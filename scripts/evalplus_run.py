"""EvalPlus's checker for `valta eval --plus`. Run it with the Python that has evalplus installed.

    python scripts/evalplus_run.py problems humaneval problems.json
    python scripts/evalplus_run.py score humaneval samples.jsonl

`problems` writes each task's id, prompt and entry point. `score` cleans every reply with EvalPlus's
sanitizer, then runs the base tests and the extra tests on it. EvalPlus writes the verdicts next to the
samples, in samples-sanitized_eval_results.json.

EvalPlus limits each test input with signal.setitimer, which Windows does not have. The call fails inside
EvalPlus's own try block, so on Windows every task would be scored as a failure. Here the same limit is a
timer thread that interrupts the main thread. Windows starts every EvalPlus worker by running this file
again, so the replacement is in place in each process. EvalPlus's memory limit uses `resource`, which
Windows does not have either, so it is turned off there.
"""

from __future__ import annotations

import _thread
import contextlib
import json
import os
import sys
import threading

if sys.platform == "win32":
    os.environ["EVALPLUS_MAX_MEMORY_BYTES"] = "-1"

import evalplus.eval  # noqa: E402

DATASETS = ("humaneval", "mbpp")


@contextlib.contextmanager
def _time_limit(seconds: float):
    timer = threading.Timer(seconds, _thread.interrupt_main)
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


if sys.platform == "win32":
    evalplus.eval.time_limit = _time_limit


def _problems(dataset: str) -> dict:
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    return get_human_eval_plus() if dataset == "humaneval" else get_mbpp_plus()


def write_problems(dataset: str, out: str) -> None:
    rows = [
        {"task_id": task_id, "prompt": problem["prompt"], "entry_point": problem["entry_point"]}
        for task_id, problem in _problems(dataset).items()
    ]
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False)


def score(dataset: str, samples: str) -> None:
    from evalplus.evaluate import evaluate
    from evalplus.sanitize import sanitize

    problems = _problems(dataset)
    target = samples[: -len(".jsonl")] + "-sanitized.jsonl"
    with open(samples, encoding="utf-8") as source, open(target, "w", encoding="utf-8") as out:
        for line in source:
            if not line.strip():
                continue
            sample = json.loads(line)
            entry_point = problems[sample["task_id"]]["entry_point"]
            solution = sanitize(sample["solution"], entrypoint=entry_point)
            out.write(json.dumps({"task_id": sample["task_id"], "solution": solution}, ensure_ascii=False) + "\n")
    evaluate(dataset=dataset, samples=target)


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[0] not in ("problems", "score") or argv[1] not in DATASETS:
        print(__doc__, file=sys.stderr)
        return 2
    command, dataset, path = argv
    if command == "problems":
        write_problems(dataset, path)
    else:
        score(dataset, path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
