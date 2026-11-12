"""HumanEval+ and MBPP+ through EvalPlus: 164 or 378 tasks, each with many more tests than the twelve-task eval.

The twelve tasks in evals.py stay the gate inside `valta train`. They are too easy to tell two strong models
apart; this is the test for choosing a model.

Valta asks the model itself, through the same client as every other call, so the host, the context size and
the thinking setting are Valta's own, and the root rules come first as always. EvalPlus lives in its own
environment, away from the training packages, and does the rest through scripts/evalplus_run.py: it cleans
each reply with its sanitizer and runs its base and extra tests. Everything is kept under data/evalplus/,
apart from the twelve-task scores in data/state.json. EvalPlus runs model-written code on this machine with a
timeout; that is not a sandbox.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from valta.config import Settings, project_root
from valta.evals import EVAL_MAX_CTX
from valta.ollama_client import OllamaError
from valta.prompt import build_system

DATASETS = {"humaneval": "HumanEval+", "mbpp": "MBPP+"}
# EvalPlus's own instruction for chat models, so scores stay comparable with published ones.
INSTRUCTION = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
# EvalPlus names its verdicts after the samples file; releases differ on the separator.
RESULTS_FILES = ("samples-sanitized.eval_results.json", "samples-sanitized_eval_results.json")
EVALPLUS_SOURCE = "https://github.com/evalplus/evalplus/archive/26d6d00bb1fd0fa37f39c99d5290da67891d1c5e.zip"
INSTALL_HINT = (
    "EvalPlus is not installed. It gets its own environment, away from the training packages:\n"
    "  py -3.13 -m venv .venv-eval\n"
    '  $env:SETUPTOOLS_SCM_PRETEND_VERSION = "0.3.2.dev0"\n'
    f'  .\\.venv-eval\\Scripts\\python.exe -m pip install "evalplus @ {EVALPLUS_SOURCE}"\n'
    "Or set VALTA_EVALPLUS_PYTHON to a Python that has evalplus."
)


class EvalPlusMissing(RuntimeError):
    pass


@dataclass
class PlusResult:
    model: str
    dataset: str
    total: int
    base_passed: int
    plus_passed: int
    folder: Path | None = None

    @property
    def score(self) -> float:
        return self.plus_passed / self.total if self.total else 0.0

    def summary(self) -> str:
        base = self.base_passed / self.total if self.total else 0.0
        return (
            f"{self.model}: {DATASETS[self.dataset]} {self.plus_passed}/{self.total} ({self.score:.1%}), "
            f"base tests alone {self.base_passed}/{self.total} ({base:.1%})"
        )


def find_evalplus_python(settings: Settings) -> str | None:
    """VALTA_EVALPLUS_PYTHON, then .venv-eval in the project, then this Python if it has evalplus."""
    found = os.environ.get("VALTA_EVALPLUS_PYTHON")
    if found:
        return found
    for candidate in (
        settings.root / ".venv-eval" / "Scripts" / "python.exe",
        settings.root / ".venv-eval" / "bin" / "python",
    ):
        if candidate.is_file():
            return str(candidate)
    if importlib.util.find_spec("evalplus") is not None:
        return sys.executable
    return None


def run_plus(
    settings: Settings,
    client,
    host: str,
    model: str,
    dataset: str = "humaneval",
    num_ctx: int | None = None,
    python: str | None = None,
    on_progress=None,
    run_script=None,
) -> PlusResult:
    """Ask `model` every task at temperature 0, then have EvalPlus score the replies."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset}. Use one of: {', '.join(DATASETS)}.")
    python = python or find_evalplus_python(settings)
    if not python:
        raise EvalPlusMissing(INSTALL_HINT)
    run_script = run_script or _run_script
    folder = _run_folder(settings, dataset, model)
    problems_path = folder / "problems.json"
    run_script(python, "problems", dataset, problems_path)
    problems = json.loads(problems_path.read_text(encoding="utf-8"))
    samples = folder / "samples.jsonl"
    ctx = min(num_ctx or settings.teacher_num_ctx, EVAL_MAX_CTX)
    _ask_every_task(settings, client, host, model, problems, samples, ctx, on_progress)
    run_script(python, "score", dataset, samples)
    verdicts = next((folder / name for name in RESULTS_FILES if (folder / name).is_file()), None)
    if verdicts is None:
        raise RuntimeError(f"EvalPlus finished without writing its verdicts in {folder}.")
    result = read_results(verdicts, model, dataset)
    result.folder = folder
    _record(settings, host, result)
    return result


def _ask_every_task(settings, client, host, model, problems, samples: Path, ctx: int, on_progress) -> None:
    """One reply per task, in EvalPlus's samples format. A failed request is an empty reply, which fails."""
    system = build_system(settings, role="You are a careful Python programmer.")
    errors = 0
    with samples.open("w", encoding="utf-8") as handle:
        for index, problem in enumerate(problems, 1):
            user = f"{INSTRUCTION}\n```python\n{problem['prompt'].strip()}\n```\n"
            note = ""
            try:
                reply = client.chat(
                    host,
                    model,
                    [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    ctx,
                    temperature=0.0,
                )
            except OllamaError as exc:
                errors += 1
                reply, note = "", f"model error: {exc}"[:200]
            handle.write(json.dumps({"task_id": problem["task_id"], "solution": reply}, ensure_ascii=False) + "\n")
            handle.flush()
            if on_progress is not None:
                on_progress(index, len(problems), problem["task_id"], note)
    if problems and errors == len(problems):
        raise OllamaError(f"Every request to {model} on {host} failed. Is the server running with that model?")


def _run_script(python: str, command: str, dataset: str, path: Path) -> None:
    script = project_root() / "scripts" / "evalplus_run.py"
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    completed = subprocess.run([python, str(script), command, dataset, str(path)], env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"EvalPlus stopped with exit code {completed.returncode} during `{command}`.")


def _run_folder(settings: Settings, dataset: str, model: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9]+", "-", model).strip("-").lower()[:60]
    folder = settings.evalplus_dir / f"{stamp}-{dataset}-{slug}"
    folder.mkdir(parents=True)
    return folder


def read_results(path: Path, model: str, dataset: str) -> PlusResult:
    """Count EvalPlus's verdicts. A task passes the plus score only when its base and extra tests both pass."""
    data = json.loads(path.read_text(encoding="utf-8"))
    result = PlusResult(model=model, dataset=dataset, total=0, base_passed=0, plus_passed=0)
    for rows in data.get("eval", {}).values():
        for row in rows:
            base = row.get("base_status") == "pass"
            result.total += 1
            result.base_passed += base
            result.plus_passed += base and row.get("plus_status") == "pass"
    return result


def _record(settings: Settings, host: str, result: PlusResult) -> None:
    entry = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": result.model,
        "host": host,
        "dataset": result.dataset,
        "total": result.total,
        "base": result.base_passed,
        "plus": result.plus_passed,
        "options": settings.model_options,
        "folder": result.folder.name if result.folder else None,
    }
    with (settings.evalplus_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def plus_history(settings: Settings, dataset: str | None = None) -> list[dict]:
    """Every recorded run, oldest first, optionally for one dataset."""
    path = settings.evalplus_dir / "history.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if dataset is None or row.get("dataset") == dataset:
            rows.append(row)
    return rows
