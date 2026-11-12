"""A small, fixed coding eval. It tells you whether a fine-tune made a model better or worse.

Each task asks for one Python function. The reply is run against hidden asserts in a temporary
folder with a timeout. The score is the share of tasks that pass. Add your own tasks as JSON files
in data/evals/ (a list of {"id", "prompt", "tests"}); `tests` is Python that runs after
`from solution import *`.

The tasks here are never used as training data. `contaminated` drops any training pair that
contains one of these prompts.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from valta.config import Settings
from valta.ollama_client import OllamaError
from valta.prompt import build_system

RUN_TIMEOUT_SEC = 10
EVAL_MAX_CTX = 8192
_SUFFIX = "Reply with one ```python code block that holds the complete function and any imports. No explanation."
REPLY_SUFFIX = _SUFFIX  # the instruction every eval prompt ends with; practice problems use it too


@dataclass
class EvalTask:
    id: str
    prompt: str
    tests: str


@dataclass
class EvalResult:
    model: str
    passed: int
    total: int
    details: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def summary(self) -> str:
        return f"{self.model}: {self.passed}/{self.total} ({self.score:.0%})"


def _task(task_id: str, prompt: str, tests: str) -> EvalTask:
    return EvalTask(task_id, f"{prompt}\n\n{_SUFFIX}", tests.strip() + "\n")


BUILTIN_TASKS: list[EvalTask] = [
    _task(
        "fizzbuzz",
        "Write fizzbuzz(n) that returns a list of strings for 1..n: 'Fizz' for multiples of 3, 'Buzz' for 5, "
        "'FizzBuzz' for both, otherwise the number as a string.",
        """
assert fizzbuzz(0) == []
assert fizzbuzz(5) == ['1', '2', 'Fizz', '4', 'Buzz']
assert fizzbuzz(15)[-1] == 'FizzBuzz'
assert fizzbuzz(15)[8] == 'Fizz'
""",
    ),
    _task(
        "is_palindrome",
        "Write is_palindrome(s) that returns True when s reads the same forwards and backwards, "
        "ignoring case and ignoring every character that is not a letter or digit.",
        """
assert is_palindrome('A man, a plan, a canal: Panama') is True
assert is_palindrome('race a car') is False
assert is_palindrome('') is True
assert is_palindrome('No lemon, no melon') is True
""",
    ),
    _task(
        "flatten",
        "Write flatten(items) that takes arbitrarily nested lists and returns one flat list in order.",
        """
assert flatten([]) == []
assert flatten([1, [2, [3, [4]]], 5]) == [1, 2, 3, 4, 5]
assert flatten([[], [[]], [1]]) == [1]
assert flatten(['a', ['b']]) == ['a', 'b']
""",
    ),
    _task(
        "word_count",
        "Write word_count(text) that returns a dict of word -> count. Split on whitespace, lowercase every word, "
        "strip leading and trailing punctuation (the characters in string.punctuation), and skip empty results.",
        """
assert word_count('') == {}
assert word_count('The cat. the CAT!') == {'the': 2, 'cat': 2}
assert word_count('  hello,   world;  hello ') == {'hello': 2, 'world': 1}
assert word_count('... !!!') == {}
""",
    ),
    _task(
        "two_sum",
        "Write two_sum(nums, target) that returns a tuple (i, j) with i < j such that nums[i] + nums[j] == target. "
        "Exactly one such pair exists.",
        """
assert two_sum([2, 7, 11, 15], 9) == (0, 1)
assert two_sum([3, 2, 4], 6) == (1, 2)
assert two_sum([3, 3], 6) == (0, 1)
assert two_sum([-1, -2, -3, -4, -5], -8) == (2, 4)
""",
    ),
    _task(
        "merge_intervals",
        "Write merge_intervals(intervals) that takes a list of [start, end] lists in any order and returns the merged "
        "list sorted by start. Intervals that overlap or touch (for example [1, 2] and [2, 3]) are merged.",
        """
assert merge_intervals([]) == []
assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
assert merge_intervals([[1, 2], [2, 3]]) == [[1, 3]]
assert merge_intervals([[5, 6], [1, 2]]) == [[1, 2], [5, 6]]
assert merge_intervals([[1, 10], [2, 3]]) == [[1, 10]]
""",
    ),
    _task(
        "roman_to_int",
        "Write roman_to_int(s) that converts a Roman numeral (I, V, X, L, C, D, M, with subtractive pairs like IV and CM) "
        "to an integer.",
        """
assert roman_to_int('III') == 3
assert roman_to_int('IV') == 4
assert roman_to_int('IX') == 9
assert roman_to_int('LVIII') == 58
assert roman_to_int('MCMXCIV') == 1994
""",
    ),
    _task(
        "is_balanced",
        "Write is_balanced(s) that returns True when every (), [] and {} in s is closed in the right order. "
        "Ignore all other characters.",
        """
assert is_balanced('') is True
assert is_balanced('([]{})') is True
assert is_balanced('(]') is False
assert is_balanced('((') is False
assert is_balanced('a(b[c]d)e') is True
assert is_balanced(')(') is False
""",
    ),
    _task(
        "chunk",
        "Write chunk(items, size) that splits a list into consecutive lists of length size; the last one may be shorter. "
        "Raise ValueError when size is less than 1.",
        """
assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
assert chunk([], 3) == []
assert chunk([1, 2], 5) == [[1, 2]]
try:
    chunk([1], 0)
    raise SystemExit('no ValueError')
except ValueError:
    pass
""",
    ),
    _task(
        "fib",
        "Write fib(n) returning the nth Fibonacci number with fib(0) = 0 and fib(1) = 1. It must be fast for n = 90.",
        """
assert fib(0) == 0
assert fib(1) == 1
assert fib(10) == 55
assert fib(90) == 2880067194370816120
""",
    ),
    _task(
        "group_anagrams",
        "Write group_anagrams(words) that groups words that are anagrams of each other. Return a list of groups; "
        "sort each group alphabetically and sort the list of groups by each group's first word.",
        """
assert group_anagrams([]) == []
assert group_anagrams(['eat', 'tea', 'tan', 'ate', 'nat', 'bat']) == [['ate', 'eat', 'tea'], ['bat'], ['nat', 'tan']]
assert group_anagrams(['a']) == [['a']]
""",
    ),
    _task(
        "parse_kv",
        "Write parse_kv(text) that parses lines of 'key=value' into a dict. Split each line on the first '=' only, "
        "strip spaces around the key and the value, and ignore blank lines and lines that start with '#'.",
        """
assert parse_kv('') == {}
assert parse_kv('a=1\\nb = two words ') == {'a': '1', 'b': 'two words'}
assert parse_kv('# note\\n\\nx=a=b') == {'x': 'a=b'}
assert parse_kv('  # indented comment\\nk=v') == {'k': 'v'}
""",
    ),
]


def load_tasks(settings: Settings | None = None) -> list[EvalTask]:
    """The built-in tasks, plus any JSON task files the user dropped in data/evals/."""
    tasks = list(BUILTIN_TASKS)
    if settings is None or not settings.evals_dir.is_dir():
        return tasks
    seen = {task.id for task in tasks}
    for path in sorted(settings.evals_dir.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for item in raw if isinstance(raw, list) else []:
            try:
                task_id, prompt, tests = str(item["id"]), str(item["prompt"]), str(item["tests"])
            except (KeyError, TypeError):
                continue
            if task_id in seen:
                continue
            seen.add(task_id)
            tasks.append(EvalTask(task_id, prompt, tests if tests.endswith("\n") else tests + "\n"))
    return tasks


def eval_prompts(settings: Settings | None = None) -> list[str]:
    return [task.prompt for task in load_tasks(settings)]


def _normal(text: str) -> str:
    return " ".join(text.lower().split())


def contaminated(text: str, prompts: list[str]) -> bool:
    """True when a training text contains an eval prompt (or the other way round)."""
    candidate = _normal(text)
    if not candidate:
        return False
    for prompt in prompts:
        # The shared suffix would make every prompt look alike, so compare the task wording only.
        core = _normal(prompt.replace(_SUFFIX, ""))
        if len(core) >= 20 and (core in candidate or (len(candidate) >= 20 and candidate in core)):
            return True
    return False


_FENCE = re.compile(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(reply: str) -> str | None:
    for block in _FENCE.findall(reply):
        if "def " in block:
            return block.strip() + "\n"
    if "def " in reply and "```" not in reply:
        return reply.strip() + "\n"
    return None


def check_solution(code: str, tests: str, timeout: int = RUN_TIMEOUT_SEC) -> tuple[bool, str]:
    """Run code plus tests in a throwaway folder. This is a timeout and a clean environment, not a sandbox."""
    with tempfile.TemporaryDirectory(prefix="valta-eval-") as folder:
        work = Path(folder)
        (work / "solution.py").write_text(code, encoding="utf-8")
        (work / "check.py").write_text("from solution import *\n" + tests, encoding="utf-8")
        keep = {"PATH", "SYSTEMROOT", "TEMP", "TMP"}
        env = {key: value for key, value in os.environ.items() if key.upper() in keep}
        try:
            completed = subprocess.run(
                [sys.executable, "-E", "-s", "check.py"],
                cwd=work,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                text=True,
                timeout=timeout,
                env=env,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return False, f"timed out after {timeout}s"
        if completed.returncode == 0:
            return True, ""
        lines = (completed.stderr or completed.stdout).strip().splitlines()
        return False, (lines[-1] if lines else f"exit {completed.returncode}")[:200]


def run_eval(
    settings: Settings,
    client,
    host: str,
    model: str,
    tasks: list[EvalTask] | None = None,
    num_ctx: int | None = None,
    on_progress=None,
) -> EvalResult:
    """Ask `model` every task at temperature 0 and score the replies."""
    tasks = tasks if tasks is not None else load_tasks(settings)
    ctx = min(num_ctx or settings.self_num_ctx, EVAL_MAX_CTX)
    system = build_system(settings, role="You are a careful Python programmer.")
    result = EvalResult(model=model, passed=0, total=len(tasks))
    errors = 0
    for task in tasks:
        try:
            reply = client.chat(
                host,
                model,
                [{"role": "system", "content": system}, {"role": "user", "content": task.prompt}],
                ctx,
                temperature=0.0,
            )
        except OllamaError as exc:
            errors += 1
            result.details.append((task.id, False, f"model error: {exc}"[:200]))
            continue
        code = extract_code(reply)
        if code is None:
            ok, note = False, "no code block"
        else:
            ok, note = check_solution(code, task.tests)
        result.passed += 1 if ok else 0
        result.details.append((task.id, ok, note))
        if on_progress is not None:
            on_progress(task.id, ok, note)
    if tasks and errors == len(tasks):
        raise OllamaError(f"Every eval request to {model} on {host} failed. Is the server running with that model?")
    return result


def should_promote(baseline: EvalResult | None, candidate: EvalResult, tolerance: float) -> tuple[bool, str]:
    """Promote unless the candidate scores more than `tolerance` below the baseline."""
    if baseline is None:
        return True, "No baseline score, so there is nothing to compare against."
    drop = baseline.score - candidate.score
    if drop > tolerance + 1e-9:
        return False, (
            f"It scored {candidate.passed}/{candidate.total}, {drop:.0%} below {baseline.summary()}, "
            f"more than the {tolerance:.0%} allowed."
        )
    return True, f"It scored {candidate.passed}/{candidate.total}, within {tolerance:.0%} of {baseline.summary()} or better."
