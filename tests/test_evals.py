import json
from pathlib import Path

import pytest

from valta.config import load_settings
from valta.evals import (
    BUILTIN_TASKS,
    EvalResult,
    EvalTask,
    check_solution,
    contaminated,
    extract_code,
    load_tasks,
    run_eval,
    should_promote,
)
from valta.ollama_client import OllamaError

REFERENCE = {
    "fizzbuzz": """
def fizzbuzz(n):
    out = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            out.append('FizzBuzz')
        elif i % 3 == 0:
            out.append('Fizz')
        elif i % 5 == 0:
            out.append('Buzz')
        else:
            out.append(str(i))
    return out
""",
    "is_palindrome": """
def is_palindrome(s):
    cleaned = [c.lower() for c in s if c.isalnum()]
    return cleaned == cleaned[::-1]
""",
    "flatten": """
def flatten(items):
    out = []
    for item in items:
        if isinstance(item, list):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out
""",
    "word_count": """
import string

def word_count(text):
    counts = {}
    for raw in text.split():
        word = raw.lower().strip(string.punctuation)
        if word:
            counts[word] = counts.get(word, 0) + 1
    return counts
""",
    "two_sum": """
def two_sum(nums, target):
    seen = {}
    for j, value in enumerate(nums):
        if target - value in seen:
            return (seen[target - value], j)
        seen[value] = j
""",
    "merge_intervals": """
def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged
""",
    "roman_to_int": """
def roman_to_int(s):
    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    for i, ch in enumerate(s):
        if i + 1 < len(s) and values[ch] < values[s[i + 1]]:
            total -= values[ch]
        else:
            total += values[ch]
    return total
""",
    "is_balanced": """
def is_balanced(s):
    pairs = {')': '(', ']': '[', '}': '{'}
    stack = []
    for ch in s:
        if ch in '([{':
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack
""",
    "chunk": """
def chunk(items, size):
    if size < 1:
        raise ValueError('size must be at least 1')
    return [items[i:i + size] for i in range(0, len(items), size)]
""",
    "fib": """
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
""",
    "group_anagrams": """
def group_anagrams(words):
    groups = {}
    for word in words:
        groups.setdefault(''.join(sorted(word)), []).append(word)
    return sorted(sorted(group) for group in groups.values())
""",
    "parse_kv": """
def parse_kv(text):
    out = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        key, _, value = stripped.partition('=')
        out[key.strip()] = value.strip()
    return out
""",
}


def _settings(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "config.toml").write_text((root / "config.toml").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "rules").mkdir()
    (tmp_path / "rules" / "root.md").write_text((root / "rules" / "root.md").read_text(encoding="utf-8"), encoding="utf-8")
    return load_settings(tmp_path)


class _FakeModel:
    """Answers with the reference solution for tasks it 'knows' and with a wrong function for the rest."""

    def __init__(self, knows):
        self.knows = set(knows)
        self.temperatures = []

    def chat(self, host, model, messages, num_ctx, temperature=0.4):
        self.temperatures.append(temperature)
        prompt = messages[-1]["content"]
        for task_id, code in REFERENCE.items():
            if f"Write {task_id}(" in prompt:
                if task_id in self.knows:
                    return f"```python\n{code}\n```"
                return f"```python\ndef {task_id}(*args, **kwargs):\n    return None\n```"
        raise AssertionError("unexpected prompt")


def test_every_builtin_task_has_a_working_reference():
    assert {task.id for task in BUILTIN_TASKS} == set(REFERENCE)
    for task in BUILTIN_TASKS:
        ok, note = check_solution(REFERENCE[task.id], task.tests)
        assert ok, f"{task.id}: {note}"


def test_wrong_and_hanging_solutions_fail():
    task = next(t for t in BUILTIN_TASKS if t.id == "fizzbuzz")
    ok, _ = check_solution("def fizzbuzz(n):\n    return []\n", task.tests)
    assert not ok
    ok, note = check_solution("def fizzbuzz(n):\n    while True:\n        pass\n", task.tests, timeout=1)
    assert not ok and "timed out" in note
    ok, _ = check_solution("def fizzbuzz(n) return\n", task.tests)
    assert not ok


def test_extract_code():
    assert extract_code("Sure!\n```python\ndef f():\n    return 1\n```\nDone.") == "def f():\n    return 1\n"
    assert extract_code("```\ndef g(): pass\n```") == "def g(): pass\n"
    assert extract_code("no code here") is None
    assert extract_code("```python\nx = 1\n```") is None
    assert extract_code("def h():\n    return 2") == "def h():\n    return 2\n"


def test_run_eval_scores_and_asks_for_temperature_zero(tmp_path: Path):
    settings = _settings(tmp_path)
    knows = [task.id for task in BUILTIN_TASKS][:6]
    model = _FakeModel(knows)
    result = run_eval(settings, model, "http://x", "fake")
    assert result.total == len(BUILTIN_TASKS)
    assert result.passed == 6
    assert result.score == pytest.approx(6 / len(BUILTIN_TASKS))
    assert set(model.temperatures) == {0.0}
    assert {task_id for task_id, ok, _ in result.details if ok} == set(knows)


def test_run_eval_raises_when_the_server_never_answers(tmp_path: Path):
    settings = _settings(tmp_path)

    class _Down:
        def chat(self, *args, **kwargs):
            raise OllamaError("connection refused")

    with pytest.raises(OllamaError):
        run_eval(settings, _Down(), "http://x", "fake")


def test_promotion_gate():
    baseline = EvalResult("old", passed=8, total=10)
    same = EvalResult("new", passed=8, total=10)
    one_worse = EvalResult("new", passed=7, total=10)
    two_worse = EvalResult("new", passed=6, total=10)
    assert should_promote(baseline, same, 0.1)[0]
    assert should_promote(baseline, one_worse, 0.1)[0]
    assert not should_promote(baseline, two_worse, 0.1)[0]
    assert should_promote(baseline, two_worse, 0.2)[0]
    assert should_promote(None, two_worse, 0.1)[0]
    assert should_promote(baseline, EvalResult("new", passed=10, total=10), 0.0)[0]


def test_training_text_that_contains_an_eval_prompt_is_flagged():
    prompts = [task.prompt for task in BUILTIN_TASKS]
    assert contaminated(prompts[0], prompts)
    assert contaminated("Please help. " + prompts[3] + " Thanks.", prompts)
    assert not contaminated("How do I package a Python project?", prompts)
    assert not contaminated("", prompts)


def test_user_tasks_load_from_json(tmp_path: Path):
    settings = _settings(tmp_path)
    (settings.evals_dir / "mine.json").write_text(
        json.dumps(
            [
                {"id": "double", "prompt": "Write double(x) returning x * 2.", "tests": "assert double(2) == 4"},
                {"id": "fizzbuzz", "prompt": "duplicate id is ignored", "tests": "assert True"},
                {"prompt": "missing id is ignored", "tests": "assert True"},
            ]
        ),
        encoding="utf-8",
    )
    (settings.evals_dir / "broken.json").write_text("{not json", encoding="utf-8")
    tasks = load_tasks(settings)
    assert len(tasks) == len(BUILTIN_TASKS) + 1
    mine = next(task for task in tasks if task.id == "double")
    assert isinstance(mine, EvalTask)
    ok, _ = check_solution("def double(x):\n    return x * 2\n", mine.tests)
    assert ok


def test_a_solution_cannot_read_the_keyboard():
    task = next(t for t in BUILTIN_TASKS if t.id == "fizzbuzz")
    ok, note = check_solution("def fizzbuzz(n):\n    return input()\n", task.tests, timeout=5)
    assert not ok and "timed out" not in note
