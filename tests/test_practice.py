from pathlib import Path

import pytest

from valta.config import load_settings
from valta.evals import BUILTIN_TASKS, REPLY_SUFFIX
from valta.memory import Memory
from valta.ollama_client import OllamaError
from valta.practice import SOURCE, Practice, Problem, parse_problem, risky, too_close, vet
from valta.train import build_pairs


def _settings(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "config.toml").write_text((root / "config.toml").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "rules").mkdir()
    (tmp_path / "rules" / "root.md").write_text((root / "rules" / "root.md").read_text(encoding="utf-8"), encoding="utf-8")
    return load_settings(tmp_path)


GOOD = "```python\ndef rotate_left(items, k):\n    if not items:\n        return []\n    k %= len(items)\n    return items[k:] + items[:k]\n```"
BAD = "```python\ndef rotate_left(items, k):\n    return items[k:] + items[:k]\n```"
PROMPT = (
    "Write rotate_left(items, k) that returns a new list with the items rotated left by k places. "
    "k may be larger than the length, and an empty list stays empty."
)
TESTS = (
    "assert rotate_left([1, 2, 3], 1) == [2, 3, 1]\n"
    "assert rotate_left([1, 2, 3], 4) == [2, 3, 1]\n"
    "assert rotate_left([], 3) == []\n"
    "assert rotate_left([1], 0) == [1]\n"
)


def problem_reply(name="rotate_left", prompt=PROMPT, tests=TESTS) -> str:
    return f"NAME: {name}\nPROMPT: {prompt}\nTESTS:\n```python\n{tests}```"


class _Models:
    """Scripted teacher and student. The teacher writes problems and solves them; the student only solves."""

    def __init__(self, settings, problems, teacher_solutions, student_solutions=()):
        self.settings = settings
        self.problems = list(problems)
        self.teacher = list(teacher_solutions)
        self.student = list(student_solutions)
        self.teacher_asked_to_solve = 0

    def chat(self, host, model, messages, num_ctx, temperature=0.4):
        prompt = messages[-1]["content"]
        if "Write one new Python practice problem" in prompt:
            if not self.problems:
                raise OllamaError("teacher is down")
            return self.problems.pop(0)
        if host == self.settings.self_host:
            if not self.student:
                raise OllamaError("student is down")
            return self.student.pop(0)
        self.teacher_asked_to_solve += 1
        return self.teacher.pop(0)


def _session(tmp_path, problems, teacher, student=(), **kwargs):
    settings = _settings(tmp_path)
    memory = Memory(settings)
    models = _Models(settings, problems, teacher, student)
    kwargs.setdefault("student_model", "valta")
    return Practice(settings, memory, models, seed=0, **kwargs), memory, models, settings


def test_a_verified_problem_the_student_fails_is_saved(tmp_path: Path):
    session, memory, _models, _settings_ = _session(tmp_path, [problem_reply()], [GOOD], [BAD])
    outcome = session.once()
    assert outcome.kind == "saved"
    rows = memory.all_examples()
    assert len(rows) == 1 and rows[0]["source_id"] == SOURCE
    assert rows[0]["user_text"].startswith("Write rotate_left(")
    assert rows[0]["user_text"].endswith(REPLY_SUFFIX)
    assert rows[0]["assistant_text"] == GOOD


def test_a_problem_the_student_already_solves_teaches_nothing(tmp_path: Path):
    session, memory, _models, _s = _session(tmp_path, [problem_reply()], [GOOD], [GOOD])
    assert session.once().kind == "known"
    assert memory.all_examples() == []
    (tmp_path / "again").mkdir()
    kept, memory, _models, _s = _session(tmp_path / "again", [problem_reply()], [GOOD], [GOOD], keep_known=True)
    assert kept.once().kind == "saved"
    assert len(memory.all_examples()) == 1


def test_without_a_student_every_verified_problem_is_kept(tmp_path: Path):
    session, memory, _models, _s = _session(tmp_path, [problem_reply()], [GOOD], student_model=None)
    assert session.once().kind == "saved"
    assert len(memory.all_examples()) == 1


@pytest.mark.parametrize(
    "reply",
    [
        "I cannot do that.",
        problem_reply(tests="assert True\nassert True\nassert True\n"),
        problem_reply(tests="import os\nassert rotate_left([1], 0) == [1]\nassert rotate_left([], 1) == []\nassert rotate_left([1, 2], 1) == [2, 1]\n"),
        problem_reply(tests="assert rotate_left([1, 2], 1) == [2, 1]\nassert rotate_left([], 1) == []\n"),
        problem_reply(tests="assert rotate_left([1, 2], 1) == [2, 1\nassert 1\nassert 2\n"),
        problem_reply(name="fizzbuzz", prompt="Write fizzbuzz(n) that returns the fizzbuzz strings from one up to n, as a list."),
        problem_reply(prompt="Write rotate_left(items, k) that moves things along. " + BUILTIN_TASKS[1].prompt.split("\n")[0]),
        problem_reply(prompt="Write other_name(x) that does something useful with the value it is given here."),
    ],
)
def test_unsound_problems_are_rejected_before_anyone_solves_them(tmp_path: Path, reply: str):
    session, memory, models, _s = _session(tmp_path, [reply], [GOOD], [BAD])
    outcome = session.once()
    assert outcome.kind == "rejected", outcome
    assert memory.all_examples() == []
    assert models.teacher_asked_to_solve == 0


def test_tests_the_teacher_cannot_pass_are_dropped(tmp_path: Path):
    wrong = TESTS.replace("== [2, 3, 1]\nassert rotate_left([1, 2, 3], 4)", "== [3, 2, 1]\nassert rotate_left([1, 2, 3], 4)")
    session, memory, models, _s = _session(tmp_path, [problem_reply(tests=wrong)], [GOOD, GOOD], [BAD])
    outcome = session.once()
    assert outcome.kind == "unsolved"
    assert models.teacher_asked_to_solve == 2  # it gets a second try
    assert memory.all_examples() == []


def test_a_second_try_can_pass(tmp_path: Path):
    session, memory, _models, _s = _session(tmp_path, [problem_reply()], [BAD, GOOD], [BAD])
    assert session.once().kind == "saved"
    assert memory.all_examples()[0]["assistant_text"] == GOOD


def test_a_solution_that_touches_the_system_is_never_run(tmp_path: Path):
    marker = tmp_path / "ran.txt"
    # Correct in every other way, so only the safety check can stop it.
    nasty = (
        "```python\n"
        "def rotate_left(items, k):\n"
        f"    open({str(marker)!r}, 'w').write('ran')\n"
        "    if not items:\n"
        "        return []\n"
        "    k %= len(items)\n"
        "    return items[k:] + items[:k]\n"
        "```"
    )
    session, memory, _models, _s = _session(tmp_path, [problem_reply()], [nasty, nasty], [BAD])
    assert session.once().kind == "unsolved"
    assert not marker.exists()
    assert memory.all_examples() == []


def test_a_name_is_never_written_twice(tmp_path: Path):
    session, memory, _models, settings = _session(tmp_path, [problem_reply()], [GOOD], [BAD])
    assert session.once().kind == "saved"
    again = Practice(settings, memory, _Models(settings, [problem_reply()], [GOOD], [BAD]), student_model="valta", seed=1)
    outcome = again.once()
    assert outcome.kind == "rejected" and "name" in outcome.note


def test_a_student_server_that_stops_does_not_stop_practice(tmp_path: Path):
    session, memory, _models, _s = _session(tmp_path, [problem_reply()], [GOOD], [])
    outcome = session.once()
    assert outcome.kind == "saved" and "not checked" in outcome.note
    assert session.student_model is None
    assert session.stats.notes
    assert len(memory.all_examples()) == 1


def test_a_teacher_server_that_stops_ends_the_run_cleanly(tmp_path: Path):
    session, memory, _models, _s = _session(tmp_path, [problem_reply()], [GOOD], [BAD])
    stats = session.run(5)
    assert stats.saved == 1
    assert "teacher is down" in stats.error
    assert stats.tried == 1


def test_run_reports_progress_and_counts(tmp_path: Path):
    seen = []
    session, _memory, _models, _s = _session(tmp_path, [problem_reply(), "junk"], [GOOD], [BAD])
    session.run(2, on_progress=lambda index, outcome: seen.append((index, outcome.kind)))
    assert seen == [(1, "saved"), (2, "rejected")]
    assert session.stats.summary() == "Tried 2: saved 1, already known 0, rejected 1, unsolved 0."


def test_saved_practice_reaches_the_training_file(tmp_path: Path):
    session, memory, _models, settings = _session(tmp_path, [problem_reply()], [GOOD], [BAD])
    session.once()
    path, ids = build_pairs(settings, memory, client=None)
    assert "rotate_left" in path.read_text(encoding="utf-8")
    assert len(ids) == 1


def test_parse_problem_variants():
    plain = "NAME: `chunk_text`\nPROMPT: Write chunk_text(s, n) that\nsplits s into pieces of n.\nTESTS:\n```\nassert 1\n```"
    problem = parse_problem(plain)
    assert problem is not None
    assert problem.name == "chunk_text"
    assert problem.prompt == "Write chunk_text(s, n) that splits s into pieces of n."
    assert problem.tests == "assert 1\n"
    assert parse_problem("NAME: x\nPROMPT: y") is None
    assert parse_problem("") is None


def test_risky_code_is_spotted_without_blocking_normal_code():
    assert risky("import os\nx = 1")
    assert risky("from subprocess import run")
    assert risky("data = open('f').read()")
    assert risky("exec('1')")
    assert risky("x = input()")
    assert not risky("import re\npattern = re.compile('a')\nimport collections\nfrom itertools import chain")
    assert not risky("def pos(x):\n    return x.position\n")
    assert not risky("reopen = 1\ncosine = 2\n")


def test_close_paraphrases_of_an_eval_task_are_flagged():
    blocked = [task.prompt for task in BUILTIN_TASKS]
    close = (
        "Write is_palindrome(s) that returns True when s reads the same forwards and backwards, "
        "ignoring case and ignoring every character that is not a letter or a digit."
    )
    assert too_close(close, blocked)
    assert not too_close(PROMPT, blocked)


def test_vet_accepts_a_sound_problem_and_names_the_reason_otherwise():
    blocked = [task.prompt for task in BUILTIN_TASKS]
    assert vet(Problem("rotate_left", PROMPT, TESTS), set(), blocked) is None
    assert vet(Problem("rotate_left", PROMPT, TESTS), {"rotate_left"}, blocked) == "name already used"
    assert vet(Problem("rotate_left", "short", TESTS), set(), blocked)
