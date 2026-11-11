from pathlib import Path

import pytest

from valta import cli
from valta.config import load_settings
from valta.memory import Memory

GOOD = "```python\ndef rotate_left(items, k):\n    if not items:\n        return []\n    k %= len(items)\n    return items[k:] + items[:k]\n```"
BAD = "```python\ndef rotate_left(items, k):\n    return items[k:] + items[:k]\n```"
PROBLEM = (
    "NAME: rotate_left\n"
    "PROMPT: Write rotate_left(items, k) that returns a new list with the items rotated left by k places. "
    "k may be larger than the length, and an empty list stays empty.\n"
    "TESTS:\n```python\n"
    "assert rotate_left([1, 2, 3], 1) == [2, 3, 1]\n"
    "assert rotate_left([1, 2, 3], 4) == [2, 3, 1]\n"
    "assert rotate_left([], 3) == []\n"
    "assert rotate_left([1], 0) == [1]\n"
    "```"
)


class _Models:
    def __init__(self, settings, student_up=True):
        self.settings = settings
        self.student_up = student_up
        self.problems = [PROBLEM]

    def healthy(self, host):
        return self.student_up or host != self.settings.self_host

    def chat(self, host, model, messages, num_ctx, temperature=0.4):
        if "Write one new Python practice problem" in messages[-1]["content"]:
            if not self.problems:
                from valta.ollama_client import OllamaError

                raise OllamaError("no more problems")
            return self.problems.pop(0)
        return BAD if host == self.settings.self_host else GOOD


def _setup(tmp_path: Path, monkeypatch, student_up=True):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "config.toml").write_text((root / "config.toml").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "rules").mkdir()
    (tmp_path / "rules" / "root.md").write_text((root / "rules" / "root.md").read_text(encoding="utf-8"), encoding="utf-8")
    settings = load_settings(tmp_path)
    memory = Memory(settings)
    models = _Models(settings, student_up)
    monkeypatch.setattr(cli, "_stack", lambda project=None: (settings, memory, None, models))
    return settings, memory


def test_practice_command_saves_and_reports(tmp_path: Path, monkeypatch, capsys):
    settings, memory = _setup(tmp_path, monkeypatch)
    cli.practice(count=2, train_after=False, keep_known=False)
    out = capsys.readouterr().out
    assert "saved" in out and "rotate_left" in out
    assert "Tried 1: saved 1" in out  # the teacher ran out of problems after one, which ends the run
    assert "1 example(s) are waiting" in out
    assert len(memory.all_examples()) == 1


def test_practice_with_the_student_server_down_says_so(tmp_path: Path, monkeypatch, capsys):
    _setup(tmp_path, monkeypatch, student_up=False)
    cli.practice(count=1, train_after=False, keep_known=False)
    assert "GTX 1660 server is down" in capsys.readouterr().out


def test_practice_train_only_runs_when_enough_examples_are_waiting(tmp_path: Path, monkeypatch, capsys):
    settings, _memory = _setup(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(cli, "_train", lambda *args, **kwargs: calls.append(args))
    settings.train_threshold = 5
    cli.practice(count=1, train_after=True, keep_known=False)
    assert calls == []
    assert "Not enough yet" in capsys.readouterr().out

    (tmp_path / "second").mkdir()
    settings, _memory = _setup(tmp_path / "second", monkeypatch)
    settings.train_threshold = 1
    cli.practice(count=1, train_after=True, keep_known=False)
    assert len(calls) == 1


def test_practice_refuses_when_the_teacher_server_is_down(tmp_path: Path, monkeypatch):
    settings, _memory = _setup(tmp_path, monkeypatch)
    down = _Models(settings)
    down.healthy = lambda host: False
    monkeypatch.setattr(cli, "_stack", lambda project=None: (settings, _memory, None, down))
    with pytest.raises(Exception, match="RX 6600"):
        cli.practice(count=1, train_after=False, keep_known=False)
