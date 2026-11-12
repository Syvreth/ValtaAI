import json
from pathlib import Path

import pytest

from valta import cli, plus_eval
from valta.config import load_settings
from valta.ollama_client import OllamaError
from valta.plus_eval import EvalPlusMissing, find_evalplus_python, plus_history, read_results, run_plus
from valta.prompt import load_rules

PROBLEMS = [
    {"task_id": "HumanEval/0", "prompt": "def add(a, b):\n    \"\"\"Return a + b.\"\"\"\n", "entry_point": "add"},
    {"task_id": "HumanEval/1", "prompt": "def neg(x):\n    \"\"\"Return -x.\"\"\"\n", "entry_point": "neg"},
]


def _settings(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "config.toml").write_text((root / "config.toml").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "rules").mkdir()
    (tmp_path / "rules" / "root.md").write_text((root / "rules" / "root.md").read_text(encoding="utf-8"), encoding="utf-8")
    return load_settings(tmp_path)


def _verdicts(statuses):
    return {
        "eval": {
            task_id: [{"task_id": task_id, "base_status": base, "plus_status": plus}]
            for task_id, (base, plus) in statuses.items()
        }
    }


class _Model:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def chat(self, host, model, messages, num_ctx, temperature=0.4):
        self.calls.append({"host": host, "model": model, "messages": messages, "num_ctx": num_ctx, "temperature": temperature})
        if self.fail:
            raise OllamaError("connection refused")
        return "```python\ndef add(a, b):\n    return a + b\n```"


class _Runner:
    """Stands in for scripts/evalplus_run.py: writes the problems, then verdicts for whatever was replied."""

    def __init__(self):
        self.commands = []
        self.samples = []

    def __call__(self, python, command, dataset, path):
        self.commands.append((python, command, dataset))
        if command == "problems":
            path.write_text(json.dumps(PROBLEMS), encoding="utf-8")
            return
        self.samples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        verdicts = _verdicts({"HumanEval/0": ("pass", "pass"), "HumanEval/1": ("pass", "fail")})
        (path.parent / plus_eval.RESULTS_FILES[0]).write_text(json.dumps(verdicts), encoding="utf-8")


def test_read_results_counts_a_plus_pass_only_when_both_test_sets_pass(tmp_path: Path):
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(_verdicts({"a": ("pass", "pass"), "b": ("pass", "fail"), "c": ("fail", "pass"), "d": ("fail", None)})),
        encoding="utf-8",
    )
    result = read_results(path, "m", "humaneval")
    assert (result.total, result.base_passed, result.plus_passed) == (4, 2, 1)
    assert result.score == 0.25
    assert "HumanEval+ 1/4 (25.0%)" in result.summary()
    assert "base tests alone 2/4 (50.0%)" in result.summary()


def test_run_plus_asks_at_temperature_zero_and_keeps_results_apart(tmp_path: Path):
    settings = _settings(tmp_path)
    model, runner = _Model(), _Runner()
    seen = []
    result = run_plus(
        settings,
        model,
        "http://teacher",
        "qwen:7b",
        num_ctx=32768,
        python="py-eval",
        run_script=runner,
        on_progress=lambda index, total, task_id, note: seen.append((index, total, task_id)),
    )
    assert [command for _python, command, _dataset in runner.commands] == ["problems", "score"]
    assert {call["host"] for call in model.calls} == {"http://teacher"}
    assert {call["temperature"] for call in model.calls} == {0.0}
    assert {call["num_ctx"] for call in model.calls} == {8192}
    first = model.calls[0]["messages"]
    assert first[0]["role"] == "system" and first[0]["content"].startswith(load_rules(settings))
    assert plus_eval.INSTRUCTION in first[1]["content"] and "def add(a, b):" in first[1]["content"]
    assert [sample["task_id"] for sample in runner.samples] == ["HumanEval/0", "HumanEval/1"]
    assert seen == [(1, 2, "HumanEval/0"), (2, 2, "HumanEval/1")]
    assert (result.total, result.base_passed, result.plus_passed) == (2, 2, 1)
    assert result.folder.parent == settings.evalplus_dir
    history = plus_history(settings, "humaneval")
    assert len(history) == 1 and history[0]["model"] == "qwen:7b" and history[0]["plus"] == 1
    assert history[0]["options"] == settings.model_options  # the sampling options the score was taken with
    assert plus_history(settings, "mbpp") == []
    assert not settings.state_path.exists()  # the twelve-task history in state.json is left alone


def test_run_plus_raises_when_every_request_fails(tmp_path: Path):
    settings = _settings(tmp_path)
    runner = _Runner()
    with pytest.raises(OllamaError):
        run_plus(settings, _Model(fail=True), "http://teacher", "m", python="py-eval", run_script=runner)
    assert [command for _python, command, _dataset in runner.commands] == ["problems"]


def test_run_plus_says_how_to_install_evalplus(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setattr(plus_eval, "find_evalplus_python", lambda settings: None)
    with pytest.raises(EvalPlusMissing, match=".venv-eval"):
        run_plus(settings, _Model(), "http://teacher", "m", run_script=_Runner())


def test_find_evalplus_python_prefers_the_variable_then_venv_eval(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path)
    monkeypatch.setenv("VALTA_EVALPLUS_PYTHON", "C:/elsewhere/python.exe")
    assert find_evalplus_python(settings) == "C:/elsewhere/python.exe"
    monkeypatch.delenv("VALTA_EVALPLUS_PYTHON")
    venv_python = tmp_path / ".venv-eval" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")
    assert find_evalplus_python(settings) == str(venv_python)


class _Servers:
    def __init__(self, settings):
        self.settings = settings

    def healthy(self, host):
        return True

    def tags(self, host):
        return [{"name": self.settings.teacher_model}, {"name": "valta:latest"}]


def _cli(tmp_path: Path, monkeypatch):
    settings = _settings(tmp_path)
    calls = []

    def fake_run_plus(settings, client, host, model, dataset, num_ctx=None, on_progress=None):
        calls.append((host, model, dataset, num_ctx))
        return plus_eval.PlusResult(model, dataset, 164, 150, 140, folder=settings.evalplus_dir / "run")

    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    monkeypatch.setattr(cli, "OllamaClient", _Servers)
    monkeypatch.setattr(cli, "run_plus", fake_run_plus)
    return settings, calls


def test_eval_plus_uses_the_teacher_server_unless_personal_is_asked_for(tmp_path: Path, monkeypatch, capsys):
    settings, calls = _cli(tmp_path, monkeypatch)
    cli.eval_cmd(model="", teacher=False, plus=True, mbpp=False, personal=False)
    assert calls[-1] == (settings.heavy_host, settings.teacher_model, "humaneval", settings.teacher_num_ctx)
    assert "HumanEval+ 140/164" in capsys.readouterr().out
    cli.eval_cmd(model="valta", teacher=False, plus=True, mbpp=True, personal=True)
    assert calls[-1] == (settings.self_host, "valta", "mbpp", settings.self_num_ctx)


def test_eval_plus_refuses_a_model_that_is_not_pulled_and_odd_flag_mixes(tmp_path: Path, monkeypatch):
    _, calls = _cli(tmp_path, monkeypatch)
    with pytest.raises(Exception, match="ollama pull qwen3.5:9b"):
        cli.eval_cmd(model="qwen3.5:9b", teacher=False, plus=True, mbpp=False, personal=False)
    with pytest.raises(Exception, match="go with --plus"):
        cli.eval_cmd(model="", teacher=False, plus=False, mbpp=True, personal=False)
    with pytest.raises(Exception, match="not both"):
        cli.eval_cmd(model="", teacher=True, plus=True, mbpp=False, personal=True)
    assert calls == []
