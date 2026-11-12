import json
from pathlib import Path

from valta.chat import _capture_correction
from valta.config import load_settings
from valta.engineer import format_solution, run_goal
from valta.evals import BUILTIN_TASKS
from valta.memory import Memory
from valta.tools import ToolRunner
from valta.train import build_pairs, tokenize_example
from valta.vectors import Vectors


def _settings(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "config.toml").write_text((root / "config.toml").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "rules").mkdir()
    (tmp_path / "rules" / "root.md").write_text((root / "rules" / "root.md").read_text(encoding="utf-8"), encoding="utf-8")
    return load_settings(tmp_path)


class _ChatMLTokenizer:
    """Just enough of a Hugging Face tokenizer: ChatML text, one id per character."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) for char in text]}


def _decode(ids):
    return "".join(chr(i) for i in ids)


def test_only_the_answer_is_trained_on():
    messages = [
        {"role": "system", "content": "RULES"},
        {"role": "user", "content": "Q?"},
        {"role": "assistant", "content": "ANSWER"},
    ]
    item = tokenize_example(_ChatMLTokenizer(), messages, max_len=500)
    assert item is not None
    assert len(item["input_ids"]) == len(item["labels"]) == len(item["attention_mask"])
    trained = _decode([i for i in item["labels"] if i != -100])
    assert trained == "ANSWER<|im_end|>\n"
    assert "RULES" not in trained and "Q?" not in trained
    assert _decode(item["input_ids"]).startswith("<|im_start|>system\nRULES")


def test_examples_longer_than_the_limit_are_skipped_not_cut():
    messages = [
        {"role": "system", "content": "RULES"},
        {"role": "user", "content": "Q?"},
        {"role": "assistant", "content": "A" * 200},
    ]
    assert tokenize_example(_ChatMLTokenizer(), messages, max_len=100) is None
    assert tokenize_example(_ChatMLTokenizer(), messages, max_len=1000) is not None


def test_dataset_is_cumulative_deduplicated_and_free_of_eval_prompts(tmp_path: Path):
    settings = _settings(tmp_path)
    memory = Memory(settings)
    first = memory.add_example("How do I package it?", "Use pyproject.toml.")
    memory.mark_examples_used([first])  # a used example must still be trained on again
    memory.add_example("How do I package it?", "Use pyproject.toml.")  # exact duplicate
    memory.add_example("What editor do I like?", "Neovim.")
    memory.add_example(BUILTIN_TASKS[0].prompt, "def fizzbuzz(n): return []")  # would leak the eval
    path, ids = build_pairs(settings, memory, client=None)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    users = [row["messages"][1]["content"] for row in rows]
    assert users.count("How do I package it?") == 1
    assert "What editor do I like?" in users
    assert BUILTIN_TASKS[0].prompt not in users
    assert first in ids


def test_a_bare_no_is_not_a_correction(tmp_path: Path):
    settings = _settings(tmp_path)
    memory = Memory(settings)
    history = [{"role": "user", "content": "What is my dog's name?"}, {"role": "assistant", "content": "Rex?"}]
    _capture_correction(memory, history, "no, that's wrong")
    assert memory.example_count() == 0
    _capture_correction(memory, history, "correction: Her name is Biscuit.")
    rows = memory.all_examples()
    assert len(rows) == 1
    assert rows[0]["user_text"] == "What is my dog's name?"
    assert rows[0]["assistant_text"] == "Her name is Biscuit."


class _Scripted:
    def __init__(self, replies):
        self.replies = list(replies)

    def healthy(self, host):
        return True

    def chat(self, host, model, messages, num_ctx):
        return self.replies.pop(0)


_GOOD_RUN = [
    "Add hello.py.",
    "ACTION write_file\nPATH hello.py\nCONTENT\nprint('hi')\nEND",
    "ACTION done\nSUMMARY Added hello.py\nEND",
    "PASS The file is in place.",
]


def test_a_verified_job_becomes_a_training_example(tmp_path: Path):
    settings = _settings(tmp_path)
    memory = Memory(settings)
    project = tmp_path / "proj"
    project.mkdir()
    tools = ToolRunner(settings, memory, Vectors(settings), lambda text: [0.3, 0.1, 0.2], project)
    assert run_goal(settings, memory, tools, _Scripted(_GOOD_RUN), "add hello.py")
    rows = memory.all_examples()
    assert len(rows) == 1
    assert rows[0]["source_id"] == "verified"
    assert rows[0]["user_text"] == "add hello.py"
    assert rows[0]["assistant_text"] == format_solution({"hello.py": "print('hi')"})
    # the same success again is not saved twice
    assert run_goal(settings, memory, tools, _Scripted(_GOOD_RUN), "add hello.py")
    assert len(memory.all_examples()) == 1


def test_failed_jobs_and_edits_to_valta_itself_are_not_saved(tmp_path: Path):
    settings = _settings(tmp_path)
    memory = Memory(settings)
    project = tmp_path / "proj"
    project.mkdir()
    tools = ToolRunner(settings, memory, Vectors(settings), lambda text: [0.3, 0.1, 0.2], project)
    failed = _GOOD_RUN[:-1] + ["FAIL It does not meet the goal."]
    settings.repair_rounds = 0
    assert run_goal(settings, memory, tools, _Scripted(failed), "add hello.py") is False
    assert memory.all_examples() == []
    # a job aimed at Valta's own folder waits for approval, so it is never trained on
    own = ToolRunner(settings, memory, Vectors(settings), lambda text: [0.3, 0.1, 0.2], settings.root)
    assert run_goal(settings, memory, own, _Scripted(_GOOD_RUN), "add hello.py")
    assert memory.all_examples() == []


def test_gguf_conversion_uses_this_python_and_a_supported_outtype(tmp_path: Path, monkeypatch):
    import subprocess
    import sys

    from valta.train import convert_to_gguf

    calls = []
    monkeypatch.setenv("LLAMA_CPP_CONVERT", "convert_hf_to_gguf.py")
    monkeypatch.setattr(subprocess, "run", lambda argv, check: calls.append(argv))
    out = convert_to_gguf(tmp_path / "merged", tmp_path / "models" / "x.gguf")
    assert out == tmp_path / "models" / "x.gguf"
    [argv] = calls
    assert argv[0] == sys.executable
    assert argv[argv.index("--outtype") + 1] in {"f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto"}


def test_converter_is_found_in_a_local_llama_cpp_clone(tmp_path: Path, monkeypatch):
    import shutil

    from valta.train import find_converter

    script = tmp_path / "llama.cpp" / "convert_hf_to_gguf.py"
    script.parent.mkdir()
    script.write_text("", encoding="utf-8")
    monkeypatch.delenv("LLAMA_CPP_CONVERT", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert find_converter() == str(script)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "nowhere"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    assert find_converter() is None
    monkeypatch.delenv("LOCALAPPDATA")
    home_script = tmp_path / "home" / "AppData" / "Local" / "llama.cpp" / "convert_hf_to_gguf.py"
    home_script.parent.mkdir(parents=True)
    home_script.write_text("", encoding="utf-8")
    assert find_converter() == str(home_script)


def test_resume_finds_the_newest_merge_and_the_examples_it_trained_on(tmp_path: Path):
    from valta.train import examples_before, latest_merged

    settings = _settings(tmp_path)
    assert latest_merged(settings) is None
    for name in ("20261101000000", "20261112001329", "not-a-run"):
        (settings.merged_dir / name).mkdir(parents=True)
        (settings.merged_dir / name / "config.json").write_text("{}", encoding="utf-8")
    (settings.merged_dir / "20991231000000").mkdir()  # no config.json: an unfinished merge
    assert latest_merged(settings).name == "20261112001329"
    memory = Memory(settings)
    old = memory.add_example("q1", "a1")
    memory.conn.execute("UPDATE training_examples SET created_at = ? WHERE id = ?", ("2026-11-11T23:00:00+00:00", old))
    new = memory.add_example("q2", "a2")
    memory.conn.execute("UPDATE training_examples SET created_at = ? WHERE id = ?", ("2026-11-12T01:00:00+00:00", new))
    memory.conn.commit()
    assert examples_before(memory, "20261112001329") == [old]


def test_ollama_create_uploads_a_local_gguf_and_names_it_in_files(tmp_path: Path, monkeypatch):
    import contextlib
    import hashlib

    import httpx

    from valta.ollama_client import OllamaClient, parse_modelfile

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"weights")
    digest = "sha256:" + hashlib.sha256(b"weights").hexdigest()
    assert parse_modelfile(f"FROM {gguf.as_posix()}\nPARAMETER num_ctx 32768\n") == (gguf.as_posix(), {"num_ctx": 32768})

    sent = {}

    class _Head:
        status_code = 404

    class _Posted:
        status_code = 201
        text = ""

    def fake_post(url, content, timeout):
        sent["blob_url"] = url
        sent["blob"] = b"".join(content)
        return _Posted()

    class _Stream:
        status_code = 200

        def iter_lines(self):
            return iter(['{"status": "success"}'])

    @contextlib.contextmanager
    def fake_stream(method, url, json, timeout):
        sent["create"] = json
        yield _Stream()

    monkeypatch.setattr(httpx, "head", lambda url, timeout: _Head())
    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(httpx, "stream", fake_stream)
    client = OllamaClient(settings=None)
    client.create("http://h", "valta-candidate", f"FROM {gguf.as_posix()}\nPARAMETER num_ctx 32768\n")
    assert sent["blob_url"] == f"http://h/api/blobs/{digest}"
    assert sent["blob"] == b"weights"
    assert sent["create"] == {
        "model": "valta-candidate",
        "stream": True,
        "parameters": {"num_ctx": 32768},
        "files": {"model.gguf": digest},
    }
    client.create("http://h", "valta", "FROM qwen2.5-coder:3b\n")
    assert sent["create"]["from"] == "qwen2.5-coder:3b" and "files" not in sent["create"]


def test_default_gate_refuses_a_model_that_lost_a_task(tmp_path: Path):
    from valta.evals import EvalResult, should_promote

    settings = _settings(tmp_path)
    assert settings.eval_tolerance == 0.0
    baseline = EvalResult("qwen2.5-coder:3b", passed=10, total=12)
    assert not should_promote(baseline, EvalResult("valta-candidate", passed=9, total=12), settings.eval_tolerance)[0]
    assert should_promote(baseline, EvalResult("valta-candidate", passed=10, total=12), settings.eval_tolerance)[0]


def test_rollback_after_the_first_promotion_returns_to_the_base_model(tmp_path: Path):
    import pytest

    from valta.config import load_state, save_state
    from valta.train import rollback

    settings = _settings(tmp_path)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    save_state(settings, {"personal_model": "valta"})

    class _NoCreate:
        def create(self, *args):
            raise AssertionError("rolling back to the base model should not create anything")

    rollback(settings, _NoCreate())
    assert load_state(settings)["personal_model"] == settings.self_model
    with pytest.raises(RuntimeError):
        rollback(settings, _NoCreate())
