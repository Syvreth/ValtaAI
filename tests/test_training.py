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
