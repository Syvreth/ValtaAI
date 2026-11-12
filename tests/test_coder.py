from pathlib import Path

import pytest

from valta.config import load_settings
from valta.engineer import parse_action, run_goal
from valta.memory import Memory
from valta.repomap import repo_map
from valta.tools import ToolError, ToolRunner
from valta.vectors import Vectors

CALC = '''class Calc:
    def __init__(self, start=0):
        self.value = start

    def add(self, amount):
        self.value += amount
        return self.value


def double(x, *rest, scale=2, **extra):
    return x * scale
'''


def _settings(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "config.toml").write_text((root / "config.toml").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "rules").mkdir()
    (tmp_path / "rules" / "root.md").write_text((root / "rules" / "root.md").read_text(encoding="utf-8"), encoding="utf-8")
    return load_settings(tmp_path)


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "proj"
    (project / "tests").mkdir(parents=True)
    (project / "calc.py").write_text(CALC, encoding="utf-8")
    (project / "notes.md").write_text("# Notes\n", encoding="utf-8")
    (project / "tests" / "test_calc.py").write_text(
        "def helper():\n    return 1\n\ndef test_add():\n    pass\n\ndef test_double():\n    pass\n", encoding="utf-8"
    )
    for skipped in (".venv/lib/x.py", ".venv-eval/y.py", "data/state.json", "__pycache__/calc.cpython-313.pyc"):
        path = project / skipped
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x = 1\n", encoding="utf-8")
    return project


def test_repo_map_lists_files_and_outlines_python(tmp_path: Path):
    text = repo_map(_project(tmp_path))
    assert text.splitlines()[0] == "calc.py (11 lines)"
    assert "  class Calc\n    def __init__(start)\n    def add(amount)\n  def double(x, *rest, scale, **extra)" in text
    assert "notes.md" in text
    assert "tests/test_calc.py (8 lines, 2 tests)" in text and "helper" not in text
    assert ".venv" not in text and "data/" not in text and "pycache" not in text


def test_repo_map_drops_arguments_then_outlines_then_files_to_stay_short(tmp_path: Path):
    project = _project(tmp_path)
    full = repo_map(project)
    without_args = repo_map(project, max_chars=len(full) + 60)
    assert "def add(amount)" in full
    assert "def add" in without_args and "def add(amount)" not in without_args
    tight = repo_map(project, max_chars=len("calc.py (11 lines)\nnotes.md\ntests/test_calc.py (8 lines, 2 tests)\n") + 75)
    assert "class Calc" not in tight and "left out" in tight and "notes.md" in tight
    for index in range(40):
        (project / f"module_{index:02}.txt").write_text("", encoding="utf-8")
    assert repo_map(project, max_chars=300).splitlines()[-1].endswith("more files")


def _tools(tmp_path: Path, project: Path) -> ToolRunner:
    settings = _settings(tmp_path)
    return ToolRunner(settings, Memory(settings), Vectors(settings), lambda text: [0.1, 0.2], project)


def test_replace_swaps_exactly_one_block_and_keeps_line_endings(tmp_path: Path):
    project = _project(tmp_path)
    tools = _tools(tmp_path, project)
    assert tools.replace_text("calc.py", "        return self.value\n", "        return round(self.value, 2)\n") == (
        "Replaced one block in calc.py"
    )
    assert "return round(self.value, 2)" in (project / "calc.py").read_text(encoding="utf-8")
    (project / "crlf.py").write_bytes(b"a = 1\r\nb = 2\r\n")
    tools.replace_text("crlf.py", "b = 2", "b = 3\nc = 4")
    assert (project / "crlf.py").read_bytes() == b"a = 1\r\nb = 3\r\nc = 4\r\n"


def test_replace_leaves_the_rest_of_a_mixed_line_ending_file_alone(tmp_path: Path):
    project = _project(tmp_path)
    tools = _tools(tmp_path, project)
    (project / "mixed.py").write_bytes(b"a = 1\r\nb = 2\nc = 3\nd = 4\r\n")
    tools.replace_text("mixed.py", "b = 2\nc = 3", "b = 20\nc = 30")
    assert (project / "mixed.py").read_bytes() == b"a = 1\r\nb = 20\nc = 30\nd = 4\r\n"
    tools.replace_text("mixed.py", "a = 1\nb = 20", "a = 10\nb = 2")  # a block that spans both endings matches
    assert (project / "mixed.py").read_bytes() == b"a = 10\r\nb = 2\nc = 30\nd = 4\r\n"


def test_replace_refuses_a_missing_or_repeated_block_and_protected_files(tmp_path: Path):
    project = _project(tmp_path)
    tools = _tools(tmp_path, project)
    before = (project / "calc.py").read_text(encoding="utf-8")
    with pytest.raises(ToolError, match="not found"):
        tools.replace_text("calc.py", "return self.total", "return 0")
    with pytest.raises(ToolError, match="appears 3 times"):
        tools.replace_text("calc.py", "self.value", "self.total")
    with pytest.raises(ToolError, match="write_file"):
        tools.replace_text("missing.py", "a", "b")
    with pytest.raises(ToolError, match="protected"):
        tools.replace_text(".venv-eval/y.py", "x = 1", "x = 2")
    assert (project / "calc.py").read_text(encoding="utf-8") == before


def test_parse_action_reads_a_replace_and_keeps_indentation():
    action = parse_action(
        "I will fix it.\nACTION replace\nPATH calc.py\nOLD\n\n        return self.value\nNEW\n        return 0\n\nEND"
    )
    assert action is not None and action.name == "replace"
    assert (action.path, action.old, action.new) == ("calc.py", "        return self.value", "        return 0")


def test_file_text_that_looks_like_an_action_or_end_stays_file_text():
    # Editing Valta's own engineer.py: the new text holds ACTION lines and an END line of its own.
    new = 'HELP = """\nACTION read_file\nPATH relative/path\nEND\n"""'
    action = parse_action(f"ACTION replace\nPATH valta/engineer.py\nOLD\nHELP = ''\nNEW\n{new}\nEND\n")
    assert action is not None and action.name == "replace"
    assert (action.path, action.old, action.new) == ("valta/engineer.py", "HELP = ''", new)

    sql = "CREATE TRIGGER t AFTER INSERT ON x\nBEGIN\n  SELECT 1;\nEND\n;"
    action = parse_action(f"ACTION write_file\nPATH schema.sql\nCONTENT\n{sql}\nEND")
    assert action is not None and (action.name, action.path, action.content) == ("write_file", "schema.sql", sql)


def test_the_last_action_wins_after_talk_or_an_example():
    action = parse_action(
        "I could list the folder:\nACTION list_directory\nPATH .\nEND\n"
        "But I know the file, so:\nACTION read_file\nPATH calc.py\nEND"
    )
    assert action is not None and (action.name, action.path) == ("read_file", "calc.py")
    assert parse_action("Nothing to do here.") is None


def test_a_replace_without_new_is_sent_back_instead_of_deleting_the_old_lines():
    from valta.engineer import REPLACE_SHAPE, _example_action

    action = parse_action("ACTION replace\nPATH calc.py\nOLD\n        return self.value\nEND")
    assert action is not None and action.problem == REPLACE_SHAPE
    assert _example_action(action) == REPLACE_SHAPE


def test_facts_queued_at_exit_are_waited_for(monkeypatch):
    import threading

    import valta.chat as chat

    release = threading.Event()
    seen = []

    def slow_capture(group):
        release.wait(5)
        seen.extend(group)

    from types import SimpleNamespace

    monkeypatch.setattr(chat, "_capture_facts", slow_capture)
    exchange = SimpleNamespace(tools=None, host="http://h", model="m")
    chat._queue_facts(exchange)
    assert chat.wait_for_facts(timeout=0.05) is False  # still extracting
    release.set()
    chat.save_pending_facts(timeout=5)
    assert seen == [exchange] and chat.wait_for_facts(timeout=0)


class _Scripted:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def healthy(self, host):
        return True

    def chat(self, host, model, messages, num_ctx):
        self.prompts.append(messages)
        return self.replies.pop(0)


def test_the_crew_edits_an_existing_file_in_place(tmp_path: Path):
    settings = _settings(tmp_path)
    memory = Memory(settings)
    project = _project(tmp_path)
    tools = ToolRunner(settings, memory, Vectors(settings), lambda text: [0.3, 0.1, 0.2], project)
    client = _Scripted(
        [
            "Make double triple by default.",
            "ACTION read_file\nPATH calc.py\nEND",
            "ACTION replace\nPATH calc.py\nOLD\ndef double(x, *rest, scale=2, **extra):\nNEW\ndef double(x, *rest, scale=3, **extra):\nEND",
            "ACTION done\nSUMMARY scale is 3 now\nEND",
            "PASS It triples.",
        ]
    )
    assert run_goal(settings, memory, tools, client, "make double use scale 3", "python -m compileall -q .")
    assert "scale=3" in (project / "calc.py").read_text(encoding="utf-8")
    planner_user = client.prompts[0][-1]["content"]
    coder_opening = client.prompts[1][1]["content"]
    assert "Project map:\ncalc.py (11 lines)" in planner_user and "def double" in coder_opening
    [example] = memory.all_examples()
    assert "class Calc" in example["assistant_text"] and "scale=3" in example["assistant_text"]


class _ToolModel:
    """Plans and reviews in text, and codes through scripted tool calls (or text, for a string)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []

    def healthy(self, host):
        return True

    def chat(self, host, model, messages, num_ctx):
        return "Change the scale." if "You are the planner" in messages[0]["content"] else "PASS It works."

    def chat_with_tools(self, host, model, messages, num_ctx, tools):
        self.sent.append((messages, tools))
        reply = self.replies.pop(0)
        if isinstance(reply, str):
            return {"role": "assistant", "content": reply, "tool_calls": []}
        name, args = reply
        return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def test_the_coder_can_work_through_native_tool_calls(tmp_path: Path):
    from valta.engineer import CODER_TOOLS

    settings = _settings(tmp_path)
    settings.native_tools = True
    memory = Memory(settings)
    project = _project(tmp_path)
    tools = ToolRunner(settings, memory, Vectors(settings), lambda text: [0.3, 0.1, 0.2], project)
    client = _ToolModel(
        [
            ("read_file", {"path": "calc.py"}),
            ("replace", {"path": "calc.py", "old": "scale=2", "new": "scale=3"}),
            ("done", {"summary": "scale is 3 now"}),
        ]
    )
    assert run_goal(settings, memory, tools, client, "make double use scale 3", "python -m compileall -q .")
    assert "scale=3" in (project / "calc.py").read_text(encoding="utf-8")
    messages, offered = client.sent[1]
    assert offered == CODER_TOOLS
    assert "You are the coder. Change files inside the project with your tools" in messages[0]["content"]
    assert "ACTION read_file" not in messages[1]["content"]  # no text format when tools are native
    assert messages[2]["tool_calls"][0]["function"]["name"] == "read_file"
    assert messages[3]["role"] == "tool" and messages[3]["tool_name"] == "read_file" and "class Calc" in messages[3]["content"]
    [transcript] = memory.all_transcripts()
    assert transcript["body"]["coder_tools"] == CODER_TOOLS


def test_a_native_coder_may_still_write_an_action_block_and_finish_in_prose(tmp_path: Path):
    settings = _settings(tmp_path)
    settings.native_tools = True
    memory = Memory(settings)
    project = _project(tmp_path)
    tools = ToolRunner(settings, memory, Vectors(settings), lambda text: [0.3, 0.1, 0.2], project)
    client = _ToolModel(["ACTION write_file\nPATH hello.py\nCONTENT\nprint('hi')\nEND", "I added hello.py."])
    assert run_goal(settings, memory, tools, client, "add hello.py", "python -m compileall -q .")
    assert (project / "hello.py").read_text(encoding="utf-8") == "print('hi')"
    assert client.sent[1][0][-1] == {"role": "user", "content": "Wrote hello.py"}


def test_tool_action_reads_string_arguments_and_ignores_unknown_tools():
    from valta.engineer import tool_action

    call = lambda name, args: {"content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}
    replace = tool_action(call("replace", '{"path": "a.py", "old": "x = 1", "new": "x = 2"}'))
    assert (replace.name, replace.path, replace.old, replace.new) == ("replace", "a.py", "x = 1", "x = 2")
    assert tool_action(call("list_directory", {})).path == "."
    assert tool_action(call("run_tests", {})).command == "python -m pytest -q"
    assert tool_action(call("delete_everything", {"path": "."})) is None
