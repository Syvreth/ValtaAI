from __future__ import annotations

import json
from dataclasses import dataclass

from valta.config import Settings
from valta.control import JobStopped
from valta.memory import Memory
from valta.ollama_client import OllamaError
from valta.prompt import build_system
from valta.repomap import repo_map
from valta.tools import ToolError, ToolRunner

ACTION_HELP = """
Reply with exactly one action:

ACTION read_file
PATH relative/path
END

ACTION list_directory
PATH .
END

ACTION search_memory
QUERY what you want to recall
END

ACTION replace
PATH the/real/file.py
OLD
the exact lines to change, copied from the file
NEW
the lines that take their place
END

ACTION write_file
PATH the/real/file.py
CONTENT
the full file text
END

ACTION run_tests
CMD python -m pytest -q
END

ACTION done
SUMMARY what you changed
END

To change a file that exists, read it first, then use ACTION replace with lines copied exactly from it, indentation included. OLD must appear once in the file.
Use ACTION write_file to create a file, or to rewrite one that is short (under about 80 lines).
Do not edit the root rules. Do not delete files. Do not push git. Stay inside the project folder.
You can read and write files there. When the goal asks for a file, make it. Do not refuse.
Never copy the sample PATH relative/path or the sample QUERY. Use a real project path.
""".strip()


@dataclass
class Action:
    name: str
    path: str = ""
    content: str = ""
    command: str = ""
    query: str = ""
    summary: str = ""
    old: str = ""
    new: str = ""
    problem: str = ""  # why the action cannot run as written, told back to the model


_ACTIONS = {
    "read_file", "list_directory", "search_memory", "replace", "write_file", "run_tests", "done", "ingest", "research", "work"
}
# Actions that carry file text. That text may hold lines such as `ACTION read_file` or `END`.
_FILE_ACTIONS = {"replace", "write_file"}
REPLACE_SHAPE = (
    "A replace needs a line OLD, then the exact lines to change, then a line NEW, "
    "then the lines that take their place, then END."
)


def parse_action(text: str) -> Action | None:
    """The action in a model's reply, or None.

    A reply may talk first, or show another action before the real one, and the last action wins. The one
    exception is file text: everything after a replace or write_file line belongs to it, up to its last END
    line, so file text that holds an ACTION or END line is written whole instead of being read as an action.
    """
    found = _last_action(text.splitlines())
    if found is None:
        return None
    name, lines = found
    rest = "\n".join(lines)
    if name == "done":
        return Action(name="done", summary=_section(rest, "SUMMARY"))
    if name == "write_file":
        content = _file_text(lines, _label_at(lines, "CONTENT"), "CONTENT").strip()
        return Action(name=name, path=_section(rest, "PATH"), content=content)
    if name == "replace":
        path = _section(rest, "PATH")
        old_at = _label_at(lines, "OLD")
        new_at = _label_at(lines, "NEW", after=old_at)
        if old_at is None or new_at is None:
            # Without NEW, the OLD lines would be deleted. Ask again instead.
            return Action(name=name, path=path, problem=REPLACE_SHAPE)
        old = _trim_blank(_inline(lines[old_at], "OLD") + lines[old_at + 1 : new_at])
        return Action(name=name, path=path, old=old, new=_file_text(lines, new_at, "NEW"))
    if name == "run_tests":
        return Action(name=name, command=_section(rest, "CMD") or "python -m pytest -q")
    if name == "search_memory":
        return Action(name=name, query=_section(rest, "QUERY"))
    if name == "ingest":
        return Action(name=name, path=_section(rest, "PATH"))
    if name == "research":
        return Action(name=name, query=_section(rest, "QUERY"))
    if name == "work":
        return Action(name=name, query=_section(rest, "GOAL"))
    return Action(name=name, path=_section(rest, "PATH") or ".")


def _section(text: str, label: str) -> str:
    lines = text.splitlines()
    collecting = False
    found: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == "END":
            break
        if stripped.startswith(label + " ") or stripped == label:
            collecting = True
            if stripped.startswith(label + " "):
                found = [stripped[len(label) + 1 :].strip()]
                if label != "CONTENT":
                    return found[0]
            else:
                found = []
            continue
        if collecting:
            found.append(line)
    return "\n".join(found).strip()


def _action_name(line: str) -> str | None:
    """The action a line names, as in `ACTION read_file`, or None."""
    start = line.rfind("ACTION ")
    if start < 0:
        return None
    name = line[start + len("ACTION ") :].strip().lower()
    return name if name in _ACTIONS else None


def _last_action(lines: list[str]) -> tuple[str, list[str]] | None:
    """The last action's name and the lines after its ACTION line, skipping over file text (see parse_action)."""
    chosen = None
    index = 0
    while index < len(lines):
        name = _action_name(lines[index])
        if name is None:
            index += 1
            continue
        if name in _FILE_ACTIONS:
            return name, lines[index + 1 :]
        end = next((at for at in range(index + 1, len(lines)) if lines[at].strip() == "END"), len(lines) - 1)
        chosen = (name, lines[index + 1 : end + 1])
        index = end + 1
    return chosen


def _label_at(lines: list[str], label: str, after: int | None = -1) -> int | None:
    """Index of the first LABEL line after `after`: the label alone on its line, else the label and text."""
    if after is None:
        return None
    candidates = range(after + 1, len(lines))
    exact = next((at for at in candidates if lines[at].strip() == label), None)
    if exact is not None:
        return exact
    return next((at for at in candidates if lines[at].startswith(label + " ")), None)


def _inline(line: str, label: str) -> list[str]:
    """Text written on the label's own line, as in `OLD x = 1`."""
    return [line[len(label) + 1 :]] if line.startswith(label + " ") else []


def _file_text(lines: list[str], start: int | None, label: str) -> str:
    """The lines after the LABEL line at `start` up to the last END line, indentation kept. An END line inside
    the text is kept, since a reply's last END closes its action. Blank lines at either end are dropped."""
    if start is None:
        return ""
    body = _inline(lines[start], label) + lines[start + 1 :]
    end = max((at for at, line in enumerate(body) if line.strip() == "END"), default=len(body))
    return _trim_blank(body[:end])


def _trim_blank(lines: list[str]) -> str:
    """The lines joined, without blank lines at either end. Indentation is kept: OLD must match the file exactly."""
    lines = list(lines)
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _emit(on_event, agent: str, text: str) -> None:
    if on_event is not None:
        on_event(agent, text)
        return
    print(f"[{agent}]\n{text}\n")


def _checkpoint(control) -> None:
    if control is not None:
        control.checkpoint()


def run_goal(
    settings: Settings,
    memory: Memory,
    tools: ToolRunner,
    client,
    goal: str,
    test_cmd: str = "python -m pytest -q",
    on_event=None,
    control=None,
) -> bool:
    """Plan, edit, test, and repair. Root rules stay in force for every agent."""
    if not client.healthy(settings.heavy_host):
        raise OllamaError("The teacher model is not up. Run scripts/start-ollama.ps1, then valta models pull.")
    memory.ensure_crew()
    memory.ensure_project(tools.project)
    memory.add_goal(goal)
    lessons = _lesson_text(tools, goal)
    _checkpoint(control)
    planner_input = f"Goal:\n{goal}\n\nProject map:\n{_project_map(tools)}\n\nLessons:\n{lessons}"
    plan = _ask(settings, client, role=PLANNER_ROLE, user=planner_input)
    _emit(on_event, "planner", plan)
    feedback = "Carry out the plan, one tool call at a time." if settings.native_tools else "Carry out the plan. Use one ACTION at a time."
    summary = ""
    test_output = ""
    passed = False
    stopped = False
    writes: dict[str, str] = {}
    rounds: list[dict] = []
    failed_attempts: list[dict[str, str]] = []
    try:
        for round_index in range(settings.repair_rounds + 1):
            _checkpoint(control)
            if round_index:
                _emit(on_event, "repair", f"Attempt {round_index + 1}. {feedback}")
            round_log: dict = {"messages": [], "tests": "", "verdict": ""}
            rounds.append(round_log)
            summary, wrote = _coder_round(
                settings, client, tools, goal, plan, feedback, on_event, control, writes,
                transcript=round_log["messages"],
            )
            _checkpoint(control)
            test_output = _run_checks(tools, test_cmd)
            round_log["tests"] = test_output
            _emit(on_event, "tester", test_output)
            if not wrote:
                passed = False
                feedback = NOTHING_CHANGED + " Then finish with done."
            elif _tests_passed(test_output):
                verdict = _ask(
                    settings,
                    client,
                    role=REVIEWER_ROLE,
                    user=f"Goal:\n{goal}\n\nPlan:\n{plan}\n\nCoder summary:\n{summary}\n\nTests:\n{test_output}",
                )
                round_log["verdict"] = verdict
                _emit(on_event, "reviewer", verdict)
                passed = verdict.strip().upper().startswith("PASS")
                if passed:
                    break
                feedback = f"The reviewer said the goal is not met.\n{verdict}"
            else:
                feedback = f"The tests failed. Repair the code.\n{test_output}"
            if not passed and wrote:
                failed_attempts.append(dict(writes))
            if round_index == settings.repair_rounds:
                break
    except JobStopped:
        stopped = True
        _emit(on_event, "valta", "Stopped.")
    status = "stopped" if stopped else "passed" if passed else "failed"
    if passed:
        _save_verified_example(settings, memory, tools, goal, writes)
    if not _is_own_code(settings, tools):
        _save_transcript(settings, memory, tools, goal, status, planner_input, plan, rounds)
        if passed:
            _save_repair_preferences(memory, goal, failed_attempts, writes)
    tools.save_note(
        text=f"Goal: {goal}\nStatus: {status}\nSummary: {summary}\n\n{test_output}",
        title=goal[:200],
        kind="lesson",
    )
    if not stopped:
        _emit(on_event, "valta", f"{status}. The lesson is in memory.")
    return passed


VERIFIED_MAX_CHARS = 4000


def format_solution(writes: dict[str, str]) -> str:
    """The answer a model should give for a goal: each file, in a fenced block under its path."""
    return "\n\n".join(f"{path}\n```\n{content}\n```" for path, content in writes.items())


def _save_verified_example(settings, memory, tools, goal: str, writes: dict[str, str]) -> None:
    """A job whose tests and reviewer both passed is a verified (goal, solution) pair for fine-tuning.

    Valta's own code is skipped here: those edits wait for approval and can be denied and restored.
    """
    if not writes:
        return
    if _is_own_code(settings, tools):
        return
    answer = format_solution(writes)
    if len(answer) > VERIFIED_MAX_CHARS or memory.example_exists(goal, answer):
        return
    memory.add_example(goal, answer, source_id="verified")


def _is_own_code(settings, tools) -> bool:
    """Jobs on Valta's own code wait for approval and can be denied, so they are not training data."""
    return tools.project.resolve() == settings.root.resolve()


def _save_transcript(settings, memory, tools, goal, status, planner_input, plan, rounds) -> None:
    """Every message of every round, untrimmed, so the coder can later be trained on how it worked."""
    body = {
        "planner": {"system": PLANNER_ROLE, "user": planner_input, "plan": plan},
        "coder_system": CODER_TOOLS_ROLE if settings.native_tools else CODER_ROLE,
        "coder_tools": CODER_TOOLS if settings.native_tools else None,
        "reviewer_system": REVIEWER_ROLE,
        "rounds": rounds,
    }
    try:
        memory.add_transcript(goal, status, settings.teacher_model, str(tools.project), body)
    except Exception as exc:  # logging must never fail a job
        print(f"[memory] could not save the transcript: {exc}")


def _save_repair_preferences(memory, goal: str, failed_attempts: list[dict[str, str]], final: dict[str, str]) -> None:
    """Each round that failed its tests or review, against the round that passed, is a preference pair."""
    chosen = format_solution(final)
    if len(chosen) > VERIFIED_MAX_CHARS:
        return
    for attempt in failed_attempts:
        rejected = format_solution(attempt)
        if len(rejected) <= VERIFIED_MAX_CHARS:
            memory.add_preference(goal, chosen, rejected, source="repair")


PLANNER_ROLE = "You are the planner. Write a short plan. Do not edit files."
CODER_ROLE = (
    "You are the coder. You can read and change files inside the project. Change a file that exists with "
    "ACTION replace, and create a file with ACTION write_file. Do not refuse. Follow the action format. "
    "Never use the sample path relative/path."
)
CODER_TOOLS_ROLE = (
    "You are the coder. Change files inside the project with your tools, one call at a time. Change a file that "
    "exists with replace: read it first and copy the old lines exactly, indentation included. Create a file, or "
    "rewrite a short one (under about 80 lines), with write_file. Call done once the files are changed. "
    "Do not edit the root rules. Do not delete files. Do not push git. Stay inside the project folder. Do not refuse."
)
REVIEWER_ROLE = "You are the reviewer. Reply PASS or FAIL, then one sentence."
NOTHING_CHANGED = "No file was changed yet. Create a file with write_file, or change one with replace."


def _tool(name: str, description: str, optional: tuple[str, ...] = (), **params: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {key: {"type": "string", "description": text} for key, text in params.items()},
                "required": [key for key in params if key not in optional],
            },
        },
    }


# The coder's actions as tools for Ollama's tool calling (settings.native_tools).
CODER_TOOLS = [
    _tool("read_file", "Read a file in the project.", path="Path relative to the project folder."),
    _tool("list_directory", "List a folder in the project.", optional=("path",), path="Folder; . is the project folder."),
    _tool("search_memory", "Search Valta's memory: lessons, notes and facts.", query="What to look for."),
    _tool(
        "replace",
        "Change a file that exists: swap one block for new text. old must be copied exactly from the file and appear once.",
        path="Path relative to the project folder.",
        old="The exact lines to change, copied from the file.",
        new="The lines that take their place.",
    ),
    _tool(
        "write_file",
        "Create a file, or rewrite a short one, with its full text.",
        path="Path relative to the project folder.",
        content="The full file text.",
    ),
    _tool(
        "run_tests",
        "Run the tests. Allowed: python -m pytest and python -m compileall.",
        optional=("command",),
        command="For example: python -m pytest -q",
    ),
    _tool("done", "Finish, once the files are changed.", summary="What you changed."),
]


def tool_action(message: dict) -> Action | None:
    """The first tool call in a reply as an Action, or an ACTION block in its text when the model wrote one instead."""
    calls = message.get("tool_calls") or []
    if not calls:
        return parse_action(message.get("content") or "")
    function = calls[0].get("function") or {}
    args = function.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {}
    if not isinstance(args, dict):
        args = {}

    def get(key: str) -> str:
        return "" if args.get(key) is None else str(args[key])

    name = str(function.get("name") or "")
    if name in {"read_file", "list_directory"}:
        return Action(name=name, path=get("path") or ("." if name == "list_directory" else ""))
    if name == "search_memory":
        return Action(name=name, query=get("query"))
    if name == "replace":
        return Action(name=name, path=get("path"), old=get("old"), new=get("new"))
    if name == "write_file":
        return Action(name=name, path=get("path"), content=get("content"))
    if name == "run_tests":
        return Action(name=name, command=get("command") or "python -m pytest -q")
    if name == "done":
        return Action(name=name, summary=get("summary"))
    return None


def _coder_round(
    settings, client, tools, goal, plan, feedback, on_event=None, control=None, writes=None, transcript=None
) -> tuple[str, bool]:
    """One coder pass. `transcript`, when given, receives every message untrimmed, in order."""
    native = settings.native_tools
    opening = f"Goal:\n{goal}\n\nPlan:\n{plan}\n\nProject map:\n{_project_map(tools)}\n\n{feedback}"
    if not native:
        opening += f"\n\n{ACTION_HELP}"
    history = [{"role": "user", "content": opening}]
    if transcript is not None:
        transcript.append(history[0])
    summary = ""
    wrote = False
    for _step in range(settings.work_max_steps):
        _checkpoint(control)
        if native:
            reply = _ask_tools(settings, client, history)
            action = tool_action(reply)
            if action is None and wrote and not reply["tool_calls"] and reply["content"].strip():
                action = Action(name="done", summary=reply["content"].strip())  # a tool-calling model ends in prose
            if reply["tool_calls"]:
                reply = {**reply, "tool_calls": reply["tool_calls"][:1]}  # one call per turn, one result each
            else:
                reply = {"role": "assistant", "content": reply["content"]}
        else:
            text = _ask(settings, client, role=CODER_ROLE, user="", history=history)
            reply, action = {"role": "assistant", "content": text}, parse_action(text)
        if action is None:
            result = "That was not one of your tools. Call one." if native else "That was not an action. " + ACTION_HELP
        elif action.name == "done":
            if not wrote:
                result = NOTHING_CHANGED
                _emit(on_event, "coder", result)
            else:
                summary = action.summary or "Finished the edit pass."
                _emit(on_event, "coder", summary)
                if transcript is not None:
                    transcript.append(reply)
                return summary, True
        else:
            rejected = _example_action(action)
            if rejected:
                result = rejected
            else:
                result = _execute(tools, action)
                if str(result).startswith(("Wrote ", "Replaced ")):
                    wrote = True
                    if writes is not None:
                        writes[action.path] = action.content if action.name == "write_file" else tools.read_file(action.path)
            detail = action.path or action.command or action.query or action.name
            _emit(on_event, "coder", f"{action.name} {detail}\n{result}")
        if reply.get("tool_calls"):
            called = (reply["tool_calls"][0].get("function") or {}).get("name", "")
            answer = {"role": "tool", "content": str(result), "tool_name": called}
        else:
            answer = {"role": "user", "content": str(result)}
        exchange = [reply, answer]
        history.extend(exchange)
        if transcript is not None:
            transcript.extend(exchange)
        history = _trim_history(history)
    return summary or "Stopped at the step limit before the coder said it was done.", wrote


CODER_RECENT_TURNS = 3


def _trim_history(history: list[dict]) -> list[dict]:
    """Keep the opening message (goal, plan, action format) and the last few exchanges.

    The opening message is a user turn and each exchange is an assistant reply plus a user result,
    so the trimmed list still alternates roles.
    """
    keep = CODER_RECENT_TURNS * 2
    if len(history) <= keep + 1:
        return history
    return [history[0], *history[-keep:]]


def _example_action(action: Action) -> str | None:
    if action.problem:
        return action.problem
    path = action.path.replace("\\", "/")
    if action.name in {"read_file", "write_file", "replace"} and path in {"relative/path", "the/real/file.py"}:
        return "That PATH is the sample from the instructions. Use a real file from the project map."
    if action.name == "search_memory" and action.query.strip() == "what you want to recall":
        return "That QUERY is the sample from the instructions. Write the file, or search for a real topic."
    if action.name == "write_file" and action.content.strip() in {"", "file contents", "the full file text"}:
        return "That CONTENT is the sample from the instructions. Put the real file text after CONTENT."
    if action.name == "replace" and action.old.strip() in {"", "the exact lines to change, copied from the file"}:
        return "That OLD is the sample from the instructions. Read the file, then copy the exact lines to change."
    return None


def _execute(tools: ToolRunner, action: Action) -> str:
    try:
        if action.name == "read_file":
            return tools.read_file(action.path)
        if action.name == "list_directory":
            return "\n".join(tools.list_directory(action.path or "."))
        if action.name == "search_memory":
            records = tools.search_memory(action.query)
            return "\n\n".join(f"{record.title}\n{record.text}" for record in records) or "No memory matched."
        if action.name == "write_file":
            if not action.path:
                return "PATH is required."
            return tools.write_file(action.path, action.content)
        if action.name == "replace":
            if not action.path:
                return "PATH is required."
            return tools.replace_text(action.path, action.old, action.new)
        if action.name == "run_tests":
            return tools.run_tests(action.command)
    except ToolError as exc:
        return str(exc)
    return "That action is not available."


def _run_checks(tools: ToolRunner, test_cmd: str) -> str:
    result = tools.run_tests(test_cmd)
    if result.startswith("exit 5") and "pytest" in test_cmd:
        follow = tools.run_tests("python -m compileall .")
        return result + "\n\nNo pytest tests were collected. Compile check:\n" + follow
    return result


def _tests_passed(output: str) -> bool:
    if "Compile check:" in output:
        return output.split("Compile check:", 1)[1].lstrip().startswith("exit 0")
    return output.startswith("exit 0")


def _lesson_text(tools: ToolRunner, goal: str) -> str:
    try:
        records = tools.search_memory(goal, kind="lesson", k=3)
    except Exception:
        return "None yet."
    if not records:
        return "None yet."
    return "\n\n".join(record.text for record in records)


def _project_map(tools: ToolRunner) -> str:
    try:
        return repo_map(tools.project)
    except OSError as exc:
        return f"Could not map the project: {exc}"


def _ask_tools(settings: Settings, client, history: list) -> dict:
    messages = [{"role": "system", "content": build_system(settings, role=CODER_TOOLS_ROLE)}, *history]
    return client.chat_with_tools(
        settings.heavy_host, settings.teacher_model, messages, settings.teacher_num_ctx, CODER_TOOLS
    )


def _ask(settings: Settings, client, role: str, user: str, history: list | None = None) -> str:
    messages = [{"role": "system", "content": build_system(settings, role=role)}]
    if history:
        messages.extend(history)
    elif user:
        messages.append({"role": "user", "content": user})
    return client.chat(settings.heavy_host, settings.teacher_model, messages, settings.teacher_num_ctx)
