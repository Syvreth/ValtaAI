from __future__ import annotations

from dataclasses import dataclass

from valta.config import Settings
from valta.memory import Memory
from valta.ollama_client import OllamaError
from valta.prompt import build_system
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

ACTION write_file
PATH relative/path
CONTENT
file contents
END

ACTION run_tests
CMD python -m pytest -q
END

ACTION done
SUMMARY what you changed
END

Do not edit the root rules. Do not delete files. Do not push git. Stay inside the project folder.
""".strip()


@dataclass
class Action:
    name: str
    path: str = ""
    content: str = ""
    command: str = ""
    query: str = ""
    summary: str = ""


def parse_action(text: str) -> Action | None:
    marker = "ACTION "
    start = text.rfind(marker)
    if start < 0:
        return None
    body = text[start + len(marker) :]
    name, _, rest = body.partition("\n")
    name = name.strip().lower()
    if name not in {"read_file", "list_directory", "search_memory", "write_file", "run_tests", "done", "ingest", "research", "work"}:
        return None
    if name == "done":
        return Action(name="done", summary=_section(rest, "SUMMARY"))
    if name == "write_file":
        return Action(name=name, path=_section(rest, "PATH"), content=_section(rest, "CONTENT"))
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


def run_goal(
    settings: Settings,
    memory: Memory,
    tools: ToolRunner,
    client,
    goal: str,
    test_cmd: str = "python -m pytest -q",
) -> bool:
    """Plan, edit, test, and repair. Root rules stay in force for every agent."""
    if not client.healthy(settings.heavy_host):
        raise OllamaError("The teacher model is not up. Run scripts/start-ollama.ps1, then valta models pull.")
    memory.ensure_crew()
    memory.ensure_project(tools.project)
    memory.add_goal(goal)
    lessons = _lesson_text(tools, goal)
    listing = _safe_list(tools)
    plan = _ask(
        settings,
        client,
        role="You are the planner. Write a short plan. Do not edit files.",
        user=f"Goal:\n{goal}\n\nProject files:\n{listing}\n\nLessons:\n{lessons}",
    )
    print(f"[planner]\n{plan}\n")
    feedback = "Carry out the plan. Use one ACTION at a time."
    summary = ""
    test_output = ""
    passed = False
    for round_index in range(settings.repair_rounds + 1):
        summary = _coder_round(settings, client, tools, goal, plan, feedback)
        test_output = _run_checks(tools, test_cmd)
        print(f"[tester]\n{test_output}\n")
        if _tests_passed(test_output):
            verdict = _ask(
                settings,
                client,
                role="You are the reviewer. Reply PASS or FAIL, then one sentence.",
                user=f"Goal:\n{goal}\n\nPlan:\n{plan}\n\nCoder summary:\n{summary}\n\nTests:\n{test_output}",
            )
            print(f"[reviewer]\n{verdict}\n")
            passed = verdict.strip().upper().startswith("PASS")
            if passed:
                break
            feedback = f"The reviewer said the goal is not met.\n{verdict}"
        else:
            feedback = f"The tests failed. Repair the code.\n{test_output}"
        if round_index == settings.repair_rounds:
            break
    status = "passed" if passed else "failed"
    tools.save_note(
        text=f"Goal: {goal}\nStatus: {status}\nSummary: {summary}\n\n{test_output}",
        title=goal[:200],
        kind="lesson",
    )
    print(f"[valta] {status}. The lesson is in memory.")
    return passed


def _coder_round(settings: Settings, client, tools: ToolRunner, goal: str, plan: str, feedback: str) -> str:
    history = [{"role": "user", "content": f"Goal:\n{goal}\n\nPlan:\n{plan}\n\n{feedback}\n\n{ACTION_HELP}"}]
    summary = ""
    for _step in range(settings.work_max_steps):
        reply = _ask(
            settings,
            client,
            role="You are the coder. Change only files inside the project. Follow the action format.",
            user="",
            history=history,
        )
        action = parse_action(reply)
        if action is None:
            result = "That was not an action. " + ACTION_HELP
        elif action.name == "done":
            summary = action.summary or "Finished the edit pass."
            print(f"[coder] {summary}")
            return summary
        else:
            result = _execute(tools, action)
            print(f"[coder] {action.name} {action.path or action.command or action.query}")
        history.append({"role": "assistant", "content": reply})
        history.append({"role": "user", "content": str(result)})
        history = history[-8:]
    return summary or "Stopped at the step limit before ACTION done."


def _execute(tools: ToolRunner, action: Action) -> str:
    try:
        if action.name == "read_file":
            return tools.read_file(action.path)
        if action.name == "list_directory":
            return "\n".join(tools.list_directory(action.path or "."))
        if action.name == "search_memory":
            records = tools.search_memory(action.query)
            return "\n\n".join(f"{record.title}\n{record.text[:800]}" for record in records) or "No memory matched."
        if action.name == "write_file":
            if not action.path:
                return "PATH is required."
            return tools.write_file(action.path, action.content)
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
    return "\n\n".join(record.text[:800] for record in records)


def _safe_list(tools: ToolRunner) -> str:
    try:
        names = tools.list_directory(".")
    except ToolError as exc:
        return str(exc)
    return "\n".join(names[:80])


def _ask(settings: Settings, client, role: str, user: str, history: list | None = None) -> str:
    messages = [{"role": "system", "content": build_system(settings, role=role)}]
    if history:
        messages.extend(history)
    elif user:
        messages.append({"role": "user", "content": user})
    return client.chat(settings.heavy_host, settings.teacher_model, messages, settings.teacher_num_ctx)
