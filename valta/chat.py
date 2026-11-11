from __future__ import annotations

import re
import threading
from pathlib import Path

from valta.config import Settings, load_state
from valta.engineer import parse_action, run_goal
from valta.ingest import ingest_path
from valta.memory import Memory
from valta.ollama_client import OllamaClient, OllamaError
from valta.prompt import build_system
from valta.research import research_loop, research_once
from valta.tools import ToolError, ToolRunner

CHAT_ROLE = """
You are Valta, in one ongoing chat. The user should not have to run commands.
The first line of every reply is exactly one of:
TALK
INGEST <path>
RESEARCH <topic>
WORK <goal>
DEPLOY <what>
Use TALK when they are conversing. Use INGEST when they want a file or folder read into memory. Use RESEARCH when they want you to look something up. Use WORK when they want software created or changed, or when they ask you to create, edit, read, or list files. Use DEPLOY when they want a finished project released. If the first line is TALK, the rest of the reply is what you say. If it is a task, the first line is the whole reply.
You have a filesystem. Do not claim that you cannot create or read files. Do not research how to make a file the user already asked you to make.
""".strip()

_DOC_SUFFIXES = {".pdf", ".epub", ".txt", ".md", ".markdown"}

BUILD_CUES = (
    "code",
    "function",
    "bug",
    "implement",
    "refactor",
    "script",
    "compile",
    "repo",
    "error",
    "build",
    "python",
    "project",
    "research",
    "look up",
)
PERSONAL_CUES = (
    "remember",
    "about me",
    "my name",
    "i like",
    "i prefer",
    "who am i",
    "i am",
    "my goal",
)


def find_docs(text: str) -> list[Path]:
    raw_candidates = re.findall(r'"([^"]+)"', text)
    raw_candidates += re.findall(r"[A-Za-z]:\\[^\r\n\"']+", text)
    stripped = text.strip().strip('"')
    if stripped:
        raw_candidates.append(stripped)
    found: list[Path] = []
    seen: set[str] = set()
    for raw in raw_candidates:
        path = Path(raw.strip().rstrip(".,)")).expanduser()
        if path.suffix.lower() not in _DOC_SUFFIXES or not path.exists():
            continue
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            found.append(path)
    return found


def interpret(answer: str) -> tuple[str, str, str]:
    """Return kind, argument, and the spoken text when the reply is just conversation."""
    cleaned = answer.strip()
    if not cleaned:
        return "talk", "", ""
    line, _, rest = cleaned.partition("\n")
    line = line.strip()
    upper = line.upper()
    if upper == "TALK":
        return "talk", "", rest.strip()
    for kind in ("INGEST ", "RESEARCH ", "WORK ", "DEPLOY "):
        if upper.startswith(kind):
            return kind.strip().lower(), line.split(" ", 1)[1].strip().strip('"'), ""
    action = parse_action(cleaned)
    if action and action.name in {"ingest", "research", "work"}:
        return action.name, action.path or action.query, ""
    return "talk", "", cleaned


def read_mentioned_files(text: str, settings: Settings, memory: Memory, tools: ToolRunner) -> list[str]:
    notes = []
    for path in find_docs(text):
        count = ingest_path(path, settings, memory, tools)
        if count:
            notes.append(f"Read {count} chunks from {path}.")
        else:
            notes.append(f"{path.name} was already in memory.")
    return notes


def perform_task(kind: str, arg: str, settings: Settings, memory: Memory, tools: ToolRunner, client: OllamaClient) -> str:
    if kind == "ingest":
        count = ingest_path(Path(arg.strip().strip('"')), settings, memory, tools)
        if count:
            return f"Read {count} chunks from {arg}."
        return f"{arg} was already in memory or had no text."
    if kind == "research":
        filed = research_once(settings, memory, tools, client, topic=arg)
        return f"Filed {filed} notes about {arg}."
    if kind == "work":
        project = settings.root if _owns_the_work(arg) else work_folder(arg, tools.project)
        runner = tools if project.resolve() == tools.project.resolve() else ToolRunner(
            settings, memory, tools.vectors, tools.embedder, project
        )
        passed = run_goal(settings, memory, runner, client, arg)
        if passed:
            return f"Finished: {arg}"
        return f"Could not finish cleanly: {arg}"
    if kind == "deploy":
        memory.add_record(text=f"Deploy requested: {arg}", kind="fact", title="deploy requested")
        return f"{arg} is waiting for approval in the control center."
    return ""


def _owns_the_work(goal: str) -> bool:
    lowered = goal.lower()
    return any(phrase in lowered for phrase in ("your code", "yourself", "your own", "valta itself"))


def route_message(text: str, personal_ready: bool) -> str:
    lowered = text.lower()
    if any(cue in lowered for cue in BUILD_CUES):
        return "teacher"
    if personal_ready and any(cue in lowered for cue in PERSONAL_CUES):
        return "personal"
    return "teacher"


def _format_memory(records) -> list[str]:
    blocks = []
    for record in records:
        source = f" ({record.url})" if record.url else ""
        title = record.title or record.kind
        blocks.append(f"[{record.kind}] {title}{source}\n{record.text}")
    return blocks


_FILE_CUES = (
    "create a file",
    "make a file",
    "write a file",
    "save a file",
    "edit the file",
    "edit a file",
    "change the file",
    "read the file",
    "list the files",
    "list files",
    "show the files",
    "on my desktop",
    "on the desktop",
    "to my desktop",
    "to the desktop",
)
_FILENAME = re.compile(
    r"(?:called|named)\s+[\"'`]?([A-Za-z0-9][\w .()-]*\.[A-Za-z0-9]{1,8})[\"'`]?",
    re.IGNORECASE,
)
_BARE_FILENAME = re.compile(
    r"\b([A-Za-z0-9][\w.-]*\.(?:txt|md|py|json|html|css|js|csv|toml|xml|yml|yaml))\b",
    re.IGNORECASE,
)


def asks_for_files(text: str) -> bool:
    lowered = text.lower()
    return any(cue in lowered for cue in _FILE_CUES)


def work_folder(text: str, project: Path, home: Path | None = None) -> Path:
    """The folder the user named, or the folder Valta was started from."""
    return _named_directory(text, home or Path.home()) or project


def write_requested_file(
    text: str,
    settings: Settings,
    memory: Memory,
    tools: ToolRunner,
    home: Path | None = None,
) -> str | None:
    """Write a file the user named. Returns None when the message is not that request."""
    if not re.search(r"\b(?:make|create|write|save|add)\b", text, re.IGNORECASE):
        return None
    if not re.search(r"\bfile\b", text, re.IGNORECASE) and not _BARE_FILENAME.search(text):
        return None
    found = _FILENAME.search(text) or _BARE_FILENAME.search(text)
    if not found:
        return None
    name = found.group(1).strip()
    relative = Path(name)
    if relative.is_absolute() or any(part in {".", ".."} for part in relative.parts):
        return None
    folder = work_folder(text, tools.project, home)
    if not folder.is_dir():
        return f"{folder} is not a folder I can write in."
    content = _requested_content(text)
    runner = tools
    if folder.resolve() != tools.project.resolve():
        runner = ToolRunner(settings, memory, tools.vectors, tools.embedder, folder)
    try:
        runner.write_file(name, content)
    except ToolError as exc:
        return str(exc)
    return f"Wrote {name} in {folder}."


def _named_directory(text: str, home: Path) -> Path | None:
    lowered = text.lower()
    desktop_phrases = ("on my desktop", "on the desktop", "to my desktop", "to the desktop", "in my desktop", "my desktop")
    if any(phrase in lowered for phrase in desktop_phrases):
        desktop = home / "Desktop"
        if desktop.is_dir():
            return desktop
    for raw in re.findall(r"[A-Za-z]:\\[^\r\n\"']+", text):
        path = Path(raw.strip().rstrip(".,)"))
        if path.suffix and path.parent.is_dir():
            return path.parent
        if path.is_dir():
            return path
    return None


def _requested_content(text: str) -> str:
    match = re.search(
        r"\b(?:that says|containing|with the text|with the contents|with content)\b\s+[\"']?(.*)[\"']?\s*$",
        text,
        re.IGNORECASE,
    )
    if not match:
        return ""
    return match.group(1).strip().strip("\"'")


def reply_once(
    settings: Settings,
    memory: Memory,
    tools: ToolRunner,
    client: OllamaClient,
    history: list,
    user_text: str,
    on_task=None,
) -> str:
    written = write_requested_file(user_text, settings, memory, tools)
    if written:
        tools.save_note(text=written, title=user_text[:200], kind="chat")
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": written})
        return written
    state = load_state(settings)
    personal_model = state.get("personal_model") or settings.self_model
    personal_ready = personal_model == "valta" and client.healthy(settings.self_host)
    route = route_message(user_text, personal_ready)
    if route == "personal":
        host, model, num_ctx = settings.self_host, personal_model, settings.self_num_ctx
        role = "You are Valta's personal model. Talk about the user from memory.\n" + CHAT_ROLE
    else:
        host, model, num_ctx = settings.heavy_host, settings.teacher_model, settings.teacher_num_ctx
        role = CHAT_ROLE
    role += (
        f"\nThe working folder is {tools.project}. "
        "File work belongs on a WORK line, aimed at that folder unless the user named another one."
    )
    if not client.healthy(host):
        raise OllamaError(
            f"Ollama is not running at {host}. Start both servers with scripts/start-ollama.ps1."
        )
    notes = read_mentioned_files(user_text, settings, memory, tools)
    if notes:
        user_text = user_text + "\n\n" + "\n".join(notes)
    records = tools.search_memory(user_text)
    system = build_system(settings, role=role, memory_blocks=_format_memory(records))
    messages = [{"role": "system", "content": system}, *history, {"role": "user", "content": user_text}]
    answer = client.chat(host, model, messages, num_ctx)
    kind, arg, spoken = interpret(answer)
    if kind in {"talk", "research"} and asks_for_files(user_text):
        kind, arg, spoken = "work", user_text.strip(), ""
    if kind == "talk":
        final = spoken or answer
    elif on_task is not None:
        on_task(kind, arg)
        final = {
            "work": f"Queued that with the crew: {arg}",
            "research": f"Researching {arg}. Watch the Live tab.",
            "ingest": f"Reading {arg}.",
            "deploy": f"{arg} is waiting for approval.",
        }.get(kind, arg)
    else:
        try:
            result = perform_task(kind, arg, settings, memory, tools, client)
        except (OSError, OllamaError, ValueError) as exc:
            result = str(exc)
        final = client.chat(
            host,
            model,
            [
                {"role": "system", "content": build_system(settings, role="Tell the user what you just did. Plain sentences. Do not emit a task line.")},
                {"role": "user", "content": result},
            ],
            num_ctx,
        )
        final = interpret(final)[2] or final
    tools.save_note(text=final, title=user_text[:200], kind="chat")
    _capture_goal(memory, user_text)
    _capture_correction(memory, history, user_text)
    _capture_uncertainty(memory, user_text, final)
    _capture_facts(settings, client, host, model, num_ctx, tools, user_text, final)
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": final})
    return final


def _capture_goal(memory: Memory, user_text: str) -> None:
    lowered = user_text.lower().strip()
    if lowered.startswith("my goal is "):
        memory.add_goal(user_text.split(" ", 3)[-1].strip())


def _capture_correction(memory: Memory, history: list, user_text: str) -> None:
    # Only an explicit "correction: <the right answer>" is a training target. A bare "no, that's wrong"
    # says nothing about what the right answer is, so it must not become one.
    lowered = user_text.lower().strip()
    if not lowered.startswith("correction:"):
        return
    prior_user = ""
    for message in reversed(history):
        if message["role"] == "user":
            prior_user = message["content"]
            break
    desired = user_text.split(":", 1)[-1].strip()
    if prior_user and desired:
        memory.add_example(prior_user, desired)


def _capture_uncertainty(memory: Memory, user_text: str, answer: str) -> None:
    lowered = answer.lower()
    if "i don't know" in lowered or "i do not know" in lowered or "i'm not sure" in lowered:
        memory.enqueue_topic(user_text[:240], source="unanswered")


def _capture_facts(settings, client, host, model, num_ctx, tools: ToolRunner, user_text: str, answer: str) -> None:
    system = build_system(settings, role="Extract durable facts about the user. Do not invent any.")
    prompt = (
        "List up to 3 durable facts about the user from this exchange. "
        "One fact per line. If there are none, reply NONE.\n\n"
        f"User: {user_text}\nValta: {answer}"
    )
    try:
        raw = client.chat(
            host,
            model,
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            num_ctx,
        )
    except OllamaError:
        return
    if raw.strip().upper() == "NONE":
        return
    for line in raw.splitlines():
        fact = line.lstrip("- ").strip()
        if fact and fact.upper() != "NONE":
            tools.save_note(text=fact, title="fact", kind="fact")


def chat_loop(settings: Settings, memory: Memory, tools: ToolRunner, client: OllamaClient) -> None:
    memory.ensure_default_agent()
    memory.ensure_project(tools.project)
    history: list[dict] = []
    stop = threading.Event()
    worker = threading.Thread(
        target=research_loop,
        args=(settings, memory, tools, client, stop),
        kwargs={"delay_first": True},
        daemon=True,
    )
    worker.start()
    print("Valta. Talk normally. Mention a file path and it will be read. /exit leaves.")
    try:
        _chat_turns(settings, memory, tools, client, history)
    finally:
        stop.set()


def _chat_turns(settings, memory, tools, client, history: list[dict]) -> None:
    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            return
        if user_text.startswith("/read "):
            try:
                print(tools.read_file(user_text[6:].strip()))
            except Exception as exc:
                print(exc)
            continue
        if user_text.startswith("/ls"):
            target = user_text[3:].strip() or "."
            try:
                for name in tools.list_directory(target):
                    print(name)
            except Exception as exc:
                print(exc)
            continue
        try:
            answer = reply_once(settings, memory, tools, client, history, user_text)
        except (OllamaError, OSError) as exc:
            print(exc)
            continue
        print(f"Valta: {answer}")


def live(settings: Settings, memory: Memory, tools: ToolRunner, client: OllamaClient) -> None:
    chat_loop(settings, memory, tools, client)
