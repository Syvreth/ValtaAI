from __future__ import annotations

import queue
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from valta.config import Settings, load_state
from valta.engineer import parse_action, run_goal
from valta.ingest import ingest_path
from valta.memory import Memory
from valta.ollama_client import OllamaClient, OllamaError
from valta.prompt import build_system
from valta.research import research_loop, research_once
from valta.tools import ToolRunner

CHAT_ROLE = """
You are Valta, in one ongoing chat. The user should not have to run commands.
The first line of every reply is exactly one of:
TALK
INGEST <path>
RESEARCH <topic>
WORK <goal>
Use TALK when they are conversing. Use INGEST when they want a file or folder read into memory. Use RESEARCH when they want you to look something up. Use WORK when they want software created or changed, or when they ask you to create, edit, read, or list files. If the first line is TALK, the rest of the reply is what you say. If it is a task, the first line is the whole reply.
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
    for kind in ("INGEST ", "RESEARCH ", "WORK "):
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


def asks_for_files(text: str) -> bool:
    lowered = text.lower()
    return any(cue in lowered for cue in _FILE_CUES)


def work_folder(text: str, project: Path, home: Path | None = None) -> Path:
    """The folder the user named, or the folder Valta was started from."""
    return _named_directory(text, home or Path.home()) or project


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


def reply_once(
    settings: Settings,
    memory: Memory,
    tools: ToolRunner,
    client: OllamaClient,
    history: list,
    user_text: str,
    on_task=None,
) -> str:
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
    _queue_facts(_Exchange(settings, client, host, model, num_ctx, tools, user_text, final))
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
    prior_answer = ""
    for message in reversed(history):
        if message["role"] == "assistant" and not prior_user and not prior_answer:
            prior_answer = message["content"]
        elif message["role"] == "user":
            prior_user = message["content"]
            break
    desired = user_text.split(":", 1)[-1].strip()
    if prior_user and desired:
        memory.add_example(prior_user, desired)
        if prior_answer:
            # The reply being corrected is the worse answer to the same question.
            memory.add_preference(prior_user, desired, prior_answer, source="correction")


def _capture_uncertainty(memory: Memory, user_text: str, answer: str) -> None:
    lowered = answer.lower()
    if "i don't know" in lowered or "i do not know" in lowered or "i'm not sure" in lowered:
        memory.enqueue_topic(user_text[:240], source="unanswered")


FACTS_PER_EXCHANGE = 3


@dataclass
class _Exchange:
    settings: Settings
    client: OllamaClient
    host: str
    model: str
    num_ctx: int
    tools: ToolRunner
    user_text: str
    answer: str


_fact_queue: queue.Queue = queue.Queue()
_fact_worker: threading.Thread | None = None
_fact_worker_lock = threading.Lock()


def _queue_facts(exchange: _Exchange) -> None:
    """Fact extraction is a whole model call, so it runs on one background thread instead of holding up the reply."""
    global _fact_worker
    _fact_queue.put(exchange)
    with _fact_worker_lock:
        if _fact_worker is None or not _fact_worker.is_alive():
            _fact_worker = threading.Thread(target=_drain_facts, daemon=True)
            _fact_worker.start()


def _drain_facts() -> None:
    """Exchanges that queued up while the model was busy are read together, in one call per model."""
    while True:
        batch = [_fact_queue.get()]
        while True:
            try:
                batch.append(_fact_queue.get_nowait())
            except queue.Empty:
                break
        groups: dict[tuple, list[_Exchange]] = {}
        for exchange in batch:
            groups.setdefault((id(exchange.tools), exchange.host, exchange.model), []).append(exchange)
        for group in groups.values():
            try:
                _capture_facts(group)
            except Exception as exc:  # one failed extraction must not stop the ones queued after it
                print(f"[memory] could not extract facts: {exc}")
        for _ in batch:
            _fact_queue.task_done()


def wait_for_facts(timeout: float | None = None) -> bool:
    """Block until every queued exchange has been read for facts, or `timeout` seconds pass. True when none is left."""
    with _fact_queue.all_tasks_done:
        return _fact_queue.all_tasks_done.wait_for(lambda: not _fact_queue.unfinished_tasks, timeout)


def save_pending_facts(timeout: float = 120) -> None:
    """Call on the way out. The fact thread dies with the program, so let it finish what is queued first."""
    if not _fact_queue.unfinished_tasks:
        return
    print("Saving what Valta learned from the last messages...", flush=True)
    if not wait_for_facts(timeout):
        print(f"Stopped waiting after {timeout:.0f} seconds. Facts from the last messages were not saved.")


def _capture_facts(exchanges: list[_Exchange]) -> None:
    first = exchanges[0]
    limit = FACTS_PER_EXCHANGE * len(exchanges)
    system = build_system(first.settings, role="Extract durable facts about the user. Do not invent any.")
    source = "this exchange" if len(exchanges) == 1 else "these exchanges"
    prompt = (
        f"List up to {limit} durable facts about the user from {source}. "
        "One fact per line. If there are none, reply NONE.\n\n"
        + "\n\n".join(f"User: {exchange.user_text}\nValta: {exchange.answer}" for exchange in exchanges)
    )
    try:
        raw = first.client.chat(
            first.host,
            first.model,
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            first.num_ctx,
        )
    except OllamaError:
        return
    if raw.strip().upper() == "NONE":
        return
    facts = [line.lstrip("- ").strip() for line in raw.splitlines()]
    for fact in [fact for fact in facts if fact and fact.upper() != "NONE"][:limit]:
        first.tools.save_note(text=fact, title="fact", kind="fact")


def chat_loop(settings: Settings, memory: Memory, tools: ToolRunner, client: OllamaClient) -> None:
    memory.ensure_default_agent()
    memory.ensure_project(tools.project)
    history: list[dict] = []
    stop = threading.Event()
    if settings.research_interval_sec > 0:
        threading.Thread(
            target=research_loop,
            args=(settings, memory, tools, client, stop),
            kwargs={"delay_first": True},
            daemon=True,
        ).start()
    print("Valta. Talk normally. Mention a file path and it will be read. /exit leaves.")
    try:
        _chat_turns(settings, memory, tools, client, history)
    finally:
        stop.set()
        save_pending_facts()


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
