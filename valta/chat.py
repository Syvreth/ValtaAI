from __future__ import annotations

import threading

from valta.config import Settings, load_state
from valta.memory import Memory
from valta.ollama_client import OllamaClient, OllamaError
from valta.prompt import build_system
from valta.research import research_loop
from valta.tools import ToolRunner

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
        blocks.append(f"[{record.kind}] {title}{source}\n{record.text[:1200]}")
    return blocks


def reply_once(settings: Settings, memory: Memory, tools: ToolRunner, client: OllamaClient, history: list, user_text: str) -> str:
    state = load_state(settings)
    personal_model = state.get("personal_model") or settings.self_model
    personal_ready = personal_model == "valta" and client.healthy(settings.self_host)
    route = route_message(user_text, personal_ready)
    if route == "personal":
        host, model, num_ctx = settings.self_host, personal_model, settings.self_num_ctx
        role = "You are Valta's personal model. Talk about the user from memory."
    else:
        host, model, num_ctx = settings.heavy_host, settings.teacher_model, settings.teacher_num_ctx
        role = "You are Valta. Help with research and with building software."
    if not client.healthy(host):
        raise OllamaError(
            f"Ollama is not running at {host}. Start both servers with scripts/start-ollama.ps1."
        )
    records = tools.search_memory(user_text)
    system = build_system(settings, role=role, memory_blocks=_format_memory(records))
    messages = [{"role": "system", "content": system}, *history[-8:], {"role": "user", "content": user_text}]
    answer = client.chat(host, model, messages, num_ctx)
    tools.save_note(text=answer, title=user_text[:200], kind="chat")
    _capture_goal(memory, user_text)
    _capture_correction(memory, history, user_text)
    _capture_uncertainty(memory, user_text, answer)
    _capture_facts(settings, client, host, model, num_ctx, tools, user_text, answer)
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": answer})
    return answer


def _capture_goal(memory: Memory, user_text: str) -> None:
    lowered = user_text.lower().strip()
    if lowered.startswith("my goal is "):
        memory.add_goal(user_text.split(" ", 3)[-1].strip())


def _capture_correction(memory: Memory, history: list, user_text: str) -> None:
    lowered = user_text.lower().strip()
    if not (lowered.startswith("correction:") or lowered.startswith("no,")):
        return
    prior_user = ""
    for message in reversed(history):
        if message["role"] == "user":
            prior_user = message["content"]
            break
    desired = user_text.split(":", 1)[-1].strip() if lowered.startswith("correction:") else user_text
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
    print("Valta. /exit to leave, /read <path>, /ls <path>.")
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
    stop = threading.Event()
    worker = threading.Thread(
        target=research_loop,
        args=(settings, memory, tools, client, stop),
        daemon=True,
    )
    worker.start()
    try:
        chat_loop(settings, memory, tools, client)
    finally:
        stop.set()
