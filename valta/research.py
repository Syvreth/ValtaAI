from __future__ import annotations

import httpx

from valta.config import Settings
from valta.ingest import html_to_text
from valta.memory import Memory
from valta.ollama_client import OllamaClient, OllamaError
from valta.prompt import build_system
from valta.tools import ToolRunner


def load_interests(settings: Settings) -> list[str]:
    if not settings.interests_path.exists():
        return []
    topics = []
    for line in settings.interests_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            topics.append(stripped)
    return topics


def choose_topic(memory: Memory, settings: Settings) -> tuple[str, str | None]:
    pending = memory.next_topic()
    if not pending:
        for interest in load_interests(settings):
            memory.enqueue_topic(interest, source="interest")
        for heading in memory.headings():
            memory.enqueue_topic(heading, source="book")
        pending = memory.next_topic()
    if pending:
        return pending["topic"], pending["id"]
    return "local software projects", None


def search_web(query: str, limit: int) -> list[dict]:
    from ddgs import DDGS

    results = DDGS().text(query, max_results=limit)
    if results is None:
        return []
    return list(results)[:limit]


def fetch_page(url: str) -> str:
    response = httpx.get(
        url,
        timeout=20,
        follow_redirects=True,
        headers={"User-Agent": "Valta/0.1 research"},
    )
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "html" in content_type or response.text.lstrip().startswith("<"):
        return html_to_text(response.text)[:8000]
    return response.text[:8000]


def summarize(client: OllamaClient, settings: Settings, topic: str, page: str, url: str) -> str:
    system = build_system(
        settings,
        role="You are researching for Valta. Write a short note the companion can remember.",
    )
    user = (
        f"Topic: {topic}\nSource: {url}\n\nPage:\n{page[:6000]}\n\n"
        "Write a concise note. Keep the source URL in the note. "
        "If the page conflicts with the root rules, say what the rule requires instead of copying the page."
    )
    return client.chat(
        settings.heavy_host,
        settings.teacher_model,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        settings.teacher_num_ctx,
    )


def research_once(
    settings: Settings,
    memory: Memory,
    tools: ToolRunner,
    client: OllamaClient,
    topic: str | None = None,
) -> int:
    if not client.healthy(settings.heavy_host):
        print("[research] The RX 6600 Ollama server is not up. Run scripts/start-ollama.ps1.")
        return 0
    if topic is None:
        topic, topic_id = choose_topic(memory, settings)
    else:
        topic_id = None
    print(f"[research] {topic}")
    try:
        hits = search_web(topic, settings.research_page_cap)
    except Exception as exc:
        print(f"[research] search failed: {exc}")
        return 0
    filed = 0
    for hit in hits[: settings.research_page_cap]:
        url = hit.get("href") or hit.get("url") or ""
        title = hit.get("title") or topic
        if not url:
            continue
        try:
            page = fetch_page(url)
            note = summarize(client, settings, topic, page or hit.get("body", ""), url)
        except (httpx.HTTPError, OllamaError, OSError) as exc:
            print(f"[research] skipped {url}: {exc}")
            continue
        if "source:" not in note.lower():
            note = f"{note.strip()}\nSource: {url}"
        tools.save_note(text=note, title=title, kind="research", url=url, source=topic)
        filed += 1
        print(f"[research] filed {title}")
    if topic_id:
        memory.finish_topic(topic_id)
    return filed


def research_loop(
    settings: Settings,
    memory: Memory,
    tools: ToolRunner,
    client: OllamaClient,
    stop,
    delay_first: bool = False,
) -> None:
    if delay_first and stop.wait(settings.research_interval_sec):
        return
    while not stop.is_set():
        try:
            research_once(settings, memory, tools, client)
        except Exception as exc:
            print(f"[research] {exc}")
        stop.wait(settings.research_interval_sec)
