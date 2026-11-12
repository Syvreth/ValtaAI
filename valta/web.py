"""Learn from a documentation website: `valta learn <url>`.

Reads pages under the address you give, the way a careful person would: it follows robots.txt (including
any Crawl-delay), waits between requests, stays on the same site and under the same path, reads only
HTML, and stops at a page limit. It never logs in and never sends cookies. Each page's main text is split
into chunks and stored in memory as kind "web" with its URL, so it is used at once when Valta answers
(memory retrieval), and later as source passages for training questions.

What you point it at is your call. Prefer documentation whose license allows reuse, and stay within a
site's terms of use.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Callable
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx

from valta.config import Settings
from valta.ingest import chunk_text, html_to_text
from valta.memory import Memory

USER_AGENT = "Valta/0.1 (personal documentation reader; respects robots.txt)"
MAX_PAGES_LIMIT = 500
MAX_PAGE_BYTES = 2_000_000
MIN_PAGE_CHARS = 200
SKIPPED_EXTENSIONS = {
    ".pdf", ".zip", ".gz", ".tar", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".css", ".js",
    ".json", ".xml", ".mp4", ".mp3", ".woff", ".woff2", ".ttf", ".exe", ".whl", ".txt",
}

# fetch(url) -> (status_code, content_type, body_text)
Fetch = Callable[[str], tuple[int, str, str]]


@dataclass
class LearnStats:
    pages: int = 0
    chunks: int = 0
    skipped: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def summary(self) -> str:
        reasons = ", ".join(f"{reason} {count}" for reason, count in sorted(self.skipped.items())) or "none"
        return f"Learned {self.pages} page(s) as {self.chunks} chunk(s). Skipped: {reasons}."


class _Page(HTMLParser):
    """Links, the <title>, and the text inside <main> or <article> when the page has one."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[str] = []
        self.title = ""
        self._in_title = False
        self._main_depth = 0
        self._main_html: list[str] = []
        self.has_main = False

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)
        elif tag == "title":
            self._in_title = True
        if tag in {"main", "article"}:
            self._main_depth += 1
            self.has_main = True
        if self._main_depth:
            self._main_html.append(self.get_starttag_text() or "")

    def handle_endtag(self, tag):
        if self._main_depth:
            self._main_html.append(f"</{tag}>")
        if tag == "title":
            self._in_title = False
        if tag in {"main", "article"} and self._main_depth:
            self._main_depth -= 1

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if self._main_depth:
            self._main_html.append(data.replace("<", "&lt;").replace(">", "&gt;"))

    def main_text(self, raw: str) -> str:
        if self.has_main:
            text = html_to_text("".join(self._main_html))
            if len(text) >= MIN_PAGE_CHARS:
                return text
        return html_to_text(raw)


def normalize(url: str) -> str:
    """Drop the #fragment and the ?query, so one page is not read twice under different addresses."""
    url, _fragment = urldefrag(url)
    parsed = urlparse(url)
    return parsed._replace(query="", params="").geturl()


def scope_of(start_url: str) -> tuple[str, str, str]:
    """(scheme, host, path prefix). A start of /3/library/os.html reads under /3/library/."""
    parsed = urlparse(start_url)
    path = parsed.path or "/"
    prefix = path if path.endswith("/") else path.rsplit("/", 1)[0] + "/"
    return parsed.scheme, parsed.netloc.lower(), prefix


def in_scope(url: str, scope: tuple[str, str, str]) -> bool:
    parsed = urlparse(url)
    scheme, host, prefix = scope
    if parsed.scheme not in {"http", "https"} or parsed.scheme != scheme or parsed.netloc.lower() != host:
        return False
    path = parsed.path or "/"
    if not (path.startswith(prefix) or path + "/" == prefix):
        return False
    lowered = path.lower()
    return not any(lowered.endswith(extension) for extension in SKIPPED_EXTENSIONS)


def http_fetch(url: str) -> tuple[int, str, str]:
    with httpx.stream(
        "GET", url, timeout=20, follow_redirects=True, headers={"User-Agent": USER_AGENT}
    ) as response:
        content_type = response.headers.get("content-type", "")
        if response.status_code >= 400 or "html" not in content_type.lower():
            return response.status_code, content_type, ""
        body = b""
        for block in response.iter_bytes():
            body += block
            if len(body) > MAX_PAGE_BYTES:
                return 413, content_type, ""
        return response.status_code, content_type, body.decode(response.encoding or "utf-8", errors="replace")


def load_robots(start_url: str, fetch_text: Callable[[str], tuple[int, str]] | None = None) -> RobotFileParser:
    """robots.txt for the site. Missing (404) allows everything; forbidden (401/403) allows nothing."""
    scheme, host, _prefix = scope_of(start_url)
    robots_url = f"{scheme}://{host}/robots.txt"
    parser = RobotFileParser(robots_url)
    if fetch_text is None:
        def fetch_text(url: str) -> tuple[int, str]:
            response = httpx.get(url, timeout=20, follow_redirects=True, headers={"User-Agent": USER_AGENT})
            return response.status_code, response.text

    try:
        status, text = fetch_text(robots_url)
    except httpx.HTTPError:
        status, text = 404, ""
    if status in (401, 403):
        parser.disallow_all = True
    elif status >= 400:
        parser.allow_all = True
    else:
        parser.parse(text.splitlines())
    return parser


def learn_site(
    settings: Settings,
    memory: Memory,
    tools,
    start_url: str,
    max_pages: int = 30,
    delay: float = 1.0,
    fetch: Fetch | None = None,
    robots: RobotFileParser | None = None,
    sleep: Callable[[float], None] = time.sleep,
    on_page: Callable[[str, int], None] | None = None,
) -> LearnStats:
    """Read up to `max_pages` pages under `start_url` and store their text in memory."""
    if urlparse(start_url).scheme not in {"http", "https"}:
        raise ValueError("Give a full http:// or https:// address.")
    max_pages = max(1, min(int(max_pages), MAX_PAGES_LIMIT))
    fetch = fetch or http_fetch
    robots = robots or load_robots(start_url)
    crawl_delay = robots.crawl_delay(USER_AGENT)
    wait = max(float(delay), float(crawl_delay or 0))
    scope = scope_of(start_url)
    start = normalize(start_url)
    queue = [start]
    queued = {start}
    stats = LearnStats()
    first = True
    while queue and stats.pages < max_pages:
        url = queue.pop(0)
        if not robots.can_fetch(USER_AGENT, url):
            stats.skip("robots.txt")
            continue
        if not first:
            sleep(wait)
        first = False
        try:
            status, _content_type, body = fetch(url)
        except httpx.HTTPError:
            stats.skip("network error")
            continue
        if status >= 400 or not body:
            stats.skip("not an HTML page" if status < 400 else f"HTTP {status}")
            continue
        page = _Page()
        try:
            page.feed(body)
        except Exception:
            stats.skip("unreadable HTML")
            continue
        for href in page.links:
            link = normalize(urljoin(url, href))
            if link not in queued and in_scope(link, scope):
                queued.add(link)
                queue.append(link)
        if memory.source_exists(url):
            stats.skip("already learned")
            continue
        text = page.main_text(body).replace("¶", "")  # ¶ permalink markers on documentation headings
        if len(text) < MIN_PAGE_CHARS:
            stats.skip("too little text")
            continue
        title = " ".join(page.title.split()) or url
        chunks = chunk_text(text, settings.chunk_chars, settings.chunk_overlap)
        for index, chunk in enumerate(chunks, start=1):
            tools.save_note(text=chunk, title=f"{title} #{index}", kind="web", url=url, source=url)
        stats.pages += 1
        stats.chunks += len(chunks)
        if on_page is not None:
            on_page(url, len(chunks))
    return stats
