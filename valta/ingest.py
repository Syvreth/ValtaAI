from __future__ import annotations

import shutil
from html.parser import HTMLParser
from pathlib import Path

from valta.config import Settings
from valta.memory import Memory
from valta.tools import ToolRunner


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self._skip:
            self._skip -= 1
        if tag in {"p", "div", "br", "h1", "h2", "h3", "li"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        return " ".join("".join(self.parts).split())


def html_to_text(raw: str) -> str:
    parser = _HTMLText()
    parser.feed(raw)
    return parser.text()


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    cleaned = text.replace("\r\n", "\n").strip()
    if not cleaned:
        return []
    if len(cleaned) <= size:
        return [cleaned]
    chunks = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + size)
        if end < len(cleaned):
            window = start + max(1, size // 2)
            split = cleaned.rfind("\n\n", window, end)
            if split <= start:
                split = cleaned.rfind(" ", window, end)
            if split > start:
                end = split
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(cleaned):
            break
        next_start = end - overlap
        start = next_start if next_start > start else end
    return chunks


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if suffix == ".epub":
        from ebooklib import ITEM_DOCUMENT, epub

        book = epub.read_epub(str(path))
        parts = []
        for item in book.get_items_of_type(ITEM_DOCUMENT):
            raw = item.get_content().decode("utf-8", errors="ignore")
            parts.append(html_to_text(raw))
        return "\n\n".join(part for part in parts if part)
    if suffix in {".txt", ".md", ".markdown"}:
        return path.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"Unsupported file type: {path.suffix}")


def headings(text: str) -> list[str]:
    found = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                found.append(title)
    return found


def ingest_path(path: Path, settings: Settings, memory: Memory, tools: ToolRunner) -> int:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    files = [path] if path.is_file() else [
        item for item in path.rglob("*") if item.suffix.lower() in {".pdf", ".epub", ".txt", ".md", ".markdown"}
    ]
    stored = 0
    for file_path in files:
        source = str(file_path)
        if memory.source_exists(source):
            print(f"Already ingested: {file_path.name}")
            continue
        destination = file_path
        if settings.library.resolve() not in file_path.parents and file_path.parent != settings.library.resolve():
            destination = settings.library / file_path.name
            if not destination.exists():
                shutil.copy2(file_path, destination)
            source = str(destination.resolve())
            if memory.source_exists(source):
                print(f"Already ingested: {destination.name}")
                continue
        text = read_document(destination)
        for heading in headings(text):
            memory.enqueue_topic(heading, source=source)
        chunks = chunk_text(text, settings.chunk_chars, settings.chunk_overlap)
        for index, chunk in enumerate(chunks, start=1):
            title = f"{destination.stem} #{index}"
            tools.save_note(text=chunk, title=title, kind="book", source=source)
            stored += 1
        print(f"Ingested {destination.name}: {len(chunks)} chunks")
    return stored
