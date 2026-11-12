from __future__ import annotations

import re
import shlex
import subprocess
import sys
import threading
from pathlib import Path

from valta.config import Settings
from valta.memory import Memory, Record
from valta.vectors import Vectors


class ToolError(RuntimeError):
    pass


class ToolRunner:
    """Tools shared by chat, research, and later agents. The root rules file cannot be changed."""

    def __init__(
        self,
        settings: Settings,
        memory: Memory,
        vectors: Vectors,
        embedder,
        project: Path,
    ):
        self.settings = settings
        self.memory = memory
        self.vectors = vectors
        self.embedder = embedder
        self.project = project.resolve()
        self.rules_path = settings.rules_path.resolve()
        self._lock = threading.Lock()

    def call(self, name: str, **kwargs):
        tools = {
            "search_memory": self.search_memory,
            "save_note": self.save_note,
            "read_file": self.read_file,
            "list_directory": self.list_directory,
            "write_file": self.write_file,
            "replace_text": self.replace_text,
            "run_tests": self.run_tests,
        }
        if name not in tools:
            raise ToolError(f"Unknown tool: {name}")
        return tools[name](**kwargs)

    def search_memory(self, query: str, kind: str | None = None, k: int | None = None) -> list[Record]:
        with self._lock:
            return self._search_memory(query, kind, k)

    def _search_memory(self, query: str, kind: str | None, k: int | None) -> list[Record]:
        limit = k or self.settings.top_k
        try:
            embedding = self.embedder(query)
            ids = self.vectors.query(embedding, limit, kind=kind)
            records = self.memory.get_many(ids)
            if records:
                return records
        except Exception:
            pass
        if kind:
            return [record for record in self.memory.recent(limit * 3, kind=kind)][:limit]
        return self.memory.search_text(query, limit) or self.memory.recent(limit)

    def save_note(
        self,
        text: str,
        title: str = "",
        kind: str = "research",
        url: str | None = None,
        source: str | None = None,
    ) -> Record:
        with self._lock:
            return self._save_note(text, title, kind, url, source)

    def _save_note(self, text: str, title: str, kind: str, url: str | None, source: str | None) -> Record:
        record = self.memory.add_record(text=text, kind=kind, title=title, url=url, source=source)
        try:
            embedding = self.embedder(text)
            self.vectors.upsert(record.id, embedding, kind, text)
        except Exception as exc:
            print(f"[memory] stored '{title or kind}' without a vector: {exc}")
        return record

    def read_file(self, path: str) -> str:
        resolved = self._inside_project(Path(path))
        if not resolved.is_file():
            raise ToolError(f"Not a file: {resolved}")
        return resolved.read_text(encoding="utf-8", errors="replace")

    def list_directory(self, path: str = ".") -> list[str]:
        resolved = self._inside_project(Path(path))
        if not resolved.is_dir():
            raise ToolError(f"Not a directory: {resolved}")
        return sorted(item.name for item in resolved.iterdir())

    def write_file(self, path: str, content: str) -> str:
        resolved = self._inside_project(Path(path))
        self.refuse_rules_mutation(resolved)
        if self._protected(resolved):
            raise ToolError("That path is protected. Root rules, memory, and the virtualenv stay as they are.")
        if resolved.exists() and resolved.is_dir():
            raise ToolError(f"Not a file: {resolved}")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return f"Wrote {resolved.relative_to(self.project)}"

    def replace_text(self, path: str, old: str, new: str) -> str:
        """Swap one block of an existing file. `old` must appear exactly once. Only that block changes: a line
        of `old` matches whether the file ends it with \\n or \\r\\n, and the new lines take the block's ending."""
        resolved = self._inside_project(Path(path))
        self.refuse_rules_mutation(resolved)
        if self._protected(resolved):
            raise ToolError("That path is protected. Root rules, memory, and the virtualenv stay as they are.")
        if not resolved.is_file():
            raise ToolError(f"Not a file: {path}. Use write_file to create it.")
        old, new = old.replace("\r\n", "\n"), new.replace("\r\n", "\n")
        if not old.strip():
            raise ToolError("OLD is empty. Copy the exact lines to change from the file.")
        try:
            text = resolved.read_bytes().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(f"{path} is not UTF-8 text, so it cannot be edited in place.") from exc
        pattern = re.compile(r"\r?\n".join(re.escape(line) for line in old.split("\n")))
        matches = list(pattern.finditer(text))
        if not matches:
            raise ToolError(f"OLD was not found in {path}. Copy it exactly from read_file, indentation included.")
        if len(matches) > 1:
            raise ToolError(
                f"OLD appears {len(matches)} times in {path}. Include more of the lines around it so it matches once."
            )
        match = matches[0]
        crlf = text.count("\r\n")
        if "\n" in match.group(0):
            newline = "\r\n" if "\r\n" in match.group(0) else "\n"
        else:  # a one-line block: take the file's usual ending
            newline = "\r\n" if crlf > text.count("\n") - crlf else "\n"
        text = text[: match.start()] + new.replace("\n", newline) + text[match.end() :]
        resolved.write_bytes(text.encode("utf-8"))
        return f"Replaced one block in {resolved.relative_to(self.project)}"

    def run_tests(self, command: str = "python -m pytest -q") -> str:
        argv = self._allowed_test_command(command)
        try:
            completed = subprocess.run(
                argv,
                cwd=self.project,
                capture_output=True,
                text=True,
                timeout=180,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return "exit timeout\nThe test command ran longer than 180 seconds."
        output = (completed.stdout + "\n" + completed.stderr).strip()
        if len(output) > 4000:
            output = output[-4000:]
        return f"exit {completed.returncode}\n{output}"

    def forget(self, record_id: str) -> bool:
        with self._lock:
            return self._forget(record_id)

    def _forget(self, record_id: str) -> bool:
        removed = self.memory.delete_record(record_id)
        try:
            self.vectors.delete(record_id)
        except Exception:
            pass
        return removed

    def refuse_rules_mutation(self, path: Path) -> None:
        if path.resolve() == self.rules_path:
            raise ToolError("The root rules cannot be edited or deleted by an agent.")

    def _protected(self, path: Path) -> bool:
        resolved = path.resolve()
        if resolved == self.rules_path:
            return True
        try:
            parts = resolved.relative_to(self.project).parts
        except ValueError:
            return True
        if not parts:
            return True
        if parts[0] in {".venv", ".venv-eval", ".git"}:
            return True
        if self.project.resolve() == self.settings.root.resolve() and parts[0] in {"rules", "data"}:
            return True
        return False

    def _allowed_test_command(self, command: str) -> list[str]:
        if any(token in command for token in ("|", "&", ";", ">", "<", "`")):
            raise ToolError("That command needs approval. Tests run without a shell.")
        tokens = shlex.split(command, posix=(sys.platform != "win32"))
        if len(tokens) < 3 or tokens[0] != "python" or tokens[1] != "-m":
            raise ToolError("That command needs approval. Allowed: python -m pytest, python -m compileall.")
        if tokens[2] not in {"pytest", "compileall"}:
            raise ToolError("That command needs approval. Allowed: python -m pytest, python -m compileall.")
        return [sys.executable, *tokens[1:]]

    def _refuse_symlink(self, path: Path) -> None:
        candidate = path if path.is_absolute() else (self.project / path)
        cursor = candidate
        while True:
            if cursor.is_symlink():
                raise ToolError("That path is a symbolic link. Valta will not follow it.")
            if cursor == self.project or self.project not in cursor.parents:
                break
            parent = cursor.parent
            if parent == cursor:
                break
            cursor = parent

    def _inside_project(self, path: Path) -> Path:
        self._refuse_symlink(path)
        resolved = path.resolve() if path.is_absolute() else (self.project / path).resolve()
        if resolved != self.project and self.project not in resolved.parents:
            raise ToolError("That path is outside the project folder.")
        return resolved
