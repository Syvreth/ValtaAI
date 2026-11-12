"""A compact map of a project for the planner and the coder.

Every file is listed. Python files also show their line count and their classes and functions, read
with the standard-library parser, so the model can name real paths and see which files are short.
Test files show how many tests they hold instead of every test name. The map is capped, so a large
project cannot crowd out the goal and the plan: argument lists go first, then outlines.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path

MAP_MAX_CHARS = 8000
SKIP_DIRS = {"data", "__pycache__", "node_modules", "venv", "env", "dist", "build", "target", "chroma"}
PARSE_MAX_BYTES = 1_000_000
_CUT_NOTE = "(classes and functions left out for the rest, to keep the map short)"


@dataclass
class _Entry:
    head: str
    items: list[tuple[int, str, str, str]] = field(default_factory=list)  # depth, "class" or "def", name, args

    def lines(self, with_args: bool) -> list[str]:
        out = [self.head]
        for depth, keyword, name, args in self.items:
            out.append("  " * depth + f"{keyword} {name}" + (f"({args})" if keyword == "def" and with_args else ""))
        return out


def repo_map(root: Path, max_chars: int = MAP_MAX_CHARS) -> str:
    entries = [_entry(path, path.relative_to(root).as_posix()) for path in _files(root)]
    if not entries:
        return "(no files yet)"
    budget = max_chars - len(_CUT_NOTE) - 1
    heads = [entry.head for entry in entries]
    if _size(heads) > budget:
        return _heads_only(heads, budget)
    for with_args in (True, False):
        lines = [line for entry in entries for line in entry.lines(with_args)]
        if _size(lines) <= budget:
            return "\n".join(lines)
    lines = []
    remaining = _size(heads)
    outlines = True
    for entry in entries:
        remaining -= len(entry.head) + 1
        block = entry.lines(with_args=False)
        if outlines and len(block) > 1 and _size(lines) + _size(block) + remaining > budget:
            outlines = False
        lines.extend(block if outlines else [entry.head])
    if not outlines:
        lines.append(_CUT_NOTE)
    return "\n".join(lines)


def _size(lines: list[str]) -> int:
    return sum(len(line) + 1 for line in lines)


def _files(root: Path):
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(
            name for name in dirs if not name.startswith(".") and name not in SKIP_DIRS and not name.endswith(".egg-info")
        )
        for name in sorted(files):
            path = Path(current) / name
            if not path.is_symlink() and not name.endswith((".pyc", ".pyo")):
                yield path


def _heads_only(heads: list[str], budget: int) -> str:
    kept: list[str] = []
    for head in heads:
        if _size(kept) + len(head) + 1 > budget:
            break
        kept.append(head)
    return "\n".join([*kept, f"... and {len(heads) - len(kept)} more files"])


def _entry(path: Path, name: str) -> _Entry:
    if path.suffix != ".py":
        return _Entry(name)
    try:
        if path.stat().st_size > PARSE_MAX_BYTES:
            return _Entry(name)
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return _Entry(name)
    length = f"{len(source.splitlines())} lines"
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return _Entry(f"{name} ({length}, does not parse)")
    if Path(name).name.startswith("test_"):
        tests = sum(
            1
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test")
        )
        return _Entry(f"{name} ({length}, {tests} test{'' if tests == 1 else 's'})")
    entry = _Entry(f"{name} ({length})")
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            entry.items.append((1, "class", node.name, ""))
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    entry.items.append((2, "def", item.name, _args(item, skip_self=True)))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            entry.items.append((1, "def", node.name, _args(node)))
    return entry


def _args(node: ast.FunctionDef | ast.AsyncFunctionDef, skip_self: bool = False) -> str:
    args = node.args
    names = [arg.arg for arg in (*args.posonlyargs, *args.args)]
    if skip_self and names and names[0] in {"self", "cls"}:
        names = names[1:]
    if args.vararg:
        names.append("*" + args.vararg.arg)
    names += [arg.arg for arg in args.kwonlyargs]
    if args.kwarg:
        names.append("**" + args.kwarg.arg)
    text = ", ".join(names)
    return text if len(text) <= 60 else text[:57] + "..."
