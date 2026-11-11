from __future__ import annotations

import difflib
import os
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from valta.chat import _owns_the_work
from valta.control import JobControl, JobStopped
from valta.engineer import run_goal

SKIP_DIRS = {
    ".venv",
    "venv",
    ".git",
    "data",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "chroma",
    "node_modules",
    ".next",
    "dist",
    "build",
    "target",
}
SKIP_SUFFIXES = {".png", ".gguf", ".db", ".bin", ".pyc"}
# Files above this size are left out of the snapshot. They are neither restored nor deleted on deny.
SNAPSHOT_MAX_BYTES = 2_000_000


@dataclass
class Job:
    id: str
    kind: str
    goal: str
    status: str = "queued"
    log: list[tuple[str, str]] = field(default_factory=list)
    diff: str = ""
    changed: list[str] = field(default_factory=list)
    sensitive: bool = False
    control: JobControl = field(default_factory=JobControl)
    project: Path = field(default_factory=Path.cwd)
    before: dict[str, bytes] = field(default_factory=dict)


class JobManager:
    def __init__(self, settings, memory, tools, client) -> None:
        self.settings = settings
        self.memory = memory
        self.tools = tools
        self.client = client
        self.jobs: list[Job] = []
        self.approvals: list[Job] = []
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def enqueue(self, kind: str, goal: str, project: Path | None = None) -> Job:
        target = project or self.tools.project
        job = Job(
            id=uuid.uuid4().hex[:8],
            kind=kind,
            goal=goal,
            sensitive=kind == "deploy" or (kind == "work" and self._touches_valta(goal, target)),
            project=target,
        )
        with self._lock:
            if kind == "deploy":
                job.status = "awaiting approval"
                self.approvals.append(job)
            else:
                self.jobs.append(job)
        self._wake.set()
        return job

    def _touches_valta(self, goal: str, project: Path) -> bool:
        """Work on Valta's own code waits for approval, however the goal is worded.

        Launching Valta from its own folder makes that folder the project, so a plain
        "build me a calculator" would otherwise edit Valta without asking.
        """
        if _owns_the_work(goal):
            return True
        try:
            return Path(project).resolve() == Path(self.settings.root).resolve()
        except OSError:
            return False

    def running(self) -> Job | None:
        with self._lock:
            for job in self.jobs:
                if job.status == "running":
                    return job
        return None

    def pause(self) -> None:
        job = self.running()
        if job:
            job.control.request_pause()
            job.status = "paused"

    def resume(self) -> None:
        with self._lock:
            for job in self.jobs:
                if job.status == "paused":
                    job.control.request_resume()
                    job.status = "running"
                    return

    def stop(self) -> None:
        job = self.running() or self._paused()
        if job:
            job.control.request_stop()

    def approve(self, job_id: str) -> str:
        job = self._approval(job_id)
        if job is None:
            return "No approval is waiting with that id."
        job.status = "approved"
        self.memory.add_record(
            text=f"Approved {job.kind}: {job.goal}",
            kind="fact",
            title=f"approved {job.kind}",
        )
        return f"Approved {job.kind} {job.id}."

    def deny(self, job_id: str) -> str:
        job = self._approval(job_id)
        if job is None:
            return "No approval is waiting with that id."
        if job.kind == "work" and job.before:
            restore(job.project, job.before)
        job.status = "denied"
        return f"Denied {job.kind} {job.id}. Changes were restored." if job.before else f"Denied {job.kind} {job.id}."

    def _approval(self, job_id: str) -> Job | None:
        with self._lock:
            for job in self.approvals:
                if job.id.startswith(job_id) and job.status == "awaiting approval":
                    return job
        return None

    def _paused(self) -> Job | None:
        with self._lock:
            for job in self.jobs:
                if job.status == "paused":
                    return job
        return None

    def _loop(self) -> None:
        while True:
            job = self._next()
            if job is None:
                self._wake.wait(timeout=0.5)
                self._wake.clear()
                continue
            self._run(job)

    def _next(self) -> Job | None:
        with self._lock:
            for job in self.jobs:
                if job.status == "queued":
                    job.status = "running"
                    return job
        return None

    def _run(self, job: Job) -> None:
        try:
            if job.kind == "research":
                from valta.research import research_once

                filed = research_once(self.settings, self.memory, self.tools, self.client, topic=job.goal)
                job.log.append(("research", f"Filed {filed} notes about {job.goal}."))
                job.status = "passed"
                return
            if job.kind != "work":
                job.status = "passed"
                return
            job.before = snapshot(job.project)
            runner = self.tools
            if job.project.resolve() != self.tools.project.resolve():
                from valta.tools import ToolRunner

                runner = ToolRunner(
                    self.settings,
                    self.memory,
                    self.tools.vectors,
                    self.tools.embedder,
                    job.project,
                )

            def on_event(agent: str, text: str) -> None:
                model = self.settings.teacher_model
                job.log.append((f"{agent} · {short_model(model)}", text))

            passed = run_goal(
                self.settings,
                self.memory,
                runner,
                self.client,
                job.goal,
                on_event=on_event,
                control=job.control,
            )
            job.diff, job.changed = diff_since(job.project, job.before)
            if job.control.stop:
                job.status = "stopped"
            elif job.sensitive and passed:
                job.status = "awaiting approval"
                with self._lock:
                    self.approvals.append(job)
            else:
                job.status = "passed" if passed else "failed"
        except JobStopped:
            job.status = "stopped"
        except Exception as exc:
            job.log.append(("valta", str(exc)))
            job.status = "failed"


def short_model(name: str) -> str:
    tail = name.split("/")[-1]
    return tail.replace(":Q4_K_M", "").replace("-GGUF", "")[:32]


def snapshot(root: Path) -> dict[str, bytes]:
    """Exact bytes of every project file worth restoring, keyed by POSIX relative path.

    Bytes, not text: a decode and re-encode would garble non-UTF-8 files and, on Windows,
    turn LF line endings into CRLF when the files are put back.
    """
    saved: dict[str, bytes] = {}
    if not root.is_dir():
        return saved
    for current, dirs, files in os.walk(root):
        # Prune in place so os.walk never descends into node_modules, .venv and the like.
        dirs[:] = [name for name in dirs if name not in SKIP_DIRS]
        for name in files:
            path = Path(current) / name
            if path.suffix.lower() in SKIP_SUFFIXES or path.is_symlink():
                continue
            try:
                if path.stat().st_size > SNAPSHOT_MAX_BYTES:
                    continue
                data = path.read_bytes()
            except OSError:
                continue
            saved[path.relative_to(root).as_posix()] = data
    return saved


def _diff_lines(data: bytes | None) -> list[str] | None:
    """Lines to show in a diff, or None for a binary file."""
    if not data:
        return []
    if b"\0" in data:
        return None
    return data.decode("utf-8", errors="replace").splitlines()


def diff_since(root: Path, before: dict[str, bytes]) -> tuple[str, list[str]]:
    after = snapshot(root)
    changed: list[str] = []
    chunks: list[str] = []
    for rel in sorted(set(before) | set(after)):
        old = before.get(rel)
        new = after.get(rel)
        if old == new:
            continue
        changed.append(rel)
        old_lines, new_lines = _diff_lines(old), _diff_lines(new)
        if old_lines is None or new_lines is None:
            chunks.append(f"Binary file {rel} changed")
            continue
        chunks.extend(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
                lineterm="",
            )
        )
    git = git_diff(root)
    text = "\n".join(chunks)
    if git:
        text = (text + "\n\n# git diff\n" + git).strip()
    return text[:50000], changed


def git_diff(root: Path) -> str:
    if not (root / ".git").exists():
        return ""
    try:
        completed = subprocess.run(
            ["git", "diff", "--", "."],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=20,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout[-20000:]


def restore(root: Path, before: dict[str, bytes]) -> None:
    after = snapshot(root)
    for rel, data in before.items():
        if after.get(rel) != data:
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    for rel in after:
        if rel not in before:
            (root / rel).unlink(missing_ok=True)
