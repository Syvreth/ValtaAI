from __future__ import annotations

from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static, TabbedContent, TabPane

from valta.chat import reply_once, work_folder
from valta.jobs import JobManager
from valta.research import research_loop


class ValtaApp(App):
    """Terminal control center for chat, jobs, memory, models, and approvals."""

    TITLE = "Valta"
    SUB_TITLE = "Control center"
    ENABLE_COMMAND_PALETTE = False
    CSS = """
    Screen { background: $surface; }
    #model-bar {
        height: 2;
        padding: 0 1;
        background: $boost;
        color: $text;
    }
    #body { height: 1fr; }
    #chat-pane {
        width: 1fr;
        height: 1fr;
        border: solid $primary;
        border-title-align: left;
    }
    #side {
        width: 1fr;
        height: 1fr;
        border: solid $accent;
        border-title-align: left;
    }
    RichLog { height: 1fr; background: $surface; }
    #prompt { dock: bottom; margin: 0 1 1 1; }
    TabbedContent { height: 1fr; }
    """
    BINDINGS = [
        Binding("f1", "tab('live')", "Live", show=False),
        Binding("f2", "tab('diff')", "Diff", show=False),
        Binding("f3", "tab('jobs')", "Jobs", show=False),
        Binding("f4", "tab('memory')", "Memory", show=False),
        Binding("f5", "tab('models')", "Models", show=False),
        Binding("f6", "tab('approvals')", "Approvals", show=False),
        Binding("ctrl+p", "pause", "Pause"),
        Binding("ctrl+r", "resume", "Resume", show=False),
        Binding("ctrl+x", "stop_job", "Stop"),
        Binding("ctrl+a", "approve", "Approve"),
        Binding("ctrl+d", "deny", "Deny"),
    ]

    def __init__(self, settings, memory, tools, client, start_research: bool = True) -> None:
        super().__init__()
        self.settings = settings
        self.memory = memory
        self.tools = tools
        self.client = client
        self.jobs = JobManager(settings, memory, tools, client)
        self.start_research = start_research
        self._stop = None
        self._alive = True
        self.history: list[dict] = []
        self._status_line = "Checking the model servers..."
        self._loaded: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(self._status_line, id="model-bar")
        with Horizontal(id="body"):
            with Vertical(id="chat-pane"):
                yield RichLog(id="chat-log", highlight=True, markup=True, wrap=True, max_lines=400)
            with TabbedContent(id="side"):
                with TabPane("Live", id="live"):
                    yield RichLog(id="live-log", highlight=True, markup=True, wrap=True)
                with TabPane("Diff", id="diff"):
                    yield RichLog(id="diff-log", highlight=False, markup=False, wrap=False)
                with TabPane("Jobs", id="jobs"):
                    yield RichLog(id="jobs-log", highlight=True, markup=False, wrap=True)
                with TabPane("Memory", id="memory"):
                    yield RichLog(id="memory-log", highlight=True, markup=True, wrap=True)
                with TabPane("Models", id="models"):
                    yield RichLog(id="models-log", highlight=True, markup=True, wrap=True)
                with TabPane("Approvals", id="approvals"):
                    yield RichLog(id="approvals-log", highlight=True, markup=False, wrap=True)
        yield Input(placeholder="Tell Valta an objective, or ask a question", id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        self.query_one("#chat-pane").border_title = "Session"
        self.query_one("#side").border_title = "Operations"
        chat.write("[bold]Valta control center[/bold]")
        chat.write("Give an objective here. The crew reports under Live.")
        chat.write("F1 live   F2 diff   F3 jobs   F4 memory   F5 models   F6 approvals")
        chat.write("Ctrl+P pause   Ctrl+R resume   Ctrl+X stop   Ctrl+A approve   Ctrl+D deny")
        self.jobs.start()
        self.set_interval(1.0, self.refresh_panels)
        import threading

        threading.Thread(target=self._watch_models, daemon=True).start()
        self.refresh_panels()
        if self.start_research:
            self._stop = threading.Event()
            threading.Thread(
                target=research_loop,
                args=(self.settings, self.memory, self.tools, self.client, self._stop),
                kwargs={"delay_first": True},
                daemon=True,
            ).start()

    def on_unmount(self) -> None:
        self._alive = False
        if self._stop is not None:
            self._stop.set()

    def _watch_models(self) -> None:
        import time

        while self._alive:
            try:
                line = self._model_line()
                loaded = self._loaded_lines()
            except Exception as exc:
                line = str(exc)
                loaded = []
            if self._alive:
                try:
                    self.call_from_thread(self._store_status, line, loaded)
                except Exception:
                    return
            time.sleep(2)

    def _store_status(self, line: str, loaded: list[str]) -> None:
        self._status_line = line
        self._loaded = loaded

    def action_tab(self, tab: str) -> None:
        self.query_one(TabbedContent).active = tab
        self.refresh_panels()

    def action_pause(self) -> None:
        self.jobs.pause()
        self._note("Paused the running job.")

    def action_resume(self) -> None:
        self.jobs.resume()
        self._note("Resumed.")

    def action_stop_job(self) -> None:
        self.jobs.stop()
        self._note("Stopping the running job.")

    def action_approve(self) -> None:
        pending = self._first_approval()
        if pending is None:
            self._note("Nothing is waiting for approval.")
            return
        self._note(self.jobs.approve(pending.id))

    def action_deny(self) -> None:
        pending = self._first_approval()
        if pending is None:
            self._note("Nothing is waiting for approval.")
            return
        self._note(self.jobs.deny(pending.id))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.clear()
        if not text:
            return
        self.query_one("#chat-log", RichLog).write(f"[bold]You[/bold]  {escape(text)}")
        self.run_worker(lambda: self._respond(text), thread=True, exclusive=False)

    def _respond(self, text: str) -> None:
        def on_task(kind: str, arg: str) -> None:
            project = self.settings.root if kind == "work" and _self(arg) else work_folder(arg, self.tools.project)
            if kind == "ingest":
                from pathlib import Path

                from valta.ingest import ingest_path

                count = ingest_path(Path(arg.strip().strip('"')), self.settings, self.memory, self.tools)
                self.call_from_thread(self._note, f"Read {count} chunks from {arg}.")
                return
            job = self.jobs.enqueue(kind, arg, project)
            self.call_from_thread(self._note, f"Queued {job.kind} {job.id}: {arg}")

        try:
            answer = reply_once(
                self.settings,
                self.memory,
                self.tools,
                self.client,
                self.history,
                text,
                on_task=on_task,
            )
        except Exception as exc:
            answer = str(exc)
        self.call_from_thread(self._say, answer)

    def _say(self, text: str) -> None:
        self.query_one("#chat-log", RichLog).write(f"[bold]Valta[/bold]  {escape(text)}")

    def _note(self, text: str) -> None:
        self.query_one("#chat-log", RichLog).write(f"[dim]{escape(text)}[/dim]")

    def refresh_panels(self) -> None:
        bar = self.query_one("#model-bar", Static)
        bar.update(self._status_line)
        self._fill_log("#live-log", self._live_lines())
        self._fill_log("#diff-log", self._diff_lines())
        self._fill_log("#jobs-log", self._job_lines())
        self._fill_log("#memory-log", self._memory_lines())
        self._fill_log("#models-log", self._model_lines())
        self._fill_log("#approvals-log", self._approval_lines())

    def _fill_log(self, selector: str, lines: list[str]) -> None:
        log = self.query_one(selector, RichLog)
        key = "\n".join(lines)
        if getattr(log, "_valta_key", None) == key:
            return
        log._valta_key = key
        log.clear()
        for line in lines:
            log.write(line)
        log.scroll_end(animate=False)

    def _live_lines(self) -> list[str]:
        lines = []
        for job in self.jobs.jobs:
            if job.kind != "work" and job.status not in {"running", "paused"}:
                continue
            lines.append(f"[bold]{escape(job.id)}[/bold]  {escape(job.status)}  {escape(job.goal)}")
            for agent, text in job.log[-12:]:
                lines.append(f"[cyan]{escape(agent)}[/cyan]\n{escape(text[:1200])}")
        return lines or ["No work is running. Give Valta an objective in the prompt."]

    def _diff_lines(self) -> list[str]:
        lines = []
        for job in reversed(self.jobs.jobs):
            if not job.changed and not job.diff:
                continue
            lines.append(f"{job.id}  {job.goal}")
            lines.append("Changed: " + ", ".join(job.changed) if job.changed else "No file changes.")
            if job.diff:
                lines.append(job.diff[:8000])
        return lines or ["Diffs appear here after a job edits files."]

    def _job_lines(self) -> list[str]:
        if not self.jobs.jobs:
            return ["The queue is empty."]
        return [f"{job.id}  {job.status:<18}  {job.kind:<10}  {job.goal}" for job in self.jobs.jobs]

    def _memory_lines(self) -> list[str]:
        rows = self.memory.recent(30)
        if not rows:
            return ["Memory is empty."]
        lines = []
        for record in rows:
            title = record.title or record.kind
            lines.append(f"[bold]{escape(record.kind)}[/bold]  {escape(title)}\n{escape(record.text[:500])}")
        return lines

    def _model_lines(self) -> list[str]:
        return [
            "Planner, coder, tester, and reviewer use the teacher model.",
            "Conversation about you uses the personal model once it has been trained.",
            *self._status_line.splitlines(),
            *self._loaded,
        ]

    def _approval_lines(self) -> list[str]:
        waiting = [job for job in self.jobs.approvals if job.status == "awaiting approval"]
        if not waiting:
            return ["No sensitive action is waiting. Self-improvements and deploys stop here until you approve."]
        lines = []
        for job in waiting:
            lines.append(f"{job.id}  {job.kind}  {job.goal}")
            lines.append("Ctrl+A keeps the verified edits in the project. Ctrl+D restores the previous files. Nothing is pushed.")
        return lines

    def _model_line(self) -> str:
        teacher = "up" if self.client.healthy(self.settings.heavy_host) else "down"
        personal = "up" if self.client.healthy(self.settings.self_host) else "down"
        return (
            f"Teacher   {teacher:<4}  {self.settings.teacher_model}   {self.settings.heavy_host}\n"
            f"Personal  {personal:<4}  {self.settings.self_model}   {self.settings.self_host}"
        )

    def _loaded_lines(self) -> list[str]:
        lines = []
        for label, host in (("Teacher", self.settings.heavy_host), ("Personal", self.settings.self_host)):
            if not self.client.healthy(host):
                lines.append(f"{label}: server down")
                continue
            try:
                loaded = self.client.ps(host)
            except Exception as exc:
                lines.append(f"{label}: {exc}")
                continue
            if not loaded:
                lines.append(f"{label}: idle")
            for item in loaded:
                lines.append(f"{label}: {item.get('name', '')}  {item.get('processor', '')}")
        return lines

    def _first_approval(self):
        for job in self.jobs.approvals:
            if job.status == "awaiting approval":
                return job
        return None


def _self(goal: str) -> bool:
    from valta.chat import _owns_the_work

    return _owns_the_work(goal)
