from __future__ import annotations

import functools
import os
import sys
import threading
import time

from rich.console import Group
from rich.markdown import Markdown
from rich.padding import Padding
from rich.table import Table
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.color import Color
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import Input, RichLog, Static, TabbedContent, TabPane

from valta.chat import reply_once, work_folder
from valta.config import load_state
from valta.jobs import JobManager, short_model
from valta.research import research_loop

# Older Windows consoles lack the fancy glyphs; Windows Terminal, VS Code and everything else have them.
FANCY = sys.platform != "win32" or any(os.environ.get(key) for key in ("WT_SESSION", "TERM_PROGRAM", "ConEmuANSI"))
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏" if FANCY else "|/-\\"
DOT = "●" if FANCY else "*"
CARET = "❯" if FANCY else ">"
DEFAULT_THEME = "tokyo-night"
NARROW = 100  # Below this many columns the activity panel takes the whole screen instead of sharing it.

TABS = (
    ("live", "Activity"),
    ("diff", "Changes"),
    ("memory", "Memory"),
    ("models", "Models"),
    ("approvals", "Approvals"),
)

STATUS_COLORS = {
    "running": "primary",
    "queued": "muted",
    "paused": "warning",
    "awaiting approval": "warning",
    "passed": "success",
    "approved": "success",
    "failed": "error",
    "denied": "error",
    "stopped": "muted",
}
KIND_COLORS = {"book": "accent", "research": "primary", "fact": "success", "lesson": "warning"}


def one_line(text: str, limit: int) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _until_closed(method):
    """Timers and worker callbacks can fire while the app is shutting down, after its widgets are gone."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except NoMatches:
            return None

    return wrapper


class Pane(RichLog):
    """A RichLog that keeps what it was given and lays it out again when its width changes, so text rewraps."""

    def __init__(self, *, keep: int = 600, **kwargs) -> None:
        super().__init__(min_width=1, wrap=True, highlight=False, markup=False, **kwargs)
        self.can_focus = False
        self._entries: list = []
        self._keep = keep
        self._laid_out_at = 0

    def add(self, item) -> None:
        self._entries.append(item)
        del self._entries[: -self._keep]
        self.write(item)

    def show(self, items: list, *, home: bool = False) -> None:
        """Replace everything. Keeps the scroll position unless `home` asks for the top."""
        self._entries = list(items)
        top = 0 if home else self.scroll_y
        self._lay_out()
        if not self.auto_scroll:
            self.scroll_to(y=top, animate=False)

    def _lay_out(self) -> None:
        self.clear()
        for item in self._entries:
            self.write(item)

    def on_resize(self, event: events.Resize) -> None:
        width = event.size.width
        if self._size_known and width and width != self._laid_out_at:
            top = self.scroll_y
            self._lay_out()
            if not self.auto_scroll:
                self.scroll_to(y=top, animate=False)
        self._laid_out_at = width


class ValtaApp(App):
    """Terminal control center. One calm chat column; the crew's activity opens beside it when there is some."""

    TITLE = "Valta"
    ENABLE_COMMAND_PALETTE = False
    AUTO_FOCUS = "#prompt"
    CSS = """
    Screen { background: $surface; }

    #bar { height: 1; padding: 0 2; background: $panel; }
    #brand { width: auto; text-style: bold; color: $primary; }
    #health { width: 1fr; height: 1; text-align: right; }

    #body { height: 1fr; }
    #main { width: 1fr; height: 1fr; }

    Pane {
        overflow-x: hidden;
        scrollbar-size-vertical: 1;
        scrollbar-gutter: stable;
        scrollbar-background: $surface;
        scrollbar-background-hover: $surface;
        scrollbar-background-active: $surface;
        scrollbar-color: $panel;
        scrollbar-color-hover: $primary 60%;
        scrollbar-color-active: $primary;
        scrollbar-corner-color: $surface;
    }
    #chat-log { height: 1fr; padding: 1 2 0 2; background: $surface; }
    #status { height: 1; padding: 0 2; background: $surface; }

    #composer { height: auto; padding: 1 2 0 2; background: $panel; }
    #entry { height: 1; }
    #caret { width: 2; color: $primary; text-style: bold; }
    #prompt { width: 1fr; height: 1; border: none; background: $panel; padding: 0; }
    #prompt:focus { border: none; background: $panel; }
    #hints { height: 2; padding: 1 0 0 0; }

    #side {
        display: none;
        width: 46%;
        min-width: 40;
        max-width: 76;
        background: $panel;
        border-left: solid $primary 40%;
    }
    #side.-open { display: block; }
    #tabs { height: 1fr; }
    #side TabPane { padding: 0; }
    #side Pane { height: 1fr; padding: 1 2; background: $panel; scrollbar-background: $panel; scrollbar-corner-color: $panel; scrollbar-color: $surface; }
    #side-foot { height: 1; padding: 0 2; }

    #body.-narrow #side.-open { width: 1fr; max-width: 100%; border-left: none; }
    #body.-narrow.-panel #main { display: none; }
    """
    BINDINGS = [
        Binding("ctrl+o", "toggle_panel", "Activity", show=False),
        Binding("escape", "close_panel", "Close", show=False),
        Binding("f1", "tab('live')", "Activity", show=False),
        Binding("f2", "tab('diff')", "Changes", show=False),
        Binding("f3", "tab('memory')", "Memory", show=False),
        Binding("f4", "tab('models')", "Models", show=False),
        Binding("f5", "tab('approvals')", "Approvals", show=False),
        Binding("ctrl+p", "pause", "Pause", show=False),
        Binding("ctrl+r", "resume", "Resume", show=False),
        Binding("ctrl+x", "stop_job", "Stop", show=False),
        Binding("ctrl+a", "approve", "Approve", show=False),
        Binding("ctrl+d", "deny", "Deny", show=False),
        Binding("ctrl+q", "quit", "Quit", show=False),
        Binding("up", "history(-1)", "Previous", show=False),
        Binding("down", "history(1)", "Next", show=False),
        Binding("pageup", "scroll_chat(-1)", "Scroll up", show=False),
        Binding("pagedown", "scroll_chat(1)", "Scroll down", show=False),
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
        self._health: dict[str, bool | None] = {"teacher": None, "personal": None}
        self._loaded: dict[str, list[str]] = {"teacher": [], "personal": []}
        self._personal_name = settings.self_model
        self._busy = 0
        self._frame = 0
        self._typed: list[str] = []
        self._typed_at = 0
        self._jobs_seen = 0
        self._approvals_seen: set[str] = set()
        self._shown: dict[str, object] = {}
        self._palette: tuple[str, dict[str, str]] = ("", {})
        name = os.environ.get("VALTA_THEME") or getattr(settings, "ui_theme", "") or DEFAULT_THEME
        try:
            self.theme = name
        except Exception:
            pass  # An unknown theme name leaves Textual's default in place.

    # ---------------------------------------------------------------- layout

    def compose(self) -> ComposeResult:
        with Horizontal(id="bar"):
            yield Static("valta", id="brand")
            yield Static("", id="health")
        with Horizontal(id="body"):
            with Vertical(id="main"):
                yield Pane(id="chat-log", auto_scroll=True)
                yield Static("", id="status")
                with Vertical(id="composer"):
                    with Horizontal(id="entry"):
                        yield Static(CARET, id="caret")
                        yield Input(placeholder="Ask anything, or give Valta an objective", id="prompt")
                    yield Static("", id="hints")
            with Vertical(id="side"):
                with TabbedContent(id="tabs"):
                    for tab_id, label in TABS:
                        with TabPane(label, id=tab_id):
                            yield Pane(id=f"{tab_id}-log", auto_scroll=False)
                yield Static("", id="side-foot")

    def on_mount(self) -> None:
        self._welcome()
        self._fit(self.size.width)
        self.jobs.start()
        self.set_interval(1.0, self.refresh_panels)
        self.set_interval(0.12, self._tick)
        threading.Thread(target=self._watch_models, daemon=True).start()
        self.refresh_panels()
        self._tick()
        if self.start_research and self.settings.research_interval_sec > 0:
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

    def on_resize(self, event: events.Resize) -> None:
        self._fit(event.size.width)

    def _fit(self, width: int) -> None:
        self.query_one("#body").set_class(width < NARROW, "-narrow")

    # ---------------------------------------------------------------- colors

    def _color(self, name: str) -> str:
        """A theme color as #rrggbb for Rich text. Rebuilt only when the theme changes."""
        if self._palette[0] != self.theme:
            variables = self.get_css_variables()

            def pick(key: str, fallback: str) -> str:
                try:
                    return Color.parse(variables.get(key, fallback)).hex[:7]
                except Exception:
                    return fallback

            text, ground = pick("foreground", "#dddddd"), pick("surface", "#222222")
            muted = Color.parse(text).blend(Color.parse(ground), 0.45).hex[:7]
            colors = {key: pick(key, "#888888") for key in ("primary", "accent", "success", "warning", "error")}
            colors.update(muted=muted, text=text)
            self._palette = (self.theme, colors)
        return self._palette[1].get(name, "")

    def _dot(self, state: bool | str | None) -> Text:
        """A status dot: green up, red down, grey unknown. Strings are job statuses."""
        if isinstance(state, str):
            color = self._color(STATUS_COLORS.get(state, "muted"))
        else:
            color = self._color("success" if state else "error" if state is False else "muted")
        dot = Text()
        dot.append(f"{DOT} ", style=color)
        return dot

    # ---------------------------------------------------------------- chat

    def _chat(self) -> Pane:
        return self.query_one("#chat-log", Pane)

    def _welcome(self) -> None:
        chat = self._chat()
        chat.add(Text("Remembers what you tell it, looks things up, and builds software.", style="dim"))
        chat.add("")
        for line in ("Build me a calculator", "Research how vector databases work", "Remember that I prefer Python"):
            row = Text("  try  ", style="dim")
            row.append(line)
            chat.add(row)
        chat.add("")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.clear()
        if not text:
            return
        if not self._typed or self._typed[-1] != text:
            self._typed.append(text)
        self._typed_at = len(self._typed)
        self._user(text)
        self._busy += 1
        self.run_worker(lambda: self._respond(text), thread=True, exclusive=False)

    def _user(self, text: str) -> None:
        line = Text(f"{CARET} ", style=f"bold {self._color('primary')}")
        line.append(text, style="bold")
        self._chat().add(line)

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
            self.call_from_thread(self._note, f"Queued {job.kind} {job.id}: {one_line(arg, 80)}")

        failed = False
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
            answer, failed = str(exc), True
        self.call_from_thread(self._say, answer, failed)

    @_until_closed
    def _say(self, text: str, failed: bool = False) -> None:
        self._busy = max(0, self._busy - 1)
        chat = self._chat()
        body = Text(text, style=self._color("error")) if failed else Markdown(text)
        chat.add(Padding(body, (0, 0, 0, 2)))
        chat.add("")

    @_until_closed
    def _note(self, text: str) -> None:
        self._chat().add(Padding(Text(text, style="dim"), (0, 0, 0, 2)))

    def action_history(self, step: int) -> None:
        prompt = self.query_one("#prompt", Input)
        if not prompt.has_focus or not self._typed:
            return
        self._typed_at = min(max(self._typed_at + step, 0), len(self._typed))
        prompt.value = self._typed[self._typed_at] if self._typed_at < len(self._typed) else ""
        prompt.cursor_position = len(prompt.value)

    def action_scroll_chat(self, direction: int) -> None:
        if direction < 0:
            self._chat().scroll_page_up(animate=False)
        else:
            self._chat().scroll_page_down(animate=False)

    # ---------------------------------------------------------------- panel

    def _open_panel(self, tab: str | None = None) -> None:
        self.query_one("#side").add_class("-open")
        self.query_one("#body").add_class("-panel")
        if tab:
            self.query_one(TabbedContent).active = tab
        self._draw_panels()

    def action_toggle_panel(self) -> None:
        if self.query_one("#side").has_class("-open"):
            self.action_close_panel()
        else:
            self._open_panel()

    def action_close_panel(self) -> None:
        self.query_one("#side").remove_class("-open")
        self.query_one("#body").remove_class("-panel")
        self.query_one("#prompt", Input).focus()

    def action_tab(self, tab: str) -> None:
        self._open_panel(tab)
        self.query_one("#prompt", Input).focus()

    # ---------------------------------------------------------------- job controls

    def action_pause(self) -> None:
        self.jobs.pause()
        self._note("Paused.")

    def action_resume(self) -> None:
        self.jobs.resume()
        self._note("Resumed.")

    def action_stop_job(self) -> None:
        self.jobs.stop()
        self._note("Stopping the running job.")

    def action_approve(self) -> None:
        pending = self._first_approval()
        self._note(self.jobs.approve(pending.id) if pending else "Nothing is waiting for approval.")

    def action_deny(self) -> None:
        pending = self._first_approval()
        self._note(self.jobs.deny(pending.id) if pending else "Nothing is waiting for approval.")

    # ---------------------------------------------------------------- live status

    def _watch_models(self) -> None:
        while self._alive:
            try:
                probe = self._probe()
            except Exception:
                probe = {"teacher": (None, []), "personal": (None, [])}
            if self._alive:
                try:
                    self.call_from_thread(self._store_status, probe)
                except Exception:
                    return
            time.sleep(2)

    def _probe(self) -> dict[str, tuple[bool | None, list[str]]]:
        """Is each model server up, and what has it loaded. One request pair per host."""
        found: dict[str, tuple[bool | None, list[str]]] = {}
        for key, host in (("teacher", self.settings.heavy_host), ("personal", self.settings.self_host)):
            if not self.client.healthy(host):
                found[key] = (False, [])
                continue
            try:
                loaded = [f"{m.get('name', '')}  {m.get('processor', '')}".strip() for m in self.client.ps(host)]
            except Exception:
                loaded = []
            found[key] = (True, loaded)
        return found

    @_until_closed
    def _store_status(self, probe: dict[str, tuple[bool | None, list[str]]]) -> None:
        self._health = {key: value[0] for key, value in probe.items()}
        self._loaded = {key: value[1] for key, value in probe.items()}
        try:
            self._personal_name = load_state(self.settings).get("personal_model") or self.settings.self_model
        except Exception:
            pass
        self._draw_bar()

    def _draw_bar(self) -> None:
        health = Text()
        for label, key, model in (
            ("teacher", "teacher", short_model(self.settings.teacher_model)),
            ("personal", "personal", short_model(self._personal_name)),
        ):
            if health:
                health.append("    ")
            health.append_text(self._dot(self._health.get(key)))
            health.append(label)
            health.append(f" {model}", style="dim")
        self._set("#health", health)

    def _set(self, selector: str, content) -> None:
        """Update a Static only when what it shows has changed."""
        if self._shown.get(selector) != content:
            self._shown[selector] = content
            self.query_one(selector, Static).update(content)

    def _active_job(self):
        return next((job for job in self.jobs.jobs if job.status in {"running", "paused"}), None)

    @_until_closed
    def _tick(self) -> None:
        """Several times a second: the spinner line and the key hints. Kept cheap on purpose."""
        self._frame = (self._frame + 1) % len(SPINNER)
        job = self._active_job()
        primary = self._color("primary")
        status = Text()
        if job is not None and job.status == "paused":
            status.append(f"{DOT} paused", style=self._color("warning"))
            status.append(f"  {one_line(job.goal, 60)}", style="dim")
        elif job is not None:
            agent = job.log[-1][0].split(" · ")[0] if job.log else "planning"
            status.append(f"{SPINNER[self._frame]} {agent}", style=primary)
            status.append(f"  {one_line(job.goal, 60)}", style="dim")
        elif self._busy:
            status.append(f"{SPINNER[self._frame]} thinking", style=primary)
        self._set("#status", status)

        # At most three hints, most relevant first.
        keys: list[tuple[str, str]] = []
        if self._first_approval() is not None:
            keys += [("ctrl+a", "approve"), ("ctrl+d", "deny")]
        if job is not None:
            keys += [("ctrl+r", "resume")] if job.status == "paused" else [("ctrl+p", "pause")]
            keys += [("ctrl+x", "stop")]
        keys += [("ctrl+o", "activity"), ("ctrl+q", "quit")]
        hints = Text()
        for key, label in keys[:3]:
            if hints:
                hints.append("   ")
            hints.append(key, style=f"bold {primary}")
            hints.append(f" {label}", style="dim")
        self._set("#hints", hints)

        foot = Text()
        foot.append("esc", style=f"bold {primary}")
        foot.append(" close   ", style="dim")
        foot.append("f1-f5", style=f"bold {primary}")
        foot.append(" switch tab", style="dim")
        self._set("#side-foot", foot)

    # ---------------------------------------------------------------- panels

    @_until_closed
    def refresh_panels(self) -> None:
        self._draw_panels()
        self._auto_open([job for job in self.jobs.approvals if job.status == "awaiting approval"])

    def _draw_panels(self) -> None:
        self._draw_bar()
        jobs = self.jobs.jobs
        self._fill(
            "#live-log",
            (tuple((job.id, job.status, len(job.log)) for job in jobs), self.theme),
            self._live_lines,
        )
        self._fill(
            "#diff-log",
            (tuple((job.id, len(job.diff), len(job.changed)) for job in jobs), self.theme),
            self._diff_lines,
        )
        self._fill("#memory-log", (tuple(record.id for record in self.memory.recent(30)), self.theme), self._memory_lines)
        self._fill(
            "#models-log",
            (self._health, self._loaded, self._personal_name, self.theme),
            self._model_lines,
        )
        self._fill(
            "#approvals-log",
            (tuple((job.id, job.status) for job in self.jobs.approvals), self.theme),
            self._approval_lines,
        )
        self._badge("live", "Activity", sum(1 for job in jobs if job.status in {"running", "paused"}))
        self._badge("approvals", "Approvals", sum(1 for job in self.jobs.approvals if job.status == "awaiting approval"))

    def _fill(self, selector: str, key, build) -> None:
        if self._shown.get(selector) == key:
            return
        self._shown[selector] = key
        self.query_one(selector, Pane).show(build(), home=selector in ("#memory-log", "#models-log"))

    def _badge(self, tab_id: str, label: str, count: int) -> None:
        text = f"{label} ({count})" if count else label
        if self._shown.get(f"tab:{tab_id}") == text:
            return
        try:
            self.query_one(TabbedContent).get_tab(tab_id).label = text
            self._shown[f"tab:{tab_id}"] = text
        except Exception:
            pass

    def _auto_open(self, waiting: list) -> None:
        """Open the panel when something needs a look: a new job or an approval. Closing it sticks until then."""
        pending = {job.id for job in waiting}
        new_job = len(self.jobs.jobs) > self._jobs_seen
        starts_work = new_job and self.jobs.jobs[-1].kind == "work"
        self._jobs_seen = len(self.jobs.jobs)
        if pending - self._approvals_seen:
            self._approvals_seen |= pending
            self._open_panel("approvals")
        elif starts_work:
            self._open_panel("live")

    def _live_lines(self) -> list:
        jobs = [job for job in self.jobs.jobs if job.kind == "work" or job.status in {"running", "paused"}]
        if not jobs:
            return [Text("Nothing is running.\nGive Valta an objective in the prompt.", style="dim")]
        active = [job for job in jobs if job.status in {"running", "paused", "queued"}]
        done = [job for job in reversed(jobs) if job not in active]
        accent = self._color("accent")
        items: list = []
        for job in (active + done)[:6]:
            head = Text()
            head.append_text(self._dot(job.status))
            head.append(one_line(job.goal, 70), style="bold")
            rows = Table.grid(padding=(0, 2))
            rows.add_column(width=9, no_wrap=True)
            rows.add_column(ratio=1)
            for agent, text in job.log[-(8 if job in active else 3) :]:
                rows.add_row(Text(agent.split(" · ")[0], style=f"bold {accent}"), Text(one_line(text, 400)))
            items.append(Group(head, Text(f"  {job.status} · {job.id}", style="dim"), Padding(rows, (0, 0, 0, 2)), ""))
        return items

    def _diff_lines(self) -> list:
        items: list = []
        green, red, accent = self._color("success"), self._color("error"), self._color("accent")
        for job in reversed(self.jobs.jobs):
            if not job.changed and not job.diff:
                continue
            head = Text(one_line(job.goal, 70), style="bold")
            head.append(f"  {job.id}", style="dim")
            items += [head, Text(", ".join(job.changed) if job.changed else "No file changes.", style="dim")]
            for raw in job.diff[:8000].splitlines():
                if raw.startswith(("+++", "---", "@@")):
                    items.append(Text(raw, style=accent))
                elif raw.startswith("+"):
                    items.append(Text(raw, style=green))
                elif raw.startswith("-"):
                    items.append(Text(raw, style=red))
                else:
                    items.append(Text(raw, style="dim"))
            items.append("")
        return items or [Text("Changes appear here after a job edits files.", style="dim")]

    def _memory_lines(self) -> list:
        records = self.memory.recent(30)
        if not records:
            return [Text("Memory is empty.", style="dim")]
        items: list = []
        for record in records:
            head = Text()
            head.append(f"{record.kind:<9}", style=self._color(KIND_COLORS.get(record.kind, "muted")))
            head.append(one_line(record.title or record.kind, 60), style="bold")
            items.append(Group(head, Padding(Text(one_line(record.text, 160), style="dim"), (0, 0, 0, 9)), ""))
        return items

    def _model_lines(self) -> list:
        items: list = []
        for label, key, model, host in (
            ("Teacher", "teacher", self.settings.teacher_model, self.settings.heavy_host),
            ("Personal", "personal", self._personal_name, self.settings.self_host),
        ):
            up = self._health.get(key)
            head = self._dot(up)
            head.append(label, style="bold")
            head.append(f"  {'up' if up else 'down' if up is False else 'checking'}", style="dim")
            lines = [head, Text(f"  {model}"), Text(f"  {host}", style="dim")]
            lines += [Text(f"  loaded  {item}", style="dim") for item in self._loaded.get(key, [])]
            items.append(Group(*lines, ""))
        items.append(
            Text("The teacher plans, codes and reviews. The personal model talks about you once it is trained.", style="dim")
        )
        return items

    def _approval_lines(self) -> list:
        waiting = [job for job in self.jobs.approvals if job.status == "awaiting approval"]
        if not waiting:
            return [Text("Nothing is waiting.\nChanges to Valta's own code stop here for you.", style="dim")]
        primary = self._color("primary")
        items: list = []
        for job in waiting:
            head = self._dot("awaiting approval")
            head.append(f"{job.kind}  ", style="bold")
            head.append(one_line(job.goal, 60))
            lines = [head]
            if job.changed:
                lines.append(Text("  " + ", ".join(job.changed[:6]), style="dim"))
            action = Text("  ")
            action.append("ctrl+a", style=f"bold {primary}")
            action.append(" keep   ", style="dim")
            action.append("ctrl+d", style=f"bold {primary}")
            action.append(" restore the old files", style="dim")
            lines += [action, Text("  Nothing is pushed.", style="dim")]
            items.append(Group(*lines, ""))
        return items

    def _first_approval(self):
        for job in self.jobs.approvals:
            if job.status == "awaiting approval":
                return job
        return None


def _self(goal: str) -> bool:
    from valta.chat import _owns_the_work

    return _owns_the_work(goal)
