import asyncio
from pathlib import Path

from textual.widgets import Static, TabbedContent

from valta.config import load_settings
from valta.jobs import Job
from valta.memory import Memory
from valta.tools import ToolRunner
from valta.ui import ValtaApp
from valta.vectors import Vectors


class _Pilot:
    def healthy(self, host):
        return True

    def ps(self, host):
        return []

    def chat(self, host, model, messages, num_ctx):
        return "TALK\nReady."


def _settings(tmp_path: Path, **changes):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "config.toml").write_text((root / "config.toml").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "rules").mkdir()
    (tmp_path / "rules" / "root.md").write_text((root / "rules" / "root.md").read_text(encoding="utf-8"), encoding="utf-8")
    settings = load_settings(tmp_path)
    for key, value in changes.items():
        setattr(settings, key, value)
    return settings


def _app(tmp_path: Path, **changes) -> ValtaApp:
    settings = _settings(tmp_path, **changes)
    memory = Memory(settings)
    tools = ToolRunner(settings, memory, Vectors(settings), lambda text: [0.1, 0.2, 0.3], tmp_path)
    return ValtaApp(settings, memory, tools, _Pilot(), start_research=False)


def _job(job_id: str, goal: str, status: str, kind: str = "work") -> Job:
    # Never "queued": the job manager's thread would pick that up and try to run it.
    return Job(id=job_id, kind=kind, goal=goal, status=status)


def _text(app: ValtaApp, selector: str) -> str:
    return str(app.query_one(selector, Static).content)


def test_control_center_composes_and_the_panel_starts_closed(tmp_path: Path):
    app = _app(tmp_path)

    async def run():
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.pause()
            assert app.query_one("#chat-log").lines
            assert not app.query_one("#side").display
            assert app.focused is app.query_one("#prompt")
            await pilot.press("f5")
            await pilot.pause()
            assert app.query_one("#side").display
            assert app.query_one(TabbedContent).active == "approvals"
            assert app.query_one("#approvals-log").lines
            await pilot.press("escape")
            await pilot.pause()
            assert not app.query_one("#side").display
            await pilot.press("ctrl+o")
            await pilot.pause()
            assert app.query_one("#side").display
            await pilot.press("f1")
            await pilot.pause()
            assert app.query_one(TabbedContent).active == "live"
            assert app.query_one("#live-log").lines
            await pilot.press("ctrl+o")
            await pilot.pause()
            assert not app.query_one("#side").display

    asyncio.run(run())


def test_a_reply_lands_in_the_chat_and_arrow_keys_recall_what_was_typed(tmp_path: Path):
    app = _app(tmp_path)

    async def run():
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.pause()
            before = len(app.query_one("#chat-log").lines)
            await pilot.press(*"hello there", "enter")
            for _ in range(100):
                await pilot.pause(0.05)
                if app._busy == 0:
                    break
            assert app._busy == 0
            assert len(app.query_one("#chat-log").lines) > before
            await pilot.press(*"second", "enter")
            for _ in range(100):
                await pilot.pause(0.05)
                if app._busy == 0:
                    break
            await pilot.press("up")
            assert app.query_one("#prompt").value == "second"
            await pilot.press("up")
            assert app.query_one("#prompt").value == "hello there"
            await pilot.press("down", "down")
            assert app.query_one("#prompt").value == ""

    asyncio.run(run())


def test_new_work_opens_activity_and_an_approval_opens_approvals(tmp_path: Path):
    app = _app(tmp_path)

    async def run():
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.pause()
            assert not app.query_one("#side").display
            app.jobs.jobs.append(_job("aaaa1111", "Build me a calculator", "running"))
            app.refresh_panels()
            app._tick()
            await pilot.pause()
            assert app.query_one("#side").display
            assert app.query_one(TabbedContent).active == "live"
            assert "Build me a calculator" in _text(app, "#status")
            assert "ctrl+p" in _text(app, "#hints")

            await pilot.press("escape")
            app.refresh_panels()
            await pilot.pause()
            assert not app.query_one("#side").display  # closing it sticks until something new happens

            waiting = _job("bbbb2222", "Add a status command", "awaiting approval")
            app.jobs.jobs.append(waiting)
            app.jobs.approvals.append(waiting)
            app.refresh_panels()
            app._tick()
            await pilot.pause()
            assert app.query_one("#side").display
            assert app.query_one(TabbedContent).active == "approvals"
            assert "ctrl+a" in _text(app, "#hints")
            assert app._first_approval() is waiting

    asyncio.run(run())


def test_a_paused_job_offers_resume(tmp_path: Path):
    app = _app(tmp_path)

    async def run():
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.pause()
            app.jobs.jobs.append(_job("cccc3333", "Long task", "paused"))
            app.refresh_panels()
            app._tick()
            await pilot.pause()
            assert "paused" in _text(app, "#status")
            assert "ctrl+r" in _text(app, "#hints")
            assert "ctrl+p" not in _text(app, "#hints")

    asyncio.run(run())


def test_the_health_bar_shows_both_servers(tmp_path: Path):
    app = _app(tmp_path)

    async def run():
        async with app.run_test(size=(120, 36)) as pilot:
            for _ in range(60):
                await pilot.pause(0.05)
                if app._health.get("teacher") is not None:
                    break
            shown = _text(app, "#health")
            assert "teacher" in shown and "personal" in shown
            assert app._health == {"teacher": True, "personal": True}

    asyncio.run(run())


def test_on_a_narrow_terminal_the_panel_takes_the_whole_width(tmp_path: Path):
    app = _app(tmp_path)

    async def run():
        async with app.run_test(size=(80, 28)) as pilot:
            await pilot.pause()
            assert app.query_one("#main").display
            await pilot.press("f3")
            await pilot.pause()
            assert app.query_one("#side").display
            assert not app.query_one("#main").display
            await pilot.press("escape")
            await pilot.pause()
            assert app.query_one("#main").display
            assert app.focused is app.query_one("#prompt")

    asyncio.run(run())


def test_chat_text_rewraps_when_the_panel_narrows_it(tmp_path: Path):
    app = _app(tmp_path)

    async def run():
        async with app.run_test(size=(120, 36)) as pilot:
            await pilot.pause()
            app._say("word " * 60)
            await pilot.pause()
            wide = len(app.query_one("#chat-log").lines)
            await pilot.press("ctrl+o")
            await pilot.pause()
            await pilot.pause()
            assert len(app.query_one("#chat-log").lines) > wide

    asyncio.run(run())


def test_an_unknown_theme_falls_back_instead_of_crashing(tmp_path: Path):
    app = _app(tmp_path, ui_theme="no-such-theme")

    async def run():
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            assert app.query_one("#chat-log").lines

    asyncio.run(run())
