from __future__ import annotations

import os
import sys
from pathlib import Path

import typer

from valta.chat import chat_loop, live
from valta.config import load_settings
from valta.engineer import run_goal
from valta.ingest import ingest_path
from valta.memory import Memory
from valta.ollama_client import OllamaClient, OllamaError
from valta.prompt import load_rules
from valta.research import research_once
from valta.tools import ToolRunner
from valta.train import (
    build_pairs,
    convert_to_gguf,
    create_valta,
    free_training_gpu,
    merge_adapter,
    rollback,
    run_qlora,
    write_modelfile,
)
from valta.vectors import Vectors

app = typer.Typer(help="Valta, a companion that grows.", no_args_is_help=True)
models_app = typer.Typer(help="Ollama models.")
app.add_typer(models_app, name="models")


def _stack(project: Path | None = None):
    settings = load_settings()
    memory = Memory(settings)
    vectors = Vectors(settings)
    client = OllamaClient(settings)

    def embed(text: str) -> list[float]:
        return client.embed(settings.heavy_host, settings.embed_model, text)

    tools = ToolRunner(
        settings,
        memory,
        vectors,
        embed,
        project or Path.cwd(),
    )
    return settings, memory, tools, client


@app.command()
def chat():
    """Talk to Valta. Research and coding use the 7B. Personal talk uses the fine-tune once it exists."""
    settings, memory, tools, client = _stack()
    chat_loop(settings, memory, tools, client)


@app.command("live")
def live_cmd():
    """Chat, and research in the background."""
    settings, memory, tools, client = _stack()
    live(settings, memory, tools, client)


@app.command()
def ingest(path: Path):
    """Read a PDF, EPUB, TXT, or MD file, or a folder of them, into memory."""
    settings, memory, tools, _client = _stack()
    count = ingest_path(path, settings, memory, tools)
    typer.echo(f"Stored {count} chunks.")


@app.command()
def work(
    goal: str,
    project: Path | None = typer.Option(None, "--project", help="Folder Valta may change."),
    tests: str = typer.Option("python -m pytest -q", "--tests", help="python -m pytest or python -m compileall."),
):
    """Plan, edit, test, and repair a software project."""
    settings, memory, tools, client = _stack(project)
    ok = run_goal(settings, memory, tools, client, goal, tests)
    if not ok:
        raise typer.Exit(code=1)


@app.command()
def improve(goal: str):
    """Improve Valta's own code. Root rules and stored memory stay protected."""
    root = load_settings().root
    settings, memory, tools, client = _stack(root)
    ok = run_goal(settings, memory, tools, client, goal, "python -m pytest -q")
    if not ok:
        raise typer.Exit(code=1)


@app.command()
def research():
    """Run one research cycle, capped by research_page_cap."""
    settings, memory, tools, client = _stack()
    filed = research_once(settings, memory, tools, client)
    typer.echo(f"Filed {filed} notes.")


@app.command("memory")
def memory_cmd(limit: int = 20):
    """List what Valta has stored."""
    settings = load_settings()
    memory = Memory(settings)
    rows = memory.recent(limit)
    if not rows:
        typer.echo("Memory is empty.")
        return
    for record in rows:
        title = record.title or record.kind
        typer.echo(f"{record.created_at}  {record.kind:10}  {record.id[:8]}  {title}")


@app.command()
def forget(record_id: str):
    """Delete one memory and its vector. Use the id prefix shown by `valta memory`."""
    settings, memory, tools, _client = _stack()
    matches = memory.find_ids(record_id)
    if len(matches) != 1:
        raise typer.BadParameter("Give one id from `valta memory`.")
    if tools.forget(matches[0]):
        typer.echo(f"Forgot {matches[0]}")
    else:
        raise typer.BadParameter("That record is already gone.")


@app.command()
def rules():
    """Print the root rules loaded on every model call."""
    settings = load_settings()
    typer.echo(load_rules(settings))


@app.command()
def doctor():
    """Check both Ollama servers and where models are loaded."""
    settings = load_settings()
    client = OllamaClient(settings)
    load_rules(settings)
    typer.echo(f"Root rules: {settings.rules_path}")
    for label, host in (("RX 6600", settings.heavy_host), ("GTX 1660", settings.self_host)):
        typer.echo(f"\n{label}  {host}")
        if not client.healthy(host):
            typer.echo("  down. Run scripts/start-ollama.ps1 after quitting the Ollama tray app.")
            continue
        try:
            typer.echo(f"  version {client.version(host)}")
            loaded = client.ps(host)
            if not loaded:
                typer.echo("  no model loaded")
            for item in loaded:
                name = item.get("name", "")
                processor = item.get("processor") or item.get("size_vram") or ""
                typer.echo(f"  loaded {name}  {processor}")
            names = ", ".join(item.get("name", "") for item in client.tags(host)) or "(none pulled)"
            typer.echo(f"  installed: {names}")
        except OllamaError as exc:
            typer.echo(f"  {exc}")


@models_app.command("pull")
def models_pull():
    """Pull the teacher, the embed model, the personal base, and any extra Hugging Face tags."""
    settings = load_settings()
    client = OllamaClient(settings)
    if not client.healthy(settings.heavy_host):
        raise typer.BadParameter("The RX 6600 server is down. Run scripts/start-ollama.ps1.")
    for model in (settings.teacher_model, settings.embed_model, *settings.extra_models):
        typer.echo(f"Pulling {model} on {settings.heavy_host}")
        client.pull(settings.heavy_host, model)
    if client.healthy(settings.self_host):
        typer.echo(f"Pulling {settings.self_model} on {settings.self_host}")
        client.pull(settings.self_host, settings.self_model)
    else:
        typer.echo("The GTX 1660 server is down, so the 3B was not pulled.")


@app.command()
def train(
    rollback_model: bool = typer.Option(False, "--rollback", help="Restore the previous valta Modelfile."),
    create_only: bool = typer.Option(False, "--create-only", help="Register an existing GGUF as valta."),
):
    """Fine-tune the 3B on the GTX 1660, then `ollama create valta`."""
    settings, memory, _tools, client = _stack()
    if rollback_model:
        rollback(settings, client)
        return
    if create_only:
        gguf = os.environ.get("VALTA_GGUF")
        if not gguf:
            raise typer.BadParameter("Set VALTA_GGUF to the .gguf file.")
        modelfile = write_modelfile(settings, Path(gguf))
        create_valta(settings, client, modelfile)
        return
    waiting = memory.example_count()
    if waiting < settings.train_threshold:
        typer.echo(
            f"{waiting} correction(s) queued. The threshold is {settings.train_threshold}. Training with what exists."
        )
    try:
        dataset, example_ids = build_pairs(settings, memory, client)
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc)) from exc
    free_training_gpu(settings, client)
    adapter = run_qlora(settings, dataset)
    merged = merge_adapter(settings, adapter)
    gguf_path = settings.models_dir / f"{adapter.name}.gguf"
    gguf = convert_to_gguf(merged, gguf_path)
    modelfile = write_modelfile(settings, gguf)
    create_valta(settings, client, modelfile)
    memory.mark_examples_used(example_ids)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    app()
