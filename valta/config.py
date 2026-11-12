from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass
class Settings:
    root: Path
    heavy_host: str
    self_host: str
    teacher_model: str
    self_model: str
    embed_model: str
    hf_base: str
    teacher_num_ctx: int
    self_num_ctx: int
    research_interval_sec: int
    research_page_cap: int
    train_threshold: int
    chunk_chars: int
    chunk_overlap: int
    top_k: int
    work_max_steps: int
    repair_rounds: int
    extra_models: list[str]
    train_max_len: int = 1024
    eval_tolerance: float = 0.0
    ui_theme: str = "tokyo-night"
    external_max_pairs: int = 2000

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def library(self) -> Path:
        return self.data / "library"

    @property
    def rules_path(self) -> Path:
        return self.root / "rules" / "root.md"

    @property
    def db_path(self) -> Path:
        return self.data / "valta.db"

    @property
    def chroma_path(self) -> Path:
        return self.data / "chroma"

    @property
    def state_path(self) -> Path:
        return self.data / "state.json"

    @property
    def interests_path(self) -> Path:
        return self.data / "interests.txt"

    @property
    def models_dir(self) -> Path:
        return self.data / "models"

    @property
    def adapters_dir(self) -> Path:
        return self.data / "adapters"

    @property
    def merged_dir(self) -> Path:
        return self.data / "merged"

    @property
    def datasets_dir(self) -> Path:
        return self.data / "datasets"

    @property
    def evals_dir(self) -> Path:
        return self.data / "evals"

    def ensure_dirs(self) -> None:
        for path in (
            self.library,
            self.chroma_path,
            self.models_dir,
            self.adapters_dir,
            self.merged_dir,
            self.datasets_dir,
            self.evals_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        if not self.interests_path.exists():
            self.interests_path.write_text(
                "# One research topic per line. Lines starting with # are ignored.\n",
                encoding="utf-8",
            )


def load_settings(root: Path | None = None) -> Settings:
    root = root or project_root()
    path = root / "config.toml"
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    settings = Settings(
        root=root,
        heavy_host=raw["heavy_host"],
        self_host=raw["self_host"],
        teacher_model=raw["teacher_model"],
        self_model=raw["self_model"],
        embed_model=raw["embed_model"],
        hf_base=raw["hf_base"],
        teacher_num_ctx=int(raw["teacher_num_ctx"]),
        self_num_ctx=int(raw["self_num_ctx"]),
        research_interval_sec=int(raw["research_interval_sec"]),
        research_page_cap=int(raw["research_page_cap"]),
        train_threshold=int(raw["train_threshold"]),
        chunk_chars=int(raw["chunk_chars"]),
        chunk_overlap=int(raw["chunk_overlap"]),
        top_k=int(raw["top_k"]),
        work_max_steps=int(raw.get("work_max_steps", 8)),
        repair_rounds=int(raw.get("repair_rounds", 2)),
        extra_models=list(raw.get("extra_models", [])),
        train_max_len=int(raw.get("train_max_len", 1024)),
        eval_tolerance=float(raw.get("eval_tolerance", 0.0)),
        ui_theme=str(raw.get("ui_theme", "tokyo-night")),
        external_max_pairs=int(raw.get("external_max_pairs", 2000)),
    )
    settings.ensure_dirs()
    return settings


def load_state(settings: Settings) -> dict:
    if not settings.state_path.exists():
        return {"personal_model": settings.self_model, "previous_modelfile": None}
    return json.loads(settings.state_path.read_text(encoding="utf-8"))


def save_state(settings: Settings, state: dict) -> None:
    settings.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
