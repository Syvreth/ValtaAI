from __future__ import annotations

from valta.config import Settings


def load_rules(settings: Settings) -> str:
    path = settings.rules_path
    if not path.exists():
        raise FileNotFoundError(f"Root rules are missing: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise FileNotFoundError(f"Root rules are empty: {path}")
    return text


def build_system(settings: Settings, role: str = "", memory_blocks: list[str] | None = None) -> str:
    """Root rules come first. A role and retrieved memory cannot replace them."""
    parts = [load_rules(settings)]
    if role.strip():
        parts.append("Role:\n" + role.strip())
    blocks = [block.strip() for block in (memory_blocks or []) if block and block.strip()]
    if blocks:
        parts.append("Memory:\n" + "\n\n".join(blocks))
    return "\n\n".join(parts)
