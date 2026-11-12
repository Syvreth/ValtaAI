"""Import conversations from a public Hugging Face dataset, such as BAAI/Infinity-Instruct.

The dataset is streamed, so nothing close to its full size is downloaded: records are read one at a time,
filtered, converted to Valta's message format, and stored in the `external_examples` table until `limit`
are kept. `valta train` then mixes in at most `external_max_pairs` of them, so a public dataset cannot
drown out your corrections and verified work.

Records are expected in the ShareGPT shape Infinity-Instruct uses:
    {"id": ..., "conversations": [{"from": "human", "value": ...}, {"from": "gpt", "value": ...}],
     "langdetect": "en", "label": {"ability_en": [...], "cate_ability_en": [...]}, ...}
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Iterable

from valta.config import Settings
from valta.evals import contaminated, eval_prompts
from valta.memory import Memory

ROLE_NAMES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "model": "assistant",
}
SKIPPED_ROLES = {"system"}  # the root rules are the system message of every training pair
CHARS_PER_TOKEN = 3  # a cautious estimate; tokenize_example measures exactly at training time
RULES_TOKEN_ALLOWANCE = 200


class DatasetAccessError(RuntimeError):
    pass


@dataclass
class ImportStats:
    seen: int = 0
    kept: int = 0
    duplicate: int = 0
    skipped: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def summary(self) -> str:
        reasons = ", ".join(f"{reason} {count}" for reason, count in sorted(self.skipped.items())) or "none"
        return f"Read {self.seen}, kept {self.kept}, already imported {self.duplicate}. Skipped: {reasons}."


def to_messages(turns) -> list[dict] | None:
    """ShareGPT turns as user/assistant messages, or None when they are not a usable conversation.

    A usable conversation alternates user and assistant, starts with the user, and ends with an
    assistant answer, which is the part training learns from.
    """
    if not isinstance(turns, list):
        return None
    messages: list[dict] = []
    for turn in turns:
        if not isinstance(turn, dict):
            return None
        speaker = str(turn.get("from") or turn.get("role") or "").strip().lower()
        text = turn.get("value", turn.get("content"))
        if speaker in SKIPPED_ROLES:
            continue
        role = ROLE_NAMES.get(speaker)
        if role is None or not isinstance(text, str) or not text.strip():
            return None
        expected = "user" if len(messages) % 2 == 0 else "assistant"
        if role != expected:
            return None
        messages.append({"role": role, "content": text.strip()})
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        return None
    return messages


def label_words(label) -> list[str]:
    """Every string inside a record's `label`, lowercased, whatever its nesting."""
    if isinstance(label, str):
        return [label.lower()]
    if isinstance(label, dict):
        return [word for value in label.values() for word in label_words(value)]
    if isinstance(label, (list, tuple)):
        return [word for value in label for word in label_words(value)]
    return []


def matches_abilities(record: dict, abilities: list[str]) -> bool:
    if not abilities:
        return True
    words = label_words(record.get("label"))
    return any(ability.lower() in word for ability in abilities for word in words)


def record_id(record: dict) -> str:
    raw = record.get("id")
    if raw not in (None, ""):
        return str(raw)
    blob = json.dumps(record.get("conversations"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def stream_dataset(name: str, config: str | None, split: str) -> Iterable[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DatasetAccessError('The datasets package is missing. Run: pip install -e ".[train]"') from exc
    try:
        return load_dataset(name, config, split=split, streaming=True)
    except Exception as exc:
        raise _access_error(name, exc) from exc


def _access_error(name: str, exc: Exception) -> DatasetAccessError:
    text = str(exc)
    lowered = text.lower()
    if any(cue in lowered for cue in ("gated", "401", "403", "unauthorized", "authenticated", "access to dataset")):
        return DatasetAccessError(
            f"{name} needs you to accept its terms and sign in. Open https://huggingface.co/datasets/{name}, "
            "log in, accept the conditions, then run `.\\.venv\\Scripts\\hf.exe auth login` and paste your "
            f"own access token.\n{text[:300]}"
        )
    return DatasetAccessError(f"Could not open {name}: {text[:400]}")


def import_dataset(
    settings: Settings,
    memory: Memory,
    name: str,
    config: str | None = None,
    split: str = "train",
    lang: str | None = "en",
    abilities: list[str] | None = None,
    limit: int = 2000,
    scan_limit: int | None = None,
    records: Iterable[dict] | None = None,
    on_progress: Callable[[ImportStats], None] | None = None,
) -> ImportStats:
    """Keep up to `limit` conversations that pass the filters. `records` replaces streaming, for tests."""
    from valta.train import conflicts_with_rules

    source = records if records is not None else stream_dataset(name, config, split)
    dataset_key = f"{name}:{config}" if config else name
    blocked = eval_prompts(settings)
    max_chars = max(settings.train_max_len - RULES_TOKEN_ALLOWANCE, 64) * CHARS_PER_TOKEN
    stats = ImportStats()
    try:
        for record in source:
            if stats.kept >= limit or (scan_limit is not None and stats.seen >= scan_limit):
                break
            stats.seen += 1
            if on_progress is not None and stats.seen % 500 == 0:
                on_progress(stats)
            if lang and str(record.get("langdetect") or "").lower() != lang.lower():
                stats.skip("language")
                continue
            if not matches_abilities(record, abilities or []):
                stats.skip("ability")
                continue
            messages = to_messages(record.get("conversations"))
            if messages is None:
                stats.skip("format")
                continue
            if sum(len(message["content"]) for message in messages) > max_chars:
                stats.skip("too long")
                continue
            texts = [message["content"] for message in messages]
            if any(conflicts_with_rules(text) for text in texts):
                stats.skip("rule conflict")
                continue
            if any(contaminated(message["content"], blocked) for message in messages if message["role"] == "user"):
                stats.skip("eval overlap")
                continue
            if memory.add_external(dataset_key, record_id(record), messages):
                stats.kept += 1
            else:
                stats.duplicate += 1
    except DatasetAccessError:
        raise
    except Exception as exc:
        if records is None:
            raise _access_error(name, exc) from exc
        raise
    return stats


def peek_dataset(name: str, config: str | None, split: str = "train", count: int = 5) -> list[dict]:
    """The first few records, cut down to what helps choose filters: language, labels, source, first question."""
    shown = []
    for record in stream_dataset(name, config, split):
        turns = record.get("conversations") or []
        first = next((turn.get("value", "") for turn in turns if isinstance(turn, dict)), "")
        shown.append(
            {
                "id": record.get("id"),
                "langdetect": record.get("langdetect"),
                "source": record.get("source"),
                "label": record.get("label"),
                "turns": len(turns),
                "first": str(first)[:160],
            }
        )
        if len(shown) >= count:
            break
    return shown
