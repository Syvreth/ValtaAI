"""Import conversations from a public Hugging Face dataset, such as BAAI/Infinity-Instruct.

The dataset is streamed, so nothing close to its full size is downloaded: records are read one at a time,
filtered, converted to Valta's message format, and stored in the `external_examples` table until `limit`
are kept. `valta train` then mixes in at most `external_max_pairs` of them, so a public dataset cannot
drown out your corrections and verified work.

Records may come in any of the common layouts (see record_messages):
    ShareGPT, as Infinity-Instruct and KodCode use:
        {"id": ..., "conversations": [{"from": "human", "value": ...}, {"from": "gpt", "value": ...}],
         "langdetect": "en", "label": {"ability_en": [...], "cate_ability_en": [...]}, ...}
    chat messages, as tulu-3 uses: {"messages": [{"role": "user", "content": ...}, ...], "source": ...}
    one question column and one answer column: instruction/response, instruction + input/output,
        query/answer, problem/solution, question/answer, prompt/completion.
"""

from __future__ import annotations

import hashlib
import json
import re
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
# For datasets with one question and one answer per record. The first column present with text wins.
QUESTION_COLUMNS = ("instruction", "query", "problem", "question", "prompt")
ANSWER_COLUMNS = ("response", "output", "answer", "solution", "completion")
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


def record_messages(record: dict) -> list[dict] | None:
    """A record as user and assistant messages, whatever common layout its dataset uses, or None.

    For a question-and-answer record, an `input` column (the code to change, in InstructCoder) is added under
    the question, and code without a fence gets one, so every answer shows code the way chat replies do.
    """
    if record.get("conversations") is not None:
        return to_messages(record["conversations"])
    if record.get("messages") is not None:
        return to_messages(record["messages"])
    question = _first_text(record, QUESTION_COLUMNS)
    answer = _first_text(record, ANSWER_COLUMNS)
    if question is None or answer is None:
        return None
    extra = record.get("input")
    if isinstance(extra, str) and extra.strip():
        question = f"{question.strip()}\n\n{_fenced(extra)}"
    return to_messages([{"role": "user", "content": question}, {"role": "assistant", "content": _fenced(answer)}])


def _first_text(record: dict, columns: tuple[str, ...]) -> str | None:
    return next((record[column] for column in columns if isinstance(record.get(column), str) and record[column].strip()), None)


def _fenced(text: str) -> str:
    """Bare code in a fence. Text that already has a fence, or has no code, is returned as it is."""
    stripped = text.strip()
    if "```" in stripped or not has_code(stripped):
        return stripped
    return f"```\n{stripped}\n```"


_FENCE = re.compile(r"```.*?(```|$)", re.S)
_CODE_HINT = re.compile(r"^\s|[=;{}]|[(\[,)\]]\s*$")  # indented, an assignment or brace, or ends like code
_IDENTIFIER = re.compile(r"`[^`]*`|\S*[_./\\]\S*")  # `inline code`, snake_case, dotted.names and paths
# Frequent English words that are rare as words in other languages ("a" and "i" are common in Spanish and Catalan).
_COMMON_ENGLISH = {"the", "and", "is", "to", "of", "in", "you", "it", "that", "for", "with", "this", "can", "are", "be"}


def looks_english(text: str) -> bool:
    """A guess from the prose, for records with no `langdetect` field. Fenced code and lines that look like
    code are set aside first. Too little prose to judge counts as English."""
    lines = _FENCE.sub(" ", text).splitlines()
    prose = _IDENTIFIER.sub(" ", " ".join(line for line in lines if not (_CODE_HINT.search(line) or _CODE_LINE.match(line))))
    letters = [char for char in prose if char.isalpha()]
    if len(letters) >= 40 and sum(char.isascii() for char in letters) / len(letters) < 0.9:
        return False
    words = re.findall(r"[^\W\d_]+", prose.lower())  # whole words, accents included: "año" is not "a" + "o"
    return len(words) < 12 or sum(word in _COMMON_ENGLISH for word in words) / len(words) >= 0.08


def matches_where(record: dict, conditions: list[str]) -> bool:
    """True with no conditions, or when any COLUMN=TEXT condition finds TEXT in that column, ignoring case."""
    if not conditions:
        return True
    for condition in conditions:
        column, _, text = condition.partition("=")
        value = record.get(column.strip())
        if value is not None and text.strip().lower() in str(value).lower():
            return True
    return False


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


_CODE_LINE = re.compile(r"^\s*(?:def|class|import|from\s+\S+\s+import|function|const|let|#include|public|fn)\b", re.M)


def has_code(text: str) -> bool:
    """A fenced code block, or a line that starts like code in a common language."""
    return "```" in text or bool(_CODE_LINE.search(text))


def record_id(record: dict) -> str:
    raw = record.get("id")
    if raw not in (None, ""):
        return str(raw)
    if record.get("conversations") is not None:  # how ShareGPT records without an id have always been keyed
        blob = json.dumps(record.get("conversations"), sort_keys=True, ensure_ascii=False)
    else:
        blob = json.dumps(record, sort_keys=True, ensure_ascii=False, default=str)
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
    min_answer: int = 0,
    code_only: bool = False,
    where: list[str] | None = None,
) -> ImportStats:
    """Keep up to `limit` conversations that pass the filters. `records` replaces streaming, for tests.

    `lang` is checked against a record's `langdetect` field. A record without one is kept, except that for
    "en" its prose has to look English (see looks_english), since many datasets mix languages. `min_answer`
    skips conversations whose final answer is shorter than that many characters, such as one-line "describe
    this function" replies. `code_only` keeps only final answers that contain code. `where` keeps records
    where any COLUMN=TEXT condition holds, such as lang=python or source=wildchat.
    """
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
            detected = record.get("langdetect")
            if lang and detected is not None and str(detected).lower() != lang.lower():
                stats.skip("language")
                continue
            if not matches_abilities(record, abilities or []):
                stats.skip("ability")
                continue
            if not matches_where(record, where or []):
                stats.skip("where")
                continue
            messages = record_messages(record)
            if messages is None:
                stats.skip("format")
                continue
            if lang and lang.lower() == "en" and detected is None and not looks_english(
                "\n".join(message["content"] for message in messages)
            ):
                stats.skip("language")
                continue
            if sum(len(message["content"]) for message in messages) > max_chars:
                stats.skip("too long")
                continue
            answer = messages[-1]["content"]
            if min_answer and len(answer) < min_answer:
                stats.skip("short answer")
                continue
            if code_only and not has_code(answer):
                stats.skip("no code")
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
    """The first few records, cut down to what helps choose filters: columns, language, labels, source, and
    the first question as Valta reads it. `turns` is 0 when Valta cannot read the record's layout."""
    shown = []
    for record in stream_dataset(name, config, split):
        messages = record_messages(record) or []
        shown.append(
            {
                "id": record.get("id"),
                "columns": sorted(record),
                "langdetect": record.get("langdetect"),
                "source": record.get("source"),
                "label": record.get("label"),
                "turns": len(messages),
                "first": messages[0]["content"][:160] if messages else "",
            }
        )
        if len(shown) >= count:
            break
    return shown
