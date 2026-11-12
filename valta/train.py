from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from valta.config import Settings, load_state, save_state
from valta.evals import EvalResult, contaminated, eval_prompts, run_eval, should_promote
from valta.memory import Memory
from valta.ollama_client import OllamaClient, OllamaError
from valta.prompt import build_system, load_rules

CANDIDATE_NAME = "valta-candidate"

CONFLICT_PHRASES = (
    "ignore the root rules",
    "edit the root rules",
    "delete the root rules",
    "hide the root rules",
    "forget everything",
    "delete all memory",
)


def conflicts_with_rules(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in CONFLICT_PHRASES)


def build_pairs(settings: Settings, memory: Memory, client: OllamaClient | None = None) -> tuple[Path, list[str]]:
    """Training text is question-answer pairs. Raw ebook text is not the corpus.

    Every fine-tune starts again from the base model, so the dataset is cumulative: all corrections and
    all verified solutions ever saved, not only the new ones. Pairs that contain an eval prompt are dropped,
    so the eval stays a fair test.
    """
    rules = load_rules(settings)
    blocked = eval_prompts(settings)
    pairs: list[dict] = []
    example_ids: list[str] = []
    seen: set[tuple[str, str]] = set()
    dropped = 0

    def add(user_text: str, assistant_text: str) -> bool:
        nonlocal dropped
        key = (user_text.strip(), assistant_text.strip())
        if not key[0] or not key[1] or key in seen:
            return False
        if conflicts_with_rules(user_text) or conflicts_with_rules(assistant_text) or contaminated(user_text, blocked):
            dropped += 1
            return False
        seen.add(key)
        pairs.append(_pair(rules, user_text, assistant_text))
        return True

    for row in memory.all_examples():
        add(row["user_text"], row["assistant_text"])
        example_ids.append(row["id"])

    for title, text in research_pairs(memory.recent(limit=200, kind="research")):
        add(f"What did you learn about {title}?", text)

    if client and client.healthy(settings.heavy_host):
        for record in memory.recent(limit=20, kind="book"):
            drafted = _qa_from_passage(settings, client, record.text)
            if drafted:
                add(drafted[0], drafted[1])

    if not pairs:
        raise RuntimeError("Nothing to train on yet. Ingest notes, research, or correct a reply first.")

    settings.datasets_dir.mkdir(parents=True, exist_ok=True)
    path = settings.datasets_dir / "train.jsonl"

    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    note = f" ({dropped} dropped for rule conflicts or eval overlap)" if dropped else ""
    print(f"Wrote {len(pairs)} training pairs to {path}{note}")
    return path, example_ids


# Research notes are background reading, not corrections or verified work. A cap keeps them from
# outnumbering the examples that matter.
RESEARCH_MAX_PAIRS = 20
_NOTE_LABEL = re.compile(
    r"^\s*(?:\*\*)?\s*note(?:\s+for\s+companion)?\s*(?::\s*\*\*|\*\*\s*:|:)\s*", re.IGNORECASE
)


def clean_note(text: str) -> str:
    """Drop a leading "**Note:**" style label, so the model does not learn to open every answer with one."""
    return _NOTE_LABEL.sub("", text, count=1).strip()


def research_pairs(records) -> list[tuple[str, str]]:
    """One (title, text) per research topic, newest note first, at most RESEARCH_MAX_PAIRS.

    Research runs revisit the same topic, and five different answers to one question teach the
    model nothing but the shape of an answer. `records` is expected newest first.
    """
    seen: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for record in records:
        title = " ".join((record.title or "").split())
        key = title.lower()
        text = clean_note(record.text)
        if not title or not text or key in seen:
            continue
        seen.add(key)
        pairs.append((title, text))
        if len(pairs) >= RESEARCH_MAX_PAIRS:
            break
    return pairs


def _pair(rules: str, user_text: str, assistant_text: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": rules},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    }


def _qa_from_passage(settings: Settings, client: OllamaClient, passage: str) -> tuple[str, str] | None:
    system = build_system(settings, role="Turn a passage into one question and one answer.")
    user = (
        "Write one question a person might ask, then an answer in your own words. "
        "Format exactly:\nQ: ...\nA: ...\n"
        "If the passage conflicts with the root rules, reply SKIP.\n\n"
        f"Passage:\n{passage}"
    )
    try:
        reply = client.chat(
            settings.heavy_host,
            settings.teacher_model,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            settings.teacher_num_ctx,
        )
    except OllamaError:
        return None
    if reply.strip().upper().startswith("SKIP") or "\nA:" not in reply and "A:" not in reply:
        return None
    question, _, answer = reply.partition("A:")
    question = question.replace("Q:", "", 1).strip()
    answer = answer.strip()
    if not question or not answer:
        return None
    return question, answer


def render_text(messages: list[dict]) -> str:
    lines = []
    for message in messages:
        lines.append(f"{message['role'].upper()}:\n{message['content']}")
    return "\n\n".join(lines)


def tokenize_example(tokenizer, messages: list[dict], max_len: int) -> dict | None:
    """Tokenize one conversation with the model's own chat template, training only on the final answer.

    The system and user turns are masked (-100), so the loss is not spent on the root rules that open every
    pair. Returns None when the example is longer than max_len: a half-cut answer would teach the model to
    never stop, so it is skipped instead.
    """
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    if not full_text.startswith(prompt_text):
        # A template that does not nest. Train on the whole text rather than guess where the answer starts.
        prompt_text = ""
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"] if prompt_text else []
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_len or len(prompt_ids) >= len(full_ids):
        return None
    labels = [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids):])
    return {"input_ids": list(full_ids), "attention_mask": [1] * len(full_ids), "labels": labels}


def run_qlora(settings: Settings, dataset_path: Path) -> Path:
    """QLoRA on the GTX 1660. 4-bit NF4, batch size 1, fp16, short context."""
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install a CUDA build of PyTorch first "
            "(see pytorch.org for the current command), then:\n"
            "  pip install -e \".[train]\""
        ) from exc

    if not torch.cuda.is_available():
        hint = ""
        if getattr(torch.version, "cuda", None) is None:
            hint = (
                " This PyTorch is a CPU-only build. Reinstall a CUDA build "
                "(see pytorch.org for the current command), then run this again."
            )
        raise RuntimeError("CUDA is not available. QLoRA runs on the GTX 1660." + hint)

    from datasets import Dataset

    compute = torch.float16
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute,
    )
    tokenizer = AutoTokenizer.from_pretrained(settings.hf_base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = []
    skipped = 0
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = tokenize_example(tokenizer, json.loads(line)["messages"], settings.train_max_len)
        if item is None:
            skipped += 1
        else:
            rows.append(item)
    if skipped:
        print(f"Skipped {skipped} example(s) longer than train_max_len={settings.train_max_len} tokens.")
    if not rows:
        raise RuntimeError(
            f"Every example is longer than train_max_len={settings.train_max_len} tokens. "
            "Raise it in config.toml, or shorten the data. A 6 GB card cannot take much more than 2048."
        )
    print(f"Training on {len(rows)} example(s).")
    dataset = Dataset.from_list(rows)

    model = AutoModelForCausalLM.from_pretrained(
        settings.hf_base,
        quantization_config=quant,
        device_map={"": 0},
        torch_dtype=compute,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )

    stamp = datetime.now(timezone.utc).strftime(STAMP_FORMAT)
    output = settings.adapters_dir / stamp
    output.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(output),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True,
        bf16=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8),
    )
    trainer.train()
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    del model
    del trainer
    torch.cuda.empty_cache()
    print(f"Saved adapter to {output}")
    return output


def merge_adapter(settings: Settings, adapter_dir: Path) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged = settings.merged_dir / adapter_dir.name
    if merged.exists():
        shutil.rmtree(merged)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base = AutoModelForCausalLM.from_pretrained(
        settings.hf_base,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    peft_model = PeftModel.from_pretrained(base, adapter_dir)
    merged_model = peft_model.merge_and_unload()
    merged.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(merged)
    tokenizer.save_pretrained(merged)
    print(f"Merged weights to {merged}")
    return merged


def find_converter() -> str | None:
    """llama.cpp's convert_hf_to_gguf.py: LLAMA_CPP_CONVERT, then PATH, then a clone in %LOCALAPPDATA%\\llama.cpp."""
    found = os.environ.get("LLAMA_CPP_CONVERT") or shutil.which("convert_hf_to_gguf.py")
    if found:
        return found
    # Some terminals start without LOCALAPPDATA, so the home folder's AppData\Local is tried as well.
    roots = [os.environ.get("LOCALAPPDATA"), str(Path.home() / "AppData" / "Local")]
    for root in roots:
        if root:
            default = Path(root) / "llama.cpp" / "convert_hf_to_gguf.py"
            if default.is_file():
                return str(default)
    return None


def convert_to_gguf(merged_dir: Path, gguf_path: Path) -> Path:
    convert = find_converter()
    if not convert:
        raise RuntimeError(
            "The adapter is saved, and the merged weights are in data/merged. "
            "Clone llama.cpp into %LOCALAPPDATA%\\llama.cpp (or set LLAMA_CPP_CONVERT to its "
            "convert_hf_to_gguf.py), then run `valta train --resume` to convert, score and promote "
            "without training again."
        )
    import subprocess
    import sys

    gguf_path.parent.mkdir(parents=True, exist_ok=True)
    # sys.executable, not "python": the converter needs this venv's torch and transformers, and a bare
    # "python" can be a different install entirely. The converter cannot write Q4_K_M (its --outtype
    # choices are f32, f16, bf16, q8_0, tq1_0, tq2_0, auto); Q8_0 keeps a 3B near full quality and
    # still fits a 6 GB card.
    subprocess.run(
        [sys.executable, convert, str(merged_dir), "--outfile", str(gguf_path), "--outtype", GGUF_OUTTYPE],
        check=True,
    )
    return gguf_path


GGUF_OUTTYPE = "q8_0"


STAMP_FORMAT = "%Y%m%d%H%M%S"


def latest_merged(settings: Settings) -> Path | None:
    """The newest merged-weights folder. Folder names are UTC timestamps, so the largest name is the newest."""
    if not settings.merged_dir.is_dir():
        return None
    runs = []
    for path in settings.merged_dir.iterdir():
        try:
            datetime.strptime(path.name, STAMP_FORMAT)
        except ValueError:
            continue
        if path.is_dir() and (path / "config.json").is_file():
            runs.append(path)
    return max(runs, key=lambda path: path.name, default=None)


def examples_before(memory: Memory, stamp: str) -> list[str]:
    """Ids of the examples saved before a run's timestamp: the ones that run trained on."""
    cutoff = datetime.strptime(stamp, STAMP_FORMAT).replace(tzinfo=timezone.utc)
    return [
        row["id"]
        for row in memory.all_examples()
        if datetime.fromisoformat(row["created_at"]) < cutoff
    ]


def modelfile_text(settings: Settings, gguf_path: Path) -> str:
    return f"FROM {gguf_path.as_posix()}\nPARAMETER num_ctx {settings.self_num_ctx}\n"


def write_modelfile(settings: Settings, gguf_path: Path) -> str:
    current = settings.models_dir / "valta.Modelfile"
    previous = settings.models_dir / "valta.previous.Modelfile"
    if current.exists():
        previous.write_text(current.read_text(encoding="utf-8"), encoding="utf-8")
        state = load_state(settings)
        state["previous_modelfile"] = str(previous)
        save_state(settings, state)
    modelfile = modelfile_text(settings, gguf_path)
    current.write_text(modelfile, encoding="utf-8")
    return modelfile


def create_valta(settings: Settings, client: OllamaClient, modelfile: str) -> None:
    client.create(settings.self_host, "valta", modelfile)
    state = load_state(settings)
    state["personal_model"] = "valta"
    save_state(settings, state)
    print("Personal model is now `valta` on the GTX 1660.")


def baseline_model(settings: Settings) -> str:
    """The model the next fine-tune has to beat: whatever personal model is active now."""
    return load_state(settings).get("personal_model") or settings.self_model


def score_baseline(settings: Settings, client: OllamaClient) -> EvalResult:
    model = baseline_model(settings)
    print(f"Scoring the current personal model ({model}) before training...")
    result = run_eval(settings, client, settings.self_host, model, on_progress=_progress)
    print(result.summary())
    return result


def gate_candidate(
    settings: Settings,
    client: OllamaClient,
    modelfile: str,
    baseline: EvalResult | None,
) -> tuple[bool, EvalResult, str]:
    """Load the new weights under a temporary name, score them, and decide. Nothing is replaced yet."""
    client.create(settings.self_host, CANDIDATE_NAME, modelfile)
    try:
        print("Scoring the new model...")
        candidate = run_eval(settings, client, settings.self_host, CANDIDATE_NAME, on_progress=_progress)
    finally:
        try:
            client.delete(settings.self_host, CANDIDATE_NAME)
        except OllamaError:
            pass
    print(candidate.summary())
    promote, reason = should_promote(baseline, candidate, settings.eval_tolerance)
    return promote, candidate, reason


def record_eval(
    settings: Settings,
    adapter: str,
    baseline: EvalResult | None,
    candidate: EvalResult,
    promoted: bool,
) -> None:
    state = load_state(settings)
    history = list(state.get("eval_history", []))
    history.append(
        {
            "adapter": adapter,
            "time": datetime.now(timezone.utc).isoformat(),
            "baseline_model": baseline.model if baseline else None,
            "baseline": baseline.score if baseline else None,
            "candidate": candidate.score,
            "promoted": promoted,
        }
    )
    state["eval_history"] = history[-50:]
    save_state(settings, state)


def _progress(task_id: str, ok: bool, note: str) -> None:
    print(f"  {'pass' if ok else 'FAIL'}  {task_id}" + (f"  ({note})" if note and not ok else ""), flush=True)


def rollback(settings: Settings, client: OllamaClient) -> None:
    previous = settings.models_dir / "valta.previous.Modelfile"
    if not previous.exists():
        # The first promotion has nothing before it but the base model, so go back to that.
        state = load_state(settings)
        if state.get("personal_model") == settings.self_model:
            raise RuntimeError(f"The personal model is already the base model, {settings.self_model}.")
        state["personal_model"] = settings.self_model
        save_state(settings, state)
        print(f"Personal model is back to the base model, {settings.self_model}. The `valta` model is kept in Ollama.")
        return
    modelfile = previous.read_text(encoding="utf-8")
    client.create(settings.self_host, "valta", modelfile)
    current = settings.models_dir / "valta.Modelfile"
    current.write_text(modelfile, encoding="utf-8")
    print("Rolled `valta` back to the previous Modelfile.")


def free_training_gpu(settings: Settings, client: OllamaClient) -> None:
    """Unload whatever is sitting on the 1660 so QLoRA can use the card."""
    if not client.healthy(settings.self_host):
        print("The GTX 1660 Ollama server is not running. Training can use the card.")
        return
    names = [item.get("name", "") for item in client.ps(settings.self_host)]
    state = load_state(settings)
    names.append(state.get("personal_model") or settings.self_model)
    names.append("valta")
    names.append(CANDIDATE_NAME)
    seen = set()
    for name in names:
        short = name.split(":")[0] if name else ""
        if not name or name in seen:
            continue
        seen.add(name)
        try:
            client.unload(settings.self_host, name)
            print(f"Unloaded {name} from the GTX 1660.")
        except OllamaError:
            if short and short not in seen:
                seen.add(short)
                try:
                    client.unload(settings.self_host, short)
                except OllamaError:
                    pass
