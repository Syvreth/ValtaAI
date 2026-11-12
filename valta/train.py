from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from valta.config import Settings, load_state, save_state
from valta.evals import EvalResult, contaminated, eval_prompts, load_tasks, run_eval, should_promote
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


def build_pairs(
    settings: Settings,
    memory: Memory,
    client: OllamaClient | None = None,
    max_examples: int = 0,
) -> tuple[Path, list[str]]:
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

    external = 0
    external_limit = settings.external_max_pairs
    if max_examples > 0:
        # A smaller, faster run trims the imported dataset first; your own examples are always kept.
        external_limit = min(external_limit, max(0, max_examples - len(pairs)))
    for messages in memory.external_sample(external_limit):
        key = (json.dumps(messages, ensure_ascii=False), "")
        user_turns = [message["content"] for message in messages if message["role"] == "user"]
        if key in seen:
            continue
        if any(conflicts_with_rules(message["content"]) for message in messages) or any(
            contaminated(text, blocked) for text in user_turns
        ):
            dropped += 1
            continue
        seen.add(key)
        pairs.append({"messages": [{"role": "system", "content": rules}, *messages]})
        external += 1

    if client and settings.doc_qa_pairs > 0 and client.healthy(settings.heavy_host):
        for record in document_passages(memory, settings.doc_qa_pairs):
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
    from_datasets = f", {external} of them from imported datasets" if external else ""
    print(f"Wrote {len(pairs)} training pairs to {path}{from_datasets}{note}")
    return path, example_ids


# Research notes are background reading, not corrections or verified work. A cap keeps them from
# outnumbering the examples that matter.
RESEARCH_MAX_PAIRS = 20
# "**Note:**", "Note for Valta Companion:", "**Note on Python:**" at the very start of a note.
_NOTE_LABEL = re.compile(
    r"^\s*(?:\*\*)?\s*note(?:\s+(?:for|on|about)\s+[^:*\n]{1,40})?\s*(?::\s*\*\*|\*\*\s*:|:)\s*",
    re.IGNORECASE,
)
# A first line that is only a bold heading with "note" in it, such as "**Python Download Page Note**".
_NOTE_HEADING = re.compile(r"^\s*\*\*[^*\n]{0,60}\bnotes?\b[^*\n]{0,60}\*\*\s*:?[ \t]*(?:\n|$)", re.IGNORECASE)


def clean_note(text: str) -> str:
    """Drop a leading "**Note:**" style label, so the model does not learn to open every answer with one."""
    text = _NOTE_HEADING.sub("", text, count=1)
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


def document_passages(memory: Memory, limit: int):
    """The newest passages from ingested books and learned web pages, at most `limit` in all.

    Each becomes one question and answer written by the teacher at training time, so `limit`
    (doc_qa_pairs) also sets how many teacher calls a run makes before training starts.
    """
    records = memory.recent(limit=limit, kind="book") + memory.recent(limit=limit, kind="web")
    records.sort(key=lambda record: record.created_at, reverse=True)
    return records[:limit]


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


TRAIN_SEED = 42
RUN_DATASET = "dataset.jsonl"


def run_qlora(
    settings: Settings,
    dataset_path: Path | None,
    resume_dir: Path | None = None,
    benchmark_steps: int = 0,
) -> Path:
    """QLoRA on the GTX 1660. 4-bit NF4, batch size 1, fp16, short context.

    Progress is saved every `checkpoint_steps` optimizer steps. `resume_dir` continues an unfinished
    run from its last checkpoint, on the exact dataset that run started with (kept in the run folder),
    so an interrupted night of training is not lost.

    `benchmark_steps` trains only that many steps in a scratch folder, prints seconds per step and
    peak GPU memory, and keeps nothing, to measure a setting before committing a night to it.
    """
    if resume_dir is not None:
        dataset_path = resume_dir / RUN_DATASET
        if not dataset_path.is_file():
            raise RuntimeError(f"{resume_dir} has no {RUN_DATASET}, so it cannot be resumed.")
    if dataset_path is None:
        raise RuntimeError("No training dataset was given.")
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainerCallback,
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
    if settings.train_fp32_embeddings:
        model = prepare_model_for_kbit_training(model)
    else:
        prepare_lean(model)
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

    if benchmark_steps > 0:
        output = settings.adapters_dir / "benchmark"  # not a timestamp, so --resume never picks it up
        shutil.rmtree(output, ignore_errors=True)
        output.mkdir(parents=True, exist_ok=True)
    elif resume_dir is not None:
        output = resume_dir
    else:
        stamp = datetime.now(timezone.utc).strftime(STAMP_FORMAT)
        output = settings.adapters_dir / stamp
        output.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dataset_path, output / RUN_DATASET)
    checkpoint = latest_checkpoint(output) if benchmark_steps <= 0 else None
    if resume_dir is not None:
        print(f"Resuming training from {checkpoint}" if checkpoint else f"No checkpoint in {output}; starting it over.")
    save_steps = max(int(settings.checkpoint_steps), 0) if benchmark_steps <= 0 else 0
    timer = _StepTimer(TrainerCallback)
    args = TrainingArguments(
        max_steps=benchmark_steps if benchmark_steps > 0 else -1,
        output_dir=str(output),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True,
        bf16=False,
        logging_steps=1,
        save_strategy="steps" if save_steps else "no",
        save_steps=save_steps or 500,
        save_total_limit=1,
        seed=TRAIN_SEED,
        data_seed=TRAIN_SEED,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8),
        callbacks=[timer],
    )
    torch.cuda.reset_peak_memory_stats()
    trainer.train(resume_from_checkpoint=str(checkpoint) if checkpoint else None)
    if benchmark_steps > 0:
        total = torch.cuda.get_device_properties(0).total_memory
        print(
            benchmark_report(
                timer.durations,
                torch.cuda.max_memory_allocated(),
                torch.cuda.max_memory_reserved(),
                total,
                settings,
            )
        )
        del model
        del trainer
        torch.cuda.empty_cache()
        shutil.rmtree(output, ignore_errors=True)
        return output
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    del model
    del trainer
    torch.cuda.empty_cache()
    for leftover in output.glob("checkpoint-*"):
        shutil.rmtree(leftover, ignore_errors=True)  # the final adapter is saved; checkpoints were only for resuming
    print(f"Saved adapter to {output}")
    return output


def _StepTimer(base):
    """A Trainer callback (built on transformers' TrainerCallback) that records seconds per optimizer step."""
    import time

    class StepTimer(base):
        def __init__(self) -> None:
            self.durations: list[float] = []
            self._started = 0.0

        def on_step_begin(self, args, state, control, **kwargs):
            self._started = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            self.durations.append(time.perf_counter() - self._started)

    return StepTimer()


def benchmark_report(durations: list[float], peak_allocated: int, peak_reserved: int, total: int, settings) -> str:
    """Seconds per step (the first step, which warms up, is left out) and peak GPU memory."""
    steady = sorted(durations[1:] or durations)
    if not steady:
        return "No steps finished."
    median = steady[len(steady) // 2]
    gib = 1024**3
    lines = [
        f"Benchmark: {len(durations)} step(s), median {median:.1f} s/step (first step {durations[0]:.1f} s, warm-up).",
        f"Peak GPU memory: {peak_allocated / gib:.2f} GB in use, {peak_reserved / gib:.2f} GB reserved, "
        f"of {total / gib:.2f} GB on the card.",
        f"Settings: train_max_len={settings.train_max_len}, train_fp32_embeddings={settings.train_fp32_embeddings}.",
    ]
    if peak_reserved > total * 0.95:
        lines.append(
            "Reserved memory is at the card's limit, so Windows may be spilling into system RAM (Task Manager: "
            "Shared GPU memory). Try a lower train_max_len."
        )
    return "\n".join(lines)


def prepare_lean(model) -> int:
    """Get a 4-bit model ready for LoRA without peft's blanket fp32 upcast.

    peft's prepare_model_for_kbit_training casts every 16-bit weight to fp32, including the frozen
    word-embedding table: about 311M numbers for Qwen2.5-3B, roughly 0.6 GB more on a 6 GB card for
    weights that never train. Here the base model is frozen, only the small 1-D weights (norms and
    biases) go to fp32 for stability, and gradient checkpointing is turned on. Returns how many
    parameters were upcast.
    """
    import torch

    upcast = 0
    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1 and param.dtype in (torch.float16, torch.bfloat16):
            param.data = param.data.float()
            upcast += param.numel()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    return upcast


def latest_checkpoint(run_dir: Path) -> Path | None:
    """The highest-numbered checkpoint-N folder the Trainer saved in a run, if any."""
    found = []
    for path in run_dir.glob("checkpoint-*"):
        suffix = path.name.split("-", 1)[1]
        if path.is_dir() and suffix.isdigit():
            found.append((int(suffix), path))
    return max(found)[1] if found else None


def adapter_finished(run_dir: Path) -> bool:
    return (run_dir / "adapter_config.json").is_file()


def latest_run(settings: Settings) -> Path | None:
    """The newest adapters/<stamp> folder that either finished or can be resumed from a checkpoint.

    A folder with neither (a run stopped before its first checkpoint) has nothing to continue from.
    """
    if not settings.adapters_dir.is_dir():
        return None
    runs = []
    for path in settings.adapters_dir.iterdir():
        try:
            datetime.strptime(path.name, STAMP_FORMAT)
        except ValueError:
            continue
        if path.is_dir() and (adapter_finished(path) or latest_checkpoint(path)):
            runs.append(path)
    return max(runs, key=lambda path: path.name, default=None)


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
    """The current personal model's eval score. Reused when neither the model's weights nor the eval
    tasks changed since it was last scored, which saves about ten minutes on the GTX 1660."""
    model = baseline_model(settings)
    fingerprint = baseline_fingerprint(settings, client, model)
    cached = load_state(settings).get("baseline_cache") or {}
    if fingerprint and cached.get("fingerprint") == fingerprint:
        result = EvalResult(model, int(cached["passed"]), int(cached["total"]))
        print(f"Reusing the score of {model} from {cached.get('time', 'earlier')}: {result.summary()}")
        return result
    print(f"Scoring the current personal model ({model}) before training...")
    result = run_eval(settings, client, settings.self_host, model, on_progress=_progress)
    print(result.summary())
    _save_baseline(settings, fingerprint, result)
    return result


def remember_baseline(settings: Settings, client: OllamaClient, model: str, result: EvalResult) -> None:
    """Record a score for `model` as it is now, so the next run need not score it again."""
    try:
        _save_baseline(settings, baseline_fingerprint(settings, client, model), result)
    except Exception:
        pass  # only a shortcut; the next run scores the model instead


def baseline_fingerprint(settings: Settings, client: OllamaClient, model: str) -> str | None:
    """The model's Ollama digest plus a hash of the eval tasks. None when the digest is unknown."""
    try:
        tags = client.tags(settings.self_host)
    except Exception:
        return None
    names = {model, f"{model}:latest"}
    digest = next(
        (tag.get("digest") for tag in tags if tag.get("name") in names or tag.get("model") in names),
        None,
    )
    if not digest:
        return None
    tasks = [(task.id, task.prompt, task.tests) for task in load_tasks(settings)]
    task_hash = hashlib.sha256(json.dumps(tasks, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    return f"{model}|{digest}|{task_hash}"


def _save_baseline(settings: Settings, fingerprint: str | None, result: EvalResult) -> None:
    if not fingerprint:
        return
    state = load_state(settings)
    state["baseline_cache"] = {
        "fingerprint": fingerprint,
        "passed": result.passed,
        "total": result.total,
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_state(settings, state)


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
