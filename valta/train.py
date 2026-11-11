from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from valta.config import Settings, load_state, save_state
from valta.memory import Memory
from valta.ollama_client import OllamaClient, OllamaError
from valta.prompt import build_system, load_rules


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
    """Training text is question-answer pairs. Raw ebook text is not the corpus."""
    rules = load_rules(settings)
    pairs: list[dict] = []
    example_ids: list[str] = []

    for row in memory.unused_examples():
        user_text = row["user_text"]
        assistant_text = row["assistant_text"]
        if conflicts_with_rules(user_text) or conflicts_with_rules(assistant_text):
            continue
        pairs.append(_pair(rules, user_text, assistant_text))
        example_ids.append(row["id"])

    for record in memory.recent(limit=200, kind="research"):
        if conflicts_with_rules(record.text):
            continue
        question = f"What did you learn about {record.title or 'this topic'}?"
        pairs.append(_pair(rules, question, record.text))

    if client and client.healthy(settings.heavy_host):
        for record in memory.recent(limit=20, kind="book"):
            drafted = _qa_from_passage(settings, client, record.text)
            if not drafted or conflicts_with_rules(drafted[0]) or conflicts_with_rules(drafted[1]):
                continue
            pairs.append(_pair(rules, drafted[0], drafted[1]))

    if not pairs:
        raise RuntimeError("Nothing to train on yet. Ingest notes, research, or correct a reply first.")

    settings.datasets_dir.mkdir(parents=True, exist_ok=True)
    path = settings.datasets_dir / "train.jsonl"
    import json

    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Wrote {len(pairs)} training pairs to {path}")
    return path, example_ids


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


def run_qlora(settings: Settings, dataset_path: Path) -> Path:
    """QLoRA on the GTX 1660. 4-bit NF4, batch size 1, fp16, short context."""
    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install them with:\n"
            "  pip install -r requirements-train.txt"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. QLoRA runs on the GTX 1660.")

    import json

    from datasets import Dataset

    rows = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append({"text": render_text(json.loads(line)["messages"])})
    dataset = Dataset.from_list(rows)

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

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=settings.self_num_ctx)

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
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
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
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


def convert_to_gguf(merged_dir: Path, gguf_path: Path) -> Path:
    convert = os.environ.get("LLAMA_CPP_CONVERT") or shutil.which("convert_hf_to_gguf.py")
    if not convert:
        raise RuntimeError(
            "The adapter is saved, and the merged weights are in data/merged. "
            "Set LLAMA_CPP_CONVERT to llama.cpp's convert_hf_to_gguf.py, run it, "
            "then `valta train --create-only` with VALTA_GGUF pointing at the .gguf file."
        )
    import subprocess

    gguf_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["python", convert, str(merged_dir), "--outfile", str(gguf_path), "--outtype", "q4_k_m"],
        check=True,
    )
    return gguf_path


def write_modelfile(settings: Settings, gguf_path: Path) -> str:
    current = settings.models_dir / "valta.Modelfile"
    previous = settings.models_dir / "valta.previous.Modelfile"
    if current.exists():
        previous.write_text(current.read_text(encoding="utf-8"), encoding="utf-8")
        state = load_state(settings)
        state["previous_modelfile"] = str(previous)
        save_state(settings, state)
    modelfile = f"FROM {gguf_path.as_posix()}\nPARAMETER num_ctx {settings.self_num_ctx}\n"
    current.write_text(modelfile, encoding="utf-8")
    return modelfile


def create_valta(settings: Settings, client: OllamaClient, modelfile: str) -> None:
    client.create(settings.self_host, "valta", modelfile)
    state = load_state(settings)
    state["personal_model"] = "valta"
    save_state(settings, state)
    print("Personal model is now `valta` on the GTX 1660.")


def rollback(settings: Settings, client: OllamaClient) -> None:
    previous = settings.models_dir / "valta.previous.Modelfile"
    if not previous.exists():
        raise RuntimeError("There is no previous Modelfile to roll back to.")
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
