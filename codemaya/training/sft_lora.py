"""Stage 1 — Supervised fine-tuning of Code LLaMA-Instruct with LoRA.

The paper: Code LLaMA-Instruct 7B is LoRA-fine-tuned on the MEL dataset with
prompts tokenized from NL descriptions, MEL scripts as targets, and cross-entropy
loss (the standard causal-LM objective). This module assembles tokenizer + model
(+ optional 4-bit QLoRA) + LoRA adapters + a TRL SFTTrainer and, when run for
real, trains and saves the adapter.

Targets trl>=1.8 / transformers>=5 / peft>=0.19. On CUDA with bitsandbytes it
does 4-bit QLoRA; otherwise it falls back to full-precision LoRA (CPU/macOS).

Note: `run()` only calls `.train()` in non-smoke mode. `--smoke` assembles the
whole pipeline on a tiny random model and returns WITHOUT training — use it to
verify the code path without a GPU or a real 7B download.
"""
from __future__ import annotations

from pathlib import Path

from codemaya.prompts import format_example
from codemaya.utils.io import ensure_dir, read_jsonl
from codemaya.utils.logging import get_logger

log = get_logger("sft")

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"  # smoke only


def _four_bit_available(cfg) -> bool:
    if not cfg.sft.load_in_4bit:
        return False
    try:
        import bitsandbytes  # noqa: F401
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def load_tokenizer(base: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(base: str, cfg, use_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if cfg.sft.bf16 else torch.float32
    if use_4bit:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training

        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base, quantization_config=quant, device_map="auto")
        model = prepare_model_for_kbit_training(model)
        log.info("loaded %s in 4-bit (QLoRA)", base)
    else:
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype)
        log.info("loaded %s in %s (full-precision LoRA)", base, dtype)
    model.config.use_cache = False
    return model


def apply_lora(model, cfg):
    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(
        r=int(cfg.lora.r), lora_alpha=int(cfg.lora.alpha),
        lora_dropout=float(cfg.lora.dropout),
        target_modules=list(cfg.lora.target_modules),
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


def load_split(cfg, split: str, limit: int | None = None):
    from datasets import Dataset

    path = Path(cfg.paths.dataset) / f"{split}.jsonl"
    rows = list(read_jsonl(path)) if path.exists() else []
    if limit:
        rows = rows[:limit]
    texts = [format_example(r["prompt"], r["response"]) for r in rows]
    return Dataset.from_dict({"text": texts})


def build_trainer(cfg, model, tokenizer, train_ds, eval_ds):
    from trl import SFTConfig, SFTTrainer

    args = SFTConfig(
        output_dir=str(Path(cfg.paths.checkpoints) / "sft_lora"),
        per_device_train_batch_size=int(cfg.sft.batch_size),
        gradient_accumulation_steps=int(cfg.sft.grad_accum),
        num_train_epochs=float(cfg.sft.epochs),
        learning_rate=float(cfg.sft.lr),
        warmup_ratio=float(cfg.sft.warmup_ratio),
        bf16=bool(cfg.sft.bf16),
        logging_steps=10, save_strategy="epoch",
        max_length=int(cfg.model.max_seq_len),
        dataset_text_field="text", packing=False,
        report_to="none", seed=int(cfg.seed),
    )
    has_eval = eval_ds is not None and len(eval_ds) > 0
    return SFTTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=eval_ds if has_eval else None,
        processing_class=tokenizer,
    )


def run(cfg, args) -> None:
    base = TINY_MODEL if args.smoke else cfg.model.base
    use_4bit = False if args.smoke else _four_bit_available(cfg)

    tokenizer = load_tokenizer(base)
    model = load_model(base, cfg, use_4bit)
    model = apply_lora(model, cfg)

    limit = 4 if args.smoke else None
    train_ds = load_split(cfg, "train", limit)
    eval_ds = load_split(cfg, "val", limit)
    log.info("train=%d eval=%d examples", len(train_ds), len(eval_ds))

    trainer = build_trainer(cfg, model, tokenizer, train_ds, eval_ds)

    if args.smoke:
        log.info("smoke: trainer assembled on %s — NOT training. OK.", base)
        return

    log.info("starting SFT training …")
    trainer.train()
    out = ensure_dir(Path(cfg.paths.checkpoints) / "sft_lora")
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    log.info("saved LoRA adapter -> %s", out)
