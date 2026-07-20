"""Inference — load the fine-tuned policy and generate MEL from a prompt.

`load_policy` returns (tokenizer, model) with the SFT LoRA adapter (and the
stage-2 contrastive adapter, if present) applied on top of the base model. When
no adapter exists yet (or in smoke mode) it applies a fresh LoRA so the model is
still well-formed and — importantly for stage-2 — has trainable adapter params.

`generate_mel` renders the instruction with the shared Code-LLaMA-Instruct
template and decodes the completion.
"""
from __future__ import annotations

from pathlib import Path

from codemaya.prompts import format_prompt
from codemaya.utils.logging import get_logger

log = get_logger("infer")

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _lora_config(cfg):
    from peft import LoraConfig

    return LoraConfig(
        r=int(cfg.lora.r), lora_alpha=int(cfg.lora.alpha),
        lora_dropout=float(cfg.lora.dropout),
        target_modules=list(cfg.lora.target_modules),
        bias="none", task_type="CAUSAL_LM",
    )


def load_policy(cfg, base: str | None = None, smoke: bool = False):
    """Load tokenizer + base model + LoRA adapter(s). Returns (tokenizer, model)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = base or (TINY_MODEL if smoke else cfg.model.base)
    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if cfg.sft.bf16 and not smoke else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype)

    # prefer the stage-2 aligned adapter, else the SFT adapter, else a fresh LoRA
    sft = Path(cfg.paths.checkpoints) / "sft_lora"
    aligned = Path(cfg.paths.checkpoints) / "contrastive_lora"
    adapter = aligned if aligned.exists() else (sft if sft.exists() else None)

    if adapter is not None and not smoke:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=True)
        log.info("loaded adapter: %s", adapter)
    else:
        from peft import get_peft_model
        model = get_peft_model(model, _lora_config(cfg))
        log.info("no saved adapter — applied a fresh LoRA (base=%s)", base)

    return tokenizer, model


def generate_mel(model, tokenizer, prompt: str, max_new_tokens: int = 256,
                 device: str = "cpu") -> str:
    import torch

    ids = tokenizer(format_prompt(prompt), return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0, ids.size(1):], skip_special_tokens=True).strip()


def _prompt_from_args(args) -> str | None:
    rest = getattr(args, "rest", []) or []
    if "--prompt" in rest:
        i = rest.index("--prompt")
        if i + 1 < len(rest):
            return rest[i + 1]
    return None


def run(cfg, args) -> None:
    tokenizer, model = load_policy(cfg, smoke=args.smoke)
    prompt = _prompt_from_args(args) or "Create a polygon cube of width 2 in Maya."
    mtok = 16 if args.smoke else int(cfg.model.max_seq_len)
    mel = generate_mel(model, tokenizer, prompt, max_new_tokens=mtok)
    log.info("prompt: %s", prompt)
    print("---- generated MEL ----")
    print(mel if mel else "(empty)")
    if args.smoke:
        log.info("smoke: generation path OK (%d chars)", len(mel))
