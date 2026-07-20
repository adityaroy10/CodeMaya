"""Prompt formatting for Code LLaMA-Instruct.

One place defines how a (natural-language prompt, MEL response) pair becomes the
instruction-tuning string, so dataset construction, SFT training, and inference
all agree on the exact format. Code LLaMA-Instruct uses the Llama-2 chat
template: `<s>[INST] <<SYS>>..<</SYS>> {user} [/INST] {assistant} </s>`.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are CodeMaya, an expert assistant for Autodesk Maya. Given a request, "
    "you reply with a correct, minimal MEL script (or Python-for-Maya when asked) "
    "that accomplishes it. Output only the script unless an explanation is requested."
)


def format_prompt(user_prompt: str) -> str:
    """Render the instruction half (no answer) — used at inference time."""
    return f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{user_prompt.strip()} [/INST]"


def format_example(user_prompt: str, response: str) -> str:
    """Render a full training example (instruction + answer)."""
    return f"{format_prompt(user_prompt)} {response.strip()} </s>"
