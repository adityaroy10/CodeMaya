"""Build the instruction-tuning dataset from scraped Maya docs.

For each scraped command record we derive candidate *responses* (MEL examples,
the synopsis as an informational answer), ask Gemini for the natural-language
*prompt* each response answers, then emit (prompt, response, type) pairs. Pairs
are shuffled deterministically and split into train/val/test JSONL.

Matches the paper's dataset: a mixture of MEL scripts, Python scripts, and
general informational content, with prompts generated backward from responses.
"""
from __future__ import annotations

import random
from pathlib import Path

from codemaya.data.gemini_client import make_client
from codemaya.utils.io import ensure_dir, read_jsonl, write_jsonl
from codemaya.utils.logging import get_logger

log = get_logger("build_dataset")


def responses_from_record(rec: dict) -> list[tuple[str, str]]:
    """Yield (response_text, response_type) candidates from one command record."""
    out: list[tuple[str, str]] = []
    for ex in rec.get("examples", []):
        ex = ex.strip()
        if not ex:
            continue
        # crude MEL-vs-python split: python-for-maya uses cmds./import
        rtype = "python" if ("cmds." in ex or "import maya" in ex) else "mel"
        out.append((ex, rtype))
    synopsis = (rec.get("synopsis") or "").strip()
    if synopsis:
        cmd = rec.get("command", "")
        flag_note = ""
        if rec.get("flags"):
            names = ", ".join(f["long"] for f in rec["flags"][:6] if f.get("long"))
            if names:
                flag_note = f" Key flags: {names}."
        out.append((f"`{cmd}`: {synopsis}{flag_note}", "info"))
    return out


def build_pairs(records: list[dict], client, max_prompt_tokens: int) -> list[dict]:
    pairs: list[dict] = []
    for rec in records:
        for response, rtype in responses_from_record(rec):
            prompt = client.backward_prompt(response, rtype)
            # cheap length guard (paper: prompts are short, peak < 25 tokens)
            if len(prompt.split()) > max_prompt_tokens:
                continue
            pairs.append({
                "prompt": prompt,
                "response": response,
                "type": rtype,
                "command": rec.get("command", ""),
                "source": "maya_docs",
            })
    return pairs


def split_pairs(pairs: list[dict], splits: dict, seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * splits["train"])
    n_val = int(n * splits["val"])
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def run(cfg, args) -> None:
    raw_path = Path(cfg.paths.raw_docs) / "commands.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found — run `codemaya scrape` first (or `scrape --smoke`).")

    records = list(read_jsonl(raw_path))
    log.info("loaded %d command records", len(records))

    client = make_client(cfg.dataset.gemini_model, cfg.paths.cache, mock=args.smoke)
    if args.smoke:
        log.info("smoke: using MockGeminiClient (no API calls)")

    pairs = build_pairs(records, client, int(cfg.dataset.max_prompt_tokens))
    log.info("built %d (prompt, response) pairs", len(pairs))

    by_type: dict[str, int] = {}
    for p in pairs:
        by_type[p["type"]] = by_type.get(p["type"], 0) + 1
    log.info("pairs by type: %s", by_type)

    splits = split_pairs(pairs, cfg.dataset.splits.to_dict(), int(cfg.seed))
    out_dir = ensure_dir(cfg.paths.dataset)
    for name, rows in splits.items():
        n = write_jsonl(Path(out_dir) / f"{name}.jsonl", rows)
        log.info("wrote %s split: %d examples", name, n)
