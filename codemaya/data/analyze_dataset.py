"""Dataset analysis — reproduces the paper's dataset insights.

Computes, over the built dataset:
  - response-type counts (MEL / python / info)                      [Sec 4.1]
  - prompt-length distribution in tokens                            [Fig. 3]
  - most frequent functions/commands across responses              [Table 2]
  - explainability split: explained vs code-only responses         [Table 3]

Writes `results/dataset_stats.json`, consumed by the visualization notebook.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from codemaya.utils.io import ensure_dir, read_jsonl, write_json
from codemaya.utils.logging import get_logger

log = get_logger("analyze")

# identifier-like tokens (MEL commands, python calls): a word optionally followed by "("
_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\b")
# MEL/python comment markers signal a natural-language explanation is present
_COMMENT_RE = re.compile(r"(^|\s)(//|#)")

# tokens that are language keywords / noise, not "functions" worth ranking
_STOP = {"the", "and", "for", "you", "this", "with", "that", "int", "float",
         "string", "true", "false", "None", "return", "import", "from", "def"}


def prompt_len_tokens(prompt: str, tokenizer=None) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(prompt))
    return len(prompt.split())


def histogram(values: list[int], bin_width: int = 5, max_val: int = 100) -> dict:
    counts: Counter = Counter()
    for v in values:
        b = min(v // bin_width * bin_width, max_val)
        counts[b] += 1
    bins = sorted(counts)
    return {"bin_width": bin_width, "bins": bins, "counts": [counts[b] for b in bins]}


def top_functions(responses: list[str], k: int = 15) -> list[list]:
    c: Counter = Counter()
    for r in responses:
        for tok in _CALL_RE.findall(r):
            if tok not in _STOP:
                c[tok] += 1
    return [[name, n] for name, n in c.most_common(k)]


def explainability(rows: list[dict]) -> dict:
    explained = code_only = 0
    for row in rows:
        resp, rtype = row.get("response", ""), row.get("type", "")
        if rtype == "info" or _COMMENT_RE.search(resp):
            explained += 1
        else:
            code_only += 1
    return {"explained": explained, "code_only": code_only}


def _load_tokenizer(cfg):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(cfg.model.base)
    except Exception as exc:  # noqa: BLE001 - tokenizer is optional, fall back to whitespace
        log.warning("tokenizer unavailable (%s); using whitespace token counts", exc)
        return None


def run(cfg, args) -> None:
    ds_dir = Path(cfg.paths.dataset)
    rows: list[dict] = []
    for split in ("train", "val", "test"):
        p = ds_dir / f"{split}.jsonl"
        if p.exists():
            rows.extend(read_jsonl(p))
    if not rows:
        raise FileNotFoundError(f"No dataset splits in {ds_dir} — run `build-dataset` first.")
    log.info("loaded %d examples", len(rows))

    tokenizer = None if args.smoke else _load_tokenizer(cfg)
    lengths = [prompt_len_tokens(r["prompt"], tokenizer) for r in rows]
    responses = [r["response"] for r in rows]

    by_type: Counter = Counter(r["type"] for r in rows)
    stats = {
        "n_total": len(rows),
        "by_type": dict(by_type),
        "prompt_len": {
            "min": min(lengths), "max": max(lengths),
            "mean": round(sum(lengths) / len(lengths), 2),
            "histogram": histogram(lengths),
        },
        "top_functions": top_functions(responses),
        "explainability": explainability(rows),
    }

    out = ensure_dir(cfg.paths.results)
    path = Path(out) / "dataset_stats.json"
    write_json(path, stats)
    log.info("by_type=%s | mean_prompt_len=%.1f | explained=%d code_only=%d",
             dict(by_type), stats["prompt_len"]["mean"],
             stats["explainability"]["explained"], stats["explainability"]["code_only"])
    log.info("wrote %s", path)
