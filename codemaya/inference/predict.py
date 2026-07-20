"""Batch prediction — generate + render over the test set into generations.jsonl.

For each held-out prompt: generate a MEL script with the policy, render it to an
image + OBJ, and (when the test row carries a reference MEL response) render that
as ground truth. The resulting `results/generations.jsonl` is the single input
every eval-* stage consumes:

    {model, prompt, mel, image, obj, gt_image?, gt_obj?}

Real runs use the fine-tuned policy + the configured renderer; `--smoke` uses a
tiny model + fallback renderer on a couple of prompts.
"""
from __future__ import annotations

from pathlib import Path

from codemaya.inference.generate import generate_mel, load_policy
from codemaya.rendering.render import render_mel
from codemaya.utils.io import ensure_dir, read_jsonl, write_jsonl
from codemaya.utils.logging import get_logger

log = get_logger("predict")


def _model_tag(cfg, args) -> str:
    rest = getattr(args, "rest", []) or []
    if "--model-tag" in rest:
        return rest[rest.index("--model-tag") + 1]
    return "smoke" if args.smoke else "finetuned"


def _test_rows(cfg, args) -> list[dict]:
    path = Path(cfg.paths.dataset) / "test.jsonl"
    rows = list(read_jsonl(path)) if path.exists() else [
        {"prompt": "Create a polygon cube of width 2 in Maya.", "response": "polyCube -w 2;", "type": "mel"},
        {"prompt": "Make a sphere of radius 1.5.", "response": "sphere -r 1.5;", "type": "mel"},
    ]
    limit = int(cfg.eval.test_prompts)
    if args.smoke:
        limit = 2
    return rows[:limit]


def run(cfg, args) -> None:
    tag = _model_tag(cfg, args)
    tokenizer, model = load_policy(cfg, smoke=args.smoke)
    rows = _test_rows(cfg, args)
    renders_dir = ensure_dir(Path(cfg.paths.renders) / tag)
    mtok = 32 if args.smoke else int(cfg.model.max_seq_len)

    out_rows = []
    for i, row in enumerate(rows):
        mel = generate_mel(model, tokenizer, row["prompt"], max_new_tokens=mtok) or "polyCube;"
        pred = render_mel(mel, cfg, renders_dir / f"pred_{i}")
        rec = {"model": tag, "prompt": row["prompt"], "mel": mel,
               "image": pred["image"], "obj": pred["obj"]}
        # render the reference response as ground truth for geometry/visual metrics
        if row.get("type") == "mel" and row.get("response"):
            gt = render_mel(row["response"], cfg, renders_dir / f"gt_{i}")
            rec["gt_image"], rec["gt_obj"] = gt["image"], gt["obj"]
        out_rows.append(rec)
        if (i + 1) % 10 == 0:
            log.info("  predicted %d/%d", i + 1, len(rows))

    out = Path(ensure_dir(cfg.paths.results)) / "generations.jsonl"
    n = write_jsonl(out, out_rows)
    log.info("wrote %d generations (model=%s) -> %s", n, tag, out)
