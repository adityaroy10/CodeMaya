"""Evaluation — semantic (CLIP) and visual (DINOv2) similarity.

Per model, over the generated renders:
  - **CLIP semantic similarity** (paper 4.8): cosine(prompt, rendered image) —
    does the render *mean* what the prompt asked for.
  - **DINOv2 visual similarity** (paper 4.3): cosine(rendered image, ground-truth
    render) — does the render *look like* the reference.

Reads `results/generations.jsonl` (`{model, prompt, image, gt_image}`); writes
`results/semantic_visual.json`.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from codemaya.evaluation.embeddings import cosine_sim
from codemaya.utils.io import ensure_dir, read_jsonl, write_json
from codemaya.utils.logging import get_logger

log = get_logger("eval.semvis")


def _load_images(paths):
    from PIL import Image
    return [Image.open(p).convert("RGB") for p in paths]


def clip_semantic(prompts, images, clip) -> float:
    t = clip.encode_text(prompts)
    v = clip.encode_image(images)
    return float(cosine_sim(t, v).mean())


def dino_visual(gen_images, gt_images, dino) -> float:
    a = dino.encode_image(gen_images)
    b = dino.encode_image(gt_images)
    return float(cosine_sim(a, b).mean())


def _build_encoders(cfg, smoke: bool):
    if smoke:
        from codemaya.evaluation.embeddings import MockEncoder
        return MockEncoder(), MockEncoder()
    from codemaya.evaluation.embeddings import ClipEncoder, DinoEncoder
    return ClipEncoder(cfg.eval.clip_model), DinoEncoder(cfg.eval.dino_model)


def run(cfg, args) -> None:
    clip, dino = _build_encoders(cfg, args.smoke)

    if args.smoke:
        from PIL import Image
        rows = [{"model": "finetuned", "prompt": "a red cube",
                 "img": Image.new("RGB", (64, 64)), "gt": Image.new("RGB", (64, 64))}
                for _ in range(3)]
        by_model = {"finetuned": rows}
    else:
        gen_path = Path(cfg.paths.results) / "generations.jsonl"
        by_model = defaultdict(list)
        for r in read_jsonl(gen_path):
            r = dict(r)
            r["img"] = _load_images([r["image"]])[0]
            r["gt"] = _load_images([r["gt_image"]])[0] if r.get("gt_image") else r["img"]
            by_model[r["model"]].append(r)

    results = {}
    for model, rows in by_model.items():
        prompts = [r["prompt"] for r in rows]
        imgs = [r["img"] for r in rows]
        gts = [r["gt"] for r in rows]
        results[model] = {
            "clip_semantic": round(clip_semantic(prompts, imgs, clip), 4),
            "dino_visual": round(dino_visual(imgs, gts, dino), 4),
            "n": len(rows),
        }
        log.info("%-10s | CLIP=%.4f DINOv2=%.4f (n=%d)", model,
                 results[model]["clip_semantic"], results[model]["dino_visual"], len(rows))

    out = ensure_dir(cfg.paths.results)
    write_json(Path(out) / "semantic_visual.json", results)
