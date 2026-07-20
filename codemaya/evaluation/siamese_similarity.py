"""Evaluation — Siamese-ViT pairwise similarity (paper 4.9 / Table 5).

Embeds rendered objects with a ViT and scores pairwise cosine similarity,
averaged within/between classes. The paper's separation — cube-vs-cube ~0.91,
sphere-vs-sphere ~0.90, cube-vs-sphere ~0.34 — shows the metric reliably tells
matching shapes from mismatching ones.

MockEncoder + tiny render set in smoke mode (offline); real path uses
`eval.vit_model`.
"""
from __future__ import annotations

from itertools import combinations, product
from pathlib import Path

from codemaya.evaluation.embeddings import cosine_sim
from codemaya.utils.io import ensure_dir, write_json
from codemaya.utils.logging import get_logger

log = get_logger("eval.siamese")

DEFAULT_CLASSES = ["cube", "sphere"]


def _mean_pairwise(emb_a, emb_b, cross: bool) -> float:
    import torch
    scores = []
    if cross:
        pairs = list(product(range(len(emb_a)), range(len(emb_b))))
        for i, j in pairs:
            scores.append(cosine_sim(emb_a[i:i + 1], emb_b[j:j + 1]))
    else:
        for i, j in combinations(range(len(emb_a)), 2):
            scores.append(cosine_sim(emb_a[i:i + 1], emb_a[j:j + 1]))
    return float(torch.stack(scores).mean()) if scores else 0.0


def pairwise_similarity_table(embeddings_by_class: dict) -> dict:
    """Average similarity for each within-class and cross-class pairing."""
    classes = list(embeddings_by_class)
    table = {}
    for c in classes:
        table[f"{c}-vs-{c}"] = round(_mean_pairwise(embeddings_by_class[c],
                                                    embeddings_by_class[c], cross=False), 4)
    for a, b in combinations(classes, 2):
        table[f"{a}-vs-{b}"] = round(_mean_pairwise(embeddings_by_class[a],
                                                    embeddings_by_class[b], cross=True), 4)
    return table


def run(cfg, args) -> None:
    classes = DEFAULT_CLASSES
    if args.smoke:
        from codemaya.evaluation.embeddings import MockEncoder
        encoder = MockEncoder()
        n_per, views = 3, 1
    else:
        from codemaya.evaluation.embeddings import ViTEncoder
        encoder = ViTEncoder(cfg.eval.vit_model)
        n_per, views = 15, int(cfg.render.viewpoints)

    from codemaya.evaluation.object_detection import render_primitive_dataset
    images, labels = render_primitive_dataset(cfg, classes, n_per, views)
    feats = encoder.encode_image(images)

    emb_by_class = {}
    for ci, c in enumerate(classes):
        idx = [i for i, l in enumerate(labels) if l == ci]
        emb_by_class[c] = feats[idx]

    table = pairwise_similarity_table(emb_by_class)
    write_json(Path(ensure_dir(cfg.paths.results)) / "siamese.json", table)
    for pair, score in table.items():
        log.info("siamese similarity | %-16s : %.4f", pair, score)
