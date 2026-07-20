"""Aggregate per-metric result files into the paper's summary tables.

Reads whatever `results/*.json` the eval stages produced and assembles:
  - **Table 1** — syntax validity per model.
  - **Table 6** — overall fine-tuned-vs-baseline comparison across syntax,
    geometry (Chamfer/Hausdorff/volume/area), CLIP semantic similarity, and
    prompt-guided detection accuracy.

Writes `results/summary.json` and a human-readable `results/summary.md`.
"""
from __future__ import annotations

from pathlib import Path

from codemaya.utils.io import ensure_dir, read_json, write_json
from codemaya.utils.logging import get_logger

log = get_logger("aggregate")

RESULT_FILES = ["syntax_validity", "semantic_visual", "geometry",
                "detection", "siamese", "dataset_stats"]


def _load_all(results_dir: Path) -> dict:
    loaded = {}
    for name in RESULT_FILES:
        p = results_dir / f"{name}.json"
        if p.exists():
            loaded[name] = read_json(p)
    return loaded


def _table1(data: dict) -> list[dict]:
    rows = []
    for model, s in (data.get("syntax_validity") or {}).items():
        rows.append({"model": model, "syntax_valid_pct": s["valid_pct"], "n": s["n"]})
    return sorted(rows, key=lambda r: r["syntax_valid_pct"])


def _table6(data: dict) -> list[dict]:
    """One row per model, pulling each metric where available."""
    syntax = data.get("syntax_validity") or {}
    sem = data.get("semantic_visual") or {}
    geo = data.get("geometry") or {}
    det = data.get("detection") or {}
    models = set(syntax) | set(sem) | set(geo)
    rows = []
    for m in sorted(models):
        row = {"model": m}
        if m in syntax:
            row["syntax_valid_pct"] = syntax[m]["valid_pct"]
        if m in sem:
            row["clip_semantic"] = sem[m]["clip_semantic"]
            row["dino_visual"] = sem[m]["dino_visual"]
        if m in geo:
            row.update({k: geo[m][k] for k in ("chamfer", "hausdorff", "volume_ratio", "area_ratio")})
        rows.append(row)
    # detection is model-agnostic in this impl (one classifier); attach separately
    if det:
        rows.append({"model": "prompt_guided_detection", "accuracy": det.get("accuracy")})
    return rows


def _to_markdown(title: str, rows: list[dict]) -> str:
    if not rows:
        return f"### {title}\n\n_(no data)_\n"
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = "\n".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows)
    return f"### {title}\n\n{head}\n{sep}\n{body}\n"


def run(cfg, args) -> None:
    results_dir = Path(cfg.paths.results)
    data = _load_all(results_dir)
    if not data:
        log.warning("no result files in %s — run the eval-* stages first", results_dir)

    summary = {
        "table1_syntax_validity": _table1(data),
        "table6_overall": _table6(data),
        "siamese": data.get("siamese"),
        "dataset": data.get("dataset_stats", {}).get("by_type"),
        "sources": sorted(data),
    }
    ensure_dir(results_dir)
    write_json(results_dir / "summary.json", summary)

    md = "# CodeMaya — Results Summary\n\n"
    md += _to_markdown("Table 1 — Syntax validity", summary["table1_syntax_validity"]) + "\n"
    md += _to_markdown("Table 6 — Overall comparison", summary["table6_overall"]) + "\n"
    if summary["siamese"]:
        md += _to_markdown("Siamese similarity",
                           [{"pair": k, "score": v} for k, v in summary["siamese"].items()])
    (results_dir / "summary.md").write_text(md, encoding="utf-8")

    log.info("aggregated %d result file(s) -> %s", len(data), results_dir / "summary.md")
    for r in summary["table1_syntax_validity"]:
        log.info("  syntax | %-10s %5.1f%%", r["model"], r["syntax_valid_pct"])
