"""Unified command-line entrypoint for the CodeMaya pipeline.

    python -m codemaya.cli <command> [--config PATH] [--set k.v=x ...] [opts]

Handlers lazy-import their stage module so `--help` and light commands work
without pulling in torch / transformers / trimesh.

Pipeline order:
    scrape -> build-dataset -> analyze-dataset -> train-sft ->
    train-contrastive -> render -> infer -> eval-* -> aggregate
"""
from __future__ import annotations

import argparse
import importlib
from typing import Callable

from codemaya.config import load_config
from codemaya.utils.logging import get_logger

log = get_logger("cli")

# command -> (module, function). Function signature: fn(cfg, args) -> None
COMMANDS: dict[str, tuple[str, str]] = {
    "scrape":            ("codemaya.data.scrape_maya_docs", "run"),
    "build-dataset":     ("codemaya.data.build_dataset", "run"),
    "analyze-dataset":   ("codemaya.data.analyze_dataset", "run"),
    "train-sft":         ("codemaya.training.sft_lora", "run"),
    "train-contrastive": ("codemaya.training.contrastive_align", "run"),
    "render":            ("codemaya.rendering.render", "run"),
    "infer":             ("codemaya.inference.generate", "run"),
    "predict":           ("codemaya.inference.predict", "run"),
    "eval-syntax":       ("codemaya.evaluation.syntax_validity", "run"),
    "eval-semantic":     ("codemaya.evaluation.semantic_visual", "run"),
    "eval-geometry":     ("codemaya.evaluation.geometry", "run"),
    "eval-detection":    ("codemaya.evaluation.object_detection", "run"),
    "eval-siamese":      ("codemaya.evaluation.siamese_similarity", "run"),
    "aggregate":         ("codemaya.evaluation.aggregate", "run"),
}


def _resolve(command: str) -> Callable:
    module_name, func_name = COMMANDS[command]
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="codemaya", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=list(COMMANDS), help="pipeline stage to run")
    p.add_argument("--config", default=None, help="path to YAML config")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="k.v=x", help="override a config key (repeatable)")
    p.add_argument("--smoke", action="store_true",
                   help="smoke mode: assemble/verify without heavy execution")
    return p


def main(argv: list[str] | None = None) -> int:
    # parse_known_args (not a REMAINDER positional) so global optionals like
    # --smoke / --set are always recognized; anything left over is stage-specific
    # (e.g. `infer --prompt "..."`) and handed to the stage on args.rest.
    args, extra = build_parser().parse_known_args(argv)
    args.rest = extra
    cfg = load_config(args.config, args.overrides)
    log.info("command=%s smoke=%s", args.command, args.smoke)
    fn = _resolve(args.command)
    fn(cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
