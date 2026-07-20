"""Evaluation — MEL syntax validity (paper Table 1).

Measures the fraction of generated scripts that are syntactically valid MEL,
per model (base Code LLaMA / Gemini / fine-tuned). Two checkers:

- **maya** (`--set eval.syntax_backend=maya`): the ground truth — pipe each
  script through `mayapy` and see whether `mel.eval` raises. Needs Maya.
- **heuristic** (default): offline structural check — balanced brackets/quotes,
  semicolon-terminated statements, command-like statement heads, no dangling
  operators. Approximates parse-ability without a Maya install.

Reads generations from `results/generations.jsonl` (`{model, prompt, mel}` per
line); writes per-model validity to `results/syntax_validity.json`.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

from codemaya.utils.io import ensure_dir, read_jsonl, write_json
from codemaya.utils.logging import get_logger

log = get_logger("eval.syntax")

_PAIRS = {")": "(", "]": "[", "}": "{"}
_OPENERS = set("([{")
_CTRL = {"if", "else", "for", "while", "do", "switch", "case", "default",
         "proc", "global", "return", "break", "continue", "{", "}"}


def _brackets_balanced(s: str) -> bool:
    stack = []
    for ch in s:
        if ch in _OPENERS:
            stack.append(ch)
        elif ch in _PAIRS:
            if not stack or stack.pop() != _PAIRS[ch]:
                return False
    return not stack


def _quotes_balanced(s: str) -> bool:
    # count unescaped double quotes; MEL strings use "
    return len(re.findall(r'(?<!\\)"', s)) % 2 == 0


def validate_mel_heuristic(mel: str) -> dict:
    """Structural MEL validity check. Returns {valid, errors}."""
    errors: list[str] = []
    code = mel.strip()
    if not code:
        return {"valid": False, "errors": ["empty"]}
    if not _brackets_balanced(code):
        errors.append("unbalanced brackets")
    if not _quotes_balanced(code):
        errors.append("unbalanced quotes")

    # strip comments, then require each statement to be command-like & ;-terminated
    body = re.sub(r"//[^\n]*", "", code)
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    # remove block braces to isolate statements
    stmts = [s.strip() for s in re.split(r";|\{|\}", body) if s.strip()]
    if not stmts:
        errors.append("no statements")
    for st in stmts:
        head = st.split()[0] if st.split() else ""
        if head in _CTRL or head.startswith(("$", "//")):
            continue
        # a command/assignment head must start with a letter or $
        if not re.match(r"^[A-Za-z_$]", head):
            errors.append(f"bad statement head: {st[:30]!r}")
            break
    # at least one ';' terminator for a non-control script
    if ";" not in code and not any(k in code for k in ("{", "proc")):
        errors.append("no statement terminator")

    return {"valid": len(errors) == 0, "errors": errors}


_MAYA_CHECK = (
    "import sys, maya.standalone as s; s.initialize('python');\n"
    "import maya.mel as mel\n"
    "code=open(sys.argv[1]).read()\n"
    "import maya.cmds as cmds\n"
    "try:\n"
    "    mel.eval(code); print('VALID')\n"
    "except Exception as e:\n"
    "    sys.stderr.write(str(e)); print('INVALID')\n"
)


def validate_mel_maya(mel: str, mayapy: str) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".mel", delete=False) as mf:
        mf.write(mel)
        mel_path = mf.name
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as pf:
        pf.write(_MAYA_CHECK)
        py_path = pf.name
    try:
        out = subprocess.run([mayapy, py_path, mel_path], capture_output=True,
                             text=True, timeout=120)
        valid = "VALID" in out.stdout and "INVALID" not in out.stdout
        return {"valid": valid, "errors": [] if valid else [out.stderr.strip()[:200]]}
    except Exception as exc:  # noqa: BLE001
        return {"valid": False, "errors": [f"maya check failed: {exc}"]}


def validate_mel(mel: str, cfg) -> dict:
    backend = cfg.eval.get("syntax_backend", "heuristic")
    if backend == "maya":
        return validate_mel_maya(mel, cfg.render.mayapy)
    return validate_mel_heuristic(mel)


def score_generations(rows: list[dict], cfg) -> dict:
    by_model: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        res = validate_mel(r.get("mel", ""), cfg)
        by_model[r.get("model", "unknown")].append(res["valid"])
    return {m: {"valid_pct": round(100 * sum(v) / len(v), 1), "n": len(v)}
            for m, v in by_model.items()}


_SMOKE_SAMPLES = [
    {"model": "finetuned", "mel": 'polyCube -w 2 -name "box";'},
    {"model": "finetuned", "mel": "sphere -r 1;"},
    {"model": "finetuned", "mel": "string $s = `polySphere -r 2`;"},
    {"model": "base", "mel": 'polyCube -w 2 -name "box;'},          # unbalanced quote
    {"model": "base", "mel": "sphere -r ((1;"},                      # unbalanced paren
    {"model": "base", "mel": ""},                                    # empty
]


def run(cfg, args) -> None:
    gen_path = Path(cfg.paths.results) / "generations.jsonl"
    if args.smoke or not gen_path.exists():
        if not args.smoke:
            log.warning("%s not found — using built-in smoke samples", gen_path)
        rows = _SMOKE_SAMPLES
    else:
        rows = list(read_jsonl(gen_path))

    scores = score_generations(rows, cfg)
    out = ensure_dir(cfg.paths.results)
    write_json(Path(out) / "syntax_validity.json", scores)
    for model, s in scores.items():
        log.info("syntax validity | %-10s : %5.1f%%  (n=%d)", model, s["valid_pct"], s["n"])
