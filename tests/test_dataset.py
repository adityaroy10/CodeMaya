"""Tests for dataset construction and analysis (offline, MockGemini)."""
from codemaya.data.analyze_dataset import explainability, histogram, top_functions
from codemaya.data.build_dataset import build_pairs, responses_from_record, split_pairs
from codemaya.data.gemini_client import MockGeminiClient

RECORD = {
    "command": "sphere",
    "synopsis": "Creates a NURBS sphere.",
    "flags": [{"long": "radius"}, {"long": "spans"}],
    "examples": ["sphere -r 5;", "import maya.cmds as cmds\ncmds.sphere(r=5)"],
}


def test_responses_from_record_types():
    resp = responses_from_record(RECORD)
    types = [t for _, t in resp]
    assert types.count("mel") == 1        # "sphere -r 5;"
    assert types.count("python") == 1     # the cmds. example
    assert types.count("info") == 1       # synopsis
    # info response carries the flag summary
    info = [r for r, t in resp if t == "info"][0]
    assert "radius" in info


def test_build_pairs_with_mock(tmp_path):
    client = MockGeminiClient("mock", tmp_path)
    pairs = build_pairs([RECORD], client, max_prompt_tokens=256)
    assert len(pairs) == 3
    assert all(p["prompt"] and p["response"] for p in pairs)
    assert {p["type"] for p in pairs} == {"mel", "python", "info"}


def test_split_proportions():
    pairs = [{"i": i} for i in range(100)]
    splits = split_pairs(pairs, {"train": 0.9, "val": 0.05, "test": 0.05}, seed=42)
    assert len(splits["train"]) == 90
    assert len(splits["val"]) == 5
    assert len(splits["test"]) == 5
    # deterministic under fixed seed
    again = split_pairs(pairs, {"train": 0.9, "val": 0.05, "test": 0.05}, seed=42)
    assert [r["i"] for r in splits["train"]] == [r["i"] for r in again["train"]]


def test_analyze_helpers():
    h = histogram([1, 2, 3, 22, 99, 250], bin_width=5, max_val=100)
    assert h["counts"][h["bins"].index(0)] == 3          # 1,2,3 -> bin 0
    assert 100 in h["bins"]                              # 250 clamps to 100

    fns = top_functions(["polyCube -w 1;", "polyCube -h 2;", "sphere -r 1;"])
    names = [n for n, _ in fns]
    assert names[0] == "polyCube"                        # most frequent

    ex = explainability([
        {"response": "// make a cube\npolyCube;", "type": "mel"},   # explained (comment)
        {"response": "sphere -r 1;", "type": "mel"},                # code-only
        {"response": "sphere info", "type": "info"},                # explained (info)
    ])
    assert ex == {"explained": 2, "code_only": 1}
