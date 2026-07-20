"""Tests for config loading and IO helpers."""
from codemaya.config import load_config
from codemaya.utils.io import read_jsonl, write_jsonl


def test_config_defaults_and_dotted_access():
    cfg = load_config()
    assert cfg.model.base.startswith("codellama/")
    assert isinstance(cfg.lora.target_modules, list)
    assert cfg.sft.batch_size == 4


def test_config_override_coercion():
    cfg = load_config(overrides=["sft.lr=0.001", "sft.epochs=1", "render.backend=maya"])
    assert cfg.sft.lr == 0.001 and isinstance(cfg.sft.lr, float)
    assert cfg.sft.epochs == 1 and isinstance(cfg.sft.epochs, int)
    assert cfg.render.backend == "maya"


def test_config_nested_override_creates_path():
    cfg = load_config(overrides=["eval.syntax_backend=maya"])
    assert cfg.eval.get("syntax_backend") == "maya"


def test_jsonl_roundtrip(tmp_path):
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    p = tmp_path / "sub" / "data.jsonl"
    n = write_jsonl(p, rows)
    assert n == 2
    assert list(read_jsonl(p)) == rows
