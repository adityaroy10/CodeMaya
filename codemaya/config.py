"""Configuration loading.

A single YAML file (configs/default.yaml) drives every stage. Access is via
attribute-style dotted lookup so call sites read `cfg.sft.lr` rather than
`cfg["sft"]["lr"]`. CLI overrides arrive as `key.subkey=value` strings.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


class Config:
    """Recursive attribute wrapper around a nested dict."""

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getattr__(self, key: str) -> Any:
        try:
            val = self._data[key]
        except KeyError as exc:
            raise AttributeError(f"No config key '{key}'") from exc
        return Config(val) if isinstance(val, dict) else val

    def __getitem__(self, key: str) -> Any:
        return self.__getattr__(key)

    def get(self, key: str, default: Any = None) -> Any:
        val = self._data.get(key, default)
        return Config(val) if isinstance(val, dict) else val

    def to_dict(self) -> dict[str, Any]:
        return self._data


def _coerce(value: str) -> Any:
    """Turn a CLI string into a python literal when possible ("3" -> 3)."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _set_dotted(data: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = data
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_config(path: str | os.PathLike | None = None,
                overrides: list[str] | None = None) -> Config:
    """Load YAML config, apply `key.sub=value` overrides, return a Config."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"Override must be key=value, got: {ov!r}")
        key, raw = ov.split("=", 1)
        _set_dotted(data, key.strip(), _coerce(raw.strip()))

    return Config(data)
