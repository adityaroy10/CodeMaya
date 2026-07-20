"""Gemini client for backward prompt generation.

The paper: "we employed gemini-2.0-flash to generate natural language prompts
that closely reflect the intent and structure of each response." Given a
scraped Maya response (a MEL snippet, a synopsis, a flag description), we ask
Gemini for the concise natural-language question a user would ask to get it.

`GeminiClient` calls the real API (needs GOOGLE_API_KEY); `MockGeminiClient`
returns deterministic templated prompts so the pipeline runs offline (smoke
mode + tests). Both cache by content hash so re-runs never re-call the API.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from codemaya.utils.io import ensure_dir
from codemaya.utils.logging import get_logger

log = get_logger("gemini")

_INSTRUCTION = (
    "You are helping build a dataset for an assistant that writes Autodesk Maya "
    "{rtype} for artists. Below is a response taken from Maya documentation. "
    "Write ONE concise, natural first-person question a Maya user would ask that "
    "this response answers. Output only the question, no preamble.\n\n"
    "Response:\n{response}\n"
)


def _key(model: str, text: str) -> str:
    return hashlib.md5(f"{model}::{text}".encode("utf-8")).hexdigest()


class _BaseClient:
    def __init__(self, model_name: str, cache_dir: str | Path):
        self.model_name = model_name
        self.cache_dir = ensure_dir(Path(cache_dir) / "gemini")

    def _cached(self, text: str) -> str | None:
        p = self.cache_dir / f"{_key(self.model_name, text)}.txt"
        return p.read_text(encoding="utf-8") if p.exists() else None

    def _store(self, text: str, result: str) -> None:
        (self.cache_dir / f"{_key(self.model_name, text)}.txt").write_text(
            result, encoding="utf-8")

    def backward_prompt(self, response: str, rtype: str) -> str:
        """Generate the NL prompt that `response` answers (cached)."""
        instruction = _INSTRUCTION.format(rtype=rtype, response=response[:4000])
        hit = self._cached(instruction)
        if hit is not None:
            return hit
        result = self._generate(instruction).strip()
        self._store(instruction, result)
        return result

    def _generate(self, instruction: str) -> str:  # pragma: no cover - overridden
        raise NotImplementedError


class GeminiClient(_BaseClient):
    """Real gemini-2.0-flash client."""

    def __init__(self, model_name: str, cache_dir: str | Path, api_key: str | None = None):
        super().__init__(model_name, cache_dir)
        import google.generativeai as genai  # lazy: only when actually calling the API

        key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "No Gemini API key. Set GOOGLE_API_KEY, or use MockGeminiClient / --smoke.")
        genai.configure(api_key=key)
        self._model = genai.GenerativeModel(model_name)

    def _generate(self, instruction: str) -> str:
        from tenacity import retry, stop_after_attempt, wait_exponential

        @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, max=30))
        def _call() -> str:
            resp = self._model.generate_content(instruction)
            return resp.text

        return _call()


class MockGeminiClient(_BaseClient):
    """Offline stand-in: deterministic templated prompts, no network."""

    _TEMPLATES = {
        "mel": "What is the MEL command to {hint}?",
        "python": "How do I {hint} in Maya using Python?",
        "info": "How does {hint} work in Maya?",
    }

    def backward_prompt(self, response: str, rtype: str) -> str:
        first_line = response.strip().splitlines()[0] if response.strip() else "do this"
        hint = " ".join(first_line.replace("//", "").replace("#", "").split()[:8]).lower() or "do this"
        return self._TEMPLATES.get(rtype, self._TEMPLATES["info"]).format(hint=hint)

    def _generate(self, instruction: str) -> str:  # not used, backward_prompt overridden
        return "How do I do this in Maya?"


def make_client(model_name: str, cache_dir: str | Path, mock: bool = False) -> _BaseClient:
    if mock:
        return MockGeminiClient(model_name, cache_dir)
    return GeminiClient(model_name, cache_dir)
