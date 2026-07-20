"""Scrape the Autodesk Maya MEL command reference.

The paper builds its dataset from "publicly accessible sources such as
Autodesk's official Maya documentation". This module fetches the MEL command
index, then each command page, and parses it into structured records:

    {command, url, category, synopsis, return_type, flags[], examples[]}

Records are written as JSONL to `paths.raw_docs/commands.jsonl` and feed the
dataset builder (build_dataset.py), which turns each command's synopsis / flags
/ examples into candidate responses and asks Gemini for matching prompts.

Politeness: caches every page to `paths.cache`, sleeps `request_delay_sec`
between live requests, sends a descriptive User-Agent. Respect Autodesk's terms
of use and robots.txt. Parsing is decoupled from fetching so it is unit-testable
offline (see parse_command_html / parse_index).
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from urllib.parse import urljoin

from codemaya.utils.io import ensure_dir, write_jsonl
from codemaya.utils.logging import get_logger

log = get_logger("scrape")

FIXTURE = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "sample_command.html"


# --------------------------------------------------------------------- fetch --
def _cache_path(cache_dir: Path, url: str) -> Path:
    h = hashlib.md5(url.encode("utf-8")).hexdigest()
    return cache_dir / f"{h}.html"


def fetch(url: str, cfg, *, session=None) -> str:
    """GET `url` with caching + retry + politeness delay. Returns HTML text."""
    cache_dir = ensure_dir(cfg.paths.cache)
    cpath = _cache_path(cache_dir, url)
    if cpath.exists():
        return cpath.read_text(encoding="utf-8")

    import requests
    from tenacity import retry, stop_after_attempt, wait_exponential

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, max=30))
    def _get(u: str) -> str:
        sess = session or requests
        resp = sess.get(u, headers={"User-Agent": cfg.scrape.user_agent}, timeout=30)
        resp.raise_for_status()
        return resp.text

    time.sleep(float(cfg.scrape.request_delay_sec))
    html = _get(url)
    cpath.write_text(html, encoding="utf-8")
    return html


# --------------------------------------------------------------------- parse --
def _clean(text: str) -> str:
    return " ".join(text.split()).strip()


def parse_index(html: str, base_url: str) -> list[dict]:
    """Extract (command, url) pairs from the MEL command index page."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    out, seen = [], set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        # command pages are relative .html links, not anchors or externals
        if not href.endswith(".html") or href.startswith(("http", "#", "index")):
            continue
        name = _clean(a.get_text()) or Path(href).stem
        url = urljoin(base_url + "/", href)
        if url not in seen:
            seen.add(url)
            out.append({"command": name, "url": url})
    return out


def parse_command_html(html: str, url: str = "") -> dict:
    """Parse a single MEL command page into a structured record.

    Defensive against layout drift: tries labelled sections first, then falls
    back to the first meaningful paragraph / any <pre> blocks.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")

    # command name: <h1>, else <title>, else url stem
    name = ""
    if soup.h1:
        name = _clean(soup.h1.get_text())
    if not name and soup.title:
        name = _clean(soup.title.get_text()).split("|")[0].strip()
    if not name and url:
        name = Path(url).stem

    def section_text(*labels: str) -> str:
        for tag in soup.find_all(["h2", "h3", "h4", "strong", "b"]):
            if _clean(tag.get_text()).lower().rstrip(":") in labels:
                node = tag.find_next(["p", "div", "pre"])
                if node:
                    return _clean(node.get_text())
        return ""

    synopsis = section_text("synopsis", "description") or (
        _clean(soup.p.get_text()) if soup.p else "")
    return_type = section_text("return value", "returns", "return type")

    # flags: rows of a table whose header mentions "flag"/"long name"
    flags: list[dict] = []
    for table in soup.find_all("table"):
        header = _clean(table.get_text()[:120]).lower()
        if "flag" in header or "long name" in header:
            for tr in table.find_all("tr"):
                cells = [_clean(td.get_text()) for td in tr.find_all(["td", "th"])]
                cells = [c for c in cells if c]
                if len(cells) >= 2 and cells[0].lower() not in ("long name", "flag"):
                    flags.append({"long": cells[0], "short": cells[1] if len(cells) > 1 else "",
                                  "type": cells[2] if len(cells) > 2 else "",
                                  "desc": cells[3] if len(cells) > 3 else ""})

    # examples: MEL/python code in <pre> blocks
    examples = [ _clean(pre.get_text()) for pre in soup.find_all("pre") if _clean(pre.get_text()) ]

    return {
        "command": name,
        "url": url,
        "synopsis": synopsis,
        "return_type": return_type,
        "flags": flags,
        "examples": examples,
    }


# ----------------------------------------------------------------------- run --
def run(cfg, args) -> None:
    """CLI entry: scrape the command reference into raw_docs/commands.jsonl."""
    out_dir = ensure_dir(cfg.paths.raw_docs)
    out_path = Path(out_dir) / "commands.jsonl"

    if args.smoke:
        log.info("smoke: parsing bundled fixture, no network")
        rec = parse_command_html(FIXTURE.read_text(encoding="utf-8"), "fixture://sample")
        n = write_jsonl(out_path, [rec])
        log.info("wrote %d record(s) -> %s (command=%s, %d flags, %d examples)",
                 n, out_path, rec["command"], len(rec["flags"]), len(rec["examples"]))
        return

    base = cfg.scrape.base_url
    index_url = urljoin(base + "/", cfg.scrape.index_page)
    log.info("fetching command index: %s", index_url)
    commands = parse_index(fetch(index_url, cfg), base)
    cap = cfg.scrape.max_commands
    if cap:
        commands = commands[: int(cap)]
    log.info("found %d commands to scrape", len(commands))

    records = []
    for i, cmd in enumerate(commands, 1):
        try:
            rec = parse_command_html(fetch(cmd["url"], cfg), cmd["url"])
            records.append(rec)
            if i % 25 == 0:
                log.info("  scraped %d/%d", i, len(commands))
        except Exception as exc:  # noqa: BLE001 - one bad page shouldn't kill the run
            log.warning("failed %s: %s", cmd["url"], exc)

    n = write_jsonl(out_path, records)
    log.info("wrote %d command records -> %s", n, out_path)
