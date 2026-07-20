"""Tests for the Maya-docs scraper's pure parsing functions (no network)."""
from pathlib import Path

from codemaya.data.scrape_maya_docs import parse_command_html, parse_index

FIXTURE = Path(__file__).parent / "fixtures" / "sample_command.html"


def test_parse_command_html_extracts_fields():
    rec = parse_command_html(FIXTURE.read_text(encoding="utf-8"), "http://x/sphere.html")
    assert rec["command"] == "sphere"
    assert "NURBS sphere" in rec["synopsis"]
    assert rec["return_type"].startswith("string[]")

    # three flags, correctly mapped long/short/type
    assert len(rec["flags"]) == 3
    radius = rec["flags"][0]
    assert radius["long"] == "radius" and radius["short"] == "r" and radius["type"] == "linear"

    # two MEL examples captured verbatim-ish
    assert len(rec["examples"]) == 2
    assert 'sphere -r 5' in rec["examples"][0]


def test_parse_index_extracts_command_links():
    html = """
    <html><body>
      <a href="sphere.html">sphere</a>
      <a href="polyCube.html">polyCube</a>
      <a href="index_all.html">index</a>
      <a href="http://external.com/x.html">ext</a>
      <a href="#top">top</a>
    </body></html>
    """
    items = parse_index(html, "http://x/Commands")
    names = {i["command"] for i in items}
    assert "sphere" in names and "polyCube" in names
    # index / external / anchor links are excluded
    assert "index" not in names and "ext" not in names
    assert all(i["url"].startswith("http://x/Commands/") for i in items)
