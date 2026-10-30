import pytest

from jarvoos import skills


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVOOS_HOME", str(tmp_path))


def test_parse_normal():
    meta, body = skills.parse("---\nname: x\ndescription: does x\n---\nBody here\n")
    assert meta == {"name": "x", "description": "does x"} and body == "Body here"


def test_parse_missing_closing_marker_does_not_crash_or_eat_the_body():
    meta, body = skills.parse("---\nname: x\ndescription: does x\n\nStep 1\nStep 2")
    assert meta["name"] == "x" and body == "Step 1\nStep 2"
    meta, body = skills.parse("---\nname: only-header")
    assert meta == {"name": "only-header"} and body == ""


def test_parse_no_frontmatter_crlf_bom_and_quotes():
    assert skills.parse("just text") == ({}, "just text")
    meta, body = skills.parse('﻿---\r\nname: "q"\r\ndescription: \'d\'\r\n---\r\nB\r\n')
    assert meta == {"name": "q", "description": "d"} and body == "B"
    assert skills.parse("") == ({}, "")


def test_bundled_skill_loads_with_source():
    s = skills.get("web-research")
    assert s and s.source == "bundled" and "web_search" in s.body and s.description


def test_user_skill_overrides_bundled_and_falls_back_on_name(tmp_path):
    d = tmp_path / "skills"
    d.mkdir()
    (d / "web-research.md").write_text("---\nname: web-research\ndescription: mine\n---\ncustom")
    (d / "notes.md").write_text("First line is the description\nmore")
    got = {s.name: s for s in skills.load_all()}
    assert got["web-research"].source == "user" and got["web-research"].body == "custom"
    assert got["notes"].description == "First line is the description" and got["notes"].source == "user"
    assert skills.get("nope") is None
    assert "- web-research: mine" in skills.index()
