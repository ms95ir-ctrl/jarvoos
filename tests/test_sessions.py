import json
from pathlib import Path

from jarvoos.sessions import SessionStore


def msgs(*pairs):
    return [{"role": r, "content": c} for r, c in pairs]


def test_append_load_roundtrip(tmp_path):
    s = SessionStore(tmp_path)
    meta = s.new(Path("/proj"), "local", "qwen")
    for m in msgs(("user", "hello there\nsecond line"), ("assistant", "hi")):
        s.append(meta.id, m)
    loaded, history = s.load(meta.id)
    assert history == msgs(("user", "hello there\nsecond line"), ("assistant", "hi"))
    assert loaded.title == "hello there" and loaded.turns == 1
    assert (loaded.cwd, loaded.provider, loaded.model) == ("/proj", "local", "qwen")


def test_auto_title_truncated_and_manual_title_kept(tmp_path):
    s = SessionStore(tmp_path)
    a = s.new(Path("/p"), "l", "m")
    s.append(a.id, {"role": "user", "content": "x" * 200})
    assert len(s.load(a.id)[0].title) <= 60
    s.set_title(a.id, "mine")
    s.append(a.id, {"role": "user", "content": "other"})
    assert s.load(a.id)[0].title == "mine"


def test_truncated_last_line_is_tolerated_and_repaired(tmp_path):
    s = SessionStore(tmp_path)
    meta = s.new(Path("/p"), "l", "m")
    s.append(meta.id, {"role": "user", "content": "one"})
    path = tmp_path / f"{meta.id}.jsonl"
    with open(path, "ab") as f:
        f.write(b'{"role": "assistant", "con')
    assert s.load(meta.id)[1] == [{"role": "user", "content": "one"}]
    s.append(meta.id, {"role": "assistant", "content": "two"})
    assert [m["content"] for m in s.load(meta.id)[1]] == ["one", "two"]
    assert all(json.loads(line) for line in path.read_text().splitlines()[:1])


def test_rewrite_replaces_messages_and_keeps_header(tmp_path):
    s = SessionStore(tmp_path)
    meta = s.new(Path("/p"), "l", "m")
    for i in range(4):
        s.append(meta.id, {"role": "user", "content": f"m{i}"})
    s.rewrite(meta.id, msgs(("user", "summary")))
    loaded, history = s.load(meta.id)
    assert history == msgs(("user", "summary")) and loaded.cwd == "/p"


def test_latest_list_prune_and_resolve(tmp_path):
    s = SessionStore(tmp_path)
    ids = []
    for i, cwd in enumerate(["/a", "/b", "/a"]):
        m = s.new(Path(cwd), "l", "m")
        s.append(m.id, {"role": "user", "content": f"t{i}"})
        ids.append(m.id)
        import os
        os.utime(tmp_path / f"{m.id}.jsonl", (1000 + i, 1000 + i))
    assert s.latest() == ids[2]
    assert s.latest(Path("/b")) == ids[1]
    assert s.latest(Path("/zzz")) is None
    assert [m.id for m in s.list(2)] == [ids[2], ids[1]]
    assert s.resolve(ids[0][:12]) in (ids[0], None)
    s.prune(keep=1)
    assert [m.id for m in s.list(10)] == [ids[2]]


def test_empty_sessions_are_not_listed_or_resumed(tmp_path):
    s = SessionStore(tmp_path)
    s.new(Path("/p"), "l", "m")
    assert s.list() == [] and s.latest() is None
