import httpx
import pytest

from jarvoos import detect


def fake_servers(monkeypatch, servers: dict, other=None):
    """servers: {'127.0.0.1:8090': {'/health': (200, {...})}}. Unknown hosts refuse; unknown paths 404."""
    def handler(req: httpx.Request) -> httpx.Response:
        port = req.url.port or (443 if req.url.scheme == "https" else 80)
        routes = servers.get(f"{req.url.host}:{port}")
        if other and req.url.host == "openrouter.ai":
            routes = other
        if routes is None:
            raise httpx.ConnectError("refused", request=req)
        hit = routes.get(req.url.path)
        if callable(hit):
            hit = hit(req)
        if hit is None:
            return httpx.Response(404, json={"error": "not found"})
        return httpx.Response(hit[0], json=hit[1])
    monkeypatch.setattr(detect, "TRANSPORT", httpx.MockTransport(handler))


STRATA = {"/health": (200, {"service": "strata", "max_context": 131072, "api_key": False}),
          "/v1/models": (200, {"data": [{"id": "qwen3-30b"}]})}
LLAMA = {"/health": (200, {"status": "ok"}),
         "/props": (200, {"default_generation_settings": {"n_ctx": 32768}, "total_slots": 4}),
         "/v1/models": (200, {"data": [{"id": "gguf-model"}]})}
OLLAMA = {"/api/tags": (200, {"models": [{"name": "llama3.1:8b"}]}), "/v1/models": (200, {"data": [{"id": "llama3.1:8b"}]})}
LMSTUDIO = {"/api/v0/models": (200, {"data": [
    {"id": "embed", "type": "embeddings", "state": "loaded", "loaded_context_length": 512},
    {"id": "gemma", "type": "llm", "state": "loaded", "loaded_context_length": 16384, "max_context_length": 131072},
    {"id": "other", "type": "llm", "state": "not-loaded", "max_context_length": 8192}]}),
    "/v1/models": (200, {"data": []})}


def test_scan_fingerprints_every_backend(monkeypatch):
    monkeypatch.delenv("JARVOOS_DETECT_HOSTS", raising=False)
    fake_servers(monkeypatch, {"127.0.0.1:8090": STRATA, "127.0.0.1:8081": LLAMA, "127.0.0.1:11434": OLLAMA, "127.0.0.1:1234": LMSTUDIO})
    found = {f.kind: f for f in detect.scan()}
    assert set(found) == {"strata", "llamacpp", "ollama", "lmstudio"}
    s = found["strata"]
    assert (s.base_url, s.models, s.n_ctx, s.needs_key) == ("http://127.0.0.1:8090/v1", ["qwen3-30b"], 131072, False)
    assert found["llamacpp"].n_ctx == 8192 and found["llamacpp"].models == ["gguf-model"]
    assert found["ollama"].models == ["llama3.1:8b"] and "4096" in found["ollama"].label
    lm = found["lmstudio"]
    assert lm.models == ["gemma", "other"] and lm.n_ctx == 16384 and lm.base_url == "http://127.0.0.1:1234/v1"


def test_scan_order_strata_first_and_nothing_found(monkeypatch):
    monkeypatch.delenv("JARVOOS_DETECT_HOSTS", raising=False)
    fake_servers(monkeypatch, {"127.0.0.1:8081": LLAMA, "127.0.0.1:8090": STRATA})
    assert [f.kind for f in detect.scan()] == ["strata", "llamacpp"]
    fake_servers(monkeypatch, {})
    assert detect.scan() == []


def test_llamacpp_loading_and_strata_key(monkeypatch):
    monkeypatch.delenv("JARVOOS_DETECT_HOSTS", raising=False)
    fake_servers(monkeypatch, {"127.0.0.1:8080": {"/health": (503, {"error": {"code": 503, "message": "Loading model"}})},
                               "127.0.0.1:8090": {"/health": (200, {"service": "strata", "max_context": 4096, "api_key": True})}})
    found = {f.kind: f for f in detect.scan()}
    assert found["llamacpp"].loading and found["llamacpp"].models == []
    assert found["strata"].needs_key


def test_extra_urls_and_env_hosts(monkeypatch):
    monkeypatch.setenv("JARVOOS_DETECT_HOSTS", "10.0.0.5")
    fake_servers(monkeypatch, {"10.0.0.5:8090": STRATA, "192.168.1.9:8000": {"/v1/models": (200, {"data": [{"id": "x", "meta": {"n_ctx": 2048}}]})}})
    found = detect.scan(["http://192.168.1.9:8000/v1"])
    by = {f.base_url: f for f in found}
    assert by["http://10.0.0.5:8090/v1"].kind == "strata"
    g = by["http://192.168.1.9:8000/v1"]
    assert g.kind == "openai" and g.n_ctx == 2048 and not g.needs_key


def test_probe_url_toggles_v1(monkeypatch):
    fake_servers(monkeypatch, {"127.0.0.1:8090": STRATA})
    up = detect.probe_url("http://127.0.0.1:8090")
    assert up.ok and up.base_url == "http://127.0.0.1:8090/v1" and up.kind == "strata" and up.n_ctx == 131072 and up.models == ["qwen3-30b"]
    # a /v1 URL on a server that serves /models at the root
    fake_servers(monkeypatch, {"h:9": {"/models": (200, {"data": [{"id": "m"}]})}})
    up = detect.probe_url("http://h:9/v1")
    assert up.ok and up.base_url == "http://h:9" and up.models == ["m"]


def test_probe_url_failure_kinds(monkeypatch):
    fake_servers(monkeypatch, {"127.0.0.1:8090": {"/v1/models": (401, {"error": "no"})}, "127.0.0.1:7": {}})
    assert detect.probe_url("http://127.0.0.1:1/v1").status == "refused"
    assert "running" in detect.probe_url("http://127.0.0.1:1/v1").hint
    up = detect.probe_url("http://127.0.0.1:8090/v1")
    assert (up.ok, up.status, up.needs_key) == (False, "unauthorized", True)
    up = detect.probe_url("http://127.0.0.1:7/v1")
    assert up.status == "not_found" and "404" in up.hint or "/models" in up.hint
    assert detect.probe_url("not a url ://").status == "bad_url"
    assert detect.probe_url("ftp://x/v1").status == "bad_url"


def test_probe_url_sends_key_and_loading(monkeypatch):
    seen = {}
    def models(req):
        seen["auth"] = req.headers.get("authorization")
        return 200, {"data": [{"id": "m"}]}
    fake_servers(monkeypatch, {"s:1": {"/v1/models": models}, "l:1": {"/v1/models": (503, {"error": {"message": "Loading model"}})}})
    assert detect.probe_url("http://s:1/v1", key="k123").ok and seen["auth"] == "Bearer k123"
    assert detect.probe_url("http://l:1/v1").status == "loading"


def test_openrouter_check_key(monkeypatch):
    def auth(req):
        return (200, {"data": {"label": "sk-or-v1-abc"}}) if req.headers.get("authorization") == "Bearer good" else (401, {"error": {"message": "User not found."}})
    fake_servers(monkeypatch, {}, other={"/api/v1/auth/key": auth})
    assert detect.openrouter_check_key("good")[0] is True
    ok, msg = detect.openrouter_check_key("bad")
    assert not ok and "rejected" in msg
    fake_servers(monkeypatch, {})
    ok, msg = detect.openrouter_check_key("good")
    assert not ok and msg.startswith("unreachable")


def test_openrouter_models_filter_cache_and_stale(monkeypatch):
    calls = []
    payload = {"data": [
        {"id": "b/tools", "context_length": 128000, "pricing": {"prompt": "0.0000003", "completion": "0.0000012"}, "supported_parameters": ["tools", "temperature"]},
        {"id": "a/notools", "context_length": 8000, "pricing": {"prompt": "0", "completion": "0"}, "supported_parameters": ["temperature"]},
        {"id": "c/auto", "context_length": 0, "pricing": {"prompt": "-1", "completion": "-1"}, "supported_parameters": ["tools"]}]}
    def models(req):
        calls.append(1)
        return 200, payload
    monkeypatch.setattr(detect, "_cache", {"t": 0.0, "models": None})
    fake_servers(monkeypatch, {}, other={"/api/v1/models": models})
    ms = detect.openrouter_models()
    assert [m.id for m in ms] == ["b/tools", "c/auto"]
    assert ms[0].context == 128000 and ms[0].price_in == pytest.approx(0.3) and ms[0].price_out == pytest.approx(1.2)
    assert ms[1].price_in == 0
    assert len(detect.openrouter_models(tools_only=False)) == 3 and len(calls) == 1  # cached
    # expired + network down -> stale list
    detect._cache["t"] -= detect.MODELS_TTL_S + 1
    fake_servers(monkeypatch, {})
    assert [m.id for m in detect.openrouter_models()] == ["b/tools", "c/auto"]
    # nothing cached and network down -> empty
    monkeypatch.setattr(detect, "_cache", {"t": 0.0, "models": None})
    assert detect.openrouter_models() == []
