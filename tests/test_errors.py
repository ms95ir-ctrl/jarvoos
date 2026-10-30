import json

import pytest

from jarvoos.config import ProviderSpec
from jarvoos.errors import ProviderError, classify_http, friendly, from_stream_error, should_failover

OR = ProviderSpec("openrouter", "openrouter", "https://openrouter.ai/api/v1", "deepseek/deepseek-v4-flash", "OPENROUTER_API_KEY")
STRATA = ProviderSpec("local", "strata", "http://127.0.0.1:8090/v1", "qwen3")
LLAMA = ProviderSpec("llama", "llamacpp", "http://127.0.0.1:8081/v1", "m")


def body(msg, **extra):
    return json.dumps({"error": {"message": msg, **extra}})


CLASSIFY = [
    (401, "bad key", "auth", True),
    (403, "forbidden", "auth", True),
    (402, "insufficient credits", "billing", True),
    (404, "no such model", "not_found", True),
    (408, "slow", "timeout", True),
    (429, "slow down", "rate_limit", True),
    (500, "oops", "server", True),
    (502, "bad gateway", "server", True),
    (503, "loading", "server", True),
    (400, "maximum context length is 4096 tokens", "context", False),
    (400, "prompt is too long: 9000 tokens", "context", False),
    (400, "request exceeds the available context size (n_ctx)", "context", False),
    (413, "too big", "context", False),
    (400, "unknown parameter foo", "bad_request", False),
    (422, "invalid", "bad_request", False),
]


@pytest.mark.parametrize("status,msg,kind,failover", CLASSIFY)
def test_classify_http(status, msg, kind, failover):
    e = classify_http(status, body(msg), "p", "http://x/v1")
    assert (e.kind, e.status, e.provider, e.url) == (kind, status, "p", "http://x/v1")
    assert should_failover(e) is failover and e.fatal_for_failover is (not failover)
    assert msg in e.raw


def test_retry_after_header_and_openrouter_metadata_are_clamped():
    assert classify_http(429, body("x"), "p", "u", {"retry-after": "7"}).retry_after == 7
    assert classify_http(429, body("x"), "p", "u", {"retry-after": "9999"}).retry_after == 60
    assert classify_http(429, body("x", metadata={"retry_after_seconds": 12}), "p", "u").retry_after == 12
    assert classify_http(429, body("x", metadata={"retry_after_seconds": 1e6}), "p", "u").retry_after == 60
    assert classify_http(429, body("x"), "p", "u", {"retry-after": "Wed, 21 Oct"}).retry_after is None


def test_non_json_body_is_kept_as_raw():
    e = classify_http(502, "<html>Bad gateway</html>", "p", "u")
    assert e.kind == "server" and "Bad gateway" in e.raw


def test_stream_error_chunk():
    e = from_stream_error({"code": 429, "message": "limited"}, "p", "u")
    assert e.kind == "rate_limit" and e.status == 429
    e = from_stream_error({"message": "the context window was exceeded"}, "p", "u")
    assert e.kind == "context"
    e = from_stream_error("upstream died", "p", "u")
    assert e.kind == "server" and "upstream died" in e.raw


def test_other_kinds_do_not_fail_over_or_do():
    for k in ("network", "timeout", "server", "rate_limit", "auth", "billing", "not_found", "empty"):
        assert should_failover(ProviderError(k))
    for k in ("context", "bad_request", "not_configured"):
        assert not should_failover(ProviderError(k))


def mk(kind, **kw):
    return ProviderError(kind, kw.pop("m", "m"), url=kw.pop("url", "http://127.0.0.1:8090/v1"), **kw)


FRIENDLY = [
    (mk("network"), STRATA, ["Local server at http://127.0.0.1:8090/v1 isn't answering", "Strata", "jarvoos doctor"]),
    (mk("network", url="http://127.0.0.1:8081/v1"), LLAMA, ["llama-server running", "jarvoos doctor"]),
    (mk("network", url="https://openrouter.ai/api/v1"), OR, ["Can't reach OpenRouter", "internet"]),
    (mk("timeout"), STRATA, ["stopped responding", "/model"]),
    (mk("auth"), OR, ["rejected the API key", "https://openrouter.ai/keys", "jarvoos setup"]),
    (mk("auth"), STRATA, ["rejected the API key"]),
    (mk("billing"), OR, ["out of credits", "https://openrouter.ai/settings/credits"]),
    (mk("rate_limit", retry_after=12), OR, ["rate limiting", "12s"]),
    (mk("server", status=503), OR, ["internal error", "503"]),
    (mk("context"), STRATA, ["context window", "/compress"]),
    (mk("not_found"), STRATA, ["no model 'qwen3'", "/v1"]),
    (mk("empty", raw="no answer (finish=length), 400 chars of reasoning"), STRATA, ["only thought", "thinking off"]),
    (mk("empty", raw="no answer (finish=stop), 0 chars of reasoning"), STRATA, ["empty reply"]),
    (mk("bad_request"), STRATA, ["rejected the request"]),
    (mk("not_configured"), STRATA, ["jarvoos setup"]),
]


@pytest.mark.parametrize("err,spec,needles", FRIENDLY, ids=[f"{e.kind}-{s.kind}-{i}" for i, (e, s, _) in enumerate(FRIENDLY)])
def test_friendly(err, spec, needles):
    msg, details = friendly(err, spec)
    for n in needles:
        assert n in msg
    assert "\n" not in msg


def test_friendly_details_are_truncated_and_single_line():
    e = ProviderError("server", "x", raw="line one\nline two " + "z" * 1000)
    _, details = friendly(e, STRATA)
    assert len(details) <= 303 and "\n" not in details and details.startswith("line one line two")
