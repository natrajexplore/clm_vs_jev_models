import json

import httpx
import pytest

from clm_jev_model_comparison.models.jev import JevClient

CRITERIA = {"returns": "Exchanges, wrong or damaged items", "billing": "Charges, invoices"}


def _ok_response(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    state = body["state"]
    choice = "returns" if "size" in state else "billing"
    other = "billing" if choice == "returns" else "returns"
    return httpx.Response(200, json={
        "model": "jev-1.13.0",
        "answers": {"label": {"type": "choice", "choice": choice, "confidence": 0.8,
                              "probabilities": {choice: 0.9, other: 0.1}, "usage": {"input_tokens": 42}}},
    })


def _client(handler, tmp_path=None, **kw) -> JevClient:
    return JevClient(model="jev-latest", base_url="https://api.test/v1/systemone", backoff_s=0.0,
                     cache_path=tmp_path / "cache.jsonl" if tmp_path else None,
                     transport=httpx.MockTransport(handler), **kw)


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")


def test_request_shape_and_parsing():
    seen = []

    def handler(request):
        seen.append(request)
        return _ok_response(request)

    answers = _client(handler).choice(["wrong size shoes", "double charged"], "Which team?", CRITERIA)
    assert [a.choice for a in answers] == ["returns", "billing"]  # input order preserved
    assert answers[0].probabilities == {"returns": 0.9, "billing": 0.1}
    assert answers[0].input_tokens == 42 and answers[0].model == "jev-1.13.0"

    body = json.loads(seen[0].content)
    assert seen[0].headers["authorization"] == "Bearer test-key"
    assert body["model"] == "jev-latest"
    assert body["questions"]["label"] == {"type": "choice", "instructions": "Which team?", "criteria": CRITERIA}


def test_retries_on_rate_limit():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(429) if calls["n"] < 3 else _ok_response(request)

    [answer] = _client(handler).choice(["wrong size"], "Which team?", CRITERIA)
    assert answer.choice == "returns" and calls["n"] == 3


def test_non_retryable_error_raises():
    with pytest.raises(RuntimeError, match="422"):
        _client(lambda r: httpx.Response(422, text="bad")).choice(["x"], "q", CRITERIA)


def test_cache_avoids_second_request(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return _ok_response(request)

    first = _client(handler, tmp_path).choice(["wrong size"], "q", CRITERIA)
    second = _client(handler, tmp_path).choice(["wrong size"], "q", CRITERIA)
    assert calls["n"] == 1
    assert not first[0].cached and second[0].cached
    assert second[0].latency_s == first[0].latency_s


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY")
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        _client(_ok_response)
