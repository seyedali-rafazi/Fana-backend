from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
import httpx
import pytest

from app import drafts, llm
from app.analysis import analyze, parse_now
from app.api import app
from app.classify import clear_classification_cache
from app.config import Settings
from app.llm import CLASSIFY_BATCH_SIZE, LLMClient, MessageToClassify


client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_classify_cache() -> None:
    clear_classification_cache()
    yield
    clear_classification_cache()


def _llm_settings(model: str = "test-model") -> Settings:
    return Settings(
        urgent_sla_business_hours=2.0,
        llm_base_url="http://llm.test/v1",
        llm_api_key="test-key",
        llm_model=model,
        llm_timeout_seconds=5.0,
    )


def test_retry_after_429_succeeds_on_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def _fake_post(self: httpx.Client, url: str, **kwargs: object) -> httpx.Response:
        nonlocal calls
        calls += 1
        req = httpx.Request("POST", url)
        if calls == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "0.1"},
                json={"error": {"message": "Rate limit reached"}},
                request=req,
            )
        content = json.dumps(
            [
                {"id": "M001", "category": "delivery", "severity": "medium"},
                {"id": "M002", "category": "payment", "severity": "high"},
            ]
        )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
            request=req,
        )

    monkeypatch.setattr(httpx.Client, "post", _fake_post)
    llm_client = LLMClient(_llm_settings(), inter_batch_pause=0.0)
    results = llm_client.classify(
        [
            MessageToClassify("M001", "Order", "Where is my order"),
            MessageToClassify("M002", "Payment", "Charged twice"),
        ]
    )

    assert calls == 2
    assert results["M001"].category == "delivery"
    assert results["M001"].severity == "medium"
    assert results["M002"].category == "payment"
    assert results["M002"].severity == "high"


@pytest.mark.parametrize("status_code", [429, 503])
def test_persistent_http_error_falls_back_cleanly_on_api_analyze(
    sample_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)

    calls = 0
    orig_post = httpx.Client.post

    def _always_fail(self: httpx.Client, url: object, **kwargs: object) -> httpx.Response:
        if "llm.test" not in str(url):
            return orig_post(self, url, **kwargs)  # type: ignore[arg-type]
        nonlocal calls
        calls += 1
        return httpx.Response(
            status_code,
            headers={"Retry-After": "0.05"},
            json={"error": {"message": "Service temporarily throttled"}},
            request=httpx.Request("POST", str(url)),
        )

    monkeypatch.setattr(httpx.Client, "post", _always_fail)

    with sample_csv.open("rb") as fh:
        response = client.post(
            "/api/analyze",
            params={"now": "2026-10-26T12:00:00Z"},
            files={"file": ("export.csv", fh, "text/csv")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["total_unique_messages"] == 3
    by_id = {m["message_id"]: m for m in body["messages"]}
    # Falls back to keyword classifier instead of crashing or collapsing all to "other"
    assert by_id["M001"]["category"] == "delivery"
    assert by_id["M002"]["category"] == "payment"
    assert by_id["M004"]["category"] == "account"
    assert calls >= 1


def test_partial_batch_failure_preserves_succeeded_and_falls_back_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Build enough unique rows to span 2 batches (CLASSIFY_BATCH_SIZE + 2)
    total_rows = CLASSIFY_BATCH_SIZE + 2
    lines = ["message_id,customer_id,channel,received_at,answered_at,subject,body,order_ref,severity_hint"]
    for i in range(1, total_rows + 1):
        # Batch 1 messages mention delivery/refund; Batch 2 mention account/payment keywords
        if i <= CLASSIFY_BATCH_SIZE:
            lines.append(f"M{i:03d},C{i},email,2026-10-26T08:00:00Z,,Issue {i},General inquiry #{i},,low")
        else:
            lines.append(f"M{i:03d},C{i},email,2026-10-26T08:00:00Z,,Password {i},I forgot my password for login,,low")

    csv_path = tmp_path / "two_batches.csv"
    csv_path.write_text("\n".join(lines), encoding="utf-8")

    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    batch_attempts = 0

    def _first_ok_second_fails(self: httpx.Client, url: str, **kwargs: object) -> httpx.Response:
        nonlocal batch_attempts
        batch_attempts += 1
        req = httpx.Request("POST", url)
        if batch_attempts == 1:
            # Succeeded batch returns LLM category "product_quality" (non-keyword!) so we prove LLM result is kept
            payload = kwargs.get("json") or {}
            user_content = payload["messages"][1]["content"]  # type: ignore[index]
            items = json.loads(user_content)
            reply = [{"id": item["id"], "category": "product_quality", "severity": "medium"} for item in items]
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": json.dumps(reply)}}]},
                request=req,
            )
        return httpx.Response(
            429,
            headers={"Retry-After": "0.1"},
            json={"error": {"message": "Rate limit exceeded"}},
            request=req,
        )

    monkeypatch.setattr(httpx.Client, "post", _first_ok_second_fails)

    report = analyze(csv_path, now=parse_now("2026-10-26T10:00:00Z"), settings=_llm_settings())
    by_id = {m["message_id"]: m for m in report["messages"]}

    # Batch 1 preserved LLM's product_quality classification
    assert by_id["M001"]["category"] == "product_quality"
    assert by_id[f"M{CLASSIFY_BATCH_SIZE:03d}"]["category"] == "product_quality"
    # Batch 2 fell back to keyword classification ("account" for "password/login")
    assert by_id[f"M{CLASSIFY_BATCH_SIZE + 1:03d}"]["category"] == "account"
    assert by_id[f"M{total_rows:03d}"]["category"] == "account"

    # Second analyze call: Batch 1 is cached -> ONLY Batch 2 is retried (now 1st post fails? make it succeed!)
    calls_before_recovery = batch_attempts
    recovery_calls = 0

    def _recover_remaining(self: httpx.Client, url: str, **kwargs: object) -> httpx.Response:
        nonlocal recovery_calls
        recovery_calls += 1
        payload = kwargs.get("json") or {}
        items = json.loads(payload["messages"][1]["content"])  # type: ignore[index]
        # Only failed Batch 2 items should be sent (not the cached Batch 1 items!)
        assert len(items) == 2
        reply = [{"id": item["id"], "category": "other", "severity": "low"} for item in items]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(reply)}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.Client, "post", _recover_remaining)
    report2 = analyze(csv_path, now=parse_now("2026-10-26T10:00:00Z"), settings=_llm_settings())
    by_id2 = {m["message_id"]: m for m in report2["messages"]}
    assert calls_before_recovery >= 2
    assert recovery_calls == 1
    # Cached batch kept product_quality; previously-failed batch wasn't poisoned by fallback cache!
    assert by_id2["M001"]["category"] == "product_quality"
    assert by_id2[f"M{total_rows:03d}"]["category"] == "other"


def test_cache_hits_cost_zero_http_requests(
    sample_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    http_calls = 0

    def _ok_post(self: httpx.Client, url: str, **kwargs: object) -> httpx.Response:
        nonlocal http_calls
        http_calls += 1
        payload = kwargs.get("json") or {}
        items = json.loads(payload["messages"][1]["content"])  # type: ignore[index]
        reply = [{"id": item["id"], "category": "refund", "severity": "low"} for item in items]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(reply)}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.Client, "post", _ok_post)
    settings = _llm_settings()

    first = analyze(sample_csv, now=parse_now("2026-10-26T10:00:00Z"), settings=settings)
    assert http_calls == 1
    assert first["category_counts"].get("refund") == 3

    second = analyze(sample_csv, now=parse_now("2026-10-26T10:00:00Z"), settings=settings)
    assert http_calls == 1  # Zero additional HTTP requests on repeat analyze!
    assert second["category_counts"].get("refund") == 3


def test_api_analyze_does_not_prefetch_drafts(
    sample_csv: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefetch_called = False

    def _fail_if_called(*args: object, **kwargs: object) -> None:
        nonlocal prefetch_called
        prefetch_called = True
        raise AssertionError("drafts.prefetch must NOT be called during /api/analyze")

    monkeypatch.setattr(drafts, "prefetch", _fail_if_called)
    monkeypatch.setattr("app.analysis.prefetch", _fail_if_called)

    with sample_csv.open("rb") as fh:
        response = client.post(
            "/api/analyze",
            params={"now": "2026-10-26T12:00:00Z"},
            files={"file": ("export.csv", fh, "text/csv")},
        )

    assert response.status_code == 200
    assert prefetch_called is False


def test_vercel_and_env_wall_seconds_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VERCEL", raising=False)
    monkeypatch.delenv("LLM_CLASSIFY_WALL_SECONDS", raising=False)
    assert LLMClient(_llm_settings())._max_wall_seconds == llm.MAX_CLASSIFY_WALL_SECONDS

    monkeypatch.setenv("VERCEL", "1")
    assert LLMClient(_llm_settings())._max_wall_seconds == llm.VERCEL_CLASSIFY_WALL_SECONDS

    monkeypatch.setenv("LLM_CLASSIFY_WALL_SECONDS", "25")
    assert LLMClient(_llm_settings())._max_wall_seconds == 25.0
