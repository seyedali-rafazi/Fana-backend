from __future__ import annotations

from pathlib import Path

import pytest

SAMPLE_CSV = """\
message_id,customer_id,channel,received_at,answered_at,subject,body,order_ref,severity_hint
M001,C100,email,2026-10-26T08:00:00Z,2026-10-26T09:30:00Z,Where is my order,"Hi, my order BZ-10234 has not arrived yet.",BZ-10234,medium
M002,C200,app,2026-10-26T08:10:00Z,,Charged twice,I was charged twice for the same order.,,high
M003,C100,email,2026-10-26T08:45:00Z,,Where is my order,"  hi, my order   BZ-10234 has NOT arrived yet. ",,medium
M004,C300,web_form,2026-10-26T09:00:00Z,2026-10-26T09:05:00Z,Password,"I forgot my password, the reset link
does not work.",,low
"""


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("URGENT_SLA_BUSINESS_HOURS", "2")


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    path = tmp_path / "export.csv"
    path.write_text(SAMPLE_CSV, encoding="utf-8")
    return path
