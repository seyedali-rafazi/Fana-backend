"""Runtime configuration, read from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass

CATEGORIES: tuple[str, ...] = (
    "delivery",
    "payment",
    "refund",
    "product_quality",
    "account",
    "other",
)


@dataclass(frozen=True)
class Settings:
    urgent_sla_business_hours: float
    llm_base_url: str | None
    llm_api_key: str | None
    llm_model: str
    llm_timeout_seconds: float


def _optional(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def get_settings() -> Settings:
    return Settings(
        urgent_sla_business_hours=float(os.environ.get("URGENT_SLA_BUSINESS_HOURS", "2")),
        llm_base_url=_optional("LLM_BASE_URL"),
        llm_api_key=_optional("LLM_API_KEY"),
        llm_model=os.environ.get("LLM_MODEL", "").strip() or "gpt-4o-mini",
        llm_timeout_seconds=float(os.environ.get("LLM_TIMEOUT_SECONDS", "30")),
    )
