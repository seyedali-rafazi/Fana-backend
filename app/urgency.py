"""Urgency rule for the support dashboard."""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from .sla import business_minutes_until_breach


def is_urgent(
    severity_hint: str,
    received_at: datetime,
    answered_at: datetime | None,
    now: datetime,
    sla_hours: float,
) -> bool:
    """A message is urgent when it is high severity, still unanswered at
    ``now``, and has waited at least ``sla_hours`` business hours.

    severity_hint is assigned by the support system and is reliable.
    """
    if severity_hint != "high":
        return False
    if answered_at is not None and not pd.isna(answered_at) and answered_at <= now:
        return False
    return business_minutes_until_breach(received_at, now, sla_hours) <= 0
