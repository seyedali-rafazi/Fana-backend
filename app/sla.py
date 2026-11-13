"""Business-hours arithmetic for the support SLA."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import math
from zoneinfo import ZoneInfo

from .ingest import jalali_to_gregorian

TEHRAN_TZ = ZoneInfo("Asia/Tehran")
BUSINESS_START = time(9, 0)
BUSINESS_END = time(17, 0)
# Saturday=5, Sunday=6, Monday=0, Tuesday=1, Wednesday=2 (Thursday=3, Friday=4 are off)
BUSINESS_WEEKDAYS = frozenset({5, 6, 0, 1, 2})

JALALI_PUBLIC_HOLIDAYS_1405: tuple[tuple[int, int, int], ...] = (
    (1405, 1, 1),
    (1405, 1, 2),
    (1405, 1, 3),
    (1405, 1, 4),
    (1405, 1, 12),
    (1405, 1, 13),
    (1405, 1, 25),
    (1405, 3, 7),
    (1405, 3, 14),
    (1405, 3, 15),
    (1405, 4, 5),
    (1405, 4, 6),
    (1405, 5, 15),
    (1405, 5, 23),
    (1405, 5, 25),
    (1405, 6, 1),
    (1405, 6, 10),
    (1405, 8, 24),
    (1405, 10, 3),
    (1405, 10, 17),
    (1405, 11, 5),
    (1405, 11, 22),
    (1405, 12, 10),
    (1405, 12, 20),
    (1405, 12, 21),
    (1405, 12, 29),
)

PUBLIC_HOLIDAYS: frozenset[date] = frozenset(
    jalali_to_gregorian(y, m, d) for y, m, d in JALALI_PUBLIC_HOLIDAYS_1405
)

_SECONDS_PER_BUSINESS_DAY = (
    datetime.combine(date(2000, 1, 1), BUSINESS_END)
    - datetime.combine(date(2000, 1, 1), BUSINESS_START)
).total_seconds()


def _to_tehran(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(TEHRAN_TZ)


def is_business_day(day: datetime | date) -> bool:
    """Return True if ``day`` (in ``Asia/Tehran``) is a Sat–Wed non-holiday."""
    if isinstance(day, datetime):
        local_date = _to_tehran(day).date() if day.tzinfo is not None else day.date()
    else:
        local_date = day
    return local_date.weekday() in BUSINESS_WEEKDAYS and local_date not in PUBLIC_HOLIDAYS


def business_seconds_between(start: datetime, end: datetime) -> float:
    """Number of business seconds in the half-open interval ``[start, end)``."""
    start_local = _to_tehran(start)
    end_local = _to_tehran(end)
    if end_local <= start_local:
        return 0.0

    start_date = start_local.date()
    end_date = end_local.date()

    if start_date == end_date:
        if not is_business_day(start_date):
            return 0.0
        opens = datetime.combine(start_date, BUSINESS_START, tzinfo=TEHRAN_TZ)
        closes = datetime.combine(start_date, BUSINESS_END, tzinfo=TEHRAN_TZ)
        lo = max(opens, start_local)
        hi = min(closes, end_local)
        return max(0.0, (hi - lo).total_seconds())

    total_seconds = 0.0

    if is_business_day(start_date):
        opens_start = datetime.combine(start_date, BUSINESS_START, tzinfo=TEHRAN_TZ)
        closes_start = datetime.combine(start_date, BUSINESS_END, tzinfo=TEHRAN_TZ)
        lo = max(opens_start, start_local)
        if closes_start > lo:
            total_seconds += (closes_start - lo).total_seconds()

    full_days = (end_date - start_date).days - 1
    if full_days > 0:
        first_mid = start_date + timedelta(days=1)
        last_mid = end_date - timedelta(days=1)
        full_weeks, rem = divmod(full_days, 7)
        biz_days = full_weeks * len(BUSINESS_WEEKDAYS)
        start_wd = first_mid.weekday()
        for offset in range(rem):
            if (start_wd + offset) % 7 in BUSINESS_WEEKDAYS:
                biz_days += 1
        holidays_on_workdays = sum(
            1
            for holiday in PUBLIC_HOLIDAYS
            if first_mid <= holiday <= last_mid and holiday.weekday() in BUSINESS_WEEKDAYS
        )
        total_seconds += (biz_days - holidays_on_workdays) * _SECONDS_PER_BUSINESS_DAY

    if is_business_day(end_date):
        opens_end = datetime.combine(end_date, BUSINESS_START, tzinfo=TEHRAN_TZ)
        closes_end = datetime.combine(end_date, BUSINESS_END, tzinfo=TEHRAN_TZ)
        hi = min(closes_end, end_local)
        if hi > opens_end:
            total_seconds += (hi - opens_end).total_seconds()

    return total_seconds


def business_hours_between(start: datetime, end: datetime) -> float:
    """Number of business hours in the half-open interval ``[start, end)``."""
    return business_seconds_between(start, end) / 3600.0


def business_minutes_between(start: datetime, end: datetime) -> int:
    """Whole business minutes in the half-open interval ``[start, end)``."""
    return int(business_seconds_between(start, end) // 60)


def business_minutes_until_breach(
    received_at: datetime,
    now: datetime,
    sla_hours: float = 2.0,
) -> int:
    """Business minutes left until SLA breach; ``0`` or negative means overdue."""
    elapsed_seconds = business_seconds_between(received_at, now)
    remaining_seconds = (sla_hours * 3600.0) - elapsed_seconds
    if remaining_seconds > 0:
        return math.ceil(remaining_seconds / 60.0)
    return math.floor(remaining_seconds / 60.0)
