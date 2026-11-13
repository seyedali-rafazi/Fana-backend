"""Loading the support-system CSV export into a DataFrame."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path
import re
from typing import IO
from zoneinfo import ZoneInfo

import pandas as pd

from .text import normalize_digits

COLUMNS: list[str] = [
    "message_id",
    "customer_id",
    "channel",
    "received_at",
    "answered_at",
    "subject",
    "body",
    "order_ref",
    "severity_hint",
]

_TEHRAN_TZ = ZoneInfo("Asia/Tehran")
_JALALI_PATTERN = re.compile(
    r"^\s*(\d{4})/(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?"
    r"(?:\s*([Zz]|[+-]\d{2}:?\d{2}))?\s*$"
)


def _jalali_to_gregorian_raw(jy: int, jm: int, jd: int) -> date:
    jy += 1595
    days = (
        -355668
        + (365 * jy)
        + ((jy // 33) * 8)
        + (((jy % 33) + 3) // 4)
        + jd
        + ((jm - 1) * 31 if jm < 7 else ((jm - 7) * 30) + 186)
    )
    gy = 400 * (days // 146097)
    days %= 146097
    if days > 36524:
        days -= 1
        gy += 100 * (days // 36524)
        days %= 36524
        if days >= 365:
            days += 1
    gy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        gy += (days - 1) // 365
        days = (days - 1) % 365
    gd = days + 1
    leap = (gy % 4 == 0 and gy % 100 != 0) or (gy % 400 == 0)
    month_days = (0, 31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    gm = 0
    for gm in range(1, 13):
        days_in_month = month_days[gm]
        if gd <= days_in_month:
            break
        gd -= days_in_month
    return date(gy, gm, gd)


def is_jalali_leap(year: int) -> bool:
    """Return True if ``year`` is a leap year in the Solar Hijri calendar."""
    if year < 1:
        raise ValueError(f"invalid Jalali year: {year}")
    return _jalali_to_gregorian_raw(year, 12, 30) < _jalali_to_gregorian_raw(year + 1, 1, 1)


def jalali_to_gregorian(year: int, month: int, day: int) -> date:
    """Convert a validated Jalali (Solar Hijri) date to a Gregorian ``date``."""
    if year < 1:
        raise ValueError(f"invalid Jalali year: {year}")
    if not 1 <= month <= 12:
        raise ValueError(f"invalid Jalali month: {month}")
    if month <= 6:
        max_day = 31
    elif month <= 11:
        max_day = 30
    else:
        max_day = 30 if is_jalali_leap(year) else 29
    if not 1 <= day <= max_day:
        raise ValueError(f"invalid Jalali day {day} for {year}/{month:02d}")
    return _jalali_to_gregorian_raw(year, month, day)


def _parse_offset(offset: str | None) -> tzinfo:
    if not offset:
        return _TEHRAN_TZ
    if offset.upper() == "Z":
        return timezone.utc
    compact = offset.replace(":", "")
    sign = 1 if compact[0] == "+" else -1
    hours = int(compact[1:3])
    minutes = int(compact[3:5])
    if hours > 23 or minutes > 59:
        raise ValueError(f"invalid UTC offset: {offset}")
    return timezone(timedelta(minutes=sign * (hours * 60 + minutes)))


def parse_jalali_timestamp(value: str) -> pd.Timestamp:
    """Parse a Jalali timestamp string into a tz-aware UTC ``pd.Timestamp``."""
    normalized = normalize_digits(value).strip()
    match = _JALALI_PATTERN.match(normalized)
    if match is None:
        raise ValueError(f"invalid Jalali timestamp: {value!r}")

    year = int(match.group(1))
    month = int(match.group(2))
    day = int(match.group(3))
    hour = int(match.group(4))
    minute = int(match.group(5))
    second = int(match.group(6) or 0)
    offset = match.group(7)

    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        raise ValueError(f"invalid time in Jalali timestamp: {value!r}")

    greg_date = jalali_to_gregorian(year, month, day)
    tz = _parse_offset(offset)
    dt = datetime(
        greg_date.year,
        greg_date.month,
        greg_date.day,
        hour,
        minute,
        second,
        tzinfo=tz,
    ).astimezone(timezone.utc)
    return pd.Timestamp(dt)


def _try_parse_jalali(value: str) -> pd.Timestamp | pd._libs.tslibs.nattype.NaTType:
    try:
        return parse_jalali_timestamp(value)
    except ValueError:
        return pd.NaT


def parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse ISO-8601 and Jalali timestamps into tz-aware UTC values.

    Jalali timestamps without an explicit offset are interpreted as
    ``Asia/Tehran`` local time and converted to UTC. Invalid or blank entries
    coerce to ``NaT``.
    """
    if len(values) == 0:
        return pd.Series([], index=values.index, dtype="datetime64[ns, UTC]")

    cleaned = values.where(values.notna(), "").astype(str).map(lambda v: normalize_digits(v).strip())
    is_jalali = cleaned.str.match(_JALALI_PATTERN)

    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
    if (~is_jalali).any():
        result.loc[~is_jalali] = pd.to_datetime(
            cleaned[~is_jalali], errors="coerce", utc=True, format="ISO8601"
        )
    if is_jalali.any():
        result.loc[is_jalali] = pd.to_datetime(
            cleaned[is_jalali].map(_try_parse_jalali), errors="coerce", utc=True
        )
    return result


def load_messages(source: str | Path | IO[bytes] | IO[str]) -> pd.DataFrame:
    """Read a support export and return one row per message, in file order."""
    df = pd.read_csv(source, dtype=str, keep_default_na=False)

    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"export is missing columns: {', '.join(missing)}")

    df = df[COLUMNS].fillna("")
    for col in ("message_id", "customer_id", "channel", "order_ref", "severity_hint"):
        df[col] = df[col].str.strip()
    df["severity_hint"] = df["severity_hint"].str.lower()

    df = df[df["message_id"] != ""]

    df["received_at"] = parse_timestamps(df["received_at"])
    df["answered_at"] = parse_timestamps(df["answered_at"])

    # discard malformed rows
    df = df.dropna(subset=["received_at"])

    return df.reset_index(drop=True)
