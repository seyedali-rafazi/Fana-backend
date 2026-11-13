from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from app.sla import (
    BUSINESS_END,
    BUSINESS_START,
    BUSINESS_WEEKDAYS,
    PUBLIC_HOLIDAYS,
    TEHRAN_TZ,
    business_hours_between,
    business_minutes_between,
    business_minutes_until_breach,
    is_business_day,
)
from app.urgency import is_urgent


def _utc(year: int, month: int, day: int, hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)


def _tehran(year: int, month: int, day: int, hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=TEHRAN_TZ)


def test_business_weekdays_and_holidays_configured() -> None:
    # Sat=5, Sun=6, Mon=0, Tue=1, Wed=2; Thu=3, Fri=4 off
    assert BUSINESS_WEEKDAYS == frozenset({5, 6, 0, 1, 2})
    assert len(PUBLIC_HOLIDAYS) == 26
    for holiday in PUBLIC_HOLIDAYS:
        assert is_business_day(holiday) is False


def test_tehran_timezone_window() -> None:
    # Monday 2026-10-26: 05:30 UTC == 09:00 Tehran, 13:30 UTC == 17:00 Tehran
    assert business_hours_between(_utc(2026, 10, 26, 4, 0), _utc(2026, 10, 26, 5, 30)) == 0.0
    assert business_hours_between(_utc(2026, 10, 26, 5, 30), _utc(2026, 10, 26, 9, 0)) == 3.5
    assert business_minutes_between(_utc(2026, 10, 26, 5, 30), _utc(2026, 10, 26, 13, 30)) == 480
    assert business_hours_between(_utc(2026, 10, 26, 13, 30), _utc(2026, 10, 26, 16, 0)) == 0.0


def test_skips_thursday_friday_not_saturday_sunday() -> None:
    # Wednesday 2026-10-21 16:00 Tehran -> Saturday 2026-10-24 10:00 Tehran
    # 1h on Wed + 1h on Sat; Thursday (2026-10-22) and Friday (2026-10-23) skipped
    start = _tehran(2026, 10, 21, 16, 0)
    end = _tehran(2026, 10, 24, 10, 0)
    assert business_hours_between(start, end) == 2.0
    assert business_minutes_between(start, end) == 120

    # Saturday 09:00 Tehran -> Sunday 17:00 Tehran (both are full business days)
    sat_open = _tehran(2026, 10, 24, 9, 0)
    sun_close = _tehran(2026, 10, 25, 17, 0)
    assert business_hours_between(sat_open, sun_close) == 16.0
    assert business_minutes_between(sat_open, sun_close) == 960


def test_skips_consecutive_nowruz_holidays_1405() -> None:
    # 1405/01/01..04 = 2026-03-21 (Sat) .. 2026-03-24 (Tue)
    # Wednesday 2026-03-18 16:00 Tehran -> Wednesday 2026-03-25 10:00 Tehran
    # 1h on Wed 03-18 + 0 on Thu/Fri + 0 on Sat..Tue holidays + 1h on Wed 03-25
    start = _tehran(2026, 3, 18, 16, 0)
    end = _tehran(2026, 3, 25, 10, 0)
    assert business_hours_between(start, end) == 2.0
    assert business_minutes_between(start, end) == 120


def test_skips_single_1405_holiday() -> None:
    # 1405/06/01 = Sunday 2026-08-23
    # Saturday 2026-08-22 16:00 Tehran -> Monday 2026-08-24 10:00 Tehran
    start = _tehran(2026, 8, 22, 16, 0)
    end = _tehran(2026, 8, 24, 10, 0)
    assert business_hours_between(start, end) == 2.0


def test_exact_two_hour_boundary_and_minutes_until_breach() -> None:
    received = _tehran(2026, 10, 26, 9, 0)

    # 119 minutes elapsed -> 1 minute left, not urgent
    t_119 = _tehran(2026, 10, 26, 10, 59)
    assert business_minutes_until_breach(received, t_119, sla_hours=2.0) == 1
    assert is_urgent("high", received, None, t_119, sla_hours=2.0) is False

    # 119m 30s elapsed -> still 1 minute left (> 0), not urgent
    t_119_30 = _tehran(2026, 10, 26, 10, 59, 30)
    assert business_minutes_until_breach(received, t_119_30, sla_hours=2.0) == 1
    assert is_urgent("high", received, None, t_119_30, sla_hours=2.0) is False

    # Exact 2.0h (120m) boundary -> 0 minutes left, urgent
    t_120 = _tehran(2026, 10, 26, 11, 0)
    assert business_minutes_until_breach(received, t_120, sla_hours=2.0) == 0
    assert is_urgent("high", received, None, t_120, sla_hours=2.0) is True

    # 121m elapsed -> -1 minute left, urgent
    t_121 = _tehran(2026, 10, 26, 11, 1)
    assert business_minutes_until_breach(received, t_121, sla_hours=2.0) == -1
    assert is_urgent("high", received, None, t_121, sla_hours=2.0) is True


def test_business_hours_empty_and_reversed_intervals() -> None:
    t = _utc(2026, 10, 26, 10, 0)
    assert business_hours_between(t, t) == 0.0
    assert business_minutes_between(t, t) == 0
    assert business_hours_between(t + timedelta(hours=1), t) == 0.0


def test_multi_week_holiday_span_matches_day_by_day_reference() -> None:
    # Span across Farvardin 1405 (multiple holidays and weekends)
    start = _tehran(2026, 3, 20, 12, 0)  # Friday before Nowruz
    end = _tehran(2026, 4, 15, 13, 30)  # 26 days later, day after 1405/01/25 holiday

    expected_seconds = 0.0
    cur_date: date = start.date()
    while cur_date <= end.date():
        if is_business_day(cur_date):
            opens = datetime.combine(cur_date, BUSINESS_START, tzinfo=TEHRAN_TZ)
            closes = datetime.combine(cur_date, BUSINESS_END, tzinfo=TEHRAN_TZ)
            lo = max(opens, start)
            hi = min(closes, end)
            if hi > lo:
                expected_seconds += (hi - lo).total_seconds()
        cur_date += timedelta(days=1)

    assert business_hours_between(start, end) == expected_seconds / 3600.0
    assert business_minutes_between(start, end) == int(expected_seconds // 60)
