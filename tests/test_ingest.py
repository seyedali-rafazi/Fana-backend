from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from app.ingest import is_jalali_leap, jalali_to_gregorian, load_messages, parse_timestamps


def test_loads_all_clean_rows(sample_csv: Path) -> None:
    df = load_messages(sample_csv)
    assert list(df["message_id"]) == ["M001", "M002", "M003", "M004"]
    assert "does not work" in df.loc[3, "body"]


def test_iso_timestamps_are_utc(sample_csv: Path) -> None:
    df = load_messages(sample_csv)
    assert df.loc[0, "received_at"] == pd.Timestamp("2026-10-26T08:00:00Z")
    assert str(df["received_at"].dt.tz) == "UTC"
    assert pd.isna(df.loc[1, "answered_at"])


def test_jalali_to_gregorian_and_leap_years() -> None:
    assert jalali_to_gregorian(1405, 1, 1).isoformat() == "2026-03-21"
    assert jalali_to_gregorian(1405, 5, 23).isoformat() == "2026-08-14"
    assert jalali_to_gregorian(1405, 6, 1).isoformat() == "2026-08-23"
    assert jalali_to_gregorian(1405, 6, 31).isoformat() == "2026-09-22"
    assert jalali_to_gregorian(1405, 7, 1).isoformat() == "2026-09-23"
    assert jalali_to_gregorian(1403, 12, 30).isoformat() == "2025-03-20"

    assert is_jalali_leap(1399) is True
    assert is_jalali_leap(1403) is True
    assert is_jalali_leap(1404) is False
    assert is_jalali_leap(1405) is False
    assert is_jalali_leap(1408) is True

    with pytest.raises(ValueError):
        jalali_to_gregorian(1405, 13, 1)
    with pytest.raises(ValueError):
        jalali_to_gregorian(1405, 7, 31)
    with pytest.raises(ValueError):
        jalali_to_gregorian(1405, 12, 30)


def test_parse_timestamps_mixed_iso_and_jalali() -> None:
    series = pd.Series(
        [
            "2026-08-14T05:36:00Z",
            "2026-08-14T09:06:00+03:30",
            "۱۴۰۵/۰۵/۲۳ ۱۱:۵۲",
            "1405/06/01 23:20:15",
            "١٤٠٥/٠١/١٨ ١٤:٠٦",
            "1400/05/01 12:00",  # pre-2022 Tehran DST (+04:30)
            "1405/05/23 08:00Z",
            "",
            "not-a-timestamp",
            "1405/12/30 10:00",  # invalid non-leap Esfand 30
        ]
    )
    parsed = parse_timestamps(series)
    assert str(parsed.dt.tz) == "UTC"
    assert parsed.iloc[0] == pd.Timestamp("2026-08-14T05:36:00Z")
    assert parsed.iloc[1] == pd.Timestamp("2026-08-14T05:36:00Z")
    assert parsed.iloc[2] == pd.Timestamp("2026-08-14T08:22:00Z")
    assert parsed.iloc[3] == pd.Timestamp("2026-08-23T19:50:15Z")
    assert parsed.iloc[4] == pd.Timestamp("2026-04-07T10:36:00Z")
    assert parsed.iloc[5] == pd.Timestamp("2021-07-23T07:30:00Z")
    assert parsed.iloc[6] == pd.Timestamp("2026-08-14T08:00:00Z")
    assert pd.isna(parsed.iloc[7])
    assert pd.isna(parsed.iloc[8])
    assert pd.isna(parsed.iloc[9])


def test_parse_timestamps_empty_series() -> None:
    parsed = parse_timestamps(pd.Series([], dtype=str))
    assert len(parsed) == 0
    assert str(parsed.dt.tz) == "UTC"


def test_load_messages_keeps_jalali_and_drops_malformed_rows() -> None:
    csv_text = (
        "message_id,customer_id,channel,received_at,answered_at,subject,body,order_ref,severity_hint\n"
        "M1,C1,email,2026-08-14T05:36:00Z,۱۴۰۵/۰۵/۲۳ ۱۱:۵۲,subj,body,,high\n"
        "M2,C2,app,1405/06/01 23:20,,subj,body,BZ-۴۱۸۱۹۱,HIGH\n"
        "   ,C3,app,2026-08-14T06:00:00Z,,subj,body,,low\n"
        "M4,C4,app,invalid-date,,subj,body,,low\n"
        ",,,,,,,,\n"
        "   \n"
    )
    df = load_messages(io.StringIO(csv_text))
    assert list(df["message_id"]) == ["M1", "M2"]
    assert df.loc[0, "answered_at"] == pd.Timestamp("2026-08-14T08:22:00Z")
    assert df.loc[1, "received_at"] == pd.Timestamp("2026-08-23T19:50:00Z")
    assert pd.isna(df.loc[1, "answered_at"])
    assert df.loc[1, "severity_hint"] == "high"
