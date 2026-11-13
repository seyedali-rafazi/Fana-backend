from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.dedup import mark_duplicates, normalize_body
from app.ingest import load_messages


def test_normalize_body_whitespace_and_case() -> None:
    assert normalize_body("  Hello   THERE\n friend ") == "hello there friend"


def test_normalize_body_arabic_zwnj_and_digits() -> None:
    a = "سلام کالای مرج\u200cوعی به انبار رسیده و تأیید شده اما مبلغ ۵,۲۰۰,۰۰۰ تومان هن\u200cوز واریز نشده."
    b = "سلام كالاي مرجوعی به انبار رسيده و تاييد شده اما مبلغ 5,200,000 تومان هنوز واريز نشده."
    c = "سلام کالای مرجوعی به انبار رسیده و تأیید شده اما مبلغ ٥,٢٠٠,٠٠٠ تومان هنوز واریز نشده.\u200c"
    assert normalize_body(a) == normalize_body(b) == normalize_body(c)
    assert normalize_body("لطفاً پیگیری كنید") == normalize_body("لطفا پیگیری کنید")
    assert normalize_body("ا\u0653ب هٔ ك ى ة") == "آب ه ک ی ه"


def test_resent_message_marked_as_duplicate(sample_csv: Path) -> None:
    df = load_messages(sample_csv)
    dups = mark_duplicates(df)
    assert dups.tolist() == [None, None, "M001", None]


def test_same_text_from_different_customers_is_not_duplicate() -> None:
    df = pd.DataFrame(
        {
            "message_id": ["A", "B"],
            "customer_id": ["C1", "C2"],
            "body": ["Where is my parcel?", "Where is my parcel?"],
            "received_at": pd.to_datetime(["2026-10-26T08:00:00Z", "2026-10-26T08:05:00Z"]),
        }
    )
    assert mark_duplicates(df).tolist() == [None, None]


def test_persian_arabic_zwnj_duplicates_and_canonical_selection() -> None:
    df = pd.DataFrame(
        {
            "message_id": ["M_LATER", "M_EARLIEST", "M_TIE_SECOND"],
            "customer_id": ["C77", "C77", "C77"],
            "body": [
                "مبلغ ۵,۲۰۰,۰۰۰ تومان مرج\u200cوعی واریز نشده",
                "مبلغ 5,200,000 تومان مرجوعی واريز نشده",
                "مبلغ ٥,٢٠٠,٠٠٠ تومان مرجوعى واریز نشده\u200c",
            ],
            "received_at": pd.to_datetime(
                [
                    "2026-08-18T12:00:00Z",
                    "2026-08-18T08:00:00Z",
                    "2026-08-18T08:00:00Z",
                ]
            ),
        }
    )
    assert mark_duplicates(df).tolist() == ["M_EARLIEST", None, "M_EARLIEST"]
