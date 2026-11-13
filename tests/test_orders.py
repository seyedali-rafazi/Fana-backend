from __future__ import annotations

from app.orders import extract_order_ref, find_order_ref


def test_order_ref_from_field() -> None:
    assert extract_order_ref("BZ-48219", "") == "BZ-48219"


def test_order_ref_from_body_when_field_empty() -> None:
    assert extract_order_ref("", "my order bz-77310 is late") == "BZ-77310"
    assert extract_order_ref("", "no order mentioned") is None


def test_order_ref_persian_and_arabic_digits() -> None:
    assert find_order_ref("BZ-۴۱۸۱۹۱") == "BZ-418191"
    assert find_order_ref("BZ-٩٧١٩٢") == "BZ-97192"
    assert find_order_ref("bz-4۱۸19١") == "BZ-418191"
    assert extract_order_ref("BZ-۴۱۸۱۹۱", "سفارش BZ-111111") == "BZ-418191"
    assert (
        extract_order_ref("", "کالای سفارش BZ-۴۱۸۱۹۱ رو 14 روز پیش پس دادم")
        == "BZ-418191"
    )
    assert (
        extract_order_ref("", "سفارش bz-٩٧١٩٢ هنوز نرسیده")
        == "BZ-97192"
    )
