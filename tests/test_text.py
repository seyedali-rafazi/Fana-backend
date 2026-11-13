from __future__ import annotations

from app.text import normalize_digits, normalize_persian


def test_normalize_digits() -> None:
    assert normalize_digits("۰۱۲۳۴۵۶۷۸۹") == "0123456789"
    assert normalize_digits("٠١٢٣٤٥٦٧٨٩") == "0123456789"
    assert normalize_digits("BZ-۴1٨9") == "BZ-4189"
    assert normalize_digits("") == ""
    assert normalize_digits(None) == ""


def test_normalize_persian() -> None:
    assert normalize_persian("كبير ى ة أ إ ٱ ؤ ۀ") == "کبیر ی ه ا ا ا و ه"
    assert normalize_persian("پيگيري\u200c مرج\u200cوعی", strip_zwnj=False) == "پیگیری\u200c مرج\u200cوعی"
    assert normalize_persian("پيگيري\u200c مرج\u200cوعی", strip_zwnj=True) == "پیگیری مرجوعی"
    assert normalize_persian("لطفاً فـــوراً", strip_zwnj=True) == "لطفا فورا"
    assert normalize_persian(None) == ""
