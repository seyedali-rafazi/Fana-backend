"""Shared text normalization helpers for Persian, Arabic, and mixed exports."""
from __future__ import annotations

import re
import unicodedata

_DIGIT_TRANSLATION = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
    "01234567890123456789",
)

_ARABIC_TO_PERSIAN = str.maketrans(
    {
        "\u064a": "\u06cc",  # ي -> ی (Arabic Yeh -> Persian Yeh)
        "\u0649": "\u06cc",  # ى -> ی (Alef Maksura -> Persian Yeh)
        "\u0643": "\u06a9",  # ك -> ک (Arabic Kaf -> Persian Keheh)
        "\u06aa": "\u06a9",  # ڪ -> ک (Swash Kaf -> Persian Keheh)
        "\u0629": "\u0647",  # ة -> ه (Teh Marbuta -> Heh)
        "\u06c0": "\u0647",  # ۀ -> ه (Heh with Yeh above -> Heh)
        "\u0623": "\u0627",  # أ -> ا (Alef with Hamza above -> Alef)
        "\u0625": "\u0627",  # إ -> ا (Alef with Hamza below -> Alef)
        "\u0671": "\u0627",  # ٱ -> ا (Alef Wasla -> Alef)
        "\u0624": "\u0648",  # ؤ -> و (Waw with Hamza above -> Waw)
        "\u0640": "",        # ـ (Arabic Tatweel / Kashida)
    }
)

_ZERO_WIDTH_CHARS = re.compile(r"[\u200b-\u200f\ufeff]")
_DIACRITICS = re.compile(r"[\u064b-\u065f\u0670]")


def normalize_digits(text: str | None) -> str:
    """Convert Persian (۰-۹) and Eastern Arabic (٠-٩) digits to ASCII 0-9."""
    if not text:
        return ""
    return text.translate(_DIGIT_TRANSLATION)


def normalize_persian(text: str | None, *, strip_zwnj: bool = False) -> str:
    """Normalize Unicode NFC, digits, and Arabic letter variants to standard Persian.

    When ``strip_zwnj`` is True, zero-width non-joiner and related invisible
    formatting characters are also removed.
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalize_digits(normalized)
    normalized = normalized.translate(_ARABIC_TO_PERSIAN)
    normalized = _DIACRITICS.sub("", normalized)
    if strip_zwnj:
        normalized = _ZERO_WIDTH_CHARS.sub("", normalized)
    return normalized
