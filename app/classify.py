"""Category & severity classification with LLM caching and deterministic rule fallback."""
from __future__ import annotations

import hashlib
import logging
import re

import pandas as pd

from .config import Settings
from .llm import Classification, LLMClient, MessageToClassify, VALID_SEVERITIES
from .text import normalize_persian

logger = logging.getLogger(__name__)

# Maps (llm_model, sha256(normalized_subject + '\0' + normalized_body)) -> LLM Classification.
# Fallback rule results are intentionally NOT stored here so future uploads can retry the LLM.
_CLASSIFY_CACHE: dict[tuple[str, str], Classification] = {}

_PROMPT_INJECTION_RE = re.compile(r"system note\b.*", re.IGNORECASE | re.DOTALL)

_WEBSITE_BUG_RE = re.compile(
    r"(website search|search page|won'?t load|nothing loads|<img|<script|"
    r"تغییر زبان برنامه|باگ سایت|صفحه سبد|نحوه دریافت مجوز)",
    re.IGNORECASE,
)

KEYWORDS: dict[str, tuple[str, ...]] = {
    "refund": (
        "refund", "money back", "return shipping", "return request", "bargasht", "pas bedid",
        "pas bedam", "bargarde", "morjui", "morjuii", "cancel shod",
        "مرجوع", "بازگشت وجه", "برگشت پول", "پول برنگشته", "استرداد", "عودت", "پس بدم", "برگرده",
    ),
    "payment": (
        "payment", "paid", "charged", "charge", "card", "wallet", "top-up", "topped up",
        "invoice", "factor", "cash on delivery", "gift card", "pardakht", "kif pul", "kif pool",
        "kart hedye", "sharj",
        "پرداخت", "برداشت", "کارت", "کیف پول", "فاکتور", "تراکنش", "درگاه",
        "ارزش افزوده", "مالیات", "کد تخفیف", "پول کسر", "از حسابم کم",
    ),
    "account": (
        "account", "password", "login", "log in", "sign in", "restore access", "otp",
        "hesab", "ramz", "vorood",
        "کنترل حساب", "شماره موبایل", "حسابم", "حساب ", "مسدود", "رمز", "ورود", "پسورد", "کد تایید",
    ),
    "delivery": (
        "delivery", "deliver", "delivered", "arrive", "arrived", "shipping", "courier",
        "tracking", "package", "parcel", "warehouse", "tahvil", "ersal", "post",
        "peyk", "nareside", "narside", "sefaresh", "morsule",
        "ارسال", "تحویل", "مرسوله", "پیک", "بسته", "پست", "کد رهگیری", "رهگیری", "آدرس اشتباه",
    ),
    "product_quality": (
        "broken", "damaged", "defective", "quality", "fake", "not working", "wrong item",
        "different item", "manual", "install", "colour", "color", "garanti", "warranty",
        "kharab", "rang", "taviz", "dangerous", "swollen", "smoke", "burning",
        "خراب", "شکسته", "کیفیت", "معیوب", "تقلبی", "کار نمی", "رنگ", "لک", "راهنما",
        "نصب", "مشخصات", "تعویض", "بو ", "باد کرده", "دود", "سوختگی", "خطرناک", "گارانتی",
    ),
}

CATEGORY_ORDER: tuple[str, ...] = ("account", "refund", "payment", "delivery", "product_quality")

_DELIVERED_MISSING_RE = re.compile(
    r"(تحویل.{0,20}(ثبت|شده|نشان).{0,55}نرسیده|"
    r"تحویل داده نشده.{0,20}تحویل.?شده|"
    r"status shode delivered|"
    r"(marked|shown|shows).{0,25}delivered.{0,55}(not received|never received|missing|nothing|haven'?t received|have not received))",
    re.IGNORECASE,
)

_HIGH_SEVERITY_PATTERNS: tuple[re.Pattern[str], ...] = (
    _DELIVERED_MISSING_RE,
    re.compile(
        r"(خطرناک|باد کرده|دود کرد|بوی سوختگی|آتش|جرقه|برق.?گرفت|"
        r"dangerous|swollen battery|smoke|burning smell|electric shock|caught fire)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(برداشت غیرمجاز|دوبار (کم|کسر|برداشت)|دو بار.{0,20}(کسر|کم|برداشت)|"
        r"کسر وجه بدون ثبت سفارش|(پول|مبلغ|وجه).{0,45}(کسر|کم|برداشت).{0,55}(سفارشی|سفارش).{0,25}(ثبت نشده|نشد)|"
        r"do bar.{0,25}kam shod|"
        r"charged twice|double charge|unauthori[sz]ed|did not authori[sz]e)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(کنترل حساب را در دست|تغییر شماره موبایل حساب توسط فرد دیگر|"
        r"بدون اطلاع من عوض شده|حسابم خالی از موجودی|لینک مشکوک.*رمزم|"
        r"password was changed|unfamiliar device|unrecognised login|unrecognized login|"
        r"someone else.{0,25}(account|access)|hacked|hack shode)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(استرداد مبلغ بالا|لغو شده.{0,45}(چند هفته|هنوز خبری نیست)|"
        r"cancel shod.{0,45}bayad bargarde.{0,30}nashode)",
        re.IGNORECASE,
    ),
)

_LOW_ROUTINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(not a big issue|just letting you know|faghat khastam begam|"
        r"شدنیه؟|چطور .*(عوض کنم|پاک کنم|وارد کنم|بگیرم)|"
        r"قابل مرجوع هستن|چند ساله|do you accept|how do i|can i|"
        r"could i get|در روزهای تعطیل هم انجام میشه|امکان تغییر زبان)",
        re.IGNORECASE,
    ),
)

_MEDIUM_SEVERITY_MARKERS: tuple[str, ...] = (
    "refund", "broken", "damaged", "defective", "missing", "not working", "late",
    "unavailable", "stuck", "top-up", "topped up", "sharj", "blocked", "msdod",
    "مرجوع", "رد درخواست", "خراب", "شکسته", "نرسید", "نیامد", "معیوب", "کیفیت",
    "مسدود", "آدرس اشتباه", "تماس نگرفت", "پیک",
)


def clear_classification_cache() -> None:
    """Clear the in-memory LLM classification cache."""
    _CLASSIFY_CACHE.clear()


def _clean_for_rules(subject: str | None, body: str | None) -> str:
    combined = f"{subject or ''} {body or ''}"
    combined = _PROMPT_INJECTION_RE.sub("", combined)
    return normalize_persian(combined, strip_zwnj=True).lower()


def _normalized_cache_key(model: str, subject: str | None, body: str | None) -> tuple[str, str]:
    norm_subj = " ".join(normalize_persian(subject or "", strip_zwnj=True).lower().split())
    norm_body = " ".join(normalize_persian(body or "", strip_zwnj=True).lower().split())
    digest = hashlib.sha256(f"{norm_subj}\0{norm_body}".encode("utf-8")).hexdigest()
    return (model, digest)


def classify_keywords(subject: str, body: str) -> str:
    text = _clean_for_rules(subject, body)
    if _WEBSITE_BUG_RE.search(text):
        return "other"

    # Explicit multi-intent priority: "restore access first" -> account
    if "restore access first" in text:
        return "account"

    # Explicit refund intent takes precedence over wallet/bank-account mentions
    if any(p in text for p in ("مرجوع", "استرداد", "عودت", "بازگشت وجه", "برگشت پول", "پس بدم", "morjui")):
        return "refund"

    # Payment debit / wallet / stored card takes precedence over "حسابم" bank-account phrasing
    if any(
        p in text
        for p in (
            "از حسابم کم", "پول کسر", "برداشت", "کیف پول", "کارت بانکی",
            "kif pul", "kif pool", "sharj",
        )
    ):
        return "payment"

    # Delivery tracking checks take precedence over "کار نمی" ("not working")
    if any(p in text for p in ("کد رهگیری", "رهگیری", "بسته ارسال")) and "بسته" in text:
        return "delivery"

    for category in CATEGORY_ORDER:
        if any(word in text for word in KEYWORDS[category]):
            return category
    return "other"


def classify_severity(subject: str, body: str, severity_hint: str | None = None) -> str:
    text = _clean_for_rules(subject, body)
    if any(pattern.search(text) for pattern in _HIGH_SEVERITY_PATTERNS):
        return "high"

    if any(pattern.search(text) for pattern in _LOW_ROUTINE_PATTERNS):
        return "low"

    if any(marker in text for marker in _MEDIUM_SEVERITY_MARKERS):
        return "medium"

    hint = str(severity_hint).strip().lower() if severity_hint is not None else ""
    if hint in ("low", "medium"):
        return hint
    return "low"


def _rule_fallback(subject: str, body: str, severity_hint: str | None = None) -> Classification:
    return Classification(
        category=classify_keywords(subject, body),
        severity=classify_severity(subject, body, severity_hint),
    )


def classify_messages_detailed(df: pd.DataFrame, settings: Settings) -> list[Classification]:
    """Return a ``Classification`` per row of ``df``, using LLM cache + fallback."""
    if df.empty:
        return []

    subjects = [str(s or "") for s in df["subject"]]
    bodies = [str(b or "") for b in df["body"]]
    message_ids = [str(m) for m in df["message_id"]]
    hints = (
        [None if pd.isna(h) else str(h) for h in df["severity_hint"]]
        if "severity_hint" in df.columns
        else [None] * len(df)
    )

    if not settings.llm_base_url:
        return [_rule_fallback(s, b, h) for s, b, h in zip(subjects, bodies, hints)]

    keys = [_normalized_cache_key(settings.llm_model, s, b) for s, b in zip(subjects, bodies)]
    llm_results_by_key: dict[tuple[str, str], Classification] = {}
    pending_by_key: dict[tuple[str, str], MessageToClassify] = {}
    hints_by_key: dict[tuple[str, str], str | None] = {}

    for mid, subj, body, hint, key in zip(message_ids, subjects, bodies, hints, keys):
        hints_by_key[key] = hint
        cached = _CLASSIFY_CACHE.get(key)
        if cached is not None:
            llm_results_by_key[key] = cached
        elif key not in pending_by_key:
            pending_by_key[key] = MessageToClassify(message_id=mid, subject=subj, body=body)

    if pending_by_key:
        pending_list = list(pending_by_key.values())
        id_to_msg = {msg.message_id: msg for msg in pending_list}
        id_to_key = {msg.message_id: key for key, msg in pending_by_key.items()}
        try:
            client = LLMClient(settings)
            classified = client.classify(pending_list)
        except Exception as exc:  # noqa: BLE001 — degrade gracefully to keyword rules
            logger.warning("LLM classification unavailable, falling back to rules: %s", exc)
            classified = {}

        for mid, cls in classified.items():
            key = id_to_key.get(mid)
            if key is not None:
                msg = id_to_msg[mid]
                sev = cls.severity if cls.severity in VALID_SEVERITIES else classify_severity(
                    msg.subject, msg.body, hints_by_key.get(key)
                )
                normalized_cls = Classification(category=cls.category, severity=sev)
                _CLASSIFY_CACHE[key] = normalized_cls
                llm_results_by_key[key] = normalized_cls

    output: list[Classification] = []
    for subj, body, hint, key in zip(subjects, bodies, hints, keys):
        hit = llm_results_by_key.get(key)
        if hit is not None:
            output.append(hit)
        else:
            output.append(_rule_fallback(subj, body, hint))
    return output


def classify_messages(df: pd.DataFrame, settings: Settings) -> pd.Series:
    """Return a category per row of ``df``."""
    detailed = classify_messages_detailed(df, settings)
    return pd.Series(
        [item.category for item in detailed],
        index=df.index,
        dtype=object,
    )
