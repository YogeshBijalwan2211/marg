"""Script-level and number normalisation, applied BEFORE translation.

 - Any Unicode decimal digit (Devanagari, Bengali, Tamil...) -> ASCII
 - "3,00,000" / "300,000"                               -> 300000
 - "3 lakh", "2.5 lakh", "1 crore", "तीन लाख", "teen hazaar" -> plain integers
"""
from __future__ import annotations
import re
import unicodedata
from decimal import Decimal

_WORD_NUMS = {
    "ek": 1, "एक": 1, "do": 2, "दो": 2, "teen": 3, "तीन": 3,
    "char": 4, "chaar": 4, "चार": 4, "paanch": 5, "panch": 5, "पांच": 5, "पाँच": 5,
    "chhe": 6, "cheh": 6, "छह": 6, "छः": 6, "saat": 7, "सात": 7,
    "aath": 8, "आठ": 8, "nau": 9, "नौ": 9, "das": 10, "दस": 10,
}
_LAKH = ["lakh", "lakhs", "lac", "lacs", "lak", "laakh", "lakkh",
         "लाख", "লাখ", "লক্ষ", "લાખ", "ಲಕ್ಷ", "ലക്ഷം", "ଲକ୍ଷ", "ਲੱਖ", "லட்சம்", "லட்சம", "లక్ష", "లక్షలు", "لاکھ"]
_CRORE = ["crore", "crores", "cr", "करोड़", "करोड", "कोटी", "কোটি", "કરોડ", "ಕೋಟಿ", "കോടി", "କୋଟି", "ਕਰੋੜ", "கோடி", "కోటి", "کروڑ"]
_THOUSAND = ["hazaar", "hazar", "hajar", "thousand", "हज़ार", "हजार", "হাজার", "હજાર", "ಸಾವಿರ", "ആയിരം", "ହଜାର",
             "ਹਜ਼ਾਰ", "ஆயிரம்", "వేయి", "హజార్", "ہزار"]
# NFC-normalise the keys: the input is NFC'd, and precomposed nukta letters (ड़, ज़) decompose under NFC
_UNITS = {unicodedata.normalize("NFC", w): v
          for words, v in ((_LAKH, 100_000), (_CRORE, 10_000_000), (_THOUSAND, 1_000)) for w in words}
_WORD_NUMS = {unicodedata.normalize("NFC", k): v for k, v in _WORD_NUMS.items()}

_INDIC = r"\u0600-\u06FF\u0900-\u0DFF"       # Arabic/Urdu + all Indic blocks (matras are not \w)
_NOT_BEFORE = rf"(?<![\w{_INDIC}])"
_NOT_AFTER = rf"(?![\w{_INDIC}])"

_grouped = re.compile(r"(?<![\d,])(?:\d{1,2}(?:,\d{2})+,\d{3}|\d{1,3}(?:,\d{3})+)(?![\d,])")
_words = "|".join(sorted(map(re.escape, _WORD_NUMS), key=len, reverse=True))
_units = "|".join(sorted(map(re.escape, _UNITS), key=len, reverse=True))
_amount = re.compile(
    rf"{_NOT_BEFORE}(?:(?P<num>\d+(?:\.\d+)?)|(?P<word>{_words}))\s*(?P<unit>{_units}){_NOT_AFTER}",
    re.IGNORECASE,
)


def to_ascii_digits(text: str) -> str:
    return "".join(
        str(unicodedata.digit(ch)) if unicodedata.category(ch) == "Nd" else ch for ch in text
    )


def _fmt(d: Decimal) -> str:
    return str(int(d)) if d == d.to_integral_value() else format(d.normalize(), "f")


_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0F\u20E3]")


def clean_text(text: str) -> str:
    """NFC, strip emoji, collapse whitespace. (ZWJ/ZWNJ are kept: they are meaningful in Indic scripts.)"""
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"\s+", " ", _EMOJI.sub(" ", text)).strip()


def normalise_text(text: str) -> tuple[str, list[tuple[str, str]]]:
    """clean_text + number normalisation. Returns (text, [(original_span, replacement), ...])."""
    return normalise_numbers(clean_text(text))


def normalise_numbers(text: str) -> tuple[str, list[tuple[str, str]]]:
    log: list[tuple[str, str]] = []
    text = to_ascii_digits(text)

    def strip_commas(m: re.Match) -> str:
        out = m.group(0).replace(",", "")
        log.append((m.group(0), out))
        return out

    text = _grouped.sub(strip_commas, text)

    def expand(m: re.Match) -> str:
        base = Decimal(m.group("num")) if m.group("num") else Decimal(_WORD_NUMS[m.group("word").lower()])
        out = _fmt(base * _UNITS[m.group("unit").lower()])
        log.append((m.group(0), out))
        return out

    text = _amount.sub(expand, text)
    return text, log


def format_indian(n: str) -> str:
    """'300000' -> '3,00,000'. Leaves decimals and short numbers alone."""
    whole, dot, frac = n.partition(".")
    if not whole.isdigit() or len(whole) <= 3:
        return n
    head, tail = whole[:-3], whole[-3:]
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts + [tail]) + dot + frac
