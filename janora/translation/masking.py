"""Placeholder masking so schemes, amounts, numbers and URLs survive MT unchanged.

Kinds: S = scheme name, M = money (₹/Rs/INR amount), N = plain number, U = URL.

Two placeholder styles (TranslationConfig.placeholder_style / output_placeholder_style):
  "alpha"  : ZXS1X-like Latin tokens. Fine for Indic->English (the model copies them), but some
             En->Indic models TRANSLITERATE them into the target script (seen: Bengali turned
             ZXN3X into জেডএক্সএন3এক্স), so the token is lost.
  "digits" : 5-digit numeric tokens (91003 = scheme #3, 92001 = money #1 ...). Models copy digits;
             native-digit output (৯১০০৩) is mapped back to ASCII before matching.
"""
from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass, field

from .glossary import Glossary
from .normalise import format_indian, to_ascii_digits

_URL = re.compile(r"https?://[^\s)>\]]+|www\.[^\s)>\]]+")
_MONEY = re.compile(r"(?:₹|Rs\.?|INR)\s?\d[\d,]*(?:\.\d+)?", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")

_KIND_CODE = {"S": "1", "M": "2", "N": "3", "U": "4"}
_CODE_KIND = {v: k for k, v in _KIND_CODE.items()}


@dataclass
class Slot:
    kind: str
    original: str
    scheme_id: str | None = None
    key: str = ""                 # style-independent id, e.g. "S2"


@dataclass
class Masked:
    text: str
    slots: dict[str, Slot] = field(default_factory=dict)   # placeholder token -> Slot


class Masker:
    def __init__(self, glossary: Glossary | None = None, fmt: str = "ZX{kind}{idx}X",
                 indian_grouping: bool = True, style: str = "alpha"):
        assert style in ("alpha", "digits")
        self.glossary = glossary or Glossary()
        self.fmt, self.style, self.indian_grouping = fmt, style, indian_grouping
        # tolerant matchers: MT sometimes changes case or inserts spaces (alpha) / uses native digits (digits)
        self._alpha = re.compile(r"Z\s?X\s?([SMNU])\s?(\d+)\s?X", re.IGNORECASE)
        self._digits = re.compile(r"(?<!\d)9([1-4])(\d{3})(?!\d)")

    # -- token construction / recognition -----------------------------------
    def _ph(self, kind: str, idx: int) -> str:
        """Token used DURING masking. Always alpha: numeric tokens would be re-matched by the
        number regex of a later masking step. `_restyle` converts to digits at the end."""
        return self.fmt.format(kind=kind, idx=idx)

    def _restyle(self, masked: "Masked") -> "Masked":
        if self.style != "digits":
            return masked
        remap = {tok: f"9{_KIND_CODE[sl.kind]}{int(sl.key[1:]):03d}" for tok, sl in masked.slots.items()}
        text = masked.text
        for tok, new in remap.items():                  # tokens are unique and not prefixes of each other
            text = text.replace(tok, new)
        return Masked(text, {remap[tok]: sl for tok, sl in masked.slots.items()})

    @property
    def split_pattern(self) -> re.Pattern:
        """Capturing pattern that isolates placeholders (used to keep them out of transliteration)."""
        if self.style == "digits":
            return re.compile(r"((?<!\d)9[1-4]\d{3}(?!\d))")
        return re.compile(r"(Z\s?X\s?[SMNU]\s?\d+\s?X)", re.IGNORECASE)

    def _norm(self, text: str) -> str:
        return to_ascii_digits(text) if self.style == "digits" else text

    def _regex(self) -> re.Pattern:
        return self._digits if self.style == "digits" else self._alpha

    @staticmethod
    def _key(style: str, m: re.Match) -> str:
        if style == "digits":
            return f"{_CODE_KIND[m.group(1)]}{int(m.group(2))}"
        return f"{m.group(1).upper()}{int(m.group(2))}"

    def strip_placeholders(self, text: str) -> str:
        return self._regex().sub("", self._norm(text))

    # -- masking ---------------------------------------------------------------
    def mask(self, text: str, normaliser=None) -> Masked:
        """`text` should already be cleaned (clean_text). Scheme names are matched on the raw
        text FIRST; `normaliser` (e.g. number normalisation) then runs on what is left, so a name
        like '... up to Rs.5 Lakh' is not mangled before it can be matched."""
        slots: dict[str, Slot] = {}
        counter = {"i": 0}

        def put(kind: str, original: str, scheme_id: str | None = None) -> str:
            counter["i"] += 1
            ph = self._ph(kind, counter["i"])
            slots[ph] = Slot(kind, original, scheme_id, f"{kind}{counter['i']}")
            return ph

        # order matters: URLs, schemes, [normaliser], money, bare numbers
        text = _URL.sub(lambda m: put("U", m.group(0)), text)
        if self.glossary.pattern is not None:
            text = self.glossary.pattern.sub(
                lambda m: put("S", m.group(0), self.glossary.scheme_id(m.group(0))), text)
        if normaliser is not None:
            text = normaliser(text)
        text = _MONEY.sub(lambda m: put("M", m.group(0)), text)
        text = _NUMBER.sub(lambda m: put("N", m.group(0)), text)
        return self._restyle(Masked(text, slots))

    # -- verification ------------------------------------------------------
    def placeholders_in(self, text: str) -> Counter:
        return Counter(self._key(self.style, m) for m in self._regex().finditer(self._norm(text)))

    def verify(self, source: Masked, translated: str) -> tuple[bool, list[str], list[str]]:
        want = Counter(s.key for tok, s in source.slots.items() if tok in source.text)
        got = self.placeholders_in(translated)
        missing = sorted((want - got).elements())
        extra = sorted((got - want).elements())
        return (not missing and not extra), missing, extra

    def stray_digits(self, source: Masked, translated: str) -> bool:
        """True if MT invented/changed digits outside the placeholders (native digits count as ASCII)."""
        digits = lambda t: Counter(re.findall(r"[0-9]", to_ascii_digits(self.strip_placeholders(t))))
        return digits(source.text) != digits(translated)

    # -- unmasking -----------------------------------------------------------
    def unmask(self, translated: str, source: Masked, *, scheme_lang: str | None = None,
               group_money: bool = False) -> str:
        by_key = {s.key: s for s in source.slots.values()}

        def back(m: re.Match) -> str:
            slot = by_key.get(self._key(self.style, m))
            if slot is None:
                return m.group(0)
            if slot.kind == "S" and slot.scheme_id:
                return self.glossary.render(slot.scheme_id, scheme_lang)
            if slot.kind == "M" and group_money and self.indian_grouping:
                sym = re.match(r"\D*", slot.original).group(0)
                num = slot.original[len(sym):].replace(",", "")
                return f"{sym}{format_indian(num)}"
            return slot.original

        return self._regex().sub(back, self._norm(translated))
