"""Placeholder masking so schemes, amounts, numbers and URLs survive MT unchanged.

Kinds: S = scheme name, M = money (₹/Rs/INR amount), N = plain number, U = URL.
"""
from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass, field

from .glossary import Glossary
from .normalise import format_indian

_URL = re.compile(r"https?://[^\s)>\]]+|www\.[^\s)>\]]+")
_MONEY = re.compile(r"(?:₹|Rs\.?|INR)\s?\d[\d,]*(?:\.\d+)?", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")


@dataclass
class Slot:
    kind: str
    original: str
    scheme_id: str | None = None


@dataclass
class Masked:
    text: str
    slots: dict[str, Slot] = field(default_factory=dict)   # placeholder -> Slot


class Masker:
    def __init__(self, glossary: Glossary | None = None, fmt: str = "ZX{kind}{idx}X",
                 indian_grouping: bool = True):
        self.glossary = glossary or Glossary()
        self.fmt = fmt
        self.indian_grouping = indian_grouping
        # tolerant matcher: MT sometimes changes case or inserts spaces
        self._find = re.compile(r"Z\s?X\s?([SMNU])\s?(\d+)\s?X", re.IGNORECASE)

    def _ph(self, kind: str, idx: int) -> str:
        return self.fmt.format(kind=kind, idx=idx)

    def mask(self, text: str, normaliser=None) -> Masked:
        """`text` should already be cleaned (clean_text). Scheme names are matched on the raw
        text FIRST; `normaliser` (e.g. number normalisation) then runs on what is left, so a name
        like '... up to Rs.5 Lakh' is not mangled before it can be matched."""
        slots: dict[str, Slot] = {}
        counter = {"i": 0}

        def put(kind: str, original: str, scheme_id: str | None = None) -> str:
            counter["i"] += 1
            ph = self._ph(kind, counter["i"])
            slots[ph] = Slot(kind, original, scheme_id)
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
        return Masked(text, slots)

    # -- verification ------------------------------------------------------
    def placeholders_in(self, text: str) -> Counter:
        return Counter(f"{m.group(1).upper()}{m.group(2)}" for m in self._find.finditer(text))

    def verify(self, source: Masked, translated: str) -> tuple[bool, list[str], list[str]]:
        want = Counter(k[2:-1] for k in source.slots if k in source.text)  # e.g. "S1"
        got = self.placeholders_in(translated)
        missing = sorted((want - got).elements())
        extra = sorted((got - want).elements())
        return (not missing and not extra), missing, extra

    def stray_digits(self, source: Masked, translated: str) -> bool:
        """True if MT invented/changed digits outside the placeholders."""
        strip = lambda t: Counter(re.findall(r"\d", self._find.sub("", t)))
        return strip(source.text) != strip(translated)

    # -- unmasking -----------------------------------------------------------
    def unmask(self, translated: str, source: Masked, *, scheme_lang: str | None = None,
               group_money: bool = False) -> str:
        by_key = {k[2:-1]: s for k, s in source.slots.items()}  # "S1" -> Slot

        def back(m: re.Match) -> str:
            key = f"{m.group(1).upper()}{m.group(2)}"
            slot = by_key.get(key)
            if slot is None:
                return m.group(0)
            if slot.kind == "S" and slot.scheme_id:
                return self.glossary.render(slot.scheme_id, scheme_lang)
            if slot.kind == "M" and group_money and self.indian_grouping:
                sym = re.match(r"\D*", slot.original).group(0)
                num = slot.original[len(sym):].replace(",", "")
                return f"{sym}{format_indian(num)}"
            return slot.original

        return self._find.sub(back, translated)
