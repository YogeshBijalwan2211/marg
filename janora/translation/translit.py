"""Romanised -> native script transliteration (IndicXlit), applied only to
non-placeholder segments so ZX..X tokens are never transliterated."""
from __future__ import annotations
import re
import warnings
from .langid import XLIT_CODE

_PH = re.compile(r"(Z\s?X\s?[SMNU]\s?\d+\s?X)", re.IGNORECASE)


class Transliterator:
    def __init__(self):
        self._engines: dict[str, object] = {}
        self.available = True      # becomes False if IndicXlit is not installed

    def _engine(self, code: str):
        if code not in self._engines:
            from ai4bharat.transliteration import XlitEngine  # type: ignore
            self._engines[code] = XlitEngine(code, beam_width=10, src_script_type="en")
        return self._engines[code]

    def _one(self, seg: str, code: str) -> str:
        if not seg.strip():
            return seg
        out = self._engine(code).translit_sentence(seg)
        return out[code] if isinstance(out, dict) else out

    def transliterate(self, text: str, roman_lang: str) -> str:
        code = XLIT_CODE.get(roman_lang.split("_")[0])
        if code is None or not self.available:
            return text
        try:
            self._engine(code)
        except ImportError:        # e.g. fairseq does not build on Python 3.13 / Windows
            self.available = False
            warnings.warn("IndicXlit not installed: romanised input is sent to MT untransliterated. "
                          "Install ai4bharat-transliteration (Python <=3.10 / Colab) or use --no-translit.")
            return text
        parts = _PH.split(text)
        return "".join(p if _PH.fullmatch(p or "") else self._one(p, code) for p in parts)
