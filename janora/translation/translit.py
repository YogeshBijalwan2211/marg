"""Indic roman/native-script transliteration around the NLLB translation pivot.

IndicXlit supports Roman-to-native and native-to-Roman conversions. It is a
runtime dependency for romanised Indic input/output; the MT model itself does
not accept Latin-script Hindi/Bengali/etc. as a target/source language tag.
"""
from __future__ import annotations

import re

from .langid import XLIT_CODE

_INDIC_RUN = re.compile(r"[\u0600-\u06FF\u0900-\u0DFF]+")


class Transliterator:
    def __init__(self):
        self._engines: dict[tuple[str, str], object] = {}

    def _engine(self, code: str, source_script: str):
        key = (code, source_script)
        if key not in self._engines:
            from ai4bharat.transliteration import XlitEngine  # type: ignore

            self._engines[key] = XlitEngine(
                code,
                beam_width=10,
                src_script_type=source_script,
                rescore=False if source_script == "indic" else True,
            )
        return self._engines[key]

    def _to_native(self, segment: str, code: str) -> str:
        if not segment.strip():
            return segment
        result = self._engine(code, "roman").translit_sentence(segment)
        return result[code] if isinstance(result, dict) else result

    def transliterate(self, text: str, roman_lang: str) -> str:
        """Roman-script Indic text -> native script, keeping placeholders intact."""
        code = XLIT_CODE.get(roman_lang.split("_")[0])
        if code is None:
            return text
        parts = re.split(r"(ZX[SMNU]\d+X)", text, flags=re.IGNORECASE)
        return "".join(
            part if re.fullmatch(r"ZX[SMNU]\d+X", part, flags=re.IGNORECASE)
            else self._to_native(part, code)
            for part in parts
        )

    def _to_roman_word(self, word: str, code: str) -> str:
        result = self._engine(code, "indic").translit_word(
            word, lang_code=code, topk=1
        )
        if isinstance(result, dict):
            result = result.get(code, [])
        if isinstance(result, list):
            return result[0] if result else word
        return result or word

    def transliterate_to_roman(self, text: str, roman_lang: str) -> str:
        """Native script -> Roman script, without rewriting English words or punctuation."""
        code = XLIT_CODE.get(roman_lang.split("_")[0])
        if code is None:
            return text

        def convert_run(match: re.Match) -> str:
            return "".join(
                self._to_roman_word(part, code) if part.strip() else part
                for part in re.split(r"(\s+)", match.group(0))
            )

        return _INDIC_RUN.sub(convert_run, text)
