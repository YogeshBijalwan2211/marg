"""Indic roman/native-script transliteration around the NLLB translation pivot.

IndicXlit is optional at runtime: if it is unavailable, romanised input is
left untouched and the caller can rely on the pipeline's quality/confirmation
gates. NLLB itself does not accept romanised Hindi as hin_Latn.
"""
from __future__ import annotations

import re

from .langid import XLIT_CODE

# Restrict reverse transliteration to Indic/Arabic-script runs so that English
# words, numbers, placeholders, URLs and punctuation are not rewritten.
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
        result = self._engine(code, "en").translit_sentence(segment)
        return result[code] if isinstance(result, dict) else result

    def transliterate(self, text: str, roman_lang: str) -> str:
        """Roman-script Indic text -> the language's native script."""
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
        """Native script -> Roman script, preserving English text and formatting."""
        code = XLIT_CODE.get(roman_lang.split("_")[0])
        if code is None:
            return text
        return _INDIC_RUN.sub(
            lambda match: " ".join(
                self._to_roman_word(word, code)
                for word in re.findall(r"[^\s]+", match.group(0))
            ),
            text,
        )
