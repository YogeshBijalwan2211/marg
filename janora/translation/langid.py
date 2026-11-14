"""Language identification and mapping to NLLB/FLORES language tags.

IndicLID is the preferred identifier when installed. A script-aware heuristic
fallback keeps the package usable without external language-ID model files.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_SCRIPTS = [
    (0x0900, 0x097F, "hin_Deva", "Deva"),
    (0x0980, 0x09FF, "ben_Beng", "Beng"),
    (0x0A00, 0x0A7F, "pan_Guru", "Guru"),
    (0x0A80, 0x0AFF, "guj_Gujr", "Gujr"),
    (0x0B00, 0x0B7F, "ory_Orya", "Orya"),
    (0x0B80, 0x0BFF, "tam_Taml", "Taml"),
    (0x0C00, 0x0C7F, "tel_Telu", "Telu"),
    (0x0C80, 0x0CFF, "kan_Knda", "Knda"),
    (0x0D00, 0x0D7F, "mal_Mlym", "Mlym"),
    (0x0600, 0x06FF, "urd_Arab", "Arab"),
]
_HI_ROMAN = set("""hai hain hoon hu mera meri mere mujhe mujhko kya ke ki ka ko se mein nahi nahin
kaise kitna kitni kitne aur hum aap apna apni yojana scheme paisa paise saal umar chahiye milega
milegi milta kab kahan kaun kisan kisaan pension aay aamdani tankhwah bete beti pita mata patni
lakh hazaar liye wala wali raha rahi karna karni kar sakta sakti""".split())
_EN = set("""the is are was my i me for what how can of and to in a an do does am have has will which
who where when if it this that with on at be""".split())

# Dataset language labels -> NLLB/FLORES language codes.
LABEL_TO_CODE = {
    "english": "eng_Latn",
    "hindi": "hin_Deva",
    "hinglish": "hin_Latn",
    "bengali": "ben_Beng",
    "assamese": "asm_Beng",
    "telugu": "tel_Telu",
    "marathi": "mar_Deva",
    "tamil": "tam_Taml",
    "urdu": "urd_Arab",
    "gujarati": "guj_Gujr",
    "kannada": "kan_Knda",
    "odia": "ory_Orya",
    "malayalam": "mal_Mlym",
    "punjabi": "pan_Guru",
    "nepali": "npi_Deva",
}

_DEV_MARKERS = {
    "mar_Deva": {
        "आहे", "आहेत", "मी", "माझे", "माझी", "माझा", "मला", "नाही", "आणि",
        "काय", "साठी", "मध्ये", "वय", "वर्षे", "वर्षांचा", "राहतो", "राहते",
        "आमच्या", "आमचे", "कुटुंबाचे", "उत्पन्न", "मिळेल", "आहोत",
    },
    "npi_Deva": {
        "छ", "छु", "छन्", "मेरो", "हुन्छ", "लागि", "बस्छु", "बस्छे", "भएको",
        "हामी", "तपाईं", "हुँ", "म", "मेरा", "हाम्रो", "परिवारको", "आम्दानी",
        "उमेर", "वर्षको", "कुनै", "छैन", "के",
    },
    "hin_Deva": {
        "है", "हैं", "मेरी", "हूँ", "हूं", "क्या", "लिए", "में", "मैं", "साल",
        "उम्र", "कोई", "परिवार", "सालाना",
    },
}


def label_to_code(label: str, text: str) -> str:
    """Map dataset labels to NLLB codes, noticing Indic languages typed in Latin."""
    try:
        code = LABEL_TO_CODE[label.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset language label: {label!r}") from exc
    if code == "eng_Latn":
        return code
    letters = [char for char in text if char.isalpha()]
    indic = sum(
        any(low <= ord(char) <= high for low, high, *_ in _SCRIPTS)
        for char in letters
    )
    if letters and indic / len(letters) < 0.12:
        return code.split("_")[0] + "_Latn"
    return code


@dataclass
class LangResult:
    code: str
    confidence: float
    backend: str = "heuristic"

    @property
    def is_roman_indic(self) -> bool:
        return self.code.endswith("_Latn") and not self.code.startswith("eng")


class LangID:
    def __init__(self, use_indiclid: bool = True):
        self._model = None
        if use_indiclid:
            try:
                from ai4bharat.IndicLID import IndicLID  # type: ignore
                self._model = IndicLID(input_threshold=0.5, roman_lid_threshold=0.6)
            except Exception:
                self._model = None

    def detect(self, text: str) -> LangResult:
        if self._model is not None:
            try:
                _, code, score, _ = self._model.batch_predict([text], 1)[0]
                return LangResult(code, float(score), "indiclid")
            except Exception:
                pass
        return self._heuristic(text)

    @staticmethod
    def _heuristic(text: str) -> LangResult:
        letters = [char for char in text if char.isalpha()]
        if not letters:
            return LangResult("eng_Latn", 0.0)
        counts: dict[tuple[str, str], int] = {}
        for char in letters:
            ordinal = ord(char)
            for low, high, lang, script in _SCRIPTS:
                if low <= ordinal <= high:
                    counts[(lang, script)] = counts.get((lang, script), 0) + 1
                    break
        if counts:
            (lang, _), count = max(counts.items(), key=lambda item: item[1])
            if count >= 3 and count / len(letters) >= 0.12:
                confidence = min(0.99, 0.5 + count / len(letters))
                if lang == "ben_Beng" and re.search("[\u09F0\u09F1]", text):
                    lang = "asm_Beng"
                elif lang == "hin_Deva":
                    tokens = set(re.findall(r"[\u0900-\u097F]+", text))
                    scores = {key: len(tokens & values) for key, values in _DEV_MARKERS.items()}
                    best = max(scores, key=lambda key: (scores[key], key == "hin_Deva"))
                    if scores[best] > 0:
                        lang = best
                    confidence = min(confidence, 0.9)
                return LangResult(lang, confidence)

        tokens = re.findall(r"[a-z]+", text.lower())
        if not tokens:
            return LangResult("eng_Latn", 0.5)
        hindi_count = sum(token in _HI_ROMAN for token in tokens)
        english_count = sum(token in _EN for token in tokens)
        if hindi_count >= 1 and hindi_count / len(tokens) >= 0.15 and hindi_count >= english_count:
            return LangResult("hin_Latn", min(0.95, 0.5 + hindi_count / len(tokens)))
        return LangResult("eng_Latn", min(0.95, 0.6 + english_count / max(len(tokens), 1)))


NATIVE_SCRIPT = {
    "hin": "Deva", "ben": "Beng", "pan": "Guru", "guj": "Gujr",
    "ory": "Orya", "tam": "Taml", "tel": "Telu", "kan": "Knda",
    "mal": "Mlym", "mar": "Deva", "asm": "Beng", "npi": "Deva",
    "urd": "Arab",
}
XLIT_CODE = {
    "asm": "as", "npi": "ne", "hin": "hi", "ben": "bn", "pan": "pa",
    "guj": "gu", "ory": "or", "tam": "ta", "tel": "te", "kan": "kn",
    "mal": "ml", "mar": "mr", "urd": "ur",
}


def native_code(roman_code: str) -> str:
    """Convert an Indic Latin-script label (e.g. hin_Latn) to native-script code."""
    stem = roman_code.split("_")[0]
    return f"{stem}_{NATIVE_SCRIPT.get(stem, 'Deva')}"
