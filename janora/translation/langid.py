"""Language identification.

Primary: IndicLID (handles romanised Hinglish). Fallback: a dependency-free
script detector + romanised-Hindi lexicon heuristic. Replace the heuristic with
the real model for production; the heuristic exists so the pipeline runs anywhere.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

_SCRIPTS = [  # (lo, hi, lang, script)
    (0x0900, 0x097F, "hin_Deva", "Deva"), (0x0980, 0x09FF, "ben_Beng", "Beng"),
    (0x0A00, 0x0A7F, "pan_Guru", "Guru"), (0x0A80, 0x0AFF, "guj_Gujr", "Gujr"),
    (0x0B00, 0x0B7F, "ory_Orya", "Orya"), (0x0B80, 0x0BFF, "tam_Taml", "Taml"),
    (0x0C00, 0x0C7F, "tel_Telu", "Telu"), (0x0C80, 0x0CFF, "kan_Knda", "Knda"),
    (0x0D00, 0x0D7F, "mal_Mlym", "Mlym"), (0x0600, 0x06FF, "urd_Arab", "Arab"),
]
_HI_ROMAN = set("""hai hain hoon hu mera meri mere mujhe mujhko kya ke ki ka ko se mein nahi nahin
kaise kitna kitni kitne aur hum aap apna apni yojana scheme paisa paise saal umar chahiye milega
milegi milta kab kahan kaun kisan kisaan pension aay aamdani tankhwah bete beti pita mata patni
lakh hazaar liye wala wali raha rahi karna karni kar sakta sakti""".split())
_EN = set("""the is are was my i me for what how can of and to in a an do does am have has will which
who where when if it this that with on at be""".split())


# labels used in this repo's data/splits/*.jsonl ("language" column) -> FLORES codes
LABEL_TO_CODE = {
    "english": "eng_Latn", "hindi": "hin_Deva", "hinglish": "hin_Latn", "bengali": "ben_Beng",
    "assamese": "asm_Beng", "telugu": "tel_Telu", "marathi": "mar_Deva", "tamil": "tam_Taml",
    "urdu": "urd_Arab", "gujarati": "guj_Gujr", "kannada": "kan_Knda", "odia": "ory_Orya",
    "malayalam": "mal_Mlym", "punjabi": "pan_Guru", "nepali": "npi_Deva",
}

# Devanagari is shared by Hindi / Marathi / Nepali: a small marker lexicon (heuristic fallback only;
# IndicLID is better). Tuned on the VAL split only.
_DEV_MARKERS = {
    "mar_Deva": {"आहे", "आहेत", "मी", "माझे", "माझी", "माझा", "मला", "नाही", "आणि", "काय", "साठी", "मध्ये", "वय",
                 "वर्षे", "वर्षांचा", "राहतो", "राहते", "आमच्या", "आमचे", "कुटुंबाचे", "उत्पन्न", "मिळेल", "आहोत"},
    "npi_Deva": {"छ", "छु", "छन्", "मेरो", "हुन्छ", "लागि", "बस्छु", "बस्छे", "भएको", "हामी", "तपाईं", "हुँ", "म",
                 "मेरा", "हाम्रो", "परिवारको", "आम्दानी", "उमेर", "वर्षको", "कुनै", "छैन", "के"},
    "hin_Deva": {"है", "हैं", "मेरी", "हूँ", "हूं", "क्या", "लिए", "में", "मैं", "साल", "उम्र", "कोई", "परिवार", "सालाना"},
}


def label_to_code(label: str, text: str) -> str:
    """Dataset label + actual text -> FLORES code, script-aware.
    A 'bengali' query typed in Latin letters ("aamaar bayas 52") is ben_Latn, not ben_Beng:
    the pipeline must transliterate it before MT."""
    code = LABEL_TO_CODE[label.lower()]
    if code == "eng_Latn":
        return code
    letters = [c for c in text if c.isalpha()]
    indic = sum(any(lo <= ord(c) <= hi for lo, hi, *_ in _SCRIPTS) for c in letters)
    if letters and indic / len(letters) < 0.12:
        return code.split("_")[0] + "_Latn"
    return code


@dataclass
class LangResult:
    code: str               # e.g. "hin_Deva", "hin_Latn", "eng_Latn"
    confidence: float
    backend: str = "heuristic"

    @property
    def is_roman_indic(self) -> bool:
        return self.code.endswith("_Latn") and not self.code.startswith("eng")


class LangID:
    def __init__(self, use_indiclid: bool = True):
        self._model = None
        if use_indiclid:
            try:  # pip install from the AI4Bharat IndicLID repo; see README
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
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return LangResult("eng_Latn", 0.0)
        counts: dict[tuple[str, str], int] = {}
        latin = 0
        for c in letters:
            o = ord(c)
            for lo, hi, lang, _s in _SCRIPTS:
                if lo <= o <= hi:
                    counts[(lang, _s)] = counts.get((lang, _s), 0) + 1
                    break
            else:
                latin += 1
        if counts:
            (lang, _), n = max(counts.items(), key=lambda kv: kv[1])
            # code-mixed text is common (Indic sentence + Latin state names/typos), so a modest share
            # of Indic letters is enough: English never contains them.
            if n >= 3 and n / len(letters) >= 0.12:
                conf = min(0.99, 0.5 + n / len(letters))
                if lang == "ben_Beng" and re.search("[\u09F0\u09F1]", text):   # ৰ ৱ occur only in Assamese
                    lang = "asm_Beng"
                elif lang == "hin_Deva":
                    toks = set(re.findall(r"[\u0900-\u097F]+", text))
                    score = {k: len(toks & v) for k, v in _DEV_MARKERS.items()}
                    best = max(score, key=lambda k: (score[k], k == "hin_Deva"))
                    if score[best] > 0:
                        lang = best
                    conf = min(conf, 0.9)         # Devanagari language choice is only a guess here
                return LangResult(lang, conf)
        toks = re.findall(r"[a-z]+", text.lower())
        if not toks:
            return LangResult("eng_Latn", 0.5)
        hi = sum(t in _HI_ROMAN for t in toks)
        en = sum(t in _EN for t in toks)
        if hi >= 1 and hi / len(toks) >= 0.15 and hi >= en:
            return LangResult("hin_Latn", min(0.95, 0.5 + hi / len(toks)))
        return LangResult("eng_Latn", min(0.95, 0.6 + en / max(len(toks), 1)))


NATIVE_SCRIPT = {"hin": "Deva", "ben": "Beng", "pan": "Guru", "guj": "Gujr", "ory": "Orya",
                 "tam": "Taml", "tel": "Telu", "kan": "Knda", "mal": "Mlym", "mar": "Deva",
                 "asm": "Beng", "npi": "Deva"}
XLIT_CODE = {"asm": "as", "npi": "ne", "hin": "hi", "ben": "bn", "pan": "pa", "guj": "gu", "ory": "or", "tam": "ta",
             "tel": "te", "kan": "kn", "mal": "ml", "mar": "mr"}


def native_code(roman_code: str) -> str:
    """'hin_Latn' -> 'hin_Deva'"""
    stem = roman_code.split("_")[0]
    return f"{stem}_{NATIVE_SCRIPT.get(stem, 'Deva')}"
