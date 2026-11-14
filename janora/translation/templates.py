"""Fixed safety / UX phrases. These must NOT be machine-translated.

!! The non-English strings below are DRAFTS written to make the pipeline runnable.
!! Have your human translator review/replace every one before the demo or evaluation,
!! and add the other languages. Anything missing falls back to English.
"""
from __future__ import annotations

TEMPLATES: dict[str, dict[str, str]] = {
    "likely_eligible": {
        "eng_Latn": "Likely eligible. Please verify on the official portal.",
        "hin_Deva": "आप संभवतः पात्र हैं। कृपया आधिकारिक पोर्टल पर इसकी पुष्टि करें।",
    },
    "likely_not_eligible": {
        "eng_Latn": "Likely not eligible based on the details provided. Please verify on the official portal.",
        "hin_Deva": "दी गई जानकारी के अनुसार आप संभवतः पात्र नहीं हैं। कृपया आधिकारिक पोर्टल पर पुष्टि करें।",
    },
    "need_more_info": {
        "eng_Latn": "I need more information to decide. {question}",
        "hin_Deva": "निर्णय के लिए मुझे और जानकारी चाहिए। {question}",
    },
    "not_found": {
        "eng_Latn": "I could not find this information.",
        "hin_Deva": "मुझे यह जानकारी नहीं मिल सकी।",
    },
    "verify_disclaimer": {
        "eng_Latn": "Please verify on the official portal before applying.",
        "hin_Deva": "आवेदन करने से पहले कृपया आधिकारिक पोर्टल पर पुष्टि करें।",
    },
    "confirm_slot": {
        "eng_Latn": "{slot}: {value}. Is this correct?",
        "hin_Deva": "{slot}: {value}। क्या यह सही है?",
    },
    "confirm_language": {
        "eng_Latn": "Which language would you like to use? / आप किस भाषा में बात करना चाहेंगे?",
        "hin_Deva": "आप किस भाषा में बात करना चाहेंगे? / Which language would you like to use?",
    },
    "translation_uncertain": {
        "eng_Latn": "The translation may be inaccurate, so the English answer is shown below.",
        "hin_Deva": "अनुवाद में त्रुटि हो सकती है, इसलिए अंग्रेज़ी उत्तर नीचे दिया गया है।",
    },
}

SLOT_LABELS: dict[str, dict[str, str]] = {
    "income": {"eng_Latn": "Annual income", "hin_Deva": "वार्षिक आय"},
    "age": {"eng_Latn": "Age", "hin_Deva": "आयु"},
    "state": {"eng_Latn": "State", "hin_Deva": "राज्य"},
    "category": {"eng_Latn": "Category", "hin_Deva": "श्रेणी"},
}


def render(key: str, lang: str, **vars) -> tuple[str, bool]:
    """Return (text, used_fallback)."""
    options = TEMPLATES[key]
    text, fallback = (options[lang], False) if lang in options else (options["eng_Latn"], True)
    return text.format(**vars), fallback


def slot_label(slot: str, lang: str) -> str:
    d = SLOT_LABELS.get(slot, {})
    return d.get(lang) or d.get("eng_Latn") or slot
