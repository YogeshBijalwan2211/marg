import re

import pytest

from janora.translation import (
    Glossary, SchemeEntry, TranslationConfig, TranslationLayer
)
from janora.translation.backends import MTBackend, NLLBBackend
from janora.translation.langid import LangID
from janora.translation.normalise import format_indian, normalise_text


class FakeMT(MTBackend):
    """Dictionary translator that preserves placeholders and records code pairs."""

    def __init__(self, table, drop_placeholders=False, mangle_digits=False):
        self.table = table
        self.drop = drop_placeholders
        self.mangle = mangle_digits
        self.calls = []

    def translate(self, sentences, src, tgt):
        self.calls.append((src, tgt))
        out = []
        for sentence in sentences:
            translated = self.table.get(sentence, sentence)
            if self.drop:
                translated = re.sub(r"ZX[A-Z]\\d+X", "", translated)
            if self.mangle:
                translated += " 7"
            out.append(translated)
        return out


class FakeXlit:
    def transliterate(self, text, lang):
        return text

    def transliterate_to_roman(self, text, lang):
        return f"roman:{text}"


class FakeTokenizer:
    unk_token_id = 0
    lang_code_to_id = {"hin_Deva": 17, "eng_Latn": 9}

    def convert_tokens_to_ids(self, token):
        return self.lang_code_to_id.get(token, self.unk_token_id)


GLOSS = Glossary([
    SchemeEntry(
        "S1", "PM-KISAN",
        {"hin_Deva": ["पीएम किसान"], "hin_Latn": ["pm kisan"]},
        {"hin_Deva": "पीएम-किसान"},
    )
])


def layer(table, **kwargs):
    config = TranslationConfig(round_trip=kwargs.pop("round_trip", False))
    return TranslationLayer(
        config, GLOSS, FakeMT(table, **kwargs),
        LangID(use_indiclid=False), FakeXlit()
    )


@pytest.mark.parametrize("raw,expected", [
    ("मेरी आय ३ लाख है", "मेरी आय 300000 है"),
    ("income 3,00,000", "income 300000"),
    ("income 2.5 lakh", "income 250000"),
    ("1 crore", "10000000"),
    ("teen lakh", "300000"),
    ("तीन लाख", "300000"),
    ("5 hazaar", "5000"),
    ("10,20,30", "10,20,30"),
    ("3 लाखों", "3 लाखों"),
])
def test_normalise(raw, expected):
    assert normalise_text(raw)[0] == expected


def test_indian_format():
    assert format_indian("300000") == "3,00,000"
    assert format_indian("10000000") == "1,00,00,000"
    assert format_indian("999") == "999"


def test_langid():
    detector = LangID(use_indiclid=False)
    assert detector.detect("मेरी आय 3 लाख है").code == "hin_Deva"
    assert detector.detect("meri income 3 lakh hai kya mujhe milega").code == "hin_Latn"
    assert detector.detect("What is the income limit for this scheme?").code == "eng_Latn"
    assert detector.detect("எனது வயது 40").code == "tam_Taml"


def test_nllb_backend_validates_codes_using_tokenizer_mapping():
    backend = object.__new__(NLLBBackend)
    backend.tok = FakeTokenizer()
    assert backend._language_id("hin_Deva") == 17
    assert backend._language_id("eng_Latn") == 9
    with pytest.raises(ValueError, match="not supported"):
        backend._language_id("hin_Latn")
    with pytest.raises(ValueError, match="not supported"):
        backend._language_id("foo_Bar")


def test_input_hindi_roundtrip_numbers_and_scheme():
    source = "पीएम किसान के लिए मेरी आय ३ लाख है।"
    masked_source = "ZXS1X के लिए मेरी आय ZXN2X है।"
    result = layer({
        masked_source: "My income is ZXN2X for ZXS1X."
    }).to_english(source)
    assert result.english == "My income is 300000 for PM-KISAN."
    assert not result.issues and not result.needs_confirmation


def test_roman_hindi_input_uses_native_nllb_code():
    translator = layer({})
    result = translator.to_english("meri income 3 lakh hai")
    assert ("hin_Deva", "eng_Latn") in translator.backend.calls


def test_roman_hindi_output_uses_native_nllb_then_romanizes():
    translator = layer({"Hello.": "नमस्ते।"})
    result = translator.from_english("Hello.", "hin_Latn")
    assert ("eng_Latn", "hin_Deva") in translator.backend.calls
    assert result.text.startswith("roman:")


def test_input_placeholder_loss_flags():
    result = layer({}, drop_placeholders=True).to_english(
        "पीएम किसान के लिए मेरी आय ३ लाख है।"
    )
    assert "placeholder_mismatch" in result.issues
    assert result.needs_confirmation


def test_stray_digit_flags():
    result = layer({}, mangle_digits=True).to_english("मेरी आय ३ लाख है")
    assert "number_changed" in result.issues


def test_english_passthrough_still_normalises():
    result = layer({}).to_english("my income is 3 lakh")
    assert result.english == "my income is 300000"
    assert result.lang.code == "eng_Latn"


def test_low_confidence_asks_language():
    assert layer({}).to_english("12345").ask_language


def test_output_uses_native_name_and_indian_grouping():
    masked = "You may get ZXM2X from ZXS1X."
    result = layer({
        masked: "आपको ZXS1X से ZXM2X मिल सकते हैं।"
    }).from_english("You may get ₹6000 from PM-KISAN.", "hin_Deva")
    assert result.text == "आपको पीएम-किसान से ₹6,000 मिल सकते हैं।"
    assert not result.issues


def test_output_gate_falls_back_to_english():
    result = layer({}, drop_placeholders=True).from_english(
        "Benefit is ₹6000.", "hin_Deva"
    )
    assert result.fell_back_to_english
    assert result.text == "Benefit is ₹6000."


def test_round_trip_flags_low_similarity():
    translator = layer({}, round_trip=True)
    translator.cfg.round_trip_min_chrf = 99.0
    translator.backend.table = {
        "Farmers get money.": "किसानों को पैसा मिलता है।",
        "किसानों को पैसा मिलता है।": "Totally different words here",
    }
    result = translator.from_english("Farmers get money.", "hin_Deva")
    assert "low_round_trip_similarity" in result.issues
    assert result.show_english_alongside


def test_disclaimer_is_human_template():
    result = layer({"Apply online.": "ऑनलाइन आवेदन करें।"}).answer_with_disclaimer(
        "Apply online.", "hin_Deva"
    )
    assert "आधिकारिक पोर्टल" in result.text


def test_missing_language_template_falls_back_and_flags():
    result = layer({}).answer_with_disclaimer("Apply online.", "tam_Taml")
    assert "template_missing_for_language" in result.issues


def test_confirm_slot_romanizes_output_language():
    result = layer({}).confirm_slot("age", "40", "hin_Latn")
    assert result.startswith("roman:")


def test_noisy_real_inputs():
    noisy = [
        ("आय 2 करोड़ है", "आय 20000000 है"),
        ("आय 1 हज़ार", "आय 1000"),
        ("0.9 lak", "90000"),
        ("வருமானம் ௦.௭ லட்சம்", "வருமானம் 70000"),
        ("آمدنی 2 لاکھ", "آمدنی 200000"),
        ("hello 😊🙏  there", "hello there"),
    ]
    for raw, expected in noisy:
        assert normalise_text(raw)[0] == expected


def test_scheme_name_with_amount_is_matched_before_number_normalisation():
    glossary = Glossary([
        SchemeEntry("S9", "6% Interest Subsidy on Loans up to Rs.5 Lakh")
    ])
    translator = TranslationLayer(
        TranslationConfig(round_trip=False), glossary, FakeMT({}),
        LangID(use_indiclid=False), FakeXlit()
    )
    from janora.translation.normalise import clean_text, normalise_numbers
    masked = translator.masker.mask(
        clean_text("मुझे 6% Interest Subsidy on Loans up to Rs.5 Lakh चाहिए, आय 3 लाख"),
        normaliser=lambda value: normalise_numbers(value)[0],
    )
    assert any(slot.scheme_id == "S9" for slot in masked.slots.values())
    assert "300000" in translator.masker.unmask(masked.text, masked)


def test_label_to_code_is_script_aware():
    from janora.translation.langid import label_to_code
    assert label_to_code("bengali", "আমার বয়স 52") == "ben_Beng"
    assert label_to_code("bengali", "aamaar bayas 52 bachha") == "ben_Latn"
    assert label_to_code("english", "anything") == "eng_Latn"


def test_assamese_marathi_nepali_vs_hindi():
    detector = LangID(use_indiclid=False)
    assert detector.detect("মোৰ বয়স ৫২ বছৰ").code == "asm_Beng"
    assert detector.detect("माझे वय ४० वर्षे आहे").code == "mar_Deva"
    assert detector.detect("मेरो उमेर ४० वर्ष हो र म बिहारमा बस्छु").code == "npi_Deva"
    assert detector.detect("मेरी उम्र 40 साल है").code == "hin_Deva"
