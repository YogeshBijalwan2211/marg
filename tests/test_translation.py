import re
import pytest
from janora.translation import Glossary, SchemeEntry, TranslationConfig, TranslationLayer
from janora.translation.backends import MTBackend
from janora.translation.langid import LangID
from janora.translation.normalise import format_indian, normalise_text


class FakeMT(MTBackend):
    """Dictionary 'translator' that preserves placeholders; can be made lossy."""
    def __init__(self, table, drop_placeholders=False, mangle_digits=False):
        self.table, self.drop, self.mangle = table, drop_placeholders, mangle_digits
    def translate(self, sents, src, tgt):
        out = []
        for s in sents:
            t = self.table.get(s, s)
            if self.drop: t = re.sub(r"ZX[A-Z]\d+X", "", t)
            if self.mangle: t = t + " 7"
            out.append(t)
        return out


class FakeXlit:
    def transliterate(self, text, lang): return text


GLOSS = Glossary([SchemeEntry("S1", "PM-KISAN", {"hin_Deva": ["पीएम किसान"], "hin_Latn": ["pm kisan"]},
                              {"hin_Deva": "पीएम-किसान"})])


def layer(table, **kw):
    cfg = TranslationConfig(round_trip=kw.pop("round_trip", False))
    return TranslationLayer(cfg, GLOSS, FakeMT(table, **kw), LangID(use_indiclid=False), FakeXlit())


@pytest.mark.parametrize("raw,expected", [
    ("मेरी आय ३ लाख है", "मेरी आय 300000 है"),
    ("income 3,00,000", "income 300000"),
    ("income 2.5 lakh", "income 250000"),
    ("1 crore", "10000000"),
    ("teen lakh", "300000"),
    ("तीन लाख", "300000"),
    ("5 hazaar", "5000"),
    ("10,20,30", "10,20,30"),       # a list, not a grouped number
    ("3 लाखों", "3 लाखों"),          # inflected form is left alone
])
def test_normalise(raw, expected):
    assert normalise_text(raw)[0] == expected


def test_indian_format():
    assert format_indian("300000") == "3,00,000"
    assert format_indian("10000000") == "1,00,00,000"
    assert format_indian("999") == "999"


def test_langid():
    ld = LangID(use_indiclid=False)
    assert ld.detect("मेरी आय 3 लाख है").code == "hin_Deva"
    assert ld.detect("meri income 3 lakh hai kya mujhe milega").code == "hin_Latn"
    assert ld.detect("What is the income limit for this scheme?").code == "eng_Latn"
    assert ld.detect("எனது வயது 40").code == "tam_Taml"


def test_input_hindi_roundtrip_numbers_and_scheme():
    src = "पीएम किसान के लिए मेरी आय ३ लाख है।"
    # what the MT sees after normalisation + masking
    masked_src = "ZXS1X के लिए मेरी आय ZXN2X है।"
    l = layer({masked_src: "My income is ZXN2X for ZXS1X."})
    r = l.to_english(src)
    assert r.english == "My income is 300000 for PM-KISAN."
    assert not r.issues and not r.needs_confirmation


def test_input_placeholder_loss_flags():
    src = "पीएम किसान के लिए मेरी आय ३ लाख है।"
    r = layer({}, drop_placeholders=True).to_english(src)
    assert "placeholder_mismatch" in r.issues and r.needs_confirmation


def test_stray_digit_flags():
    r = layer({}, mangle_digits=True).to_english("मेरी आय ३ लाख है")
    assert "number_changed" in r.issues


def test_english_passthrough_still_normalises():
    r = layer({}).to_english("my income is 3 lakh")
    assert r.english == "my income is 300000" and r.lang.code == "eng_Latn"


def test_low_confidence_asks_language():
    r = layer({}).to_english("12345")
    assert r.ask_language


def test_output_uses_native_name_and_indian_grouping():
    en = "You may get ₹6000 from PM-KISAN."
    masked = "You may get ZXM2X from ZXS1X."
    l = layer({masked: "आपको ZXS1X से ZXM2X मिल सकते हैं।"})
    r = l.from_english(en, "hin_Deva")
    assert r.text == "आपको पीएम-किसान से ₹6,000 मिल सकते हैं।"
    assert not r.issues


def test_output_gate_falls_back_to_english():
    r = layer({}, drop_placeholders=True).from_english("Benefit is ₹6000.", "hin_Deva")
    assert r.fell_back_to_english and r.text == "Benefit is ₹6000."


def test_round_trip_flags_low_similarity():
    l = layer({}, round_trip=True)
    l.cfg.round_trip_min_chrf = 99.0
    # fake MT is identity -> perfect similarity, so force a low-sim case via table
    l.backend.table = {"Farmers get money.": "किसानों को पैसा मिलता है।", "किसानों को पैसा मिलता है।": "Totally different words here"}
    r = l.from_english("Farmers get money.", "hin_Deva")
    assert "low_round_trip_similarity" in r.issues and r.show_english_alongside


def test_disclaimer_is_human_template():
    r = layer({"Apply online.": "ऑनलाइन आवेदन करें।"}).answer_with_disclaimer("Apply online.", "hin_Deva")
    assert "आधिकारिक पोर्टल" in r.text


def test_missing_language_template_falls_back_and_flags():
    r = layer({}).answer_with_disclaimer("Apply online.", "tam_Taml")
    assert "template_missing_for_language" in r.issues


def test_confirm_slot():
    assert layer({}).confirm_slot("income", "₹3,00,000", "hin_Deva") == "वार्षिक आय: ₹3,00,000। क्या यह सही है?"


# ---- regressions found on the real MARG data -------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("आय 2 करोड़ है", "आय 20000000 है"),                 # nukta letters decompose under NFC
    ("आय 1 हज़ार", "आय 1000"),
    ("0.9 lak", "90000"),                                  # typo'd unit
    ("வருமானம் ௦.௭ லட்சம்", "வருமானம் 70000"),           # Tamil digits + Tamil 'lakh'
    ("آمدنی 2 لاکھ", "آمدنی 200000"),                     # Urdu
    ("hello 😊🙏  there", "hello there"),                   # emoji + extra spaces
])
def test_noisy_real_inputs(raw, expected):
    assert normalise_text(raw)[0] == expected


def test_scheme_name_with_amount_is_matched_before_number_normalisation():
    g = Glossary([SchemeEntry("S9", "6% Interest Subsidy on Loans up to Rs.5 Lakh")])
    l = TranslationLayer(TranslationConfig(round_trip=False), g, FakeMT({}), LangID(use_indiclid=False), FakeXlit())
    from janora.translation.normalise import clean_text, normalise_numbers
    m = l.masker.mask(clean_text("मुझे 6% Interest Subsidy on Loans up to Rs.5 Lakh चाहिए, आय 3 लाख"),
                      normaliser=lambda t: normalise_numbers(t)[0])
    assert any(s.scheme_id == "S9" for s in m.slots.values())
    assert "300000" in l.masker.unmask(m.text, m)


def test_label_to_code_is_script_aware():
    from janora.translation.langid import label_to_code
    assert label_to_code("bengali", "আমার বয়স 52") == "ben_Beng"
    assert label_to_code("bengali", "aamaar bayas 52 bachha") == "ben_Latn"
    assert label_to_code("english", "anything") == "eng_Latn"


def test_assamese_marathi_nepali_vs_hindi():
    ld = LangID(use_indiclid=False)
    assert ld.detect("মোৰ বয়স ৫২ বছৰ").code == "asm_Beng"
    assert ld.detect("माझे वय ४० वर्षे आहे").code == "mar_Deva"
    assert ld.detect("मेरो उमेर ४० वर्ष हो र म बिहारमा बस्छु").code == "npi_Deva"
    assert ld.detect("मेरी उम्र 40 साल है").code == "hin_Deva"


def test_translit_degrades_gracefully_without_indicxlit():
    from janora.translation.translit import Transliterator
    t = Transliterator()
    with pytest.warns(UserWarning):
        assert t.transliterate("meri income ZXN1X hai", "hin_Latn") == "meri income ZXN1X hai"
    assert t.available is False
