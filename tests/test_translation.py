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
    cfg = TranslationConfig(round_trip=kw.pop("round_trip", False), output_placeholder_style="alpha")
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


# ---- MT drops content from long inputs (seen with IndicTrans2/NLLB on real queries) ------------
class DropsOnLongInput(MTBackend):
    """Loses placeholders whenever a piece has more than 8 words, like real MT on run-on queries."""
    def translate(self, sents, src, tgt):
        return [re.sub(r"ZX[A-Z]\d+X", "", s) if len(s.split()) > 8 else s for s in sents]


def test_split_chunks_balanced_and_keeps_placeholders():
    from janora.translation.pipeline import split_chunks
    chunks = split_chunks("a b c d e f g h i j k l m ZXN1X, n o", 6)
    assert all(len(c.split()) <= 6 for c in chunks)
    assert sum("ZXN1X" in c for c in chunks) == 1
    assert " ".join(chunks).replace(",", "").split() == "a b c d e f g h i j k l m ZXN1X n o".split()


def test_chunk_retry_recovers_dropped_number():
    cfg = TranslationConfig(round_trip=False, chunk_words=12, small_chunk_words=6)
    l = TranslationLayer(cfg, GLOSS, DropsOnLongInput(), LangID(use_indiclid=False), FakeXlit())
    r = l.to_english("a b c d e f g h i j k आय 3 लाख", forced_lang="hin_Deva")   # one 13-word unpunctuated piece
    assert not r.issues and "300000" in r.english


# ---- repetition loops (seen: Malayalam output "ആൻഡ് ആൻഡ് ആൻഡ് ...") -------------------------
def test_has_repetition():
    from janora.translation.pipeline import has_repetition
    assert has_repetition("under the and and and and and board")
    assert has_repetition("x a b a b a b a b y")
    assert not has_repetition("pension shall be provided to workers who are registered")
    assert not has_repetition("55 55 वर्ष")


class LoopsOnLongInput(MTBackend):
    def translate(self, sents, src, tgt):
        return [("and " * 6 + s) if len(s.split()) > 8 else s for s in sents]


def test_repetition_triggers_chunk_retry_then_recovers():
    cfg = TranslationConfig(round_trip=False)
    l = TranslationLayer(cfg, GLOSS, LoopsOnLongInput(), LangID(use_indiclid=False), FakeXlit())
    r = l.from_english("Pension is paid to workers who are registered under the board after 55 years of age.", "hin_Deva")
    assert not r.fell_back_to_english and "repetition_loop" not in r.issues


def test_persistent_repetition_falls_back_to_english():
    class AlwaysLoops(MTBackend):
        def translate(self, sents, src, tgt): return ["and and and and and " + s for s in sents]
    l = TranslationLayer(TranslationConfig(round_trip=False), GLOSS, AlwaysLoops(), LangID(use_indiclid=False), FakeXlit())
    r = l.from_english("Pension is paid.", "hin_Deva")
    assert r.fell_back_to_english and "repetition_loop" in r.issues and r.text == "Pension is paid."


# ---- placeholders transliterated by En->Indic models (seen: Bengali ZXN3X -> জেডএক্সএন3এক্স) -----
class TransliteratesAlpha(MTBackend):
    """Mimics the observed failure: Latin ZX..X tokens get rewritten in Bengali letters.
    Numeric tokens are copied (optionally as native digits)."""
    def __init__(self, native_digits=False): self.native = native_digits
    def translate(self, sents, src, tgt):
        out = []
        for s in sents:
            s = re.sub(r"ZX([SMNU])(\d+)X", lambda m: "জেডএক্স" + m.group(1) + m.group(2) + "এক্স", s)
            if self.native:
                s = s.translate(str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯"))
            out.append(s)
        return out


ANSWER = "Pension of ₹250000 is paid to PM-KISAN workers aged 55 years. Official link: https://www.myscheme.gov.in/schemes/x"


def test_alpha_placeholders_lost_when_model_transliterates():
    cfg = TranslationConfig(round_trip=False, output_placeholder_style="alpha")
    r = TranslationLayer(cfg, GLOSS, TransliteratesAlpha(), LangID(use_indiclid=False), FakeXlit()).from_english(ANSWER, "ben_Beng")
    assert r.fell_back_to_english                      # the failure seen on Bengali/Gujarati/Kannada/Odia/Urdu


@pytest.mark.parametrize("native", [False, True])
def test_digit_placeholders_survive_and_restore(native):
    cfg = TranslationConfig(round_trip=False, output_placeholder_style="digits")
    r = TranslationLayer(cfg, GLOSS, TransliteratesAlpha(native), LangID(use_indiclid=False), FakeXlit()).from_english(ANSWER, "ben_Beng")
    assert not r.fell_back_to_english and not r.issues
    assert "₹2,50,000" in r.text and "55" in r.text
    assert "https://www.myscheme.gov.in/schemes/x" in r.text and "PM-KISAN" in r.text


def test_digit_style_flags_lost_token():
    class DropsDigits(MTBackend):
        def translate(self, sents, src, tgt): return [re.sub(r"9[1-4]\d{3}", "", s) for s in sents]
    cfg = TranslationConfig(round_trip=False, output_placeholder_style="digits")
    r = TranslationLayer(cfg, GLOSS, DropsDigits(), LangID(use_indiclid=False), FakeXlit()).from_english(ANSWER, "ben_Beng")
    assert r.fell_back_to_english and "placeholder_mismatch" in r.issues
