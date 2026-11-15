import re
from janora.baseline.flow import JanoraFlow
from janora.baseline.retrieve import SchemeIndex
from janora.baseline.rules import check
from janora.baseline.slots import extract_slots, merge
from janora.translation import Glossary, TranslationConfig, TranslationLayer
from janora.translation.backends import MTBackend
from janora.translation.langid import LangID

STATES = ["gujarat", "tamil nadu", "uttarakhand"]
SCHEMES = [
    {"id": "a", "name": "Farmer Pension Yojana", "category": "Agriculture", "topics": [], "state": None, "level": "central",
     "description": "Pension for small farmers.", "url": "https://x/a", "rule": {"occupation": ["farmer", "entrepreneur"]}},
    {"id": "b", "name": "Girls Scholarship", "category": "Education", "topics": [], "state": "gujarat", "level": "state",
     "description": "Scholarship for girl students.", "url": "https://x/b", "rule": {"gender": "female", "occupation": "student", "state": "gujarat"}},
    {"id": "c", "name": "Low Income Housing", "category": "Housing", "topics": [], "state": None, "level": "central",
     "description": "Housing aid.", "url": "https://x/c", "rule": {"max_income": 300000}},
]


def test_rules_three_valued_and_list_values():
    assert check({"occupation": ["farmer", "entrepreneur"]}, {"occupation": "farmer"})[0] == "yes"
    assert check({"occupation": ["farmer", "entrepreneur"]}, {"occupation": "student"})[0] == "no"
    assert check({"max_income": 300000}, {})[0] == "unknown"             # missing means unknown, never false
    assert check({"max_income": 300000, "gender": "female"}, {"income": 500000})[0] == "no"
    assert check({"disability": True}, {"disability": False})[0] == "no"


def test_slots_extraction_and_typo_state():
    s = extract_slots("I am a 29 year old male farmer from Uttarakhadn, income 290000, I live in a village", STATES)
    assert (s["age"], s["income"], s["gender"], s["occupation"], s["residence"]) == (29, 290000, "male", "farmer", "rural")
    assert s["state"] == "uttarakhand" and s["disability"] is None and s["bpl"] is None
    assert extract_slots("I am a woman", STATES)["gender"] == "female"        # 'woman' must not match 'man'
    assert extract_slots("I am a farmer in the city", STATES)["residence"] == "urban"


def test_merge_never_erases():
    assert merge({"age": 29, "state": "gujarat"}, {"age": None, "state": "tamil nadu"}) == {"age": 29, "state": "tamil nadu"}


class FakeMT(MTBackend):
    def translate(self, sents, src, tgt):
        # the layer masks the number first, so the MT sees ZXN1X in place of 20
        table = {"मैं गुजरात की ZXN1X साल की छात्रा हूँ": "I am a ZXN1X year old female student from Gujarat"}
        return [table.get(s, s) for s in sents]


def test_full_flow_english_and_hindi_roundtrip():
    index = SchemeIndex(SCHEMES)
    layer = TranslationLayer(TranslationConfig(round_trip=False, output_placeholder_style="alpha"), Glossary(), FakeMT(), LangID(use_indiclid=False))
    flow = JanoraFlow(layer, index)
    en = flow.ask("I am a farmer, which scheme can I get?")
    assert en["output"].target_lang == "eng_Latn" and "Farmer Pension Yojana" in en["output"].text
    assert "Please verify on the official portal" in en["output"].text
    hi = flow.ask("मैं गुजरात की 20 साल की छात्रा हूँ", lang="hin_Deva")
    assert hi["profile"]["gender"] == "female" and hi["profile"]["state"] == "gujarat" and hi["profile"]["age"] == 20
    assert "Girls Scholarship" in hi["answer_en"] and hi["input"].lang.code == "hin_Deva"
    assert "आधिकारिक पोर्टल" in hi["output"].text or hi["output"].fell_back_to_english   # safety line in Hindi (or English fallback)
