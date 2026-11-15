"""End-to-end flow:  user text -> English -> slots -> retrieval -> eligibility -> English answer -> user's language.

Retrieval, slot extraction and eligibility here are simple BASELINES so the whole chain runs; replace them
with the team's trained components. The translation layer is the real one.
"""
from __future__ import annotations
import re
from collections import Counter

from janora.translation.langid import native_code
from janora.translation.pipeline import TranslationLayer

from .retrieve import SchemeIndex
from .rules import check
from .slots import extract_slots, merge

_ASK = {"state": "Which state do you live in?", "income": "What is your annual family income?",
        "age": "What is your age?", "occupation": "What is your occupation (farmer, student, worker...)?",
        "caste": "Which category do you belong to (SC, ST, OBC or General)?", "gender": "Are you male or female?",
        "residence": "Do you live in a rural or an urban area?", "disability": "Do you have a disability?",
        "bpl": "Do you have a BPL card?"}


def first_sentence(text: str, limit: int = 220) -> str:
    s = re.split(r"(?<=[.!?])\s+", (text or "").strip())[0]
    return s if len(s) <= limit else s[:limit].rsplit(" ", 1)[0] + "..."


class JanoraFlow:
    def __init__(self, layer: TranslationLayer, index: SchemeIndex, top_k: int = 3):
        self.layer, self.index, self.top_k = layer, index, top_k

    # -- English-only core (what the team's components will replace) ----------------------------------
    def reason(self, english: str, profile: dict | None = None) -> dict:
        slots = extract_slots(english, self.index.states)
        profile = merge(profile, slots)
        hits = self.index.search(english, k=60, state=profile.get("state"))
        yes, maybe = [], []
        for s, score in hits:
            status, unknown, _ = check(s["rule"], profile)
            if status == "yes":
                yes.append((s, score))
            elif status == "unknown":
                maybe.append((s, score, unknown))
        need = Counter(f for _, _, u in maybe[:10] for f in u)          # which missing fact matters most
        ask = need.most_common(1)[0][0] if need else None
        return {"profile": profile, "yes": yes, "maybe": maybe, "ask": ask}

    def compose_english(self, res: dict) -> str:
        p = res["profile"]
        known = ", ".join(f"{k} {v}" for k, v in p.items() if v is not None and k != "income") \
            + (f", annual income ₹{p['income']}" if p.get("income") else "")
        parts = []
        if res["yes"]:
            parts.append(f"Based on what you told me ({known}), you appear to be eligible for these schemes:" if known
                         else "You appear to be eligible for these schemes:")
            for i, (s, _) in enumerate(res["yes"][: self.top_k], 1):
                parts.append(f"{i}. {s['name']}. {first_sentence(s.get('description', ''))} Official link: {s['url']}")
        elif res["maybe"]:
            parts.append("I could not confirm eligibility yet, but these schemes may suit you:")
            for i, (s, _, unk) in enumerate(res["maybe"][: self.top_k], 1):
                parts.append(f"{i}. {s['name']}. {first_sentence(s.get('description', ''))} I still need: {', '.join(unk)}.")
        else:
            parts.append("I could not find a matching scheme for the details you gave.")
        if res["ask"]:
            parts.append(_ASK[res["ask"]])
        return "\n".join(parts)

    # -- full chain -----------------------------------------------------------------------------
    def ask(self, text: str, profile: dict | None = None, lang: str | None = None) -> dict:
        r = self.layer.to_english(text, forced_lang=lang)
        res = self.reason(r.english, profile)
        answer_en = self.compose_english(res)
        target = native_code(r.lang.code) if r.lang.is_roman_indic else r.lang.code   # Hinglish -> Devanagari
        out = self.layer.answer_with_disclaimer(answer_en, target)
        return {"input": r, "profile": res["profile"], "answer_en": answer_en, "output": out, "reasoning": res}
