"""Regex slot extraction from ENGLISH text (BASELINE stand-in for the team's slot extractor).
Returns only what is stated; anything not mentioned stays None (= unknown, never false).
Known gaps: no negation ("I am not a farmer"), no third-party handling ("my father is a farmer"),
no correction handling ("49... oops 47"): these are what the real extractor/Profile State Manager must do."""
from __future__ import annotations
import re
from difflib import SequenceMatcher

PROFILE_FIELDS = ("age", "income", "state", "gender", "caste", "disability", "bpl", "residence", "occupation")

_OCC = [("farmer", r"farmer|farming|agricultur|cultivat|kisan"),
        ("student", r"student|stdnt|studying|college|school|scholarship"),
        ("worker", r"worker|labou?r|daily wage|construction|mason|driver"),
        ("entrepreneur", r"entrepreneur|business|startup|start-up|shop ?owner|self[- ]employed")]
_CASTE = [("sc", r"\bsc\b|scheduled caste"), ("st", r"\bst\b|scheduled tribe|tribal"),
          ("obc", r"\bobc\b|other backward|backward class"), ("general", r"general (?:category|caste)|\bgeneral\b|\bunreserved\b")]


def _state_match(text: str, states: list[str]) -> str | None:
    low = re.sub(r"[^a-z& ]", " ", text.lower())
    toks = low.split()
    best, best_r = None, 0.0
    for st in sorted(states, key=len, reverse=True):
        if re.search(rf"\b{re.escape(st)}\b", low):
            return st
        n = len(st.split())
        for i in range(len(toks) - n + 1):               # typo tolerance ("uttarakhadn")
            r = SequenceMatcher(None, " ".join(toks[i:i + n]), st).ratio()
            if r > best_r:
                best, best_r = st, r
    return best if best_r >= 0.86 else None


def extract_slots(text: str, states: list[str]) -> dict:
    """`text` should be number-normalised English (TranslationLayer.to_english output)."""
    t = text
    low = t.lower()
    out = {k: None for k in PROFILE_FIELDS}

    m = (re.search(r"\b(?:age|aged)\b\D{0,8}(\d{1,3})\b", low) or re.search(r"\b(\d{1,3})\s*(?:years?|yrs?|yo|y/o)\b", low)
         or re.search(r"\bi am (\d{1,3})\b", low))
    if m and 0 < int(m.group(1)) < 120:
        out["age"] = int(m.group(1))

    inc = [float(x) for x in re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w])", low.replace(",", "")) if float(x) >= 1000]
    if inc:
        out["income"] = int(inc[0])

    if re.search(r"\b(female|woman|women|lady|girl|she)\b", low):
        out["gender"] = "female"
    elif re.search(r"\b(male|man|boy|gentleman)\b", low):
        out["gender"] = "male"

    out["state"] = _state_match(low, states)

    for val, pat in _CASTE:
        if re.search(pat, low):
            out["caste"] = val
            break
    for val, pat in _OCC:
        if re.search(pat, low):
            out["occupation"] = val
            break
    if re.search(r"disab|handicap|divyang|differently abled|blind|deaf|wheelchair|physically challenged", low):
        out["disability"] = True
    if re.search(r"\bbpl\b|below poverty|poverty line", low):
        out["bpl"] = True
    if re.search(r"village|rural|countryside", low):
        out["residence"] = "rural"
    elif re.search(r"\bcity\b|urban|\btown\b|metro", low):
        out["residence"] = "urban"
    return out


def merge(profile: dict | None, new: dict) -> dict:
    """Minimal Profile State Manager stand-in: newly stated facts overwrite; missing never erases."""
    merged = dict(profile or {})
    for k, v in new.items():
        if v is not None:
            merged[k] = v
    return merged
