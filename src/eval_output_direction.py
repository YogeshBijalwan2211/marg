"""Evaluate the OUTPUT direction: English answer -> user's language (En->Indic), then back.

Run from the repo root (needs the en-indic model; for IndicTrans2 accept the terms of
ai4bharat/indictrans2-en-indic-dist-200M on Hugging Face and run huggingface_hub.login() once).

  python src/eval_output_direction.py --split val --n 30 --backend indictrans2 --show 1
  python src/eval_output_direction.py --split val --n 30 --backend nllb
  python src/eval_output_direction.py --debug ben_Beng --placeholders digits   # why a language falls back

Answers are built from REAL scheme records of the chosen split (data/rag_schemes.json + the frozen
split map): description, an eligibility sentence from the scheme's rule, and the official link,
i.e. the kind of text the generator will produce. No human references exist yet, so the checks are:

  fallback%   gate: a placeholder (scheme name / amount / URL) was lost even after retries -> English shown
  english%    answer translated but flagged (low round-trip similarity or changed digits) -> English shown too
  numbers%    numbers of the English answer that survive in the output (Indian grouping accounted for)
  script%     output is really in the target script (catches 'model just copied the English')
  rt_chrF     round-trip similarity (back-translation vs original), the automatic quality flag
  name%       Hindi only: the glossary's Hindi scheme name was inserted
Tune on `val`; run `--split test` once at the end.
"""
import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from janora.translation import Glossary, TranslationConfig, TranslationLayer  # noqa: E402
from janora.translation.langid import LangID, _SCRIPTS  # noqa: E402
from janora.translation.normalise import normalise_text  # noqa: E402
from janora.translation.templates import TEMPLATES  # noqa: E402

TARGETS = ["hin_Deva", "ben_Beng", "asm_Beng", "guj_Gujr", "kan_Knda", "mal_Mlym", "mar_Deva",
           "npi_Deva", "ory_Orya", "pan_Guru", "tam_Taml", "tel_Telu", "urd_Arab"]


def pct(a, b):
    return round(100 * a / b, 1) if b else None


def eligibility_sentence(rule: dict, state: str) -> str:
    bits = []
    if rule.get("state") or state:
        bits.append(f"residents of {(rule.get('state') or state).title()}")
    if rule.get("min_age") and rule.get("max_age"):
        bits.append(f"aged {rule['min_age']} to {rule['max_age']} years")
    elif rule.get("min_age"):
        bits.append(f"aged {rule['min_age']} years or above")
    elif rule.get("max_age"):
        bits.append(f"aged up to {rule['max_age']} years")
    if rule.get("max_income"):
        bits.append(f"with annual family income up to ₹{int(rule['max_income'])}")
    if rule.get("occupation"):
        bits.append(f"working as {rule['occupation']}")
    return ("This scheme is for " + ", ".join(bits) + ".") if bits else ""


def build_answer(s: dict) -> str:
    desc = " ".join(re.split(r"(?<=[.!?])\s+", s["description"].strip())[:2])
    parts = [f"You may be eligible for {s['name']}.", desc, eligibility_sentence(s.get("rule") or {}, s.get("state") or ""),
             f"Official link: {s['url']}"]
    return " ".join(p for p in parts if p)


def load_answers(split: str, n: int, seed: int):
    schemes = json.load(open("data/rag_schemes.json", encoding="utf-8"))
    split_map = json.load(open("data/registry/scheme_split_v1.json", encoding="utf-8"))
    pool = [s for s in schemes if split_map.get(s["id"]) == split and len(s.get("description", "")) >= 80]
    random.Random(seed).shuffle(pool)
    return [(s, build_answer(s)) for s in pool[:n]]


def script_share(text: str, tgt: str, glossary_names: list[str]) -> float:
    """Share of letters in the target script, ignoring URLs and scheme names (those stay Latin)."""
    text = re.sub(r"https?://\S+", " ", text)
    for nm in glossary_names:
        text = text.replace(nm, " ")
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    suffix = tgt.split("_")[1]
    ranges = [(lo, hi) for lo, hi, _l, sc in _SCRIPTS if sc == suffix]
    return sum(any(lo <= ord(c) <= hi for lo, hi in ranges) for c in letters) / len(letters)


def numbers(text: str) -> collections.Counter:
    return collections.Counter(re.findall(r"\d+(?:\.\d+)?", normalise_text(text)[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=30, help="answers per language")
    ap.add_argument("--backend", default="indictrans2", choices=["indictrans2", "nllb"])
    ap.add_argument("--langs", nargs="*", default=TARGETS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-round-trip", action="store_true")
    ap.add_argument("--placeholders", default="digits", choices=["alpha", "digits"],
                    help="output-side placeholder style; 'digits' avoids models transliterating ZX..X tokens")
    ap.add_argument("--debug", metavar="LANG", help="e.g. ben_Beng: print the RAW masked source and RAW MT output "
                    "for the first answer (to see WHY a language falls back to English), then exit")
    ap.add_argument("--show", type=int, default=1, help="print this many example answers per language")
    a = ap.parse_args()

    gl = Glossary.from_json("data/registry/glossary.json")
    layer = TranslationLayer(TranslationConfig(backend=a.backend, round_trip=not a.no_round_trip,
                                                  output_placeholder_style=a.placeholders), gl,
                             langid=LangID(use_indiclid=False))
    answers = load_answers(a.split, a.n, a.seed)
    if a.debug:
        from janora.translation.pipeline import split_sentences, split_chunks
        masked = layer.out_masker.mask(answers[0][1])
        print("MASKED SOURCE:\n", masked.text, "\n")
        for name, pieces in (("sentences", split_sentences(masked.text)), ("chunks(12)", split_chunks(masked.text, 12))):
            out = layer.backend.translate(pieces, "eng_Latn", a.debug)
            print(f"RAW MT OUTPUT ({name}):")
            for src, o in zip(pieces, out):
                print("  src:", src[:150], "\n  out:", o[:200])
            print("  placeholders found:", dict(layer.out_masker.placeholders_in(" ".join(out))),
                  " expected:", sorted(sl.key for sl in masked.slots.values()), "\n")
        return
    print(f"backend={a.backend} split={a.split} answers={len(answers)} "
          f"(e.g. {answers[0][1][:110]!r}...)\n")
    rows = []
    for tgt in a.langs:
        c = collections.Counter()
        scores = []
        shown = 0
        for s, en in answers:
            res = layer.from_english(en, tgt)
            c["n"] += 1
            c["fallback"] += res.fell_back_to_english
            c["english_shown"] += res.show_english_alongside and not res.fell_back_to_english
            if res.round_trip_score is not None:
                scores.append(res.round_trip_score)
            if not res.fell_back_to_english:
                want, got = numbers(en), numbers(res.text)
                c["num_hit"] += sum((want & got).values())
                c["num_tot"] += sum(want.values())
                names = [gl.render(s["id"], tgt), s["name"]]
                c["script_n"] += 1
                c["script_ok"] += script_share(res.text, tgt, names) >= 0.5
                if tgt == "hin_Deva" and s["id"] in gl.entries:
                    c["name_n"] += 1
                    c["name_ok"] += gl.render(s["id"], tgt) in res.text
            if shown < a.show:
                shown += 1
                print(f"--- {tgt} example (issues={res.issues}, rt_chrF={None if res.round_trip_score is None else round(res.round_trip_score, 1)})")
                print(res.text[:400], "\n")
        rows.append((tgt, c, sum(scores) / len(scores) if scores else None))

    print(f"{'target':9} {'n':>3} {'fallback%':>10} {'english%':>9} {'numbers%':>9} {'script%':>8} {'rt_chrF':>8} {'name%':>6}")
    for tgt, c, rt in rows:
        print(f"{tgt:9} {c['n']:3} {pct(c['fallback'], c['n']):10} {pct(c['english_shown'], c['n']):9} "
              f"{str(pct(c['num_hit'], c['num_tot'])):>9} {str(pct(c['script_ok'], c['script_n'])):>8} "
              f"{'-' if rt is None else round(rt, 1):>8} {str(pct(c['name_ok'], c['name_n'])):>6}")
    have = sorted({l for t in TEMPLATES.values() for l in t})
    missing = [t for t in a.langs if t not in have]
    print(f"\nHuman-translated templates exist for: {have}\nMISSING (translator needed): {missing}")
    print("Note: name% is only meaningful for hin_Deva (the glossary has Hindi names only); other languages keep the English name.")


if __name__ == "__main__":
    main()
