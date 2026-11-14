"""Evaluate the translation layer on this repo's query splits.

Run from the repo root.

  # 1) OFFLINE checks, no models needed (language ID, number normalisation, glossary masking):
  python src/eval_translation.py offline --split val

  # 2) END-TO-END check with a real MT model (needs GPU/CPU + model downloads):
  python src/eval_translation.py mt --split val --backend indictrans2 --limit 300
  python src/eval_translation.py mt --split val --backend nllb --limit 300
  python src/eval_translation.py mt --split val --backend indictrans2 --no-translit   # ablation

Golden rule from the project: tune on `val`; run `--split test` ONCE at the end.

The rows carry `true_profile` (age, income, state ...), so translation quality is measured as
"did the user's facts survive into the English text?" (slot preservation), per language.
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
from janora.translation.langid import LangID, label_to_code  # noqa: E402
from janora.translation.normalise import clean_text, normalise_numbers  # noqa: E402


def load(split):
    with open(f"data/splits/{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def facts_present(text, profile, missing=()):
    """Which STATED profile facts appear in `text` (already number-normalised)? age/income only.
    Fields listed in the row's `missing_fields` were never said by the user, so they are skipped."""
    nums = set(re.findall(r"\d+(?:\.\d+)?", text.replace(",", "")))
    res = {}
    if profile.get("age") is not None and "age" not in missing:
        res["age"] = str(profile["age"]) in nums
    # `missing_fields` does not reliably list income, so score income only on rows where the text
    # actually contains a large figure (>= 1000); then ask: is that figure the true income?
    if profile.get("income") is not None and any(float(x) >= 1000 for x in nums):
        res["income"] = str(int(profile["income"])) in nums
    return res


def pct(a, b):
    return round(100 * a / b, 1) if b else None


def offline(args):
    rows = [r for r in load(args.split) if r.get("row_type") != "offtopic"]
    gl = Glossary.from_json("data/registry/glossary.json")
    ld = LangID(use_indiclid=not args.heuristic_langid)
    layer = TranslationLayer(TranslationConfig(), gl, langid=ld)
    by = collections.defaultdict(lambda: collections.Counter())
    confusions = collections.Counter()
    for r in rows:
        lang, gold = r["language"], label_to_code(r["language"], r["query"])
        c = by[lang]
        c["n"] += 1
        c["roman"] += gold.endswith("_Latn") and not gold.startswith("eng")
        pred = ld.detect(r["query"]).code
        c["langid_ok"] += pred == gold
        if pred != gold:
            confusions[(gold, pred)] += 1
        masked = layer.masker.mask(clean_text(r["query"]), normaliser=lambda t: normalise_numbers(t)[0])
        normalised = layer.masker.unmask(masked.text, masked)
        for k, ok in facts_present(normalised, r.get("true_profile") or {}, r.get("missing_fields") or []).items():
            c[f"{k}_n"] += 1
            c[f"{k}_ok"] += ok
        if r["row_type"] == "named" and r.get("target_slug"):
            c["named_n"] += 1
            c["named_masked"] += any(s.scheme_id == r["target_slug"] for s in masked.slots.values())
    print(f"split={args.split}   (offtopic rows excluded)")
    print("langID is script-aware: a query typed in Latin letters counts as <lang>_Latn.")
    print(f"{'language':10} {'n':>5} {'langID%':>8} {'age%':>6} {'income%':>8} {'named->glossary%':>17}")
    tot = collections.Counter()
    for lang in sorted(by):
        c = by[lang]; tot.update(c)
        print(f"{lang:10} {c['n']:5} {pct(c['langid_ok'], c['n']):8} {pct(c['age_ok'], c['age_n']):6} "
              f"{pct(c['income_ok'], c['income_n']):8} {str(pct(c['named_masked'], c['named_n'])):>17}")
    print(f"{'ALL':10} {tot['n']:5} {pct(tot['langid_ok'], tot['n']):8} {pct(tot['age_ok'], tot['age_n']):6} "
          f"{pct(tot['income_ok'], tot['income_n']):8} {str(pct(tot['named_masked'], tot['named_n'])):>17}")
    print(f"romanised non-English/non-Hinglish-style rows: {tot['roman']} of {tot['n']}")
    print("\ntop language-ID confusions (gold -> predicted):")
    for (g, p), n in confusions.most_common(8):
        print(f"  {g} -> {p}: {n}")


def mt(args):
    rows = [r for r in load(args.split) if r["row_type"] != "offtopic" and r["language"] != "english"]
    if args.limit:                       # file is sorted by language: sample randomly, not rows[:N]
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.limit]
    all_rows = load(args.split)
    gl = Glossary.from_json("data/registry/glossary.json")
    cfg = TranslationConfig(backend=args.backend, transliterate_roman=not args.no_translit)
    layer = TranslationLayer(cfg, gl, langid=LangID(use_indiclid=False))
    by = collections.defaultdict(collections.Counter)
    shown = 0
    for r in rows:
        res = layer.to_english(r["query"], forced_lang=label_to_code(r["language"], r["query"]))
        c = by[r["language"]]
        c["n"] += 1
        c["gate_fail"] += bool(res.issues)
        if res.issues and shown < args.show_failures:
            shown += 1
            print(f"\n[gate fail] {r['language']} issues={res.issues}\n  query  : {r['query'][:200]}"
                  f"\n  english: {res.english[:200]}")
        # compare facts in the ENGLISH output with the gold profile
        eng, _ = normalise_numbers(res.english)
        for k, ok in facts_present(eng, r.get("true_profile") or {}, r.get("missing_fields") or []).items():
            c[f"{k}_n"] += 1
            c[f"{k}_ok"] += ok
        st = (r.get("true_profile") or {}).get("state")
        if st:
            c["state_n"] += 1
            c["state_ok"] += st.lower() in res.english.lower()
    eng = [r for r in all_rows if r["language"] == "english" and r["row_type"] != "offtopic"
           and (r.get("true_profile") or {}).get("state")]
    base = pct(sum(r["true_profile"]["state"].lower() in r["query"].lower() for r in eng), len(eng))
    print(f"\n(state% ceiling: in the {len(eng)} English rows, which need no translation, the state name "
          f"appears in the text only {base}% of the time because of typos/abbreviations)")
    print(f"backend={args.backend} translit={not args.no_translit} split={args.split}")
    print(f"{'language':10} {'n':>5} {'age%':>6} {'income%':>8} {'state%':>7} {'gate-fail%':>11}")
    for lang in sorted(by):
        c = by[lang]
        print(f"{lang:10} {c['n']:5} {pct(c['age_ok'], c['age_n']):6} {pct(c['income_ok'], c['income_n']):8} "
              f"{pct(c['state_ok'], c['state_n']):7} {pct(c['gate_fail'], c['n']):11}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("offline"); a.add_argument("--split", default="val")
    a.add_argument("--heuristic-langid", action="store_true", help="skip IndicLID, use built-in heuristic")
    m = sub.add_parser("mt"); m.add_argument("--split", default="val")
    m.add_argument("--backend", default="indictrans2", choices=["indictrans2", "nllb"])
    m.add_argument("--limit", type=int, default=0, help="random sample size; 0 = all non-English rows")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--show-failures", type=int, default=0, help="print N rows that tripped a quality gate")
    m.add_argument("--no-translit", action="store_true")
    args = ap.parse_args()
    offline(args) if args.cmd == "offline" else mt(args)
