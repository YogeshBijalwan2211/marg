"""Evaluate JANORA translation on the repository's query splits.

Examples:
  python src/eval_translation.py offline --split val
  python src/eval_translation.py mt --split val --backend nllb --limit 300
  python src/eval_translation.py mt --split val --backend nllb --no-translit --limit 300

Tune thresholds on validation data; run the test split only once for the final report.
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from janora.translation import Glossary, TranslationConfig, TranslationLayer  # noqa: E402
from janora.translation.langid import LangID, label_to_code  # noqa: E402
from janora.translation.normalise import clean_text, normalise_numbers  # noqa: E402


def load(split):
    path = Path("data/splits") / f"{split}.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def facts_present(text, profile, missing=()):
    """Return which stated age/income facts survived in the translated text."""
    nums = set(re.findall(r"\d+(?:\.\d+)?", text.replace(",", "")))
    result = {}
    if profile.get("age") is not None and "age" not in missing:
        result["age"] = str(profile["age"]) in nums
    if profile.get("income") is not None and any(float(x) >= 1000 for x in nums):
        result["income"] = str(int(profile["income"])) in nums
    return result


def pct(numerator, denominator):
    return round(100 * numerator / denominator, 1) if denominator else None


def offline(args):
    rows = [r for r in load(args.split) if r.get("row_type") != "offtopic"]
    glossary = Glossary.from_json("data/registry/glossary.json")
    langid = LangID(use_indiclid=not args.heuristic_langid)
    layer = TranslationLayer(TranslationConfig(backend="nllb"), glossary, langid=langid)
    by_language = collections.defaultdict(collections.Counter)
    confusions = collections.Counter()
    for row in rows:
        label = row["language"]
        gold = label_to_code(label, row["query"])
        counts = by_language[label]
        counts["n"] += 1
        counts["roman"] += gold.endswith("_Latn") and not gold.startswith("eng")
        predicted = langid.detect(row["query"]).code
        counts["langid_ok"] += predicted == gold
        if predicted != gold:
            confusions[(gold, predicted)] += 1
        masked = layer.masker.mask(
            clean_text(row["query"]),
            normaliser=lambda value: normalise_numbers(value)[0],
        )
        restored = layer.masker.unmask(masked.text, masked)
        for key, passed in facts_present(
            restored, row.get("true_profile") or {}, row.get("missing_fields") or []
        ).items():
            counts[f"{key}_n"] += 1
            counts[f"{key}_ok"] += passed
        if row["row_type"] == "named" and row.get("target_slug"):
            counts["named_n"] += 1
            counts["named_masked"] += any(
                slot.scheme_id == row["target_slug"] for slot in masked.slots.values()
            )

    print(f"split={args.split} (off-topic rows excluded)")
    print("Language ID is script-aware: Latin-script Indic input is labelled <lang>_Latn.")
    print(f"{'language':10} {'n':>5} {'langID%':>8} {'age%':>6} {'income%':>8} {'named->glossary%':>17}")
    total = collections.Counter()
    for label in sorted(by_language):
        counts = by_language[label]
        total.update(counts)
        print(
            f"{label:10} {counts['n']:5} {pct(counts['langid_ok'], counts['n']):8} "
            f"{pct(counts['age_ok'], counts['age_n']):6} "
            f"{pct(counts['income_ok'], counts['income_n']):8} "
            f"{str(pct(counts['named_masked'], counts['named_n'])):>17}"
        )
    print(
        f"{'ALL':10} {total['n']:5} {pct(total['langid_ok'], total['n']):8} "
        f"{pct(total['age_ok'], total['age_n']):6} "
        f"{pct(total['income_ok'], total['income_n']):8} "
        f"{str(pct(total['named_masked'], total['named_n'])):>17}"
    )
    print(f"Romanised non-English rows: {total['roman']} of {total['n']}")
    print("\nTop language-ID confusions (gold -> predicted):")
    for (gold, predicted), count in confusions.most_common(8):
        print(f"  {gold} -> {predicted}: {count}")


def mt(args):
    rows = [
        row for row in load(args.split)
        if row["row_type"] != "offtopic" and row["language"] != "english"
    ]
    if args.limit:
        rows = rows[:args.limit]
    glossary = Glossary.from_json("data/registry/glossary.json")
    config = TranslationConfig(
        backend=args.backend,
        transliterate_roman=not args.no_translit,
    )
    layer = TranslationLayer(
        config, glossary, langid=LangID(use_indiclid=False)
    )
    by_language = collections.defaultdict(collections.Counter)
    for row in rows:
        result = layer.to_english(
            row["query"],
            forced_lang=label_to_code(row["language"], row["query"]),
        )
        counts = by_language[row["language"]]
        counts["n"] += 1
        counts["gate_fail"] += bool(result.issues)
        english, _ = normalise_numbers(result.english)
        for key, passed in facts_present(
            english, row.get("true_profile") or {}, row.get("missing_fields") or []
        ).items():
            counts[f"{key}_n"] += 1
            counts[f"{key}_ok"] += passed
        state = (row.get("true_profile") or {}).get("state")
        if state:
            counts["state_n"] += 1
            counts["state_ok"] += state.lower() in result.english.lower()

    print(
        f"backend={args.backend} translit={not args.no_translit} split={args.split}"
    )
    print(f"{'language':10} {'n':>5} {'age%':>6} {'income%':>8} {'state%':>7} {'gate-fail%':>11}")
    for label in sorted(by_language):
        counts = by_language[label]
        print(
            f"{label:10} {counts['n']:5} {pct(counts['age_ok'], counts['age_n']):6} "
            f"{pct(counts['income_ok'], counts['income_n']):8} "
            f"{pct(counts['state_ok'], counts['state_n']):7} "
            f"{pct(counts['gate_fail'], counts['n']):11}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    offline_parser = subparsers.add_parser("offline")
    offline_parser.add_argument("--split", default="val")
    offline_parser.add_argument(
        "--heuristic-langid", action="store_true",
        help="skip IndicLID and use the built-in heuristic",
    )
    mt_parser = subparsers.add_parser("mt")
    mt_parser.add_argument("--split", default="val")
    mt_parser.add_argument(
        "--backend", default="nllb", choices=["nllb", "indictrans2"]
    )
    mt_parser.add_argument("--limit", type=int, default=0)
    mt_parser.add_argument("--no-translit", action="store_true")
    arguments = parser.parse_args()
    offline(arguments) if arguments.cmd == "offline" else mt(arguments)
