"""Measure the BASELINE flow on ENGLISH rows of a split (no MT model needed):
  slots     per-field coverage and precision of extracted facts vs true_profile
  retrieval 'named' rows: is the target scheme in the top-k of TF-IDF search?
  eligible  'standard'/'vague' rows: share of the top-3 recommended schemes that are truly eligible
            under the TRUE profile, and recall@10 of the dataset's relevant_slugs
Tune on val; run test once at the end.   python src/eval_flow.py --split val
"""
import argparse
import ast
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from janora.baseline.flow import JanoraFlow  # noqa: E402
from janora.baseline.retrieve import SchemeIndex  # noqa: E402
from janora.baseline.rules import check  # noqa: E402
from janora.baseline.slots import PROFILE_FIELDS, extract_slots  # noqa: E402
from janora.translation import TranslationLayer  # noqa: E402
from janora.translation.normalise import normalise_text  # noqa: E402

lst = lambda v: v if isinstance(v, list) else ast.literal_eval(v)
pct = lambda a, b: round(100 * a / b, 1) if b else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    a = ap.parse_args()
    index = SchemeIndex.from_json("data/rag_schemes.json")
    flow = JanoraFlow(TranslationLayer(), index)
    rows = [json.loads(l) for l in open(f"data/splits/{a.split}.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["language"] == "english" and r["row_type"] != "offtopic"]
    slot = collections.defaultdict(collections.Counter)
    named = collections.Counter()
    elig = collections.Counter()
    for r in rows:
        text, _ = normalise_text(r["query"])
        truth = r["true_profile"]
        got = extract_slots(text, index.states)
        for f in PROFILE_FIELDS:
            if got[f] is not None:
                slot[f]["extracted"] += 1
                slot[f]["correct"] += got[f] == truth.get(f)
            slot[f]["rows"] += 1
        res = flow.reason(text)
        if r["row_type"] == "named" and r["target_slug"] not in (None, "None"):
            ids = [s["id"] for s, _ in index.search(text, k=50, state=res["profile"].get("state"))]
            named["n"] += 1
            for k in (1, 5, 10):
                named[f"top{k}"] += r["target_slug"] in ids[:k]
        else:
            top = [s for s, _ in res["yes"][:3]]
            if top:
                elig["rows_with_answer"] += 1
                elig["top3_total"] += len(top)
                elig["top3_truly_eligible"] += sum(check(s["rule"], truth)[0] == "yes" for s in top)
            elig["rows"] += 1
            rel = set(lst(r["relevant_slugs"]))
            if rel:
                ranked = [s["id"] for s, _ in res["yes"][:10]]
                elig["rel_n"] += 1
                elig["rel_hit10"] += bool(rel & set(ranked))
    print(f"split={a.split}  English rows={len(rows)}\n\nSLOTS (facts stated in the query vs true_profile)")
    print(f"{'field':12} {'extracted':>10} {'precision%':>11}")
    for f in PROFILE_FIELDS:
        c = slot[f]
        print(f"{f:12} {pct(c['extracted'], c['rows']):>10} {str(pct(c['correct'], c['extracted'])):>11}")
    print(f"\nRETRIEVAL on 'named' rows (n={named['n']}): target in top-1 {pct(named['top1'], named['n'])}%  "
          f"top-5 {pct(named['top5'], named['n'])}%  top-10 {pct(named['top10'], named['n'])}%")
    print(f"\nELIGIBILITY on other rows (n={elig['rows']}): rows where the system recommends something: "
          f"{pct(elig['rows_with_answer'], elig['rows'])}%")
    print(f"  top-3 recommended that are truly eligible (true profile): {pct(elig['top3_truly_eligible'], elig['top3_total'])}%")
    print(f"  any dataset-relevant scheme in the top-10 recommended: {pct(elig['rel_hit10'], elig['rel_n'])}%")


if __name__ == "__main__":
    main()
