"""Translation evaluation: chrF++, number/entity preservation, glossary accuracy,
language-ID accuracy, and an IndicTrans2-vs-NLLB comparison CLI.

Test TSV columns: id <TAB> source <TAB> reference_en <TAB> src_lang [<TAB> glossary_terms ('|' separated)]
Usage: python -m janora.translation.evaluate test.tsv --glossary glossary.json --backends indictrans2 nllb
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from collections import Counter

from .normalise import normalise_text


def _nums(text: str) -> Counter:
    return Counter(re.findall(r"\d+(?:\.\d+)?", normalise_text(text)[0]))


def number_preservation(hyps: list[str], refs: list[str]) -> float:
    """Share of reference numbers that appear in the hypothesis (multiset recall)."""
    hit = tot = 0
    for h, r in zip(hyps, refs):
        rn = _nums(r)
        tot += sum(rn.values())
        hit += sum((rn & _nums(h)).values())
    return hit / tot if tot else 1.0


def glossary_accuracy(hyps: list[str], terms: list[list[str]]) -> float:
    hit = tot = 0
    for h, ts in zip(hyps, terms):
        for t in ts:
            tot += 1
            hit += t.lower() in h.lower()
    return hit / tot if tot else 1.0


def chrfpp(hyps: list[str], refs: list[str]) -> float:
    from sacrebleu.metrics import CHRF
    return CHRF(word_order=2).corpus_score(hyps, [refs]).score


def comet(srcs: list[str], hyps: list[str], refs: list[str], model: str = "Unbabel/wmt22-comet-da") -> float:
    from comet import download_model, load_from_checkpoint  # pip install unbabel-comet
    m = load_from_checkpoint(download_model(model))
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    return float(m.predict(data, batch_size=16, gpus=0).system_score)


def langid_accuracy(predicted: list[str], gold: list[str]) -> float:
    return sum(p == g for p, g in zip(predicted, gold)) / max(len(gold), 1)


def _load(path: str):
    with open(path, encoding="utf-8") as f:
        return [r for r in csv.reader(f, delimiter="\t") if r and not r[0].startswith("#")]


def main() -> None:
    from .config import TranslationConfig
    from .glossary import Glossary
    from .pipeline import TranslationLayer
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv")
    ap.add_argument("--glossary")
    ap.add_argument("--backends", nargs="+", default=["indictrans2", "nllb"])
    ap.add_argument("--no-translit", action="store_true", help="ablation: skip Hinglish transliteration")
    ap.add_argument("--comet", action="store_true")
    a = ap.parse_args()
    rows = _load(a.tsv)
    srcs, refs, langs = [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows]
    terms = [(r[4].split("|") if len(r) > 4 and r[4] else []) for r in rows]
    gl = Glossary.from_json(a.glossary) if a.glossary else Glossary()
    report = {}
    for b in a.backends:
        cfg = TranslationConfig(backend=b, transliterate_roman=not a.no_translit)
        layer = TranslationLayer(cfg, gl)
        res = [layer.to_english(s, forced_lang=l) for s, l in zip(srcs, langs)]
        hyps = [r.english for r in res]
        report[b] = {
            "chrF++": chrfpp(hyps, refs),
            "number_preservation": number_preservation(hyps, refs),
            "glossary_accuracy": glossary_accuracy(hyps, terms),
            "gate_failure_rate": sum(bool(r.issues) for r in res) / len(res),
            **({"COMET": comet(srcs, hyps, refs)} if a.comet else {}),
        }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
