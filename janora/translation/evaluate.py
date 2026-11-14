"""Translation evaluation: chrF++, number/entity preservation, glossary accuracy,
and language identification.

Test TSV columns: id<TAB>source<TAB>reference_en<TAB>src_lang[<TAB>glossary_terms]
Usage: python -m janora.translation.evaluate test.tsv --glossary glossary.json
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


def number_preservation(hypotheses: list[str], references: list[str]) -> float:
    """Share of reference numbers that appear in the hypothesis (multiset recall)."""
    hits = total = 0
    for hypothesis, reference in zip(hypotheses, references):
        reference_numbers = _nums(reference)
        total += sum(reference_numbers.values())
        hits += sum((reference_numbers & _nums(hypothesis)).values())
    return hits / total if total else 1.0


def glossary_accuracy(hypotheses: list[str], terms: list[list[str]]) -> float:
    hits = total = 0
    for hypothesis, row_terms in zip(hypotheses, terms):
        for term in row_terms:
            total += 1
            hits += term.lower() in hypothesis.lower()
    return hits / total if total else 1.0


def chrfpp(hypotheses: list[str], references: list[str]) -> float:
    from sacrebleu.metrics import CHRF
    return CHRF(word_order=2).corpus_score(hypotheses, [references]).score


def comet(
    sources: list[str], hypotheses: list[str], references: list[str],
    model: str = "Unbabel/wmt22-comet-da",
) -> float:
    from comet import download_model, load_from_checkpoint
    metric = load_from_checkpoint(download_model(model))
    data = [
        {"src": source, "mt": hypothesis, "ref": reference}
        for source, hypothesis, reference in zip(sources, hypotheses, references)
    ]
    return float(metric.predict(data, batch_size=16, gpus=0).system_score)


def langid_accuracy(predicted: list[str], gold: list[str]) -> float:
    return sum(p == g for p, g in zip(predicted, gold)) / max(len(gold), 1)


def _load(path: str):
    with open(path, encoding="utf-8") as file:
        return [
            row for row in csv.reader(file, delimiter="\t")
            if row and not row[0].startswith("#")
        ]


def main() -> None:
    from .config import TranslationConfig
    from .glossary import Glossary
    from .pipeline import TranslationLayer

    parser = argparse.ArgumentParser()
    parser.add_argument("tsv")
    parser.add_argument("--glossary")
    parser.add_argument("--no-translit", action="store_true")
    parser.add_argument("--comet", action="store_true")
    args = parser.parse_args()

    rows = _load(args.tsv)
    sources = [row[1] for row in rows]
    references = [row[2] for row in rows]
    languages = [row[3] for row in rows]
    terms = [
        row[4].split("|") if len(row) > 4 and row[4] else []
        for row in rows
    ]
    glossary = Glossary.from_json(args.glossary) if args.glossary else Glossary()
    config = TranslationConfig(transliterate_roman=not args.no_translit)
    layer = TranslationLayer(config, glossary)

    results = [layer.to_english(source, forced_lang=lang)
               for source, lang in zip(sources, languages)]
    hypotheses = [result.english for result in results]
    report = {
        "backend": config.backend,
        "chrF++": chrfpp(hypotheses, references),
        "number_preservation": number_preservation(hypotheses, references),
        "glossary_accuracy": glossary_accuracy(hypotheses, terms),
        "gate_failure_rate": sum(bool(result.issues) for result in results) / max(len(results), 1),
    }
    if args.comet:
        report["COMET"] = comet(sources, hypotheses, references)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
