"""Build the scheme-name glossary used by the translation layer's masking step.

Run from the repo root:   python src/build_glossary.py
Reads  : data/en/schemes_list.jsonl, data/hi/schemes_list.jsonl, data/registry/scheme_registry.csv
Writes : data/registry/glossary.json

One entry per slug. English name -> matched in any input; Hindi name -> matched in Hindi input AND used
when the answer is rendered in Hindi. (Other languages have no names in this repo yet: add them when scraped.)
The glossary covers ALL schemes (names are not labels), so building it does not touch the frozen split.
"""
import csv
import json
from pathlib import Path

DATA = Path("data")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    reg = {}
    with open(DATA / "registry" / "scheme_registry.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            reg[r["slug"]] = r["name"].strip()
    hi = {x["slug"]: (x.get("schemeName") or "").strip() for x in read_jsonl(DATA / "hi" / "schemes_list.jsonl")}
    out, seen_en = [], {}
    for slug, name_en in reg.items():
        if not name_en:
            continue
        key = name_en.lower()
        if key in seen_en:          # two slugs share one English name: keep the first, report below
            continue
        seen_en[key] = slug
        entry = {"id": slug, "name_en": name_en, "aliases": {}, "native_name": {}}
        if hi.get(slug):
            entry["aliases"]["hin_Deva"] = [hi[slug]]
            entry["native_name"]["hin_Deva"] = hi[slug]
        out.append(entry)
    path = DATA / "registry" / "glossary.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(out)} schemes -> {path}  ({sum('hin_Deva' in e['aliases'] for e in out)} with Hindi names, "
          f"{len(reg) - len(out)} skipped as empty/duplicate names)")


if __name__ == "__main__":
    main()
