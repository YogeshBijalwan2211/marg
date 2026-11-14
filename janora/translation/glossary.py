"""Scheme-name glossary built from the scheme registry.

Registry JSON format (list of objects):
[{"id": "S0001", "name_en": "PM-KISAN",
  "aliases": {"hin_Deva": ["पीएम किसान", "प्रधानमंत्री किसान सम्मान निधि"],
              "hin_Latn": ["pm kisan", "kisan samman nidhi"]},
  "native_name": {"hin_Deva": "पीएम-किसान"}}]
`aliases` are matched in user input; `native_name` is what is inserted in output
(falls back to the English name if absent).
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field


@dataclass
class SchemeEntry:
    id: str
    name_en: str
    aliases: dict[str, list[str]] = field(default_factory=dict)
    native_name: dict[str, str] = field(default_factory=dict)


class Glossary:
    def __init__(self, entries: list[SchemeEntry] | None = None):
        self.entries = {e.id: e for e in (entries or [])}
        self._pattern: re.Pattern | None = None
        self._lookup: dict[str, str] = {}
        self._build()

    @classmethod
    def from_json(cls, path: str) -> "Glossary":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return cls([SchemeEntry(r["id"], r["name_en"], r.get("aliases", {}), r.get("native_name", {})) for r in raw])

    def _build(self) -> None:
        self._lookup = {}
        for e in self.entries.values():
            names = [e.name_en] + [a for v in e.aliases.values() for a in v]
            names += list(e.native_name.values())
            for n in names:
                if n and n.strip():
                    self._lookup[n.strip().lower()] = e.id
        if not self._lookup:
            self._pattern = None
            return
        alts = "|".join(re.escape(n) for n in sorted(self._lookup, key=len, reverse=True))
        # lookarounds instead of \b: safe for Devanagari matras
        self._pattern = re.compile(rf"(?<![\w\u0900-\u097F])(?:{alts})(?![\w\u0900-\u097F])", re.IGNORECASE)

    @property
    def pattern(self) -> re.Pattern | None:
        return self._pattern

    def scheme_id(self, surface: str) -> str | None:
        return self._lookup.get(surface.strip().lower())

    def render(self, scheme_id: str, lang: str | None) -> str:
        e = self.entries[scheme_id]
        if lang and lang in e.native_name:
            return e.native_name[lang]
        return e.name_en
