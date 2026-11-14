# JANORA translation layer (marg repo)

Layout added to the repo root:

```
janora/translation/        the package (normalise, masking, langid, translit, backends, templates, pipeline, evaluate)
src/build_glossary.py      data/en + data/hi + registry  ->  data/registry/glossary.json
src/eval_translation.py    offline checks and end-to-end MT evaluation on data/splits/*.jsonl
tests/test_translation.py  31 tests (fake MT backend, no GPU needed)
requirements-translation.txt
```

## Use
```python
from janora.translation import TranslationLayer, TranslationConfig, Glossary
layer = TranslationLayer(TranslationConfig(backend="indictrans2"), Glossary.from_json("data/registry/glossary.json"))

r = layer.to_english("meri income 3 lakh hai, kya mujhe pm kisan milega?")   # before intent / slot extraction
out = layer.answer_with_disclaimer(english_answer, "hin_Deva")               # after the grounded generator
```

## Commands (from the repo root)
```
pip install -r requirements-translation.txt
python src/build_glossary.py                         # rerun whenever the scheme list changes
python -m pytest tests
python src/eval_translation.py offline --split val   # no models needed
python src/eval_translation.py mt --split val --backend indictrans2 --limit 300
python src/eval_translation.py mt --split val --backend nllb --limit 300
python src/eval_translation.py mt --split val --backend indictrans2 --no-translit   # ablation
```
Tune on `val`; run `--split test` once at the end (project rule).

## Measured so far (offline, heuristic language ID, no MT model)
| | val | test |
|---|---|---|
| language ID (script-aware) | 96.1% | 95.4% |
| age recovered after normalisation | 96.8% | 95.1% |
| income recovered (rows that state a figure) | 93.1% | 94.1% |

English rows score the same as the other 13 languages, so the number normaliser is not the weak point.
Not measured yet: anything that needs the MT models (translation quality, round-trip gate, transliteration).

## Known gaps
* ~6.5% of queries are non-Hindi languages typed in Latin letters ("aamaar bayas 52"). The built-in heuristic calls
  them English; IndicLID is needed (and IndicXlit then transliterates them).
* Scheme-name masking is exact-match. Typo'd/truncated names in noisy queries are not masked (about 25% of "named" rows).
* The glossary has Hindi names only; other languages need names scraped, otherwise they fall back to English names.
* `templates.py` Hindi strings are drafts for your human translator to review; other languages are missing.
* IndicLID / IndicTransToolkit / IndicXlit wrappers were not run (no model downloads in my sandbox).
