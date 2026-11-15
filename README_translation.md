# JANORA translation layer

Pivot design: user language -> English (pipeline runs in English) -> user language.

## Interface for the rest of the pipeline
```python
from janora.translation import TranslationLayer, TranslationConfig, Glossary
layer = TranslationLayer(TranslationConfig(backend="indictrans2"),
                         Glossary.from_json("data/registry/glossary.json"))

r = layer.to_english(user_text)                      # BEFORE intent / slot extraction
r.english, r.lang.code, r.issues, r.needs_confirmation   # needs_confirmation -> echo the slot back to the user
layer.confirm_slot("income", "₹3,00,000", "hin_Deva")    # human-written confirmation prompt

out = layer.answer_with_disclaimer(english_answer, "hin_Deva")   # AFTER the grounded generator
out.text, out.show_english_alongside, out.fell_back_to_english, out.issues
layer.template("likely_eligible", "hin_Deva")                    # fixed safety phrases: never machine-translated
```
If a translation cannot be verified (a scheme name / number / URL is lost, or the model loops), the layer returns the
ENGLISH answer with `fell_back_to_english=True` instead of a corrupted translation.

## Commands (repo root)
```
python src/build_glossary.py
python -m pytest tests
python src/eval_translation.py offline --split val --heuristic-langid          # no models
python src/eval_translation.py mt --split val --backend indictrans2 --no-translit --show-failures 10
python src/eval_output_direction.py --split val --n 5 --backend indictrans2    # English -> user language
python src/eval_output_direction.py --debug ben_Beng                           # raw model output for one language
```
`transformers` must be `<5` (IndicTransToolkit breaks on 5.x). IndicTrans2 models are gated on Hugging Face: accept the
terms of both `indictrans2-indic-en-dist-200M` and `indictrans2-en-indic-dist-200M`, then `huggingface_hub.login()`.

## Results (val split)
Input side, 559 non-English queries, IndicTrans2 vs NLLB-200: see `results/translation_val_input_side.md`
(age 97.6 vs 95.8, income 94.9 vs 95.0, gate-fail 1.4% vs 3.2%).
Offline (no MT model): language ID 96.1% val / 95.4% test; age 96.8%; income 93.1%.

Output side (English -> language): Hindi, Assamese, Marathi, Nepali, Punjabi, Tamil, Telugu and Malayalam translated with
round-trip chrF 69-82. Bengali, Gujarati, Kannada, Odia and Urdu first fell back to English because the model transliterated
the Latin placeholder tokens (e.g. Bengali ZXN3X -> জেডএক্সএন3এক্স). Fix: numeric placeholder tokens (`output_placeholder_style="digits"`,
now the default); verified on Bengali (all placeholders recovered). **Not yet re-measured for Gujarati, Kannada, Odia, Urdu**
and the full output-side table has not been completed.

## Known limitations
* Output side: only the Bengali fix is verified; other languages need `eval_output_direction.py` re-run. A language that still
  fails falls back to English (no wrong answer, but not in the user's language).
* Malayalam produced a repetition loop in one answer; a repetition gate now retries then falls back to English.
* ~6.5% of queries are non-Hindi languages typed in Latin letters; the built-in language ID calls them English (needs IndicLID + IndicXlit).
* Scheme-name masking is exact-match (typo'd names in noisy queries are not masked); glossary has Hindi names only.
* Hindi strings in `templates.py` are drafts; other languages' safety templates are missing (translator needed).
* No human-translated test set yet, so chrF++/COMET are not computed; the reported numbers are proxy checks.
* Round-trip threshold (chrF 40) is a placeholder, not calibrated.
