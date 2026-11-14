# JANORA translation layer

This module converts multilingual user queries to an English pivot for intent,
slot extraction, and grounded reasoning, then translates answers back into the
user's language.

## Components

- `janora/translation/`: language identification, transliteration, number
  normalisation, scheme-name masking, NLLB backend, and the translation pipeline.
- `data/registry/glossary.json`: canonical scheme names and aliases.
- `src/eval_translation.py`: offline checks and end-to-end translation evaluation.
- `tests/test_translation.py`: model-free regression tests.

## Default model: NLLB-200

The default backend is `facebook/nllb-200-distilled-600M`. It is public on
Hugging Face and does not require IndicTransToolkit or Hugging Face login.
NLLB uses FLORES-200 language codes (for example, `hin_Deva`, `ben_Beng`,
`tam_Taml`, and `eng_Latn`). Latin-script Indic text is transliterated to its
native script before translation when the optional Indic transliteration
package is available.

NLLB-200 supports 200 languages, not every language. Support for a language
does not guarantee equal translation quality. Hinglish is not a separate NLLB
language; its handling relies on language detection and transliteration, so
mixed Hindi-English inputs must be evaluated on project examples.

## Install

Install a PyTorch wheel suitable for your environment first (CPU or CUDA), then:

```bash
pip install -r requirements-translation.txt
```

For Google Colab, select a GPU runtime if available. Check it with:

```python
import torch
print(torch.__version__)
print("CUDA:", torch.cuda.is_available())
```

The model can run on CPU, but inference is slower. The first run downloads the
model weights (roughly a few GB).

## Use

```python
from janora.translation import TranslationLayer, TranslationConfig, Glossary

glossary = Glossary.from_json("data/registry/glossary.json")
layer = TranslationLayer(TranslationConfig(), glossary)  # NLLB by default

query = layer.to_english(
    "meri income 3 lakh hai, kya mujhe pm kisan milega?"
)
if query.needs_confirmation or query.ask_language:
    # Ask the user to confirm critical extracted facts or their language.
    pass

# After generating a grounded English answer:
answer = layer.answer_with_disclaimer(english_answer, query.lang.code)
```

## Run checks and evaluation

Run from the repository root:

```bash
python -m pytest tests
python src/eval_translation.py offline --split val
python src/eval_translation.py mt --split val --backend nllb --limit 300
python src/eval_translation.py mt --split val --backend nllb --no-translit --limit 300
```

Use the validation split for tuning. Run the test split only once for the final
project report. Model-based evaluation requires downloading the NLLB model.

## Important deployment notes

- NLLB-200's model card specifies a CC-BY-NC 4.0 licence. Review this before
  any commercial deployment.
- Critical fields such as age, income, and state should be confirmed with the
  user; translation must not be the sole source of eligibility decisions.
- The Hindi safety/disclaimer templates are human-authored. Other-language
  safety templates should be reviewed by fluent speakers before deployment.
- Glossary aliases are currently data-dependent. Add native aliases for each
  language as the scheme registry grows.
