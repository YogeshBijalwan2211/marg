# JANORA translation layer

This module translates multilingual user queries into an English pivot for intent,
slot extraction, and grounded reasoning, then translates answers into the user's
language.

## Components

- `janora/translation/`: language identification, Indic transliteration, number
  normalisation, scheme-name masking, NLLB backend, and the translation pipeline.
- `data/registry/glossary.json`: canonical scheme names and aliases.
- `src/eval_translation.py`: offline checks and end-to-end translation evaluation.
- `janora/translation/evaluate.py`: chrF++, number preservation, glossary accuracy,
  and optional COMET scoring.
- `tests/test_translation.py`: regression tests that do not download the NLLB model.

## Default translation model

The only supported machine-translation backend in this branch is
`facebook/nllb-200-distilled-600M`. The model repository is public; inference
does not require a Hugging Face login. The Hugging Face model page currently
identifies the checkpoint with the FLORES-200 language set and displays 196
language tags. It is not a model for every language, and quality varies by
language pair.

NLLB uses language tags such as `eng_Latn`, `hin_Deva`, `ben_Beng`,
`tam_Taml`, and `urd_Arab`. The backend validates codes against the loaded
tokenizer instead of maintaining a separate hard-coded list.

Hinglish is not a separate NLLB language tag. Latin-script Indic input is
transliterated to native script before translation, and Latin-script Indic
output is translated into native script and transliterated back. This requires
the optional IndicXlit package and its model files. Mixed Hindi-English
sentences should still be tested with actual JANORA queries.

## Installation

Install a PyTorch build suitable for your local/Colab CPU or CUDA runtime, then:

```bash
pip install -r requirements-translation.txt
```

For Colab, choose a GPU runtime when available and check it with:

```python
import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

The model runs on CPU too, but inference is slower. The first run downloads
the model weights (about 2.5 GB for this checkpoint).

## Use

```python
from janora.translation import TranslationLayer, TranslationConfig, Glossary

glossary = Glossary.from_json("data/registry/glossary.json")
layer = TranslationLayer(TranslationConfig(), glossary)  # NLLB by default

query = layer.to_english(
    "meri income 3 lakh hai, kya mujhe pm kisan milega?"
)
if query.needs_confirmation or query.ask_language:
    # Confirm critical facts or ask the user to confirm their language.
    pass

# After the grounded English answer has been generated:
answer = layer.answer_with_disclaimer(english_answer, query.lang.code)
```

## Checks and evaluation

Run from the repository root:

```bash
python -m pytest tests
python src/eval_translation.py offline --split val
python src/eval_translation.py mt --split val --backend nllb --limit 300
python src/eval_translation.py mt --split val --backend nllb --no-translit --limit 300
```

Use the validation split to tune parameters. Run the test split only once for
the final project report. Model-based evaluation downloads NLLB weights and
may take longer on CPU.

## Deployment and safety notes

- The model card lists the checkpoint licence as CC-BY-NC 4.0. Review its
  non-commercial restriction before any commercial deployment.
- Translation must not be the sole basis for eligibility decisions. Confirm
  critical values such as age, income, and state with the user.
- The Hindi safety/disclaimer templates are human-authored. Have fluent
  speakers review them and prepare reviewed templates for other target languages.
- Add native scheme aliases to the glossary as the multilingual registry grows.
