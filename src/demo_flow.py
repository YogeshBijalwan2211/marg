"""Run the whole chain on one query:  python src/demo_flow.py "meri income 3 lakh hai ..." [--lang hin_Latn]
Needs the translation models for non-English input (IndicTrans2; HF login). English input needs no model."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from janora.baseline.flow import JanoraFlow  # noqa: E402
from janora.baseline.retrieve import SchemeIndex  # noqa: E402
from janora.translation import Glossary, TranslationConfig, TranslationLayer  # noqa: E402
from janora.translation.langid import LangID  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--lang", help="force the input language code, e.g. hin_Deva")
    ap.add_argument("--backend", default="indictrans2", choices=["indictrans2", "nllb"])
    ap.add_argument("--no-translit", action="store_true")
    a = ap.parse_args()
    layer = TranslationLayer(TranslationConfig(backend=a.backend, transliterate_roman=not a.no_translit),
                             Glossary.from_json("data/registry/glossary.json"), langid=LangID(use_indiclid=False))
    flow = JanoraFlow(layer, SchemeIndex.from_json("data/rag_schemes.json"))
    res = flow.ask(a.query, lang=a.lang)
    print("DETECTED LANGUAGE:", res["input"].lang.code, "| issues:", res["input"].issues)
    print("ENGLISH   :", res["input"].english)
    print("PROFILE   :", {k: v for k, v in res["profile"].items() if v is not None})
    print("\nENGLISH ANSWER:\n" + res["answer_en"])
    print(f"\nFINAL ANSWER ({res['output'].target_lang}, fell_back_to_english={res['output'].fell_back_to_english}):\n"
          + res["output"].text)
