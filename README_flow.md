# End-to-end demo flow (baseline stand-ins + the real translation layer)

user text -> `to_english` -> slots -> retrieval -> eligibility -> English answer -> `answer_with_disclaimer` -> user's language

`janora/baseline/` holds simple BASELINES so the chain runs end to end. Replace each with the team's trained component:
| file | stand-in for | what it does |
|---|---|---|
| `slots.py` | slot extractor + Profile State Manager | regex facts from English; missing = unknown; no negation / third-party / correction handling |
| `retrieve.py` | retriever + reranker | TF-IDF over scheme records, state pre-filter |
| `rules.py` | constraint matching | three-valued rule check on each scheme's `rule` (list-valued rules supported) |
| `flow.py` | orchestration | ranks yes > unknown, asks the one missing fact most schemes need, adds safety line |

```
python src/demo_flow.py "I am a 29 year old male farmer from Gujarat, income 2.9 lakh"     # English: no model needed
python src/demo_flow.py "meri income 3 lakh hai, main Gujarat ka kisan hoon" --lang hin_Latn  # needs IndicTrans2 + HF login
python src/eval_flow.py --split val                                                         # English rows, no model
```

## Measured (English rows; val / test run once)
* Rule engine reproduces the dataset's `n_eligible_total` on 100% of rows (val 1763, train 8244): it follows the dataset's own definition.
* Slot precision when a fact is extracted: age 95.6/96.5, income 100/100, state 99.9/100, gender 100/100, residence 99.7/100, occupation 97.3/97.9.
* Named-scheme retrieval, target in top-1: 97.5 / 99.0 %.
* Top-3 recommendations that are truly eligible under the true profile: 98.4 / 99.1 %; a dataset-relevant scheme in the top-10: 88.7 / 87.8 %.

## Read these numbers carefully
* Eligibility and relevance labels come from the same `rule` fields the engine uses, so they check slot extraction and consistency,
  NOT whether the real government eligibility is right (rules can be incomplete: e.g. a pension for farmers aged 60+ is recommended to a
  29-year-old because its `rule` only says occupation=farmer; the constraint extractor must capture such conditions).
* Slot "extracted" is the share of rows where a fact was found; many facts are simply not stated in a query.
* The non-English path (`--lang`) through real MT was NOT run here: the unit test uses a fake translator. Run `demo_flow.py` on a machine with the models.
