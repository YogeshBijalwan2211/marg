"""JANORA translation layer: user language <-> English pivot, with quality gates."""
from __future__ import annotations
import re
from dataclasses import dataclass, field

from .backends import MTBackend, make_backend
from .config import TranslationConfig
from .glossary import Glossary
from .langid import LangID, LangResult, native_code
from .masking import Masked, Masker
from .normalise import clean_text, normalise_numbers, normalise_text
from .templates import render, slot_label
from .translit import Transliterator

ENGLISH = "eng_Latn"

# issue codes (quality gates)
LOW_LANG_CONF = "low_language_confidence"
PLACEHOLDER_MISMATCH = "placeholder_mismatch"
NUMBER_CHANGED = "number_changed"
LOW_ROUND_TRIP = "low_round_trip_similarity"
TEMPLATE_FALLBACK = "template_missing_for_language"

_SENT = re.compile(r"(?<=[.!?।])\s+")


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENT.split(text.strip()) if s]


def chrf_similarity(hyp: str, ref: str) -> float:
    """0-100. chrF if sacrebleu is installed, else difflib ratio."""
    try:
        from sacrebleu.metrics import CHRF
        return CHRF().sentence_score(hyp, [ref]).score
    except Exception:
        import difflib
        return 100 * difflib.SequenceMatcher(None, hyp.lower(), ref.lower()).ratio()


@dataclass
class InputResult:
    original: str
    lang: LangResult
    normalised: str
    english: str
    issues: list[str] = field(default_factory=list)
    needs_confirmation: bool = False   # caller should echo-confirm before saving slots
    ask_language: bool = False         # caller should show templates 'confirm_language'


@dataclass
class OutputResult:
    text: str
    english: str
    target_lang: str
    issues: list[str] = field(default_factory=list)
    round_trip_score: float | None = None
    show_english_alongside: bool = False
    fell_back_to_english: bool = False


class TranslationLayer:
    def __init__(self, cfg: TranslationConfig | None = None, glossary: Glossary | None = None,
                 backend: MTBackend | None = None, langid: LangID | None = None,
                 transliterator: Transliterator | None = None):
        self.cfg = cfg or TranslationConfig()
        self.glossary = glossary or Glossary()
        self.masker = Masker(self.glossary, self.cfg.placeholder_fmt, self.cfg.indian_grouping)
        self._backend = backend
        self.langid = langid or LangID()
        self.translit = transliterator or Transliterator()

    @property
    def backend(self) -> MTBackend:           # lazy: models load on first use
        if self._backend is None:
            self._backend = make_backend(self.cfg)
        return self._backend

    # ------------------------------------------------------------------ core
    def _mt(self, masked: Masked, text: str, src: str, tgt: str) -> tuple[str, list[str]]:
        """Translate masked text; verify placeholders; retry once unsplit."""
        issues: list[str] = []
        attempts = [lambda: " ".join(self.backend.translate(split_sentences(text) or [text], src, tgt)),
                    lambda: self.backend.translate([text], src, tgt)[0]]
        out = ""
        for attempt in attempts:
            out = attempt()
            ok, _, _ = self.masker.verify(masked, out)
            if ok:
                break
        else:
            issues.append(PLACEHOLDER_MISMATCH)
        if self.masker.stray_digits(masked, out):
            issues.append(NUMBER_CHANGED)
        return out, issues

    # ----------------------------------------------------------- input side
    def to_english(self, text: str, forced_lang: str | None = None) -> InputResult:
        lang = LangResult(forced_lang, 1.0, "forced") if forced_lang else self.langid.detect(text)
        issues: list[str] = []
        ask_language = lang.confidence < self.cfg.langid_min_conf
        if ask_language:
            issues.append(LOW_LANG_CONF)

        if lang.code == ENGLISH:
            normalised, _ = normalise_text(text)
            return InputResult(text, lang, normalised, normalised, issues, False, ask_language)

        masked = self.masker.mask(clean_text(text), normaliser=lambda t: normalise_numbers(t)[0])
        normalised = self.masker.unmask(masked.text, masked)      # cleaned + numbers normalised
        src_text, mt_src = masked.text, lang.code
        if lang.is_roman_indic:
            mt_src = native_code(lang.code)
            if self.cfg.transliterate_roman:
                src_text = self.translit.transliterate(src_text, lang.code)

        translated, mt_issues = self._mt(masked, src_text, mt_src, ENGLISH)
        issues += mt_issues
        english = self.masker.unmask(translated, masked)
        needs_conf = bool(set(issues) & {PLACEHOLDER_MISMATCH, NUMBER_CHANGED})
        return InputResult(text, lang, normalised, english, issues, needs_conf, ask_language)

    def confirm_slot(self, slot: str, value: str, lang: str) -> str:
        """Echo-confirmation prompt in the user's language (critical slots: income, age, state, category)."""
        text, _ = render("confirm_slot", lang, slot=slot_label(slot, lang), value=value)
        return text

    # ---------------------------------------------------------- output side
    def template(self, key: str, lang: str, **vars) -> str:
        return render(key, lang, **vars)[0]

    def from_english(self, answer_en: str, target_lang: str) -> OutputResult:
        if target_lang == ENGLISH:
            return OutputResult(answer_en, answer_en, target_lang)
        masked = self.masker.mask(answer_en)
        translated, issues = self._mt(masked, masked.text, ENGLISH, target_lang)
        if PLACEHOLDER_MISMATCH in issues:           # gate: show original English
            return OutputResult(answer_en, answer_en, target_lang, issues,
                                show_english_alongside=True, fell_back_to_english=True)
        score, show_en = None, False
        if self.cfg.round_trip:
            back = " ".join(self.backend.translate(split_sentences(translated) or [translated],
                                                   target_lang, ENGLISH))
            score = chrf_similarity(back, masked.text)
            if score < self.cfg.round_trip_min_chrf:
                issues.append(LOW_ROUND_TRIP)
                show_en = True
        if NUMBER_CHANGED in issues:
            show_en = True
        final = self.masker.unmask(translated, masked, scheme_lang=target_lang,
                                   group_money=self.cfg.indian_grouping)
        return OutputResult(final, answer_en, target_lang, issues, score, show_en)

    def answer_with_disclaimer(self, body_en: str, target_lang: str) -> OutputResult:
        """Machine-translated body + human-translated safety line."""
        res = self.from_english(body_en, target_lang)
        text, fb = render("verify_disclaimer", target_lang)
        res.text = f"{res.text}\n\n{text}"
        if fb:
            res.issues.append(TEMPLATE_FALLBACK)
        if res.show_english_alongside and not res.fell_back_to_english:
            note, _ = render("translation_uncertain", target_lang)
            res.text += f"\n\n{note}\n{res.english}"
        return res
