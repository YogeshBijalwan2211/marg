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

LOW_LANG_CONF = "low_language_confidence"
PLACEHOLDER_MISMATCH = "placeholder_mismatch"
NUMBER_CHANGED = "number_changed"
LOW_ROUND_TRIP = "low_round_trip_similarity"
TEMPLATE_FALLBACK = "template_missing_for_language"
XLIT_UNAVAILABLE = "roman_transliteration_unavailable"

_SENT = re.compile(r"(?<=[.!?।])\s+")


def split_sentences(text: str) -> list[str]:
    return [sentence for sentence in _SENT.split(text.strip()) if sentence]


def chrf_similarity(hyp: str, ref: str) -> float:
    """Return 0-100 chrF; use a character-similarity fallback when sacrebleu is absent."""
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
    needs_confirmation: bool = False
    ask_language: bool = False


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
    def __init__(
        self,
        cfg: TranslationConfig | None = None,
        glossary: Glossary | None = None,
        backend: MTBackend | None = None,
        langid: LangID | None = None,
        transliterator: Transliterator | None = None,
    ):
        self.cfg = cfg or TranslationConfig()
        self.glossary = glossary or Glossary()
        self.masker = Masker(
            self.glossary, self.cfg.placeholder_fmt, self.cfg.indian_grouping
        )
        self._backend = backend
        self.langid = langid or LangID()
        self.translit = transliterator or Transliterator()

    @property
    def backend(self) -> MTBackend:
        """Lazily load the NLLB model only when translation is requested."""
        if self._backend is None:
            self._backend = make_backend(self.cfg)
        return self._backend

    def _mt(
        self, masked: Masked, text: str, src: str, tgt: str
    ) -> tuple[str, list[str]]:
        """Translate masked text, retry once unsplit, and verify placeholders/digits."""
        issues: list[str] = []
        attempts = [
            lambda: " ".join(
                self.backend.translate(split_sentences(text) or [text], src, tgt)
            ),
            lambda: self.backend.translate([text], src, tgt)[0],
        ]
        output = ""
        for attempt in attempts:
            output = attempt()
            valid, _, _ = self.masker.verify(masked, output)
            if valid:
                break
        else:
            issues.append(PLACEHOLDER_MISMATCH)
        if self.masker.stray_digits(masked, output):
            issues.append(NUMBER_CHANGED)
        return output, issues

    @staticmethod
    def _model_language(language: str) -> str:
        """Map roman-script Indic tags to their NLLB native-script language tags."""
        if language.endswith("_Latn") and not language.startswith("eng"):
            return native_code(language)
        return language

    def to_english(self, text: str, forced_lang: str | None = None) -> InputResult:
        lang = (
            LangResult(forced_lang, 1.0, "forced")
            if forced_lang else self.langid.detect(text)
        )
        issues: list[str] = []
        ask_language = lang.confidence < self.cfg.langid_min_conf
        if ask_language:
            issues.append(LOW_LANG_CONF)

        if lang.code == ENGLISH:
            normalised, _ = normalise_text(text)
            return InputResult(
                text, lang, normalised, normalised, issues, False, ask_language
            )

        masked = self.masker.mask(
            clean_text(text),
            normaliser=lambda value: normalise_numbers(value)[0],
        )
        normalised = self.masker.unmask(masked.text, masked)
        source_text = masked.text
        model_source = lang.code

        if lang.is_roman_indic:
            model_source = self._model_language(lang.code)
            if self.cfg.transliterate_roman:
                try:
                    source_text = self.translit.transliterate(source_text, lang.code)
                except Exception:
                    # Do not pass Latin characters to NLLB while falsely labelling
                    # them as a native Indic script. Ask the caller to confirm/retry.
                    issues.append(XLIT_UNAVAILABLE)
                    return InputResult(
                        text, lang, normalised, normalised, issues, True, ask_language
                    )

        translated, mt_issues = self._mt(
            masked, source_text, model_source, ENGLISH
        )
        issues.extend(mt_issues)
        english = self.masker.unmask(translated, masked)
        needs_confirmation = bool(
            set(issues) & {PLACEHOLDER_MISMATCH, NUMBER_CHANGED, XLIT_UNAVAILABLE}
        )
        return InputResult(
            text, lang, normalised, english, issues, needs_confirmation, ask_language
        )

    def confirm_slot(self, slot: str, value: str, lang: str) -> str:
        model_lang = self._model_language(lang)
        text, fallback = render(
            "confirm_slot", model_lang,
            slot=slot_label(slot, model_lang), value=value,
        )
        if model_lang != lang:
            try:
                return self.translit.transliterate_to_roman(text, lang)
            except Exception:
                text, _ = render(
                    "confirm_slot", ENGLISH,
                    slot=slot, value=value,
                )
        return text

    def template(self, key: str, lang: str, **variables) -> str:
        model_lang = self._model_language(lang)
        text, _ = render(key, model_lang, **variables)
        if model_lang != lang:
            try:
                return self.translit.transliterate_to_roman(text, lang)
            except Exception:
                text, _ = render(key, ENGLISH, **variables)
        return text

    def from_english(self, answer_en: str, target_lang: str) -> OutputResult:
        if target_lang == ENGLISH:
            return OutputResult(answer_en, answer_en, target_lang)

        model_target = self._model_language(target_lang)
        masked = self.masker.mask(answer_en)
        translated, issues = self._mt(
            masked, masked.text, ENGLISH, model_target
        )
        if PLACEHOLDER_MISMATCH in issues:
            return OutputResult(
                answer_en, answer_en, target_lang, issues,
                show_english_alongside=True, fell_back_to_english=True,
            )

        score, show_english = None, False
        if self.cfg.round_trip:
            back = " ".join(
                self.backend.translate(
                    split_sentences(translated) or [translated],
                    model_target, ENGLISH,
                )
            )
            score = chrf_similarity(back, masked.text)
            if score < self.cfg.round_trip_min_chrf:
                issues.append(LOW_ROUND_TRIP)
                show_english = True
        if NUMBER_CHANGED in issues:
            show_english = True

        if model_target != target_lang:
            try:
                translated = self.translit.transliterate_to_roman(
                    translated, target_lang
                )
            except Exception:
                issues.append(XLIT_UNAVAILABLE)
                return OutputResult(
                    answer_en, answer_en, target_lang, issues,
                    show_english_alongside=True, fell_back_to_english=True,
                )

        final_text = self.masker.unmask(
            translated, masked, scheme_lang=target_lang,
            group_money=self.cfg.indian_grouping,
        )
        return OutputResult(
            final_text, answer_en, target_lang, issues, score, show_english
        )

    def answer_with_disclaimer(self, body_en: str, target_lang: str) -> OutputResult:
        """Machine-translated body + a fixed, human-authored safety line."""
        result = self.from_english(body_en, target_lang)
        model_lang = self._model_language(target_lang)
        disclaimer, fallback = render("verify_disclaimer", model_lang)
        if model_lang != target_lang:
            try:
                disclaimer = self.translit.transliterate_to_roman(
                    disclaimer, target_lang
                )
            except Exception:
                disclaimer, _ = render("verify_disclaimer", ENGLISH)
                fallback = True
        result.text = f"{result.text}\n\n{disclaimer}"
        if fallback:
            result.issues.append(TEMPLATE_FALLBACK)
        if result.show_english_alongside and not result.fell_back_to_english:
            note, note_fallback = render("translation_uncertain", model_lang)
            if model_lang != target_lang:
                try:
                    note = self.translit.transliterate_to_roman(note, target_lang)
                except Exception:
                    note, _ = render("translation_uncertain", ENGLISH)
                    note_fallback = True
            result.text += f"\n\n{note}\n{result.english}"
            if note_fallback:
                result.issues.append(TEMPLATE_FALLBACK)
        return result
