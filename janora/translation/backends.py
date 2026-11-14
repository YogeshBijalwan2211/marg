"""NLLB-200 machine-translation backend used by JANORA."""
from __future__ import annotations

from abc import ABC, abstractmethod

from .config import TranslationConfig


class MTBackend(ABC):
    name = "base"

    @abstractmethod
    def translate(self, sentences: list[str], src: str, tgt: str) -> list[str]:
        """Translate sentences using NLLB/FLORES codes (e.g. hin_Deva -> eng_Latn)."""


def _device(cfg: TranslationConfig) -> str:
    if cfg.device:
        return cfg.device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class NLLBBackend(MTBackend):
    """NLLB-200 sequence-to-sequence backend with tokenizer-driven code validation."""

    name = "nllb"

    def __init__(self, cfg: TranslationConfig):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.cfg = cfg
        self.device = _device(cfg)
        self.tok = AutoTokenizer.from_pretrained(cfg.nllb_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.nllb_model)
        self.model.to(self.device)
        self.model.eval()

    def _language_id(self, code: str) -> int:
        """Resolve a language tag using the loaded tokenizer, not a hand-maintained list."""
        language_ids = getattr(self.tok, "lang_code_to_id", None)
        if language_ids is not None:
            token_id = language_ids.get(code)
            if token_id is None:
                raise ValueError(
                    f"Language code {code!r} is not supported by {self.cfg.nllb_model}. "
                    "Use an NLLB/FLORES language tag (for example hin_Deva or eng_Latn)."
                )
            return token_id

        token_id = self.tok.convert_tokens_to_ids(code)
        if token_id is None or token_id == self.tok.unk_token_id:
            raise ValueError(
                f"Language code {code!r} is not recognised by the NLLB tokenizer. "
                "Use an NLLB/FLORES language tag (for example hin_Deva or eng_Latn)."
            )
        return token_id

    def translate(self, sentences: list[str], src: str, tgt: str) -> list[str]:
        import torch

        if not sentences:
            return []

        # Validate both sides before setting src_lang or starting generation.
        self._language_id(src)
        target_id = self._language_id(tgt)
        self.tok.src_lang = src

        translated: list[str] = []
        for start in range(0, len(sentences), self.cfg.batch_size):
            batch = sentences[start:start + self.cfg.batch_size]
            encoded = self.tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.cfg.max_input_length,
            ).to(self.device)
            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    forced_bos_token_id=target_id,
                    num_beams=self.cfg.num_beams,
                    max_new_tokens=self.cfg.max_new_tokens,
                )
            translated.extend(
                self.tok.batch_decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
            )
        return translated


def make_backend(cfg: TranslationConfig) -> MTBackend:
    if cfg.backend != "nllb":
        raise ValueError(
            f"Unsupported translation backend {cfg.backend!r}; JANORA is configured "
            "to use NLLB-200 ('nllb')."
        )
    return NLLBBackend(cfg)
