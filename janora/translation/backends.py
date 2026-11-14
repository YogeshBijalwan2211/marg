"""MT backends with a common interface so IndicTrans2 and NLLB-200 can be compared."""
from __future__ import annotations
from abc import ABC, abstractmethod
from .config import TranslationConfig


class MTBackend(ABC):
    name = "base"

    @abstractmethod
    def translate(self, sentences: list[str], src: str, tgt: str) -> list[str]:
        """src/tgt are FLORES codes: eng_Latn, hin_Deva, tam_Taml ..."""


def _device(cfg: TranslationConfig) -> str:
    if cfg.device:
        return cfg.device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class IndicTrans2Backend(MTBackend):
    name = "indictrans2"

    def __init__(self, cfg: TranslationConfig):
        import torch  # noqa: F401
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        try:
            from IndicTransToolkit.processor import IndicProcessor
        except ImportError:  # newer toolkit versions
            from IndicTransToolkit import IndicProcessor  # type: ignore
        self.cfg, self.device = cfg, _device(cfg)
        self.ip = IndicProcessor(inference=True)
        self._models: dict[str, tuple] = {}
        self._AutoTok, self._AutoModel = AutoTokenizer, AutoModelForSeq2SeqLM

    def _load(self, name: str):
        if name not in self._models:
            tok = self._AutoTok.from_pretrained(name, trust_remote_code=True)
            model = self._AutoModel.from_pretrained(name, trust_remote_code=True).to(self.device).eval()
            self._models[name] = (tok, model)
        return self._models[name]

    def translate(self, sentences, src, tgt):
        import torch
        name = self.cfg.indictrans2_indic_en if tgt == "eng_Latn" else self.cfg.indictrans2_en_indic
        tok, model = self._load(name)
        out: list[str] = []
        for i in range(0, len(sentences), self.cfg.batch_size):
            batch = self.ip.preprocess_batch(sentences[i:i + self.cfg.batch_size], src_lang=src, tgt_lang=tgt)
            enc = tok(batch, truncation=True, padding="longest", return_tensors="pt",
                      return_attention_mask=True).to(self.device)
            with torch.inference_mode():
                gen = model.generate(**enc, use_cache=False, min_length=0, num_beams=self.cfg.num_beams,
                                     max_length=self.cfg.max_length, num_return_sequences=1)
            dec = tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            out.extend(self.ip.postprocess_batch(dec, lang=tgt))
        return out


class NLLBBackend(MTBackend):
    name = "nllb"

    def __init__(self, cfg: TranslationConfig):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.cfg, self.device = cfg, _device(cfg)
        self.tok = AutoTokenizer.from_pretrained(cfg.nllb_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.nllb_model).to(self.device).eval()

    def translate(self, sentences, src, tgt):
        import torch
        self.tok.src_lang = src
        out: list[str] = []
        for i in range(0, len(sentences), self.cfg.batch_size):
            enc = self.tok(sentences[i:i + self.cfg.batch_size], return_tensors="pt",
                           padding=True, truncation=True).to(self.device)
            with torch.inference_mode():
                gen = self.model.generate(**enc, forced_bos_token_id=self.tok.convert_tokens_to_ids(tgt),
                                          num_beams=self.cfg.num_beams, max_length=self.cfg.max_length)
            out.extend(self.tok.batch_decode(gen, skip_special_tokens=True))
        return out


def make_backend(cfg: TranslationConfig) -> MTBackend:
    return {"indictrans2": IndicTrans2Backend, "nllb": NLLBBackend}[cfg.backend](cfg)
