"""Machine-translation backends for JANORA.

NLLB-200 is the default backend. IndicTrans2 is retained only as an optional
legacy/comparison backend; it is not imported or installed for NLLB usage.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from .config import TranslationConfig

# FLORES-200 language codes supported by the NLLB-200 model used here.
# Keep this explicit: accepting arbitrary codes can silently produce invalid
# target token IDs or confusing generation errors.
NLLB_LANGUAGES = frozenset({
    "ace_Arab", "ace_Latn", "acm_Arab", "acq_Arab", "aeb_Arab", "af_Latn",
    "ajp_Arab", "aka_Latn", "amh_Ethi", "apc_Arab", "arb_Arab", "ars_Arab",
    "ary_Arab", "arz_Arab", "asm_Beng", "ast_Latn", "awa_Deva", "ayr_Latn",
    "azb_Arab", "azj_Latn", "bak_Cyrl", "bam_Latn", "ban_Latn", "bel_Cyrl",
    "bem_Latn", "ben_Beng", "bho_Deva", "bjn_Arab", "bjn_Latn", "bod_Tibt",
    "bos_Latn", "bug_Latn", "bul_Cyrl", "cat_Latn", "ceb_Latn", "ces_Latn",
    "cjk_Latn", "ckb_Arab", "crh_Latn", "cym_Latn", "dan_Latn", "deu_Latn",
    "dik_Latn", "dyu_Latn", "dzo_Tibt", "ell_Grek", "eng_Latn", "epo_Latn",
    "est_Latn", "eus_Latn", "ewe_Latn", "fao_Latn", "fij_Latn", "fin_Latn",
    "fon_Latn", "fra_Latn", "fur_Latn", "fuv_Latn", "gaz_Latn", "gla_Latn",
    "gle_Latn", "glg_Latn", "grn_Latn", "guj_Gujr", "hat_Latn", "hau_Latn",
    "heb_Hebr", "hin_Deva", "hne_Deva", "hrv_Latn", "hun_Latn", "hye_Armn",
    "ibo_Latn", "ilo_Latn", "ind_Latn", "isl_Latn", "ita_Latn", "jav_Latn",
    "jpn_Jpan", "kab_Latn", "kac_Latn", "kam_Latn", "kan_Knda", "kas_Arab",
    "kas_Deva", "kat_Geor", "kaz_Cyrl", "kbp_Latn", "kea_Latn", "khk_Cyrl",
    "khm_Khmr", "kik_Latn", "kin_Latn", "kir_Cyrl", "kmb_Latn", "kmr_Latn",
    "knc_Arab", "knc_Latn", "kon_Latn", "kor_Hang", "lao_Laoo", "lij_Latn",
    "lim_Latn", "lin_Latn", "lit_Latn", "lmo_Latn", "ltg_Latn", "ltz_Latn",
    "lua_Latn", "lug_Latn", "luo_Latn", "lus_Latn", "lvs_Latn", "mag_Deva",
    "mai_Deva", "mal_Mlym", "mar_Deva", "min_Latn", "mkd_Cyrl", "plt_Latn",
    "mlt_Latn", "mni_Beng", "mos_Latn", "mri_Latn", "mya_Mymr", "nld_Latn",
    "nno_Latn", "nob_Latn", "npi_Deva", "nso_Latn", "nus_Latn", "nya_Latn",
    "oci_Latn", "ory_Orya", "pag_Latn", "pan_Guru", "pap_Latn", "pbt_Arab",
    "pes_Arab", "pol_Latn", "por_Latn", "prs_Arab", "quy_Latn", "ron_Latn",
    "run_Latn", "rus_Cyrl", "sag_Latn", "san_Deva", "sat_Beng", "scn_Latn",
    "shn_Mymr", "sin_Sinh", "slk_Latn", "slv_Latn", "smo_Latn", "sna_Latn",
    "snd_Arab", "som_Latn", "sot_Latn", "spa_Latn", "als_Latn", "srd_Latn",
    "srp_Cyrl", "ssw_Latn", "sun_Latn", "swe_Latn", "swh_Latn", "szl_Latn",
    "tam_Taml", "taq_Latn", "taq_Tfng", "tat_Cyrl", "tel_Telu", "tgk_Cyrl",
    "tgl_Latn", "tha_Thai", "tir_Ethi", "tpi_Latn", "tsn_Latn", "tso_Latn",
    "tuk_Latn", "tum_Latn", "tur_Latn", "twi_Latn", "tzm_Tfng", "uig_Arab",
    "ukr_Cyrl", "umb_Latn", "urd_Arab", "uzn_Latn", "vec_Latn", "vie_Latn",
    "war_Latn", "wol_Latn", "xho_Latn", "ydd_Hebr", "yor_Latn", "yue_Hant",
    "zho_Hans", "zho_Hant", "zsm_Latn", "zul_Latn",
})


class MTBackend(ABC):
    name = "base"

    @abstractmethod
    def translate(self, sentences: list[str], src: str, tgt: str) -> list[str]:
        """Translate sentences using FLORES codes (e.g. hin_Deva -> eng_Latn)."""


def _device(cfg: TranslationConfig) -> str:
    if cfg.device:
        return cfg.device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class NLLBBackend(MTBackend):
    """NLLB-200 sequence-to-sequence backend with explicit language control."""

    name = "nllb"

    def __init__(self, cfg: TranslationConfig):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.cfg = cfg
        self.device = _device(cfg)
        self.tok = AutoTokenizer.from_pretrained(cfg.nllb_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.nllb_model)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _validate_language(code: str) -> None:
        if code not in NLLB_LANGUAGES:
            raise ValueError(
                f"Language code {code!r} is not supported by the configured "
                "NLLB-200 backend. Use a FLORES-200 code such as hin_Deva "
                "or eng_Latn. Romanised Indic input must be transliterated "
                "to its native script before NLLB translation."
            )

    def translate(self, sentences: list[str], src: str, tgt: str) -> list[str]:
        import torch

        if not sentences:
            return []
        self._validate_language(src)
        self._validate_language(tgt)

        # NLLB's tokenizer uses the source language token at the start and
        # forced_bos_token_id to select the target language.
        self.tok.src_lang = src
        target_id = self.tok.convert_tokens_to_ids(tgt)
        if target_id is None or target_id == self.tok.unk_token_id:
            raise ValueError(f"Tokenizer does not recognise NLLB target code {tgt!r}")

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


class IndicTrans2Backend(MTBackend):
    """Optional legacy IndicTrans2 backend; does not load unless explicitly selected."""

    name = "indictrans2"

    def __init__(self, cfg: TranslationConfig):
        import torch  # noqa: F401
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        try:
            from IndicTransToolkit.processor import IndicProcessor
        except ImportError:
            from IndicTransToolkit import IndicProcessor  # type: ignore

        self.cfg, self.device = cfg, _device(cfg)
        self.ip = IndicProcessor(inference=True)
        self._models: dict[str, tuple] = {}
        self._AutoTok, self._AutoModel = AutoTokenizer, AutoModelForSeq2SeqLM

    def _load(self, name: str):
        if name not in self._models:
            tok = self._AutoTok.from_pretrained(name, trust_remote_code=True)
            model = self._AutoModel.from_pretrained(name, trust_remote_code=True)
            self._models[name] = (tok, model.to(self.device).eval())
        return self._models[name]

    def translate(self, sentences: list[str], src: str, tgt: str) -> list[str]:
        import torch
        name = self.cfg.indictrans2_indic_en if tgt == "eng_Latn" else self.cfg.indictrans2_en_indic
        tok, model = self._load(name)
        out: list[str] = []
        for i in range(0, len(sentences), self.cfg.batch_size):
            batch = self.ip.preprocess_batch(
                sentences[i:i + self.cfg.batch_size], src_lang=src, tgt_lang=tgt
            )
            enc = tok(
                batch, truncation=True, padding="longest", return_tensors="pt",
                return_attention_mask=True,
            ).to(self.device)
            with torch.inference_mode():
                gen = model.generate(
                    **enc, use_cache=False, min_length=0, num_beams=self.cfg.num_beams,
                    max_length=self.cfg.max_length, num_return_sequences=1,
                )
            dec = tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            out.extend(self.ip.postprocess_batch(dec, lang=tgt))
        return out


def make_backend(cfg: TranslationConfig) -> MTBackend:
    backends = {"nllb": NLLBBackend, "indictrans2": IndicTrans2Backend}
    try:
        backend_cls = backends[cfg.backend]
    except KeyError as exc:
        raise ValueError(
            f"Unknown translation backend {cfg.backend!r}. "
            f"Choose one of: {', '.join(sorted(backends))}."
        ) from exc
    return backend_cls(cfg)
