"""Configuration for the JANORA translation layer."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TranslationConfig:
    # NLLB-200 is the supported default. IndicTrans2 is an optional legacy backend.
    backend: str = "nllb"
    nllb_model: str = "facebook/nllb-200-distilled-600M"

    # Optional IndicTrans2 model IDs, used only with backend="indictrans2".
    indictrans2_indic_en: str = "ai4bharat/indictrans2-indic-en-dist-200M"
    indictrans2_en_indic: str = "ai4bharat/indictrans2-en-indic-dist-200M"

    device: str | None = None  # None -> CUDA when available, otherwise CPU
    num_beams: int = 4
    batch_size: int = 8
    max_input_length: int = 512
    max_new_tokens: int = 256
    # Kept for the legacy IndicTrans2 backend.
    max_length: int = 256

    # Input pipeline
    transliterate_roman: bool = True
    langid_min_conf: float = 0.60

    # Output pipeline
    round_trip: bool = True
    round_trip_min_chrf: float = 40.0  # Calibrate on the validation split.
    indian_grouping: bool = True

    # Alphanumeric placeholders are intentionally easy for MT models to copy.
    placeholder_fmt: str = "ZX{kind}{idx}X"
