"""Configuration for the JANORA translation layer."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class TranslationConfig:
    # --- models -----------------------------------------------------------
    backend: str = "indictrans2"          # "indictrans2" | "nllb"
    indictrans2_indic_en: str = "ai4bharat/indictrans2-indic-en-dist-200M"
    indictrans2_en_indic: str = "ai4bharat/indictrans2-en-indic-dist-200M"
    nllb_model: str = "facebook/nllb-200-distilled-600M"
    device: str | None = None             # None -> cuda if available else cpu
    num_beams: int = 5
    max_length: int = 256
    batch_size: int = 16

    # retry tiers when the MT output loses a placeholder (content dropped from long inputs)
    chunk_words: int = 12
    small_chunk_words: int = 6

    # --- input pipeline ---------------------------------------------------
    transliterate_roman: bool = True      # Hinglish -> Devanagari before MT (ablate this!)
    langid_min_conf: float = 0.60         # below this -> ask the user to confirm language

    # --- output pipeline --------------------------------------------------
    round_trip: bool = True
    round_trip_min_chrf: float = 40.0     # 0-100; CALIBRATE on your validation set
    indian_grouping: bool = True          # 300000 -> 3,00,000 for rupee amounts

    # --- placeholders -----------------------------------------------------
    # Alphanumeric on purpose: MT models tend to copy unknown Latin tokens
    # verbatim but mangle brackets. Verify empirically with the masking ablation.
    placeholder_fmt: str = "ZX{kind}{idx}X"
    placeholder_style: str = "alpha"          # input side (Indic -> English): "alpha" works (gate-fail 1.4%)
    output_placeholder_style: str = "digits"  # output side (English -> Indic): "alpha" tokens get transliterated by some models (Bengali)
