"""CPU numerical reference for Rubin-style LUT-W3A8 and LUT-W3A16 models."""

from .activation import fake_quantize_activation_k64
from .e4m3 import quantize_e4m3
from .lloyd_max import fit_lut_tiles
from .linear import CachedLUTLinear, reference_linear_lookup
from .qwen import convert_qwen_to_w3a16, convert_qwen_to_w3a8
from .weight import LUTWeight, quantize_weight, reconstruct_weight

__all__ = [
    "CachedLUTLinear",
    "LUTWeight",
    "convert_qwen_to_w3a8",
    "convert_qwen_to_w3a16",
    "fake_quantize_activation_k64",
    "fit_lut_tiles",
    "quantize_e4m3",
    "quantize_weight",
    "reconstruct_weight",
    "reference_linear_lookup",
]
