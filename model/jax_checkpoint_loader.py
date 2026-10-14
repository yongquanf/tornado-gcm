# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Load JAX NeuralGCM checkpoint into a PyTorch model.

Provides:
  - build_jax_neural_components(): construct PyTorch NN modules matching
    the JAX deterministic_2_8_deg checkpoint architecture
  - load_jax_checkpoint(): load weights from JAX pickle checkpoint
  - build_jax_aligned_model(): build full NeuralGCMModel with JAX architecture

Architecture (from JAX param tree — 229 learnable params):
  DECODER: EPD(784→384→259) + positional(8,128,64)
  ENCODER[0]: EPD(1052→384→193) + positional(8,128,64) + orography(4223)
  ENCODER[1]: same as ENCODER[0]
  PHYSICS: EPD(2365→384→192) + 3 surface embeddings
    + VerticalConvTower(9→64→64→64→64→32) + positional(8,128,64)
    + corrector orography(4094)

Weight mapping:
  JAX Haiku Linear: w shape (in, out), b shape (out,)
  PyTorch nn.Linear: weight shape (out, in), bias shape (out,)
  → PT weight = JAX w.T;  PT bias = JAX b

  JAX Conv1d: w shape (kernel, in_ch, out_ch), b shape (out_ch, 1)
  PyTorch Conv1d: weight shape (out_ch, in_ch, kernel), bias shape (out_ch,)
  → PT weight = JAX w.permute(2,1,0);  PT bias = JAX b.squeeze()
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tornado_gcm.neural.standard_layers import Mlp, Epd, ConvLevel
from tornado_gcm.neural.towers import ForwardTower, VerticalConvTower
from tornado_gcm import scales

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Constants from JAX Gin config (deterministic_2_8_deg)
# ═══════════════════════════════════════════════════════════════════════════

LATENT_SIZE = 384
LAYER_SIZE = 384
NUM_BLOCKS = 5
N_SIGMA_LAYERS = 32
N_CNN_FEATURES = 32
SURFACE_MODEL_LATENT_SIZE = 8
SURFACE_MODEL_LAYER_SIZE = 8
SURFACE_MODEL_OUTPUT_SIZE = 8
POSITIONAL_LATENT_SIZE = 8

# ── Resolution → grid / architecture specs ───────────────────────────────
# Each JAX checkpoint uses a specific spectral grid with corresponding
# positional embedding dimensions.
_JAX_RESOLUTION_SPECS: dict[str, dict] = {
    "2.8": {
        "grid_name": "TL63",
        "nodal_lon": 128,
        "nodal_lat": 64,
        "max_wavenumber": 63,
        "positional_channels": 8,
        "gin_stem": "deterministic_2_8_deg",
        # Architecture dimensions matching JAX checkpoint tensor shapes
        "physics_epd_input": 2365,
        "encoder_epd_input": 1052,
        "decoder_epd_input": 784,
        "corrector_orog_size": 4094,
        "encoder_orog_size": 4223,
        "surface_embed_inputs": [4, 1, 5],  # land, sea, ice
    },
    "1.4": {
        "grid_name": "TL127",
        "nodal_lon": 256,
        "nodal_lat": 128,
        "max_wavenumber": 127,
        "positional_channels": 32,
        "gin_stem": "deterministic_1_4_deg",
        "physics_epd_input": 2389,
        "encoder_epd_input": 1076,
        "decoder_epd_input": 297,
        "corrector_orog_size": 16127,
        "encoder_orog_size": 16639,
        "surface_embed_inputs": [1],  # sea only (SST); land/ice use bias params
        # JAX NodalLandSeaIceEmbedding "bias-only" mode: land / sea_ice
        # contributions are per-surface constant parameters (8,1,1) broadcast
        # across lon/lat, not learned Epd towers.
        "land_ice_biases": True,
        # Decoder has its own learned_orography parameter here too.
        "decoder_orog_size": 16639,
    },
    "0.7": {
        "grid_name": "TL255",
        "nodal_lon": 512,
        "nodal_lat": 256,
        "max_wavenumber": 255,
        "positional_channels": 32,
        "gin_stem": "deterministic_0_7_deg",
        "physics_epd_input": 2389,
        "encoder_epd_input": 1076,
        "decoder_epd_input": 297,
        "corrector_orog_size": 65023,
        "encoder_orog_size": 66047,
        "surface_embed_inputs": [1],  # sea only (SST); land/ice use bias params
        "land_ice_biases": True,
        "decoder_orog_size": 66047,
    },
}

# EPD dimensions (from JAX param shapes)
PHYSICS_EPD_INPUT = 2365
PHYSICS_EPD_OUTPUT = 192
DECODER_EPD_INPUT = 784
DECODER_EPD_OUTPUT = 259
ENCODER_EPD_INPUT = 1052
ENCODER_EPD_OUTPUT = 193

# Surface embedding input sizes (land=4, sea=1, ice=5)
SURFACE_EMBED_INPUTS = [4, 1, 5]

# Volume conv tower architecture
VOLUME_CONV_INPUT_CH = 9
VOLUME_CONV_HIDDEN_CH = [64, 64, 64, 64]
VOLUME_CONV_OUTPUT_CH = 32
VOLUME_CONV_KERNEL = 5

# Reference temperatures from Gin config
REFERENCE_TEMPERATURES = np.array([
    223.58614815, 211.47405876, 205.87815406, 206.40755302, 210.43452345,
    214.5683887, 218.75303863, 223.23145107, 227.9710687, 232.85381503,
    237.53588735, 242.05068293, 246.29986585, 250.14294113, 253.74839535,
    256.98024283, 259.94441031, 262.7041158, 265.21752838, 267.62333985,
    269.94462121, 272.10056439, 274.12518288, 275.99833711, 277.72759392,
    279.3292128, 280.79178708, 282.13507065, 283.41832023, 284.7682506,
    286.33945487, 288.06707666,
])

# ── Normalization constants from Gin config ──────────────────────────────
# advance/ and embedding_model/ ShiftAndNormalize share identical values.
_NORM_SCALES_NON_CNN = {
    'cos_latitude': 0.3036,
    'divergence': 0.05232, 'divergence_del2': 108.7,
    'divergence_dlat': 2.176, 'divergence_dlon': 1.488,
    'geopotential_at_surface': 0.00926, 'geopotential_at_surface_del2': 4.511,
    'geopotential_at_surface_dlat': 0.1074, 'geopotential_at_surface_dlon': 0.08614,
    'land_sea_mask': 0.4403, 'learned_positional_features': 1.0,
    'log_surface_pressure': 0.1096, 'log_surface_pressure_del2': 48.61,
    'log_surface_pressure_dlat': 1.095, 'log_surface_pressure_dlon': 0.9575,
    'memory_divergence': 0.05232, 'memory_log_surface_pressure': 0.1123,
    'memory_specific_cloud_ice_water_content': 7.845e-06,
    'memory_specific_cloud_liquid_water_content': 1.648e-05,
    'memory_specific_humidity': 0.003298, 'memory_temperature_variation': 14.73,
    'memory_u': 0.01472, 'memory_v': 0.01001, 'memory_vorticity': 0.2168,
    'pressure': 1.643, 'radiation': 0.2867,
    'sea_ice_cover': 0.387, 'sea_surface_temperature': 11.93,
    'sin_latitude': 0.7043,
    'specific_cloud_ice_water_content': 7.845e-06,
    'specific_cloud_ice_water_content_del2': 0.01029,
    'specific_cloud_ice_water_content_dlat': 0.000203,
    'specific_cloud_ice_water_content_dlon': 0.0001601,
    'specific_cloud_liquid_water_content': 1.648e-05,
    'specific_cloud_liquid_water_content_del2': 0.02065,
    'specific_cloud_liquid_water_content_dlat': 0.0004055,
    'specific_cloud_liquid_water_content_dlon': 0.000309,
    'specific_humidity': 0.003281, 'specific_humidity_del2': 0.8403,
    'specific_humidity_dlat': 0.01955, 'specific_humidity_dlon': 0.01361,
    'surface_embedding': 1.0,
    'temperature_variation': 14.73, 'temperature_variation_del2': 2386.0,
    'temperature_variation_dlat': 62.26, 'temperature_variation_dlon': 50.23,
    'u': 0.01472, 'u_del2': 6.478, 'u_dlat': 0.682, 'u_dlon': 0.1073,
    'v': 0.01001, 'v_del2': 5.028, 'v_dlat': 0.3098, 'v_dlon': 0.1431,
    'vorticity': 0.2168, 'vorticity_del2': 290.5,
    'vorticity_dlat': 5.741, 'vorticity_dlon': 4.558,
}
_NORM_SHIFTS_NON_CNN = {
    'cos_latitude': 0.642,
    'divergence': -0.0, 'divergence_del2': 0.006,
    'divergence_dlat': 0.001, 'divergence_dlon': -0.0,
    'geopotential_at_surface': 0.004, 'geopotential_at_surface_del2': -0.068,
    'geopotential_at_surface_dlat': -0.011, 'geopotential_at_surface_dlon': -0.0,
    'land_sea_mask': 0.334, 'learned_positional_features': 0.0,
    'log_surface_pressure': 1.716, 'log_surface_pressure_del2': 0.819,
    'log_surface_pressure_dlat': 0.135, 'log_surface_pressure_dlon': 0.0,
    'memory_divergence': -0.0, 'memory_log_surface_pressure': 1.716,
    'memory_specific_cloud_ice_water_content': 0.0,
    'memory_specific_cloud_liquid_water_content': 0.0,
    'memory_specific_humidity': 0.003298, 'memory_temperature_variation': -4.946,
    'memory_u': 0.007, 'memory_v': -0.0, 'memory_vorticity': -0.002,
    'pressure': 2.798, 'radiation': 0.214,
    'sea_ice_cover': 0.24, 'sea_surface_temperature': 285.14,
    'sin_latitude': 0.0,
    'specific_cloud_ice_water_content': 0.0,
    'specific_cloud_ice_water_content_del2': 0.0,
    'specific_cloud_ice_water_content_dlat': 0.0,
    'specific_cloud_ice_water_content_dlon': -0.0,
    'specific_cloud_liquid_water_content': 0.0,
    'specific_cloud_liquid_water_content_del2': 0.0,
    'specific_cloud_liquid_water_content_dlat': 0.0,
    'specific_cloud_liquid_water_content_dlon': 0.0,
    'specific_humidity': 0.0, 'specific_humidity_del2': 0.0,
    'specific_humidity_dlat': 0.0, 'specific_humidity_dlon': 0.0,
    'surface_embedding': 0.0,
    'temperature_variation': -5.126, 'temperature_variation_del2': 46.171,
    'temperature_variation_dlat': 5.054, 'temperature_variation_dlon': 0.0,
    'u': 0.007, 'u_del2': 0.147, 'u_dlat': 0.002, 'u_dlon': 0.0,
    'v': -0.0, 'v_del2': -0.001, 'v_dlat': -0.0, 'v_dlon': 0.0,
    'vorticity': -0.002, 'vorticity_del2': -0.399,
    'vorticity_dlat': 0.036, 'vorticity_dlon': -0.0,
}
# CNN1D keys all have scale=0.1, shift=0.0
_ADVANCE_SCALES = {**{f'CNN1D_{i}': 0.1 for i in range(64)}, **_NORM_SCALES_NON_CNN}
_ADVANCE_SHIFTS = {**{f'CNN1D_{i}': 0.0 for i in range(64)}, **_NORM_SHIFTS_NON_CNN}
# Embedding model uses the same normalization constants
_EMBEDDING_SCALES = _ADVANCE_SCALES
_EMBEDDING_SHIFTS = _ADVANCE_SHIFTS

# InverseLevelScale for specific_humidity (32 per-level values)
_ADVANCE_INVERSE_LEVEL_SCALES = [
    8.825e-05, 7.041e-05, 0.0001044, 0.0001832, 0.0007474, 0.002588,
    0.007067, 0.01524, 0.02838, 0.04518, 0.06902, 0.09718, 0.1316,
    0.1716, 0.2182, 0.2748, 0.3379, 0.4073, 0.4611, 0.5177, 0.5798,
    0.6547, 0.7397, 0.8362, 0.9373, 1.041, 1.149, 1.265, 1.394,
    1.551, 1.71, 1.792,
]

# Output tendency transforms from JAX gin:
# DivCurlNeuralParameterization.tendency_transform_module =
#   @div_curl_tendency_outputs/SequentialTransform
# with:
#   1) InverseShiftAndNormalize(global_scale=0.02)
#   2) LevelScale(keys_to_scale=['specific_humidity'])
_GLOBAL_OUT_SCALE = 0.02
_TENDENCY_OUT_SCALES = {
    'log_surface_pressure': 0.0497,
    'temperature_variation': 33.62,
    'u': 0.0576,
    'v': 0.0506,
    'specific_cloud_ice_water_content': 0.0007559999999999999,
    'specific_cloud_liquid_water_content': 0.0014856000000000001,
    'specific_humidity': 0.005841,
}
_TENDENCY_OUT_SHIFTS = {
    'log_surface_pressure': 0.0,
    'temperature_variation': 0.0,
    'u': 0.0,
    'v': 0.0,
    'specific_cloud_ice_water_content': 0.0,
    'specific_cloud_liquid_water_content': 0.0,
    'specific_humidity': 0.0,
}
_TENDENCY_Q_LEVEL_SCALES = [
    0.0001361,
    0.0002096,
    0.000281,
    0.0004981,
    0.001449,
    0.004681,
    0.01312,
    0.02983,
    0.05774,
    0.09596,
    0.1486,
    0.211,
    0.2882,
    0.3727,
    0.47,
    0.5779,
    0.6897,
    0.8138,
    0.9033,
    0.9937,
    1.095,
    1.191,
    1.299,
    1.411,
    1.498,
    1.549,
    1.578,
    1.599,
    1.628,
    1.679,
    1.783,
    1.882,
]

# Approximate gelu matching JAX's gelu(approximate=True)
def _gelu_approx(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate='tanh')


# ═══════════════════════════════════════════════════════════════════════════
# EPD tower builder
# ═══════════════════════════════════════════════════════════════════════════

def build_epd_tower(
    input_size: int,
    output_size: int,
    latent_size: int = LATENT_SIZE,
    num_blocks: int = NUM_BLOCKS,
    process_n_hidden: int = 3,
    encode_bias: bool = True,
    decode_bias: bool = False,
    decode_n_hidden: int = 0,
) -> Epd:
    """Build EPD tower matching JAX EpdTower architecture.

    JAX Gin parameters:
      encode/MlpUniform.num_hidden_layers = 0, with_bias = True
      process/MlpUniform.num_hidden_layers = 3, with_bias = True
      decode/MlpUniform.num_hidden_layers = 0, with_bias = False

    Args:
        input_size: input features dimension.
        output_size: output features dimension.
        latent_size: hidden/latent dimension (LATENT_SIZE=384).
        num_blocks: number of residual process blocks (NUM_BLOCKS=5).
        process_n_hidden: hidden layers count per process MLP
            (JAX num_hidden_layers=3 → 4 linears).
        encode_bias: bias in encode MLP.
        decode_bias: bias in decode MLP.
        decode_n_hidden: hidden layers in decode MLP (0 for main, 1 for surface).
    """
    encode = Mlp(
        input_size, latent_size,
        intermediate_sizes=(),
        activation=_gelu_approx,
        activate_final=False,
        use_bias=encode_bias,
    )
    decode = Mlp(
        latent_size, output_size,
        intermediate_sizes=(latent_size,) * decode_n_hidden,
        activation=_gelu_approx,
        activate_final=False,
        use_bias=decode_bias,
    )
    process_blocks = [
        Mlp(
            latent_size, latent_size,
            intermediate_sizes=(latent_size,) * process_n_hidden,
            activation=_gelu_approx,
            activate_final=False,
            use_bias=True,
        )
        for _ in range(num_blocks)
    ]
    # JAX EpdTower: post_encode_activation = None
    return Epd(encode, decode, process_blocks)


def build_surface_epd(
    input_size: int,
    output_size: int = SURFACE_MODEL_OUTPUT_SIZE,
) -> Epd:
    """Build surface embedding EPD (small, hidden=8, 1 process block).

    JAX Gin:
      surface_model/EpdTower.num_process_blocks = 1
      surface_model/EpdTower.latent_size = 8
      surface_model_encode: num_hidden_layers=0, with_bias=True
      surface_model_process: num_hidden_layers=3, with_bias=True
      surface_model_decode: num_hidden_layers=1, with_bias=False
    """
    return build_epd_tower(
        input_size, output_size,
        latent_size=SURFACE_MODEL_LATENT_SIZE,
        num_blocks=1,
        process_n_hidden=3,
        encode_bias=True,
        decode_bias=False,
        decode_n_hidden=1,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Volume embedding (VerticalConvTower)
# ═══════════════════════════════════════════════════════════════════════════

def build_volume_conv_tower() -> VerticalConvTower:
    """Build VerticalConvTower matching JAX volume embedding.

    Architecture: 5 ConvLevel layers with kernel_size=5
      conv_0: (9  → 64)
      conv_1: (64 → 64)
      conv_2: (64 → 64)
      conv_3: (64 → 64)
      conv_4: (64 → 32)

    JAX uses gelu activation (ACTIVATION = @gelu in gin config).
    JAX jax.nn.gelu defaults to approximate=True (tanh), so use the tanh
    approximation in PT to match.
    """
    return VerticalConvTower(
        input_channels=VOLUME_CONV_INPUT_CH,
        output_channels=VOLUME_CONV_OUTPUT_CH,
        hidden_channels=VOLUME_CONV_HIDDEN_CH,
        kernel_size=VOLUME_CONV_KERNEL,
        activation=lambda x: F.gelu(x, approximate='tanh'),
        activate_final=False,
        use_bias=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# JaxNeuralComponents — all JAX-aligned learnable modules
# ═══════════════════════════════════════════════════════════════════════════

class JaxNeuralComponents(nn.Module):
    """All learnable neural network components matching the JAX checkpoint.

    This module holds 229 parameters matching the JAX deterministic_2_8_deg
    checkpoint. It does NOT implement the full forward pass (feature
    extraction pipeline varies by context). Instead, it provides the
    correctly-shaped nn.Module tree for weight loading.

    Submodule tree:
      physics_epd         — Epd(2365→384→192), 5 blocks
      physics_positional   — Parameter(8, 128, 64)
      physics_surface_embeds — ModuleList of 3 small Epd towers
      physics_volume_embed — VerticalConvTower(9→32)
      physics_corrector_orog — Parameter(4094)
      encoder_0_epd        — Epd(1052→384→193), 5 blocks
      encoder_0_positional — Parameter(8, 128, 64)
      encoder_0_orography  — Parameter(4223)
      encoder_1_epd        — Epd(1052→384→193), 5 blocks
      encoder_1_positional — Parameter(8, 128, 64)
      encoder_1_orography  — Parameter(4223)
      decoder_epd          — Epd(784→384→259), 5 blocks
      decoder_positional   — Parameter(8, 128, 64)
    """

    def __init__(self, resolution: str = "2.8"):
        super().__init__()
        _spec = _JAX_RESOLUTION_SPECS[resolution]
        _pl = _spec["positional_channels"]
        _nl = _spec["nodal_lon"]
        _na = _spec["nodal_lat"]
        _pi = _spec["physics_epd_input"]
        _ei = _spec["encoder_epd_input"]
        _di = _spec["decoder_epd_input"]
        _co = _spec["corrector_orog_size"]
        _eo = _spec["encoder_orog_size"]
        _se = _spec["surface_embed_inputs"]
        # Optional: decoder learned_orography (det 1.4°/0.7°).
        _do = _spec.get("decoder_orog_size")
        # Optional: JAX bias-only surface embedding (det 1.4°/0.7°).
        _land_ice_biases = bool(_spec.get("land_ice_biases", False))
        self._land_ice_biases = _land_ice_biases

        # ── Physics Parameterization ─────────────────────────────────
        self.physics_epd = build_epd_tower(_pi, PHYSICS_EPD_OUTPUT)
        self.physics_positional = nn.Parameter(
            torch.randn(_pl, _nl, _na) * 0.01
        )
        self.physics_surface_embeds = nn.ModuleList([
            build_surface_epd(in_size) for in_size in _se
        ])
        if _land_ice_biases:
            # Per-surface constant bias params (JAX ``land_params`` /
            # ``sea_ice_params``): shape (output_size, 1, 1) broadcast to
            # (output_size, lon, lat) inside EmbeddingSurfaceFeatures.
            self.physics_land_bias = nn.Parameter(
                torch.zeros(SURFACE_MODEL_OUTPUT_SIZE, 1, 1)
            )
            self.physics_sea_ice_bias = nn.Parameter(
                torch.zeros(SURFACE_MODEL_OUTPUT_SIZE, 1, 1)
            )
        self.physics_volume_embed = build_volume_conv_tower()
        self.physics_corrector_orog = nn.Parameter(torch.zeros(_co))

        # ── Encoder 0 ────────────────────────────────────────────────
        self.encoder_0_epd = build_epd_tower(_ei, ENCODER_EPD_OUTPUT)
        self.encoder_0_positional = nn.Parameter(
            torch.randn(_pl, _nl, _na) * 0.01
        )
        self.encoder_0_orography = nn.Parameter(torch.zeros(_eo))

        # ── Encoder 1 ────────────────────────────────────────────────
        self.encoder_1_epd = build_epd_tower(_ei, ENCODER_EPD_OUTPUT)
        self.encoder_1_positional = nn.Parameter(
            torch.randn(_pl, _nl, _na) * 0.01
        )
        self.encoder_1_orography = nn.Parameter(torch.zeros(_eo))

        # ── Decoder ──────────────────────────────────────────────────
        self.decoder_epd = build_epd_tower(_di, DECODER_EPD_OUTPUT)
        self.decoder_positional = nn.Parameter(
            torch.randn(_pl, _nl, _na) * 0.01
        )
        if _do is not None:
            # Some checkpoints (det 1.4°/0.7°) also store a decoder-side
            # learned_orography; loaded here to remove the "unmapped" warning
            # and stay bit-consistent with the JAX checkpoint.
            self.decoder_orography = nn.Parameter(torch.zeros(_do))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════════════════
# Weight conversion: JAX checkpoint → PyTorch state_dict
# ═══════════════════════════════════════════════════════════════════════════

_JAX_ROOT = "params/stochastic_modular_step_model/~/"
_DECODER_PREFIX = _JAX_ROOT + "dimensional_learned_primitive_to_weatherbench_decoder/"
_ENCODER_PREFIX = _JAX_ROOT + "dimensional_learned_weatherbench_to_primitive_with_memory_encoder/~/"
_PHYSICS_PREFIX = _JAX_ROOT + "stochastic_physics_parameterization_step/~/"


def _parse_jax_param_path(path: str) -> dict[str, Any] | None:
    """Parse a JAX Haiku parameter path into structured info.

    Returns dict with keys: component, subpath, param_type, or None if not a param.
    """
    if not path.startswith("params/"):
        return None

    # Strip params prefix
    p = path[len("params/stochastic_modular_step_model/~/"):]

    result = {"full_path": path}

    # Determine component
    if p.startswith("dimensional_learned_primitive_to_weatherbench_decoder/"):
        result["component"] = "decoder"
        p = p[len("dimensional_learned_primitive_to_weatherbench_decoder/"):]
    elif p.startswith("dimensional_learned_weatherbench_to_primitive_with_memory_encoder/~/"):
        result["component"] = "encoder"
        p = p[len("dimensional_learned_weatherbench_to_primitive_with_memory_encoder/~/"):]
    elif p.startswith("stochastic_physics_parameterization_step/~/"):
        result["component"] = "physics"
        p = p[len("stochastic_physics_parameterization_step/~/"):]
    else:
        return None

    result["subpath"] = p
    return result


def _epd_tower_jax_to_pt(
    jax_subpath: str,
    pt_prefix: str,
) -> tuple[str, str] | None:
    """Map EPD tower JAX subpath to PT state_dict key.

    Args:
        jax_subpath: path under epd_tower/, e.g.
            'encode_tower/~/mlp_uniform/~/linear_0/w'
        pt_prefix: PyTorch prefix, e.g. 'physics_epd'

    Returns:
        (pt_key, param_type) where param_type is 'linear_w', 'linear_b'.
    """
    # Encode tower
    m = re.match(r'encode_tower/~/mlp_uniform/~/linear_(\d+)/(w|b)', jax_subpath)
    if m:
        idx, wb = int(m.group(1)), m.group(2)
        pt_name = "weight" if wb == "w" else "bias"
        return f"{pt_prefix}.encode.layers.{idx}.{pt_name}", f"linear_{wb}"

    # Process tower (first block has no suffix, subsequent have _1, _2, ...)
    m = re.match(r'process_tower(?:_(\d+))?/~/mlp_uniform/~/linear_(\d+)/(w|b)', jax_subpath)
    if m:
        block_suffix, lin_idx, wb = m.group(1), int(m.group(2)), m.group(3)
        block_idx = int(block_suffix) if block_suffix else 0
        pt_name = "weight" if wb == "w" else "bias"
        return f"{pt_prefix}.process_blocks.{block_idx}.layers.{lin_idx}.{pt_name}", f"linear_{wb}"

    # Decode tower
    m = re.match(r'decode_tower/~/mlp_uniform/~/linear_(\d+)/(w|b)', jax_subpath)
    if m:
        idx, wb = int(m.group(1)), m.group(2)
        pt_name = "weight" if wb == "w" else "bias"
        return f"{pt_prefix}.decode.layers.{idx}.{pt_name}", f"linear_{wb}"

    return None


def _surface_epd_jax_to_pt(
    jax_subpath: str,
    pt_prefix: str,
) -> tuple[str, str] | None:
    """Map surface model EPD tower JAX subpath to PT key.

    Surface tower names: surface_model_encode_tower, surface_model_process_tower,
    surface_model_decode_tower.
    """
    # Encode tower
    m = re.match(r'surface_model_encode_tower/~/mlp_uniform/~/linear_(\d+)/(w|b)', jax_subpath)
    if m:
        idx, wb = int(m.group(1)), m.group(2)
        pt_name = "weight" if wb == "w" else "bias"
        return f"{pt_prefix}.encode.layers.{idx}.{pt_name}", f"linear_{wb}"

    # Process tower
    m = re.match(r'surface_model_process_tower/~/mlp_uniform/~/linear_(\d+)/(w|b)', jax_subpath)
    if m:
        idx, wb = int(m.group(1)), m.group(2)
        pt_name = "weight" if wb == "w" else "bias"
        return f"{pt_prefix}.process_blocks.0.layers.{idx}.{pt_name}", f"linear_{wb}"

    # Decode tower
    m = re.match(r'surface_model_decode_tower/~/mlp_uniform/~/linear_(\d+)/(w|b)', jax_subpath)
    if m:
        idx, wb = int(m.group(1)), m.group(2)
        pt_name = "weight" if wb == "w" else "bias"
        return f"{pt_prefix}.decode.layers.{idx}.{pt_name}", f"linear_{wb}"

    return None


def _vertical_conv_jax_to_pt(
    jax_subpath: str,
    pt_prefix: str,
) -> tuple[str, str] | None:
    """Map vertical conv tower JAX subpath to PT key.

    JAX: conv_level/w (kernel, in, out), conv_level/b (out, 1)
         conv_level_1/w, conv_level_1/b, ...
    PT:  layers.0.conv.weight (out, in, kernel), layers.0.conv.bias (out,)
    """
    m = re.match(r'conv_level(?:_(\d+))?/(w|b)', jax_subpath)
    if m:
        suffix, wb = m.group(1), m.group(2)
        idx = int(suffix) if suffix else 0
        if wb == "w":
            return f"{pt_prefix}.layers.{idx}.conv.weight", "conv_w"
        else:
            return f"{pt_prefix}.layers.{idx}.conv.bias", "conv_b"
    return None


def build_weight_mapping(jax_params: dict) -> dict[str, tuple[str, str]]:
    """Build JAX path → (PT state_dict key, param_type) mapping.

    Iterate all JAX param leaves and determine the corresponding PT key.
    param_type is one of: 'linear_w', 'linear_b', 'conv_w', 'conv_b',
    'positional', 'orography'.

    Returns:
        {jax_path: (pt_key, param_type)}
    """
    mapping: dict[str, tuple[str, str]] = {}

    for jax_path in jax_params:
        if not jax_path.startswith("params/"):
            continue

        parsed = _parse_jax_param_path(jax_path)
        if parsed is None:
            continue

        comp = parsed["component"]
        sub = parsed["subpath"]

        # ── DECODER ──────────────────────────────────────────────
        if comp == "decoder":
            # EPD tower
            if sub.startswith("nodal_mapping/~/epd_tower/"):
                epd_sub = sub[len("nodal_mapping/~/epd_tower/"):]
                result = _epd_tower_jax_to_pt(epd_sub, "decoder_epd")
                if result:
                    mapping[jax_path] = result
                    continue
            # Learned positional features
            if "learned_positional_features/learned_positional_features" in sub:
                mapping[jax_path] = ("decoder_positional", "positional")
                continue
            # Learned orography (present in det 1.4°/0.7° decoders)
            if "learned_orography/orography" in sub:
                mapping[jax_path] = ("decoder_orography", "orography")
                continue

        # ── ENCODER ──────────────────────────────────────────────
        elif comp == "encoder":
            # Determine sub-encoder index (0 or 1)
            # encoder_0: learned_weatherbench_to_primitive_encoder/...
            # encoder_1: learned_weatherbench_to_primitive_encoder_1/...
            enc_idx = None
            enc_sub = None
            if sub.startswith("learned_weatherbench_to_primitive_encoder_1/"):
                enc_idx = 1
                enc_sub = sub[len("learned_weatherbench_to_primitive_encoder_1/"):]
            elif sub.startswith("learned_weatherbench_to_primitive_encoder/"):
                enc_idx = 0
                enc_sub = sub[len("learned_weatherbench_to_primitive_encoder/"):]

            if enc_idx is not None and enc_sub is not None:
                prefix = f"encoder_{enc_idx}"

                # EPD tower
                if enc_sub.startswith("nodal_mapping/~/epd_tower/"):
                    epd_s = enc_sub[len("nodal_mapping/~/epd_tower/"):]
                    result = _epd_tower_jax_to_pt(epd_s, f"{prefix}_epd")
                    if result:
                        mapping[jax_path] = result
                        continue

                # Learned positional features
                if "learned_positional_features/learned_positional_features" in enc_sub:
                    mapping[jax_path] = (f"{prefix}_positional", "positional")
                    continue

                # Learned orography
                if "learned_orography/orography" in enc_sub:
                    mapping[jax_path] = (f"{prefix}_orography", "orography")
                    continue

        # ── PHYSICS ──────────────────────────────────────────────
        elif comp == "physics":
            # Corrector orography
            if "custom_coords_corrector" in sub and "learned_orography/orography" in sub:
                mapping[jax_path] = ("physics_corrector_orog", "orography")
                continue

            # Main EPD tower
            if sub.startswith("div_curl_neural_parameterization/nodal_mapping/~/epd_tower/"):
                epd_s = sub[len("div_curl_neural_parameterization/nodal_mapping/~/epd_tower/"):]
                result = _epd_tower_jax_to_pt(epd_s, "physics_epd")
                if result:
                    mapping[jax_path] = result
                    continue

            # Surface embeddings
            if "embedding_surface_features" in sub:
                # Per-surface constant bias params (JAX bias-only mode used by
                # deterministic 1.4°/0.7°): ``nodal_land_sea_ice_embedding/
                # {land,sea_ice}_params`` — shape (output_size, 1, 1).
                if sub.endswith("nodal_land_sea_ice_embedding/land_params"):
                    mapping[jax_path] = ("physics_land_bias", "orography")
                    continue
                if sub.endswith("nodal_land_sea_ice_embedding/sea_ice_params"):
                    mapping[jax_path] = ("physics_sea_ice_bias", "orography")
                    continue

                # Extract embedding index: modal_to_nodal_embedding (0),
                #   modal_to_nodal_embedding_1 (1), modal_to_nodal_embedding_2 (2)
                m = re.search(
                    r'modal_to_nodal_embedding(?:_(\d+))?/nodal_mapping/~/epd_tower/(.*)',
                    sub,
                )
                if m:
                    embed_suffix = m.group(1)
                    embed_idx = int(embed_suffix) if embed_suffix else 0
                    epd_sub = m.group(2)
                    result = _surface_epd_jax_to_pt(
                        epd_sub, f"physics_surface_embeds.{embed_idx}",
                    )
                    if result:
                        mapping[jax_path] = result
                        continue

            # Volume embedding (VerticalConvTower)
            if "embedding_volume_features" in sub:
                m = re.search(r'vertical_conv_tower/~/(.*)', sub)
                if m:
                    conv_sub = m.group(1)
                    result = _vertical_conv_jax_to_pt(
                        conv_sub, "physics_volume_embed",
                    )
                    if result:
                        mapping[jax_path] = result
                        continue

            # Learned positional features
            if "learned_positional_features/learned_positional_features" in sub:
                mapping[jax_path] = ("physics_positional", "positional")
                continue

        logger.warning("Unmapped JAX param: %s", jax_path)

    return mapping


def _convert_param(
    jax_value: np.ndarray,
    param_type: str,
) -> torch.Tensor:
    """Convert a single JAX parameter to PyTorch format.

    Handles:
      - linear_w: transpose (in, out) → (out, in)
      - linear_b: no change
      - conv_w: permute (kernel, in, out) → (out, in, kernel)
      - conv_b: squeeze (out, 1) → (out,)
      - positional: no change
      - orography: no change
    """
    t = torch.from_numpy(np.array(jax_value))

    if param_type == "linear_w":
        # JAX: (in_features, out_features) → PT: (out_features, in_features)
        return t.T.contiguous()
    elif param_type == "linear_b":
        return t
    elif param_type == "conv_w":
        # JAX: (kernel, in_ch, out_ch) → PT: (out_ch, in_ch, kernel)
        return t.permute(2, 1, 0).contiguous()
    elif param_type == "conv_b":
        # JAX: (out_ch, 1) → PT: (out_ch,)
        return t.squeeze(-1)
    elif param_type in ("positional", "orography"):
        return t
    else:
        raise ValueError(f"Unknown param_type: {param_type}")


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def load_jax_checkpoint(checkpoint_path: str | Path) -> dict:
    """Load a JAX NeuralGCM pickle checkpoint.

    Returns:
        Flat dict mapping 'params/...' paths to numpy arrays.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)

    # Flatten nested dict to 'a/b/c' paths
    flat = {}

    def _flatten(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _flatten(v, f"{prefix}{k}/")
        elif isinstance(obj, (np.ndarray, np.generic)):
            flat[prefix.rstrip("/")] = obj
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _flatten(v, f"{prefix}[{i}]/")
        elif isinstance(obj, (int, float, str)):
            flat[prefix.rstrip("/")] = obj

    _flatten(data)
    return flat


def build_jax_neural_components(resolution: str = "2.8") -> JaxNeuralComponents:
    """Build a JaxNeuralComponents module with correct architecture.

    Returns the module with random weights; call load_jax_weights() to
    populate from a JAX checkpoint.
    """
    return JaxNeuralComponents(resolution=resolution)


def load_jax_weights(
    model: JaxNeuralComponents,
    checkpoint_path: str | Path,
    strict: bool = True,
) -> dict[str, str]:
    """Load JAX checkpoint weights into PyTorch JaxNeuralComponents.

    Args:
        model: JaxNeuralComponents to load into.
        checkpoint_path: path to JAX pickle checkpoint.
        strict: if True, raise on unmapped / missing params.

    Returns:
        Summary dict with stats: n_mapped, n_unmapped, n_missing.
    """
    flat_params = load_jax_checkpoint(checkpoint_path)

    # Filter to actual param arrays
    param_arrays = {
        k: v for k, v in flat_params.items()
        if k.startswith("params/") and isinstance(v, np.ndarray)
    }

    mapping = build_weight_mapping(param_arrays)

    # Build PyTorch state_dict
    pt_state = {}
    mapped_jax_keys = set()
    shape_mismatches = []

    for jax_path, (pt_key, param_type) in mapping.items():
        jax_val = param_arrays[jax_path]
        pt_tensor = _convert_param(jax_val, param_type)
        pt_state[pt_key] = pt_tensor
        mapped_jax_keys.add(jax_path)

    # Check for unmapped JAX params
    unmapped = set(param_arrays.keys()) - mapped_jax_keys
    if unmapped:
        logger.warning("Unmapped JAX params (%d):", len(unmapped))
        for p in sorted(unmapped):
            logger.warning("  %s", p)

    # Load into model
    model_sd = model.state_dict()
    missing_in_jax = set(model_sd.keys()) - set(pt_state.keys())
    extra_in_jax = set(pt_state.keys()) - set(model_sd.keys())

    # Validate shapes
    for key in set(pt_state.keys()) & set(model_sd.keys()):
        if pt_state[key].shape != model_sd[key].shape:
            shape_mismatches.append(
                f"  {key}: PT expects {model_sd[key].shape}, "
                f"got {pt_state[key].shape}"
            )

    if shape_mismatches:
        msg = "Shape mismatches:\n" + "\n".join(shape_mismatches)
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    if missing_in_jax:
        msg = f"Missing in JAX checkpoint ({len(missing_in_jax)}): {sorted(missing_in_jax)}"
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    if extra_in_jax:
        msg = f"Extra from JAX ({len(extra_in_jax)}): {sorted(extra_in_jax)}"
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    # Actually load (allow partial if not strict)
    model.load_state_dict(pt_state, strict=strict)

    summary = {
        "n_mapped": len(mapped_jax_keys),
        "n_unmapped_jax": len(unmapped),
        "n_missing_in_jax": len(missing_in_jax),
        "n_extra_from_jax": len(extra_in_jax),
        "n_shape_mismatches": len(shape_mismatches),
    }
    return summary


def build_and_load_jax_model(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[JaxNeuralComponents, dict]:
    """Build model + load JAX weights in one call.

    Args:
        checkpoint_path: path to JAX pickle checkpoint.
        device: torch device.

    Returns:
        (model, summary_dict)
    """
    model = build_jax_neural_components()
    summary = load_jax_weights(model, checkpoint_path, strict=False)
    model = model.to(device)
    model.eval()
    return model, summary


# ═══════════════════════════════════════════════════════════════════════════
# Utility to add full model building to exp_utils
# ═══════════════════════════════════════════════════════════════════════════

def build_jax_aligned_experiment_model(
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
    resolution: str = "2.8",
):
    """Build a full NeuralGCMModel with architecture matching the JAX checkpoint.

    This wires the complete JAX-aligned feature extraction pipeline
    (CombinedFeatures → ShiftAndNormalize → InverseLevelScale → SoftClip)
    producing 2365 features for the physics EPD.

    When checkpoint_path is provided, loads JAX weights into the model.

    Args:
        checkpoint_path: path to JAX .pkl checkpoint.
        device: torch device string.
        resolution: one of "2.8", "1.4", "0.7".

    Returns:
        (model, neural_components, coords, config, summary)
    """
    # Auto-detect resolution from checkpoint filename to prevent mismatch.
    # e.g. deterministic_2_8_deg.pkl → "2.8"
    if checkpoint_path is not None:
        _m = re.search(r'deterministic_(\d+)_(\d+)_deg', str(checkpoint_path))
        if _m:
            _ckpt_str = f"{int(_m.group(1))}.{int(_m.group(2))}"
            _ckpt_res = {"2.8": "2.8", "1.4": "1.4", "0.7": "0.7"}.get(_ckpt_str)
            if _ckpt_res is not None and _ckpt_res != resolution:
                logger.warning(
                    "Checkpoint resolution (%s) does not match requested "
                    "resolution (%s). Using checkpoint resolution.",
                    _ckpt_res, resolution,
                )
                resolution = _ckpt_res

    from tornado_gcm.scripts.data_preprocessing import build_coords_from_csv
    from tornado_gcm.core import coordinate_systems as core_coords
    from tornado_gcm.core import filtering
    from tornado_gcm.core import primitive_equations as pe_mod
    from tornado_gcm.core import spherical_harmonic
    from tornado_gcm.core import sigma_coordinates as sigma_mod
    from tornado_gcm.neural.fixers import (
        F64ConservationFixer, build_physical_ctx,
    )
    from tornado_gcm.neural.parameterizations import JaxAlignedDivCurlParameterization
    from tornado_gcm.neural.features import (
        CombinedFeatures,
        VelocityAndPrognostics,
        MemoryVelocityAndValues,
        PressureFeatures,
        RadiationFeatures,
        LatitudeFeatures,
        FloatDataFeatures,
        EmbeddingSurfaceFeatures,
        EmbeddingVolumeFeatures,
        LearnedPositionalFeatures,
    )
    from tornado_gcm.neural.transforms import (
        ShiftAndNormalize,
        InverseShiftAndNormalize,
        InverseLevelScale,
        SoftClip,
        SequentialTransform,
        ToModalDiffOperators,
        TendencyTransform,
    )
    from tornado_gcm.model.api import ModelConfig, NeuralGCMModel
    from tornado_gcm.precision.policy import PrecisionPolicy
    from tornado_gcm.units import SimUnits
    import xarray as xr

    # Build coordinates matching the JAX checkpoint grid.
    # The checkpoint dycore uses equidistant sigma layers, so we must not
    # use the hybrid-CSV-derived sigma profile here.
    _spec = _JAX_RESOLUTION_SPECS[resolution]
    _coords_csv, _hybrid = build_coords_from_csv(
        n_levels=N_SIGMA_LAYERS, resolution=_spec["grid_name"]
    )
    sigma = sigma_mod.SigmaCoordinates.equidistant(
        N_SIGMA_LAYERS, dtype=np.float64
    )
    coords = core_coords.CoordinateSystem(
        horizontal=_coords_csv.horizontal,
        vertical=sigma,
    )
    logger.info(
        "Using equidistant sigma vertical levels to match JAX checkpoint "
        "(layers=%d, max dsigma=%.6f, min dsigma=%.6f)",
        sigma.layers,
        float(np.max(sigma.layer_thickness)),
        float(np.min(sigma.layer_thickness)),
    )
    grid = coords.horizontal

    # ── Physics specs from JAX SimUnits (nondimensional constants) ────
    physics_specs = SimUnits.from_si()

    # ── Load auxiliary data from checkpoint (geopotential, land/sea mask) ──
    _aux_geopotential = None
    _aux_land_sea_mask = None
    _reference_datetime_str = "1979-01-01T00"  # JAX default (model_builder.py)
    _gin_cfg = ""
    _reference_gin_cfg = ""
    _reference_gin_path = (
        Path(__file__).resolve().parents[2]
        / "neuralgcm"
        / "neuralgcm"
        / "reference_code"
        / "paper_configs"
        / f"{_spec['gin_stem']}.gin"
    )
    if _reference_gin_path.exists():
        _reference_gin_cfg = _reference_gin_path.read_text(encoding="utf-8")
    if checkpoint_path is not None:
        with open(checkpoint_path, "rb") as _f:
            _ckpt_aux = pickle.load(_f)
        # Extract reference_datetime_str from gin config if explicitly set
        _gin_cfg = _ckpt_aux.get("model_config_str", "")
        import re as _re
        _rdt_match = _re.search(
            r"get_model_specs\.reference_datetime_str\s*=\s*'([^']+)'", _gin_cfg
        )
        if _rdt_match:
            _reference_datetime_str = _rdt_match.group(1)
            logger.info("Reference datetime from checkpoint gin config: %s", _reference_datetime_str)
        else:
            logger.info("Using default reference datetime: %s", _reference_datetime_str)
        _aux_dict = _ckpt_aux.get("aux_ds_dict")
        if _aux_dict is not None:
            try:
                _aux_ds = xr.Dataset.from_dict(_aux_dict)
                if "geopotential_at_surface" in _aux_ds:
                    _aux_geopotential = torch.from_numpy(
                        _aux_ds["geopotential_at_surface"].values.copy()
                    ).float().unsqueeze(0)  # (1, lon, lat)
                    # Checkpoint stores geopotential in SI units (m²/s²).
                    # JAX internally uses nondimensional units (÷ vel_scale²).
                    # vel_scale² = (radius * 2 * Omega)²
                    _R = scales.RADIUS.magnitude       # m
                    _Om = scales.OMEGA.magnitude       # rad/s
                    _vel_sq = (_R * 2.0 * _Om) ** 2   # m²/s²
                    _aux_geopotential = _aux_geopotential / _vel_sq
                    logger.info("Loaded geopotential_at_surface from checkpoint "
                                "(nondim range [%.4f, %.4f])",
                                _aux_geopotential.min().item(),
                                _aux_geopotential.max().item())
                if "land_sea_mask" in _aux_ds:
                    _aux_land_sea_mask = torch.from_numpy(
                        _aux_ds["land_sea_mask"].values.copy()
                    ).float().unsqueeze(0)  # (1, lon, lat)
                    logger.info("Loaded land_sea_mask from checkpoint")
            except Exception as _e:
                logger.warning("Could not load aux data from checkpoint: %s", _e)

    # Build neural components
    components = build_jax_neural_components(resolution=resolution)

    # ── Build feature modules ────────────────────────────────────────

    # Fields used by VelocityAndPrognostics (advance path)
    advance_fields = [
        "divergence", "vorticity", "u", "v",
        "temperature_variation", "log_surface_pressure",
        "specific_humidity", "specific_cloud_liquid_water_content",
        "specific_cloud_ice_water_content",
    ]
    grad_module = ToModalDiffOperators(grid)

    velocity_progs = VelocityAndPrognostics(
        grid=grid,
        fields_to_include=advance_fields,
        compute_gradients_fn=grad_module,
    )

    memory_module = MemoryVelocityAndValues(
        grid=grid,
        fields_to_include=advance_fields,
        compute_gradients_fn=None,  # No gradients for memory
    )

    pressure_module = PressureFeatures(
        grid=grid,
        sigma_centers=coords.vertical.centers,
    )

    radiation_module = RadiationFeatures(grid=grid, reference_datetime_str=_reference_datetime_str)
    latitude_module = LatitudeFeatures(grid=grid)

    # FloatDataFeatures(with_grads): geopotential_at_surface
    geopotential_data = (_aux_geopotential if _aux_geopotential is not None
                         else torch.zeros(1, grid.nodal_shape[-2], grid.nodal_shape[-1]))
    float_with_grads = FloatDataFeatures(
        grid=grid,
        covariates={"geopotential_at_surface": geopotential_data},
        compute_gradients_fn=ToModalDiffOperators(grid),
    )

    # FloatDataFeatures(without_grads): land_sea_mask
    land_sea_mask_data = (_aux_land_sea_mask if _aux_land_sea_mask is not None
                          else torch.zeros(1, grid.nodal_shape[-2], grid.nodal_shape[-1]))
    float_without_grads = FloatDataFeatures(
        grid=grid,
        covariates={"land_sea_mask": land_sea_mask_data},
        compute_gradients_fn=None,
    )

    # EmbeddingSurfaceFeatures
    # 2.8° has 3 surface embeds (land, sea, ice) via separate Epds [4,1,5].
    # Det 1.4°/0.7° use "bias-only" mode: one Epd for sea (SST) + per-surface
    # constant bias parameters (``land_params`` / ``sea_ice_params``, shape
    # (8,1,1)) broadcast across the nodal grid — matches JAX
    # ``NodalLandSeaIceEmbedding`` with land/sea_ice embeddings=None.
    _n_surf = len(_spec["surface_embed_inputs"])
    _land_ice_biases = bool(_spec.get("land_ice_biases", False))
    _surface_land_bias = None
    _surface_sea_ice_bias = None
    if _n_surf >= 3:
        surface_land_epd = ForwardTower(
            neural_net=components.physics_surface_embeds[0], feature_axis=-3)
        surface_sea_epd = ForwardTower(
            neural_net=components.physics_surface_embeds[1], feature_axis=-3)
        surface_ice_epd = ForwardTower(
            neural_net=components.physics_surface_embeds[2], feature_axis=-3)
    elif _land_ice_biases:
        # Single sea-only Epd; land + sea_ice use constant bias params.
        surface_sea_epd = ForwardTower(
            neural_net=components.physics_surface_embeds[0], feature_axis=-3)
        surface_land_epd = None
        surface_ice_epd = None
        _surface_land_bias = components.physics_land_bias
        _surface_sea_ice_bias = components.physics_sea_ice_bias
    else:
        _shared = ForwardTower(
            neural_net=components.physics_surface_embeds[0], feature_axis=-3)
        surface_land_epd = surface_sea_epd = surface_ice_epd = _shared
    # Use shared normalization scales/shifts
    norm_pair = (_ADVANCE_SCALES, _ADVANCE_SHIFTS)
    surface_embedding = EmbeddingSurfaceFeatures(
        grid=grid,
        land_epd=surface_land_epd,
        sea_epd=surface_sea_epd,
        sea_ice_epd=surface_ice_epd,
        land_sea_mask=land_sea_mask_data.clone(),
        land_norm=norm_pair,
        sea_norm=norm_pair,
        sea_ice_norm=norm_pair,
        land_bias=_surface_land_bias,
        sea_ice_bias=_surface_sea_ice_bias,
    )

    # EmbeddingVolumeFeatures
    volume_embedding = EmbeddingVolumeFeatures(
        grid=grid,
        conv_tower=components.physics_volume_embed,
        sigma_centers=coords.vertical.centers,
        n_output=N_CNN_FEATURES,
        feature_name="CNN1D",
        embedding_scales=_EMBEDDING_SCALES,
        embedding_shifts=_EMBEDDING_SHIFTS,
    )

    # LearnedPositionalFeatures
    positional_module = LearnedPositionalFeatures(
        positional_param=components.physics_positional,
    )

    # ── CombinedFeatures ─────────────────────────────────────────────
    feature_extractor = CombinedFeatures(
        feature_modules=[
            surface_embedding,
            volume_embedding,
            pressure_module,
            radiation_module,
            latitude_module,
            velocity_progs,
            memory_module,
            float_with_grads,
            float_without_grads,
            positional_module,
        ],
    )

    # ── Feature transforms: ShiftAndNormalize → InverseLevelScale → SoftClip
    feature_transform = SequentialTransform(transforms=[
        ShiftAndNormalize(shifts=_ADVANCE_SHIFTS, scales=_ADVANCE_SCALES),
        InverseLevelScale(
            keys_to_scale=[
                "specific_humidity",
                "specific_humidity_del2",
                "specific_humidity_dlat",
                "specific_humidity_dlon",
            ],
            scales=_ADVANCE_INVERSE_LEVEL_SCALES,
        ),
        SoftClip(max_value=16.0, hinge_softness=1.0),
    ])

    # ── Physics EPD tower ────────────────────────────────────────────
    physics_tower = ForwardTower(
        neural_net=components.physics_epd,
        feature_axis=-3,
    )

    # ── Tendency output transforms (JAX DivCurlNeuralParameterization) ──
    tendency_transform = SequentialTransform(transforms=[
        InverseShiftAndNormalize(
            shifts=_TENDENCY_OUT_SHIFTS,
            scales=_TENDENCY_OUT_SCALES,
            global_scale=_GLOBAL_OUT_SCALE,
        ),
        TendencyTransform(
            level_scales={
                "specific_humidity": torch.tensor(
                    _TENDENCY_Q_LEVEL_SCALES, dtype=torch.float32,
                )
            }
        ),
    ])

    # ── Full parameterization ────────────────────────────────────────
    parameterization = JaxAlignedDivCurlParameterization(
        grid=grid,
        neural_net=physics_tower,
        feature_extractor=feature_extractor,
        feature_transform=feature_transform,
        tendency_transform=tendency_transform,
        n_levels=N_SIGMA_LAYERS,
        filter_fn=None,
    )

    # The fixers conserve a *physical* quantity when given the integral's
    # context.  The surface-geopotential term is intentionally omitted here:
    # its nondimensional convention (checkpoint stores SI geopotential divided
    # by vel_scale^2) has not been reconciled with the energy integral's
    # convention, and a wrong constant term would bias every correction.
    # Recorded as ISSUE-008 in revision/plan/00_issues.md.
    _phys_ctx = build_physical_ctx(
        coords=coords,
        physics_specs=physics_specs,
        reference_temperature=REFERENCE_TEMPERATURES,
        orography=None,
    )
    fixer = F64ConservationFixer(physical_ctx=_phys_ctx)
    policy = PrecisionPolicy()

    # Nondimensional dt from JAX checkpoint (Gin config):
    #   dt = 0.525024 ≈ 3600s / 6856.83s (scale [time]) = 1 hour.
    #   N_INNER_DYCORE_STEPS = 5 from Gin config.
    _JAX_DT_NONDIM = 0.525024

    # Orography in modal space: nondimensionalize geopotential (m²/s²)
    # Aux geopotential is already converted to nondimensional units above.
    #
    # JAX FilteredCustomOrography chain:
    #   1. Load nodal orography from dataset
    #   2. Spectral interpolate from dataset native coords → target coords
    #   3. Apply orography/DataExponentialFilter (modal-space filter)
    #
    # PT equivalent: grid.to_modal() (steps 1+2, same-grid shortcut).
    # The orography/DataExponentialFilter is applied below in the gin-parsing
    # section if its attenuation > 0 (for deterministic_2_8_deg it is 0.0).
    orography_modal = None
    if _aux_geopotential is not None:
        _geopot_nodal = _aux_geopotential.squeeze(0)  # already nondim, (lon, lat)
        orography_modal = grid.to_modal(_geopot_nodal.float())
        logger.info("Orography modal shape: %s, range [%.4f, %.4f], l2=%.6e",
                     orography_modal.shape, orography_modal.min().item(),
                     orography_modal.max().item(),
                     orography_modal.norm(p=2).item())

    corrector_coords = None
    corrector_orography = None
    if _gin_cfg or _reference_gin_cfg:
        def _extract_gin_value(key: str, default: Any = None) -> Any:
            for gin_text in (_gin_cfg, _reference_gin_cfg):
                if not gin_text:
                    continue
                match = re.search(rf"^{re.escape(key)}\s*=\s*(.+)$", gin_text, re.MULTILINE)
                if not match:
                    continue
                raw = match.group(1).strip()
                # Resolve gin macro references (%MACRO_NAME) recursively
                # Macros are defined as "MACRO_NAME = value" in the same gin text.
                _max_macro_depth = 10
                for _ in range(_max_macro_depth):
                    m = re.match(r'^%(\w+(?:/\w+)*)$', raw)
                    if not m:
                        break
                    macro_name = m.group(1)
                    macro_match = re.search(
                        rf"^{re.escape(macro_name)}\s*=\s*(.+)$",
                        gin_text, re.MULTILINE,
                    )
                    if macro_match:
                        raw = macro_match.group(1).strip()
                    else:
                        logger.warning("Unresolved gin macro: %s", raw)
                        break
                if raw == "None":
                    return None
                if (raw.startswith("'") and raw.endswith("'")) or (
                    raw.startswith('"') and raw.endswith('"')
                ):
                    return raw[1:-1]
                try:
                    if any(ch in raw for ch in (".", "e", "E")):
                        return float(raw)
                    return int(raw)
                except ValueError:
                    return raw
            return default

        # ── Extract orography correction & filter settings from gin ─────
        # JAX: LearnedOrography(base=FilteredCustomOrography, correction_scale)
        #      FilteredCustomOrography applies orography/DataExponentialFilter
        correction_scale = _extract_gin_value(
            "LearnedOrography.correction_scale", 1e-05
        )
        data_filter_atten = _extract_gin_value(
            "orography/DataExponentialFilter.attenuation", 0.0
        )
        data_filter_order = int(_extract_gin_value(
            "orography/DataExponentialFilter.order", 1
        ))
        data_filter_cutoff = float(_extract_gin_value(
            "orography/DataExponentialFilter.cutoff", 0.0
        ))
        logger.info(
            "Gin orography: correction_scale=%.2e, "
            "data_filter(atten=%.4f, order=%d, cutoff=%.2f)",
            correction_scale, data_filter_atten, data_filter_order, data_filter_cutoff,
        )
        print(
            f"[oro-loader] correction_scale={correction_scale:.2e}, "
            f"data_filter_atten={data_filter_atten}, "
            f"data_filter_order={data_filter_order}, "
            f"data_filter_cutoff={data_filter_cutoff}"
        )

        # ── Prepare orography data filter (JAX orography/DataExponentialFilter) ──
        # JAX applies this filter AFTER spectral interpolation to the target
        # coords (corrector grid). We build the filter fn here and apply it
        # to corrector_orography below (after interpolation).
        # For deterministic_2_8_deg, DATA_FILTER_ATTENUATION = 0.0 → identity.
        _oro_filter_fn = None
        if data_filter_atten > 0:
            _oro_filter_fn = filtering.exponential_filter(
                grid,
                attenuation=data_filter_atten,
                order=data_filter_order,
                cutoff=data_filter_cutoff,
            )
            logger.info(
                "Orography DataExponentialFilter ready: atten=%.4f, order=%d, "
                "cutoff=%.2f (will apply on corrector grid after interpolation)",
                data_filter_atten, data_filter_order, data_filter_cutoff,
            )
        elif orography_modal is not None:
            logger.info(
                "Orography DataExponentialFilter skipped (atten=%.4f ≤ 0)",
                data_filter_atten,
            )

        custom_lw = _extract_gin_value(
            "smaller/GridWithWavenumbers.longitude_wavenumbers"
        )
        # Skip corrector grid when it's within 1 wavenumber of the main grid —
        # the dealiasing margin is negligible and the mismatch causes shape errors
        # (e.g. state at 127×65 vs corrector at 125×64) for scripts that call
        # dycore methods directly without spectral-interpolation.
        if (custom_lw is not None
                and abs(int(custom_lw) - coords.horizontal.longitude_wavenumbers) > 1):
            custom_dealias = str(_extract_gin_value(
                "smaller/GridWithWavenumbers.dealiasing", "quadratic"
            ))
            custom_lat_spacing = str(_extract_gin_value(
                "smaller/GridWithWavenumbers.latitude_spacing", "gauss"
            ))
            custom_lon_offset = float(_extract_gin_value(
                "smaller/GridWithWavenumbers.longitude_offset", 0.0
            ))
            custom_radius = _extract_gin_value(
                "smaller/GridWithWavenumbers.radius", None
            )
            corrector_grid = spherical_harmonic.Grid.with_wavenumbers(
                longitude_wavenumbers=int(custom_lw),
                dealiasing=custom_dealias,
                latitude_spacing=custom_lat_spacing,
                longitude_offset=custom_lon_offset,
                radius=float(1.0 if custom_radius is None else custom_radius),
            )
            corrector_coords = core_coords.CoordinateSystem(
                horizontal=corrector_grid,
                vertical=coords.vertical,
            )

            if orography_modal is not None:
                corrector_orography = core_coords.get_spectral_interpolate_fn(
                    coords, corrector_coords,
                )(orography_modal)
            else:
                corrector_orography = torch.zeros(
                    corrector_coords.modal_shape[1:], dtype=torch.float32,
                )

            # ── Apply orography data filter on corrector grid ──────────
            # JAX FilteredCustomOrography applies DataExponentialFilter
            # AFTER spectral interpolation to the target coords.
            # This matches: filtered_modal_orography(..., coords=corrector_coords).
            if corrector_orography is not None and _oro_filter_fn is not None:
                _oro_l2_before = corrector_orography.norm(p=2).item()
                corrector_orography = _oro_filter_fn(corrector_orography)
                _oro_l2_after = corrector_orography.norm(p=2).item()
                logger.info(
                    "Corrector orography after DataExponentialFilter: "
                    "l2=%.6e (was %.6e, Δ=%.2e)",
                    _oro_l2_after, _oro_l2_before,
                    _oro_l2_before - _oro_l2_after,
                )

            if checkpoint_path is not None:
                flat_params = load_jax_checkpoint(checkpoint_path)
                param_arrays = {
                    k: v for k, v in flat_params.items()
                    if k.startswith("params/") and isinstance(v, np.ndarray)
                }
                mapping = build_weight_mapping(param_arrays)
                correction_vec = None
                for jax_path, (pt_key, param_type) in mapping.items():
                    if pt_key == "physics_corrector_orog":
                        correction_vec = _convert_param(
                            param_arrays[jax_path], param_type,
                        ).float()
                        break
                if correction_vec is not None:
                    # ── Apply learned orography correction ──────────────
                    # JAX: nodal_orography_from_ds converts geopotential→height
                    # (÷g_SI), then nondimensionalizes. LearnedOrography adds
                    # correction*scale in HEIGHT space.
                    #
                    # PT: corrector_orography is in nondim GEOPOTENTIAL (Φ/vel²).
                    # The Showstopper B fix (÷_g_nondim) converts to height space
                    # but runs AFTER this block. So we must scale the correction
                    # by _g_nondim so it lands in the right units after division.
                    #
                    # Without scaling: correction_contribution = corr*scale / g
                    #                   ≈ 1.125e-4 / 72.36 ≈ 1.56e-6 (72× too small)
                    _g_nondim = float(physics_specs.g)
                    _corr_stats = {
                        'min': correction_vec.min().item(),
                        'max': correction_vec.max().item(),
                        'mean': correction_vec.mean().item(),
                        'std': correction_vec.std().item(),
                        'l2': correction_vec.norm(p=2).item(),
                    }
                    logger.info(
                        "Loaded corrector correction: shape=%s, "
                        "min=%.6e, max=%.6e, mean=%.6e, std=%.6e, l2=%.6e",
                        tuple(correction_vec.shape),
                        _corr_stats['min'], _corr_stats['max'],
                        _corr_stats['mean'], _corr_stats['std'],
                        _corr_stats['l2'],
                    )
                    print(
                        f"[oro-loader] correction_vec: shape={tuple(correction_vec.shape)}, "
                        f"min={_corr_stats['min']:.4e}, max={_corr_stats['max']:.4e}, "
                        f"mean={_corr_stats['mean']:.4e}, std={_corr_stats['std']:.4e}, "
                        f"l2={_corr_stats['l2']:.4e}"
                    )

                    correction_modal = torch.zeros_like(corrector_orography)
                    correction_modal[torch.as_tensor(
                        corrector_grid.mask, dtype=torch.bool,
                    )] = correction_vec.to(
                        dtype=correction_modal.dtype,
                        device=correction_modal.device,
                    )
                    _oro_l2_before_corr = corrector_orography.norm(p=2).item()
                    _corr_contrib_raw = (correction_modal * correction_scale).norm(p=2).item()
                    _corr_contrib_scaled = (correction_modal * correction_scale * _g_nondim).norm(p=2).item()
                    corrector_orography = (
                        corrector_orography
                        + correction_modal * correction_scale * _g_nondim
                    )
                    _oro_l2_after_corr = corrector_orography.norm(p=2).item()
                    logger.info(
                        "Corrector orography: l2=%.6e (base=%.6e, "
                        "correction*%.2e*g=%.6e, Δ=%.2e, shape=%s, "
                        "range [%.6e, %.6e])",
                        _oro_l2_after_corr, _oro_l2_before_corr,
                        correction_scale, _corr_contrib_scaled,
                        _oro_l2_before_corr - _oro_l2_after_corr,
                        tuple(corrector_orography.shape),
                        corrector_orography.min().item(),
                        corrector_orography.max().item(),
                    )
                    print(
                        f"[oro-loader] corrector_orography final: "
                        f"l2={_oro_l2_after_corr:.6e} (base={_oro_l2_before_corr:.6e}, "
                        f"correction*{correction_scale:.2e}*g={_corr_contrib_scaled:.6e}, "
                        f"raw_correction_l2={_corr_contrib_raw:.6e}, "
                        f"g_nondim={_g_nondim:.4f}, "
                        f"delta={_oro_l2_before_corr - _oro_l2_after_corr:.6e})"
                    )
                else:
                    print("[oro-loader] WARNING: correction_vec is None — "
                          "learned orography correction NOT applied!")

    # ── Compute filter attenuations matching JAX gin config ─────────
    # deterministic_2_8_deg.gin: NUM_SUBSTEPS=1 → corrector ctor `dt` equals
    # full physics nondim dt. ExponentialFilter uses exponential_step_filter(
    # grid, dt, tau, ...) → attenuation = dt / tau (same _JAX_DT_NONDIM here).
    # (IMEX substep size is dt / dycore_substeps separately in time_integrator.)
    _DYCORE_SUBSTEPS = 5
    _dycore_tau_nondim = physics_specs.nondimensionalize(
        120 * scales.units.minute)  # DYCORE_TAU = '120 minutes'
    _stability_tau_nondim = physics_specs.nondimensionalize(
        4 * scales.units.minute)    # STABILITY_TAU = '4 minutes'

    _dycore_filter_atten = _JAX_DT_NONDIM / _dycore_tau_nondim
    _stability_filter_atten = _JAX_DT_NONDIM / _stability_tau_nondim
    logger.info(
        "Filter attenuations (physics_dt=%.8f): dycore=%.6f (tau=%.5f), "
        "stability=%.6f (tau=%.5f)",
        _JAX_DT_NONDIM,
        _dycore_filter_atten,
        _dycore_tau_nondim,
        _stability_filter_atten,
        _stability_tau_nondim,
    )

    # JAX DivCurlNeuralParameterization.filter_module = @ml/SequentialStepFilter
    # with (@stability/ExponentialFilter,), i.e. apply stability filtering to
    # parameterization modal tendencies before dycore coupling.
    parameterization.filter_fn = filtering.exponential_filter(
        grid,
        attenuation=_stability_filter_atten,
        order=10,
        cutoff=0.4,
    )

    # ── Orography unit correction (Showstopper B fix) ───────────────
    # ``orography_modal`` / ``corrector_orography`` above hold nondimensional
    # surface GEOPOTENTIAL (Φ/vel²). The dynamical core
    # (PrimitiveEquationsSigma) treats ``orography`` as a HEIGHT-like field and
    # reconstructs surface geopotential as ``orography * g`` (and the orography
    # tendency as ``-g·∇²(orography)``) with ``g = physics_specs.g`` (nondim
    # ≈ 72.36). Passing nondim geopotential directly therefore over-scales the
    # orographic forcing by g_nondim and blows up the dycore in one outer step
    # (validated: dividing the dycore orography by g_nondim restores stability).
    # Divide here so the dycore's ``* g`` recovers the correct Φ/vel².
    _g_nondim = float(physics_specs.g)
    if orography_modal is not None:
        orography_modal = orography_modal / _g_nondim
    if corrector_orography is not None:
        corrector_orography = corrector_orography / _g_nondim
    logger.info("Orography rescaled by 1/g_nondim (g=%.4f) for dycore consistency",
                 _g_nondim)

    model_cfg = ModelConfig(
        coords=coords,
        corrector_coords=corrector_coords,
        dt=_JAX_DT_NONDIM,
        dycore_substeps=_DYCORE_SUBSTEPS,
        physics_specs=physics_specs,
        reference_temperature=REFERENCE_TEMPERATURES,
        orography=orography_modal,
        corrector_orography=corrector_orography,
        cloud_keys=(
            "specific_cloud_liquid_water_content",
            "specific_cloud_ice_water_content",
        ),
        precision_policy=policy,
        # Dycore filter: JAX gin DYCORE_TAU='120 minutes', order=3, cutoff=0
        filter_attenuation=_dycore_filter_atten,
        filter_order=3,
        filter_cutoff=0.0,
        # Stability filter: JAX gin STABILITY_TAU='4 minutes', order=10, cutoff=0.4
        stability_filter_attenuation=_stability_filter_atten,
        stability_filter_order=10,
        stability_filter_cutoff=0.4,
    )
    model = NeuralGCMModel(model_cfg, parameterization, fixer)

    # Load weights if checkpoint provided
    summary = None
    if checkpoint_path is not None:
        summary = load_jax_weights(components, checkpoint_path, strict=False)
        logger.info("Loaded JAX weights: %s", summary)

    model = model.to(device)
    # Move pre-converted static tensors to GPU so hot-path .to(device=…)
    # calls are no-ops, eliminating CUDA-graph DeviceCopy partitions.
    coords.horizontal.to(device)
    if hasattr(coords.vertical, "to"):
        coords.vertical.to(device)
    # coords may be a different instance than model._dycore_pe.coords.
    # Migrate the high-precision Grid here too so that the returned
    # coords object is fully device-consistent.
    if hasattr(coords, "_horizontal_einsum_highest") and hasattr(
        coords._horizontal_einsum_highest, "to"
    ):
        coords._horizontal_einsum_highest.to(device)

    # Build resolution-aware config dict with training-essential keys.
    # dt_hours and peak_lr come from PAPER_RESOLUTIONS in exp_utils.
    _res_dt = {"2.8": 1.0, "1.4": 0.5, "0.7": 0.25}.get(resolution, 1.0)
    _res_peak_lr = {"2.8": 0.002, "1.4": 0.002, "0.7": 0.001}.get(resolution, 0.002)
    _res_total_steps = {"2.8": 38000, "1.4": 26000, "0.7": 25000}.get(resolution, 38000)
    config = {
        "resolution": resolution,
        "n_levels": N_SIGMA_LAYERS,
        "dt_hours": _res_dt,
        "grid": _spec["grid_name"],
        "grid_lon": _spec["nodal_lon"],
        "grid_lat": _spec["nodal_lat"],
        "max_wavenumber": _spec["max_wavenumber"],
        "latent_size": LATENT_SIZE,
        "num_blocks": NUM_BLOCKS,
        "peak_lr": _res_peak_lr,
        "total_steps": _res_total_steps,
        "hidden_size": 64,  # used by training scripts
    }
    return model, components, coords, config, summary
