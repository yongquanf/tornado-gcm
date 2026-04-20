# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Nearest-time forcing selection — PyTorch analogue of ``DynamicDataForcing``.

``neuralgcm.legacy.forcings.DynamicDataForcing`` (after nondimensionalize) picks
the forcing row whose ``sim_time`` is closest to the requested time, using
``jnp.interp`` + ``round`` along ``time_axis=0``.

This module reproduces that index rule on NumPy buffers, then returns
``torch.float32`` tensors on the requested device for ``NeuralGCMModel.step``.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
import torch

ForcingFn = Callable[[dict[str, Any], float], dict[str, torch.Tensor]]


def _time_axis_length(arr: np.ndarray) -> int:
    return int(arr.shape[0])


def _select_time0(arr: np.ndarray, idx: int) -> np.ndarray:
    """Take index ``idx`` along leading (time) axis, dropping that axis."""
    if arr.ndim == 0:
        return arr
    return np.take(arr, idx, axis=0)


def build_dynamic_data_forcing_torch(
    forcing_data_nondim: Mapping[str, Any],
    *,
    dt_tolerance: float,
    device: torch.device | str,
) -> ForcingFn:
    """Build ``forcing_fn(forcing_batch, sim_time)`` matching JAX rounding rule.

    Args:
        forcing_data_nondim: nondimensional forcing dict (NumPy or array-like),
            including ``sim_time`` with leading time dimension ``T >= 1``.
        dt_tolerance: nondim tolerance (same units as ``sim_time``); if the
            selected row's ``sim_time`` differs from ``sim_time`` by more than
            this, values are replaced with NaN (JAX behaviour).
        device: torch device for outputs.
    """
    data = {k: np.asarray(v) for k, v in forcing_data_nondim.items()}
    if "sim_time" not in data:
        raise KeyError("forcing_data must contain 'sim_time'")
    times = np.squeeze(np.asarray(data["sim_time"]))
    if times.ndim == 0:
        times = times.reshape(1)
    times_1d = times.reshape(-1)
    t_len = times_1d.shape[0]
    for k, v in data.items():
        if k == "sim_time":
            continue
        if v.shape[0] != t_len:
            raise ValueError(
                f"Leading time length mismatch for {k}: "
                f"expected {t_len} from sim_time, got {v.shape[0]}"
            )

    dev = torch.device(device) if isinstance(device, str) else device

    def forcing_fn(_forcing_in: dict[str, Any], sim_time: float | torch.Tensor) -> dict[str, torch.Tensor]:
        del _forcing_in  # use captured ``data`` (same as JAX: caller passes full series)
        st = float(sim_time.item()) if torch.is_tensor(sim_time) else float(sim_time)
        if t_len == 1:
            idx = 0
        else:
            approx = float(np.interp(st, times_1d, np.arange(t_len, dtype=np.float64)))
            idx = int(int(np.round(approx)))
            idx = max(0, min(t_len - 1, idx))
        row_sim = float(np.asarray(_select_time0(data["sim_time"], idx)).reshape(-1)[0])
        valid = abs(row_sim - st) <= float(dt_tolerance)
        out: dict[str, torch.Tensor] = {}
        for k, v in data.items():
            sl = _select_time0(v, idx)
            t = torch.from_numpy(np.array(sl, copy=True)).to(torch.float32).to(dev)
            if not valid:
                t = torch.full_like(t, float("nan"))
            out[k] = t
        return out

    return forcing_fn
