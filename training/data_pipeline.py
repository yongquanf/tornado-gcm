# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Data pipeline: xarray I/O, sim_time, forcing strategies, regridding.

Implements:
  - inputs_from_xarray / forcings_from_xarray: xarray → model-ready conversion
  - sim_time computation: (t - t_ref) / T_scale nondimensionalization
  - Dynamic forcing: DynamicDataForcing, PersistenceDataForcing,
    AnomalyPersistence, Climatology
  - Regridder: bilinear horizontal interpolation
  - NearestNaNFiller: sphere-aware nearest-neighbor NaN filling
  - selective_temporal_shift: time-shift forcing variables
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# sim_time computation
# ═══════════════════════════════════════════════════════════════════════════


def datetime64_to_sim_time(
    times: np.ndarray,
    reference_datetime: np.datetime64,
    time_scale_hours: float = 1.0,
) -> np.ndarray:
    """Convert datetime64 array to nondimensional simulation time.

    sim_time = (t - t_ref) / (time_scale_hours * 3600s)

    Args:
        times: array of np.datetime64.
        reference_datetime: t_ref.
        time_scale_hours: T_scale in hours (default 1.0 → hours).

    Returns:
        1-D float64 array of nondimensional times.
    """
    delta = (times - reference_datetime).astype("timedelta64[s]").astype(np.float64)
    return delta / (time_scale_hours * 3600.0)


def sim_time_to_datetime64(
    sim_time: np.ndarray,
    reference_datetime: np.datetime64,
    time_scale_hours: float = 1.0,
) -> np.ndarray:
    """Convert nondimensional sim_time back to datetime64."""
    seconds = (sim_time * time_scale_hours * 3600.0).astype(np.int64)
    return reference_datetime + seconds.astype("timedelta64[s]")


def with_sim_time(
    data: dict[str, np.ndarray],
    times: np.ndarray,
    reference_datetime: np.datetime64 | None = None,
    time_scale_hours: float = 1.0,
) -> dict[str, np.ndarray]:
    """Add 'sim_time' key to data dict if not present.

    Args:
        data: dict with (T, ...) arrays.
        times: 1-D datetime64 time coordinate.
        reference_datetime: if None, uses times[0].
        time_scale_hours: nondimensionalization scale.

    Returns:
        data dict with 'sim_time' added (shape broadcastable to other fields).
    """
    if "sim_time" in data:
        return data
    if reference_datetime is None:
        reference_datetime = times[0]
    st = datetime64_to_sim_time(times, reference_datetime, time_scale_hours)
    data = dict(data)
    # Reshape for broadcasting: (T, 1, 1, 1) if 4D fields, else (T,)
    sample_key = next((k for k in data if k != "sim_time"), None)
    if sample_key is not None and data[sample_key].ndim > 1:
        shape = [len(st)] + [1] * (data[sample_key].ndim - 1)
        st = st.reshape(shape)
    data["sim_time"] = st
    return data


# ═══════════════════════════════════════════════════════════════════════════
# xarray ↔ model conversion
# ═══════════════════════════════════════════════════════════════════════════


def inputs_from_xarray(
    dataset,
    input_variables: Sequence[str] | None = None,
    reference_datetime=None,
    time_scale_hours: float = 1.0,
) -> dict[str, np.ndarray]:
    """Convert xarray Dataset to model input dict.

    Handles dimension reordering (time, level, lon, lat) and surface
    variable expansion (add singleton level dim).

    Args:
        dataset: xarray.Dataset with time, [level], longitude, latitude dims.
        input_variables: subset of variables to include (None = all).
        reference_datetime: for sim_time computation.
        time_scale_hours: sim_time scale.

    Returns:
        dict of variable_name → numpy array (T, [L], lon, lat).
    """
    import xarray as xr

    if input_variables is not None:
        dataset = dataset[list(input_variables)]

    result = {}
    for name in dataset.data_vars:
        var = dataset[name]
        dims = list(var.dims)

        # Reorder to (time, level, lon, lat) or (time, lon, lat)
        target_dims = []
        for d in ["time", "level", "longitude", "latitude"]:
            # Accept common aliases
            for alias in [d, d[:3], d.replace("itude", "")]:
                if alias in dims:
                    target_dims.append(alias)
                    break
        if target_dims:
            var = var.transpose(*target_dims)

        arr = var.values
        # Add singleton level dim for surface variables
        has_level = any(d in dims for d in ["level", "lev"])
        if not has_level and arr.ndim >= 2:
            arr = np.expand_dims(arr, axis=-3 if arr.ndim >= 3 else 1)

        result[name] = arr

    # Add sim_time
    if "time" in dataset.dims:
        times = dataset["time"].values
        result = with_sim_time(result, times, reference_datetime, time_scale_hours)

    # ── Validation ────────────────────────────────────────────────────
    if input_variables is not None:
        missing = set(input_variables) - set(result.keys()) - {"sim_time"}
        if missing:
            logger.warning(
                f"Missing variables in dataset: {sorted(missing)}. "
                f"Available: {sorted(dataset.data_vars)}"
            )

    for name, arr in result.items():
        if name == "sim_time":
            continue
        nan_count = np.isnan(arr).sum()
        if nan_count > 0:
            nan_pct = 100.0 * nan_count / arr.size
            logger.warning(
                f"Variable '{name}' contains {nan_count} NaN values "
                f"({nan_pct:.2f}% of {arr.size} elements)"
            )

    return result


def forcings_from_xarray(
    dataset,
    forcing_variables: Sequence[str] | None = None,
    reference_datetime=None,
    time_scale_hours: float = 1.0,
) -> dict[str, np.ndarray]:
    """Convert xarray Dataset to forcing dict (same logic as inputs)."""
    return inputs_from_xarray(
        dataset, forcing_variables, reference_datetime, time_scale_hours
    )


# ═══════════════════════════════════════════════════════════════════════════
# Dynamic forcing strategies
# ═══════════════════════════════════════════════════════════════════════════


class DynamicDataForcing:
    """Look up forcing data by sim_time with nearest-time interpolation.

    Stores (times, data) and returns the slice closest to the query time.
    Falls back to NaN if the closest time exceeds dt_tolerance.

    Args:
        times: 1-D sim_time array of available forcing times.
        data: dict of forcing_name → (T, ...) arrays.
        dt_tolerance: max allowed time mismatch (in sim_time units).
    """

    def __init__(
        self,
        times: np.ndarray,
        data: dict[str, np.ndarray],
        dt_tolerance: float = 1.0,
    ):
        self.times = np.asarray(times, dtype=np.float64).ravel()
        self.data = data
        self.dt_tolerance = dt_tolerance

    def __call__(self, sim_time: float) -> dict[str, torch.Tensor]:
        """Return forcing dict for the given sim_time."""
        idx = int(np.argmin(np.abs(self.times - sim_time)))
        dt = abs(self.times[idx] - sim_time)

        result = {}
        for key, arr in self.data.items():
            val = arr[idx]
            if dt > self.dt_tolerance:
                val = np.full_like(val, np.nan)
            result[key] = torch.from_numpy(val).float()
        return result


class PersistenceDataForcing:
    """Always return the first-timestep forcing (persistence forecast)."""

    def __init__(self, data: dict[str, np.ndarray]):
        self._forcing = {
            key: torch.from_numpy(arr[0]).float() for key, arr in data.items()
        }

    def __call__(self, sim_time: float) -> dict[str, torch.Tensor]:
        return dict(self._forcing)


class Climatology:
    """Look up forcing from a climatological dataset by day-of-year.

    Args:
        climatology: dict of variable → (365 or 366, ...) daily climatology arrays.
        times_dayofyear: 1-D int array of day-of-year (1-366) for each row.
    """

    def __init__(
        self,
        climatology: dict[str, np.ndarray],
        times_dayofyear: np.ndarray | None = None,
    ):
        self.climatology = climatology
        if times_dayofyear is None:
            first_key = next(iter(climatology))
            self.times_dayofyear = np.arange(1, climatology[first_key].shape[0] + 1)
        else:
            self.times_dayofyear = times_dayofyear

    def __call__(self, dayofyear: int) -> dict[str, torch.Tensor]:
        idx = int(np.argmin(np.abs(self.times_dayofyear - dayofyear)))
        return {
            key: torch.from_numpy(arr[idx]).float()
            for key, arr in self.climatology.items()
        }


class AnomalyPersistence:
    """Anomaly persistence forcing: anomaly(t0) + climatology(t).

    Computes the deviation from climatology at the initial time and
    adds it to the climatological state at each forecast time.
    Sea-ice cover is clipped to [0, 1].
    """

    def __init__(
        self,
        initial_data: dict[str, np.ndarray],
        climatology: Climatology,
        initial_dayofyear: int,
        clip_variables: Sequence[str] = ("sea_ice_cover",),
    ):
        clim_t0 = climatology(initial_dayofyear)
        self.anomaly = {}
        for key in initial_data:
            if key in clim_t0:
                self.anomaly[key] = (
                    torch.from_numpy(initial_data[key]).float() - clim_t0[key]
                )
        self.climatology = climatology
        self.clip_variables = set(clip_variables)

    def __call__(self, dayofyear: int) -> dict[str, torch.Tensor]:
        clim = self.climatology(dayofyear)
        result = {}
        for key, clim_val in clim.items():
            val = clim_val
            if key in self.anomaly:
                val = val + self.anomaly[key]
            if key in self.clip_variables:
                val = val.clamp(0.0, 1.0)
            result[key] = val
        return result


# ═══════════════════════════════════════════════════════════════════════════
# selective_temporal_shift
# ═══════════════════════════════════════════════════════════════════════════


def selective_temporal_shift(
    data: dict[str, np.ndarray],
    variables: Sequence[str],
    shift: int,
) -> dict[str, np.ndarray]:
    """Shift selected variables in time relative to the rest.

    Positive shift: selected variables use data from ``shift`` steps earlier.
    Negative shift: selected variables use data from ``|shift|`` steps later.
    The time axis is truncated to the overlap region.

    Args:
        data: dict of variable → (T, ...) arrays.
        variables: variable names to shift.
        shift: number of time steps to shift (positive = earlier).

    Returns:
        New data dict with adjusted arrays and truncated time dimension.
    """
    if shift == 0:
        return data

    result = {}
    sample_key = next(iter(data))
    T = data[sample_key].shape[0]

    if shift > 0:
        for key, arr in data.items():
            if key in variables:
                result[key] = arr[:T - shift]  # earlier data
            else:
                result[key] = arr[shift:]  # aligned by truncating front
    else:
        abs_shift = abs(shift)
        for key, arr in data.items():
            if key in variables:
                result[key] = arr[abs_shift:]  # later data
            else:
                result[key] = arr[:T - abs_shift]  # truncate back

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Regridder (bilinear)
# ═══════════════════════════════════════════════════════════════════════════


class BilinearRegridder:
    """Bilinear horizontal regridding from source to target lon/lat grid.

    Precomputes interpolation indices and weights for efficient regridding.

    Args:
        source_lon: 1-D source longitude array (radians or degrees).
        source_lat: 1-D source latitude array (radians or degrees).
        target_lon: 1-D target longitude array.
        target_lat: 1-D target latitude array.
        degrees: if True, convert to radians internally.
    """

    def __init__(
        self,
        source_lon: np.ndarray,
        source_lat: np.ndarray,
        target_lon: np.ndarray,
        target_lat: np.ndarray,
        degrees: bool = True,
    ):
        if degrees:
            source_lon = np.deg2rad(source_lon)
            source_lat = np.deg2rad(source_lat)
            target_lon = np.deg2rad(target_lon)
            target_lat = np.deg2rad(target_lat)

        self._lon_indices, self._lon_weights = self._interp_weights(
            source_lon, target_lon, periodic=True
        )
        self._lat_indices, self._lat_weights = self._interp_weights(
            source_lat, target_lat, periodic=False
        )

    @staticmethod
    def _interp_weights(
        source: np.ndarray, target: np.ndarray, periodic: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute linear interpolation indices and weights."""
        n = len(source)
        indices = np.searchsorted(source, target) - 1

        if periodic:
            indices = indices % n
            next_idx = (indices + 1) % n
            span = np.where(
                next_idx > indices,
                source[next_idx] - source[indices],
                source[next_idx] + 2 * np.pi - source[indices],
            )
            frac = np.where(
                target >= source[indices],
                target - source[indices],
                target + 2 * np.pi - source[indices],
            )
        else:
            indices = np.clip(indices, 0, n - 2)
            next_idx = indices + 1
            span = source[next_idx] - source[indices]
            frac = target - source[indices]

        weights = np.where(span > 0, frac / span, 0.0)
        return np.stack([indices, next_idx], axis=-1), weights

    def __call__(self, field: np.ndarray) -> np.ndarray:
        """Regrid field with shape (..., lon_src, lat_src).

        Returns: (..., lon_tgt, lat_tgt).
        """
        # Latitude interpolation first
        lat_idx = self._lat_indices  # (lat_tgt, 2)
        lat_w = self._lat_weights  # (lat_tgt,)
        f_lat = (
            (1 - lat_w)[np.newaxis, :] * field[..., lat_idx[:, 0]]
            + lat_w[np.newaxis, :] * field[..., lat_idx[:, 1]]
        )  # (..., lon_src, lat_tgt)

        # Longitude interpolation
        lon_idx = self._lon_indices  # (lon_tgt, 2)
        lon_w = self._lon_weights  # (lon_tgt,)
        result = (
            (1 - lon_w)[:, np.newaxis] * f_lat[..., lon_idx[:, 0], :]
            + lon_w[:, np.newaxis] * f_lat[..., lon_idx[:, 1], :]
        )
        return result


# ═══════════════════════════════════════════════════════════════════════════
# NaN filling (nearest-neighbor on sphere)
# ═══════════════════════════════════════════════════════════════════════════


class NearestNaNFiller:
    """Fill NaN values using nearest-neighbor interpolation on the sphere.

    Uses haversine distance to find the closest non-NaN point.

    Args:
        lon: 1-D longitude in degrees.
        lat: 1-D latitude in degrees.
    """

    def __init__(self, lon: np.ndarray, lat: np.ndarray):
        self.lon_rad = np.deg2rad(lon)
        self.lat_rad = np.deg2rad(lat)
        self._lon_grid, self._lat_grid = np.meshgrid(
            self.lon_rad, self.lat_rad, indexing="ij"
        )
        self._coords = np.stack(
            [self._lat_grid.ravel(), self._lon_grid.ravel()], axis=-1
        )

    def fill(self, field: np.ndarray) -> np.ndarray:
        """Fill NaN values in field (..., lon, lat).

        For each spatial slice, finds NaN locations and replaces with
        the value of the nearest non-NaN point (haversine distance).
        """
        shape = field.shape
        spatial = field.reshape(-1, shape[-2], shape[-1])
        result = spatial.copy()

        for i in range(spatial.shape[0]):
            slab = spatial[i]  # (lon, lat)
            flat = slab.ravel()
            nan_mask = np.isnan(flat)
            if not nan_mask.any():
                continue
            if nan_mask.all():
                continue  # nothing to fill from

            valid_coords = self._coords[~nan_mask]
            nan_coords = self._coords[nan_mask]
            valid_values = flat[~nan_mask]

            # Haversine distance to all valid points
            dlat = nan_coords[:, 0:1] - valid_coords[:, 0:1].T
            dlon = nan_coords[:, 1:2] - valid_coords[:, 1:2].T
            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(nan_coords[:, 0:1])
                * np.cos(valid_coords[:, 0:1].T)
                * np.sin(dlon / 2) ** 2
            )
            dist = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            nearest_idx = np.argmin(dist, axis=1)

            filled = flat.copy()
            filled[nan_mask] = valid_values[nearest_idx]
            result[i] = filled.reshape(shape[-2], shape[-1])

        return result.reshape(shape)
