# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Evaluation runner for NeuralGCM PyTorch training.

Implements:
  - get_forecast_starts: equispaced start times with 0z/12z alternation (M2)
  - EvalMetrics: per-variable, per-lead-time RMSE and ACC
  - EvaluationRunner: in-training evaluation loop (M1)

References JAX:
  - experiment.py: _get_datetime_forecast_starts + run_evaluation
  - experimental/training/data_loading.py: _get_datetime_forecast_starts (diurnal)
  - metrics.py: TransformedL2Loss, RMSE, SpatialBiasRMSE
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _model_device(model: nn.Module) -> torch.device:
    """Infer the active device for a model or wrapped model."""
    raw = model
    while hasattr(raw, "module"):
        raw = raw.module
    try:
        return next(raw.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _move_state_to_device(state, device: torch.device):
    """Move a State-like tree to the given device if needed."""
    if hasattr(state, "tree_map"):
        return state.tree_map(lambda t: t.to(device))
    return state


def _move_forcing_to_device(forcing, device: torch.device):
    """Move nested forcing tensors/arrays to the given device."""
    if forcing is None:
        return None
    moved = {}
    for key, value in forcing.items():
        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            tensor = torch.as_tensor(value)
        moved[key] = tensor.to(device=device, dtype=torch.float32)
    return moved


# ═══════════════════════════════════════════════════════════════════════════
# M2: Forecast start time sampling
# ═══════════════════════════════════════════════════════════════════════════


def get_forecast_starts_equispaced(
    sample_count: int,
    first_start: np.datetime64,
    last_start: np.datetime64,
    alternate_0z_12z: bool = True,
) -> np.ndarray:
    """Get equispaced forecast start times, matching JAX reference.

    Produces ``sample_count`` evenly-distributed times from ``first_start``
    to ``last_start``, rounded to midnight, with alternating 0z/12z init
    to match ECMWF operational practice.

    Args:
        sample_count: number of forecast start times.
        first_start: earliest start (should be at 00:00 UTC).
        last_start: latest start.
        alternate_0z_12z: if True, alternate between 0z and 12z init.

    Returns:
        np.ndarray of datetime64 start times, shape (sample_count,).
    """
    if sample_count <= 0:
        return np.array([], dtype="datetime64[ns]")

    # Round up to next midnight
    one_day = np.timedelta64(1, "D")
    stop = np.datetime64(last_start, "D") + one_day

    # Equispaced from first_start (inclusive) to stop (exclusive)
    start = np.datetime64(first_start, "ns")
    stop_ns = np.datetime64(stop, "ns")
    total_ns = (stop_ns - start).astype(np.int64)
    offsets = np.linspace(0, total_ns, sample_count + 1)[:-1].astype(np.int64)
    start_times = start + offsets.astype("timedelta64[ns]")

    # Round to nearest midnight
    start_times = start_times.astype("datetime64[D]").astype("datetime64[ns]")

    if alternate_0z_12z:
        twelve_hours = np.timedelta64(12, "h").astype("timedelta64[ns]")
        parity = (np.arange(sample_count) % 2).astype(np.int64)
        start_times = start_times + parity * twelve_hours

    return start_times


def get_forecast_starts_balanced(
    sample_count: int,
    candidates: np.ndarray,
    balance_diurnal_cycle: bool = True,
) -> np.ndarray:
    """Get forecast start times balanced across diurnal hours.

    Matches JAX experimental ``_get_datetime_forecast_starts`` with
    ``balance_diurnal_cycle=True``.

    Args:
        sample_count: number of start times to select.
        candidates: array of datetime64 (available data times).
        balance_diurnal_cycle: distribute samples evenly across hours of day.

    Returns:
        np.ndarray of datetime64 start times, sorted.
    """
    if sample_count <= 0:
        return np.array([], dtype="datetime64[ns]")

    candidates = candidates.astype("datetime64[ns]")

    if not balance_diurnal_cycle:
        indices = np.linspace(0, len(candidates) - 1, sample_count).astype(int)
        return candidates[indices]

    # Group by hour-of-day offset
    days = candidates.astype("datetime64[D]").astype("datetime64[ns]")
    offsets = candidates - days  # timedelta from midnight
    unique_offsets = np.unique(offsets)
    n_offsets = len(unique_offsets)

    if n_offsets == 0:
        raise ValueError("No candidates available.")

    # Distribute sample_count across unique offsets
    counts = [
        sample_count // n_offsets + (1 if i < sample_count % n_offsets else 0)
        for i in range(n_offsets)
    ]

    starts = []
    for offset, count in zip(unique_offsets, counts):
        if count == 0:
            continue
        group = candidates[offsets == offset]
        if count > len(group):
            raise ValueError(
                f"Offset {offset} underrepresented: need {count}, have {len(group)}"
            )
        stride = len(group) / count
        idx = np.round(np.arange(count) * stride).astype(int)
        idx = np.clip(idx, 0, len(group) - 1)
        starts.append(group[idx])

    result = np.sort(np.concatenate(starts))
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Metric computation
# ═══════════════════════════════════════════════════════════════════════════


def _state_to_tensor_dict(state) -> dict[str, torch.Tensor]:
    """Convert a State to a flat dict of named tensors."""
    from tornado_gcm.core.primitive_equations import State

    if isinstance(state, State):
        d = {
            "vorticity": state.vorticity,
            "divergence": state.divergence,
            "temperature_variation": state.temperature_variation,
            "log_surface_pressure": state.log_surface_pressure,
        }
        for k, v in state.tracers.items():
            d[f"tracer_{k}"] = v
        return d
    elif isinstance(state, dict):
        return state
    elif isinstance(state, torch.Tensor):
        return {"field": state}
    else:
        raise TypeError(f"Unsupported state type: {type(state)}")


@torch.no_grad()
def compute_rmse(
    predictions: list,
    targets: list,
    variable_names: Sequence[str] | None = None,
) -> dict[str, list[float]]:
    """Compute per-variable, per-lead-time RMSE.

    Args:
        predictions: list of K predicted states.
        targets: list of K target states.
        variable_names: optional filter to specific variables.

    Returns:
        dict mapping variable name → list of RMSE values (one per lead time).
    """
    result: dict[str, list[float]] = {}
    K = min(len(predictions), len(targets))

    for k in range(K):
        pred_d = _state_to_tensor_dict(predictions[k])
        targ_d = _state_to_tensor_dict(targets[k])

        for name in pred_d:
            if variable_names and name not in variable_names:
                continue
            if name not in targ_d:
                continue

            p, t = pred_d[name], targ_d[name]
            if torch.is_complex(p):
                # For spectral coefficients, compute in physical meaning
                diff = p - t
                mse = (diff * diff.conj()).real.mean().item()
            else:
                mse = ((p - t) ** 2).float().mean().item()

            rmse = math.sqrt(max(mse, 0.0))
            result.setdefault(name, []).append(rmse)

    return result


@torch.no_grad()
def compute_anomaly_correlation(
    predictions: list,
    targets: list,
    climatology: dict[str, torch.Tensor] | None = None,
    variable_names: Sequence[str] | None = None,
) -> dict[str, list[float]]:
    """Compute per-variable, per-lead-time Anomaly Correlation Coefficient.

    ACC = corr(pred - clim, target - clim), averaged over spatial dims.
    When climatology is None, uses the target mean as a proxy.

    Args:
        predictions: list of K predicted states.
        targets: list of K target states.
        climatology: optional dict of climatological means.
        variable_names: optional filter.

    Returns:
        dict mapping variable name → list of ACC values per lead time.
    """
    result: dict[str, list[float]] = {}
    K = min(len(predictions), len(targets))

    for k in range(K):
        pred_d = _state_to_tensor_dict(predictions[k])
        targ_d = _state_to_tensor_dict(targets[k])

        for name in pred_d:
            if variable_names and name not in variable_names:
                continue
            if name not in targ_d:
                continue

            p = pred_d[name].float()
            t = targ_d[name].float()

            if torch.is_complex(p):
                # Skip ACC for complex spectral data
                continue

            # Anomaly: subtract climatology or target mean
            if climatology and name in climatology:
                clim = climatology[name].float().to(p.device)
            else:
                clim = t.mean(dim=tuple(range(t.ndim)), keepdim=True)

            p_anom = p - clim
            t_anom = t - clim

            # Pearson correlation over all spatial dims
            p_flat = p_anom.reshape(-1)
            t_flat = t_anom.reshape(-1)

            cov = (p_flat * t_flat).sum()
            p_std = p_flat.norm()
            t_std = t_flat.norm()

            denom = p_std * t_std
            acc = (cov / denom).item() if denom > 0 else 0.0
            acc = max(-1.0, min(1.0, acc))  # clamp

            result.setdefault(name, []).append(acc)

    return result


@torch.no_grad()
def compute_bias(
    predictions: list,
    targets: list,
    variable_names: Sequence[str] | None = None,
) -> dict[str, list[float]]:
    """Compute per-variable, per-lead-time spatial mean bias.

    Returns:
        dict mapping variable name → list of bias values per lead time.
    """
    result: dict[str, list[float]] = {}
    K = min(len(predictions), len(targets))

    for k in range(K):
        pred_d = _state_to_tensor_dict(predictions[k])
        targ_d = _state_to_tensor_dict(targets[k])

        for name in pred_d:
            if variable_names and name not in variable_names:
                continue
            if name not in targ_d:
                continue

            p, t = pred_d[name], targ_d[name]
            if torch.is_complex(p):
                bias = (p - t).abs().mean().item()
            else:
                bias = (p - t).float().mean().item()

            result.setdefault(name, []).append(bias)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# EvalResult container
# ═══════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class EvalResult:
    """Container for evaluation metrics.

    Attributes:
        rmse: per-variable, per-lead-time RMSE values.
        acc: per-variable, per-lead-time ACC values (may be empty).
        bias: per-variable, per-lead-time mean bias.
        eval_loss: scalar evaluation loss (if loss_fn provided).
        n_samples: number of forecast samples evaluated.
        elapsed_seconds: wall time for evaluation.
    """

    rmse: dict[str, list[float]] = dataclasses.field(default_factory=dict)
    acc: dict[str, list[float]] = dataclasses.field(default_factory=dict)
    bias: dict[str, list[float]] = dataclasses.field(default_factory=dict)
    eval_loss: float = 0.0
    n_samples: int = 0
    elapsed_seconds: float = 0.0

    def summary(self, variables: Sequence[str] | None = None) -> dict[str, float]:
        """Flatten to a scalar-valued dict for logging.

        Returns keys like "rmse/temperature_variation/t+3",
        "acc/vorticity/t+1", "bias/divergence/t+5".
        """
        out: dict[str, float] = {}
        if self.eval_loss > 0:
            out["eval_loss"] = self.eval_loss
        for metric_name, metric_dict in [
            ("rmse", self.rmse),
            ("acc", self.acc),
            ("bias", self.bias),
        ]:
            for var, values in metric_dict.items():
                if variables and var not in variables:
                    continue
                for t, v in enumerate(values):
                    out[f"{metric_name}/{var}/t+{t + 1}"] = v
                # Also report mean over lead times
                if values:
                    out[f"{metric_name}/{var}/mean"] = sum(values) / len(values)
        out["n_samples"] = float(self.n_samples)
        out["eval_seconds"] = self.elapsed_seconds
        return out

    def log_summary(
        self,
        step: int,
        prefix: str = "eval",
        variables: Sequence[str] | None = None,
        max_lead_times: int = 4,
    ) -> None:
        """Log a concise summary to the logger."""
        parts = [f"step={step}"]
        if self.eval_loss > 0:
            parts.append(f"loss={self.eval_loss:.6f}")

        # Report first N lead times for each variable
        for var, values in self.rmse.items():
            if variables and var not in variables:
                continue
            shown = values[:max_lead_times]
            rmse_str = ", ".join(f"{v:.4f}" for v in shown)
            parts.append(f"rmse/{var}=[{rmse_str}]")

        for var, values in self.acc.items():
            if variables and var not in variables:
                continue
            shown = values[:max_lead_times]
            acc_str = ", ".join(f"{v:.4f}" for v in shown)
            parts.append(f"acc/{var}=[{acc_str}]")

        logger.info(f"[{prefix}] {' | '.join(parts)}")


# ═══════════════════════════════════════════════════════════════════════════
# M1: EvaluationRunner
# ═══════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class EvalConfig:
    """Configuration for periodic evaluation.

    Attributes:
        eval_cadence: evaluate every N training steps.
        n_eval_samples: number of forecast start times to evaluate.
        rollout_steps: evaluation rollout length K.
        compute_acc: whether to compute anomaly correlation.
        variable_names: optional filter to specific variables.
    """

    eval_cadence: int = 1000
    n_eval_samples: int = 20
    rollout_steps: int | None = None  # None = use training K
    compute_acc: bool = True
    variable_names: Sequence[str] | None = None


class EvaluationRunner:
    """In-training evaluation loop with streaming metrics.

    Runs autoregressive forecasts from equispaced start times,
    computes RMSE and ACC per variable per lead time, and
    optionally evaluates a loss function.

    Matches JAX ``experiment.run_evaluation()``:
      - Equispaced start time sampling (0z/12z alternation)
      - Per-cadence triggering
      - Streaming metric aggregation (no need to hold all forecasts)
      - Supports both regular params and EMA params

    Usage in training loop::

        eval_runner = EvaluationRunner(model, config=EvalConfig(eval_cadence=500))
        eval_runner.set_eval_data(eval_states, eval_forcings)

        for step in range(total_steps):
            # ... train step ...
            result = eval_runner.maybe_evaluate(step, rollout_steps=K)
            if result is not None:
                result.log_summary(step)

    Args:
        model: the model (nn.Module with step() or forward()).
        config: EvalConfig controlling frequency and scope.
        loss_fn: optional loss module (same interface as training loss).
        ema_params: optional EMA parameter dict for separate evaluation.
        climatology: optional climatological means for ACC.
    """

    def __init__(
        self,
        model: nn.Module,
        config: EvalConfig | None = None,
        loss_fn: nn.Module | None = None,
        ema_params: dict[str, torch.Tensor] | None = None,
        climatology: dict[str, torch.Tensor] | None = None,
    ):
        self.model = model
        self.config = config or EvalConfig()
        self.loss_fn = loss_fn
        self.ema_params = ema_params
        self.climatology = climatology

        # Eval data: list of (initial_state, target_states, forcings)
        self._eval_samples: list[tuple[Any, list, list | None]] = []

    def set_eval_data(
        self,
        states: list,
        forcings: list[dict[str, torch.Tensor]] | dict[str, np.ndarray] | None = None,
        rollout_steps: int | None = None,
    ) -> None:
        """Set evaluation data from a list of sequential states.

        Samples ``n_eval_samples`` equispaced starting points from the
        state sequence and builds (init, targets, forcings) tuples.

        Args:
            states: sequential list of states (length >= rollout_steps + 1).
            forcings: optional per-timestep forcing dicts (list) or a
                forcing_dict mapping variable names to time-indexed arrays.
            rollout_steps: evaluation rollout K (overrides config).
        """
        K = rollout_steps or self.config.rollout_steps or 16
        n = self.config.n_eval_samples
        total = len(states) - K
        device = _model_device(self.model)

        if total <= 0:
            logger.warning(
                f"Not enough states for evaluation: {len(states)} states, "
                f"K={K}. Need at least {K + 1}."
            )
            return

        # Detect forcing_dict (dict[str, ndarray]) vs list-of-dicts
        is_forcing_dict = (
            forcings is not None
            and isinstance(forcings, dict)
            and all(isinstance(v, np.ndarray) for v in forcings.values())
        )

        # Equispaced indices
        if n >= total:
            indices = list(range(total))
        else:
            indices = np.linspace(0, total - 1, n).astype(int).tolist()

        self._eval_samples = []
        for idx in indices:
            init = _move_state_to_device(states[idx], device)
            targets = [
                _move_state_to_device(s, device)
                for s in states[idx + 1 : idx + 1 + K]
            ]
            frc = None
            if forcings is not None:
                if is_forcing_dict:
                    # Build per-step forcing dicts from arrays
                    frc = []
                    for t_idx in range(idx + 1, idx + 1 + K):
                        step_forcing = {}
                        for k, arr in forcings.items():
                            t_clamp = min(t_idx, arr.shape[0] - 1)
                            step_forcing[k] = torch.as_tensor(
                                arr[t_clamp], dtype=torch.float32, device=device
                            )
                        frc.append(step_forcing)
                else:
                    frc = [
                        _move_forcing_to_device(f, device)
                        for f in forcings[idx + 1 : idx + 1 + K]
                    ]
            self._eval_samples.append((init, targets, frc))

        logger.info(
            f"Evaluation data: {len(self._eval_samples)} samples, K={K}"
        )

    def set_eval_data_from_dataloader(
        self,
        dataloader,
        n_samples: int | None = None,
        rollout_steps: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Set evaluation data from a DataLoader.

        Extracts the first ``n_samples`` batches from the dataloader.
        """
        from tornado_gcm.scripts.data_preprocessing import batch_to_train_pair

        n = n_samples or self.config.n_eval_samples
        K = rollout_steps or self.config.rollout_steps or 16
        self._eval_samples = []

        for i, batch_dict in enumerate(dataloader):
            if i >= n:
                break
            initial, targets = batch_to_train_pair(
                batch_dict, K,
                tracer_keys=("specific_humidity",),
                batch_idx=0,
            )
            if device is not None:
                initial = initial.tree_map(lambda t: t.to(device))
                targets = [s.tree_map(lambda t: t.to(device)) for s in targets]

            # Extract forcings
            forcing_keys = [k for k in batch_dict if k.startswith("forcing_")]
            frc_list = None
            if forcing_keys:
                frc_list = []
                for t_idx in range(1, K + 1):
                    step_forcing = {}
                    for k in forcing_keys:
                        name = k[len("forcing_"):]
                        val = batch_dict[k][0, t_idx]
                        if device is not None:
                            val = val.to(device)
                        step_forcing[name] = val
                    frc_list.append(step_forcing)

            self._eval_samples.append((initial, targets, frc_list))

        logger.info(
            f"Evaluation data from DataLoader: {len(self._eval_samples)} samples"
        )

    def should_evaluate(self, step: int) -> bool:
        """Check if evaluation should run at this step."""
        if not self._eval_samples:
            return False
        return (step + 1) % self.config.eval_cadence == 0

    def maybe_evaluate(
        self,
        step: int,
        rollout_steps: int | None = None,
        use_ema: bool = False,
    ) -> EvalResult | None:
        """Evaluate if cadence triggers; return None otherwise.

        Args:
            step: current training step.
            rollout_steps: override rollout K.
            use_ema: if True and ema_params set, evaluate with EMA weights.

        Returns:
            EvalResult if evaluation ran, None if skipped.
        """
        if not self.should_evaluate(step):
            return None
        return self.evaluate(rollout_steps=rollout_steps, use_ema=use_ema)

    @torch.no_grad()
    def evaluate(
        self,
        rollout_steps: int | None = None,
        use_ema: bool = False,
    ) -> EvalResult:
        """Run full evaluation over all stored samples.

        Uses streaming aggregation: metrics accumulated sample by sample.
        No need to hold all forecasts in memory.

        Args:
            rollout_steps: override rollout K.
            use_ema: evaluate with EMA parameters.

        Returns:
            EvalResult with aggregated metrics.
        """
        t0 = time.time()
        self.model.eval()

        # Optionally swap to EMA parameters
        original_params = None
        if use_ema and self.ema_params:
            original_params = {}
            for name, param in self.model.named_parameters():
                original_params[name] = param.data.clone()
                if name in self.ema_params:
                    param.data.copy_(self.ema_params[name])

        # Streaming metric accumulators
        all_rmse: dict[str, list[list[float]]] = {}
        all_acc: dict[str, list[list[float]]] = {}
        all_bias: dict[str, list[list[float]]] = {}
        total_loss = 0.0
        n_loss = 0

        try:
            device = _model_device(self.model)
            for init_state, targets, forcings in self._eval_samples:
                init_state = _move_state_to_device(init_state, device)
                targets = [_move_state_to_device(s, device) for s in targets]
                forcings = (
                    [_move_forcing_to_device(f, device) for f in forcings]
                    if forcings is not None else None
                )
                K = rollout_steps or len(targets)
                K = min(K, len(targets))

                # Autoregressive rollout
                predictions = []
                state = init_state
                for i in range(K):
                    frc = forcings[i] if forcings is not None and i < len(forcings) else None
                    state = self.model.step(state, forcings=frc)
                    predictions.append(state)

                # Compute metrics
                rmse = compute_rmse(
                    predictions, targets[:K],
                    variable_names=self.config.variable_names,
                )
                for var, vals in rmse.items():
                    all_rmse.setdefault(var, []).append(vals)

                if self.config.compute_acc:
                    acc = compute_anomaly_correlation(
                        predictions, targets[:K],
                        climatology=self.climatology,
                        variable_names=self.config.variable_names,
                    )
                    for var, vals in acc.items():
                        all_acc.setdefault(var, []).append(vals)

                bias = compute_bias(
                    predictions, targets[:K],
                    variable_names=self.config.variable_names,
                )
                for var, vals in bias.items():
                    all_bias.setdefault(var, []).append(vals)

                # Optional loss
                if self.loss_fn is not None:
                    loss_dict = self.loss_fn(
                        predictions=predictions,
                        targets=targets[:K],
                        step=0,
                    )
                    if isinstance(loss_dict, dict) and "total" in loss_dict:
                        total_loss += loss_dict["total"].item()
                    elif isinstance(loss_dict, torch.Tensor):
                        total_loss += loss_dict.item()
                    n_loss += 1

        finally:
            # Restore original parameters
            if original_params is not None:
                for name, param in self.model.named_parameters():
                    if name in original_params:
                        param.data.copy_(original_params[name])
            self.model.train()

        # Aggregate: average across samples per lead time
        def _aggregate(metric_dict):
            agg = {}
            for var, sample_lists in metric_dict.items():
                # sample_lists: list of lists, each inner list has K values
                max_k = max(len(s) for s in sample_lists) if sample_lists else 0
                means = []
                for t in range(max_k):
                    vals = [s[t] for s in sample_lists if t < len(s)]
                    means.append(sum(vals) / len(vals) if vals else 0.0)
                agg[var] = means
            return agg

        result = EvalResult(
            rmse=_aggregate(all_rmse),
            acc=_aggregate(all_acc),
            bias=_aggregate(all_bias),
            eval_loss=total_loss / max(n_loss, 1),
            n_samples=len(self._eval_samples),
            elapsed_seconds=time.time() - t0,
        )

        return result

    def update_ema_params(self, ema_params: dict[str, torch.Tensor]) -> None:
        """Update stored EMA parameters (call from trainer after each step)."""
        self.ema_params = ema_params


# ═══════════════════════════════════════════════════════════════════════════
# Weather Benchmark Evaluation (Day-3/Day-5 standard metrics)
# ═══════════════════════════════════════════════════════════════════════════

# Standard lead times for weather forecast evaluation
LEAD_HOURS_DAY3 = 72
LEAD_HOURS_DAY5 = 120
LEAD_HOURS_DAY10 = 240

# WeatherBench2 standard variables and thresholds
WB2_HEADLINE_VARS = [
    "geopotential_500",    # Z500 geopotential height at 500 hPa
    "temperature_850",     # T850 temperature at 850 hPa
    "temperature_500",     # T500 temperature at 500 hPa
    "specific_humidity_700",
]

# ACC thresholds for "useful" forecast skill
ACC_USEFUL_THRESHOLD = 0.6    # Day-3 typically > 0.6
ACC_MARGINAL_THRESHOLD = 0.5  # Day-5 typically > 0.5


@dataclasses.dataclass
class WeatherBenchmarkResult:
    """Standard weather forecast verification metrics.

    Attributes:
        day3_rmse: per-variable RMSE at Day-3 (72h) lead time.
        day5_rmse: per-variable RMSE at Day-5 (120h) lead time.
        day10_rmse: per-variable RMSE at Day-10 (240h) lead time.
        day3_acc: per-variable ACC at Day-3.
        day5_acc: per-variable ACC at Day-5.
        rmse_by_lead: per-variable RMSE for all lead times dict[var, list[float]].
        acc_by_lead: per-variable ACC for all lead times dict[var, list[float]].
        n_samples: number of forecast cases evaluated.
        dt_hours: model output time step in hours.
    """
    day3_rmse: dict[str, float] = dataclasses.field(default_factory=dict)
    day5_rmse: dict[str, float] = dataclasses.field(default_factory=dict)
    day10_rmse: dict[str, float] = dataclasses.field(default_factory=dict)
    day3_acc: dict[str, float] = dataclasses.field(default_factory=dict)
    day5_acc: dict[str, float] = dataclasses.field(default_factory=dict)
    rmse_by_lead: dict[str, list[float]] = dataclasses.field(default_factory=dict)
    acc_by_lead: dict[str, list[float]] = dataclasses.field(default_factory=dict)
    n_samples: int = 0
    dt_hours: float = 6.0

    def headline_summary(self) -> dict[str, float]:
        """Return headline metrics for logging / paper tables."""
        out: dict[str, float] = {}
        for var, val in self.day3_rmse.items():
            out[f"day3_rmse/{var}"] = val
        for var, val in self.day5_rmse.items():
            out[f"day5_rmse/{var}"] = val
        for var, val in self.day3_acc.items():
            out[f"day3_acc/{var}"] = val
        for var, val in self.day5_acc.items():
            out[f"day5_acc/{var}"] = val
        out["n_samples"] = float(self.n_samples)
        return out

    def log_headline(self, step: int, prefix: str = "benchmark") -> None:
        """Log headline Day-3/Day-5 metrics."""
        parts = [f"step={step}"]
        for var in sorted(self.day3_rmse):
            r3 = self.day3_rmse.get(var, float("nan"))
            r5 = self.day5_rmse.get(var, float("nan"))
            parts.append(f"{var}: Day3={r3:.2f} Day5={r5:.2f}")
        for var in sorted(self.day3_acc):
            a3 = self.day3_acc.get(var, float("nan"))
            a5 = self.day5_acc.get(var, float("nan"))
            parts.append(f"ACC/{var}: Day3={a3:.4f} Day5={a5:.4f}")
        logger.info(f"[{prefix}] {' | '.join(parts)}")


def _lead_time_index(target_hours: float, dt_hours: float) -> int | None:
    """Return the 0-based index for a given lead time, or None if unavailable."""
    idx = round(target_hours / dt_hours) - 1  # 0-indexed
    return idx if idx >= 0 else None


def extract_day_metrics(
    eval_result: EvalResult,
    dt_hours: float = 6.0,
) -> WeatherBenchmarkResult:
    """Extract Day-3/Day-5/Day-10 metrics from a full EvalResult.

    Args:
        eval_result: result from EvaluationRunner.evaluate().
        dt_hours: output time step in hours (6h for standard NeuralGCM).

    Returns:
        WeatherBenchmarkResult with headline metrics extracted.
    """
    idx3 = _lead_time_index(LEAD_HOURS_DAY3, dt_hours)
    idx5 = _lead_time_index(LEAD_HOURS_DAY5, dt_hours)
    idx10 = _lead_time_index(LEAD_HOURS_DAY10, dt_hours)

    result = WeatherBenchmarkResult(
        rmse_by_lead=dict(eval_result.rmse),
        acc_by_lead=dict(eval_result.acc),
        n_samples=eval_result.n_samples,
        dt_hours=dt_hours,
    )

    for var, vals in eval_result.rmse.items():
        if idx3 is not None and idx3 < len(vals):
            result.day3_rmse[var] = vals[idx3]
        if idx5 is not None and idx5 < len(vals):
            result.day5_rmse[var] = vals[idx5]
        if idx10 is not None and idx10 < len(vals):
            result.day10_rmse[var] = vals[idx10]

    for var, vals in eval_result.acc.items():
        if idx3 is not None and idx3 < len(vals):
            result.day3_acc[var] = vals[idx3]
        if idx5 is not None and idx5 < len(vals):
            result.day5_acc[var] = vals[idx5]

    return result


@dataclasses.dataclass
class WeatherBenchmarkConfig:
    """Configuration for standard weather benchmark evaluation.

    Attributes:
        eval_cadence: evaluate every N training steps.
        n_eval_samples: number of forecast cases.
        forecast_hours: total forecast length in hours.
        dt_hours: model output time step in hours.
        compute_acc: whether to compute ACC.
        headline_vars: variable names to report in headlines.
    """
    eval_cadence: int = 5000
    n_eval_samples: int = 50
    forecast_hours: float = 360.0  # 15 days
    dt_hours: float = 6.0
    compute_acc: bool = True
    headline_vars: Sequence[str] | None = None


class WeatherBenchmarkRunner:
    """Standard weather forecast benchmark evaluation.

    Wraps EvaluationRunner with Day-3/Day-5 metric extraction and
    standard WeatherBench2 reporting conventions.

    Usage::

        benchmark = WeatherBenchmarkRunner(model, dt_hours=6.0)
        benchmark.set_eval_data(eval_states, eval_forcings)

        # During training:
        result = benchmark.maybe_evaluate(step)
        if result is not None:
            result.log_headline(step)
            print(result.day3_rmse)  # {'temperature_variation': 1.23, ...}

    Args:
        model: the forecast model.
        config: WeatherBenchmarkConfig.
        loss_fn: optional loss function.
        ema_params: optional EMA parameters.
        climatology: optional climatological means for ACC.
    """

    def __init__(
        self,
        model: nn.Module,
        config: WeatherBenchmarkConfig | None = None,
        loss_fn: nn.Module | None = None,
        ema_params: dict[str, torch.Tensor] | None = None,
        climatology: dict[str, torch.Tensor] | None = None,
    ):
        self.config = config or WeatherBenchmarkConfig()
        rollout_steps = int(self.config.forecast_hours / self.config.dt_hours)

        self._eval_runner = EvaluationRunner(
            model=model,
            config=EvalConfig(
                eval_cadence=self.config.eval_cadence,
                n_eval_samples=self.config.n_eval_samples,
                rollout_steps=rollout_steps,
                compute_acc=self.config.compute_acc,
                variable_names=self.config.headline_vars,
            ),
            loss_fn=loss_fn,
            ema_params=ema_params,
            climatology=climatology,
        )

    def set_eval_data(
        self,
        states: list,
        forcings: list | None = None,
    ) -> None:
        """Set evaluation data (same interface as EvaluationRunner)."""
        self._eval_runner.set_eval_data(states, forcings)

    def set_eval_data_from_dataloader(
        self,
        dataloader,
        n_samples: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Set evaluation data from a DataLoader."""
        self._eval_runner.set_eval_data_from_dataloader(
            dataloader, n_samples=n_samples, device=device,
        )

    def should_evaluate(self, step: int) -> bool:
        """Check if benchmark evaluation should run at this step."""
        return self._eval_runner.should_evaluate(step)

    def maybe_evaluate(
        self,
        step: int,
        use_ema: bool = True,
    ) -> WeatherBenchmarkResult | None:
        """Evaluate if cadence triggers; return None otherwise."""
        if not self.should_evaluate(step):
            return None
        return self.evaluate(use_ema=use_ema)

    def evaluate(self, use_ema: bool = True) -> WeatherBenchmarkResult:
        """Run full benchmark evaluation and extract Day-3/Day-5 metrics."""
        eval_result = self._eval_runner.evaluate(use_ema=use_ema)
        return extract_day_metrics(eval_result, dt_hours=self.config.dt_hours)

    def update_ema_params(self, ema_params: dict[str, torch.Tensor]) -> None:
        """Update stored EMA parameters."""
        self._eval_runner.update_ema_params(ema_params)
