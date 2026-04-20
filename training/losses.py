# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Physics-constrained losses for NeuralGCM training.

Implements:
  - TrajectoryMSE / TrajectoryMAE: standard prediction losses
  - CRPSLoss: Continuous Ranked Probability Score for ensembles
  - BatchedParameterLoss: aggregated per-variable weighted loss
  - AggregationState: running statistics accumulator
  - PhysicsConstrainedLoss: L_pred + λ_E·L_E + λ_M·L_M + λ_H·L_H
  - SumLoss: weighted combination of multiple loss terms
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
from tornado_gcm.precision.zone_cast import f64_math


class TrajectoryMSE(nn.Module):
    """Mean squared error over a rollout trajectory.

    Supports both plain Tensors and State objects (computes MSE over all fields).
    """

    def _state_mse(self, pred, tgt) -> torch.Tensor:
        """MSE between two State objects."""
        from tornado_gcm.core.primitive_equations import State
        total = torch.tensor(0.0, device=pred.vorticity.device)
        total = total + ((pred.vorticity - tgt.vorticity) ** 2).mean()
        total = total + ((pred.divergence - tgt.divergence) ** 2).mean()
        total = total + ((pred.temperature_variation - tgt.temperature_variation) ** 2).mean()
        total = total + ((pred.log_surface_pressure - tgt.log_surface_pressure) ** 2).mean()
        return total

    def forward(
        self,
        predictions: list,
        targets: list,
    ) -> torch.Tensor:
        from tornado_gcm.core.primitive_equations import State
        total = torch.tensor(0.0)
        for pred, tgt in zip(predictions, targets):
            if isinstance(pred, State):
                total = total + self._state_mse(pred, tgt)
            else:
                total = total + ((pred - tgt) ** 2).mean()
        return total / len(predictions)


class TrajectoryMAE(nn.Module):
    """Mean absolute error over a rollout trajectory."""

    def _state_mae(self, pred, tgt) -> torch.Tensor:
        from tornado_gcm.core.primitive_equations import State
        total = torch.tensor(0.0, device=pred.vorticity.device)
        total = total + (pred.vorticity - tgt.vorticity).abs().mean()
        total = total + (pred.divergence - tgt.divergence).abs().mean()
        total = total + (pred.temperature_variation - tgt.temperature_variation).abs().mean()
        total = total + (pred.log_surface_pressure - tgt.log_surface_pressure).abs().mean()
        return total

    def forward(self, predictions: list, targets: list) -> torch.Tensor:
        from tornado_gcm.core.primitive_equations import State
        total = torch.tensor(0.0)
        for pred, tgt in zip(predictions, targets):
            if isinstance(pred, State):
                total = total + self._state_mae(pred, tgt)
            else:
                total = total + (pred - tgt).abs().mean()
        return total / len(predictions)


def _safe_sqrt(x: torch.Tensor) -> torch.Tensor:
    """Sqrt with safe gradient at zero (avoids NaN grad)."""
    eps = torch.finfo(x.dtype).eps
    return torch.sqrt(x.clamp(min=eps))


class CRPSLoss(nn.Module):
    r"""Continuous Ranked Probability Score for ensemble predictions.

    .. math::
        \text{CRPS} = E\|X - Y\| - \frac{1}{2} E\|X - X'\|

    where X, X' are ensemble members and Y is the target.
    The spread term weight defaults to 0.5 (energy score convention).

    Expects predictions with an ensemble dimension (dim 0).
    """

    def __init__(
        self,
        spread_term_weight: float = 0.5,
        variable_weights: dict[str, float] | None = None,
    ):
        super().__init__()
        self.spread_term_weight = spread_term_weight
        self.variable_weights = variable_weights or {}

    def _crps_tensor(
        self, pred_ensemble: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """CRPS for a single tensor with ensemble dim 0.

        Args:
            pred_ensemble: (E, ...) ensemble predictions.
            target: (...) single target.
        """
        e = pred_ensemble.shape[0]
        # Skill: E[|X - Y|]
        skill = (pred_ensemble - target.unsqueeze(0)).abs().mean(dim=0)
        # Spread: E[|X - X'|] via cyclic shift (matches JAX pshuffle)
        pred_shifted = pred_ensemble.roll(shifts=1, dims=0)
        spread = (pred_ensemble - pred_shifted).abs().mean(dim=0)
        crps = skill - self.spread_term_weight * spread
        return crps.mean()

    def forward(
        self,
        predictions: list[torch.Tensor | dict],
        targets: list[torch.Tensor | dict],
    ) -> torch.Tensor:
        """Compute CRPS over a trajectory.

        predictions: list of (E, ...) ensemble tensors or dicts of such.
        targets: list of (...) single-member targets or dicts.
        """
        total = torch.tensor(0.0)
        for pred, tgt in zip(predictions, targets):
            if isinstance(pred, dict) and isinstance(tgt, dict):
                for key in pred:
                    w = self.variable_weights.get(key, 1.0)
                    total = total + w * self._crps_tensor(pred[key], tgt[key])
            else:
                total = total + self._crps_tensor(pred, tgt)
        return total / max(len(predictions), 1)


class AggregationState:
    """Running aggregation of weighted statistics for loss computation.

    Accumulates sum_of(statistic * weight) and sum_of(weight) independently,
    enabling correct mean computation across multiple batches.
    """

    def __init__(self):
        self.sum_weighted: dict[str, torch.Tensor] = {}
        self.sum_weights: dict[str, torch.Tensor] = {}

    def update(
        self,
        values: dict[str, torch.Tensor],
        weights: dict[str, torch.Tensor] | None = None,
    ) -> None:
        for key, val in values.items():
            w = weights[key] if weights is not None and key in weights else torch.ones_like(val)
            if key in self.sum_weighted:
                self.sum_weighted[key] = self.sum_weighted[key] + val * w
                self.sum_weights[key] = self.sum_weights[key] + w
            else:
                self.sum_weighted[key] = val * w
                self.sum_weights[key] = w.clone()

    def mean(self) -> dict[str, torch.Tensor]:
        result = {}
        for key in self.sum_weighted:
            result[key] = self.sum_weighted[key] / self.sum_weights[key].clamp(min=1e-12)
        return result

    def __add__(self, other: "AggregationState") -> "AggregationState":
        merged = AggregationState()
        all_keys = set(self.sum_weighted) | set(other.sum_weighted)
        for key in all_keys:
            if key in self.sum_weighted and key in other.sum_weighted:
                merged.sum_weighted[key] = self.sum_weighted[key] + other.sum_weighted[key]
                merged.sum_weights[key] = self.sum_weights[key] + other.sum_weights[key]
            elif key in self.sum_weighted:
                merged.sum_weighted[key] = self.sum_weighted[key].clone()
                merged.sum_weights[key] = self.sum_weights[key].clone()
            else:
                merged.sum_weighted[key] = other.sum_weighted[key].clone()
                merged.sum_weights[key] = other.sum_weights[key].clone()
        return merged


class BatchedParameterLoss(nn.Module):
    """Per-variable weighted loss with aggregation support.

    Computes a weighted sum of per-variable losses, where each variable
    can use a different loss function (MSE or MAE) and weight.
    """

    def __init__(
        self,
        variable_weights: dict[str, float] | None = None,
        loss_type: str = "mse",
    ):
        super().__init__()
        self.variable_weights = variable_weights or {}
        self.loss_type = loss_type

    def _compute_one(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mae":
            return (pred - tgt).abs().mean()
        return ((pred - tgt) ** 2).mean()

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        agg_state: AggregationState | None = None,
    ) -> tuple[torch.Tensor, AggregationState]:
        """Compute weighted per-variable loss.

        Returns (total_loss, aggregation_state).
        """
        if agg_state is None:
            agg_state = AggregationState()

        values = {}
        total = torch.tensor(0.0)
        for key in predictions:
            if key not in targets:
                continue
            val = self._compute_one(predictions[key], targets[key])
            w = self.variable_weights.get(key, 1.0)
            values[key] = val
            total = total + w * val

        agg_state.update(values)
        return total, agg_state


class SumLoss(nn.Module):
    """Weighted combination of multiple loss terms."""

    def __init__(
        self,
        terms: dict[str, nn.Module],
        term_weights: dict[str, float] | None = None,
    ):
        super().__init__()
        self.terms = nn.ModuleDict(terms)
        self.term_weights = term_weights or {}

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        result = {}
        total = torch.tensor(0.0)
        for name, loss_fn in self.terms.items():
            val = loss_fn(**kwargs)
            if isinstance(val, dict):
                val = val.get("total", sum(val.values()))
            w = self.term_weights.get(name, 1.0)
            result[name] = val
            total = total + w * val
        result["total"] = total
        return result


class EnergyConservationLoss(nn.Module):
    r"""Relative energy drift loss (Z2 — computed in FP64).

    .. math::
        \mathcal{L}_E = \frac{1}{K}\sum_{k=1}^{K}
            \left(\frac{|\hat{E}_k - \hat{E}_{k-1}|}{|\hat{E}_0|}\right)^2
    """

    @f64_math
    def forward(self, energies: torch.Tensor) -> torch.Tensor:
        """Args: energies — shape (K+1,) total energy at each saved step."""
        e0 = energies[0].abs().clamp(min=1e-12)
        diffs = (energies[1:] - energies[:-1]).abs() / e0
        return (diffs ** 2).mean()


class MassConservationLoss(nn.Module):
    r"""Relative dry-air mass drift loss (Z2 — FP64).

    .. math::
        \mathcal{L}_M = \frac{1}{K}\sum_{k=1}^{K}
            \left(\frac{|\hat{M}_k^d - \hat{M}_0^d|}{|\hat{M}_0^d|}\right)^2
    """

    @f64_math
    def forward(self, masses: torch.Tensor) -> torch.Tensor:
        """Args: masses — shape (K+1,) total dry mass at each step."""
        m0 = masses[0].abs().clamp(min=1e-12)
        drifts = (masses[1:] - masses[0]).abs() / m0
        return (drifts ** 2).mean()


class HydrologicalClosureLoss(nn.Module):
    r"""Penalize non-physical precipitation/evaporation.

    .. math::
        \mathcal{L}_H = \frac{1}{K}\sum_{k=1}^{K}
            \bigl[\max(0, -\hat{P}_k)^2 + \max(0, \hat{E}_k)^2\bigr]

    Convention: P >= 0 (precipitation), E <= 0 (evaporation).
    """

    def forward(
        self,
        precipitation: torch.Tensor,
        evaporation: torch.Tensor,
    ) -> torch.Tensor:
        neg_precip = torch.clamp(-precipitation, min=0.0)
        pos_evap = torch.clamp(evaporation, min=0.0)
        return (neg_precip ** 2 + pos_evap ** 2).mean()


class PhysicsConstrainedLoss(nn.Module):
    r"""Combined loss: prediction + physics constraints.

    .. math::
        \mathcal{L} = \mathcal{L}_{\text{pred}}
            + \lambda_E \mathcal{L}_E
            + \lambda_M \mathcal{L}_M
            + \lambda_H \mathcal{L}_H

    During warmup (first ``warmup_steps``), only :math:`\mathcal{L}_{\text{pred}}`
    is applied to let the model learn basic dynamics before adding constraints.

    Typical λ values: 1e-3 ~ 1e-2.
    """

    def __init__(
        self,
        lambda_e: float = 1e-3,
        lambda_m: float = 1e-3,
        lambda_h: float = 1e-3,
        warmup_steps: int = 1000,
    ):
        super().__init__()
        self.lambda_e = lambda_e
        self.lambda_m = lambda_m
        self.lambda_h = lambda_h
        self.warmup_steps = warmup_steps

        self.pred_loss = TrajectoryMSE()
        self.energy_loss = EnergyConservationLoss()
        self.mass_loss = MassConservationLoss()
        self.hydro_loss = HydrologicalClosureLoss()

    def forward(
        self,
        predictions: list[torch.Tensor],
        targets: list[torch.Tensor],
        energies: torch.Tensor | None = None,
        masses: torch.Tensor | None = None,
        precipitation: torch.Tensor | None = None,
        evaporation: torch.Tensor | None = None,
        step: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Returns dict with 'total', 'pred', 'energy', 'mass', 'hydro' keys.
        """
        device = predictions[0].device if isinstance(predictions[0], torch.Tensor) else predictions[0].vorticity.device
        l_pred = self.pred_loss(predictions, targets)

        past_warmup = step >= self.warmup_steps
        zero = torch.tensor(0.0, device=device)

        l_e = self.energy_loss(energies) if (past_warmup and energies is not None) else zero
        l_m = self.mass_loss(masses) if (past_warmup and masses is not None) else zero
        l_h = (
            self.hydro_loss(precipitation, evaporation)
            if (past_warmup and precipitation is not None and evaporation is not None)
            else zero
        )

        total = l_pred + self.lambda_e * l_e + self.lambda_m * l_m + self.lambda_h * l_h

        return {
            "total": total,
            "pred": l_pred,
            "energy": l_e,
            "mass": l_m,
            "hydro": l_h,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Spectral-domain losses (training fidelity — matching JAX original)
# ═══════════════════════════════════════════════════════════════════════════


def _state_to_modal_dict(state) -> dict[str, torch.Tensor]:
    """Extract modal (spectral) fields from a State object as a dict."""
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
    if isinstance(state, dict):
        return state
    raise TypeError(f"Cannot extract modal fields from {type(state)}")


def _spectral_norm(x: torch.Tensor) -> torch.Tensor:
    """Compute spectral norm per total wavenumber l.

    Input x has shape (..., m, l) where m is the longitudinal wavenumber axis.
    Returns shape (..., 1, l): sqrt(sum_m |x_{m,l}|^2).
    """
    return _safe_sqrt((x * x).sum(dim=-2, keepdim=True))


class SpectralMSELoss(nn.Module):
    r"""MSE loss in spherical harmonic modal space with progressive wavenumber filtering.

    Implements the equivalent of NeuralGCM's ``TransformedL2Loss`` in spectral
    basis with lead-time-dependent high-wavenumber masking to avoid the
    "double penalty" problem.

    The mask for total wavenumber :math:`l` at rollout step :math:`k` is:

    .. math::
        w(l, k) = \sigma\!\left(\frac{l_{\max}(k) - l}{\tau}\right),
        \quad l_{\max}(k) = L \cdot \max\!\bigl(0,\; 1 - \alpha\,k\bigr)

    where :math:`\sigma` is the sigmoid function, :math:`\alpha` controls the
    decay rate, and :math:`\tau` controls smoothness.

    Args:
        total_wavenumbers: maximum total wavenumber L of the spectral grid.
        alpha: decay rate of the wavenumber cutoff per rollout step.
            Default 0.05 means at step 20 cutoff reaches zero.
        tau: sigmoid smoothness (higher = sharper cutoff). Default 2.0.
        variable_weights: optional per-field weight dict.
    """

    def __init__(
        self,
        total_wavenumbers: int,
        alpha: float = 0.05,
        tau: float = 2.0,
        variable_weights: dict[str, float] | None = None,
    ):
        super().__init__()
        self.L = total_wavenumbers
        self.alpha = alpha
        self.tau = tau
        self.variable_weights = variable_weights or {}

    def _wavenumber_mask(self, rollout_step: int, device: torch.device) -> torch.Tensor:
        """Compute wavenumber mask for a given rollout step. Shape (1, L)."""
        l = torch.arange(self.L, dtype=torch.float32, device=device)
        l_cutoff = self.L * max(0.0, 1.0 - self.alpha * rollout_step)
        mask = torch.sigmoid((l_cutoff - l) / self.tau)
        return mask.unsqueeze(0)  # (1, L)

    def forward(
        self,
        predictions: list,
        targets: list,
    ) -> torch.Tensor:
        """Compute spectral MSE with progressive wavenumber filtering.

        Args:
            predictions: list of K State objects (modal space).
            targets: list of K State objects (modal space).
        """
        total = torch.tensor(0.0)
        device = None
        for k, (pred, tgt) in enumerate(zip(predictions, targets)):
            pred_d = _state_to_modal_dict(pred)
            tgt_d = _state_to_modal_dict(tgt)
            if device is None:
                device = next(iter(pred_d.values())).device
            mask = self._wavenumber_mask(k, device)  # (1, L)
            for key in pred_d:
                if key not in tgt_d:
                    continue
                w = self.variable_weights.get(key, 1.0)
                err = pred_d[key] - tgt_d[key]
                # err shape: (..., m, l) — mask broadcasts over leading dims and m
                masked_sq = (err ** 2) * mask
                total = total + w * masked_sq.mean()
        return total / max(len(predictions), 1)


class TransformedL2Loss(nn.Module):
    r"""L2 loss with time rescaling and per-variable weighting.

    Faithful port of JAX NeuralGCM's ``TransformedL2Loss`` configured with
    ``LegacyTimeRescaling`` + ``PerVariableRescaling`` components.

    Time rescaling applies a single fixed factor uniformly to all timesteps:

    .. math::
        \text{scale} = \begin{cases}
            1 & \text{if } n = 1 \\
            \frac{1}{\sqrt{(n-1) \cdot s}} & \text{if } n > 1
        \end{cases}

    where :math:`n` is the trajectory length and :math:`s` is ``steps_per_save``.

    By Parseval's theorem for orthonormal spherical harmonics, computing L2 in
    modal (spectral) space is equivalent to computing in nodal (physical) space.

    Args:
        steps_per_save: number of model inner steps between saved trajectory
            points.  Used by LegacyTimeRescaling.
        variable_weights: optional per-field weight dict (PerVariableRescaling).
    """

    def __init__(
        self,
        steps_per_save: int = 1,
        variable_weights: dict[str, float] | None = None,
    ):
        super().__init__()
        self.steps_per_save = steps_per_save
        self.variable_weights = variable_weights or {}

    def forward(
        self,
        predictions: list,
        targets: list,
    ) -> torch.Tensor:
        n = len(predictions)
        # LegacyTimeRescaling: fixed scale² applied uniformly to all timesteps.
        # JAX: scale = 1/sqrt((n-1)*steps_per_save)  →  scale² = 1/((n-1)*s)
        time_scale_sq = 1.0 if n <= 1 else 1.0 / ((n - 1) * self.steps_per_save)

        total = torch.tensor(0.0)
        for pred, tgt in zip(predictions, targets):
            pred_d = _state_to_modal_dict(pred)
            tgt_d = _state_to_modal_dict(tgt)
            for key in pred_d:
                if key not in tgt_d:
                    continue
                w = self.variable_weights.get(key, 1.0)
                err = pred_d[key] - tgt_d[key]
                # Combined transform:  (error × scale × √w)²  =  error² × scale² × w
                total = total + time_scale_sq * w * (err ** 2).mean()
        # Mean over trajectory timesteps (matches JAX jnp.mean over all dims)
        return total / max(n, 1)


class SpectralNormMatchingLoss(nn.Module):
    r"""MSE on per-wavenumber spectral norms — encourages power spectrum fidelity.

    Implements the equivalent of NeuralGCM's ``TransformedL2SpectrumLoss``.
    For each field and level, computes

    .. math::
        \text{norm}_l = \sqrt{\sum_m |x_{m,l}|^2}

    then takes MSE between predicted and target norms.

    Args:
        variable_weights: optional per-field weight dict.
    """

    def __init__(self, variable_weights: dict[str, float] | None = None):
        super().__init__()
        self.variable_weights = variable_weights or {}

    def forward(
        self,
        predictions: list,
        targets: list,
    ) -> torch.Tensor:
        total = torch.tensor(0.0)
        for pred, tgt in zip(predictions, targets):
            pred_d = _state_to_modal_dict(pred)
            tgt_d = _state_to_modal_dict(tgt)
            for key in pred_d:
                if key not in tgt_d:
                    continue
                w = self.variable_weights.get(key, 1.0)
                pred_norm = _spectral_norm(pred_d[key])   # (..., 1, l)
                tgt_norm = _spectral_norm(tgt_d[key])
                total = total + w * ((pred_norm - tgt_norm) ** 2).mean()
        return total / max(len(predictions), 1)


class BatchMeanBiasLoss(nn.Module):
    r"""Bias penalty on batch-mean spectral amplitude.

    Implements the equivalent of NeuralGCM's ``BatchMeanSquaredBias``.
    Computes MSE between rollout-averaged spectral amplitudes of prediction
    and target, penalizing systematic biases in each spherical harmonic
    coefficient.

    .. math::
        \mathcal{L}_{\text{bias}} = \text{MSE}\!\bigl(
            \overline{|\hat{x}|},\;\overline{|\hat{y}|}
        \bigr)

    where :math:`\overline{\cdot}` is the rollout-time average.

    Args:
        variable_weights: optional per-field weight dict.
    """

    def __init__(self, variable_weights: dict[str, float] | None = None):
        super().__init__()
        self.variable_weights = variable_weights or {}

    def forward(
        self,
        predictions: list,
        targets: list,
    ) -> torch.Tensor:
        # Accumulate spectral amplitudes across rollout steps
        pred_sum: dict[str, torch.Tensor] = {}
        tgt_sum: dict[str, torch.Tensor] = {}
        n = len(predictions)
        for pred, tgt in zip(predictions, targets):
            pred_d = _state_to_modal_dict(pred)
            tgt_d = _state_to_modal_dict(tgt)
            for key in pred_d:
                if key not in tgt_d:
                    continue
                pa = pred_d[key].abs()
                ta = tgt_d[key].abs()
                if key in pred_sum:
                    pred_sum[key] = pred_sum[key] + pa
                    tgt_sum[key] = tgt_sum[key] + ta
                else:
                    pred_sum[key] = pa
                    tgt_sum[key] = ta

        total = torch.tensor(0.0)
        for key in pred_sum:
            w = self.variable_weights.get(key, 1.0)
            pred_mean = pred_sum[key] / n
            tgt_mean = tgt_sum[key] / n
            total = total + w * ((pred_mean - tgt_mean) ** 2).mean()
        return total


class NeuralGCMLoss(nn.Module):
    r"""Full NeuralGCM-faithful combined loss.

    .. math::
        \mathcal{L} = \lambda_{\text{l2}}\,\mathcal{L}_{\text{transformed-l2}}
                     + \lambda_{\text{spec}}\,\mathcal{L}_{\text{spectrum}}
                     + \lambda_{\text{bias}}\,\mathcal{L}_{\text{bias}}
                     + \lambda_E\,\mathcal{L}_E
                     + \lambda_M\,\mathcal{L}_M
                     + \lambda_H\,\mathcal{L}_H

    The first three terms reproduce the NeuralGCM paper's training objectives
    (transformed L2 with time rescaling, power spectrum matching, and bias
    penalization).  The last three are physics conservation constraints
    (PyTorch-only extension).
    """

    def __init__(
        self,
        steps_per_save: int = 1,
        lambda_l2: float = 1.0,
        lambda_spec: float = 1.0,
        lambda_bias: float = 1.0,
        lambda_e: float = 1e-3,
        lambda_m: float = 1e-3,
        lambda_h: float = 1e-3,
        warmup_steps: int = 1000,
    ):
        super().__init__()
        self.lambda_l2 = lambda_l2
        self.lambda_spec = lambda_spec
        self.lambda_bias = lambda_bias
        self.lambda_e = lambda_e
        self.lambda_m = lambda_m
        self.lambda_h = lambda_h
        self.warmup_steps = warmup_steps

        self.transformed_l2 = TransformedL2Loss(steps_per_save=steps_per_save)
        self.spectrum_match = SpectralNormMatchingLoss()
        self.bias_loss = BatchMeanBiasLoss()
        self.energy_loss = EnergyConservationLoss()
        self.mass_loss = MassConservationLoss()
        self.hydro_loss = HydrologicalClosureLoss()

    def forward(
        self,
        predictions: list,
        targets: list,
        energies: torch.Tensor | None = None,
        masses: torch.Tensor | None = None,
        precipitation: torch.Tensor | None = None,
        evaporation: torch.Tensor | None = None,
        step: int = 0,
    ) -> dict[str, torch.Tensor]:
        from tornado_gcm.core.primitive_equations import State
        device = (
            predictions[0].vorticity.device
            if isinstance(predictions[0], State)
            else predictions[0].device
        )

        l_l2 = self.transformed_l2(predictions, targets)
        l_spec = self.spectrum_match(predictions, targets)
        l_bias = self.bias_loss(predictions, targets)

        past_warmup = step >= self.warmup_steps
        zero = torch.tensor(0.0, device=device)

        l_e = self.energy_loss(energies) if (past_warmup and energies is not None) else zero
        l_m = self.mass_loss(masses) if (past_warmup and masses is not None) else zero
        l_h = (
            self.hydro_loss(precipitation, evaporation)
            if (past_warmup and precipitation is not None and evaporation is not None)
            else zero
        )

        total = (
            self.lambda_l2 * l_l2
            + self.lambda_spec * l_spec
            + self.lambda_bias * l_bias
            + self.lambda_e * l_e
            + self.lambda_m * l_m
            + self.lambda_h * l_h
        )

        return {
            "total": total,
            "transformed_l2": l_l2,
            "spectrum_match": l_spec,
            "bias": l_bias,
            "energy": l_e,
            "mass": l_m,
            "hydro": l_h,
        }
