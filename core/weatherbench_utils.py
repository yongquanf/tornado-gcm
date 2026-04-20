# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""WeatherBench-compatible state structure — PyTorch implementation."""

from __future__ import annotations

import dataclasses

import torch


@dataclasses.dataclass
class State:
    """A WeatherBench state with velocity components.

    Attributes:
        u: zonal wind (layers, lon, lat).
        v: meridional wind (layers, lon, lat).
        t: temperature (layers, lon, lat).
        z: geopotential (layers, lon, lat).
        sim_time: simulation time (scalar).
        tracers: additional tracer fields.
        diagnostics: additional diagnostic fields.
    """

    u: torch.Tensor
    v: torch.Tensor
    t: torch.Tensor
    z: torch.Tensor
    sim_time: float = 0.0
    tracers: dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    diagnostics: dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)

    def __add__(self, other: State) -> State:
        return State(
            u=self.u + other.u,
            v=self.v + other.v,
            t=self.t + other.t,
            z=self.z + other.z,
            sim_time=self.sim_time,
            tracers={k: self.tracers[k] + other.tracers[k] for k in self.tracers},
            diagnostics=self.diagnostics,
        )

    def __mul__(self, scalar: float) -> State:
        return State(
            u=self.u * scalar,
            v=self.v * scalar,
            t=self.t * scalar,
            z=self.z * scalar,
            sim_time=self.sim_time,
            tracers={k: v * scalar for k, v in self.tracers.items()},
            diagnostics=self.diagnostics,
        )

    def __rmul__(self, scalar: float) -> State:
        return self.__mul__(scalar)
