# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""SimUnits: physical constants and scaling routines for NeuralGCM PyTorch."""

from __future__ import annotations

import dataclasses
from typing import Protocol

from tornado_gcm import scales
from tornado_gcm import typing as typing_
import numpy as np

Quantity = typing_.Numeric
Numeric = typing_.Numeric


class SimUnitsProtocol(Protocol):
    """Protocol for a class that handles dimensionalization of quantities."""

    radius: float
    angular_velocity: float
    gravity_acceleration: float
    ideal_gas_constant: float
    water_vapor_gas_constant: float
    water_vapor_isobaric_heat_capacity: float
    kappa: float
    scale: scales.ScaleProtocol

    @property
    def R(self) -> float: ...

    @property
    def R_vapor(self) -> float: ...

    @property
    def g(self) -> float: ...

    @property
    def Cp(self) -> float: ...

    @property
    def Cp_vapor(self) -> float: ...

    def nondimensionalize(self, quantity: scales.Quantity) -> float: ...

    def nondimensionalize_timedelta64(
        self, timedelta: np.timedelta64
    ) -> float: ...

    def dimensionalize(
        self, value: float, unit: scales.Unit
    ) -> scales.Quantity: ...

    def dimensionalize_timedelta64(self, value: float) -> np.timedelta64: ...

    @classmethod
    def from_si(cls, **kwargs) -> "SimUnits": ...


@dataclasses.dataclass(frozen=True)
class SimUnits:
    """Physical constants and scaling routines.

    Stores non-dimensional physical constants and provides routines for
    dimensionalization and non-dimensionalization of quantities.
    """

    radius: float
    angular_velocity: float
    gravity_acceleration: float
    ideal_gas_constant: float
    water_vapor_gas_constant: float
    water_vapor_isobaric_heat_capacity: float
    kappa: float
    scale: scales.ScaleProtocol

    @property
    def R(self) -> float:
        return self.ideal_gas_constant

    @property
    def R_vapor(self) -> float:
        return self.water_vapor_gas_constant

    @property
    def g(self) -> float:
        return self.gravity_acceleration

    @property
    def Cp(self) -> float:
        return self.ideal_gas_constant / self.kappa

    @property
    def Cp_vapor(self) -> float:
        return self.water_vapor_isobaric_heat_capacity

    def nondimensionalize(self, quantity: scales.Quantity) -> float:
        return self.scale.nondimensionalize(quantity)

    def nondimensionalize_timedelta64(
        self, timedelta: np.timedelta64
    ) -> float:
        base_unit = "s"
        return self.scale.nondimensionalize(
            timedelta / np.timedelta64(1, base_unit) * scales.units(base_unit)
        )

    def dimensionalize(
        self, value: float, unit: scales.Unit
    ) -> scales.Quantity:
        return self.scale.dimensionalize(value, unit)

    def dimensionalize_timedelta64(self, value: float) -> np.timedelta64:
        base_unit = "s"
        dt = self.scale.dimensionalize(value, scales.units(base_unit)).m
        if isinstance(dt, np.ndarray):
            return dt.astype(f"timedelta64[{base_unit}]")
        else:
            return np.timedelta64(int(dt), base_unit)

    @classmethod
    def from_si(
        cls,
        radius_si: scales.Quantity = scales.RADIUS,
        angular_velocity_si: scales.Quantity = scales.ANGULAR_VELOCITY,
        gravity_acceleration_si: scales.Quantity = scales.GRAVITY_ACCELERATION,
        ideal_gas_constant_si: scales.Quantity = scales.IDEAL_GAS_CONSTANT,
        water_vapor_gas_constant_si: scales.Quantity = scales.IDEAL_GAS_CONSTANT_H20,
        water_vapor_isobaric_heat_capacity_si: scales.Quantity = scales.WATER_VAPOR_CP,
        kappa_si: scales.Quantity = scales.KAPPA,
        scale: scales.ScaleProtocol = scales.DEFAULT_SCALE,
    ) -> SimUnits:
        """Constructs SimUnits from SI constants. Default: NEURALGCM_V1_SCALE."""
        return cls(
            scale.nondimensionalize(radius_si),
            scale.nondimensionalize(angular_velocity_si),
            scale.nondimensionalize(gravity_acceleration_si),
            scale.nondimensionalize(ideal_gas_constant_si),
            scale.nondimensionalize(water_vapor_gas_constant_si),
            scale.nondimensionalize(water_vapor_isobaric_heat_capacity_si),
            scale.nondimensionalize(kappa_si),
            scale,
        )
