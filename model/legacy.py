"""Legacy compatibility: PressureLevelModel wrapper.

Provides a simplified API that accepts pressure-level inputs (like ERA5)
and wraps the full NeuralGCM AtmosphericModel for backward compatibility
with code that expects a direct pressure-level interface.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import torch
import torch.nn as nn


@dataclasses.dataclass
class PressureLevelConfig:
    """Configuration for pressure-level model.

    Attributes:
        pressure_levels: 1-D array of target pressure levels in Pa.
        input_variables: list of input variable names.
        forcing_variables: list of forcing variable names.
        output_variables: list of output variable names.
        dt_hours: time step in hours.
    """

    pressure_levels: list[float] = dataclasses.field(
        default_factory=lambda: [
            100, 200, 300, 500, 700, 850, 925, 1000,
        ]
    )
    input_variables: list[str] = dataclasses.field(
        default_factory=lambda: [
            "u_component_of_wind",
            "v_component_of_wind",
            "temperature",
            "specific_humidity",
            "geopotential",
        ]
    )
    forcing_variables: list[str] = dataclasses.field(
        default_factory=lambda: [
            "sea_surface_temperature",
            "sea_ice_cover",
            "geopotential_at_surface",
        ]
    )
    output_variables: list[str] = dataclasses.field(
        default_factory=lambda: [
            "u_component_of_wind",
            "v_component_of_wind",
            "temperature",
            "specific_humidity",
            "geopotential",
        ]
    )
    dt_hours: float = 6.0


class PressureLevelModel(nn.Module):
    """Pressure-level interface to NeuralGCM (legacy API).

    Wraps the full AtmosphericModel (or any model with encode/decode)
    to provide a simplified predict interface:
        inputs (pressure-level dict) → encode → step N times → decode → outputs

    This mirrors the legacy API from the JAX codebase where users work
    directly with pressure-level xarray Datasets.
    """

    def __init__(
        self,
        atmosphere_model,
        config: PressureLevelConfig | None = None,
    ):
        super().__init__()
        self.atmosphere = atmosphere_model
        self.config = config or PressureLevelConfig()

    @property
    def input_variables(self) -> list[str]:
        return self.config.input_variables

    @property
    def forcing_variables(self) -> list[str]:
        return self.config.forcing_variables

    @property
    def output_variables(self) -> list[str]:
        return self.config.output_variables

    def encode(self, inputs: dict[str, torch.Tensor]) -> dict:
        """Encode pressure-level inputs to model state."""
        if hasattr(self.atmosphere, "encode"):
            return self.atmosphere.encode(inputs)
        return inputs

    def decode(self, state) -> dict[str, torch.Tensor]:
        """Decode model state to pressure-level outputs."""
        if hasattr(self.atmosphere, "decode"):
            return self.atmosphere.decode(state)
        if isinstance(state, dict):
            return state
        # Fallback: extract fields from State
        result = {}
        if hasattr(state, "vorticity"):
            result["vorticity"] = state.vorticity
            result["divergence"] = state.divergence
            result["temperature_variation"] = state.temperature_variation
            result["log_surface_pressure"] = state.log_surface_pressure
        return result

    def advance(self, state, forcings: dict[str, torch.Tensor] | None = None):
        """Advance model state by one outer step."""
        if hasattr(self.atmosphere, "step"):
            return self.atmosphere.step(state, forcings=forcings)
        return self.atmosphere(state)

    @torch.no_grad()
    def predict(
        self,
        inputs: dict[str, torch.Tensor],
        forcings: dict[str, torch.Tensor] | None = None,
        outer_steps: int = 1,
    ) -> list[dict[str, torch.Tensor]]:
        """Run prediction from pressure-level inputs.

        Args:
            inputs: dict of variable → tensor (levels, lon, lat).
            forcings: dict of forcing variable → tensor.
            outer_steps: number of output time steps.

        Returns:
            list of output dicts (one per outer step).
        """
        state = self.encode(inputs)
        outputs = []
        for _ in range(outer_steps):
            state = self.advance(state, forcings=forcings)
            outputs.append(self.decode(state))
        return outputs

    def inputs_from_xarray(self, dataset) -> dict[str, torch.Tensor]:
        """Convert xarray Dataset to model inputs."""
        from pytorch_src.training.data_pipeline import inputs_from_xarray
        import numpy as np
        data = inputs_from_xarray(dataset, self.config.input_variables)
        return {k: torch.from_numpy(v).float() for k, v in data.items()
                if isinstance(v, np.ndarray)}

    def forcings_from_xarray(self, dataset) -> dict[str, torch.Tensor]:
        """Convert xarray Dataset to forcing dict."""
        from pytorch_src.training.data_pipeline import forcings_from_xarray
        import numpy as np
        data = forcings_from_xarray(dataset, self.config.forcing_variables)
        return {k: torch.from_numpy(v).float() for k, v in data.items()
                if isinstance(v, np.ndarray)}
