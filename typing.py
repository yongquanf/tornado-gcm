# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Commonly used type aliases for the PyTorch NeuralGCM codebase."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Generic, Mapping, Optional, TypeVar, Union

import torch
import numpy as np

# ---------------------------------------------------------------------------
# Array / numeric types
# ---------------------------------------------------------------------------
Tensor = torch.Tensor
Array = Union[np.ndarray, torch.Tensor]
ArrayOrArrayTuple = Union[Array, tuple[Array, ...]]
Numeric = Union[float, int, torch.Tensor]

PRNGKeyArray = Any  # torch.Generator or seed integer

# ---------------------------------------------------------------------------
# Generic state types
# ---------------------------------------------------------------------------
PyTreeState = TypeVar("PyTreeState")
Pytree = Any
PyTreeMemory = Pytree
PyTreeDiagnostics = Pytree

AuxFeatures = dict[str, Any]
DataState = dict[str, Any]
ForcingData = dict[str, Any]


# ---------------------------------------------------------------------------
# Key with cos-lat factor (used in spectral transformations)
# ---------------------------------------------------------------------------
@dataclasses.dataclass(eq=True, order=True, frozen=True)
class KeyWithCosLatFactor:
    """A key described by `name` and an integer `factor_order`."""

    name: str
    factor_order: int
    filter_strength: float = 0.0


# ---------------------------------------------------------------------------
# RandomnessState  (replaces tree_math.struct version)
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class RandomnessState:
    """Representation of random states on the sphere.

    Attributes:
        core: internal representation of the random state.
        nodal_value: random field values in the nodal representation.
        modal_value: random field values in the modal representation.
        prng_key: underlying PRNG key (torch.Generator or seed).
        prng_step: optional iteration counter for PRNG key management.
    """

    core: Pytree | None = None
    nodal_value: Pytree | None = None
    modal_value: Pytree | None = None
    prng_key: PRNGKeyArray | None = None
    prng_step: int | None = None


# ---------------------------------------------------------------------------
# ModelState  (replaces tree_math.struct Generic version)
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class ModelState(Generic[PyTreeState]):
    """PyTreeState decomposed into deterministic and perturbation components.

    Attributes:
        state: Prognostic variables describing the state of the atmosphere.
        memory: Optional model fields/predictions providing past time context.
        diagnostics: Optional diagnostic values computed in the model space.
        randomness: An optional random field for stochastic perturbation.
    """

    state: PyTreeState
    memory: Pytree = None
    diagnostics: Pytree = dataclasses.field(default_factory=dict)
    randomness: RandomnessState = dataclasses.field(
        default_factory=RandomnessState
    )


# ---------------------------------------------------------------------------
# TrajectoryRepresentations
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class TrajectoryRepresentations:
    """Dataclass that holds trajectories in all default representations."""

    data_nodal_trajectory: Pytree
    data_modal_trajectory: Pytree
    model_nodal_trajectory: Pytree
    model_modal_trajectory: Pytree

    def get_representation(
        self, *, is_nodal: bool, is_encoded: bool
    ) -> Pytree:
        """Retrieves representation based on `is_nodal`, `is_encoded`."""
        key = (is_nodal, is_encoded)
        lookup = {
            (True, True): self.model_nodal_trajectory,
            (True, False): self.data_nodal_trajectory,
            (False, True): self.model_modal_trajectory,
            (False, False): self.data_modal_trajectory,
        }
        return lookup[key]


# ---------------------------------------------------------------------------
# Callable type aliases
# ---------------------------------------------------------------------------
State = TypeVar("State")
StateFn = Callable[[State], State]
InverseFn = Callable[[State, Tensor], State]
StepFn = Callable[[State, State], State]
FilterFn = Callable[[State, State, State], tuple[State, State]]

ScanFn = Callable[..., Any]
PytreeFn = Callable[[Pytree], Pytree]
PyTreeTermsFn = Callable[[PyTreeState], PyTreeState]
PyTreeInverseFn = Callable[[PyTreeState, Numeric], PyTreeState]
TimeStepFn = Callable[[PyTreeState], PyTreeState]
PyTreeFilterFn = Callable[[PyTreeState], PyTreeState]
PyTreeStepFilterFn = Callable[[PyTreeState, PyTreeState], PyTreeState]
PyTreeStepFilterModule = Callable[..., PyTreeStepFilterFn]

Forcing = Pytree
ForcingFn = Callable[[ForcingData, float], Forcing]
ForcingModule = Callable[..., ForcingFn]

PostProcessFn = Callable[..., Any]
Params = Optional[Mapping[str, Mapping[str, Array]]]

StepFn = Callable[[PyTreeState, Optional[Forcing]], PyTreeState]  # type: ignore[assignment]
StepModule = Callable[..., StepFn]
CorrectorFn = Callable[
    [PyTreeState, Optional[PyTreeState], Optional[Forcing]], PyTreeState
]
CorrectorModule = Callable[..., CorrectorFn]
ParameterizationFn = Callable[
    [
        PyTreeState,
        Optional[PyTreeMemory],
        Optional[PyTreeDiagnostics],
        Optional[RandomnessState],
        Optional[Forcing],
    ],
    PyTreeState,
]
ParameterizationModule = Callable[..., ParameterizationFn]
TrajectoryFn = Callable[..., tuple[Any, Any]]
TransformFn = Callable[[Pytree], Pytree]
TransformModule = Callable[..., TransformFn]

GatingFactory = Callable[..., Callable[[Tensor, Tensor], Tensor]]
TowerFactory = Callable[..., Callable[..., Any]]
LayerFactory = Callable[..., Callable[..., Any]]

EmbeddingFn = Callable[
    [
        Pytree,
        Optional[PyTreeMemory],
        Optional[PyTreeDiagnostics],
        Optional[RandomnessState],
        Optional[Forcing],
    ],
    Pytree,
]
EmbeddingModule = Callable[..., EmbeddingFn]
