# API Reference

This page lists the stable public entry points. All internal imports use the
absolute `tornado_gcm.*` namespace.

## Core dynamics — `tornado_gcm.core`

| Symbol | Module | Purpose |
|---|---|---|
| `Grid` | `core.spherical_harmonic` | Horizontal spectral/nodal grid |
| `RealSphericalHarmonics` | `core.spherical_harmonic` | Real spherical-harmonic transforms |
| `CoordinateSystem` | `core.coordinate_systems` | Horizontal grid + vertical coordinate |
| `SigmaCoordinates` | `core.sigma_coordinates` | Normalized-pressure vertical levels |
| `HybridCoordinates` | `core.hybrid_coordinates` | Hybrid sigma-pressure levels |
| `LayerCoordinates` | `core.layer_coordinates` | Multi-layer shallow-water levels |
| `State` | `core.primitive_equations` | Modal prognostic state |
| `PrimitiveEquationsSigma` | `core.primitive_equations` | Sigma-coordinate primitive equations |
| `imex_rk_sil3` | `core.time_integration` | IMEX-RK3-SIL3 integrator |
| `ExponentialFilter` | `core.filtering` | Modal exponential filter |

```python
from tornado_gcm.core.spherical_harmonic import Grid
from tornado_gcm.core.sigma_coordinates import SigmaCoordinates
from tornado_gcm.core.coordinate_systems import CoordinateSystem

grid = Grid.construct(max_wavenumber=21, gaussian_nodes=16)
vertical = SigmaCoordinates.equidistant(layers=13)
coords = CoordinateSystem(horizontal=grid, vertical=vertical)

nodal = grid.to_nodal(modal_field)
modal = grid.to_modal(nodal_field)
```

## Model assembly — `tornado_gcm.model`

| Symbol | Module | Purpose |
|---|---|---|
| `NeuralGCMModel` | `model.api` | Coupling hub: dycore + parameterization + fixer |
| `ModelConfig` | `model.api` | Model configuration dataclass |
| `AtmosphericModel` | `model.atmosphere` | Encoder/decoder/stochastic facade |
| `build_jax_aligned_experiment_model` | `model.jax_checkpoint_loader` | Build a complete JAX-aligned model |
| `load_jax_checkpoint` | `model.jax_checkpoint_loader` | Flatten a JAX pickle to `params/...` |
| `build_jax_neural_components` | `model.jax_checkpoint_loader` | Build the neural component module |
| `load_jax_weights` | `model.jax_checkpoint_loader` | Load converted weights |

```python
from tornado_gcm.model.api import ModelConfig, NeuralGCMModel

model = NeuralGCMModel(
    config=ModelConfig(coords=coords, dt=dt, physics_specs=sim_units,
                       reference_temperature=reference_temperature),
    parameterization=parameterization,
    conservation_fixer=fixer,
)
final_state, trajectory = model(initial_state, outer_steps, inner_steps)
```

## Neural components — `tornado_gcm.neural`

| Symbol | Module | Purpose |
|---|---|---|
| `WeatherbenchToPrimitiveEncoder` | `neural.encoders_decoders` | WeatherBench dict → modal `State` |
| `PrimitiveToWeatherbenchDecoder` | `neural.encoders_decoders` | Modal `State` → WeatherBench dict |
| `CombinedFeatures` | `neural.features` | Compose feature modules |
| `ForwardTower`, `ColumnTower` | `neural.towers` | Feature-axis towers |
| `JaxAlignedDivCurlParameterization` | `neural.parameterizations` | Production learned physics |
| `F64ConservationFixer` | `neural.fixers` | Combined mass/energy fixer |

## Precision — `tornado_gcm.precision`

| Symbol | Module | Purpose |
|---|---|---|
| `PrecisionZone` | `precision.policy` | Z0–Z3 enum |
| `PrecisionPolicy` | `precision.policy` | Zone dtype configuration |
| `zone_cast` | `precision.zone_cast` | Boundary cast with optional audit |
| `f64_math`, `einsum_highest` | `precision.zone_cast` | FP64 compute helpers |
| `SDAConfig` | `precision.sda` | YAML-backed SDA configuration |
| `SDAController` | `precision.sda` | Profile → schedule → hot-swap |
| `PrecisionBenchmark` | `precision.benchmarks` | CLI benchmark workload |
| `ZoneDiscovery` | `precision.zone_discovery` | Automatic zone assignment |

```python
from tornado_gcm.precision.sda import SDAConfig, SDAController

cfg = SDAConfig.from_yaml("configs/sda_neuralgcm_1.4deg.yaml")
controller = SDAController(cfg)
controller.apply()
```

## Training — `tornado_gcm.training`

| Symbol | Module | Purpose |
|---|---|---|
| `MixedPrecisionTrainer` | `training.trainer` | Training loop with per-zone precision |
| `SDATrainer` | `training.sda_trainer` | Trainer + SDA controller |
| `TrajectoryMSE`, `PhysicsConstrainedLoss` | `training.losses` | Loss functions |
| `TrajectoryDataset`, `build_dataloader` | `training.data_loading` | Windowed datasets |
| `inputs_from_xarray` | `training.data_pipeline` | xarray dataset → tensor dict |
| `save_checkpoint`, `load_checkpoint` | `training.checkpointing` | Checkpoint I/O |
| `EvaluationRunner` | `training.evaluation` | Forecast evaluation |

```python
from tornado_gcm.training.trainer import MixedPrecisionTrainer

trainer = MixedPrecisionTrainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    policy=policy,
)
metrics = trainer.train_step(batch, rollout_steps=4)
```

## Inference — `tornado_gcm.inference`

| Symbol | Module | Purpose |
|---|---|---|
| `MPInference` | `inference.runner` | Mixed-precision rollout |
| `SDAInference` | `inference.sda_runner` | Adaptive/sparse rollout |
| `InferenceRunner` | `inference.production` | Chunked production runner |
| `ZarrWriter`, `NumpyWriter` | `inference.production` | Output writers |

```python
from tornado_gcm.inference.runner import MPInference

runner = MPInference(model)
states = runner.run(initial_state, outer_steps=40)
```

## Distributed — `tornado_gcm.distributed`

| Symbol | Module | Purpose |
|---|---|---|
| `setup_distributed`, `cleanup_distributed` | `distributed.ddp` | Process-group lifecycle |
| `wrap_ddp`, `wrap_fsdp` | `distributed.ddp` | Model wrappers |
| `all_reduce_metrics` | `distributed.ddp` | Metric aggregation |
| `create_device_mesh` | `distributed.dtensor_sharding` | DTensor mesh |
| `maybe_enable_dtensor_sharding` | `distributed.dtensor_sharding` | Env-driven sharding |

## CLI

```text
tornado-gcm train       # present but not wired in 0.1.0
tornado-gcm infer       # present but not wired in 0.1.0
tornado-gcm benchmark   # functional
```
