# Quick Start

## CLI

The command-line entry point is available either as `tornado-gcm` after
installation or as `python cli.py` from the source tree.

```bash
tornado-gcm --help

# Precision / performance benchmark (wired in 0.1.0)
tornado-gcm benchmark \
  --config configs/sda_neuralgcm_2.8deg_full.yaml \
  --steps 20 \
  --device cuda \
  --report bench.txt
```

!!! note "CLI status in 0.1.0"
    The `benchmark` subcommand is wired. The `train` and `infer`
    subcommands currently reference library APIs that are not present in
    this release; use the Python APIs below instead.

## Load JAX weights

Convert a JAX NeuralGCM pickle into a PyTorch neural-component module:

```python
from tornado_gcm.model.jax_checkpoint_loader import (
    build_jax_neural_components,
    load_jax_weights,
)

model = build_jax_neural_components(resolution="2.8")
summary = load_jax_weights(
    model,
    "checkpoints/deterministic_2_8_deg.pkl",
    strict=False,
)
print(summary)
```

`load_jax_weights` maps Haiku parameter paths to PyTorch keys and transposes
Linear and Conv1d weights as needed. For a complete model wired to the
JAX-aligned feature pipeline, use:

```python
from tornado_gcm.model.jax_checkpoint_loader import (
    build_jax_aligned_experiment_model,
)

model, neural, coords, config, summary = build_jax_aligned_experiment_model(
    "checkpoints/deterministic_2_8_deg.pkl",
    device="cuda",
    resolution="2.8",
)
```

!!! warning "Missing scripts package"
    In the 0.1.0 release, the full JAX-aligned builder imports
    `tornado_gcm.scripts.data_preprocessing`, which is not shipped. The
    `build_jax_neural_components` + `load_jax_weights` path above does not
    depend on it.

## Build an initial state

WeatherBench-style data is converted to a modal primitive-equation state by
the encoder. Inputs are supplied as an in-memory `xarray.Dataset` with
`time`, `level`, `longitude`, and `latitude` dimensions:

```python
from tornado_gcm.neural.encoders_decoders import WeatherbenchToPrimitiveEncoder
from tornado_gcm.training.data_pipeline import inputs_from_xarray

wb_dict = inputs_from_xarray(dataset)      # dict of torch tensors
encoder = WeatherbenchToPrimitiveEncoder(
    coords=coords,
    physics_specs=config.physics_specs,
    reference_temperatures=reference_temperatures,
    input_pressure_levels=input_pressure_levels,
    orography=orography,
)
state = encoder(wb_dict)                   # modal core.primitive_equations.State
```

## Run a forecast

```python
from tornado_gcm.inference.runner import MPInference

runner = MPInference(model)
trajectory = runner.run(state, outer_steps=40)
```

`MPInference.run` returns a list of states, one per saved outer step. Use
`run_chunked` for long rollouts or `run_streaming` to receive results
incrementally. `SDAInference` adds adaptive and sparse variants.

## Precision configuration

```python
from tornado_gcm.precision.sda import SDAConfig

cfg = SDAConfig.from_yaml("configs/sda_neuralgcm_2.8deg_full.yaml")
print(cfg.policy.z1_compute_dtype, cfg.policy.z3_compute_dtype)
```

See [Precision Zones](precision.md) for the full YAML schema.
