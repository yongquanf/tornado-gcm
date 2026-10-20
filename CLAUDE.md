# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

TornadoGCM is a pure-PyTorch reimplementation of NeuralGCM: a spectral primitive-equation dycore coupled to learned physics, organized as a **Precision-Zoned Hybrid Architecture (PZHA)** with four precision zones and an optional **SDA** control plane that adapts precision during training/inference. The Google JAX NeuralGCM codebase is the reference; most module and class names are intentionally 1:1 with it (`neuralgcm` → `neural/`, `dinosaur` → `core/`), and pre-trained JAX pickles are converted at load time. The `README.md` gives the user-facing overview.

## Commands

```bash
# Install (the checkout directory must be named `tornado_gcm` — see Packaging)
pip install -e ".[dev,data,gpu]"
pre-commit install

# Format / lint (configs in pyproject.toml and .pre-commit-config.yaml)
pre-commit run --all-files
black . && isort . && ruff check .

# Compile smoke check — the check CI actually enforces
python -m compileall -q .

# Tests: there is no tests/ directory yet, so `pytest` has nothing to collect
# and CI skips its pytest step. Once tests exist:
pytest
pytest tests/test_file.py::test_name -v
pytest --cov=tornado_gcm --cov-report=term-missing

# CLI (works without installing; cli.py bootstraps the package)
python cli.py --help
python cli.py benchmark -c configs/sda_neuralgcm_2.8deg_full.yaml -s 20 -d cuda -r bench.txt

# Documentation (MkDocs Material; Read the Docs builds from .readthedocs.yaml)
pip install -r docs/requirements.txt
mkdocs serve
mkdocs build --strict

# Regenerate the end-to-end architecture diagram (runs a collision check)
python docs/assets/diagram_src/build_architecture_svg.py

# Release artifacts (wheel + staged sdist)
python -m build --wheel --no-isolation --outdir dist
python .github/scripts/build_sdist.py --output dist
python -m twine check dist/*
```

Releases are tag-driven: pushes to `main` run `.github/workflows/release.yml`
(build + docs), while pushing a `vX.Y.Z` tag publishes to PyPI via Trusted
Publishing and triggers the release docs build. PyPI versions are immutable,
so every release needs a version bump in `pyproject.toml` and a matching tag.

## Packaging and imports

- The repository root **is** the `tornado_gcm` package; there is no `src/` nesting. `pyproject.toml` lists packages explicitly with `package-dir = { "" = ".." }`, so the checkout directory must be named exactly `tornado_gcm`. Do not replace this with recursive package discovery: the parent directory is an arbitrary workspace, and discovery scans all of it.
- `python -m build --wheel` works from the repo root. `python -m build --sdist` does not produce a usable sdist because setuptools cannot emit files outside the project root; for a release, build the sdist from a staged standard-layout copy (a `tornado_gcm/` subdirectory containing the package).
- Documentation is MkDocs Material (`mkdocs.yml`, `docs/`), built by Read the Docs from `.readthedocs.yaml` (which installs `docs/requirements.txt`).
- All internal imports are absolute (`from tornado_gcm.core import ...`); relative imports are not used.
- `cli.py` injects a synthetic `tornado_gcm` module into `sys.modules`, so `python cli.py ...` works from an uninstalled source tree.
- Every Python file starts with the two-line copyright/SPDX header.

## Architecture

### One model step

```
WeatherBench dict (u, v, t, z, tracers)
  → WeatherbenchToPrimitiveEncoder            # nodal → modal
  → core.primitive_equations.State             # spectral coefficients
  → NeuralGCMModel.step()                      # one physics step
      1. learned physics parameterization                     Z3 / BF16
      2. dycore + IMEX-RK3-SIL3 × dycore_substeps, + filter  Z1 / TF32
      3. conservation fix (dry-air mass, total energy)       Z2 / FP64
  → PrimitiveToWeatherbenchDecoder             # modal → nodal
```

`NeuralGCMModel.forward()` returns `(final_state, trajectory)`. The physics tendency is computed **once per outer step** and held constant across the dycore substeps; filtering happens after each substep. `inference.runner.MPInference.run()` is a no-grad wrapper around this loop.

### State and grids (`core/`)

- `core.primitive_equations.State` is **modal**: `vorticity`, `divergence`, `temperature_variation` are `(layers, m, l)`; `log_surface_pressure` is `(1, m, l)`; `tracers` is a dict of the same shape; `sim_time` is a scalar. The `m` axis is ordered `[0, +1, -1, +2, -2, ...]`, `l` is ascending, and entries with `|m| > l` are structural zeros — the modal array is not square-symmetric.
- Nodal tensors use `(..., lon, lat)` with the vertical axis at `-3`; layers run top of atmosphere → surface. Neural code converts modal→nodal internally.
- `core.coordinate_systems.CoordinateSystem` combines a `spherical_harmonic.Grid` with a vertical coordinate (`SigmaCoordinates`, `HybridCoordinates` for `P = a + b·P_surface`, `LayerCoordinates`, or `PressureCoordinates`). Coordinate arithmetic is FP64 (Z0).
- `Grid.construct(max_wavenumber, gaussian_nodes)` and the `T21`/`T42`/`T213`/`T425` factories build grids. Key transforms: `to_nodal`/`to_modal`, `d_dlon`, `cos_lat_d_dlat`, `div_cos_lat`, `curl_cos_lat`, `laplacian`, `clip_wavenumbers` (drops the top wavenumber after tendencies to control aliasing).
- Model quantities are nondimensionalized through `units.SimUnits` / `scales.py` (`NEURALGCM_V1_SCALE`). The JAX-aligned builder uses `_JAX_DT_NONDIM = 0.525024` (≈ 1 hour) and 5 dycore substeps.

### Model assembly and checkpoint conversion (`model/`)

- `model/api.py::NeuralGCMModel` (configured by `ModelConfig`) is the single coupling hub; it owns the dycore, parameterization, filters, conservation fixer, and optional encoder/decoder/stochastic/forcing.
- `model/atmosphere.py::AtmosphericModel` is an encoder/decoder/stochastic facade that delegates `step` to `NeuralGCMModel`; it is not wired into the current training/inference paths. `model/legacy.py::PressureLevelModel` is legacy. `model/__init__.py` exports only `NeuralGCMModel`.
- `model/jax_checkpoint_loader.py` implements the checkpoint path: JAX pickle → flat `params/...` dict (`load_jax_checkpoint`), Haiku-path→PyTorch-key mapping (`build_weight_mapping`), then transposition/permutation (`_convert_param`; Linear `(in,out)→(out,in)`, Conv1d `(k,in,out)→(out,in,k)`). `load_jax_weights(..., strict=False)` loads into `JaxNeuralComponents`.
- `build_jax_aligned_experiment_model(checkpoint_path, device, resolution)` is the full-model entry point and returns `(model, neural_components, coords, config, summary)`.

### Neural package (`neural/`)

- Mirrors JAX neuralgcm. The production parameterization is `JaxAlignedDivCurlParameterization`: `CombinedFeatures` → `SequentialTransform(ShiftAndNormalize → InverseLevelScale → SoftClip)` → `pack_features` → `ForwardTower(Epd)` → inverse normalization/tendency transform → div/curl → modal `State`.
- The EPD input width is fixed at **2365 features**, and dict keys are packed in **alphabetically sorted order**; both are load-bearing. `ForwardTower` acts on the feature axis `-3`; `ColumnTower` flattens level×feature per column.
- `neural/fixers.py` holds the Z2 conservation fixers (`GlobalDryAirMassFixer`, `GlobalEnergyFixer`, `F64ConservationFixer`, Kahan summation).

### Precision (`precision/`)

- Zones are declared in `precision/policy.py`: **Z0** numerical foundation (FP64 bases/coords/implicit matrices), **Z1** dynamics (SHT/einsums; FP32 storage with a TF32 toggle), **Z2** conservation fixers (FP64 compute / FP32 storage), **Z3** neural networks (BF16 compute/params, FP32 master).
- Runtime boundaries go through `precision/zone_cast.py`: `zone_cast`, `zone_boundary`, `f64_math`, `einsum_highest` (FP64 intermediates). Fine sub-zones (`Z1_SHT`, `Z1_IMEX`, `Z1_DIAGNOSTIC`, …) resolve through `PrecisionPolicy.get_effective_dtype()`.
- `precision/sda.py` provides `SDAConfig.from_yaml()` and `SDAController` (profiler → scheduler → precision hot-swap every `check_interval`). `precision/zone_discovery.py` optionally infers per-module/sub-zone dtypes from a `torch.export`/FX graph (spectral clustering + sensitivity analysis) and exports a policy. `scheduler.py` supports `milestone`, `loss_aware`, and `stability_aware` modes.
- `precision/storage/`, `precision/sparse/`, and `precision/accelerator/` are optional data-plane features (mixed-precision storage, sparse spectral splits, `torch.compile` wrapper, optional Triton SHT/fixer kernels for CUDA).
- Enforcement is partial by design: real BF16 for the NN happens via `torch.autocast("cuda", dtype=torch.bfloat16)` in the trainer; TF32 is a global backend switch, not per-zone; several `z3_param_dtype`/`z3_master_dtype` fields are declarative unless explicit hooks or `bake_precision` are applied.

### Training, inference, distributed

- `training/trainer.py::MixedPrecisionTrainer(model, optimizer, loss_fn, policy=..., max_grad_norm=..., ema_num_steps=...)` exposes `train_step(batch, rollout_steps=1)`; it has **no** `train()`, `fit()`, or checkpoint methods. `training/sda_trainer.py::SDATrainer(model, optimizer, loss_fn, sda_config=...)` wraps it and adds `fit(data_iter, total_steps, ...)` plus the SDA controller. Checkpoint I/O lives separately in `training/checkpointing.py`.
- Data: `training/data_pipeline.inputs_from_xarray(ds)` consumes an in-memory `xarray.Dataset` with dims `time/level/longitude/latitude`; `training/data_loading.TrajectoryDataset` takes dicts of numpy arrays plus timestamps. There is no zarr/netCDF file opener in this package — callers must load the dataset.
- `inference/runner.py::MPInference(model, policy=None, monitor=None)` provides `run`, `run_chunked`, `run_streaming`; `SDAInference` adds adaptive/sparse variants. `inference/production.py::InferenceRunner(engine, config, forcing_fn, post_process_fn)` adds chunking, NaN guard/rollback, and `NumpyWriter`/`ZarrWriter`/`AsyncZarrWriter` output. Initial conditions must be encoded explicitly.
- `distributed/` is standalone: `setup_distributed`/`wrap_ddp`/`wrap_fsdp` and DTensor sharding (`NEURALGCM_DTENSOR_ENABLED`). Only `all_reduce_metrics` and `get_distributed_sampler` are wired into the trainer.

### Configs and data

- `configs/sda_neuralgcm_{0.7,1.4,2.8}.yaml` plus `sda_debug_fp32.yaml`. `SDAConfig.from_yaml()` reads only `policy`, `profiler`, `scheduler`, `accelerator`, `storage`, and `sparse`. Other top-level blocks (`model`, `training`, `data`, `fixer` in the 2.8° "full" config) are silently ignored — they document an intended fuller pipeline, not current behavior.
- `data/ecmwf137_hybrid_levels.csv` and `data/ufs127_hybrid_levels.csv` are vertical-level definitions.

## Conventions and invariants

- Python ≥ 3.9, `from __future__ import annotations`, line length 100. Ruff selects `E,F,W,I,UP`; `E501` and `F401` are intentionally ignored (line length is black's job; `__init__` re-exports).
- Static tensors are registered as `nn.Module` buffers (or through a `_Buf` submodule), and `State`/`CoordinateSystem` are registered as pytrees so Dynamo/`torch.compile` can trace without graph breaks. Avoid `cached_property` writes in traced code; guard device moves with `torch.compiler.is_compiling()`.
- Do not hand-roll spectral transforms: go through `Grid`/`RealSphericalHarmonics`, and call `clip_wavenumbers` after tendency computations.
- Use `zone_cast` / `f64_math` / `einsum_highest` when crossing a meaningful precision boundary instead of ad-hoc `.to(dtype)`.
- `State` arithmetic preserves `sim_time`; all tracers must keep identical keys across `tree_map`, addition, and tendencies.
- Sort dict keys before packing features; the 2365-channel EPD layout depends on it.

## Known gaps in this snapshot

These are verified current-state issues, not design intent — check them before relying on the corresponding path:

- `tornado_gcm.scripts.*` is referenced but absent from this release: `model/jax_checkpoint_loader.py` imports `scripts.data_preprocessing.build_coords_from_csv`, `training/evaluation.py` imports `batch_to_train_pair`, and `precision/community_tuner.py` imports `scripts.experiments.exp_utils`. Those code paths fail at call time.
- The CLI is out of sync with the library. `benchmark` is wired (though `PrecisionBenchmark` measures a synthetic 1024² matmul, not the model). `train` passes `total_steps`/`batch_size` to trainers whose constructors require `optimizer`/`loss_fn`, and calls nonexistent `train`/`load_checkpoint`/`save_checkpoint`. `infer` calls nonexistent `MPInference.from_state_dict`/`rollout` and `load_initial_condition`/`save_forecast`. Use the library APIs documented above.
- README examples drift from the code (`AtmosphereModel` vs `AtmosphericModel`; `MPInference(device=...)`; `rollout`).
- Time units are inconsistent: the JAX-aligned model path is nondimensional (≈0.525 ≈ 1 h outer step), while `training/evaluation.py` and the CLI assume 6-hour steps, and YAML `dt:` values are not read by `SDAConfig`. Verify lead-time semantics before comparing forecasts.
- `precision/accelerator/compile_backend.py::register_state_pytree()` re-registers `State` even though `core/primitive_equations.py` already registers it; the duplicate registration raises `ValueError` despite the function's "idempotent" docstring.
- `distributed/ddp.py` imports `TransformerEncoderBlock` from `neural/transformer_layers.py`, which defines `TransformerBlock`/`TransformerEncoder`/`TransformerDecoder` — the FSDP auto-wrap path with that class fails.
- There is no `tests/` directory; CI's pytest step is skipped, and its lint/mypy steps are non-blocking.
