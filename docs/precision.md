# Precision Zones

TornadoGCM assigns computation to four precision zones. The policy is
declared in `precision/policy.py` and can be overridden per module or
sub-zone.

| Zone | Role | Default compute | Default storage |
|---|---|---|---|
| **Z0** | Numerical foundation: Legendre/Fourier bases, sigma coordinates, implicit matrices | FP64 | FP64 |
| **Z1** | Dynamics: spherical-harmonic transforms, primitive-equation einsums | FP32 (TF32 optional) | FP32 |
| **Z2** | Conservation fixers: dry-air mass, total energy | FP64 | FP32 |
| **Z3** | Neural networks: MLP, CNN, Transformer | BF16 | BF16 params (FP32 master) |

Fine-grained sub-zones (`Z1_SHT`, `Z1_IMEX`, `Z1_DIAGNOSTIC`, `Z2_FIXER`,
`Z3_NEURAL`, …) are resolved by `PrecisionPolicy.get_effective_dtype()`.

## Configuration

`SDAConfig.from_yaml()` reads six top-level sections: `policy`, `profiler`,
`scheduler`, `accelerator`, `storage`, and `sparse`.

```yaml
policy:
  z0_dtype: float64
  z1_compute_dtype: float32
  z1_use_tf32: true
  z1_storage_dtype: float32
  z2_compute_dtype: float64
  z2_storage_dtype: float32
  z3_compute_dtype: bfloat16
  z3_param_dtype: bfloat16
  z3_master_dtype: float32
  data_load_dtype: float32
  sht_adaptive_threshold: 42
  enable_precision_audit: true
  audit_every_n_steps: 100

scheduler:
  mode: loss_aware        # milestone | loss_aware | stability_aware
  check_interval: 100
  loss_plateau_threshold: 0.01
  loss_plateau_patience: 3
  nan_trigger_count: 2
```

Ready-to-use profiles live in `configs/`. The 2.8° "full" profile also
contains `model`, `training`, `data`, and `fixer` blocks; these are not
consumed by `SDAConfig` in this release.

## Runtime casting

Use `precision/zone_cast.py` at explicit boundaries:

```python
from tornado_gcm.precision.zone_cast import zone_cast, f64_math, einsum_highest

# Cast a tensor between zone dtypes
y = zone_cast(x, PrecisionZone.Z3_NEURAL_NETWORK,
              PrecisionZone.Z1_DYNAMICS_CORE, policy)

# Run a function with FP64 inputs, returning the original dtype
@f64_math
def conserve(x):
    ...

# Einsum with FP64 intermediates
out = einsum_highest("...hml,...hml->...", a, b)
```

`PrecisionMonitor` optionally audits precision loss on downcasts.

## SDA control plane

`SDAController` coordinates three components:

1. `SDAProfiler` collects gradient, spectral, and per-zone timing metrics.
2. `PrecisionScheduler` decides when to change phase.
3. The active policy is hot-swapped and TF32 settings are re-applied.

Scheduling modes:

- `milestone` — fixed phase fractions (60% / 30% / 10% by default).
- `loss_aware` — advance on a loss plateau.
- `stability_aware` — escalate precision on NaN or gradient explosion.

## Automatic zone discovery

`precision/zone_discovery.py` can infer a policy instead of hand-writing
one. It supports spectral clustering over per-module numerical drift, a
hardware-level greedy assignment, and a DAG path that exports the model to
FX, groups tensors, classifies sub-zones, and solves a budget-constrained
assignment. Results convert to a `PrecisionPolicy` or `SDAConfig`.

## Enforcement notes

The policy is partly declarative in this release:

- BF16 neural execution is applied with `torch.autocast("cuda", bfloat16)`
  in the trainer.
- TF32 is a global PyTorch backend switch, not a per-zone setting.
- `z3_param_dtype` and `z3_master_dtype` describe the intended scheme;
  training keeps FP32 weights unless `bake_precision` or module precision
  hooks are applied.
- `sht_adaptive_threshold` selects FP64 spherical-harmonic transforms for
  wavenumbers above the threshold unless the transform is explicitly
  rewired.

Treat these as implementation details to verify when measuring precision
or performance.
