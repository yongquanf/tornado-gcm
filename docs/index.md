# TornadoGCM

**Pure-PyTorch NeuralGCM with a Precision-Zoned Hybrid Architecture (PZHA).**

TornadoGCM is an independent PyTorch reimplementation of
[NeuralGCM](https://github.com/google-research/neuralgcm): a spectral
primitive-equation dycore coupled to learned physics parameterizations.
Computation is organized into four precision zones, and an optional SDA
control plane adapts precision during training and inference.

!!! note "Independent research port"
    This project is not affiliated with or endorsed by Google Research or
    DeepMind. NeuralGCM is described in Kochkov et al., *Nature* 632,
    1060–1066 (2024). See the repository's `LICENSE` and `LICENSE-WEIGHTS`
    files for licensing terms.

## Highlights

- **Pure PyTorch** — GPU execution, DDP/FSDP helpers, and `torch.compile`
  compatibility.
- **PZHA mixed precision** — Z0 FP64 numerical foundation, Z1 TF32 dynamics,
  Z2 FP64 conservation fixers, and Z3 BF16 neural networks.
- **SDA control plane** — profiling, phase scheduling, and runtime precision
  policies driven by YAML configuration.
- **JAX checkpoint conversion** — load pre-trained NeuralGCM weights directly
  from the original JAX pickle format.
- **Optional Triton kernels** — fused spherical-harmonic transform and
  conservation-reduction kernels on CUDA devices.

## Install

```bash
pip install tornado-gcm
```

Optional extras:

```bash
pip install "tornado-gcm[data]"   # xarray / zarr / netCDF4
pip install "tornado-gcm[gpu]"    # Triton kernels
```

## A first look

```python
from tornado_gcm.inference.runner import MPInference
from tornado_gcm.precision.sda import SDAConfig

cfg = SDAConfig.from_yaml("configs/sda_neuralgcm_2.8deg_full.yaml")
runner = MPInference(model)
trajectory = runner.run(initial_state, outer_steps=40)
```

`model` is a `NeuralGCMModel` and `initial_state` is a modal
`core.primitive_equations.State`. See [Quick Start](quickstart.md) for
checkpoint loading, state construction, and the CLI.

## Documentation map

- [Installation](installation.md)
- [Quick Start](quickstart.md)
- [Architecture](architecture.md)
- [Precision Zones](precision.md)
- [API Reference](api.md)
- [Development](development.md)
- [Changelog](changelog.md)
