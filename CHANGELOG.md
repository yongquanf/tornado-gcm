# Changelog

All notable changes to TornadoGCM are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versions follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Added
- Planned: FP8 support for Z3 neural zone (H100 only)
- Planned: ONNX export for deployment

---

## [0.1.0] - 2026-04-20

### Added
- Pure PyTorch reimplementation of NeuralGCM spectral primitive-equations dycore
- **Precision-Zoned Hybrid Architecture (PZHA)**: four computation zones (Z0 FP64, Z1 TF32, Z2 FP64, Z3 BF16)
- **SDA control plane**: Self-adaptive Dynamic Accuracy scheduler (`precision/sda.py`, `precision/scheduler.py`)
- Spherical Harmonic Transform (SHT) with optional Triton-fused kernels (`precision/accelerator/triton_sht.py`)
- JAX checkpoint loader — load pre-trained NeuralGCM weights from original JAX format (`model/jax_checkpoint_loader.py`)
- Mixed-precision inference runners: `MPInference` and `SDAInference`
- Variable-precision 3-phase training: `MixedPrecisionTrainer`, `SDATrainer`
- Distributed training: DDP (`distributed/ddp.py`) and DTensor sharding (`distributed/dtensor_sharding.py`)
- SDA YAML configs for 0.7°, 1.4°, and 2.8° resolutions
- CLI entry point with `train`, `infer`, `benchmark` subcommands
- Precision benchmarking and zone-discovery utilities

### Changed
- Internal package namespace renamed from `pytorch_src` to `tornado_gcm`

[Unreleased]: https://github.com/yongquanf/tornado-gcm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yongquanf/tornado-gcm/releases/tag/v0.1.0
