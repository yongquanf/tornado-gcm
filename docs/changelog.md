# Changelog

The authoritative changelog lives at the repository root:
[CHANGELOG.md](https://github.com/yongquanf/tornado-gcm/blob/main/CHANGELOG.md).

## 0.1.0

- Pure PyTorch reimplementation of the NeuralGCM spectral
  primitive-equations dycore.
- Precision-Zoned Hybrid Architecture with four zones (Z0 FP64, Z1 TF32,
  Z2 FP64, Z3 BF16).
- SDA control plane with profiling, scheduling, and runtime precision
  policies.
- JAX checkpoint conversion for pre-trained NeuralGCM weights.
- Mixed-precision inference runners and training loops.
- DDP/FSDP and DTensor distributed helpers.
- SDA YAML profiles for 0.7°, 1.4°, and 2.8° grids.
