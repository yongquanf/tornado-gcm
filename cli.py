# Copyright 2026 yongquan fu
# SPDX-License-Identifier: Apache-2.0
"""Command-line interface for TornadoGCM."""

from __future__ import annotations

import logging
import pathlib
from typing import Optional

import click

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(package_name="tornado-gcm")
def main() -> None:
    """TornadoGCM: PyTorch NeuralGCM with Precision-Zoned Hybrid Architecture."""


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="SDA YAML config file.")
@click.option("--checkpoint", "-k", default=None, type=click.Path(), help="Path to load/save model checkpoint.")
@click.option("--jax-weights", default=None, type=click.Path(exists=True), help="JAX checkpoint to convert and initialise from.")
@click.option("--steps", "-s", default=10000, show_default=True, help="Total training steps.")
@click.option("--batch-size", "-b", default=4, show_default=True, help="Base batch size.")
@click.option("--device", "-d", default="cuda", show_default=True, help="Torch device (cuda / cpu).")
@click.option("--use-sda/--no-sda", default=True, show_default=True, help="Enable SDA precision scheduler.")
def train(
    config: str,
    checkpoint: Optional[str],
    jax_weights: Optional[str],
    steps: int,
    batch_size: int,
    device: str,
    use_sda: bool,
) -> None:
    """Train the NeuralGCM model with mixed-precision PZHA."""
    import torch
    from tornado_gcm.precision.sda import SDAConfig, SDAController
    from tornado_gcm.precision.policy import PrecisionPolicy

    logger.info("Loading SDA config from %s", config)
    sda_cfg = SDAConfig.from_yaml(config)

    if jax_weights:
        logger.info("Converting JAX checkpoint: %s", jax_weights)
        from tornado_gcm.model.jax_checkpoint_loader import load_jax_checkpoint
        model = load_jax_checkpoint(jax_weights, device=device)
    else:
        raise click.UsageError("--jax-weights is required to initialise the model.")

    if use_sda:
        from tornado_gcm.training.sda_trainer import SDATrainer
        trainer = SDATrainer(model=model, sda_config=sda_cfg, total_steps=steps, batch_size=batch_size)
    else:
        from tornado_gcm.training.trainer import MixedPrecisionTrainer
        policy = PrecisionPolicy.from_config(sda_cfg.policy)
        trainer = MixedPrecisionTrainer(model=model, policy=policy, total_steps=steps, batch_size=batch_size)

    if checkpoint and pathlib.Path(checkpoint).exists():
        logger.info("Resuming from checkpoint: %s", checkpoint)
        trainer.load_checkpoint(checkpoint)

    logger.info("Starting training for %d steps", steps)
    trainer.train()

    if checkpoint:
        logger.info("Saving checkpoint to %s", checkpoint)
        trainer.save_checkpoint(checkpoint)


# ---------------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------------

@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="SDA YAML config file.")
@click.option("--checkpoint", "-k", required=True, type=click.Path(exists=True), help="Model checkpoint (.pt).")
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True), help="Input initial condition (zarr or netCDF4).")
@click.option("--output", "-o", "output_path", default="forecast.zarr", show_default=True, help="Output path for forecast (zarr).")
@click.option("--steps", "-s", default=40, show_default=True, help="Number of 6-hourly forecast steps.")
@click.option("--device", "-d", default="cuda", show_default=True, help="Torch device.")
@click.option("--use-sda/--no-sda", default=True, show_default=True, help="Enable SDA-controlled inference.")
def infer(
    config: str,
    checkpoint: str,
    input_path: str,
    output_path: str,
    steps: int,
    device: str,
    use_sda: bool,
) -> None:
    """Run autoregressive weather forecast from an initial condition."""
    import torch
    from tornado_gcm.precision.sda import SDAConfig

    logger.info("Loading config and checkpoint")
    sda_cfg = SDAConfig.from_yaml(config)

    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)

    if use_sda:
        from tornado_gcm.inference.sda_runner import SDAInference
        runner = SDAInference.from_state_dict(state_dict, sda_cfg, device=device)
    else:
        from tornado_gcm.inference.runner import MPInference
        runner = MPInference.from_state_dict(state_dict, device=device)

    logger.info("Loading initial condition from %s", input_path)
    from tornado_gcm.inference.production import load_initial_condition, save_forecast
    initial_state = load_initial_condition(input_path, device=device)

    logger.info("Running %d-step forecast", steps)
    trajectory = runner.rollout(initial_state, steps=steps)

    logger.info("Saving forecast to %s", output_path)
    save_forecast(trajectory, output_path)
    logger.info("Done.")


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------

@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="SDA YAML config file.")
@click.option("--steps", "-s", default=20, show_default=True, help="Number of warm-up + benchmark steps.")
@click.option("--device", "-d", default="cuda", show_default=True, help="Torch device.")
@click.option("--report", "-r", default="benchmark_report.txt", show_default=True, help="Output report file.")
def benchmark(config: str, steps: int, device: str, report: str) -> None:
    """Run precision/performance benchmarks and write a summary report."""
    from tornado_gcm.precision.sda import SDAConfig
    from tornado_gcm.precision.benchmarks import PrecisionBenchmark

    logger.info("Loading config: %s", config)
    sda_cfg = SDAConfig.from_yaml(config)

    bench = PrecisionBenchmark(sda_config=sda_cfg, device=device, n_steps=steps)
    logger.info("Running benchmark (%d steps)…", steps)
    result = bench.run()

    report_str = result.report()
    click.echo(report_str)
    pathlib.Path(report).write_text(report_str, encoding="utf-8")
    logger.info("Report saved to %s", report)


if __name__ == "__main__":
    main()
