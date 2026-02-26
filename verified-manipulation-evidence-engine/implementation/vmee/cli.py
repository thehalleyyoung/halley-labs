"""
Command-line interface for the VMEE pipeline.

Provides commands for running the full pipeline, individual components,
evaluation, and calibration.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import click

from vmee.config import VMEEConfig, load_config, default_config, save_config, config_hash


def setup_logging(level: str) -> None:
    """Configure logging for the pipeline."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@click.group()
@click.version_option(version="0.1.0", prog_name="vmee")
def main():
    """Verified Manipulation Evidence Engine (VMEE).

    Formally certified manipulation evidence via causal-Bayesian inference
    and adversarial evaluation.
    """
    pass


@main.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to TOML config file")
@click.option("--output", "-o", type=click.Path(), default="output", help="Output directory")
@click.option("--log-level", "-l", default="INFO", help="Logging level")
@click.option("--seed", "-s", type=int, default=None, help="Random seed override")
@click.option("--dry-run", is_flag=True, help="Validate config without running")
def run(config: Optional[str], output: str, log_level: str, seed: Optional[int], dry_run: bool):
    """Run the full VMEE evidence generation pipeline."""
    setup_logging(log_level)
    logger = logging.getLogger("vmee.cli")

    if config:
        cfg = load_config(config)
    else:
        cfg = default_config()

    cfg.output_dir = output
    cfg.log_level = log_level
    if seed is not None:
        cfg.seed = seed
    cfg.dry_run = dry_run

    errors = cfg.validate()
    if errors:
        for e in errors:
            logger.error(f"Config error: {e}")
        sys.exit(1)

    logger.info(f"VMEE pipeline starting (config hash: {config_hash(cfg)})")
    logger.info(f"Output directory: {output}")

    if dry_run:
        logger.info("Dry run mode - config is valid, exiting")
        return

    os.makedirs(output, exist_ok=True)

    from vmee.lob.simulator import LOBSimulator
    from vmee.causal.discovery import CausalDiscoveryEngine
    from vmee.bayesian.engine import BayesianInferenceEngine
    from vmee.temporal.monitor import TemporalMonitor
    from vmee.proof.bridge import ProofBridge
    from vmee.evidence.assembler import EvidenceAssembler

    start_time = time.time()

    logger.info("Step 1/6: Generating synthetic market data...")
    sim = LOBSimulator(cfg.lob)
    market_data = sim.generate_trading_day()

    logger.info("Step 2/6: Running causal discovery...")
    causal_engine = CausalDiscoveryEngine(cfg.causal)
    causal_result = causal_engine.discover(market_data)

    logger.info("Step 3/6: Running Bayesian inference...")
    bayesian_engine = BayesianInferenceEngine(cfg.bayesian)
    bayesian_result = bayesian_engine.infer(market_data, causal_result)

    logger.info("Step 4/6: Monitoring temporal specifications...")
    monitor = TemporalMonitor(cfg.temporal)
    temporal_result = monitor.monitor(market_data)

    logger.info("Step 5/6: Generating SMT proofs...")
    bridge = ProofBridge(cfg.proof)
    proof_result = bridge.generate_proofs(bayesian_result, temporal_result, causal_result)

    logger.info("Step 6/6: Assembling evidence bundle...")
    assembler = EvidenceAssembler(cfg.evidence)
    bundle = assembler.assemble(
        causal_result=causal_result,
        bayesian_result=bayesian_result,
        temporal_result=temporal_result,
        proof_result=proof_result,
        config=cfg,
    )

    bundle_path = os.path.join(output, "evidence_bundle.json")
    assembler.save_bundle(bundle, bundle_path)

    elapsed = time.time() - start_time
    logger.info(f"Pipeline complete in {elapsed:.1f}s. Bundle: {bundle_path}")


@main.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to TOML config file")
@click.option("--output", "-o", type=click.Path(), default="output", help="Output directory")
@click.option("--log-level", "-l", default="INFO", help="Logging level")
def generate_data(config: Optional[str], output: str, log_level: str):
    """Generate synthetic market data with planted manipulations."""
    setup_logging(log_level)
    logger = logging.getLogger("vmee.cli")

    cfg = load_config(config) if config else default_config()
    cfg.output_dir = output

    from vmee.lob.simulator import LOBSimulator
    from vmee.lob.manipulation import ManipulationPlanter

    sim = LOBSimulator(cfg.lob)
    data = sim.generate_trading_day()

    planter = ManipulationPlanter(seed=cfg.seed)
    data_with_manipulation = planter.plant_spoofing(data)

    os.makedirs(output, exist_ok=True)
    output_path = os.path.join(output, "market_data.json")
    with open(output_path, "w") as f:
        json.dump(data.to_dict(), f, indent=2, default=str)

    logger.info(f"Generated market data: {output_path}")


@main.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to TOML config file")
@click.option("--output", "-o", type=click.Path(), default="evaluation_results")
@click.option("--log-level", "-l", default="INFO", help="Logging level")
@click.option("--scenarios", "-n", type=int, default=None, help="Number of scenarios per type")
def evaluate(config: Optional[str], output: str, log_level: str, scenarios: Optional[int]):
    """Run evaluation and benchmarking suite."""
    setup_logging(log_level)
    logger = logging.getLogger("vmee.cli")

    cfg = load_config(config) if config else default_config()
    if scenarios:
        cfg.evaluation.num_scenarios_per_type = scenarios

    from vmee.evaluation.benchmark import BenchmarkRunner

    runner = BenchmarkRunner(cfg)
    results = runner.run_full_evaluation()

    os.makedirs(output, exist_ok=True)
    output_path = os.path.join(output, "evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Evaluation complete: {output_path}")


@main.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to TOML config file")
@click.option("--output", "-o", type=click.Path(), default="output")
@click.option("--log-level", "-l", default="INFO", help="Logging level")
@click.option("--hours", "-t", type=float, default=None, help="Training hours override")
def adversarial(config: Optional[str], output: str, log_level: str, hours: Optional[float]):
    """Run adversarial RL stress-testing."""
    setup_logging(log_level)
    logger = logging.getLogger("vmee.cli")

    cfg = load_config(config) if config else default_config()
    if hours:
        cfg.adversarial.training_hours = hours

    from vmee.adversarial.trainer import AdversarialTrainer

    trainer = AdversarialTrainer(cfg.adversarial, cfg.lob)
    results = trainer.train()

    os.makedirs(output, exist_ok=True)
    output_path = os.path.join(output, "adversarial_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Adversarial training complete: {output_path}")


@main.command()
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to TOML config file")
@click.option("--output", "-o", type=click.Path(), default="output")
@click.option("--log-level", "-l", default="INFO", help="Logging level")
def calibrate(config: Optional[str], output: str, log_level: str):
    """Run sim-to-real calibration."""
    setup_logging(log_level)
    logger = logging.getLogger("vmee.cli")

    cfg = load_config(config) if config else default_config()

    from vmee.calibration.calibrator import SimToRealCalibrator

    calibrator = SimToRealCalibrator(cfg.calibration, cfg.lob)
    results = calibrator.calibrate()

    os.makedirs(output, exist_ok=True)
    output_path = os.path.join(output, "calibration_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Calibration complete: {output_path}")


@main.command()
@click.argument("bundle_path", type=click.Path(exists=True))
@click.option("--log-level", "-l", default="INFO", help="Logging level")
def verify(bundle_path: str, log_level: str):
    """Verify an evidence bundle."""
    setup_logging(log_level)
    logger = logging.getLogger("vmee.cli")

    from vmee.evidence.verifier import BundleVerifier

    verifier = BundleVerifier()
    result = verifier.verify_bundle(bundle_path)

    if result.is_valid:
        logger.info("Bundle verification PASSED")
        for check in result.checks:
            logger.info(f"  ✓ {check.name}: {check.status}")
    else:
        logger.error("Bundle verification FAILED")
        for check in result.checks:
            status = "✓" if check.passed else "✗"
            logger.info(f"  {status} {check.name}: {check.status}")
        sys.exit(1)


@main.command()
@click.option("--output", "-o", type=click.Path(), default="default_config.toml")
def init_config(output: str):
    """Generate a default configuration file."""
    cfg = default_config()
    save_config(cfg, output)
    click.echo(f"Default configuration written to: {output}")


if __name__ == "__main__":
    main()
