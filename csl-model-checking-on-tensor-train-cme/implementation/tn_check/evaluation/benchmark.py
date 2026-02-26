"""
Benchmarking and ground-truth comparison.

For small systems (≤ 12 species), computes exact probability distributions
via sparse matrix exponentiation and compares against TT-compressed results.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

from tn_check.tensor.mps import MPS
from tn_check.tensor.mpo import MPO
from tn_check.tensor.operations import (
    mps_to_dense, mpo_to_dense, mps_distance, mps_total_variation_distance,
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a benchmark comparison."""
    model_name: str
    num_species: int
    state_space_size: int
    tt_bond_dims: list[int]
    tt_params: int
    compression_ratio: float
    max_absolute_error: float = 0.0
    total_variation_distance: float = 0.0
    l2_error: float = 0.0
    tt_time_seconds: float = 0.0
    exact_time_seconds: float = 0.0
    tt_memory_bytes: int = 0
    exact_memory_bytes: int = 0


def ground_truth_check(
    generator_mpo: MPO,
    initial_state_mps: MPS,
    time_point: float,
    tt_result: MPS,
    model_name: str = "unknown",
) -> BenchmarkResult:
    """
    Compare TT-compressed result against exact computation.

    Only feasible for small systems (state space ≤ 10^6).

    Args:
        generator_mpo: CME generator as MPO.
        initial_state_mps: Initial state as MPS.
        time_point: Time point for comparison.
        tt_result: TT-compressed result to validate.
        model_name: Name for reporting.

    Returns:
        BenchmarkResult with error metrics.
    """
    total_size = 1
    for d in generator_mpo.physical_dims_in:
        total_size *= d

    if total_size > 1_000_000:
        logger.warning(f"State space {total_size} too large for exact computation")
        return BenchmarkResult(
            model_name=model_name,
            num_species=generator_mpo.num_sites,
            state_space_size=total_size,
            tt_bond_dims=list(tt_result.bond_dims),
            tt_params=tt_result.total_params,
            compression_ratio=tt_result.compression_ratio,
        )

    # Exact computation via sparse matrix exponentiation
    t_start = time.time()
    Q_dense = mpo_to_dense(generator_mpo)
    Q_sparse = sparse.csc_matrix(Q_dense)
    p0_dense = mps_to_dense(initial_state_mps)
    p_exact = expm_multiply(Q_sparse * time_point, p0_dense)
    exact_time = time.time() - t_start

    # TT result
    p_tt = mps_to_dense(tt_result)

    # Error metrics
    max_abs_error = float(np.max(np.abs(p_exact - p_tt)))
    tv_distance = 0.5 * float(np.sum(np.abs(p_exact - p_tt)))
    l2_error = float(np.linalg.norm(p_exact - p_tt))

    result = BenchmarkResult(
        model_name=model_name,
        num_species=generator_mpo.num_sites,
        state_space_size=total_size,
        tt_bond_dims=list(tt_result.bond_dims),
        tt_params=tt_result.total_params,
        compression_ratio=tt_result.compression_ratio,
        max_absolute_error=max_abs_error,
        total_variation_distance=tv_distance,
        l2_error=l2_error,
        exact_time_seconds=exact_time,
        exact_memory_bytes=Q_dense.nbytes + p0_dense.nbytes,
    )

    logger.info(
        f"Ground truth check [{model_name}]: "
        f"max_err={max_abs_error:.2e}, TV={tv_distance:.2e}, "
        f"L2={l2_error:.2e}"
    )

    return result


def compressibility_survey(
    generator_mpo: MPO,
    initial_state_mps: MPS,
    time_point: float,
    tolerances: list[float] = None,
    max_bond_dims: list[int] = None,
) -> dict:
    """
    Survey TT-compressibility of a CME probability distribution.

    For each tolerance level, computes the effective TT rank needed
    to represent the probability distribution at the given time.

    Args:
        generator_mpo: CME generator as MPO.
        initial_state_mps: Initial state.
        time_point: Time at which to measure compressibility.
        tolerances: List of truncation tolerances.
        max_bond_dims: List of max bond dimensions to test.

    Returns:
        Dictionary with compressibility data.
    """
    if tolerances is None:
        tolerances = [1e-4, 1e-6, 1e-8]
    if max_bond_dims is None:
        max_bond_dims = [10, 20, 50, 100, 200]

    from tn_check.tensor.operations import mps_compress, mps_zip_up
    from tn_check.tensor.mpo import identity_mpo
    from tn_check.tensor.operations import mpo_addition, mpo_scalar_multiply

    results = {
        "num_species": generator_mpo.num_sites,
        "physical_dims": list(generator_mpo.physical_dims_in),
        "time_point": time_point,
        "tolerance_sweep": {},
        "bond_dim_sweep": {},
    }

    # Quick evolution for compressibility check
    # Use a few uniformization steps
    q = 1.0  # placeholder rate
    from tn_check.integrator.tdvp import TDVPIntegrator
    from tn_check.config import IntegratorConfig

    integrator = TDVPIntegrator(
        generator=generator_mpo,
        config=IntegratorConfig(dt=min(0.1, time_point / 10)),
    )

    state = initial_state_mps.copy()
    n_steps = max(1, int(time_point / integrator.config.dt))
    for step in range(min(n_steps, 100)):
        state = integrator.step(state, integrator.config.dt)

    # Sweep over tolerances
    for tol in tolerances:
        compressed, err = mps_compress(state, tolerance=tol)
        results["tolerance_sweep"][tol] = {
            "bond_dims": list(compressed.bond_dims),
            "max_bond_dim": compressed.max_bond_dim,
            "truncation_error": err,
            "total_params": compressed.total_params,
        }

    # Sweep over bond dimensions
    for chi in max_bond_dims:
        compressed, err = mps_compress(state, max_bond_dim=chi)
        results["bond_dim_sweep"][chi] = {
            "bond_dims": list(compressed.bond_dims),
            "truncation_error": err,
            "total_params": compressed.total_params,
        }

    return results
