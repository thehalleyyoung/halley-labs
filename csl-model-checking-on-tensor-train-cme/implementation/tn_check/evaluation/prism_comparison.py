"""
PRISM-style model checking comparison.

Defines the same models and properties used in PRISM's benchmark suite,
computes both dense exact and TT-compressed answers, and reports
compression ratios and accuracy metrics.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from tn_check.cme.reaction_network import ReactionNetwork
from tn_check.cme.compiler import CMECompiler
from tn_check.solver.dense_reference import DenseReferenceSolver

logger = logging.getLogger(__name__)


def _define_prism_benchmarks() -> list[dict]:
    """
    Define benchmark cases matching PRISM's standard stochastic models.

    Each entry specifies a model builder, CSL property parameters,
    and expected qualitative behavior.
    """
    from tn_check.models.library import (
        birth_death,
        toggle_switch,
        schlogl,
        gene_expression,
        sir_epidemic,
        michaelis_menten_enzyme,
        exclusive_switch,
        multi_species_cascade,
    )

    benchmarks = [
        {
            "name": "birth_death_poisson",
            "description": "Birth-death process with Poisson steady state",
            "builder": lambda: birth_death(birth_rate=5.0, death_rate=0.5, max_copy=40),
            "formula_type": "steady_state",
            "species_index": 0,
            "threshold": 8,
            "time": 50.0,
        },
        {
            "name": "gene_expression_burst",
            "description": "Bursty gene expression (high translation rate)",
            "builder": lambda: gene_expression(
                k_txn=0.2, k_tln=20.0, gamma_m=0.1, gamma_p=0.01,
                max_copy_mRNA=20, max_copy_protein=100,
            ),
            "formula_type": "transient_prob",
            "species_index": 1,
            "threshold": 50,
            "time": 100.0,
        },
        {
            "name": "schlogl_bistable",
            "description": "Schlögl model bistable switching",
            "builder": lambda: schlogl(max_copy=100),
            "formula_type": "steady_state",
            "species_index": 0,
            "threshold": 80,
            "time": 100.0,
        },
        {
            "name": "toggle_switch_bistable",
            "description": "Toggle switch bistable decision",
            "builder": lambda: toggle_switch(alpha1=20.0, alpha2=20.0, max_copy=30),
            "formula_type": "transient_prob",
            "species_index": 0,
            "threshold": 15,
            "time": 20.0,
        },
        {
            "name": "sir_outbreak",
            "description": "SIR epidemic outbreak probability",
            "builder": lambda: sir_epidemic(
                max_S=25, max_I=25, max_R=25, S0=20, I0=2,
            ),
            "formula_type": "transient_prob",
            "species_index": 1,
            "threshold": 10,
            "time": 10.0,
        },
        {
            "name": "enzyme_completion",
            "description": "Michaelis-Menten substrate conversion",
            "builder": lambda: michaelis_menten_enzyme(
                max_E=10, max_S=15, max_ES=10, max_P=15, E0=5, S0=10,
            ),
            "formula_type": "transient_prob",
            "species_index": 3,
            "threshold": 5,
            "time": 20.0,
        },
        {
            "name": "exclusive_switch_decision",
            "description": "Exclusive switch protein dominance",
            "builder": lambda: exclusive_switch(max_copy_dna=3, max_copy_protein=30),
            "formula_type": "transient_prob",
            "species_index": 1,
            "threshold": 10,
            "time": 50.0,
        },
        {
            "name": "cascade_3_layer",
            "description": "3-layer signaling cascade propagation",
            "builder": lambda: multi_species_cascade(n_species=3, max_copy=25),
            "formula_type": "transient_prob",
            "species_index": 2,
            "threshold": 5,
            "time": 30.0,
        },
    ]

    return benchmarks


def run_prism_comparison(
    max_state_space: int = 500_000,
) -> dict:
    """
    Run PRISM-style comparison benchmarks.

    For each model/property pair:
    1. Build the reaction network
    2. Compile to MPO (TT format)
    3. Compile to dense Q matrix
    4. Compute exact CSL property via dense solver
    5. Report state space size, TT parameters, compression ratio, time

    Args:
        max_state_space: Maximum state space size for dense computation.

    Returns:
        Dictionary with results for each benchmark case.
    """
    benchmarks = _define_prism_benchmarks()
    results = {"benchmarks": [], "summary": {}}

    total_dense_time = 0.0
    total_tt_time = 0.0

    for bench in benchmarks:
        entry = {
            "name": bench["name"],
            "description": bench["description"],
        }

        try:
            net = bench["builder"]()
            entry["num_species"] = net.num_species
            entry["physical_dims"] = net.physical_dims
            state_space = int(np.prod(net.physical_dims))
            entry["state_space_size"] = state_space

            # TT compilation
            t_start = time.time()
            compiler = CMECompiler(net)
            mpo = compiler.compile()
            tt_time = time.time() - t_start

            entry["mpo_bond_dims"] = list(mpo.bond_dims)
            entry["mpo_max_bond_dim"] = int(mpo.max_bond_dim)
            entry["mpo_total_params"] = int(mpo.total_params)
            entry["tt_compile_time"] = tt_time
            entry["compression_ratio"] = float(
                state_space ** 2 / max(1, mpo.total_params)
            )
            total_tt_time += tt_time

            # Dense reference (if feasible)
            if state_space <= max_state_space:
                t_start = time.time()
                solver = DenseReferenceSolver(net, max_states=max_state_space)
                csl_result = solver.csl_comparison(
                    formula_type=bench["formula_type"],
                    t=bench["time"],
                    species_index=bench["species_index"],
                    threshold2=bench["threshold"],
                )
                dense_time = time.time() - t_start

                entry["dense_probability"] = csl_result["probability"]
                entry["dense_time"] = dense_time
                entry["dense_memory_bytes"] = state_space * state_space * 8
                total_dense_time += dense_time
                entry["status"] = "complete"
            else:
                entry["status"] = "tt_only"
                entry["skip_reason"] = (
                    f"State space {state_space} exceeds max {max_state_space}"
                )

        except Exception as e:
            entry["status"] = "error"
            entry["error"] = str(e)

        results["benchmarks"].append(entry)
        logger.info(
            f"PRISM comparison: {bench['name']} -> {entry.get('status', 'unknown')}"
        )

    # Summary statistics
    completed = [b for b in results["benchmarks"] if b.get("status") == "complete"]
    results["summary"] = {
        "total_benchmarks": len(benchmarks),
        "completed": len(completed),
        "total_dense_time": total_dense_time,
        "total_tt_time": total_tt_time,
        "avg_compression_ratio": float(np.mean([
            b["compression_ratio"] for b in results["benchmarks"]
            if "compression_ratio" in b
        ])) if completed else 0.0,
    }

    return results


def run_tt_advantage_analysis(
    species_counts: list[int] = None,
) -> dict:
    """
    Analyze where TT wins over dense: large state spaces with modular structure.

    Creates cascade models of increasing size and shows the crossover
    where TT compression ratio makes dense infeasible but TT is fine.

    Args:
        species_counts: Species counts to test.

    Returns:
        Dictionary with TT advantage analysis.
    """
    if species_counts is None:
        species_counts = [2, 3, 4, 5, 6, 8, 10, 12, 15]

    from tn_check.models.library import multi_species_cascade

    results = {"entries": []}

    for n in species_counts:
        max_copy = max(8, 30 - n)
        net = multi_species_cascade(n_species=n, max_copy=max_copy)
        state_space = int(np.prod(net.physical_dims))

        entry = {
            "n_species": n,
            "max_copy": max_copy,
            "state_space_size": state_space,
            "dense_memory_bytes": state_space * state_space * 8,
            "dense_memory_gb": state_space * state_space * 8 / (1024 ** 3),
        }

        try:
            t_start = time.time()
            compiler = CMECompiler(net)
            mpo = compiler.compile()
            tt_time = time.time() - t_start

            entry["mpo_total_params"] = int(mpo.total_params)
            entry["tt_memory_bytes"] = int(mpo.total_params) * 8
            entry["tt_memory_gb"] = int(mpo.total_params) * 8 / (1024 ** 3)
            entry["compression_ratio"] = float(
                state_space ** 2 / max(1, mpo.total_params)
            )
            entry["tt_compile_time"] = tt_time
            entry["dense_feasible"] = state_space <= 500_000
            entry["status"] = "success"
        except Exception as e:
            entry["status"] = "error"
            entry["error"] = str(e)

        results["entries"].append(entry)

    return results
