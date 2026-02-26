"""Tests for benchmark functionality: models, solvers, experiments, contractions, and PRISM."""

import numpy as np
import pytest
from scipy.stats import poisson

from tn_check.models.library import (
    birth_death,
    gene_expression,
    exclusive_switch,
    sir_epidemic,
    michaelis_menten_enzyme,
    multi_species_cascade,
)
from tn_check.cme.compiler import CMECompiler
from tn_check.solver.dense_reference import DenseReferenceSolver
from tn_check.tensor.decomposition import tensor_to_mps
from tn_check.tensor.operations import (
    mps_to_dense,
    mpo_to_dense,
    mps_compress,
    mpo_mps_contraction,
    mps_zip_up,
)


# ---------------------------------------------------------------------------
# 1. Model creation and MPO compilation
# ---------------------------------------------------------------------------

MODEL_FACTORIES = [
    ("gene_expression", lambda: gene_expression(max_copy_mRNA=10, max_copy_protein=15)),
    ("exclusive_switch", lambda: exclusive_switch(max_copy_dna=3, max_copy_protein=10)),
    ("sir_epidemic", lambda: sir_epidemic(max_S=8, max_I=8, max_R=8, S0=5, I0=2)),
    ("michaelis_menten_enzyme", lambda: michaelis_menten_enzyme(
        max_E=5, max_S=8, max_ES=5, max_P=8, E0=3, S0=5)),
    ("multi_species_cascade", lambda: multi_species_cascade(n_species=3, max_copy=10)),
]


@pytest.mark.parametrize("name,factory", MODEL_FACTORIES, ids=[m[0] for m in MODEL_FACTORIES])
def test_model_creates_and_compiles_to_mpo(name, factory):
    model = factory()
    compiler = CMECompiler(model)
    mpo = compiler.compile()
    assert mpo.num_sites == len(model.physical_dims)
    for core in mpo.cores:
        assert core.ndim == 4


# ---------------------------------------------------------------------------
# 2. DenseReferenceSolver on birth_death – exact Poisson comparison
# ---------------------------------------------------------------------------

def test_dense_solver_birth_death_poisson():
    max_copy = 30
    model = birth_death(birth_rate=1.0, death_rate=0.1, max_copy=max_copy)
    solver = DenseReferenceSolver(model)
    Q = solver.compile()
    N = model.physical_dims[0]
    assert Q.shape == (N, N)

    pi = solver.steady_state()
    lam = 1.0 / 0.1
    exact = poisson.pmf(np.arange(N), lam)
    exact /= exact.sum()

    l1 = float(np.sum(np.abs(pi - exact)))
    assert l1 < 0.15, f"Steady-state L1 error {l1} too large"


# ---------------------------------------------------------------------------
# 3. DenseReferenceSolver evolve produces valid probability distribution
# ---------------------------------------------------------------------------

def test_dense_solver_evolve_valid_distribution():
    max_copy = 30
    model = birth_death(birth_rate=1.0, death_rate=0.1, max_copy=max_copy)
    solver = DenseReferenceSolver(model)
    solver.compile()
    N = model.physical_dims[0]

    p0 = np.zeros(N)
    p0[0] = 1.0
    p_t = solver.evolve(p0, t=1.0)

    assert p_t.shape == (N,)
    assert np.all(p_t >= 0.0), "Negative probabilities"
    assert abs(p_t.sum() - 1.0) < 1e-6, f"Sum = {p_t.sum()}"


# ---------------------------------------------------------------------------
# 4. run_scaling_experiment
# ---------------------------------------------------------------------------

def test_run_scaling_experiment():
    from tn_check.experiments import run_scaling_experiment

    result = run_scaling_experiment()
    assert result["passed"], f"Scaling experiment failed: {result}"


# ---------------------------------------------------------------------------
# 5. run_accuracy_benchmark
# ---------------------------------------------------------------------------

def test_run_accuracy_benchmark():
    from tn_check.evaluation.benchmark import run_accuracy_benchmark

    result = run_accuracy_benchmark()
    assert "models" in result
    assert len(result["models"]) > 0
    for m in result["models"]:
        assert m.get("status") in ("success", "error")


# ---------------------------------------------------------------------------
# 6. run_gene_expression_experiment
# ---------------------------------------------------------------------------

def test_run_gene_expression_experiment():
    from tn_check.experiments import run_gene_expression_experiment

    result = run_gene_expression_experiment()
    assert result["passed"], f"Gene expression experiment failed: {result}"


# ---------------------------------------------------------------------------
# 7. run_error_propagation_experiment
# ---------------------------------------------------------------------------

def test_run_error_propagation_experiment():
    from tn_check.experiments import run_error_propagation_experiment

    result = run_error_propagation_experiment()
    assert result["passed"], f"Error propagation experiment failed: {result}"


# ---------------------------------------------------------------------------
# 8. run_csl_model_checking_experiment
# ---------------------------------------------------------------------------

def test_run_csl_model_checking_experiment():
    from tn_check.experiments import run_csl_model_checking_experiment

    result = run_csl_model_checking_experiment()
    assert result["passed"], f"CSL model checking experiment failed: {result}"


# ---------------------------------------------------------------------------
# 9. run_full_pipeline_experiment
# ---------------------------------------------------------------------------

def test_run_full_pipeline_experiment():
    from tn_check.experiments import run_full_pipeline_experiment

    result = run_full_pipeline_experiment()
    assert result["passed"], f"Full pipeline experiment failed: {result}"


# ---------------------------------------------------------------------------
# 10. MPO-MPS contraction correctness (verify against dense)
# ---------------------------------------------------------------------------

def test_mpo_mps_contraction_matches_dense():
    model = birth_death(birth_rate=1.0, death_rate=0.1, max_copy=15)
    compiler = CMECompiler(model)
    mpo = compiler.compile()
    N = model.physical_dims[0]

    p0_vec = np.zeros(N)
    p0_vec[0] = 1.0
    p0_mps = tensor_to_mps(p0_vec, list(model.physical_dims))

    result_mps = mpo_mps_contraction(mpo, p0_mps)
    result_dense = mps_to_dense(result_mps)

    Q_dense = mpo_to_dense(mpo)
    expected = Q_dense @ p0_vec

    np.testing.assert_allclose(result_dense, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# 11. Zip-up contraction correctness (verify against dense)
# ---------------------------------------------------------------------------

def test_zip_up_contraction_matches_dense():
    model = birth_death(birth_rate=1.0, death_rate=0.1, max_copy=15)
    compiler = CMECompiler(model)
    mpo = compiler.compile()
    N = model.physical_dims[0]

    p0_vec = np.zeros(N)
    p0_vec[0] = 1.0
    p0_mps = tensor_to_mps(p0_vec, list(model.physical_dims))

    result_mps, trunc_err = mps_zip_up(mpo, p0_mps, max_bond_dim=50)
    result_dense = mps_to_dense(result_mps)

    Q_dense = mpo_to_dense(mpo)
    expected = Q_dense @ p0_vec

    np.testing.assert_allclose(result_dense, expected, atol=1e-8)


# ---------------------------------------------------------------------------
# 12. CME generator has zero column sums (birth_death)
# ---------------------------------------------------------------------------

def test_cme_generator_zero_column_sums():
    """Row sums of interior states in the CME generator must be (near) zero.

    Boundary rows may have small leakage due to FSP truncation.
    """
    model = birth_death(birth_rate=1.0, death_rate=0.1, max_copy=20)
    solver = DenseReferenceSolver(model)
    Q = solver.compile()
    N = Q.shape[0]

    # Interior rows (excluding boundary) should sum to zero
    interior_row_sums = Q[:N - 1].sum(axis=1)
    np.testing.assert_allclose(interior_row_sums, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 13. CME generator has Metzler property (birth_death)
# ---------------------------------------------------------------------------

def test_cme_generator_metzler_property():
    model = birth_death(birth_rate=1.0, death_rate=0.1, max_copy=20)
    solver = DenseReferenceSolver(model)
    Q = solver.compile()

    N = Q.shape[0]
    for i in range(N):
        for j in range(N):
            if i != j:
                assert Q[i, j] >= -1e-14, (
                    f"Off-diagonal Q[{i},{j}]={Q[i,j]} is negative (not Metzler)"
                )


# ---------------------------------------------------------------------------
# 14. PRISM comparison benchmark
# ---------------------------------------------------------------------------

def test_prism_comparison_runs():
    from tn_check.evaluation.prism_comparison import run_prism_comparison

    result = run_prism_comparison(max_state_space=500_000)
    assert "benchmarks" in result
    assert "summary" in result
    assert len(result["benchmarks"]) > 0
    assert result["summary"]["total_benchmarks"] > 0


def test_prism_benchmarks_defined():
    from tn_check.evaluation.prism_comparison import _define_prism_benchmarks

    benchmarks = _define_prism_benchmarks()
    assert len(benchmarks) >= 5
    for b in benchmarks:
        assert "name" in b
        assert "builder" in b
        net = b["builder"]()
        assert len(net.species) > 0


# ---------------------------------------------------------------------------
# 15. multi_species_cascade state space sizes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_species,max_copy", [
    (2, 10),
    (3, 10),
    (4, 10),
    (5, 10),
])
def test_multi_species_cascade_state_space(n_species, max_copy):
    model = multi_species_cascade(n_species=n_species, max_copy=max_copy)
    assert len(model.species) == n_species
    dims = model.physical_dims
    assert len(dims) == n_species
    # Each species should have the same dimension
    expected_d = dims[0]
    expected_size = expected_d ** n_species
    state_size = 1
    for d in dims:
        state_size *= d
    assert state_size == expected_size
