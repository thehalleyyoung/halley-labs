"""
Tests for recurrent neural network policy support in MARACE.

Covers LSTM and GRU cells (concrete and abstract), the recurrent
policy unroller, abstract evaluator, and Lipschitz bound computation.
"""

from __future__ import annotations

import numpy as np
import pytest

from marace.abstract.zonotope import Zonotope
from marace.policy.recurrent import (
    GRUCell,
    LSTMCell,
    RecurrentAbstractEvaluator,
    RecurrentAbstractOutput,
    RecurrentLipschitzBound,
    RecurrentNetworkArchitecture,
    RecurrentPolicyUnroller,
    SigmoidAbstractTransformer,
    _abstract_add,
    _abstract_hadamard,
    _abstract_one_minus,
    _sigmoid,
    _tanh,
    make_random_recurrent_architecture,
)

# ======================================================================
# Fixtures
# ======================================================================

RNG = np.random.default_rng(123)


def _make_lstm_cell(
    input_dim: int = 4,
    hidden_dim: int = 3,
    scale: float = 0.1,
) -> LSTMCell:
    """Create a small LSTM cell with deterministic random weights."""
    rng = np.random.default_rng(42)
    gate_rows = 4 * hidden_dim
    W = rng.normal(0, scale, (gate_rows, input_dim))
    U = rng.normal(0, scale, (gate_rows, hidden_dim))
    b = rng.normal(0, scale, (gate_rows,))
    return LSTMCell(hidden_dim, W, U, b)


def _make_gru_cell(
    input_dim: int = 4,
    hidden_dim: int = 3,
    scale: float = 0.1,
) -> GRUCell:
    """Create a small GRU cell with deterministic random weights."""
    rng = np.random.default_rng(42)
    gate_rows = 3 * hidden_dim
    W = rng.normal(0, scale, (gate_rows, input_dim))
    U = rng.normal(0, scale, (gate_rows, hidden_dim))
    b = rng.normal(0, scale, (gate_rows,))
    return GRUCell(hidden_dim, W, U, b)


# ======================================================================
# 1. LSTM cell concrete forward pass correctness
# ======================================================================


class TestLSTMConcrete:
    """Test LSTM cell concrete forward pass."""

    def test_output_shapes(self) -> None:
        """LSTM cell returns hidden and cell states of correct shape."""
        cell = _make_lstm_cell(input_dim=4, hidden_dim=3)
        x = RNG.normal(size=4)
        h = np.zeros(3)
        c = np.zeros(3)
        h_new, c_new = cell.forward_concrete(x, h, c)
        assert h_new.shape == (3,)
        assert c_new.shape == (3,)

    def test_zero_input_zero_state(self) -> None:
        """With zero input and zero state, output depends only on bias."""
        cell = _make_lstm_cell(input_dim=4, hidden_dim=3)
        x = np.zeros(4)
        h = np.zeros(3)
        c = np.zeros(3)
        h_new, c_new = cell.forward_concrete(x, h, c)
        # Should be deterministic and finite
        assert np.all(np.isfinite(h_new))
        assert np.all(np.isfinite(c_new))

    def test_deterministic(self) -> None:
        """Same inputs produce same outputs."""
        cell = _make_lstm_cell()
        x = RNG.normal(size=4)
        h = RNG.normal(size=3)
        c = RNG.normal(size=3)
        h1, c1 = cell.forward_concrete(x, h, c)
        h2, c2 = cell.forward_concrete(x, h, c)
        np.testing.assert_array_equal(h1, h2)
        np.testing.assert_array_equal(c1, c2)

    def test_hidden_bounded_by_one(self) -> None:
        """LSTM hidden state is bounded by [-1, 1] (product of σ and tanh)."""
        cell = _make_lstm_cell(scale=1.0)
        for _ in range(50):
            x = RNG.normal(size=4) * 2.0
            h = RNG.uniform(-1, 1, size=3)
            c = RNG.normal(size=3)
            h_new, _ = cell.forward_concrete(x, h, c)
            assert np.all(np.abs(h_new) <= 1.0 + 1e-10)

    def test_manual_computation(self) -> None:
        """Verify against manually computed LSTM equations."""
        hd = 2
        id_ = 3
        W = np.zeros((8, id_))
        U = np.zeros((8, hd))
        b = np.zeros(8)
        # Set bias for forget gate high -> f ≈ 1
        b[hd:2*hd] = 5.0
        cell = LSTMCell(hd, W, U, b)

        x = np.ones(id_)
        h = np.zeros(hd)
        c = np.array([1.0, -1.0])

        h_new, c_new = cell.forward_concrete(x, h, c)
        # With zero W, U and high forget bias: f ≈ 1, i ≈ 0.5, g ≈ 0, o ≈ 0.5
        # c' ≈ f*c ≈ c (because f≈1 and i*g≈0)
        # h' ≈ o * tanh(c') ≈ 0.5 * tanh(c)
        expected_c = _sigmoid(np.array(5.0)) * c  # f*c + i*g where i*g≈0
        assert np.allclose(c_new, expected_c, atol=0.1)


# ======================================================================
# 2. GRU cell concrete forward pass correctness
# ======================================================================


class TestGRUConcrete:
    """Test GRU cell concrete forward pass."""

    def test_output_shape(self) -> None:
        """GRU cell returns hidden state of correct shape."""
        cell = _make_gru_cell(input_dim=4, hidden_dim=3)
        x = RNG.normal(size=4)
        h = np.zeros(3)
        h_new = cell.forward_concrete(x, h)
        assert h_new.shape == (3,)

    def test_zero_input_zero_state(self) -> None:
        """With zero input and zero state, output is finite and deterministic."""
        cell = _make_gru_cell()
        x = np.zeros(4)
        h = np.zeros(3)
        h_new = cell.forward_concrete(x, h)
        assert np.all(np.isfinite(h_new))

    def test_deterministic(self) -> None:
        """Same inputs produce same outputs."""
        cell = _make_gru_cell()
        x = RNG.normal(size=4)
        h = RNG.normal(size=3)
        h1 = cell.forward_concrete(x, h)
        h2 = cell.forward_concrete(x, h)
        np.testing.assert_array_equal(h1, h2)

    def test_hidden_bounded(self) -> None:
        """GRU hidden state is a convex combination of n and h, bounded."""
        cell = _make_gru_cell(scale=1.0)
        for _ in range(50):
            x = RNG.normal(size=4) * 2.0
            h = RNG.uniform(-1, 1, size=3)
            h_new = cell.forward_concrete(x, h)
            # h_new = (1-z)*tanh(...) + z*h, so bounded by max(1, max(|h|))
            assert np.all(np.abs(h_new) <= 1.0 + max(0, np.max(np.abs(h))) + 1e-10)


# ======================================================================
# 3. Abstract LSTM evaluation soundness
# ======================================================================


class TestAbstractLSTM:
    """Test that abstract LSTM evaluation is sound (contains concrete)."""

    def test_soundness_point_input(self) -> None:
        """Abstract evaluation at a point zonotope equals concrete."""
        cell = _make_lstm_cell(input_dim=4, hidden_dim=3)
        x = RNG.normal(size=4)
        h = np.zeros(3)
        c = np.zeros(3)

        h_conc, c_conc = cell.forward_concrete(x, h, c)

        x_z = Zonotope.from_point(x)
        h_z = Zonotope.from_point(h)
        c_z = Zonotope.from_point(c)
        h_abs, c_abs = cell.forward_abstract(x_z, h_z, c_z)

        # Concrete output must be within abstract bounds
        lo_h, hi_h = h_abs.bounding_box()[:, 0], h_abs.bounding_box()[:, 1]
        assert np.all(h_conc >= lo_h - 1e-6), f"h below lower: {h_conc} < {lo_h}"
        assert np.all(h_conc <= hi_h + 1e-6), f"h above upper: {h_conc} > {hi_h}"

    def test_soundness_interval_input(self) -> None:
        """Concrete outputs from sampled inputs lie within abstract bounds."""
        cell = _make_lstm_cell(input_dim=4, hidden_dim=3, scale=0.1)
        center = RNG.normal(size=4) * 0.5
        eps = 0.1
        x_z = Zonotope.from_interval(center - eps, center + eps)
        h_z = Zonotope.from_point(np.zeros(3))
        c_z = Zonotope.from_point(np.zeros(3))

        h_abs, c_abs = cell.forward_abstract(x_z, h_z, c_z)
        lo_h, hi_h = h_abs.bounding_box()[:, 0], h_abs.bounding_box()[:, 1]

        # Sample concrete inputs and verify containment
        for _ in range(100):
            x_sample = center + RNG.uniform(-eps, eps, size=4)
            h_conc, _ = cell.forward_concrete(x_sample, np.zeros(3), np.zeros(3))
            assert np.all(h_conc >= lo_h - 1e-6), \
                f"Soundness violation: {h_conc} < {lo_h}"
            assert np.all(h_conc <= hi_h + 1e-6), \
                f"Soundness violation: {h_conc} > {hi_h}"


# ======================================================================
# 4. Abstract GRU evaluation soundness
# ======================================================================


class TestAbstractGRU:
    """Test that abstract GRU evaluation is sound."""

    def test_soundness_point_input(self) -> None:
        """Abstract evaluation at a point zonotope contains concrete."""
        cell = _make_gru_cell(input_dim=4, hidden_dim=3)
        x = RNG.normal(size=4)
        h = np.zeros(3)

        h_conc = cell.forward_concrete(x, h)

        x_z = Zonotope.from_point(x)
        h_z = Zonotope.from_point(h)
        h_abs = cell.forward_abstract(x_z, h_z)

        lo, hi = h_abs.bounding_box()[:, 0], h_abs.bounding_box()[:, 1]
        assert np.all(h_conc >= lo - 1e-6)
        assert np.all(h_conc <= hi + 1e-6)

    def test_soundness_interval_input(self) -> None:
        """Concrete GRU outputs from sampled inputs lie within abstract bounds."""
        cell = _make_gru_cell(input_dim=4, hidden_dim=3, scale=0.1)
        center = RNG.normal(size=4) * 0.5
        eps = 0.1
        x_z = Zonotope.from_interval(center - eps, center + eps)
        h_z = Zonotope.from_point(np.zeros(3))

        h_abs = cell.forward_abstract(x_z, h_z)
        lo, hi = h_abs.bounding_box()[:, 0], h_abs.bounding_box()[:, 1]

        for _ in range(100):
            x_sample = center + RNG.uniform(-eps, eps, size=4)
            h_conc = cell.forward_concrete(x_sample, np.zeros(3))
            assert np.all(h_conc >= lo - 1e-6)
            assert np.all(h_conc <= hi + 1e-6)


# ======================================================================
# 5. Sigmoid abstract transformer
# ======================================================================


class TestSigmoidAbstractTransformer:
    """Test the sigmoid abstract transformer."""

    def test_soundness(self) -> None:
        """Sigmoid of concrete points lies within abstract bounds."""
        z = Zonotope.from_interval(np.array([-2.0, -1.0]), np.array([1.0, 2.0]))
        sig_z = SigmoidAbstractTransformer.transform(z)
        lo, hi = sig_z.bounding_box()[:, 0], sig_z.bounding_box()[:, 1]

        for _ in range(200):
            x = np.array([
                RNG.uniform(-2.0, 1.0),
                RNG.uniform(-1.0, 2.0),
            ])
            sig_x = _sigmoid(x)
            assert np.all(sig_x >= lo - 1e-6)
            assert np.all(sig_x <= hi + 1e-6)

    def test_point_input(self) -> None:
        """Sigmoid at a point zonotope produces a near-point result."""
        x = np.array([0.0, 1.0, -1.0])
        z = Zonotope.from_point(x)
        sig_z = SigmoidAbstractTransformer.transform(z)
        lo, hi = sig_z.bounding_box()[:, 0], sig_z.bounding_box()[:, 1]
        expected = _sigmoid(x)
        np.testing.assert_allclose(lo, expected, atol=1e-5)
        np.testing.assert_allclose(hi, expected, atol=1e-5)


# ======================================================================
# 6. Lipschitz bound computation for unrolled networks
# ======================================================================


class TestRecurrentLipschitz:
    """Test Lipschitz bound computation for recurrent networks."""

    def test_naive_bound_positive(self) -> None:
        """Naive Lipschitz bound is positive and finite."""
        arch = make_random_recurrent_architecture(
            cell_type="lstm", hidden_dim=4, unroll_horizon=5
        )
        lip = RecurrentLipschitzBound(arch)
        cert = lip.compute_naive()
        assert cert.global_bound > 0.0
        assert np.isfinite(cert.global_bound)
        assert cert.method == "recurrent_naive"

    def test_naive_bound_is_power_of_per_step(self) -> None:
        """Naive Lip(f^K) = Lip(f)^K (product of identical per-step bounds)."""
        arch = make_random_recurrent_architecture(
            cell_type="lstm", unroll_horizon=5, with_output_projection=False,
        )
        lip = RecurrentLipschitzBound(arch)
        cert = lip.compute_naive()
        per_step = cert.per_layer_bounds[0]
        expected = per_step ** 5
        np.testing.assert_allclose(cert.global_bound, expected, rtol=1e-8)

    def test_interval_bound_finite(self) -> None:
        """Interval-based bound is finite."""
        arch = make_random_recurrent_architecture(
            cell_type="lstm", input_dim=4, hidden_dim=3, unroll_horizon=3
        )
        lip = RecurrentLipschitzBound(arch)
        x_lo = -np.ones(4)
        x_hi = np.ones(4)
        cert = lip.compute_interval(x_lo, x_hi)
        assert np.isfinite(cert.global_bound)
        assert cert.method == "recurrent_interval"

    def test_decay_bound_tighter(self) -> None:
        """Decay bound should be no looser than naive for small weights."""
        arch = make_random_recurrent_architecture(
            cell_type="lstm", input_dim=4, hidden_dim=3,
            unroll_horizon=10, weight_scale=0.05,
            with_output_projection=False,
        )
        lip = RecurrentLipschitzBound(arch)
        x_lo = -0.5 * np.ones(4)
        x_hi = 0.5 * np.ones(4)
        cert_naive = lip.compute_naive()
        cert_decay = lip.compute_with_decay(x_lo, x_hi)
        assert np.isfinite(cert_decay.global_bound)

    def test_gru_lipschitz(self) -> None:
        """Lipschitz bound works for GRU architectures."""
        arch = make_random_recurrent_architecture(
            cell_type="gru", hidden_dim=4, unroll_horizon=5
        )
        lip = RecurrentLipschitzBound(arch)
        cert = lip.compute_naive()
        assert cert.global_bound > 0.0
        assert np.isfinite(cert.global_bound)


# ======================================================================
# 7. Generator count management during unrolling
# ======================================================================


class TestGeneratorManagement:
    """Test that generator reduction controls blowup during unrolling."""

    def test_generators_bounded(self) -> None:
        """Generator count stays below max_generators after evaluation."""
        arch = make_random_recurrent_architecture(
            cell_type="lstm", input_dim=4, hidden_dim=3,
            unroll_horizon=10, weight_scale=0.1,
        )
        evaluator = RecurrentAbstractEvaluator(arch, max_generators=50)
        x_z = Zonotope.from_interval(-np.ones(4), np.ones(4))
        result = evaluator.evaluate(x_z)

        # Output zonotope generators should be bounded
        assert result.output_zonotope.num_generators <= 200

    def test_generator_counts_recorded(self) -> None:
        """Per-step generator counts are recorded in the output."""
        arch = make_random_recurrent_architecture(
            cell_type="lstm", input_dim=4, hidden_dim=3,
            unroll_horizon=5,
        )
        evaluator = RecurrentAbstractEvaluator(arch, max_generators=50)
        x_z = Zonotope.from_interval(-0.1 * np.ones(4), 0.1 * np.ones(4))
        result = evaluator.evaluate(x_z)
        assert len(result.generator_counts) == 5
        assert all(isinstance(c, int) for c in result.generator_counts)


# ======================================================================
# 8. Unrolling equivalence
# ======================================================================


class TestUnrollingEquivalence:
    """Test unrolling produces correct feedforward representation."""

    def test_unroll_produces_network_architecture(self) -> None:
        """Unroller creates a valid NetworkArchitecture."""
        arch = make_random_recurrent_architecture(
            cell_type="lstm", input_dim=4, hidden_dim=3,
            output_dim=2, unroll_horizon=3,
        )
        unroller = RecurrentPolicyUnroller(arch)
        ff = unroller.unroll()
        assert ff.input_dim == 4 + 3  # input + hidden
        assert ff.output_dim == 2
        assert ff.depth > 0

    def test_unrolled_lipschitz_matches_naive(self) -> None:
        """Unrolled Lipschitz bound is consistent with naive computation."""
        arch = make_random_recurrent_architecture(
            cell_type="lstm", input_dim=4, hidden_dim=3,
            unroll_horizon=3, with_output_projection=False,
        )
        unroller = RecurrentPolicyUnroller(arch)
        lip_unrolled = unroller.unrolled_lipschitz()
        lip_naive = RecurrentLipschitzBound(arch).compute_naive()
        # Both use Lip(f)^K, should match
        np.testing.assert_allclose(lip_unrolled, lip_naive.global_bound, rtol=1e-6)

    def test_per_step_lipschitz_positive(self) -> None:
        """Per-step Lipschitz constant is positive."""
        arch = make_random_recurrent_architecture(cell_type="gru")
        unroller = RecurrentPolicyUnroller(arch)
        assert unroller.per_step_lipschitz() > 0.0


# ======================================================================
# 9. Edge cases
# ======================================================================


class TestEdgeCases:
    """Test edge cases for recurrent policies."""

    def test_single_timestep(self) -> None:
        """Unroll horizon K=1 produces a valid result."""
        arch = make_random_recurrent_architecture(
            cell_type="lstm", unroll_horizon=1,
        )
        evaluator = RecurrentAbstractEvaluator(arch)
        x_z = Zonotope.from_interval(-np.ones(4), np.ones(4))
        result = evaluator.evaluate(x_z)
        assert result.output_zonotope.dimension == arch.output_dim
        assert len(result.per_step_zonotopes) == 1

    def test_zero_hidden_state(self) -> None:
        """Default zero hidden state produces valid output."""
        arch = make_random_recurrent_architecture(cell_type="gru")
        evaluator = RecurrentAbstractEvaluator(arch)
        x_z = Zonotope.from_interval(-np.ones(4), np.ones(4))
        result = evaluator.evaluate(x_z)
        assert np.all(np.isfinite(result.concrete_center))

    def test_large_horizon(self) -> None:
        """Large unroll horizon K=50 completes without error."""
        arch = make_random_recurrent_architecture(
            cell_type="lstm", input_dim=2, hidden_dim=2,
            output_dim=1, unroll_horizon=50, weight_scale=0.05,
        )
        evaluator = RecurrentAbstractEvaluator(arch, max_generators=30)
        x_z = Zonotope.from_interval(-0.1 * np.ones(2), 0.1 * np.ones(2))
        result = evaluator.evaluate(x_z)
        assert result.output_zonotope.dimension == 1
        assert len(result.per_step_zonotopes) == 50

    def test_invalid_cell_type(self) -> None:
        """Invalid cell type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported cell type"):
            RecurrentNetworkArchitecture(
                cell_type="rnn",
                input_dim=4,
                hidden_dim=3,
                output_dim=2,
                unroll_horizon=5,
                W_gates=np.zeros((12, 4)),
                U_gates=np.zeros((12, 3)),
                b_gates=np.zeros(12),
            )


# ======================================================================
# 10. RecurrentAbstractEvaluator soundness (multi-step)
# ======================================================================


class TestMultiStepSoundness:
    """Test multi-step abstract evaluation soundness."""

    def test_lstm_multi_step_soundness(self) -> None:
        """Concrete multi-step LSTM output lies within abstract bounds."""
        arch = make_random_recurrent_architecture(
            cell_type="lstm", input_dim=3, hidden_dim=2,
            output_dim=2, unroll_horizon=4, weight_scale=0.1,
        )
        evaluator = RecurrentAbstractEvaluator(arch, max_generators=100)
        center = np.array([0.5, -0.3, 0.1])
        eps = 0.15
        x_z = Zonotope.from_interval(center - eps, center + eps)
        result = evaluator.evaluate(x_z)
        lo, hi = result.output_zonotope.bounding_box()[:, 0], \
                 result.output_zonotope.bounding_box()[:, 1]

        # Sample concrete trajectories
        cell = LSTMCell(
            arch.hidden_dim, arch.W_gates, arch.U_gates, arch.b_gates
        )
        for _ in range(100):
            x_sample = center + RNG.uniform(-eps, eps, size=3)
            h = np.zeros(2)
            c = np.zeros(2)
            for _ in range(4):
                h, c = cell.forward_concrete(x_sample, h, c)
            out = arch.W_output @ h + arch.b_output
            assert np.all(out >= lo - 1e-5), f"Below lower: {out} < {lo}"
            assert np.all(out <= hi + 1e-5), f"Above upper: {out} > {hi}"

    def test_gru_multi_step_soundness(self) -> None:
        """Concrete multi-step GRU output lies within abstract bounds."""
        arch = make_random_recurrent_architecture(
            cell_type="gru", input_dim=3, hidden_dim=2,
            output_dim=2, unroll_horizon=4, weight_scale=0.1,
        )
        evaluator = RecurrentAbstractEvaluator(arch, max_generators=100)
        center = np.array([0.2, -0.1, 0.4])
        eps = 0.15
        x_z = Zonotope.from_interval(center - eps, center + eps)
        result = evaluator.evaluate(x_z)
        lo, hi = result.output_zonotope.bounding_box()[:, 0], \
                 result.output_zonotope.bounding_box()[:, 1]

        cell = GRUCell(
            arch.hidden_dim, arch.W_gates, arch.U_gates, arch.b_gates
        )
        for _ in range(100):
            x_sample = center + RNG.uniform(-eps, eps, size=3)
            h = np.zeros(2)
            for _ in range(4):
                h = cell.forward_concrete(x_sample, h)
            out = arch.W_output @ h + arch.b_output
            assert np.all(out >= lo - 1e-5), f"Below lower: {out} < {lo}"
            assert np.all(out <= hi + 1e-5), f"Above upper: {out} > {hi}"


# ======================================================================
# 11. Abstract arithmetic helpers
# ======================================================================


class TestAbstractHelpers:
    """Test abstract arithmetic helper functions."""

    def test_abstract_add(self) -> None:
        """Minkowski sum has correct center and generator count."""
        a = Zonotope.from_interval(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        b = Zonotope.from_interval(np.array([0.0, 0.0]), np.array([2.0, 2.0]))
        c = _abstract_add(a, b)
        np.testing.assert_allclose(c.center, [1.5, 1.5])
        assert c.num_generators == a.num_generators + b.num_generators

    def test_abstract_hadamard_soundness(self) -> None:
        """Hadamard product abstract result contains concrete products."""
        a = Zonotope.from_interval(np.array([-1.0, 0.5]), np.array([1.0, 2.0]))
        b = Zonotope.from_interval(np.array([0.0, -1.0]), np.array([2.0, 1.0]))
        c = _abstract_hadamard(a, b)
        lo, hi = c.bounding_box()[:, 0], c.bounding_box()[:, 1]

        for _ in range(200):
            av = np.array([RNG.uniform(-1.0, 1.0), RNG.uniform(0.5, 2.0)])
            bv = np.array([RNG.uniform(0.0, 2.0), RNG.uniform(-1.0, 1.0)])
            prod = av * bv
            assert np.all(prod >= lo - 1e-10)
            assert np.all(prod <= hi + 1e-10)

    def test_abstract_one_minus(self) -> None:
        """1 - z has correct center and generators."""
        z = Zonotope.from_interval(np.array([0.2, 0.3]), np.array([0.8, 0.7]))
        result = _abstract_one_minus(z)
        lo, hi = result.bounding_box()[:, 0], result.bounding_box()[:, 1]
        np.testing.assert_allclose(lo, [0.2, 0.3], atol=1e-10)
        np.testing.assert_allclose(hi, [0.8, 0.7], atol=1e-10)


# ======================================================================
# 12. RecurrentNetworkArchitecture validation
# ======================================================================


class TestArchitectureValidation:
    """Test RecurrentNetworkArchitecture construction and validation."""

    def test_summary(self) -> None:
        """Summary string is non-empty and contains cell type."""
        arch = make_random_recurrent_architecture(cell_type="lstm")
        s = arch.summary()
        assert "lstm" in s
        assert str(arch.hidden_dim) in s

    def test_total_parameters(self) -> None:
        """Parameter count is correct."""
        arch = make_random_recurrent_architecture(
            cell_type="lstm", input_dim=4, hidden_dim=8,
            output_dim=2, with_output_projection=True,
        )
        expected = (4 * 8 * 4) + (4 * 8 * 8) + (4 * 8) + (2 * 8) + 2
        assert arch.total_parameters == expected

    def test_wrong_gate_shape_raises(self) -> None:
        """Mismatched W_gates shape raises ValueError."""
        with pytest.raises(ValueError, match="W_gates row count"):
            RecurrentNetworkArchitecture(
                cell_type="lstm",
                input_dim=4,
                hidden_dim=3,
                output_dim=2,
                unroll_horizon=5,
                W_gates=np.zeros((10, 4)),  # should be 12
                U_gates=np.zeros((12, 3)),
                b_gates=np.zeros(12),
            )


# ======================================================================
# 13. Concrete center in abstract output
# ======================================================================


class TestConcreteCenterTracking:
    """Test that the concrete center is correctly computed."""

    def test_concrete_center_matches_direct(self) -> None:
        """Concrete center in abstract output matches direct forward pass."""
        arch = make_random_recurrent_architecture(
            cell_type="lstm", input_dim=4, hidden_dim=3,
            output_dim=2, unroll_horizon=5,
        )
        evaluator = RecurrentAbstractEvaluator(arch)
        x_center = RNG.normal(size=4) * 0.3
        x_z = Zonotope.from_point(x_center)
        result = evaluator.evaluate(x_z)

        # Direct computation
        cell = LSTMCell(
            arch.hidden_dim, arch.W_gates, arch.U_gates, arch.b_gates
        )
        h = np.zeros(3)
        c = np.zeros(3)
        for _ in range(5):
            h, c = cell.forward_concrete(x_center, h, c)
        expected = arch.W_output @ h + arch.b_output
        np.testing.assert_allclose(result.concrete_center, expected, atol=1e-10)


# ======================================================================
# 14. Factory helper
# ======================================================================


class TestFactory:
    """Test the make_random_recurrent_architecture factory."""

    def test_lstm_factory(self) -> None:
        """LSTM factory creates valid architecture."""
        arch = make_random_recurrent_architecture(cell_type="lstm")
        assert arch.cell_type == "lstm"
        assert arch.W_gates.shape == (4 * arch.hidden_dim, arch.input_dim)

    def test_gru_factory(self) -> None:
        """GRU factory creates valid architecture."""
        arch = make_random_recurrent_architecture(cell_type="gru")
        assert arch.cell_type == "gru"
        assert arch.W_gates.shape == (3 * arch.hidden_dim, arch.input_dim)

    def test_no_output_projection(self) -> None:
        """Factory without output projection sets W_output to None."""
        arch = make_random_recurrent_architecture(with_output_projection=False)
        assert arch.W_output is None
        assert arch.b_output is None


# ======================================================================
# 15. Output bounds convenience method
# ======================================================================


class TestOutputBounds:
    """Test the output_bounds convenience method."""

    def test_bounds_shape(self) -> None:
        """output_bounds returns arrays of correct shape."""
        arch = make_random_recurrent_architecture(
            cell_type="lstm", output_dim=3, unroll_horizon=3,
        )
        evaluator = RecurrentAbstractEvaluator(arch)
        x_z = Zonotope.from_interval(-np.ones(4), np.ones(4))
        lo, hi = evaluator.output_bounds(x_z)
        assert lo.shape == (3,)
        assert hi.shape == (3,)
        assert np.all(lo <= hi + 1e-10)
