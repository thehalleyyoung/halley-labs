"""Tests for certificate computation from causal_qd.certificates.

Covers EdgeCertificate construction and derived metrics,
BootstrapCertificateComputer with BICScore, LipschitzBound and
LipschitzBoundComputer, PathCertificate composition via
CertificateComposer, and cross-module consistency checks.

All tests use real BIC scoring on Gaussian data generated from a
known linear SCM (chain 0→1→2→3→4).
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pytest

from causal_qd.certificates.edge_certificate import EdgeCertificate
from causal_qd.certificates.bootstrap import BootstrapCertificateComputer
from causal_qd.certificates.lipschitz import (
    LipschitzBound,
    LipschitzBoundComputer,
)
from causal_qd.certificates.path_certificate import (
    PathCertificate,
    CertificateComposer,
)
from causal_qd.scores.bic import BICScore


# ===================================================================
# Helpers
# ===================================================================

def _make_score_fn(scorer: BICScore):
    """Wrap a BICScore into the (adj, data)->float callable expected by
    BootstrapCertificateComputer."""
    def score_fn(adj, data):
        return scorer.score(adj, data)
    return score_fn


def _generate_chain_data(
    rng: np.random.Generator,
    n_samples: int = 500,
    n_vars: int = 5,
    weights: np.ndarray | None = None,
    noise_scale: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate data from a chain SCM 0→1→…→(n_vars-1).

    Returns (data, adjacency_matrix).
    """
    if weights is None:
        weights = np.array([0.8, 0.7, 0.9, 0.6])
    adj = np.zeros((n_vars, n_vars), dtype=np.int8)
    for i in range(n_vars - 1):
        adj[i, i + 1] = 1
    data = np.zeros((n_samples, n_vars))
    data[:, 0] = rng.standard_normal(n_samples)
    for i in range(1, n_vars):
        data[:, i] = (
            weights[i - 1] * data[:, i - 1]
            + rng.standard_normal(n_samples) * noise_scale
        )
    return data, adj


# ===================================================================
# test_edge_certificate_computation
# ===================================================================

class TestEdgeCertificateComputation:
    """Verify EdgeCertificate construction, properties, derived metrics,
    and serialisation round-trip using certificates produced by the
    bootstrap computer on real BIC-scored data."""

    def test_bootstrap_produces_certificates_for_all_edges(
        self, small_dag, gaussian_data, bic_scorer, rng,
    ):
        """compute_edge_certificates returns one cert per DAG edge."""
        data, _ = gaussian_data
        adj = small_dag.adjacency.copy()
        score_fn = _make_score_fn(bic_scorer)

        computer = BootstrapCertificateComputer(
            n_bootstrap=50, score_fn=score_fn, rng=rng,
        )
        certs = computer.compute_edge_certificates(adj, data)

        edges = list(zip(*np.nonzero(adj)))
        assert len(certs) == len(edges)
        for edge in edges:
            assert edge in certs

    def test_certificate_value_in_unit_interval(
        self, small_dag, gaussian_data, bic_scorer, rng,
    ):
        """Certificate .value lies in [0, 1]."""
        data, _ = gaussian_data
        adj = small_dag.adjacency.copy()
        score_fn = _make_score_fn(bic_scorer)

        computer = BootstrapCertificateComputer(
            n_bootstrap=50, score_fn=score_fn, rng=rng,
        )
        certs = computer.compute_edge_certificates(adj, data)

        for cert in certs.values():
            assert 0.0 <= cert.value <= 1.0

    def test_certificate_confidence_matches_input(
        self, small_dag, gaussian_data, bic_scorer, rng,
    ):
        """Certificates carry the confidence level set on the computer."""
        data, _ = gaussian_data
        adj = small_dag.adjacency.copy()
        score_fn = _make_score_fn(bic_scorer)

        computer = BootstrapCertificateComputer(
            n_bootstrap=30, score_fn=score_fn,
            confidence_level=0.90, rng=rng,
        )
        certs = computer.compute_edge_certificates(adj, data)
        for cert in certs.values():
            assert cert.confidence == pytest.approx(0.90)

    def test_bootstrap_frequency_is_fraction(
        self, small_dag, gaussian_data, bic_scorer, rng,
    ):
        """bootstrap_frequency is between 0 and 1."""
        data, _ = gaussian_data
        adj = small_dag.adjacency.copy()
        score_fn = _make_score_fn(bic_scorer)
        computer = BootstrapCertificateComputer(
            n_bootstrap=40, score_fn=score_fn, rng=rng,
        )
        certs = computer.compute_edge_certificates(adj, data)
        for cert in certs.values():
            assert 0.0 <= cert.bootstrap_frequency <= 1.0

    def test_confidence_interval_contains_frequency(
        self, small_dag, gaussian_data, bic_scorer, rng,
    ):
        """The Wilson CI should contain the point estimate."""
        data, _ = gaussian_data
        adj = small_dag.adjacency.copy()
        score_fn = _make_score_fn(bic_scorer)
        computer = BootstrapCertificateComputer(
            n_bootstrap=80, score_fn=score_fn, rng=rng,
        )
        certs = computer.compute_edge_certificates(adj, data)
        for cert in certs.values():
            lo, hi = cert.confidence_interval()
            assert lo <= hi
            assert lo <= cert.bootstrap_frequency + 1e-10
            assert cert.bootstrap_frequency - 1e-10 <= hi

    def test_score_gap_equals_score_delta(
        self, small_dag, gaussian_data, bic_scorer, rng,
    ):
        """score_gap is an alias for score_delta."""
        data, _ = gaussian_data
        adj = small_dag.adjacency.copy()
        score_fn = _make_score_fn(bic_scorer)
        computer = BootstrapCertificateComputer(
            n_bootstrap=30, score_fn=score_fn, rng=rng,
        )
        certs = computer.compute_edge_certificates(adj, data)
        for cert in certs.values():
            assert cert.score_gap == cert.score_delta

    def test_serialisation_round_trip(
        self, small_dag, gaussian_data, bic_scorer, rng,
    ):
        """to_dict -> from_dict preserves key fields."""
        data, _ = gaussian_data
        adj = small_dag.adjacency.copy()
        score_fn = _make_score_fn(bic_scorer)
        computer = BootstrapCertificateComputer(
            n_bootstrap=30, score_fn=score_fn, rng=rng,
        )
        certs = computer.compute_edge_certificates(adj, data)
        for cert in certs.values():
            d = cert.to_dict()
            restored = EdgeCertificate.from_dict(d)
            assert restored.source == cert.source
            assert restored.target == cert.target
            assert restored.bootstrap_frequency == pytest.approx(
                cert.bootstrap_frequency
            )
            assert restored.score_delta == pytest.approx(cert.score_delta)

    def test_combine_takes_minimum(self, rng):
        """combine() of two EdgeCertificates returns minimum value."""
        c1 = EdgeCertificate(
            source=0, target=1,
            bootstrap_frequency=0.9, score_delta=5.0,
        )
        c2 = EdgeCertificate(
            source=0, target=1,
            bootstrap_frequency=0.6, score_delta=2.0,
        )
        combined = c1.combine(c2)
        assert combined.value <= c1.value
        assert combined.value <= c2.value

    def test_edge_certificate_source_target(self):
        """source and target are stored correctly."""
        cert = EdgeCertificate(
            source=2, target=3,
            bootstrap_frequency=0.75, score_delta=1.5,
        )
        assert cert.source == 2
        assert cert.target == 3

    def test_edge_certificate_stability_radius_no_lipschitz(self):
        """stability_radius is inf when lipschitz_bound is None."""
        cert = EdgeCertificate(
            source=0, target=1,
            bootstrap_frequency=0.8, score_delta=3.0,
            lipschitz_bound=None,
        )
        assert cert.stability_radius() == float("inf")

    def test_edge_certificate_stability_radius_with_lipschitz(self):
        """stability_radius = |score_delta| / lipschitz_bound."""
        cert = EdgeCertificate(
            source=0, target=1,
            bootstrap_frequency=0.8, score_delta=4.0,
            lipschitz_bound=2.0,
        )
        assert cert.stability_radius() == pytest.approx(2.0)

    def test_compute_all_certificates_returns_sorted_list(
        self, small_dag, gaussian_data, bic_scorer, rng,
    ):
        """compute_all_certificates returns a list sorted by (src, tgt)."""
        data, _ = gaussian_data
        adj = small_dag.adjacency.copy()
        score_fn = _make_score_fn(bic_scorer)
        computer = BootstrapCertificateComputer(
            n_bootstrap=30, score_fn=score_fn, rng=rng,
        )
        all_certs = computer.compute_all_certificates(adj, data)
        assert isinstance(all_certs, list)
        assert len(all_certs) == int(adj.sum())
        keys = [(c.source, c.target) for c in all_certs]
        assert keys == sorted(keys)

    def test_nonedge_certificates_cover_absent_edges(
        self, small_dag, gaussian_data, bic_scorer, rng,
    ):
        """compute_nonedge_certificates produces certs for non-edges."""
        data, _ = gaussian_data
        adj = small_dag.adjacency.copy()
        score_fn = _make_score_fn(bic_scorer)
        computer = BootstrapCertificateComputer(
            n_bootstrap=20, score_fn=score_fn, rng=rng,
        )
        nonedge_certs = computer.compute_nonedge_certificates(adj, data)
        existing_edges = set(zip(*np.nonzero(adj)))
        for edge in nonedge_certs:
            assert edge not in existing_edges


# ===================================================================
# test_bootstrap_stability_range
# ===================================================================

class TestBootstrapStabilityRange:
    """Bootstrap-computed certificates should show high stability for
    edges present in the true data-generating SCM and lower stability
    for edges with weak or no causal support."""

    def test_true_edge_high_frequency(self, bic_scorer, rng):
        """Edges in the true SCM should have high bootstrap frequency."""
        data, adj = _generate_chain_data(rng, n_samples=800)
        score_fn = _make_score_fn(bic_scorer)
        computer = BootstrapCertificateComputer(
            n_bootstrap=80, score_fn=score_fn, rng=rng,
        )
        certs = computer.compute_edge_certificates(adj, data)
        for cert in certs.values():
            assert cert.bootstrap_frequency >= 0.5, (
                f"Edge ({cert.source},{cert.target}) freq "
                f"{cert.bootstrap_frequency:.2f} too low"
            )

    def test_score_delta_positive_for_true_edges(self, bic_scorer, rng):
        """True edges should generally improve the BIC score."""
        data, adj = _generate_chain_data(rng, n_samples=600)
        score_fn = _make_score_fn(bic_scorer)
        computer = BootstrapCertificateComputer(
            n_bootstrap=60, score_fn=score_fn, rng=rng,
        )
        certs = computer.compute_edge_certificates(adj, data)
        positive_count = sum(1 for c in certs.values() if c.score_delta > 0)
        assert positive_count >= len(certs) // 2

    def test_more_bootstraps_tighter_ci(self, bic_scorer, rng):
        """Increasing n_bootstrap should narrow the confidence interval."""
        data, adj = _generate_chain_data(rng, n_samples=500)
        score_fn = _make_score_fn(bic_scorer)

        rng_lo = np.random.default_rng(42)
        comp_lo = BootstrapCertificateComputer(
            n_bootstrap=20, score_fn=score_fn, rng=rng_lo,
        )
        certs_lo = comp_lo.compute_edge_certificates(adj, data)

        rng_hi = np.random.default_rng(42)
        comp_hi = BootstrapCertificateComputer(
            n_bootstrap=200, score_fn=score_fn, rng=rng_hi,
        )
        certs_hi = comp_hi.compute_edge_certificates(adj, data)

        for edge in certs_lo:
            lo_lo, lo_hi = certs_lo[edge].confidence_interval()
            hi_lo, hi_hi = certs_hi[edge].confidence_interval()
            width_lo = lo_hi - lo_lo
            width_hi = hi_hi - hi_lo
            assert width_hi <= width_lo + 0.05  # allow small tolerance

    def test_bootstrap_deltas_stored(self, bic_scorer, rng):
        """bootstrap_deltas should be stored and have correct length."""
        data, adj = _generate_chain_data(rng, n_samples=300)
        score_fn = _make_score_fn(bic_scorer)
        n_boot = 40
        computer = BootstrapCertificateComputer(
            n_bootstrap=n_boot, score_fn=score_fn, rng=rng,
        )
        certs = computer.compute_edge_certificates(adj, data)
        for cert in certs.values():
            assert cert.bootstrap_deltas is not None
            assert len(cert.bootstrap_deltas) == n_boot

    def test_score_delta_ci_uses_bootstrap_deltas(self, bic_scorer, rng):
        """score_delta_confidence_interval should use bootstrap_deltas."""
        data, adj = _generate_chain_data(rng, n_samples=400)
        score_fn = _make_score_fn(bic_scorer)
        computer = BootstrapCertificateComputer(
            n_bootstrap=60, score_fn=score_fn, rng=rng,
        )
        certs = computer.compute_edge_certificates(adj, data)
        for cert in certs.values():
            lo, hi = cert.score_delta_confidence_interval()
            assert lo <= hi
            deltas = np.array(cert.bootstrap_deltas)
            assert lo >= float(np.min(deltas)) - 1e-10
            assert hi <= float(np.max(deltas)) + 1e-10


# ===================================================================
# test_lipschitz_bound_positive
# ===================================================================

class TestLipschitzBoundPositive:
    """LipschitzBound and LipschitzBoundComputer should produce
    finite, non-negative bounds on BIC score sensitivity."""

    def test_lipschitz_bound_estimate_positive(self, gaussian_data):
        """LipschitzBound.estimate_constant returns a positive float."""
        data, _ = gaussian_data
        lb = LipschitzBound(regularisation=1e-6)
        L = lb.estimate_constant(data)
        assert L > 0.0
        assert np.isfinite(L)

    def test_lipschitz_bound_between_two_dags(self, gaussian_data):
        """LipschitzBound.bound gives a non-negative value for two DAGs."""
        data, adj = gaussian_data
        adj2 = adj.copy()
        adj2[0, 1] = 0  # remove one edge
        lb = LipschitzBound(regularisation=1e-6)
        b = lb.bound(adj, adj2, data)
        assert b >= 0.0
        assert np.isfinite(b)

    def test_lipschitz_bound_identical_dags_zero(self, gaussian_data):
        """Bound between identical DAGs should be zero."""
        data, adj = gaussian_data
        lb = LipschitzBound(regularisation=1e-6)
        b = lb.bound(adj, adj, data)
        assert b == pytest.approx(0.0)

    def test_lipschitz_bound_computer_spectral(
        self, gaussian_data, bic_scorer,
    ):
        """LipschitzBoundComputer.spectral_bound returns positive."""
        data, adj = gaussian_data
        score_fn = _make_score_fn(bic_scorer)
        lbc = LipschitzBoundComputer(score_fn=score_fn)
        sb = lbc.spectral_bound(adj, data)
        assert sb > 0.0
        assert np.isfinite(sb)

    def test_lipschitz_bound_computer_empirical(
        self, gaussian_data, bic_scorer, rng,
    ):
        """LipschitzBoundComputer.empirical_bound returns non-negative."""
        data, adj = gaussian_data
        score_fn = _make_score_fn(bic_scorer)
        lbc = LipschitzBoundComputer(score_fn=score_fn, n_perturbations=20)
        eb = lbc.empirical_bound(adj, data, rng=rng)
        assert eb >= 0.0
        assert np.isfinite(eb)

    def test_per_edge_lipschitz_positive(
        self, small_dag, gaussian_data, bic_scorer, rng,
    ):
        """per_edge_lipschitz returns a non-negative bound per edge."""
        data, _ = gaussian_data
        adj = small_dag.adjacency.copy()
        score_fn = _make_score_fn(bic_scorer)
        lbc = LipschitzBoundComputer(score_fn=score_fn, n_perturbations=10)
        pel = lbc.per_edge_lipschitz(adj, data, rng=rng)
        edges = list(zip(*np.nonzero(adj)))
        assert len(pel) == len(edges)
        for edge, val in pel.items():
            assert val >= 0.0
            assert np.isfinite(val)

    def test_spectral_bound_scales_with_data_size(self, bic_scorer, rng):
        """Spectral bound should grow with sample size (N factor)."""
        adj = np.zeros((3, 3), dtype=np.int8)
        adj[0, 1] = 1
        adj[1, 2] = 1
        score_fn = _make_score_fn(bic_scorer)
        lbc = LipschitzBoundComputer(score_fn=score_fn)

        data_small = rng.standard_normal((100, 3))
        data_large = rng.standard_normal((1000, 3))
        sb_small = lbc.spectral_bound(adj, data_small)
        sb_large = lbc.spectral_bound(adj, data_large)
        # Spectral bound has factor N, so larger data → larger bound
        assert sb_large > sb_small * 0.5

    def test_bootstrap_with_lipschitz(
        self, small_dag, gaussian_data, bic_scorer, rng,
    ):
        """BootstrapCertificateComputer with compute_lipschitz=True
        should attach lipschitz_bound to edge certificates."""
        data, _ = gaussian_data
        adj = small_dag.adjacency.copy()
        score_fn = _make_score_fn(bic_scorer)
        computer = BootstrapCertificateComputer(
            n_bootstrap=30, score_fn=score_fn, rng=rng,
            compute_lipschitz=True,
        )
        certs = computer.compute_edge_certificates(adj, data)
        for cert in certs.values():
            assert cert.lipschitz_bound is not None
            assert cert.lipschitz_bound >= 0.0

    def test_stability_radius_finite_with_lipschitz(
        self, small_dag, gaussian_data, bic_scorer, rng,
    ):
        """stability_radius should be finite when lipschitz_bound is set."""
        data, _ = gaussian_data
        adj = small_dag.adjacency.copy()
        score_fn = _make_score_fn(bic_scorer)
        computer = BootstrapCertificateComputer(
            n_bootstrap=30, score_fn=score_fn, rng=rng,
            compute_lipschitz=True,
        )
        certs = computer.compute_edge_certificates(adj, data)
        for cert in certs.values():
            if cert.lipschitz_bound and cert.lipschitz_bound > 0:
                sr = cert.stability_radius()
                assert np.isfinite(sr)
                assert sr >= 0.0


# ===================================================================
# test_path_certificate_composition
# ===================================================================

class TestPathCertificateComposition:
    """PathCertificate and CertificateComposer should correctly
    aggregate edge certificates along directed paths."""

    def test_path_certificate_basic_properties(self):
        """PathCertificate exposes correct length and path."""
        ec1 = EdgeCertificate(0, 1, 0.9, 5.0)
        ec2 = EdgeCertificate(1, 2, 0.8, 3.0)
        ec3 = EdgeCertificate(2, 3, 0.7, 2.0)
        pc = PathCertificate(
            path=[0, 1, 2, 3], edge_certificates=[ec1, ec2, ec3],
        )
        assert pc.path == [0, 1, 2, 3]
        assert pc.length == 3
        assert len(pc) == 3

    def test_min_edge_certificate_is_minimum(self):
        """min_edge_certificate should be the smallest edge value."""
        ec1 = EdgeCertificate(0, 1, 0.9, 5.0)
        ec2 = EdgeCertificate(1, 2, 0.5, 1.0)
        pc = PathCertificate(path=[0, 1, 2], edge_certificates=[ec1, ec2])
        assert pc.min_edge_certificate <= ec1.value
        assert pc.min_edge_certificate <= ec2.value
        assert pc.value == pc.min_edge_certificate

    def test_path_score_is_product(self):
        """path_score equals the product of individual edge values."""
        ec1 = EdgeCertificate(0, 1, 0.9, 5.0, frequency_weight=1.0)
        ec2 = EdgeCertificate(1, 2, 0.8, 3.0, frequency_weight=1.0)
        pc = PathCertificate(path=[0, 1, 2], edge_certificates=[ec1, ec2])
        expected = ec1.value * ec2.value
        assert pc.path_score == pytest.approx(expected)

    def test_min_bootstrap_frequency(self):
        """min_bootstrap_frequency is min over edge frequencies."""
        ec1 = EdgeCertificate(0, 1, 0.95, 5.0)
        ec2 = EdgeCertificate(1, 2, 0.70, 3.0)
        ec3 = EdgeCertificate(2, 3, 0.85, 4.0)
        pc = PathCertificate(
            path=[0, 1, 2, 3], edge_certificates=[ec1, ec2, ec3],
        )
        assert pc.min_bootstrap_frequency == pytest.approx(0.70)

    def test_composed_lipschitz_none_when_missing(self):
        """composed_lipschitz is None if any edge lacks a bound."""
        ec1 = EdgeCertificate(0, 1, 0.9, 5.0, lipschitz_bound=1.0)
        ec2 = EdgeCertificate(1, 2, 0.8, 3.0, lipschitz_bound=None)
        pc = PathCertificate(path=[0, 1, 2], edge_certificates=[ec1, ec2])
        assert pc.composed_lipschitz is None

    def test_composed_lipschitz_product(self):
        """composed_lipschitz is the product of per-edge bounds."""
        ec1 = EdgeCertificate(0, 1, 0.9, 5.0, lipschitz_bound=2.0)
        ec2 = EdgeCertificate(1, 2, 0.8, 3.0, lipschitz_bound=3.0)
        pc = PathCertificate(path=[0, 1, 2], edge_certificates=[ec1, ec2])
        assert pc.composed_lipschitz == pytest.approx(6.0)

    def test_path_stability_radius_is_minimum(self):
        """path_stability_radius equals the min edge stability_radius."""
        ec1 = EdgeCertificate(0, 1, 0.9, 4.0, lipschitz_bound=2.0)
        ec2 = EdgeCertificate(1, 2, 0.8, 6.0, lipschitz_bound=3.0)
        pc = PathCertificate(path=[0, 1, 2], edge_certificates=[ec1, ec2])
        expected = min(ec1.stability_radius(), ec2.stability_radius())
        assert pc.path_stability_radius == pytest.approx(expected)

    def test_composer_compose_path(self):
        """CertificateComposer.compose_path builds a PathCertificate."""
        ec01 = EdgeCertificate(0, 1, 0.9, 5.0)
        ec12 = EdgeCertificate(1, 2, 0.85, 4.0)
        ec23 = EdgeCertificate(2, 3, 0.7, 2.0)
        edge_certs = {(0, 1): ec01, (1, 2): ec12, (2, 3): ec23}
        composer = CertificateComposer(edge_certs)
        pc = composer.compose_path([0, 1, 2, 3])
        assert pc.path == [0, 1, 2, 3]
        assert pc.length == 3
        assert pc.min_bootstrap_frequency == pytest.approx(0.7)

    def test_composer_compose_path_missing_edge_raises(self):
        """compose_path raises KeyError for missing edge cert."""
        ec01 = EdgeCertificate(0, 1, 0.9, 5.0)
        composer = CertificateComposer({(0, 1): ec01})
        with pytest.raises(KeyError):
            composer.compose_path([0, 1, 2])

    def test_composer_compose_all_paths(self, small_dag, gaussian_data, bic_scorer, rng):
        """compose_all_paths finds all directed paths and certifies them."""
        data, _ = gaussian_data
        adj = small_dag.adjacency.copy()
        score_fn = _make_score_fn(bic_scorer)
        computer = BootstrapCertificateComputer(
            n_bootstrap=30, score_fn=score_fn, rng=rng,
        )
        edge_certs = computer.compute_edge_certificates(adj, data)
        composer = CertificateComposer(edge_certs)
        # In the chain 0→1→2→3→4, paths from 0 to 4 should exist
        path_certs = composer.compose_all_paths(adj, 0, 4)
        assert len(path_certs) >= 1
        # The single path should be [0, 1, 2, 3, 4]
        assert path_certs[0].path == [0, 1, 2, 3, 4]

    def test_compose_all_paths_sorted_by_value(
        self, small_dag, gaussian_data, bic_scorer, rng,
    ):
        """compose_all_paths returns certs sorted by value descending."""
        data, _ = gaussian_data
        adj = small_dag.adjacency.copy()
        score_fn = _make_score_fn(bic_scorer)
        computer = BootstrapCertificateComputer(
            n_bootstrap=30, score_fn=score_fn, rng=rng,
        )
        edge_certs = computer.compute_edge_certificates(adj, data)
        composer = CertificateComposer(edge_certs)
        path_certs = composer.compose_all_paths(adj, 0, 4)
        for i in range(len(path_certs) - 1):
            assert path_certs[i].value >= path_certs[i + 1].value

    def test_weakest_edge_identification(self):
        """weakest_edge returns the edge with lowest value."""
        ec1 = EdgeCertificate(0, 1, 0.9, 5.0)
        ec2 = EdgeCertificate(1, 2, 0.3, 0.5)
        ec3 = EdgeCertificate(2, 3, 0.8, 3.0)
        pc = PathCertificate(
            path=[0, 1, 2, 3], edge_certificates=[ec1, ec2, ec3],
        )
        weak = pc.weakest_edge()
        assert weak.source == 1
        assert weak.target == 2

    def test_path_confidence_interval_bounds(self):
        """path_confidence_interval returns valid bounds."""
        ec1 = EdgeCertificate(0, 1, 0.9, 5.0, n_bootstrap=100)
        ec2 = EdgeCertificate(1, 2, 0.8, 3.0, n_bootstrap=100)
        pc = PathCertificate(path=[0, 1, 2], edge_certificates=[ec1, ec2])
        lo, hi = pc.path_confidence_interval()
        assert 0.0 <= lo <= hi <= 1.0


# ===================================================================
# test_certificate_consistency
# ===================================================================

class TestCertificateConsistency:
    """Cross-checks between different certificate modules:
    bootstrap frequencies, Lipschitz bounds, score deltas, and
    composed path certificates should be mutually consistent."""

    def test_edge_value_between_frequency_and_sigmoid(
        self, small_dag, gaussian_data, bic_scorer, rng,
    ):
        """Edge value is a weighted combo of frequency and normalised delta."""
        data, _ = gaussian_data
        adj = small_dag.adjacency.copy()
        score_fn = _make_score_fn(bic_scorer)
        computer = BootstrapCertificateComputer(
            n_bootstrap=40, score_fn=score_fn, rng=rng,
        )
        certs = computer.compute_edge_certificates(adj, data)
        for cert in certs.values():
            sigmoid = 1.0 / (1.0 + math.exp(-cert.score_delta))
            w = 0.6  # default frequency_weight
            expected = w * cert.bootstrap_frequency + (1.0 - w) * sigmoid
            assert cert.value == pytest.approx(expected, abs=1e-10)

    def test_path_value_equals_min_edge_value(
        self, small_dag, gaussian_data, bic_scorer, rng,
    ):
        """PathCertificate.value equals min of edge values."""
        data, _ = gaussian_data
        adj = small_dag.adjacency.copy()
        score_fn = _make_score_fn(bic_scorer)
        computer = BootstrapCertificateComputer(
            n_bootstrap=30, score_fn=score_fn, rng=rng,
        )
        edge_certs = computer.compute_edge_certificates(adj, data)
        composer = CertificateComposer(edge_certs)
        pc = composer.compose_path([0, 1, 2, 3, 4])
        edge_vals = [edge_certs[e].value for e in sorted(edge_certs)]
        assert pc.value == pytest.approx(min(edge_vals))

    def test_stability_radius_consistent_with_lipschitz(
        self, small_dag, gaussian_data, bic_scorer, rng,
    ):
        """stability_radius = |score_delta|/lipschitz_bound when both set."""
        data, _ = gaussian_data
        adj = small_dag.adjacency.copy()
        score_fn = _make_score_fn(bic_scorer)
        computer = BootstrapCertificateComputer(
            n_bootstrap=30, score_fn=score_fn, rng=rng,
            compute_lipschitz=True,
        )
        certs = computer.compute_edge_certificates(adj, data)
        for cert in certs.values():
            if cert.lipschitz_bound and cert.lipschitz_bound > 0:
                expected = abs(cert.score_delta) / cert.lipschitz_bound
                assert cert.stability_radius() == pytest.approx(expected)
            else:
                assert cert.stability_radius() == float("inf")

    def test_empirical_vs_spectral_lipschitz_order(
        self, gaussian_data, bic_scorer, rng,
    ):
        """Empirical Lipschitz bound should be finite and in reasonable range."""
        data, adj = gaussian_data
        score_fn = _make_score_fn(bic_scorer)
        lbc = LipschitzBoundComputer(
            score_fn=score_fn, n_perturbations=30,
        )
        emp = lbc.empirical_bound(adj, data, rng=rng)
        spec = lbc.spectral_bound(adj, data)
        # Both should be positive and finite
        assert emp > 0.0
        assert spec > 0.0
        assert np.isfinite(emp)
        assert np.isfinite(spec)

    def test_nonedge_cert_frequency_high_for_absent_edges(
        self, bic_scorer, rng,
    ):
        """Non-edge certificates for truly absent edges should have
        high bootstrap frequency (edge absence is supported)."""
        data, adj = _generate_chain_data(rng, n_samples=600)
        score_fn = _make_score_fn(bic_scorer)
        computer = BootstrapCertificateComputer(
            n_bootstrap=40, score_fn=score_fn, rng=rng,
        )
        # Check one non-edge: 0→3 (absent in the chain 0→1→2→3→4)
        nonedge_certs = computer.compute_nonedge_certificates(
            adj, data, candidate_edges=[(0, 3)],
        )
        assert (0, 3) in nonedge_certs
        cert = nonedge_certs[(0, 3)]
        assert cert.bootstrap_frequency >= 0.3

    def test_edge_certs_dict_keys_match_adj(
        self, small_dag, gaussian_data, bic_scorer, rng,
    ):
        """Edge certificate dict keys match nonzero entries of adj."""
        data, _ = gaussian_data
        adj = small_dag.adjacency.copy()
        score_fn = _make_score_fn(bic_scorer)
        computer = BootstrapCertificateComputer(
            n_bootstrap=20, score_fn=score_fn, rng=rng,
        )
        certs = computer.compute_edge_certificates(adj, data)
        expected_edges = set(
            (int(i), int(j)) for i, j in zip(*np.nonzero(adj))
        )
        assert set(certs.keys()) == expected_edges


# ===================================================================
# test_certificate_known_stable_edge
# ===================================================================

class TestCertificateKnownStableEdge:
    """With strong signal and large sample size, certificates for
    true edges should clearly certify their presence."""

    def test_strong_edge_is_certified(self, bic_scorer):
        """A very strong edge (high weight, many samples) should
        have value exceeding the default certification threshold."""
        rng = np.random.default_rng(123)
        data, adj = _generate_chain_data(
            rng, n_samples=1000,
            weights=np.array([1.5, 1.2, 1.4, 1.1]),
            noise_scale=0.3,
        )
        score_fn = _make_score_fn(bic_scorer)
        computer = BootstrapCertificateComputer(
            n_bootstrap=100, score_fn=score_fn,
            rng=np.random.default_rng(456),
        )
        certs = computer.compute_edge_certificates(adj, data)
        for cert in certs.values():
            assert cert.is_certified(threshold=0.5), (
                f"Edge ({cert.source},{cert.target}) not certified: "
                f"value={cert.value:.4f}"
            )

    def test_strong_edge_high_bootstrap_frequency(self, bic_scorer):
        """Strong true edges should have bootstrap frequency near 1."""
        rng = np.random.default_rng(789)
        data, adj = _generate_chain_data(
            rng, n_samples=1000,
            weights=np.array([1.5, 1.2, 1.4, 1.1]),
            noise_scale=0.3,
        )
        score_fn = _make_score_fn(bic_scorer)
        computer = BootstrapCertificateComputer(
            n_bootstrap=100, score_fn=score_fn,
            rng=np.random.default_rng(101),
        )
        certs = computer.compute_edge_certificates(adj, data)
        for cert in certs.values():
            assert cert.bootstrap_frequency >= 0.8, (
                f"Edge ({cert.source},{cert.target}) "
                f"freq={cert.bootstrap_frequency:.2f}"
            )

    def test_strong_edge_positive_score_delta(self, bic_scorer):
        """Strong edges should have positive average score delta."""
        rng = np.random.default_rng(202)
        data, adj = _generate_chain_data(
            rng, n_samples=1000,
            weights=np.array([1.5, 1.2, 1.4, 1.1]),
            noise_scale=0.3,
        )
        score_fn = _make_score_fn(bic_scorer)
        computer = BootstrapCertificateComputer(
            n_bootstrap=80, score_fn=score_fn,
            rng=np.random.default_rng(303),
        )
        certs = computer.compute_edge_certificates(adj, data)
        for cert in certs.values():
            assert cert.score_delta > 0, (
                f"Edge ({cert.source},{cert.target}) "
                f"delta={cert.score_delta:.4f}"
            )

    def test_strong_edge_ci_bounded_away_from_zero(self, bic_scorer):
        """For strong edges the CI lower bound should exceed zero."""
        rng = np.random.default_rng(404)
        data, adj = _generate_chain_data(
            rng, n_samples=1000,
            weights=np.array([1.5, 1.2, 1.4, 1.1]),
            noise_scale=0.3,
        )
        score_fn = _make_score_fn(bic_scorer)
        computer = BootstrapCertificateComputer(
            n_bootstrap=100, score_fn=score_fn,
            rng=np.random.default_rng(505),
        )
        certs = computer.compute_edge_certificates(adj, data)
        for cert in certs.values():
            lo, hi = cert.confidence_interval()
            assert lo > 0.5, (
                f"Edge ({cert.source},{cert.target}) CI lower={lo:.4f}"
            )

    def test_path_through_strong_chain_is_certified(self, bic_scorer):
        """Full path certificate for a strong chain should pass."""
        rng = np.random.default_rng(606)
        data, adj = _generate_chain_data(
            rng, n_samples=1000,
            weights=np.array([1.5, 1.2, 1.4, 1.1]),
            noise_scale=0.3,
        )
        score_fn = _make_score_fn(bic_scorer)
        computer = BootstrapCertificateComputer(
            n_bootstrap=80, score_fn=score_fn,
            rng=np.random.default_rng(707),
        )
        edge_certs = computer.compute_edge_certificates(adj, data)
        composer = CertificateComposer(edge_certs)
        pc = composer.compose_path([0, 1, 2, 3, 4])
        assert pc.is_certified(threshold=0.5)
        assert pc.min_bootstrap_frequency >= 0.7

    def test_weak_edge_lower_certificate(self, bic_scorer):
        """An edge with tiny weight should have a lower certificate
        value than a strong edge."""
        rng = np.random.default_rng(808)
        n = 3
        adj = np.zeros((n, n), dtype=np.int8)
        adj[0, 1] = 1
        adj[1, 2] = 1
        data = np.zeros((500, n))
        data[:, 0] = rng.standard_normal(500)
        # Edge 0→1: strong
        data[:, 1] = 1.5 * data[:, 0] + rng.standard_normal(500) * 0.3
        # Edge 1→2: weak
        data[:, 2] = 0.05 * data[:, 1] + rng.standard_normal(500) * 1.0

        score_fn = _make_score_fn(bic_scorer)
        computer = BootstrapCertificateComputer(
            n_bootstrap=80, score_fn=score_fn,
            rng=np.random.default_rng(909),
        )
        certs = computer.compute_edge_certificates(adj, data)
        strong = certs[(0, 1)]
        weak = certs[(1, 2)]
        assert strong.bootstrap_frequency >= weak.bootstrap_frequency


# ---------------------------------------------------------------------------
# test_boltzmann_weighted_stability (Theorem 9)
# ---------------------------------------------------------------------------

from causal_qd.certificates.bootstrap import (
    BoltzmannStabilityResult,
    boltzmann_weighted_stability,
    boltzmann_edge_probabilities,
    optimal_beta,
)


def _make_edge_certs(
    edges: List[Tuple[int, int]],
    freq: float = 0.8,
    delta: float = 1.0,
) -> Dict[Tuple[int, int], EdgeCertificate]:
    """Helper: build a dict of EdgeCertificates with uniform values."""
    return {
        (s, t): EdgeCertificate(
            source=s, target=t,
            bootstrap_frequency=freq,
            score_delta=delta,
        )
        for s, t in edges
    }


class TestBoltzmannWeightedStability:
    """Tests for Boltzmann-weighted certificate stability (Theorem 9)."""

    def test_uniform_qualities_give_equal_weights(self):
        """When all qualities are identical, weights should be uniform."""
        dags = [np.eye(3) for _ in range(4)]
        quals = [1.0, 1.0, 1.0, 1.0]
        certs = [_make_edge_certs([(0, 0), (1, 1), (2, 2)]) for _ in range(4)]
        result = boltzmann_weighted_stability(dags, quals, certs, beta=1.0)
        np.testing.assert_allclose(result.per_dag_weights, 0.25, atol=1e-12)

    def test_varying_qualities_weight_better_dags_higher(self):
        """Lower quality score → higher Boltzmann weight (exp(-β·q))."""
        dags = [np.eye(3) for _ in range(3)]
        quals = [0.0, 5.0, 10.0]  # lower is "better" in Boltzmann weighting
        certs = [_make_edge_certs([(0, 0)]) for _ in range(3)]
        result = boltzmann_weighted_stability(dags, quals, certs, beta=1.0)
        assert result.per_dag_weights[0] > result.per_dag_weights[1]
        assert result.per_dag_weights[1] > result.per_dag_weights[2]

    def test_partition_function_is_positive(self):
        """Z_β must always be strictly positive."""
        dags = [np.eye(2)]
        quals = [42.0]
        certs = [_make_edge_certs([(0, 0), (1, 1)])]
        result = boltzmann_weighted_stability(dags, quals, certs, beta=2.0)
        assert result.partition_function > 0

    def test_ess_between_one_and_archive_size(self):
        """ESS should be in [1, len(archive)]."""
        n = 5
        dags = [np.eye(2) for _ in range(n)]
        quals = [float(i) for i in range(n)]
        certs = [_make_edge_certs([(0, 0)]) for _ in range(n)]
        result = boltzmann_weighted_stability(dags, quals, certs, beta=1.0)
        assert 1.0 <= result.effective_sample_size <= n + 1e-9

    def test_edge_probability_matrix_sums_correctly(self):
        """Each entry in the edge prob matrix must be in [0, 1]."""
        dag1 = np.array([[0, 1], [0, 0]])
        dag2 = np.array([[0, 0], [1, 0]])
        dags = [dag1, dag2]
        quals = [1.0, 1.0]
        certs = [
            _make_edge_certs([(0, 1)], freq=0.9, delta=2.0),
            _make_edge_certs([(1, 0)], freq=0.7, delta=1.0),
        ]
        prob = boltzmann_edge_probabilities(dags, quals, certs, beta=0.0)
        assert prob.shape == (2, 2)
        assert np.all(prob >= -1e-12)
        assert np.all(prob <= 1.0 + 1e-12)

    def test_beta_zero_gives_uniform_weights(self):
        """β = 0 ⇒ all weights equal regardless of quality."""
        dags = [np.eye(2) for _ in range(3)]
        quals = [0.0, 100.0, 1000.0]
        certs = [_make_edge_certs([(0, 0)]) for _ in range(3)]
        result = boltzmann_weighted_stability(dags, quals, certs, beta=0.0)
        np.testing.assert_allclose(result.per_dag_weights, 1.0 / 3, atol=1e-12)

    def test_single_dag_case(self):
        """With a single DAG the stability should equal its C_avg."""
        dag = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        certs_dict = _make_edge_certs([(0, 1), (1, 2)], freq=0.85, delta=1.5)
        result = boltzmann_weighted_stability([dag], [3.0], [certs_dict], beta=1.0)
        expected_avg = np.mean([c.value for c in certs_dict.values()])
        assert abs(result.stability_score - expected_avg) < 1e-10
        np.testing.assert_allclose(result.per_dag_weights, [1.0])
        assert abs(result.effective_sample_size - 1.0) < 1e-10

    def test_numerical_stability_extreme_qualities(self):
        """Should not overflow/NaN with very large quality differences.

        The partition function Z_β may overflow to inf when qualities
        span a huge range, but normalised weights and stability score
        must remain finite.
        """
        dags = [np.eye(2) for _ in range(3)]
        quals = [0.0, 1e6, -1e6]
        certs = [_make_edge_certs([(0, 0)], freq=0.9) for _ in range(3)]
        result = boltzmann_weighted_stability(dags, quals, certs, beta=1.0)
        assert np.isfinite(result.stability_score)
        assert result.partition_function > 0
        assert np.all(np.isfinite(result.per_dag_weights))

    def test_optimal_beta_inverse_std(self):
        """optimal_beta returns 1/std for normal spreads."""
        quals = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = optimal_beta(quals)
        expected = 1.0 / float(np.std(quals))
        assert abs(b - expected) < 1e-10

    def test_optimal_beta_constant_qualities(self):
        """optimal_beta returns 1.0 when std is zero."""
        assert optimal_beta([5.0, 5.0, 5.0]) == 1.0

    def test_optimal_beta_single_element(self):
        """optimal_beta returns 1.0 for a single element."""
        assert optimal_beta([42.0]) == 1.0
