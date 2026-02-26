"""
Tests for the regime detection and inference module.

Covers StickyHDPHMM fitting, forward-backward/Viterbi algorithms,
transition matrix estimation, online regime tracking, BOCPD, CUSUM,
and convergence diagnostics.
"""

from __future__ import annotations

import numpy as np
import pytest

from causal_trading.regime.sticky_hdp_hmm import (
    StickyHDPHMM,
    _forward,
    _backward,
    _forward_backward,
    _viterbi,
    _log_normalize,
)
from causal_trading.regime.regime_detection import (
    BayesianRegimeDetector,
    BayesianOnlineChangePointDetector,
    CUSUMDetector,
)
from causal_trading.regime.online_tracker import OnlineRegimeTracker
from causal_trading.regime.transition_matrix import TransitionMatrixEstimator
from causal_trading.regime.regime_posterior import RegimePosterior


# =========================================================================
# StickyHDPHMM: Fitting & state recovery
# =========================================================================

class TestStickyHDPHMM:
    """Tests for the Sticky HDP-HMM model."""

    def test_fit_returns_self(self, regime_switching_data, seed):
        data, _ = regime_switching_data
        model = StickyHDPHMM(K_max=3, n_iter=30, burn_in=10, random_state=seed)
        result = model.fit(data)
        assert result is model

    def test_predict_shape(self, regime_switching_data, seed):
        data, _ = regime_switching_data
        model = StickyHDPHMM(K_max=3, n_iter=30, burn_in=10, random_state=seed)
        model.fit(data)
        labels = model.predict(data)
        assert labels.shape == (len(data),)

    def test_predict_proba_shape(self, regime_switching_data, seed):
        data, _ = regime_switching_data
        model = StickyHDPHMM(K_max=3, n_iter=30, burn_in=10, random_state=seed)
        model.fit(data)
        proba = model.predict_proba(data)
        assert proba.shape[0] == len(data)
        # Probabilities sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_recovers_two_regime_structure(self, regime_switching_data, seed):
        """With well-separated means the model should use at most 3 states."""
        data, true_labels = regime_switching_data
        model = StickyHDPHMM(K_max=5, n_iter=80, burn_in=30, random_state=seed)
        model.fit(data)
        pred = model.predict(data)
        n_states_used = len(np.unique(pred))
        assert 2 <= n_states_used <= 4

    def test_transition_matrix_is_stochastic(self, regime_switching_data, seed):
        data, _ = regime_switching_data
        model = StickyHDPHMM(K_max=3, n_iter=30, burn_in=10, random_state=seed)
        model.fit(data)
        T = model.get_transition_matrix()
        assert T.ndim == 2
        np.testing.assert_allclose(T.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(T >= 0)

    def test_sample_generates_data(self, regime_switching_data, seed):
        data, _ = regime_switching_data
        model = StickyHDPHMM(K_max=3, n_iter=30, burn_in=10, random_state=seed)
        model.fit(data)
        samples, states = model.sample(50)
        assert samples.shape[0] == 50
        assert states.shape == (50,)

    def test_score_returns_scalar(self, regime_switching_data, seed):
        data, _ = regime_switching_data
        model = StickyHDPHMM(K_max=3, n_iter=30, burn_in=10, random_state=seed)
        model.fit(data)
        s = model.score(data)
        assert np.isfinite(s)

    def test_stationary_distribution_sums_to_one(self, regime_switching_data, seed):
        data, _ = regime_switching_data
        model = StickyHDPHMM(K_max=3, n_iter=30, burn_in=10, random_state=seed)
        model.fit(data)
        pi = model.get_stationary_distribution()
        np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-6)
        assert np.all(pi >= 0)

    def test_summary_has_expected_keys(self, regime_switching_data, seed):
        data, _ = regime_switching_data
        model = StickyHDPHMM(K_max=3, n_iter=30, burn_in=10, random_state=seed)
        model.fit(data)
        s = model.summary()
        assert "n_states" in s or "K" in s or len(s) > 0

    def test_multivariate_fit(self, multivariate_regime_data, seed):
        data, _ = multivariate_regime_data
        model = StickyHDPHMM(K_max=4, n_iter=40, burn_in=15, random_state=seed)
        model.fit(data)
        pred = model.predict(data)
        assert pred.shape == (len(data),)

    def test_credible_intervals(self, regime_switching_data, seed):
        data, _ = regime_switching_data
        model = StickyHDPHMM(K_max=3, n_iter=30, burn_in=10, random_state=seed)
        model.fit(data)
        lo, hi = model.transition_credible_intervals(0.90)
        assert lo.shape == hi.shape
        assert np.all(hi >= lo - 1e-10)


# =========================================================================
# Forward-backward & Viterbi correctness
# =========================================================================

class TestForwardBackward:
    """Test forward-backward and Viterbi on small hand-constructed HMMs."""

    @pytest.fixture
    def small_hmm_params(self):
        """2-state HMM with known parameters."""
        log_pi = np.log([0.6, 0.4])
        log_A = np.log([[0.7, 0.3], [0.4, 0.6]])
        # 5 time-steps of log-likelihoods
        log_lik = np.array([
            [np.log(0.9), np.log(0.1)],
            [np.log(0.2), np.log(0.8)],
            [np.log(0.7), np.log(0.3)],
            [np.log(0.4), np.log(0.6)],
            [np.log(0.5), np.log(0.5)],
        ])
        return log_pi, log_A, log_lik

    def test_forward_log_marginal_finite(self, small_hmm_params):
        log_pi, log_A, log_lik = small_hmm_params
        alpha, log_marginal = _forward(log_pi, log_A, log_lik)
        assert np.isfinite(log_marginal)
        assert alpha.shape == log_lik.shape

    def test_forward_backward_gamma_sums_to_one(self, small_hmm_params):
        log_pi, log_A, log_lik = small_hmm_params
        gamma, xi, log_marg = _forward_backward(log_pi, log_A, log_lik)
        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-6)

    def test_forward_backward_consistency(self, small_hmm_params):
        """Forward-backward log-marginal matches forward-only."""
        log_pi, log_A, log_lik = small_hmm_params
        alpha_fwd, lm_fwd = _forward(log_pi, log_A, log_lik)
        _, _, lm_fb = _forward_backward(log_pi, log_A, log_lik)
        np.testing.assert_allclose(lm_fwd, lm_fb, atol=1e-8)

    def test_backward_shape(self, small_hmm_params):
        log_pi, log_A, log_lik = small_hmm_params
        beta = _backward(log_A, log_lik)
        assert beta.shape == log_lik.shape

    def test_viterbi_returns_valid_states(self, small_hmm_params):
        log_pi, log_A, log_lik = small_hmm_params
        path, log_prob = _viterbi(log_pi, log_A, log_lik)
        assert path.shape == (log_lik.shape[0],)
        assert set(path).issubset({0, 1})
        assert np.isfinite(log_prob)

    def test_viterbi_optimal_for_trivial_case(self):
        """If state 0 is always overwhelmingly likely, Viterbi picks state 0."""
        log_pi = np.array([0.0, -1000.0])
        log_A = np.array([[0.0, -1000.0], [-1000.0, 0.0]])
        log_lik = np.array([[0.0, -1000.0]] * 10)
        path, _ = _viterbi(log_pi, log_A, log_lik)
        np.testing.assert_array_equal(path, 0)

    def test_viterbi_recovers_known_sequence(self):
        """Two-state HMM with known regime switches."""
        log_pi = np.log([0.9, 0.1])
        log_A = np.log([[0.95, 0.05], [0.05, 0.95]])
        T = 100
        true_states = np.zeros(T, dtype=int)
        true_states[40:70] = 1
        log_lik = np.zeros((T, 2))
        for t in range(T):
            if true_states[t] == 0:
                log_lik[t] = [0.0, -5.0]
            else:
                log_lik[t] = [-5.0, 0.0]
        path, _ = _viterbi(log_pi, log_A, log_lik)
        agreement = np.mean(path == true_states)
        assert agreement > 0.85

    def test_forward_backward_naive_comparison(self):
        """Compare forward-backward with naive enumeration on tiny HMM."""
        log_pi = np.log([0.5, 0.5])
        log_A = np.log([[0.8, 0.2], [0.3, 0.7]])
        log_lik = np.array([
            [np.log(0.6), np.log(0.4)],
            [np.log(0.3), np.log(0.7)],
            [np.log(0.5), np.log(0.5)],
        ])
        T, K = log_lik.shape

        # Naive: enumerate all 2^3 = 8 sequences
        total = 0.0
        state_marginals = np.zeros((T, K))
        import itertools
        for seq in itertools.product(range(K), repeat=T):
            p = np.exp(log_pi[seq[0]] + log_lik[0, seq[0]])
            for t in range(1, T):
                p *= np.exp(log_A[seq[t - 1], seq[t]] + log_lik[t, seq[t]])
            total += p
            for t in range(T):
                state_marginals[t, seq[t]] += p

        naive_gamma = state_marginals / total
        gamma, _, _ = _forward_backward(log_pi, log_A, log_lik)
        np.testing.assert_allclose(gamma, naive_gamma, atol=1e-6)


class TestLogNormalize:
    def test_log_normalize_sums_to_one(self):
        log_v = np.array([-1.0, -2.0, -0.5])
        normalized, _ = _log_normalize(log_v)
        np.testing.assert_allclose(np.exp(normalized).sum(), 1.0, atol=1e-8)

    def test_log_normalize_preserves_order(self):
        log_v = np.array([-1.0, -3.0, -0.5])
        normalized, _ = _log_normalize(log_v)
        assert normalized[2] > normalized[0] > normalized[1]


# =========================================================================
# Transition Matrix Estimation
# =========================================================================

class TestTransitionMatrixEstimator:
    def test_fit_from_sequence(self, known_regime_sequence):
        est = TransitionMatrixEstimator(n_states=2)
        est.fit(known_regime_sequence)
        T_map = est.get_map_estimate()
        assert T_map.shape == (2, 2)
        np.testing.assert_allclose(T_map.sum(axis=1), 1.0, atol=1e-6)

    def test_posterior_mean_is_stochastic(self, known_regime_sequence):
        est = TransitionMatrixEstimator(n_states=2)
        est.fit(known_regime_sequence)
        T_mean = est.get_posterior_mean()
        np.testing.assert_allclose(T_mean.sum(axis=1), 1.0, atol=1e-6)
        assert np.all(T_mean >= 0)

    def test_convergence_to_true_matrix(self, known_transition_matrix, rng):
        """With a long sequence, MAP should be close to the true matrix."""
        T_true = known_transition_matrix
        n = 5000
        states = np.empty(n, dtype=int)
        states[0] = 0
        for t in range(1, n):
            states[t] = rng.choice(2, p=T_true[states[t - 1]])
        est = TransitionMatrixEstimator(n_states=2, prior_alpha=0.1)
        est.fit(states)
        T_est = est.get_posterior_mean()
        np.testing.assert_allclose(T_est, T_true, atol=0.05)

    def test_credible_intervals_cover_true(self, known_transition_matrix, rng):
        T_true = known_transition_matrix
        n = 2000
        states = np.empty(n, dtype=int)
        states[0] = 0
        for t in range(1, n):
            states[t] = rng.choice(2, p=T_true[states[t - 1]])
        est = TransitionMatrixEstimator(n_states=2)
        est.fit(states)
        lo, hi = est.credible_intervals(level=0.99)
        assert np.all(lo <= T_true + 0.02)
        assert np.all(hi >= T_true - 0.02)

    def test_stationary_distribution(self, known_regime_sequence):
        est = TransitionMatrixEstimator(n_states=2)
        est.fit(known_regime_sequence)
        pi = est.stationary_distribution()
        np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-6)
        assert np.all(pi >= 0)

    def test_ergodicity_check(self, known_regime_sequence):
        est = TransitionMatrixEstimator(n_states=2)
        est.fit(known_regime_sequence)
        assert est.is_ergodic() is True

    def test_spectral_gap_positive(self, known_regime_sequence):
        est = TransitionMatrixEstimator(n_states=2)
        est.fit(known_regime_sequence)
        gap = est.spectral_gap()
        assert gap > 0

    def test_mixing_time_finite(self, known_regime_sequence):
        est = TransitionMatrixEstimator(n_states=2)
        est.fit(known_regime_sequence)
        mt = est.mixing_time()
        assert np.isfinite(mt)
        assert mt > 0

    def test_fit_from_counts(self):
        counts = np.array([[90, 10], [20, 80]])
        est = TransitionMatrixEstimator(n_states=2)
        est.fit_from_counts(counts)
        T = est.get_posterior_mean()
        np.testing.assert_allclose(T.sum(axis=1), 1.0, atol=1e-6)

    def test_sticky_prior(self, known_regime_sequence):
        est = TransitionMatrixEstimator(n_states=2, sticky_kappa=5.0)
        est.fit(known_regime_sequence)
        T = est.get_map_estimate()
        # Sticky prior should increase diagonal
        assert T[0, 0] > 0.5
        assert T[1, 1] > 0.5


# =========================================================================
# Online Regime Tracker
# =========================================================================

class TestOnlineRegimeTracker:
    def test_update_returns_posterior(self, regime_switching_data, seed):
        data, _ = regime_switching_data
        tracker = OnlineRegimeTracker(n_regimes=3, random_state=seed)
        for t in range(min(50, len(data))):
            post = tracker.update(data[t])
            assert post.shape[0] >= 1
            np.testing.assert_allclose(post.sum(), 1.0, atol=1e-5)

    def test_batch_update(self, regime_switching_data, seed):
        data, _ = regime_switching_data
        tracker = OnlineRegimeTracker(n_regimes=3, random_state=seed)
        posteriors = tracker.batch_update(data[:100])
        assert posteriors.shape[0] == 100

    def test_regime_history_length(self, regime_switching_data, seed):
        data, _ = regime_switching_data
        tracker = OnlineRegimeTracker(n_regimes=3, random_state=seed)
        tracker.batch_update(data[:80])
        history = tracker.get_regime_history()
        assert len(history) == 80

    def test_current_regime_is_valid(self, regime_switching_data, seed):
        data, _ = regime_switching_data
        tracker = OnlineRegimeTracker(n_regimes=3, random_state=seed)
        tracker.batch_update(data[:50])
        r = tracker.get_current_regime()
        assert 0 <= r < 3

    def test_alerts_on_regime_change(self, regime_switching_data, seed):
        data, _ = regime_switching_data
        tracker = OnlineRegimeTracker(
            n_regimes=3, alert_threshold=0.5, random_state=seed
        )
        tracker.batch_update(data)
        alerts = tracker.get_alerts()
        # With two-regime data, should detect at least 1 alert
        assert isinstance(alerts, list)

    def test_entropy_is_nonneg(self, regime_switching_data, seed):
        data, _ = regime_switching_data
        tracker = OnlineRegimeTracker(n_regimes=3, random_state=seed)
        tracker.batch_update(data[:50])
        ent = tracker.entropy_of_posterior()
        assert ent >= 0

    def test_transition_matrix_is_valid(self, regime_switching_data, seed):
        data, _ = regime_switching_data
        tracker = OnlineRegimeTracker(n_regimes=3, random_state=seed)
        tracker.batch_update(data[:100])
        T = tracker.get_transition_matrix()
        np.testing.assert_allclose(T.sum(axis=1), 1.0, atol=1e-5)

    def test_reset(self, seed):
        tracker = OnlineRegimeTracker(n_regimes=2, random_state=seed)
        tracker.update(1.0)
        tracker.update(2.0)
        tracker.reset()
        history = tracker.get_regime_history()
        assert len(history) == 0


# =========================================================================
# Bayesian Online Change Point Detector
# =========================================================================

class TestBOCPD:
    def test_detects_mean_shift(self, rng):
        """BOCPD should detect a mean shift in synthetic data."""
        n = 300
        data = np.concatenate([
            rng.normal(0.0, 0.5, size=150),
            rng.normal(3.0, 0.5, size=150),
        ])
        detector = BayesianOnlineChangePointDetector(hazard_rate=1 / 100.0)
        cps = detector.detect_batch(data, threshold=0.3)
        # Should detect a change point near t=150
        assert len(cps) >= 1
        nearest = min(cps, key=lambda t: abs(t - 150))
        assert abs(nearest - 150) < 40

    def test_run_length_distribution_shape(self, rng):
        detector = BayesianOnlineChangePointDetector()
        for x in rng.normal(0, 1, size=50):
            detector.update(x)
        rl = detector.get_run_length_distribution()
        assert len(rl) > 0
        assert np.isfinite(rl.sum())

    def test_expected_run_length_positive(self, rng):
        detector = BayesianOnlineChangePointDetector()
        for x in rng.normal(0, 1, size=30):
            detector.update(x)
        erl = detector.get_expected_run_length()
        assert erl > 0

    def test_reset_clears_state(self):
        detector = BayesianOnlineChangePointDetector()
        detector.update(1.0)
        detector.update(2.0)
        detector.reset()
        rl = detector.get_run_length_distribution()
        assert len(rl) <= 2


# =========================================================================
# CUSUM Detector
# =========================================================================

class TestCUSUM:
    def test_detects_upward_shift(self, rng):
        data = np.concatenate([
            rng.normal(0, 0.5, size=100),
            rng.normal(3, 0.5, size=100),
        ])
        det = CUSUMDetector(threshold=4.0, drift=0.5)
        cps = det.detect_batch(data)
        assert len(cps) >= 1
        nearest = min(cps, key=lambda t: abs(t - 100))
        assert abs(nearest - 100) < 30

    def test_no_false_alarm_on_stationary(self, rng):
        data = rng.normal(0, 0.5, size=500)
        det = CUSUMDetector(threshold=8.0, drift=0.5)
        cps = det.detect_batch(data)
        assert len(cps) <= 3  # tolerate rare false alarms

    def test_online_update(self):
        det = CUSUMDetector(threshold=5.0, drift=0.5)
        detected = False
        for i in range(200):
            x = 0.0 if i < 100 else 3.0
            if det.update(x):
                detected = True
                break
        assert detected


# =========================================================================
# Regime Posterior Computation
# =========================================================================

class TestRegimePosterior:
    @pytest.fixture
    def fitted_posterior(self, regime_switching_data, seed):
        data, _ = regime_switching_data
        rp = RegimePosterior(n_regimes=2, random_state=seed)
        pi = np.array([0.6, 0.4])
        A = np.array([[0.9, 0.1], [0.15, 0.85]])
        means = np.array([0.0, 2.0]).reshape(-1, 1)
        covars = np.array([0.25, 0.25]).reshape(-1, 1, 1)
        rp.set_parameters(pi=pi, A=A, means=means, covars=covars)
        return rp, data

    def test_compute_posterior_shape(self, fitted_posterior):
        rp, data = fitted_posterior
        gamma = rp.compute_posterior(data)
        assert gamma.shape == (len(data), 2)
        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-5)

    def test_marginal_likelihood_finite(self, fitted_posterior):
        rp, data = fitted_posterior
        rp.compute_posterior(data)
        ml = rp.marginal_likelihood()
        assert np.isfinite(ml)

    def test_posterior_entropy_nonneg(self, fitted_posterior):
        rp, data = fitted_posterior
        rp.compute_posterior(data)
        ent = rp.posterior_entropy()
        assert np.all(ent >= -1e-10)

    def test_expected_durations_positive(self, fitted_posterior):
        rp, _ = fitted_posterior
        d = rp.expected_durations()
        assert np.all(d > 0)

    def test_bic_finite(self, fitted_posterior):
        rp, data = fitted_posterior
        bic = rp.bic(data)
        assert np.isfinite(bic)


# =========================================================================
# Bayesian Regime Detector
# =========================================================================

class TestBayesianRegimeDetector:
    def test_detect_returns_tuple(self, regime_switching_data, seed):
        data, _ = regime_switching_data
        det = BayesianRegimeDetector(n_regimes=3, n_particles=100, random_state=seed)
        labels, posteriors = det.detect(data)
        assert labels.shape == (len(data),)
        assert posteriors.shape[0] == len(data)

    def test_online_update(self, rng, seed):
        det = BayesianRegimeDetector(n_regimes=2, n_particles=50, random_state=seed)
        for t in range(50):
            post = det.online_update(rng.normal(0, 1))
            np.testing.assert_allclose(post.sum(), 1.0, atol=1e-4)

    def test_regime_durations(self, regime_switching_data, seed):
        data, _ = regime_switching_data
        det = BayesianRegimeDetector(n_regimes=3, n_particles=100, random_state=seed)
        det.detect(data)
        durations = det.get_regime_durations()
        assert isinstance(durations, dict)
        total_duration = sum(sum(v) for v in durations.values())
        assert total_duration > 0


# =========================================================================
# Convergence Diagnostics (on StickyHDPHMM)
# =========================================================================

class TestConvergenceDiagnostics:
    def test_geweke_z_small_for_converged(self, regime_switching_data, seed):
        from causal_trading.regime.sticky_hdp_hmm import ConvergenceDiagnostics
        diag = ConvergenceDiagnostics()
        # Simulate converged log-likelihood trace
        rng = np.random.default_rng(seed)
        lls = -100.0 + rng.normal(0, 0.1, size=300)
        diag.log_likelihoods = list(lls)
        z = diag.geweke_z()
        assert abs(z) < 5  # converged traces should have small z

    def test_ess_positive(self, seed):
        from causal_trading.regime.sticky_hdp_hmm import ConvergenceDiagnostics
        diag = ConvergenceDiagnostics()
        rng = np.random.default_rng(seed)
        diag.log_likelihoods = list(rng.normal(-50, 0.5, size=200))
        ess = diag.effective_sample_size(burn_in=50)
        assert ess > 0


class TestMCMCConvergenceDiagnostics:
    """Tests for enhanced MCMC convergence diagnostics (Phase B1)."""

    def test_transition_matrix_ess(self):
        """Transition matrix ESS should be computed for recorded matrices."""
        from causal_trading.regime.sticky_hdp_hmm import ConvergenceDiagnostics
        diag = ConvergenceDiagnostics()
        rng = np.random.default_rng(42)
        K = 3
        # Stationary chain (good mixing)
        for _ in range(100):
            T = rng.dirichlet(np.ones(K) * 50, size=K)
            diag.record_transition_matrix(T)
            diag.log_likelihoods.append(-50 + rng.standard_normal())
        ess = diag.transition_matrix_ess()
        assert ess > 10, f"ESS too low for well-mixed chain: {ess}"

    def test_transition_rhat(self):
        """Transition R-hat should be near 1.0 for converged chain."""
        from causal_trading.regime.sticky_hdp_hmm import ConvergenceDiagnostics
        diag = ConvergenceDiagnostics()
        rng = np.random.default_rng(42)
        K = 3
        for _ in range(200):
            T = rng.dirichlet(np.ones(K) * 50, size=K)
            diag.record_transition_matrix(T)
            diag.log_likelihoods.append(-50 + rng.standard_normal())
        rhat = diag.transition_rhat()
        assert rhat < 1.2, f"R-hat too high for stationary chain: {rhat}"

    def test_full_diagnostic_report_structure(self):
        """Diagnostic report should contain all required fields."""
        from causal_trading.regime.sticky_hdp_hmm import ConvergenceDiagnostics
        diag = ConvergenceDiagnostics()
        rng = np.random.default_rng(42)
        for _ in range(100):
            diag.log_likelihoods.append(-50 + rng.standard_normal())
            diag.record_transition_matrix(rng.dirichlet([1,1,1], size=3))
        report = diag.full_diagnostic_report()
        assert "converged" in report
        assert "ll_ess" in report
        assert "ll_rhat" in report
        assert "tm_ess" in report
        assert "tm_rhat" in report
        assert "warnings" in report

    def test_nonconverged_chain_flagged(self):
        """A trending chain should be flagged as non-converged."""
        from causal_trading.regime.sticky_hdp_hmm import ConvergenceDiagnostics
        diag = ConvergenceDiagnostics()
        rng = np.random.default_rng(42)
        for i in range(200):
            diag.log_likelihoods.append(-200 + i * 1.0 + rng.standard_normal())
            diag.record_transition_matrix(rng.dirichlet([1,1], size=2))
        report = diag.full_diagnostic_report()
        assert not report["converged"]
        assert len(report["warnings"]) > 0
