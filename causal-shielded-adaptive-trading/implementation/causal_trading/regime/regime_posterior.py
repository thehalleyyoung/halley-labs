"""
Regime posterior computation: full posterior P(Z | X), marginal likelihood,
posterior predictive distributions, credible intervals for regime assignments,
and model comparison metrics (BIC, WAIC, LOO-CV).

References
----------
- Gelman, A. et al. (2013). Bayesian Data Analysis, 3rd edition.
- Vehtari, A., Gelman, A., & Gabry, J. (2017).
  Practical Bayesian model evaluation using leave-one-out cross-validation
  and WAIC. Statistics and Computing, 27(5), 1413-1432.
- Watanabe, S. (2010). Asymptotic equivalence of Bayes cross validation
  and widely applicable information criterion in singular learning theory.
  JMLR, 11, 3571-3594.
"""

from __future__ import annotations

from typing import Optional, Tuple, List, Dict, Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.special import logsumexp, gammaln, digamma


class RegimePosterior:
    """Full posterior computation for regime models.

    Given a fitted HMM (transition matrix, emission parameters, initial
    distribution), computes the posterior over hidden state sequences,
    marginal likelihoods, predictive distributions, and model comparison
    criteria.

    Parameters
    ----------
    n_regimes : int
        Number of regimes.
    n_posterior_samples : int
        Number of MCMC samples for posterior summaries.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_regimes: int = 5,
        n_posterior_samples: int = 1000,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_regimes = n_regimes
        self.n_posterior_samples = n_posterior_samples
        self.random_state = random_state

        self._rng = np.random.default_rng(random_state)

        # Model parameters (set via set_parameters or fit)
        self._pi: Optional[NDArray] = None         # (K,) initial dist
        self._A: Optional[NDArray] = None           # (K, K) transition
        self._means: Optional[NDArray] = None       # (K, D) emission means
        self._covars: Optional[NDArray] = None      # (K, D, D) emission covars

        # Cached results
        self._log_alpha: Optional[NDArray] = None
        self._log_beta: Optional[NDArray] = None
        self._log_gamma: Optional[NDArray] = None
        self._log_xi: Optional[NDArray] = None
        self._log_evidence: Optional[float] = None
        self._state_samples: Optional[NDArray] = None  # (S, T) samples
        self._X: Optional[NDArray] = None

    # ------------------------------------------------------------------
    # Parameter setting
    # ------------------------------------------------------------------

    def set_parameters(
        self,
        pi: NDArray,
        A: NDArray,
        means: NDArray,
        covars: NDArray,
    ) -> "RegimePosterior":
        """Set model parameters directly.

        Parameters
        ----------
        pi : (K,) initial state distribution
        A : (K, K) transition matrix
        means : (K, D) emission means
        covars : (K, D, D) emission covariance matrices
        """
        self._pi = np.asarray(pi, dtype=np.float64)
        self._A = np.asarray(A, dtype=np.float64)
        self._means = np.asarray(means, dtype=np.float64)
        self._covars = np.asarray(covars, dtype=np.float64)
        self.n_regimes = len(pi)
        # Invalidate cache
        self._log_alpha = None
        self._log_beta = None
        self._log_gamma = None
        self._log_xi = None
        self._log_evidence = None
        self._state_samples = None
        return self

    # ------------------------------------------------------------------
    # Emission log-likelihood
    # ------------------------------------------------------------------

    def _compute_log_lik(self, X: NDArray) -> NDArray:
        """Compute (T, K) log-emission likelihood matrix."""
        T = X.shape[0]
        D = X.shape[1] if X.ndim > 1 else 1
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        K = self.n_regimes
        log_lik = np.zeros((T, K))
        for k in range(K):
            mu = self._means[k]
            if self._covars.ndim == 3:
                cov = self._covars[k]
            else:
                cov = np.diag(self._covars[k]) if self._covars[k].ndim == 1 else self._covars[k]
            cov = 0.5 * (cov + cov.T) + 1e-8 * np.eye(D)
            try:
                log_lik[:, k] = stats.multivariate_normal.logpdf(X, mean=mu, cov=cov)
            except np.linalg.LinAlgError:
                log_lik[:, k] = -1e10
        return log_lik

    # ------------------------------------------------------------------
    # Forward-backward
    # ------------------------------------------------------------------

    def _forward(self, log_pi: NDArray, log_A: NDArray,
                 log_lik: NDArray) -> Tuple[NDArray, float]:
        T, K = log_lik.shape
        log_alpha = np.full((T, K), -np.inf)
        log_alpha[0] = log_pi + log_lik[0]
        for t in range(1, T):
            for k in range(K):
                log_alpha[t, k] = logsumexp(log_alpha[t - 1] + log_A[:, k]) + log_lik[t, k]
        return log_alpha, float(logsumexp(log_alpha[-1]))

    def _backward(self, log_A: NDArray, log_lik: NDArray) -> NDArray:
        T, K = log_lik.shape
        log_beta = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = logsumexp(
                    log_A[k, :] + log_lik[t + 1] + log_beta[t + 1]
                )
        return log_beta

    def _run_forward_backward(self, X: NDArray) -> None:
        """Run full forward-backward and cache all results."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self._X = X
        T, D = X.shape
        K = self.n_regimes

        log_pi = np.log(self._pi + 1e-300)
        log_A = np.log(self._A + 1e-300)
        log_lik = self._compute_log_lik(X)

        self._log_alpha, self._log_evidence = self._forward(log_pi, log_A, log_lik)
        self._log_beta = self._backward(log_A, log_lik)

        # State posteriors gamma_t(k) = P(z_t = k | X)
        self._log_gamma = self._log_alpha + self._log_beta
        self._log_gamma -= logsumexp(self._log_gamma, axis=1, keepdims=True)

        # Pairwise marginals xi_t(j, k) = P(z_t=j, z_{t+1}=k | X)
        self._log_xi = np.full((T - 1, K, K), -np.inf)
        for t in range(T - 1):
            for j in range(K):
                for k in range(K):
                    self._log_xi[t, j, k] = (
                        self._log_alpha[t, j]
                        + log_A[j, k]
                        + log_lik[t + 1, k]
                        + self._log_beta[t + 1, k]
                    )
            self._log_xi[t] -= logsumexp(self._log_xi[t].ravel())

    # ------------------------------------------------------------------
    # Public posterior methods
    # ------------------------------------------------------------------

    def compute_posterior(self, X: NDArray) -> NDArray:
        """Compute the full posterior P(z_t = k | X) for all t.

        Parameters
        ----------
        X : (T,) or (T, D) observations

        Returns
        -------
        gamma : (T, K) state posteriors
        """
        self._run_forward_backward(X)
        return np.exp(self._log_gamma)

    def marginal_likelihood(self, X: Optional[NDArray] = None) -> float:
        """Compute log P(X | model) = log marginal likelihood.

        Parameters
        ----------
        X : observations (optional; uses cached if available)

        Returns
        -------
        log_evidence : float
        """
        if X is not None:
            self._run_forward_backward(X)
        if self._log_evidence is None:
            raise RuntimeError("Run compute_posterior(X) first.")
        return self._log_evidence

    def pairwise_marginals(self, X: Optional[NDArray] = None) -> NDArray:
        """Compute pairwise marginals P(z_t=j, z_{t+1}=k | X).

        Returns
        -------
        xi : (T-1, K, K) pairwise marginal probabilities
        """
        if X is not None:
            self._run_forward_backward(X)
        if self._log_xi is None:
            raise RuntimeError("Run compute_posterior(X) first.")
        return np.exp(self._log_xi)

    # ------------------------------------------------------------------
    # Posterior sampling (forward-filter backward-sample)
    # ------------------------------------------------------------------

    def sample_state_sequences(
        self, X: NDArray, n_samples: Optional[int] = None
    ) -> NDArray:
        """Draw state sequences from the posterior P(Z | X) via FFBS.

        Parameters
        ----------
        X : (T,) or (T, D) observations
        n_samples : int (default: self.n_posterior_samples)

        Returns
        -------
        samples : (n_samples, T) state sequence samples
        """
        if n_samples is None:
            n_samples = self.n_posterior_samples
        self._run_forward_backward(X)
        T, K = self._log_alpha.shape
        log_A = np.log(self._A + 1e-300)

        samples = np.zeros((n_samples, T), dtype=np.int64)
        for s in range(n_samples):
            # Sample z_T
            log_p = self._log_alpha[-1].copy()
            log_p -= logsumexp(log_p)
            p = np.exp(log_p)
            p = np.clip(p, 0, None)
            p /= p.sum()
            samples[s, -1] = self._rng.choice(K, p=p)

            # Backward sample
            for t in range(T - 2, -1, -1):
                log_p = self._log_alpha[t] + log_A[:, samples[s, t + 1]]
                log_p -= logsumexp(log_p)
                p = np.exp(log_p)
                p = np.clip(p, 0, None)
                p /= p.sum()
                samples[s, t] = self._rng.choice(K, p=p)

        self._state_samples = samples
        return samples

    # ------------------------------------------------------------------
    # Credible intervals for regime assignments
    # ------------------------------------------------------------------

    def credible_intervals(
        self, X: NDArray, level: float = 0.95, n_samples: Optional[int] = None
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Compute credible intervals for regime assignment probabilities.

        For each time step t, we report the MAP regime and the
        probability interval around it.

        Parameters
        ----------
        X : observations
        level : credible level
        n_samples : number of posterior samples

        Returns
        -------
        map_states : (T,) MAP regime assignments
        lower : (T, K) lower credible bounds on P(z_t=k)
        upper : (T, K) upper credible bounds on P(z_t=k)
        """
        gamma = self.compute_posterior(X)
        T, K = gamma.shape

        # For credible intervals, use posterior samples
        if n_samples is None:
            n_samples = self.n_posterior_samples
        samples = self.sample_state_sequences(X, n_samples)

        # For each t, compute empirical distribution over states
        lower = np.zeros((T, K))
        upper = np.zeros((T, K))
        map_states = np.argmax(gamma, axis=1)

        alpha_lo = (1.0 - level) / 2.0
        alpha_hi = 1.0 - alpha_lo

        for t in range(T):
            counts = np.bincount(samples[:, t], minlength=K).astype(np.float64)
            probs = counts / n_samples
            # Bootstrap-style CI using the Wilson score interval
            for k in range(K):
                p_hat = probs[k]
                n = n_samples
                z = stats.norm.ppf(alpha_hi)
                denom = 1.0 + z ** 2 / n
                center = (p_hat + z ** 2 / (2.0 * n)) / denom
                halfwidth = z * np.sqrt(
                    (p_hat * (1 - p_hat) + z ** 2 / (4.0 * n)) / n
                ) / denom
                lower[t, k] = max(0.0, center - halfwidth)
                upper[t, k] = min(1.0, center + halfwidth)

        return map_states, lower, upper

    # ------------------------------------------------------------------
    # Posterior predictive distribution
    # ------------------------------------------------------------------

    def posterior_predictive(
        self, X: NDArray, n_ahead: int = 1
    ) -> Tuple[NDArray, NDArray]:
        """Posterior predictive distribution P(x_{T+h} | X) for h=1..n_ahead.

        Returns the mean and variance of the predictive distribution
        at each step ahead, marginalised over the regime posterior.

        Parameters
        ----------
        X : (T,) or (T, D) observations
        n_ahead : int
            Number of steps ahead to predict.

        Returns
        -------
        pred_means : (n_ahead, D) predictive means
        pred_vars : (n_ahead, D, D) predictive covariance matrices
        """
        gamma = self.compute_posterior(X)
        K = self.n_regimes
        D = self._means.shape[1] if self._means.ndim > 1 else 1

        # Posterior over z_T
        p_zT = gamma[-1]  # (K,)

        pred_means = np.zeros((n_ahead, D))
        pred_vars = np.zeros((n_ahead, D, D))

        p_z = p_zT.copy()
        for h in range(n_ahead):
            # Propagate through transition matrix
            p_z_next = p_z @ self._A  # (K,)

            # Predictive mean: E[x] = sum_k p(z=k) mu_k
            mean = np.zeros(D)
            for k in range(K):
                mean += p_z_next[k] * self._means[k].ravel()[:D]
            pred_means[h] = mean

            # Predictive variance: law of total variance
            # Var[x] = E[Var[x|z]] + Var[E[x|z]]
            E_var = np.zeros((D, D))
            for k in range(K):
                if self._covars.ndim == 3:
                    E_var += p_z_next[k] * self._covars[k][:D, :D]
                else:
                    E_var += p_z_next[k] * np.eye(D)

            var_E = np.zeros((D, D))
            for k in range(K):
                diff = self._means[k].ravel()[:D] - mean
                var_E += p_z_next[k] * np.outer(diff, diff)

            pred_vars[h] = E_var + var_E
            p_z = p_z_next

        return pred_means, pred_vars

    # ------------------------------------------------------------------
    # Model comparison: BIC
    # ------------------------------------------------------------------

    def bic(self, X: NDArray) -> float:
        """Bayesian Information Criterion.

        BIC = -2 * log P(X | theta_MAP) + p * log(T)

        where p is the number of free parameters.

        Parameters
        ----------
        X : observations

        Returns
        -------
        bic_value : float (lower is better)
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        T, D = X.shape
        K = self.n_regimes

        log_lik = self.marginal_likelihood(X)

        # Number of free parameters
        n_transition = K * (K - 1)         # transition matrix
        n_initial = K - 1                   # initial distribution
        n_emission = K * D + K * D * (D + 1) // 2  # means + covariances
        n_params = n_transition + n_initial + n_emission

        return -2.0 * log_lik + n_params * np.log(T)

    # ------------------------------------------------------------------
    # Model comparison: WAIC
    # ------------------------------------------------------------------

    def waic(self, X: NDArray, n_samples: Optional[int] = None) -> Tuple[float, float]:
        """Widely Applicable Information Criterion (Watanabe 2010).

        WAIC = -2 * (lppd - p_waic)

        Parameters
        ----------
        X : observations
        n_samples : number of posterior samples

        Returns
        -------
        waic_value : float (lower is better)
        p_waic : float (effective number of parameters)
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        T, D = X.shape

        if n_samples is None:
            n_samples = min(self.n_posterior_samples, 500)
        samples = self.sample_state_sequences(X, n_samples)

        # Compute log-pointwise predictive density
        log_lik_matrix = np.zeros((n_samples, T))
        for s in range(n_samples):
            for t in range(T):
                k = samples[s, t]
                mu = self._means[k]
                if self._covars.ndim == 3:
                    cov = self._covars[k]
                else:
                    cov = np.eye(D)
                cov = 0.5 * (cov + cov.T) + 1e-8 * np.eye(D)
                try:
                    log_lik_matrix[s, t] = stats.multivariate_normal.logpdf(
                        X[t], mean=mu, cov=cov
                    )
                except Exception:
                    log_lik_matrix[s, t] = -1e10

        # lppd = sum_t log(1/S * sum_s p(y_t | theta_s))
        lppd = np.sum(logsumexp(log_lik_matrix, axis=0) - np.log(n_samples))

        # p_waic2 = sum_t Var_s[log p(y_t | theta_s)]
        p_waic = float(np.sum(np.var(log_lik_matrix, axis=0, ddof=1)))

        waic_val = -2.0 * (lppd - p_waic)
        return float(waic_val), p_waic

    # ------------------------------------------------------------------
    # Model comparison: LOO-CV (PSIS-LOO)
    # ------------------------------------------------------------------

    def loo_cv(self, X: NDArray, n_samples: Optional[int] = None) -> Tuple[float, NDArray]:
        """Approximate leave-one-out cross-validation via Pareto-smoothed
        importance sampling (PSIS-LOO).

        Parameters
        ----------
        X : observations
        n_samples : number of posterior samples

        Returns
        -------
        loo_value : float (estimated out-of-sample log predictive density)
        loo_pointwise : (T,) per-observation LOO log-densities
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        T, D = X.shape

        if n_samples is None:
            n_samples = min(self.n_posterior_samples, 500)
        samples = self.sample_state_sequences(X, n_samples)

        # Compute log-likelihood for each sample and observation
        log_lik_matrix = np.zeros((n_samples, T))
        for s in range(n_samples):
            for t in range(T):
                k = samples[s, t]
                mu = self._means[k]
                if self._covars.ndim == 3:
                    cov = self._covars[k]
                else:
                    cov = np.eye(D)
                cov = 0.5 * (cov + cov.T) + 1e-8 * np.eye(D)
                try:
                    log_lik_matrix[s, t] = stats.multivariate_normal.logpdf(
                        X[t], mean=mu, cov=cov
                    )
                except Exception:
                    log_lik_matrix[s, t] = -1e10

        # Importance weights for LOO: w_s^{-t} = 1 / p(y_t | theta_s)
        # log(w_s^{-t}) = -log_lik_matrix[s, t]
        loo_pointwise = np.zeros(T)
        for t in range(T):
            log_ratios = -log_lik_matrix[:, t]
            # Pareto smoothing of the tail
            log_ratios_sorted = np.sort(log_ratios)
            # Simple truncation instead of full PSIS
            M = max(int(min(n_samples * 0.2, 3 * np.sqrt(n_samples))), 5)
            tail = log_ratios_sorted[-M:]
            # Fit generalized Pareto to the tail
            if len(tail) > 1 and np.std(tail) > 1e-10:
                threshold = log_ratios_sorted[-M - 1] if M < n_samples else log_ratios_sorted[0]
                exceedances = tail - threshold
                # Method of moments for GPD shape
                mean_exc = np.mean(exceedances)
                var_exc = np.var(exceedances)
                if mean_exc > 0 and var_exc > 0:
                    xi = 0.5 * (1.0 - mean_exc ** 2 / var_exc)
                    xi = np.clip(xi, -0.5, 1.0)
                else:
                    xi = 0.0
                # Replace tail with smoothed values
                if xi < 0.7:  # k-hat diagnostic: good if < 0.7
                    # Use smoothed weights
                    for i in range(M):
                        p_i = (i + 0.5) / M
                        if abs(xi) > 1e-6:
                            smoothed = threshold + mean_exc * ((1 - p_i) ** (-xi) - 1) / xi
                        else:
                            smoothed = threshold - mean_exc * np.log(1 - p_i)
                        log_ratios[np.argsort(log_ratios)[-(i + 1)]] = smoothed

            # Normalise importance weights
            log_w = log_ratios - logsumexp(log_ratios)
            # LOO estimate
            loo_pointwise[t] = logsumexp(log_w + log_lik_matrix[:, t])

        loo_value = float(np.sum(loo_pointwise))
        return loo_value, loo_pointwise

    # ------------------------------------------------------------------
    # Posterior entropy and uncertainty
    # ------------------------------------------------------------------

    def posterior_entropy(self, X: Optional[NDArray] = None) -> NDArray:
        """Shannon entropy of the state posterior at each time step.

        Returns
        -------
        H : (T,) entropy values (higher = more uncertain)
        """
        if X is not None:
            gamma = self.compute_posterior(X)
        elif self._log_gamma is not None:
            gamma = np.exp(self._log_gamma)
        else:
            raise RuntimeError("Run compute_posterior(X) first.")
        gamma_safe = np.clip(gamma, 1e-300, None)
        return -np.sum(gamma_safe * np.log(gamma_safe), axis=1)

    def mutual_information_states(self, X: Optional[NDArray] = None) -> float:
        """Mutual information I(z_t; z_{t+1} | X) averaged over time.

        Measures how much knowing the current regime tells us about
        the next regime.
        """
        if X is not None:
            self._run_forward_backward(X)
        if self._log_gamma is None or self._log_xi is None:
            raise RuntimeError("Run compute_posterior(X) first.")

        gamma = np.exp(self._log_gamma)
        xi = np.exp(self._log_xi)
        T = gamma.shape[0]
        K = gamma.shape[1]

        mi_sum = 0.0
        for t in range(T - 1):
            for j in range(K):
                for k in range(K):
                    if xi[t, j, k] > 1e-300:
                        mi_sum += xi[t, j, k] * (
                            np.log(xi[t, j, k] + 1e-300)
                            - np.log(gamma[t, j] + 1e-300)
                            - np.log(gamma[t + 1, k] + 1e-300)
                        )
        return float(mi_sum / max(T - 1, 1))

    # ------------------------------------------------------------------
    # Expected regime durations
    # ------------------------------------------------------------------

    def expected_durations(self) -> NDArray:
        """Expected duration in each regime under the transition matrix.

        For a geometric sojourn distribution:
        E[duration_k] = 1 / (1 - A[k,k])

        Returns
        -------
        durations : (K,) expected durations
        """
        if self._A is None:
            raise RuntimeError("Set parameters first.")
        diag = np.diag(self._A)
        return 1.0 / (1.0 - diag + 1e-300)

    # ------------------------------------------------------------------
    # Compare multiple models
    # ------------------------------------------------------------------

    @staticmethod
    def compare_models(
        models: List["RegimePosterior"],
        X: NDArray,
        criterion: str = "bic",
    ) -> Dict[str, Any]:
        """Compare multiple regime models on the same data.

        Parameters
        ----------
        models : list of RegimePosterior instances
        X : observations
        criterion : one of 'bic', 'waic', 'loo'

        Returns
        -------
        results : dict with rankings and scores
        """
        scores = []
        for i, model in enumerate(models):
            if criterion == "bic":
                score = model.bic(X)
            elif criterion == "waic":
                score, _ = model.waic(X)
            elif criterion == "loo":
                score, _ = model.loo_cv(X)
                score = -score  # negate so lower is better
            else:
                raise ValueError(f"Unknown criterion: {criterion}")
            scores.append(score)

        scores_arr = np.array(scores)
        ranking = np.argsort(scores_arr)
        best = int(ranking[0])
        delta = scores_arr - scores_arr[best]

        # Akaike-style weights (for BIC/WAIC)
        log_weights = -0.5 * delta
        log_weights -= logsumexp(log_weights)
        weights = np.exp(log_weights)

        return {
            "criterion": criterion,
            "scores": scores_arr,
            "ranking": ranking,
            "best_model": best,
            "delta": delta,
            "weights": weights,
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, X: NDArray) -> Dict[str, Any]:
        """Comprehensive summary of the posterior analysis.

        Parameters
        ----------
        X : observations

        Returns
        -------
        summary_dict : dict
        """
        gamma = self.compute_posterior(X)
        map_states = np.argmax(gamma, axis=1)
        n_active = len(np.unique(map_states))
        entropy = self.posterior_entropy()

        return {
            "n_regimes": self.n_regimes,
            "n_active_regimes": n_active,
            "log_marginal_likelihood": self._log_evidence,
            "bic": self.bic(X),
            "mean_posterior_entropy": float(entropy.mean()),
            "max_posterior_entropy": float(entropy.max()),
            "expected_durations": self.expected_durations(),
            "state_frequencies": np.bincount(map_states, minlength=self.n_regimes) / len(map_states),
        }

    def __repr__(self) -> str:
        has_params = self._pi is not None
        return (
            f"RegimePosterior(n_regimes={self.n_regimes}, "
            f"has_parameters={has_params})"
        )
