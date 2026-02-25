"""
Spiked random matrix models and BBP phase transition.

Implements the Baik-Ben Arous-Péché transition for spiked covariance
matrices, signal detection thresholds, and spiked NTK models for
studying how finite-width neural networks detect planted features.
"""

import numpy as np
from scipy import stats


class SpikedCovarianceModel:
    """Spiked covariance model: Σ = σ²I + Σ_k θ_k v_k v_k^T.

    Models a population covariance with a few eigenvalues (spikes)
    separated from the bulk σ² background.

    Parameters
    ----------
    N : int
        Number of samples.
    P : int
        Dimension.
    spikes : list of float
        Spike strengths θ₁, θ₂, ... (added to σ²).
    sigma_sq : float
        Noise variance.
    """

    def __init__(self, N: int, P: int, spikes: list, sigma_sq: float = 1.0):
        self.N = N
        self.P = P
        self.spikes = list(spikes)
        self.sigma_sq = sigma_sq
        self.gamma = N / P

    def population_eigenvalues(self) -> np.ndarray:
        """Return the population covariance eigenvalues.

        The population covariance Σ has eigenvalues σ² + θ_k for
        k = 1, ..., r and σ² for the remaining P - r dimensions.

        Returns
        -------
        np.ndarray
            All P population eigenvalues (sorted descending).
        """
        eigs = np.full(self.P, self.sigma_sq)
        for i, theta in enumerate(self.spikes):
            if i < self.P:
                eigs[i] = self.sigma_sq + theta
        return np.sort(eigs)[::-1]

    def sample_eigenvalues(self, n_trials: int = 100) -> np.ndarray:
        """Sample eigenvalues from the spiked model.

        Generates X ~ N(0, Σ) and computes eigenvalues of (1/N) X^T X.

        Parameters
        ----------
        n_trials : int
            Number of independent trials.

        Returns
        -------
        np.ndarray
            Array of shape (n_trials, P) with eigenvalues per trial.
        """
        pop_eigs = self.population_eigenvalues()
        sqrt_cov = np.diag(np.sqrt(pop_eigs))

        all_eigs = np.zeros((n_trials, self.P))
        for trial in range(n_trials):
            Z = np.random.randn(self.N, self.P)
            X = Z @ sqrt_cov
            sample_cov = X.T @ X / self.N
            all_eigs[trial] = np.sort(np.linalg.eigvalsh(sample_cov))[::-1]

        return all_eigs

    def asymptotic_spike_locations(self) -> list:
        """Predict asymptotic locations of outlier eigenvalues.

        Uses the BBP formula: if θ > σ²√γ (supercritical),
        the sample spike lands at σ²(1 + θ/σ²)(1 + γσ²/θ).

        Returns
        -------
        list of dict
            For each spike: {'theta': float, 'supercritical': bool, 'location': float}.
        """
        bbp = BBPTransition(self.gamma, self.sigma_sq)
        threshold = bbp.critical_threshold()
        results = []

        for theta in self.spikes:
            if theta > threshold:
                loc = bbp.spike_location(theta)
                results.append({
                    "theta": theta,
                    "supercritical": True,
                    "location": loc,
                })
            else:
                # Subcritical: spike merges with the bulk edge
                edge = self.sigma_sq * (1.0 + np.sqrt(self.gamma)) ** 2
                results.append({
                    "theta": theta,
                    "supercritical": False,
                    "location": edge,
                })

        return results


class BBPTransition:
    """Baik-Ben Arous-Péché phase transition for spiked random matrices.

    The BBP transition describes a sharp threshold: spikes with
    strength θ > θ_c = σ²√γ produce outlier eigenvalues, while
    weaker spikes are invisible in the sample spectrum.

    Parameters
    ----------
    gamma : float
        Aspect ratio N/P.
    sigma_sq : float
        Noise variance.
    """

    def __init__(self, gamma: float, sigma_sq: float = 1.0):
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        self.gamma = gamma
        self.sigma_sq = sigma_sq

    def critical_threshold(self) -> float:
        """BBP critical threshold θ_c = σ²√γ.

        Spikes with θ > θ_c produce outlier eigenvalues; spikes
        with θ ≤ θ_c are undetectable from the sample covariance.

        Returns
        -------
        float
            Critical spike strength.
        """
        return self.sigma_sq * np.sqrt(self.gamma)

    def spike_location(self, theta: float) -> float:
        """Predicted location of outlier eigenvalue for supercritical spike.

        For θ > θ_c: ℓ(θ) = σ²(1 + θ/σ²)(1 + γσ²/θ)
                            = (σ² + θ)(1 + γσ²/θ).

        Parameters
        ----------
        theta : float
            Spike strength.

        Returns
        -------
        float
            Asymptotic location of the corresponding sample eigenvalue.
        """
        if theta <= 0:
            raise ValueError("theta must be positive")
        s = self.sigma_sq
        g = self.gamma
        return (s + theta) * (1.0 + g * s / theta)

    def spike_eigenvector_overlap(self, theta: float) -> float:
        """Cosine-squared overlap between sample and population eigenvectors.

        For θ > θ_c:
        |⟨v̂, v⟩|² = (1 - γσ⁴/θ²) / (1 + γσ²/θ).

        For θ ≤ θ_c, the overlap vanishes asymptotically.

        Parameters
        ----------
        theta : float
            Spike strength.

        Returns
        -------
        float
            Squared cosine overlap ∈ [0, 1].
        """
        if theta <= self.critical_threshold():
            return 0.0
        s = self.sigma_sq
        g = self.gamma
        numerator = 1.0 - g * s ** 2 / theta ** 2
        denominator = 1.0 + g * s / theta
        return max(0.0, numerator / denominator)

    def phase_diagram(self, theta_range: np.ndarray) -> dict:
        """Compute the BBP phase diagram over a range of spike strengths.

        Parameters
        ----------
        theta_range : np.ndarray
            Spike strengths to evaluate.

        Returns
        -------
        dict
            Phase diagram data: spike locations, overlaps, and phase labels.
        """
        theta_range = np.asarray(theta_range, dtype=float)
        threshold = self.critical_threshold()
        edge = self.sigma_sq * (1.0 + np.sqrt(self.gamma)) ** 2

        locations = np.zeros_like(theta_range)
        overlaps = np.zeros_like(theta_range)
        phases = []

        for i, theta in enumerate(theta_range):
            if theta > threshold:
                locations[i] = self.spike_location(theta)
                overlaps[i] = self.spike_eigenvector_overlap(theta)
                phases.append("supercritical")
            else:
                locations[i] = edge
                overlaps[i] = 0.0
                phases.append("subcritical")

        return {
            "theta_range": theta_range,
            "critical_threshold": threshold,
            "spike_locations": locations,
            "eigenvector_overlaps": overlaps,
            "phases": phases,
            "bulk_edge": edge,
        }

    def detection_power(
        self, theta: float, N: int, alpha: float = 0.05
    ) -> float:
        """Statistical power of detecting a spike at significance level α.

        For a supercritical spike, the largest eigenvalue has
        Gaussian fluctuations of order N^{-1/2} around ℓ(θ).
        Power ≈ Φ((ℓ(θ) - λ⁺ - z_α σ_TW) / σ_TW).

        Parameters
        ----------
        theta : float
            Spike strength.
        N : int
            Sample size.
        alpha : float
            Significance level.

        Returns
        -------
        float
            Approximate detection power ∈ [0, 1].
        """
        threshold = self.critical_threshold()
        if theta <= threshold:
            return alpha  # no power beyond size

        edge = self.sigma_sq * (1.0 + np.sqrt(self.gamma)) ** 2
        spike_loc = self.spike_location(theta)

        # Fluctuation scale at the edge: O(N^{-2/3})
        sigma_edge = (1.0 + np.sqrt(self.gamma)) ** (4.0 / 3.0) * self.gamma ** (-1.0 / 6.0) * N ** (-2.0 / 3.0)

        # Critical value for TW test at level alpha
        # TW mean ≈ -1.77, std ≈ 0.81 (for β=2)
        tw_mean = -1.77
        tw_std = 0.81
        z_alpha = tw_mean + tw_std * stats.norm.ppf(1 - alpha)
        critical_value = edge + z_alpha * sigma_edge

        # Supercritical spike has Gaussian fluctuations of order N^{-1/2}
        sigma_spike = (spike_loc - edge) / np.sqrt(N) if N > 0 else 1.0
        sigma_spike = max(sigma_spike, sigma_edge)

        power = 1.0 - stats.norm.cdf(
            (critical_value - spike_loc) / sigma_spike
        )
        return float(np.clip(power, 0.0, 1.0))

    def transition_sharpness(
        self, theta_range: np.ndarray, N_values: list
    ) -> dict:
        """Measure how sharply the BBP transition occurs for finite N.

        Parameters
        ----------
        theta_range : np.ndarray
            Range of spike strengths around the threshold.
        N_values : list of int
            Sample sizes to evaluate.

        Returns
        -------
        dict
            Sharpness analysis across N values.
        """
        threshold = self.critical_threshold()
        results = {}

        for N in N_values:
            powers = np.array([self.detection_power(max(t, 1e-10), N) for t in theta_range])
            # Find the width of the transition region (10%–90% power)
            idx_10 = np.searchsorted(powers, 0.10)
            idx_90 = np.searchsorted(powers, 0.90)

            if idx_10 < len(theta_range) and idx_90 < len(theta_range):
                width = theta_range[min(idx_90, len(theta_range) - 1)] - theta_range[idx_10]
            else:
                width = float("inf")

            results[N] = {
                "powers": powers,
                "transition_width": float(width),
                "threshold": float(threshold),
            }

        return results


class SignalDetectionThreshold:
    """Optimal thresholds for signal detection in spiked models.

    Parameters
    ----------
    gamma : float
        Aspect ratio.
    sigma_sq : float
        Noise variance.
    """

    def __init__(self, gamma: float, sigma_sq: float = 1.0):
        self.gamma = gamma
        self.sigma_sq = sigma_sq
        self.bbp = BBPTransition(gamma, sigma_sq)

    def optimal_threshold(self) -> float:
        """Optimal hard threshold for eigenvalue-based signal detection.

        The optimal threshold sits at the upper MP edge plus a margin
        accounting for TW fluctuations.

        Returns
        -------
        float
            Optimal detection threshold for eigenvalues.
        """
        edge = self.sigma_sq * (1.0 + np.sqrt(self.gamma)) ** 2
        # Median of TW₂ ≈ -1.27; add this to account for fluctuations
        return edge

    def detection_boundary(self, n_signals_range: np.ndarray) -> np.ndarray:
        """Minimum detectable spike strength as a function of number of signals.

        For k signals, the effective aspect ratio changes slightly
        as the detected signals are deflated.

        Parameters
        ----------
        n_signals_range : np.ndarray
            Number of planted signals to consider.

        Returns
        -------
        np.ndarray
            Minimum detectable θ for each number of signals.
        """
        n_signals_range = np.asarray(n_signals_range, dtype=int)
        thresholds = np.zeros(len(n_signals_range))

        for i, k in enumerate(n_signals_range):
            # Effective ratio after removing k dimensions
            gamma_eff = self.gamma
            thresholds[i] = self.sigma_sq * np.sqrt(gamma_eff)

        return thresholds

    def information_theoretic_limit(self, theta: float) -> dict:
        """Information-theoretic limit for spike detection.

        The mutual information between the spike and the data
        determines the fundamental detection limit.

        For a rank-1 spike: I(v; X) = (1/2) log(1 + N·θ²/(σ⁴·(1+γ)))

        Parameters
        ----------
        theta : float
            Spike strength.

        Returns
        -------
        dict
            {'mutual_information': float, 'detectable': bool, 'snr': float}.
        """
        s = self.sigma_sq
        g = self.gamma
        snr = theta ** 2 / (s ** 2 * (1 + g))

        mutual_info = 0.5 * np.log(1.0 + snr)
        detectable = theta > self.bbp.critical_threshold()

        return {
            "mutual_information": float(mutual_info),
            "detectable": detectable,
            "snr": float(snr),
            "bbp_threshold": float(self.bbp.critical_threshold()),
        }


class SpikedNTKModel:
    """Spiked NTK model: how neural tangent kernels interact with data structure.

    Models how planted features in the data appear (or fail to appear)
    in the NTK spectrum as a function of network width.

    Parameters
    ----------
    N : int
        Number of samples.
    P : int
        Input dimension.
    d_model : int
        Network width (hidden layer size).
    """

    def __init__(self, N: int, P: int, d_model: int):
        self.N = N
        self.P = P
        self.d_model = d_model
        self.gamma = N / P

    def ntk_with_planted_features(
        self,
        X: np.ndarray,
        feature_directions: np.ndarray,
        strengths: np.ndarray,
    ) -> np.ndarray:
        """Compute the NTK matrix for data with planted features.

        X = signal + noise, where signal lives in the span of
        feature_directions with given strengths.

        The NTK at initialization (lazy regime) for a two-layer ReLU
        network is: Θ(x_i, x_j) = (1/m) x_i^T x_j (π - arccos(x̂_i · x̂_j))/(2π)
        where x̂ = x/‖x‖ and m is the width.

        Parameters
        ----------
        X : np.ndarray
            Data matrix (N × P).
        feature_directions : np.ndarray
            Feature directions (k × P), assumed orthonormal.
        strengths : np.ndarray
            Signal strengths for each feature.

        Returns
        -------
        np.ndarray
            NTK matrix (N × N).
        """
        N = X.shape[0]
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        X_hat = X / norms

        # NTK kernel for two-layer ReLU: K(x, y) = ‖x‖‖y‖ · k_arc(x̂·ŷ)
        # where k_arc(t) = (1/(2π)) (t(π - arccos(t)) + √(1 - t²))
        gram = X_hat @ X_hat.T
        gram = np.clip(gram, -1.0, 1.0)

        angle_part = (np.pi - np.arccos(gram)) / (2 * np.pi)
        sqrt_part = np.sqrt(np.maximum(1.0 - gram ** 2, 0.0)) / (2 * np.pi)

        ntk = (norms @ norms.T) * (gram * angle_part + sqrt_part) / self.d_model

        return ntk

    def feature_visibility(self, strengths: np.ndarray, width: int) -> dict:
        """Assess whether features are visible in the NTK spectrum.

        A feature with strength θ is visible if θ exceeds the BBP
        threshold for the effective aspect ratio γ_eff = N/width.

        Parameters
        ----------
        strengths : np.ndarray
            Feature strengths.
        width : int
            Network width.

        Returns
        -------
        dict
            Visibility assessment for each feature.
        """
        gamma_eff = self.N / width
        bbp = BBPTransition(gamma_eff)
        threshold = bbp.critical_threshold()

        visibility = []
        for theta in strengths:
            visible = theta > threshold
            if visible:
                overlap = bbp.spike_eigenvector_overlap(theta)
            else:
                overlap = 0.0
            visibility.append({
                "strength": float(theta),
                "visible": visible,
                "overlap": float(overlap),
                "threshold": float(threshold),
            })

        return {
            "gamma_effective": float(gamma_eff),
            "bbp_threshold": float(threshold),
            "features": visibility,
        }

    def lazy_regime_blindness(self, strengths: np.ndarray, width: int) -> dict:
        """Analyze feature blindness in the lazy (NTK) regime.

        In the lazy regime, the NTK is fixed at initialization.
        Features weaker than the BBP threshold for γ = N/width are
        invisible, limiting what the network can learn.

        Parameters
        ----------
        strengths : np.ndarray
            Feature strengths.
        width : int
            Network width.

        Returns
        -------
        dict
            Analysis of which features are blind to the lazy regime.
        """
        vis = self.feature_visibility(strengths, width)
        n_visible = sum(1 for f in vis["features"] if f["visible"])
        n_blind = len(strengths) - n_visible

        # Width needed to see each feature
        min_widths = []
        for theta in strengths:
            # Need γ_eff = N/m such that θ > σ²√γ_eff
            # => m > N σ⁴ / θ²
            min_width = int(np.ceil(self.N / (theta ** 2))) if theta > 0 else float("inf")
            min_widths.append(min_width)

        return {
            "n_visible": n_visible,
            "n_blind": n_blind,
            "current_width": width,
            "features": vis["features"],
            "min_widths_needed": min_widths,
        }

    def rich_regime_alignment(
        self,
        strengths: np.ndarray,
        width: int,
        lr: float = 0.01,
        steps: int = 100,
    ) -> dict:
        """Simulate feature learning in the rich (mean-field) regime.

        In the rich regime, the NTK evolves during training and can
        align with task-relevant features, overcoming the BBP limitation.

        This is a simplified model: tracks how the effective spike
        strength grows during training due to feature learning.

        Parameters
        ----------
        strengths : np.ndarray
            Initial feature strengths.
        width : int
            Network width.
        lr : float
            Learning rate.
        steps : int
            Number of training steps.

        Returns
        -------
        dict
            Feature alignment trajectory.
        """
        strengths = np.asarray(strengths, dtype=float)
        gamma_eff = self.N / width

        # Model: effective strength grows as θ_eff(t) = θ + α·t·θ/(1 + β·t)
        # where α depends on lr and β on regularization
        alpha = lr * width / self.N  # feature learning rate
        beta = 0.01  # regularization

        trajectory = np.zeros((steps, len(strengths)))
        bbp_threshold = np.sqrt(gamma_eff)

        for t in range(steps):
            theta_eff = strengths + alpha * t * strengths / (1.0 + beta * t)
            trajectory[t] = theta_eff

        # When does each feature cross the threshold?
        crossing_times = []
        for j in range(len(strengths)):
            crossed = np.where(trajectory[:, j] > bbp_threshold)[0]
            if len(crossed) > 0:
                crossing_times.append(int(crossed[0]))
            else:
                crossing_times.append(None)

        return {
            "initial_strengths": strengths.tolist(),
            "final_strengths": trajectory[-1].tolist(),
            "bbp_threshold": float(bbp_threshold),
            "crossing_times": crossing_times,
            "trajectory": trajectory,
        }


class PlantedFeatureDetector:
    """Detect planted features in NTK eigenvalue spectra.

    Parameters
    ----------
    gamma : float
        Aspect ratio.
    sigma_sq : float
        Noise variance.
    """

    def __init__(self, gamma: float, sigma_sq: float = 1.0):
        self.gamma = gamma
        self.sigma_sq = sigma_sq
        self.bbp = BBPTransition(gamma, sigma_sq)

    def detect_planted_signal(self, eigenvalues: np.ndarray) -> dict:
        """Detect outlier eigenvalues (spikes) in the NTK spectrum.

        Eigenvalues above the MP upper edge are candidates for
        planted signals.

        Parameters
        ----------
        eigenvalues : np.ndarray
            Observed eigenvalues (e.g., of the NTK).

        Returns
        -------
        dict
            Detected signals and their properties.
        """
        eigenvalues = np.sort(np.asarray(eigenvalues, dtype=float))[::-1]
        edge = self.sigma_sq * (1.0 + np.sqrt(self.gamma)) ** 2

        # Detect outliers above the edge (with a small margin for fluctuations)
        margin = edge * 0.05
        outliers = eigenvalues[eigenvalues > edge + margin]

        signals = []
        for eig in outliers:
            theta_est = self.estimate_signal_strength(eig)
            overlap = self.bbp.spike_eigenvector_overlap(theta_est) if theta_est > 0 else 0.0
            signals.append({
                "eigenvalue": float(eig),
                "estimated_strength": float(theta_est),
                "estimated_overlap": float(overlap),
            })

        return {
            "n_signals_detected": len(signals),
            "bulk_edge": float(edge),
            "signals": signals,
        }

    def estimate_signal_strength(self, outlier_eigenvalue: float) -> float:
        """Estimate the population spike strength from an observed outlier.

        Inverts the BBP formula: given ℓ = (σ² + θ)(1 + γσ²/θ),
        solve for θ.

        Parameters
        ----------
        outlier_eigenvalue : float
            Observed outlier eigenvalue.

        Returns
        -------
        float
            Estimated spike strength θ.
        """
        ell = outlier_eigenvalue
        s = self.sigma_sq
        g = self.gamma

        # ℓ = (s + θ)(1 + gs/θ) = s + θ + gs + gs²/θ
        # θ² + (s + gs - ℓ)θ + gs² = 0
        a = 1.0
        b = s * (1.0 + g) - ell
        c = g * s ** 2

        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return 0.0

        # Take the larger root (physical solution)
        theta = (-b + np.sqrt(discriminant)) / (2 * a)
        return max(float(theta), 0.0)

    def feature_learning_evidence(
        self, ntk_eigenvalues_over_time: list
    ) -> dict:
        """Track emergence of spikes during training as evidence of feature learning.

        As the network trains in the rich regime, new spikes should
        emerge in the NTK spectrum, indicating feature learning.

        Parameters
        ----------
        ntk_eigenvalues_over_time : list of np.ndarray
            NTK eigenvalues at successive training steps.

        Returns
        -------
        dict
            Evidence of feature learning from spectral evolution.
        """
        edge = self.sigma_sq * (1.0 + np.sqrt(self.gamma)) ** 2
        margin = edge * 0.05

        n_outliers_over_time = []
        max_eig_over_time = []
        estimated_strengths_over_time = []

        for eigs in ntk_eigenvalues_over_time:
            eigs = np.sort(np.asarray(eigs, dtype=float))[::-1]
            outliers = eigs[eigs > edge + margin]
            n_outliers_over_time.append(len(outliers))
            max_eig_over_time.append(float(eigs[0]) if len(eigs) > 0 else 0.0)

            if len(outliers) > 0:
                strengths = [self.estimate_signal_strength(e) for e in outliers]
                estimated_strengths_over_time.append(strengths)
            else:
                estimated_strengths_over_time.append([])

        # Detect if number of outliers increased over time
        if len(n_outliers_over_time) >= 2:
            feature_learning_detected = n_outliers_over_time[-1] > n_outliers_over_time[0]
            spike_growth = max_eig_over_time[-1] - max_eig_over_time[0]
        else:
            feature_learning_detected = False
            spike_growth = 0.0

        return {
            "feature_learning_detected": feature_learning_detected,
            "n_outliers_over_time": n_outliers_over_time,
            "max_eigenvalue_over_time": max_eig_over_time,
            "estimated_strengths_over_time": estimated_strengths_over_time,
            "spike_growth": float(spike_growth),
            "bulk_edge": float(edge),
        }
