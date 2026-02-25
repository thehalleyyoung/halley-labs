"""
Tracy-Widom distribution and spectral edge analysis.

Implements the Tracy-Widom distribution via Painlevé II,
Airy kernel Fredholm determinants, and tools for analyzing
eigenvalue fluctuations at the spectral edge of random matrices.
"""

import numpy as np
from scipy import integrate, interpolate, special


class AiryKernelComputer:
    """Compute the Airy kernel and its Fredholm determinant.

    The Airy kernel K_Ai(x, y) governs the local eigenvalue
    statistics at the soft edge of random matrices.

    Parameters
    ----------
    grid_size : int
        Number of grid points for discretization.
    x_max : float
        Upper limit of the computational domain.
    """

    def __init__(self, grid_size: int = 200, x_max: float = 10.0):
        self.grid_size = grid_size
        self.x_max = x_max

    def airy_function(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the Airy function Ai(x).

        Parameters
        ----------
        x : np.ndarray
            Evaluation points.

        Returns
        -------
        np.ndarray
            Ai(x) values.
        """
        ai, _, _, _ = special.airy(x)
        return ai

    def airy_derivative(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the Airy function derivative Ai'(x).

        Parameters
        ----------
        x : np.ndarray
            Evaluation points.

        Returns
        -------
        np.ndarray
            Ai'(x) values.
        """
        _, ai_prime, _, _ = special.airy(x)
        return ai_prime

    def airy_kernel(self, x: float, y: float) -> float:
        """Evaluate the Airy kernel K_Ai(x, y).

        K_Ai(x, y) = [Ai(x)Ai'(y) - Ai'(x)Ai(y)] / (x - y)

        At x = y, uses the limit: K_Ai(x, x) = Ai'(x)² - x·Ai(x)².

        Parameters
        ----------
        x, y : float
            Kernel arguments.

        Returns
        -------
        float
            Kernel value.
        """
        if abs(x - y) < 1e-12:
            ai = self.airy_function(np.array([x]))[0]
            ai_p = self.airy_derivative(np.array([x]))[0]
            return ai_p ** 2 - x * ai ** 2

        ai_x = self.airy_function(np.array([x]))[0]
        ai_y = self.airy_function(np.array([y]))[0]
        aip_x = self.airy_derivative(np.array([x]))[0]
        aip_y = self.airy_derivative(np.array([y]))[0]

        return (ai_x * aip_y - aip_x * ai_y) / (x - y)

    def kernel_matrix(self, grid: np.ndarray) -> np.ndarray:
        """Build the discretized Airy kernel matrix on a grid.

        Parameters
        ----------
        grid : np.ndarray
            Grid points.

        Returns
        -------
        np.ndarray
            Kernel matrix K[i, j] = K_Ai(grid[i], grid[j]) * Δx.
        """
        n = len(grid)
        dx = grid[1] - grid[0] if n > 1 else 1.0
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self.airy_kernel(grid[i], grid[j]) * dx
        return K

    def fredholm_determinant(self, s: float, beta: int = 2) -> float:
        """Compute the Fredholm determinant det(I - K_Ai) on [s, ∞).

        For β = 2, this gives the Tracy-Widom CDF: F₂(s) = det(I - K_Ai|_{[s,∞)}).

        Parameters
        ----------
        s : float
            Lower limit of the integration interval.
        beta : int
            Dyson index (1, 2, or 4).

        Returns
        -------
        float
            Fredholm determinant value.
        """
        grid = np.linspace(s, s + self.x_max, self.grid_size)
        K = self.kernel_matrix(grid)

        if beta == 1:
            K = K / 2.0
        elif beta == 4:
            K = K * 2.0

        eigenvalues = np.linalg.eigvalsh(K)
        # det(I - K) = prod(1 - λ_i)
        log_det = np.sum(np.log(np.maximum(1.0 - eigenvalues, 1e-300)))
        return np.exp(log_det)


class TracyWidomDistribution:
    """The Tracy-Widom distribution for the largest eigenvalue.

    The TW distribution describes the fluctuations of the largest
    eigenvalue of large random matrices. For GUE (β=2), it is
    expressed through the Painlevé II equation.

    Parameters
    ----------
    beta : int
        Dyson index: 1 (GOE), 2 (GUE), or 4 (GSE).
    """

    def __init__(self, beta: int = 2):
        if beta not in (1, 2, 4):
            raise ValueError("beta must be 1, 2, or 4")
        self.beta = beta
        self._s_grid = None
        self._cdf_grid = None
        self._pdf_grid = None
        self._build_tables()

    def _build_tables(self, s_min: float = -8.0, s_max: float = 6.0, n_points: int = 1000):
        """Precompute CDF and PDF tables via Painlevé II solution."""
        self._s_grid = np.linspace(s_min, s_max, n_points)

        if self.beta == 2:
            q_values = self.painleve_ii_solution(self._s_grid)
            # F₂(s) = exp(-∫_s^∞ (x - s) q²(x) dx)
            cdf = np.zeros(n_points)
            for i, s in enumerate(self._s_grid):
                # Integrate from s to s_max
                mask = self._s_grid >= s
                x_sub = self._s_grid[mask]
                q_sub = q_values[mask]
                integrand = (x_sub - s) * q_sub ** 2
                cdf[i] = np.exp(-np.trapz(integrand, x_sub))
            self._cdf_grid = cdf
        elif self.beta == 1:
            # TW₁ approximation via relationship to TW₂
            # F₁(s) ≈ F₂(s)^{1/2} · exp(-(1/2)∫_s^∞ q(x) dx)
            q_values = self.painleve_ii_solution(self._s_grid)
            cdf_2 = np.zeros(n_points)
            q_integral = np.zeros(n_points)
            for i, s in enumerate(self._s_grid):
                mask = self._s_grid >= s
                x_sub = self._s_grid[mask]
                q_sub = q_values[mask]
                integrand = (x_sub - s) * q_sub ** 2
                cdf_2[i] = np.exp(-np.trapz(integrand, x_sub))
                q_integral[i] = np.trapz(q_sub, x_sub)
            self._cdf_grid = np.sqrt(cdf_2) * np.exp(-0.5 * q_integral)
        elif self.beta == 4:
            # TW₄ via relationship: F₄(s) = (F₂(s) · F₂(s/√2 · 2^{1/3}))^{1/2} approx
            q_values = self.painleve_ii_solution(self._s_grid)
            cdf_2 = np.zeros(n_points)
            for i, s in enumerate(self._s_grid):
                mask = self._s_grid >= s
                x_sub = self._s_grid[mask]
                q_sub = q_values[mask]
                integrand = (x_sub - s) * q_sub ** 2
                cdf_2[i] = np.exp(-np.trapz(integrand, x_sub))
            # F₄(s) ≈ cosh(∫_s^∞ q(x) dx) · F₂(s)^{1/2}
            q_int = np.zeros(n_points)
            for i, s in enumerate(self._s_grid):
                mask = self._s_grid >= s
                q_int[i] = np.trapz(q_values[mask], self._s_grid[mask])
            self._cdf_grid = np.cosh(q_int / 2) * np.sqrt(cdf_2)

        # Normalize CDF
        self._cdf_grid = np.clip(self._cdf_grid, 0.0, 1.0)
        if self._cdf_grid[-1] > 0:
            self._cdf_grid /= self._cdf_grid[-1]

        # PDF via numerical differentiation
        self._pdf_grid = np.gradient(self._cdf_grid, self._s_grid)
        self._pdf_grid = np.maximum(self._pdf_grid, 0.0)

    def painleve_ii_solution(self, s_range: np.ndarray) -> np.ndarray:
        """Solve the Painlevé II equation q'' = sq + 2q³.

        With boundary condition q(s) ~ Ai(s) as s → +∞.

        Parameters
        ----------
        s_range : np.ndarray
            Grid points (should be sorted ascending).

        Returns
        -------
        np.ndarray
            Solution q(s) at each grid point.
        """
        s_sorted = np.sort(s_range)
        s_max = s_sorted[-1]

        # Initial condition from Airy function at large s
        ai_val, ai_prime, _, _ = special.airy(s_max)

        # Integrate backward from s_max to s_min
        # System: q' = p, p' = s*q + 2*q³
        def rhs(s, y):
            q, p = y
            return [p, s * q + 2.0 * q ** 3]

        s_backward = s_sorted[::-1]
        sol = integrate.solve_ivp(
            rhs,
            [s_max, s_sorted[0]],
            [ai_val, ai_prime],
            t_eval=s_backward,
            method="RK45",
            rtol=1e-10,
            atol=1e-12,
            max_step=0.05,
        )

        q_backward = sol.y[0]
        q_forward = q_backward[::-1]

        # Interpolate to original grid
        if len(q_forward) == len(s_sorted):
            q_interp = np.interp(s_range, s_sorted, q_forward)
        else:
            s_sol = sol.t[::-1]
            q_interp = np.interp(s_range, s_sol, q_forward[: len(s_sol)])

        return q_interp

    def cdf(self, s: np.ndarray) -> np.ndarray:
        """Evaluate the Tracy-Widom CDF F_β(s).

        Parameters
        ----------
        s : np.ndarray or float
            Evaluation points.

        Returns
        -------
        np.ndarray
            CDF values.
        """
        s = np.atleast_1d(np.asarray(s, dtype=float))
        interp_fn = interpolate.interp1d(
            self._s_grid, self._cdf_grid,
            kind="cubic", bounds_error=False, fill_value=(0.0, 1.0)
        )
        return interp_fn(s)

    def pdf(self, s: np.ndarray) -> np.ndarray:
        """Evaluate the Tracy-Widom PDF f_β(s).

        Parameters
        ----------
        s : np.ndarray or float
            Evaluation points.

        Returns
        -------
        np.ndarray
            PDF values.
        """
        s = np.atleast_1d(np.asarray(s, dtype=float))
        interp_fn = interpolate.interp1d(
            self._s_grid, self._pdf_grid,
            kind="cubic", bounds_error=False, fill_value=(0.0, 0.0)
        )
        return np.maximum(interp_fn(s), 0.0)

    def quantile(self, p: np.ndarray) -> np.ndarray:
        """Inverse CDF (quantile function) of the Tracy-Widom distribution.

        Parameters
        ----------
        p : np.ndarray or float
            Probability values in [0, 1].

        Returns
        -------
        np.ndarray
            Quantile values.
        """
        p = np.atleast_1d(np.asarray(p, dtype=float))
        interp_fn = interpolate.interp1d(
            self._cdf_grid, self._s_grid,
            kind="linear", bounds_error=False,
            fill_value=(self._s_grid[0], self._s_grid[-1])
        )
        return interp_fn(p)

    def mean(self) -> float:
        """Expected value E[TW_β].

        Returns
        -------
        float
            Mean of the Tracy-Widom distribution.
        """
        return float(np.trapz(self._s_grid * self._pdf_grid, self._s_grid))

    def variance(self) -> float:
        """Variance Var[TW_β].

        Returns
        -------
        float
            Variance of the Tracy-Widom distribution.
        """
        mu = self.mean()
        return float(np.trapz((self._s_grid - mu) ** 2 * self._pdf_grid, self._s_grid))

    def skewness(self) -> float:
        """Skewness of the Tracy-Widom distribution.

        Returns
        -------
        float
            Skewness.
        """
        mu = self.mean()
        sigma = np.sqrt(self.variance())
        if sigma < 1e-15:
            return 0.0
        m3 = float(np.trapz((self._s_grid - mu) ** 3 * self._pdf_grid, self._s_grid))
        return m3 / sigma ** 3

    def sample(self, n_samples: int) -> np.ndarray:
        """Sample from the Tracy-Widom distribution via inverse CDF.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        np.ndarray
            Samples from TW_β.
        """
        u = np.random.uniform(0, 1, n_samples)
        return self.quantile(u)


class EdgeFluctuation:
    """Analyze eigenvalue fluctuations at the spectral edge.

    For an N×P Wishart matrix with γ = N/P, the largest eigenvalue
    satisfies: N^{2/3}(λ_max - λ⁺)/μ → TW_β.

    Parameters
    ----------
    N : int
        Matrix dimension.
    gamma : float
        Aspect ratio N/P.
    """

    def __init__(self, N: int, gamma: float = 1.0):
        self.N = N
        self.gamma = gamma
        self.tw = TracyWidomDistribution(beta=2)

    def edge_location(self) -> float:
        """Upper edge of the Marchenko-Pastur support.

        λ⁺ = (1 + √γ)² (with σ² = 1).

        Returns
        -------
        float
            Edge location.
        """
        return (1.0 + np.sqrt(self.gamma)) ** 2

    def edge_scaling(self) -> float:
        """Compute the N^{2/3} edge scaling factor.

        The centering/scaling is:
        μ_N = (1 + √γ)(1/√N + 1/√P)^{1/3} ≈ (1+√γ)^{4/3} γ^{-1/6} N^{-2/3}.

        Returns
        -------
        float
            Scaling factor for the edge fluctuations.
        """
        sq_gamma = np.sqrt(self.gamma)
        return (1.0 + sq_gamma) ** (4.0 / 3.0) * self.gamma ** (-1.0 / 6.0) * self.N ** (-2.0 / 3.0)

    def largest_eigenvalue_distribution(self, eigenvalues: np.ndarray) -> np.ndarray:
        """Rescale eigenvalues to Tracy-Widom scale.

        Computes (λ_i - λ⁺) / μ_N for comparison with TW.

        Parameters
        ----------
        eigenvalues : np.ndarray
            Observed largest eigenvalues (from multiple trials).

        Returns
        -------
        np.ndarray
            Rescaled eigenvalues on the TW scale.
        """
        edge = self.edge_location()
        scale = self.edge_scaling()
        return (np.asarray(eigenvalues) - edge) / scale

    def test_tw_fit(self, eigenvalues: np.ndarray, n_trials: int = 1) -> dict:
        """Test whether the largest eigenvalue follows the Tracy-Widom distribution.

        Parameters
        ----------
        eigenvalues : np.ndarray
            Largest eigenvalues from n_trials independent realizations.
        n_trials : int
            Number of trials (used for context, eigenvalues should have this length).

        Returns
        -------
        dict
            Test results including rescaled statistics and TW comparison.
        """
        rescaled = self.largest_eigenvalue_distribution(eigenvalues)
        tw_cdf_vals = self.tw.cdf(rescaled)

        n = len(rescaled)
        empirical_cdf = np.arange(1, n + 1) / n
        sorted_rescaled = np.sort(rescaled)
        tw_cdf_sorted = self.tw.cdf(sorted_rescaled)

        ks_stat = np.max(np.abs(empirical_cdf - tw_cdf_sorted))

        return {
            "rescaled_eigenvalues": rescaled,
            "mean_rescaled": float(np.mean(rescaled)),
            "std_rescaled": float(np.std(rescaled)),
            "tw_mean": self.tw.mean(),
            "tw_std": np.sqrt(self.tw.variance()),
            "ks_statistic": float(ks_stat),
        }

    def outlier_detection(
        self, eigenvalues: np.ndarray, significance: float = 0.05
    ) -> dict:
        """Detect outlier eigenvalues beyond the Tracy-Widom threshold.

        An eigenvalue is an outlier if it exceeds the (1 - α) quantile
        of the TW distribution after rescaling.

        Parameters
        ----------
        eigenvalues : np.ndarray
            Observed eigenvalues (sorted ascending).
        significance : float
            Significance level for outlier detection.

        Returns
        -------
        dict
            {'threshold': float, 'outliers': np.ndarray, 'n_outliers': int}.
        """
        eigenvalues = np.sort(np.asarray(eigenvalues, dtype=float))
        tw_quantile = self.tw.quantile(np.array([1.0 - significance]))[0]
        threshold = self.edge_location() + tw_quantile * self.edge_scaling()

        outliers = eigenvalues[eigenvalues > threshold]

        return {
            "threshold": float(threshold),
            "outliers": outliers,
            "n_outliers": len(outliers),
            "significance": significance,
        }


class SpectralEdgeAnalyzer:
    """Tools for analyzing spectral edge behavior and phase transitions.

    Parameters
    ----------
    beta : int
        Dyson index.
    """

    def __init__(self, beta: int = 2):
        self.beta = beta
        self.tw = TracyWidomDistribution(beta=beta)

    def phase_transition_at_edge(
        self, eigenvalues: np.ndarray, gamma: float
    ) -> dict:
        """Detect a phase transition at the spectral edge.

        Checks whether the eigenvalue distribution transitions
        from MP bulk behavior to outlier behavior, indicating
        a BBP-type phase transition.

        Parameters
        ----------
        eigenvalues : np.ndarray
            Observed eigenvalues.
        gamma : float
            Aspect ratio.

        Returns
        -------
        dict
            Phase transition analysis results.
        """
        eigenvalues = np.sort(np.asarray(eigenvalues, dtype=float))
        edge_loc = (1.0 + np.sqrt(gamma)) ** 2
        N = len(eigenvalues)

        # Count eigenvalues above the MP edge
        n_above = np.sum(eigenvalues > edge_loc * 1.01)

        # Edge statistics
        ef = EdgeFluctuation(N, gamma)
        rescaled_top = ef.largest_eigenvalue_distribution(eigenvalues[-5:])

        # If top eigenvalue is far from TW prediction, indicates phase transition
        tw_99 = self.tw.quantile(np.array([0.99]))[0]
        top_rescaled = rescaled_top[-1]

        return {
            "edge_location": float(edge_loc),
            "n_outliers_above_edge": int(n_above),
            "top_eigenvalue": float(eigenvalues[-1]),
            "top_rescaled": float(top_rescaled),
            "tw_99_quantile": float(tw_99),
            "phase_transition_detected": top_rescaled > tw_99 * 2,
        }

    def edge_exponents(
        self, eigenvalues_list: list, N_list: list
    ) -> dict:
        """Measure scaling exponents at the spectral edge.

        Fits the exponent α in |λ_max - λ⁺| ~ N^{-α} across
        multiple matrix sizes.

        Parameters
        ----------
        eigenvalues_list : list of np.ndarray
            Largest eigenvalues for each matrix size.
        N_list : list of int
            Matrix dimensions.

        Returns
        -------
        dict
            {'exponent': float, 'expected_exponent': 2/3, 'fit_error': float}.
        """
        deviations = []
        for eigs, N in zip(eigenvalues_list, N_list):
            gamma = 1.0  # assume square matrices for simplicity
            edge = (1.0 + np.sqrt(gamma)) ** 2
            mean_dev = np.mean(np.abs(np.asarray(eigs) - edge))
            deviations.append(mean_dev)

        log_N = np.log(np.array(N_list, dtype=float))
        log_dev = np.log(np.array(deviations) + 1e-30)

        # Linear fit: log(dev) = -α log(N) + c
        coeffs = np.polyfit(log_N, log_dev, 1)
        alpha = -coeffs[0]

        return {
            "exponent": float(alpha),
            "expected_exponent": 2.0 / 3.0,
            "fit_error": float(abs(alpha - 2.0 / 3.0)),
            "deviations": deviations,
        }

    def soft_edge_vs_hard_edge(self, eigenvalues: np.ndarray) -> dict:
        """Classify whether the spectral edge is soft or hard.

        A soft edge has density vanishing as √(λ⁺ - λ) (MP-like),
        while a hard edge has density diverging as 1/√λ (Jacobi-like).

        Parameters
        ----------
        eigenvalues : np.ndarray
            Observed eigenvalues.

        Returns
        -------
        dict
            Edge classification and supporting statistics.
        """
        eigenvalues = np.sort(np.asarray(eigenvalues, dtype=float))
        n = len(eigenvalues)

        # Examine density near the upper edge
        top_fraction = eigenvalues[int(0.95 * n):]
        spacings = np.diff(top_fraction)

        # Soft edge: spacings grow as we approach the edge
        # Hard edge: spacings shrink
        if len(spacings) < 3:
            return {"edge_type": "undetermined", "evidence": "too few eigenvalues"}

        first_half = np.mean(spacings[: len(spacings) // 2])
        second_half = np.mean(spacings[len(spacings) // 2:])

        if second_half > first_half * 1.2:
            edge_type = "soft"
        elif second_half < first_half * 0.8:
            edge_type = "hard"
        else:
            edge_type = "intermediate"

        # Examine lower edge
        bottom_fraction = eigenvalues[: int(0.05 * n) + 1]
        if len(bottom_fraction) > 1 and bottom_fraction[0] > 1e-10:
            lower_edge = "hard"
        else:
            lower_edge = "at_zero"

        return {
            "upper_edge_type": edge_type,
            "lower_edge_type": lower_edge,
            "spacing_ratio": float(second_half / (first_half + 1e-30)),
        }

    def eigenvalue_spacing_at_edge(
        self, eigenvalues: np.ndarray, n_edge: int = 10
    ) -> dict:
        """Compute nearest-neighbor spacing statistics at the spectral edge.

        Parameters
        ----------
        eigenvalues : np.ndarray
            Observed eigenvalues (sorted).
        n_edge : int
            Number of edge eigenvalues to analyze.

        Returns
        -------
        dict
            Spacing statistics at the upper edge.
        """
        eigenvalues = np.sort(np.asarray(eigenvalues, dtype=float))
        n = len(eigenvalues)
        n_edge = min(n_edge, n - 1)

        top_eigs = eigenvalues[-(n_edge + 1):]
        spacings = np.diff(top_eigs)

        # Normalize by mean spacing
        mean_spacing = np.mean(spacings)
        if mean_spacing > 0:
            normalized_spacings = spacings / mean_spacing
        else:
            normalized_spacings = spacings

        return {
            "raw_spacings": spacings,
            "normalized_spacings": normalized_spacings,
            "mean_spacing": float(mean_spacing),
            "std_spacing": float(np.std(spacings)),
            "min_spacing": float(np.min(spacings)),
            "max_spacing": float(np.max(spacings)),
            "n_edge": n_edge,
        }
