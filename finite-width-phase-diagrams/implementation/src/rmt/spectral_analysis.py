"""
Advanced spectral analysis tools for NTK matrices using random matrix theory.

Provides empirical spectral distribution analysis, Wigner semicircle comparison,
spectral rigidity statistics, spectral flow tracking, NTK spectrum classification,
random matrix ensemble generation, and spectral comparison metrics.
"""

import numpy as np
from scipy import stats, optimize, integrate, interpolate
from scipy.special import gammaln
from typing import Optional, Callable, List, Tuple, Dict, Any


# ---------------------------------------------------------------------------
# 1. EmpiricalSpectralDistribution
# ---------------------------------------------------------------------------

class EmpiricalSpectralDistribution:
    r"""Empirical spectral distribution tools for a given set of eigenvalues.

    Provides KDE-based density estimation, CDF, Stieltjes transform,
    moments, log-potential, spectral entropy, effective rank,
    participation ratio, level-spacing statistics, spectral unfolding,
    and number variance.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Array of eigenvalues (real).
    """

    def __init__(self, eigenvalues: np.ndarray):
        self.eigenvalues = np.sort(np.asarray(eigenvalues, dtype=np.float64))
        self.N = len(self.eigenvalues)

    # ----- KDE density -----------------------------------------------------

    def density(
        self,
        x_range: np.ndarray,
        bandwidth: str = 'silverman',
    ) -> np.ndarray:
        r"""Kernel density estimate (KDE) of the spectral density.

        Parameters
        ----------
        x_range : np.ndarray
            Grid of points at which to evaluate the density.
        bandwidth : str
            Bandwidth selection method ('silverman' or 'scott').

        Returns
        -------
        np.ndarray
            Estimated spectral density at each point in x_range.
        """
        x_range = np.asarray(x_range, dtype=np.float64)
        if self.N == 0:
            return np.zeros_like(x_range)

        kde = stats.gaussian_kde(self.eigenvalues, bw_method=bandwidth)
        return kde(x_range)

    # ----- Empirical CDF ---------------------------------------------------

    def empirical_cdf(self, x: np.ndarray) -> np.ndarray:
        r"""Empirical CDF F_N(x) = #{λ_i ≤ x} / N.

        Parameters
        ----------
        x : np.ndarray
            Points at which to evaluate the CDF.

        Returns
        -------
        np.ndarray
            CDF values.
        """
        x = np.asarray(x, dtype=np.float64)
        return np.array([np.sum(self.eigenvalues <= xi) / self.N for xi in x])

    # ----- Stieltjes transform ---------------------------------------------

    def stieltjes_transform(self, z: complex) -> complex:
        r"""Empirical Stieltjes transform m_N(z) = (1/N) Σ 1/(λ_i - z).

        Parameters
        ----------
        z : complex
            Point in the complex plane (typically Im(z) > 0).

        Returns
        -------
        complex
            Stieltjes transform value.
        """
        z = complex(z)
        return complex(np.mean(1.0 / (self.eigenvalues - z)))

    # ----- Moments ---------------------------------------------------------

    def moments(self, k: int) -> float:
        r"""k-th spectral moment m_k = (1/N) Σ λ_i^k.

        Parameters
        ----------
        k : int
            Moment order.

        Returns
        -------
        float
            k-th moment.
        """
        return float(np.mean(self.eigenvalues ** k))

    # ----- Log-potential ---------------------------------------------------

    def log_potential(self, x: np.ndarray) -> np.ndarray:
        r"""Log-potential U(x) = -(1/N) Σ ln|x - λ_i|.

        Parameters
        ----------
        x : np.ndarray
            Points at which to evaluate.

        Returns
        -------
        np.ndarray
            Log-potential values.
        """
        x = np.asarray(x, dtype=np.float64)
        result = np.empty_like(x)
        for i, xi in enumerate(x):
            diffs = np.abs(xi - self.eigenvalues)
            diffs = np.maximum(diffs, 1e-300)
            result[i] = -np.mean(np.log(diffs))
        return result

    # ----- Spectral entropy ------------------------------------------------

    def entropy(self) -> float:
        r"""Spectral entropy -Σ p_i ln p_i where p_i = λ_i / Σ λ_j.

        Only defined for non-negative eigenvalues.

        Returns
        -------
        float
            Spectral entropy.
        """
        eigs = self.eigenvalues[self.eigenvalues > 0]
        if len(eigs) == 0:
            return 0.0
        p = eigs / np.sum(eigs)
        return float(-np.sum(p * np.log(p + 1e-300)))

    # ----- Effective rank --------------------------------------------------

    def effective_rank(self) -> float:
        r"""Effective rank = exp(entropy).

        Measures the effective dimensionality of the spectrum.

        Returns
        -------
        float
            Effective rank.
        """
        return float(np.exp(self.entropy()))

    # ----- Participation ratio ---------------------------------------------

    def participation_ratio(
        self,
        eigenvectors: np.ndarray,
    ) -> np.ndarray:
        r"""Inverse participation ratio IPR = Σ |v_i|^4 for each eigenvector.

        Small IPR → delocalized; IPR ≈ 1 → localized.

        Parameters
        ----------
        eigenvectors : np.ndarray, shape (N, N)
            Matrix of eigenvectors (columns).

        Returns
        -------
        np.ndarray
            IPR for each eigenvector.
        """
        V = np.asarray(eigenvectors, dtype=np.float64)
        ipr = np.sum(V ** 4, axis=0)
        return ipr

    # ----- Level spacing distribution --------------------------------------

    def level_spacing_distribution(self) -> Dict[str, np.ndarray]:
        r"""Nearest-neighbor level spacing distribution P(s).

        After unfolding to unit mean spacing, computes the histogram of
        spacings and compares with Wigner surmise P_GOE(s) = (π/2) s exp(-πs²/4).

        Returns
        -------
        dict
            ``spacings``, ``s_bins``, ``P_s`` (histogram),
            ``wigner_surmise`` (GOE prediction).
        """
        unfolded = self.unfolded_spectrum()
        spacings = np.diff(unfolded)

        if len(spacings) == 0:
            return {
                'spacings': np.array([]),
                's_bins': np.array([]),
                'P_s': np.array([]),
                'wigner_surmise': np.array([]),
            }

        # Normalize to unit mean
        mean_spacing = np.mean(spacings)
        if mean_spacing > 1e-14:
            s = spacings / mean_spacing
        else:
            s = spacings

        hist, bin_edges = np.histogram(s, bins=50, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Wigner surmise (GOE)
        wigner = (np.pi / 2) * bin_centers * np.exp(-np.pi * bin_centers ** 2 / 4)

        return {
            'spacings': s,
            's_bins': bin_centers,
            'P_s': hist,
            'wigner_surmise': wigner,
        }

    # ----- Unfolded spectrum -----------------------------------------------

    def unfolded_spectrum(self) -> np.ndarray:
        r"""Unfold the spectrum to have unit mean spacing.

        Uses the integrated density of states (staircase function) to map
        eigenvalues to a uniform sequence.

        Returns
        -------
        np.ndarray
            Unfolded eigenvalues with approximately unit mean spacing.
        """
        if self.N <= 1:
            return self.eigenvalues.copy()

        # Smooth the staircase with a polynomial fit
        ranks = np.arange(1, self.N + 1, dtype=float)
        # Fit polynomial of degree min(5, N-1) to (eigenvalue, rank) pairs
        deg = min(5, self.N - 1)
        coeffs = np.polyfit(self.eigenvalues, ranks, deg)
        unfolded = np.polyval(coeffs, self.eigenvalues)
        return unfolded

    # ----- Number variance -------------------------------------------------

    def number_variance(self, L_range: np.ndarray) -> np.ndarray:
        r"""Number variance Σ²(L) = Var[N(L)] for interval of length L.

        N(L) counts eigenvalues in an interval of length L (in unfolded
        units).  The variance is averaged over the center position.

        Parameters
        ----------
        L_range : np.ndarray
            Array of interval lengths (in unfolded units).

        Returns
        -------
        np.ndarray
            Number variance for each L.
        """
        unfolded = self.unfolded_spectrum()
        L_range = np.asarray(L_range, dtype=np.float64)
        sigma2 = np.empty(len(L_range))

        for i, L in enumerate(L_range):
            counts = []
            # Slide window across the unfolded spectrum
            centers = np.linspace(
                unfolded[0] + L / 2, unfolded[-1] - L / 2,
                min(200, self.N)
            )
            for c in centers:
                n_in = np.sum((unfolded >= c - L / 2) & (unfolded < c + L / 2))
                counts.append(n_in)
            counts = np.array(counts, dtype=float)
            sigma2[i] = np.var(counts) if len(counts) > 1 else 0.0

        return sigma2


# ---------------------------------------------------------------------------
# 2. WignerSemicircle
# ---------------------------------------------------------------------------

class WignerSemicircle:
    r"""Wigner semicircle distribution for comparison with empirical spectra.

    The semicircle law ρ(x) = (1/(2πR²)) √(4R² - x²) describes the
    limiting spectral density of GOE/GUE matrices with variance R².

    Parameters
    ----------
    radius : float
        Semicircle radius (default 2.0 for unit-variance entries).
    """

    def __init__(self, radius: float = 2.0):
        if radius <= 0:
            raise ValueError("Radius must be positive.")
        self.R = radius

    def density(self, x: np.ndarray) -> np.ndarray:
        r"""Semicircle density ρ(x) = (1/(2πR²)) √(4R² - x²).

        Parameters
        ----------
        x : np.ndarray
            Points at which to evaluate.

        Returns
        -------
        np.ndarray
            Density values (zero outside [-2R, 2R]).
        """
        x = np.asarray(x, dtype=np.float64)
        rho = np.zeros_like(x)
        mask = np.abs(x) < 2 * self.R
        rho[mask] = np.sqrt(4 * self.R ** 2 - x[mask] ** 2) / (2 * np.pi * self.R ** 2)
        return rho

    def stieltjes(self, z: complex) -> complex:
        r"""Stieltjes transform m(z) = (-z + √(z² - 4R²)) / (2R²).

        Parameters
        ----------
        z : complex
            Point in the complex plane.

        Returns
        -------
        complex
            Stieltjes transform value.
        """
        z = complex(z)
        disc = z ** 2 - 4 * self.R ** 2
        sqrt_disc = np.lib.scimath.sqrt(disc)
        # Choose branch with Im(m) > 0 when Im(z) > 0
        m = (-z + sqrt_disc) / (2 * self.R ** 2)
        if z.imag > 0 and m.imag < 0:
            m = (-z - sqrt_disc) / (2 * self.R ** 2)
        return complex(m)

    def r_transform(self, z: complex) -> complex:
        r"""R-transform R(z) = z for the semicircle.

        The R-transform encodes the free cumulants; for the semicircle
        all free cumulants beyond the first are zero, giving R(z) = z.

        Parameters
        ----------
        z : complex
            Argument.

        Returns
        -------
        complex
            R-transform value.
        """
        return z

    def compare_with_empirical(
        self,
        eigenvalues: np.ndarray,
    ) -> Dict[str, Any]:
        r"""Compare empirical eigenvalue distribution with semicircle.

        Computes KS distance, Wasserstein distance, and QQ plot data.

        Parameters
        ----------
        eigenvalues : np.ndarray
            Empirical eigenvalues.

        Returns
        -------
        dict
            ``ks_statistic``, ``ks_pvalue``, ``wasserstein``,
            ``qq_empirical``, ``qq_theoretical``.
        """
        eigs = np.sort(np.asarray(eigenvalues, dtype=np.float64))
        N = len(eigs)

        # Rescale to match semicircle: set variance to R²
        var = np.var(eigs) if N > 1 else 1.0
        if var > 1e-14:
            scaled = eigs * self.R / np.sqrt(var)
        else:
            scaled = eigs

        # Semicircle CDF (numerical)
        x_grid = np.linspace(-2 * self.R - 0.1, 2 * self.R + 0.1, 1000)
        density_vals = self.density(x_grid)
        cdf_vals = np.cumsum(density_vals) * (x_grid[1] - x_grid[0])
        cdf_vals = np.clip(cdf_vals, 0, 1)

        # KS test
        empirical_cdf = np.arange(1, N + 1) / N
        theo_cdf = np.interp(scaled, x_grid, cdf_vals)
        ks_stat = float(np.max(np.abs(empirical_cdf - theo_cdf)))

        # Wasserstein (L1 between sorted quantiles)
        theo_quantiles = np.interp(empirical_cdf, cdf_vals, x_grid)
        w1 = float(np.mean(np.abs(scaled - theo_quantiles)))

        return {
            'ks_statistic': ks_stat,
            'ks_pvalue': float(np.exp(-2 * N * ks_stat ** 2)),  # Approximate
            'wasserstein': w1,
            'qq_empirical': scaled,
            'qq_theoretical': theo_quantiles,
        }


# ---------------------------------------------------------------------------
# 3. SpectralRigidity
# ---------------------------------------------------------------------------

class SpectralRigidity:
    r"""Spectral rigidity statistics for eigenvalue correlations.

    Measures long-range eigenvalue correlations via number variance Σ²(L),
    spectral rigidity Δ₃(L), and comparison with GOE/GUE/Poisson
    universality classes.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Sorted eigenvalues.
    """

    def __init__(self, eigenvalues: np.ndarray):
        self.eigenvalues = np.sort(np.asarray(eigenvalues, dtype=np.float64))
        self.N = len(self.eigenvalues)
        self._esd = EmpiricalSpectralDistribution(self.eigenvalues)

    # ----- Number variance -------------------------------------------------

    def number_variance(self, L_range: np.ndarray) -> np.ndarray:
        r"""Number variance Σ²(L) = Var[#{eigenvalues in interval of length L}].

        Parameters
        ----------
        L_range : np.ndarray
            Interval lengths (in unfolded units).

        Returns
        -------
        np.ndarray
            Number variance.
        """
        return self._esd.number_variance(L_range)

    # ----- Spectral rigidity Δ₃ --------------------------------------------

    def spectral_rigidity_delta3(self, L_range: np.ndarray) -> np.ndarray:
        r"""Dyson–Mehta spectral rigidity Δ₃(L).

        Δ₃(L) = min_{a,b} (1/L) ∫₀^L [N(E) - aE - b]² dE

        where N(E) is the staircase function.

        Parameters
        ----------
        L_range : np.ndarray
            Interval lengths.

        Returns
        -------
        np.ndarray
            Spectral rigidity for each L.
        """
        unfolded = self._esd.unfolded_spectrum()
        L_range = np.asarray(L_range, dtype=np.float64)
        delta3 = np.empty(len(L_range))

        for idx, L in enumerate(L_range):
            if L <= 0:
                delta3[idx] = 0.0
                continue

            # Average over starting points
            vals = []
            starts = np.linspace(
                unfolded[0], unfolded[-1] - L,
                min(100, max(1, int(self.N / 2)))
            )
            for E0 in starts:
                E1 = E0 + L
                mask = (unfolded >= E0) & (unfolded < E1)
                eigs_in = unfolded[mask]
                if len(eigs_in) < 2:
                    continue

                # Staircase N(E) within [E0, E1]
                E_pts = np.linspace(E0, E1, 50)
                N_E = np.array([np.sum(unfolded <= e) for e in E_pts], dtype=float)

                # Best-fit line N(E) ≈ a E + b
                coeffs = np.polyfit(E_pts, N_E, 1)
                fitted = np.polyval(coeffs, E_pts)
                residual = np.mean((N_E - fitted) ** 2)
                vals.append(residual / L if L > 0 else 0)

            delta3[idx] = float(np.mean(vals)) if vals else 0.0

        return delta3

    # ----- GOE prediction --------------------------------------------------

    def goe_prediction(self, L: np.ndarray) -> np.ndarray:
        r"""GOE number variance prediction.

        Σ²_GOE(L) ≈ (2/π²)(ln(2πL) + γ + 1 - π²/8)

        for L >> 1, where γ ≈ 0.5772 is the Euler–Mascheroni constant.

        Parameters
        ----------
        L : np.ndarray
            Interval lengths.

        Returns
        -------
        np.ndarray
            GOE number variance.
        """
        L = np.asarray(L, dtype=np.float64)
        gamma_em = 0.5772156649
        result = np.zeros_like(L)
        mask = L > 0
        result[mask] = (2 / np.pi ** 2) * (
            np.log(2 * np.pi * L[mask]) + gamma_em + 1 - np.pi ** 2 / 8
        )
        return np.maximum(result, 0)

    # ----- GUE prediction --------------------------------------------------

    def gue_prediction(self, L: np.ndarray) -> np.ndarray:
        r"""GUE number variance prediction.

        Σ²_GUE(L) ≈ (1/π²)(ln(2πL) + γ + 1)

        Parameters
        ----------
        L : np.ndarray
            Interval lengths.

        Returns
        -------
        np.ndarray
            GUE number variance.
        """
        L = np.asarray(L, dtype=np.float64)
        gamma_em = 0.5772156649
        result = np.zeros_like(L)
        mask = L > 0
        result[mask] = (1 / np.pi ** 2) * (
            np.log(2 * np.pi * L[mask]) + gamma_em + 1
        )
        return np.maximum(result, 0)

    # ----- Poisson prediction ----------------------------------------------

    def poisson_prediction(self, L: np.ndarray) -> np.ndarray:
        r"""Poisson number variance Σ²_Poisson(L) = L.

        For uncorrelated eigenvalues (integrable or localized systems).

        Parameters
        ----------
        L : np.ndarray
            Interval lengths.

        Returns
        -------
        np.ndarray
            Poisson prediction (= L).
        """
        return np.asarray(L, dtype=np.float64).copy()

    # ----- Classify statistics ---------------------------------------------

    def classify_statistics(
        self,
        L_range: np.ndarray,
    ) -> Dict[str, Any]:
        r"""Classify eigenvalue statistics as GOE, GUE, or Poisson.

        Compares the empirical Σ²(L) with predictions from each
        universality class and selects the best fit.

        Parameters
        ----------
        L_range : np.ndarray
            Interval lengths for comparison.

        Returns
        -------
        dict
            ``classification`` (str), ``residuals`` (dict of MSE per class).
        """
        sigma2_emp = self.number_variance(L_range)
        sigma2_goe = self.goe_prediction(L_range)
        sigma2_gue = self.gue_prediction(L_range)
        sigma2_poi = self.poisson_prediction(L_range)

        mse_goe = float(np.mean((sigma2_emp - sigma2_goe) ** 2))
        mse_gue = float(np.mean((sigma2_emp - sigma2_gue) ** 2))
        mse_poi = float(np.mean((sigma2_emp - sigma2_poi) ** 2))

        residuals = {'GOE': mse_goe, 'GUE': mse_gue, 'Poisson': mse_poi}
        best = min(residuals, key=residuals.get)

        return {
            'classification': best,
            'residuals': residuals,
            'sigma2_empirical': sigma2_emp,
        }

    # ----- Brody parameter -------------------------------------------------

    def brody_parameter(self) -> Dict[str, float]:
        r"""Fit the Brody distribution P(s) = (β+1) a s^β exp(-a s^{β+1}).

        β = 0 → Poisson, β = 1 → GOE (Wigner surmise).

        Returns
        -------
        dict
            ``beta`` (Brody parameter), ``a`` (normalization).
        """
        spacings = np.diff(self._esd.unfolded_spectrum())
        if len(spacings) < 5:
            return {'beta': 0.0, 'a': 1.0}

        mean_s = np.mean(spacings)
        if mean_s > 1e-14:
            s = spacings / mean_s
        else:
            return {'beta': 0.0, 'a': 1.0}

        # MLE fit for Brody parameter
        def _neg_log_lik(beta):
            if beta < -0.5 or beta > 5:
                return 1e10
            bp1 = beta + 1
            from scipy.special import gamma as gamma_fn
            a = bp1 * (gamma_fn(1 + 1 / bp1)) ** bp1
            log_p = np.log(bp1 + 1e-14) + np.log(a + 1e-14) + beta * np.log(s + 1e-14) - a * s ** bp1
            return -np.sum(log_p)

        result = optimize.minimize_scalar(_neg_log_lik, bounds=(0, 3), method='bounded')
        beta = float(result.x)
        from scipy.special import gamma as gamma_fn
        a = (beta + 1) * (gamma_fn(1 + 1 / (beta + 1))) ** (beta + 1)

        return {'beta': beta, 'a': float(a)}


# ---------------------------------------------------------------------------
# 4. SpectralFlowAnalysis
# ---------------------------------------------------------------------------

class SpectralFlowAnalysis:
    r"""Track eigenvalue evolution as a function of a control parameter.

    Identifies level crossings, avoided crossings, spectral gap evolution,
    topological transitions, Berry phases, spectral zeta functions, and
    heat kernel traces.
    """

    def __init__(self):
        pass

    # ----- Track eigenvalues -----------------------------------------------

    def track_eigenvalues(
        self,
        kernel_fn: Callable[[float], np.ndarray],
        parameter_range: np.ndarray,
    ) -> np.ndarray:
        r"""Track eigenvalues of K(α) as α varies.

        Parameters
        ----------
        kernel_fn : callable
            Function α → K(α) returning a symmetric matrix.
        parameter_range : np.ndarray
            Values of the control parameter α.

        Returns
        -------
        np.ndarray, shape (len(parameter_range), N)
            Eigenvalue trajectories (sorted ascending at each α).
        """
        params = np.asarray(parameter_range, dtype=np.float64)
        K0 = kernel_fn(params[0])
        N = K0.shape[0]
        trajectories = np.empty((len(params), N))

        for i, alpha in enumerate(params):
            K = kernel_fn(alpha)
            eigs = np.linalg.eigvalsh(K)
            trajectories[i] = eigs

        return trajectories

    # ----- Level crossings -------------------------------------------------

    def level_crossings(
        self,
        eigenvalue_trajectories: np.ndarray,
    ) -> List[Dict[str, Any]]:
        r"""Find level crossings and avoided crossings.

        A crossing occurs when two eigenvalue trajectories swap order.

        Parameters
        ----------
        eigenvalue_trajectories : np.ndarray, shape (T, N)
            Eigenvalue trajectories.

        Returns
        -------
        list of dict
            Each dict has ``step``, ``level_pair``, ``gap``.
        """
        T, N = eigenvalue_trajectories.shape
        crossings = []

        for t in range(1, T):
            for k in range(N - 1):
                gap_prev = eigenvalue_trajectories[t - 1, k + 1] - eigenvalue_trajectories[t - 1, k]
                gap_curr = eigenvalue_trajectories[t, k + 1] - eigenvalue_trajectories[t, k]
                # Avoided crossing: gap reaches a minimum
                if t >= 2:
                    gap_prev2 = eigenvalue_trajectories[t - 2, k + 1] - eigenvalue_trajectories[t - 2, k]
                    if gap_prev < gap_prev2 and gap_prev < gap_curr:
                        crossings.append({
                            'step': t - 1,
                            'level_pair': (k, k + 1),
                            'gap': float(gap_prev),
                        })

        return crossings

    # ----- Spectral gap evolution ------------------------------------------

    def spectral_gap_evolution(
        self,
        eigenvalue_trajectories: np.ndarray,
    ) -> np.ndarray:
        r"""Track the spectral gap Δ = λ₁ - λ₀ over parameter range.

        Parameters
        ----------
        eigenvalue_trajectories : np.ndarray, shape (T, N)
            Eigenvalue trajectories.

        Returns
        -------
        np.ndarray, shape (T,)
            Spectral gap at each parameter value.
        """
        if eigenvalue_trajectories.shape[1] < 2:
            return np.zeros(eigenvalue_trajectories.shape[0])
        return eigenvalue_trajectories[:, 1] - eigenvalue_trajectories[:, 0]

    # ----- Topological transitions -----------------------------------------

    def topological_transitions(
        self,
        eigenvalue_trajectories: np.ndarray,
    ) -> List[Dict[str, Any]]:
        r"""Detect changes in spectral topology (gap closings/openings).

        A topological transition occurs when a spectral gap closes and
        reopens, potentially changing the topological index.

        Parameters
        ----------
        eigenvalue_trajectories : np.ndarray, shape (T, N)
            Eigenvalue trajectories.

        Returns
        -------
        list of dict
            Transitions with ``step``, ``gap_minimum``, ``levels``.
        """
        T, N = eigenvalue_trajectories.shape
        transitions = []

        for k in range(N - 1):
            gaps = eigenvalue_trajectories[:, k + 1] - eigenvalue_trajectories[:, k]
            # Find minima in the gap
            for t in range(1, T - 1):
                if gaps[t] < gaps[t - 1] and gaps[t] < gaps[t + 1]:
                    if gaps[t] < 0.01 * np.mean(gaps):
                        transitions.append({
                            'step': t,
                            'gap_minimum': float(gaps[t]),
                            'levels': (k, k + 1),
                        })

        return transitions

    # ----- Avoided crossing analysis ---------------------------------------

    def avoided_crossing_analysis(
        self,
        eigenvalue_pair: np.ndarray,
        parameter_range: np.ndarray,
    ) -> Dict[str, Any]:
        r"""Analyze the gap at an avoided crossing between two levels.

        Fits the two-level model: E_± = ε̄ ± √(δ² + V²) where δ is the
        bare splitting and V is the coupling.

        Parameters
        ----------
        eigenvalue_pair : np.ndarray, shape (T, 2)
            The two eigenvalue trajectories.
        parameter_range : np.ndarray
            Parameter values.

        Returns
        -------
        dict
            ``gap_minimum``, ``coupling_V``, ``crossing_parameter``.
        """
        gaps = eigenvalue_pair[:, 1] - eigenvalue_pair[:, 0]
        min_idx = np.argmin(gaps)
        gap_min = float(gaps[min_idx])
        crossing_param = float(parameter_range[min_idx])

        # Coupling V = gap_min / 2 from the two-level model
        V = gap_min / 2.0

        return {
            'gap_minimum': gap_min,
            'coupling_V': float(V),
            'crossing_parameter': crossing_param,
            'min_index': int(min_idx),
        }

    # ----- Berry phase -----------------------------------------------------

    def berry_phase(
        self,
        eigenvector_trajectories: np.ndarray,
    ) -> np.ndarray:
        r"""Geometric (Berry) phase around a loop in parameter space.

        γ = -Im Σ_t ln ⟨v(t)|v(t+1)⟩

        Parameters
        ----------
        eigenvector_trajectories : np.ndarray, shape (T, N)
            Eigenvector at each parameter step (single level).

        Returns
        -------
        np.ndarray
            Berry phase (one value per level if 2D input; scalar if 1D).
        """
        V = np.asarray(eigenvector_trajectories, dtype=np.complex128)
        if V.ndim == 1:
            return np.array([0.0])

        T = V.shape[0]
        phase = 0.0
        for t in range(T - 1):
            overlap = np.dot(np.conj(V[t]), V[t + 1])
            if abs(overlap) > 1e-14:
                phase -= np.angle(overlap)

        # Close the loop
        overlap = np.dot(np.conj(V[-1]), V[0])
        if abs(overlap) > 1e-14:
            phase -= np.angle(overlap)

        return np.array([float(phase)])

    # ----- Spectral zeta function ------------------------------------------

    def spectral_zeta_function(
        self,
        eigenvalues: np.ndarray,
        s_range: np.ndarray,
    ) -> np.ndarray:
        r"""Spectral zeta function ζ(s) = Σ λ_i^{-s}.

        Only sums over positive eigenvalues.

        Parameters
        ----------
        eigenvalues : np.ndarray
            Eigenvalues.
        s_range : np.ndarray
            Values of s at which to evaluate.

        Returns
        -------
        np.ndarray
            ζ(s) values.
        """
        eigs = np.asarray(eigenvalues, dtype=np.float64)
        eigs = eigs[eigs > 1e-14]
        s_range = np.asarray(s_range, dtype=np.float64)

        zeta = np.empty(len(s_range))
        for i, s in enumerate(s_range):
            zeta[i] = np.sum(eigs ** (-s))
        return zeta

    # ----- Spectral determinant --------------------------------------------

    def spectral_determinant(self, eigenvalues: np.ndarray) -> float:
        r"""Regularized spectral determinant det' = exp(-ζ'(0)).

        Uses ζ'(0) = -Σ ln λ_i (sum over positive eigenvalues).

        Parameters
        ----------
        eigenvalues : np.ndarray
            Eigenvalues.

        Returns
        -------
        float
            Regularized determinant.
        """
        eigs = np.asarray(eigenvalues, dtype=np.float64)
        eigs = eigs[eigs > 1e-14]
        if len(eigs) == 0:
            return 0.0

        # ζ'(0) = -Σ ln λ_i
        zeta_prime_0 = -np.sum(np.log(eigs))
        return float(np.exp(-zeta_prime_0))

    # ----- Heat kernel trace -----------------------------------------------

    def heat_kernel_trace(
        self,
        eigenvalues: np.ndarray,
        t_range: np.ndarray,
    ) -> np.ndarray:
        r"""Heat kernel trace K(t) = Σ exp(-λ_i t).

        Parameters
        ----------
        eigenvalues : np.ndarray
            Eigenvalues.
        t_range : np.ndarray
            Time values.

        Returns
        -------
        np.ndarray
            K(t) values.
        """
        eigs = np.asarray(eigenvalues, dtype=np.float64)
        t_range = np.asarray(t_range, dtype=np.float64)

        K_t = np.empty(len(t_range))
        for i, t in enumerate(t_range):
            K_t[i] = np.sum(np.exp(-eigs * t))
        return K_t


# ---------------------------------------------------------------------------
# 5. NTKSpectrumAnalyzer
# ---------------------------------------------------------------------------

class NTKSpectrumAnalyzer:
    r"""Spectrum analysis specialized for neural tangent kernel matrices.

    Provides bulk and edge comparisons with RMT predictions, spike
    detection (BBP), effective dimension, generalization prediction from
    spectrum, double-descent prediction, and phase classification.

    Parameters
    ----------
    ntk_matrix : np.ndarray, shape (N, N)
        Neural tangent kernel matrix (symmetric positive semi-definite).
    """

    def __init__(self, ntk_matrix: np.ndarray):
        self.K = np.asarray(ntk_matrix, dtype=np.float64)
        self.N = self.K.shape[0]
        self.eigenvalues = np.linalg.eigvalsh(self.K)
        self._esd = EmpiricalSpectralDistribution(self.eigenvalues)

    # ----- Bulk analysis ---------------------------------------------------

    def bulk_analysis(
        self,
        gamma: Optional[float] = None,
    ) -> Dict[str, Any]:
        r"""Compare bulk spectrum with Marchenko-Pastur law.

        Parameters
        ----------
        gamma : float or None
            Aspect ratio N/P.  If None, inferred from matrix size.

        Returns
        -------
        dict
            ``gamma``, ``mp_edges``, ``empirical_edges``,
            ``ks_distance``, ``bulk_match``.
        """
        if gamma is None:
            gamma = 1.0

        # MP edges: λ± = σ² (1 ± √γ)²
        sigma2 = np.mean(self.eigenvalues)  # rough scale
        lam_minus = sigma2 * (1 - np.sqrt(gamma)) ** 2
        lam_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2

        emp_min = float(self.eigenvalues[0])
        emp_max = float(self.eigenvalues[-1])

        # KS distance in the bulk
        x_grid = np.linspace(max(emp_min, 0), emp_max, 200)
        emp_cdf = self._esd.empirical_cdf(x_grid)

        # MP CDF (approximate via numerical integration of density)
        mp_density = np.zeros_like(x_grid)
        for i, x in enumerate(x_grid):
            if lam_minus <= x <= lam_plus and lam_plus > lam_minus:
                mp_density[i] = (np.sqrt((lam_plus - x) * (x - lam_minus))
                                 / (2 * np.pi * gamma * sigma2 * x + 1e-14))
        mp_cdf = np.cumsum(mp_density) * (x_grid[1] - x_grid[0]) if len(x_grid) > 1 else mp_density

        ks = float(np.max(np.abs(emp_cdf - mp_cdf)))

        return {
            'gamma': gamma,
            'mp_edges': (float(lam_minus), float(lam_plus)),
            'empirical_edges': (emp_min, emp_max),
            'ks_distance': ks,
            'bulk_match': ks < 0.1,
        }

    # ----- Edge analysis ---------------------------------------------------

    def edge_analysis(self) -> Dict[str, Any]:
        r"""Compare spectral edges with Tracy-Widom predictions.

        Returns
        -------
        dict
            ``max_eigenvalue``, ``tw_scaled``, ``edge_statistics``.
        """
        lam_max = float(self.eigenvalues[-1])
        lam_mean = float(np.mean(self.eigenvalues))
        lam_std = float(np.std(self.eigenvalues))

        # TW scaling: (λ_max - μ_edge) / (σ N^{-2/3})
        N = self.N
        if lam_std > 1e-14 and N > 0:
            tw_scaled = (lam_max - lam_mean) / (lam_std * N ** (-2.0 / 3))
        else:
            tw_scaled = 0.0

        return {
            'max_eigenvalue': lam_max,
            'tw_scaled': float(tw_scaled),
            'mean': lam_mean,
            'std': lam_std,
        }

    # ----- Spike detection (BBP) -------------------------------------------

    def spike_detection(
        self,
        significance: float = 0.05,
    ) -> Dict[str, Any]:
        r"""Detect outlier eigenvalues via the BBP (Baik-Ben Arous-Péché) transition.

        An eigenvalue is a spike if it exceeds the bulk edge
        λ_+ = σ² (1 + √γ)² by more than a threshold determined by
        Tracy-Widom fluctuations.

        Parameters
        ----------
        significance : float
            Significance level for spike detection.

        Returns
        -------
        dict
            ``n_spikes``, ``spike_eigenvalues``, ``bulk_edge``.
        """
        sigma2 = np.median(self.eigenvalues)
        gamma = 1.0
        lam_plus = sigma2 * (1 + np.sqrt(gamma)) ** 2

        # TW threshold (approximate: TW_1 95th percentile ≈ 0.98)
        tw_threshold = 0.98 * (sigma2 * gamma ** (1.0 / 6) * self.N ** (-2.0 / 3))

        spike_threshold = lam_plus + tw_threshold / significance
        spikes = self.eigenvalues[self.eigenvalues > spike_threshold]

        return {
            'n_spikes': int(len(spikes)),
            'spike_eigenvalues': spikes,
            'bulk_edge': float(lam_plus),
            'threshold': float(spike_threshold),
        }

    # ----- Condition number ------------------------------------------------

    def eigenvalue_condition_number(self) -> float:
        r"""Condition number κ = λ_max / λ_min.

        Returns
        -------
        float
            Condition number (inf if λ_min ≈ 0).
        """
        pos_eigs = self.eigenvalues[self.eigenvalues > 1e-14]
        if len(pos_eigs) == 0:
            return np.inf
        return float(pos_eigs[-1] / pos_eigs[0])

    # ----- Spectral norm ratio ---------------------------------------------

    def spectral_norm_ratio(self) -> float:
        r"""Spectral norm ratio ||K|| / tr(K) indicating concentration.

        Ratio ≈ 1/N for flat spectrum; ratio ≈ 1 for rank-1 matrix.

        Returns
        -------
        float
            Spectral norm ratio.
        """
        trace = np.sum(self.eigenvalues)
        if trace < 1e-14:
            return np.inf
        return float(self.eigenvalues[-1] / trace)

    # ----- Kernel alignment ------------------------------------------------

    def kernel_alignment(self, target_kernel: np.ndarray) -> float:
        r"""Kernel alignment ⟨K₁, K₂⟩_F / (||K₁||_F ||K₂||_F).

        Parameters
        ----------
        target_kernel : np.ndarray
            Target kernel matrix.

        Returns
        -------
        float
            Alignment ∈ [-1, 1].
        """
        K2 = np.asarray(target_kernel, dtype=np.float64)
        norm1 = np.linalg.norm(self.K, 'fro')
        norm2 = np.linalg.norm(K2, 'fro')
        if norm1 < 1e-14 or norm2 < 1e-14:
            return 0.0
        return float(np.sum(self.K * K2) / (norm1 * norm2))

    # ----- Effective dimension ---------------------------------------------

    def effective_dimension(self, regularization: float = 0.0) -> float:
        r"""Effective dimension d_eff = tr(K(K + λI)^{-1}).

        Parameters
        ----------
        regularization : float
            Ridge parameter λ ≥ 0.

        Returns
        -------
        float
            Effective dimension.
        """
        reg_eigs = self.eigenvalues + regularization
        d_eff = np.sum(self.eigenvalues / np.maximum(reg_eigs, 1e-14))
        return float(d_eff)

    # ----- Spectral gap to ridge -------------------------------------------

    def spectral_gap_to_ridge(
        self,
        lambda_range: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        r"""How the spectral gap affects the optimal ridge parameter.

        Computes d_eff(λ) and the bias-variance trade-off as a function
        of the ridge parameter λ.

        Parameters
        ----------
        lambda_range : np.ndarray
            Ridge parameter values.

        Returns
        -------
        dict
            ``lambda_values``, ``effective_dimension``, ``shrinkage``.
        """
        lams = np.asarray(lambda_range, dtype=np.float64)
        d_eff = np.array([self.effective_dimension(l) for l in lams])

        # Shrinkage: average λ_i / (λ_i + λ)
        shrinkage = np.empty(len(lams))
        for i, lam in enumerate(lams):
            shrinkage[i] = float(np.mean(
                self.eigenvalues / (self.eigenvalues + lam + 1e-14)
            ))

        return {
            'lambda_values': lams,
            'effective_dimension': d_eff,
            'shrinkage': shrinkage,
        }

    # ----- Generalization from spectrum ------------------------------------

    def generalization_from_spectrum(
        self,
        n_train: int,
        noise_var: float,
    ) -> Dict[str, float]:
        r"""Predict test error from the NTK spectrum.

        For kernel ridge regression with regularization λ = 0:
            Bias² ≈ Σ_{k>n} c_k²
            Variance ≈ (σ²/n) Σ_{k≤n} 1

        Parameters
        ----------
        n_train : int
            Number of training samples.
        noise_var : float
            Noise variance σ².

        Returns
        -------
        dict
            ``bias_squared``, ``variance``, ``test_error``.
        """
        sorted_eigs = np.sort(self.eigenvalues)[::-1]
        N = len(sorted_eigs)

        # Rough spectral decomposition
        if n_train >= N:
            bias_sq = 0.0
            variance = noise_var * np.sum(1.0 / (sorted_eigs + 1e-14)) / n_train
        else:
            # Bias from unlearned modes
            bias_sq = float(np.sum(sorted_eigs[n_train:]))
            variance = float(noise_var * n_train / (n_train + 1e-14))

        return {
            'bias_squared': float(bias_sq),
            'variance': float(variance),
            'test_error': float(bias_sq + variance),
        }

    # ----- Double descent prediction ---------------------------------------

    def double_descent_prediction(
        self,
        n_train_range: np.ndarray,
        noise_var: float,
    ) -> Dict[str, np.ndarray]:
        r"""Predict double descent from spectral structure.

        Test error peaks near n_train ≈ N (interpolation threshold).

        Parameters
        ----------
        n_train_range : np.ndarray
            Array of training set sizes.
        noise_var : float
            Noise variance.

        Returns
        -------
        dict
            ``n_train``, ``test_error``, ``bias``, ``variance``.
        """
        ns = np.asarray(n_train_range, dtype=int)
        errors = np.empty(len(ns))
        biases = np.empty(len(ns))
        variances = np.empty(len(ns))

        for i, n in enumerate(ns):
            result = self.generalization_from_spectrum(int(n), noise_var)
            errors[i] = result['test_error']
            biases[i] = result['bias_squared']
            variances[i] = result['variance']

        return {
            'n_train': ns,
            'test_error': errors,
            'bias': biases,
            'variance': variances,
        }

    # ----- Phase from spectrum ---------------------------------------------

    def phase_from_spectrum(self) -> Dict[str, Any]:
        r"""Classify lazy/rich/critical regime from spectral shape.

        - Lazy: spectrum close to MP law (bulk-dominated)
        - Rich: significant spikes above bulk
        - Critical: power-law decay in spectrum

        Returns
        -------
        dict
            ``phase`` (str), ``spectral_indicators``.
        """
        sorted_eigs = np.sort(self.eigenvalues)[::-1]
        N = len(sorted_eigs)

        if N < 3:
            return {'phase': 'undetermined', 'spectral_indicators': {}}

        # Indicator 1: spike ratio (largest / median)
        median_eig = np.median(sorted_eigs)
        spike_ratio = sorted_eigs[0] / (median_eig + 1e-14)

        # Indicator 2: effective rank / N
        eff_rank = self._esd.effective_rank()
        rank_ratio = eff_rank / N

        # Indicator 3: power-law exponent (fit log-log)
        ranks = np.arange(1, N + 1, dtype=float)
        pos_mask = sorted_eigs > 1e-14
        if np.sum(pos_mask) > 3:
            log_rank = np.log(ranks[pos_mask])
            log_eig = np.log(sorted_eigs[pos_mask])
            coeffs = np.polyfit(log_rank, log_eig, 1)
            power_law_exp = -coeffs[0]
        else:
            power_law_exp = 0.0

        # Classification
        if spike_ratio > 10 and rank_ratio < 0.3:
            phase = 'rich'
        elif 0.8 < power_law_exp < 2.0 and rank_ratio < 0.5:
            phase = 'critical'
        else:
            phase = 'lazy'

        return {
            'phase': phase,
            'spectral_indicators': {
                'spike_ratio': float(spike_ratio),
                'rank_ratio': float(rank_ratio),
                'power_law_exponent': float(power_law_exp),
                'effective_rank': float(eff_rank),
            },
        }


# ---------------------------------------------------------------------------
# 6. MatrixEnsembleGenerator
# ---------------------------------------------------------------------------

class MatrixEnsembleGenerator:
    r"""Generate random matrices from standard RMT ensembles.

    Provides GOE, GUE, Wishart, spiked Wishart, correlated Wishart,
    Jacobi, circular, and NTK-like ensembles for benchmarking.

    Parameters
    ----------
    N : int
        Matrix dimension.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, N: int, seed: Optional[int] = None):
        if N <= 0:
            raise ValueError("N must be positive.")
        self.N = N
        self.rng = np.random.default_rng(seed)

    # ----- GOE -------------------------------------------------------------

    def goe(self) -> np.ndarray:
        r"""Gaussian Orthogonal Ensemble: H = (A + A^T) / (2√N).

        Returns
        -------
        np.ndarray, shape (N, N)
            GOE random matrix.
        """
        A = self.rng.normal(size=(self.N, self.N))
        H = (A + A.T) / (2 * np.sqrt(self.N))
        return H

    # ----- GUE -------------------------------------------------------------

    def gue(self) -> np.ndarray:
        r"""Gaussian Unitary Ensemble: H = (A + A†) / (2√N).

        Returns
        -------
        np.ndarray, shape (N, N)
            GUE random matrix (Hermitian, complex).
        """
        A = (self.rng.normal(size=(self.N, self.N))
             + 1j * self.rng.normal(size=(self.N, self.N)))
        H = (A + A.conj().T) / (2 * np.sqrt(self.N))
        return H

    # ----- Wishart ---------------------------------------------------------

    def wishart(self, gamma: float) -> np.ndarray:
        r"""Wishart matrix W = (1/P) X X^T where X is N × P.

        Parameters
        ----------
        gamma : float
            Aspect ratio γ = N/P.

        Returns
        -------
        np.ndarray, shape (N, N)
            Wishart random matrix.
        """
        if gamma <= 0:
            raise ValueError("gamma must be positive.")
        P = max(int(self.N / gamma), 1)
        X = self.rng.normal(size=(self.N, P)) / np.sqrt(P)
        return X @ X.T

    # ----- Spiked Wishart --------------------------------------------------

    def spiked_wishart(
        self,
        gamma: float,
        spikes: np.ndarray,
    ) -> np.ndarray:
        r"""Spiked Wishart model: population covariance Σ = I + Σ_k θ_k v_k v_k^T.

        Parameters
        ----------
        gamma : float
            Aspect ratio.
        spikes : np.ndarray
            Spike strengths θ_k.

        Returns
        -------
        np.ndarray, shape (N, N)
            Spiked Wishart matrix.
        """
        spikes = np.asarray(spikes, dtype=np.float64)
        P = max(int(self.N / gamma), 1)

        # Population covariance
        Sigma = np.eye(self.N)
        for k, theta in enumerate(spikes):
            if k >= self.N:
                break
            v = np.zeros(self.N)
            v[k] = 1.0
            Sigma += theta * np.outer(v, v)

        L = np.linalg.cholesky(Sigma)
        X = (L @ self.rng.normal(size=(self.N, P))) / np.sqrt(P)
        return X @ X.T

    # ----- Correlated Wishart ----------------------------------------------

    def correlated_wishart(
        self,
        gamma: float,
        population_covariance: np.ndarray,
    ) -> np.ndarray:
        r"""Wishart with structured population covariance.

        Parameters
        ----------
        gamma : float
            Aspect ratio.
        population_covariance : np.ndarray, shape (N, N)
            Population covariance Σ.

        Returns
        -------
        np.ndarray, shape (N, N)
            Correlated Wishart matrix.
        """
        Sigma = np.asarray(population_covariance, dtype=np.float64)
        P = max(int(self.N / gamma), 1)

        L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(self.N))
        X = (L @ self.rng.normal(size=(self.N, P))) / np.sqrt(P)
        return X @ X.T

    # ----- Jacobi ensemble -------------------------------------------------

    def jacobi_ensemble(self, m1: int, m2: int) -> np.ndarray:
        r"""Jacobi (MANOVA) ensemble.

        Eigenvalues of A(A+B)^{-1} where A ~ Wishart(N, m1), B ~ Wishart(N, m2).

        Parameters
        ----------
        m1, m2 : int
            Degrees of freedom for the two Wishart matrices.

        Returns
        -------
        np.ndarray, shape (N, N)
            Jacobi ensemble matrix.
        """
        X1 = self.rng.normal(size=(self.N, m1))
        X2 = self.rng.normal(size=(self.N, m2))
        A = X1 @ X1.T
        B = X2 @ X2.T
        AB_inv = np.linalg.solve(A + B + 1e-10 * np.eye(self.N), np.eye(self.N))
        return A @ AB_inv

    # ----- Circular ensemble -----------------------------------------------

    def circular_ensemble(self, beta: int = 2) -> np.ndarray:
        r"""Circular ensemble (CUE for β=2, COE for β=1).

        Parameters
        ----------
        beta : int
            Dyson index (1 = COE, 2 = CUE).

        Returns
        -------
        np.ndarray, shape (N, N)
            Unitary (or orthogonal) random matrix.
        """
        if beta == 2:
            # CUE: QR decomposition of complex Gaussian matrix
            Z = (self.rng.normal(size=(self.N, self.N))
                 + 1j * self.rng.normal(size=(self.N, self.N)))
            Q, R = np.linalg.qr(Z)
            # Fix phases for Haar measure
            d = np.diag(R)
            ph = d / np.abs(d + 1e-300)
            return Q * ph
        else:
            # COE: QR of real Gaussian
            Z = self.rng.normal(size=(self.N, self.N))
            Q, R = np.linalg.qr(Z)
            d = np.diag(R)
            ph = np.sign(d)
            ph[ph == 0] = 1
            return Q * ph

    # ----- NTK ensemble ----------------------------------------------------

    def ntk_ensemble(
        self,
        depth: int = 2,
        activation: str = 'relu',
    ) -> np.ndarray:
        r"""Generate a random NTK matrix from a random neural network.

        Constructs the empirical NTK Θ(x, x') = Σ_p ∂f/∂θ_p(x) ∂f/∂θ_p(x')
        for a random network evaluated on N random inputs.

        Parameters
        ----------
        depth : int
            Network depth.
        activation : str
            Activation function.

        Returns
        -------
        np.ndarray, shape (N, N)
            Empirical NTK matrix.
        """
        d = max(self.N, 5)
        width = self.N
        X = self.rng.normal(size=(self.N, d)) / np.sqrt(d)

        # Build NTK via forward pass and Jacobians
        # For simplicity, use the recursive NTK formula for fully-connected nets
        # K^0 = (1/d) X X^T
        K = X @ X.T / d
        K_dot = K.copy()  # derivative kernel

        for l in range(depth):
            # For ReLU: K^{l+1} = (1/2π) [√(1-c²) + (π-arccos(c))c]
            # where c_{ij} = K^l_{ij} / √(K^l_{ii} K^l_{jj})
            diag = np.sqrt(np.maximum(np.diag(K), 1e-14))
            C = K / np.outer(diag, diag)
            C = np.clip(C, -1 + 1e-10, 1 - 1e-10)

            if activation == 'relu':
                angle = np.arccos(C)
                K_new = (1 / (2 * np.pi)) * (
                    np.sqrt(1 - C ** 2) + (np.pi - angle) * C
                ) * np.outer(diag, diag)
                K_dot_new = (1 / (2 * np.pi)) * (np.pi - angle)
            elif activation == 'tanh':
                K_new = (2 / np.pi) * np.arcsin(
                    2 * C / np.sqrt((1 + 2 * np.diag(K)[None, :])
                                    * (1 + 2 * np.diag(K)[:, None]))
                )
                K_dot_new = np.ones_like(K)  # simplified
            else:
                K_new = K
                K_dot_new = np.ones_like(K)

            # NTK recursion: Θ^{l+1} = Θ^l ⊙ K_dot + K_new
            K_dot = K_dot * K_dot_new
            K = K_new

        # Full NTK = sum of per-layer contributions (simplified)
        ntk = K + K_dot * (X @ X.T / d)
        # Symmetrize
        ntk = 0.5 * (ntk + ntk.T)
        return ntk


# ---------------------------------------------------------------------------
# 7. SpectralComparisonTool
# ---------------------------------------------------------------------------

class SpectralComparisonTool:
    r"""Tools for comparing two spectral measures.

    Provides Wasserstein distance, Kolmogorov-Smirnov test, QQ plot data,
    L^p spectral distance, and log-determinant ratio.
    """

    def __init__(self):
        pass

    # ----- Wasserstein distance --------------------------------------------

    def wasserstein_distance(
        self,
        eigs1: np.ndarray,
        eigs2: np.ndarray,
    ) -> float:
        r"""W₁ (earth mover's) distance between spectral measures.

        W₁ = ∫ |F₁(x) - F₂(x)| dx  =  (1/N) Σ |λ_{(i)}^1 - λ_{(i)}^2|

        for equally sized sorted samples.

        Parameters
        ----------
        eigs1, eigs2 : np.ndarray
            Two sets of eigenvalues.

        Returns
        -------
        float
            Wasserstein-1 distance.
        """
        e1 = np.sort(np.asarray(eigs1, dtype=np.float64))
        e2 = np.sort(np.asarray(eigs2, dtype=np.float64))

        # If different sizes, interpolate to common grid
        if len(e1) != len(e2):
            n = max(len(e1), len(e2))
            q1 = np.interp(
                np.linspace(0, 1, n),
                np.linspace(0, 1, len(e1)),
                e1
            )
            q2 = np.interp(
                np.linspace(0, 1, n),
                np.linspace(0, 1, len(e2)),
                e2
            )
        else:
            q1, q2 = e1, e2

        return float(np.mean(np.abs(q1 - q2)))

    # ----- Kolmogorov-Smirnov test -----------------------------------------

    def kolmogorov_smirnov(
        self,
        eigs1: np.ndarray,
        eigs2: np.ndarray,
    ) -> Dict[str, float]:
        r"""Two-sample Kolmogorov-Smirnov test between spectral distributions.

        Parameters
        ----------
        eigs1, eigs2 : np.ndarray
            Two sets of eigenvalues.

        Returns
        -------
        dict
            ``statistic``, ``pvalue``.
        """
        result = stats.ks_2samp(eigs1, eigs2)
        return {
            'statistic': float(result.statistic),
            'pvalue': float(result.pvalue),
        }

    # ----- QQ plot data ----------------------------------------------------

    def qq_plot_data(
        self,
        empirical_eigs: np.ndarray,
        theoretical_quantiles: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        r"""Prepare QQ plot data from empirical and theoretical quantiles.

        Parameters
        ----------
        empirical_eigs : np.ndarray
            Sorted empirical eigenvalues.
        theoretical_quantiles : np.ndarray
            Theoretical quantiles (same length).

        Returns
        -------
        dict
            ``empirical``, ``theoretical`` (sorted arrays for QQ plot).
        """
        e = np.sort(np.asarray(empirical_eigs, dtype=np.float64))
        t = np.sort(np.asarray(theoretical_quantiles, dtype=np.float64))

        # Match lengths
        n = min(len(e), len(t))
        if len(e) > n:
            e = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(e)), e)
        if len(t) > n:
            t = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(t)), t)

        return {
            'empirical': e,
            'theoretical': t,
        }

    # ----- L^p spectral distance -------------------------------------------

    def spectral_distance(
        self,
        eigs1: np.ndarray,
        eigs2: np.ndarray,
        p: float = 2.0,
    ) -> float:
        r"""L^p distance between spectral densities.

        Approximated via sorted quantile matching:
            d_p = ((1/N) Σ |λ_{(i)}^1 - λ_{(i)}^2|^p)^{1/p}.

        Parameters
        ----------
        eigs1, eigs2 : np.ndarray
            Two sets of eigenvalues.
        p : float
            Exponent (default 2).

        Returns
        -------
        float
            L^p distance.
        """
        e1 = np.sort(np.asarray(eigs1, dtype=np.float64))
        e2 = np.sort(np.asarray(eigs2, dtype=np.float64))

        if len(e1) != len(e2):
            n = max(len(e1), len(e2))
            e1 = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(e1)), e1)
            e2 = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(e2)), e2)

        return float(np.mean(np.abs(e1 - e2) ** p) ** (1.0 / p))

    # ----- Log-determinant ratio -------------------------------------------

    def log_determinant_ratio(
        self,
        eigs1: np.ndarray,
        eigs2: np.ndarray,
    ) -> float:
        r"""Log-determinant ratio ln det(K₁) / det(K₂) = Σ ln(λ_i^1 / λ_i^2).

        Parameters
        ----------
        eigs1, eigs2 : np.ndarray
            Eigenvalues of the two matrices (must be positive).

        Returns
        -------
        float
            Log-determinant ratio.
        """
        e1 = np.sort(np.asarray(eigs1, dtype=np.float64))
        e2 = np.sort(np.asarray(eigs2, dtype=np.float64))

        e1 = np.maximum(e1, 1e-300)
        e2 = np.maximum(e2, 1e-300)

        if len(e1) != len(e2):
            n = min(len(e1), len(e2))
            e1 = e1[-n:]
            e2 = e2[-n:]

        return float(np.sum(np.log(e1) - np.log(e2)))
