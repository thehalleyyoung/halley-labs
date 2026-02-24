"""
Marchenko-Pastur law and related spectral tools.

Implements the Stieltjes transform, Marchenko-Pastur density,
resolvent computations, and free convolution for analyzing
bulk eigenvalue distributions of large random matrices and NTKs.
"""

import numpy as np
from scipy import integrate, stats


class StieltjesTransform:
    """Stieltjes transform for the Marchenko-Pastur distribution.

    The Stieltjes transform m(z) encodes the spectral density of a
    probability measure via m(z) = ∫ dμ(x)/(x - z). For the
    Marchenko-Pastur law with ratio γ = N/P, it satisfies a
    quadratic self-consistency equation.

    Parameters
    ----------
    gamma : float
        Aspect ratio N/P (number of samples / dimension).
    """

    def __init__(self, gamma: float):
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        self.gamma = gamma

    def evaluate(self, z: complex) -> complex:
        """Evaluate the Stieltjes transform m(z) of the MP distribution.

        Solves m(z) = [-(z - γ + 1) + sqrt((z - γ - 1)² - 4γ)] / (2γz)
        choosing the branch with Im[m(z)] > 0 when Im[z] > 0.

        Parameters
        ----------
        z : complex
            Point in the upper half-plane (or real axis with small imaginary part).

        Returns
        -------
        complex
            The Stieltjes transform m(z).
        """
        z = complex(z)
        g = self.gamma
        discriminant = (z - g - 1) ** 2 - 4 * g
        sqrt_disc = np.lib.scimath.sqrt(discriminant)

        # Choose branch so that Im[m] > 0 when Im[z] > 0
        m_plus = (-(z - g + 1) + sqrt_disc) / (2 * g * z)
        m_minus = (-(z - g + 1) - sqrt_disc) / (2 * g * z)

        if z.imag > 0:
            return m_plus if m_plus.imag > 0 else m_minus
        elif z.imag < 0:
            return m_plus if m_plus.imag < 0 else m_minus
        else:
            # Real z: return the one with positive imaginary part limit
            return m_plus

    def density_from_stieltjes(self, x_range: np.ndarray, eta: float = 1e-6) -> np.ndarray:
        """Recover the spectral density from the Stieltjes transform.

        Uses ρ(x) = (1/π) Im[m(x + iη)] as η → 0⁺.

        Parameters
        ----------
        x_range : np.ndarray
            Real-valued grid points.
        eta : float
            Small imaginary regularization.

        Returns
        -------
        np.ndarray
            Spectral density at each point in x_range.
        """
        density = np.zeros_like(x_range, dtype=float)
        for i, x in enumerate(x_range):
            m = self.evaluate(complex(x, eta))
            density[i] = m.imag / np.pi
        return density

    def inverse_stieltjes(self, m: complex) -> complex:
        """Invert the Stieltjes transform: given m, find z such that m(z) = m.

        From the self-consistency equation:
        z = -1/m + γ/(1 + γm)  (for the MP law with σ²=1).

        Parameters
        ----------
        m : complex
            Value of the Stieltjes transform.

        Returns
        -------
        complex
            The corresponding z value.
        """
        m = complex(m)
        return -1.0 / m + self.gamma / (1.0 + self.gamma * m)

    def derivative(self, z: complex, dz: float = 1e-8) -> complex:
        """Compute m'(z) via finite difference.

        Parameters
        ----------
        z : complex
            Evaluation point.
        dz : float
            Step size for finite difference.

        Returns
        -------
        complex
            Approximate derivative m'(z).
        """
        z = complex(z)
        return (self.evaluate(z + dz) - self.evaluate(z - dz)) / (2 * dz)


class MarchenkoPasturLaw:
    """The Marchenko-Pastur distribution for sample covariance eigenvalues.

    For an N×P matrix X with i.i.d. entries of variance σ²/P,
    the empirical spectral distribution of X^T X converges to
    the MP law as N, P → ∞ with N/P → γ.

    Parameters
    ----------
    gamma : float
        Aspect ratio N/P.
    sigma_sq : float
        Variance parameter (default 1.0).
    """

    def __init__(self, gamma: float, sigma_sq: float = 1.0):
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if sigma_sq <= 0:
            raise ValueError("sigma_sq must be positive")
        self.gamma = gamma
        self.sigma_sq = sigma_sq
        self._stieltjes = StieltjesTransform(gamma)

    def support(self) -> tuple:
        """Return the support [λ⁻, λ⁺] of the MP distribution.

        λ± = σ²(1 ± √γ)².

        Returns
        -------
        tuple
            (lambda_minus, lambda_plus).
        """
        s = self.sigma_sq
        sq = np.sqrt(self.gamma)
        return (s * (1 - sq) ** 2, s * (1 + sq) ** 2)

    def density(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the Marchenko-Pastur density.

        ρ(x) = 1/(2πγσ²x) * √((λ⁺ - x)(x - λ⁻))  for x ∈ [λ⁻, λ⁺].

        Parameters
        ----------
        x : np.ndarray
            Points at which to evaluate the density.

        Returns
        -------
        np.ndarray
            Density values (zero outside the support).
        """
        x = np.asarray(x, dtype=float)
        lam_m, lam_p = self.support()
        density = np.zeros_like(x)
        mask = (x >= lam_m) & (x <= lam_p) & (x > 0)
        xm = x[mask]
        density[mask] = (
            np.sqrt((lam_p - xm) * (xm - lam_m))
            / (2 * np.pi * self.gamma * self.sigma_sq * xm)
        )
        return density

    def bulk_eigenvalue_cdf(self, x: np.ndarray) -> np.ndarray:
        """CDF of the MP distribution via numerical integration.

        Parameters
        ----------
        x : np.ndarray
            Points at which to evaluate the CDF.

        Returns
        -------
        np.ndarray
            CDF values.
        """
        x = np.asarray(x, dtype=float)
        lam_m, lam_p = self.support()
        cdf = np.zeros_like(x)
        p0 = self.atom_at_zero()

        for i, xi in enumerate(x):
            if xi <= lam_m:
                cdf[i] = p0 if xi >= 0 else 0.0
            elif xi >= lam_p:
                cdf[i] = 1.0
            else:
                integral, _ = integrate.quad(
                    lambda t: self._density_scalar(t), lam_m, xi
                )
                cdf[i] = p0 + integral
        return cdf

    def _density_scalar(self, x: float) -> float:
        """Scalar version of density for integration."""
        lam_m, lam_p = self.support()
        if x <= lam_m or x >= lam_p or x <= 0:
            return 0.0
        return (
            np.sqrt((lam_p - x) * (x - lam_m))
            / (2 * np.pi * self.gamma * self.sigma_sq * x)
        )

    def moments(self, k: int) -> float:
        """Compute the k-th moment of the MP distribution.

        The moments are related to Narayana numbers:
        M_k = σ^{2k} Σ_{j=1}^{k} (1/k) C(k,j) C(k,j-1) γ^{j-1}

        For γ=1, these reduce to Catalan numbers: C_k = (2k)! / ((k+1)! k!).

        Parameters
        ----------
        k : int
            Moment order (k ≥ 1).

        Returns
        -------
        float
            The k-th moment.
        """
        if k < 0:
            raise ValueError("k must be non-negative")
        if k == 0:
            return 1.0

        from scipy.special import comb
        moment = 0.0
        for j in range(1, k + 1):
            narayana = comb(k, j, exact=True) * comb(k, j - 1, exact=True) / k
            moment += narayana * self.gamma ** (j - 1)
        return moment * self.sigma_sq ** k

    def free_entropy(self) -> float:
        """Compute the free entropy (log-potential) of the MP distribution.

        Σ = ∫∫ log|x - y| dμ(x) dμ(y) computed analytically.
        For σ²=1: Σ = (γ+1)/(2γ) log((1+√γ)²) + log(γ)/2 - 3/2 - (log σ² terms).

        Returns
        -------
        float
            Free entropy value.
        """
        g = self.gamma
        s = self.sigma_sq
        lam_m, lam_p = self.support()

        # Numerical integration of ∫ log(x) ρ(x) dx
        def integrand(x):
            d = self._density_scalar(x)
            if d <= 0:
                return 0.0
            return np.log(x) * d

        result, _ = integrate.quad(integrand, lam_m, lam_p, limit=100)
        return result

    def fit_to_empirical(self, eigenvalues: np.ndarray) -> dict:
        """Fit gamma and sigma_sq to empirical eigenvalues via moment matching.

        Uses the first two moments of the empirical distribution to
        estimate γ and σ².

        Parameters
        ----------
        eigenvalues : np.ndarray
            Observed eigenvalues.

        Returns
        -------
        dict
            Fitted parameters {'gamma': float, 'sigma_sq': float}.
        """
        eigenvalues = np.asarray(eigenvalues, dtype=float)
        m1 = np.mean(eigenvalues)
        m2 = np.mean(eigenvalues ** 2)

        # For MP: E[λ] = σ², E[λ²] = σ⁴(1 + γ)
        sigma_sq = m1
        if sigma_sq > 0:
            gamma = m2 / (sigma_sq ** 2) - 1.0
            gamma = max(gamma, 0.01)
        else:
            gamma = 1.0
            sigma_sq = 1.0

        self.gamma = gamma
        self.sigma_sq = sigma_sq
        self._stieltjes = StieltjesTransform(gamma)
        return {"gamma": gamma, "sigma_sq": sigma_sq}

    def ks_test(self, eigenvalues: np.ndarray) -> dict:
        """Kolmogorov-Smirnov test of eigenvalues against MP distribution.

        Parameters
        ----------
        eigenvalues : np.ndarray
            Observed eigenvalues.

        Returns
        -------
        dict
            {'statistic': float, 'p_value': float, 'reject_at_005': bool}.
        """
        eigenvalues = np.sort(np.asarray(eigenvalues, dtype=float))
        n = len(eigenvalues)
        empirical_cdf = np.arange(1, n + 1) / n
        theoretical_cdf = self.bulk_eigenvalue_cdf(eigenvalues)

        ks_stat = np.max(np.abs(empirical_cdf - theoretical_cdf))
        # Approximate p-value via Kolmogorov distribution
        sqrt_n = np.sqrt(n)
        p_value = stats.kstwobign.sf(ks_stat * sqrt_n)

        return {
            "statistic": float(ks_stat),
            "p_value": float(p_value),
            "reject_at_005": p_value < 0.05,
        }

    def atom_at_zero(self) -> float:
        """Point mass at zero when γ > 1.

        When N > P (γ > 1), there are N - P zero eigenvalues,
        contributing a mass of max(0, 1 - 1/γ) at the origin.

        Returns
        -------
        float
            Mass at zero.
        """
        return max(0.0, 1.0 - 1.0 / self.gamma)


class ResolventComputer:
    """Compute the resolvent (Green's function) of a matrix.

    Given a symmetric matrix K (or its eigenvalues), computes
    R(z) = (K - zI)⁻¹ and derived spectral quantities.

    Parameters
    ----------
    matrix_or_eigenvalues : np.ndarray
        Either a square symmetric matrix or a 1-D array of eigenvalues.
    """

    def __init__(self, matrix_or_eigenvalues: np.ndarray):
        arr = np.asarray(matrix_or_eigenvalues, dtype=float)
        if arr.ndim == 2:
            if arr.shape[0] != arr.shape[1]:
                raise ValueError("Matrix must be square")
            self.eigenvalues = np.linalg.eigvalsh(arr)
            self._matrix = arr
        elif arr.ndim == 1:
            self.eigenvalues = np.sort(arr)
            self._matrix = None
        else:
            raise ValueError("Input must be 1-D or 2-D array")
        self.N = len(self.eigenvalues)

    def resolvent(self, z: complex) -> np.ndarray:
        """Compute the full resolvent matrix R(z) = (K - zI)⁻¹.

        Only available when initialized with a matrix.

        Parameters
        ----------
        z : complex
            Spectral parameter (should not be a real eigenvalue).

        Returns
        -------
        np.ndarray
            The resolvent matrix.
        """
        if self._matrix is None:
            raise ValueError("Full resolvent requires matrix input, not just eigenvalues")
        return np.linalg.inv(self._matrix - z * np.eye(self.N))

    def trace_resolvent(self, z: complex) -> complex:
        """Compute the normalized trace of the resolvent: (1/N) tr R(z).

        This equals the empirical Stieltjes transform:
        m_N(z) = (1/N) Σ_i 1/(λ_i - z).

        Parameters
        ----------
        z : complex
            Spectral parameter.

        Returns
        -------
        complex
            Normalized trace of the resolvent.
        """
        z = complex(z)
        return np.mean(1.0 / (self.eigenvalues - z))

    def resolvent_identity(self, z1: complex, z2: complex, tol: float = 1e-6) -> dict:
        """Verify the resolvent identity R(z1) - R(z2) = (z1-z2) R(z1) R(z2).

        Checks this at the level of the trace: tests that
        m(z1) - m(z2) ≈ (z1 - z2) * (1/N) tr[R(z1) R(z2)].

        Parameters
        ----------
        z1, z2 : complex
            Two spectral parameters.
        tol : float
            Tolerance for the identity check.

        Returns
        -------
        dict
            {'lhs': complex, 'rhs': complex, 'error': float, 'verified': bool}.
        """
        m1 = self.trace_resolvent(z1)
        m2 = self.trace_resolvent(z2)
        lhs = m1 - m2

        # (1/N) tr[R(z1) R(z2)] = (1/N) Σ_i 1/((λ_i - z1)(λ_i - z2))
        trace_product = np.mean(
            1.0 / ((self.eigenvalues - z1) * (self.eigenvalues - z2))
        )
        rhs = (z1 - z2) * trace_product

        error = abs(lhs - rhs)
        return {
            "lhs": complex(lhs),
            "rhs": complex(rhs),
            "error": float(error),
            "verified": error < tol,
        }

    def spectral_density_from_resolvent(
        self, x_range: np.ndarray, eta: float = 0.01
    ) -> np.ndarray:
        """Compute the empirical spectral density from the resolvent.

        ρ_N(x) = (1/π) Im[m_N(x + iη)].

        Parameters
        ----------
        x_range : np.ndarray
            Real-valued grid.
        eta : float
            Imaginary regularization.

        Returns
        -------
        np.ndarray
            Smoothed spectral density.
        """
        x_range = np.asarray(x_range, dtype=float)
        density = np.zeros_like(x_range)
        for i, x in enumerate(x_range):
            m = self.trace_resolvent(complex(x, eta))
            density[i] = m.imag / np.pi
        return density


class FreeConvolution:
    """Additive free convolution of spectral measures.

    Computes the spectral distribution of A + B when A and B are
    freely independent, using subordination and R-transform methods.
    """

    def __init__(self):
        pass

    def additive_free_convolution(
        self,
        stieltjes1: StieltjesTransform,
        stieltjes2: StieltjesTransform,
        z_range: np.ndarray,
    ) -> np.ndarray:
        """Compute the free additive convolution via R-transforms.

        If μ = μ_A ⊞ μ_B, then R_μ(z) = R_A(z) + R_B(z),
        where R(z) = m⁻¹(z) - 1/z is the R-transform.

        Parameters
        ----------
        stieltjes1, stieltjes2 : StieltjesTransform
            Stieltjes transforms of the two measures.
        z_range : np.ndarray
            Grid of complex points at which to evaluate the result.

        Returns
        -------
        np.ndarray
            Spectral density of the free convolution.
        """
        eta = 0.01
        densities = np.zeros(len(z_range))

        for i, x in enumerate(z_range):
            z = complex(x, eta)
            m1 = stieltjes1.evaluate(z)
            m2 = stieltjes2.evaluate(z)

            # R-transform: R(m) = m^{-1}(m) - 1/m
            # Free convolution Stieltjes transform via fixed-point iteration
            m_conv = self._free_convolution_fixed_point(stieltjes1, stieltjes2, z)
            densities[i] = m_conv.imag / np.pi

        return densities

    def _free_convolution_fixed_point(
        self,
        st1: StieltjesTransform,
        st2: StieltjesTransform,
        z: complex,
        max_iter: int = 200,
        tol: float = 1e-10,
    ) -> complex:
        """Solve for m of the free convolution via subordination iteration.

        The subordination functions w1(z), w2(z) satisfy:
        w1 + w2 = z + 1/m, m = m_1(w1) = m_2(w2).

        Parameters
        ----------
        st1, st2 : StieltjesTransform
            Component Stieltjes transforms.
        z : complex
            Evaluation point.
        max_iter : int
            Maximum iterations.
        tol : float
            Convergence tolerance.

        Returns
        -------
        complex
            Stieltjes transform of the free convolution at z.
        """
        # Initialize with the average
        m = (st1.evaluate(z) + st2.evaluate(z)) / 2

        for _ in range(max_iter):
            if abs(m) < 1e-15:
                m = complex(0, 0.1)

            # Subordination: w_i(z) = 1/(m) - R_i(m) where R is implicit
            w1 = z - st2.inverse_stieltjes(m) + 1.0 / m if abs(m) > 1e-15 else z
            m_new = st1.evaluate(w1)

            if abs(m_new - m) < tol:
                return m_new
            m = m_new

        return m

    def subordination_function(
        self,
        m_A: StieltjesTransform,
        m_B: StieltjesTransform,
        z: complex,
    ) -> tuple:
        """Compute the subordination functions w_A(z), w_B(z).

        These satisfy: m_A(w_A(z)) = m_B(w_B(z)) = m_{A⊞B}(z),
        and w_A(z) + w_B(z) = z + 1/m_{A⊞B}(z).

        Parameters
        ----------
        m_A, m_B : StieltjesTransform
            Stieltjes transforms.
        z : complex
            Evaluation point.

        Returns
        -------
        tuple
            (w_A, w_B, m_conv).
        """
        m_conv = self._free_convolution_fixed_point(m_A, m_B, z)

        if abs(m_conv) < 1e-15:
            return (z, z, m_conv)

        # w_A = F_B(m_conv) where F_B = 1/m_B^{-1}
        w_A = m_B.inverse_stieltjes(m_conv)
        w_B = z + 1.0 / m_conv - w_A

        return (w_A, w_B, m_conv)

    def density_of_sum(
        self,
        eigs_A: np.ndarray,
        eigs_B: np.ndarray,
        x_range: np.ndarray,
        eta: float = 0.05,
    ) -> np.ndarray:
        """Approximate spectral density of A + B (freely independent).

        Uses the empirical Stieltjes transforms of A and B to compute
        the free convolution density.

        Parameters
        ----------
        eigs_A, eigs_B : np.ndarray
            Eigenvalues of the two matrices.
        x_range : np.ndarray
            Grid for density evaluation.
        eta : float
            Regularization.

        Returns
        -------
        np.ndarray
            Density of the free sum.
        """
        eigs_A = np.asarray(eigs_A, dtype=float)
        eigs_B = np.asarray(eigs_B, dtype=float)

        densities = np.zeros(len(x_range))
        for i, x in enumerate(x_range):
            z = complex(x, eta)
            # Empirical Stieltjes transforms
            m_A = np.mean(1.0 / (eigs_A - z))
            m_B = np.mean(1.0 / (eigs_B - z))

            # Approximate via R-transform addition on the empirical measures
            # R(w) = K(w) - 1/w where K = m^{-1}
            # For empirical measures, use direct subordination iteration
            m_conv = (m_A + m_B) / 2  # simple initialization
            for _ in range(100):
                if abs(m_conv) < 1e-15:
                    break
                # Blue function: B(m) = 1/m + R(m)
                # For empirical: approximate R from m
                w = z - 1.0 / m_conv + 1.0 / m_A
                m_new = np.mean(1.0 / (eigs_A - w))
                if abs(m_new - m_conv) < 1e-10:
                    m_conv = m_new
                    break
                m_conv = 0.5 * m_conv + 0.5 * m_new

            densities[i] = m_conv.imag / np.pi

        return densities


class BulkEigenvaluePrediction:
    """Predict bulk eigenvalue statistics from the Marchenko-Pastur law.

    Parameters
    ----------
    gamma : float
        Aspect ratio N/P.
    sigma_sq : float
        Variance scale.
    """

    def __init__(self, gamma: float, sigma_sq: float = 1.0):
        self.gamma = gamma
        self.sigma_sq = sigma_sq
        self.mp = MarchenkoPasturLaw(gamma, sigma_sq)

    def predict_spectrum(self, n_eigenvalues: int) -> np.ndarray:
        """Predict quantile locations of eigenvalues under the MP law.

        Returns the quantiles q_i = F^{-1}((i - 0.5)/n) for i = 1, ..., n.

        Parameters
        ----------
        n_eigenvalues : int
            Number of eigenvalue quantiles to predict.

        Returns
        -------
        np.ndarray
            Predicted eigenvalue locations.
        """
        lam_m, lam_p = self.mp.support()
        # Build a fine grid for CDF inversion
        x_grid = np.linspace(lam_m + 1e-8, lam_p - 1e-8, 2000)
        cdf_grid = self.mp.bulk_eigenvalue_cdf(x_grid)

        p0 = self.mp.atom_at_zero()
        quantiles = (np.arange(1, n_eigenvalues + 1) - 0.5) / n_eigenvalues

        predicted = np.zeros(n_eigenvalues)
        for i, q in enumerate(quantiles):
            if q <= p0:
                predicted[i] = 0.0
            else:
                # Interpolate to find quantile
                idx = np.searchsorted(cdf_grid, q)
                idx = min(idx, len(x_grid) - 1)
                predicted[i] = x_grid[idx]

        return predicted

    def predict_ntk_spectrum(
        self, kernel_matrix: np.ndarray, n_samples: int
    ) -> dict:
        """Predict NTK spectrum and compare to Marchenko-Pastur.

        Parameters
        ----------
        kernel_matrix : np.ndarray
            Empirical NTK matrix (N × N).
        n_samples : int
            Number of data samples (P).

        Returns
        -------
        dict
            Comparison of empirical vs predicted spectra.
        """
        eigenvalues = np.linalg.eigvalsh(kernel_matrix)
        N = kernel_matrix.shape[0]
        gamma_emp = N / n_samples

        mp_fit = MarchenkoPasturLaw(gamma_emp)
        fit_result = mp_fit.fit_to_empirical(eigenvalues)
        ks_result = mp_fit.ks_test(eigenvalues)

        return {
            "empirical_eigenvalues": eigenvalues,
            "fitted_gamma": fit_result["gamma"],
            "fitted_sigma_sq": fit_result["sigma_sq"],
            "predicted_support": mp_fit.support(),
            "ks_test": ks_result,
        }

    def condition_number_prediction(self, gamma: float) -> float:
        """Predict the condition number λ⁺/λ⁻ from the MP law.

        κ = ((1 + √γ)/(1 - √γ))² for γ < 1.

        Parameters
        ----------
        gamma : float
            Aspect ratio.

        Returns
        -------
        float
            Predicted condition number (inf if γ ≥ 1).
        """
        if gamma >= 1.0:
            return float("inf")
        sq = np.sqrt(gamma)
        return ((1 + sq) / (1 - sq)) ** 2

    def spectral_gap_prediction(self, gamma: float, sigma: float = 1.0) -> float:
        """Predict the gap between 0 and λ⁻ in the MP distribution.

        gap = σ²(1 - √γ)² for γ < 1, or 0 for γ ≥ 1.

        Parameters
        ----------
        gamma : float
            Aspect ratio.
        sigma : float
            Standard deviation.

        Returns
        -------
        float
            Spectral gap.
        """
        if gamma >= 1.0:
            return 0.0
        return sigma ** 2 * (1 - np.sqrt(gamma)) ** 2
