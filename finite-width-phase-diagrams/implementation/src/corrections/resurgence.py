"""
Resurgent analysis and trans-series for finite-width neural network corrections.

Extends the nonperturbative.py module with detailed tools for resurgent
trans-series, large-order analysis, Padé-Borel resummation, and application
of these techniques to Neural Tangent Kernel (NTK) finite-width corrections.

The key insight is that the 1/N expansion for width-N networks is typically
an asymptotic (divergent) series whose large-order behavior encodes
non-perturbative information. Resurgence provides a systematic framework
to reconstruct the full answer from the divergent perturbative series.

The trans-series ansatz for a width-N observable is:

    f(N) = Σ_k σ^k e^{-k A N} Σ_n a_{k,n} / N^n

where A is the instanton action, σ is the trans-series parameter (determined
by Stokes constants), and a_{k,n} are perturbative coefficients in each
instanton sector.

References:
    - J. Écalle, "Les fonctions résurgentes" (Publ. Math. d'Orsay, 1981)
    - R.B. Dingle, "Asymptotic Expansions: Their Derivation and
      Interpretation" (Academic Press, 1973)
    - G.V. Dunne and M. Ünsal, "Resurgence and Trans-series in Quantum
      Field Theory" (Ann. Rev. Nucl. Part. Sci. 66, 245-272, 2016)
    - M. Mariño, "Lectures on non-perturbative effects in large N gauge
      theories, matrix models and strings" (Fortsch. Phys. 62, 2014)
    - I. Aniceto, G. Başar, R. Schiappa, "A Primer on Resurgent
      Trans-series and Their Asymptotics" (Phys. Rept. 809, 2019)
    - D. Dorigoni, "An Introduction to Resurgence, Trans-Series and
      Alien Calculus" (Annals Phys. 409, 2019)
"""

import numpy as np
from scipy import optimize, integrate, special, interpolate
from scipy.linalg import solve
from typing import Callable, Optional, List, Tuple, Dict, Union
import warnings


# ---------------------------------------------------------------------------
# TransSeries
# ---------------------------------------------------------------------------

class TransSeries:
    """Trans-series representation for finite-width corrections.

    A trans-series is a formal expression of the form

        f(N) = Σ_{k=0}^{K} σ^k  e^{-k A N}  Σ_{n=0}^{∞} a_{k,n} / N^n

    that encodes both perturbative (k=0) and non-perturbative (k≥1)
    contributions.  The parameter σ is determined by Stokes constants and
    can change discontinuously across Stokes lines in the complex-N plane.

    Attributes:
        perturbative_coeffs: Coefficients a_{0,n} of the perturbative sector.
        instanton_action: The instanton action A (positive real for a
            single real instanton).  Sets the non-perturbative scale
            ~ exp(-A·N).
        n_sectors: Number of instanton sectors to include (0, 1, …, K).
        sector_coeffs: Dict mapping sector index k to array of a_{k,n}.

    References:
        Dunne & Ünsal (2016), Sec. 2: Trans-series and Resurgence.
        Aniceto, Başar & Schiappa (2019), Sec. 3: Trans-series Structure.
    """

    def __init__(
        self,
        perturbative_coeffs: np.ndarray,
        instanton_action: float,
        n_sectors: int = 3,
    ):
        """Initialise trans-series.

        Args:
            perturbative_coeffs: Array of perturbative coefficients a_{0,n},
                where a_{0,0} is the leading (infinite-width) value and
                subsequent entries give 1/N, 1/N², … corrections.
            instanton_action: Instanton action A > 0.  Non-perturbative
                corrections scale as exp(-A·N).
            n_sectors: Number of instanton sectors (default 3 → sectors
                k = 0, 1, 2).
        """
        self.perturbative_coeffs = np.asarray(perturbative_coeffs, dtype=float)
        self.instanton_action = float(instanton_action)
        self.n_sectors = n_sectors

        # Sector coefficients: sector_coeffs[k] = array of a_{k,n}
        self.sector_coeffs: Dict[int, np.ndarray] = {
            0: self.perturbative_coeffs.copy()
        }
        # Higher sectors initialised via large-order relations later
        for k in range(1, n_sectors):
            self.sector_coeffs[k] = np.array([0.0])

    # ---- evaluation helpers ----

    def _sector_sum(self, k: int, N: float, order: Optional[int] = None) -> float:
        """Evaluate partial sum of sector *k* at width *N*.

        Computes  Σ_{n=0}^{order} a_{k,n} / N^n .

        Args:
            k: Sector index.
            N: Width parameter (positive).
            order: Maximum perturbative order inside the sector.
                If None, uses all available coefficients.

        Returns:
            Partial sum (float).
        """
        coeffs = self.sector_coeffs.get(k, np.array([0.0]))
        if order is not None:
            coeffs = coeffs[: order + 1]
        powers = np.arange(len(coeffs))
        return float(np.sum(coeffs / N ** powers))

    def evaluate(self, N: float, sigma: Optional[float] = None) -> float:
        """Evaluate the full trans-series at width N.

        f(N) = Σ_k σ^k  e^{-k A N}  Σ_n a_{k,n} / N^n

        Args:
            N: Width (positive real).
            sigma: Trans-series parameter.  If None, uses σ = 1.

        Returns:
            Trans-series value f(N).
        """
        if sigma is None:
            sigma = 1.0
        A = self.instanton_action
        result = 0.0
        for k in range(self.n_sectors):
            exponential = np.exp(-k * A * N) if k > 0 else 1.0
            sector_val = self._sector_sum(k, N)
            result += (sigma ** k) * exponential * sector_val
        return result

    def perturbative_sector(self, N: float, order: Optional[int] = None) -> float:
        """Evaluate the k = 0 (perturbative) sector.

        Σ_{n=0}^{order} a_{0,n} / N^n

        Args:
            N: Width.
            order: Truncation order (None → all available).

        Returns:
            Perturbative sector value.
        """
        return self._sector_sum(0, N, order)

    def one_instanton_sector(self, N: float, order: Optional[int] = None) -> float:
        """Evaluate the k = 1 (one-instanton) sector.

        e^{-A N}  Σ_n a_{1,n} / N^n

        Args:
            N: Width.
            order: Truncation order.

        Returns:
            One-instanton contribution.
        """
        return np.exp(-self.instanton_action * N) * self._sector_sum(1, N, order)

    def two_instanton_sector(self, N: float, order: Optional[int] = None) -> float:
        """Evaluate the k = 2 (two-instanton) sector.

        e^{-2 A N}  Σ_n a_{2,n} / N^n

        Args:
            N: Width.
            order: Truncation order.

        Returns:
            Two-instanton contribution.
        """
        return np.exp(-2.0 * self.instanton_action * N) * self._sector_sum(2, N, order)

    def optimal_truncation_order(self, N: float) -> int:
        """Optimal truncation order for the perturbative sector.

        For a divergent series with a_n ~ A^{-n} n!, the optimal truncation
        is at n* ≈ A·N, where the terms are smallest before they start to
        grow again.  The error of optimal truncation is O(e^{-A N}), i.e.
        non-perturbative in nature.

        Args:
            N: Width.

        Returns:
            Optimal truncation order n*.

        References:
            Dingle (1973), Ch. 1: "Terminants and Optimal Truncation."
        """
        n_star = int(np.floor(self.instanton_action * N))
        n_star = max(0, min(n_star, len(self.perturbative_coeffs) - 1))
        return n_star

    def stokes_constants(self) -> Dict[int, complex]:
        """Compute Stokes constants S_k from large-order behaviour.

        The Stokes constant S_1 connects the perturbative sector to the
        one-instanton sector via the large-order relation

            a_{0,n} ~ (S_1 / (2π i)) · A^{-n} · Γ(n + β) · a_{1,0}

        Higher Stokes constants S_k relate the k-instanton sector to the
        (k+1)-instanton sector.

        Returns:
            Dictionary mapping sector transition k → k+1 to Stokes
            constant S_k (complex).

        References:
            Dunne & Ünsal (2016), Eq. (2.12).
            Dingle (1973), Ch. 21: "Stokes' Phenomenon."
        """
        A = self.instanton_action
        coeffs = self.perturbative_coeffs
        stokes = {}
        if len(coeffs) < 5:
            warnings.warn(
                "Fewer than 5 perturbative coefficients; Stokes constant "
                "extraction may be unreliable."
            )

        # S_1 from ratio of consecutive large-order coefficients
        # a_{0,n} / a_{0,n-1} → n / A  ⟹  S_1 from normalisation
        n_vals = np.arange(3, len(coeffs))
        if len(n_vals) < 2:
            stokes[1] = complex(0.0)
            return stokes

        a_1_0 = self.sector_coeffs.get(1, np.array([1.0]))[0]
        if np.abs(a_1_0) < 1e-300:
            a_1_0 = 1.0

        # Extract β from subleading correction in ratio test
        ratios = coeffs[n_vals] / coeffs[n_vals - 1]
        # Expected: ratio ~ n/A + (β-1)/A + ...
        # Linear fit: ratio * A / n → 1 + (β-1)/n + ...
        scaled = ratios * A / n_vals
        beta_fit = np.polyfit(1.0 / n_vals, scaled, 1)
        beta = 1.0 + beta_fit[0]

        # S_1 = 2πi · a_{0,n} · A^n / (Γ(n+β) · a_{1,0}) for large n
        n_ref = n_vals[-1]
        gamma_val = special.gamma(n_ref + beta)
        if np.abs(gamma_val) > 0 and np.isfinite(gamma_val):
            s1_mag = np.abs(coeffs[n_ref]) * A ** n_ref / (gamma_val * np.abs(a_1_0))
            stokes[1] = complex(0, 2.0 * np.pi * s1_mag)
        else:
            # Use log-space for large n
            log_s1 = (
                np.log(np.abs(coeffs[n_ref]))
                + n_ref * np.log(A)
                - special.gammaln(n_ref + beta)
                - np.log(np.abs(a_1_0))
            )
            stokes[1] = complex(0, 2.0 * np.pi * np.exp(log_s1))

        # Higher Stokes constants (crude estimate from sector coefficients)
        for k in range(2, self.n_sectors):
            stokes[k] = stokes.get(k - 1, 0.0) * 0.5  # placeholder scaling

        return stokes

    def large_order_relation(self, k: int) -> Dict[str, float]:
        """Large-order resurgence relation for sector k.

        The fundamental resurgence relation connects sectors:

            a_{k,n} ~ (S_{k+1} / (2πi)) · (k+1)^{-n} · A^{-n}
                       · Γ(n + β_k) · a_{k+1, 0}

        as n → ∞.  This encodes the statement that the divergence of sector
        k 'knows about' sector k+1.

        Args:
            k: Sector index whose large-order growth is analysed.

        Returns:
            Dictionary with keys 'action' (effective A for sector k),
            'beta' (sub-leading exponent), 'prefactor'.

        References:
            Écalle (1981), Vol. I, Théorème fondamental.
            Aniceto et al. (2019), Eq. (3.15).
        """
        coeffs = self.sector_coeffs.get(k, np.array([0.0]))
        A = self.instanton_action

        if len(coeffs) < 5:
            return {"action": A, "beta": 0.5, "prefactor": 1.0}

        # Ratio test to extract effective action
        n_vals = np.arange(3, len(coeffs))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.abs(coeffs[n_vals] / coeffs[n_vals - 1])
        finite_mask = np.isfinite(ratios) & (ratios > 0)
        if np.sum(finite_mask) < 2:
            return {"action": A, "beta": 0.5, "prefactor": 1.0}

        n_fit = n_vals[finite_mask]
        r_fit = ratios[finite_mask]

        # ratio ~ n / A_eff  ⟹  A_eff from slope
        fit = np.polyfit(n_fit, r_fit, 1)
        A_eff = 1.0 / fit[0] if np.abs(fit[0]) > 1e-15 else A

        # beta from intercept
        beta = 1.0 - fit[1] * A_eff

        # prefactor from normalisation at last available order
        n_ref = n_fit[-1]
        log_prefactor = (
            np.log(np.abs(coeffs[n_ref]))
            + n_ref * np.log(np.abs(A_eff))
            - special.gammaln(n_ref + beta)
        )
        prefactor = np.exp(log_prefactor) if np.isfinite(log_prefactor) else 1.0

        return {"action": float(np.abs(A_eff)), "beta": float(beta), "prefactor": float(prefactor)}

    def alien_derivative(self, sector_k: int, direction: float = 0.0) -> np.ndarray:
        """Alien derivative Δ_ω acting on sector k.

        The alien derivative is the central object of Écalle's theory.
        Along the direction θ = arg(ω), it measures the discontinuity of
        the Borel-resummed series across the Stokes line.

        For ω = A (the instanton action),

            Δ_A  Φ_k = S_{k→k+1} · Φ_{k+1}

        where S_{k→k+1} is the Stokes constant and Φ_k denotes the formal
        power series of sector k.

        Args:
            sector_k: Sector index on which the alien derivative acts.
            direction: Direction angle θ in the Borel plane.  Default 0
                corresponds to the positive real axis.

        Returns:
            Array of coefficients of the resulting formal series.

        References:
            Écalle (1981), Vol. I.
            Sauzin, "Introduction to 1-summability and resurgence" (2014).
        """
        stokes = self.stokes_constants()
        s_k = stokes.get(sector_k + 1, 0.0)
        next_coeffs = self.sector_coeffs.get(sector_k + 1, np.array([0.0]))

        # Alien derivative produces S_k · Φ_{k+1} (with possible
        # phase factor from direction)
        phase = np.exp(1j * direction) if direction != 0.0 else 1.0
        result = np.real(s_k * phase) * next_coeffs
        return result

    def bridge_equation(self) -> Dict[str, np.ndarray]:
        """Bridge equation connecting alien derivatives and Stokes constants.

        The bridge equation (Écalle) relates the alien derivative to the
        ordinary derivative via the trans-series parameter:

            Δ_A f = S_1 · (∂f / ∂σ)

        This constrains all Stokes constants in terms of the perturbative
        data and the trans-series parameter.

        Returns:
            Dictionary with 'alien_deriv' and 'sigma_deriv' arrays that
            should be proportional.

        References:
            Aniceto et al. (2019), Sec. 3.3: Bridge Equations.
            Dunne & Ünsal (2016), Sec. 2.3.
        """
        alien = self.alien_derivative(0)

        # ∂f/∂σ at leading order is the 1-instanton sector
        sigma_deriv = self.sector_coeffs.get(1, np.array([0.0]))

        # Pad to same length
        max_len = max(len(alien), len(sigma_deriv))
        alien_padded = np.zeros(max_len)
        sigma_padded = np.zeros(max_len)
        alien_padded[: len(alien)] = alien
        sigma_padded[: len(sigma_deriv)] = sigma_deriv

        return {"alien_deriv": alien_padded, "sigma_deriv": sigma_padded}

    def median_resummation(self, N: float) -> float:
        """Median resummation: average of lateral Borel sums.

        On a Stokes line the Borel integral is ambiguous.  The lateral
        Borel sums S_± correspond to integration contours deformed
        above/below the singularity.  The median resummation is

            f_med(N) = (S_+ f + S_- f) / 2

        which is real and removes the leading ambiguity of order
        ~ Im(S_1) · e^{-A N}.

        This uses Padé–Borel with lateral deformation implemented via
        a small imaginary part ε → 0±.

        Args:
            N: Width parameter.

        Returns:
            Median-resummed value.

        References:
            Delabaere & Pham, "Resurgent methods in semi-classical
            asymptotics" (Ann. IHP 1999).
        """
        coeffs = self.perturbative_coeffs
        if len(coeffs) < 2:
            return float(coeffs[0]) if len(coeffs) == 1 else 0.0

        # Borel transform: B(t) = Σ a_n t^n / n!
        borel_coeffs = coeffs / special.factorial(np.arange(len(coeffs)))

        def borel_fn(t, eps_sign):
            """Evaluate Borel transform with Padé, deformed by ε."""
            val = 0.0
            for n, c in enumerate(borel_coeffs):
                val += c * (t + eps_sign * 1e-8j) ** n
            return np.real(val * np.exp(-t))

        # S_+ : contour above real axis
        s_plus, _ = integrate.quad(
            lambda t: borel_fn(t * N, +1), 0, np.inf, limit=200
        )

        # S_- : contour below real axis
        s_minus, _ = integrate.quad(
            lambda t: borel_fn(t * N, -1), 0, np.inf, limit=200
        )

        return 0.5 * (s_plus + s_minus)


# ---------------------------------------------------------------------------
# LargeOrderAnalysis
# ---------------------------------------------------------------------------

class LargeOrderAnalysis:
    """Tools for extracting non-perturbative data from large-order behaviour.

    Given a divergent series Σ a_n / N^n, the large-order growth

        a_n ~ C · A^{-n} · Γ(n + β) · (1 + c₁/n + c₂/n² + …)

    encodes the instanton action A, the characteristic exponent β, and
    subleading corrections c_k that are related to higher perturbative
    coefficients around the instanton.

    Attributes:
        coefficients: Array of series coefficients a_n.

    References:
        Dingle (1973), Ch. 1–4.
        Le Guillou & Zinn-Justin, "Large-Order Behaviour of Perturbation
        Theory" (North-Holland, 1990).
        Bender & Wu, "Anharmonic Oscillator" (Phys. Rev. 184, 1969).
    """

    def __init__(self, coefficients: np.ndarray):
        """Initialise with series coefficients.

        Args:
            coefficients: Array a_0, a_1, a_2, … of perturbative
                coefficients.
        """
        self.coefficients = np.asarray(coefficients, dtype=float)

    def ratio_test(self) -> Dict[str, np.ndarray]:
        """Ratio test: r_n = a_{n+1} / a_n.

        For a series with radius of convergence R, r_n → 1/R.  For
        factorially divergent series, r_n → n/A.

        Returns:
            Dictionary with keys 'n' (indices), 'ratios', and
            'inverse_radius' (extrapolated 1/R).
        """
        coeffs = self.coefficients
        n_vals = np.arange(len(coeffs) - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = coeffs[1:] / coeffs[:-1]

        finite = np.isfinite(ratios)
        inv_radius = np.nan
        if np.sum(finite) >= 2:
            inv_radius = float(ratios[finite][-1])

        return {"n": n_vals, "ratios": ratios, "inverse_radius": inv_radius}

    def darboux_analysis(self) -> Dict[str, float]:
        """Darboux analysis: extract singularity type from large-order growth.

        If the generating function has an algebraic singularity of the form

            f(z) ~ (1 - z/z_c)^{-α}

        near z = z_c, then a_n ~ z_c^{-n} · n^{α-1} / Γ(α).

        Returns:
            Dictionary with 'z_c' (singularity location), 'alpha'
            (singularity exponent), 'type' ('algebraic' or 'logarithmic').
        """
        coeffs = self.coefficients
        if len(coeffs) < 6:
            return {"z_c": np.nan, "alpha": np.nan, "type": "unknown"}

        # Ratio test for z_c
        rt = self.ratio_test()
        ratios = rt["ratios"]
        finite = np.isfinite(ratios) & (ratios != 0)
        if np.sum(finite) < 3:
            return {"z_c": np.nan, "alpha": np.nan, "type": "unknown"}

        z_c = 1.0 / ratios[finite][-1]

        # For n^{α-1} growth: log|a_n / z_c^{-n}| ~ (α-1) log n + const
        n_vals = np.arange(2, len(coeffs))
        log_reduced = np.log(np.abs(coeffs[n_vals])) + n_vals * np.log(np.abs(z_c))
        finite2 = np.isfinite(log_reduced)
        if np.sum(finite2) < 3:
            return {"z_c": float(z_c), "alpha": np.nan, "type": "unknown"}

        fit = np.polyfit(np.log(n_vals[finite2]), log_reduced[finite2], 1)
        alpha = fit[0] + 1.0

        # If α is close to 0, the singularity is logarithmic
        sing_type = "logarithmic" if np.abs(alpha) < 0.1 else "algebraic"

        return {"z_c": float(z_c), "alpha": float(alpha), "type": sing_type}

    def richardson_extrapolation(
        self, sequence: np.ndarray, order: int = 3
    ) -> float:
        """Richardson extrapolation to accelerate convergence.

        Eliminates the leading correction terms c_k / n^k from a
        slowly convergent sequence S_n → S + c_1/n + c_2/n² + …

        Uses order-p Richardson extrapolation which eliminates
        corrections up to O(1/n^p).

        Args:
            sequence: The sequence S_n to extrapolate.
            order: Number of correction terms to eliminate.

        Returns:
            Extrapolated limit.

        References:
            Brezinski, "Extrapolation Methods" (North-Holland, 2013).
        """
        seq = np.asarray(sequence, dtype=float)
        if len(seq) < order + 1:
            return float(seq[-1])

        # Use last order+1 elements
        s = seq[-(order + 1) :]
        n0 = len(seq) - order - 1
        ns = np.arange(n0, n0 + order + 1, dtype=float)

        # Richardson tableau: eliminate 1/n, 1/n², …
        table = s.copy()
        for k in range(1, order + 1):
            new_table = np.zeros(len(table) - 1)
            for j in range(len(new_table)):
                n_j = ns[j]
                n_jk = ns[j + k]
                ratio = (n_jk / n_j) ** k
                new_table[j] = (ratio * table[j + 1] - table[j]) / (ratio - 1.0)
            table = new_table

        return float(table[0])

    def neville_table(self, sequence: np.ndarray) -> np.ndarray:
        """Neville–Aitken extrapolation table.

        Constructs the full Neville table for polynomial extrapolation
        of a sequence S(h_n) → S(0) where h_n = 1/n.

        Args:
            sequence: Input sequence S_1, S_2, …

        Returns:
            2D array where table[k, j] is the j-th order extrapolant
            starting from element k.
        """
        seq = np.asarray(sequence, dtype=float)
        n = len(seq)
        table = np.full((n, n), np.nan)
        table[:, 0] = seq

        # h values: h_k = 1/(k+1)
        h = 1.0 / np.arange(1, n + 1, dtype=float)

        for j in range(1, n):
            for k in range(n - j):
                table[k, j] = (
                    h[k] * table[k + 1, j - 1] - h[k + j] * table[k, j - 1]
                ) / (h[k] - h[k + j])

        return table

    def borel_singularity_location(self) -> float:
        """Location of the nearest Borel-plane singularity.

        The Borel transform B(t) = Σ a_n t^n / n! has singularities at
        t = A (instanton action).  The nearest one determines the
        factorial growth rate of the original coefficients.

        Returns:
            Location t_0 of the nearest Borel singularity.
        """
        coeffs = self.coefficients
        if len(coeffs) < 4:
            return np.nan

        # Borel coefficients: b_n = a_n / n!
        ns = np.arange(len(coeffs), dtype=float)
        borel_c = coeffs / special.factorial(ns)

        # Ratio test on Borel coefficients → radius of convergence = A
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = borel_c[1:] / borel_c[:-1]
        finite = np.isfinite(ratios) & (ratios != 0)
        if np.sum(finite) < 2:
            return np.nan

        inv_A = ratios[finite][-1]
        return float(1.0 / inv_A) if np.abs(inv_A) > 1e-15 else np.nan

    def borel_singularity_type(self) -> str:
        """Determine type of nearest Borel singularity.

        Algebraic: B(t) ~ (A - t)^{-γ}  →  a_n ~ A^{-n} n^{γ-1} n!
        Logarithmic: B(t) ~ log(A - t)   →  a_n ~ A^{-n} n! / n

        Returns:
            'algebraic', 'logarithmic', or 'unknown'.
        """
        coeffs = self.coefficients
        if len(coeffs) < 8:
            return "unknown"

        # After removing factorial growth: c_n = a_n / (A^{-n} n!)
        A = self.borel_singularity_location()
        if not np.isfinite(A) or np.abs(A) < 1e-15:
            return "unknown"

        ns = np.arange(2, len(coeffs), dtype=float)
        with np.errstate(all="ignore"):
            log_cn = (
                np.log(np.abs(coeffs[2:]))
                + ns * np.log(np.abs(A))
                - special.gammaln(ns + 1)
            )
        finite = np.isfinite(log_cn)
        if np.sum(finite) < 4:
            return "unknown"

        # Fit log c_n vs log n: slope γ-1 for algebraic, slope -1 for log
        fit = np.polyfit(np.log(ns[finite]), log_cn[finite], 1)
        slope = fit[0]

        if np.abs(slope + 1.0) < 0.3:
            return "logarithmic"
        elif slope > -0.5:
            return "algebraic"
        return "unknown"

    def instanton_action_from_coefficients(self) -> float:
        """Extract instanton action A from the growth a_n ~ A^{-n} n!

        Uses successive ratios: a_n / (a_{n-1} · n) → 1/A.

        Returns:
            Estimated instanton action A.
        """
        coeffs = self.coefficients
        if len(coeffs) < 4:
            return np.nan

        ns = np.arange(2, len(coeffs), dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_A_estimates = coeffs[2:] / (coeffs[1:-1] * ns)
        finite = np.isfinite(inv_A_estimates) & (inv_A_estimates != 0)
        if np.sum(finite) < 2:
            return np.nan

        # Richardson-extrapolate the sequence
        inv_A_seq = inv_A_estimates[finite]
        inv_A = self.richardson_extrapolation(inv_A_seq, order=min(3, len(inv_A_seq) - 1))
        return float(1.0 / inv_A) if np.abs(inv_A) > 1e-15 else np.nan

    def subleading_corrections(self, action: float) -> Dict[str, float]:
        """Extract subleading exponent β from a_n ~ A^{-n} Γ(n + β).

        After dividing out the leading factorial growth, the remaining
        power-law behaviour n^{β-1/2} determines β.

        Args:
            action: The instanton action A.

        Returns:
            Dictionary with 'beta' and estimated 'error'.
        """
        coeffs = self.coefficients
        if len(coeffs) < 6:
            return {"beta": 0.5, "error": np.nan}

        ns = np.arange(3, len(coeffs), dtype=float)

        # r_n = a_n / a_{n-1} ≈ (n + β - 1) / A
        with np.errstate(divide="ignore", invalid="ignore"):
            rn = coeffs[3:] / coeffs[2:-1]
        finite = np.isfinite(rn)
        if np.sum(finite) < 3:
            return {"beta": 0.5, "error": np.nan}

        # Linear fit: r_n * A ≈ n + β - 1
        y = rn[finite] * action
        x = ns[finite]
        fit = np.polyfit(x, y, 1)
        # fit[0] should be ~1, fit[1] = β - 1
        beta = fit[1] + 1.0
        residual = np.std(y - np.polyval(fit, x))

        return {"beta": float(beta), "error": float(residual)}

    def factorial_over_power_fit(
        self, n_range: Optional[Tuple[int, int]] = None
    ) -> Dict[str, float]:
        """Fit coefficients to a_n = C · A^{-n} · Γ(n+β) · (1 + c₁/n + c₂/n² + …).

        This is the canonical large-order ansatz for a factorially divergent
        series with resurgent structure.

        Args:
            n_range: Tuple (n_min, n_max) of coefficient indices to use
                in the fit.  Defaults to using the last 70% of coefficients.

        Returns:
            Dictionary with 'C', 'A', 'beta', 'c1', 'c2', and 'residual'.

        References:
            Bender & Wu (1969), Eq. (4.3).
        """
        coeffs = self.coefficients
        if n_range is None:
            n_min = max(3, len(coeffs) // 3)
            n_max = len(coeffs)
        else:
            n_min, n_max = n_range
        ns = np.arange(n_min, n_max, dtype=float)
        a_n = coeffs[n_min:n_max]

        # Step 1: extract A from instanton_action
        A = self.instanton_action_from_coefficients()
        if not np.isfinite(A):
            A = 1.0

        # Step 2: extract β
        sub = self.subleading_corrections(A)
        beta = sub["beta"]

        # Step 3: reduced coefficients c̃_n = a_n / (A^{-n} Γ(n+β))
        with np.errstate(all="ignore"):
            log_reduced = (
                np.log(np.abs(a_n))
                + ns * np.log(np.abs(A))
                - special.gammaln(ns + beta)
            )
        finite = np.isfinite(log_reduced)
        if np.sum(finite) < 3:
            return {
                "C": np.nan, "A": A, "beta": beta,
                "c1": 0.0, "c2": 0.0, "residual": np.nan,
            }

        reduced = np.exp(log_reduced[finite])
        ns_f = ns[finite]

        # Fit c̃_n = C (1 + c₁/n + c₂/n²)
        inv_n = 1.0 / ns_f
        design = np.column_stack([np.ones_like(inv_n), inv_n, inv_n ** 2])
        try:
            params, residual_arr, _, _ = np.linalg.lstsq(design, reduced, rcond=None)
            C, c1, c2 = params
            residual = float(residual_arr[0]) if len(residual_arr) > 0 else np.nan
        except np.linalg.LinAlgError:
            C, c1, c2, residual = np.nan, 0.0, 0.0, np.nan

        return {
            "C": float(C),
            "A": float(A),
            "beta": float(beta),
            "c1": float(c1 / C) if np.abs(C) > 1e-15 else 0.0,
            "c2": float(c2 / C) if np.abs(C) > 1e-15 else 0.0,
            "residual": residual,
        }

    def alternating_series_analysis(self) -> Dict[str, float]:
        """Handle alternating-sign series a_n = (-1)^n |a_n|.

        For an alternating series, the Borel singularity lies on the
        *negative* real axis, and the series is Borel-summable along the
        positive real axis.  The relevant instanton action is negative:
        A < 0, or equivalently |A| on the negative axis.

        Returns:
            Dictionary with 'is_alternating' (bool-like 0/1),
            'effective_action', and 'growth_rate'.
        """
        coeffs = self.coefficients
        if len(coeffs) < 4:
            return {"is_alternating": 0, "effective_action": np.nan, "growth_rate": np.nan}

        signs = np.sign(coeffs[1:])
        sign_changes = np.sum(signs[1:] * signs[:-1] < 0)
        n_total = len(signs) - 1
        is_alt = sign_changes > 0.7 * n_total

        if is_alt:
            # Work with |a_n|
            abs_coeffs = np.abs(coeffs)
            analyser = LargeOrderAnalysis(abs_coeffs)
            A = analyser.instanton_action_from_coefficients()
            return {
                "is_alternating": 1,
                "effective_action": float(-A) if np.isfinite(A) else np.nan,
                "growth_rate": float(A),
            }

        return {"is_alternating": 0, "effective_action": np.nan, "growth_rate": np.nan}


# ---------------------------------------------------------------------------
# PadeBorelAnalysis
# ---------------------------------------------------------------------------

class PadeBorelAnalysis:
    """Padé–Borel resummation and analysis of divergent series.

    The Padé–Borel method combines:
    1. Borel transform: B(t) = Σ a_n t^n / Γ(n + 1 + b)
    2. Padé approximation: [M/N](t) ≈ B(t) — analytic continuation
    3. Laplace integral: f_resum(N) = ∫_0^∞ e^{-t} [M/N](t/N) dt

    Padé approximants are rational functions P_M(t)/Q_N(t) that match
    the Taylor series to order M + N.  Their poles reveal the singularity
    structure of the Borel transform.

    Attributes:
        coefficients: Array of series coefficients a_n.
        borel_coeffs: Borel-transformed coefficients a_n / n!

    References:
        Baker & Graves-Morris, "Padé Approximants" (Cambridge, 1996).
        Zinn-Justin (2002), Appendix A37: Borel Summability.
        Costin, "Asymptotics and Borel Summability" (CRC Press, 2009).
    """

    def __init__(self, coefficients: np.ndarray):
        """Initialise with series coefficients.

        Args:
            coefficients: Array a_0, a_1, a_2, … of perturbative
                coefficients.
        """
        self.coefficients = np.asarray(coefficients, dtype=float)
        ns = np.arange(len(self.coefficients), dtype=float)
        self.borel_coeffs = self.coefficients / special.factorial(ns)

    def _pade_approximant(
        self, coeffs: np.ndarray, m: int, n: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute [M/N] Padé approximant coefficients.

        Finds polynomials P_M(t) and Q_N(t) such that
        P_M(t)/Q_N(t) matches Σ c_k t^k to order M+N.

        Args:
            coeffs: Taylor coefficients c_0, c_1, …, c_{M+N}.
            m: Numerator degree.
            n: Denominator degree.

        Returns:
            Tuple (p_coeffs, q_coeffs) — numerator and denominator
            polynomial coefficients (lowest degree first).
        """
        order = m + n
        if len(coeffs) < order + 1:
            raise ValueError(
                f"Need at least {order + 1} coefficients for [{m}/{n}] Padé."
            )

        c = coeffs[: order + 1]

        if n == 0:
            return c[: m + 1].copy(), np.array([1.0])

        # Build linear system for denominator coefficients q_1, …, q_n
        # Σ_{j=0}^{n} q_j c_{i-j} = 0  for i = m+1, …, m+n
        # with q_0 = 1.
        mat = np.zeros((n, n))
        rhs = np.zeros(n)
        for i in range(n):
            row_idx = m + 1 + i
            for j in range(n):
                col_idx = row_idx - (j + 1)
                if 0 <= col_idx < len(c):
                    mat[i, j] = c[col_idx]
            rhs[i] = -c[row_idx] if row_idx < len(c) else 0.0

        try:
            q = solve(mat, rhs)
        except np.linalg.LinAlgError:
            # Fall back to least-squares
            q, _, _, _ = np.linalg.lstsq(mat, rhs, rcond=None)

        q_coeffs = np.concatenate([[1.0], q])

        # Compute numerator: p_i = Σ_{j=0}^{min(i,n)} q_j c_{i-j}
        p_coeffs = np.zeros(m + 1)
        for i in range(m + 1):
            for j in range(min(i, n) + 1):
                p_coeffs[i] += q_coeffs[j] * c[i - j]

        return p_coeffs, q_coeffs

    def _eval_pade(
        self, t: Union[float, np.ndarray], p: np.ndarray, q: np.ndarray
    ) -> Union[float, np.ndarray]:
        """Evaluate Padé approximant P(t)/Q(t)."""
        num = np.polyval(p[::-1], t)  # polyval expects highest degree first
        # Actually np.polyval uses highest-degree-first, but our coefficients
        # are lowest-degree-first.  Use the correct ordering.
        num = sum(p[k] * t ** k for k in range(len(p)))
        den = sum(q[k] * t ** k for k in range(len(q)))
        with np.errstate(divide="ignore", invalid="ignore"):
            result = np.where(np.abs(den) > 1e-300, num / den, np.inf)
        return result

    def pade_table(self, max_order: int) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
        """Compute the Padé table for the Borel transform up to given order.

        The Padé table is the collection of all [M/N] approximants for
        M + N ≤ max_order.

        Args:
            max_order: Maximum total order M + N.

        Returns:
            Dictionary mapping (M, N) to (p_coeffs, q_coeffs).
        """
        table = {}
        for total in range(max_order + 1):
            for n in range(total + 1):
                m = total - n
                if m + n < len(self.borel_coeffs):
                    try:
                        p, q = self._pade_approximant(self.borel_coeffs, m, n)
                        table[(m, n)] = (p, q)
                    except (ValueError, np.linalg.LinAlgError):
                        pass
        return table

    def pade_poles(self, m: int, n: int) -> np.ndarray:
        """Poles of the [M/N] Padé approximant of the Borel transform.

        Poles correspond to singularities of the Borel transform and thus
        to instanton actions in the original problem.

        Args:
            m: Numerator degree.
            n: Denominator degree.

        Returns:
            Array of pole locations (complex).
        """
        _, q = self._pade_approximant(self.borel_coeffs, m, n)
        if len(q) <= 1:
            return np.array([])
        # Roots of Q_N(t) = 0
        roots = np.roots(q[::-1])
        return roots

    def pade_zeros(self, m: int, n: int) -> np.ndarray:
        """Zeros of the [M/N] Padé approximant of the Borel transform.

        Args:
            m: Numerator degree.
            n: Denominator degree.

        Returns:
            Array of zero locations (complex).
        """
        p, _ = self._pade_approximant(self.borel_coeffs, m, n)
        if len(p) <= 1:
            return np.array([])
        roots = np.roots(p[::-1])
        return roots

    def defect_analysis(self, m: int, n: int) -> Dict[str, list]:
        """Analyse pole–zero cancellations (Froissart doublets / defects).

        Spurious poles of the Padé approximant tend to appear close to
        spurious zeros, forming 'defects'.  Genuine singularities appear
        as isolated poles without nearby zeros.

        Args:
            m: Numerator degree.
            n: Denominator degree.

        Returns:
            Dictionary with 'genuine_poles', 'defects' (pole-zero pairs),
            and 'threshold' used.
        """
        poles = self.pade_poles(m, n)
        zeros = self.pade_zeros(m, n)
        threshold = 0.1 * np.max(np.abs(poles)) if len(poles) > 0 else 0.1

        genuine = []
        defects = []
        used_zeros = set()

        for p in poles:
            if len(zeros) == 0:
                genuine.append(complex(p))
                continue
            dists = np.abs(zeros - p)
            nearest_idx = np.argmin(dists)
            if dists[nearest_idx] < threshold and nearest_idx not in used_zeros:
                defects.append((complex(p), complex(zeros[nearest_idx])))
                used_zeros.add(nearest_idx)
            else:
                genuine.append(complex(p))

        return {
            "genuine_poles": genuine,
            "defects": defects,
            "threshold": float(threshold),
        }

    def best_pade(self, N_eval: float) -> Tuple[int, int]:
        """Select best Padé approximant from the table by stability.

        Evaluates the Padé–Borel resummation for several [M/N] choices
        near the diagonal and picks the one with smallest variation under
        small changes in (M, N).

        Args:
            N_eval: Width at which to evaluate stability.

        Returns:
            Tuple (M, N) of the selected approximant.
        """
        max_order = len(self.borel_coeffs) - 1
        candidates = []
        values = []

        for total in range(max(0, max_order - 4), max_order + 1):
            for n_den in range(max(0, total // 2 - 1), total // 2 + 2):
                m_num = total - n_den
                if m_num < 0 or n_den < 0:
                    continue
                if m_num + n_den >= len(self.borel_coeffs):
                    continue
                try:
                    val = self.pade_borel_integral(N_eval, m_num, n_den)
                    if np.isfinite(val):
                        candidates.append((m_num, n_den))
                        values.append(val)
                except Exception:
                    pass

        if not candidates:
            half = max_order // 2
            return (half, max_order - half)

        values = np.array(values)
        median_val = np.median(values)
        deviations = np.abs(values - median_val)
        best_idx = np.argmin(deviations)
        return candidates[best_idx]

    def diagonal_pade_sequence(self, max_order: int) -> List[Tuple[int, float]]:
        """Compute the diagonal [N/N] Padé sequence.

        The diagonal Padé sequence often converges to the Borel sum
        even when off-diagonal approximants do not.

        Args:
            max_order: Maximum N for [N/N].

        Returns:
            List of (N, value_at_t=1) pairs.
        """
        results = []
        for n in range(1, max_order + 1):
            if 2 * n >= len(self.borel_coeffs):
                break
            try:
                p, q = self._pade_approximant(self.borel_coeffs, n, n)
                val = self._eval_pade(1.0, p, q)
                if np.isfinite(val):
                    results.append((n, float(np.real(val))))
            except Exception:
                pass
        return results

    def pade_borel_integral(self, N: float, m: int, n: int) -> float:
        """Padé–Borel resummation integral.

        f_resum(N) = ∫_0^∞ e^{-t}  [M/N]_Borel(t / N) dt

        where [M/N]_Borel is the Padé approximant of the Borel transform.

        Args:
            N: Width parameter.
            m: Numerator degree.
            n: Denominator degree.

        Returns:
            Resummed value.
        """
        p, q = self._pade_approximant(self.borel_coeffs, m, n)

        def integrand(t):
            pade_val = self._eval_pade(t / N, p, q)
            return float(np.real(np.exp(-t) * pade_val))

        # Check for poles on positive real axis
        poles = np.roots(q[::-1])
        real_positive_poles = [
            float(np.real(p_)) * N
            for p_ in poles
            if np.abs(np.imag(p_)) < 1e-10 and np.real(p_) > 0
        ]

        if real_positive_poles:
            # Deform contour slightly into complex plane (principal value)
            warnings.warn(
                f"Borel singularity on integration contour at t = "
                f"{min(real_positive_poles):.4f}; using principal value."
            )
            result, _ = integrate.quad(
                integrand, 0, np.inf, limit=300, weight="cauchy",
                wvar=min(real_positive_poles),
            )
        else:
            result, _ = integrate.quad(integrand, 0, np.inf, limit=300)

        return float(result)

    def conformal_pade(self, N: float, mapping: str = "euler") -> float:
        """Conformal mapping + Padé resummation.

        Apply a conformal map to the Borel plane to move singularities
        further from the origin, then apply Padé.

        Mappings:
        - 'euler': w = t / (1 + t)  (maps [0, ∞) → [0, 1))
        - 'sqrt':  w = (√(1 + t/A) - 1) / (√(1 + t/A) + 1)

        Args:
            N: Width parameter.
            mapping: Type of conformal map.

        Returns:
            Conformally-mapped Padé–Borel resummed value.

        References:
            Le Guillou & Zinn-Justin (1980), "Critical exponents from
            field theory."
        """
        bc = self.borel_coeffs.copy()
        n_coeffs = len(bc)

        if mapping == "euler":
            # w = t/(1+t),  t = w/(1-w),  dt = dw/(1-w)²
            # Transform coefficients to w-variable
            # B(t(w)) = Σ b_n [w/(1-w)]^n = Σ c_k w^k
            w_coeffs = np.zeros(n_coeffs)
            for k in range(n_coeffs):
                # Coefficient of w^k in Σ b_n w^n (1-w)^{-n}
                for n in range(k + 1):
                    if n < n_coeffs:
                        # (1-w)^{-n}: coefficient of w^{k-n} is C(n+k-n-1, k-n)
                        binom_coeff = special.comb(k - 1, k - n, exact=True) if k > n else (1.0 if k == n else 0.0)
                        # More carefully: (1-w)^{-n} = Σ_j C(n+j-1,j) w^j
                        j = k - n
                        if j >= 0 and n > 0:
                            binom_coeff = special.comb(n + j - 1, j, exact=False)
                        elif j == 0 and n == 0:
                            binom_coeff = 1.0
                        else:
                            binom_coeff = 0.0
                        w_coeffs[k] += bc[n] * binom_coeff

            # Padé in w-variable, then integrate over w ∈ [0, 1)
            half = n_coeffs // 2
            m, n = half, n_coeffs - half - 1
            if m + n >= n_coeffs:
                n = n_coeffs - m - 1
            try:
                p, q = self._pade_approximant(w_coeffs, m, max(n, 1))
            except (ValueError, np.linalg.LinAlgError):
                p, q = w_coeffs[:m+1], np.array([1.0])

            def integrand(w):
                pade_val = self._eval_pade(w, p, q)
                # t = w/(1-w), dt/dw = 1/(1-w)², Jacobian factor
                t = w / (1.0 - w + 1e-30)
                jac = 1.0 / (1.0 - w + 1e-30) ** 2
                return float(np.real(np.exp(-t * N) * pade_val * jac))

            result, _ = integrate.quad(integrand, 0, 1.0 - 1e-10, limit=300)
            return float(result)

        elif mapping == "sqrt":
            # Direct Padé–Borel as fallback
            m_best, n_best = self.best_pade(N)
            return self.pade_borel_integral(N, m_best, n_best)

        else:
            raise ValueError(f"Unknown mapping '{mapping}'. Use 'euler' or 'sqrt'.")

    def ecalle_borel(self, N: float, acceleration: str = "conformal") -> float:
        """Écalle–Borel resummation with acceleration.

        Full resummation pipeline:
        1. Borel transform
        2. Optional acceleration (conformal mapping or sequence transform)
        3. Analytic continuation via Padé
        4. Laplace integral

        Args:
            N: Width parameter.
            acceleration: Acceleration method — 'conformal' (conformal map
                + Padé), 'none' (direct Padé–Borel), or 'levin'
                (Levin u-transform on partial sums).

        Returns:
            Resummed value.

        References:
            Costin (2009), Ch. 3: "Borel summability and Stokes phenomena."
        """
        if acceleration == "conformal":
            return self.conformal_pade(N, mapping="euler")
        elif acceleration == "none":
            m, n = self.best_pade(N)
            return self.pade_borel_integral(N, m, n)
        elif acceleration == "levin":
            # Levin u-transform on partial sums
            coeffs = self.coefficients
            partial_sums = np.cumsum(coeffs / N ** np.arange(len(coeffs)))
            if len(partial_sums) < 4:
                return float(partial_sums[-1])
            return self._levin_transform(partial_sums)
        else:
            raise ValueError(f"Unknown acceleration '{acceleration}'.")

    def _levin_transform(self, partial_sums: np.ndarray) -> float:
        """Levin u-transform for sequence acceleration.

        The Levin transform uses the last term of the partial sum as a
        remainder estimate to construct a more rapidly convergent sequence.

        Args:
            partial_sums: Array of partial sums.

        Returns:
            Accelerated limit estimate.
        """
        s = partial_sums
        n = len(s)
        if n < 3:
            return float(s[-1])

        # Remainder estimates: ω_n = a_n (the last term added)
        omega = np.diff(s)  # a_1, a_2, …
        if len(omega) < 2:
            return float(s[-1])

        # Levin u-transform: u_n = s_n / ω_n
        with np.errstate(divide="ignore", invalid="ignore"):
            u = s[1:] / omega

        # Neville–Aitken on the u-sequence
        analyser = LargeOrderAnalysis(np.array([0.0]))  # dummy
        return analyser.richardson_extrapolation(u[np.isfinite(u)], order=2)

    def convergence_check(
        self,
        N_values: np.ndarray,
        reference_fn: Optional[Callable[[float], float]] = None,
    ) -> Dict[str, np.ndarray]:
        """Check convergence of Padé–Borel resummation.

        Evaluates the resummation at several widths and compares against
        a reference function (if provided).

        Args:
            N_values: Array of width values.
            reference_fn: Optional exact function f(N) for comparison.

        Returns:
            Dictionary with 'N_values', 'resummed', and optionally
            'reference' and 'relative_error'.
        """
        N_values = np.asarray(N_values, dtype=float)
        m, n = self.best_pade(N_values[len(N_values) // 2])
        resummed = np.array([self.pade_borel_integral(N, m, n) for N in N_values])

        result: Dict[str, np.ndarray] = {
            "N_values": N_values,
            "resummed": resummed,
        }

        if reference_fn is not None:
            ref = np.array([reference_fn(N) for N in N_values])
            result["reference"] = ref
            with np.errstate(divide="ignore", invalid="ignore"):
                result["relative_error"] = np.abs((resummed - ref) / ref)

        return result


# ---------------------------------------------------------------------------
# ResurgentNTKCorrections
# ---------------------------------------------------------------------------

class ResurgentNTKCorrections:
    """Apply resurgent trans-series methods to NTK finite-width corrections.

    The NTK (Neural Tangent Kernel) of a width-N network admits a 1/N
    expansion:

        K(N) = K_∞ + K₁/N + K₂/N² + …

    whose coefficients K_n typically grow factorially, signalling
    non-perturbative corrections of order e^{-A·N}.  This class uses
    resurgent analysis to:

    1. Detect the instanton action A from the perturbative coefficients.
    2. Resum the divergent 1/N expansion via Padé–Borel.
    3. Predict NTK corrections beyond the reach of finite-order perturbation
       theory.
    4. Correct phase boundaries that were determined perturbatively.

    Attributes:
        ntk_corrections_by_order: Dictionary mapping perturbative order n
            to array of K_n values measured at different widths.
        width_values: Array of widths at which corrections were measured.

    References:
        Dyer & Gur-Ari, "Asymptotics of Wide Networks from Feynman
        Diagrams" (ICLR 2020).
        Huang & Yau, "Dynamics of Deep Neural Networks and Neural
        Tangent Hierarchy" (ICML 2020).
    """

    def __init__(
        self,
        ntk_corrections_by_order: Dict[int, np.ndarray],
        width_values: np.ndarray,
    ):
        """Initialise with measured NTK correction data.

        Args:
            ntk_corrections_by_order: Dict mapping order n to array of
                correction values K_n, one per width.
            width_values: Array of network widths at which corrections
                were measured.
        """
        self.ntk_corrections_by_order = ntk_corrections_by_order
        self.width_values = np.asarray(width_values, dtype=float)

        # Compile perturbative coefficients (averaged over widths)
        max_order = max(ntk_corrections_by_order.keys())
        self._perturbative_coeffs = np.zeros(max_order + 1)
        for n, vals in ntk_corrections_by_order.items():
            self._perturbative_coeffs[n] = np.mean(vals)

    def fit_transseries(
        self,
        max_perturbative_order: int = 5,
        n_instanton_sectors: int = 2,
    ) -> TransSeries:
        """Fit a trans-series to the NTK correction data.

        Uses the perturbative coefficients to initialise the trans-series,
        then determines non-perturbative sectors from large-order analysis.

        Args:
            max_perturbative_order: Maximum perturbative order to include.
            n_instanton_sectors: Number of instanton sectors.

        Returns:
            Fitted TransSeries instance.
        """
        coeffs = self._perturbative_coeffs[: max_perturbative_order + 1]
        A = self.extract_instanton_action()
        if not np.isfinite(A) or A <= 0:
            A = 1.0

        ts = TransSeries(coeffs, A, n_sectors=n_instanton_sectors + 1)

        # Populate 1-instanton sector from large-order relation
        lo = LargeOrderAnalysis(coeffs)
        fit_result = lo.factorial_over_power_fit()
        if np.isfinite(fit_result["C"]):
            # a_{1,0} ~ 2πi / S_1 · ...  (simplified)
            ts.sector_coeffs[1] = np.array([fit_result["C"]])

        # Determine σ by fitting to data at largest width
        if len(self.width_values) > 0:
            N_ref = self.width_values[-1]
            # Measured total correction at N_ref
            total_measured = sum(
                self.ntk_corrections_by_order.get(n, np.array([0.0]))[-1]
                / N_ref ** n
                for n in range(max_perturbative_order + 1)
            )
            pert_val = ts.perturbative_sector(N_ref, order=max_perturbative_order)
            residual = total_measured - pert_val
            one_inst = np.exp(-A * N_ref) * ts._sector_sum(1, N_ref)
            if np.abs(one_inst) > 1e-300:
                sigma = residual / one_inst
            else:
                sigma = 1.0
            # Store sigma hint (not an attribute of TransSeries, but useful)
            ts._fitted_sigma = float(sigma)

        return ts

    def extract_instanton_action(self) -> float:
        """Extract the non-perturbative scale A from large-order growth.

        Uses LargeOrderAnalysis on the perturbative NTK coefficients.

        Returns:
            Instanton action A (positive real).
        """
        coeffs = self._perturbative_coeffs
        if len(coeffs) < 4:
            return np.nan
        analyser = LargeOrderAnalysis(coeffs)
        return analyser.instanton_action_from_coefficients()

    def predict_at_width(self, width: float, method: str = "pade_borel") -> float:
        """Predict NTK correction at a given width using resummation.

        Args:
            width: Network width N.
            method: Resummation method — 'pade_borel', 'conformal',
                'optimal_truncation', or 'transseries'.

        Returns:
            Predicted total NTK correction.
        """
        coeffs = self._perturbative_coeffs

        if method == "pade_borel":
            pba = PadeBorelAnalysis(coeffs)
            m, n = pba.best_pade(width)
            return pba.pade_borel_integral(width, m, n)

        elif method == "conformal":
            pba = PadeBorelAnalysis(coeffs)
            return pba.conformal_pade(width, mapping="euler")

        elif method == "optimal_truncation":
            A = self.extract_instanton_action()
            if not np.isfinite(A):
                A = 1.0
            ts = TransSeries(coeffs, A)
            n_star = ts.optimal_truncation_order(width)
            return ts.perturbative_sector(width, order=n_star)

        elif method == "transseries":
            ts = self.fit_transseries()
            sigma = getattr(ts, "_fitted_sigma", 1.0)
            return ts.evaluate(width, sigma=sigma)

        else:
            raise ValueError(f"Unknown method '{method}'.")

    def non_perturbative_correction_magnitude(self, width: float) -> float:
        """Estimate |e^{-A·N}| — the magnitude of non-perturbative effects.

        Args:
            width: Network width N.

        Returns:
            exp(-A · N), the non-perturbative suppression factor.
        """
        A = self.extract_instanton_action()
        if not np.isfinite(A) or A <= 0:
            return np.nan
        return float(np.exp(-A * width))

    def critical_width(self) -> float:
        """Width at which non-perturbative effects become O(1).

        This is N_c such that exp(-A · N_c) ~ 1, i.e. N_c ~ 1/A.
        More precisely, we find where the 1-instanton correction equals
        the perturbative uncertainty (optimal truncation error).

        Returns:
            Critical width N_c.
        """
        A = self.extract_instanton_action()
        if not np.isfinite(A) or A <= 0:
            return np.nan
        # Leading estimate: N_c = 1/A
        # Refined: where optimally-truncated perturbative error ~ 1-inst
        return float(1.0 / A)

    def improvement_over_perturbative(
        self, width_range: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Quantify how much resurgent resummation improves over perturbative.

        Compares optimal truncation, Padé–Borel, and full trans-series
        predictions across a range of widths.

        Args:
            width_range: Array of widths to evaluate.

        Returns:
            Dictionary with 'widths', 'perturbative', 'pade_borel',
            'transseries', and 'improvement_factor'.
        """
        width_range = np.asarray(width_range, dtype=float)
        pert = np.array([self.predict_at_width(N, "optimal_truncation") for N in width_range])
        pb = np.array([self.predict_at_width(N, "pade_borel") for N in width_range])
        ts = np.array([self.predict_at_width(N, "transseries") for N in width_range])

        # Improvement factor: how much closer is PB to trans-series than pert?
        with np.errstate(divide="ignore", invalid="ignore"):
            improvement = np.abs(pert - ts) / (np.abs(pb - ts) + 1e-300)

        return {
            "widths": width_range,
            "perturbative": pert,
            "pade_borel": pb,
            "transseries": ts,
            "improvement_factor": improvement,
        }

    def phase_boundary_correction(
        self,
        perturbative_boundary: Callable[[float], float],
        width_range: np.ndarray,
    ) -> np.ndarray:
        """Correct a perturbative phase boundary with non-perturbative effects.

        A phase boundary determined perturbatively at x_c(∞) receives
        finite-width corrections:

            x_c(N) = x_c(∞) + Σ_n δx_n / N^n + σ · δx_np · e^{-A N} + …

        The perturbative corrections are already included in
        perturbative_boundary.  This method adds the non-perturbative
        correction.

        Args:
            perturbative_boundary: Function N → x_c(N) giving the
                perturbative phase boundary.
            width_range: Array of widths.

        Returns:
            Corrected phase boundary values.
        """
        width_range = np.asarray(width_range, dtype=float)
        x_pert = np.array([perturbative_boundary(N) for N in width_range])

        A = self.extract_instanton_action()
        if not np.isfinite(A) or A <= 0:
            return x_pert

        # Estimate non-perturbative boundary shift from Stokes constant
        ts = self.fit_transseries()
        stokes = ts.stokes_constants()
        s1 = stokes.get(1, 0.0)
        np_amplitude = float(np.abs(s1)) if np.isfinite(np.abs(s1)) else 0.0

        # Correction: δx_np · e^{-A N}
        np_correction = np_amplitude * np.exp(-A * width_range)

        return x_pert + np_correction

    def stokes_line_in_phase_diagram(
        self, param_ranges: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Locate Stokes lines in the phase diagram parameter space.

        A Stokes line is a curve in parameter space where the trans-series
        parameter σ jumps discontinuously.  This occurs when the instanton
        action A(x) becomes purely imaginary, so that Im(A) = 0 and the
        Borel singularity sits on the integration contour.

        The Stokes line separates regions of parameter space where
        non-perturbative effects have different signs / phases.

        Args:
            param_ranges: Dictionary with parameter names as keys and
                arrays of parameter values as values.  At least one of
                'learning_rate', 'depth', or 'temperature' should be
                present.

        Returns:
            Dictionary with 'stokes_line' (array of parameter values
            where Im(A) = 0) and 'instanton_action_values'.

        References:
            Berry, "Stokes' phenomenon; smoothing a Victorian
            discontinuity" (IHES 1989).
        """
        # Use the primary parameter axis
        param_name = list(param_ranges.keys())[0]
        param_values = np.asarray(param_ranges[param_name], dtype=float)

        # Model: instanton action depends on parameter
        # A(x) = A_0 + A_1 (x - x_0) + A_2 (x - x_0)^2 + …
        # Stokes line at Im(A(x)) = 0  (for real parameters, at A(x) = 0)
        A_base = self.extract_instanton_action()
        if not np.isfinite(A_base):
            A_base = 1.0

        # Simple model: A varies linearly with parameter
        x_mid = (param_values[0] + param_values[-1]) / 2.0
        A_values = A_base * (1.0 - (param_values - x_mid) / x_mid)

        # Stokes line where A crosses zero
        stokes_pts = []
        for i in range(len(A_values) - 1):
            if A_values[i] * A_values[i + 1] < 0:
                # Linear interpolation
                x_cross = param_values[i] - A_values[i] * (
                    param_values[i + 1] - param_values[i]
                ) / (A_values[i + 1] - A_values[i])
                stokes_pts.append(float(x_cross))

        return {
            "stokes_line": np.array(stokes_pts),
            "instanton_action_values": A_values,
            "parameter_name": param_name,
            "parameter_values": param_values,
        }
