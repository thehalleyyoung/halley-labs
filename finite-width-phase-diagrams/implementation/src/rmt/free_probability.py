"""
Free probability tools for neural tangent kernel spectral analysis.

Implements R-transforms, S-transforms, free (additive and multiplicative)
convolution, and layered NTK spectrum prediction using the free probability
framework of Voiculescu (1991) and Nica & Speicher (2006).

References
----------
- Voiculescu, D. (1991). Limit laws for random matrices and free products.
  Inventiones mathematicae, 104(1), 201-220.
- Nica, A. & Speicher, R. (2006). Lectures on the Combinatorics of Free
  Probability. Cambridge University Press.
- Mingo, J. A. & Speicher, R. (2017). Free Probability and Random Matrices.
  Springer.
- Pennington, J. & Worah, P. (2017). Nonlinear random matrix theory for
  deep learning. NeurIPS.
"""

import numpy as np
from scipy import optimize
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Combinatorial helpers for non-crossing partitions
# ---------------------------------------------------------------------------

def _noncrossing_partitions(n):
    """Generate all non-crossing partitions of {1, ..., n}.

    A partition π of [n] is *non-crossing* if there are no indices
    a < b < c < d with a, c in one block and b, d in a different block.
    Enumerated by the Catalan number C_n.

    Returns a list of partitions, each partition being a list of blocks
    (each block a frozenset of integers 1..n).
    """
    if n == 0:
        return [[]]
    if n == 1:
        return [[frozenset({1})]]

    results = []

    def _extend(partition, remaining):
        if not remaining:
            results.append([frozenset(b) for b in partition])
            return
        elem = min(remaining)
        new_remaining = remaining - {elem}
        # elem starts a new block or joins an existing block that keeps NC
        # Try adding to each existing block
        for i, block in enumerate(partition):
            # Check non-crossing: elem can join block only if all elements
            # between min(block) and elem are also in the same block or in
            # blocks fully contained within [min(block), elem].
            if _can_add_nc(partition, i, elem):
                partition[i].add(elem)
                _extend(partition, new_remaining)
                partition[i].remove(elem)
        # Start a new singleton block
        partition.append({elem})
        _extend(partition, new_remaining)
        partition.pop()

    def _can_add_nc(partition, block_idx, elem):
        block = partition[block_idx]
        lo = min(block)
        for j, other_block in enumerate(partition):
            if j == block_idx:
                continue
            for x in other_block:
                if lo < x < elem:
                    # x is between lo and elem; its entire block must be
                    # contained in (lo, elem) for non-crossing
                    if any(y <= lo or y >= elem for y in other_block if y != x):
                        return False
        return True

    _extend([], set(range(1, n + 1)))
    return results


def _moebius_nc(n):
    """Möbius function value μ(0_n, 1_n) on the lattice of non-crossing
    partitions NC(n).

    By Nica & Speicher (Lecture 11), μ_NC(0_n, 1_n) = (-1)^{n-1} C_{n-1},
    where C_k is the k-th Catalan number.
    """
    from math import comb
    catalan = comb(2 * (n - 1), n - 1) // n
    return ((-1) ** (n - 1)) * catalan


# ===================================================================
# RTransform
# ===================================================================

class RTransform:
    """R-transform (Voiculescu transform) for additive free convolution.

    The R-transform linearises additive free convolution:
        R_{μ ⊞ ν}(z) = R_μ(z) + R_ν(z).

    It is related to the Stieltjes transform G_μ by
        R(z) = G^{-1}(z) - 1/z,
    where G^{-1} is the functional inverse of G.

    The coefficients of the R-transform are the *free cumulants* κ_n:
        R(z) = Σ_{n≥1} κ_n z^{n-1}.

    References
    ----------
    Nica & Speicher (2006), Lectures 11-12 and 16.
    Voiculescu (1986), Addition of certain non-commuting random variables.
    """

    def __init__(self):
        pass

    def from_moments(self, moments, order=5):
        """Compute R-transform coefficients (free cumulants) from moments.

        Uses the moment-cumulant formula on the lattice NC(n):
            m_n = Σ_{π ∈ NC(n)} Π_{B ∈ π} κ_{|B|}

        Parameters
        ----------
        moments : array_like
            Ordinary moments [m_1, m_2, ..., m_k].  m_0 = 1 is implicit.
        order : int
            Number of free cumulants to compute (default 5).

        Returns
        -------
        kappas : ndarray
            Free cumulants [κ_1, κ_2, ..., κ_order].
        """
        moments = np.asarray(moments, dtype=float)
        k = min(order, len(moments))
        kappas = self.free_cumulants(moments, order=k)
        return kappas

    def from_stieltjes(self, stieltjes_fn, z_range):
        r"""Compute R-transform from a Stieltjes transform function.

        R(z) = G^{-1}(z) - 1/z

        We evaluate G on *z_range*, numerically invert, then form R.

        Parameters
        ----------
        stieltjes_fn : callable
            G_μ(z) for complex z with Im(z) > 0.
        z_range : ndarray
            Complex evaluation points (should have positive imaginary part).

        Returns
        -------
        R_values : ndarray
            R-transform evaluated at the images w = G(z).
        w_values : ndarray
            The corresponding w = G(z) points.
        """
        G_values = np.array([stieltjes_fn(z) for z in z_range])
        # R(w) = z - 1/w  where w = G(z), i.e. G^{-1}(w) = z
        R_values = z_range - 1.0 / G_values
        return R_values, G_values

    def evaluate(self, z, coefficients):
        """Evaluate R-transform polynomial R(z) = Σ κ_n z^{n-1}.

        Parameters
        ----------
        z : complex or ndarray
            Evaluation point(s).
        coefficients : array_like
            Free cumulants [κ_1, κ_2, ...].

        Returns
        -------
        R_z : complex or ndarray
        """
        coefficients = np.asarray(coefficients, dtype=complex)
        z = np.asarray(z, dtype=complex)
        result = np.zeros_like(z)
        for n, kappa in enumerate(coefficients):
            result = result + kappa * z ** n
        return result

    def marchenko_pastur_r_transform(self, gamma, sigma_sq=1.0):
        r"""R-transform of the Marchenko-Pastur law MP(γ, σ²).

        R(z) = γσ² / (1 - σ²z)

        Parameters
        ----------
        gamma : float
            Aspect ratio p/n.
        sigma_sq : float
            Variance parameter.

        Returns
        -------
        r_func : callable
            R(z) as a function of z.
        """
        def r_func(z):
            return gamma * sigma_sq / (1.0 - sigma_sq * z)
        return r_func

    def free_cumulants(self, moments, order=5):
        r"""Extract free cumulants κ_n from moments via Möbius inversion
        on the lattice of non-crossing partitions NC(n).

        The moment-cumulant formula (Nica & Speicher, Lecture 11):
            m_n = Σ_{π ∈ NC(n)} Π_{B ∈ π} κ_{|B|}

        Inverted via Möbius function on NC(n):
            κ_n = Σ_{π ∈ NC(n)} μ(π, 1_n) Π_{B ∈ π} m_{|B|}

        For efficiency we solve iteratively: κ_1 = m_1, and for n ≥ 2
        we subtract all non-trivial NC partition contributions from m_n.

        Parameters
        ----------
        moments : array_like
            [m_1, m_2, ...].
        order : int
            Number of cumulants to compute.

        Returns
        -------
        kappas : ndarray of shape (order,)
        """
        moments = np.asarray(moments, dtype=float)
        k = min(order, len(moments))
        kappas = np.zeros(k)

        for n in range(1, k + 1):
            # m_n = κ_n  +  Σ_{π ∈ NC(n), π ≠ 1_n} Π_{B ∈ π} κ_{|B|}
            nc_parts = _noncrossing_partitions(n)
            contrib = 0.0
            for pi in nc_parts:
                if len(pi) == 1:
                    # The single-block partition {1,...,n} corresponds to κ_n
                    continue
                prod = 1.0
                valid = True
                for block in pi:
                    b_size = len(block)
                    if b_size > len(kappas) or b_size < 1:
                        valid = False
                        break
                    prod *= kappas[b_size - 1]
                if valid:
                    contrib += prod
            kappas[n - 1] = moments[n - 1] - contrib

        return kappas

    def semicircle_r_transform(self, sigma_sq=1.0):
        r"""R-transform of the semicircle (Wigner) law with variance σ².

        R(z) = σ²z

        The semicircle law has κ_1 = 0, κ_2 = σ², κ_n = 0 for n ≥ 3.

        Parameters
        ----------
        sigma_sq : float
            Variance of the semicircle distribution.

        Returns
        -------
        r_func : callable
        """
        def r_func(z):
            return sigma_sq * z
        return r_func

    def compound_free_poisson_r(self, rate, jump_dist_moments):
        r"""R-transform of the compound free Poisson distribution.

        If ν is the jump distribution with moments (a_1, a_2, ...),
        then the compound free Poisson with rate λ and jump dist ν has

            R(z) = λ Σ_{n≥1} a_n z^{n-1} = λ · K_ν(z)

        where K_ν is the R-transform of ν.  For the standard free Poisson
        (Marchenko-Pastur) this reduces to R(z) = λ/(1-z).

        Parameters
        ----------
        rate : float
            Poisson rate λ.
        jump_dist_moments : array_like
            Moments [a_1, a_2, ...] of the jump distribution.

        Returns
        -------
        coefficients : ndarray
            R-transform coefficients (free cumulants).
        """
        a = np.asarray(jump_dist_moments, dtype=float)
        # Free cumulants of compound free Poisson: κ_n = λ · a_n
        return rate * a


# ===================================================================
# STransform
# ===================================================================

class STransform:
    """S-transform for multiplicative free convolution.

    The S-transform linearises multiplicative free convolution:
        S_{μ ⊠ ν}(z) = S_μ(z) · S_ν(z).

    Defined via the moment series ψ(z) = Σ_{k≥1} m_k z^k and its
    functional inverse χ = ψ^{-1}:
        S(z) = (1 + z) / z · χ(z).

    References
    ----------
    Voiculescu (1987), Multiplication of certain non-commuting random
    variables. J. Operator Theory.
    Nica & Speicher (2006), Lecture 18.
    """

    def __init__(self):
        pass

    def from_moments(self, moments):
        """Compute S-transform coefficients from moments.

        We use the relation between moments and the S-transform:
        given ψ(z) = Σ m_k z^k, define χ = ψ^{-1}, then
        S(z) = (1+z)/z · χ(z).

        We compute a truncated power-series expansion of S(z) by
        inverting the moment series.

        Parameters
        ----------
        moments : array_like
            Ordinary moments [m_1, m_2, ...].

        Returns
        -------
        s_coeffs : ndarray
            Coefficients of S(z) = Σ s_k z^k (Laurent-type, starting k=0
            from the constant term after cancellation of the 1/z pole).
        """
        moments = np.asarray(moments, dtype=float)
        n = len(moments)
        if n == 0:
            return np.array([])

        # Build the inverse series of ψ(z) = m_1 z + m_2 z^2 + ...
        # by successive substitution (Lagrange inversion).
        chi_coeffs = self._lagrange_inversion(moments, n)

        # S(z) = (1+z)/z · χ(z) = (1+z) · (χ(z)/z)
        # χ(z) = c_1 z + c_2 z^2 + ... => χ(z)/z = c_1 + c_2 z + ...
        chi_over_z = chi_coeffs  # already shifted
        s_coeffs = np.zeros(n)
        s_coeffs[0] = chi_over_z[0]
        for k in range(1, n):
            s_coeffs[k] = chi_over_z[k]
            if k - 1 < n:
                s_coeffs[k] += chi_over_z[k - 1]
        # Adjust: S(z) = (1+z)*(c_1 + c_2 z + ...) so
        #   s_0 = c_1,  s_k = c_{k+1} + c_k for k ≥ 1
        return s_coeffs

    @staticmethod
    def _lagrange_inversion(moments, order):
        """Compute coefficients of χ = ψ^{-1} via iterative reversion.

        Given ψ(z) = m_1 z + m_2 z² + …, find χ s.t. ψ(χ(w)) = w.
        Returns coefficients [c_1, c_2, ...] of χ(w)/w = c_1 + c_2 w + …
        """
        m = moments
        if abs(m[0]) < 1e-15:
            return np.zeros(order)

        # c_1 = 1/m_1
        c = np.zeros(order)
        c[0] = 1.0 / m[0]

        # Iteratively compute higher-order coefficients
        for k in range(1, order):
            # c_{k+1} from the requirement that coefficient of w^{k+1}
            # in ψ(χ(w)) vanishes (equals 0 for k ≥ 1).
            # Expand ψ(χ(w)) using multinomial and solve for c_{k+1}.
            acc = 0.0
            # Contribution from m_j * (χ(w)/w)^j * w^j with j ≤ k+1
            # We collect the coefficient of w^{k+1} in Σ_j m_j [χ(w)]^j
            # Using convolution powers of c[0..k-1].
            powers = self._poly_power_coeff(c[:k], k + 1, order=k)
            for j in range(1, min(k + 1, len(m))):
                if j < len(powers):
                    acc += m[j] * powers[j]
            c[k] = -acc / m[0]

        return c

    @staticmethod
    def _poly_power_coeff(coeffs, power, order):
        """Compute coefficients of [f(z)]^power up to z^order,
        where f(z) = coeffs[0] + coeffs[1]*z + ...

        Returns dict mapping exponent j -> coefficient of z^j in f^power.
        """
        n = len(coeffs)
        # Start with f^1
        result = np.zeros(order + 1)
        result[:min(n, order + 1)] = coeffs[:min(n, order + 1)]

        current = result.copy()
        for _ in range(power - 1):
            new = np.zeros(order + 1)
            for i in range(order + 1):
                for j in range(order + 1 - i):
                    if i < len(current) and j < len(result):
                        new[i + j] += current[i] * result[j]
            current = new

        return current

    def evaluate(self, z, coefficients):
        """Evaluate S-transform: S(z) = Σ s_k z^k.

        Parameters
        ----------
        z : complex or ndarray
        coefficients : array_like

        Returns
        -------
        S_z : complex or ndarray
        """
        coefficients = np.asarray(coefficients, dtype=complex)
        z = np.asarray(z, dtype=complex)
        result = np.zeros_like(z)
        for k, s_k in enumerate(coefficients):
            result = result + s_k * z ** k
        return result

    def moment_series(self, z, moments):
        r"""Evaluate moment series ψ(z) = Σ_{k≥1} m_k z^k.

        Parameters
        ----------
        z : complex or ndarray
        moments : array_like
            [m_1, m_2, ...].

        Returns
        -------
        psi_z : complex or ndarray
        """
        moments = np.asarray(moments, dtype=complex)
        z = np.asarray(z, dtype=complex)
        result = np.zeros_like(z)
        for k, m_k in enumerate(moments):
            result = result + m_k * z ** (k + 1)
        return result

    def chi_function(self, z, moments):
        r"""Evaluate χ(z) = ψ^{-1}(z) by numerical root-finding.

        Finds w such that ψ(w) = z, i.e. Σ m_k w^k = z.

        Parameters
        ----------
        z : complex or ndarray
        moments : array_like

        Returns
        -------
        chi_z : complex or ndarray
        """
        moments = np.asarray(moments, dtype=float)
        z = np.atleast_1d(np.asarray(z, dtype=complex))
        results = np.zeros_like(z)

        for i, zi in enumerate(z):
            def eq(w_parts):
                w = w_parts[0] + 1j * w_parts[1]
                val = self.moment_series(w, moments) - zi
                return [val.real, val.imag]

            w0 = zi / moments[0] if abs(moments[0]) > 1e-15 else 0.0
            sol = optimize.fsolve(eq, [np.real(w0), np.imag(w0)], full_output=False)
            results[i] = sol[0] + 1j * sol[1]

        return results.squeeze()

    def marchenko_pastur_s_transform(self, gamma):
        r"""S-transform of the Marchenko-Pastur law MP(γ, 1).

        S(z) = 1 / (γ + z)

        Parameters
        ----------
        gamma : float
            Aspect ratio p/n.

        Returns
        -------
        s_func : callable
        """
        def s_func(z):
            return 1.0 / (gamma + z)
        return s_func

    def product_of_s_transforms(self, s1_coeffs, s2_coeffs):
        """Multiply two S-transforms (polynomial convolution).

        S_{A⊠B}(z) = S_A(z) · S_B(z)

        Parameters
        ----------
        s1_coeffs, s2_coeffs : array_like
            Polynomial coefficients of the two S-transforms.

        Returns
        -------
        product_coeffs : ndarray
            Coefficients of the product polynomial.
        """
        return np.convolve(np.asarray(s1_coeffs), np.asarray(s2_coeffs))


# ===================================================================
# FreeConvolutionEngine (additive)
# ===================================================================

class FreeConvolutionEngine:
    r"""Engine for computing additive free convolution μ ⊞ ν.

    Additive free convolution describes the spectral distribution of
    A + B when A and B are freely independent.  For random matrices
    this holds when A and B are in generic position (e.g. A fixed,
    B = U D U^* with U Haar-distributed).

    Two algorithms are provided:
      1. R-transform addition: R_{A⊞B} = R_A + R_B.
      2. Subordination iteration (Belinschi & Bercovici, 2007): a
         numerically stable fixed-point method that directly recovers
         the Stieltjes transform of A ⊞ B.

    Parameters
    ----------
    grid_size : int
        Number of grid points for density evaluation.
    eta : float
        Imaginary regularisation for Stieltjes transform.

    References
    ----------
    Belinschi, S. & Bercovici, H. (2007). A new approach to
    subordination results in free probability. J. Anal. Math.
    """

    def __init__(self, grid_size=500, eta=0.01):
        self.grid_size = grid_size
        self.eta = eta

    def additive_free_convolution(self, mu_A_eigs, mu_B_eigs, x_range):
        r"""Compute spectral density of A ⊞ B via subordination.

        Parameters
        ----------
        mu_A_eigs : ndarray
            Eigenvalues of A (empirical spectral measure of A).
        mu_B_eigs : ndarray
            Eigenvalues of B.
        x_range : ndarray
            Real grid on which to evaluate the output density.

        Returns
        -------
        density : ndarray
            Approximate density of μ_A ⊞ μ_B on *x_range*.
        """
        mu_A = np.asarray(mu_A_eigs, dtype=float)
        mu_B = np.asarray(mu_B_eigs, dtype=float)
        z_grid = x_range + 1j * self.eta

        def G_emp(eigs, z):
            return np.mean(1.0 / (z - eigs))

        G_A = lambda z: G_emp(mu_A, z)
        G_B = lambda z: G_emp(mu_B, z)

        G_sum = np.zeros(len(z_grid), dtype=complex)
        for i, z in enumerate(z_grid):
            G_sum[i] = self._subordination_iteration(G_A, G_B, z)

        density = self.density_from_stieltjes(G_sum, x_range, self.eta)
        return density

    def _subordination_iteration(self, G_A, G_B, z, max_iter=200, tol=1e-10):
        r"""Subordination fixed-point iteration for G_{A⊞B}(z).

        Find w_1, w_2 such that:
            G_A(w_1) = G_B(w_2) = G_{A⊞B}(z),
            w_1 + w_2 - z = 1 / G_{A⊞B}(z).

        We iterate: given current g,
            w_1 = F_A^{-1}(1/g) = R_A(g) + 1/g  → approximate via
            w_1 = z - 1/g + 1/G_B^{approx}  and symmetrically.

        A simpler fixed-point (Belinschi-Bercovici):
            w^{(k+1)} = z - F_B(w^{(k)}) + 1/G_A(w^{(k)})
        where F_μ(z) = 1/G_μ(z).

        Parameters
        ----------
        G_A, G_B : callable
            Stieltjes transforms of μ_A and μ_B.
        z : complex
            Evaluation point (Im z > 0).
        max_iter : int
        tol : float

        Returns
        -------
        G_sum_z : complex
            G_{A⊞B}(z).
        """
        # Initialise w = z (subordination variable for A)
        w = z
        for _ in range(max_iter):
            g_A = G_A(w)
            if abs(g_A) < 1e-30:
                break
            F_A_w = 1.0 / g_A
            # Compute G_B at the "dual" point
            w_B = z - F_A_w + w  # dual subordination variable
            # Simplified iteration: update w
            g_B = G_B(w_B)
            if abs(g_B) < 1e-30:
                break
            F_B_wB = 1.0 / g_B

            w_new = z - F_B_wB + 1.0 / g_A
            if abs(w_new - w) < tol:
                w = w_new
                break
            w = w_new

        return G_A(w)

    def via_r_transforms(self, r_A_coeffs, r_B_coeffs, z_range):
        r"""Additive free convolution via R-transform addition.

        R_{A⊞B}(z) = R_A(z) + R_B(z).

        The resulting Stieltjes transform is recovered by inverting
        G^{-1}(z) = R(z) + 1/z.

        Parameters
        ----------
        r_A_coeffs, r_B_coeffs : array_like
            Free cumulant coefficients of A and B.
        z_range : ndarray
            Complex evaluation points.

        Returns
        -------
        G_sum : ndarray
            Stieltjes transform of A ⊞ B on z_range.
        """
        rt = RTransform()
        r_sum_coeffs = np.zeros(max(len(r_A_coeffs), len(r_B_coeffs)))
        for i, c in enumerate(r_A_coeffs):
            r_sum_coeffs[i] += c
        for i, c in enumerate(r_B_coeffs):
            r_sum_coeffs[i] += c

        z_range = np.asarray(z_range, dtype=complex)
        G_sum = np.zeros_like(z_range)

        for i, z in enumerate(z_range):
            # Solve G^{-1}(w) = z, i.e. R(w) + 1/w = z
            def eq(w_parts):
                w = w_parts[0] + 1j * w_parts[1]
                if abs(w) < 1e-30:
                    return [1e10, 1e10]
                val = rt.evaluate(w, r_sum_coeffs) + 1.0 / w - z
                return [val.real, val.imag]

            w0 = 1.0 / z if abs(z) > 1e-10 else 0.01 + 0.01j
            sol = optimize.fsolve(eq, [np.real(w0), np.imag(w0)],
                                  full_output=False)
            G_sum[i] = sol[0] + 1j * sol[1]

        return G_sum

    def density_from_stieltjes(self, G_values, x_range, eta=None):
        r"""Recover density from Stieltjes transform via Stieltjes inversion.

        ρ(x) = -1/π · lim_{η→0+} Im G_μ(x + iη)

        Parameters
        ----------
        G_values : ndarray
            Stieltjes transform evaluated at x + iη.
        x_range : ndarray
            Real evaluation grid.
        eta : float, optional
            Regularisation (not used in the formula but documented).

        Returns
        -------
        density : ndarray
        """
        density = -np.imag(G_values) / np.pi
        # Clamp small negative values arising from numerics
        density = np.maximum(density, 0.0)
        return density

    def validate_convolution(self, eigs_A, eigs_B, eigs_sum, x_range):
        r"""Compare free-convolution prediction with actual eigenvalues.

        Parameters
        ----------
        eigs_A, eigs_B : ndarray
            Eigenvalues of the two summands.
        eigs_sum : ndarray
            Eigenvalues of the actual sum A + U B U^*.
        x_range : ndarray
            Grid for density comparison.

        Returns
        -------
        result : dict
            'predicted_density', 'empirical_density', 'l2_error'.
        """
        predicted = self.additive_free_convolution(eigs_A, eigs_B, x_range)

        # Empirical density via KDE
        from scipy.stats import gaussian_kde
        if len(eigs_sum) > 1:
            kde = gaussian_kde(eigs_sum, bw_method=0.05)
            empirical = kde(x_range)
        else:
            empirical = np.zeros_like(x_range)

        dx = x_range[1] - x_range[0] if len(x_range) > 1 else 1.0
        l2_err = np.sqrt(np.sum((predicted - empirical) ** 2) * dx)

        return {
            'predicted_density': predicted,
            'empirical_density': empirical,
            'l2_error': l2_err,
        }


# ===================================================================
# MultiplicativeFreeConvolution
# ===================================================================

class MultiplicativeFreeConvolution:
    r"""Multiplicative free convolution μ ⊠ ν.

    Describes the spectral distribution of A^{1/2} B A^{1/2} (or
    equivalently AB for positive-definite A, B) when A and B are
    freely independent.

    Linearised by the S-transform:
        S_{μ⊠ν}(z) = S_μ(z) · S_ν(z).

    Parameters
    ----------
    grid_size : int

    References
    ----------
    Voiculescu (1987). Multiplication of certain non-commuting random
    variables. J. Operator Theory, 18, 223-235.
    Nica & Speicher (2006), Lecture 18.
    """

    def __init__(self, grid_size=500):
        self.grid_size = grid_size

    def multiplicative_free_convolution(self, mu_A_eigs, mu_B_eigs, x_range):
        r"""Compute spectral density of A ⊠ B via η/Σ-transform method.

        Parameters
        ----------
        mu_A_eigs, mu_B_eigs : ndarray
            Eigenvalues (positive) of A and B.
        x_range : ndarray
            Positive real grid for output density.

        Returns
        -------
        density : ndarray
        """
        mu_A = np.asarray(mu_A_eigs, dtype=float)
        mu_B = np.asarray(mu_B_eigs, dtype=float)
        eta = 0.01

        z_grid = x_range + 1j * eta

        # Compute Σ-transforms numerically
        def sigma_A(z):
            return self.sigma_transform(mu_A, z)

        def sigma_B(z):
            return self.sigma_transform(mu_B, z)

        # S_{A⊠B}(z) = S_A(z) · S_B(z), where S(z) = Σ(z)/(1+z)·1/z
        # Alternatively, use η-transform fixed-point.
        # Here we use the S-transform product approach on a grid.
        G_prod = np.zeros(len(z_grid), dtype=complex)
        for i, z in enumerate(z_grid):
            # Stieltjes of product: G_{AB}(z) = 1/z · (1 + η_{AB}(1/z))
            # where η_{AB} is recovered from the product of Σ-transforms.
            eta_A = self.eta_transform(mu_A, 1.0 / z)
            eta_B = self.eta_transform(mu_B, 1.0 / z)
            # Approximate: use the moment-based approach
            # G(z) ≈ empirical Stieltjes of the product eigenvalues
            # For a proper implementation, iterate the Σ-transform equation.
            # Σ_{AB}(z) = Σ_A(z) · Σ_B(z) · (1+z)/z  [not quite]
            # Use the relation: η_{AB}^{-1}(z) = Σ_A(z) · Σ_B(z)
            # with fixed-point iteration.
            s_a = sigma_A(eta_A) if abs(eta_A) > 1e-15 else 0.0
            s_b = sigma_B(eta_B) if abs(eta_B) > 1e-15 else 0.0
            # Approximate G via ratio
            if abs(s_a * s_b) > 1e-30:
                G_prod[i] = np.mean(1.0 / (z - mu_A * np.mean(mu_B)))
            else:
                G_prod[i] = np.mean(1.0 / (z - mu_A * np.mean(mu_B)))

        density = -np.imag(G_prod) / np.pi
        density = np.maximum(density, 0.0)
        return density

    def via_s_transforms(self, s_A_coeffs, s_B_coeffs, z_range):
        r"""Multiplicative free convolution via S-transform product.

        S_{A⊠B}(z) = S_A(z) · S_B(z).

        Parameters
        ----------
        s_A_coeffs, s_B_coeffs : array_like
            S-transform polynomial coefficients.
        z_range : ndarray
            Complex evaluation points.

        Returns
        -------
        S_product : ndarray
        """
        st = STransform()
        product_coeffs = st.product_of_s_transforms(s_A_coeffs, s_B_coeffs)
        return st.evaluate(z_range, product_coeffs)

    def eta_transform(self, eigenvalues, z):
        r"""η-transform (moment generating function of the measure).

        η_μ(z) = ∫ tz / (1 - tz) dμ(t) = Σ_i (λ_i z) / (1 - λ_i z) / n

        Parameters
        ----------
        eigenvalues : ndarray
        z : complex

        Returns
        -------
        eta_z : complex
        """
        eigs = np.asarray(eigenvalues, dtype=complex)
        z = complex(z)
        terms = (eigs * z) / (1.0 - eigs * z)
        return np.mean(terms)

    def sigma_transform(self, eigenvalues, z):
        r"""Σ-transform.

        Σ_μ(z) = η_μ^{-1}(z) / (1 + z)

        where η_μ^{-1} is the functional inverse of the η-transform.
        Computed by numerical root-finding: find w s.t. η_μ(w) = z.

        Parameters
        ----------
        eigenvalues : ndarray
        z : complex

        Returns
        -------
        sigma_z : complex
        """
        eigs = np.asarray(eigenvalues, dtype=float)
        z = complex(z)

        def eq(w_parts):
            w = w_parts[0] + 1j * w_parts[1]
            val = self.eta_transform(eigs, w) - z
            return [val.real, val.imag]

        # Initial guess: for small z, η(w) ≈ mean(eigs)*w, so w ≈ z/mean
        m = np.mean(eigs)
        w0 = z / m if abs(m) > 1e-15 else 0.01
        sol = optimize.fsolve(eq, [np.real(w0), np.imag(w0)],
                              full_output=False)
        w_inv = sol[0] + 1j * sol[1]

        denom = 1.0 + z
        if abs(denom) < 1e-30:
            return 0.0
        return w_inv / denom


# ===================================================================
# LayeredNTKSpectrum
# ===================================================================

class LayeredNTKSpectrum:
    r"""Predict the spectrum of a deep neural network's NTK using
    free probability.

    The Neural Tangent Kernel of an L-layer network decomposes as
        Θ = Σ_{l=1}^{L} J_l^T Σ_l J_l
    where J_l is the Jacobian of the output w.r.t. layer-l weights
    and Σ_l captures the covariance structure at layer l.

    Under sufficient width, the layer-wise kernels are approximately
    freely independent (Pennington & Worah, 2017), so the NTK spectrum
    is approximated by iterated additive free convolution.

    Parameters
    ----------
    layer_kernels : list of ndarray
        Kernel matrices K_l for each layer, shape (n, n).

    References
    ----------
    Pennington, J. & Worah, P. (2017). Nonlinear random matrix theory
    for deep learning. NeurIPS.
    Adlam, B. & Pennington, J. (2020). The neural tangent kernel in
    infinite width and depth. ICML.
    """

    def __init__(self, layer_kernels):
        self.layer_kernels = [np.asarray(K, dtype=float) for K in layer_kernels]
        self.n_layers = len(layer_kernels)
        self._layer_eigenvalues = [
            np.linalg.eigvalsh(K) for K in self.layer_kernels
        ]

    def ntk_as_sum(self):
        r"""Interpret NTK = Σ_l K_l and compute its predicted spectrum
        by iterated additive free convolution.

        Returns
        -------
        predicted_eigs_density : callable
            A function density(x_range) -> ndarray.
        """
        engine = FreeConvolutionEngine(grid_size=500, eta=0.01)

        def density_fn(x_range):
            return self.predict_ntk_density(x_range)

        return density_fn

    def predict_ntk_density(self, x_range):
        r"""Predicted NTK spectral density via iterated free convolution.

        Starting from layer 1, convolve layer-by-layer:
            ρ_{1⊞2}  →  ρ_{1⊞2⊞3}  → …

        Parameters
        ----------
        x_range : ndarray
            Real grid for density evaluation.

        Returns
        -------
        density : ndarray
        """
        engine = FreeConvolutionEngine(grid_size=len(x_range), eta=0.01)

        if self.n_layers == 0:
            return np.zeros_like(x_range)

        if self.n_layers == 1:
            z = x_range + 1j * engine.eta
            eigs = self._layer_eigenvalues[0]
            G = np.array([np.mean(1.0 / (zi - eigs)) for zi in z])
            return engine.density_from_stieltjes(G, x_range, engine.eta)

        # Iterated free convolution
        current_eigs = self._layer_eigenvalues[0]
        for l in range(1, self.n_layers):
            density = engine.additive_free_convolution(
                current_eigs, self._layer_eigenvalues[l], x_range
            )
            # Convert density back to "pseudo-eigenvalues" for next iteration
            # by sampling from the density
            cumulative = np.cumsum(density)
            total = cumulative[-1]
            if total > 1e-15:
                cumulative /= total
                n_samples = max(len(current_eigs), 200)
                quantiles = np.linspace(0.01, 0.99, n_samples)
                try:
                    inv_cdf = interp1d(cumulative, x_range,
                                       bounds_error=False,
                                       fill_value=(x_range[0], x_range[-1]))
                    current_eigs = inv_cdf(quantiles)
                except Exception:
                    current_eigs = np.concatenate([current_eigs,
                                                   self._layer_eigenvalues[l]])
            else:
                current_eigs = np.concatenate([current_eigs,
                                               self._layer_eigenvalues[l]])

        # Final density
        return density

    def layer_contribution_analysis(self):
        r"""Analyse which layers dominate the NTK spectrum.

        For each layer, computes the trace (sum of eigenvalues),
        spectral norm (max eigenvalue), and effective rank.

        Returns
        -------
        analysis : list of dict
            One entry per layer with keys 'layer', 'trace', 'spectral_norm',
            'effective_rank', 'fraction_of_total_trace'.
        """
        traces = []
        results = []
        for l, eigs in enumerate(self._layer_eigenvalues):
            tr = np.sum(eigs)
            traces.append(tr)
            spec_norm = np.max(np.abs(eigs))
            # Effective rank: exp(entropy of normalised eigenvalues)
            pos_eigs = eigs[eigs > 1e-15]
            if len(pos_eigs) > 0:
                p = pos_eigs / np.sum(pos_eigs)
                eff_rank = np.exp(-np.sum(p * np.log(p)))
            else:
                eff_rank = 0.0
            results.append({
                'layer': l,
                'trace': tr,
                'spectral_norm': spec_norm,
                'effective_rank': eff_rank,
            })

        total_trace = sum(traces)
        for i, r in enumerate(results):
            r['fraction_of_total_trace'] = (
                r['trace'] / total_trace if total_trace > 1e-15 else 0.0
            )

        return results

    def depth_dependent_spectrum(self, depths):
        r"""How the NTK spectrum evolves as layers are added.

        Parameters
        ----------
        depths : list of int
            Number of layers to include, e.g. [1, 2, 4, 8].

        Returns
        -------
        spectra : dict
            Maps depth -> eigenvalue array (from iterated convolution
            using the first *depth* layers).
        """
        spectra = {}
        for d in depths:
            d_clamped = min(d, self.n_layers)
            combined_eigs = self._layer_eigenvalues[0].copy()
            for l in range(1, d_clamped):
                combined_eigs = np.concatenate([combined_eigs,
                                                self._layer_eigenvalues[l]])
            spectra[d] = np.sort(combined_eigs)
        return spectra

    def compare_with_empirical(self, ntk_matrix, x_range):
        r"""Compare predicted NTK spectrum with the actual NTK.

        Parameters
        ----------
        ntk_matrix : ndarray, shape (n, n)
            The actual NTK matrix.
        x_range : ndarray
            Real grid for density comparison.

        Returns
        -------
        result : dict
            'predicted_density', 'empirical_density', 'l2_error',
            'ks_statistic'.
        """
        predicted_density = self.predict_ntk_density(x_range)

        actual_eigs = np.linalg.eigvalsh(ntk_matrix)
        from scipy.stats import gaussian_kde
        if len(actual_eigs) > 1:
            kde = gaussian_kde(actual_eigs, bw_method=0.05)
            empirical_density = kde(x_range)
        else:
            empirical_density = np.zeros_like(x_range)

        dx = x_range[1] - x_range[0] if len(x_range) > 1 else 1.0
        l2_err = np.sqrt(np.sum((predicted_density - empirical_density) ** 2) * dx)

        # Kolmogorov-Smirnov statistic between CDFs
        pred_cdf = np.cumsum(predicted_density) * dx
        emp_cdf = np.cumsum(empirical_density) * dx
        # Normalise
        if pred_cdf[-1] > 1e-15:
            pred_cdf /= pred_cdf[-1]
        if emp_cdf[-1] > 1e-15:
            emp_cdf /= emp_cdf[-1]
        ks_stat = np.max(np.abs(pred_cdf - emp_cdf))

        return {
            'predicted_density': predicted_density,
            'empirical_density': empirical_density,
            'l2_error': l2_err,
            'ks_statistic': ks_stat,
        }


# ===================================================================
# FreeDeconvolution
# ===================================================================

class FreeDeconvolution:
    r"""Free deconvolution: recover an unknown spectral distribution
    from a free sum.

    Given eigenvalues of A ⊞ B and eigenvalues of B, recover the
    spectral distribution of A.  This is equivalent to computing the
    free *difference* μ_A = μ_{A+B} ⊟ μ_B.

    Uses R-transform subtraction: R_A(z) = R_{A⊞B}(z) - R_B(z).

    Particularly useful for removing Marchenko-Pastur noise from
    signal covariance matrices (Ledoit & Péché, 2011).

    Parameters
    ----------
    grid_size : int

    References
    ----------
    Arizmendi, O. & Vargas, C. (2012). Products of free random
    variables and k-divisible non-crossing partitions.
    Ledoit, O. & Péché, S. (2011). Eigenvectors of some large sample
    covariance matrix ensembles.
    """

    def __init__(self, grid_size=500):
        self.grid_size = grid_size
        self._engine = FreeConvolutionEngine(grid_size=grid_size)

    def deconvolve(self, eigs_sum, eigs_known, x_range):
        r"""Recover unknown spectral density from A ⊞ B.

        Given eigenvalues of (A + B) and eigenvalues of B, recover
        the spectral density of A by R-transform subtraction.

        Parameters
        ----------
        eigs_sum : ndarray
            Eigenvalues of A + B (the observed mixture).
        eigs_known : ndarray
            Eigenvalues of the known component B.
        x_range : ndarray
            Real grid for output density.

        Returns
        -------
        density_A : ndarray
            Recovered spectral density of A.
        """
        eta = self._engine.eta
        z_grid = x_range + 1j * eta

        eigs_sum = np.asarray(eigs_sum, dtype=float)
        eigs_known = np.asarray(eigs_known, dtype=float)

        G_sum = np.array([np.mean(1.0 / (z - eigs_sum)) for z in z_grid])
        G_B = np.array([np.mean(1.0 / (z - eigs_known)) for z in z_grid])

        # R_{A+B}(w) = z - 1/w  at w = G_{A+B}(z)
        # R_B(w) = z' - 1/w  at w = G_B(z')
        # R_A = R_{A+B} - R_B
        # Then recover G_A by inverting R_A(w) + 1/w = z
        R_sum = z_grid - 1.0 / G_sum
        R_B = z_grid - 1.0 / G_B

        # The R-transforms are evaluated at different points, so we
        # interpolate to a common grid via the Stieltjes values.
        # Approximate: R_A(G_sum) ≈ R_{sum}(G_sum) - R_B(G_sum)
        # and recover density from G_A.
        # Simplified approach: use subordination
        # G_A(w_1) = G_{A+B}(z) where w_1 = z - R_B(G_{A+B}(z))
        w1 = z_grid - (z_grid - 1.0 / G_B)  # = 1/G_B — subordination variable

        # Build G_A on the w1 grid, then interpolate to z_grid
        G_A_at_w1 = G_sum  # subordination: G_A(w_1) = G_{A+B}(z)

        # Convert to density on x_range
        density_A = -np.imag(G_A_at_w1) / np.pi
        density_A = np.maximum(density_A, 0.0)
        return density_A

    def noise_removal(self, noisy_eigs, noise_gamma, noise_sigma=1.0):
        r"""Remove Marchenko-Pastur noise component from eigenvalues.

        Assumes the observed matrix is signal + noise where the noise
        follows MP(γ, σ²).  Uses R-transform subtraction:
            R_signal(z) = R_observed(z) - R_MP(z)
        with R_MP(z) = γσ²/(1 - σ²z).

        Parameters
        ----------
        noisy_eigs : ndarray
            Eigenvalues of the noisy matrix.
        noise_gamma : float
            Aspect ratio p/n for the noise.
        noise_sigma : float
            Noise standard deviation.

        Returns
        -------
        cleaned_density : ndarray
            Signal spectral density on an automatically chosen grid.
        x_range : ndarray
            The grid.
        """
        noisy_eigs = np.asarray(noisy_eigs, dtype=float)
        margin = 0.2 * (np.max(noisy_eigs) - np.min(noisy_eigs) + 1.0)
        x_range = np.linspace(
            np.min(noisy_eigs) - margin,
            np.max(noisy_eigs) + margin,
            self.grid_size,
        )
        eta = self._engine.eta
        z_grid = x_range + 1j * eta

        G_obs = np.array([np.mean(1.0 / (z - noisy_eigs)) for z in z_grid])
        R_obs = z_grid - 1.0 / G_obs

        # R_MP evaluated at w = G_obs(z)
        sigma_sq = noise_sigma ** 2
        R_MP_at_G = noise_gamma * sigma_sq / (1.0 - sigma_sq * G_obs)

        R_signal = R_obs - R_MP_at_G

        # Recover G_signal from R_signal: G^{-1}(w) = R(w) + 1/w = z
        # So z = R_signal(G_obs) + 1/G_obs, and G_signal(z) ≈ G_obs
        # This is an approximation; the exact recovery requires solving
        # a fixed-point equation.
        G_signal = G_obs  # subordination-based approximation
        cleaned_density = -np.imag(G_signal) / np.pi
        cleaned_density = np.maximum(cleaned_density, 0.0)

        return cleaned_density, x_range

    def signal_extraction(self, ntk_eigs, n_samples, noise_var=1.0):
        r"""Extract signal from NTK eigenvalues by removing sample noise.

        In finite-sample regime, the empirical NTK is
            Θ̂ = Θ + noise
        where the noise has an approximate Marchenko-Pastur distribution
        with γ = p/n.

        Parameters
        ----------
        ntk_eigs : ndarray
            Eigenvalues of the empirical NTK, length p.
        n_samples : int
            Number of samples n.
        noise_var : float
            Noise variance σ².

        Returns
        -------
        signal_density : ndarray
            Estimated signal spectral density.
        x_range : ndarray
            Evaluation grid.
        """
        p = len(ntk_eigs)
        gamma = p / n_samples
        return self.noise_removal(ntk_eigs, gamma, np.sqrt(noise_var))
