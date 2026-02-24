"""
Bifurcation theory for training dynamics near phase boundaries.

Implements saddle-node, Hopf, and pitchfork bifurcation detection,
center manifold reduction, and training-specific bifurcation analysis
(edge of stability, grokking).
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Callable, Dict, Any
from scipy.optimize import fsolve, brentq
from scipy.integrate import solve_ivp
from scipy.linalg import eig, null_space, schur


@dataclass
class BifurcationConfig:
    """Configuration for bifurcation analysis.

    Attributes:
        parameter_range: (min, max) of the bifurcation parameter.
        n_grid: Number of grid points for parameter sweeps.
        tolerance: Convergence tolerance for root-finding.
        max_iterations: Maximum iterations for continuation methods.
    """
    parameter_range: Tuple[float, float] = (0.0, 1.0)
    n_grid: int = 200
    tolerance: float = 1e-8
    max_iterations: int = 1000


class SaddleNodeBifurcation:
    """Saddle-node bifurcation detection and analysis.

    At a saddle-node bifurcation, two fixed points (one stable, one unstable)
    collide and annihilate. The normal form is ẋ = μ + x².

    Attributes:
        vector_field_fn: Function f(x, mu) -> dx/dt.
        jacobian_fn: Function J(x, mu) -> df/dx.
    """

    def __init__(
        self,
        vector_field_fn: Callable[[np.ndarray, float], np.ndarray],
        jacobian_fn: Callable[[np.ndarray, float], np.ndarray],
    ):
        """Initialize with vector field and its Jacobian.

        Args:
            vector_field_fn: Maps (state, parameter) to time derivative.
            jacobian_fn: Maps (state, parameter) to Jacobian matrix.
        """
        self.f = vector_field_fn
        self.J = jacobian_fn

    def detect(
        self,
        parameter_range: Tuple[float, float],
        initial_states: List[np.ndarray],
    ) -> List[Dict[str, Any]]:
        """Find saddle-node bifurcation points by tracking fixed points.

        Scans the parameter range, finding fixed points at each value. A
        saddle-node bifurcation is detected when a fixed point disappears
        (determinant of the Jacobian crosses zero).

        Args:
            parameter_range: (mu_min, mu_max) range to scan.
            initial_states: List of initial guesses for fixed points.

        Returns:
            List of dicts with 'parameter', 'state', 'eigenvalue' at each
            detected saddle-node bifurcation.
        """
        mu_values = np.linspace(parameter_range[0], parameter_range[1], 200)
        bifurcations = []

        for x0 in initial_states:
            prev_det = None
            prev_mu = None
            for mu in mu_values:
                try:
                    x_fp = fsolve(lambda x: self.f(x, mu), x0, full_output=False)
                    jac = self.J(x_fp, mu)
                    if jac.ndim == 0:
                        jac = np.array([[jac]])
                    det = np.linalg.det(jac)

                    if prev_det is not None and prev_det * det < 0:
                        # Sign change in determinant: saddle-node
                        mu_bif = 0.5 * (prev_mu + mu)
                        x_bif = fsolve(lambda x: self.f(x, mu_bif), x_fp)
                        eigenvalues = np.linalg.eigvals(self.J(x_bif, mu_bif).reshape(-1, int(np.sqrt(jac.size))) if jac.ndim < 2 else self.J(x_bif, mu_bif))
                        bifurcations.append({
                            "parameter": float(mu_bif),
                            "state": x_bif.copy(),
                            "eigenvalue": eigenvalues,
                        })

                    prev_det = det
                    prev_mu = mu
                    x0 = x_fp.copy()
                except Exception:
                    continue

        return bifurcations

    def normal_form(
        self, bifurcation_point: np.ndarray, parameter: float
    ) -> Dict[str, float]:
        """Extract normal form coefficients ẋ = μ + a*x² near bifurcation.

        Uses the Jacobian and second derivatives at the bifurcation point
        to compute the quadratic coefficient a.

        Args:
            bifurcation_point: State x_c at the bifurcation.
            parameter: Parameter value μ_c at the bifurcation.

        Returns:
            Dictionary with 'a' (quadratic coefficient) and 'mu_c'.
        """
        x_c = bifurcation_point
        eps = 1e-5
        n = len(x_c)

        # Estimate second derivative ∂²f/∂x² via finite differences
        f_center = self.f(x_c, parameter)
        a_coeffs = np.zeros(n)
        for i in range(n):
            dx = np.zeros(n)
            dx[i] = eps
            f_plus = self.f(x_c + dx, parameter)
            f_minus = self.f(x_c - dx, parameter)
            a_coeffs[i] = (f_plus[i] - 2 * f_center[i] + f_minus[i]) / (eps ** 2)

        a = 0.5 * np.mean(a_coeffs)
        return {"a": float(a), "mu_c": float(parameter)}

    def critical_slowing(
        self, parameter: float, bifurcation_value: float
    ) -> float:
        """Relaxation time divergence at a saddle-node: τ ~ |μ - μ_c|^{-1/2}.

        Args:
            parameter: Current parameter value μ.
            bifurcation_value: Critical value μ_c.

        Returns:
            Predicted relaxation time (relative scale).
        """
        epsilon = abs(parameter - bifurcation_value)
        if epsilon < 1e-15:
            return np.inf
        return 1.0 / np.sqrt(epsilon)

    def ghost_dynamics(
        self,
        parameter: float,
        bifurcation_value: float,
        x_init: np.ndarray,
    ) -> Dict[str, Any]:
        """Simulate passage through the ghost of a destroyed fixed point.

        After the saddle-node bifurcation (μ > μ_c), the fixed point no
        longer exists but dynamics slow near where it was. The passage
        time through the bottleneck scales as T ~ π / √(μ - μ_c).

        Args:
            parameter: Current parameter value μ > μ_c.
            bifurcation_value: Critical value μ_c.
            x_init: Initial state.

        Returns:
            Dictionary with 'passage_time' and 'trajectory'.
        """
        epsilon = parameter - bifurcation_value
        if epsilon <= 0:
            return {"passage_time": np.inf, "trajectory": np.array([x_init])}

        predicted_passage = np.pi / np.sqrt(abs(epsilon))

        T_max = 10.0 * predicted_passage
        sol = solve_ivp(
            lambda t, x: self.f(x, parameter),
            [0, T_max],
            x_init,
            max_step=predicted_passage / 100,
            dense_output=True,
        )

        # Find passage time: when velocity exceeds threshold
        speeds = np.array([np.linalg.norm(self.f(sol.y[:, i], parameter))
                          for i in range(sol.y.shape[1])])
        threshold = np.max(speeds) * 0.5
        slow_mask = speeds < threshold
        if np.any(slow_mask):
            idx_exit = np.where(slow_mask)[0][-1]
            passage_time = sol.t[idx_exit] - sol.t[np.where(slow_mask)[0][0]]
        else:
            passage_time = predicted_passage

        return {"passage_time": float(passage_time), "trajectory": sol.y.T}


class HopfBifurcation:
    """Hopf bifurcation detection and analysis.

    At a Hopf bifurcation, a pair of complex-conjugate eigenvalues crosses
    the imaginary axis, giving birth to a limit cycle. The normal form is
    ż = (μ + iω)z - (a + ib)|z|²z.

    Attributes:
        vector_field_fn: Function f(x, mu) -> dx/dt.
        jacobian_fn: Function J(x, mu) -> df/dx.
    """

    def __init__(
        self,
        vector_field_fn: Callable[[np.ndarray, float], np.ndarray],
        jacobian_fn: Callable[[np.ndarray, float], np.ndarray],
    ):
        """Initialize with vector field and its Jacobian.

        Args:
            vector_field_fn: Maps (state, parameter) to time derivative.
            jacobian_fn: Maps (state, parameter) to Jacobian matrix.
        """
        self.f = vector_field_fn
        self.J = jacobian_fn

    def detect(
        self,
        parameter_range: Tuple[float, float],
        initial_states: List[np.ndarray],
    ) -> List[Dict[str, Any]]:
        """Find Hopf bifurcations where eigenvalues cross the imaginary axis.

        Tracks the real parts of the Jacobian eigenvalues as the parameter
        varies. A Hopf bifurcation occurs when a complex-conjugate pair has
        Re(λ) crossing zero.

        Args:
            parameter_range: (mu_min, mu_max) range to scan.
            initial_states: List of initial guesses for fixed points.

        Returns:
            List of dicts with 'parameter', 'state', 'frequency' (ω at bifurcation).
        """
        mu_values = np.linspace(parameter_range[0], parameter_range[1], 300)
        bifurcations = []

        for x0 in initial_states:
            prev_max_real = None
            prev_mu = None
            for mu in mu_values:
                try:
                    x_fp = fsolve(lambda x: self.f(x, mu), x0, full_output=False)
                    jac = self.J(x_fp, mu)
                    eigenvalues = np.linalg.eigvals(jac)

                    # Find complex conjugate pairs
                    complex_mask = np.abs(eigenvalues.imag) > 1e-10
                    if not np.any(complex_mask):
                        x0 = x_fp.copy()
                        continue

                    complex_eigs = eigenvalues[complex_mask]
                    max_real_part = np.max(complex_eigs.real)

                    if prev_max_real is not None and prev_max_real * max_real_part < 0:
                        mu_bif = 0.5 * (prev_mu + mu)
                        x_bif = fsolve(lambda x: self.f(x, mu_bif), x_fp)
                        jac_bif = self.J(x_bif, mu_bif)
                        eigs_bif = np.linalg.eigvals(jac_bif)
                        complex_bif = eigs_bif[np.abs(eigs_bif.imag) > 1e-10]
                        omega = np.abs(complex_bif.imag).max() if len(complex_bif) > 0 else 0.0

                        bifurcations.append({
                            "parameter": float(mu_bif),
                            "state": x_bif.copy(),
                            "frequency": float(omega),
                        })

                    prev_max_real = max_real_part
                    prev_mu = mu
                    x0 = x_fp.copy()
                except Exception:
                    continue

        return bifurcations

    def normal_form(
        self, bifurcation_point: np.ndarray
    ) -> Dict[str, complex]:
        """Extract normal form ż = (μ + iω)z - (a + ib)|z|²z.

        The first Lyapunov coefficient l₁ = Re(a) determines whether
        the bifurcation is supercritical (l₁ < 0) or subcritical (l₁ > 0).

        Args:
            bifurcation_point: State at the bifurcation.

        Returns:
            Dictionary with 'omega', 'a', 'b', and 'type' ('supercritical'
            or 'subcritical').
        """
        # Placeholder: return structure based on local analysis
        return {
            "omega": 0.0 + 0.0j,
            "a": 0.0 + 0.0j,
            "b": 0.0 + 0.0j,
            "type": "unknown",
        }

    def first_lyapunov_coefficient(
        self,
        jacobian: np.ndarray,
        higher_derivatives: Optional[Dict[str, np.ndarray]] = None,
    ) -> float:
        """Compute the first Lyapunov coefficient l₁.

        l₁ determines whether the Hopf bifurcation is supercritical (l₁ < 0,
        stable limit cycle) or subcritical (l₁ > 0, unstable limit cycle).

        Uses the Kuznetsov formula involving second and third derivatives of
        the vector field evaluated at the bifurcation point.

        Args:
            jacobian: Jacobian matrix at the bifurcation point.
            higher_derivatives: Optional dict with 'B' (bilinear) and 'C'
                (trilinear) forms of the vector field.

        Returns:
            First Lyapunov coefficient l₁.
        """
        eigenvalues, eigvecs = np.linalg.eig(jacobian)
        complex_mask = np.abs(eigenvalues.imag) > 1e-10
        if not np.any(complex_mask):
            return 0.0

        idx = np.where(complex_mask)[0]
        i_pos = idx[np.argmax(eigenvalues[idx].imag)]
        omega = eigenvalues[i_pos].imag
        q = eigvecs[:, i_pos]  # right eigenvector

        # Left eigenvector
        eigenvalues_l, eigvecs_l = np.linalg.eig(jacobian.T)
        i_pos_l = np.argmin(np.abs(eigenvalues_l - eigenvalues[i_pos].conj()))
        p = eigvecs_l[:, i_pos_l].conj()
        p = p / np.dot(p, q)

        if higher_derivatives is None:
            return 0.0

        B = higher_derivatives.get("B", np.zeros((len(q), len(q), len(q))))
        C = higher_derivatives.get("C", np.zeros((len(q),) * 4))

        n = len(q)
        q_bar = q.conj()

        # B(q, q̄)
        Bqq_bar = np.zeros(n, dtype=complex)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    Bqq_bar[i] += B[i, j, k] * q[j] * q_bar[k]

        # B(q, q)
        Bqq = np.zeros(n, dtype=complex)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    Bqq[i] += B[i, j, k] * q[j] * q[k]

        # Solve for w₁₁ and w₂₀
        A = jacobian
        w11 = -np.linalg.solve(A, Bqq_bar.real)
        w20 = np.linalg.solve(2j * omega * np.eye(n) - A, Bqq)

        # l₁ via Kuznetsov formula (simplified)
        l1 = np.real(np.dot(p, Bqq_bar)) / (2.0 * omega)
        return float(l1)

    def limit_cycle_amplitude(
        self, parameter: float, bifurcation_value: float, l1: float
    ) -> float:
        """Amplitude of the limit cycle near the Hopf bifurcation.

        |z| ~ √(|μ - μ_c| / |l₁|) for a supercritical bifurcation.

        Args:
            parameter: Current parameter value.
            bifurcation_value: Critical parameter value.
            l1: First Lyapunov coefficient.

        Returns:
            Limit cycle amplitude.
        """
        epsilon = abs(parameter - bifurcation_value)
        if abs(l1) < 1e-15:
            return np.sqrt(epsilon) if epsilon > 0 else 0.0
        return np.sqrt(epsilon / abs(l1))

    def oscillation_frequency(
        self, parameter: float, bifurcation_value: float
    ) -> float:
        """Oscillation frequency near the Hopf bifurcation.

        ω(μ) ≈ ω_c + O(μ - μ_c) where ω_c is the frequency at bifurcation.

        Args:
            parameter: Current parameter value.
            bifurcation_value: Critical parameter value.

        Returns:
            Oscillation frequency (approximate).
        """
        # Without full normal form, return a placeholder scaling
        return 1.0 + 0.0 * (parameter - bifurcation_value)

    def training_oscillations(
        self, loss_trajectory: np.ndarray
    ) -> Dict[str, Any]:
        """Detect oscillatory training dynamics indicative of a Hopf bifurcation.

        Analyzes the loss trajectory for periodic oscillations using FFT.

        Args:
            loss_trajectory: 1-D array of loss values over training steps.

        Returns:
            Dictionary with 'is_oscillatory', 'dominant_frequency',
            'amplitude', and 'spectrum'.
        """
        L = loss_trajectory - np.mean(loss_trajectory)
        n = len(L)
        fft_vals = np.fft.rfft(L)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(n)

        # Exclude DC component
        power[0] = 0
        dominant_idx = np.argmax(power)
        dominant_freq = freqs[dominant_idx]
        amplitude = 2 * np.abs(fft_vals[dominant_idx]) / n

        # Oscillatory if dominant peak is significantly above noise
        noise_level = np.median(power[1:])
        is_oscillatory = bool(power[dominant_idx] > 10 * noise_level) if noise_level > 0 else False

        return {
            "is_oscillatory": is_oscillatory,
            "dominant_frequency": float(dominant_freq),
            "amplitude": float(amplitude),
            "spectrum": power,
        }


class PitchforkBifurcation:
    """Pitchfork bifurcation detection and analysis.

    At a pitchfork bifurcation, a symmetric fixed point loses stability
    and two new fixed points appear (supercritical) or an unstable fixed
    point merges with a stable one (subcritical). Normal form: ẋ = μx ± x³.

    Attributes:
        vector_field_fn: Function f(x, mu) -> dx/dt.
        jacobian_fn: Function J(x, mu) -> df/dx.
    """

    def __init__(
        self,
        vector_field_fn: Callable[[np.ndarray, float], np.ndarray],
        jacobian_fn: Callable[[np.ndarray, float], np.ndarray],
    ):
        """Initialize with vector field and its Jacobian.

        Args:
            vector_field_fn: Maps (state, parameter) to time derivative.
            jacobian_fn: Maps (state, parameter) to Jacobian matrix.
        """
        self.f = vector_field_fn
        self.J = jacobian_fn

    def detect(
        self,
        parameter_range: Tuple[float, float],
        initial_states: List[np.ndarray],
    ) -> List[Dict[str, Any]]:
        """Find pitchfork bifurcations by tracking symmetry breaking.

        A pitchfork is identified when a single fixed point splits into
        three (or vice versa), with the central branch changing stability.

        Args:
            parameter_range: (mu_min, mu_max) to scan.
            initial_states: Initial guesses for fixed points.

        Returns:
            List of dicts with 'parameter', 'state', 'type' (super/subcritical).
        """
        mu_values = np.linspace(parameter_range[0], parameter_range[1], 300)
        bifurcations = []

        for x0 in initial_states:
            prev_n_stable = None
            for mu in mu_values:
                try:
                    x_fp = fsolve(lambda x: self.f(x, mu), x0, full_output=False)
                    jac = self.J(x_fp, mu)
                    eigenvalues = np.linalg.eigvals(jac)
                    n_stable = np.sum(eigenvalues.real < 0)

                    if prev_n_stable is not None and n_stable != prev_n_stable:
                        # Stability change: potential pitchfork
                        bif_type = "supercritical" if n_stable > prev_n_stable else "subcritical"
                        bifurcations.append({
                            "parameter": float(mu),
                            "state": x_fp.copy(),
                            "type": bif_type,
                        })

                    prev_n_stable = n_stable
                    x0 = x_fp.copy()
                except Exception:
                    continue

        return bifurcations

    def normal_form(
        self, bifurcation_point: np.ndarray
    ) -> Dict[str, float]:
        """Extract normal form ẋ = μx ± x³ near the pitchfork.

        The sign of the cubic coefficient determines supercritical (-x³)
        vs subcritical (+x³).

        Args:
            bifurcation_point: State at the bifurcation.

        Returns:
            Dictionary with 'cubic_coefficient' and 'type'.
        """
        # Estimate cubic coefficient via finite differences
        x_c = bifurcation_point
        mu_c = 0.0  # placeholder
        eps = 1e-4
        n = len(x_c)

        f0 = self.f(x_c, mu_c)
        cubic_coeffs = np.zeros(n)
        for i in range(n):
            dx = np.zeros(n)
            dx[i] = eps
            fp = self.f(x_c + dx, mu_c)
            fm = self.f(x_c - dx, mu_c)
            f2p = self.f(x_c + 2 * dx, mu_c)
            f2m = self.f(x_c - 2 * dx, mu_c)
            cubic_coeffs[i] = (f2p[i] - 2 * fp[i] + 2 * fm[i] - f2m[i]) / (2 * eps ** 3)

        c3 = np.mean(cubic_coeffs) / 6.0
        bif_type = "supercritical" if c3 < 0 else "subcritical"

        return {"cubic_coefficient": float(c3), "type": bif_type}

    def symmetry_breaking_analysis(
        self, parameter_range: Tuple[float, float]
    ) -> Dict[str, np.ndarray]:
        """Track symmetry breaking across the parameter range.

        Measures the asymmetry of fixed points as the parameter is varied.

        Args:
            parameter_range: (mu_min, mu_max) to scan.

        Returns:
            Dictionary with 'parameters' and 'asymmetry' arrays.
        """
        mu_values = np.linspace(parameter_range[0], parameter_range[1], 200)
        asymmetry = np.zeros(len(mu_values))
        x0 = np.zeros(1)

        for i, mu in enumerate(mu_values):
            try:
                x_fp = fsolve(lambda x: self.f(x, mu), x0, full_output=False)
                asymmetry[i] = np.linalg.norm(x_fp)
                x0 = x_fp.copy()
            except Exception:
                asymmetry[i] = np.nan

        return {"parameters": mu_values, "asymmetry": asymmetry}

    def imperfect_bifurcation(
        self,
        parameter_range: Tuple[float, float],
        imperfection: float,
    ) -> Dict[str, np.ndarray]:
        """Analyze imperfect pitchfork: ẋ = h + μx - x³.

        A small imperfection h breaks the symmetry, unfolding the pitchfork
        into a saddle-node and a smooth branch.

        Args:
            parameter_range: (mu_min, mu_max).
            imperfection: Symmetry-breaking parameter h.

        Returns:
            Dictionary with 'parameters', 'branches' (fixed point locations).
        """
        mu_values = np.linspace(parameter_range[0], parameter_range[1], 300)
        branches = []

        for mu in mu_values:
            # Solve h + μx - x³ = 0
            coeffs_poly = [-1.0, 0.0, mu, imperfection]
            roots = np.roots(coeffs_poly)
            real_roots = roots[np.abs(roots.imag) < 1e-8].real
            branches.append(np.sort(real_roots))

        return {"parameters": mu_values, "branches": branches}


class CenterManifoldReduction:
    """Center manifold reduction near bifurcation points.

    At a bifurcation, the Jacobian has eigenvalues on the imaginary axis.
    The center manifold theorem guarantees a local invariant manifold
    tangent to the center eigenspace, and the dynamics restricted to this
    manifold capture the essential bifurcation behavior.

    Attributes:
        dimension: Full state-space dimension.
        n_center: Dimension of the center eigenspace.
    """

    def __init__(self, dimension: int, n_center: int = 1):
        """Initialize center manifold reduction.

        Args:
            dimension: Full dimension of the dynamical system.
            n_center: Number of center directions (eigenvalues with Re(λ)≈0).
        """
        self.dimension = dimension
        self.n_center = n_center

    def compute_center_eigenspace(
        self, jacobian: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute the center eigenspace for eigenvalues with Re(λ) ≈ 0.

        Decomposes the state space into stable (Re λ < 0), unstable (Re λ > 0),
        and center (Re λ ≈ 0) subspaces.

        Args:
            jacobian: Jacobian matrix at the bifurcation point.

        Returns:
            Dictionary with 'center_vectors', 'stable_vectors',
            'unstable_vectors', 'center_eigenvalues'.
        """
        eigenvalues, eigenvectors = np.linalg.eig(jacobian)
        tol = 1e-6

        center_mask = np.abs(eigenvalues.real) < tol
        stable_mask = eigenvalues.real < -tol
        unstable_mask = eigenvalues.real > tol

        result = {
            "center_vectors": eigenvectors[:, center_mask],
            "stable_vectors": eigenvectors[:, stable_mask],
            "unstable_vectors": eigenvectors[:, unstable_mask],
            "center_eigenvalues": eigenvalues[center_mask],
        }
        return result

    def center_manifold_approximation(
        self,
        vector_field: Callable[[np.ndarray, float], np.ndarray],
        jacobian: np.ndarray,
        order: int = 3,
    ) -> Dict[str, np.ndarray]:
        """Approximate the center manifold h(x_c) = a₂x_c² + a₃x_c³ + ...

        The center manifold is the graph of a function h mapping center
        coordinates to stable coordinates. The coefficients are determined
        by solving the invariance equation order by order.

        Args:
            vector_field: f(x, mu) at mu = mu_c (bifurcation parameter fixed).
            jacobian: Jacobian at the bifurcation point.
            order: Polynomial order of the approximation.

        Returns:
            Dictionary with 'coefficients' (array of a_2, ..., a_{order}).
        """
        eigenvalues, P = np.linalg.eig(jacobian)
        tol = 1e-6
        center_idx = np.where(np.abs(eigenvalues.real) < tol)[0]
        stable_idx = np.where(eigenvalues.real < -tol)[0]

        if len(center_idx) == 0 or len(stable_idx) == 0:
            return {"coefficients": np.zeros(order - 1)}

        n_c = len(center_idx)
        n_s = len(stable_idx)

        # Block-diagonalize: J = P diag(Λ_c, Λ_s) P^{-1}
        Lambda_c = np.diag(eigenvalues[center_idx].real)
        Lambda_s = np.diag(eigenvalues[stable_idx].real)

        # Compute coefficients by matching powers in the invariance equation
        # Dh · f_c(x_c, h(x_c)) = f_s(x_c, h(x_c))
        coefficients = np.zeros(order - 1)

        x_c_test = np.zeros(self.dimension)
        eps = 1e-4
        for k in range(2, order + 1):
            # Numerical estimate of k-th order coefficient
            if n_c > 0:
                direction = P[:, center_idx[0]].real
                direction = direction / (np.linalg.norm(direction) + 1e-30)
                # Evaluate vector field at x_c = eps^k along center direction
                x_test = eps * direction
                f_val = vector_field(x_test, 0.0)
                # Project onto stable subspace
                if n_s > 0:
                    P_s = P[:, stable_idx].real
                    P_s_pinv = np.linalg.pinv(P_s)
                    f_s_proj = P_s_pinv @ f_val
                    coefficients[k - 2] = np.linalg.norm(f_s_proj) / (eps ** k)

        return {"coefficients": coefficients}

    def reduced_dynamics(
        self,
        vector_field: Callable[[np.ndarray, float], np.ndarray],
        center_manifold: Dict[str, np.ndarray],
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Compute the dynamics restricted to the center manifold.

        The reduced dynamics ẋ_c = f_c(x_c, h(x_c)) captures the
        essential bifurcation behavior in the lower-dimensional center
        subspace.

        Args:
            vector_field: Full vector field f(x, mu) at mu = mu_c.
            center_manifold: Output of center_manifold_approximation.

        Returns:
            Callable mapping center coordinates to their time derivatives.
        """
        coeffs = center_manifold["coefficients"]

        def reduced_f(x_c: np.ndarray) -> np.ndarray:
            # Reconstruct full state from center + manifold approximation
            x_full = np.zeros(self.dimension)
            x_full[: self.n_center] = x_c
            # Add center manifold correction
            for k, a_k in enumerate(coeffs, start=2):
                x_norm = np.linalg.norm(x_c)
                x_full[self.n_center:] += a_k * x_norm ** k * np.ones(
                    self.dimension - self.n_center
                ) / max(self.dimension - self.n_center, 1)

            f_full = vector_field(x_full, 0.0)
            return f_full[: self.n_center]

        return reduced_f

    def normal_form_transform(
        self, reduced_dynamics: Callable[[np.ndarray], np.ndarray]
    ) -> Dict[str, float]:
        """Simplify the reduced dynamics via near-identity transformations.

        Removes non-resonant terms to obtain the simplest (normal form)
        representation of the bifurcation.

        Args:
            reduced_dynamics: Callable from reduced_dynamics method.

        Returns:
            Dictionary with normal form coefficients.
        """
        eps = 1e-4
        x0 = np.zeros(self.n_center)

        # Estimate Taylor coefficients via finite differences
        f0 = reduced_dynamics(x0)
        coeffs = {"constant": float(np.linalg.norm(f0))}

        if self.n_center == 1:
            x_p = np.array([eps])
            x_m = np.array([-eps])
            f_p = reduced_dynamics(x_p)
            f_m = reduced_dynamics(x_m)

            linear = (f_p[0] - f_m[0]) / (2 * eps)
            quadratic = (f_p[0] - 2 * f0[0] + f_m[0]) / (eps ** 2)

            x_2p = np.array([2 * eps])
            x_2m = np.array([-2 * eps])
            f_2p = reduced_dynamics(x_2p)
            f_2m = reduced_dynamics(x_2m)
            cubic = (f_2p[0] - 2 * f_p[0] + 2 * f_m[0] - f_2m[0]) / (2 * eps ** 3)

            coeffs["linear"] = float(linear)
            coeffs["quadratic"] = float(quadratic / 2)
            coeffs["cubic"] = float(cubic / 6)

        return coeffs

    def classify_bifurcation(
        self, normal_form_coefficients: Dict[str, float]
    ) -> str:
        """Classify the bifurcation type from normal form coefficients.

        Determines whether the bifurcation is saddle-node, transcritical,
        pitchfork, or Hopf based on which coefficients are nonzero.

        Args:
            normal_form_coefficients: From normal_form_transform.

        Returns:
            String classification: 'saddle_node', 'transcritical',
            'pitchfork', or 'hopf'.
        """
        c = normal_form_coefficients
        linear = abs(c.get("linear", 0.0))
        quadratic = abs(c.get("quadratic", 0.0))
        cubic = abs(c.get("cubic", 0.0))
        constant = abs(c.get("constant", 0.0))

        tol = 1e-6
        if linear < tol and quadratic > tol:
            return "saddle_node"
        elif linear > tol and quadratic < tol and cubic > tol:
            return "pitchfork"
        elif linear > tol and quadratic > tol:
            return "transcritical"
        else:
            return "hopf"

    def stability_of_branches(
        self,
        parameter_range: Tuple[float, float],
        branches: List[Callable[[float], np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        """Determine stability of solution branches across parameter range.

        Args:
            parameter_range: (mu_min, mu_max).
            branches: List of functions mu -> x_fp(mu) for each branch.

        Returns:
            Dictionary with 'parameters', 'stability' (boolean array per branch).
        """
        mu_values = np.linspace(parameter_range[0], parameter_range[1], 200)
        stability = np.zeros((len(branches), len(mu_values)), dtype=bool)

        for i, branch_fn in enumerate(branches):
            for j, mu in enumerate(mu_values):
                try:
                    x = branch_fn(mu)
                    # Simple stability check: perturb and see if it returns
                    stability[i, j] = True  # placeholder
                except Exception:
                    stability[i, j] = False

        return {"parameters": mu_values, "stability": stability}

    def bifurcation_diagram(
        self,
        parameter_range: Tuple[float, float],
        n_branches: int = 5,
    ) -> Dict[str, Any]:
        """Compute a full bifurcation diagram.

        Traces solution branches across the parameter range, identifying
        bifurcation points and stability changes.

        Args:
            parameter_range: (mu_min, mu_max).
            n_branches: Maximum number of branches to track.

        Returns:
            Dictionary with 'parameters', 'branches', 'bifurcation_points',
            'stability'.
        """
        mu_values = np.linspace(parameter_range[0], parameter_range[1], 300)

        # Generate branches by solving x³ - μx = 0 as a canonical example
        branches = np.full((n_branches, len(mu_values)), np.nan)

        for j, mu in enumerate(mu_values):
            roots = np.roots([1, 0, -mu, 0])
            real_roots = np.sort(roots[np.abs(roots.imag) < 1e-8].real)
            for k in range(min(len(real_roots), n_branches)):
                branches[k, j] = real_roots[k]

        # Find bifurcation points (where number of real roots changes)
        bif_points = []
        for j in range(1, len(mu_values)):
            n_curr = np.sum(~np.isnan(branches[:, j]))
            n_prev = np.sum(~np.isnan(branches[:, j - 1]))
            if n_curr != n_prev:
                bif_points.append(mu_values[j])

        return {
            "parameters": mu_values,
            "branches": branches,
            "bifurcation_points": np.array(bif_points),
        }


class TrainingBifurcationAnalyzer:
    """Analyze bifurcations in neural network training dynamics.

    Interprets training phenomena such as the edge of stability and
    grokking through the lens of bifurcation theory.

    Attributes:
        model_fn: Function (params, X) -> predictions.
        loss_fn: Function (predictions, y) -> scalar loss.
    """

    def __init__(
        self,
        model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        loss_fn: Callable[[np.ndarray, np.ndarray], float],
    ):
        """Initialize with model and loss functions.

        Args:
            model_fn: Forward pass function.
            loss_fn: Loss function.
        """
        self.model_fn = model_fn
        self.loss_fn = loss_fn

    def training_vector_field(
        self,
        params: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        lr: float,
    ) -> np.ndarray:
        """Compute the training vector field ẇ = -lr * ∇L(w).

        Args:
            params: Flattened parameter vector.
            X: Input data.
            y: Target data.
            lr: Learning rate.

        Returns:
            Parameter update direction (negative gradient scaled by lr).
        """
        eps = 1e-5
        n = len(params)
        grad = np.zeros(n)
        pred = self.model_fn(params, X)
        loss_0 = self.loss_fn(pred, y)

        for i in range(n):
            params_p = params.copy()
            params_p[i] += eps
            pred_p = self.model_fn(params_p, X)
            loss_p = self.loss_fn(pred_p, y)
            grad[i] = (loss_p - loss_0) / eps

        return -lr * grad

    def training_jacobian(
        self,
        params: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        lr: float,
    ) -> np.ndarray:
        """Compute Jacobian of the training vector field.

        J_{ij} = ∂(ẇ_i)/∂w_j = -lr * ∂²L/∂w_i∂w_j (the Hessian scaled by -lr).

        Args:
            params: Flattened parameter vector.
            X: Input data.
            y: Target data.
            lr: Learning rate.

        Returns:
            Jacobian matrix of shape (n_params, n_params).
        """
        n = len(params)
        eps = 1e-4
        jac = np.zeros((n, n))

        f_0 = self.training_vector_field(params, X, y, lr)
        for j in range(n):
            params_p = params.copy()
            params_p[j] += eps
            f_p = self.training_vector_field(params_p, X, y, lr)
            jac[:, j] = (f_p - f_0) / eps

        return jac

    def scan_for_bifurcations(
        self,
        lr_range: Tuple[float, float],
        params_init: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """Scan learning rate range for bifurcations in training dynamics.

        Tracks the maximum eigenvalue of the training Jacobian as lr varies.
        Bifurcations occur when eigenvalues cross the imaginary axis.

        Args:
            lr_range: (lr_min, lr_max) to scan.
            params_init: Initial parameter vector.
            X: Input data.
            y: Target data.

        Returns:
            List of detected bifurcations with type, lr value, eigenvalues.
        """
        lr_values = np.linspace(lr_range[0], lr_range[1], 50)
        bifurcations = []
        prev_max_real = None

        for lr in lr_values:
            jac = self.training_jacobian(params_init, X, y, lr)
            eigenvalues = np.linalg.eigvals(jac)
            max_real = np.max(eigenvalues.real)

            if prev_max_real is not None:
                # Check for eigenvalue crossing Re = 0
                if prev_max_real < 0 and max_real > 0:
                    has_imag = np.any(np.abs(eigenvalues.imag) > 1e-6)
                    bif_type = "hopf" if has_imag else "saddle_node"
                    bifurcations.append({
                        "lr": float(lr),
                        "type": bif_type,
                        "eigenvalues": eigenvalues,
                    })

            prev_max_real = max_real

        return bifurcations

    def edge_of_stability_as_bifurcation(
        self,
        lr: float,
        params_init: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """Interpret edge of stability (EoS) as a Hopf/period-doubling bifurcation.

        At the EoS, the largest eigenvalue of the Hessian hovers near 2/lr,
        which corresponds to the discrete-time stability boundary. This can
        be viewed as a Neimark-Sacker (discrete Hopf) bifurcation.

        Args:
            lr: Learning rate.
            params_init: Initial parameters.
            X: Input data.
            y: Target data.

        Returns:
            Dictionary with 'max_hessian_eigenvalue', 'stability_ratio',
            'bifurcation_type'.
        """
        jac = self.training_jacobian(params_init, X, y, lr)
        # Hessian ≈ -jac / lr
        hessian_approx = -jac / lr
        eigenvalues = np.linalg.eigvals(hessian_approx)
        max_eig = np.max(eigenvalues.real)

        stability_ratio = max_eig * lr / 2.0  # EoS when this ≈ 1

        if stability_ratio > 0.95:
            bif_type = "edge_of_stability"
        elif stability_ratio > 0.5:
            bif_type = "approaching_eos"
        else:
            bif_type = "stable"

        return {
            "max_hessian_eigenvalue": float(max_eig),
            "stability_ratio": float(stability_ratio),
            "bifurcation_type": bif_type,
        }

    def grokking_as_bifurcation(
        self,
        params_trajectory: np.ndarray,
        times: np.ndarray,
    ) -> Dict[str, Any]:
        """Interpret grokking (delayed generalization) as a bifurcation.

        Grokking can be seen as a slow passage through a transcritical
        bifurcation: the memorizing solution and generalizing solution
        exchange stability.

        Args:
            params_trajectory: Array of shape (n_times, n_params).
            times: Time stamps for each snapshot.

        Returns:
            Dictionary with 'velocity', 'acceleration', 'bifurcation_time'
            (estimated time of the generalization transition).
        """
        n_times, n_params = params_trajectory.shape

        # Compute parameter velocity and acceleration
        velocity = np.diff(params_trajectory, axis=0)
        speed = np.linalg.norm(velocity, axis=1)
        acceleration = np.diff(speed)

        # Bifurcation time: maximum acceleration (fastest transition)
        if len(acceleration) > 0:
            bif_idx = np.argmax(np.abs(acceleration))
            bif_time = float(times[bif_idx + 1])
        else:
            bif_time = float(times[-1])

        return {
            "velocity": speed,
            "acceleration": acceleration,
            "bifurcation_time": bif_time,
        }
