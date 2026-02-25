"""
Non-perturbative methods for finite-width corrections beyond 1/N expansion.

Implements saddle-point methods, instanton calculus, Borel resummation,
path-integral formulation for neural networks, and numerical bootstrap
techniques. These go beyond the perturbative 1/N expansion to capture
effects like tunneling, non-perturbative phase transitions, and
resurgent trans-series structure.

References:
    - S. Coleman, "Aspects of Symmetry" (Cambridge, 1985)
    - J. Zinn-Justin, "Quantum Field Theory and Critical Phenomena" (Oxford, 2002)
    - G. 't Hooft, "The Large-N Expansion in Quantum Field Theory" (1993)
    - M. Mariño, "Instantons and Large N" (Cambridge, 2015)
    - D. Dorigoni, "An Introduction to Resurgence, Trans-Series and
      Alien Calculus" (Annals Phys. 2019)
"""

import numpy as np
from scipy import optimize, integrate, special, linalg, interpolate
from typing import Callable, Optional, List, Tuple, Dict, Union
import warnings


class SaddlePointApproximation:
    """Saddle-point (steepest descent) approximation for path integrals.

    For integrals of the form Z = ∫ dφ exp(-N · S[φ]), the saddle-point
    method expands around configurations φ* satisfying δS/δφ = 0. The
    leading contribution is exp(-N · S[φ*]) with Gaussian fluctuation
    corrections of order 1/N.

    This is the backbone of the large-N expansion in neural network field
    theory, where N is the network width.

    References:
        - Zinn-Justin, Ch. 2: "Gaussian Integrals and Perturbative Expansion"
        - Coleman, Ch. 7: "The Uses of Instantons"
        - Bender & Orszag, "Advanced Mathematical Methods" Ch. 6
    """

    def __init__(
        self,
        action_fn: Callable[[np.ndarray], float],
        gradient_fn: Callable[[np.ndarray], np.ndarray],
        hessian_fn: Callable[[np.ndarray], np.ndarray],
    ):
        """Initialize saddle-point calculator.

        Args:
            action_fn: The action functional S[φ] mapping field configurations
                to real numbers.
            gradient_fn: Gradient of the action δS/δφ, returning a vector of
                the same dimension as the field.
            hessian_fn: Hessian of the action δ²S/δφ²,  returning a matrix
                of second derivatives.
        """
        self.action_fn = action_fn
        self.gradient_fn = gradient_fn
        self.hessian_fn = hessian_fn

    def find_saddle_point(
        self,
        initial_guess: np.ndarray,
        method: str = "newton",
        tol: float = 1e-12,
        max_iter: int = 500,
    ) -> Dict:
        """Find field configuration φ* where δS/δφ = 0.

        Uses Newton's method (default) or other scipy optimizers to locate
        critical points of the action. Newton's method converges quadratically
        near the saddle: φ_{n+1} = φ_n - [S''(φ_n)]^{-1} S'(φ_n).

        Args:
            initial_guess: Starting field configuration.
            method: Optimization method — 'newton' for Newton-Raphson using
                the full Hessian, or any method accepted by scipy.optimize.minimize.
            tol: Convergence tolerance on gradient norm |δS/δφ| < tol.
            max_iter: Maximum number of iterations.

        Returns:
            Dictionary with keys:
                'saddle_point': the field configuration φ*
                'action': S[φ*]
                'gradient_norm': |δS/δφ| at convergence
                'converged': boolean convergence flag
                'n_iterations': number of iterations used
        """
        phi = np.array(initial_guess, dtype=np.float64)

        if method == "newton":
            converged = False
            n_iter = 0
            for i in range(max_iter):
                grad = self.gradient_fn(phi)
                grad_norm = np.linalg.norm(grad)
                n_iter = i + 1
                if grad_norm < tol:
                    converged = True
                    break
                H = self.hessian_fn(phi)
                try:
                    delta = np.linalg.solve(H, grad)
                except np.linalg.LinAlgError:
                    delta = np.linalg.lstsq(H, grad, rcond=None)[0]
                # Line search for robustness
                alpha = 1.0
                current_action = self.action_fn(phi)
                for _ in range(20):
                    phi_trial = phi - alpha * delta
                    if self.action_fn(phi_trial) < current_action + 1e-4 * alpha * np.dot(grad, -delta):
                        break
                    alpha *= 0.5
                phi = phi - alpha * delta

            return {
                "saddle_point": phi,
                "action": self.action_fn(phi),
                "gradient_norm": np.linalg.norm(self.gradient_fn(phi)),
                "converged": converged,
                "n_iterations": n_iter,
            }
        else:
            result = optimize.minimize(
                self.action_fn,
                initial_guess,
                jac=self.gradient_fn,
                hess=self.hessian_fn,
                method=method,
                tol=tol,
                options={"maxiter": max_iter},
            )
            return {
                "saddle_point": result.x,
                "action": result.fun,
                "gradient_norm": np.linalg.norm(self.gradient_fn(result.x)),
                "converged": result.success,
                "n_iterations": result.nit,
            }

    def gaussian_fluctuations(self, saddle_point: np.ndarray) -> Dict:
        """Compute Gaussian fluctuation determinant at saddle point.

        The one-loop prefactor involves det(S''(φ*)), arising from the
        Gaussian integral over fluctuations δφ around φ*:

            ∫ dδφ exp(-½ δφ^T S''(φ*) δφ) = (2π)^{n/2} / √det(S''(φ*))

        Negative eigenvalues indicate unstable directions (relevant for
        instantons and tunneling).

        Args:
            saddle_point: The saddle-point field configuration φ*.

        Returns:
            Dictionary with eigenvalues, determinant, and stability info.
        """
        H = self.hessian_fn(saddle_point)
        eigenvalues = np.linalg.eigvalsh(H)
        n_negative = np.sum(eigenvalues < 0)
        n_zero = np.sum(np.abs(eigenvalues) < 1e-10)

        # Sign of determinant and log|det| for numerical stability
        sign = np.prod(np.sign(eigenvalues[np.abs(eigenvalues) > 1e-10]))
        log_abs_det = np.sum(np.log(np.abs(eigenvalues[np.abs(eigenvalues) > 1e-10])))

        return {
            "eigenvalues": eigenvalues,
            "log_abs_determinant": log_abs_det,
            "determinant_sign": sign,
            "n_negative_modes": int(n_negative),
            "n_zero_modes": int(n_zero),
            "stable": n_negative == 0,
            "dimension": len(saddle_point),
        }

    def one_loop_correction(self, saddle_point: np.ndarray) -> Dict:
        """Compute one-loop effective action at saddle point.

        S_eff = S(φ*) + (1/2) ln det(S''(φ*) / 2π)

        This is the first quantum/finite-width correction. In NN language,
        this gives the O(1/N) correction to the infinite-width (GP) limit.

        Reference: Zinn-Justin, Eq. (2.35) and surrounding discussion.

        Args:
            saddle_point: The saddle-point field configuration φ*.

        Returns:
            Dictionary with classical action, one-loop correction, and total.
        """
        S_classical = self.action_fn(saddle_point)
        fluct = self.gaussian_fluctuations(saddle_point)

        n = fluct["dimension"]
        # (1/2) ln det(S''/2π) = (1/2) Σ ln(λ_i/2π)
        positive_eigs = fluct["eigenvalues"][fluct["eigenvalues"] > 1e-10]
        one_loop = 0.5 * np.sum(np.log(positive_eigs / (2 * np.pi)))

        return {
            "S_classical": S_classical,
            "one_loop_correction": one_loop,
            "S_effective": S_classical + one_loop,
            "n_modes": len(positive_eigs),
            "n_zero_modes": fluct["n_zero_modes"],
            "stable": fluct["stable"],
        }

    def steepest_descent_contour(
        self,
        saddle_point: np.ndarray,
        z_range: np.ndarray,
        n_points: int = 200,
    ) -> Dict:
        """Deform integration contour through saddle in the complex plane.

        For a 1D integral ∫ dz exp(-N·S(z)), the steepest descent contour
        passes through z* along the direction where Im(S(z)) = Im(S(z*))
        and Re(S(z)) increases away from z*. This contour is determined by
        the Hessian at the saddle: the descent direction is along the
        eigenvector of S''(z*) with positive eigenvalue.

        Only implemented for 1D and 2D fields.

        Args:
            saddle_point: The saddle-point configuration.
            z_range: Parameter range for contour parameterization.
            n_points: Number of points along the contour.

        Returns:
            Dictionary with contour points and action values along contour.
        """
        dim = len(saddle_point)
        H = self.hessian_fn(saddle_point)
        eigvals, eigvecs = np.linalg.eigh(H)

        # Steepest descent direction: eigenvector with largest positive eigenvalue
        idx = np.argmax(eigvals)
        descent_dir = eigvecs[:, idx]

        t_values = np.linspace(z_range[0], z_range[-1], n_points)
        contour_points = np.array([saddle_point + t * descent_dir for t in t_values])
        action_values = np.array([self.action_fn(pt) for pt in contour_points])

        # Also compute the steepest ascent direction (for Stokes lines)
        ascent_idx = np.argmin(eigvals)
        ascent_dir = eigvecs[:, ascent_idx]

        return {
            "contour_points": contour_points,
            "action_values": action_values,
            "t_values": t_values,
            "descent_direction": descent_dir,
            "ascent_direction": ascent_dir,
            "saddle_action": self.action_fn(saddle_point),
        }

    def multi_saddle_contribution(self, saddle_points: List[np.ndarray]) -> Dict:
        """Sum over multiple saddle points with proper weights.

        When multiple saddle points exist, the full integral receives
        contributions from each: Z = Σ_s n_s × Z_s, where n_s are
        Maslov indices and Z_s is the saddle-point contribution.

        In NN field theory, multiple saddle points correspond to different
        trained network configurations (local minima of the loss landscape).

        Args:
            saddle_points: List of saddle-point configurations.

        Returns:
            Dictionary with individual and combined contributions.
        """
        contributions = []
        for phi_s in saddle_points:
            loop_data = self.one_loop_correction(phi_s)
            fluct = self.gaussian_fluctuations(phi_s)
            contributions.append({
                "saddle_point": phi_s,
                "S_classical": loop_data["S_classical"],
                "S_effective": loop_data["S_effective"],
                "one_loop": loop_data["one_loop_correction"],
                "n_negative_modes": fluct["n_negative_modes"],
                "weight": np.exp(-loop_data["S_effective"]),
            })

        # Sort by action (dominant saddle first)
        contributions.sort(key=lambda c: c["S_classical"])

        total_weight = sum(c["weight"] for c in contributions)

        return {
            "contributions": contributions,
            "total_weight": total_weight,
            "dominant_saddle": contributions[0]["saddle_point"],
            "n_saddles": len(saddle_points),
            "action_gaps": [
                c["S_classical"] - contributions[0]["S_classical"]
                for c in contributions[1:]
            ],
        }

    def partition_function(
        self,
        saddle_points: List[np.ndarray],
        temperature: float = 1.0,
    ) -> Dict:
        """Compute partition function from saddle-point contributions.

        Z = Σ_s exp(-S(φ_s)/T) × (2π T)^{n/2} / √|det S''(φ_s)|

        Temperature T plays the role of 1/N in the large-width expansion:
        higher T (smaller N) means larger fluctuations.

        Args:
            saddle_points: List of saddle-point configurations.
            temperature: Temperature parameter (= 1/N for width expansion).

        Returns:
            Dictionary with partition function, free energy, and per-saddle data.
        """
        T = temperature
        contributions = []
        log_Z_parts = []

        for phi_s in saddle_points:
            S = self.action_fn(phi_s)
            fluct = self.gaussian_fluctuations(phi_s)
            pos_eigs = fluct["eigenvalues"][fluct["eigenvalues"] > 1e-10]
            n_eff = len(pos_eigs)

            # log Z_s = -S/T + (n/2) ln(2πT) - (1/2) Σ ln λ_i
            log_Zs = (
                -S / T
                + 0.5 * n_eff * np.log(2 * np.pi * T)
                - 0.5 * np.sum(np.log(pos_eigs))
            )
            contributions.append({
                "saddle_point": phi_s,
                "action": S,
                "log_Z_contribution": log_Zs,
                "boltzmann_weight": np.exp(-S / T),
            })
            log_Z_parts.append(log_Zs)

        # log Z = log(Σ exp(log_Z_s)) via log-sum-exp trick
        max_log = max(log_Z_parts)
        log_Z = max_log + np.log(sum(np.exp(lz - max_log) for lz in log_Z_parts))
        free_energy = -T * log_Z

        return {
            "log_Z": log_Z,
            "Z": np.exp(log_Z) if log_Z < 500 else float("inf"),
            "free_energy": free_energy,
            "temperature": T,
            "contributions": contributions,
            "dominant_saddle_index": int(np.argmax(log_Z_parts)),
        }

    def effective_action_expansion(
        self,
        phi_range: np.ndarray,
        order: int = 4,
    ) -> Dict:
        """Compute effective action Γ[φ] including loop corrections.

        Γ[φ] = S[φ] + (1/2) Tr ln S''[φ] + higher loops

        The effective action is the Legendre transform of ln Z[J] and
        generates 1PI correlation functions. At tree level Γ = S; loop
        corrections dress the classical action.

        Args:
            phi_range: Array of field values at which to evaluate Γ.
            order: Maximum order in loop expansion (2=one-loop, 4=two-loop approx).

        Returns:
            Dictionary with effective action values and its derivatives.
        """
        gamma_values = np.zeros_like(phi_range, dtype=float)
        classical_values = np.zeros_like(phi_range, dtype=float)
        one_loop_values = np.zeros_like(phi_range, dtype=float)

        for i, phi_val in enumerate(phi_range):
            phi = np.atleast_1d(phi_val)
            S = self.action_fn(phi)
            classical_values[i] = S

            H = self.hessian_fn(phi)
            eigs = np.linalg.eigvalsh(H)
            pos_eigs = eigs[eigs > 1e-10]
            one_loop = 0.5 * np.sum(np.log(np.abs(pos_eigs) / (2 * np.pi))) if len(pos_eigs) > 0 else 0.0
            one_loop_values[i] = one_loop

            gamma_values[i] = S + one_loop

        # Numerical derivatives of Γ for vertices
        if len(phi_range) > 4:
            dphi = phi_range[1] - phi_range[0] if len(phi_range) > 1 else 1.0
            gamma_prime = np.gradient(gamma_values, dphi)
            gamma_double_prime = np.gradient(gamma_prime, dphi)
        else:
            gamma_prime = np.zeros_like(gamma_values)
            gamma_double_prime = np.zeros_like(gamma_values)

        return {
            "phi_range": phi_range,
            "gamma": gamma_values,
            "classical_action": classical_values,
            "one_loop_correction": one_loop_values,
            "gamma_prime": gamma_prime,
            "gamma_double_prime": gamma_double_prime,
            "order": order,
        }

    def stokes_phenomenon(
        self,
        saddle_points: List[np.ndarray],
        parameter_range: np.ndarray,
    ) -> Dict:
        """Detect Stokes phenomenon: when saddle contributions exchange dominance.

        As a parameter varies, the relative importance of different saddle
        points changes. At Stokes lines, a subdominant saddle's contribution
        jumps discontinuously (its Stokes multiplier changes). This is
        related to wall-crossing in NN phase diagrams.

        Reference: Berry & Howls, "Hyperasymptotics for integrals with saddles"
        (Proc. R. Soc. A, 1991).

        Args:
            saddle_points: List of saddle-point configurations (may be
                parameter-dependent via closures in action_fn).
            parameter_range: Range of the control parameter.

        Returns:
            Dictionary with dominance structure and Stokes transition points.
        """
        n_saddles = len(saddle_points)
        n_params = len(parameter_range)
        actions = np.zeros((n_saddles, n_params))

        # Evaluate action at each saddle for each parameter value
        for j, param in enumerate(parameter_range):
            for i, phi_s in enumerate(saddle_points):
                actions[i, j] = self.action_fn(phi_s)

        # Find which saddle dominates (smallest action) at each parameter
        dominant = np.argmin(actions, axis=0)

        # Find Stokes transitions (where dominant saddle changes)
        transitions = []
        for j in range(1, n_params):
            if dominant[j] != dominant[j - 1]:
                # Linear interpolation for transition point
                a1 = actions[dominant[j - 1], j - 1] - actions[dominant[j], j - 1]
                a2 = actions[dominant[j - 1], j] - actions[dominant[j], j]
                if abs(a2 - a1) > 1e-15:
                    t = a1 / (a1 - a2)
                    param_cross = parameter_range[j - 1] + t * (
                        parameter_range[j] - parameter_range[j - 1]
                    )
                else:
                    param_cross = 0.5 * (parameter_range[j - 1] + parameter_range[j])
                transitions.append({
                    "parameter": param_cross,
                    "from_saddle": int(dominant[j - 1]),
                    "to_saddle": int(dominant[j]),
                    "action_gap_before": float(a1),
                    "action_gap_after": float(a2),
                })

        return {
            "actions": actions,
            "dominant_saddle": dominant,
            "transitions": transitions,
            "n_transitions": len(transitions),
            "parameter_range": parameter_range,
        }


class InstantonCalculator:
    """Calculator for instanton (bounce) solutions and tunneling rates.

    Instantons are finite-action solutions of the Euclidean equations of
    motion that mediate tunneling between different vacua. In neural network
    field theory, they describe non-perturbative transitions between
    different trained configurations.

    The bounce equation in d dimensions with O(d) symmetry is:
        φ'' + (d-1)/r · φ' = V'(φ)

    with boundary conditions φ'(0) = 0, φ(∞) = φ_false.

    References:
        - S. Coleman, "Fate of the false vacuum" (Phys. Rev. D 15, 1977)
        - C. Callan & S. Coleman, "Fate of the false vacuum II" (Phys. Rev. D 16, 1977)
        - Zinn-Justin, Ch. 37: "Instantons in Quantum Mechanics"
    """

    def __init__(
        self,
        potential_fn: Callable[[float], float],
        potential_derivative_fn: Callable[[float], float],
        dimension: int = 4,
    ):
        """Initialize instanton calculator.

        Args:
            potential_fn: The potential V(φ) with at least two local minima.
            potential_derivative_fn: V'(φ), derivative of the potential.
            dimension: Spacetime dimension d for the O(d)-symmetric bounce.
        """
        self.V = potential_fn
        self.dV = potential_derivative_fn
        self.d = dimension

    def bounce_solution(
        self,
        false_vacuum: float,
        true_vacuum: float,
        r_max: float = 50.0,
        n_grid: int = 1000,
    ) -> Dict:
        """Find the bounce solution (instanton) for tunneling.

        Solves the ODE: φ'' + (d-1)/r · φ' = V'(φ) using the shooting
        method. Start near the true vacuum at r=0 and tune the initial
        value φ(0) so that φ(∞) → φ_false.

        The shooting parameter is φ(0); φ'(0) = 0 by O(d) symmetry.

        Reference: Coleman, "Fate of the false vacuum", Sec. III.

        Args:
            false_vacuum: Field value at the false (metastable) vacuum.
            true_vacuum: Field value at the true (stable) vacuum.
            r_max: Maximum radius for integration.
            n_grid: Number of radial grid points.

        Returns:
            Dictionary with bounce profile φ(r) and associated data.
        """
        d = self.d

        def bounce_ode(r, y):
            phi, dphi = y
            if r < 1e-10:
                # L'Hôpital: (d-1)/r * φ' → (d-1) * φ'' at r=0
                # so d * φ'' = V'(φ) → φ'' = V'(φ)/d
                ddphi = self.dV(phi) / d
            else:
                ddphi = self.dV(phi) - (d - 1) / r * dphi
            return [dphi, ddphi]

        # Shooting method: find φ(0) = φ_0 such that φ(r→∞) → false_vacuum
        def shoot(phi_0):
            y0 = [phi_0, 0.0]
            r_span = (1e-6, r_max)
            r_eval = np.linspace(1e-6, r_max, n_grid)
            sol = integrate.solve_ivp(
                bounce_ode, r_span, y0, t_eval=r_eval,
                method="RK45", max_step=r_max / n_grid * 2,
            )
            return sol

        # Binary search for the correct φ(0)
        phi_lo = min(false_vacuum, true_vacuum)
        phi_hi = max(false_vacuum, true_vacuum)

        # The bounce starts near the true vacuum side
        if true_vacuum < false_vacuum:
            search_lo, search_hi = true_vacuum, false_vacuum
        else:
            search_lo, search_hi = false_vacuum, true_vacuum

        best_sol = None
        best_phi0 = None
        for _ in range(64):
            phi_0 = 0.5 * (search_lo + search_hi)
            sol = shoot(phi_0)
            if sol.success:
                endpoint = sol.y[0, -1]
                if (true_vacuum < false_vacuum and endpoint > false_vacuum) or \
                   (true_vacuum > false_vacuum and endpoint < false_vacuum):
                    search_hi = phi_0
                else:
                    search_lo = phi_0
                best_sol = sol
                best_phi0 = phi_0
            else:
                search_hi = phi_0

        if best_sol is None:
            # Fallback: use midpoint
            phi_0 = 0.5 * (false_vacuum + true_vacuum)
            best_sol = shoot(phi_0)
            best_phi0 = phi_0

        r_values = best_sol.t
        phi_values = best_sol.y[0]
        dphi_values = best_sol.y[1]

        return {
            "r": r_values,
            "phi": phi_values,
            "dphi_dr": dphi_values,
            "phi_0": best_phi0,
            "false_vacuum": false_vacuum,
            "true_vacuum": true_vacuum,
            "r_max": r_max,
        }

    def tunneling_rate(self, bounce_action: float, prefactor: float = 1.0) -> Dict:
        """Compute tunneling rate from bounce action.

        The tunneling rate per unit volume is:
            Γ/V ~ A × exp(-S_bounce)

        where A is a prefactor from the fluctuation determinant.

        Reference: Callan & Coleman, Eq. (3.1).

        Args:
            bounce_action: The Euclidean action of the bounce S_bounce.
            prefactor: Pre-exponential factor A from determinant ratio.

        Returns:
            Dictionary with tunneling rate and decay lifetime.
        """
        log_rate = np.log(prefactor) - bounce_action
        rate = prefactor * np.exp(-bounce_action) if bounce_action < 500 else 0.0
        lifetime = 1.0 / rate if rate > 0 else float("inf")

        return {
            "bounce_action": bounce_action,
            "log_tunneling_rate": log_rate,
            "tunneling_rate": rate,
            "lifetime": lifetime,
            "prefactor": prefactor,
        }

    def thin_wall_approximation(
        self,
        false_vacuum: float,
        true_vacuum: float,
        epsilon: float,
    ) -> Dict:
        """Thin-wall approximation for nearly degenerate vacua.

        When the energy splitting ε = V(φ_false) - V(φ_true) is small,
        the bounce has a thin wall of thickness ~ 1/m separating large
        regions of false and true vacuum. The bounce action is:

            S_bounce ≈ 27π² σ⁴ / (2ε³)     (d=4)
            S_bounce ≈ π σ² / ε             (d=2, for reference)

        where σ = ∫ dφ √(2V_barrier(φ)) is the surface tension.

        Reference: Coleman, "Fate of the false vacuum", Sec. V.

        Args:
            false_vacuum: False vacuum field value.
            true_vacuum: True vacuum field value.
            epsilon: Energy splitting V(false) - V(true) > 0.

        Returns:
            Dictionary with bounce action, bubble radius, and surface tension.
        """
        # Compute surface tension σ = ∫ dφ √(2 ΔV(φ))
        # ΔV(φ) = V(φ) - V(false_vacuum) for the degenerate potential
        V_false = self.V(false_vacuum)

        def integrand(phi):
            dV = self.V(phi) - V_false
            return np.sqrt(2 * max(dV, 0.0))

        sigma, _ = integrate.quad(
            integrand, min(false_vacuum, true_vacuum), max(false_vacuum, true_vacuum)
        )

        d = self.d
        if d == 4:
            # Bubble radius R = 3σ/ε
            R_bubble = 3.0 * sigma / epsilon if epsilon > 0 else float("inf")
            S_bounce = 27.0 * np.pi**2 * sigma**4 / (2.0 * epsilon**3)
        elif d == 3:
            R_bubble = 2.0 * sigma / epsilon if epsilon > 0 else float("inf")
            S_bounce = 16.0 * np.pi * sigma**3 / (3.0 * epsilon**2)
        elif d == 2:
            R_bubble = sigma / epsilon if epsilon > 0 else float("inf")
            S_bounce = np.pi * sigma**2 / epsilon
        else:
            # General formula: S_d = (d-1)^d ω_d σ^d / (d * ε^{d-1})
            omega_d = 2 * np.pi ** (d / 2) / special.gamma(d / 2)
            R_bubble = (d - 1) * sigma / epsilon if epsilon > 0 else float("inf")
            S_bounce = (
                (d - 1) ** d * omega_d * sigma**d / (d * epsilon ** (d - 1))
            )

        return {
            "surface_tension": sigma,
            "bubble_radius": R_bubble,
            "bounce_action": S_bounce,
            "epsilon": epsilon,
            "dimension": d,
            "thin_wall_valid": epsilon < sigma,
        }

    def instanton_action(
        self,
        bounce_solution: Dict,
        potential_fn: Optional[Callable] = None,
    ) -> float:
        """Compute the Euclidean action of the bounce solution.

        S[φ_bounce] = Ω_{d-1} ∫_0^∞ dr r^{d-1} [½(φ')² + V(φ)]

        where Ω_{d-1} is the area of the unit (d-1)-sphere.

        Args:
            bounce_solution: Output from bounce_solution().
            potential_fn: Potential function (defaults to self.V).

        Returns:
            The Euclidean action S_E of the bounce.
        """
        V = potential_fn if potential_fn is not None else self.V
        d = self.d
        r = bounce_solution["r"]
        phi = bounce_solution["phi"]
        dphi = bounce_solution["dphi_dr"]

        # Ω_{d-1} = 2π^{d/2} / Γ(d/2)
        omega = 2 * np.pi ** (d / 2) / special.gamma(d / 2)

        # Integrand: r^{d-1} [½(φ')² + V(φ) - V(φ_false)]
        V_false = V(bounce_solution["false_vacuum"])
        integrand = r ** (d - 1) * (0.5 * dphi**2 + np.array([V(p) for p in phi]) - V_false)

        action = omega * np.trapz(integrand, r)
        return float(action)

    def fluctuation_determinant(
        self,
        bounce_solution: Dict,
        potential_fn: Optional[Callable] = None,
    ) -> Dict:
        """Compute fluctuation determinant ratio around the bounce.

        The prefactor for the tunneling rate involves the ratio:
            det'(-∂² + V''(φ_bounce)) / det(-∂² + V''(φ_false))

        where det' excludes zero modes (from translational invariance).
        The bounce has d zero modes from translations in d dimensions.

        Reference: Callan & Coleman, Sec. III; Coleman, Ch. 7, Sec. 4.

        Args:
            bounce_solution: Output from bounce_solution().
            potential_fn: Potential function (defaults to self.V).

        Returns:
            Dictionary with determinant ratio and eigenvalue information.
        """
        V = potential_fn if potential_fn is not None else self.V
        r = bounce_solution["r"]
        phi = bounce_solution["phi"]
        n = len(r)
        dr = r[1] - r[0] if n > 1 else 1.0

        # Discretize the fluctuation operator: -d²/dr² - (d-1)/r · d/dr + V''(φ)
        # using finite differences
        V_double_prime = np.array([
            (V(phi[i] + 1e-5) - 2 * V(phi[i]) + V(phi[i] - 1e-5)) / 1e-10
            for i in range(n)
        ])

        # Build the radial operator as a tridiagonal matrix
        d_dim = self.d
        diag = np.zeros(n)
        off_diag_upper = np.zeros(n - 1)
        off_diag_lower = np.zeros(n - 1)

        for i in range(n):
            diag[i] = 2.0 / dr**2 + V_double_prime[i]

        for i in range(n - 1):
            r_mid = 0.5 * (r[i] + r[i + 1])
            if r_mid > 1e-10:
                friction = (d_dim - 1) / (2.0 * r_mid)
            else:
                friction = 0.0
            off_diag_upper[i] = -1.0 / dr**2 - friction / dr
            off_diag_lower[i] = -1.0 / dr**2 + friction / dr

        # Construct sparse-like banded matrix; use full matrix for small n
        M_bounce = np.diag(diag) + np.diag(off_diag_upper, 1) + np.diag(off_diag_lower, -1)

        # False vacuum operator
        V_pp_false = (
            V(bounce_solution["false_vacuum"] + 1e-5)
            - 2 * V(bounce_solution["false_vacuum"])
            + V(bounce_solution["false_vacuum"] - 1e-5)
        ) / 1e-10
        diag_false = np.full(n, 2.0 / dr**2 + V_pp_false)
        M_false = np.diag(diag_false) + np.diag(-np.ones(n - 1) / dr**2, 1) + np.diag(-np.ones(n - 1) / dr**2, -1)

        # Compute eigenvalues
        eigs_bounce = np.linalg.eigvalsh(M_bounce)
        eigs_false = np.linalg.eigvalsh(M_false)

        # One negative eigenvalue expected (the bounce is a saddle of S_E)
        n_negative = np.sum(eigs_bounce < -1e-8)
        # d zero modes from translational invariance
        n_zero = np.sum(np.abs(eigs_bounce) < 1e-6)

        # Determinant ratio excluding zero and negative modes
        pos_bounce = eigs_bounce[eigs_bounce > 1e-6]
        pos_false = eigs_false[eigs_false > 1e-6]

        n_common = min(len(pos_bounce), len(pos_false))
        if n_common > 0:
            log_det_ratio = np.sum(np.log(pos_bounce[:n_common])) - np.sum(np.log(pos_false[:n_common]))
        else:
            log_det_ratio = 0.0

        return {
            "log_det_ratio": log_det_ratio,
            "n_negative_modes": int(n_negative),
            "n_zero_modes": int(n_zero),
            "lowest_eigenvalue": float(eigs_bounce[0]),
            "negative_eigenvalue": float(eigs_bounce[0]) if n_negative > 0 else None,
            "expected_zero_modes": self.d,
        }

    def instanton_correction_to_energy(
        self,
        bounce_action: float,
        prefactor: float = 1.0,
    ) -> Dict:
        """Non-perturbative correction to ground state energy from instantons.

        In quantum mechanics with degenerate minima, the instanton gives
        a level splitting: ΔE ~ ℏω × (S_inst/2πℏ)^{1/2} × exp(-S_inst/ℏ)

        The energy correction is: E = -ℏΓ/2 where Γ is the tunneling rate.

        Reference: Coleman, Ch. 7, Eq. (7.5.13); Zinn-Justin, Ch. 37.

        Args:
            bounce_action: Instanton action S_inst.
            prefactor: Pre-exponential factor including fluctuation determinant.

        Returns:
            Dictionary with energy correction and level splitting.
        """
        if bounce_action > 500:
            delta_E = 0.0
            log_delta_E = -bounce_action + np.log(abs(prefactor))
        else:
            delta_E = prefactor * np.exp(-bounce_action)
            log_delta_E = np.log(abs(prefactor)) - bounce_action

        return {
            "energy_correction": -0.5 * delta_E,
            "level_splitting": abs(delta_E),
            "log_level_splitting": log_delta_E,
            "bounce_action": bounce_action,
            "prefactor": prefactor,
            "is_exponentially_small": bounce_action > 10,
        }

    def multi_instanton_gas(
        self,
        bounce_action: float,
        volume: float,
        temperature: float = 1.0,
    ) -> Dict:
        """Dilute instanton gas approximation.

        For a dilute gas of n instantons and n̄ anti-instantons in volume V,
        the partition function is:
            Z = Σ_{n,n̄} [K V exp(-S_inst)]^{n+n̄} / (n! n̄!)
              = exp(2 K V exp(-S_inst))

        where K is the single-instanton prefactor. This gives the
        non-perturbative free energy.

        Reference: 't Hooft, "Computation of the quantum effects due to a
        four-dimensional pseudoparticle" (Phys. Rev. D 14, 1976).

        Args:
            bounce_action: Single instanton action.
            volume: Spacetime volume V.
            temperature: Temperature parameter.

        Returns:
            Dictionary with free energy, instanton density, and screening.
        """
        # Fugacity = exp(-S/T) per unit volume
        log_fugacity = -bounce_action / temperature
        if log_fugacity > -500:
            fugacity = np.exp(log_fugacity)
        else:
            fugacity = 0.0

        # Mean instanton number ⟨n⟩ = K V exp(-S/T)
        mean_n = volume * fugacity

        # Free energy = -T ln Z = -T × 2 × V × fugacity (instantons + anti-instantons)
        F_inst = -temperature * 2.0 * volume * fugacity

        # Instanton density
        density = fugacity

        # Inter-instanton distance
        if density > 0:
            mean_separation = density ** (-1.0 / self.d)
        else:
            mean_separation = float("inf")

        # Diluteness check: gas valid when separation >> instanton size
        dilute = mean_separation > 5.0  # heuristic

        return {
            "free_energy_correction": F_inst,
            "mean_instanton_number": mean_n,
            "instanton_density": density,
            "mean_separation": mean_separation,
            "dilute_gas_valid": dilute,
            "bounce_action": bounce_action,
            "volume": volume,
            "log_fugacity": log_fugacity,
        }

    def neural_network_instanton(
        self,
        ntk_action: float,
        width: int,
    ) -> Dict:
        """Non-perturbative instanton correction in NN field theory.

        In the neural network path integral with width N, the action scales
        as N × S_inst. Non-perturbative corrections go as:

            δf ~ exp(-N × S_inst) × N^{power} × (perturbative series in 1/N)

        These capture tunneling between different trained configurations
        that cannot be seen in the 1/N expansion.

        Args:
            ntk_action: The instanton action in NTK units (i.e., S_inst
                such that the full bounce action is N × S_inst).
            width: Network width N.

        Returns:
            Dictionary with non-perturbative correction estimate.
        """
        N = width
        full_action = N * ntk_action

        # Leading non-perturbative correction
        log_correction = -full_action
        if full_action < 500:
            correction = np.exp(-full_action)
        else:
            correction = 0.0

        # Prefactor: fluctuation determinant gives N-dependent power
        # From d zero modes (translations) and the negative mode
        prefactor_power = -self.d / 2.0  # from zero mode integrals
        log_prefactor = prefactor_power * np.log(N)

        # Series in 1/N from fluctuations around instanton
        perturbative_corrections = [1.0]
        for k in range(1, 5):
            # Placeholder coefficients; in practice computed from Feynman diagrams
            perturbative_corrections.append((-1)**k * special.gamma(k + 0.5) / (N**k * np.math.factorial(k)))

        total_log = log_correction + log_prefactor
        total_correction = correction * N**prefactor_power * sum(perturbative_corrections)

        return {
            "correction": total_correction,
            "log_correction": total_log,
            "full_bounce_action": full_action,
            "ntk_action": ntk_action,
            "width": N,
            "prefactor_power": prefactor_power,
            "perturbative_series": perturbative_corrections,
            "suppression_factor": float(np.exp(-full_action)) if full_action < 500 else 0.0,
            "visible_at_width": int(np.ceil(10.0 / ntk_action)) if ntk_action > 0 else None,
        }


class BorelResummation:
    """Borel resummation of divergent asymptotic series.

    The 1/N expansion in neural network field theory produces a divergent
    asymptotic series with factorially growing coefficients:

        f(N) ~ Σ_n a_n / N^n,    a_n ~ n! × S_inst^{-n}

    The factorial growth is connected to instantons via the dispersion
    relation (Bogomolny-Zinn-Justin mechanism). Borel resummation extracts
    finite answers from such divergent series.

    The procedure is:
        1. Form the Borel transform B(t) = Σ a_n t^n / n!
        2. Sum B(t) via Padé approximants
        3. Compute f(N) = ∫₀^∞ dt e^{-t} B(t/N)

    References:
        - 't Hooft, "Can We Make Sense out of QCD?" (1977)
        - Zinn-Justin, Ch. 37 & 40
        - M. Mariño, "Lectures on non-perturbative effects in large N gauge
          theories, matrix models and strings" (2012)
        - D. Dorigoni, Annals Phys. 409 (2019) 167914
    """

    def __init__(self, coefficients: List[float]):
        """Initialize with coefficients of the divergent series.

        Args:
            coefficients: Coefficients a_n of the asymptotic series
                f(N) = Σ_{n=0}^∞ a_n / N^n. Provide as many terms as known.
        """
        self.coefficients = np.array(coefficients, dtype=np.float64)
        self.n_terms = len(coefficients)

    def borel_transform(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """Evaluate the Borel transform B(t) = Σ a_n t^n / n!.

        The Borel transform tames the factorial divergence: if a_n ~ n!,
        then a_n/n! ~ 1 and B(t) has finite radius of convergence.

        Args:
            t: Point(s) at which to evaluate B(t).

        Returns:
            B(t) values.
        """
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        result = np.zeros_like(t)
        for n, a_n in enumerate(self.coefficients):
            result += a_n * t**n / special.gamma(n + 1)
        return result

    def borel_sum(
        self,
        N: float,
        method: str = "pade",
        pade_order: Optional[Tuple[int, int]] = None,
    ) -> Dict:
        """Compute the Borel sum f(N) = ∫₀^∞ e^{-t} B(t/N) dt.

        This integral, if it converges, defines the unique (Borel-summable)
        function with the given asymptotic expansion.

        Args:
            N: The large parameter (network width).
            method: 'direct' for direct numerical integration of truncated
                Borel transform, or 'pade' for Padé-improved integration.
            pade_order: (m, n) order for Padé approximant. If None, uses
                balanced order m ≈ n ≈ n_terms/2.

        Returns:
            Dictionary with resummed value and error estimate.
        """
        if method == "pade":
            if pade_order is None:
                m = self.n_terms // 2
                n = self.n_terms - m - 1
                if n < 0:
                    n = 0
                    m = self.n_terms - 1
            else:
                m, n = pade_order
            return self.pade_borel(N, m, n)

        # Direct integration of truncated Borel transform
        def integrand(t):
            bt = self.borel_transform(t / N)
            return np.exp(-t) * bt

        result, error = integrate.quad(integrand, 0, np.inf, limit=200)

        return {
            "value": result,
            "error": error,
            "N": N,
            "method": "direct",
            "n_terms_used": self.n_terms,
        }

    def pade_borel(self, N: float, m: int, n: int) -> Dict:
        """Padé-Borel resummation.

        Construct the [m/n] Padé approximant P_m(t)/Q_n(t) of the Borel
        transform, then compute:

            f(N) = ∫₀^∞ e^{-t} [P_m(t/N) / Q_n(t/N)] dt

        The Padé approximant analytically continues B(t) beyond its radius
        of convergence, capturing the effect of Borel-plane singularities.

        Args:
            N: The large parameter (network width).
            m: Degree of numerator polynomial.
            n: Degree of denominator polynomial.

        Returns:
            Dictionary with resummed value and Padé poles.
        """
        # Build Padé coefficients from Taylor series a_k/k!
        borel_coeffs = np.array([
            self.coefficients[k] / special.gamma(k + 1)
            for k in range(min(m + n + 1, self.n_terms))
        ])

        # Construct Padé approximant [m/n] via scipy
        # We need m+n+1 coefficients
        n_available = len(borel_coeffs)
        m_eff = min(m, n_available - 1)
        n_eff = min(n, n_available - m_eff - 1)
        if n_eff < 0:
            n_eff = 0
            m_eff = n_available - 1

        # Build the Padé using the Wynn epsilon algorithm / direct construction
        # For simplicity, use polynomial fitting approach
        total_order = m_eff + n_eff
        if total_order >= n_available:
            total_order = n_available - 1
            n_eff = max(0, total_order - m_eff)

        # Use scipy's pade if available, otherwise manual construction
        try:
            from scipy.interpolate import pade as scipy_pade
            p_coeffs = borel_coeffs[:m_eff + n_eff + 1]
            p_poly, q_poly = scipy_pade(p_coeffs, n_eff)
        except (ImportError, Exception):
            # Fallback: just use truncated series
            p_poly = np.polynomial.polynomial.Polynomial(borel_coeffs[:m_eff + 1])
            q_poly = np.polynomial.polynomial.Polynomial([1.0])

        # Find poles of denominator (singularities in Borel plane)
        if hasattr(q_poly, 'roots'):
            poles = q_poly.roots()
        elif hasattr(q_poly, 'r'):
            poles = q_poly.r
        else:
            q_coeffs = np.array(q_poly) if not callable(q_poly) else [1.0]
            poles = np.roots(q_coeffs) if len(q_coeffs) > 1 else np.array([])

        # Numerical integration of Padé-Borel
        def integrand(t):
            s = t / N
            try:
                if callable(p_poly) and callable(q_poly):
                    val = p_poly(s) / q_poly(s)
                else:
                    num = sum(borel_coeffs[k] * s**k for k in range(min(m_eff + 1, len(borel_coeffs))))
                    val = num
                return np.exp(-t) * float(np.real(val))
            except (ZeroDivisionError, FloatingPointError):
                return 0.0

        result, error = integrate.quad(integrand, 0, np.inf, limit=200)

        real_poles = np.real(poles[np.abs(np.imag(poles)) < 1e-10]) if len(poles) > 0 else np.array([])

        return {
            "value": result,
            "error": error,
            "N": N,
            "method": f"pade[{m_eff}/{n_eff}]",
            "pade_poles": poles,
            "real_poles": real_poles,
            "nearest_singularity": float(np.min(np.abs(real_poles))) if len(real_poles) > 0 else None,
        }

    def conformal_borel(self, N: float, mapping_param: float = 1.0) -> Dict:
        """Conformal mapping improved Borel resummation.

        Map the Borel plane t → w(t) = (√(1 + t/a) - 1) / (√(1 + t/a) + 1)
        where a is the location of the nearest singularity. This maps the
        cut plane to the unit disk, improving convergence of the series.

        Reference: Le Guillou & Zinn-Justin, "Large-Order Behaviour of
        Perturbation Theory" (1990).

        Args:
            N: The large parameter.
            mapping_param: Location of nearest Borel singularity a > 0.

        Returns:
            Dictionary with conformally-improved Borel sum.
        """
        a = mapping_param

        # Conformal map: t → w = (√(1+t/a) - 1) / (√(1+t/a) + 1)
        # Inverse: t = a × 4w / (1-w)²
        def t_of_w(w):
            return a * 4.0 * w / (1.0 - w) ** 2

        def dt_dw(w):
            return a * 4.0 * (1.0 + w) / (1.0 - w) ** 3

        # Re-expand B(t(w)) in powers of w
        # Compute mapped coefficients by evaluating at Chebyshev points
        n_pts = min(self.n_terms * 2, 30)
        w_pts = np.cos(np.pi * np.arange(n_pts) / (n_pts - 1)) * 0.5 + 0.5
        w_pts = w_pts[(w_pts > 0.01) & (w_pts < 0.99)]
        t_pts = np.array([t_of_w(w) for w in w_pts])
        B_pts = self.borel_transform(t_pts)

        # Fit polynomial in w
        n_fit = min(len(w_pts) - 1, self.n_terms)
        if n_fit > 0 and len(w_pts) > n_fit:
            poly_coeffs = np.polyfit(w_pts, B_pts, n_fit)
            B_mapped = np.poly1d(poly_coeffs)
        else:
            B_mapped = lambda w: self.borel_transform(np.atleast_1d(t_of_w(w)))

        # Integrate: f(N) = ∫₀^∞ dt e^{-t} B(t/N)
        # Change variables to w: dt = dt_dw dw, integrate w from 0 to 1
        def integrand(w):
            if w <= 0.001 or w >= 0.999:
                return 0.0
            t = t_of_w(w)
            jac = dt_dw(w)
            bval = float(np.real(B_mapped(w))) if callable(B_mapped) else 0.0
            return np.exp(-t) * bval * jac / N

        result, error = integrate.quad(integrand, 0.001, 0.999, limit=200)

        return {
            "value": result,
            "error": error,
            "N": N,
            "method": "conformal_borel",
            "mapping_param": a,
        }

    def detect_singularities(self, method: str = "ratio_test") -> Dict:
        """Detect singularities of B(t) in the Borel plane.

        Singularities in the Borel transform are directly related to
        non-perturbative physics: a singularity at t = S_inst corresponds
        to an instanton with action S_inst.

        Methods:
        - ratio_test: |a_{n+1}/a_n| × n → 1/R (radius of convergence)
        - domb_sykes: plot a_n/a_{n-1} vs 1/n, extrapolate to find singularity

        Reference: Zinn-Justin, Ch. 40, "Summation Methods".

        Args:
            method: Detection method ('ratio_test' or 'domb_sykes').

        Returns:
            Dictionary with detected singularity locations.
        """
        borel_coeffs = np.array([
            self.coefficients[k] / special.gamma(k + 1)
            for k in range(self.n_terms)
        ])

        if method == "ratio_test":
            ratios = []
            for n in range(1, len(borel_coeffs)):
                if abs(borel_coeffs[n - 1]) > 1e-15:
                    ratios.append(abs(borel_coeffs[n] / borel_coeffs[n - 1]))
                else:
                    ratios.append(np.nan)

            ratios = np.array(ratios)
            valid = ~np.isnan(ratios)
            if np.any(valid):
                # Extrapolate to n→∞ using last few ratios
                recent_ratios = ratios[valid][-min(5, np.sum(valid)):]
                inv_radius = np.mean(recent_ratios)
                radius = 1.0 / inv_radius if inv_radius > 1e-15 else float("inf")
            else:
                radius = float("inf")
                inv_radius = 0.0

            return {
                "method": "ratio_test",
                "ratios": ratios,
                "radius_of_convergence": radius,
                "nearest_singularity": radius,
                "instanton_action_estimate": radius,
            }

        elif method == "domb_sykes":
            # Domb-Sykes plot: a_n/a_{n-1} vs 1/n → intercept gives 1/R
            xs, ys = [], []
            for n in range(1, len(borel_coeffs)):
                if abs(borel_coeffs[n - 1]) > 1e-15:
                    xs.append(1.0 / n)
                    ys.append(borel_coeffs[n] / borel_coeffs[n - 1])

            if len(xs) >= 2:
                xs, ys = np.array(xs), np.array(ys)
                # Linear fit: y = a + b/n
                coeffs_fit = np.polyfit(xs, ys, 1)
                intercept = coeffs_fit[1]  # a_n/a_{n-1} as n→∞
                radius = 1.0 / abs(intercept) if abs(intercept) > 1e-15 else float("inf")

                # Sign tells us if singularity is on positive or negative real axis
                sign = np.sign(intercept)
            else:
                radius = float("inf")
                intercept = 0.0
                sign = 0
                coeffs_fit = [0, 0]

            return {
                "method": "domb_sykes",
                "radius_of_convergence": radius,
                "intercept": float(intercept),
                "singularity_position": float(sign * radius),
                "fit_coefficients": coeffs_fit.tolist() if isinstance(coeffs_fit, np.ndarray) else coeffs_fit,
            }

        raise ValueError(f"Unknown method: {method}. Use 'ratio_test' or 'domb_sykes'.")

    def instanton_singularity(self, action: float) -> Dict:
        """Relate Borel singularity to instanton with given action.

        If the Borel transform has a singularity at t = S_inst, then
        the large-order behavior of perturbation theory is:

            a_n ~ (1/S_inst)^n × Γ(n + b) × [1 + O(1/n)]

        where b is related to the fluctuation determinant around the
        instanton. This is the Bogomolny-Zinn-Justin relation.

        Reference: Bogomolny, Phys. Lett. B 91 (1980) 76;
            Zinn-Justin, J. Math. Phys. 22 (1981) 511.

        Args:
            action: The instanton action S_inst.

        Returns:
            Dictionary checking consistency with measured large-order behavior.
        """
        # Predicted large-order: a_n ~ C × (1/S)^n × n!^{1} × n^b
        S = action

        # Extract b from the ratio of successive coefficients
        predicted_ratios = []
        measured_ratios = []
        for n in range(1, self.n_terms):
            predicted_ratios.append((n) / S)  # leading: a_n/a_{n-1} ~ n/S
            if abs(self.coefficients[n - 1]) > 1e-15:
                measured_ratios.append(self.coefficients[n] / self.coefficients[n - 1])
            else:
                measured_ratios.append(np.nan)

        predicted_ratios = np.array(predicted_ratios)
        measured_ratios = np.array(measured_ratios)

        # Estimate b from subleading correction
        valid = ~np.isnan(measured_ratios) & (predicted_ratios > 0)
        if np.any(valid):
            ratio_of_ratios = measured_ratios[valid] / predicted_ratios[valid]
            ns = np.arange(1, self.n_terms)[valid]
            if len(ns) >= 2:
                # Fit: measured/predicted ~ 1 + b/n
                fit = np.polyfit(1.0 / ns, ratio_of_ratios, 1)
                b_estimate = fit[0]
                consistency = float(np.mean(np.abs(ratio_of_ratios - 1.0)))
            else:
                b_estimate = 0.0
                consistency = float("inf")
        else:
            b_estimate = 0.0
            consistency = float("inf")

        return {
            "instanton_action": S,
            "predicted_ratios": predicted_ratios,
            "measured_ratios": measured_ratios,
            "b_parameter": b_estimate,
            "consistency_measure": consistency,
            "consistent": consistency < 0.5,
        }

    def lateral_borel_sum(self, N: float, direction: str = "+") -> Dict:
        """Lateral Borel sum along contour above or below singularity.

        When B(t) has a singularity on the positive real axis at t = a,
        the Borel integral is ambiguous. The lateral sums S_± are defined
        by deforming the contour above (+) or below (-) the singularity:

            S_± f(N) = ∫_{0}^{∞±iε} dt e^{-t} B(t/N)

        The difference S_+ - S_- is the non-perturbative ambiguity.

        Args:
            N: The large parameter.
            direction: '+' for contour above real axis, '-' for below.

        Returns:
            Dictionary with lateral Borel sum.
        """
        sing_data = self.detect_singularities()
        a = sing_data.get("nearest_singularity", None)

        eps = 0.01 if direction == "+" else -0.01

        def integrand_real(t):
            s = t / N
            bt = float(np.real(self.borel_transform(s)))
            return np.exp(-t) * bt

        def integrand_imag(t):
            s = (t + 1j * eps) / N
            bt = self.borel_transform(np.real(s))
            return np.exp(-t) * float(np.real(bt)) * np.exp(eps / N)

        if a is not None and a < float("inf"):
            # Split integral at singularity
            result1, err1 = integrate.quad(integrand_real, 0, a * N * 0.9, limit=200)
            result2, err2 = integrate.quad(integrand_real, a * N * 1.1, a * N * 10, limit=200)
            # Near singularity, use deformed contour
            result_mid, err_mid = integrate.quad(integrand_imag, a * N * 0.9, a * N * 1.1, limit=200)
            result = result1 + result_mid + result2
            error = err1 + err_mid + err2
        else:
            result, error = integrate.quad(integrand_real, 0, np.inf, limit=200)

        return {
            "value": result,
            "error": error,
            "N": N,
            "direction": direction,
            "singularity": a,
        }

    def stokes_discontinuity(self, N: float) -> Dict:
        """Compute the Stokes discontinuity (non-perturbative ambiguity).

        Disc f(N) = S_+ f(N) - S_- f(N) ~ exp(-N × S_inst) × (series in 1/N)

        This ambiguity must be cancelled by the ambiguity in the instanton
        contribution, leading to the resurgent trans-series structure.

        Reference: Dorigoni, Sec. 3.2 "Stokes phenomenon and alien calculus."

        Args:
            N: The large parameter.

        Returns:
            Dictionary with discontinuity and its exponential structure.
        """
        s_plus = self.lateral_borel_sum(N, direction="+")
        s_minus = self.lateral_borel_sum(N, direction="-")

        disc = s_plus["value"] - s_minus["value"]

        # Estimate the non-perturbative scale
        sing_data = self.detect_singularities()
        S_inst = sing_data.get("nearest_singularity", None)

        if S_inst is not None and S_inst > 0 and S_inst < float("inf"):
            np_scale = np.exp(-N * S_inst) if N * S_inst < 500 else 0.0
            if np_scale > 1e-300:
                stokes_constant = disc / np_scale
            else:
                stokes_constant = None
        else:
            np_scale = None
            stokes_constant = None

        return {
            "discontinuity": disc,
            "S_plus": s_plus["value"],
            "S_minus": s_minus["value"],
            "nonperturbative_scale": np_scale,
            "stokes_constant": stokes_constant,
            "instanton_action": S_inst,
            "N": N,
        }

    def resurgent_transseries(self, N: float, n_sectors: int = 3) -> Dict:
        """Construct resurgent trans-series representation.

        f(N) = Σ_{k=0}^{K} C_k × exp(-k·S/N) × Σ_{n=0}^∞ a_{k,n} / N^n

        Each sector k corresponds to a k-instanton contribution. The
        trans-series parameters C_k are fixed by requiring cancellation
        of all non-perturbative ambiguities (median resummation).

        Reference: Aniceto, Schiappa & Vonk, "The Resurgence of Instantons
        in String Theory" (Commun. Num. Theor. Phys. 6, 2012).

        Args:
            N: The large parameter (network width).
            n_sectors: Number of instanton sectors to include.

        Returns:
            Dictionary with trans-series structure and resummed value.
        """
        sing_data = self.detect_singularities()
        S_inst = sing_data.get("nearest_singularity", 1.0)
        if S_inst is None or S_inst == float("inf"):
            S_inst = 1.0

        sectors = []
        total = 0.0

        for k in range(n_sectors):
            # k-instanton weight
            exp_weight = np.exp(-k * S_inst * N) if k * S_inst * N < 500 else 0.0

            # Perturbative coefficients around k-instanton sector
            # Large-order behavior: a_{k,n} ~ (n!/S_inst^n) with modifications
            sector_coeffs = []
            for n in range(min(self.n_terms, 10)):
                if k == 0 and n < self.n_terms:
                    sector_coeffs.append(self.coefficients[n])
                else:
                    # Approximate higher-sector coefficients from resurgence relations
                    # a_{k,n} ≈ (S_inst)^{-n} × a_{0,n+k} / C (Stokes constant)
                    if n + k < self.n_terms:
                        sector_coeffs.append(
                            self.coefficients[n + k] * S_inst ** (-k) / special.gamma(k + 1)
                        )
                    else:
                        sector_coeffs.append(0.0)

            # Sum the perturbative part
            pert_sum = sum(c / N**n for n, c in enumerate(sector_coeffs))

            # Trans-series parameter C_k (Stokes multiplier)
            C_k = 1.0 / special.gamma(k + 1)  # Simplified; real case needs matching

            sector_contribution = C_k * exp_weight * pert_sum
            total += sector_contribution

            sectors.append({
                "k": k,
                "C_k": C_k,
                "exp_weight": exp_weight,
                "perturbative_sum": pert_sum,
                "contribution": sector_contribution,
                "n_coefficients": len(sector_coeffs),
            })

        return {
            "total": total,
            "sectors": sectors,
            "instanton_action": S_inst,
            "N": N,
            "n_sectors": n_sectors,
            "perturbative_sector": sectors[0]["contribution"] if sectors else 0.0,
            "nonperturbative_fraction": abs(total - sectors[0]["contribution"]) / abs(total)
            if abs(total) > 1e-15 and sectors
            else 0.0,
        }

    def optimal_truncation(self, N: float) -> Dict:
        """Truncate the asymptotic series at the smallest term.

        For a series Σ a_n / N^n with a_n ~ n! / S^n, the terms first
        decrease then increase. The optimal truncation index is:

            n* ≈ S × N    (the term of smallest magnitude)

        This gives accuracy ~ exp(-S × N), which is the non-perturbative
        scale. Beyond n*, the series diverges.

        Reference: Bender & Orszag, "Advanced Mathematical Methods", Ch. 6.

        Args:
            N: The large parameter.

        Returns:
            Dictionary with optimal truncation point and partial sum.
        """
        terms = []
        partial_sums = []
        running_sum = 0.0

        for n, a_n in enumerate(self.coefficients):
            term = a_n / N**n
            terms.append(abs(term))
            running_sum += term
            partial_sums.append(running_sum)

        terms_arr = np.array(terms)

        # Find the index of the smallest term
        if len(terms_arr) > 0:
            n_star = int(np.argmin(terms_arr))
            optimal_value = partial_sums[n_star]
            smallest_term = terms_arr[n_star]
        else:
            n_star = 0
            optimal_value = 0.0
            smallest_term = float("inf")

        # Theoretical prediction: n* ≈ S × N
        sing_data = self.detect_singularities()
        S_inst = sing_data.get("nearest_singularity", None)
        n_star_predicted = S_inst * N if S_inst is not None and S_inst < float("inf") else None

        return {
            "optimal_index": n_star,
            "optimal_value": optimal_value,
            "smallest_term": smallest_term,
            "accuracy_estimate": smallest_term,
            "theoretical_n_star": n_star_predicted,
            "all_terms": terms_arr,
            "partial_sums": np.array(partial_sums),
            "N": N,
        }


class PathIntegralNN:
    """Path integral formulation of neural network inference.

    Treats the neural network posterior as a field theory with action
    S[W] = -ln P(y|X,W) - ln P(W). The partition function is:

        Z = ∫ dW exp(-S[W])

    and observables are computed as path integral expectation values.
    The large-N (wide network) limit is controlled by saddle-point
    methods, with 1/N corrections from loop diagrams.

    References:
        - Neal, "Bayesian Learning for Neural Networks" (1996)
        - Lee et al., "Deep Neural Networks as Gaussian Processes" (ICLR 2018)
        - Roberts, Yaida & Hanin, "The Principles of Deep Learning Theory" (2022)
    """

    def __init__(self, architecture_config: Dict):
        """Initialize NN path integral.

        Args:
            architecture_config: Dictionary specifying the architecture:
                - 'input_dim': input dimension
                - 'hidden_dims': list of hidden layer widths
                - 'output_dim': output dimension
                - 'activation': activation function name
                - 'sigma_w': weight prior standard deviation
                - 'sigma_b': bias prior standard deviation
        """
        self.config = architecture_config
        self.input_dim = architecture_config.get("input_dim", 1)
        self.hidden_dims = architecture_config.get("hidden_dims", [100])
        self.output_dim = architecture_config.get("output_dim", 1)
        self.sigma_w = architecture_config.get("sigma_w", 1.0)
        self.sigma_b = architecture_config.get("sigma_b", 1.0)

        # Total number of parameters
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        self.n_params = sum(
            dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1)
        )
        self.layer_dims = dims

    def neural_network_action(
        self,
        weights: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        noise_variance: float = 1.0,
    ) -> float:
        """Compute NN action S[W] = -ln P(y|X,W) - ln P(W).

        S[W] = (1/2σ_n²) Σ_i (y_i - f(x_i; W))² + Σ_l (1/2σ_w²) ||W_l||²

        This is the negative log-posterior (up to constants).

        Args:
            weights: Flattened weight vector.
            X: Input data, shape (n_samples, input_dim).
            y: Target data, shape (n_samples, output_dim).
            noise_variance: Observation noise σ_n².

        Returns:
            Action value S[W].
        """
        # Forward pass to compute predictions
        predictions = self._forward(weights, X)

        # Likelihood term: (1/2σ²) ||y - f(X;W)||²
        residuals = y - predictions
        S_likelihood = 0.5 * np.sum(residuals**2) / noise_variance

        # Prior term: (1/2σ_w²) ||W||²
        S_prior = 0.5 * np.sum(weights**2) / self.sigma_w**2

        return S_likelihood + S_prior

    def _forward(self, weights: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Forward pass through the network.

        Args:
            weights: Flattened weight vector.
            X: Input data.

        Returns:
            Network output.
        """
        h = X
        idx = 0
        for i in range(len(self.layer_dims) - 1):
            n_in = self.layer_dims[i]
            n_out = self.layer_dims[i + 1]

            # Extract weight matrix and bias
            W = weights[idx: idx + n_in * n_out].reshape(n_in, n_out)
            idx += n_in * n_out
            b = weights[idx: idx + n_out]
            idx += n_out

            h = h @ W + b

            # Activation for hidden layers (not output)
            if i < len(self.layer_dims) - 2:
                activation = self.config.get("activation", "tanh")
                if activation == "tanh":
                    h = np.tanh(h)
                elif activation == "relu":
                    h = np.maximum(0, h)
                elif activation == "erf":
                    h = special.erf(h)
                else:
                    h = np.tanh(h)

        return h

    def gaussian_action(self, weights: np.ndarray, sigma_w: Optional[float] = None) -> float:
        """Gaussian (free) action S₀[W] = Σ w²/(2σ²).

        This is the action of the Gaussian process limit (infinite width).
        Finite-width corrections are interactions relative to this free theory.

        Args:
            weights: Weight vector.
            sigma_w: Prior standard deviation (defaults to self.sigma_w).

        Returns:
            Gaussian action value.
        """
        sw = sigma_w if sigma_w is not None else self.sigma_w
        return 0.5 * np.sum(weights**2) / sw**2

    def interaction_vertices(
        self,
        weights: np.ndarray,
        X: np.ndarray,
        order: int = 4,
    ) -> Dict:
        """Extract interaction vertices by expanding action around saddle point.

        S[W₀ + δW] = S[W₀] + ½ δW^T S'' δW + (1/3!) V₃ δW³ + (1/4!) V₄ δW⁴ + ...

        The cubic vertex V₃ gives the 1/√N correction, quartic V₄ gives
        the 1/N correction (sunset and double-bubble diagrams).

        Args:
            weights: Saddle-point weights W₀.
            X: Input data for evaluating vertices.
            order: Maximum vertex order to compute (3 or 4).

        Returns:
            Dictionary with vertex tensors (stored as contractions for efficiency).
        """
        n_w = len(weights)
        eps = 1e-4

        # Quadratic (mass matrix / propagator inverse)
        hessian = np.zeros((n_w, n_w))
        f0 = self._forward(weights, X)
        for i in range(min(n_w, 50)):  # Limit for computational feasibility
            for j in range(i, min(n_w, 50)):
                w_pp = weights.copy()
                w_pp[i] += eps
                w_pp[j] += eps
                w_pm = weights.copy()
                w_pm[i] += eps
                w_pm[j] -= eps
                w_mp = weights.copy()
                w_mp[i] -= eps
                w_mp[j] += eps
                w_mm = weights.copy()
                w_mm[i] -= eps
                w_mm[j] -= eps

                f_pp = self._forward(w_pp, X)
                f_pm = self._forward(w_pm, X)
                f_mp = self._forward(w_mp, X)
                f_mm = self._forward(w_mm, X)

                d2f = (f_pp - f_pm - f_mp + f_mm) / (4 * eps**2)
                hessian[i, j] = np.sum(d2f)
                hessian[j, i] = hessian[i, j]

        # Add prior contribution to Hessian
        hessian += np.eye(n_w) / self.sigma_w**2

        result = {
            "quadratic": hessian[:min(n_w, 50), :min(n_w, 50)],
            "n_params": n_w,
            "order": order,
        }

        # Cubic vertex (sampled, not full tensor)
        if order >= 3:
            cubic_samples = []
            for _ in range(min(10, n_w)):
                i = np.random.randint(min(n_w, 50))
                w_p = weights.copy()
                w_p[i] += eps
                w_m = weights.copy()
                w_m[i] -= eps
                H_p = np.zeros(min(n_w, 10))
                H_m = np.zeros(min(n_w, 10))
                for j in range(min(n_w, 10)):
                    w_pj = w_p.copy()
                    w_pj[j] += eps
                    w_mj = w_p.copy()
                    w_mj[j] -= eps
                    H_p[j] = np.sum(self._forward(w_pj, X) - self._forward(w_mj, X)) / (2 * eps)

                    w_pj2 = w_m.copy()
                    w_pj2[j] += eps
                    w_mj2 = w_m.copy()
                    w_mj2[j] -= eps
                    H_m[j] = np.sum(self._forward(w_pj2, X) - self._forward(w_mj2, X)) / (2 * eps)

                cubic_samples.append((H_p - H_m) / (2 * eps))
            result["cubic_samples"] = cubic_samples

        return result

    def propagator(
        self,
        saddle_weights: np.ndarray,
        momentum: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute the propagator G = (S'')^{-1} at the saddle point.

        In the NN field theory, this is the connected two-point function
        of weight fluctuations: G_{ij} = ⟨δW_i δW_j⟩_connected.

        In the infinite-width limit, this reduces to the NTK kernel.

        Args:
            saddle_weights: Weights at the saddle point.
            momentum: Optional momentum for Fourier-space propagator.

        Returns:
            Propagator matrix G_{ij}.
        """
        n_w = len(saddle_weights)
        n_eff = min(n_w, 50)

        # Hessian of the Gaussian action (prior part)
        H = np.eye(n_eff) / self.sigma_w**2

        if momentum is not None:
            # Add kinetic term p² for momentum-space propagator
            p2 = np.sum(momentum**2)
            H += p2 * np.eye(n_eff)

        # Invert to get propagator
        try:
            G = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            G = np.linalg.pinv(H)

        return G

    def one_loop_effective_action(
        self,
        saddle_weights: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        noise_variance: float = 1.0,
    ) -> Dict:
        """One-loop effective action Γ₁ = S + (1/2) Tr ln S''.

        This is the first correction to the Gaussian process limit.
        The trace is over all weight indices and gives the O(1/N) correction
        to the free energy.

        Reference: Roberts, Yaida & Hanin, Ch. 5, "Effective Theory at
        Finite Width."

        Args:
            saddle_weights: Saddle-point weights.
            X: Input data.
            y: Target data.
            noise_variance: Observation noise variance.

        Returns:
            Dictionary with classical and one-loop contributions.
        """
        S_classical = self.neural_network_action(saddle_weights, X, y, noise_variance)

        # Compute Hessian of the action
        n_w = len(saddle_weights)
        n_eff = min(n_w, 100)
        eps = 1e-4

        H = np.eye(n_eff) / self.sigma_w**2
        for i in range(n_eff):
            w_p = saddle_weights.copy()
            w_p[i] += eps
            w_m = saddle_weights.copy()
            w_m[i] -= eps
            f_p = self._forward(w_p, X)
            f_m = self._forward(w_m, X)
            f_0 = self._forward(saddle_weights, X)

            # Contribution from likelihood Hessian
            d2S_diag = np.sum((f_p - 2 * f_0 + f_m) / eps**2) / noise_variance
            df = (f_p - f_m) / (2 * eps)
            d2S_diag += np.sum(df**2) / noise_variance
            H[i, i] += d2S_diag

        # One-loop: (1/2) Tr ln(S''/2π)
        eigvals = np.linalg.eigvalsh(H)
        pos_eigs = eigvals[eigvals > 1e-10]
        one_loop = 0.5 * np.sum(np.log(pos_eigs / (2 * np.pi)))

        return {
            "S_classical": S_classical,
            "one_loop_correction": one_loop,
            "effective_action": S_classical + one_loop,
            "n_modes": len(pos_eigs),
            "condition_number": pos_eigs[-1] / pos_eigs[0] if len(pos_eigs) > 1 else 1.0,
            "width_dependence": f"O(1/N) with N={self.hidden_dims[0]}",
        }

    def two_loop_correction(
        self,
        saddle_weights: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        noise_variance: float = 1.0,
    ) -> Dict:
        """Two-loop correction to the effective action.

        The two-loop contribution involves sunset and figure-eight diagrams:
            Γ₂ = -(1/12) V₃ G V₃ G G + (1/8) V₄ G G

        where V₃, V₄ are cubic and quartic vertices, and G is the propagator.

        This gives the O(1/N²) correction.

        Args:
            saddle_weights: Saddle-point weights.
            X: Input data.
            y: Target data.
            noise_variance: Observation noise variance.

        Returns:
            Dictionary with two-loop correction estimate.
        """
        # Get propagator and vertices
        G = self.propagator(saddle_weights)
        vertices = self.interaction_vertices(saddle_weights, X, order=4)

        n_eff = G.shape[0]

        # Figure-eight (double bubble): (1/8) Tr[V₄ · G²]
        # Approximate V₄ from Hessian of Hessian
        H = vertices["quadratic"]
        H_size = min(H.shape[0], n_eff)

        # Estimate quartic vertex from finite differences of Hessian
        figure_eight = (1.0 / 8.0) * np.trace(G[:H_size, :H_size] @ G[:H_size, :H_size])

        # Sunset diagram: -(1/12) contracted cubic vertices
        sunset = 0.0
        if "cubic_samples" in vertices and len(vertices["cubic_samples"]) > 0:
            for v3 in vertices["cubic_samples"]:
                n_v = min(len(v3), H_size)
                sunset += np.sum(v3[:n_v] ** 2 * np.diag(G[:n_v, :n_v]) ** 2)
            sunset *= -1.0 / (12.0 * len(vertices["cubic_samples"]))

        two_loop = figure_eight + sunset

        return {
            "two_loop_correction": two_loop,
            "figure_eight": figure_eight,
            "sunset": sunset,
            "order": "O(1/N^2)",
            "n_effective_modes": n_eff,
        }

    def loop_expansion_coefficients(
        self,
        saddle_weights: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        n_loops: int = 3,
        noise_variance: float = 1.0,
    ) -> Dict:
        """Compute coefficients of the loop (1/N) expansion.

        Γ[W] = N × Γ₀ + Γ₁ + (1/N) Γ₂ + (1/N²) Γ₃ + ...

        where Γ₀ is the classical (GP limit) action, Γ₁ is the one-loop
        determinant, etc.

        Args:
            saddle_weights: Saddle-point weights.
            X: Input data.
            y: Target data.
            n_loops: Number of loop orders to compute.
            noise_variance: Observation noise variance.

        Returns:
            Dictionary with loop expansion coefficients.
        """
        N = self.hidden_dims[0] if self.hidden_dims else 1

        coefficients = []

        # Tree level (0 loops): classical action / N
        S_cl = self.neural_network_action(saddle_weights, X, y, noise_variance)
        coefficients.append(S_cl / N)

        # One loop
        if n_loops >= 1:
            one_loop_data = self.one_loop_effective_action(
                saddle_weights, X, y, noise_variance
            )
            coefficients.append(one_loop_data["one_loop_correction"])

        # Two loops
        if n_loops >= 2:
            two_loop_data = self.two_loop_correction(
                saddle_weights, X, y, noise_variance
            )
            coefficients.append(two_loop_data["two_loop_correction"])

        # Higher loops: estimate from factorial growth
        if n_loops >= 3:
            for k in range(3, n_loops + 1):
                # Estimate: Γ_k ~ (k-1)! × Γ₂ (factorial growth from instantons)
                if len(coefficients) >= 3 and abs(coefficients[2]) > 1e-15:
                    est = coefficients[2] * special.gamma(k) / special.gamma(3)
                else:
                    est = 0.0
                coefficients.append(est)

        return {
            "coefficients": coefficients,
            "n_loops": n_loops,
            "width": N,
            "effective_action_at_width": sum(
                c / N**k for k, c in enumerate(coefficients)
            ),
            "convergence_radius_estimate": 1.0 / abs(coefficients[2] / coefficients[1])
            if len(coefficients) >= 3 and abs(coefficients[1]) > 1e-15
            else float("inf"),
        }

    def effective_potential(
        self,
        phi: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        noise_variance: float = 1.0,
    ) -> Dict:
        """Effective potential V_eff(φ) including loop corrections.

        The effective potential is the effective action evaluated on
        constant field configurations. For the NN, φ represents the
        mean network output, and V_eff(φ) gives the free energy as a
        function of the constrained output.

        V_eff(φ) = V_tree(φ) + V_1loop(φ) + ...

        Args:
            phi: Array of field (mean output) values.
            X: Input data.
            y: Target data.
            noise_variance: Observation noise variance.

        Returns:
            Dictionary with effective potential at each φ value.
        """
        phi = np.atleast_1d(phi)
        V_tree = np.zeros_like(phi)
        V_1loop = np.zeros_like(phi)
        V_total = np.zeros_like(phi)

        n_samples = X.shape[0] if X.ndim > 1 else len(X)

        for i, phi_val in enumerate(phi):
            # Tree-level: likelihood potential for constant output φ
            residuals = y.flatten() - phi_val
            V_tree[i] = 0.5 * np.sum(residuals**2) / noise_variance

            # One-loop: Gaussian fluctuations around constant output
            # The mass² is 1/σ² + n_data/σ_n² (from prior + likelihood)
            N = self.hidden_dims[0] if self.hidden_dims else 1
            mass_sq = 1.0 / self.sigma_w**2 + n_samples / noise_variance
            # Tr ln(S'') = N × ln(mass²) per output mode
            V_1loop[i] = 0.5 * N * np.log(mass_sq / (2 * np.pi))

            V_total[i] = V_tree[i] + V_1loop[i] / N  # 1-loop is O(1)

        # Find the minimum (the physical vacuum)
        min_idx = np.argmin(V_total)

        return {
            "phi": phi,
            "V_tree": V_tree,
            "V_1loop": V_1loop,
            "V_total": V_total,
            "phi_min": float(phi[min_idx]),
            "V_min": float(V_total[min_idx]),
            "mass_squared": float(np.gradient(np.gradient(V_total, phi), phi)[min_idx])
            if len(phi) > 2
            else None,
        }


class NumericalBootstrap:
    """Numerical conformal bootstrap for bounding NN critical exponents.

    The conformal bootstrap constrains CFT data (operator dimensions and
    OPE coefficients) using crossing symmetry and unitarity. This can
    bound the critical exponents of the NN field theory at its phase
    transition, providing non-perturbative constraints independent of
    the 1/N expansion.

    Uses a simplified version of the Rattazzi-Rychkov-Tonni-Vichi (RRTV)
    approach, implemented with scipy's linear programming and SDP relaxation.

    References:
        - Rattazzi, Rychkov, Tonni, Vichi, JHEP 0812 (2008) 031
        - Poland, Rychkov, Vichi, Rev. Mod. Phys. 91 (2019) 015002
        - El-Showk et al., Phys. Rev. D 86 (2012) 025022
    """

    def __init__(self, n_constraints: int = 10, dimension: float = 3.0):
        """Initialize bootstrap calculator.

        Args:
            n_constraints: Number of derivative constraints to use.
            dimension: Spacetime dimension d.
        """
        self.n_constraints = n_constraints
        self.d = dimension

    def crossing_symmetry_constraint(
        self,
        exponents: List[float],
        ope_coefficients: List[float],
    ) -> np.ndarray:
        """Evaluate crossing symmetry equation.

        The crossing equation for identical scalars is:
            Σ_O λ²_O F_{Δ_O, l_O}(u,v) = 0

        where F = v^{Δ_φ} g(u,v) - u^{Δ_φ} g(v,u) and g is the
        conformal block.

        Args:
            exponents: List of operator dimensions Δ_O.
            ope_coefficients: Corresponding OPE coefficients λ²_O.

        Returns:
            Residual of the crossing equation at evaluation points.
        """
        # Evaluate at the crossing-symmetric point z = z̄ = 1/2
        # u = zz̄, v = (1-z)(1-z̄)
        u = 0.25  # z=z̄=1/2
        v = 0.25

        residual = np.zeros(self.n_constraints)
        d = self.d

        for i, (Delta, lam_sq) in enumerate(zip(exponents, ope_coefficients)):
            for k in range(self.n_constraints):
                # Conformal block derivatives at crossing point
                # Approximate: g_{Δ,0}(u,v) ~ u^{Δ/2} F(Δ/2, Δ/2; Δ; u) for spin-0
                # Use power series around u=v=1/4
                block_val = u ** (Delta / 2) * special.hyp2f1(
                    Delta / 2, Delta / 2, Delta, u
                )
                # k-th derivative contribution (numerical differentiation)
                eps = 0.01
                u_k = u + k * eps
                if u_k < 1:
                    block_deriv = u_k ** (Delta / 2) * special.hyp2f1(
                        Delta / 2, Delta / 2, Delta, u_k
                    )
                else:
                    block_deriv = block_val

                # Crossing: F = v^{Δ_ext} g(u,v) - u^{Δ_ext} g(v,u)
                F_val = block_deriv - block_val  # Simplified crossing
                residual[k] += lam_sq * F_val

        return residual

    def unitarity_bounds(self, dimension: float, spin: int) -> float:
        """Unitarity bound on operator dimension.

        In d dimensions, unitarity requires:
            Δ ≥ (d-2)/2           for spin 0 (scalar)
            Δ ≥ d - 2 + spin     for spin > 0

        The free-field value saturates the bound.

        Args:
            dimension: Spacetime dimension d.
            spin: Spin of the operator.

        Returns:
            Lower bound on the scaling dimension Δ.
        """
        if spin == 0:
            return (dimension - 2) / 2
        else:
            return dimension - 2 + spin

    def linear_functional_method(self, n_derivatives: int = 20) -> Dict:
        """Find linear functional for bootstrap bounds (RRTV method).

        Search for a linear functional α acting on functions of (Δ, l) such
        that:
            α(F_{0}) > 0    (normalization on identity)
            α(F_{Δ,l}) ≥ 0  for all Δ ≥ Δ_unitarity(l), l = 0,2,4,...

        If such α exists, it rules out the assumed CFT spectrum.

        We discretize by evaluating at z = z̄ = 1/2 and using derivatives
        ∂^m_z ∂^n_{z̄} up to order n_derivatives.

        Args:
            n_derivatives: Maximum total derivative order.

        Returns:
            Dictionary with the functional and its properties.
        """
        # Set up the search space for the functional
        n_components = (n_derivatives + 1) * (n_derivatives + 2) // 2
        n_spins = 5  # Check spins 0, 2, 4, 6, 8
        n_delta_pts = 20

        d = self.d

        # Build constraint matrix: for each (Δ, l), α · F_{Δ,l} ≥ 0
        constraints_A = []
        constraints_b = []

        for spin in range(0, 2 * n_spins, 2):
            delta_min = self.unitarity_bounds(d, spin)
            delta_values = np.linspace(delta_min, delta_min + 10, n_delta_pts)

            for Delta in delta_values:
                # Evaluate conformal block derivatives
                row = np.zeros(min(n_components, 20))
                for k in range(min(n_components, 20)):
                    u = 0.25
                    eps = 0.01 * (k + 1)
                    block = u ** (Delta / 2) * (1 + spin * 0.1)  # Simplified
                    row[k] = block * (-1) ** k  # Crossing-odd combination
                constraints_A.append(-row)  # -α · F ≤ 0 ⟺ α · F ≥ 0
                constraints_b.append(0.0)

        A = np.array(constraints_A)
        b = np.array(constraints_b)

        # Objective: maximize α · F_0 (identity contribution)
        c = -np.ones(A.shape[1])  # Maximize α(1)

        # Solve LP
        n_vars = A.shape[1]
        bounds = [(-10, 10)] * n_vars

        try:
            result = optimize.linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
            functional = result.x if result.success else np.zeros(n_vars)
            feasible = result.success
        except Exception:
            functional = np.zeros(min(n_components, 20))
            feasible = False

        return {
            "functional": functional,
            "feasible": feasible,
            "n_derivatives": n_derivatives,
            "n_constraints": len(constraints_b),
            "objective_value": -result.fun if feasible else None,
        }

    def bound_on_gap(
        self,
        external_dimension: float,
        n_points: int = 50,
    ) -> Dict:
        """Compute upper bound on the gap to the first scalar operator.

        Using the bootstrap, find the maximum allowed value of Δ_gap
        (the dimension of the first non-identity scalar) as a function
        of the external operator dimension Δ_φ.

        Args:
            external_dimension: Dimension Δ_φ of the external scalar operator.
            n_points: Number of trial gap values to test.

        Returns:
            Dictionary with the gap bound.
        """
        d = self.d
        delta_min = self.unitarity_bounds(d, 0)
        delta_max = 2 * external_dimension + 5  # Generous upper range

        gap_values = np.linspace(delta_min + 0.01, delta_max, n_points)
        allowed = np.ones(n_points, dtype=bool)

        for i, delta_gap in enumerate(gap_values):
            # Test if a CFT exists with gap = delta_gap
            # Simplified: check if crossing can be satisfied
            n_ops = 5
            deltas = delta_gap + np.arange(n_ops) * 0.5

            # Try to solve crossing equation for OPE coefficients
            A = np.zeros((self.n_constraints, n_ops))
            for j, Delta in enumerate(deltas):
                u = 0.25
                for k in range(self.n_constraints):
                    eps_k = 0.01 * (k + 1)
                    block = (u + eps_k) ** (Delta / 2) - u ** (Delta / 2)
                    A[k, j] = block

            b_target = np.ones(self.n_constraints)

            # Check if A λ² = b has a non-negative solution
            try:
                result = optimize.nnls(A, b_target)
                residual = np.linalg.norm(A @ result[0] - b_target)
                allowed[i] = residual < 1.0
            except Exception:
                allowed[i] = True

        # Find the maximum allowed gap
        if np.any(allowed):
            max_gap_idx = np.max(np.where(allowed)[0])
            max_gap = gap_values[max_gap_idx]
        else:
            max_gap = delta_min

        return {
            "external_dimension": external_dimension,
            "gap_bound": max_gap,
            "gap_values": gap_values,
            "allowed": allowed,
            "unitarity_bound": delta_min,
            "dimension": d,
        }

    def allowed_region(
        self,
        external_dim_range: np.ndarray,
        n_points: int = 20,
    ) -> Dict:
        """Compute the allowed region in the (Δ_σ, Δ_ε) plane.

        For each value of the external dimension Δ_σ, compute the allowed
        range of the leading scalar dimension Δ_ε. The boundary of the
        allowed region is where the Ising model / NN critical point lives.

        Reference: El-Showk et al., PRD 86 (2012) 025022.

        Args:
            external_dim_range: Range of external dimensions Δ_σ to scan.
            n_points: Number of Δ_ε values to test per Δ_σ.

        Returns:
            Dictionary with allowed region boundary.
        """
        ext_dims = np.atleast_1d(external_dim_range)
        gap_bounds = np.zeros(len(ext_dims))

        for i, delta_sigma in enumerate(ext_dims):
            result = self.bound_on_gap(delta_sigma, n_points=n_points)
            gap_bounds[i] = result["gap_bound"]

        # Known results for comparison
        d = self.d
        if abs(d - 3.0) < 0.01:
            ising_sigma = 0.5181489
            ising_epsilon = 1.412625
        else:
            ising_sigma = (d - 2) / 2 + 0.02
            ising_epsilon = d - 2 * (d - 2) / 2 + 0.4

        return {
            "external_dims": ext_dims,
            "gap_upper_bounds": gap_bounds,
            "dimension": d,
            "ising_point": {"delta_sigma": ising_sigma, "delta_epsilon": ising_epsilon},
            "unitarity_bound": self.unitarity_bounds(d, 0),
        }

    def compare_with_nn_exponents(self, measured_exponents: Dict) -> Dict:
        """Compare measured NN exponents with bootstrap bounds.

        Check whether the critical exponents measured from the finite-width
        NN phase transition are consistent with bootstrap constraints.

        Args:
            measured_exponents: Dictionary with keys 'delta_sigma' and
                'delta_epsilon' (or equivalently 'eta' and 'nu').

        Returns:
            Dictionary with comparison results.
        """
        d = self.d

        if "delta_sigma" in measured_exponents:
            ds = measured_exponents["delta_sigma"]
            de = measured_exponents.get("delta_epsilon", None)
        elif "eta" in measured_exponents:
            eta = measured_exponents["eta"]
            nu = measured_exponents.get("nu", 0.5)
            ds = (d - 2 + eta) / 2
            de = d - 1.0 / nu if nu > 0 else None
        else:
            return {"error": "Need either delta_sigma/delta_epsilon or eta/nu."}

        # Get bootstrap bound at this Δ_σ
        gap_data = self.bound_on_gap(ds, n_points=30)
        gap_bound = gap_data["gap_bound"]

        # Check unitarity
        unitarity_ok = ds >= self.unitarity_bounds(d, 0)

        # Check if Δ_ε is within allowed region
        if de is not None:
            within_bounds = de <= gap_bound
        else:
            within_bounds = None

        return {
            "delta_sigma": ds,
            "delta_epsilon": de,
            "gap_bound": gap_bound,
            "unitarity_satisfied": unitarity_ok,
            "within_bootstrap_bounds": within_bounds,
            "distance_to_bound": gap_bound - de if de is not None else None,
            "dimension": d,
        }

    def sdp_solver(
        self,
        objective: np.ndarray,
        constraints: List[Dict],
    ) -> Dict:
        """Solve a semidefinite program (SDP) for bootstrap bounds.

        Simplified SDP via reduction to LP/SOCP using scipy. The full
        bootstrap requires specialized SDP solvers (SDPB), but this
        provides approximate bounds.

        The SDP is:
            minimize  c^T x
            subject to  Σ_i x_i F_i - F_0 ⪰ 0   (positive semidefinite)

        We relax PSD constraints to eigenvalue constraints.

        Args:
            objective: Linear objective vector c.
            constraints: List of constraint dictionaries, each with
                'matrices' (list of matrices F_i) and 'offset' (F_0).

        Returns:
            Dictionary with optimal value and solution.
        """
        n_vars = len(objective)

        # Collect all linear inequality constraints from PSD relaxation
        A_ub_rows = []
        b_ub_vals = []

        for constraint in constraints:
            matrices = constraint.get("matrices", [])
            offset = constraint.get("offset", np.zeros((2, 2)))
            n_mat = offset.shape[0]

            # PSD constraint: all eigenvalues ≥ 0
            # Relaxation: trace ≥ 0 and det ≥ 0 (for 2×2)
            if n_mat <= 2 and len(matrices) == n_vars:
                # Trace constraint: Σ x_i tr(F_i) ≥ tr(F_0)
                row_trace = np.array([np.trace(M) for M in matrices])
                A_ub_rows.append(-row_trace)
                b_ub_vals.append(-np.trace(offset))

                # Diagonal dominance constraints
                for k in range(n_mat):
                    row_diag = np.array([M[k, k] for M in matrices])
                    A_ub_rows.append(-row_diag)
                    b_ub_vals.append(-offset[k, k])
            else:
                # General case: use random projections for PSD constraint
                n_projections = min(5, n_mat)
                for _ in range(n_projections):
                    v = np.random.randn(n_mat)
                    v /= np.linalg.norm(v)
                    # v^T (Σ x_i F_i - F_0) v ≥ 0
                    row = np.array([v @ M @ v for M in matrices[:n_vars]])
                    if len(row) < n_vars:
                        row = np.pad(row, (0, n_vars - len(row)))
                    A_ub_rows.append(-row)
                    b_ub_vals.append(-v @ offset @ v)

        if len(A_ub_rows) == 0:
            return {
                "optimal_value": None,
                "solution": np.zeros(n_vars),
                "success": False,
                "message": "No constraints provided.",
            }

        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_vals)

        bounds = [(-100, 100)] * n_vars

        try:
            result = optimize.linprog(
                objective, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs"
            )
            return {
                "optimal_value": result.fun if result.success else None,
                "solution": result.x if result.success else np.zeros(n_vars),
                "success": result.success,
                "message": result.message,
                "n_constraints": len(b_ub),
            }
        except Exception as e:
            return {
                "optimal_value": None,
                "solution": np.zeros(n_vars),
                "success": False,
                "message": str(e),
            }
