"""
Free energy landscape analysis for neural network mean-field theory.

Provides tools for computing free energy surfaces, solving saddle-point equations,
finding minimum energy paths, detecting phase transitions, and classifying
transition order from order-parameter behavior.
"""

import numpy as np
from scipy import optimize, interpolate, integrate, linalg
from scipy.ndimage import minimum_filter
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple, Dict, Any


@dataclass
class SaddlePointResult:
    """Result from a saddle-point solve."""
    solution: np.ndarray
    converged: bool
    residual: float
    iterations: int
    jacobian: Optional[np.ndarray] = None


@dataclass
class PathResult:
    """Result from a minimum-energy-path calculation."""
    images: np.ndarray
    energies: np.ndarray
    barrier_height: float
    converged: bool
    iterations: int


@dataclass
class TransitionInfo:
    """Information about a detected phase transition."""
    location: float
    order: int  # 1 = first-order, 2 = second-order
    latent_heat: Optional[float] = None
    critical_exponents: Optional[Dict[str, float]] = None
    landau_coefficients: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------------
# 1. FreeEnergyLandscape
# ---------------------------------------------------------------------------

class FreeEnergyLandscape:
    """Compute free energy as a function of order parameters.

    The free energy in mean-field theory takes the form
        F(q, q̂) = -T ln Z(q, q̂)
    where the partition function is evaluated via a saddle-point
    approximation over the conjugate order parameters.

    Parameters
    ----------
    temperature : float
        Temperature T > 0.
    coupling_matrix : array_like or None
        Symmetric coupling matrix J_{ij}.  If *None* a 2×2 identity is used.
    external_field : array_like or None
        External field h_i.  If *None* the field is zero.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        coupling_matrix: Optional[np.ndarray] = None,
        external_field: Optional[np.ndarray] = None,
    ):
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        self.temperature = temperature
        self.beta = 1.0 / temperature

        if coupling_matrix is not None:
            self.J = np.asarray(coupling_matrix, dtype=np.float64)
            if self.J.ndim != 2 or self.J.shape[0] != self.J.shape[1]:
                raise ValueError("coupling_matrix must be square.")
            self.dim = self.J.shape[0]
        else:
            self.dim = 2
            self.J = np.eye(self.dim, dtype=np.float64)

        if external_field is not None:
            self.h = np.asarray(external_field, dtype=np.float64).ravel()
            if self.h.shape[0] != self.dim:
                raise ValueError("external_field length must match coupling_matrix dimension.")
        else:
            self.h = np.zeros(self.dim, dtype=np.float64)

    # ----- core energy -------------------------------------------------

    def compute_free_energy(self, order_params: np.ndarray) -> float:
        r"""Compute F(q) = -T \ln Z via mean-field approximation.

        The mean-field free energy density is
            F = -½ q^T J q  -  h^T q  +  T * Σ_i s(q_i)
        where s(q) = ½[(1+q)ln(1+q) + (1−q)ln(1−q)] is the binary-entropy
        contribution (Ising-type mean-field).
        """
        q = np.asarray(order_params, dtype=np.float64).ravel()
        if q.shape[0] != self.dim:
            raise ValueError(f"Expected {self.dim} order parameters, got {q.shape[0]}.")

        # Clip to avoid log(0)
        q_clip = np.clip(q, -1.0 + 1e-14, 1.0 - 1e-14)

        energy_term = -0.5 * q_clip @ self.J @ q_clip - self.h @ q_clip

        # Entropic (mean-field) term: T * Σ s(q_i)
        sp = 1.0 + q_clip
        sm = 1.0 - q_clip
        entropy_term = 0.5 * (sp * np.log(sp) + sm * np.log(sm))
        entropic = self.temperature * np.sum(entropy_term)

        return float(energy_term + entropic)

    # ----- 2-D surface -------------------------------------------------

    def free_energy_surface(
        self,
        q_range: Tuple[float, float] = (-0.99, 0.99),
        qhat_range: Tuple[float, float] = (-0.99, 0.99),
        resolution: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the free energy on a 2-D grid of the first two order
        parameters, holding any remaining parameters at zero.

        Returns (Q1, Q2, F) meshgrids.
        """
        q1 = np.linspace(q_range[0], q_range[1], resolution)
        q2 = np.linspace(qhat_range[0], qhat_range[1], resolution)
        Q1, Q2 = np.meshgrid(q1, q2, indexing="ij")
        F = np.empty_like(Q1)

        base = np.zeros(self.dim)
        for i in range(resolution):
            for j in range(resolution):
                params = base.copy()
                params[0] = Q1[i, j]
                if self.dim > 1:
                    params[1] = Q2[i, j]
                F[i, j] = self.compute_free_energy(params)

        return Q1, Q2, F

    # ----- gradient & hessian ------------------------------------------

    def gradient_of_free_energy(self, order_params: np.ndarray) -> np.ndarray:
        r"""Analytic gradient ∂F/∂q_i.

        ∂F/∂q_i = -(J q)_i - h_i + T * arctanh(q_i)
        """
        q = np.asarray(order_params, dtype=np.float64).ravel()
        q_clip = np.clip(q, -1.0 + 1e-14, 1.0 - 1e-14)
        grad = -(self.J @ q_clip) - self.h + self.temperature * np.arctanh(q_clip)
        return grad

    def hessian_of_free_energy(self, order_params: np.ndarray) -> np.ndarray:
        r"""Analytic Hessian ∂²F/∂q_i∂q_j.

        H_{ij} = -J_{ij} + T/(1 - q_i²) δ_{ij}
        """
        q = np.asarray(order_params, dtype=np.float64).ravel()
        q_clip = np.clip(q, -1.0 + 1e-14, 1.0 - 1e-14)
        diag = self.temperature / (1.0 - q_clip ** 2)
        H = -self.J.copy() + np.diag(diag)
        return H

    # ----- minima search -----------------------------------------------

    def find_minima(
        self,
        n_starts: int = 20,
        bounds: Optional[Tuple[float, float]] = None,
        tol: float = 1e-10,
    ) -> List[Dict[str, Any]]:
        """Multi-start L-BFGS-B optimisation to find all local minima.

        Returns a list of dicts with keys 'position', 'energy', 'hessian_eigenvalues'.
        """
        if bounds is None:
            lo, hi = -0.95, 0.95
        else:
            lo, hi = bounds
        bnds = [(lo, hi)] * self.dim

        raw_minima: List[Tuple[float, np.ndarray]] = []
        rng = np.random.default_rng(42)

        for _ in range(n_starts):
            x0 = rng.uniform(lo, hi, size=self.dim)
            res = optimize.minimize(
                self.compute_free_energy,
                x0,
                jac=self.gradient_of_free_energy,
                method="L-BFGS-B",
                bounds=bnds,
                options={"ftol": tol, "gtol": tol, "maxiter": 2000},
            )
            if res.success:
                raw_minima.append((float(res.fun), res.x.copy()))

        # Deduplicate
        unique: List[Tuple[float, np.ndarray]] = []
        for e, x in sorted(raw_minima, key=lambda t: t[0]):
            if not any(np.allclose(x, u[1], atol=1e-6) for u in unique):
                unique.append((e, x))

        results = []
        for e, x in unique:
            H = self.hessian_of_free_energy(x)
            eigvals = np.linalg.eigvalsh(H)
            results.append({
                "position": x,
                "energy": e,
                "hessian_eigenvalues": eigvals,
            })
        return results

    # ----- thermodynamic derivatives -----------------------------------

    def entropy_from_free_energy(
        self,
        order_params: np.ndarray,
        temperature: Optional[float] = None,
        dT: float = 1e-5,
    ) -> float:
        """Numerical entropy S = -∂F/∂T via centred finite difference."""
        T0 = temperature if temperature is not None else self.temperature

        saved = self.temperature, self.beta
        self.temperature = T0 + dT
        self.beta = 1.0 / self.temperature
        Fp = self.compute_free_energy(order_params)

        self.temperature = T0 - dT
        self.beta = 1.0 / self.temperature
        Fm = self.compute_free_energy(order_params)

        self.temperature, self.beta = saved

        return float(-(Fp - Fm) / (2.0 * dT))

    def internal_energy(self, order_params: np.ndarray) -> float:
        """Internal energy U = F + T S."""
        F = self.compute_free_energy(order_params)
        S = self.entropy_from_free_energy(order_params)
        return F + self.temperature * S


# ---------------------------------------------------------------------------
# 2. SaddlePointSolver
# ---------------------------------------------------------------------------

class SaddlePointSolver:
    """Solve saddle-point equations arising from mean-field actions.

    Given an action S(q, q̂), the saddle-point equations are
        ∂S/∂q = 0,  ∂S/∂q̂ = 0
    which determine the mean-field order parameters.

    Parameters
    ----------
    landscape : FreeEnergyLandscape or None
        If provided, the solver uses its gradient as the equation system.
    tol : float
        Convergence tolerance on the residual norm.
    max_iter : int
        Maximum Newton iterations.
    """

    def __init__(
        self,
        landscape: Optional[FreeEnergyLandscape] = None,
        tol: float = 1e-12,
        max_iter: int = 500,
    ):
        self.landscape = landscape
        self.tol = tol
        self.max_iter = max_iter

    # ----- Newton solver -----------------------------------------------

    def solve_saddle_point(
        self,
        initial_guess: np.ndarray,
        method: str = "newton",
        equations: Optional[Callable] = None,
        jacobian: Optional[Callable] = None,
    ) -> SaddlePointResult:
        """Solve the saddle-point equations starting from *initial_guess*.

        Parameters
        ----------
        method : {'newton', 'hybr', 'broyden'}
        equations : callable, optional
            F(x) = 0 system.  Defaults to landscape gradient.
        jacobian : callable, optional
            Jacobian of *equations*.  Defaults to landscape Hessian.
        """
        x0 = np.asarray(initial_guess, dtype=np.float64).ravel()

        if equations is None:
            if self.landscape is None:
                raise ValueError("Provide either a landscape or explicit equations.")
            equations = self.landscape.gradient_of_free_energy
        if jacobian is None and self.landscape is not None:
            jacobian = self.landscape.hessian_of_free_energy

        if method == "newton":
            return self._newton_solve(x0, equations, jacobian)
        elif method in ("hybr", "broyden"):
            return self._scipy_solve(x0, equations, jacobian, method)
        else:
            raise ValueError(f"Unknown method '{method}'.")

    def _newton_solve(
        self,
        x: np.ndarray,
        F: Callable,
        J: Optional[Callable],
    ) -> SaddlePointResult:
        eps_fd = 1e-7
        for it in range(1, self.max_iter + 1):
            fx = F(x)
            res = np.linalg.norm(fx)
            if res < self.tol:
                jac = J(x) if J is not None else self._numerical_jacobian(F, x, eps_fd)
                return SaddlePointResult(x.copy(), True, res, it, jac)

            if J is not None:
                jac = J(x)
            else:
                jac = self._numerical_jacobian(F, x, eps_fd)

            try:
                dx = np.linalg.solve(jac, -fx)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(jac, -fx, rcond=None)[0]

            # Line-search (backtracking Armijo)
            alpha = 1.0
            for _ in range(30):
                x_new = x + alpha * dx
                if np.linalg.norm(F(x_new)) < res:
                    break
                alpha *= 0.5
            x = x_new

        return SaddlePointResult(x.copy(), False, float(np.linalg.norm(F(x))), self.max_iter, None)

    @staticmethod
    def _numerical_jacobian(F: Callable, x: np.ndarray, eps: float) -> np.ndarray:
        n = x.shape[0]
        f0 = F(x)
        m = f0.shape[0]
        J = np.empty((m, n))
        for j in range(n):
            xp = x.copy()
            xp[j] += eps
            J[:, j] = (F(xp) - f0) / eps
        return J

    def _scipy_solve(
        self,
        x0: np.ndarray,
        equations: Callable,
        jacobian: Optional[Callable],
        method: str,
    ) -> SaddlePointResult:
        jac_arg = jacobian if jacobian is not None else False
        if method == "broyden":
            method = "broyden1"
            sol = optimize.root(equations, x0, method=method, tol=self.tol,
                                options={"maxiter": self.max_iter})
        else:
            sol = optimize.root(equations, x0, jac=jac_arg, method=method,
                                tol=self.tol, options={"maxiter": self.max_iter})
        return SaddlePointResult(
            solution=sol.x.copy(),
            converged=sol.success,
            residual=float(np.linalg.norm(sol.fun)),
            iterations=sol.nfev,
            jacobian=None,
        )

    # ----- saddle point from arbitrary action --------------------------

    def saddle_point_from_action(
        self,
        action_func: Callable[[np.ndarray], float],
        variables: np.ndarray,
        eps: float = 1e-7,
    ) -> SaddlePointResult:
        """Find the saddle point of an arbitrary scalar action S(x).

        Uses numerical gradient and Hessian of *action_func*.
        """
        x0 = np.asarray(variables, dtype=np.float64).ravel()

        def grad(x: np.ndarray) -> np.ndarray:
            g = np.empty_like(x)
            s0 = action_func(x)
            for i in range(x.shape[0]):
                xp = x.copy(); xp[i] += eps
                g[i] = (action_func(xp) - s0) / eps
            return g

        def hess(x: np.ndarray) -> np.ndarray:
            n = x.shape[0]
            H = np.empty((n, n))
            for i in range(n):
                for j in range(i, n):
                    xpp = x.copy(); xpp[i] += eps; xpp[j] += eps
                    xpm = x.copy(); xpm[i] += eps; xpm[j] -= eps
                    xmp = x.copy(); xmp[i] -= eps; xmp[j] += eps
                    xmm = x.copy(); xmm[i] -= eps; xmm[j] -= eps
                    H[i, j] = (action_func(xpp) - action_func(xpm)
                                - action_func(xmp) + action_func(xmm)) / (4.0 * eps * eps)
                    H[j, i] = H[i, j]
            return H

        return self._newton_solve(x0, grad, hess)

    # ----- continuation method -----------------------------------------

    def continuation_method(
        self,
        equations: Callable[[np.ndarray, float], np.ndarray],
        param_name: str,
        param_range: np.ndarray,
        initial_solution: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Track solutions of F(x; λ) = 0 as λ varies along *param_range*.

        Uses natural-parameter continuation: solve at each λ using the
        previous solution as the initial guess.

        Returns dict with 'parameters', 'solutions', 'residuals'.
        """
        param_range = np.asarray(param_range, dtype=np.float64)
        n_steps = param_range.shape[0]

        if initial_solution is None:
            initial_solution = np.zeros(2)
        x = np.asarray(initial_solution, dtype=np.float64).ravel()
        dim = x.shape[0]

        solutions = np.empty((n_steps, dim))
        residuals = np.empty(n_steps)

        for k, lam in enumerate(param_range):
            def eqs(xv, _lam=lam):
                return equations(xv, _lam)

            res = optimize.root(eqs, x, method="hybr",
                                tol=self.tol, options={"maxiter": self.max_iter})
            x = res.x.copy()
            solutions[k] = x
            residuals[k] = float(np.linalg.norm(res.fun))

        return {
            "parameters": param_range,
            "solutions": solutions,
            "residuals": residuals,
        }

    # ----- bifurcation detection ---------------------------------------

    def bifurcation_detection(
        self,
        solutions_vs_param: Dict[str, np.ndarray],
        threshold: float = 1e-4,
    ) -> List[Dict[str, Any]]:
        """Detect bifurcation points where the Jacobian becomes singular.

        Parameters
        ----------
        solutions_vs_param : dict
            Output of :meth:`continuation_method`.
        threshold : float
            Eigenvalue magnitude threshold for near-singularity.

        Returns list of dicts with 'parameter', 'solution', 'type'.
        """
        params = solutions_vs_param["parameters"]
        sols = solutions_vs_param["solutions"]
        n = sols.shape[0]

        bifurcations: List[Dict[str, Any]] = []

        for k in range(n):
            lam = params[k]
            x = sols[k]

            # Numerical Jacobian w.r.t. x at this (x, λ)
            dim = x.shape[0]
            eps = 1e-7
            J = np.empty((dim, dim))
            if self.landscape is not None:
                J = self.landscape.hessian_of_free_energy(x)
            else:
                # fall back to finite differences of gradient
                def _grad(xv):
                    return self.landscape.gradient_of_free_energy(xv) if self.landscape else xv
                for j in range(dim):
                    xp = x.copy(); xp[j] += eps
                    xm = x.copy(); xm[j] -= eps
                    J[:, j] = (_grad(xp) - _grad(xm)) / (2.0 * eps)

            eigvals = np.linalg.eigvals(J)
            min_abs = np.min(np.abs(eigvals))

            if min_abs < threshold:
                # Classify: saddle-node vs pitchfork
                n_small = np.sum(np.abs(eigvals) < threshold)
                bif_type = "pitchfork" if n_small >= 2 else "saddle-node"

                # Check for sign change in smallest eigenvalue to refine
                bifurcations.append({
                    "parameter": float(lam),
                    "solution": x.copy(),
                    "type": bif_type,
                    "min_eigenvalue": float(min_abs),
                })

        # Merge nearby detections
        merged: List[Dict[str, Any]] = []
        for b in bifurcations:
            if not merged or abs(b["parameter"] - merged[-1]["parameter"]) > 5 * abs(
                params[1] - params[0] if len(params) > 1 else 1.0
            ):
                merged.append(b)
        return merged


# ---------------------------------------------------------------------------
# 3. FreeEnergyBarrier
# ---------------------------------------------------------------------------

class FreeEnergyBarrier:
    """Compute barriers between free-energy minima using path methods.

    Implements the Nudged Elastic Band (NEB) and simplified string methods
    to find minimum energy paths (MEPs) on the free-energy landscape.

    Parameters
    ----------
    landscape : FreeEnergyLandscape
        The free-energy landscape on which to compute barriers.
    spring_constant : float
        NEB spring constant between images.
    """

    def __init__(
        self,
        landscape: FreeEnergyLandscape,
        spring_constant: float = 10.0,
    ):
        self.landscape = landscape
        self.k_spring = spring_constant

    # ----- public interface --------------------------------------------

    def compute_barrier(
        self,
        minimum_a: np.ndarray,
        minimum_b: np.ndarray,
        method: str = "neb",
        n_images: int = 20,
        max_iter: int = 500,
    ) -> PathResult:
        """Compute the barrier between two minima.

        Parameters
        ----------
        method : {'neb', 'string'}
        """
        if method == "neb":
            return self.minimum_energy_path(minimum_a, minimum_b, n_images, max_iter)
        elif method == "string":
            return self.string_method(minimum_a, minimum_b, n_images, max_iter)
        else:
            raise ValueError(f"Unknown method '{method}'.")

    # ----- Nudged Elastic Band -----------------------------------------

    def minimum_energy_path(
        self,
        start: np.ndarray,
        end: np.ndarray,
        n_images: int = 20,
        max_iter: int = 500,
        tol: float = 1e-6,
        dt: float = 0.01,
    ) -> PathResult:
        """Nudged Elastic Band method for the minimum-energy path.

        Implements the climbing-image NEB after an initial relaxation phase.
        """
        start = np.asarray(start, dtype=np.float64).ravel()
        end = np.asarray(end, dtype=np.float64).ravel()
        dim = start.shape[0]

        # Linear interpolation for initial path
        images = np.array([
            start + t * (end - start) for t in np.linspace(0, 1, n_images)
        ])  # shape (n_images, dim)

        energies = np.array([self.landscape.compute_free_energy(im) for im in images])

        converged = False
        for iteration in range(1, max_iter + 1):
            forces = np.zeros_like(images)

            for i in range(1, n_images - 1):
                # Tangent vector
                tau = images[i + 1] - images[i - 1]
                tau_norm = np.linalg.norm(tau)
                if tau_norm < 1e-15:
                    tau_hat = np.zeros(dim)
                else:
                    tau_hat = tau / tau_norm

                # True gradient (potential force)
                grad = self.landscape.gradient_of_free_energy(images[i])

                # Perpendicular component of true force
                grad_perp = grad - np.dot(grad, tau_hat) * tau_hat

                # Spring force along tangent
                spring_force = self.k_spring * (
                    np.linalg.norm(images[i + 1] - images[i])
                    - np.linalg.norm(images[i] - images[i - 1])
                ) * tau_hat

                forces[i] = -grad_perp + spring_force

            # Climbing image: highest-energy interior image
            interior_energies = energies[1:-1]
            ci_idx = np.argmax(interior_energies) + 1
            grad_ci = self.landscape.gradient_of_free_energy(images[ci_idx])
            tau_ci = images[ci_idx + 1] - images[ci_idx - 1]
            tn = np.linalg.norm(tau_ci)
            if tn > 1e-15:
                tau_ci_hat = tau_ci / tn
            else:
                tau_ci_hat = np.zeros(dim)
            forces[ci_idx] = -grad_ci + 2.0 * np.dot(grad_ci, tau_ci_hat) * tau_ci_hat

            # Velocity Verlet-like update
            max_force = np.max(np.linalg.norm(forces[1:-1], axis=1))
            if max_force < tol:
                converged = True
                break

            # Adaptive step
            step = dt * forces
            max_step = np.max(np.abs(step))
            if max_step > 0.1:
                step *= 0.1 / max_step
            images[1:-1] += step[1:-1]

            # Clip to valid domain
            images = np.clip(images, -0.999, 0.999)

            # Recompute energies
            energies = np.array([self.landscape.compute_free_energy(im) for im in images])

        barrier = float(np.max(energies) - min(energies[0], energies[-1]))
        return PathResult(images, energies, barrier, converged, iteration)

    # ----- String method -----------------------------------------------

    def string_method(
        self,
        start: np.ndarray,
        end: np.ndarray,
        n_images: int = 20,
        max_iter: int = 500,
        tol: float = 1e-6,
        dt: float = 0.005,
    ) -> PathResult:
        """Simplified string method for the minimum-energy path.

        At each step: (1) evolve images by steepest descent on the potential,
        (2) re-parametrise by equal arc-length.
        """
        start = np.asarray(start, dtype=np.float64).ravel()
        end = np.asarray(end, dtype=np.float64).ravel()

        images = np.array([
            start + t * (end - start) for t in np.linspace(0, 1, n_images)
        ])

        converged = False
        for iteration in range(1, max_iter + 1):
            # 1. Steepest descent on free energy (interior images only)
            old_images = images.copy()
            for i in range(1, n_images - 1):
                grad = self.landscape.gradient_of_free_energy(images[i])
                images[i] -= dt * grad

            # Clip
            images = np.clip(images, -0.999, 0.999)

            # 2. Re-parametrise by equal arc-length
            images = self._reparametrise(images)

            # Convergence check
            max_shift = np.max(np.linalg.norm(images - old_images, axis=1))
            if max_shift < tol:
                converged = True
                break

        energies = np.array([self.landscape.compute_free_energy(im) for im in images])
        barrier = float(np.max(energies) - min(energies[0], energies[-1]))
        return PathResult(images, energies, barrier, converged, iteration)

    @staticmethod
    def _reparametrise(images: np.ndarray) -> np.ndarray:
        """Re-distribute images at equal arc-length along the path."""
        n = images.shape[0]
        # Compute cumulative arc-length
        diffs = np.diff(images, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        total = cum[-1]
        if total < 1e-15:
            return images

        # Target arc-lengths
        target = np.linspace(0, total, n)

        # Interpolate each coordinate
        new_images = np.empty_like(images)
        new_images[0] = images[0]
        new_images[-1] = images[-1]
        for d in range(images.shape[1]):
            interp_func = interpolate.interp1d(cum, images[:, d], kind="cubic")
            new_images[:, d] = interp_func(target)

        return new_images

    # ----- barrier height and transition rate --------------------------

    @staticmethod
    def barrier_height(path_energies: np.ndarray) -> float:
        """Extract the barrier height from a sequence of energies along a path."""
        e = np.asarray(path_energies)
        ref = min(e[0], e[-1])
        return float(np.max(e) - ref)

    @staticmethod
    def transition_rate(barrier: float, temperature: float, attempt_freq: float = 1.0) -> float:
        r"""Kramers escape rate  r = ν₀ exp(-ΔF / T).

        Parameters
        ----------
        barrier : float
            Free-energy barrier height ΔF.
        temperature : float
            Temperature T.
        attempt_freq : float
            Pre-exponential attempt frequency ν₀.
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        return attempt_freq * np.exp(-barrier / temperature)


# ---------------------------------------------------------------------------
# 4. PhaseTransitionDetector
# ---------------------------------------------------------------------------

class PhaseTransitionDetector:
    """Detect phase transitions from free-energy data.

    Provides methods for locating first-order transitions (Maxwell
    construction), spinodal lines, and critical points.
    """

    def __init__(self, tol: float = 1e-8):
        self.tol = tol

    # ----- crossing detection ------------------------------------------

    def detect_crossing(
        self,
        free_energies_phase1: np.ndarray,
        free_energies_phase2: np.ndarray,
        param_range: np.ndarray,
    ) -> Optional[float]:
        """Find the parameter value where two free-energy branches cross.

        Uses linear interpolation to find the zero of ΔF = F₁ - F₂.
        """
        f1 = np.asarray(free_energies_phase1, dtype=np.float64)
        f2 = np.asarray(free_energies_phase2, dtype=np.float64)
        p = np.asarray(param_range, dtype=np.float64)

        diff = f1 - f2
        sign_changes = np.where(np.diff(np.sign(diff)))[0]

        if len(sign_changes) == 0:
            return None

        # Refine the first crossing via bisection
        idx = sign_changes[0]
        a, b = p[idx], p[idx + 1]
        fa, fb = diff[idx], diff[idx + 1]

        # Linear interpolation
        crossing = a - fa * (b - a) / (fb - fa)

        # Refine with interpolation of the full curve
        try:
            interp_diff = interpolate.interp1d(p, diff, kind="cubic")
            result = optimize.brentq(interp_diff, a, b, xtol=self.tol)
            return float(result)
        except (ValueError, RuntimeError):
            return float(crossing)

    # ----- Maxwell construction ----------------------------------------

    def maxwell_construction(
        self,
        free_energy_func: Callable[[float], float],
        param_range: np.ndarray,
        resolution: int = 500,
    ) -> Dict[str, Any]:
        """Maxwell equal-area construction for a first-order transition.

        Finds parameter value λ* such that the equal-area rule is satisfied
        for the free-energy curve F(λ).

        Returns dict with 'transition_param', 'coexistence_values',
        'area_balance'.
        """
        p = np.asarray(param_range, dtype=np.float64)
        lam = np.linspace(p[0], p[-1], resolution)
        F = np.array([free_energy_func(l) for l in lam])

        # Pressure / derivative: dF/dλ
        dF = np.gradient(F, lam)
        d2F = np.gradient(dF, lam)

        # Find inflection points (spinodals)
        sign_d2 = np.sign(d2F)
        inflection_idx = np.where(np.diff(sign_d2))[0]

        if len(inflection_idx) < 2:
            return {
                "transition_param": None,
                "coexistence_values": None,
                "area_balance": None,
            }

        sp1_idx, sp2_idx = inflection_idx[0], inflection_idx[-1]
        lam1, lam2 = lam[sp1_idx], lam[sp2_idx]

        # Equal-area: find λ* such that ∫_{lam1}^{λ*} (F - F_tang) dλ
        #   = ∫_{λ*}^{lam2} (F_tang - F) dλ
        # where F_tang is the common-tangent line.

        def area_diff(lam_star):
            # Tangent line connecting F(lam1) to F(lam2) via lam_star
            F1 = free_energy_func(lam1)
            F2 = free_energy_func(lam2)
            slope = (F2 - F1) / (lam2 - lam1)
            tang = lambda l: F1 + slope * (l - lam1)

            # Area above and below
            mask_left = (lam >= lam1) & (lam <= lam_star)
            mask_right = (lam >= lam_star) & (lam <= lam2)

            area_above = np.trapz(F[mask_left] - tang(lam[mask_left]), lam[mask_left])
            area_below = np.trapz(tang(lam[mask_right]) - F[mask_right], lam[mask_right])
            return area_above - area_below

        try:
            lam_star = optimize.brentq(area_diff, lam1, lam2, xtol=self.tol)
        except ValueError:
            lam_star = 0.5 * (lam1 + lam2)

        return {
            "transition_param": float(lam_star),
            "coexistence_values": (float(lam1), float(lam2)),
            "area_balance": float(area_diff(lam_star)),
        }

    # ----- spinodal lines ----------------------------------------------

    def spinodal_lines(
        self,
        free_energy_func: Callable[[np.ndarray], float],
        param_range: np.ndarray,
        order_param_range: Tuple[float, float] = (-0.99, 0.99),
        resolution: int = 200,
    ) -> Dict[str, np.ndarray]:
        """Find spinodal decomposition boundaries where ∂²F/∂q² = 0.

        Returns dict with 'param_values', 'upper_spinodal', 'lower_spinodal'.
        """
        params = np.asarray(param_range, dtype=np.float64)
        q_vals = np.linspace(order_param_range[0], order_param_range[1], resolution)

        upper_spinodals = []
        lower_spinodals = []

        eps = 1e-6
        for lam in params:
            # Build ∂²F/∂q² as a function of q at this λ
            d2F = np.empty(resolution)
            for i, q in enumerate(q_vals):
                qv = np.array([q, lam])
                fp = free_energy_func(np.array([q + eps, lam]))
                fm = free_energy_func(np.array([q - eps, lam]))
                f0 = free_energy_func(qv)
                d2F[i] = (fp - 2 * f0 + fm) / (eps ** 2)

            # Find zeros of d2F
            sign_changes = np.where(np.diff(np.sign(d2F)))[0]
            zeros = []
            for idx in sign_changes:
                # Linear interpolation
                q_zero = q_vals[idx] - d2F[idx] * (q_vals[idx + 1] - q_vals[idx]) / (
                    d2F[idx + 1] - d2F[idx]
                )
                zeros.append(q_zero)

            if len(zeros) >= 2:
                lower_spinodals.append(min(zeros))
                upper_spinodals.append(max(zeros))
            elif len(zeros) == 1:
                lower_spinodals.append(zeros[0])
                upper_spinodals.append(zeros[0])
            else:
                lower_spinodals.append(np.nan)
                upper_spinodals.append(np.nan)

        return {
            "param_values": params,
            "upper_spinodal": np.array(upper_spinodals),
            "lower_spinodal": np.array(lower_spinodals),
        }

    # ----- critical point ----------------------------------------------

    def critical_point(
        self,
        free_energy_func: Callable[[np.ndarray], float],
        param_ranges: Tuple[Tuple[float, float], Tuple[float, float]],
        resolution: int = 80,
    ) -> Optional[Dict[str, float]]:
        """Locate the critical point where ∂²F/∂q² = 0 and ∂³F/∂q³ = 0.

        Searches a 2-D grid of (q, λ) for the simultaneous vanishing of
        the second and third derivatives.
        """
        q_lo, q_hi = param_ranges[0]
        lam_lo, lam_hi = param_ranges[1]

        q_grid = np.linspace(q_lo, q_hi, resolution)
        lam_grid = np.linspace(lam_lo, lam_hi, resolution)

        eps = 1e-5

        def derivs(q, lam):
            """Return (d2F, d3F) at (q, lam)."""
            pts = [free_energy_func(np.array([q + k * eps, lam])) for k in range(-2, 3)]
            d2 = (pts[3] - 2 * pts[2] + pts[1]) / eps ** 2
            d3 = (pts[4] - 2 * pts[3] + 2 * pts[1] - pts[0]) / (2 * eps ** 3)
            return d2, d3

        # Coarse grid search for region where both are small
        best_obj = np.inf
        best_q, best_lam = 0.0, 0.0

        for q in q_grid:
            for lam in lam_grid:
                d2, d3 = derivs(q, lam)
                obj = d2 ** 2 + d3 ** 2
                if obj < best_obj:
                    best_obj = obj
                    best_q, best_lam = q, lam

        # Refine with optimisation
        def objective(x):
            d2, d3 = derivs(x[0], x[1])
            return d2 ** 2 + d3 ** 2

        res = optimize.minimize(
            objective,
            [best_q, best_lam],
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-14, "maxiter": 5000},
        )

        if res.fun > 1e-4:
            return None

        return {
            "q_critical": float(res.x[0]),
            "param_critical": float(res.x[1]),
            "residual": float(res.fun),
        }


# ---------------------------------------------------------------------------
# 5. TransitionClassifier
# ---------------------------------------------------------------------------

class TransitionClassifier:
    """Classify phase transitions as first-order or second-order.

    Uses order-parameter discontinuities, latent-heat calculations,
    critical-exponent extraction, and Landau-expansion fitting.
    """

    def __init__(self, tol: float = 1e-6):
        self.tol = tol

    # ----- classify from order parameter -------------------------------

    def classify(
        self,
        order_param_vs_control: Tuple[np.ndarray, np.ndarray],
    ) -> TransitionInfo:
        """Classify the transition by analysing the order parameter q(λ).

        A discontinuous jump ⇒ first-order; continuous but non-analytic ⇒
        second-order.

        Parameters
        ----------
        order_param_vs_control : (control_values, order_param_values)
        """
        control, q = (
            np.asarray(order_param_vs_control[0], dtype=np.float64),
            np.asarray(order_param_vs_control[1], dtype=np.float64),
        )

        # Detect jumps
        dq = np.abs(np.diff(q))
        dc = np.abs(np.diff(control))
        # Normalise derivative
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = dq / dc

        max_slope_idx = np.argmax(slope)
        max_slope = slope[max_slope_idx]

        # Heuristic: if slope > 10× median → likely discontinuity
        median_slope = np.median(slope[np.isfinite(slope)])
        jump = dq[max_slope_idx]

        if max_slope > 10 * median_slope and jump > 0.01:
            order = 1
        else:
            order = 2

        loc = float(0.5 * (control[max_slope_idx] + control[max_slope_idx + 1]))

        return TransitionInfo(
            location=loc,
            order=order,
            latent_heat=float(jump) if order == 1 else None,
        )

    # ----- latent heat -------------------------------------------------

    def latent_heat(
        self,
        free_energies: np.ndarray,
        temperature_range: np.ndarray,
    ) -> Optional[float]:
        """Compute latent heat L = T_c ΔS for a first-order transition.

        Uses the entropy jump at the transition temperature.
        """
        F = np.asarray(free_energies, dtype=np.float64)
        T = np.asarray(temperature_range, dtype=np.float64)

        # Entropy: S = -dF/dT
        S = -np.gradient(F, T)

        # Find largest jump in entropy
        dS = np.abs(np.diff(S))
        idx = np.argmax(dS)
        delta_S = dS[idx]
        T_c = 0.5 * (T[idx] + T[idx + 1])

        if delta_S < self.tol:
            return None

        return float(T_c * delta_S)

    # ----- critical exponents -----------------------------------------

    def critical_exponents(
        self,
        order_param_vs_control: Tuple[np.ndarray, np.ndarray],
        critical_point: float,
    ) -> Dict[str, float]:
        r"""Extract critical exponents β, γ, δ, α near a second-order transition.

        β:  q ~ |λ - λ_c|^β          (order parameter)
        γ:  χ ~ |λ - λ_c|^{-γ}       (susceptibility ∝ dq/dh)
        α:  C ~ |λ - λ_c|^{-α}       (specific heat from d²F/dT²)

        Uses log-log fitting in the scaling region.
        """
        control, q = (
            np.asarray(order_param_vs_control[0], dtype=np.float64),
            np.asarray(order_param_vs_control[1], dtype=np.float64),
        )
        lam_c = critical_point

        # Select data near but not at the critical point
        eps = control - lam_c
        mask = (np.abs(eps) > 1e-8) & (np.abs(eps) < 0.3 * (control[-1] - control[0]))
        # Use one side (above Tc)
        above = mask & (eps > 0)
        below = mask & (eps < 0)

        exponents: Dict[str, float] = {}

        # β from q ~ |ε|^β (above critical point)
        if np.sum(above) > 5:
            log_eps = np.log(np.abs(eps[above]))
            log_q = np.log(np.abs(q[above]) + 1e-30)
            try:
                coeffs = np.polyfit(log_eps, log_q, 1)
                exponents["beta"] = float(coeffs[0])
            except (np.linalg.LinAlgError, ValueError):
                exponents["beta"] = np.nan
        else:
            exponents["beta"] = np.nan

        # γ from susceptibility χ = dq/dλ ~ |ε|^{-γ}
        dq_dlam = np.gradient(q, control)
        chi = np.abs(dq_dlam)

        if np.sum(above) > 5:
            log_chi = np.log(chi[above] + 1e-30)
            log_eps_a = np.log(np.abs(eps[above]))
            try:
                coeffs = np.polyfit(log_eps_a, log_chi, 1)
                exponents["gamma"] = float(-coeffs[0])
            except (np.linalg.LinAlgError, ValueError):
                exponents["gamma"] = np.nan
        else:
            exponents["gamma"] = np.nan

        # α from specific heat C = -T d²F/dT² ~ |ε|^{-α}
        # Approximate via second derivative of q as proxy
        d2q = np.gradient(dq_dlam, control)
        C_proxy = np.abs(d2q)

        if np.sum(above) > 5:
            log_C = np.log(C_proxy[above] + 1e-30)
            try:
                coeffs = np.polyfit(np.log(np.abs(eps[above])), log_C, 1)
                exponents["alpha"] = float(-coeffs[0])
            except (np.linalg.LinAlgError, ValueError):
                exponents["alpha"] = np.nan
        else:
            exponents["alpha"] = np.nan

        # δ from q ~ h^{1/δ} at T=Tc — requires field data, estimate from scaling relation
        # Widom relation: δ = 1 + γ/β
        beta = exponents.get("beta", np.nan)
        gamma = exponents.get("gamma", np.nan)
        if np.isfinite(beta) and np.isfinite(gamma) and abs(beta) > 1e-10:
            exponents["delta"] = 1.0 + gamma / beta
        else:
            exponents["delta"] = np.nan

        return exponents

    # ----- Landau expansion --------------------------------------------

    def landau_expansion(
        self,
        free_energy_data: Tuple[np.ndarray, np.ndarray],
        order_param_range: Optional[Tuple[float, float]] = None,
        max_order: int = 6,
    ) -> Dict[str, float]:
        r"""Fit Landau expansion F(q) = a₀ + a₂ q² + a₄ q⁴ + a₆ q⁶ + …

        Only even powers are used (Z₂ symmetry assumed).  The sign pattern
        of the coefficients distinguishes first- and second-order transitions:
          * a₂ > 0, a₄ > 0  →  disordered (no transition)
          * a₂ < 0, a₄ > 0  →  second-order transition
          * a₂ > 0, a₄ < 0, a₆ > 0  →  first-order transition

        Parameters
        ----------
        free_energy_data : (q_values, F_values)
        order_param_range : tuple, optional
            Restrict fit to this q range.
        max_order : int
            Highest even power to include (default 6).

        Returns dict with keys 'a0', 'a2', 'a4', … and 'transition_type'.
        """
        q_vals, F_vals = (
            np.asarray(free_energy_data[0], dtype=np.float64),
            np.asarray(free_energy_data[1], dtype=np.float64),
        )

        if order_param_range is not None:
            mask = (q_vals >= order_param_range[0]) & (q_vals <= order_param_range[1])
            q_vals = q_vals[mask]
            F_vals = F_vals[mask]

        # Build design matrix with even powers: q^0, q^2, q^4, …
        powers = list(range(0, max_order + 1, 2))
        A = np.column_stack([q_vals ** p for p in powers])

        # Least-squares fit
        coeffs, residuals, rank, sv = np.linalg.lstsq(A, F_vals, rcond=None)

        result: Dict[str, float] = {}
        for p, c in zip(powers, coeffs):
            result[f"a{p}"] = float(c)

        # Classify
        a2 = result.get("a2", 0.0)
        a4 = result.get("a4", 0.0)
        a6 = result.get("a6", 0.0)

        if a2 < 0 and a4 > 0:
            result["transition_type"] = "second_order"
        elif a4 < 0 and a6 > 0:
            result["transition_type"] = "first_order"
        elif a2 > 0 and a4 > 0:
            result["transition_type"] = "none"
        else:
            result["transition_type"] = "undetermined"

        # If second-order, estimate Tc from a2(T) = a2_0 (T - Tc)
        # a2 changes sign at Tc ⇒ Tc ≈ -a0/a2 (rough proxy)
        if result["transition_type"] == "second_order" and abs(a2) > 1e-15:
            q_min_sq = -a2 / (2.0 * a4) if a4 > 0 else 0.0
            result["q_min_squared"] = float(max(q_min_sq, 0.0))

        # Fit residual
        F_fit = A @ coeffs
        result["fit_residual"] = float(np.sqrt(np.mean((F_vals - F_fit) ** 2)))

        return result
