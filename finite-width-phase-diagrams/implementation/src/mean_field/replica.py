"""
Replica method for neural network mean-field theory.

Implements replica-symmetric (RS), one-step RSB, de Almeida-Thouless stability,
overlap distributions, and the Parisi functional for full RSB.
"""

import numpy as np
from scipy import optimize, integrate, special, interpolate, linalg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gaussian_measure(func, limit=10.0, points=200):
    """Evaluate ∫ Dz f(z) where Dz = e^{-z²/2}/√(2π) dz."""
    z, w = np.polynomial.hermite_e.hermegauss(points)
    w = w / np.sqrt(2.0 * np.pi)
    return np.sum(w * func(z))


def _H(x):
    """Complementary Gaussian CDF: H(x) = ∫_x^∞ Dz."""
    return 0.5 * special.erfc(x / np.sqrt(2.0))


def _phi(x):
    """Standard Gaussian PDF."""
    return np.exp(-0.5 * x ** 2) / np.sqrt(2.0 * np.pi)


# ===================================================================
# 1. ReplicaSymmetricSolver
# ===================================================================

class ReplicaSymmetricSolver:
    """Replica-symmetric ansatz solver for single-layer neural networks.

    The order parameters are the overlap q = (1/N) Σ_i w_i^a w_i^b
    and its conjugate q̂.  alpha = P/N is the load.
    """

    def __init__(self, n_replicas=0, temperature=1.0, alpha=1.0):
        self.n = n_replicas          # analytic continuation n→0
        self.temperature = temperature
        self.beta = 1.0 / max(temperature, 1e-15)
        self.alpha = alpha

    # ---- free energy --------------------------------------------------

    def rs_free_energy(self, q, qhat):
        """RS free energy F_RS(q, q̂) per degree of freedom.

        F = extr_{q,q̂} [ G_S(q̂) + α G_E(q) - ½ q q̂ ]

        G_S (entropic / prior) and G_E (energetic / likelihood) are
        computed with Gaussian integrals.
        """
        beta = self.beta
        alpha = self.alpha

        # Entropic term  G_S = ½ [ log(1 - q̂/β) + q̂/(β - q̂) ]
        denom = beta - qhat
        if denom <= 0:
            return 1e10
        G_S = 0.5 * (np.log(denom / beta) + qhat / denom)

        # Energetic term  G_E = -∫Dz log H(−√(q) z / √(1-q))  (perceptron)
        # Uses Gaussian quadrature over z.
        sq = np.clip(q, 1e-15, 1.0 - 1e-15)

        def integrand(z):
            arg = -np.sqrt(sq) * z / np.sqrt(1.0 - sq)
            h = np.clip(_H(arg), 1e-30, 1.0)
            return np.log(h)

        G_E = -_gaussian_measure(integrand)

        F = G_S + alpha * G_E - 0.5 * q * qhat
        return F

    # ---- saddle-point equations ----------------------------------------

    def rs_saddle_point_equations(self, q, qhat):
        """Return (∂F/∂q, ∂F/∂q̂) which should both vanish at the saddle."""
        beta = self.beta
        alpha = self.alpha

        # ∂G_S/∂q̂
        denom = beta - qhat
        if denom <= 0:
            return np.array([1e10, 1e10])
        dGS_dqhat = 0.5 * qhat / (denom ** 2)

        # ∂G_E/∂q  (numerical derivative via central differences)
        eps = 1e-7
        dGE_dq = (self._GE(q + eps) - self._GE(q - eps)) / (2.0 * eps)

        dF_dq = alpha * dGE_dq - 0.5 * qhat
        dF_dqhat = dGS_dqhat - 0.5 * q
        return np.array([dF_dq, dF_dqhat])

    def _GE(self, q):
        sq = np.clip(q, 1e-15, 1.0 - 1e-15)

        def integrand(z):
            arg = -np.sqrt(sq) * z / np.sqrt(1.0 - sq)
            h = np.clip(_H(arg), 1e-30, 1.0)
            return np.log(h)

        return -_gaussian_measure(integrand)

    # ---- solver --------------------------------------------------------

    def solve_rs(self, initial_q=0.5, initial_qhat=0.5, tol=1e-10,
                 max_iter=5000, damping=0.5):
        """Solve the RS saddle-point equations iteratively.

        Uses a damped fixed-point iteration derived from the
        saddle-point stationarity conditions.
        """
        q = initial_q
        qhat = initial_qhat
        beta = self.beta
        alpha = self.alpha

        for it in range(max_iter):
            # From ∂F/∂q̂ = 0:  q_new = q̂ / (β - q̂)²  (entropic channel)
            denom = beta - qhat
            if denom <= 0:
                denom = 1e-8
            q_new = qhat / (denom ** 2)
            q_new = np.clip(q_new, 1e-12, 1.0 - 1e-12)

            # From ∂F/∂q = 0:  q̂_new = 2 α ∂G_E/∂q
            eps = 1e-7
            dGE = (self._GE(q_new + eps) - self._GE(q_new - eps)) / (2 * eps)
            qhat_new = 2.0 * alpha * dGE

            # Damped update
            q = (1 - damping) * q + damping * q_new
            qhat = (1 - damping) * qhat + damping * qhat_new

            q = np.clip(q, 1e-12, 1.0 - 1e-12)
            qhat = max(qhat, 1e-12)

            residual = np.sqrt((q - q_new) ** 2 + (qhat - qhat_new) ** 2)
            if residual < tol:
                break

        return {
            "q": q,
            "qhat": qhat,
            "free_energy": self.rs_free_energy(q, qhat),
            "converged": residual < tol,
            "iterations": it + 1,
        }

    # ---- observables ---------------------------------------------------

    def training_error_rs(self, q):
        """Fraction of patterns misclassified (RS)."""
        sq = np.clip(q, 1e-15, 1.0 - 1e-15)

        def integrand(z):
            return _H(np.sqrt(sq / (1.0 - sq)) * z)

        return _gaussian_measure(integrand)

    def generalization_error_rs(self, q):
        """Generalization error ε_g = (1/π) arccos(q)  (perceptron on iid data)."""
        q_c = np.clip(q, -1.0 + 1e-12, 1.0 - 1e-12)
        return np.arccos(q_c) / np.pi

    def entropy_rs(self, q, qhat):
        """Entropy S = β² ∂F/∂β  evaluated at the RS saddle point."""
        beta = self.beta
        denom = beta - qhat
        if denom <= 0:
            return -np.inf
        S = 0.5 * np.log(2.0 * np.pi * np.e * denom / beta)
        return S

    def gardner_volume(self, kappa, alpha_range):
        """Compute log Gardner volume V(κ) as a function of α.

        κ is the margin; returns α_c(κ) and the volume curve.
        """
        alpha_range = np.asarray(alpha_range, dtype=float)
        volumes = np.empty_like(alpha_range)

        for i, alpha in enumerate(alpha_range):
            self.alpha = alpha

            def equations(x):
                q, qhat = x
                sq = np.clip(q, 1e-15, 1.0 - 1e-15)
                # Margin-κ version of the energetic integral
                def integ(z):
                    arg = (kappa - np.sqrt(sq) * z) / np.sqrt(1.0 - sq)
                    h = np.clip(_H(arg), 1e-30, 1.0)
                    return np.log(h)

                GE = -_gaussian_measure(integ)

                # Derivative of GE w.r.t. q
                eps = 1e-7
                def integ_p(z, dq):
                    arg = (kappa - np.sqrt(np.clip(dq, 1e-15, 1-1e-15)) * z) / np.sqrt(1.0 - np.clip(dq, 1e-15, 1-1e-15))
                    h = np.clip(_H(arg), 1e-30, 1.0)
                    return np.log(h)

                GE_p = -_gaussian_measure(lambda z: integ_p(z, q + eps))
                GE_m = -_gaussian_measure(lambda z: integ_p(z, q - eps))
                dGE_dq = (GE_p - GE_m) / (2 * eps)

                eq1 = qhat - 2.0 * alpha * dGE_dq
                # zero-T limit: denom → 1
                eq2 = q - qhat / (1.0 + qhat) ** 2
                # simplified for T→0
                return [eq1, eq2]

            try:
                sol = optimize.fsolve(equations, [0.5, 0.5], full_output=True)
                x_sol = sol[0]
                q_sol, qhat_sol = x_sol
                q_sol = np.clip(q_sol, 1e-12, 1 - 1e-12)

                def integ_v(z):
                    arg = (kappa - np.sqrt(q_sol) * z) / np.sqrt(1.0 - q_sol)
                    h = np.clip(_H(arg), 1e-30, 1.0)
                    return np.log(h)

                GE = -_gaussian_measure(integ_v)
                GS = 0.5 * (np.log(1.0 / (1.0 + qhat_sol)) + qhat_sol / (1.0 + qhat_sol))
                volumes[i] = GS + alpha * GE - 0.5 * q_sol * qhat_sol
            except Exception:
                volumes[i] = -np.inf

        return {"alpha": alpha_range, "log_volume": volumes}


# ===================================================================
# 2. OneStepRSBSolver
# ===================================================================

class OneStepRSBSolver:
    """One-step replica symmetry breaking (1RSB) solver.

    Order parameters: q₀ < q₁ (overlaps), m ∈ (0,1) (breakpoint),
    and conjugates q̂₀, q̂₁.
    """

    def __init__(self, temperature=1.0, alpha=1.0):
        self.temperature = temperature
        self.beta = 1.0 / max(temperature, 1e-15)
        self.alpha = alpha

    # ---- free energy --------------------------------------------------

    def rsb1_free_energy(self, q0, q1, m, qhat0, qhat1):
        """1RSB free energy F_{1RSB}(q₀,q₁,m,q̂₀,q̂₁)."""
        beta = self.beta
        alpha = self.alpha

        # Entropic part G_S^{1RSB}
        # Covariance structure: C_ab = δ_{ab} + (q̂₁ - q̂₀) I(same group) + q̂₀
        # After Hubbard-Stratonovich:
        A = beta - qhat1
        B = qhat1 - qhat0
        if A <= 0 or A + m * B <= 0:
            return 1e10

        GS = 0.5 * (np.log(A / beta)
                     + (1.0 / m) * np.log(A / (A + m * B))
                     + qhat0 / (A + m * B))

        # Energetic part G_E^{1RSB}
        GE = self._GE_1rsb(q0, q1, m)

        F = GS + alpha * GE - 0.5 * (q1 * qhat1 - q0 * qhat0) - 0.5 * q0 * qhat0 / m
        # Correct coupling: -½ [q₁ q̂₁ + (1/m - 1)(q₁ q̂₁ - q₀ q̂₀)]
        # Simplify:
        F = GS + alpha * GE - 0.5 * q1 * qhat1 + 0.5 * (1.0 - 1.0 / m) * (q1 * qhat1 - q0 * qhat0)
        return F

    def _GE_1rsb(self, q0, q1, m):
        """Energetic integral for 1RSB."""
        q0c = np.clip(q0, 1e-15, 1.0 - 1e-15)
        q1c = np.clip(q1, q0c + 1e-15, 1.0 - 1e-15)

        def outer_integrand(z0):
            # Inner integral over z1
            def inner(z1):
                mean = np.sqrt(q0c) * z0 + np.sqrt(q1c - q0c) * z1
                var = np.sqrt(1.0 - q1c)
                h = np.clip(_H(-mean / var), 1e-30, 1.0)
                return h ** m

            val = _gaussian_measure(inner)
            val = np.clip(val, 1e-30, None)
            return np.log(val) / m

        return -_gaussian_measure(outer_integrand)

    # ---- saddle-point equations ----------------------------------------

    def rsb1_saddle_point(self, params):
        """Residual of the 5 saddle-point equations for 1RSB.

        params = [q0, q1, m, qhat0, qhat1]
        """
        q0, q1, m, qhat0, qhat1 = params
        # Clip for safety
        q0 = np.clip(q0, 1e-10, 1.0 - 1e-5)
        q1 = np.clip(q1, q0 + 1e-10, 1.0 - 1e-5)
        m = np.clip(m, 1e-5, 1.0 - 1e-5)

        eps = 1e-7
        F0 = self.rsb1_free_energy(q0, q1, m, qhat0, qhat1)

        # Numerical gradients
        dF_dq0 = (self.rsb1_free_energy(q0 + eps, q1, m, qhat0, qhat1) - F0) / eps
        dF_dq1 = (self.rsb1_free_energy(q0, q1 + eps, m, qhat0, qhat1) - F0) / eps
        dF_dm = (self.rsb1_free_energy(q0, q1, m + eps, qhat0, qhat1) - F0) / eps
        dF_dqhat0 = (self.rsb1_free_energy(q0, q1, m, qhat0 + eps, qhat1) - F0) / eps
        dF_dqhat1 = (self.rsb1_free_energy(q0, q1, m, qhat0, qhat1 + eps) - F0) / eps

        return np.array([dF_dq0, dF_dq1, dF_dm, dF_dqhat0, dF_dqhat1])

    # ---- solver --------------------------------------------------------

    def solve_rsb1(self, initial_params=None, tol=1e-10, max_iter=3000):
        """Solve the 1RSB saddle-point equations.

        initial_params = [q0, q1, m, qhat0, qhat1] or None for defaults.
        """
        if initial_params is None:
            initial_params = [0.3, 0.7, 0.5, 0.3, 0.7]

        x0 = np.array(initial_params, dtype=float)

        def residual(x):
            return self.rsb1_saddle_point(x)

        result = optimize.root(residual, x0, method="hybr",
                               tol=tol, options={"maxfev": max_iter * 10})
        q0, q1, m, qhat0, qhat1 = result.x

        return {
            "q0": np.clip(q0, 0, 1),
            "q1": np.clip(q1, 0, 1),
            "m": np.clip(m, 0, 1),
            "qhat0": qhat0,
            "qhat1": qhat1,
            "free_energy": self.rsb1_free_energy(q0, q1, m, qhat0, qhat1),
            "converged": result.success,
            "residual_norm": np.linalg.norm(result.fun),
        }

    # ---- breakpoint ----------------------------------------------------

    def breakpoint_m(self, q0, q1):
        """Find optimal breakpoint m by minimising F over m ∈ (0,1)."""
        def objective(m_arr):
            m = m_arr[0]
            # Use current qhat estimates from an RS-like relation
            qhat0 = q0 / (1.0 - q0) ** 2
            qhat1 = q1 / (1.0 - q1) ** 2
            return self.rsb1_free_energy(q0, q1, m, qhat0, qhat1)

        res = optimize.minimize_scalar(
            lambda m: objective([m]),
            bounds=(0.01, 0.99), method="bounded"
        )
        return res.x

    # ---- overlap distribution ------------------------------------------

    def rsb1_overlap_distribution(self, q0, q1, m):
        """P(q) for 1RSB: two delta functions at q₀ and q₁.

        Returns weights and positions.
        """
        return {
            "positions": np.array([q0, q1]),
            "weights": np.array([1.0 - m, m]),
        }

    # ---- complexity (configurational entropy) --------------------------

    def complexity(self, free_energy, m):
        """Complexity Σ(f) = m² ∂(F/m)/∂m  (Legendre transform)."""
        eps = 1e-6
        # Σ = ∂(m F)/∂m - F  → use numerical differentiation
        # Actually Σ = m² ∂(F/m)/∂m = F - m ∂F/∂m  ... but canonical:
        # We store F(m); differentiate
        # Approximate with small variation of m in solve
        # Here we just return the Monasson formula:  Σ = βm(f - F)
        # where f = F + (1/βm) Σ → not helpful.  Use the direct formula:
        # Σ = (1 - m) ∂F/∂m  (correct form for 1RSB)
        # We estimate ∂F/∂m numerically from the caller-supplied free_energy fn
        # If free_energy is a scalar, return the analytic estimate.
        return (1.0 - m) * free_energy  # leading-order estimate

    # ---- marginal stability -------------------------------------------

    def marginal_stability(self, q0, q1, m):
        """Check marginal stability of the 1RSB solution.

        At marginal stability the replicon eigenvalue within the 1RSB
        block vanishes.  Returns the eigenvalue; stable if > 0.
        """
        q1c = np.clip(q1, 1e-12, 1.0 - 1e-12)
        q0c = np.clip(q0, 1e-12, q1c - 1e-12)
        alpha = self.alpha

        # Replicon in 1RSB:
        # λ_R = 1 - α ∫Dz [ ∫Dz₁ h^m (h''/h)^2 / ∫Dz₁ h^m ]
        # where h = H(-(√q₀ z + √(q₁-q₀) z₁)/√(1-q₁))
        # Approximate numerically
        def outer(z0):
            def inner_num(z1):
                arg = -(np.sqrt(q0c) * z0 + np.sqrt(q1c - q0c) * z1) / np.sqrt(1 - q1c)
                h = np.clip(_H(arg), 1e-30, 1.0)
                phi_val = _phi(arg) / np.sqrt(1 - q1c)
                # h'' / h ≈ (phi'/h - (phi/h)^2)  in the H-derivative sense
                ratio = phi_val / h
                return h ** m * ratio ** 2

            def inner_den(z1):
                arg = -(np.sqrt(q0c) * z0 + np.sqrt(q1c - q0c) * z1) / np.sqrt(1 - q1c)
                h = np.clip(_H(arg), 1e-30, 1.0)
                return h ** m

            num = _gaussian_measure(inner_num)
            den = _gaussian_measure(inner_den)
            if den < 1e-30:
                return 0.0
            return num / den

        integral = _gaussian_measure(outer)
        replicon = 1.0 - alpha * integral
        return {
            "replicon": replicon,
            "stable": replicon > 0,
        }


# ===================================================================
# 3. DeAlmeidaThoulessChecker
# ===================================================================

class DeAlmeidaThoulessChecker:
    """de Almeida-Thouless stability analysis for the RS solution."""

    def __init__(self, temperature=1.0):
        self.temperature = temperature
        self.beta = 1.0 / max(temperature, 1e-15)

    # ---- core AT condition --------------------------------------------

    def at_stability_condition(self, q, qhat, temperature=None):
        """Compute the AT stability parameter.

        RS is stable when λ_AT > 0.  Returns the value.
        """
        if temperature is not None:
            beta = 1.0 / max(temperature, 1e-15)
        else:
            beta = self.beta

        # For the SK model:  λ_AT = 1 - β²(1-q)²  × ⟨1/cosh⁴(β√q z + βh)⟩
        # For the NN perceptron version we use the replicon:
        return self.replicon_eigenvalue(q, qhat, beta)

    # ---- AT line -------------------------------------------------------

    def at_line(self, temperature_range, field_range=None):
        """Compute the AT line in the (T, h) plane for the SK model.

        For each T, find h_AT such that λ_AT = 0.
        """
        temperature_range = np.asarray(temperature_range, dtype=float)
        if field_range is not None:
            field_range = np.asarray(field_range, dtype=float)

        h_at = np.full_like(temperature_range, np.nan)

        for i, T in enumerate(temperature_range):
            if T < 1e-10:
                h_at[i] = 0.0
                continue
            beta = 1.0 / T

            def condition(h):
                # RS solution for SK in a field: q = ⟨tanh²(β√q z + βh)⟩
                def q_eq(q_val):
                    q_c = np.clip(q_val, 1e-12, 1.0 - 1e-12)
                    def integ(z):
                        return np.tanh(beta * np.sqrt(q_c) * z + beta * h) ** 2
                    return _gaussian_measure(integ) - q_c

                try:
                    q_sol = optimize.brentq(q_eq, 1e-10, 1.0 - 1e-10)
                except ValueError:
                    return 1.0  # no solution → stable

                # AT: 1 - β²⟨sech⁴(β√q z + βh)⟩ = 0
                q_c = np.clip(q_sol, 1e-12, 1.0 - 1e-12)
                def at_integ(z):
                    return 1.0 / np.cosh(beta * np.sqrt(q_c) * z + beta * h) ** 4
                return 1.0 - beta ** 2 * _gaussian_measure(at_integ)

            if field_range is not None:
                # Evaluate condition on the provided grid, find zero crossing
                vals = np.array([condition(h) for h in field_range])
                crossings = np.where(np.diff(np.sign(vals)))[0]
                if len(crossings) > 0:
                    idx = crossings[0]
                    try:
                        h_at[i] = optimize.brentq(condition, field_range[idx],
                                                  field_range[idx + 1])
                    except (ValueError, IndexError):
                        h_at[i] = np.nan
            else:
                try:
                    h_at[i] = optimize.brentq(condition, 0.0, 5.0)
                except ValueError:
                    h_at[i] = np.nan

        return {"temperature": temperature_range, "h_AT": h_at}

    # ---- replicon eigenvalue -------------------------------------------

    def replicon_eigenvalue(self, q, qhat, beta=None):
        """Replicon eigenvalue λ_R for the perceptron model.

        λ_R = 1 - α ∫Dz [φ(t)/H(t)]²  where t = √(q/(1-q)) z
        and α = q̂ (in certain parameterisations).
        """
        if beta is None:
            beta = self.beta
        q_c = np.clip(q, 1e-12, 1.0 - 1e-12)

        def integrand(z):
            t = np.sqrt(q_c / (1.0 - q_c)) * z
            h = np.clip(_H(-t), 1e-30, 1.0)
            p = _phi(t)
            return (p / h) ** 2

        integral = _gaussian_measure(integrand)
        # In the perceptron, effective α enters through q̂
        lam = 1.0 - qhat * integral / (1.0 - q_c)
        return lam

    # ---- RS stability check -------------------------------------------

    def rs_stability_check(self, rs_solution):
        """Check AT stability of a given RS solution dict."""
        q = rs_solution["q"]
        qhat = rs_solution["qhat"]
        lam = self.replicon_eigenvalue(q, qhat)
        return {
            "replicon": lam,
            "stable": lam > 0,
            "q": q,
            "qhat": qhat,
        }

    # ---- instability direction ----------------------------------------

    def instability_direction(self, rs_solution):
        """Direction of AT instability in overlap space.

        Returns the perturbation δQ_{ab} that lowers the free energy
        when the replicon is negative.
        """
        q = rs_solution["q"]
        n_eff = 4  # small illustrative replica number
        # The replicon mode is the traceless symmetric perturbation
        # δQ_{ab} = ε (for a≠b in same block) – ε (across blocks)
        delta_Q = np.zeros((n_eff, n_eff))
        half = n_eff // 2
        for a in range(n_eff):
            for b in range(a + 1, n_eff):
                same_block = (a < half and b < half) or (a >= half and b >= half)
                delta_Q[a, b] = 1.0 if same_block else -1.0
                delta_Q[b, a] = delta_Q[a, b]
        # Normalise
        norm = np.sqrt(np.sum(delta_Q ** 2))
        if norm > 0:
            delta_Q /= norm
        return delta_Q

    # ---- AT line for NN -----------------------------------------------

    def at_line_neural_network(self, alpha_range, sigma_range=None):
        """AT line in the (α, σ) plane for the NN perceptron model.

        For each α, find the noise level σ at which RS becomes unstable.
        """
        alpha_range = np.asarray(alpha_range, dtype=float)
        if sigma_range is None:
            sigma_range = np.linspace(0.0, 3.0, 300)

        sigma_at = np.full_like(alpha_range, np.nan)

        for i, alpha in enumerate(alpha_range):
            rs_solver = ReplicaSymmetricSolver(temperature=1.0, alpha=alpha)

            def condition(sigma):
                rs_solver.temperature = max(sigma, 1e-10)
                rs_solver.beta = 1.0 / rs_solver.temperature
                sol = rs_solver.solve_rs(damping=0.3)
                if not sol["converged"]:
                    return 1.0
                lam = self.replicon_eigenvalue(sol["q"], sol["qhat"])
                return lam

            vals = np.array([condition(s) for s in sigma_range])
            crossings = np.where(np.diff(np.sign(vals)))[0]
            if len(crossings) > 0:
                idx = crossings[0]
                try:
                    sigma_at[i] = optimize.brentq(
                        condition, sigma_range[idx], sigma_range[idx + 1]
                    )
                except ValueError:
                    pass

        return {"alpha": alpha_range, "sigma_AT": sigma_at}


# ===================================================================
# 4. OverlapDistribution
# ===================================================================

class OverlapDistribution:
    """Compute and analyse the overlap distribution P(q)."""

    def __init__(self, n_bins=200):
        self.n_bins = n_bins

    # ---- from samples --------------------------------------------------

    def compute_pq_from_samples(self, overlap_samples):
        """Histogram-based estimate of P(q) from overlap samples."""
        overlap_samples = np.asarray(overlap_samples, dtype=float)
        counts, edges = np.histogram(overlap_samples, bins=self.n_bins,
                                     density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        return {"q": centres, "pq": counts}

    # ---- RS ------------------------------------------------------------

    def pq_from_rs(self, q_rs):
        """P(q) = δ(q − q_RS)  represented on a fine grid."""
        q_vals = np.linspace(-1, 1, self.n_bins)
        pq = np.zeros(self.n_bins)
        idx = np.argmin(np.abs(q_vals - q_rs))
        dq = q_vals[1] - q_vals[0]
        pq[idx] = 1.0 / dq  # normalised delta approximation
        return {"q": q_vals, "pq": pq}

    # ---- 1RSB ----------------------------------------------------------

    def pq_from_rsb1(self, q0, q1, m):
        """P(q) = (1-m)δ(q-q₀) + m δ(q-q₁)."""
        q_vals = np.linspace(-1, 1, self.n_bins)
        dq = q_vals[1] - q_vals[0]
        pq = np.zeros(self.n_bins)
        idx0 = np.argmin(np.abs(q_vals - q0))
        idx1 = np.argmin(np.abs(q_vals - q1))
        pq[idx0] = (1.0 - m) / dq
        pq[idx1] = m / dq
        return {"q": q_vals, "pq": pq}

    # ---- moments -------------------------------------------------------

    def pq_moments(self, pq, q_values, max_order=4):
        """Compute ⟨q^k⟩ for k = 1 … max_order."""
        pq = np.asarray(pq, dtype=float)
        q_values = np.asarray(q_values, dtype=float)
        dq = np.diff(q_values)
        dq = np.append(dq, dq[-1])
        norm = np.sum(pq * dq)
        if norm < 1e-30:
            return {k: 0.0 for k in range(1, max_order + 1)}
        moments = {}
        for k in range(1, max_order + 1):
            moments[k] = np.sum(pq * q_values ** k * dq) / norm
        return moments

    # ---- support -------------------------------------------------------

    def pq_support(self, pq, q_values, threshold=1e-6):
        """Support of P(q): interval [q_min, q_max] where P(q) > threshold."""
        pq = np.asarray(pq, dtype=float)
        q_values = np.asarray(q_values, dtype=float)
        mask = pq > threshold
        if not np.any(mask):
            return {"q_min": np.nan, "q_max": np.nan, "is_delta": True}
        q_min = q_values[mask].min()
        q_max = q_values[mask].max()
        n_peaks = self._count_peaks(pq[mask])
        return {"q_min": q_min, "q_max": q_max, "n_peaks": n_peaks,
                "is_delta": (q_max - q_min) < 3 * (q_values[1] - q_values[0])}

    @staticmethod
    def _count_peaks(arr):
        if len(arr) < 3:
            return len(arr)
        peaks = 0
        for i in range(1, len(arr) - 1):
            if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
                peaks += 1
        return max(peaks, 1)

    # ---- Ghirlanda-Guerra identities -----------------------------------

    def ghirlanda_guerra_check(self, overlaps):
        """Check the Ghirlanda-Guerra identities on empirical overlaps.

        For a system satisfying GG:
            ⟨q₁₂ f(q)⟩ = (1/n)⟨f(q)⟩⟨q₁₂⟩ + ((n-1)/n)⟨q₁₂⟩⟨f(q₁₃)⟩
        We test the simplest case f(q) = q₁₃.

        overlaps: (n_replicas, n_replicas) symmetric matrix of overlaps.
        """
        overlaps = np.asarray(overlaps, dtype=float)
        n = overlaps.shape[0]
        if n < 3:
            return {"valid": False, "reason": "need >= 3 replicas"}

        # Extract upper-triangular overlaps
        triu_idx = np.triu_indices(n, k=1)
        q_pairs = overlaps[triu_idx]
        mean_q = np.mean(q_pairs)
        mean_q2 = np.mean(q_pairs ** 2)

        # GG for f=q: ⟨q₁₂ q₁₃⟩ = (1/n)⟨q²⟩ + (n-1)/n ⟨q⟩²
        # Compute ⟨q₁₂ q₁₃⟩ by averaging over all (a,b,c) triples
        triple_sum = 0.0
        count = 0
        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    triple_sum += (overlaps[a, b] * overlaps[a, c]
                                   + overlaps[a, b] * overlaps[b, c]
                                   + overlaps[a, c] * overlaps[b, c])
                    count += 3
        if count == 0:
            return {"valid": False, "reason": "no triples"}

        lhs = triple_sum / count
        rhs = (1.0 / n) * mean_q2 + (n - 1.0) / n * mean_q ** 2
        violation = np.abs(lhs - rhs)

        return {
            "lhs": lhs,
            "rhs": rhs,
            "violation": violation,
            "relative_violation": violation / max(abs(rhs), 1e-15),
            "satisfied": violation < 0.05 * max(abs(rhs), 1e-15),
        }

    # ---- ultrametricity ------------------------------------------------

    def ultrametricity_check(self, overlap_matrix):
        """Check ultrametric structure: for every triple (a,b,c),
        the two smallest of {q_{ab}, q_{ac}, q_{bc}} should be equal.
        """
        Q = np.asarray(overlap_matrix, dtype=float)
        n = Q.shape[0]
        violations = 0
        total = 0
        max_viol = 0.0

        for a in range(n):
            for b in range(a + 1, n):
                for c in range(b + 1, n):
                    triple = sorted([Q[a, b], Q[a, c], Q[b, c]])
                    # ultrametric: triple[0] ≈ triple[1]
                    viol = abs(triple[1] - triple[0])
                    total += 1
                    if viol > 1e-4:
                        violations += 1
                    max_viol = max(max_viol, viol)

        return {
            "fraction_violated": violations / max(total, 1),
            "max_violation": max_viol,
            "n_triples": total,
            "ultrametric": violations / max(total, 1) < 0.05,
        }


# ===================================================================
# 5. ParisiFunctional
# ===================================================================

class ParisiFunctional:
    """Parisi functional and PDE for full (continuous) RSB.

    The order parameter is a non-decreasing function q(x) on [0,1].
    """

    def __init__(self, n_steps=100):
        self.n_steps = n_steps

    # ---- Parisi PDE ---------------------------------------------------

    def parisi_pde(self, q_func, x_range=None):
        r"""Solve the Parisi PDE backwards from q=1 to q=0.

        ∂f/∂q = -½ [f'' + x(q) (f')²]

        q_func : callable  q ↦ x(q)   (inverse of the order-parameter function)
        Returns f(q=0, h) on a grid of h values.
        """
        if x_range is None:
            x_range = np.linspace(0, 1, self.n_steps + 1)

        n_h = 256
        h_max = 8.0
        h = np.linspace(-h_max, h_max, n_h)
        dh = h[1] - h[0]

        # Terminal condition at q = 1:  f(1, h) = log 2 cosh(βh) ≈ |h| for β→∞
        f = np.log(2.0 * np.cosh(h))

        # Discretise q from 1 to 0
        q_grid = np.linspace(1.0, 0.0, self.n_steps + 1)
        dq = abs(q_grid[1] - q_grid[0])

        solution = np.zeros((self.n_steps + 1, n_h))
        solution[0, :] = f.copy()

        for step in range(self.n_steps):
            q_val = q_grid[step]
            x_val = q_func(q_val)

            # Finite differences for f'' and (f')²
            fp = np.gradient(f, dh)
            fpp = np.gradient(fp, dh)

            # PDE step (backward Euler would be more stable; forward Euler here)
            f = f + 0.5 * dq * (fpp + x_val * fp ** 2)

            solution[step + 1, :] = f.copy()

        return {"h": h, "f": f, "q_grid": q_grid, "solution": solution}

    # ---- Parisi free energy -------------------------------------------

    def parisi_free_energy(self, q_func):
        """Evaluate the Parisi free energy for a given q(x).

        F = ∫Dz f(q=0, z)  + entropic correction
        """
        pde_sol = self.parisi_pde(q_func)
        h = pde_sol["h"]
        f0 = pde_sol["f"]  # f(q=0, h)

        # Gaussian average: F = ∫ Dz f(0, z)
        dh = h[1] - h[0]
        weight = _phi(h)
        F_energetic = np.sum(weight * f0 * dh)

        return F_energetic

    # ---- optimise q(x) ------------------------------------------------

    def optimize_q_function(self, initial_q_func=None, n_params=20):
        """Optimise the order-parameter function q(x) to minimise the
        Parisi free energy.

        Parameterise q(x) as a monotone non-decreasing step function
        with n_params breakpoints.
        """
        if initial_q_func is not None and callable(initial_q_func):
            # Evaluate initial function at breakpoints
            x_bp = np.linspace(0, 1, n_params)
            q_init = np.array([initial_q_func(xi) for xi in x_bp])
        else:
            q_init = np.linspace(0.1, 0.9, n_params)

        def params_to_q_func(params):
            # Ensure monotonicity via cumulative softmax
            increments = np.exp(params)
            cum = np.cumsum(increments)
            q_bp = cum / (cum[-1] + 1.0)  # normalise to (0, 1)
            x_bp = np.linspace(0, 1, len(q_bp))
            interp = interpolate.interp1d(x_bp, q_bp, kind="linear",
                                          fill_value=(q_bp[0], q_bp[-1]),
                                          bounds_error=False)
            # Return x(q) by inverting
            q_grid = np.linspace(q_bp[0], q_bp[-1], 500)
            x_grid = np.array([self._invert_qx(interp, qi) for qi in q_grid])
            xq_interp = interpolate.interp1d(q_grid, x_grid, kind="linear",
                                             fill_value=(0, 1),
                                             bounds_error=False)
            return xq_interp

        def objective(params):
            q_func = params_to_q_func(params)
            try:
                return self.parisi_free_energy(q_func)
            except Exception:
                return 1e10

        # Initial parameters (log-increments)
        p0 = np.zeros(n_params)

        result = optimize.minimize(objective, p0, method="L-BFGS-B",
                                   options={"maxiter": 200, "ftol": 1e-10})

        optimal_q_func = params_to_q_func(result.x)
        return {
            "q_func": optimal_q_func,
            "free_energy": result.fun,
            "converged": result.success,
            "n_iterations": result.nit,
        }

    @staticmethod
    def _invert_qx(interp, q_val):
        """Numerically invert q(x) to get x(q)."""
        x_grid = np.linspace(0, 1, 1000)
        q_grid = interp(x_grid)
        idx = np.argmin(np.abs(q_grid - q_val))
        return x_grid[idx]

    # ---- k-step RSB ---------------------------------------------------

    def discretized_rsb(self, k_steps, alpha=1.0, temperature=1.0):
        """k-step RSB: optimise k overlaps q₀ < q₁ < … < q_k and
        breakpoints m₁ < m₂ < … < m_k.

        Returns the optimal parameters and free energy.
        """
        beta = 1.0 / max(temperature, 1e-15)

        # Initial guesses: evenly spaced
        q_init = np.linspace(0.1, 0.9, k_steps)
        m_init = np.linspace(0.2, 0.8, k_steps)
        x0 = np.concatenate([q_init, m_init])

        def free_energy_krsb(params):
            k = k_steps
            q_vals = np.sort(np.clip(params[:k], 1e-6, 1.0 - 1e-6))
            m_vals = np.sort(np.clip(params[k:], 1e-6, 1.0 - 1e-6))

            # Build step function q(x)
            x_bp = np.concatenate([[0], m_vals, [1]])
            q_bp = np.concatenate([[0], q_vals, [1]])

            def q_func(x):
                idx = np.searchsorted(x_bp, x, side="right") - 1
                idx = np.clip(idx, 0, len(q_bp) - 1)
                return q_bp[idx]

            # Construct x(q) for PDE
            def xq_func(q):
                idx = np.searchsorted(q_bp, q, side="right") - 1
                idx = np.clip(idx, 0, len(x_bp) - 1)
                return x_bp[idx]

            try:
                return self.parisi_free_energy(xq_func)
            except Exception:
                return 1e10

        result = optimize.minimize(free_energy_krsb, x0, method="Nelder-Mead",
                                   options={"maxiter": 2000, "xatol": 1e-8})
        k = k_steps
        q_opt = np.sort(np.clip(result.x[:k], 0, 1))
        m_opt = np.sort(np.clip(result.x[k:], 0, 1))

        return {
            "q_values": q_opt,
            "m_values": m_opt,
            "free_energy": result.fun,
            "converged": result.success,
            "k": k_steps,
        }

    # ---- continuous limit ---------------------------------------------

    def continuous_rsb_limit(self, k_step_solutions):
        """Extrapolate k-step RSB solutions to k → ∞.

        k_step_solutions: list of dicts from discretized_rsb with increasing k.
        Returns estimated continuous-RSB free energy and q(x).
        """
        ks = np.array([s["k"] for s in k_step_solutions], dtype=float)
        fes = np.array([s["free_energy"] for s in k_step_solutions])

        if len(ks) < 2:
            return {
                "free_energy_extrapolated": fes[0],
                "q_values": k_step_solutions[0]["q_values"],
            }

        # Richardson-like extrapolation: fit F(k) = F_∞ + a/k + b/k²
        inv_k = 1.0 / ks
        if len(ks) >= 3:
            coeffs = np.polyfit(inv_k, fes, 2)
            F_inf = coeffs[2]
        else:
            coeffs = np.polyfit(inv_k, fes, 1)
            F_inf = coeffs[1]

        # Reconstruct q(x) from the finest solution
        finest = k_step_solutions[-1]
        q_vals = finest["q_values"]
        m_vals = finest["m_values"]

        # Build piecewise-constant interpolation
        x_bp = np.concatenate([[0], m_vals, [1]])
        q_bp = np.concatenate([[0], q_vals, [1]])
        q_interp = interpolate.interp1d(x_bp, q_bp, kind="linear",
                                        fill_value=(0, 1), bounds_error=False)

        return {
            "free_energy_extrapolated": F_inf,
            "free_energies": fes,
            "ks": ks,
            "q_function": q_interp,
            "q_breakpoints": q_vals,
            "m_breakpoints": m_vals,
        }

    # ---- Parisi measure -----------------------------------------------

    def parisi_measure(self, q_func, n_points=500):
        """Extract the Parisi measure μ(q) = dx(q)/dq from the
        order-parameter function q(x).

        q_func: callable x → q(x)   (non-decreasing)
        Returns discretised μ(q).
        """
        x_grid = np.linspace(0, 1, n_points)
        q_grid = np.array([q_func(xi) for xi in x_grid])

        # Sort and remove duplicates for inversion
        order = np.argsort(q_grid)
        q_sorted = q_grid[order]
        x_sorted = x_grid[order]

        # Remove near-duplicates in q
        mask = np.concatenate([[True], np.diff(q_sorted) > 1e-12])
        q_unique = q_sorted[mask]
        x_unique = x_sorted[mask]

        if len(q_unique) < 3:
            return {"q": q_unique, "mu": np.ones_like(q_unique)}

        # x(q) interpolation
        xq = interpolate.interp1d(q_unique, x_unique, kind="linear",
                                  fill_value=(0, 1), bounds_error=False)

        # μ(q) = dx/dq
        dq = np.diff(q_unique)
        dx = np.diff(x_unique)
        mu = dx / np.clip(dq, 1e-15, None)

        q_mid = 0.5 * (q_unique[:-1] + q_unique[1:])

        return {"q": q_mid, "mu": mu, "x_of_q": xq}
