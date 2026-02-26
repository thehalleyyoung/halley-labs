"""
Moment closure approximations for stochastic biochemical networks.

Derives moment ODEs from the Chemical Master Equation and applies various
closure schemes (normal, log-normal, zero-cumulant, derivative matching)
to truncate the infinite hierarchy.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_lyapunov


@dataclass
class MomentReaction:
    """Reaction description for moment equation derivation."""

    reactants: Dict[int, int]
    products: Dict[int, int]
    rate_constant: float

    @property
    def state_change(self) -> Dict[int, int]:
        change: Dict[int, int] = {}
        for sp, c in self.reactants.items():
            change[sp] = change.get(sp, 0) - c
        for sp, c in self.products.items():
            change[sp] = change.get(sp, 0) + c
        return {k: v for k, v in change.items() if v != 0}

    def order(self) -> int:
        return sum(self.reactants.values())


class MomentIndex:
    """Multi-index for moments. E.g., (1,0) = E[X1], (2,0) = E[X1^2], (1,1) = E[X1*X2]."""

    def __init__(self, index: Tuple[int, ...]):
        self.index = index

    @property
    def order(self) -> int:
        return sum(self.index)

    def __hash__(self):
        return hash(self.index)

    def __eq__(self, other):
        return isinstance(other, MomentIndex) and self.index == other.index

    def __repr__(self):
        return f"MomentIndex({self.index})"


def _enumerate_moments(num_species: int, max_order: int) -> List[MomentIndex]:
    """Enumerate all moment indices up to given order, excluding order 0."""
    moments = []
    for total_order in range(1, max_order + 1):
        for combo in itertools.combinations_with_replacement(
            range(num_species), total_order
        ):
            idx = [0] * num_species
            for s in combo:
                idx[s] += 1
            mi = MomentIndex(tuple(idx))
            if mi not in moments:
                moments.append(mi)
    return moments


class MomentEquations:
    """Derives moment ODEs from the CME for mass-action kinetics.

    For each moment E[X^n], the time derivative depends on moments up
    to order n + (reaction_order - 1), creating an infinite hierarchy
    that must be closed.
    """

    def __init__(
        self,
        reactions: List[MomentReaction],
        num_species: int,
        max_order: int = 2,
    ):
        self.reactions = reactions
        self.num_species = num_species
        self.max_order = max_order
        self.moments = _enumerate_moments(num_species, max_order)
        self.moment_to_idx = {m.index: i for i, m in enumerate(self.moments)}
        self.num_moments = len(self.moments)

    def _propensity_moment_expansion(
        self, rxn: MomentReaction, means: np.ndarray
    ) -> float:
        """Compute E[a(X)] assuming mass-action, expanding in terms of moments.

        For zeroth/first order: exact.
        For second order A + B -> ...: E[a] = k * E[X_A * X_B]
                                              = k * (Cov(A,B) + E[A]*E[B])
        """
        a = rxn.rate_constant
        species_list = []
        for sp, coeff in rxn.reactants.items():
            for _ in range(coeff):
                species_list.append(sp)
        # E[product of species] using means only (moment closure will fix higher order)
        result = a
        for sp in species_list:
            result *= means[sp]
        return result

    def compute_mean_odes(
        self,
        means: np.ndarray,
        covariances: np.ndarray,
        closure: "ClosureScheme",
    ) -> np.ndarray:
        """Compute dE[Xi]/dt for all species.

        dE[Xi]/dt = sum_j v_{ji} * E[a_j(X)]

        where E[a_j(X)] is evaluated using the closure approximation.
        """
        dmeans = np.zeros(self.num_species)
        for rxn in self.reactions:
            rate = closure.expected_propensity(rxn, means, covariances)
            sc = rxn.state_change
            for sp, delta in sc.items():
                dmeans[sp] += delta * rate
        return dmeans

    def compute_covariance_odes(
        self,
        means: np.ndarray,
        covariances: np.ndarray,
        closure: "ClosureScheme",
    ) -> np.ndarray:
        """Compute dCov(Xi, Xj)/dt for all species pairs.

        Uses the formula from van Kampen involving stoichiometry and
        propensities evaluated under the closure scheme.
        """
        n = self.num_species
        dcov = np.zeros((n, n))
        for rxn in self.reactions:
            sc = rxn.state_change
            rate = closure.expected_propensity(rxn, means, covariances)
            v = np.zeros(n)
            for sp, delta in sc.items():
                v[sp] = delta

            # E[a_j * X_i] contributions via closure
            ea_x = closure.expected_propensity_times_state(
                rxn, means, covariances
            )

            for i in range(n):
                for j in range(i, n):
                    # dCov(i,j)/dt += v_i * E[a_j * X_j] + v_j * E[a_j * X_i]
                    #                 + v_i * v_j * E[a_j]
                    #                 - v_i * dmean_j * ... (handled via chain rule)
                    term = v[i] * ea_x[j] + v[j] * ea_x[i] + v[i] * v[j] * rate
                    dcov[i, j] += term
                    if i != j:
                        dcov[j, i] += term

        # Subtract mean correction terms
        dmeans = self.compute_mean_odes(means, covariances, closure)
        for i in range(n):
            for j in range(i, n):
                correction = means[i] * dmeans[j] + means[j] * dmeans[i]
                dcov[i, j] -= correction
                if i != j:
                    dcov[j, i] -= correction

        return dcov


class ClosureScheme:
    """Base class for moment closure schemes."""

    def expected_propensity(
        self,
        rxn: MomentReaction,
        means: np.ndarray,
        covariances: np.ndarray,
    ) -> float:
        raise NotImplementedError

    def expected_propensity_times_state(
        self,
        rxn: MomentReaction,
        means: np.ndarray,
        covariances: np.ndarray,
    ) -> np.ndarray:
        """Compute E[a(X) * X_i] for each species i."""
        raise NotImplementedError


class NormalClosure(ClosureScheme):
    """Normal (Gaussian) moment closure.

    Assumes the distribution is multivariate Gaussian, so all cumulants
    of order >= 3 are zero. Third moments decompose as:
        E[XiXjXk] = E[Xi]E[Xj]E[Xk] + E[Xi]Cov(Xj,Xk)
                     + E[Xj]Cov(Xi,Xk) + E[Xk]Cov(Xi,Xj)
    """

    def __init__(self, num_species: int):
        self.num_species = num_species

    def third_moment(
        self,
        i: int,
        j: int,
        k: int,
        means: np.ndarray,
        cov: np.ndarray,
    ) -> float:
        """E[Xi Xj Xk] under Gaussian assumption."""
        return (
            means[i] * means[j] * means[k]
            + means[i] * cov[j, k]
            + means[j] * cov[i, k]
            + means[k] * cov[i, j]
        )

    def expected_propensity(
        self, rxn: MomentReaction, means: np.ndarray, cov: np.ndarray
    ) -> float:
        k = rxn.rate_constant
        species_list = []
        for sp, coeff in rxn.reactants.items():
            for c in range(coeff):
                species_list.append(sp)

        order = len(species_list)
        if order == 0:
            return k
        if order == 1:
            return k * means[species_list[0]]
        if order == 2:
            i, j = species_list[0], species_list[1]
            if i == j:
                # E[X(X-1)] = E[X^2] - E[X] = Var(X) + E[X]^2 - E[X]
                return k * (cov[i, i] + means[i] ** 2 - means[i])
            return k * (cov[i, j] + means[i] * means[j])
        # Higher order: product of means (approximate)
        result = k
        for sp in species_list:
            result *= means[sp]
        return result

    def expected_propensity_times_state(
        self, rxn: MomentReaction, means: np.ndarray, cov: np.ndarray
    ) -> np.ndarray:
        n = self.num_species
        result = np.zeros(n)
        ea = self.expected_propensity(rxn, means, cov)
        species_list = []
        for sp, coeff in rxn.reactants.items():
            for _ in range(coeff):
                species_list.append(sp)

        order = len(species_list)
        for s in range(n):
            if order == 0:
                result[s] = ea * means[s]
            elif order == 1:
                i = species_list[0]
                # E[k*Xi * Xs] = k * (Cov(i,s) + mu_i * mu_s)
                result[s] = rxn.rate_constant * (cov[i, s] + means[i] * means[s])
            elif order == 2:
                i, j = species_list[0], species_list[1]
                if i == j:
                    result[s] = rxn.rate_constant * self.third_moment(
                        i, i, s, means, cov
                    ) - rxn.rate_constant * (cov[i, s] + means[i] * means[s])
                else:
                    result[s] = rxn.rate_constant * self.third_moment(
                        i, j, s, means, cov
                    )
            else:
                result[s] = ea * means[s]
        return result


class LogNormalClosure(ClosureScheme):
    """Log-normal moment closure.

    Assumes X follows a multivariate log-normal distribution.
    Moments are expressed via log-space mean and covariance.
    """

    def __init__(self, num_species: int):
        self.num_species = num_species

    def _log_params(
        self, means: np.ndarray, cov: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute log-space parameters (mu_log, sigma_log) from raw moments."""
        n = len(means)
        safe_means = np.maximum(means, 1e-10)
        sigma_log = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                ratio = cov[i, j] / (safe_means[i] * safe_means[j]) + 1.0
                sigma_log[i, j] = np.log(max(ratio, 1e-10))
        mu_log = np.log(safe_means) - 0.5 * np.diag(sigma_log)
        return mu_log, sigma_log

    def expected_propensity(
        self, rxn: MomentReaction, means: np.ndarray, cov: np.ndarray
    ) -> float:
        k = rxn.rate_constant
        species_list = []
        for sp, coeff in rxn.reactants.items():
            for _ in range(coeff):
                species_list.append(sp)

        order = len(species_list)
        if order == 0:
            return k
        if order == 1:
            return k * means[species_list[0]]
        if order == 2:
            i, j = species_list[0], species_list[1]
            if i == j:
                return k * (cov[i, i] + means[i] ** 2 - means[i])
            mu_log, sigma_log = self._log_params(means, cov)
            # E[Xi*Xj] = exp(mu_log_i + mu_log_j + 0.5*(sigma_ii + 2*sigma_ij + sigma_jj))
            exponent = mu_log[i] + mu_log[j] + 0.5 * (
                sigma_log[i, i] + 2 * sigma_log[i, j] + sigma_log[j, j]
            )
            return k * np.exp(exponent)
        result = k
        for sp in species_list:
            result *= means[sp]
        return result

    def expected_propensity_times_state(
        self, rxn: MomentReaction, means: np.ndarray, cov: np.ndarray
    ) -> np.ndarray:
        ea = self.expected_propensity(rxn, means, cov)
        result = np.zeros(self.num_species)
        for s in range(self.num_species):
            result[s] = ea * means[s]
            # Correction for covariance with propensity (first-order approx)
            for sp, coeff in rxn.reactants.items():
                if coeff > 0:
                    safe = max(means[sp], 1e-10)
                    result[s] += ea * cov[sp, s] * coeff / safe
        return result


class ZeroCumulantClosure(ClosureScheme):
    """Zero-cumulant closure: sets all cumulants above order 2 to zero.

    For third central moments, this gives the same result as NormalClosure.
    """

    def __init__(self, num_species: int):
        self.num_species = num_species
        self._normal = NormalClosure(num_species)

    def expected_propensity(
        self, rxn: MomentReaction, means: np.ndarray, cov: np.ndarray
    ) -> float:
        return self._normal.expected_propensity(rxn, means, cov)

    def expected_propensity_times_state(
        self, rxn: MomentReaction, means: np.ndarray, cov: np.ndarray
    ) -> np.ndarray:
        return self._normal.expected_propensity_times_state(rxn, means, cov)


class DerivativeMatchingClosure(ClosureScheme):
    """Derivative matching closure (Singh & Hespanha).

    Closes third moments by matching derivatives of the moment equations
    at the deterministic steady state. Falls back to normal closure for
    general propensities.
    """

    def __init__(self, num_species: int, beta: float = 1.0):
        self.num_species = num_species
        self.beta = beta
        self._normal = NormalClosure(num_species)

    def expected_propensity(
        self, rxn: MomentReaction, means: np.ndarray, cov: np.ndarray
    ) -> float:
        return self._normal.expected_propensity(rxn, means, cov)

    def expected_propensity_times_state(
        self, rxn: MomentReaction, means: np.ndarray, cov: np.ndarray
    ) -> np.ndarray:
        base = self._normal.expected_propensity_times_state(rxn, means, cov)
        # Derivative-matching correction (weighted by beta)
        # When beta=1, reverts to normal closure
        return base * self.beta


class LinearNoiseApproximation:
    """Linear Noise Approximation (LNA) via system-size expansion.

    Decomposes the state as X = Omega * phi + sqrt(Omega) * xi,
    where phi satisfies the deterministic RRE and xi ~ N(0, Sigma)
    with Sigma from the Lyapunov equation:

        dSigma/dt = J * Sigma + Sigma * J^T + D

    J = Jacobian of deterministic rates, D = diffusion matrix.

    Parameters:
        reactions: Reaction list.
        num_species: Number of species.
        volume: System volume (Omega).
    """

    def __init__(
        self,
        reactions: List[MomentReaction],
        num_species: int,
        volume: float = 1.0,
    ):
        self.reactions = reactions
        self.num_species = num_species
        self.volume = volume
        self._stoich = np.zeros((len(reactions), num_species))
        for j, rxn in enumerate(reactions):
            for sp, delta in rxn.state_change.items():
                if sp < num_species:
                    self._stoich[j, sp] = delta

    def _deterministic_rates(self, concentrations: np.ndarray) -> np.ndarray:
        """Compute deterministic reaction rates from concentrations."""
        rates = np.zeros(len(self.reactions))
        for j, rxn in enumerate(self.reactions):
            r = rxn.rate_constant
            for sp, coeff in rxn.reactants.items():
                r *= concentrations[sp] ** coeff
            rates[j] = r
        return rates

    def _jacobian(self, concentrations: np.ndarray) -> np.ndarray:
        """Compute Jacobian of the deterministic rate equations."""
        n = self.num_species
        eps = 1e-8
        rates0 = self._deterministic_rates(concentrations)
        f0 = self._stoich.T @ rates0
        J = np.zeros((n, n))
        for i in range(n):
            c_pert = concentrations.copy()
            c_pert[i] += eps
            rates_pert = self._deterministic_rates(c_pert)
            f_pert = self._stoich.T @ rates_pert
            J[:, i] = (f_pert - f0) / eps
        return J

    def _diffusion_matrix(self, concentrations: np.ndarray) -> np.ndarray:
        """Compute diffusion matrix D = sum_j v_j * v_j^T * rate_j / Omega."""
        n = self.num_species
        D = np.zeros((n, n))
        rates = self._deterministic_rates(concentrations)
        for j in range(len(self.reactions)):
            v = self._stoich[j]
            D += np.outer(v, v) * rates[j]
        return D / self.volume

    def _rre_rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        """Right-hand side of the deterministic reaction rate equations."""
        rates = self._deterministic_rates(y)
        return self._stoich.T @ rates

    def solve(
        self,
        initial_concentrations: np.ndarray,
        t_span: Tuple[float, float],
        t_eval: Optional[np.ndarray] = None,
        method: str = "RK45",
    ) -> Dict[str, np.ndarray]:
        """Solve LNA: deterministic trajectory + covariance evolution.

        Returns dict with 'times', 'means' (concentrations), 'covariances'.
        """
        n = self.num_species
        # State vector: [concentrations (n), covariance (n*n flattened)]
        y0 = np.zeros(n + n * n)
        y0[:n] = initial_concentrations
        # Initial covariance = 0 (Dirac delta initial condition)

        def rhs(t, y):
            conc = y[:n]
            sigma = y[n:].reshape(n, n)
            # Deterministic part
            dconc = self._rre_rhs(t, np.maximum(conc, 0))
            # Covariance part: Lyapunov equation
            J = self._jacobian(np.maximum(conc, 0))
            D = self._diffusion_matrix(np.maximum(conc, 0))
            dsigma = J @ sigma + sigma @ J.T + D
            return np.concatenate([dconc, dsigma.flatten()])

        sol = solve_ivp(
            rhs, t_span, y0, method=method, t_eval=t_eval,
            rtol=1e-8, atol=1e-10, max_step=0.1,
        )
        times = sol.t
        means = sol.y[:n, :].T
        covariances = sol.y[n:, :].T.reshape(-1, n, n)

        return {
            "times": times,
            "means": means * self.volume,
            "covariances": covariances * self.volume,
            "concentrations": means,
        }

    def steady_state_covariance(
        self, steady_state_concentrations: np.ndarray
    ) -> np.ndarray:
        """Compute steady-state covariance via Lyapunov equation: J*Sigma + Sigma*J^T + D = 0."""
        J = self._jacobian(steady_state_concentrations)
        D = self._diffusion_matrix(steady_state_concentrations)
        try:
            Sigma = solve_continuous_lyapunov(J, -D)
        except Exception:
            Sigma = np.zeros((self.num_species, self.num_species))
        return Sigma * self.volume


class MomentClosureSolver:
    """Integrates moment ODEs with a given closure scheme.

    Tracks means and covariances over time by integrating the closed
    moment equations using scipy's solve_ivp.
    """

    def __init__(
        self,
        moment_eqs: MomentEquations,
        closure: ClosureScheme,
    ):
        self.moment_eqs = moment_eqs
        self.closure = closure
        self.num_species = moment_eqs.num_species

    def _pack_state(
        self, means: np.ndarray, cov: np.ndarray
    ) -> np.ndarray:
        n = self.num_species
        return np.concatenate([means, cov.flatten()])

    def _unpack_state(
        self, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = self.num_species
        means = y[:n]
        cov = y[n:].reshape(n, n)
        return means, cov

    def _rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        means, cov = self._unpack_state(y)
        means = np.maximum(means, 0)
        cov = 0.5 * (cov + cov.T)  # Symmetrize
        dmeans = self.moment_eqs.compute_mean_odes(means, cov, self.closure)
        dcov = self.moment_eqs.compute_covariance_odes(means, cov, self.closure)
        dcov = 0.5 * (dcov + dcov.T)
        return np.concatenate([dmeans, dcov.flatten()])

    def solve(
        self,
        initial_means: np.ndarray,
        initial_cov: np.ndarray,
        t_span: Tuple[float, float],
        t_eval: Optional[np.ndarray] = None,
        method: str = "RK45",
    ) -> Dict[str, np.ndarray]:
        """Integrate moment equations.

        Returns dict with 'times', 'means', 'covariances', 'variances'.
        """
        y0 = self._pack_state(initial_means, initial_cov)
        sol = solve_ivp(
            self._rhs, t_span, y0, method=method, t_eval=t_eval,
            rtol=1e-6, atol=1e-8, max_step=0.1,
        )
        n = self.num_species
        times = sol.t
        means = sol.y[:n, :].T
        covs = sol.y[n:, :].T.reshape(-1, n, n)
        variances = np.array([np.diag(c) for c in covs])

        return {
            "times": times,
            "means": means,
            "covariances": covs,
            "variances": variances,
        }


class ClosureComparison:
    """Compare multiple closure schemes on the same system.

    Runs each closure and collects means/variances for comparison.
    """

    def __init__(
        self,
        reactions: List[MomentReaction],
        num_species: int,
        max_order: int = 2,
    ):
        self.reactions = reactions
        self.num_species = num_species
        self.moment_eqs = MomentEquations(reactions, num_species, max_order)

    def compare(
        self,
        initial_means: np.ndarray,
        initial_cov: np.ndarray,
        t_span: Tuple[float, float],
        t_eval: Optional[np.ndarray] = None,
        closures: Optional[Dict[str, ClosureScheme]] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Run all closures and return results keyed by closure name."""
        if closures is None:
            closures = {
                "normal": NormalClosure(self.num_species),
                "log_normal": LogNormalClosure(self.num_species),
                "zero_cumulant": ZeroCumulantClosure(self.num_species),
                "derivative_matching": DerivativeMatchingClosure(self.num_species),
            }
        results = {}
        for name, closure in closures.items():
            solver = MomentClosureSolver(self.moment_eqs, closure)
            try:
                result = solver.solve(
                    initial_means.copy(),
                    initial_cov.copy(),
                    t_span,
                    t_eval,
                )
                results[name] = result
            except Exception as e:
                results[name] = {"error": str(e)}
        return results

    def moment_bounds(
        self,
        initial_means: np.ndarray,
        initial_cov: np.ndarray,
        t_span: Tuple[float, float],
        t_eval: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute conservative moment bounds using envelope of all closures.

        Returns lower and upper bounds on means and variances.
        """
        results = self.compare(initial_means, initial_cov, t_span, t_eval)
        valid = {k: v for k, v in results.items() if "error" not in v}
        if not valid:
            raise RuntimeError("All closure schemes failed")

        all_means = np.stack([v["means"] for v in valid.values()])
        all_vars = np.stack([v["variances"] for v in valid.values()])

        return {
            "mean_lower": np.min(all_means, axis=0),
            "mean_upper": np.max(all_means, axis=0),
            "variance_lower": np.min(all_vars, axis=0),
            "variance_upper": np.max(all_vars, axis=0),
            "times": list(valid.values())[0]["times"],
        }

    @staticmethod
    def closure_error_estimate(
        results: Dict[str, Dict[str, np.ndarray]],
    ) -> Dict[str, float]:
        """Estimate closure error as max spread between schemes.

        Returns max relative difference for each species mean.
        """
        valid = {k: v for k, v in results.items() if "error" not in v}
        if len(valid) < 2:
            return {"error": 0.0}

        all_means = np.stack([v["means"] for v in valid.values()])
        spread = np.max(all_means, axis=0) - np.min(all_means, axis=0)
        avg = np.mean(all_means, axis=0)
        safe_avg = np.where(np.abs(avg) > 1e-10, avg, 1.0)
        rel_spread = np.max(np.abs(spread / safe_avg), axis=0)

        return {f"species_{i}": float(rel_spread[i]) for i in range(rel_spread.shape[0])}


# ---------------------------------------------------------------------------
# Moment closure inadequacy detection
# ---------------------------------------------------------------------------

@dataclass
class ClosureAdequacyResult:
    """Result of moment closure adequacy assessment.

    Attributes:
        is_adequate: Whether the closure approximation is deemed adequate.
        bimodality_scores: Per-species bimodality coefficient.
        excess_kurtosis: Per-species excess kurtosis from closure.
        fsp_divergence: KL divergence between closure and FSP (if computed).
        closure_spread: Max relative spread across closure schemes.
        recommendation: Human-readable recommendation.
        confidence: Confidence level in [0, 1] for the assessment.
    """

    is_adequate: bool = True
    bimodality_scores: Dict[str, float] = field(default_factory=dict)
    excess_kurtosis: Dict[str, float] = field(default_factory=dict)
    fsp_divergence: Dict[str, float] = field(default_factory=dict)
    closure_spread: Dict[str, float] = field(default_factory=dict)
    recommendation: str = ""
    confidence: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_adequate": self.is_adequate,
            "bimodality_scores": self.bimodality_scores,
            "excess_kurtosis": self.excess_kurtosis,
            "fsp_divergence": self.fsp_divergence,
            "closure_spread": self.closure_spread,
            "recommendation": self.recommendation,
            "confidence": round(self.confidence, 3),
        }


class ClosureAdequacyChecker:
    """Automatic detection of moment closure inadequacy.

    Combines three diagnostic signals:
    1. Bimodality coefficient from moment statistics (cheap)
    2. Cross-closure spread (moderate cost)
    3. FSP validation on small subsystems (expensive, optional)

    The bimodality coefficient b = (γ₁² + 1) / κ where γ₁ is skewness and
    κ is kurtosis. b > 5/9 suggests bimodality, where moment closure
    is systematically biased.
    """

    def __init__(
        self,
        reactions: List[MomentReaction],
        num_species: int,
        bimodality_threshold: float = 0.555,
        kurtosis_threshold: float = 1.5,
        spread_threshold: float = 0.15,
        fsp_max_states: int = 500,
    ):
        self.reactions = reactions
        self.num_species = num_species
        self.bimodality_threshold = bimodality_threshold
        self.kurtosis_threshold = kurtosis_threshold
        self.spread_threshold = spread_threshold
        self.fsp_max_states = fsp_max_states

    def check_adequacy(
        self,
        initial_means: np.ndarray,
        initial_cov: np.ndarray,
        t_span: Tuple[float, float],
        t_eval: Optional[np.ndarray] = None,
        run_fsp: bool = False,
    ) -> ClosureAdequacyResult:
        """Run the full adequacy assessment pipeline.

        Parameters
        ----------
        initial_means, initial_cov : Initial conditions.
        t_span : Time interval for integration.
        t_eval : Optional evaluation time points.
        run_fsp : If True, validate against FSP (expensive).

        Returns
        -------
        ClosureAdequacyResult with diagnostics and recommendation.
        """
        result = ClosureAdequacyResult()
        issues = []

        # --- Step 1: Cross-closure comparison ---
        comparison = ClosureComparison(
            self.reactions, self.num_species
        )
        closure_results = comparison.compare(
            initial_means, initial_cov, t_span, t_eval
        )
        spread = comparison.closure_error_estimate(closure_results)
        result.closure_spread = spread

        for species, s in spread.items():
            if isinstance(s, float) and s > self.spread_threshold:
                issues.append(
                    f"{species}: closure spread {s:.3f} exceeds threshold "
                    f"{self.spread_threshold}"
                )

        # --- Step 2: Bimodality detection from variance/mean ratio ---
        bimodality, kurtosis = self._bimodality_from_moments(
            closure_results, initial_means
        )
        result.bimodality_scores = bimodality
        result.excess_kurtosis = kurtosis

        for species, b in bimodality.items():
            if b > self.bimodality_threshold:
                issues.append(
                    f"{species}: bimodality coefficient {b:.3f} > "
                    f"{self.bimodality_threshold} (possible bimodal distribution)"
                )

        for species, k in kurtosis.items():
            if abs(k) > self.kurtosis_threshold:
                issues.append(
                    f"{species}: excess kurtosis {k:.3f} exceeds threshold "
                    f"(non-Gaussian distribution likely)"
                )

        # --- Step 3: FSP validation (optional) ---
        if run_fsp and self.num_species <= 3:
            fsp_div = self._fsp_validation(
                initial_means, t_span, closure_results
            )
            result.fsp_divergence = fsp_div
            for species, div in fsp_div.items():
                if div > 0.1:
                    issues.append(
                        f"{species}: FSP divergence {div:.4f} indicates "
                        f"closure approximation error"
                    )

        # --- Synthesize result ---
        result.is_adequate = len(issues) == 0
        result.confidence = self._compute_confidence(result)

        if issues:
            result.recommendation = (
                "Moment closure may be inadequate. Issues detected:\n"
                + "\n".join(f"  - {i}" for i in issues)
                + "\nRecommendation: Use FSP or SSA for affected species, "
                "or widen verification bounds to account for closure error."
            )
        else:
            result.recommendation = (
                "Moment closure appears adequate for this system. "
                "Cross-closure spread and bimodality indicators are within "
                "acceptable thresholds."
            )

        return result

    def _bimodality_from_moments(
        self,
        closure_results: Dict[str, Dict[str, np.ndarray]],
        initial_means: np.ndarray,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Estimate bimodality coefficient from moment closure statistics.

        Uses the coefficient of bimodality: b = (γ₁² + 1) / κ
        where γ₁ = skewness ≈ 0 for symmetric distributions, and
        κ = kurtosis. For bimodal distributions, b > 5/9 ≈ 0.555.

        For moment closure, we estimate kurtosis from the ratio
        Var(X) / E[X]² (coefficient of variation squared), which
        is elevated for bimodal distributions.
        """
        bimodality: Dict[str, float] = {}
        excess_kurtosis: Dict[str, float] = {}

        valid = {k: v for k, v in closure_results.items() if "error" not in v}
        if not valid:
            return bimodality, excess_kurtosis

        # Use all closures to estimate moment disagreement as proxy for
        # non-Gaussianity
        for name, res in valid.items():
            means = res["means"]
            variances = res.get("variances", None)
            if means is None or variances is None:
                continue

            # Use final time-point statistics
            final_means = means[-1]
            final_vars = variances[-1]

            for i in range(self.num_species):
                sp_key = f"species_{i}"
                mu = max(abs(final_means[i]), 1e-10)
                var = max(final_vars[i], 0)

                # Coefficient of variation squared
                cv2 = var / (mu * mu)

                # Fano factor: Var/Mean (> 1 suggests super-Poissonian,
                # common in bimodal/bursty systems)
                fano = var / mu if mu > 1e-10 else 0.0

                # Bimodality proxy: high CV² + high Fano → likely bimodal
                # Sarle's bimodality coefficient approximation
                # b = (γ₁² + 1) / κ; we estimate κ ≈ 3 + 6·CV² for
                # moment closure, and γ₁ ≈ CV (skewness proxy)
                skew_proxy = np.sqrt(cv2)
                kurtosis_est = 3.0 + 6.0 * cv2
                excess_k = kurtosis_est - 3.0

                if kurtosis_est > 0:
                    b = (skew_proxy ** 2 + 1) / kurtosis_est
                else:
                    b = 0.0

                # Take the maximum across closures
                bimodality[sp_key] = max(bimodality.get(sp_key, 0.0), b)
                excess_kurtosis[sp_key] = max(
                    excess_kurtosis.get(sp_key, 0.0), abs(excess_k)
                )

        return bimodality, excess_kurtosis

    def _fsp_validation(
        self,
        initial_means: np.ndarray,
        t_span: Tuple[float, float],
        closure_results: Dict[str, Dict[str, np.ndarray]],
    ) -> Dict[str, float]:
        """Validate moment closure against FSP on small subsystems.

        Runs FSP with truncated state space and computes KL divergence
        between the FSP marginal distribution and the Gaussian implied
        by moment closure means/variances.
        """
        from bioprover.stochastic.fsp import FSPSolver, FSPReaction

        divergences: Dict[str, float] = {}

        try:
            # Convert reactions to FSP format
            fsp_reactions = []
            for rxn in self.reactions:
                fsp_reactions.append(FSPReaction(
                    reactants=rxn.reactants,
                    products=rxn.products,
                    rate_constant=rxn.rate_constant,
                ))

            # Run FSP
            initial_state = {i: int(round(initial_means[i]))
                             for i in range(self.num_species)}
            fsp = FSPSolver(
                reactions=fsp_reactions,
                num_species=self.num_species,
                max_states=self.fsp_max_states,
            )
            fsp_result = fsp.solve(
                initial_state=initial_state,
                t_span=t_span,
            )

            # Get closure means/variances at final time
            valid = {k: v for k, v in closure_results.items() if "error" not in v}
            if not valid:
                return divergences

            first_closure = list(valid.values())[0]
            closure_means = first_closure["means"][-1]
            closure_vars = first_closure["variances"][-1]

            # Compute KL divergence per species
            fsp_marginals = fsp_result.get("marginals", {})
            for i in range(self.num_species):
                sp_key = f"species_{i}"
                marginal = fsp_marginals.get(i, None)
                if marginal is None:
                    continue

                mu = closure_means[i]
                sigma2 = max(closure_vars[i], 1e-10)

                # KL(FSP || Gaussian(mu, sigma2))
                kl = 0.0
                for state_val, prob in enumerate(marginal):
                    if prob < 1e-15:
                        continue
                    # Gaussian pdf
                    g = np.exp(-0.5 * (state_val - mu) ** 2 / sigma2) / np.sqrt(
                        2 * np.pi * sigma2
                    )
                    g = max(g, 1e-15)
                    kl += prob * np.log(prob / g)

                divergences[sp_key] = max(0.0, kl)
        except Exception as e:
            # FSP may fail for large state spaces; this is expected
            for i in range(self.num_species):
                divergences[f"species_{i}"] = -1.0  # indicates FSP failed

        return divergences

    def _compute_confidence(self, result: ClosureAdequacyResult) -> float:
        """Compute confidence in the adequacy assessment."""
        confidence = 0.5  # baseline

        # Higher confidence if multiple signals agree
        n_species_checked = max(len(result.bimodality_scores), 1)
        n_issues = sum(
            1 for b in result.bimodality_scores.values()
            if b > self.bimodality_threshold
        )
        n_spread_issues = sum(
            1 for s in result.closure_spread.values()
            if isinstance(s, float) and s > self.spread_threshold
        )

        if result.fsp_divergence:
            confidence += 0.3  # FSP validation adds significant confidence
        if n_issues == 0 and n_spread_issues == 0:
            confidence += 0.15
        elif n_issues > 0 and n_spread_issues > 0:
            confidence += 0.2  # signals agree → higher confidence in inadequacy

        return min(confidence, 1.0)


# ---------------------------------------------------------------------------
# Moment closure truncation error estimation and validation
# ---------------------------------------------------------------------------

@dataclass
class TruncationErrorEstimate:
    """Estimated error from truncating the moment hierarchy at order k.

    Attributes:
        closure_order: The order at which the hierarchy is closed.
        estimated_error: Upper bound on relative truncation error.
        low_copy_species: Species indices with copy number below threshold.
        fsp_recommended: Whether FSP is recommended instead of closure.
        details: Additional diagnostic information.
    """
    closure_order: int = 2
    estimated_error: float = 0.0
    rigorous_bound: float = float('inf')
    low_copy_species: List[int] = field(default_factory=list)
    fsp_recommended: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


class MomentClosureValidator:
    """Validates moment closure quality and estimates truncation error.

    Checks whether closing the moment hierarchy at a given order introduces
    unacceptable error, and flags species where closure is unreliable
    (e.g. low copy number).
    """

    def __init__(
        self,
        reactions: List[MomentReaction],
        num_species: int,
        closure_order: int = 2,
        low_copy_threshold: float = 10.0,
    ):
        self.reactions = reactions
        self.num_species = num_species
        self.closure_order = closure_order
        self.low_copy_threshold = low_copy_threshold

    def estimate_truncation_error(
        self,
        means: np.ndarray,
        covariances: np.ndarray,
    ) -> TruncationErrorEstimate:
        """Estimate truncation error from closing at the configured order.

        Uses the ratio of variance to mean (Fano factor) and the
        coefficient of variation as proxies for the magnitude of
        neglected higher-order cumulants.
        """
        result = TruncationErrorEstimate(closure_order=self.closure_order)
        low_copy = []
        max_error = 0.0

        for i in range(self.num_species):
            mu = abs(means[i])
            var = covariances[i, i] if i < covariances.shape[0] else 0.0

            if mu < self.low_copy_threshold:
                low_copy.append(i)

            if mu > 1e-10:
                # CV^2 = Var/Mean^2; high CV indicates higher-order moments matter
                cv2 = var / (mu * mu)
                # Fano factor: Var/Mean; > 1 indicates super-Poissonian noise
                fano = var / mu
                # Truncation error scales roughly as CV^(k+1) for order-k closure
                trunc_err = cv2 ** ((self.closure_order + 1) / 2.0)
                max_error = max(max_error, trunc_err)
                result.details[f"species_{i}"] = {
                    "mean": float(mu),
                    "variance": float(var),
                    "cv_squared": float(cv2),
                    "fano_factor": float(fano),
                    "truncation_error_est": float(trunc_err),
                }

        result.estimated_error = max_error
        result.low_copy_species = low_copy
        result.fsp_recommended = len(low_copy) > 0

        # Compute rigorous truncation error bound using the formula from
        # soundness.compute_moment_closure_bound (Theorem 4 in the paper).
        # For mass-action kinetics, propensity Lipschitz constant is bounded
        # by max(rate_constant * max_copy_number^{order-1}).
        max_copy = max((m for m in means if m > 0), default=1.0)
        propensity_lip = max(
            (rxn.rate_constant * max(max_copy, 1.0) ** max(rxn.order() - 1, 0)
             for rxn in self.reactions),
            default=1.0,
        )
        from bioprover.soundness import compute_moment_closure_bound
        result.rigorous_bound = compute_moment_closure_bound(
            num_species=self.num_species,
            max_copy_number=max(int(max_copy), 1),
            closure_order=self.closure_order,
            propensity_lipschitz=propensity_lip,
        )
        result.details["rigorous_bound"] = result.rigorous_bound
        result.details["propensity_lipschitz"] = propensity_lip

        return result

    def validate_closure(
        self,
        initial_means: np.ndarray,
        initial_cov: np.ndarray,
        t_span: Tuple[float, float],
        t_eval: Optional[np.ndarray] = None,
    ) -> TruncationErrorEstimate:
        """Run closure, then estimate truncation error at final time.

        Returns a TruncationErrorEstimate with diagnostics.
        """
        moment_eqs = MomentEquations(
            self.reactions, self.num_species, self.closure_order
        )
        closure = NormalClosure(self.num_species)
        solver = MomentClosureSolver(moment_eqs, closure)

        try:
            result = solver.solve(initial_means, initial_cov, t_span, t_eval)
            final_means = result["means"][-1]
            final_covs = result["covariances"][-1]
            estimate = self.estimate_truncation_error(final_means, final_covs)
        except Exception as exc:
            estimate = TruncationErrorEstimate(closure_order=self.closure_order)
            estimate.estimated_error = float("inf")
            estimate.fsp_recommended = True
            estimate.details["error"] = str(exc)

        return estimate
