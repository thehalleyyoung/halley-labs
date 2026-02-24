"""
Statistical mechanics models of neural networks.

Implements partition function estimation, free energy computation,
entropy estimation, replica method, cavity method, order parameters,
phase transitions, and Boltzmann machine analysis.
"""

import numpy as np
from scipy.optimize import brentq, minimize, minimize_scalar, root_scalar
from scipy.integrate import quad, dblquad
from scipy.special import erf, expit, logsumexp as scipy_logsumexp
from scipy.linalg import eigvalsh
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Callable
import warnings


def _logsumexp(a: np.ndarray) -> float:
    """Numerically stable log-sum-exp."""
    a_max = np.max(a)
    if not np.isfinite(a_max):
        return float(a_max)
    return float(a_max + np.log(np.sum(np.exp(a - a_max))))


@dataclass
class StatMechReport:
    """Report from statistical mechanics analysis."""
    partition_function_log: float = 0.0
    free_energy: float = 0.0
    entropy: float = 0.0
    energy: float = 0.0
    temperature: float = 1.0
    order_parameters: Dict[str, float] = field(default_factory=dict)
    phase: str = "unknown"
    phase_transition_detected: bool = False
    critical_temperature: float = 0.0
    replica_results: Dict[str, float] = field(default_factory=dict)
    cavity_results: Dict[str, float] = field(default_factory=dict)
    boltzmann_results: Dict[str, Any] = field(default_factory=dict)
    susceptibility: float = 0.0
    specific_heat: float = 0.0
    magnetization: float = 0.0


@dataclass
class ModelSpec:
    """Specification for statistical mechanics analysis."""
    depth: int = 5
    width: int = 100
    sigma_w: float = 1.0
    sigma_b: float = 0.0
    activation: str = "relu"
    temperature: float = 1.0
    n_classes: int = 2
    input_dim: int = 10
    dataset_size: int = 1000


class PartitionFunctionEstimator:
    """Estimate partition function via importance sampling and related methods."""

    def __init__(self, n_samples: int = 50000):
        self.n_samples = n_samples

    def estimate_log_Z_importance_sampling(
        self, energy_fn: Callable, dim: int, temperature: float = 1.0,
        proposal_std: float = 1.0
    ) -> Dict[str, float]:
        """Estimate log Z using importance sampling with Gaussian proposal."""
        beta = 1.0 / (temperature + 1e-12)
        samples = np.random.randn(self.n_samples, dim) * proposal_std
        energies = np.array([energy_fn(s) for s in samples])

        log_weights = -beta * energies
        log_proposal = -0.5 * np.sum(samples ** 2, axis=1) / proposal_std ** 2 \
                       - 0.5 * dim * np.log(2 * np.pi * proposal_std ** 2)

        log_importance_weights = log_weights - log_proposal
        log_Z = _logsumexp(log_importance_weights) - np.log(self.n_samples)

        weights = np.exp(log_importance_weights - np.max(log_importance_weights))
        weights /= np.sum(weights) + 1e-30
        ess = 1.0 / (np.sum(weights ** 2) + 1e-30)

        mean_energy = np.sum(weights * energies)
        var_energy = np.sum(weights * (energies - mean_energy) ** 2)

        return {
            "log_Z": float(log_Z),
            "mean_energy": float(mean_energy),
            "var_energy": float(var_energy),
            "effective_sample_size": float(ess),
            "ess_fraction": float(ess / self.n_samples),
        }

    def estimate_log_Z_annealed(
        self, energy_fn: Callable, dim: int,
        n_temperatures: int = 20, max_temperature: float = 10.0
    ) -> Dict[str, float]:
        """Annealed importance sampling for log Z estimation."""
        betas = np.linspace(0, 1.0, n_temperatures)
        temperatures = 1.0 / (betas + 1e-12)
        temperatures[0] = max_temperature

        samples = np.random.randn(self.n_samples // 10, dim)
        log_weights = np.zeros(len(samples))

        for i in range(1, len(betas)):
            db = betas[i] - betas[i - 1]
            energies = np.array([energy_fn(s) for s in samples])
            log_weights -= db * energies

            current_beta = betas[i]
            noise = np.random.randn(*samples.shape) * 0.1
            samples_proposed = samples + noise
            energies_proposed = np.array([energy_fn(s) for s in samples_proposed])

            log_accept = -current_beta * (energies_proposed - energies)
            accept = np.log(np.random.rand(len(samples))) < log_accept
            samples[accept] = samples_proposed[accept]

        log_Z = _logsumexp(log_weights) - np.log(len(samples))
        log_Z_high_T = 0.5 * dim * np.log(2 * np.pi * max_temperature)
        log_Z += log_Z_high_T

        return {
            "log_Z": float(log_Z),
            "n_temperatures": n_temperatures,
            "log_Z_high_T": float(log_Z_high_T),
        }

    def estimate_log_Z_wang_landau(
        self, energy_fn: Callable, dim: int,
        energy_range: Tuple[float, float] = (-10.0, 10.0),
        n_bins: int = 100, n_iterations: int = 50000
    ) -> Dict[str, float]:
        """Wang-Landau algorithm for density of states estimation."""
        bin_edges = np.linspace(energy_range[0], energy_range[1], n_bins + 1)
        log_g = np.zeros(n_bins)
        histogram = np.zeros(n_bins)
        ln_f = 1.0

        state = np.random.randn(dim) * 0.5
        energy = energy_fn(state)

        def get_bin(e):
            idx = int((e - energy_range[0]) / (energy_range[1] - energy_range[0]) * n_bins)
            return max(0, min(n_bins - 1, idx))

        for iteration in range(n_iterations):
            proposal = state + np.random.randn(dim) * 0.3
            e_proposal = energy_fn(proposal)

            bin_current = get_bin(energy)
            bin_proposal = get_bin(e_proposal)

            if (energy_range[0] <= e_proposal <= energy_range[1] and
                np.random.rand() < np.exp(log_g[bin_current] - log_g[bin_proposal])):
                state = proposal
                energy = e_proposal
                bin_current = bin_proposal

            log_g[bin_current] += ln_f
            histogram[bin_current] += 1

            if iteration > 0 and iteration % 10000 == 0:
                if np.min(histogram[histogram > 0]) > 0.8 * np.mean(histogram[histogram > 0]):
                    ln_f *= 0.5
                    histogram[:] = 0

        log_g -= np.max(log_g)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]

        return {
            "log_density_of_states": log_g.tolist(),
            "energy_bins": bin_centers.tolist(),
            "final_ln_f": float(ln_f),
        }


class FreeEnergyComputer:
    """Compute free energy using various approximations."""

    def __init__(self, n_samples: int = 30000):
        self.n_samples = n_samples

    def annealed_free_energy(
        self, model_spec: ModelSpec
    ) -> Dict[str, float]:
        """Compute annealed free energy F_ann = -T * <log Z>."""
        T = model_spec.temperature
        beta = 1.0 / (T + 1e-12)
        N = model_spec.width
        D = model_spec.input_dim
        P = model_spec.dataset_size

        log_Z_weight = 0.5 * N * D * np.log(2 * np.pi * model_spec.sigma_w ** 2)
        log_Z_bias = 0.5 * N * np.log(2 * np.pi * (model_spec.sigma_b ** 2 + 1e-6)) \
            if model_spec.sigma_b > 0 else 0.0
        log_Z_total = (log_Z_weight + log_Z_bias) * model_spec.depth

        alpha = P / N
        entropy_data = 0.5 * P * np.log(2 * np.pi * np.e)
        F_ann = -T * (log_Z_total - beta * entropy_data)

        return {
            "F_annealed": float(F_ann),
            "log_Z_weight": float(log_Z_total),
            "entropy_data_term": float(entropy_data),
            "alpha": float(alpha),
            "temperature": float(T),
        }

    def quenched_free_energy_replica(
        self, model_spec: ModelSpec, n_replicas: int = 0
    ) -> Dict[str, float]:
        """Compute quenched free energy using replica trick.

        F_quenched = -T * E[log Z] ≈ -T * lim_{n->0} (E[Z^n] - 1) / n
        For replica-symmetric ansatz.
        """
        T = model_spec.temperature
        N = model_spec.width
        D = model_spec.input_dim
        P = model_spec.dataset_size
        alpha = P / N

        sigma_w2 = model_spec.sigma_w ** 2

        def rs_free_energy(q):
            """Replica-symmetric free energy as function of overlap q."""
            if q <= 0 or q >= 1:
                return 1e10
            energetic = -0.5 * alpha * np.log(1 - q + 1e-12)
            entropic = -0.5 * (np.log(1 - q + 1e-12) + q / (1 - q + 1e-12))
            return float(energetic + entropic)

        result = minimize_scalar(rs_free_energy, bounds=(0.01, 0.99), method="bounded")
        q_star = float(result.x)
        f_rs = float(result.fun)

        F_quenched = -T * (f_rs + 0.5 * np.log(2 * np.pi * sigma_w2) * D)

        return {
            "F_quenched": float(F_quenched),
            "q_star": float(q_star),
            "f_replica_symmetric": float(f_rs),
            "alpha": float(alpha),
            "is_rs_stable": bool(q_star < 0.95),
        }

    def mean_field_free_energy(
        self, model_spec: ModelSpec
    ) -> Dict[str, float]:
        """Compute free energy in mean field approximation."""
        T = model_spec.temperature
        N = model_spec.width
        D = model_spec.input_dim
        sigma_w2 = model_spec.sigma_w ** 2

        q = sigma_w2 * D / N
        if model_spec.activation == "relu":
            q_eff = q / 2.0
        elif model_spec.activation == "tanh":
            q_eff = q * (2.0 / np.pi)
        else:
            q_eff = q

        F_mf = -0.5 * N * np.log(2 * np.pi * np.e * q_eff + 1e-12)
        S_mf = 0.5 * N * (1 + np.log(2 * np.pi * q_eff + 1e-12))
        E_mf = F_mf + T * S_mf

        return {
            "F_mean_field": float(F_mf),
            "S_mean_field": float(S_mf),
            "E_mean_field": float(E_mf),
            "q_effective": float(q_eff),
        }


class EntropyEstimator:
    """Estimate entropy from free energy and energy."""

    def __init__(self, n_samples: int = 30000):
        self.n_samples = n_samples

    def thermodynamic_entropy(
        self, free_energy: float, energy: float, temperature: float
    ) -> float:
        """Compute entropy from S = (E - F) / T."""
        return (energy - free_energy) / (temperature + 1e-12)

    def microcanonical_entropy(
        self, energy_fn: Callable, dim: int,
        energy_target: float, energy_width: float = 0.5
    ) -> float:
        """Estimate microcanonical entropy S(E) = log(Omega(E))."""
        samples = np.random.randn(self.n_samples, dim)
        energies = np.array([energy_fn(s) for s in samples])
        in_shell = np.sum(np.abs(energies - energy_target) < energy_width)
        total_volume = (2 * np.pi) ** (dim / 2.0) / gamma_fn_approx(dim / 2.0 + 1)
        fraction = in_shell / self.n_samples

        if fraction > 0:
            return float(np.log(fraction * total_volume + 1e-30))
        return float(-np.inf)

    def temperature_derivative_entropy(
        self, energy_fn: Callable, dim: int,
        temperature: float, delta_t: float = 0.1
    ) -> Dict[str, float]:
        """Estimate entropy via numerical derivative of free energy."""
        pf = PartitionFunctionEstimator(self.n_samples)

        result_t = pf.estimate_log_Z_importance_sampling(energy_fn, dim, temperature)
        result_tp = pf.estimate_log_Z_importance_sampling(energy_fn, dim, temperature + delta_t)
        result_tm = pf.estimate_log_Z_importance_sampling(energy_fn, dim,
                                                           max(0.01, temperature - delta_t))

        F_t = -temperature * result_t["log_Z"]
        F_tp = -(temperature + delta_t) * result_tp["log_Z"]
        F_tm = -max(0.01, temperature - delta_t) * result_tm["log_Z"]

        S = -(F_tp - F_tm) / (2 * delta_t)
        C = temperature * (F_tp - 2 * F_t + F_tm) / (delta_t ** 2)

        return {
            "entropy": float(S),
            "specific_heat": float(C),
            "free_energy": float(F_t),
        }


def gamma_fn_approx(x: float) -> float:
    """Approximate gamma function for positive real x."""
    if x <= 0:
        return 1.0
    try:
        from scipy.special import gamma
        return float(gamma(x))
    except (ImportError, OverflowError):
        if x > 170:
            return float("inf")
        result = 1.0
        while x > 1:
            x -= 1
            result *= x
        return result


class ReplicaMethod:
    """Replica method for computing typical-case behavior."""

    def __init__(self, n_samples: int = 20000):
        self.n_samples = n_samples

    def replica_symmetric_equations(
        self, alpha: float, sigma_w2: float, temperature: float = 1.0
    ) -> Dict[str, float]:
        """Solve replica-symmetric saddle point equations.

        For a simple perceptron-like model with Gaussian weights.
        """
        beta = 1.0 / (temperature + 1e-12)

        def saddle_point(params):
            q, qhat = params
            if q <= 0 or q >= sigma_w2:
                return [1.0, 1.0]
            eq1 = q - sigma_w2 * self._compute_overlap_integral(qhat, beta)
            eq2 = qhat - alpha * beta / (1 - beta * (sigma_w2 - q) + 1e-12)
            return [eq1, eq2]

        best_q, best_qhat = 0.5 * sigma_w2, 1.0
        best_residual = float("inf")

        for q_init in np.linspace(0.1 * sigma_w2, 0.9 * sigma_w2, 5):
            for qhat_init in [0.1, 0.5, 1.0, 2.0, 5.0]:
                try:
                    from scipy.optimize import fsolve
                    solution = fsolve(saddle_point, [q_init, qhat_init], full_output=True)
                    if solution[2] == 1:
                        residual = np.sum(np.array(solution[1]["fvec"]) ** 2)
                        if residual < best_residual:
                            best_residual = residual
                            best_q = float(solution[0][0])
                            best_qhat = float(solution[0][1])
                except Exception:
                    continue

        free_energy = self._rs_free_energy(best_q, best_qhat, alpha, sigma_w2, beta)

        return {
            "q_star": float(best_q),
            "qhat_star": float(best_qhat),
            "free_energy": float(free_energy),
            "overlap": float(best_q / sigma_w2),
            "generalization_error": float(1.0 - best_q / sigma_w2),
            "alpha": float(alpha),
        }

    def _compute_overlap_integral(self, qhat: float, beta: float) -> float:
        """Compute overlap integral in RS equations."""
        n = min(self.n_samples, 10000)
        z = np.random.randn(n)
        effective_field = np.sqrt(max(qhat, 0)) * z
        return float(np.mean(np.tanh(beta * effective_field) ** 2))

    def _rs_free_energy(self, q: float, qhat: float, alpha: float,
                        sigma_w2: float, beta: float) -> float:
        """Compute replica-symmetric free energy."""
        term1 = -0.5 * q * qhat
        term2 = 0.5 * alpha * np.log(1 + beta * (sigma_w2 - q) + 1e-12)
        n = min(self.n_samples, 10000)
        z = np.random.randn(n)
        effective_field = np.sqrt(max(qhat, 0)) * z
        term3 = np.mean(np.log(2 * np.cosh(beta * effective_field) + 1e-12))
        return float(term1 + term2 + term3)

    def check_rs_stability(self, q: float, qhat: float, alpha: float,
                            sigma_w2: float, beta: float) -> Dict[str, float]:
        """Check de Almeida-Thouless stability of RS solution."""
        n = min(self.n_samples, 10000)
        z = np.random.randn(n)
        effective_field = np.sqrt(max(qhat, 0)) * z
        sech4_mean = float(np.mean(1.0 / np.cosh(beta * effective_field) ** 4))

        at_parameter = alpha * beta ** 2 * sech4_mean / \
                       (1 + beta * (sigma_w2 - q) + 1e-12) ** 2

        return {
            "at_parameter": float(at_parameter),
            "is_rs_stable": bool(at_parameter < 1.0),
            "sech4_mean": float(sech4_mean),
        }

    def learning_curve(self, sigma_w2: float, alpha_range: Tuple[float, float] = (0.1, 10.0),
                       n_points: int = 30, temperature: float = 0.0) -> Dict[str, Any]:
        """Compute learning curve: generalization error vs alpha."""
        alphas = np.linspace(alpha_range[0], alpha_range[1], n_points)
        gen_errors = []
        overlaps = []
        free_energies = []

        for alpha in alphas:
            result = self.replica_symmetric_equations(alpha, sigma_w2, max(temperature, 0.01))
            gen_errors.append(result["generalization_error"])
            overlaps.append(result["overlap"])
            free_energies.append(result["free_energy"])

        return {
            "alphas": alphas.tolist(),
            "generalization_errors": gen_errors,
            "overlaps": overlaps,
            "free_energies": free_energies,
        }


class CavityMethod:
    """Cavity method (belief propagation) for neural network analysis."""

    def __init__(self, n_iterations: int = 100, damping: float = 0.5):
        self.n_iterations = n_iterations
        self.damping = damping

    def run_bp_perceptron(
        self, X: np.ndarray, y: np.ndarray, sigma_w2: float = 1.0,
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """Run belief propagation for a single-layer perceptron."""
        P, N = X.shape
        beta = 1.0 / (temperature + 1e-12)

        a_means = np.zeros(N)
        a_vars = np.ones(N) * sigma_w2
        omega = np.zeros(P)
        V = np.ones(P) * sigma_w2

        converged = False
        for iteration in range(self.n_iterations):
            old_a_means = a_means.copy()

            for mu in range(P):
                omega[mu] = X[mu] @ a_means
                V[mu] = X[mu] ** 2 @ a_vars

            g = np.zeros(P)
            dg = np.zeros(P)
            for mu in range(P):
                v_total = V[mu] + 1e-6
                z = y[mu] * omega[mu] / np.sqrt(v_total)
                g[mu] = y[mu] * self._gaussian_ratio(z) / np.sqrt(v_total)
                dg[mu] = -(g[mu] ** 2 + g[mu] * y[mu] * omega[mu] / v_total)

            new_a_means = sigma_w2 * (X.T @ g)
            new_a_vars = sigma_w2 / (1 - sigma_w2 * (X ** 2).T @ dg + 1e-6)
            new_a_vars = np.clip(new_a_vars, 1e-6, 10 * sigma_w2)

            a_means = self.damping * old_a_means + (1 - self.damping) * new_a_means
            a_vars = np.clip(new_a_vars, 1e-6, 10 * sigma_w2)

            if np.max(np.abs(a_means - old_a_means)) < 1e-6:
                converged = True
                break

        predictions = np.sign(X @ a_means)
        accuracy = float(np.mean(predictions == y))
        overlap = float(np.mean(a_means ** 2) / sigma_w2)

        return {
            "means": a_means,
            "variances": a_vars,
            "predictions": predictions,
            "accuracy": accuracy,
            "overlap": overlap,
            "converged": converged,
            "n_iterations_used": iteration + 1,
            "free_energy_estimate": float(-np.sum(np.log(a_vars + 1e-12)) / (2 * N)),
        }

    def _gaussian_ratio(self, z: float) -> float:
        """Compute Gaussian integral ratio phi(z)/Phi(z)."""
        z = np.clip(z, -30, 30)
        phi = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
        Phi = 0.5 * (1 + erf(z / np.sqrt(2)))
        return phi / (Phi + 1e-30)

    def run_tap_equations(
        self, J: np.ndarray, h: np.ndarray, temperature: float = 1.0
    ) -> Dict[str, Any]:
        """Run TAP (Thouless-Anderson-Palmer) mean field equations."""
        N = len(h)
        beta = 1.0 / (temperature + 1e-12)

        m = np.tanh(beta * h)

        converged = False
        for iteration in range(self.n_iterations):
            old_m = m.copy()

            local_field = h + J @ m - beta * np.diag(J ** 2) @ m * (1 - m ** 2)
            new_m = np.tanh(beta * local_field)
            m = self.damping * old_m + (1 - self.damping) * new_m

            if np.max(np.abs(m - old_m)) < 1e-8:
                converged = True
                break

        magnetization = float(np.mean(np.abs(m)))
        overlap = float(np.mean(m ** 2))
        energy = float(-0.5 * m @ J @ m - h @ m)

        entropy = float(-np.sum(
            0.5 * (1 + m) * np.log(0.5 * (1 + m) + 1e-12) +
            0.5 * (1 - m) * np.log(0.5 * (1 - m) + 1e-12)
        ))

        free_energy = energy - temperature * entropy

        return {
            "magnetizations": m,
            "mean_magnetization": float(magnetization),
            "overlap": float(overlap),
            "energy": float(energy),
            "entropy": float(entropy),
            "free_energy": float(free_energy),
            "converged": converged,
        }


class OrderParameterTracker:
    """Track order parameters for neural network ensembles."""

    def __init__(self, n_samples: int = 10000):
        self.n_samples = n_samples

    def compute_overlap(self, w1: np.ndarray, w2: np.ndarray) -> float:
        """Compute overlap between two weight configurations."""
        norm1 = np.linalg.norm(w1) + 1e-12
        norm2 = np.linalg.norm(w2) + 1e-12
        return float(np.dot(w1.ravel(), w2.ravel()) / (norm1 * norm2))

    def compute_magnetization(self, weights: np.ndarray, reference: np.ndarray) -> float:
        """Compute magnetization relative to reference configuration."""
        return self.compute_overlap(weights, reference)

    def compute_edwards_anderson(self, weight_samples: List[np.ndarray]) -> float:
        """Compute Edwards-Anderson order parameter q_EA = <q^2>."""
        n = len(weight_samples)
        if n < 2:
            return 0.0
        overlaps = []
        for i in range(min(n, 50)):
            for j in range(i + 1, min(n, 50)):
                q = self.compute_overlap(weight_samples[i], weight_samples[j])
                overlaps.append(q ** 2)
        return float(np.mean(overlaps)) if overlaps else 0.0

    def compute_overlap_distribution(
        self, weight_samples: List[np.ndarray], n_bins: int = 50
    ) -> Dict[str, Any]:
        """Compute distribution of overlaps P(q)."""
        n = len(weight_samples)
        overlaps = []
        for i in range(min(n, 100)):
            for j in range(i + 1, min(n, 100)):
                q = self.compute_overlap(weight_samples[i], weight_samples[j])
                overlaps.append(q)

        if not overlaps:
            return {"bins": np.zeros(n_bins).tolist(), "counts": np.zeros(n_bins).tolist()}

        overlaps = np.array(overlaps)
        counts, bin_edges = np.histogram(overlaps, bins=n_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        return {
            "bins": bin_centers.tolist(),
            "counts": counts.tolist(),
            "mean_overlap": float(np.mean(overlaps)),
            "std_overlap": float(np.std(overlaps)),
            "q_EA": float(np.mean(overlaps ** 2)),
        }

    def track_training_order_params(
        self, weight_trajectory: List[np.ndarray],
        reference: Optional[np.ndarray] = None
    ) -> Dict[str, List[float]]:
        """Track order parameters during training."""
        if reference is None:
            reference = weight_trajectory[-1]

        magnetizations = []
        self_overlaps = []
        norms = []

        for w in weight_trajectory:
            magnetizations.append(self.compute_magnetization(w, reference))
            self_overlaps.append(float(np.mean(w ** 2)))
            norms.append(float(np.linalg.norm(w)))

        return {
            "magnetizations": magnetizations,
            "self_overlaps": self_overlaps,
            "norms": norms,
        }


class PhaseTransitionDetector:
    """Detect phase transitions from order parameter behavior."""

    def __init__(self):
        pass

    def detect_from_order_parameter(
        self, temperatures: np.ndarray, order_params: np.ndarray,
        method: str = "derivative"
    ) -> Dict[str, Any]:
        """Detect phase transition from order parameter vs temperature."""
        if len(temperatures) < 5:
            return {"detected": False, "reason": "insufficient data"}

        sort_idx = np.argsort(temperatures)
        T = temperatures[sort_idx]
        op = order_params[sort_idx]

        if method == "derivative":
            dop_dT = np.gradient(op, T)
            d2op_dT2 = np.gradient(dop_dT, T)
            peak_idx = np.argmax(np.abs(dop_dT))
            Tc = T[peak_idx]

            is_first_order = abs(dop_dT[peak_idx]) > 5 * np.mean(np.abs(dop_dT))
            transition_type = "first_order" if is_first_order else "continuous"

        elif method == "binder":
            m2 = order_params
            m4 = order_params ** 2
            binder = 1 - m4 / (3 * m2 ** 2 + 1e-12)
            mid_idx = len(binder) // 2
            Tc = T[mid_idx]
            transition_type = "continuous"
        else:
            Tc = T[len(T) // 2]
            transition_type = "unknown"

        return {
            "detected": True,
            "Tc": float(Tc),
            "transition_type": transition_type,
            "order_parameter_at_Tc": float(np.interp(Tc, T, op)),
        }

    def compute_susceptibility(
        self, temperatures: np.ndarray, order_params: np.ndarray
    ) -> np.ndarray:
        """Compute generalized susceptibility chi = d(order_param)/dT."""
        return np.gradient(order_params, temperatures)

    def compute_specific_heat(
        self, temperatures: np.ndarray, energies: np.ndarray
    ) -> np.ndarray:
        """Compute specific heat C = dE/dT."""
        return np.gradient(energies, temperatures)

    def find_critical_temperature(
        self, compute_order_param: Callable,
        T_range: Tuple[float, float] = (0.1, 5.0),
        n_points: int = 50
    ) -> Dict[str, float]:
        """Find critical temperature by scanning."""
        T_vals = np.linspace(T_range[0], T_range[1], n_points)
        order_params = np.array([compute_order_param(T) for T in T_vals])

        dop = np.gradient(order_params, T_vals)
        peak_idx = np.argmax(np.abs(dop))

        return {
            "Tc": float(T_vals[peak_idx]),
            "max_susceptibility": float(np.abs(dop[peak_idx])),
            "order_param_at_Tc": float(order_params[peak_idx]),
        }


class BoltzmannMachineAnalyzer:
    """Analyze Restricted Boltzmann Machines."""

    def __init__(self, n_samples: int = 20000):
        self.n_samples = n_samples

    def compute_rbm_partition_function_bound(
        self, W: np.ndarray, a: np.ndarray, b: np.ndarray
    ) -> Dict[str, float]:
        """Compute bounds on RBM partition function.

        W: visible-hidden weights (n_visible, n_hidden)
        a: visible biases
        b: hidden biases
        """
        n_visible, n_hidden = W.shape

        log_Z_upper = n_visible * np.log(2) + np.sum(np.log(2 * np.cosh(b)))
        for j in range(n_hidden):
            log_Z_upper += np.log(np.cosh(np.linalg.norm(W[:, j])))

        log_Z_lower = np.sum(np.log(2 * np.cosh(a)))
        for j in range(n_hidden):
            log_Z_lower += np.log(2 * np.cosh(b[j]))

        log_Z_ais = self._ais_estimate(W, a, b)

        return {
            "log_Z_upper_bound": float(log_Z_upper),
            "log_Z_lower_bound": float(log_Z_lower),
            "log_Z_ais_estimate": float(log_Z_ais),
            "bound_gap": float(log_Z_upper - log_Z_lower),
            "n_visible": n_visible,
            "n_hidden": n_hidden,
        }

    def _ais_estimate(self, W: np.ndarray, a: np.ndarray, b: np.ndarray,
                      n_betas: int = 100) -> float:
        """Annealed importance sampling estimate of log Z."""
        n_visible, n_hidden = W.shape
        betas = np.linspace(0, 1, n_betas)
        n_chains = min(100, self.n_samples // 100)

        v = (np.random.rand(n_chains, n_visible) > 0.5).astype(float)
        log_weights = np.zeros(n_chains)

        for k in range(1, len(betas)):
            beta_k = betas[k]
            beta_km1 = betas[k - 1]

            hidden_input = v @ W + b
            log_p_k = v @ (beta_k * a) + np.sum(np.log(1 + np.exp(
                np.clip(beta_k * hidden_input, -500, 500))), axis=1)
            log_p_km1 = v @ (beta_km1 * a) + np.sum(np.log(1 + np.exp(
                np.clip(beta_km1 * hidden_input, -500, 500))), axis=1)
            log_weights += log_p_k - log_p_km1

            h_prob = expit(beta_k * hidden_input)
            h = (np.random.rand(n_chains, n_hidden) < h_prob).astype(float)
            v_prob = expit(beta_k * (h @ W.T + a))
            v = (np.random.rand(n_chains, n_visible) < v_prob).astype(float)

        log_Z_base = n_visible * np.log(2) + n_hidden * np.log(2)
        log_Z = log_Z_base + _logsumexp(log_weights) - np.log(n_chains)

        return float(log_Z)

    def compute_free_energy_gap(
        self, W: np.ndarray, a: np.ndarray, b: np.ndarray,
        data: np.ndarray
    ) -> Dict[str, float]:
        """Compute free energy gap between data and model."""
        hidden_input = data @ W + b
        free_energy_data = -data @ a - np.sum(
            np.log(1 + np.exp(np.clip(hidden_input, -500, 500))), axis=1)
        mean_fe_data = float(np.mean(free_energy_data))

        n_visible = W.shape[0]
        random_data = (np.random.rand(self.n_samples, n_visible) > 0.5).astype(float)
        hidden_input_rand = random_data @ W + b
        free_energy_random = -random_data @ a - np.sum(
            np.log(1 + np.exp(np.clip(hidden_input_rand, -500, 500))), axis=1)
        mean_fe_random = float(np.mean(free_energy_random))

        return {
            "mean_fe_data": mean_fe_data,
            "mean_fe_random": mean_fe_random,
            "fe_gap": float(mean_fe_random - mean_fe_data),
            "relative_gap": float((mean_fe_random - mean_fe_data) /
                                  (abs(mean_fe_random) + 1e-12)),
        }

    def analyze_rbm_spectrum(self, W: np.ndarray) -> Dict[str, Any]:
        """Analyze weight matrix spectrum of RBM."""
        singular_values = svdvals(W)
        n_visible, n_hidden = W.shape

        effective_temp = float(np.sqrt(np.mean(W ** 2)))
        spectral_norm = float(singular_values[0]) if len(singular_values) > 0 else 0.0

        sv_normalized = singular_values / (np.sum(singular_values) + 1e-12)
        entropy = -np.sum(sv_normalized * np.log(sv_normalized + 1e-12))
        effective_rank = np.exp(entropy)

        return {
            "singular_values": singular_values.tolist(),
            "spectral_norm": spectral_norm,
            "effective_temperature": effective_temp,
            "effective_rank": float(effective_rank),
            "condition_number": float(singular_values[0] / (singular_values[-1] + 1e-12))
                if len(singular_values) > 1 else 1.0,
        }


class StatMechAnalyzer:
    """Main statistical mechanics analyzer for neural networks."""

    def __init__(self, n_samples: int = 20000):
        self.n_samples = n_samples
        self.pf_estimator = PartitionFunctionEstimator(n_samples)
        self.fe_computer = FreeEnergyComputer(n_samples)
        self.entropy_estimator = EntropyEstimator(n_samples)
        self.replica = ReplicaMethod(n_samples)
        self.cavity = CavityMethod()
        self.order_tracker = OrderParameterTracker(n_samples)
        self.phase_detector = PhaseTransitionDetector()
        self.boltzmann = BoltzmannMachineAnalyzer(n_samples)

    def analyze(self, model_spec: ModelSpec) -> StatMechReport:
        """Full statistical mechanics analysis."""
        report = StatMechReport()
        report.temperature = model_spec.temperature

        fe_ann = self.fe_computer.annealed_free_energy(model_spec)
        report.free_energy = fe_ann["F_annealed"]

        fe_mf = self.fe_computer.mean_field_free_energy(model_spec)
        report.energy = fe_mf["E_mean_field"]
        report.entropy = fe_mf["S_mean_field"]

        alpha = model_spec.dataset_size / model_spec.width
        try:
            replica_result = self.replica.replica_symmetric_equations(
                alpha, model_spec.sigma_w ** 2, model_spec.temperature
            )
            report.replica_results = replica_result
            report.order_parameters["overlap"] = replica_result["overlap"]
            report.order_parameters["q_star"] = replica_result["q_star"]
            report.magnetization = replica_result["overlap"]
        except Exception as e:
            report.replica_results = {"error": str(e)}

        N = model_spec.width
        D = model_spec.input_dim
        X = np.random.randn(min(model_spec.dataset_size, 200), D) / np.sqrt(D)
        w_true = np.random.randn(D) * model_spec.sigma_w / np.sqrt(D)
        y = np.sign(X @ w_true)

        try:
            cavity_result = self.cavity.run_bp_perceptron(
                X, y, model_spec.sigma_w ** 2, model_spec.temperature
            )
            report.cavity_results = {
                "accuracy": cavity_result["accuracy"],
                "overlap": cavity_result["overlap"],
                "converged": cavity_result["converged"],
                "n_iterations": cavity_result["n_iterations_used"],
            }
        except Exception as e:
            report.cavity_results = {"error": str(e)}

        temperatures = np.linspace(0.1, 5.0, 20)
        order_params = []
        for T in temperatures:
            try:
                spec_T = ModelSpec(
                    depth=model_spec.depth, width=model_spec.width,
                    sigma_w=model_spec.sigma_w, sigma_b=model_spec.sigma_b,
                    temperature=T, input_dim=model_spec.input_dim,
                    dataset_size=model_spec.dataset_size
                )
                result = self.replica.replica_symmetric_equations(
                    alpha, model_spec.sigma_w ** 2, T
                )
                order_params.append(result["overlap"])
            except Exception:
                order_params.append(0.0)

        order_params = np.array(order_params)
        transition = self.phase_detector.detect_from_order_parameter(temperatures, order_params)
        if transition["detected"]:
            report.phase_transition_detected = True
            report.critical_temperature = transition["Tc"]

        if report.magnetization > 0.5:
            report.phase = "ordered"
        elif report.magnetization < 0.1:
            report.phase = "disordered"
        else:
            report.phase = "critical"

        susceptibility = self.phase_detector.compute_susceptibility(temperatures, order_params)
        report.susceptibility = float(np.max(np.abs(susceptibility)))

        energies_temp = -np.cumsum(order_params)
        specific_heat = self.phase_detector.compute_specific_heat(temperatures, energies_temp)
        report.specific_heat = float(np.max(np.abs(specific_heat)))

        W_rbm = np.random.randn(D, N) * model_spec.sigma_w / np.sqrt(D)
        a_rbm = np.zeros(D)
        b_rbm = np.zeros(N)
        try:
            rbm_result = self.boltzmann.compute_rbm_partition_function_bound(W_rbm, a_rbm, b_rbm)
            report.boltzmann_results = rbm_result
        except Exception as e:
            report.boltzmann_results = {"error": str(e)}

        log_Z = report.free_energy / (-model_spec.temperature) if model_spec.temperature > 0 else 0
        report.partition_function_log = log_Z

        return report

    def scan_phase_diagram(
        self, model_spec: ModelSpec,
        sigma_w_range: Tuple[float, float] = (0.5, 3.0),
        alpha_range: Tuple[float, float] = (0.1, 10.0),
        n_grid: int = 10
    ) -> Dict[str, Any]:
        """Scan phase diagram in (sigma_w, alpha) space."""
        sigma_w_vals = np.linspace(sigma_w_range[0], sigma_w_range[1], n_grid)
        alpha_vals = np.linspace(alpha_range[0], alpha_range[1], n_grid)

        phase_map = np.zeros((n_grid, n_grid))
        overlap_map = np.zeros((n_grid, n_grid))

        for i, sw in enumerate(sigma_w_vals):
            for j, alpha in enumerate(alpha_vals):
                try:
                    result = self.replica.replica_symmetric_equations(
                        alpha, sw ** 2, model_spec.temperature
                    )
                    overlap_map[i, j] = result["overlap"]
                    if result["overlap"] > 0.9:
                        phase_map[i, j] = 2
                    elif result["overlap"] > 0.1:
                        phase_map[i, j] = 1
                    else:
                        phase_map[i, j] = 0
                except Exception:
                    phase_map[i, j] = -1

        return {
            "sigma_w_values": sigma_w_vals.tolist(),
            "alpha_values": alpha_vals.tolist(),
            "phase_map": phase_map.tolist(),
            "overlap_map": overlap_map.tolist(),
        }
