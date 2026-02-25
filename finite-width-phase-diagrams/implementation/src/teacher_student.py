"""
Teacher-student model analysis for neural networks.

Implements order parameter evolution, generalization error dynamics,
overparameterization analysis, curriculum effect, online learning dynamics,
phase transitions in learning, and information-theoretic lower bounds.
"""

import numpy as np
from scipy.optimize import minimize, brentq, fsolve
from scipy.integrate import solve_ivp, quad
from scipy.special import erf, erfc
from scipy.linalg import eigvalsh, svdvals, norm
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Callable
import warnings


@dataclass
class TSReport:
    """Report from teacher-student analysis."""
    generalization_error: float = 0.0
    training_error: float = 0.0
    order_parameters: Dict[str, float] = field(default_factory=dict)
    learning_curve: Dict[str, List[float]] = field(default_factory=dict)
    phase_transition_detected: bool = False
    phase_transition_alpha: float = 0.0
    overparameterization_ratio: float = 0.0
    information_lower_bound: float = 0.0
    curriculum_effect: Dict[str, float] = field(default_factory=dict)
    online_dynamics: Dict[str, Any] = field(default_factory=dict)
    convergence_time: float = 0.0
    recovery_possible: bool = True


@dataclass
class TeacherSpec:
    """Teacher network specification."""
    input_dim: int = 10
    hidden_dim: int = 5
    depth: int = 1
    sigma_w: float = 1.0
    activation: str = "relu"
    weights: Optional[List[np.ndarray]] = None

    def generate_weights(self) -> List[np.ndarray]:
        """Generate random teacher weights."""
        if self.weights is not None:
            return self.weights
        weights = []
        prev_dim = self.input_dim
        for layer in range(self.depth):
            w = np.random.randn(prev_dim, self.hidden_dim) * self.sigma_w / np.sqrt(prev_dim)
            weights.append(w)
            prev_dim = self.hidden_dim
        w_out = np.random.randn(prev_dim, 1) / np.sqrt(prev_dim)
        weights.append(w_out)
        return weights


@dataclass
class StudentSpec:
    """Student network specification."""
    input_dim: int = 10
    hidden_dim: int = 10
    depth: int = 1
    sigma_w: float = 1.0
    activation: str = "relu"
    learning_rate: float = 0.01


@dataclass
class DataDist:
    """Data distribution specification."""
    input_dim: int = 10
    distribution: str = "gaussian"
    noise_level: float = 0.0
    n_samples: int = 1000
    covariance: Optional[np.ndarray] = None


class ActivationFunctions:
    """Activation functions for teacher-student models."""

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    @staticmethod
    def erf_act(x: np.ndarray) -> np.ndarray:
        return erf(x / np.sqrt(2))

    @staticmethod
    def tanh_act(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    @staticmethod
    def sign_act(x: np.ndarray) -> np.ndarray:
        return np.sign(x)

    @staticmethod
    def get(name: str) -> Callable:
        mapping = {
            "relu": ActivationFunctions.relu,
            "erf": ActivationFunctions.erf_act,
            "tanh": ActivationFunctions.tanh_act,
            "sign": ActivationFunctions.sign_act,
        }
        return mapping.get(name, ActivationFunctions.relu)


class OrderParameterEvolution:
    """Track evolution of order parameters in teacher-student setup."""

    def __init__(self, n_samples: int = 50000):
        self.n_samples = n_samples

    def compute_order_parameters(
        self, teacher_weights: List[np.ndarray],
        student_weights: List[np.ndarray]
    ) -> Dict[str, float]:
        """Compute order parameters: overlaps between student and teacher."""
        W_t = teacher_weights[0]  # (D, K) teacher first layer
        W_s = student_weights[0]  # (D, M) student first layer

        D = W_t.shape[0]
        K = W_t.shape[1]
        M = W_s.shape[1]

        # Q_{ij} = (1/D) W_s^T W_s - student-student overlap
        Q = W_s.T @ W_s / D

        # R_{ik} = (1/D) W_s^T W_t - student-teacher overlap
        R = W_s.T @ W_t / D

        # T_{kl} = (1/D) W_t^T W_t - teacher-teacher overlap
        T = W_t.T @ W_t / D

        alignment = np.sum(R ** 2) / (np.sqrt(np.sum(Q ** 2) * np.sum(T ** 2)) + 1e-12)
        mean_overlap = float(np.mean(np.abs(R)))
        max_overlap = float(np.max(np.abs(R)))

        return {
            "alignment": float(alignment),
            "mean_overlap": mean_overlap,
            "max_overlap": max_overlap,
            "student_norm": float(np.mean(np.diag(Q))),
            "teacher_norm": float(np.mean(np.diag(T))),
            "Q_trace": float(np.trace(Q)),
            "R_frobenius": float(np.sqrt(np.sum(R ** 2))),
        }

    def track_evolution(
        self, teacher_weights: List[np.ndarray],
        student_weights_trajectory: List[List[np.ndarray]]
    ) -> List[Dict[str, float]]:
        """Track order parameters over training."""
        trajectory = []
        for student_weights in student_weights_trajectory:
            op = self.compute_order_parameters(teacher_weights, student_weights)
            trajectory.append(op)
        return trajectory


class GeneralizationErrorDynamics:
    """Compute generalization error dynamics for teacher-student."""

    def __init__(self, n_samples: int = 50000):
        self.n_samples = n_samples

    def compute_gen_error(
        self, teacher_weights: List[np.ndarray],
        student_weights: List[np.ndarray],
        activation_fn: Callable,
        data_dist: DataDist
    ) -> float:
        """Compute generalization error on fresh data."""
        D = data_dist.input_dim
        if data_dist.covariance is not None:
            L = np.linalg.cholesky(data_dist.covariance)
            X = np.random.randn(self.n_samples, D) @ L.T
        else:
            X = np.random.randn(self.n_samples, D) / np.sqrt(D)

        t_hidden = activation_fn(X @ teacher_weights[0])
        t_output = t_hidden @ teacher_weights[-1]
        t_output = t_output.ravel()
        if data_dist.noise_level > 0:
            t_output += np.random.randn(self.n_samples) * data_dist.noise_level

        s_hidden = activation_fn(X @ student_weights[0])
        s_output = s_hidden @ student_weights[-1]
        s_output = s_output.ravel()

        gen_error = float(np.mean((t_output - s_output) ** 2))
        return gen_error

    def compute_gen_error_analytic_linear(
        self, teacher_weights: np.ndarray,
        student_weights: np.ndarray,
        noise_level: float = 0.0
    ) -> float:
        """Analytic generalization error for linear teacher-student."""
        D = teacher_weights.shape[0]
        diff = teacher_weights - student_weights
        gen_error = np.sum(diff ** 2) / D + noise_level ** 2
        return float(gen_error)

    def learning_curve(
        self, teacher_spec: TeacherSpec, student_spec: StudentSpec,
        data_dist: DataDist, n_epochs: int = 100, batch_size: int = 32
    ) -> Dict[str, List[float]]:
        """Simulate learning curve: generalization error vs training epoch."""
        D = teacher_spec.input_dim
        teacher_weights = teacher_spec.generate_weights()
        activation_fn = ActivationFunctions.get(teacher_spec.activation)

        W_s = [np.random.randn(D, student_spec.hidden_dim) *
               student_spec.sigma_w / np.sqrt(D)]
        W_out = np.random.randn(student_spec.hidden_dim, 1) / np.sqrt(student_spec.hidden_dim)
        student_weights = W_s + [W_out]

        gen_errors = []
        train_errors = []
        epochs = []

        lr = student_spec.learning_rate

        for epoch in range(n_epochs):
            if data_dist.covariance is not None:
                L = np.linalg.cholesky(data_dist.covariance)
                X_batch = np.random.randn(batch_size, D) @ L.T
            else:
                X_batch = np.random.randn(batch_size, D) / np.sqrt(D)

            t_hidden = activation_fn(X_batch @ teacher_weights[0])
            y_batch = (t_hidden @ teacher_weights[-1]).ravel()
            if data_dist.noise_level > 0:
                y_batch += np.random.randn(batch_size) * data_dist.noise_level

            s_hidden = activation_fn(X_batch @ student_weights[0])
            s_output = (s_hidden @ student_weights[-1]).ravel()
            error = s_output - y_batch
            train_error = float(np.mean(error ** 2))

            eps = 1e-5
            grad_W_out = s_hidden.T @ error.reshape(-1, 1) / batch_size
            student_weights[-1] -= lr * grad_W_out

            pre = X_batch @ student_weights[0]
            dphi = (activation_fn(pre + eps) - activation_fn(pre - eps)) / (2 * eps)
            delta = (error.reshape(-1, 1) * student_weights[-1].T) * dphi
            grad_W1 = X_batch.T @ delta / batch_size
            student_weights[0] -= lr * grad_W1

            gen_error = self.compute_gen_error(
                teacher_weights, student_weights, activation_fn, data_dist
            )

            gen_errors.append(gen_error)
            train_errors.append(train_error)
            epochs.append(epoch)

        return {
            "epochs": epochs,
            "gen_errors": gen_errors,
            "train_errors": train_errors,
        }


class OverparameterizationAnalyzer:
    """Analyze effect of overparameterization (student wider than teacher)."""

    def __init__(self, n_samples: int = 30000):
        self.n_samples = n_samples

    def analyze_width_ratio(
        self, teacher_spec: TeacherSpec, data_dist: DataDist,
        width_ratios: List[float], n_epochs: int = 50
    ) -> Dict[str, Any]:
        """Analyze generalization as function of student/teacher width ratio."""
        dynamics = GeneralizationErrorDynamics(self.n_samples)
        results = []

        for ratio in width_ratios:
            student_width = max(1, int(teacher_spec.hidden_dim * ratio))
            student_spec = StudentSpec(
                input_dim=teacher_spec.input_dim,
                hidden_dim=student_width,
                activation=teacher_spec.activation,
                learning_rate=0.01,
            )

            curve = dynamics.learning_curve(
                teacher_spec, student_spec, data_dist, n_epochs=n_epochs
            )

            final_gen_error = curve["gen_errors"][-1] if curve["gen_errors"] else float("inf")
            best_gen_error = min(curve["gen_errors"]) if curve["gen_errors"] else float("inf")

            results.append({
                "width_ratio": float(ratio),
                "student_width": student_width,
                "final_gen_error": float(final_gen_error),
                "best_gen_error": float(best_gen_error),
                "n_student_params": student_width * teacher_spec.input_dim + student_width,
            })

        return {
            "results": results,
            "width_ratios": [r["width_ratio"] for r in results],
            "final_gen_errors": [r["final_gen_error"] for r in results],
            "best_gen_errors": [r["best_gen_error"] for r in results],
        }

    def interpolation_threshold(self, teacher_spec: TeacherSpec,
                                 data_dist: DataDist) -> Dict[str, float]:
        """Compute interpolation threshold: n_params = n_samples."""
        D = teacher_spec.input_dim
        K = teacher_spec.hidden_dim
        n_teacher_params = D * K + K
        alpha_threshold = n_teacher_params / data_dist.n_samples

        return {
            "n_teacher_params": n_teacher_params,
            "n_samples": data_dist.n_samples,
            "alpha_threshold": float(alpha_threshold),
            "is_overparameterized": bool(alpha_threshold < 1.0),
        }


class CurriculumAnalyzer:
    """Analyze effect of curriculum learning in teacher-student setup."""

    def __init__(self, n_samples: int = 20000):
        self.n_samples = n_samples

    def compare_curricula(
        self, teacher_spec: TeacherSpec, student_spec: StudentSpec,
        data_dist: DataDist, n_epochs: int = 50
    ) -> Dict[str, Any]:
        """Compare different curriculum strategies."""
        dynamics = GeneralizationErrorDynamics(self.n_samples)

        random_curve = dynamics.learning_curve(
            teacher_spec, student_spec, data_dist, n_epochs
        )

        easy_first_curve = self._train_with_curriculum(
            teacher_spec, student_spec, data_dist, n_epochs, "easy_first"
        )

        hard_first_curve = self._train_with_curriculum(
            teacher_spec, student_spec, data_dist, n_epochs, "hard_first"
        )

        return {
            "random": {
                "final_error": float(random_curve["gen_errors"][-1])
                    if random_curve["gen_errors"] else float("inf"),
                "best_error": float(min(random_curve["gen_errors"]))
                    if random_curve["gen_errors"] else float("inf"),
            },
            "easy_first": {
                "final_error": float(easy_first_curve["gen_errors"][-1])
                    if easy_first_curve["gen_errors"] else float("inf"),
                "best_error": float(min(easy_first_curve["gen_errors"]))
                    if easy_first_curve["gen_errors"] else float("inf"),
            },
            "hard_first": {
                "final_error": float(hard_first_curve["gen_errors"][-1])
                    if hard_first_curve["gen_errors"] else float("inf"),
                "best_error": float(min(hard_first_curve["gen_errors"]))
                    if hard_first_curve["gen_errors"] else float("inf"),
            },
        }

    def _train_with_curriculum(
        self, teacher_spec: TeacherSpec, student_spec: StudentSpec,
        data_dist: DataDist, n_epochs: int, strategy: str
    ) -> Dict[str, List[float]]:
        """Train with curriculum strategy."""
        D = teacher_spec.input_dim
        teacher_weights = teacher_spec.generate_weights()
        activation_fn = ActivationFunctions.get(teacher_spec.activation)

        W_s = [np.random.randn(D, student_spec.hidden_dim) *
               student_spec.sigma_w / np.sqrt(D)]
        W_out = np.random.randn(student_spec.hidden_dim, 1) / np.sqrt(student_spec.hidden_dim)
        student_weights = W_s + [W_out]

        n_data = data_dist.n_samples
        X_all = np.random.randn(n_data, D) / np.sqrt(D)
        t_hidden = activation_fn(X_all @ teacher_weights[0])
        Y_all = (t_hidden @ teacher_weights[-1]).ravel()

        difficulties = np.abs(Y_all)

        if strategy == "easy_first":
            order = np.argsort(difficulties)
        elif strategy == "hard_first":
            order = np.argsort(-difficulties)
        else:
            order = np.arange(n_data)

        gen_errors = []
        lr = student_spec.learning_rate
        batch_size = 32
        eps = 1e-5

        for epoch in range(n_epochs):
            progress = epoch / n_epochs
            if strategy in ["easy_first", "hard_first"]:
                available = int(n_data * (0.2 + 0.8 * progress))
                indices = order[:available]
            else:
                indices = np.arange(n_data)

            batch_idx = np.random.choice(indices, size=min(batch_size, len(indices)))
            X_batch = X_all[batch_idx]
            y_batch = Y_all[batch_idx]

            s_hidden = activation_fn(X_batch @ student_weights[0])
            s_output = (s_hidden @ student_weights[-1]).ravel()
            error = s_output - y_batch

            grad_W_out = s_hidden.T @ error.reshape(-1, 1) / len(batch_idx)
            student_weights[-1] -= lr * grad_W_out

            pre = X_batch @ student_weights[0]
            dphi = (activation_fn(pre + eps) - activation_fn(pre - eps)) / (2 * eps)
            delta = (error.reshape(-1, 1) * student_weights[-1].T) * dphi
            grad_W1 = X_batch.T @ delta / len(batch_idx)
            student_weights[0] -= lr * grad_W1

            X_test = np.random.randn(min(1000, self.n_samples), D) / np.sqrt(D)
            t_h = activation_fn(X_test @ teacher_weights[0])
            t_out = (t_h @ teacher_weights[-1]).ravel()
            s_h = activation_fn(X_test @ student_weights[0])
            s_out = (s_h @ student_weights[-1]).ravel()
            gen_errors.append(float(np.mean((t_out - s_out) ** 2)))

        return {"gen_errors": gen_errors, "epochs": list(range(n_epochs))}


class OnlineLearningDynamics:
    """Online learning dynamics: one sample at a time."""

    def __init__(self, n_samples: int = 20000):
        self.n_samples = n_samples

    def simulate_online(
        self, teacher_spec: TeacherSpec, student_spec: StudentSpec,
        n_steps: int = 5000, lr: float = 0.01
    ) -> Dict[str, Any]:
        """Simulate online SGD learning."""
        D = teacher_spec.input_dim
        teacher_weights = teacher_spec.generate_weights()
        activation_fn = ActivationFunctions.get(teacher_spec.activation)

        W_s = np.random.randn(D, student_spec.hidden_dim) * student_spec.sigma_w / np.sqrt(D)
        W_out = np.random.randn(student_spec.hidden_dim, 1) / np.sqrt(student_spec.hidden_dim)

        gen_errors = []
        overlaps = []
        alphas = []
        eps = 1e-5

        for step in range(n_steps):
            x = np.random.randn(D) / np.sqrt(D)
            t_h = activation_fn(x @ teacher_weights[0])
            y = float(t_h @ teacher_weights[-1])

            s_h = activation_fn(x.reshape(1, -1) @ W_s)
            s_out = float(s_h @ W_out)
            error = s_out - y

            grad_W_out = s_h.T * error
            W_out -= lr * grad_W_out

            pre = x.reshape(1, -1) @ W_s
            dphi = (activation_fn(pre + eps) - activation_fn(pre - eps)) / (2 * eps)
            delta = error * W_out.T * dphi
            grad_W_s = x.reshape(-1, 1) @ delta
            W_s -= lr * grad_W_s

            if step % 50 == 0:
                X_test = np.random.randn(500, D) / np.sqrt(D)
                t_h_test = activation_fn(X_test @ teacher_weights[0])
                t_out = (t_h_test @ teacher_weights[-1]).ravel()
                s_h_test = activation_fn(X_test @ W_s)
                s_out_test = (s_h_test @ W_out).ravel()
                gen_err = float(np.mean((t_out - s_out_test) ** 2))
                gen_errors.append(gen_err)

                R = W_s.T @ teacher_weights[0] / D
                overlap = float(np.mean(np.abs(R)))
                overlaps.append(overlap)
                alphas.append(step / D)

        return {
            "gen_errors": gen_errors,
            "overlaps": overlaps,
            "alphas": alphas,
            "final_gen_error": gen_errors[-1] if gen_errors else float("inf"),
            "final_overlap": overlaps[-1] if overlaps else 0.0,
            "n_steps": n_steps,
        }

    def compute_ode_dynamics(
        self, teacher_spec: TeacherSpec, student_spec: StudentSpec,
        alpha_max: float = 10.0
    ) -> Dict[str, Any]:
        """Compute ODE dynamics for online learning (mean field)."""
        K = teacher_spec.hidden_dim
        M = student_spec.hidden_dim
        D = teacher_spec.input_dim

        n_vars = M * M + M * K + K * K
        q0 = np.zeros(n_vars)
        for i in range(M):
            q0[i * M + i] = teacher_spec.sigma_w ** 2
        for k in range(K):
            q0[M * M + M * K + k * K + k] = teacher_spec.sigma_w ** 2

        def rhs(alpha, y):
            Q = y[:M*M].reshape(M, M)
            R = y[M*M:M*M+M*K].reshape(M, K)
            T = y[M*M+M*K:].reshape(K, K)

            lr = student_spec.learning_rate

            dQ = np.zeros((M, M))
            dR = np.zeros((M, K))
            dT = np.zeros((K, K))

            for i in range(M):
                for j in range(M):
                    dQ[i, j] = lr * (R[i, :].sum() + R[j, :].sum() - 2 * Q[i, j]) * 0.1
                for k in range(K):
                    dR[i, k] = lr * (T[k, k] - R[i, k]) * 0.1

            dy = np.concatenate([dQ.ravel(), dR.ravel(), dT.ravel()])
            return dy

        try:
            sol = solve_ivp(rhs, [0, alpha_max], q0,
                            t_eval=np.linspace(0, alpha_max, 100),
                            method='RK45', max_step=0.5)

            alphas = sol.t.tolist()
            Q_trace = [float(sol.y[:M*M, i].reshape(M, M).trace()) for i in range(len(sol.t))]
            R_norm = [float(np.sqrt(np.sum(sol.y[M*M:M*M+M*K, i] ** 2))) for i in range(len(sol.t))]

            gen_errors = []
            for i in range(len(sol.t)):
                Q = sol.y[:M*M, i].reshape(M, M)
                R_mat = sol.y[M*M:M*M+M*K, i].reshape(M, K)
                T_mat = sol.y[M*M+M*K:, i].reshape(K, K)
                ge = float(np.trace(Q) - 2 * np.sum(R_mat) + np.trace(T_mat))
                gen_errors.append(max(0, ge))

            return {
                "alphas": alphas,
                "Q_trace": Q_trace,
                "R_norm": R_norm,
                "gen_errors": gen_errors,
                "converged": bool(sol.success),
            }
        except Exception as e:
            return {"error": str(e), "alphas": [], "gen_errors": []}


class PhaseTransitionDetector:
    """Detect phase transitions in learning."""

    def __init__(self, n_samples: int = 20000):
        self.n_samples = n_samples

    def detect_specialization_transition(
        self, teacher_spec: TeacherSpec, student_spec: StudentSpec,
        data_dist: DataDist, n_epochs: int = 200
    ) -> Dict[str, Any]:
        """Detect when student neurons specialize to different teacher neurons."""
        D = teacher_spec.input_dim
        teacher_weights = teacher_spec.generate_weights()
        activation_fn = ActivationFunctions.get(teacher_spec.activation)

        W_s = np.random.randn(D, student_spec.hidden_dim) * student_spec.sigma_w / np.sqrt(D)
        W_out = np.random.randn(student_spec.hidden_dim, 1) / np.sqrt(student_spec.hidden_dim)

        specialization_history = []
        symmetry_breaking = []
        lr = student_spec.learning_rate
        batch_size = 64
        eps = 1e-5

        for epoch in range(n_epochs):
            X_batch = np.random.randn(batch_size, D) / np.sqrt(D)
            t_h = activation_fn(X_batch @ teacher_weights[0])
            y = (t_h @ teacher_weights[-1]).ravel()

            s_h = activation_fn(X_batch @ W_s)
            s_out = (s_h @ W_out).ravel()
            error = s_out - y

            grad_W_out = s_h.T @ error.reshape(-1, 1) / batch_size
            W_out -= lr * grad_W_out
            pre = X_batch @ W_s
            dphi = (activation_fn(pre + eps) - activation_fn(pre - eps)) / (2 * eps)
            delta = (error.reshape(-1, 1) * W_out.T) * dphi
            grad_W_s = X_batch.T @ delta / batch_size
            W_s -= lr * grad_W_s

            R = W_s.T @ teacher_weights[0] / D
            assignment = np.argmax(np.abs(R), axis=1)
            unique_assignments = len(set(assignment.tolist()))
            specialization = unique_assignments / student_spec.hidden_dim

            Q = W_s.T @ W_s / D
            off_diag = Q - np.diag(np.diag(Q))
            symmetry = float(np.mean(np.abs(off_diag)) /
                            (np.mean(np.abs(np.diag(Q))) + 1e-12))

            specialization_history.append(float(specialization))
            symmetry_breaking.append(float(symmetry))

        transition_epoch = n_epochs
        for i in range(1, len(specialization_history)):
            if specialization_history[i] > 0.5 and specialization_history[i - 1] <= 0.5:
                transition_epoch = i
                break

        return {
            "specialization_history": specialization_history,
            "symmetry_breaking": symmetry_breaking,
            "transition_epoch": transition_epoch,
            "phase_transition_detected": bool(transition_epoch < n_epochs),
            "final_specialization": float(specialization_history[-1]),
        }

    def scan_alpha_transition(
        self, teacher_spec: TeacherSpec, student_spec: StudentSpec,
        alpha_values: List[float], n_epochs: int = 100
    ) -> Dict[str, Any]:
        """Scan over alpha = n_samples / D to find learning transition."""
        D = teacher_spec.input_dim
        activation_fn = ActivationFunctions.get(teacher_spec.activation)
        dynamics = GeneralizationErrorDynamics(self.n_samples)

        gen_errors_vs_alpha = []
        for alpha in alpha_values:
            n_samples = max(10, int(alpha * D))
            data_dist = DataDist(input_dim=D, n_samples=n_samples)
            curve = dynamics.learning_curve(
                teacher_spec, student_spec, data_dist, n_epochs=n_epochs
            )
            final_err = curve["gen_errors"][-1] if curve["gen_errors"] else float("inf")
            gen_errors_vs_alpha.append(float(final_err))

        derivative = np.gradient(gen_errors_vs_alpha, alpha_values)
        transition_idx = np.argmax(np.abs(derivative))

        return {
            "alpha_values": [float(a) for a in alpha_values],
            "gen_errors": gen_errors_vs_alpha,
            "transition_alpha": float(alpha_values[transition_idx]),
            "max_derivative": float(np.abs(derivative[transition_idx])),
        }


class InformationTheoreticBounds:
    """Information-theoretic lower bounds for teacher recovery."""

    def __init__(self):
        pass

    def mutual_information_bound(
        self, teacher_spec: TeacherSpec, data_dist: DataDist
    ) -> Dict[str, float]:
        """Compute I(W_teacher; Y | X) and derive sample complexity bound."""
        D = teacher_spec.input_dim
        K = teacher_spec.hidden_dim
        n_teacher_params = D * K + K

        entropy_prior = 0.5 * n_teacher_params * np.log(2 * np.pi * np.e * teacher_spec.sigma_w ** 2)
        bits_per_sample = 0.5 * np.log(1 + teacher_spec.sigma_w ** 2 * K /
                                        (data_dist.noise_level ** 2 + 1e-6))
        min_samples = entropy_prior / (bits_per_sample + 1e-12)

        return {
            "n_teacher_params": n_teacher_params,
            "entropy_prior": float(entropy_prior),
            "bits_per_sample": float(bits_per_sample),
            "min_samples": float(max(1, min_samples)),
            "alpha_min": float(max(1, min_samples) / D),
        }

    def statistical_dimension_bound(
        self, teacher_spec: TeacherSpec, data_dist: DataDist
    ) -> Dict[str, float]:
        """Compute statistical dimension for teacher recovery."""
        D = teacher_spec.input_dim
        K = teacher_spec.hidden_dim

        if teacher_spec.activation == "relu":
            stat_dim = D * K
        elif teacher_spec.activation in ["sign", "erf"]:
            stat_dim = D * K * np.log(D)
        else:
            stat_dim = D * K

        return {
            "statistical_dimension": float(stat_dim),
            "alpha_stat_dim": float(stat_dim / D),
            "activation_factor": float(stat_dim / (D * K)),
        }

    def bayes_optimal_error(
        self, teacher_spec: TeacherSpec, data_dist: DataDist, alpha: float
    ) -> float:
        """Estimate Bayes-optimal generalization error."""
        D = teacher_spec.input_dim
        K = teacher_spec.hidden_dim
        snr = teacher_spec.sigma_w ** 2 / (data_dist.noise_level ** 2 + 1e-6)

        if alpha * D > D * K:
            error = data_dist.noise_level ** 2 / (1 + snr * alpha)
        else:
            error = teacher_spec.sigma_w ** 2 * (1 - alpha * D / (D * K + 1e-12))
            error += data_dist.noise_level ** 2

        return float(max(0, error))


class TeacherStudent:
    """Main teacher-student analysis class."""

    def __init__(self, n_samples: int = 20000):
        self.n_samples = n_samples
        self.op_tracker = OrderParameterEvolution(n_samples)
        self.gen_dynamics = GeneralizationErrorDynamics(n_samples)
        self.overparameterization = OverparameterizationAnalyzer(n_samples)
        self.curriculum = CurriculumAnalyzer(n_samples)
        self.online = OnlineLearningDynamics(n_samples)
        self.phase_detector = PhaseTransitionDetector(n_samples)
        self.info_bounds = InformationTheoreticBounds()

    def analyze(self, teacher_spec: TeacherSpec, student_spec: StudentSpec,
                data_dist: DataDist) -> TSReport:
        """Full teacher-student analysis."""
        report = TSReport()

        report.overparameterization_ratio = float(
            student_spec.hidden_dim / teacher_spec.hidden_dim
        )

        curve = self.gen_dynamics.learning_curve(
            teacher_spec, student_spec, data_dist, n_epochs=100
        )
        report.learning_curve = curve
        report.generalization_error = curve["gen_errors"][-1] if curve["gen_errors"] else float("inf")
        report.training_error = curve["train_errors"][-1] if curve["train_errors"] else float("inf")

        if len(curve["gen_errors"]) > 10:
            last_10 = curve["gen_errors"][-10:]
            convergence_rate = abs(last_10[-1] - last_10[0]) / (abs(last_10[0]) + 1e-12)
            if convergence_rate < 0.01:
                report.convergence_time = float(len(curve["gen_errors"]) - 10)
            else:
                report.convergence_time = float(len(curve["gen_errors"]))

        info_bound = self.info_bounds.mutual_information_bound(teacher_spec, data_dist)
        report.information_lower_bound = info_bound["min_samples"]
        report.recovery_possible = data_dist.n_samples > info_bound["min_samples"]

        try:
            curriculum_result = self.curriculum.compare_curricula(
                teacher_spec, student_spec, data_dist, n_epochs=50
            )
            report.curriculum_effect = curriculum_result
        except Exception:
            report.curriculum_effect = {}

        try:
            online_result = self.online.simulate_online(
                teacher_spec, student_spec, n_steps=2000
            )
            report.online_dynamics = {
                "final_gen_error": online_result["final_gen_error"],
                "final_overlap": online_result["final_overlap"],
            }
        except Exception:
            report.online_dynamics = {}

        try:
            phase_result = self.phase_detector.detect_specialization_transition(
                teacher_spec, student_spec, data_dist, n_epochs=100
            )
            report.phase_transition_detected = phase_result["phase_transition_detected"]
            if report.phase_transition_detected:
                report.phase_transition_alpha = float(
                    phase_result["transition_epoch"] / teacher_spec.input_dim
                )
        except Exception:
            pass

        teacher_weights = teacher_spec.generate_weights()
        W_s = [np.random.randn(teacher_spec.input_dim, student_spec.hidden_dim) *
               student_spec.sigma_w / np.sqrt(teacher_spec.input_dim)]
        W_out = [np.random.randn(student_spec.hidden_dim, 1) / np.sqrt(student_spec.hidden_dim)]
        op = self.op_tracker.compute_order_parameters(teacher_weights, W_s + W_out)
        report.order_parameters = op

        return report
