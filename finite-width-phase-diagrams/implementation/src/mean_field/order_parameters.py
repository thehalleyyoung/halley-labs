"""
Mean-field order parameter computation for neural network phase diagrams.

Provides overlap, correlation, response, fixed-point iteration, multi-fixed-point
detection, stability analysis, and a top-level solver for phase diagram construction.
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import linalg as sla
from scipy import optimize as sopt
from scipy.spatial import distance as sdist
from scipy.cluster.hierarchy import fcluster, linkage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. OverlapParameter
# ---------------------------------------------------------------------------


class OverlapParameter:
    """Compute the Edwards-Anderson overlap q = (1/N) sum_i sigma_i^a sigma_i^b.

    Parameters
    ----------
    normalize : bool
        If True, divide by the system size N.
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def compute_overlap(
        self, config_a: np.ndarray, config_b: np.ndarray
    ) -> float:
        """Overlap between two replica configurations.

        Parameters
        ----------
        config_a, config_b : ndarray of shape (N,)
            Spin / activation configurations for replicas a and b.

        Returns
        -------
        q : float
            Overlap scalar.
        """
        config_a = np.asarray(config_a, dtype=np.float64)
        config_b = np.asarray(config_b, dtype=np.float64)
        if config_a.shape != config_b.shape:
            raise ValueError("Configurations must have the same shape.")
        dot = np.dot(config_a, config_b)
        if self.normalize:
            dot /= config_a.shape[0]
        return float(dot)

    def compute_self_overlap(self, config: np.ndarray) -> float:
        """Self-overlap q_EA = (1/N) sum_i sigma_i^2.

        Parameters
        ----------
        config : ndarray of shape (N,)

        Returns
        -------
        q_self : float
        """
        config = np.asarray(config, dtype=np.float64)
        val = np.dot(config, config)
        if self.normalize:
            val /= config.shape[0]
        return float(val)

    def compute_overlap_matrix(self, configs: np.ndarray) -> np.ndarray:
        """Full overlap matrix Q_{ab} for a set of configurations.

        Parameters
        ----------
        configs : ndarray of shape (n_replicas, N)

        Returns
        -------
        Q : ndarray of shape (n_replicas, n_replicas)
        """
        configs = np.asarray(configs, dtype=np.float64)
        n_replicas, N = configs.shape
        Q = configs @ configs.T
        if self.normalize:
            Q /= N
        return Q

    def thermal_average_overlap(
        self, configs: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> float:
        """Thermally averaged overlap <q> over a set of configuration pairs.

        For each distinct pair (a, b) with a < b, q_{ab} is computed and then
        averaged with optional Boltzmann weights w_a * w_b.

        Parameters
        ----------
        configs : ndarray of shape (n_replicas, N)
        weights : ndarray of shape (n_replicas,), optional
            Boltzmann weights (need not be normalised).

        Returns
        -------
        q_avg : float
        """
        configs = np.asarray(configs, dtype=np.float64)
        n_replicas, N = configs.shape
        if n_replicas < 2:
            raise ValueError("Need at least two configurations.")

        Q = self.compute_overlap_matrix(configs)

        if weights is None:
            weights = np.ones(n_replicas)
        weights = np.asarray(weights, dtype=np.float64)
        weights /= weights.sum()

        total_q = 0.0
        total_w = 0.0
        for a in range(n_replicas):
            for b in range(a + 1, n_replicas):
                w = weights[a] * weights[b]
                total_q += w * Q[a, b]
                total_w += w
        if total_w == 0.0:
            return 0.0
        return float(total_q / total_w)

    def overlap_distribution(
        self, configs: np.ndarray, bins: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Histogram P(q) of the overlap distribution.

        Parameters
        ----------
        configs : ndarray of shape (n_replicas, N)
        bins : int

        Returns
        -------
        q_vals : ndarray – bin centres
        p_vals : ndarray – normalised histogram
        """
        Q = self.compute_overlap_matrix(configs)
        n = Q.shape[0]
        overlaps = Q[np.triu_indices(n, k=1)]
        counts, edges = np.histogram(overlaps, bins=bins, density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        return centres, counts


# ---------------------------------------------------------------------------
# 2. CorrelationFunction
# ---------------------------------------------------------------------------


class CorrelationFunction:
    """Compute correlation functions C(t, t') = (1/N) sum_i <sigma_i(t) sigma_i(t')>.

    Parameters
    ----------
    normalize : bool
        Divide by N.
    subtract_mean : bool
        If True, compute connected correlations by default.
    """

    def __init__(self, normalize: bool = True, subtract_mean: bool = False):
        self.normalize = normalize
        self.subtract_mean = subtract_mean

    def equal_time_correlation(self, configs_t: np.ndarray) -> float:
        """Equal-time (auto-)correlation averaged over samples.

        Parameters
        ----------
        configs_t : ndarray of shape (n_samples, N)
            Configurations at a single time-step across samples.

        Returns
        -------
        C : float
        """
        configs_t = np.asarray(configs_t, dtype=np.float64)
        n_samples, N = configs_t.shape

        if self.subtract_mean:
            mean_config = configs_t.mean(axis=0)
            fluct = configs_t - mean_config
            C = np.mean(np.sum(fluct ** 2, axis=1))
        else:
            C = np.mean(np.sum(configs_t ** 2, axis=1))

        if self.normalize:
            C /= N
        return float(C)

    def two_time_correlation(
        self, configs_t: np.ndarray, configs_tp: np.ndarray
    ) -> float:
        """Two-time correlation C(t, t').

        Parameters
        ----------
        configs_t : ndarray of shape (n_samples, N)
        configs_tp : ndarray of shape (n_samples, N)

        Returns
        -------
        C : float
        """
        configs_t = np.asarray(configs_t, dtype=np.float64)
        configs_tp = np.asarray(configs_tp, dtype=np.float64)
        if configs_t.shape != configs_tp.shape:
            raise ValueError("Configuration arrays must have the same shape.")
        n_samples, N = configs_t.shape

        if self.subtract_mean:
            mean_t = configs_t.mean(axis=0)
            mean_tp = configs_tp.mean(axis=0)
            C = np.mean(np.sum((configs_t - mean_t) * (configs_tp - mean_tp), axis=1))
        else:
            C = np.mean(np.sum(configs_t * configs_tp, axis=1))

        if self.normalize:
            C /= N
        return float(C)

    def connected_correlation(
        self, configs_t: np.ndarray, configs_tp: np.ndarray
    ) -> float:
        """Connected (cumulant) correlation C_c = C(t,t') - <sigma(t)><sigma(t')>.

        Parameters
        ----------
        configs_t, configs_tp : ndarray of shape (n_samples, N)

        Returns
        -------
        C_conn : float
        """
        configs_t = np.asarray(configs_t, dtype=np.float64)
        configs_tp = np.asarray(configs_tp, dtype=np.float64)
        n_samples, N = configs_t.shape

        mean_prod = np.mean(np.sum(configs_t * configs_tp, axis=1))
        prod_mean = np.sum(configs_t.mean(axis=0) * configs_tp.mean(axis=0))
        C_conn = mean_prod - prod_mean
        if self.normalize:
            C_conn /= N
        return float(C_conn)

    def correlation_length_from_spatial(
        self,
        correlations: np.ndarray,
        distances: np.ndarray,
        fit_range: Optional[Tuple[float, float]] = None,
    ) -> float:
        """Extract correlation length xi from spatial decay C(r) ~ exp(-r/xi).

        A linear fit to log(|C(r)|) vs r is performed over *fit_range*.

        Parameters
        ----------
        correlations : ndarray of shape (n_distances,)
        distances : ndarray of shape (n_distances,)
        fit_range : (r_min, r_max), optional
            Restrict the fit to this distance window. Default: use all
            points where |C| > 0.

        Returns
        -------
        xi : float
            Correlation length (positive).  Returns np.inf if the fit
            slope is essentially zero.
        """
        correlations = np.asarray(correlations, dtype=np.float64)
        distances = np.asarray(distances, dtype=np.float64)

        mask = np.abs(correlations) > 1e-30
        if fit_range is not None:
            mask &= (distances >= fit_range[0]) & (distances <= fit_range[1])
        if mask.sum() < 2:
            logger.warning("Not enough points for correlation-length fit.")
            return np.inf

        log_c = np.log(np.abs(correlations[mask]))
        r = distances[mask]

        coeffs = np.polyfit(r, log_c, 1)
        slope = coeffs[0]
        if np.abs(slope) < 1e-15:
            return np.inf
        xi = -1.0 / slope
        if xi < 0:
            logger.warning("Negative correlation length; taking absolute value.")
            xi = np.abs(xi)
        return float(xi)

    def correlation_matrix(self, configs: np.ndarray) -> np.ndarray:
        """Site-resolved equal-time correlation matrix C_{ij}.

        Parameters
        ----------
        configs : ndarray of shape (n_samples, N)

        Returns
        -------
        C : ndarray of shape (N, N)
        """
        configs = np.asarray(configs, dtype=np.float64)
        n_samples, N = configs.shape
        if self.subtract_mean:
            configs = configs - configs.mean(axis=0, keepdims=True)
        C = (configs.T @ configs) / n_samples
        return C

    def autocorrelation_function(
        self, trajectory: np.ndarray, max_lag: Optional[int] = None
    ) -> np.ndarray:
        """Temporal autocorrelation from a single-site trajectory.

        Parameters
        ----------
        trajectory : ndarray of shape (T,) or (T, N)
            If 2-d, the autocorrelation is averaged over the N sites.
        max_lag : int, optional
            Maximum lag; defaults to T // 2.

        Returns
        -------
        acf : ndarray of shape (max_lag + 1,)
        """
        trajectory = np.asarray(trajectory, dtype=np.float64)
        if trajectory.ndim == 1:
            trajectory = trajectory[:, None]
        T, N = trajectory.shape
        if max_lag is None:
            max_lag = T // 2

        mean = trajectory.mean(axis=0, keepdims=True)
        fluct = trajectory - mean
        var = np.mean(fluct ** 2, axis=0)
        var[var < 1e-30] = 1.0  # avoid division by zero

        acf = np.zeros(max_lag + 1, dtype=np.float64)
        for lag in range(max_lag + 1):
            if lag == 0:
                acf[0] = 1.0
            else:
                c = np.mean(fluct[: T - lag] * fluct[lag:], axis=0) / var
                acf[lag] = np.mean(c)
        return acf


# ---------------------------------------------------------------------------
# 3. ResponseFunction
# ---------------------------------------------------------------------------


class ResponseFunction:
    """Compute linear response K_{ij} = delta <sigma_i> / delta h_j.

    Parameters
    ----------
    epsilon : float
        Step size for numerical differentiation when needed.
    """

    def __init__(self, epsilon: float = 1e-5):
        self.epsilon = epsilon

    def compute_response(
        self, magnetizations: np.ndarray, fields: np.ndarray
    ) -> np.ndarray:
        """Estimate the response matrix from paired (m, h) measurements.

        Uses least-squares: if multiple measurements are available,
        K = dM / dH is obtained via a pseudo-inverse.

        Parameters
        ----------
        magnetizations : ndarray of shape (n_measurements, N)
            Mean magnetisation vectors for different applied fields.
        fields : ndarray of shape (n_measurements, N)
            Corresponding applied-field vectors.

        Returns
        -------
        K : ndarray of shape (N, N)
            Response (susceptibility) matrix.
        """
        magnetizations = np.asarray(magnetizations, dtype=np.float64)
        fields = np.asarray(fields, dtype=np.float64)
        if magnetizations.shape != fields.shape:
            raise ValueError("magnetizations and fields must share shape.")

        # K = M @ pinv(H)  (each row is a measurement)
        K = magnetizations.T @ np.linalg.pinv(fields.T)
        return K

    def susceptibility_from_response(self, response_matrix: np.ndarray) -> float:
        """Scalar susceptibility chi = (1/N) Tr(K).

        Parameters
        ----------
        response_matrix : ndarray of shape (N, N)

        Returns
        -------
        chi : float
        """
        response_matrix = np.asarray(response_matrix, dtype=np.float64)
        N = response_matrix.shape[0]
        return float(np.trace(response_matrix) / N)

    def dynamic_response(
        self,
        magnetizations_t: np.ndarray,
        fields_t: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        """Time-dependent integrated response R(t, t_w).

        For each waiting time t_w and observation time t > t_w the
        response is estimated as delta m(t) / delta h(t_w).

        Parameters
        ----------
        magnetizations_t : ndarray of shape (T, N)
            Magnetisation trajectory.
        fields_t : ndarray of shape (T, N)
            Applied field trajectory.
        times : ndarray of shape (T,)
            Time stamps.

        Returns
        -------
        R : ndarray of shape (T, T)
            R[t, tw] = integrated response at time t to a field at tw.
        """
        magnetizations_t = np.asarray(magnetizations_t, dtype=np.float64)
        fields_t = np.asarray(fields_t, dtype=np.float64)
        times = np.asarray(times, dtype=np.float64)
        T, N = magnetizations_t.shape

        R = np.zeros((T, T), dtype=np.float64)
        for tw in range(T):
            h_norm = np.linalg.norm(fields_t[tw])
            if h_norm < 1e-30:
                continue
            for t in range(tw, T):
                dm = magnetizations_t[t] - magnetizations_t[tw]
                R[t, tw] = np.dot(dm, fields_t[tw]) / (N * h_norm ** 2)
        return R

    def fluctuation_dissipation_ratio(
        self,
        correlation: np.ndarray,
        response: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        """Fluctuation-dissipation ratio X = T R / (dC/dt).

        Parameters
        ----------
        correlation : ndarray of shape (T,)
        response : ndarray of shape (T,)
        temperature : float

        Returns
        -------
        X : ndarray of shape (T - 1,)
        """
        correlation = np.asarray(correlation, dtype=np.float64)
        response = np.asarray(response, dtype=np.float64)
        dC = np.diff(correlation)
        mask = np.abs(dC) > 1e-30
        X = np.full(len(dC), np.nan)
        R_mid = 0.5 * (response[:-1] + response[1:])
        X[mask] = temperature * R_mid[mask] / dC[mask]
        return X

    def numerical_response(
        self,
        model_fn: Callable[[np.ndarray], np.ndarray],
        field: np.ndarray,
    ) -> np.ndarray:
        """Numerical response via central differences.

        Parameters
        ----------
        model_fn : callable
            Maps field vector h -> magnetisation vector m.
        field : ndarray of shape (N,)

        Returns
        -------
        K : ndarray of shape (N, N)
        """
        field = np.asarray(field, dtype=np.float64)
        N = field.shape[0]
        K = np.zeros((N, N), dtype=np.float64)
        for j in range(N):
            h_plus = field.copy()
            h_minus = field.copy()
            h_plus[j] += self.epsilon
            h_minus[j] -= self.epsilon
            m_plus = model_fn(h_plus)
            m_minus = model_fn(h_minus)
            K[:, j] = (m_plus - m_minus) / (2.0 * self.epsilon)
        return K


# ---------------------------------------------------------------------------
# 4. FixedPointIterator
# ---------------------------------------------------------------------------


class FixedPointIterator:
    """Self-consistent fixed-point iteration for mean-field equations.

    Solves x = F(x) by repeated application of the map F.

    Parameters
    ----------
    max_iter : int
        Default maximum iterations.
    tol : float
        Default convergence tolerance on ||x_{k+1} - x_k||.
    verbose : bool
        Log progress every *log_every* steps.
    log_every : int
        Logging interval.
    """

    def __init__(
        self,
        max_iter: int = 10000,
        tol: float = 1e-10,
        verbose: bool = False,
        log_every: int = 500,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.log_every = log_every

    def iterate(
        self,
        initial_params: np.ndarray,
        equations: Callable[[np.ndarray], np.ndarray],
        max_iter: Optional[int] = None,
        tol: Optional[float] = None,
    ) -> Dict:
        """Plain fixed-point iteration x <- F(x).

        Parameters
        ----------
        initial_params : ndarray
        equations : callable  x -> F(x)
        max_iter : int, optional
        tol : float, optional

        Returns
        -------
        result : dict with keys
            'solution', 'converged', 'iterations', 'residual', 'history'
        """
        max_iter = max_iter or self.max_iter
        tol = tol or self.tol
        x = np.asarray(initial_params, dtype=np.float64).copy()
        history: List[np.ndarray] = [x.copy()]

        converged = False
        residual = np.inf
        for k in range(1, max_iter + 1):
            x_new = np.asarray(equations(x), dtype=np.float64)
            residual = float(np.linalg.norm(x_new - x))
            x = x_new
            history.append(x.copy())
            if self.verbose and k % self.log_every == 0:
                logger.info("iter %d  residual=%.3e", k, residual)
            if residual < tol:
                converged = True
                break

        return {
            "solution": x,
            "converged": converged,
            "iterations": k if converged else max_iter,
            "residual": residual,
            "history": np.array(history),
        }

    def damped_iteration(
        self,
        initial: np.ndarray,
        equations: Callable[[np.ndarray], np.ndarray],
        damping: float = 0.5,
        max_iter: Optional[int] = None,
        tol: Optional[float] = None,
    ) -> Dict:
        """Damped (under-relaxed) iteration x <- (1-alpha)*x + alpha*F(x).

        Parameters
        ----------
        initial : ndarray
        equations : callable
        damping : float in (0, 1]
        max_iter, tol : optional overrides

        Returns
        -------
        result : dict
        """
        if not 0.0 < damping <= 1.0:
            raise ValueError("damping must be in (0, 1].")
        max_iter = max_iter or self.max_iter
        tol = tol or self.tol
        x = np.asarray(initial, dtype=np.float64).copy()
        history: List[np.ndarray] = [x.copy()]

        converged = False
        residual = np.inf
        for k in range(1, max_iter + 1):
            x_new = np.asarray(equations(x), dtype=np.float64)
            x_mixed = (1.0 - damping) * x + damping * x_new
            residual = float(np.linalg.norm(x_mixed - x))
            x = x_mixed
            history.append(x.copy())
            if self.verbose and k % self.log_every == 0:
                logger.info("damped iter %d  residual=%.3e", k, residual)
            if residual < tol:
                converged = True
                break

        return {
            "solution": x,
            "converged": converged,
            "iterations": k if converged else max_iter,
            "residual": residual,
            "history": np.array(history),
        }

    def anderson_mixing(
        self,
        initial: np.ndarray,
        equations: Callable[[np.ndarray], np.ndarray],
        m: int = 5,
        damping: float = 0.5,
        max_iter: Optional[int] = None,
        tol: Optional[float] = None,
    ) -> Dict:
        """Anderson mixing (type-I) acceleration for fixed-point iteration.

        Maintains a window of *m* previous iterates and residuals and
        determines an optimal linear combination to reduce the residual.

        Parameters
        ----------
        initial : ndarray
        equations : callable
        m : int
            Mixing history depth.
        damping : float
            Mixing parameter (0, 1].
        max_iter, tol : optional

        Returns
        -------
        result : dict
        """
        max_iter = max_iter or self.max_iter
        tol = tol or self.tol
        x = np.asarray(initial, dtype=np.float64).copy()
        dim = x.size

        X_hist: List[np.ndarray] = []
        R_hist: List[np.ndarray] = []
        history: List[np.ndarray] = [x.copy()]

        converged = False
        residual = np.inf
        for k in range(1, max_iter + 1):
            Fx = np.asarray(equations(x), dtype=np.float64)
            res = Fx - x
            residual = float(np.linalg.norm(res))

            if residual < tol:
                converged = True
                x = Fx
                history.append(x.copy())
                break

            X_hist.append(x.copy())
            R_hist.append(res.copy())

            mk = min(len(X_hist), m)
            if mk < 2:
                x = x + damping * res
            else:
                # Build the matrix of residual differences
                dR = np.zeros((dim, mk - 1), dtype=np.float64)
                dX = np.zeros((dim, mk - 1), dtype=np.float64)
                for j in range(mk - 1):
                    dR[:, j] = R_hist[-1] - R_hist[-(j + 2)]
                    dX[:, j] = X_hist[-1] - X_hist[-(j + 2)]

                # Solve least-squares: min || res - dR @ gamma ||
                gamma, _, _, _ = np.linalg.lstsq(dR, res, rcond=None)
                x_bar = x - dX @ gamma
                r_bar = res - dR @ gamma
                x = x_bar + damping * r_bar

            history.append(x.copy())
            if self.verbose and k % self.log_every == 0:
                logger.info("anderson iter %d  residual=%.3e", k, residual)

            # Trim history
            if len(X_hist) > m:
                X_hist.pop(0)
                R_hist.pop(0)

        return {
            "solution": x,
            "converged": converged,
            "iterations": k if converged else max_iter,
            "residual": residual,
            "history": np.array(history),
        }

    @staticmethod
    def detect_convergence(
        history: np.ndarray, window: int = 50, tol: float = 1e-10
    ) -> Dict:
        """Analyse convergence from an iteration history.

        Parameters
        ----------
        history : ndarray of shape (n_iters, dim)
        window : int
            Number of trailing iterates to inspect.
        tol : float

        Returns
        -------
        info : dict
            'converged' : bool
            'rate' : float – estimated convergence rate (ratio of successive residuals)
            'final_residual' : float
            'oscillating' : bool – True if sign changes suggest oscillation
        """
        history = np.asarray(history, dtype=np.float64)
        if history.ndim == 1:
            history = history[:, None]
        n = history.shape[0]
        if n < 3:
            return {
                "converged": False,
                "rate": np.nan,
                "final_residual": np.nan,
                "oscillating": False,
            }

        tail = history[-window:]
        diffs = np.linalg.norm(np.diff(tail, axis=0), axis=1)
        final_residual = float(diffs[-1])

        # Convergence rate: geometric mean of successive residual ratios
        ratios = []
        for i in range(1, len(diffs)):
            if diffs[i - 1] > 1e-30:
                ratios.append(diffs[i] / diffs[i - 1])
        rate = float(np.median(ratios)) if ratios else np.nan

        # Oscillation detection: check if the direction of change alternates
        if len(tail) >= 4:
            directions = np.diff(tail, axis=0)
            dot_products = np.array(
                [
                    np.dot(directions[i], directions[i + 1])
                    for i in range(len(directions) - 1)
                ]
            )
            frac_negative = np.mean(dot_products < 0)
            oscillating = bool(frac_negative > 0.4)
        else:
            oscillating = False

        return {
            "converged": final_residual < tol,
            "rate": rate,
            "final_residual": final_residual,
            "oscillating": oscillating,
        }


# ---------------------------------------------------------------------------
# 5. MultiFixedPointDetector
# ---------------------------------------------------------------------------


class MultiFixedPointDetector:
    """Find multiple fixed points by scanning initial conditions.

    Parameters
    ----------
    iterator : FixedPointIterator
        Iterator used to converge each trial.
    tol_cluster : float
        Tolerance for clustering duplicate solutions.
    """

    def __init__(
        self,
        iterator: Optional[FixedPointIterator] = None,
        tol_cluster: float = 1e-6,
    ):
        self.iterator = iterator or FixedPointIterator()
        self.tol_cluster = tol_cluster

    def scan_initial_conditions(
        self,
        equations: Callable[[np.ndarray], np.ndarray],
        param_ranges: List[Tuple[float, float]],
        n_samples: int = 100,
        method: str = "damped",
        damping: float = 0.5,
        seed: Optional[int] = None,
    ) -> Dict:
        """Run fixed-point iteration from many random initial conditions.

        Parameters
        ----------
        equations : callable
        param_ranges : list of (lo, hi) for each dimension
        n_samples : int
        method : str – 'plain', 'damped', or 'anderson'
        damping : float
        seed : int, optional

        Returns
        -------
        result : dict
            'solutions' : ndarray of unique fixed points
            'all_solutions' : list of dicts from iterator
            'n_converged' : int
        """
        rng = np.random.default_rng(seed)
        dim = len(param_ranges)
        lo = np.array([r[0] for r in param_ranges])
        hi = np.array([r[1] for r in param_ranges])

        all_solutions: List[Dict] = []
        converged_pts: List[np.ndarray] = []

        for _ in range(n_samples):
            x0 = rng.uniform(lo, hi)
            if method == "anderson":
                res = self.iterator.anderson_mixing(x0, equations, damping=damping)
            elif method == "damped":
                res = self.iterator.damped_iteration(x0, equations, damping=damping)
            else:
                res = self.iterator.iterate(x0, equations)
            all_solutions.append(res)
            if res["converged"]:
                converged_pts.append(res["solution"])

        if converged_pts:
            unique = self.cluster_fixed_points(np.array(converged_pts))
        else:
            unique = np.empty((0, dim))

        return {
            "solutions": unique,
            "all_solutions": all_solutions,
            "n_converged": len(converged_pts),
        }

    def cluster_fixed_points(
        self,
        solutions: np.ndarray,
        tol: Optional[float] = None,
    ) -> np.ndarray:
        """Cluster nearby solutions and return unique representatives.

        Parameters
        ----------
        solutions : ndarray of shape (n, dim)
        tol : float, optional

        Returns
        -------
        unique : ndarray of shape (n_unique, dim)
        """
        tol = tol if tol is not None else self.tol_cluster
        solutions = np.asarray(solutions, dtype=np.float64)
        if solutions.ndim == 1:
            solutions = solutions[:, None]
        n = solutions.shape[0]
        if n == 0:
            return solutions
        if n == 1:
            return solutions.copy()

        dist_matrix = sdist.cdist(solutions, solutions)
        Z = linkage(sdist.squareform(dist_matrix + dist_matrix.T, checks=False), method="complete")
        labels = fcluster(Z, t=tol, criterion="distance")

        unique = []
        for lab in np.unique(labels):
            members = solutions[labels == lab]
            unique.append(members.mean(axis=0))
        return np.array(unique)

    def basin_of_attraction(
        self,
        equations: Callable[[np.ndarray], np.ndarray],
        fixed_points: np.ndarray,
        grid_points: np.ndarray,
        method: str = "damped",
        damping: float = 0.5,
    ) -> np.ndarray:
        """Determine basin of attraction on a grid.

        For each grid point, iterate to convergence and assign to the
        nearest fixed point.

        Parameters
        ----------
        equations : callable
        fixed_points : ndarray of shape (n_fp, dim)
        grid_points : ndarray of shape (n_grid, dim)
        method : str
        damping : float

        Returns
        -------
        labels : ndarray of shape (n_grid,)
            Index into fixed_points for each grid point, or -1 if not
            converged.
        """
        fixed_points = np.asarray(fixed_points, dtype=np.float64)
        grid_points = np.asarray(grid_points, dtype=np.float64)
        n_grid = grid_points.shape[0]
        labels = np.full(n_grid, -1, dtype=int)

        for i in range(n_grid):
            if method == "anderson":
                res = self.iterator.anderson_mixing(grid_points[i], equations, damping=damping)
            elif method == "damped":
                res = self.iterator.damped_iteration(grid_points[i], equations, damping=damping)
            else:
                res = self.iterator.iterate(grid_points[i], equations)

            if res["converged"]:
                dists = np.linalg.norm(fixed_points - res["solution"], axis=1)
                labels[i] = int(np.argmin(dists))

        return labels


# ---------------------------------------------------------------------------
# 6. FixedPointStabilityAnalyzer
# ---------------------------------------------------------------------------


class FixedPointStabilityAnalyzer:
    """Stability analysis of fixed points of the map x -> F(x).

    Parameters
    ----------
    epsilon : float
        Step size for numerical Jacobian.
    """

    def __init__(self, epsilon: float = 1e-7):
        self.epsilon = epsilon

    def compute_jacobian(
        self,
        equations: Callable[[np.ndarray], np.ndarray],
        fixed_point: np.ndarray,
        epsilon: Optional[float] = None,
    ) -> np.ndarray:
        """Numerical Jacobian J_{ij} = dF_i / dx_j via central differences.

        Parameters
        ----------
        equations : callable
        fixed_point : ndarray of shape (dim,)
        epsilon : float, optional

        Returns
        -------
        J : ndarray of shape (dim, dim)
        """
        eps = epsilon if epsilon is not None else self.epsilon
        x = np.asarray(fixed_point, dtype=np.float64)
        dim = x.size
        J = np.zeros((dim, dim), dtype=np.float64)
        for j in range(dim):
            x_p = x.copy()
            x_m = x.copy()
            x_p[j] += eps
            x_m[j] -= eps
            F_p = np.asarray(equations(x_p), dtype=np.float64)
            F_m = np.asarray(equations(x_m), dtype=np.float64)
            J[:, j] = (F_p - F_m) / (2.0 * eps)
        return J

    def eigenvalue_analysis(
        self, jacobian: np.ndarray
    ) -> Dict:
        """Eigenvalue decomposition of the Jacobian.

        Parameters
        ----------
        jacobian : ndarray of shape (dim, dim)

        Returns
        -------
        result : dict
            'eigenvalues' : complex ndarray
            'eigenvectors' : complex ndarray (columns)
            'spectral_radius' : float
            'max_real_part' : float
        """
        eigenvalues, eigenvectors = np.linalg.eig(jacobian)
        spectral_radius = float(np.max(np.abs(eigenvalues)))
        max_real = float(np.max(eigenvalues.real))
        return {
            "eigenvalues": eigenvalues,
            "eigenvectors": eigenvectors,
            "spectral_radius": spectral_radius,
            "max_real_part": max_real,
        }

    def classify_fixed_point(
        self, eigenvalues: np.ndarray
    ) -> Dict:
        """Classify a fixed point from the eigenvalues of the Jacobian of the
        *map* x -> F(x).

        For a discrete map, the fixed point is stable if all |lambda| < 1.

        Parameters
        ----------
        eigenvalues : complex ndarray

        Returns
        -------
        info : dict
            'stable' : bool
            'classification' : str
            'n_unstable' : int
            'n_marginal' : int
            'spectral_radius' : float
        """
        eigenvalues = np.asarray(eigenvalues)
        abs_eig = np.abs(eigenvalues)
        spectral_radius = float(np.max(abs_eig))

        marginal_tol = 1e-8
        n_unstable = int(np.sum(abs_eig > 1.0 + marginal_tol))
        n_marginal = int(np.sum(np.abs(abs_eig - 1.0) <= marginal_tol))

        stable = n_unstable == 0 and n_marginal == 0

        # Sub-classify based on eigenvalue structure
        has_complex = np.any(np.abs(eigenvalues.imag) > marginal_tol)
        all_real_neg = np.all(eigenvalues.real < 0) and not has_complex

        if stable and not has_complex:
            classification = "stable_node"
        elif stable and has_complex:
            classification = "stable_spiral"
        elif n_marginal > 0 and n_unstable == 0:
            classification = "marginally_stable"
        elif n_unstable > 0 and not has_complex:
            classification = "unstable_node"
        elif n_unstable > 0 and has_complex:
            classification = "unstable_spiral"
        else:
            classification = "saddle"

        return {
            "stable": stable,
            "classification": classification,
            "n_unstable": n_unstable,
            "n_marginal": n_marginal,
            "spectral_radius": spectral_radius,
        }

    def lyapunov_exponents(
        self,
        equations: Callable[[np.ndarray], np.ndarray],
        fixed_point: np.ndarray,
        trajectory_length: int = 1000,
        perturbation: float = 1e-8,
    ) -> np.ndarray:
        """Estimate Lyapunov exponents by tracking perturbation growth along
        a trajectory starting near the fixed point.

        Uses QR-based reorthonormalisation every step.

        Parameters
        ----------
        equations : callable
        fixed_point : ndarray of shape (dim,)
        trajectory_length : int
        perturbation : float

        Returns
        -------
        lyap : ndarray of shape (dim,)
            Lyapunov exponents sorted descending.
        """
        x = np.asarray(fixed_point, dtype=np.float64).copy()
        dim = x.size
        # Start with a small kick so that we are not exactly at FP
        rng = np.random.default_rng(42)
        x = x + perturbation * rng.standard_normal(dim)

        Q = np.eye(dim, dtype=np.float64)
        lyap_sum = np.zeros(dim, dtype=np.float64)

        for t in range(trajectory_length):
            # Propagate the point
            x_new = np.asarray(equations(x), dtype=np.float64)

            # Jacobian at current point
            J = self.compute_jacobian(equations, x)

            # Propagate perturbation vectors
            Z = J @ Q
            Q_new, R = np.linalg.qr(Z)

            # Accumulate log of diagonal of R
            diag_R = np.abs(np.diag(R))
            diag_R[diag_R < 1e-300] = 1e-300
            lyap_sum += np.log(diag_R)

            Q = Q_new
            x = x_new

        lyap = lyap_sum / trajectory_length
        return np.sort(lyap)[::-1]

    def full_stability_report(
        self,
        equations: Callable[[np.ndarray], np.ndarray],
        fixed_point: np.ndarray,
    ) -> Dict:
        """Combined report: Jacobian, eigenvalues, classification, Lyapunov exponents.

        Parameters
        ----------
        equations : callable
        fixed_point : ndarray

        Returns
        -------
        report : dict
        """
        J = self.compute_jacobian(equations, fixed_point)
        eig_info = self.eigenvalue_analysis(J)
        classification = self.classify_fixed_point(eig_info["eigenvalues"])
        lyap = self.lyapunov_exponents(equations, fixed_point)
        return {
            "jacobian": J,
            **eig_info,
            **classification,
            "lyapunov_exponents": lyap,
        }


# ---------------------------------------------------------------------------
# 7. OrderParameterSolver
# ---------------------------------------------------------------------------


class OrderParameterSolver:
    """Orchestrate mean-field order-parameter computation.

    Combines fixed-point iteration, multi-solution detection, and
    stability analysis to construct phase diagrams.

    Parameters
    ----------
    iterator : FixedPointIterator, optional
    detector : MultiFixedPointDetector, optional
    stability : FixedPointStabilityAnalyzer, optional
    overlap : OverlapParameter, optional
    correlation : CorrelationFunction, optional
    response : ResponseFunction, optional
    """

    def __init__(
        self,
        iterator: Optional[FixedPointIterator] = None,
        detector: Optional[MultiFixedPointDetector] = None,
        stability: Optional[FixedPointStabilityAnalyzer] = None,
        overlap: Optional[OverlapParameter] = None,
        correlation: Optional[CorrelationFunction] = None,
        response: Optional[ResponseFunction] = None,
    ):
        self.iterator = iterator or FixedPointIterator()
        self.detector = detector or MultiFixedPointDetector(iterator=self.iterator)
        self.stability = stability or FixedPointStabilityAnalyzer()
        self.overlap = overlap or OverlapParameter()
        self.correlation = correlation or CorrelationFunction()
        self.response = response or ResponseFunction()

    def solve_mean_field(
        self,
        equations: Callable[[np.ndarray], np.ndarray],
        params: np.ndarray,
        method: str = "anderson",
        damping: float = 0.5,
    ) -> Dict:
        """Solve a single mean-field self-consistency equation.

        Tries Anderson mixing first; falls back to damped iteration if
        Anderson does not converge.

        Parameters
        ----------
        equations : callable  x -> F(x)
        params : ndarray – initial guess
        method : str
        damping : float

        Returns
        -------
        result : dict
            Includes 'solution', 'converged', 'stability', 'iterations'.
        """
        params = np.asarray(params, dtype=np.float64)

        if method == "anderson":
            res = self.iterator.anderson_mixing(params, equations, damping=damping)
            if not res["converged"]:
                logger.info("Anderson mixing did not converge; falling back to damped iteration.")
                res = self.iterator.damped_iteration(params, equations, damping=damping)
        elif method == "damped":
            res = self.iterator.damped_iteration(params, equations, damping=damping)
        else:
            res = self.iterator.iterate(params, equations)

        if res["converged"]:
            stab = self.stability.full_stability_report(equations, res["solution"])
            res["stability"] = stab
        else:
            res["stability"] = None

        return res

    def phase_diagram_scan(
        self,
        equations_factory: Callable[..., Callable[[np.ndarray], np.ndarray]],
        param1_range: np.ndarray,
        param2_range: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        scan_n_samples: int = 20,
        param_dim: int = 1,
    ) -> Dict:
        """Scan a 2-d parameter space and collect fixed points + stability.

        Parameters
        ----------
        equations_factory : callable
            (p1, p2) -> equations callable.  Given external parameters,
            returns the self-consistency map F.
        param1_range : ndarray
        param2_range : ndarray
        initial_guess : ndarray, optional
            If None, use the detector to scan initial conditions.
        scan_n_samples : int
            Number of initial-condition samples when using the detector.
        param_dim : int
            Dimension of the order parameter vector.

        Returns
        -------
        diagram : dict
            'param1' : ndarray
            'param2' : ndarray
            'solutions' : 2-d object array of solution lists
            'n_solutions' : int ndarray
            'stability' : 2-d object array
        """
        p1 = np.asarray(param1_range, dtype=np.float64)
        p2 = np.asarray(param2_range, dtype=np.float64)
        n1, n2 = len(p1), len(p2)

        sol_grid = np.empty((n1, n2), dtype=object)
        stab_grid = np.empty((n1, n2), dtype=object)
        n_sol_grid = np.zeros((n1, n2), dtype=int)

        for i, v1 in enumerate(p1):
            for j, v2 in enumerate(p2):
                eqs = equations_factory(v1, v2)
                if initial_guess is not None:
                    res = self.solve_mean_field(eqs, initial_guess)
                    solutions = [res["solution"]] if res["converged"] else []
                    stabilities = [res["stability"]] if res["converged"] else []
                else:
                    ranges = [(-2.0, 2.0)] * param_dim
                    det_res = self.detector.scan_initial_conditions(
                        eqs, ranges, n_samples=scan_n_samples
                    )
                    solutions = list(det_res["solutions"])
                    stabilities = []
                    for sol in solutions:
                        stab = self.stability.full_stability_report(eqs, sol)
                        stabilities.append(stab)

                sol_grid[i, j] = solutions
                stab_grid[i, j] = stabilities
                n_sol_grid[i, j] = len(solutions)

            logger.info("phase_diagram_scan: row %d/%d done", i + 1, n1)

        return {
            "param1": p1,
            "param2": p2,
            "solutions": sol_grid,
            "n_solutions": n_sol_grid,
            "stability": stab_grid,
        }

    def detect_phase_transitions(
        self,
        solutions: Dict,
        order_param_index: int = 0,
    ) -> Dict:
        """Detect phase transitions from a phase-diagram scan.

        Looks for:
        * Changes in the number of stable solutions (first-order-like).
        * Divergence of susceptibility / vanishing of stability eigenvalue
          (second-order-like).

        Parameters
        ----------
        solutions : dict
            Output of :meth:`phase_diagram_scan`.
        order_param_index : int
            Which component of the order-parameter vector to track.

        Returns
        -------
        transitions : dict
            'first_order_lines' : list of (i, j) indices where n_solutions changes
            'second_order_lines' : list of (i, j) where spectral radius crosses 1
            'order_parameter_field' : ndarray of shape (n1, n2)
            'susceptibility_field' : ndarray of shape (n1, n2)
        """
        sol_grid = solutions["solutions"]
        stab_grid = solutions["stability"]
        n1, n2 = solutions["n_solutions"].shape

        op_field = np.full((n1, n2), np.nan)
        chi_field = np.full((n1, n2), np.nan)

        for i in range(n1):
            for j in range(n2):
                sols = sol_grid[i, j]
                stabs = stab_grid[i, j]
                if sols is not None and len(sols) > 0:
                    # Pick the first stable solution, or the first solution
                    chosen_idx = 0
                    if stabs:
                        for idx, st in enumerate(stabs):
                            if st is not None and st.get("stable", False):
                                chosen_idx = idx
                                break
                    sol = np.asarray(sols[chosen_idx])
                    if sol.size > order_param_index:
                        op_field[i, j] = sol[order_param_index]
                    else:
                        op_field[i, j] = float(sol.flat[0])

                    # Susceptibility ~ 1 / (1 - spectral_radius)
                    if stabs and stabs[chosen_idx] is not None:
                        sr = stabs[chosen_idx].get("spectral_radius", 0.0)
                        denom = np.abs(1.0 - sr)
                        chi_field[i, j] = 1.0 / max(denom, 1e-15)

        # First-order: changes in number of solutions
        n_sol = solutions["n_solutions"]
        first_order = []
        for i in range(n1):
            for j in range(n2):
                if i > 0 and n_sol[i, j] != n_sol[i - 1, j]:
                    first_order.append((i, j))
                if j > 0 and n_sol[i, j] != n_sol[i, j - 1]:
                    first_order.append((i, j))

        # Second-order: spectral radius crossing 1
        second_order = []
        for i in range(n1):
            for j in range(n2):
                stabs = stab_grid[i, j]
                if not stabs:
                    continue
                for st in stabs:
                    if st is not None:
                        sr = st.get("spectral_radius", 0.0)
                        if np.abs(sr - 1.0) < 0.05:
                            second_order.append((i, j))
                            break

        return {
            "first_order_lines": first_order,
            "second_order_lines": second_order,
            "order_parameter_field": op_field,
            "susceptibility_field": chi_field,
        }

    def compute_order_parameters_from_configs(
        self,
        configs: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> Dict:
        """Convenience: compute all standard order parameters from raw configs.

        Parameters
        ----------
        configs : ndarray of shape (n_replicas, N)
        weights : ndarray of shape (n_replicas,), optional

        Returns
        -------
        ops : dict
        """
        configs = np.asarray(configs, dtype=np.float64)
        Q = self.overlap.compute_overlap_matrix(configs)
        q_avg = self.overlap.thermal_average_overlap(configs, weights)

        corr_eq = self.correlation.equal_time_correlation(configs)
        C_mat = self.correlation.correlation_matrix(configs)

        # Magnetisation and susceptibility
        m = configs.mean(axis=0)
        m_avg = float(np.mean(m))
        m2 = float(np.mean(m ** 2))
        chi = configs.shape[1] * (m2 - m_avg ** 2)

        return {
            "overlap_matrix": Q,
            "mean_overlap": q_avg,
            "equal_time_correlation": corr_eq,
            "correlation_matrix": C_mat,
            "magnetisation": m,
            "mean_magnetisation": m_avg,
            "susceptibility": chi,
        }

    def free_energy_landscape(
        self,
        equations: Callable[[np.ndarray], np.ndarray],
        grid_1d: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Approximate free-energy landscape F(q) by integrating the
        self-consistency residual along a 1-d order-parameter axis.

        F(q) = - integral_0^q [F(q') - q'] dq'

        Parameters
        ----------
        equations : callable
        grid_1d : ndarray of shape (n_points,)

        Returns
        -------
        q_vals : ndarray
        F_vals : ndarray
        """
        grid_1d = np.asarray(grid_1d, dtype=np.float64)
        residuals = np.array(
            [float(np.asarray(equations(np.atleast_1d(q)))[0] - q) for q in grid_1d]
        )
        F_vals = -np.cumsum(residuals) * np.gradient(grid_1d)
        F_vals -= F_vals.min()
        return grid_1d, F_vals
