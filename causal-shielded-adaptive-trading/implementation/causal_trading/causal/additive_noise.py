"""
Additive Noise Models (ANM) for bivariate causal direction inference.

Implements the approach of Hoyer et al. (2009): fit Y = f(X) + N and test
whether the residuals N are independent of X.  If independence holds in one
direction but not the other the causal direction is identifiable.

Provides multiple regression backends (linear, polynomial, Gaussian-process /
kernel-ridge) and independence tests via HSIC.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.linalg import cho_solve, cho_factor, solve_triangular


# ---------------------------------------------------------------------------
# Kernel helpers (used for kernel ridge regression)
# ---------------------------------------------------------------------------

def _rbf_kernel(
    X: NDArray, Y: Optional[NDArray] = None, bandwidth: float = 1.0
) -> NDArray:
    """Compute Gaussian RBF kernel matrix."""
    if Y is None:
        Y = X
    sq_dists = cdist(X.reshape(-1, 1), Y.reshape(-1, 1), "sqeuclidean")
    return np.exp(-sq_dists / (2.0 * bandwidth ** 2))


def _polynomial_kernel(
    X: NDArray,
    Y: Optional[NDArray] = None,
    degree: int = 3,
    coef0: float = 1.0,
) -> NDArray:
    if Y is None:
        Y = X
    return (X.reshape(-1, 1) @ Y.reshape(1, -1) + coef0) ** degree


def _median_bandwidth(X: NDArray) -> float:
    """Median heuristic for Gaussian kernel bandwidth."""
    dists = cdist(X.reshape(-1, 1), X.reshape(-1, 1), "sqeuclidean")
    med = np.median(dists[np.triu_indices_from(dists, k=1)])
    return float(np.sqrt(med / 2.0)) if med > 0 else 1.0


# ---------------------------------------------------------------------------
# Regression model types
# ---------------------------------------------------------------------------

class RegressionType(Enum):
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    KRR = "kernel_ridge"
    GP = "gp"


@dataclass
class RegressionResult:
    """Container for a fitted regression."""
    reg_type: RegressionType
    predict: Callable[[NDArray], NDArray]
    residuals: NDArray
    params: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0  # R² or log-likelihood


# ---------------------------------------------------------------------------
# Regression backends
# ---------------------------------------------------------------------------

def _fit_linear(X: NDArray, Y: NDArray) -> RegressionResult:
    """OLS regression Y = a*X + b."""
    A = np.column_stack([X, np.ones_like(X)])
    coeffs, res, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    pred = A @ coeffs
    residuals = Y - pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Y - Y.mean()) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    def predict(x: NDArray) -> NDArray:
        return np.column_stack([x, np.ones_like(x)]) @ coeffs

    return RegressionResult(
        reg_type=RegressionType.LINEAR,
        predict=predict,
        residuals=residuals,
        params={"coeffs": coeffs},
        score=r2,
    )


def _fit_polynomial(
    X: NDArray, Y: NDArray, degree: int = 3
) -> RegressionResult:
    """Polynomial regression Y = sum_d a_d * X^d."""
    coeffs = np.polyfit(X, Y, degree)
    poly = np.poly1d(coeffs)
    pred = poly(X)
    residuals = Y - pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Y - Y.mean()) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    def predict(x: NDArray) -> NDArray:
        return poly(x)

    return RegressionResult(
        reg_type=RegressionType.POLYNOMIAL,
        predict=predict,
        residuals=residuals,
        params={"coeffs": coeffs, "degree": degree},
        score=r2,
    )


def _fit_kernel_ridge(
    X: NDArray,
    Y: NDArray,
    bandwidth: Optional[float] = None,
    reg_lambda: float = 0.1,
) -> RegressionResult:
    """Kernel ridge regression with Gaussian kernel."""
    if bandwidth is None:
        bandwidth = _median_bandwidth(X)
    K = _rbf_kernel(X, bandwidth=bandwidth)
    n = len(X)
    alpha = np.linalg.solve(K + reg_lambda * np.eye(n), Y)
    pred = K @ alpha
    residuals = Y - pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Y - Y.mean()) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    def predict(x: NDArray, _X=X, _alpha=alpha, _bw=bandwidth) -> NDArray:
        Ks = _rbf_kernel(x, _X, bandwidth=_bw)
        return Ks @ _alpha

    return RegressionResult(
        reg_type=RegressionType.KRR,
        predict=predict,
        residuals=residuals,
        params={"bandwidth": bandwidth, "lambda": reg_lambda},
        score=r2,
    )


def _fit_gp(
    X: NDArray,
    Y: NDArray,
    bandwidth: Optional[float] = None,
    noise_var: float = 0.1,
) -> RegressionResult:
    """Gaussian process regression (mean prediction) with RBF kernel.

    Computes leave-one-out cross-validation residuals to avoid overfitting.
    """
    if bandwidth is None:
        bandwidth = _median_bandwidth(X)
    n = len(X)
    K = _rbf_kernel(X, bandwidth=bandwidth)
    K_noisy = K + noise_var * np.eye(n)

    L = cho_factor(K_noisy)
    alpha = cho_solve(L, Y)
    pred = K @ alpha

    # LOO residuals: r_i = (y_i - f_{-i}(x_i)) = alpha_i / K_inv_ii
    K_inv = cho_solve(L, np.eye(n))
    K_inv_diag = np.diag(K_inv)
    K_inv_diag = np.maximum(K_inv_diag, 1e-10)
    residuals = alpha / K_inv_diag

    # Marginal log-likelihood
    log_lik = (
        -0.5 * Y @ alpha
        - np.sum(np.log(np.diag(L[0])))
        - 0.5 * n * np.log(2 * np.pi)
    )

    def predict(
        x: NDArray, _X=X, _alpha=alpha, _bw=bandwidth
    ) -> NDArray:
        Ks = _rbf_kernel(x, _X, bandwidth=_bw)
        return Ks @ _alpha

    return RegressionResult(
        reg_type=RegressionType.GP,
        predict=predict,
        residuals=residuals,
        params={
            "bandwidth": bandwidth,
            "noise_var": noise_var,
            "log_marginal_likelihood": float(log_lik),
        },
        score=float(log_lik),
    )


# ---------------------------------------------------------------------------
# HSIC-based independence test (lightweight, used internally)
# ---------------------------------------------------------------------------

def _hsic_test_statistic(
    X: NDArray,
    Y: NDArray,
    bandwidth_x: Optional[float] = None,
    bandwidth_y: Optional[float] = None,
) -> float:
    """Unbiased HSIC estimator (Song et al. 2012)."""
    n = len(X)
    if n < 4:
        return 0.0
    if bandwidth_x is None:
        bandwidth_x = _median_bandwidth(X)
    if bandwidth_y is None:
        bandwidth_y = _median_bandwidth(Y)

    Kx = _rbf_kernel(X, bandwidth=bandwidth_x)
    Ky = _rbf_kernel(Y, bandwidth=bandwidth_y)

    # Zero diagonals for unbiased estimator
    np.fill_diagonal(Kx, 0.0)
    np.fill_diagonal(Ky, 0.0)

    term1 = np.sum(Kx * Ky)
    term2 = np.sum(Kx) * np.sum(Ky) / ((n - 1) * (n - 2))
    term3 = 2.0 * np.sum(Kx @ Ky) / (n - 2)

    stat = (term1 + term2 - term3) / (n * (n - 3))
    return float(stat)


def _hsic_permutation_pvalue(
    X: NDArray,
    Y: NDArray,
    n_permutations: int = 500,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Permutation-based p-value for HSIC independence test."""
    rng = np.random.default_rng(seed)
    stat = _hsic_test_statistic(X, Y)
    null_stats = np.empty(n_permutations)
    for i in range(n_permutations):
        perm_y = rng.permutation(Y)
        null_stats[i] = _hsic_test_statistic(X, perm_y)
    pvalue = float(np.mean(null_stats >= stat))
    return stat, pvalue


# ---------------------------------------------------------------------------
# Additive Noise Model
# ---------------------------------------------------------------------------

class AdditiveNoiseModel:
    """Fit and evaluate an additive noise model  Y = f(X) + N.

    Parameters
    ----------
    regression : str or RegressionType
        Regression backend: ``"linear"``, ``"polynomial"``, ``"krr"``
        (kernel ridge), or ``"gp"`` (Gaussian process).
    degree : int
        Polynomial degree (only for ``"polynomial"``).
    bandwidth : float or None
        Kernel bandwidth (for KRR / GP); ``None`` uses median heuristic.
    reg_lambda : float
        Regularisation strength (KRR).
    noise_var : float
        Observation noise variance (GP).
    """

    def __init__(
        self,
        regression: Union[str, RegressionType] = "gp",
        degree: int = 3,
        bandwidth: Optional[float] = None,
        reg_lambda: float = 1e-3,
        noise_var: float = 0.1,
    ) -> None:
        if isinstance(regression, str):
            regression = RegressionType(regression)
        self.regression = regression
        self.degree = degree
        self.bandwidth = bandwidth
        self.reg_lambda = reg_lambda
        self.noise_var = noise_var

        self.result_: Optional[RegressionResult] = None
        self.noise_distribution_: Optional[stats.rv_continuous] = None

    def fit(self, X: NDArray, Y: NDArray) -> "AdditiveNoiseModel":
        """Fit the regression f and compute residuals N = Y - f(X)."""
        X = np.asarray(X, dtype=np.float64).ravel()
        Y = np.asarray(Y, dtype=np.float64).ravel()

        if self.regression == RegressionType.LINEAR:
            self.result_ = _fit_linear(X, Y)
        elif self.regression == RegressionType.POLYNOMIAL:
            self.result_ = _fit_polynomial(X, Y, degree=self.degree)
        elif self.regression == RegressionType.KRR:
            self.result_ = _fit_kernel_ridge(
                X, Y, bandwidth=self.bandwidth, reg_lambda=self.reg_lambda
            )
        elif self.regression == RegressionType.GP:
            self.result_ = _fit_gp(
                X, Y, bandwidth=self.bandwidth, noise_var=self.noise_var
            )
        else:
            raise ValueError(f"Unknown regression type: {self.regression}")

        self._estimate_noise_distribution()
        return self

    @property
    def residuals(self) -> NDArray:
        if self.result_ is None:
            raise RuntimeError("Model not fitted.")
        return self.result_.residuals

    def predict(self, X: NDArray) -> NDArray:
        if self.result_ is None:
            raise RuntimeError("Model not fitted.")
        return self.result_.predict(np.asarray(X).ravel())

    def _estimate_noise_distribution(self) -> None:
        """Fit a Gaussian KDE to the residuals for density estimation."""
        residuals = self.residuals
        if len(residuals) < 5:
            self.noise_distribution_ = stats.norm(
                loc=np.mean(residuals), scale=np.std(residuals) + 1e-12
            )
            return
        try:
            kde = stats.gaussian_kde(residuals)
            self.noise_distribution_ = kde
        except np.linalg.LinAlgError:
            self.noise_distribution_ = stats.norm(
                loc=np.mean(residuals), scale=np.std(residuals) + 1e-12
            )

    def independence_test(
        self,
        X: NDArray,
        n_permutations: int = 500,
        seed: Optional[int] = None,
    ) -> Tuple[float, float]:
        """Test independence between residuals and X using HSIC.

        Returns
        -------
        (statistic, p_value) : tuple[float, float]
        """
        X = np.asarray(X).ravel()
        return _hsic_permutation_pvalue(
            X, self.residuals, n_permutations=n_permutations, seed=seed
        )

    def noise_log_likelihood(self) -> float:
        """Log-likelihood of residuals under the fitted noise distribution."""
        residuals = self.residuals
        if self.noise_distribution_ is None:
            return float("-inf")
        if hasattr(self.noise_distribution_, "logpdf"):
            return float(np.sum(self.noise_distribution_.logpdf(residuals)))
        # KDE
        return float(np.sum(np.log(self.noise_distribution_.evaluate(residuals) + 1e-300)))

    def bic_score(self, n_params: Optional[int] = None) -> float:
        """BIC score for model selection: -2 * logL + k * log(n)."""
        n = len(self.residuals)
        log_lik = self.noise_log_likelihood()
        if n_params is None:
            if self.regression == RegressionType.LINEAR:
                n_params = 2
            elif self.regression == RegressionType.POLYNOMIAL:
                n_params = self.degree + 1
            else:
                n_params = int(np.sqrt(n))
        return float(-2.0 * log_lik + n_params * np.log(n))


# ---------------------------------------------------------------------------
# Model selection across regression types
# ---------------------------------------------------------------------------

def select_best_model(
    X: NDArray, Y: NDArray, candidates: Optional[List[str]] = None
) -> AdditiveNoiseModel:
    """Select the best ANM regression type via BIC.

    Parameters
    ----------
    X, Y : array-like
        Cause and effect data.
    candidates : list of str, optional
        Regression types to try; defaults to all available.

    Returns
    -------
    AdditiveNoiseModel
        Fitted model with lowest BIC.
    """
    if candidates is None:
        candidates = ["linear", "polynomial", "krr", "gp"]

    best_bic = np.inf
    best_model: Optional[AdditiveNoiseModel] = None

    for cand in candidates:
        try:
            m = AdditiveNoiseModel(regression=cand)
            m.fit(X, Y)
            bic = m.bic_score()
            if bic < best_bic:
                best_bic = bic
                best_model = m
        except Exception:
            continue

    if best_model is None:
        best_model = AdditiveNoiseModel(regression="linear")
        best_model.fit(X, Y)
    return best_model


# ---------------------------------------------------------------------------
# ANM-based causal direction test (Hoyer et al. 2009)
# ---------------------------------------------------------------------------

@dataclass
class DirectionTestResult:
    """Result of an ANM-based causal direction test."""
    direction: str  # "X->Y", "Y->X", or "undetermined"
    p_forward: float  # p-value testing N ⊥ X in model Y = f(X) + N
    p_backward: float  # p-value testing N ⊥ Y in model X = g(Y) + N
    hsic_forward: float
    hsic_backward: float
    model_forward: Optional[AdditiveNoiseModel] = None
    model_backward: Optional[AdditiveNoiseModel] = None
    confidence: float = 0.0


class ANMDirectionTest:
    """Bivariate causal direction test using additive noise models.

    Fits Y = f(X) + N_1 and X = g(Y) + N_2, then tests independence of
    the residuals from the input variable.  If one direction yields
    independent residuals and the other does not, the causal direction is
    identified.

    Parameters
    ----------
    regression : str
        Regression backend for the ANM.
    alpha : float
        Significance level for the independence test.
    n_permutations : int
        Number of permutations for the HSIC test.
    auto_select_model : bool
        If True, select the best regression type via BIC.
    """

    def __init__(
        self,
        regression: str = "gp",
        alpha: float = 0.05,
        n_permutations: int = 500,
        auto_select_model: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.regression = regression
        self.alpha = alpha
        self.n_permutations = n_permutations
        self.auto_select_model = auto_select_model
        self.seed = seed

    def test(self, X: NDArray, Y: NDArray) -> DirectionTestResult:
        """Run the causal direction test.

        Parameters
        ----------
        X, Y : array-like, shape (n,)
            Observations of the two variables.

        Returns
        -------
        DirectionTestResult
        """
        X = np.asarray(X, dtype=np.float64).ravel()
        Y = np.asarray(Y, dtype=np.float64).ravel()

        # Forward: Y = f(X) + N
        if self.auto_select_model:
            model_fwd = select_best_model(X, Y)
        else:
            model_fwd = AdditiveNoiseModel(regression=self.regression)
            model_fwd.fit(X, Y)
        hsic_fwd, p_fwd = model_fwd.independence_test(
            X, n_permutations=self.n_permutations, seed=self.seed
        )

        # Backward: X = g(Y) + N
        if self.auto_select_model:
            model_bwd = select_best_model(Y, X)
        else:
            model_bwd = AdditiveNoiseModel(regression=self.regression)
            model_bwd.fit(Y, X)
        hsic_bwd, p_bwd = model_bwd.independence_test(
            Y, n_permutations=self.n_permutations, seed=self.seed
        )

        # Decision
        fwd_indep = p_fwd > self.alpha
        bwd_indep = p_bwd > self.alpha

        if fwd_indep and not bwd_indep:
            direction = "X->Y"
            confidence = (1.0 - p_bwd) * p_fwd
        elif bwd_indep and not fwd_indep:
            direction = "Y->X"
            confidence = (1.0 - p_fwd) * p_bwd
        elif fwd_indep and bwd_indep:
            # Both independent: use multiple tiebreakers
            resid_fwd = model_fwd.residuals
            resid_bwd = model_bwd.residuals
            # 1. Normalized residual variance (lower = better fit)
            var_fwd = np.var(resid_fwd) / (np.var(Y) + 1e-10)
            var_bwd = np.var(resid_bwd) / (np.var(X) + 1e-10)
            # 2. HSIC (lower = more independent)
            score_fwd = hsic_fwd
            score_bwd = hsic_bwd
            # 3. Higher p-value = more independent
            if abs(p_fwd - p_bwd) > 0.05:
                if p_fwd > p_bwd:
                    direction = "X->Y"
                    confidence = p_fwd - p_bwd
                else:
                    direction = "Y->X"
                    confidence = p_bwd - p_fwd
            elif abs(var_fwd - var_bwd) > 0.01:
                if var_fwd < var_bwd:
                    direction = "X->Y"
                    confidence = max(0.0, (var_bwd - var_fwd) / (var_bwd + var_fwd + 1e-10))
                else:
                    direction = "Y->X"
                    confidence = max(0.0, (var_fwd - var_bwd) / (var_fwd + var_bwd + 1e-10))
            elif hsic_fwd < hsic_bwd:
                direction = "X->Y"
                confidence = max(0.0, (hsic_bwd - hsic_fwd) / (hsic_bwd + hsic_fwd + 1e-10))
            elif hsic_bwd < hsic_fwd:
                direction = "Y->X"
                confidence = max(0.0, (hsic_fwd - hsic_bwd) / (hsic_fwd + hsic_bwd + 1e-10))
            else:
                direction = "undetermined"
                confidence = 0.0
        else:
            direction = "undetermined"
            confidence = 0.0

        return DirectionTestResult(
            direction=direction,
            p_forward=p_fwd,
            p_backward=p_bwd,
            hsic_forward=hsic_fwd,
            hsic_backward=hsic_bwd,
            model_forward=model_fwd,
            model_backward=model_bwd,
            confidence=confidence,
        )

    def test_multivariate(
        self,
        data: NDArray,
        variable_names: Optional[List[str]] = None,
    ) -> Dict[Tuple[str, str], DirectionTestResult]:
        """Run pairwise ANM direction tests on all variable pairs.

        Parameters
        ----------
        data : ndarray, shape (n, p)
            Observations.
        variable_names : list of str, optional
            Column names; defaults to X0, X1, ...

        Returns
        -------
        dict[(str, str), DirectionTestResult]
        """
        data = np.asarray(data, dtype=np.float64)
        n, p = data.shape
        if variable_names is None:
            variable_names = [f"X{i}" for i in range(p)]

        results: Dict[Tuple[str, str], DirectionTestResult] = {}
        for i in range(p):
            for j in range(i + 1, p):
                result = self.test(data[:, i], data[:, j])
                pair = (variable_names[i], variable_names[j])
                results[pair] = result
        return results

    def infer_dag(
        self,
        data: NDArray,
        variable_names: Optional[List[str]] = None,
    ) -> "nx.DiGraph":
        """Build a DAG from pairwise ANM direction tests.

        Adds a directed edge for every pair whose direction is determined,
        then prunes any cycles greedily (by removing the least-confident
        edge on each cycle).
        """
        import networkx as nx

        data = np.asarray(data, dtype=np.float64)
        _, p = data.shape
        if variable_names is None:
            variable_names = [f"X{i}" for i in range(p)]

        pairwise = self.test_multivariate(data, variable_names)
        G = nx.DiGraph()
        G.add_nodes_from(variable_names)

        edges_with_conf: List[Tuple[str, str, float]] = []
        for (vi, vj), result in pairwise.items():
            if result.direction == "X->Y":
                edges_with_conf.append((vi, vj, result.confidence))
            elif result.direction == "Y->X":
                edges_with_conf.append((vj, vi, result.confidence))

        # Sort by confidence descending and add if no cycle created
        edges_with_conf.sort(key=lambda t: t[2], reverse=True)
        for u, v, conf in edges_with_conf:
            G.add_edge(u, v, confidence=conf)
            if not nx.is_directed_acyclic_graph(G):
                G.remove_edge(u, v)

        return G


# ---------------------------------------------------------------------------
# Multivariate ANM fitting
# ---------------------------------------------------------------------------

class MultivariateANM:
    """Fit additive noise models for every variable given its candidate
    parents in a DAG.

    Parameters
    ----------
    graph : nx.DiGraph
        Causal DAG.
    regression : str
        Regression backend.
    """

    def __init__(
        self,
        graph: "nx.DiGraph",
        regression: str = "gp",
    ) -> None:
        import networkx as nx
        self.graph = graph
        self.regression = regression
        self.models_: Dict[str, AdditiveNoiseModel] = {}

    def fit(self, data: Dict[str, NDArray]) -> "MultivariateANM":
        """Fit an ANM for every non-root node."""
        import networkx as nx

        for node in nx.topological_sort(self.graph):
            parents = list(self.graph.predecessors(node))
            if not parents:
                continue
            Y = data[node]
            # Build multivariate input
            X_multi = np.column_stack([data[p] for p in parents])
            # For simplicity, project to 1-D via first PC
            if X_multi.shape[1] > 1:
                X_proj = self._project(X_multi)
            else:
                X_proj = X_multi.ravel()
            model = AdditiveNoiseModel(regression=self.regression)
            model.fit(X_proj, Y)
            self.models_[node] = model
        return self

    @staticmethod
    def _project(X: NDArray) -> NDArray:
        """Project multivariate parents onto first principal component."""
        X_centered = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        return (X_centered @ Vt[0]).ravel()

    def residuals(self) -> Dict[str, NDArray]:
        return {node: m.residuals for node, m in self.models_.items()}

    def test_all_independence(
        self,
        data: Dict[str, NDArray],
        n_permutations: int = 300,
    ) -> Dict[str, Tuple[float, float]]:
        """Independence test of residuals vs parents for all nodes."""
        results: Dict[str, Tuple[float, float]] = {}
        for node, model in self.models_.items():
            parents = list(self.graph.predecessors(node))
            X_multi = np.column_stack([data[p] for p in parents])
            if X_multi.shape[1] > 1:
                X_proj = self._project(X_multi)
            else:
                X_proj = X_multi.ravel()
            stat, pval = model.independence_test(
                X_proj, n_permutations=n_permutations
            )
            results[node] = (stat, pval)
        return results
