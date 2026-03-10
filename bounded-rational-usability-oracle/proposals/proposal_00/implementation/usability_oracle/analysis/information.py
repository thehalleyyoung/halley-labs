"""
usability_oracle.analysis.information — Information-theoretic analysis.

Provides channel capacity analysis for UI communication channels,
rate-distortion analysis for abstraction quality, mutual information
estimation, and information-bottleneck computation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

_EPS = 1e-30


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ChannelAnalysis:
    """Results of information-theoretic channel analysis.

    Attributes:
        capacity: Channel capacity in bits.
        optimal_input: Capacity-achieving input distribution.
        mutual_information: MI for the given input distribution.
        redundancy: Capacity - actual MI (unused channel capacity).
        efficiency: MI / Capacity ratio.
    """
    capacity: float = 0.0
    optimal_input: Optional[np.ndarray] = None
    mutual_information: float = 0.0
    redundancy: float = 0.0
    efficiency: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"Channel Analysis:\n"
            f"  Capacity: {self.capacity:.4f} bits\n"
            f"  MI:       {self.mutual_information:.4f} bits\n"
            f"  Efficiency: {self.efficiency:.2%}\n"
            f"  Redundancy: {self.redundancy:.4f} bits"
        )


@dataclass
class RateDistortionResult:
    """Rate-distortion analysis result."""
    rate: float = 0.0
    distortion: float = 0.0
    rd_curve: list[tuple[float, float]] = field(default_factory=list)
    optimal_encoder: Optional[np.ndarray] = None


@dataclass
class InformationBottleneckResult:
    """Information bottleneck result."""
    compressed_mi: float = 0.0
    relevant_mi: float = 0.0
    beta: float = 1.0
    encoder: Optional[np.ndarray] = None
    n_effective_clusters: int = 0


# ---------------------------------------------------------------------------
# Core information measures
# ---------------------------------------------------------------------------

def _entropy(p: np.ndarray) -> float:
    """Shannon entropy H(X) in bits."""
    p = np.asarray(p, dtype=float).ravel()
    p = p[p > _EPS]
    return float(-np.sum(p * np.log2(p)))


def _joint_entropy(pxy: np.ndarray) -> float:
    """Joint entropy H(X,Y) from joint distribution."""
    vals = pxy.ravel()
    vals = vals[vals > _EPS]
    return float(-np.sum(vals * np.log2(vals)))


def _conditional_entropy(pxy: np.ndarray) -> float:
    """Conditional entropy H(Y|X) = H(X,Y) - H(X)."""
    px = pxy.sum(axis=1)
    return _joint_entropy(pxy) - _entropy(px)


def _mutual_information(pxy: np.ndarray) -> float:
    """Mutual information I(X;Y) = H(X) + H(Y) - H(X,Y)."""
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    return _entropy(px) + _entropy(py) - _joint_entropy(pxy)


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL divergence D_KL(P || Q) in bits."""
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    mask = p > _EPS
    q_safe = np.maximum(q[mask], _EPS)
    return float(np.sum(p[mask] * np.log2(p[mask] / q_safe)))


def _cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """Cross-entropy H(P, Q) = -sum p * log2(q)."""
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    q_safe = np.maximum(q, _EPS)
    return float(-np.sum(p * np.log2(q_safe)))


def _jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence JSD(P || Q) in bits."""
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


# ---------------------------------------------------------------------------
# Blahut-Arimoto for channel capacity
# ---------------------------------------------------------------------------

def _blahut_arimoto_capacity(
    W: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> tuple[float, np.ndarray]:
    """Compute channel capacity and optimal input distribution.

    Parameters:
        W: Channel matrix P(Y|X), shape (|X|, |Y|).

    Returns:
        (capacity_in_bits, optimal_input_distribution)
    """
    n_x, n_y = W.shape
    p = np.full(n_x, 1.0 / n_x)
    capacity = 0.0

    for _ in range(max_iter):
        q = p @ W
        q = np.maximum(q, _EPS)

        # c(x) = sum_y W(y|x) log2(W(y|x) / q(y))
        c = np.zeros(n_x)
        for i in range(n_x):
            for j in range(n_y):
                if W[i, j] > _EPS:
                    c[i] += W[i, j] * math.log2(W[i, j] / q[j])

        exp_c = np.exp2(c)
        p_new = p * exp_c
        total = p_new.sum()
        p_new = p_new / total if total > _EPS else np.full(n_x, 1.0 / n_x)

        new_cap = 0.5 * (float(np.sum(p_new * c)) + float(np.log2(total)))
        if abs(new_cap - capacity) < tol:
            return max(new_cap, 0.0), p_new
        capacity = new_cap
        p = p_new

    return max(capacity, 0.0), p


# ---------------------------------------------------------------------------
# Rate-distortion via Blahut-Arimoto
# ---------------------------------------------------------------------------

def _blahut_arimoto_rd(
    source_dist: np.ndarray,
    distortion_matrix: np.ndarray,
    beta: float,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> tuple[float, float, np.ndarray]:
    """Compute a single point on the rate-distortion curve.

    Returns (rate, distortion, encoder q(x_hat|x)).
    """
    p = np.asarray(source_dist, dtype=float).ravel()
    D = np.asarray(distortion_matrix, dtype=float)
    n_x, n_xhat = D.shape

    m = np.full(n_xhat, 1.0 / n_xhat)
    q_cond = np.zeros((n_x, n_xhat))

    for _ in range(max_iter):
        for i in range(n_x):
            log_q = np.log(np.maximum(m, _EPS)) - beta * D[i, :]
            log_q -= np.max(log_q)
            q_row = np.exp(log_q)
            q_row /= q_row.sum()
            q_cond[i, :] = q_row

        m_new = p @ q_cond
        m_new = np.maximum(m_new, _EPS)
        m_new /= m_new.sum()

        if np.max(np.abs(m_new - m)) < tol:
            m = m_new
            break
        m = m_new

    rate = 0.0
    distortion = 0.0
    for i in range(n_x):
        for j in range(n_xhat):
            if q_cond[i, j] > _EPS and m[j] > _EPS:
                rate += p[i] * q_cond[i, j] * math.log2(q_cond[i, j] / m[j])
            distortion += p[i] * q_cond[i, j] * D[i, j]

    return max(rate, 0.0), distortion, q_cond


# ---------------------------------------------------------------------------
# Information Bottleneck
# ---------------------------------------------------------------------------

def _information_bottleneck(
    pxy: np.ndarray,
    n_clusters: int,
    beta: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> tuple[np.ndarray, float, float]:
    """Information bottleneck method.

    Finds a compressed representation T of X that preserves information
    about Y, by minimising I(X;T) - beta * I(T;Y).

    Parameters:
        pxy: Joint distribution P(X, Y).
        n_clusters: Number of compressed clusters |T|.
        beta: Trade-off parameter.

    Returns:
        (encoder q(t|x), I(X;T), I(T;Y))
    """
    px = pxy.sum(axis=1)
    py_given_x = pxy / np.maximum(px[:, np.newaxis], _EPS)
    n_x = pxy.shape[0]
    n_y = pxy.shape[1]
    n_t = min(n_clusters, n_x)

    # Random initialisation
    rng = np.random.RandomState(42)
    q_t_given_x = rng.dirichlet(np.ones(n_t), size=n_x)

    for _ in range(max_iter):
        # p(t) = sum_x p(x) q(t|x)
        pt = px @ q_t_given_x
        pt = np.maximum(pt, _EPS)

        # p(y|t) = sum_x p(x) q(t|x) p(y|x) / p(t)
        py_given_t = np.zeros((n_t, n_y))
        for t in range(n_t):
            for x in range(n_x):
                py_given_t[t, :] += px[x] * q_t_given_x[x, t] * py_given_x[x, :]
            py_given_t[t, :] /= max(pt[t], _EPS)

        # Update q(t|x) ∝ p(t) exp(-beta * D_KL(p(y|x) || p(y|t)))
        q_new = np.zeros((n_x, n_t))
        for x in range(n_x):
            for t in range(n_t):
                kl = 0.0
                for y in range(n_y):
                    if py_given_x[x, y] > _EPS and py_given_t[t, y] > _EPS:
                        kl += py_given_x[x, y] * math.log(py_given_x[x, y] / py_given_t[t, y])
                q_new[x, t] = pt[t] * math.exp(-beta * kl)
            row_sum = q_new[x, :].sum()
            if row_sum > _EPS:
                q_new[x, :] /= row_sum
            else:
                q_new[x, :] = 1.0 / n_t

        if np.max(np.abs(q_new - q_t_given_x)) < tol:
            q_t_given_x = q_new
            break
        q_t_given_x = q_new

    # Compute I(X;T) and I(T;Y)
    pt = px @ q_t_given_x
    pt = np.maximum(pt, _EPS)

    i_xt = 0.0
    for x in range(n_x):
        for t in range(n_t):
            if q_t_given_x[x, t] > _EPS and pt[t] > _EPS:
                i_xt += px[x] * q_t_given_x[x, t] * math.log2(q_t_given_x[x, t] / pt[t])

    # Recompute p(y|t) with final q
    py_given_t = np.zeros((n_t, n_y))
    for t in range(n_t):
        for x in range(n_x):
            py_given_t[t, :] += px[x] * q_t_given_x[x, t] * py_given_x[x, :]
        if pt[t] > _EPS:
            py_given_t[t, :] /= pt[t]

    i_ty = 0.0
    py = pxy.sum(axis=0)
    py = np.maximum(py, _EPS)
    for t in range(n_t):
        for y in range(n_y):
            joint = pt[t] * py_given_t[t, y]
            if joint > _EPS and py[y] > _EPS:
                i_ty += joint * math.log2(joint / (pt[t] * py[y]))

    return q_t_given_x, max(i_xt, 0.0), max(i_ty, 0.0)


# ---------------------------------------------------------------------------
# InformationAnalyzer
# ---------------------------------------------------------------------------

class InformationAnalyzer:
    """Information-theoretic analysis of UI interaction channels.

    Treats the UI as a noisy communication channel between designer intent
    and user comprehension, analysing capacity, efficiency, and bottlenecks.
    """

    def __init__(self, max_iter: int = 500, tol: float = 1e-8) -> None:
        self._max_iter = max_iter
        self._tol = tol

    # ------------------------------------------------------------------
    # Channel capacity analysis
    # ------------------------------------------------------------------

    def channel_analysis(
        self,
        channel_matrix: np.ndarray,
        input_distribution: Optional[np.ndarray] = None,
    ) -> ChannelAnalysis:
        """Analyse a UI communication channel.

        Parameters:
            channel_matrix: P(perceived_state | actual_state), shape (n_actual, n_perceived).
            input_distribution: Prior distribution over actual states.
                Uniform if not provided.
        """
        W = np.asarray(channel_matrix, dtype=float)
        if W.ndim != 2 or W.shape[0] == 0 or W.shape[1] == 0:
            return ChannelAnalysis()

        # Normalise rows
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > _EPS, row_sums, 1.0)
        W = W / row_sums

        # Channel capacity
        capacity, opt_input = _blahut_arimoto_capacity(W, self._max_iter, self._tol)

        # MI for given or uniform input
        if input_distribution is not None:
            p_x = np.asarray(input_distribution, dtype=float).ravel()
        else:
            p_x = np.full(W.shape[0], 1.0 / W.shape[0])

        p_x = p_x / p_x.sum()
        pxy = p_x[:, np.newaxis] * W
        mi = _mutual_information(pxy)

        redundancy = max(capacity - mi, 0.0)
        efficiency = mi / capacity if capacity > _EPS else 0.0

        return ChannelAnalysis(
            capacity=capacity,
            optimal_input=opt_input,
            mutual_information=mi,
            redundancy=redundancy,
            efficiency=efficiency,
            metadata={"n_inputs": W.shape[0], "n_outputs": W.shape[1]},
        )

    # ------------------------------------------------------------------
    # Rate-distortion curve
    # ------------------------------------------------------------------

    def rate_distortion_curve(
        self,
        source_dist: np.ndarray,
        distortion_matrix: np.ndarray,
        n_points: int = 20,
        beta_range: tuple[float, float] = (0.01, 50.0),
    ) -> RateDistortionResult:
        """Compute the rate-distortion curve R(D).

        Parameters:
            source_dist: Source distribution p(x).
            distortion_matrix: d(x, x_hat) matrix.
            n_points: Number of points on the curve.
            beta_range: Range of Lagrange multiplier beta.
        """
        betas = np.logspace(
            math.log10(beta_range[0]),
            math.log10(beta_range[1]),
            n_points,
        )

        rd_curve: list[tuple[float, float]] = []
        last_encoder = None

        for beta in betas:
            rate, dist, encoder = _blahut_arimoto_rd(
                source_dist, distortion_matrix, float(beta),
                self._max_iter, self._tol,
            )
            rd_curve.append((rate, dist))
            last_encoder = encoder

        # Sort by distortion
        rd_curve.sort(key=lambda x: x[1])

        return RateDistortionResult(
            rate=rd_curve[-1][0] if rd_curve else 0.0,
            distortion=rd_curve[-1][1] if rd_curve else 0.0,
            rd_curve=rd_curve,
            optimal_encoder=last_encoder,
        )

    # ------------------------------------------------------------------
    # Information bottleneck
    # ------------------------------------------------------------------

    def information_bottleneck(
        self,
        joint_distribution: np.ndarray,
        n_clusters: int = 5,
        beta: float = 1.0,
    ) -> InformationBottleneckResult:
        """Apply the information bottleneck method.

        Finds a compressed representation of UI states that preserves
        task-relevant information.
        """
        pxy = np.asarray(joint_distribution, dtype=float)
        if pxy.ndim != 2 or pxy.shape[0] == 0:
            return InformationBottleneckResult()

        total = pxy.sum()
        if total > _EPS:
            pxy = pxy / total

        encoder, i_xt, i_ty = _information_bottleneck(
            pxy, n_clusters, beta, max_iter=200,
        )

        # Count effective clusters (those with non-negligible probability)
        pt = pxy.sum(axis=1) @ encoder
        n_eff = int(np.sum(pt > 0.01))

        return InformationBottleneckResult(
            compressed_mi=i_xt,
            relevant_mi=i_ty,
            beta=beta,
            encoder=encoder,
            n_effective_clusters=n_eff,
        )

    # ------------------------------------------------------------------
    # Mutual information estimation
    # ------------------------------------------------------------------

    def estimate_mi(
        self,
        x_samples: np.ndarray,
        y_samples: np.ndarray,
        n_bins: int = 20,
    ) -> float:
        """Estimate mutual information from samples using histogram binning.

        Parameters:
            x_samples: 1-D array of X samples.
            y_samples: 1-D array of Y samples.
            n_bins: Number of histogram bins per dimension.
        """
        x = np.asarray(x_samples, dtype=float).ravel()
        y = np.asarray(y_samples, dtype=float).ravel()
        n = min(len(x), len(y))
        if n < 10:
            return 0.0

        x = x[:n]
        y = y[:n]

        # 2D histogram to estimate joint distribution
        hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=n_bins)
        pxy = hist_2d / n
        pxy = np.maximum(pxy, _EPS)

        return _mutual_information(pxy)

    # ------------------------------------------------------------------
    # Transfer entropy (Granger causality in info-theory)
    # ------------------------------------------------------------------

    @staticmethod
    def transfer_entropy(
        source: np.ndarray,
        target: np.ndarray,
        lag: int = 1,
        n_bins: int = 10,
    ) -> float:
        """Estimate transfer entropy from source to target time series.

        TE(X->Y) = H(Y_{t+1} | Y_t) - H(Y_{t+1} | Y_t, X_t)
        """
        x = np.asarray(source, dtype=float).ravel()
        y = np.asarray(target, dtype=float).ravel()
        n = min(len(x), len(y)) - lag
        if n < 20:
            return 0.0

        y_past = y[:n]
        x_past = x[:n]
        y_future = y[lag:lag + n]

        # Discretize
        def _bin(arr: np.ndarray) -> np.ndarray:
            mn, mx = arr.min(), arr.max()
            if mx - mn < _EPS:
                return np.zeros(len(arr), dtype=int)
            return np.clip(((arr - mn) / (mx - mn) * (n_bins - 1)).astype(int), 0, n_bins - 1)

        yp_b = _bin(y_past)
        xp_b = _bin(x_past)
        yf_b = _bin(y_future)

        # Joint and conditional entropies via counting
        def _entropy_from_indices(*arrs: np.ndarray) -> float:
            combined = np.column_stack(arrs)
            _, counts = np.unique(combined, axis=0, return_counts=True)
            p = counts / counts.sum()
            return float(-np.sum(p * np.log2(p + _EPS)))

        h_yf_yp = _entropy_from_indices(yf_b, yp_b)
        h_yp = _entropy_from_indices(yp_b)
        h_yf_yp_xp = _entropy_from_indices(yf_b, yp_b, xp_b)
        h_yp_xp = _entropy_from_indices(yp_b, xp_b)

        # TE = H(Y_f, Y_p) - H(Y_p) - H(Y_f, Y_p, X_p) + H(Y_p, X_p)
        te = (h_yf_yp - h_yp) - (h_yf_yp_xp - h_yp_xp)
        return max(te, 0.0)

    # ------------------------------------------------------------------
    # Divergence measures
    # ------------------------------------------------------------------

    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """KL divergence D_KL(P || Q) in bits."""
        return _kl_divergence(p, q)

    @staticmethod
    def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """Jensen-Shannon divergence in bits."""
        return _jensen_shannon_divergence(p, q)

    @staticmethod
    def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
        """Cross-entropy H(P, Q) in bits."""
        return _cross_entropy(p, q)
