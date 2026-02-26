"""
Synthetic market data generation with regime-switching dynamics.

Generates time series from known Regime-Indexed Structural Causal Models
(RI-SCMs) so that downstream algorithms can be evaluated against ground
truth.

Key properties of the generated data:

* **Regime-switching** – a latent Markov chain over 3-5 regimes drives
  structural changes in the data-generating process.
* **Controlled invariance** – a configurable fraction of causal edges is
  shared across all regimes ("invariant"), while the rest are
  regime-specific.
* **Realistic marginals** – returns exhibit fat tails (Student-*t*)
  and GARCH-like volatility clustering.
* **Ground truth** – regime labels, adjacency matrices, and invariant
  edge sets are recorded for evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class GroundTruth:
    """Ground-truth labels for a synthetic dataset."""
    regime_labels: NDArray                     # (T,) int
    regime_transition_matrix: NDArray          # (K, K)
    adjacency_matrices: Dict[int, NDArray]     # regime → (p, p) bool
    invariant_edges: NDArray                   # (p, p) bool
    invariant_features: List[int]
    regime_specific_features: Dict[int, List[int]]
    causal_coefficients: Dict[int, NDArray]    # regime → (p, p)
    noise_scales: Dict[int, NDArray]           # regime → (p,)
    n_regimes: int
    n_features: int


@dataclass
class SyntheticDataset:
    """Container for a synthetic market dataset."""
    returns: NDArray      # (T,) – portfolio-level returns
    features: NDArray     # (T, p) – observed feature matrix
    prices: NDArray       # (T,) – cumulated price series
    volumes: NDArray      # (T,) – synthetic volume series
    ground_truth: GroundTruth
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Regime transition logic
# ---------------------------------------------------------------------------

def _build_transition_matrix(
    n_regimes: int,
    persistence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> NDArray:
    """Create a row-stochastic Markov transition matrix.

    Each regime stays in itself with probability *persistence*; the
    remaining mass is spread uniformly over other regimes.
    """
    rng = rng or np.random.default_rng()
    P = np.full((n_regimes, n_regimes), (1.0 - persistence) / (n_regimes - 1))
    np.fill_diagonal(P, persistence)
    # Small random perturbation
    noise = rng.dirichlet(np.ones(n_regimes) * 50, size=n_regimes)
    P = 0.9 * P + 0.1 * noise
    # Re-normalise
    P /= P.sum(axis=1, keepdims=True)
    return P


def _simulate_regime_sequence(
    P: NDArray, T: int, rng: Optional[np.random.Generator] = None
) -> NDArray:
    """Draw a regime sequence of length T from the Markov chain."""
    rng = rng or np.random.default_rng()
    n_regimes = P.shape[0]
    regimes = np.zeros(T, dtype=int)

    # Stationary distribution as initial state
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.real(eigenvectors[:, idx])
    pi = pi / pi.sum()
    regimes[0] = rng.choice(n_regimes, p=np.abs(pi))

    for t in range(1, T):
        regimes[t] = rng.choice(n_regimes, p=P[regimes[t - 1]])

    return regimes


# ---------------------------------------------------------------------------
# Causal graph generation
# ---------------------------------------------------------------------------

def _generate_dag(
    p: int,
    edge_density: float = 0.15,
    rng: Optional[np.random.Generator] = None,
) -> NDArray:
    """Generate a random DAG adjacency matrix (lower-triangular)."""
    rng = rng or np.random.default_rng()
    A = np.zeros((p, p), dtype=bool)
    for i in range(1, p):
        for j in range(i):
            if rng.random() < edge_density:
                A[i, j] = True
    return A


def _generate_regime_graphs(
    p: int,
    n_regimes: int,
    invariant_ratio: float = 0.4,
    edge_density: float = 0.15,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Dict[int, NDArray], NDArray]:
    """Build per-regime DAGs sharing a common invariant skeleton.

    Returns
    -------
    adj : dict mapping regime → (p, p) bool adjacency
    invariant : (p, p) bool – edges present in *all* regime graphs
    """
    rng = rng or np.random.default_rng()

    base = _generate_dag(p, edge_density, rng)
    n_edges = int(base.sum())
    n_invariant = max(1, int(invariant_ratio * n_edges))

    # Choose invariant edge positions
    edge_positions = list(zip(*np.where(base)))
    rng.shuffle(edge_positions)
    invariant_set = set(edge_positions[:n_invariant])

    invariant = np.zeros((p, p), dtype=bool)
    for i, j in invariant_set:
        invariant[i, j] = True

    adj: Dict[int, NDArray] = {}
    for r in range(n_regimes):
        A_r = invariant.copy()
        # Add regime-specific edges
        for i, j in edge_positions[n_invariant:]:
            if rng.random() < 0.5:
                A_r[i, j] = True
        # Randomly add a few novel edges
        for _ in range(max(1, n_edges // 5)):
            i = rng.integers(1, p)
            j = rng.integers(0, i)
            A_r[i, j] = True
        adj[r] = A_r

    return adj, invariant


def _generate_coefficients(
    adj: NDArray,
    coeff_range: Tuple[float, float] = (0.2, 0.8),
    rng: Optional[np.random.Generator] = None,
) -> NDArray:
    """Assign random linear coefficients to edges in a DAG."""
    rng = rng or np.random.default_rng()
    p = adj.shape[0]
    W = np.zeros((p, p), dtype=np.float64)
    lo, hi = coeff_range
    for i in range(p):
        for j in range(p):
            if adj[i, j]:
                sign = rng.choice([-1.0, 1.0])
                W[i, j] = sign * rng.uniform(lo, hi)
    return W


# ---------------------------------------------------------------------------
# GARCH-like volatility
# ---------------------------------------------------------------------------

def _simulate_garch_vol(
    T: int,
    omega: float = 1e-5,
    alpha: float = 0.08,
    beta: float = 0.90,
    rng: Optional[np.random.Generator] = None,
) -> NDArray:
    """Simulate a GARCH(1,1) conditional-variance process.

    Returns (T,) array of conditional standard deviations.
    """
    rng = rng or np.random.default_rng()
    var = np.zeros(T, dtype=np.float64)
    var[0] = omega / (1.0 - alpha - beta) if alpha + beta < 1.0 else omega * 100
    eps = rng.standard_normal(T)

    for t in range(1, T):
        var[t] = omega + alpha * (eps[t - 1] * np.sqrt(var[t - 1])) ** 2 + beta * var[t - 1]
        var[t] = max(var[t], 1e-10)

    return np.sqrt(var)


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

class SyntheticMarketGenerator:
    """Generate synthetic market data from RI-SCMs.

    Parameters
    ----------
    n_features : int
        Number of observed variables (default 30).
    n_regimes : int
        Number of latent regimes (default 3).
    invariant_ratio : float
        Fraction of causal edges that are shared across regimes.
    edge_density : float
        Expected density of the causal DAG.
    snr : float
        Signal-to-noise ratio controlling residual variance.
    regime_persistence : float
        Diagonal of the Markov transition matrix.
    fat_tail_df : float
        Degrees of freedom for Student-*t* noise (∞ → Gaussian).
    use_garch : bool
        Add GARCH(1,1) volatility clustering.
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_features: int = 30,
        n_regimes: int = 3,
        invariant_ratio: float = 0.4,
        edge_density: float = 0.15,
        snr: float = 2.0,
        regime_persistence: float = 0.97,
        fat_tail_df: float = 5.0,
        use_garch: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.n_features = n_features
        self.n_regimes = n_regimes
        self.invariant_ratio = invariant_ratio
        self.edge_density = edge_density
        self.snr = snr
        self.regime_persistence = regime_persistence
        self.fat_tail_df = fat_tail_df
        self.use_garch = use_garch
        self.rng = np.random.default_rng(seed)

        self._ground_truth: Optional[GroundTruth] = None
        self._dataset: Optional[SyntheticDataset] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        T: int = 5000,
        n_regimes: Optional[int] = None,
        n_features: Optional[int] = None,
    ) -> SyntheticDataset:
        """Generate a complete synthetic dataset.

        Parameters
        ----------
        T : int
            Number of time steps.
        n_regimes : int or None
            Override instance default.
        n_features : int or None
            Override instance default.

        Returns
        -------
        SyntheticDataset
        """
        K = n_regimes or self.n_regimes
        p = n_features or self.n_features

        # 1. Regime dynamics
        P = _build_transition_matrix(K, self.regime_persistence, self.rng)
        regimes = _simulate_regime_sequence(P, T, self.rng)

        # 2. Causal graphs
        adj, invariant = _generate_regime_graphs(
            p, K, self.invariant_ratio, self.edge_density, self.rng
        )

        # 3. Coefficients
        causal_coefs: Dict[int, NDArray] = {}
        for r in range(K):
            causal_coefs[r] = _generate_coefficients(adj[r], rng=self.rng)

        # 4. Noise scales per regime
        noise_scales: Dict[int, NDArray] = {}
        for r in range(K):
            base_scale = 1.0 / self.snr
            noise_scales[r] = base_scale * (
                0.5 + self.rng.exponential(0.5, size=p)
            )

        # 5. GARCH vol (single series applied to all features)
        if self.use_garch:
            vol = _simulate_garch_vol(T, rng=self.rng)
        else:
            vol = np.ones(T)

        # 6. Simulate features from the SCM
        features = np.zeros((T, p), dtype=np.float64)
        for t in range(T):
            r = regimes[t]
            W = causal_coefs[r]
            sigma = noise_scales[r] * vol[t]

            # Topological-order sampling (lower-triangular W)
            x = np.zeros(p)
            for i in range(p):
                parent_effect = W[i, :i] @ x[:i] if i > 0 else 0.0
                if np.isfinite(self.fat_tail_df) and self.fat_tail_df > 2:
                    noise = (
                        self.rng.standard_t(self.fat_tail_df) * sigma[i]
                    )
                else:
                    noise = self.rng.normal(0, sigma[i])
                x[i] = parent_effect + noise
            features[t] = x

        # 7. Construct returns and prices
        returns = self._features_to_returns(features, regimes, causal_coefs)
        prices = 100.0 * np.exp(np.cumsum(returns))
        volumes = self._generate_volumes(T, regimes, vol)

        # 8. Identify invariant / regime-specific feature indices
        invariant_feats = self._identify_invariant_features(adj, K)
        regime_specific_feats = self._identify_regime_specific(adj, invariant_feats, K)

        gt = GroundTruth(
            regime_labels=regimes,
            regime_transition_matrix=P,
            adjacency_matrices=adj,
            invariant_edges=invariant,
            invariant_features=invariant_feats,
            regime_specific_features=regime_specific_feats,
            causal_coefficients=causal_coefs,
            noise_scales=noise_scales,
            n_regimes=K,
            n_features=p,
        )

        dataset = SyntheticDataset(
            returns=returns,
            features=features,
            prices=prices,
            volumes=volumes,
            ground_truth=gt,
            metadata={
                "T": T,
                "snr": self.snr,
                "fat_tail_df": self.fat_tail_df,
                "use_garch": self.use_garch,
                "regime_persistence": self.regime_persistence,
            },
        )
        self._ground_truth = gt
        self._dataset = dataset

        logger.info(
            "Generated synthetic dataset: T=%d, p=%d, K=%d, "
            "%d invariant features",
            T, p, K, len(invariant_feats),
        )
        return dataset

    def get_ground_truth(self) -> GroundTruth:
        """Return ground-truth labels from the last generation."""
        if self._ground_truth is None:
            raise RuntimeError("Call generate() first.")
        return self._ground_truth

    def get_regime_data(
        self, regime: int
    ) -> Tuple[NDArray, NDArray]:
        """Return (features, returns) restricted to a single regime."""
        if self._dataset is None:
            raise RuntimeError("Call generate() first.")
        mask = self._ground_truth.regime_labels == regime  # type: ignore[union-attr]
        return self._dataset.features[mask], self._dataset.returns[mask]

    def reseed(self, seed: int) -> None:
        """Re-seed the generator."""
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _features_to_returns(
        self,
        features: NDArray,
        regimes: NDArray,
        causal_coefs: Dict[int, NDArray],
    ) -> NDArray:
        """Construct a synthetic return series as a function of features.

        Returns are a linear combination of the first few features (acting
        as "alpha signals") plus noise.
        """
        T, p = features.shape
        n_alpha = min(5, p)
        returns = np.zeros(T, dtype=np.float64)

        for t in range(T):
            r = regimes[t]
            W = causal_coefs[r]
            alpha_weights = W[0, :n_alpha] if p > 1 else np.array([0.5])
            signal = features[t, :n_alpha] @ alpha_weights
            noise = self.rng.normal(0, 0.01)
            returns[t] = np.clip(signal * 0.01 + noise, -0.10, 0.10)

        return returns

    def _generate_volumes(
        self,
        T: int,
        regimes: NDArray,
        vol: NDArray,
    ) -> NDArray:
        """Synthetic volume: higher in high-vol / regime-transition periods."""
        base_volume = 1e6
        volumes = np.zeros(T, dtype=np.float64)
        for t in range(T):
            regime_factor = 1.0 + 0.3 * regimes[t]
            vol_factor = vol[t] / np.mean(vol)
            transition = 1.0
            if t > 0 and regimes[t] != regimes[t - 1]:
                transition = 2.0
            volumes[t] = (
                base_volume
                * regime_factor
                * vol_factor
                * transition
                * self.rng.lognormal(0, 0.3)
            )
        return volumes

    @staticmethod
    def _identify_invariant_features(
        adj: Dict[int, NDArray], n_regimes: int
    ) -> List[int]:
        """Features whose parent sets are identical across all regimes."""
        p = adj[0].shape[0]
        invariant: List[int] = []
        for j in range(p):
            parents = [
                frozenset(np.where(adj[r][:, j])[0]) for r in range(n_regimes)
            ]
            if len(set(parents)) == 1:
                invariant.append(j)
        return invariant

    @staticmethod
    def _identify_regime_specific(
        adj: Dict[int, NDArray],
        invariant: List[int],
        n_regimes: int,
    ) -> Dict[int, List[int]]:
        """For each regime, list features whose parents differ from the
        invariant skeleton."""
        p = adj[0].shape[0]
        result: Dict[int, List[int]] = {}
        for r in range(n_regimes):
            specific = [j for j in range(p) if j not in invariant]
            if specific:
                result[r] = specific
        return result
