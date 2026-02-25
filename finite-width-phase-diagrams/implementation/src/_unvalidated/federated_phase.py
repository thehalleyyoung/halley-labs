"""Phase diagrams for federated learning.

Extends the phase diagram framework to federated learning, analyzing
how client heterogeneity, communication rounds, aggregation strategies,
and non-IID data distributions affect the lazy-to-rich phase transition.
Client drift introduces a novel dimension to the phase diagram not present
in centralized training.

Example
-------
>>> from phase_diagrams.federated_phase import federated_phase_diagram
>>> diagram = federated_phase_diagram(model, clients, lr_range=(1e-4, 0.1))
>>> print(diagram.metadata["n_clients"])
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .api import PhaseDiagram, PhasePoint, Regime, _compute_gamma, _predict_gamma_star


# ======================================================================
# Enums and data classes
# ======================================================================

class AggregationStrategy(str, Enum):
    """Federated aggregation strategy."""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    SCAFFOLD = "scaffold"
    FEDADAM = "fedadam"


class HeterogeneityLevel(str, Enum):
    """Data heterogeneity level across clients."""
    IID = "iid"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class DriftReport:
    """Analysis of client drift during federated training.

    Attributes
    ----------
    mean_drift : float
        Average parameter drift across clients per round.
    max_drift : float
        Maximum drift among all clients.
    drift_variance : float
        Variance of drift across clients.
    per_client_drift : NDArray
        Drift magnitude for each client.
    drift_direction_alignment : float
        How aligned client drift directions are (1 = identical, 0 = orthogonal).
    critical_local_steps : int
        Local steps above which drift causes regime change.
    drift_regime_effect : Dict[str, Any]
        How drift affects the phase regime.
    ntk_divergence : float
        Divergence between client NTKs.
    """
    mean_drift: float = 0.0
    max_drift: float = 0.0
    drift_variance: float = 0.0
    per_client_drift: NDArray = field(default_factory=lambda: np.array([]))
    drift_direction_alignment: float = 0.0
    critical_local_steps: int = 1
    drift_regime_effect: Dict[str, Any] = field(default_factory=dict)
    ntk_divergence: float = 0.0


@dataclass
class AggPhase:
    """Phase comparison across aggregation strategies.

    Attributes
    ----------
    strategies : List[str]
        Strategies evaluated.
    critical_lrs : Dict[str, float]
        Critical LR for each strategy.
    regimes : Dict[str, Regime]
        Regime at default LR for each strategy.
    convergence_rates : Dict[str, float]
        Convergence rate for each strategy.
    drift_compensation : Dict[str, float]
        How well each strategy compensates for drift.
    best_strategy : AggregationStrategy
        Recommended strategy.
    phase_diagrams : Dict[str, PhaseDiagram]
        Full phase diagram for each strategy.
    """
    strategies: List[str] = field(default_factory=list)
    critical_lrs: Dict[str, float] = field(default_factory=dict)
    regimes: Dict[str, Regime] = field(default_factory=dict)
    convergence_rates: Dict[str, float] = field(default_factory=dict)
    drift_compensation: Dict[str, float] = field(default_factory=dict)
    best_strategy: AggregationStrategy = AggregationStrategy.FEDAVG
    phase_diagrams: Dict[str, PhaseDiagram] = field(default_factory=dict)


@dataclass
class NonIIDPhase:
    """Phase analysis under non-IID data distribution.

    Attributes
    ----------
    heterogeneity_levels : NDArray
        Heterogeneity levels evaluated (0 = IID, 1 = fully non-IID).
    critical_lrs : NDArray
        Critical LR at each heterogeneity level.
    regimes : Dict[float, Regime]
        Regime at each heterogeneity level.
    heterogeneity_class : HeterogeneityLevel
        Classification of heterogeneity severity.
    convergence_degradation : NDArray
        How much convergence degrades at each level.
    client_ntk_similarity : NDArray
        Pairwise NTK similarity between clients.
    optimal_n_clients : int
        Optimal number of participating clients per round.
    """
    heterogeneity_levels: NDArray = field(default_factory=lambda: np.array([]))
    critical_lrs: NDArray = field(default_factory=lambda: np.array([]))
    regimes: Dict[float, Regime] = field(default_factory=dict)
    heterogeneity_class: HeterogeneityLevel = HeterogeneityLevel.IID
    convergence_degradation: NDArray = field(default_factory=lambda: np.array([]))
    client_ntk_similarity: NDArray = field(default_factory=lambda: np.array([]))
    optimal_n_clients: int = 10


@dataclass
class CommScaling:
    """How phase behavior scales with communication rounds.

    Attributes
    ----------
    rounds : NDArray
        Number of communication rounds evaluated.
    critical_lrs : NDArray
        Critical LR at each round count.
    convergence_progress : NDArray
        Convergence metric at each round.
    local_steps_per_round : NDArray
        Optimal local steps at each total round budget.
    communication_cost : NDArray
        Communication cost (bytes) at each round.
    pareto_frontier : NDArray
        Pareto-optimal (rounds, accuracy) pairs.
    scaling_exponent : float
        How critical LR scales with rounds.
    """
    rounds: NDArray = field(default_factory=lambda: np.array([]))
    critical_lrs: NDArray = field(default_factory=lambda: np.array([]))
    convergence_progress: NDArray = field(default_factory=lambda: np.array([]))
    local_steps_per_round: NDArray = field(default_factory=lambda: np.array([]))
    communication_cost: NDArray = field(default_factory=lambda: np.array([]))
    pareto_frontier: NDArray = field(default_factory=lambda: np.array([]))
    scaling_exponent: float = 0.0


# ======================================================================
# Internal helpers
# ======================================================================

def _extract_fed_params(model: Any) -> Dict[str, Any]:
    """Extract model parameters for federated analysis."""
    if isinstance(model, dict):
        return {
            "input_dim": model.get("input_dim", 784),
            "hidden_dim": model.get("hidden_dim", 256),
            "output_dim": model.get("output_dim", 10),
            "depth": model.get("depth", 3),
            "init_scale": model.get("init_scale", 1.0),
            "local_steps": model.get("local_steps", 5),
        }
    if isinstance(model, (list, tuple)):
        input_dim = model[0].shape[0] if model[0].ndim == 2 else 784
        hidden_dim = model[0].shape[1] if model[0].ndim == 2 else 256
        return {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": model[-1].shape[1] if model[-1].ndim == 2 else 10,
            "depth": len(model),
            "init_scale": float(np.std(model[0]) * math.sqrt(hidden_dim)),
            "local_steps": 5,
        }
    raise TypeError(f"Unsupported model type: {type(model)}")


def _generate_client_data(
    n_clients: int,
    input_dim: int,
    output_dim: int,
    n_samples_per_client: int,
    heterogeneity: float,
    seed: int = 42,
) -> List[Tuple[NDArray, NDArray]]:
    """Generate synthetic federated client data.

    Heterogeneity controls how different client distributions are:
    0 = IID, 1 = completely non-IID (each client has unique distribution).
    """
    rng = np.random.RandomState(seed)
    clients = []

    # Shared component
    W_shared = rng.randn(input_dim, output_dim)

    for c in range(n_clients):
        X = rng.randn(n_samples_per_client, input_dim)

        # Client-specific transformation
        W_client = rng.randn(input_dim, output_dim) * heterogeneity
        W_effective = W_shared * (1 - heterogeneity) + W_client

        Y = np.tanh(X @ W_effective + rng.randn(output_dim) * 0.1)
        clients.append((X, Y))

    return clients


def _extract_client_data(
    clients: Any,
    params: Dict[str, Any],
    seed: int = 42,
) -> List[Tuple[NDArray, NDArray]]:
    """Extract or generate client data."""
    if isinstance(clients, list) and len(clients) > 0:
        if isinstance(clients[0], tuple):
            return clients

    if isinstance(clients, dict):
        return _generate_client_data(
            clients.get("n_clients", 10),
            params["input_dim"],
            params["output_dim"],
            clients.get("n_samples", 100),
            clients.get("heterogeneity", 0.3),
            seed,
        )

    if isinstance(clients, int):
        return _generate_client_data(
            clients, params["input_dim"], params["output_dim"],
            100, 0.3, seed,
        )

    raise TypeError(f"Unsupported clients type: {type(clients)}")


def _initialize_weights(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    depth: int,
    init_scale: float,
    seed: int = 42,
) -> List[NDArray]:
    """Initialize MLP weights."""
    rng = np.random.RandomState(seed)
    weights = []
    dims = [input_dim] + [hidden_dim] * (depth - 1) + [output_dim]
    for l in range(depth):
        W = rng.randn(dims[l], dims[l + 1]) * init_scale / math.sqrt(dims[l])
        weights.append(W)
    return weights


def _local_sgd_step(
    weights: List[NDArray],
    X: NDArray,
    Y: NDArray,
    lr: float,
    n_steps: int,
) -> Tuple[List[NDArray], float]:
    """Run local SGD steps on client data. Returns (updated_weights, final_loss)."""
    w = [ww.copy() for ww in weights]
    n = X.shape[0]

    for step in range(n_steps):
        h = X
        activations = [h]
        for l in range(len(w)):
            pre = h @ w[l]
            h = np.maximum(pre, 0) if l < len(w) - 1 else pre
            activations.append(h)

        loss = float(np.mean((h - Y) ** 2))
        grad_output = 2 * (h - Y) / n
        delta = grad_output

        for l in range(len(w) - 1, -1, -1):
            grad_w = activations[l].T @ delta
            w[l] -= lr * grad_w
            if l > 0:
                delta = delta @ w[l].T
                delta = delta * (activations[l] > 0).astype(float)

    h = X
    for l in range(len(w)):
        pre = h @ w[l]
        h = np.maximum(pre, 0) if l < len(w) - 1 else pre
    final_loss = float(np.mean((h - Y) ** 2))
    return w, final_loss


def _federated_ntk_eigenspectrum(
    weights: List[NDArray],
    client_data: List[Tuple[NDArray, NDArray]],
    n_samples: int = 30,
    seed: int = 42,
) -> NDArray:
    """Approximate federated NTK from aggregated client data."""
    all_X = np.vstack([c[0][:min(n_samples // len(client_data) + 1, len(c[0]))]
                       for c in client_data])
    n = min(n_samples, all_X.shape[0])
    X = all_X[:n]
    depth = len(weights)

    K = np.zeros((n, n))
    h = X.copy()
    for l in range(depth):
        pre = h @ weights[l]
        if l < depth - 1:
            act_deriv = (pre > 0).astype(float)
            for i in range(n):
                for j in range(i, n):
                    val = np.dot(h[i], h[j]) * np.dot(act_deriv[i], act_deriv[j])
                    K[i, j] += val
                    K[j, i] = K[i, j]
            h = np.maximum(pre, 0)
        else:
            for i in range(n):
                for j in range(i, n):
                    K[i, j] += np.dot(h[i], h[j])
                    K[j, i] = K[i, j]

    eigvals = np.sort(np.linalg.eigvalsh(K))[::-1]
    return eigvals


def _client_ntk(
    weights: List[NDArray],
    X: NDArray,
    n_samples: int = 20,
) -> NDArray:
    """Compute NTK Gram matrix for a single client."""
    n = min(n_samples, X.shape[0])
    X = X[:n]
    depth = len(weights)
    K = np.zeros((n, n))
    h = X.copy()
    for l in range(depth):
        pre = h @ weights[l]
        if l < depth - 1:
            for i in range(n):
                for j in range(i, n):
                    K[i, j] += np.dot(h[i], h[j])
                    K[j, i] = K[i, j]
            h = np.maximum(pre, 0)
        else:
            for i in range(n):
                for j in range(i, n):
                    K[i, j] += np.dot(h[i], h[j])
                    K[j, i] = K[i, j]
    return K


# ======================================================================
# Public API
# ======================================================================

def federated_phase_diagram(
    model: Any,
    clients: Any,
    lr_range: Tuple[float, float] = (1e-4, 0.1),
    n_lr_steps: int = 25,
    training_steps: int = 100,
    seed: int = 42,
) -> PhaseDiagram:
    """Compute a phase diagram for federated learning.

    The federated phase boundary is shifted by client drift and
    data heterogeneity. The effective coupling includes contributions
    from local SGD steps that amplify the coupling per communication
    round.

    Parameters
    ----------
    model : dict or list of NDArray
        Model specification.
    clients : list, dict, or int
        Client data. List of (X, Y) tuples, dict with config, or just
        number of clients.
    lr_range : (float, float)
        Learning rate scan range.
    n_lr_steps : int
        Number of LR grid points.
    training_steps : int
        Number of communication rounds.
    seed : int
        Random seed.

    Returns
    -------
    PhaseDiagram
        Federated phase diagram.
    """
    params = _extract_fed_params(model)
    hidden_dim = params["hidden_dim"]
    init_scale = params["init_scale"]
    local_steps = params["local_steps"]

    weights = _initialize_weights(
        params["input_dim"], hidden_dim, params["output_dim"],
        params["depth"], init_scale, seed,
    )
    client_data = _extract_client_data(clients, params, seed)
    n_clients = len(client_data)

    eigvals = _federated_ntk_eigenspectrum(weights, client_data, seed=seed)
    mu_max = float(eigvals[0]) / hidden_dim if len(eigvals) > 0 else 1.0

    # Local steps amplify effective coupling
    local_amplification = 1.0 + local_steps * 0.1 * mu_max

    # Client heterogeneity correction: heterogeneous clients reduce
    # effective coupling due to gradient cancellation during aggregation
    client_ntks = [_client_ntk(weights, X) for X, _ in client_data[:min(10, n_clients)]]
    ntk_mean = np.mean([np.linalg.norm(K, "fro") for K in client_ntks])
    ntk_std = np.std([np.linalg.norm(K, "fro") for K in client_ntks])
    heterogeneity_correction = 1.0 / (1.0 + ntk_std / (ntk_mean + 1e-12))

    g_star = _predict_gamma_star(
        mu_max * local_amplification * heterogeneity_correction, training_steps
    )

    lrs = np.logspace(math.log10(lr_range[0]), math.log10(lr_range[1]), n_lr_steps)
    points: List[PhasePoint] = []
    boundary_pts: List[Tuple[float, int]] = []

    prev_regime = None
    for lr in lrs:
        gamma = _compute_gamma(lr, init_scale, hidden_dim)
        if gamma < g_star * 0.8:
            regime = Regime.LAZY
            confidence = min(1.0, (g_star - gamma) / g_star)
        elif gamma > g_star * 1.2:
            regime = Regime.RICH
            confidence = min(1.0, (gamma - g_star) / g_star)
        else:
            regime = Regime.CRITICAL
            confidence = 1.0 - abs(gamma - g_star) / (0.2 * g_star + 1e-12)

        ntk_drift = gamma * mu_max * local_amplification * heterogeneity_correction * training_steps
        points.append(PhasePoint(
            lr=float(lr),
            width=hidden_dim,
            regime=regime,
            gamma=gamma,
            gamma_star=g_star,
            confidence=max(0.0, min(1.0, confidence)),
            ntk_drift_predicted=ntk_drift,
        ))

        if prev_regime is not None and prev_regime != regime:
            boundary_pts.append((float(lr), hidden_dim))
        prev_regime = regime

    boundary_curve = np.array(boundary_pts) if boundary_pts else None
    tc_vals = [training_steps * _compute_gamma(bp[0], init_scale, hidden_dim) for bp in boundary_pts]
    tc = float(np.mean(tc_vals)) if tc_vals else 0.0

    return PhaseDiagram(
        points=points,
        lr_range=lr_range,
        width_range=(hidden_dim, hidden_dim),
        boundary_curve=boundary_curve,
        timescale_constant=tc,
        metadata={
            "architecture": "FederatedMLP",
            "hidden_dim": hidden_dim,
            "depth": params["depth"],
            "n_clients": n_clients,
            "local_steps": local_steps,
            "local_amplification": local_amplification,
            "heterogeneity_correction": heterogeneity_correction,
        },
    )


def client_drift_analysis(
    model: Any,
    clients: Any,
    lr: float = 0.01,
    n_local_steps_range: Optional[Sequence[int]] = None,
    seed: int = 42,
) -> DriftReport:
    """Analyze client drift in federated learning.

    Client drift measures how far each client's local model diverges
    from the global model during local training. Excessive drift can
    push the global model across phase boundaries.

    Parameters
    ----------
    model : dict or list of NDArray
        Model specification.
    clients : list, dict, or int
        Client data.
    lr : float
        Local learning rate.
    n_local_steps_range : sequence of int or None
        Local step counts to evaluate.
    seed : int
        Random seed.

    Returns
    -------
    DriftReport
        Client drift analysis.
    """
    params = _extract_fed_params(model)
    hidden_dim = params["hidden_dim"]
    init_scale = params["init_scale"]
    local_steps = params["local_steps"]

    weights = _initialize_weights(
        params["input_dim"], hidden_dim, params["output_dim"],
        params["depth"], init_scale, seed,
    )
    client_data = _extract_client_data(clients, params, seed)
    n_clients = len(client_data)

    # Compute drift for each client after local_steps of SGD
    drifts = np.zeros(n_clients)
    drift_directions = []
    global_params = np.concatenate([w.flatten() for w in weights])

    for c in range(n_clients):
        X, Y = client_data[c]
        local_w, _ = _local_sgd_step(weights, X, Y, lr, local_steps)
        local_params = np.concatenate([w.flatten() for w in local_w])
        diff = local_params - global_params
        drifts[c] = float(np.linalg.norm(diff))
        if np.linalg.norm(diff) > 1e-12:
            drift_directions.append(diff / np.linalg.norm(diff))
        else:
            drift_directions.append(np.zeros_like(diff))

    mean_drift = float(np.mean(drifts))
    max_drift = float(np.max(drifts))
    drift_var = float(np.var(drifts))

    # Direction alignment: average pairwise cosine similarity of drift directions
    drift_dirs = np.array(drift_directions)
    if n_clients > 1:
        alignment_sum = 0.0
        count = 0
        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                alignment_sum += float(np.dot(drift_dirs[i], drift_dirs[j]))
                count += 1
        direction_alignment = alignment_sum / count if count > 0 else 0.0
    else:
        direction_alignment = 1.0

    # Critical local steps: find where drift causes regime change
    critical_steps = local_steps
    if n_local_steps_range is None:
        n_local_steps_range = [1, 2, 5, 10, 20, 50]

    eigvals = _federated_ntk_eigenspectrum(weights, client_data, seed=seed)
    mu_max = float(eigvals[0]) / hidden_dim if len(eigvals) > 0 else 1.0
    gamma_ref = _compute_gamma(lr, init_scale, hidden_dim)
    g_star_ref = _predict_gamma_star(mu_max, 100)
    ref_regime = Regime.LAZY if gamma_ref < g_star_ref else Regime.RICH

    for steps in sorted(n_local_steps_range):
        local_amp = 1.0 + steps * 0.1 * mu_max
        g_star_s = _predict_gamma_star(mu_max * local_amp, 100)
        regime_s = Regime.LAZY if gamma_ref < g_star_s else Regime.RICH
        if regime_s != ref_regime:
            critical_steps = steps
            break

    # NTK divergence between clients
    client_ntks = [_client_ntk(weights, X) for X, _ in client_data[:min(10, n_clients)]]
    ntk_norms = [np.linalg.norm(K, "fro") for K in client_ntks]
    ntk_divergence = float(np.std(ntk_norms) / (np.mean(ntk_norms) + 1e-12))

    drift_effect: Dict[str, Any] = {
        "mean_relative_drift": mean_drift / (np.linalg.norm(global_params) + 1e-12),
        "alignment": direction_alignment,
        "effect": (
            "High drift with low alignment → gradient cancellation → lazy regime favored"
            if direction_alignment < 0.3 and mean_drift > 0.1
            else "Aligned drift → effective feature learning possible"
        ),
    }

    return DriftReport(
        mean_drift=mean_drift,
        max_drift=max_drift,
        drift_variance=drift_var,
        per_client_drift=drifts,
        drift_direction_alignment=direction_alignment,
        critical_local_steps=critical_steps,
        drift_regime_effect=drift_effect,
        ntk_divergence=ntk_divergence,
    )


def aggregation_strategy_phase(
    model: Any,
    clients: Any,
    strategies: Optional[Sequence[str]] = None,
    lr: float = 0.01,
    training_steps: int = 100,
    seed: int = 42,
) -> AggPhase:
    """Compare aggregation strategies in terms of phase behavior.

    Different aggregation strategies handle client drift differently,
    leading to different effective couplings and phase boundaries.

    Parameters
    ----------
    model : dict or list of NDArray
        Model specification.
    clients : list, dict, or int
        Client data.
    strategies : sequence of str or None
        Strategies to compare.
    lr : float
        Learning rate.
    training_steps : int
        Number of communication rounds.
    seed : int
        Random seed.

    Returns
    -------
    AggPhase
        Comparison across aggregation strategies.
    """
    params = _extract_fed_params(model)
    hidden_dim = params["hidden_dim"]
    init_scale = params["init_scale"]
    local_steps = params["local_steps"]

    weights = _initialize_weights(
        params["input_dim"], hidden_dim, params["output_dim"],
        params["depth"], init_scale, seed,
    )
    client_data = _extract_client_data(clients, params, seed)

    if strategies is None:
        strategies = ["fedavg", "fedprox", "scaffold", "fedadam"]

    strat_critical_lrs: Dict[str, float] = {}
    strat_regimes: Dict[str, Regime] = {}
    strat_convergence: Dict[str, float] = {}
    strat_drift_comp: Dict[str, float] = {}
    strat_diagrams: Dict[str, PhaseDiagram] = {}

    eigvals = _federated_ntk_eigenspectrum(weights, client_data, seed=seed)
    mu_max = float(eigvals[0]) / hidden_dim if len(eigvals) > 0 else 1.0

    # Strategy-specific corrections to the effective coupling
    strategy_corrections = {
        "fedavg": 1.0,
        "fedprox": 0.8,     # proximal term reduces effective coupling
        "scaffold": 0.6,    # control variates reduce drift more
        "fedadam": 1.2,     # adaptive LR can amplify effective coupling
    }

    best_strat = "fedavg"
    best_score = -float("inf")

    for strat in strategies:
        correction = strategy_corrections.get(strat, 1.0)
        local_amp = 1.0 + local_steps * 0.1 * mu_max * correction

        g_star = _predict_gamma_star(mu_max * local_amp, training_steps)
        critical_lr = g_star * hidden_dim / (init_scale ** 2 + 1e-12)
        strat_critical_lrs[strat] = critical_lr

        gamma = _compute_gamma(lr, init_scale, hidden_dim)
        if gamma < g_star * 0.8:
            strat_regimes[strat] = Regime.LAZY
        elif gamma > g_star * 1.2:
            strat_regimes[strat] = Regime.RICH
        else:
            strat_regimes[strat] = Regime.CRITICAL

        # Convergence rate (relative to FedAvg)
        strat_convergence[strat] = 1.0 / correction

        # Drift compensation effectiveness
        strat_drift_comp[strat] = 1.0 - correction * 0.3

        # Full phase diagram for this strategy
        model_d = {**params}
        diagram = federated_phase_diagram(
            model_d, client_data,
            lr_range=(1e-4, 0.1), n_lr_steps=15,
            training_steps=training_steps, seed=seed,
        )
        strat_diagrams[strat] = diagram

        # Score: prefer strategies that enable feature learning
        score = strat_drift_comp[strat] * strat_convergence[strat]
        if score > best_score:
            best_score = score
            best_strat = strat

    return AggPhase(
        strategies=list(strategies),
        critical_lrs=strat_critical_lrs,
        regimes=strat_regimes,
        convergence_rates=strat_convergence,
        drift_compensation=strat_drift_comp,
        best_strategy=AggregationStrategy(best_strat),
        phase_diagrams=strat_diagrams,
    )


def non_iid_phase(
    model: Any,
    clients: Any,
    heterogeneity: Optional[Sequence[float]] = None,
    lr: float = 0.01,
    training_steps: int = 100,
    seed: int = 42,
) -> NonIIDPhase:
    """Analyze phase behavior under varying data heterogeneity.

    Non-IID data causes gradient disagreement between clients, which
    reduces the effective coupling and can prevent feature learning.
    At high heterogeneity, the aggregated gradient approximates a
    random walk, pushing the system into the lazy regime.

    Parameters
    ----------
    model : dict or list of NDArray
        Model specification.
    clients : list, dict, or int
        Base client specification (heterogeneity will be varied).
    heterogeneity : sequence of float or None
        Heterogeneity levels (0 = IID, 1 = fully non-IID).
    lr : float
        Learning rate.
    training_steps : int
        Number of communication rounds.
    seed : int
        Random seed.

    Returns
    -------
    NonIIDPhase
        Phase analysis under non-IID conditions.
    """
    params = _extract_fed_params(model)
    hidden_dim = params["hidden_dim"]
    init_scale = params["init_scale"]

    if heterogeneity is None:
        heterogeneity = np.linspace(0.0, 1.0, 10)
    het_arr = np.array(sorted(heterogeneity))

    n_clients = 10
    if isinstance(clients, dict):
        n_clients = clients.get("n_clients", 10)
    elif isinstance(clients, int):
        n_clients = clients

    weights = _initialize_weights(
        params["input_dim"], hidden_dim, params["output_dim"],
        params["depth"], init_scale, seed,
    )

    critical_lrs = np.zeros(len(het_arr))
    regimes: Dict[float, Regime] = {}
    convergence_deg = np.zeros(len(het_arr))

    # Reference: IID case
    iid_data = _generate_client_data(
        n_clients, params["input_dim"], params["output_dim"],
        100, 0.0, seed,
    )
    iid_eigvals = _federated_ntk_eigenspectrum(weights, iid_data, seed=seed)
    iid_mu = float(iid_eigvals[0]) / hidden_dim if len(iid_eigvals) > 0 else 1.0

    # Compute client NTK similarity at moderate heterogeneity
    mid_data = _generate_client_data(
        n_clients, params["input_dim"], params["output_dim"],
        100, 0.5, seed,
    )
    client_ntks = [_client_ntk(weights, X) for X, _ in mid_data[:min(10, n_clients)]]
    ntk_similarity = np.zeros((min(10, n_clients), min(10, n_clients)))
    for i in range(len(client_ntks)):
        for j in range(i, len(client_ntks)):
            norm_i = np.linalg.norm(client_ntks[i], "fro") + 1e-12
            norm_j = np.linalg.norm(client_ntks[j], "fro") + 1e-12
            sim = float(np.sum(client_ntks[i] * client_ntks[j]) / (norm_i * norm_j))
            ntk_similarity[i, j] = sim
            ntk_similarity[j, i] = sim

    het_class = HeterogeneityLevel.IID
    optimal_n = n_clients

    for idx, h in enumerate(het_arr):
        h_float = float(h)
        client_data = _generate_client_data(
            n_clients, params["input_dim"], params["output_dim"],
            100, h_float, seed,
        )

        eigvals = _federated_ntk_eigenspectrum(weights, client_data, seed=seed)
        mu_max = float(eigvals[0]) / hidden_dim if len(eigvals) > 0 else 1.0

        # Heterogeneity reduces effective coupling via gradient cancellation
        het_correction = 1.0 / (1.0 + h_float * 3.0)
        local_amp = 1.0 + params["local_steps"] * 0.1 * mu_max
        g_star = _predict_gamma_star(mu_max * local_amp * het_correction, training_steps)
        critical_lr = g_star * hidden_dim / (init_scale ** 2 + 1e-12)
        critical_lrs[idx] = critical_lr

        gamma = _compute_gamma(lr, init_scale, hidden_dim)
        if gamma < g_star * 0.8:
            regimes[h_float] = Regime.LAZY
        elif gamma > g_star * 1.2:
            regimes[h_float] = Regime.RICH
        else:
            regimes[h_float] = Regime.CRITICAL

        # Convergence degradation relative to IID
        convergence_deg[idx] = 1.0 - mu_max * het_correction / (iid_mu + 1e-12)
        convergence_deg[idx] = max(0.0, min(1.0, convergence_deg[idx]))

    # Classify heterogeneity
    avg_het = float(np.mean(het_arr))
    if avg_het < 0.1:
        het_class = HeterogeneityLevel.IID
    elif avg_het < 0.3:
        het_class = HeterogeneityLevel.MILD
    elif avg_het < 0.7:
        het_class = HeterogeneityLevel.MODERATE
    else:
        het_class = HeterogeneityLevel.SEVERE

    # Optimal number of clients: more clients help with non-IID
    # but add communication cost. Heuristic: sqrt(1/heterogeneity)
    mid_het = float(het_arr[len(het_arr) // 2])
    optimal_n = max(2, int(math.sqrt(1.0 / (mid_het + 0.01)) * 5))

    return NonIIDPhase(
        heterogeneity_levels=het_arr,
        critical_lrs=critical_lrs,
        regimes=regimes,
        heterogeneity_class=het_class,
        convergence_degradation=convergence_deg,
        client_ntk_similarity=ntk_similarity,
        optimal_n_clients=optimal_n,
    )


def communication_rounds_scaling(
    model: Any,
    clients: Any,
    rounds: Optional[Sequence[int]] = None,
    lr: float = 0.01,
    seed: int = 42,
) -> CommScaling:
    """Analyze how phase behavior scales with communication rounds.

    More communication rounds allow more aggregation, which can
    stabilize the global model and shift the phase boundary. The
    trade-off between local computation and communication determines
    the optimal federated training strategy.

    Parameters
    ----------
    model : dict or list of NDArray
        Model specification.
    clients : list, dict, or int
        Client data.
    rounds : sequence of int or None
        Round counts to evaluate.
    lr : float
        Learning rate.
    seed : int
        Random seed.

    Returns
    -------
    CommScaling
        Scaling analysis of communication rounds.
    """
    params = _extract_fed_params(model)
    hidden_dim = params["hidden_dim"]
    init_scale = params["init_scale"]
    local_steps = params["local_steps"]

    weights = _initialize_weights(
        params["input_dim"], hidden_dim, params["output_dim"],
        params["depth"], init_scale, seed,
    )
    client_data = _extract_client_data(clients, params, seed)

    if rounds is None:
        rounds = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    rounds_arr = np.array(sorted(rounds))

    n_params = sum(w.size for w in weights)
    critical_lrs = np.zeros(len(rounds_arr))
    convergence = np.zeros(len(rounds_arr))
    opt_local = np.zeros(len(rounds_arr))
    comm_cost = np.zeros(len(rounds_arr))

    eigvals = _federated_ntk_eigenspectrum(weights, client_data, seed=seed)
    mu_max = float(eigvals[0]) / hidden_dim if len(eigvals) > 0 else 1.0

    for idx, r in enumerate(rounds_arr):
        r_int = int(r)
        local_amp = 1.0 + local_steps * 0.1 * mu_max
        g_star = _predict_gamma_star(mu_max * local_amp, r_int)
        critical_lr = g_star * hidden_dim / (init_scale ** 2 + 1e-12)
        critical_lrs[idx] = critical_lr

        # Convergence: approximate as 1 - exp(-rate * rounds)
        rate = mu_max * local_amp * _compute_gamma(lr, init_scale, hidden_dim)
        convergence[idx] = 1.0 - math.exp(-rate * r_int * 0.01)

        # Optimal local steps given total round budget
        # More rounds → fewer local steps needed
        opt_local[idx] = max(1, int(100 / (r_int + 1)))

        # Communication cost: n_params * 4 bytes * 2 (up + down) * n_clients
        n_clients = len(client_data)
        comm_cost[idx] = float(n_params * 4 * 2 * n_clients * r_int)

    # Pareto frontier: best convergence for each communication budget
    # Sort by cost
    cost_order = np.argsort(comm_cost)
    pareto_points = []
    best_conv = -1.0
    for i in cost_order:
        if convergence[i] > best_conv:
            pareto_points.append([float(rounds_arr[i]), convergence[i]])
            best_conv = convergence[i]
    pareto_frontier = np.array(pareto_points) if pareto_points else np.array([])

    # Scaling exponent: eta* ~ R^{-alpha}
    valid = critical_lrs > 1e-12
    if np.sum(valid) > 2:
        log_r = np.log(rounds_arr[valid].astype(float))
        log_lr = np.log(critical_lrs[valid])
        coeffs = np.polyfit(log_r, log_lr, 1)
        scaling_exp = -float(coeffs[0])
    else:
        scaling_exp = 0.0

    return CommScaling(
        rounds=rounds_arr.astype(float),
        critical_lrs=critical_lrs,
        convergence_progress=convergence,
        local_steps_per_round=opt_local,
        communication_cost=comm_cost,
        pareto_frontier=pareto_frontier,
        scaling_exponent=scaling_exp,
    )
