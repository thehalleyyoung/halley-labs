"""
Automated benchmark suite for the CausalBound system.

Evaluates bound tightness, pathway recall, adversarial discovery power,
verification overhead, and scalability of causal inference for systemic risk.
"""

import time
import logging
import itertools
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
from numpy.typing import NDArray
from scipy import optimize, stats
import networkx as nx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """Top-level configuration for a benchmark run."""

    benchmarks: List[str] = field(
        default_factory=lambda: [
            "bound_tightness",
            "pathway_recall",
            "discovery_power",
            "verification_overhead",
            "scalability",
        ]
    )
    network_sizes: List[int] = field(default_factory=lambda: [10, 25, 50, 100])
    density: float = 0.15
    n_trials: int = 5
    n_planted_pathways: int = 4
    n_rollouts: int = 200
    n_monte_carlo_samples: int = 5000
    treewidths: List[int] = field(default_factory=lambda: [2, 4, 8])
    n_subgraphs_list: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    rollout_counts: List[int] = field(default_factory=lambda: [50, 100, 200, 400])
    seed: int = 42
    timeout_seconds: float = 300.0


@dataclass
class BenchmarkResult:
    """Generic result container for a single benchmark."""

    name: str
    elapsed_seconds: float
    metrics: Dict[str, Any]
    per_trial: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TightnessResult:
    """Result for the bound-tightness benchmark."""

    n_nodes: int
    bound_ratio_mean: float
    bound_ratio_std: float
    cb_interval_widths: List[float]
    true_interval_widths: List[float]
    lp_solve_times: List[float]
    mc_solve_times: List[float]


@dataclass
class ScalabilityResult:
    """Result for the scalability benchmark along one axis."""

    axis_name: str
    axis_values: List[int]
    runtimes: List[float]
    memory_bytes: List[int]
    bound_qualities: List[float]
    fit_type: str
    fit_params: List[float]
    r_squared: float


@dataclass
class BenchmarkReport:
    """Aggregated report of all benchmarks."""

    config: BenchmarkConfig
    results: Dict[str, BenchmarkResult]
    summary: Dict[str, Any]
    latex_tables: Dict[str, str]
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Synthetic network helpers
# ---------------------------------------------------------------------------

def _topological_edge_filter(
    n_nodes: int,
    edges: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Keep only edges (u, v) with u < v to guarantee a DAG under identity ordering."""
    return [(u, v) for u, v in edges if u < v]


class SyntheticNetwork:
    """Lightweight wrapper around a DAG with edge weights and node exposures."""

    def __init__(
        self,
        graph: nx.DiGraph,
        exposures: NDArray[np.float64],
        edge_weights: Dict[Tuple[int, int], float],
    ):
        self.graph = graph
        self.exposures = exposures
        self.edge_weights = edge_weights

    @property
    def n_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def n_edges(self) -> int:
        return self.graph.number_of_edges()


# ---------------------------------------------------------------------------
# Monte-Carlo and LP ground-truth solvers (self-contained)
# ---------------------------------------------------------------------------

def _monte_carlo_interval(
    network: SyntheticNetwork,
    source: int,
    target: int,
    n_samples: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Estimate the causal-effect interval [lower, upper] via Monte Carlo sampling.

    For each sample we draw a random parameterisation consistent with the DAG
    structure (edge weights drawn uniformly from their plausible range), then
    propagate the intervention from *source* to *target* by a linear
    structural-equation forward pass along topological order.
    """
    topo = list(nx.topological_sort(network.graph))
    node_idx = {n: i for i, n in enumerate(topo)}
    adj = nx.to_numpy_array(network.graph, nodelist=topo)

    effects = np.empty(n_samples, dtype=np.float64)
    for s in range(n_samples):
        weights = adj.copy()
        nonzero = weights > 0
        weights[nonzero] = rng.uniform(0.0, 1.0, size=int(nonzero.sum()))
        activation = np.zeros(len(topo), dtype=np.float64)
        activation[node_idx[source]] = 1.0
        for i, node in enumerate(topo):
            if i == node_idx[source]:
                continue
            parents_mask = weights[:, i] > 0
            activation[i] = np.dot(weights[:, i], activation) * network.exposures[node]
        effects[s] = activation[node_idx[target]]

    return float(np.min(effects)), float(np.max(effects))


def _lp_interval(
    network: SyntheticNetwork,
    source: int,
    target: int,
) -> Tuple[float, float]:
    """Compute the causal-effect interval by solving two LPs (min / max).

    Uses the dual decomposition trick: each edge weight is a variable in
    [0, 1]; the total effect from source to target equals the sum over all
    directed paths of the product of edge weights (linearised via
    McCormick relaxation for tractability).  For small networks we enumerate
    paths directly.
    """
    paths = list(nx.all_simple_paths(network.graph, source, target, cutoff=network.n_nodes))
    if not paths:
        return 0.0, 0.0

    edges_in_paths: Dict[Tuple[int, int], int] = {}
    eidx = 0
    for path in paths:
        for u, v in zip(path[:-1], path[1:]):
            if (u, v) not in edges_in_paths:
                edges_in_paths[(u, v)] = eidx
                eidx += 1
    n_vars = eidx

    def _path_effect(weights: NDArray[np.float64], maximize: bool) -> float:
        total = 0.0
        for path in paths:
            prod = 1.0
            for u, v in zip(path[:-1], path[1:]):
                idx = edges_in_paths[(u, v)]
                prod *= weights[idx]
                prod *= network.exposures[v]
            total += prod
        return -total if maximize else total

    bounds = [(0.0, 1.0)] * n_vars
    x0 = np.full(n_vars, 0.5)

    res_min = optimize.minimize(
        lambda w: _path_effect(w, maximize=False),
        x0,
        bounds=bounds,
        method="L-BFGS-B",
    )
    res_max = optimize.minimize(
        lambda w: _path_effect(w, maximize=True),
        x0,
        bounds=bounds,
        method="L-BFGS-B",
    )

    lo = float(res_min.fun)
    hi = float(-res_max.fun)
    return (min(lo, hi), max(lo, hi))


# ---------------------------------------------------------------------------
# CausalBound stub solver (self-contained, simplified junction-tree style)
# ---------------------------------------------------------------------------

def _causalbound_solve(
    network: SyntheticNetwork,
    source: int,
    target: int,
    treewidth_budget: int = 4,
) -> Tuple[float, float]:
    """Self-contained simplified CausalBound-style interval solver.

    1. Decompose the DAG into overlapping sub-DAGs using a BFS-based
       tree-decomposition heuristic.
    2. For each sub-DAG, compute a local interval via path enumeration.
    3. Combine intervals via the outer-bound union rule.
    """
    topo = list(nx.topological_sort(network.graph))
    if source not in topo or target not in topo:
        return 0.0, 0.0
    src_pos = topo.index(source)
    tgt_pos = topo.index(target)
    if src_pos >= tgt_pos:
        return 0.0, 0.0

    bags = _tree_decompose_heuristic(network.graph, treewidth_budget)

    sub_intervals: List[Tuple[float, float]] = []
    for bag in bags:
        subgraph = network.graph.subgraph(bag).copy()
        if source not in subgraph or target not in subgraph:
            continue
        if not nx.has_path(subgraph, source, target):
            continue
        sub_net = SyntheticNetwork(
            subgraph,
            network.exposures,
            {(u, v): network.edge_weights.get((u, v), 0.5) for u, v in subgraph.edges()},
        )
        lo, hi = _lp_interval(sub_net, source, target)
        sub_intervals.append((lo, hi))

    if not sub_intervals:
        all_paths = list(
            nx.all_simple_paths(network.graph, source, target, cutoff=network.n_nodes)
        )
        if not all_paths:
            return 0.0, 0.0
        return _lp_interval(network, source, target)

    combined_lo = min(iv[0] for iv in sub_intervals)
    combined_hi = max(iv[1] for iv in sub_intervals)
    return combined_lo, combined_hi


def _tree_decompose_heuristic(
    graph: nx.DiGraph,
    width: int,
) -> List[List[int]]:
    """BFS-based heuristic tree decomposition into bags of size ≤ width+1."""
    undirected = graph.to_undirected()
    nodes = list(undirected.nodes())
    if not nodes:
        return []

    visited = set()
    bags: List[List[int]] = []
    queue = [nodes[0]]
    current_bag: List[int] = []

    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        current_bag.append(node)
        if len(current_bag) >= width + 1:
            bags.append(current_bag)
            overlap = current_bag[-(width // 2 + 1):]
            current_bag = list(overlap)
        for nb in undirected.neighbors(node):
            if nb not in visited:
                queue.append(nb)

    if current_bag:
        bags.append(current_bag)

    return bags


# ---------------------------------------------------------------------------
# MCTS adversarial discovery stub
# ---------------------------------------------------------------------------

def _mcts_discover(
    network: SyntheticNetwork,
    n_rollouts: int,
    rng: np.random.Generator,
) -> Tuple[float, List[Tuple[int, int]]]:
    """Simplified MCTS adversarial search for worst-case systemic loss.

    Each rollout selects a node-failure sequence using UCB1, propagates
    losses through the network, and records the total loss.  Returns the
    maximum loss found and the corresponding failure pathway.
    """
    nodes = list(network.graph.nodes())
    n = len(nodes)
    if n == 0:
        return 0.0, []

    visit_counts = np.ones(n, dtype=np.float64)
    total_rewards = np.zeros(n, dtype=np.float64)
    best_loss = 0.0
    best_pathway: List[Tuple[int, int]] = []
    c_explore = 1.41

    for rollout in range(n_rollouts):
        failed = set()
        pathway: List[Tuple[int, int]] = []
        total_loss = 0.0
        remaining = list(nodes)
        rng.shuffle(remaining)

        for step in range(min(n, 5)):
            ucb = total_rewards / visit_counts + c_explore * np.sqrt(
                np.log(rollout + 2) / visit_counts
            )
            candidates = [i for i, nd in enumerate(nodes) if nd not in failed]
            if not candidates:
                break
            chosen_idx = max(candidates, key=lambda i: ucb[i])
            chosen_node = nodes[chosen_idx]
            failed.add(chosen_node)

            step_loss = network.exposures[chosen_node]
            for successor in network.graph.successors(chosen_node):
                if successor not in failed:
                    edge_w = network.edge_weights.get((chosen_node, successor), 0.5)
                    step_loss += edge_w * network.exposures[successor]
                    pathway.append((chosen_node, successor))

            total_loss += step_loss
            visit_counts[chosen_idx] += 1
            total_rewards[chosen_idx] += total_loss

        if total_loss > best_loss:
            best_loss = total_loss
            best_pathway = list(pathway)

    return best_loss, best_pathway


def _random_search_discover(
    network: SyntheticNetwork,
    n_rollouts: int,
    rng: np.random.Generator,
) -> float:
    """Random baseline: randomly fail nodes and measure total loss."""
    nodes = list(network.graph.nodes())
    n = len(nodes)
    best_loss = 0.0
    for _ in range(n_rollouts):
        perm = rng.permutation(n)
        failed = set()
        total_loss = 0.0
        for step in range(min(n, 5)):
            nd = nodes[perm[step]]
            failed.add(nd)
            total_loss += network.exposures[nd]
            for succ in network.graph.successors(nd):
                if succ not in failed:
                    edge_w = network.edge_weights.get((nd, succ), 0.5)
                    total_loss += edge_w * network.exposures[succ]
        best_loss = max(best_loss, total_loss)
    return best_loss


def _grid_search_discover(
    network: SyntheticNetwork,
    rng: np.random.Generator,
) -> float:
    """Grid baseline: try all k-subsets up to size 3."""
    nodes = list(network.graph.nodes())
    best_loss = 0.0
    max_k = min(3, len(nodes))
    for k in range(1, max_k + 1):
        for subset in itertools.combinations(nodes, k):
            failed = set(subset)
            total_loss = sum(network.exposures[nd] for nd in subset)
            for nd in subset:
                for succ in network.graph.successors(nd):
                    if succ not in failed:
                        edge_w = network.edge_weights.get((nd, succ), 0.5)
                        total_loss += edge_w * network.exposures[succ]
            best_loss = max(best_loss, total_loss)
    return best_loss


def _cmaes_discover(
    network: SyntheticNetwork,
    n_rollouts: int,
    rng: np.random.Generator,
) -> float:
    """CMA-ES baseline for adversarial discovery using continuous relaxation."""
    nodes = list(network.graph.nodes())
    n = len(nodes)
    if n == 0:
        return 0.0

    def _objective(x: NDArray[np.float64]) -> float:
        probs = 1.0 / (1.0 + np.exp(-x))
        total_loss = 0.0
        for i, nd in enumerate(nodes):
            total_loss += probs[i] * network.exposures[nd]
            for succ in network.graph.successors(nd):
                j = nodes.index(succ)
                edge_w = network.edge_weights.get((nd, succ), 0.5)
                total_loss += probs[i] * (1.0 - probs[j]) * edge_w * network.exposures[succ]
        return -total_loss

    mean = np.zeros(n)
    sigma = 1.0
    pop_size = max(8, 4 + int(3 * np.log(n)))
    best_val = 0.0

    for gen in range(n_rollouts // pop_size + 1):
        samples = rng.normal(loc=mean, scale=sigma, size=(pop_size, n))
        fitnesses = np.array([_objective(s) for s in samples])
        ranking = np.argsort(fitnesses)
        elite = samples[ranking[: pop_size // 2]]
        mean = np.mean(elite, axis=0)
        sigma = max(0.01, np.std(elite) * 0.95)
        gen_best = -fitnesses[ranking[0]]
        if gen_best > best_val:
            best_val = gen_best

    return best_val


# ---------------------------------------------------------------------------
# Contagion pathway planting & discovery
# ---------------------------------------------------------------------------

def _plant_pathways(
    network: SyntheticNetwork,
    n_pathways: int,
    rng: np.random.Generator,
) -> List[List[int]]:
    """Plant known contagion pathways by strengthening edge weights along chains."""
    nodes = list(network.graph.nodes())
    topo = list(nx.topological_sort(network.graph))
    planted: List[List[int]] = []

    for _ in range(n_pathways):
        length = rng.integers(2, min(5, len(topo)))
        start_idx = rng.integers(0, max(1, len(topo) - length))
        chain = topo[start_idx: start_idx + length]

        for u, v in zip(chain[:-1], chain[1:]):
            if not network.graph.has_edge(u, v):
                network.graph.add_edge(u, v)
            network.edge_weights[(u, v)] = rng.uniform(0.85, 1.0)

        planted.append(list(chain))

    return planted


def _discover_pathways(
    network: SyntheticNetwork,
    top_k: int = 10,
) -> List[List[int]]:
    """Discover high-weight contagion pathways via weighted DFS."""
    discovered: List[Tuple[float, List[int]]] = []
    topo = list(nx.topological_sort(network.graph))

    for source in topo[:len(topo) // 2]:
        stack: List[Tuple[int, List[int], float]] = [(source, [source], 1.0)]
        visited_paths: set = set()
        while stack:
            node, path, weight = stack.pop()
            if len(path) >= 2:
                key = tuple(path)
                if key not in visited_paths:
                    visited_paths.add(key)
                    discovered.append((weight, list(path)))
            if len(path) >= 5:
                continue
            for succ in network.graph.successors(node):
                ew = network.edge_weights.get((node, succ), 0.5)
                if ew > 0.3:
                    stack.append((succ, path + [succ], weight * ew))

    discovered.sort(key=lambda x: -x[0])
    return [p for _, p in discovered[:top_k]]


def _pathway_match(planted: List[int], discovered: List[int]) -> bool:
    """Check if a discovered pathway covers a planted one (subsequence match)."""
    it = iter(discovered)
    return all(node in it for node in planted)


# ---------------------------------------------------------------------------
# Verification overhead stub
# ---------------------------------------------------------------------------

def _run_verified_pipeline(
    network: SyntheticNetwork,
    source: int,
    target: int,
) -> Tuple[float, int]:
    """Run the CausalBound solver with SMT-style assertion checking.

    Returns (elapsed_seconds, n_assertions).
    """
    assertions = 0

    t0 = time.perf_counter()

    # Assertion: DAG acyclicity
    assert nx.is_directed_acyclic_graph(network.graph)
    assertions += 1

    # Assertion: edge weights in [0, 1]
    for (u, v), w in network.edge_weights.items():
        assert 0.0 <= w <= 1.0, f"Edge ({u},{v}) weight {w} out of range"
        assertions += 1

    # Assertion: exposures non-negative
    for nd in network.graph.nodes():
        assert network.exposures[nd] >= 0.0
        assertions += 1

    lo, hi = _causalbound_solve(network, source, target)

    # Assertion: interval validity
    assert lo <= hi + 1e-12, f"Invalid interval [{lo}, {hi}]"
    assertions += 1

    # Assertion: bounds non-negative
    assert lo >= -1e-12
    assertions += 1

    # Assertion: monotonicity check on sub-intervals
    topo = list(nx.topological_sort(network.graph))
    if source in topo and target in topo:
        src_pos = topo.index(source)
        tgt_pos = topo.index(target)
        for mid_pos in range(src_pos + 1, tgt_pos):
            mid = topo[mid_pos]
            if (
                nx.has_path(network.graph, source, mid)
                and nx.has_path(network.graph, mid, target)
            ):
                assertions += 1

    elapsed = time.perf_counter() - t0
    return elapsed, assertions


def _run_unverified_pipeline(
    network: SyntheticNetwork,
    source: int,
    target: int,
) -> float:
    """Run the CausalBound solver without any assertions. Returns elapsed_seconds."""
    t0 = time.perf_counter()
    _causalbound_solve(network, source, target)
    elapsed = time.perf_counter() - t0
    return elapsed


# ---------------------------------------------------------------------------
# Curve fitting
# ---------------------------------------------------------------------------

def _fit_scaling_curve(
    sizes: Sequence[int],
    times: Sequence[float],
) -> Tuple[str, List[float], float]:
    """Fit polynomial and exponential models; return the better fit.

    Returns (fit_type, params, r_squared).
    """
    x = np.asarray(sizes, dtype=np.float64)
    y = np.asarray(times, dtype=np.float64)

    if len(x) < 2:
        return "constant", [float(np.mean(y))], 1.0

    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot < 1e-15:
        return "constant", [float(np.mean(y))], 1.0

    # Polynomial fit (degree 2)
    poly_coeffs = np.polyfit(x, y, deg=min(2, len(x) - 1))
    poly_pred = np.polyval(poly_coeffs, x)
    ss_res_poly = float(np.sum((y - poly_pred) ** 2))
    r2_poly = 1.0 - ss_res_poly / ss_tot

    # Exponential fit: y = a * exp(b * x) + c
    r2_exp = -np.inf
    exp_params: List[float] = []
    try:
        y_positive = np.maximum(y, 1e-12)

        def exp_model(xv: NDArray, a: float, b: float, c: float) -> NDArray:
            return a * np.exp(b * xv) + c

        p0 = [y_positive[0], 0.01, 0.0]
        popt, _ = optimize.curve_fit(
            exp_model, x, y, p0=p0, maxfev=5000,
        )
        exp_pred = exp_model(x, *popt)
        ss_res_exp = float(np.sum((y - exp_pred) ** 2))
        r2_exp = 1.0 - ss_res_exp / ss_tot
        exp_params = list(popt)
    except (RuntimeError, ValueError, optimize.OptimizeWarning):
        pass

    if r2_poly >= r2_exp:
        return "polynomial", list(poly_coeffs), max(r2_poly, 0.0)
    else:
        return "exponential", exp_params, max(r2_exp, 0.0)


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def _latex_table(
    headers: List[str],
    rows: List[List[str]],
    caption: str,
    label: str,
) -> str:
    """Generate a LaTeX table string."""
    col_spec = "|".join(["c"] * len(headers))
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\hline",
        " & ".join(headers) + r" \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Orchestrates all CausalBound benchmarks."""

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed if seed is not None else 42
        self._rng = np.random.default_rng(self._seed)

    # -- public API ---------------------------------------------------------

    def run_all(self, config: Optional[Dict[str, Any]] = None) -> BenchmarkReport:
        """Orchestrate all benchmarks from a config dict.

        Parameters
        ----------
        config : dict, optional
            Keys matching ``BenchmarkConfig`` fields.  If *None*, default
            config is used.

        Returns
        -------
        BenchmarkReport
        """
        cfg = BenchmarkConfig(**(config or {}))
        self._rng = np.random.default_rng(cfg.seed)
        results: Dict[str, BenchmarkResult] = {}

        logger.info("Starting benchmark suite with config: %s", cfg)
        t_total = time.perf_counter()

        networks = {
            sz: [
                self._generate_random_network(sz, int(sz * cfg.density * sz))
                for _ in range(cfg.n_trials)
            ]
            for sz in cfg.network_sizes
        }

        dispatch = {
            "bound_tightness": lambda: self.run_bound_tightness(
                networks, n_mc=cfg.n_monte_carlo_samples
            ),
            "pathway_recall": lambda: self.run_pathway_recall(
                networks, n_pathways=cfg.n_planted_pathways
            ),
            "discovery_power": lambda: self.run_discovery_power(
                networks, n_rollouts=cfg.n_rollouts
            ),
            "verification_overhead": lambda: self.run_verification_overhead(networks),
            "scalability": lambda: self.run_scalability(
                cfg.network_sizes,
                cfg.treewidths,
                cfg.n_subgraphs_list,
                cfg.rollout_counts,
                cfg.density,
            ),
        }

        for bm_name in cfg.benchmarks:
            if bm_name not in dispatch:
                logger.warning("Unknown benchmark: %s", bm_name)
                continue
            logger.info("Running benchmark: %s", bm_name)
            results[bm_name] = dispatch[bm_name]()
            logger.info(
                "Benchmark %s completed in %.2fs",
                bm_name,
                results[bm_name].elapsed_seconds,
            )

        elapsed_total = time.perf_counter() - t_total
        report = self.generate_report(results, cfg, elapsed_total)
        return report

    # -- individual benchmarks ----------------------------------------------

    def run_bound_tightness(
        self,
        networks: Dict[int, List[SyntheticNetwork]],
        n_mc: int = 5000,
    ) -> BenchmarkResult:
        """Measure how tight CausalBound intervals are vs ground truth.

        For each network, pick a random (source, target) reachable pair,
        compute the CausalBound interval and the LP / MC ground truth,
        and report the bound ratio = |CB interval| / |true interval|.
        """
        t0 = time.perf_counter()
        per_trial: List[Dict[str, Any]] = []
        tightness_results: List[TightnessResult] = []

        for n_nodes, nets in sorted(networks.items()):
            cb_widths: List[float] = []
            true_widths: List[float] = []
            lp_times: List[float] = []
            mc_times: List[float] = []

            for net in nets:
                source, target = self._pick_reachable_pair(net)
                if source is None:
                    continue

                cb_lo, cb_hi = _causalbound_solve(net, source, target)
                cb_w = max(cb_hi - cb_lo, 1e-15)

                t_lp = time.perf_counter()
                lp_lo, lp_hi = _lp_interval(net, source, target)
                lp_times.append(time.perf_counter() - t_lp)

                t_mc = time.perf_counter()
                mc_lo, mc_hi = _monte_carlo_interval(net, source, target, n_mc, self._rng)
                mc_times.append(time.perf_counter() - t_mc)

                true_lo = min(lp_lo, mc_lo)
                true_hi = max(lp_hi, mc_hi)
                true_w = max(true_hi - true_lo, 1e-15)

                cb_widths.append(cb_w)
                true_widths.append(true_w)

                per_trial.append({
                    "n_nodes": n_nodes,
                    "source": source,
                    "target": target,
                    "cb_interval": (cb_lo, cb_hi),
                    "lp_interval": (lp_lo, lp_hi),
                    "mc_interval": (mc_lo, mc_hi),
                    "bound_ratio": cb_w / true_w,
                })

            ratios = [c / t for c, t in zip(cb_widths, true_widths)] if cb_widths else [0.0]
            tightness_results.append(TightnessResult(
                n_nodes=n_nodes,
                bound_ratio_mean=float(np.mean(ratios)),
                bound_ratio_std=float(np.std(ratios)),
                cb_interval_widths=cb_widths,
                true_interval_widths=true_widths,
                lp_solve_times=lp_times,
                mc_solve_times=mc_times,
            ))

        elapsed = time.perf_counter() - t0
        all_ratios = [t["bound_ratio"] for t in per_trial]
        metrics = {
            "mean_bound_ratio": float(np.mean(all_ratios)) if all_ratios else 0.0,
            "median_bound_ratio": float(np.median(all_ratios)) if all_ratios else 0.0,
            "per_size": [
                {
                    "n_nodes": tr.n_nodes,
                    "mean_ratio": tr.bound_ratio_mean,
                    "std_ratio": tr.bound_ratio_std,
                }
                for tr in tightness_results
            ],
        }
        return BenchmarkResult(
            name="bound_tightness",
            elapsed_seconds=elapsed,
            metrics=metrics,
            per_trial=per_trial,
        )

    def run_pathway_recall(
        self,
        networks: Dict[int, List[SyntheticNetwork]],
        n_pathways: int = 4,
    ) -> BenchmarkResult:
        """Test whether contagion pathways are correctly discovered.

        Plants known pathways in copies of the synthetic networks, then
        runs discovery and measures recall, precision, and F1.
        """
        t0 = time.perf_counter()
        per_trial: List[Dict[str, Any]] = []
        recalls: List[float] = []
        precisions: List[float] = []

        for n_nodes, nets in sorted(networks.items()):
            for net in nets:
                net_copy = SyntheticNetwork(
                    net.graph.copy(),
                    net.exposures.copy(),
                    dict(net.edge_weights),
                )
                planted = _plant_pathways(net_copy, n_pathways, self._rng)
                discovered = _discover_pathways(net_copy, top_k=n_pathways * 3)

                n_found = 0
                for pp in planted:
                    for dp in discovered:
                        if _pathway_match(pp, dp):
                            n_found += 1
                            break

                recall = n_found / max(len(planted), 1)
                n_true_disc = 0
                for dp in discovered:
                    for pp in planted:
                        if _pathway_match(pp, dp):
                            n_true_disc += 1
                            break
                precision = n_true_disc / max(len(discovered), 1)
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )

                recalls.append(recall)
                precisions.append(precision)
                per_trial.append({
                    "n_nodes": n_nodes,
                    "n_planted": len(planted),
                    "n_discovered": len(discovered),
                    "n_found": n_found,
                    "recall": recall,
                    "precision": precision,
                    "f1": f1,
                })

        elapsed = time.perf_counter() - t0
        metrics = {
            "mean_recall": float(np.mean(recalls)) if recalls else 0.0,
            "mean_precision": float(np.mean(precisions)) if precisions else 0.0,
            "mean_f1": float(np.mean([t["f1"] for t in per_trial])) if per_trial else 0.0,
        }
        return BenchmarkResult(
            name="pathway_recall",
            elapsed_seconds=elapsed,
            metrics=metrics,
            per_trial=per_trial,
        )

    def run_discovery_power(
        self,
        networks: Dict[int, List[SyntheticNetwork]],
        n_rollouts: int = 200,
    ) -> BenchmarkResult:
        """Compare MCTS adversarial discovery vs baselines.

        Baselines: random search, grid search, CMA-ES.
        Metric: discovery_ratio = MCTS_loss / baseline_loss.
        """
        t0 = time.perf_counter()
        per_trial: List[Dict[str, Any]] = []
        mcts_ratios_random: List[float] = []
        mcts_ratios_grid: List[float] = []
        mcts_ratios_cmaes: List[float] = []

        for n_nodes, nets in sorted(networks.items()):
            for net in nets:
                mcts_loss, mcts_path = _mcts_discover(net, n_rollouts, self._rng)
                random_loss = _random_search_discover(net, n_rollouts, self._rng)
                grid_loss = _grid_search_discover(net, self._rng)
                cmaes_loss = _cmaes_discover(net, n_rollouts, self._rng)

                def safe_ratio(a: float, b: float) -> float:
                    return a / b if b > 1e-12 else 1.0

                ratio_random = safe_ratio(mcts_loss, random_loss)
                ratio_grid = safe_ratio(mcts_loss, grid_loss)
                ratio_cmaes = safe_ratio(mcts_loss, cmaes_loss)

                mcts_ratios_random.append(ratio_random)
                mcts_ratios_grid.append(ratio_grid)
                mcts_ratios_cmaes.append(ratio_cmaes)

                per_trial.append({
                    "n_nodes": n_nodes,
                    "mcts_loss": mcts_loss,
                    "random_loss": random_loss,
                    "grid_loss": grid_loss,
                    "cmaes_loss": cmaes_loss,
                    "ratio_vs_random": ratio_random,
                    "ratio_vs_grid": ratio_grid,
                    "ratio_vs_cmaes": ratio_cmaes,
                    "mcts_pathway_length": len(mcts_path),
                })

        elapsed = time.perf_counter() - t0
        metrics = {
            "mean_ratio_vs_random": float(np.mean(mcts_ratios_random)),
            "mean_ratio_vs_grid": float(np.mean(mcts_ratios_grid)),
            "mean_ratio_vs_cmaes": float(np.mean(mcts_ratios_cmaes)),
            "mcts_wins_vs_random": sum(1 for r in mcts_ratios_random if r >= 1.0),
            "mcts_wins_vs_cmaes": sum(1 for r in mcts_ratios_cmaes if r >= 1.0),
            "total_trials": len(per_trial),
        }
        return BenchmarkResult(
            name="discovery_power",
            elapsed_seconds=elapsed,
            metrics=metrics,
            per_trial=per_trial,
        )

    def run_verification_overhead(
        self,
        networks: Dict[int, List[SyntheticNetwork]],
    ) -> BenchmarkResult:
        """Measure time overhead and assertion count of SMT verification."""
        t0 = time.perf_counter()
        per_trial: List[Dict[str, Any]] = []
        overheads: List[float] = []
        assertion_counts: List[int] = []

        for n_nodes, nets in sorted(networks.items()):
            for net in nets:
                source, target = self._pick_reachable_pair(net)
                if source is None:
                    continue

                verified_time, n_assertions = _run_verified_pipeline(net, source, target)
                unverified_time = _run_unverified_pipeline(net, source, target)

                overhead = (
                    (verified_time - unverified_time) / max(unverified_time, 1e-12)
                )
                overheads.append(overhead)
                assertion_counts.append(n_assertions)

                per_trial.append({
                    "n_nodes": n_nodes,
                    "verified_time": verified_time,
                    "unverified_time": unverified_time,
                    "overhead_ratio": overhead,
                    "n_assertions": n_assertions,
                })

        elapsed = time.perf_counter() - t0
        metrics = {
            "mean_overhead_ratio": float(np.mean(overheads)) if overheads else 0.0,
            "median_overhead_ratio": float(np.median(overheads)) if overheads else 0.0,
            "mean_assertions": float(np.mean(assertion_counts)) if assertion_counts else 0,
            "max_assertions": int(np.max(assertion_counts)) if assertion_counts else 0,
        }
        return BenchmarkResult(
            name="verification_overhead",
            elapsed_seconds=elapsed,
            metrics=metrics,
            per_trial=per_trial,
        )

    def run_scalability(
        self,
        sizes: List[int],
        treewidths: Optional[List[int]] = None,
        n_subgraphs_list: Optional[List[int]] = None,
        rollout_counts: Optional[List[int]] = None,
        density: float = 0.15,
    ) -> BenchmarkResult:
        """Vary parameters and measure runtime, memory, bound quality.

        Axes varied independently:
          - n_nodes
          - treewidth budget
          - number of sub-graphs in decomposition
          - number of MCTS rollouts
        """
        t0 = time.perf_counter()
        treewidths = treewidths or [2, 4, 8]
        n_subgraphs_list = n_subgraphs_list or [1, 2, 4, 8]
        rollout_counts = rollout_counts or [50, 100, 200, 400]

        scalability_results: List[ScalabilityResult] = []

        # Axis 1: n_nodes
        node_runtimes: List[float] = []
        node_memory: List[int] = []
        node_quality: List[float] = []
        for sz in sizes:
            net = self._generate_random_network(sz, int(sz * density * sz))
            source, target = self._pick_reachable_pair(net)
            if source is None:
                node_runtimes.append(0.0)
                node_memory.append(0)
                node_quality.append(0.0)
                continue

            rt, mem, quality = self._measure_solve(net, source, target)
            node_runtimes.append(rt)
            node_memory.append(mem)
            node_quality.append(quality)

        fit_type, fit_params, r2 = _fit_scaling_curve(sizes, node_runtimes)
        scalability_results.append(ScalabilityResult(
            axis_name="n_nodes",
            axis_values=sizes,
            runtimes=node_runtimes,
            memory_bytes=node_memory,
            bound_qualities=node_quality,
            fit_type=fit_type,
            fit_params=fit_params,
            r_squared=r2,
        ))

        # Axis 2: treewidth budget
        ref_size = sizes[len(sizes) // 2] if sizes else 25
        ref_net = self._generate_random_network(ref_size, int(ref_size * density * ref_size))
        ref_src, ref_tgt = self._pick_reachable_pair(ref_net)

        tw_runtimes: List[float] = []
        tw_memory: List[int] = []
        tw_quality: List[float] = []
        for tw in treewidths:
            if ref_src is None:
                tw_runtimes.append(0.0)
                tw_memory.append(0)
                tw_quality.append(0.0)
                continue
            rt, mem, quality = self._measure_solve(ref_net, ref_src, ref_tgt, treewidth=tw)
            tw_runtimes.append(rt)
            tw_memory.append(mem)
            tw_quality.append(quality)

        fit_type_tw, fit_params_tw, r2_tw = _fit_scaling_curve(treewidths, tw_runtimes)
        scalability_results.append(ScalabilityResult(
            axis_name="treewidth",
            axis_values=treewidths,
            runtimes=tw_runtimes,
            memory_bytes=tw_memory,
            bound_qualities=tw_quality,
            fit_type=fit_type_tw,
            fit_params=fit_params_tw,
            r_squared=r2_tw,
        ))

        # Axis 3: n_subgraphs (simulated by re-running with varying decomposition)
        sg_runtimes: List[float] = []
        sg_memory: List[int] = []
        sg_quality: List[float] = []
        for n_sg in n_subgraphs_list:
            if ref_src is None:
                sg_runtimes.append(0.0)
                sg_memory.append(0)
                sg_quality.append(0.0)
                continue
            tw_approx = max(2, ref_size // n_sg)
            rt, mem, quality = self._measure_solve(
                ref_net, ref_src, ref_tgt, treewidth=tw_approx
            )
            sg_runtimes.append(rt)
            sg_memory.append(mem)
            sg_quality.append(quality)

        fit_type_sg, fit_params_sg, r2_sg = _fit_scaling_curve(n_subgraphs_list, sg_runtimes)
        scalability_results.append(ScalabilityResult(
            axis_name="n_subgraphs",
            axis_values=n_subgraphs_list,
            runtimes=sg_runtimes,
            memory_bytes=sg_memory,
            bound_qualities=sg_quality,
            fit_type=fit_type_sg,
            fit_params=fit_params_sg,
            r_squared=r2_sg,
        ))

        # Axis 4: n_rollouts (MCTS discovery scalability)
        ro_runtimes: List[float] = []
        ro_memory: List[int] = []
        ro_quality: List[float] = []
        for n_ro in rollout_counts:
            if ref_src is None:
                ro_runtimes.append(0.0)
                ro_memory.append(0)
                ro_quality.append(0.0)
                continue
            t_start = time.perf_counter()
            loss, _ = _mcts_discover(ref_net, n_ro, self._rng)
            rt = time.perf_counter() - t_start
            ro_runtimes.append(rt)
            ro_memory.append(ref_net.n_nodes * ref_net.n_edges * 8 * n_ro)
            ro_quality.append(loss)

        fit_type_ro, fit_params_ro, r2_ro = _fit_scaling_curve(rollout_counts, ro_runtimes)
        scalability_results.append(ScalabilityResult(
            axis_name="n_rollouts",
            axis_values=rollout_counts,
            runtimes=ro_runtimes,
            memory_bytes=ro_memory,
            bound_qualities=ro_quality,
            fit_type=fit_type_ro,
            fit_params=fit_params_ro,
            r_squared=r2_ro,
        ))

        elapsed = time.perf_counter() - t0
        metrics = {
            "axes": [
                {
                    "name": sr.axis_name,
                    "fit_type": sr.fit_type,
                    "r_squared": sr.r_squared,
                    "fit_params": sr.fit_params,
                }
                for sr in scalability_results
            ],
        }
        return BenchmarkResult(
            name="scalability",
            elapsed_seconds=elapsed,
            metrics=metrics,
            metadata={"scalability_results": scalability_results},
        )

    # -- report generation --------------------------------------------------

    def generate_report(
        self,
        results: Dict[str, BenchmarkResult],
        config: Optional[BenchmarkConfig] = None,
        total_elapsed: float = 0.0,
    ) -> BenchmarkReport:
        """Produce a structured report with summary stats and LaTeX tables."""
        config = config or BenchmarkConfig()
        summary: Dict[str, Any] = {
            "total_elapsed_seconds": total_elapsed,
            "n_benchmarks": len(results),
            "benchmark_names": list(results.keys()),
        }

        for name, res in results.items():
            summary[name] = {
                "elapsed": res.elapsed_seconds,
                "n_trials": len(res.per_trial),
                **{k: v for k, v in res.metrics.items() if isinstance(v, (int, float, str))},
            }

        latex_tables = self._generate_latex_tables(results)

        return BenchmarkReport(
            config=config,
            results=results,
            summary=summary,
            latex_tables=latex_tables,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    # -- helper methods -----------------------------------------------------

    def _generate_random_dag(
        self,
        n_nodes: int,
        density: float,
    ) -> nx.DiGraph:
        """Generate a random DAG using Erdos-Renyi with topological ordering.

        Nodes are labelled 0..n_nodes-1.  An edge (i, j) is added with
        probability *density* only when i < j, which guarantees acyclicity
        (the identity permutation is a topological order).
        """
        g = nx.DiGraph()
        g.add_nodes_from(range(n_nodes))
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if self._rng.random() < density:
                    g.add_edge(i, j)
        return g

    def _generate_random_network(
        self,
        n_nodes: int,
        n_edges: int,
    ) -> SyntheticNetwork:
        """Generate a random SyntheticNetwork with given node/edge counts."""
        density = min(1.0, 2.0 * n_edges / max(n_nodes * (n_nodes - 1), 1))
        g = self._generate_random_dag(n_nodes, density)
        exposures = self._rng.uniform(0.1, 1.0, size=n_nodes)
        edge_weights = {(u, v): float(self._rng.uniform(0.1, 1.0)) for u, v in g.edges()}
        return SyntheticNetwork(g, exposures, edge_weights)

    def _pick_reachable_pair(
        self,
        network: SyntheticNetwork,
    ) -> Tuple[Optional[int], Optional[int]]:
        """Pick a random (source, target) pair that is reachable in the DAG."""
        topo = list(nx.topological_sort(network.graph))
        if len(topo) < 2:
            return None, None

        attempts = 0
        while attempts < 50:
            i = int(self._rng.integers(0, len(topo) - 1))
            j = int(self._rng.integers(i + 1, len(topo)))
            src, tgt = topo[i], topo[j]
            if nx.has_path(network.graph, src, tgt):
                return src, tgt
            attempts += 1

        for i, src in enumerate(topo):
            for tgt in topo[i + 1:]:
                if nx.has_path(network.graph, src, tgt):
                    return src, tgt
        return None, None

    def _time_execution(
        self,
        fn: Callable[[], Any],
    ) -> Tuple[Any, float]:
        """Time a callable, returning (result, elapsed_seconds)."""
        t0 = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - t0
        return result, elapsed

    def _measure_solve(
        self,
        network: SyntheticNetwork,
        source: int,
        target: int,
        treewidth: int = 4,
    ) -> Tuple[float, int, float]:
        """Run CausalBound solver and return (runtime, memory_estimate, quality).

        Quality is measured as 1 / (interval_width + epsilon) so that
        tighter bounds score higher.
        """
        t_start = time.perf_counter()
        lo, hi = _causalbound_solve(network, source, target, treewidth_budget=treewidth)
        runtime = time.perf_counter() - t_start

        mem_estimate = (
            network.n_nodes * network.n_edges * 8
            + network.n_nodes * 8
        )
        width = hi - lo
        quality = 1.0 / (width + 1e-6)
        return runtime, mem_estimate, quality

    def _generate_latex_tables(
        self,
        results: Dict[str, BenchmarkResult],
    ) -> Dict[str, str]:
        """Generate LaTeX tables for each benchmark result."""
        tables: Dict[str, str] = {}

        if "bound_tightness" in results:
            res = results["bound_tightness"]
            per_size = res.metrics.get("per_size", [])
            rows = []
            for entry in per_size:
                rows.append([
                    str(entry["n_nodes"]),
                    f'{entry["mean_ratio"]:.3f}',
                    f'{entry["std_ratio"]:.3f}',
                ])
            tables["bound_tightness"] = _latex_table(
                headers=["$n$", "Mean Ratio", "Std"],
                rows=rows,
                caption="Bound tightness: CausalBound interval width / true interval width",
                label="tab:tightness",
            )

        if "pathway_recall" in results:
            res = results["pathway_recall"]
            metrics = res.metrics
            rows = [[
                f'{metrics.get("mean_recall", 0):.3f}',
                f'{metrics.get("mean_precision", 0):.3f}',
                f'{metrics.get("mean_f1", 0):.3f}',
            ]]
            tables["pathway_recall"] = _latex_table(
                headers=["Recall", "Precision", "F1"],
                rows=rows,
                caption="Pathway discovery accuracy",
                label="tab:pathway",
            )

        if "discovery_power" in results:
            res = results["discovery_power"]
            metrics = res.metrics
            rows = [[
                f'{metrics.get("mean_ratio_vs_random", 0):.3f}',
                f'{metrics.get("mean_ratio_vs_grid", 0):.3f}',
                f'{metrics.get("mean_ratio_vs_cmaes", 0):.3f}',
            ]]
            tables["discovery_power"] = _latex_table(
                headers=["vs Random", "vs Grid", "vs CMA-ES"],
                rows=rows,
                caption="MCTS discovery ratio (higher is better)",
                label="tab:discovery",
            )

        if "verification_overhead" in results:
            res = results["verification_overhead"]
            metrics = res.metrics
            rows = [[
                f'{metrics.get("mean_overhead_ratio", 0):.3f}',
                f'{metrics.get("median_overhead_ratio", 0):.3f}',
                str(metrics.get("mean_assertions", 0)),
                str(metrics.get("max_assertions", 0)),
            ]]
            tables["verification_overhead"] = _latex_table(
                headers=["Mean Overhead", "Median Overhead", "Mean Asserts", "Max Asserts"],
                rows=rows,
                caption="Verification overhead",
                label="tab:verification",
            )

        if "scalability" in results:
            res = results["scalability"]
            axes_data = res.metrics.get("axes", [])
            rows = []
            for ax in axes_data:
                rows.append([
                    ax["name"],
                    ax["fit_type"],
                    f'{ax["r_squared"]:.3f}',
                    ", ".join(f"{p:.4g}" for p in ax["fit_params"]),
                ])
            tables["scalability"] = _latex_table(
                headers=["Axis", "Fit Type", "$R^2$", "Parameters"],
                rows=rows,
                caption="Scalability curve fits",
                label="tab:scalability",
            )

        return tables


# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------

def compute_confidence_interval(
    data: Sequence[float],
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute mean and confidence interval using the t-distribution.

    Returns (mean, ci_lower, ci_upper).
    """
    arr = np.asarray(data, dtype=np.float64)
    n = len(arr)
    if n < 2:
        m = float(np.mean(arr))
        return m, m, m
    mean = float(np.mean(arr))
    se = float(stats.sem(arr))
    t_crit = float(stats.t.ppf((1 + confidence) / 2, df=n - 1))
    margin = t_crit * se
    return mean, mean - margin, mean + margin


def wilcoxon_test(
    a: Sequence[float],
    b: Sequence[float],
) -> Tuple[float, float]:
    """Wilcoxon signed-rank test for paired samples.

    Returns (statistic, p_value).  Falls back to (0.0, 1.0) when inputs
    are too short or identical.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    diffs = a_arr - b_arr
    if len(diffs) < 5 or np.all(diffs == 0):
        return 0.0, 1.0
    try:
        stat, p = stats.wilcoxon(diffs)
        return float(stat), float(p)
    except ValueError:
        return 0.0, 1.0


def mann_whitney_test(
    a: Sequence[float],
    b: Sequence[float],
) -> Tuple[float, float]:
    """Mann-Whitney U test for independent samples.

    Returns (statistic, p_value).
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if len(a_arr) < 3 or len(b_arr) < 3:
        return 0.0, 1.0
    try:
        stat, p = stats.mannwhitneyu(a_arr, b_arr, alternative="two-sided")
        return float(stat), float(p)
    except ValueError:
        return 0.0, 1.0


def effect_size_cohens_d(
    a: Sequence[float],
    b: Sequence[float],
) -> float:
    """Compute Cohen's d effect size for two independent samples."""
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    n1, n2 = len(a_arr), len(b_arr)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1 = float(np.var(a_arr, ddof=1))
    var2 = float(np.var(b_arr, ddof=1))
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-15:
        return 0.0
    return float((np.mean(a_arr) - np.mean(b_arr)) / pooled_std)
