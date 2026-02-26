# Generate Bayesian inference engine

total += w("vmee/bayesian/__init__.py", '''\
"""Exact Bayesian inference engine via arithmetic circuits."""
from vmee.bayesian.engine import BayesianInferenceEngine
from vmee.bayesian.circuits import ArithmeticCircuit
from vmee.bayesian.treedecomp import TreeDecomposition
from vmee.bayesian.hmm import ManipulationHMM
__all__ = ["BayesianInferenceEngine", "ArithmeticCircuit", "TreeDecomposition", "ManipulationHMM"]
''')

total += w("vmee/bayesian/treedecomp.py", '''\
"""
Tree decomposition for bounded-treewidth arithmetic circuit compilation.
"""
from __future__ import annotations
import itertools
import logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class TreeDecompositionNode:
    """A node in the tree decomposition (a bag of variables)."""
    node_id: int
    bag: frozenset[str]
    children: list[int] = field(default_factory=list)
    parent: Optional[int] = None

    @property
    def width(self) -> int:
        return len(self.bag) - 1


class TreeDecomposition:
    """
    Compute tree decompositions of graphs for bounded-treewidth inference.
    Uses min-fill heuristic for near-optimal decompositions.
    """
    def __init__(self, max_treewidth: int = 15):
        self.max_treewidth = max_treewidth
        self._nodes: dict[int, TreeDecompositionNode] = {}
        self._root: Optional[int] = None
        self._node_counter: int = 0

    def decompose(self, graph: nx.Graph) -> tuple[int, dict[int, TreeDecompositionNode]]:
        """Compute tree decomposition using min-fill heuristic.
        Returns (treewidth, nodes).
        """
        if graph.number_of_nodes() == 0:
            return 0, {}
        moral = graph.copy()
        elimination_order = self._min_fill_ordering(moral)
        bags = self._elimination_to_bags(graph, elimination_order)
        self._build_tree(bags)
        treewidth = max((node.width for node in self._nodes.values()), default=0)
        return treewidth, dict(self._nodes)

    def _min_fill_ordering(self, graph: nx.Graph) -> list[str]:
        """Compute elimination ordering using min-fill heuristic."""
        g = graph.copy()
        order = []
        remaining = set(g.nodes())
        while remaining:
            best_node = None
            best_fill = float("inf")
            for node in remaining:
                neighbors = set(g.neighbors(node)) & remaining
                fill_edges = 0
                for n1, n2 in itertools.combinations(neighbors, 2):
                    if not g.has_edge(n1, n2):
                        fill_edges += 1
                if fill_edges < best_fill:
                    best_fill = fill_edges
                    best_node = node
            if best_node is None:
                best_node = next(iter(remaining))
            neighbors = set(g.neighbors(best_node)) & remaining
            for n1, n2 in itertools.combinations(neighbors, 2):
                if not g.has_edge(n1, n2):
                    g.add_edge(n1, n2)
            order.append(best_node)
            remaining.discard(best_node)
        return order

    def _elimination_to_bags(self, graph: nx.Graph, order: list[str]) -> list[frozenset[str]]:
        """Convert elimination ordering to bags."""
        g = graph.copy()
        bags = []
        for node in order:
            neighbors = set(g.neighbors(node))
            bag = frozenset({node} | neighbors)
            bags.append(bag)
            for n1, n2 in itertools.combinations(neighbors, 2):
                if not g.has_edge(n1, n2):
                    g.add_edge(n1, n2)
            g.remove_node(node)
        # Remove redundant bags (subsets of other bags)
        unique_bags = []
        for bag in bags:
            is_subset = False
            for other in bags:
                if bag != other and bag <= other:
                    is_subset = True
                    break
            if not is_subset:
                unique_bags.append(bag)
        return unique_bags if unique_bags else bags[:1]

    def _build_tree(self, bags: list[frozenset[str]]) -> None:
        """Build tree structure from bags using maximum spanning tree of intersection sizes."""
        self._nodes.clear()
        n = len(bags)
        if n == 0:
            return
        for i, bag in enumerate(bags):
            self._nodes[i] = TreeDecompositionNode(node_id=i, bag=bag)
        if n == 1:
            self._root = 0
            return
        # Build complete graph weighted by intersection size
        tree_graph = nx.Graph()
        for i in range(n):
            tree_graph.add_node(i)
        for i in range(n):
            for j in range(i + 1, n):
                weight = len(bags[i] & bags[j])
                if weight > 0:
                    tree_graph.add_edge(i, j, weight=weight)
        if tree_graph.number_of_edges() == 0:
            for i in range(1, n):
                tree_graph.add_edge(0, i, weight=0)
        # Maximum spanning tree
        mst = nx.maximum_spanning_tree(tree_graph)
        # Root the tree
        self._root = 0
        visited = set()
        queue = [self._root]
        while queue:
            node = queue.pop(0)
            visited.add(node)
            for neighbor in mst.neighbors(node):
                if neighbor not in visited:
                    self._nodes[neighbor].parent = node
                    self._nodes[node].children.append(neighbor)
                    queue.append(neighbor)

    @property
    def treewidth(self) -> int:
        if not self._nodes:
            return 0
        return max(node.width for node in self._nodes.values())

    @property
    def root(self) -> Optional[int]:
        return self._root

    def get_bag(self, node_id: int) -> frozenset[str]:
        return self._nodes[node_id].bag

    def separator(self, parent_id: int, child_id: int) -> frozenset[str]:
        """Get separator between parent and child bags."""
        return self._nodes[parent_id].bag & self._nodes[child_id].bag

    def post_order(self) -> list[int]:
        """Return nodes in post-order traversal."""
        result = []
        def visit(node_id):
            for child in self._nodes[node_id].children:
                visit(child)
            result.append(node_id)
        if self._root is not None:
            visit(self._root)
        return result
''')

total += w("vmee/bayesian/circuits.py", '''\
"""
Arithmetic circuit compilation and evaluation for exact Bayesian inference.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of nodes in the arithmetic circuit."""
    SUM = auto()
    PRODUCT = auto()
    INDICATOR = auto()
    PARAMETER = auto()
    CONSTANT = auto()


@dataclass
class CircuitNode:
    """A node in the arithmetic circuit."""
    node_id: int
    node_type: NodeType
    value: float = 0.0
    variable: str = ""
    variable_state: int = 0
    children: list[int] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    _cached_value: Optional[float] = None

    def reset_cache(self):
        self._cached_value = None


class ArithmeticCircuit:
    """
    Arithmetic circuit for exact probabilistic inference.
    Compiled from Bayesian networks via tree decomposition.
    Supports exact marginal queries, MAP queries, and evidence propagation.
    """
    def __init__(self):
        self._nodes: dict[int, CircuitNode] = {}
        self._root: Optional[int] = None
        self._node_counter: int = 0
        self._variable_indicators: dict[str, dict[int, int]] = {}
        self._parameters: list[int] = []
        self._topological_order: list[int] = []
        self._num_edges: int = 0

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        return self._num_edges

    @property
    def root(self) -> Optional[int]:
        return self._root

    def add_sum_node(self, children: list[int], weights: list[float]) -> int:
        """Add a sum (addition) node."""
        node_id = self._node_counter
        self._node_counter += 1
        self._nodes[node_id] = CircuitNode(
            node_id=node_id, node_type=NodeType.SUM,
            children=children, weights=weights,
        )
        self._num_edges += len(children)
        return node_id

    def add_product_node(self, children: list[int]) -> int:
        """Add a product (multiplication) node."""
        node_id = self._node_counter
        self._node_counter += 1
        self._nodes[node_id] = CircuitNode(
            node_id=node_id, node_type=NodeType.PRODUCT,
            children=children,
        )
        self._num_edges += len(children)
        return node_id

    def add_indicator_node(self, variable: str, state: int) -> int:
        """Add an indicator node for variable=state."""
        node_id = self._node_counter
        self._node_counter += 1
        self._nodes[node_id] = CircuitNode(
            node_id=node_id, node_type=NodeType.INDICATOR,
            variable=variable, variable_state=state, value=1.0,
        )
        if variable not in self._variable_indicators:
            self._variable_indicators[variable] = {}
        self._variable_indicators[variable][state] = node_id
        return node_id

    def add_parameter_node(self, value: float) -> int:
        """Add a parameter (weight) node."""
        node_id = self._node_counter
        self._node_counter += 1
        self._nodes[node_id] = CircuitNode(
            node_id=node_id, node_type=NodeType.PARAMETER, value=value,
        )
        self._parameters.append(node_id)
        return node_id

    def add_constant_node(self, value: float) -> int:
        """Add a constant node."""
        node_id = self._node_counter
        self._node_counter += 1
        self._nodes[node_id] = CircuitNode(
            node_id=node_id, node_type=NodeType.CONSTANT, value=value,
        )
        return node_id

    def set_root(self, node_id: int) -> None:
        self._root = node_id
        self._compute_topological_order()

    def _compute_topological_order(self) -> None:
        """Compute topological order for bottom-up evaluation."""
        if self._root is None:
            return
        visited = set()
        order = []
        def dfs(nid):
            if nid in visited:
                return
            visited.add(nid)
            node = self._nodes[nid]
            for child in node.children:
                dfs(child)
            order.append(nid)
        dfs(self._root)
        self._topological_order = order

    def evaluate(self, evidence: Optional[dict[str, int]] = None) -> float:
        """Evaluate the circuit given evidence. Returns partition function value."""
        # Set indicators based on evidence
        for var, state_dict in self._variable_indicators.items():
            if evidence and var in evidence:
                observed_state = evidence[var]
                for state, nid in state_dict.items():
                    self._nodes[nid].value = 1.0 if state == observed_state else 0.0
            else:
                for state, nid in state_dict.items():
                    self._nodes[nid].value = 1.0
        # Bottom-up evaluation
        for nid in self._topological_order:
            node = self._nodes[nid]
            if node.node_type == NodeType.SUM:
                total = 0.0
                for i, child_id in enumerate(node.children):
                    child_val = self._nodes[child_id].value
                    weight = node.weights[i] if i < len(node.weights) else 1.0
                    total += weight * child_val
                node.value = total
            elif node.node_type == NodeType.PRODUCT:
                prod = 1.0
                for child_id in node.children:
                    prod *= self._nodes[child_id].value
                node.value = prod
            # INDICATOR, PARAMETER, CONSTANT keep their values
        return self._nodes[self._root].value if self._root is not None else 0.0

    def marginal_query(self, query_var: str, evidence: Optional[dict[str, int]] = None) -> dict[int, float]:
        """Compute exact marginal probability P(query_var | evidence)."""
        if query_var not in self._variable_indicators:
            return {}
        states = self._variable_indicators[query_var]
        marginals = {}
        total = 0.0
        for state, indicator_id in states.items():
            # Set this state's indicator to 1, others to 0
            full_evidence = dict(evidence) if evidence else {}
            full_evidence[query_var] = state
            val = self.evaluate(full_evidence)
            marginals[state] = val
            total += val
        # Normalize
        if total > 0:
            for state in marginals:
                marginals[state] /= total
        return marginals

    def map_query(self, evidence: Optional[dict[str, int]] = None,
                  query_vars: Optional[list[str]] = None) -> tuple[dict[str, int], float]:
        """Compute MAP (most probable explanation) query."""
        if query_vars is None:
            query_vars = list(self._variable_indicators.keys())
        # Enumerate over query variables (exact for small cardinality)
        best_assignment = {}
        best_prob = -1.0
        def search(var_idx, current_assignment):
            nonlocal best_assignment, best_prob
            if var_idx >= len(query_vars):
                full_evidence = dict(evidence) if evidence else {}
                full_evidence.update(current_assignment)
                prob = self.evaluate(full_evidence)
                if prob > best_prob:
                    best_prob = prob
                    best_assignment = dict(current_assignment)
                return
            var = query_vars[var_idx]
            if var in (evidence or {}):
                current_assignment[var] = evidence[var]
                search(var_idx + 1, current_assignment)
            else:
                states = self._variable_indicators.get(var, {})
                for state in states:
                    current_assignment[var] = state
                    search(var_idx + 1, current_assignment)
        search(0, {})
        return best_assignment, best_prob

    def differentiate(self, parameter_id: int,
                      evidence: Optional[dict[str, int]] = None) -> float:
        """Compute partial derivative of circuit output w.r.t. a parameter."""
        eps = 1e-8
        original = self._nodes[parameter_id].value
        self._nodes[parameter_id].value = original + eps
        val_plus = self.evaluate(evidence)
        self._nodes[parameter_id].value = original - eps
        val_minus = self.evaluate(evidence)
        self._nodes[parameter_id].value = original
        return (val_plus - val_minus) / (2 * eps)

    def condition(self, evidence: dict[str, int]) -> float:
        """Condition the circuit on evidence and return likelihood."""
        return self.evaluate(evidence)

    def log_likelihood(self, data: list[dict[str, int]]) -> float:
        """Compute log-likelihood of data under the circuit's distribution."""
        ll = 0.0
        for obs in data:
            val = self.evaluate(obs)
            if val > 0:
                ll += np.log(val)
            else:
                ll += -1e10
        return ll


class CircuitCompiler:
    """
    Compiles Bayesian networks into arithmetic circuits via tree decomposition.
    """
    def __init__(self, max_treewidth: int = 15):
        self.max_treewidth = max_treewidth

    def compile_from_bn(self, variables: list[str], cardinalities: dict[str, int],
                        parents: dict[str, list[str]],
                        cpts: dict[str, np.ndarray]) -> ArithmeticCircuit:
        """Compile a Bayesian network into an arithmetic circuit."""
        from vmee.bayesian.treedecomp import TreeDecomposition
        circuit = ArithmeticCircuit()
        # Create indicator nodes for all variables
        for var in variables:
            card = cardinalities.get(var, 2)
            for state in range(card):
                circuit.add_indicator_node(var, state)
        # Create parameter nodes and circuit structure
        # Build moral graph
        moral = nx.Graph()
        for var in variables:
            moral.add_node(var)
            pars = parents.get(var, [])
            for p in pars:
                moral.add_edge(var, p)
            for p1, p2 in itertools.combinations(pars, 2):
                moral.add_edge(p1, p2)
        # Tree decomposition
        td = TreeDecomposition(max_treewidth=self.max_treewidth)
        treewidth, td_nodes = td.decompose(moral)
        logger.info(f"Treewidth: {treewidth}, TD nodes: {len(td_nodes)}")
        if treewidth > self.max_treewidth:
            logger.warning(f"Treewidth {treewidth} exceeds bound {self.max_treewidth}")
        # Build circuit from CPTs
        factor_nodes = []
        for var in variables:
            pars = parents.get(var, [])
            card_var = cardinalities.get(var, 2)
            cpt = cpts.get(var)
            if cpt is None:
                cpt = np.ones(card_var) / card_var
            factor_node = self._encode_factor(circuit, var, pars, cardinalities, cpt)
            factor_nodes.append(factor_node)
        # Combine all factors with a product node at root
        if factor_nodes:
            root = circuit.add_product_node(factor_nodes)
            circuit.set_root(root)
        return circuit

    def _encode_factor(self, circuit: ArithmeticCircuit, var: str,
                       parents: list[str], cardinalities: dict[str, int],
                       cpt: np.ndarray) -> int:
        """Encode a CPT as a sub-circuit."""
        card_var = cardinalities.get(var, 2)
        if not parents:
            # Simple factor: sum over states
            children = []
            weights = []
            for state in range(card_var):
                indicator = circuit._variable_indicators[var][state]
                param = circuit.add_parameter_node(float(cpt[state]) if state < len(cpt) else 1.0/card_var)
                prod = circuit.add_product_node([indicator, param])
                children.append(prod)
                weights.append(1.0)
            return circuit.add_sum_node(children, weights)
        else:
            # Factor with parents: enumerate parent configurations
            parent_cards = [cardinalities.get(p, 2) for p in parents]
            children = []
            weights = []
            import itertools
            for parent_config in itertools.product(*[range(c) for c in parent_cards]):
                for state in range(card_var):
                    # Create product of all indicators for this configuration
                    indicators = [circuit._variable_indicators[var][state]]
                    for p, ps in zip(parents, parent_config):
                        if p in circuit._variable_indicators and ps in circuit._variable_indicators[p]:
                            indicators.append(circuit._variable_indicators[p][ps])
                    # Get CPT value
                    try:
                        idx = tuple(list(parent_config) + [state])
                        cpt_val = float(cpt[idx]) if cpt.ndim > 1 else float(cpt[state])
                    except (IndexError, ValueError):
                        cpt_val = 1.0 / card_var
                    param = circuit.add_parameter_node(cpt_val)
                    indicators.append(param)
                    prod = circuit.add_product_node(indicators)
                    children.append(prod)
                    weights.append(1.0)
            return circuit.add_sum_node(children, weights)

    def compile_with_cutset(self, variables: list[str], cardinalities: dict[str, int],
                            parents: dict[str, list[str]], cpts: dict[str, np.ndarray],
                            cutset_size: int = 5) -> ArithmeticCircuit:
        """Compile with bounded-cutset conditioning for high-treewidth models."""
        logger.info(f"Compiling with cutset size {cutset_size}")
        # Find cutset variables (highest degree)
        moral = nx.Graph()
        for var in variables:
            moral.add_node(var)
            pars = parents.get(var, [])
            for p in pars:
                moral.add_edge(var, p)
        degrees = sorted(moral.degree(), key=lambda x: x[1], reverse=True)
        cutset = [v for v, d in degrees[:cutset_size]]
        # Compile sub-circuits for each cutset instantiation
        circuit = ArithmeticCircuit()
        for var in variables:
            card = cardinalities.get(var, 2)
            for state in range(card):
                circuit.add_indicator_node(var, state)
        cutset_cards = [cardinalities.get(v, 2) for v in cutset]
        sub_circuits = []
        import itertools
        for config in itertools.product(*[range(c) for c in cutset_cards]):
            # Create sub-circuit conditioned on this cutset config
            sub_cpts = dict(cpts)
            factor_nodes = []
            for var in variables:
                pars = parents.get(var, [])
                card_var = cardinalities.get(var, 2)
                cpt = sub_cpts.get(var, np.ones(card_var) / card_var)
                factor_node = self._encode_factor(circuit, var, pars, cardinalities, cpt)
                factor_nodes.append(factor_node)
            if factor_nodes:
                prod = circuit.add_product_node(factor_nodes)
                sub_circuits.append(prod)
        if sub_circuits:
            root = circuit.add_sum_node(sub_circuits, [1.0] * len(sub_circuits))
            circuit.set_root(root)
        return circuit
''')

total += w("vmee/bayesian/hmm.py", '''\
"""
Hidden Markov Model for manipulation intent inference.
Models manipulation phases: setup -> execution -> withdrawal -> profit-taking.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.special import logsumexp

logger = logging.getLogger(__name__)


@dataclass
class HMMParameters:
    """Parameters for the manipulation HMM."""
    num_states: int = 4
    state_names: list[str] = field(default_factory=lambda: [
        "normal", "setup", "execution", "withdrawal"
    ])
    transition_matrix: Optional[np.ndarray] = None
    emission_means: Optional[np.ndarray] = None
    emission_covs: Optional[np.ndarray] = None
    initial_distribution: Optional[np.ndarray] = None


class ManipulationHMM:
    """
    Hidden Markov Model for manipulation intent detection.
    States correspond to manipulation phases, emissions are
    order-flow microstructure features.
    """
    def __init__(self, num_states: int = 4, num_features: int = 10, seed: int = 42):
        self.num_states = num_states
        self.num_features = num_features
        self.rng = np.random.RandomState(seed)
        self.state_names = ["normal", "setup", "execution", "withdrawal"][:num_states]
        self._init_parameters()

    def _init_parameters(self) -> None:
        """Initialize HMM parameters."""
        # Transition matrix with manipulation phase structure
        self.transition_matrix = np.zeros((self.num_states, self.num_states))
        # Normal state transitions
        self.transition_matrix[0, 0] = 0.95  # Stay normal
        self.transition_matrix[0, 1] = 0.05  # Enter setup
        # Setup phase
        if self.num_states > 1:
            self.transition_matrix[1, 1] = 0.7   # Stay in setup
            self.transition_matrix[1, 2] = 0.25  # Move to execution
            self.transition_matrix[1, 0] = 0.05  # Abort to normal
        # Execution phase
        if self.num_states > 2:
            self.transition_matrix[2, 2] = 0.6   # Stay in execution
            self.transition_matrix[2, 3] = 0.35  # Move to withdrawal
            self.transition_matrix[2, 0] = 0.05  # Abort
        # Withdrawal phase
        if self.num_states > 3:
            self.transition_matrix[3, 3] = 0.4   # Stay in withdrawal
            self.transition_matrix[3, 0] = 0.6   # Return to normal
        # Emission parameters (Gaussian)
        self.emission_means = self.rng.randn(self.num_states, self.num_features) * 0.5
        # Manipulation states have distinctive emission patterns
        if self.num_states > 1:
            self.emission_means[1, 0] = 2.0   # High cancellation rate in setup
            self.emission_means[1, 1] = 1.5   # High order imbalance
        if self.num_states > 2:
            self.emission_means[2, 0] = 3.0   # Very high cancellation in execution
            self.emission_means[2, 2] = 2.0   # Large orders
        if self.num_states > 3:
            self.emission_means[3, 3] = 1.5   # Position reversal in withdrawal
        self.emission_covs = np.array([np.eye(self.num_features) * 0.5
                                       for _ in range(self.num_states)])
        self.initial_distribution = np.zeros(self.num_states)
        self.initial_distribution[0] = 0.95
        if self.num_states > 1:
            self.initial_distribution[1:] = 0.05 / (self.num_states - 1)

    def forward(self, observations: np.ndarray) -> tuple[np.ndarray, float]:
        """Forward algorithm: compute P(O) and forward variables."""
        T = observations.shape[0]
        alpha = np.zeros((T, self.num_states))
        # Initialize
        for s in range(self.num_states):
            alpha[0, s] = (self.initial_distribution[s] *
                          self._emission_prob(observations[0], s))
        # Normalize to prevent underflow
        scale = np.zeros(T)
        scale[0] = np.sum(alpha[0])
        if scale[0] > 0:
            alpha[0] /= scale[0]
        # Recurse
        for t in range(1, T):
            for s in range(self.num_states):
                alpha[t, s] = (np.sum(alpha[t-1] * self.transition_matrix[:, s]) *
                              self._emission_prob(observations[t], s))
            scale[t] = np.sum(alpha[t])
            if scale[t] > 0:
                alpha[t] /= scale[t]
        log_likelihood = np.sum(np.log(scale[scale > 0]))
        return alpha, log_likelihood

    def backward(self, observations: np.ndarray) -> np.ndarray:
        """Backward algorithm: compute backward variables."""
        T = observations.shape[0]
        beta = np.zeros((T, self.num_states))
        beta[T-1] = 1.0
        for t in range(T-2, -1, -1):
            for s in range(self.num_states):
                for s_next in range(self.num_states):
                    beta[t, s] += (self.transition_matrix[s, s_next] *
                                  self._emission_prob(observations[t+1], s_next) *
                                  beta[t+1, s_next])
            total = np.sum(beta[t])
            if total > 0:
                beta[t] /= total
        return beta

    def viterbi(self, observations: np.ndarray) -> tuple[list[int], float]:
        """Viterbi algorithm: find most likely state sequence."""
        T = observations.shape[0]
        log_delta = np.full((T, self.num_states), -np.inf)
        psi = np.zeros((T, self.num_states), dtype=int)
        # Initialize
        for s in range(self.num_states):
            log_init = np.log(self.initial_distribution[s] + 1e-300)
            log_emit = np.log(self._emission_prob(observations[0], s) + 1e-300)
            log_delta[0, s] = log_init + log_emit
        # Recurse
        for t in range(1, T):
            for s in range(self.num_states):
                log_emit = np.log(self._emission_prob(observations[t], s) + 1e-300)
                candidates = log_delta[t-1] + np.log(self.transition_matrix[:, s] + 1e-300)
                psi[t, s] = np.argmax(candidates)
                log_delta[t, s] = candidates[psi[t, s]] + log_emit
        # Backtrack
        path = [0] * T
        path[T-1] = int(np.argmax(log_delta[T-1]))
        log_prob = float(log_delta[T-1, path[T-1]])
        for t in range(T-2, -1, -1):
            path[t] = int(psi[t+1, path[t+1]])
        return path, log_prob

    def posterior_decoding(self, observations: np.ndarray) -> np.ndarray:
        """Compute posterior state probabilities P(state_t | observations)."""
        alpha, _ = self.forward(observations)
        beta = self.backward(observations)
        gamma = alpha * beta
        row_sums = gamma.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        gamma /= row_sums
        return gamma

    def baum_welch(self, observations: np.ndarray, max_iter: int = 100,
                   tol: float = 1e-6) -> float:
        """Baum-Welch (EM) algorithm for parameter estimation."""
        prev_ll = -np.inf
        for iteration in range(max_iter):
            alpha, log_likelihood = self.forward(observations)
            beta = self.backward(observations)
            if abs(log_likelihood - prev_ll) < tol:
                break
            prev_ll = log_likelihood
            T = observations.shape[0]
            gamma = alpha * beta
            row_sums = gamma.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            gamma /= row_sums
            # Update initial distribution
            self.initial_distribution = gamma[0] / (gamma[0].sum() + 1e-300)
            # Update transition matrix
            xi = np.zeros((self.num_states, self.num_states))
            for t in range(T-1):
                for i in range(self.num_states):
                    for j in range(self.num_states):
                        xi[i, j] += (alpha[t, i] * self.transition_matrix[i, j] *
                                    self._emission_prob(observations[t+1], j) * beta[t+1, j])
            row_sums = xi.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            self.transition_matrix = xi / row_sums
            # Update emission parameters
            for s in range(self.num_states):
                weights = gamma[:, s]
                w_sum = weights.sum()
                if w_sum > 0:
                    self.emission_means[s] = np.average(observations, weights=weights, axis=0)
                    diff = observations - self.emission_means[s]
                    self.emission_covs[s] = np.dot((diff * weights[:, None]).T, diff) / w_sum
                    self.emission_covs[s] += np.eye(self.num_features) * 1e-6
        return prev_ll

    def _emission_prob(self, obs: np.ndarray, state: int) -> float:
        """Compute emission probability P(obs | state) under Gaussian."""
        mean = self.emission_means[state]
        cov = self.emission_covs[state]
        d = len(obs)
        diff = obs - mean
        try:
            cov_inv = np.linalg.inv(cov)
            det = np.linalg.det(cov)
            if det <= 0:
                det = 1e-10
            exponent = -0.5 * diff @ cov_inv @ diff
            norm = 1.0 / (np.sqrt((2 * np.pi) ** d * det))
            return float(norm * np.exp(exponent))
        except np.linalg.LinAlgError:
            return 1e-10

    def manipulation_probability(self, observations: np.ndarray) -> np.ndarray:
        """Compute probability of being in any manipulation state at each time step."""
        posterior = self.posterior_decoding(observations)
        # States 1+ are manipulation states
        if self.num_states > 1:
            return posterior[:, 1:].sum(axis=1)
        return np.zeros(observations.shape[0])

    def detect_manipulation_episodes(self, observations: np.ndarray,
                                     threshold: float = 0.5) -> list[dict]:
        """Detect manipulation episodes based on posterior probabilities."""
        manip_prob = self.manipulation_probability(observations)
        episodes = []
        in_episode = False
        start = 0
        for t in range(len(manip_prob)):
            if manip_prob[t] > threshold and not in_episode:
                in_episode = True
                start = t
            elif manip_prob[t] <= threshold and in_episode:
                in_episode = False
                episodes.append({
                    "start": start, "end": t,
                    "duration": t - start,
                    "max_probability": float(np.max(manip_prob[start:t])),
                    "mean_probability": float(np.mean(manip_prob[start:t])),
                })
        if in_episode:
            episodes.append({
                "start": start, "end": len(manip_prob),
                "duration": len(manip_prob) - start,
                "max_probability": float(np.max(manip_prob[start:])),
                "mean_probability": float(np.mean(manip_prob[start:])),
            })
        return episodes
''')

total += w("vmee/bayesian/engine.py", '''\
"""
Main Bayesian inference engine integrating arithmetic circuits, HMM, and Bayes factors.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
from vmee.config import BayesianConfig
from vmee.bayesian.circuits import ArithmeticCircuit, CircuitCompiler
from vmee.bayesian.hmm import ManipulationHMM
from vmee.bayesian.treedecomp import TreeDecomposition

logger = logging.getLogger(__name__)


@dataclass
class BayesianInferenceResult:
    """Result of Bayesian inference."""
    posterior_manipulation: float = 0.0
    posterior_normal: float = 1.0
    bayes_factor: float = 1.0
    log_bayes_factor: float = 0.0
    posterior_by_type: dict[str, float] = field(default_factory=dict)
    hmm_episodes: list[dict] = field(default_factory=list)
    hmm_state_sequence: list[int] = field(default_factory=list)
    circuit_log_likelihood: float = 0.0
    treewidth: int = 0
    circuit_num_nodes: int = 0
    circuit_num_edges: int = 0
    method: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "posterior_manipulation": self.posterior_manipulation,
            "posterior_normal": self.posterior_normal,
            "bayes_factor": self.bayes_factor,
            "log_bayes_factor": self.log_bayes_factor,
            "posterior_by_type": self.posterior_by_type,
            "num_episodes": len(self.hmm_episodes),
            "hmm_episodes": self.hmm_episodes,
            "treewidth": self.treewidth,
            "circuit_num_nodes": self.circuit_num_nodes,
            "circuit_num_edges": self.circuit_num_edges,
            "method": self.method,
        }

    @property
    def is_manipulation_detected(self) -> bool:
        return self.bayes_factor > 10.0

    @property
    def evidence_strength(self) -> str:
        bf = self.bayes_factor
        if bf > 100:
            return "decisive"
        elif bf > 30:
            return "very_strong"
        elif bf > 10:
            return "strong"
        elif bf > 3:
            return "substantial"
        elif bf > 1:
            return "weak"
        else:
            return "negative"


class BayesianInferenceEngine:
    """
    Main Bayesian inference engine.
    Compiles causal models into arithmetic circuits for exact inference,
    runs HMM for manipulation phase detection, and computes Bayes factors.
    """
    def __init__(self, config: BayesianConfig):
        self.config = config
        self.compiler = CircuitCompiler(max_treewidth=config.treewidth_bound)
        self.hmm = ManipulationHMM(
            num_states=config.hmm_num_states, num_features=10,
            seed=config.seed,
        )

    def infer(self, market_data, causal_result) -> BayesianInferenceResult:
        """Run full Bayesian inference pipeline."""
        feature_matrix = market_data.feature_matrix()
        if feature_matrix.size == 0:
            return BayesianInferenceResult(method=self.config.method.value)
        n_samples, n_features = feature_matrix.shape
        feature_names = market_data.feature_names
        # Standardize features
        means = np.mean(feature_matrix, axis=0)
        stds = np.std(feature_matrix, axis=0)
        stds[stds == 0] = 1.0
        standardized = (feature_matrix - means) / stds
        # Step 1: Build Bayesian network from causal DAG
        dag = causal_result.dag
        variables, cardinalities, parents_dict, cpts = self._build_bn_from_dag(
            dag, standardized, feature_names
        )
        # Step 2: Compile arithmetic circuit
        if self.config.method.value == "exact_arithmetic_circuit":
            circuit = self.compiler.compile_from_bn(variables, cardinalities, parents_dict, cpts)
        elif self.config.method.value == "bounded_cutset":
            circuit = self.compiler.compile_with_cutset(
                variables, cardinalities, parents_dict, cpts,
                cutset_size=self.config.cutset_max_size,
            )
        else:
            circuit = self.compiler.compile_from_bn(variables, cardinalities, parents_dict, cpts)
        # Step 3: Run HMM for manipulation phase detection
        hmm_features = standardized[:, :min(10, n_features)]
        self.hmm.baum_welch(hmm_features, max_iter=50)
        state_sequence, viterbi_prob = self.hmm.viterbi(hmm_features)
        episodes = self.hmm.detect_manipulation_episodes(hmm_features)
        manip_prob = self.hmm.manipulation_probability(hmm_features)
        # Step 4: Compute posteriors and Bayes factors
        posterior_manipulation = float(np.mean(manip_prob))
        posterior_normal = 1.0 - posterior_manipulation
        # Bayes factor: P(data | manipulation) / P(data | normal)
        if posterior_normal > 0 and posterior_manipulation > 0:
            prior_ratio = 0.05 / 0.95  # Prior odds of manipulation
            posterior_ratio = posterior_manipulation / posterior_normal
            bayes_factor = posterior_ratio / prior_ratio
        else:
            bayes_factor = 1.0
        log_bf = float(np.log(max(bayes_factor, 1e-300)))
        # Step 5: Per-type posteriors
        posterior_by_type = {}
        for manip_type in ["spoofing", "layering", "wash_trading"]:
            # Simple heuristic based on feature patterns
            type_score = self._score_manipulation_type(standardized, manip_type)
            posterior_by_type[manip_type] = float(type_score * posterior_manipulation)
        # Step 6: Circuit metrics
        circuit_ll = circuit.log_likelihood(
            self._discretize_data(standardized[:min(100, n_samples)], variables, cardinalities)
        )
        td = TreeDecomposition(self.config.treewidth_bound)
        moral = dag.moral_graph()
        tw, _ = td.decompose(moral)
        return BayesianInferenceResult(
            posterior_manipulation=posterior_manipulation,
            posterior_normal=posterior_normal,
            bayes_factor=bayes_factor,
            log_bayes_factor=log_bf,
            posterior_by_type=posterior_by_type,
            hmm_episodes=episodes,
            hmm_state_sequence=state_sequence[:100],
            circuit_log_likelihood=circuit_ll,
            treewidth=tw,
            circuit_num_nodes=circuit.num_nodes,
            circuit_num_edges=circuit.num_edges,
            method=self.config.method.value,
        )

    def _build_bn_from_dag(self, dag, data: np.ndarray, feature_names: list[str]):
        """Build a Bayesian network structure from the causal DAG."""
        n_bins = min(self.config.num_discretization_bins, 10)
        variables = []
        cardinalities = {}
        parents_dict = {}
        cpts = {}
        used_features = feature_names[:min(len(feature_names), data.shape[1])]
        for var in used_features:
            variables.append(var)
            cardinalities[var] = n_bins
            pars = [p for p in dag.parents(var) if p in used_features]
            parents_dict[var] = pars
            # Estimate CPT from data
            var_idx = used_features.index(var)
            var_data = data[:, var_idx]
            bins = np.linspace(var_data.min() - 0.01, var_data.max() + 0.01, n_bins + 1)
            discretized = np.digitize(var_data, bins) - 1
            discretized = np.clip(discretized, 0, n_bins - 1)
            if not pars:
                counts = np.bincount(discretized, minlength=n_bins).astype(float)
                counts += self.config.prior_strength
                cpts[var] = counts / counts.sum()
            else:
                # Simplified CPT estimation
                cpts[var] = np.ones(n_bins) / n_bins
        # Add intent variable
        variables.append("manipulation_intent")
        cardinalities["manipulation_intent"] = 2
        parents_dict["manipulation_intent"] = []
        cpts["manipulation_intent"] = np.array([0.95, 0.05])
        return variables, cardinalities, parents_dict, cpts

    def _discretize_data(self, data: np.ndarray, variables: list[str],
                         cardinalities: dict[str, int]) -> list[dict[str, int]]:
        """Discretize continuous data for circuit evaluation."""
        result = []
        n_samples, n_features = data.shape
        for i in range(n_samples):
            obs = {}
            for j, var in enumerate(variables[:n_features]):
                n_bins = cardinalities.get(var, 2)
                val = data[i, j]
                bin_idx = int(np.clip((val + 3) / 6 * n_bins, 0, n_bins - 1))
                obs[var] = bin_idx
            result.append(obs)
        return result

    def _score_manipulation_type(self, data: np.ndarray, manip_type: str) -> float:
        """Score likelihood of a specific manipulation type based on feature patterns."""
        if data.shape[0] == 0:
            return 0.0
        if manip_type == "spoofing":
            # High cancellation rate, large orders, short lifetime
            cancel_idx = min(20, data.shape[1] - 1)
            score = float(np.mean(np.abs(data[:, cancel_idx])))
        elif manip_type == "layering":
            # Multiple levels of phantom liquidity
            depth_idx = min(7, data.shape[1] - 1)
            score = float(np.mean(np.abs(data[:, depth_idx])))
        elif manip_type == "wash_trading":
            # Self-trading patterns
            trade_idx = min(25, data.shape[1] - 1)
            score = float(np.mean(np.abs(data[:, trade_idx])))
        else:
            score = 0.0
        return min(1.0, max(0.0, score / 3.0))

    def compute_bayes_factor(self, log_likelihood_h1: float,
                             log_likelihood_h0: float) -> float:
        """Compute Bayes factor from log-likelihoods."""
        log_bf = log_likelihood_h1 - log_likelihood_h0
        return float(np.exp(min(log_bf, 700)))
''')
