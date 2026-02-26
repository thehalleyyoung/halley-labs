"""GNN-based circuit encoder for BioProver.

Converts :class:`BioModel` instances to bipartite species–reaction graphs
and computes fixed-dimensional embeddings via a from-scratch GraphSAGE
message-passing implementation using only NumPy and SciPy (no PyTorch /
TensorFlow).  All inference is designed for CPU execution.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / configuration
# ---------------------------------------------------------------------------

_EPS = 1e-8

SPECIES_TYPE_MAP: Dict[str, int] = {
    "PROTEIN": 0,
    "MRNA": 1,
    "SMALL_MOLECULE": 2,
    "COMPLEX": 3,
    "PROMOTER": 4,
}

REACTION_TYPE_MAP: Dict[str, int] = {
    "MASS_ACTION": 0,
    "HILL_ACTIVATION": 1,
    "HILL_REPRESSION": 2,
    "MICHAELIS_MENTEN": 3,
    "CONSTITUTIVE": 4,
    "LINEAR_DEGRADATION": 5,
    "DIMER_FORMATION": 6,
    "OTHER": 7,
}


class EdgeType(IntEnum):
    """Encodes the role of an edge in the bipartite graph."""
    REACTANT = 0
    PRODUCT = 1
    MODIFIER = 2


# ---------------------------------------------------------------------------
# Graph data structure
# ---------------------------------------------------------------------------


@dataclass
class CircuitGraph:
    """Bipartite graph representation of a biological circuit.

    The graph has two node types – *species* and *reactions* – with directed
    edges carrying stoichiometric and interaction-type information.

    Attributes
    ----------
    species_features : np.ndarray
        ``(n_species, species_feat_dim)``
    reaction_features : np.ndarray
        ``(n_reactions, reaction_feat_dim)``
    edge_index : np.ndarray
        ``(2, n_edges)`` – row 0 = source, row 1 = target.  Indices in
        ``[0, n_species + n_reactions)``.
    edge_features : np.ndarray
        ``(n_edges, edge_feat_dim)``
    species_names : list[str]
    reaction_names : list[str]
    adjacency : scipy.sparse.csr_matrix
        Sparse adjacency of shape ``(n_nodes, n_nodes)``.
    """

    species_features: np.ndarray
    reaction_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    species_names: List[str]
    reaction_names: List[str]
    adjacency: sparse.csr_matrix = field(repr=False)

    @property
    def n_species(self) -> int:
        return self.species_features.shape[0]

    @property
    def n_reactions(self) -> int:
        return self.reaction_features.shape[0]

    @property
    def n_nodes(self) -> int:
        return self.n_species + self.n_reactions

    @property
    def n_edges(self) -> int:
        return self.edge_index.shape[1]

    @property
    def node_features(self) -> np.ndarray:
        """Concatenated node features (species first, then reactions).

        Species and reaction feature vectors are right-padded / truncated to
        the same dimensionality.
        """
        dim = max(self.species_features.shape[1], self.reaction_features.shape[1])
        sp = _pad_to(self.species_features, dim)
        rx = _pad_to(self.reaction_features, dim)
        return np.vstack([sp, rx])


def _pad_to(arr: np.ndarray, target_cols: int) -> np.ndarray:
    if arr.shape[1] >= target_cols:
        return arr[:, :target_cols]
    pad_width = target_cols - arr.shape[1]
    return np.pad(arr, ((0, 0), (0, pad_width)))


# ---------------------------------------------------------------------------
# BioModel → CircuitGraph conversion
# ---------------------------------------------------------------------------


def biomodel_to_graph(model: Any) -> CircuitGraph:
    """Convert a *BioModel* to a :class:`CircuitGraph`.

    Parameters
    ----------
    model
        ``BioModel`` instance with ``.species``, ``.reactions``, and
        ``.species_names`` attributes.
    """
    species_list = model.species
    reactions_list = model.reactions
    sp_names = [s.name for s in species_list]
    rx_names = [r.name for r in reactions_list]
    sp_idx = {name: i for i, name in enumerate(sp_names)}
    n_sp = len(sp_names)

    # -- species node features -----------------------------------------------
    sp_feats: List[np.ndarray] = []
    for sp in species_list:
        st = SPECIES_TYPE_MAP.get(sp.species_type.name, len(SPECIES_TYPE_MAP))
        init_conc = sp.initial_concentration
        lb = sp.concentration_bounds.lower if sp.concentration_bounds else 0.0
        ub = sp.concentration_bounds.upper if sp.concentration_bounds else 1e6
        cn = float(sp.copy_number) if sp.copy_number is not None else -1.0
        sp_feats.append(np.array([st, init_conc, lb, ub, cn], dtype=np.float64))
    species_features = np.stack(sp_feats) if sp_feats else np.empty((0, 5))

    # -- reaction node features ----------------------------------------------
    rx_feats: List[np.ndarray] = []
    for rxn in reactions_list:
        rt = _classify_kinetic_law(rxn.kinetic_law)
        params = rxn.kinetic_law.parameters if rxn.kinetic_law else {}
        k_vals = list(params.values()) if params else [0.0]
        rate_vals = [float(v) for v in k_vals]
        hill = _extract_hill_coeff(params)
        rx_feats.append(np.array(
            [rt, np.mean(rate_vals), np.max(rate_vals), hill],
            dtype=np.float64,
        ))
    reaction_features = np.stack(rx_feats) if rx_feats else np.empty((0, 4))

    # -- edges ---------------------------------------------------------------
    srcs: List[int] = []
    dsts: List[int] = []
    e_feats: List[np.ndarray] = []

    for j, rxn in enumerate(reactions_list):
        rx_node = n_sp + j
        # reactant → reaction
        for entry in rxn.reactants:
            si = sp_idx.get(entry.species_name)
            if si is not None:
                srcs.append(si)
                dsts.append(rx_node)
                e_feats.append(np.array([
                    float(entry.coefficient),
                    float(EdgeType.REACTANT),
                ], dtype=np.float64))
        # reaction → product
        for entry in rxn.products:
            si = sp_idx.get(entry.species_name)
            if si is not None:
                srcs.append(rx_node)
                dsts.append(si)
                e_feats.append(np.array([
                    float(entry.coefficient),
                    float(EdgeType.PRODUCT),
                ], dtype=np.float64))
        # modifier → reaction
        if rxn.modifiers:
            for mod in rxn.modifiers:
                si = sp_idx.get(mod)
                if si is not None:
                    srcs.append(si)
                    dsts.append(rx_node)
                    e_feats.append(np.array([
                        1.0,
                        float(EdgeType.MODIFIER),
                    ], dtype=np.float64))

    edge_index = np.array([srcs, dsts], dtype=np.int64) if srcs else np.empty((2, 0), dtype=np.int64)
    edge_features = np.stack(e_feats) if e_feats else np.empty((0, 2))

    # -- sparse adjacency ----------------------------------------------------
    n_nodes = n_sp + len(rx_names)
    if srcs:
        data = np.ones(len(srcs), dtype=np.float64)
        adj = sparse.csr_matrix(
            (data, (np.array(srcs), np.array(dsts))),
            shape=(n_nodes, n_nodes),
        )
    else:
        adj = sparse.csr_matrix((n_nodes, n_nodes), dtype=np.float64)

    return CircuitGraph(
        species_features=species_features,
        reaction_features=reaction_features,
        edge_index=edge_index,
        edge_features=edge_features,
        species_names=sp_names,
        reaction_names=rx_names,
        adjacency=adj,
    )


def _classify_kinetic_law(kl: Any) -> int:
    if kl is None:
        return REACTION_TYPE_MAP["OTHER"]
    name = type(kl).__name__
    mapping = {
        "MassAction": "MASS_ACTION",
        "HillActivation": "HILL_ACTIVATION",
        "HillRepression": "HILL_REPRESSION",
        "MichaelisMenten": "MICHAELIS_MENTEN",
        "ConstitutiveProduction": "CONSTITUTIVE",
        "LinearDegradation": "LINEAR_DEGRADATION",
        "DimerFormation": "DIMER_FORMATION",
    }
    return REACTION_TYPE_MAP.get(mapping.get(name, "OTHER"), REACTION_TYPE_MAP["OTHER"])


def _extract_hill_coeff(params: Dict[str, Any]) -> float:
    for k, v in params.items():
        if k.lower() in ("n", "hill", "hill_coefficient"):
            return float(v)
    return 1.0


# ---------------------------------------------------------------------------
# Neighbour sampler
# ---------------------------------------------------------------------------


class NeighborSampler:
    """Sample fixed-size neighbourhoods from a sparse adjacency matrix.

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        Adjacency matrix (directed).
    num_samples : int
        Maximum number of neighbours to sample per node.  ``-1`` = all.
    """

    def __init__(self, adj: sparse.csr_matrix, num_samples: int = -1) -> None:
        self.adj = adj
        self.num_samples = num_samples
        self._rng = np.random.RandomState(0)

    def sample(self, node_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(sampled_neighbor_ids, mapping)`` for each node in *node_ids*.

        Parameters
        ----------
        node_ids
            1-D array of node indices.

        Returns
        -------
        (neighbours, offsets)
            ``neighbours`` is a flat 1-D array; ``offsets`` of length
            ``len(node_ids) + 1`` marks the boundary for each node.
        """
        all_nbrs: List[np.ndarray] = []
        offsets = [0]
        for nid in node_ids:
            row = self.adj.getrow(nid)
            nbr_ids = row.indices.copy()
            if self.num_samples > 0 and len(nbr_ids) > self.num_samples:
                nbr_ids = self._rng.choice(nbr_ids, size=self.num_samples, replace=False)
            all_nbrs.append(nbr_ids)
            offsets.append(offsets[-1] + len(nbr_ids))
        if all_nbrs:
            neighbours = np.concatenate(all_nbrs)
        else:
            neighbours = np.empty(0, dtype=np.int64)
        return neighbours, np.array(offsets, dtype=np.int64)

    def sample_transpose(self, node_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sample from the *incoming* edges (transpose of adj)."""
        adj_t = self.adj.T.tocsr()
        sampler_t = NeighborSampler(adj_t, self.num_samples)
        sampler_t._rng = self._rng
        return sampler_t.sample(node_ids)


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------


def _mean_aggregate(
    node_features: np.ndarray,
    neighbours: np.ndarray,
    offsets: np.ndarray,
    n_target: int,
) -> np.ndarray:
    """Mean aggregation over variable-size neighbourhoods."""
    dim = node_features.shape[1]
    out = np.zeros((n_target, dim), dtype=np.float64)
    for i in range(n_target):
        start, end = offsets[i], offsets[i + 1]
        if end > start:
            out[i] = node_features[neighbours[start:end]].mean(axis=0)
    return out


def _max_pool_aggregate(
    node_features: np.ndarray,
    neighbours: np.ndarray,
    offsets: np.ndarray,
    n_target: int,
    W_pool: np.ndarray,
    b_pool: np.ndarray,
) -> np.ndarray:
    """Max-pool aggregation with a learnable linear transform."""
    dim_out = W_pool.shape[1]
    out = np.zeros((n_target, dim_out), dtype=np.float64)
    for i in range(n_target):
        start, end = offsets[i], offsets[i + 1]
        if end > start:
            nbr_feats = node_features[neighbours[start:end]]
            transformed = np.maximum(0.0, nbr_feats @ W_pool + b_pool)
            out[i] = transformed.max(axis=0)
    return out


def _lstm_aggregate(
    node_features: np.ndarray,
    neighbours: np.ndarray,
    offsets: np.ndarray,
    n_target: int,
    Wi: np.ndarray, Wf: np.ndarray, Wc: np.ndarray, Wo: np.ndarray,
    Ui: np.ndarray, Uf: np.ndarray, Uc: np.ndarray, Uo: np.ndarray,
    bi: np.ndarray, bf: np.ndarray, bc: np.ndarray, bo: np.ndarray,
) -> np.ndarray:
    """LSTM aggregation with a single-step unrolling per neighbour."""
    dim_h = Wi.shape[1]
    out = np.zeros((n_target, dim_h), dtype=np.float64)
    for i in range(n_target):
        start, end = offsets[i], offsets[i + 1]
        h = np.zeros(dim_h, dtype=np.float64)
        c = np.zeros(dim_h, dtype=np.float64)
        idx_range = np.arange(start, end)
        np.random.shuffle(idx_range)
        for j in idx_range:
            x = node_features[neighbours[j]]
            ig = _sigmoid(x @ Wi + h @ Ui + bi)
            fg = _sigmoid(x @ Wf + h @ Uf + bf)
            cand = np.tanh(x @ Wc + h @ Uc + bc)
            og = _sigmoid(x @ Wo + h @ Uo + bo)
            c = fg * c + ig * cand
            h = og * np.tanh(c)
        out[i] = h
    return out


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ---------------------------------------------------------------------------
# GraphSAGE layer
# ---------------------------------------------------------------------------


@dataclass
class GraphSAGELayerWeights:
    """Weights for a single GraphSAGE layer."""

    W_self: np.ndarray  # (in_dim, out_dim)
    W_neigh: np.ndarray  # (in_dim, out_dim) or (agg_dim, out_dim)
    bias: np.ndarray  # (out_dim,)
    # Optional pool weights (used only by max-pool / LSTM aggregators)
    W_pool: Optional[np.ndarray] = None
    b_pool: Optional[np.ndarray] = None
    lstm_params: Optional[Dict[str, np.ndarray]] = None


class GraphSAGELayer:
    """A single GraphSAGE message-passing layer (CPU / NumPy only).

    Parameters
    ----------
    weights : GraphSAGELayerWeights
    aggregator : str
        ``"mean"``, ``"max_pool"``, or ``"lstm"``.
    activation : str
        ``"relu"`` or ``"none"``.
    """

    def __init__(
        self,
        weights: GraphSAGELayerWeights,
        aggregator: str = "mean",
        activation: str = "relu",
    ) -> None:
        self.weights = weights
        self.aggregator = aggregator
        self.activation = activation

    def forward(
        self,
        node_features: np.ndarray,
        neighbours: np.ndarray,
        offsets: np.ndarray,
        node_ids: np.ndarray,
    ) -> np.ndarray:
        """Compute updated embeddings for *node_ids*.

        Parameters
        ----------
        node_features : (n_nodes, in_dim)
        neighbours : flat array from :class:`NeighborSampler`
        offsets : array from :class:`NeighborSampler`
        node_ids : indices to update

        Returns
        -------
        np.ndarray of shape ``(len(node_ids), out_dim)``
        """
        n = len(node_ids)
        w = self.weights

        # Self features
        h_self = node_features[node_ids] @ w.W_self  # (n, out_dim)

        # Neighbour aggregation
        if self.aggregator == "mean":
            h_neigh_raw = _mean_aggregate(node_features, neighbours, offsets, n)
            h_neigh = h_neigh_raw @ w.W_neigh
        elif self.aggregator == "max_pool":
            assert w.W_pool is not None and w.b_pool is not None
            h_neigh = _max_pool_aggregate(
                node_features, neighbours, offsets, n, w.W_pool, w.b_pool,
            )
            h_neigh = h_neigh @ w.W_neigh
        elif self.aggregator == "lstm":
            assert w.lstm_params is not None
            lp = w.lstm_params
            h_neigh_raw = _lstm_aggregate(
                node_features, neighbours, offsets, n,
                lp["Wi"], lp["Wf"], lp["Wc"], lp["Wo"],
                lp["Ui"], lp["Uf"], lp["Uc"], lp["Uo"],
                lp["bi"], lp["bf"], lp["bc"], lp["bo"],
            )
            h_neigh = h_neigh_raw @ w.W_neigh
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator!r}")

        h = h_self + h_neigh + w.bias  # (n, out_dim)

        if self.activation == "relu":
            h = np.maximum(0.0, h)

        # L2-normalise rows
        norms = np.linalg.norm(h, axis=1, keepdims=True)
        h = h / np.maximum(norms, _EPS)
        return h


# ---------------------------------------------------------------------------
# Multi-layer GraphSAGE encoder
# ---------------------------------------------------------------------------


class GraphSAGEEncoder:
    """Multi-layer GraphSAGE encoder with skip connections.

    Parameters
    ----------
    layers : list[GraphSAGELayer]
    sampler : NeighborSampler
    use_skip : bool
        If *True*, add skip (residual) connections between layers when the
        dimensionality matches.
    """

    def __init__(
        self,
        layers: List[GraphSAGELayer],
        sampler: NeighborSampler,
        use_skip: bool = True,
    ) -> None:
        self.layers = layers
        self.sampler = sampler
        self.use_skip = use_skip

    def encode(
        self,
        node_features: np.ndarray,
        node_ids: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Run multi-layer message passing and return embeddings.

        Parameters
        ----------
        node_features : (n_nodes, feat_dim)
        node_ids : which nodes to compute embeddings for.
            Default = all nodes.

        Returns
        -------
        np.ndarray of shape ``(len(node_ids), embed_dim)``
        """
        n_nodes = node_features.shape[0]
        if node_ids is None:
            node_ids = np.arange(n_nodes, dtype=np.int64)

        h = node_features.copy()
        for layer in self.layers:
            nbrs, offs = self.sampler.sample(node_ids)
            h_new_partial = layer.forward(h, nbrs, offs, node_ids)
            out_dim = h_new_partial.shape[1]

            # Expand to full matrix so subsequent layers see updated values
            h_new = np.zeros((n_nodes, out_dim), dtype=np.float64)
            # Copy existing features (truncated/padded to new dim)
            copy_dim = min(h.shape[1], out_dim)
            h_new[:, :copy_dim] = h[:, :copy_dim]
            for idx_i, nid in enumerate(node_ids):
                h_new[nid] = h_new_partial[idx_i]

            # Skip connection when dims match
            if self.use_skip and h.shape[1] == out_dim:
                h = h + h_new
            else:
                h = h_new

        return h[node_ids]

    @classmethod
    def from_config(
        cls,
        in_dim: int,
        hidden_dims: List[int],
        adj: sparse.csr_matrix,
        aggregator: str = "mean",
        num_samples: int = 10,
        use_skip: bool = True,
        rng: Optional[np.random.RandomState] = None,
    ) -> "GraphSAGEEncoder":
        """Build an encoder with Xavier-initialised random weights.

        Parameters
        ----------
        in_dim : input feature dimensionality
        hidden_dims : list of hidden dimensions for each layer
        adj : sparse adjacency
        aggregator : ``"mean"`` | ``"max_pool"`` | ``"lstm"``
        num_samples : neighbours per node per layer
        use_skip : add residual connections
        rng : random state
        """
        if rng is None:
            rng = np.random.RandomState(42)

        dims = [in_dim] + hidden_dims
        layers: List[GraphSAGELayer] = []
        for i in range(len(dims) - 1):
            d_in, d_out = dims[i], dims[i + 1]
            scale = math.sqrt(2.0 / (d_in + d_out))
            W_self = rng.randn(d_in, d_out).astype(np.float64) * scale
            W_neigh = rng.randn(d_in, d_out).astype(np.float64) * scale
            bias = np.zeros(d_out, dtype=np.float64)
            W_pool = None
            b_pool = None
            lstm_params = None
            if aggregator == "max_pool":
                W_pool = rng.randn(d_in, d_out).astype(np.float64) * scale
                b_pool = np.zeros(d_out, dtype=np.float64)
            elif aggregator == "lstm":
                lstm_params = _init_lstm_params(d_in, d_out, rng)

            act = "relu" if i < len(dims) - 2 else "none"
            lw = GraphSAGELayerWeights(
                W_self=W_self, W_neigh=W_neigh, bias=bias,
                W_pool=W_pool, b_pool=b_pool, lstm_params=lstm_params,
            )
            layers.append(GraphSAGELayer(lw, aggregator=aggregator, activation=act))

        sampler = NeighborSampler(adj, num_samples=num_samples)
        return cls(layers, sampler, use_skip=use_skip)


def _init_lstm_params(
    d_in: int, d_h: int, rng: np.random.RandomState,
) -> Dict[str, np.ndarray]:
    scale = math.sqrt(2.0 / (d_in + d_h))
    params: Dict[str, np.ndarray] = {}
    for gate in ("i", "f", "c", "o"):
        params[f"W{gate}"] = rng.randn(d_in, d_h).astype(np.float64) * scale
        params[f"U{gate}"] = rng.randn(d_h, d_h).astype(np.float64) * scale
        params[f"b{gate}"] = np.zeros(d_h, dtype=np.float64)
    return params


# ---------------------------------------------------------------------------
# Graph-level readout
# ---------------------------------------------------------------------------


def readout_mean(embeddings: np.ndarray) -> np.ndarray:
    """Mean readout over all node embeddings → single graph vector."""
    return embeddings.mean(axis=0)


def readout_max(embeddings: np.ndarray) -> np.ndarray:
    """Max readout over all node embeddings → single graph vector."""
    return embeddings.max(axis=0)


def readout_attention(
    embeddings: np.ndarray,
    W_att: np.ndarray,
    b_att: np.ndarray,
) -> np.ndarray:
    """Attention-weighted readout.

    Parameters
    ----------
    embeddings : (n_nodes, dim)
    W_att : (dim, 1)
    b_att : (1,)

    Returns
    -------
    np.ndarray of shape ``(dim,)``
    """
    scores = embeddings @ W_att + b_att  # (n, 1)
    weights = _softmax(scores.ravel())  # (n,)
    return (embeddings * weights[:, None]).sum(axis=0)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / (e.sum() + _EPS)


class GraphReadout:
    """Configurable readout that combines mean, max, and/or attention."""

    def __init__(
        self,
        mode: str = "mean",
        W_att: Optional[np.ndarray] = None,
        b_att: Optional[np.ndarray] = None,
    ) -> None:
        self.mode = mode
        self.W_att = W_att
        self.b_att = b_att

    def __call__(self, embeddings: np.ndarray) -> np.ndarray:
        if self.mode == "mean":
            return readout_mean(embeddings)
        elif self.mode == "max":
            return readout_max(embeddings)
        elif self.mode == "attention":
            assert self.W_att is not None and self.b_att is not None
            return readout_attention(embeddings, self.W_att, self.b_att)
        elif self.mode == "mean_max":
            return np.concatenate([readout_mean(embeddings), readout_max(embeddings)])
        else:
            raise ValueError(f"Unknown readout mode: {self.mode!r}")


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


@dataclass
class BatchedGraphs:
    """A batched representation of multiple :class:`CircuitGraph` instances."""

    node_features: np.ndarray  # (total_nodes, feat_dim)
    adjacency: sparse.csr_matrix
    batch_ids: np.ndarray  # (total_nodes,) – which graph each node belongs to
    graph_offsets: np.ndarray  # (n_graphs + 1,)
    n_graphs: int


def batch_graphs(graphs: List[CircuitGraph]) -> BatchedGraphs:
    """Combine multiple :class:`CircuitGraph` into a single batched graph.

    Parameters
    ----------
    graphs : list of CircuitGraph

    Returns
    -------
    BatchedGraphs
    """
    if not graphs:
        return BatchedGraphs(
            node_features=np.empty((0, 0)),
            adjacency=sparse.csr_matrix((0, 0)),
            batch_ids=np.empty(0, dtype=np.int64),
            graph_offsets=np.array([0], dtype=np.int64),
            n_graphs=0,
        )

    feat_dim = max(g.node_features.shape[1] for g in graphs)
    all_feats: List[np.ndarray] = []
    all_rows: List[np.ndarray] = []
    all_cols: List[np.ndarray] = []
    batch_ids: List[np.ndarray] = []
    offsets = [0]

    offset = 0
    for gid, g in enumerate(graphs):
        nf = _pad_to(g.node_features, feat_dim)
        all_feats.append(nf)

        if g.edge_index.shape[1] > 0:
            all_rows.append(g.edge_index[0] + offset)
            all_cols.append(g.edge_index[1] + offset)

        batch_ids.append(np.full(g.n_nodes, gid, dtype=np.int64))
        offset += g.n_nodes
        offsets.append(offset)

    node_features = np.vstack(all_feats)
    total = node_features.shape[0]

    if all_rows:
        rows = np.concatenate(all_rows)
        cols = np.concatenate(all_cols)
        data = np.ones(len(rows), dtype=np.float64)
        adjacency = sparse.csr_matrix((data, (rows, cols)), shape=(total, total))
    else:
        adjacency = sparse.csr_matrix((total, total), dtype=np.float64)

    return BatchedGraphs(
        node_features=node_features,
        adjacency=adjacency,
        batch_ids=np.concatenate(batch_ids),
        graph_offsets=np.array(offsets, dtype=np.int64),
        n_graphs=len(graphs),
    )


def unbatch_embeddings(
    embeddings: np.ndarray, batched: BatchedGraphs,
) -> List[np.ndarray]:
    """Split batched embeddings back into per-graph arrays."""
    result: List[np.ndarray] = []
    for i in range(batched.n_graphs):
        start = batched.graph_offsets[i]
        end = batched.graph_offsets[i + 1]
        result.append(embeddings[start:end])
    return result


# ---------------------------------------------------------------------------
# Feature normalization for node features
# ---------------------------------------------------------------------------


class NodeFeatureNormalizer:
    """Per-feature Z-score normaliser fitted on training graphs."""

    def __init__(self) -> None:
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    def fit(self, graphs: List[CircuitGraph]) -> None:
        all_feats = np.vstack([g.node_features for g in graphs])
        self._mean = all_feats.mean(axis=0)
        self._std = all_feats.std(axis=0) + _EPS

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self._mean is None or self._std is None:
            return features
        return (features - self._mean) / self._std

    def fit_transform(self, graphs: List[CircuitGraph]) -> List[np.ndarray]:
        self.fit(graphs)
        return [self.transform(g.node_features) for g in graphs]

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean": self._mean.tolist() if self._mean is not None else None,
            "std": self._std.tolist() if self._std is not None else None,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "NodeFeatureNormalizer":
        obj = cls()
        if d["mean"] is not None:
            obj._mean = np.asarray(d["mean"])
            obj._std = np.asarray(d["std"])
        return obj


# ---------------------------------------------------------------------------
# Convenience: full encode pipeline
# ---------------------------------------------------------------------------


class CircuitEncoder:
    """End-to-end encoder: BioModel → fixed-dim embedding vector.

    Parameters
    ----------
    encoder : GraphSAGEEncoder
    readout : GraphReadout
    normalizer : NodeFeatureNormalizer or None
    """

    def __init__(
        self,
        encoder: GraphSAGEEncoder,
        readout: GraphReadout,
        normalizer: Optional[NodeFeatureNormalizer] = None,
    ) -> None:
        self.encoder = encoder
        self.readout = readout
        self.normalizer = normalizer

    def encode_model(self, model: Any) -> np.ndarray:
        """Encode a single :class:`BioModel` to a 1-D embedding."""
        graph = biomodel_to_graph(model)
        feats = graph.node_features
        if self.normalizer is not None:
            feats = self.normalizer.transform(feats)

        # Re-create encoder's sampler for this graph's adjacency
        sampler = NeighborSampler(graph.adjacency, self.encoder.sampler.num_samples)
        old_sampler = self.encoder.sampler
        self.encoder.sampler = sampler
        try:
            node_embeds = self.encoder.encode(feats)
        finally:
            self.encoder.sampler = old_sampler

        return self.readout(node_embeds)

    def encode_graph(self, graph: CircuitGraph) -> np.ndarray:
        """Encode a pre-built :class:`CircuitGraph`."""
        feats = graph.node_features
        if self.normalizer is not None:
            feats = self.normalizer.transform(feats)

        sampler = NeighborSampler(graph.adjacency, self.encoder.sampler.num_samples)
        old_sampler = self.encoder.sampler
        self.encoder.sampler = sampler
        try:
            node_embeds = self.encoder.encode(feats)
        finally:
            self.encoder.sampler = old_sampler

        return self.readout(node_embeds)

    def encode_batch(self, models: List[Any]) -> List[np.ndarray]:
        """Encode a list of :class:`BioModel` to embeddings."""
        return [self.encode_model(m) for m in models]

    @classmethod
    def from_config(
        cls,
        in_dim: int = 8,
        hidden_dims: Optional[List[int]] = None,
        readout_mode: str = "mean",
        aggregator: str = "mean",
        num_samples: int = 10,
        rng: Optional[np.random.RandomState] = None,
    ) -> "CircuitEncoder":
        """Build a :class:`CircuitEncoder` with random weights (for shape validation)."""
        if hidden_dims is None:
            hidden_dims = [32, 32]
        if rng is None:
            rng = np.random.RandomState(42)

        dummy_adj = sparse.csr_matrix((1, 1), dtype=np.float64)
        sage = GraphSAGEEncoder.from_config(
            in_dim, hidden_dims, dummy_adj,
            aggregator=aggregator, num_samples=num_samples,
            rng=rng,
        )
        ro = GraphReadout(mode=readout_mode)
        return cls(sage, ro)
