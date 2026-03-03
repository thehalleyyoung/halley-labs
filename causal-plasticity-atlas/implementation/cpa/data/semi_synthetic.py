"""Semi-synthetic benchmark datasets.

Provides loaders and generators for semi-synthetic datasets based on
real-world causal network structures (Sachs, ALARM, Insurance) with
synthetically generated multi-context samples.

Each generator creates a base linear-Gaussian SCM from a known
network topology and then introduces controlled plasticity
variations across contexts.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from cpa.core.types import PlasticityClass


# ===================================================================
# DAG constructors for well-known networks
# ===================================================================


def _create_sachs_dag() -> Tuple[NDArray, List[str]]:
    """Create the 11-node, 17-edge Sachs protein signalling DAG.

    Network from Sachs et al. (2005) "Causal Protein-Signaling
    Networks Derived from Multiparameter Single-Cell Data".

    Returns
    -------
    (adj, names) : tuple
        Adjacency matrix (11, 11) and variable names.
    """
    names = [
        "Raf", "Mek", "Plcg", "PIP2", "PIP3",
        "Erk", "Akt", "PKA", "PKC", "P38", "Jnk",
    ]
    p = len(names)
    adj = np.zeros((p, p), dtype=np.float64)

    edges = [
        (0, 1),   # Raf → Mek
        (1, 5),   # Mek → Erk
        (2, 3),   # Plcg → PIP2
        (2, 4),   # Plcg → PIP3
        (4, 3),   # PIP3 → PIP2
        (7, 0),   # PKA → Raf
        (7, 1),   # PKA → Mek
        (7, 5),   # PKA → Erk
        (7, 6),   # PKA → Akt
        (7, 9),   # PKA → P38
        (7, 10),  # PKA → Jnk
        (8, 0),   # PKC → Raf
        (8, 1),   # PKC → Mek
        (8, 7),   # PKC → PKA
        (8, 9),   # PKC → P38
        (8, 10),  # PKC → Jnk
        (8, 2),   # PKC → Plcg
    ]
    for i, j in edges:
        adj[i, j] = 1.0

    return adj, names


def _create_alarm_dag() -> Tuple[NDArray, List[str]]:
    """Create a simplified 20-node ALARM medical diagnosis DAG.

    Simplified from the original 37-node ALARM network (Beinlich et al., 1989).

    Returns
    -------
    (adj, names) : tuple
    """
    names = [
        "HYPOVOLEMIA", "LVFAILURE", "ANAPHYLAXIS", "INSUFFANESTH",
        "PULMEMBOLUS", "INTUBATION", "KINKEDTUBE", "DISCONNECT",
        "MINVOLSET", "VENTMACH", "VENTTUBE", "VENTLUNG",
        "VENTALV", "ARTCO2", "CATECHOL", "HR",
        "STROKEVOLUME", "CO", "BP", "HISTORY",
    ]
    p = len(names)
    adj = np.zeros((p, p), dtype=np.float64)

    edges = [
        (0, 16),   # HYPOVOLEMIA → STROKEVOLUME
        (0, 14),   # HYPOVOLEMIA → CATECHOL
        (1, 16),   # LVFAILURE → STROKEVOLUME
        (1, 19),   # LVFAILURE → HISTORY
        (2, 14),   # ANAPHYLAXIS → CATECHOL
        (3, 14),   # INSUFFANESTH → CATECHOL
        (4, 11),   # PULMEMBOLUS → VENTLUNG
        (5, 11),   # INTUBATION → VENTLUNG
        (5, 10),   # INTUBATION → VENTTUBE
        (6, 10),   # KINKEDTUBE → VENTTUBE
        (7, 10),   # DISCONNECT → VENTTUBE
        (8, 9),    # MINVOLSET → VENTMACH
        (9, 10),   # VENTMACH → VENTTUBE
        (10, 11),  # VENTTUBE → VENTLUNG
        (11, 12),  # VENTLUNG → VENTALV
        (12, 13),  # VENTALV → ARTCO2
        (13, 14),  # ARTCO2 → CATECHOL
        (14, 15),  # CATECHOL → HR
        (16, 17),  # STROKEVOLUME → CO
        (15, 17),  # HR → CO
        (17, 18),  # CO → BP
    ]
    for i, j in edges:
        adj[i, j] = 1.0

    return adj, names


def _create_insurance_dag() -> Tuple[NDArray, List[str]]:
    """Create a simplified 15-node Insurance network DAG.

    Simplified from the original 27-node Insurance evaluation network.

    Returns
    -------
    (adj, names) : tuple
    """
    names = [
        "Age", "SocioEcon", "GoodStudent", "RiskAversion",
        "VehicleYear", "MakeModel", "Antilock", "DrivQuality",
        "DrivHist", "Accident", "RuggedAuto", "Cushioning",
        "MedCost", "ILiCost", "PropCost",
    ]
    p = len(names)
    adj = np.zeros((p, p), dtype=np.float64)

    edges = [
        (0, 1),    # Age → SocioEcon
        (0, 2),    # Age → GoodStudent
        (0, 3),    # Age → RiskAversion
        (1, 2),    # SocioEcon → GoodStudent
        (1, 4),    # SocioEcon → VehicleYear
        (1, 5),    # SocioEcon → MakeModel
        (3, 7),    # RiskAversion → DrivQuality
        (3, 6),    # RiskAversion → Antilock
        (4, 6),    # VehicleYear → Antilock
        (4, 10),   # VehicleYear → RuggedAuto
        (5, 10),   # MakeModel → RuggedAuto
        (5, 11),   # MakeModel → Cushioning
        (7, 9),    # DrivQuality → Accident
        (7, 8),    # DrivQuality → DrivHist
        (6, 9),    # Antilock → Accident
        (10, 11),  # RuggedAuto → Cushioning
        (9, 12),   # Accident → MedCost
        (9, 13),   # Accident → ILiCost
        (11, 12),  # Cushioning → MedCost
        (9, 14),   # Accident → PropCost
        (10, 14),  # RuggedAuto → PropCost
    ]
    for i, j in edges:
        adj[i, j] = 1.0

    return adj, names


# ===================================================================
# Context variation
# ===================================================================


def _add_context_variation(
    base_weights: NDArray,
    base_dag: NDArray,
    variation_type: str,
    rng: np.random.Generator,
    strength: float = 0.5,
) -> Tuple[NDArray, NDArray]:
    """Add plasticity-inducing variation to a base network.

    Parameters
    ----------
    base_weights : NDArray
        Weighted adjacency matrix.
    base_dag : NDArray
        Binary adjacency matrix.
    variation_type : str
        One of ``"parametric"``, ``"structural"``, ``"emergence"``,
        ``"mixed"``.
    rng : Generator
    strength : float
        Variation magnitude.

    Returns
    -------
    (new_weights, new_dag) : tuple
    """
    p = base_dag.shape[0]
    new_dag = base_dag.copy()
    new_weights = base_weights.copy()

    if variation_type == "parametric":
        edges = np.argwhere(base_dag != 0)
        for i, j in edges:
            new_weights[i, j] += rng.normal(0, strength)

    elif variation_type == "structural":
        edges = list(zip(*np.where(base_dag != 0)))
        n_change = max(1, int(len(edges) * 0.2))
        for _ in range(n_change):
            if edges and rng.random() < 0.5:
                idx = rng.integers(len(edges))
                ei, ej = edges[idx]
                new_dag[ei, ej] = 0.0
                new_weights[ei, ej] = 0.0
            else:
                tries = 0
                while tries < 20:
                    i_new = rng.integers(p)
                    j_new = rng.integers(p)
                    if i_new != j_new and new_dag[i_new, j_new] == 0:
                        new_dag[i_new, j_new] = 1.0
                        if _has_cycle(new_dag):
                            new_dag[i_new, j_new] = 0.0
                        else:
                            new_weights[i_new, j_new] = rng.normal(0, 1.0)
                            break
                    tries += 1

    elif variation_type == "emergence":
        n_emerge = max(1, int(p * 0.15))
        for _ in range(n_emerge):
            tries = 0
            while tries < 20:
                i_new = rng.integers(p)
                j_new = rng.integers(p)
                if i_new != j_new and new_dag[i_new, j_new] == 0:
                    new_dag[i_new, j_new] = 1.0
                    if _has_cycle(new_dag):
                        new_dag[i_new, j_new] = 0.0
                    else:
                        new_weights[i_new, j_new] = rng.normal(0, 1.5)
                        break
                tries += 1

    elif variation_type == "mixed":
        new_weights, new_dag = _add_context_variation(
            new_weights, new_dag, "parametric", rng, strength * 0.5
        )
        new_weights, new_dag = _add_context_variation(
            new_weights, new_dag, "structural", rng, strength * 0.5
        )

    return new_weights, new_dag


def _has_cycle(adj: NDArray) -> bool:
    """Check for cycles via topological sort."""
    p = adj.shape[0]
    in_deg = np.sum(adj != 0, axis=0).astype(int)
    queue = [i for i in range(p) if in_deg[i] == 0]
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for j in range(p):
            if adj[node, j] != 0:
                in_deg[j] -= 1
                if in_deg[j] == 0:
                    queue.append(j)
    return visited != p


def _sample_from_weighted_dag(
    adj_weights: NDArray,
    dag: NDArray,
    n: int,
    noise_std: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> NDArray:
    """Sample from a linear-Gaussian SCM."""
    rng = rng or np.random.default_rng()
    p = dag.shape[0]

    in_deg = np.sum(dag != 0, axis=0).astype(int)
    queue = [i for i in range(p) if in_deg[i] == 0]
    order: List[int] = []
    temp_deg = in_deg.copy()
    while queue:
        node = queue.pop(0)
        order.append(node)
        for j in range(p):
            if dag[node, j] != 0:
                temp_deg[j] -= 1
                if temp_deg[j] == 0:
                    queue.append(j)

    data = np.zeros((n, p), dtype=np.float64)
    for j in order:
        parents = np.where(dag[:, j] != 0)[0]
        noise = rng.normal(0, noise_std, size=n)
        if len(parents) == 0:
            data[:, j] = noise
        else:
            data[:, j] = sum(adj_weights[pa, j] * data[:, pa] for pa in parents) + noise
    return data


def _generate_multi_context(
    dag_fn,
    n_per_context: int,
    n_contexts: int,
    seed: Optional[int],
    variation_types: Optional[List[str]] = None,
) -> Tuple[List[NDArray], List[NDArray], List[str]]:
    """Core multi-context generation logic."""
    rng = np.random.default_rng(seed)
    base_dag, names = dag_fn()
    base_weights = np.zeros_like(base_dag)
    edges = np.argwhere(base_dag != 0)
    for i, j in edges:
        w = rng.uniform(0.5, 2.0)
        if rng.random() < 0.5:
            w = -w
        base_weights[i, j] = w

    if variation_types is None:
        variation_types = ["parametric", "structural", "emergence", "mixed", "parametric"]

    datasets: List[NDArray] = []
    dag_list: List[NDArray] = []
    labels: List[str] = []

    for k in range(n_contexts):
        if k == 0:
            w_k = base_weights.copy()
            d_k = base_dag.copy()
        else:
            vtype = variation_types[k % len(variation_types)]
            w_k, d_k = _add_context_variation(
                base_weights, base_dag, vtype, rng, strength=0.3 + 0.1 * k
            )

        data = _sample_from_weighted_dag(w_k, d_k, n_per_context, rng=rng)
        datasets.append(data)
        dag_list.append(d_k)
        labels.append(f"ctx_{k}")

    return datasets, dag_list, labels


# ===================================================================
# SemiSyntheticLoader
# ===================================================================


class SemiSyntheticLoader:
    """Load and manage semi-synthetic benchmark datasets.

    Can generate data on-the-fly from known network topologies
    or load from a data directory.

    Parameters
    ----------
    data_dir : str or Path or None
        Directory containing pre-generated datasets.
    """

    def __init__(self, data_dir: Optional[str] = None) -> None:
        self._data_dir = Path(data_dir) if data_dir else None
        self._generators = {
            "sachs": (_create_sachs_dag, 11),
            "alarm": (_create_alarm_dag, 20),
            "insurance": (_create_insurance_dag, 15),
        }

    def load(self, name: str, n_samples: int = 1000, seed: Optional[int] = None) -> Tuple[NDArray, NDArray]:
        """Load or generate a named semi-synthetic dataset.

        Parameters
        ----------
        name : str
            Dataset name (``"sachs"``, ``"alarm"``, or ``"insurance"``).
        n_samples : int
            Number of samples if generating.
        seed : int or None

        Returns
        -------
        (data, true_dag)
        """
        if self._data_dir is not None:
            data_path = self._data_dir / f"{name}_data.npy"
            dag_path = self._data_dir / f"{name}_dag.npy"
            if data_path.exists() and dag_path.exists():
                return np.load(data_path), np.load(dag_path)

        if name not in self._generators:
            raise ValueError(
                f"Unknown dataset {name!r}. Available: {list(self._generators)}"
            )

        dag_fn, _ = self._generators[name]
        dag, names = dag_fn()
        rng = np.random.default_rng(seed)
        weights = np.zeros_like(dag)
        edges = np.argwhere(dag != 0)
        for i, j in edges:
            w = rng.uniform(0.5, 2.0)
            if rng.random() < 0.5:
                w = -w
            weights[i, j] = w

        data = _sample_from_weighted_dag(weights, dag, n_samples, rng=rng)
        return data, dag

    def available_datasets(self) -> List[str]:
        """List names of all available semi-synthetic datasets."""
        available = list(self._generators.keys())
        if self._data_dir is not None and self._data_dir.exists():
            for f in self._data_dir.glob("*_dag.npy"):
                name = f.stem.replace("_dag", "")
                if name not in available:
                    available.append(name)
        return sorted(available)


# ===================================================================
# Module-level convenience functions
# ===================================================================


def sachs_network(
    n_samples: int = 1000,
    n_per_context: int = 1000,
    n_contexts: int = 5,
    seed: Optional[int] = None,
) -> Tuple[List[NDArray], List[NDArray], List[str]]:
    """Generate multi-context data from the Sachs protein-signalling network.

    Parameters
    ----------
    n_samples : int
        Ignored if n_per_context is given (kept for API compat).
    n_per_context : int
        Samples per context.
    n_contexts : int
        Number of contexts.
    seed : int or None

    Returns
    -------
    (datasets, dags, labels)
    """
    return _generate_multi_context(
        _create_sachs_dag, n_per_context, n_contexts, seed
    )


def alarm_network(
    n_samples: int = 500,
    n_per_context: int = 500,
    n_contexts: int = 5,
    seed: Optional[int] = None,
) -> Tuple[List[NDArray], List[NDArray], List[str]]:
    """Generate multi-context data from the ALARM monitoring network."""
    return _generate_multi_context(
        _create_alarm_dag, n_per_context, n_contexts, seed
    )


def insurance_network(
    n_samples: int = 500,
    n_per_context: int = 500,
    n_contexts: int = 3,
    seed: Optional[int] = None,
) -> Tuple[List[NDArray], List[NDArray], List[str]]:
    """Generate multi-context data from the Insurance evaluation network."""
    return _generate_multi_context(
        _create_insurance_dag, n_per_context, n_contexts, seed
    )
