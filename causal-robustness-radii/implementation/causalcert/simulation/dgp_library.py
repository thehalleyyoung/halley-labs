"""
Library of standard data-generating processes for causal inference benchmarks.

Each DGP provides:
- A ground-truth causal DAG
- True average treatment effect (ATE)
- True robustness radius (when analytically known)
- Data generation via configurable SCM engines

DGPs
----
- :class:`LaLondeDGP` — Classic job-training evaluation (8 variables).
- :class:`SmokingBirthweightDGP` — Smoking/birthweight (12 variables).
- :class:`IHDPSimulation` — Infant health & development (25 variables).
- :class:`InstrumentDGP` — IV setting with compliance types.
- :class:`MediationDGP` — Mediation with direct and indirect effects.
- :class:`ConfoundedDGP` — Strong confounding setup.
- :class:`FaithfulnessViolationDGP` — Near-faithfulness violations.
- :class:`SparseHighDimDGP` — Sparse high-dimensional (p=50).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from causalcert.types import AdjacencyMatrix, NodeId, NodeSet, VariableType
from causalcert.simulation.types import DGPSpec, GroundTruth
from causalcert.simulation.engines import (
    LinearGaussianEngine,
    NonlinearEngine,
    MixedTypeEngine,
    InterventionalEngine,
    _topological_order,
    _parents,
)
from causalcert.simulation.noise_models import GaussianNoise


# ============================================================================
# Helpers
# ============================================================================


def _adj_from_edges(n: int, edges: list[tuple[int, int]]) -> AdjacencyMatrix:
    """Build an adjacency matrix from an edge list."""
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        adj[u, v] = 1
    return adj


def _linear_total_effect(
    adj: AdjacencyMatrix,
    weights: NDArray[np.float64],
    treatment: int,
    outcome: int,
) -> float:
    """True total effect in a linear SEM via matrix inversion."""
    n = adj.shape[0]
    M = np.eye(n) - weights.T
    try:
        inv_M = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return 0.0
    return float(inv_M[treatment, outcome])


def _find_confounders(
    adj: AdjacencyMatrix, treatment: int, outcome: int
) -> NodeSet:
    """Find common ancestors of treatment and outcome (confounders)."""
    n = adj.shape[0]
    # BFS ancestors
    def _ancestors(v: int) -> set[int]:
        visited: set[int] = set()
        stack = list(np.nonzero(adj[:, v])[0].astype(int))
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(
                    int(p) for p in np.nonzero(adj[:, node])[0] if p not in visited
                )
        return visited

    anc_t = _ancestors(treatment)
    anc_y = _ancestors(outcome)
    return frozenset(anc_t & anc_y)


def _find_mediators(
    adj: AdjacencyMatrix, treatment: int, outcome: int
) -> NodeSet:
    """Find nodes on directed paths from treatment to outcome."""
    from collections import deque

    n = adj.shape[0]
    # Find all nodes reachable from treatment that can reach outcome
    # Forward from treatment
    fwd: set[int] = set()
    queue = deque([treatment])
    while queue:
        v = queue.popleft()
        for c in np.nonzero(adj[v])[0]:
            c = int(c)
            if c not in fwd:
                fwd.add(c)
                queue.append(c)
    # Backward from outcome
    bwd: set[int] = set()
    queue = deque([outcome])
    while queue:
        v = queue.popleft()
        for p in np.nonzero(adj[:, v])[0]:
            p = int(p)
            if p not in bwd:
                bwd.add(p)
                queue.append(p)
    mediators = fwd & bwd - {treatment, outcome}
    return frozenset(mediators)


# ============================================================================
# LaLondeDGP
# ============================================================================


@dataclass(slots=True)
class LaLondeDGP:
    """Classic LaLonde job-training evaluation DGP (8 variables).

    Variables: age, education, black, hispanic, married,
    nodegree, treatment, earnings.

    Parameters
    ----------
    n_samples : int
        Number of observations.
    treatment_effect : float
        True ATE of the job-training programme.
    noise_scale : float
        Noise standard deviation.
    seed : int
        Random seed.
    """

    n_samples: int = 1000
    treatment_effect: float = 1794.0
    noise_scale: float = 1.0
    seed: int = 42

    # Node indices
    AGE: int = 0
    EDUCATION: int = 1
    BLACK: int = 2
    HISPANIC: int = 3
    MARRIED: int = 4
    NODEGREE: int = 5
    TREATMENT: int = 6
    EARNINGS: int = 7
    N_NODES: int = 8

    _names: tuple[str, ...] = (
        "age", "education", "black", "hispanic",
        "married", "nodegree", "treatment", "earnings",
    )

    def dag(self) -> AdjacencyMatrix:
        """Return the ground-truth DAG."""
        edges = [
            (self.AGE, self.EDUCATION),
            (self.AGE, self.MARRIED),
            (self.AGE, self.TREATMENT),
            (self.AGE, self.EARNINGS),
            (self.EDUCATION, self.NODEGREE),
            (self.EDUCATION, self.TREATMENT),
            (self.EDUCATION, self.EARNINGS),
            (self.BLACK, self.TREATMENT),
            (self.BLACK, self.EARNINGS),
            (self.HISPANIC, self.TREATMENT),
            (self.HISPANIC, self.EARNINGS),
            (self.MARRIED, self.EARNINGS),
            (self.NODEGREE, self.TREATMENT),
            (self.NODEGREE, self.EARNINGS),
            (self.TREATMENT, self.EARNINGS),
        ]
        return _adj_from_edges(self.N_NODES, edges)

    def generate(self, rng: np.random.Generator | None = None) -> pd.DataFrame:
        """Generate observational data.

        Parameters
        ----------
        rng : Generator | None
            Random state.

        Returns
        -------
        pd.DataFrame
        """
        rng = rng if rng is not None else np.random.default_rng(self.seed)
        n = self.n_samples

        age = rng.normal(25.0, 7.0, n)
        education = 10.0 + 0.3 * age + rng.normal(0, 2.0, n)
        black = rng.binomial(1, 0.4, n).astype(float)
        hispanic = rng.binomial(1, 0.1, n).astype(float)
        married = (
            0.2 + 0.01 * age + rng.normal(0, 0.3, n) > 0.5
        ).astype(float)
        nodegree = (education < 12).astype(float)

        # Treatment propensity
        logit_t = (
            -2.0 + 0.02 * age + 0.1 * education
            + 0.3 * black + 0.2 * hispanic - 0.5 * nodegree
        )
        prob_t = 1.0 / (1.0 + np.exp(-logit_t))
        treatment = rng.binomial(1, prob_t).astype(float)

        # Outcome
        earnings = (
            5000.0 + 200.0 * age + 500.0 * education
            - 1000.0 * black - 500.0 * hispanic + 800.0 * married
            - 2000.0 * nodegree
            + self.treatment_effect * treatment
            + rng.normal(0, self.noise_scale * 3000.0, n)
        )

        data = np.column_stack([
            age, education, black, hispanic, married,
            nodegree, treatment, earnings,
        ])
        return pd.DataFrame(data, columns=list(self._names))

    def ground_truth(self) -> GroundTruth:
        """Return ground-truth causal quantities."""
        adj = self.dag()
        confounders = _find_confounders(adj, self.TREATMENT, self.EARNINGS)
        return GroundTruth(
            true_ate=self.treatment_effect,
            true_dag=adj,
            true_robustness_radius=2,
            confounders=confounders,
            mediators=frozenset(),
            valid_adjustment_sets=(
                frozenset({
                    self.AGE, self.EDUCATION, self.BLACK,
                    self.HISPANIC, self.NODEGREE,
                }),
            ),
        )


# ============================================================================
# SmokingBirthweightDGP
# ============================================================================


@dataclass(slots=True)
class SmokingBirthweightDGP:
    """Smoking and birthweight DGP (12 variables).

    Parameters
    ----------
    n_samples : int
        Number of observations.
    treatment_effect : float
        True effect of smoking on birthweight (grams).
    noise_scale : float
        Noise scale.
    seed : int
        Random seed.
    """

    n_samples: int = 1000
    treatment_effect: float = -200.0
    noise_scale: float = 1.0
    seed: int = 42

    # Node indices
    MAGE: int = 0       # maternal age
    MEDU: int = 1       # maternal education
    INCOME: int = 2     # household income
    PRENATAL: int = 3   # prenatal care
    ALCOHOL: int = 4    # alcohol use
    SMOKING: int = 5    # smoking (treatment)
    GESTAGE: int = 6    # gestational age
    PARITY: int = 7     # parity
    MHEIGHT: int = 8    # maternal height
    MWEIGHT: int = 9    # maternal weight
    MBMI: int = 10      # maternal BMI
    BIRTHWT: int = 11   # birthweight (outcome)
    N_NODES: int = 12

    _names: tuple[str, ...] = (
        "mage", "medu", "income", "prenatal", "alcohol",
        "smoking", "gestage", "parity", "mheight", "mweight",
        "mbmi", "birthwt",
    )

    def dag(self) -> AdjacencyMatrix:
        """Return the ground-truth DAG."""
        edges = [
            (self.MAGE, self.MEDU), (self.MAGE, self.INCOME),
            (self.MAGE, self.PARITY), (self.MAGE, self.SMOKING),
            (self.MAGE, self.GESTAGE),
            (self.MEDU, self.INCOME), (self.MEDU, self.SMOKING),
            (self.MEDU, self.PRENATAL),
            (self.INCOME, self.PRENATAL), (self.INCOME, self.SMOKING),
            (self.PRENATAL, self.GESTAGE),
            (self.ALCOHOL, self.SMOKING), (self.ALCOHOL, self.BIRTHWT),
            (self.SMOKING, self.GESTAGE), (self.SMOKING, self.BIRTHWT),
            (self.GESTAGE, self.BIRTHWT),
            (self.PARITY, self.BIRTHWT),
            (self.MHEIGHT, self.MWEIGHT), (self.MHEIGHT, self.BIRTHWT),
            (self.MWEIGHT, self.MBMI),
            (self.MBMI, self.BIRTHWT),
        ]
        return _adj_from_edges(self.N_NODES, edges)

    def generate(self, rng: np.random.Generator | None = None) -> pd.DataFrame:
        """Generate observational data."""
        rng = rng if rng is not None else np.random.default_rng(self.seed)
        n = self.n_samples
        ns = self.noise_scale

        mage = rng.normal(28.0, 5.0, n)
        medu = 12.0 + 0.2 * mage + rng.normal(0, 2.0 * ns, n)
        income = 30000 + 1000 * mage + 2000 * medu + rng.normal(0, 5000 * ns, n)
        parity = np.clip(rng.poisson(0.05 * mage, n), 0, 6).astype(float)
        prenatal = 1.0 / (1.0 + np.exp(-(0.01 * income + 0.1 * medu - 3)))
        prenatal = rng.binomial(1, np.clip(prenatal, 0.01, 0.99)).astype(float)
        alcohol = rng.binomial(1, 0.15, n).astype(float)

        logit_s = -1.5 - 0.02 * mage - 0.1 * medu - 0.00001 * income + 0.5 * alcohol
        prob_s = 1.0 / (1.0 + np.exp(-logit_s))
        smoking = rng.binomial(1, np.clip(prob_s, 0.01, 0.99)).astype(float)

        gestage = 39.0 + 0.05 * mage + 0.5 * prenatal - 0.8 * smoking + rng.normal(
            0, 1.5 * ns, n
        )
        mheight = rng.normal(163.0, 7.0, n)
        mweight = 50.0 + 0.5 * mheight + rng.normal(0, 8.0 * ns, n)
        mbmi = mweight / (mheight / 100.0) ** 2

        birthwt = (
            -2000.0 + 100.0 * gestage + 50.0 * parity
            + 5.0 * mheight + 10.0 * mbmi - 50.0 * alcohol
            + self.treatment_effect * smoking
            + rng.normal(0, 300.0 * ns, n)
        )

        data = np.column_stack([
            mage, medu, income, prenatal, alcohol, smoking,
            gestage, parity, mheight, mweight, mbmi, birthwt,
        ])
        return pd.DataFrame(data, columns=list(self._names))

    def ground_truth(self) -> GroundTruth:
        """Return ground-truth causal quantities."""
        adj = self.dag()
        return GroundTruth(
            true_ate=self.treatment_effect,
            true_dag=adj,
            true_robustness_radius=3,
            confounders=frozenset({
                self.MAGE, self.MEDU, self.INCOME, self.ALCOHOL,
            }),
            mediators=frozenset({self.GESTAGE}),
            valid_adjustment_sets=(
                frozenset({
                    self.MAGE, self.MEDU, self.INCOME,
                    self.ALCOHOL, self.PARITY, self.MHEIGHT, self.MBMI,
                }),
            ),
        )


# ============================================================================
# IHDPSimulation
# ============================================================================


@dataclass(slots=True)
class IHDPSimulation:
    """Infant Health and Development Program simulation (25 variables).

    Semi-synthetic DGP inspired by Hill (2011).  Generates 6 continuous
    covariates, 19 binary covariates, 1 binary treatment, and a
    continuous outcome.

    Parameters
    ----------
    n_samples : int
        Number of observations.
    treatment_effect : float
        True average treatment effect.
    noise_scale : float
        Noise scale.
    seed : int
        Random seed.
    """

    n_samples: int = 1000
    treatment_effect: float = 4.0
    noise_scale: float = 1.0
    seed: int = 42
    n_covariates: int = 23
    N_NODES: int = 25  # 23 covariates + treatment + outcome
    TREATMENT: int = 23
    OUTCOME: int = 24

    def dag(self) -> AdjacencyMatrix:
        """Return a plausible causal DAG for IHDP."""
        n = self.N_NODES
        edges: list[tuple[int, int]] = []
        # Covariates 0-5 continuous, 6-22 binary
        # Covariates influence treatment and outcome
        for j in range(self.n_covariates):
            if j < 10:
                edges.append((j, self.TREATMENT))
            edges.append((j, self.OUTCOME))
        edges.append((self.TREATMENT, self.OUTCOME))
        # Some covariate inter-dependencies
        for j in range(1, 6):
            edges.append((0, j))
        for j in range(7, 12):
            edges.append((6, j))
        return _adj_from_edges(n, edges)

    def generate(self, rng: np.random.Generator | None = None) -> pd.DataFrame:
        """Generate semi-synthetic IHDP data."""
        rng = rng if rng is not None else np.random.default_rng(self.seed)
        n = self.n_samples

        # Continuous covariates (0-5)
        X = np.zeros((n, self.n_covariates), dtype=np.float64)
        X[:, 0] = rng.normal(0, 1, n)
        for j in range(1, 6):
            X[:, j] = 0.3 * X[:, 0] + rng.normal(0, 1, n)

        # Binary covariates (6-22)
        X[:, 6] = rng.binomial(1, 0.5, n).astype(float)
        for j in range(7, min(12, self.n_covariates)):
            prob = 1.0 / (1.0 + np.exp(-(-0.5 + 0.5 * X[:, 6])))
            X[:, j] = rng.binomial(1, prob).astype(float)
        for j in range(12, self.n_covariates):
            X[:, j] = rng.binomial(1, 0.3, n).astype(float)

        # Treatment
        logit = -0.5 + 0.3 * X[:, :10].sum(axis=1) / 10.0
        prob_t = 1.0 / (1.0 + np.exp(-logit))
        treatment = rng.binomial(1, np.clip(prob_t, 0.01, 0.99)).astype(float)

        # Outcome — nonlinear response surface
        beta = rng.normal(0, 0.5, self.n_covariates)
        mu = X @ beta + np.sin(X[:, 0]) + X[:, 1] * X[:, 2]
        outcome = mu + self.treatment_effect * treatment + rng.normal(
            0, self.noise_scale, n
        )

        data = np.column_stack([X, treatment, outcome])
        cols = [f"X{i}" for i in range(self.n_covariates)] + ["treatment", "outcome"]
        return pd.DataFrame(data, columns=cols)

    def ground_truth(self) -> GroundTruth:
        """Return ground-truth causal quantities."""
        adj = self.dag()
        confounders = frozenset(range(10))  # first 10 affect treatment
        return GroundTruth(
            true_ate=self.treatment_effect,
            true_dag=adj,
            true_robustness_radius=None,
            confounders=confounders,
            mediators=frozenset(),
            valid_adjustment_sets=(frozenset(range(self.n_covariates)),),
        )


# ============================================================================
# InstrumentDGP
# ============================================================================


@dataclass(slots=True)
class InstrumentDGP:
    """Instrumental variables DGP with compliance types.

    DAG: Z → T → Y, U → T, U → Y  (U unobserved)

    Variables: instrument (Z), confounder (U), compliance_type,
    treatment (T), outcome (Y).

    Parameters
    ----------
    n_samples : int
        Number of observations.
    compliance_rate : float
        Proportion of compliers.
    true_late : float
        True local average treatment effect (LATE).
    noise_scale : float
        Noise scale.
    seed : int
        Random seed.
    """

    n_samples: int = 1000
    compliance_rate: float = 0.6
    true_late: float = 2.0
    noise_scale: float = 1.0
    seed: int = 42

    Z: int = 0
    U: int = 1
    T: int = 2
    Y: int = 3
    N_NODES: int = 4

    _names: tuple[str, ...] = ("instrument", "confounder", "treatment", "outcome")

    def dag(self) -> AdjacencyMatrix:
        """Return the IV DAG (including the unobserved U)."""
        edges = [
            (self.Z, self.T),
            (self.U, self.T),
            (self.U, self.Y),
            (self.T, self.Y),
        ]
        return _adj_from_edges(self.N_NODES, edges)

    def observed_dag(self) -> AdjacencyMatrix:
        """Return DAG over observed variables only (Z, T, Y)."""
        edges = [(0, 1), (1, 2)]  # Z→T→Y
        return _adj_from_edges(3, edges)

    def generate(self, rng: np.random.Generator | None = None) -> pd.DataFrame:
        """Generate IV data with compliance heterogeneity."""
        rng = rng if rng is not None else np.random.default_rng(self.seed)
        n = self.n_samples

        z = rng.binomial(1, 0.5, n).astype(float)
        u = rng.normal(0, 1, n)

        # Compliance type: complier (C), always-taker (AT), never-taker (NT)
        comp_prob = np.full(n, self.compliance_rate)
        at_prob = np.full(n, (1.0 - self.compliance_rate) / 2)
        # Complier: T = Z; AT: T = 1; NT: T = 0
        comp_type = rng.choice(3, n, p=[
            self.compliance_rate,
            (1 - self.compliance_rate) / 2,
            (1 - self.compliance_rate) / 2,
        ])
        treatment = np.where(
            comp_type == 0, z,
            np.where(comp_type == 1, 1.0, 0.0)
        )
        # Add confounder effect to treatment
        treatment = (treatment + 0.3 * u > 0.5).astype(float)

        outcome = (
            1.0 + self.true_late * treatment + 0.5 * u
            + rng.normal(0, self.noise_scale, n)
        )

        data = np.column_stack([z, u, treatment, outcome])
        return pd.DataFrame(data, columns=list(self._names))

    def ground_truth(self) -> GroundTruth:
        """Return ground-truth causal quantities."""
        return GroundTruth(
            true_ate=self.true_late,
            true_dag=self.dag(),
            true_robustness_radius=1,
            confounders=frozenset({self.U}),
            mediators=frozenset(),
            valid_adjustment_sets=(),
        )


# ============================================================================
# MediationDGP
# ============================================================================


@dataclass(slots=True)
class MediationDGP:
    """Mediation DGP with direct and indirect effects.

    DAG: C → T → M → Y, T → Y, C → Y, C → T

    Parameters
    ----------
    n_samples : int
        Number of observations.
    direct_effect : float
        True direct effect of T on Y.
    indirect_effect : float
        True indirect effect T → M → Y.
    noise_scale : float
        Noise scale.
    seed : int
        Random seed.
    """

    n_samples: int = 1000
    direct_effect: float = 1.0
    indirect_effect: float = 0.5
    noise_scale: float = 1.0
    seed: int = 42

    C: int = 0
    T: int = 1
    M: int = 2
    Y: int = 3
    N_NODES: int = 4

    _names: tuple[str, ...] = ("confounder", "treatment", "mediator", "outcome")

    def dag(self) -> AdjacencyMatrix:
        """Return the mediation DAG."""
        edges = [
            (self.C, self.T), (self.C, self.Y),
            (self.T, self.M), (self.T, self.Y),
            (self.M, self.Y),
        ]
        return _adj_from_edges(self.N_NODES, edges)

    def generate(self, rng: np.random.Generator | None = None) -> pd.DataFrame:
        """Generate mediation data."""
        rng = rng if rng is not None else np.random.default_rng(self.seed)
        n = self.n_samples

        c = rng.normal(0, 1, n)
        t = 0.5 * c + rng.normal(0, self.noise_scale, n)
        # Mediator: coefficient chosen so indirect = t_to_m * m_to_y = indirect_effect
        t_to_m = 1.0
        m_to_y = self.indirect_effect / t_to_m
        m = t_to_m * t + rng.normal(0, self.noise_scale, n)
        y = (
            self.direct_effect * t + m_to_y * m + 0.3 * c
            + rng.normal(0, self.noise_scale, n)
        )

        data = np.column_stack([c, t, m, y])
        return pd.DataFrame(data, columns=list(self._names))

    @property
    def total_effect(self) -> float:
        """Total causal effect = direct + indirect."""
        return self.direct_effect + self.indirect_effect

    def ground_truth(self) -> GroundTruth:
        """Return ground-truth causal quantities."""
        adj = self.dag()
        return GroundTruth(
            true_ate=self.total_effect,
            true_dag=adj,
            true_robustness_radius=2,
            confounders=frozenset({self.C}),
            mediators=frozenset({self.M}),
            valid_adjustment_sets=(frozenset({self.C}),),
        )


# ============================================================================
# ConfoundedDGP
# ============================================================================


@dataclass(slots=True)
class ConfoundedDGP:
    """Strong confounding setup.

    Multiple confounders with varying strengths create a challenging
    identification problem.

    DAG: U1 → T, U1 → Y, U2 → T, U2 → Y, U3 → T, U3 → Y,
         U1 → U2, T → Y

    Parameters
    ----------
    n_samples : int
        Number of observations.
    treatment_effect : float
        True ATE.
    n_confounders : int
        Number of confounders.
    confound_strength : float
        Confounding coefficient magnitude.
    noise_scale : float
        Noise scale.
    seed : int
        Random seed.
    """

    n_samples: int = 1000
    treatment_effect: float = 1.0
    n_confounders: int = 3
    confound_strength: float = 2.0
    noise_scale: float = 1.0
    seed: int = 42

    @property
    def N_NODES(self) -> int:
        return self.n_confounders + 2  # confounders + T + Y

    @property
    def TREATMENT(self) -> int:
        return self.n_confounders

    @property
    def OUTCOME(self) -> int:
        return self.n_confounders + 1

    def dag(self) -> AdjacencyMatrix:
        """Return the confounded DAG."""
        n = self.N_NODES
        edges: list[tuple[int, int]] = []
        for j in range(self.n_confounders):
            edges.append((j, self.TREATMENT))
            edges.append((j, self.OUTCOME))
        # Chain among confounders
        for j in range(self.n_confounders - 1):
            edges.append((j, j + 1))
        edges.append((self.TREATMENT, self.OUTCOME))
        return _adj_from_edges(n, edges)

    def generate(self, rng: np.random.Generator | None = None) -> pd.DataFrame:
        """Generate confounded data."""
        rng = rng if rng is not None else np.random.default_rng(self.seed)
        n = self.n_samples
        k = self.n_confounders

        U = np.zeros((n, k), dtype=np.float64)
        U[:, 0] = rng.normal(0, 1, n)
        for j in range(1, k):
            U[:, j] = 0.5 * U[:, j - 1] + rng.normal(0, 1, n)

        # Treatment
        logit = U @ np.full(k, self.confound_strength / k)
        prob_t = 1.0 / (1.0 + np.exp(-logit))
        t = rng.binomial(1, np.clip(prob_t, 0.01, 0.99)).astype(float)

        # Outcome
        y = (
            self.treatment_effect * t
            + U @ np.full(k, self.confound_strength / k)
            + rng.normal(0, self.noise_scale, n)
        )

        data = np.column_stack([U, t, y])
        cols = [f"U{i}" for i in range(k)] + ["treatment", "outcome"]
        return pd.DataFrame(data, columns=cols)

    def ground_truth(self) -> GroundTruth:
        """Return ground-truth causal quantities."""
        adj = self.dag()
        confounders = frozenset(range(self.n_confounders))
        return GroundTruth(
            true_ate=self.treatment_effect,
            true_dag=adj,
            true_robustness_radius=self.n_confounders,
            confounders=confounders,
            mediators=frozenset(),
            valid_adjustment_sets=(confounders,),
        )


# ============================================================================
# FaithfulnessViolationDGP
# ============================================================================


@dataclass(slots=True)
class FaithfulnessViolationDGP:
    """DGP with near-faithfulness violations via path cancellation.

    Two causal paths T → Y with nearly equal but opposite-sign effects,
    making the total effect close to zero despite strong individual paths.

    DAG: T → M1 → Y, T → M2 → Y, C → T, C → Y

    Parameters
    ----------
    n_samples : int
        Number of observations.
    path1_strength : float
        Strength of the T → M1 → Y path.
    cancellation_fraction : float
        How much of path1 is cancelled by path2 (1.0 = exact cancellation).
    noise_scale : float
        Noise scale.
    seed : int
        Random seed.
    """

    n_samples: int = 1000
    path1_strength: float = 2.0
    cancellation_fraction: float = 0.95
    noise_scale: float = 1.0
    seed: int = 42

    C: int = 0
    T: int = 1
    M1: int = 2
    M2: int = 3
    Y: int = 4
    N_NODES: int = 5

    _names: tuple[str, ...] = ("confounder", "treatment", "med1", "med2", "outcome")

    def dag(self) -> AdjacencyMatrix:
        """Return the DAG with two near-cancelling paths."""
        edges = [
            (self.C, self.T), (self.C, self.Y),
            (self.T, self.M1), (self.T, self.M2),
            (self.M1, self.Y), (self.M2, self.Y),
        ]
        return _adj_from_edges(self.N_NODES, edges)

    @property
    def true_ate(self) -> float:
        """Total effect after cancellation."""
        path2_str = -self.path1_strength * self.cancellation_fraction
        return self.path1_strength + path2_str

    def generate(self, rng: np.random.Generator | None = None) -> pd.DataFrame:
        """Generate data with near-faithfulness violation."""
        rng = rng if rng is not None else np.random.default_rng(self.seed)
        n = self.n_samples

        path2_strength = -self.path1_strength * self.cancellation_fraction

        c = rng.normal(0, 1, n)
        t = 0.5 * c + rng.normal(0, self.noise_scale, n)
        m1 = t + rng.normal(0, self.noise_scale, n)
        m2 = t + rng.normal(0, self.noise_scale, n)
        y = (
            self.path1_strength * m1 + path2_strength * m2
            + 0.3 * c + rng.normal(0, self.noise_scale, n)
        )

        data = np.column_stack([c, t, m1, m2, y])
        return pd.DataFrame(data, columns=list(self._names))

    def ground_truth(self) -> GroundTruth:
        """Return ground-truth causal quantities."""
        adj = self.dag()
        return GroundTruth(
            true_ate=self.true_ate,
            true_dag=adj,
            true_robustness_radius=1,
            confounders=frozenset({self.C}),
            mediators=frozenset({self.M1, self.M2}),
            valid_adjustment_sets=(frozenset({self.C}),),
        )


# ============================================================================
# SparseHighDimDGP
# ============================================================================


@dataclass(slots=True)
class SparseHighDimDGP:
    """Sparse high-dimensional DGP (p=50 by default).

    Only a small subset of variables are causally relevant; the rest are
    independent noise variables.

    Parameters
    ----------
    n_samples : int
        Number of observations.
    n_total : int
        Total number of variables.
    n_relevant : int
        Number of causally relevant variables.
    treatment_effect : float
        True ATE.
    noise_scale : float
        Noise scale.
    seed : int
        Random seed.
    """

    n_samples: int = 1000
    n_total: int = 50
    n_relevant: int = 8
    treatment_effect: float = 1.5
    noise_scale: float = 1.0
    seed: int = 42

    @property
    def TREATMENT(self) -> int:
        return self.n_relevant - 2

    @property
    def OUTCOME(self) -> int:
        return self.n_relevant - 1

    def dag(self) -> AdjacencyMatrix:
        """Return the sparse DAG (relevant nodes form a chain + confounders)."""
        n = self.n_total
        r = self.n_relevant
        edges: list[tuple[int, int]] = []
        # Relevant chain: 0 → 1 → … → treatment → outcome
        for j in range(r - 1):
            edges.append((j, j + 1))
        # Add confounders from first few to treatment and outcome
        for j in range(min(3, r - 2)):
            edges.append((j, self.TREATMENT))
            edges.append((j, self.OUTCOME))
        return _adj_from_edges(n, edges)

    def generate(self, rng: np.random.Generator | None = None) -> pd.DataFrame:
        """Generate sparse high-dimensional data."""
        rng = rng if rng is not None else np.random.default_rng(self.seed)
        n = self.n_samples
        r = self.n_relevant

        data = np.zeros((n, self.n_total), dtype=np.float64)
        # Relevant variables via linear chain
        data[:, 0] = rng.normal(0, 1, n)
        for j in range(1, r - 2):
            data[:, j] = 0.5 * data[:, j - 1] + rng.normal(0, self.noise_scale, n)

        # Treatment
        logit = 0.3 * data[:, :min(3, r - 2)].sum(axis=1) + 0.3 * data[:, r - 3]
        prob_t = 1.0 / (1.0 + np.exp(-logit))
        data[:, self.TREATMENT] = rng.binomial(
            1, np.clip(prob_t, 0.01, 0.99)
        ).astype(float)

        # Outcome
        data[:, self.OUTCOME] = (
            self.treatment_effect * data[:, self.TREATMENT]
            + 0.3 * data[:, :min(3, r - 2)].sum(axis=1)
            + rng.normal(0, self.noise_scale, n)
        )

        # Irrelevant noise variables
        for j in range(r, self.n_total):
            data[:, j] = rng.normal(0, 1, n)

        cols = [f"X{i}" for i in range(self.n_total)]
        return pd.DataFrame(data, columns=cols)

    def ground_truth(self) -> GroundTruth:
        """Return ground-truth causal quantities."""
        adj = self.dag()
        confounders = frozenset(range(min(3, self.n_relevant - 2)))
        return GroundTruth(
            true_ate=self.treatment_effect,
            true_dag=adj,
            true_robustness_radius=None,
            confounders=confounders,
            mediators=frozenset(),
            valid_adjustment_sets=(confounders,),
        )


# ============================================================================
# Registry / convenience
# ============================================================================

DGP_REGISTRY: dict[str, type] = {
    "lalonde": LaLondeDGP,
    "smoking_birthweight": SmokingBirthweightDGP,
    "ihdp": IHDPSimulation,
    "instrument": InstrumentDGP,
    "mediation": MediationDGP,
    "confounded": ConfoundedDGP,
    "faithfulness_violation": FaithfulnessViolationDGP,
    "sparse_highdim": SparseHighDimDGP,
}


def list_dgps() -> list[str]:
    """Return the names of all registered DGPs."""
    return sorted(DGP_REGISTRY.keys())


def create_dgp(name: str, **kwargs: Any) -> Any:
    """Instantiate a DGP by name.

    Parameters
    ----------
    name : str
        DGP identifier (see :func:`list_dgps`).
    **kwargs
        Forwarded to the DGP constructor.

    Returns
    -------
    DGP instance

    Raises
    ------
    ValueError
        If *name* is not registered.
    """
    key = name.strip().lower()
    if key not in DGP_REGISTRY:
        valid = ", ".join(sorted(DGP_REGISTRY))
        raise ValueError(f"Unknown DGP {name!r}; choose from {valid}")
    return DGP_REGISTRY[key](**kwargs)
