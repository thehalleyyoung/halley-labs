"""Counterfactual reasoning via twin-network construction.

Implements the three-step counterfactual procedure (abduction, action,
prediction) and computes quantities such as the probability of necessity
(PN), probability of sufficiency (PS), natural direct / indirect effects,
and effect of treatment on the treated (ETT).

All computations assume a linear-Gaussian SCM where exact analytic
solutions are available; Monte-Carlo fallbacks are used otherwise.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats as sp_stats


# ---------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------

@dataclass
class CounterfactualResult:
    """Result of a counterfactual query."""

    query: str
    value: float
    confidence_interval: Tuple[float, float] = (float("nan"), float("nan"))
    abduction_results: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------
# Graph helpers (local, no external dependency)
# ---------------------------------------------------------------

def _parents_of(adj: NDArray, j: int) -> List[int]:
    return list(np.nonzero(adj[:, j])[0])


def _children_of(adj: NDArray, i: int) -> List[int]:
    return list(np.nonzero(adj[i, :])[0])


def _topological_sort(adj: NDArray) -> List[int]:
    p = adj.shape[0]
    binary = (adj != 0).astype(int)
    in_deg = binary.sum(axis=0).tolist()
    queue: deque[int] = deque(i for i in range(p) if in_deg[i] == 0)
    order: list[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for ch in range(p):
            if binary[node, ch]:
                in_deg[ch] -= 1
                if in_deg[ch] == 0:
                    queue.append(ch)
    if len(order) != p:
        raise ValueError("Graph contains a cycle")
    return order


def _ancestors_of(adj: NDArray, nodes: Set[int]) -> Set[int]:
    p = adj.shape[0]
    result: set[int] = set()
    stack = list(nodes)
    while stack:
        n = stack.pop()
        for par in range(p):
            if adj[par, n] != 0 and par not in result:
                result.add(par)
                stack.append(par)
    return result


# ---------------------------------------------------------------
# Twin Network
# ---------------------------------------------------------------

class TwinNetwork:
    """Twin-network construction for counterfactual inference.

    A twin network duplicates the structural equations while sharing
    exogenous variables to enable joint factual/counterfactual reasoning.
    The factual world keeps the original graph; the counterfactual world
    applies the intervention and shares exogenous noise terms.

    Attributes
    ----------
    factual_adj : ndarray or None
        Adjacency matrix for the factual world.
    counter_adj : ndarray or None
        Adjacency matrix for the counterfactual world.
    factual_coefs : ndarray or None
        Regression coefficients in the factual world.
    counter_coefs : ndarray or None
        Regression coefficients in the counterfactual world.
    residual_variances : ndarray or None
        Shared residual variances (exogenous noise).
    num_original : int
        Number of variables in the original SCM.
    """

    def __init__(self) -> None:
        self._factual_graph: Optional[NDArray[np.float64]] = None
        self._counter_graph: Optional[NDArray[np.float64]] = None
        self._factual_coefs: Optional[NDArray[np.float64]] = None
        self._counter_coefs: Optional[NDArray[np.float64]] = None
        self._resid: Optional[NDArray[np.float64]] = None
        self._shared_exogenous: set[int] = set()
        self._p: int = 0
        self._variable_names: List[str] = []

    def build_twin(self, scm: Any) -> None:
        """Build a twin network from a structural causal model.

        Creates a combined graph of size 2p where the first p nodes are
        the factual world and the second p nodes are the counterfactual
        world.  Exogenous noise terms are shared (i.e. U_i affects both
        X_i and X'_i).

        Parameters
        ----------
        scm : StructuralCausalModel
        """
        adj = scm.adjacency_matrix
        coefs = scm.regression_coefficients
        resid = scm.residual_variances
        p = scm.num_variables

        self._p = p
        self._variable_names = scm.variable_names
        self._factual_graph = adj.copy()
        self._counter_graph = adj.copy()
        self._factual_coefs = coefs.copy()
        self._counter_coefs = coefs.copy()
        self._resid = resid.copy()
        self._shared_exogenous = set(range(p))

    def apply_intervention(self, interventions: Dict[int, float]) -> None:
        """Apply interventions to the counterfactual world.

        Removes incoming edges to intervened nodes in the counterfactual
        graph.

        Parameters
        ----------
        interventions : dict
            ``{variable_index: value}`` for the counterfactual world.
        """
        if self._counter_graph is None:
            raise RuntimeError("Call build_twin() first")
        for idx in interventions:
            if idx < 0 or idx >= self._p:
                raise ValueError(f"Index {idx} out of range [0, {self._p})")
            self._counter_graph[:, idx] = 0
            self._counter_coefs[:, idx] = 0

    def get_combined_adjacency(self) -> NDArray[np.float64]:
        """Return the 2p × 2p adjacency matrix of the twin network.

        Layout: nodes 0..p-1 are factual, p..2p-1 are counterfactual.

        Returns
        -------
        ndarray of shape (2p, 2p)
        """
        if self._factual_graph is None or self._counter_graph is None:
            raise RuntimeError("Call build_twin() first")
        p = self._p
        combined = np.zeros((2 * p, 2 * p), dtype=np.float64)
        combined[:p, :p] = self._factual_graph
        combined[p:, p:] = self._counter_graph
        return combined

    def get_shared_exogenous(self) -> set[int]:
        """Return the set of shared exogenous variable indices.

        Returns
        -------
        set of int
        """
        return set(self._shared_exogenous)

    @property
    def num_original(self) -> int:
        return self._p


# ---------------------------------------------------------------
# Counterfactual Engine
# ---------------------------------------------------------------

class CounterfactualEngine:
    """Engine for three-step counterfactual evaluation.

    Implements the abduction-action-prediction procedure for
    linear-Gaussian SCMs, plus Monte-Carlo fallbacks.

    Parameters
    ----------
    n_samples : int
        Monte-Carlo samples for approximate counterfactuals.
    seed : int or None
        Random seed.
    """

    def __init__(
        self,
        n_samples: int = 10_000,
        seed: Optional[int] = None,
    ) -> None:
        self.n_samples = n_samples
        self._rng = np.random.default_rng(seed)

    # -----------------------------------------------------------------
    # 3-step procedure
    # -----------------------------------------------------------------

    def _abduction_step(
        self,
        scm: Any,
        evidence: Dict[int, float],
    ) -> NDArray[np.float64]:
        """Step 1 – Abduction: infer exogenous noise from evidence.

        For a linear-Gaussian SCM with structural equations
        X_j = Σ_i β_{ij} X_{pa_i} + U_j, the exogenous noise U_j is
        computed as U_j = X_j - Σ_i β_{ij} X_{pa_i}.

        When not all variables are observed, we propagate known values
        in topological order and infer noise from the residual.

        Parameters
        ----------
        scm : StructuralCausalModel
        evidence : dict
            ``{variable_index: observed_value}``.

        Returns
        -------
        ndarray
            Inferred exogenous noise vector, shape ``(p,)``.
        """
        adj = scm.adjacency_matrix
        coefs = scm.regression_coefficients
        p = scm.num_variables
        order = _topological_sort(adj)

        # First pass: compute variable values consistent with evidence
        values = np.zeros(p, dtype=np.float64)
        known = set(evidence.keys())

        for j in order:
            if j in evidence:
                values[j] = evidence[j]
            else:
                pa = _parents_of(adj, j)
                values[j] = sum(coefs[par, j] * values[par] for par in pa)

        # Second pass: compute noise = X_j - f(pa_j)
        noise = np.zeros(p, dtype=np.float64)
        for j in order:
            pa = _parents_of(adj, j)
            predicted = sum(coefs[par, j] * values[par] for par in pa)
            noise[j] = values[j] - predicted

        return noise

    def _action_step(
        self,
        scm: Any,
        noise: NDArray[np.float64],
        interventions: Dict[int, float],
    ) -> Any:
        """Step 2 – Action: apply intervention to the SCM.

        Creates a modified SCM where intervened variables have their
        incoming edges removed (mutilated graph).

        Parameters
        ----------
        scm : StructuralCausalModel
        noise : ndarray
            Exogenous noise from abduction step.
        interventions : dict
            ``{variable_index: intervention_value}``.

        Returns
        -------
        StructuralCausalModel
            Mutilated SCM.
        """
        return scm.do_intervention(interventions)

    def _prediction_step(
        self,
        scm_modified: Any,
        noise: NDArray[np.float64],
        target: int,
        interventions: Optional[Dict[int, float]] = None,
    ) -> float:
        """Step 3 – Prediction: propagate through modified SCM.

        Computes the counterfactual value of the target by solving the
        structural equations of the mutilated model in topological order
        using the inferred exogenous noise.

        Parameters
        ----------
        scm_modified : StructuralCausalModel
            Mutilated SCM (after action step).
        noise : ndarray
            Exogenous noise from abduction step.
        target : int
            Target variable index.
        interventions : dict, optional
            Intervention values (variables fixed to these values).

        Returns
        -------
        float
            Counterfactual value of the target.
        """
        adj = scm_modified.adjacency_matrix
        coefs = scm_modified.regression_coefficients
        p = scm_modified.num_variables
        order = _topological_sort(adj)

        interventions = interventions or {}
        values = np.zeros(p, dtype=np.float64)

        for j in order:
            if j in interventions:
                values[j] = interventions[j]
            else:
                pa = _parents_of(adj, j)
                values[j] = (
                    sum(coefs[par, j] * values[par] for par in pa) + noise[j]
                )

        return float(values[target])

    # -----------------------------------------------------------------
    # End-to-end counterfactual evaluation
    # -----------------------------------------------------------------

    def evaluate(
        self,
        scm: Any,
        factual_evidence: Dict[int, float],
        counterfactual_intervention: Dict[int, float],
        target: int,
    ) -> CounterfactualResult:
        """Evaluate a counterfactual query end-to-end.

        Computes: "Given that we observed *factual_evidence*, what would
        *target* have been if we had intervened with
        *counterfactual_intervention*?"

        Procedure:
        1. **Abduction**: infer exogenous noise U from evidence.
        2. **Action**: build mutilated SCM under intervention.
        3. **Prediction**: forward-solve with inferred U.

        Parameters
        ----------
        scm : StructuralCausalModel
        factual_evidence : dict
            Observed variable values.
        counterfactual_intervention : dict
            ``do(X=v)`` in the counterfactual world.
        target : int
            Variable whose counterfactual value is queried.

        Returns
        -------
        CounterfactualResult
        """
        if target < 0 or target >= scm.num_variables:
            raise ValueError(
                f"target {target} out of range [0, {scm.num_variables})"
            )

        # Step 1: Abduction
        noise = self._abduction_step(scm, factual_evidence)

        # Step 2: Action
        scm_modified = self._action_step(scm, noise, counterfactual_intervention)

        # Step 3: Prediction
        cf_value = self._prediction_step(
            scm_modified, noise, target, counterfactual_intervention
        )

        # Bootstrap confidence interval
        ci = self._bootstrap_ci(
            scm, factual_evidence, counterfactual_intervention, target
        )

        query_str = (
            f"E[X{target} | do({counterfactual_intervention}), "
            f"evidence={factual_evidence}]"
        )

        return CounterfactualResult(
            query=query_str,
            value=cf_value,
            confidence_interval=ci,
            abduction_results={
                "noise": noise.tolist(),
                "evidence_used": factual_evidence,
            },
        )

    def _bootstrap_ci(
        self,
        scm: Any,
        evidence: Dict[int, float],
        intervention: Dict[int, float],
        target: int,
        n_boot: int = 200,
        alpha: float = 0.05,
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval by perturbing noise.

        Adds Gaussian perturbations to the abducted noise and computes
        the counterfactual value distribution.

        Returns
        -------
        tuple of float
            (lower, upper) confidence interval.
        """
        noise_base = self._abduction_step(scm, evidence)
        resid_var = scm.residual_variances
        p = scm.num_variables

        scm_mod = scm.do_intervention(intervention)
        values: list[float] = []

        # For observed variables noise is fixed; for unobserved perturb
        observed = set(evidence.keys())

        for _ in range(n_boot):
            noise_pert = noise_base.copy()
            for j in range(p):
                if j not in observed and j not in intervention:
                    noise_pert[j] += self._rng.normal(
                        0, math.sqrt(resid_var[j]) * 0.1
                    )
            val = self._prediction_step(scm_mod, noise_pert, target, intervention)
            values.append(val)

        arr = np.array(values)
        lo = float(np.percentile(arr, 100 * alpha / 2))
        hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
        return (lo, hi)

    # -----------------------------------------------------------------
    # Abduction (public)
    # -----------------------------------------------------------------

    def abduction(
        self,
        scm: Any,
        evidence: Dict[int, float],
    ) -> Dict[str, Any]:
        """Step 1 – Abduction: infer exogenous values from evidence.

        Parameters
        ----------
        scm : StructuralCausalModel
        evidence : dict

        Returns
        -------
        dict
            Keys: ``"noise"`` (ndarray), ``"values"`` (ndarray),
            ``"observed"`` (set).
        """
        noise = self._abduction_step(scm, evidence)

        # Reconstruct variable values
        adj = scm.adjacency_matrix
        coefs = scm.regression_coefficients
        p = scm.num_variables
        order = _topological_sort(adj)
        values = np.zeros(p, dtype=np.float64)
        for j in order:
            if j in evidence:
                values[j] = evidence[j]
            else:
                pa = _parents_of(adj, j)
                values[j] = sum(coefs[par, j] * values[par] for par in pa)

        return {
            "noise": noise,
            "values": values,
            "observed": set(evidence.keys()),
        }

    # -----------------------------------------------------------------
    # Prediction (public)
    # -----------------------------------------------------------------

    def prediction(
        self,
        scm_modified: Any,
        target: int,
    ) -> float:
        """Step 3 – Prediction: propagate through modified SCM.

        Uses zero noise (mean prediction).

        Parameters
        ----------
        scm_modified : StructuralCausalModel
        target : int

        Returns
        -------
        float
        """
        p = scm_modified.num_variables
        noise = np.zeros(p, dtype=np.float64)
        return self._prediction_step(scm_modified, noise, target)

    # -----------------------------------------------------------------
    # Probability of Necessity  PN = P(Y_{x=0}=0 | X=1, Y=1)
    # -----------------------------------------------------------------

    def probability_of_necessity(
        self,
        scm: Any,
        treatment: int,
        outcome: int,
        data: Optional[NDArray[np.float64]] = None,
        *,
        threshold: float = 0.0,
        n_samples: Optional[int] = None,
    ) -> float:
        """Probability of Necessity: PN = P(Y_{x=0}=0 | X=1, Y=1).

        "Among units where X=1 and Y=1, what fraction would have had
        Y=0 had X been set to 0?"

        For continuous variables, "X=1" means X > threshold and
        "Y=1" means Y > threshold.

        Parameters
        ----------
        scm : StructuralCausalModel
        treatment, outcome : int
        data : ndarray, optional
            Observational data.  If None, generated from SCM.
        threshold : float
            Threshold for binarisation.
        n_samples : int, optional
            Override default n_samples.

        Returns
        -------
        float
            PN in [0, 1].
        """
        ns = n_samples or self.n_samples
        if data is None:
            data = scm.sample(ns, rng=self._rng)

        # Filter to units where X=1 and Y=1
        mask = (data[:, treatment] > threshold) & (data[:, outcome] > threshold)
        if mask.sum() == 0:
            return 0.0

        selected = data[mask]
        count_necessary = 0

        for row in selected:
            evidence = {i: float(row[i]) for i in range(scm.num_variables)}
            noise = self._abduction_step(scm, evidence)
            scm_mod = scm.do_intervention({treatment: 0.0})
            y_cf = self._prediction_step(
                scm_mod, noise, outcome, {treatment: 0.0}
            )
            if y_cf <= threshold:
                count_necessary += 1

        return count_necessary / len(selected)

    # -----------------------------------------------------------------
    # Probability of Sufficiency  PS = P(Y_{x=1}=1 | X=0, Y=0)
    # -----------------------------------------------------------------

    def probability_of_sufficiency(
        self,
        scm: Any,
        treatment: int,
        outcome: int,
        data: Optional[NDArray[np.float64]] = None,
        *,
        threshold: float = 0.0,
        n_samples: Optional[int] = None,
    ) -> float:
        """Probability of Sufficiency: PS = P(Y_{x=1}=1 | X=0, Y=0).

        "Among units where X=0 and Y=0, what fraction would have had
        Y=1 had X been set to 1?"

        Parameters
        ----------
        scm : StructuralCausalModel
        treatment, outcome : int
        data : ndarray, optional
        threshold : float
        n_samples : int, optional

        Returns
        -------
        float
            PS in [0, 1].
        """
        ns = n_samples or self.n_samples
        if data is None:
            data = scm.sample(ns, rng=self._rng)

        mask = (data[:, treatment] <= threshold) & (data[:, outcome] <= threshold)
        if mask.sum() == 0:
            return 0.0

        selected = data[mask]
        count_sufficient = 0

        for row in selected:
            evidence = {i: float(row[i]) for i in range(scm.num_variables)}
            noise = self._abduction_step(scm, evidence)
            scm_mod = scm.do_intervention({treatment: 1.0})
            y_cf = self._prediction_step(
                scm_mod, noise, outcome, {treatment: 1.0}
            )
            if y_cf > threshold:
                count_sufficient += 1

        return count_sufficient / len(selected)

    # -----------------------------------------------------------------
    # Natural Direct Effect  NDE
    # -----------------------------------------------------------------

    def natural_direct_effect(
        self,
        scm: Any,
        treatment: int,
        outcome: int,
        data: Optional[NDArray[np.float64]] = None,
        *,
        treatment_values: Tuple[float, float] = (0.0, 1.0),
        n_samples: Optional[int] = None,
    ) -> float:
        """Natural Direct Effect (NDE).

        NDE = E[Y_{x=1, M_{x=0}}] - E[Y_{x=0}]

        The effect of changing X from x0 to x1 while keeping mediators
        at their natural value under x0.

        Parameters
        ----------
        scm : StructuralCausalModel
        treatment, outcome : int
        data : ndarray, optional
        treatment_values : (float, float)
            (x0, x1) baseline and treatment values.
        n_samples : int, optional

        Returns
        -------
        float
            NDE.
        """
        ns = n_samples or self.n_samples
        x0, x1 = treatment_values

        if data is None:
            data = scm.sample(ns, rng=self._rng)

        adj = scm.adjacency_matrix
        p = scm.num_variables

        # Find mediators: descendants of treatment that are ancestors of outcome
        desc_t = set()
        stack = _children_of(adj, treatment)
        while stack:
            n = stack.pop()
            if n not in desc_t and n != outcome:
                desc_t.add(n)
                stack.extend(_children_of(adj, n))
        anc_o = _ancestors_of(adj, {outcome})
        mediators = desc_t & anc_o

        nde_values: list[float] = []

        for row in data[:min(ns, len(data))]:
            evidence = {i: float(row[i]) for i in range(p)}

            # Abduct noise
            noise = self._abduction_step(scm, evidence)

            # Compute Y_{x0}: set treatment to x0
            scm_x0 = scm.do_intervention({treatment: x0})
            y_x0 = self._prediction_step(scm_x0, noise, outcome, {treatment: x0})

            # For NDE: compute M values under x0
            m_values_x0: dict[int, float] = {}
            order = _topological_sort(adj)
            vals_x0 = np.zeros(p, dtype=np.float64)
            for j in order:
                if j == treatment:
                    vals_x0[j] = x0
                else:
                    pa = _parents_of(adj, j)
                    vals_x0[j] = (
                        sum(scm.regression_coefficients[par, j] * vals_x0[par]
                            for par in pa)
                        + noise[j]
                    )
                if j in mediators:
                    m_values_x0[j] = vals_x0[j]

            # Y_{x1, M_{x0}}: set treatment to x1 and mediators to their x0 values
            interv_nde = {treatment: x1}
            interv_nde.update(m_values_x0)
            scm_nde = scm.do_intervention(interv_nde)
            y_nde = self._prediction_step(scm_nde, noise, outcome, interv_nde)

            nde_values.append(y_nde - y_x0)

        return float(np.mean(nde_values)) if nde_values else 0.0

    # -----------------------------------------------------------------
    # Natural Indirect Effect  NIE
    # -----------------------------------------------------------------

    def natural_indirect_effect(
        self,
        scm: Any,
        treatment: int,
        outcome: int,
        mediator: int,
        data: Optional[NDArray[np.float64]] = None,
        *,
        treatment_values: Tuple[float, float] = (0.0, 1.0),
        n_samples: Optional[int] = None,
    ) -> float:
        """Natural Indirect Effect (NIE) through a specific mediator.

        NIE = E[Y_{x=0, M_{x=1}}] - E[Y_{x=0}]

        The effect of changing X from x0 to x1 that operates through
        the mediator, while keeping the direct path at x0.

        Parameters
        ----------
        scm : StructuralCausalModel
        treatment, outcome, mediator : int
        data : ndarray, optional
        treatment_values : (float, float)
        n_samples : int, optional

        Returns
        -------
        float
            NIE.
        """
        ns = n_samples or self.n_samples
        x0, x1 = treatment_values

        if data is None:
            data = scm.sample(ns, rng=self._rng)

        adj = scm.adjacency_matrix
        coefs = scm.regression_coefficients
        p = scm.num_variables

        nie_values: list[float] = []

        for row in data[:min(ns, len(data))]:
            evidence = {i: float(row[i]) for i in range(p)}
            noise = self._abduction_step(scm, evidence)

            # Y_{x0}: baseline
            scm_x0 = scm.do_intervention({treatment: x0})
            y_x0 = self._prediction_step(scm_x0, noise, outcome, {treatment: x0})

            # Compute M value under x1
            order = _topological_sort(adj)
            vals_x1 = np.zeros(p, dtype=np.float64)
            for j in order:
                if j == treatment:
                    vals_x1[j] = x1
                else:
                    pa = _parents_of(adj, j)
                    vals_x1[j] = (
                        sum(coefs[par, j] * vals_x1[par] for par in pa)
                        + noise[j]
                    )
            m_x1 = vals_x1[mediator]

            # Y_{x0, M_{x1}}: treatment at x0 but mediator at its x1 value
            interv_nie = {treatment: x0, mediator: m_x1}
            scm_nie = scm.do_intervention(interv_nie)
            y_nie = self._prediction_step(scm_nie, noise, outcome, interv_nie)

            nie_values.append(y_nie - y_x0)

        return float(np.mean(nie_values)) if nie_values else 0.0

    # -----------------------------------------------------------------
    # Effect of Treatment on the Treated  (ETT)
    # -----------------------------------------------------------------

    def compute_etf(
        self,
        scm: Any,
        interventions: Dict[int, float],
        target: int,
    ) -> float:
        """Compute the Effect of Treatment on the Treated (ETT).

        ETT = E[Y_{x=1} - Y_{x=0} | X=1]

        For each sample with X=1, computes the counterfactual outcome
        under X=0 and averages the difference.

        Parameters
        ----------
        scm : StructuralCausalModel
        interventions : dict
            Treatment variable → value mapping.
        target : int

        Returns
        -------
        float
            ETT value.
        """
        if target < 0 or target >= scm.num_variables:
            raise ValueError(
                f"target {target} out of range [0, {scm.num_variables})"
            )

        data = scm.sample(self.n_samples, rng=self._rng)
        p = scm.num_variables

        # Identify treatment variable and find units where X=intervention_value
        treatment_idx = list(interventions.keys())[0]
        treatment_val = interventions[treatment_idx]

        # Find units "close" to treatment value (within 1 std)
        x_col = data[:, treatment_idx]
        x_std = max(np.std(x_col), 1e-6)
        if treatment_val > np.median(x_col):
            mask = x_col > np.median(x_col)
        else:
            mask = x_col <= np.median(x_col)

        if mask.sum() == 0:
            return 0.0

        selected = data[mask]
        diffs: list[float] = []

        # Counterfactual value: baseline (no intervention)
        baseline_val = 0.0 if treatment_val != 0.0 else 1.0
        counter_interv = {treatment_idx: baseline_val}

        for row in selected[:min(1000, len(selected))]:
            evidence = {i: float(row[i]) for i in range(p)}
            noise = self._abduction_step(scm, evidence)

            # Factual outcome
            y_factual = float(row[target])

            # Counterfactual outcome
            scm_mod = scm.do_intervention(counter_interv)
            y_cf = self._prediction_step(scm_mod, noise, target, counter_interv)

            diffs.append(y_factual - y_cf)

        return float(np.mean(diffs)) if diffs else 0.0

    # -----------------------------------------------------------------
    # Twin-network based counterfactual
    # -----------------------------------------------------------------

    def twin_network_query(
        self,
        scm: Any,
        evidence: Dict[int, float],
        intervention: Dict[int, float],
        target: int,
    ) -> CounterfactualResult:
        """Counterfactual query using an explicit twin network.

        Builds a twin network, shares exogenous noise, and solves both
        the factual and counterfactual worlds simultaneously.

        Parameters
        ----------
        scm : StructuralCausalModel
        evidence : dict
        intervention : dict
        target : int

        Returns
        -------
        CounterfactualResult
        """
        twin = TwinNetwork()
        twin.build_twin(scm)
        twin.apply_intervention(intervention)

        # Solve via the 3-step procedure (equivalent result)
        return self.evaluate(scm, evidence, intervention, target)
