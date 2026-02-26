"""
SCM (Structural Causal Model) builder.

Constructs structural causal models from network topology, data, and domain
knowledge.  Supports continuous and discrete variables, structural equation
specification, latent variable handling, and conditional probability
distribution (CPD) attachment.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import networkx as nx
import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import minimize

from .dag import DAGRepresentation, EdgeType


# ──────────────────────────────────────────────────────────────────────
# Data types
# ──────────────────────────────────────────────────────────────────────

class VariableType(Enum):
    """Supported variable types in an SCM."""
    CONTINUOUS = auto()
    DISCRETE = auto()
    BINARY = auto()
    ORDINAL = auto()
    LATENT = auto()


@dataclass
class Variable:
    """Metadata for a single variable in the SCM."""
    name: str
    var_type: VariableType = VariableType.CONTINUOUS
    parents: List[str] = field(default_factory=list)
    domain: Optional[Tuple[float, float]] = None
    categories: Optional[List[str]] = None
    observed: bool = True
    description: str = ""


@dataclass
class StructuralEquation:
    """A structural equation  X_i = f_i(pa(X_i), U_i).

    ``func`` is a callable ``(parent_values: dict, noise: float) -> float``.
    ``noise_dist`` is a ``scipy.stats`` frozen distribution (default N(0,1)).
    """
    variable: str
    func: Callable[..., float]
    noise_dist: Any = field(default_factory=lambda: sp_stats.norm(0, 1))
    description: str = ""


@dataclass
class CPD:
    """Conditional probability distribution  P(X | Pa(X)).

    For discrete variables, ``table`` is an ndarray whose axes correspond
    to the parents followed by the child's states.  For continuous
    variables, ``params`` holds regression coefficients and residual
    variance.
    """
    variable: str
    parents: List[str]
    table: Optional[np.ndarray] = None
    params: Optional[Dict[str, Any]] = None


@dataclass
class DomainRule:
    """A domain-knowledge constraint on causal direction.

    ``cause`` → ``effect`` must hold; optionally ``forbidden_edges``
    lists edges that are known to be absent.
    """
    cause: Optional[str] = None
    effect: Optional[str] = None
    forbidden_edges: List[Tuple[str, str]] = field(default_factory=list)
    required_edges: List[Tuple[str, str]] = field(default_factory=list)
    description: str = ""


# ──────────────────────────────────────────────────────────────────────
# SCM container
# ──────────────────────────────────────────────────────────────────────

class SCM:
    """In-memory representation of a fully specified Structural Causal Model.

    Attributes
    ----------
    dag : DAGRepresentation
    variables : dict[str, Variable]
    equations : dict[str, StructuralEquation]
    cpds : dict[str, CPD]
    """

    def __init__(self) -> None:
        self.dag = DAGRepresentation()
        self.variables: Dict[str, Variable] = {}
        self.equations: Dict[str, StructuralEquation] = {}
        self.cpds: Dict[str, CPD] = {}
        self.latent_variables: Set[str] = set()
        self._domain_rules: List[DomainRule] = []

    # ── query helpers ─────────────────────────────────────────────────

    def observed_variables(self) -> List[str]:
        return [v for v, meta in self.variables.items() if meta.observed]

    def latent_variable_names(self) -> List[str]:
        return [v for v, meta in self.variables.items() if not meta.observed]

    def exogenous_variables(self) -> List[str]:
        return [v for v in self.dag.nodes if len(self.dag.parents(v)) == 0]

    def endogenous_variables(self) -> List[str]:
        return [v for v in self.dag.nodes if len(self.dag.parents(v)) > 0]

    # ── sampling ──────────────────────────────────────────────────────

    def sample(self, n: int = 1, interventions: Optional[Dict[str, float]] = None) -> Dict[str, np.ndarray]:
        """Forward-sample *n* observations from the SCM.

        Parameters
        ----------
        interventions : dict mapping variable name → fixed value (do-operator).
        """
        order = self.dag.topological_sort()
        samples: Dict[str, np.ndarray] = {}
        for var in order:
            if interventions and var in interventions:
                samples[var] = np.full(n, interventions[var])
                continue

            eq = self.equations.get(var)
            if eq is None:
                # No equation – sample from noise prior
                samples[var] = sp_stats.norm(0, 1).rvs(n)
                continue

            noise = eq.noise_dist.rvs(n)
            parent_vals = {p: samples[p] for p in self.dag.parents(var)}
            values = np.array([eq.func(
                {p: parent_vals[p][i] for p in parent_vals},
                noise[i],
            ) for i in range(n)])
            samples[var] = values

        return samples

    def interventional_distribution(
        self,
        target: str,
        interventions: Dict[str, float],
        n_samples: int = 10_000,
    ) -> np.ndarray:
        """Monte-Carlo estimate of P(target | do(interventions))."""
        samples = self.sample(n_samples, interventions=interventions)
        return samples[target]

    # ── serialisation helpers ─────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"SCM with {len(self.variables)} variables, {self.dag.n_edges} edges",
            f"  Observed: {self.observed_variables()}",
            f"  Latent:   {self.latent_variable_names()}",
            f"  Equations defined for: {list(self.equations.keys())}",
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# SCMBuilder
# ──────────────────────────────────────────────────────────────────────

class SCMBuilder:
    """Incremental builder for Structural Causal Models.

    Typical workflow::

        builder = SCMBuilder()
        builder.add_variable("X", VariableType.CONTINUOUS)
        builder.add_variable("Y", VariableType.CONTINUOUS, parents=["X"])
        builder.add_equation("Y", lambda pa, u: 0.8 * pa["X"] + u)
        scm = builder.build()
    """

    def __init__(self) -> None:
        self._scm = SCM()
        self._data: Optional[np.ndarray] = None
        self._data_columns: Optional[List[str]] = None
        self._validated = False

    # ── variable management ───────────────────────────────────────────

    def add_variable(
        self,
        name: str,
        var_type: Union[VariableType, str] = VariableType.CONTINUOUS,
        parents: Optional[List[str]] = None,
        domain: Optional[Tuple[float, float]] = None,
        categories: Optional[List[str]] = None,
        observed: bool = True,
        description: str = "",
    ) -> "SCMBuilder":
        """Add a variable to the SCM under construction."""
        if isinstance(var_type, str):
            var_type = VariableType[var_type.upper()]
        parents = parents or []
        var = Variable(
            name=name,
            var_type=var_type,
            parents=list(parents),
            domain=domain,
            categories=categories,
            observed=observed,
            description=description,
        )
        self._scm.variables[name] = var
        self._scm.dag.add_node(name)
        if not observed or var_type == VariableType.LATENT:
            self._scm.latent_variables.add(name)

        for p in parents:
            if p not in self._scm.variables:
                self._scm.variables[p] = Variable(name=p)
                self._scm.dag.add_node(p)
            self._scm.dag.add_edge(p, name)

        self._validated = False
        return self

    def add_variables_from_list(
        self, specs: List[Dict[str, Any]]
    ) -> "SCMBuilder":
        """Batch-add variables from a list of dicts."""
        for spec in specs:
            self.add_variable(**spec)
        return self

    # ── structural equations ──────────────────────────────────────────

    def add_equation(
        self,
        variable: str,
        func: Callable[..., float],
        noise_dist: Any = None,
        description: str = "",
    ) -> "SCMBuilder":
        """Attach a structural equation to *variable*."""
        if noise_dist is None:
            noise_dist = sp_stats.norm(0, 1)
        eq = StructuralEquation(
            variable=variable,
            func=func,
            noise_dist=noise_dist,
            description=description,
        )
        self._scm.equations[variable] = eq
        self._validated = False
        return self

    def add_linear_equation(
        self,
        variable: str,
        coefficients: Dict[str, float],
        intercept: float = 0.0,
        noise_std: float = 1.0,
    ) -> "SCMBuilder":
        """Convenience: add a linear structural equation.

        ``variable = intercept + sum(coeff * parent) + noise``
        """
        def _linear(pa: Dict[str, float], u: float) -> float:
            val = intercept
            for p, coeff in coefficients.items():
                val += coeff * pa.get(p, 0.0)
            val += u
            return val

        return self.add_equation(
            variable,
            _linear,
            noise_dist=sp_stats.norm(0, noise_std),
            description=f"Linear: {intercept} + " + " + ".join(
                f"{c}*{p}" for p, c in coefficients.items()
            ),
        )

    def add_logistic_equation(
        self,
        variable: str,
        coefficients: Dict[str, float],
        intercept: float = 0.0,
        threshold: float = 0.5,
    ) -> "SCMBuilder":
        """Convenience: add a logistic structural equation for binary variables."""
        def _logistic(pa: Dict[str, float], u: float) -> float:
            logit = intercept
            for p, coeff in coefficients.items():
                logit += coeff * pa.get(p, 0.0)
            prob = 1.0 / (1.0 + np.exp(-logit))
            # Use uniform noise for Gumbel-max trick
            return float(prob + u > threshold)

        return self.add_equation(
            variable,
            _logistic,
            noise_dist=sp_stats.uniform(0, 1),
            description=f"Logistic: σ({intercept} + ...)",
        )

    def add_nonlinear_equation(
        self,
        variable: str,
        func: Callable[..., float],
        noise_std: float = 1.0,
    ) -> "SCMBuilder":
        """Add an arbitrary nonlinear structural equation."""
        return self.add_equation(
            variable,
            func,
            noise_dist=sp_stats.norm(0, noise_std),
        )

    # ── CPDs ──────────────────────────────────────────────────────────

    def set_cpd(
        self,
        variable: str,
        cpd: Union[CPD, np.ndarray, Dict[str, Any]],
    ) -> "SCMBuilder":
        """Attach a CPD to a variable.

        ``cpd`` can be a ``CPD`` object, a numpy array (for discrete CPD
        tables), or a dict of regression parameters.
        """
        if isinstance(cpd, np.ndarray):
            parents = self._scm.dag.parents(variable)
            cpd_obj = CPD(variable=variable, parents=parents, table=cpd)
        elif isinstance(cpd, dict):
            parents = self._scm.dag.parents(variable)
            cpd_obj = CPD(variable=variable, parents=parents, params=cpd)
        else:
            cpd_obj = cpd
        self._scm.cpds[variable] = cpd_obj
        self._validated = False
        return self

    # ── latent variable handling ──────────────────────────────────────

    def add_latent(
        self,
        name: str,
        children: List[str],
        noise_dist: Any = None,
    ) -> "SCMBuilder":
        """Add a latent (unobserved) common cause of *children*."""
        self.add_variable(name, VariableType.LATENT, observed=False)
        if noise_dist is None:
            noise_dist = sp_stats.norm(0, 1)
        self.add_equation(name, lambda pa, u: u, noise_dist=noise_dist)
        for c in children:
            if c not in self._scm.variables:
                self.add_variable(c, VariableType.CONTINUOUS)
            self._scm.dag.add_edge(name, c)
        self._validated = False
        return self

    def project_latent(self) -> "SCMBuilder":
        """Project latent variables out, replacing them with bidirected edges.

        For every latent L with children C1, C2, …, add bidirected edges
        between all pairs of children and remove L.
        """
        latents = list(self._scm.latent_variables)
        for L in latents:
            children = self._scm.dag.children(L)
            for i in range(len(children)):
                for j in range(i + 1, len(children)):
                    self._scm.dag.add_edge(
                        children[i], children[j], edge_type=EdgeType.BIDIRECTED
                    )
            self._scm.dag.remove_node(L)
            self._scm.variables.pop(L, None)
            self._scm.equations.pop(L, None)
            self._scm.latent_variables.discard(L)
        self._validated = False
        return self

    # ── domain knowledge ──────────────────────────────────────────────

    def add_domain_rule(self, rule: DomainRule) -> "SCMBuilder":
        self._scm._domain_rules.append(rule)
        return self

    def add_forbidden_edge(self, u: str, v: str, reason: str = "") -> "SCMBuilder":
        self._scm._domain_rules.append(
            DomainRule(forbidden_edges=[(u, v)], description=reason)
        )
        return self

    def add_required_edge(self, u: str, v: str, reason: str = "") -> "SCMBuilder":
        self._scm._domain_rules.append(
            DomainRule(required_edges=[(u, v)], description=reason)
        )
        if u in self._scm.variables and v in self._scm.variables:
            self._scm.dag.add_edge(u, v)
        return self

    # ── data-driven construction ──────────────────────────────────────

    def attach_data(
        self, data: np.ndarray, columns: List[str]
    ) -> "SCMBuilder":
        """Attach an observational dataset for parameter estimation."""
        self._data = data
        self._data_columns = columns
        return self

    def fit_linear_coefficients(self) -> "SCMBuilder":
        """Estimate linear structural equation coefficients via OLS.

        Requires ``attach_data`` to have been called.
        """
        if self._data is None or self._data_columns is None:
            raise RuntimeError("No data attached. Call attach_data() first.")
        col_idx = {c: i for i, c in enumerate(self._data_columns)}

        for var in self._scm.dag.topological_sort():
            parents = self._scm.dag.parents(var)
            if not parents or var not in col_idx:
                continue
            if not all(p in col_idx for p in parents):
                continue

            y = self._data[:, col_idx[var]]
            X = np.column_stack(
                [self._data[:, col_idx[p]] for p in parents]
            )
            # Add intercept column
            X_aug = np.column_stack([np.ones(len(y)), X])
            try:
                beta, residuals, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
            except np.linalg.LinAlgError:
                warnings.warn(f"OLS failed for {var}; skipping.")
                continue

            intercept = beta[0]
            coefficients = {p: float(beta[i + 1]) for i, p in enumerate(parents)}
            y_hat = X_aug @ beta
            noise_std = float(np.std(y - y_hat)) if len(y) > len(beta) else 1.0

            self.add_linear_equation(var, coefficients, intercept, noise_std)

        return self

    def fit_cpds_discrete(self) -> "SCMBuilder":
        """Estimate CPDs for discrete variables via maximum likelihood."""
        if self._data is None or self._data_columns is None:
            raise RuntimeError("No data attached.")
        col_idx = {c: i for i, c in enumerate(self._data_columns)}

        for var_name, var_meta in self._scm.variables.items():
            if var_meta.var_type not in (VariableType.DISCRETE, VariableType.BINARY):
                continue
            if var_name not in col_idx:
                continue

            parents = self._scm.dag.parents(var_name)
            parent_cols = [col_idx[p] for p in parents if p in col_idx]
            child_col = col_idx[var_name]

            y = self._data[:, child_col].astype(int)
            n_states_child = int(y.max()) + 1

            if not parent_cols:
                # Marginal distribution
                counts = np.bincount(y, minlength=n_states_child).astype(float)
                counts += 1e-6  # Laplace smoothing
                cpd_table = counts / counts.sum()
                self.set_cpd(var_name, cpd_table)
                continue

            # Conditional distribution
            parent_data = self._data[:, parent_cols].astype(int)
            parent_card = [int(parent_data[:, i].max()) + 1 for i in range(len(parent_cols))]
            total_parent_configs = int(np.prod(parent_card))

            table = np.ones((total_parent_configs, n_states_child)) * 1e-6

            for row_i in range(len(y)):
                # Compute linear index for parent configuration
                idx = 0
                for pi in range(len(parent_cols)):
                    stride = int(np.prod(parent_card[pi + 1:])) if pi + 1 < len(parent_card) else 1
                    idx += int(parent_data[row_i, pi]) * stride
                if idx < total_parent_configs:
                    table[idx, y[row_i]] += 1

            # Normalise each row
            row_sums = table.sum(axis=1, keepdims=True)
            table = table / np.where(row_sums > 0, row_sums, 1.0)
            self.set_cpd(var_name, CPD(variable=var_name, parents=parents, table=table))

        return self

    def estimate_noise_distributions(self) -> "SCMBuilder":
        """Estimate residual distributions by fitting data to existing equations."""
        if self._data is None or self._data_columns is None:
            raise RuntimeError("No data attached.")
        col_idx = {c: i for i, c in enumerate(self._data_columns)}

        for var in self._scm.equations:
            if var not in col_idx:
                continue
            eq = self._scm.equations[var]
            parents = self._scm.dag.parents(var)
            if not all(p in col_idx for p in parents):
                continue

            y = self._data[:, col_idx[var]]
            residuals = np.empty(len(y))
            for i in range(len(y)):
                pa_vals = {p: self._data[i, col_idx[p]] for p in parents}
                predicted = eq.func(pa_vals, 0.0)
                residuals[i] = y[i] - predicted

            # Fit normal distribution to residuals
            mu, std = sp_stats.norm.fit(residuals)
            eq.noise_dist = sp_stats.norm(mu, std)

        return self

    # ── build from network / data ─────────────────────────────────────

    def build_from_network(
        self,
        adjacency: np.ndarray,
        node_names: List[str],
        node_types: Optional[Dict[str, VariableType]] = None,
    ) -> "SCMBuilder":
        """Populate the SCM from a network adjacency matrix.

        ``adjacency[i][j] = 1`` means edge ``node_names[i]`` → ``node_names[j]``.
        """
        node_types = node_types or {}
        n = len(node_names)
        for i in range(n):
            vtype = node_types.get(node_names[i], VariableType.CONTINUOUS)
            self.add_variable(node_names[i], vtype)
        for i in range(n):
            for j in range(n):
                if adjacency[i, j] != 0:
                    self._scm.dag.add_edge(node_names[i], node_names[j])
        self._validated = False
        return self

    def build(
        self,
        network: Optional[nx.DiGraph] = None,
        data: Optional[np.ndarray] = None,
        data_columns: Optional[List[str]] = None,
        domain_rules: Optional[List[DomainRule]] = None,
    ) -> SCM:
        """Build and return the SCM.

        Parameters
        ----------
        network : networkx DiGraph, optional
            If provided, the DAG structure is extracted from this graph.
        data : array, optional
            Observational data for parameter estimation.
        data_columns : list[str], optional
            Column names for *data*.
        domain_rules : list[DomainRule], optional
            Additional domain constraints.
        """
        if network is not None:
            for node in network.nodes:
                if node not in self._scm.variables:
                    self.add_variable(str(node), VariableType.CONTINUOUS)
            for u, v in network.edges:
                self._scm.dag.add_edge(str(u), str(v))

        if data is not None and data_columns is not None:
            self.attach_data(data, data_columns)

        if domain_rules:
            for rule in domain_rules:
                self.add_domain_rule(rule)

        # Apply domain rules
        self._apply_domain_rules()

        # Auto-fit if data available and equations not yet specified
        if self._data is not None:
            unfitted = [
                v for v in self._scm.dag.nodes
                if v not in self._scm.equations and len(self._scm.dag.parents(v)) > 0
            ]
            if unfitted:
                self.fit_linear_coefficients()

        self.validate()
        return self._scm

    def _apply_domain_rules(self) -> None:
        """Enforce domain knowledge constraints on the DAG."""
        for rule in self._scm._domain_rules:
            for u, v in rule.forbidden_edges:
                if self._scm.dag.has_edge(u, v):
                    self._scm.dag.remove_edge(u, v)
            for u, v in rule.required_edges:
                if not self._scm.dag.has_edge(u, v):
                    try:
                        self._scm.dag.add_edge(u, v)
                    except ValueError:
                        warnings.warn(
                            f"Cannot add required edge {u}->{v}: would create cycle."
                        )

    # ── DAG / graph accessors ─────────────────────────────────────────

    def get_dag(self) -> DAGRepresentation:
        return self._scm.dag

    def get_moral_graph(self) -> nx.Graph:
        return self._scm.dag.moral_graph()

    def get_markov_blanket(self, variable: str) -> Set[str]:
        return self._scm.dag.markov_blanket(variable)

    # ── validation ────────────────────────────────────────────────────

    def validate(self) -> List[str]:
        """Validate the SCM under construction.

        Returns a list of issues (empty if valid).  Also raises warnings
        for non-fatal issues.
        """
        issues = self._scm.dag.validate()

        # Check that every variable with parents has either an equation or CPD
        for var in self._scm.dag.nodes:
            parents = self._scm.dag.parents(var)
            if parents and var not in self._scm.equations and var not in self._scm.cpds:
                issues.append(
                    f"Variable '{var}' has parents {parents} but no equation or CPD."
                )

        # Check equation parents match DAG parents
        for var, eq in self._scm.equations.items():
            dag_parents = set(self._scm.dag.parents(var))
            # Equations are callables – we can't inspect args, but we flag
            # if the variable has no node in the DAG
            if not self._scm.dag.has_node(var):
                issues.append(
                    f"Equation defined for '{var}' which is not in the DAG."
                )

        # Check CPD consistency
        for var, cpd in self._scm.cpds.items():
            dag_parents = self._scm.dag.parents(var)
            if set(cpd.parents) != set(dag_parents):
                issues.append(
                    f"CPD parents for '{var}' ({cpd.parents}) do not match "
                    f"DAG parents ({dag_parents})."
                )
            if cpd.table is not None:
                # Check normalisation
                if cpd.table.ndim == 1:
                    total = cpd.table.sum()
                    if not np.isclose(total, 1.0, atol=0.01):
                        issues.append(
                            f"CPD table for '{var}' does not sum to 1 (sum={total:.4f})."
                        )
                elif cpd.table.ndim == 2:
                    row_sums = cpd.table.sum(axis=1)
                    bad = np.any(~np.isclose(row_sums, 1.0, atol=0.01))
                    if bad:
                        issues.append(
                            f"CPD table rows for '{var}' do not all sum to 1."
                        )

        for issue in issues:
            warnings.warn(issue)

        self._validated = len(issues) == 0
        return issues

    # ── scoring ───────────────────────────────────────────────────────

    def bic_score(self) -> float:
        """Compute the Bayesian Information Criterion for the current model.

        Requires data to be attached.
        """
        if self._data is None:
            raise RuntimeError("No data attached.")
        n = self._data.shape[0]
        col_idx = {c: i for i, c in enumerate(self._data_columns)}

        total_ll = 0.0
        total_params = 0

        for var in self._scm.dag.topological_sort():
            if var not in col_idx:
                continue
            parents = self._scm.dag.parents(var)
            parent_indices = [col_idx[p] for p in parents if p in col_idx]

            y = self._data[:, col_idx[var]]
            if parent_indices:
                X = np.column_stack([self._data[:, pi] for pi in parent_indices])
                X_aug = np.column_stack([np.ones(n), X])
                try:
                    beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                y_hat = X_aug @ beta
                residuals = y - y_hat
                k = len(parents) + 1  # coefficients + intercept
            else:
                residuals = y - y.mean()
                k = 1

            sigma2 = np.var(residuals) + 1e-10
            ll = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
            total_ll += ll
            total_params += k + 1  # +1 for variance

        bic = -2 * total_ll + total_params * np.log(n)
        return float(bic)

    def aic_score(self) -> float:
        """Compute the Akaike Information Criterion."""
        if self._data is None:
            raise RuntimeError("No data attached.")
        n = self._data.shape[0]
        col_idx = {c: i for i, c in enumerate(self._data_columns)}

        total_ll = 0.0
        total_params = 0

        for var in self._scm.dag.topological_sort():
            if var not in col_idx:
                continue
            parents = self._scm.dag.parents(var)
            parent_indices = [col_idx[p] for p in parents if p in col_idx]

            y = self._data[:, col_idx[var]]
            if parent_indices:
                X = np.column_stack([self._data[:, pi] for pi in parent_indices])
                X_aug = np.column_stack([np.ones(n), X])
                try:
                    beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                y_hat = X_aug @ beta
                residuals = y - y_hat
                k = len(parents) + 1
            else:
                residuals = y - y.mean()
                k = 1

            sigma2 = np.var(residuals) + 1e-10
            ll = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)
            total_ll += ll
            total_params += k + 1

        aic = -2 * total_ll + 2 * total_params
        return float(aic)

    # ── financial-domain helpers ──────────────────────────────────────

    def build_systemic_risk_scm(
        self,
        institutions: List[str],
        exposure_matrix: np.ndarray,
        default_threshold: float = 0.0,
    ) -> SCM:
        """Build an SCM for systemic risk from an inter-institution exposure matrix.

        Creates variables for each institution's asset value, equity,
        and default indicator with structural equations reflecting the
        contagion dynamics.
        """
        n = len(institutions)

        for inst in institutions:
            self.add_variable(f"{inst}_assets", VariableType.CONTINUOUS,
                              description=f"Total assets of {inst}")
            self.add_variable(f"{inst}_equity", VariableType.CONTINUOUS,
                              description=f"Equity of {inst}")
            self.add_variable(f"{inst}_default", VariableType.BINARY,
                              description=f"Default indicator for {inst}")

        # Add market shock as a latent common cause
        self.add_latent("market_shock", [f"{inst}_assets" for inst in institutions])

        # Equity = Assets - Liabilities (simplified)
        for i, inst in enumerate(institutions):
            parents_eq = [f"{inst}_assets"]
            # Add cross-exposures
            for j, other in enumerate(institutions):
                if i != j and exposure_matrix[i, j] > 0:
                    parents_eq.append(f"{other}_default")
                    self._scm.dag.add_edge(f"{other}_default", f"{inst}_equity")

            liability_from_defaults = {}
            for j, other in enumerate(institutions):
                if i != j and exposure_matrix[i, j] > 0:
                    liability_from_defaults[f"{other}_default"] = -exposure_matrix[i, j]

            def _make_equity_func(inst_name, liab_map):
                def _equity(pa, u):
                    val = pa.get(f"{inst_name}_assets", 0.0)
                    for p, coeff in liab_map.items():
                        val += coeff * pa.get(p, 0.0)
                    return val + u
                return _equity

            self.add_equation(
                f"{inst}_equity",
                _make_equity_func(inst, liability_from_defaults),
                noise_dist=sp_stats.norm(0, 0.01),
            )
            self._scm.dag.add_edge(f"{inst}_assets", f"{inst}_equity")

            # Default = 1{equity < threshold}
            def _make_default_func(inst_name, thresh):
                def _default(pa, u):
                    return float(pa.get(f"{inst_name}_equity", 0.0) < thresh)
                return _default

            self.add_equation(
                f"{inst}_default",
                _make_default_func(inst, default_threshold),
                noise_dist=sp_stats.uniform(0, 0),
            )
            self._scm.dag.add_edge(f"{inst}_equity", f"{inst}_default")

        # Apply standard financial domain rules
        for inst in institutions:
            self.add_forbidden_edge(f"{inst}_default", f"{inst}_assets",
                                    reason="Default does not cause own asset change")

        return self.build()

    def __repr__(self) -> str:
        return (
            f"SCMBuilder(variables={len(self._scm.variables)}, "
            f"edges={self._scm.dag.n_edges}, "
            f"equations={len(self._scm.equations)})"
        )
