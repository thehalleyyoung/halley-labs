"""
SMTEncoder: encode causal-inference constraints as QF_LRA assertions for Z3.

Provides methods to translate LP bounds, conditional independence relations,
normalization constraints, marginalization consistency, and causal-polytope
membership into Z3 expressions suitable for incremental verification.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import z3


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------

def _smt_name(prefix: str, *parts: str) -> str:
    """Build a deterministic, collision-free SMT variable name."""
    sanitised = [p.replace(" ", "_").replace(".", "_") for p in parts]
    return f"{prefix}__{'_'.join(sanitised)}"


def _real(name: str) -> z3.ArithRef:
    return z3.Real(name)


def _bool(name: str) -> z3.BoolRef:
    return z3.Bool(name)


# ---------------------------------------------------------------------------
# Data structures used by the encoder
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DAGSpec:
    """Lightweight specification of a directed acyclic graph."""
    nodes: FrozenSet[str]
    edges: FrozenSet[Tuple[str, str]]

    @staticmethod
    def from_edge_list(edges: List[Tuple[str, str]]) -> "DAGSpec":
        nodes: set[str] = set()
        for u, v in edges:
            nodes.add(u)
            nodes.add(v)
        return DAGSpec(frozenset(nodes), frozenset(edges))

    def parents(self, node: str) -> Set[str]:
        return {u for u, v in self.edges if v == node}

    def children(self, node: str) -> Set[str]:
        return {v for u, v in self.edges if u == node}

    def ancestors(self, node: str) -> Set[str]:
        visited: set[str] = set()
        stack = list(self.parents(node))
        while stack:
            cur = stack.pop()
            if cur not in visited:
                visited.add(cur)
                stack.extend(self.parents(cur))
        return visited

    def descendants(self, node: str) -> Set[str]:
        visited: set[str] = set()
        stack = list(self.children(node))
        while stack:
            cur = stack.pop()
            if cur not in visited:
                visited.add(cur)
                stack.extend(self.children(cur))
        return visited


# ---------------------------------------------------------------------------
# SMTEncoder
# ---------------------------------------------------------------------------

class SMTEncoder:
    """
    Encode causal-inference claims as Z3 QF_LRA formulas.

    Each ``encode_*`` method returns a ``z3.BoolRef`` that can be asserted
    into the solver. The encoder keeps a registry of created SMT variables
    so that the same real-valued variable is reused when the same symbolic
    name appears in different constraints.
    """

    def __init__(self, epsilon: float = 1e-9) -> None:
        self.epsilon = epsilon
        self._var_cache: Dict[str, z3.ExprRef] = {}
        self._assertion_log: List[Tuple[str, z3.BoolRef]] = []

    # ------------------------------------------------------------------
    # Variable management
    # ------------------------------------------------------------------

    def real_var(self, name: str) -> z3.ArithRef:
        """Return (or create) a Z3 Real variable with the given name."""
        if name not in self._var_cache:
            self._var_cache[name] = z3.Real(name)
        return self._var_cache[name]  # type: ignore[return-value]

    def bool_var(self, name: str) -> z3.BoolRef:
        """Return (or create) a Z3 Bool variable with the given name."""
        if name not in self._var_cache:
            self._var_cache[name] = z3.Bool(name)
        return self._var_cache[name]  # type: ignore[return-value]

    def int_var(self, name: str) -> z3.ArithRef:
        """Return (or create) a Z3 Int variable with the given name."""
        if name not in self._var_cache:
            self._var_cache[name] = z3.Int(name)
        return self._var_cache[name]  # type: ignore[return-value]

    def _rv(self, val: float) -> z3.RatNumRef:
        """Shorthand for a Z3 rational literal."""
        return z3.RealVal(str(val))

    def _log(self, tag: str, expr: z3.BoolRef) -> z3.BoolRef:
        self._assertion_log.append((tag, expr))
        return expr

    # ------------------------------------------------------------------
    # 1. Bound claims
    # ------------------------------------------------------------------

    def encode_bound_claim(
        self,
        var_name: str,
        lower: float,
        upper: float,
    ) -> z3.BoolRef:
        """
        Assert ``lower <= var <= upper`` with probability-range guard.

        Returns a conjunction:
          lb == lower ∧ ub == upper ∧ lb <= ub ∧ 0 <= lb ∧ ub <= 1
        """
        lb = self.real_var(f"{var_name}_lb")
        ub = self.real_var(f"{var_name}_ub")
        return self._log(
            "bound",
            z3.And(
                lb == self._rv(lower),
                ub == self._rv(upper),
                lb <= ub,
                lb >= self._rv(0.0),
                ub <= self._rv(1.0),
            ),
        )

    def encode_bound_tightness(
        self,
        var_name: str,
        lower: float,
        upper: float,
        witness_lower: float,
        witness_upper: float,
    ) -> z3.BoolRef:
        """
        Assert that the witness values certify tightness:
        ``witness_lower >= lower`` and ``witness_upper <= upper``.
        """
        wl = self.real_var(f"{var_name}_wl")
        wu = self.real_var(f"{var_name}_wu")
        lb = self.real_var(f"{var_name}_lb")
        ub = self.real_var(f"{var_name}_ub")
        return self._log(
            "tightness",
            z3.And(
                lb == self._rv(lower),
                ub == self._rv(upper),
                wl == self._rv(witness_lower),
                wu == self._rv(witness_upper),
                wl >= lb,
                wu <= ub,
                wl <= wu,
            ),
        )

    # ------------------------------------------------------------------
    # 2. Conditional independence  (d-separation)
    # ------------------------------------------------------------------

    def encode_dsep(
        self,
        x: str,
        y: str,
        z_set: List[str],
        dag: DAGSpec,
    ) -> z3.BoolRef:
        """
        Encode the d-separation predicate dsep(X, Y | Z) in *dag*.

        Uses an edge-indicator / reachability formulation.  For each node
        we create a Boolean ``reach_<n>`` that is true iff *n* is reachable
        from *x* through non-blocked paths. *z_set* members act as
        blocking nodes.  d-separation holds iff ``¬reach_y``.
        """
        prefix = _smt_name("dsep", x, y, *sorted(z_set))
        nodes = sorted(dag.nodes)
        z_nodes = set(z_set)

        reach: Dict[str, z3.BoolRef] = {
            n: self.bool_var(f"{prefix}_reach_{n}") for n in nodes
        }

        clauses: List[z3.BoolRef] = []

        # Source reachable from itself
        clauses.append(reach[x] == z3.BoolVal(True))

        # Blocking: conditioning set unreachable
        for c in z_nodes:
            if c in reach:
                clauses.append(reach[c] == z3.BoolVal(False))

        # Edge propagation (u -> v): if u reachable and edge exists
        for u, v in dag.edges:
            if v not in z_nodes and v != x:
                clauses.append(z3.Implies(reach[u], reach[v]))

        # Reverse edge propagation for non-causal paths (v-structures)
        for u, v in dag.edges:
            if u not in z_nodes and u != x:
                desc_in_z = dag.descendants(v) & z_nodes
                if desc_in_z:
                    clauses.append(z3.Implies(reach[v], reach[u]))

        # d-separation claim: y not reachable
        clauses.append(reach[y] == z3.BoolVal(False))

        return self._log("dsep", z3.And(*clauses))

    # ------------------------------------------------------------------
    # 3. Normalization
    # ------------------------------------------------------------------

    def encode_normalization(
        self,
        distribution: Dict[str, float],
        prefix: str = "norm",
    ) -> z3.BoolRef:
        """
        Assert that *distribution* sums to 1 ± ε and all entries ≥ 0.
        """
        eps = self._rv(self.epsilon)
        smt_vars: List[z3.ArithRef] = []
        clauses: List[z3.BoolRef] = []
        for name, val in distribution.items():
            v = self.real_var(f"{prefix}_{name}")
            clauses.append(v == self._rv(val))
            clauses.append(v >= self._rv(0.0))
            smt_vars.append(v)

        total = z3.Sum(smt_vars) if smt_vars else self._rv(0.0)
        clauses.append(total >= self._rv(1.0) - eps)
        clauses.append(total <= self._rv(1.0) + eps)
        return self._log("normalization", z3.And(*clauses))

    def encode_normalization_from_values(
        self,
        prefix: str,
        values: Dict[str, float],
    ) -> z3.BoolRef:
        """Convenience wrapper around :meth:`encode_normalization`."""
        return self.encode_normalization(values, prefix=prefix)

    def encode_sub_normalization(
        self,
        distribution: Dict[str, float],
        prefix: str = "subnorm",
    ) -> z3.BoolRef:
        """Assert that distribution sums to ≤ 1 (sub-probability)."""
        eps = self._rv(self.epsilon)
        smt_vars: List[z3.ArithRef] = []
        clauses: List[z3.BoolRef] = []
        for name, val in distribution.items():
            v = self.real_var(f"{prefix}_{name}")
            clauses.append(v == self._rv(val))
            clauses.append(v >= self._rv(0.0))
            smt_vars.append(v)

        total = z3.Sum(smt_vars) if smt_vars else self._rv(0.0)
        clauses.append(total <= self._rv(1.0) + eps)
        return self._log("sub_normalization", z3.And(*clauses))

    # ------------------------------------------------------------------
    # 4. Marginalization consistency
    # ------------------------------------------------------------------

    def encode_marginalization(
        self,
        joint: Dict[str, float],
        marginal: Dict[str, float],
        prefix: str = "marg",
    ) -> z3.BoolRef:
        """
        Assert marginalization consistency: for every marginal entry *k*,
        the sum of joint entries whose key contains *k* equals the
        marginal value.
        """
        clauses: List[z3.BoolRef] = []
        eps = self._rv(self.epsilon)

        # Fix joint values
        joint_vars: Dict[str, z3.ArithRef] = {}
        for jk, jv in joint.items():
            v = self.real_var(f"{prefix}_j_{jk}")
            clauses.append(v == self._rv(jv))
            clauses.append(v >= self._rv(0.0))
            joint_vars[jk] = v

        # Fix marginal values
        marginal_vars: Dict[str, z3.ArithRef] = {}
        for mk, mv in marginal.items():
            v = self.real_var(f"{prefix}_m_{mk}")
            clauses.append(v == self._rv(mv))
            clauses.append(v >= self._rv(0.0))
            marginal_vars[mk] = v

        # Marginalization: sum of joint entries containing key == marginal
        for mk, mv_var in marginal_vars.items():
            matching = [
                jvar for jk, jvar in joint_vars.items() if mk in jk
            ]
            if matching:
                s = z3.Sum(matching)
                clauses.append(s >= mv_var - eps)
                clauses.append(s <= mv_var + eps)
            else:
                clauses.append(mv_var <= eps)

        return self._log("marginalization", z3.And(*clauses))

    def encode_marginalization_check(
        self,
        prefix: str,
        joint_values: Dict[str, float],
        marginal_values: Dict[str, float],
        separator_vars: List[str],
    ) -> z3.BoolRef:
        """
        Verify marginalization over *separator_vars*.

        Groups joint entries by their separator-key projection, sums each
        group, and checks equality with the corresponding marginal entry.
        """
        clauses: List[z3.BoolRef] = []
        eps = self._rv(self.epsilon)

        joint_smt: Dict[str, z3.ArithRef] = {}
        for k, v in joint_values.items():
            sv = self.real_var(f"{prefix}_jt_{k}")
            clauses.append(sv == self._rv(v))
            joint_smt[k] = sv

        marginal_smt: Dict[str, z3.ArithRef] = {}
        for k, v in marginal_values.items():
            sv = self.real_var(f"{prefix}_mg_{k}")
            clauses.append(sv == self._rv(v))
            marginal_smt[k] = sv

        # group joint entries by separator key
        groups: Dict[str, List[z3.ArithRef]] = {}
        for jk, jv in joint_smt.items():
            sep_key = "|".join(
                s for s in separator_vars if s in jk
            )
            groups.setdefault(sep_key, []).append(jv)

        for mk, mv in marginal_smt.items():
            sep_key = "|".join(
                s for s in separator_vars if s in mk
            )
            if sep_key in groups:
                total = z3.Sum(groups[sep_key])
                clauses.append(total >= mv - eps)
                clauses.append(total <= mv + eps)

        return self._log("marg_check", z3.And(*clauses) if clauses else z3.BoolVal(True))

    # ------------------------------------------------------------------
    # 5. Causal-polytope membership
    # ------------------------------------------------------------------

    def encode_causal_polytope(
        self,
        point: List[float],
        A: List[List[float]],
        b: List[float],
        prefix: str = "poly",
    ) -> z3.BoolRef:
        """
        Encode membership ``A·x ≤ b`` for a given *point* x.
        """
        n = len(point)
        x_vars = [self.real_var(f"{prefix}_x_{j}") for j in range(n)]
        clauses: List[z3.BoolRef] = []

        for j in range(n):
            clauses.append(x_vars[j] == self._rv(point[j]))
            clauses.append(x_vars[j] >= self._rv(0.0))
            clauses.append(x_vars[j] <= self._rv(1.0))

        for i, (row, rhs) in enumerate(zip(A, b)):
            lhs = z3.Sum(
                [self._rv(row[j]) * x_vars[j] for j in range(min(len(row), n))]
            )
            clauses.append(lhs <= self._rv(rhs) + self._rv(self.epsilon))

        return self._log("polytope", z3.And(*clauses))

    def encode_vertex_enumeration(
        self,
        point: List[float],
        vertices: List[List[float]],
        prefix: str = "vex",
    ) -> z3.BoolRef:
        """
        Encode that *point* is a convex combination of *vertices*.

        Introduces weight variables λ_i ≥ 0 with Σλ_i = 1 and
        Σλ_i·v_i = point.
        """
        n = len(point)
        k = len(vertices)
        lam = [self.real_var(f"{prefix}_lam_{i}") for i in range(k)]
        clauses: List[z3.BoolRef] = []

        for i in range(k):
            clauses.append(lam[i] >= self._rv(0.0))

        # Σλ = 1
        clauses.append(z3.Sum(lam) == self._rv(1.0))

        # Σλ_i * v_ij = point_j for each coordinate j
        for j in range(n):
            weighted = z3.Sum(
                [lam[i] * self._rv(vertices[i][j]) for i in range(k)]
            )
            eps = self._rv(self.epsilon)
            clauses.append(weighted >= self._rv(point[j]) - eps)
            clauses.append(weighted <= self._rv(point[j]) + eps)

        return self._log("vertex_enum", z3.And(*clauses))

    # ------------------------------------------------------------------
    # 6. Message consistency
    # ------------------------------------------------------------------

    def encode_message_consistency(
        self,
        prefix: str,
        sender_potential: Dict[str, float],
        separator_vars: List[str],
        message: Dict[str, float],
    ) -> z3.BoolRef:
        """
        Verify that *message* is the correct marginalisation of
        *sender_potential* onto *separator_vars*.
        """
        clauses: List[z3.BoolRef] = []
        eps = self._rv(self.epsilon)

        pot_smt: Dict[str, z3.ArithRef] = {}
        for k, v in sender_potential.items():
            sv = self.real_var(f"{prefix}_sp_{k}")
            clauses.append(sv == self._rv(v))
            clauses.append(sv >= self._rv(0.0))
            pot_smt[k] = sv

        msg_smt: Dict[str, z3.ArithRef] = {}
        for k, v in message.items():
            sv = self.real_var(f"{prefix}_msg_{k}")
            clauses.append(sv == self._rv(v))
            msg_smt[k] = sv

        # For each message entry, sum matching potential entries
        for mk, mv in msg_smt.items():
            matching = [
                pv for pk, pv in pot_smt.items()
                if mk in pk or pk in separator_vars
            ]
            if matching:
                total = z3.Sum(matching)
                clauses.append(total >= mv - eps)
                clauses.append(total <= mv + eps)

        return self._log(
            "msg_consistency",
            z3.And(*clauses) if clauses else z3.BoolVal(True),
        )

    # ------------------------------------------------------------------
    # 7. Intervention encoding
    # ------------------------------------------------------------------

    def encode_intervention_consistency(
        self,
        target_var: str,
        intervened_value: float,
        pre_dist: Dict[str, float],
        post_dist: Dict[str, float],
        dag: DAGSpec,
        prefix: str = "intv",
    ) -> z3.BoolRef:
        """
        Verify that the post-intervention distribution is consistent
        with the truncated factorisation (do-calculus rule 2).

        For non-descendants of the target, the conditional should be
        unchanged.
        """
        clauses: List[z3.BoolRef] = []
        eps = self._rv(self.epsilon)

        target_descendants = dag.descendants(target_var)

        for k in pre_dist:
            if k not in target_descendants and k != target_var:
                pre_v = self.real_var(f"{prefix}_pre_{k}")
                post_v = self.real_var(f"{prefix}_post_{k}")
                clauses.append(pre_v == self._rv(pre_dist[k]))
                clauses.append(post_v == self._rv(post_dist.get(k, 0.0)))
                clauses.append(z3.And(
                    post_v >= pre_v - eps,
                    post_v <= pre_v + eps,
                ))

        # Intervened variable is fixed
        tv = self.real_var(f"{prefix}_do_{target_var}")
        clauses.append(tv == self._rv(intervened_value))

        return self._log(
            "intervention",
            z3.And(*clauses) if clauses else z3.BoolVal(True),
        )

    # ------------------------------------------------------------------
    # 8. Conditional probability encoding
    # ------------------------------------------------------------------

    def encode_conditional(
        self,
        joint_key: str,
        marginal_key: str,
        joint_val: float,
        marginal_val: float,
        cond_val: float,
        prefix: str = "cond",
    ) -> z3.BoolRef:
        """
        Verify P(A|B) = P(A,B) / P(B) within tolerance.
        """
        jv = self.real_var(f"{prefix}_joint_{joint_key}")
        mv = self.real_var(f"{prefix}_marg_{marginal_key}")
        cv = self.real_var(f"{prefix}_cond_{joint_key}_{marginal_key}")
        eps = self._rv(self.epsilon)

        return self._log(
            "conditional",
            z3.And(
                jv == self._rv(joint_val),
                mv == self._rv(marginal_val),
                cv == self._rv(cond_val),
                mv > self._rv(0.0),
                # P(A|B)*P(B) ≈ P(A,B)
                cv * mv >= jv - eps,
                cv * mv <= jv + eps,
            ),
        )

    # ------------------------------------------------------------------
    # 9. Chain rule encoding
    # ------------------------------------------------------------------

    def encode_chain_rule(
        self,
        factors: List[Tuple[str, float]],
        joint_name: str,
        joint_val: float,
        prefix: str = "chain",
    ) -> z3.BoolRef:
        """
        Verify that joint = Π factor_i within tolerance.
        """
        clauses: List[z3.BoolRef] = []
        eps = self._rv(self.epsilon)

        factor_vars: List[z3.ArithRef] = []
        for fname, fval in factors:
            v = self.real_var(f"{prefix}_f_{fname}")
            clauses.append(v == self._rv(fval))
            clauses.append(v >= self._rv(0.0))
            clauses.append(v <= self._rv(1.0))
            factor_vars.append(v)

        jv = self.real_var(f"{prefix}_joint_{joint_name}")
        clauses.append(jv == self._rv(joint_val))

        # Product constraint: encode iteratively
        if factor_vars:
            product = factor_vars[0]
            for fv in factor_vars[1:]:
                product = product * fv
            clauses.append(jv >= product - eps)
            clauses.append(jv <= product + eps)

        return self._log("chain_rule", z3.And(*clauses))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_assertion_log(self) -> List[Tuple[str, str]]:
        """Return stringified log of all assertions produced."""
        return [(tag, str(expr)) for tag, expr in self._assertion_log]

    def get_variable_count(self) -> int:
        return len(self._var_cache)

    def clear_cache(self) -> None:
        """Reset the variable cache (start fresh namespace)."""
        self._var_cache.clear()
        self._assertion_log.clear()
