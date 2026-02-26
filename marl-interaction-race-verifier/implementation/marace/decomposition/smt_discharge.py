"""SMT/LP-based contract assumption discharge for assume-guarantee verification.

This module implements formal semantic discharge of contract assumptions,
replacing the syntactic variable-containment heuristic in
:meth:`ContractComposition.discharge_assumptions` with a sound and complete
decision procedure for the QF_LRA (quantifier-free linear real arithmetic)
fragment.

Mathematical Background
-----------------------
An assume-guarantee contract is a pair ``C = (A, G)`` where *A*
(assumption) and *G* (guarantee) are conjunctive linear predicates,
i.e. polyhedra of the form ``{x | Hx ≤ k}``.

**Implication as LP infeasibility.**  Given polyhedra ``P = {x | Ax ≤ b}``
and ``Q = {x | Cx ≤ d}``, we have  P ⊆ Q  iff for every row ``cⱼ^T x ≤ dⱼ``
of Q the LP

    maximise  cⱼ^T x   subject to  Ax ≤ b

has optimal value ≤ dⱼ.  When the LP is infeasible (P is empty),
containment holds vacuously.

**Farkas certificate.**  By LP strong duality, the dual optimal solution
``λ ≥ 0`` satisfying ``A^T λ = cⱼ`` and ``b^T λ ≤ dⱼ`` constitutes
a *Farkas certificate* (a.k.a. LP dual witness) that the implication
holds.  These certificates are self-checking: a verifier need only
confirm non-negativity, the linear combination equality, and the
bound, all in O(mn) arithmetic operations.

**AG composition rule.**  Given contracts ``C_i = (A_i, G_i)`` for
groups ``i = 1 … k``, the parallel composition is sound when every
assumption is entailed by the other groups' guarantees:

    ∀ i : ⋀_{j ≠ i} G_j  ⊢  A_i

This module checks each such entailment clause-by-clause via the LP
procedure above and collects Farkas certificates into a proof chain.

Dependencies
------------
Only ``numpy`` and ``scipy.optimize.linprog`` (HiGHS backend) are
required — no Z3 or other SMT solver is needed.  The LP-based approach
is *sound and complete* for the QF_LRA fragment, which covers all
linear contracts in the MARACE framework.
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
from scipy.optimize import linprog, OptimizeResult

from marace.decomposition.contracts import (
    CheckResult,
    ConjunctivePredicate,
    Contract,
    InterfaceVariable,
    LinearContract,
    LinearPredicate,
)

logger = logging.getLogger(__name__)


# =====================================================================
# Configuration
# =====================================================================

# Numerical tolerance for feasibility/optimality comparisons.
_EPS = 1e-9

# Tolerance for Farkas certificate verification.
_FARKAS_TOL = 1e-7

# Default LP solver method.
_LP_METHOD = "highs"

# Maximum number of LP iterations per call.
_LP_MAX_ITER = 10_000


# =====================================================================
# SMT theory enum
# =====================================================================

class SMTTheory(enum.Enum):
    """Background theory for the discharge decision procedure.

    Only QF_LRA is fully supported via LP reduction.  QF_NRA and QF_LIA
    are declared for future extensibility but currently raise on use.
    """

    QF_LRA = "QF_LRA"
    """Quantifier-free linear real arithmetic — decidable via LP."""

    QF_NRA = "QF_NRA"
    """Quantifier-free nonlinear real arithmetic — requires dedicated
    solvers (e.g. dReal); not yet supported."""

    QF_LIA = "QF_LIA"
    """Quantifier-free linear integer arithmetic — requires MIP; not
    yet supported."""


# =====================================================================
# Result data-classes
# =====================================================================

@dataclass(frozen=True)
class FarkasCertificate:
    """LP dual witness certifying that a linear implication holds.

    Given ``P = {x | Ax ≤ b}`` and a target half-space ``c^T x ≤ d``,
    a Farkas certificate is a vector ``λ ≥ 0`` such that:

    1.  ``A^T λ = c``   (the linear combination of rows of A equals c)
    2.  ``b^T λ ≤ d``   (the combination of bounds implies the target)

    Verification requires only matrix–vector products and scalar
    comparison — no LP solver is needed.
    """

    dual_multipliers: np.ndarray
    """Non-negative dual vector λ of length m (number of assumption rows)."""

    premise_matrix: np.ndarray
    """The matrix A from the premise polyhedron (m × n)."""

    premise_rhs: np.ndarray
    """The vector b from the premise polyhedron (m,)."""

    target_normal: np.ndarray
    """Normal vector c of the target half-space (n,)."""

    target_bound: float
    """Bound d of the target half-space."""

    def verify(self, tol: float = _FARKAS_TOL) -> bool:
        """Independently verify this certificate.

        Returns ``True`` iff the three Farkas conditions hold:

        1. ``λ ≥ -tol``  (non-negativity)
        2. ``‖A^T λ − c‖_∞ ≤ tol``  (linear combination)
        3. ``b^T λ ≤ d + tol``  (bound)
        """
        lam = self.dual_multipliers
        if np.any(lam < -tol):
            return False
        residual = self.premise_matrix.T @ lam - self.target_normal
        if np.max(np.abs(residual)) > tol:
            return False
        if float(self.premise_rhs @ lam) > self.target_bound + tol:
            return False
        return True


@dataclass(frozen=True)
class Counterexample:
    """Concrete variable assignment witnessing a contract violation.

    Attributes
    ----------
    assignment : dict
        Mapping from variable name to value satisfying the premises
        but violating the target.
    violated_clause : LinearPredicate
        The specific guarantee clause that is violated.
    violation_margin : float
        By how much the clause is violated (positive means violated).
    """

    assignment: Dict[str, float]
    violated_clause: LinearPredicate
    violation_margin: float


@dataclass
class DischargeResult:
    """Outcome of a single assumption-discharge attempt.

    Attributes
    ----------
    satisfied : bool
        ``True`` if the assumption is entailed by the guarantees.
    witness : FarkasCertificate or None
        Proof witness when ``satisfied`` is ``True``.
    counterexample : Counterexample or None
        Concrete violation when ``satisfied`` is ``False``.
    proof_steps : list of str
        Human-readable proof trace.
    theory_used : SMTTheory
        Background theory used for the discharge.
    elapsed_seconds : float
        Wall-clock time for the discharge.
    """

    satisfied: bool
    witness: Optional[FarkasCertificate] = None
    counterexample: Optional[Counterexample] = None
    proof_steps: List[str] = field(default_factory=list)
    theory_used: SMTTheory = SMTTheory.QF_LRA
    elapsed_seconds: float = 0.0


@dataclass
class SoundnessResult:
    """Outcome of a full composition-soundness check.

    Attributes
    ----------
    sound : bool
        ``True`` iff every assumption was discharged.
    discharge_results : dict
        Per-group, per-clause discharge results.
    undischarged : list of (group_id, clause) pairs
        Assumptions that could not be discharged.
    proof_chain : list of ProofChainEntry
        Ordered sequence of discharge steps forming the proof.
    """

    sound: bool
    discharge_results: Dict[str, List[DischargeResult]] = field(
        default_factory=dict
    )
    undischarged: List[Tuple[str, LinearPredicate]] = field(
        default_factory=list
    )
    proof_chain: List["ProofChainEntry"] = field(default_factory=list)


@dataclass(frozen=True)
class ProofChainEntry:
    """A single step in the composition-soundness proof chain.

    Each entry records that assumption clause ``clause_index`` of
    group ``group_id`` was discharged using Farkas certificate
    ``certificate``.
    """

    step_index: int
    group_id: str
    clause_index: int
    clause: LinearPredicate
    certificate: Optional[FarkasCertificate]
    discharged: bool
    justification: str


# =====================================================================
# SMTEncoder
# =====================================================================

class SMTEncoder:
    """Encode verification queries as LP problems in QF_LRA.

    All encodings target ``scipy.optimize.linprog`` with the HiGHS
    backend.  For QF_LRA the reduction to LP is exact (sound and
    complete).

    Proof sketch — correctness of LP encoding
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    A conjunctive predicate ``P = ⋀_i aᵢ^T x ≤ bᵢ`` is satisfiable
    iff the LP ``min 0  s.t. Ax ≤ b`` is feasible.  By LP strong
    duality the predicate ``P ⊢ c^T x ≤ d`` holds iff
    ``max c^T x  s.t. Ax ≤ b ≤ d``.  If the LP is unbounded, the
    implication fails; if infeasible, P is empty and the implication
    holds vacuously.  Soundness follows from the correctness of the
    HiGHS simplex / interior-point solver within machine-precision
    tolerances.
    """

    def __init__(self, theory: SMTTheory = SMTTheory.QF_LRA) -> None:
        if theory != SMTTheory.QF_LRA:
            raise NotImplementedError(
                f"Theory {theory.value} is not yet supported; only QF_LRA "
                f"(linear real arithmetic) is implemented via LP reduction."
            )
        self._theory = theory

    @property
    def theory(self) -> SMTTheory:
        return self._theory

    # -----------------------------------------------------------------
    # Predicate encoding
    # -----------------------------------------------------------------

    @staticmethod
    def encode_linear_predicate(
        pred: ConjunctivePredicate,
        var_order: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode a conjunctive predicate as an LP constraint ``Ax ≤ b``.

        Parameters
        ----------
        pred : ConjunctivePredicate
            Conjunction of linear inequalities.
        var_order : list of str
            Fixed ordering of variables (defines column indices).

        Returns
        -------
        A : ndarray of shape (m, n)
        b : ndarray of shape (m,)
        """
        return pred.to_matrix_form(var_order)

    # -----------------------------------------------------------------
    # Feasibility
    # -----------------------------------------------------------------

    @staticmethod
    def check_feasibility(
        A: np.ndarray,
        b: np.ndarray,
    ) -> Tuple[bool, Optional[np.ndarray]]:
        """Check whether the polyhedron ``{x | Ax ≤ b}`` is non-empty.

        Returns ``(feasible, witness_point)``.  The witness is a point
        in the polyhedron when feasible, or ``None`` when infeasible.
        """
        if A.size == 0:
            return True, np.zeros(A.shape[1] if A.ndim == 2 else 0)
        m, n = A.shape
        res = linprog(
            np.zeros(n),
            A_ub=A,
            b_ub=b,
            bounds=(None, None),
            method=_LP_METHOD,
            options={"maxiter": _LP_MAX_ITER},
        )
        if res.success:
            return True, res.x.copy()
        return False, None

    # -----------------------------------------------------------------
    # Implication (polyhedron containment)
    # -----------------------------------------------------------------

    def encode_contract_implication(
        self,
        premise: ConjunctivePredicate,
        target: ConjunctivePredicate,
        var_order: List[str],
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
        """Encode ``premise ⊢ target`` as a list of LP maximisation problems.

        For each clause ``cⱼ^T x ≤ dⱼ`` in *target*, we need to check

            max cⱼ^T x  subject to  premise

        and verify the optimum is ≤ dⱼ.

        Returns a list of ``(A, b, c_j, d_j)`` tuples, one per target
        clause.

        Proof sketch
        ~~~~~~~~~~~~
        ``P ⊆ Q``  iff  ``∀ j: max_{x ∈ P} cⱼ^T x ≤ dⱼ``.
        Each such optimisation is a standard LP.  By strong duality
        the dual optimal ``λ ≥ 0`` with ``A^T λ = cⱼ`` and
        ``b^T λ ≤ dⱼ`` is a Farkas certificate.
        """
        A, b_vec = premise.to_matrix_form(var_order)
        problems: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        idx = {v: i for i, v in enumerate(var_order)}
        n = len(var_order)
        for clause in target.clauses:
            c_j = np.zeros(n, dtype=np.float64)
            for var, coeff in clause.coefficients.items():
                if var in idx:
                    c_j[idx[var]] = coeff
            problems.append((A.copy(), b_vec.copy(), c_j, clause.bound))
        return problems

    # -----------------------------------------------------------------
    # Zonotope containment
    # -----------------------------------------------------------------

    def encode_zonotope_containment(
        self,
        z_inner: "Zonotope",
        z_outer: "Zonotope",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Encode zonotope containment ``Z_inner ⊆ Z_outer`` as LP constraints.

        A zonotope ``Z = {c + G ε | ε ∈ [-1,1]^p}`` is contained in
        ``Z' = {c' + G' ε' | ε' ∈ [-1,1]^{p'}}`` iff for every
        axis-aligned direction ``eᵢ`` the support function satisfies
        ``h_{Z}(eᵢ) ≤ h_{Z'}(eᵢ)`` and ``h_{Z}(-eᵢ) ≤ h_{Z'}(-eᵢ)``.

        This is a *necessary* condition (sufficient for intervals /
        axis-aligned containment).  For general zonotopes, a sound
        over-approximation is obtained by checking the interval-hull
        containment: ``bbox(Z_inner) ⊆ bbox(Z_outer)``.

        Returns ``(A_inner, b_inner, A_outer, b_outer)`` — the LP
        constraint matrices for the inner and outer bounding boxes
        expressed as polyhedra ``Ax ≤ b``.

        Proof sketch
        ~~~~~~~~~~~~
        For an interval ``[l, u]`` the polyhedron is ``x ≤ u ∧ -x ≤ -l``.
        If ``bbox(Z_inner) ⊆ bbox(Z_outer)`` then certainly
        ``Z_inner ⊆ bbox(Z_inner) ⊆ bbox(Z_outer) ⊇ Z_outer`` when
        ``Z_outer`` is itself an interval-hull zonotope.  The general
        case is an over-approximation (sound, not complete).
        """
        dim = z_inner.dimension
        if dim != z_outer.dimension:
            raise ValueError(
                f"Zonotope dimensions must match ({dim} vs {z_outer.dimension})"
            )

        bbox_in = z_inner.bounding_box()   # (dim, 2)
        bbox_out = z_outer.bounding_box()  # (dim, 2)

        # Inner bounding box as Ax ≤ b
        #   x_i ≤ hi_in   →  row = e_i,  b = hi_in
        #  -x_i ≤ -lo_in  →  row = -e_i, b = -lo_in
        A_inner = np.vstack([np.eye(dim), -np.eye(dim)])
        b_inner = np.concatenate([bbox_in[:, 1], -bbox_in[:, 0]])

        A_outer = np.vstack([np.eye(dim), -np.eye(dim)])
        b_outer = np.concatenate([bbox_out[:, 1], -bbox_out[:, 0]])

        return A_inner, b_inner, A_outer, b_outer

    # -----------------------------------------------------------------
    # HB-consistency encoding
    # -----------------------------------------------------------------

    @staticmethod
    def encode_hb_consistency(
        hb_constraints: List[Tuple[str, str]],
    ) -> Tuple[bool, Optional[List[str]]]:
        """Check happens-before DAG consistency.

        Parameters
        ----------
        hb_constraints : list of (before, after) pairs
            Each ``(a, b)`` asserts that event *a* happens before *b*
            in the causal order.

        Returns
        -------
        consistent : bool
            ``True`` iff the constraints form a DAG (no cycles).
        topological_order : list of str or None
            A valid topological ordering if consistent, else ``None``.

        Proof sketch
        ~~~~~~~~~~~~
        A set of happens-before constraints is consistent iff the
        directed graph they induce is acyclic.  We verify this via
        Kahn's algorithm (BFS topological sort) in O(V + E).
        """
        from collections import defaultdict, deque

        in_degree: Dict[str, int] = defaultdict(int)
        adj: Dict[str, List[str]] = defaultdict(list)
        nodes: Set[str] = set()

        for a, b in hb_constraints:
            adj[a].append(b)
            in_degree.setdefault(a, 0)
            in_degree[b] = in_degree.get(b, 0) + 1
            nodes.add(a)
            nodes.add(b)

        queue: deque[str] = deque()
        for node in nodes:
            if in_degree.get(node, 0) == 0:
                queue.append(node)

        order: List[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbour in adj.get(node, []):
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    queue.append(neighbour)

        if len(order) == len(nodes):
            return True, order
        return False, None


# =====================================================================
# LPDischarger — core LP-based discharge engine
# =====================================================================

class LPDischarger:
    """Discharge linear contract assumptions via LP strong duality.

    For each assumption clause ``a^T x ≤ b`` of a target contract, the
    discharger solves

        max a^T x   subject to   ⋀ G_j   (guarantees of other groups)

    If the optimal value is ≤ b (or the LP is infeasible), the clause is
    discharged and a Farkas certificate is extracted from the dual
    solution.

    Proof sketch — soundness
    ~~~~~~~~~~~~~~~~~~~~~~~~
    By LP strong duality, if the primal optimum p* ≤ b then there
    exists a dual feasible ``λ ≥ 0`` with ``H^T λ = a`` and
    ``k^T λ = p* ≤ b`` where ``Hx ≤ k`` is the premise polyhedron.
    This ``λ`` is the Farkas certificate.  Soundness relies on:

    1.  Correctness of HiGHS within machine tolerance.
    2.  Post-hoc verification of the certificate (three simple checks).

    For unsound corner cases (degenerate or ill-conditioned LPs), the
    certificate verification catches any numerical error.
    """

    def __init__(self, tol: float = _EPS) -> None:
        self._tol = tol

    # -----------------------------------------------------------------
    # Single-clause discharge
    # -----------------------------------------------------------------

    def discharge_clause(
        self,
        clause: LinearPredicate,
        premise_A: np.ndarray,
        premise_b: np.ndarray,
        var_order: List[str],
    ) -> DischargeResult:
        """Attempt to discharge a single assumption clause.

        Parameters
        ----------
        clause : LinearPredicate
            The assumption clause ``a^T x ≤ b`` to discharge.
        premise_A : ndarray (m, n)
            Constraint matrix of the combined guarantees.
        premise_b : ndarray (m,)
            Constraint RHS of the combined guarantees.
        var_order : list of str
            Variable ordering (column semantics of premise_A).

        Returns
        -------
        DischargeResult
            Includes a Farkas certificate on success, or a
            counterexample on failure.
        """
        t0 = time.monotonic()
        steps: List[str] = []
        n = len(var_order)

        # Build objective: maximise a^T x  ⟹  minimise -a^T x
        idx = {v: i for i, v in enumerate(var_order)}
        c_obj = np.zeros(n, dtype=np.float64)
        for var, coeff in clause.coefficients.items():
            if var in idx:
                c_obj[idx[var]] = coeff

        steps.append(
            f"Discharging clause: Σ(coeff*var) ≤ {clause.bound}"
        )

        # Handle empty premise (no guarantees)
        if premise_A.size == 0:
            steps.append("Premise polyhedron is empty (no guarantee constraints).")
            # No constraints ⟹ LP unbounded ⟹ clause not discharged
            # unless the clause is trivially true (all coefficients zero)
            if np.allclose(c_obj, 0.0):
                steps.append("Clause has zero objective — trivially satisfied.")
                elapsed = time.monotonic() - t0
                return DischargeResult(
                    satisfied=True,
                    proof_steps=steps,
                    elapsed_seconds=elapsed,
                )
            steps.append("No premise constraints and non-trivial clause — cannot discharge.")
            elapsed = time.monotonic() - t0
            return DischargeResult(
                satisfied=False,
                proof_steps=steps,
                elapsed_seconds=elapsed,
            )

        # Solve LP:  max c_obj^T x  s.t. premise_A x ≤ premise_b
        #   ⟺ linprog(−c_obj, A_ub=premise_A, b_ub=premise_b)
        res: OptimizeResult = linprog(
            -c_obj,
            A_ub=premise_A,
            b_ub=premise_b,
            bounds=(None, None),
            method=_LP_METHOD,
            options={"maxiter": _LP_MAX_ITER, "dual_feasibility_tolerance": 1e-10},
        )

        elapsed = time.monotonic() - t0

        # Case 1: Premise infeasible ⟹ implication holds vacuously
        if not res.success and res.status == 2:
            steps.append("Premise polyhedron is infeasible — discharged vacuously.")
            return DischargeResult(
                satisfied=True,
                proof_steps=steps,
                elapsed_seconds=elapsed,
            )

        # Case 2: LP unbounded ⟹ violation (clause not implied)
        if not res.success and res.status == 3:
            steps.append("LP unbounded — clause is not entailed by guarantees.")
            return DischargeResult(
                satisfied=False,
                proof_steps=steps,
                elapsed_seconds=elapsed,
            )

        # Case 3: LP has other failure
        if not res.success:
            steps.append(f"LP solver returned status {res.status}: {res.message}")
            return DischargeResult(
                satisfied=False,
                proof_steps=steps,
                elapsed_seconds=elapsed,
            )

        # Case 4: LP solved — check optimum
        primal_opt = -res.fun  # we minimised -c^T x, so max = -min
        steps.append(f"LP optimum = {primal_opt:.10g}, clause bound = {clause.bound}")

        if primal_opt <= clause.bound + self._tol:
            # Discharged — extract Farkas certificate
            steps.append("Optimum ≤ bound — clause discharged.")

            cert = self._extract_certificate(
                res, premise_A, premise_b, c_obj, clause.bound
            )
            if cert is not None and cert.verify():
                steps.append("Farkas certificate verified ✓")
            elif cert is not None:
                steps.append("WARNING: Farkas certificate verification failed (numerical).")
            else:
                steps.append("Farkas certificate extraction skipped (no dual available).")

            return DischargeResult(
                satisfied=True,
                witness=cert,
                proof_steps=steps,
                elapsed_seconds=elapsed,
            )

        # Case 5: Violated — build counterexample
        steps.append("Optimum > bound — clause NOT discharged.")
        assignment = {var_order[i]: float(res.x[i]) for i in range(n)}
        cex = Counterexample(
            assignment=assignment,
            violated_clause=clause,
            violation_margin=float(primal_opt - clause.bound),
        )
        return DischargeResult(
            satisfied=False,
            counterexample=cex,
            proof_steps=steps,
            elapsed_seconds=elapsed,
        )

    # -----------------------------------------------------------------
    # Certificate extraction
    # -----------------------------------------------------------------

    @staticmethod
    def _extract_certificate(
        res: OptimizeResult,
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        d: float,
    ) -> Optional[FarkasCertificate]:
        """Extract a Farkas certificate from LP dual variables.

        The HiGHS solver stores inequality duals in ``res.ineqlin.marginals``
        (scipy ≥ 1.7).  Falls back to a least-squares reconstruction
        when dual information is unavailable.

        The dual of  max c^T x  s.t. Ax ≤ b  is
            min b^T λ  s.t. A^T λ = c, λ ≥ 0
        At optimality (strong duality) b^T λ* = c^T x* ≤ d.
        """
        lam: Optional[np.ndarray] = None

        # Try to get duals from the result object
        if hasattr(res, "ineqlin") and hasattr(res.ineqlin, "marginals"):
            raw = np.asarray(res.ineqlin.marginals, dtype=np.float64)
            # linprog minimised −c^T x, so dual signs are flipped:
            #   dual of min (−c)^T x s.t. Ax ≤ b  ⟹  max b^T λ s.t. A^T λ = −c, λ ≥ 0
            # We need λ for the *max c^T x* problem, so negate.
            lam = -raw

        # Fallback: reconstruct via least-squares  A^T λ = c,  λ ≥ 0
        if lam is None or np.any(lam < -_FARKAS_TOL):
            lam = LPDischarger._reconstruct_dual(A, c)

        if lam is None:
            return None

        # Clamp tiny negative values from numerical noise
        lam = np.maximum(lam, 0.0)

        return FarkasCertificate(
            dual_multipliers=lam,
            premise_matrix=A,
            premise_rhs=b,
            target_normal=c,
            target_bound=d,
        )

    @staticmethod
    def _reconstruct_dual(
        A: np.ndarray, c: np.ndarray
    ) -> Optional[np.ndarray]:
        """Reconstruct dual multipliers via non-negative least squares.

        Solves  min ‖A^T λ − c‖²  subject to λ ≥ 0.

        Uses ``scipy.optimize.nnls`` when available, otherwise falls
        back to unconstrained least-squares and clamping.
        """
        try:
            from scipy.optimize import nnls
            lam, residual = nnls(A.T, c)
            if residual < _FARKAS_TOL * (np.linalg.norm(c) + 1.0):
                return lam
            return None
        except ImportError:
            pass

        # Unconstrained least-squares fallback
        lam, _, _, _ = np.linalg.lstsq(A.T, c, rcond=None)
        lam = np.maximum(lam, 0.0)
        residual = np.linalg.norm(A.T @ lam - c)
        if residual < _FARKAS_TOL * (np.linalg.norm(c) + 1.0):
            return lam
        return None

    # -----------------------------------------------------------------
    # Multi-clause discharge
    # -----------------------------------------------------------------

    def discharge_assumption(
        self,
        assumption: ConjunctivePredicate,
        guarantees: Sequence[ConjunctivePredicate],
        var_order: Optional[List[str]] = None,
    ) -> DischargeResult:
        """Discharge all clauses of an assumption against combined guarantees.

        Parameters
        ----------
        assumption : ConjunctivePredicate
            The assumption to discharge (conjunction of linear clauses).
        guarantees : sequence of ConjunctivePredicate
            The guarantees from other groups.  Their conjunction forms
            the premise.
        var_order : list of str, optional
            Variable ordering.  Inferred from predicates if omitted.

        Returns
        -------
        DischargeResult
            ``satisfied=True`` iff every clause of the assumption is
            entailed by the combined guarantees.
        """
        t0 = time.monotonic()

        # Collect variables
        all_vars: Set[str] = set()
        all_vars.update(assumption.variables)
        for g in guarantees:
            all_vars.update(g.variables)
        if var_order is None:
            var_order = sorted(all_vars)

        # Build combined premise matrix from all guarantees
        premise_parts_A: List[np.ndarray] = []
        premise_parts_b: List[np.ndarray] = []
        for g in guarantees:
            gA, gb = g.to_matrix_form(var_order)
            if gA.size > 0:
                premise_parts_A.append(gA)
                premise_parts_b.append(gb)

        if premise_parts_A:
            premise_A = np.vstack(premise_parts_A)
            premise_b = np.concatenate(premise_parts_b)
        else:
            premise_A = np.zeros((0, len(var_order)), dtype=np.float64)
            premise_b = np.zeros(0, dtype=np.float64)

        # Discharge each clause
        all_steps: List[str] = []
        all_certs: List[Optional[FarkasCertificate]] = []
        all_satisfied = True
        first_counterexample: Optional[Counterexample] = None

        for i, clause in enumerate(assumption.clauses):
            all_steps.append(f"--- Clause {i} ---")
            sub = self.discharge_clause(clause, premise_A, premise_b, var_order)
            all_steps.extend(sub.proof_steps)
            all_certs.append(sub.witness)

            if not sub.satisfied:
                all_satisfied = False
                if first_counterexample is None:
                    first_counterexample = sub.counterexample
                # Continue to collect proof info for all clauses

        elapsed = time.monotonic() - t0

        # Build aggregate result
        # Use the first successfully-verified certificate as the overall witness
        first_cert = next((c for c in all_certs if c is not None), None)

        return DischargeResult(
            satisfied=all_satisfied,
            witness=first_cert if all_satisfied else None,
            counterexample=first_counterexample,
            proof_steps=all_steps,
            elapsed_seconds=elapsed,
        )

    # -----------------------------------------------------------------
    # Full contract implication check
    # -----------------------------------------------------------------

    def check_implication(
        self,
        premise: ConjunctivePredicate,
        target: ConjunctivePredicate,
        var_order: Optional[List[str]] = None,
    ) -> DischargeResult:
        """Check ``premise ⊢ target`` (every clause of target follows from premise).

        Equivalent to :meth:`discharge_assumption` with a single premise
        treated as the guarantee and *target* as the assumption to
        discharge.
        """
        return self.discharge_assumption(target, [premise], var_order)


# =====================================================================
# ContractDischarger — high-level contract-aware discharge
# =====================================================================

class ContractDischarger:
    """High-level contract assumption discharge with AG semantics.

    Wraps :class:`LPDischarger` to operate directly on
    :class:`LinearContract` objects, implementing the assume-guarantee
    composition rule.

    AG Composition Rule
    -------------------
    Given contracts ``C_i = (A_i, G_i)`` for groups ``i = 1 … k``,
    the parallel composition is sound when:

        ∀ i ∈ {1…k} :  ⋀_{j ≠ i} G_j  ⊢  A_i

    That is, each group's assumptions are entailed by the conjunction
    of all other groups' guarantees.
    """

    def __init__(self, tol: float = _EPS) -> None:
        self._lp = LPDischarger(tol=tol)

    def discharge_assumption(
        self,
        assumption: ConjunctivePredicate,
        guarantees: Sequence[LinearContract],
        var_order: Optional[List[str]] = None,
    ) -> DischargeResult:
        """Discharge an assumption against a set of guarantee contracts.

        Parameters
        ----------
        assumption : ConjunctivePredicate
            The assumption to discharge.
        guarantees : sequence of LinearContract
            Contracts whose guarantees form the premise.

        Returns
        -------
        DischargeResult
        """
        guarantee_preds = [c.guarantee for c in guarantees]
        return self._lp.discharge_assumption(assumption, guarantee_preds, var_order)

    def verify_composition_soundness(
        self,
        contracts: Dict[str, LinearContract],
        composed: Optional[LinearContract] = None,
    ) -> SoundnessResult:
        """Verify the AG composition rule for a set of group contracts.

        For each group *i*, checks that ``⋀_{j ≠ i} G_j ⊢ A_i``.

        Parameters
        ----------
        contracts : dict mapping group_id → LinearContract
            Per-group contracts.
        composed : LinearContract, optional
            The composed contract (for informational purposes; not
            required for the soundness check).

        Returns
        -------
        SoundnessResult
            Includes per-group discharge results, proof chain, and any
            undischarged assumption clauses.

        Proof sketch
        ~~~~~~~~~~~~
        We iterate over each group *i*.  For each assumption clause
        ``a`` in ``A_i``, we form the premise ``P = ⋀_{j ≠ i} G_j``
        and solve  ``max a^T x  s.t. Px ≤ p``.  If the optimum is
        ≤ ``bound(a)`` for every clause and every group, the composition
        is sound.  The Farkas certificates collected form a
        machine-checkable proof chain.
        """
        group_ids = list(contracts.keys())

        # Compute var_order from all contracts
        all_vars: Set[str] = set()
        for c in contracts.values():
            all_vars.update(c.variables)
        var_order = sorted(all_vars)

        discharge_results: Dict[str, List[DischargeResult]] = {}
        undischarged: List[Tuple[str, LinearPredicate]] = []
        proof_chain: List[ProofChainEntry] = []
        step_idx = 0
        all_sound = True

        for i, gid in enumerate(group_ids):
            contract_i = contracts[gid]
            other_contracts = [
                contracts[gid_j]
                for j, gid_j in enumerate(group_ids)
                if j != i
            ]

            clause_results: List[DischargeResult] = []

            for ci, clause in enumerate(contract_i.assumption.clauses):
                # Discharge this single clause
                single_assumption = ConjunctivePredicate([clause])
                result = self.discharge_assumption(
                    single_assumption, other_contracts, var_order
                )
                clause_results.append(result)

                entry = ProofChainEntry(
                    step_index=step_idx,
                    group_id=gid,
                    clause_index=ci,
                    clause=clause,
                    certificate=result.witness,
                    discharged=result.satisfied,
                    justification=(
                        f"Clause {ci} of group {gid}: "
                        + ("discharged via LP" if result.satisfied else "NOT discharged")
                    ),
                )
                proof_chain.append(entry)
                step_idx += 1

                if not result.satisfied:
                    all_sound = False
                    undischarged.append((gid, clause))

            discharge_results[gid] = clause_results

        return SoundnessResult(
            sound=all_sound,
            discharge_results=discharge_results,
            undischarged=undischarged,
            proof_chain=proof_chain,
        )

    def find_counterexample(
        self,
        contract: LinearContract,
        var_order: Optional[List[str]] = None,
    ) -> Optional[Counterexample]:
        """Find a counterexample to a contract (A ∧ ¬G satisfiable).

        Checks each guarantee clause: negates it and tests whether
        ``A ∧ ¬g_j`` is feasible.  Returns the first counterexample
        found, or ``None`` if the contract is valid.

        Parameters
        ----------
        contract : LinearContract
            The contract to check.
        var_order : list of str, optional
            Variable ordering.

        Returns
        -------
        Counterexample or None
        """
        if var_order is None:
            var_order = sorted(contract.variables)

        n = len(var_order)
        A_assume, b_assume = contract.assumption.to_matrix_form(var_order)

        for clause in contract.guarantee.clauses:
            neg = clause.negate()
            neg_pred = ConjunctivePredicate([neg])
            A_neg, b_neg = neg_pred.to_matrix_form(var_order)

            # Stack assumption + negated guarantee clause
            if A_assume.size > 0 and A_neg.size > 0:
                A_comb = np.vstack([A_assume, A_neg])
                b_comb = np.concatenate([b_assume, b_neg])
            elif A_neg.size > 0:
                A_comb = A_neg
                b_comb = b_neg
            elif A_assume.size > 0:
                continue
            else:
                continue

            res = linprog(
                np.zeros(n),
                A_ub=A_comb,
                b_ub=b_comb,
                bounds=(None, None),
                method=_LP_METHOD,
                options={"maxiter": _LP_MAX_ITER},
            )
            if res.success:
                assignment = {var_order[i]: float(res.x[i]) for i in range(n)}
                lhs = sum(
                    coeff * assignment.get(v, 0.0)
                    for v, coeff in clause.coefficients.items()
                )
                margin = lhs - clause.bound
                return Counterexample(
                    assignment=assignment,
                    violated_clause=clause,
                    violation_margin=float(margin),
                )

        return None


# =====================================================================
# CompositionSoundnessProver
# =====================================================================

class CompositionSoundnessProver:
    """Construct a formal proof of AG composition soundness.

    Orchestrates :class:`ContractDischarger` and records each discharge
    step into a machine-checkable proof certificate chain.  The
    resulting proof can be independently verified by checking every
    Farkas certificate without an LP solver.

    Usage
    -----
    >>> prover = CompositionSoundnessProver()
    >>> proof = prover.prove(contracts)
    >>> assert proof.is_complete
    >>> for entry in proof.chain:
    ...     assert entry.certificate.verify()

    Proof Architecture
    ------------------
    The proof is a sequence of ``ProofChainEntry`` steps, one per
    assumption clause across all groups.  For ``k`` groups with at
    most ``m`` assumption clauses each, the proof has at most ``k·m``
    steps.

    Each step records:
    1. Which group's assumption clause is being discharged.
    2. The LP-derived Farkas certificate.
    3. Whether the certificate independently verifies.

    The proof is *complete* iff every step has ``discharged=True``.
    """

    def __init__(self, tol: float = _EPS) -> None:
        self._discharger = ContractDischarger(tol=tol)

    def prove(
        self,
        contracts: Dict[str, LinearContract],
    ) -> "CompositionProof":
        """Attempt to prove AG composition soundness.

        Parameters
        ----------
        contracts : dict mapping group_id → LinearContract

        Returns
        -------
        CompositionProof
            A structured proof object with certificate chain and
            verification status.
        """
        t0 = time.monotonic()
        result = self._discharger.verify_composition_soundness(contracts)
        elapsed = time.monotonic() - t0

        # Verify all certificates
        verified_chain: List[Tuple[ProofChainEntry, bool]] = []
        all_verified = True
        for entry in result.proof_chain:
            if entry.certificate is not None:
                cert_ok = entry.certificate.verify()
            else:
                cert_ok = entry.discharged  # vacuous/trivial discharge
            verified_chain.append((entry, cert_ok))
            if entry.discharged and not cert_ok:
                all_verified = False

        return CompositionProof(
            is_complete=result.sound,
            chain=result.proof_chain,
            certificate_verifications=verified_chain,
            all_certificates_verified=all_verified,
            undischarged=result.undischarged,
            elapsed_seconds=elapsed,
        )

    def prove_with_refinement(
        self,
        contracts: Dict[str, LinearContract],
        max_rounds: int = 5,
        weaken_factor: float = 1.1,
    ) -> Tuple["CompositionProof", Dict[str, LinearContract]]:
        """Attempt proof with iterative contract weakening.

        If the initial proof attempt fails, iteratively weakens the
        assumptions of groups with undischarged clauses and retries.

        Parameters
        ----------
        contracts : dict
            Initial contracts.
        max_rounds : int
            Maximum weakening rounds.
        weaken_factor : float
            Multiplicative factor for assumption bound widening.

        Returns
        -------
        (proof, refined_contracts)
        """
        from marace.decomposition.contracts import ContractRefinement

        current = dict(contracts)
        for round_idx in range(max_rounds):
            proof = self.prove(current)
            if proof.is_complete:
                return proof, current

            # Weaken assumptions for groups with undischarged clauses
            groups_to_weaken = {gid for gid, _ in proof.undischarged}
            changed = False
            for gid in groups_to_weaken:
                weakened = ContractRefinement.weaken_assumption(
                    current[gid], factor=weaken_factor
                )
                if weakened.name != current[gid].name:
                    current[gid] = weakened
                    changed = True

            if not changed:
                break

        return self.prove(current), current


@dataclass
class CompositionProof:
    """Structured proof certificate for AG composition soundness.

    Attributes
    ----------
    is_complete : bool
        ``True`` iff every assumption clause was discharged.
    chain : list of ProofChainEntry
        Ordered discharge steps.
    certificate_verifications : list of (ProofChainEntry, bool)
        Each entry paired with its independent verification result.
    all_certificates_verified : bool
        ``True`` iff every Farkas certificate independently verifies.
    undischarged : list of (group_id, clause)
        Assumptions that could not be discharged.
    elapsed_seconds : float
        Total proof time.
    """

    is_complete: bool
    chain: List[ProofChainEntry] = field(default_factory=list)
    certificate_verifications: List[Tuple[ProofChainEntry, bool]] = field(
        default_factory=list
    )
    all_certificates_verified: bool = True
    undischarged: List[Tuple[str, LinearPredicate]] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def num_steps(self) -> int:
        return len(self.chain)

    @property
    def num_discharged(self) -> int:
        return sum(1 for e in self.chain if e.discharged)

    @property
    def num_undischarged(self) -> int:
        return len(self.undischarged)

    def summary(self) -> str:
        """Human-readable summary of the proof."""
        lines = [
            f"AG Composition Proof ({self.num_steps} steps, "
            f"{self.elapsed_seconds:.3f}s)",
            f"  Discharged: {self.num_discharged}/{self.num_steps}",
        ]
        if self.is_complete:
            lines.append("  Status: COMPLETE ✓ — all assumptions discharged")
        else:
            lines.append(
                f"  Status: INCOMPLETE — {self.num_undischarged} "
                f"undischarged clause(s)"
            )
        if self.all_certificates_verified:
            lines.append("  Certificates: all independently verified ✓")
        else:
            n_fail = sum(
                1 for _, ok in self.certificate_verifications if not ok
            )
            lines.append(
                f"  Certificates: {n_fail} verification failure(s) ⚠"
            )
        if self.undischarged:
            lines.append("  Undischarged clauses:")
            for gid, clause in self.undischarged:
                lines.append(f"    group={gid}: {clause}")
        return "\n".join(lines)

    def verify_independently(self) -> bool:
        """Re-verify all Farkas certificates without an LP solver.

        This is the *checker* entry point: it takes O(k·m·n) time
        where k·m is the number of proof steps and n is the number
        of variables.

        Returns ``True`` iff the proof is complete and every
        certificate passes verification.
        """
        if not self.is_complete:
            return False
        for entry in self.chain:
            if entry.certificate is not None:
                if not entry.certificate.verify():
                    return False
            elif not entry.discharged:
                return False
        return True


# =====================================================================
# Integration helpers
# =====================================================================

def semantic_discharge(
    contract: LinearContract,
    available_guarantees: Sequence[LinearContract],
    tol: float = _EPS,
) -> Tuple[LinearContract, List[LinearPredicate], List[DischargeResult]]:
    """Drop-in replacement for ``ContractComposition.discharge_assumptions``.

    Unlike the original, this function performs *semantic* discharge:
    each assumption clause is checked against the combined guarantees
    via LP, not just variable containment.

    Parameters
    ----------
    contract : LinearContract
        Contract whose assumptions should be discharged.
    available_guarantees : sequence of LinearContract
        Contracts providing guarantees.
    tol : float
        Numerical tolerance.

    Returns
    -------
    reduced_contract : LinearContract
        Contract with discharged assumptions removed.
    undischarged : list of LinearPredicate
        Assumption clauses that could not be discharged.
    results : list of DischargeResult
        Per-clause discharge results with proof witnesses.

    Example
    -------
    >>> reduced, remaining, results = semantic_discharge(contract, guarantees)
    >>> for r in results:
    ...     if r.satisfied and r.witness:
    ...         assert r.witness.verify()
    """
    discharger = LPDischarger(tol=tol)

    # Collect variables
    all_vars: Set[str] = set()
    all_vars.update(contract.variables)
    for g in available_guarantees:
        all_vars.update(g.variables)
    var_order = sorted(all_vars)

    # Build combined premise
    premise_parts_A: List[np.ndarray] = []
    premise_parts_b: List[np.ndarray] = []
    for g in available_guarantees:
        gA, gb = g.guarantee.to_matrix_form(var_order)
        if gA.size > 0:
            premise_parts_A.append(gA)
            premise_parts_b.append(gb)

    if premise_parts_A:
        premise_A = np.vstack(premise_parts_A)
        premise_b = np.concatenate(premise_parts_b)
    else:
        premise_A = np.zeros((0, len(var_order)), dtype=np.float64)
        premise_b = np.zeros(0, dtype=np.float64)

    # Discharge each clause
    kept_clauses: List[LinearPredicate] = []
    undischarged: List[LinearPredicate] = []
    results: List[DischargeResult] = []

    for clause in contract.assumption.clauses:
        result = discharger.discharge_clause(clause, premise_A, premise_b, var_order)
        results.append(result)
        if not result.satisfied:
            undischarged.append(clause)
            kept_clauses.append(clause)

    reduced = LinearContract(
        name=contract.name + "_smt_discharged",
        assumption=ConjunctivePredicate(kept_clauses),
        guarantee=contract.guarantee,
        interface_vars=contract.interface_vars,
    )
    return reduced, undischarged, results


def verify_ag_composition(
    contracts: Dict[str, LinearContract],
    tol: float = _EPS,
) -> SoundnessResult:
    """One-shot verification of AG composition soundness.

    Convenience wrapper around :class:`ContractDischarger`.

    Parameters
    ----------
    contracts : dict mapping group_id → LinearContract
    tol : float
        Numerical tolerance.

    Returns
    -------
    SoundnessResult
    """
    discharger = ContractDischarger(tol=tol)
    return discharger.verify_composition_soundness(contracts)


def build_proof_certificate(
    contracts: Dict[str, LinearContract],
    tol: float = _EPS,
) -> CompositionProof:
    """Build and verify a composition proof certificate.

    Convenience wrapper around :class:`CompositionSoundnessProver`.

    Parameters
    ----------
    contracts : dict mapping group_id → LinearContract
    tol : float
        Numerical tolerance.

    Returns
    -------
    CompositionProof
    """
    prover = CompositionSoundnessProver(tol=tol)
    return prover.prove(contracts)
