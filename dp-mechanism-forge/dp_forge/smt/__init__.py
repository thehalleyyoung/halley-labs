"""
SMT-based verification for differential privacy mechanisms.

This package implements Satisfiability Modulo Theories (SMT) based
verification of DP guarantees. It encodes privacy constraints as
formulas in the theory of linear real arithmetic (LRA) and uses
DPLL(T)-style solving to check satisfiability.

Key capabilities:
- **Encoding**: Translate DP constraints (privacy ratios, probability
  simplex, adjacency) into SMT formulas over LRA.
- **Verification**: Check whether a mechanism satisfies (ε, δ)-DP by
  checking unsatisfiability of the negation.
- **Counterexample generation**: Extract concrete counterexamples from
  satisfying assignments.
- **Theory combination**: Combine LRA with other theories (e.g.,
  nonlinear arithmetic for Rényi divergence).

Architecture:
    1. **SMTEncoder** — Encodes DP mechanism constraints as SMT formulas.
    2. **DPLLTSolver** — DPLL(T) solver for checking satisfiability.
    3. **TheorySolver** — Theory solver for linear real arithmetic.
    4. **SMTVerifier** — High-level DP verification via SMT.
    5. **CounterexampleExtractor** — Extracts counterexamples from models.

Example::

    from dp_forge.smt import SMTVerifier, SMTConfig

    verifier = SMTVerifier(config=SMTConfig(theory="LRA"))
    result = verifier.verify(mechanism, epsilon=1.0, delta=0.0)
    if result.verified:
        print("Mechanism is ε-DP (SMT-verified)")
    else:
        print(f"Violation: {result.counterexample}")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt

from dp_forge.types import (
    AdjacencyRelation,
    Formula,
    Predicate,
    PrivacyBudget,
    VerifyResult,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Theory(Enum):
    """SMT background theories."""

    LRA = auto()       # Linear Real Arithmetic
    LIA = auto()       # Linear Integer Arithmetic
    NRA = auto()       # Nonlinear Real Arithmetic
    QF_LRA = auto()    # Quantifier-Free LRA
    QF_NRA = auto()    # Quantifier-Free NRA
    AUFLIRA = auto()   # Arrays, Uninterpreted Functions, LIA+LRA

    def __repr__(self) -> str:
        return f"Theory.{self.name}"


class SolverResult(Enum):
    """Result of an SMT satisfiability check."""

    SAT = auto()
    UNSAT = auto()
    UNKNOWN = auto()
    TIMEOUT = auto()

    def __repr__(self) -> str:
        return f"SolverResult.{self.name}"


class PropagationRule(Enum):
    """Theory propagation rules used in DPLL(T)."""

    UNIT_PROPAGATION = auto()
    THEORY_PROPAGATION = auto()
    BOOLEAN_CONSTRAINT_PROPAGATION = auto()
    CONFLICT_DRIVEN_LEARNING = auto()

    def __repr__(self) -> str:
        return f"PropagationRule.{self.name}"


class EncodeStrategy(Enum):
    """Strategy for encoding DP constraints as SMT formulas."""

    DIRECT = auto()
    LOG_SPACE = auto()
    RATIO = auto()
    DIFFERENCE = auto()

    def __repr__(self) -> str:
        return f"EncodeStrategy.{self.name}"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SMTConfig:
    """Configuration for SMT-based verification.

    Attributes:
        theory: Background theory for the SMT solver.
        encode_strategy: Strategy for encoding DP constraints.
        timeout_seconds: Maximum solver time.
        produce_proofs: Whether to generate unsatisfiability proofs.
        produce_models: Whether to generate satisfying assignments.
        random_seed: Random seed for solver heuristics.
        verbose: Verbosity level.
    """

    theory: Theory = Theory.QF_LRA
    encode_strategy: EncodeStrategy = EncodeStrategy.RATIO
    timeout_seconds: float = 120.0
    produce_proofs: bool = True
    produce_models: bool = True
    random_seed: int = 42
    verbose: int = 1

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {self.timeout_seconds}")

    def __repr__(self) -> str:
        return (
            f"SMTConfig(theory={self.theory.name}, "
            f"encode={self.encode_strategy.name}, "
            f"timeout={self.timeout_seconds}s)"
        )


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


@dataclass
class SMTVariable:
    """An SMT variable declaration.

    Attributes:
        name: Variable name.
        sort: SMT sort (e.g., 'Real', 'Int', 'Bool').
        lower_bound: Optional lower bound.
        upper_bound: Optional upper bound.
    """

    name: str
    sort: str = "Real"
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Variable name must be non-empty")
        if self.sort not in ("Real", "Int", "Bool"):
            raise ValueError(f"Unsupported sort: {self.sort}")

    def __repr__(self) -> str:
        bounds = ""
        if self.lower_bound is not None or self.upper_bound is not None:
            lb = self.lower_bound if self.lower_bound is not None else "-∞"
            ub = self.upper_bound if self.upper_bound is not None else "∞"
            bounds = f" ∈ [{lb}, {ub}]"
        return f"SMTVariable({self.name}: {self.sort}{bounds})"


@dataclass
class SMTConstraint:
    """An SMT constraint (assertion).

    Attributes:
        formula: The constraint formula.
        label: Optional label for tracking in proofs.
        is_soft: Whether this is a soft constraint (for MaxSMT).
        weight: Weight for soft constraints.
    """

    formula: Formula
    label: Optional[str] = None
    is_soft: bool = False
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.is_soft and self.weight <= 0:
            raise ValueError(f"Soft constraint weight must be > 0, got {self.weight}")

    def __repr__(self) -> str:
        soft = ", soft" if self.is_soft else ""
        label = f", label={self.label!r}" if self.label else ""
        return f"SMTConstraint({self.formula.expr!r}{label}{soft})"


@dataclass
class SMTModel:
    """A satisfying assignment (model) from the SMT solver.

    Attributes:
        assignments: Mapping from variable names to values.
        is_complete: Whether all variables are assigned.
    """

    assignments: Dict[str, float]
    is_complete: bool = True

    def get(self, name: str) -> Optional[float]:
        """Get the value assigned to a variable."""
        return self.assignments.get(name)

    def __repr__(self) -> str:
        n = len(self.assignments)
        complete = "complete" if self.is_complete else "partial"
        return f"SMTModel(vars={n}, {complete})"


@dataclass
class SMTProof:
    """An unsatisfiability proof from the SMT solver.

    Attributes:
        proof_steps: Sequence of proof step descriptions.
        core_constraints: Unsatisfiable core (subset of input constraints).
        interpolant: Optional Craig interpolant derived from the proof.
    """

    proof_steps: List[str] = field(default_factory=list)
    core_constraints: List[str] = field(default_factory=list)
    interpolant: Optional[Formula] = None

    @property
    def size(self) -> int:
        """Number of proof steps."""
        return len(self.proof_steps)

    @property
    def core_size(self) -> int:
        """Size of the unsatisfiable core."""
        return len(self.core_constraints)

    def __repr__(self) -> str:
        itp = ", with_interpolant" if self.interpolant else ""
        return f"SMTProof(steps={self.size}, core={self.core_size}{itp})"


@dataclass
class SMTCheckResult:
    """Result of an SMT satisfiability check.

    Attributes:
        result: SAT, UNSAT, UNKNOWN, or TIMEOUT.
        model: Satisfying assignment (if SAT).
        proof: Unsatisfiability proof (if UNSAT and proofs enabled).
        solving_time: Time spent solving in seconds.
        num_conflicts: Number of DPLL conflicts encountered.
        num_decisions: Number of DPLL decisions made.
    """

    result: SolverResult
    model: Optional[SMTModel] = None
    proof: Optional[SMTProof] = None
    solving_time: float = 0.0
    num_conflicts: int = 0
    num_decisions: int = 0

    def __post_init__(self) -> None:
        if self.result == SolverResult.SAT and self.model is None:
            raise ValueError("model must be provided when result is SAT")

    @property
    def is_sat(self) -> bool:
        """Whether the formula is satisfiable."""
        return self.result == SolverResult.SAT

    @property
    def is_unsat(self) -> bool:
        """Whether the formula is unsatisfiable."""
        return self.result == SolverResult.UNSAT

    def __repr__(self) -> str:
        return (
            f"SMTCheckResult(result={self.result.name}, "
            f"time={self.solving_time:.2f}s, "
            f"conflicts={self.num_conflicts})"
        )


@dataclass
class SMTVerifyResult:
    """Result of SMT-based DP verification.

    Attributes:
        verified: Whether the mechanism satisfies DP.
        check_result: Underlying SMT check result.
        counterexample: Concrete counterexample if verification failed.
        privacy_budget: The privacy budget that was verified.
        encoding_size: Number of SMT constraints in the encoding.
    """

    verified: bool
    check_result: SMTCheckResult
    counterexample: Optional[Tuple[int, int, int, float]] = None
    privacy_budget: Optional[PrivacyBudget] = None
    encoding_size: int = 0

    def to_verify_result(self) -> VerifyResult:
        """Convert to the standard VerifyResult type."""
        return VerifyResult(valid=self.verified, violation=self.counterexample)

    def __repr__(self) -> str:
        return (
            f"SMTVerifyResult(verified={self.verified}, "
            f"encoding={self.encoding_size} constraints, "
            f"time={self.check_result.solving_time:.2f}s)"
        )


# ---------------------------------------------------------------------------
# Protocols (interfaces)
# ---------------------------------------------------------------------------


@runtime_checkable
class TheorySolverProtocol(Protocol):
    """Protocol for theory solvers in DPLL(T)."""

    def check_consistency(
        self, literals: List[Formula]
    ) -> Tuple[bool, Optional[List[Formula]]]:
        """Check if a set of theory literals is consistent.

        Returns:
            Tuple of (is_consistent, conflict_clause_or_none).
        """
        ...

    def propagate(self, literals: List[Formula]) -> List[Formula]:
        """Theory propagation: derive new literals from current assignment."""
        ...


# ---------------------------------------------------------------------------
# Public API classes
# ---------------------------------------------------------------------------


class SMTEncoder:
    """Encode DP mechanism constraints as SMT formulas.

    Translates mechanism probability tables and privacy constraints
    into quantifier-free linear real arithmetic (QF_LRA) formulas.
    """

    def __init__(self, config: Optional[SMTConfig] = None) -> None:
        self.config = config or SMTConfig()

    def encode_mechanism(
        self,
        mechanism: npt.NDArray[np.float64],
        adjacency: AdjacencyRelation,
        budget: PrivacyBudget,
    ) -> Tuple[List[SMTVariable], List[SMTConstraint]]:
        """Encode a mechanism as SMT variables and constraints.

        Args:
            mechanism: The n × k probability table.
            adjacency: Adjacency relation.
            budget: Privacy budget.

        Returns:
            Tuple of (variables, constraints).
        """
        from dp_forge.smt.encoder import SMTEncoderImpl
        impl = SMTEncoderImpl(self.config)
        return impl.encode_mechanism(mechanism, adjacency, budget)

    def encode_privacy_violation(
        self,
        n: int,
        k: int,
        adjacency: AdjacencyRelation,
        budget: PrivacyBudget,
    ) -> Tuple[List[SMTVariable], List[SMTConstraint]]:
        """Encode the negation of DP (existence of a violation).

        The conjunction of mechanism constraints and these constraints
        is satisfiable iff the mechanism violates DP.

        Args:
            n: Number of inputs.
            k: Number of output bins.
            adjacency: Adjacency relation.
            budget: Privacy budget.

        Returns:
            Tuple of (variables, constraints).
        """
        from dp_forge.smt.encoder import SMTEncoderImpl
        impl = SMTEncoderImpl(self.config)
        return impl.encode_privacy_violation(n, k, adjacency, budget)

    def encode_formula(self, formula: Formula) -> List[SMTConstraint]:
        """Convert a generic formula to SMT constraints.

        Args:
            formula: Formula to encode.

        Returns:
            List of SMT constraints.
        """
        from dp_forge.smt.encoder import SMTEncoderImpl
        impl = SMTEncoderImpl(self.config)
        return impl.encode_formula(formula)


class DPLLTSolver:
    """DPLL(T) solver for satisfiability modulo theories.

    Implements the core DPLL(T) loop with Boolean search (DPLL)
    combined with theory checking (T-solver).
    """

    def __init__(self, config: Optional[SMTConfig] = None) -> None:
        self.config = config or SMTConfig()

    def check_sat(
        self,
        variables: List[SMTVariable],
        constraints: List[SMTConstraint],
    ) -> SMTCheckResult:
        """Check satisfiability of the given constraints.

        Delegates to the DPLLTSolverImpl from dp_forge.smt.dpll_t.

        Args:
            variables: Variable declarations.
            constraints: Constraints to check.

        Returns:
            SMTCheckResult with SAT/UNSAT/UNKNOWN and model/proof.
        """
        import time as _time
        from dp_forge.smt.dpll_t import DPLLTSolverImpl, Literal, LiteralPolarity
        from dp_forge.smt.theory_solver import (
            FeasibilityChecker,
            LinearConstraint,
            parse_linear_constraint,
        )

        start = _time.time()

        impl = DPLLTSolverImpl(
            timeout=self.config.timeout_seconds,
            produce_proofs=self.config.produce_proofs,
            produce_models=self.config.produce_models,
        )

        # Register variable bounds as theory atoms
        for v in variables:
            if v.lower_bound is not None:
                lb_formula = Formula(
                    expr=f"{v.name} >= {v.lower_bound}",
                    variables=frozenset({v.name}),
                )
                atom_id = impl.new_theory_atom(lb_formula)
                impl.add_unit(Literal(atom_id, LiteralPolarity.POSITIVE))
            if v.upper_bound is not None:
                ub_formula = Formula(
                    expr=f"{v.name} <= {v.upper_bound}",
                    variables=frozenset({v.name}),
                )
                atom_id = impl.new_theory_atom(ub_formula)
                impl.add_unit(Literal(atom_id, LiteralPolarity.POSITIVE))

        # Register constraints as theory atoms with unit clauses
        for c in constraints:
            atom_id = impl.new_theory_atom(c.formula)
            impl.add_unit(Literal(atom_id, LiteralPolarity.POSITIVE))

        # Run DPLL(T) loop
        result_str, bool_model, proof_steps = impl.solve()
        elapsed = _time.time() - start

        if result_str == "SAT":
            # Extract real-valued model via feasibility checker
            checker = FeasibilityChecker()
            lra_constraints: List[LinearConstraint] = []
            var_bounds: Dict[str, tuple] = {}
            for v in variables:
                var_bounds[v.name] = (v.lower_bound, v.upper_bound)
            for c in constraints:
                lc = parse_linear_constraint(c.formula)
                if lc is not None:
                    lra_constraints.append(lc)
            _, model_vals = checker.check(lra_constraints, var_bounds)
            assignments = model_vals if model_vals is not None else {}
            return SMTCheckResult(
                result=SolverResult.SAT,
                model=SMTModel(assignments=assignments, is_complete=True),
                solving_time=elapsed,
            )
        elif result_str == "UNSAT":
            proof = SMTProof(
                proof_steps=proof_steps or ["DPLL(T) determined UNSAT"],
                core_constraints=[c.label or "" for c in constraints if c.label],
            )
            return SMTCheckResult(
                result=SolverResult.UNSAT,
                proof=proof if self.config.produce_proofs else None,
                solving_time=elapsed,
            )
        else:
            return SMTCheckResult(
                result=SolverResult.UNKNOWN,
                solving_time=elapsed,
            )

    def check_sat_assuming(
        self,
        variables: List[SMTVariable],
        constraints: List[SMTConstraint],
        assumptions: List[Formula],
    ) -> SMTCheckResult:
        """Check satisfiability under additional assumptions.

        Args:
            variables: Variable declarations.
            constraints: Constraints to check.
            assumptions: Additional assumed literals.

        Returns:
            SMTCheckResult.
        """
        extra = [SMTConstraint(formula=a) for a in assumptions]
        return self.check_sat(variables, constraints + extra)


class SMTVerifier:
    """High-level SMT-based DP verification.

    Encodes DP verification as an SMT problem: the mechanism satisfies
    (ε, δ)-DP iff the negation (existence of a violation) is UNSAT.
    """

    def __init__(self, config: Optional[SMTConfig] = None) -> None:
        self.config = config or SMTConfig()

    def verify(
        self,
        mechanism: npt.NDArray[np.float64],
        epsilon: float,
        delta: float = 0.0,
        adjacency: Optional[AdjacencyRelation] = None,
    ) -> SMTVerifyResult:
        """Verify DP guarantees via SMT.

        Args:
            mechanism: The n × k probability table.
            epsilon: Privacy parameter ε.
            delta: Privacy parameter δ.
            adjacency: Adjacency relation (defaults to Hamming-1).

        Returns:
            SMTVerifyResult with verification outcome.
        """
        n, k = mechanism.shape
        if adjacency is None:
            adjacency = AdjacencyRelation.hamming_distance_1(n)
        budget = PrivacyBudget(epsilon=epsilon, delta=delta)

        encoder = SMTEncoder(self.config)
        variables, constraints = encoder.encode_mechanism(mechanism, adjacency, budget)

        solver = DPLLTSolver(self.config)
        check_result = solver.check_sat(variables, constraints)

        # If the mechanism constraints + DP constraints are satisfiable,
        # the mechanism satisfies DP (the encoding is of the DP conditions).
        # For verification, we encode the violation and check if it's UNSAT.
        # Re-encode as violation check
        from dp_forge.smt.encoder import SMTEncoderImpl
        enc = SMTEncoderImpl(self.config)
        from dp_forge.smt.encoder import MechanismEncoder
        mech_enc = MechanismEncoder()
        m_vars, m_cons, var_names = mech_enc.encode_fixed_mechanism(mechanism)

        from dp_forge.smt.encoder import PrivacyConstraintEncoder
        priv_enc = PrivacyConstraintEncoder(self.config.encode_strategy)
        v_vars, v_cons = priv_enc.encode_violation(
            n, k, epsilon, delta, adjacency, var_names
        )

        all_vars = m_vars + v_vars
        all_cons = m_cons + v_cons

        viol_result = solver.check_sat(all_vars, all_cons)

        if viol_result.is_unsat:
            return SMTVerifyResult(
                verified=True,
                check_result=viol_result,
                privacy_budget=budget,
                encoding_size=len(all_cons),
            )
        else:
            cex = None
            if viol_result.model is not None:
                extractor = CounterexampleExtractor()
                cex = extractor.extract(viol_result.model, n, k)
            return SMTVerifyResult(
                verified=False,
                check_result=viol_result,
                counterexample=cex,
                privacy_budget=budget,
                encoding_size=len(all_cons),
            )

    def verify_with_interpolant(
        self,
        mechanism: npt.NDArray[np.float64],
        epsilon: float,
        delta: float = 0.0,
        adjacency: Optional[AdjacencyRelation] = None,
    ) -> Tuple[SMTVerifyResult, Optional[Formula]]:
        """Verify DP and extract an interpolant from the UNSAT proof.

        If verification succeeds (UNSAT), derives a Craig interpolant
        from the proof that can be used for CEGAR refinement.

        Returns:
            Tuple of (result, interpolant_or_none).
        """
        result = self.verify(mechanism, epsilon, delta, adjacency)
        interpolant = None
        if result.verified and result.check_result.proof is not None:
            interpolant = result.check_result.proof.interpolant
        return result, interpolant


class CounterexampleExtractor:
    """Extract concrete counterexamples from SMT models."""

    def extract(
        self,
        model: SMTModel,
        n: int,
        k: int,
    ) -> Tuple[int, int, int, float]:
        """Extract a concrete DP violation from an SMT model.

        Args:
            model: Satisfying assignment from the SMT solver.
            n: Number of inputs.
            k: Number of output bins.

        Returns:
            Tuple of (i, i_prime, j_worst, magnitude).
        """
        best_i, best_ip, best_j = 0, 1, 0
        best_mag = 0.0

        for i in range(n):
            for ip in range(n):
                if i == ip:
                    continue
                for j in range(k):
                    p_ij = model.get(f"p_{i}_{j}")
                    p_ipj = model.get(f"p_{ip}_{j}")
                    if p_ij is not None and p_ipj is not None and p_ipj > 1e-15:
                        ratio = p_ij / p_ipj
                        if ratio > best_mag:
                            best_mag = ratio
                            best_i, best_ip, best_j = i, ip, j

        import math as _math
        log_mag = _math.log(best_mag) if best_mag > 0 else 0.0
        return (best_i, best_ip, best_j, log_mag)


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------


def smt_verify(
    mechanism: npt.NDArray[np.float64],
    epsilon: float,
    delta: float = 0.0,
    *,
    config: Optional[SMTConfig] = None,
) -> SMTVerifyResult:
    """Convenience function for SMT-based DP verification.

    Args:
        mechanism: The n × k probability table.
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ.
        config: Optional SMT configuration.

    Returns:
        SMTVerifyResult with verification outcome.
    """
    verifier = SMTVerifier(config=config)
    return verifier.verify(mechanism, epsilon, delta)


def encode_dp_constraints(
    n: int,
    k: int,
    epsilon: float,
    delta: float = 0.0,
    *,
    adjacency: Optional[AdjacencyRelation] = None,
) -> Tuple[List[SMTVariable], List[SMTConstraint]]:
    """Encode (ε, δ)-DP constraints as SMT formulas.

    Utility function for generating SMT encodings of DP constraints
    without tying to a specific mechanism.

    Args:
        n: Number of inputs.
        k: Number of output bins.
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ.
        adjacency: Adjacency relation.

    Returns:
        Tuple of (variables, constraints).
    """
    if adjacency is None:
        adjacency = AdjacencyRelation.hamming_distance_1(n)
    budget = PrivacyBudget(epsilon=epsilon, delta=delta)
    encoder = SMTEncoder()
    return encoder.encode_privacy_violation(n, k, adjacency, budget)


__all__ = [
    # Enums
    "Theory",
    "SolverResult",
    "PropagationRule",
    "EncodeStrategy",
    # Config
    "SMTConfig",
    # Data types
    "SMTVariable",
    "SMTConstraint",
    "SMTModel",
    "SMTProof",
    "SMTCheckResult",
    "SMTVerifyResult",
    # Protocols
    "TheorySolverProtocol",
    # Classes
    "SMTEncoder",
    "DPLLTSolver",
    "SMTVerifier",
    "CounterexampleExtractor",
    # Functions
    "smt_verify",
    "encode_dp_constraints",
]
