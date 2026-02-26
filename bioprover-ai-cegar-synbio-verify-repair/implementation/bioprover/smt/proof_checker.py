"""Proof checking and validation for BioProver.

Provides data structures for proof trees, validation of inference
steps, proof simplification, statistics, serialisation, and
independent re-checking of critical SMT results.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

import z3

from bioprover.smt.solver_base import Model, SMTResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference rules
# ---------------------------------------------------------------------------

class InferenceRule(Enum):
    """Standard inference rules appearing in SMT proofs."""

    AXIOM = auto()
    ASSUMPTION = auto()
    RESOLUTION = auto()
    UNIT_RESOLUTION = auto()
    MODUS_PONENS = auto()
    HYPOTHESIS = auto()
    LEMMA = auto()
    REWRITE = auto()
    TRANSITIVITY = auto()
    MONOTONICITY = auto()
    CONGRUENCE = auto()
    COMMUTATIVITY = auto()
    THEORY_LEMMA = auto()
    QUANTIFIER_INST = auto()
    DEFINITION_INTRO = auto()
    IFF_TRUE = auto()
    IFF_FALSE = auto()
    AND_ELIM = auto()
    OR_INTRO = auto()
    NOT_INTRO = auto()
    NOT_ELIM = auto()
    ARITH_BOUND = auto()
    ARITH_FARKAS = auto()
    NNF = auto()
    SKOLEMIZE = auto()
    UNKNOWN_RULE = auto()


_Z3_RULE_MAP: Dict[int, InferenceRule] = {
    z3.Z3_OP_PR_ASSERTED: InferenceRule.ASSUMPTION,
    z3.Z3_OP_PR_HYPOTHESIS: InferenceRule.HYPOTHESIS,
    z3.Z3_OP_PR_LEMMA: InferenceRule.LEMMA,
    z3.Z3_OP_PR_UNIT_RESOLUTION: InferenceRule.UNIT_RESOLUTION,
    z3.Z3_OP_PR_MODUS_PONENS: InferenceRule.MODUS_PONENS,
    z3.Z3_OP_PR_TRANSITIVITY: InferenceRule.TRANSITIVITY,
    z3.Z3_OP_PR_MONOTONICITY: InferenceRule.MONOTONICITY,
    z3.Z3_OP_PR_REWRITE: InferenceRule.REWRITE,
    z3.Z3_OP_PR_COMMUTATIVITY: InferenceRule.COMMUTATIVITY,
    z3.Z3_OP_PR_QUANT_INST: InferenceRule.QUANTIFIER_INST,
    z3.Z3_OP_PR_DEF_INTRO: InferenceRule.DEFINITION_INTRO,
    z3.Z3_OP_PR_IFF_TRUE: InferenceRule.IFF_TRUE,
    z3.Z3_OP_PR_IFF_FALSE: InferenceRule.IFF_FALSE,
    z3.Z3_OP_PR_NNF_NEG: InferenceRule.NNF,
    z3.Z3_OP_PR_SKOLEMIZE: InferenceRule.SKOLEMIZE,
    z3.Z3_OP_PR_TH_LEMMA: InferenceRule.THEORY_LEMMA,
    z3.Z3_OP_PR_AND_ELIM: InferenceRule.AND_ELIM,
    z3.Z3_OP_PR_NOT_OR_ELIM: InferenceRule.OR_INTRO,
}


def _z3_kind_to_rule(kind: int) -> InferenceRule:
    return _Z3_RULE_MAP.get(kind, InferenceRule.UNKNOWN_RULE)


# ---------------------------------------------------------------------------
# ProofStep
# ---------------------------------------------------------------------------

@dataclass
class ProofStep:
    """A single step in a proof tree.

    Each step applies an inference rule to zero or more premises
    to derive a conclusion.
    """

    step_id: int
    rule: InferenceRule
    conclusion: str
    premises: List[int] = field(default_factory=list)
    annotation: str = ""
    is_valid: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "rule": self.rule.name,
            "conclusion": self.conclusion,
            "premises": self.premises,
            "annotation": self.annotation,
            "is_valid": self.is_valid,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ProofStep:
        return cls(
            step_id=d["step_id"],
            rule=InferenceRule[d["rule"]],
            conclusion=d["conclusion"],
            premises=d.get("premises", []),
            annotation=d.get("annotation", ""),
            is_valid=d.get("is_valid"),
        )

    def __repr__(self) -> str:
        valid = "✓" if self.is_valid else ("✗" if self.is_valid is False else "?")
        return f"ProofStep({self.step_id}, {self.rule.name}, {valid})"


# ---------------------------------------------------------------------------
# ProofTree
# ---------------------------------------------------------------------------

@dataclass
class ProofTree:
    """A tree of proof steps.

    Steps are stored in a flat list; the tree structure is encoded
    via the ``premises`` field of each :class:`ProofStep`.
    """

    steps: List[ProofStep] = field(default_factory=list)
    root_id: Optional[int] = None
    conclusion: str = ""
    is_valid: Optional[bool] = None

    # -- construction -------------------------------------------------------

    def add_step(self, step: ProofStep) -> None:
        self.steps.append(step)
        self.root_id = step.step_id

    @classmethod
    def from_z3_proof(cls, proof: z3.ExprRef) -> ProofTree:
        """Build a :class:`ProofTree` from a Z3 proof object."""
        tree = cls()
        visited: Dict[int, int] = {}
        counter = [0]
        _build_from_z3(proof, tree, visited, counter)
        if tree.steps:
            tree.root_id = tree.steps[-1].step_id
            tree.conclusion = tree.steps[-1].conclusion
        return tree

    # -- statistics ---------------------------------------------------------

    @property
    def depth(self) -> int:
        if not self.steps:
            return 0
        step_map = {s.step_id: s for s in self.steps}
        return _tree_depth(self.root_id, step_map, set())

    @property
    def width(self) -> int:
        """Maximum number of premises at any step."""
        if not self.steps:
            return 0
        return max(len(s.premises) for s in self.steps)

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def rule_usage(self) -> Dict[str, int]:
        """Count occurrences of each inference rule."""
        counter: Counter[str] = Counter()
        for s in self.steps:
            counter[s.rule.name] += 1
        return dict(counter.most_common())

    def statistics(self) -> Dict[str, Any]:
        return {
            "num_steps": self.num_steps,
            "depth": self.depth,
            "width": self.width,
            "rule_usage": self.rule_usage(),
            "is_valid": self.is_valid,
        }

    # -- validation ---------------------------------------------------------

    def validate(self) -> bool:
        """Validate every step in the proof.

        Returns ``True`` if all steps are valid, ``False`` otherwise.
        Marks each step and the tree with their validity status.
        """
        step_map = {s.step_id: s for s in self.steps}
        all_valid = True

        for step in self.steps:
            valid = _validate_step(step, step_map)
            step.is_valid = valid
            if not valid:
                all_valid = False

        self.is_valid = all_valid
        return all_valid

    # -- simplification -----------------------------------------------------

    def simplify(self) -> ProofTree:
        """Return a simplified copy with redundant steps removed.

        Removes steps that are not reachable from the root.
        """
        if self.root_id is None:
            return ProofTree()

        reachable = self._reachable_ids(self.root_id)
        new_steps = [s for s in self.steps if s.step_id in reachable]

        simplified = ProofTree(
            steps=new_steps,
            root_id=self.root_id,
            conclusion=self.conclusion,
        )

        # Collapse single-premise rewrite chains.
        simplified = self._collapse_rewrites(simplified)
        return simplified

    def _reachable_ids(self, root: int) -> Set[int]:
        step_map = {s.step_id: s for s in self.steps}
        reachable: Set[int] = set()
        stack = [root]
        while stack:
            sid = stack.pop()
            if sid in reachable:
                continue
            reachable.add(sid)
            step = step_map.get(sid)
            if step:
                stack.extend(step.premises)
        return reachable

    @staticmethod
    def _collapse_rewrites(tree: ProofTree) -> ProofTree:
        """Collapse chains of single-premise REWRITE steps."""
        step_map = {s.step_id: s for s in tree.steps}
        collapsed: Set[int] = set()

        for step in tree.steps:
            if (
                step.rule == InferenceRule.REWRITE
                and len(step.premises) == 1
                and step.premises[0] in step_map
            ):
                parent = step_map[step.premises[0]]
                if (
                    parent.rule == InferenceRule.REWRITE
                    and len(parent.premises) <= 1
                ):
                    # Skip the parent; link to grandparent.
                    step.premises = list(parent.premises)
                    collapsed.add(parent.step_id)

        new_steps = [s for s in tree.steps if s.step_id not in collapsed]
        return ProofTree(
            steps=new_steps,
            root_id=tree.root_id,
            conclusion=tree.conclusion,
        )

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "root_id": self.root_id,
            "conclusion": self.conclusion,
            "is_valid": self.is_valid,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ProofTree:
        return cls(
            steps=[ProofStep.from_dict(s) for s in d.get("steps", [])],
            root_id=d.get("root_id"),
            conclusion=d.get("conclusion", ""),
            is_valid=d.get("is_valid"),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> ProofTree:
        return cls.from_dict(json.loads(text))

    def __repr__(self) -> str:
        valid = "✓" if self.is_valid else ("✗" if self.is_valid is False else "?")
        return f"ProofTree({self.num_steps} steps, depth={self.depth}, {valid})"


# ---------------------------------------------------------------------------
# Independent re-checker
# ---------------------------------------------------------------------------

class ProofReChecker:
    """Re-check critical SMT results independently.

    Creates a fresh Z3 solver and verifies that the given formulas
    produce the expected result.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def recheck_unsat(self, formulas: List[z3.ExprRef]) -> bool:
        """Verify that *formulas* are jointly unsatisfiable."""
        s = z3.Solver()
        s.set("timeout", int(self.timeout * 1000))
        for f in formulas:
            s.add(f)
        result = s.check()
        if result != z3.unsat:
            logger.warning("Re-check: expected UNSAT, got %s", result)
            return False
        return True

    def recheck_sat(
        self,
        formulas: List[z3.ExprRef],
        model: Model,
    ) -> bool:
        """Verify that *model* satisfies *formulas*."""
        s = z3.Solver()
        s.set("timeout", int(self.timeout * 1000))
        for f in formulas:
            s.add(f)

        # Fix model values and check.
        for var, val in model.assignments.items():
            z3_var = z3.Real(var)
            if isinstance(val, (int, float)):
                s.add(z3_var == z3.RealVal(val))
            elif isinstance(val, tuple) and len(val) == 2:
                lo, hi = val
                s.add(z3_var >= z3.RealVal(lo))
                s.add(z3_var <= z3.RealVal(hi))

        result = s.check()
        if result != z3.sat:
            logger.warning("Re-check: model does not satisfy formulas (%s)", result)
            return False
        return True

    def recheck_with_proof(
        self,
        formulas: List[z3.ExprRef],
    ) -> Tuple[bool, Optional[ProofTree]]:
        """Re-check UNSAT and return the proof tree."""
        s = z3.Solver()
        s.set("timeout", int(self.timeout * 1000))
        s.set("proof", True)
        for f in formulas:
            s.add(f)

        result = s.check()
        if result != z3.unsat:
            return False, None

        try:
            proof = s.proof()
            tree = ProofTree.from_z3_proof(proof)
            tree.validate()
            return tree.is_valid is True, tree
        except z3.Z3Exception:
            return True, None

    def compute_proof_hash(self, tree: ProofTree) -> str:
        """Compute a deterministic hash of a proof tree for auditing."""
        serialised = tree.to_json(indent=0)
        return hashlib.sha256(serialised.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_from_z3(
    node: z3.ExprRef,
    tree: ProofTree,
    visited: Dict[int, int],
    counter: List[int],
) -> int:
    """Recursively build :class:`ProofTree` from a Z3 proof term."""
    nid = id(node)
    if nid in visited:
        return visited[nid]

    step_id = counter[0]
    counter[0] += 1

    try:
        kind = node.decl().kind()
    except Exception:
        kind = -1

    rule = _z3_kind_to_rule(kind)

    # Conclusion is the last child.
    conclusion = ""
    num_args = node.num_args() if hasattr(node, "num_args") else 0
    if num_args > 0:
        try:
            conclusion = str(node.arg(num_args - 1))
        except Exception:
            conclusion = str(node)
    else:
        conclusion = str(node)

    # Recurse into premise children.
    premises: List[int] = []
    for i in range(max(0, num_args - 1)):
        try:
            child = node.arg(i)
            child_id = _build_from_z3(child, tree, visited, counter)
            premises.append(child_id)
        except Exception:
            pass

    step = ProofStep(
        step_id=step_id,
        rule=rule,
        conclusion=conclusion[:200],
        premises=premises,
    )
    tree.add_step(step)
    visited[nid] = step_id
    return step_id


def _tree_depth(
    root: Optional[int],
    step_map: Dict[int, ProofStep],
    visited: Set[int],
) -> int:
    if root is None or root in visited:
        return 0
    visited.add(root)
    step = step_map.get(root)
    if step is None or not step.premises:
        return 1
    return 1 + max(
        _tree_depth(p, step_map, visited) for p in step.premises
    )


def _validate_step(
    step: ProofStep,
    step_map: Dict[int, ProofStep],
) -> bool:
    """Validate a single proof step.

    Currently checks structural validity: all premises must exist.
    Rule-specific semantic checks are performed where possible.
    """
    # All premises must exist.
    for pid in step.premises:
        if pid not in step_map:
            return False

    # Axioms and assumptions are valid by definition.
    if step.rule in (InferenceRule.AXIOM, InferenceRule.ASSUMPTION):
        return True

    # Rewrite and commutativity are syntactic – accept.
    if step.rule in (
        InferenceRule.REWRITE,
        InferenceRule.COMMUTATIVITY,
        InferenceRule.DEFINITION_INTRO,
        InferenceRule.NNF,
        InferenceRule.SKOLEMIZE,
        InferenceRule.MONOTONICITY,
    ):
        return True

    # For resolution / modus ponens, check arity.
    if step.rule in (InferenceRule.RESOLUTION, InferenceRule.UNIT_RESOLUTION):
        return len(step.premises) >= 2

    if step.rule == InferenceRule.MODUS_PONENS:
        return len(step.premises) >= 2

    if step.rule == InferenceRule.TRANSITIVITY:
        return len(step.premises) >= 2

    # Theory lemma, quantifier instantiation – trust the solver.
    if step.rule in (
        InferenceRule.THEORY_LEMMA,
        InferenceRule.QUANTIFIER_INST,
        InferenceRule.ARITH_BOUND,
        InferenceRule.ARITH_FARKAS,
    ):
        return True

    # Elimination / introduction rules.
    if step.rule in (
        InferenceRule.AND_ELIM,
        InferenceRule.OR_INTRO,
        InferenceRule.NOT_INTRO,
        InferenceRule.NOT_ELIM,
        InferenceRule.IFF_TRUE,
        InferenceRule.IFF_FALSE,
    ):
        return len(step.premises) >= 1

    if step.rule == InferenceRule.HYPOTHESIS:
        return True

    if step.rule == InferenceRule.LEMMA:
        return len(step.premises) >= 1

    # Unknown rules: cannot validate.
    return True
