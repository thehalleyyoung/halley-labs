"""Proof composition for compositional verification.

Builds proof trees from individual module verification results and
composition rules, validates composed proofs, generates certificates,
and localises failures when composition is unsuccessful.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from bioprover.compositional.contracts import (
    Contract,
    ContractRefinement,
    RefinementResult,
    SatisfactionResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Composition rules
# ---------------------------------------------------------------------------

class CompositionRule(Enum):
    """Rule used to compose sub-proofs."""
    SEQUENTIAL = auto()    # Cascade / pipeline
    PARALLEL = auto()      # Independent modules
    FEEDBACK = auto()      # Circular AG
    REFINEMENT = auto()    # Contract refinement
    CONJUNCTION = auto()   # Contract conjunction
    QUOTIENT = auto()      # Quotient decomposition


# ---------------------------------------------------------------------------
# Composable proof
# ---------------------------------------------------------------------------

@dataclass
class ComposableProof:
    """A single verification result that can be composed with others.

    Attributes:
        module_name: Name of the verified module.
        contract: The contract the module was verified against.
        result: Verification outcome.
        method: How the verification was performed.
        metadata: Additional diagnostics.
    """
    module_name: str
    contract: Contract
    result: SatisfactionResult
    method: str = "cegar"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.result.satisfied


# ---------------------------------------------------------------------------
# Proof tree
# ---------------------------------------------------------------------------

@dataclass
class ProofTreeNode:
    """A node in the proof tree.

    Leaf nodes wrap a ComposableProof. Internal nodes apply a
    CompositionRule to child nodes.
    """
    label: str
    rule: Optional[CompositionRule] = None
    proof: Optional[ComposableProof] = None
    children: List[ProofTreeNode] = field(default_factory=list)
    contract: Optional[Contract] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return self.proof is not None and not self.children

    @property
    def is_valid(self) -> bool:
        if self.is_leaf:
            return self.proof is not None and self.proof.is_valid
        return all(c.is_valid for c in self.children)


class ProofTree:
    """A hierarchical proof tree for compositional verification.

    Leaf nodes are individual module proofs. Internal nodes correspond
    to composition rules (sequential, parallel, feedback).
    """

    def __init__(self, root: Optional[ProofTreeNode] = None) -> None:
        self.root = root

    # -- construction -------------------------------------------------------

    @staticmethod
    def leaf(proof: ComposableProof) -> ProofTreeNode:
        """Create a leaf node from a module proof."""
        return ProofTreeNode(
            label=proof.module_name,
            proof=proof,
            contract=proof.contract,
        )

    @staticmethod
    def compose(
        rule: CompositionRule,
        children: List[ProofTreeNode],
        composed_contract: Contract,
        label: Optional[str] = None,
    ) -> ProofTreeNode:
        """Create an internal node by composing child proofs."""
        node_label = label or f"{rule.name}({', '.join(c.label for c in children)})"
        return ProofTreeNode(
            label=node_label,
            rule=rule,
            children=list(children),
            contract=composed_contract,
        )

    # -- sequential composition ---------------------------------------------

    @staticmethod
    def sequential(
        stages: List[ProofTreeNode],
        system_contract: Contract,
        label: Optional[str] = None,
    ) -> ProofTreeNode:
        """Compose proofs for a cascade / pipeline.

        The output signals of stage i feed into the input signals of
        stage i+1. The composition is valid if every stage satisfies
        its contract and the interfaces are compatible.
        """
        return ProofTree.compose(
            CompositionRule.SEQUENTIAL,
            stages,
            system_contract,
            label=label or "sequential",
        )

    # -- parallel composition -----------------------------------------------

    @staticmethod
    def parallel(
        branches: List[ProofTreeNode],
        system_contract: Contract,
        label: Optional[str] = None,
    ) -> ProofTreeNode:
        """Compose proofs for independent parallel modules."""
        return ProofTree.compose(
            CompositionRule.PARALLEL,
            branches,
            system_contract,
            label=label or "parallel",
        )

    # -- feedback composition -----------------------------------------------

    @staticmethod
    def feedback(
        modules: List[ProofTreeNode],
        system_contract: Contract,
        circular_ag_result: Optional[Any] = None,
        label: Optional[str] = None,
    ) -> ProofTreeNode:
        """Compose proofs with circular dependencies.

        Requires a valid circular AG result to justify the circular
        reasoning.
        """
        node = ProofTree.compose(
            CompositionRule.FEEDBACK,
            modules,
            system_contract,
            label=label or "feedback",
        )
        if circular_ag_result is not None:
            node.metadata["circular_ag_result"] = circular_ag_result
        return node

    # -- traversal ----------------------------------------------------------

    def leaves(self) -> List[ProofTreeNode]:
        """Return all leaf nodes."""
        if self.root is None:
            return []
        return self._collect_leaves(self.root)

    def depth(self) -> int:
        """Return the depth of the proof tree."""
        if self.root is None:
            return 0
        return self._depth(self.root)

    def node_count(self) -> int:
        """Total number of nodes."""
        if self.root is None:
            return 0
        return self._count(self.root)

    # -- visualisation ------------------------------------------------------

    def ascii_render(self) -> str:
        """Render the proof tree as an ASCII diagram."""
        if self.root is None:
            return "(empty proof tree)"
        lines: List[str] = []
        self._render_node(self.root, lines, prefix="", is_last=True)
        return "\n".join(lines)

    def _render_node(
        self,
        node: ProofTreeNode,
        lines: List[str],
        prefix: str,
        is_last: bool,
    ) -> None:
        connector = "└── " if is_last else "├── "
        status = "✓" if node.is_valid else "✗"

        if node.is_leaf:
            lines.append(f"{prefix}{connector}[{status}] {node.label}")
        else:
            rule_name = node.rule.name if node.rule else "?"
            lines.append(
                f"{prefix}{connector}[{status}] {node.label} ({rule_name})"
            )

        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(node.children):
            self._render_node(
                child, lines, child_prefix, is_last=(i == len(node.children) - 1)
            )

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _collect_leaves(node: ProofTreeNode) -> List[ProofTreeNode]:
        if node.is_leaf:
            return [node]
        result: List[ProofTreeNode] = []
        for child in node.children:
            result.extend(ProofTree._collect_leaves(child))
        return result

    @staticmethod
    def _depth(node: ProofTreeNode) -> int:
        if node.is_leaf:
            return 1
        return 1 + max((ProofTree._depth(c) for c in node.children), default=0)

    @staticmethod
    def _count(node: ProofTreeNode) -> int:
        return 1 + sum(ProofTree._count(c) for c in node.children)


# ---------------------------------------------------------------------------
# Proof validator
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    """A problem found during proof validation."""
    severity: str  # "error" | "warning"
    location: str  # path in the proof tree
    message: str


class ProofValidator:
    """Validate a composed proof tree for soundness.

    Checks:
        1. All leaf proofs are valid (modules satisfy their contracts).
        2. Composition rules are correctly applied.
        3. Interface compatibility between connected modules.
        4. For feedback composition, a valid circular AG result exists.
    """

    def validate(self, tree: ProofTree) -> List[ValidationIssue]:
        """Validate the entire proof tree.

        Returns a list of issues (empty = valid proof).
        """
        if tree.root is None:
            return [ValidationIssue("error", "/", "Empty proof tree.")]
        issues: List[ValidationIssue] = []
        self._validate_node(tree.root, issues, path="/")
        return issues

    def is_valid(self, tree: ProofTree) -> bool:
        issues = self.validate(tree)
        return not any(i.severity == "error" for i in issues)

    # -- recursive validation -----------------------------------------------

    def _validate_node(
        self,
        node: ProofTreeNode,
        issues: List[ValidationIssue],
        path: str,
    ) -> None:
        current_path = f"{path}{node.label}/"

        if node.is_leaf:
            self._validate_leaf(node, issues, current_path)
        else:
            self._validate_internal(node, issues, current_path)
            for child in node.children:
                self._validate_node(child, issues, current_path)

    def _validate_leaf(
        self,
        node: ProofTreeNode,
        issues: List[ValidationIssue],
        path: str,
    ) -> None:
        if node.proof is None:
            issues.append(ValidationIssue("error", path, "Leaf has no proof."))
            return
        if not node.proof.is_valid:
            issues.append(ValidationIssue(
                "error", path,
                f"Module '{node.proof.module_name}' failed verification: "
                f"{node.proof.result.message}"
            ))

    def _validate_internal(
        self,
        node: ProofTreeNode,
        issues: List[ValidationIssue],
        path: str,
    ) -> None:
        if node.rule is None:
            issues.append(
                ValidationIssue("error", path, "Internal node has no composition rule.")
            )
            return

        if not node.children:
            issues.append(
                ValidationIssue("error", path, "Internal node has no children.")
            )
            return

        # Rule-specific checks
        if node.rule == CompositionRule.SEQUENTIAL:
            self._validate_sequential(node, issues, path)
        elif node.rule == CompositionRule.PARALLEL:
            self._validate_parallel(node, issues, path)
        elif node.rule == CompositionRule.FEEDBACK:
            self._validate_feedback(node, issues, path)

    def _validate_sequential(
        self,
        node: ProofTreeNode,
        issues: List[ValidationIssue],
        path: str,
    ) -> None:
        """Validate sequential (cascade) composition.

        Check that outputs of stage i overlap with inputs of stage i+1.
        """
        for i in range(len(node.children) - 1):
            c_out = node.children[i]
            c_in = node.children[i + 1]

            out_names = self._get_output_names(c_out)
            in_names = self._get_input_names(c_in)

            if not out_names & in_names:
                issues.append(ValidationIssue(
                    "warning", path,
                    f"No shared signals between stage {i} ({c_out.label}) "
                    f"and stage {i+1} ({c_in.label})."
                ))

    def _validate_parallel(
        self,
        node: ProofTreeNode,
        issues: List[ValidationIssue],
        path: str,
    ) -> None:
        """Validate parallel composition.

        Check that parallel modules do not write to the same outputs.
        """
        all_outputs: Set[str] = set()
        for child in node.children:
            child_outputs = self._get_output_names(child)
            overlap = all_outputs & child_outputs
            if overlap:
                issues.append(ValidationIssue(
                    "error", path,
                    f"Output conflict in parallel composition: "
                    f"{overlap} written by multiple modules."
                ))
            all_outputs |= child_outputs

    def _validate_feedback(
        self,
        node: ProofTreeNode,
        issues: List[ValidationIssue],
        path: str,
    ) -> None:
        """Validate feedback composition.

        Requires a circular AG result in metadata.
        """
        ag_result = node.metadata.get("circular_ag_result")
        if ag_result is None:
            issues.append(ValidationIssue(
                "error", path,
                "Feedback composition requires a circular AG result."
            ))
            return

        if hasattr(ag_result, "proof_valid") and not ag_result.proof_valid:
            issues.append(ValidationIssue(
                "error", path,
                "Circular AG proof is not valid."
            ))

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _get_output_names(node: ProofTreeNode) -> Set[str]:
        if node.contract is not None:
            return set(node.contract.output_names)
        return set()

    @staticmethod
    def _get_input_names(node: ProofTreeNode) -> Set[str]:
        if node.contract is not None:
            return set(node.contract.input_names)
        return set()


# ---------------------------------------------------------------------------
# Proof certificate
# ---------------------------------------------------------------------------

@dataclass
class ProofCertificate:
    """Self-contained certificate for a composed verification result.

    Contains all information needed to independently verify the proof:
    individual module certificates, composition rules, and contracts.
    """
    system_name: str
    system_contract: Optional[Contract]
    module_certificates: List[ModuleCertificate]
    composition_rules: List[Dict[str, Any]]
    valid: bool
    validation_issues: List[ValidationIssue] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_name": self.system_name,
            "system_contract": (
                self.system_contract.to_dict() if self.system_contract else None
            ),
            "module_certificates": [mc.to_dict() for mc in self.module_certificates],
            "composition_rules": self.composition_rules,
            "valid": self.valid,
            "validation_issues": [
                {"severity": i.severity, "location": i.location, "message": i.message}
                for i in self.validation_issues
            ],
            "metadata": self.metadata,
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), indent=2, **kwargs)


@dataclass
class ModuleCertificate:
    """Certificate for an individual module verification."""
    module_name: str
    contract: Contract
    satisfied: bool
    method: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_name": self.module_name,
            "contract": self.contract.to_dict(),
            "satisfied": self.satisfied,
            "method": self.method,
            "diagnostics": self.diagnostics,
        }


def generate_certificate(
    tree: ProofTree,
    system_name: str = "system",
) -> ProofCertificate:
    """Generate a proof certificate from a validated proof tree.

    Args:
        tree: The proof tree to certify.
        system_name: Name for the composed system.

    Returns:
        A ProofCertificate.
    """
    validator = ProofValidator()
    issues = validator.validate(tree)

    module_certs: List[ModuleCertificate] = []
    for leaf in tree.leaves():
        if leaf.proof is not None:
            module_certs.append(ModuleCertificate(
                module_name=leaf.proof.module_name,
                contract=leaf.proof.contract,
                satisfied=leaf.proof.is_valid,
                method=leaf.proof.method,
                diagnostics=leaf.proof.metadata,
            ))

    composition_rules = _extract_rules(tree.root) if tree.root else []

    return ProofCertificate(
        system_name=system_name,
        system_contract=tree.root.contract if tree.root else None,
        module_certificates=module_certs,
        composition_rules=composition_rules,
        valid=not any(i.severity == "error" for i in issues),
        validation_issues=issues,
    )


# ---------------------------------------------------------------------------
# Compositional counterexample construction
# ---------------------------------------------------------------------------

@dataclass
class CompositionalCounterexample:
    """Counterexample localised to specific modules and interfaces.

    When composition fails, this identifies *where* the failure is.
    """
    failed_modules: List[str]
    failed_interfaces: List[Tuple[str, str]]
    root_cause: str
    module_counterexamples: Dict[str, Any] = field(default_factory=dict)
    suggested_fixes: List[str] = field(default_factory=list)


def localize_failure(tree: ProofTree) -> Optional[CompositionalCounterexample]:
    """Analyse a failed proof tree to localise the failure.

    Returns:
        A CompositionalCounterexample if the proof is invalid, else None.
    """
    if tree.root is None:
        return CompositionalCounterexample(
            failed_modules=[],
            failed_interfaces=[],
            root_cause="Empty proof tree.",
        )

    if tree.root.is_valid:
        return None

    failed_modules: List[str] = []
    module_cex: Dict[str, Any] = {}
    failed_interfaces: List[Tuple[str, str]] = []

    # Collect failed leaves
    for leaf in tree.leaves():
        if not leaf.is_valid and leaf.proof is not None:
            failed_modules.append(leaf.proof.module_name)
            if leaf.proof.result.counterexample is not None:
                module_cex[leaf.proof.module_name] = leaf.proof.result.counterexample

    # Identify failed interfaces at internal nodes
    _find_interface_failures(tree.root, failed_interfaces)

    root_cause = _diagnose_root_cause(tree.root, failed_modules, failed_interfaces)

    suggestions: List[str] = []
    for mod in failed_modules:
        suggestions.append(f"Re-verify module '{mod}' with refined abstraction.")
    for src, dst in failed_interfaces:
        suggestions.append(
            f"Check interface compatibility between '{src}' and '{dst}'."
        )

    return CompositionalCounterexample(
        failed_modules=failed_modules,
        failed_interfaces=failed_interfaces,
        root_cause=root_cause,
        module_counterexamples=module_cex,
        suggested_fixes=suggestions,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_rules(node: Optional[ProofTreeNode]) -> List[Dict[str, Any]]:
    """Extract composition rules from the proof tree."""
    if node is None:
        return []
    rules: List[Dict[str, Any]] = []
    if node.rule is not None:
        rules.append({
            "rule": node.rule.name,
            "label": node.label,
            "children": [c.label for c in node.children],
        })
    for child in node.children:
        rules.extend(_extract_rules(child))
    return rules


def _find_interface_failures(
    node: ProofTreeNode,
    failed: List[Tuple[str, str]],
) -> None:
    """Find interface mismatches at composition nodes."""
    if node.is_leaf or node.rule is None:
        return

    if node.rule == CompositionRule.SEQUENTIAL:
        for i in range(len(node.children) - 1):
            a = node.children[i]
            b = node.children[i + 1]
            if not a.is_valid or not b.is_valid:
                failed.append((a.label, b.label))

    for child in node.children:
        _find_interface_failures(child, failed)


def _diagnose_root_cause(
    root: ProofTreeNode,
    failed_modules: List[str],
    failed_interfaces: List[Tuple[str, str]],
) -> str:
    """Generate a human-readable root-cause diagnosis."""
    if not failed_modules and not failed_interfaces:
        return "No specific failure identified."

    parts: List[str] = []
    if failed_modules:
        parts.append(
            f"Module(s) failed verification: {', '.join(failed_modules)}."
        )
    if failed_interfaces:
        iface_strs = [f"{a}→{b}" for a, b in failed_interfaces]
        parts.append(
            f"Interface mismatch(es): {', '.join(iface_strs)}."
        )
    return " ".join(parts)
