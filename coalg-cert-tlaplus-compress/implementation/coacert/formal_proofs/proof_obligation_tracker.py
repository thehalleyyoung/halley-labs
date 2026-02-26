"""
Unified proof obligation tracking for CoaCert-TLA.

Collects and tracks proof obligations from all proof modules —
coherence, conformance, minimality, preservation, categorical diagrams,
and CTL*\\X preservation — and provides a single aggregate view.

This allows a user to ask: "Is the entire certification pipeline sound?"
and receive a single answer with a dependency chain showing which
theorems depend on which lemmas.

PROOF DEPENDENCY CHAIN:
  1. T-Fair Coherence (tfair_theorem.py)
     └─ Prerequisite for all downstream proofs
  2. Categorical Diagrams (categorical_diagram.py)
     └─ Naturality + unit + multiplication compatibility
     └─ Depends on: T-Fair Coherence
  3. Streett Acceptance Preservation (ctl_star_preservation.py)
     └─ Depends on: T-Fair Coherence
  4. CTL*\\X Preservation (ctl_star_preservation.py)
     └─ Depends on: T-Fair Coherence + Streett Acceptance
     └─ Uses: BCG88 + coalgebra morphism condition
  5. Minimality (minimality_proof.py)
     └─ Myhill-Nerode + partition refinement
     └─ Independent of coherence (but relevant for quotient quality)
  6. Conformance (conformance_certificate.py)
     └─ W-method depth sufficiency
     └─ Depends on: Minimality (for state bound)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Obligation status and categories
# ---------------------------------------------------------------------------

class ObligationCategory(Enum):
    """Category of a proof obligation."""
    COHERENCE = auto()
    CONFORMANCE = auto()
    MINIMALITY = auto()
    PRESERVATION = auto()
    CATEGORICAL_DIAGRAM = auto()
    CTL_STAR = auto()
    STREETT_ACCEPTANCE = auto()


class DischargeStatus(Enum):
    """Discharge status of a proof obligation."""
    PENDING = auto()
    DISCHARGED = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class ProofObligation:
    """A single proof obligation tracked by the system.

    Attributes:
        obligation_id: Unique identifier.
        category: Which proof module this obligation belongs to.
        description: Human-readable description of what must be proved.
        status: Current discharge status.
        depends_on: IDs of obligations that must be discharged first.
        discharged_by: Description of the witness or argument that discharged it.
        discharged_at: Timestamp when discharged.
        error_detail: If failed, explanation of the failure.
        source_module: Python module that generated this obligation.
    """
    obligation_id: str = ""
    category: ObligationCategory = ObligationCategory.COHERENCE
    description: str = ""
    status: DischargeStatus = DischargeStatus.PENDING
    depends_on: List[str] = field(default_factory=list)
    discharged_by: str = ""
    discharged_at: Optional[float] = None
    error_detail: str = ""
    source_module: str = ""

    def discharge(self, witness_description: str) -> None:
        """Mark this obligation as discharged."""
        self.status = DischargeStatus.DISCHARGED
        self.discharged_by = witness_description
        self.discharged_at = time.monotonic()

    def fail(self, detail: str) -> None:
        """Mark this obligation as failed."""
        self.status = DischargeStatus.FAILED
        self.error_detail = detail

    def skip(self, reason: str) -> None:
        """Mark this obligation as skipped."""
        self.status = DischargeStatus.SKIPPED
        self.discharged_by = reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.obligation_id,
            "category": self.category.name,
            "description": self.description,
            "status": self.status.name,
            "depends_on": self.depends_on,
            "discharged_by": self.discharged_by,
            "error_detail": self.error_detail,
            "source_module": self.source_module,
        }


# ---------------------------------------------------------------------------
# Proof obligation tracker
# ---------------------------------------------------------------------------

class ProofObligationTracker:
    """Unified tracker for all proof obligations in the certification pipeline.

    Registers obligations from every proof module, tracks their discharge
    status, enforces the dependency chain, and computes an aggregate
    proof hash covering the entire certification.

    Usage:
        tracker = ProofObligationTracker()
        tracker.register("coh-1", ObligationCategory.COHERENCE,
                         "T-Fair coherence for pair 0")
        tracker.register("cat-nat", ObligationCategory.CATEGORICAL_DIAGRAM,
                         "Naturality square commutes",
                         depends_on=["coh-1"])
        tracker.discharge("coh-1", "Exhaustive saturation check")
        tracker.discharge("cat-nat", "Diagram verification passed")
        summary = tracker.summary()
    """

    def __init__(self) -> None:
        self._obligations: Dict[str, ProofObligation] = {}
        self._registration_order: List[str] = []

    @property
    def obligations(self) -> List[ProofObligation]:
        """All obligations in registration order."""
        return [self._obligations[oid] for oid in self._registration_order]

    def register(
        self,
        obligation_id: str,
        category: ObligationCategory,
        description: str,
        depends_on: Optional[List[str]] = None,
        source_module: str = "",
    ) -> ProofObligation:
        """Register a new proof obligation.

        Parameters
        ----------
        obligation_id : str
            Unique identifier for this obligation.
        category : ObligationCategory
            Which proof module this belongs to.
        description : str
            What must be proved.
        depends_on : list of str, optional
            IDs of obligations that must be discharged first.
        source_module : str
            Python module that generated this obligation.

        Returns
        -------
        ProofObligation
            The registered obligation.
        """
        if obligation_id in self._obligations:
            logger.warning(
                "Obligation %s already registered; overwriting", obligation_id
            )
            self._registration_order = [
                oid for oid in self._registration_order if oid != obligation_id
            ]

        obl = ProofObligation(
            obligation_id=obligation_id,
            category=category,
            description=description,
            depends_on=list(depends_on or []),
            source_module=source_module,
        )
        self._obligations[obligation_id] = obl
        self._registration_order.append(obligation_id)
        return obl

    def discharge(
        self,
        obligation_id: str,
        witness_description: str,
    ) -> bool:
        """Discharge an obligation if all dependencies are met.

        Returns True if successfully discharged, False if dependencies
        are not yet discharged or the obligation is not found.
        """
        obl = self._obligations.get(obligation_id)
        if obl is None:
            logger.warning("Unknown obligation: %s", obligation_id)
            return False

        # Check dependencies
        for dep_id in obl.depends_on:
            dep = self._obligations.get(dep_id)
            if dep is None:
                obl.fail(f"Dependency {dep_id} not registered")
                return False
            if dep.status != DischargeStatus.DISCHARGED:
                obl.fail(
                    f"Dependency {dep_id} not discharged (status: {dep.status.name})"
                )
                return False

        obl.discharge(witness_description)
        return True

    def fail(self, obligation_id: str, detail: str) -> None:
        """Mark an obligation as failed."""
        obl = self._obligations.get(obligation_id)
        if obl is not None:
            obl.fail(detail)

    def skip(self, obligation_id: str, reason: str) -> None:
        """Mark an obligation as skipped."""
        obl = self._obligations.get(obligation_id)
        if obl is not None:
            obl.skip(reason)

    def get(self, obligation_id: str) -> Optional[ProofObligation]:
        """Get an obligation by ID."""
        return self._obligations.get(obligation_id)

    def by_category(
        self, category: ObligationCategory
    ) -> List[ProofObligation]:
        """Get all obligations in a given category."""
        return [
            obl for obl in self.obligations if obl.category == category
        ]

    def pending(self) -> List[ProofObligation]:
        """Get all pending obligations."""
        return [
            obl for obl in self.obligations
            if obl.status == DischargeStatus.PENDING
        ]

    def discharged(self) -> List[ProofObligation]:
        """Get all discharged obligations."""
        return [
            obl for obl in self.obligations
            if obl.status == DischargeStatus.DISCHARGED
        ]

    def failed(self) -> List[ProofObligation]:
        """Get all failed obligations."""
        return [
            obl for obl in self.obligations
            if obl.status == DischargeStatus.FAILED
        ]

    def all_discharged(self) -> bool:
        """Check if all registered obligations are discharged."""
        return all(
            obl.status == DischargeStatus.DISCHARGED
            for obl in self.obligations
        )

    def dependency_chain(self) -> List[List[str]]:
        """Compute the dependency chain as layers (topological sort).

        Returns a list of layers, where each layer contains obligation IDs
        whose dependencies are all in earlier layers. This gives the order
        in which obligations should be discharged.
        """
        remaining: Set[str] = set(self._obligations.keys())
        layers: List[List[str]] = []

        while remaining:
            # Find obligations with all deps satisfied (in earlier layers or not remaining)
            ready: List[str] = []
            for oid in sorted(remaining):
                obl = self._obligations[oid]
                deps_satisfied = all(
                    dep_id not in remaining for dep_id in obl.depends_on
                )
                if deps_satisfied:
                    ready.append(oid)

            if not ready:
                # Circular dependency — add all remaining as one layer
                layers.append(sorted(remaining))
                break

            layers.append(ready)
            remaining -= set(ready)

        return layers

    def compute_proof_hash(self) -> str:
        """Compute an aggregate SHA-256 hash over all obligations.

        This hash covers the obligation IDs, their statuses, and their
        witness descriptions, providing a tamper-evident summary of the
        entire proof state.
        """
        hasher = hashlib.sha256()
        for oid in self._registration_order:
            obl = self._obligations[oid]
            hasher.update(obl.obligation_id.encode())
            hasher.update(obl.category.name.encode())
            hasher.update(obl.status.name.encode())
            hasher.update(obl.discharged_by.encode())
            hasher.update(obl.error_detail.encode())
        return hasher.hexdigest()

    def summary(self) -> Dict[str, Any]:
        """Generate a unified proof summary.

        Returns a dictionary with:
        - Aggregate counts by status
        - Per-category breakdown
        - Dependency chain
        - Aggregate proof hash
        """
        total = len(self._obligations)
        by_status: Dict[str, int] = {}
        for obl in self.obligations:
            key = obl.status.name
            by_status[key] = by_status.get(key, 0) + 1

        by_category: Dict[str, Dict[str, int]] = {}
        for obl in self.obligations:
            cat = obl.category.name
            if cat not in by_category:
                by_category[cat] = {"total": 0, "discharged": 0, "failed": 0, "pending": 0, "skipped": 0}
            by_category[cat]["total"] += 1
            by_category[cat][obl.status.name.lower()] = (
                by_category[cat].get(obl.status.name.lower(), 0) + 1
            )

        proof_hash = self.compute_proof_hash()

        return {
            "total_obligations": total,
            "by_status": by_status,
            "by_category": by_category,
            "all_discharged": self.all_discharged(),
            "dependency_chain": self.dependency_chain(),
            "proof_hash": proof_hash,
            "obligations": [obl.to_dict() for obl in self.obligations],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the full summary to JSON."""
        return json.dumps(self.summary(), indent=indent)
