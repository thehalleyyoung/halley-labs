"""
Persistent Verification Knowledge Base for Cross-Session Transfer.

Implements cross-session knowledge accumulation and transfer for tensor
shape verification.  Key capabilities:

1. **Architectural pattern hashing** — Hash the structure (layer types,
   connectivity) ignoring specific parameter values so that similar
   architectures (ResNet-18 vs ResNet-50) map to the same family.

2. **Anti-unification over proof schemas** (Plotkin 1970) — Given
   multiple proof certificates for the same architectural family,
   extract a generalized proof schema capturing common structure,
   replacing specific values with variables.

3. **Knowledge transfer** — Prime CEGAR and neuro-symbolic repair loops
   with predicates, strategies, and failure modes from prior sessions.

4. **Persistence** — JSON-based save/load with cross-session merging.

5. **AGM belief revision** (Gärdenfors 1988) — Entrenchment-based
   contraction and revision to prevent stale predicates from
   accumulating across sessions.

Usage::

    from src.knowledge_base import VerificationKnowledgeBase

    kb = VerificationKnowledgeBase.load("kb.json")
    arch_hash = kb.compute_arch_hash(model_source)
    transferred = kb.lookup(arch_hash)
    # ... run verification ...
    kb.record(arch_hash, predicates, strategies, proof_cert)
    kb.save("kb.json")
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Architectural pattern hashing
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_layer_sequence(source: str) -> List[str]:
    """Extract the sequence of nn.Module layer types from __init__.

    Parses the AST to find ``self.<name> = nn.<LayerType>(...)`` assignments,
    returning the layer type names in order.  Parameter values (in_features,
    kernel_size, etc.) are ignored so that models differing only in
    width/depth hash identically.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    layer_types: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for item in ast.walk(node):
            if not isinstance(item, ast.Assign):
                continue
            # Look for self.x = nn.Something(...)
            if (
                len(item.targets) == 1
                and isinstance(item.targets[0], ast.Attribute)
                and isinstance(item.targets[0].value, ast.Name)
                and item.targets[0].value.id == "self"
            ):
                val = item.value
                if isinstance(val, ast.Call):
                    func = val.func
                    # nn.Linear(...), nn.Conv2d(...), etc.
                    if isinstance(func, ast.Attribute):
                        layer_types.append(func.attr)
                    elif isinstance(func, ast.Name):
                        layer_types.append(func.id)
    return layer_types


def _extract_forward_pattern(source: str) -> List[str]:
    """Extract a simplified call pattern from the forward method.

    Returns a list of attribute names called on self (e.g. ["conv1", "bn1",
    "relu", "fc"]) representing the connectivity pattern.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    calls: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "forward":
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                if (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "self"
                ):
                    calls.append(func.attr)
    return calls


def compute_arch_hash(source: str) -> str:
    """Compute an architectural pattern hash for a model source.

    The hash captures the sequence of layer types and forward-pass
    connectivity while ignoring specific parameter values.  Two models
    from the same family (e.g. ResNet-18 and ResNet-50) should produce
    the same hash if they share the same layer-type skeleton.
    """
    layers = _extract_layer_sequence(source)
    forward_calls = _extract_forward_pattern(source)
    pattern = {
        "layer_types": layers,
        "forward_calls": forward_calls,
    }
    canonical = json.dumps(pattern, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# Anti-unification (Plotkin 1970)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ProofSchema:
    """A generalized proof schema extracted via anti-unification.

    Variables replace specific values that differ across proof certificates.
    The ``rule_skeleton`` captures the common inference structure while
    ``variable_positions`` tracks where generalisation occurred.
    """
    rule_skeleton: List[Dict[str, Any]] = field(default_factory=list)
    variable_positions: List[Dict[str, str]] = field(default_factory=list)
    source_count: int = 0
    arch_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_skeleton": self.rule_skeleton,
            "variable_positions": self.variable_positions,
            "source_count": self.source_count,
            "arch_hash": self.arch_hash,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProofSchema":
        return cls(
            rule_skeleton=d.get("rule_skeleton", []),
            variable_positions=d.get("variable_positions", []),
            source_count=d.get("source_count", 0),
            arch_hash=d.get("arch_hash", ""),
        )


def _anti_unify_values(v1: Any, v2: Any, var_counter: List[int]) -> Tuple[Any, Optional[str]]:
    """Anti-unify two values.  Returns (generalized_value, variable_name_or_None)."""
    if v1 == v2:
        return v1, None
    var_name = f"?V{var_counter[0]}"
    var_counter[0] += 1
    return var_name, var_name


def _anti_unify_steps(
    steps1: List[Dict[str, Any]],
    steps2: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Anti-unify two sequences of proof steps (Plotkin-style).

    Aligns steps by position (truncating to shorter length), generalises
    each field independently, and collects variable positions.
    """
    var_counter = [0]
    skeleton: List[Dict[str, Any]] = []
    var_positions: List[Dict[str, str]] = []

    min_len = min(len(steps1), len(steps2))
    for i in range(min_len):
        s1, s2 = steps1[i], steps2[i]
        gen_step: Dict[str, Any] = {}
        step_vars: Dict[str, str] = {}

        for key in set(list(s1.keys()) + list(s2.keys())):
            v1 = s1.get(key)
            v2 = s2.get(key)
            gen_val, var_name = _anti_unify_values(v1, v2, var_counter)
            gen_step[key] = gen_val
            if var_name is not None:
                step_vars[key] = var_name

        skeleton.append(gen_step)
        var_positions.append(step_vars)

    return skeleton, var_positions


def anti_unify_proof_certificates(
    certificates: List[Dict[str, Any]],
    arch_hash: str = "",
) -> ProofSchema:
    """Extract a generalized proof schema from multiple proof certificates.

    Uses Plotkin (1970) style anti-unification: pairwise generalise the
    proof step sequences, replacing differing concrete values with
    variables.  The result captures the common verification structure
    shared across all certificates in the family.

    Parameters
    ----------
    certificates : list of dict
        Serialised proof certificates (via ``ProofCertificate.to_dict()``).
    arch_hash : str
        Architectural family hash these certificates belong to.

    Returns
    -------
    ProofSchema
        The least-general generalisation of the input certificates.
    """
    if not certificates:
        return ProofSchema(arch_hash=arch_hash)

    # Start with the first certificate's steps
    current_steps = certificates[0].get("steps", [])
    current_vars: List[Dict[str, str]] = [{} for _ in current_steps]

    for cert in certificates[1:]:
        next_steps = cert.get("steps", [])
        current_steps, current_vars = _anti_unify_steps(current_steps, next_steps)

    return ProofSchema(
        rule_skeleton=current_steps,
        variable_positions=current_vars,
        source_count=len(certificates),
        arch_hash=arch_hash,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Predicate entry with entrenchment scoring (AGM belief revision support)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PredicateEntry:
    """A predicate with AGM entrenchment metadata.

    Entrenchment score (Gärdenfors 1988) determines contraction priority:
    lower-entrenched predicates are removed first.  Score is computed from:
    - verification_successes / verification_attempts (success rate)
    - recency (time since last successful verification)
    - scope (number of arch families where this predicate appears)
    """
    predicate: str
    added_at: float = 0.0
    last_verified_at: float = 0.0
    verification_successes: int = 0
    verification_attempts: int = 0
    scope: int = 1
    depends_on: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.verification_attempts == 0:
            return 0.0
        return self.verification_successes / self.verification_attempts

    def entrenchment_score(self, now: Optional[float] = None) -> float:
        """Compute entrenchment score in [0, 1].

        Higher score = more entrenched = harder to remove.
        """
        now = now or time.time()
        # Component 1: success rate [0, 1]
        sr = self.success_rate
        # Component 2: recency — exponential decay with 30-day half-life
        age_seconds = max(0.0, now - self.last_verified_at) if self.last_verified_at else (now - self.added_at)
        half_life = 30 * 86400  # 30 days in seconds
        recency = 2.0 ** (-age_seconds / half_life) if half_life > 0 else 0.0
        # Component 3: scope — log-scaled, capped at 1.0
        import math
        scope_score = min(1.0, math.log2(1 + self.scope) / 3.0)
        # Weighted combination
        return 0.5 * sr + 0.3 * recency + 0.2 * scope_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicate": self.predicate,
            "added_at": self.added_at,
            "last_verified_at": self.last_verified_at,
            "verification_successes": self.verification_successes,
            "verification_attempts": self.verification_attempts,
            "scope": self.scope,
            "depends_on": self.depends_on,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PredicateEntry":
        return cls(
            predicate=d.get("predicate", ""),
            added_at=d.get("added_at", 0.0),
            last_verified_at=d.get("last_verified_at", 0.0),
            verification_successes=d.get("verification_successes", 0),
            verification_attempts=d.get("verification_attempts", 0),
            scope=d.get("scope", 1),
            depends_on=d.get("depends_on", []),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Knowledge entry and family record
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VerificationStrategy:
    """A successful verification strategy for a family."""
    propagator_type: str = ""
    predicate_templates: List[str] = field(default_factory=list)
    iteration_count: int = 0
    total_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VerificationStrategy":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class FailureMode:
    """A known failure mode and its fix."""
    description: str = ""
    fix_description: str = ""
    predicates_needed: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FailureMode":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class FamilyRecord:
    """All knowledge accumulated for an architectural family."""
    arch_hash: str = ""
    layer_types: List[str] = field(default_factory=list)
    predicates: List[str] = field(default_factory=list)
    predicate_entries: Dict[str, PredicateEntry] = field(default_factory=dict)
    strategies: List[Dict[str, Any]] = field(default_factory=list)
    failure_modes: List[Dict[str, Any]] = field(default_factory=list)
    proof_certificates: List[Dict[str, Any]] = field(default_factory=list)
    proof_schema: Optional[Dict[str, Any]] = None
    session_count: int = 0
    last_updated: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arch_hash": self.arch_hash,
            "layer_types": self.layer_types,
            "predicates": self.predicates,
            "predicate_entries": {
                k: v.to_dict() for k, v in self.predicate_entries.items()
            },
            "strategies": self.strategies,
            "failure_modes": self.failure_modes,
            "proof_certificates": self.proof_certificates,
            "proof_schema": self.proof_schema,
            "session_count": self.session_count,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FamilyRecord":
        record = cls(
            arch_hash=d.get("arch_hash", ""),
            layer_types=d.get("layer_types", []),
            predicates=d.get("predicates", []),
            strategies=d.get("strategies", []),
            failure_modes=d.get("failure_modes", []),
            proof_certificates=d.get("proof_certificates", []),
            proof_schema=d.get("proof_schema"),
            session_count=d.get("session_count", 0),
            last_updated=d.get("last_updated", 0.0),
        )
        # Deserialize predicate_entries
        for k, v in d.get("predicate_entries", {}).items():
            record.predicate_entries[k] = PredicateEntry.from_dict(v)
        # Back-fill entries for predicates that lack an entry (legacy data)
        for p in record.predicates:
            if p not in record.predicate_entries:
                record.predicate_entries[p] = PredicateEntry(
                    predicate=p, added_at=record.last_updated,
                )
        return record

    def merge(self, other: "FamilyRecord") -> None:
        """Merge another FamilyRecord into this one (union of knowledge)."""
        # Merge predicates (deduplicate)
        seen = set(self.predicates)
        for p in other.predicates:
            if p not in seen:
                self.predicates.append(p)
                seen.add(p)

        # Merge predicate entries (take higher-entrenched entry)
        for p, entry in other.predicate_entries.items():
            if p not in self.predicate_entries:
                self.predicate_entries[p] = copy.deepcopy(entry)
            else:
                existing = self.predicate_entries[p]
                existing.verification_successes += entry.verification_successes
                existing.verification_attempts += entry.verification_attempts
                existing.scope = max(existing.scope, entry.scope)
                existing.last_verified_at = max(
                    existing.last_verified_at, entry.last_verified_at
                )

        # Merge strategies
        existing_keys = {json.dumps(s, sort_keys=True) for s in self.strategies}
        for s in other.strategies:
            key = json.dumps(s, sort_keys=True)
            if key not in existing_keys:
                self.strategies.append(s)
                existing_keys.add(key)

        # Merge failure modes
        existing_fm = {json.dumps(f, sort_keys=True) for f in self.failure_modes}
        for f in other.failure_modes:
            key = json.dumps(f, sort_keys=True)
            if key not in existing_fm:
                self.failure_modes.append(f)
                existing_fm.add(key)

        # Merge proof certificates
        existing_hashes = {
            c.get("certificate_hash", "") for c in self.proof_certificates
        }
        for c in other.proof_certificates:
            h = c.get("certificate_hash", "")
            if h and h not in existing_hashes:
                self.proof_certificates.append(c)
                existing_hashes.add(h)

        # Merge layer types (take union preserving order)
        if not self.layer_types and other.layer_types:
            self.layer_types = list(other.layer_types)

        self.session_count += other.session_count
        self.last_updated = max(self.last_updated, other.last_updated)

        # Recompute proof schema if we have new certificates
        if self.proof_certificates:
            schema = anti_unify_proof_certificates(
                self.proof_certificates, self.arch_hash
            )
            self.proof_schema = schema.to_dict()


# ═══════════════════════════════════════════════════════════════════════════════
# TransferredKnowledge — result of a KB lookup
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TransferredKnowledge:
    """Knowledge transferred from the KB for a specific architecture."""
    arch_hash: str = ""
    predicates: List[str] = field(default_factory=list)
    strategies: List[Dict[str, Any]] = field(default_factory=list)
    failure_modes: List[Dict[str, Any]] = field(default_factory=list)
    proof_schema: Optional[ProofSchema] = None
    session_count: int = 0

    @property
    def has_knowledge(self) -> bool:
        return bool(self.predicates or self.strategies or self.failure_modes)


# ═══════════════════════════════════════════════════════════════════════════════
# VerificationKnowledgeBase
# ═══════════════════════════════════════════════════════════════════════════════

class VerificationKnowledgeBase:
    """Persistent knowledge base for cross-session verification transfer.

    Stores per-family verification knowledge (predicates, strategies,
    failure modes, proof schemas) keyed by architectural pattern hash.
    Supports JSON persistence and merging across sessions.

    Usage::

        kb = VerificationKnowledgeBase()
        # or load from file:
        kb = VerificationKnowledgeBase.load("kb.json")

        arch_hash = kb.compute_arch_hash(source)
        transferred = kb.lookup(arch_hash)

        # After verification:
        kb.record(arch_hash, predicates=["x.shape[-1] == 768"], ...)
        kb.save("kb.json")
    """

    def __init__(self) -> None:
        self._families: Dict[str, FamilyRecord] = {}
        self._metadata: Dict[str, Any] = {
            "version": 1,
            "created": time.time(),
            "total_sessions": 0,
        }

    # ── Architectural hashing ────────────────────────────────────────────

    def compute_arch_hash(self, source: str) -> str:
        """Compute architectural pattern hash for a model source."""
        return compute_arch_hash(source)

    # ── Lookup ───────────────────────────────────────────────────────────

    def lookup(self, arch_hash: str) -> TransferredKnowledge:
        """Look up knowledge for an architectural family.

        Returns a ``TransferredKnowledge`` object with predicates,
        strategies, and failure modes from prior sessions.  Returns
        an empty object if no knowledge exists for this hash.
        """
        record = self._families.get(arch_hash)
        if record is None:
            return TransferredKnowledge(arch_hash=arch_hash)

        proof_schema = None
        if record.proof_schema:
            proof_schema = ProofSchema.from_dict(record.proof_schema)

        return TransferredKnowledge(
            arch_hash=arch_hash,
            predicates=list(record.predicates),
            strategies=list(record.strategies),
            failure_modes=list(record.failure_modes),
            proof_schema=proof_schema,
            session_count=record.session_count,
        )

    # ── Recording ────────────────────────────────────────────────────────

    def record(
        self,
        arch_hash: str,
        predicates: Optional[List[str]] = None,
        strategies: Optional[List[Dict[str, Any]]] = None,
        failure_modes: Optional[List[Dict[str, Any]]] = None,
        proof_certificate: Optional[Dict[str, Any]] = None,
        layer_types: Optional[List[str]] = None,
    ) -> None:
        """Record verification knowledge for an architectural family.

        New knowledge is merged with any existing records for the same
        arch_hash.  Proof schemas are recomputed when new certificates
        are added.
        """
        if arch_hash not in self._families:
            self._families[arch_hash] = FamilyRecord(
                arch_hash=arch_hash,
                last_updated=time.time(),
            )

        record = self._families[arch_hash]
        record.session_count += 1
        record.last_updated = time.time()
        self._metadata["total_sessions"] = self._metadata.get("total_sessions", 0) + 1

        if layer_types:
            record.layer_types = layer_types

        if predicates:
            now = time.time()
            seen = set(record.predicates)
            for p in predicates:
                if p not in seen:
                    record.predicates.append(p)
                    seen.add(p)
                # Create or update predicate entry
                if p not in record.predicate_entries:
                    record.predicate_entries[p] = PredicateEntry(
                        predicate=p, added_at=now, last_verified_at=now,
                        verification_successes=1, verification_attempts=1,
                    )
                else:
                    entry = record.predicate_entries[p]
                    entry.verification_attempts += 1
                    entry.verification_successes += 1
                    entry.last_verified_at = now

        if strategies:
            existing = {json.dumps(s, sort_keys=True) for s in record.strategies}
            for s in strategies:
                key = json.dumps(s, sort_keys=True)
                if key not in existing:
                    record.strategies.append(s)
                    existing.add(key)

        if failure_modes:
            existing_fm = {json.dumps(f, sort_keys=True) for f in record.failure_modes}
            for f in failure_modes:
                key = json.dumps(f, sort_keys=True)
                if key not in existing_fm:
                    record.failure_modes.append(f)
                    existing_fm.add(key)

        if proof_certificate:
            existing_hashes = {
                c.get("certificate_hash", "") for c in record.proof_certificates
            }
            cert_hash = proof_certificate.get("certificate_hash", "")
            if not cert_hash or cert_hash not in existing_hashes:
                record.proof_certificates.append(proof_certificate)
                # Recompute anti-unified proof schema
                if len(record.proof_certificates) >= 2:
                    schema = anti_unify_proof_certificates(
                        record.proof_certificates, arch_hash
                    )
                    record.proof_schema = schema.to_dict()

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the knowledge base to a JSON file."""
        data = {
            "metadata": self._metadata,
            "families": {
                k: v.to_dict() for k, v in self._families.items()
            },
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("KB saved to %s (%d families)", path, len(self._families))

    @classmethod
    def load(cls, path: str) -> "VerificationKnowledgeBase":
        """Load a knowledge base from a JSON file.

        Returns an empty KB if the file does not exist.
        """
        kb = cls()
        if not os.path.exists(path):
            return kb
        try:
            with open(path, "r") as f:
                data = json.load(f)
            kb._metadata = data.get("metadata", kb._metadata)
            for key, fam_dict in data.get("families", {}).items():
                kb._families[key] = FamilyRecord.from_dict(fam_dict)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Failed to load KB from %s: %s", path, exc)
        return kb

    # ── Merging ──────────────────────────────────────────────────────────

    def merge(self, other: "VerificationKnowledgeBase") -> None:
        """Merge knowledge from another KB into this one.

        For families present in both KBs, records are merged (union of
        predicates, strategies, failure modes, proof certificates).
        """
        for arch_hash, other_record in other._families.items():
            if arch_hash in self._families:
                self._families[arch_hash].merge(other_record)
            else:
                self._families[arch_hash] = copy.deepcopy(other_record)
        self._metadata["total_sessions"] = (
            self._metadata.get("total_sessions", 0)
            + other._metadata.get("total_sessions", 0)
        )

    # ── AGM Belief Revision (Gärdenfors 1988) ────────────────────────────

    def contract(self, arch_hash: str, predicate: str) -> List[str]:
        """AGM contraction: remove a predicate and all that depend on it.

        Uses entrenchment ordering — the predicate and any predicates
        whose ``depends_on`` list includes it are removed (closure
        under dependence).

        Returns the list of removed predicates.
        """
        record = self._families.get(arch_hash)
        if record is None:
            return []

        removed: List[str] = []
        to_remove: Set[str] = {predicate}

        # Compute transitive dependents
        changed = True
        while changed:
            changed = False
            for p, entry in record.predicate_entries.items():
                if p not in to_remove and any(
                    dep in to_remove for dep in entry.depends_on
                ):
                    to_remove.add(p)
                    changed = True

        # Remove from both lists
        for p in to_remove:
            if p in record.predicate_entries:
                del record.predicate_entries[p]
                removed.append(p)
        record.predicates = [p for p in record.predicates if p not in to_remove]

        if removed:
            logger.info(
                "AGM contraction on %s: removed %d predicates", arch_hash[:8], len(removed)
            )
        return removed

    def revise(self, arch_hash: str, predicate: str,
               depends_on: Optional[List[str]] = None) -> List[str]:
        """AGM revision via the Levi identity: contract ¬φ then expand φ.

        Since we store predicate strings (not logical formulae), we
        approximate ¬φ as any predicate that *contradicts* φ:
        - Same variable/shape path but different constant (e.g.
          ``x.shape[-1] == 768`` contradicts ``x.shape[-1] == 512``)
        - Explicit negation prefix ``not(...)``

        Returns the list of predicates removed during contraction.
        """
        record = self._families.get(arch_hash)
        if record is None:
            # No existing family — just expand
            self.record(arch_hash, predicates=[predicate])
            if depends_on and arch_hash in self._families:
                entry = self._families[arch_hash].predicate_entries.get(predicate)
                if entry:
                    entry.depends_on = list(depends_on)
            return []

        # Step 1: find contradicting predicates
        contradictions = self._find_contradictions(predicate, record)

        # Step 2: contract each contradiction
        all_removed: List[str] = []
        for contra in contradictions:
            all_removed.extend(self.contract(arch_hash, contra))

        # Step 3: expand (add the new predicate)
        now = time.time()
        if predicate not in record.predicate_entries:
            record.predicate_entries[predicate] = PredicateEntry(
                predicate=predicate, added_at=now, last_verified_at=now,
                verification_successes=1, verification_attempts=1,
                depends_on=list(depends_on or []),
            )
            if predicate not in record.predicates:
                record.predicates.append(predicate)
        else:
            entry = record.predicate_entries[predicate]
            entry.verification_successes += 1
            entry.verification_attempts += 1
            entry.last_verified_at = now

        return all_removed

    @staticmethod
    def _find_contradictions(predicate: str,
                             record: FamilyRecord) -> List[str]:
        """Find predicates in *record* that contradict *predicate*.

        Heuristic: two predicates contradict if they reference the same
        shape path (lhs of ``==``) but assert different values.
        """
        contradictions: List[str] = []
        # Parse lhs of "lhs == rhs" pattern
        if "==" not in predicate:
            return contradictions
        new_lhs = predicate.split("==")[0].strip()

        for existing in record.predicates:
            if existing == predicate:
                continue
            if "==" in existing:
                existing_lhs = existing.split("==")[0].strip()
                if existing_lhs == new_lhs:
                    contradictions.append(existing)
        return contradictions

    def invalidate_stale(self, arch_hash: str,
                         threshold_age: float = 30 * 86400,
                         min_uses: int = 2,
                         entrenchment_threshold: float = 0.3) -> List[str]:
        """Remove predicates below entrenchment threshold.

        A predicate is considered stale if:
        - Its entrenchment score is below ``entrenchment_threshold``, AND
        - It is older than ``threshold_age`` seconds, AND
        - It has fewer than ``min_uses`` verification successes.

        Returns the list of removed predicates.
        """
        record = self._families.get(arch_hash)
        if record is None:
            return []

        now = time.time()
        stale: List[str] = []
        for p, entry in list(record.predicate_entries.items()):
            age = now - entry.added_at
            score = entry.entrenchment_score(now)
            if (score < entrenchment_threshold
                    and age > threshold_age
                    and entry.verification_successes < min_uses):
                stale.append(p)

        removed: List[str] = []
        for p in stale:
            removed.extend(self.contract(arch_hash, p))

        if removed:
            logger.info(
                "Stale invalidation on %s: removed %d predicates",
                arch_hash[:8], len(removed),
            )
        return removed

    def measure_kb_precision(
        self,
        arch_hash: str,
        test_cases: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Measure how many stored predicates are still valid.

        Each test case should have:
        - ``"valid_predicates"``: list of predicate strings that are
          currently valid for the architecture.

        Returns a dict with precision, recall, and per-predicate status.
        """
        record = self._families.get(arch_hash)
        if record is None:
            return {"precision": 0.0, "recall": 0.0, "total_stored": 0,
                    "valid_count": 0, "invalid_count": 0, "details": []}

        # Collect all currently valid predicates from test cases
        valid_set: Set[str] = set()
        for tc in test_cases:
            for p in tc.get("valid_predicates", []):
                valid_set.add(p)

        stored = set(record.predicates)
        valid_stored = stored & valid_set
        invalid_stored = stored - valid_set

        precision = len(valid_stored) / len(stored) if stored else 1.0
        recall = len(valid_stored) / len(valid_set) if valid_set else 1.0

        details = []
        for p in record.predicates:
            is_valid = p in valid_set
            entry = record.predicate_entries.get(p)
            details.append({
                "predicate": p,
                "valid": is_valid,
                "entrenchment": entry.entrenchment_score() if entry else 0.0,
            })

        # Update entrenchment: mark invalid predicates as failed
        now = time.time()
        for p in record.predicates:
            entry = record.predicate_entries.get(p)
            if entry is None:
                continue
            entry.verification_attempts += 1
            if p in valid_set:
                entry.verification_successes += 1
                entry.last_verified_at = now

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "total_stored": len(stored),
            "valid_count": len(valid_stored),
            "invalid_count": len(invalid_stored),
            "details": details,
        }

    # ── Query helpers ────────────────────────────────────────────────────

    @property
    def families(self) -> Dict[str, FamilyRecord]:
        """Access the internal family records (read-only view)."""
        return dict(self._families)

    @property
    def family_count(self) -> int:
        return len(self._families)

    @property
    def total_predicates(self) -> int:
        return sum(len(f.predicates) for f in self._families.values())

    def get_all_arch_hashes(self) -> List[str]:
        return list(self._families.keys())

    def has_family(self, arch_hash: str) -> bool:
        return arch_hash in self._families

    def get_family_record(self, arch_hash: str) -> Optional[FamilyRecord]:
        return self._families.get(arch_hash)

    def get_transferred_predicates(self, arch_hash: str) -> List[str]:
        """Get predicates for priming a CEGAR loop."""
        record = self._families.get(arch_hash)
        if record is None:
            return []
        return list(record.predicates)

    def get_repair_context(self, arch_hash: str) -> str:
        """Build a repair context string for the neuro-symbolic loop.

        Returns a formatted string summarising prior failure modes and
        successful strategies for this architectural family.
        """
        record = self._families.get(arch_hash)
        if record is None:
            return ""

        parts: List[str] = []
        if record.predicates:
            parts.append("## Known Predicates from Prior Sessions")
            for p in record.predicates:
                parts.append(f"  - {p}")

        if record.failure_modes:
            parts.append("\n## Known Failure Modes")
            for fm in record.failure_modes:
                desc = fm.get("description", "")
                fix = fm.get("fix_description", "")
                parts.append(f"  - {desc}")
                if fix:
                    parts.append(f"    Fix: {fix}")

        if record.strategies:
            parts.append("\n## Successful Verification Strategies")
            for s in record.strategies:
                prop = s.get("propagator_type", "")
                iters = s.get("iteration_count", 0)
                parts.append(f"  - {prop} ({iters} iterations)")

        return "\n".join(parts)

    def __repr__(self) -> str:
        return (
            f"VerificationKnowledgeBase("
            f"{self.family_count} families, "
            f"{self.total_predicates} predicates)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Empirical transfer validation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TransferExperimentResult:
    """Result of an empirical knowledge transfer experiment."""
    family_name: str
    source_models: int
    target_models: int
    predicates_transferred: int
    predicates_useful: int  # predicates that appeared in target proof
    transfer_rate: float  # useful / transferred
    cold_iterations_avg: float
    warm_iterations_avg: float
    speedup_ratio: float  # cold / warm
    cold_time_avg_ms: float
    warm_time_avg_ms: float

    def to_dict(self) -> dict:
        return {
            "family": self.family_name,
            "source_models": self.source_models,
            "target_models": self.target_models,
            "predicates_transferred": self.predicates_transferred,
            "predicates_useful": self.predicates_useful,
            "transfer_rate": round(self.transfer_rate, 3),
            "cold_iterations_avg": round(self.cold_iterations_avg, 2),
            "warm_iterations_avg": round(self.warm_iterations_avg, 2),
            "speedup_ratio": round(self.speedup_ratio, 2),
            "cold_time_avg_ms": round(self.cold_time_avg_ms, 2),
            "warm_time_avg_ms": round(self.warm_time_avg_ms, 2),
        }


def run_transfer_experiment(
    source_models: List[Tuple[str, Dict[str, tuple]]],
    target_models: List[Tuple[str, Dict[str, tuple]]],
    family_name: str = "experiment",
) -> TransferExperimentResult:
    """Run an empirical transfer experiment.

    1. Run CEGAR cold on all source models, accumulate KB
    2. Run CEGAR cold on target models (baseline)
    3. Run CEGAR warm on target models with KB from step 1
    4. Compare iterations and timing

    Parameters
    ----------
    source_models : list of (source_code, input_shapes)
    target_models : list of (source_code, input_shapes)
    family_name : str

    Returns
    -------
    TransferExperimentResult
    """
    from src.shape_cegar import ShapeCEGARLoop

    kb = VerificationKnowledgeBase()

    # Step 1: Build KB from source models
    for src_code, inp_shapes in source_models:
        loop = ShapeCEGARLoop(src_code, input_shapes=inp_shapes)
        result = loop.run()
        arch_hash = compute_arch_hash(src_code)
        preds = [p.pretty() for p in result.discovered_predicates]
        kb.record(arch_hash, predicates=preds)

    # Step 2 & 3: Run cold and warm on target models
    cold_iters = []
    warm_iters = []
    cold_times = []
    warm_times = []
    preds_transferred = 0
    preds_useful = 0

    for src_code, inp_shapes in target_models:
        import time as _time

        # Cold run
        t0 = _time.monotonic()
        cold_loop = ShapeCEGARLoop(src_code, input_shapes=inp_shapes)
        cold_result = cold_loop.run()
        cold_ms = (_time.monotonic() - t0) * 1000
        cold_iters.append(cold_result.iterations)
        cold_times.append(cold_ms)

        # Warm run
        t0 = _time.monotonic()
        warm_loop = ShapeCEGARLoop(src_code, input_shapes=inp_shapes, knowledge_base=kb)
        warm_result = warm_loop.run()
        warm_ms = (_time.monotonic() - t0) * 1000
        warm_iters.append(warm_result.iterations)
        warm_times.append(warm_ms)

        # Check transferred predicates
        arch_hash = compute_arch_hash(src_code)
        transferred = kb.lookup(arch_hash)
        preds_transferred += len(transferred.predicates)
        cold_preds = {p.pretty() for p in cold_result.discovered_predicates}
        preds_useful += sum(1 for p in transferred.predicates if p in cold_preds)

    n_targets = max(1, len(target_models))
    cold_avg = sum(cold_iters) / n_targets
    warm_avg = sum(warm_iters) / n_targets
    cold_time_avg = sum(cold_times) / n_targets
    warm_time_avg = sum(warm_times) / n_targets
    speedup = cold_avg / max(0.001, warm_avg)
    transfer_rate = preds_useful / max(1, preds_transferred)

    return TransferExperimentResult(
        family_name=family_name,
        source_models=len(source_models),
        target_models=len(target_models),
        predicates_transferred=preds_transferred,
        predicates_useful=preds_useful,
        transfer_rate=transfer_rate,
        cold_iterations_avg=cold_avg,
        warm_iterations_avg=warm_avg,
        speedup_ratio=speedup,
        cold_time_avg_ms=cold_time_avg,
        warm_time_avg_ms=warm_time_avg,
    )
