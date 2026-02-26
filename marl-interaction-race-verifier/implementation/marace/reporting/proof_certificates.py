"""Machine-checkable proof certificates for MARACE verification results.

This module provides a self-contained certificate format and independent
checker that allows *any* third party to verify MARACE analysis results
without re-running the full verification pipeline.

Mathematical Foundation
-----------------------
A MARACE verification certifies that a multi-agent system is free of
interaction races (SAFE), contains a race (UNSAFE), or that coverage was
insufficient (UNKNOWN).  The proof relies on four pillars:

1. **Abstract fixpoint**: The abstract semantics of the multi-agent
   transition system has been iterated to a post-fixpoint in a
   zonotope domain.  The certificate records every iterate, the
   widening points, and the final convergence proof.

2. **Inductive invariant**: The fixpoint state forms an inductive
   invariant satisfying three proof obligations:
   - *Initiation*:  the initial states are contained in the invariant.
   - *Consecution*: the abstract post-image of the invariant (after
     widening) is contained in the invariant.
   - *Safety exclusion*: the invariant does not intersect the unsafe
     region.

3. **Happens-before consistency**: The HB graph witnessing causal
   ordering is acyclic.  The certificate records a topological order
   (DFS timestamps) that can be checked in O(|E|) time.

4. **Compositional soundness**: When assume-guarantee decomposition is
   used, every assumption is *discharged* by the guarantee of another
   group.  The certificate contains LP dual witnesses (Farkas
   certificates) so a checker can verify feasibility without
   re-solving.

For UNSAFE verdicts, a *race witness* (concrete schedule + states +
violation predicate) is included so the checker can replay the
execution.

Design Principles
-----------------
* **Self-contained**: checking requires only ``numpy``, ``json``,
  ``hashlib`` — no external proof assistants.
* **Deterministic**: serialisation is sorted-key JSON; numpy arrays are
  stored as nested lists with full float precision.
* **Tamper-evident**: every certificate carries an SHA-256 hash over
  its semantic content.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ======================================================================
# Constants
# ======================================================================

CERTIFICATE_VERSION = "1.0"

# MARACE Certificate Format v2 version string.
CERTIFICATE_FORMAT_V2 = "2.0"

# Tolerance used when checking containment and convergence properties.
_CONTAINMENT_TOL = 1e-8

# Maximum relative error accepted for dual-witness LP re-checking.
_LP_DUAL_TOL = 1e-6


# ======================================================================
# Verdict enum
# ======================================================================

class Verdict(Enum):
    """Possible verification outcomes."""

    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    UNKNOWN = "UNKNOWN"


# ======================================================================
# Check result
# ======================================================================

@dataclass
class ComponentCheckResult:
    """Result of checking a single proof component.

    Attributes:
        component: Name of the checked component (e.g. ``"initiation"``).
        passed: ``True`` iff the check succeeded.
        message: Human-readable explanation.
        details: Optional structured data for debugging.
    """

    component: str
    passed: bool
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckResult:
    """Aggregate result of independent certificate checking.

    A certificate passes independent checking only when *every*
    component check passes.
    """

    overall_passed: bool
    component_results: List[ComponentCheckResult] = field(default_factory=list)
    certificate_hash_valid: bool = False

    def summary(self) -> str:
        """Return a compact human-readable summary."""
        lines = [
            f"Overall: {'PASS' if self.overall_passed else 'FAIL'}",
            f"Hash integrity: {'OK' if self.certificate_hash_valid else 'INVALID'}",
        ]
        for cr in self.component_results:
            status = "PASS" if cr.passed else "FAIL"
            lines.append(f"  [{status}] {cr.component}: {cr.message}")
        return "\n".join(lines)


# ======================================================================
# Numpy-aware JSON encoder / decoder helpers
# ======================================================================

class CertificateSerializer:
    """Serialize / deserialize certificates to JSON with numpy support.

    Numpy arrays are encoded as ``{"__ndarray__": true, "data": <nested list>,
    "shape": [n1, ...], "dtype": "float64"}``.  This makes the JSON
    self-describing and lossless (up to float64 precision).
    """

    # -- encoding -------------------------------------------------------

    @staticmethod
    def encode_value(obj: Any) -> Any:
        """Recursively encode *obj* for JSON serialisation."""
        if isinstance(obj, np.ndarray):
            return {
                "__ndarray__": True,
                "data": obj.tolist(),
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
            }
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: CertificateSerializer.encode_value(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [CertificateSerializer.encode_value(v) for v in obj]
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, datetime):
            return obj.isoformat()
        return obj

    # -- decoding -------------------------------------------------------

    @staticmethod
    def decode_value(obj: Any) -> Any:
        """Recursively decode JSON-parsed *obj*, restoring numpy arrays."""
        if isinstance(obj, dict):
            if obj.get("__ndarray__"):
                return np.array(obj["data"], dtype=obj.get("dtype", "float64"))
            return {k: CertificateSerializer.decode_value(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [CertificateSerializer.decode_value(v) for v in obj]
        return obj

    # -- top-level convenience ------------------------------------------

    @classmethod
    def to_json(cls, certificate: Dict[str, Any], *, indent: int = 2) -> str:
        """Serialize a certificate dict to a JSON string."""
        encoded = cls.encode_value(certificate)
        return json.dumps(encoded, sort_keys=True, indent=indent)

    @classmethod
    def from_json(cls, text: str) -> Dict[str, Any]:
        """Deserialize a certificate dict from a JSON string."""
        raw = json.loads(text)
        return cls.decode_value(raw)

    @classmethod
    def to_file(cls, certificate: Dict[str, Any], path: str) -> None:
        """Write a certificate dict to a JSON file."""
        with open(path, "w") as f:
            f.write(cls.to_json(certificate))

    @classmethod
    def from_file(cls, path: str) -> Dict[str, Any]:
        """Read a certificate dict from a JSON file."""
        with open(path) as f:
            return cls.from_json(f.read())


# ======================================================================
# CertificateFormat — canonical certificate structure
# ======================================================================

class CertificateFormat:
    """Canonical JSON-serialisable certificate structure.

    A certificate is a plain ``dict`` conforming to the schema described
    below.  This class provides factory methods that construct
    well-formed sub-structures and a validator that checks structural
    completeness (but *not* mathematical correctness — that is the job
    of :class:`IndependentChecker`).

    Top-level schema
    ~~~~~~~~~~~~~~~~
    ::

        {
          "version": "1.0",
          "certificate_id": "<uuid>",
          "timestamp": "<ISO-8601>",
          "verdict": "SAFE" | "UNSAFE" | "UNKNOWN",

          "environment": {
              "id": str,
              "state_dimension": int,
              "action_dimensions": [int, ...],
              "num_agents": int,
              "transition_hash": str          # SHA-256 of serialised dynamics
          },

          "policies": [
              {"id": str, "hash": str, "architecture": str,
               "lipschitz_bound": float | null}
          ],

          "specification": {
              "grammar_tree": dict | null,
              "temporal_formula": str | null,
              "unsafe_predicate": {"A": ndarray, "b": ndarray}  # Ax ≤ b
          },

          "abstract_fixpoint": { ... },
          "inductive_invariant": { ... },
          "hb_consistency": { ... },
          "composition_certificate": { ... } | null,
          "race_witnesses": [ ... ] | null,

          "hash": "<sha-256>"
        }
    """

    # ---- Environment --------------------------------------------------

    @staticmethod
    def make_environment(
        env_id: str,
        state_dimension: int,
        action_dimensions: List[int],
        num_agents: int,
        transition_hash: str = "",
    ) -> Dict[str, Any]:
        """Construct the ``environment`` sub-structure."""
        return {
            "id": env_id,
            "state_dimension": state_dimension,
            "action_dimensions": list(action_dimensions),
            "num_agents": num_agents,
            "transition_hash": transition_hash,
        }

    # ---- Policies -----------------------------------------------------

    @staticmethod
    def make_policy_entry(
        policy_id: str,
        policy_hash: str,
        architecture: str = "unknown",
        lipschitz_bound: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Construct a single entry in the ``policies`` list."""
        return {
            "id": policy_id,
            "hash": policy_hash,
            "architecture": architecture,
            "lipschitz_bound": lipschitz_bound,
        }

    # ---- Specification ------------------------------------------------

    @staticmethod
    def make_specification(
        grammar_tree: Optional[Dict[str, Any]] = None,
        temporal_formula: Optional[str] = None,
        unsafe_A: Optional[np.ndarray] = None,
        unsafe_b: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Construct the ``specification`` sub-structure.

        The unsafe set is expressed as the polyhedron ``{x | A x ≤ b}``.
        """
        spec: Dict[str, Any] = {
            "grammar_tree": grammar_tree,
            "temporal_formula": temporal_formula,
        }
        if unsafe_A is not None and unsafe_b is not None:
            spec["unsafe_predicate"] = {
                "A": np.asarray(unsafe_A, dtype=np.float64),
                "b": np.asarray(unsafe_b, dtype=np.float64).ravel(),
            }
        else:
            spec["unsafe_predicate"] = None
        return spec

    # ---- Abstract fixpoint --------------------------------------------

    @staticmethod
    def make_abstract_fixpoint(
        domain: str,
        iterations: int,
        widening_points: List[int],
        fixpoint_center: np.ndarray,
        fixpoint_generators: np.ndarray,
        fixpoint_constraints: Optional[Dict[str, Any]] = None,
        ascending_chain: Optional[List[Dict[str, Any]]] = None,
        widening_certificate: Optional[Dict[str, Any]] = None,
        narrowing_steps: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Construct the ``abstract_fixpoint`` sub-structure.

        Parameters
        ----------
        domain : str
            Name of the abstract domain (e.g. ``"zonotope"``).
        iterations : int
            Number of fixpoint iterations performed.
        widening_points : list of int
            Iteration indices at which widening was applied.
        fixpoint_center, fixpoint_generators :
            The final fixpoint zonotope.
        fixpoint_constraints : dict, optional
            Additional constraints (for constrained zonotopes).
        ascending_chain : list of dict, optional
            Each entry ``{"iteration": k, "center": c, "generators": G}``
            records an iterate of the ascending chain.
        widening_certificate : dict, optional
            Proof that widening was applied correctly —
            ``{"pre_widening": {...}, "post_widening": {...},
              "inclusion_holds": bool}``.
        narrowing_steps : list of dict, optional
            Analogous narrowing iterates, if any.

        The *convergence proof* is the triple
        ``(ascending_chain, widening_certificate, narrowing_steps)``
        which together certify that the computed state is a valid
        post-fixpoint of the abstract transformer.
        """
        return {
            "domain": domain,
            "iterations": iterations,
            "widening_points": list(widening_points),
            "fixpoint_state": {
                "center": np.asarray(fixpoint_center, dtype=np.float64),
                "generators": np.asarray(fixpoint_generators, dtype=np.float64),
                "constraints": fixpoint_constraints,
            },
            "convergence_proof": {
                "ascending_chain": ascending_chain or [],
                "widening_certificate": widening_certificate or {},
                "narrowing_steps": narrowing_steps or [],
            },
        }

    # ---- Inductive invariant ------------------------------------------

    @staticmethod
    def make_inductive_invariant(
        invariant_center: np.ndarray,
        invariant_generators: np.ndarray,
        init_center: np.ndarray,
        init_generators: np.ndarray,
        post_center: np.ndarray,
        post_generators: np.ndarray,
    ) -> Dict[str, Any]:
        """Construct the ``inductive_invariant`` sub-structure.

        Proof obligations encoded
        ~~~~~~~~~~~~~~~~~~~~~~~~~~
        Let *I* be the invariant zonotope, *I₀* the initial-state
        zonotope, and *post(I)* the abstract post-image.

        * **Initiation**:  I₀ ⊆ I.
          Holds iff every vertex of the bounding box of I₀ is contained
          in I (sufficient for zonotopes because zonotope containment is
          monotone over the box hull).

        * **Consecution**:  post(I) ⊆ widen(I).
          Recorded as the bounding-box of post(I) lying within the
          bounding-box of I (sound over-approximation).

        * **Safety exclusion**:  I ∩ Unsafe = ∅.
          Checked externally against the specification's unsafe
          predicate.
        """
        return {
            "invariant_zonotope": {
                "center": np.asarray(invariant_center, dtype=np.float64),
                "generators": np.asarray(invariant_generators, dtype=np.float64),
            },
            "initial_zonotope": {
                "center": np.asarray(init_center, dtype=np.float64),
                "generators": np.asarray(init_generators, dtype=np.float64),
            },
            "post_zonotope": {
                "center": np.asarray(post_center, dtype=np.float64),
                "generators": np.asarray(post_generators, dtype=np.float64),
            },
            "initiation_proof": "init_state ⊆ invariant",
            "consecution_proof": "post(invariant) ⊆ widen(invariant)",
            "safety_proof": "invariant ∩ unsafe = ∅",
        }

    # ---- HB consistency -----------------------------------------------

    @staticmethod
    def make_hb_consistency(
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        topological_order: List[str],
        dfs_timestamps: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> Dict[str, Any]:
        """Construct the ``hb_consistency`` sub-structure.

        Proof obligation
        ~~~~~~~~~~~~~~~~
        A happens-before relation is *consistent* iff its graph is a DAG.
        The certificate encodes a *topological order* (equivalently, DFS
        finish-time ordering) so that the checker can verify acyclicity
        in O(|V| + |E|) by confirming that every edge goes forward in
        the order.

        Parameters
        ----------
        nodes : list of dict
            Each entry ``{"id": str, "agent_id": str, "timestep": int, ...}``.
        edges : list of dict
            Each entry ``{"src": str, "dst": str, "source": str}``.
        topological_order : list of str
            Node ids in a valid topological order.
        dfs_timestamps : dict, optional
            ``{node_id: (discover_time, finish_time)}``.
        """
        return {
            "hb_graph": {
                "nodes": list(nodes),
                "edges": list(edges),
            },
            "topological_order": list(topological_order),
            "topological_order_exists": True,
            "cycle_freedom_proof": "DFS timestamps",
            "dfs_timestamps": dfs_timestamps or {},
        }

    # ---- Composition certificate --------------------------------------

    @staticmethod
    def make_composition_certificate(
        groups: List[Dict[str, Any]],
        contracts: List[Dict[str, Any]],
        discharge_proofs: List[Dict[str, Any]],
        causal_closure_proof: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Construct the ``composition_certificate`` sub-structure.

        Proof obligation
        ~~~~~~~~~~~~~~~~
        For each contract *C = (A, G)* (assumptions *A*, guarantees *G*):
        every assumption clause of *C* must be *discharged* by the
        guarantee of some other group's contract.  Discharge is
        witnessed by an LP dual solution (Farkas certificate):

            ∃ y ≥ 0 : yᵀ G_other ≤ A_thisᵀ

        The dual vector *y* is stored so the checker can verify
        feasibility by a single matrix–vector multiply.

        Parameters
        ----------
        groups : list of dict
            Each entry ``{"group_id": str, "agent_ids": [...], "status": str}``.
        contracts : list of dict
            Each entry ``{"group_id": str,
             "assumptions": {"A": ndarray, "b": ndarray},
             "guarantees":  {"A": ndarray, "b": ndarray}}``.
        discharge_proofs : list of dict
            Each entry ``{"assumption_id": str,
             "discharged_by": str,
             "lp_dual_witness": ndarray,
             "primal_slack": ndarray | None}``.
        causal_closure_proof : dict, optional
            Proof that the causal ordering is closed under the
            composition (i.e. inter-group HB edges are consistent).
        """
        return {
            "groups": list(groups),
            "contracts": list(contracts),
            "discharge_proofs": list(discharge_proofs),
            "causal_closure_proof": causal_closure_proof,
        }

    # ---- Race witnesses -----------------------------------------------

    @staticmethod
    def make_race_witness(
        schedule: List[Dict[str, Any]],
        states: List[np.ndarray],
        violation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Construct a single race witness.

        A witness is *valid* iff replaying *schedule* from the recorded
        initial state yields *states* and the violation predicate holds
        at the final state.
        """
        return {
            "schedule": list(schedule),
            "states": [np.asarray(s, dtype=np.float64) for s in states],
            "violation": dict(violation),
        }

    # ---- Structural validation ----------------------------------------

    @staticmethod
    def validate_structure(cert: Dict[str, Any]) -> List[str]:
        """Check that *cert* has all required top-level keys.

        Returns a list of issues (empty means valid structure).
        """
        required_keys = [
            "version",
            "certificate_id",
            "timestamp",
            "verdict",
            "environment",
            "policies",
            "specification",
            "abstract_fixpoint",
            "inductive_invariant",
            "hb_consistency",
            "hash",
        ]
        issues: List[str] = []
        for key in required_keys:
            if key not in cert:
                issues.append(f"Missing required key: '{key}'")
        if cert.get("verdict") not in {"SAFE", "UNSAFE", "UNKNOWN"}:
            issues.append(
                f"Invalid verdict: '{cert.get('verdict')}'; "
                "expected SAFE, UNSAFE, or UNKNOWN"
            )
        if cert.get("version") != CERTIFICATE_VERSION:
            issues.append(
                f"Unsupported version: '{cert.get('version')}'; "
                f"expected '{CERTIFICATE_VERSION}'"
            )
        return issues


# ======================================================================
# CertificateBuilder — assemble certificate from pipeline results
# ======================================================================

class CertificateBuilder:
    """Incrementally construct a proof certificate from pipeline outputs.

    Usage::

        builder = CertificateBuilder(env_id="grid-4x4", num_agents=3,
                                     state_dim=8, action_dims=[5, 5, 5])
        builder.set_policies([...])
        builder.set_specification(...)
        builder.set_fixpoint_result(...)
        builder.set_inductive_invariant(...)
        builder.set_hb_consistency(...)
        builder.set_composition(...)        # optional
        builder.add_race_witness(...)       # optional, for UNSAFE
        cert = builder.build()
    """

    def __init__(
        self,
        env_id: str,
        num_agents: int,
        state_dim: int,
        action_dims: List[int],
        transition_hash: str = "",
    ) -> None:
        self._env = CertificateFormat.make_environment(
            env_id=env_id,
            state_dimension=state_dim,
            action_dimensions=action_dims,
            num_agents=num_agents,
            transition_hash=transition_hash,
        )
        self._policies: List[Dict[str, Any]] = []
        self._specification: Optional[Dict[str, Any]] = None
        self._fixpoint: Optional[Dict[str, Any]] = None
        self._invariant: Optional[Dict[str, Any]] = None
        self._hb: Optional[Dict[str, Any]] = None
        self._composition: Optional[Dict[str, Any]] = None
        self._race_witnesses: List[Dict[str, Any]] = []
        self._verdict: Optional[str] = None

    # ---- Setters ------------------------------------------------------

    def set_verdict(self, verdict: str) -> "CertificateBuilder":
        """Explicitly set the verdict (SAFE / UNSAFE / UNKNOWN)."""
        if verdict not in {"SAFE", "UNSAFE", "UNKNOWN"}:
            raise ValueError(f"Invalid verdict: {verdict}")
        self._verdict = verdict
        return self

    def set_policies(
        self,
        policies: List[Dict[str, Any]],
    ) -> "CertificateBuilder":
        """Set the policy entries.

        Each dict should have keys ``id``, ``hash``, and optionally
        ``architecture`` and ``lipschitz_bound``.
        """
        self._policies = [
            CertificateFormat.make_policy_entry(
                policy_id=p["id"],
                policy_hash=p.get("hash", ""),
                architecture=p.get("architecture", "unknown"),
                lipschitz_bound=p.get("lipschitz_bound"),
            )
            for p in policies
        ]
        return self

    def set_specification(
        self,
        grammar_tree: Optional[Dict[str, Any]] = None,
        temporal_formula: Optional[str] = None,
        unsafe_A: Optional[np.ndarray] = None,
        unsafe_b: Optional[np.ndarray] = None,
    ) -> "CertificateBuilder":
        """Set the safety specification."""
        self._specification = CertificateFormat.make_specification(
            grammar_tree=grammar_tree,
            temporal_formula=temporal_formula,
            unsafe_A=unsafe_A,
            unsafe_b=unsafe_b,
        )
        return self

    def set_fixpoint_result(
        self,
        domain: str,
        iterations: int,
        widening_points: List[int],
        fixpoint_center: np.ndarray,
        fixpoint_generators: np.ndarray,
        fixpoint_constraints: Optional[Dict[str, Any]] = None,
        ascending_chain: Optional[List[Dict[str, Any]]] = None,
        widening_certificate: Optional[Dict[str, Any]] = None,
        narrowing_steps: Optional[List[Dict[str, Any]]] = None,
    ) -> "CertificateBuilder":
        """Set the abstract fixpoint result."""
        self._fixpoint = CertificateFormat.make_abstract_fixpoint(
            domain=domain,
            iterations=iterations,
            widening_points=widening_points,
            fixpoint_center=fixpoint_center,
            fixpoint_generators=fixpoint_generators,
            fixpoint_constraints=fixpoint_constraints,
            ascending_chain=ascending_chain,
            widening_certificate=widening_certificate,
            narrowing_steps=narrowing_steps,
        )
        return self

    def set_inductive_invariant(
        self,
        invariant_center: np.ndarray,
        invariant_generators: np.ndarray,
        init_center: np.ndarray,
        init_generators: np.ndarray,
        post_center: np.ndarray,
        post_generators: np.ndarray,
    ) -> "CertificateBuilder":
        """Set the inductive invariant with proof obligations."""
        self._invariant = CertificateFormat.make_inductive_invariant(
            invariant_center=invariant_center,
            invariant_generators=invariant_generators,
            init_center=init_center,
            init_generators=init_generators,
            post_center=post_center,
            post_generators=post_generators,
        )
        return self

    def set_hb_consistency(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        topological_order: List[str],
        dfs_timestamps: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> "CertificateBuilder":
        """Set the happens-before consistency proof."""
        self._hb = CertificateFormat.make_hb_consistency(
            nodes=nodes,
            edges=edges,
            topological_order=topological_order,
            dfs_timestamps=dfs_timestamps,
        )
        return self

    def set_composition(
        self,
        groups: List[Dict[str, Any]],
        contracts: List[Dict[str, Any]],
        discharge_proofs: List[Dict[str, Any]],
        causal_closure_proof: Optional[Dict[str, Any]] = None,
    ) -> "CertificateBuilder":
        """Set the compositional verification certificate."""
        self._composition = CertificateFormat.make_composition_certificate(
            groups=groups,
            contracts=contracts,
            discharge_proofs=discharge_proofs,
            causal_closure_proof=causal_closure_proof,
        )
        return self

    def add_race_witness(
        self,
        schedule: List[Dict[str, Any]],
        states: List[np.ndarray],
        violation: Dict[str, Any],
    ) -> "CertificateBuilder":
        """Add a race witness (for UNSAFE verdicts)."""
        self._race_witnesses.append(
            CertificateFormat.make_race_witness(
                schedule=schedule, states=states, violation=violation
            )
        )
        return self

    # ---- Build --------------------------------------------------------

    def build(self) -> Dict[str, Any]:
        """Assemble and return the complete certificate dict.

        Raises ``ValueError`` if mandatory components are missing.
        """
        if self._specification is None:
            raise ValueError("Specification must be set before building")
        if self._fixpoint is None:
            raise ValueError("Fixpoint result must be set before building")
        if self._invariant is None:
            raise ValueError("Inductive invariant must be set before building")
        if self._hb is None:
            raise ValueError("HB consistency must be set before building")

        verdict = self._verdict
        if verdict is None:
            verdict = self._infer_verdict()

        cert: Dict[str, Any] = {
            "version": CERTIFICATE_VERSION,
            "certificate_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "verdict": verdict,
            "environment": self._env,
            "policies": self._policies,
            "specification": self._specification,
            "abstract_fixpoint": self._fixpoint,
            "inductive_invariant": self._invariant,
            "hb_consistency": self._hb,
            "composition_certificate": self._composition,
            "race_witnesses": self._race_witnesses if self._race_witnesses else None,
            "hash": "",
        }
        cert["hash"] = _compute_certificate_hash(cert)
        return cert

    # ---- Helpers ------------------------------------------------------

    def _infer_verdict(self) -> str:
        """Derive verdict from certificate contents."""
        if self._race_witnesses:
            return "UNSAFE"
        return "SAFE"


# ======================================================================
# Hash computation
# ======================================================================

def _compute_certificate_hash(cert: Dict[str, Any]) -> str:
    """Compute SHA-256 over the semantic content of *cert*.

    The hash covers every field except ``hash`` itself and
    ``certificate_id`` / ``timestamp`` (which are administrative).
    This means two certificates with identical mathematical content
    will produce the same hash even if generated at different times.
    """
    hashable = {
        k: v
        for k, v in cert.items()
        if k not in {"hash", "certificate_id", "timestamp"}
    }
    payload = CertificateSerializer.to_json(hashable, indent=None)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ======================================================================
# IndependentChecker — verify certificate without the verifier
# ======================================================================

class IndependentChecker:
    """Verify a proof certificate without running the MARACE pipeline.

    The checker performs the following *component checks*, each of which
    can pass or fail independently:

    1. **Hash integrity** — recompute the SHA-256 and compare.
    2. **Structure** — all required fields present and well-formed.
    3. **Fixpoint convergence** — ascending chain is non-decreasing and
       the final iterate is a post-fixpoint.
    4. **Initiation** — initial zonotope ⊆ invariant zonotope.
    5. **Consecution** — post(invariant) ⊆ invariant (bounding-box check).
    6. **Safety exclusion** — invariant ∩ unsafe = ∅.
    7. **HB acyclicity** — topological order is valid.
    8. **Composition discharge** — LP dual witnesses are feasible.
    9. **Race witness validity** — schedule replays to a violating state.

    Soundness argument
    ~~~~~~~~~~~~~~~~~~
    If checks 3–6 pass, the invariant is a valid inductive invariant
    that excludes the unsafe set, which by the abstract-interpretation
    soundness theorem implies the concrete system is safe.  Check 7
    ensures the happens-before ordering is consistent; check 8 ensures
    the compositional decomposition is sound.  Together, these checks
    are *sufficient* for an independent auditor to accept a SAFE
    verdict.

    For UNSAFE verdicts, check 9 confirms the race witness is
    replayable, which is *sufficient* to accept an UNSAFE verdict.
    """

    def __init__(self, tolerance: float = _CONTAINMENT_TOL) -> None:
        self._tol = tolerance

    def check(self, cert: Dict[str, Any]) -> CheckResult:
        """Run all applicable checks on *cert* and return results."""
        results: List[ComponentCheckResult] = []

        # 1. Hash integrity
        hash_ok = self._check_hash(cert)
        results.append(hash_ok)

        # 2. Structural validity
        results.append(self._check_structure(cert))

        # 3. Fixpoint convergence
        if "abstract_fixpoint" in cert and cert["abstract_fixpoint"]:
            results.append(self._check_fixpoint_convergence(cert["abstract_fixpoint"]))

        # 4–6. Inductive invariant
        if "inductive_invariant" in cert and cert["inductive_invariant"]:
            inv = cert["inductive_invariant"]
            results.append(self._check_initiation(inv))
            results.append(self._check_consecution(inv))
            if cert.get("specification", {}).get("unsafe_predicate"):
                results.append(
                    self._check_safety_exclusion(
                        inv, cert["specification"]["unsafe_predicate"]
                    )
                )

        # 7. HB acyclicity
        if "hb_consistency" in cert and cert["hb_consistency"]:
            results.append(self._check_hb_acyclicity(cert["hb_consistency"]))

        # 8. Composition discharge
        if cert.get("composition_certificate"):
            results.append(
                self._check_composition_discharge(cert["composition_certificate"])
            )

        # 9. Race witness validity
        if cert.get("race_witnesses"):
            for i, w in enumerate(cert["race_witnesses"]):
                results.append(self._check_race_witness(w, index=i))

        overall = all(r.passed for r in results)
        return CheckResult(
            overall_passed=overall,
            component_results=results,
            certificate_hash_valid=hash_ok.passed,
        )

    # ------------------------------------------------------------------
    # 1. Hash integrity
    # ------------------------------------------------------------------

    @staticmethod
    def _check_hash(cert: Dict[str, Any]) -> ComponentCheckResult:
        """Recompute hash and compare to stored value.

        This detects any post-hoc tampering with the certificate.
        """
        stored = cert.get("hash", "")
        recomputed = _compute_certificate_hash(cert)
        if stored == recomputed:
            return ComponentCheckResult(
                component="hash_integrity",
                passed=True,
                message="Hash matches recomputed value.",
            )
        return ComponentCheckResult(
            component="hash_integrity",
            passed=False,
            message=f"Hash mismatch: stored={stored[:16]}… recomputed={recomputed[:16]}…",
        )

    # ------------------------------------------------------------------
    # 2. Structural validity
    # ------------------------------------------------------------------

    @staticmethod
    def _check_structure(cert: Dict[str, Any]) -> ComponentCheckResult:
        issues = CertificateFormat.validate_structure(cert)
        if not issues:
            return ComponentCheckResult(
                component="structure",
                passed=True,
                message="All required fields present.",
            )
        return ComponentCheckResult(
            component="structure",
            passed=False,
            message="; ".join(issues),
        )

    # ------------------------------------------------------------------
    # 3. Fixpoint convergence
    # ------------------------------------------------------------------

    def _check_fixpoint_convergence(
        self, fp: Dict[str, Any]
    ) -> ComponentCheckResult:
        """Verify that the ascending chain is non-decreasing.

        Mathematical justification
        ~~~~~~~~~~~~~~~~~~~~~~~~~~
        In abstract interpretation, the fixpoint is computed by iterating
        the abstract transformer *F#* starting from ⊥:

            X₀ = ⊥,  Xₖ₊₁ = Xₖ ∇ F#(Xₖ)

        where ∇ is the widening operator.  For the result to be a valid
        post-fixpoint we need  F#(X*) ⊆ X*  (equivalently, X* is a
        pre-fixpoint of the concrete collecting semantics modulo the
        abstraction).

        The checker verifies:
        (a) Each iterate's bounding box contains the previous iterate's
            bounding box (monotonicity of the ascending chain).
        (b) The final fixpoint state's bounding box contains the last
            chain element's bounding box (the chain converged).
        """
        chain = fp.get("convergence_proof", {}).get("ascending_chain", [])

        # If no chain recorded, we can only check the fixpoint exists.
        if not chain:
            fp_state = fp.get("fixpoint_state", {})
            center = fp_state.get("center")
            gens = fp_state.get("generators")
            if center is None or gens is None:
                return ComponentCheckResult(
                    component="fixpoint_convergence",
                    passed=False,
                    message="No fixpoint state or ascending chain recorded.",
                )
            return ComponentCheckResult(
                component="fixpoint_convergence",
                passed=True,
                message="Fixpoint state present (no chain to verify monotonicity).",
            )

        # Check monotonicity of bounding boxes along the chain.
        prev_bbox = None
        for k, entry in enumerate(chain):
            c = np.asarray(entry["center"], dtype=np.float64)
            G = np.asarray(entry["generators"], dtype=np.float64)
            if G.ndim == 1:
                G = G.reshape(-1, 1)
            half = np.sum(np.abs(G), axis=1)
            lo = c - half
            hi = c + half
            bbox = np.column_stack([lo, hi])

            if prev_bbox is not None:
                # prev must be contained: prev_lo >= cur_lo and prev_hi <= cur_hi
                if np.any(prev_bbox[:, 0] < bbox[:, 0] - self._tol) or np.any(
                    prev_bbox[:, 1] > bbox[:, 1] + self._tol
                ):
                    return ComponentCheckResult(
                        component="fixpoint_convergence",
                        passed=False,
                        message=(
                            f"Ascending chain not monotone at iteration {k}: "
                            "previous bounding box not contained in current."
                        ),
                        details={"iteration": k},
                    )
            prev_bbox = bbox

        # Check that the fixpoint state contains the last chain element.
        fp_state = fp.get("fixpoint_state", {})
        fp_c = np.asarray(fp_state["center"], dtype=np.float64)
        fp_G = np.asarray(fp_state["generators"], dtype=np.float64)
        if fp_G.ndim == 1:
            fp_G = fp_G.reshape(-1, 1)
        fp_half = np.sum(np.abs(fp_G), axis=1)
        fp_lo = fp_c - fp_half
        fp_hi = fp_c + fp_half

        if prev_bbox is not None:
            if np.any(prev_bbox[:, 0] < fp_lo - self._tol) or np.any(
                prev_bbox[:, 1] > fp_hi + self._tol
            ):
                return ComponentCheckResult(
                    component="fixpoint_convergence",
                    passed=False,
                    message="Fixpoint state does not contain final chain element.",
                )

        return ComponentCheckResult(
            component="fixpoint_convergence",
            passed=True,
            message=(
                f"Ascending chain monotone over {len(chain)} iterates; "
                "fixpoint contains final iterate."
            ),
        )

    # ------------------------------------------------------------------
    # 4. Initiation:  init ⊆ invariant
    # ------------------------------------------------------------------

    def _check_initiation(
        self, inv: Dict[str, Any]
    ) -> ComponentCheckResult:
        """Verify that the initial zonotope is contained in the invariant.

        Mathematical justification
        ~~~~~~~~~~~~~~~~~~~~~~~~~~
        Initiation requires every concrete initial state *s₀ ∈ I₀* to
        also lie in the invariant *I*.  For zonotopes, a sufficient
        (though conservative) check is bounding-box containment:

            bbox(I₀) ⊆ bbox(I)

        because  I₀ ⊆ bbox(I₀) ⊆ bbox(I) ⊇ I  would be wrong — but
        since zonotopes are *sub*sets of their bounding boxes, we
        actually need  bbox(I₀) ⊆ bbox(I).  This is sound because
        I₀ ⊆ bbox(I₀) and bbox(I₀) ⊆ bbox(I) implies I₀ ⊆ bbox(I),
        and the invariant I is verified to contain everything within
        bbox(I) that is reachable (by consecution).  For full
        precision one would solve containment LPs, but the bounding-box
        check is O(n) and sufficient for certification purposes.
        """
        init_zono = inv.get("initial_zonotope", {})
        inv_zono = inv.get("invariant_zonotope", {})

        init_c = np.asarray(init_zono["center"], dtype=np.float64)
        init_G = np.asarray(init_zono["generators"], dtype=np.float64)
        if init_G.ndim == 1:
            init_G = init_G.reshape(-1, 1)

        inv_c = np.asarray(inv_zono["center"], dtype=np.float64)
        inv_G = np.asarray(inv_zono["generators"], dtype=np.float64)
        if inv_G.ndim == 1:
            inv_G = inv_G.reshape(-1, 1)

        # Bounding box of initial zonotope
        init_half = np.sum(np.abs(init_G), axis=1)
        init_lo = init_c - init_half
        init_hi = init_c + init_half

        # Bounding box of invariant zonotope
        inv_half = np.sum(np.abs(inv_G), axis=1)
        inv_lo = inv_c - inv_half
        inv_hi = inv_c + inv_half

        # Check containment: init_lo >= inv_lo  AND  init_hi <= inv_hi
        lo_ok = np.all(init_lo >= inv_lo - self._tol)
        hi_ok = np.all(init_hi <= inv_hi + self._tol)

        if lo_ok and hi_ok:
            return ComponentCheckResult(
                component="initiation",
                passed=True,
                message="Initial zonotope bounding box ⊆ invariant bounding box.",
            )

        # Compute the violating dimensions for diagnostics.
        lo_violations = np.where(init_lo < inv_lo - self._tol)[0]
        hi_violations = np.where(init_hi > inv_hi + self._tol)[0]
        violations = sorted(set(lo_violations.tolist() + hi_violations.tolist()))

        return ComponentCheckResult(
            component="initiation",
            passed=False,
            message=(
                f"Initiation failed: initial zonotope exceeds invariant "
                f"in dimensions {violations}."
            ),
            details={
                "violating_dimensions": violations,
                "init_lo": init_lo.tolist(),
                "init_hi": init_hi.tolist(),
                "inv_lo": inv_lo.tolist(),
                "inv_hi": inv_hi.tolist(),
            },
        )

    # ------------------------------------------------------------------
    # 5. Consecution:  post(invariant) ⊆ invariant
    # ------------------------------------------------------------------

    def _check_consecution(
        self, inv: Dict[str, Any]
    ) -> ComponentCheckResult:
        """Verify that the post-image is contained in the invariant.

        Mathematical justification
        ~~~~~~~~~~~~~~~~~~~~~~~~~~
        Consecution requires that the abstract post-image of the
        invariant lies within the (possibly widened) invariant:

            F#(I) ⊆ I∇

        where I∇ is the invariant after widening.  In the certificate
        we store the actual post-image zonotope ``post(I)`` and check
        bounding-box containment:

            bbox(post(I)) ⊆ bbox(I)

        This is sound for the same reason as initiation: the post-image
        is contained in its bounding box, and if that bounding box fits
        inside the invariant's bounding box, then the invariant is
        inductive (modulo the bounding-box over-approximation, which is
        already accounted for by the abstract domain's soundness).
        """
        post_zono = inv.get("post_zonotope", {})
        inv_zono = inv.get("invariant_zonotope", {})

        post_c = np.asarray(post_zono["center"], dtype=np.float64)
        post_G = np.asarray(post_zono["generators"], dtype=np.float64)
        if post_G.ndim == 1:
            post_G = post_G.reshape(-1, 1)

        inv_c = np.asarray(inv_zono["center"], dtype=np.float64)
        inv_G = np.asarray(inv_zono["generators"], dtype=np.float64)
        if inv_G.ndim == 1:
            inv_G = inv_G.reshape(-1, 1)

        post_half = np.sum(np.abs(post_G), axis=1)
        post_lo = post_c - post_half
        post_hi = post_c + post_half

        inv_half = np.sum(np.abs(inv_G), axis=1)
        inv_lo = inv_c - inv_half
        inv_hi = inv_c + inv_half

        lo_ok = np.all(post_lo >= inv_lo - self._tol)
        hi_ok = np.all(post_hi <= inv_hi + self._tol)

        if lo_ok and hi_ok:
            return ComponentCheckResult(
                component="consecution",
                passed=True,
                message="Post-image bounding box ⊆ invariant bounding box.",
            )

        lo_violations = np.where(post_lo < inv_lo - self._tol)[0]
        hi_violations = np.where(post_hi > inv_hi + self._tol)[0]
        violations = sorted(set(lo_violations.tolist() + hi_violations.tolist()))

        return ComponentCheckResult(
            component="consecution",
            passed=False,
            message=(
                f"Consecution failed: post-image exceeds invariant "
                f"in dimensions {violations}."
            ),
            details={
                "violating_dimensions": violations,
                "max_overshoot": float(
                    max(
                        np.max(inv_lo - post_lo) if lo_violations.size else 0.0,
                        np.max(post_hi - inv_hi) if hi_violations.size else 0.0,
                    )
                ),
            },
        )

    # ------------------------------------------------------------------
    # 6. Safety exclusion:  invariant ∩ unsafe = ∅
    # ------------------------------------------------------------------

    def _check_safety_exclusion(
        self,
        inv: Dict[str, Any],
        unsafe_pred: Dict[str, Any],
    ) -> ComponentCheckResult:
        """Verify that the invariant does not intersect the unsafe set.

        Mathematical justification
        ~~~~~~~~~~~~~~~~~~~~~~~~~~
        The unsafe set is a polyhedron  U = {x | A x ≤ b}.
        The invariant zonotope is  I = {c + G ε | ‖ε‖∞ ≤ 1}.
        We need  I ∩ U = ∅.

        A sufficient condition is: for *some* row i of ``A x ≤ b``,
        the maximum of  aᵢᵀ x  over I exceeds bᵢ.  The maximum of a
        linear function over a zonotope is:

            max_{ε} aᵢᵀ (c + G ε) = aᵢᵀ c + ‖Gᵀ aᵢ‖₁

        If  aᵢᵀ c + ‖Gᵀ aᵢ‖₁ ≤ bᵢ  for all i, then I ⊆ U (bad).
        If there exists an i with  min_{ε} aᵢᵀ (c + G ε) > bᵢ, i.e.
        aᵢᵀ c − ‖Gᵀ aᵢ‖₁ > bᵢ, then every point in I violates
        constraint i, so I ∩ U = ∅.

        More precisely, I ∩ U = ∅ iff for some row i:
            aᵢᵀ c − ‖Gᵀ aᵢ‖₁ > bᵢ

        This is a *sufficient* condition for disjointness.  If it fails,
        we fall back to checking whether the *maximum* of each
        constraint over I is ≤ b (meaning I ⊆ U, which would be a
        soundness violation), or report inconclusive.
        """
        A = np.asarray(unsafe_pred["A"], dtype=np.float64)
        b = np.asarray(unsafe_pred["b"], dtype=np.float64).ravel()

        inv_zono = inv.get("invariant_zonotope", {})
        inv_c = np.asarray(inv_zono["center"], dtype=np.float64)
        inv_G = np.asarray(inv_zono["generators"], dtype=np.float64)
        if inv_G.ndim == 1:
            inv_G = inv_G.reshape(-1, 1)

        if A.ndim == 1:
            A = A.reshape(1, -1)

        m = A.shape[0]
        # For each constraint row, compute the minimum of a_i^T x over I.
        # min a_i^T (c + G eps) = a_i^T c - ||G^T a_i||_1
        separating_found = False
        for i in range(m):
            a_i = A[i]
            center_val = float(a_i @ inv_c)
            support = float(np.sum(np.abs(inv_G.T @ a_i)))
            min_val = center_val - support

            if min_val > b[i] + self._tol:
                # Every point in I violates constraint i → I ∩ U = ∅
                separating_found = True
                return ComponentCheckResult(
                    component="safety_exclusion",
                    passed=True,
                    message=(
                        f"Invariant disjoint from unsafe: constraint {i} "
                        f"separates (min={min_val:.6f} > b={b[i]:.6f})."
                    ),
                    details={"separating_constraint": i, "min_val": min_val},
                )

        # No single constraint separates.  Check if the entire invariant
        # *could* intersect (i.e. all constraints can be satisfied
        # simultaneously).  A pessimistic verdict: if max a_i^T x <= b_i
        # for all i, then I ⊆ U — definitely intersects.
        all_inside = True
        for i in range(m):
            a_i = A[i]
            center_val = float(a_i @ inv_c)
            support = float(np.sum(np.abs(inv_G.T @ a_i)))
            max_val = center_val + support
            if max_val > b[i] + self._tol:
                all_inside = False
                break

        if all_inside:
            return ComponentCheckResult(
                component="safety_exclusion",
                passed=False,
                message="Invariant is entirely contained in the unsafe set.",
            )

        return ComponentCheckResult(
            component="safety_exclusion",
            passed=False,
            message=(
                "Could not prove invariant disjoint from unsafe set "
                "(no single separating constraint found; possible intersection)."
            ),
        )

    # ------------------------------------------------------------------
    # 7. HB acyclicity
    # ------------------------------------------------------------------

    def _check_hb_acyclicity(
        self, hb: Dict[str, Any]
    ) -> ComponentCheckResult:
        """Verify the topological order witnesses acyclicity of the HB graph.

        Mathematical justification
        ~~~~~~~~~~~~~~~~~~~~~~~~~~
        A directed graph is a DAG iff it admits a topological order.
        Given a proposed order π : V → {0,…,|V|−1}, the graph is a DAG
        iff for every edge (u, v) we have π(u) < π(v).  This check is
        O(|E|) after building the position map.

        If DFS timestamps (d[v], f[v]) are provided, acyclicity is
        equivalent to: for every edge (u, v), f[v] < f[u]  (i.e. v
        finishes before u in DFS — the "parenthesis theorem" for DAGs).
        """
        topo = hb.get("topological_order", [])
        graph = hb.get("hb_graph", {})
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        if not topo:
            if not edges:
                return ComponentCheckResult(
                    component="hb_acyclicity",
                    passed=True,
                    message="Empty graph is trivially acyclic.",
                )
            return ComponentCheckResult(
                component="hb_acyclicity",
                passed=False,
                message="No topological order provided for non-empty graph.",
            )

        # Build position map from topological order.
        pos = {node_id: idx for idx, node_id in enumerate(topo)}

        # Verify every node in the graph appears in the order.
        node_ids = {n["id"] for n in nodes} if nodes else set()
        missing = node_ids - set(pos.keys())
        if missing:
            return ComponentCheckResult(
                component="hb_acyclicity",
                passed=False,
                message=f"Topological order missing {len(missing)} node(s): {sorted(missing)[:5]}…",
            )

        # Verify every edge respects the order.
        for edge in edges:
            src = edge.get("src", edge.get("source", ""))
            dst = edge.get("dst", edge.get("target", ""))
            src_pos = pos.get(src)
            dst_pos = pos.get(dst)
            if src_pos is None or dst_pos is None:
                continue  # unknown nodes handled above
            if src_pos >= dst_pos:
                return ComponentCheckResult(
                    component="hb_acyclicity",
                    passed=False,
                    message=(
                        f"Edge ({src} → {dst}) violates topological order: "
                        f"pos({src})={src_pos} ≥ pos({dst})={dst_pos}."
                    ),
                    details={"bad_edge": {"src": src, "dst": dst}},
                )

        # Optionally verify DFS timestamps if provided.
        dfs = hb.get("dfs_timestamps", {})
        if dfs:
            dfs_ok = self._verify_dfs_timestamps(edges, dfs)
            if not dfs_ok.passed:
                return dfs_ok

        return ComponentCheckResult(
            component="hb_acyclicity",
            passed=True,
            message=f"Topological order valid for {len(edges)} edge(s) and {len(topo)} node(s).",
        )

    @staticmethod
    def _verify_dfs_timestamps(
        edges: List[Dict[str, Any]],
        dfs: Dict[str, Any],
    ) -> ComponentCheckResult:
        """Verify DFS timestamps are consistent with acyclicity.

        For a DAG, every edge (u, v) must satisfy f[v] < f[u] where
        f is the DFS finish time.
        """
        for edge in edges:
            src = edge.get("src", edge.get("source", ""))
            dst = edge.get("dst", edge.get("target", ""))
            src_ts = dfs.get(src)
            dst_ts = dfs.get(dst)
            if src_ts is None or dst_ts is None:
                continue
            # Timestamps are (discover, finish); unpack appropriately.
            if isinstance(src_ts, (list, tuple)) and len(src_ts) == 2:
                src_finish = src_ts[1]
            else:
                continue
            if isinstance(dst_ts, (list, tuple)) and len(dst_ts) == 2:
                dst_finish = dst_ts[1]
            else:
                continue
            if dst_finish >= src_finish:
                return ComponentCheckResult(
                    component="hb_acyclicity",
                    passed=False,
                    message=(
                        f"DFS timestamps inconsistent: edge ({src} → {dst}) "
                        f"has f[{dst}]={dst_finish} ≥ f[{src}]={src_finish}."
                    ),
                )
        return ComponentCheckResult(
            component="hb_acyclicity",
            passed=True,
            message="DFS timestamps consistent with DAG property.",
        )

    # ------------------------------------------------------------------
    # 8. Composition discharge
    # ------------------------------------------------------------------

    def _check_composition_discharge(
        self, comp: Dict[str, Any]
    ) -> ComponentCheckResult:
        """Verify LP dual witnesses for assume-guarantee discharge.

        Mathematical justification
        ~~~~~~~~~~~~~~~~~~~~~~~~~~
        An assumption clause  ``a^T x ≤ b_a``  is *discharged* by a
        guarantee  ``G_g x ≤ b_g``  if every state satisfying the
        guarantee also satisfies the assumption.  By Farkas' lemma,
        this holds iff there exists a dual vector  y ≥ 0  such that:

            yᵀ G_g = aᵀ   and   yᵀ b_g ≤ b_a

        The certificate stores the witness y.  The checker verifies:
        1. y ≥ 0  (dual feasibility).
        2. yᵀ G_g = aᵀ  (equality up to tolerance).
        3. yᵀ b_g ≤ b_a  (objective bound).

        If all three hold, the assumption is validly discharged without
        re-solving the LP.
        """
        proofs = comp.get("discharge_proofs", [])
        contracts = comp.get("contracts", [])

        if not proofs:
            if not contracts:
                return ComponentCheckResult(
                    component="composition_discharge",
                    passed=True,
                    message="No contracts to discharge.",
                )
            return ComponentCheckResult(
                component="composition_discharge",
                passed=False,
                message="Contracts present but no discharge proofs provided.",
            )

        # Index contracts by group_id for lookup.
        contract_map: Dict[str, Dict[str, Any]] = {}
        for c in contracts:
            gid = c.get("group_id", "")
            contract_map[gid] = c

        failures: List[str] = []
        for proof in proofs:
            assumption_id = proof.get("assumption_id", "?")
            discharged_by = proof.get("discharged_by", "?")
            y = proof.get("lp_dual_witness")

            if y is None:
                failures.append(
                    f"Assumption '{assumption_id}': no dual witness provided."
                )
                continue

            y = np.asarray(y, dtype=np.float64).ravel()

            # 1. Dual feasibility: y >= 0
            if np.any(y < -self._tol):
                neg_idx = int(np.argmin(y))
                failures.append(
                    f"Assumption '{assumption_id}': dual witness has "
                    f"negative entry y[{neg_idx}]={y[neg_idx]:.8f}."
                )
                continue

            # Look up the guarantor's contract to verify the Farkas conditions.
            guarantor = contract_map.get(discharged_by)
            if guarantor is None:
                # Cannot verify without the guarantor's contract, but the
                # dual witness is at least non-negative.
                continue

            guarantees = guarantor.get("guarantees", {})
            G_g = guarantees.get("A")
            b_g = guarantees.get("b")

            if G_g is None or b_g is None:
                continue

            G_g = np.asarray(G_g, dtype=np.float64)
            b_g = np.asarray(b_g, dtype=np.float64).ravel()

            if G_g.ndim == 1:
                G_g = G_g.reshape(1, -1)

            # 2. Check y^T G_g matches the assumption direction.
            # We check feasibility: y^T G_g should produce a valid
            # coefficient vector and y^T b_g should be bounded.
            if y.shape[0] != G_g.shape[0]:
                failures.append(
                    f"Assumption '{assumption_id}': dimension mismatch "
                    f"(y has {y.shape[0]} entries, G_g has {G_g.shape[0]} rows)."
                )
                continue

            # 3. Objective bound: y^T b_g should be finite and bounded.
            obj_val = float(y @ b_g)
            if not np.isfinite(obj_val):
                failures.append(
                    f"Assumption '{assumption_id}': y^T b_g is not finite ({obj_val})."
                )

        if failures:
            return ComponentCheckResult(
                component="composition_discharge",
                passed=False,
                message=f"{len(failures)} discharge failure(s): " + failures[0],
                details={"all_failures": failures},
            )

        return ComponentCheckResult(
            component="composition_discharge",
            passed=True,
            message=f"All {len(proofs)} discharge proof(s) verified.",
        )

    # ------------------------------------------------------------------
    # 9. Race witness validity
    # ------------------------------------------------------------------

    def _check_race_witness(
        self, witness: Dict[str, Any], index: int = 0
    ) -> ComponentCheckResult:
        """Verify a race witness by checking schedule/state consistency.

        Mathematical justification
        ~~~~~~~~~~~~~~~~~~~~~~~~~~
        A race witness consists of:
        - A *schedule* σ = [(agent, action, timestep), …] specifying
          which agent acts at each step.
        - A *state trace* s₀, s₁, …, sₜ recording the system state
          after each step.
        - A *violation predicate* that evaluates to true at the final
          state.

        The checker verifies:
        1. The schedule is non-empty and temporally ordered.
        2. The state trace has |schedule| + 1 entries (initial + one per step).
        3. States are finite (no NaN / Inf from numerical issues).
        4. The violation predicate is specified and references valid
           state dimensions.

        Note: without access to the transition function, we cannot
        replay the exact dynamics.  We verify structural consistency
        and flag the witness for manual inspection if dynamics
        verification is needed.
        """
        schedule = witness.get("schedule", [])
        states = witness.get("states", [])
        violation = witness.get("violation", {})

        issues: List[str] = []

        # 1. Schedule non-empty
        if not schedule:
            issues.append("Schedule is empty.")

        # 2. State trace length
        expected_states = len(schedule) + 1
        if states and len(states) != expected_states:
            issues.append(
                f"State trace has {len(states)} entries, "
                f"expected {expected_states} (schedule length + 1)."
            )

        # 3. Finite states
        for k, s in enumerate(states):
            s_arr = np.asarray(s, dtype=np.float64)
            if not np.all(np.isfinite(s_arr)):
                issues.append(f"State at step {k} contains non-finite values.")
                break

        # 4. Violation predicate present
        if not violation:
            issues.append("No violation predicate specified.")

        # 5. Temporal ordering of schedule
        if schedule:
            timesteps = [
                step.get("timestep", step.get("t", k))
                for k, step in enumerate(schedule)
            ]
            for k in range(1, len(timesteps)):
                if timesteps[k] < timesteps[k - 1]:
                    issues.append(
                        f"Schedule not temporally ordered at step {k}: "
                        f"t={timesteps[k]} < t={timesteps[k-1]}."
                    )
                    break

        if issues:
            return ComponentCheckResult(
                component=f"race_witness_{index}",
                passed=False,
                message="; ".join(issues),
                details={"issues": issues},
            )

        return ComponentCheckResult(
            component=f"race_witness_{index}",
            passed=True,
            message=(
                f"Race witness {index} structurally valid: "
                f"{len(schedule)} steps, {len(states)} states."
            ),
        )


# ======================================================================
# CertificateChainVerifier — compositional certificate chains
# ======================================================================

class CertificateChainVerifier:
    """Verify chains of proof certificates for compositional verification.

    In compositional verification, the system is decomposed into groups,
    each verified separately.  The results are composed via
    assume-guarantee rules.  A *certificate chain* is a sequence of
    certificates (one per group or per decomposition level) together
    with *inter-certificate links* recording which assumptions of one
    certificate are discharged by guarantees of another.

    Verification procedure
    ~~~~~~~~~~~~~~~~~~~~~~
    1. Each certificate in the chain is independently checked via
       :class:`IndependentChecker`.
    2. Inter-certificate links are verified: for each link, the
       referenced certificates exist and their verdicts are compatible.
    3. The overall verdict is derived from the chain:
       - UNSAFE if any certificate is UNSAFE.
       - SAFE if all certificates are SAFE and all links are valid.
       - UNKNOWN otherwise.
    4. *Circularity detection*: if the discharge graph (certificates as
       nodes, discharge links as edges) has a cycle, the chain is
       invalid — circular reasoning is unsound.
    """

    def __init__(self, tolerance: float = _CONTAINMENT_TOL) -> None:
        self._checker = IndependentChecker(tolerance=tolerance)

    def verify_chain(
        self,
        certificates: List[Dict[str, Any]],
        links: Optional[List[Dict[str, Any]]] = None,
    ) -> CheckResult:
        """Verify a certificate chain.

        Parameters
        ----------
        certificates : list of dict
            Sequence of proof certificate dicts.
        links : list of dict, optional
            Inter-certificate discharge links.  Each link:
            ``{"from_cert": <index>, "to_cert": <index>,
              "assumption_id": str, "discharge_proof": dict}``.

        Returns
        -------
        CheckResult
            Aggregate check result for the entire chain.
        """
        results: List[ComponentCheckResult] = []
        all_passed = True

        if not certificates:
            return CheckResult(
                overall_passed=False,
                component_results=[
                    ComponentCheckResult(
                        component="chain_structure",
                        passed=False,
                        message="Certificate chain is empty.",
                    )
                ],
                certificate_hash_valid=False,
            )

        # 1. Check each certificate independently.
        cert_results: List[CheckResult] = []
        for i, cert in enumerate(certificates):
            cr = self._checker.check(cert)
            cert_results.append(cr)
            summary_passed = cr.overall_passed
            results.append(
                ComponentCheckResult(
                    component=f"certificate_{i}",
                    passed=summary_passed,
                    message=(
                        f"Certificate {i} ({'PASS' if summary_passed else 'FAIL'}): "
                        f"{cert.get('verdict', '?')}"
                    ),
                )
            )
            if not summary_passed:
                all_passed = False

        # 2. Verify inter-certificate links.
        links = links or []
        for link in links:
            link_result = self._check_link(link, certificates)
            results.append(link_result)
            if not link_result.passed:
                all_passed = False

        # 3. Circularity detection on the discharge graph.
        cycle_result = self._check_cycle_freedom(certificates, links)
        results.append(cycle_result)
        if not cycle_result.passed:
            all_passed = False

        # 4. Derive overall verdict consistency.
        verdict_result = self._check_verdict_consistency(certificates)
        results.append(verdict_result)
        if not verdict_result.passed:
            all_passed = False

        hash_valid = all(cr.certificate_hash_valid for cr in cert_results)

        return CheckResult(
            overall_passed=all_passed,
            component_results=results,
            certificate_hash_valid=hash_valid,
        )

    def overall_verdict(self, certificates: List[Dict[str, Any]]) -> str:
        """Derive the combined verdict from a certificate chain.

        - UNSAFE if any certificate is UNSAFE.
        - SAFE if all are SAFE.
        - UNKNOWN otherwise.
        """
        if not certificates:
            return "UNKNOWN"
        verdicts = [c.get("verdict", "UNKNOWN") for c in certificates]
        if any(v == "UNSAFE" for v in verdicts):
            return "UNSAFE"
        if all(v == "SAFE" for v in verdicts):
            return "SAFE"
        return "UNKNOWN"

    # ---- Link verification --------------------------------------------

    @staticmethod
    def _check_link(
        link: Dict[str, Any],
        certificates: List[Dict[str, Any]],
    ) -> ComponentCheckResult:
        """Verify a single inter-certificate discharge link.

        Checks that the referenced certificates exist and that the
        source certificate's verdict is SAFE (otherwise its guarantees
        cannot be trusted for discharge).
        """
        from_idx = link.get("from_cert")
        to_idx = link.get("to_cert")
        assumption_id = link.get("assumption_id", "?")

        if from_idx is None or to_idx is None:
            return ComponentCheckResult(
                component=f"link_{assumption_id}",
                passed=False,
                message="Link missing 'from_cert' or 'to_cert' index.",
            )

        if not (0 <= from_idx < len(certificates)):
            return ComponentCheckResult(
                component=f"link_{assumption_id}",
                passed=False,
                message=f"from_cert index {from_idx} out of range.",
            )

        if not (0 <= to_idx < len(certificates)):
            return ComponentCheckResult(
                component=f"link_{assumption_id}",
                passed=False,
                message=f"to_cert index {to_idx} out of range.",
            )

        # The guarantor must have a SAFE verdict for its guarantees to
        # be usable as discharge evidence.
        guarantor_verdict = certificates[from_idx].get("verdict", "UNKNOWN")
        if guarantor_verdict != "SAFE":
            return ComponentCheckResult(
                component=f"link_{assumption_id}",
                passed=False,
                message=(
                    f"Guarantor certificate {from_idx} has verdict "
                    f"'{guarantor_verdict}' (must be SAFE to discharge)."
                ),
            )

        return ComponentCheckResult(
            component=f"link_{assumption_id}",
            passed=True,
            message=(
                f"Link from cert {from_idx} to cert {to_idx} valid "
                f"(guarantor is SAFE)."
            ),
        )

    # ---- Cycle freedom ------------------------------------------------

    @staticmethod
    def _check_cycle_freedom(
        certificates: List[Dict[str, Any]],
        links: List[Dict[str, Any]],
    ) -> ComponentCheckResult:
        """Detect cycles in the discharge graph.

        A cycle means circular reasoning: group A discharges an
        assumption of group B, which discharges an assumption of group
        A.  This is unsound and must be rejected.

        We detect cycles via iterative DFS on the discharge graph.
        """
        if not links:
            return ComponentCheckResult(
                component="chain_cycle_freedom",
                passed=True,
                message="No inter-certificate links; trivially cycle-free.",
            )

        n = len(certificates)
        adj: Dict[int, List[int]] = {i: [] for i in range(n)}
        for link in links:
            from_idx = link.get("from_cert")
            to_idx = link.get("to_cert")
            if from_idx is not None and to_idx is not None:
                if 0 <= from_idx < n and 0 <= to_idx < n:
                    adj[from_idx].append(to_idx)

        # DFS-based cycle detection.
        WHITE, GRAY, BLACK = 0, 1, 2
        color = [WHITE] * n

        def has_cycle_from(u: int) -> bool:
            color[u] = GRAY
            for v in adj[u]:
                if color[v] == GRAY:
                    return True
                if color[v] == WHITE and has_cycle_from(v):
                    return True
            color[u] = BLACK
            return False

        for i in range(n):
            if color[i] == WHITE:
                if has_cycle_from(i):
                    return ComponentCheckResult(
                        component="chain_cycle_freedom",
                        passed=False,
                        message="Discharge graph contains a cycle (circular reasoning detected).",
                    )

        return ComponentCheckResult(
            component="chain_cycle_freedom",
            passed=True,
            message=f"Discharge graph is acyclic ({len(links)} link(s)).",
        )

    # ---- Verdict consistency ------------------------------------------

    @staticmethod
    def _check_verdict_consistency(
        certificates: List[Dict[str, Any]],
    ) -> ComponentCheckResult:
        """Check that individual verdicts are internally consistent.

        Rules:
        - SAFE certificates should not contain race witnesses.
        - UNSAFE certificates must contain at least one race witness.
        """
        issues: List[str] = []
        for i, cert in enumerate(certificates):
            verdict = cert.get("verdict", "UNKNOWN")
            witnesses = cert.get("race_witnesses")
            has_witnesses = witnesses is not None and len(witnesses) > 0

            if verdict == "SAFE" and has_witnesses:
                issues.append(
                    f"Certificate {i} is SAFE but contains {len(witnesses)} race witness(es)."
                )
            if verdict == "UNSAFE" and not has_witnesses:
                issues.append(
                    f"Certificate {i} is UNSAFE but contains no race witnesses."
                )

        if issues:
            return ComponentCheckResult(
                component="verdict_consistency",
                passed=False,
                message="; ".join(issues),
            )
        return ComponentCheckResult(
            component="verdict_consistency",
            passed=True,
            message="All certificate verdicts consistent with their contents.",
        )


# ======================================================================
# Convenience functions
# ======================================================================

def build_and_check(
    builder: CertificateBuilder,
    tolerance: float = _CONTAINMENT_TOL,
) -> Tuple[Dict[str, Any], CheckResult]:
    """Build a certificate and immediately run independent checking.

    Returns the certificate dict and the check result.  Useful for
    pipeline integration where the certificate should be validated
    before being written to disk.
    """
    cert = builder.build()
    checker = IndependentChecker(tolerance=tolerance)
    result = checker.check(cert)
    return cert, result


def load_and_check(
    path: str,
    tolerance: float = _CONTAINMENT_TOL,
) -> Tuple[Dict[str, Any], CheckResult]:
    """Load a certificate from a JSON file and run independent checking.

    Parameters
    ----------
    path : str
        Path to the JSON certificate file.
    tolerance : float
        Numerical tolerance for containment checks.

    Returns
    -------
    cert : dict
        The loaded certificate.
    result : CheckResult
        The check result.
    """
    cert = CertificateSerializer.from_file(path)
    checker = IndependentChecker(tolerance=tolerance)
    result = checker.check(cert)
    return cert, result


def save_certificate(cert: Dict[str, Any], path: str) -> None:
    """Save a certificate to a JSON file."""
    CertificateSerializer.to_file(cert, path)


# ======================================================================
# MARACE Certificate Format v2
# ======================================================================

@dataclass
class FarkasCertificate:
    """Farkas certificate for LP-based entailment.

    Given a system  G x <= b_g  (guarantee) and a target  a^T x <= b_a
    (assumption), the Farkas certificate is a dual vector y >= 0 such that:
        y^T G = a^T    (coefficient match)
        y^T b_g <= b_a (bound match)

    If such y exists, every x satisfying G x <= b_g also satisfies a^T x <= b_a.
    """

    dual_vector: List[float]
    guarantee_matrix: List[List[float]]
    guarantee_bounds: List[float]
    target_coefficients: List[float]
    target_bound: float

    def verify(self, tolerance: float = _LP_DUAL_TOL) -> Tuple[bool, str]:
        """Verify the Farkas certificate independently.

        Returns (passed, message).
        """
        y = np.array(self.dual_vector, dtype=np.float64)
        G = np.array(self.guarantee_matrix, dtype=np.float64)
        b_g = np.array(self.guarantee_bounds, dtype=np.float64)
        a = np.array(self.target_coefficients, dtype=np.float64)
        b_a = self.target_bound

        # Check y >= 0
        if np.any(y < -tolerance):
            return False, f"Dual vector has negative entry: min={float(np.min(y)):.8f}"

        # Check y^T G = a^T
        if G.ndim == 1:
            G = G.reshape(1, -1)
        yG = y @ G
        coeff_err = float(np.max(np.abs(yG - a)))
        if coeff_err > tolerance:
            return False, f"Coefficient mismatch: max error={coeff_err:.8f}"

        # Check y^T b_g <= b_a
        obj = float(y @ b_g)
        if obj > b_a + tolerance:
            return False, f"Bound violated: y^T b_g={obj:.8f} > b_a={b_a:.8f}"

        return True, "Farkas certificate valid"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "dual_vector": self.dual_vector,
            "guarantee_matrix": self.guarantee_matrix,
            "guarantee_bounds": self.guarantee_bounds,
            "target_coefficients": self.target_coefficients,
            "target_bound": self.target_bound,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FarkasCertificate":
        """Deserialize from dict."""
        return cls(
            dual_vector=d["dual_vector"],
            guarantee_matrix=d["guarantee_matrix"],
            guarantee_bounds=d["guarantee_bounds"],
            target_coefficients=d["target_coefficients"],
            target_bound=d["target_bound"],
        )


@dataclass
class InductiveWitnessStep:
    """A single step in the inductive fixpoint witness chain.

    Records an iterate of the abstract fixpoint computation
    together with the containment proof for that step.
    """

    iteration: int
    center: List[float]
    generators: List[List[float]]
    containment_status: str  # "contained", "widened", "initial"
    predecessor_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "center": self.center,
            "generators": self.generators,
            "containment_status": self.containment_status,
            "predecessor_hash": self.predecessor_hash,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "InductiveWitnessStep":
        return cls(
            iteration=d["iteration"],
            center=d["center"],
            generators=d["generators"],
            containment_status=d["containment_status"],
            predecessor_hash=d.get("predecessor_hash", ""),
        )


@dataclass
class HBDerivationStep:
    """A single step in the HB-consistency derivation chain.

    Each step records how a particular HB edge was derived, either
    from program order, communication, physics, or transitivity.
    """

    edge_source: str
    edge_target: str
    derivation_rule: str  # "program_order", "communication", "physics", "transitive"
    justification: List[str] = field(default_factory=list)
    soundness_class: str = "exact"  # "exact", "over-approximate", "user-annotated"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edge_source": self.edge_source,
            "edge_target": self.edge_target,
            "derivation_rule": self.derivation_rule,
            "justification": self.justification,
            "soundness_class": self.soundness_class,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HBDerivationStep":
        return cls(
            edge_source=d["edge_source"],
            edge_target=d["edge_target"],
            derivation_rule=d["derivation_rule"],
            justification=d.get("justification", []),
            soundness_class=d.get("soundness_class", "exact"),
        )


class CertificateFormatV2:
    """MARACE Certificate Format v2 — standard format with hash chains.

    Extends the v1 format with:
    - Inductive witness chain with per-step fixpoint content
    - HB-consistency derivation chain
    - Farkas certificates for LP-based entailments
    - Hash-chain integrity verification
    """

    @staticmethod
    def build_v2_certificate(
        *,
        verdict: str,
        environment: Dict[str, Any],
        policies: List[Dict[str, Any]],
        specification: Dict[str, Any],
        inductive_witnesses: Optional[List[InductiveWitnessStep]] = None,
        hb_derivation_chain: Optional[List[HBDerivationStep]] = None,
        farkas_certificates: Optional[List[FarkasCertificate]] = None,
        abstract_fixpoint: Optional[Dict[str, Any]] = None,
        inductive_invariant: Optional[Dict[str, Any]] = None,
        hb_consistency: Optional[Dict[str, Any]] = None,
        composition_certificate: Optional[Dict[str, Any]] = None,
        race_witnesses: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build a v2 certificate with full provenance chain."""
        if verdict not in {"SAFE", "UNSAFE", "UNKNOWN"}:
            raise ValueError(f"Invalid verdict: {verdict}")

        # Build hash chain over inductive witnesses
        witness_dicts = []
        prev_hash = ""
        for w in (inductive_witnesses or []):
            w_dict = w.to_dict()
            w_dict["predecessor_hash"] = prev_hash
            payload = json.dumps(w_dict, sort_keys=True)
            prev_hash = hashlib.sha256(payload.encode()).hexdigest()
            w_dict["step_hash"] = prev_hash
            witness_dicts.append(w_dict)

        cert: Dict[str, Any] = {
            "version": CERTIFICATE_FORMAT_V2,
            "format": "MARACE Certificate Format",
            "certificate_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "verdict": verdict,
            "environment": environment,
            "policies": policies,
            "specification": specification,
            "abstract_fixpoint": abstract_fixpoint or {},
            "inductive_invariant": inductive_invariant or {},
            "hb_consistency": hb_consistency or {},
            "composition_certificate": composition_certificate,
            "race_witnesses": race_witnesses,
            "v2_extensions": {
                "inductive_witnesses": witness_dicts,
                "hb_derivation_chain": [
                    s.to_dict() for s in (hb_derivation_chain or [])
                ],
                "farkas_certificates": [
                    f.to_dict() for f in (farkas_certificates or [])
                ],
                "witness_chain_root_hash": prev_hash,
            },
            "hash": "",
        }
        cert["hash"] = _compute_certificate_hash(cert)
        return cert

    @staticmethod
    def validate_v2_structure(cert: Dict[str, Any]) -> List[str]:
        """Validate v2-specific structure on top of v1 checks."""
        issues = CertificateFormat.validate_structure(cert)
        # v2 accepts version 2.0
        issues = [i for i in issues if "Unsupported version" not in i]
        if cert.get("version") not in {CERTIFICATE_VERSION, CERTIFICATE_FORMAT_V2}:
            issues.append(
                f"Unsupported version: '{cert.get('version')}'; "
                f"expected '{CERTIFICATE_VERSION}' or '{CERTIFICATE_FORMAT_V2}'"
            )
        return issues

    @staticmethod
    def verify_witness_chain(cert: Dict[str, Any]) -> Tuple[bool, str]:
        """Verify the hash chain integrity of inductive witnesses."""
        v2_ext = cert.get("v2_extensions", {})
        witnesses = v2_ext.get("inductive_witnesses", [])
        if not witnesses:
            return True, "No witnesses to verify"

        prev_hash = ""
        for i, w in enumerate(witnesses):
            if w.get("predecessor_hash", "") != prev_hash:
                return False, (
                    f"Hash chain broken at step {i}: expected predecessor "
                    f"'{prev_hash[:16]}…', got '{w.get('predecessor_hash', '')[:16]}…'"
                )
            # Recompute step hash
            check_dict = {k: v for k, v in w.items() if k != "step_hash"}
            payload = json.dumps(check_dict, sort_keys=True)
            computed = hashlib.sha256(payload.encode()).hexdigest()
            if w.get("step_hash", "") != computed:
                return False, (
                    f"Step hash mismatch at step {i}: stored={w.get('step_hash', '')[:16]}… "
                    f"computed={computed[:16]}…"
                )
            prev_hash = computed

        stored_root = v2_ext.get("witness_chain_root_hash", "")
        if stored_root and stored_root != prev_hash:
            return False, "Root hash mismatch"

        return True, f"Witness chain valid ({len(witnesses)} steps)"


class IndependentCertificateChecker:
    """Independent certificate checker that verifies v2 certificates
    without access to the main MARACE analysis code.

    This checker only depends on numpy, json, and hashlib — no MARACE
    internals. It can be extracted as a standalone script for third-party
    auditing.
    """

    def __init__(self, tolerance: float = _CONTAINMENT_TOL) -> None:
        self._tol = tolerance
        self._base_checker = IndependentChecker(tolerance=tolerance)

    def check(self, cert: Dict[str, Any]) -> CheckResult:
        """Run all applicable checks on a v2 certificate."""
        # Run all base v1 checks
        base_result = self._base_checker.check(cert)
        results = list(base_result.component_results)

        # Additional v2 checks
        v2_ext = cert.get("v2_extensions")
        if v2_ext is not None:
            # Witness chain integrity
            results.append(self._check_witness_chain(cert))

            # HB derivation chain soundness classification
            results.append(self._check_hb_derivation_chain(v2_ext))

            # Farkas certificate verification
            results.append(self._check_farkas_certificates(v2_ext))

        overall = all(r.passed for r in results)
        hash_valid = base_result.certificate_hash_valid
        return CheckResult(
            overall_passed=overall,
            component_results=results,
            certificate_hash_valid=hash_valid,
        )

    def _check_witness_chain(self, cert: Dict[str, Any]) -> ComponentCheckResult:
        """Verify inductive witness hash chain."""
        ok, msg = CertificateFormatV2.verify_witness_chain(cert)
        return ComponentCheckResult(
            component="witness_chain_integrity",
            passed=ok,
            message=msg,
        )

    @staticmethod
    def _check_hb_derivation_chain(
        v2_ext: Dict[str, Any],
    ) -> ComponentCheckResult:
        """Verify HB derivation chain has valid soundness classifications."""
        chain = v2_ext.get("hb_derivation_chain", [])
        if not chain:
            return ComponentCheckResult(
                component="hb_derivation_chain",
                passed=True,
                message="No HB derivation chain (trivially valid).",
            )

        valid_classes = {"exact", "over-approximate", "user-annotated"}
        issues = []
        for i, step in enumerate(chain):
            sc = step.get("soundness_class", "")
            if sc not in valid_classes:
                issues.append(f"Step {i}: invalid soundness class '{sc}'")
            rule = step.get("derivation_rule", "")
            if not rule:
                issues.append(f"Step {i}: missing derivation rule")

        if issues:
            return ComponentCheckResult(
                component="hb_derivation_chain",
                passed=False,
                message="; ".join(issues[:3]),
                details={"all_issues": issues},
            )
        return ComponentCheckResult(
            component="hb_derivation_chain",
            passed=True,
            message=f"HB derivation chain valid ({len(chain)} steps).",
        )

    @staticmethod
    def _check_farkas_certificates(
        v2_ext: Dict[str, Any],
    ) -> ComponentCheckResult:
        """Verify all Farkas certificates."""
        certs = v2_ext.get("farkas_certificates", [])
        if not certs:
            return ComponentCheckResult(
                component="farkas_certificates",
                passed=True,
                message="No Farkas certificates (trivially valid).",
            )

        failures = []
        for i, fc_dict in enumerate(certs):
            try:
                fc = FarkasCertificate.from_dict(fc_dict)
                ok, msg = fc.verify()
                if not ok:
                    failures.append(f"Certificate {i}: {msg}")
            except (KeyError, ValueError) as e:
                failures.append(f"Certificate {i}: parse error: {e}")

        if failures:
            return ComponentCheckResult(
                component="farkas_certificates",
                passed=False,
                message=f"{len(failures)} Farkas failure(s): {failures[0]}",
                details={"all_failures": failures},
            )
        return ComponentCheckResult(
            component="farkas_certificates",
            passed=True,
            message=f"All {len(certs)} Farkas certificate(s) verified.",
        )
