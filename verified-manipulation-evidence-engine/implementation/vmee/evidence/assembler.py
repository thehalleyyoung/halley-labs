"""
Evidence bundle assembly and verification.

Packages causal subgraph, Bayesian posteriors, temporal violations,
and SMT proof certificates into a self-contained JSON evidence bundle.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class VerificationCheck:
    """Result of a single verification check on a bundle."""
    name: str
    passed: bool
    status: str
    details: str = ""


@dataclass
class VerificationResult:
    """Result of verifying an evidence bundle."""
    is_valid: bool
    checks: List[VerificationCheck]
    bundle_hash: str = ""


class EvidenceAssembler:
    """Assembles evidence bundles from pipeline results."""

    def __init__(self, config=None):
        self.config = config

    def assemble(
        self,
        causal_result: Any,
        bayesian_result: Any,
        temporal_result: Any,
        proof_result: Any,
        config: Any = None,
    ) -> Dict:
        """Assemble a complete evidence bundle."""
        bundle = {
            "version": "0.1.0",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "causal_subgraph": self._extract_causal(causal_result),
            "bayesian_evidence": self._extract_bayesian(bayesian_result),
            "temporal_violations": self._extract_temporal(temporal_result),
            "proof_certificates": self._extract_proofs(proof_result),
            "soundness": self._extract_soundness(
                causal_result, bayesian_result, temporal_result, proof_result
            ),
        }

        # Compute bundle hash
        content = json.dumps(bundle, sort_keys=True, default=str)
        bundle["bundle_hash"] = hashlib.sha256(content.encode()).hexdigest()

        return bundle

    def save_bundle(self, bundle: Dict, path: str) -> None:
        """Save bundle to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(bundle, f, indent=2, default=str)
        logger.info(f"Evidence bundle saved: {path}")

    def _extract_causal(self, result: Any) -> Dict:
        """Extract causal subgraph from discovery result."""
        dag = getattr(result, 'dag', None)
        effects = getattr(result, 'identified_effects', [])
        bound = getattr(result, 'finite_sample_bound', None)

        causal_info = {
            "dag": {
                "nodes": list(dag.nodes()) if dag else [],
                "edges": list(dag.edges()) if dag else [],
            },
            "identified_effects": [
                {
                    "treatment": e.treatment,
                    "outcome": e.outcome,
                    "method": e.method,
                    "adjustment_set": list(e.adjustment_set),
                    "identified": e.identified,
                }
                for e in effects
            ],
        }

        if bound:
            causal_info["finite_sample_bound"] = {
                "correctness_probability": bound.correctness_probability,
                "sample_size": bound.sample_size,
                "significance_level": bound.significance_level,
            }

        return causal_info

    def _extract_bayesian(self, result: Any) -> Dict:
        """Extract Bayesian evidence from inference result."""
        posteriors = getattr(result, 'posteriors', {})
        return {
            "posteriors": {
                case_id: {
                    "distribution": post.distribution,
                    "bayes_factor": post.bayes_factor,
                    "treewidth": post.treewidth,
                    "circuit_size": post.circuit_size,
                }
                for case_id, post in posteriors.items()
            },
            "inference_method": getattr(result, 'method', 'exact_arithmetic_circuit'),
            "mean_treewidth": getattr(result, 'mean_treewidth', 0),
            "mean_circuit_size": getattr(result, 'mean_circuit_size', 0),
        }

    def _extract_temporal(self, result: Any) -> List[Dict]:
        """Extract temporal violations."""
        violations = getattr(result, 'violations', [])
        return [
            {
                "formula_name": v.formula_name,
                "timestamp": v.timestamp,
                "signal_value": v.signal_value,
                "threshold": v.threshold,
                "fragment_verified": getattr(result, 'fragment_verified', False),
            }
            for v in violations
        ]

    def _extract_proofs(self, result: Any) -> List[Dict]:
        """Extract proof certificates."""
        certs = getattr(result, 'certificates', [])
        return [
            {
                "status": c.status.name,
                "solver": c.solver_name,
                "num_variables": c.num_variables,
                "num_constraints": c.num_constraints,
                "precision_bits": c.precision_bits,
                "proof_time_seconds": c.proof_time_seconds,
                "level": c.level,
                "certificate_hash": c.certificate_hash,
                "translation_validated": (
                    c.translation_validation.valid
                    if c.translation_validation else None
                ),
            }
            for c in certs
        ]

    def _extract_soundness(self, causal, bayesian, temporal, proof) -> Dict:
        """Run composition soundness check and extract results."""
        from vmee.composition.soundness import CompositionFramework

        framework = CompositionFramework()
        result = framework.check_soundness(causal, bayesian, temporal, proof)

        return {
            "sound": result.sound,
            "object_level_claims": result.object_level_claims,
            "meta_level_claims": result.meta_level_claims,
            "compatibility": {
                "status": result.compatibility.status.name,
                "causal_bayesian": result.compatibility.causal_bayesian_compatible,
                "bayesian_temporal": result.compatibility.bayesian_temporal_compatible,
                "identification_inference": result.compatibility.identification_inference_compatible,
            },
        }


class BundleVerifier:
    """Verifies evidence bundles for correctness."""

    def verify_bundle(self, bundle_path: str) -> VerificationResult:
        """Verify an evidence bundle from a JSON file."""
        with open(bundle_path, "r") as f:
            bundle = json.load(f)
        return self.verify(bundle)

    def verify(self, bundle: Dict) -> VerificationResult:
        """Verify an evidence bundle."""
        checks = []

        # Check structure
        required_keys = ["version", "causal_subgraph", "bayesian_evidence",
                         "temporal_violations", "proof_certificates"]
        for key in required_keys:
            checks.append(VerificationCheck(
                name=f"Structure: {key}",
                passed=key in bundle,
                status="present" if key in bundle else "missing",
            ))

        # Check posteriors are valid distributions
        bayesian = bundle.get("bayesian_evidence", {})
        posteriors = bayesian.get("posteriors", {})
        for case_id, post in posteriors.items():
            dist = post.get("distribution", {})
            total = sum(dist.values())
            valid = abs(total - 1.0) < 1e-6
            checks.append(VerificationCheck(
                name=f"Posterior normalization: {case_id}",
                passed=valid,
                status=f"sum={total:.6f}" if valid else f"INVALID sum={total:.6f}",
            ))

        # Check proof certificates
        proofs = bundle.get("proof_certificates", [])
        for i, proof in enumerate(proofs):
            checks.append(VerificationCheck(
                name=f"Proof {i}: {proof.get('solver', 'unknown')}",
                passed=proof.get("status") == "PROVED",
                status=proof.get("status", "UNKNOWN"),
            ))

        # Check soundness
        soundness = bundle.get("soundness", {})
        checks.append(VerificationCheck(
            name="Compositional soundness",
            passed=soundness.get("sound", False),
            status="SOUND" if soundness.get("sound") else "NOT SOUND",
        ))

        is_valid = all(c.passed for c in checks)
        return VerificationResult(
            is_valid=is_valid,
            checks=checks,
            bundle_hash=bundle.get("bundle_hash", ""),
        )
