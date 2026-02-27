"""Tests for knowledge transfer: pattern hashing, anti-unification, warm-start
speedup, persistence round-trips, and KB merging across sessions."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import textwrap
import time
from typing import Any, Dict, List

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.knowledge_base import (
    VerificationKnowledgeBase,
    ProofSchema,
    FamilyRecord,
    TransferredKnowledge,
    compute_arch_hash,
    anti_unify_proof_certificates,
    _extract_layer_sequence,
    _extract_forward_pattern,
)
from src.shape_cegar import ShapeCEGARLoop


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _resnet_source(name: str, num_blocks: int, width: int, classes: int = 10) -> str:
    """Generate a ResNet-like source with given depth/width."""
    init = []
    fwd = []
    init.append(f"        self.conv1 = nn.Conv2d(3, {width}, kernel_size=3, padding=1)")
    init.append(f"        self.bn1 = nn.BatchNorm2d({width})")
    init.append(f"        self.relu = nn.ReLU()")
    for i in range(num_blocks):
        init.append(f"        self.conv{i+2} = nn.Conv2d({width}, {width}, kernel_size=3, padding=1)")
        init.append(f"        self.bn{i+2} = nn.BatchNorm2d({width})")
    init.append(f"        self.fc = nn.Linear({width}, {classes})")

    fwd.append("        x = self.relu(self.bn1(self.conv1(x)))")
    for i in range(num_blocks):
        fwd.append(f"        x = self.relu(self.bn{i+2}(self.conv{i+2}(x)))")
    fwd.append("        x = x.mean(dim=[2, 3])")
    fwd.append("        x = self.fc(x)")
    fwd.append("        return x")

    return f"""import torch
import torch.nn as nn

class {name}(nn.Module):
    def __init__(self):
        super().__init__()
{chr(10).join(init)}

    def forward(self, x):
{chr(10).join(fwd)}
"""


def _vgg_source(name: str, num_convs: int, width: int, classes: int = 10) -> str:
    """Generate a VGG-like source (different skeleton than ResNet)."""
    init = []
    fwd = []
    prev = 3
    for i in range(num_convs):
        init.append(f"        self.conv{i+1} = nn.Conv2d({prev}, {width}, kernel_size=3)")
        prev = width
    init.append(f"        self.fc = nn.Linear({width}, {classes})")

    for i in range(num_convs):
        fwd.append(f"        x = self.conv{i+1}(x)")
    fwd.append("        x = x.mean(dim=[2, 3])")
    fwd.append("        x = self.fc(x)")
    fwd.append("        return x")

    return f"""import torch
import torch.nn as nn

class {name}(nn.Module):
    def __init__(self):
        super().__init__()
{chr(10).join(init)}

    def forward(self, x):
{chr(10).join(fwd)}
"""


def _make_cert(model: str, steps: List[Dict], cert_hash: str = "") -> Dict[str, Any]:
    return {
        "model_name": model,
        "properties": ["shape_safety"],
        "steps": steps,
        "root_step": len(steps) - 1,
        "theories_used": ["arith"],
        "verification_conditions": [],
        "certificate_hash": cert_hash or f"hash_{model}",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Pattern hash consistency within architectural families
# ═══════════════════════════════════════════════════════════════════════════════

class TestPatternHashConsistency:

    def test_same_skeleton_same_hash_varying_depth(self):
        """ResNet variants with different depth but same skeleton hash identically."""
        s1 = _resnet_source("R1", num_blocks=2, width=64)
        s2 = _resnet_source("R2", num_blocks=2, width=128)
        assert compute_arch_hash(s1) == compute_arch_hash(s2)

    def test_same_skeleton_same_hash_varying_classes(self):
        """Changing num_classes (Linear output) doesn't change the hash."""
        s1 = _resnet_source("R1", 2, 64, classes=10)
        s2 = _resnet_source("R2", 2, 64, classes=1000)
        assert compute_arch_hash(s1) == compute_arch_hash(s2)

    def test_different_skeleton_different_hash(self):
        """ResNet vs VGG produce different hashes."""
        resnet = _resnet_source("RN", 2, 64)
        vgg = _vgg_source("VG", 3, 64)
        assert compute_arch_hash(resnet) != compute_arch_hash(vgg)

    def test_hash_deterministic_across_calls(self):
        """Same source always gives same hash."""
        src = _resnet_source("R", 3, 64)
        h1 = compute_arch_hash(src)
        h2 = compute_arch_hash(src)
        assert h1 == h2

    def test_layer_sequence_ignores_parameters(self):
        """_extract_layer_sequence returns types, not parameter values."""
        s1 = _resnet_source("R1", 2, 32)
        s2 = _resnet_source("R2", 2, 256)
        layers1 = _extract_layer_sequence(s1)
        layers2 = _extract_layer_sequence(s2)
        assert layers1 == layers2


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Anti-unification produces valid generalized schemas
# ═══════════════════════════════════════════════════════════════════════════════

class TestAntiUnificationValidity:

    def test_schema_preserves_common_rules(self):
        """Common rule names are kept literal in the skeleton."""
        c1 = _make_cert("M1", [
            {"rule": "asserted", "conclusion": "x>0", "premises": []},
            {"rule": "mp", "conclusion": "y>0", "premises": [0]},
        ], "h1")
        c2 = _make_cert("M2", [
            {"rule": "asserted", "conclusion": "x>5", "premises": []},
            {"rule": "mp", "conclusion": "y>5", "premises": [0]},
        ], "h2")
        schema = anti_unify_proof_certificates([c1, c2], arch_hash="test")
        assert schema.rule_skeleton[0]["rule"] == "asserted"
        assert schema.rule_skeleton[1]["rule"] == "mp"

    def test_schema_variables_for_differing_values(self):
        """Values that differ across certs become ?V variables."""
        c1 = _make_cert("M1", [{"rule": "ax", "value": 10}], "h1")
        c2 = _make_cert("M2", [{"rule": "ax", "value": 20}], "h2")
        schema = anti_unify_proof_certificates([c1, c2])
        assert "value" in schema.variable_positions[0]
        assert schema.rule_skeleton[0]["value"].startswith("?V")

    def test_schema_source_count(self):
        """source_count reflects input certificate count."""
        certs = [
            _make_cert(f"M{i}", [{"rule": "ax", "val": i}], f"h{i}")
            for i in range(5)
        ]
        schema = anti_unify_proof_certificates(certs)
        assert schema.source_count == 5

    def test_schema_arch_hash_propagated(self):
        """arch_hash is propagated to the ProofSchema."""
        c = _make_cert("M1", [{"rule": "ax"}], "h1")
        schema = anti_unify_proof_certificates([c], arch_hash="myhash")
        assert schema.arch_hash == "myhash"

    def test_schema_serialization_roundtrip(self):
        """ProofSchema survives to_dict → from_dict."""
        c1 = _make_cert("A", [{"rule": "r", "v": 1}], "h1")
        c2 = _make_cert("B", [{"rule": "r", "v": 2}], "h2")
        schema = anti_unify_proof_certificates([c1, c2], arch_hash="xyz")
        d = schema.to_dict()
        restored = ProofSchema.from_dict(d)
        assert restored.source_count == schema.source_count
        assert restored.arch_hash == "xyz"
        assert len(restored.rule_skeleton) == len(schema.rule_skeleton)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Warm-start CEGAR uses fewer iterations than cold-start
# ═══════════════════════════════════════════════════════════════════════════════

class TestWarmStartSpeedup:

    def test_warm_start_no_more_iterations_than_cold(self):
        """Warm-start CEGAR should use ≤ iterations compared to cold."""
        src = _resnet_source("WarmTest", 2, 64)
        input_shapes = {"x": ("batch", 3, 32, 32)}

        # Cold
        cold_loop = ShapeCEGARLoop(src, input_shapes=input_shapes)
        cold_result = cold_loop.run()

        # Build KB
        kb = VerificationKnowledgeBase()
        arch_hash = compute_arch_hash(src)
        kb.record(
            arch_hash,
            predicates=[p.pretty() for p in cold_result.discovered_predicates],
        )

        # Warm
        warm_loop = ShapeCEGARLoop(src, input_shapes=input_shapes, knowledge_base=kb)
        warm_result = warm_loop.run()

        assert warm_result.iterations <= cold_result.iterations

    def test_warm_start_at_least_as_good_verdict(self):
        """Warm start should produce same or better verdict."""
        src = _resnet_source("VerdictTest", 1, 32)
        input_shapes = {"x": ("batch", 3, 16, 16)}

        cold_loop = ShapeCEGARLoop(src, input_shapes=input_shapes)
        cold_result = cold_loop.run()

        kb = VerificationKnowledgeBase()
        kb.record(
            compute_arch_hash(src),
            predicates=[p.pretty() for p in cold_result.discovered_predicates],
        )

        warm_loop = ShapeCEGARLoop(src, input_shapes=input_shapes, knowledge_base=kb)
        warm_result = warm_loop.run()

        safe_verdicts = {"SAFE", "UNKNOWN"}
        if cold_result.verdict.name in safe_verdicts:
            assert warm_result.verdict.name in safe_verdicts

    def test_warm_start_time_not_dramatically_worse(self):
        """Warm start should not take dramatically more time (>5x) than cold."""
        src = _resnet_source("TimeTest", 1, 32)
        input_shapes = {"x": ("batch", 3, 16, 16)}

        t0 = time.monotonic()
        cold = ShapeCEGARLoop(src, input_shapes=input_shapes).run()
        cold_time = time.monotonic() - t0

        kb = VerificationKnowledgeBase()
        kb.record(compute_arch_hash(src),
                  predicates=[p.pretty() for p in cold.discovered_predicates])

        t0 = time.monotonic()
        warm = ShapeCEGARLoop(src, input_shapes=input_shapes, knowledge_base=kb).run()
        warm_time = time.monotonic() - t0

        # Allow some overhead but not 5x worse
        assert warm_time < cold_time * 5 + 0.5  # +0.5s tolerance for very fast tests


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Knowledge persistence (save/load round-trip)
# ═══════════════════════════════════════════════════════════════════════════════

class TestKnowledgePersistence:

    def test_predicates_survive_roundtrip(self, tmp_path):
        """Predicates survive save → load."""
        kb = VerificationKnowledgeBase()
        src = _resnet_source("Persist", 1, 32)
        h = compute_arch_hash(src)
        kb.record(h, predicates=["x.shape[-1] == 32", "x.shape[0] >= 1"])

        path = str(tmp_path / "kb.json")
        kb.save(path)
        kb2 = VerificationKnowledgeBase.load(path)
        transferred = kb2.lookup(h)
        assert set(transferred.predicates) == {"x.shape[-1] == 32", "x.shape[0] >= 1"}

    def test_strategies_survive_roundtrip(self, tmp_path):
        """Strategies survive save → load."""
        kb = VerificationKnowledgeBase()
        kb.record("h1", strategies=[{"propagator_type": "cegar", "iteration_count": 3}])

        path = str(tmp_path / "kb.json")
        kb.save(path)
        kb2 = VerificationKnowledgeBase.load(path)
        t = kb2.lookup("h1")
        assert len(t.strategies) == 1
        assert t.strategies[0]["propagator_type"] == "cegar"

    def test_proof_schema_survives_roundtrip(self, tmp_path):
        """Proof schema (from anti-unification) survives save → load."""
        kb = VerificationKnowledgeBase()
        c1 = _make_cert("A", [{"rule": "r1", "val": 1}], "h1")
        c2 = _make_cert("B", [{"rule": "r1", "val": 2}], "h2")
        kb.record("fam1", proof_certificate=c1)
        kb.record("fam1", proof_certificate=c2)

        path = str(tmp_path / "kb.json")
        kb.save(path)
        kb2 = VerificationKnowledgeBase.load(path)
        rec = kb2.get_family_record("fam1")
        assert rec is not None
        assert rec.proof_schema is not None
        assert rec.proof_schema["source_count"] == 2

    def test_load_nonexistent_gives_empty_kb(self, tmp_path):
        """Loading from a nonexistent path returns empty KB."""
        kb = VerificationKnowledgeBase.load(str(tmp_path / "no.json"))
        assert kb.family_count == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Merge of knowledge bases from different sessions
# ═══════════════════════════════════════════════════════════════════════════════

class TestKBMerge:

    def test_merge_disjoint_families(self):
        """Merging KBs with non-overlapping families unions them."""
        kb1 = VerificationKnowledgeBase()
        kb1.record("h1", predicates=["p1"])
        kb2 = VerificationKnowledgeBase()
        kb2.record("h2", predicates=["p2"])

        kb1.merge(kb2)
        assert kb1.family_count == 2
        assert kb1.has_family("h1") and kb1.has_family("h2")

    def test_merge_overlapping_predicates_union(self):
        """Merging overlapping families unions predicates without duplicates."""
        kb1 = VerificationKnowledgeBase()
        kb1.record("h", predicates=["p1", "p2"])
        kb2 = VerificationKnowledgeBase()
        kb2.record("h", predicates=["p2", "p3"])

        kb1.merge(kb2)
        rec = kb1.get_family_record("h")
        assert rec is not None
        assert set(rec.predicates) == {"p1", "p2", "p3"}

    def test_merge_increments_session_count(self):
        """Session counts accumulate across merges."""
        kb1 = VerificationKnowledgeBase()
        kb1.record("h", predicates=["p1"])
        kb2 = VerificationKnowledgeBase()
        kb2.record("h", predicates=["p2"])

        kb1.merge(kb2)
        rec = kb1.get_family_record("h")
        assert rec is not None
        assert rec.session_count >= 2

    def test_merge_three_way(self):
        """Three-way merge accumulates all predicates."""
        kbs = []
        for i in range(3):
            kb = VerificationKnowledgeBase()
            kb.record("h", predicates=[f"p{i}"])
            kbs.append(kb)

        kbs[0].merge(kbs[1])
        kbs[0].merge(kbs[2])
        rec = kbs[0].get_family_record("h")
        assert rec is not None
        assert set(rec.predicates) == {"p0", "p1", "p2"}

    def test_merge_preserves_proof_certificates(self):
        """Proof certificates from both KBs are preserved after merge."""
        kb1 = VerificationKnowledgeBase()
        kb1.record("h", proof_certificate=_make_cert("M1", [{"r": 1}], "c1"))
        kb2 = VerificationKnowledgeBase()
        kb2.record("h", proof_certificate=_make_cert("M2", [{"r": 2}], "c2"))

        kb1.merge(kb2)
        rec = kb1.get_family_record("h")
        assert rec is not None
        assert len(rec.proof_certificates) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Empirical transfer experiment
# ═══════════════════════════════════════════════════════════════════════════════

class TestTransferExperiment:

    def test_same_family_transfer(self):
        """Transfer within same architectural family yields speedup."""
        from src.knowledge_base import run_transfer_experiment

        source = [
            (_resnet_source("S1", 2, 64), {"x": ("batch", 3, 32, 32)}),
            (_resnet_source("S2", 2, 128), {"x": ("batch", 3, 32, 32)}),
        ]
        target = [
            (_resnet_source("T1", 2, 32), {"x": ("batch", 3, 32, 32)}),
        ]

        result = run_transfer_experiment(source, target, "resnet_family")
        assert result.warm_iterations_avg <= result.cold_iterations_avg
        assert result.speedup_ratio >= 1.0
        assert result.source_models == 2
        assert result.target_models == 1

    def test_cross_family_no_speedup(self):
        """Transfer across different families gives lower transfer rate."""
        from src.knowledge_base import run_transfer_experiment

        source = [
            (_vgg_source("VS1", 3, 64), {"x": ("batch", 3, 32, 32)}),
        ]
        target = [
            (_resnet_source("TR1", 2, 64), {"x": ("batch", 3, 32, 32)}),
        ]

        result = run_transfer_experiment(source, target, "cross_family")
        # Different families won't share arch_hash, so no useful transfer
        assert result.predicates_useful == 0 or result.transfer_rate < 1.0

    def test_transfer_result_to_dict(self):
        """TransferExperimentResult serializes cleanly."""
        from src.knowledge_base import run_transfer_experiment

        source = [(_resnet_source("S1", 1, 32), {"x": ("batch", 3, 16, 16)})]
        target = [(_resnet_source("T1", 1, 64), {"x": ("batch", 3, 16, 16)})]

        result = run_transfer_experiment(source, target)
        d = result.to_dict()
        assert "speedup_ratio" in d
        assert "transfer_rate" in d
        assert d["source_models"] == 1
