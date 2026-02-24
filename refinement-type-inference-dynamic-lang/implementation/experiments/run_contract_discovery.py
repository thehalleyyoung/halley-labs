"""
Contract Discovery Experiment — CEGAR loop discovering shape predicates
for models with SYMBOLIC (under-specified) dimensions.

Demonstrates the CEGAR loop's core value: when input dimensions are
symbolic (e.g., ("batch", "features")), the loop iteratively discovers
what dimension values are required (e.g., features==768).

Outputs: experiments/contract_discovery_results.json
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.shape_cegar import run_shape_cegar, PREDICATE_QUALITY_THRESHOLD

RESULTS_FILE = Path(__file__).parent / "contract_discovery_results.json"

# ═══════════════════════════════════════════════════════════════════════════════
# Test cases — 8 models with symbolic (under-specified) dimensions
# ═══════════════════════════════════════════════════════════════════════════════

TEST_CASES: List[Dict[str, Any]] = [
    # ── 1. Simple MLP with symbolic input ──
    {
        "name": "mlp_symbolic_features",
        "description": (
            "nn.Linear(768, 256): input has symbolic 'features' dim. "
            "CEGAR should discover features==768."
        ),
        "has_bug": False,
        "code": """\
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 256)

    def forward(self, x):
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", "features")},
        "expected_predicates": ["x.shape[-1] == 768"],
    },
    # ── 2. CNN with symbolic channels ──
    {
        "name": "cnn_symbolic_channels",
        "description": (
            "nn.Conv2d(3, 64, 3): input has symbolic 'channels' dim. "
            "CEGAR should discover channels==3."
        ),
        "has_bug": False,
        "code": """\
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)

    def forward(self, x):
        return self.conv(x)
""",
        "input_shapes": {"x": ("batch", "channels", "height", "width")},
        "expected_predicates": ["x.shape[1] == 3"],
    },
    # ── 3. Transformer encoder with symbolic embed_dim ──
    {
        "name": "transformer_symbolic_embed",
        "description": (
            "Transformer projections with in_features=512. "
            "CEGAR should discover embed_dim==512."
        ),
        "has_bug": False,
        "code": """\
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 512)
        self.k_proj = nn.Linear(512, 512)
        self.v_proj = nn.Linear(512, 512)
        self.out_proj = nn.Linear(512, 512)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.out_proj(v)
""",
        "input_shapes": {"x": ("batch", "seq_len", "embed_dim")},
        "expected_predicates": ["x.shape[-1] == 512"],
    },
    # ── 4. Multi-layer MLP with intermediate shape discovery ──
    {
        "name": "multilayer_mlp_symbolic",
        "description": (
            "3-layer MLP (784->256->128->10) with symbolic input. "
            "CEGAR should discover features==784."
        ),
        "has_bug": False,
        "code": """\
import torch.nn as nn

class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
""",
        "input_shapes": {"x": ("batch", "features")},
        "expected_predicates": ["x.shape[-1] == 784"],
    },
    # ── 5. ResNet-like with symbolic in_channels ──
    {
        "name": "resnet_symbolic_channels",
        "description": (
            "ResNet-style block with Conv2d(64, 64, 3). "
            "CEGAR should discover in_channels==64."
        ),
        "has_bug": False,
        "code": """\
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out
""",
        "input_shapes": {"x": ("batch", "in_channels", "h", "w")},
        "expected_predicates": ["x.shape[1] == 64"],
    },
    # ── 6. Autoencoder with matching encoder/decoder dims ──
    {
        "name": "autoencoder_symbolic",
        "description": (
            "Autoencoder (784->256->64->256->784) with symbolic input. "
            "CEGAR should discover features==784."
        ),
        "has_bug": False,
        "code": """\
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Linear(784, 256)
        self.enc2 = nn.Linear(256, 64)
        self.dec1 = nn.Linear(64, 256)
        self.dec2 = nn.Linear(256, 784)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.enc1(x))
        x = self.relu(self.enc2(x))
        x = self.relu(self.dec1(x))
        return self.dec2(x)
""",
        "input_shapes": {"x": ("batch", "features")},
        "expected_predicates": ["x.shape[-1] == 784"],
    },
    # ── 7. Model with BOTH a bug AND symbolic dims ──
    {
        "name": "buggy_mlp_symbolic",
        "description": (
            "MLP with symbolic input AND a real bug: "
            "fc1 outputs 256 but fc2 expects 128. "
            "CEGAR should find the bug AND discover input contracts."
        ),
        "has_bug": True,
        "code": """\
import torch.nn as nn

class BuggyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
""",
        "input_shapes": {"x": ("batch", "features")},
        "expected_predicates": [],
    },
    # ── 8. Correct model — only symbolic dims, no bugs ──
    {
        "name": "wide_net_symbolic",
        "description": (
            "Wide network (1024->512->256->128) with symbolic input. "
            "CEGAR should discover all contracts and prove safe."
        ),
        "has_bug": False,
        "code": """\
import torch.nn as nn

class WideNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
""",
        "input_shapes": {"x": ("batch", "features")},
        "expected_predicates": ["x.shape[-1] == 1024"],
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_one(tc: Dict[str, Any]) -> Dict[str, Any]:
    """Run CEGAR with quality filter on a single test case."""
    t0 = time.monotonic()
    try:
        result = run_shape_cegar(
            tc["code"],
            input_shapes=tc["input_shapes"],
            enable_quality_filter=True,
        )
        predicates_pretty = [p.pretty() for p in result.discovered_predicates]
        contracts_pretty = [c.pretty() for c in result.contracts_inferred]
        status = result.final_status.name
        iterations = result.iterations
        real_bugs = len(result.real_bugs)
        quality_report = result.predicate_quality_report or {}
    except Exception as e:
        predicates_pretty = []
        contracts_pretty = []
        status = f"ERROR: {e}"
        iterations = 0
        real_bugs = 0
        quality_report = {}

    elapsed = (time.monotonic() - t0) * 1000

    return {
        "name": tc["name"],
        "description": tc["description"],
        "has_bug": tc["has_bug"],
        "status": status,
        "iterations": iterations,
        "predicates_discovered": predicates_pretty,
        "num_predicates": len(predicates_pretty),
        "contracts_inferred": contracts_pretty,
        "num_contracts": len(contracts_pretty),
        "real_bugs_found": real_bugs,
        "expected_predicates": tc["expected_predicates"],
        "time_ms": round(elapsed, 2),
        "predicate_quality_report": quality_report,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 76)
    print("  CEGAR Contract Discovery — Symbolic Dimension Predicate Inference")
    print("=" * 76)

    results: List[Dict[str, Any]] = []

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"\n{'─' * 76}")
        print(f"  [{i}/{len(TEST_CASES)}] {tc['name']}")
        print(f"  {tc['description']}")
        print(f"  input_shapes = {tc['input_shapes']}")
        print(f"{'─' * 76}")

        r = run_one(tc)
        results.append(r)

        # Pretty-print result
        mark = "✓" if (
            (r["has_bug"] and r["status"] == "REAL_BUG_FOUND")
            or (not r["has_bug"] and r["status"] == "SAFE")
        ) else "✗"

        print(f"  {mark} Status:      {r['status']}")
        print(f"    Iterations:  {r['iterations']}")
        print(f"    Predicates:  {r['num_predicates']}")
        for p in r["predicates_discovered"]:
            print(f"      → {p}")
        print(f"    Contracts:   {r['num_contracts']}")
        for c in r["contracts_inferred"]:
            print(f"      → {c}")
        if r["real_bugs_found"]:
            print(f"    Real bugs:   {r['real_bugs_found']}")
        print(f"    Time:        {r['time_ms']:.1f}ms")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 76}")
    print("  SUMMARY")
    print(f"{'=' * 76}")

    total_preds = sum(r["num_predicates"] for r in results)
    total_contracts = sum(r["num_contracts"] for r in results)
    total_iters = sum(r["iterations"] for r in results)
    total_time = sum(r["time_ms"] for r in results)
    safe_count = sum(1 for r in results if r["status"] == "SAFE")
    bug_count = sum(1 for r in results if r["status"] == "REAL_BUG_FOUND")
    correct = sum(
        1 for r in results
        if (r["has_bug"] and r["status"] == "REAL_BUG_FOUND")
        or (not r["has_bug"] and r["status"] == "SAFE")
    )

    print(f"  Models tested:          {len(results)}")
    print(f"  Correct verdicts:       {correct}/{len(results)}")
    print(f"  Proven SAFE:            {safe_count}")
    print(f"  REAL_BUG_FOUND:         {bug_count}")
    print(f"  Total predicates:       {total_preds}")
    print(f"  Total contracts:        {total_contracts}")
    print(f"  Total CEGAR iterations: {total_iters}")
    print(f"  Total time:             {total_time:.1f}ms")

    # ── Write JSON ───────────────────────────────────────────────────────
    output = {
        "experiment": "contract_discovery",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_test_cases": len(TEST_CASES),
        "predicate_quality_threshold": PREDICATE_QUALITY_THRESHOLD,
        "summary": {
            "models_tested": len(results),
            "correct_verdicts": correct,
            "proven_safe": safe_count,
            "bugs_found": bug_count,
            "total_predicates_discovered": total_preds,
            "total_contracts_inferred": total_contracts,
            "total_cegar_iterations": total_iters,
            "total_time_ms": round(total_time, 2),
        },
        "per_model": results,
    }

    RESULTS_FILE.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
