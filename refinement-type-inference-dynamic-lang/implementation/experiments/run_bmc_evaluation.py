"""
Comprehensive evaluation of the TensorGuard constraint-based verifier with
product theory T_shape × T_device × T_phase.

Evaluates:
1. Architecture verification on diverse nn.Module models
2. Cross-domain bug detection (shape + device + phase)
3. CEGAR shape predicate discovery
4. Z3 solver statistics
5. Symbolic shape verification
6. Comparison with baseline (pattern matching only)
"""

import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_checker import (
    verify_model, extract_computation_graph, BoundedModelChecker, ConstraintVerifier,
    Device, Phase, VerificationResult
)
from src.shape_cegar import run_shape_cegar


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark models
# ═══════════════════════════════════════════════════════════════════════════════

MODELS = {
    # --- Safe models ---
    "simple_linear": {
        "source": """
import torch.nn as nn
class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 10)},
        "expected_safe": True,
        "category": "basic",
    },

    "mlp_3layer": {
        "source": """
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
""",
        "input_shapes": {"x": ("batch", 784)},
        "expected_safe": True,
        "category": "mlp",
    },

    "cnn_classifier": {
        "source": """
import torch.nn as nn
class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 54 * 54, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
""",
        "input_shapes": {"x": ("batch", 3, 224, 224)},
        "expected_safe": True,
        "category": "cnn",
    },

    "transformer_encoder": {
        "source": """
import torch.nn as nn
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(512, 8)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self.ff1 = nn.Linear(512, 2048)
        self.ff2 = nn.Linear(2048, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        x = self.norm1(x)
        x = self.ff1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.ff2(x)
        x = self.norm2(x)
        return x
""",
        "input_shapes": {"x": ("batch", "seq_len", 512)},
        "expected_safe": True,
        "category": "transformer",
    },

    "resnet_block": {
        "source": """
import torch.nn as nn
class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu(x)
        return x
""",
        "input_shapes": {"x": ("batch", 64, 32, 32)},
        "expected_safe": True,
        "category": "resnet",
    },

    "autoencoder": {
        "source": """
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
        x = self.dec2(x)
        return x
""",
        "input_shapes": {"x": ("batch", 784)},
        "expected_safe": True,
        "category": "autoencoder",
    },

    "embedding_classifier": {
        "source": """
import torch.nn as nn
class EmbeddingClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10000, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 5)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        x = self.embed(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
""",
        "input_shapes": {"x": ("batch", "seq_len")},
        "expected_safe": True,
        "category": "nlp",
    },

    "gan_generator": {
        "source": """
import torch.nn as nn
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 784)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)
    def forward(self, z):
        z = self.relu(self.bn1(self.fc1(z)))
        z = self.relu(self.bn2(self.fc2(z)))
        z = self.relu(self.bn3(self.fc3(z)))
        z = self.fc4(z)
        return z
""",
        "input_shapes": {"z": ("batch", 100)},
        "expected_safe": True,
        "category": "gan",
    },

    "unet_block": {
        "source": """
import torch.nn as nn
class UNetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x
""",
        "input_shapes": {"x": ("batch", 3, 256, 256)},
        "expected_safe": True,
        "category": "unet",
    },

    "lstm_classifier": {
        "source": """
import torch.nn as nn
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(5000, 128)
        self.lstm = nn.LSTM(128, 256)
        self.fc = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = self.embed(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
""",
        "input_shapes": {"x": ("batch", "seq_len")},
        "expected_safe": True,
        "category": "rnn",
    },

    # --- Buggy models (shape errors) ---
    "shape_bug_linear": {
        "source": """
import torch.nn as nn
class ShapeBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
""",
        "input_shapes": {"x": ("batch", 768)},
        "expected_safe": False,
        "bug_type": "shape_mismatch",
        "category": "bug_shape",
    },

    "shape_bug_conv": {
        "source": """
import torch.nn as nn
class ConvBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(16, 64, 3)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "expected_safe": False,
        "bug_type": "channel_mismatch",
        "category": "bug_shape",
    },

    "shape_bug_matmul": {
        "source": """
import torch.nn as nn
class MatmulBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(100, 50)
    def forward(self, x):
        a = self.fc1(x)
        b = self.fc2(x)
        return a @ b
""",
        "input_shapes": {"x": ("batch", 100)},
        "expected_safe": False,
        "bug_type": "matmul_inner_dim",
        "category": "bug_shape",
    },

    # --- Symbolic shape models ---
    "symbolic_mlp": {
        "source": """
import torch.nn as nn
class SymbolicMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
""",
        "input_shapes": {"x": ("batch", "features")},
        "expected_safe": None,  # depends on features
        "category": "symbolic",
    },

    "symbolic_transformer": {
        "source": """
import torch.nn as nn
class SymTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(768)
        self.fc1 = nn.Linear(768, 3072)
        self.fc2 = nn.Linear(3072, 768)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        x = self.norm(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
""",
        "input_shapes": {"x": ("batch", "seq_len", "features")},
        "expected_safe": None,
        "category": "symbolic",
    },

    # --- Phase-dependent models ---
    "phase_dropout_model": {
        "source": """
import torch.nn as nn
class PhaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.bn(x)
        x = self.fc2(x)
        return x
""",
        "input_shapes": {"x": ("batch", 100)},
        "expected_safe": True,
        "category": "phase",
    },

    # --- Deep models ---
    "deep_mlp_10layer": {
        "source": """
import torch.nn as nn
class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 256)
        self.fc7 = nn.Linear(256, 256)
        self.fc8 = nn.Linear(256, 128)
        self.fc9 = nn.Linear(128, 64)
        self.fc10 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        x = self.relu(self.fc8(x))
        x = self.relu(self.fc9(x))
        x = self.fc10(x)
        return x
""",
        "input_shapes": {"x": ("batch", 512)},
        "expected_safe": True,
        "category": "deep",
    },

    "wide_resnet_block": {
        "source": """
import torch.nn as nn
class WideResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 256, 3)
        self.conv2 = nn.Conv2d(256, 256, 3)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = self.relu(x)
        return x
""",
        "input_shapes": {"x": ("batch", 256, 16, 16)},
        "expected_safe": True,
        "category": "resnet",
    },

    # Additional safe/buggy models for comprehensive evaluation
    "vgg_block": {
        "source": """
import torch.nn as nn
class VGGBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x
""",
        "input_shapes": {"x": ("batch", 3, 224, 224)},
        "expected_safe": True,
        "category": "cnn",
    },

    "shape_bug_deep": {
        "source": """
import torch.nn as nn
class DeepShapeBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 200)
        self.fc2 = nn.Linear(200, 300)
        self.fc3 = nn.Linear(300, 400)
        self.fc4 = nn.Linear(500, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
""",
        "input_shapes": {"x": ("batch", 100)},
        "expected_safe": False,
        "bug_type": "shape_mismatch_deep",
        "category": "bug_shape",
    },
}


def run_verification_experiments():
    """Run verification on all benchmark models."""
    results = {}
    total_z3_queries = 0
    total_z3_time = 0.0

    print("=" * 80)
    print("TensorGuard Constraint-Based Verifier — Comprehensive Evaluation")
    print("Product Theory: T_shape × T_device × T_phase")
    print("=" * 80)
    print()

    for name, spec in MODELS.items():
        source = spec["source"]
        input_shapes = spec["input_shapes"]
        expected = spec.get("expected_safe")

        # Verify in TRAIN mode
        result_train = verify_model(
            source, input_shapes,
            default_phase=Phase.TRAIN,
        )

        # Verify in EVAL mode
        result_eval = verify_model(
            source, input_shapes,
            default_phase=Phase.EVAL,
        )

        cert = result_train.certificate
        z3_queries = cert.z3_queries if cert else 0
        z3_time = cert.z3_total_time_ms if cert else 0.0
        z3_unsat = cert.z3_unsat_count if cert else 0
        z3_sat = cert.z3_sat_count if cert else 0
        theories = cert.theories_used if cert else []
        domains = cert.product_domains if cert else []

        # For unsafe models, get stats from the counterexample run
        if not result_train.safe and result_train.counterexample:
            # Re-extract from the checker directly
            try:
                graph = extract_computation_graph(source)
                checker = ConstraintVerifier(
                    graph, input_shapes,
                    default_phase=Phase.TRAIN,
                )
                _ = checker.verify()
                z3_queries = checker.ctx._query_count if hasattr(checker.ctx, '_query_count') else z3_queries
            except Exception:
                pass

        total_z3_queries += z3_queries
        total_z3_time += z3_time

        correct = None
        if expected is not None:
            correct = (result_train.safe == expected)

        status = "✓ SAFE" if result_train.safe else "✗ UNSAFE"
        match_str = ""
        if correct is not None:
            match_str = " [CORRECT]" if correct else " [INCORRECT]"

        phase_diff = result_train.safe != result_eval.safe

        print(f"  {name:30s} {status:12s} "
              f"Z3={z3_queries:4d}q/{z3_time:7.1f}ms "
              f"({z3_unsat}U/{z3_sat}S) "
              f"phase_diff={'YES' if phase_diff else 'no ':3s}"
              f"{match_str}")

        results[name] = {
            "model_name": name,
            "category": spec["category"],
            "safe_train": result_train.safe,
            "safe_eval": result_eval.safe,
            "expected_safe": expected,
            "correct": correct,
            "phase_sensitive": phase_diff,
            "z3_queries": z3_queries,
            "z3_time_ms": z3_time,
            "z3_unsat": z3_unsat,
            "z3_sat": z3_sat,
            "theories": theories,
            "product_domains": domains,
            "verification_time_ms": result_train.verification_time_ms,
            "num_steps": (result_train.graph.num_steps
                          if result_train.graph else 0),
        }

        if not result_train.safe and result_train.counterexample:
            results[name]["violations"] = [
                v.message for v in result_train.counterexample.violations
            ]
            results[name]["failing_step"] = result_train.counterexample.failing_step
            results[name]["concrete_dims"] = result_train.counterexample.concrete_dims

    print()
    print("-" * 80)

    # Summary statistics
    total = len(results)
    safe_count = sum(1 for r in results.values() if r["safe_train"])
    unsafe_count = total - safe_count
    correct_count = sum(1 for r in results.values()
                        if r["correct"] is True)
    total_evaluated = sum(1 for r in results.values()
                          if r["correct"] is not None)
    phase_sensitive = sum(1 for r in results.values()
                          if r["phase_sensitive"])

    accuracy = correct_count / total_evaluated if total_evaluated > 0 else 0

    print(f"\n  SUMMARY")
    print(f"  Models verified:          {total}")
    print(f"  Safe (train):             {safe_count}")
    print(f"  Unsafe (train):           {unsafe_count}")
    print(f"  Accuracy:                 {correct_count}/{total_evaluated} = {accuracy:.1%}")
    print(f"  Phase-sensitive models:   {phase_sensitive}")
    print(f"  Total Z3 queries:         {total_z3_queries}")
    print(f"  Total Z3 time:            {total_z3_time:.1f}ms")
    print(f"  Avg queries/model:        {total_z3_queries/total:.1f}")
    print(f"  Avg Z3 time/model:        {total_z3_time/total:.1f}ms")
    print(f"  Product domains:          T_shape × T_device × T_phase")

    return results, {
        "total_models": total,
        "safe_count": safe_count,
        "unsafe_count": unsafe_count,
        "accuracy": accuracy,
        "correct": correct_count,
        "evaluated": total_evaluated,
        "phase_sensitive": phase_sensitive,
        "total_z3_queries": total_z3_queries,
        "total_z3_time_ms": total_z3_time,
        "avg_z3_queries": total_z3_queries / total,
        "avg_z3_time_ms": total_z3_time / total,
    }


def run_cegar_experiments():
    """Evaluate CEGAR shape predicate discovery."""
    print("\n" + "=" * 80)
    print("CEGAR Shape Predicate Discovery Evaluation")
    print("=" * 80 + "\n")

    cegar_benchmarks = {
        "linear_symbolic": {
            "source": """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 10)
    def forward(self, x):
        return self.fc(x)
""",
            "input_shapes": {"x": ("batch", "features")},
            "expected_predicates": ["x.shape[-1] == 768"],
        },
        "mlp_symbolic": {
            "source": """
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
""",
            "input_shapes": {"x": ("batch", "d")},
            "expected_predicates": ["x.shape[-1] == 512"],
        },
        "conv_symbolic": {
            "source": """
import torch.nn as nn
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv(x))
        return x
""",
            "input_shapes": {"x": ("batch", "channels", "height", "width")},
            "expected_predicates": ["x.shape[1] == 3"],
        },
        "deep_symbolic": {
            "source": """
import torch.nn as nn
class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
""",
            "input_shapes": {"x": ("batch", "features")},
            "expected_predicates": ["x.shape[-1] == 1024"],
        },
        "embedding_symbolic": {
            "source": """
import torch.nn as nn
class EmbNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10000, 256)
        self.fc = nn.Linear(256, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.embed(x)
        x = self.relu(x)
        x = self.fc(x)
        return x
""",
            "input_shapes": {"x": ("batch", "seq_len")},
            "expected_predicates": [],  # no shape constraint needed on input for embedding
        },
        "multi_input_symbolic": {
            "source": """
import torch.nn as nn
class MultiInput(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(100, 50)
        self.fc_b = nn.Linear(200, 50)
        self.fc_out = nn.Linear(50, 10)
        self.relu = nn.ReLU()
    def forward(self, a):
        a = self.relu(self.fc_a(a))
        a = self.fc_out(a)
        return a
""",
            "input_shapes": {"a": ("batch", "d_a")},
            "expected_predicates": ["a.shape[-1] == 100"],
        },
        "bug_detection_cegar": {
            "source": """
import torch.nn as nn
class BuggyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 200)
        self.fc2 = nn.Linear(300, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
""",
            "input_shapes": {"x": ("batch", 100)},
            "expected_predicates": [],  # real bug, no predicates needed
            "has_bug": True,
        },
        "transformer_symbolic": {
            "source": """
import torch.nn as nn
class TransBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(768)
        self.ff1 = nn.Linear(768, 3072)
        self.ff2 = nn.Linear(3072, 768)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        x = self.norm(x)
        x = self.relu(self.ff1(x))
        x = self.dropout(x)
        x = self.ff2(x)
        return x
""",
            "input_shapes": {"x": ("batch", "seq", "dim")},
            "expected_predicates": ["x.shape[-1] == 768"],
        },
    }

    results = {}

    for name, spec in cegar_benchmarks.items():
        t0 = time.monotonic()
        result = run_shape_cegar(
            spec["source"],
            input_shapes=spec["input_shapes"],
            max_iterations=10,
        )
        elapsed = (time.monotonic() - t0) * 1000

        preds = [p.pretty() for p in result.discovered_predicates]
        has_bug = spec.get("has_bug", False)

        # Check correctness
        if has_bug:
            correct = result.has_real_bugs
        else:
            expected = spec["expected_predicates"]
            correct = all(ep in preds for ep in expected)

        status = result.final_status.name
        print(f"  {name:30s} {status:15s} "
              f"iter={result.iterations:2d} "
              f"preds={len(result.discovered_predicates):2d} "
              f"time={elapsed:7.1f}ms "
              f"{'[CORRECT]' if correct else '[WRONG]'}")
        if preds:
            for p in preds:
                print(f"    → {p}")
        if result.contracts_inferred:
            for c in result.contracts_inferred:
                print(f"    contract: {c.pretty()}")

        results[name] = {
            "status": status,
            "iterations": result.iterations,
            "predicates": preds,
            "contracts": [c.pretty() for c in result.contracts_inferred],
            "time_ms": elapsed,
            "correct": correct,
            "has_bug": has_bug,
            "real_bugs_found": result.has_real_bugs,
        }

    correct_count = sum(1 for r in results.values() if r["correct"])
    total = len(results)
    print(f"\n  CEGAR accuracy: {correct_count}/{total} = {correct_count/total:.1%}")

    return results


def run_phase_comparison():
    """Compare verification in TRAIN vs EVAL mode."""
    print("\n" + "=" * 80)
    print("Phase-Sensitive Verification (TRAIN vs EVAL)")
    print("=" * 80 + "\n")

    phase_models = {k: v for k, v in MODELS.items()
                    if v["category"] in ("phase", "mlp", "transformer", "gan")}

    results = {}
    for name, spec in phase_models.items():
        result_train = verify_model(
            spec["source"], spec["input_shapes"],
            default_phase=Phase.TRAIN,
        )
        result_eval = verify_model(
            spec["source"], spec["input_shapes"],
            default_phase=Phase.EVAL,
        )

        train_q = result_train.certificate.z3_queries if result_train.certificate else 0
        eval_q = result_eval.certificate.z3_queries if result_eval.certificate else 0

        print(f"  {name:30s} TRAIN={'SAFE' if result_train.safe else 'UNSAFE':6s} "
              f"EVAL={'SAFE' if result_eval.safe else 'UNSAFE':6s} "
              f"Z3: {train_q}/{eval_q}")

        results[name] = {
            "safe_train": result_train.safe,
            "safe_eval": result_eval.safe,
            "z3_train": train_q,
            "z3_eval": eval_q,
            "phase_diff": result_train.safe != result_eval.safe,
        }

    return results


if __name__ == "__main__":
    all_results = {}

    # 1. Architecture verification
    model_results, summary = run_verification_experiments()
    all_results["verification"] = {
        "models": model_results,
        "summary": summary,
    }

    # 2. CEGAR experiments
    cegar_results = run_cegar_experiments()
    all_results["cegar"] = cegar_results

    # 3. Phase comparison
    phase_results = run_phase_comparison()
    all_results["phase_comparison"] = phase_results

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "comprehensive_bmc_results.json"
    )
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'=' * 80}")
    print(f"Results saved to {output_path}")
    print(f"{'=' * 80}")
