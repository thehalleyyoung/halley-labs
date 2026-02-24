"""
Bug Class Registry for Intent-Apparent Bug Detection.

Loads bug class definitions from bugclasses.jsonl and provides a structured
API for pattern checkers to look up bug classes by category.

Each bug class represents a pattern where the programmer's intent is apparent
from the code structure but the code doesn't match that intent — producing
a silent semantic error rather than a crash.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class BugCategory(Enum):
    """Categories of intent-apparent ML bugs."""
    SHAPE = auto()          # Tensor shape mismatches, reshape errors
    DEVICE = auto()         # CPU/CUDA placement mismatches
    GRADIENT = auto()       # Broken gradient flow, detach issues
    OPTIMIZER = auto()      # Optimizer misconfiguration, param lifecycle
    SEMANTIC = auto()       # Wrong loss, wrong axis, wrong normalization
    DTYPE = auto()          # Integer truncation, fp16 overflow
    LIFECYCLE = auto()      # Module registration, parameter lifecycle
    ALIASING = auto()       # In-place aliasing, expand vs clone
    CONTROLFLOW = auto()    # Training loop protocol violations


# Map from bug class name keywords to categories
_CATEGORY_KEYWORDS: Dict[BugCategory, List[str]] = {
    BugCategory.SHAPE: [
        "shape", "reshape", "view", "flatten", "squeeze", "cat", "concat",
        "conv", "matmul", "broadcast", "spatial", "dim mismatch",
        "element-count", "fold", "unfold", "interpolat", "output_padding",
        "grid_sample", "pad",
    ],
    BugCategory.DEVICE: ["device", "cpu", "cuda", "placement"],
    BugCategory.GRADIENT: [
        "gradient", "detach", "differentiable", "backward", "grad",
        "hard gating", "non-differentiable",
    ],
    BugCategory.OPTIMIZER: [
        "optimizer", "zero_grad", "param group", "learning rate",
        "scheduler", "frozen", "unfreezing", "weight decay",
        "gradient accumulation", "gradient clipping", "gradscaler",
    ],
    BugCategory.SEMANTIC: [
        "softmax", "wrong dimension", "axis confusion", "loss",
        "cross_entropy", "bce", "attention mask", "batch_first",
        "layernorm", "batch_norm", "multihead", "activation",
        "dropout", "training state", "ignore_index",
    ],
    BugCategory.DTYPE: ["float16", "fp16", "overflow", "integer", "truncation", "dtype"],
    BugCategory.LIFECYCLE: [
        "parameter", "modulelist", "moduledict", "registered",
        "plain python", "dead param", "replaced", ".data",
    ],
    BugCategory.ALIASING: [
        "alias", "in-place", "expand", "weight sharing",
        "list multiplication", "residual", "skip",
    ],
    BugCategory.CONTROLFLOW: [
        "torchscript", "trace", "hidden state", "rnn", "leakage",
        "misaligned", "paired transform",
    ],
}


@dataclass(frozen=True)
class BugClassDef:
    """A single bug class definition loaded from bugclasses.jsonl."""
    id: int
    name: str
    code_example: str
    z3_strategy: str
    category: BugCategory

    @property
    def short_name(self) -> str:
        """Short identifier for the bug class."""
        return self.name.split("(")[0].strip().lower().replace(" ", "_")[:50]


def _classify_bug(name: str) -> BugCategory:
    """Classify a bug class name into a category using keyword matching."""
    name_lower = name.lower()
    best_cat = BugCategory.SEMANTIC  # default
    best_score = 0
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in name_lower)
        if score > best_score:
            best_score = score
            best_cat = cat
    return best_cat


class BugClassRegistry:
    """Registry of all known intent-apparent ML bug classes.

    Loads from bugclasses.jsonl and provides lookup by category, ID, or keyword.
    """

    def __init__(self, bugclasses_path: Optional[Path] = None):
        self._classes: List[BugClassDef] = []
        self._by_category: Dict[BugCategory, List[BugClassDef]] = {
            cat: [] for cat in BugCategory
        }
        self._by_id: Dict[int, BugClassDef] = {}

        if bugclasses_path is None:
            bugclasses_path = Path(__file__).parent.parent.parent / "bugclasses.jsonl"

        if bugclasses_path.exists():
            self._load(bugclasses_path)
        else:
            logger.warning(f"bugclasses.jsonl not found at {bugclasses_path}")

    def _load(self, path: Path) -> None:
        """Load bug classes from JSONL file."""
        with open(path, "r") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    category = _classify_bug(data["bug_class"])
                    bc = BugClassDef(
                        id=i,
                        name=data["bug_class"],
                        code_example=data.get("code_example", ""),
                        z3_strategy=data.get("how_z3", ""),
                        category=category,
                    )
                    self._classes.append(bc)
                    self._by_category[category].append(bc)
                    self._by_id[i] = bc
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse line {i}: {e}")

    @property
    def all_classes(self) -> List[BugClassDef]:
        return list(self._classes)

    def by_category(self, cat: BugCategory) -> List[BugClassDef]:
        return self._by_category.get(cat, [])

    def by_id(self, bug_id: int) -> Optional[BugClassDef]:
        return self._by_id.get(bug_id)

    def search(self, keyword: str) -> List[BugClassDef]:
        """Search bug classes by keyword in name."""
        kw = keyword.lower()
        return [bc for bc in self._classes if kw in bc.name.lower()]

    def categories_summary(self) -> Dict[str, int]:
        """Return count of bug classes per category."""
        return {cat.name: len(classes) for cat, classes in self._by_category.items()}
