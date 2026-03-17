"""
usability_oracle.benchmarks.datasets — Benchmark dataset management.

Provides loading, saving, listing, and synthetic generation of benchmark
case collections.  Includes a set of built-in test cases covering the
five bottleneck types.

NOTE — Import requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~
This module depends on project-internal packages that are **not**
independently pip-installable.  The following intra-project imports must
be resolvable before this file can be loaded:

  - usability_oracle.core.enums        (BottleneckType, RegressionVerdict)
  - usability_oracle.benchmarks.suite   (BenchmarkCase)
  - usability_oracle.benchmarks.generators (SyntheticUIGenerator)
  - usability_oracle.benchmarks.mutations  (MutationGenerator)

To satisfy them, install the package in editable/development mode from
the ``implementation/`` directory::

    cd implementation && pip install -e ".[dev]"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from usability_oracle.core.enums import BottleneckType, RegressionVerdict
from usability_oracle.benchmarks.suite import BenchmarkCase
from usability_oracle.benchmarks.generators import SyntheticUIGenerator
from usability_oracle.benchmarks.mutations import MutationGenerator


# ---------------------------------------------------------------------------
# Built-in dataset definitions
# ---------------------------------------------------------------------------

_BUILTIN_DATASETS: dict[str, list[dict[str, Any]]] = {
    "smoke": [
        {"name": "identical-forms", "category": "form", "verdict": "neutral",
         "desc": "Two identical 4-field forms"},
        {"name": "extra-button", "category": "form", "verdict": "regression",
         "mutation": "choice_paralysis", "severity": 0.4},
        {"name": "improved-layout", "category": "form", "verdict": "improvement",
         "mutation": "motor_difficulty", "severity": -0.3},
    ],
    "perceptual": [
        {"name": f"perceptual-{s}", "category": "perceptual",
         "verdict": "regression" if s > 0.2 else "neutral",
         "mutation": "perceptual_overload", "severity": s}
        for s in [0.1, 0.3, 0.5, 0.7, 0.9]
    ],
    "choice": [
        {"name": f"choice-{s}", "category": "choice",
         "verdict": "regression" if s > 0.2 else "neutral",
         "mutation": "choice_paralysis", "severity": s}
        for s in [0.1, 0.3, 0.5, 0.7, 0.9]
    ],
    "motor": [
        {"name": f"motor-{s}", "category": "motor",
         "verdict": "regression" if s > 0.2 else "neutral",
         "mutation": "motor_difficulty", "severity": s}
        for s in [0.1, 0.3, 0.5, 0.7, 0.9]
    ],
    "memory": [
        {"name": f"memory-{s}", "category": "memory",
         "verdict": "regression" if s > 0.2 else "neutral",
         "mutation": "memory_decay", "severity": s}
        for s in [0.1, 0.3, 0.5, 0.7, 0.9]
    ],
    "interference": [
        {"name": f"interference-{s}", "category": "interference",
         "verdict": "regression" if s > 0.2 else "neutral",
         "mutation": "cross_channel_interference", "severity": s}
        for s in [0.1, 0.3, 0.5, 0.7, 0.9]
    ],
}

_MUTATION_FN_MAP: dict[str, str] = {
    "perceptual_overload": "apply_perceptual_overload",
    "choice_paralysis": "apply_choice_paralysis",
    "motor_difficulty": "apply_motor_difficulty",
    "memory_decay": "apply_memory_decay",
    "cross_channel_interference": "apply_interference",
}


# ---------------------------------------------------------------------------
# DatasetManager
# ---------------------------------------------------------------------------

class DatasetManager:
    """Load, save, and generate benchmark case collections.

    Parameters:
        data_dir: Directory for cached / saved datasets.
        seed: RNG seed for reproducibility.
    """

    def __init__(self, data_dir: Path | None = None, seed: int = 42) -> None:
        self._data_dir = data_dir or Path("benchmark_data")
        self._seed = seed
        self._gen = SyntheticUIGenerator(seed=seed)
        self._mut = MutationGenerator(seed=seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, name: str) -> list[BenchmarkCase]:
        """Load a dataset by *name*.

        First checks built-in datasets, then looks in *data_dir*.
        """
        if name in _BUILTIN_DATASETS:
            return self._load_builtin(name)
        path = self._data_dir / f"{name}.json"
        if path.exists():
            return self._load_from_file(path)
        raise FileNotFoundError(f"Dataset '{name}' not found (checked built-ins and {self._data_dir})")

    def list_available(self) -> list[str]:
        """Return names of all available datasets."""
        names = sorted(_BUILTIN_DATASETS.keys())
        if self._data_dir.exists():
            for p in sorted(self._data_dir.glob("*.json")):
                n = p.stem
                if n not in names:
                    names.append(n)
        return names

    def generate_synthetic(self, n_cases: int = 20, config: dict[str, Any] | None = None) -> list[BenchmarkCase]:
        """Generate *n_cases* synthetic benchmark cases.

        Each case creates a base UI, applies a random mutation, and records
        the expected verdict.
        """
        config = config or {}
        form_fields = config.get("form_fields", 6)
        nav_items = config.get("nav_items", 8)
        cases: list[BenchmarkCase] = []

        generators = [
            ("form", lambda: self._gen.generate_form(n_fields=form_fields)),
            ("navigation", lambda: self._gen.generate_navigation(n_items=nav_items)),
            ("dashboard", lambda: self._gen.generate_dashboard(n_widgets=6)),
            ("search", lambda: self._gen.generate_search_results(n_results=8)),
            ("settings", lambda: self._gen.generate_settings_page(n_settings=8)),
        ]

        for i in range(n_cases):
            cat, gen_fn = generators[i % len(generators)]
            base = gen_fn()
            mutated, mutation_name = self._mut.apply_random_mutation(base, seed=self._seed + i)
            cases.append(BenchmarkCase(
                name=f"synthetic-{cat}-{i:04d}",
                source_a=base,
                source_b=mutated,
                expected_verdict=RegressionVerdict.REGRESSION,
                category=cat,
                metadata={"mutation": mutation_name, "index": i},
            ))

        # Add some no-change cases
        n_neutral = max(1, n_cases // 5)
        for i in range(n_neutral):
            cat, gen_fn = generators[i % len(generators)]
            tree = gen_fn()
            cases.append(BenchmarkCase(
                name=f"synthetic-{cat}-neutral-{i:04d}",
                source_a=tree,
                source_b=tree,
                expected_verdict=RegressionVerdict.NEUTRAL,
                category=cat,
                metadata={"mutation": None, "index": n_cases + i},
            ))

        return cases

    def save(self, cases: list[BenchmarkCase], path: Path) -> None:
        """Persist *cases* to a JSON file at *path*."""
        path.parent.mkdir(parents=True, exist_ok=True)
        serialised = []
        for c in cases:
            serialised.append({
                "name": c.name,
                "source_a": _serialize_source(c.source_a),
                "source_b": _serialize_source(c.source_b),
                "task_spec": c.task_spec,
                "expected_verdict": c.expected_verdict.value,
                "category": c.category,
                "metadata": c.metadata,
            })
        path.write_text(json.dumps(serialised, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_builtin(self, name: str) -> list[BenchmarkCase]:
        specs = _BUILTIN_DATASETS[name]
        cases: list[BenchmarkCase] = []
        for spec in specs:
            verdict = RegressionVerdict(spec["verdict"])
            base = self._gen.generate_form(n_fields=4)
            mutation = spec.get("mutation")
            severity = spec.get("severity", 0.5)
            if mutation and severity > 0:
                fn_name = _MUTATION_FN_MAP.get(mutation)
                if fn_name:
                    mutated = getattr(self._mut, fn_name)(base, severity)
                else:
                    mutated = base
            else:
                mutated = base
            cases.append(BenchmarkCase(
                name=spec["name"],
                source_a=base,
                source_b=mutated,
                expected_verdict=verdict,
                category=spec.get("category", "general"),
                metadata=spec,
            ))
        return cases

    def _load_from_file(self, path: Path) -> list[BenchmarkCase]:
        data = json.loads(path.read_text(encoding="utf-8"))
        cases: list[BenchmarkCase] = []
        for item in data:
            cases.append(BenchmarkCase(
                name=item["name"],
                source_a=item.get("source_a"),
                source_b=item.get("source_b"),
                task_spec=item.get("task_spec"),
                expected_verdict=RegressionVerdict(item["expected_verdict"]),
                category=item.get("category", "general"),
                metadata=item.get("metadata", {}),
            ))
        return cases


def _serialize_source(source: Any) -> Any:
    """Best-effort serialisation of an accessibility tree or raw data."""
    if source is None:
        return None
    if hasattr(source, "root") and hasattr(source.root, "id"):
        return {"type": "accessibility_tree", "root_id": source.root.id}
    return str(source)
