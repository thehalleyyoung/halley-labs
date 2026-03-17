from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_SRC = REPO_ROOT / "implementation" / "src"
if str(IMPLEMENTATION_SRC) not in sys.path:
    sys.path.insert(0, str(IMPLEMENTATION_SRC))

from taintflow.integrations.sklearn_interceptor import (  # noqa: E402
    AuditedEstimator,
    AuditedPipeline,
    PipelineAuditor,
    get_audit_log,
    reset_audit_log,
)


ScenarioFactory = Callable[[], Any]


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    dataset: str
    transforms: tuple[ScenarioFactory, ...]
    classifier: ScenarioFactory
    leaky: bool
    api: str


@dataclass(frozen=True)
class OverheadResult:
    scenario: str
    vanilla_median_s: float
    audited_median_s: float
    ratio: float


def load_dataset(name: str) -> tuple[pd.DataFrame, np.ndarray]:
    loaders: dict[str, Callable[..., tuple[np.ndarray, np.ndarray]]] = {
        "iris": load_iris,
        "wine": load_wine,
        "breast_cancer": load_breast_cancer,
    }
    X, y = loaders[name](return_X_y=True)
    return pd.DataFrame(X), np.asarray(y)


def clone_steps(
    transforms: tuple[ScenarioFactory, ...],
    classifier: ScenarioFactory,
) -> list[Any]:
    return [factory() for factory in transforms] + [classifier()]


def record_to_dict(record: Any) -> dict[str, Any]:
    return {
        "estimator": record.estimator_class,
        "method": record.method,
        "input_rows": record.input_shape[0],
        "output_rows": record.output_shape[0],
        "wall_time_ms": record.wall_time_ms,
    }


def run_scenario(spec: ScenarioSpec) -> dict[str, Any]:
    X, y = load_dataset(spec.dataset)
    full_rows = len(X)
    stratify = y if len(np.unique(y)) < 20 else None
    steps = clone_steps(spec.transforms, spec.classifier)

    reset_audit_log()

    if spec.leaky:
        Xt: Any = X
        for estimator in steps[:-1]:
            Xt = AuditedEstimator(estimator).fit_transform(Xt, y)
        X_train, X_test, y_train, y_test = train_test_split(
            Xt,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )
        audited_classifier = AuditedEstimator(steps[-1])
        audited_classifier.fit(X_train, y_train)
        score = audited_classifier.score(X_test, y_test)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=stratify,
        )
        named_steps = [(f"s{i}", estimator) for i, estimator in enumerate(steps)]
        if spec.api == "audited_pipeline":
            pipeline = AuditedPipeline(named_steps)
            pipeline.fit(X_train, y_train)
            score = pipeline.score(X_test, y_test)
        elif spec.api == "pipeline_auditor":
            pipeline = Pipeline(named_steps)
            auditor = PipelineAuditor(pipeline)
            auditor.audit_fit(X_train, y_train)
            score = pipeline.score(X_test, y_test)
        else:
            raise ValueError(f"Unknown API: {spec.api}")

    records = get_audit_log().records
    detected_pre_split_fit = any(
        record.method in {"fit", "fit_transform"} and record.input_shape[0] == full_rows
        for record in records
    )

    return {
        "name": spec.name,
        "dataset": spec.dataset,
        "expected_leaky": spec.leaky,
        "detected_pre_split_fit": detected_pre_split_fit,
        "score": score,
        "api": spec.api,
        "full_rows": full_rows,
        "record_count": len(records),
        "records": [record_to_dict(record) for record in records],
    }


def compute_confusion(results: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "tp": sum(r["expected_leaky"] and r["detected_pre_split_fit"] for r in results),
        "tn": sum((not r["expected_leaky"]) and (not r["detected_pre_split_fit"]) for r in results),
        "fp": sum((not r["expected_leaky"]) and r["detected_pre_split_fit"] for r in results),
        "fn": sum(r["expected_leaky"] and (not r["detected_pre_split_fit"]) for r in results),
    }


def compute_metrics(confusion: dict[str, int]) -> dict[str, float]:
    tp = confusion["tp"]
    tn = confusion["tn"]
    fp = confusion["fp"]
    fn = confusion["fn"]
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_paired_score_deltas(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_suffix: dict[str, dict[str, float]] = {}
    for result in results:
        suffix = result["name"].split("_", 1)[1]
        pair = by_suffix.setdefault(suffix, {})
        pair["leaky" if result["expected_leaky"] else "clean"] = result["score"]

    deltas: list[dict[str, Any]] = []
    for suffix, pair in sorted(by_suffix.items()):
        if "leaky" in pair and "clean" in pair:
            deltas.append(
                {
                    "pair": suffix,
                    "leaky_score": pair["leaky"],
                    "clean_score": pair["clean"],
                    "delta": pair["leaky"] - pair["clean"],
                }
            )
    return deltas


def benchmark_overhead(spec: ScenarioSpec, repeats: int = 10) -> OverheadResult:
    X, y = load_dataset(spec.dataset)
    stratify = y if len(np.unique(y)) < 20 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )
    vanilla_times: list[float] = []
    audited_times: list[float] = []

    for _ in range(repeats):
        steps = clone_steps(spec.transforms, spec.classifier)
        named_steps = [(f"s{i}", estimator) for i, estimator in enumerate(steps)]

        t0 = time.perf_counter()
        vanilla = Pipeline(named_steps)
        vanilla.fit(X_train, y_train)
        vanilla.score(X_test, y_test)
        vanilla_times.append(time.perf_counter() - t0)

        steps = clone_steps(spec.transforms, spec.classifier)
        named_steps = [(f"s{i}", estimator) for i, estimator in enumerate(steps)]
        t0 = time.perf_counter()
        reset_audit_log()
        audited = AuditedPipeline(named_steps)
        audited.fit(X_train, y_train)
        audited.score(X_test, y_test)
        audited_times.append(time.perf_counter() - t0)

    vanilla_median = statistics.median(vanilla_times)
    audited_median = statistics.median(audited_times)
    return OverheadResult(
        scenario=spec.name,
        vanilla_median_s=vanilla_median,
        audited_median_s=audited_median,
        ratio=audited_median / vanilla_median,
    )


def synthetic_feature_selection_stress_test() -> dict[str, float]:
    rng = np.random.RandomState(42)
    X = rng.normal(size=(400, 1000))
    y = rng.randint(0, 2, size=400)

    selector = SelectKBest(f_classif, k=20)
    X_selected = selector.fit_transform(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )
    leaky_classifier = LogisticRegression(max_iter=2000)
    leaky_classifier.fit(X_train, y_train)
    leaky_accuracy = leaky_classifier.score(X_test, y_test)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )
    clean_pipeline = Pipeline(
        [
            ("selector", SelectKBest(f_classif, k=20)),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )
    clean_pipeline.fit(X_train, y_train)
    clean_accuracy = clean_pipeline.score(X_test, y_test)

    return {
        "leaky_accuracy": leaky_accuracy,
        "clean_accuracy": clean_accuracy,
        "delta": leaky_accuracy - clean_accuracy,
    }


def build_report() -> dict[str, Any]:
    warnings.filterwarnings("ignore")

    scenarios = [
        ScenarioSpec(
            name="leaky_scaler_iris",
            dataset="iris",
            transforms=(StandardScaler,),
            classifier=lambda: LogisticRegression(max_iter=500),
            leaky=True,
            api="manual",
        ),
        ScenarioSpec(
            name="clean_scaler_iris",
            dataset="iris",
            transforms=(StandardScaler,),
            classifier=lambda: LogisticRegression(max_iter=500),
            leaky=False,
            api="audited_pipeline",
        ),
        ScenarioSpec(
            name="leaky_pca_wine",
            dataset="wine",
            transforms=(lambda: PCA(n_components=5),),
            classifier=lambda: LogisticRegression(max_iter=500),
            leaky=True,
            api="manual",
        ),
        ScenarioSpec(
            name="clean_pca_wine",
            dataset="wine",
            transforms=(lambda: PCA(n_components=5),),
            classifier=lambda: LogisticRegression(max_iter=500),
            leaky=False,
            api="pipeline_auditor",
        ),
        ScenarioSpec(
            name="leaky_selectk_bc",
            dataset="breast_cancer",
            transforms=(lambda: SelectKBest(f_classif, k=10),),
            classifier=lambda: LogisticRegression(max_iter=1500),
            leaky=True,
            api="manual",
        ),
        ScenarioSpec(
            name="clean_selectk_bc",
            dataset="breast_cancer",
            transforms=(lambda: SelectKBest(f_classif, k=10),),
            classifier=lambda: LogisticRegression(max_iter=1500),
            leaky=False,
            api="audited_pipeline",
        ),
        ScenarioSpec(
            name="leaky_minmax_pca_wine",
            dataset="wine",
            transforms=(MinMaxScaler, lambda: PCA(n_components=5)),
            classifier=lambda: LogisticRegression(max_iter=500),
            leaky=True,
            api="manual",
        ),
        ScenarioSpec(
            name="clean_minmax_pca_wine",
            dataset="wine",
            transforms=(MinMaxScaler, lambda: PCA(n_components=5)),
            classifier=lambda: LogisticRegression(max_iter=500),
            leaky=False,
            api="pipeline_auditor",
        ),
    ]

    scenario_results = [run_scenario(spec) for spec in scenarios]
    confusion = compute_confusion(scenario_results)
    paired_deltas = compute_paired_score_deltas(scenario_results)
    overhead_specs = [spec for spec in scenarios if spec.name in {"clean_scaler_iris", "clean_pca_wine"}]
    overhead = [asdict(benchmark_overhead(spec)) for spec in overhead_specs]

    return {
        "benchmark": "live_audit_recovery",
        "methodology": {
            "detection_rule": "Flag a scenario when any audited fit or fit_transform consumes the full dataset before train_test_split.",
            "scope": "Live sklearn execution through AuditedEstimator, AuditedPipeline, and PipelineAuditor.",
            "note": "This validates the executable wrapper/replay layer only; it is not an end-to-end evaluation of the taintflow audit CLI.",
        },
        "scenarios": scenario_results,
        "confusion": confusion,
        "metrics": compute_metrics(confusion),
        "paired_holdout_score_deltas": paired_deltas,
        "overhead": overhead,
        "synthetic_feature_selection_inflation": synthetic_feature_selection_stress_test(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. Defaults to stdout.",
    )
    args = parser.parse_args()

    report = build_report()
    payload = json.dumps(report, indent=2)
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
