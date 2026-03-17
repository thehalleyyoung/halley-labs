# TaintFlow: ML Pipeline Leakage Auditor

TaintFlow is a research prototype for auditing train/test leakage in Python ML workflows.

The strongest executable evidence in the current repository is the **live sklearn audit-replay layer** under `implementation/src/taintflow/integrations/sklearn_interceptor.py`: it records `fit`, `fit_transform`, `transform`, `predict`, and `score` events from real sklearn execution through `AuditedEstimator`, `AuditedPipeline`, and `PipelineAuditor`.

## What this repo demonstrates today

- A quantitative leakage-analysis design based on partition taints and information-flow bounds.
- Executable sklearn wrappers that emit stage-level audit logs on live pipeline runs.
- A reproducible benchmark script, `benchmarks/live_audit_recovery.py`, that exercises the wrapper layer on real sklearn pipelines.
- A synthetic stress test showing that pre-split feature selection can materially inflate held-out accuracy.

## What is still incomplete

The repository also contains a broader `taintflow audit` CLI story built around runtime tracing, DAG construction, and abstract interpretation. That end-to-end path is **not** the main empirical claim of the current paper or README, because it has not yet been validated in a controlled benchmark in this snapshot.

## Recovery benchmark

Run the grounded benchmark with:

```bash
python3 benchmarks/live_audit_recovery.py \
  --output benchmarks/live_audit_recovery_results.json
```

In the checked-in results file:

- a benchmark-side checker over live audit logs separated **4 leaky** and **4 clean** sklearn scenarios with `TP=4`, `TN=4`, `FP=0`, `FN=0`;
- the four paired real-dataset scenarios had **zero held-out score delta**, which is exactly why structural auditing is useful;
- a synthetic random-label feature-selection stress test increased test accuracy from **0.5833** to **0.7417** (`+0.1583`) when `SelectKBest` was fitted before the split;
- audited execution overhead was near-1x on two representative clean scenarios: **1.115x** on `clean_scaler_iris` and **1.014x** on `clean_pca_wine`.

These numbers come directly from `benchmarks/live_audit_recovery_results.json`.

## Minimal API examples

Audit a new pipeline as it runs:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from taintflow.integrations.sklearn_interceptor import AuditedPipeline

pipe = AuditedPipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=500)),
])

pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)
```

Replay an existing sklearn pipeline through the audit log:

```python
from sklearn.pipeline import Pipeline

from taintflow.integrations.sklearn_interceptor import PipelineAuditor

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=500)),
])

auditor = PipelineAuditor(pipe)
auditor.audit_fit(X_train, y_train)
```

## Repository layout

- `implementation/src/taintflow/`: prototype implementation.
- `benchmarks/live_audit_recovery.py`: grounded live-execution benchmark used by the paper.
- `benchmarks/live_audit_recovery_results.json`: checked-in output from that benchmark.
- `tool_paper.tex`: paper source.
- `groundings.json`: grounded claims and experiment provenance for the current story.

## Suggested interpretation

TaintFlow is most compelling today as a **validated audit substrate for sklearn leakage replay** plus a **research direction for quantitative information-flow analysis**. The current repository shows that the wrappers execute, log useful stage structure, and support lightweight leakage checks even when score-based heuristics are silent.
