# Theory Overview

This document describes the formal foundations of TaintFlow's leakage detection.

## 1. Partition-Taint Lattice

TaintFlow tracks which data partition (train, test, or both) has influenced each
column at each point in a pipeline. The **partition-taint lattice** is the
four-element lattice:

```
        ⊤  (Both / Leaked)
       / \
    Train  Test
       \ /
        ⊥  (Untainted)
```

Each element represents a taint level:

| Element | Meaning |
|---------|---------|
| `⊥`    | Column has not been influenced by any partition-specific data. |
| `Train` | Column has been influenced only by training data. |
| `Test`  | Column has been influenced only by test data. |
| `⊤`    | Column has been influenced by **both** partitions — this is leakage. |

The **join** (least upper bound) operation models information merging: if a
transformer's `fit()` sees training data and its `transform()` is applied to test
data, the output column's taint is `Train ⊔ Test = ⊤`.

**Key property:** The lattice is finite, so all fixed-point computations
terminate in at most four iterations.

## 2. Fit-Transform Decomposition

Every scikit-learn estimator follows a `fit` / `transform` (or `predict`)
protocol. TaintFlow decomposes each estimator into:

1. **Fit phase** — The estimator observes data and stores learned parameters
   (e.g., mean and variance for `StandardScaler`). TaintFlow records which
   partition's rows were seen.

2. **Transform phase** — The estimator applies its stored parameters to new data.
   The output taint is the join of:
   - The taint of the **input columns** being transformed.
   - The taint of the **fitted parameters** (from the fit phase).

This decomposition is the core abstraction that lets TaintFlow detect leakage
without executing the pipeline numerically.

### Example: StandardScaler

```
fit(X_full)          → params tainted with ⊤ (saw both train and test rows)
transform(X_test)    → output taint = ⊤ ⊔ Test = ⊤  (LEAKED)
```

vs. the correct pipeline:

```
fit(X_train)         → params tainted with Train
transform(X_test)    → output taint = Train ⊔ Test = ⊤?
```

Wait — even the correct pipeline shows `⊤`? No: in a proper `Pipeline`, the
`transform` of test data uses parameters fitted **only** on training data, and
the test rows are never seen during `fit`. TaintFlow distinguishes these cases
by tracking **parameter taint** separately from **row taint**. In the correct
pipeline, the parameter taint is `Train`, and the row taint of the test set is
`Test`. The output is parameterised by `(param_taint=Train, row_taint=Test)`,
which is the expected and non-leaky state. Leakage occurs only when
`param_taint ⊒ Test` (parameters have seen test data).

## 3. Channel Capacity Bounds

Not all leakage is equally severe. A `StandardScaler` fitted on 1000 samples
where 200 are test rows leaks some information, but far less than a target
encoder that directly maps `y_test` values into features.

TaintFlow quantifies leakage severity using **channel capacity** from information
theory. Each leaky transformer is modelled as a noisy channel from the test
partition to the model's input, and TaintFlow computes an upper bound on the
mutual information transmitted.

### Supported Models

| Model | Applicable To | Bound |
|-------|--------------|-------|
| **Gaussian Channel** | Mean/variance estimators (StandardScaler, Normalizer) | `C ≤ (d/2) · log(1 + n_test/n_train)` where `d` is dimensionality |
| **Counting Bound** | Categorical encoders (OrdinalEncoder, LabelEncoder) | `C ≤ log(k)` where `k` is the number of categories |
| **Mutual Information** | General transformers | Estimated via KSG estimator on sampled data |

These bounds let users triage warnings: a leakage with capacity 0.01 bits is
negligible, while one with 5.3 bits is critical.

## 4. PI-DAG Extraction

TaintFlow parses a Python script or sklearn `Pipeline` object into a
**Pipeline Information DAG (PI-DAG)** — a directed acyclic graph where:

- **Nodes** are operations: `fit`, `transform`, `predict`, `train_test_split`,
  and data-loading steps.
- **Edges** represent data flow: which columns flow from one operation to the
  next, annotated with their taint.

The PI-DAG is the structure over which taint propagation is computed via a
standard worklist algorithm.

## 5. Soundness Theorem

> **Theorem (Soundness).** If TaintFlow reports no leakage warnings for a
> pipeline, then no test-partition information flows into the model's training
> inputs.

More precisely: let `P` be a pipeline and `σ` be any concrete data split. If
TaintFlow's abstract interpretation yields `param_taint(t) ⊑ Train` for every
transformer `t` in `P`, then for all concrete executions under `σ`, the
numerical parameters of `t` are a function only of the training rows.

**Proof sketch.** By induction on the PI-DAG topology. The base case is
`train_test_split`, which assigns `Train` and `Test` taints correctly. The
inductive step follows from the monotonicity of the join operation and the
correctness of the fit-transform decomposition (each abstract transformer
over-approximates the concrete information flow). ∎

The converse does not hold: TaintFlow may report false positives (leakage
warnings for pipelines that are technically safe), but it will never miss true
leakage. This is the standard soundness-over-completeness trade-off in abstract
interpretation.
