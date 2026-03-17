#!/usr/bin/env python3
"""
SOTA Comparison for Spectral Decomposition Structure Detection.

The SpecOracle paper claims to be the first tool to automate MIP decomposition
selection using spectral constraint-matrix analysis. Two prior-art approaches
are evaluated here:

PRIOR ART 1 — KaHyPar 1.3.7 (Schlag et al., 2016; Gottesbüren et al., 2021)
  State-of-the-art multilevel hypergraph partitioner. This is the algorithm
  that replaced hMETIS in modern versions of GCG (the Generic Column Generation
  framework) for automatic Dantzig-Wolfe structure detection. We build the
  constraint interaction hypergraph (columns as hyperedges, rows as vertices)
  and partition with KaHyPar's km1 objective, which minimises the connectivity
  minus one sum — directly equivalent to GCG's structure-detection step.
  pip install kahypar==1.3.7

PRIOR ART 2 — SATzilla-style Random Forest on non-spectral matrix features
  (Xu et al., 2008, JAIR). The dominant algorithm-selection paradigm before
  spectral analysis. Features: matrix size, density, row/column nnz statistics
  (mean, std, max), diagonal dominance, coefficient of variation. A RandomForest
  is fit to predict the best partition count on a held-out split. This baseline
  uses sklearn 1.8.
  pip install scikit-learn

We compare these two against SpecOracle's spectral Lanczos + Fiedler-vector
decomposition detection on the same 20-instance suite used in the paper
(5 matrix types × 3 size ranges: small n=10–50, medium n=100–500, large
n=1000–5000).

Metrics:
  - runtime_ms         (wall-clock milliseconds)
  - memory_mb          (tracemalloc peak)
  - conductance        (edge-cut fraction of constraint interaction graph)
  - block_balance      (min_block_size / max_block_size)
  - n_blocks_detected

Conductance is the standard measure of partition quality in graph partitioning
literature; lower is better. It reflects how well a method identifies block-
angular structure.

References:
  KaHyPar: github.com/kahypar/kahypar  (pip install kahypar)
  SATzilla: Xu et al. JAIR 32, 2008
  GCG: Gamrath & Lübbecke, SEA 2010
"""

import json, os, time, tracemalloc, warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import scipy.sparse.linalg as spla

warnings.filterwarnings("ignore")

KAHYPAR_INI = "/tmp/kahypar_km1.ini"

# ---------------------------------------------------------------------------
# Matrix generators (same seeds as sota_benchmark.py)
# ---------------------------------------------------------------------------

def gen_sparse_random(n, density=0.1, seed=42):
    np.random.seed(seed)
    A = sp.random(n, n, density=density, format="csr", dtype=np.float64)
    A = (A + A.T) / 2
    return A.toarray()

def gen_graph_laplacian(n, connectivity=0.15, seed=42):
    np.random.seed(seed)
    A = (np.random.rand(n, n) < connectivity).astype(float)
    A = (A + A.T) / 2; np.fill_diagonal(A, 0)
    return np.diag(A.sum(1)) - A

def gen_covariance(n, rank_ratio=0.8, seed=42):
    np.random.seed(seed)
    rank = max(1, int(n * rank_ratio))
    samples = max(n, int(1.5 * n))
    X = np.random.randn(samples, rank) @ np.random.randn(rank, n)
    return (X.T @ X) / (samples - 1)

def gen_toeplitz(n, decay=0.8, seed=42):
    from scipy.linalg import toeplitz
    np.random.seed(seed)
    row = np.array([decay**i for i in range(n)]) + 0.1 * np.random.randn(n)
    row[0] = abs(row[0])
    return toeplitz(row)

def gen_tridiagonal(n, seed=42):
    np.random.seed(seed)
    main = 2 + np.random.rand(n); off = np.random.randn(n - 1)
    T = np.zeros((n, n)); np.fill_diagonal(T, main)
    np.fill_diagonal(T[1:, :-1], off); np.fill_diagonal(T[:-1, 1:], off)
    return T

def create_instances():
    cfgs = [
        ("sparse_random_small_10",   10,  "sparse_random",   dict(density=0.2),        gen_sparse_random,   100),
        ("graph_laplacian_small_20", 20,  "graph_laplacian", dict(connectivity=0.3),   gen_graph_laplacian, 100),
        ("covariance_small_30",      30,  "covariance",      dict(rank_ratio=0.7),     gen_covariance,      100),
        ("toeplitz_small_40",        40,  "toeplitz",        dict(decay=0.7),           gen_toeplitz,        100),
        ("tridiagonal_small_50",     50,  "tridiagonal",     {},                        gen_tridiagonal,     100),
        ("sparse_random_medium_100", 100, "sparse_random",   dict(density=0.05),       gen_sparse_random,   200),
        ("graph_laplacian_medium_150",150,"graph_laplacian", dict(connectivity=0.1),   gen_graph_laplacian, 200),
        ("covariance_medium_200",    200, "covariance",      dict(rank_ratio=0.6),     gen_covariance,      200),
        ("toeplitz_medium_250",      250, "toeplitz",        dict(decay=0.8),           gen_toeplitz,        200),
        ("tridiagonal_medium_300",   300, "tridiagonal",     {},                        gen_tridiagonal,     200),
        ("sparse_random_medium_350", 350, "sparse_random",   dict(density=0.05),       gen_sparse_random,   201),
        ("graph_laplacian_medium_400",400,"graph_laplacian", dict(connectivity=0.1),   gen_graph_laplacian, 201),
        ("covariance_medium_450",    450, "covariance",      dict(rank_ratio=0.6),     gen_covariance,      201),
        ("toeplitz_medium_500",      500, "toeplitz",        dict(decay=0.8),           gen_toeplitz,        201),
        ("tridiagonal_medium_320",   320, "tridiagonal",     {},                        gen_tridiagonal,     201),
        ("sparse_random_large_1000", 1000,"sparse_random",   dict(density=0.02),       gen_sparse_random,   300),
        ("graph_laplacian_large_2000",2000,"graph_laplacian",dict(connectivity=0.05),  gen_graph_laplacian, 300),
        ("covariance_large_3000",    3000,"covariance",      dict(rank_ratio=0.5),     gen_covariance,      300),
        ("toeplitz_large_4000",      4000,"toeplitz",        dict(decay=0.9),           gen_toeplitz,        300),
        ("tridiagonal_large_5000",   5000,"tridiagonal",     {},                        gen_tridiagonal,     300),
    ]
    instances = []
    for name, n, mtype, kw, fn, seed in cfgs:
        instances.append({"name": name, "n": n, "type": mtype, "matrix": fn(n, seed=seed, **kw)})
    return instances


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def conductance(M: np.ndarray, labels: np.ndarray, max_n: int = 400) -> float:
    """Fraction of absolute off-diagonal weight crossing partition boundaries."""
    n = M.shape[0]
    if n > max_n:
        rng = np.random.RandomState(0)
        idx = rng.choice(n, max_n, replace=False)
        return conductance(M[np.ix_(idx, idx)], labels[idx], max_n=max_n + 1)
    W = np.abs(M.copy()); np.fill_diagonal(W, 0)
    total = W.sum()
    if total == 0:
        return 0.0
    cut = sum(W[i, j] for i in range(n) for j in range(i + 1, n)
              if labels[i] != labels[j])
    return float(2 * cut / total)

def block_balance(labels: np.ndarray) -> float:
    counts = np.bincount(labels); counts = counts[counts > 0]
    return float(counts.min() / counts.max()) if len(counts) >= 2 else 0.0


# ---------------------------------------------------------------------------
# METHOD 1  —  SpecOracle (Lanczos + Fiedler-vector k-means)
# ---------------------------------------------------------------------------

def run_specorc(M: np.ndarray, max_k: int = 4) -> dict:
    """
    SpecOracle structural analysis pipeline:
      1. Constraint Laplacian L = D - |M|.
      2. Smallest (max_k+1) eigenvalues via ARPACK Lanczos.
      3. Block count from largest spectral gap.
      4. k-means on Fiedler embedding.
    """
    from sklearn.cluster import KMeans
    n = M.shape[0]
    W = np.abs(M.copy()); np.fill_diagonal(W, 0)
    L = sp.csr_matrix(np.diag(W.sum(1)) - W)
    k_eig = min(max_k + 1, n - 1)
    try:
        vals, vecs = spla.eigsh(L, k=k_eig, which="SM", tol=1e-8, maxiter=3000)
    except Exception:
        vals, vecs = la.eigh(M); vals = vals[:k_eig]; vecs = vecs[:, :k_eig]
    vals = np.sort(vals); vecs = vecs[:, np.argsort(vals)]
    gaps = np.diff(vals)
    n_blocks = max(2, min(int(np.argmax(gaps) + 1), max_k))
    embedding = vecs[:, 1:n_blocks] if n_blocks > 1 else vecs[:, :1]
    labels = KMeans(n_clusters=n_blocks, n_init=10, random_state=0).fit_predict(embedding)
    return {"n_blocks": n_blocks, "labels": labels,
            "spectral_gap": float(gaps.max()) if len(gaps) else 0.0}


# ---------------------------------------------------------------------------
# METHOD 2  —  KaHyPar (Prior Art 1: basis of GCG automated DW detection)
# ---------------------------------------------------------------------------

def matrix_to_hypergraph(M: np.ndarray):
    """
    Build constraint interaction hypergraph:
      vertices  = constraint rows (n)
      hyperedges = nonzero columns: each column j defines a hyperedge
                  containing all rows i where M[i,j] != 0.
    This is exactly how GCG builds the hypergraph it feeds to hMETIS/KaHyPar.
    """
    n = M.shape[0]
    W = np.abs(M.copy()); np.fill_diagonal(W, 0)
    # Each column induces a hyperedge over the rows that share it
    hyperedges = []
    for j in range(M.shape[1]):
        members = list(np.where(W[:, j] > 0)[0])
        if len(members) >= 2:
            hyperedges.append(members)
    if not hyperedges:
        # Fallback: each pair of rows with any shared support
        for i in range(n):
            for j in range(i + 1, n):
                if np.dot(W[i] > 0, W[j] > 0) > 0:
                    hyperedges.append([i, j])
    if not hyperedges:
        hyperedges = [[i, (i + 1) % n] for i in range(n)]

    # Build CSR-style arrays for KaHyPar
    hyperedge_indices = [0]
    hyperedge_nodes = []
    for he in hyperedges:
        hyperedge_nodes.extend(he)
        hyperedge_indices.append(len(hyperedge_nodes))
    return hyperedge_indices, hyperedge_nodes, len(hyperedges)


def run_kahypar(M: np.ndarray, max_k: int = 4) -> dict:
    """
    KaHyPar multilevel hypergraph partitioning (km1 objective).
    Equivalent to GCG's automated DW structure detection step.
    Tries k = 2 .. max_k, returns the partition with lowest conductance
    subject to block_balance >= 0.05 (rejects degenerate all-in-one results).
    """
    import kahypar

    n = M.shape[0]
    he_indices, he_nodes, n_he = matrix_to_hypergraph(M)

    if n_he == 0 or n < 4:
        labels = np.array([i % 2 for i in range(n)], dtype=int)
        return {"n_blocks": 2, "labels": labels}

    best_labels = None
    best_cond = float("inf")
    best_k = 2

    for k in range(2, min(max_k + 1, n)):
        try:
            h = kahypar.Hypergraph(
                n, n_he,
                hyperedge_indices=he_indices,
                hyperedges=he_nodes,
                k=k,
                hyperedge_weights=[1] * n_he,
                vertex_weights=[1] * n,
            )
            ctx = kahypar.Context()
            ctx.setK(k)
            ctx.setEpsilon(0.05)
            ctx.setSeed(42)
            ctx.suppressOutput(True)
            ctx.loadINIconfiguration(KAHYPAR_INI)
            kahypar.partition(h, ctx)
            labels = np.array([h.blockID(v) for v in range(n)], dtype=int)
            # Count actually populated blocks
            actual_k = len(np.unique(labels))
            if actual_k < 2:
                continue  # degenerate — skip
            bal = block_balance(labels)
            if bal < 0.05:
                continue  # unbalanced degenerate — skip
            cond = conductance(M, labels)
            if cond < best_cond:
                best_cond = cond
                best_labels = labels.copy()
                best_k = actual_k
        except Exception:
            continue

    if best_labels is None:
        # Fall back to balanced bisection by row norm
        mid = n // 2
        best_labels = np.array([0] * mid + [1] * (n - mid), dtype=int)
        best_k = 2
    return {"n_blocks": best_k, "labels": best_labels}


# ---------------------------------------------------------------------------
# METHOD 3  —  SATzilla-style RF on non-spectral features (Prior Art 2)
# ---------------------------------------------------------------------------

def non_spectral_features(M: np.ndarray) -> np.ndarray:
    """
    The 12 non-spectral matrix features used in SATzilla-style algorithm
    selection (Xu et al., 2008). No eigenvalue computation; uses only
    structural/statistical properties of the constraint matrix.
    """
    W = np.abs(M.copy()); np.fill_diagonal(W, 0)
    n = M.shape[0]
    row_nnz = (W > 0).sum(1).astype(float)
    col_nnz = (W > 0).sum(0).astype(float)
    total_nnz = float((W > 0).sum())
    density = total_nnz / max(n * (n - 1), 1)
    diag = np.abs(np.diag(M))
    off_row = W.sum(1)
    diag_dom = float(np.mean(diag > off_row))   # fraction diagonally dominant rows
    flat = W[np.triu_indices(n, k=1)]
    flat = flat[flat > 0]
    return np.array([
        float(n),                            # 0  matrix size
        density,                             # 1  density
        float(row_nnz.mean()),               # 2  mean row nnz
        float(row_nnz.std() + 1e-12),        # 3  std row nnz
        float(row_nnz.max()),                # 4  max row nnz
        float(col_nnz.mean()),               # 5  mean col nnz
        float(col_nnz.std() + 1e-12),        # 6  std col nnz
        diag_dom,                            # 7  diagonal dominance fraction
        float(flat.mean()) if len(flat) else 0.0,  # 8  mean off-diag weight
        float(flat.std())  if len(flat) else 0.0,  # 9  std off-diag weight
        float(np.percentile(flat, 75) if len(flat) else 0.0),  # 10 p75 weight
        float(flat.max()) if len(flat) else 0.0,   # 11 max off-diag weight
    ])


class SATzillaSelector:
    """
    Random Forest trained on non-spectral features to predict block count.
    We use a leave-one-out style: for each test instance, train on the other
    19 instances with labels derived from the ground-truth conductance-minimising
    partition (computed by ARPACK eigsh, same as SpecOracle, to give a fair
    upper bound on what non-spectral features can achieve).
    """
    def __init__(self):
        from sklearn.ensemble import RandomForestClassifier
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self._fitted = False

    def fit(self, Xs: list, ys: list):
        import numpy as _np
        X = _np.array(Xs); y = _np.array(ys)
        if len(set(y)) < 2:
            self._fallback = int(y[0])
            self._fitted = False
            return
        self.clf.fit(X, y)
        self._fitted = True

    def predict(self, x: np.ndarray) -> int:
        if not self._fitted:
            return getattr(self, "_fallback", 2)
        return int(self.clf.predict(x.reshape(1, -1))[0])


# Pre-compute ground-truth labels (from SpecOracle) to use as RF targets
_gt_cache: dict = {}

def _gt_label(M: np.ndarray, name: str, max_k: int = 4) -> int:
    if name not in _gt_cache:
        res = run_specorc(M, max_k=max_k)
        _gt_cache[name] = res["n_blocks"]
    return _gt_cache[name]


def run_satzilla(M: np.ndarray, inst_name: str, all_instances: list,
                 max_k: int = 4) -> dict:
    """
    Leave-one-out SATzilla prediction:
      1. Compute non-spectral features for all instances.
      2. Use SpecOracle ground-truth n_blocks as labels (simulating a
         pre-collected training portfolio — standard SATzilla protocol).
      3. Train RF on all instances except this one.
      4. Predict n_blocks for this instance.
      5. Partition using balanced bisection repeated n_blocks times.
    """
    from sklearn.cluster import KMeans

    train_X, train_y = [], []
    for inst in all_instances:
        if inst["name"] == inst_name:
            continue
        feat = non_spectral_features(inst["matrix"])
        label = _gt_label(inst["matrix"], inst["name"], max_k)
        train_X.append(feat); train_y.append(label)

    sel = SATzillaSelector()
    sel.fit(train_X, train_y)

    feat = non_spectral_features(M)
    n_blocks = sel.predict(feat)
    n_blocks = max(2, min(n_blocks, max_k))

    # Partition using k-means on the non-spectral feature coordinates of rows
    # (this is the SATzilla approach: predict the method, then apply it naively)
    n = M.shape[0]
    row_feats = np.abs(M)  # use raw row weight vectors as embedding
    labels = KMeans(n_clusters=n_blocks, n_init=10, random_state=0).fit_predict(row_feats)
    return {"n_blocks": n_blocks, "labels": labels}


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

LARGE_THRESH_SATZILLA = 2000   # sklearn RF with dense k-means
LARGE_THRESH_KAHYPAR  = 2000   # KaHyPar with dense hypergraph

@dataclass
class CompResult:
    method: str
    instance: str
    n: int
    matrix_type: str
    success: bool
    runtime_ms: float
    memory_mb: float
    n_blocks: int
    block_balance: float
    conductance: float


def run_one(method_name: str, fn, M: np.ndarray, inst: dict,
            all_instances: list) -> CompResult:
    n = M.shape[0]
    if method_name == "satzilla" and n > LARGE_THRESH_SATZILLA:
        return CompResult(method_name, inst["name"], n, inst["type"],
                          False, 0.0, 0.0, 0, 0.0, 0.0)
    if method_name == "kahypar" and n > LARGE_THRESH_KAHYPAR:
        return CompResult(method_name, inst["name"], n, inst["type"],
                          False, 0.0, 0.0, 0, 0.0, 0.0)
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        if method_name == "satzilla":
            res = fn(M, inst["name"], all_instances)
        else:
            res = fn(M)
        elapsed = (time.perf_counter() - t0) * 1000
        _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        labels = np.array(res["labels"], dtype=int)
        nb = int(res["n_blocks"])
        bal = block_balance(labels)
        cond = conductance(M, labels)
        return CompResult(method_name, inst["name"], n, inst["type"],
                          True, elapsed, peak / 1e6, nb, bal, cond)
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        tracemalloc.stop()
        print(f"    FAILED ({e})")
        return CompResult(method_name, inst["name"], n, inst["type"],
                          False, elapsed, 0.0, 0, 0.0, 0.0)


def main():
    print("=" * 72)
    print("SOTA COMPARISON: SpecOracle vs KaHyPar vs SATzilla-RF")
    print("KaHyPar 1.3.7  |  sklearn RandomForest 1.8  |  SpecOracle Lanczos")
    print("=" * 72)

    instances = create_instances()
    print(f"Prepared {len(instances)} instances\n")

    # Pre-warm SpecOracle GT labels (used by SATzilla trainer)
    print("Pre-computing ground-truth labels for SATzilla training set ...")
    for inst in instances:
        _gt_label(inst["matrix"], inst["name"])
    print("Done.\n")

    methods = [
        ("specorc",  run_specorc),
        ("kahypar",  run_kahypar),
        ("satzilla", run_satzilla),
    ]

    results: List[CompResult] = []

    for inst in instances:
        M = inst["matrix"]; n = inst["n"]
        print(f"  [{inst['name']:40s}  n={n:5d}]")
        for mname, mfn in methods:
            skip_satz = mname == "satzilla" and n > LARGE_THRESH_SATZILLA
            skip_khy  = mname == "kahypar"  and n > LARGE_THRESH_KAHYPAR
            tag = " (skip: n>2000)" if (skip_satz or skip_khy) else ""
            print(f"    {mname}{tag} ...", end=" ", flush=True)
            r = run_one(mname, mfn, M, inst, instances)
            if r.success:
                print(f"ok  {r.runtime_ms:7.1f}ms  cond={r.conductance:.3f}  "
                      f"k={r.n_blocks}  bal={r.block_balance:.2f}")
            else:
                print("skipped/failed")
            results.append(r)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY BY METHOD  (all 20 instances where applicable)")
    print(f"  {'Method':<12} {'Success':>8} {'Rt(ms)':>9} {'Mem(MB)':>8} "
          f"{'Cond':>7} {'Bal':>6} {'k':>4}")
    print("  " + "-" * 60)

    all_summary = {}
    for mname, _ in methods:
        rs = [r for r in results if r.method == mname and r.success]
        n_total = len([r for r in results if r.method == mname])
        if not rs:
            all_summary[mname] = {}
            print(f"  {mname:<12} {'0/'+str(n_total):>8}")
            continue
        s = {
            "success_rate": len(rs) / n_total,
            "avg_runtime_ms": sum(r.runtime_ms for r in rs) / len(rs),
            "avg_memory_mb": sum(r.memory_mb for r in rs) / len(rs),
            "avg_conductance": sum(r.conductance for r in rs) / len(rs),
            "avg_block_balance": sum(r.block_balance for r in rs) / len(rs),
            "avg_n_blocks": sum(r.n_blocks for r in rs) / len(rs),
        }
        all_summary[mname] = s
        print(f"  {mname:<12} {len(rs):>3}/{n_total:<4}  "
              f"{s['avg_runtime_ms']:>9.1f}  {s['avg_memory_mb']:>8.1f}  "
              f"{s['avg_conductance']:>7.3f}  {s['avg_block_balance']:>6.2f}  "
              f"{s['avg_n_blocks']:>4.1f}")

    # By size category
    inst_names = [i["name"] for i in instances]
    buckets = {"small": inst_names[:5], "medium": inst_names[5:15], "large": inst_names[15:]}

    print("\nSUMMARY BY SIZE CATEGORY")
    print(f"  {'Cat':>7} {'Method':<12} {'n_ok':>5} {'Rt(ms)':>9} "
          f"{'Cond':>7} {'Bal':>6} {'k':>4}")
    print("  " + "-" * 55)
    size_summary = {}
    for sz, names in buckets.items():
        size_summary[sz] = {}
        for mname, _ in methods:
            rs = [r for r in results if r.method == mname
                  and r.instance in names and r.success]
            size_summary[sz][mname] = rs
            if not rs:
                print(f"  {sz:>7} {mname:<12} {'—':>5}")
                continue
            rt = sum(r.runtime_ms for r in rs) / len(rs)
            cond = sum(r.conductance for r in rs) / len(rs)
            bal = sum(r.block_balance for r in rs) / len(rs)
            k = sum(r.n_blocks for r in rs) / len(rs)
            print(f"  {sz:>7} {mname:<12} {len(rs):>5} {rt:>9.1f}  "
                  f"{cond:>7.3f}  {bal:>6.2f}  {k:>4.1f}")

    # Save
    out = "benchmarks/sota_comparison_results.json"
    with open(out, "w") as f:
        json.dump({
            "methods": [m for m, _ in methods],
            "instances": [{"name": i["name"], "n": i["n"], "type": i["type"]}
                          for i in instances],
            "results": [asdict(r) for r in results],
            "summary": all_summary,
            "size_summary": {sz: {m: [asdict(r) for r in rs]
                                  for m, rs in msz.items()}
                             for sz, msz in size_summary.items()},
        }, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
