# CausalCert

Structural stress-testing for causal DAGs via per-edge fragility scores and robustness radii.

## Recovered paper story

CausalCert is now documented around a **fragility-first** story instead of a broad benchmark-superiority story.

The strongest grounded evidence currently in this repository is:

1. the formal theory and solver formulations in `tool_paper.tex`
2. a real Python implementation whose full test suite currently passes
3. end-to-end example scripts that surface nontrivial fragility rankings
4. a smaller recovery harness that demonstrates a certified radius on a tiny live example

What the repository does **not** currently justify is a strong benchmark paper about minimum-edit radii across many DAG families. The radius machinery exists, but several larger end-to-end example paths still collapse to degenerate `"[0,0]"` results or fail to produce an ATE. The paper and docs now say that plainly.

## What we can honestly claim today

- `cd implementation && python3 -m pytest tests -q` passes with **1365 passed, 1 skipped**
- `cd implementation && python3 examples/quickstart.py` completes in about **1.57 s**, returns radius `"[0,0]"`, and emits nontrivial top fragility scores `0.7900 / 0.7375 / 0.3150`
- `cd implementation && python3 examples/published_dag_analysis.py --samples 500 --max-nodes 8 --top-k 10` completes **11/11** built-in DAG analyses and surfaces high-impact directed pairs, though every run still reports radius `"[0,0]"` and no ATE
- `python3 benchmarks/live_fragility_recovery.py` gives a tiny end-to-end witness where the current implementation returns a **certified radius of 1**

Taken together, that is enough for a compelling and truthful story:

- the theory is substantial,
- the implementation is real,
- fragility ranking is already useful,
- radius computation is promising but not yet reliable enough for large benchmark claims.

## Reproducible commands

```bash
cd implementation && python3 -m pytest tests -q
cd implementation && python3 examples/quickstart.py
cd implementation && python3 examples/published_dag_analysis.py --samples 500 --max-nodes 8 --top-k 10
python3 benchmarks/live_fragility_recovery.py
```

## Grounded artifacts

- `tool_paper.tex` / `tool_paper.pdf`
- `groundings.json`
- `implementation/examples/quickstart.py`
- `implementation/examples/published_dag_analysis.py`
- `benchmarks/live_fragility_recovery.py`
- `benchmarks/live_fragility_recovery_results.json`

## Practical takeaway

If you want to understand the evidence-backed contribution, start with the paper and the fragility-oriented example scripts. The most defensible next step is to repair the larger radius-evaluation pipeline so that the already-implemented solver ideas can be validated on broader benchmark families without overclaiming.
