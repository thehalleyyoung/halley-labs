# RegSynth

**Recovered, evidence-backed story:** RegSynth is a sizable Rust prototype for
formalizing a machine-readable fragment of multi-jurisdictional AI compliance,
checking synthetic obligation sets for satisfiability, exploring Pareto tradeoffs,
and emitting certificate artifacts. The executable core that genuinely works today
is a synthetic benchmark/analysis pipeline, not a validated large-scale legal
reasoning system.

## Honest takeaway

The strongest truthful story in this checkout is:

- a compiled Rust workspace and `regsynth` CLI exist;
- the CLI benchmark subcommand runs on small synthetic instances;
- the bundled Python harnesses generate synthetic multi-objective scenarios and
  let us compare MaxSMT-style search with lighter baselines;
- the Pareto validation harness demonstrates that the frontier logic and the
  trajectory-domination theorem are executable ideas;
- the repo does **not** currently justify earlier claims about industrial-scale
  runtime, 300+ obligation workloads, or benchmark superiority.

## What we reran

All claims below are tied to commands rerun in this repository on 2026-03-17.

| Command | Result |
| --- | --- |
| `cd implementation && cargo test --release` | Build completed, but test suite is not fully green: **68 passed / 2 failed**. The remaining failures are `pareto_cert::tests::generate_pareto_cert_two_points` and `verifier::tests::verify_pareto_cert` in `crates/regsynth-certificate`. |
| `./implementation/target/release/regsynth benchmark --iterations 2 --num-obligations 10 --num-jurisdictions 2 --output-format json` | **2/2 feasible** toy runs; average **14.5 constraints**, average frontier size **1.0**, and all reported stage timings rounded to **0 ms** on this tiny synthetic input. |
| `python3 run_benchmark.py` | Re-ran the bundled 8-scenario synthetic benchmark and refreshed `benchmarks/real_benchmark_results.json`. **MaxSMT_Synthesis succeeded on 5/8 scenarios (62.5%)**, averaged **0.3219 hypervolume** and **10.8 ms**. **Epsilon_Constraint** had the best average hypervolume (**0.4828**). MaxSMT failed on `finserv_consumer_protection` and `environmental_multimedia` because the current script still contains symbolic bounds that do not parse as floats. |
| `python3 benchmarks/pareto_dominance_validation.py --runs 5 --bf-runs 10 --traj-runs 20` | Validation harness completed and rewrote `benchmarks/results/pareto_dominance_validation.json`. Summary: **100% dominance correctness**, mean hypervolume ratio **0.9840** versus brute force (minimum **0.9158**), trajectory domination observed in **15%** of sampled instances with mean hypervolume improvement **21.84%**. |

## Strongest executable core

### 1. Synthetic synthesis and trade-off exploration
The `regsynth` binary and the Python benchmark harness both exercise the same
high-level story: represent obligations as constraints, solve synthetic
instances, and examine Pareto trade-offs across cost/time/risk-style objectives.
The CLI benchmark is small, but it is real and executable.

### 2. Pareto and trajectory logic
`benchmarks/pareto_dominance_validation.py` is currently the clearest executable
evidence for the paper's optimization claims. It checks non-domination directly,
compares the MaxSMT-style enumerator against brute force on small instances, and
shows concrete cases where joint trajectory reasoning beats greedy per-step
selection.

### 3. A substantial implementation artifact
The Rust workspace in `implementation/` builds in release mode and contains the
DSL, encoding, solver, Pareto, planner, certificate, and CLI crates described
in the paper. That is a meaningful artifact even though some certificate tests
still fail and the benchmark numbers are only synthetic.

## What we intentionally do **not** claim anymore

These older claims were removed or downgraded because the reruns above do not
support them:

- no claim of solving 300+ obligation / 5+ jurisdiction workloads in fixed wall
  clock budgets;
- no claim that MaxSMT is the best-performing optimizer in the shipped benchmark
  harness;
- no claim that the benchmark suite represents full real-world regulatory
  encodings;
- no claim of production-ready infeasibility-certificate validation while the
  current certificate test failures remain;
- no claim that the repository's top-level Rust example files are turnkey entry
  points (they are source examples, but they are not wired into Cargo as
  runnable examples in this checkout).

## Broken or limited pieces we found

- `run_benchmark.py` had stale absolute paths from another checkout. It was
  repaired to use repo-relative paths so the benchmark can be rerun here.
- `cargo test --release` still exposes two real certificate-related failures.
- The default `benchmarks/highdim_pareto_benchmark.py` run is expensive and its
  generated recommendation text overstates current evidence. We inspected that
  script and its historical result file, but we did **not** use them as part of
  the recovered narrative.

## Repository areas worth reading

- `implementation/` — Rust workspace for the DSL, encoding, solvers, Pareto
  frontier logic, planner, certificate generation, and CLI.
- `benchmarks/sota_benchmark.py` — synthetic multi-objective comparison harness.
- `benchmarks/pareto_dominance_validation.py` — executable Pareto/frontier and
  trajectory validation harness.
- `benchmarks/real_benchmark_results.json` — refreshed output from the rerun
  benchmark wrapper.
- `groundings.json` — recovered grounding map with exact commands and results.
- `tool_paper.tex` / `tool_paper.pdf` — paper rewritten to match only the
  evidence above.

## Minimal reproduction

```bash
cd implementation
cargo test --release

cd ..
./implementation/target/release/regsynth benchmark --iterations 2 --num-obligations 10 --num-jurisdictions 2 --output-format json
python3 run_benchmark.py
python3 benchmarks/pareto_dominance_validation.py --runs 5 --bf-runs 10 --traj-runs 20
```

## Current status

RegSynth is best understood as a **prototype synthesis engine plus synthetic
validation harness**. That is still a compelling story: the repo contains real
code for formal encodings, Pareto reasoning, and temporal trajectory analysis.
The honest limitation is that the current evidence stops well short of proving
industrial-scale legal compliance performance.
