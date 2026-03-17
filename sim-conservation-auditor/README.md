# ConservationLint

ConservationLint is best understood today as a **tested Rust workspace for conservation-law computations, numerical integrators, detectors, and runnable physics examples**.

This README intentionally avoids the earlier unsupported story about externally validated commercial-simulator performance or broad superiority claims. The strongest honest evidence in this repository is the executable core that we reran on 2026-03-17.

## Honest status

What is well supported now:

- a modular Rust implementation of conservation-related types, laws, integrators, detectors, and a runtime monitor;
- a real end-to-end CLI audit path over JSON traces, with JSON/SARIF/Markdown/HTML reporting;
- runnable examples that demonstrate expected conservation behavior and failure modes;
- a large passing workspace test suite;
- a small Python benchmark harness that models multi-law vs single-law detection on synthetic Kepler traces.

What is **not** currently supported by rerun evidence:

- claims about accuracy on MATLAB, ANSYS, Gazebo, COMSOL, Unity, or other commercial simulators;
- claims of state-of-the-art detection rates on external workloads;
- the old 25-kernel `benchmarks/run_benchmarks.sh` story.

## Strongest executable core

The most convincing working part of the repo is the Rust workspace in `implementation/`:

- `sim-laws`: conservation quantities such as energy, momentum, angular momentum, mass, charge, vorticity, and symplectic structure;
- `sim-integrator`: explicit, symplectic, implicit, and composition integrators;
- `sim-detect`: statistical and drift-detection utilities;
- `sim-monitor`: a runtime event monitor for conservation drift;
- `sim-cli`: a small CLI layer;
- `examples/`: standalone executable demonstrations.

That core is reinforced by runnable examples rather than by external benchmark reproduction.

## Rerun evidence

All claims below are tied to commands rerun in this repository on 2026-03-17.

### 1) Workspace tests pass

```bash
cargo test --manifest-path implementation/Cargo.toml --workspace
```

Observed result:

- workspace tests completed successfully after small numeric/test repairs;
- 381 tests passed across the workspace;
- one doctest is ignored;
- no test failures remained.

### 2) The CLI now performs real multi-law audits

```bash
cargo run --manifest-path implementation/Cargo.toml -p sim-cli -- \
  audit examples/traces/clean_orbit_trace.json \
  --config examples/traces/audit_strict_config.json \
  --format json

cargo run --manifest-path implementation/Cargo.toml -p sim-cli -- \
  audit examples/traces/drift_orbit_trace.json \
  --config examples/traces/audit_strict_config.json \
  --format sarif
```

Observed result:

- the bundled clean two-law trace passed with `0` violating laws in about `0.116 ms`;
- the bundled drifting two-law trace failed with `2` violating laws in about `0.389 ms`;
- the SARIF output contained `2` populated results, one per violated law;
- the `report` subcommand also rendered the same audit JSON as Markdown and HTML.

This is materially stronger evidence than the previous `list laws` check: the CLI is now a real audit entry point, not just a registry browser.

### 3) The CLI still exposes a conservation-law registry

```bash
cargo run --manifest-path implementation/Cargo.toml -p sim-cli -- list laws
```

Observed result:

- the CLI listed 8 laws: energy, linear momentum, angular momentum, mass, charge, symplectic form, vorticity, and center-of-mass.

This is modest evidence, but it confirms the CLI binary builds and the law registry is wired up.

### 4) The Kepler integrator example cleanly distinguishes drift behavior

```bash
cargo run --manifest-path implementation/Cargo.toml --example benchmark_integrators
```

Observed result on the shipped 100,000-step Kepler setup:

| Integrator | Max `|ΔE/E|` | Max `|ΔL/L|` | Final `|ΔE/E|` |
|---|---:|---:|---:|
| Forward Euler | `9.7763e-1` | `3.8643e-1` | `9.7610e-1` |
| RK4 | `3.5120e-6` | `3.8382e-7` | `3.5336e-6` |
| Velocity Verlet | `7.4136e-4` | `3.3862e-14` | `1.8638e-4` |
| Yoshida 4th order | `4.0995e-7` | `8.0491e-14` | `4.0855e-7` |
| Yoshida 6th order | `1.3186e-2` | `6.7168e-14` | `9.6148e-3` |

Honest takeaway: the example really does show that the repository can reproduce qualitatively different conservation behavior across integrators. It does **not** prove universal method rankings.

### 5) The N-body audit example preserves invariants to tight tolerances

```bash
cargo run --manifest-path implementation/Cargo.toml --example nbody_audit
```

Observed result on the shipped 50,000-step three-body example:

- max relative energy drift: `5.8920e-7`;
- max momentum deviation: `2.7437e-14`;
- max angular-momentum deviation: `9.9920e-15`;
- `0` checks exceeded the example tolerance `1e-6`.

This is the clearest end-to-end demonstration that the repository can integrate a nontrivial system and audit multiple conserved quantities at runtime.

### 6) The fluid example demonstrates both preserved and intentionally dissipative behavior

```bash
cargo run --manifest-path implementation/Cargo.toml --example fluid_audit
```

Observed result:

- linear advection preserved mass to machine precision (`max |Δm/m| = 6.6527e-16`);
- the upwind scheme lost energy (`max |ΔE/E| = 1.1295e-1`), which the example labels as expected dissipation;
- the Burgers example reports expected shock-related energy loss and a mass warning/failure.

Honest takeaway: the fluid example is useful as a didactic audit of what should and should not be conserved under the chosen numerics.

### 7) The Python baseline benchmark is executable, but it is a model benchmark

```bash
python3 benchmarks/baseline_comparison.py --output benchmarks/baseline_comparison_results.rerun.json
```

Observed summary:

| Method | Detection rate | False-positive rate | Avg latency (steps) | Laws monitored |
|---|---:|---:|---:|---|
| adaptive_threshold | 60% | 0% | 1275.3 | energy |
| cusum | 70% | 0% | 439.9 | energy |
| zscore | 40% | 0% | 40.0 | energy |
| gromacs_style | 60% | 0% | 1342.8 | energy |
| conservationlint | 100% | 0% | 319.9 | angular_momentum, energy |

Important caveat: this script does **not** invoke the Rust CLI. It benchmarks a Python-side model of the intended multi-law detector on synthetic traces. It is useful as an executable design artifact, not as external validation of deployed tool performance.

### 8) The CLI stays fast on a larger generated trace

```bash
python3 benchmarks/generate_cli_trace.py --output /tmp/sim-large-trace.json --samples 10000 --mode drift
cargo run --manifest-path implementation/Cargo.toml -p sim-cli -- \
  audit /tmp/sim-large-trace.json \
  --config examples/traces/audit_strict_config.json \
  --format json
```

Observed result:

- the generated trace contained `20,000` total samples across two monitored laws;
- the CLI completed the audit in about `7.2 ms`;
- both laws were flagged as violating under the strict config.

Honest takeaway: the new CLI path is not just functional, it is also lightweight enough to be practical on moderate trace sizes.

## Broken benchmark path that we did not try to spin into evidence

We also reran:

```bash
bash benchmarks/run_benchmarks.sh --quick
```

Observed result:

- the script failed immediately because it expects a binary at repo-root `target/release/conservation-lint`;
- this repository actually builds under `implementation/target/...` when using the documented workspace manifest path;
- separate inspection also shows that `benchmarks/kernels/` is missing.

So the old benchmark-runner narrative is currently broken and should not be treated as evidence.

## Practical conclusion

The honest story is not “ConservationLint beat production tools on a large benchmark.”

The honest story is:

> this repository contains a real, tested implementation of conservation-law utilities and numerics, plus runnable examples and a working multi-law audit CLI that demonstrate conservation-preserving and conservation-breaking behavior in controlled settings.

That is still a useful and interesting artifact—especially for readers interested in numerical methods, invariant monitoring, and executable examples of conservation-aware simulation tooling.

## Reproducing the verified evidence

```bash
cargo build --manifest-path implementation/Cargo.toml --workspace
cargo test --manifest-path implementation/Cargo.toml --workspace
cargo run --manifest-path implementation/Cargo.toml -p sim-cli -- audit implementation/examples/traces/clean_orbit_trace.json --config implementation/examples/traces/audit_strict_config.json --format json
cargo run --manifest-path implementation/Cargo.toml -p sim-cli -- audit implementation/examples/traces/drift_orbit_trace.json --config implementation/examples/traces/audit_strict_config.json --format sarif
cargo run --manifest-path implementation/Cargo.toml -p sim-cli -- list laws
cargo run --manifest-path implementation/Cargo.toml --example benchmark_integrators
cargo run --manifest-path implementation/Cargo.toml --example nbody_audit
cargo run --manifest-path implementation/Cargo.toml --example fluid_audit
python3 benchmarks/generate_cli_trace.py --output /tmp/sim-large-trace.json --samples 10000 --mode drift
python3 benchmarks/baseline_comparison.py --output benchmarks/baseline_comparison_results.rerun.json
bash benchmarks/run_benchmarks.sh --quick   # expected to fail in current repo state
```

## Limitations

- The strongest evidence is from unit tests and shipped examples, not external simulation corpora.
- The commercial-simulator benchmark scripts are synthetic/model-based and should not be read as measurements on proprietary tools.
- The new CLI audit path is grounded on bundled/generated trace fixtures, not on external simulator exports.
- The benchmark runner script in `benchmarks/run_benchmarks.sh` is not currently usable as-is.
