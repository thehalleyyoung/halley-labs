# Changelog

All notable changes to BiCut will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2025-07-18

### Added

- **Bilevel IR & Parser**: Typed intermediate representation with leader/follower
  variable scoping, integrality annotations, and constraint classification.
  TOML-based problem specification format.

- **Structural Analysis** (`bicut-core`): Convexity detection, three-tier
  constraint qualification verification (LICQ, MFCQ, Slater), boundedness
  analysis, coupling strength classification, and dependency graph analysis.

- **Reformulation Selection**: Automatic strategy selection based on problem
  signature mapping to KKT, strong duality, value function, or CCG.

- **KKT Reformulation Pass**: Complementarity encoding via big-M (with
  automatic bound-tightening LPs), SOS1 sets, or indicator constraints.

- **Strong Duality Pass**: Primal-dual pairing for LP lower levels,
  eliminating complementarity constraints entirely.

- **Value Function Pass**: Parametric LP with piecewise-linear value function
  for continuous lower levels; sampling-based approximation for MILP.

- **Column-and-Constraint Generation (CCG) Pass**: Iterative decomposition
  for general bilevel programs with finite convergence guarantee.

- **Bilevel Intersection Cuts** (`bicut-cuts`): Novel valid inequalities from
  Balas's framework applied to the bilevel-infeasible set. Separation oracle
  with parametric caching (>90% hit rate). Cut pool with age-based purging.

- **Value Function Oracle** (`bicut-value-function`): Exact parametric LP
  evaluation, critical region computation, Gomory–Johnson lifting, sampling-
  based MILP approximation with error bounds.

- **LP Solver** (`bicut-lp`): Revised simplex, dual simplex, and interior
  point methods. Tableau operations, basis management, MPS parser.

- **Solver Backends**: Emission to Gurobi (.lp), SCIP (.lp), and HiGHS (.mps)
  with backend-specific optimizations (indicator constraints, SOS1 sets,
  lazy constraints).

- **Correctness Certificates**: Machine-checkable proofs of reformulation
  validity including CQ status, boundedness, and integrality verification.

- **Branch-and-Cut Framework** (`bicut-branch-cut`): Node management,
  branching strategies, cut callbacks, bounding, and heuristics.

- **Benchmark Harness** (`bicut-bench`): BOBILib integration, MibS comparison,
  performance profiling, and automated reporting.

- **CLI** (`bicut-cli`): Commands for compile, solve, cut-loop, analyze,
  and bench. TOML configuration support.

- **Documentation**: Comprehensive README with architecture diagrams,
  theory overview, CLI reference, and examples.

- **Tool Paper**: LaTeX document with SOTA comparison, 4 experiments,
  and real citations.

### Technical Details

- Rust workspace with 9 crates, ~62,000 lines of code
- Dual-licensed: MIT / Apache-2.0
- Targets: BOBILib (2600+ MIBLP instances)
- Baselines: MibS, CPLEX/Gurobi native bilevel, KKT reformulation
