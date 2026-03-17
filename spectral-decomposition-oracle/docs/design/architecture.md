# Architecture — Spectral Decomposition Oracle

## Overview

The Spectral Decomposition Oracle is a system that uses spectral analysis of
constraint matrices to guide the selection of decomposition methods for
mathematical optimization problems. Given an LP/MIP, it extracts spectral
features from the constraint hypergraph Laplacian, classifies the problem, and
recommends the best decomposition strategy (Benders, Dantzig-Wolfe, or
Lagrangian relaxation) — or predicts that decomposition is futile.

## Crate Dependency Diagram

```
                  ┌──────────────────┐
                  │  spectral-types  │  Common types: sparse matrices, config,
                  │                  │  error types, numerical tolerances
                  └────────┬─────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
     ┌────────▼───┐  ┌────▼──────┐  ┌──▼──────────┐
     │matrix-decomp│  │spectral-  │  │optimization │
     │             │  │  core     │  │             │
     │ LU, QR,    │  │ Hypergraph│  │ LP solvers  │
     │ Cholesky,  │  │ Laplacian │  │ Benders, DW │
     │ SVD,       │  │ Eigensolve│  │ Lagrangian  │
     │ Lanczos,   │  │ Features  │  │ Partition   │
     │ LOBPCG     │  │ Clustering│  │             │
     └────────┬───┘  └────┬──────┘  └──────┬──────┘
              │            │               │
              └────────────┼───────────────┘
                           │
                    ┌──────▼──────┐
                    │   oracle    │
                    │             │
                    │ Classifier  │
                    │ Futility    │
                    │ Evaluation  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ certificate │
                    │             │
                    │ L3 bound    │
                    │ Davis-Kahan │
                    │ Scaling law │
                    │ Dual check  │
                    │ Reports     │
                    └──────┬──────┘
                           │
                ┌──────────▼──────────┐
                │ spectral-decomp-cli │
                │                     │
                │ CLI entry point     │
                │ I/O, orchestration  │
                └─────────────────────┘
```

## Data Flow

```
  LP/MIP Instance (MPS/LP file)
         │
         ▼
  ┌─── Parse constraint matrix A ───┐
  │                                  │
  ▼                                  ▼
  Syntactic features           Constraint hypergraph
  (rows, cols, density,        (rows → hyperedges,
   structure stats)             cols → vertices)
                                     │
                                     ▼
                               Normalized Laplacian
                               (clique or Bolla)
                                     │
                                     ▼
                               Eigensolve (k=8)
                               (ARPACK → LOBPCG)
                                     │
                                     ▼
                               8 spectral features
                                     │
         ┌───────────────────────────┘
         ▼
  Combined feature vector (spectral + syntactic)
         │
         ├──► Ensemble classifier ──► Method choice
         │                            {Benders, DW, Lagrangian}
         │
         └──► Futility predictor ──► P(futile)
                                      │
         ┌────────────────────────────┘
         ▼
  Decomposition execution (or bail if futile)
         │
         ▼
  Certificate: dual feasibility, L3 bound, T2 scaling law
         │
         ▼
  JSON report with visualization data
```

## Key Design Decisions

1. **Hypergraph, not graph.** Constraints naturally form hyperedges (one
   constraint touches multiple variables). We use the hypergraph Laplacian
   rather than reducing to a graph first, with a threshold-based fallback:
   clique expansion when d_max ≤ 200, Bolla incidence otherwise.

2. **Spectral features over structural heuristics.** The 8 spectral features
   (spectral gap, Cheeger estimate, etc.) capture global connectivity
   properties that simple syntactic features (density, degree distribution)
   miss. Ablation shows ≥5pp improvement.

3. **ARPACK-first eigensolve with LOBPCG fallback.** ARPACK (implicit restart
   Lanczos with shift-invert) is fast for well-conditioned problems. LOBPCG
   handles the cases where ARPACK fails to converge, particularly for
   near-singular Laplacians.

4. **Futility prediction as a first-class concern.** Rather than always
   attempting decomposition, the oracle estimates P(futile) and can recommend
   direct solve. This saves compute on problems where the constraint structure
   is too tightly coupled for decomposition to help.

5. **Formal certificates.** Every recommendation is accompanied by a
   verifiable certificate: Davis-Kahan perturbation bounds, the L3 partition
   bound, and the T2 scaling law. This makes the oracle auditable.

6. **Calibrated probabilities.** Temperature scaling ensures the classifier's
   confidence estimates are well-calibrated, not just discriminative. This
   matters for the futility threshold and for downstream decision-making.

7. **Seven-crate workspace.** The system is split into focused crates to
   enforce separation of concerns: types, matrix algebra, spectral analysis,
   optimization, ML oracle, certificates, and CLI. Each crate has a clear API
   boundary and can be tested independently.
