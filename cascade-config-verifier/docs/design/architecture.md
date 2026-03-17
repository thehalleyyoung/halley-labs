# CascadeVerify Architecture

## Overview

CascadeVerify is structured as a 10-crate Rust workspace implementing a
two-tier static analysis pipeline for detecting cascade risks in microservice
configurations.

## Data Flow

```
Config Files (YAML/JSON)
    │
    ▼
┌─────────────┐
│ cascade-    │  Parse K8s, Istio, Envoy, Helm, Kustomize
│ config      │  Resolve references, expand templates
└─────┬───────┘
      │  Unified config model
      ▼
┌─────────────┐
│ cascade-    │  Build RTIG from config model
│ graph       │  Compute graph properties (SCC, treewidth, symmetry)
└─────┬───────┘
      │  RTIG
      ▼
┌─────────────┐   ┌──────────────┐
│ cascade-    │   │ cascade-     │
│ analysis    │──▶│ bmc          │  Encode RTIG as QF_LIA, run BMC
│ (Tier 1+2)  │   │              │  Enumerate MUS via MARCO
└─────┬───────┘   └──────────────┘
      │  Cascade risks
      ▼
┌─────────────┐   ┌──────────────┐
│ cascade-    │──▶│ cascade-     │
│ repair      │   │ maxsat       │  Encode repair as weighted MaxSAT
│             │   │              │  Enumerate Pareto frontier
└─────┬───────┘   └──────────────┘
      │  Repair patches
      ▼
┌─────────────┐
│ cascade-    │  SARIF, JUnit, JSON, YAML, terminal output
│ verify      │  CI/CD integration, caching, diffing
└─────┬───────┘
      │
      ▼
┌─────────────┐
│ cascade-    │  CLI entry point
│ cli         │
└─────────────┘
```

## Key Design Decisions

1. **Two-tier architecture**: Tier 1 (graph) is fast enough for CI/CD;
   Tier 2 (BMC) is exhaustive for audit.

2. **Monotonicity exploitation**: The CB-free model enables sound antichain
   pruning, which is the key enabler for practical MUS enumeration.

3. **MaxSAT for repair**: Natural expression of "change as little as possible
   while guaranteeing cascade-freedom" — better than CEGIS for this domain.

4. **Rust workspace**: Type safety, performance, and the crate boundary
   enforces clean module separation.

5. **Built-in domain semantics**: No manual specification required — the tool
   understands K8s/Istio/Envoy retry and timeout primitives natively.
