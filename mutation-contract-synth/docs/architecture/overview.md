# MutSpec Architecture Overview

## System Architecture

MutSpec is organized as a Rust workspace of 10 crates, each handling a
distinct stage of the mutation-contract synthesis pipeline.

```
┌─────────────────────────────────────────────────────────────────┐
│                          CLI (mutspec)                          │
│  mutate │ analyze │ synthesize │ verify │ report │ config       │
├─────────┴─────────┴────────────┴────────┴────────┴─────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ gap-analysis  │  │contract-synth│  │      coverage        │  │
│  │  ┌─────────┐  │  │ ┌──────────┐│  │ ┌──────┐ ┌────────┐ │  │
│  │  │ witness │  │  │ │ lattice  ││  │ │ dom  │ │subsump.│ │  │
│  │  │ ranking │  │  │ │ walk     ││  │ │ set  │ │scoring │ │  │
│  │  │ equiv   │  │  │ └──────────┘│  │ └──────┘ └────────┘ │  │
│  │  └─────────┘  │  └──────────────┘  └──────────────────────┘  │
│  └──────────────┘                                               │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │program-analys│  │  smt-solver  │  │    mutation-core      │  │
│  │ parser, CFG  │  │ incremental  │  │ operators, kill mat.  │  │
│  │ SSA, WP eng  │  │ s-exp parser │  │ mutant descriptors    │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐                            │
│  │pit-integration│  │   test-gen   │                            │
│  │ PIT XML parse │  │ test types   │                            │
│  └──────────────┘  └──────────────┘                            │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                      shared-types                           ││
│  │  AST · IR (SSA) · Formula · Contract · Config · Errors      ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Input**: Source program (`.ms` files) + test suite or PIT XML report
2. **Parsing** (`program-analysis`): Source → AST → IR (SSA form)
3. **Mutation** (`mutation-core`): IR → Mutant set + Kill matrix
4. **WP Differencing** (`program-analysis`): Original ⊕ Mutant → Error predicates
5. **Synthesis** (`contract-synth`): Error predicates → Discrimination lattice → Contracts
6. **Verification** (`smt-solver`): Contracts → SMT queries → Verified contracts
7. **Gap Analysis** (`gap-analysis`): Survived mutants + Contracts → Bug reports
8. **Output** (`cli`): SARIF reports, JML annotations, JSON specifications

## Crate Dependency Graph

```
shared-types ← mutation-core ← program-analysis ← contract-synth
     ↑              ↑                  ↑                ↑
     │              │                  │                │
     ├── smt-solver ┘                  │                │
     ├── coverage ─────────────────────┘                │
     ├── gap-analysis ──────────────────────────────────┘
     ├── pit-integration
     ├── test-gen
     └── cli (depends on all above)
```
