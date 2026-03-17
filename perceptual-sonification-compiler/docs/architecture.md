# SoniType Architecture

## Overview

SoniType is a domain-specific language (DSL) and optimizing compiler for
perceptually-grounded data sonification. It treats sonification as a lossy
coding problem over a psychoacoustically-constrained perceptual channel.

## Compilation Pipeline

```
                    ┌─────────────────────────────────────────┐
                    │           SoniType Compiler              │
                    └─────────────────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
              ┌──────────┐   ┌──────────────┐  ┌──────────────┐
              │ Frontend  │   │  Optimizer   │  │   Backend    │
              └──────────┘   └──────────────┘  └──────────────┘
                    │                │                │
         ┌─────────┼─────────┐     │         ┌──────┼──────┐
         ▼         ▼         ▼     ▼         ▼      ▼      ▼
      ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
      │Lexer │ │Parser│ │Type  │ │Cost  │ │Code  │ │WCET  │ │Render│
      │      │ │      │ │Check │ │Model │ │Gen   │ │      │ │      │
      └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘
```

## Phase 1: Frontend (sonitype-dsl)

The frontend parses declarative `.soni` specifications through:

1. **Lexer** (`lexer.rs`): Tokenizes the input into a stream of typed tokens
2. **Parser** (`parser.rs`): Builds an abstract syntax tree (AST) from tokens
3. **Desugaring** (`desugar.rs`): Expands syntactic sugar into core forms
4. **Type Inference** (`type_inference.rs`): Infers perceptual type qualifiers
5. **Type Checking** (`type_system.rs`): Verifies psychoacoustic constraints

### Perceptual Type System

The type system treats psychoacoustic constraints as first-class type qualifiers:

- **Masking clearance**: Ensures streams don't mask each other across Bark bands
- **JND validation**: Verifies parameter differences exceed just-noticeable thresholds
- **Stream segregation**: Checks Bregman's criteria for auditory stream formation
- **Cognitive load**: Enforces Cowan's 4±1 working memory limit

## Phase 2: Optimizer (sonitype-optimizer)

The optimizer maximizes psychoacoustically-constrained mutual information I_ψ(D;A):

1. **Constraint Propagation** (`propagation.rs`): Prunes infeasible regions
2. **Bark-Band Decomposition** (`decomposition.rs`): Decomposes across frequency bands
3. **Branch-and-Bound** (`branch_and_bound.rs`): Searches parameter space
4. **Pareto Optimization** (`pareto.rs`): Balances multiple objectives
5. **Mutual Information** (`mutual_information.rs`): Computes I_ψ objective

## Phase 3: Backend (sonitype-codegen + sonitype-renderer)

The backend compiles the optimized audio graph:

1. **Lowering** (`lowering.rs`): Converts IR to concrete audio operations
2. **Scheduling** (`scheduler.rs`): Orders operations for buffer efficiency
3. **Code Generation** (`codegen.rs`): Emits the runtime audio graph
4. **WCET Analysis** (`wcet.rs`): Bounds worst-case execution time
5. **Rendering** (`render.rs`): Executes the audio graph in real-time

## Crate Dependency Graph

```
sonitype-cli
├── sonitype-dsl
│   ├── sonitype-core
│   └── sonitype-psychoacoustic
│       └── sonitype-core
├── sonitype-ir
│   ├── sonitype-core
│   └── sonitype-psychoacoustic
├── sonitype-optimizer
│   ├── sonitype-core
│   ├── sonitype-ir
│   └── sonitype-psychoacoustic
├── sonitype-codegen
│   ├── sonitype-core
│   ├── sonitype-ir
│   └── sonitype-optimizer
├── sonitype-renderer
│   └── sonitype-core
├── sonitype-stdlib
│   ├── sonitype-core
│   └── sonitype-psychoacoustic
├── sonitype-accessibility
│   ├── sonitype-core
│   └── sonitype-psychoacoustic
└── sonitype-streaming
    ├── sonitype-core
    └── sonitype-renderer
```

## Key Data Structures

### Audio Graph (IR)

The intermediate representation is a directed acyclic graph where:
- **Nodes** represent audio processing operations (oscillators, filters, mixers)
- **Edges** represent buffer connections between nodes
- **Annotations** carry psychoacoustic metadata (Bark-band occupancy, masking margins)

### Perceptual Resource Vector

Each stream carries a resource vector:
- 24-dimensional Bark-band spectral energy
- Cognitive load count
- Temporal density measure

The resource algebra supports composition via the ⊕ operator with cross-band
masking interaction correction.

## Psychoacoustic Models

See [psychoacoustic_models.md](psychoacoustic_models.md) for detailed model documentation.
