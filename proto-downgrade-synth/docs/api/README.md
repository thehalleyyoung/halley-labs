# NegSynth API Reference

## Core Types (`negsyn-types`)

### Protocol Types

- `HandshakePhase` — Enum representing TLS/SSH handshake phases
- `ProtocolVersion` — Protocol version with major/minor fields and ordering
- `CipherSuite` — Cipher suite with key exchange, authentication, encryption, MAC
- `NegotiationState` — Current state of cipher-suite negotiation
- `NegotiationLTS` — Labeled transition system for negotiation flows
- `TransitionLabel` — Labels on LTS transitions (client/server/adversary actions)
- `SecurityLevel` — Security classification (Critical, High, Medium, Low, None)

### Symbolic Types

- `SymbolicState` — Full symbolic execution state (memory, constraints, negotiation)
- `SymbolicValue` — Symbolic expressions over bitvectors, booleans, arrays
- `PathConstraint` — Branch condition along a symbolic path
- `SymbolicMemory` — Symbolic memory model with regions and permissions
- `ExecutionTree` — Tree of symbolic execution paths

### SMT Types

- `SmtExpr` — SMT-LIB expression AST
- `SmtFormula` — Complete SMT formula with declarations and assertions
- `SmtResult` — Solver result (SAT/UNSAT/Unknown/Timeout)
- `SmtModel` — Satisfying assignment from solver
- `SmtSort` — SMT sort (Bool, BitVec, Array, Int, Real)

### Adversary Types

- `BoundedDYAdversary` — Bounded Dolev-Yao adversary model
- `AdversaryBudget` — Budget (max messages, interceptions, modifications)
- `KnowledgeSet` — Accumulated adversary knowledge
- `MessageTerm` — Dolev-Yao message algebra terms
- `AdversaryAction` — Individual adversary actions (intercept, modify, inject)

### Certificate Types

- `Certificate` — Bounded-completeness certificate
- `AttackTrace` — Concrete attack trace with protocol steps
- `AnalysisResult` — Union of attack trace or certificate
- `BoundsSpec` — Analysis bounds (depth k, budget n)

### Configuration Types

- `AnalysisConfig` — Top-level analysis configuration
- `SlicerConfig` — Protocol-aware slicer parameters
- `MergeConfig` — Merge operator configuration
- `ExtractionConfig` — State machine extraction parameters
- `EncodingConfig` — SMT encoding options
- `ConcretizerConfig` — CEGAR concretization settings

## Pipeline Crates

### `negsyn-slicer`
Protocol-aware program slicing. Reduces LLVM IR to negotiation-relevant subset.

### `negsyn-merge`
Protocol-aware merge operator. Exploits algebraic structure for path reduction.

### `negsyn-extract`
Bisimulation-quotient state machine extraction from symbolic execution traces.

### `negsyn-encode`
Dolev-Yao + SMT constraint encoding in combined BV+Arrays+UF+LIA theory.

### `negsyn-concrete`
CEGAR-based concretization producing byte-level attack traces or certificates.

### `negsyn-proto-tls`
TLS 1.0–1.3 protocol models: handshake state machines, cipher suite negotiation, extensions.

### `negsyn-proto-ssh`
SSH v2 protocol models: key exchange, algorithm negotiation, extensions.

### `negsyn-eval`
Evaluation harness with CVE oracle, coverage metrics, and benchmark framework.

### `negsyn-cli`
Command-line interface: analyze, verify, diff, replay, benchmark, inspect.
