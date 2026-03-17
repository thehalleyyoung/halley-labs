# NegSynth Architecture

## Pipeline Overview

NegSynth implements a six-phase pipeline that transforms C source code into
either concrete downgrade attack traces or bounded-completeness certificates.

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐
│  C Source    │───▶│ LLVM Bitcode │───▶│ Protocol-Aware│
│  (OpenSSL,  │    │ (via Clang/  │    │ Slicer        │
│  WolfSSL,   │    │  wllvm)      │    │               │
│  libssh2)   │    └──────────────┘    └───────┬───────┘
└─────────────┘                                │
                                    Negotiation Slice
                                    (~3-7K instructions)
                                               │
                                    ┌──────────▼───────────┐
                                    │  KLEE + Protocol-    │
                                    │  Aware Merge         │
                                    │  Operator            │
                                    └──────────┬───────────┘
                                               │
                                    Symbolic Execution Traces
                                    (10-100x path reduction)
                                               │
                                    ┌──────────▼───────────┐
                                    │  State Machine       │
                                    │  Extractor           │
                                    │  (Bisimulation       │
                                    │   quotient)          │
                                    └──────────┬───────────┘
                                               │
                                    ┌──────────▼───────────┐
                                    │  DY+SMT Encoder      │
                                    │  (BV+Arrays+UF+LIA)  │
                                    └──────────┬───────────┘
                                               │
                                    ┌──────────▼───────────┐
                                    │  Z3 + CEGAR Loop     │
                                    └──────────┬───────────┘
                                               │
                              ┌────────────────┴────────────────┐
                              ▼                                 ▼
                   Attack Trace                    Bounded-Completeness
                   (byte-level)                    Certificate (k, n)
```

## Phase Details

### Phase 1: Protocol-Aware Slicing (`negsyn-slicer`)

The slicer identifies negotiation-relevant code in the target library using:

- **Protocol-specific taint tracking**: Seeds from cipher suite arrays, version
  fields, and extension parsers
- **Context-sensitive points-to analysis**: Resolves indirect calls through
  SSL_METHOD vtables and callback chains
- **Call graph construction**: Handles macro-generated dispatch tables
- **Validation**: Ensures the slice preserves all negotiation-observable behaviors

Target: reduce 100K+ LoC to ~3-7K instruction slice (≤2% of source).

### Phase 2: Symbolic Execution with Merge (`negsyn-merge`)

Executes the negotiation slice symbolically using KLEE, extended with the
protocol-aware merge operator that exploits four algebraic properties:

1. **Finite outcome spaces**: Cipher suites form enumerated sets
2. **Lattice-ordered preferences**: Security levels form a lattice
3. **Monotonic state progression**: Handshake phases progress acyclically
4. **Deterministic selection**: Given matching capabilities, outcome is determined

This achieves O(n) symbolic paths where generic merging produces O(2^n).

### Phase 3: State Machine Extraction (`negsyn-extract`)

Extracts a labeled transition system (LTS) from symbolic execution traces:

- Observation function maps symbolic states to protocol-observable behaviors
- Bisimulation quotient reduces the LTS to its minimal form
- Preserves trace equivalence with the original program execution

### Phase 4: DY+SMT Encoding (`negsyn-encode`)

Encodes the state machine together with a bounded Dolev-Yao adversary:

- **Bitvector theory**: Wire-format byte encoding
- **Array theory**: Cipher suite lists and message buffers
- **Uninterpreted functions**: Cryptographic operations
- **Linear integer arithmetic**: Message counters and sequence numbers
- **Dolev-Yao rules**: Adversary knowledge accumulation

### Phase 5: CEGAR Solving (`negsyn-concrete`)

Iterative solving with counterexample-guided abstraction refinement:

- SAT → concretize to byte-level attack trace
- UNSAT → generate bounded-completeness certificate
- Refinement failure → add constraints and re-solve
- Timeout → decompose and retry with incremental solving

### Phase 6: Validation

For attack traces: replay via TLS-Attacker to confirm exploitability.
For certificates: validate coverage metrics and bound adequacy.

## Crate Dependencies

```
negsyn-types ────┬── negsyn-slicer
                 ├── negsyn-merge
                 ├── negsyn-extract
                 ├── negsyn-encode
                 ├── negsyn-concrete
                 ├── negsyn-proto-tls
                 ├── negsyn-proto-ssh
                 ├── negsyn-eval ──── (all above)
                 └── negsyn-cli ───── (all above)
```
