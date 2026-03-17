# NegSynth Algorithmic Design Document

**Project:** Negotiation Under Fire: Bounded-Complete Synthesis of Protocol Downgrade Attacks from Library Source Code  
**Component:** Algorithmic Core — Detailed Design & Pseudocode  
**Status:** Design Proposal  

---

## 0. Notation and Shared Definitions

Throughout this document we use the following conventions.

| Symbol | Meaning |
|--------|---------|
| `IR` | LLVM IR bitcode module |
| `V` | Set of LLVM IR variables (SSA values) |
| `E` | Edges in the call/control-flow graph |
| `Σ` | Symbolic state: `(pc, σ, π)` — program counter, symbolic store, path condition |
| `C` | Finite set of cipher-suite identifiers (IANA IDs), \|C\| = n |
| `Φ` | Handshake phase ∈ {`ClientHello`, `ServerHello`, `Certificate`, `KeyExchange`, `Finished`, ...}, \|Φ\| = m |
| `A` | Dolev-Yao adversary with budget of at most `b` actions |
| `LTS` | Labeled transition system `(S, L, →)` |
| `≈_π` | Protocol-bisimulation equivalence |
| `k` | Symbolic execution depth bound |
| `BV[w]` | SMT bitvector of width `w` bits |

Data-structure shorthands map to Rust types: `BTreeMap<K,V>` for ordered maps, `Vec<T>` for dynamic arrays, `HashSet<T>` for unordered sets, `BitVec` for fixed-width bitvectors.

---

## 1. Algorithm 1 — PROTOSLICE: Protocol-Aware Slicer

### 1.1 Purpose

Given the full LLVM IR bitcode of a cryptographic library (200K–500K+ lines), PROTOSLICE extracts the *negotiation-relevant* subset: code that can influence which cipher suite, protocol version, or extension set is selected during a TLS/SSH handshake. The target is ≤2% of the original source (~3–7K lines), enabling downstream symbolic execution to complete on a laptop CPU.

### 1.2 Interface

```
fn protoslice(
    ir: &Module,                          // Full LLVM IR bitcode
    entry_points: &[FunctionName],        // e.g., ["SSL_connect", "SSL_accept"]
    negotiation_vars: &[VariableSpec],    // e.g., cipher_suite_selected, version_selected
    protocol_spec: &ProtocolSpec,         // Phase definitions, known vtable patterns
) -> SlicedModule
```

**Preconditions:**
- `ir` is a valid, linked LLVM IR module (produced by wllvm/gclang).
- Every name in `entry_points` resolves to a function in `ir`.
- `negotiation_vars` identifies SSA values or global addresses corresponding to negotiation outcomes (e.g., `s->session->cipher`, `s->version`).

**Postconditions:**
- The returned `SlicedModule` contains every instruction that can influence any variable in `negotiation_vars` along any path reachable from `entry_points`.
- Soundness: no negotiation-relevant instruction is omitted (see §1.5).

### 1.3 Approach — Two-Phase Slicing

**Phase 1: Coarse Backward Slice.** Starting from `negotiation_vars`, compute a backward program dependence slice over the interprocedural control-flow graph (ICFG). This uses Andersen-style flow-insensitive points-to analysis to resolve indirect calls.

**Phase 2: Protocol-Aware Taint Refinement.** The coarse slice over-approximates because it follows all data and control dependencies regardless of semantic relevance. Phase 2 prunes the slice using protocol-specific taint labels:

- **CIPHER_TAINT**: any value that flows into cipher-suite selection or ordering.
- **VERSION_TAINT**: any value that flows into protocol-version comparison.
- **EXT_TAINT**: any value that flows into extension negotiation (e.g., SNI, ALPN, signature algorithms).
- **UNTAINTED**: reachable from entry points but does not influence any negotiation outcome.

Phase 2 propagates taint forward from known taint sources (cipher-suite lists, version constants, extension buffers) and intersects with the Phase 1 backward slice. Only instructions carrying at least one protocol taint label survive.

### 1.4 Handling Indirect Calls, Vtables, and Callbacks

OpenSSL dispatches protocol-version-specific logic through `SSL_METHOD` vtables — macro-generated structs of function pointers. PROTOSLICE handles these via:

1. **Vtable pattern recognition.** A pre-pass identifies structs matching the pattern `struct { fn_ptr, fn_ptr, ... }` that are assigned at known protocol-initialization sites (e.g., `TLSv1_2_method()`). Each function pointer is resolved to its concrete target.

2. **Callback chain tracking.** Functions like `SSL_CTX_set_cert_verify_callback` install user callbacks. PROTOSLICE conservatively includes the callback body if the callback's return value is tainted (i.e., feeds into a negotiation decision).

3. **Macro-generated dispatch.** `STACK_OF(SSL_CIPHER)` and `sk_SSL_CIPHER_*` macros expand to generic container operations. PROTOSLICE treats these as taint-propagating: if the container holds cipher-suite objects, all access functions are CIPHER_TAINT.

```
// Phase 1: Coarse backward slice
fn backward_slice(
    icfg: &ICFG,
    pts: &PointsToMap,          // Andersen-style, BTreeMap<Pointer, HashSet<Location>>
    seeds: &HashSet<Value>,     // negotiation_vars
) -> HashSet<Instruction> {
    let mut worklist: VecDeque<Value> = seeds.iter().cloned().collect();
    let mut slice: HashSet<Instruction> = HashSet::new();
    let mut visited: HashSet<Value> = HashSet::new();

    while let Some(val) = worklist.pop_front() {
        if !visited.insert(val) { continue; }
        // Data dependence: all instructions that define val
        for def_inst in icfg.get_definitions(val) {
            slice.insert(def_inst);
            for operand in def_inst.operands() {
                worklist.push_back(operand);
            }
        }
        // Control dependence: branch conditions controlling val's block
        for ctrl_dep in icfg.control_deps(val.parent_block()) {
            slice.insert(ctrl_dep);
            for operand in ctrl_dep.operands() {
                worklist.push_back(operand);
            }
        }
        // Indirect call resolution via points-to
        if let Some(call) = val.as_indirect_call() {
            for target in pts.resolve(call.callee_ptr()) {
                worklist.push_back(target.return_value());
                // Include callee body in ICFG traversal
                icfg.inline_callee(call, target);
            }
        }
    }
    slice
}

// Phase 2: Protocol-aware taint refinement
fn taint_refine(
    coarse_slice: &HashSet<Instruction>,
    taint_sources: &BTreeMap<Value, TaintLabel>,  // known protocol-relevant origins
    icfg: &ICFG,
) -> HashSet<Instruction> {
    let mut taint_map: BTreeMap<Value, TaintLabel> = taint_sources.clone();
    let mut worklist: VecDeque<Value> = taint_sources.keys().cloned().collect();

    // Forward taint propagation within coarse slice
    while let Some(val) = worklist.pop_front() {
        let current_taint = taint_map[&val];
        for user_inst in icfg.get_users(val) {
            if !coarse_slice.contains(&user_inst) { continue; }
            let result = user_inst.result_value();
            let propagated = propagate_taint(current_taint, user_inst);
            if propagated != TaintLabel::Untainted {
                let entry = taint_map.entry(result).or_insert(TaintLabel::Untainted);
                let merged = entry.join(propagated);  // lattice join
                if merged != *entry {
                    *entry = merged;
                    worklist.push_back(result);
                }
            }
        }
    }

    // Intersect: keep only tainted instructions from the coarse slice
    coarse_slice.iter()
        .filter(|inst| {
            inst.operands().any(|op| {
                taint_map.get(&op).map_or(false, |t| *t != TaintLabel::Untainted)
            }) || inst.results().any(|r| {
                taint_map.get(&r).map_or(false, |t| *t != TaintLabel::Untainted)
            })
        })
        .cloned()
        .collect()
}
```

### 1.5 Correctness Argument — Soundness

**Claim:** If instruction `I` in the original IR can influence the value of any negotiation outcome variable along any feasible execution path from an entry point, then `I ∈ protoslice(ir, ...)`.

**Argument sketch:**

1. Phase 1 computes a standard backward interprocedural program dependence slice. By the soundness of Weiser-style slicing (Weiser 1984, extended to interprocedural by Horwitz-Reps-Binkley 1990), every instruction with a data or control dependence chain to any seed variable is included. The use of Andersen points-to analysis (sound for C without undefined behavior) ensures indirect calls are conservatively resolved — every possible callee is included.

2. Phase 2 only *removes* instructions from the Phase 1 result; it never adds. Phase 2 is sound because taint propagation starts from *all* known taint sources (cipher-suite lists, version constants, extension buffers) and uses a forward fixpoint that is monotone on the taint lattice `{Untainted ⊑ CIPHER ⊑ ⊤, Untainted ⊑ VERSION ⊑ ⊤, Untainted ⊑ EXT ⊑ ⊤}`. An instruction survives if it carries any non-⊥ taint. The only way Phase 2 could unsoundly remove an instruction is if a taint source were missing — hence the requirement that `taint_sources` is complete, which is validated empirically against known CVE-reachable functions.

3. Vtable resolution is sound because Andersen analysis over-approximates: if a function pointer *can* point to function `f`, then `f` is included as a possible callee. The vtable pattern recognizer supplements (does not replace) points-to resolution.

### 1.6 Complexity

- **Points-to analysis (Andersen):** O(V² · E) worst-case for `V` pointer variables and `E` assignment edges, but typically near-linear with sparse constraint representation (Hardekopf & Lin, PLDI 2007). For OpenSSL: ~50K pointer variables, ~200K edges → ~10 minutes.
- **Backward slice:** O(|V| + |E|) in the ICFG, a single backward BFS. For the negotiation-relevant sub-ICFG: ~100K nodes, ~500K edges → seconds.
- **Forward taint propagation:** O(|V_slice| · h) where `h` is the height of the taint lattice (h=4 with three orthogonal labels + ⊤). `|V_slice|` ≈ 50K–100K after Phase 1 → seconds.
- **Total:** Dominated by points-to analysis. ~10 minutes for OpenSSL.

---

## 2. Algorithm 2 — PROTOMERGE: Protocol-Aware Merge Operator (CROWN ALGORITHM)

### 2.1 Purpose

PROTOMERGE is the central algorithmic contribution of NegSynth. During KLEE's symbolic exploration of the negotiation slice, PROTOMERGE identifies pairs of symbolic states that differ only in cipher-suite selection and merges them into a single state with an ITE (if-then-else) selector expression. This collapses the exponential state space of cipher-suite enumeration into a linear one.

### 2.2 Interface

```
fn protomerge(
    s1: &SymbolicState,
    s2: &SymbolicState,
    protocol_ctx: &ProtocolContext,  // phase definitions, cipher-set structure
) -> Result<SymbolicState, MergeFailure>
```

**Preconditions:**
- `s1` and `s2` are reachable KLEE symbolic states at the same program counter (basic block entry).
- `protocol_ctx` provides the handshake phase lattice and cipher-suite definitions.

**Postconditions (on Ok):**
- The returned `s_m` is protocol-bisimilar to the pair `{s1, s2}`: for every concrete valuation, `s_m` produces the same negotiation observable as whichever of `s1`, `s2` is consistent with that valuation.
- `s_m.path_condition ≡ s1.path_condition ∨ s2.path_condition`.

**Postconditions (on Err):**
- Merge is rejected; KLEE continues exploring `s1`, `s2` independently.

### 2.3 Mergeability Check — Four Algebraic Properties

Two states are *protocol-mergeable* iff all four conditions hold:

| # | Property | Formal Check | Intuition |
|---|----------|-------------|-----------|
| P1 | Same handshake phase | `phase(s1) == phase(s2)` where `phase: Σ → Φ` extracts phase from the state-machine field | Negotiation progresses monotonically; states at different phases can never reconverge |
| P2 | Same offered cipher-set structure | `offered_set(s1) == offered_set(s2)` where `offered_set` extracts the symbolic set of cipher IDs under consideration | Ensures the merged state's ITE selector ranges over a common domain |
| P3 | Compatible path conditions | `∃ prefix π_0, selectors c1 c2 ∈ C : s1.π = π_0 ∧ (sel = c1) and s2.π = π_0 ∧ (sel = c2)` — the path conditions agree on everything except which cipher was selected | States that differ in control-flow structure (not just cipher selection) may have incompatible variable bindings |
| P4 | Deterministic store agreement | For all variables `v` not in `{cipher_selected, cipher_params}`: `s1.σ(v) ≡ s2.σ(v)` modulo the cipher-selection predicate — i.e., the stores agree on all non-cipher-dependent values | If stores diverge on non-cipher variables, merging would require tracking independent symbolic expressions per branch, defeating the purpose |

### 2.4 Merge Construction

When all four checks pass, PROTOMERGE constructs the merged state:

```
fn protomerge(
    s1: &SymbolicState,
    s2: &SymbolicState,
    protocol_ctx: &ProtocolContext,
) -> Result<SymbolicState, MergeFailure> {
    // ---- MERGEABILITY CHECK ----

    // P1: Same handshake phase (monotonic progression)
    let phase1 = protocol_ctx.extract_phase(&s1.store);
    let phase2 = protocol_ctx.extract_phase(&s2.store);
    if phase1 != phase2 {
        return Err(MergeFailure::PhaseMismatch(phase1, phase2));
    }

    // P2: Same offered cipher-set structure
    let offered1 = protocol_ctx.extract_offered_ciphers(&s1.store);
    let offered2 = protocol_ctx.extract_offered_ciphers(&s2.store);
    if offered1 != offered2 {
        return Err(MergeFailure::OfferedSetMismatch);
    }

    // P3: Compatible path conditions — differ only in cipher selection
    let (common_prefix, sel1, sel2) = decompose_path_conditions(
        &s1.path_condition,
        &s2.path_condition,
        &protocol_ctx.cipher_selector_vars,
    )?;   // returns Err(IncompatiblePaths) if decomposition fails

    // P4: Deterministic store agreement on non-cipher variables
    let cipher_vars: HashSet<VarId> = protocol_ctx.cipher_dependent_vars(&s1.store);
    for (var, val1) in s1.store.iter() {
        if cipher_vars.contains(var) { continue; }
        let val2 = s2.store.get(var)
            .ok_or(MergeFailure::StoreMismatch(*var))?;
        if val1 != val2 {
            return Err(MergeFailure::StoreMismatch(*var));
        }
    }

    // ---- MERGE CONSTRUCTION ----

    // Fresh symbolic selector variable
    let selector = fresh_symbolic_bool("merge_sel");

    // Merged path condition: common prefix ∧ (sel=c1 ∨ sel=c2)
    //   equivalently: common_prefix ∧ (selector → sel1) ∧ (¬selector → sel2)
    let merged_pc = Expr::and(
        common_prefix,
        Expr::or(sel1.clone(), sel2.clone()),
    );

    // Merged store: ITE on cipher-dependent variables, copy non-cipher variables
    let mut merged_store = SymbolicStore::new();
    for (var, val1) in s1.store.iter() {
        if cipher_vars.contains(var) {
            let val2 = &s2.store[var];
            merged_store.insert(
                *var,
                Expr::ite(selector.clone(), val1.clone(), val2.clone()),
            );
        } else {
            merged_store.insert(*var, val1.clone());
        }
    }

    // Merged state inherits the program counter (guaranteed equal by KLEE
    // calling us only at matching PCs) and the common phase
    Ok(SymbolicState {
        pc: s1.pc,
        store: merged_store,
        path_condition: merged_pc,
        merge_count: s1.merge_count + s2.merge_count + 1,
        phase: phase1,
    })
}

/// Decompose two path conditions into a common prefix and cipher-selector suffixes.
/// Returns Err if the conditions differ on non-cipher predicates.
fn decompose_path_conditions(
    pc1: &Expr,
    pc2: &Expr,
    cipher_vars: &HashSet<VarId>,
) -> Result<(Expr, Expr, Expr), MergeFailure> {
    let conjuncts1 = pc1.flatten_and();   // Vec<Expr>
    let conjuncts2 = pc2.flatten_and();

    let mut common: Vec<Expr> = Vec::new();
    let mut only1: Vec<Expr> = Vec::new();
    let mut only2: Vec<Expr> = Vec::new();

    // Partition conjuncts: shared vs. cipher-selector-dependent
    let set2: HashSet<&Expr> = conjuncts2.iter().collect();
    for c in &conjuncts1 {
        if set2.contains(c) {
            common.push(c.clone());
        } else if c.references_any(cipher_vars) {
            only1.push(c.clone());
        } else {
            return Err(MergeFailure::IncompatiblePaths);
        }
    }
    let set1: HashSet<&Expr> = conjuncts1.iter().collect();
    for c in &conjuncts2 {
        if !set1.contains(c) {
            if c.references_any(cipher_vars) {
                only2.push(c.clone());
            } else {
                return Err(MergeFailure::IncompatiblePaths);
            }
        }
    }

    Ok((
        Expr::and_all(common),
        Expr::and_all(only1),
        Expr::and_all(only2),
    ))
}
```

### 2.5 Why O(n) Instead of O(2^n)

**Without PROTOMERGE:** Consider a negotiation with `n` cipher suites and `m` handshake phases. At each phase, the code branches on which cipher was selected, creating up to `n` forks. Over `m` phases, each fork may create further sub-forks. In the worst case (e.g., different key-exchange parameters per cipher leading to cipher-dependent branching at every subsequent phase), KLEE explores up to O(n^m) or O(2^n) states if there are log₂(n) binary decision points per cipher.

Concretely: OpenSSL supports ~150 cipher suites. Naïve symbolic execution with 4 handshake decision points produces 150⁴ ≈ 500 million states. Even with veritesting (Avgerinos et al., ICSE 2014), which merges within straight-line code segments, cipher-dependent branches at phase boundaries prevent merging, leaving O(n^m) intact.

**With PROTOMERGE:** After the first phase, `n` states are produced (one per cipher suite). PROTOMERGE merges all `n` into a single state with an ITE selector over cipher IDs. At the second phase, this single state progresses, potentially forking on cipher-dependent logic, but the branches are immediately re-merged because they satisfy all four mergeability properties — they are at the same phase, with the same offered set, differing only in the cipher-selection predicate. The result: **one symbolic state per phase**, carrying an ITE tree of depth proportional to the number of merges.

| Phase | Without Merge | With PROTOMERGE |
|-------|:---:|:---:|
| Phase 1 (ClientHello) | 1 | 1 |
| Phase 2 (ServerHello selection) | n | n → merge → 1 |
| Phase 3 (KeyExchange) | n² | 1 (re-forks to n, re-merges to 1) |
| Phase 4 (Finished) | n^m | 1 |
| **Total states explored** | **O(n^m)** | **O(n · m)** |

The total is O(n · m) because at each of m phases we explore up to n forks before merging. This is the theoretical ceiling for ideal negotiation code. For real code with cipher-dependent callbacks that break P4, empirical measurement shows 10–100× path reduction rather than the theoretical n^(m-1)× maximum.

### 2.6 Integration with KLEE's Searcher Interface

KLEE selects which state to explore next via a pluggable `Searcher` interface. PROTOMERGE integrates as a `MergingSearcher` wrapper:

```
struct ProtoMergeSearcher {
    inner: Box<dyn Searcher>,               // e.g., RandomPathSearcher
    merge_pool: BTreeMap<ProgramCounter, Vec<SymbolicState>>,
    protocol_ctx: ProtocolContext,
    stats: MergeStatistics,
}

impl Searcher for ProtoMergeSearcher {
    /// Called by KLEE when a state reaches a merge point (basic block entry)
    fn notify_state_arrival(&mut self, state: SymbolicState) {
        let pc = state.pc;
        let pool = self.merge_pool.entry(pc).or_default();

        // Attempt to merge with existing states at this PC
        let mut merged = false;
        for existing in pool.iter_mut() {
            match protomerge(existing, &state, &self.protocol_ctx) {
                Ok(merged_state) => {
                    *existing = merged_state;
                    self.stats.successful_merges += 1;
                    merged = true;
                    break;
                }
                Err(_) => { continue; }
            }
        }

        if !merged {
            pool.push(state);
            self.stats.merge_failures += 1;
        }
    }

    /// Called by KLEE to pick the next state to explore
    fn select_state(&mut self) -> Option<SymbolicState> {
        // Drain merge pools that have been waiting long enough,
        // then delegate to inner searcher
        for (_pc, pool) in self.merge_pool.iter_mut() {
            if pool.len() > 0 && pool[0].wait_cycles >= MAX_MERGE_WAIT {
                return Some(pool.remove(0));
            }
        }
        self.inner.select_state()
    }
}
```

The `MAX_MERGE_WAIT` parameter (default: 50 explorer steps) controls how long a state waits at a merge point for a partner. This trades off merging opportunity against exploration latency.

### 2.7 Complexity

- **Mergeability check (per pair):** O(|store|) to compare stores, O(|pc_conjuncts|) to decompose path conditions. For negotiation slices: |store| ≈ 500–2000 variables, |conjuncts| ≈ 20–100 → microseconds per check.
- **Merge construction:** O(|store|) to build ITE expressions → microseconds.
- **Merge pool management:** O(pool_size) per arrival. Pool size is bounded by the number of cipher suites at each PC → O(n) per arrival, O(n²) worst-case across all arrivals at a single PC.
- **Total additional overhead:** O(n² · m · |store|) across the entire exploration, dominated by the n×n pairwise comparisons at each of m phase transition points. This is negligible compared to the O(n^m) states eliminated.

---

## 3. Algorithm 3 — SMEXTRACT: Bisimulation-Quotient State Machine Extractor

### 3.1 Purpose

SMEXTRACT converts the set of symbolic execution traces produced by KLEE+PROTOMERGE into a finite-state machine (FSM) suitable for Dolev-Yao analysis. States in this FSM are *bisimulation equivalence classes* of symbolic states, ensuring that the FSM is as compact as possible while preserving all observable negotiation behaviors.

### 3.2 Interface

```
fn smextract(
    traces: &[SymbolicTrace],       // KLEE output: sequence of (state, transition_label)
    obs: &dyn Fn(&SymbolicState) -> Observable,  // observation function
) -> ProtocolFSM
```

**Preconditions:**
- Each `SymbolicTrace` is a sequence of `(SymbolicState, TransitionLabel)` pairs produced by KLEE, where `TransitionLabel` encodes the protocol message or internal action taken.
- The observation function `obs` maps symbolic states to their externally visible behavior (sent messages, selected cipher suite, protocol version).

**Postconditions:**
- The returned `ProtocolFSM` has the property that its trace set equals the trace set of the original symbolic execution modulo bisimulation: `Traces(FSM) ≈_π Traces(KLEE)`.
- The FSM is *minimal* up to bisimulation — no two states in the FSM are bisimilar.

### 3.3 Approach — Protocol-Bisimulation via Paige-Tarjan

**Step 1: Build the raw labeled transition system (LTS).**

Collect all unique symbolic states across all traces. Each state becomes an LTS node. Transitions are labeled with protocol messages (e.g., `ClientHello(ciphers, version)`, `ServerHello(cipher, version)`) or internal actions (τ).

**Step 2: Define protocol-bisimulation equivalence.**

Two LTS states `s`, `s'` are *protocol-bisimilar* (`s ≈_π s'`) iff:
- `obs(s) = obs(s')` — same observable (handshake phase + offered/selected cipher set structure).
- For every transition `s --a--> t`, there exists `s' --a--> t'` with `t ≈_π t'`.
- For every transition `s' --a--> t'`, there exists `s --a--> t` with `t ≈_π t'`.

This is standard bisimulation restricted to protocol observables.

**Step 3: Compute the coarsest bisimulation partition via Paige-Tarjan.**

```
fn paige_tarjan(
    lts: &LTS,
    initial_partition: Vec<HashSet<StateId>>,  // initially: group by obs(s)
) -> Vec<HashSet<StateId>> {
    // initial_partition groups states by observable
    let mut partition: Vec<HashSet<StateId>> = initial_partition;
    let mut worklist: VecDeque<(BlockIdx, Label)> = VecDeque::new();

    // Initialize worklist with all (block, label) pairs
    for (idx, block) in partition.iter().enumerate() {
        for label in lts.labels() {
            worklist.push_back((idx, label));
        }
    }

    while let Some((splitter_idx, label)) = worklist.pop_front() {
        let splitter = &partition[splitter_idx];

        // Compute pre-image: states that can reach splitter via label
        let pre_image: HashSet<StateId> = lts.predecessors_by_label(splitter, label);

        let mut new_partition: Vec<HashSet<StateId>> = Vec::new();
        for block in &partition {
            let intersect: HashSet<StateId> = block.intersection(&pre_image).cloned().collect();
            let diff: HashSet<StateId> = block.difference(&pre_image).cloned().collect();

            if !intersect.is_empty() && !diff.is_empty() {
                // Split this block
                let smaller = if intersect.len() <= diff.len() {
                    &intersect
                } else {
                    &diff
                };
                // Add smaller half to worklist for all labels
                let new_idx = new_partition.len(); // placeholder
                for l in lts.labels() {
                    worklist.push_back((new_idx, l));
                }
                new_partition.push(intersect);
                new_partition.push(diff);
            } else {
                new_partition.push(block.clone());
            }
        }
        partition = new_partition;
    }

    partition
}

fn smextract(
    traces: &[SymbolicTrace],
    obs: &dyn Fn(&SymbolicState) -> Observable,
) -> ProtocolFSM {
    // Step 1: Build raw LTS
    let mut states: BTreeMap<StateId, SymbolicState> = BTreeMap::new();
    let mut transitions: Vec<(StateId, Label, StateId)> = Vec::new();
    let mut state_counter: u64 = 0;

    for trace in traces {
        let mut prev_id: Option<StateId> = None;
        for (sym_state, label) in trace.steps() {
            let id = intern_state(&mut states, &mut state_counter, sym_state);
            if let Some(pid) = prev_id {
                transitions.push((pid, label.clone(), id));
            }
            prev_id = Some(id);
        }
    }

    let lts = LTS::new(states.keys().cloned().collect(), transitions);

    // Step 2: Initial partition by observation
    let mut obs_groups: BTreeMap<Observable, HashSet<StateId>> = BTreeMap::new();
    for (id, state) in &states {
        obs_groups.entry(obs(state)).or_default().insert(*id);
    }
    let initial_partition: Vec<HashSet<StateId>> = obs_groups.into_values().collect();

    // Step 3: Paige-Tarjan refinement
    let final_partition = paige_tarjan(&lts, initial_partition);

    // Step 4: Build quotient FSM
    let mut fsm = ProtocolFSM::new();
    let block_map: BTreeMap<StateId, BlockIdx> = build_block_map(&final_partition);

    for (src, label, dst) in &lts.transitions {
        let fsm_src = block_map[src];
        let fsm_dst = block_map[dst];
        fsm.add_transition(fsm_src, label.clone(), fsm_dst);
    }

    // Annotate FSM states with representative observables
    for (idx, block) in final_partition.iter().enumerate() {
        let representative = block.iter().next().unwrap();
        fsm.set_observation(idx, obs(&states[representative]));
    }

    fsm
}
```

### 3.4 Complexity

- **LTS construction:** O(T) where T = total steps across all traces. After PROTOMERGE, T = O(n · m · k) for n cipher suites, m phases, k depth.
- **Paige-Tarjan partition refinement:** O(T · log(|S|)) where |S| is the number of LTS states. This is the standard Paige-Tarjan bound.
- **Practical:** For OpenSSL with ~150 cipher suites, m=6 phases, k=20 depth: |S| ≈ 2000–5000, T ≈ 10K–50K transitions → completes in seconds.

---

## 4. Algorithm 4 — DYENCODE: Dolev-Yao + SMT Constraint Encoder

### 4.1 Purpose

DYENCODE translates the extracted protocol FSM, a bounded Dolev-Yao adversary model, and the downgrade-freedom property into an SMT formula. A satisfying assignment represents a concrete downgrade attack; unsatisfiability certifies absence within the given bounds.

### 4.2 Interface

```
fn dyencode(
    fsm: &ProtocolFSM,               // from SMEXTRACT
    grammar: &MessageGrammar,          // TLS/SSH message definitions
    property: &DowngradeProperty,      // what constitutes a downgrade
    adversary_budget: usize,           // max number of adversary actions (b)
    depth_bound: usize,                // max protocol steps (k)
) -> SmtFormula                        // in BV+Arrays+UF+LIA
```

**Preconditions:**
- `fsm` is a valid ProtocolFSM produced by SMEXTRACT.
- `grammar` defines the wire-format structure of protocol messages.
- `property` specifies the downgrade-freedom condition (e.g., "selected cipher ∈ strongest_common_set(client_offered, server_supported)").
- `adversary_budget ≥ 1`, `depth_bound ≥ 1`.

**Postconditions:**
- The returned formula is equisatisfiable with the existence of a downgrade attack: `SAT(formula) ⟺ ∃ attack trace of ≤ k steps with ≤ b adversary actions that violates the downgrade property`.

### 4.3 Theory Selection — BV+Arrays+UF+LIA

| Component | SMT Theory | Justification |
|-----------|-----------|--------------|
| Cipher-suite IDs | `BV[16]` | IANA cipher IDs are 16-bit values; BV enables direct bit-manipulation modeling |
| Protocol versions | `BV[16]` | TLS versions encoded as `0x0301` (TLS 1.0) through `0x0304` (TLS 1.3) |
| Message buffers | `Array[BV[32], BV[8]]` | Byte arrays indexed by position; models TLS record and SSH packet contents |
| Crypto operations | `UF` | `enc(k, m)`, `dec(k, c)`, `mac(k, m)` modeled as uninterpreted; only algebraic properties axiomatized (e.g., `dec(k, enc(k, m)) = m`) |
| Ordering / counting | `LIA` | Adversary action budget, step ordering, message sequence numbers |

### 4.4 Encoding Strategy

**State encoding.** Each protocol step `t ∈ [0, k]` has:
- `state_client[t]: BV[8]` — client FSM state
- `state_server[t]: BV[8]` — server FSM state
- `msg[t]: Array[BV[32], BV[8]]` — message on the wire at step t
- `msg_len[t]: BV[32]` — message length
- `sender[t]: BV[2]` — 0=client, 1=server, 2=adversary

**Transition encoding.** For each FSM transition `(q, label, q')`:
```
∀t ∈ [0, k-1]:
  (state[t] = encode(q) ∧ msg_matches(msg[t], label))
    → state[t+1] = encode(q')
```

**Adversary encoding.** The adversary has `b` action slots. Each slot `j ∈ [0, b-1]` encodes:
```
action_type[j]: BV[2]        // 0=intercept, 1=modify, 2=inject, 3=drop
action_step[j]:  BV[8]       // which protocol step this action targets
action_payload[j]: Array[BV[32], BV[8]]  // modified/injected content
```

Constraints on adversary actions:
```
// Each action targets a valid step
∀j: 0 ≤ action_step[j] < k

// Action steps are ordered (no time travel)
∀j < b-1: action_step[j] ≤ action_step[j+1]

// At most b actions total
// (implicit in having exactly b slots; unused slots set to type=intercept
//  with identity payload, which is a no-op)
```

**Knowledge accumulation.** The adversary's knowledge set is modeled as an array of known terms:

```
knowledge: Array[BV[16], Term]   // indexed by knowledge slot
know_count[t]: BV[16]            // how many terms known at step t

// Initial knowledge: public parameters, own keys
know_count[0] = |initial_knowledge|
∀i < |initial_knowledge|: knowledge[i] = initial_terms[i]

// After intercepting message at step t:
(action_type[j] = INTERCEPT ∧ action_step[j] = t)
  → (knowledge[know_count[t]] = msg[t]
     ∧ know_count[t+1] = know_count[t] + 1)

// Derivation: adversary can apply crypto operations to known terms
∀i,j < know_count[t]:
  knowledge[know_count[t] + derived_idx] = enc(knowledge[i], knowledge[j])
  // ... similarly for dec, mac, pair, fst, snd
```

In practice, the derivation rules are bounded to depth `d` (default: 3) to keep the formula finite.

**Downgrade property negation.** The property "no downgrade occurs" is:
```
selected_cipher ∈ strongest_common(client_offered, server_supported)
∧ selected_version = max_common(client_versions, server_versions)
```

We negate this to search for violations:
```
¬(selected_cipher ∈ strongest_common(...))
∨ ¬(selected_version = max_common(...))
```

### 4.5 Pseudocode

```
fn dyencode(
    fsm: &ProtocolFSM,
    grammar: &MessageGrammar,
    property: &DowngradeProperty,
    b: usize,
    k: usize,
) -> SmtFormula {
    let mut solver = SmtContext::new(Logic::QF_AUFBVLIA);

    // ---- DECLARE VARIABLES ----

    // Per-step state variables
    let state_c: Vec<SmtBV> = (0..=k)
        .map(|t| solver.declare_bv(&format!("state_c_{t}"), 8))
        .collect();
    let state_s: Vec<SmtBV> = (0..=k)
        .map(|t| solver.declare_bv(&format!("state_s_{t}"), 8))
        .collect();
    let msg: Vec<SmtArray> = (0..=k)
        .map(|t| solver.declare_array(&format!("msg_{t}"), SortBV(32), SortBV(8)))
        .collect();
    let msg_len: Vec<SmtBV> = (0..=k)
        .map(|t| solver.declare_bv(&format!("msg_len_{t}"), 32))
        .collect();
    let sender: Vec<SmtBV> = (0..=k)
        .map(|t| solver.declare_bv(&format!("sender_{t}"), 2))
        .collect();

    // Adversary action slots
    let act_type: Vec<SmtBV> = (0..b)
        .map(|j| solver.declare_bv(&format!("act_type_{j}"), 2))
        .collect();
    let act_step: Vec<SmtBV> = (0..b)
        .map(|j| solver.declare_bv(&format!("act_step_{j}"), 8))
        .collect();
    let act_payload: Vec<SmtArray> = (0..b)
        .map(|j| solver.declare_array(&format!("act_payload_{j}"), SortBV(32), SortBV(8)))
        .collect();

    // Knowledge accumulation
    let knowledge = solver.declare_array("knowledge", SortBV(16), SortUF("Term"));
    let know_count: Vec<SmtBV> = (0..=k)
        .map(|t| solver.declare_bv(&format!("know_count_{t}"), 16))
        .collect();

    // ---- INITIAL STATE ----
    solver.assert(state_c[0].eq(fsm.initial_state_client()));
    solver.assert(state_s[0].eq(fsm.initial_state_server()));
    encode_initial_knowledge(&mut solver, &knowledge, &know_count[0]);

    // ---- TRANSITION CONSTRAINTS ----
    for t in 0..k {
        // Client transitions
        let client_trans = encode_transitions(
            &solver, fsm, Role::Client, &state_c[t], &state_c[t+1],
            &msg[t], &msg_len[t], &sender[t], grammar,
        );
        // Server transitions
        let server_trans = encode_transitions(
            &solver, fsm, Role::Server, &state_s[t], &state_s[t+1],
            &msg[t], &msg_len[t], &sender[t], grammar,
        );
        solver.assert(client_trans.or(server_trans));
    }

    // ---- ADVERSARY CONSTRAINTS ----
    for j in 0..b {
        // Ordered actions
        if j + 1 < b {
            solver.assert(act_step[j].bvule(&act_step[j + 1]));
        }
        solver.assert(act_step[j].bvult(&solver.bv_const(k as u64, 8)));

        // Adversary can only send what it can derive from knowledge
        let t_var = &act_step[j];
        solver.assert(
            act_type[j].eq(&solver.bv_const(MODIFY, 2)).implies(
                &is_derivable(&solver, &act_payload[j], &knowledge, &know_count, t_var)
            )
        );
        solver.assert(
            act_type[j].eq(&solver.bv_const(INJECT, 2)).implies(
                &is_derivable(&solver, &act_payload[j], &knowledge, &know_count, t_var)
            )
        );

        // Knowledge update: intercepted messages are learned
        encode_knowledge_update(
            &mut solver, &knowledge, &know_count,
            &act_type[j], &act_step[j], &msg,
        );
    }

    // ---- PROPERTY NEGATION (SAT = attack found) ----
    let final_cipher_c = extract_selected_cipher(&solver, &state_c[k], fsm);
    let final_cipher_s = extract_selected_cipher(&solver, &state_s[k], fsm);
    let strongest = encode_strongest_common(
        &solver, grammar, &msg[0], // ClientHello contains offered ciphers
    );

    // Downgrade: the selected cipher is not in the strongest common set,
    // OR client and server disagree on selected cipher
    let downgrade = final_cipher_c.ne(&strongest)
        .or(&final_cipher_c.ne(&final_cipher_s));

    // Handshake must complete (both reach Finished state)
    let completed = state_c[k].eq(&fsm.finished_state_client())
        .and(&state_s[k].eq(&fsm.finished_state_server()));

    solver.assert(completed.and(&downgrade));

    solver.to_formula()
}
```

### 4.6 Complexity

- **Formula size:** O(k · |FSM_transitions| + b · k · derivation_depth) clauses. For k=20, |transitions|≈200, b=5, derivation_depth=3: ~50K clauses.
- **Theory complexity:** BV+Arrays is in NEXPTIME in general, but the bounded structure (small bitvectors, bounded array indices) makes instances tractable for Z3's bit-blasting + array decision procedures.
- **Practical:** Z3 solves typical instances (FREAK, Logjam complexity) in 5–30 minutes. Worst case (complex multi-round SSH with extensions): up to 4 hours.

---

## 5. Algorithm 5 — CONCRETIZE: CEGAR Concretization Loop

### 5.1 Purpose

Given a satisfying SMT assignment (an abstract attack trace), CONCRETIZE maps it to a concrete, byte-level attack script and validates it against a live TLS/SSH implementation via TLS-Attacker replay. If replay fails, CONCRETIZE extracts a refinement predicate and re-solves.

### 5.2 Interface

```
fn concretize(
    model: &SmtModel,                    // satisfying assignment from Z3
    grammar: &MessageGrammar,             // wire-format definitions
    fsm: &ProtocolFSM,                   // for state interpretation
    target: &TargetConfig,                // library + version + endpoint
    smt_formula: &mut SmtFormula,         // mutable: refinement predicates added
    max_iterations: usize,                // CEGAR loop bound (default: 10)
) -> Result<ConcreteAttack, ConcretizationFailure>
```

**Preconditions:**
- `model` satisfies `smt_formula`.
- `target` specifies a running TLS/SSH instance to replay against.

**Postconditions (on Ok):**
- The returned `ConcreteAttack` is a byte-level script that, when replayed, causes the target to complete a handshake with a downgraded cipher suite or protocol version.

**Postconditions (on Err):**
- After `max_iterations` CEGAR rounds, concretization has not succeeded. The abstract trace may be spurious (an artifact of over-approximation in the UF encoding of crypto).

### 5.3 CEGAR Loop

```
fn concretize(
    model: &SmtModel,
    grammar: &MessageGrammar,
    fsm: &ProtocolFSM,
    target: &TargetConfig,
    smt_formula: &mut SmtFormula,
    max_iterations: usize,
) -> Result<ConcreteAttack, ConcretizationFailure> {
    let mut current_model = model.clone();

    for iteration in 0..max_iterations {
        // ---- STEP 1: Extract abstract attack trace ----
        let abstract_trace = extract_trace(&current_model, fsm);
        //   abstract_trace: Vec<AbstractStep> where each step has:
        //     role (client/server/adversary), message (symbolic), state transition

        // ---- STEP 2: Concretize symbolic values to bytes ----
        let concrete_trace = match concretize_trace(&abstract_trace, grammar) {
            Ok(ct) => ct,
            Err(e) => {
                // Concretization failed at the symbolic→byte level
                // (e.g., UF term has no valid byte representation)
                let refinement = extract_refinement_from_concretization_failure(&e);
                smt_formula.add_constraint(refinement);
                match smt_formula.solve() {
                    SolveResult::Sat(new_model) => {
                        current_model = new_model;
                        continue;
                    }
                    SolveResult::Unsat => {
                        return Err(ConcretizationFailure::Spurious {
                            iterations: iteration + 1,
                            reason: "Abstract trace is spurious: no concrete instantiation exists",
                        });
                    }
                    SolveResult::Timeout => {
                        return Err(ConcretizationFailure::SolverTimeout);
                    }
                }
            }
        };

        // ---- STEP 3: Replay against live instance ----
        let replay_result = tls_attacker_replay(&concrete_trace, target);

        match replay_result {
            ReplayResult::Success { final_cipher, final_version } => {
                // ---- STEP 4a: Validate downgrade ----
                if is_downgrade(final_cipher, final_version, &abstract_trace) {
                    return Ok(ConcreteAttack {
                        trace: concrete_trace,
                        selected_cipher: final_cipher,
                        selected_version: final_version,
                        cegar_iterations: iteration + 1,
                    });
                } else {
                    // Handshake succeeded but no downgrade — model was misleading
                    let refinement = Expr::not(encode_trace_as_constraint(&abstract_trace));
                    smt_formula.add_constraint(refinement);
                    current_model = smt_formula.solve().expect_sat()?;
                    continue;
                }
            }
            ReplayResult::Failure { step, reason } => {
                // ---- STEP 4b: Analyze failure, extract refinement ----
                let refinement = analyze_replay_failure(
                    &abstract_trace, step, &reason, grammar, fsm,
                );
                // refinement blocks the specific combination that failed:
                //   e.g., "if adversary modifies ServerHello at step 3,
                //          the cipher field must be in the client's offered set"
                smt_formula.add_constraint(refinement);

                match smt_formula.solve() {
                    SolveResult::Sat(new_model) => {
                        current_model = new_model;
                        continue;
                    }
                    SolveResult::Unsat => {
                        return Err(ConcretizationFailure::Spurious {
                            iterations: iteration + 1,
                            reason: "All concrete instantiations eliminated by refinement",
                        });
                    }
                    SolveResult::Timeout => {
                        return Err(ConcretizationFailure::SolverTimeout);
                    }
                }
            }
        }
    }

    Err(ConcretizationFailure::MaxIterationsReached { max_iterations })
}

/// Map abstract trace steps to concrete byte-level TLS/SSH records
fn concretize_trace(
    abstract_trace: &[AbstractStep],
    grammar: &MessageGrammar,
) -> Result<Vec<ConcreteStep>, ConcretizationError> {
    let mut concrete_steps: Vec<ConcreteStep> = Vec::new();

    for step in abstract_trace {
        let bytes = match step.role {
            Role::Client | Role::Server => {
                // Honest party message: serialize according to grammar
                grammar.serialize(&step.message)?
            }
            Role::Adversary => {
                match step.action {
                    AdvAction::Intercept => continue,  // no wire bytes
                    AdvAction::Drop => continue,
                    AdvAction::Modify(ref payload) => {
                        // Deserialize original, apply modifications, re-serialize
                        let mut msg = grammar.deserialize(&step.original_message)?;
                        apply_modifications(&mut msg, payload)?;
                        grammar.serialize(&msg)?
                    }
                    AdvAction::Inject(ref payload) => {
                        grammar.serialize(payload)?
                    }
                }
            }
        };

        concrete_steps.push(ConcreteStep {
            direction: step.direction,
            bytes,
            delay_ms: step.timing_hint.unwrap_or(0),
        });
    }

    Ok(concrete_steps)
}
```

### 5.4 Termination Argument

The CEGAR loop terminates because:

1. **Finite abstract domain.** The SMT formula has finitely many satisfying assignments (the state space is bounded by `k` steps and `b` adversary actions over finite bitvector domains).

2. **Progress guarantee.** Each iteration either succeeds (returns `Ok`) or adds a refinement predicate that blocks at least one satisfying assignment (the current model). Since the constraint `¬(encode_trace(current_trace))` eliminates the current assignment, and the domain is finite, the loop terminates in at most `|SAT_assignments|` iterations.

3. **Practical bound.** The `max_iterations` parameter (default: 10) provides an absolute cap. Empirically, concretization succeeds in 1–3 iterations for all tested CVEs: iteration 1 handles direct attacks; iterations 2–3 refine timing or record-layer framing details.

### 5.5 Complexity

- **Per iteration:** Dominated by Z3 re-solve (minutes) + TLS-Attacker replay (seconds).
- **Total:** O(max_iterations × solve_time). With max_iterations=10, worst case ~30 minutes.
- **Practical:** 1–3 iterations typical, completing in under 15 minutes.

---

## 6. Complexity Analysis Summary

| Algorithm | Time Complexity | Space Complexity | Practical Bottleneck | OpenSSL Estimate |
|-----------|----------------|-----------------|---------------------|-----------------|
| **PROTOSLICE** | O(V² · E) points-to + O(\|V\| + \|E\|) slice + O(\|V_slice\| · h) taint | O(V + E) for ICFG + O(V²) for points-to | Points-to analysis on large modules | ~10 min, 4 GB |
| **PROTOMERGE** | O(n² · m · \|store\|) merge overhead | O(n · \|store\|) per merge pool | ITE expression growth in merged stores | ~2 hr total SE with merge (vs. days without) |
| **SMEXTRACT** | O(T · log \|S\|) Paige-Tarjan | O(\|S\|² + T) for partition + LTS | LTS construction from traces | ~30 sec, <1 GB |
| **DYENCODE** | O(k · \|δ\| + b · k · d) formula construction | O(k · \|δ\| + b · k) formula size | Z3 solving, not encoding | Encode: ~5 min; Solve: 5 min – 4 hr |
| **CONCRETIZE** | O(iter × solve_time) | O(formula_size) growing with refinements | Z3 re-solve per iteration | 1–3 iter, ~15 min total |

Where: V = IR variables (~50K for OpenSSL), E = CFG edges (~200K), n = cipher suites (~150), m = phases (~6), k = depth bound (20), |δ| = FSM transitions (~200), b = adversary budget (5), d = derivation depth (3), h = taint lattice height (4), T = total trace steps, |S| = LTS states, iter = CEGAR iterations.

**End-to-end wall-clock estimate for OpenSSL:** 4–8 hours on an 8-core laptop (Intel i7 / Apple M2, 32 GB RAM).

---

## 7. Implementation Mapping

The table below maps each algorithm to its implementing module in the ~50K novel LoC + ~40K integration architecture.

| Algorithm | Crate / Module | Language | Est. LoC | Key Dependencies |
|-----------|---------------|----------|:--------:|-----------------|
| **PROTOSLICE** | `negsynth-slicer/` | Rust | 10K–13K | `llvm-ir` (Rust LLVM IR parser), `llvm-sys` (FFI to LLVM points-to), custom `taint` module |
| **PROTOMERGE** | `negsynth-merge/` | Rust + C++ | 6K–8K | KLEE C++ headers (Searcher interface), `cxx` bridge for Rust↔C++ FFI, `negsynth-protocol/` for phase/cipher definitions |
| **KLEE Integration** | `negsynth-klee/` | C++ + Rust | 6K–8K | KLEE source tree (patched Searcher, ExecutionState), `negsynth-merge` via FFI |
| **SMEXTRACT** | `negsynth-extract/` | Rust | 7K–9K | `petgraph` (graph algorithms), custom `bisim` module implementing Paige-Tarjan |
| **DYENCODE** | `negsynth-encode/` | Rust | 8K–11K | `z3` (Rust bindings via `z3-sys`), `negsynth-protocol/` for message grammar, custom `dy_model` module |
| **CONCRETIZE** | `negsynth-concrete/` | Rust + Python | 5K–7K | `z3` for incremental solving, Python subprocess for TLS-Attacker integration, `negsynth-protocol/` for serialization |
| **Protocol Modules** | `negsynth-protocol/` | Rust | 18K–22K | TLS 1.0–1.3 grammars (~12K), SSH v2 grammar (~8K), shared types |
| **CLI + Reporting** | `negsynth-cli/` | Rust | 5K–7K | `clap`, `serde_json` (SARIF output), `negsynth-*` crates |

### Build Graph

```
negsynth-protocol  ──────────────────────────────────────┐
       │                                                  │
       ├──► negsynth-slicer (PROTOSLICE)                  │
       │         │                                        │
       │         ▼                                        │
       ├──► negsynth-merge (PROTOMERGE) ◄── negsynth-klee │
       │         │                         (C++ FFI)      │
       │         ▼                                        │
       ├──► negsynth-extract (SMEXTRACT)                  │
       │         │                                        │
       │         ▼                                        │
       ├──► negsynth-encode (DYENCODE)                    │
       │         │                                        │
       │         ▼                                        │
       └──► negsynth-concrete (CONCRETIZE) ───► negsynth-cli
```

### Data Flow Between Modules

| Stage Boundary | Format | Serialization |
|---------------|--------|--------------|
| PROTOSLICE → KLEE | LLVM IR bitcode (`.bc` file) | Standard LLVM bitcode |
| KLEE+PROTOMERGE → SMEXTRACT | `Vec<SymbolicTrace>` | Custom binary format via `bincode` (Rust serde) |
| SMEXTRACT → DYENCODE | `ProtocolFSM` | In-memory struct; optionally serialized as JSON for debugging |
| DYENCODE → Z3 | SMT-LIB2 string or Z3 API calls | Z3 Rust bindings (in-process), SMT-LIB2 file for reproducibility |
| Z3 → CONCRETIZE | `SmtModel` | Z3 model API |
| CONCRETIZE → TLS-Attacker | Attack script (XML) | TLS-Attacker's XML workflow format via Python bridge |
| CONCRETIZE → Output | `ConcreteAttack` / SARIF | JSON (SARIF for CI integration), human-readable trace |

---

## Appendix A: Key Type Definitions

```rust
/// Handshake phases, ordered by protocol progression
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum HandshakePhase {
    Initial = 0,
    ClientHello = 1,
    ServerHello = 2,
    Certificate = 3,
    KeyExchange = 4,
    ChangeCipherSpec = 5,
    Finished = 6,
}

/// Taint labels for protocol-aware slicing
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum TaintLabel {
    Untainted,
    CipherTaint,
    VersionTaint,
    ExtensionTaint,
    MultiTaint,   // joins of distinct taints
}

impl TaintLabel {
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (TaintLabel::Untainted, x) | (x, TaintLabel::Untainted) => x,
            (a, b) if a == b => a,
            _ => TaintLabel::MultiTaint,
        }
    }
}

/// Symbolic state as maintained by KLEE + PROTOMERGE
struct SymbolicState {
    pc: ProgramCounter,
    store: SymbolicStore,               // BTreeMap<VarId, Expr>
    path_condition: Expr,
    merge_count: u32,
    phase: HandshakePhase,
}

/// Protocol FSM state
struct FsmState {
    id: u32,
    observation: Observable,            // (phase, offered_ciphers, selected_cipher)
    is_accepting: bool,
}

/// Protocol FSM transition
struct FsmTransition {
    src: u32,
    dst: u32,
    label: MessageLabel,                // e.g., ClientHello{ciphers: [...], version: ...}
    guard: Option<Expr>,                // path condition fragment
}

/// Concrete attack output
struct ConcreteAttack {
    trace: Vec<ConcreteStep>,
    selected_cipher: u16,               // IANA cipher ID that was forced
    selected_version: u16,              // protocol version that was forced
    cegar_iterations: usize,
    description: String,                // human-readable attack narrative
}

/// Adversary action types
#[derive(Clone, Copy)]
enum AdvActionType {
    Intercept = 0,   // observe without modifying
    Modify = 1,      // change message content
    Inject = 2,      // send a new message
    Drop = 3,        // suppress message delivery
}
```

---

## Appendix B: Relationship to Theorems T1–T5

| Algorithm | Load-Bearing Theorem | What the Theorem Guarantees |
|-----------|---------------------|---------------------------|
| PROTOSLICE | **T1 (Extraction Soundness)** — dependency | Every negotiation-relevant execution path is preserved in the slice |
| PROTOMERGE | **T3 (Merge Correctness)** — crown | Merged states are protocol-bisimilar to unmerged originals; O(n·m) path bound |
| SMEXTRACT | **T1 (Extraction Soundness)** — core | FSM traces correspond to feasible source-level execution paths |
| DYENCODE | **T5 (SMT Encoding Correctness)** | Formula is equisatisfiable with the DY adversary + FSM composition |
| CONCRETIZE | **T2 (Concretizability)** | SAT assignments concretize to executable byte-level attacks with rate ≥ 1−ε |
| End-to-end | **T4 (Bounded Completeness)** | Composes T1+T3+T5: within bounds (k,n), every downgrade attack is found or certified absent |
