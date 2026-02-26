; ═══════════════════════════════════════════════════════════════════
; LITMUS∞ SMT-LIB2 Encoding — Axiomatic Memory Model Verification
; ═══════════════════════════════════════════════════════════════════
;
; Theory: Axiomatic weak memory model verification (Alglave et al. 2014)
;
; Sorts:
;   Int — event timestamps (topological ordering for acyclicity)
;   Int — read-from values (index into per-address store list)
;   Bool — coherence order relation (total order on stores per address)
;
; Relations encoded:
;   po  — program order (model-dependent preservation)
;   rf  — read-from (which store each load reads)
;   co  — coherence order (total order on stores per address)
;   fr  — from-reads (derived: load reads from s_i, s_j co-after s_i)
;
; Safety property:
;   UNSAT ⟹ forbidden outcome unreachable under model (SAFE)
;   SAT   ⟹ forbidden outcome reachable (UNSAFE, witness provided)
;
; Memory model: PTX-GPU
; Pattern: coww_fence
; Forbidden outcome: {"r0": 1}
; ═══════════════════════════════════════════════════════════════════

; ── Theory Declarations ─────────────────────────────────────────────
; Axiomatic memory model framework (Alglave, Maranget, Tautschnig 2014)
;
; An execution X = (E, po, rf, co) where:
;   E   = set of memory events (reads/writes with addresses and values)
;   po  = program order (per-thread total order on events)
;   rf  = read-from function mapping each read to the write it reads
;   co  = coherence order (per-address total order on writes)
;   fr  = from-reads: derived as rf^{-1}; co
;
; A memory model M defines which executions are consistent via
; acyclicity constraints on combinations of these relations.
;
; Models encoded:
;   TSO   — acyclic(po ∪ rf ∪ co ∪ fr), po relaxes W→R only
;   PSO   — additionally relaxes W→W
;   ARM   — relaxes all cross-address po except deps and fences
;   RISC-V — like ARM with asymmetric fence.{pred}.{succ}
;   GPU    — like ARM with scoped fences (workgroup/device/system)
;
; Encoding strategy:
;   Acyclicity is encoded via integer timestamps:
;   For each event e, ts(e) ∈ ℤ with ts(a) < ts(b) for each preserved edge a→b.
;   A cycle exists iff the conjunction of all ts constraints is unsatisfiable.
;   We conjoin the forbidden outcome constraint and check satisfiability.

; ── Pattern: coww_fence ─────────────────────────
; Threads: 3
; Addresses: x
; Memory model: PTX-GPU
; Forbidden outcome: {'r0': 1}
;
; Operations:
;   e0: T0 Store x = 1
;   e1: T0 Fence
;   e2: T1 Store x = 2
;   e3: T2 Load x → r0

(set-logic QF_LIA)

(declare-fun rf_3 () Int)
(declare-fun co_x_2_1 () Bool)
(declare-fun co_x_1_2 () Bool)
(declare-fun ts_0 () Int)
(declare-fun ts_1 () Int)
(declare-fun ts_2 () Int)
(declare-fun ts_3 () Int)
(declare-fun ts_init_x () Int)
(assert (>= rf_3 0))
(assert (< rf_3 3))
(assert (or co_x_1_2 co_x_2_1))
(assert (not (and co_x_1_2 co_x_2_1)))
(assert (>= ts_0 0))
(assert (< ts_0 50))
(assert (>= ts_1 0))
(assert (< ts_1 50))
(assert (>= ts_2 0))
(assert (< ts_2 50))
(assert (>= ts_3 0))
(assert (< ts_3 50))
(assert (>= ts_init_x 0))
(assert (=> (= rf_3 0) (< ts_init_x ts_3)))
(assert (=> (= rf_3 1) (< ts_0 ts_3)))
(assert (=> (= rf_3 2) (< ts_2 ts_3)))
(assert (< ts_init_x ts_0))
(assert (< ts_init_x ts_2))
(assert (=> co_x_1_2 (< ts_0 ts_2)))
(assert (=> co_x_2_1 (< ts_2 ts_0)))
(assert (=> (= rf_3 0) (< ts_3 ts_0)))
(assert (=> (= rf_3 0) (< ts_3 ts_2)))
(assert (=> (and (= rf_3 1) co_x_1_2) (< ts_3 ts_2)))
(assert (=> (and (= rf_3 2) co_x_2_1) (< ts_3 ts_0)))
(assert (and (or (= rf_3 1))))

(check-sat)
(get-model)

; ── SAT Witness (counterexample execution) ──
; Status: SAT — forbidden outcome IS reachable under PTX-GPU
;   rf_3 = 1
;   co_x_1_2 = True
;   co_x_2_1 = False

(exit)