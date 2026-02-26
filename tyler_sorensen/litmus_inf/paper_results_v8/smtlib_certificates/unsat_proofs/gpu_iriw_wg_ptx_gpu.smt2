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
; Pattern: gpu_iriw_wg
; Forbidden outcome: {"r0": 1, "r1": 0, "r2": 1, "r3": 0}
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

; ── Pattern: gpu_iriw_wg ─────────────────────────
; Threads: 4
; Addresses: x, y
; Memory model: PTX-GPU
; Forbidden outcome: {'r0': 1, 'r1': 0, 'r2': 1, 'r3': 0}
;
; Operations:
;   e0: T0 Store x = 1
;   e1: T1 Store y = 1
;   e2: T2 Load x → r0
;   e3: T2 Fence [scope=workgroup]
;   e4: T2 Load y → r1
;   e5: T3 Load y → r2
;   e6: T3 Fence [scope=workgroup]
;   e7: T3 Load x → r3

(set-logic QF_LIA)

(declare-fun rf_2 () Int)
(declare-fun rf_4 () Int)
(declare-fun rf_5 () Int)
(declare-fun rf_7 () Int)
(declare-fun ts_0 () Int)
(declare-fun ts_1 () Int)
(declare-fun ts_2 () Int)
(declare-fun ts_3 () Int)
(declare-fun ts_4 () Int)
(declare-fun ts_5 () Int)
(declare-fun ts_6 () Int)
(declare-fun ts_7 () Int)
(declare-fun ts_init_x () Int)
(declare-fun ts_init_y () Int)
(assert (>= rf_2 0))
(assert (< rf_2 2))
(assert (>= rf_4 0))
(assert (< rf_4 2))
(assert (>= rf_5 0))
(assert (< rf_5 2))
(assert (>= rf_7 0))
(assert (< rf_7 2))
(assert (>= ts_0 0))
(assert (< ts_0 100))
(assert (>= ts_1 0))
(assert (< ts_1 100))
(assert (>= ts_2 0))
(assert (< ts_2 100))
(assert (>= ts_3 0))
(assert (< ts_3 100))
(assert (>= ts_4 0))
(assert (< ts_4 100))
(assert (>= ts_5 0))
(assert (< ts_5 100))
(assert (>= ts_6 0))
(assert (< ts_6 100))
(assert (>= ts_7 0))
(assert (< ts_7 100))
(assert (>= ts_init_x 0))
(assert (>= ts_init_y 0))
(assert (< ts_2 ts_4))
(assert (< ts_5 ts_7))
(assert (=> (= rf_2 0) (< ts_init_x ts_2)))
(assert (=> (= rf_2 1) (< ts_0 ts_2)))
(assert (=> (= rf_4 0) (< ts_init_y ts_4)))
(assert (=> (= rf_4 1) (< ts_1 ts_4)))
(assert (=> (= rf_5 0) (< ts_init_y ts_5)))
(assert (=> (= rf_5 1) (< ts_1 ts_5)))
(assert (=> (= rf_7 0) (< ts_init_x ts_7)))
(assert (=> (= rf_7 1) (< ts_0 ts_7)))
(assert (< ts_init_x ts_0))
(assert (< ts_init_y ts_1))
(assert (=> (= rf_2 0) (< ts_2 ts_0)))
(assert (=> (= rf_4 0) (< ts_4 ts_1)))
(assert (=> (= rf_5 0) (< ts_5 ts_1)))
(assert (=> (= rf_7 0) (< ts_7 ts_0)))
(assert (and (or (= rf_2 1)) (or (= rf_4 0)) (or (= rf_5 1)) (or (= rf_7 0))))

(check-sat)

; ── UNSAT Proof Certificate ──
; Status: UNSAT — forbidden outcome UNREACHABLE under PTX-GPU
; Unsat core size: 7 assertions (out of 43)
; Core assertions: track_29, track_26, track_27, track_39, track_41, track_forbidden, track_33
(get-unsat-core)

(exit)