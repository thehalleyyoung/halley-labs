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
; Memory model: ARM
; Pattern: gpu_rwc_dev
; Forbidden outcome: {"r0": 1, "r1": 0, "r2": 0}
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

; ── Pattern: gpu_rwc_dev ─────────────────────────
; Threads: 3
; Addresses: x, y
; Memory model: ARM
; Forbidden outcome: {'r0': 1, 'r1': 0, 'r2': 0}
;
; Operations:
;   e0: T0 Store x = 1
;   e1: T1 Load x → r0
;   e2: T1 Fence [scope=device]
;   e3: T1 Load y → r1
;   e4: T2 Store y = 1
;   e5: T2 Fence [scope=device]
;   e6: T2 Load x → r2

(set-logic QF_LIA)

(declare-fun rf_1 () Int)
(declare-fun rf_3 () Int)
(declare-fun rf_6 () Int)
(declare-fun ts_0 () Int)
(declare-fun ts_1 () Int)
(declare-fun ts_2 () Int)
(declare-fun ts_3 () Int)
(declare-fun ts_4 () Int)
(declare-fun ts_5 () Int)
(declare-fun ts_6 () Int)
(declare-fun ts_init_x () Int)
(declare-fun ts_init_y () Int)
(assert (>= rf_1 0))
(assert (< rf_1 2))
(assert (>= rf_3 0))
(assert (< rf_3 2))
(assert (>= rf_6 0))
(assert (< rf_6 2))
(assert (>= ts_0 0))
(assert (< ts_0 90))
(assert (>= ts_1 0))
(assert (< ts_1 90))
(assert (>= ts_2 0))
(assert (< ts_2 90))
(assert (>= ts_3 0))
(assert (< ts_3 90))
(assert (>= ts_4 0))
(assert (< ts_4 90))
(assert (>= ts_5 0))
(assert (< ts_5 90))
(assert (>= ts_6 0))
(assert (< ts_6 90))
(assert (>= ts_init_x 0))
(assert (>= ts_init_y 0))
(assert (< ts_1 ts_3))
(assert (< ts_4 ts_6))
(assert (=> (= rf_1 0) (< ts_init_x ts_1)))
(assert (=> (= rf_1 1) (< ts_0 ts_1)))
(assert (=> (= rf_3 0) (< ts_init_y ts_3)))
(assert (=> (= rf_3 1) (< ts_4 ts_3)))
(assert (=> (= rf_6 0) (< ts_init_x ts_6)))
(assert (=> (= rf_6 1) (< ts_0 ts_6)))
(assert (< ts_init_x ts_0))
(assert (< ts_init_y ts_4))
(assert (=> (= rf_1 0) (< ts_1 ts_0)))
(assert (=> (= rf_3 0) (< ts_3 ts_4)))
(assert (=> (= rf_6 0) (< ts_6 ts_0)))
(assert (and (or (= rf_1 1)) (or (= rf_3 0)) (or (= rf_6 0))))

(check-sat)

; ── UNSAT Proof Certificate ──
; Status: UNSAT — forbidden outcome UNREACHABLE under ARM
; Unsat core size: 6 assertions (out of 36)
; Core assertions: track_33, track_34, track_22, track_forbidden, track_23, track_25
(get-unsat-core)

(exit)