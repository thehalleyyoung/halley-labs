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
; Memory model: OpenCL-Dev
; Pattern: sb_3thread_fence
; Forbidden outcome: {"r0": 0, "r1": 0, "r2": 0}
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

; ── Pattern: sb_3thread_fence ─────────────────────────
; Threads: 3
; Addresses: x, y, z
; Memory model: OpenCL-Dev
; Forbidden outcome: {'r0': 0, 'r1': 0, 'r2': 0}
;
; Operations:
;   e0: T0 Store x = 1
;   e1: T0 Fence
;   e2: T0 Load y → r0
;   e3: T1 Store y = 1
;   e4: T1 Fence
;   e5: T1 Load z → r1
;   e6: T2 Store z = 1
;   e7: T2 Fence
;   e8: T2 Load x → r2

(set-logic QF_LIA)

(declare-fun rf_2 () Int)
(declare-fun rf_5 () Int)
(declare-fun rf_8 () Int)
(declare-fun ts_0 () Int)
(declare-fun ts_1 () Int)
(declare-fun ts_2 () Int)
(declare-fun ts_3 () Int)
(declare-fun ts_4 () Int)
(declare-fun ts_5 () Int)
(declare-fun ts_6 () Int)
(declare-fun ts_7 () Int)
(declare-fun ts_8 () Int)
(declare-fun ts_init_x () Int)
(declare-fun ts_init_y () Int)
(declare-fun ts_init_z () Int)
(assert (>= rf_2 0))
(assert (< rf_2 2))
(assert (>= rf_5 0))
(assert (< rf_5 2))
(assert (>= rf_8 0))
(assert (< rf_8 2))
(assert (>= ts_0 0))
(assert (< ts_0 120))
(assert (>= ts_1 0))
(assert (< ts_1 120))
(assert (>= ts_2 0))
(assert (< ts_2 120))
(assert (>= ts_3 0))
(assert (< ts_3 120))
(assert (>= ts_4 0))
(assert (< ts_4 120))
(assert (>= ts_5 0))
(assert (< ts_5 120))
(assert (>= ts_6 0))
(assert (< ts_6 120))
(assert (>= ts_7 0))
(assert (< ts_7 120))
(assert (>= ts_8 0))
(assert (< ts_8 120))
(assert (>= ts_init_x 0))
(assert (>= ts_init_y 0))
(assert (>= ts_init_z 0))
(assert (< ts_0 ts_2))
(assert (< ts_3 ts_5))
(assert (< ts_6 ts_8))
(assert (=> (= rf_2 0) (< ts_init_y ts_2)))
(assert (=> (= rf_2 1) (< ts_3 ts_2)))
(assert (=> (= rf_5 0) (< ts_init_z ts_5)))
(assert (=> (= rf_5 1) (< ts_6 ts_5)))
(assert (=> (= rf_8 0) (< ts_init_x ts_8)))
(assert (=> (= rf_8 1) (< ts_0 ts_8)))
(assert (< ts_init_x ts_0))
(assert (< ts_init_y ts_3))
(assert (< ts_init_z ts_6))
(assert (=> (= rf_2 0) (< ts_2 ts_3)))
(assert (=> (= rf_5 0) (< ts_5 ts_6)))
(assert (=> (= rf_8 0) (< ts_8 ts_0)))
(assert (and (or (= rf_2 0)) (or (= rf_5 0)) (or (= rf_8 0))))

(check-sat)

; ── UNSAT Proof Certificate ──
; Status: UNSAT — forbidden outcome UNREACHABLE under OpenCL-Dev
; Unsat core size: 7 assertions (out of 43)
; Core assertions: track_41, track_40, track_forbidden, track_29, track_27, track_39, track_28
(get-unsat-core)

(exit)