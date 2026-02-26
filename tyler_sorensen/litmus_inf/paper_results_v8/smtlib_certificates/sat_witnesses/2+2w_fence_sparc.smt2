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
; Memory model: PSO
; Pattern: 2+2w_fence
; Forbidden outcome: {"r0": 2, "r1": 1}
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

; ── Pattern: 2+2w_fence ─────────────────────────
; Threads: 3
; Addresses: x, y
; Memory model: PSO
; Forbidden outcome: {'r0': 2, 'r1': 1}
;
; Operations:
;   e0: T0 Store x = 1
;   e1: T0 Fence
;   e2: T0 Store y = 1
;   e3: T1 Store y = 2
;   e4: T1 Fence
;   e5: T1 Store x = 2
;   e6: T2 Load y → r0
;   e7: T2 Fence
;   e8: T2 Load x → r1

(set-logic QF_LIA)

(declare-fun rf_6 () Int)
(declare-fun rf_8 () Int)
(declare-fun co_x_2_1 () Bool)
(declare-fun co_x_1_2 () Bool)
(declare-fun co_y_2_1 () Bool)
(declare-fun co_y_1_2 () Bool)
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
(assert (>= rf_6 0))
(assert (< rf_6 3))
(assert (>= rf_8 0))
(assert (< rf_8 3))
(assert (or co_x_1_2 co_x_2_1))
(assert (not (and co_x_1_2 co_x_2_1)))
(assert (or co_y_1_2 co_y_2_1))
(assert (not (and co_y_1_2 co_y_2_1)))
(assert (>= ts_0 0))
(assert (< ts_0 110))
(assert (>= ts_1 0))
(assert (< ts_1 110))
(assert (>= ts_2 0))
(assert (< ts_2 110))
(assert (>= ts_3 0))
(assert (< ts_3 110))
(assert (>= ts_4 0))
(assert (< ts_4 110))
(assert (>= ts_5 0))
(assert (< ts_5 110))
(assert (>= ts_6 0))
(assert (< ts_6 110))
(assert (>= ts_7 0))
(assert (< ts_7 110))
(assert (>= ts_8 0))
(assert (< ts_8 110))
(assert (>= ts_init_x 0))
(assert (>= ts_init_y 0))
(assert (< ts_0 ts_2))
(assert (< ts_3 ts_5))
(assert (< ts_6 ts_8))
(assert (=> (= rf_6 0) (< ts_init_y ts_6)))
(assert (=> (= rf_6 1) (< ts_2 ts_6)))
(assert (=> (= rf_6 2) (< ts_3 ts_6)))
(assert (=> (= rf_8 0) (< ts_init_x ts_8)))
(assert (=> (= rf_8 1) (< ts_0 ts_8)))
(assert (=> (= rf_8 2) (< ts_5 ts_8)))
(assert (< ts_init_x ts_0))
(assert (< ts_init_x ts_5))
(assert (=> co_x_1_2 (< ts_0 ts_5)))
(assert (=> co_x_2_1 (< ts_5 ts_0)))
(assert (< ts_init_y ts_2))
(assert (< ts_init_y ts_3))
(assert (=> co_y_1_2 (< ts_2 ts_3)))
(assert (=> co_y_2_1 (< ts_3 ts_2)))
(assert (=> (= rf_6 0) (< ts_6 ts_2)))
(assert (=> (= rf_6 0) (< ts_6 ts_3)))
(assert (=> (and (= rf_6 1) co_y_1_2) (< ts_6 ts_3)))
(assert (=> (and (= rf_6 2) co_y_2_1) (< ts_6 ts_2)))
(assert (=> (= rf_8 0) (< ts_8 ts_0)))
(assert (=> (= rf_8 0) (< ts_8 ts_5)))
(assert (=> (and (= rf_8 1) co_x_1_2) (< ts_8 ts_5)))
(assert (=> (and (= rf_8 2) co_x_2_1) (< ts_8 ts_0)))
(assert (and (or (= rf_6 2)) (or (= rf_8 1))))

(check-sat)
(get-model)

; ── SAT Witness (counterexample execution) ──
; Status: SAT — forbidden outcome IS reachable under PSO
;   rf_6 = 2
;   rf_8 = 1
;   co_x_1_2 = True
;   co_x_2_1 = False
;   co_y_1_2 = False
;   co_y_2_1 = True

(exit)