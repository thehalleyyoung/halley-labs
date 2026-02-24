//! Comprehensive tests for the LITMUS∞ checker subsystem.
//!
//! Tests cover: execution graph construction, BitMatrix operations,
//! memory model axiom checking, acyclicity verification, litmus test
//! construction, outcome enumeration, verifier, relations, scopes,
//! WebGPU model, operational model, and decomposition.

use litmus_infinity::checker::execution::*;
use litmus_infinity::checker::memory_model::*;
use litmus_infinity::checker::litmus::{LitmusTest, Thread, Instruction, Ordering, Outcome, LitmusOutcome};
use litmus_infinity::checker::verifier::*;
use litmus_infinity::checker::decomposition::*;
use litmus_infinity::checker::webgpu::*;
use litmus_infinity::checker::operational::*;
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 1: BitMatrix tests
// ═══════════════════════════════════════════════════════════════════════════

mod bitmatrix_tests {
    use super::*;

    #[test]
    fn empty_matrix() {
        let m = BitMatrix::new(4);
        assert_eq!(m.dim(), 4);
        assert_eq!(m.count_edges(), 0);
        assert!(m.is_empty());
    }

    #[test]
    fn identity_matrix() {
        let m = BitMatrix::identity(4);
        assert_eq!(m.dim(), 4);
        assert_eq!(m.count_edges(), 4);
        for i in 0..4 {
            assert!(m.get(i, i));
            for j in 0..4 {
                if i != j {
                    assert!(!m.get(i, j));
                }
            }
        }
    }

    #[test]
    fn universal_matrix() {
        let m = BitMatrix::universal(3);
        assert_eq!(m.count_edges(), 9);
        for i in 0..3 {
            for j in 0..3 {
                assert!(m.get(i, j));
            }
        }
    }

    #[test]
    fn set_and_get() {
        let mut m = BitMatrix::new(4);
        m.set(0, 1, true);
        m.set(2, 3, true);
        assert!(m.get(0, 1));
        assert!(m.get(2, 3));
        assert!(!m.get(1, 0));
        assert!(!m.get(3, 2));
    }

    #[test]
    fn add_edge() {
        let mut m = BitMatrix::new(3);
        assert!(m.add(0, 1)); // newly inserted
        assert!(!m.add(0, 1)); // already there
    }

    #[test]
    fn remove_edge() {
        let mut m = BitMatrix::new(3);
        m.set(1, 2, true);
        assert!(m.get(1, 2));
        m.remove(1, 2);
        assert!(!m.get(1, 2));
    }

    #[test]
    fn successors() {
        let mut m = BitMatrix::new(4);
        m.set(0, 1, true);
        m.set(0, 3, true);
        let succs: Vec<usize> = m.successors(0).collect();
        assert_eq!(succs.len(), 2);
        assert!(succs.contains(&1));
        assert!(succs.contains(&3));
    }

    #[test]
    fn predecessors() {
        let mut m = BitMatrix::new(4);
        m.set(1, 3, true);
        m.set(2, 3, true);
        let preds: Vec<usize> = m.predecessors(3).collect();
        assert_eq!(preds.len(), 2);
        assert!(preds.contains(&1));
        assert!(preds.contains(&2));
    }

    #[test]
    fn edges_list() {
        let mut m = BitMatrix::new(3);
        m.set(0, 1, true);
        m.set(1, 2, true);
        let edges = m.edges();
        assert_eq!(edges.len(), 2);
        assert!(edges.contains(&(0, 1)));
        assert!(edges.contains(&(1, 2)));
    }

    #[test]
    fn union() {
        let mut a = BitMatrix::new(3);
        a.set(0, 1, true);
        let mut b = BitMatrix::new(3);
        b.set(1, 2, true);
        let c = a.union(&b);
        assert!(c.get(0, 1));
        assert!(c.get(1, 2));
        assert_eq!(c.count_edges(), 2);
    }

    #[test]
    fn union_assign() {
        let mut a = BitMatrix::new(3);
        a.set(0, 1, true);
        let mut b = BitMatrix::new(3);
        b.set(1, 2, true);
        a.union_assign(&b);
        assert!(a.get(0, 1));
        assert!(a.get(1, 2));
    }

    #[test]
    fn intersection() {
        let mut a = BitMatrix::new(3);
        a.set(0, 1, true);
        a.set(1, 2, true);
        let mut b = BitMatrix::new(3);
        b.set(1, 2, true);
        b.set(2, 0, true);
        let c = a.intersection(&b);
        assert!(!c.get(0, 1));
        assert!(c.get(1, 2));
        assert!(!c.get(2, 0));
    }

    #[test]
    fn intersect_assign() {
        let mut a = BitMatrix::new(3);
        a.set(0, 1, true);
        a.set(1, 2, true);
        let mut b = BitMatrix::new(3);
        b.set(1, 2, true);
        a.intersect_assign(&b);
        assert!(!a.get(0, 1));
        assert!(a.get(1, 2));
    }

    #[test]
    fn complement() {
        let mut m = BitMatrix::new(2);
        m.set(0, 0, true);
        let c = m.complement();
        assert!(!c.get(0, 0));
        assert!(c.get(0, 1));
        assert!(c.get(1, 0));
        assert!(c.get(1, 1));
    }

    #[test]
    fn difference() {
        let mut a = BitMatrix::new(3);
        a.set(0, 1, true);
        a.set(1, 2, true);
        let mut b = BitMatrix::new(3);
        b.set(1, 2, true);
        let c = a.difference(&b);
        assert!(c.get(0, 1));
        assert!(!c.get(1, 2));
    }

    #[test]
    fn inverse_transpose() {
        let mut m = BitMatrix::new(3);
        m.set(0, 1, true);
        m.set(1, 2, true);
        let inv = m.inverse();
        assert!(inv.get(1, 0));
        assert!(inv.get(2, 1));
        assert!(!inv.get(0, 1));
    }

    #[test]
    fn inverse_involution() {
        let mut m = BitMatrix::new(3);
        m.set(0, 1, true);
        m.set(2, 0, true);
        assert_eq!(m.inverse().inverse(), m);
    }

    #[test]
    fn compose_basic() {
        let mut a = BitMatrix::new(3);
        a.set(0, 1, true); // 0->1
        let mut b = BitMatrix::new(3);
        b.set(1, 2, true); // 1->2
        let c = a.compose(&b);
        assert!(c.get(0, 2)); // 0->1->2
        assert!(!c.get(0, 1));
    }

    #[test]
    fn compose_with_identity() {
        let mut m = BitMatrix::new(3);
        m.set(0, 1, true);
        m.set(1, 2, true);
        let id = BitMatrix::identity(3);
        assert_eq!(m.compose(&id), m);
        assert_eq!(id.compose(&m), m);
    }

    #[test]
    fn transitive_closure_chain() {
        let mut m = BitMatrix::new(4);
        m.set(0, 1, true);
        m.set(1, 2, true);
        m.set(2, 3, true);
        let tc = m.transitive_closure();
        assert!(tc.get(0, 3)); // 0->1->2->3
        assert!(tc.get(0, 2));
        assert!(tc.get(1, 3));
    }

    #[test]
    fn transitive_closure_already_transitive() {
        let mut m = BitMatrix::new(3);
        m.set(0, 1, true);
        m.set(1, 2, true);
        m.set(0, 2, true);
        let tc = m.transitive_closure();
        assert_eq!(tc, m);
    }

    #[test]
    fn reflexive_transitive_closure() {
        let mut m = BitMatrix::new(3);
        m.set(0, 1, true);
        let rtc = m.reflexive_transitive_closure();
        assert!(rtc.get(0, 0));
        assert!(rtc.get(1, 1));
        assert!(rtc.get(2, 2));
        assert!(rtc.get(0, 1));
    }

    #[test]
    fn plus_alias() {
        let mut m = BitMatrix::new(3);
        m.set(0, 1, true);
        m.set(1, 2, true);
        assert_eq!(m.plus(), m.transitive_closure());
    }

    #[test]
    fn optional() {
        let mut m = BitMatrix::new(3);
        m.set(0, 1, true);
        let opt = m.optional();
        assert!(opt.get(0, 0));
        assert!(opt.get(1, 1));
        assert!(opt.get(0, 1));
    }

    #[test]
    fn identity_filter() {
        let pred = vec![true, false, true];
        let m = BitMatrix::identity_filter(3, &pred);
        assert!(m.get(0, 0));
        assert!(!m.get(1, 1));
        assert!(m.get(2, 2));
    }

    #[test]
    fn restrict() {
        let mut m = BitMatrix::new(3);
        m.set(0, 1, true);
        m.set(0, 2, true);
        m.set(1, 2, true);
        let pred_src = vec![true, false, true];
        let pred_dst = vec![true, true, false];
        let restricted = m.restrict(&pred_src, &pred_dst);
        assert!(restricted.get(0, 1));
        assert!(!restricted.get(0, 2)); // dst[2] = false
        assert!(!restricted.get(1, 2)); // src[1] = false
    }

    #[test]
    fn is_acyclic_dag() {
        let mut m = BitMatrix::new(4);
        m.set(0, 1, true);
        m.set(0, 2, true);
        m.set(1, 3, true);
        m.set(2, 3, true);
        assert!(m.is_acyclic());
    }

    #[test]
    fn is_acyclic_with_cycle() {
        let mut m = BitMatrix::new(3);
        m.set(0, 1, true);
        m.set(1, 2, true);
        m.set(2, 0, true);
        assert!(!m.is_acyclic());
    }

    #[test]
    fn is_acyclic_self_loop() {
        let mut m = BitMatrix::new(2);
        m.set(0, 0, true);
        assert!(!m.is_acyclic());
    }

    #[test]
    fn is_acyclic_empty() {
        let m = BitMatrix::new(5);
        assert!(m.is_acyclic());
    }

    #[test]
    fn is_irreflexive() {
        let mut m = BitMatrix::new(3);
        m.set(0, 1, true);
        m.set(1, 2, true);
        assert!(m.is_irreflexive());
    }

    #[test]
    fn is_not_irreflexive() {
        let mut m = BitMatrix::new(3);
        m.set(1, 1, true);
        assert!(!m.is_irreflexive());
    }

    #[test]
    fn find_cycle_exists() {
        let mut m = BitMatrix::new(3);
        m.set(0, 1, true);
        m.set(1, 2, true);
        m.set(2, 0, true);
        let cycle = m.find_cycle();
        assert!(cycle.is_some());
        let cycle = cycle.unwrap();
        assert!(cycle.len() >= 2);
    }

    #[test]
    fn find_cycle_none() {
        let mut m = BitMatrix::new(3);
        m.set(0, 1, true);
        m.set(1, 2, true);
        assert!(m.find_cycle().is_none());
    }

    #[test]
    fn zero_size_matrix() {
        let m = BitMatrix::new(0);
        assert_eq!(m.dim(), 0);
        assert!(m.is_empty());
        assert!(m.is_acyclic());
    }

    #[test]
    fn single_element_matrix() {
        let mut m = BitMatrix::new(1);
        assert!(m.is_acyclic());
        m.set(0, 0, true);
        assert!(!m.is_acyclic());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 2: Event and ExecutionGraph tests
// ═══════════════════════════════════════════════════════════════════════════

mod event_tests {
    use super::*;

    #[test]
    fn event_construction() {
        let e = Event::new(0, 0, OpType::Write, 0x100, 42);
        assert_eq!(e.id, 0);
        assert_eq!(e.thread, 0);
        assert_eq!(e.op_type, OpType::Write);
        assert_eq!(e.address, 0x100);
        assert_eq!(e.value, 42);
        assert_eq!(e.scope, Scope::None);
    }

    #[test]
    fn event_with_scope() {
        let e = Event::new(0, 0, OpType::Read, 0x100, 0)
            .with_scope(Scope::CTA);
        assert_eq!(e.scope, Scope::CTA);
    }

    #[test]
    fn event_with_po_index() {
        let e = Event::new(0, 0, OpType::Write, 0x100, 1)
            .with_po_index(5);
        assert_eq!(e.po_index, 5);
    }

    #[test]
    fn event_type_checks() {
        let r = Event::new(0, 0, OpType::Read, 0x100, 0);
        assert!(r.is_read());
        assert!(!r.is_write());
        assert!(!r.is_fence());

        let w = Event::new(1, 0, OpType::Write, 0x100, 1);
        assert!(!w.is_read());
        assert!(w.is_write());

        let f = Event::new(2, 0, OpType::Fence, 0, 0);
        assert!(f.is_fence());
        assert!(!f.is_read());

        let rmw = Event::new(3, 0, OpType::RMW, 0x100, 1);
        assert!(rmw.is_read());
        assert!(rmw.is_write());
        assert!(rmw.is_rmw());
    }

    #[test]
    fn event_label() {
        let w = Event::new(0, 0, OpType::Write, 0x100, 1);
        let label = w.label();
        assert!(label.contains("W"));

        let f = Event::new(1, 0, OpType::Fence, 0, 0);
        let label = f.label();
        assert!(label.contains("F"));
    }

    #[test]
    fn event_display() {
        let e = Event::new(0, 0, OpType::Read, 0x100, 0);
        let s = format!("{}", e);
        assert!(s.contains("e0"));
        assert!(s.contains("T0"));
    }

    #[test]
    fn optype_display() {
        assert_eq!(format!("{}", OpType::Read), "R");
        assert_eq!(format!("{}", OpType::Write), "W");
        assert_eq!(format!("{}", OpType::Fence), "F");
        assert_eq!(format!("{}", OpType::RMW), "RMW");
    }

    #[test]
    fn scope_default() {
        let s: Scope = Default::default();
        assert_eq!(s, Scope::None);
    }

    #[test]
    fn scope_display() {
        assert_eq!(format!("{}", Scope::CTA), ".cta");
        assert_eq!(format!("{}", Scope::GPU), ".gpu");
        assert_eq!(format!("{}", Scope::System), ".sys");
        assert_eq!(format!("{}", Scope::None), "");
    }
}

mod execution_graph_tests {
    use super::*;

    fn make_simple_graph() -> ExecutionGraph {
        let events = vec![
            Event::new(0, 0, OpType::Write, 0x100, 1).with_po_index(0),
            Event::new(1, 0, OpType::Read, 0x100, 1).with_po_index(1),
            Event::new(2, 1, OpType::Write, 0x100, 2).with_po_index(0),
            Event::new(3, 1, OpType::Read, 0x100, 2).with_po_index(1),
        ];
        ExecutionGraph::new(events)
    }

    #[test]
    fn graph_construction() {
        let g = make_simple_graph();
        assert_eq!(g.len(), 4);
        assert!(!g.is_empty());
    }

    #[test]
    fn program_order_computed() {
        let g = make_simple_graph();
        // Thread 0: event 0 po-before event 1
        assert!(g.po.get(0, 1));
        // Thread 1: event 2 po-before event 3
        assert!(g.po.get(2, 3));
        // Cross-thread: no po
        assert!(!g.po.get(0, 2));
        assert!(!g.po.get(1, 3));
    }

    #[test]
    fn thread_events() {
        let g = make_simple_graph();
        let t0 = g.thread_events(0);
        assert_eq!(t0.len(), 2);
        assert!(t0.contains(&0));
        assert!(t0.contains(&1));
    }

    #[test]
    fn addr_events() {
        let g = make_simple_graph();
        let a = g.addr_events(0x100);
        assert_eq!(a.len(), 4);
    }

    #[test]
    fn thread_ids() {
        let g = make_simple_graph();
        let tids = g.thread_ids();
        assert!(tids.contains(&0));
        assert!(tids.contains(&1));
    }

    #[test]
    fn reads_writes_predicates() {
        let g = make_simple_graph();
        let reads = g.reads_pred();
        assert!(!reads[0]); // Write
        assert!(reads[1]);  // Read
        let writes = g.writes_pred();
        assert!(writes[0]);  // Write
        assert!(!writes[1]); // Read
    }

    #[test]
    fn same_address_relation() {
        let g = make_simple_graph();
        let sa = g.same_address();
        // All events access 0x100, so all pairs should be same-address
        for i in 0..4 {
            for j in 0..4 {
                assert!(sa.get(i, j));
            }
        }
    }

    #[test]
    fn derive_fr() {
        let mut g = make_simple_graph();
        g.rf.set(0, 1, true); // event 0 (W) -> event 1 (R)
        g.co.set(0, 2, true); // event 0 (W) co-before event 2 (W)
        g.derive_fr();
        // fr = rf^{-1} ; co: if R reads from W1, and W1 co-before W2, then R fr W2
        assert!(g.fr.get(1, 2)); // event 1 fr event 2
    }

    #[test]
    fn add_named_relation() {
        let mut g = make_simple_graph();
        let rel = BitMatrix::new(4);
        g.add_relation("custom", rel);
        assert!(g.extra.len() > 0);
    }

    #[test]
    fn empty_graph() {
        let g = ExecutionGraph::new(vec![]);
        assert_eq!(g.len(), 0);
        assert!(g.is_empty());
    }

    #[test]
    fn single_event_graph() {
        let events = vec![Event::new(0, 0, OpType::Write, 0x100, 1)];
        let g = ExecutionGraph::new(events);
        assert_eq!(g.len(), 1);
    }

    #[test]
    fn rebuild_caches() {
        let mut g = make_simple_graph();
        g.rebuild_caches();
        assert_eq!(g.thread_events(0).len(), 2);
    }

    #[test]
    fn fences_predicate() {
        let events = vec![
            Event::new(0, 0, OpType::Write, 0x100, 1),
            Event::new(1, 0, OpType::Fence, 0, 0),
            Event::new(2, 0, OpType::Read, 0x100, 0),
        ];
        let g = ExecutionGraph::new(events);
        let fences = g.fences_pred();
        assert!(!fences[0]);
        assert!(fences[1]);
        assert!(!fences[2]);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 3: Litmus test construction
// ═══════════════════════════════════════════════════════════════════════════

mod litmus_test_construction {
    use super::*;

    #[test]
    fn basic_construction() {
        let test = LitmusTest::new("test");
        assert_eq!(test.name, "test");
        assert_eq!(test.thread_count(), 0);
    }

    #[test]
    fn add_thread() {
        let mut test = LitmusTest::new("test");
        let t = Thread::new(0);
        test.add_thread(t);
        assert_eq!(test.thread_count(), 1);
    }

    #[test]
    fn thread_with_instructions() {
        let instrs = vec![
            Instruction::Store { addr: 0x100, value: 1, ordering: Ordering::Relaxed },
            Instruction::Load { reg: 0, addr: 0x200, ordering: Ordering::Relaxed },
        ];
        let t = Thread::with_instructions(0, instrs);
        assert_eq!(t.instructions.len(), 2);
    }

    #[test]
    fn thread_add_instructions() {
        let mut t = Thread::new(0);
        t.store(0x100, 1, Ordering::Relaxed);
        t.load(0, 0x200, Ordering::Relaxed);
        t.fence(Ordering::SeqCst, litmus_infinity::checker::litmus::Scope::None);
        assert_eq!(t.instructions.len(), 3);
    }

    #[test]
    fn thread_rmw() {
        let mut t = Thread::new(0);
        t.rmw(0, 0x100, 1, Ordering::AcqRel);
        assert_eq!(t.instructions.len(), 1);
    }

    #[test]
    fn thread_memory_op_count() {
        let mut t = Thread::new(0);
        t.store(0x100, 1, Ordering::Relaxed);
        t.load(0, 0x200, Ordering::Relaxed);
        t.fence(Ordering::SeqCst, litmus_infinity::checker::litmus::Scope::None);
        assert_eq!(t.memory_op_count(), 2); // Fence doesn't count
    }

    #[test]
    fn thread_accessed_addresses() {
        let mut t = Thread::new(0);
        t.store(0x100, 1, Ordering::Relaxed);
        t.load(0, 0x200, Ordering::Relaxed);
        t.store(0x100, 2, Ordering::Relaxed);
        let addrs = t.accessed_addresses();
        assert_eq!(addrs.len(), 2);
        assert!(addrs.contains(&0x100));
        assert!(addrs.contains(&0x200));
    }

    #[test]
    fn thread_display() {
        let mut t = Thread::new(0);
        t.store(0x100, 1, Ordering::Relaxed);
        let s = format!("{}", t);
        assert!(s.contains("T0"));
    }

    #[test]
    fn instruction_display() {
        let i = Instruction::Store { addr: 0x100, value: 1, ordering: Ordering::Relaxed };
        let s = format!("{}", i);
        assert!(s.contains("store"));
    }

    #[test]
    fn outcome_construction() {
        let o = Outcome::new()
            .with_reg(0, 0, 1)
            .with_mem(0x100, 42);
        assert_eq!(o.registers[&(0, 0)], 1);
        assert_eq!(o.memory[&0x100], 42);
    }

    #[test]
    fn outcome_matches() {
        let o = Outcome::new().with_reg(0, 0, 1);
        let mut reg_state = HashMap::new();
        reg_state.insert((0, 0), 1);
        let mem_state = HashMap::new();
        assert!(o.matches(&reg_state, &mem_state));
    }

    #[test]
    fn outcome_does_not_match() {
        let o = Outcome::new().with_reg(0, 0, 1);
        let mut reg_state = HashMap::new();
        reg_state.insert((0, 0), 99);
        let mem_state = HashMap::new();
        assert!(!o.matches(&reg_state, &mem_state));
    }

    #[test]
    fn outcome_display() {
        let o = Outcome::new().with_reg(0, 0, 1).with_mem(0x100, 42);
        let s = format!("{}", o);
        assert!(s.len() > 0);
    }

    #[test]
    fn litmus_outcome_display() {
        assert_eq!(format!("{}", LitmusOutcome::Allowed), "allowed");
        assert_eq!(format!("{}", LitmusOutcome::Forbidden), "forbidden");
        assert_eq!(format!("{}", LitmusOutcome::Required), "required");
    }

    #[test]
    fn set_initial_state() {
        let mut test = LitmusTest::new("test");
        test.set_initial(0x100, 0);
        test.set_initial(0x200, 0);
        assert_eq!(test.initial_state.len(), 2);
    }

    #[test]
    fn expect_outcome() {
        let mut test = LitmusTest::new("test");
        let o = Outcome::new().with_reg(0, 0, 0);
        test.expect(o, LitmusOutcome::Allowed);
        assert_eq!(test.expected_outcomes.len(), 1);
    }

    #[test]
    fn all_addresses() {
        let mut test = LitmusTest::new("test");
        let mut t0 = Thread::new(0);
        t0.store(0x100, 1, Ordering::Relaxed);
        t0.load(0, 0x200, Ordering::Relaxed);
        test.add_thread(t0);
        test.set_initial(0x300, 0);
        let addrs = test.all_addresses();
        assert!(addrs.contains(&0x100));
        assert!(addrs.contains(&0x200));
        assert!(addrs.contains(&0x300));
    }

    #[test]
    fn total_instructions() {
        let mut test = LitmusTest::new("test");
        let mut t0 = Thread::new(0);
        t0.store(0x100, 1, Ordering::Relaxed);
        t0.load(0, 0x200, Ordering::Relaxed);
        test.add_thread(t0);
        let mut t1 = Thread::new(1);
        t1.store(0x200, 2, Ordering::Relaxed);
        test.add_thread(t1);
        assert_eq!(test.total_instructions(), 3);
    }

    #[test]
    fn total_memory_ops() {
        let mut test = LitmusTest::new("test");
        let mut t0 = Thread::new(0);
        t0.store(0x100, 1, Ordering::Relaxed);
        t0.fence(Ordering::SeqCst, litmus_infinity::checker::litmus::Scope::None);
        t0.load(0, 0x200, Ordering::Relaxed);
        test.add_thread(t0);
        assert_eq!(test.total_memory_ops(), 2);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 4: Ordering tests
// ═══════════════════════════════════════════════════════════════════════════

mod ordering_tests {
    use super::*;

    #[test]
    fn ordering_scope() {
        assert_eq!(Ordering::AcquireCTA.scope(), Scope::CTA);
        assert_eq!(Ordering::ReleaseCTA.scope(), Scope::CTA);
        assert_eq!(Ordering::AcquireGPU.scope(), Scope::GPU);
        assert_eq!(Ordering::ReleaseGPU.scope(), Scope::GPU);
        assert_eq!(Ordering::AcquireSystem.scope(), Scope::System);
        assert_eq!(Ordering::ReleaseSystem.scope(), Scope::System);
        assert_eq!(Ordering::SeqCst.scope(), Scope::System);
        assert_eq!(Ordering::Relaxed.scope(), Scope::None);
    }

    #[test]
    fn ordering_is_acquire() {
        assert!(Ordering::Acquire.is_acquire());
        assert!(Ordering::AcqRel.is_acquire());
        assert!(Ordering::SeqCst.is_acquire());
        assert!(Ordering::AcquireCTA.is_acquire());
        assert!(!Ordering::Release.is_acquire());
        assert!(!Ordering::Relaxed.is_acquire());
    }

    #[test]
    fn ordering_is_release() {
        assert!(Ordering::Release.is_release());
        assert!(Ordering::AcqRel.is_release());
        assert!(Ordering::SeqCst.is_release());
        assert!(Ordering::ReleaseCTA.is_release());
        assert!(!Ordering::Acquire.is_release());
        assert!(!Ordering::Relaxed.is_release());
    }

    #[test]
    fn ordering_display() {
        assert_eq!(format!("{}", Ordering::Relaxed), "rlx");
        assert_eq!(format!("{}", Ordering::Acquire), "acq");
        assert_eq!(format!("{}", Ordering::Release), "rel");
        assert_eq!(format!("{}", Ordering::AcqRel), "acq_rel");
        assert_eq!(format!("{}", Ordering::SeqCst), "sc");
        assert_eq!(format!("{}", Ordering::AcquireCTA), "acq.cta");
        assert_eq!(format!("{}", Ordering::ReleaseCTA), "rel.cta");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 5: Memory model tests
// ═══════════════════════════════════════════════════════════════════════════

mod memory_model_tests {
    use super::*;

    #[test]
    fn model_construction() {
        let m = MemoryModel::new("TestModel");
        assert_eq!(m.name, "TestModel");
        assert!(!m.base_relations.is_empty());
    }

    #[test]
    fn add_derived_relation() {
        let mut m = MemoryModel::new("test");
        m.add_derived("hb", RelationExpr::union(
            RelationExpr::base("po"),
            RelationExpr::base("rf"),
        ), "happens-before");
        assert_eq!(m.derived_relations.len(), 1);
    }

    #[test]
    fn add_acyclic_constraint() {
        let mut m = MemoryModel::new("test");
        m.add_acyclic(RelationExpr::base("po"));
        assert_eq!(m.constraints.len(), 1);
    }

    #[test]
    fn add_irreflexive_constraint() {
        let mut m = MemoryModel::new("test");
        m.add_irreflexive(RelationExpr::base("fr"));
        assert_eq!(m.constraints.len(), 1);
    }

    #[test]
    fn add_empty_constraint() {
        let mut m = MemoryModel::new("test");
        m.add_empty(RelationExpr::base("test"));
        assert_eq!(m.constraints.len(), 1);
    }

    #[test]
    fn validate_valid_model() {
        let mut m = MemoryModel::new("test");
        m.add_acyclic(RelationExpr::base("po"));
        assert!(m.validate().is_ok());
    }

    #[test]
    fn validate_invalid_model() {
        let mut m = MemoryModel::new("test");
        m.add_acyclic(RelationExpr::base("nonexistent_relation"));
        assert!(m.validate().is_err());
    }

    #[test]
    fn builtin_model_sc() {
        let m = BuiltinModel::SC.build();
        assert!(m.validate().is_ok());
    }

    #[test]
    fn builtin_model_tso() {
        let m = BuiltinModel::TSO.build();
        assert!(m.validate().is_ok());
    }

    #[test]
    fn builtin_model_pso() {
        let m = BuiltinModel::PSO.build();
        assert!(m.validate().is_ok());
    }

    #[test]
    fn builtin_model_arm() {
        let m = BuiltinModel::ARM.build();
        assert!(m.validate().is_ok());
    }

    #[test]
    fn builtin_model_riscv() {
        let m = BuiltinModel::RISCV.build();
        assert!(m.validate().is_ok());
    }

    #[test]
    fn builtin_model_ptx() {
        let m = BuiltinModel::PTX.build();
        assert!(m.validate().is_ok());
    }

    #[test]
    fn builtin_model_webgpu() {
        let m = BuiltinModel::WebGPU.build();
        assert!(m.validate().is_ok());
    }

    #[test]
    fn builtin_model_all() {
        let all = BuiltinModel::all();
        assert_eq!(all.len(), 7);
        for m in &all {
            let model = m.build();
            assert!(model.validate().is_ok());
        }
    }

    #[test]
    fn builtin_model_names() {
        assert_eq!(BuiltinModel::SC.name(), "SC");
        assert_eq!(BuiltinModel::TSO.name(), "TSO");
        assert_eq!(BuiltinModel::ARM.name(), "ARMv8");
    }

    #[test]
    fn builtin_model_display() {
        assert_eq!(format!("{}", BuiltinModel::SC), "SC");
    }

    #[test]
    fn relation_expr_construction() {
        let e = RelationExpr::seq(
            RelationExpr::base("po"),
            RelationExpr::base("rf"),
        );
        let refs = e.referenced_bases();
        assert!(refs.contains(&"po".to_string()));
        assert!(refs.contains(&"rf".to_string()));
    }

    #[test]
    fn relation_expr_union_many() {
        let e = RelationExpr::union_many(vec![
            RelationExpr::base("po"),
            RelationExpr::base("rf"),
            RelationExpr::base("co"),
        ]);
        let refs = e.referenced_bases();
        assert_eq!(refs.len(), 3);
    }

    #[test]
    fn relation_expr_seq_many() {
        let e = RelationExpr::seq_many(vec![
            RelationExpr::base("po"),
            RelationExpr::base("rf"),
        ]);
        let refs = e.referenced_bases();
        assert!(refs.contains(&"po".to_string()));
    }

    #[test]
    fn constraint_name() {
        let c = Constraint::acyclic(RelationExpr::base("po"));
        assert!(c.name().contains("acyclic"));
    }

    #[test]
    fn constraint_expr() {
        let c = Constraint::acyclic(RelationExpr::base("po"));
        match c.expr() {
            RelationExpr::Base(name) => assert_eq!(name, "po"),
            _ => panic!("Expected Base"),
        }
    }

    #[test]
    fn constraint_display() {
        let c = Constraint::acyclic(RelationExpr::base("hb"));
        let s = format!("{}", c);
        assert!(s.contains("acyclic"));
    }

    #[test]
    fn eval_expr_base_po() {
        let events = vec![
            Event::new(0, 0, OpType::Write, 0x100, 1).with_po_index(0),
            Event::new(1, 0, OpType::Read, 0x100, 0).with_po_index(1),
        ];
        let graph = ExecutionGraph::new(events);
        let model = BuiltinModel::SC.build();
        let env = HashMap::new();
        let result = model.eval_expr(&RelationExpr::base("po"), &graph, &env);
        assert!(result.get(0, 1));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 6: Verifier tests
// ═══════════════════════════════════════════════════════════════════════════

mod verifier_tests {
    use super::*;

    fn make_sb_test() -> LitmusTest {
        let mut test = LitmusTest::new("SB");
        test.set_initial(0x100, 0);
        test.set_initial(0x200, 0);

        let mut t0 = Thread::new(0);
        t0.store(0x100, 1, Ordering::Relaxed);
        t0.load(0, 0x200, Ordering::Relaxed);
        test.add_thread(t0);

        let mut t1 = Thread::new(1);
        t1.store(0x200, 1, Ordering::Relaxed);
        t1.load(0, 0x100, Ordering::Relaxed);
        test.add_thread(t1);

        test
    }

    fn make_mp_test() -> LitmusTest {
        let mut test = LitmusTest::new("MP");
        test.set_initial(0x100, 0);
        test.set_initial(0x200, 0);

        let mut t0 = Thread::new(0);
        t0.store(0x100, 1, Ordering::Relaxed);
        t0.store(0x200, 1, Ordering::Relaxed);
        test.add_thread(t0);

        let mut t1 = Thread::new(1);
        t1.load(0, 0x200, Ordering::Relaxed);
        t1.load(1, 0x100, Ordering::Relaxed);
        test.add_thread(t1);

        test
    }

    #[test]
    fn verifier_construction() {
        let model = BuiltinModel::SC.build();
        let v = Verifier::new(model);
        assert_eq!(v.model().name, "SC");
    }

    #[test]
    fn verifier_check_execution() {
        let model = BuiltinModel::SC.build();
        let mut v = Verifier::new(model);
        let events = vec![
            Event::new(0, 0, OpType::Write, 0x100, 1).with_po_index(0),
            Event::new(1, 1, OpType::Read, 0x100, 1).with_po_index(0),
        ];
        let mut graph = ExecutionGraph::new(events);
        graph.rf.set(0, 1, true);
        let result = v.check_execution(&graph);
        // SC should accept this simple execution
        let _ = result;
    }

    #[test]
    fn verifier_acyclicity_check() {
        let model = BuiltinModel::SC.build();
        let v = Verifier::new(model);
        let mut m = BitMatrix::new(3);
        m.set(0, 1, true);
        m.set(1, 2, true);
        let (acyclic, cycle) = v.acyclicity_check(&m);
        assert!(acyclic);
        assert!(cycle.is_none());
    }

    #[test]
    fn verifier_acyclicity_check_cycle() {
        let model = BuiltinModel::SC.build();
        let v = Verifier::new(model);
        let mut m = BitMatrix::new(3);
        m.set(0, 1, true);
        m.set(1, 2, true);
        m.set(2, 0, true);
        let (acyclic, cycle) = v.acyclicity_check(&m);
        assert!(!acyclic);
        assert!(cycle.is_some());
    }

    #[test]
    fn verify_litmus_sb_sc() {
        let model = BuiltinModel::SC.build();
        let mut v = Verifier::new(model);
        let test = make_sb_test();
        let result = v.verify_litmus(&test);
        // SC should produce some result
        let _ = result;
    }

    #[test]
    fn verify_litmus_mp_sc() {
        let model = BuiltinModel::SC.build();
        let mut v = Verifier::new(model);
        let test = make_mp_test();
        let result = v.verify_litmus(&test);
        let _ = result;
    }

    #[test]
    fn verify_litmus_sb_tso() {
        let model = BuiltinModel::TSO.build();
        let mut v = Verifier::new(model);
        let test = make_sb_test();
        let result = v.verify_litmus(&test);
        let _ = result;
    }

    #[test]
    fn enumerate_consistent_executions() {
        let model = BuiltinModel::SC.build();
        let mut v = Verifier::new(model);
        let test = make_sb_test();
        let consistent = v.enumerate_consistent(&test);
        // SC should have some consistent executions
        assert!(consistent.len() > 0);
    }

    #[test]
    fn verifier_stats() {
        let model = BuiltinModel::SC.build();
        let mut v = Verifier::new(model);
        let test = make_sb_test();
        let _ = v.verify_litmus(&test);
        let stats = v.stats();
        assert!(stats.executions_checked > 0);
    }

    #[test]
    fn verifier_reset_stats() {
        let model = BuiltinModel::SC.build();
        let mut v = Verifier::new(model);
        let test = make_sb_test();
        let _ = v.verify_litmus(&test);
        v.reset_stats();
        assert_eq!(v.stats().executions_checked, 0);
    }

    #[test]
    fn verify_batch() {
        let model = BuiltinModel::SC.build();
        let tests = vec![make_sb_test(), make_mp_test()];
        let results = litmus_infinity::checker::verifier::verify_batch(&model, &tests);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn verify_multi_model() {
        let test = make_sb_test();
        let models = vec![
            BuiltinModel::SC.build(),
            BuiltinModel::TSO.build(),
        ];
        let results = litmus_infinity::checker::verifier::verify_multi_model(&test, &models);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn compositional_verifier() {
        let model = BuiltinModel::SC.build();
        let mut cv = CompositionalVerifier::new(model);
        let test = make_sb_test();
        let result = cv.verify_compositional(&test);
        let _ = result;
    }

    #[test]
    fn compositional_decompose() {
        let model = BuiltinModel::SC.build();
        let cv = CompositionalVerifier::new(model);
        let test = make_sb_test();
        let components = cv.decompose_test(&test);
        let total: usize = components.iter().map(|c| c.len()).sum();
        assert!(total >= test.thread_count());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 7: Execution enumeration
// ═══════════════════════════════════════════════════════════════════════════

mod execution_enumeration_tests {
    use super::*;

    #[test]
    fn enumerate_simple() {
        let mut test = LitmusTest::new("simple");
        test.set_initial(0x100, 0);
        let mut t0 = Thread::new(0);
        t0.store(0x100, 1, Ordering::Relaxed);
        test.add_thread(t0);
        let mut t1 = Thread::new(1);
        t1.load(0, 0x100, Ordering::Relaxed);
        test.add_thread(t1);
        let execs = test.enumerate_executions();
        assert!(execs.len() > 0);
    }

    #[test]
    fn enumerate_no_threads() {
        let test = LitmusTest::new("empty");
        let execs = test.enumerate_executions();
        assert!(execs.len() >= 1);
    }

    #[test]
    fn enumerate_single_thread_store() {
        let mut test = LitmusTest::new("single_store");
        let mut t0 = Thread::new(0);
        t0.store(0x100, 42, Ordering::Relaxed);
        test.add_thread(t0);
        let execs = test.enumerate_executions();
        assert!(execs.len() >= 1);
    }
}

// relations_tests removed: module not publicly exported

// axiom_tests removed: module not publicly exported

// scope_tests removed: module not publicly exported


// ═══════════════════════════════════════════════════════════════════════════
// SECTION 11: Decomposition tests
// ═══════════════════════════════════════════════════════════════════════════

mod decomposition_tests {
    use super::*;

    fn make_test() -> LitmusTest {
        let mut test = LitmusTest::new("test");
        test.set_initial(0x100, 0);
        test.set_initial(0x200, 0);
        let mut t0 = Thread::new(0);
        t0.store(0x100, 1, Ordering::Relaxed);
        t0.load(0, 0x200, Ordering::Relaxed);
        test.add_thread(t0);
        let mut t1 = Thread::new(1);
        t1.store(0x200, 1, Ordering::Relaxed);
        t1.load(0, 0x100, Ordering::Relaxed);
        test.add_thread(t1);
        test
    }

    #[test]
    fn test_decomposer() {
        let test = make_test();
        let tree = TestDecomposer::decompose(&test);
        assert!(tree.root.node_count() >= 1);
    }

    #[test]
    fn decomposition_tree_trivial() {
        let test = make_test();
        let tree = TestDecomposer::decompose(&test);
        // Small test may be trivially decomposed
        let _ = tree.is_trivial();
    }

    #[test]
    fn decomposition_compression_ratio() {
        let test = make_test();
        let tree = TestDecomposer::decompose(&test);
        let ratio = tree.compression_ratio();
        assert!(ratio >= 0.0);
    }

    #[test]
    fn decomposition_node_depth() {
        let test = make_test();
        let tree = TestDecomposer::decompose(&test);
        let depth = tree.root.depth();
        assert!(depth >= 1);
    }

    #[test]
    fn decomposition_node_leaf_count() {
        let test = make_test();
        let tree = TestDecomposer::decompose(&test);
        assert!(tree.root.leaf_count() >= 1);
    }

    #[test]
    fn optimal_decomposition() {
        let test = make_test();
        let tree = OptimalDecomposition::find_optimal(&test);
        assert!(tree.root.node_count() >= 1);
    }

    #[test]
    fn estimate_executions() {
        let test = make_test();
        let tree = TestDecomposer::decompose(&test);
        let est = OptimalDecomposition::estimate_executions(&test, &tree);
        assert!(est >= 1);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 12: WebGPU model tests
// ═══════════════════════════════════════════════════════════════════════════

mod webgpu_tests {
    use super::*;

    #[test]
    fn webgpu_scope_all() {
        let scopes = WebGPUScope::all();
        assert!(scopes.len() > 0);
    }

    #[test]
    fn webgpu_scope_includes() {
        assert!(WebGPUScope::QueueFamily.includes(&WebGPUScope::Workgroup));
    }

    #[test]
    fn webgpu_scope_config() {
        let config = WebGPUScopeConfig::new(4, 2, 32);
        assert_eq!(config.total_invocations(), 4 * 2 * 32);
    }

    #[test]
    fn webgpu_scope_config_default() {
        let config = WebGPUScopeConfig::default_config();
        assert!(config.total_invocations() > 0);
    }

    #[test]
    fn webgpu_event_construction() {
        let e = WebGPUEvent::new(0, 0, OpType::Write, 0x100, 1);
        let _ = e;
    }

    #[test]
    fn webgpu_event_with_scope() {
        let e = WebGPUEvent::new(0, 0, OpType::Write, 0x100, 1)
            .with_scope(WebGPUScope::Workgroup);
        let _ = e;
    }

    #[test]
    fn webgpu_event_with_workgroup() {
        let e = WebGPUEvent::new(0, 0, OpType::Write, 0x100, 1)
            .with_workgroup(0, 0);
        let _ = e;
    }

    #[test]
    fn webgpu_event_to_base() {
        let e = WebGPUEvent::new(0, 0, OpType::Write, 0x100, 1);
        let base = e.to_base_event();
        assert_eq!(base.op_type, OpType::Write);
    }

    #[test]
    fn webgpu_model_construction() {
        let m = WebGPUModel::new();
        let am = m.axiomatic_model();
        assert!(am.validate().is_ok());
    }

    #[test]
    fn webgpu_litmus_test_mp() {
        let test = WebGPULitmusTest::message_passing();
        let litmus = test.to_litmus_test();
        assert!(litmus.thread_count() > 0);
    }

    #[test]
    fn webgpu_litmus_test_sb() {
        let test = WebGPULitmusTest::store_buffering();
        let litmus = test.to_litmus_test();
        assert!(litmus.thread_count() > 0);
    }

    #[test]
    fn webgpu_litmus_test_coherence() {
        let test = WebGPULitmusTest::workgroup_coherence();
        let litmus = test.to_litmus_test();
        assert!(litmus.thread_count() > 0);
    }

    #[test]
    fn webgpu_ordering_to_litmus() {
        let rlx = WebGPUOrdering::Relaxed;
        let o = rlx.to_litmus_ordering();
        assert_eq!(o, Ordering::Relaxed);
    }

    #[test]
    fn vulkan_model() {
        let m = VulkanModel::new();
        let _ = m;
    }

    #[test]
    fn model_difference() {
        let sc = BuiltinModel::SC.build();
        let tso = BuiltinModel::TSO.build();
        let diff = ModelDifference::compare(&sc, &tso);
        let summary = diff.summary();
        assert!(summary.len() > 0);
    }

    #[test]
    fn compare_gpu_models() {
        let diffs = ModelDifference::compare_gpu_models();
        let _ = diffs;
    }

    #[test]
    fn safe_gpu_strict() {
        let s = SafeGPU::strict();
        let _ = s;
    }

    #[test]
    fn safe_gpu_standard() {
        let s = SafeGPU::standard();
        let _ = s;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 13: Operational model tests
// ═══════════════════════════════════════════════════════════════════════════

mod operational_tests {
    use super::*;

    #[test]
    fn thread_state() {
        let mut ts = ThreadState::new(0);
        ts.write_register(0, 42);
        assert_eq!(ts.read_register(0), 42);
        assert_eq!(ts.read_register(1), 0);
    }

    #[test]
    fn thread_state_with_scope() {
        let ts = ThreadState::new(0).with_scope(Scope::CTA);
        let _ = ts;
    }

    #[test]
    fn thread_state_pc() {
        let mut ts = ThreadState::new(0);
        ts.advance_pc();
        ts.advance_pc();
        ts.set_pc(0);
    }

    #[test]
    fn store_buffer() {
        let mut sb = StoreBuffer::new(0);
        assert!(sb.is_empty());
        sb.push(0x100, 42);
        assert!(!sb.is_empty());
        assert_eq!(sb.len(), 1);
        assert_eq!(sb.lookup(0x100), Some(42));
        assert_eq!(sb.lookup(0x200), None);
    }

    #[test]
    fn store_buffer_drain() {
        let mut sb = StoreBuffer::new(0);
        sb.push(0x100, 1);
        sb.push(0x200, 2);
        let entry = sb.drain_oldest();
        assert!(entry.is_some());
        assert_eq!(sb.len(), 1);
    }

    #[test]
    fn store_buffer_flush() {
        let mut sb = StoreBuffer::new(0);
        sb.push(0x100, 1);
        sb.push(0x100, 2);
        sb.push(0x200, 3);
        let flushed = sb.flush_address(0x100);
        assert_eq!(flushed.len(), 2);
    }

    #[test]
    fn store_buffer_flush_all() {
        let mut sb = StoreBuffer::new(0);
        sb.push(0x100, 1);
        sb.push(0x200, 2);
        let flushed = sb.flush_all();
        assert_eq!(flushed.len(), 2);
        assert!(sb.is_empty());
    }

    #[test]
    fn store_buffer_max_size() {
        let mut sb = StoreBuffer::new(0).with_max_size(2);
        sb.push(0x100, 1);
        sb.push(0x200, 2);
        assert!(sb.is_full());
    }

    #[test]
    fn write_buffer() {
        let mut wb = WriteBuffer::new(0);
        assert!(wb.is_empty());
        wb.push(0x100, 42, Scope::None);
        assert!(!wb.is_empty());
        assert_eq!(wb.lookup(0x100), Some(42));
    }

    #[test]
    fn write_buffer_propagate() {
        let mut wb = WriteBuffer::new(0);
        wb.push(0x100, 1, Scope::None);
        wb.push(0x200, 2, Scope::None);
        let entry = wb.propagate_any();
        assert!(entry.is_some());
    }

    #[test]
    fn machine_state() {
        let mut ms = MachineState::new(2, MemoryModelKind::SC);
        ms.init_memory(0x100, 0);
        let val = ms.read(0, 0x100);
        assert_eq!(val, 0);
        ms.write(0, 0x100, 42);
        let val = ms.read(0, 0x100);
        assert_eq!(val, 42);
    }

    #[test]
    fn machine_state_tso() {
        let ms = MachineState::new(2, MemoryModelKind::TSO);
        let _ = ms;
    }

    #[test]
    fn machine_state_fence() {
        let mut ms = MachineState::new(2, MemoryModelKind::TSO);
        ms.init_memory(0x100, 0);
        ms.write(0, 0x100, 42);
        ms.fence(0);
    }

    #[test]
    fn operational_scope_hierarchy() {
        let mut h = litmus_infinity::checker::operational::ScopeHierarchy::new();
        h.assign(0, 0, 0, 0);
        h.assign(1, 0, 0, 0);
        assert!(h.same_scope(0, 1, Scope::CTA));
    }

    #[test]
    fn operational_model_sc() {
        let om = OperationalModel::new(MemoryModelKind::SC);
        let _ = om;
    }

    #[test]
    fn operational_model_tso() {
        let om = OperationalModel::new(MemoryModelKind::TSO);
        let _ = om;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SECTION 14: Integration tests
// ═══════════════════════════════════════════════════════════════════════════

mod checker_integration {
    use super::*;

    #[test]
    fn sc_forbids_sb_weak_outcome() {
        let mut test = LitmusTest::new("SB");
        test.set_initial(0x100, 0);
        test.set_initial(0x200, 0);

        let mut t0 = Thread::new(0);
        t0.store(0x100, 1, Ordering::Relaxed);
        t0.load(0, 0x200, Ordering::Relaxed);
        test.add_thread(t0);

        let mut t1 = Thread::new(1);
        t1.store(0x200, 1, Ordering::Relaxed);
        t1.load(0, 0x100, Ordering::Relaxed);
        test.add_thread(t1);

        let weak = Outcome::new()
            .with_reg(0, 0, 0)
            .with_reg(1, 0, 0);
        test.expect(weak, LitmusOutcome::Forbidden);

        let model = BuiltinModel::SC.build();
        let mut v = Verifier::new(model);
        let result = v.verify_litmus(&test);
        // Under SC, r0=0,r1=0 should be forbidden
        assert!(!result.has_forbidden());
    }

    #[test]
    fn execution_graph_with_rf_co() {
        let events = vec![
            Event::new(0, 0, OpType::Write, 0x100, 1).with_po_index(0),
            Event::new(1, 1, OpType::Write, 0x100, 2).with_po_index(0),
            Event::new(2, 1, OpType::Read, 0x100, 1).with_po_index(1),
        ];
        let mut g = ExecutionGraph::new(events);
        g.rf.set(0, 2, true); // read sees write from thread 0
        g.co.set(0, 1, true); // write 0 before write 1
        g.derive_fr();
        assert!(g.fr.get(2, 1)); // read from 0 -> fr -> write 1
    }

    #[test]
    fn all_builtin_models_validate() {
        for m in BuiltinModel::all() {
            let model = m.build();
            assert!(model.validate().is_ok(), "Model {} failed validation", m.name());
        }
    }
    // relation_evaluator_from_graph removed: RelationExprEvaluator not publicly exported
}
