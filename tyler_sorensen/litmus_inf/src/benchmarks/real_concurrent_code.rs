//! Real concurrent code pattern extraction for litmus test generation.
//!
//! Extracts litmus tests from patterns found in real concurrent code:
//!   - Linux kernel synchronization primitives
//!   - Tokio async runtime patterns
//!   - Crossbeam concurrent data structures
//!   - Lock-free data structure patterns (Treiber stack, MS queue)
//!   - RCU-like patterns
//!   - LKMM (Linux Kernel Memory Model) patterns

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

use crate::checker::litmus::{
    LitmusTest, Thread, Instruction, Ordering, Outcome, Scope, LitmusOutcome,
};
use crate::checker::execution::{Address, Value};

// ---------------------------------------------------------------------------
// Pattern taxonomy
// ---------------------------------------------------------------------------

/// Source of the concurrent pattern.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternSource {
    LinuxKernel,
    Tokio,
    Crossbeam,
    StdLibrary,
    LockFreeAlgorithm,
    Academic,
    Custom(String),
}

/// A concurrent code pattern that can be converted to a litmus test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrentPattern {
    pub name: String,
    pub source: PatternSource,
    pub description: String,
    pub category: PatternCategory,
    pub test: Option<LitmusTest>,
    pub num_threads: usize,
    pub num_locations: usize,
    pub bug_class: Option<BugClass>,
}

/// Category of concurrent pattern.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatternCategory {
    LockFree,
    WaitFree,
    Blocking,
    RCU,
    SeqLock,
    SpinLock,
    PublishSubscribe,
    ProducerConsumer,
    DoubleCheckedLocking,
    Initialization,
    ReferenceCount,
    MemoryReclamation,
    SignalHandling,
}

/// Bug class that a litmus test targets.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BugClass {
    DataRace,
    UseAfterFree,
    TornRead,
    StaleRead,
    LostUpdate,
    ABAViolation,
    OrderingViolation,
    PublicationFailure,
    InitializationRace,
}

/// Lock-free data structure patterns.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LockFreePattern {
    TreiberStack,
    MichaelScottQueue,
    HarrisList,
    SkipList,
    ChaseLevDeque,
    MPMCQueue,
    SPSCQueue,
    MPSCQueue,
}

/// RCU pattern variants.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RcuPattern {
    ClassicRcu,
    SrCu,       // Sleepable RCU
    TreeRcu,    // Tree-based RCU
    RcuReplace, // RCU pointer replacement
    RcuDereference,
}

/// Linux kernel synchronization patterns.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KernelPattern {
    SmpMb,
    SmpWmb,
    SmpRmb,
    SmpStoreMbLoad,
    LockUnlock,
    SpinLockIrq,
    CompletionWait,
    WaitQueueWake,
    PerCpuAccess,
    AtomicBitOps,
}

// ---------------------------------------------------------------------------
// Extractor
// ---------------------------------------------------------------------------

/// Extracts concurrent patterns and converts them to litmus tests.
pub struct RealCodeExtractor {
    patterns: Vec<ConcurrentPattern>,
}

impl RealCodeExtractor {
    pub fn new() -> Self {
        Self { patterns: Vec::new() }
    }

    /// Generate all known patterns.
    pub fn extract_all(&mut self) -> &[ConcurrentPattern] {
        self.extract_linux_kernel_patterns();
        self.extract_lock_free_patterns();
        self.extract_rcu_patterns();
        self.extract_tokio_patterns();
        self.extract_crossbeam_patterns();
        self.extract_std_patterns();
        self.extract_classic_bug_patterns();
        &self.patterns
    }

    /// Get all extracted patterns.
    pub fn patterns(&self) -> &[ConcurrentPattern] {
        &self.patterns
    }

    /// Get all litmus tests from extracted patterns.
    pub fn litmus_tests(&self) -> Vec<&LitmusTest> {
        self.patterns.iter()
            .filter_map(|p| p.test.as_ref())
            .collect()
    }

    // -----------------------------------------------------------------------
    // Linux Kernel Memory Model (LKMM) patterns
    // -----------------------------------------------------------------------

    fn extract_linux_kernel_patterns(&mut self) {
        // smp_store_mb + smp_load_acquire — publish/subscribe
        self.patterns.push(self.make_publish_subscribe());
        
        // smp_wmb + smp_rmb — write barrier / read barrier pair
        self.patterns.push(self.make_wmb_rmb_pair());
        
        // WRITE_ONCE / READ_ONCE with smp_mb
        self.patterns.push(self.make_write_once_read_once());
        
        // spin_lock / spin_unlock pattern
        self.patterns.push(self.make_spin_lock_pattern());
        
        // Completion variable pattern
        self.patterns.push(self.make_completion_pattern());
        
        // Per-CPU access with preempt disable
        self.patterns.push(self.make_per_cpu_pattern());
        
        // Wait queue wake pattern
        self.patterns.push(self.make_waitqueue_pattern());
        
        // Atomic bitops (test_and_set, test_and_clear)
        self.patterns.push(self.make_atomic_bitops());
        
        // Memory barrier pairing (smp_mb__before_atomic / after)
        self.patterns.push(self.make_mb_before_after_atomic());
        
        // smp_store_release + smp_load_acquire MP pattern
        self.patterns.push(self.make_release_acquire_mp());
        
        // Linux rcu_assign_pointer / rcu_dereference
        self.patterns.push(self.make_rcu_pointer_pattern());
        
        // kfree_rcu pattern (delayed free after grace period)
        self.patterns.push(self.make_kfree_rcu_pattern());
    }

    fn make_publish_subscribe(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("LKMM-publish-subscribe");
        
        // Thread 0: writer (publisher)
        let mut t0 = Thread::new(0);
        t0.store(0x10, 42, Ordering::Relaxed);  // data = 42
        t0.store(0x18, 42, Ordering::Relaxed);  // data2 = 42
        t0.fence(Ordering::Release, Scope::None); // smp_wmb()
        t0.store(0x20, 1, Ordering::Release);   // flag = 1 (smp_store_release)
        test.add_thread(t0);
        
        // Thread 1: reader (subscriber)
        let mut t1 = Thread::new(1);
        t1.load(0, 0x20, Ordering::Acquire); // smp_load_acquire(&flag)
        t1.load(1, 0x10, Ordering::Relaxed); // data
        t1.load(2, 0x18, Ordering::Relaxed); // data2
        test.add_thread(t1);
        
        // Thread 2: another reader
        let mut t2 = Thread::new(2);
        t2.load(0, 0x20, Ordering::Acquire);
        t2.load(1, 0x10, Ordering::Relaxed);
        test.add_thread(t2);
        
        // Thread 3: concurrent writer
        let mut t3 = Thread::new(3);
        t3.store(0x10, 99, Ordering::Relaxed);
        t3.fence(Ordering::Release, Scope::None);
        t3.store(0x20, 2, Ordering::Release);
        test.add_thread(t3);
        
        // Forbidden: see flag=1 but data=0
        let outcome = Outcome::new()
            .with_reg(1, 0, 1)
            .with_reg(1, 1, 0);
        test.expect(outcome, LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "LKMM-publish-subscribe".into(),
            source: PatternSource::LinuxKernel,
            description: "Linux kernel publish/subscribe via smp_store_release + smp_load_acquire".into(),
            category: PatternCategory::PublishSubscribe,
            test: Some(test),
            num_threads: 4,
            num_locations: 3,
            bug_class: Some(BugClass::PublicationFailure),
        }
    }

    fn make_wmb_rmb_pair(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("LKMM-wmb-rmb");
        
        // Thread 0: two writes with wmb
        let mut t0 = Thread::new(0);
        t0.store(0x10, 1, Ordering::Relaxed); // WRITE_ONCE(data, 1)
        t0.fence(Ordering::Release, Scope::None); // smp_wmb()
        t0.store(0x20, 1, Ordering::Relaxed); // WRITE_ONCE(flag, 1)
        test.add_thread(t0);
        
        // Thread 1: two reads with rmb
        let mut t1 = Thread::new(1);
        t1.load(0, 0x20, Ordering::Relaxed);  // READ_ONCE(flag)
        t1.fence(Ordering::Acquire, Scope::None); // smp_rmb()
        t1.load(1, 0x10, Ordering::Relaxed);  // READ_ONCE(data)
        test.add_thread(t1);
        
        // Thread 2: observer
        let mut t2 = Thread::new(2);
        t2.load(0, 0x20, Ordering::Relaxed);
        t2.load(1, 0x10, Ordering::Relaxed);
        test.add_thread(t2);
        
        // Forbidden: T1 sees flag=1, data=0
        let outcome = Outcome::new()
            .with_reg(1, 0, 1)
            .with_reg(1, 1, 0);
        test.expect(outcome, LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "LKMM-wmb-rmb".into(),
            source: PatternSource::LinuxKernel,
            description: "Write memory barrier + read memory barrier pair (smp_wmb/smp_rmb)".into(),
            category: PatternCategory::PublishSubscribe,
            test: Some(test),
            num_threads: 3,
            num_locations: 2,
            bug_class: Some(BugClass::OrderingViolation),
        }
    }

    fn make_write_once_read_once(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("LKMM-WRITE_ONCE-READ_ONCE");
        
        // Multiple writers and readers using WRITE_ONCE/READ_ONCE with full barrier
        let mut t0 = Thread::new(0);
        t0.store(0x10, 1, Ordering::Relaxed); // WRITE_ONCE(x, 1)
        t0.fence(Ordering::SeqCst, Scope::None); // smp_mb()
        t0.store(0x20, 1, Ordering::Relaxed); // WRITE_ONCE(y, 1)
        test.add_thread(t0);
        
        let mut t1 = Thread::new(1);
        t1.load(0, 0x20, Ordering::Relaxed); // READ_ONCE(y)
        t1.fence(Ordering::SeqCst, Scope::None); // smp_mb()
        t1.load(1, 0x10, Ordering::Relaxed); // READ_ONCE(x)
        test.add_thread(t1);
        
        let mut t2 = Thread::new(2);
        t2.store(0x20, 2, Ordering::Relaxed);
        t2.fence(Ordering::SeqCst, Scope::None);
        t2.store(0x10, 2, Ordering::Relaxed);
        test.add_thread(t2);
        
        let mut t3 = Thread::new(3);
        t3.load(0, 0x10, Ordering::Relaxed);
        t3.fence(Ordering::SeqCst, Scope::None);
        t3.load(1, 0x20, Ordering::Relaxed);
        test.add_thread(t3);
        
        // Forbidden: both barriers ineffective
        let outcome = Outcome::new()
            .with_reg(1, 0, 1)
            .with_reg(1, 1, 0)
            .with_reg(3, 0, 2)
            .with_reg(3, 1, 0);
        test.expect(outcome, LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "LKMM-WRITE_ONCE-READ_ONCE".into(),
            source: PatternSource::LinuxKernel,
            description: "WRITE_ONCE/READ_ONCE with smp_mb() full barrier pattern".into(),
            category: PatternCategory::PublishSubscribe,
            test: Some(test),
            num_threads: 4,
            num_locations: 2,
            bug_class: Some(BugClass::OrderingViolation),
        }
    }

    fn make_spin_lock_pattern(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("LKMM-spinlock");
        
        let lock_addr = 0x00u64;
        let data_addr = 0x10u64;
        let data2_addr = 0x18u64;
        
        for t in 0..4 {
            let mut thread = Thread::new(t);
            // spin_lock: CAS(lock, 0, 1) with acquire
            thread.rmw(0, lock_addr, 1, Ordering::Acquire);
            // Critical section
            thread.store(data_addr, (t + 1) as Value, Ordering::Relaxed);
            thread.load(1, data_addr, Ordering::Relaxed);
            thread.store(data2_addr, (t + 1) as Value, Ordering::Relaxed);
            thread.load(2, data2_addr, Ordering::Relaxed);
            // spin_unlock: store(lock, 0) with release
            thread.store(lock_addr, 0, Ordering::Release);
            test.add_thread(thread);
        }
        
        test.expect(Outcome::new(), LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "LKMM-spinlock".into(),
            source: PatternSource::LinuxKernel,
            description: "Spin lock with acquire/release semantics, 4 contending threads".into(),
            category: PatternCategory::SpinLock,
            test: Some(test),
            num_threads: 4,
            num_locations: 3,
            bug_class: Some(BugClass::DataRace),
        }
    }

    fn make_completion_pattern(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("LKMM-completion");
        
        // Thread 0: does work, then complete()
        let mut t0 = Thread::new(0);
        t0.store(0x10, 1, Ordering::Relaxed); // work result
        t0.store(0x18, 1, Ordering::Relaxed); // more work
        t0.fence(Ordering::Release, Scope::None);
        t0.store(0x20, 1, Ordering::Release); // completion.done = 1
        test.add_thread(t0);
        
        // Threads 1-3: wait_for_completion()
        for t in 1..4 {
            let mut thread = Thread::new(t);
            thread.load(0, 0x20, Ordering::Acquire); // wait on done
            thread.load(1, 0x10, Ordering::Relaxed); // read result
            thread.load(2, 0x18, Ordering::Relaxed); // read result2
            test.add_thread(thread);
        }
        
        let outcome = Outcome::new()
            .with_reg(1, 0, 1)
            .with_reg(1, 1, 0);
        test.expect(outcome, LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "LKMM-completion".into(),
            source: PatternSource::LinuxKernel,
            description: "Completion variable: one thread completes, others wait".into(),
            category: PatternCategory::PublishSubscribe,
            test: Some(test),
            num_threads: 4,
            num_locations: 3,
            bug_class: Some(BugClass::OrderingViolation),
        }
    }

    fn make_per_cpu_pattern(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("LKMM-per-cpu");
        
        // Per-CPU variable access with preempt_disable/enable acting as compiler barriers
        for t in 0..4 {
            let mut thread = Thread::new(t);
            let per_cpu_loc = (t * 0x40) as u64; // per-CPU variable (different cache lines)
            let shared_loc = 0x100u64;
            
            // preempt_disable (compiler barrier)
            thread.fence(Ordering::AcqRel, Scope::None);
            // Access per-CPU variable
            thread.load(0, per_cpu_loc, Ordering::Relaxed);
            thread.store(per_cpu_loc, (t + 1) as Value, Ordering::Relaxed);
            // Also touch shared
            thread.load(1, shared_loc, Ordering::Acquire);
            // preempt_enable (compiler barrier)
            thread.fence(Ordering::AcqRel, Scope::None);
            
            test.add_thread(thread);
        }
        
        test.expect(Outcome::new(), LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "LKMM-per-cpu".into(),
            source: PatternSource::LinuxKernel,
            description: "Per-CPU variable access pattern with preempt_disable barriers".into(),
            category: PatternCategory::Initialization,
            test: Some(test),
            num_threads: 4,
            num_locations: 5,
            bug_class: Some(BugClass::DataRace),
        }
    }

    fn make_waitqueue_pattern(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("LKMM-waitqueue-wake");
        
        let condition = 0x10u64;
        let wq_lock = 0x20u64;
        let data = 0x30u64;
        
        // Thread 0: sleeper (wait_event)
        let mut t0 = Thread::new(0);
        t0.rmw(0, wq_lock, 1, Ordering::Acquire); // lock waitqueue
        t0.load(1, condition, Ordering::Acquire);   // check condition
        t0.store(wq_lock, 0, Ordering::Release);    // unlock
        // After wakeup:
        t0.load(2, data, Ordering::Acquire);
        test.add_thread(t0);
        
        // Thread 1: waker (wake_up)
        let mut t1 = Thread::new(1);
        t1.store(data, 42, Ordering::Release);
        t1.fence(Ordering::SeqCst, Scope::None); // smp_mb()
        t1.store(condition, 1, Ordering::Release); // set condition
        test.add_thread(t1);
        
        // Thread 2: another sleeper
        let mut t2 = Thread::new(2);
        t2.rmw(0, wq_lock, 1, Ordering::Acquire);
        t2.load(1, condition, Ordering::Acquire);
        t2.store(wq_lock, 0, Ordering::Release);
        t2.load(2, data, Ordering::Acquire);
        test.add_thread(t2);
        
        // Thread 3: another waker
        let mut t3 = Thread::new(3);
        t3.store(data, 99, Ordering::Release);
        t3.fence(Ordering::SeqCst, Scope::None);
        t3.store(condition, 2, Ordering::Release);
        test.add_thread(t3);
        
        let outcome = Outcome::new()
            .with_reg(0, 1, 1)
            .with_reg(0, 2, 0);
        test.expect(outcome, LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "LKMM-waitqueue-wake".into(),
            source: PatternSource::LinuxKernel,
            description: "Wait queue + wake_up pattern with condition variable".into(),
            category: PatternCategory::ProducerConsumer,
            test: Some(test),
            num_threads: 4,
            num_locations: 3,
            bug_class: Some(BugClass::OrderingViolation),
        }
    }

    fn make_atomic_bitops(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("LKMM-atomic-bitops");
        
        let flags = 0x10u64;
        let data = 0x20u64;
        
        // Thread 0: test_and_set_bit
        let mut t0 = Thread::new(0);
        t0.rmw(0, flags, 1, Ordering::AcqRel); // test_and_set_bit(0, &flags)
        t0.store(data, 1, Ordering::Relaxed);
        test.add_thread(t0);
        
        // Thread 1: test_and_clear_bit
        let mut t1 = Thread::new(1);
        t1.rmw(0, flags, 0, Ordering::AcqRel); // test_and_clear_bit(0, &flags)
        t1.load(1, data, Ordering::Relaxed);
        test.add_thread(t1);
        
        // Thread 2: set_bit + test_bit
        let mut t2 = Thread::new(2);
        t2.rmw(0, flags, 2, Ordering::AcqRel);
        t2.load(1, data, Ordering::Relaxed);
        test.add_thread(t2);
        
        // Thread 3: observer
        let mut t3 = Thread::new(3);
        t3.load(0, flags, Ordering::Acquire);
        t3.load(1, data, Ordering::Relaxed);
        test.add_thread(t3);
        
        test.expect(Outcome::new(), LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "LKMM-atomic-bitops".into(),
            source: PatternSource::LinuxKernel,
            description: "Atomic bit operations (test_and_set_bit, test_and_clear_bit)".into(),
            category: PatternCategory::LockFree,
            test: Some(test),
            num_threads: 4,
            num_locations: 2,
            bug_class: Some(BugClass::LostUpdate),
        }
    }

    fn make_mb_before_after_atomic(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("LKMM-mb-before-after-atomic");
        
        let counter = 0x10u64;
        let data = 0x20u64;
        let flag = 0x30u64;
        
        // Thread 0: smp_mb__before_atomic + atomic_inc
        let mut t0 = Thread::new(0);
        t0.store(data, 1, Ordering::Relaxed);
        t0.fence(Ordering::SeqCst, Scope::None); // smp_mb__before_atomic()
        t0.rmw(0, counter, 1, Ordering::Relaxed); // atomic_inc
        test.add_thread(t0);
        
        // Thread 1: atomic_inc + smp_mb__after_atomic
        let mut t1 = Thread::new(1);
        t1.rmw(0, counter, 1, Ordering::Relaxed); // atomic_inc
        t1.fence(Ordering::SeqCst, Scope::None); // smp_mb__after_atomic()
        t1.load(1, data, Ordering::Relaxed);
        test.add_thread(t1);
        
        // Thread 2: another participant
        let mut t2 = Thread::new(2);
        t2.store(flag, 1, Ordering::Relaxed);
        t2.fence(Ordering::SeqCst, Scope::None);
        t2.rmw(0, counter, 1, Ordering::Relaxed);
        test.add_thread(t2);
        
        // Thread 3: observer
        let mut t3 = Thread::new(3);
        t3.load(0, counter, Ordering::Acquire);
        t3.load(1, data, Ordering::Relaxed);
        t3.load(2, flag, Ordering::Relaxed);
        test.add_thread(t3);
        
        test.expect(Outcome::new(), LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "LKMM-mb-before-after-atomic".into(),
            source: PatternSource::LinuxKernel,
            description: "smp_mb__before_atomic / smp_mb__after_atomic barrier pairing".into(),
            category: PatternCategory::LockFree,
            test: Some(test),
            num_threads: 4,
            num_locations: 3,
            bug_class: Some(BugClass::OrderingViolation),
        }
    }

    fn make_release_acquire_mp(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("LKMM-release-acquire-MP");
        
        // 6-thread message passing chain with release/acquire
        let mut t0 = Thread::new(0);
        t0.store(0x10, 1, Ordering::Relaxed);
        t0.store(0x18, 1, Ordering::Relaxed);
        t0.store(0x20, 1, Ordering::Release); // flag
        test.add_thread(t0);
        
        let mut t1 = Thread::new(1);
        t1.load(0, 0x20, Ordering::Acquire);
        t1.store(0x28, 1, Ordering::Release);
        test.add_thread(t1);
        
        let mut t2 = Thread::new(2);
        t2.load(0, 0x28, Ordering::Acquire);
        t2.store(0x30, 1, Ordering::Release);
        test.add_thread(t2);
        
        let mut t3 = Thread::new(3);
        t3.load(0, 0x30, Ordering::Acquire);
        t3.store(0x38, 1, Ordering::Release);
        test.add_thread(t3);
        
        let mut t4 = Thread::new(4);
        t4.load(0, 0x38, Ordering::Acquire);
        t4.store(0x40, 1, Ordering::Release);
        test.add_thread(t4);
        
        let mut t5 = Thread::new(5);
        t5.load(0, 0x40, Ordering::Acquire);
        t5.load(1, 0x10, Ordering::Relaxed);
        t5.load(2, 0x18, Ordering::Relaxed);
        test.add_thread(t5);
        
        let outcome = Outcome::new()
            .with_reg(5, 0, 1)
            .with_reg(5, 1, 0);
        test.expect(outcome, LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "LKMM-release-acquire-MP".into(),
            source: PatternSource::LinuxKernel,
            description: "6-thread release/acquire message passing chain".into(),
            category: PatternCategory::PublishSubscribe,
            test: Some(test),
            num_threads: 6,
            num_locations: 7,
            bug_class: Some(BugClass::OrderingViolation),
        }
    }

    fn make_rcu_pointer_pattern(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("LKMM-rcu-pointer");
        
        let ptr_loc = 0x10u64;
        let old_data = 0x20u64;
        let new_data = 0x30u64;
        
        // Thread 0: updater - rcu_assign_pointer
        let mut t0 = Thread::new(0);
        t0.store(new_data, 42, Ordering::Relaxed); // init new object
        t0.fence(Ordering::Release, Scope::None);
        t0.store(ptr_loc, new_data, Ordering::Release); // rcu_assign_pointer
        test.add_thread(t0);
        
        // Thread 1: reader - rcu_dereference
        let mut t1 = Thread::new(1);
        t1.load(0, ptr_loc, Ordering::Acquire); // p = rcu_dereference(ptr)
        t1.load(1, new_data, Ordering::Relaxed); // *p (simplified)
        test.add_thread(t1);
        
        // Thread 2: another reader
        let mut t2 = Thread::new(2);
        t2.load(0, ptr_loc, Ordering::Acquire);
        t2.load(1, new_data, Ordering::Relaxed);
        test.add_thread(t2);
        
        // Thread 3: reclaimer
        let mut t3 = Thread::new(3);
        t3.load(0, ptr_loc, Ordering::Acquire);
        t3.fence(Ordering::SeqCst, Scope::None); // synchronize_rcu()
        t3.store(old_data, 0, Ordering::Relaxed); // kfree(old)
        test.add_thread(t3);
        
        let outcome = Outcome::new()
            .with_reg(1, 0, new_data)
            .with_reg(1, 1, 0);
        test.expect(outcome, LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "LKMM-rcu-pointer".into(),
            source: PatternSource::LinuxKernel,
            description: "RCU pointer publish: rcu_assign_pointer + rcu_dereference".into(),
            category: PatternCategory::RCU,
            test: Some(test),
            num_threads: 4,
            num_locations: 3,
            bug_class: Some(BugClass::UseAfterFree),
        }
    }

    fn make_kfree_rcu_pattern(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("LKMM-kfree-rcu");
        
        let ptr = 0x10u64;
        let obj_data = 0x20u64;
        let new_obj = 0x30u64;
        let grace_period_done = 0x40u64;
        
        // Thread 0: updater
        let mut t0 = Thread::new(0);
        t0.store(new_obj, 99, Ordering::Relaxed);
        t0.fence(Ordering::Release, Scope::None);
        t0.store(ptr, new_obj, Ordering::Release);
        test.add_thread(t0);
        
        // Thread 1: RCU reader (in read-side critical section)
        let mut t1 = Thread::new(1);
        t1.load(0, ptr, Ordering::Acquire); // rcu_read_lock + rcu_dereference
        t1.load(1, obj_data, Ordering::Relaxed); // use old object
        test.add_thread(t1);
        
        // Thread 2: grace period (synchronize_rcu)
        let mut t2 = Thread::new(2);
        t2.fence(Ordering::SeqCst, Scope::None);
        t2.store(grace_period_done, 1, Ordering::Release);
        test.add_thread(t2);
        
        // Thread 3: reclaimer (kfree_rcu callback)
        let mut t3 = Thread::new(3);
        t3.load(0, grace_period_done, Ordering::Acquire);
        t3.store(obj_data, 0xDEAD, Ordering::Relaxed); // free (poison)
        test.add_thread(t3);
        
        // Forbidden: reader sees freed data
        let outcome = Outcome::new()
            .with_reg(1, 1, 0xDEAD);
        test.expect(outcome, LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "LKMM-kfree-rcu".into(),
            source: PatternSource::LinuxKernel,
            description: "kfree_rcu pattern: delayed free after RCU grace period".into(),
            category: PatternCategory::MemoryReclamation,
            test: Some(test),
            num_threads: 4,
            num_locations: 4,
            bug_class: Some(BugClass::UseAfterFree),
        }
    }

    // -----------------------------------------------------------------------
    // Lock-free data structure patterns
    // -----------------------------------------------------------------------

    fn extract_lock_free_patterns(&mut self) {
        self.patterns.push(self.make_treiber_stack_full());
        self.patterns.push(self.make_michael_scott_queue_full());
        self.patterns.push(self.make_harris_list());
        self.patterns.push(self.make_chase_lev_deque());
        self.patterns.push(self.make_spsc_queue());
        self.patterns.push(self.make_mpmc_bounded_queue());
    }

    fn make_treiber_stack_full(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("Treiber-stack-full");
        
        let head = 0x00u64;
        let node0_data = 0x10u64;
        let node0_next = 0x18u64;
        let node1_data = 0x20u64;
        let node1_next = 0x28u64;
        
        // Thread 0: push node0
        let mut t0 = Thread::new(0);
        t0.store(node0_data, 42, Ordering::Relaxed);
        t0.load(0, head, Ordering::Relaxed); // old_head = head
        t0.store(node0_next, 0, Ordering::Relaxed); // node0.next = old_head
        t0.rmw(1, head, node0_data, Ordering::AcqRel); // CAS(head, old_head, node0)
        test.add_thread(t0);
        
        // Thread 1: push node1
        let mut t1 = Thread::new(1);
        t1.store(node1_data, 99, Ordering::Relaxed);
        t1.load(0, head, Ordering::Relaxed);
        t1.store(node1_next, 0, Ordering::Relaxed);
        t1.rmw(1, head, node1_data, Ordering::AcqRel);
        test.add_thread(t1);
        
        // Thread 2: pop
        let mut t2 = Thread::new(2);
        t2.load(0, head, Ordering::Acquire);
        t2.load(1, node0_data, Ordering::Relaxed);
        t2.load(2, node0_next, Ordering::Relaxed);
        t2.rmw(3, head, 0, Ordering::AcqRel);
        test.add_thread(t2);
        
        // Thread 3: pop
        let mut t3 = Thread::new(3);
        t3.load(0, head, Ordering::Acquire);
        t3.load(1, node1_data, Ordering::Relaxed);
        t3.rmw(2, head, 0, Ordering::AcqRel);
        test.add_thread(t3);
        
        test.expect(Outcome::new(), LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "Treiber-stack-full".into(),
            source: PatternSource::LockFreeAlgorithm,
            description: "Full Treiber stack with 2 pushers and 2 poppers".into(),
            category: PatternCategory::LockFree,
            test: Some(test),
            num_threads: 4,
            num_locations: 5,
            bug_class: Some(BugClass::ABAViolation),
        }
    }

    fn make_michael_scott_queue_full(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("MS-queue-full");
        
        let head = 0x00u64;
        let tail = 0x08u64;
        let node0_val = 0x10u64;
        let node0_next = 0x18u64;
        let node1_val = 0x20u64;
        let node1_next = 0x28u64;
        
        // Thread 0: enqueue val=42
        let mut t0 = Thread::new(0);
        t0.store(node0_val, 42, Ordering::Relaxed);
        t0.store(node0_next, 0, Ordering::Relaxed);
        t0.load(0, tail, Ordering::Acquire); // read tail
        // CAS tail->next = new_node
        t0.rmw(1, node0_next, node0_val, Ordering::AcqRel);
        // CAS tail = new_node
        t0.rmw(2, tail, node0_val, Ordering::AcqRel);
        test.add_thread(t0);
        
        // Thread 1: enqueue val=99
        let mut t1 = Thread::new(1);
        t1.store(node1_val, 99, Ordering::Relaxed);
        t1.store(node1_next, 0, Ordering::Relaxed);
        t1.load(0, tail, Ordering::Acquire);
        t1.rmw(1, node1_next, node1_val, Ordering::AcqRel);
        t1.rmw(2, tail, node1_val, Ordering::AcqRel);
        test.add_thread(t1);
        
        // Thread 2: dequeue
        let mut t2 = Thread::new(2);
        t2.load(0, head, Ordering::Acquire);
        t2.load(1, node0_val, Ordering::Relaxed);
        t2.load(2, node0_next, Ordering::Acquire);
        t2.rmw(3, head, 0, Ordering::AcqRel);
        test.add_thread(t2);
        
        // Thread 3: dequeue
        let mut t3 = Thread::new(3);
        t3.load(0, head, Ordering::Acquire);
        t3.load(1, node1_val, Ordering::Relaxed);
        t3.rmw(2, head, 0, Ordering::AcqRel);
        test.add_thread(t3);
        
        test.expect(Outcome::new(), LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "MS-queue-full".into(),
            source: PatternSource::LockFreeAlgorithm,
            description: "Michael-Scott queue: 2 enqueuers + 2 dequeuers".into(),
            category: PatternCategory::LockFree,
            test: Some(test),
            num_threads: 4,
            num_locations: 6,
            bug_class: Some(BugClass::ABAViolation),
        }
    }

    fn make_harris_list(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("Harris-linked-list");
        
        let head = 0x00u64;
        let node_a = 0x10u64;
        let node_b = 0x20u64;
        let node_a_next = 0x18u64;
        let node_b_next = 0x28u64;
        
        // Thread 0: insert node_a
        let mut t0 = Thread::new(0);
        t0.store(node_a, 10, Ordering::Relaxed); // key
        t0.load(0, head, Ordering::Acquire);
        t0.store(node_a_next, 0, Ordering::Relaxed);
        t0.rmw(1, head, node_a, Ordering::AcqRel); // CAS head
        test.add_thread(t0);
        
        // Thread 1: insert node_b
        let mut t1 = Thread::new(1);
        t1.store(node_b, 20, Ordering::Relaxed);
        t1.load(0, head, Ordering::Acquire);
        t1.store(node_b_next, 0, Ordering::Relaxed);
        t1.rmw(1, head, node_b, Ordering::AcqRel);
        test.add_thread(t1);
        
        // Thread 2: delete (mark + unlink)
        let mut t2 = Thread::new(2);
        t2.load(0, head, Ordering::Acquire);
        t2.load(1, node_a, Ordering::Relaxed);
        // Mark: set low bit of next pointer
        t2.rmw(2, node_a_next, 1, Ordering::AcqRel);
        // Unlink: CAS prev.next
        t2.rmw(3, head, 0, Ordering::AcqRel);
        test.add_thread(t2);
        
        // Thread 3: find
        let mut t3 = Thread::new(3);
        t3.load(0, head, Ordering::Acquire);
        t3.load(1, node_a, Ordering::Relaxed);
        t3.load(2, node_a_next, Ordering::Acquire);
        t3.load(3, node_b, Ordering::Relaxed);
        test.add_thread(t3);
        
        test.expect(Outcome::new(), LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "Harris-linked-list".into(),
            source: PatternSource::LockFreeAlgorithm,
            description: "Harris lock-free linked list: insert, delete, find concurrent".into(),
            category: PatternCategory::LockFree,
            test: Some(test),
            num_threads: 4,
            num_locations: 5,
            bug_class: Some(BugClass::ABAViolation),
        }
    }

    fn make_chase_lev_deque(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("Chase-Lev-deque");
        
        let top = 0x00u64;
        let bottom = 0x08u64;
        let buf0 = 0x10u64;
        let buf1 = 0x18u64;
        let buf2 = 0x20u64;
        
        // Thread 0: owner (push + pop from bottom)
        let mut t0 = Thread::new(0);
        // push
        t0.load(0, bottom, Ordering::Relaxed);
        t0.store(buf0, 42, Ordering::Relaxed);
        t0.fence(Ordering::Release, Scope::None);
        t0.rmw(1, bottom, 1, Ordering::Relaxed); // bottom++
        // push another
        t0.store(buf1, 99, Ordering::Relaxed);
        t0.rmw(2, bottom, 1, Ordering::Relaxed);
        test.add_thread(t0);
        
        // Thread 1: stealer (steal from top)
        let mut t1 = Thread::new(1);
        t1.load(0, top, Ordering::Acquire);
        t1.fence(Ordering::SeqCst, Scope::None);
        t1.load(1, bottom, Ordering::Acquire);
        t1.load(2, buf0, Ordering::Relaxed); // read element
        t1.rmw(3, top, 1, Ordering::SeqCst); // CAS top++
        test.add_thread(t1);
        
        // Thread 2: another stealer
        let mut t2 = Thread::new(2);
        t2.load(0, top, Ordering::Acquire);
        t2.fence(Ordering::SeqCst, Scope::None);
        t2.load(1, bottom, Ordering::Acquire);
        t2.load(2, buf1, Ordering::Relaxed);
        t2.rmw(3, top, 1, Ordering::SeqCst);
        test.add_thread(t2);
        
        // Thread 3: owner pop
        let mut t3 = Thread::new(3);
        t3.load(0, bottom, Ordering::Relaxed);
        t3.rmw(1, bottom, 0, Ordering::Relaxed); // bottom--
        t3.fence(Ordering::SeqCst, Scope::None);
        t3.load(2, top, Ordering::SeqCst);
        t3.load(3, buf2, Ordering::Relaxed);
        test.add_thread(t3);
        
        test.expect(Outcome::new(), LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "Chase-Lev-deque".into(),
            source: PatternSource::LockFreeAlgorithm,
            description: "Chase-Lev work-stealing deque: 1 owner + 2 stealers".into(),
            category: PatternCategory::LockFree,
            test: Some(test),
            num_threads: 4,
            num_locations: 5,
            bug_class: Some(BugClass::DataRace),
        }
    }

    fn make_spsc_queue(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("SPSC-ring-buffer");
        
        let head = 0x00u64; // producer writes, consumer reads
        let tail = 0x08u64; // consumer writes, producer reads
        let buf0 = 0x10u64;
        let buf1 = 0x18u64;
        let buf2 = 0x20u64;
        
        // Thread 0: producer
        let mut t0 = Thread::new(0);
        t0.load(0, tail, Ordering::Acquire); // check space
        t0.store(buf0, 1, Ordering::Relaxed);
        t0.store(buf1, 2, Ordering::Relaxed);
        t0.fence(Ordering::Release, Scope::None);
        t0.rmw(1, head, 2, Ordering::Release); // head += 2
        test.add_thread(t0);
        
        // Thread 1: consumer
        let mut t1 = Thread::new(1);
        t1.load(0, head, Ordering::Acquire);
        t1.load(1, buf0, Ordering::Relaxed);
        t1.load(2, buf1, Ordering::Relaxed);
        t1.rmw(3, tail, 2, Ordering::Release); // tail += 2
        test.add_thread(t1);
        
        // Forbidden: consumer sees head advanced but data not written
        let outcome = Outcome::new()
            .with_reg(1, 0, 2) // head = 2 (2 items)
            .with_reg(1, 1, 0); // but buf[0] = 0
        test.expect(outcome, LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "SPSC-ring-buffer".into(),
            source: PatternSource::LockFreeAlgorithm,
            description: "Single-producer single-consumer ring buffer".into(),
            category: PatternCategory::ProducerConsumer,
            test: Some(test),
            num_threads: 2,
            num_locations: 5,
            bug_class: Some(BugClass::StaleRead),
        }
    }

    fn make_mpmc_bounded_queue(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("MPMC-bounded-queue");
        
        let head = 0x00u64;
        let tail = 0x08u64;
        let slot0_seq = 0x10u64;
        let slot0_data = 0x18u64;
        let slot1_seq = 0x20u64;
        let slot1_data = 0x28u64;
        
        // Thread 0: producer 0
        let mut t0 = Thread::new(0);
        t0.load(0, tail, Ordering::Relaxed);
        t0.load(1, slot0_seq, Ordering::Acquire);
        t0.store(slot0_data, 42, Ordering::Relaxed);
        t0.store(slot0_seq, 1, Ordering::Release);
        t0.rmw(2, tail, 1, Ordering::Relaxed);
        test.add_thread(t0);
        
        // Thread 1: producer 1
        let mut t1 = Thread::new(1);
        t1.load(0, tail, Ordering::Relaxed);
        t1.load(1, slot1_seq, Ordering::Acquire);
        t1.store(slot1_data, 99, Ordering::Relaxed);
        t1.store(slot1_seq, 1, Ordering::Release);
        t1.rmw(2, tail, 1, Ordering::Relaxed);
        test.add_thread(t1);
        
        // Thread 2: consumer 0
        let mut t2 = Thread::new(2);
        t2.load(0, head, Ordering::Relaxed);
        t2.load(1, slot0_seq, Ordering::Acquire);
        t2.load(2, slot0_data, Ordering::Relaxed);
        t2.store(slot0_seq, 2, Ordering::Release);
        t2.rmw(3, head, 1, Ordering::Relaxed);
        test.add_thread(t2);
        
        // Thread 3: consumer 1
        let mut t3 = Thread::new(3);
        t3.load(0, head, Ordering::Relaxed);
        t3.load(1, slot1_seq, Ordering::Acquire);
        t3.load(2, slot1_data, Ordering::Relaxed);
        t3.store(slot1_seq, 2, Ordering::Release);
        t3.rmw(3, head, 1, Ordering::Relaxed);
        test.add_thread(t3);
        
        test.expect(Outcome::new(), LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "MPMC-bounded-queue".into(),
            source: PatternSource::LockFreeAlgorithm,
            description: "Bounded MPMC queue with sequence numbers (Dmitry Vyukov style)".into(),
            category: PatternCategory::ProducerConsumer,
            test: Some(test),
            num_threads: 4,
            num_locations: 6,
            bug_class: Some(BugClass::LostUpdate),
        }
    }

    // -----------------------------------------------------------------------
    // RCU patterns
    // -----------------------------------------------------------------------

    fn extract_rcu_patterns(&mut self) {
        self.patterns.push(self.make_classic_rcu());
        self.patterns.push(self.make_srcu_pattern());
        self.patterns.push(self.make_rcu_list_traversal());
    }

    fn make_classic_rcu(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("RCU-classic");
        
        let gp_counter = 0x00u64;
        let ptr = 0x10u64;
        let old_obj = 0x20u64;
        let new_obj = 0x30u64;
        
        // Thread 0: updater
        let mut t0 = Thread::new(0);
        t0.store(new_obj, 42, Ordering::Relaxed);
        t0.fence(Ordering::Release, Scope::None);
        t0.store(ptr, new_obj, Ordering::Release); // rcu_assign_pointer
        // synchronize_rcu (simplified)
        t0.rmw(0, gp_counter, 1, Ordering::SeqCst);
        t0.fence(Ordering::SeqCst, Scope::None);
        t0.store(old_obj, 0, Ordering::Relaxed); // kfree(old)
        test.add_thread(t0);
        
        // Threads 1-3: readers in RCU read-side critical sections
        for t in 1..4 {
            let mut thread = Thread::new(t);
            // rcu_read_lock (just a preempt_disable, no memory barrier)
            thread.load(0, gp_counter, Ordering::Relaxed);
            // p = rcu_dereference(ptr)
            thread.load(1, ptr, Ordering::Acquire);
            // read *p
            thread.load(2, new_obj, Ordering::Relaxed);
            thread.load(3, old_obj, Ordering::Relaxed);
            // rcu_read_unlock
            test.add_thread(thread);
        }
        
        // Forbidden: reader sees freed old object
        let outcome = Outcome::new()
            .with_reg(1, 3, 0); // old_obj freed
        test.expect(outcome, LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "RCU-classic".into(),
            source: PatternSource::LinuxKernel,
            description: "Classic RCU: 1 updater + 3 readers with grace period".into(),
            category: PatternCategory::RCU,
            test: Some(test),
            num_threads: 4,
            num_locations: 4,
            bug_class: Some(BugClass::UseAfterFree),
        }
    }

    fn make_srcu_pattern(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("SRCU-sleepable");
        
        let srcu_lock_count = 0x00u64;
        let srcu_unlock_count = 0x08u64;
        let ptr = 0x10u64;
        let data = 0x20u64;
        
        // Thread 0: SRCU reader
        let mut t0 = Thread::new(0);
        t0.rmw(0, srcu_lock_count, 1, Ordering::Acquire); // srcu_read_lock
        t0.load(1, ptr, Ordering::Acquire);
        t0.load(2, data, Ordering::Relaxed);
        t0.rmw(3, srcu_unlock_count, 1, Ordering::Release); // srcu_read_unlock
        test.add_thread(t0);
        
        // Thread 1: another SRCU reader
        let mut t1 = Thread::new(1);
        t1.rmw(0, srcu_lock_count, 1, Ordering::Acquire);
        t1.load(1, ptr, Ordering::Acquire);
        t1.load(2, data, Ordering::Relaxed);
        t1.rmw(3, srcu_unlock_count, 1, Ordering::Release);
        test.add_thread(t1);
        
        // Thread 2: updater
        let mut t2 = Thread::new(2);
        t2.store(data, 42, Ordering::Relaxed);
        t2.fence(Ordering::Release, Scope::None);
        t2.store(ptr, data, Ordering::Release);
        test.add_thread(t2);
        
        // Thread 3: synchronize_srcu (waits for readers)
        let mut t3 = Thread::new(3);
        t3.load(0, srcu_lock_count, Ordering::Acquire);
        t3.load(1, srcu_unlock_count, Ordering::Acquire);
        t3.fence(Ordering::SeqCst, Scope::None);
        test.add_thread(t3);
        
        test.expect(Outcome::new(), LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "SRCU-sleepable".into(),
            source: PatternSource::LinuxKernel,
            description: "Sleepable RCU with explicit lock/unlock counts".into(),
            category: PatternCategory::RCU,
            test: Some(test),
            num_threads: 4,
            num_locations: 4,
            bug_class: Some(BugClass::UseAfterFree),
        }
    }

    fn make_rcu_list_traversal(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("RCU-list-traversal");
        
        let head = 0x00u64;
        let node_a = 0x10u64;
        let node_a_next = 0x18u64;
        let node_b = 0x20u64;
        let node_b_next = 0x28u64;
        let new_node = 0x30u64;
        let new_next = 0x38u64;
        
        // Thread 0: list_add_rcu (insert new_node after node_a)
        let mut t0 = Thread::new(0);
        t0.store(new_node, 42, Ordering::Relaxed);
        t0.load(0, node_a_next, Ordering::Relaxed); // save node_a->next
        t0.store(new_next, 0, Ordering::Relaxed); // new->next = node_a->next
        t0.fence(Ordering::Release, Scope::None);
        t0.store(node_a_next, new_node, Ordering::Release); // node_a->next = new
        test.add_thread(t0);
        
        // Thread 1: list_del_rcu (remove node_b)
        let mut t1 = Thread::new(1);
        t1.load(0, node_b_next, Ordering::Relaxed);
        t1.store(node_a_next, 0, Ordering::Release); // prev->next = node_b->next
        test.add_thread(t1);
        
        // Thread 2: RCU reader traversing list
        let mut t2 = Thread::new(2);
        t2.load(0, head, Ordering::Acquire);
        t2.load(1, node_a, Ordering::Relaxed);
        t2.load(2, node_a_next, Ordering::Acquire); // rcu_dereference
        t2.load(3, new_node, Ordering::Relaxed);
        test.add_thread(t2);
        
        // Thread 3: another reader
        let mut t3 = Thread::new(3);
        t3.load(0, head, Ordering::Acquire);
        t3.load(1, node_a_next, Ordering::Acquire);
        t3.load(2, node_b, Ordering::Relaxed);
        t3.load(3, node_b_next, Ordering::Acquire);
        test.add_thread(t3);
        
        test.expect(Outcome::new(), LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "RCU-list-traversal".into(),
            source: PatternSource::LinuxKernel,
            description: "RCU-protected list: concurrent insert, delete, and traversal".into(),
            category: PatternCategory::RCU,
            test: Some(test),
            num_threads: 4,
            num_locations: 7,
            bug_class: Some(BugClass::UseAfterFree),
        }
    }

    // -----------------------------------------------------------------------
    // Tokio async runtime patterns
    // -----------------------------------------------------------------------

    fn extract_tokio_patterns(&mut self) {
        self.patterns.push(self.make_tokio_task_spawn());
        self.patterns.push(self.make_tokio_oneshot());
        self.patterns.push(self.make_tokio_notify());
    }

    fn make_tokio_task_spawn(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("Tokio-task-spawn");
        
        let task_state = 0x00u64;
        let join_handle = 0x08u64;
        let result = 0x10u64;
        let waker = 0x18u64;
        
        // Thread 0: spawner
        let mut t0 = Thread::new(0);
        t0.store(task_state, 1, Ordering::Release); // SCHEDULED
        t0.store(join_handle, 1, Ordering::Release);
        test.add_thread(t0);
        
        // Thread 1: worker (executes task)
        let mut t1 = Thread::new(1);
        t1.load(0, task_state, Ordering::Acquire);
        t1.store(result, 42, Ordering::Relaxed);
        t1.store(task_state, 2, Ordering::Release); // COMPLETE
        // Wake joiner
        t1.load(1, waker, Ordering::Acquire);
        test.add_thread(t1);
        
        // Thread 2: joiner (awaits result)
        let mut t2 = Thread::new(2);
        t2.store(waker, 1, Ordering::Release);
        t2.load(0, task_state, Ordering::Acquire);
        t2.load(1, result, Ordering::Relaxed);
        test.add_thread(t2);
        
        // Thread 3: canceller
        let mut t3 = Thread::new(3);
        t3.rmw(0, task_state, 3, Ordering::AcqRel); // CANCELLED
        t3.load(1, result, Ordering::Relaxed);
        test.add_thread(t3);
        
        let outcome = Outcome::new()
            .with_reg(2, 0, 2) // sees COMPLETE
            .with_reg(2, 1, 0); // but not result
        test.expect(outcome, LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "Tokio-task-spawn".into(),
            source: PatternSource::Tokio,
            description: "Tokio task lifecycle: spawn, execute, join, cancel".into(),
            category: PatternCategory::ProducerConsumer,
            test: Some(test),
            num_threads: 4,
            num_locations: 4,
            bug_class: Some(BugClass::StaleRead),
        }
    }

    fn make_tokio_oneshot(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("Tokio-oneshot-channel");
        
        let state = 0x00u64;
        let value = 0x10u64;
        let tx_dropped = 0x18u64;
        
        // Thread 0: sender
        let mut t0 = Thread::new(0);
        t0.store(value, 42, Ordering::Relaxed);
        t0.fence(Ordering::Release, Scope::None);
        t0.store(state, 1, Ordering::Release); // VALUE_SENT
        test.add_thread(t0);
        
        // Thread 1: receiver
        let mut t1 = Thread::new(1);
        t1.load(0, state, Ordering::Acquire);
        t1.load(1, value, Ordering::Relaxed);
        test.add_thread(t1);
        
        // Thread 2: sender drop (without sending)
        let mut t2 = Thread::new(2);
        t2.store(tx_dropped, 1, Ordering::Release);
        t2.rmw(0, state, 2, Ordering::AcqRel); // TX_DROPPED
        test.add_thread(t2);
        
        // Thread 3: receiver checking for drop
        let mut t3 = Thread::new(3);
        t3.load(0, state, Ordering::Acquire);
        t3.load(1, tx_dropped, Ordering::Relaxed);
        test.add_thread(t3);
        
        let outcome = Outcome::new()
            .with_reg(1, 0, 1)
            .with_reg(1, 1, 0);
        test.expect(outcome, LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "Tokio-oneshot-channel".into(),
            source: PatternSource::Tokio,
            description: "Tokio oneshot channel: send/recv with drop detection".into(),
            category: PatternCategory::ProducerConsumer,
            test: Some(test),
            num_threads: 4,
            num_locations: 3,
            bug_class: Some(BugClass::StaleRead),
        }
    }

    fn make_tokio_notify(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("Tokio-notify");
        
        let state = 0x00u64;
        let waiters = 0x08u64;
        let data = 0x10u64;
        
        // Thread 0: notifier
        let mut t0 = Thread::new(0);
        t0.store(data, 42, Ordering::Relaxed);
        t0.fence(Ordering::Release, Scope::None);
        t0.rmw(0, state, 1, Ordering::AcqRel); // notify
        test.add_thread(t0);
        
        // Threads 1-3: waiters
        for t in 1..4 {
            let mut thread = Thread::new(t);
            thread.rmw(0, waiters, 1, Ordering::AcqRel); // register as waiter
            thread.load(1, state, Ordering::Acquire); // check notification
            thread.load(2, data, Ordering::Relaxed);
            test.add_thread(thread);
        }
        
        let outcome = Outcome::new()
            .with_reg(1, 1, 1)
            .with_reg(1, 2, 0);
        test.expect(outcome, LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "Tokio-notify".into(),
            source: PatternSource::Tokio,
            description: "Tokio Notify: one notifier, multiple waiters".into(),
            category: PatternCategory::PublishSubscribe,
            test: Some(test),
            num_threads: 4,
            num_locations: 3,
            bug_class: Some(BugClass::OrderingViolation),
        }
    }

    // -----------------------------------------------------------------------
    // Crossbeam patterns
    // -----------------------------------------------------------------------

    fn extract_crossbeam_patterns(&mut self) {
        self.patterns.push(self.make_crossbeam_epoch());
        self.patterns.push(self.make_crossbeam_channel());
    }

    fn make_crossbeam_epoch(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("Crossbeam-epoch-GC");
        
        let global_epoch = 0x00u64;
        let local_epoch_0 = 0x10u64;
        let local_epoch_1 = 0x18u64;
        let garbage_bag = 0x20u64;
        let data = 0x28u64;
        
        // Thread 0: pin + access + unpin
        let mut t0 = Thread::new(0);
        t0.load(0, global_epoch, Ordering::SeqCst);
        t0.store(local_epoch_0, 1, Ordering::Release); // pin: store local epoch
        t0.fence(Ordering::SeqCst, Scope::None);
        t0.load(1, data, Ordering::Acquire); // access protected data
        t0.store(local_epoch_0, 0, Ordering::Release); // unpin
        test.add_thread(t0);
        
        // Thread 1: pin + access + unpin
        let mut t1 = Thread::new(1);
        t1.load(0, global_epoch, Ordering::SeqCst);
        t1.store(local_epoch_1, 1, Ordering::Release);
        t1.fence(Ordering::SeqCst, Scope::None);
        t1.load(1, data, Ordering::Acquire);
        t1.store(local_epoch_1, 0, Ordering::Release);
        test.add_thread(t1);
        
        // Thread 2: try_advance epoch
        let mut t2 = Thread::new(2);
        t2.load(0, local_epoch_0, Ordering::Acquire);
        t2.load(1, local_epoch_1, Ordering::Acquire);
        t2.rmw(2, global_epoch, 1, Ordering::SeqCst); // advance
        test.add_thread(t2);
        
        // Thread 3: collector (free old garbage)
        let mut t3 = Thread::new(3);
        t3.load(0, global_epoch, Ordering::Acquire);
        t3.load(1, garbage_bag, Ordering::Acquire);
        t3.store(data, 0xDEAD, Ordering::Relaxed); // free
        test.add_thread(t3);
        
        let outcome = Outcome::new()
            .with_reg(0, 1, 0xDEAD); // reader sees freed data
        test.expect(outcome, LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "Crossbeam-epoch-GC".into(),
            source: PatternSource::Crossbeam,
            description: "Crossbeam epoch-based garbage collection: pin/unpin/advance/collect".into(),
            category: PatternCategory::MemoryReclamation,
            test: Some(test),
            num_threads: 4,
            num_locations: 5,
            bug_class: Some(BugClass::UseAfterFree),
        }
    }

    fn make_crossbeam_channel(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("Crossbeam-channel");
        
        let head = 0x00u64;
        let tail = 0x08u64;
        let lap = 0x10u64;
        let slot0 = 0x18u64;
        let slot1 = 0x20u64;
        
        // Thread 0: sender 1
        let mut t0 = Thread::new(0);
        t0.load(0, tail, Ordering::Relaxed);
        t0.store(slot0, 42, Ordering::Relaxed);
        t0.rmw(1, tail, 1, Ordering::Release);
        test.add_thread(t0);
        
        // Thread 1: sender 2
        let mut t1 = Thread::new(1);
        t1.load(0, tail, Ordering::Relaxed);
        t1.store(slot1, 99, Ordering::Relaxed);
        t1.rmw(1, tail, 1, Ordering::Release);
        test.add_thread(t1);
        
        // Thread 2: receiver 1
        let mut t2 = Thread::new(2);
        t2.load(0, head, Ordering::Relaxed);
        t2.load(1, tail, Ordering::Acquire);
        t2.load(2, slot0, Ordering::Relaxed);
        t2.rmw(3, head, 1, Ordering::Release);
        test.add_thread(t2);
        
        // Thread 3: receiver 2
        let mut t3 = Thread::new(3);
        t3.load(0, head, Ordering::Relaxed);
        t3.load(1, tail, Ordering::Acquire);
        t3.load(2, slot1, Ordering::Relaxed);
        t3.rmw(3, head, 1, Ordering::Release);
        test.add_thread(t3);
        
        test.expect(Outcome::new(), LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "Crossbeam-channel".into(),
            source: PatternSource::Crossbeam,
            description: "Crossbeam bounded channel: 2 senders + 2 receivers".into(),
            category: PatternCategory::ProducerConsumer,
            test: Some(test),
            num_threads: 4,
            num_locations: 5,
            bug_class: Some(BugClass::LostUpdate),
        }
    }

    // -----------------------------------------------------------------------
    // Std library patterns
    // -----------------------------------------------------------------------

    fn extract_std_patterns(&mut self) {
        self.patterns.push(self.make_arc_pattern());
        self.patterns.push(self.make_once_cell_pattern());
        self.patterns.push(self.make_condvar_pattern());
    }

    fn make_arc_pattern(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("Std-Arc-drop");
        
        let ref_count = 0x00u64;
        let data = 0x10u64;
        let weak_count = 0x18u64;
        
        // Threads 0-2: clone + use + drop Arc
        for t in 0..3 {
            let mut thread = Thread::new(t);
            // clone: increment refcount
            thread.rmw(0, ref_count, 1, Ordering::Relaxed);
            // use data
            thread.load(1, data, Ordering::Acquire);
            // drop: decrement refcount
            thread.rmw(2, ref_count, 0, Ordering::Release); // fetch_sub(1, Release)
            test.add_thread(thread);
        }
        
        // Thread 3: weak reference
        let mut t3 = Thread::new(3);
        t3.rmw(0, weak_count, 1, Ordering::Relaxed);
        t3.load(1, ref_count, Ordering::Acquire); // check strong count
        t3.load(2, data, Ordering::Acquire); // upgrade attempt
        t3.rmw(3, weak_count, 0, Ordering::Release);
        test.add_thread(t3);
        
        test.expect(Outcome::new(), LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "Std-Arc-drop".into(),
            source: PatternSource::StdLibrary,
            description: "Arc reference counting: clone, use, drop with weak upgrade".into(),
            category: PatternCategory::ReferenceCount,
            test: Some(test),
            num_threads: 4,
            num_locations: 3,
            bug_class: Some(BugClass::UseAfterFree),
        }
    }

    fn make_once_cell_pattern(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("Std-OnceCell");
        
        let state = 0x00u64; // 0=EMPTY, 1=RUNNING, 2=COMPLETE
        let value = 0x10u64;
        
        // Threads 0-3: all try to initialize
        for t in 0..4 {
            let mut thread = Thread::new(t);
            // try CAS EMPTY -> RUNNING
            thread.rmw(0, state, 1, Ordering::AcqRel);
            // If we won: write value
            thread.store(value, (t + 1) as Value, Ordering::Relaxed);
            thread.fence(Ordering::Release, Scope::None);
            // Mark COMPLETE
            thread.store(state, 2, Ordering::Release);
            // If we lost: read value
            thread.load(1, value, Ordering::Acquire);
            test.add_thread(thread);
        }
        
        // Forbidden: sees COMPLETE but wrong value
        let outcome = Outcome::new()
            .with_reg(1, 0, 1) // lost the CAS (state was already 1)
            .with_reg(1, 1, 0); // but value not yet written
        test.expect(outcome, LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "Std-OnceCell".into(),
            source: PatternSource::StdLibrary,
            description: "OnceCell/OnceLock: 4 threads racing to initialize".into(),
            category: PatternCategory::DoubleCheckedLocking,
            test: Some(test),
            num_threads: 4,
            num_locations: 2,
            bug_class: Some(BugClass::InitializationRace),
        }
    }

    fn make_condvar_pattern(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("Std-Condvar");
        
        let mutex = 0x00u64;
        let condition = 0x10u64;
        let data = 0x18u64;
        let data2 = 0x20u64;
        
        // Thread 0: producer
        let mut t0 = Thread::new(0);
        t0.rmw(0, mutex, 1, Ordering::Acquire); // lock
        t0.store(data, 42, Ordering::Relaxed);
        t0.store(data2, 99, Ordering::Relaxed);
        t0.store(condition, 1, Ordering::Relaxed);
        t0.store(mutex, 0, Ordering::Release); // unlock
        // notify_all (implied by unlock)
        test.add_thread(t0);
        
        // Threads 1-3: consumers (wait on condvar)
        for t in 1..4 {
            let mut thread = Thread::new(t);
            thread.rmw(0, mutex, 1, Ordering::Acquire);
            thread.load(1, condition, Ordering::Relaxed);
            thread.load(2, data, Ordering::Relaxed);
            thread.load(3, data2, Ordering::Relaxed);
            thread.store(mutex, 0, Ordering::Release);
            test.add_thread(thread);
        }
        
        let outcome = Outcome::new()
            .with_reg(1, 1, 1) // condition set
            .with_reg(1, 2, 0); // but data not visible
        test.expect(outcome, LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "Std-Condvar".into(),
            source: PatternSource::StdLibrary,
            description: "Condvar: producer sets data, notifies consumers via mutex".into(),
            category: PatternCategory::ProducerConsumer,
            test: Some(test),
            num_threads: 4,
            num_locations: 4,
            bug_class: Some(BugClass::OrderingViolation),
        }
    }

    // -----------------------------------------------------------------------
    // Classic concurrency bug patterns
    // -----------------------------------------------------------------------

    fn extract_classic_bug_patterns(&mut self) {
        self.patterns.push(self.make_double_checked_locking());
        self.patterns.push(self.make_aba_problem());
        self.patterns.push(self.make_signal_handler_race());
    }

    fn make_double_checked_locking(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("Bug-double-checked-lock");
        
        let instance = 0x00u64; // pointer to singleton
        let lock = 0x08u64;
        let data = 0x10u64;
        let init_done = 0x18u64;
        
        // All threads: double-checked locking pattern
        for t in 0..4 {
            let mut thread = Thread::new(t);
            // First check (no lock)
            thread.load(0, instance, Ordering::Acquire);
            // Lock
            thread.rmw(1, lock, 1, Ordering::Acquire);
            // Second check
            thread.load(2, instance, Ordering::Relaxed);
            // Initialize (if null)
            thread.store(data, (t + 1) as Value, Ordering::Relaxed);
            thread.fence(Ordering::Release, Scope::None);
            thread.store(instance, data, Ordering::Release);
            // Unlock
            thread.store(lock, 0, Ordering::Release);
            test.add_thread(thread);
        }
        
        test.expect(Outcome::new(), LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "Bug-double-checked-lock".into(),
            source: PatternSource::Academic,
            description: "Double-checked locking pattern — classic bug without barriers".into(),
            category: PatternCategory::DoubleCheckedLocking,
            test: Some(test),
            num_threads: 4,
            num_locations: 4,
            bug_class: Some(BugClass::InitializationRace),
        }
    }

    fn make_aba_problem(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("Bug-ABA");
        
        let head = 0x00u64;
        let node_a = 0x10u64;
        let node_b = 0x20u64;
        
        // Thread 0: read head (sees A), gets preempted
        let mut t0 = Thread::new(0);
        t0.load(0, head, Ordering::Acquire); // sees A
        // (preempted here)
        // try CAS(head, A, B) — succeeds even though A was recycled
        t0.rmw(1, head, node_b, Ordering::AcqRel);
        test.add_thread(t0);
        
        // Thread 1: pop A
        let mut t1 = Thread::new(1);
        t1.load(0, head, Ordering::Acquire);
        t1.rmw(1, head, node_b, Ordering::AcqRel); // remove A
        test.add_thread(t1);
        
        // Thread 2: free A, push C (which reuses A's address)
        let mut t2 = Thread::new(2);
        t2.store(node_a, 0xDEAD, Ordering::Relaxed); // "free" A
        t2.store(node_a, 0xBEEF, Ordering::Relaxed); // reuse as C
        t2.rmw(0, head, node_a, Ordering::AcqRel); // push "C" (same addr as A)
        test.add_thread(t2);
        
        // Thread 3: observer
        let mut t3 = Thread::new(3);
        t3.load(0, head, Ordering::Acquire);
        t3.load(1, node_a, Ordering::Relaxed);
        t3.load(2, node_b, Ordering::Relaxed);
        test.add_thread(t3);
        
        test.expect(Outcome::new(), LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "Bug-ABA".into(),
            source: PatternSource::Academic,
            description: "ABA problem: CAS succeeds spuriously after node recycling".into(),
            category: PatternCategory::LockFree,
            test: Some(test),
            num_threads: 4,
            num_locations: 3,
            bug_class: Some(BugClass::ABAViolation),
        }
    }

    fn make_signal_handler_race(&self) -> ConcurrentPattern {
        let mut test = LitmusTest::new("Bug-signal-handler-race");
        
        let sig_flag = 0x00u64;
        let shared_data = 0x10u64;
        let lock = 0x18u64;
        
        // Thread 0: main thread
        let mut t0 = Thread::new(0);
        t0.rmw(0, lock, 1, Ordering::Acquire);
        t0.store(shared_data, 1, Ordering::Relaxed);
        t0.load(1, sig_flag, Ordering::Relaxed); // check for signal
        t0.store(shared_data, 2, Ordering::Relaxed);
        t0.store(lock, 0, Ordering::Release);
        test.add_thread(t0);
        
        // Thread 1: signal handler (interrupts thread 0)
        let mut t1 = Thread::new(1);
        t1.store(sig_flag, 1, Ordering::Relaxed);
        // Signal handler touches shared data without lock!
        t1.load(0, shared_data, Ordering::Relaxed);
        t1.store(shared_data, 99, Ordering::Relaxed);
        test.add_thread(t1);
        
        // Thread 2: another thread
        let mut t2 = Thread::new(2);
        t2.rmw(0, lock, 1, Ordering::Acquire);
        t2.load(1, shared_data, Ordering::Relaxed);
        t2.store(lock, 0, Ordering::Release);
        test.add_thread(t2);
        
        // Thread 3: observer
        let mut t3 = Thread::new(3);
        t3.load(0, shared_data, Ordering::Relaxed);
        t3.load(1, sig_flag, Ordering::Relaxed);
        test.add_thread(t3);
        
        test.expect(Outcome::new(), LitmusOutcome::Forbidden);
        
        ConcurrentPattern {
            name: "Bug-signal-handler-race".into(),
            source: PatternSource::Academic,
            description: "Signal handler race: handler accesses shared data without synchronization".into(),
            category: PatternCategory::SignalHandling,
            test: Some(test),
            num_threads: 4,
            num_locations: 3,
            bug_class: Some(BugClass::DataRace),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_all() {
        let mut extractor = RealCodeExtractor::new();
        let patterns = extractor.extract_all();
        assert!(patterns.len() >= 25, "Should extract many patterns, got {}", patterns.len());
    }

    #[test]
    fn test_all_patterns_have_tests() {
        let mut extractor = RealCodeExtractor::new();
        extractor.extract_all();
        let tests = extractor.litmus_tests();
        assert!(!tests.is_empty());
        assert_eq!(tests.len(), extractor.patterns().len());
    }

    #[test]
    fn test_pattern_sources() {
        let mut extractor = RealCodeExtractor::new();
        extractor.extract_all();
        let sources: Vec<_> = extractor.patterns().iter()
            .map(|p| &p.source)
            .collect();
        assert!(sources.contains(&&PatternSource::LinuxKernel));
        assert!(sources.contains(&&PatternSource::Tokio));
        assert!(sources.contains(&&PatternSource::Crossbeam));
        assert!(sources.contains(&&PatternSource::LockFreeAlgorithm));
    }

    #[test]
    fn test_thread_counts() {
        let mut extractor = RealCodeExtractor::new();
        extractor.extract_all();
        for pattern in extractor.patterns() {
            if let Some(test) = &pattern.test {
                assert!(test.threads.len() >= 2,
                    "Pattern {} should have at least 2 threads", pattern.name);
            }
        }
    }

    #[test]
    fn test_linux_kernel_patterns() {
        let mut extractor = RealCodeExtractor::new();
        extractor.extract_all();
        let kernel_patterns: Vec<_> = extractor.patterns().iter()
            .filter(|p| p.source == PatternSource::LinuxKernel)
            .collect();
        assert!(kernel_patterns.len() >= 10, "Should have many kernel patterns");
    }
}
