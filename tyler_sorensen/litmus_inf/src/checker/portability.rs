//! Architecture portability checker for concurrent code patterns.
//!
//! Given a concurrent access pattern (e.g., a lock-free queue, spinlock,
//! message-passing idiom), this module:
//! 1. Extracts the implicit litmus test
//! 2. Checks it against all memory models
//! 3. Reports which architectures are safe vs. broken
//! 4. Recommends minimal fences to fix each broken architecture
//!
//! This is the primary user-facing feature that distinguishes LITMUS∞
//! from herd7/diy: practitioners describe their concurrent pattern and
//! get an actionable portability report.

use std::collections::HashMap;
use std::fmt;
use serde::{Serialize, Deserialize};

use super::litmus::{LitmusTest, Thread, Instruction, Ordering, Outcome, LitmusOutcome, Scope};
use super::memory_model::BuiltinModel;
use super::verifier::Verifier;

/// A concurrent access pattern described at a high level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrentPattern {
    /// Human-readable name
    pub name: String,
    /// Description of the pattern
    pub description: String,
    /// The access operations per thread
    pub threads: Vec<ThreadPattern>,
    /// The "bad" outcome we want to prevent
    pub forbidden_outcome: ForbiddenOutcome,
}

/// Operations in a single thread of a concurrent pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadPattern {
    pub ops: Vec<AccessOp>,
}

/// A single memory access operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessOp {
    /// Write a value to a named variable
    Write { var: String, val: u64 },
    /// Read a named variable into a named register
    Read { var: String, reg: String },
    /// A fence with given ordering
    Fence { ordering: String },
}

/// Description of the outcome to prevent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForbiddenOutcome {
    /// Register values that constitute the bug: reg_name -> expected_value
    pub register_values: HashMap<String, u64>,
    /// Memory values that constitute the bug: var_name -> expected_value
    pub memory_values: HashMap<String, u64>,
}

/// Result of checking a pattern's portability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortabilityReport {
    pub pattern_name: String,
    pub architectures: Vec<ArchResult>,
    pub summary: String,
}

/// Result for a single architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchResult {
    pub architecture: String,
    pub model: String,
    pub safe: bool,
    pub total_executions: usize,
    pub forbidden_found: usize,
    pub recommended_fences: Vec<FenceRecommendation>,
}

/// A specific fence recommendation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FenceRecommendation {
    pub thread_id: usize,
    pub after_op: usize,
    pub fence_type: String,
    pub instruction: String,
}

impl fmt::Display for PortabilityReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════════╗")?;
        writeln!(f, "║  LITMUS∞ Portability Report                  ║")?;
        writeln!(f, "╠══════════════════════════════════════════════╣")?;
        writeln!(f, "║  Pattern: {:<35}║", self.pattern_name)?;
        writeln!(f, "╠══════════════════════════════════════════════╣")?;
        for arch in &self.architectures {
            let status = if arch.safe { "✓ SAFE" } else { "✗ BROKEN" };
            writeln!(f, "║  {:<12} {:<8} ({} execs checked) ║",
                     arch.architecture, status, arch.total_executions)?;
            if !arch.safe {
                for fence in &arch.recommended_fences {
                    writeln!(f, "║    Fix: {} after op {} in T{}       ║",
                             fence.instruction, fence.after_op, fence.thread_id)?;
                }
            }
        }
        writeln!(f, "╠══════════════════════════════════════════════╣")?;
        writeln!(f, "║  {}  ║", self.summary)?;
        writeln!(f, "╚══════════════════════════════════════════════╝")?;
        Ok(())
    }
}

/// Maps architecture names to memory models and fence instructions.
struct ArchMapping {
    name: &'static str,
    model: BuiltinModel,
    /// Given an ordering type (e.g., "store-load"), return the fence instruction.
    fence_map: fn(&str) -> &'static str,
}

fn x86_fence(ordering: &str) -> &'static str {
    match ordering {
        "store-load" => "MFENCE",
        "store-store" => "SFENCE",
        "load-load" => "LFENCE",
        _ => "MFENCE",
    }
}

fn arm_fence(ordering: &str) -> &'static str {
    match ordering {
        "store-load" => "DMB ISH",
        "store-store" => "DMB ISHST",
        "load-load" => "DMB ISHLD",
        "load-store" => "DMB ISH",
        _ => "DMB ISH",
    }
}

fn riscv_fence(ordering: &str) -> &'static str {
    match ordering {
        "store-load" => "fence rw,rw",
        "store-store" => "fence w,w",
        "load-load" => "fence r,r",
        "load-store" => "fence r,w",
        _ => "fence rw,rw",
    }
}

fn pso_fence(ordering: &str) -> &'static str {
    match ordering {
        "store-load" => "MEMBAR #StoreLoad",
        "store-store" => "MEMBAR #StoreStore",
        _ => "MEMBAR #StoreLoad|#StoreStore",
    }
}

const ARCH_MAPPINGS: &[(&str, &str)] = &[
    ("x86 (TSO)", "TSO"),
    ("SPARC (PSO)", "PSO"),
    ("ARM (ARMv8)", "ARM"),
    ("RISC-V (RVWMO)", "RISC-V"),
];

/// Check a concurrent pattern for portability across architectures.
pub fn check_portability(pattern: &ConcurrentPattern) -> PortabilityReport {
    let litmus = pattern_to_litmus(pattern);
    let mut arch_results = Vec::new();

    let models = vec![
        ("x86 (TSO)", BuiltinModel::TSO, x86_fence as fn(&str) -> &'static str),
        ("SPARC (PSO)", BuiltinModel::PSO, pso_fence as fn(&str) -> &'static str),
        ("ARM (ARMv8)", BuiltinModel::ARM, arm_fence as fn(&str) -> &'static str),
        ("RISC-V (RVWMO)", BuiltinModel::RISCV, riscv_fence as fn(&str) -> &'static str),
    ];

    let mut broken_count = 0;

    for (arch_name, model, fence_fn) in &models {
        let mem_model = model.build();
        let mut verifier = Verifier::new(mem_model);
        let result = verifier.verify_litmus(&litmus);

        let safe = result.forbidden_observed.is_empty();
        if !safe {
            broken_count += 1;
        }

        let recommended_fences = if !safe {
            recommend_fences_for_pattern(pattern, fence_fn)
        } else {
            vec![]
        };

        arch_results.push(ArchResult {
            architecture: arch_name.to_string(),
            model: format!("{:?}", model),
            safe,
            total_executions: result.total_executions,
            forbidden_found: result.forbidden_observed.len(),
            recommended_fences,
        });
    }

    let summary = if broken_count == 0 {
        "Pattern is portable across all tested architectures".to_string()
    } else {
        format!("Pattern breaks on {}/{} architectures — fences needed",
                broken_count, models.len())
    };

    PortabilityReport {
        pattern_name: pattern.name.clone(),
        architectures: arch_results,
        summary,
    }
}

/// Convert a high-level concurrent pattern to a litmus test.
fn pattern_to_litmus(pattern: &ConcurrentPattern) -> LitmusTest {
    let mut var_addrs: HashMap<String, u64> = HashMap::new();
    let mut next_addr: u64 = 0x100;

    // Assign addresses to variables
    for tp in &pattern.threads {
        for op in &tp.ops {
            let var = match op {
                AccessOp::Write { var, .. } => var.clone(),
                AccessOp::Read { var, .. } => var.clone(),
                AccessOp::Fence { .. } => continue,
            };
            if !var_addrs.contains_key(&var) {
                var_addrs.insert(var, next_addr);
                next_addr += 0x100;
            }
        }
    }

    let mut test = LitmusTest::new(&pattern.name);
    for (&ref var, &addr) in &var_addrs {
        test.set_initial(addr, 0);
    }

    let mut reg_map: HashMap<String, (usize, usize)> = HashMap::new();
    let mut next_reg_per_thread: Vec<usize> = vec![0; pattern.threads.len()];

    for (tid, tp) in pattern.threads.iter().enumerate() {
        let mut thread = Thread::new(tid);
        for op in &tp.ops {
            match op {
                AccessOp::Write { var, val } => {
                    let addr = var_addrs[var];
                    thread.store(addr, *val, Ordering::Relaxed);
                }
                AccessOp::Read { var, reg } => {
                    let addr = var_addrs[var];
                    let reg_idx = next_reg_per_thread[tid];
                    next_reg_per_thread[tid] += 1;
                    thread.load(reg_idx, addr, Ordering::Relaxed);
                    reg_map.insert(reg.clone(), (tid, reg_idx));
                }
                AccessOp::Fence { ordering } => {
                    let ord = match ordering.as_str() {
                        "acquire" => Ordering::Acquire,
                        "release" => Ordering::Release,
                        "seq_cst" => Ordering::SeqCst,
                        _ => Ordering::SeqCst,
                    };
                    thread.fence(ord, Scope::System);
                }
            }
        }
        test.add_thread(thread);
    }

    // Set up forbidden outcome
    let mut outcome = Outcome::new();
    for (reg_name, &val) in &pattern.forbidden_outcome.register_values {
        if let Some(&(tid, reg_idx)) = reg_map.get(reg_name) {
            outcome = outcome.with_reg(tid, reg_idx, val);
        }
    }
    for (var_name, &val) in &pattern.forbidden_outcome.memory_values {
        if let Some(&addr) = var_addrs.get(var_name) {
            outcome = outcome.with_mem(addr, val);
        }
    }
    test.expect(outcome, LitmusOutcome::Forbidden);

    test
}

/// Recommend fences based on the pattern's access structure.
fn recommend_fences_for_pattern(
    pattern: &ConcurrentPattern,
    fence_fn: &dyn Fn(&str) -> &'static str,
) -> Vec<FenceRecommendation> {
    let mut recs = Vec::new();
    for (tid, tp) in pattern.threads.iter().enumerate() {
        let ops = &tp.ops;
        for i in 0..ops.len().saturating_sub(1) {
            let ordering_type = match (&ops[i], &ops[i + 1]) {
                (AccessOp::Write { .. }, AccessOp::Read { .. }) => Some("store-load"),
                (AccessOp::Write { .. }, AccessOp::Write { .. }) => Some("store-store"),
                (AccessOp::Read { .. }, AccessOp::Read { .. }) => Some("load-load"),
                (AccessOp::Read { .. }, AccessOp::Write { .. }) => Some("load-store"),
                _ => None,
            };
            if let Some(ord_type) = ordering_type {
                // Only recommend if the two ops touch different variables
                let var_a = match &ops[i] {
                    AccessOp::Write { var, .. } | AccessOp::Read { var, .. } => Some(var),
                    _ => None,
                };
                let var_b = match &ops[i + 1] {
                    AccessOp::Write { var, .. } | AccessOp::Read { var, .. } => Some(var),
                    _ => None,
                };
                if var_a != var_b {
                    recs.push(FenceRecommendation {
                        thread_id: tid,
                        after_op: i,
                        fence_type: ord_type.to_string(),
                        instruction: fence_fn(ord_type).to_string(),
                    });
                }
            }
        }
    }
    recs
}

/// Common concurrent patterns for quick checking.
pub fn builtin_patterns() -> Vec<ConcurrentPattern> {
    vec![
        spinlock_pattern(),
        message_passing_pattern(),
        double_checked_locking_pattern(),
        seqlock_reader_pattern(),
        producer_consumer_pattern(),
    ]
}

/// Spinlock acquire-release pattern
pub fn spinlock_pattern() -> ConcurrentPattern {
    let mut forbidden_regs = HashMap::new();
    forbidden_regs.insert("r0".to_string(), 0);
    forbidden_regs.insert("r1".to_string(), 0);

    ConcurrentPattern {
        name: "Spinlock (store-buffer)".to_string(),
        description: "Two threads each set a flag and check the other's — \
                       the classic mutual exclusion check (Peterson/Dekker style)".to_string(),
        threads: vec![
            ThreadPattern {
                ops: vec![
                    AccessOp::Write { var: "flag0".to_string(), val: 1 },
                    AccessOp::Read { var: "flag1".to_string(), reg: "r0".to_string() },
                ],
            },
            ThreadPattern {
                ops: vec![
                    AccessOp::Write { var: "flag1".to_string(), val: 1 },
                    AccessOp::Read { var: "flag0".to_string(), reg: "r1".to_string() },
                ],
            },
        ],
        forbidden_outcome: ForbiddenOutcome {
            register_values: forbidden_regs,
            memory_values: HashMap::new(),
        },
    }
}

/// Message passing pattern
pub fn message_passing_pattern() -> ConcurrentPattern {
    let mut forbidden_regs = HashMap::new();
    forbidden_regs.insert("flag_val".to_string(), 1);
    forbidden_regs.insert("data_val".to_string(), 0);

    ConcurrentPattern {
        name: "Message Passing".to_string(),
        description: "Producer writes data then sets flag; consumer reads flag then data. \
                       Checks store-to-load ordering.".to_string(),
        threads: vec![
            ThreadPattern {
                ops: vec![
                    AccessOp::Write { var: "data".to_string(), val: 42 },
                    AccessOp::Write { var: "flag".to_string(), val: 1 },
                ],
            },
            ThreadPattern {
                ops: vec![
                    AccessOp::Read { var: "flag".to_string(), reg: "flag_val".to_string() },
                    AccessOp::Read { var: "data".to_string(), reg: "data_val".to_string() },
                ],
            },
        ],
        forbidden_outcome: ForbiddenOutcome {
            register_values: forbidden_regs,
            memory_values: HashMap::new(),
        },
    }
}

/// Double-checked locking pattern
pub fn double_checked_locking_pattern() -> ConcurrentPattern {
    let mut forbidden_regs = HashMap::new();
    forbidden_regs.insert("obj_val".to_string(), 0);
    forbidden_regs.insert("init_val".to_string(), 1);

    ConcurrentPattern {
        name: "Double-Checked Locking".to_string(),
        description: "Thread 0 initializes object then sets initialized flag. \
                       Thread 1 checks flag then reads object. Classic DCL bug.".to_string(),
        threads: vec![
            ThreadPattern {
                ops: vec![
                    AccessOp::Write { var: "object".to_string(), val: 1 },
                    AccessOp::Write { var: "initialized".to_string(), val: 1 },
                ],
            },
            ThreadPattern {
                ops: vec![
                    AccessOp::Read { var: "initialized".to_string(), reg: "init_val".to_string() },
                    AccessOp::Read { var: "object".to_string(), reg: "obj_val".to_string() },
                ],
            },
        ],
        forbidden_outcome: ForbiddenOutcome {
            register_values: forbidden_regs,
            memory_values: HashMap::new(),
        },
    }
}

/// Seqlock reader pattern
pub fn seqlock_reader_pattern() -> ConcurrentPattern {
    let mut forbidden_regs = HashMap::new();
    forbidden_regs.insert("r0".to_string(), 1);
    forbidden_regs.insert("r1".to_string(), 1);

    ConcurrentPattern {
        name: "Seqlock Reader".to_string(),
        description: "Load-buffering pattern: each thread reads then writes. \
                       Tests whether speculative loads can see future stores.".to_string(),
        threads: vec![
            ThreadPattern {
                ops: vec![
                    AccessOp::Read { var: "x".to_string(), reg: "r0".to_string() },
                    AccessOp::Write { var: "y".to_string(), val: 1 },
                ],
            },
            ThreadPattern {
                ops: vec![
                    AccessOp::Read { var: "y".to_string(), reg: "r1".to_string() },
                    AccessOp::Write { var: "x".to_string(), val: 1 },
                ],
            },
        ],
        forbidden_outcome: ForbiddenOutcome {
            register_values: forbidden_regs,
            memory_values: HashMap::new(),
        },
    }
}

/// Producer-consumer with write ordering
pub fn producer_consumer_pattern() -> ConcurrentPattern {
    let mut forbidden_mem = HashMap::new();
    forbidden_mem.insert("x".to_string(), 1);
    forbidden_mem.insert("y".to_string(), 1);

    ConcurrentPattern {
        name: "Producer-Consumer Write Ordering".to_string(),
        description: "Two threads write to two locations in opposite orders. \
                       Tests store-store ordering (2+2W pattern).".to_string(),
        threads: vec![
            ThreadPattern {
                ops: vec![
                    AccessOp::Write { var: "x".to_string(), val: 1 },
                    AccessOp::Write { var: "y".to_string(), val: 2 },
                ],
            },
            ThreadPattern {
                ops: vec![
                    AccessOp::Write { var: "y".to_string(), val: 1 },
                    AccessOp::Write { var: "x".to_string(), val: 2 },
                ],
            },
        ],
        forbidden_outcome: ForbiddenOutcome {
            register_values: HashMap::new(),
            memory_values: forbidden_mem,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spinlock_portability() {
        let pattern = spinlock_pattern();
        let report = check_portability(&pattern);
        // x86 (TSO) allows SB-like outcome → not safe without fence
        assert!(!report.architectures[0].safe, "Spinlock SB should break on TSO");
        // All weaker models should also break
        for arch in &report.architectures {
            assert!(!arch.safe, "Spinlock SB should break on {}", arch.architecture);
        }
    }

    #[test]
    fn test_message_passing_portability() {
        let pattern = message_passing_pattern();
        let report = check_portability(&pattern);
        // x86 (TSO) should be safe for MP
        assert!(report.architectures[0].safe, "MP should be safe on TSO");
        // PSO, ARM, RISC-V should break
        assert!(!report.architectures[1].safe, "MP should break on PSO");
        assert!(!report.architectures[2].safe, "MP should break on ARM");
        assert!(!report.architectures[3].safe, "MP should break on RISC-V");
    }

    #[test]
    fn test_seqlock_portability() {
        let pattern = seqlock_reader_pattern();
        let report = check_portability(&pattern);
        // TSO and PSO should be safe for LB, ARM/RISC-V should break
        assert!(report.architectures[0].safe, "LB should be safe on TSO");
        assert!(report.architectures[1].safe, "LB should be safe on PSO");
        assert!(!report.architectures[2].safe, "LB should break on ARM");
        assert!(!report.architectures[3].safe, "LB should break on RISC-V");
    }

    #[test]
    fn test_fence_recommendations_present() {
        let pattern = spinlock_pattern();
        let report = check_portability(&pattern);
        for arch in &report.architectures {
            if !arch.safe {
                assert!(!arch.recommended_fences.is_empty(),
                        "Broken arch {} should have fence recommendations",
                        arch.architecture);
            }
        }
    }

    #[test]
    fn test_all_builtin_patterns() {
        for pattern in builtin_patterns() {
            let report = check_portability(&pattern);
            assert!(!report.architectures.is_empty());
            assert_eq!(report.architectures.len(), 4);
        }
    }

    #[test]
    fn test_report_display() {
        let pattern = spinlock_pattern();
        let report = check_portability(&pattern);
        let display = format!("{}", report);
        assert!(display.contains("LITMUS∞"));
        assert!(display.contains("Spinlock"));
    }
}
