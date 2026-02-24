//! Constant-time analysis for LITMUS∞.
//!
//! Detects secret-dependent branching, secret-dependent memory accesses,
//! and variable-time instructions. Provides taint tracking, control flow
//! graph construction, and constant-time transformation suggestions.

use std::collections::{HashMap, HashSet, BTreeMap, VecDeque};
use std::fmt;

// ═══════════════════════════════════════════════════════════════════════
// Security levels and configuration
// ═══════════════════════════════════════════════════════════════════════

/// Security level for data classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SecurityLevel {
    /// Public data, safe to branch on.
    Public,
    /// Secret data, must not influence timing.
    Secret,
    /// Mixed (contains both public and secret components).
    Mixed,
}

impl SecurityLevel {
    /// Join (least upper bound) of two security levels.
    pub fn join(self, other: Self) -> Self {
        match (self, other) {
            (SecurityLevel::Secret, _) | (_, SecurityLevel::Secret) => SecurityLevel::Secret,
            (SecurityLevel::Mixed, _) | (_, SecurityLevel::Mixed) => SecurityLevel::Mixed,
            _ => SecurityLevel::Public,
        }
    }
}

/// Target architecture for timing model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetArch {
    X86,
    ARM,
    PTX,
    SPIRV,
    Generic,
}

/// Configuration for constant-time analysis.
#[derive(Debug, Clone)]
pub struct CtConfig {
    /// Secret annotations for variables.
    pub secret_annotations: HashMap<String, SecurityLevel>,
    /// Target architecture.
    pub target_arch: TargetArch,
    /// Maximum analysis depth.
    pub analysis_depth: usize,
    /// Check for secret-dependent memory accesses.
    pub check_memory_access: bool,
    /// Check for secret-dependent branching.
    pub check_branching: bool,
    /// Check for variable-time instructions.
    pub check_timing: bool,
}

impl Default for CtConfig {
    fn default() -> Self {
        CtConfig {
            secret_annotations: HashMap::new(),
            target_arch: TargetArch::Generic,
            analysis_depth: 100,
            check_memory_access: true,
            check_branching: true,
            check_timing: true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Program representation
// ═══════════════════════════════════════════════════════════════════════

/// A variable in the program.
#[derive(Debug, Clone)]
pub struct Variable {
    /// Variable name.
    pub name: String,
    /// Security level.
    pub level: SecurityLevel,
    /// Type size in bytes.
    pub size: usize,
}

/// Binary operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add, Sub, Mul, Div, Mod,
    And, Or, Xor, Shl, Shr,
    Eq, Ne, Lt, Le, Gt, Ge,
}

/// Unary operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Not, Neg, BitNot,
}

/// An instruction in the program.
#[derive(Debug, Clone)]
pub enum Instruction {
    /// Load from memory.
    Load { dst: String, addr: String, secret_dep: bool },
    /// Store to memory.
    Store { addr: String, val: String, secret_dep: bool },
    /// Binary operation.
    BinaryOp { op: BinOp, dst: String, src1: String, src2: String },
    /// Unary operation.
    UnaryOp { op: UnaryOp, dst: String, src: String },
    /// Conditional branch.
    Branch { cond: String, true_target: usize, false_target: usize },
    /// Function call.
    Call { func: String, args: Vec<String>, dst: Option<String> },
    /// Return.
    Return { val: Option<String> },
    /// Fence.
    Fence,
    /// No operation.
    Nop,
    /// Assignment.
    Assign { dst: String, src: String },
    /// Load immediate.
    LoadImm { dst: String, value: i64 },
}

/// A function in the program.
#[derive(Debug, Clone)]
pub struct Function {
    /// Function name.
    pub name: String,
    /// Parameters.
    pub params: Vec<Variable>,
    /// Local variables.
    pub locals: Vec<Variable>,
    /// Instructions.
    pub instructions: Vec<Instruction>,
}

/// A program to analyze.
#[derive(Debug, Clone)]
pub struct Program {
    /// Instructions.
    pub instructions: Vec<Instruction>,
    /// Variables.
    pub variables: HashMap<String, Variable>,
    /// Functions.
    pub functions: Vec<Function>,
    /// Control flow graph.
    pub cfg: ControlFlowGraph,
}

// ═══════════════════════════════════════════════════════════════════════
// ControlFlowGraph
// ═══════════════════════════════════════════════════════════════════════

/// A basic block in the control flow graph.
#[derive(Debug, Clone)]
pub struct BasicBlock {
    /// Block ID.
    pub id: usize,
    /// Instructions in this block.
    pub instructions: Vec<usize>, // indices into program instructions
    /// Successor blocks.
    pub successors: Vec<usize>,
    /// Predecessor blocks.
    pub predecessors: Vec<usize>,
}

/// Control flow graph.
#[derive(Debug, Clone)]
pub struct ControlFlowGraph {
    /// Basic blocks.
    pub blocks: Vec<BasicBlock>,
    /// Entry block ID.
    pub entry: usize,
    /// Exit block IDs.
    pub exits: Vec<usize>,
    /// Dominator tree: block -> immediate dominator.
    pub dominators: HashMap<usize, usize>,
    /// Post-dominator tree.
    pub post_dominators: HashMap<usize, usize>,
}

impl ControlFlowGraph {
    /// Create a new empty CFG.
    pub fn new() -> Self {
        ControlFlowGraph {
            blocks: Vec::new(),
            entry: 0,
            exits: Vec::new(),
            dominators: HashMap::new(),
            post_dominators: HashMap::new(),
        }
    }

    /// Build CFG from instructions.
    pub fn build(instructions: &[Instruction]) -> Self {
        if instructions.is_empty() {
            return Self::new();
        }

        // Find leaders (start of basic blocks)
        let mut leaders = HashSet::new();
        leaders.insert(0);

        for (i, instr) in instructions.iter().enumerate() {
            match instr {
                Instruction::Branch { true_target, false_target, .. } => {
                    leaders.insert(*true_target);
                    leaders.insert(*false_target);
                    if i + 1 < instructions.len() {
                        leaders.insert(i + 1);
                    }
                }
                Instruction::Return { .. } => {
                    if i + 1 < instructions.len() {
                        leaders.insert(i + 1);
                    }
                }
                _ => {}
            }
        }

        let mut sorted_leaders: Vec<usize> = leaders.into_iter()
            .filter(|&l| l < instructions.len())
            .collect();
        sorted_leaders.sort();

        // Create basic blocks
        let mut blocks = Vec::new();
        let mut block_map: HashMap<usize, usize> = HashMap::new();

        for (idx, &leader) in sorted_leaders.iter().enumerate() {
            let end = sorted_leaders.get(idx + 1).copied().unwrap_or(instructions.len());
            let instrs: Vec<usize> = (leader..end).collect();
            block_map.insert(leader, idx);
            blocks.push(BasicBlock {
                id: idx,
                instructions: instrs,
                successors: Vec::new(),
                predecessors: Vec::new(),
            });
        }

        // Add edges
        let num_blocks = blocks.len();
        for (idx, block) in blocks.iter_mut().enumerate() {
            if let Some(&last_instr_idx) = block.instructions.last() {
                if last_instr_idx < instructions.len() {
                    match &instructions[last_instr_idx] {
                        Instruction::Branch { true_target, false_target, .. } => {
                            if let Some(&bid) = block_map.get(true_target) {
                                block.successors.push(bid);
                            }
                            if let Some(&bid) = block_map.get(false_target) {
                                if !block.successors.contains(&bid) {
                                    block.successors.push(bid);
                                }
                            }
                        }
                        Instruction::Return { .. } => {
                            // No successors
                        }
                        _ => {
                            // Fall through to next block
                            if idx + 1 < num_blocks {
                                block.successors.push(idx + 1);
                            }
                        }
                    }
                }
            }
        }

        // Add predecessors
        let succs: Vec<Vec<usize>> = blocks.iter().map(|b| b.successors.clone()).collect();
        for (i, succ_list) in succs.iter().enumerate() {
            for &succ in succ_list {
                if succ < blocks.len() {
                    blocks[succ].predecessors.push(i);
                }
            }
        }

        let exits: Vec<usize> = blocks.iter()
            .filter(|b| b.successors.is_empty())
            .map(|b| b.id)
            .collect();

        let mut cfg = ControlFlowGraph {
            blocks,
            entry: 0,
            exits,
            dominators: HashMap::new(),
            post_dominators: HashMap::new(),
        };

        cfg.compute_dominators();
        cfg
    }

    /// Compute dominator tree using iterative algorithm.
    fn compute_dominators(&mut self) {
        let n = self.blocks.len();
        if n == 0 { return; }

        // Initialize
        let mut doms: Vec<Option<usize>> = vec![None; n];
        doms[self.entry] = Some(self.entry);

        let mut changed = true;
        let order = self.reverse_postorder();

        while changed {
            changed = false;
            for &b in &order {
                if b == self.entry { continue; }
                let preds = &self.blocks[b].predecessors;

                let mut new_idom = None;
                for &p in preds {
                    if doms[p].is_some() {
                        new_idom = match new_idom {
                            None => Some(p),
                            Some(cur) => Some(self.intersect_doms(&doms, cur, p)),
                        };
                    }
                }

                if new_idom != doms[b] {
                    doms[b] = new_idom;
                    changed = true;
                }
            }
        }

        self.dominators.clear();
        for (i, dom) in doms.iter().enumerate() {
            if let Some(d) = dom {
                if *d != i {
                    self.dominators.insert(i, *d);
                }
            }
        }
    }

    fn intersect_doms(&self, doms: &[Option<usize>], mut a: usize, mut b: usize) -> usize {
        while a != b {
            while a > b {
                a = doms[a].unwrap_or(a);
                if a == doms[a].unwrap_or(a) { break; }
            }
            while b > a {
                b = doms[b].unwrap_or(b);
                if b == doms[b].unwrap_or(b) { break; }
            }
            if a == b { break; }
            // Prevent infinite loops
            if doms[a] == Some(a) || doms[b] == Some(b) { break; }
        }
        a.min(b)
    }

    fn reverse_postorder(&self) -> Vec<usize> {
        let n = self.blocks.len();
        let mut visited = vec![false; n];
        let mut order = Vec::new();

        fn dfs(cfg: &ControlFlowGraph, node: usize, visited: &mut Vec<bool>, order: &mut Vec<usize>) {
            if visited[node] { return; }
            visited[node] = true;
            for &succ in &cfg.blocks[node].successors {
                if succ < visited.len() {
                    dfs(cfg, succ, visited, order);
                }
            }
            order.push(node);
        }

        dfs(self, self.entry, &mut visited, &mut order);
        order.reverse();
        order
    }

    /// Check if block a dominates block b.
    pub fn dominates(&self, a: usize, b: usize) -> bool {
        if a == b { return true; }
        let mut current = b;
        while let Some(&dom) = self.dominators.get(&current) {
            if dom == a { return true; }
            if dom == current { break; }
            current = dom;
        }
        false
    }
}

// ═══════════════════════════════════════════════════════════════════════
// TaintTracker — forward taint propagation
// ═══════════════════════════════════════════════════════════════════════

/// Forward taint tracking for secret propagation.
#[derive(Debug, Clone)]
pub struct TaintTracker {
    /// Taint status per variable.
    taint: HashMap<String, SecurityLevel>,
    /// Track implicit flows through control dependencies.
    pub track_implicit: bool,
    /// Current control dependency taint.
    control_taint: SecurityLevel,
}

impl TaintTracker {
    /// Create a new taint tracker.
    pub fn new() -> Self {
        TaintTracker {
            taint: HashMap::new(),
            track_implicit: true,
            control_taint: SecurityLevel::Public,
        }
    }

    /// Mark a variable as tainted with a given level.
    pub fn taint_var(&mut self, var: &str, level: SecurityLevel) {
        self.taint.insert(var.to_string(), level);
    }

    /// Check if a variable is tainted (secret).
    pub fn is_tainted(&self, var: &str) -> bool {
        matches!(self.taint.get(var), Some(SecurityLevel::Secret) | Some(SecurityLevel::Mixed))
    }

    /// Get the security level of a variable.
    pub fn level_of(&self, var: &str) -> SecurityLevel {
        self.taint.get(var).copied().unwrap_or(SecurityLevel::Public)
    }

    /// Propagate taint through an instruction.
    pub fn propagate(&mut self, instr: &Instruction) {
        match instr {
            Instruction::BinaryOp { dst, src1, src2, .. } => {
                let level = self.level_of(src1).join(self.level_of(src2));
                let final_level = level.join(self.control_taint);
                self.taint.insert(dst.clone(), final_level);
            }
            Instruction::UnaryOp { dst, src, .. } => {
                let level = self.level_of(src).join(self.control_taint);
                self.taint.insert(dst.clone(), level);
            }
            Instruction::Load { dst, addr, .. } => {
                let level = self.level_of(addr).join(self.control_taint);
                self.taint.insert(dst.clone(), level);
            }
            Instruction::Store { addr, val, .. } => {
                // Store taint propagates to memory
                let _level = self.level_of(val).join(self.level_of(addr));
            }
            Instruction::Assign { dst, src } => {
                let level = self.level_of(src).join(self.control_taint);
                self.taint.insert(dst.clone(), level);
            }
            Instruction::LoadImm { dst, .. } => {
                self.taint.insert(dst.clone(), self.control_taint);
            }
            Instruction::Call { dst, args, .. } => {
                // Conservative: output is tainted if any arg is
                let level = args.iter()
                    .map(|a| self.level_of(a))
                    .fold(SecurityLevel::Public, SecurityLevel::join);
                if let Some(d) = dst {
                    self.taint.insert(d.clone(), level.join(self.control_taint));
                }
            }
            _ => {}
        }
    }

    /// Set control dependency taint (when entering a branch).
    pub fn enter_branch(&mut self, cond: &str) {
        if self.track_implicit {
            self.control_taint = self.control_taint.join(self.level_of(cond));
        }
    }

    /// Exit a branch (restore control taint).
    pub fn exit_branch(&mut self) {
        self.control_taint = SecurityLevel::Public;
    }

    /// Get all tainted variables.
    pub fn tainted_vars(&self) -> Vec<String> {
        self.taint.iter()
            .filter(|(_, &level)| level == SecurityLevel::Secret || level == SecurityLevel::Mixed)
            .map(|(name, _)| name.clone())
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Violation types
// ═══════════════════════════════════════════════════════════════════════

/// Severity of a violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

/// A branch-related constant-time violation.
#[derive(Debug, Clone)]
pub struct BranchViolation {
    /// Location (instruction index).
    pub location: usize,
    /// The tainted condition variable.
    pub condition: String,
    /// Severity.
    pub severity: Severity,
    /// Description.
    pub description: String,
}

/// A memory access constant-time violation.
#[derive(Debug, Clone)]
pub struct MemoryViolation {
    /// Location (instruction index).
    pub location: usize,
    /// The access pattern.
    pub access_pattern: String,
    /// Cache impact description.
    pub cache_impact: String,
    /// Severity.
    pub severity: Severity,
    /// Description.
    pub description: String,
}

/// A timing-related violation.
#[derive(Debug, Clone)]
pub struct TimingViolation {
    /// Location (instruction index).
    pub location: usize,
    /// The variable-time operation.
    pub operation: String,
    /// Estimated timing variation (cycles).
    pub timing_variation: f64,
    /// Severity.
    pub severity: Severity,
    /// Description.
    pub description: String,
}

// ═══════════════════════════════════════════════════════════════════════
// Analyzers
// ═══════════════════════════════════════════════════════════════════════

/// Branch analyzer: detect secret-dependent branches.
#[derive(Debug)]
pub struct BranchAnalyzer;

impl BranchAnalyzer {
    /// Analyze all branches in the program for secret-dependent conditions.
    pub fn analyze(program: &Program, taint: &TaintTracker) -> Vec<BranchViolation> {
        let mut violations = Vec::new();

        for (i, instr) in program.instructions.iter().enumerate() {
            if let Instruction::Branch { cond, .. } = instr {
                if taint.is_tainted(cond) {
                    violations.push(BranchViolation {
                        location: i,
                        condition: cond.clone(),
                        severity: Severity::High,
                        description: format!(
                            "Branch at instruction {} depends on secret variable '{}'",
                            i, cond
                        ),
                    });
                }
            }
        }

        violations
    }
}

/// Memory access analyzer: detect secret-dependent memory accesses.
#[derive(Debug)]
pub struct MemoryAccessAnalyzer;

impl MemoryAccessAnalyzer {
    /// Analyze memory accesses for secret-dependent addresses.
    pub fn analyze(program: &Program, taint: &TaintTracker) -> Vec<MemoryViolation> {
        let mut violations = Vec::new();

        for (i, instr) in program.instructions.iter().enumerate() {
            match instr {
                Instruction::Load { addr, .. } if taint.is_tainted(addr) => {
                    violations.push(MemoryViolation {
                        location: i,
                        access_pattern: format!("Load from secret-dependent address '{}'", addr),
                        cache_impact: "May cause secret-dependent cache misses".to_string(),
                        severity: Severity::Critical,
                        description: format!(
                            "Secret-dependent load at instruction {}",
                            i
                        ),
                    });
                }
                Instruction::Store { addr, .. } if taint.is_tainted(addr) => {
                    violations.push(MemoryViolation {
                        location: i,
                        access_pattern: format!("Store to secret-dependent address '{}'", addr),
                        cache_impact: "May cause secret-dependent cache evictions".to_string(),
                        severity: Severity::Critical,
                        description: format!(
                            "Secret-dependent store at instruction {}",
                            i
                        ),
                    });
                }
                _ => {}
            }
        }

        violations
    }
}

/// Instruction timing model.
#[derive(Debug, Clone)]
pub struct InstructionTiming {
    /// Variable-time operations for the target architecture.
    pub variable_time_ops: HashMap<BinOp, f64>,
}

impl InstructionTiming {
    /// Create timing model for a target architecture.
    pub fn for_arch(arch: TargetArch) -> Self {
        let mut variable_time_ops = HashMap::new();
        match arch {
            TargetArch::X86 => {
                variable_time_ops.insert(BinOp::Div, 30.0);
                variable_time_ops.insert(BinOp::Mod, 30.0);
            }
            TargetArch::ARM => {
                variable_time_ops.insert(BinOp::Div, 20.0);
                variable_time_ops.insert(BinOp::Mul, 5.0);
            }
            TargetArch::PTX => {
                variable_time_ops.insert(BinOp::Div, 40.0);
            }
            _ => {
                variable_time_ops.insert(BinOp::Div, 25.0);
                variable_time_ops.insert(BinOp::Mod, 25.0);
            }
        }
        InstructionTiming { variable_time_ops }
    }

    /// Check if an operation has data-dependent timing.
    pub fn is_variable_time(&self, op: BinOp) -> bool {
        self.variable_time_ops.contains_key(&op)
    }

    /// Get timing variation for an operation.
    pub fn timing_variation(&self, op: BinOp) -> f64 {
        self.variable_time_ops.get(&op).copied().unwrap_or(0.0)
    }
}

/// Timing model analyzer.
#[derive(Debug)]
pub struct TimingModelAnalyzer;

impl TimingModelAnalyzer {
    /// Analyze timing violations.
    pub fn analyze(
        program: &Program,
        taint: &TaintTracker,
        timing: &InstructionTiming,
    ) -> Vec<TimingViolation> {
        let mut violations = Vec::new();

        for (i, instr) in program.instructions.iter().enumerate() {
            if let Instruction::BinaryOp { op, src1, src2, .. } = instr {
                if timing.is_variable_time(*op) {
                    if taint.is_tainted(src1) || taint.is_tainted(src2) {
                        violations.push(TimingViolation {
                            location: i,
                            operation: format!("{:?}", op),
                            timing_variation: timing.timing_variation(*op),
                            severity: Severity::Medium,
                            description: format!(
                                "Variable-time {:?} on secret data at instruction {}",
                                op, i
                            ),
                        });
                    }
                }
            }
        }

        violations
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CtReport — analysis report
// ═══════════════════════════════════════════════════════════════════════

/// Constant-time analysis report.
#[derive(Debug, Clone)]
pub struct CtReport {
    /// Branch violations.
    pub branch_violations: Vec<BranchViolation>,
    /// Memory access violations.
    pub memory_violations: Vec<MemoryViolation>,
    /// Timing violations.
    pub timing_violations: Vec<TimingViolation>,
    /// Overall result.
    pub is_constant_time: bool,
    /// Summary.
    pub summary: String,
    /// Remediation suggestions.
    pub remediations: Vec<String>,
}

impl CtReport {
    /// Total number of violations.
    pub fn total_violations(&self) -> usize {
        self.branch_violations.len()
            + self.memory_violations.len()
            + self.timing_violations.len()
    }

    /// Maximum severity.
    pub fn max_severity(&self) -> Severity {
        let mut max = Severity::Low;
        for v in &self.branch_violations { max = max.max(v.severity); }
        for v in &self.memory_violations { max = max.max(v.severity); }
        for v in &self.timing_violations { max = max.max(v.severity); }
        max
    }
}

impl fmt::Display for CtReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Constant-Time Analysis Report")?;
        writeln!(f, "  Constant-time: {}", if self.is_constant_time { "YES" } else { "NO" })?;
        writeln!(f, "  Branch violations: {}", self.branch_violations.len())?;
        writeln!(f, "  Memory violations: {}", self.memory_violations.len())?;
        writeln!(f, "  Timing violations: {}", self.timing_violations.len())?;
        if !self.remediations.is_empty() {
            writeln!(f, "  Remediations:")?;
            for r in &self.remediations {
                writeln!(f, "    - {}", r)?;
            }
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ConstantTimeAnalyzer — main entry point
// ═══════════════════════════════════════════════════════════════════════

/// Main constant-time analyzer.
#[derive(Debug)]
pub struct ConstantTimeAnalyzer {
    config: CtConfig,
}

impl ConstantTimeAnalyzer {
    /// Create a new analyzer.
    pub fn new(config: CtConfig) -> Self {
        ConstantTimeAnalyzer { config }
    }

    /// Analyze a program for constant-time violations.
    pub fn analyze(&self, program: &Program) -> CtReport {
        // Initialize taint tracker
        let mut taint = TaintTracker::new();
        for (name, &level) in &self.config.secret_annotations {
            taint.taint_var(name, level);
        }
        for (name, var) in &program.variables {
            if var.level == SecurityLevel::Secret {
                taint.taint_var(name, SecurityLevel::Secret);
            }
        }

        // Propagate taint through instructions
        for instr in &program.instructions {
            taint.propagate(instr);
        }

        // Run analyses
        let mut branch_violations = Vec::new();
        let mut memory_violations = Vec::new();
        let mut timing_violations = Vec::new();

        if self.config.check_branching {
            branch_violations = BranchAnalyzer::analyze(program, &taint);
        }

        if self.config.check_memory_access {
            memory_violations = MemoryAccessAnalyzer::analyze(program, &taint);
        }

        if self.config.check_timing {
            let timing = InstructionTiming::for_arch(self.config.target_arch);
            timing_violations = TimingModelAnalyzer::analyze(program, &taint, &timing);
        }

        let is_constant_time = branch_violations.is_empty()
            && memory_violations.is_empty()
            && timing_violations.is_empty();

        // Generate remediations
        let mut remediations = Vec::new();
        for v in &branch_violations {
            remediations.push(format!(
                "Replace branch on '{}' at instruction {} with conditional move (cmov)",
                v.condition, v.location
            ));
        }
        for v in &memory_violations {
            remediations.push(format!(
                "Use constant-time table lookup at instruction {} (mask all indices)",
                v.location
            ));
        }
        for v in &timing_violations {
            remediations.push(format!(
                "Replace {:?} at instruction {} with constant-time alternative",
                v.operation, v.location
            ));
        }

        let summary = if is_constant_time {
            "Program is constant-time".to_string()
        } else {
            format!(
                "Found {} violation(s): {} branch, {} memory, {} timing",
                branch_violations.len() + memory_violations.len() + timing_violations.len(),
                branch_violations.len(),
                memory_violations.len(),
                timing_violations.len(),
            )
        };

        CtReport {
            branch_violations,
            memory_violations,
            timing_violations,
            is_constant_time,
            summary,
            remediations,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ConstantTimeTransformer — suggest transformations
// ═══════════════════════════════════════════════════════════════════════

/// Suggest constant-time transformations.
#[derive(Debug)]
pub struct ConstantTimeTransformer;

impl ConstantTimeTransformer {
    /// Transform a branch into a conditional move.
    pub fn branch_to_cmov(cond: &str, true_val: &str, false_val: &str, dst: &str) -> Vec<Instruction> {
        // result = cond ? true_val : false_val
        // Implemented as: mask = -cond; result = (true_val & mask) | (false_val & ~mask)
        vec![
            Instruction::UnaryOp {
                op: UnaryOp::Neg,
                dst: "__mask".to_string(),
                src: cond.to_string(),
            },
            Instruction::BinaryOp {
                op: BinOp::And,
                dst: "__t1".to_string(),
                src1: true_val.to_string(),
                src2: "__mask".to_string(),
            },
            Instruction::UnaryOp {
                op: UnaryOp::BitNot,
                dst: "__nmask".to_string(),
                src: "__mask".to_string(),
            },
            Instruction::BinaryOp {
                op: BinOp::And,
                dst: "__t2".to_string(),
                src1: false_val.to_string(),
                src2: "__nmask".to_string(),
            },
            Instruction::BinaryOp {
                op: BinOp::Or,
                dst: dst.to_string(),
                src1: "__t1".to_string(),
                src2: "__t2".to_string(),
            },
        ]
    }

    /// Create a constant-time table lookup.
    pub fn ct_table_lookup(index: &str, table_size: usize, dst: &str) -> Vec<Instruction> {
        // Scan all entries, select the matching one
        // For each entry i: mask = ct_eq(index, i); result |= table[i] & mask
        let mut instrs = Vec::new();
        instrs.push(Instruction::LoadImm {
            dst: dst.to_string(),
            value: 0,
        });
        // In practice, you'd unroll this for the table size
        instrs
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_program() -> Program {
        let mut variables = HashMap::new();
        variables.insert("secret".to_string(), Variable {
            name: "secret".to_string(),
            level: SecurityLevel::Secret,
            size: 8,
        });
        variables.insert("public".to_string(), Variable {
            name: "public".to_string(),
            level: SecurityLevel::Public,
            size: 8,
        });
        variables.insert("result".to_string(), Variable {
            name: "result".to_string(),
            level: SecurityLevel::Public,
            size: 8,
        });
        variables.insert("addr".to_string(), Variable {
            name: "addr".to_string(),
            level: SecurityLevel::Public,
            size: 8,
        });

        let instructions = vec![
            Instruction::Assign { dst: "result".to_string(), src: "secret".to_string() },
            Instruction::Branch {
                cond: "result".to_string(),
                true_target: 2,
                false_target: 3,
            },
            Instruction::LoadImm { dst: "result".to_string(), value: 1 },
            Instruction::LoadImm { dst: "result".to_string(), value: 0 },
        ];

        let cfg = ControlFlowGraph::build(&instructions);

        Program {
            instructions,
            variables,
            functions: Vec::new(),
            cfg,
        }
    }

    #[test]
    fn test_taint_propagation() {
        let mut taint = TaintTracker::new();
        taint.taint_var("secret", SecurityLevel::Secret);

        let instr = Instruction::BinaryOp {
            op: BinOp::Add,
            dst: "result".to_string(),
            src1: "secret".to_string(),
            src2: "public".to_string(),
        };
        taint.propagate(&instr);
        assert!(taint.is_tainted("result"));
        assert!(!taint.is_tainted("public"));
    }

    #[test]
    fn test_taint_not_propagated_from_public() {
        let mut taint = TaintTracker::new();

        let instr = Instruction::BinaryOp {
            op: BinOp::Add,
            dst: "result".to_string(),
            src1: "a".to_string(),
            src2: "b".to_string(),
        };
        taint.propagate(&instr);
        assert!(!taint.is_tainted("result"));
    }

    #[test]
    fn test_branch_violation_detection() {
        let program = make_test_program();
        let mut taint = TaintTracker::new();
        taint.taint_var("secret", SecurityLevel::Secret);

        // Propagate through assign
        taint.propagate(&program.instructions[0]);

        let violations = BranchAnalyzer::analyze(&program, &taint);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].condition, "result");
    }

    #[test]
    fn test_memory_access_violation() {
        let mut variables = HashMap::new();
        variables.insert("secret_addr".to_string(), Variable {
            name: "secret_addr".to_string(),
            level: SecurityLevel::Secret,
            size: 8,
        });

        let instructions = vec![
            Instruction::Load {
                dst: "val".to_string(),
                addr: "secret_addr".to_string(),
                secret_dep: true,
            },
        ];

        let cfg = ControlFlowGraph::build(&instructions);
        let program = Program {
            instructions,
            variables,
            functions: Vec::new(),
            cfg,
        };

        let mut taint = TaintTracker::new();
        taint.taint_var("secret_addr", SecurityLevel::Secret);

        let violations = MemoryAccessAnalyzer::analyze(&program, &taint);
        assert_eq!(violations.len(), 1);
    }

    #[test]
    fn test_timing_violation() {
        let instructions = vec![
            Instruction::BinaryOp {
                op: BinOp::Div,
                dst: "result".to_string(),
                src1: "secret".to_string(),
                src2: "divisor".to_string(),
            },
        ];

        let cfg = ControlFlowGraph::build(&instructions);
        let program = Program {
            instructions,
            variables: HashMap::new(),
            functions: Vec::new(),
            cfg,
        };

        let mut taint = TaintTracker::new();
        taint.taint_var("secret", SecurityLevel::Secret);

        let timing = InstructionTiming::for_arch(TargetArch::X86);
        let violations = TimingModelAnalyzer::analyze(&program, &taint, &timing);
        assert_eq!(violations.len(), 1);
    }

    #[test]
    fn test_cfg_construction() {
        let instructions = vec![
            Instruction::LoadImm { dst: "x".to_string(), value: 0 },
            Instruction::Branch {
                cond: "c".to_string(),
                true_target: 2,
                false_target: 3,
            },
            Instruction::LoadImm { dst: "x".to_string(), value: 1 },
            Instruction::LoadImm { dst: "x".to_string(), value: 2 },
        ];

        let cfg = ControlFlowGraph::build(&instructions);
        assert!(cfg.blocks.len() >= 2);
    }

    #[test]
    fn test_full_analysis() {
        let program = make_test_program();
        let mut config = CtConfig::default();
        config.secret_annotations.insert("secret".to_string(), SecurityLevel::Secret);

        let analyzer = ConstantTimeAnalyzer::new(config);
        let report = analyzer.analyze(&program);

        assert!(!report.is_constant_time);
        assert!(report.total_violations() > 0);
    }

    #[test]
    fn test_constant_time_program() {
        let mut variables = HashMap::new();
        variables.insert("a".to_string(), Variable {
            name: "a".to_string(),
            level: SecurityLevel::Public,
            size: 8,
        });

        let instructions = vec![
            Instruction::BinaryOp {
                op: BinOp::Add,
                dst: "result".to_string(),
                src1: "a".to_string(),
                src2: "a".to_string(),
            },
        ];

        let cfg = ControlFlowGraph::build(&instructions);
        let program = Program {
            instructions,
            variables,
            functions: Vec::new(),
            cfg,
        };

        let config = CtConfig::default();
        let analyzer = ConstantTimeAnalyzer::new(config);
        let report = analyzer.analyze(&program);

        assert!(report.is_constant_time);
    }

    #[test]
    fn test_security_level_join() {
        assert_eq!(SecurityLevel::Public.join(SecurityLevel::Public), SecurityLevel::Public);
        assert_eq!(SecurityLevel::Public.join(SecurityLevel::Secret), SecurityLevel::Secret);
        assert_eq!(SecurityLevel::Secret.join(SecurityLevel::Public), SecurityLevel::Secret);
    }

    #[test]
    fn test_instruction_timing() {
        let timing = InstructionTiming::for_arch(TargetArch::X86);
        assert!(timing.is_variable_time(BinOp::Div));
        assert!(!timing.is_variable_time(BinOp::Add));
    }

    #[test]
    fn test_branch_to_cmov() {
        let instrs = ConstantTimeTransformer::branch_to_cmov("c", "t", "f", "r");
        assert_eq!(instrs.len(), 5);
    }
}
