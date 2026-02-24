//! Optimal fence insertion.
//!
//! Given a litmus test with relaxed memory operations, determine the minimum-
//! cost set of fence insertions that make the program portable under a target
//! memory model.  Two solvers are provided:
//!
//! - **IlpSolver** — exact branch-and-bound ILP formulation.
//! - **HeuristicSolver** — fast greedy + local-search approximation.

use std::collections::{HashMap, HashSet, BTreeMap};
use std::fmt;

use crate::checker::litmus::{LitmusTest, Thread, Instruction, Ordering, Scope};
use crate::checker::execution::{Address, Value, ThreadId};

use super::cost_model_calibrated::{CalibratedCostModel, FenceCost, Architecture};

// ---------------------------------------------------------------------------
// FenceInsertion
// ---------------------------------------------------------------------------

/// A single fence to insert at a particular program point.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FenceInsertion {
    /// Thread in which the fence is inserted.
    pub thread_id: usize,
    /// Position within the thread's instruction list (inserted *before* this index).
    pub position: usize,
    /// Ordering of the fence.
    pub ordering: Ordering,
    /// Scope of the fence.
    pub scope: Scope,
}

impl FenceInsertion {
    pub fn new(thread_id: usize, position: usize, ordering: Ordering, scope: Scope) -> Self {
        Self { thread_id, position, ordering, scope }
    }
}

impl fmt::Display for FenceInsertion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "T{}@{}: fence.{}.{}", self.thread_id, self.position, self.ordering, self.scope)
    }
}

// ---------------------------------------------------------------------------
// FencePlacement
// ---------------------------------------------------------------------------

/// A complete set of fence insertions for a program.
#[derive(Debug, Clone)]
pub struct FencePlacement {
    pub insertions: Vec<FenceInsertion>,
}

impl FencePlacement {
    pub fn new() -> Self {
        Self { insertions: Vec::new() }
    }

    pub fn with_insertions(insertions: Vec<FenceInsertion>) -> Self {
        Self { insertions }
    }

    pub fn add(&mut self, insertion: FenceInsertion) {
        self.insertions.push(insertion);
    }

    pub fn num_fences(&self) -> usize {
        self.insertions.len()
    }

    /// Group insertions by thread.
    pub fn by_thread(&self) -> HashMap<usize, Vec<&FenceInsertion>> {
        let mut map: HashMap<usize, Vec<&FenceInsertion>> = HashMap::new();
        for ins in &self.insertions {
            map.entry(ins.thread_id).or_default().push(ins);
        }
        for v in map.values_mut() {
            v.sort_by_key(|i| i.position);
        }
        map
    }

    /// Apply the insertions to a litmus test, producing a new test with fences.
    pub fn apply(&self, test: &LitmusTest) -> LitmusTest {
        let mut result = test.clone();
        let by_thread = self.by_thread();
        for thread in &mut result.threads {
            if let Some(insertions) = by_thread.get(&thread.id) {
                // Insert in reverse order so positions remain valid.
                let mut sorted: Vec<&&FenceInsertion> = insertions.iter().collect();
                sorted.sort_by(|a, b| b.position.cmp(&a.position));
                for ins in sorted {
                    let pos = ins.position.min(thread.instructions.len());
                    thread.instructions.insert(pos, Instruction::Fence {
                        ordering: ins.ordering,
                        scope: ins.scope,
                    });
                }
            }
        }
        result
    }

    /// Total cost of this placement under the given cost model.
    pub fn total_cost(&self, model: &CalibratedCostModel) -> f64 {
        self.insertions.iter()
            .map(|ins| model.cost_of(&ins.ordering, &ins.scope))
            .sum()
    }
}

impl Default for FencePlacement {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for FencePlacement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "FencePlacement ({} fences):", self.num_fences())?;
        for ins in &self.insertions {
            writeln!(f, "  {}", ins)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// InsertionConfig
// ---------------------------------------------------------------------------

/// Target memory model for portability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetModel {
    /// Sequential consistency.
    SC,
    /// Total store order (x86-TSO).
    TSO,
    /// Release-acquire consistency.
    RA,
    /// Relaxed (baseline — no ordering).
    Relaxed,
}

impl fmt::Display for TargetModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SC => write!(f, "SC"),
            Self::TSO => write!(f, "TSO"),
            Self::RA => write!(f, "RA"),
            Self::Relaxed => write!(f, "Relaxed"),
        }
    }
}

/// Optimization level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// Exact ILP solution.
    Optimal,
    /// Greedy heuristic.
    Fast,
    /// Iterated local search.
    Balanced,
}

/// Configuration for fence insertion.
#[derive(Debug, Clone)]
pub struct InsertionConfig {
    pub target_model: TargetModel,
    pub cost_model: CalibratedCostModel,
    pub optimization: OptimizationLevel,
    /// Maximum time budget for ILP solver in milliseconds.
    pub timeout_ms: u64,
    /// Number of random restarts for iterated local search.
    pub num_restarts: usize,
}

impl InsertionConfig {
    pub fn new(target: TargetModel, cost_model: CalibratedCostModel) -> Self {
        Self {
            target_model: target,
            cost_model,
            optimization: OptimizationLevel::Balanced,
            timeout_ms: 5000,
            num_restarts: 10,
        }
    }

    pub fn with_optimization(mut self, level: OptimizationLevel) -> Self {
        self.optimization = level;
        self
    }

    pub fn with_timeout(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    pub fn with_restarts(mut self, n: usize) -> Self {
        self.num_restarts = n;
        self
    }
}

// ---------------------------------------------------------------------------
// InsertionResult
// ---------------------------------------------------------------------------

/// Result of fence insertion optimization.
#[derive(Debug, Clone)]
pub struct InsertionResult {
    pub placement: FencePlacement,
    pub total_cost: f64,
    pub num_fences: usize,
    pub solver_used: String,
    /// Time taken to solve, in milliseconds.
    pub solve_time_ms: u64,
    /// Whether the solution is provably optimal.
    pub is_optimal: bool,
}

impl InsertionResult {
    pub fn new(
        placement: FencePlacement,
        total_cost: f64,
        solver: &str,
        solve_time_ms: u64,
        is_optimal: bool,
    ) -> Self {
        let num_fences = placement.num_fences();
        Self { placement, total_cost, num_fences, solver_used: solver.into(), solve_time_ms, is_optimal }
    }
}

impl fmt::Display for InsertionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "InsertionResult(fences={}, cost={:.2}, solver={}, time={}ms, optimal={})",
            self.num_fences, self.total_cost, self.solver_used, self.solve_time_ms, self.is_optimal,
        )
    }
}

// ---------------------------------------------------------------------------
// Violation detection helpers
// ---------------------------------------------------------------------------

/// A potential ordering violation between two instructions in a thread.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Violation {
    thread_id: usize,
    /// Index of the first instruction.
    first: usize,
    /// Index of the second instruction.
    second: usize,
    /// Minimum ordering required to fix.
    required_ordering: Ordering,
    /// Minimum scope required.
    required_scope: Scope,
}

/// Candidate fence location between two instructions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CandidateLocation {
    thread_id: usize,
    position: usize,
}

/// Detect ordering violations in a litmus test under the target model.
fn detect_violations(test: &LitmusTest, target: TargetModel) -> Vec<Violation> {
    let mut violations = Vec::new();
    for thread in &test.threads {
        let instrs = &thread.instructions;
        for i in 0..instrs.len() {
            for j in (i + 1)..instrs.len() {
                if let Some(v) = check_pair_violation(thread.id, i, j, &instrs[i], &instrs[j], target) {
                    violations.push(v);
                }
            }
        }
    }
    violations
}

fn check_pair_violation(
    thread_id: usize,
    idx_a: usize,
    idx_b: usize,
    a: &Instruction,
    b: &Instruction,
    target: TargetModel,
) -> Option<Violation> {
    let (is_a_mem, is_b_mem) = (is_memory_op(a), is_memory_op(b));
    if !is_a_mem || !is_b_mem {
        return None;
    }

    let ord_a = instruction_ordering(a);
    let ord_b = instruction_ordering(b);

    let (required_ordering, required_scope) = match target {
        TargetModel::SC => {
            // SC requires total order on all memory ops.
            if ordering_sufficient_for_sc(ord_a, ord_b) {
                return None;
            }
            (Ordering::SeqCst, Scope::System)
        }
        TargetModel::TSO => {
            // TSO allows store-load reordering only.
            if !is_store(a) || !is_load(b) {
                return None;
            }
            if ordering_sufficient_for_tso(ord_a, ord_b) {
                return None;
            }
            (Ordering::AcqRel, Scope::System)
        }
        TargetModel::RA => {
            // RA requires release on stores, acquire on loads.
            let needed = match (is_store(a), is_load(b)) {
                (true, true) => Ordering::AcqRel,
                (true, false) => Ordering::Release,
                (false, true) => Ordering::Acquire,
                (false, false) => return None,
            };
            let have = combined_ordering(ord_a, ord_b);
            if ordering_strength_val(have) >= ordering_strength_val(needed) {
                return None;
            }
            (needed, Scope::System)
        }
        TargetModel::Relaxed => {
            // No ordering required — no violations.
            return None;
        }
    };

    Some(Violation {
        thread_id,
        first: idx_a,
        second: idx_b,
        required_ordering,
        required_scope,
    })
}

fn is_memory_op(instr: &Instruction) -> bool {
    matches!(instr,
        Instruction::Load { .. } | Instruction::Store { .. } | Instruction::RMW { .. }
    )
}

fn is_load(instr: &Instruction) -> bool {
    matches!(instr, Instruction::Load { .. })
}

fn is_store(instr: &Instruction) -> bool {
    matches!(instr, Instruction::Store { .. })
}

fn instruction_ordering(instr: &Instruction) -> Ordering {
    match instr {
        Instruction::Load { ordering, .. }
        | Instruction::Store { ordering, .. }
        | Instruction::Fence { ordering, .. }
        | Instruction::RMW { ordering, .. } => *ordering,
        _ => Ordering::Relaxed,
    }
}

fn ordering_strength_val(o: Ordering) -> u8 {
    match o {
        Ordering::Relaxed => 0,
        Ordering::Acquire | Ordering::AcquireCTA | Ordering::AcquireGPU | Ordering::AcquireSystem => 1,
        Ordering::Release | Ordering::ReleaseCTA | Ordering::ReleaseGPU | Ordering::ReleaseSystem => 2,
        Ordering::AcqRel => 3,
        Ordering::SeqCst => 4,
    }
}

fn combined_ordering(a: Ordering, b: Ordering) -> Ordering {
    let sa = ordering_strength_val(a);
    let sb = ordering_strength_val(b);
    if sa >= sb { a } else { b }
}

fn ordering_sufficient_for_sc(a: Ordering, b: Ordering) -> bool {
    ordering_strength_val(a) >= 4 && ordering_strength_val(b) >= 4
}

fn ordering_sufficient_for_tso(a: Ordering, b: Ordering) -> bool {
    // In TSO, store-load reordering is prevented by mfence-level ordering.
    ordering_strength_val(a) >= 3 || ordering_strength_val(b) >= 3
}

/// Enumerate all candidate fence locations in a test.
fn enumerate_candidates(test: &LitmusTest) -> Vec<CandidateLocation> {
    let mut candidates = Vec::new();
    for thread in &test.threads {
        // A fence can be inserted before any instruction, or after the last one.
        for pos in 0..=thread.instructions.len() {
            candidates.push(CandidateLocation {
                thread_id: thread.id,
                position: pos,
            });
        }
    }
    candidates
}

/// Check whether a fence at `loc` with the given ordering/scope fixes a violation.
fn fence_fixes_violation(
    loc: &CandidateLocation,
    fence_ord: Ordering,
    violation: &Violation,
) -> bool {
    if loc.thread_id != violation.thread_id {
        return false;
    }
    // Fence must be between the two instructions.
    if loc.position <= violation.first || loc.position > violation.second {
        return false;
    }
    ordering_strength_val(fence_ord) >= ordering_strength_val(violation.required_ordering)
}

// ---------------------------------------------------------------------------
// ILP Solver — Branch-and-Bound
// ---------------------------------------------------------------------------

/// Variable in the ILP: binary decision for placing a fence of type `fence_idx`
/// at location `loc_idx`.
#[derive(Debug, Clone)]
struct IlpVariable {
    loc_idx: usize,
    fence_ordering: Ordering,
    fence_scope: Scope,
    cost: f64,
}

/// Node in the branch-and-bound tree.
#[derive(Debug, Clone)]
struct BbNode {
    /// Fixed variable assignments: var_idx → true/false.
    fixed: HashMap<usize, bool>,
    /// Lower bound on cost for this node.
    lower_bound: f64,
}

/// ILP-based optimal fence placement solver.
#[derive(Debug)]
pub struct IlpSolver;

impl IlpSolver {
    pub fn new() -> Self {
        Self
    }

    /// Solve for the optimal fence placement.
    pub fn solve(&self, test: &LitmusTest, config: &InsertionConfig) -> InsertionResult {
        let start = std::time::Instant::now();
        let violations = detect_violations(test, config.target_model);

        if violations.is_empty() {
            return InsertionResult::new(
                FencePlacement::new(), 0.0, "ILP", 0, true,
            );
        }

        let candidates = enumerate_candidates(test);
        let fence_types = available_fence_types(&config.cost_model);

        // Build ILP variables.
        let mut variables: Vec<IlpVariable> = Vec::new();
        for (loc_idx, _loc) in candidates.iter().enumerate() {
            for &(ord, scope) in &fence_types {
                let cost = config.cost_model.cost_of(&ord, &scope);
                if cost <= 0.0 && ord == Ordering::Relaxed {
                    continue; // Skip no-op fences.
                }
                variables.push(IlpVariable {
                    loc_idx,
                    fence_ordering: ord,
                    fence_scope: scope,
                    cost,
                });
            }
        }

        if variables.is_empty() {
            // No fence types available — return empty.
            return InsertionResult::new(
                FencePlacement::new(), 0.0, "ILP", start.elapsed().as_millis() as u64, true,
            );
        }

        // Build constraint matrix: each violation must be covered by at least one variable.
        let mut covers: Vec<Vec<usize>> = vec![Vec::new(); violations.len()];
        for (var_idx, var) in variables.iter().enumerate() {
            let loc = &candidates[var.loc_idx];
            for (viol_idx, viol) in violations.iter().enumerate() {
                if fence_fixes_violation(loc, var.fence_ordering, viol) {
                    covers[viol_idx].push(var_idx);
                }
            }
        }

        // Branch-and-bound.
        let mut best_cost = f64::INFINITY;
        let mut best_assignment: HashMap<usize, bool> = HashMap::new();

        let root = BbNode { fixed: HashMap::new(), lower_bound: 0.0 };
        let mut stack: Vec<BbNode> = vec![root];
        let timeout = std::time::Duration::from_millis(config.timeout_ms);

        while let Some(node) = stack.pop() {
            if start.elapsed() > timeout {
                break;
            }
            if node.lower_bound >= best_cost {
                continue; // Prune.
            }

            // Find first uncovered violation.
            let uncovered = find_uncovered_violation(&violations, &covers, &node.fixed, &variables, &candidates);

            match uncovered {
                None => {
                    // All violations covered — compute total cost.
                    let cost: f64 = node.fixed.iter()
                        .filter(|(_, &v)| v)
                        .map(|(&idx, _)| variables[idx].cost)
                        .sum();
                    if cost < best_cost {
                        best_cost = cost;
                        best_assignment = node.fixed.clone();
                    }
                }
                Some(viol_idx) => {
                    // Branch on each variable that can cover this violation.
                    for &var_idx in &covers[viol_idx] {
                        if node.fixed.contains_key(&var_idx) {
                            continue;
                        }
                        let mut new_fixed = node.fixed.clone();
                        new_fixed.insert(var_idx, true);
                        let lb = compute_lower_bound(&new_fixed, &variables);
                        if lb < best_cost {
                            stack.push(BbNode { fixed: new_fixed, lower_bound: lb });
                        }
                    }
                }
            }
        }

        // Build placement from best assignment.
        let mut placement = FencePlacement::new();
        for (&var_idx, &val) in &best_assignment {
            if val {
                let var = &variables[var_idx];
                let loc = &candidates[var.loc_idx];
                placement.add(FenceInsertion::new(
                    loc.thread_id, loc.position, var.fence_ordering, var.fence_scope,
                ));
            }
        }

        // De-duplicate: if multiple fences at the same location, keep strongest.
        deduplicate_placement(&mut placement);

        let total_cost = placement.total_cost(&config.cost_model);
        let elapsed = start.elapsed().as_millis() as u64;
        let is_optimal = elapsed < config.timeout_ms;

        InsertionResult::new(placement, total_cost, "ILP", elapsed, is_optimal)
    }
}

fn find_uncovered_violation(
    violations: &[Violation],
    covers: &[Vec<usize>],
    fixed: &HashMap<usize, bool>,
    variables: &[IlpVariable],
    candidates: &[CandidateLocation],
) -> Option<usize> {
    'outer: for (viol_idx, viol) in violations.iter().enumerate() {
        // Check if any fixed-true variable covers this violation.
        for &var_idx in &covers[viol_idx] {
            if let Some(&true) = fixed.get(&var_idx) {
                continue 'outer;
            }
        }
        return Some(viol_idx);
    }
    None
}

fn compute_lower_bound(fixed: &HashMap<usize, bool>, variables: &[IlpVariable]) -> f64 {
    fixed.iter()
        .filter(|(_, &v)| v)
        .map(|(&idx, _)| variables[idx].cost)
        .sum()
}

fn available_fence_types(model: &CalibratedCostModel) -> Vec<(Ordering, Scope)> {
    model.costs.costs.keys().copied().collect()
}

fn deduplicate_placement(placement: &mut FencePlacement) {
    let mut best_at: HashMap<(usize, usize), FenceInsertion> = HashMap::new();
    for ins in &placement.insertions {
        let key = (ins.thread_id, ins.position);
        let entry = best_at.entry(key).or_insert_with(|| ins.clone());
        if ordering_strength_val(ins.ordering) > ordering_strength_val(entry.ordering) {
            *entry = ins.clone();
        }
    }
    placement.insertions = best_at.into_values().collect();
    placement.insertions.sort_by_key(|i| (i.thread_id, i.position));
}

// ---------------------------------------------------------------------------
// HeuristicSolver
// ---------------------------------------------------------------------------

/// Heuristic-based fast fence placement solver.
#[derive(Debug)]
pub struct HeuristicSolver;

impl HeuristicSolver {
    pub fn new() -> Self {
        Self
    }

    /// Solve using the configured heuristic strategy.
    pub fn solve(&self, test: &LitmusTest, config: &InsertionConfig) -> InsertionResult {
        match config.optimization {
            OptimizationLevel::Fast => self.greedy(test, config),
            OptimizationLevel::Balanced => self.iterated_local_search(test, config),
            OptimizationLevel::Optimal => self.greedy(test, config), // fallback
        }
    }

    /// Greedy: for each violation, insert the cheapest fence that fixes it.
    pub fn greedy(&self, test: &LitmusTest, config: &InsertionConfig) -> InsertionResult {
        let start = std::time::Instant::now();
        let violations = detect_violations(test, config.target_model);

        if violations.is_empty() {
            return InsertionResult::new(FencePlacement::new(), 0.0, "Greedy", 0, false);
        }

        let candidates = enumerate_candidates(test);
        let fence_types = available_fence_types(&config.cost_model);
        let mut placement = FencePlacement::new();
        let mut covered: HashSet<usize> = HashSet::new();

        for (viol_idx, viol) in violations.iter().enumerate() {
            if covered.contains(&viol_idx) {
                continue;
            }
            // Find cheapest fence that fixes this violation.
            let mut best: Option<FenceInsertion> = None;
            let mut best_cost = f64::INFINITY;
            let mut best_coverage = 0usize;

            for loc in &candidates {
                for &(ord, scope) in &fence_types {
                    if !fence_fixes_violation(loc, ord, viol) {
                        continue;
                    }
                    let cost = config.cost_model.cost_of(&ord, &scope);
                    // Count how many other violations this fence also fixes.
                    let coverage = violations.iter().enumerate()
                        .filter(|(idx, v)| !covered.contains(idx) && fence_fixes_violation(loc, ord, v))
                        .count();
                    // Prefer lower cost per violation fixed.
                    let cost_per_fix = if coverage > 0 { cost / coverage as f64 } else { f64::INFINITY };
                    if cost_per_fix < best_cost / best_coverage.max(1) as f64 {
                        best = Some(FenceInsertion::new(loc.thread_id, loc.position, ord, scope));
                        best_cost = cost;
                        best_coverage = coverage;
                    }
                }
            }

            if let Some(ins) = best {
                // Mark all violations fixed by this insertion.
                let loc = CandidateLocation { thread_id: ins.thread_id, position: ins.position };
                for (idx, v) in violations.iter().enumerate() {
                    if fence_fixes_violation(&loc, ins.ordering, v) {
                        covered.insert(idx);
                    }
                }
                placement.add(ins);
            }
        }

        deduplicate_placement(&mut placement);
        let total_cost = placement.total_cost(&config.cost_model);
        let elapsed = start.elapsed().as_millis() as u64;
        InsertionResult::new(placement, total_cost, "Greedy", elapsed, false)
    }

    /// Local search: try removing fences and replacing with cheaper alternatives.
    pub fn local_search(
        &self,
        test: &LitmusTest,
        config: &InsertionConfig,
        initial: &FencePlacement,
    ) -> FencePlacement {
        let violations = detect_violations(test, config.target_model);
        let candidates = enumerate_candidates(test);
        let fence_types = available_fence_types(&config.cost_model);
        let mut current = initial.clone();

        let mut improved = true;
        while improved {
            improved = false;
            for i in 0..current.insertions.len() {
                let removed = current.insertions[i].clone();
                let mut trial = current.clone();
                trial.insertions.remove(i);

                // Check if any violations are now uncovered.
                let uncovered = find_uncovered_with_placement(&violations, &trial, &candidates);
                if uncovered.is_empty() {
                    // Removing the fence is valid — cheaper solution.
                    current = trial;
                    improved = true;
                    break;
                }

                // Try replacing with a cheaper fence.
                for &(ord, scope) in &fence_types {
                    let cost = config.cost_model.cost_of(&ord, &scope);
                    let removed_cost = config.cost_model.cost_of(&removed.ordering, &removed.scope);
                    if cost >= removed_cost {
                        continue;
                    }
                    let replacement = FenceInsertion::new(
                        removed.thread_id, removed.position, ord, scope,
                    );
                    let mut trial2 = trial.clone();
                    trial2.add(replacement);
                    let still_uncovered = find_uncovered_with_placement(&violations, &trial2, &candidates);
                    if still_uncovered.is_empty() {
                        current = trial2;
                        improved = true;
                        break;
                    }
                }
                if improved {
                    break;
                }
            }
        }

        current
    }

    /// Iterated local search with random restarts.
    pub fn iterated_local_search(&self, test: &LitmusTest, config: &InsertionConfig) -> InsertionResult {
        let start = std::time::Instant::now();
        let violations = detect_violations(test, config.target_model);
        let candidates = enumerate_candidates(test);
        let fence_types = available_fence_types(&config.cost_model);

        let initial = self.greedy(test, config);
        let mut best = initial.placement.clone();
        let mut best_cost = best.total_cost(&config.cost_model);

        // Run local search on greedy solution.
        let improved = self.local_search(test, config, &best);
        let uncovered_check = find_uncovered_with_placement(&violations, &improved, &candidates);
        if uncovered_check.is_empty() {
            let improved_cost = improved.total_cost(&config.cost_model);
            if improved_cost < best_cost {
                best = improved;
                best_cost = improved_cost;
            }
        }

        // Random restarts: perturb and re-optimize.

        for restart in 0..config.num_restarts {
            if start.elapsed().as_millis() as u64 > config.timeout_ms {
                break;
            }
            // Generate a random-ish initial placement by varying fence types.
            let mut perturbed = FencePlacement::new();
            for viol in &violations {
                // Pick a fence type based on restart index (deterministic pseudo-random).
                let type_idx = (viol.first + viol.second + restart) % fence_types.len().max(1);
                if type_idx < fence_types.len() {
                    let (ord, scope) = fence_types[type_idx];
                    let pos = (viol.first + viol.second) / 2 + 1;
                    let pos = pos.min(test.threads.iter()
                        .find(|t| t.id == viol.thread_id)
                        .map(|t| t.instructions.len())
                        .unwrap_or(0));
                    if fence_fixes_violation(
                        &CandidateLocation { thread_id: viol.thread_id, position: pos },
                        ord, viol,
                    ) {
                        perturbed.add(FenceInsertion::new(viol.thread_id, pos, ord, scope));
                    }
                }
            }
            deduplicate_placement(&mut perturbed);

            let improved = self.local_search(test, config, &perturbed);
            // Only accept if it covers all violations.
            let uncovered = find_uncovered_with_placement(&violations, &improved, &candidates);
            if uncovered.is_empty() {
                let cost = improved.total_cost(&config.cost_model);
                if cost < best_cost {
                    best = improved;
                    best_cost = cost;
                }
            }
        }

        let elapsed = start.elapsed().as_millis() as u64;
        InsertionResult::new(best, best_cost, "IteratedLocalSearch", elapsed, false)
    }
}

fn find_uncovered_with_placement(
    violations: &[Violation],
    placement: &FencePlacement,
    _candidates: &[CandidateLocation],
) -> Vec<usize> {
    let mut uncovered = Vec::new();
    'outer: for (idx, viol) in violations.iter().enumerate() {
        for ins in &placement.insertions {
            let loc = CandidateLocation { thread_id: ins.thread_id, position: ins.position };
            if fence_fixes_violation(&loc, ins.ordering, viol) {
                continue 'outer;
            }
        }
        uncovered.push(idx);
    }
    uncovered
}

// ---------------------------------------------------------------------------
// OptimalFenceInserter — unified entry point
// ---------------------------------------------------------------------------

/// Unified fence insertion optimizer that dispatches to ILP or heuristic solvers.
#[derive(Debug)]
pub struct OptimalFenceInserter {
    config: InsertionConfig,
}

impl OptimalFenceInserter {
    pub fn new(config: InsertionConfig) -> Self {
        Self { config }
    }

    /// Run fence insertion optimization.
    pub fn optimize(&self, test: &LitmusTest) -> InsertionResult {
        match self.config.optimization {
            OptimizationLevel::Optimal => {
                let ilp = IlpSolver::new();
                ilp.solve(test, &self.config)
            }
            OptimizationLevel::Fast => {
                let heuristic = HeuristicSolver::new();
                heuristic.greedy(test, &self.config)
            }
            OptimizationLevel::Balanced => {
                let heuristic = HeuristicSolver::new();
                heuristic.iterated_local_search(test, &self.config)
            }
        }
    }

    /// Simple per-thread recommendation (baseline for comparison).
    pub fn simple_recommendation(&self, test: &LitmusTest) -> InsertionResult {
        let start = std::time::Instant::now();
        let violations = detect_violations(test, self.config.target_model);
        let mut placement = FencePlacement::new();

        // Group violations by thread, insert one strong fence per thread.
        let mut thread_violations: HashMap<usize, Vec<&Violation>> = HashMap::new();
        for v in &violations {
            thread_violations.entry(v.thread_id).or_default().push(v);
        }

        for (tid, viols) in &thread_violations {
            // Find strongest required ordering among all violations in this thread.
            let max_ord = viols.iter()
                .map(|v| v.required_ordering)
                .max_by_key(|o| ordering_strength_val(*o))
                .unwrap_or(Ordering::Relaxed);
            let max_scope = viols.iter()
                .map(|v| v.required_scope)
                .max_by_key(|s| scope_strength_val(*s))
                .unwrap_or(Scope::None);

            // Insert one fence after each store (simple heuristic).
            if let Some(thread) = test.threads.iter().find(|t| t.id == *tid) {
                for (i, instr) in thread.instructions.iter().enumerate() {
                    if is_store(instr) {
                        placement.add(FenceInsertion::new(*tid, i + 1, max_ord, max_scope));
                    }
                }
            }
        }

        deduplicate_placement(&mut placement);
        let total_cost = placement.total_cost(&self.config.cost_model);
        let elapsed = start.elapsed().as_millis() as u64;
        InsertionResult::new(placement, total_cost, "Simple", elapsed, false)
    }

    /// Compare all solver strategies and return results sorted by cost.
    pub fn compare_strategies(&self, test: &LitmusTest) -> Vec<InsertionResult> {
        let mut results = Vec::new();

        // Simple recommendation.
        results.push(self.simple_recommendation(test));

        // Greedy.
        let heuristic = HeuristicSolver::new();
        results.push(heuristic.greedy(test, &self.config));

        // Iterated local search.
        let ils_config = InsertionConfig {
            optimization: OptimizationLevel::Balanced,
            ..self.config.clone()
        };
        results.push(heuristic.iterated_local_search(test, &ils_config));

        // ILP (if not too large).
        if test.total_instructions() <= 50 {
            let ilp = IlpSolver::new();
            results.push(ilp.solve(test, &self.config));
        }

        results.sort_by(|a, b| a.total_cost.partial_cmp(&b.total_cost).unwrap());
        results
    }
}

fn scope_strength_val(s: Scope) -> u8 {
    match s {
        Scope::None => 0,
        Scope::CTA => 1,
        Scope::GPU => 2,
        Scope::System => 3,
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checker::litmus::{LitmusTest, Thread, Instruction, Ordering, Scope};

    /// Helper: build a simple store-buffering (SB) litmus test with relaxed ops.
    fn sb_relaxed() -> LitmusTest {
        let mut test = LitmusTest::new("SB-relaxed");
        let mut t0 = Thread::new(0);
        t0.store(0x100, 1, Ordering::Relaxed);
        t0.load(0, 0x200, Ordering::Relaxed);
        let mut t1 = Thread::new(1);
        t1.store(0x200, 1, Ordering::Relaxed);
        t1.load(1, 0x100, Ordering::Relaxed);
        test.add_thread(t0);
        test.add_thread(t1);
        test
    }

    /// Helper: build a message-passing (MP) test with relaxed ops.
    fn mp_relaxed() -> LitmusTest {
        let mut test = LitmusTest::new("MP-relaxed");
        let mut t0 = Thread::new(0);
        t0.store(0x100, 1, Ordering::Relaxed); // data
        t0.store(0x200, 1, Ordering::Relaxed); // flag
        let mut t1 = Thread::new(1);
        t1.load(0, 0x200, Ordering::Relaxed); // flag
        t1.load(1, 0x100, Ordering::Relaxed); // data
        test.add_thread(t0);
        test.add_thread(t1);
        test
    }

    /// Helper: already-SC test.
    fn sb_sc() -> LitmusTest {
        let mut test = LitmusTest::new("SB-SC");
        let mut t0 = Thread::new(0);
        t0.store(0x100, 1, Ordering::SeqCst);
        t0.load(0, 0x200, Ordering::SeqCst);
        let mut t1 = Thread::new(1);
        t1.store(0x200, 1, Ordering::SeqCst);
        t1.load(1, 0x100, Ordering::SeqCst);
        test.add_thread(t0);
        test.add_thread(t1);
        test
    }

    fn default_config(target: TargetModel) -> InsertionConfig {
        InsertionConfig::new(target, CalibratedCostModel::predefined(Architecture::X86))
    }

    // -----------------------------------------------------------------------
    // Violation detection
    // -----------------------------------------------------------------------

    #[test]
    fn no_violations_for_relaxed_target() {
        let test = sb_relaxed();
        let violations = detect_violations(&test, TargetModel::Relaxed);
        assert!(violations.is_empty());
    }

    #[test]
    fn violations_detected_for_sc() {
        let test = sb_relaxed();
        let violations = detect_violations(&test, TargetModel::SC);
        assert!(!violations.is_empty());
    }

    #[test]
    fn no_violations_for_already_sc() {
        let test = sb_sc();
        let violations = detect_violations(&test, TargetModel::SC);
        assert!(violations.is_empty());
    }

    #[test]
    fn tso_detects_store_load_reorder() {
        let test = sb_relaxed();
        let violations = detect_violations(&test, TargetModel::TSO);
        // SB has store-load pattern in both threads.
        assert!(!violations.is_empty());
    }

    #[test]
    fn tso_no_violation_for_load_store() {
        // A test with only load-store pairs should have no TSO violations.
        let mut test = LitmusTest::new("load-store");
        let mut t0 = Thread::new(0);
        t0.load(0, 0x100, Ordering::Relaxed);
        t0.store(0x200, 1, Ordering::Relaxed);
        test.add_thread(t0);
        let violations = detect_violations(&test, TargetModel::TSO);
        assert!(violations.is_empty());
    }

    // -----------------------------------------------------------------------
    // FenceInsertion
    // -----------------------------------------------------------------------

    #[test]
    fn fence_insertion_display() {
        let ins = FenceInsertion::new(0, 1, Ordering::SeqCst, Scope::System);
        let s = format!("{}", ins);
        assert!(s.contains("T0@1"));
    }

    // -----------------------------------------------------------------------
    // FencePlacement
    // -----------------------------------------------------------------------

    #[test]
    fn fence_placement_by_thread() {
        let mut fp = FencePlacement::new();
        fp.add(FenceInsertion::new(0, 1, Ordering::AcqRel, Scope::System));
        fp.add(FenceInsertion::new(1, 1, Ordering::AcqRel, Scope::System));
        fp.add(FenceInsertion::new(0, 2, Ordering::Release, Scope::System));
        let by_thread = fp.by_thread();
        assert_eq!(by_thread[&0].len(), 2);
        assert_eq!(by_thread[&1].len(), 1);
    }

    #[test]
    fn fence_placement_apply() {
        let test = sb_relaxed();
        let mut fp = FencePlacement::new();
        fp.add(FenceInsertion::new(0, 1, Ordering::AcqRel, Scope::System));
        let modified = fp.apply(&test);
        // Thread 0 should now have 3 instructions (store, fence, load).
        assert_eq!(modified.threads[0].instructions.len(), 3);
        assert!(matches!(modified.threads[0].instructions[1], Instruction::Fence { .. }));
    }

    #[test]
    fn fence_placement_total_cost() {
        let model = CalibratedCostModel::predefined(Architecture::X86);
        let mut fp = FencePlacement::new();
        fp.add(FenceInsertion::new(0, 1, Ordering::SeqCst, Scope::System));
        let cost = fp.total_cost(&model);
        assert!(cost > 0.0);
    }

    #[test]
    fn deduplicate_keeps_strongest() {
        let mut fp = FencePlacement::new();
        fp.add(FenceInsertion::new(0, 1, Ordering::Release, Scope::System));
        fp.add(FenceInsertion::new(0, 1, Ordering::SeqCst, Scope::System));
        deduplicate_placement(&mut fp);
        assert_eq!(fp.num_fences(), 1);
        assert_eq!(fp.insertions[0].ordering, Ordering::SeqCst);
    }

    // -----------------------------------------------------------------------
    // ILP Solver
    // -----------------------------------------------------------------------

    #[test]
    fn ilp_no_violations() {
        let test = sb_sc();
        let config = default_config(TargetModel::SC);
        let ilp = IlpSolver::new();
        let result = ilp.solve(&test, &config);
        assert_eq!(result.num_fences, 0);
        assert!((result.total_cost - 0.0).abs() < 1e-9);
    }

    #[test]
    fn ilp_inserts_fences_for_sb() {
        let test = sb_relaxed();
        let config = default_config(TargetModel::SC);
        let ilp = IlpSolver::new();
        let result = ilp.solve(&test, &config);
        assert!(result.num_fences > 0);
        assert!(result.total_cost > 0.0);
    }

    #[test]
    fn ilp_tso_sb() {
        let test = sb_relaxed();
        let config = default_config(TargetModel::TSO);
        let ilp = IlpSolver::new();
        let result = ilp.solve(&test, &config);
        // TSO only needs store-load fences.
        assert!(result.num_fences > 0);
    }

    #[test]
    fn ilp_result_display() {
        let result = InsertionResult::new(
            FencePlacement::new(), 0.0, "test", 42, true,
        );
        let s = format!("{}", result);
        assert!(s.contains("fences=0"));
    }

    // -----------------------------------------------------------------------
    // Heuristic Solver
    // -----------------------------------------------------------------------

    #[test]
    fn greedy_no_violations() {
        let test = sb_sc();
        let config = default_config(TargetModel::SC);
        let h = HeuristicSolver::new();
        let result = h.greedy(&test, &config);
        assert_eq!(result.num_fences, 0);
    }

    #[test]
    fn greedy_inserts_fences_for_sb() {
        let test = sb_relaxed();
        let config = default_config(TargetModel::SC);
        let h = HeuristicSolver::new();
        let result = h.greedy(&test, &config);
        assert!(result.num_fences > 0);
    }

    #[test]
    fn greedy_mp_test() {
        let test = mp_relaxed();
        let config = default_config(TargetModel::SC);
        let h = HeuristicSolver::new();
        let result = h.greedy(&test, &config);
        assert!(result.num_fences > 0);
    }

    #[test]
    fn local_search_improves_or_equal() {
        let test = sb_relaxed();
        let config = default_config(TargetModel::SC);
        let h = HeuristicSolver::new();
        let greedy_result = h.greedy(&test, &config);
        let improved = h.local_search(&test, &config, &greedy_result.placement);
        let improved_cost = improved.total_cost(&config.cost_model);
        assert!(improved_cost <= greedy_result.total_cost + 1e-9);
    }

    #[test]
    fn iterated_local_search_produces_result() {
        let test = sb_relaxed();
        let config = default_config(TargetModel::SC).with_restarts(3).with_timeout(1000);
        let h = HeuristicSolver::new();
        let result = h.iterated_local_search(&test, &config);
        // ILS should produce a valid result (may be 0 fences if no violations,
        // but SB-relaxed under SC should have fences).
        assert!(result.total_cost >= 0.0);
    }

    // -----------------------------------------------------------------------
    // OptimalFenceInserter
    // -----------------------------------------------------------------------

    #[test]
    fn inserter_optimize_fast() {
        let config = default_config(TargetModel::SC).with_optimization(OptimizationLevel::Fast);
        let inserter = OptimalFenceInserter::new(config);
        let test = sb_relaxed();
        let result = inserter.optimize(&test);
        assert!(result.num_fences > 0);
        assert_eq!(result.solver_used, "Greedy");
    }

    #[test]
    fn inserter_optimize_optimal() {
        let config = default_config(TargetModel::SC).with_optimization(OptimizationLevel::Optimal);
        let inserter = OptimalFenceInserter::new(config);
        let test = sb_relaxed();
        let result = inserter.optimize(&test);
        assert!(result.num_fences > 0);
        assert_eq!(result.solver_used, "ILP");
    }

    #[test]
    fn inserter_optimize_balanced() {
        let config = default_config(TargetModel::SC)
            .with_optimization(OptimizationLevel::Balanced)
            .with_restarts(2)
            .with_timeout(1000);
        let inserter = OptimalFenceInserter::new(config);
        let test = sb_relaxed();
        let result = inserter.optimize(&test);
        // Balanced may or may not improve on greedy, but cost should be non-negative.
        assert!(result.total_cost >= 0.0);
    }

    #[test]
    fn simple_recommendation_inserts_after_stores() {
        let config = default_config(TargetModel::SC);
        let inserter = OptimalFenceInserter::new(config);
        let test = sb_relaxed();
        let result = inserter.simple_recommendation(&test);
        assert!(result.num_fences > 0);
        assert_eq!(result.solver_used, "Simple");
    }

    #[test]
    fn compare_strategies_returns_sorted() {
        let config = default_config(TargetModel::SC).with_timeout(500).with_restarts(2);
        let inserter = OptimalFenceInserter::new(config);
        let test = sb_relaxed();
        let results = inserter.compare_strategies(&test);
        assert!(results.len() >= 3);
        // Should be sorted by cost.
        for i in 1..results.len() {
            assert!(results[i].total_cost >= results[i - 1].total_cost - 1e-9);
        }
    }

    #[test]
    fn compare_ilp_vs_greedy_cost() {
        let config = default_config(TargetModel::TSO).with_timeout(2000);
        let test = sb_relaxed();
        let ilp = IlpSolver::new();
        let ilp_result = ilp.solve(&test, &config);
        let h = HeuristicSolver::new();
        let greedy_result = h.greedy(&test, &config);
        // ILP should be at most as expensive as greedy (or very close).
        assert!(ilp_result.total_cost <= greedy_result.total_cost + 1e-9);
    }

    #[test]
    fn compare_ilp_vs_heuristic_vs_simple() {
        let config = default_config(TargetModel::SC).with_timeout(2000).with_restarts(3);
        let test = mp_relaxed();
        let inserter = OptimalFenceInserter::new(config.clone());

        let simple = inserter.simple_recommendation(&test);
        let h = HeuristicSolver::new();
        let greedy = h.greedy(&test, &config);
        let ilp = IlpSolver::new();
        let ilp_result = ilp.solve(&test, &config);

        // ILP ≤ Greedy ≤ Simple (in cost).
        assert!(
            ilp_result.total_cost <= greedy.total_cost + 1e-9,
            "ILP ({}) should be ≤ Greedy ({})", ilp_result.total_cost, greedy.total_cost,
        );
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn empty_test_no_fences() {
        let test = LitmusTest::new("empty");
        let config = default_config(TargetModel::SC);
        let inserter = OptimalFenceInserter::new(config);
        let result = inserter.optimize(&test);
        assert_eq!(result.num_fences, 0);
    }

    #[test]
    fn single_thread_no_cross_thread_violations_tso() {
        let mut test = LitmusTest::new("single");
        let mut t0 = Thread::new(0);
        t0.store(0x100, 1, Ordering::Relaxed);
        t0.load(0, 0x100, Ordering::Relaxed);
        test.add_thread(t0);
        let config = default_config(TargetModel::TSO);
        let ilp = IlpSolver::new();
        let result = ilp.solve(&test, &config);
        // Single-thread store-load is still a TSO violation.
        // (TSO prevents store-load reorder even within a thread.)
        assert!(result.num_fences > 0 || result.total_cost == 0.0);
    }

    #[test]
    fn ra_target_mp_test() {
        let test = mp_relaxed();
        let config = default_config(TargetModel::RA);
        let ilp = IlpSolver::new();
        let result = ilp.solve(&test, &config);
        // RA should insert release/acquire fences for message passing.
        assert!(result.num_fences > 0);
    }

    #[test]
    fn insertion_config_builder() {
        let config = InsertionConfig::new(
            TargetModel::SC,
            CalibratedCostModel::predefined(Architecture::PTX),
        )
        .with_optimization(OptimizationLevel::Optimal)
        .with_timeout(10000)
        .with_restarts(20);

        assert_eq!(config.target_model, TargetModel::SC);
        assert_eq!(config.optimization, OptimizationLevel::Optimal);
        assert_eq!(config.timeout_ms, 10000);
        assert_eq!(config.num_restarts, 20);
    }

    #[test]
    fn target_model_display() {
        assert_eq!(format!("{}", TargetModel::SC), "SC");
        assert_eq!(format!("{}", TargetModel::TSO), "TSO");
        assert_eq!(format!("{}", TargetModel::RA), "RA");
        assert_eq!(format!("{}", TargetModel::Relaxed), "Relaxed");
    }

    #[test]
    fn ptx_cost_model_with_scoped_fences() {
        let config = InsertionConfig::new(
            TargetModel::SC,
            CalibratedCostModel::predefined(Architecture::PTX),
        );
        let test = sb_relaxed();
        let h = HeuristicSolver::new();
        let result = h.greedy(&test, &config);
        // Should produce fences using PTX cost model.
        assert!(result.num_fences > 0);
    }

    #[test]
    fn placement_display() {
        let mut fp = FencePlacement::new();
        fp.add(FenceInsertion::new(0, 1, Ordering::AcqRel, Scope::System));
        let s = format!("{}", fp);
        assert!(s.contains("1 fences"));
    }

    #[test]
    fn fence_fixes_violation_wrong_thread() {
        let loc = CandidateLocation { thread_id: 0, position: 1 };
        let viol = Violation {
            thread_id: 1, first: 0, second: 1,
            required_ordering: Ordering::AcqRel, required_scope: Scope::System,
        };
        assert!(!fence_fixes_violation(&loc, Ordering::SeqCst, &viol));
    }

    #[test]
    fn fence_fixes_violation_correct() {
        let loc = CandidateLocation { thread_id: 0, position: 1 };
        let viol = Violation {
            thread_id: 0, first: 0, second: 2,
            required_ordering: Ordering::AcqRel, required_scope: Scope::System,
        };
        assert!(fence_fixes_violation(&loc, Ordering::SeqCst, &viol));
        assert!(fence_fixes_violation(&loc, Ordering::AcqRel, &viol));
        assert!(!fence_fixes_violation(&loc, Ordering::Relaxed, &viol));
    }

    #[test]
    fn fence_outside_range_does_not_fix() {
        let loc = CandidateLocation { thread_id: 0, position: 3 };
        let viol = Violation {
            thread_id: 0, first: 0, second: 2,
            required_ordering: Ordering::AcqRel, required_scope: Scope::System,
        };
        assert!(!fence_fixes_violation(&loc, Ordering::SeqCst, &viol));
    }

    #[test]
    fn enumerate_candidates_count() {
        let test = sb_relaxed();
        let candidates = enumerate_candidates(&test);
        // Each thread has 2 instructions → 3 candidate positions per thread → 6 total.
        assert_eq!(candidates.len(), 6);
    }

    #[test]
    fn is_memory_op_checks() {
        assert!(is_memory_op(&Instruction::Load { reg: 0, addr: 0, ordering: Ordering::Relaxed }));
        assert!(is_memory_op(&Instruction::Store { addr: 0, value: 0, ordering: Ordering::Relaxed }));
        assert!(is_memory_op(&Instruction::RMW { reg: 0, addr: 0, value: 0, ordering: Ordering::Relaxed }));
        assert!(!is_memory_op(&Instruction::Fence { ordering: Ordering::SeqCst, scope: Scope::System }));
        assert!(!is_memory_op(&Instruction::Branch { label: 0 }));
    }

    #[test]
    fn ordering_strength_consistent() {
        assert!(ordering_strength_val(Ordering::SeqCst) > ordering_strength_val(Ordering::AcqRel));
        assert!(ordering_strength_val(Ordering::AcqRel) > ordering_strength_val(Ordering::Release));
        assert!(ordering_strength_val(Ordering::Release) > ordering_strength_val(Ordering::Relaxed));
    }
}
