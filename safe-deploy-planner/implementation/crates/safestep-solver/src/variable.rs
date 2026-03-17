// SAT variable, literal, assignment, and variable management types.

use smallvec::SmallVec;
use std::collections::HashMap;
use std::fmt;

// ── Variable ──────────────────────────────────────────────────────────────────

/// A SAT variable represented as a positive integer index (1-based).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Variable(pub u32);

impl Variable {
    /// Create a new variable with the given 1-based index.
    #[inline]
    pub fn new(index: u32) -> Self {
        debug_assert!(index > 0, "Variable index must be positive");
        Variable(index)
    }

    /// Return the 1-based index of this variable.
    #[inline]
    pub fn index(self) -> u32 {
        self.0
    }

    /// Return the 0-based array index for this variable.
    #[inline]
    pub fn array_index(self) -> usize {
        (self.0 - 1) as usize
    }

    /// Create the positive literal of this variable.
    #[inline]
    pub fn positive(self) -> Literal {
        Literal::positive(self)
    }

    /// Create the negative literal of this variable.
    #[inline]
    pub fn negative(self) -> Literal {
        Literal::negative(self)
    }
}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "x{}", self.0)
    }
}

// ── Literal ───────────────────────────────────────────────────────────────────

/// A literal is a variable with a polarity. Encoded as u32: var * 2 + (1 if negative, 0 if positive).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Literal(pub u32);

impl Literal {
    /// Create a positive literal for the given variable.
    #[inline]
    pub fn positive(var: Variable) -> Self {
        Literal(var.0 * 2)
    }

    /// Create a negative literal for the given variable.
    #[inline]
    pub fn negative(var: Variable) -> Self {
        Literal(var.0 * 2 + 1)
    }

    /// Create a literal from variable index and sign. sign=true means positive.
    #[inline]
    pub fn from_dimacs(dimacs: i32) -> Self {
        debug_assert!(dimacs != 0, "DIMACS literal cannot be zero");
        if dimacs > 0 {
            Literal::positive(Variable::new(dimacs as u32))
        } else {
            Literal::negative(Variable::new((-dimacs) as u32))
        }
    }

    /// Convert to DIMACS format (positive or negative integer).
    #[inline]
    pub fn to_dimacs(self) -> i32 {
        let var = self.var().0 as i32;
        if self.is_positive() {
            var
        } else {
            -var
        }
    }

    /// Get the variable of this literal.
    #[inline]
    pub fn var(self) -> Variable {
        Variable(self.0 >> 1)
    }

    /// Whether this literal is positive.
    #[inline]
    pub fn is_positive(self) -> bool {
        self.0 & 1 == 0
    }

    /// Whether this literal is negative.
    #[inline]
    pub fn is_negative(self) -> bool {
        self.0 & 1 == 1
    }

    /// Return the negation of this literal.
    #[inline]
    pub fn negated(self) -> Literal {
        Literal(self.0 ^ 1)
    }

    /// The polarity of this literal (true = positive).
    #[inline]
    pub fn polarity(self) -> bool {
        self.is_positive()
    }

    /// Internal code used for indexing watch lists etc.
    #[inline]
    pub fn code(self) -> usize {
        self.0 as usize
    }

    /// Create from internal code.
    #[inline]
    pub fn from_code(code: u32) -> Self {
        Literal(code)
    }

    /// Evaluate this literal under a given truth value for its variable.
    #[inline]
    pub fn eval(self, var_value: bool) -> bool {
        if self.is_positive() {
            var_value
        } else {
            !var_value
        }
    }
}

impl fmt::Debug for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_positive() {
            write!(f, "+x{}", self.var().0)
        } else {
            write!(f, "-x{}", self.var().0)
        }
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_positive() {
            write!(f, "x{}", self.var().0)
        } else {
            write!(f, "¬x{}", self.var().0)
        }
    }
}

// ── LiteralVec ────────────────────────────────────────────────────────────────

/// SmallVec-based literal storage (inline for small clauses).
pub type LiteralVec = SmallVec<[Literal; 4]>;

/// Helper to create a LiteralVec from a slice.
pub fn lit_vec(lits: &[Literal]) -> LiteralVec {
    SmallVec::from_slice(lits)
}

// ── VarLabel ──────────────────────────────────────────────────────────────────

/// Human-readable label for a variable, used for debugging.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarLabel {
    pub name: String,
    pub metadata: Option<String>,
}

impl VarLabel {
    pub fn new(name: impl Into<String>) -> Self {
        VarLabel {
            name: name.into(),
            metadata: None,
        }
    }

    pub fn with_metadata(name: impl Into<String>, meta: impl Into<String>) -> Self {
        VarLabel {
            name: name.into(),
            metadata: Some(meta.into()),
        }
    }
}

impl fmt::Display for VarLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)?;
        if let Some(ref m) = self.metadata {
            write!(f, " ({})", m)?;
        }
        Ok(())
    }
}

// ── VariableManager ───────────────────────────────────────────────────────────

/// Allocates and tracks variables.
#[derive(Debug, Clone)]
pub struct VariableManager {
    next_index: u32,
    labels: HashMap<Variable, VarLabel>,
    name_to_var: HashMap<String, Variable>,
}

impl VariableManager {
    /// Create a new variable manager.
    pub fn new() -> Self {
        VariableManager {
            next_index: 1,
            labels: HashMap::new(),
            name_to_var: HashMap::new(),
        }
    }

    /// Allocate a new unnamed variable.
    pub fn new_variable(&mut self) -> Variable {
        let var = Variable::new(self.next_index);
        self.next_index += 1;
        var
    }

    /// Allocate a named variable.
    pub fn new_named_variable(&mut self, name: impl Into<String>) -> Variable {
        let var = self.new_variable();
        let name_str: String = name.into();
        self.name_to_var.insert(name_str.clone(), var);
        self.labels.insert(var, VarLabel::new(name_str));
        var
    }

    /// Allocate `n` new unnamed variables.
    pub fn new_variables(&mut self, n: u32) -> Vec<Variable> {
        (0..n).map(|_| self.new_variable()).collect()
    }

    /// Return the total number of allocated variables.
    pub fn variable_count(&self) -> u32 {
        self.next_index - 1
    }

    /// Get the label for a variable, if any.
    pub fn label(&self, var: Variable) -> Option<&VarLabel> {
        self.labels.get(&var)
    }

    /// Set or update the label for a variable.
    pub fn set_label(&mut self, var: Variable, label: VarLabel) {
        let name = label.name.clone();
        self.labels.insert(var, label);
        self.name_to_var.insert(name, var);
    }

    /// Look up a variable by name.
    pub fn var_by_name(&self, name: &str) -> Option<Variable> {
        self.name_to_var.get(name).copied()
    }

    /// Get all variable-label pairs.
    pub fn all_labels(&self) -> impl Iterator<Item = (Variable, &VarLabel)> {
        self.labels.iter().map(|(&v, l)| (v, l))
    }

    /// Check if a variable index is valid (allocated).
    pub fn is_valid(&self, var: Variable) -> bool {
        var.0 >= 1 && var.0 < self.next_index
    }

    /// Get the maximum variable code (for sizing arrays indexed by literal codes).
    pub fn max_literal_code(&self) -> usize {
        (self.next_index * 2) as usize
    }
}

impl Default for VariableManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── DecisionLevel ─────────────────────────────────────────────────────────────

/// Decision level for assignments. Level 0 means top-level / unit propagation before any decision.
pub type DecisionLevel = u32;

// ── Reason ────────────────────────────────────────────────────────────────────

/// Why a literal was assigned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    /// Made as a decision.
    Decision,
    /// Propagated from the given clause.
    Propagation(u32), // ClauseId index
    /// Assumed (external assumption).
    Assumption,
}

// ── Assignment ────────────────────────────────────────────────────────────────

/// A partial or total variable assignment.
#[derive(Debug, Clone)]
pub struct Assignment {
    /// Per-variable truth value: None = unassigned.
    values: Vec<Option<bool>>,
    /// Per-variable decision level.
    levels: Vec<DecisionLevel>,
    /// Per-variable reason.
    reasons: Vec<Reason>,
    /// Number of assigned variables.
    num_assigned: usize,
    /// Total number of variables.
    num_vars: usize,
}

impl Assignment {
    /// Create a new empty assignment for `num_vars` variables.
    pub fn new(num_vars: usize) -> Self {
        Assignment {
            values: vec![None; num_vars],
            levels: vec![0; num_vars],
            reasons: vec![Reason::Decision; num_vars],
            num_assigned: 0,
            num_vars,
        }
    }

    /// Get the truth value of a variable (None if unassigned).
    #[inline]
    pub fn get(&self, var: Variable) -> Option<bool> {
        let idx = var.array_index();
        if idx < self.values.len() {
            self.values[idx]
        } else {
            None
        }
    }

    /// Set the truth value of a variable.
    pub fn set(&mut self, var: Variable, value: bool, level: DecisionLevel, reason: Reason) {
        let idx = var.array_index();
        if idx >= self.values.len() {
            self.resize(idx + 1);
        }
        if self.values[idx].is_none() {
            self.num_assigned += 1;
        }
        self.values[idx] = Some(value);
        self.levels[idx] = level;
        self.reasons[idx] = reason;
    }

    /// Unset a variable's assignment.
    pub fn unset(&mut self, var: Variable) {
        let idx = var.array_index();
        if idx < self.values.len() && self.values[idx].is_some() {
            self.values[idx] = None;
            self.levels[idx] = 0;
            self.reasons[idx] = Reason::Decision;
            self.num_assigned -= 1;
        }
    }

    /// Flip the truth value of an assigned variable.
    pub fn flip(&mut self, var: Variable) {
        let idx = var.array_index();
        if let Some(ref mut v) = self.values[idx] {
            *v = !*v;
        }
    }

    /// Get the decision level of a variable's assignment.
    pub fn level(&self, var: Variable) -> DecisionLevel {
        let idx = var.array_index();
        if idx < self.levels.len() {
            self.levels[idx]
        } else {
            0
        }
    }

    /// Get the reason for a variable's assignment.
    pub fn reason(&self, var: Variable) -> Reason {
        let idx = var.array_index();
        if idx < self.reasons.len() {
            self.reasons[idx]
        } else {
            Reason::Decision
        }
    }

    /// Evaluate a literal under this assignment (None if variable unassigned).
    #[inline]
    pub fn eval_literal(&self, lit: Literal) -> Option<bool> {
        self.get(lit.var()).map(|v| lit.eval(v))
    }

    /// Whether all variables are assigned.
    pub fn is_complete(&self) -> bool {
        self.num_assigned == self.num_vars
    }

    /// Return the number of assigned variables.
    pub fn num_assigned(&self) -> usize {
        self.num_assigned
    }

    /// Return iterator over unassigned variables (1-based).
    pub fn unassigned_variables(&self) -> Vec<Variable> {
        self.values
            .iter()
            .enumerate()
            .filter_map(|(i, v)| {
                if v.is_none() {
                    Some(Variable::new((i + 1) as u32))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Number of variables in this assignment.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Grow the assignment to accommodate more variables.
    pub fn resize(&mut self, new_size: usize) {
        if new_size > self.values.len() {
            self.values.resize(new_size, None);
            self.levels.resize(new_size, 0);
            self.reasons.resize(new_size, Reason::Decision);
        }
        if new_size > self.num_vars {
            self.num_vars = new_size;
        }
    }

    /// Extract the assignment as a mapping from variable to bool (only assigned vars).
    pub fn to_map(&self) -> HashMap<Variable, bool> {
        let mut map = HashMap::new();
        for (i, v) in self.values.iter().enumerate() {
            if let Some(val) = v {
                map.insert(Variable::new((i + 1) as u32), *val);
            }
        }
        map
    }

    /// Clone the values only (not levels/reasons).
    pub fn clone_values(&self) -> Vec<Option<bool>> {
        self.values.clone()
    }

    /// Reset all assignments.
    pub fn clear(&mut self) {
        for v in self.values.iter_mut() {
            *v = None;
        }
        for l in self.levels.iter_mut() {
            *l = 0;
        }
        for r in self.reasons.iter_mut() {
            *r = Reason::Decision;
        }
        self.num_assigned = 0;
    }
}

// ── Phase Saver ───────────────────────────────────────────────────────────────

/// Saves the last polarity used for each variable (phase saving heuristic).
#[derive(Debug, Clone)]
pub struct PhaseSaver {
    phases: Vec<bool>,
}

impl PhaseSaver {
    pub fn new(num_vars: usize) -> Self {
        PhaseSaver {
            phases: vec![false; num_vars],
        }
    }

    /// Save the phase of a variable.
    pub fn save(&mut self, var: Variable, positive: bool) {
        let idx = var.array_index();
        if idx >= self.phases.len() {
            self.phases.resize(idx + 1, false);
        }
        self.phases[idx] = positive;
    }

    /// Get the saved phase of a variable.
    pub fn get(&self, var: Variable) -> bool {
        let idx = var.array_index();
        if idx < self.phases.len() {
            self.phases[idx]
        } else {
            false
        }
    }

    /// Resize to accommodate more variables.
    pub fn resize(&mut self, n: usize) {
        if n > self.phases.len() {
            self.phases.resize(n, false);
        }
    }
}

// ── VSIDS Activity ────────────────────────────────────────────────────────────

/// VSIDS variable activity scores for variable selection heuristic.
#[derive(Debug, Clone)]
pub struct VsidsActivity {
    activities: Vec<f64>,
    increment: f64,
    decay: f64,
    /// Heap order for variable selection (binary heap, max-activity at top).
    heap: Vec<Variable>,
    /// Position of each variable in the heap (-1 if not in heap).
    heap_pos: Vec<i32>,
}

impl VsidsActivity {
    pub fn new(num_vars: usize, decay: f64, increment: f64) -> Self {
        let mut heap = Vec::with_capacity(num_vars);
        let mut heap_pos = vec![-1i32; num_vars];
        for i in 0..num_vars {
            let var = Variable::new((i + 1) as u32);
            heap.push(var);
            heap_pos[i] = i as i32;
        }
        VsidsActivity {
            activities: vec![0.0; num_vars],
            increment,
            decay,
            heap,
            heap_pos,
        }
    }

    /// Bump the activity of a variable.
    pub fn bump(&mut self, var: Variable) {
        let idx = var.array_index();
        if idx >= self.activities.len() {
            self.resize(idx + 1);
        }
        self.activities[idx] += self.increment;

        // Rescale if activities get too large.
        if self.activities[idx] > 1e100 {
            for a in self.activities.iter_mut() {
                *a *= 1e-100;
            }
            self.increment *= 1e-100;
        }

        // Percolate up in the heap.
        if self.heap_pos[idx] >= 0 {
            self.percolate_up(self.heap_pos[idx] as usize);
        }
    }

    /// Apply decay to the increment.
    pub fn decay(&mut self) {
        self.increment /= self.decay;
    }

    /// Get activity of a variable.
    pub fn activity(&self, var: Variable) -> f64 {
        let idx = var.array_index();
        if idx < self.activities.len() {
            self.activities[idx]
        } else {
            0.0
        }
    }

    /// Insert a variable into the heap (if not already present).
    pub fn insert(&mut self, var: Variable) {
        let idx = var.array_index();
        if idx >= self.heap_pos.len() {
            self.resize(idx + 1);
        }
        if self.heap_pos[idx] < 0 {
            let pos = self.heap.len();
            self.heap.push(var);
            self.heap_pos[idx] = pos as i32;
            self.percolate_up(pos);
        }
    }

    /// Remove and return the variable with highest activity.
    pub fn pop_max(&mut self) -> Option<Variable> {
        if self.heap.is_empty() {
            return None;
        }
        let top = self.heap[0];
        let top_idx = top.array_index();
        let last = self.heap.len() - 1;
        self.swap_heap(0, last);
        self.heap.pop();
        self.heap_pos[top_idx] = -1;
        if !self.heap.is_empty() {
            self.percolate_down(0);
        }
        Some(top)
    }

    /// Check if the heap is empty.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Resize to accommodate more variables.
    pub fn resize(&mut self, new_size: usize) {
        while self.activities.len() < new_size {
            self.activities.push(0.0);
        }
        while self.heap_pos.len() < new_size {
            self.heap_pos.push(-1);
        }
    }

    fn percolate_up(&mut self, mut pos: usize) {
        while pos > 0 {
            let parent = (pos - 1) / 2;
            let pos_var = self.heap[pos];
            let par_var = self.heap[parent];
            if self.activities[pos_var.array_index()] > self.activities[par_var.array_index()] {
                self.swap_heap(pos, parent);
                pos = parent;
            } else {
                break;
            }
        }
    }

    fn percolate_down(&mut self, mut pos: usize) {
        let len = self.heap.len();
        loop {
            let left = 2 * pos + 1;
            let right = 2 * pos + 2;
            let mut largest = pos;

            if left < len
                && self.activities[self.heap[left].array_index()]
                    > self.activities[self.heap[largest].array_index()]
            {
                largest = left;
            }
            if right < len
                && self.activities[self.heap[right].array_index()]
                    > self.activities[self.heap[largest].array_index()]
            {
                largest = right;
            }
            if largest != pos {
                self.swap_heap(pos, largest);
                pos = largest;
            } else {
                break;
            }
        }
    }

    fn swap_heap(&mut self, a: usize, b: usize) {
        self.heap.swap(a, b);
        let va = self.heap[a].array_index();
        let vb = self.heap[b].array_index();
        self.heap_pos[va] = a as i32;
        self.heap_pos[vb] = b as i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_basic() {
        let v = Variable::new(1);
        assert_eq!(v.index(), 1);
        assert_eq!(v.array_index(), 0);
        assert_eq!(format!("{}", v), "x1");
    }

    #[test]
    fn test_literal_positive() {
        let v = Variable::new(3);
        let lit = Literal::positive(v);
        assert!(lit.is_positive());
        assert!(!lit.is_negative());
        assert_eq!(lit.var(), v);
        assert_eq!(lit.to_dimacs(), 3);
    }

    #[test]
    fn test_literal_negative() {
        let v = Variable::new(5);
        let lit = Literal::negative(v);
        assert!(lit.is_negative());
        assert!(!lit.is_positive());
        assert_eq!(lit.var(), v);
        assert_eq!(lit.to_dimacs(), -5);
    }

    #[test]
    fn test_literal_negation() {
        let lit = Literal::from_dimacs(3);
        let neg = lit.negated();
        assert_eq!(neg.to_dimacs(), -3);
        assert_eq!(neg.negated(), lit);
    }

    #[test]
    fn test_literal_from_dimacs() {
        let pos = Literal::from_dimacs(7);
        assert!(pos.is_positive());
        assert_eq!(pos.var(), Variable::new(7));

        let neg = Literal::from_dimacs(-4);
        assert!(neg.is_negative());
        assert_eq!(neg.var(), Variable::new(4));
    }

    #[test]
    fn test_literal_eval() {
        let pos = Literal::positive(Variable::new(1));
        assert!(pos.eval(true));
        assert!(!pos.eval(false));

        let neg = Literal::negative(Variable::new(1));
        assert!(!neg.eval(true));
        assert!(neg.eval(false));
    }

    #[test]
    fn test_variable_manager() {
        let mut mgr = VariableManager::new();
        assert_eq!(mgr.variable_count(), 0);

        let v1 = mgr.new_variable();
        assert_eq!(v1, Variable::new(1));
        assert_eq!(mgr.variable_count(), 1);

        let v2 = mgr.new_named_variable("deploy_A");
        assert_eq!(v2, Variable::new(2));
        assert_eq!(mgr.label(v2).unwrap().name, "deploy_A");
        assert_eq!(mgr.var_by_name("deploy_A"), Some(v2));
    }

    #[test]
    fn test_variable_manager_batch() {
        let mut mgr = VariableManager::new();
        let vars = mgr.new_variables(5);
        assert_eq!(vars.len(), 5);
        assert_eq!(mgr.variable_count(), 5);
        assert_eq!(vars[0], Variable::new(1));
        assert_eq!(vars[4], Variable::new(5));
    }

    #[test]
    fn test_assignment_basic() {
        let mut asgn = Assignment::new(3);
        assert!(asgn.get(Variable::new(1)).is_none());
        assert!(!asgn.is_complete());

        asgn.set(Variable::new(1), true, 0, Reason::Decision);
        assert_eq!(asgn.get(Variable::new(1)), Some(true));
        assert_eq!(asgn.num_assigned(), 1);

        asgn.set(Variable::new(2), false, 1, Reason::Propagation(0));
        asgn.set(Variable::new(3), true, 1, Reason::Propagation(1));
        assert!(asgn.is_complete());
        assert_eq!(asgn.level(Variable::new(2)), 1);
    }

    #[test]
    fn test_assignment_unset() {
        let mut asgn = Assignment::new(2);
        asgn.set(Variable::new(1), true, 0, Reason::Decision);
        assert_eq!(asgn.num_assigned(), 1);
        asgn.unset(Variable::new(1));
        assert_eq!(asgn.num_assigned(), 0);
        assert!(asgn.get(Variable::new(1)).is_none());
    }

    #[test]
    fn test_assignment_flip() {
        let mut asgn = Assignment::new(1);
        asgn.set(Variable::new(1), true, 0, Reason::Decision);
        assert_eq!(asgn.get(Variable::new(1)), Some(true));
        asgn.flip(Variable::new(1));
        assert_eq!(asgn.get(Variable::new(1)), Some(false));
    }

    #[test]
    fn test_assignment_eval_literal() {
        let mut asgn = Assignment::new(2);
        asgn.set(Variable::new(1), true, 0, Reason::Decision);
        let pos = Literal::positive(Variable::new(1));
        let neg = Literal::negative(Variable::new(1));
        assert_eq!(asgn.eval_literal(pos), Some(true));
        assert_eq!(asgn.eval_literal(neg), Some(false));
        assert_eq!(
            asgn.eval_literal(Literal::positive(Variable::new(2))),
            None
        );
    }

    #[test]
    fn test_assignment_unassigned_variables() {
        let mut asgn = Assignment::new(4);
        asgn.set(Variable::new(1), true, 0, Reason::Decision);
        asgn.set(Variable::new(3), false, 0, Reason::Decision);
        let unassigned = asgn.unassigned_variables();
        assert_eq!(unassigned, vec![Variable::new(2), Variable::new(4)]);
    }

    #[test]
    fn test_assignment_to_map() {
        let mut asgn = Assignment::new(3);
        asgn.set(Variable::new(1), true, 0, Reason::Decision);
        asgn.set(Variable::new(3), false, 0, Reason::Decision);
        let map = asgn.to_map();
        assert_eq!(map.len(), 2);
        assert_eq!(map[&Variable::new(1)], true);
        assert_eq!(map[&Variable::new(3)], false);
    }

    #[test]
    fn test_phase_saver() {
        let mut ps = PhaseSaver::new(3);
        assert!(!ps.get(Variable::new(1)));
        ps.save(Variable::new(1), true);
        assert!(ps.get(Variable::new(1)));
        ps.save(Variable::new(1), false);
        assert!(!ps.get(Variable::new(1)));
    }

    #[test]
    fn test_vsids_activity() {
        let mut act = VsidsActivity::new(3, 0.95, 1.0);
        act.bump(Variable::new(1));
        act.bump(Variable::new(1));
        act.bump(Variable::new(2));
        assert!(act.activity(Variable::new(1)) > act.activity(Variable::new(2)));

        let top = act.pop_max().unwrap();
        assert_eq!(top, Variable::new(1));
    }

    #[test]
    fn test_vsids_decay() {
        let mut act = VsidsActivity::new(2, 0.5, 1.0);
        act.bump(Variable::new(1)); // activity = 1.0
        act.decay(); // increment becomes 2.0
        act.bump(Variable::new(2)); // activity = 2.0
        assert!(act.activity(Variable::new(2)) > act.activity(Variable::new(1)));
    }

    #[test]
    fn test_vsids_insert_after_pop() {
        let mut act = VsidsActivity::new(2, 0.95, 1.0);
        act.bump(Variable::new(1));
        let v = act.pop_max().unwrap();
        assert_eq!(v, Variable::new(1));
        act.insert(Variable::new(1));
        let v2 = act.pop_max().unwrap();
        assert_eq!(v2, Variable::new(1));
    }

    #[test]
    fn test_literal_code_roundtrip() {
        let lit = Literal::from_dimacs(-5);
        let code = lit.code();
        let lit2 = Literal::from_code(code as u32);
        assert_eq!(lit, lit2);
    }

    #[test]
    fn test_lit_vec() {
        let v1 = Variable::new(1);
        let v2 = Variable::new(2);
        let lv = lit_vec(&[v1.positive(), v2.negative()]);
        assert_eq!(lv.len(), 2);
        assert!(lv[0].is_positive());
        assert!(lv[1].is_negative());
    }

    #[test]
    fn test_variable_manager_is_valid() {
        let mut mgr = VariableManager::new();
        let v = mgr.new_variable();
        assert!(mgr.is_valid(v));
        assert!(!mgr.is_valid(Variable::new(99)));
    }

    #[test]
    fn test_assignment_clear() {
        let mut asgn = Assignment::new(3);
        asgn.set(Variable::new(1), true, 0, Reason::Decision);
        asgn.set(Variable::new(2), false, 1, Reason::Propagation(0));
        assert_eq!(asgn.num_assigned(), 2);
        asgn.clear();
        assert_eq!(asgn.num_assigned(), 0);
        assert!(asgn.get(Variable::new(1)).is_none());
    }
}
