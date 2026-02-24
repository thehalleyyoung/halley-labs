#[allow(unused)]

//! Dataflow analysis for LITMUS∞.
//!
//! Implements reaching definitions, live variable analysis, available expressions,
//! very busy expressions, constant propagation, and interval analysis for
//! litmus test programs. Supports both forward and backward analysis with
//! configurable lattices and transfer functions.

use std::collections::{HashMap, HashSet, BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;

// ═══════════════════════════════════════════════════════════════════════════
// Lattice Framework
// ═══════════════════════════════════════════════════════════════════════════

/// A complete lattice for dataflow analysis.
pub trait Lattice: Clone + Eq + fmt::Debug {
    /// The bottom element (⊥).
    fn bottom() -> Self;
    /// The top element (⊤).
    fn top() -> Self;
    /// Join (least upper bound).
    fn join(&self, other: &Self) -> Self;
    /// Meet (greatest lower bound).
    fn meet(&self, other: &Self) -> Self;
    /// Partial order: self ⊑ other.
    fn leq(&self, other: &Self) -> bool;
    /// Widening operator (default: join).
    fn widen(&self, other: &Self) -> Self {
        self.join(other)
    }
    /// Narrowing operator (default: meet).
    fn narrow(&self, other: &Self) -> Self {
        self.meet(other)
    }
}

/// A powerset lattice over a finite domain.
#[derive(Clone, PartialEq, Eq)]
pub struct PowersetLattice<T: Clone + Eq + Hash> {
    pub elements: HashSet<T>,
    pub universe: HashSet<T>,
}

impl<T: Clone + Eq + Hash + fmt::Debug> fmt::Debug for PowersetLattice<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PowersetLattice({:?})", self.elements)
    }
}

impl<T: Clone + Eq + Hash + fmt::Debug> Lattice for PowersetLattice<T> {
    fn bottom() -> Self {
        PowersetLattice {
            elements: HashSet::new(),
            universe: HashSet::new(),
        }
    }

    fn top() -> Self {
        PowersetLattice {
            elements: HashSet::new(),
            universe: HashSet::new(),
        }
    }

    fn join(&self, other: &Self) -> Self {
        PowersetLattice {
            elements: self.elements.union(&other.elements).cloned().collect(),
            universe: self.universe.union(&other.universe).cloned().collect(),
        }
    }

    fn meet(&self, other: &Self) -> Self {
        PowersetLattice {
            elements: self.elements.intersection(&other.elements).cloned().collect(),
            universe: self.universe.intersection(&other.universe).cloned().collect(),
        }
    }

    fn leq(&self, other: &Self) -> bool {
        self.elements.is_subset(&other.elements)
    }
}

/// An interval lattice [lo, hi] for integer analysis.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IntervalLattice {
    Bottom,
    Interval { lo: i64, hi: i64 },
    Top,
}

impl Lattice for IntervalLattice {
    fn bottom() -> Self {
        IntervalLattice::Bottom
    }

    fn top() -> Self {
        IntervalLattice::Top
    }

    fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (IntervalLattice::Bottom, x) | (x, IntervalLattice::Bottom) => x.clone(),
            (IntervalLattice::Top, _) | (_, IntervalLattice::Top) => IntervalLattice::Top,
            (IntervalLattice::Interval { lo: a, hi: b }, IntervalLattice::Interval { lo: c, hi: d }) => {
                IntervalLattice::Interval {
                    lo: (*a).min(*c),
                    hi: (*b).max(*d),
                }
            }
        }
    }

    fn meet(&self, other: &Self) -> Self {
        match (self, other) {
            (IntervalLattice::Bottom, _) | (_, IntervalLattice::Bottom) => IntervalLattice::Bottom,
            (IntervalLattice::Top, x) | (x, IntervalLattice::Top) => x.clone(),
            (IntervalLattice::Interval { lo: a, hi: b }, IntervalLattice::Interval { lo: c, hi: d }) => {
                let lo = (*a).max(*c);
                let hi = (*b).min(*d);
                if lo <= hi {
                    IntervalLattice::Interval { lo, hi }
                } else {
                    IntervalLattice::Bottom
                }
            }
        }
    }

    fn leq(&self, other: &Self) -> bool {
        match (self, other) {
            (IntervalLattice::Bottom, _) => true,
            (_, IntervalLattice::Top) => true,
            (IntervalLattice::Top, _) => false,
            (_, IntervalLattice::Bottom) => false,
            (IntervalLattice::Interval { lo: a, hi: b }, IntervalLattice::Interval { lo: c, hi: d }) => {
                c <= a && b <= d
            }
        }
    }

    fn widen(&self, other: &Self) -> Self {
        match (self, other) {
            (IntervalLattice::Bottom, x) | (x, IntervalLattice::Bottom) => x.clone(),
            (IntervalLattice::Top, _) | (_, IntervalLattice::Top) => IntervalLattice::Top,
            (IntervalLattice::Interval { lo: a, hi: b }, IntervalLattice::Interval { lo: c, hi: d }) => {
                let lo = if c < a { i64::MIN } else { *a };
                let hi = if d > b { i64::MAX } else { *b };
                if lo == i64::MIN && hi == i64::MAX {
                    IntervalLattice::Top
                } else {
                    IntervalLattice::Interval { lo, hi }
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Control Flow Graph
// ═══════════════════════════════════════════════════════════════════════════

/// Unique identifier for a basic block.
pub type BlockId = usize;

/// A basic block in the control flow graph.
#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub id: BlockId,
    pub instructions: Vec<DataflowInstruction>,
    pub successors: Vec<BlockId>,
    pub predecessors: Vec<BlockId>,
    pub is_entry: bool,
    pub is_exit: bool,
    pub thread_id: usize,
    pub label: Option<String>,
}

impl BasicBlock {
    pub fn new(id: BlockId) -> Self {
        BasicBlock {
            id,
            instructions: Vec::new(),
            successors: Vec::new(),
            predecessors: Vec::new(),
            is_entry: false,
            is_exit: false,
            thread_id: 0,
            label: None,
        }
    }

    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }

    pub fn add_instruction(&mut self, inst: DataflowInstruction) {
        self.instructions.push(inst);
    }
}

/// An instruction for dataflow analysis purposes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DataflowInstruction {
    /// Load from address into variable.
    Load { dst: Variable, addr: MemoryAddr, ordering: MemoryOrdering },
    /// Store value to address.
    Store { addr: MemoryAddr, src: ValueExpr, ordering: MemoryOrdering },
    /// Local assignment.
    Assign { dst: Variable, expr: ValueExpr },
    /// Fence instruction.
    Fence { kind: FenceKind },
    /// Conditional branch.
    Branch { cond: ValueExpr, true_target: BlockId, false_target: BlockId },
    /// Compare and swap.
    CAS { dst: Variable, addr: MemoryAddr, expected: ValueExpr, desired: ValueExpr, ordering: MemoryOrdering },
    /// Atomic read-modify-write.
    RMW { dst: Variable, addr: MemoryAddr, op: RMWOp, operand: ValueExpr, ordering: MemoryOrdering },
    /// No-operation.
    Nop,
}

/// A variable in the program.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Variable {
    pub name: String,
    pub thread: usize,
    pub version: usize,
}

impl Variable {
    pub fn new(name: &str, thread: usize) -> Self {
        Variable {
            name: name.to_string(),
            thread,
            version: 0,
        }
    }

    pub fn with_version(mut self, version: usize) -> Self {
        self.version = version;
        self
    }
}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.version > 0 {
            write!(f, "{}_{}.{}", self.name, self.thread, self.version)
        } else {
            write!(f, "{}_{}", self.name, self.thread)
        }
    }
}

/// A memory address expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MemoryAddr {
    Named(String),
    Offset(String, i64),
    Indirect(Box<ValueExpr>),
}

impl fmt::Display for MemoryAddr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryAddr::Named(n) => write!(f, "{}", n),
            MemoryAddr::Offset(n, o) => write!(f, "{}+{}", n, o),
            MemoryAddr::Indirect(e) => write!(f, "[{:?}]", e),
        }
    }
}

/// Value expressions for dataflow.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ValueExpr {
    Const(i64),
    Var(Variable),
    BinOp(Box<ValueExpr>, BinOp, Box<ValueExpr>),
    UnaryOp(UnaryOp, Box<ValueExpr>),
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add, Sub, Mul, Div, Mod,
    And, Or, Xor, Shl, Shr,
    Eq, Ne, Lt, Le, Gt, Ge,
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Neg, Not, BitNot,
}

/// Memory ordering for loads/stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryOrdering {
    Relaxed, Acquire, Release, AcqRel, SeqCst, NonAtomic,
}

/// Fence kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FenceKind {
    Full, Acquire, Release, StoreStore, LoadLoad, LoadStore, StoreLoad,
}

/// Read-modify-write operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RMWOp {
    Add, Sub, And, Or, Xor, Exchange, Max, Min,
}

// ═══════════════════════════════════════════════════════════════════════════
// Control Flow Graph Implementation
// ═══════════════════════════════════════════════════════════════════════════

/// A control flow graph for a single thread.
#[derive(Debug, Clone)]
pub struct ControlFlowGraph {
    pub blocks: Vec<BasicBlock>,
    pub entry: BlockId,
    pub exits: Vec<BlockId>,
    pub thread_id: usize,
    pub dominators: Option<DominatorTree>,
    pub post_dominators: Option<DominatorTree>,
    pub loop_info: Option<LoopInfo>,
}

impl ControlFlowGraph {
    pub fn new(thread_id: usize) -> Self {
        let entry_block = BasicBlock {
            id: 0,
            instructions: Vec::new(),
            successors: Vec::new(),
            predecessors: Vec::new(),
            is_entry: true,
            is_exit: false,
            thread_id,
            label: Some("entry".into()),
        };
        ControlFlowGraph {
            blocks: vec![entry_block],
            entry: 0,
            exits: Vec::new(),
            thread_id,
            dominators: None,
            post_dominators: None,
            loop_info: None,
        }
    }

    pub fn add_block(&mut self) -> BlockId {
        let id = self.blocks.len();
        let mut block = BasicBlock::new(id);
        block.thread_id = self.thread_id;
        self.blocks.push(block);
        id
    }

    pub fn add_edge(&mut self, from: BlockId, to: BlockId) {
        if !self.blocks[from].successors.contains(&to) {
            self.blocks[from].successors.push(to);
        }
        if !self.blocks[to].predecessors.contains(&from) {
            self.blocks[to].predecessors.push(from);
        }
    }

    pub fn remove_edge(&mut self, from: BlockId, to: BlockId) {
        self.blocks[from].successors.retain(|&s| s != to);
        self.blocks[to].predecessors.retain(|&p| p != from);
    }

    pub fn successors(&self, block: BlockId) -> &[BlockId] {
        &self.blocks[block].successors
    }

    pub fn predecessors(&self, block: BlockId) -> &[BlockId] {
        &self.blocks[block].predecessors
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Compute reverse post-order traversal.
    pub fn reverse_postorder(&self) -> Vec<BlockId> {
        let mut visited = vec![false; self.blocks.len()];
        let mut order = Vec::new();
        self.rpo_dfs(self.entry, &mut visited, &mut order);
        order.reverse();
        order
    }

    fn rpo_dfs(&self, block: BlockId, visited: &mut Vec<bool>, order: &mut Vec<BlockId>) {
        if visited[block] {
            return;
        }
        visited[block] = true;
        for &succ in &self.blocks[block].successors {
            self.rpo_dfs(succ, visited, order);
        }
        order.push(block);
    }

    /// Compute post-order traversal on the reverse CFG.
    pub fn reverse_cfg_postorder(&self) -> Vec<BlockId> {
        let mut visited = vec![false; self.blocks.len()];
        let mut order = Vec::new();
        for &exit in &self.exits {
            self.reverse_rpo_dfs(exit, &mut visited, &mut order);
        }
        order.reverse();
        order
    }

    fn reverse_rpo_dfs(&self, block: BlockId, visited: &mut Vec<bool>, order: &mut Vec<BlockId>) {
        if visited[block] {
            return;
        }
        visited[block] = true;
        for &pred in &self.blocks[block].predecessors {
            self.reverse_rpo_dfs(pred, visited, order);
        }
        order.push(block);
    }

    /// Compute dominator tree using iterative algorithm.
    pub fn compute_dominators(&mut self) {
        let n = self.blocks.len();
        let rpo = self.reverse_postorder();
        let mut rpo_number = vec![0usize; n];
        for (i, &b) in rpo.iter().enumerate() {
            rpo_number[b] = i;
        }
        let mut idom: Vec<Option<BlockId>> = vec![None; n];
        idom[self.entry] = Some(self.entry);
        let mut changed = true;

        while changed {
            changed = false;
            for &b in &rpo {
                if b == self.entry {
                    continue;
                }
                let mut new_idom: Option<BlockId> = None;
                for &p in &self.blocks[b].predecessors {
                    if idom[p].is_some() {
                        new_idom = Some(match new_idom {
                            None => p,
                            Some(current) => {
                                self.intersect_dom(current, p, &idom, &rpo_number)
                            }
                        });
                    }
                }
                if new_idom != idom[b] {
                    idom[b] = new_idom;
                    changed = true;
                }
            }
        }

        let mut children: Vec<Vec<BlockId>> = vec![Vec::new(); n];
        for b in 0..n {
            if let Some(d) = idom[b] {
                if d != b {
                    children[d].push(b);
                }
            }
        }

        let mut depth = vec![0u32; n];
        let mut stack = vec![self.entry];
        while let Some(node) = stack.pop() {
            for &child in &children[node] {
                depth[child] = depth[node] + 1;
                stack.push(child);
            }
        }

        self.dominators = Some(DominatorTree {
            idom,
            children,
            depth,
        });
    }

    fn intersect_dom(
        &self,
        mut a: BlockId,
        mut b: BlockId,
        idom: &[Option<BlockId>],
        rpo_number: &[usize],
    ) -> BlockId {
        while a != b {
            while rpo_number[a] > rpo_number[b] {
                a = idom[a].unwrap_or(a);
            }
            while rpo_number[b] > rpo_number[a] {
                b = idom[b].unwrap_or(b);
            }
        }
        a
    }

    /// Check if block a dominates block b.
    pub fn dominates(&self, a: BlockId, b: BlockId) -> bool {
        if let Some(ref dom) = self.dominators {
            let mut current = b;
            loop {
                if current == a {
                    return true;
                }
                match dom.idom[current] {
                    Some(d) if d != current => current = d,
                    _ => return false,
                }
            }
        }
        false
    }

    /// Compute dominance frontiers.
    pub fn dominance_frontiers(&self) -> Vec<HashSet<BlockId>> {
        let n = self.blocks.len();
        let mut df: Vec<HashSet<BlockId>> = vec![HashSet::new(); n];
        let dom = match &self.dominators {
            Some(d) => d,
            None => return df,
        };
        for b in 0..n {
            let preds = &self.blocks[b].predecessors;
            if preds.len() >= 2 {
                for &p in preds {
                    let mut runner = p;
                    while Some(runner) != dom.idom[b] && runner != b {
                        df[runner].insert(b);
                        match dom.idom[runner] {
                            Some(d) if d != runner => runner = d,
                            _ => break,
                        }
                    }
                }
            }
        }
        df
    }

    /// Detect natural loops.
    pub fn detect_loops(&mut self) {
        if self.dominators.is_none() {
            self.compute_dominators();
        }
        let n = self.blocks.len();
        let mut loops = Vec::new();

        for b in 0..n {
            for &succ in &self.blocks[b].successors {
                if self.dominates(succ, b) {
                    let body = self.natural_loop_body(succ, b);
                    loops.push(NaturalLoop {
                        header: succ,
                        back_edge: (b, succ),
                        body,
                        exits: Vec::new(),
                        depth: 0,
                    });
                }
            }
        }

        for lp in &mut loops {
            for &block in &lp.body {
                for &succ in &self.blocks[block].successors {
                    if !lp.body.contains(&succ) {
                        lp.exits.push((block, succ));
                    }
                }
            }
        }

        for i in 0..loops.len() {
            let mut depth = 0u32;
            for j in 0..loops.len() {
                if i != j && loops[j].body.contains(&loops[i].header) {
                    depth += 1;
                }
            }
            loops[i].depth = depth;
        }

        self.loop_info = Some(LoopInfo { loops });
    }

    fn natural_loop_body(&self, header: BlockId, tail: BlockId) -> Vec<BlockId> {
        let mut body = HashSet::new();
        body.insert(header);
        if header == tail {
            return body.into_iter().collect();
        }
        body.insert(tail);
        let mut worklist = vec![tail];
        while let Some(node) = worklist.pop() {
            for &pred in &self.blocks[node].predecessors {
                if !body.contains(&pred) {
                    body.insert(pred);
                    worklist.push(pred);
                }
            }
        }
        let mut result: Vec<_> = body.into_iter().collect();
        result.sort();
        result
    }
}

/// Dominator tree.
#[derive(Debug, Clone)]
pub struct DominatorTree {
    pub idom: Vec<Option<BlockId>>,
    pub children: Vec<Vec<BlockId>>,
    pub depth: Vec<u32>,
}

/// Natural loop information.
#[derive(Debug, Clone)]
pub struct NaturalLoop {
    pub header: BlockId,
    pub back_edge: (BlockId, BlockId),
    pub body: Vec<BlockId>,
    pub exits: Vec<(BlockId, BlockId)>,
    pub depth: u32,
}

/// Loop information for the entire CFG.
#[derive(Debug, Clone)]
pub struct LoopInfo {
    pub loops: Vec<NaturalLoop>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Dataflow Analysis Framework
// ═══════════════════════════════════════════════════════════════════════════

/// Direction of dataflow analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisDirection {
    Forward,
    Backward,
}

/// Transfer function trait for dataflow analysis.
pub trait TransferFunction<L: Lattice> {
    fn apply(&self, block: &BasicBlock, input: &L) -> L;
    fn direction(&self) -> AnalysisDirection;
    fn initial_value(&self) -> L;
    fn boundary_value(&self) -> L;
}

/// Result of a dataflow analysis.
#[derive(Debug, Clone)]
pub struct DataflowResult<L: Lattice> {
    pub in_values: Vec<L>,
    pub out_values: Vec<L>,
    pub iterations: u32,
    pub converged: bool,
}

/// Generic dataflow analysis engine using iterative worklist algorithm.
pub struct DataflowEngine<L: Lattice, T: TransferFunction<L>> {
    transfer: T,
    max_iterations: u32,
    use_widening: bool,
    widening_threshold: u32,
    _phantom: PhantomData<L>,
}

impl<L: Lattice, T: TransferFunction<L>> DataflowEngine<L, T> {
    pub fn new(transfer: T) -> Self {
        DataflowEngine {
            transfer,
            max_iterations: 1000,
            use_widening: false,
            widening_threshold: 10,
            _phantom: PhantomData,
        }
    }

    pub fn with_max_iterations(mut self, max: u32) -> Self {
        self.max_iterations = max;
        self
    }

    pub fn with_widening(mut self, threshold: u32) -> Self {
        self.use_widening = true;
        self.widening_threshold = threshold;
        self
    }

    /// Run the dataflow analysis on a CFG.
    pub fn analyze(&self, cfg: &ControlFlowGraph) -> DataflowResult<L> {
        let n = cfg.num_blocks();
        let init = self.transfer.initial_value();
        let boundary = self.transfer.boundary_value();

        let mut in_values = vec![init.clone(); n];
        let mut out_values = vec![init.clone(); n];

        match self.transfer.direction() {
            AnalysisDirection::Forward => {
                in_values[cfg.entry] = boundary.clone();
            }
            AnalysisDirection::Backward => {
                for &exit in &cfg.exits {
                    out_values[exit] = boundary.clone();
                }
            }
        }

        let worklist_order = match self.transfer.direction() {
            AnalysisDirection::Forward => cfg.reverse_postorder(),
            AnalysisDirection::Backward => cfg.reverse_cfg_postorder(),
        };

        let mut worklist: VecDeque<BlockId> = worklist_order.into_iter().collect();
        let mut in_worklist = vec![true; n];
        let mut iterations = 0u32;
        let mut iter_counts = vec![0u32; n];

        while let Some(block_id) = worklist.pop_front() {
            in_worklist[block_id] = false;
            iterations += 1;

            if iterations > self.max_iterations {
                return DataflowResult {
                    in_values,
                    out_values,
                    iterations,
                    converged: false,
                };
            }

            match self.transfer.direction() {
                AnalysisDirection::Forward => {
                    let mut new_in = if cfg.blocks[block_id].predecessors.is_empty() {
                        in_values[block_id].clone()
                    } else {
                        let mut acc = L::bottom();
                        for &pred in &cfg.blocks[block_id].predecessors {
                            acc = acc.join(&out_values[pred]);
                        }
                        acc
                    };

                    if self.use_widening && iter_counts[block_id] >= self.widening_threshold {
                        new_in = in_values[block_id].widen(&new_in);
                    }

                    let new_out = self.transfer.apply(&cfg.blocks[block_id], &new_in);

                    if new_out != out_values[block_id] {
                        in_values[block_id] = new_in;
                        out_values[block_id] = new_out;
                        iter_counts[block_id] += 1;
                        for &succ in &cfg.blocks[block_id].successors {
                            if !in_worklist[succ] {
                                worklist.push_back(succ);
                                in_worklist[succ] = true;
                            }
                        }
                    }
                }
                AnalysisDirection::Backward => {
                    let mut new_out = if cfg.blocks[block_id].successors.is_empty() {
                        out_values[block_id].clone()
                    } else {
                        let mut acc = L::bottom();
                        for &succ in &cfg.blocks[block_id].successors {
                            acc = acc.join(&in_values[succ]);
                        }
                        acc
                    };

                    if self.use_widening && iter_counts[block_id] >= self.widening_threshold {
                        new_out = out_values[block_id].widen(&new_out);
                    }

                    let new_in = self.transfer.apply(&cfg.blocks[block_id], &new_out);

                    if new_in != in_values[block_id] {
                        out_values[block_id] = new_out;
                        in_values[block_id] = new_in;
                        iter_counts[block_id] += 1;
                        for &pred in &cfg.blocks[block_id].predecessors {
                            if !in_worklist[pred] {
                                worklist.push_back(pred);
                                in_worklist[pred] = true;
                            }
                        }
                    }
                }
            }
        }

        DataflowResult {
            in_values,
            out_values,
            iterations,
            converged: true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Definition-Use Analysis
// ═══════════════════════════════════════════════════════════════════════════

/// A definition site: (variable, block, instruction index).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Definition {
    pub var: Variable,
    pub block: BlockId,
    pub inst_index: usize,
}

/// A use site: (variable, block, instruction index).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Use {
    pub var: Variable,
    pub block: BlockId,
    pub inst_index: usize,
}

/// Definition-use chains.
#[derive(Debug, Clone)]
pub struct DefUseChains {
    pub defs: HashMap<Variable, Vec<Definition>>,
    pub uses: HashMap<Variable, Vec<Use>>,
    pub def_use: HashMap<Definition, Vec<Use>>,
    pub use_def: HashMap<Use, Vec<Definition>>,
}

impl DefUseChains {
    pub fn compute(cfg: &ControlFlowGraph) -> Self {
        let mut defs: HashMap<Variable, Vec<Definition>> = HashMap::new();
        let mut uses: HashMap<Variable, Vec<Use>> = HashMap::new();

        for block in &cfg.blocks {
            for (i, inst) in block.instructions.iter().enumerate() {
                for var in Self::defined_vars(inst) {
                    let def = Definition {
                        var: var.clone(),
                        block: block.id,
                        inst_index: i,
                    };
                    defs.entry(var).or_default().push(def);
                }

                for var in Self::used_vars(inst) {
                    let u = Use {
                        var: var.clone(),
                        block: block.id,
                        inst_index: i,
                    };
                    uses.entry(var).or_default().push(u);
                }
            }
        }

        DefUseChains {
            defs,
            uses,
            def_use: HashMap::new(),
            use_def: HashMap::new(),
        }
    }

    pub fn defined_vars(inst: &DataflowInstruction) -> Vec<Variable> {
        match inst {
            DataflowInstruction::Load { dst, .. } => vec![dst.clone()],
            DataflowInstruction::Assign { dst, .. } => vec![dst.clone()],
            DataflowInstruction::CAS { dst, .. } => vec![dst.clone()],
            DataflowInstruction::RMW { dst, .. } => vec![dst.clone()],
            _ => Vec::new(),
        }
    }

    pub fn used_vars(inst: &DataflowInstruction) -> Vec<Variable> {
        let mut result = Vec::new();
        match inst {
            DataflowInstruction::Store { src, .. } => {
                Self::collect_expr_vars(src, &mut result);
            }
            DataflowInstruction::Assign { expr, .. } => {
                Self::collect_expr_vars(expr, &mut result);
            }
            DataflowInstruction::Branch { cond, .. } => {
                Self::collect_expr_vars(cond, &mut result);
            }
            DataflowInstruction::CAS { expected, desired, .. } => {
                Self::collect_expr_vars(expected, &mut result);
                Self::collect_expr_vars(desired, &mut result);
            }
            DataflowInstruction::RMW { operand, .. } => {
                Self::collect_expr_vars(operand, &mut result);
            }
            _ => {}
        }
        result
    }

    pub fn collect_expr_vars(expr: &ValueExpr, result: &mut Vec<Variable>) {
        match expr {
            ValueExpr::Var(v) => result.push(v.clone()),
            ValueExpr::BinOp(l, _, r) => {
                Self::collect_expr_vars(l, result);
                Self::collect_expr_vars(r, result);
            }
            ValueExpr::UnaryOp(_, e) => {
                Self::collect_expr_vars(e, result);
            }
            ValueExpr::Const(_) => {}
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Reaching Definitions Analysis
// ═══════════════════════════════════════════════════════════════════════════

/// Reaching definitions lattice value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReachingDefinitions {
    pub definitions: HashSet<Definition>,
}

impl Lattice for ReachingDefinitions {
    fn bottom() -> Self {
        ReachingDefinitions { definitions: HashSet::new() }
    }
    fn top() -> Self {
        ReachingDefinitions { definitions: HashSet::new() }
    }
    fn join(&self, other: &Self) -> Self {
        ReachingDefinitions {
            definitions: self.definitions.union(&other.definitions).cloned().collect(),
        }
    }
    fn meet(&self, other: &Self) -> Self {
        ReachingDefinitions {
            definitions: self.definitions.intersection(&other.definitions).cloned().collect(),
        }
    }
    fn leq(&self, other: &Self) -> bool {
        self.definitions.is_subset(&other.definitions)
    }
}

/// Transfer function for reaching definitions.
pub struct ReachingDefinitionsTransfer;

impl TransferFunction<ReachingDefinitions> for ReachingDefinitionsTransfer {
    fn apply(&self, block: &BasicBlock, input: &ReachingDefinitions) -> ReachingDefinitions {
        let mut result = input.definitions.clone();
        for (i, inst) in block.instructions.iter().enumerate() {
            for var in DefUseChains::defined_vars(inst) {
                result.retain(|d| d.var != var);
                result.insert(Definition {
                    var,
                    block: block.id,
                    inst_index: i,
                });
            }
        }
        ReachingDefinitions { definitions: result }
    }
    fn direction(&self) -> AnalysisDirection { AnalysisDirection::Forward }
    fn initial_value(&self) -> ReachingDefinitions { ReachingDefinitions::bottom() }
    fn boundary_value(&self) -> ReachingDefinitions { ReachingDefinitions::bottom() }
}

/// Run reaching definitions analysis.
pub fn reaching_definitions(cfg: &ControlFlowGraph) -> DataflowResult<ReachingDefinitions> {
    DataflowEngine::new(ReachingDefinitionsTransfer).analyze(cfg)
}

// ═══════════════════════════════════════════════════════════════════════════
// Live Variable Analysis
// ═══════════════════════════════════════════════════════════════════════════

/// Live variables lattice value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveVariables {
    pub variables: HashSet<Variable>,
}

impl Lattice for LiveVariables {
    fn bottom() -> Self { LiveVariables { variables: HashSet::new() } }
    fn top() -> Self { LiveVariables { variables: HashSet::new() } }
    fn join(&self, other: &Self) -> Self {
        LiveVariables {
            variables: self.variables.union(&other.variables).cloned().collect(),
        }
    }
    fn meet(&self, other: &Self) -> Self {
        LiveVariables {
            variables: self.variables.intersection(&other.variables).cloned().collect(),
        }
    }
    fn leq(&self, other: &Self) -> bool {
        self.variables.is_subset(&other.variables)
    }
}

/// Transfer function for live variable analysis (backward).
pub struct LiveVariablesTransfer;

impl TransferFunction<LiveVariables> for LiveVariablesTransfer {
    fn apply(&self, block: &BasicBlock, input: &LiveVariables) -> LiveVariables {
        let mut result = input.variables.clone();
        for inst in block.instructions.iter().rev() {
            for var in &DefUseChains::defined_vars(inst) {
                result.remove(var);
            }
            for var in DefUseChains::used_vars(inst) {
                result.insert(var);
            }
        }
        LiveVariables { variables: result }
    }
    fn direction(&self) -> AnalysisDirection { AnalysisDirection::Backward }
    fn initial_value(&self) -> LiveVariables { LiveVariables::bottom() }
    fn boundary_value(&self) -> LiveVariables { LiveVariables::bottom() }
}

/// Run live variable analysis.
pub fn live_variables(cfg: &ControlFlowGraph) -> DataflowResult<LiveVariables> {
    DataflowEngine::new(LiveVariablesTransfer).analyze(cfg)
}

// ═══════════════════════════════════════════════════════════════════════════
// Available Expressions Analysis
// ═══════════════════════════════════════════════════════════════════════════

/// An expression that may be available.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AvailableExpr {
    pub expr: ValueExpr,
    pub variables: Vec<Variable>,
}

/// Available expressions lattice value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AvailableExpressions {
    pub expressions: HashSet<AvailableExpr>,
    pub is_top: bool,
}

impl Lattice for AvailableExpressions {
    fn bottom() -> Self {
        AvailableExpressions { expressions: HashSet::new(), is_top: true }
    }
    fn top() -> Self {
        AvailableExpressions { expressions: HashSet::new(), is_top: true }
    }
    fn join(&self, other: &Self) -> Self {
        if self.is_top { return other.clone(); }
        if other.is_top { return self.clone(); }
        AvailableExpressions {
            expressions: self.expressions.intersection(&other.expressions).cloned().collect(),
            is_top: false,
        }
    }
    fn meet(&self, other: &Self) -> Self {
        if self.is_top { return self.clone(); }
        if other.is_top { return other.clone(); }
        AvailableExpressions {
            expressions: self.expressions.union(&other.expressions).cloned().collect(),
            is_top: false,
        }
    }
    fn leq(&self, other: &Self) -> bool {
        if other.is_top { return true; }
        if self.is_top { return false; }
        other.expressions.is_subset(&self.expressions)
    }
}

/// Transfer function for available expressions.
pub struct AvailableExpressionsTransfer;

impl TransferFunction<AvailableExpressions> for AvailableExpressionsTransfer {
    fn apply(&self, block: &BasicBlock, input: &AvailableExpressions) -> AvailableExpressions {
        let mut result = input.expressions.clone();
        for inst in &block.instructions {
            let defined = DefUseChains::defined_vars(inst);
            for var in &defined {
                result.retain(|ae| !ae.variables.contains(var));
            }
            if let DataflowInstruction::Assign { expr, .. } = inst {
                let mut vars = Vec::new();
                DefUseChains::collect_expr_vars(expr, &mut vars);
                if !vars.is_empty() {
                    result.insert(AvailableExpr { expr: expr.clone(), variables: vars });
                }
            }
        }
        AvailableExpressions { expressions: result, is_top: false }
    }
    fn direction(&self) -> AnalysisDirection { AnalysisDirection::Forward }
    fn initial_value(&self) -> AvailableExpressions { AvailableExpressions::bottom() }
    fn boundary_value(&self) -> AvailableExpressions {
        AvailableExpressions { expressions: HashSet::new(), is_top: false }
    }
}

/// Run available expressions analysis.
pub fn available_expressions(cfg: &ControlFlowGraph) -> DataflowResult<AvailableExpressions> {
    DataflowEngine::new(AvailableExpressionsTransfer).analyze(cfg)
}

// ═══════════════════════════════════════════════════════════════════════════
// Very Busy Expressions Analysis
// ═══════════════════════════════════════════════════════════════════════════

/// Very busy expressions lattice (backward must analysis).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VeryBusyExpressions {
    pub expressions: HashSet<AvailableExpr>,
    pub is_top: bool,
}

impl Lattice for VeryBusyExpressions {
    fn bottom() -> Self { VeryBusyExpressions { expressions: HashSet::new(), is_top: true } }
    fn top() -> Self { VeryBusyExpressions { expressions: HashSet::new(), is_top: true } }
    fn join(&self, other: &Self) -> Self {
        if self.is_top { return other.clone(); }
        if other.is_top { return self.clone(); }
        VeryBusyExpressions {
            expressions: self.expressions.intersection(&other.expressions).cloned().collect(),
            is_top: false,
        }
    }
    fn meet(&self, other: &Self) -> Self {
        if self.is_top { return self.clone(); }
        if other.is_top { return other.clone(); }
        VeryBusyExpressions {
            expressions: self.expressions.union(&other.expressions).cloned().collect(),
            is_top: false,
        }
    }
    fn leq(&self, other: &Self) -> bool {
        if other.is_top { return true; }
        if self.is_top { return false; }
        other.expressions.is_subset(&self.expressions)
    }
}

/// Transfer function for very busy expressions (backward).
pub struct VeryBusyExpressionsTransfer;

impl TransferFunction<VeryBusyExpressions> for VeryBusyExpressionsTransfer {
    fn apply(&self, block: &BasicBlock, input: &VeryBusyExpressions) -> VeryBusyExpressions {
        let mut result = input.expressions.clone();
        for inst in block.instructions.iter().rev() {
            for var in &DefUseChains::defined_vars(inst) {
                result.retain(|ae| !ae.variables.contains(var));
            }
            if let DataflowInstruction::Assign { expr, .. } = inst {
                let mut vars = Vec::new();
                DefUseChains::collect_expr_vars(expr, &mut vars);
                if !vars.is_empty() {
                    result.insert(AvailableExpr { expr: expr.clone(), variables: vars });
                }
            }
        }
        VeryBusyExpressions { expressions: result, is_top: false }
    }
    fn direction(&self) -> AnalysisDirection { AnalysisDirection::Backward }
    fn initial_value(&self) -> VeryBusyExpressions { VeryBusyExpressions::bottom() }
    fn boundary_value(&self) -> VeryBusyExpressions {
        VeryBusyExpressions { expressions: HashSet::new(), is_top: false }
    }
}

/// Run very busy expressions analysis.
pub fn very_busy_expressions(cfg: &ControlFlowGraph) -> DataflowResult<VeryBusyExpressions> {
    DataflowEngine::new(VeryBusyExpressionsTransfer).analyze(cfg)
}

// ═══════════════════════════════════════════════════════════════════════════
// Constant Propagation Analysis
// ═══════════════════════════════════════════════════════════════════════════

/// Constant propagation lattice element for a single variable.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConstValue {
    Bottom,
    Const(i64),
    Top,
}

impl ConstValue {
    pub fn join(&self, other: &ConstValue) -> ConstValue {
        match (self, other) {
            (ConstValue::Bottom, x) | (x, ConstValue::Bottom) => x.clone(),
            (ConstValue::Top, _) | (_, ConstValue::Top) => ConstValue::Top,
            (ConstValue::Const(a), ConstValue::Const(b)) => {
                if a == b { ConstValue::Const(*a) } else { ConstValue::Top }
            }
        }
    }

    pub fn meet(&self, other: &ConstValue) -> ConstValue {
        match (self, other) {
            (ConstValue::Top, x) | (x, ConstValue::Top) => x.clone(),
            (ConstValue::Bottom, _) | (_, ConstValue::Bottom) => ConstValue::Bottom,
            (ConstValue::Const(a), ConstValue::Const(b)) => {
                if a == b { ConstValue::Const(*a) } else { ConstValue::Bottom }
            }
        }
    }
}

/// Constant propagation state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstantPropagation {
    pub values: HashMap<Variable, ConstValue>,
}

impl Lattice for ConstantPropagation {
    fn bottom() -> Self { ConstantPropagation { values: HashMap::new() } }
    fn top() -> Self { ConstantPropagation { values: HashMap::new() } }
    fn join(&self, other: &Self) -> Self {
        let mut result = self.values.clone();
        for (var, val) in &other.values {
            let entry = result.entry(var.clone()).or_insert(ConstValue::Bottom);
            *entry = entry.join(val);
        }
        ConstantPropagation { values: result }
    }
    fn meet(&self, other: &Self) -> Self {
        let mut result = HashMap::new();
        for var in self.values.keys().chain(other.values.keys()) {
            let a = self.values.get(var).unwrap_or(&ConstValue::Top);
            let b = other.values.get(var).unwrap_or(&ConstValue::Top);
            result.insert(var.clone(), a.meet(b));
        }
        ConstantPropagation { values: result }
    }
    fn leq(&self, other: &Self) -> bool {
        for (var, val) in &self.values {
            let other_val = other.values.get(var).unwrap_or(&ConstValue::Bottom);
            match (val, other_val) {
                (ConstValue::Bottom, _) => {}
                (_, ConstValue::Top) => {}
                (ConstValue::Const(a), ConstValue::Const(b)) if a == b => {}
                _ => return false,
            }
        }
        true
    }
}

/// Transfer function for constant propagation.
pub struct ConstantPropagationTransfer;

impl ConstantPropagationTransfer {
    fn evaluate(&self, expr: &ValueExpr, state: &ConstantPropagation) -> ConstValue {
        match expr {
            ValueExpr::Const(c) => ConstValue::Const(*c),
            ValueExpr::Var(v) => state.values.get(v).cloned().unwrap_or(ConstValue::Top),
            ValueExpr::BinOp(l, op, r) => {
                let lv = self.evaluate(l, state);
                let rv = self.evaluate(r, state);
                match (lv, rv) {
                    (ConstValue::Const(a), ConstValue::Const(b)) => {
                        let result = match op {
                            BinOp::Add => Some(a.wrapping_add(b)),
                            BinOp::Sub => Some(a.wrapping_sub(b)),
                            BinOp::Mul => Some(a.wrapping_mul(b)),
                            BinOp::Div => if b != 0 { Some(a / b) } else { None },
                            BinOp::Mod => if b != 0 { Some(a % b) } else { None },
                            BinOp::And => Some(a & b),
                            BinOp::Or => Some(a | b),
                            BinOp::Xor => Some(a ^ b),
                            BinOp::Shl => Some(a << (b as u32 & 63)),
                            BinOp::Shr => Some(a >> (b as u32 & 63)),
                            BinOp::Eq => Some(if a == b { 1 } else { 0 }),
                            BinOp::Ne => Some(if a != b { 1 } else { 0 }),
                            BinOp::Lt => Some(if a < b { 1 } else { 0 }),
                            BinOp::Le => Some(if a <= b { 1 } else { 0 }),
                            BinOp::Gt => Some(if a > b { 1 } else { 0 }),
                            BinOp::Ge => Some(if a >= b { 1 } else { 0 }),
                        };
                        match result {
                            Some(v) => ConstValue::Const(v),
                            None => ConstValue::Top,
                        }
                    }
                    (ConstValue::Bottom, _) | (_, ConstValue::Bottom) => ConstValue::Bottom,
                    _ => ConstValue::Top,
                }
            }
            ValueExpr::UnaryOp(op, e) => {
                match self.evaluate(e, state) {
                    ConstValue::Const(v) => {
                        let result = match op {
                            UnaryOp::Neg => -v,
                            UnaryOp::Not => if v == 0 { 1 } else { 0 },
                            UnaryOp::BitNot => !v,
                        };
                        ConstValue::Const(result)
                    }
                    ConstValue::Bottom => ConstValue::Bottom,
                    ConstValue::Top => ConstValue::Top,
                }
            }
        }
    }
}

impl TransferFunction<ConstantPropagation> for ConstantPropagationTransfer {
    fn apply(&self, block: &BasicBlock, input: &ConstantPropagation) -> ConstantPropagation {
        let mut state = input.clone();
        for inst in &block.instructions {
            match inst {
                DataflowInstruction::Assign { dst, expr } => {
                    let val = self.evaluate(expr, &state);
                    state.values.insert(dst.clone(), val);
                }
                DataflowInstruction::Load { dst, .. }
                | DataflowInstruction::CAS { dst, .. }
                | DataflowInstruction::RMW { dst, .. } => {
                    state.values.insert(dst.clone(), ConstValue::Top);
                }
                _ => {}
            }
        }
        state
    }
    fn direction(&self) -> AnalysisDirection { AnalysisDirection::Forward }
    fn initial_value(&self) -> ConstantPropagation { ConstantPropagation::bottom() }
    fn boundary_value(&self) -> ConstantPropagation { ConstantPropagation::bottom() }
}

/// Run constant propagation analysis.
pub fn constant_propagation(cfg: &ControlFlowGraph) -> DataflowResult<ConstantPropagation> {
    DataflowEngine::new(ConstantPropagationTransfer).analyze(cfg)
}

// ═══════════════════════════════════════════════════════════════════════════
// Interval Analysis
// ═══════════════════════════════════════════════════════════════════════════

/// Interval analysis state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntervalState {
    pub intervals: HashMap<Variable, IntervalLattice>,
}

impl Lattice for IntervalState {
    fn bottom() -> Self { IntervalState { intervals: HashMap::new() } }
    fn top() -> Self { IntervalState { intervals: HashMap::new() } }
    fn join(&self, other: &Self) -> Self {
        let mut result = self.intervals.clone();
        for (var, val) in &other.intervals {
            let entry = result.entry(var.clone()).or_insert(IntervalLattice::Bottom);
            *entry = entry.join(val);
        }
        IntervalState { intervals: result }
    }
    fn meet(&self, other: &Self) -> Self {
        let mut result = HashMap::new();
        for var in self.intervals.keys().chain(other.intervals.keys()) {
            let a = self.intervals.get(var).unwrap_or(&IntervalLattice::Top);
            let b = other.intervals.get(var).unwrap_or(&IntervalLattice::Top);
            result.insert(var.clone(), a.meet(b));
        }
        IntervalState { intervals: result }
    }
    fn leq(&self, other: &Self) -> bool {
        for (var, val) in &self.intervals {
            let other_val = other.intervals.get(var).unwrap_or(&IntervalLattice::Bottom);
            if !val.leq(other_val) { return false; }
        }
        true
    }
    fn widen(&self, other: &Self) -> Self {
        let mut result = self.intervals.clone();
        for (var, val) in &other.intervals {
            let entry = result.entry(var.clone()).or_insert(IntervalLattice::Bottom);
            *entry = entry.widen(val);
        }
        IntervalState { intervals: result }
    }
}

/// Transfer function for interval analysis.
pub struct IntervalAnalysisTransfer;

impl IntervalAnalysisTransfer {
    fn evaluate_interval(&self, expr: &ValueExpr, state: &IntervalState) -> IntervalLattice {
        match expr {
            ValueExpr::Const(c) => IntervalLattice::Interval { lo: *c, hi: *c },
            ValueExpr::Var(v) => state.intervals.get(v).cloned().unwrap_or(IntervalLattice::Top),
            ValueExpr::BinOp(l, op, r) => {
                let li = self.evaluate_interval(l, state);
                let ri = self.evaluate_interval(r, state);
                match (&li, &ri) {
                    (IntervalLattice::Interval { lo: a, hi: b },
                     IntervalLattice::Interval { lo: c, hi: d }) => {
                        match op {
                            BinOp::Add => IntervalLattice::Interval {
                                lo: a.saturating_add(*c),
                                hi: b.saturating_add(*d),
                            },
                            BinOp::Sub => IntervalLattice::Interval {
                                lo: a.saturating_sub(*d),
                                hi: b.saturating_sub(*c),
                            },
                            BinOp::Mul => {
                                let products = [a * c, a * d, b * c, b * d];
                                IntervalLattice::Interval {
                                    lo: *products.iter().min().unwrap(),
                                    hi: *products.iter().max().unwrap(),
                                }
                            }
                            _ => IntervalLattice::Top,
                        }
                    }
                    (IntervalLattice::Bottom, _) | (_, IntervalLattice::Bottom) => IntervalLattice::Bottom,
                    _ => IntervalLattice::Top,
                }
            }
            ValueExpr::UnaryOp(op, e) => {
                match self.evaluate_interval(e, state) {
                    IntervalLattice::Interval { lo, hi } => match op {
                        UnaryOp::Neg => IntervalLattice::Interval { lo: -hi, hi: -lo },
                        _ => IntervalLattice::Top,
                    },
                    other => other,
                }
            }
        }
    }
}

impl TransferFunction<IntervalState> for IntervalAnalysisTransfer {
    fn apply(&self, block: &BasicBlock, input: &IntervalState) -> IntervalState {
        let mut state = input.clone();
        for inst in &block.instructions {
            match inst {
                DataflowInstruction::Assign { dst, expr } => {
                    let interval = self.evaluate_interval(expr, &state);
                    state.intervals.insert(dst.clone(), interval);
                }
                DataflowInstruction::Load { dst, .. }
                | DataflowInstruction::CAS { dst, .. }
                | DataflowInstruction::RMW { dst, .. } => {
                    state.intervals.insert(dst.clone(), IntervalLattice::Top);
                }
                _ => {}
            }
        }
        state
    }
    fn direction(&self) -> AnalysisDirection { AnalysisDirection::Forward }
    fn initial_value(&self) -> IntervalState { IntervalState::bottom() }
    fn boundary_value(&self) -> IntervalState { IntervalState::bottom() }
}

/// Run interval analysis with widening.
pub fn interval_analysis(cfg: &ControlFlowGraph) -> DataflowResult<IntervalState> {
    DataflowEngine::new(IntervalAnalysisTransfer).with_widening(5).analyze(cfg)
}

// ═══════════════════════════════════════════════════════════════════════════
// SSA Construction
// ═══════════════════════════════════════════════════════════════════════════

/// SSA form representation.
#[derive(Debug, Clone)]
pub struct SSAForm {
    pub cfg: ControlFlowGraph,
    pub phi_functions: HashMap<BlockId, Vec<PhiFunction>>,
    pub version_counter: HashMap<String, usize>,
    pub renamed_vars: HashMap<Variable, Variable>,
}

/// A φ-function at the start of a basic block.
#[derive(Debug, Clone)]
pub struct PhiFunction {
    pub dst: Variable,
    pub args: Vec<(BlockId, Variable)>,
}

impl SSAForm {
    /// Construct SSA form from a CFG using the standard algorithm.
    pub fn construct(mut cfg: ControlFlowGraph) -> Self {
        cfg.compute_dominators();
        let df = cfg.dominance_frontiers();

        let mut phi_functions: HashMap<BlockId, Vec<PhiFunction>> = HashMap::new();
        let version_counter: HashMap<String, usize> = HashMap::new();

        let mut all_vars: HashSet<Variable> = HashSet::new();
        let mut def_sites: HashMap<Variable, HashSet<BlockId>> = HashMap::new();

        for block in &cfg.blocks {
            for inst in &block.instructions {
                for var in DefUseChains::defined_vars(inst) {
                    all_vars.insert(var.clone());
                    def_sites.entry(var).or_default().insert(block.id);
                }
            }
        }

        for var in &all_vars {
            let mut worklist: Vec<BlockId> = def_sites
                .get(var)
                .map(|s| s.iter().copied().collect())
                .unwrap_or_default();
            let mut has_phi: HashSet<BlockId> = HashSet::new();

            while let Some(block) = worklist.pop() {
                for &frontier in &df[block] {
                    if !has_phi.contains(&frontier) {
                        has_phi.insert(frontier);
                        let phi = PhiFunction {
                            dst: var.clone(),
                            args: cfg.blocks[frontier]
                                .predecessors
                                .iter()
                                .map(|&p| (p, var.clone()))
                                .collect(),
                        };
                        phi_functions.entry(frontier).or_default().push(phi);
                        if !def_sites.get(var).map(|s| s.contains(&frontier)).unwrap_or(false) {
                            worklist.push(frontier);
                        }
                    }
                }
            }
        }

        SSAForm {
            cfg,
            phi_functions,
            version_counter,
            renamed_vars: HashMap::new(),
        }
    }

    pub fn fresh_version(&mut self, name: &str) -> usize {
        let counter = self.version_counter.entry(name.to_string()).or_insert(0);
        *counter += 1;
        *counter
    }

    pub fn has_phi(&self, block: BlockId) -> bool {
        self.phi_functions.get(&block).map(|v| !v.is_empty()).unwrap_or(false)
    }

    pub fn phi_at(&self, block: BlockId) -> &[PhiFunction] {
        self.phi_functions.get(&block).map(|v| v.as_slice()).unwrap_or(&[])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Memory Dependence Analysis
// ═══════════════════════════════════════════════════════════════════════════

/// Memory access information.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemoryAccess {
    pub block: BlockId,
    pub inst_index: usize,
    pub addr: MemoryAddr,
    pub is_write: bool,
    pub ordering: MemoryOrdering,
    pub thread_id: usize,
}

/// Memory dependence between two accesses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryDependence {
    pub source: MemoryAccess,
    pub sink: MemoryAccess,
    pub kind: DependenceKind,
}

/// Kind of memory dependence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DependenceKind {
    Flow, Anti, Output, Input,
}

/// Memory dependence analyzer.
#[derive(Debug)]
pub struct MemoryDependenceAnalyzer {
    pub accesses: Vec<MemoryAccess>,
    pub dependences: Vec<MemoryDependence>,
    pub alias_analysis: AliasAnalysis,
}

impl MemoryDependenceAnalyzer {
    pub fn new() -> Self {
        MemoryDependenceAnalyzer {
            accesses: Vec::new(),
            dependences: Vec::new(),
            alias_analysis: AliasAnalysis::new(),
        }
    }

    pub fn collect_accesses(&mut self, cfg: &ControlFlowGraph) {
        for block in &cfg.blocks {
            for (i, inst) in block.instructions.iter().enumerate() {
                match inst {
                    DataflowInstruction::Load { addr, ordering, .. } => {
                        self.accesses.push(MemoryAccess {
                            block: block.id, inst_index: i,
                            addr: addr.clone(), is_write: false,
                            ordering: *ordering, thread_id: block.thread_id,
                        });
                    }
                    DataflowInstruction::Store { addr, ordering, .. } => {
                        self.accesses.push(MemoryAccess {
                            block: block.id, inst_index: i,
                            addr: addr.clone(), is_write: true,
                            ordering: *ordering, thread_id: block.thread_id,
                        });
                    }
                    DataflowInstruction::CAS { addr, ordering, .. }
                    | DataflowInstruction::RMW { addr, ordering, .. } => {
                        self.accesses.push(MemoryAccess {
                            block: block.id, inst_index: i,
                            addr: addr.clone(), is_write: true,
                            ordering: *ordering, thread_id: block.thread_id,
                        });
                    }
                    _ => {}
                }
            }
        }
    }

    pub fn compute_dependences(&mut self) {
        let n = self.accesses.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let a = &self.accesses[i];
                let b = &self.accesses[j];
                if !self.alias_analysis.may_alias(&a.addr, &b.addr) {
                    continue;
                }
                let kind = match (a.is_write, b.is_write) {
                    (true, false) => DependenceKind::Flow,
                    (false, true) => DependenceKind::Anti,
                    (true, true) => DependenceKind::Output,
                    (false, false) => DependenceKind::Input,
                };
                if kind != DependenceKind::Input {
                    self.dependences.push(MemoryDependence {
                        source: a.clone(), sink: b.clone(), kind,
                    });
                }
            }
        }
    }

    pub fn flow_dependences(&self) -> Vec<&MemoryDependence> {
        self.dependences.iter().filter(|d| d.kind == DependenceKind::Flow).collect()
    }

    pub fn anti_dependences(&self) -> Vec<&MemoryDependence> {
        self.dependences.iter().filter(|d| d.kind == DependenceKind::Anti).collect()
    }

    pub fn output_dependences(&self) -> Vec<&MemoryDependence> {
        self.dependences.iter().filter(|d| d.kind == DependenceKind::Output).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Alias Analysis
// ═══════════════════════════════════════════════════════════════════════════

/// Simple alias analysis.
#[derive(Debug)]
pub struct AliasAnalysis {
    pub known_distinct: HashSet<(String, String)>,
}

impl AliasAnalysis {
    pub fn new() -> Self {
        AliasAnalysis { known_distinct: HashSet::new() }
    }

    pub fn add_distinct(&mut self, a: &str, b: &str) {
        self.known_distinct.insert((a.to_string(), b.to_string()));
        self.known_distinct.insert((b.to_string(), a.to_string()));
    }

    pub fn may_alias(&self, a: &MemoryAddr, b: &MemoryAddr) -> bool {
        match (a, b) {
            (MemoryAddr::Named(na), MemoryAddr::Named(nb)) => {
                if na == nb { return true; }
                !self.known_distinct.contains(&(na.clone(), nb.clone()))
            }
            (MemoryAddr::Offset(na, oa), MemoryAddr::Offset(nb, ob)) => {
                if na == nb { oa == ob } else { !self.known_distinct.contains(&(na.clone(), nb.clone())) }
            }
            _ => true,
        }
    }

    pub fn must_alias(&self, a: &MemoryAddr, b: &MemoryAddr) -> bool {
        match (a, b) {
            (MemoryAddr::Named(na), MemoryAddr::Named(nb)) => na == nb,
            (MemoryAddr::Offset(na, oa), MemoryAddr::Offset(nb, ob)) => na == nb && oa == ob,
            _ => false,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Data Race Detection
// ═══════════════════════════════════════════════════════════════════════════

/// A potential data race.
#[derive(Debug, Clone)]
pub struct DataRace {
    pub access1: MemoryAccess,
    pub access2: MemoryAccess,
    pub addr: MemoryAddr,
    pub is_ordering_race: bool,
}

/// Data race detector using happens-before analysis.
#[derive(Debug)]
pub struct DataRaceDetector {
    pub races: Vec<DataRace>,
}

impl DataRaceDetector {
    pub fn new() -> Self {
        DataRaceDetector { races: Vec::new() }
    }

    pub fn detect(&mut self, analyzer: &MemoryDependenceAnalyzer) {
        for dep in &analyzer.dependences {
            if dep.kind == DependenceKind::Input { continue; }
            if dep.source.thread_id != dep.sink.thread_id {
                let has_ordering = self.has_ordering_guarantee(&dep.source, &dep.sink);
                if !has_ordering {
                    self.races.push(DataRace {
                        access1: dep.source.clone(), access2: dep.sink.clone(),
                        addr: dep.source.addr.clone(), is_ordering_race: false,
                    });
                } else if !self.has_sufficient_ordering(&dep.source, &dep.sink) {
                    self.races.push(DataRace {
                        access1: dep.source.clone(), access2: dep.sink.clone(),
                        addr: dep.source.addr.clone(), is_ordering_race: true,
                    });
                }
            }
        }
    }

    fn has_ordering_guarantee(&self, a: &MemoryAccess, b: &MemoryAccess) -> bool {
        !matches!(a.ordering, MemoryOrdering::NonAtomic) &&
        !matches!(b.ordering, MemoryOrdering::NonAtomic)
    }

    fn has_sufficient_ordering(&self, a: &MemoryAccess, b: &MemoryAccess) -> bool {
        matches!(
            (a.ordering, b.ordering),
            (MemoryOrdering::SeqCst, MemoryOrdering::SeqCst)
            | (MemoryOrdering::Release, MemoryOrdering::Acquire)
            | (MemoryOrdering::AcqRel, _)
            | (_, MemoryOrdering::AcqRel)
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Fence Placement Analysis
// ═══════════════════════════════════════════════════════════════════════════

/// Fence placement suggestion.
#[derive(Debug, Clone)]
pub struct FencePlacement {
    pub block: BlockId,
    pub position: usize,
    pub kind: FenceKind,
    pub reason: String,
    pub eliminates_races: Vec<usize>,
}

/// Optimal fence placement analyzer.
#[derive(Debug)]
pub struct FencePlacementAnalyzer {
    pub placements: Vec<FencePlacement>,
}

impl FencePlacementAnalyzer {
    pub fn new() -> Self {
        FencePlacementAnalyzer { placements: Vec::new() }
    }

    pub fn suggest_fences(&mut self, races: &[DataRace], cfg: &ControlFlowGraph) {
        let mut race_groups: HashMap<BlockId, Vec<(usize, &DataRace)>> = HashMap::new();
        for (i, race) in races.iter().enumerate() {
            race_groups.entry(race.access1.block).or_default().push((i, race));
            race_groups.entry(race.access2.block).or_default().push((i, race));
        }
        for (block_id, block_races) in &race_groups {
            let has_stores = cfg.blocks[*block_id].instructions.iter()
                .any(|inst| matches!(inst, DataflowInstruction::Store { .. }));
            let has_loads = cfg.blocks[*block_id].instructions.iter()
                .any(|inst| matches!(inst, DataflowInstruction::Load { .. }));
            let fence_kind = if has_stores && has_loads { FenceKind::Full }
                else if has_stores { FenceKind::StoreStore }
                else { FenceKind::LoadLoad };
            let race_indices: Vec<usize> = block_races.iter().map(|(i, _)| *i).collect();
            self.placements.push(FencePlacement {
                block: *block_id,
                position: cfg.blocks[*block_id].instructions.len(),
                kind: fence_kind,
                reason: format!("Eliminates {} races in block {}", race_indices.len(), block_id),
                eliminates_races: race_indices,
            });
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Taint Analysis
// ═══════════════════════════════════════════════════════════════════════════

/// Taint source.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TaintSource {
    pub variable: Variable,
    pub label: String,
}

/// Taint propagation state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaintState {
    pub tainted: HashMap<Variable, HashSet<String>>,
}

impl Lattice for TaintState {
    fn bottom() -> Self { TaintState { tainted: HashMap::new() } }
    fn top() -> Self { TaintState { tainted: HashMap::new() } }
    fn join(&self, other: &Self) -> Self {
        let mut result = self.tainted.clone();
        for (var, labels) in &other.tainted {
            let entry = result.entry(var.clone()).or_default();
            for label in labels { entry.insert(label.clone()); }
        }
        TaintState { tainted: result }
    }
    fn meet(&self, other: &Self) -> Self {
        let mut result = HashMap::new();
        for var in self.tainted.keys() {
            if let Some(other_labels) = other.tainted.get(var) {
                let labels: HashSet<String> = self.tainted[var].intersection(other_labels).cloned().collect();
                if !labels.is_empty() { result.insert(var.clone(), labels); }
            }
        }
        TaintState { tainted: result }
    }
    fn leq(&self, other: &Self) -> bool {
        for (var, labels) in &self.tainted {
            match other.tainted.get(var) {
                None => return false,
                Some(other_labels) => { if !labels.is_subset(other_labels) { return false; } }
            }
        }
        true
    }
}

/// Transfer function for taint analysis.
pub struct TaintTransfer {
    pub sources: Vec<TaintSource>,
}

impl TaintTransfer {
    fn collect_taints(&self, expr: &ValueExpr, state: &TaintState, result: &mut HashSet<String>) {
        match expr {
            ValueExpr::Var(v) => {
                if let Some(labels) = state.tainted.get(v) {
                    result.extend(labels.iter().cloned());
                }
            }
            ValueExpr::BinOp(l, _, r) => {
                self.collect_taints(l, state, result);
                self.collect_taints(r, state, result);
            }
            ValueExpr::UnaryOp(_, e) => { self.collect_taints(e, state, result); }
            ValueExpr::Const(_) => {}
        }
    }
}

impl TransferFunction<TaintState> for TaintTransfer {
    fn apply(&self, block: &BasicBlock, input: &TaintState) -> TaintState {
        let mut state = input.clone();
        for inst in &block.instructions {
            match inst {
                DataflowInstruction::Assign { dst, expr } => {
                    let mut taints = HashSet::new();
                    self.collect_taints(expr, &state, &mut taints);
                    if !taints.is_empty() { state.tainted.insert(dst.clone(), taints); }
                    else { state.tainted.remove(dst); }
                }
                DataflowInstruction::Load { dst, .. } => {
                    for source in &self.sources {
                        if source.variable == *dst {
                            state.tainted.entry(dst.clone()).or_default().insert(source.label.clone());
                        }
                    }
                }
                _ => {}
            }
        }
        state
    }
    fn direction(&self) -> AnalysisDirection { AnalysisDirection::Forward }
    fn initial_value(&self) -> TaintState { TaintState::bottom() }
    fn boundary_value(&self) -> TaintState {
        let mut tainted = HashMap::new();
        for source in &self.sources {
            tainted.entry(source.variable.clone()).or_insert_with(HashSet::new)
                .insert(source.label.clone());
        }
        TaintState { tainted }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Multi-Thread Dataflow
// ═══════════════════════════════════════════════════════════════════════════

/// Multi-threaded program CFG.
#[derive(Debug, Clone)]
pub struct MultiThreadCFG {
    pub threads: Vec<ControlFlowGraph>,
    pub shared_variables: HashSet<String>,
    pub initial_values: HashMap<String, i64>,
}

impl MultiThreadCFG {
    pub fn new() -> Self {
        MultiThreadCFG {
            threads: Vec::new(),
            shared_variables: HashSet::new(),
            initial_values: HashMap::new(),
        }
    }

    pub fn add_thread(&mut self, cfg: ControlFlowGraph) {
        self.threads.push(cfg);
    }

    pub fn add_shared(&mut self, name: &str, initial: i64) {
        self.shared_variables.insert(name.to_string());
        self.initial_values.insert(name.to_string(), initial);
    }

    pub fn analyze_per_thread(&self) -> Vec<DataflowResult<ReachingDefinitions>> {
        self.threads.iter().map(|cfg| reaching_definitions(cfg)).collect()
    }

    pub fn cross_thread_accesses(&self) -> Vec<MemoryAccess> {
        let mut accesses = Vec::new();
        for cfg in &self.threads {
            let mut analyzer = MemoryDependenceAnalyzer::new();
            analyzer.collect_accesses(cfg);
            for access in analyzer.accesses {
                if let MemoryAddr::Named(name) = &access.addr {
                    if self.shared_variables.contains(name) {
                        accesses.push(access);
                    }
                }
            }
        }
        accesses
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Happens-Before Analysis
// ═══════════════════════════════════════════════════════════════════════════

/// Vector clock for happens-before tracking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorClock {
    pub clocks: Vec<u64>,
}

impl VectorClock {
    pub fn new(num_threads: usize) -> Self {
        VectorClock { clocks: vec![0; num_threads] }
    }

    pub fn increment(&mut self, thread: usize) {
        if thread < self.clocks.len() { self.clocks[thread] += 1; }
    }

    pub fn join(&self, other: &VectorClock) -> VectorClock {
        let len = self.clocks.len().max(other.clocks.len());
        let mut result = vec![0u64; len];
        for i in 0..len {
            let a = if i < self.clocks.len() { self.clocks[i] } else { 0 };
            let b = if i < other.clocks.len() { other.clocks[i] } else { 0 };
            result[i] = a.max(b);
        }
        VectorClock { clocks: result }
    }

    pub fn happens_before(&self, other: &VectorClock) -> bool {
        let len = self.clocks.len().max(other.clocks.len());
        for i in 0..len {
            let a = if i < self.clocks.len() { self.clocks[i] } else { 0 };
            let b = if i < other.clocks.len() { other.clocks[i] } else { 0 };
            if a > b { return false; }
        }
        true
    }

    pub fn concurrent_with(&self, other: &VectorClock) -> bool {
        !self.happens_before(other) && !other.happens_before(self)
    }
}

/// Happens-before graph for memory model analysis.
#[derive(Debug)]
pub struct HappensBeforeGraph {
    pub events: Vec<HBEvent>,
    pub edges: Vec<(usize, usize)>,
    pub clocks: Vec<VectorClock>,
}

/// An event in the happens-before graph.
#[derive(Debug, Clone)]
pub struct HBEvent {
    pub thread: usize,
    pub access: MemoryAccess,
    pub clock: VectorClock,
}

impl HappensBeforeGraph {
    pub fn new(num_threads: usize) -> Self {
        HappensBeforeGraph {
            events: Vec::new(),
            edges: Vec::new(),
            clocks: (0..num_threads).map(|_| VectorClock::new(num_threads)).collect(),
        }
    }

    pub fn add_event(&mut self, thread: usize, access: MemoryAccess) -> usize {
        self.clocks[thread].increment(thread);
        let event = HBEvent {
            thread,
            access,
            clock: self.clocks[thread].clone(),
        };
        let idx = self.events.len();
        self.events.push(event);
        idx
    }

    pub fn add_sync_edge(&mut self, from: usize, to: usize) {
        self.edges.push((from, to));
        let from_thread = self.events[from].thread;
        let to_thread = self.events[to].thread;
        let from_clock = self.clocks[from_thread].clone();
        self.clocks[to_thread] = self.clocks[to_thread].join(&from_clock);
    }

    pub fn happens_before(&self, a: usize, b: usize) -> bool {
        self.events[a].clock.happens_before(&self.events[b].clock)
    }

    pub fn find_races(&self) -> Vec<(usize, usize)> {
        let mut races = Vec::new();
        for i in 0..self.events.len() {
            for j in (i + 1)..self.events.len() {
                let ei = &self.events[i];
                let ej = &self.events[j];
                if ei.thread == ej.thread { continue; }
                let same_addr = ei.access.addr == ej.access.addr;
                let one_write = ei.access.is_write || ej.access.is_write;
                let concurrent = ei.clock.concurrent_with(&ej.clock);
                if same_addr && one_write && concurrent {
                    races.push((i, j));
                }
            }
        }
        races
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Program Slicing
// ═══════════════════════════════════════════════════════════════════════════

/// A program slice criterion.
#[derive(Debug, Clone)]
pub struct SliceCriterion {
    pub block: BlockId,
    pub inst_index: usize,
    pub variables: HashSet<Variable>,
}

/// Program slicer using dataflow analysis.
#[derive(Debug)]
pub struct ProgramSlicer {
    pub slice_blocks: HashSet<BlockId>,
    pub slice_instructions: HashSet<(BlockId, usize)>,
}

impl ProgramSlicer {
    pub fn new() -> Self {
        ProgramSlicer {
            slice_blocks: HashSet::new(),
            slice_instructions: HashSet::new(),
        }
    }

    /// Compute backward slice from a criterion.
    pub fn backward_slice(&mut self, cfg: &ControlFlowGraph, criterion: &SliceCriterion) {
        let mut worklist: VecDeque<(BlockId, usize, HashSet<Variable>)> = VecDeque::new();
        worklist.push_back((criterion.block, criterion.inst_index, criterion.variables.clone()));
        let mut visited: HashSet<(BlockId, usize)> = HashSet::new();

        while let Some((block, inst_idx, needed_vars)) = worklist.pop_front() {
            if visited.contains(&(block, inst_idx)) { continue; }
            visited.insert((block, inst_idx));
            if needed_vars.is_empty() { continue; }

            self.slice_blocks.insert(block);
            self.slice_instructions.insert((block, inst_idx));

            if inst_idx > 0 {
                let inst = &cfg.blocks[block].instructions[inst_idx - 1];
                let defined = DefUseChains::defined_vars(inst);
                let used = DefUseChains::used_vars(inst);
                let mut new_needed = needed_vars.clone();
                let mut relevant = false;
                for def in &defined {
                    if needed_vars.contains(def) {
                        relevant = true;
                        new_needed.remove(def);
                        for u in &used { new_needed.insert(u.clone()); }
                    }
                }
                if relevant {
                    self.slice_instructions.insert((block, inst_idx - 1));
                }
                worklist.push_back((block, inst_idx - 1, new_needed));
            } else {
                for &pred in &cfg.blocks[block].predecessors {
                    let pred_len = cfg.blocks[pred].instructions.len();
                    worklist.push_back((pred, pred_len, needed_vars.clone()));
                }
            }
        }
    }

    pub fn get_slice(&self) -> &HashSet<(BlockId, usize)> {
        &self.slice_instructions
    }

    pub fn in_slice(&self, block: BlockId, inst: usize) -> bool {
        self.slice_instructions.contains(&(block, inst))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Dataflow Summary
// ═══════════════════════════════════════════════════════════════════════════

/// Summary of all dataflow analyses for a program.
#[derive(Debug)]
pub struct DataflowSummary {
    pub reaching_defs: Option<DataflowResult<ReachingDefinitions>>,
    pub live_vars: Option<DataflowResult<LiveVariables>>,
    pub constants: Option<DataflowResult<ConstantPropagation>>,
    pub intervals: Option<DataflowResult<IntervalState>>,
    pub races: Vec<DataRace>,
    pub fence_suggestions: Vec<FencePlacement>,
}

impl DataflowSummary {
    pub fn analyze(cfg: &ControlFlowGraph) -> Self {
        let reaching_defs = Some(reaching_definitions(cfg));
        let live_vars = Some(live_variables(cfg));
        let constants = Some(constant_propagation(cfg));
        let intervals = Some(interval_analysis(cfg));

        let mut mem_analyzer = MemoryDependenceAnalyzer::new();
        mem_analyzer.collect_accesses(cfg);
        mem_analyzer.compute_dependences();

        let mut race_detector = DataRaceDetector::new();
        race_detector.detect(&mem_analyzer);

        let mut fence_analyzer = FencePlacementAnalyzer::new();
        fence_analyzer.suggest_fences(&race_detector.races, cfg);

        DataflowSummary {
            reaching_defs, live_vars, constants, intervals,
            races: race_detector.races,
            fence_suggestions: fence_analyzer.placements,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Builder Utilities
// ═══════════════════════════════════════════════════════════════════════════

/// Builder for constructing CFGs from litmus test descriptions.
pub struct CFGBuilder {
    cfg: ControlFlowGraph,
    current_block: BlockId,
}

impl CFGBuilder {
    pub fn new(thread_id: usize) -> Self {
        CFGBuilder {
            cfg: ControlFlowGraph::new(thread_id),
            current_block: 0,
        }
    }

    pub fn load(&mut self, dst: &str, addr: &str, ordering: MemoryOrdering) -> &mut Self {
        let inst = DataflowInstruction::Load {
            dst: Variable::new(dst, self.cfg.thread_id),
            addr: MemoryAddr::Named(addr.to_string()),
            ordering,
        };
        self.cfg.blocks[self.current_block].add_instruction(inst);
        self
    }

    pub fn store(&mut self, addr: &str, value: i64, ordering: MemoryOrdering) -> &mut Self {
        let inst = DataflowInstruction::Store {
            addr: MemoryAddr::Named(addr.to_string()),
            src: ValueExpr::Const(value),
            ordering,
        };
        self.cfg.blocks[self.current_block].add_instruction(inst);
        self
    }

    pub fn store_var(&mut self, addr: &str, var: &str, ordering: MemoryOrdering) -> &mut Self {
        let inst = DataflowInstruction::Store {
            addr: MemoryAddr::Named(addr.to_string()),
            src: ValueExpr::Var(Variable::new(var, self.cfg.thread_id)),
            ordering,
        };
        self.cfg.blocks[self.current_block].add_instruction(inst);
        self
    }

    pub fn assign_const(&mut self, dst: &str, value: i64) -> &mut Self {
        let inst = DataflowInstruction::Assign {
            dst: Variable::new(dst, self.cfg.thread_id),
            expr: ValueExpr::Const(value),
        };
        self.cfg.blocks[self.current_block].add_instruction(inst);
        self
    }

    pub fn assign_binop(&mut self, dst: &str, lhs: &str, op: BinOp, rhs: &str) -> &mut Self {
        let tid = self.cfg.thread_id;
        let inst = DataflowInstruction::Assign {
            dst: Variable::new(dst, tid),
            expr: ValueExpr::BinOp(
                Box::new(ValueExpr::Var(Variable::new(lhs, tid))),
                op,
                Box::new(ValueExpr::Var(Variable::new(rhs, tid))),
            ),
        };
        self.cfg.blocks[self.current_block].add_instruction(inst);
        self
    }

    pub fn fence(&mut self, kind: FenceKind) -> &mut Self {
        let inst = DataflowInstruction::Fence { kind };
        self.cfg.blocks[self.current_block].add_instruction(inst);
        self
    }

    pub fn new_block(&mut self) -> BlockId {
        let block_id = self.cfg.add_block();
        self.cfg.add_edge(self.current_block, block_id);
        self.current_block = block_id;
        block_id
    }

    pub fn set_exit(&mut self) {
        self.cfg.blocks[self.current_block].is_exit = true;
        self.cfg.exits.push(self.current_block);
    }

    pub fn build(self) -> ControlFlowGraph {
        self.cfg
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_cfg() -> ControlFlowGraph {
        let mut builder = CFGBuilder::new(0);
        builder.store("x", 1, MemoryOrdering::Relaxed);
        builder.load("r1", "x", MemoryOrdering::Relaxed);
        builder.assign_const("r2", 42);
        builder.set_exit();
        builder.build()
    }

    #[test]
    fn test_reaching_definitions() {
        let cfg = simple_cfg();
        let result = reaching_definitions(&cfg);
        assert!(result.converged);
    }

    #[test]
    fn test_live_variables() {
        let cfg = simple_cfg();
        let result = live_variables(&cfg);
        assert!(result.converged);
    }

    #[test]
    fn test_constant_propagation() {
        let cfg = simple_cfg();
        let result = constant_propagation(&cfg);
        assert!(result.converged);
    }

    #[test]
    fn test_interval_analysis() {
        let cfg = simple_cfg();
        let result = interval_analysis(&cfg);
        assert!(result.converged);
    }

    #[test]
    fn test_dominator_tree() {
        let mut cfg = simple_cfg();
        cfg.compute_dominators();
        assert!(cfg.dominators.is_some());
    }

    #[test]
    fn test_vector_clock() {
        let mut vc1 = VectorClock::new(3);
        let mut vc2 = VectorClock::new(3);
        vc1.increment(0);
        vc2.increment(1);
        assert!(vc1.concurrent_with(&vc2));
        let joined = vc1.join(&vc2);
        assert!(vc1.happens_before(&joined));
        assert!(vc2.happens_before(&joined));
    }

    #[test]
    fn test_interval_lattice() {
        let a = IntervalLattice::Interval { lo: 1, hi: 5 };
        let b = IntervalLattice::Interval { lo: 3, hi: 8 };
        let joined = a.join(&b);
        assert_eq!(joined, IntervalLattice::Interval { lo: 1, hi: 8 });
        let met = a.meet(&b);
        assert_eq!(met, IntervalLattice::Interval { lo: 3, hi: 5 });
    }

    #[test]
    fn test_alias_analysis() {
        let mut aa = AliasAnalysis::new();
        aa.add_distinct("x", "y");
        assert!(aa.must_alias(&MemoryAddr::Named("x".into()), &MemoryAddr::Named("x".into())));
        assert!(!aa.may_alias(&MemoryAddr::Named("x".into()), &MemoryAddr::Named("y".into())));
    }

    #[test]
    fn test_powerset_lattice() {
        let mut a = PowersetLattice::<u32>::bottom();
        a.elements.insert(1);
        a.elements.insert(2);
        let mut b = PowersetLattice::<u32>::bottom();
        b.elements.insert(2);
        b.elements.insert(3);
        let joined = a.join(&b);
        assert_eq!(joined.elements.len(), 3);
        let met = a.meet(&b);
        assert_eq!(met.elements.len(), 1);
    }

    #[test]
    fn test_def_use_chains() {
        let cfg = simple_cfg();
        let chains = DefUseChains::compute(&cfg);
        assert!(!chains.defs.is_empty());
    }

    #[test]
    fn test_memory_dependence() {
        let cfg = simple_cfg();
        let mut analyzer = MemoryDependenceAnalyzer::new();
        analyzer.collect_accesses(&cfg);
        assert!(!analyzer.accesses.is_empty());
        analyzer.compute_dependences();
    }

    #[test]
    fn test_multi_thread_cfg() {
        let mut mt = MultiThreadCFG::new();
        mt.add_shared("x", 0);
        let mut b1 = CFGBuilder::new(0);
        b1.store("x", 1, MemoryOrdering::Release);
        b1.set_exit();
        mt.add_thread(b1.build());
        let mut b2 = CFGBuilder::new(1);
        b2.load("r1", "x", MemoryOrdering::Acquire);
        b2.set_exit();
        mt.add_thread(b2.build());
        let results = mt.analyze_per_thread();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_ssa_construction() {
        let cfg = simple_cfg();
        let ssa = SSAForm::construct(cfg);
        assert!(ssa.cfg.dominators.is_some());
    }

    #[test]
    fn test_program_slicer() {
        let cfg = simple_cfg();
        let mut slicer = ProgramSlicer::new();
        let criterion = SliceCriterion {
            block: 0,
            inst_index: 2,
            variables: [Variable::new("r2", 0)].into_iter().collect(),
        };
        slicer.backward_slice(&cfg, &criterion);
        assert!(!slicer.slice_instructions.is_empty());
    }

    #[test]
    fn test_dataflow_summary() {
        let cfg = simple_cfg();
        let summary = DataflowSummary::analyze(&cfg);
        assert!(summary.reaching_defs.is_some());
        assert!(summary.live_vars.is_some());
        assert!(summary.constants.is_some());
        assert!(summary.intervals.is_some());
    }

    #[test]
    fn test_happens_before_graph() {
        let mut hb = HappensBeforeGraph::new(2);
        let addr = MemoryAddr::Named("x".into());
        let a0 = hb.add_event(0, MemoryAccess {
            block: 0, inst_index: 0, addr: addr.clone(),
            is_write: true, ordering: MemoryOrdering::Release, thread_id: 0,
        });
        let a1 = hb.add_event(1, MemoryAccess {
            block: 0, inst_index: 0, addr: addr.clone(),
            is_write: false, ordering: MemoryOrdering::Acquire, thread_id: 1,
        });
        hb.add_sync_edge(a0, a1);
        assert!(hb.happens_before(a0, a1));
    }

    #[test]
    fn test_cfg_builder() {
        let mut builder = CFGBuilder::new(0);
        builder.store("x", 1, MemoryOrdering::SeqCst);
        builder.fence(FenceKind::Full);
        builder.load("r1", "y", MemoryOrdering::SeqCst);
        let _b2 = builder.new_block();
        builder.assign_const("r2", 0);
        builder.set_exit();
        let cfg = builder.build();
        assert_eq!(cfg.num_blocks(), 2);
    }
}
