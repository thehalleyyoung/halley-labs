//! Data and control dependency analysis.
//!
//! Builds program dependence graphs combining def-use chains (data dependencies)
//! and control dependencies. Supports transitive closure computation and
//! protocol-negotiation-specialized dependency extraction.

use std::collections::{HashMap, HashSet, VecDeque, BTreeSet};
use std::fmt;
use serde::{Serialize, Deserialize};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use indexmap::IndexMap;

use crate::ir::{Module, Function, BasicBlock, Instruction, Value, Type};
use crate::cfg::{CFG, DominatorTree, PostDominatorTree, ControlDependence};
use crate::{InstructionId, SlicerError, SlicerResult, NegotiationPhase};

// ---------------------------------------------------------------------------
// Dependency types
// ---------------------------------------------------------------------------

/// A data dependency: an instruction uses a value defined by another.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DataDependency {
    /// The instruction that defines the value.
    pub def: InstructionId,
    /// The instruction that uses the value.
    pub use_site: InstructionId,
    /// The register name through which the dependency flows.
    pub variable: String,
    /// Whether this is a memory-based dependency (through load/store).
    pub is_memory: bool,
}

/// A control dependency: an instruction's execution depends on a branch.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ControlDependencyEdge {
    /// The branch instruction that controls execution.
    pub branch: InstructionId,
    /// The instruction whose execution is controlled.
    pub dependent: InstructionId,
    /// The block containing the branch.
    pub branch_block: String,
    /// The block containing the dependent instruction.
    pub dependent_block: String,
}

/// Kind of dependency edge in the PDG.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DepEdgeKind {
    Data { variable: String, is_memory: bool },
    Control { branch_block: String },
}

// ---------------------------------------------------------------------------
// Dependency graph
// ---------------------------------------------------------------------------

/// Combined data and control dependency graph for a function.
pub struct DependencyGraph {
    pub function_name: String,
    graph: DiGraph<InstructionId, DepEdgeKind>,
    node_map: HashMap<InstructionId, NodeIndex>,
    /// Data dependencies indexed by defining instruction.
    data_deps: Vec<DataDependency>,
    /// Control dependencies.
    control_deps: Vec<ControlDependencyEdge>,
}

impl DependencyGraph {
    pub fn new(function_name: impl Into<String>) -> Self {
        Self {
            function_name: function_name.into(),
            graph: DiGraph::new(),
            node_map: HashMap::new(),
            data_deps: Vec::new(),
            control_deps: Vec::new(),
        }
    }

    fn get_or_create_node(&mut self, id: &InstructionId) -> NodeIndex {
        if let Some(&idx) = self.node_map.get(id) {
            idx
        } else {
            let idx = self.graph.add_node(id.clone());
            self.node_map.insert(id.clone(), idx);
            idx
        }
    }

    /// Add a data dependency.
    pub fn add_data_dep(&mut self, dep: DataDependency) {
        let def_idx = self.get_or_create_node(&dep.def);
        let use_idx = self.get_or_create_node(&dep.use_site);
        let kind = DepEdgeKind::Data {
            variable: dep.variable.clone(),
            is_memory: dep.is_memory,
        };
        self.graph.add_edge(def_idx, use_idx, kind);
        self.data_deps.push(dep);
    }

    /// Add a control dependency.
    pub fn add_control_dep(&mut self, dep: ControlDependencyEdge) {
        let branch_idx = self.get_or_create_node(&dep.branch);
        let dep_idx = self.get_or_create_node(&dep.dependent);
        let kind = DepEdgeKind::Control { branch_block: dep.branch_block.clone() };
        self.graph.add_edge(branch_idx, dep_idx, kind);
        self.control_deps.push(dep);
    }

    /// Get all instructions that an instruction depends on (backward slice seeds).
    pub fn dependencies_of(&self, id: &InstructionId) -> Vec<&InstructionId> {
        self.node_map.get(id).map(|&idx| {
            self.graph.neighbors_directed(idx, petgraph::Direction::Incoming)
                .map(|n| &self.graph[n])
                .collect()
        }).unwrap_or_default()
    }

    /// Get all instructions that depend on a given instruction.
    pub fn dependents_of(&self, id: &InstructionId) -> Vec<&InstructionId> {
        self.node_map.get(id).map(|&idx| {
            self.graph.neighbors(idx)
                .map(|n| &self.graph[n])
                .collect()
        }).unwrap_or_default()
    }

    /// Compute the transitive closure of backward dependencies from a set of seeds.
    pub fn transitive_closure_backward(&self, seeds: &[InstructionId]) -> HashSet<InstructionId> {
        let mut result = HashSet::new();
        let mut worklist: VecDeque<InstructionId> = seeds.iter().cloned().collect();

        while let Some(id) = worklist.pop_front() {
            if !result.insert(id.clone()) { continue; }
            for dep in self.dependencies_of(&id) {
                if !result.contains(dep) {
                    worklist.push_back(dep.clone());
                }
            }
        }
        result
    }

    /// Compute the transitive closure of forward dependencies from a set of seeds.
    pub fn transitive_closure_forward(&self, seeds: &[InstructionId]) -> HashSet<InstructionId> {
        let mut result = HashSet::new();
        let mut worklist: VecDeque<InstructionId> = seeds.iter().cloned().collect();

        while let Some(id) = worklist.pop_front() {
            if !result.insert(id.clone()) { continue; }
            for dep in self.dependents_of(&id) {
                if !result.contains(dep) {
                    worklist.push_back(dep.clone());
                }
            }
        }
        result
    }

    /// Total number of instruction nodes.
    pub fn num_instructions(&self) -> usize {
        self.node_map.len()
    }

    /// Total number of dependency edges.
    pub fn num_edges(&self) -> usize {
        self.graph.edge_count()
    }

    /// Number of data dependencies.
    pub fn num_data_deps(&self) -> usize {
        self.data_deps.len()
    }

    /// Number of control dependencies.
    pub fn num_control_deps(&self) -> usize {
        self.control_deps.len()
    }

    /// Get all data dependencies.
    pub fn data_dependencies(&self) -> &[DataDependency] {
        &self.data_deps
    }

    /// Get all control dependencies.
    pub fn control_dependencies(&self) -> &[ControlDependencyEdge] {
        &self.control_deps
    }
}

// ---------------------------------------------------------------------------
// Program Dependence Graph builder
// ---------------------------------------------------------------------------

/// Builder for program dependence graphs.
pub struct ProgramDependenceGraph;

impl ProgramDependenceGraph {
    /// Build a PDG for a function.
    pub fn build(func: &Function) -> DependencyGraph {
        let mut pdg = DependencyGraph::new(&func.name);

        // Build def-use chains (data dependencies).
        Self::build_data_deps(func, &mut pdg);

        // Build control dependencies.
        Self::build_control_deps(func, &mut pdg);

        pdg
    }

    /// Build data dependencies from def-use chains.
    fn build_data_deps(func: &Function, pdg: &mut DependencyGraph) {
        // Map from register name to its defining instruction.
        let mut def_map: HashMap<String, InstructionId> = HashMap::new();

        // First pass: collect all definitions.
        for (bname, block) in &func.blocks {
            for (idx, instr) in block.instructions.iter().enumerate() {
                if let Some(dest) = instr.dest() {
                    def_map.insert(
                        dest.to_string(),
                        InstructionId::new(&func.name, bname, idx),
                    );
                }
            }
        }

        // Second pass: for each use, create a data dependency to its definition.
        for (bname, block) in &func.blocks {
            for (idx, instr) in block.instructions.iter().enumerate() {
                let use_site = InstructionId::new(&func.name, bname, idx);

                for reg in instr.used_registers() {
                    if let Some(def_id) = def_map.get(reg) {
                        pdg.add_data_dep(DataDependency {
                            def: def_id.clone(),
                            use_site: use_site.clone(),
                            variable: reg.to_string(),
                            is_memory: false,
                        });
                    }
                }

                // Memory dependencies for load/store pairs.
                if let Instruction::Load { dest, ptr, .. } = instr {
                    if let Some(ptr_name) = ptr.name() {
                        // Find stores to the same pointer.
                        Self::find_store_deps(func, ptr_name, &use_site, pdg);
                    }
                }
            }
        }

        // Parameter definitions: treat parameters as defined at function entry.
        if let Some((entry_name, _)) = func.blocks.iter().next() {
            for (pname, _) in &func.params {
                let param_def = InstructionId::new(&func.name, entry_name, 0);
                // Find uses of this parameter.
                for (bname, block) in &func.blocks {
                    for (idx, instr) in block.instructions.iter().enumerate() {
                        if instr.used_registers().contains(&pname.as_str()) {
                            pdg.add_data_dep(DataDependency {
                                def: param_def.clone(),
                                use_site: InstructionId::new(&func.name, bname, idx),
                                variable: pname.clone(),
                                is_memory: false,
                            });
                        }
                    }
                }
            }
        }
    }

    /// Find store instructions that define what a load reads.
    fn find_store_deps(
        func: &Function,
        ptr_name: &str,
        load_id: &InstructionId,
        pdg: &mut DependencyGraph,
    ) {
        for (bname, block) in &func.blocks {
            for (idx, instr) in block.instructions.iter().enumerate() {
                if let Instruction::Store { ptr, .. } = instr {
                    if ptr.name() == Some(ptr_name) {
                        let store_id = InstructionId::new(&func.name, bname, idx);
                        pdg.add_data_dep(DataDependency {
                            def: store_id,
                            use_site: load_id.clone(),
                            variable: format!("*{}", ptr_name),
                            is_memory: true,
                        });
                    }
                }
            }
        }
    }

    /// Build control dependencies.
    fn build_control_deps(func: &Function, pdg: &mut DependencyGraph) {
        let cfg = CFG::from_function(func);
        let pdt = match PostDominatorTree::compute(&cfg) {
            Some(pdt) => pdt,
            None => return,
        };
        let cd = ControlDependence::compute(&cfg, &pdt);

        for (dep_block, ctrl_blocks) in &cd.dependences {
            for ctrl_block in ctrl_blocks {
                // The branch is the terminator of the control block.
                let branch_id = Self::terminator_id(func, ctrl_block);

                // All instructions in the dependent block are control-dependent on this branch.
                if let Some(block) = func.blocks.get(dep_block) {
                    for (idx, _instr) in block.instructions.iter().enumerate() {
                        if let Some(ref bid) = branch_id {
                            pdg.add_control_dep(ControlDependencyEdge {
                                branch: bid.clone(),
                                dependent: InstructionId::new(&func.name, dep_block, idx),
                                branch_block: ctrl_block.clone(),
                                dependent_block: dep_block.clone(),
                            });
                        }
                    }
                }
            }
        }
    }

    /// Get the instruction ID of the terminator in a block.
    fn terminator_id(func: &Function, block_name: &str) -> Option<InstructionId> {
        func.blocks.get(block_name).and_then(|block| {
            let len = block.instructions.len();
            if len > 0 && block.instructions[len - 1].is_terminator() {
                Some(InstructionId::new(&func.name, block_name, len - 1))
            } else {
                None
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Negotiation-specialized dependency analysis
// ---------------------------------------------------------------------------

/// Dependency analysis specialized for protocol negotiation.
pub struct NegotiationDependencyAnalysis<'a> {
    module: &'a Module,
    /// Phases to focus on.
    pub target_phases: Vec<NegotiationPhase>,
    /// Pre-built PDGs per function.
    pdgs: HashMap<String, DependencyGraph>,
}

impl<'a> NegotiationDependencyAnalysis<'a> {
    pub fn new(module: &'a Module) -> Self {
        Self {
            module,
            target_phases: vec![
                NegotiationPhase::CipherSuiteSelection,
                NegotiationPhase::VersionNegotiation,
                NegotiationPhase::ExtensionProcessing,
            ],
            pdgs: HashMap::new(),
        }
    }

    /// Build PDGs for all negotiation-relevant functions.
    pub fn build(&mut self) {
        for (fname, func) in &self.module.functions {
            if func.is_declaration { continue; }
            if self.is_relevant(func) {
                let pdg = ProgramDependenceGraph::build(func);
                self.pdgs.insert(fname.clone(), pdg);
            }
        }
    }

    /// Check if a function is relevant for the target negotiation phases.
    fn is_relevant(&self, func: &Function) -> bool {
        if func.is_negotiation_relevant() { return true; }
        for phase in &self.target_phases {
            if phase.matches_function(&func.name) { return true; }
        }
        false
    }

    /// Get the PDG for a function.
    pub fn pdg(&self, func_name: &str) -> Option<&DependencyGraph> {
        self.pdgs.get(func_name)
    }

    /// Find all instructions that influence a negotiation outcome.
    pub fn influencing_instructions(
        &self,
        outcome_func: &str,
        outcome_block: &str,
        outcome_idx: usize,
    ) -> HashSet<InstructionId> {
        let seed = InstructionId::new(outcome_func, outcome_block, outcome_idx);
        if let Some(pdg) = self.pdgs.get(outcome_func) {
            pdg.transitive_closure_backward(&[seed])
        } else {
            let mut result = HashSet::new();
            result.insert(seed);
            result
        }
    }

    /// Find all instructions influenced by a taint source.
    pub fn influenced_by(
        &self,
        source_func: &str,
        source_block: &str,
        source_idx: usize,
    ) -> HashSet<InstructionId> {
        let seed = InstructionId::new(source_func, source_block, source_idx);
        if let Some(pdg) = self.pdgs.get(source_func) {
            pdg.transitive_closure_forward(&[seed])
        } else {
            let mut result = HashSet::new();
            result.insert(seed);
            result
        }
    }

    /// Extract a dependency-based slice: the intersection of forward from sources
    /// and backward from sinks.
    pub fn extract_slice(
        &self,
        sources: &[InstructionId],
        sinks: &[InstructionId],
    ) -> HashSet<InstructionId> {
        let mut forward_set = HashSet::new();
        let mut backward_set = HashSet::new();

        for source in sources {
            if let Some(pdg) = self.pdgs.get(&source.function) {
                forward_set.extend(pdg.transitive_closure_forward(&[source.clone()]));
            }
        }

        for sink in sinks {
            if let Some(pdg) = self.pdgs.get(&sink.function) {
                backward_set.extend(pdg.transitive_closure_backward(&[sink.clone()]));
            }
        }

        if forward_set.is_empty() { return backward_set; }
        if backward_set.is_empty() { return forward_set; }
        forward_set.intersection(&backward_set).cloned().collect()
    }

    /// Statistics about the analysis.
    pub fn stats(&self) -> NegDepStats {
        let total_instrs: usize = self.pdgs.values().map(|p| p.num_instructions()).sum();
        let total_data: usize = self.pdgs.values().map(|p| p.num_data_deps()).sum();
        let total_ctrl: usize = self.pdgs.values().map(|p| p.num_control_deps()).sum();
        NegDepStats {
            num_functions_analyzed: self.pdgs.len(),
            total_instructions: total_instrs,
            total_data_deps: total_data,
            total_control_deps: total_ctrl,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NegDepStats {
    pub num_functions_analyzed: usize,
    pub total_instructions: usize,
    pub total_data_deps: usize,
    pub total_control_deps: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Module, Function, BasicBlock, Instruction, Value, Type, IntPredicate};

    fn make_dep_test_function() -> Function {
        let mut func = Function::new("ssl_select_cipher", Type::i32());
        func.add_param("ssl", Type::ptr(Type::NamedStruct("SSL".into())));

        let entry = func.add_block("entry");
        entry.push(Instruction::Load {
            dest: "cipher_list".into(),
            ty: Type::ptr(Type::i8()),
            ptr: Value::reg("ssl", Type::ptr(Type::NamedStruct("SSL".into()))),
            volatile: false,
            align: Some(8),
        });
        entry.push(Instruction::ICmp {
            dest: "is_empty".into(),
            pred: IntPredicate::Eq,
            lhs: Value::reg("cipher_list", Type::ptr(Type::i8())),
            rhs: Value::null(Type::i8()),
        });
        entry.push(Instruction::CondBr {
            cond: Value::reg("is_empty", Type::Int(1)),
            true_dest: "error".into(),
            false_dest: "select".into(),
        });

        let error = func.add_block("error");
        error.push(Instruction::Ret { value: Some(Value::int(0, 32)) });

        let select = func.add_block("select");
        select.push(Instruction::Call {
            dest: Some("chosen".into()),
            func: Value::func_ref("ssl3_choose_cipher", Type::func(Type::i32(), vec![])),
            args: vec![Value::reg("cipher_list", Type::ptr(Type::i8()))],
            ret_ty: Type::i32(),
            is_tail: false,
            calling_conv: None,
            attrs: vec![],
        });
        select.push(Instruction::Ret { value: Some(Value::reg("chosen", Type::i32())) });

        func.compute_predecessors();
        func
    }

    #[test]
    fn test_build_pdg() {
        let func = make_dep_test_function();
        let pdg = ProgramDependenceGraph::build(&func);
        assert!(pdg.num_instructions() > 0);
        assert!(pdg.num_data_deps() > 0);
    }

    #[test]
    fn test_data_dependency_chain() {
        let func = make_dep_test_function();
        let pdg = ProgramDependenceGraph::build(&func);

        // The load of cipher_list should be used by the icmp and the call.
        let load_id = InstructionId::new("ssl_select_cipher", "entry", 0);
        let deps = pdg.dependents_of(&load_id);
        assert!(!deps.is_empty());
    }

    #[test]
    fn test_control_dependency() {
        let func = make_dep_test_function();
        let pdg = ProgramDependenceGraph::build(&func);
        assert!(pdg.num_control_deps() > 0);
    }

    #[test]
    fn test_transitive_closure_backward() {
        let func = make_dep_test_function();
        let pdg = ProgramDependenceGraph::build(&func);

        // The return in "select" depends transitively on the load in "entry".
        let ret_id = InstructionId::new("ssl_select_cipher", "select", 1);
        let closure = pdg.transitive_closure_backward(&[ret_id]);
        assert!(closure.len() > 1);
    }

    #[test]
    fn test_transitive_closure_forward() {
        let func = make_dep_test_function();
        let pdg = ProgramDependenceGraph::build(&func);

        let load_id = InstructionId::new("ssl_select_cipher", "entry", 0);
        let closure = pdg.transitive_closure_forward(&[load_id]);
        assert!(closure.len() > 1);
    }

    #[test]
    fn test_negotiation_dependency_analysis() {
        let module = Module::test_module();
        let mut nda = NegotiationDependencyAnalysis::new(&module);
        nda.build();

        let stats = nda.stats();
        assert!(stats.num_functions_analyzed > 0);
    }

    #[test]
    fn test_negotiation_extract_slice() {
        let module = Module::test_module();
        let mut nda = NegotiationDependencyAnalysis::new(&module);
        nda.build();

        let sources = vec![
            InstructionId::new("ssl3_choose_cipher", "entry", 0),
        ];
        let sinks = vec![
            InstructionId::new("ssl3_choose_cipher", "select_loop", 0),
        ];
        let slice = nda.extract_slice(&sources, &sinks);
        // Should contain at least the source and possibly more.
        assert!(!slice.is_empty() || sources.is_empty());
    }

    #[test]
    fn test_dependency_graph_empty() {
        let pdg = DependencyGraph::new("empty_func");
        assert_eq!(pdg.num_instructions(), 0);
        assert_eq!(pdg.num_edges(), 0);
    }

    #[test]
    fn test_data_dep_struct() {
        let dep = DataDependency {
            def: InstructionId::new("f", "bb1", 0),
            use_site: InstructionId::new("f", "bb2", 1),
            variable: "x".into(),
            is_memory: false,
        };
        assert_eq!(dep.variable, "x");
        assert!(!dep.is_memory);
    }

    #[test]
    fn test_control_dep_struct() {
        let dep = ControlDependencyEdge {
            branch: InstructionId::new("f", "bb1", 2),
            dependent: InstructionId::new("f", "bb2", 0),
            branch_block: "bb1".into(),
            dependent_block: "bb2".into(),
        };
        assert_eq!(dep.branch_block, "bb1");
    }

    #[test]
    fn test_pdg_dependencies_of_unknown() {
        let pdg = DependencyGraph::new("test");
        let unknown = InstructionId::new("test", "unknown", 99);
        assert!(pdg.dependencies_of(&unknown).is_empty());
    }

    #[test]
    fn test_influencing_instructions() {
        let module = Module::test_module();
        let mut nda = NegotiationDependencyAnalysis::new(&module);
        nda.build();

        let influenced = nda.influencing_instructions(
            "ssl3_choose_cipher", "select_loop", 0,
        );
        assert!(!influenced.is_empty());
    }
}
