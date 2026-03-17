use serde::{Deserialize, Serialize};
use crate::address::VirtualAddress;
use crate::cfg::ControlFlowGraph;
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct FunctionId(pub u32);
impl FunctionId {
    pub fn new(id: u32) -> Self { Self(id) }
    pub fn as_usize(self) -> usize { self.0 as usize }
}
impl fmt::Display for FunctionId { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "fn{}", self.0) } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Function {
    pub id: FunctionId,
    pub name: String,
    pub entry_address: VirtualAddress,
    pub size: u64,
    pub cfg: ControlFlowGraph,
    pub is_leaf: bool,
    pub is_crypto: bool,
    pub unroll_count: Option<u32>,
    pub callees: Vec<FunctionId>,
}

impl Function {
    pub fn new(id: FunctionId, name: &str, entry: VirtualAddress) -> Self {
        Self { id, name: name.to_string(), entry_address: entry, size: 0,
               cfg: ControlFlowGraph::new(), is_leaf: true, is_crypto: false, unroll_count: None, callees: Vec::new() }
    }
    pub fn instruction_count(&self) -> usize { self.cfg.total_instructions() }
    pub fn block_count(&self) -> usize { self.cfg.block_count() }
}
impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{} ({} blocks)", self.name, self.entry_address, self.block_count())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallSite {
    pub caller: FunctionId, pub callee: FunctionId, pub address: VirtualAddress,
    pub is_indirect: bool, pub is_tail_call: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CallGraph {
    pub functions: HashMap<FunctionId, Vec<CallSite>>,
    pub callers: HashMap<FunctionId, Vec<FunctionId>>,
}

impl CallGraph {
    pub fn new() -> Self { Self::default() }
    pub fn add_call(&mut self, site: CallSite) {
        self.callers.entry(site.callee).or_default().push(site.caller);
        self.functions.entry(site.caller).or_default().push(site);
    }
    pub fn callees(&self, func: FunctionId) -> Vec<FunctionId> {
        self.functions.get(&func).map_or(Vec::new(), |s| s.iter().map(|s| s.callee).collect())
    }
    pub fn callers_of(&self, func: FunctionId) -> &[FunctionId] {
        self.callers.get(&func).map_or(&[], |v| v.as_slice())
    }
    pub fn reverse_topological_order(&self, funcs: &[FunctionId]) -> Vec<FunctionId> {
        let mut visited = std::collections::HashSet::new();
        let mut order = Vec::new();
        for &f in funcs { self.topo_visit(f, &mut visited, &mut order); }
        order
    }
    fn topo_visit(&self, func: FunctionId, visited: &mut std::collections::HashSet<FunctionId>, order: &mut Vec<FunctionId>) {
        if !visited.insert(func) { return; }
        for callee in self.callees(func) { self.topo_visit(callee, visited, order); }
        order.push(func);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Program {
    pub name: String, pub functions: Vec<Function>,
    pub call_graph: CallGraph, pub entry_function: Option<FunctionId>,
}

impl Program {
    pub fn new(name: &str) -> Self { Self { name: name.to_string(), functions: Vec::new(), call_graph: CallGraph::new(), entry_function: None } }
    pub fn add_function(&mut self, func: Function) { self.functions.push(func); }
    pub fn find_function(&self, name: &str) -> Option<&Function> { self.functions.iter().find(|f| f.name == name) }
    pub fn total_instructions(&self) -> usize { self.functions.iter().map(|f| f.instruction_count()).sum() }
    pub fn function_count(&self) -> usize { self.functions.len() }
    pub fn analysis_order(&self) -> Vec<FunctionId> {
        let ids: Vec<_> = self.functions.iter().map(|f| f.id).collect();
        self.call_graph.reverse_topological_order(&ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn test_prog() { let mut p = Program::new("t"); p.add_function(Function::new(FunctionId(0), "main", VirtualAddress(0x1000))); assert_eq!(p.function_count(), 1); }
}
