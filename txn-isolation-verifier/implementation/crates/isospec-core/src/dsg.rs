//! Direct Serialization Graph (DSG) construction and analysis.
use isospec_types::identifier::TransactionId;
use isospec_types::dependency::{Dependency, DependencyType};
use isospec_types::isolation::AnomalyClass;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::algo;
use std::collections::HashMap;

/// The Direct Serialization Graph.
#[derive(Debug, Clone)]
pub struct SerializationGraph {
    graph: DiGraph<TransactionId, DependencyType>,
    node_map: HashMap<TransactionId, NodeIndex>,
    dependencies: Vec<Dependency>,
}

impl SerializationGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
            dependencies: Vec::new(),
        }
    }

    pub fn add_transaction(&mut self, txn_id: TransactionId) {
        if !self.node_map.contains_key(&txn_id) {
            let idx = self.graph.add_node(txn_id);
            self.node_map.insert(txn_id, idx);
        }
    }

    pub fn add_dependency(&mut self, dep: Dependency) {
        self.add_transaction(dep.from_txn);
        self.add_transaction(dep.to_txn);
        let from_idx = self.node_map[&dep.from_txn];
        let to_idx = self.node_map[&dep.to_txn];
        self.graph.add_edge(from_idx, to_idx, dep.dep_type);
        self.dependencies.push(dep);
    }

    pub fn has_cycle(&self) -> bool {
        algo::is_cyclic_directed(&self.graph)
    }

    pub fn find_cycles(&self) -> Vec<Vec<TransactionId>> {
        let mut cycles = Vec::new();
        let sccs = petgraph::algo::tarjan_scc(&self.graph);
        for scc in sccs {
            if scc.len() > 1 {
                let cycle: Vec<TransactionId> = scc.iter()
                    .map(|idx| self.graph[*idx])
                    .collect();
                cycles.push(cycle);
            } else if scc.len() == 1 {
                let node = scc[0];
                if self.graph.neighbors(node).any(|n| n == node) {
                    cycles.push(vec![self.graph[node]]);
                }
            }
        }
        cycles
    }

    pub fn find_rw_cycles(&self) -> Vec<Vec<TransactionId>> {
        self.find_cycles_with_edge_type(|dt| dt.is_anti_dependency())
    }

    fn find_cycles_with_edge_type<F: Fn(DependencyType) -> bool>(&self, filter: F) -> Vec<Vec<TransactionId>> {
        let mut filtered = DiGraph::new();
        let mut fnode_map: HashMap<NodeIndex, NodeIndex> = HashMap::new();

        for node in self.graph.node_indices() {
            let new_idx = filtered.add_node(self.graph[node]);
            fnode_map.insert(node, new_idx);
        }

        for edge in self.graph.edge_indices() {
            let (src, dst) = self.graph.edge_endpoints(edge).unwrap();
            let weight = self.graph[edge];
            if filter(weight) {
                filtered.add_edge(fnode_map[&src], fnode_map[&dst], weight);
            }
        }

        let sccs = petgraph::algo::tarjan_scc(&filtered);
        let mut cycles = Vec::new();
        for scc in sccs {
            if scc.len() > 1 {
                let cycle: Vec<TransactionId> = scc.iter()
                    .map(|idx| filtered[*idx])
                    .collect();
                cycles.push(cycle);
            }
        }
        cycles
    }

    /// Detect specific anomaly classes by examining cycle structure.
    pub fn detect_anomaly(&self, anomaly: AnomalyClass) -> Vec<AnomalyInstance> {
        match anomaly {
            AnomalyClass::G0 => self.detect_g0(),
            AnomalyClass::G1a => self.detect_g1a(),
            AnomalyClass::G1b => self.detect_g1b(),
            AnomalyClass::G1c => self.detect_g1c(),
            AnomalyClass::G2Item => self.detect_g2_item(),
            AnomalyClass::G2 => self.detect_g2(),
        }
    }

    fn detect_g0(&self) -> Vec<AnomalyInstance> {
        // G0: Write-write cycle (dirty write)
        let mut instances = Vec::new();
        let cycles = self.find_cycles_of_type(DependencyType::WriteWrite);
        for cycle in cycles {
            let chain = self.deps_for_cycle(&cycle, |dt| dt == DependencyType::WriteWrite);
            instances.push(AnomalyInstance {
                anomaly_class: AnomalyClass::G0,
                involved_transactions: cycle,
                dependency_chain: chain,
                description: "Write-write cycle detected (dirty write)".into(),
            });
        }
        instances
    }

    fn detect_g1a(&self) -> Vec<AnomalyInstance> {
        // G1a: Aborted read - reading data written by an aborted transaction
        let mut instances = Vec::new();
        for dep in &self.dependencies {
            if dep.dep_type == DependencyType::WriteRead {
                instances.push(AnomalyInstance {
                    anomaly_class: AnomalyClass::G1a,
                    involved_transactions: vec![dep.from_txn, dep.to_txn],
                    dependency_chain: vec![dep.clone()],
                    description: format!("Potential aborted read: {} reads from {}", dep.to_txn, dep.from_txn),
                });
            }
        }
        instances
    }

    fn detect_g1b(&self) -> Vec<AnomalyInstance> {
        // G1b: Intermediate read
        let mut instances = Vec::new();
        for dep in &self.dependencies {
            if dep.dep_type == DependencyType::WriteRead {
                instances.push(AnomalyInstance {
                    anomaly_class: AnomalyClass::G1b,
                    involved_transactions: vec![dep.from_txn, dep.to_txn],
                    dependency_chain: vec![dep.clone()],
                    description: format!("Potential intermediate read: {} reads intermediate value from {}", dep.to_txn, dep.from_txn),
                });
            }
        }
        instances
    }

    fn detect_g1c(&self) -> Vec<AnomalyInstance> {
        // G1c: Circular information flow (ww+wr cycle)
        let mut instances = Vec::new();
        let cycles = self.find_cycles_of_type_mixed(&[DependencyType::WriteWrite, DependencyType::WriteRead]);
        for cycle in cycles {
            instances.push(AnomalyInstance {
                anomaly_class: AnomalyClass::G1c,
                involved_transactions: cycle.clone(),
                dependency_chain: self.deps_for_cycle(&cycle, |dt| dt == DependencyType::WriteWrite || dt == DependencyType::WriteRead),
                description: "Circular information flow detected".into(),
            });
        }
        instances
    }

    fn detect_g2_item(&self) -> Vec<AnomalyInstance> {
        // G2-item: Item anti-dependency cycle
        let mut instances = Vec::new();
        let cycles = self.find_rw_cycles();
        for cycle in cycles {
            if !cycle.is_empty() {
                instances.push(AnomalyInstance {
                    anomaly_class: AnomalyClass::G2Item,
                    involved_transactions: cycle.clone(),
                    dependency_chain: self.deps_for_cycle(&cycle, |_| true),
                    description: "Item anti-dependency cycle detected".into(),
                });
            }
        }
        instances
    }

    fn detect_g2(&self) -> Vec<AnomalyInstance> {
        // G2: Predicate anti-dependency (phantom)
        let mut instances = Vec::new();
        let pred_cycles = self.find_cycles_of_type_mixed(&[
            DependencyType::PredicateReadWrite,
            DependencyType::WriteRead,
            DependencyType::WriteWrite,
        ]);
        for cycle in pred_cycles {
            instances.push(AnomalyInstance {
                anomaly_class: AnomalyClass::G2,
                involved_transactions: cycle.clone(),
                dependency_chain: self.deps_for_cycle(&cycle, |dt| dt.is_predicate_level() || dt == DependencyType::WriteRead),
                description: "Predicate anti-dependency cycle (phantom) detected".into(),
            });
        }
        instances
    }

    fn find_cycles_of_type(&self, dep_type: DependencyType) -> Vec<Vec<TransactionId>> {
        self.find_cycles_with_edge_type(|dt| dt == dep_type)
    }

    fn find_cycles_of_type_mixed(&self, types: &[DependencyType]) -> Vec<Vec<TransactionId>> {
        self.find_cycles_with_edge_type(|dt| types.contains(&dt))
    }

    fn deps_for_cycle<F: Fn(DependencyType) -> bool>(&self, _cycle: &[TransactionId], filter: F) -> Vec<Dependency> {
        self.dependencies.iter()
            .filter(|d| filter(d.dep_type))
            .cloned()
            .collect()
    }

    pub fn transaction_count(&self) -> usize { self.node_map.len() }
    pub fn dependency_count(&self) -> usize { self.dependencies.len() }
    pub fn dependencies(&self) -> &[Dependency] { &self.dependencies }

    pub fn topological_order(&self) -> Option<Vec<TransactionId>> {
        algo::toposort(&self.graph, None)
            .ok()
            .map(|order| order.into_iter().map(|idx| self.graph[idx]).collect())
    }

    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph DSG {\n");
        for (txn, _) in &self.node_map {
            dot.push_str(&format!("  \"{}\";\n", txn));
        }
        for dep in &self.dependencies {
            dot.push_str(&format!("  \"{}\" -> \"{}\" [label=\"{}\"];\n",
                dep.from_txn, dep.to_txn, dep.dep_type));
        }
        dot.push_str("}\n");
        dot
    }
}

impl Default for SerializationGraph {
    fn default() -> Self { Self::new() }
}

/// An instance of a detected anomaly.
#[derive(Debug, Clone)]
pub struct AnomalyInstance {
    pub anomaly_class: AnomalyClass,
    pub involved_transactions: Vec<TransactionId>,
    pub dependency_chain: Vec<Dependency>,
    pub description: String,
}

impl std::fmt::Display for AnomalyInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {} (transactions: {:?})", self.anomaly_class, self.description, self.involved_transactions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph() {
        let g = SerializationGraph::new();
        assert!(!g.has_cycle());
        assert_eq!(g.transaction_count(), 0);
    }

    #[test]
    fn test_simple_cycle() {
        let mut g = SerializationGraph::new();
        let t1 = TransactionId::new(1);
        let t2 = TransactionId::new(2);
        g.add_dependency(Dependency::new(t1, t2, DependencyType::WriteRead));
        g.add_dependency(Dependency::new(t2, t1, DependencyType::ReadWrite));
        assert!(g.has_cycle());
        assert!(!g.find_cycles().is_empty());
    }

    #[test]
    fn test_no_cycle() {
        let mut g = SerializationGraph::new();
        let t1 = TransactionId::new(1);
        let t2 = TransactionId::new(2);
        let t3 = TransactionId::new(3);
        g.add_dependency(Dependency::new(t1, t2, DependencyType::WriteRead));
        g.add_dependency(Dependency::new(t2, t3, DependencyType::WriteRead));
        assert!(!g.has_cycle());
        assert!(g.topological_order().is_some());
    }

    #[test]
    fn test_g0_detection() {
        let mut g = SerializationGraph::new();
        let t1 = TransactionId::new(1);
        let t2 = TransactionId::new(2);
        g.add_dependency(Dependency::new(t1, t2, DependencyType::WriteWrite));
        g.add_dependency(Dependency::new(t2, t1, DependencyType::WriteWrite));
        let anomalies = g.detect_anomaly(AnomalyClass::G0);
        assert!(!anomalies.is_empty());
    }

    #[test]
    fn test_dot_output() {
        let mut g = SerializationGraph::new();
        g.add_dependency(Dependency::new(TransactionId::new(1), TransactionId::new(2), DependencyType::WriteRead));
        let dot = g.to_dot();
        assert!(dot.contains("digraph"));
        assert!(dot.contains("T1"));
        assert!(dot.contains("T2"));
    }
}
