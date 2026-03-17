//! Cycle detection algorithms for serialization graphs.
use isospec_types::identifier::TransactionId;
use isospec_types::dependency::{Dependency, DependencyType};
use std::collections::{HashMap, HashSet, VecDeque};

/// Tarjan-based cycle detection.
pub struct CycleDetector {
    adj: HashMap<TransactionId, Vec<(TransactionId, DependencyType)>>,
}

impl CycleDetector {
    pub fn new() -> Self { Self { adj: HashMap::new() } }

    pub fn add_edge(&mut self, from: TransactionId, to: TransactionId, dep_type: DependencyType) {
        self.adj.entry(from).or_default().push((to, dep_type));
    }

    pub fn from_dependencies(deps: &[Dependency]) -> Self {
        let mut det = Self::new();
        for dep in deps {
            det.add_edge(dep.from_txn, dep.to_txn, dep.dep_type);
        }
        det
    }

    pub fn find_all_cycles(&self) -> Vec<Cycle> {
        let nodes: Vec<TransactionId> = self.adj.keys().copied().collect();
        let mut all_cycles = Vec::new();

        for start in &nodes {
            let cycles = self.find_cycles_from(*start);
            for cycle in cycles {
                if !self.is_duplicate(&all_cycles, &cycle) {
                    all_cycles.push(cycle);
                }
            }
        }
        all_cycles
    }

    fn find_cycles_from(&self, start: TransactionId) -> Vec<Cycle> {
        let mut cycles = Vec::new();
        let mut path = vec![start];
        let mut visited = HashSet::new();
        visited.insert(start);
        self.dfs_cycle(start, &mut path, &mut visited, &mut cycles, start);
        cycles
    }

    fn dfs_cycle(
        &self,
        current: TransactionId,
        path: &mut Vec<TransactionId>,
        visited: &mut HashSet<TransactionId>,
        cycles: &mut Vec<Cycle>,
        target: TransactionId,
    ) {
        if let Some(neighbors) = self.adj.get(&current) {
            for (next, dep_type) in neighbors {
                if *next == target && path.len() > 1 {
                    let mut cycle_edges = Vec::new();
                    for i in 0..path.len() {
                        let from = path[i];
                        let to = if i + 1 < path.len() { path[i + 1] } else { target };
                        if let Some(neighbors) = self.adj.get(&from) {
                            if let Some((_, dt)) = neighbors.iter().find(|(n, _)| *n == to) {
                                cycle_edges.push((from, to, *dt));
                            }
                        }
                    }
                    cycle_edges.push((current, target, *dep_type));
                    cycles.push(Cycle {
                        nodes: path.clone(),
                        edges: cycle_edges,
                    });
                } else if !visited.contains(next) && path.len() < 10 {
                    visited.insert(*next);
                    path.push(*next);
                    self.dfs_cycle(*next, path, visited, cycles, target);
                    path.pop();
                    visited.remove(next);
                }
            }
        }
    }

    fn is_duplicate(&self, existing: &[Cycle], new: &Cycle) -> bool {
        existing.iter().any(|c| {
            c.nodes.len() == new.nodes.len()
                && new.nodes.iter().all(|n| c.nodes.contains(n))
        })
    }

    /// Find shortest cycle (BFS-based).
    pub fn find_shortest_cycle(&self) -> Option<Cycle> {
        let nodes: Vec<TransactionId> = self.adj.keys().copied().collect();
        let mut best: Option<Cycle> = None;

        for start in &nodes {
            if let Some(cycle) = self.bfs_shortest_cycle(*start) {
                if best.as_ref().map_or(true, |b| cycle.nodes.len() < b.nodes.len()) {
                    best = Some(cycle);
                }
            }
        }
        best
    }

    fn bfs_shortest_cycle(&self, start: TransactionId) -> Option<Cycle> {
        let mut queue: VecDeque<(TransactionId, Vec<TransactionId>)> = VecDeque::new();
        queue.push_back((start, vec![start]));
        let mut visited = HashSet::new();

        while let Some((current, path)) = queue.pop_front() {
            if let Some(neighbors) = self.adj.get(&current) {
                for (next, _) in neighbors {
                    if *next == start && path.len() > 1 {
                        return Some(Cycle { nodes: path, edges: Vec::new() });
                    }
                    if !visited.contains(next) && path.len() < 10 {
                        visited.insert(*next);
                        let mut new_path = path.clone();
                        new_path.push(*next);
                        queue.push_back((*next, new_path));
                    }
                }
            }
        }
        None
    }

    /// Check if graph has any cycle.
    pub fn has_cycle(&self) -> bool {
        let nodes: HashSet<TransactionId> = self.adj.keys().copied().collect();
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for node in &nodes {
            if !visited.contains(node) {
                if self.has_cycle_dfs(*node, &mut visited, &mut rec_stack) {
                    return true;
                }
            }
        }
        false
    }

    fn has_cycle_dfs(
        &self,
        node: TransactionId,
        visited: &mut HashSet<TransactionId>,
        rec_stack: &mut HashSet<TransactionId>,
    ) -> bool {
        visited.insert(node);
        rec_stack.insert(node);

        if let Some(neighbors) = self.adj.get(&node) {
            for (next, _) in neighbors {
                if !visited.contains(next) {
                    if self.has_cycle_dfs(*next, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(next) {
                    return true;
                }
            }
        }
        rec_stack.remove(&node);
        false
    }

    /// Count edges by type.
    pub fn edge_counts(&self) -> HashMap<DependencyType, usize> {
        let mut counts = HashMap::new();
        for neighbors in self.adj.values() {
            for (_, dt) in neighbors {
                *counts.entry(*dt).or_insert(0) += 1;
            }
        }
        counts
    }

    pub fn node_count(&self) -> usize { self.adj.len() }
    pub fn edge_count(&self) -> usize { self.adj.values().map(|v| v.len()).sum() }
}

impl Default for CycleDetector {
    fn default() -> Self { Self::new() }
}

/// A cycle in the serialization graph.
#[derive(Debug, Clone)]
pub struct Cycle {
    pub nodes: Vec<TransactionId>,
    pub edges: Vec<(TransactionId, TransactionId, DependencyType)>,
}

impl Cycle {
    pub fn length(&self) -> usize { self.nodes.len() }

    pub fn contains_edge_type(&self, dt: DependencyType) -> bool {
        self.edges.iter().any(|(_, _, t)| *t == dt)
    }

    pub fn has_anti_dependency(&self) -> bool {
        self.edges.iter().any(|(_, _, t)| t.is_anti_dependency())
    }

    pub fn consecutive_rw_count(&self) -> usize {
        let mut max_consec = 0;
        let mut current = 0;
        for (_, _, dt) in &self.edges {
            if dt.is_anti_dependency() {
                current += 1;
                max_consec = max_consec.max(current);
            } else {
                current = 0;
            }
        }
        max_consec
    }
}

impl std::fmt::Display for Cycle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let txns: Vec<String> = self.nodes.iter().map(|t| format!("{}", t)).collect();
        write!(f, "[{}]", txns.join(" -> "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_cycle() {
        let mut det = CycleDetector::new();
        det.add_edge(TransactionId::new(1), TransactionId::new(2), DependencyType::WriteRead);
        det.add_edge(TransactionId::new(2), TransactionId::new(3), DependencyType::WriteRead);
        assert!(!det.has_cycle());
    }

    #[test]
    fn test_simple_cycle() {
        let mut det = CycleDetector::new();
        det.add_edge(TransactionId::new(1), TransactionId::new(2), DependencyType::WriteRead);
        det.add_edge(TransactionId::new(2), TransactionId::new(1), DependencyType::ReadWrite);
        assert!(det.has_cycle());
        let cycles = det.find_all_cycles();
        assert!(!cycles.is_empty());
    }

    #[test]
    fn test_shortest_cycle() {
        let mut det = CycleDetector::new();
        det.add_edge(TransactionId::new(1), TransactionId::new(2), DependencyType::WriteRead);
        det.add_edge(TransactionId::new(2), TransactionId::new(3), DependencyType::WriteRead);
        det.add_edge(TransactionId::new(3), TransactionId::new(1), DependencyType::ReadWrite);
        det.add_edge(TransactionId::new(1), TransactionId::new(3), DependencyType::WriteWrite);
        det.add_edge(TransactionId::new(3), TransactionId::new(1), DependencyType::WriteWrite);
        let shortest = det.find_shortest_cycle().unwrap();
        assert!(shortest.length() <= 3);
    }

    #[test]
    fn test_from_dependencies() {
        let deps = vec![
            Dependency::new(TransactionId::new(1), TransactionId::new(2), DependencyType::WriteRead),
            Dependency::new(TransactionId::new(2), TransactionId::new(1), DependencyType::ReadWrite),
        ];
        let det = CycleDetector::from_dependencies(&deps);
        assert!(det.has_cycle());
        assert_eq!(det.node_count(), 2);
        assert_eq!(det.edge_count(), 2);
    }

    #[test]
    fn test_edge_counts() {
        let mut det = CycleDetector::new();
        det.add_edge(TransactionId::new(1), TransactionId::new(2), DependencyType::WriteRead);
        det.add_edge(TransactionId::new(2), TransactionId::new(3), DependencyType::WriteRead);
        det.add_edge(TransactionId::new(3), TransactionId::new(1), DependencyType::ReadWrite);
        let counts = det.edge_counts();
        assert_eq!(counts[&DependencyType::WriteRead], 2);
        assert_eq!(counts[&DependencyType::ReadWrite], 1);
    }
}
