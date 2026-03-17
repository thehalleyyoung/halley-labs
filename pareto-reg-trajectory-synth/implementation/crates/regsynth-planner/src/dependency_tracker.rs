use std::collections::{HashMap, HashSet, VecDeque};

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction;
use serde::{Deserialize, Serialize};

use crate::PlannerError;
use crate::PlannerResult;

// ─── Dependency Types ───────────────────────────────────────────────────────

/// Type of dependency between compliance tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DependencyType {
    /// Must complete A before starting B.
    Prerequisite,
    /// A and B share resources and cannot run simultaneously.
    ResourceConflict,
    /// Compliance with A is a regulatory prerequisite for B.
    Regulatory,
}

impl std::fmt::Display for DependencyType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Prerequisite => write!(f, "prerequisite"),
            Self::ResourceConflict => write!(f, "resource-conflict"),
            Self::Regulatory => write!(f, "regulatory"),
        }
    }
}

/// An edge in the dependency graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyEdge {
    pub dep_type: DependencyType,
    pub description: String,
    pub weight: f64,
}

impl DependencyEdge {
    pub fn new(dep_type: DependencyType) -> Self {
        Self {
            dep_type,
            description: String::new(),
            weight: 1.0,
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }
}

// ─── Task Node ──────────────────────────────────────────────────────────────

/// A task node in the dependency graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskNode {
    pub id: String,
    pub name: String,
    pub duration_days: f64,
}

impl TaskNode {
    pub fn new(id: impl Into<String>, name: impl Into<String>, duration: f64) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            duration_days: duration,
        }
    }
}

// ─── Slack Info ─────────────────────────────────────────────────────────────

/// Slack information for a task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlackInfo {
    pub task_id: String,
    pub earliest_start: f64,
    pub earliest_finish: f64,
    pub latest_start: f64,
    pub latest_finish: f64,
    pub total_slack: f64,
    pub free_slack: f64,
    pub is_critical: bool,
}

// ─── Bottleneck Info ────────────────────────────────────────────────────────

/// Information about a bottleneck task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckInfo {
    pub task_id: String,
    pub task_name: String,
    pub num_dependents: usize,
    pub num_transitive_dependents: usize,
    pub impact_score: f64,
}

// ─── Dependency Graph ───────────────────────────────────────────────────────

/// Dependency graph for compliance tasks using petgraph.
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    graph: DiGraph<TaskNode, DependencyEdge>,
    node_map: HashMap<String, NodeIndex>,
}

impl DependencyGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
        }
    }

    /// Add a task node to the graph.
    pub fn add_task(&mut self, task: TaskNode) -> NodeIndex {
        if let Some(&idx) = self.node_map.get(&task.id) {
            return idx;
        }
        let id = task.id.clone();
        let idx = self.graph.add_node(task);
        self.node_map.insert(id, idx);
        idx
    }

    /// Add a dependency edge: `from` must complete before `to` can start.
    pub fn add_dependency(
        &mut self,
        from: &str,
        to: &str,
        edge: DependencyEdge,
    ) -> PlannerResult<()> {
        let from_idx = self.node_map.get(from)
            .ok_or_else(|| PlannerError::TaskNotFound(from.to_string()))?;
        let to_idx = self.node_map.get(to)
            .ok_or_else(|| PlannerError::TaskNotFound(to.to_string()))?;
        self.graph.add_edge(*from_idx, *to_idx, edge);
        Ok(())
    }

    /// Number of tasks in the graph.
    pub fn task_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of dependency edges.
    pub fn dependency_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Get a task by ID.
    pub fn get_task(&self, id: &str) -> Option<&TaskNode> {
        self.node_map.get(id).map(|&idx| &self.graph[idx])
    }

    /// Get all direct predecessors of a task.
    pub fn predecessors(&self, task_id: &str) -> Vec<&TaskNode> {
        let Some(&idx) = self.node_map.get(task_id) else {
            return Vec::new();
        };
        self.graph.neighbors_directed(idx, Direction::Incoming)
            .map(|n| &self.graph[n])
            .collect()
    }

    /// Get all direct successors of a task.
    pub fn successors(&self, task_id: &str) -> Vec<&TaskNode> {
        let Some(&idx) = self.node_map.get(task_id) else {
            return Vec::new();
        };
        self.graph.neighbors_directed(idx, Direction::Outgoing)
            .map(|n| &self.graph[n])
            .collect()
    }

    /// Topological sort of all tasks.
    pub fn topological_sort(&self) -> PlannerResult<Vec<String>> {
        let mut in_degree: HashMap<NodeIndex, usize> = HashMap::new();
        for idx in self.graph.node_indices() {
            in_degree.insert(idx, self.graph.neighbors_directed(idx, Direction::Incoming).count());
        }

        let mut queue: VecDeque<NodeIndex> = in_degree.iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&idx, _)| idx)
            .collect();

        let mut order = Vec::new();
        while let Some(node) = queue.pop_front() {
            order.push(self.graph[node].id.clone());
            for neighbor in self.graph.neighbors_directed(node, Direction::Outgoing) {
                if let Some(deg) = in_degree.get_mut(&neighbor) {
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        if order.len() != self.graph.node_count() {
            Err(PlannerError::CircularDependency(
                "Circular dependency detected in task graph".to_string()
            ))
        } else {
            Ok(order)
        }
    }

    /// Detect cycles in the dependency graph.
    pub fn detect_cycles(&self) -> Vec<Vec<String>> {
        let mut cycles = Vec::new();
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();
        let mut path = Vec::new();

        for idx in self.graph.node_indices() {
            if !visited.contains(&idx) {
                self.dfs_find_cycles(idx, &mut visited, &mut rec_stack, &mut path, &mut cycles);
            }
        }
        cycles
    }

    fn dfs_find_cycles(
        &self,
        node: NodeIndex,
        visited: &mut HashSet<NodeIndex>,
        rec_stack: &mut HashSet<NodeIndex>,
        path: &mut Vec<NodeIndex>,
        cycles: &mut Vec<Vec<String>>,
    ) {
        visited.insert(node);
        rec_stack.insert(node);
        path.push(node);

        for neighbor in self.graph.neighbors_directed(node, Direction::Outgoing) {
            if !visited.contains(&neighbor) {
                self.dfs_find_cycles(neighbor, visited, rec_stack, path, cycles);
            } else if rec_stack.contains(&neighbor) {
                // Found a cycle: extract it
                let cycle_start = path.iter().position(|&n| n == neighbor).unwrap_or(0);
                let cycle: Vec<String> = path[cycle_start..].iter()
                    .map(|&n| self.graph[n].id.clone())
                    .collect();
                if !cycle.is_empty() {
                    cycles.push(cycle);
                }
            }
        }

        path.pop();
        rec_stack.remove(&node);
    }

    /// Find the critical path: the longest path through the graph by duration.
    pub fn find_critical_path(&self) -> PlannerResult<Vec<String>> {
        let topo = self.topological_sort()?;
        if topo.is_empty() {
            return Ok(Vec::new());
        }

        let mut earliest_finish: HashMap<&str, f64> = HashMap::new();
        let mut predecessor_on_cp: HashMap<&str, Option<&str>> = HashMap::new();

        for task_id in &topo {
            let idx = self.node_map[task_id.as_str()];
            let duration = self.graph[idx].duration_days;

            let mut best_ef = 0.0f64;
            let mut best_pred: Option<&str> = None;

            for pred_idx in self.graph.neighbors_directed(idx, Direction::Incoming) {
                let pred_id = &self.graph[pred_idx].id;
                let pred_ef = earliest_finish.get(pred_id.as_str()).copied().unwrap_or(0.0);
                if pred_ef > best_ef {
                    best_ef = pred_ef;
                    best_pred = Some(pred_id);
                }
            }

            earliest_finish.insert(task_id, best_ef + duration);
            predecessor_on_cp.insert(task_id, best_pred);
        }

        // Find task with maximum earliest finish
        let end_task = topo.iter()
            .max_by(|a, b| {
                earliest_finish[a.as_str()]
                    .partial_cmp(&earliest_finish[b.as_str()])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|s| s.as_str());

        // Trace back the critical path
        let mut cp = Vec::new();
        let mut current = end_task;
        while let Some(task_id) = current {
            cp.push(task_id.to_string());
            current = predecessor_on_cp.get(task_id).copied().flatten();
        }
        cp.reverse();
        Ok(cp)
    }

    /// Find bottleneck tasks: tasks that many other tasks depend on.
    pub fn find_bottlenecks(&self) -> Vec<BottleneckInfo> {
        let mut bottlenecks = Vec::new();

        for (id_str, &idx) in &self.node_map {
            let direct = self.graph.neighbors_directed(idx, Direction::Outgoing).count();

            // Count transitive dependents via BFS
            let mut visited = HashSet::new();
            let mut queue = VecDeque::new();
            queue.push_back(idx);
            visited.insert(idx);
            while let Some(node) = queue.pop_front() {
                for neighbor in self.graph.neighbors_directed(node, Direction::Outgoing) {
                    if visited.insert(neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
            let transitive = visited.len() - 1; // exclude self

            let duration = self.graph[idx].duration_days;
            let impact = (transitive as f64) * duration;

            bottlenecks.push(BottleneckInfo {
                task_id: id_str.to_string(),
                task_name: self.graph[idx].name.clone(),
                num_dependents: direct,
                num_transitive_dependents: transitive,
                impact_score: impact,
            });
        }

        bottlenecks.sort_by(|a, b| b.impact_score.partial_cmp(&a.impact_score).unwrap_or(std::cmp::Ordering::Equal));
        bottlenecks
    }

    /// Compute slack for each task.
    pub fn compute_slack(&self) -> PlannerResult<Vec<SlackInfo>> {
        let topo = self.topological_sort()?;
        if topo.is_empty() {
            return Ok(Vec::new());
        }

        // Forward pass
        let mut es: HashMap<&str, f64> = HashMap::new();
        let mut ef: HashMap<&str, f64> = HashMap::new();

        for task_id in &topo {
            let idx = self.node_map[task_id.as_str()];
            let duration = self.graph[idx].duration_days;

            let earliest_start = self.graph
                .neighbors_directed(idx, Direction::Incoming)
                .map(|p| ef.get(self.graph[p].id.as_str()).copied().unwrap_or(0.0))
                .fold(0.0f64, f64::max);

            es.insert(task_id, earliest_start);
            ef.insert(task_id, earliest_start + duration);
        }

        let project_duration = ef.values().cloned().fold(0.0f64, f64::max);

        // Backward pass
        let mut lf: HashMap<&str, f64> = HashMap::new();
        let mut ls: HashMap<&str, f64> = HashMap::new();

        for task_id in topo.iter().rev() {
            let idx = self.node_map[task_id.as_str()];
            let duration = self.graph[idx].duration_days;

            let latest_finish = self.graph
                .neighbors_directed(idx, Direction::Outgoing)
                .map(|s| ls.get(self.graph[s].id.as_str()).copied().unwrap_or(project_duration))
                .fold(project_duration, f64::min);

            lf.insert(task_id, latest_finish);
            ls.insert(task_id, latest_finish - duration);
        }

        // Compute slack info
        let mut result = Vec::new();
        for task_id in &topo {
            let idx = self.node_map[task_id.as_str()];
            let earliest_start = es[task_id.as_str()];
            let earliest_finish = ef[task_id.as_str()];
            let latest_finish = lf[task_id.as_str()];
            let latest_start = ls[task_id.as_str()];
            let total_slack = latest_start - earliest_start;

            // Free slack: minimum of (ES of successors - EF of this task)
            let free_slack = self.graph
                .neighbors_directed(idx, Direction::Outgoing)
                .map(|s| es.get(self.graph[s].id.as_str()).copied().unwrap_or(project_duration))
                .fold(project_duration, f64::min)
                - earliest_finish;
            let free_slack = free_slack.max(0.0);

            result.push(SlackInfo {
                task_id: task_id.clone(),
                earliest_start,
                earliest_finish,
                latest_start,
                latest_finish,
                total_slack,
                free_slack,
                is_critical: total_slack.abs() < 0.001,
            });
        }

        Ok(result)
    }

    /// Get all dependencies of a specific type.
    pub fn dependencies_of_type(&self, dep_type: DependencyType) -> Vec<(String, String)> {
        self.graph.edge_indices()
            .filter_map(|e| {
                let edge = &self.graph[e];
                if edge.dep_type == dep_type {
                    let (from, to) = self.graph.edge_endpoints(e)?;
                    Some((self.graph[from].id.clone(), self.graph[to].id.clone()))
                } else {
                    None
                }
            })
            .collect()
    }
}

impl Default for DependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_diamond_graph() -> DependencyGraph {
        let mut g = DependencyGraph::new();
        g.add_task(TaskNode::new("start", "Start", 5.0));
        g.add_task(TaskNode::new("left", "Left Path", 10.0));
        g.add_task(TaskNode::new("right", "Right Path", 3.0));
        g.add_task(TaskNode::new("end", "End", 2.0));
        g.add_dependency("start", "left", DependencyEdge::new(DependencyType::Prerequisite)).unwrap();
        g.add_dependency("start", "right", DependencyEdge::new(DependencyType::Prerequisite)).unwrap();
        g.add_dependency("left", "end", DependencyEdge::new(DependencyType::Prerequisite)).unwrap();
        g.add_dependency("right", "end", DependencyEdge::new(DependencyType::Prerequisite)).unwrap();
        g
    }

    #[test]
    fn test_add_tasks_and_deps() {
        let g = build_diamond_graph();
        assert_eq!(g.task_count(), 4);
        assert_eq!(g.dependency_count(), 4);
    }

    #[test]
    fn test_predecessors_successors() {
        let g = build_diamond_graph();
        let preds = g.predecessors("end");
        assert_eq!(preds.len(), 2);
        let succs = g.successors("start");
        assert_eq!(succs.len(), 2);
    }

    #[test]
    fn test_topological_sort() {
        let g = build_diamond_graph();
        let topo = g.topological_sort().unwrap();
        assert_eq!(topo.len(), 4);
        let start_pos = topo.iter().position(|s| s == "start").unwrap();
        let end_pos = topo.iter().position(|s| s == "end").unwrap();
        assert!(start_pos < end_pos);
    }

    #[test]
    fn test_cycle_detection_no_cycle() {
        let g = build_diamond_graph();
        let cycles = g.detect_cycles();
        assert!(cycles.is_empty());
    }

    #[test]
    fn test_cycle_detection_with_cycle() {
        let mut g = DependencyGraph::new();
        g.add_task(TaskNode::new("a", "A", 1.0));
        g.add_task(TaskNode::new("b", "B", 1.0));
        g.add_task(TaskNode::new("c", "C", 1.0));
        g.add_dependency("a", "b", DependencyEdge::new(DependencyType::Prerequisite)).unwrap();
        g.add_dependency("b", "c", DependencyEdge::new(DependencyType::Prerequisite)).unwrap();
        g.add_dependency("c", "a", DependencyEdge::new(DependencyType::Prerequisite)).unwrap();

        let cycles = g.detect_cycles();
        assert!(!cycles.is_empty());
        assert!(g.topological_sort().is_err());
    }

    #[test]
    fn test_critical_path() {
        let g = build_diamond_graph();
        let cp = g.find_critical_path().unwrap();
        // Critical path: start(5) -> left(10) -> end(2) = 17 days
        assert!(cp.contains(&"start".to_string()));
        assert!(cp.contains(&"left".to_string()));
        assert!(cp.contains(&"end".to_string()));
        assert!(!cp.contains(&"right".to_string()));
    }

    #[test]
    fn test_compute_slack() {
        let g = build_diamond_graph();
        let slacks = g.compute_slack().unwrap();

        let start_slack = slacks.iter().find(|s| s.task_id == "start").unwrap();
        assert!(start_slack.is_critical);

        let right_slack = slacks.iter().find(|s| s.task_id == "right").unwrap();
        assert!(!right_slack.is_critical);
        assert!(right_slack.total_slack > 0.0);
        // Right path slack = 10 - 3 = 7 days
        assert!((right_slack.total_slack - 7.0).abs() < 0.01);
    }

    #[test]
    fn test_find_bottlenecks() {
        let g = build_diamond_graph();
        let bottlenecks = g.find_bottlenecks();
        // "start" has 3 transitive dependents
        let start_bn = bottlenecks.iter().find(|b| b.task_id == "start").unwrap();
        assert_eq!(start_bn.num_transitive_dependents, 3);
    }

    #[test]
    fn test_dependencies_of_type() {
        let mut g = DependencyGraph::new();
        g.add_task(TaskNode::new("a", "A", 5.0));
        g.add_task(TaskNode::new("b", "B", 5.0));
        g.add_task(TaskNode::new("c", "C", 5.0));
        g.add_dependency("a", "b", DependencyEdge::new(DependencyType::Prerequisite)).unwrap();
        g.add_dependency("b", "c", DependencyEdge::new(DependencyType::Regulatory)).unwrap();

        let prereqs = g.dependencies_of_type(DependencyType::Prerequisite);
        assert_eq!(prereqs.len(), 1);
        assert_eq!(prereqs[0], ("a".to_string(), "b".to_string()));

        let reg = g.dependencies_of_type(DependencyType::Regulatory);
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn test_unknown_task_error() {
        let mut g = DependencyGraph::new();
        g.add_task(TaskNode::new("a", "A", 5.0));
        let result = g.add_dependency("a", "nonexistent", DependencyEdge::new(DependencyType::Prerequisite));
        assert!(result.is_err());
    }
}
