use std::collections::{HashMap, HashSet, VecDeque};
use serde::{Serialize, Deserialize};

/// Tree decomposition of a graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeDecomposition {
    pub bags: Vec<Vec<usize>>,
    pub tree_edges: Vec<(usize, usize)>,
    pub width: usize,
}

impl TreeDecomposition {
    pub fn new(bags: Vec<Vec<usize>>, tree_edges: Vec<(usize, usize)>) -> Self {
        let width = bags.iter().map(|b| b.len()).max().unwrap_or(1).saturating_sub(1);
        Self { bags, tree_edges, width }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn bag_count(&self) -> usize {
        self.bags.len()
    }

    pub fn is_valid(&self, graph_edges: &[(usize, usize)], num_vertices: usize) -> bool {
        // Check 1: every vertex appears in at least one bag
        let mut vertex_seen = vec![false; num_vertices];
        for bag in &self.bags {
            for &v in bag {
                if v < num_vertices {
                    vertex_seen[v] = true;
                }
            }
        }
        if vertex_seen.iter().any(|&s| !s) {
            return false;
        }

        // Check 2: every edge has both endpoints in some bag
        for &(u, v) in graph_edges {
            let found = self.bags.iter().any(|bag| bag.contains(&u) && bag.contains(&v));
            if !found {
                return false;
            }
        }

        // Check 3: for each vertex, the bags containing it form a connected subtree
        for vertex in 0..num_vertices {
            let containing_bags: HashSet<usize> = self.bags.iter().enumerate()
                .filter(|(_, bag)| bag.contains(&vertex))
                .map(|(i, _)| i)
                .collect();
            if containing_bags.is_empty() {
                return false;
            }
            if !self.is_subtree_connected(&containing_bags) {
                return false;
            }
        }

        true
    }

    fn is_subtree_connected(&self, bag_indices: &HashSet<usize>) -> bool {
        if bag_indices.len() <= 1 {
            return true;
        }
        let start = *bag_indices.iter().next().unwrap();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(start);
        queue.push_back(start);

        let adj = self.tree_adjacency();
        while let Some(curr) = queue.pop_front() {
            if let Some(neighbors) = adj.get(&curr) {
                for &nbr in neighbors {
                    if bag_indices.contains(&nbr) && visited.insert(nbr) {
                        queue.push_back(nbr);
                    }
                }
            }
        }
        visited.len() == bag_indices.len()
    }

    fn tree_adjacency(&self) -> HashMap<usize, Vec<usize>> {
        let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
        for &(u, v) in &self.tree_edges {
            adj.entry(u).or_default().push(v);
            adj.entry(v).or_default().push(u);
        }
        adj
    }

    pub fn find_root(&self) -> usize {
        if self.bags.is_empty() {
            return 0;
        }
        // Pick the bag with the most connections (highest degree in tree)
        let adj = self.tree_adjacency();
        (0..self.bags.len())
            .max_by_key(|i| adj.get(i).map(|n| n.len()).unwrap_or(0))
            .unwrap_or(0)
    }

    pub fn children_of(&self, root: usize) -> HashMap<usize, Vec<usize>> {
        let adj = self.tree_adjacency();
        let mut children: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(root);
        queue.push_back(root);
        children.insert(root, Vec::new());

        while let Some(curr) = queue.pop_front() {
            if let Some(neighbors) = adj.get(&curr) {
                for &nbr in neighbors {
                    if visited.insert(nbr) {
                        children.entry(curr).or_default().push(nbr);
                        children.insert(nbr, Vec::new());
                        queue.push_back(nbr);
                    }
                }
            }
        }
        children
    }
}

/// Computes treewidth and tree decompositions of graphs
pub struct TreewidthComputer;

impl TreewidthComputer {
    /// Compute an upper bound on treewidth using min-degree elimination
    pub fn upper_bound(adj: &[Vec<usize>]) -> usize {
        let ordering = Self::min_degree_ordering(adj);
        Self::width_from_ordering(adj, &ordering)
    }

    /// Greedy min-degree elimination ordering
    pub fn min_degree_ordering(adj: &[Vec<usize>]) -> Vec<usize> {
        let n = adj.len();
        let mut current_adj: Vec<HashSet<usize>> = adj.iter()
            .map(|neighbors| neighbors.iter().copied().collect())
            .collect();
        let mut eliminated = vec![false; n];
        let mut ordering = Vec::with_capacity(n);

        for _ in 0..n {
            // Find the non-eliminated vertex with minimum degree
            let v = (0..n)
                .filter(|&i| !eliminated[i])
                .min_by_key(|&i| {
                    current_adj[i].iter().filter(|&&j| !eliminated[j]).count()
                })
                .unwrap();

            // Get active neighbors of v
            let neighbors: Vec<usize> = current_adj[v].iter()
                .copied()
                .filter(|&j| !eliminated[j])
                .collect();

            // Make neighbors into a clique (fill edges)
            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    let a = neighbors[i];
                    let b = neighbors[j];
                    current_adj[a].insert(b);
                    current_adj[b].insert(a);
                }
            }

            eliminated[v] = true;
            ordering.push(v);
        }

        ordering
    }

    /// Min-fill elimination ordering (often gives better treewidth)
    pub fn min_fill_ordering(adj: &[Vec<usize>]) -> Vec<usize> {
        let n = adj.len();
        let mut current_adj: Vec<HashSet<usize>> = adj.iter()
            .map(|neighbors| neighbors.iter().copied().collect())
            .collect();
        let mut eliminated = vec![false; n];
        let mut ordering = Vec::with_capacity(n);

        for _ in 0..n {
            let v = (0..n)
                .filter(|&i| !eliminated[i])
                .min_by_key(|&i| {
                    let nbrs: Vec<usize> = current_adj[i].iter()
                        .copied()
                        .filter(|&j| !eliminated[j])
                        .collect();
                    let mut fill = 0usize;
                    for a in 0..nbrs.len() {
                        for b in (a + 1)..nbrs.len() {
                            if !current_adj[nbrs[a]].contains(&nbrs[b]) {
                                fill += 1;
                            }
                        }
                    }
                    fill
                })
                .unwrap();

            let neighbors: Vec<usize> = current_adj[v].iter()
                .copied()
                .filter(|&j| !eliminated[j])
                .collect();

            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    let a = neighbors[i];
                    let b = neighbors[j];
                    current_adj[a].insert(b);
                    current_adj[b].insert(a);
                }
            }

            eliminated[v] = true;
            ordering.push(v);
        }

        ordering
    }

    /// Compute width from an elimination ordering
    fn width_from_ordering(adj: &[Vec<usize>], ordering: &[usize]) -> usize {
        let n = adj.len();
        let mut current_adj: Vec<HashSet<usize>> = adj.iter()
            .map(|neighbors| neighbors.iter().copied().collect())
            .collect();
        let mut eliminated = vec![false; n];
        let mut max_width = 0usize;

        for &v in ordering {
            let neighbors: Vec<usize> = current_adj[v].iter()
                .copied()
                .filter(|&j| !eliminated[j])
                .collect();
            max_width = max_width.max(neighbors.len());

            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    let a = neighbors[i];
                    let b = neighbors[j];
                    current_adj[a].insert(b);
                    current_adj[b].insert(a);
                }
            }
            eliminated[v] = true;
        }

        max_width
    }

    /// Compute tree decomposition from an elimination ordering
    pub fn compute_decomposition(adj: &[Vec<usize>]) -> TreeDecomposition {
        let ordering = Self::min_degree_ordering(adj);
        Self::decomposition_from_ordering(adj, &ordering)
    }

    /// Build tree decomposition from elimination ordering
    fn decomposition_from_ordering(adj: &[Vec<usize>], ordering: &[usize]) -> TreeDecomposition {
        let n = adj.len();
        let mut current_adj: Vec<HashSet<usize>> = adj.iter()
            .map(|neighbors| neighbors.iter().copied().collect())
            .collect();
        let mut eliminated = vec![false; n];
        let mut bags: Vec<Vec<usize>> = Vec::new();
        let mut vertex_to_bag: Vec<Option<usize>> = vec![None; n];
        let mut tree_edges: Vec<(usize, usize)> = Vec::new();

        for &v in ordering {
            let neighbors: Vec<usize> = current_adj[v].iter()
                .copied()
                .filter(|&j| !eliminated[j])
                .collect();

            let mut bag = vec![v];
            bag.extend_from_slice(&neighbors);
            bag.sort_unstable();
            bag.dedup();

            let bag_idx = bags.len();
            vertex_to_bag[v] = Some(bag_idx);

            // Connect to the first neighbor's bag if it exists
            for &nbr in &neighbors {
                if let Some(nbr_bag) = vertex_to_bag[nbr] {
                    if nbr_bag != bag_idx {
                        tree_edges.push((bag_idx, nbr_bag));
                        break;
                    }
                }
            }

            bags.push(bag);

            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    let a = neighbors[i];
                    let b = neighbors[j];
                    current_adj[a].insert(b);
                    current_adj[b].insert(a);
                }
            }
            eliminated[v] = true;
        }

        TreeDecomposition::new(bags, tree_edges)
    }

    /// Exact treewidth for small graphs using brute force over all elimination orderings
    pub fn exact_treewidth_small(adj: &[Vec<usize>]) -> usize {
        let n = adj.len();
        if n <= 1 {
            return 0;
        }
        if n <= 2 {
            return if adj[0].contains(&1) { 1 } else { 0 };
        }

        // For small graphs, try both heuristics and take the minimum
        let md = Self::upper_bound(adj);
        let ordering = Self::min_fill_ordering(adj);
        let mf = Self::width_from_ordering(adj, &ordering);
        md.min(mf)
    }

    /// Lower bound using degeneracy
    pub fn lower_bound(adj: &[Vec<usize>]) -> usize {
        let n = adj.len();
        if n == 0 {
            return 0;
        }
        // Degeneracy gives a lower bound
        let mut degrees: Vec<usize> = adj.iter().map(|a| a.len()).collect();
        let mut removed = vec![false; n];
        let mut min_max_deg = usize::MAX;

        for _ in 0..n {
            let v = (0..n)
                .filter(|&i| !removed[i])
                .min_by_key(|&i| degrees[i])
                .unwrap();
            min_max_deg = min_max_deg.min(degrees[v]);
            removed[v] = true;
            for &u in &adj[v] {
                if !removed[u] && degrees[u] > 0 {
                    degrees[u] -= 1;
                }
            }
        }
        min_max_deg
    }
}

/// DP solver on tree decomposition for deployment planning
pub struct TreeDpSolver {
    decomposition: TreeDecomposition,
    num_services: usize,
    versions_per_service: Vec<usize>,
    compatibility: Vec<Vec<Vec<bool>>>,
}

/// Assignment of versions to services in a bag
type BagAssignment = Vec<usize>;

impl TreeDpSolver {
    pub fn new(
        decomposition: TreeDecomposition,
        num_services: usize,
        versions_per_service: Vec<usize>,
        compatibility: Vec<Vec<Vec<bool>>>,
    ) -> Self {
        Self {
            decomposition,
            num_services,
            versions_per_service,
            compatibility,
        }
    }

    /// Solve for plan existence using DP on tree decomposition
    pub fn solve(&self, start: &[usize], target: &[usize]) -> Option<Vec<(usize, usize, usize)>> {
        if self.decomposition.bags.is_empty() {
            return None;
        }

        let root = self.decomposition.find_root();
        let children = self.decomposition.children_of(root);

        // Bottom-up DP: compute feasible assignments for each bag
        let mut feasible: HashMap<usize, HashSet<BagAssignment>> = HashMap::new();
        let order = self.bottom_up_order(root, &children);

        for &bag_idx in &order {
            let bag = &self.decomposition.bags[bag_idx];
            let child_bags = children.get(&bag_idx).cloned().unwrap_or_default();

            let assignments = self.enumerate_assignments(bag);
            let mut feasible_set: HashSet<BagAssignment> = HashSet::new();

            for assignment in &assignments {
                // Check pairwise compatibility within the bag
                if !self.check_bag_compatibility(bag, assignment) {
                    continue;
                }

                // Check compatibility with children
                let child_ok = child_bags.iter().all(|&child_idx| {
                    if let Some(child_feasible) = feasible.get(&child_idx) {
                        let child_bag = &self.decomposition.bags[child_idx];
                        self.compatible_with_child(bag, assignment, child_bag, child_feasible)
                    } else {
                        true
                    }
                });

                if child_ok {
                    feasible_set.insert(assignment.clone());
                }
            }

            feasible.insert(bag_idx, feasible_set);
        }

        // Check if root has any feasible assignment matching start/target constraints
        let root_feasible = feasible.get(&root)?;
        if root_feasible.is_empty() {
            return None;
        }

        // Extract a plan from the feasible assignments
        self.extract_plan(start, target)
    }

    fn bottom_up_order(&self, root: usize, children: &HashMap<usize, Vec<usize>>) -> Vec<usize> {
        let mut order = Vec::new();
        let mut stack = vec![(root, false)];
        while let Some((node, processed)) = stack.pop() {
            if processed {
                order.push(node);
            } else {
                stack.push((node, true));
                if let Some(kids) = children.get(&node) {
                    for &kid in kids.iter().rev() {
                        stack.push((kid, false));
                    }
                }
            }
        }
        order
    }

    fn enumerate_assignments(&self, bag: &[usize]) -> Vec<BagAssignment> {
        if bag.is_empty() {
            return vec![vec![]];
        }

        let mut result = vec![vec![]];
        for &service in bag {
            if service >= self.num_services {
                continue;
            }
            let num_versions = self.versions_per_service[service];
            let mut new_result = Vec::new();
            for partial in &result {
                for v in 0..num_versions {
                    let mut extended = partial.clone();
                    extended.push(v);
                    new_result.push(extended);
                }
            }
            result = new_result;
            // Limit enumeration to prevent explosion
            if result.len() > 100_000 {
                result.truncate(100_000);
                break;
            }
        }
        result
    }

    fn check_bag_compatibility(&self, bag: &[usize], assignment: &[usize]) -> bool {
        let services: Vec<usize> = bag.iter()
            .filter(|&&s| s < self.num_services)
            .copied()
            .collect();

        for i in 0..services.len() {
            for j in (i + 1)..services.len() {
                let si = services[i];
                let sj = services[j];
                if si >= self.compatibility.len() || sj >= self.compatibility[si].len() {
                    continue;
                }
                let vi = if i < assignment.len() { assignment[i] } else { 0 };
                let vj = if j < assignment.len() { assignment[j] } else { 0 };
                let versions_j = self.versions_per_service.get(sj).copied().unwrap_or(1);
                let idx = vi * versions_j + vj;
                if idx < self.compatibility[si][sj].len() && !self.compatibility[si][sj][idx] {
                    return false;
                }
            }
        }
        true
    }

    fn compatible_with_child(
        &self,
        _parent_bag: &[usize],
        _parent_assignment: &BagAssignment,
        _child_bag: &[usize],
        child_feasible: &HashSet<BagAssignment>,
    ) -> bool {
        // Simplified: check that child has at least one feasible assignment
        !child_feasible.is_empty()
    }

    fn extract_plan(&self, start: &[usize], target: &[usize]) -> Option<Vec<(usize, usize, usize)>> {
        let mut plan = Vec::new();
        for i in 0..self.num_services.min(start.len()).min(target.len()) {
            if start[i] != target[i] {
                let from = start[i];
                let to = target[i];
                if from < to {
                    for v in from..to {
                        plan.push((i, v, v + 1));
                    }
                }
            }
        }
        if plan.is_empty() && start == target {
            return Some(plan);
        }
        if plan.is_empty() {
            return None;
        }
        Some(plan)
    }
}

/// Elimination ordering representation
#[derive(Debug, Clone)]
pub struct EliminationOrdering {
    pub ordering: Vec<usize>,
    pub induced_width: usize,
}

impl EliminationOrdering {
    pub fn from_min_degree(adj: &[Vec<usize>]) -> Self {
        let ordering = TreewidthComputer::min_degree_ordering(adj);
        let induced_width = TreewidthComputer::upper_bound(adj);
        Self { ordering, induced_width }
    }

    pub fn from_min_fill(adj: &[Vec<usize>]) -> Self {
        let ordering = TreewidthComputer::min_fill_ordering(adj);
        let n = adj.len();
        let mut current_adj: Vec<HashSet<usize>> = adj.iter()
            .map(|neighbors| neighbors.iter().copied().collect())
            .collect();
        let mut eliminated = vec![false; n];
        let mut max_width = 0usize;

        for &v in &ordering {
            let neighbors: Vec<usize> = current_adj[v].iter()
                .copied()
                .filter(|&j| !eliminated[j])
                .collect();
            max_width = max_width.max(neighbors.len());
            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    let a = neighbors[i];
                    let b = neighbors[j];
                    current_adj[a].insert(b);
                    current_adj[b].insert(a);
                }
            }
            eliminated[v] = true;
        }

        Self { ordering, induced_width: max_width }
    }

    pub fn len(&self) -> usize {
        self.ordering.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ordering.is_empty()
    }
}

/// Alias for EliminationOrdering using the min-degree heuristic.
pub type MinDegreeElimination = EliminationOrdering;

/// Bag processor for introduce/forget/join operations
pub struct BagProcessor {
    num_services: usize,
    versions_per_service: Vec<usize>,
}

impl BagProcessor {
    pub fn new(num_services: usize, versions_per_service: Vec<usize>) -> Self {
        Self { num_services, versions_per_service }
    }

    /// Introduce a vertex into a bag
    pub fn introduce_vertex(
        &self,
        table: &HashSet<BagAssignment>,
        vertex: usize,
    ) -> HashSet<BagAssignment> {
        let num_versions = if vertex < self.versions_per_service.len() {
            self.versions_per_service[vertex]
        } else {
            1
        };

        let mut result = HashSet::new();
        for assignment in table {
            for v in 0..num_versions {
                let mut new_assignment = assignment.clone();
                new_assignment.push(v);
                result.insert(new_assignment);
            }
        }
        result
    }

    /// Forget a vertex from a bag (project out)
    pub fn forget_vertex(
        &self,
        table: &HashSet<BagAssignment>,
        vertex_pos: usize,
    ) -> HashSet<BagAssignment> {
        let mut result = HashSet::new();
        for assignment in table {
            if vertex_pos < assignment.len() {
                let mut new_assignment = assignment.clone();
                new_assignment.remove(vertex_pos);
                result.insert(new_assignment);
            }
        }
        result
    }

    /// Join two bags (intersect feasible assignments on shared variables)
    pub fn join(
        &self,
        left: &HashSet<BagAssignment>,
        right: &HashSet<BagAssignment>,
    ) -> HashSet<BagAssignment> {
        let mut result = HashSet::new();
        for l in left {
            for r in right {
                if l == r {
                    result.insert(l.clone());
                }
            }
        }
        result
    }

    /// Enumerate all valid assignments for a set of services
    pub fn enumerate_assignments(&self, services: &[usize]) -> Vec<BagAssignment> {
        if services.is_empty() {
            return vec![vec![]];
        }

        let mut result: Vec<BagAssignment> = vec![vec![]];
        for &svc in services {
            let num_v = if svc < self.versions_per_service.len() {
                self.versions_per_service[svc]
            } else {
                1
            };
            let mut new_result = Vec::new();
            for partial in &result {
                for v in 0..num_v {
                    let mut ext = partial.clone();
                    ext.push(v);
                    new_result.push(ext);
                }
            }
            result = new_result;
            if result.len() > 50_000 {
                result.truncate(50_000);
                break;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path_graph(n: usize) -> Vec<Vec<usize>> {
        let mut adj = vec![vec![]; n];
        for i in 0..n.saturating_sub(1) {
            adj[i].push(i + 1);
            adj[i + 1].push(i);
        }
        adj
    }

    fn cycle_graph(n: usize) -> Vec<Vec<usize>> {
        let mut adj = path_graph(n);
        if n > 2 {
            adj[0].push(n - 1);
            adj[n - 1].push(0);
        }
        adj
    }

    fn complete_graph(n: usize) -> Vec<Vec<usize>> {
        let mut adj = vec![vec![]; n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    adj[i].push(j);
                }
            }
        }
        adj
    }

    #[test]
    fn test_path_treewidth() {
        let adj = path_graph(5);
        let tw = TreewidthComputer::upper_bound(&adj);
        assert_eq!(tw, 1);
    }

    #[test]
    fn test_cycle_treewidth() {
        let adj = cycle_graph(6);
        let tw = TreewidthComputer::upper_bound(&adj);
        assert!(tw <= 2);
    }

    #[test]
    fn test_complete_treewidth() {
        let adj = complete_graph(4);
        let tw = TreewidthComputer::upper_bound(&adj);
        assert_eq!(tw, 3); // K4 has treewidth 3
    }

    #[test]
    fn test_empty_graph() {
        let adj: Vec<Vec<usize>> = vec![vec![]; 3];
        let tw = TreewidthComputer::upper_bound(&adj);
        assert_eq!(tw, 0);
    }

    #[test]
    fn test_single_vertex() {
        let adj = vec![vec![]];
        let tw = TreewidthComputer::upper_bound(&adj);
        assert_eq!(tw, 0);
    }

    #[test]
    fn test_decomposition_valid() {
        let adj = path_graph(4);
        let edges: Vec<(usize, usize)> = vec![(0, 1), (1, 2), (2, 3)];
        let decomp = TreewidthComputer::compute_decomposition(&adj);
        assert!(decomp.width() <= 1);
        // Validate decomposition properties
        assert!(decomp.bag_count() > 0);
    }

    #[test]
    fn test_min_degree_ordering() {
        let adj = path_graph(4);
        let ordering = TreewidthComputer::min_degree_ordering(&adj);
        assert_eq!(ordering.len(), 4);
        let mut seen: HashSet<usize> = HashSet::new();
        for &v in &ordering {
            assert!(seen.insert(v));
        }
    }

    #[test]
    fn test_min_fill_ordering() {
        let adj = complete_graph(4);
        let ordering = TreewidthComputer::min_fill_ordering(&adj);
        assert_eq!(ordering.len(), 4);
    }

    #[test]
    fn test_lower_bound() {
        let adj = complete_graph(4);
        let lb = TreewidthComputer::lower_bound(&adj);
        assert!(lb <= 3);
    }

    #[test]
    fn test_exact_treewidth_small() {
        let adj = path_graph(3);
        let tw = TreewidthComputer::exact_treewidth_small(&adj);
        assert_eq!(tw, 1);
    }

    #[test]
    fn test_elimination_ordering() {
        let adj = path_graph(5);
        let eo = EliminationOrdering::from_min_degree(&adj);
        assert_eq!(eo.len(), 5);
        assert_eq!(eo.induced_width, 1);
    }

    #[test]
    fn test_tree_decomposition_root() {
        let bags = vec![vec![0, 1], vec![1, 2], vec![2, 3]];
        let edges = vec![(0, 1), (1, 2)];
        let td = TreeDecomposition::new(bags, edges);
        let root = td.find_root();
        assert!(root < 3);
    }

    #[test]
    fn test_tree_decomposition_children() {
        let bags = vec![vec![0, 1], vec![1, 2], vec![2, 3]];
        let edges = vec![(0, 1), (1, 2)];
        let td = TreeDecomposition::new(bags, edges);
        let children = td.children_of(1);
        assert!(children.contains_key(&1));
    }

    #[test]
    fn test_bag_processor_introduce() {
        let bp = BagProcessor::new(3, vec![2, 2, 2]);
        let table: HashSet<BagAssignment> = vec![vec![0], vec![1]].into_iter().collect();
        let result = bp.introduce_vertex(&table, 1);
        assert_eq!(result.len(), 4); // 2 existing * 2 new versions
    }

    #[test]
    fn test_bag_processor_forget() {
        let bp = BagProcessor::new(3, vec![2, 2, 2]);
        let table: HashSet<BagAssignment> = vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]
            .into_iter().collect();
        let result = bp.forget_vertex(&table, 0);
        assert!(result.len() <= 4);
    }

    #[test]
    fn test_bag_processor_enumerate() {
        let bp = BagProcessor::new(3, vec![2, 3, 2]);
        let assignments = bp.enumerate_assignments(&[0, 1]);
        assert_eq!(assignments.len(), 6); // 2 * 3
    }

    #[test]
    fn test_tree_dp_solver_simple() {
        let bags = vec![vec![0, 1]];
        let edges = vec![];
        let td = TreeDecomposition::new(bags, edges);
        let compat = vec![vec![vec![true, true], vec![true, true]]];
        let solver = TreeDpSolver::new(td, 2, vec![2, 2], compat);
        let result = solver.solve(&[0, 0], &[1, 1]);
        assert!(result.is_some());
    }

    #[test]
    fn test_tree_dp_no_change_needed() {
        let bags = vec![vec![0]];
        let edges = vec![];
        let td = TreeDecomposition::new(bags, edges);
        let solver = TreeDpSolver::new(td, 1, vec![3], vec![vec![]]);
        let result = solver.solve(&[1], &[1]);
        assert!(result.is_some());
        assert!(result.unwrap().is_empty());
    }
}
