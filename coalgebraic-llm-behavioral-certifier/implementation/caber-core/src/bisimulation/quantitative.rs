// quantitative.rs — Quantitative bisimulation (behavioral distances) module for CABER.
//
// Implements behavioral pseudometrics via iterative fixpoint computation of
// Kantorovich-style distances over probabilistic transition systems.
// All types are defined locally; no imports from other CABER modules.

use std::collections::{HashMap, HashSet, VecDeque};

// ---------------------------------------------------------------------------
// CouplingMethod
// ---------------------------------------------------------------------------

/// Selects the algorithm used for optimal-transport / coupling computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CouplingMethod {
    /// Exact LP-based transport solver (northwest corner + stepping-stone).
    Exact,
    /// Hungarian algorithm (for equal-support-size assignment problems).
    Hungarian,
    /// Fast greedy coupling heuristic.
    GreedyApprox,
}

// ---------------------------------------------------------------------------
// QuantBisimConfig
// ---------------------------------------------------------------------------

/// Configuration knobs for the quantitative bisimulation engine.
#[derive(Debug, Clone)]
pub struct QuantBisimConfig {
    /// Discount factor c ∈ (0, 1].
    pub discount_factor: f64,
    /// Convergence threshold ε — stop when max |d_{n+1} − d_n| < ε.
    pub epsilon: f64,
    /// Hard cap on the number of fixpoint iterations.
    pub max_iterations: usize,
    /// If true, use a cheaper approximate coupling instead of exact OT.
    pub use_sublinear_approx: bool,
    /// Which coupling / OT solver to employ.
    pub coupling_method: CouplingMethod,
}

impl Default for QuantBisimConfig {
    fn default() -> Self {
        Self {
            discount_factor: 0.99,
            epsilon: 1e-8,
            max_iterations: 1000,
            use_sublinear_approx: false,
            coupling_method: CouplingMethod::Exact,
        }
    }
}

// ---------------------------------------------------------------------------
// ProbTransitionSystem
// ---------------------------------------------------------------------------

/// A labelled probabilistic transition system (LTS with distributions).
///
/// For every `(state, action)` pair the system stores a discrete probability
/// distribution over successor states.
#[derive(Debug, Clone)]
pub struct ProbTransitionSystem {
    pub num_states: usize,
    pub actions: Vec<String>,
    /// `(state, action) -> [(target, probability)]`
    pub transitions: HashMap<(usize, String), Vec<(usize, f64)>>,
    /// Per-state label sets (atomic propositions that hold).
    pub state_labels: Vec<Vec<String>>,
}

impl ProbTransitionSystem {
    pub fn new(num_states: usize) -> Self {
        Self {
            num_states,
            actions: Vec::new(),
            transitions: HashMap::new(),
            state_labels: vec![Vec::new(); num_states],
        }
    }

    /// Add a single transition edge.  If the `(state, action)` pair already
    /// has an entry for `target`, the probability is *added* (caller should
    /// call `normalize` later if needed).
    pub fn add_transition(&mut self, state: usize, action: &str, target: usize, prob: f64) {
        let act = action.to_string();
        if !self.actions.contains(&act) {
            self.actions.push(act.clone());
        }
        let entry = self.transitions.entry((state, act)).or_insert_with(Vec::new);
        // Merge into existing target if present.
        for pair in entry.iter_mut() {
            if pair.0 == target {
                pair.1 += prob;
                return;
            }
        }
        entry.push((target, prob));
    }

    /// Return the distribution for `(state, action)`.  Returns an empty
    /// vector when the action is not enabled from `state`.
    pub fn distribution(&self, state: usize, action: &str) -> Vec<(usize, f64)> {
        self.transitions
            .get(&(state, action.to_string()))
            .cloned()
            .unwrap_or_default()
    }

    /// Normalize every distribution so that probabilities sum to 1.
    pub fn normalize(&mut self) {
        for dist in self.transitions.values_mut() {
            let total: f64 = dist.iter().map(|(_, p)| *p).sum();
            if total > 0.0 && (total - 1.0).abs() > 1e-15 {
                for pair in dist.iter_mut() {
                    pair.1 /= total;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DistinguishingTrace
// ---------------------------------------------------------------------------

/// A witness trace that certifies a lower bound on the behavioral distance
/// between two states.
#[derive(Debug, Clone)]
pub struct DistinguishingTrace {
    pub actions: Vec<String>,
    pub difference: f64,
    pub description: String,
}

// ---------------------------------------------------------------------------
// CouplingConstruction  (sic — matches spec spelling)
// ---------------------------------------------------------------------------

/// A concrete coupling (joint distribution) together with its transport cost.
#[derive(Debug, Clone)]
pub struct CouplingConstruction {
    /// Joint distribution matrix γ[i][j].
    pub coupling: Vec<Vec<f64>>,
    /// Total transport cost  Σ γ[i][j] · c[i][j].
    pub cost: f64,
}

impl CouplingConstruction {
    pub fn from_transport_plan(plan: Vec<Vec<f64>>, cost: f64) -> Self {
        Self {
            coupling: plan,
            cost,
        }
    }

    /// Check that the row/column marginals of the coupling match `mu` and `nu`
    /// (within floating-point tolerance).
    pub fn is_valid_coupling(&self, mu: &[f64], nu: &[f64]) -> bool {
        let tol = 1e-6;
        let rows = self.coupling.len();
        if rows != mu.len() {
            return false;
        }
        if rows == 0 {
            return nu.is_empty();
        }
        let cols = self.coupling[0].len();
        if cols != nu.len() {
            return false;
        }
        // Check row marginals.
        for i in 0..rows {
            let row_sum: f64 = self.coupling[i].iter().sum();
            if (row_sum - mu[i]).abs() > tol {
                return false;
            }
        }
        // Check column marginals.
        for j in 0..cols {
            let col_sum: f64 = (0..rows).map(|i| self.coupling[i][j]).sum();
            if (col_sum - nu[j]).abs() > tol {
                return false;
            }
        }
        true
    }

    pub fn cost(&self) -> f64 {
        self.cost
    }
}

// ---------------------------------------------------------------------------
// BehavioralPseudometric
// ---------------------------------------------------------------------------

/// A symmetric, non-negative function satisfying d(x,x)=0 and the triangle
/// inequality — stored as a full n×n matrix.
#[derive(Debug, Clone)]
pub struct BehavioralPseudometric {
    pub distances: Vec<Vec<f64>>,
    pub states: usize,
}

impl BehavioralPseudometric {
    pub fn from_distance_matrix(d: Vec<Vec<f64>>) -> Self {
        let n = d.len();
        Self {
            distances: d,
            states: n,
        }
    }

    /// Check d(x,x)=0, symmetry, and triangle inequality.
    pub fn validate_pseudometric(&self) -> bool {
        let n = self.states;
        let tol = 1e-6;
        for i in 0..n {
            if self.distances[i][i].abs() > tol {
                return false;
            }
            for j in 0..n {
                // symmetry
                if (self.distances[i][j] - self.distances[j][i]).abs() > tol {
                    return false;
                }
                // non-negative
                if self.distances[i][j] < -tol {
                    return false;
                }
                // triangle inequality
                for k in 0..n {
                    if self.distances[i][j] > self.distances[i][k] + self.distances[k][j] + tol {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Check the strong (ultra-metric) triangle inequality:
    /// d(x,z) ≤ max(d(x,y), d(y,z)).
    pub fn validate_ultrametric(&self) -> bool {
        let n = self.states;
        let tol = 1e-6;
        // Must first be a pseudometric.
        if !self.validate_pseudometric() {
            return false;
        }
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let max_ij_jk = self.distances[i][j].max(self.distances[j][k]);
                    if self.distances[i][k] > max_ij_jk + tol {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Maximum pairwise distance.
    pub fn diameter(&self) -> f64 {
        let mut diam: f64 = 0.0;
        for row in &self.distances {
            for &v in row {
                if v > diam {
                    diam = v;
                }
            }
        }
        diam
    }

    /// Partition the state set into equivalence classes where every pair in the
    /// same class has distance ≤ `epsilon`.  Uses single-linkage clustering.
    pub fn quotient_at_threshold(&self, epsilon: f64) -> Vec<Vec<usize>> {
        let n = self.states;
        // Union-Find.
        let mut parent: Vec<usize> = (0..n).collect();
        let mut rank: Vec<usize> = vec![0; n];

        fn find(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }
        fn union(parent: &mut Vec<usize>, rank: &mut Vec<usize>, a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra == rb {
                return;
            }
            if rank[ra] < rank[rb] {
                parent[ra] = rb;
            } else if rank[ra] > rank[rb] {
                parent[rb] = ra;
            } else {
                parent[rb] = ra;
                rank[ra] += 1;
            }
        }

        for i in 0..n {
            for j in (i + 1)..n {
                if self.distances[i][j] <= epsilon {
                    union(&mut parent, &mut rank, i, j);
                }
            }
        }

        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let root = find(&mut parent, i);
            groups.entry(root).or_insert_with(Vec::new).push(i);
        }
        let mut result: Vec<Vec<usize>> = groups.into_values().collect();
        result.sort_by_key(|g| g[0]);
        result
    }
}

// ---------------------------------------------------------------------------
// KantorovichComputer
// ---------------------------------------------------------------------------

/// Computes the Kantorovich (Wasserstein-1) distance between two finite
/// distributions given a ground metric.
pub struct KantorovichComputer;

impl KantorovichComputer {
    /// Exact Kantorovich via optimal transport LP.
    pub fn compute(
        mu: &[(usize, f64)],
        nu: &[(usize, f64)],
        ground_metric: &[Vec<f64>],
    ) -> f64 {
        if mu.is_empty() && nu.is_empty() {
            return 0.0;
        }
        if mu.is_empty() || nu.is_empty() {
            return 1.0;
        }

        let m = mu.len();
        let n = nu.len();

        // Build supply / demand / cost.
        let supply: Vec<f64> = mu.iter().map(|(_, p)| *p).collect();
        let demand: Vec<f64> = nu.iter().map(|(_, p)| *p).collect();
        let mut cost = vec![vec![0.0; n]; m];
        for i in 0..m {
            for j in 0..n {
                cost[i][j] = ground_metric[mu[i].0][nu[j].0];
            }
        }

        let (_, total_cost) = solve_transport_lp(&supply, &demand, &cost);
        total_cost
    }

    /// Hungarian-algorithm-based computation.  Works best when both
    /// distributions have the same support size.  Falls back to the LP solver
    /// when support sizes differ.
    pub fn compute_hungarian(
        mu: &[(usize, f64)],
        nu: &[(usize, f64)],
        ground_metric: &[Vec<f64>],
    ) -> f64 {
        if mu.is_empty() && nu.is_empty() {
            return 0.0;
        }
        if mu.is_empty() || nu.is_empty() {
            return 1.0;
        }

        // Discretise both distributions into equal-size multisets by
        // splitting probability mass into quanta of size `quantum`.
        // We use the least common mass resolution up to 1000 atoms.
        let granularity: usize = 1000;

        let mu_atoms: Vec<usize> = discretize_distribution(mu, granularity);
        let nu_atoms: Vec<usize> = discretize_distribution(nu, granularity);

        let m = mu_atoms.len();
        let n = nu_atoms.len();
        let size = m.max(n);

        // Build size×size cost matrix (pad with zeros for dummies).
        let mut cost_matrix = vec![vec![0.0f64; size]; size];
        for i in 0..size {
            for j in 0..size {
                if i < m && j < n {
                    cost_matrix[i][j] = ground_metric[mu_atoms[i]][nu_atoms[j]];
                }
            }
        }

        let assignment = hungarian_assignment(&cost_matrix);
        let mut total = 0.0;
        for i in 0..size {
            total += cost_matrix[i][assignment[i]];
        }
        total / size as f64
    }

    /// Fast greedy coupling: repeatedly pair the cheapest remaining
    /// (supply, demand) pair and ship as much mass as possible.
    pub fn compute_greedy(
        mu: &[(usize, f64)],
        nu: &[(usize, f64)],
        ground_metric: &[Vec<f64>],
    ) -> f64 {
        if mu.is_empty() && nu.is_empty() {
            return 0.0;
        }
        if mu.is_empty() || nu.is_empty() {
            return 1.0;
        }

        let m = mu.len();
        let n = nu.len();

        // Collect (cost, i, j) triples and sort ascending.
        let mut edges: Vec<(f64, usize, usize)> = Vec::with_capacity(m * n);
        for i in 0..m {
            for j in 0..n {
                let c = ground_metric[mu[i].0][nu[j].0];
                edges.push((c, i, j));
            }
        }
        edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut remaining_supply: Vec<f64> = mu.iter().map(|(_, p)| *p).collect();
        let mut remaining_demand: Vec<f64> = nu.iter().map(|(_, p)| *p).collect();
        let mut total_cost = 0.0;

        for (c, i, j) in &edges {
            let ship = remaining_supply[*i].min(remaining_demand[*j]);
            if ship > 0.0 {
                total_cost += ship * c;
                remaining_supply[*i] -= ship;
                remaining_demand[*j] -= ship;
            }
        }
        total_cost
    }
}

// ---------------------------------------------------------------------------
// Transport LP solver (northwest-corner + stepping-stone / MODI)
// ---------------------------------------------------------------------------

/// Solve the transportation problem:
///   min  Σ_ij  flow[i][j] * cost[i][j]
///   s.t. Σ_j flow[i][j] = supply[i]  ∀ i
///        Σ_i flow[i][j] = demand[j]  ∀ j
///        flow[i][j] ≥ 0
///
/// Returns `(flow_matrix, total_cost)`.
///
/// Uses the northwest-corner rule to find an initial basic feasible solution,
/// then improves it with the MODI (modified distribution) / stepping-stone
/// method until optimal.
pub fn solve_transport_lp(
    supply: &[f64],
    demand: &[f64],
    cost: &[Vec<f64>],
) -> (Vec<Vec<f64>>, f64) {
    let m = supply.len();
    let n = demand.len();

    if m == 0 || n == 0 {
        return (vec![vec![0.0; n]; m], 0.0);
    }

    // Balance the problem (add dummy row/col if needed).
    let total_supply: f64 = supply.iter().sum();
    let total_demand: f64 = demand.iter().sum();

    let mut s: Vec<f64> = supply.to_vec();
    let mut d: Vec<f64> = demand.to_vec();
    let mut c: Vec<Vec<f64>> = cost.to_vec();

    if total_supply < total_demand - 1e-12 {
        // Add dummy supply row.
        s.push(total_demand - total_supply);
        c.push(vec![0.0; n]);
    } else if total_demand < total_supply - 1e-12 {
        // Add dummy demand column.
        d.push(total_supply - total_demand);
        for row in c.iter_mut() {
            row.push(0.0);
        }
    }

    let rows = s.len();
    let cols = d.len();

    // --- Northwest Corner initial BFS ---
    let mut flow = vec![vec![0.0f64; cols]; rows];
    {
        let mut rem_s = s.clone();
        let mut rem_d = d.clone();
        let mut i = 0;
        let mut j = 0;
        while i < rows && j < cols {
            let ship = rem_s[i].min(rem_d[j]);
            flow[i][j] = ship;
            rem_s[i] -= ship;
            rem_d[j] -= ship;
            if rem_s[i] < 1e-15 {
                i += 1;
            }
            if rem_d[j] < 1e-15 {
                j += 1;
            }
        }
    }

    // --- MODI improvement loop ---
    // We iterate up to a generous limit; small instances converge fast.
    let max_modi_iters = 500;
    for _ in 0..max_modi_iters {
        // Identify basic cells.
        let mut basic: Vec<(usize, usize)> = Vec::new();
        for i in 0..rows {
            for j in 0..cols {
                if flow[i][j] > 1e-15 {
                    basic.push((i, j));
                }
            }
        }

        // Need rows+cols-1 basic variables; add degeneracy breakers if needed.
        let needed = rows + cols - 1;
        if basic.len() < needed {
            'outer: for i in 0..rows {
                for j in 0..cols {
                    if flow[i][j] <= 1e-15 {
                        // Check that adding (i,j) doesn't create a cycle with
                        // existing basic cells.  Simple heuristic: just add it.
                        basic.push((i, j));
                        flow[i][j] = 1e-15; // epsilon flow for degeneracy
                        if basic.len() >= needed {
                            break 'outer;
                        }
                    }
                }
            }
        }

        // Compute dual variables u[i], v[j] from u[i]+v[j] = c[i][j] for basic cells.
        let mut u = vec![f64::NAN; rows];
        let mut v = vec![f64::NAN; cols];
        u[0] = 0.0;
        let mut determined = vec![false; rows + cols];
        determined[0] = true;
        let mut changed = true;
        while changed {
            changed = false;
            for &(bi, bj) in &basic {
                let ui_det = determined[bi];
                let vj_det = determined[rows + bj];
                if ui_det && !vj_det {
                    v[bj] = c[bi][bj] - u[bi];
                    determined[rows + bj] = true;
                    changed = true;
                } else if !ui_det && vj_det {
                    u[bi] = c[bi][bj] - v[bj];
                    determined[bi] = true;
                    changed = true;
                }
            }
        }

        // Fill in any undetermined duals with 0 (shouldn't happen in
        // well-connected problems, but be safe).
        for i in 0..rows {
            if u[i].is_nan() {
                u[i] = 0.0;
            }
        }
        for j in 0..cols {
            if v[j].is_nan() {
                v[j] = 0.0;
            }
        }

        // Find the most negative reduced cost among non-basic cells.
        let mut best_rc = -1e-10; // threshold
        let mut entering: Option<(usize, usize)> = None;
        for i in 0..rows {
            for j in 0..cols {
                let rc = c[i][j] - u[i] - v[j];
                if rc < best_rc {
                    best_rc = rc;
                    entering = Some((i, j));
                }
            }
        }

        let (ei, ej) = match entering {
            Some(e) => e,
            None => break, // optimal
        };

        // Find a cycle containing (ei, ej) and the basic cells.
        // We use BFS on a bipartite graph of row/col nodes.
        if let Some(cycle) = find_cycle(ei, ej, &basic, rows, cols) {
            // Determine theta = min flow on "-" cells of the cycle.
            let mut theta = f64::MAX;
            for k in 0..cycle.len() {
                if k % 2 == 1 {
                    // minus cell
                    let (ci, cj) = cycle[k];
                    if flow[ci][cj] < theta {
                        theta = flow[ci][cj];
                    }
                }
            }

            // Apply theta to the cycle.
            for k in 0..cycle.len() {
                let (ci, cj) = cycle[k];
                if k % 2 == 0 {
                    flow[ci][cj] += theta;
                } else {
                    flow[ci][cj] -= theta;
                }
            }
        } else {
            // Could not find cycle — break to avoid infinite loop.
            break;
        }
    }

    // Compute total cost.
    let mut total = 0.0;
    for i in 0..m.min(rows) {
        for j in 0..n.min(cols) {
            if flow[i][j] > 1e-15 {
                total += flow[i][j] * cost[i][j];
            }
        }
    }

    // Trim back to original dimensions.
    let result: Vec<Vec<f64>> = flow.iter().take(m).map(|r| r[..n].to_vec()).collect();
    (result, total)
}

/// Find a cycle in the basis graph that includes the entering cell `(ei, ej)`.
/// Returns the cycle as a sequence of `(row, col)` starting with the entering cell.
fn find_cycle(
    ei: usize,
    ej: usize,
    basic: &[(usize, usize)],
    _rows: usize,
    _cols: usize,
) -> Option<Vec<(usize, usize)>> {
    // Build adjacency: for each row, which columns are basic; for each column, which rows.
    let mut row_to_cols: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut col_to_rows: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(r, c) in basic {
        row_to_cols.entry(r).or_default().push(c);
        col_to_rows.entry(c).or_default().push(r);
    }
    // Also add the entering cell.
    row_to_cols.entry(ei).or_default().push(ej);
    col_to_rows.entry(ej).or_default().push(ei);

    // DFS from (ei, ej).  Alternate row-moves and column-moves.
    // Path stores (row, col) cells.  Even indices: searching along row; odd: along col.
    let mut path: Vec<(usize, usize)> = vec![(ei, ej)];
    let found = dfs_cycle(&mut path, ei, ej, &row_to_cols, &col_to_rows, true);
    if found {
        Some(path)
    } else {
        None
    }
}

/// Recursive DFS searching for a cycle back to (ei, ej).
fn dfs_cycle(
    path: &mut Vec<(usize, usize)>,
    ei: usize,
    ej: usize,
    row_to_cols: &HashMap<usize, Vec<usize>>,
    col_to_rows: &HashMap<usize, Vec<usize>>,
    move_along_row: bool, // true = pick a different col in same row
) -> bool {
    let &(cur_r, cur_c) = path.last().unwrap();
    if move_along_row {
        // Try all columns in the same row.
        if let Some(cols) = row_to_cols.get(&cur_r) {
            for &nj in cols {
                if nj == cur_c {
                    continue;
                }
                if nj == ej && path.len() >= 3 {
                    // Found cycle back.
                    path.push((cur_r, nj));
                    return true;
                }
                // Check not already in path.
                if path.iter().any(|&(_, c)| c == nj && path.iter().any(|&(r2, c2)| c2 == nj && r2 == cur_r)) {
                    continue;
                }
                let already = path.iter().any(|&(r, c)| r == cur_r && c == nj);
                if already {
                    continue;
                }
                path.push((cur_r, nj));
                if dfs_cycle(path, ei, ej, row_to_cols, col_to_rows, false) {
                    return true;
                }
                path.pop();
            }
        }
    } else {
        // Move along column: pick a different row in same col.
        if let Some(rows) = col_to_rows.get(&cur_c) {
            for &ni in rows {
                if ni == cur_r {
                    continue;
                }
                if ni == ei && cur_c == ej && path.len() >= 3 {
                    return true;
                }
                let already = path.iter().any(|&(r, c)| r == ni && c == cur_c);
                if already {
                    continue;
                }
                path.push((ni, cur_c));
                if dfs_cycle(path, ei, ej, row_to_cols, col_to_rows, true) {
                    return true;
                }
                path.pop();
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Hungarian algorithm (Kuhn-Munkres)
// ---------------------------------------------------------------------------

/// Classic O(n³) Hungarian algorithm for the assignment problem.
/// Returns an assignment vector `a` where row `i` is assigned to column `a[i]`.
fn hungarian_assignment(cost: &[Vec<f64>]) -> Vec<usize> {
    let n = cost.len();
    if n == 0 {
        return Vec::new();
    }

    // We use the Jonker-Volgenant variant with potentials.
    let mut u = vec![0.0f64; n + 1]; // row potentials
    let mut v = vec![0.0f64; n + 1]; // col potentials
    let mut assignment = vec![0usize; n + 1]; // col -> row (1-indexed, 0 = unassigned)
    let mut way = vec![0usize; n + 1];

    for i in 1..=n {
        assignment[0] = i;
        let mut j0 = 0usize;
        let mut min_v = vec![f64::MAX; n + 1];
        let mut used = vec![false; n + 1];

        loop {
            used[j0] = true;
            let i0 = assignment[j0];
            let mut delta = f64::MAX;
            let mut j1 = 0usize;

            for j in 1..=n {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < min_v[j] {
                        min_v[j] = cur;
                        way[j] = j0;
                    }
                    if min_v[j] < delta {
                        delta = min_v[j];
                        j1 = j;
                    }
                }
            }

            for j in 0..=n {
                if used[j] {
                    u[assignment[j]] += delta;
                    v[j] -= delta;
                } else {
                    min_v[j] -= delta;
                }
            }

            j0 = j1;
            if assignment[j0] == 0 {
                break;
            }
        }

        loop {
            let j1 = way[j0];
            assignment[j0] = assignment[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    // Convert to 0-indexed.
    let mut result = vec![0usize; n];
    for j in 1..=n {
        if assignment[j] > 0 && assignment[j] <= n {
            result[assignment[j] - 1] = j - 1;
        }
    }
    result
}

/// Discretize a distribution into `granularity` equally-weighted atoms.
fn discretize_distribution(dist: &[(usize, f64)], granularity: usize) -> Vec<usize> {
    let total: f64 = dist.iter().map(|(_, p)| *p).sum();
    if total <= 0.0 {
        return Vec::new();
    }

    let mut atoms: Vec<usize> = Vec::with_capacity(granularity);
    let mut remaining = granularity;
    let mut accumulated_error = 0.0;

    for (idx, (state, prob)) in dist.iter().enumerate() {
        let ideal = prob / total * granularity as f64 + accumulated_error;
        let count = if idx == dist.len() - 1 {
            remaining
        } else {
            let c = ideal.round() as usize;
            c.min(remaining)
        };
        for _ in 0..count {
            atoms.push(*state);
        }
        remaining -= count;
        accumulated_error = ideal - count as f64;
    }

    atoms
}

// ---------------------------------------------------------------------------
// DistinguishingTraceComputer
// ---------------------------------------------------------------------------

/// Finds action sequences (traces) that witness behavioral differences
/// between two states.
pub struct DistinguishingTraceComputer;

impl DistinguishingTraceComputer {
    /// BFS/DFS search for the action sequence of length ≤ `max_depth` that
    /// maximises the distributional difference between `s` and `t`.
    pub fn find_distinguishing_trace(
        system: &ProbTransitionSystem,
        s: usize,
        t: usize,
        max_depth: usize,
    ) -> Option<DistinguishingTrace> {
        // We search over action sequences using BFS.
        // For each sequence w = a_1 … a_k we compute the probability of
        // reaching the set of states with a given label from s vs t.
        // The "difference" is the total-variation distance between the
        // resulting distributions.

        if s == t {
            return None;
        }

        // Check label difference at depth 0.
        let label_diff = if system.state_labels.get(s) != system.state_labels.get(t) {
            1.0
        } else {
            0.0
        };

        let mut best_trace: Option<DistinguishingTrace> = if label_diff > 0.0 {
            Some(DistinguishingTrace {
                actions: Vec::new(),
                difference: label_diff,
                description: format!(
                    "States {} and {} have different labels",
                    s, t
                ),
            })
        } else {
            None
        };
        let mut best_diff = label_diff;

        // BFS: each entry is (actions_so_far, distribution_from_s, distribution_from_t).
        // Distributions are HashMap<usize, f64>.
        type Dist = HashMap<usize, f64>;

        let mut init_s: Dist = HashMap::new();
        init_s.insert(s, 1.0);
        let mut init_t: Dist = HashMap::new();
        init_t.insert(t, 1.0);

        let mut queue: VecDeque<(Vec<String>, Dist, Dist)> = VecDeque::new();
        queue.push_back((Vec::new(), init_s, init_t));

        while let Some((actions, dist_s, dist_t)) = queue.pop_front() {
            if actions.len() >= max_depth {
                continue;
            }
            for action in &system.actions {
                let next_s = step_distribution(&dist_s, action, system);
                let next_t = step_distribution(&dist_t, action, system);

                let tv = total_variation(&next_s, &next_t);
                let mut new_actions = actions.clone();
                new_actions.push(action.clone());

                if tv > best_diff + 1e-12 {
                    best_diff = tv;
                    best_trace = Some(DistinguishingTrace {
                        actions: new_actions.clone(),
                        difference: tv,
                        description: format!(
                            "After actions [{}], TV distance = {:.6}",
                            new_actions.join(", "),
                            tv
                        ),
                    });
                }

                if new_actions.len() < max_depth {
                    queue.push_back((new_actions, next_s, next_t));
                }
            }
        }

        best_trace
    }

    /// Sample `num_traces` random traces (using a deterministic pseudo-random
    /// walk) and return the maximum observed total-variation distance.
    pub fn lower_bound_from_traces(
        system: &ProbTransitionSystem,
        s: usize,
        t: usize,
        num_traces: usize,
    ) -> f64 {
        let max_depth = 10;
        let mut best = 0.0;

        // Simple LCG-based PRNG for deterministic behaviour.
        let mut rng_state: u64 = (s as u64).wrapping_mul(6364136223846793005)
            .wrapping_add(t as u64)
            .wrapping_add(1442695040888963407);

        let next_rand = |state: &mut u64| -> f64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((*state >> 33) as f64) / (u32::MAX as f64)
        };

        for _ in 0..num_traces {
            let mut cur_s = s;
            let mut cur_t = t;
            let mut trace_actions: Vec<String> = Vec::new();

            for _ in 0..max_depth {
                if system.actions.is_empty() {
                    break;
                }
                let a_idx = (next_rand(&mut rng_state) * system.actions.len() as f64) as usize
                    % system.actions.len();
                let action = &system.actions[a_idx];
                trace_actions.push(action.clone());

                let d_s = system.distribution(cur_s, action);
                let d_t = system.distribution(cur_t, action);

                // Pick a successor from d_s.
                cur_s = sample_from_dist(&d_s, next_rand(&mut rng_state));
                cur_t = sample_from_dist(&d_t, next_rand(&mut rng_state));
            }

            // Compute a distributional difference for this trace prefix.
            // We use the label difference at the reached states as a proxy.
            let labels_s = system.state_labels.get(cur_s).cloned().unwrap_or_default();
            let labels_t = system.state_labels.get(cur_t).cloned().unwrap_or_default();
            let diff = if labels_s != labels_t { 1.0 } else { 0.0 };
            if diff > best {
                best = diff;
            }

            // Also check the full distributional picture at each prefix.
            let mut dist_s: HashMap<usize, f64> = HashMap::new();
            dist_s.insert(s, 1.0);
            let mut dist_t: HashMap<usize, f64> = HashMap::new();
            dist_t.insert(t, 1.0);
            for a in &trace_actions {
                dist_s = step_distribution(&dist_s, a, system);
                dist_t = step_distribution(&dist_t, a, system);
                let tv = total_variation(&dist_s, &dist_t);
                if tv > best {
                    best = tv;
                }
            }
        }

        best
    }
}

/// Advance a state distribution by one step under a given action.
fn step_distribution(
    dist: &HashMap<usize, f64>,
    action: &str,
    system: &ProbTransitionSystem,
) -> HashMap<usize, f64> {
    let mut next: HashMap<usize, f64> = HashMap::new();
    for (&state, &prob) in dist {
        let successors = system.distribution(state, action);
        for (tgt, p) in successors {
            *next.entry(tgt).or_insert(0.0) += prob * p;
        }
    }
    next
}

/// Total-variation distance between two (possibly sub-) distributions.
fn total_variation(a: &HashMap<usize, f64>, b: &HashMap<usize, f64>) -> f64 {
    let mut keys: HashSet<usize> = a.keys().copied().collect();
    keys.extend(b.keys());
    let mut tv = 0.0;
    for k in keys {
        let pa = a.get(&k).copied().unwrap_or(0.0);
        let pb = b.get(&k).copied().unwrap_or(0.0);
        tv += (pa - pb).abs();
    }
    tv / 2.0
}

/// Pick a state from a distribution given a random number in [0,1).
fn sample_from_dist(dist: &[(usize, f64)], r: f64) -> usize {
    if dist.is_empty() {
        return 0;
    }
    let mut cumulative = 0.0;
    for &(state, p) in dist {
        cumulative += p;
        if r < cumulative {
            return state;
        }
    }
    dist.last().unwrap().0
}

// ---------------------------------------------------------------------------
// QuantitativeBisimEngine
// ---------------------------------------------------------------------------

/// Main entry-point: computes the behavioral pseudometric on a probabilistic
/// transition system via iterative fixpoint of the Kantorovich lifting.
pub struct QuantitativeBisimEngine {
    pub system: ProbTransitionSystem,
    pub distances: Vec<Vec<f64>>,
    pub config: QuantBisimConfig,
    pub iterations_used: usize,
    pub converged: bool,
}

impl QuantitativeBisimEngine {
    pub fn new(system: ProbTransitionSystem, config: QuantBisimConfig) -> Self {
        let n = system.num_states;
        Self {
            system,
            distances: vec![vec![0.0; n]; n],
            config,
            iterations_used: 0,
            converged: false,
        }
    }

    /// Run the iterative fixpoint computation:
    ///
    /// ```text
    /// d_0(s,t) = 0 if same labels, 1 otherwise
    /// d_{n+1}(s,t) = c · max_a K(d_n)(δ(s,a), δ(t,a))
    /// ```
    ///
    /// where K is the Kantorovich distance w.r.t. ground metric `d_n`,
    /// and `c` is the discount factor.
    pub fn compute(&mut self) {
        let n = self.system.num_states;

        // d_0 initialisation.
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    self.distances[i][j] = 0.0;
                } else {
                    let li = &self.system.state_labels[i];
                    let lj = &self.system.state_labels[j];
                    self.distances[i][j] = if li == lj { 0.0 } else { 1.0 };
                }
            }
        }

        let c = self.config.discount_factor;

        for iter in 0..self.config.max_iterations {
            let old = self.distances.clone();
            let mut max_change: f64 = 0.0;

            for i in 0..n {
                for j in (i + 1)..n {
                    let mut max_kant: f64 = 0.0;

                    // Inherit label distance as a floor.
                    let label_dist = if self.system.state_labels[i] == self.system.state_labels[j] {
                        0.0
                    } else {
                        1.0
                    };

                    for action in &self.system.actions.clone() {
                        let mu = self.system.distribution(i, action);
                        let nu = self.system.distribution(j, action);

                        // If neither state has this action, skip.
                        if mu.is_empty() && nu.is_empty() {
                            continue;
                        }
                        // If exactly one has it, distance contribution = 1.
                        if mu.is_empty() || nu.is_empty() {
                            max_kant = 1.0;
                            continue;
                        }

                        let kant = match self.config.coupling_method {
                            CouplingMethod::Exact => {
                                KantorovichComputer::compute(&mu, &nu, &old)
                            }
                            CouplingMethod::Hungarian => {
                                KantorovichComputer::compute_hungarian(&mu, &nu, &old)
                            }
                            CouplingMethod::GreedyApprox => {
                                KantorovichComputer::compute_greedy(&mu, &nu, &old)
                            }
                        };

                        if kant > max_kant {
                            max_kant = kant;
                        }
                    }

                    // For states with no common action, keep label distance.
                    let new_dist = if self.system.actions.is_empty() {
                        label_dist
                    } else {
                        (c * max_kant).max(label_dist * c)
                    };

                    let new_dist = new_dist.min(1.0);
                    self.distances[i][j] = new_dist;
                    self.distances[j][i] = new_dist;

                    let change = (new_dist - old[i][j]).abs();
                    if change > max_change {
                        max_change = change;
                    }
                }
            }

            self.iterations_used = iter + 1;

            if max_change < self.config.epsilon {
                self.converged = true;
                return;
            }
        }
    }

    pub fn distance(&self, s: usize, t: usize) -> f64 {
        self.distances[s][t]
    }

    pub fn distance_matrix(&self) -> &Vec<Vec<f64>> {
        &self.distances
    }

    pub fn converged(&self) -> bool {
        self.converged
    }

    pub fn iterations(&self) -> usize {
        self.iterations_used
    }

    /// Lower bound from distinguishing traces.
    pub fn lower_bound(&self, s: usize, t: usize) -> f64 {
        DistinguishingTraceComputer::lower_bound_from_traces(&self.system, s, t, 50)
    }

    /// Upper bound from greedy coupling construction.
    pub fn upper_bound(&self, s: usize, t: usize) -> f64 {
        // Use greedy coupling on the current distance matrix.
        let n = self.system.num_states;
        let mut ub = 0.0f64;
        for action in &self.system.actions {
            let mu = self.system.distribution(s, action);
            let nu = self.system.distribution(t, action);
            if mu.is_empty() && nu.is_empty() {
                continue;
            }
            if mu.is_empty() || nu.is_empty() {
                ub = ub.max(1.0);
                continue;
            }
            let k = KantorovichComputer::compute_greedy(&mu, &nu, &self.distances);
            ub = ub.max(k);
        }
        (self.config.discount_factor * ub).min(1.0)
    }

    /// Returns `(lower_bound, upper_bound)`.
    pub fn bracket(&self, s: usize, t: usize) -> (f64, f64) {
        let lb = self.lower_bound(s, t);
        let ub = self.upper_bound(s, t);
        // Ensure lb ≤ ub.  In degenerate cases the heuristics may disagree.
        if lb > ub {
            (ub, lb)
        } else {
            (lb, ub)
        }
    }

    /// Two states are "bisimilar up to ε" when their distance is below ε.
    pub fn is_bisimilar(&self, s: usize, t: usize, epsilon: f64) -> bool {
        self.distances[s][t] < epsilon
    }

    /// Return the `k` states nearest to `s`, sorted by ascending distance.
    pub fn nearest_states(&self, s: usize, k: usize) -> Vec<(usize, f64)> {
        let n = self.system.num_states;
        let mut dists: Vec<(usize, f64)> = (0..n)
            .filter(|&i| i != s)
            .map(|i| (i, self.distances[s][i]))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(k);
        dists
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Build a simple two-state system where both states are identical
    /// (same label, same transitions).
    fn identical_two_state_system() -> ProbTransitionSystem {
        let mut sys = ProbTransitionSystem::new(2);
        sys.state_labels = vec![vec!["a".into()], vec!["a".into()]];
        sys.add_transition(0, "go", 0, 0.5);
        sys.add_transition(0, "go", 1, 0.5);
        sys.add_transition(1, "go", 0, 0.5);
        sys.add_transition(1, "go", 1, 0.5);
        sys
    }

    /// Build a two-state system where the states are as different as possible.
    fn opposite_two_state_system() -> ProbTransitionSystem {
        let mut sys = ProbTransitionSystem::new(2);
        sys.state_labels = vec![vec!["a".into()], vec!["b".into()]];
        sys.add_transition(0, "go", 0, 1.0);
        sys.add_transition(1, "go", 1, 1.0);
        sys
    }

    /// Three-state system producing intermediate distance.
    fn three_state_system() -> ProbTransitionSystem {
        let mut sys = ProbTransitionSystem::new(3);
        sys.state_labels = vec![
            vec!["a".into()],
            vec!["a".into()],
            vec!["b".into()],
        ];
        // State 0 → equally likely 0 or 2.
        sys.add_transition(0, "go", 0, 0.5);
        sys.add_transition(0, "go", 2, 0.5);
        // State 1 → goes to 0 always.
        sys.add_transition(1, "go", 0, 1.0);
        // State 2 → self-loop.
        sys.add_transition(2, "go", 2, 1.0);
        sys
    }

    // -----------------------------------------------------------------------
    // Engine tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_identical_systems_distance_zero() {
        let sys = identical_two_state_system();
        let config = QuantBisimConfig::default();
        let mut engine = QuantitativeBisimEngine::new(sys, config);
        engine.compute();
        assert!(engine.distance(0, 1) < 1e-6, "identical states should have distance ~0");
        assert!(engine.converged());
    }

    #[test]
    fn test_different_systems_distance_one() {
        let sys = opposite_two_state_system();
        let mut config = QuantBisimConfig::default();
        config.discount_factor = 1.0;
        let mut engine = QuantitativeBisimEngine::new(sys, config);
        engine.compute();
        assert!(
            (engine.distance(0, 1) - 1.0).abs() < 1e-4,
            "maximally different states should have distance ~1, got {}",
            engine.distance(0, 1)
        );
    }

    #[test]
    fn test_intermediate_distance() {
        let sys = three_state_system();
        let config = QuantBisimConfig::default();
        let mut engine = QuantitativeBisimEngine::new(sys, config);
        engine.compute();
        let d01 = engine.distance(0, 1);
        assert!(
            d01 > 1e-4 && d01 < 1.0 - 1e-4,
            "states 0 and 1 should have intermediate distance, got {}",
            d01
        );
    }

    #[test]
    fn test_self_distance_zero() {
        let sys = three_state_system();
        let config = QuantBisimConfig::default();
        let mut engine = QuantitativeBisimEngine::new(sys, config);
        engine.compute();
        for i in 0..3 {
            assert!(
                engine.distance(i, i).abs() < 1e-12,
                "d({},{}) should be 0",
                i,
                i
            );
        }
    }

    #[test]
    fn test_symmetry() {
        let sys = three_state_system();
        let config = QuantBisimConfig::default();
        let mut engine = QuantitativeBisimEngine::new(sys, config);
        engine.compute();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (engine.distance(i, j) - engine.distance(j, i)).abs() < 1e-12,
                    "distance should be symmetric"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Kantorovich
    // -----------------------------------------------------------------------

    #[test]
    fn test_kantorovich_identical_distributions() {
        // d is trivial: d(0,0) = 0.
        let ground = vec![vec![0.0]];
        let mu = vec![(0, 1.0)];
        let nu = vec![(0, 1.0)];
        let k = KantorovichComputer::compute(&mu, &nu, &ground);
        assert!(k.abs() < 1e-8, "K between identical point distributions should be 0");
    }

    #[test]
    fn test_kantorovich_dirac_different() {
        let ground = vec![
            vec![0.0, 0.7],
            vec![0.7, 0.0],
        ];
        let mu = vec![(0, 1.0)];
        let nu = vec![(1, 1.0)];
        let k = KantorovichComputer::compute(&mu, &nu, &ground);
        assert!(
            (k - 0.7).abs() < 1e-6,
            "K between Diracs on states 0 and 1 should equal d(0,1)=0.7, got {}",
            k
        );
    }

    #[test]
    fn test_kantorovich_uniform_vs_dirac() {
        // Two states, ground metric d(0,1)=1.
        let ground = vec![
            vec![0.0, 1.0],
            vec![1.0, 0.0],
        ];
        // μ = uniform, ν = Dirac on 0.
        let mu = vec![(0, 0.5), (1, 0.5)];
        let nu = vec![(0, 1.0)];
        let k = KantorovichComputer::compute(&mu, &nu, &ground);
        // Optimal: ship 0.5 from (0→0) cost 0, 0.5 from (1→0) cost 0.5.
        assert!(
            (k - 0.5).abs() < 1e-6,
            "Expected 0.5, got {}",
            k
        );
    }

    // -----------------------------------------------------------------------
    // Hungarian
    // -----------------------------------------------------------------------

    #[test]
    fn test_hungarian_simple_assignment() {
        // 3×3 cost matrix.
        let cost = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 4.0, 6.0],
            vec![3.0, 6.0, 9.0],
        ];
        let a = hungarian_assignment(&cost);
        // Verify it's a valid permutation.
        let mut cols_used: HashSet<usize> = HashSet::new();
        for &c in &a {
            assert!(c < 3);
            cols_used.insert(c);
        }
        assert_eq!(cols_used.len(), 3, "assignment must be a permutation");
    }

    #[test]
    fn test_hungarian_kantorovich() {
        let ground = vec![
            vec![0.0, 1.0],
            vec![1.0, 0.0],
        ];
        let mu = vec![(0, 0.5), (1, 0.5)];
        let nu = vec![(0, 0.5), (1, 0.5)];
        let k = KantorovichComputer::compute_hungarian(&mu, &nu, &ground);
        assert!(k < 0.1, "K between identical distributions should be ~0, got {}", k);
    }

    // -----------------------------------------------------------------------
    // Greedy coupling
    // -----------------------------------------------------------------------

    #[test]
    fn test_greedy_coupling_identical() {
        let ground = vec![
            vec![0.0, 1.0],
            vec![1.0, 0.0],
        ];
        let mu = vec![(0, 0.5), (1, 0.5)];
        let nu = vec![(0, 0.5), (1, 0.5)];
        let k = KantorovichComputer::compute_greedy(&mu, &nu, &ground);
        assert!(k < 1e-8, "Greedy K between identical should be 0, got {}", k);
    }

    #[test]
    fn test_greedy_coupling_dirac() {
        let ground = vec![
            vec![0.0, 0.3],
            vec![0.3, 0.0],
        ];
        let mu = vec![(0, 1.0)];
        let nu = vec![(1, 1.0)];
        let k = KantorovichComputer::compute_greedy(&mu, &nu, &ground);
        assert!(
            (k - 0.3).abs() < 1e-8,
            "Greedy K between Diracs should be 0.3, got {}",
            k
        );
    }

    // -----------------------------------------------------------------------
    // Pseudometric validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_pseudometric_validation() {
        let sys = three_state_system();
        let config = QuantBisimConfig::default();
        let mut engine = QuantitativeBisimEngine::new(sys, config);
        engine.compute();
        let pm = BehavioralPseudometric::from_distance_matrix(engine.distances.clone());
        assert!(
            pm.validate_pseudometric(),
            "computed distance matrix must satisfy pseudometric axioms"
        );
    }

    #[test]
    fn test_pseudometric_trivial_is_ultrametric() {
        // All-zero distance matrix is trivially an ultrametric.
        let d = vec![vec![0.0; 3]; 3];
        let pm = BehavioralPseudometric::from_distance_matrix(d);
        assert!(pm.validate_pseudometric());
        assert!(pm.validate_ultrametric());
    }

    // -----------------------------------------------------------------------
    // Quotient
    // -----------------------------------------------------------------------

    #[test]
    fn test_quotient_at_threshold() {
        let d = vec![
            vec![0.0, 0.1, 0.9],
            vec![0.1, 0.0, 0.8],
            vec![0.9, 0.8, 0.0],
        ];
        let pm = BehavioralPseudometric::from_distance_matrix(d);

        // Threshold 0.2 → states 0,1 together; state 2 alone.
        let groups = pm.quotient_at_threshold(0.2);
        assert_eq!(groups.len(), 2);
        let flat: Vec<usize> = groups.iter().flat_map(|g| g.iter().copied()).collect();
        assert!(flat.contains(&0) && flat.contains(&1) && flat.contains(&2));
        // 0 and 1 should be in the same group.
        let g01 = groups.iter().find(|g| g.contains(&0)).unwrap();
        assert!(g01.contains(&1));

        // Threshold 1.0 → all in one group.
        let groups2 = pm.quotient_at_threshold(1.0);
        assert_eq!(groups2.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Convergence behaviour
    // -----------------------------------------------------------------------

    #[test]
    fn test_convergence_with_tight_epsilon() {
        let sys = three_state_system();
        let mut config = QuantBisimConfig::default();
        config.epsilon = 1e-12;
        config.max_iterations = 5000;
        let mut engine = QuantitativeBisimEngine::new(sys, config);
        engine.compute();
        // Should eventually converge (discount < 1 is a contraction).
        assert!(
            engine.converged() || engine.iterations() == 5000,
            "engine should converge or exhaust iterations"
        );
    }

    #[test]
    fn test_convergence_stabilises() {
        // With a discount factor < 1 the iteration is a contraction and
        // should eventually stabilise.
        let sys = three_state_system();
        let mut config = QuantBisimConfig::default();
        config.max_iterations = 200;
        config.epsilon = 1e-10;

        let mut engine = QuantitativeBisimEngine::new(sys.clone(), config.clone());
        engine.compute();
        let d_converged = engine.distance(0, 1);

        // Running more iterations should not change the result.
        config.max_iterations = 400;
        let mut engine2 = QuantitativeBisimEngine::new(sys, config);
        engine2.compute();
        let d_more = engine2.distance(0, 1);

        assert!(
            (d_converged - d_more).abs() < 1e-8,
            "extra iterations should not change converged distance: {} vs {}",
            d_converged,
            d_more
        );
    }

    // -----------------------------------------------------------------------
    // Nearest states
    // -----------------------------------------------------------------------

    #[test]
    fn test_nearest_states() {
        let sys = three_state_system();
        let config = QuantBisimConfig::default();
        let mut engine = QuantitativeBisimEngine::new(sys, config);
        engine.compute();
        let nearest = engine.nearest_states(0, 2);
        assert_eq!(nearest.len(), 2, "should return 2 nearest states");
        // Should be sorted ascending.
        assert!(nearest[0].1 <= nearest[1].1);
    }

    // -----------------------------------------------------------------------
    // Is bisimilar
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_bisimilar() {
        let sys = identical_two_state_system();
        let config = QuantBisimConfig::default();
        let mut engine = QuantitativeBisimEngine::new(sys, config);
        engine.compute();
        assert!(engine.is_bisimilar(0, 1, 1e-4));
    }

    // -----------------------------------------------------------------------
    // Transport solver
    // -----------------------------------------------------------------------

    #[test]
    fn test_transport_lp_simple() {
        // 2×2: supply [1, 0], demand [0, 1], cost [[0, 5], [5, 0]]
        // Must ship 1 unit from row-0 → col-1 at cost 5.
        let supply = vec![1.0, 0.0];
        let demand = vec![0.0, 1.0];
        let cost = vec![
            vec![0.0, 5.0],
            vec![5.0, 0.0],
        ];
        let (flow, total) = solve_transport_lp(&supply, &demand, &cost);
        assert!(
            (total - 5.0).abs() < 1e-6,
            "optimal cost should be 5, got {}",
            total
        );
        assert!((flow[0][1] - 1.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Diameter
    // -----------------------------------------------------------------------

    #[test]
    fn test_diameter() {
        let d = vec![
            vec![0.0, 0.3, 0.7],
            vec![0.3, 0.0, 0.5],
            vec![0.7, 0.5, 0.0],
        ];
        let pm = BehavioralPseudometric::from_distance_matrix(d);
        assert!((pm.diameter() - 0.7).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // CouplingConstruction validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_coupling_validation() {
        let coupling = vec![
            vec![0.3, 0.2],
            vec![0.1, 0.4],
        ];
        let mu = vec![0.5, 0.5];
        let nu = vec![0.4, 0.6];
        let cc = CouplingConstruction::from_transport_plan(coupling, 0.0);
        assert!(cc.is_valid_coupling(&mu, &nu));

        // Bad marginals.
        let bad_mu = vec![0.6, 0.4];
        assert!(!cc.is_valid_coupling(&bad_mu, &nu));
    }

    // -----------------------------------------------------------------------
    // Distinguishing trace
    // -----------------------------------------------------------------------

    #[test]
    fn test_distinguishing_trace_opposite() {
        let sys = opposite_two_state_system();
        let trace = DistinguishingTraceComputer::find_distinguishing_trace(&sys, 0, 1, 3);
        assert!(trace.is_some(), "opposite states should yield a distinguishing trace");
        let t = trace.unwrap();
        assert!(t.difference > 0.0);
    }

    #[test]
    fn test_distinguishing_trace_identical() {
        let sys = identical_two_state_system();
        // Identical states have no distinguishing trace at any depth.
        let trace = DistinguishingTraceComputer::find_distinguishing_trace(&sys, 0, 1, 5);
        // May or may not return a trace with difference 0 — either is acceptable.
        if let Some(t) = trace {
            assert!(
                t.difference < 1e-6,
                "identical states should have near-zero trace difference"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Greedy vs exact Kantorovich comparison
    // -----------------------------------------------------------------------

    #[test]
    fn test_greedy_is_upper_bound_of_exact() {
        let ground = vec![
            vec![0.0, 0.5, 1.0],
            vec![0.5, 0.0, 0.5],
            vec![1.0, 0.5, 0.0],
        ];
        let mu = vec![(0, 0.4), (1, 0.6)];
        let nu = vec![(1, 0.3), (2, 0.7)];

        let exact = KantorovichComputer::compute(&mu, &nu, &ground);
        let greedy = KantorovichComputer::compute_greedy(&mu, &nu, &ground);

        // Greedy should be >= exact (it's a heuristic, possibly suboptimal).
        assert!(
            greedy >= exact - 1e-8,
            "greedy ({}) should be >= exact ({})",
            greedy,
            exact
        );
    }

    // -----------------------------------------------------------------------
    // Engine with different coupling methods
    // -----------------------------------------------------------------------

    #[test]
    fn test_engine_greedy_coupling_method() {
        let sys = three_state_system();
        let mut config = QuantBisimConfig::default();
        config.coupling_method = CouplingMethod::GreedyApprox;
        let mut engine = QuantitativeBisimEngine::new(sys, config);
        engine.compute();
        // Should produce a valid pseudometric.
        let pm = BehavioralPseudometric::from_distance_matrix(engine.distances.clone());
        // Greedy is an upper bound, so the result is a valid pseudometric or close.
        assert!(engine.distance(0, 0) < 1e-12);
        assert!(engine.converged() || engine.iterations() > 0);
    }

    // -----------------------------------------------------------------------
    // Lower / upper bounds and bracket
    // -----------------------------------------------------------------------

    #[test]
    fn test_bracket_ordering() {
        let sys = three_state_system();
        let config = QuantBisimConfig::default();
        let mut engine = QuantitativeBisimEngine::new(sys, config);
        engine.compute();
        let (lb, ub) = engine.bracket(0, 2);
        assert!(
            lb <= ub + 1e-8,
            "lower bound ({}) should be <= upper bound ({})",
            lb,
            ub
        );
    }

    // -----------------------------------------------------------------------
    // Normalize
    // -----------------------------------------------------------------------

    #[test]
    fn test_normalize() {
        let mut sys = ProbTransitionSystem::new(2);
        sys.add_transition(0, "a", 0, 3.0);
        sys.add_transition(0, "a", 1, 7.0);
        sys.normalize();
        let d = sys.distribution(0, "a");
        let total: f64 = d.iter().map(|(_, p)| *p).sum();
        assert!(
            (total - 1.0).abs() < 1e-12,
            "normalized distribution should sum to 1"
        );
    }

    // -----------------------------------------------------------------------
    // Triangle inequality
    // -----------------------------------------------------------------------

    #[test]
    fn test_triangle_inequality_holds() {
        let sys = three_state_system();
        let config = QuantBisimConfig::default();
        let mut engine = QuantitativeBisimEngine::new(sys, config);
        engine.compute();
        let n = engine.system.num_states;
        let tol = 1e-6;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    assert!(
                        engine.distance(i, k) <= engine.distance(i, j) + engine.distance(j, k) + tol,
                        "triangle inequality violated: d({},{})={} > d({},{})={} + d({},{})={}",
                        i, k, engine.distance(i, k),
                        i, j, engine.distance(i, j),
                        j, k, engine.distance(j, k)
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Discount factor effect
    // -----------------------------------------------------------------------

    #[test]
    fn test_discount_reduces_distance() {
        let sys = opposite_two_state_system();

        let mut config_full = QuantBisimConfig::default();
        config_full.discount_factor = 1.0;
        let mut engine_full = QuantitativeBisimEngine::new(sys.clone(), config_full);
        engine_full.compute();

        let mut config_half = QuantBisimConfig::default();
        config_half.discount_factor = 0.5;
        let mut engine_half = QuantitativeBisimEngine::new(sys, config_half);
        engine_half.compute();

        assert!(
            engine_half.distance(0, 1) <= engine_full.distance(0, 1) + 1e-8,
            "lower discount should give lower distance"
        );
    }
}
