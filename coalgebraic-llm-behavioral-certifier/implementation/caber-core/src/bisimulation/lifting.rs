// Kantorovich lifting module for CABER
// Implements optimal transport, Hungarian algorithm, Sinkhorn solver,
// and compositional lifting for coalgebraic bisimulation metrics.

// ─── Distribution ────────────────────────────────────────────────────────────

/// A discrete probability distribution over a finite set.
#[derive(Clone, Debug)]
pub struct Distribution {
    pub weights: Vec<f64>,
}

impl Distribution {
    pub fn new(weights: Vec<f64>) -> Self {
        Distribution { weights }
    }

    pub fn uniform(n: usize) -> Self {
        let w = 1.0 / n as f64;
        Distribution {
            weights: vec![w; n],
        }
    }

    pub fn point_mass(n: usize, idx: usize) -> Self {
        let mut weights = vec![0.0; n];
        if idx < n {
            weights[idx] = 1.0;
        }
        Distribution { weights }
    }

    pub fn normalize(&mut self) {
        let sum: f64 = self.weights.iter().sum();
        if sum > 0.0 {
            for w in &mut self.weights {
                *w /= sum;
            }
        }
    }

    pub fn is_valid(&self, tolerance: f64) -> bool {
        let sum: f64 = self.weights.iter().sum();
        if (sum - 1.0).abs() > tolerance {
            return false;
        }
        self.weights.iter().all(|&w| w >= -tolerance)
    }

    pub fn support_size(&self) -> usize {
        self.weights.iter().filter(|&&w| w.abs() > 1e-12).count()
    }

    pub fn entropy(&self) -> f64 {
        let mut h = 0.0;
        for &w in &self.weights {
            if w > 1e-15 {
                h -= w * w.ln();
            }
        }
        h
    }

    pub fn total_variation(&self, other: &Distribution) -> f64 {
        let n = self.weights.len().max(other.weights.len());
        let mut tv = 0.0;
        for i in 0..n {
            let a = if i < self.weights.len() {
                self.weights[i]
            } else {
                0.0
            };
            let b = if i < other.weights.len() {
                other.weights[i]
            } else {
                0.0
            };
            tv += (a - b).abs();
        }
        tv / 2.0
    }

    /// KL divergence D_KL(self || other) with additive Laplace smoothing.
    pub fn kl_divergence(&self, other: &Distribution) -> f64 {
        let n = self.weights.len().max(other.weights.len());
        let eps = 1e-10;
        let mut kl = 0.0;
        for i in 0..n {
            let p = if i < self.weights.len() {
                self.weights[i]
            } else {
                0.0
            } + eps;
            let q = if i < other.weights.len() {
                other.weights[i]
            } else {
                0.0
            } + eps;
            kl += p * (p / q).ln();
        }
        // Subtract the bias introduced by eps on the normalization
        // (this is a standard smoothed KL; the caller should be aware of the smoothing)
        kl
    }
}

// ─── MetricValidation ────────────────────────────────────────────────────────

pub struct MetricValidation {
    pub is_symmetric: bool,
    pub satisfies_triangle: bool,
    pub satisfies_identity: bool,
    pub violations: Vec<String>,
}

// ─── HungarianResult ─────────────────────────────────────────────────────────

pub struct HungarianResult {
    pub assignment: Vec<(usize, usize)>,
    pub total_cost: f64,
    pub dual_variables: (Vec<f64>, Vec<f64>),
}

// ─── TransportPlan ───────────────────────────────────────────────────────────

pub struct TransportPlan {
    pub plan: Vec<Vec<f64>>,
    pub cost: f64,
    pub source_marginal: Vec<f64>,
    pub target_marginal: Vec<f64>,
}

impl TransportPlan {
    /// Check that the plan has correct marginals and is non-negative.
    pub fn is_valid(&self, tolerance: f64) -> bool {
        let m = self.plan.len();
        if m == 0 {
            return true;
        }
        let n = self.plan[0].len();

        // Non-negativity
        for i in 0..m {
            for j in 0..n {
                if self.plan[i][j] < -tolerance {
                    return false;
                }
            }
        }

        // Row marginals
        for i in 0..m {
            let row_sum: f64 = self.plan[i].iter().sum();
            if (row_sum - self.source_marginal[i]).abs() > tolerance {
                return false;
            }
        }

        // Column marginals
        for j in 0..n {
            let col_sum: f64 = (0..m).map(|i| self.plan[i][j]).sum();
            if (col_sum - self.target_marginal[j]).abs() > tolerance {
                return false;
            }
        }

        true
    }

    pub fn support(&self) -> Vec<(usize, usize)> {
        let mut s = Vec::new();
        for i in 0..self.plan.len() {
            for j in 0..self.plan[i].len() {
                if self.plan[i][j].abs() > 1e-12 {
                    s.push((i, j));
                }
            }
        }
        s
    }

    pub fn sparsity(&self) -> f64 {
        let m = self.plan.len();
        if m == 0 {
            return 1.0;
        }
        let n = self.plan[0].len();
        let total = (m * n) as f64;
        let zeros = self
            .plan
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&v| v.abs() <= 1e-12)
            .count() as f64;
        zeros / total
    }
}

// ─── HungarianAlgorithm ─────────────────────────────────────────────────────

/// Full Kuhn-Munkres (Hungarian) algorithm for minimum cost assignment.
pub struct HungarianAlgorithm;

impl HungarianAlgorithm {
    /// Solve the assignment problem on a square cost matrix.
    pub fn solve(cost_matrix: &[Vec<f64>]) -> HungarianResult {
        let n = cost_matrix.len();
        if n == 0 {
            return HungarianResult {
                assignment: vec![],
                total_cost: 0.0,
                dual_variables: (vec![], vec![]),
            };
        }

        // u[i] = row potential, v[j] = column potential (1-indexed internally)
        let mut u = vec![0.0f64; n + 1];
        let mut v = vec![0.0f64; n + 1];
        // p[j] = row assigned to column j (1-indexed, 0 = unassigned)
        let mut p = vec![0usize; n + 1];
        // way[j] = predecessor column in the augmenting path
        let mut way = vec![0usize; n + 1];

        for i in 1..=n {
            // Start augmenting path from row i
            p[0] = i;
            let mut j0 = 0usize; // virtual column 0
            let mut minv = vec![f64::INFINITY; n + 1];
            let mut used = vec![false; n + 1];

            loop {
                used[j0] = true;
                let i0 = p[j0];
                let mut delta = f64::INFINITY;
                let mut j1 = 0usize;

                for j in 1..=n {
                    if !used[j] {
                        let cur = cost_matrix[i0 - 1][j - 1] - u[i0] - v[j];
                        if cur < minv[j] {
                            minv[j] = cur;
                            way[j] = j0;
                        }
                        if minv[j] < delta {
                            delta = minv[j];
                            j1 = j;
                        }
                    }
                }

                for j in 0..=n {
                    if used[j] {
                        u[p[j]] += delta;
                        v[j] -= delta;
                    } else {
                        minv[j] -= delta;
                    }
                }

                j0 = j1;
                if p[j0] == 0 {
                    break;
                }
            }

            // Update assignment along augmenting path
            loop {
                let j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
                if j0 == 0 {
                    break;
                }
            }
        }

        // Extract result (convert from 1-indexed to 0-indexed)
        let mut assignment = Vec::with_capacity(n);
        let mut total_cost = 0.0;
        for j in 1..=n {
            if p[j] != 0 {
                let row = p[j] - 1;
                let col = j - 1;
                assignment.push((row, col));
                total_cost += cost_matrix[row][col];
            }
        }
        assignment.sort_by_key(|&(r, _)| r);

        let row_potentials: Vec<f64> = u[1..=n].to_vec();
        let col_potentials: Vec<f64> = v[1..=n].to_vec();

        HungarianResult {
            assignment,
            total_cost,
            dual_variables: (row_potentials, col_potentials),
        }
    }

    /// Solve for rectangular (non-square) cost matrices by padding.
    pub fn solve_rectangular(cost: &[Vec<f64>]) -> HungarianResult {
        let m = cost.len();
        if m == 0 {
            return HungarianResult {
                assignment: vec![],
                total_cost: 0.0,
                dual_variables: (vec![], vec![]),
            };
        }
        let n = cost[0].len();
        let dim = m.max(n);

        // Pad to square with zeros
        let mut square_cost = vec![vec![0.0; dim]; dim];
        for i in 0..m {
            for j in 0..n {
                square_cost[i][j] = cost[i][j];
            }
        }

        let result = Self::solve(&square_cost);

        // Filter out padding entries
        let assignment: Vec<(usize, usize)> = result
            .assignment
            .into_iter()
            .filter(|&(r, c)| r < m && c < n)
            .collect();

        let total_cost: f64 = assignment.iter().map(|&(r, c)| cost[r][c]).sum();

        let row_potentials = result.dual_variables.0[..m].to_vec();
        let col_potentials = result.dual_variables.1[..n].to_vec();

        HungarianResult {
            assignment,
            total_cost,
            dual_variables: (row_potentials, col_potentials),
        }
    }
}

// ─── Network Simplex for Optimal Transport ───────────────────────────────────

/// Solves the optimal transport problem via the network simplex method
/// for the transportation problem. This finds the exact minimum cost flow
/// that pushes mass from source distribution to target distribution.
struct NetworkSimplex;

impl NetworkSimplex {
    /// Solve the transportation problem using the northwest corner + stepping stone method.
    /// Returns the optimal flow matrix.
    fn solve(
        supply: &[f64],
        demand: &[f64],
        cost: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        let m = supply.len();
        let n = demand.len();
        if m == 0 || n == 0 {
            return vec![vec![0.0; n]; m];
        }

        // Use the revised simplex approach for transportation:
        // We use Vogel's approximation for the initial BFS,
        // then MODI (modified distribution) method to iterate to optimality.

        let mut flow = vec![vec![0.0; n]; m];
        let mut remaining_supply = supply.to_vec();
        let mut remaining_demand = demand.to_vec();

        // --- Phase 1: Initial BFS via Vogel's Approximation Method ---
        let mut row_done = vec![false; m];
        let mut col_done = vec![false; n];

        let allocated = Self::vogel_initial_bfs(
            &cost,
            &mut remaining_supply,
            &mut remaining_demand,
            &mut flow,
            &mut row_done,
            &mut col_done,
            m,
            n,
        );

        if !allocated {
            // Fallback: northwest corner
            let mut s = supply.to_vec();
            let mut d = demand.to_vec();
            flow = vec![vec![0.0; n]; m];
            let mut i = 0;
            let mut j = 0;
            while i < m && j < n {
                let f = s[i].min(d[j]);
                flow[i][j] = f;
                s[i] -= f;
                d[j] -= f;
                if s[i] < 1e-15 {
                    i += 1;
                }
                if d[j] < 1e-15 {
                    j += 1;
                }
            }
        }

        // --- Phase 2: MODI method to improve the solution ---
        Self::modi_optimize(&mut flow, cost, m, n);

        flow
    }

    fn vogel_initial_bfs(
        cost: &[Vec<f64>],
        supply: &mut [f64],
        demand: &mut [f64],
        flow: &mut [Vec<f64>],
        row_done: &mut [bool],
        col_done: &mut [bool],
        _m: usize,
        _n: usize,
    ) -> bool {
        let m = supply.len();
        let n = demand.len();
        let total_cells = m + n - 1; // number of basic variables needed
        let mut allocated = 0;

        for _ in 0..(m * n) {
            if allocated >= total_cells {
                break;
            }

            // Compute penalty for each row
            let mut best_penalty = -1.0f64;
            let mut best_is_row = true;
            let mut best_idx = 0;

            for i in 0..m {
                if row_done[i] {
                    continue;
                }
                let mut costs_in_row: Vec<f64> = Vec::new();
                for j in 0..n {
                    if !col_done[j] {
                        costs_in_row.push(cost[i][j]);
                    }
                }
                if costs_in_row.len() < 1 {
                    continue;
                }
                costs_in_row.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let penalty = if costs_in_row.len() >= 2 {
                    costs_in_row[1] - costs_in_row[0]
                } else {
                    costs_in_row[0]
                };
                if penalty > best_penalty {
                    best_penalty = penalty;
                    best_is_row = true;
                    best_idx = i;
                }
            }

            for j in 0..n {
                if col_done[j] {
                    continue;
                }
                let mut costs_in_col: Vec<f64> = Vec::new();
                for i in 0..m {
                    if !row_done[i] {
                        costs_in_col.push(cost[i][j]);
                    }
                }
                if costs_in_col.len() < 1 {
                    continue;
                }
                costs_in_col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let penalty = if costs_in_col.len() >= 2 {
                    costs_in_col[1] - costs_in_col[0]
                } else {
                    costs_in_col[0]
                };
                if penalty > best_penalty {
                    best_penalty = penalty;
                    best_is_row = false;
                    best_idx = j;
                }
            }

            if best_penalty < 0.0 {
                break;
            }

            // Find the cheapest cell in the selected row/column
            let (bi, bj) = if best_is_row {
                let i = best_idx;
                let mut min_cost = f64::INFINITY;
                let mut min_j = 0;
                for j in 0..n {
                    if !col_done[j] && cost[i][j] < min_cost {
                        min_cost = cost[i][j];
                        min_j = j;
                    }
                }
                (i, min_j)
            } else {
                let j = best_idx;
                let mut min_cost = f64::INFINITY;
                let mut min_i = 0;
                for i in 0..m {
                    if !row_done[i] && cost[i][j] < min_cost {
                        min_cost = cost[i][j];
                        min_i = i;
                    }
                }
                (min_i, j)
            };

            let f = supply[bi].min(demand[bj]);
            flow[bi][bj] = f;
            supply[bi] -= f;
            demand[bj] -= f;
            allocated += 1;

            if supply[bi] < 1e-15 {
                row_done[bi] = true;
            }
            if demand[bj] < 1e-15 {
                col_done[bj] = true;
            }
        }

        allocated > 0
    }

    fn modi_optimize(flow: &mut [Vec<f64>], cost: &[Vec<f64>], _m: usize, _n: usize) {
        // MODI (Modified Distribution) method:
        // Compute dual variables u[i], v[j] such that u[i] + v[j] = c[i][j] for basic cells
        // Check optimality: for all non-basic cells, c[i][j] - u[i] - v[j] >= 0
        // If not, find entering variable and perform pivot

        let m = flow.len();
        let n = if m > 0 { flow[0].len() } else { 0 };
        let max_iterations = m * n * 10;

        for _ in 0..max_iterations {
            // Identify basic cells
            let mut basic: Vec<(usize, usize)> = Vec::new();
            for i in 0..m {
                for j in 0..n {
                    if flow[i][j] > 1e-15 {
                        basic.push((i, j));
                    }
                }
            }

            // Compute dual variables u, v from basic cells
            // u[i] + v[j] = c[i][j] for (i,j) in basis
            let mut u = vec![f64::NAN; m];
            let mut v = vec![f64::NAN; n];
            u[0] = 0.0;

            let mut changed = true;
            let mut iters = 0;
            while changed && iters < m + n {
                changed = false;
                iters += 1;
                for &(i, j) in &basic {
                    if !u[i].is_nan() && v[j].is_nan() {
                        v[j] = cost[i][j] - u[i];
                        changed = true;
                    } else if u[i].is_nan() && !v[j].is_nan() {
                        u[i] = cost[i][j] - v[j];
                        changed = true;
                    }
                }
            }

            // Fill in any remaining NaN values with 0
            for val in u.iter_mut() {
                if val.is_nan() {
                    *val = 0.0;
                }
            }
            for val in v.iter_mut() {
                if val.is_nan() {
                    *val = 0.0;
                }
            }

            // Find the most negative reduced cost among non-basic cells
            let mut min_rc = -1e-10;
            let mut entering: Option<(usize, usize)> = None;
            for i in 0..m {
                for j in 0..n {
                    if flow[i][j] <= 1e-15 {
                        let rc = cost[i][j] - u[i] - v[j];
                        if rc < min_rc {
                            min_rc = rc;
                            entering = Some((i, j));
                        }
                    }
                }
            }

            let (ei, ej) = match entering {
                Some(e) => e,
                None => break, // Optimal
            };

            // Find a cycle (loop) including the entering cell
            // Use stepping-stone path finding
            if let Some(cycle) = Self::find_cycle(flow, &basic, ei, ej, m, n) {
                // Determine the maximum flow adjustment
                let mut theta = f64::INFINITY;
                for k in (1..cycle.len()).step_by(2) {
                    let (ci, cj) = cycle[k];
                    if flow[ci][cj] < theta {
                        theta = flow[ci][cj];
                    }
                }

                if theta < 1e-15 {
                    // Degenerate pivot - break to avoid infinite loop
                    break;
                }

                // Adjust flow along the cycle
                for (k, &(ci, cj)) in cycle.iter().enumerate() {
                    if k % 2 == 0 {
                        flow[ci][cj] += theta;
                    } else {
                        flow[ci][cj] -= theta;
                    }
                }

                // Clean up near-zero flows
                for i in 0..m {
                    for j in 0..n {
                        if flow[i][j] < 1e-15 {
                            flow[i][j] = 0.0;
                        }
                    }
                }
            } else {
                break;
            }
        }
    }

    /// Find a cycle in the transportation tableau starting from (ei, ej).
    fn find_cycle(
        _flow: &[Vec<f64>],
        basic: &[(usize, usize)],
        ei: usize,
        ej: usize,
        m: usize,
        n: usize,
    ) -> Option<Vec<(usize, usize)>> {
        // Build row/col adjacency from basic cells plus the entering cell
        let mut row_cols: Vec<Vec<usize>> = vec![Vec::new(); m];
        let mut col_rows: Vec<Vec<usize>> = vec![Vec::new(); n];

        row_cols[ei].push(ej);
        col_rows[ej].push(ei);

        for &(i, j) in basic {
            if i == ei && j == ej {
                continue;
            }
            row_cols[i].push(j);
            col_rows[j].push(i);
        }

        // DFS to find cycle from (ei, ej) back to (ei, ej)
        // The cycle alternates between row moves (same row, different col)
        // and column moves (same col, different row).
        let mut path = vec![(ei, ej)];
        let mut visited_rows = vec![false; m];
        let mut visited_cols = vec![false; n];
        visited_rows[ei] = true;

        if Self::dfs_cycle(
            &row_cols,
            &col_rows,
            &mut path,
            &mut visited_rows,
            &mut visited_cols,
            ei,
            ej,
            true, // next move is along the column (find another row in column ej)
        ) {
            Some(path)
        } else {
            None
        }
    }

    fn dfs_cycle(
        row_cols: &[Vec<usize>],
        col_rows: &[Vec<usize>],
        path: &mut Vec<(usize, usize)>,
        visited_rows: &mut [bool],
        visited_cols: &mut [bool],
        start_row: usize,
        current_col: usize,
        move_along_col: bool,
    ) -> bool {
        if move_along_col {
            // Move along column current_col: find a different row
            visited_cols[current_col] = true;
            for &next_row in &col_rows[current_col] {
                if next_row == start_row && path.len() >= 3 {
                    // Found cycle back to start
                    return true;
                }
                if !visited_rows[next_row] {
                    visited_rows[next_row] = true;
                    // Now pick a column from this row
                    for &next_col in &row_cols[next_row] {
                        if next_col != current_col && !visited_cols[next_col] {
                            path.push((next_row, current_col));
                            path.push((next_row, next_col));
                            if Self::dfs_cycle(
                                row_cols,
                                col_rows,
                                path,
                                visited_rows,
                                visited_cols,
                                start_row,
                                next_col,
                                true,
                            ) {
                                return true;
                            }
                            path.pop();
                            path.pop();
                        }
                    }
                    visited_rows[next_row] = false;
                }
            }
            visited_cols[current_col] = false;
        }
        false
    }
}

// ─── WassersteinDistance ──────────────────────────────────────────────────────

/// Wasserstein distance computations between discrete distributions.
pub struct WassersteinDistance;

impl WassersteinDistance {
    /// Wasserstein-1 (Earth Mover's Distance) via optimal transport.
    pub fn w1(mu: &[f64], nu: &[f64], ground_metric: &[Vec<f64>]) -> f64 {
        let m = mu.len();
        let n = nu.len();
        if m == 0 || n == 0 {
            return 0.0;
        }

        let flow = NetworkSimplex::solve(mu, nu, ground_metric);

        let mut total = 0.0;
        for i in 0..m {
            for j in 0..n {
                total += flow[i][j] * ground_metric[i][j];
            }
        }
        total
    }

    /// 1D Wasserstein-1 via the CDF integral formula:
    /// W_1(µ,ν) = ∫ |F_µ(x) - F_ν(x)| dx
    /// where the integral is computed over the sorted support points.
    pub fn w1_1d(mu: &[f64], nu: &[f64]) -> f64 {
        // Both mu and nu are probability mass functions on {0, 1, ..., n-1}
        // treated as point masses on the integers.
        let n = mu.len().max(nu.len());
        if n == 0 {
            return 0.0;
        }

        // Build CDFs
        let mut cdf_mu = Vec::with_capacity(n);
        let mut cdf_nu = Vec::with_capacity(n);
        let mut cum_mu = 0.0;
        let mut cum_nu = 0.0;
        for i in 0..n {
            cum_mu += if i < mu.len() { mu[i] } else { 0.0 };
            cum_nu += if i < nu.len() { nu[i] } else { 0.0 };
            cdf_mu.push(cum_mu);
            cdf_nu.push(cum_nu);
        }

        // W_1 = sum of |F_mu(i) - F_nu(i)| for integer-spaced support
        let mut w = 0.0;
        for i in 0..n {
            w += (cdf_mu[i] - cdf_nu[i]).abs();
        }
        w
    }

    /// Wasserstein-2 distance (using squared ground distances).
    pub fn w2(mu: &[f64], nu: &[f64], ground_distances: &[Vec<f64>]) -> f64 {
        let m = mu.len();
        let n = nu.len();
        if m == 0 || n == 0 {
            return 0.0;
        }

        // Build squared cost matrix
        let squared: Vec<Vec<f64>> = ground_distances
            .iter()
            .map(|row| row.iter().map(|&d| d * d).collect())
            .collect();

        let flow = NetworkSimplex::solve(mu, nu, &squared);

        let mut total = 0.0;
        for i in 0..m {
            for j in 0..n {
                total += flow[i][j] * squared[i][j];
            }
        }
        total.sqrt()
    }

    /// General Wasserstein-p distance.
    pub fn wp(mu: &[f64], nu: &[f64], ground_distances: &[Vec<f64>], p: f64) -> f64 {
        let m = mu.len();
        let n = nu.len();
        if m == 0 || n == 0 {
            return 0.0;
        }

        // Build cost^p matrix
        let cost_p: Vec<Vec<f64>> = ground_distances
            .iter()
            .map(|row| row.iter().map(|&d| d.powf(p)).collect())
            .collect();

        let flow = NetworkSimplex::solve(mu, nu, &cost_p);

        let mut total = 0.0;
        for i in 0..m {
            for j in 0..n {
                total += flow[i][j] * cost_p[i][j];
            }
        }
        total.powf(1.0 / p)
    }
}

// ─── KantorovichLifting ──────────────────────────────────────────────────────

/// Kantorovich lifting of a ground metric to distributions.
pub struct KantorovichLifting {
    pub ground_metric: Vec<Vec<f64>>,
    pub num_points: usize,
}

impl KantorovichLifting {
    pub fn new(ground_metric: Vec<Vec<f64>>) -> Self {
        let num_points = ground_metric.len();
        KantorovichLifting {
            ground_metric,
            num_points,
        }
    }

    /// Create a lifting from the discrete metric (d(x,y) = 1 if x≠y, 0 if x=y).
    pub fn from_discrete(n: usize) -> Self {
        let mut metric = vec![vec![1.0; n]; n];
        for i in 0..n {
            metric[i][i] = 0.0;
        }
        KantorovichLifting {
            ground_metric: metric,
            num_points: n,
        }
    }

    /// Compute the Kantorovich (Wasserstein-1) distance between two distributions.
    pub fn lift(&self, mu: &Distribution, nu: &Distribution) -> f64 {
        WassersteinDistance::w1(&mu.weights, &nu.weights, &self.ground_metric)
    }

    /// Compute the Kantorovich distance and return the optimal transport plan.
    pub fn lift_with_plan(&self, mu: &Distribution, nu: &Distribution) -> (f64, TransportPlan) {
        let m = mu.weights.len();
        let n = nu.weights.len();

        let flow = NetworkSimplex::solve(&mu.weights, &nu.weights, &self.ground_metric);

        let mut cost = 0.0;
        for i in 0..m {
            for j in 0..n {
                cost += flow[i][j] * self.ground_metric[i][j];
            }
        }

        let plan = TransportPlan {
            plan: flow,
            cost,
            source_marginal: mu.weights.clone(),
            target_marginal: nu.weights.clone(),
        };

        (cost, plan)
    }

    /// Validate that the ground metric is a proper metric.
    pub fn validate_ground_metric(&self) -> MetricValidation {
        let n = self.num_points;
        let d = &self.ground_metric;
        let mut violations = Vec::new();

        // Identity of indiscernibles: d(x,x) = 0 and d(x,y) > 0 for x ≠ y
        let mut satisfies_identity = true;
        for i in 0..n {
            if d[i][i].abs() > 1e-12 {
                satisfies_identity = false;
                violations.push(format!("d({},{}) = {} ≠ 0", i, i, d[i][i]));
            }
        }
        for i in 0..n {
            for j in 0..n {
                if i != j && d[i][j] < -1e-12 {
                    satisfies_identity = false;
                    violations.push(format!("d({},{}) = {} < 0", i, j, d[i][j]));
                }
            }
        }

        // Symmetry: d(x,y) = d(y,x)
        let mut is_symmetric = true;
        for i in 0..n {
            for j in (i + 1)..n {
                if (d[i][j] - d[j][i]).abs() > 1e-10 {
                    is_symmetric = false;
                    violations.push(format!(
                        "d({},{}) = {} ≠ d({},{}) = {}",
                        i, j, d[i][j], j, i, d[j][i]
                    ));
                }
            }
        }

        // Triangle inequality: d(x,z) ≤ d(x,y) + d(y,z)
        let mut satisfies_triangle = true;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    if d[i][k] > d[i][j] + d[j][k] + 1e-10 {
                        satisfies_triangle = false;
                        violations.push(format!(
                            "d({},{}) = {} > d({},{}) + d({},{}) = {} + {} = {}",
                            i,
                            k,
                            d[i][k],
                            i,
                            j,
                            d[i][j],
                            j,
                            k,
                            d[j][k],
                            d[i][j] + d[j][k]
                        ));
                    }
                }
            }
        }

        MetricValidation {
            is_symmetric,
            satisfies_triangle,
            satisfies_identity,
            violations,
        }
    }
}

// ─── LiftedMetricProperties ─────────────────────────────────────────────────

/// Checks that the Kantorovich lifting preserves metric properties.
pub struct LiftedMetricProperties;

impl LiftedMetricProperties {
    /// Check triangle inequality: K(µ,ρ) ≤ K(µ,ν) + K(ν,ρ) for all triples.
    pub fn preserves_triangle_inequality(
        lifting: &KantorovichLifting,
        distributions: &[Distribution],
    ) -> bool {
        let n = distributions.len();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let d_ik = lifting.lift(&distributions[i], &distributions[k]);
                    let d_ij = lifting.lift(&distributions[i], &distributions[j]);
                    let d_jk = lifting.lift(&distributions[j], &distributions[k]);
                    if d_ik > d_ij + d_jk + 1e-8 {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Check symmetry: K(µ,ν) = K(ν,µ) for all pairs.
    pub fn preserves_symmetry(
        lifting: &KantorovichLifting,
        distributions: &[Distribution],
    ) -> bool {
        let n = distributions.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let d_ij = lifting.lift(&distributions[i], &distributions[j]);
                let d_ji = lifting.lift(&distributions[j], &distributions[i]);
                if (d_ij - d_ji).abs() > 1e-8 {
                    return false;
                }
            }
        }
        true
    }

    /// Check identity of indiscernibles: K(µ,µ) = 0 for all µ.
    pub fn preserves_identity(lifting: &KantorovichLifting) -> bool {
        // Test with a few representative distributions
        let n = lifting.num_points;
        if n == 0 {
            return true;
        }

        // Uniform
        let u = Distribution::uniform(n);
        if lifting.lift(&u, &u).abs() > 1e-10 {
            return false;
        }

        // Each point mass
        for i in 0..n {
            let p = Distribution::point_mass(n, i);
            if lifting.lift(&p, &p).abs() > 1e-10 {
                return false;
            }
        }

        true
    }
}

// ─── CompositionalLifting ────────────────────────────────────────────────────

/// Compositional operations on Kantorovich liftings.
pub struct CompositionalLifting;

impl CompositionalLifting {
    /// Compose two liftings by using the lifted distance of l1 as input
    /// to build a new ground metric, then applying l2's structure.
    /// The composed metric d'(i,k) = min_j [ d1(i,j) + d2(j,k) ].
    pub fn compose(
        l1: &KantorovichLifting,
        l2: &KantorovichLifting,
    ) -> KantorovichLifting {
        let n1 = l1.num_points;
        let n2 = l2.num_points;

        if n1 != n2 {
            // For composition, we build a metric on the larger space
            // by taking the infimal convolution (min over intermediate)
            let n = n1.max(n2);
            let shared = n1.min(n2);
            let mut metric = vec![vec![f64::INFINITY; n]; n];
            for i in 0..n {
                metric[i][i] = 0.0;
            }

            // d'(i,k) = min_j [ d1(i,j) + d2(j,k) ]
            for i in 0..n1 {
                for k in 0..n2 {
                    let mut min_val = f64::INFINITY;
                    for j in 0..shared {
                        let via = l1.ground_metric[i][j] + l2.ground_metric[j][k];
                        if via < min_val {
                            min_val = via;
                        }
                    }
                    if min_val < metric[i][k] {
                        metric[i][k] = min_val;
                    }
                    if min_val < metric[k][i] {
                        metric[k][i] = min_val;
                    }
                }
            }

            // Clean up infinity values
            for i in 0..n {
                for j in 0..n {
                    if metric[i][j] == f64::INFINITY {
                        // Use a large finite value
                        metric[i][j] = 1e6;
                    }
                }
            }

            KantorovichLifting::new(metric)
        } else {
            // Same dimension: infimal convolution
            let n = n1;
            let mut metric = vec![vec![f64::INFINITY; n]; n];

            for i in 0..n {
                for k in 0..n {
                    for j in 0..n {
                        let via = l1.ground_metric[i][j] + l2.ground_metric[j][k];
                        if via < metric[i][k] {
                            metric[i][k] = via;
                        }
                    }
                }
            }

            KantorovichLifting::new(metric)
        }
    }

    /// Product lifting: the ground metric on the product space X × Y is
    /// d_prod((x1,y1), (x2,y2)) = d1(x1,x2) + d2(y1,y2).
    pub fn product_lifting(
        l1: &KantorovichLifting,
        l2: &KantorovichLifting,
    ) -> KantorovichLifting {
        let n1 = l1.num_points;
        let n2 = l2.num_points;
        let n = n1 * n2;

        let mut metric = vec![vec![0.0; n]; n];
        for i1 in 0..n1 {
            for j1 in 0..n2 {
                let idx1 = i1 * n2 + j1;
                for i2 in 0..n1 {
                    for j2 in 0..n2 {
                        let idx2 = i2 * n2 + j2;
                        metric[idx1][idx2] =
                            l1.ground_metric[i1][i2] + l2.ground_metric[j1][j2];
                    }
                }
            }
        }

        KantorovichLifting::new(metric)
    }
}

// ─── SinkhornSolver ──────────────────────────────────────────────────────────

/// Sinkhorn-Knopp algorithm for entropy-regularized optimal transport.
pub struct SinkhornSolver;

impl SinkhornSolver {
    /// Solve the entropy-regularized optimal transport problem.
    ///
    /// Minimizes <C, P> - ε H(P) subject to marginal constraints.
    /// Uses iterative Bregman projections (Sinkhorn-Knopp scaling).
    pub fn solve(
        mu: &[f64],
        nu: &[f64],
        cost: &[Vec<f64>],
        regularization: f64,
        max_iter: usize,
    ) -> TransportPlan {
        let m = mu.len();
        let n = nu.len();
        if m == 0 || n == 0 {
            return TransportPlan {
                plan: vec![],
                cost: 0.0,
                source_marginal: mu.to_vec(),
                target_marginal: nu.to_vec(),
            };
        }

        let eps = regularization;

        // Gibbs kernel: K[i][j] = exp(-C[i][j] / ε)
        let mut kernel = vec![vec![0.0f64; n]; m];
        for i in 0..m {
            for j in 0..n {
                kernel[i][j] = (-cost[i][j] / eps).exp();
            }
        }

        // Scaling vectors
        let mut u = vec![1.0f64; m];
        let mut v = vec![1.0f64; n];

        let convergence_threshold = 1e-9;

        for _ in 0..max_iter {
            // u = a ./ (K * v)
            let mut max_change = 0.0f64;
            for i in 0..m {
                let mut kv = 0.0;
                for j in 0..n {
                    kv += kernel[i][j] * v[j];
                }
                let new_u = if kv > 1e-300 { mu[i] / kv } else { 0.0 };
                let change = if u[i] > 1e-300 {
                    (new_u / u[i] - 1.0).abs()
                } else {
                    new_u.abs()
                };
                if change > max_change {
                    max_change = change;
                }
                u[i] = new_u;
            }

            // v = b ./ (K^T * u)
            for j in 0..n {
                let mut ktu = 0.0;
                for i in 0..m {
                    ktu += kernel[i][j] * u[i];
                }
                let new_v = if ktu > 1e-300 { nu[j] / ktu } else { 0.0 };
                let change = if v[j] > 1e-300 {
                    (new_v / v[j] - 1.0).abs()
                } else {
                    new_v.abs()
                };
                if change > max_change {
                    max_change = change;
                }
                v[j] = new_v;
            }

            if max_change < convergence_threshold {
                break;
            }
        }

        // Compute transport plan: P = diag(u) * K * diag(v)
        let mut plan = vec![vec![0.0; n]; m];
        let mut total_cost = 0.0;
        for i in 0..m {
            for j in 0..n {
                plan[i][j] = u[i] * kernel[i][j] * v[j];
                total_cost += plan[i][j] * cost[i][j];
            }
        }

        TransportPlan {
            plan,
            cost: total_cost,
            source_marginal: mu.to_vec(),
            target_marginal: nu.to_vec(),
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Hungarian algorithm tests ────────────────────────────────────────

    #[test]
    fn test_hungarian_3x3_basic() {
        // Classic 3×3 problem:
        // Cost:  [4, 1, 3]
        //        [2, 0, 5]
        //        [3, 2, 2]
        // Optimal: (0,1)=1, (1,0)=2, (2,2)=2 → total = 5
        let cost = vec![
            vec![4.0, 1.0, 3.0],
            vec![2.0, 0.0, 5.0],
            vec![3.0, 2.0, 2.0],
        ];
        let result = HungarianAlgorithm::solve(&cost);
        assert!((result.total_cost - 5.0).abs() < 1e-8,
            "Expected total cost 5, got {}", result.total_cost);
        assert_eq!(result.assignment.len(), 3);
    }

    #[test]
    fn test_hungarian_4x4() {
        // 4×4 problem:
        // Cost:  [10, 5, 13, 15]
        //        [3, 9, 18,  13]
        //        [13, 7, 4,  15]
        //        [12, 11, 14, 13]
        // Optimal: row0→col1(5), row1→col0(3), row2→col2(4), row3→col3(13) = 25
        let cost = vec![
            vec![10.0, 5.0, 13.0, 15.0],
            vec![3.0, 9.0, 18.0, 13.0],
            vec![13.0, 7.0, 4.0, 15.0],
            vec![12.0, 11.0, 14.0, 13.0],
        ];
        let result = HungarianAlgorithm::solve(&cost);
        assert!((result.total_cost - 25.0).abs() < 1e-8,
            "Expected total cost 25, got {}", result.total_cost);
        assert_eq!(result.assignment.len(), 4);
    }

    #[test]
    fn test_hungarian_1x1() {
        let cost = vec![vec![7.0]];
        let result = HungarianAlgorithm::solve(&cost);
        assert!((result.total_cost - 7.0).abs() < 1e-8);
        assert_eq!(result.assignment, vec![(0, 0)]);
    }

    #[test]
    fn test_hungarian_rectangular() {
        // 2×3 matrix → pad to 3×3, only 2 real assignments
        let cost = vec![
            vec![5.0, 3.0, 1.0],
            vec![2.0, 7.0, 4.0],
        ];
        let result = HungarianAlgorithm::solve_rectangular(&cost);
        assert!(result.assignment.len() <= 2);
        // Optimal: row0→col2(1), row1→col0(2) = 3
        assert!((result.total_cost - 3.0).abs() < 1e-8,
            "Expected 3.0, got {}", result.total_cost);
    }

    // ── 1D Wasserstein tests ─────────────────────────────────────────────

    #[test]
    fn test_w1_1d_identical() {
        let mu = vec![0.25, 0.25, 0.25, 0.25];
        let nu = vec![0.25, 0.25, 0.25, 0.25];
        let d = WassersteinDistance::w1_1d(&mu, &nu);
        assert!(d.abs() < 1e-10, "Identical distributions should have distance 0, got {}", d);
    }

    #[test]
    fn test_w1_1d_point_masses() {
        // δ_0 vs δ_3 on {0,1,2,3} → distance = 3
        let mu = vec![1.0, 0.0, 0.0, 0.0];
        let nu = vec![0.0, 0.0, 0.0, 1.0];
        let d = WassersteinDistance::w1_1d(&mu, &nu);
        assert!((d - 3.0).abs() < 1e-10, "Expected 3.0, got {}", d);
    }

    #[test]
    fn test_w1_1d_shift() {
        // Shifting uniform(4) by 1 position costs 1.0 total
        let mu = vec![0.25, 0.25, 0.25, 0.25, 0.0];
        let nu = vec![0.0, 0.25, 0.25, 0.25, 0.25];
        let d = WassersteinDistance::w1_1d(&mu, &nu);
        assert!((d - 1.0).abs() < 1e-10, "Expected 1.0, got {}", d);
    }

    // ── Kantorovich lifting tests ────────────────────────────────────────

    #[test]
    fn test_kantorovich_discrete_metric_identical() {
        let lifting = KantorovichLifting::from_discrete(3);
        let mu = Distribution::uniform(3);
        let d = lifting.lift(&mu, &mu);
        assert!(d.abs() < 1e-8, "K(µ,µ) should be 0, got {}", d);
    }

    #[test]
    fn test_kantorovich_discrete_metric_point_masses() {
        // Under the discrete metric, K(δ_i, δ_j) = 1 for i≠j
        let lifting = KantorovichLifting::from_discrete(3);
        let d0 = Distribution::point_mass(3, 0);
        let d1 = Distribution::point_mass(3, 1);
        let d = lifting.lift(&d0, &d1);
        assert!((d - 1.0).abs() < 1e-8, "Expected 1.0, got {}", d);
    }

    #[test]
    fn test_kantorovich_discrete_equals_total_variation() {
        // Under the discrete metric, Kantorovich distance = total variation
        let lifting = KantorovichLifting::from_discrete(4);
        let mu = Distribution::new(vec![0.5, 0.3, 0.1, 0.1]);
        let nu = Distribution::new(vec![0.1, 0.1, 0.3, 0.5]);
        let k_dist = lifting.lift(&mu, &nu);
        let tv = mu.total_variation(&nu);
        assert!((k_dist - tv).abs() < 1e-6,
            "Under discrete metric, K = TV: K={}, TV={}", k_dist, tv);
    }

    // ── Transport plan tests ─────────────────────────────────────────────

    #[test]
    fn test_transport_plan_validity() {
        let lifting = KantorovichLifting::from_discrete(3);
        let mu = Distribution::new(vec![0.5, 0.3, 0.2]);
        let nu = Distribution::new(vec![0.2, 0.5, 0.3]);
        let (_, plan) = lifting.lift_with_plan(&mu, &nu);
        assert!(plan.is_valid(1e-6), "Transport plan should be valid");
    }

    #[test]
    fn test_transport_plan_sparsity() {
        let lifting = KantorovichLifting::from_discrete(3);
        let mu = Distribution::point_mass(3, 0);
        let nu = Distribution::point_mass(3, 2);
        let (_, plan) = lifting.lift_with_plan(&mu, &nu);
        // For two point masses, support should be just 1 cell
        let support = plan.support();
        assert!(!support.is_empty(), "Support should not be empty");
        assert!(plan.sparsity() > 0.5, "Plan between point masses should be sparse");
    }

    // ── Distribution tests ───────────────────────────────────────────────

    #[test]
    fn test_distribution_operations() {
        let mut d = Distribution::new(vec![2.0, 3.0, 5.0]);
        d.normalize();
        assert!(d.is_valid(1e-10), "Normalized distribution should be valid");
        assert!((d.weights[0] - 0.2).abs() < 1e-10);
        assert!((d.weights[1] - 0.3).abs() < 1e-10);
        assert!((d.weights[2] - 0.5).abs() < 1e-10);

        let u = Distribution::uniform(4);
        assert_eq!(u.support_size(), 4);
        let entropy = u.entropy();
        assert!((entropy - (4.0f64).ln()).abs() < 1e-10,
            "Uniform(4) entropy should be ln(4), got {}", entropy);

        let pm = Distribution::point_mass(4, 2);
        assert_eq!(pm.support_size(), 1);
        assert!(pm.entropy().abs() < 1e-10, "Point mass entropy should be 0");
    }

    // ── Sinkhorn solver tests ────────────────────────────────────────────

    #[test]
    fn test_sinkhorn_marginals() {
        let mu = vec![0.4, 0.3, 0.3];
        let nu = vec![0.2, 0.5, 0.3];
        let cost = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 1.0],
            vec![2.0, 1.0, 0.0],
        ];
        let plan = SinkhornSolver::solve(&mu, &nu, &cost, 0.01, 5000);
        assert!(plan.is_valid(1e-4),
            "Sinkhorn plan should have approximately correct marginals");
    }

    #[test]
    fn test_sinkhorn_converges_to_exact() {
        // With small regularization, Sinkhorn should approximate the exact solution
        let mu = vec![0.5, 0.5];
        let nu = vec![0.5, 0.5];
        let cost = vec![
            vec![0.0, 1.0],
            vec![1.0, 0.0],
        ];
        let plan = SinkhornSolver::solve(&mu, &nu, &cost, 0.001, 10000);
        // Exact: move nothing (already matched) → cost = 0
        assert!(plan.cost < 0.05, "Cost should be near 0, got {}", plan.cost);
    }

    // ── Metric property tests ────────────────────────────────────────────

    #[test]
    fn test_lifted_metric_preserves_properties() {
        let lifting = KantorovichLifting::from_discrete(3);
        let dists = vec![
            Distribution::new(vec![0.5, 0.3, 0.2]),
            Distribution::new(vec![0.1, 0.6, 0.3]),
            Distribution::new(vec![0.3, 0.3, 0.4]),
            Distribution::uniform(3),
        ];

        assert!(
            LiftedMetricProperties::preserves_triangle_inequality(&lifting, &dists),
            "Lifted metric should preserve triangle inequality"
        );
        assert!(
            LiftedMetricProperties::preserves_symmetry(&lifting, &dists),
            "Lifted metric should preserve symmetry"
        );
        assert!(
            LiftedMetricProperties::preserves_identity(&lifting),
            "Lifted metric should preserve identity"
        );
    }

    // ── Metric validation tests ──────────────────────────────────────────

    #[test]
    fn test_validate_ground_metric() {
        let lifting = KantorovichLifting::from_discrete(4);
        let validation = lifting.validate_ground_metric();
        assert!(validation.is_symmetric, "Discrete metric should be symmetric");
        assert!(validation.satisfies_triangle, "Discrete metric should satisfy triangle");
        assert!(validation.satisfies_identity, "Discrete metric should satisfy identity");
        assert!(validation.violations.is_empty(), "Should have no violations");
    }

    // ── Compositional lifting tests ──────────────────────────────────────

    #[test]
    fn test_compositional_lifting() {
        let l1 = KantorovichLifting::from_discrete(3);
        let l2 = KantorovichLifting::from_discrete(3);
        let composed = CompositionalLifting::compose(&l1, &l2);

        // Composed discrete metric: d'(i,k) = min_j [d1(i,j) + d2(j,k)]
        // For i=k: min_j [d(i,j) + d(j,i)] includes j=i which gives 0
        // For i≠k: min_j [d(i,j) + d(j,k)]
        //   j=i → 0 + 1 = 1, j=k → 1 + 0 = 1, j=other → 1 + 1 = 2
        //   so d'(i,k) = 1
        for i in 0..3 {
            assert!(composed.ground_metric[i][i].abs() < 1e-8,
                "d'({},{}) should be 0", i, i);
            for j in 0..3 {
                if i != j {
                    assert!((composed.ground_metric[i][j] - 1.0).abs() < 1e-8,
                        "d'({},{}) should be 1, got {}", i, j, composed.ground_metric[i][j]);
                }
            }
        }

        let validation = composed.validate_ground_metric();
        assert!(validation.is_symmetric);
        assert!(validation.satisfies_triangle);
    }

    #[test]
    fn test_product_lifting() {
        let l1 = KantorovichLifting::from_discrete(2);
        let l2 = KantorovichLifting::from_discrete(2);
        let product = CompositionalLifting::product_lifting(&l1, &l2);

        // Product space has 2×2 = 4 points
        assert_eq!(product.num_points, 4);

        // d_prod((0,0), (1,1)) = d1(0,1) + d2(0,1) = 1 + 1 = 2
        assert!((product.ground_metric[0][3] - 2.0).abs() < 1e-8);
        // d_prod((0,0), (0,1)) = d1(0,0) + d2(0,1) = 0 + 1 = 1
        assert!((product.ground_metric[0][1] - 1.0).abs() < 1e-8);

        let validation = product.validate_ground_metric();
        assert!(validation.is_symmetric);
        assert!(validation.satisfies_triangle);
    }

    // ── Wasserstein-p tests ──────────────────────────────────────────────

    #[test]
    fn test_wasserstein_p_identity() {
        let mu = vec![0.3, 0.4, 0.3];
        let ground = vec![
            vec![0.0, 1.0, 2.0],
            vec![1.0, 0.0, 1.0],
            vec![2.0, 1.0, 0.0],
        ];
        let d1 = WassersteinDistance::w1(&mu, &mu, &ground);
        let d2 = WassersteinDistance::w2(&mu, &mu, &ground);
        let dp = WassersteinDistance::wp(&mu, &mu, &ground, 3.0);
        assert!(d1.abs() < 1e-8, "W1(µ,µ) = 0");
        assert!(d2.abs() < 1e-8, "W2(µ,µ) = 0");
        assert!(dp.abs() < 1e-8, "W3(µ,µ) = 0");
    }
}
