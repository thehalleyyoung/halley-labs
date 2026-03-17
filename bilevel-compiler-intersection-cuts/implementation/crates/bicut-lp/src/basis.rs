//! Basis management: LU factorization with partial pivoting,
//! product-form and Forrest-Tomlin updates, refactorization triggers,
//! numerical stability monitoring, and eta vectors.

use std::fmt;

/// Tolerance for treating values as zero in the basis.
const ZERO_TOL: f64 = 1e-13;
/// Threshold for partial pivoting in LU factorization.
const PIVOT_THRESHOLD: f64 = 0.01;
/// Maximum eta file length before refactorization.
const MAX_ETA_COUNT: usize = 100;
/// Stability ratio triggering refactorization.
const STABILITY_THRESHOLD: f64 = 1e8;

/// Eta vector: represents an elementary column operation.
#[derive(Debug, Clone)]
pub struct EtaVector {
    pub col: usize,
    pub indices: Vec<usize>,
    pub values: Vec<f64>,
}

impl EtaVector {
    pub fn new(col: usize) -> Self {
        Self {
            col,
            indices: Vec::new(),
            values: Vec::new(),
        }
    }

    pub fn add_entry(&mut self, row: usize, value: f64) {
        self.indices.push(row);
        self.values.push(value);
    }

    /// Apply eta vector to a dense vector: v = E * v where E differs from I in column `col`.
    pub fn apply_forward(&self, v: &mut [f64]) {
        let pivot_val = self
            .indices
            .iter()
            .zip(self.values.iter())
            .find(|(&i, _)| i == self.col)
            .map(|(_, &val)| val)
            .unwrap_or(1.0);

        if pivot_val.abs() < ZERO_TOL {
            return;
        }

        let old_v_col = v[self.col];
        let mut new_v_col = 0.0;
        for (&i, &eta_val) in self.indices.iter().zip(self.values.iter()) {
            if i == self.col {
                new_v_col += eta_val * old_v_col;
            } else {
                new_v_col += eta_val * v[i];
                // Other components: v[i] stays the same since the eta column only modifies col
            }
        }
        // Actually, the eta matrix E has column `col` replaced.
        // E * v = v + (eta_col - e_col) * v[col]
        // v'[i] = v[i] + (eta[i] - delta(i,col)) * v[col]
        let factor = old_v_col;
        for (&i, &eta_val) in self.indices.iter().zip(self.values.iter()) {
            if i == self.col {
                v[i] = eta_val * factor;
            } else {
                v[i] += eta_val * factor;
            }
        }
        // Subtract the original identity contribution for col
        // Actually for non-col rows: v'[i] = v[i] + eta[i] * v[col] (since delta(i,col)=0)
        // For col row: v'[col] = eta[col] * v[col] (since delta(col,col)=1 is replaced)
        // The above loop does this. Let me re-derive:
        // We need to be careful. Let me just re-do:
        // Reset and redo properly.
        // v[i] was already modified, so restore first.
        for (&i, &eta_val) in self.indices.iter().zip(self.values.iter()) {
            if i == self.col {
                v[i] = old_v_col; // restore
            } else {
                v[i] -= eta_val * factor; // undo
            }
        }
        // Now apply properly:
        // E = I + (eta - e_col) * e_col^T
        // (E*v)[i] = v[i] + (eta[i] - delta(i,col)) * v[col]
        // For i != col: (E*v)[i] = v[i] + eta[i]*v[col]
        // For i == col: (E*v)[col] = v[col] + (eta[col] - 1)*v[col] = eta[col]*v[col]
        let v_col = v[self.col];
        for (&i, &eta_val) in self.indices.iter().zip(self.values.iter()) {
            if i == self.col {
                v[self.col] = eta_val * v_col;
            } else {
                v[i] += eta_val * v_col;
            }
        }
    }

    /// Apply eta vector transpose: v = E^T * v.
    pub fn apply_backward(&self, v: &mut [f64]) {
        // E^T = I + e_col * (eta - e_col)^T
        // (E^T * v)[col] = v[col] + sum_{i} (eta[i] - delta(i,col)) * v[i]
        // (E^T * v)[j] = v[j] for j != col
        // Wait—more carefully:
        // E = I + (eta_col - e_col) * e_col^T
        // E^T = I + e_col * (eta_col - e_col)^T
        // (E^T v)[j] = v[j] + e_col[j] * (eta_col - e_col)^T v
        //            = v[j] + delta(j,col) * sum_i (eta[i]-delta(i,col))*v[i]
        // So only v[col] changes:
        // v'[col] = v[col] + sum_i (eta[i]-delta(i,col))*v[i]
        //         = v[col] + sum_{i!=col} eta[i]*v[i] + (eta[col]-1)*v[col]
        //         = eta[col]*v[col] + sum_{i!=col} eta[i]*v[i]
        let mut new_v_col = 0.0;
        for (&i, &eta_val) in self.indices.iter().zip(self.values.iter()) {
            new_v_col += eta_val * v[i];
        }
        v[self.col] = new_v_col;
    }
}

/// Permutation vector.
#[derive(Debug, Clone)]
pub struct Permutation {
    pub perm: Vec<usize>,
    pub inv: Vec<usize>,
}

impl Permutation {
    pub fn identity(n: usize) -> Self {
        let perm: Vec<usize> = (0..n).collect();
        Self {
            inv: perm.clone(),
            perm,
        }
    }

    pub fn swap(&mut self, i: usize, j: usize) {
        self.perm.swap(i, j);
        let pi = self.perm[i];
        let pj = self.perm[j];
        self.inv[pi] = i;
        self.inv[pj] = j;
    }

    pub fn apply(&self, v: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; v.len()];
        for (i, &p) in self.perm.iter().enumerate() {
            result[i] = v[p];
        }
        result
    }

    pub fn apply_inverse(&self, v: &[f64]) -> Vec<f64> {
        let mut result = vec![0.0; v.len()];
        for (i, &p) in self.inv.iter().enumerate() {
            result[i] = v[p];
        }
        result
    }
}

/// LU factorization of the basis matrix with partial pivoting.
#[derive(Debug, Clone)]
pub struct LuFactorization {
    pub n: usize,
    /// L factor stored column-by-column (lower triangular, unit diagonal).
    pub l_col_ptrs: Vec<usize>,
    pub l_row_indices: Vec<usize>,
    pub l_values: Vec<f64>,
    /// U factor stored row-by-row (upper triangular).
    pub u_row_ptrs: Vec<usize>,
    pub u_col_indices: Vec<usize>,
    pub u_values: Vec<f64>,
    /// Row permutation.
    pub row_perm: Permutation,
    /// Column permutation.
    pub col_perm: Permutation,
    /// Growth factor estimate for stability.
    pub growth_factor: f64,
    /// Maximum entry in original matrix.
    pub max_orig_entry: f64,
    /// Number of fill-in entries.
    pub fill_in: usize,
}

impl LuFactorization {
    /// Create an LU factorization of a dense matrix (for basis of size m).
    pub fn factorize(matrix: &[Vec<f64>]) -> Result<Self, BasisError> {
        let n = matrix.len();
        if n == 0 {
            return Ok(Self::empty());
        }

        // Work on a copy
        let mut work: Vec<Vec<f64>> = matrix.to_vec();
        let mut row_perm = Permutation::identity(n);
        let mut col_perm = Permutation::identity(n);
        let mut max_orig = 0.0f64;
        for row in &work {
            for &v in row {
                max_orig = max_orig.max(v.abs());
            }
        }
        if max_orig < ZERO_TOL {
            return Err(BasisError::Singular);
        }

        // L stored as dense lower triangular
        let mut l_dense = vec![vec![0.0; n]; n];
        for i in 0..n {
            l_dense[i][i] = 1.0;
        }

        let mut max_u_entry = max_orig;

        // Gaussian elimination with partial pivoting
        for k in 0..n {
            // Find pivot: largest absolute value in column k, rows k..n
            let mut best_row = k;
            let mut best_val = work[k][k].abs();
            for i in (k + 1)..n {
                let v = work[i][k].abs();
                if v > best_val {
                    best_val = v;
                    best_row = i;
                }
            }

            // Threshold pivoting: also consider column selection
            if best_val < ZERO_TOL {
                // Try to find a nonzero in the remaining submatrix
                let mut found = false;
                for j in (k + 1)..n {
                    for i in k..n {
                        if work[i][j].abs() > ZERO_TOL {
                            // Swap columns k and j
                            for row in &mut work {
                                row.swap(k, j);
                            }
                            col_perm.swap(k, j);
                            // Now re-search for best row
                            best_row = k;
                            best_val = work[k][k].abs();
                            for ii in (k + 1)..n {
                                let v = work[ii][k].abs();
                                if v > best_val {
                                    best_val = v;
                                    best_row = ii;
                                }
                            }
                            found = true;
                            break;
                        }
                    }
                    if found {
                        break;
                    }
                }
                if !found {
                    return Err(BasisError::Singular);
                }
            }

            // Swap rows k and best_row
            if best_row != k {
                work.swap(k, best_row);
                row_perm.swap(k, best_row);
                // Also swap L entries already computed
                for j in 0..k {
                    let tmp = l_dense[k][j];
                    l_dense[k][j] = l_dense[best_row][j];
                    l_dense[best_row][j] = tmp;
                }
            }

            let pivot = work[k][k];
            if pivot.abs() < ZERO_TOL {
                return Err(BasisError::Singular);
            }

            // Eliminate below
            for i in (k + 1)..n {
                let multiplier = work[i][k] / pivot;
                l_dense[i][k] = multiplier;
                work[i][k] = 0.0;
                for j in (k + 1)..n {
                    work[i][j] -= multiplier * work[k][j];
                    max_u_entry = max_u_entry.max(work[i][j].abs());
                }
            }
        }

        // Convert L to sparse column format
        let mut l_col_ptrs = vec![0usize; n + 1];
        let mut l_row_indices = Vec::new();
        let mut l_values = Vec::new();
        for j in 0..n {
            for i in (j + 1)..n {
                if l_dense[i][j].abs() > ZERO_TOL {
                    l_row_indices.push(i);
                    l_values.push(l_dense[i][j]);
                }
            }
            l_col_ptrs[j + 1] = l_row_indices.len();
        }

        // Convert U to sparse row format
        let mut u_row_ptrs = vec![0usize; n + 1];
        let mut u_col_indices = Vec::new();
        let mut u_values = Vec::new();
        for i in 0..n {
            for j in i..n {
                if work[i][j].abs() > ZERO_TOL {
                    u_col_indices.push(j);
                    u_values.push(work[i][j]);
                }
            }
            u_row_ptrs[i + 1] = u_col_indices.len();
        }

        let growth = if max_orig > ZERO_TOL {
            max_u_entry / max_orig
        } else {
            1.0
        };

        let fill = l_row_indices.len() + u_col_indices.len();

        Ok(Self {
            n,
            l_col_ptrs,
            l_row_indices,
            l_values,
            u_row_ptrs,
            u_col_indices,
            u_values,
            row_perm,
            col_perm,
            growth_factor: growth,
            max_orig_entry: max_orig,
            fill_in: fill,
        })
    }

    fn empty() -> Self {
        Self {
            n: 0,
            l_col_ptrs: vec![0],
            l_row_indices: Vec::new(),
            l_values: Vec::new(),
            u_row_ptrs: vec![0],
            u_col_indices: Vec::new(),
            u_values: Vec::new(),
            row_perm: Permutation::identity(0),
            col_perm: Permutation::identity(0),
            growth_factor: 1.0,
            max_orig_entry: 0.0,
            fill_in: 0,
        }
    }

    /// Solve Bx = b (forward then backward substitution).
    pub fn solve(&self, rhs: &[f64]) -> Vec<f64> {
        let n = self.n;
        if n == 0 {
            return Vec::new();
        }
        // Apply row permutation
        let pb = self.row_perm.apply(rhs);

        // Forward substitution: Ly = Pb
        let mut y = pb;
        for j in 0..n {
            let start = self.l_col_ptrs[j];
            let end = self.l_col_ptrs[j + 1];
            for idx in start..end {
                let i = self.l_row_indices[idx];
                let l_val = self.l_values[idx];
                y[i] -= l_val * y[j];
            }
        }

        // Backward substitution: Uz = y
        let mut z = y;
        for i in (0..n).rev() {
            let start = self.u_row_ptrs[i];
            let end = self.u_row_ptrs[i + 1];
            // Find diagonal
            let mut diag = 0.0;
            let mut sum = 0.0;
            for idx in start..end {
                let j = self.u_col_indices[idx];
                let u_val = self.u_values[idx];
                if j == i {
                    diag = u_val;
                } else if j > i {
                    sum += u_val * z[j];
                }
            }
            if diag.abs() > ZERO_TOL {
                z[i] = (z[i] - sum) / diag;
            } else {
                z[i] = 0.0;
            }
        }

        // Apply column permutation inverse
        self.col_perm.apply_inverse(&z)
    }

    /// Solve B^T x = b (transpose system).
    pub fn solve_transpose(&self, rhs: &[f64]) -> Vec<f64> {
        let n = self.n;
        if n == 0 {
            return Vec::new();
        }

        // Apply column permutation
        let pb = self.col_perm.apply(rhs);

        // Forward substitution with U^T: U^T w = P_c * b
        let mut w = pb;
        for i in 0..n {
            let start = self.u_row_ptrs[i];
            let end = self.u_row_ptrs[i + 1];
            let mut diag = 0.0;
            for idx in start..end {
                let j = self.u_col_indices[idx];
                let u_val = self.u_values[idx];
                if j == i {
                    diag = u_val;
                }
            }
            if diag.abs() > ZERO_TOL {
                w[i] /= diag;
            }
            for idx in start..end {
                let j = self.u_col_indices[idx];
                let u_val = self.u_values[idx];
                if j > i {
                    w[j] -= u_val * w[i];
                }
            }
        }

        // Backward substitution with L^T
        for j in (0..n).rev() {
            let start = self.l_col_ptrs[j];
            let end = self.l_col_ptrs[j + 1];
            for idx in start..end {
                let i = self.l_row_indices[idx];
                let l_val = self.l_values[idx];
                w[j] -= l_val * w[i];
            }
        }

        // Apply row permutation inverse
        self.row_perm.apply_inverse(&w)
    }

    /// Check numerical stability: max |LU - PAQ| / max|A|.
    pub fn is_stable(&self) -> bool {
        self.growth_factor < STABILITY_THRESHOLD
    }
}

/// Basis representation for the simplex method.
#[derive(Debug, Clone)]
pub struct Basis {
    /// Size of the basis (number of basic variables = number of rows).
    pub m: usize,
    /// Indices of basic variables (column indices in the augmented matrix).
    pub basic_vars: Vec<usize>,
    /// LU factorization of the current basis matrix.
    pub lu: Option<LuFactorization>,
    /// Eta file for product-form updates.
    pub eta_file: Vec<EtaVector>,
    /// Number of updates since last refactorization.
    pub update_count: usize,
    /// Maximum updates before refactorization.
    pub max_updates: usize,
    /// Whether the basis needs refactorization.
    pub needs_refactorization: bool,
}

impl Basis {
    /// Create a new basis with given basic variable indices.
    pub fn new(m: usize, basic_vars: Vec<usize>) -> Self {
        Self {
            m,
            basic_vars,
            lu: None,
            eta_file: Vec::new(),
            update_count: 0,
            max_updates: MAX_ETA_COUNT,
            needs_refactorization: true,
        }
    }

    /// Create an identity basis (slack variables are basic).
    pub fn identity(m: usize, first_slack: usize) -> Self {
        let basic_vars: Vec<usize> = (first_slack..(first_slack + m)).collect();
        Self {
            m,
            basic_vars,
            lu: None,
            eta_file: Vec::new(),
            update_count: 0,
            max_updates: MAX_ETA_COUNT,
            needs_refactorization: true,
        }
    }

    /// Factorize the basis matrix from a column extraction function.
    pub fn factorize<F>(&mut self, get_column: F) -> Result<(), BasisError>
    where
        F: Fn(usize) -> Vec<f64>,
    {
        let m = self.m;
        let mut matrix = vec![vec![0.0; m]; m];
        for (j, &var) in self.basic_vars.iter().enumerate() {
            let col = get_column(var);
            for (i, &v) in col.iter().enumerate().take(m) {
                matrix[i][j] = v;
            }
        }
        self.lu = Some(LuFactorization::factorize(&matrix)?);
        self.eta_file.clear();
        self.update_count = 0;
        self.needs_refactorization = false;
        Ok(())
    }

    /// Solve Bx = b using current factorization + eta file (FTRAN).
    ///
    /// The eta file stores matrices M_i where M_i = E_i^{-1} (the inverse
    /// of the elementary basis change matrix). After k updates:
    ///   B_k^{-1} = M_k * M_{k-1} * ... * M_1 * B_0^{-1}
    /// So x = M_k * ... * M_1 * LU_solve(b)
    /// Apply M_1 first, then M_2, ..., M_k (forward order).
    pub fn solve(&self, rhs: &[f64]) -> Result<Vec<f64>, BasisError> {
        let lu = self.lu.as_ref().ok_or(BasisError::NotFactorized)?;
        let mut x = lu.solve(rhs);
        for eta in &self.eta_file {
            apply_eta_forward(&mut x, eta);
        }
        Ok(x)
    }

    /// Solve B^T y = c using current factorization + eta file (BTRAN).
    ///
    /// B_k^{-T} = B_0^{-T} * M_1^T * ... * M_k^T
    /// So y = LU_T_solve(M_1^T * ... * M_k^T * c)
    /// Apply M_k^T first, then M_{k-1}^T, ..., M_1^T (reverse order),
    /// then LU^T solve.
    pub fn solve_transpose(&self, rhs: &[f64]) -> Result<Vec<f64>, BasisError> {
        let lu = self.lu.as_ref().ok_or(BasisError::NotFactorized)?;
        let mut w = rhs.to_vec();
        for eta in self.eta_file.iter().rev() {
            apply_eta_transpose(&mut w, eta);
        }
        Ok(lu.solve_transpose(&w))
    }

    /// Perform a basis update: swap leaving variable at position `leave_pos` with `enter_var`.
    /// `pivot_column` is the FTRAN'd entering column.
    pub fn update(
        &mut self,
        leave_pos: usize,
        enter_var: usize,
        pivot_column: &[f64],
    ) -> Result<(), BasisError> {
        let pivot_val = pivot_column[leave_pos];
        if pivot_val.abs() < ZERO_TOL {
            return Err(BasisError::NumericalInstability);
        }

        // Build eta vector: column leave_pos of the eta matrix
        let mut eta = EtaVector::new(leave_pos);
        for (i, &v) in pivot_column.iter().enumerate() {
            if i == leave_pos {
                eta.add_entry(i, 1.0 / pivot_val);
            } else if v.abs() > ZERO_TOL {
                eta.add_entry(i, -v / pivot_val);
            }
        }
        self.eta_file.push(eta);
        self.basic_vars[leave_pos] = enter_var;
        self.update_count += 1;

        if self.update_count >= self.max_updates {
            self.needs_refactorization = true;
        }

        Ok(())
    }

    /// Forrest-Tomlin update: more numerically stable basis update.
    pub fn forrest_tomlin_update(
        &mut self,
        leave_pos: usize,
        enter_var: usize,
        pivot_column: &[f64],
    ) -> Result<(), BasisError> {
        let pivot_val = pivot_column[leave_pos];
        if pivot_val.abs() < ZERO_TOL {
            return Err(BasisError::NumericalInstability);
        }

        // The Forrest-Tomlin update transforms the spike column into a
        // proper triangular form. We use a sequence of row operations.
        let m = self.m;
        let mut spike = pivot_column.to_vec();

        // Build the eta vector with special pivoting order
        let mut eta = EtaVector::new(leave_pos);

        // Process rows above and below the leaving position
        // First, handle rows != leave_pos
        for i in 0..m {
            if i == leave_pos {
                continue;
            }
            if spike[i].abs() > ZERO_TOL {
                eta.add_entry(i, -spike[i] / pivot_val);
            }
        }
        eta.add_entry(leave_pos, 1.0 / pivot_val);

        // Sort entries by index for better cache behavior
        let mut pairs: Vec<(usize, f64)> = eta
            .indices
            .iter()
            .copied()
            .zip(eta.values.iter().copied())
            .collect();
        pairs.sort_by_key(|&(i, _)| i);
        eta.indices = pairs.iter().map(|&(i, _)| i).collect();
        eta.values = pairs.iter().map(|&(_, v)| v).collect();

        self.eta_file.push(eta);
        self.basic_vars[leave_pos] = enter_var;
        self.update_count += 1;

        // Forrest-Tomlin can go longer before refactorization
        if self.update_count >= self.max_updates * 3 / 2 {
            self.needs_refactorization = true;
        }

        Ok(())
    }

    /// Check if refactorization is needed.
    pub fn should_refactorize(&self) -> bool {
        self.needs_refactorization
            || self.update_count >= self.max_updates
            || self.lu.as_ref().map_or(true, |lu| !lu.is_stable())
    }

    /// Get the diagonal elements of U (for condition estimation).
    pub fn u_diag(&self) -> Vec<f64> {
        let lu = match &self.lu {
            Some(lu) => lu,
            None => return Vec::new(),
        };
        let n = lu.n;
        let mut diag = vec![0.0; n];
        for i in 0..n {
            let start = lu.u_row_ptrs[i];
            let end = lu.u_row_ptrs[i + 1];
            for idx in start..end {
                if lu.u_col_indices[idx] == i {
                    diag[i] = lu.u_values[idx];
                    break;
                }
            }
        }
        diag
    }

    /// Estimate the condition number of the basis (1-norm based).
    pub fn condition_estimate(&self) -> f64 {
        let lu = match &self.lu {
            Some(lu) => lu,
            None => return f64::INFINITY,
        };
        let n = lu.n;
        if n == 0 {
            return 1.0;
        }

        // Estimate ||B^{-1}||_1 using a few BTRAN solves
        let mut max_norm = 0.0f64;
        for k in 0..n.min(5) {
            let mut e = vec![0.0; n];
            e[k] = 1.0;
            let col = match self.solve_transpose(&e) {
                Ok(c) => c,
                Err(_) => return f64::INFINITY,
            };
            let norm: f64 = col.iter().map(|x| x.abs()).sum();
            max_norm = max_norm.max(norm);
        }

        // ||B||_1 estimate
        let b_norm = lu.max_orig_entry * (n as f64).sqrt();

        max_norm * b_norm
    }

    /// Get a copy of the basic variable indices.
    pub fn basic_indices(&self) -> &[usize] {
        &self.basic_vars
    }

    /// Check if a variable is basic.
    pub fn is_basic(&self, var: usize) -> bool {
        self.basic_vars.contains(&var)
    }

    /// Find the position of a basic variable.
    pub fn position_of(&self, var: usize) -> Option<usize> {
        self.basic_vars.iter().position(|&v| v == var)
    }
}

/// Apply the inverse of an eta matrix to a vector.
fn apply_eta_inverse(v: &mut [f64], eta: &EtaVector) {
    // Compute M^{-1} * v where M has column `col` replaced by eta entries.
    // M^{-1}[col] = v[col] / eta[col,col]
    // M^{-1}[i] = v[i] - eta[i,col] * (v[col]/eta[col,col])  for i != col
    let pivot_idx = eta.col;
    let mut pivot_entry = 1.0;
    for (&i, &eta_val) in eta.indices.iter().zip(eta.values.iter()) {
        if i == pivot_idx {
            pivot_entry = eta_val;
            break;
        }
    }

    if pivot_entry.abs() < ZERO_TOL {
        return;
    }

    let alpha = v[pivot_idx] / pivot_entry;
    for (&i, &eta_val) in eta.indices.iter().zip(eta.values.iter()) {
        if i != pivot_idx {
            v[i] -= eta_val * alpha;
        }
    }
    v[pivot_idx] = alpha;
}

/// Multiply v by the eta matrix M: v ← M * v.
/// M has column `col` replaced by eta entries; other columns are identity.
/// (M*v)[i] = v[i] + eta[i]*v[col]  for i != col (where eta[i] defaults to 0)
/// (M*v)[col] = eta[col]*v[col]
fn apply_eta_forward(v: &mut [f64], eta: &EtaVector) {
    let col = eta.col;
    let v_col = v[col];
    for (&i, &eta_val) in eta.indices.iter().zip(eta.values.iter()) {
        if i == col {
            v[col] = eta_val * v_col;
        } else {
            v[i] += eta_val * v_col;
        }
    }
}

/// Multiply v by the transpose of eta matrix M: v ← M^T * v.
/// (M^T v)[j] = v[j]                     for j != col
/// (M^T v)[col] = sum_i eta[i] * v[i]
fn apply_eta_transpose(v: &mut [f64], eta: &EtaVector) {
    let col = eta.col;
    let mut new_v_col = 0.0;
    for (&i, &eta_val) in eta.indices.iter().zip(eta.values.iter()) {
        new_v_col += eta_val * v[i];
    }
    v[col] = new_v_col;
}

/// Apply the transpose-inverse of an eta matrix to a vector.
fn apply_eta_transpose_inverse(v: &mut [f64], eta: &EtaVector) {
    // E^T has: E^T[j][i] = E[i][j]
    // For the column `col` of E: E[i][col] = eta[i]
    // E^T[col][i] = eta[i] for i in eta.indices
    // E^T[j][i] = delta(j,i) for i != col
    // (E^T v)[j] = sum_i E^T[j][i] * v[i]
    // For j != col: (E^T v)[j] = v[j] (from i=j, since E^T[j][j] = delta(j,j) = 1, and no col contribution since j != col => E^T[j][col] would need j==col)
    // Hmm, let me reconsider.
    // E^T[j][i] = E[i][j].
    // E[i][j] = delta(i,j) for j != col, and E[i][col] = eta[i] for j == col.
    // So E^T[j][i] = delta(i,j) for j != col, and E^T[col][i] = eta[i].
    // (E^T v)[j] = sum_i E^T[j][i] v[i]
    //   for j != col: = sum_i delta(i,j) v[i] = v[j]
    //   for j == col: = sum_i eta[i] v[i]
    // So E^T only modifies position col.
    // (E^{-T} v)[col] = (v[col] - sum_{i != col} eta[i]*v[i]) / eta[col]
    // (E^{-T} v)[j] = v[j] for j != col

    let pivot_idx = eta.col;
    let mut pivot_entry = 1.0;
    let mut off_diag_sum = 0.0;

    for (&i, &eta_val) in eta.indices.iter().zip(eta.values.iter()) {
        if i == pivot_idx {
            pivot_entry = eta_val;
        } else {
            off_diag_sum += eta_val * v[i];
        }
    }

    if pivot_entry.abs() > ZERO_TOL {
        v[pivot_idx] = (v[pivot_idx] - off_diag_sum) / pivot_entry;
    }
}

/// Errors in basis operations.
#[derive(Debug, Clone)]
pub enum BasisError {
    Singular,
    NotFactorized,
    NumericalInstability,
    InvalidDimension,
}

impl fmt::Display for BasisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BasisError::Singular => write!(f, "Basis is singular"),
            BasisError::NotFactorized => write!(f, "Basis not factorized"),
            BasisError::NumericalInstability => write!(f, "Numerical instability detected"),
            BasisError::InvalidDimension => write!(f, "Invalid dimension"),
        }
    }
}

impl std::error::Error for BasisError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_lu() {
        let matrix = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let lu = LuFactorization::factorize(&matrix).unwrap();
        let x = lu.solve(&[1.0, 2.0, 3.0]);
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
        assert!((x[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_simple_lu() {
        let matrix = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let lu = LuFactorization::factorize(&matrix).unwrap();
        // 2x + y = 5, x + 3y = 10 => x = 1, y = 3
        let x = lu.solve(&[5.0, 10.0]);
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_lu_with_pivoting() {
        let matrix = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let lu = LuFactorization::factorize(&matrix).unwrap();
        let x = lu.solve(&[3.0, 7.0]);
        assert!((x[0] - 7.0).abs() < 1e-10);
        assert!((x[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_singular_matrix() {
        let matrix = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        assert!(LuFactorization::factorize(&matrix).is_err());
    }

    #[test]
    fn test_lu_transpose_solve() {
        let matrix = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let lu = LuFactorization::factorize(&matrix).unwrap();
        // B^T y = c => [2 1; 1 3]^T y = [4; 7] => [2 1; 1 3] y = [4; 7]
        // Same as forward since this matrix is symmetric
        let y = lu.solve_transpose(&[4.0, 7.0]);
        let x = lu.solve(&[4.0, 7.0]);
        assert!((y[0] - x[0]).abs() < 1e-10);
        assert!((y[1] - x[1]).abs() < 1e-10);
    }

    #[test]
    fn test_basis_solve() {
        let mut basis = Basis::new(2, vec![0, 1]);
        let cols: Vec<Vec<f64>> = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        basis.factorize(|var| cols[var].clone()).unwrap();
        let x = basis.solve(&[5.0, 10.0]).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_basis_update() {
        let mut basis = Basis::new(2, vec![0, 1]);
        let cols: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        basis.factorize(|var| cols[var].clone()).unwrap();

        // Enter variable 2, leave position 0
        let pivot_col = basis.solve(&cols[2]).unwrap();
        basis.update(0, 2, &pivot_col).unwrap();
        assert_eq!(basis.basic_vars, vec![2, 1]);
    }

    #[test]
    fn test_permutation() {
        let mut p = Permutation::identity(3);
        p.swap(0, 2);
        let v = vec![1.0, 2.0, 3.0];
        let pv = p.apply(&v);
        assert!((pv[0] - 3.0).abs() < 1e-10);
        assert!((pv[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_condition_estimate() {
        let mut basis = Basis::new(2, vec![0, 1]);
        let cols: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        basis.factorize(|var| cols[var].clone()).unwrap();
        let cond = basis.condition_estimate();
        assert!(cond < 10.0);
    }

    #[test]
    fn test_3x3_lu() {
        let matrix = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 10.0],
        ];
        let lu = LuFactorization::factorize(&matrix).unwrap();
        let b = vec![14.0, 32.0, 50.0];
        let x = lu.solve(&b);
        // Check Ax = b
        for i in 0..3 {
            let row_sum: f64 = matrix[i].iter().zip(x.iter()).map(|(a, xi)| a * xi).sum();
            assert!(
                (row_sum - b[i]).abs() < 1e-8,
                "Row {} mismatch: {} vs {}",
                i,
                row_sum,
                b[i]
            );
        }
    }
}
