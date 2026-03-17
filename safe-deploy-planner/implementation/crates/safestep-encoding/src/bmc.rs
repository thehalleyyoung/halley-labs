//! Bounded Model Checking (BMC) formula encoder for deployment planning.
//!
//! Translates the multi-service deployment planning problem into SAT via
//! BMC-style unrolling: at each step exactly one service may change its
//! version, versions are monotonically non-decreasing, and every
//! intermediate state must satisfy the pairwise compatibility constraints.

use crate::formula::{Clause, Literal};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert an unsigned integer to a little-endian bit vector.
pub fn to_binary(value: usize, num_bits: usize) -> Vec<bool> {
    (0..num_bits).map(|b| (value >> b) & 1 == 1).collect()
}

/// Number of bits needed to represent values in `0..n`.
fn bits_needed(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut b = 0usize;
    let mut v = n - 1;
    while v > 0 {
        v >>= 1;
        b += 1;
    }
    b
}

// ---------------------------------------------------------------------------
// BmcEncoder
// ---------------------------------------------------------------------------

/// Main BMC encoder that maps the deployment planning problem into SAT
/// variables and clauses.
pub struct BmcEncoder {
    /// Number of services.
    pub num_services: usize,
    /// Number of versions available for each service.
    pub versions_per_service: Vec<usize>,
    /// Bits required to binary-encode each service's version.
    pub bits_per_service: Vec<usize>,
    /// Total state bits across all services.
    total_state_bits: usize,
}

impl BmcEncoder {
    /// Create a new BMC encoder.
    pub fn new(num_services: usize, versions_per_service: Vec<usize>) -> Self {
        assert_eq!(num_services, versions_per_service.len());
        let bits_per_service: Vec<usize> = versions_per_service
            .iter()
            .map(|&v| bits_needed(v))
            .collect();
        let total_state_bits: usize = bits_per_service.iter().sum();
        Self {
            num_services,
            versions_per_service,
            bits_per_service,
            total_state_bits,
        }
    }

    /// Variable index for state bit `bit` of `service` at time `step`.
    ///
    /// Layout: for each step we allocate `total_state_bits` state variables
    /// followed by `num_services` change indicator variables.
    pub fn state_var(&self, service: usize, step: usize, bit: usize) -> usize {
        let vars_per_step = self.total_state_bits + self.num_services;
        let base = 1 + step * vars_per_step; // 1-indexed SAT variables
        let offset: usize = self.bits_per_service[..service].iter().sum();
        base + offset + bit
    }

    /// Variable indicating that `service` changes at `step`.
    pub fn change_var(&self, service: usize, step: usize) -> usize {
        let vars_per_step = self.total_state_bits + self.num_services;
        let base = 1 + step * vars_per_step;
        base + self.total_state_bits + service
    }

    /// Total number of SAT variables needed for unrolling to `depth` steps.
    pub fn total_variables(&self, depth: usize) -> usize {
        let vars_per_step = self.total_state_bits + self.num_services;
        // Steps 0..=depth → depth+1 state snapshots, depth transitions
        (depth + 1) * self.total_state_bits + depth * self.num_services
            + vars_per_step // padding for auxiliary
    }

    /// Rough estimate of the number of clauses for the full encoding.
    pub fn total_clauses(&self, depth: usize) -> usize {
        let init = self.total_state_bits;
        let target = self.total_state_bits;
        let transition_per_step =
            self.num_services * self.num_services + self.total_state_bits * 4;
        init + target + depth * transition_per_step
    }

    /// Clauses that fix the initial state to `state`.
    /// `state[i]` is the initial version of service i.
    pub fn encode_initial_state(&self, state: &[usize]) -> Vec<Clause> {
        assert_eq!(state.len(), self.num_services);
        let mut clauses = Vec::new();
        for (svc, &ver) in state.iter().enumerate() {
            let bits = to_binary(ver, self.bits_per_service[svc]);
            for (b, &val) in bits.iter().enumerate() {
                let var = self.state_var(svc, 0, b) as Literal;
                clauses.push(if val { vec![var] } else { vec![-var] });
            }
        }
        clauses
    }

    /// Clauses that require the state at `step` to equal `state`.
    pub fn encode_target_state(&self, state: &[usize], step: usize) -> Vec<Clause> {
        assert_eq!(state.len(), self.num_services);
        let mut clauses = Vec::new();
        for (svc, &ver) in state.iter().enumerate() {
            let bits = to_binary(ver, self.bits_per_service[svc]);
            for (b, &val) in bits.iter().enumerate() {
                let var = self.state_var(svc, step, b) as Literal;
                clauses.push(if val { vec![var] } else { vec![-var] });
            }
        }
        clauses
    }

    /// Transition relation for time step `step` (from step to step+1).
    ///
    /// - At most one service changes (via `change_var`).
    /// - If a service does not change, its bits are copied.
    /// - If a service changes, its version increments by exactly 1.
    pub fn encode_transition(&self, step: usize) -> Vec<Clause> {
        let mut clauses = Vec::new();

        // --- at-most-one change ---
        let change_lits: Vec<Literal> = (0..self.num_services)
            .map(|s| self.change_var(s, step) as Literal)
            .collect();
        clauses.extend(pairwise_at_most_one(&change_lits));

        // --- at-least-one change (progress) ---
        clauses.push(change_lits.clone());

        // --- frame axiom: unchanged services keep their bits ---
        for svc in 0..self.num_services {
            let c = self.change_var(svc, step) as Literal;
            for b in 0..self.bits_per_service[svc] {
                let curr = self.state_var(svc, step, b) as Literal;
                let next = self.state_var(svc, step + 1, b) as Literal;
                // ¬change → (curr ↔ next)
                clauses.push(vec![c, -curr, next]);
                clauses.push(vec![c, curr, -next]);
            }
        }

        // --- change semantics: version increments by 1 ---
        for svc in 0..self.num_services {
            let nb = self.bits_per_service[svc];
            let c = self.change_var(svc, step) as Literal;
            let curr_bits: Vec<Literal> = (0..nb)
                .map(|b| self.state_var(svc, step, b) as Literal)
                .collect();
            let next_bits: Vec<Literal> = (0..nb)
                .map(|b| self.state_var(svc, step + 1, b) as Literal)
                .collect();
            // Encode: change → next = curr + 1 (ripple carry)
            let inc_clauses = encode_increment(c, &curr_bits, &next_bits);
            clauses.extend(inc_clauses);
        }

        clauses
    }

    /// Safety constraints at `step`.
    ///
    /// `safe_pairs` contains (svc_a, svc_b, compatible_version_pairs).
    /// For each pair of services we forbid all incompatible combinations.
    pub fn encode_safety_at_step(
        &self,
        step: usize,
        safe_pairs: &[(usize, usize, Vec<(usize, usize)>)],
    ) -> Vec<Clause> {
        let mut clauses = Vec::new();
        for &(svc_a, svc_b, ref compat) in safe_pairs {
            let max_a = self.versions_per_service[svc_a];
            let max_b = self.versions_per_service[svc_b];
            let compat_set: std::collections::HashSet<(usize, usize)> =
                compat.iter().copied().collect();

            for va in 0..max_a {
                for vb in 0..max_b {
                    if compat_set.contains(&(va, vb)) {
                        continue;
                    }
                    // Forbid (va, vb): at least one of the bits must differ
                    let bits_a = to_binary(va, self.bits_per_service[svc_a]);
                    let bits_b = to_binary(vb, self.bits_per_service[svc_b]);
                    let mut clause = Vec::new();
                    for (b, &val) in bits_a.iter().enumerate() {
                        let v = self.state_var(svc_a, step, b) as Literal;
                        clause.push(if val { -v } else { v });
                    }
                    for (b, &val) in bits_b.iter().enumerate() {
                        let v = self.state_var(svc_b, step, b) as Literal;
                        clause.push(if val { -v } else { v });
                    }
                    clauses.push(clause);
                }
            }
        }
        clauses
    }
}

/// Pairwise at-most-one encoding: for every pair (i, j), ¬i ∨ ¬j.
fn pairwise_at_most_one(lits: &[Literal]) -> Vec<Clause> {
    let mut clauses = Vec::new();
    for i in 0..lits.len() {
        for j in (i + 1)..lits.len() {
            clauses.push(vec![-lits[i], -lits[j]]);
        }
    }
    clauses
}

/// Encode: if `guard` is true then `next = curr + 1` (binary ripple carry).
///
/// Bits are little-endian. We use implications guarded by `guard`.
fn encode_increment(guard: Literal, curr: &[Literal], next: &[Literal]) -> Vec<Clause> {
    assert_eq!(curr.len(), next.len());
    let n = curr.len();
    if n == 0 {
        return Vec::new();
    }
    let mut clauses = Vec::new();

    // bit 0: next[0] = ¬curr[0], carry = curr[0]
    // guard → (next[0] ↔ ¬curr[0])
    clauses.push(vec![-guard, -curr[0], -next[0]]);
    clauses.push(vec![-guard, curr[0], next[0]]);

    if n == 1 {
        return clauses;
    }

    // For simplicity we encode the +1 via implications on each bit pattern.
    // For each possible current value v, if guard ∧ curr==v → next==v+1.
    let max_val = 1usize << n;
    for v in 0..max_val.saturating_sub(1) {
        let v1 = v + 1;
        let curr_bits = to_binary(v, n);
        let next_bits = to_binary(v1, n);

        // Build a blocking clause: guard ∧ (curr == v) → (next == v+1)
        // Equivalently: ¬guard ∨ ¬(curr==v) ∨ (next==v+1)
        // We split into per-next-bit implications.
        for (b, &nb) in next_bits.iter().enumerate() {
            let mut clause: Vec<Literal> = vec![-guard];
            for (cb, &cv) in curr_bits.iter().enumerate() {
                let var = curr[cb];
                clause.push(if cv { -var } else { var });
            }
            let nvar = next[b];
            clause.push(if nb { nvar } else { -nvar });
            clauses.push(clause);
        }
    }
    clauses
}

// ---------------------------------------------------------------------------
// BmcUnrolling
// ---------------------------------------------------------------------------

/// Incremental BMC unrolling manager.
pub struct BmcUnrolling {
    encoder: BmcEncoder,
    /// Clauses generated for each depth.
    clauses_by_depth: HashMap<usize, Vec<Clause>>,
    /// Current maximum unrolled depth.
    current_depth: usize,
}

impl BmcUnrolling {
    /// Create a new unrolling manager wrapping the given encoder.
    pub fn new(encoder: BmcEncoder) -> Self {
        Self {
            encoder,
            clauses_by_depth: HashMap::new(),
            current_depth: 0,
        }
    }

    /// Produce clauses for step `k` (the transition from k to k+1).
    pub fn unroll_to_depth(&mut self, k: usize) {
        for d in self.current_depth..=k {
            if !self.clauses_by_depth.contains_key(&d) {
                let clauses = self.encoder.encode_transition(d);
                self.clauses_by_depth.insert(d, clauses);
            }
        }
        if k > self.current_depth {
            self.current_depth = k;
        }
    }

    /// Get the clauses that were generated for depth `k`.
    pub fn clauses_at_depth(&self, k: usize) -> &[Clause] {
        self.clauses_by_depth
            .get(&k)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Reference to the underlying encoder.
    pub fn encoder(&self) -> &BmcEncoder {
        &self.encoder
    }

    /// Collect all clauses up to the current depth.
    pub fn all_clauses(&self) -> Vec<Clause> {
        let mut out = Vec::new();
        for d in 0..=self.current_depth {
            if let Some(c) = self.clauses_by_depth.get(&d) {
                out.extend(c.iter().cloned());
            }
        }
        out
    }

    /// Extract a plan from a satisfying assignment.
    ///
    /// Returns a sequence of `(service, from_version, to_version)` triples.
    pub fn extract_plan(
        assignment: &[bool],
        depth: usize,
        encoder: &BmcEncoder,
    ) -> Vec<(usize, usize, usize)> {
        let mut plan = Vec::new();
        for step in 0..depth {
            for svc in 0..encoder.num_services {
                let cv = Self::read_version(assignment, encoder, svc, step);
                let nv = Self::read_version(assignment, encoder, svc, step + 1);
                if nv != cv {
                    plan.push((svc, cv, nv));
                }
            }
        }
        plan
    }

    /// Read the version of `service` at `step` from the assignment.
    fn read_version(
        assignment: &[bool],
        encoder: &BmcEncoder,
        service: usize,
        step: usize,
    ) -> usize {
        let nb = encoder.bits_per_service[service];
        let mut val = 0usize;
        for b in 0..nb {
            let var_idx = encoder.state_var(service, step, b);
            if var_idx < assignment.len() && assignment[var_idx] {
                val |= 1 << b;
            }
        }
        val
    }
}

// ---------------------------------------------------------------------------
// MonotoneEncoder
// ---------------------------------------------------------------------------

/// Encodes the monotonicity constraint: version[i][k+1] >= version[i][k].
///
/// Uses a bit-by-bit MSB-to-LSB comparison.
pub struct MonotoneEncoder;

impl MonotoneEncoder {
    /// Produce clauses enforcing that the value at step+1 >= value at step.
    ///
    /// `service` and `step` determine which SAT variables to reference via
    /// the provided variable-index function `var_fn(service, step, bit)`.
    pub fn encode_monotone<F>(
        var_fn: &F,
        service: usize,
        step: usize,
        num_bits: usize,
    ) -> Vec<Clause>
    where
        F: Fn(usize, usize, usize) -> usize,
    {
        let curr: Vec<Literal> = (0..num_bits)
            .rev()
            .map(|b| var_fn(service, step, b) as Literal)
            .collect();
        let next: Vec<Literal> = (0..num_bits)
            .rev()
            .map(|b| var_fn(service, step + 1, b) as Literal)
            .collect();
        Self::encode_geq(&next, &curr)
    }

    /// Clauses for a >= b (MSB-first bit vectors).
    ///
    /// At each bit position from MSB to LSB we track whether the prefix so
    /// far is strictly greater or equal. If at any position a has a 0 while
    /// b has a 1, and the prefix was merely equal, the constraint is violated.
    fn encode_geq(a: &[Literal], b: &[Literal]) -> Vec<Clause> {
        assert_eq!(a.len(), b.len());
        let n = a.len();
        if n == 0 {
            return Vec::new();
        }
        let mut clauses = Vec::new();

        // For each bit position i (MSB-first), if all higher bits are equal
        // then a[i] >= b[i].
        // We encode: ∀prefix that is bit-wise equal ⇒ ¬(a[i]=0 ∧ b[i]=1).
        // This uses O(n²) clauses in the worst case but is straightforward.
        for i in 0..n {
            // Build: (some higher bit has a>b) ∨ ¬(a[i]=0 ∧ b[i]=1)
            // ≡ (some higher bit has a>b) ∨ a[i] ∨ ¬b[i]
            let mut clause: Vec<Literal> = Vec::new();
            for j in 0..i {
                // a[j] > b[j] means a[j]=1 ∧ b[j]=0  ⇒  a[j] ∧ ¬b[j]
                // But in a clause we need a disjunction of these pairs...
                // We'll need auxiliary variables for a clean encoding.
                // For simplicity, use the "bit-chain" method:
                // We just add a[j] ∧ ¬b[j] as a pair of literals — but that
                // doesn't work in a single clause. Instead, we'll use
                // a standard technique.
                let _ = j; // handled below
            }
            // Use the direct chain encoding:
            // If all higher bits are equal (a[j]==b[j] for j<i), then a[i] ∨ ¬b[i].
            // ∀j<i: (a[j]≠b[j]) is a disjunctive escape.
            // We encode: ∨_{j<i} (a[j] ∧ ¬b[j]) ∨ ∨_{j<i} (¬a[j] ∧ b[j])^{complement} ...
            // A simpler correct encoding: for each bit, add a clause that
            // forbids the pattern "equal prefix, then a<b at this bit".
            // This requires: for every subset of equal-prefix assignments.
            // Instead, let's use the O(n) clause approach with auxiliary "decided" vars.
            clause.clear();
        }
        clauses.clear();

        // Cleaner approach: auxiliary variables d[i] meaning "decided: a>b
        // has been determined at some bit j<=i".
        // d[0]: a[0]>b[0]  ⇒  a[0] ∧ ¬b[0]
        // constraint: ¬d[i] → a[i] ∨ ¬b[i]  (if not yet decided, can't have a<b here)
        //             d[i] ↔ d[i-1] ∨ (a[i-1]==a and b[i-1]==... )  — complex.
        // Actually the simplest correct O(n) encoding uses carry bits.
        // Let g[i] = "a[0..i] > b[0..i]" (strictly greater on the prefix).
        // Then: g[0] = a[0] ∧ ¬b[0]
        //       g[i] = g[i-1] ∨ (eq[i-1] ∧ a[i] ∧ ¬b[i])
        // where eq[i] = (a[0..i] == b[0..i]).
        //       eq[0] = (a[0] ↔ b[0])
        //       eq[i] = eq[i-1] ∧ (a[i] ↔ b[i])
        // Final constraint: g[n-1] ∨ eq[n-1]  (a >= b).

        // We'll use a simulated auxiliary variable counter starting from a
        // large offset to avoid collisions with real variables.
        // Since this is a static method we generate variables offset from a
        // deterministic base derived from the literal values.
        let base_var: i32 = a.iter().chain(b.iter()).map(|l| l.abs()).max().unwrap_or(0) + 1;
        let mut next_aux = base_var;
        let mut alloc = || {
            let v = next_aux;
            next_aux += 1;
            v
        };

        if n == 1 {
            // a[0] >= b[0] ⇔ a[0] ∨ ¬b[0]
            clauses.push(vec![a[0], -b[0]]);
            return clauses;
        }

        // eq[i]: prefix 0..=i is equal
        let mut eq: Vec<i32> = Vec::with_capacity(n);
        // gt[i]: prefix 0..=i has a > b
        let mut gt: Vec<i32> = Vec::with_capacity(n);

        for _ in 0..n {
            eq.push(alloc());
            gt.push(alloc());
        }

        // --- bit 0 ---
        // eq[0] ↔ (a[0] ↔ b[0])
        // eq[0] ↔ (a[0]∧b[0]) ∨ (¬a[0]∧¬b[0])
        clauses.push(vec![-eq[0], a[0], -b[0]]);     // eq[0] → (a[0] ∨ ¬b[0])
        clauses.push(vec![-eq[0], -a[0], b[0]]);      // eq[0] → (¬a[0] ∨ b[0])
        clauses.push(vec![eq[0], a[0], b[0]]);         // ¬eq[0] → (a[0] ∨ b[0])
        clauses.push(vec![eq[0], -a[0], -b[0]]);       // ¬eq[0] → (¬a[0] ∨ ¬b[0])

        // gt[0] ↔ a[0] ∧ ¬b[0]
        clauses.push(vec![-gt[0], a[0]]);
        clauses.push(vec![-gt[0], -b[0]]);
        clauses.push(vec![gt[0], -a[0], b[0]]);

        // --- bits 1..n-1 ---
        for i in 1..n {
            // eq[i] ↔ eq[i-1] ∧ (a[i] ↔ b[i])
            let same_i = alloc();
            // same_i ↔ (a[i] ↔ b[i])
            clauses.push(vec![-same_i, a[i], -b[i]]);
            clauses.push(vec![-same_i, -a[i], b[i]]);
            clauses.push(vec![same_i, a[i], b[i]]);
            clauses.push(vec![same_i, -a[i], -b[i]]);
            // eq[i] ↔ eq[i-1] ∧ same_i
            clauses.push(vec![-eq[i], eq[i - 1]]);
            clauses.push(vec![-eq[i], same_i]);
            clauses.push(vec![eq[i], -eq[i - 1], -same_i]);

            // gt[i] ↔ gt[i-1] ∨ (eq[i-1] ∧ a[i] ∧ ¬b[i])
            let new_gt_i = alloc();
            // new_gt_i ↔ eq[i-1] ∧ a[i] ∧ ¬b[i]
            clauses.push(vec![-new_gt_i, eq[i - 1]]);
            clauses.push(vec![-new_gt_i, a[i]]);
            clauses.push(vec![-new_gt_i, -b[i]]);
            clauses.push(vec![new_gt_i, -eq[i - 1], -a[i], b[i]]);
            // gt[i] ↔ gt[i-1] ∨ new_gt_i
            clauses.push(vec![-gt[i], gt[i - 1], new_gt_i]);
            clauses.push(vec![gt[i], -gt[i - 1]]);
            clauses.push(vec![gt[i], -new_gt_i]);
        }

        // Final: a >= b ↔ gt[n-1] ∨ eq[n-1]
        clauses.push(vec![gt[n - 1], eq[n - 1]]);

        clauses
    }

    /// Encode monotonicity for all services across a range of steps.
    pub fn encode_all_monotone<F>(
        var_fn: &F,
        num_services: usize,
        bits_per_service: &[usize],
        steps: usize,
    ) -> Vec<Clause>
    where
        F: Fn(usize, usize, usize) -> usize,
    {
        let mut clauses = Vec::new();
        for step in 0..steps {
            for svc in 0..num_services {
                clauses.extend(Self::encode_monotone(
                    var_fn,
                    svc,
                    step,
                    bits_per_service[svc],
                ));
            }
        }
        clauses
    }
}

// ---------------------------------------------------------------------------
// StepEncoder
// ---------------------------------------------------------------------------

/// Encodes at-most-one-service-changes-per-step constraints.
pub struct StepEncoder;

impl StepEncoder {
    /// Pairwise encoding: for every pair of services, at most one may change.
    pub fn encode_at_most_one_change(num_services: usize, step: usize, encoder: &BmcEncoder) -> Vec<Clause> {
        let lits: Vec<Literal> = (0..num_services)
            .map(|s| encoder.change_var(s, step) as Literal)
            .collect();
        pairwise_at_most_one(&lits)
    }

    /// Ladder (sequential) encoding for at-most-one, more compact for large n.
    ///
    /// Introduces n-1 auxiliary variables y_1..y_{n-1}.
    /// Semantics: y_i = "at least one of x_1..x_i is true".
    pub fn encode_at_most_one_ladder(
        lits: &[Literal],
        next_var: &mut i32,
    ) -> Vec<Clause> {
        let n = lits.len();
        if n <= 1 {
            return Vec::new();
        }
        if n <= 4 {
            return pairwise_at_most_one(lits);
        }

        let mut clauses = Vec::new();
        let mut y: Vec<i32> = Vec::with_capacity(n - 1);
        for _ in 0..(n - 1) {
            y.push(*next_var);
            *next_var += 1;
        }

        // x_0 → y_0
        clauses.push(vec![-lits[0], y[0]]);

        for i in 1..(n - 1) {
            // x_i → y_i
            clauses.push(vec![-lits[i], y[i]]);
            // y_{i-1} → y_i  (monotone ladder)
            clauses.push(vec![-y[i - 1], y[i]]);
            // x_i → ¬y_{i-1}  (if x_i true, nothing before was true)
            clauses.push(vec![-lits[i], -y[i - 1]]);
        }

        // x_{n-1} → ¬y_{n-2}
        clauses.push(vec![-lits[n - 1], -y[n - 2]]);

        clauses
    }

    /// Encode at-most-one for a step using the best strategy given the size.
    pub fn encode_best(num_services: usize, step: usize, encoder: &BmcEncoder) -> Vec<Clause> {
        if num_services <= 6 {
            Self::encode_at_most_one_change(num_services, step, encoder)
        } else {
            let lits: Vec<Literal> = (0..num_services)
                .map(|s| encoder.change_var(s, step) as Literal)
                .collect();
            let mut nv = encoder.total_variables(step + 2) as i32 + 1;
            Self::encode_at_most_one_ladder(&lits, &mut nv)
        }
    }
}

// ---------------------------------------------------------------------------
// CompletenessChecker
// ---------------------------------------------------------------------------

/// Computes the completeness bound k* = Σ(target[i] − start[i]).
///
/// Under monotone single-step-increment semantics, exactly k* steps are
/// required to go from `start` to `target`.
pub struct CompletenessChecker;

impl CompletenessChecker {
    /// Compute the exact number of steps required.
    pub fn completeness_bound(start: &[usize], target: &[usize]) -> usize {
        assert_eq!(start.len(), target.len());
        start
            .iter()
            .zip(target.iter())
            .map(|(&s, &t)| t.saturating_sub(s))
            .sum()
    }

    /// Check whether the given unrolling depth is sufficient to reach
    /// the target from start.
    pub fn is_complete(depth: usize, start: &[usize], target: &[usize]) -> bool {
        depth >= Self::completeness_bound(start, target)
    }

    /// Minimum number of steps to update service `svc` from `from` to `to`.
    pub fn service_steps(from: usize, to: usize) -> usize {
        to.saturating_sub(from)
    }

    /// Per-service breakdown of required steps.
    pub fn per_service_steps(start: &[usize], target: &[usize]) -> Vec<usize> {
        start
            .iter()
            .zip(target.iter())
            .map(|(&s, &t)| t.saturating_sub(s))
            .collect()
    }

    /// Validate that all target versions are >= start versions.
    pub fn is_valid_target(start: &[usize], target: &[usize]) -> bool {
        start.len() == target.len()
            && start.iter().zip(target.iter()).all(|(&s, &t)| t >= s)
    }
}

// ---------------------------------------------------------------------------
// BmcResult – helpers for working with solver output
// ---------------------------------------------------------------------------

/// A decoded deployment plan extracted from a BMC solution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeploymentPlan {
    /// Ordered list of (service_index, from_version, to_version).
    pub steps: Vec<(usize, usize, usize)>,
}

impl DeploymentPlan {
    /// Build from a raw step list.
    pub fn new(steps: Vec<(usize, usize, usize)>) -> Self {
        Self { steps }
    }

    /// Number of deployment steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether the plan is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Verify that the plan transforms `start` into `target`.
    pub fn verify(&self, start: &[usize], target: &[usize]) -> bool {
        let mut state = start.to_vec();
        for &(svc, from, to) in &self.steps {
            if svc >= state.len() || state[svc] != from || to != from + 1 {
                return false;
            }
            state[svc] = to;
        }
        state == target
    }

    /// Services touched by this plan, in order.
    pub fn services_in_order(&self) -> Vec<usize> {
        self.steps.iter().map(|&(s, _, _)| s).collect()
    }

    /// Intermediate states along the plan.
    pub fn intermediate_states(&self, start: &[usize]) -> Vec<Vec<usize>> {
        let mut states = vec![start.to_vec()];
        let mut current = start.to_vec();
        for &(svc, _from, to) in &self.steps {
            current[svc] = to;
            states.push(current.clone());
        }
        states
    }
}

// ---------------------------------------------------------------------------
// VariableAllocator
// ---------------------------------------------------------------------------

/// Simple monotonic variable allocator for SAT variable indices.
pub struct VariableAllocator {
    next: usize,
}

impl VariableAllocator {
    /// Create a new allocator starting from the given index.
    pub fn new(start: usize) -> Self {
        Self { next: start }
    }

    /// Allocate a single fresh variable.
    pub fn fresh(&mut self) -> usize {
        let v = self.next;
        self.next += 1;
        v
    }

    /// Allocate `n` contiguous variables.
    pub fn fresh_block(&mut self, n: usize) -> Vec<usize> {
        let start = self.next;
        self.next += n;
        (start..start + n).collect()
    }

    /// Current next variable (peek without allocating).
    pub fn peek(&self) -> usize {
        self.next
    }
}

// ---------------------------------------------------------------------------
// ClauseStats
// ---------------------------------------------------------------------------

/// Statistics about a clause set.
#[derive(Debug, Clone, Default)]
pub struct ClauseStats {
    pub num_clauses: usize,
    pub num_unit: usize,
    pub num_binary: usize,
    pub num_ternary: usize,
    pub num_long: usize,
    pub max_length: usize,
    pub total_literals: usize,
}

impl ClauseStats {
    /// Compute statistics from a slice of clauses.
    pub fn from_clauses(clauses: &[Clause]) -> Self {
        let mut s = Self::default();
        s.num_clauses = clauses.len();
        for c in clauses {
            s.total_literals += c.len();
            if c.len() > s.max_length {
                s.max_length = c.len();
            }
            match c.len() {
                1 => s.num_unit += 1,
                2 => s.num_binary += 1,
                3 => s.num_ternary += 1,
                _ => s.num_long += 1,
            }
        }
        s
    }

    /// Average clause length (0.0 if empty).
    pub fn avg_length(&self) -> f64 {
        if self.num_clauses == 0 {
            0.0
        } else {
            self.total_literals as f64 / self.num_clauses as f64
        }
    }

    /// Number of distinct variables referenced by the clauses.
    pub fn num_variables(clauses: &[Clause]) -> usize {
        let vars: std::collections::HashSet<i32> = clauses
            .iter()
            .flat_map(|c| c.iter().map(|l| l.abs()))
            .collect();
        vars.len()
    }
}

// ---------------------------------------------------------------------------
// DimacsPrinter
// ---------------------------------------------------------------------------

/// Format clauses in DIMACS CNF format for external solver consumption.
pub struct DimacsPrinter;

impl DimacsPrinter {
    /// Render clauses as a DIMACS-format string.
    pub fn to_dimacs(clauses: &[Clause], num_vars: usize) -> String {
        let mut out = format!("p cnf {} {}\n", num_vars, clauses.len());
        for c in clauses {
            for &lit in c {
                out.push_str(&format!("{} ", lit));
            }
            out.push_str("0\n");
        }
        out
    }

    /// Parse a DIMACS assignment line ("v 1 -2 3 0") into a bool vector.
    pub fn parse_assignment(line: &str, num_vars: usize) -> Vec<bool> {
        let mut assignment = vec![false; num_vars + 1];
        for token in line.split_whitespace() {
            if token == "v" || token == "0" {
                continue;
            }
            if let Ok(lit) = token.parse::<i32>() {
                let var = lit.unsigned_abs() as usize;
                if var <= num_vars {
                    assignment[var] = lit > 0;
                }
            }
        }
        assignment
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_binary() {
        assert_eq!(to_binary(0, 3), vec![false, false, false]);
        assert_eq!(to_binary(1, 3), vec![true, false, false]);
        assert_eq!(to_binary(5, 4), vec![true, false, true, false]);
        assert_eq!(to_binary(7, 3), vec![true, true, true]);
    }

    #[test]
    fn test_bits_needed() {
        assert_eq!(bits_needed(1), 1);
        assert_eq!(bits_needed(2), 1);
        assert_eq!(bits_needed(3), 2);
        assert_eq!(bits_needed(4), 2);
        assert_eq!(bits_needed(5), 3);
        assert_eq!(bits_needed(8), 3);
        assert_eq!(bits_needed(9), 4);
    }

    #[test]
    fn test_encoder_construction() {
        let enc = BmcEncoder::new(3, vec![4, 4, 4]);
        assert_eq!(enc.num_services, 3);
        assert_eq!(enc.bits_per_service, vec![2, 2, 2]);
        assert_eq!(enc.total_state_bits, 6);
    }

    #[test]
    fn test_state_var_indexing() {
        let enc = BmcEncoder::new(2, vec![4, 4]);
        // Service 0 has 2 bits, service 1 has 2 bits, total_state_bits=4.
        // vars_per_step = 4 + 2 = 6
        // Step 0: base = 1
        assert_eq!(enc.state_var(0, 0, 0), 1);
        assert_eq!(enc.state_var(0, 0, 1), 2);
        assert_eq!(enc.state_var(1, 0, 0), 3);
        assert_eq!(enc.state_var(1, 0, 1), 4);
        // Step 1: base = 1 + 6 = 7
        assert_eq!(enc.state_var(0, 1, 0), 7);
    }

    #[test]
    fn test_change_var_indexing() {
        let enc = BmcEncoder::new(2, vec![4, 4]);
        // change vars at step 0: base + total_state_bits = 1 + 4 = 5
        assert_eq!(enc.change_var(0, 0), 5);
        assert_eq!(enc.change_var(1, 0), 6);
    }

    #[test]
    fn test_initial_state_encoding() {
        let enc = BmcEncoder::new(2, vec![4, 4]);
        let clauses = enc.encode_initial_state(&[0, 3]);
        // Service 0 = version 0 → bits [false, false] → negative unit clauses
        // Service 1 = version 3 → bits [true, true] → positive unit clauses
        assert_eq!(clauses.len(), 4); // 2 bits × 2 services
        assert_eq!(clauses[0], vec![-(enc.state_var(0, 0, 0) as i32)]);
        assert_eq!(clauses[1], vec![-(enc.state_var(0, 0, 1) as i32)]);
        assert_eq!(clauses[2], vec![enc.state_var(1, 0, 0) as i32]);
        assert_eq!(clauses[3], vec![enc.state_var(1, 0, 1) as i32]);
    }

    #[test]
    fn test_target_state_encoding() {
        let enc = BmcEncoder::new(2, vec![4, 4]);
        let clauses = enc.encode_target_state(&[2, 1], 3);
        assert_eq!(clauses.len(), 4);
        // version 2 = bits [false, true]
        assert_eq!(clauses[0], vec![-(enc.state_var(0, 3, 0) as i32)]);
        assert_eq!(clauses[1], vec![enc.state_var(0, 3, 1) as i32]);
    }

    #[test]
    fn test_transition_produces_clauses() {
        let enc = BmcEncoder::new(3, vec![4, 4, 4]);
        let clauses = enc.encode_transition(0);
        assert!(!clauses.is_empty());
        // Should contain at-most-one pairwise constraints: C(3,2)=3
        // Plus at-least-one: 1 clause
        // Plus frame axioms: lots
        assert!(clauses.len() > 3);
    }

    #[test]
    fn test_pairwise_amo() {
        let clauses = pairwise_at_most_one(&[1, 2, 3]);
        // C(3,2) = 3 binary clauses
        assert_eq!(clauses.len(), 3);
        assert!(clauses.contains(&vec![-1, -2]));
        assert!(clauses.contains(&vec![-1, -3]));
        assert!(clauses.contains(&vec![-2, -3]));
    }

    #[test]
    fn test_safety_encoding() {
        let enc = BmcEncoder::new(2, vec![3, 3]);
        // Only (0,0) and (1,1) and (2,2) are compatible
        let compat = vec![(0, 0), (1, 1), (2, 2)];
        let clauses = enc.encode_safety_at_step(0, &[(0, 1, compat)]);
        // 3×3 - 3 = 6 incompatible pairs → 6 clauses
        assert_eq!(clauses.len(), 6);
    }

    #[test]
    fn test_completeness_checker_bound() {
        assert_eq!(
            CompletenessChecker::completeness_bound(&[0, 0, 0], &[2, 3, 1]),
            6
        );
        assert_eq!(
            CompletenessChecker::completeness_bound(&[1, 2], &[3, 4]),
            4
        );
    }

    #[test]
    fn test_completeness_checker_is_complete() {
        assert!(CompletenessChecker::is_complete(6, &[0, 0, 0], &[2, 3, 1]));
        assert!(!CompletenessChecker::is_complete(5, &[0, 0, 0], &[2, 3, 1]));
    }

    #[test]
    fn test_completeness_valid_target() {
        assert!(CompletenessChecker::is_valid_target(&[0, 1], &[2, 3]));
        assert!(!CompletenessChecker::is_valid_target(&[3, 1], &[2, 3]));
    }

    #[test]
    fn test_per_service_steps() {
        let steps = CompletenessChecker::per_service_steps(&[0, 2, 1], &[3, 5, 1]);
        assert_eq!(steps, vec![3, 3, 0]);
    }

    #[test]
    fn test_deployment_plan_verify() {
        let plan = DeploymentPlan::new(vec![(0, 0, 1), (1, 0, 1), (0, 1, 2)]);
        assert!(plan.verify(&[0, 0], &[2, 1]));
        assert!(!plan.verify(&[0, 0], &[1, 1]));
    }

    #[test]
    fn test_deployment_plan_intermediate_states() {
        let plan = DeploymentPlan::new(vec![(0, 0, 1), (1, 0, 1)]);
        let states = plan.intermediate_states(&[0, 0]);
        assert_eq!(states.len(), 3);
        assert_eq!(states[0], vec![0, 0]);
        assert_eq!(states[1], vec![1, 0]);
        assert_eq!(states[2], vec![1, 1]);
    }

    #[test]
    fn test_deployment_plan_empty() {
        let plan = DeploymentPlan::new(vec![]);
        assert!(plan.is_empty());
        assert_eq!(plan.len(), 0);
        assert!(plan.verify(&[2, 3], &[2, 3]));
    }

    #[test]
    fn test_unrolling() {
        let enc = BmcEncoder::new(2, vec![4, 4]);
        let mut unroll = BmcUnrolling::new(enc);
        unroll.unroll_to_depth(2);
        assert!(!unroll.clauses_at_depth(0).is_empty());
        assert!(!unroll.clauses_at_depth(1).is_empty());
        assert!(!unroll.clauses_at_depth(2).is_empty());
        assert!(unroll.clauses_at_depth(5).is_empty());
    }

    #[test]
    fn test_all_clauses() {
        let enc = BmcEncoder::new(2, vec![4, 4]);
        let mut unroll = BmcUnrolling::new(enc);
        unroll.unroll_to_depth(1);
        let all = unroll.all_clauses();
        let d0 = unroll.clauses_at_depth(0).len();
        let d1 = unroll.clauses_at_depth(1).len();
        assert_eq!(all.len(), d0 + d1);
    }

    #[test]
    fn test_variable_allocator() {
        let mut alloc = VariableAllocator::new(100);
        assert_eq!(alloc.peek(), 100);
        assert_eq!(alloc.fresh(), 100);
        assert_eq!(alloc.fresh(), 101);
        let block = alloc.fresh_block(5);
        assert_eq!(block, vec![102, 103, 104, 105, 106]);
        assert_eq!(alloc.peek(), 107);
    }

    #[test]
    fn test_clause_stats() {
        let clauses: Vec<Clause> = vec![
            vec![1],
            vec![1, -2],
            vec![1, 2, 3],
            vec![1, 2, 3, 4, 5],
        ];
        let stats = ClauseStats::from_clauses(&clauses);
        assert_eq!(stats.num_clauses, 4);
        assert_eq!(stats.num_unit, 1);
        assert_eq!(stats.num_binary, 1);
        assert_eq!(stats.num_ternary, 1);
        assert_eq!(stats.num_long, 1);
        assert_eq!(stats.max_length, 5);
        assert_eq!(stats.total_literals, 1 + 2 + 3 + 5);
    }

    #[test]
    fn test_clause_stats_avg() {
        let clauses: Vec<Clause> = vec![vec![1, 2], vec![3, 4, 5, 6]];
        let stats = ClauseStats::from_clauses(&clauses);
        assert!((stats.avg_length() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_clause_stats_empty() {
        let stats = ClauseStats::from_clauses(&[]);
        assert_eq!(stats.num_clauses, 0);
        assert!((stats.avg_length() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_num_variables() {
        let clauses: Vec<Clause> = vec![vec![1, -2], vec![3, -1]];
        assert_eq!(ClauseStats::num_variables(&clauses), 3);
    }

    #[test]
    fn test_dimacs_printer() {
        let clauses: Vec<Clause> = vec![vec![1, -2], vec![3, 4, -1]];
        let dimacs = DimacsPrinter::to_dimacs(&clauses, 4);
        assert!(dimacs.starts_with("p cnf 4 2\n"));
        assert!(dimacs.contains("1 -2 0"));
        assert!(dimacs.contains("3 4 -1 0"));
    }

    #[test]
    fn test_dimacs_parse_assignment() {
        let asgn = DimacsPrinter::parse_assignment("v 1 -2 3 -4 0", 4);
        assert!(asgn[1]);
        assert!(!asgn[2]);
        assert!(asgn[3]);
        assert!(!asgn[4]);
    }

    #[test]
    fn test_monotone_single_bit() {
        // a >= b for single bit: a ∨ ¬b
        let clauses = MonotoneEncoder::encode_geq(&[10], &[20]);
        assert_eq!(clauses.len(), 1);
        assert_eq!(clauses[0], vec![10, -20]);
    }

    #[test]
    fn test_monotone_two_bits() {
        let clauses = MonotoneEncoder::encode_geq(&[10, 11], &[20, 21]);
        // Should produce clauses for the auxiliary-variable based comparison.
        assert!(!clauses.is_empty());
        // The final clause ensures gt ∨ eq
        let last = clauses.last().unwrap();
        assert_eq!(last.len(), 2); // gt[1] ∨ eq[1]
    }

    #[test]
    fn test_monotone_encoder_via_var_fn() {
        let enc = BmcEncoder::new(2, vec![4, 4]);
        let var_fn = |svc: usize, step: usize, bit: usize| enc.state_var(svc, step, bit);
        let clauses = MonotoneEncoder::encode_monotone(&var_fn, 0, 0, 2);
        assert!(!clauses.is_empty());
    }

    #[test]
    fn test_step_encoder_pairwise() {
        let enc = BmcEncoder::new(4, vec![4, 4, 4, 4]);
        let clauses = StepEncoder::encode_at_most_one_change(4, 0, &enc);
        // C(4,2) = 6 pairwise constraints
        assert_eq!(clauses.len(), 6);
    }

    #[test]
    fn test_step_encoder_ladder() {
        let lits: Vec<i32> = (1..=8).collect();
        let mut nv = 100;
        let clauses = StepEncoder::encode_at_most_one_ladder(&lits, &mut nv);
        assert!(!clauses.is_empty());
        // Should introduce 7 auxiliary variables
        assert!(nv > 100);
    }

    #[test]
    fn test_step_encoder_best_small() {
        let enc = BmcEncoder::new(3, vec![4, 4, 4]);
        let clauses = StepEncoder::encode_best(3, 0, &enc);
        assert_eq!(clauses.len(), 3); // pairwise for small n
    }

    #[test]
    fn test_total_variables() {
        let enc = BmcEncoder::new(2, vec![4, 4]);
        let v = enc.total_variables(3);
        assert!(v > 0);
    }

    #[test]
    fn test_total_clauses_estimate() {
        let enc = BmcEncoder::new(2, vec![4, 4]);
        let c = enc.total_clauses(3);
        assert!(c > 0);
    }

    #[test]
    fn test_services_in_order() {
        let plan = DeploymentPlan::new(vec![(0, 0, 1), (2, 0, 1), (1, 0, 1)]);
        assert_eq!(plan.services_in_order(), vec![0, 2, 1]);
    }

    #[test]
    fn test_encode_all_monotone() {
        let enc = BmcEncoder::new(2, vec![4, 4]);
        let var_fn = |svc: usize, step: usize, bit: usize| enc.state_var(svc, step, bit);
        let clauses =
            MonotoneEncoder::encode_all_monotone(&var_fn, 2, &enc.bits_per_service, 2);
        assert!(!clauses.is_empty());
    }

    #[test]
    fn test_safety_all_compatible() {
        let enc = BmcEncoder::new(2, vec![2, 2]);
        let compat: Vec<(usize, usize)> = (0..2)
            .flat_map(|a| (0..2).map(move |b| (a, b)))
            .collect();
        let clauses = enc.encode_safety_at_step(0, &[(0, 1, compat)]);
        assert!(clauses.is_empty());
    }

    #[test]
    fn test_extract_plan_basic() {
        let enc = BmcEncoder::new(2, vec![4, 4]);
        // Build a fake assignment where service 0 goes 0→1 at step 0
        let max_var = enc.total_variables(1) + 10;
        let mut assignment = vec![false; max_var + 1];
        // Step 0: svc 0 = version 0, svc 1 = version 0
        // Step 1: svc 0 = version 1, svc 1 = version 0
        // version 1 = bits [true, false]
        let sv0_s1_b0 = enc.state_var(0, 1, 0);
        assignment[sv0_s1_b0] = true; // bit 0 = true → version 1

        let plan = BmcUnrolling::extract_plan(&assignment, 1, &enc);
        // Service 0 went from 0 to 1
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0], (0, 0, 1));
    }
}
