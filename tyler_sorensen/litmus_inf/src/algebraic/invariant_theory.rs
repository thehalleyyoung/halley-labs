/// Polynomial invariants, Molien's theorem, invariant ring computation,
/// and related invariant theory algorithms for the LITMUS∞ algebraic engine.
///
/// Provides polynomial arithmetic, group actions on polynomials, Reynolds
/// operator, invariant ring generators, Molien series computation,
/// primary/secondary invariants, Gröbner basis computation, and
/// symmetric function algorithms.
#[allow(unused)]

use std::collections::{HashMap, HashSet, BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::ops::{Add, Sub, Mul, Neg, AddAssign, SubAssign, MulAssign};

use super::types::Permutation;

// ═══════════════════════════════════════════════════════════════════════════
// Monomial and Polynomial types
// ═══════════════════════════════════════════════════════════════════════════

/// Monomial ordering strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MonomialOrder {
    /// Lexicographic order.
    Lex,
    /// Graded lexicographic order.
    GrLex,
    /// Graded reverse lexicographic order.
    GrevLex,
}

/// A monomial: coefficient × x₁^{e₁} × x₂^{e₂} × ... × xₙ^{eₙ}.
#[derive(Debug, Clone)]
pub struct Monomial {
    /// Exponent vector.
    pub exponents: Vec<u32>,
    /// Coefficient.
    pub coefficient: f64,
}

impl Monomial {
    /// Create a new monomial.
    pub fn new(exponents: Vec<u32>, coefficient: f64) -> Self {
        Self { exponents, coefficient }
    }

    /// Create a constant monomial.
    pub fn constant(n_vars: usize, value: f64) -> Self {
        Self { exponents: vec![0; n_vars], coefficient: value }
    }

    /// Create a single variable monomial: x_i.
    pub fn variable(n_vars: usize, var: usize) -> Self {
        let mut exp = vec![0; n_vars];
        exp[var] = 1;
        Self { exponents: exp, coefficient: 1.0 }
    }

    /// Total degree of the monomial.
    pub fn total_degree(&self) -> u32 {
        self.exponents.iter().sum()
    }

    /// Number of variables.
    pub fn n_vars(&self) -> usize { self.exponents.len() }

    /// Check if this is a constant (all exponents zero).
    pub fn is_constant(&self) -> bool {
        self.exponents.iter().all(|&e| e == 0)
    }

    /// Check if the coefficient is zero.
    pub fn is_zero(&self) -> bool {
        self.coefficient.abs() < 1e-15
    }

    /// Multiply two monomials.
    pub fn mul_monomial(&self, other: &Monomial) -> Monomial {
        let n = self.n_vars().max(other.n_vars());
        let mut exp = vec![0u32; n];
        for (i, &e) in self.exponents.iter().enumerate() { exp[i] += e; }
        for (i, &e) in other.exponents.iter().enumerate() { exp[i] += e; }
        Monomial::new(exp, self.coefficient * other.coefficient)
    }

    /// Evaluate the monomial at a point.
    pub fn evaluate(&self, point: &[f64]) -> f64 {
        let mut result = self.coefficient;
        for (i, &e) in self.exponents.iter().enumerate() {
            if e > 0 && i < point.len() {
                result *= point[i].powi(e as i32);
            }
        }
        result
    }

    /// Compare monomials using a given ordering.
    pub fn cmp_order(&self, other: &Monomial, order: MonomialOrder) -> std::cmp::Ordering {
        match order {
            MonomialOrder::Lex => self.exponents.cmp(&other.exponents).reverse(),
            MonomialOrder::GrLex => {
                let d1 = self.total_degree();
                let d2 = other.total_degree();
                d1.cmp(&d2).then_with(|| self.exponents.cmp(&other.exponents).reverse())
            }
            MonomialOrder::GrevLex => {
                let d1 = self.total_degree();
                let d2 = other.total_degree();
                d1.cmp(&d2).then_with(|| {
                    for i in (0..self.n_vars()).rev() {
                        let e1 = self.exponents.get(i).copied().unwrap_or(0);
                        let e2 = other.exponents.get(i).copied().unwrap_or(0);
                        if e1 != e2 { return e2.cmp(&e1); }
                    }
                    std::cmp::Ordering::Equal
                })
            }
        }
    }

    /// Check if this monomial divides another.
    pub fn divides(&self, other: &Monomial) -> bool {
        let n = self.n_vars().max(other.n_vars());
        for i in 0..n {
            let e1 = self.exponents.get(i).copied().unwrap_or(0);
            let e2 = other.exponents.get(i).copied().unwrap_or(0);
            if e1 > e2 { return false; }
        }
        true
    }

    /// LCM of two monomials (ignoring coefficients).
    pub fn lcm(&self, other: &Monomial) -> Monomial {
        let n = self.n_vars().max(other.n_vars());
        let exp: Vec<u32> = (0..n).map(|i| {
            let e1 = self.exponents.get(i).copied().unwrap_or(0);
            let e2 = other.exponents.get(i).copied().unwrap_or(0);
            e1.max(e2)
        }).collect();
        Monomial::new(exp, 1.0)
    }
}

impl PartialEq for Monomial {
    fn eq(&self, other: &Self) -> bool {
        self.exponents == other.exponents && (self.coefficient - other.coefficient).abs() < 1e-15
    }
}

impl fmt::Display for Monomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_constant() {
            return write!(f, "{:.4}", self.coefficient);
        }
        if (self.coefficient - 1.0).abs() > 1e-15 && (self.coefficient + 1.0).abs() > 1e-15 {
            write!(f, "{:.4}", self.coefficient)?;
        } else if (self.coefficient + 1.0).abs() < 1e-15 {
            write!(f, "-")?;
        }
        for (i, &e) in self.exponents.iter().enumerate() {
            if e == 1 { write!(f, "x{}", i)?; }
            else if e > 1 { write!(f, "x{}^{}", i, e)?; }
        }
        Ok(())
    }
}

/// A multivariate polynomial over the reals.
#[derive(Debug, Clone)]
pub struct Polynomial {
    /// Terms of the polynomial.
    pub terms: Vec<Monomial>,
    /// Number of variables.
    pub n_vars: usize,
}

impl Polynomial {
    /// Create a new polynomial from terms.
    pub fn new(terms: Vec<Monomial>, n_vars: usize) -> Self {
        let mut p = Self { terms, n_vars };
        p.collect_like_terms();
        p
    }

    /// Create the zero polynomial.
    pub fn zero(n_vars: usize) -> Self {
        Self { terms: Vec::new(), n_vars }
    }

    /// Create a constant polynomial.
    pub fn constant(n_vars: usize, value: f64) -> Self {
        if value.abs() < 1e-15 { return Self::zero(n_vars); }
        Self { terms: vec![Monomial::constant(n_vars, value)], n_vars }
    }

    /// Create a single variable polynomial: x_i.
    pub fn variable(n_vars: usize, var: usize) -> Self {
        Self { terms: vec![Monomial::variable(n_vars, var)], n_vars }
    }

    /// Check if the polynomial is zero.
    pub fn is_zero(&self) -> bool { self.terms.is_empty() }

    /// Total degree of the polynomial.
    pub fn total_degree(&self) -> u32 {
        self.terms.iter().map(|t| t.total_degree()).max().unwrap_or(0)
    }

    /// Check if the polynomial is homogeneous.
    pub fn is_homogeneous(&self) -> bool {
        if self.terms.len() <= 1 { return true; }
        let d = self.terms[0].total_degree();
        self.terms.iter().all(|t| t.total_degree() == d)
    }

    /// Get the leading term (with respect to a monomial order).
    pub fn leading_term(&self, order: MonomialOrder) -> Option<&Monomial> {
        self.terms.iter().max_by(|a, b| a.cmp_order(b, order))
    }

    /// Get the leading monomial (exponent vector of leading term).
    pub fn leading_monomial(&self, order: MonomialOrder) -> Option<Vec<u32>> {
        self.leading_term(order).map(|t| t.exponents.clone())
    }

    /// Get the leading coefficient.
    pub fn leading_coefficient(&self, order: MonomialOrder) -> Option<f64> {
        self.leading_term(order).map(|t| t.coefficient)
    }

    /// Add two polynomials.
    pub fn add_poly(&self, other: &Polynomial) -> Polynomial {
        let n = self.n_vars.max(other.n_vars);
        let mut terms = self.terms.clone();
        terms.extend(other.terms.iter().cloned());
        Polynomial::new(terms, n)
    }

    /// Subtract two polynomials.
    pub fn sub_poly(&self, other: &Polynomial) -> Polynomial {
        let n = self.n_vars.max(other.n_vars);
        let mut terms = self.terms.clone();
        for t in &other.terms {
            terms.push(Monomial::new(t.exponents.clone(), -t.coefficient));
        }
        Polynomial::new(terms, n)
    }

    /// Multiply two polynomials.
    pub fn mul_poly(&self, other: &Polynomial) -> Polynomial {
        let n = self.n_vars.max(other.n_vars);
        let mut terms = Vec::new();
        for a in &self.terms {
            for b in &other.terms {
                terms.push(a.mul_monomial(b));
            }
        }
        Polynomial::new(terms, n)
    }

    /// Scalar multiplication.
    pub fn scale(&self, scalar: f64) -> Polynomial {
        let terms: Vec<Monomial> = self.terms.iter()
            .map(|t| Monomial::new(t.exponents.clone(), t.coefficient * scalar))
            .collect();
        Polynomial::new(terms, self.n_vars)
    }

    /// Evaluate the polynomial at a point.
    pub fn evaluate(&self, point: &[f64]) -> f64 {
        self.terms.iter().map(|t| t.evaluate(point)).sum()
    }

    /// Collect like terms.
    pub fn collect_like_terms(&mut self) {
        let mut combined: BTreeMap<Vec<u32>, f64> = BTreeMap::new();
        for term in &self.terms {
            *combined.entry(term.exponents.clone()).or_default() += term.coefficient;
        }
        self.terms = combined.into_iter()
            .filter(|(_, c)| c.abs() > 1e-15)
            .map(|(exp, c)| Monomial::new(exp, c))
            .collect();
    }

    /// Sort terms by a monomial order.
    pub fn sort_terms(&mut self, order: MonomialOrder) {
        self.terms.sort_by(|a, b| b.cmp_order(a, order));
    }

    /// Number of terms.
    pub fn num_terms(&self) -> usize { self.terms.len() }

    /// Power: p^k.
    pub fn pow(&self, k: u32) -> Polynomial {
        if k == 0 { return Polynomial::constant(self.n_vars, 1.0); }
        let mut result = self.clone();
        for _ in 1..k {
            result = result.mul_poly(self);
        }
        result
    }
}

impl fmt::Display for Polynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() { return write!(f, "0"); }
        for (i, term) in self.terms.iter().enumerate() {
            if i > 0 && term.coefficient > 0.0 { write!(f, " + ")?; }
            else if i > 0 && term.coefficient < 0.0 { write!(f, " - ")?; }
            if i > 0 { write!(f, "{}", Monomial::new(term.exponents.clone(), term.coefficient.abs()))?; }
            else { write!(f, "{}", term)?; }
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// GroupAction on Polynomials
// ═══════════════════════════════════════════════════════════════════════════

/// Permutation action on polynomial variables.
#[derive(Debug, Clone)]
pub struct PermutationAction {
    /// Number of variables.
    n_vars: usize,
}

impl PermutationAction {
    /// Create a new action.
    pub fn new(n_vars: usize) -> Self {
        Self { n_vars }
    }

    /// Act on a monomial by permuting variables.
    pub fn act_on_monomial(&self, perm: &Permutation, mono: &Monomial) -> Monomial {
        let mut new_exp = vec![0u32; self.n_vars];
        for (i, &e) in mono.exponents.iter().enumerate() {
            if i < perm.degree() {
                let j = perm.apply(i as u32) as usize;
                if j < self.n_vars { new_exp[j] = e; }
            }
        }
        Monomial::new(new_exp, mono.coefficient)
    }

    /// Act on a polynomial by permuting variables.
    pub fn act_on_polynomial(&self, perm: &Permutation, poly: &Polynomial) -> Polynomial {
        let terms: Vec<Monomial> = poly.terms.iter()
            .map(|t| self.act_on_monomial(perm, t))
            .collect();
        Polynomial::new(terms, self.n_vars)
    }

    /// Check if a polynomial is invariant under a permutation.
    pub fn is_invariant(&self, perm: &Permutation, poly: &Polynomial) -> bool {
        let acted = self.act_on_polynomial(perm, poly);
        poly_equal(poly, &acted)
    }

    /// Check if a polynomial is invariant under a group.
    pub fn is_invariant_under_group(&self, generators: &[Permutation], poly: &Polynomial) -> bool {
        generators.iter().all(|g| self.is_invariant(g, poly))
    }

    /// Symmetrize a polynomial: Σ_{g∈G} g·f (without dividing by |G|).
    pub fn symmetrize(&self, group: &[Permutation], poly: &Polynomial) -> Polynomial {
        let mut result = Polynomial::zero(self.n_vars);
        for g in group {
            let acted = self.act_on_polynomial(g, poly);
            result = result.add_poly(&acted);
        }
        result
    }

    /// Reynolds operator: R(f) = (1/|G|) Σ_{g∈G} g·f.
    pub fn reynolds_operator(&self, group: &[Permutation], poly: &Polynomial) -> Polynomial {
        let sym = self.symmetrize(group, poly);
        sym.scale(1.0 / group.len() as f64)
    }
}

/// Check if two polynomials are approximately equal.
fn poly_equal(a: &Polynomial, b: &Polynomial) -> bool {
    if a.terms.len() != b.terms.len() { return false; }
    let mut a_map: BTreeMap<Vec<u32>, f64> = BTreeMap::new();
    let mut b_map: BTreeMap<Vec<u32>, f64> = BTreeMap::new();
    for t in &a.terms { *a_map.entry(t.exponents.clone()).or_default() += t.coefficient; }
    for t in &b.terms { *b_map.entry(t.exponents.clone()).or_default() += t.coefficient; }
    if a_map.len() != b_map.len() { return false; }
    for (exp, ca) in &a_map {
        match b_map.get(exp) {
            Some(cb) if (ca - cb).abs() < 1e-10 => {}
            _ => return false,
        }
    }
    true
}

// ═══════════════════════════════════════════════════════════════════════════
// InvariantRing
// ═══════════════════════════════════════════════════════════════════════════

/// The invariant ring R[x₁,...,xₙ]^G for a permutation group G.
#[derive(Debug, Clone)]
pub struct InvariantRing {
    /// Number of variables.
    n_vars: usize,
    /// Group elements.
    group: Vec<Permutation>,
    /// Computed generators of the invariant ring.
    generators: Vec<Polynomial>,
    /// Degrees of the generators.
    generator_degrees: Vec<u32>,
    /// The action handler.
    action: PermutationAction,
}

impl InvariantRing {
    /// Create a new invariant ring.
    pub fn new(n_vars: usize, group: Vec<Permutation>) -> Self {
        Self {
            n_vars,
            group,
            generators: Vec::new(),
            generator_degrees: Vec::new(),
            action: PermutationAction::new(n_vars),
        }
    }

    /// Compute generators up to a given degree bound.
    ///
    /// Uses the Reynolds operator to project monomials onto invariants.
    pub fn compute_generators(&mut self, max_degree: u32) -> &[Polynomial] {
        self.generators.clear();
        self.generator_degrees.clear();
        for deg in 1..=max_degree {
            let monomials = self.monomials_of_degree(deg);
            for mono in monomials {
                let poly = Polynomial::new(vec![mono], self.n_vars);
                let inv = self.action.reynolds_operator(&self.group, &poly);
                if !inv.is_zero() && !self.is_in_span(&inv) {
                    self.generator_degrees.push(deg);
                    self.generators.push(inv);
                }
            }
        }
        &self.generators
    }

    /// Generate all monomials of a given total degree.
    fn monomials_of_degree(&self, degree: u32) -> Vec<Monomial> {
        let mut result = Vec::new();
        let mut exp = vec![0u32; self.n_vars];
        self.gen_monomials_recursive(&mut exp, 0, degree, &mut result);
        result
    }

    fn gen_monomials_recursive(
        &self, exp: &mut Vec<u32>, var: usize, remaining: u32, result: &mut Vec<Monomial>
    ) {
        if var == self.n_vars - 1 {
            exp[var] = remaining;
            result.push(Monomial::new(exp.clone(), 1.0));
            exp[var] = 0;
            return;
        }
        for e in 0..=remaining {
            exp[var] = e;
            self.gen_monomials_recursive(exp, var + 1, remaining - e, result);
        }
        exp[var] = 0;
    }

    /// Check if an invariant is in the span of current generators.
    fn is_in_span(&self, poly: &Polynomial) -> bool {
        // Simple check: see if the leading monomial matches any generator
        if self.generators.is_empty() { return false; }
        for gen in &self.generators {
            if poly_equal(poly, gen) { return true; }
        }
        false
    }

    /// Get the primary invariants.
    pub fn primary_invariants(&self) -> Vec<&Polynomial> {
        self.generators.iter().take(self.n_vars).collect()
    }

    /// Get the secondary invariants.
    pub fn secondary_invariants(&self) -> Vec<&Polynomial> {
        if self.generators.len() > self.n_vars {
            self.generators.iter().skip(self.n_vars).collect()
        } else {
            Vec::new()
        }
    }

    /// Compute the Hilbert series as a vector of coefficients.
    pub fn hilbert_series(&self, max_degree: u32) -> Vec<usize> {
        let mut coeffs = vec![0usize; (max_degree + 1) as usize];
        for deg in 0..=max_degree {
            let monomials = if deg == 0 {
                vec![Monomial::constant(self.n_vars, 1.0)]
            } else {
                self.monomials_of_degree(deg)
            };
            for mono in monomials {
                let poly = Polynomial::new(vec![mono], self.n_vars);
                let inv = self.action.reynolds_operator(&self.group, &poly);
                if !inv.is_zero() {
                    coeffs[deg as usize] += 1;
                }
            }
        }
        coeffs
    }

    /// Krull dimension of the invariant ring (= n for polynomial ring).
    pub fn krull_dimension(&self) -> usize { self.n_vars }

    /// Get generators.
    pub fn get_generators(&self) -> &[Polynomial] { &self.generators }
}

// ═══════════════════════════════════════════════════════════════════════════
// MolienSeries
// ═══════════════════════════════════════════════════════════════════════════

/// Molien's theorem for computing the Hilbert series of an invariant ring.
///
/// M(t) = (1/|G|) Σ_{g∈G} 1/det(I - t·ρ(g))
#[derive(Debug, Clone)]
pub struct MolienSeries {
    /// Coefficients of the series (coefficient[d] = dim of degree-d invariants).
    pub coefficients: Vec<f64>,
    /// Number of terms computed.
    pub num_terms: usize,
}

impl MolienSeries {
    /// Compute the Molien series for a permutation group.
    ///
    /// For a permutation group, ρ(g) is the permutation matrix.
    /// det(I - t·P_g) = Π_{cycle c of g} (1 - t^{|c|}).
    pub fn compute(group: &[Permutation], max_degree: usize) -> Self {
        let order = group.len() as f64;
        let mut coefficients = vec![0.0f64; max_degree + 1];
        for g in group {
            let cycles = g.cycle_decomposition();
            // 1/det(I - tP_g) = 1/Π_c(1 - t^{|c|}) = Π_c 1/(1-t^{|c|})
            // = Π_c (1 + t^{|c|} + t^{2|c|} + ...)
            let cycle_lengths: Vec<usize> = cycles.iter().map(|c| c.len()).collect();
            // Compute product of geometric series
            let mut contribution = vec![0.0f64; max_degree + 1];
            contribution[0] = 1.0;
            for &cl in &cycle_lengths {
                let mut new_contrib = vec![0.0f64; max_degree + 1];
                for d in 0..=max_degree {
                    if contribution[d].abs() < 1e-15 { continue; }
                    let mut k = 0;
                    while d + k * cl <= max_degree {
                        new_contrib[d + k * cl] += contribution[d];
                        k += 1;
                    }
                }
                contribution = new_contrib;
            }
            for d in 0..=max_degree {
                coefficients[d] += contribution[d];
            }
        }
        // Divide by |G|
        for c in &mut coefficients {
            *c /= order;
        }
        Self { coefficients, num_terms: max_degree + 1 }
    }

    /// Get the coefficient at a specific degree.
    pub fn coefficient_at_degree(&self, d: usize) -> f64 {
        self.coefficients.get(d).copied().unwrap_or(0.0)
    }

    /// Truncate to a specific number of terms.
    pub fn truncate(&self, n: usize) -> MolienSeries {
        Self {
            coefficients: self.coefficients[..n.min(self.num_terms)].to_vec(),
            num_terms: n.min(self.num_terms),
        }
    }

    /// Check if the series represents a polynomial (finite number of terms).
    pub fn is_polynomial(&self) -> bool {
        // Check if trailing coefficients are zero
        self.coefficients.iter().rev()
            .take_while(|&&c| c.abs() < 1e-15)
            .count() > self.num_terms / 2
    }
}

impl fmt::Display for MolienSeries {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "M(t) = ")?;
        for (d, &c) in self.coefficients.iter().enumerate() {
            if c.abs() < 1e-15 { continue; }
            if d > 0 { write!(f, " + ")?; }
            if d == 0 { write!(f, "{:.0}", c)?; }
            else { write!(f, "{:.0}t^{}", c, d)?; }
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FundamentalInvariants — classical symmetric functions
// ═══════════════════════════════════════════════════════════════════════════

/// Symmetric function computations.
#[derive(Debug, Clone)]
pub struct SymmetricFunctions {
    /// Number of variables.
    n_vars: usize,
}

impl SymmetricFunctions {
    /// Create a new symmetric functions instance.
    pub fn new(n_vars: usize) -> Self { Self { n_vars } }

    /// Compute the k-th elementary symmetric polynomial e_k(x₁,...,xₙ).
    pub fn elementary(&self, k: usize) -> Polynomial {
        if k > self.n_vars { return Polynomial::zero(self.n_vars); }
        if k == 0 { return Polynomial::constant(self.n_vars, 1.0); }
        let mut terms = Vec::new();
        let mut indices = vec![0usize; k];
        // Initialize to 0,1,...,k-1
        for i in 0..k { indices[i] = i; }
        loop {
            let mut exp = vec![0u32; self.n_vars];
            for &i in &indices { exp[i] = 1; }
            terms.push(Monomial::new(exp, 1.0));
            // Next combination
            if !next_combination(&mut indices, self.n_vars) { break; }
        }
        Polynomial::new(terms, self.n_vars)
    }

    /// Compute the k-th power sum polynomial p_k(x₁,...,xₙ) = Σ xᵢ^k.
    pub fn power_sum(&self, k: u32) -> Polynomial {
        let terms: Vec<Monomial> = (0..self.n_vars).map(|i| {
            let mut exp = vec![0u32; self.n_vars];
            exp[i] = k;
            Monomial::new(exp, 1.0)
        }).collect();
        Polynomial::new(terms, self.n_vars)
    }

    /// Compute the k-th complete homogeneous symmetric polynomial h_k.
    pub fn complete_homogeneous(&self, k: u32) -> Polynomial {
        if k == 0 { return Polynomial::constant(self.n_vars, 1.0); }
        let mut terms = Vec::new();
        let mut exp = vec![0u32; self.n_vars];
        self.gen_weak_compositions(&mut exp, 0, k, &mut terms);
        Polynomial::new(terms, self.n_vars)
    }

    fn gen_weak_compositions(
        &self, exp: &mut Vec<u32>, var: usize, remaining: u32, terms: &mut Vec<Monomial>
    ) {
        if var == self.n_vars - 1 {
            exp[var] = remaining;
            terms.push(Monomial::new(exp.clone(), 1.0));
            exp[var] = 0;
            return;
        }
        for e in 0..=remaining {
            exp[var] = e;
            self.gen_weak_compositions(exp, var + 1, remaining - e, terms);
        }
        exp[var] = 0;
    }

    /// Newton's identity: express power sums in terms of elementary.
    ///
    /// p_k = Σ_{i=1}^{k-1} (-1)^{i-1} e_i p_{k-i} + (-1)^{k-1} k e_k.
    pub fn newton_identity_coefficients(&self, k: usize) -> Vec<(usize, f64)> {
        let mut coeffs = Vec::new();
        for i in 1..k {
            let sign = if (i - 1) % 2 == 0 { 1.0 } else { -1.0 };
            coeffs.push((i, sign));
        }
        let sign = if (k - 1) % 2 == 0 { 1.0 } else { -1.0 };
        coeffs.push((k, sign * k as f64));
        coeffs
    }
}

/// Next combination in lexicographic order.
fn next_combination(indices: &mut Vec<usize>, n: usize) -> bool {
    let k = indices.len();
    let mut i = k;
    while i > 0 {
        i -= 1;
        if indices[i] < n - k + i {
            indices[i] += 1;
            for j in (i + 1)..k {
                indices[j] = indices[j - 1] + 1;
            }
            return true;
        }
    }
    false
}

// ═══════════════════════════════════════════════════════════════════════════
// GroebnerBasis
// ═══════════════════════════════════════════════════════════════════════════

/// A Gröbner basis for a polynomial ideal.
#[derive(Debug, Clone)]
pub struct GroebnerBasis {
    /// Basis polynomials.
    pub basis: Vec<Polynomial>,
    /// Monomial order used.
    pub order: MonomialOrder,
    /// Number of variables.
    n_vars: usize,
}

impl GroebnerBasis {
    /// Compute a Gröbner basis from a set of polynomials (Buchberger's algorithm).
    pub fn compute(polynomials: Vec<Polynomial>, order: MonomialOrder) -> Self {
        let n_vars = polynomials.iter().map(|p| p.n_vars).max().unwrap_or(0);
        let mut basis = polynomials;
        let mut changed = true;
        let max_iterations = 100;
        let mut iteration = 0;
        while changed && iteration < max_iterations {
            changed = false;
            iteration += 1;
            let n = basis.len();
            let mut new_polys = Vec::new();
            for i in 0..n {
                for j in (i + 1)..n {
                    let s = Self::s_polynomial(&basis[i], &basis[j], order);
                    let remainder = Self::reduce_by(&s, &basis, order);
                    if !remainder.is_zero() {
                        new_polys.push(remainder);
                        changed = true;
                    }
                }
            }
            basis.extend(new_polys);
        }
        Self { basis, order, n_vars }
    }

    /// Compute the S-polynomial of two polynomials.
    pub fn s_polynomial(f: &Polynomial, g: &Polynomial, order: MonomialOrder) -> Polynomial {
        let lt_f = match f.leading_term(order) { Some(t) => t, None => return Polynomial::zero(f.n_vars) };
        let lt_g = match g.leading_term(order) { Some(t) => t, None => return Polynomial::zero(g.n_vars) };
        let lcm = lt_f.lcm(lt_g);
        let n = f.n_vars.max(g.n_vars);
        // S(f,g) = lcm/lt(f) * f - lcm/lt(g) * g
        let mut f_quotient_exp = vec![0u32; n];
        let mut g_quotient_exp = vec![0u32; n];
        for i in 0..lcm.exponents.len() {
            f_quotient_exp[i] = lcm.exponents[i] - lt_f.exponents.get(i).copied().unwrap_or(0);
            g_quotient_exp[i] = lcm.exponents[i] - lt_g.exponents.get(i).copied().unwrap_or(0);
        }
        let f_factor = Polynomial::new(
            vec![Monomial::new(f_quotient_exp, 1.0 / lt_f.coefficient)], n);
        let g_factor = Polynomial::new(
            vec![Monomial::new(g_quotient_exp, 1.0 / lt_g.coefficient)], n);
        f_factor.mul_poly(f).sub_poly(&g_factor.mul_poly(g))
    }

    /// Reduce a polynomial by a set of polynomials.
    pub fn reduce_by(poly: &Polynomial, basis: &[Polynomial], order: MonomialOrder) -> Polynomial {
        let mut remainder = poly.clone();
        let mut changed = true;
        while changed {
            changed = false;
            for b in basis {
                let lt_b = match b.leading_term(order) { Some(t) => t, None => continue };
                loop {
                    let lt_r = match remainder.leading_term(order) { Some(t) => t.clone(), None => break };
                    if !lt_b.divides(&lt_r) { break; }
                    // Subtract (lt_r / lt_b) * b from remainder
                    let n = remainder.n_vars.max(b.n_vars);
                    let mut quotient_exp = vec![0u32; n];
                    for i in 0..lt_r.exponents.len() {
                        quotient_exp[i] = lt_r.exponents.get(i).copied().unwrap_or(0)
                            - lt_b.exponents.get(i).copied().unwrap_or(0);
                    }
                    let factor = Polynomial::new(
                        vec![Monomial::new(quotient_exp, lt_r.coefficient / lt_b.coefficient)], n);
                    remainder = remainder.sub_poly(&factor.mul_poly(b));
                    changed = true;
                }
            }
        }
        remainder
    }

    /// Reduce a polynomial to normal form.
    pub fn reduce(&self, poly: &Polynomial) -> Polynomial {
        Self::reduce_by(poly, &self.basis, self.order)
    }

    /// Check ideal membership: poly ∈ ⟨basis⟩ iff reduce(poly) = 0.
    pub fn ideal_membership(&self, poly: &Polynomial) -> bool {
        self.reduce(poly).is_zero()
    }

    /// Minimize the basis.
    pub fn minimalize(&mut self) {
        let mut minimal = Vec::new();
        for i in 0..self.basis.len() {
            let lt_i = match self.basis[i].leading_term(self.order) { Some(t) => t.clone(), None => continue };
            let mut is_redundant = false;
            for j in 0..self.basis.len() {
                if i == j { continue; }
                if let Some(lt_j) = self.basis[j].leading_term(self.order) {
                    if lt_j.divides(&lt_i) {
                        is_redundant = true;
                        break;
                    }
                }
            }
            if !is_redundant { minimal.push(self.basis[i].clone()); }
        }
        self.basis = minimal;
    }

    /// Get the basis polynomials.
    pub fn get_basis(&self) -> &[Polynomial] { &self.basis }

    /// Number of basis elements.
    pub fn size(&self) -> usize { self.basis.len() }
}

// ═══════════════════════════════════════════════════════════════════════════
// InvariantDetection
// ═══════════════════════════════════════════════════════════════════════════

/// Tools for detecting and testing polynomial invariants.
#[derive(Debug, Clone)]
pub struct InvariantDetector {
    /// Number of variables.
    n_vars: usize,
    /// Group elements.
    group: Vec<Permutation>,
    /// Action handler.
    action: PermutationAction,
}

impl InvariantDetector {
    /// Create a new detector.
    pub fn new(n_vars: usize, group: Vec<Permutation>) -> Self {
        Self { n_vars, group, action: PermutationAction::new(n_vars) }
    }

    /// Check if a polynomial is invariant under the group.
    pub fn is_invariant(&self, poly: &Polynomial) -> bool {
        self.action.is_invariant_under_group(&self.group, poly)
    }

    /// Project a polynomial onto the invariant subspace.
    pub fn projection_to_invariants(&self, poly: &Polynomial) -> Polynomial {
        self.action.reynolds_operator(&self.group, poly)
    }

    /// Compute the "degree of invariance": fraction of group elements under which poly is fixed.
    pub fn degree_of_invariance(&self, poly: &Polynomial) -> f64 {
        let fixed_count = self.group.iter()
            .filter(|g| self.action.is_invariant(g, poly))
            .count();
        fixed_count as f64 / self.group.len() as f64
    }

    /// Compute the "symmetry defect": distance from being invariant.
    pub fn symmetry_defect(&self, poly: &Polynomial) -> f64 {
        let projected = self.projection_to_invariants(poly);
        let diff = poly.sub_poly(&projected);
        // Approximate L2 norm at a few random points
        let points: Vec<Vec<f64>> = (0..10).map(|i| {
            (0..self.n_vars).map(|j| ((i * 7 + j * 3 + 1) % 11) as f64 / 5.0).collect()
        }).collect();
        let mut sum_sq = 0.0;
        for pt in &points {
            let v = diff.evaluate(pt);
            sum_sq += v * v;
        }
        (sum_sq / points.len() as f64).sqrt()
    }

    /// Find a basis for the invariant subspace of homogeneous polynomials of degree d.
    pub fn invariant_subspace(&self, degree: u32) -> Vec<Polynomial> {
        let mut result = Vec::new();
        let action = PermutationAction::new(self.n_vars);
        let ring = InvariantRing::new(self.n_vars, self.group.clone());
        let monomials = ring.monomials_of_degree(degree);
        for mono in monomials {
            let poly = Polynomial::new(vec![mono], self.n_vars);
            let inv = action.reynolds_operator(&self.group, &poly);
            if !inv.is_zero() {
                let mut is_new = true;
                for existing in &result {
                    if poly_equal(&inv, existing) { is_new = false; break; }
                }
                if is_new { result.push(inv); }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn symmetric_group_s3() -> Vec<Permutation> {
        let mut result = Vec::new();
        let perms: Vec<Vec<u32>> = vec![
            vec![0,1,2], vec![0,2,1], vec![1,0,2],
            vec![1,2,0], vec![2,0,1], vec![2,1,0],
        ];
        for p in perms { result.push(Permutation::new(p)); }
        result
    }

    #[test]
    fn test_monomial_arithmetic() {
        let a = Monomial::new(vec![2, 1], 3.0);
        let b = Monomial::new(vec![1, 2], 2.0);
        let c = a.mul_monomial(&b);
        assert_eq!(c.exponents, vec![3, 3]);
        assert!((c.coefficient - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_polynomial_arithmetic() {
        let p = Polynomial::variable(3, 0).add_poly(&Polynomial::variable(3, 1));
        assert_eq!(p.num_terms(), 2);
        assert_eq!(p.total_degree(), 1);
    }

    #[test]
    fn test_reynolds_operator() {
        let group = symmetric_group_s3();
        let action = PermutationAction::new(3);
        let x0 = Polynomial::variable(3, 0);
        let inv = action.reynolds_operator(&group, &x0);
        // Reynolds of x0 under S3 = (x0 + x1 + x2) / 3
        assert!(!inv.is_zero());
        assert!(action.is_invariant_under_group(&group, &inv));
    }

    #[test]
    fn test_molien_series() {
        // S3 acting on 3 variables
        let group = symmetric_group_s3();
        let ms = MolienSeries::compute(&group, 5);
        // dim(invariants of degree 0) = 1
        assert!((ms.coefficient_at_degree(0) - 1.0).abs() < 1e-10);
        // dim(invariants of degree 1) = 1 (e_1 = x+y+z)
        assert!((ms.coefficient_at_degree(1) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_elementary_symmetric() {
        let sf = SymmetricFunctions::new(3);
        let e1 = sf.elementary(1);
        assert_eq!(e1.num_terms(), 3);
        let e2 = sf.elementary(2);
        assert_eq!(e2.num_terms(), 3);
        let e3 = sf.elementary(3);
        assert_eq!(e3.num_terms(), 1);
    }
}
