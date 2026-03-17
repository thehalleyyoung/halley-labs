//! Bitvector encoding utilities for cipher suites, versions, and sets.
//!
//! Provides efficient bitvector representations for protocol elements:
//! - 16-bit IANA cipher suite IDs
//! - Protocol version encodings
//! - Set membership via bitvector operations
//! - Security ordering comparisons
//! - Cardinality constraints

use crate::{SmtDeclaration, SmtExpr, SmtSort};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

// ─── Bitvector sort configurations ──────────────────────────────────────

/// Bitvector sort with width tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BvSort {
    pub width: u32,
}

impl BvSort {
    pub fn new(width: u32) -> Self {
        BvSort { width }
    }

    pub fn cipher_suite() -> Self {
        BvSort { width: 16 }
    }

    pub fn version() -> Self {
        BvSort { width: 16 }
    }

    pub fn phase() -> Self {
        BvSort { width: 4 }
    }

    pub fn action_type() -> Self {
        BvSort { width: 4 }
    }

    pub fn extension_id() -> Self {
        BvSort { width: 16 }
    }

    pub fn boolean() -> Self {
        BvSort { width: 1 }
    }

    pub fn security_level() -> Self {
        BvSort { width: 4 }
    }

    pub fn to_smt_sort(&self) -> SmtSort {
        SmtSort::BitVec(self.width)
    }

    pub fn max_value(&self) -> u64 {
        if self.width >= 64 {
            u64::MAX
        } else {
            (1u64 << self.width) - 1
        }
    }

    pub fn zero(&self) -> SmtExpr {
        SmtExpr::bv_lit(0, self.width)
    }

    pub fn ones(&self) -> SmtExpr {
        SmtExpr::bv_lit(self.max_value(), self.width)
    }
}

// ─── Security level encoding ────────────────────────────────────────────

/// Maps security level names to bitvector values for ordering comparisons.
#[derive(Debug, Clone)]
pub struct SecurityLevelEncoder {
    pub width: u32,
    level_values: IndexMap<String, u64>,
}

impl SecurityLevelEncoder {
    pub fn new() -> Self {
        let mut level_values = IndexMap::new();
        level_values.insert("broken".to_string(), 0);
        level_values.insert("weak".to_string(), 1);
        level_values.insert("legacy".to_string(), 2);
        level_values.insert("standard".to_string(), 3);
        level_values.insert("high".to_string(), 4);
        SecurityLevelEncoder {
            width: 4,
            level_values,
        }
    }

    pub fn encode_level(&self, level: &str) -> SmtExpr {
        let val = self.level_values.get(level).copied().unwrap_or(0);
        SmtExpr::bv_lit(val, self.width)
    }

    pub fn is_secure(&self, level_expr: SmtExpr) -> SmtExpr {
        let standard = SmtExpr::bv_lit(3, self.width);
        SmtExpr::BvUle(Box::new(standard), Box::new(level_expr))
    }

    pub fn is_downgrade(&self, selected: SmtExpr, expected: SmtExpr) -> SmtExpr {
        SmtExpr::bv_ult(selected, expected)
    }
}

impl Default for SecurityLevelEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Cipher suite bitvector encoder ─────────────────────────────────────

/// Encodes cipher suite operations as bitvector constraints.
#[derive(Debug, Clone)]
pub struct CipherSuiteEncoder {
    width: u32,
    /// Map from IANA ID to security score (used for ordering).
    security_scores: IndexMap<u16, u32>,
    /// All known cipher suite IDs.
    known_ids: BTreeSet<u16>,
}

impl CipherSuiteEncoder {
    pub fn new(cipher_ids: &[u16]) -> Self {
        let mut security_scores = IndexMap::new();
        let mut known_ids = BTreeSet::new();
        for &id in cipher_ids {
            known_ids.insert(id);
            security_scores.insert(id, Self::default_score(id));
        }
        CipherSuiteEncoder {
            width: 16,
            security_scores,
            known_ids,
        }
    }

    pub fn with_scores(scores: IndexMap<u16, u32>) -> Self {
        let known_ids: BTreeSet<u16> = scores.keys().copied().collect();
        CipherSuiteEncoder {
            width: 16,
            security_scores: scores,
            known_ids,
        }
    }

    fn default_score(id: u16) -> u32 {
        match id {
            0x0000 => 0,   // NULL
            0x0001..=0x0005 => 50,  // RC4/DES export
            0x000A => 100,  // 3DES
            0x002F => 200,  // AES_128_CBC_SHA
            0x0033 => 250,  // DHE_RSA_AES_128_CBC_SHA
            0x0035 => 220,  // AES_256_CBC_SHA
            0x009C => 350,  // AES_128_GCM_SHA256
            0x009D => 370,  // AES_256_GCM_SHA384
            0x009E => 400,  // DHE_RSA_AES_128_GCM_SHA256
            0x009F => 420,  // DHE_RSA_AES_256_GCM_SHA384
            0xC009 => 200,  // ECDHE_ECDSA_AES_128_CBC_SHA
            0xC013 => 200,  // ECDHE_RSA_AES_128_CBC_SHA
            0xC014 => 220,  // ECDHE_RSA_AES_256_CBC_SHA
            0xC023 => 350,  // ECDHE_ECDSA_AES_128_CBC_SHA256
            0xC027 => 400,  // ECDHE_RSA_AES_128_CBC_SHA256
            0xC02B => 450,  // ECDHE_ECDSA_AES_128_GCM_SHA256
            0xC02C => 470,  // ECDHE_ECDSA_AES_256_GCM_SHA384
            0xC02F => 450,  // ECDHE_RSA_AES_128_GCM_SHA256
            0xC030 => 470,  // ECDHE_RSA_AES_256_GCM_SHA384
            0xCCA8 => 480,  // ECDHE_RSA_CHACHA20_POLY1305
            0xCCA9 => 480,  // ECDHE_ECDSA_CHACHA20_POLY1305
            0x1301 => 500,  // TLS13_AES_128_GCM_SHA256
            0x1302 => 520,  // TLS13_AES_256_GCM_SHA384
            0x1303 => 510,  // TLS13_CHACHA20_POLY1305_SHA256
            _ => 100,
        }
    }

    /// Encode a cipher suite ID as a bitvector literal.
    pub fn encode_id(&self, id: u16) -> SmtExpr {
        SmtExpr::bv_lit(id as u64, self.width)
    }

    /// Encode security score for a cipher suite variable.
    pub fn encode_score_lookup(&self, cipher_var: SmtExpr) -> SmtExpr {
        let mut result = SmtExpr::bv_lit(0, 16);
        for (&id, &score) in &self.security_scores {
            let id_expr = SmtExpr::bv_lit(id as u64, self.width);
            let score_expr = SmtExpr::bv_lit(score as u64, 16);
            let cond = SmtExpr::eq(cipher_var.clone(), id_expr);
            result = SmtExpr::ite(cond, score_expr, result);
        }
        result
    }

    /// Encode "cipher is in set" as a disjunction of equalities.
    pub fn encode_membership(&self, cipher_var: SmtExpr, set: &BTreeSet<u16>) -> SmtExpr {
        if set.is_empty() {
            return SmtExpr::BoolLit(false);
        }
        let clauses: Vec<SmtExpr> = set
            .iter()
            .map(|&id| SmtExpr::eq(cipher_var.clone(), SmtExpr::bv_lit(id as u64, self.width)))
            .collect();
        SmtExpr::or(clauses)
    }

    /// Encode "selected cipher is strictly weaker than expected".
    pub fn encode_cipher_downgrade(
        &self,
        selected_var: SmtExpr,
        expected_var: SmtExpr,
    ) -> SmtExpr {
        let sel_score = self.encode_score_lookup(selected_var);
        let exp_score = self.encode_score_lookup(expected_var);
        SmtExpr::bv_ult(sel_score, exp_score)
    }

    /// Encode "strongest cipher in set" as the maximum score lookup.
    pub fn encode_strongest_in_set(&self, set: &BTreeSet<u16>) -> Option<SmtExpr> {
        set.iter()
            .filter_map(|id| self.security_scores.get(id).map(|s| (*id, *s)))
            .max_by_key(|(_, score)| *score)
            .map(|(id, _)| SmtExpr::bv_lit(id as u64, self.width))
    }

    /// Declarations needed for cipher suite encoding.
    pub fn declarations(&self) -> Vec<SmtDeclaration> {
        vec![SmtDeclaration::DeclareFun {
            name: "cipher_security_score".to_string(),
            args: vec![SmtSort::BitVec(self.width)],
            ret: SmtSort::BitVec(16),
        }]
    }

    /// Axioms constraining the security score function.
    pub fn score_axioms(&self) -> Vec<SmtExpr> {
        let mut axioms = Vec::new();
        for (&id, &score) in &self.security_scores {
            let id_expr = SmtExpr::bv_lit(id as u64, self.width);
            let score_expr = SmtExpr::bv_lit(score as u64, 16);
            let app = SmtExpr::Apply(
                "cipher_security_score".to_string(),
                vec![id_expr],
            );
            axioms.push(SmtExpr::eq(app, score_expr));
        }
        axioms
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn known_ids(&self) -> &BTreeSet<u16> {
        &self.known_ids
    }
}

// ─── Version bitvector encoder ──────────────────────────────────────────

/// Encodes protocol version operations as bitvector constraints.
#[derive(Debug, Clone)]
pub struct VersionEncoder {
    width: u32,
    version_security: IndexMap<u16, u32>,
}

impl VersionEncoder {
    pub fn new() -> Self {
        let mut version_security = IndexMap::new();
        // TLS versions: major.minor as wire encoding
        version_security.insert(0x0300, 0); // SSL 3.0
        version_security.insert(0x0301, 1); // TLS 1.0
        version_security.insert(0x0302, 2); // TLS 1.1
        version_security.insert(0x0303, 3); // TLS 1.2
        version_security.insert(0x0304, 4); // TLS 1.3
        // SSH
        version_security.insert(0x0200, 3); // SSH 2.0
        VersionEncoder {
            width: 16,
            version_security,
        }
    }

    pub fn encode_version(&self, wire_value: u16) -> SmtExpr {
        SmtExpr::bv_lit(wire_value as u64, self.width)
    }

    pub fn encode_security_level(&self, version_var: SmtExpr) -> SmtExpr {
        let mut result = SmtExpr::bv_lit(0, 4);
        for (&ver, &level) in &self.version_security {
            let ver_expr = SmtExpr::bv_lit(ver as u64, self.width);
            let level_expr = SmtExpr::bv_lit(level as u64, 4);
            let cond = SmtExpr::eq(version_var.clone(), ver_expr);
            result = SmtExpr::ite(cond, level_expr, result);
        }
        result
    }

    pub fn encode_version_downgrade(
        &self,
        selected_var: SmtExpr,
        expected_var: SmtExpr,
    ) -> SmtExpr {
        let sel_level = self.encode_security_level(selected_var);
        let exp_level = self.encode_security_level(expected_var);
        SmtExpr::bv_ult(sel_level, exp_level)
    }

    pub fn encode_membership(&self, version_var: SmtExpr, versions: &BTreeSet<u16>) -> SmtExpr {
        if versions.is_empty() {
            return SmtExpr::BoolLit(false);
        }
        let clauses: Vec<SmtExpr> = versions
            .iter()
            .map(|&v| SmtExpr::eq(version_var.clone(), SmtExpr::bv_lit(v as u64, self.width)))
            .collect();
        SmtExpr::or(clauses)
    }

    pub fn encode_max_version(&self, versions: &BTreeSet<u16>) -> Option<SmtExpr> {
        versions
            .iter()
            .filter_map(|v| self.version_security.get(v).map(|s| (*v, *s)))
            .max_by_key(|(_, sec)| *sec)
            .map(|(v, _)| SmtExpr::bv_lit(v as u64, self.width))
    }

    pub fn width(&self) -> u32 {
        self.width
    }
}

impl Default for VersionEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Set encoding via bitvector arrays ──────────────────────────────────

/// Encodes sets of values as bitvector arrays (Array BitVec[16] Bool).
#[derive(Debug, Clone)]
pub struct BvSetEncoder {
    element_width: u32,
    prefix: String,
    counter: u32,
}

impl BvSetEncoder {
    pub fn new(element_width: u32, prefix: impl Into<String>) -> Self {
        BvSetEncoder {
            element_width,
            prefix: prefix.into(),
            counter: 0,
        }
    }

    /// Create a fresh set variable.
    pub fn fresh_set_var(&mut self) -> (String, SmtDeclaration) {
        let name = format!("{}_{}", self.prefix, self.counter);
        self.counter += 1;
        let decl = SmtDeclaration::DeclareConst {
            name: name.clone(),
            sort: SmtSort::Array(
                Box::new(SmtSort::BitVec(self.element_width)),
                Box::new(SmtSort::Bool),
            ),
        };
        (name, decl)
    }

    /// Encode "element ∈ set" as array select.
    pub fn encode_contains(&self, set_var: &str, element: SmtExpr) -> SmtExpr {
        SmtExpr::select(SmtExpr::var(set_var), element)
    }

    /// Encode "set is exactly {v1, v2, ...}" by constraining every element.
    pub fn encode_exact_set(&self, set_var: &str, elements: &[u16]) -> SmtExpr {
        let elem_set: BTreeSet<u16> = elements.iter().copied().collect();
        let mut constraints = Vec::new();

        // For each known element, it's in the set iff it's in our element list
        for &elem in &elem_set {
            let elem_expr = SmtExpr::bv_lit(elem as u64, self.element_width);
            let select = SmtExpr::select(SmtExpr::var(set_var), elem_expr);
            constraints.push(select);
        }

        SmtExpr::and(constraints)
    }

    /// Encode set inclusion: every element of A is in B.
    pub fn encode_subset(&self, set_a: &str, set_b: &str) -> SmtExpr {
        let x = format!("__subset_elem_{}", self.counter);
        let elem_var = SmtExpr::var(&x);
        let a_contains = SmtExpr::select(SmtExpr::var(set_a), elem_var.clone());
        let b_contains = SmtExpr::select(SmtExpr::var(set_b), elem_var);
        SmtExpr::ForAll(
            vec![(x, SmtSort::BitVec(self.element_width))],
            Box::new(SmtExpr::implies(a_contains, b_contains)),
        )
    }

    /// Encode set intersection non-emptiness.
    pub fn encode_intersection_nonempty(&self, set_a: &str, set_b: &str) -> SmtExpr {
        let x = format!("__intersect_elem_{}", self.counter);
        let elem_var = SmtExpr::var(&x);
        let a_contains = SmtExpr::select(SmtExpr::var(set_a), elem_var.clone());
        let b_contains = SmtExpr::select(SmtExpr::var(set_b), elem_var);
        SmtExpr::Exists(
            vec![(x, SmtSort::BitVec(self.element_width))],
            Box::new(SmtExpr::and(vec![a_contains, b_contains])),
        )
    }

    /// Encode set equality.
    pub fn encode_set_eq(&self, set_a: &str, set_b: &str) -> SmtExpr {
        let x = format!("__seteq_elem_{}", self.counter);
        let elem_var = SmtExpr::var(&x);
        let a_contains = SmtExpr::select(SmtExpr::var(set_a), elem_var.clone());
        let b_contains = SmtExpr::select(SmtExpr::var(set_b), elem_var);
        SmtExpr::ForAll(
            vec![(x, SmtSort::BitVec(self.element_width))],
            Box::new(SmtExpr::eq(a_contains, b_contains)),
        )
    }
}

// ─── Cardinality constraints ────────────────────────────────────────────

/// Encodes cardinality constraints over boolean/bitvector variables.
#[derive(Debug, Clone)]
pub struct CardinalityEncoder;

impl CardinalityEncoder {
    /// Encode "at most k of these booleans are true" using a sequential counter.
    pub fn at_most_k(vars: &[SmtExpr], k: u32) -> SmtExpr {
        if vars.is_empty() || k as usize >= vars.len() {
            return SmtExpr::BoolLit(true);
        }
        if k == 0 {
            return SmtExpr::and(vars.iter().map(|v| SmtExpr::not(v.clone())).collect());
        }

        // Use integer sum encoding: sum of ite(var, 1, 0) <= k
        let summands: Vec<SmtExpr> = vars
            .iter()
            .map(|v| SmtExpr::ite(v.clone(), SmtExpr::IntLit(1), SmtExpr::IntLit(0)))
            .collect();
        let sum = SmtExpr::int_add(summands);
        SmtExpr::int_le(sum, SmtExpr::IntLit(k as i64))
    }

    /// Encode "exactly k of these booleans are true".
    pub fn exactly_k(vars: &[SmtExpr], k: u32) -> SmtExpr {
        if vars.is_empty() {
            return if k == 0 {
                SmtExpr::BoolLit(true)
            } else {
                SmtExpr::BoolLit(false)
            };
        }

        let summands: Vec<SmtExpr> = vars
            .iter()
            .map(|v| SmtExpr::ite(v.clone(), SmtExpr::IntLit(1), SmtExpr::IntLit(0)))
            .collect();
        let sum = SmtExpr::int_add(summands);
        SmtExpr::eq(sum, SmtExpr::IntLit(k as i64))
    }

    /// Encode "at least k of these booleans are true".
    pub fn at_least_k(vars: &[SmtExpr], k: u32) -> SmtExpr {
        if k == 0 {
            return SmtExpr::BoolLit(true);
        }
        if k as usize > vars.len() {
            return SmtExpr::BoolLit(false);
        }

        let summands: Vec<SmtExpr> = vars
            .iter()
            .map(|v| SmtExpr::ite(v.clone(), SmtExpr::IntLit(1), SmtExpr::IntLit(0)))
            .collect();
        let sum = SmtExpr::int_add(summands);
        SmtExpr::IntGe(Box::new(sum), Box::new(SmtExpr::IntLit(k as i64)))
    }

    /// Encode "sum of bitvector values <= bound".
    pub fn bv_sum_le(vars: &[SmtExpr], width: u32, bound: u64) -> SmtExpr {
        if vars.is_empty() {
            return SmtExpr::BoolLit(true);
        }
        let mut sum = vars[0].clone();
        for v in &vars[1..] {
            sum = SmtExpr::bv_add(sum, v.clone());
        }
        SmtExpr::bv_ule(sum, SmtExpr::bv_lit(bound, width))
    }

    /// Pairwise encoding of at-most-one (for small sets).
    pub fn at_most_one_pairwise(vars: &[SmtExpr]) -> SmtExpr {
        if vars.len() <= 1 {
            return SmtExpr::BoolLit(true);
        }
        let mut clauses = Vec::new();
        for i in 0..vars.len() {
            for j in (i + 1)..vars.len() {
                clauses.push(SmtExpr::or(vec![
                    SmtExpr::not(vars[i].clone()),
                    SmtExpr::not(vars[j].clone()),
                ]));
            }
        }
        SmtExpr::and(clauses)
    }
}

// ─── Main BvEncoder ─────────────────────────────────────────────────────

/// Top-level bitvector encoding utility combining all sub-encoders.
#[derive(Debug, Clone)]
pub struct BvEncoder {
    pub cipher_encoder: CipherSuiteEncoder,
    pub version_encoder: VersionEncoder,
    pub security_encoder: SecurityLevelEncoder,
    var_counter: u32,
}

impl BvEncoder {
    pub fn new(cipher_ids: &[u16]) -> Self {
        BvEncoder {
            cipher_encoder: CipherSuiteEncoder::new(cipher_ids),
            version_encoder: VersionEncoder::new(),
            security_encoder: SecurityLevelEncoder::new(),
            var_counter: 0,
        }
    }

    /// Create a fresh bitvector variable with the given prefix and sort.
    pub fn fresh_bv_var(&mut self, prefix: &str, sort: BvSort) -> (String, SmtDeclaration) {
        let name = format!("{}_{}", prefix, self.var_counter);
        self.var_counter += 1;
        let decl = SmtDeclaration::DeclareConst {
            name: name.clone(),
            sort: sort.to_smt_sort(),
        };
        (name, decl)
    }

    /// Create a fresh boolean variable.
    pub fn fresh_bool_var(&mut self, prefix: &str) -> (String, SmtDeclaration) {
        let name = format!("{}_{}", prefix, self.var_counter);
        self.var_counter += 1;
        let decl = SmtDeclaration::DeclareConst {
            name: name.clone(),
            sort: SmtSort::Bool,
        };
        (name, decl)
    }

    /// Create a fresh integer variable.
    pub fn fresh_int_var(&mut self, prefix: &str) -> (String, SmtDeclaration) {
        let name = format!("{}_{}", prefix, self.var_counter);
        self.var_counter += 1;
        let decl = SmtDeclaration::DeclareConst {
            name: name.clone(),
            sort: SmtSort::Int,
        };
        (name, decl)
    }

    /// Encode a cipher suite state variable at a given time step.
    pub fn cipher_var_at(&self, step: u32) -> String {
        format!("cipher_t{}", step)
    }

    /// Encode a version state variable at a given time step.
    pub fn version_var_at(&self, step: u32) -> String {
        format!("version_t{}", step)
    }

    /// Encode a phase state variable at a given time step.
    pub fn phase_var_at(&self, step: u32) -> String {
        format!("phase_t{}", step)
    }

    /// Declare state variables for a single time step.
    pub fn declare_step_vars(&self, step: u32) -> Vec<SmtDeclaration> {
        vec![
            SmtDeclaration::DeclareConst {
                name: self.cipher_var_at(step),
                sort: BvSort::cipher_suite().to_smt_sort(),
            },
            SmtDeclaration::DeclareConst {
                name: self.version_var_at(step),
                sort: BvSort::version().to_smt_sort(),
            },
            SmtDeclaration::DeclareConst {
                name: self.phase_var_at(step),
                sort: BvSort::phase().to_smt_sort(),
            },
        ]
    }

    /// Encode phase ordering constraint: phase at step+1 >= phase at step.
    pub fn encode_phase_monotonicity(&self, step: u32) -> SmtExpr {
        let curr = SmtExpr::var(self.phase_var_at(step));
        let next = SmtExpr::var(self.phase_var_at(step + 1));
        SmtExpr::bv_ule(curr, next)
    }

    /// Encode that a cipher suite variable is valid (in the known set).
    pub fn encode_valid_cipher(&self, cipher_var: SmtExpr) -> SmtExpr {
        self.cipher_encoder
            .encode_membership(cipher_var, self.cipher_encoder.known_ids())
    }

    /// Encode that a version variable is one of the supported versions.
    pub fn encode_valid_version(
        &self,
        version_var: SmtExpr,
        supported: &BTreeSet<u16>,
    ) -> SmtExpr {
        self.version_encoder
            .encode_membership(version_var, supported)
    }

    /// Encode comparison: cipher a is strictly more secure than cipher b.
    pub fn encode_cipher_stronger(
        &self,
        cipher_a: SmtExpr,
        cipher_b: SmtExpr,
    ) -> SmtExpr {
        let score_a = self.cipher_encoder.encode_score_lookup(cipher_a);
        let score_b = self.cipher_encoder.encode_score_lookup(cipher_b);
        SmtExpr::BvUgt(Box::new(score_a), Box::new(score_b))
    }

    /// Encode comparison: version a is strictly higher than version b.
    pub fn encode_version_higher(
        &self,
        version_a: SmtExpr,
        version_b: SmtExpr,
    ) -> SmtExpr {
        let level_a = self.version_encoder.encode_security_level(version_a);
        let level_b = self.version_encoder.encode_security_level(version_b);
        SmtExpr::BvUgt(Box::new(level_a), Box::new(level_b))
    }

    /// Generate all base declarations for a k-step encoding.
    pub fn base_declarations(&self, depth: u32) -> Vec<SmtDeclaration> {
        let mut decls = Vec::new();
        for step in 0..=depth {
            decls.extend(self.declare_step_vars(step));
        }
        decls.extend(self.cipher_encoder.declarations());
        decls
    }

    /// Generate base axioms (score function definitions, etc.).
    pub fn base_axioms(&self) -> Vec<SmtExpr> {
        self.cipher_encoder.score_axioms()
    }

    pub fn var_counter(&self) -> u32 {
        self.var_counter
    }
}

// ─── Bitvector utility functions ────────────────────────────────────────

/// Compute minimum bit width needed to represent n distinct values.
pub fn bits_for_values(n: u64) -> u32 {
    if n <= 1 {
        return 1;
    }
    64 - (n - 1).leading_zeros()
}

/// Create a bitvector mask with bits [hi:lo] set.
pub fn bv_mask(hi: u32, lo: u32, width: u32) -> u64 {
    assert!(hi < width && lo <= hi);
    let range = hi - lo + 1;
    let mask = if range >= 64 {
        u64::MAX
    } else {
        (1u64 << range) - 1
    };
    mask << lo
}

/// One-hot encoding: exactly one of n booleans is true.
pub fn one_hot(n: u32) -> Vec<SmtExpr> {
    let mut constraints = Vec::new();
    let vars: Vec<SmtExpr> = (0..n).map(|i| SmtExpr::var(format!("oh_{}", i))).collect();
    // At least one
    constraints.push(SmtExpr::or(vars.clone()));
    // At most one (pairwise)
    constraints.push(CardinalityEncoder::at_most_one_pairwise(&vars));
    constraints
}

/// Encode population count (hamming weight) of a bitvector as an integer.
pub fn popcount_to_int(bv_var: &str, width: u32) -> SmtExpr {
    let bits: Vec<SmtExpr> = (0..width)
        .map(|i| {
            let bit = SmtExpr::Extract(Box::new(SmtExpr::var(bv_var)), i, i);
            SmtExpr::ite(
                SmtExpr::eq(bit, SmtExpr::bv_lit(1, 1)),
                SmtExpr::IntLit(1),
                SmtExpr::IntLit(0),
            )
        })
        .collect();
    SmtExpr::int_add(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bv_sort_basics() {
        let cs = BvSort::cipher_suite();
        assert_eq!(cs.width, 16);
        assert_eq!(cs.max_value(), 65535);
        assert_eq!(format!("{}", cs.to_smt_sort()), "(_ BitVec 16)");
    }

    #[test]
    fn test_cipher_suite_encoder() {
        let enc = CipherSuiteEncoder::new(&[0x002F, 0x009C, 0x1302]);
        let id = enc.encode_id(0x002F);
        assert_eq!(format!("{}", id), "(_ bv47 16)");

        let mut set = BTreeSet::new();
        set.insert(0x002F);
        set.insert(0x009C);
        let membership = enc.encode_membership(SmtExpr::var("c"), &set);
        let s = format!("{}", membership);
        assert!(s.contains("or"));
    }

    #[test]
    fn test_version_encoder() {
        let enc = VersionEncoder::new();
        let v = enc.encode_version(0x0303);
        assert_eq!(format!("{}", v), "(_ bv771 16)");

        let mut versions = BTreeSet::new();
        versions.insert(0x0303);
        versions.insert(0x0304);
        let max = enc.encode_max_version(&versions);
        assert!(max.is_some());
    }

    #[test]
    fn test_security_level_encoder() {
        let enc = SecurityLevelEncoder::new();
        let high = enc.encode_level("high");
        assert_eq!(format!("{}", high), "(_ bv4 4)");

        let is_sec = enc.is_secure(SmtExpr::var("level"));
        let s = format!("{}", is_sec);
        assert!(s.contains("bvule"));
    }

    #[test]
    fn test_cardinality_at_most_k() {
        let vars = vec![SmtExpr::var("a"), SmtExpr::var("b"), SmtExpr::var("c")];
        let result = CardinalityEncoder::at_most_k(&vars, 1);
        let s = format!("{}", result);
        assert!(s.contains("<="));
    }

    #[test]
    fn test_cardinality_exactly_k() {
        let vars = vec![SmtExpr::var("a"), SmtExpr::var("b")];
        let result = CardinalityEncoder::exactly_k(&vars, 1);
        let s = format!("{}", result);
        assert!(s.contains("="));
    }

    #[test]
    fn test_at_most_one_pairwise() {
        let vars = vec![SmtExpr::var("a"), SmtExpr::var("b"), SmtExpr::var("c")];
        let result = CardinalityEncoder::at_most_one_pairwise(&vars);
        match &result {
            SmtExpr::And(clauses) => assert_eq!(clauses.len(), 3),
            _ => panic!("expected And"),
        }
    }

    #[test]
    fn test_bits_for_values() {
        assert_eq!(bits_for_values(1), 1);
        assert_eq!(bits_for_values(2), 1);
        assert_eq!(bits_for_values(3), 2);
        assert_eq!(bits_for_values(4), 2);
        assert_eq!(bits_for_values(5), 3);
        assert_eq!(bits_for_values(256), 8);
        assert_eq!(bits_for_values(257), 9);
    }

    #[test]
    fn test_bv_mask() {
        assert_eq!(bv_mask(3, 0, 8), 0x0F);
        assert_eq!(bv_mask(7, 4, 8), 0xF0);
        assert_eq!(bv_mask(7, 0, 8), 0xFF);
    }

    #[test]
    fn test_bv_encoder_fresh_vars() {
        let mut enc = BvEncoder::new(&[0x002F]);
        let (name1, _) = enc.fresh_bv_var("test", BvSort::cipher_suite());
        let (name2, _) = enc.fresh_bv_var("test", BvSort::cipher_suite());
        assert_ne!(name1, name2);
    }

    #[test]
    fn test_phase_monotonicity() {
        let enc = BvEncoder::new(&[]);
        let mono = enc.encode_phase_monotonicity(0);
        let s = format!("{}", mono);
        assert!(s.contains("phase_t0"));
        assert!(s.contains("phase_t1"));
        assert!(s.contains("bvule"));
    }

    #[test]
    fn test_set_encoder_contains() {
        let enc = BvSetEncoder::new(16, "cs_set");
        let result = enc.encode_contains("cs_set_0", SmtExpr::bv_lit(0x002F, 16));
        let s = format!("{}", result);
        assert!(s.contains("select"));
    }

    #[test]
    fn test_set_encoder_subset() {
        let enc = BvSetEncoder::new(16, "cs_set");
        let result = enc.encode_subset("A", "B");
        let s = format!("{}", result);
        assert!(s.contains("forall"));
        assert!(s.contains("=>"));
    }

    #[test]
    fn test_popcount() {
        let result = popcount_to_int("bv", 4);
        let s = format!("{}", result);
        assert!(s.contains("extract"));
    }

    #[test]
    fn test_cipher_downgrade_encoding() {
        let enc = CipherSuiteEncoder::new(&[0x002F, 0x1302]);
        let downgrade = enc.encode_cipher_downgrade(SmtExpr::var("sel"), SmtExpr::var("exp"));
        let s = format!("{}", downgrade);
        assert!(s.contains("bvult"));
    }

    #[test]
    fn test_strongest_in_set() {
        let enc = CipherSuiteEncoder::new(&[0x002F, 0x009C, 0x1302]);
        let mut set = BTreeSet::new();
        set.insert(0x002F);
        set.insert(0x1302);
        let strongest = enc.encode_strongest_in_set(&set);
        assert!(strongest.is_some());
    }

    #[test]
    fn test_bv_encoder_base_declarations() {
        let enc = BvEncoder::new(&[0x002F, 0x009C]);
        let decls = enc.base_declarations(3);
        // 4 steps * 3 vars each + cipher decls
        assert!(decls.len() >= 12);
    }

    #[test]
    fn test_one_hot() {
        let constraints = one_hot(3);
        assert_eq!(constraints.len(), 2);
    }

    #[test]
    fn test_bv_sum_le() {
        let vars = vec![SmtExpr::var("a"), SmtExpr::var("b")];
        let result = CardinalityEncoder::bv_sum_le(&vars, 8, 5);
        let s = format!("{}", result);
        assert!(s.contains("bvule"));
    }
}
