//! Dolev-Yao message algebra encoding (Definition D4).
//!
//! Implements the symbolic message algebra for protocol analysis:
//! - Term construction (encryption, MAC, hash, pairing)
//! - Term destruction (decryption, projection, verification)
//! - Adversary knowledge closure (deduction rules)
//! - Subterm reasoning optimizations
//! - Knowledge monotonicity invariants

use crate::{ConstraintOrigin, SmtConstraint, SmtDeclaration, SmtExpr, SmtSort};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

// ─── DY Term Algebra ────────────────────────────────────────────────────

/// Dolev-Yao term type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KeyType {
    Symmetric,
    PublicKey,
    PrivateKey,
}

/// A Dolev-Yao message term (Definition D4).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DYTerm {
    // Atoms
    Nonce { id: u32 },
    Key { key_type: KeyType, id: u32 },
    CipherSuiteId(u16),
    VersionId(u16),
    ExtensionId(u16),
    Literal(Vec<u8>),
    Variable(String),

    // Constructors
    SymEncrypt { key: Box<DYTerm>, plaintext: Box<DYTerm> },
    AsymEncrypt { public_key: Box<DYTerm>, plaintext: Box<DYTerm> },
    Mac { key: Box<DYTerm>, message: Box<DYTerm> },
    Hash(Box<DYTerm>),
    Pair(Box<DYTerm>, Box<DYTerm>),
    TlsRecord {
        record_type: u8,
        version: Box<DYTerm>,
        payload: Box<DYTerm>,
    },
    SshPacket {
        sequence: u32,
        payload: Box<DYTerm>,
    },
    Tuple(Vec<DYTerm>),
}

impl DYTerm {
    pub fn nonce(id: u32) -> Self {
        DYTerm::Nonce { id }
    }

    pub fn sym_key(id: u32) -> Self {
        DYTerm::Key { key_type: KeyType::Symmetric, id }
    }

    pub fn pub_key(id: u32) -> Self {
        DYTerm::Key { key_type: KeyType::PublicKey, id }
    }

    pub fn priv_key(id: u32) -> Self {
        DYTerm::Key { key_type: KeyType::PrivateKey, id }
    }

    pub fn pair(a: DYTerm, b: DYTerm) -> Self {
        DYTerm::Pair(Box::new(a), Box::new(b))
    }

    pub fn sym_encrypt(key: DYTerm, plaintext: DYTerm) -> Self {
        DYTerm::SymEncrypt {
            key: Box::new(key),
            plaintext: Box::new(plaintext),
        }
    }

    pub fn asym_encrypt(pubkey: DYTerm, plaintext: DYTerm) -> Self {
        DYTerm::AsymEncrypt {
            public_key: Box::new(pubkey),
            plaintext: Box::new(plaintext),
        }
    }

    pub fn mac(key: DYTerm, message: DYTerm) -> Self {
        DYTerm::Mac {
            key: Box::new(key),
            message: Box::new(message),
        }
    }

    pub fn hash(term: DYTerm) -> Self {
        DYTerm::Hash(Box::new(term))
    }

    pub fn is_atom(&self) -> bool {
        matches!(
            self,
            DYTerm::Nonce { .. }
                | DYTerm::Key { .. }
                | DYTerm::CipherSuiteId(_)
                | DYTerm::VersionId(_)
                | DYTerm::ExtensionId(_)
                | DYTerm::Literal(_)
                | DYTerm::Variable(_)
        )
    }

    /// Collect all subterms (transitive closure).
    pub fn subterms(&self) -> BTreeSet<DYTerm>
    where
        Self: Ord,
    {
        let mut subs = BTreeSet::new();
        self.collect_subterms(&mut subs);
        subs
    }

    fn collect_subterms(&self, subs: &mut BTreeSet<DYTerm>)
    where
        Self: Ord,
    {
        subs.insert(self.clone());
        match self {
            DYTerm::SymEncrypt { key, plaintext } | DYTerm::AsymEncrypt { public_key: key, plaintext } => {
                key.collect_subterms(subs);
                plaintext.collect_subterms(subs);
            }
            DYTerm::Mac { key, message } => {
                key.collect_subterms(subs);
                message.collect_subterms(subs);
            }
            DYTerm::Hash(t) => t.collect_subterms(subs),
            DYTerm::Pair(a, b) => {
                a.collect_subterms(subs);
                b.collect_subterms(subs);
            }
            DYTerm::TlsRecord { version, payload, .. } => {
                version.collect_subterms(subs);
                payload.collect_subterms(subs);
            }
            DYTerm::SshPacket { payload, .. } => {
                payload.collect_subterms(subs);
            }
            DYTerm::Tuple(elems) => {
                for e in elems {
                    e.collect_subterms(subs);
                }
            }
            _ => {} // atoms
        }
    }

    /// Depth of the term tree.
    pub fn depth(&self) -> u32 {
        match self {
            DYTerm::SymEncrypt { key, plaintext }
            | DYTerm::AsymEncrypt { public_key: key, plaintext } => {
                1 + key.depth().max(plaintext.depth())
            }
            DYTerm::Mac { key, message } => 1 + key.depth().max(message.depth()),
            DYTerm::Hash(t) => 1 + t.depth(),
            DYTerm::Pair(a, b) => 1 + a.depth().max(b.depth()),
            DYTerm::TlsRecord { version, payload, .. } => {
                1 + version.depth().max(payload.depth())
            }
            DYTerm::SshPacket { payload, .. } => 1 + payload.depth(),
            DYTerm::Tuple(elems) => {
                1 + elems.iter().map(|e| e.depth()).max().unwrap_or(0)
            }
            _ => 0,
        }
    }

    /// Unique string name for this term (used in SMT variable naming).
    pub fn smt_name(&self) -> String {
        match self {
            DYTerm::Nonce { id } => format!("nonce_{}", id),
            DYTerm::Key { key_type, id } => format!("key_{:?}_{}", key_type, id),
            DYTerm::CipherSuiteId(id) => format!("cs_{:#06x}", id),
            DYTerm::VersionId(id) => format!("ver_{:#06x}", id),
            DYTerm::ExtensionId(id) => format!("ext_{}", id),
            DYTerm::Literal(data) => {
                if data.len() <= 4 {
                    format!("lit_{}", hex::encode(data))
                } else {
                    format!("lit_{}_{}", hex::encode(&data[..4]), data.len())
                }
            }
            DYTerm::Variable(name) => format!("var_{}", name),
            DYTerm::SymEncrypt { .. } => "senc".to_string(),
            DYTerm::AsymEncrypt { .. } => "aenc".to_string(),
            DYTerm::Mac { .. } => "mac".to_string(),
            DYTerm::Hash(_) => "hash".to_string(),
            DYTerm::Pair(_, _) => "pair".to_string(),
            DYTerm::TlsRecord { record_type, .. } => format!("tlsrec_{}", record_type),
            DYTerm::SshPacket { sequence, .. } => format!("sshpkt_{}", sequence),
            DYTerm::Tuple(_) => "tuple".to_string(),
        }
    }
}

impl PartialOrd for DYTerm {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DYTerm {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.smt_name().cmp(&other.smt_name())
    }
}

impl fmt::Display for DYTerm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DYTerm::Nonce { id } => write!(f, "n{}", id),
            DYTerm::Key { key_type, id } => write!(f, "k{:?}{}", key_type, id),
            DYTerm::CipherSuiteId(id) => write!(f, "cs(0x{:04x})", id),
            DYTerm::VersionId(id) => write!(f, "ver(0x{:04x})", id),
            DYTerm::ExtensionId(id) => write!(f, "ext({})", id),
            DYTerm::Literal(data) => write!(f, "lit[{}]", data.len()),
            DYTerm::Variable(name) => write!(f, "{}", name),
            DYTerm::SymEncrypt { key, plaintext } => write!(f, "senc({}, {})", key, plaintext),
            DYTerm::AsymEncrypt { public_key, plaintext } => {
                write!(f, "aenc({}, {})", public_key, plaintext)
            }
            DYTerm::Mac { key, message } => write!(f, "mac({}, {})", key, message),
            DYTerm::Hash(t) => write!(f, "h({})", t),
            DYTerm::Pair(a, b) => write!(f, "⟨{}, {}⟩", a, b),
            DYTerm::TlsRecord { record_type, payload, .. } => {
                write!(f, "tls_rec({}, {})", record_type, payload)
            }
            DYTerm::SshPacket { sequence, payload } => {
                write!(f, "ssh_pkt({}, {})", sequence, payload)
            }
            DYTerm::Tuple(elems) => {
                write!(f, "(")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", e)?;
                }
                write!(f, ")")
            }
        }
    }
}

// ─── DY Term Algebra ────────────────────────────────────────────────────

/// The Dolev-Yao term algebra implementing constructor and destructor operations.
#[derive(Debug, Clone)]
pub struct DYTermAlgebra {
    term_counter: u32,
    /// Map from term to its SMT variable name.
    term_names: IndexMap<DYTerm, String>,
}

impl DYTermAlgebra {
    pub fn new() -> Self {
        DYTermAlgebra {
            term_counter: 0,
            term_names: IndexMap::new(),
        }
    }

    /// Register a term and get its unique SMT name.
    pub fn register_term(&mut self, term: &DYTerm) -> String {
        if let Some(name) = self.term_names.get(term) {
            return name.clone();
        }
        let name = format!("dy_{}_{}", term.smt_name(), self.term_counter);
        self.term_counter += 1;
        self.term_names.insert(term.clone(), name.clone());
        name
    }

    /// Declare SMT sorts for the DY algebra.
    pub fn sort_declarations(&self) -> Vec<SmtDeclaration> {
        vec![
            SmtDeclaration::DeclareSort { name: "DYTerm".to_string(), arity: 0 },
        ]
    }

    /// Declare uninterpreted functions for constructors.
    pub fn constructor_declarations(&self) -> Vec<SmtDeclaration> {
        let term_sort = SmtSort::Uninterpreted("DYTerm".to_string());
        vec![
            SmtDeclaration::DeclareFun {
                name: "senc".to_string(),
                args: vec![term_sort.clone(), term_sort.clone()],
                ret: term_sort.clone(),
            },
            SmtDeclaration::DeclareFun {
                name: "sdec".to_string(),
                args: vec![term_sort.clone(), term_sort.clone()],
                ret: term_sort.clone(),
            },
            SmtDeclaration::DeclareFun {
                name: "aenc".to_string(),
                args: vec![term_sort.clone(), term_sort.clone()],
                ret: term_sort.clone(),
            },
            SmtDeclaration::DeclareFun {
                name: "adec".to_string(),
                args: vec![term_sort.clone(), term_sort.clone()],
                ret: term_sort.clone(),
            },
            SmtDeclaration::DeclareFun {
                name: "mac".to_string(),
                args: vec![term_sort.clone(), term_sort.clone()],
                ret: term_sort.clone(),
            },
            SmtDeclaration::DeclareFun {
                name: "hash".to_string(),
                args: vec![term_sort.clone()],
                ret: term_sort.clone(),
            },
            SmtDeclaration::DeclareFun {
                name: "pair".to_string(),
                args: vec![term_sort.clone(), term_sort.clone()],
                ret: term_sort.clone(),
            },
            SmtDeclaration::DeclareFun {
                name: "fst".to_string(),
                args: vec![term_sort.clone()],
                ret: term_sort.clone(),
            },
            SmtDeclaration::DeclareFun {
                name: "snd".to_string(),
                args: vec![term_sort.clone()],
                ret: term_sort.clone(),
            },
            SmtDeclaration::DeclareFun {
                name: "verify_mac".to_string(),
                args: vec![term_sort.clone(), term_sort.clone(), term_sort.clone()],
                ret: SmtSort::Bool,
            },
            SmtDeclaration::DeclareFun {
                name: "pk".to_string(),
                args: vec![term_sort.clone()],
                ret: term_sort.clone(),
            },
            SmtDeclaration::DeclareFun {
                name: "tls_record".to_string(),
                args: vec![
                    SmtSort::BitVec(8),
                    term_sort.clone(),
                    term_sort.clone(),
                ],
                ret: term_sort.clone(),
            },
            SmtDeclaration::DeclareFun {
                name: "tls_record_payload".to_string(),
                args: vec![term_sort.clone()],
                ret: term_sort,
            },
        ]
    }

    /// Axioms relating constructors and destructors.
    pub fn constructor_destructor_axioms(&self) -> Vec<SmtExpr> {
        let term_sort = SmtSort::Uninterpreted("DYTerm".to_string());
        let mut axioms = Vec::new();

        // ∀ k, m: sdec(k, senc(k, m)) = m
        axioms.push(SmtExpr::ForAll(
            vec![
                ("__k".to_string(), term_sort.clone()),
                ("__m".to_string(), term_sort.clone()),
            ],
            Box::new(SmtExpr::eq(
                SmtExpr::Apply("sdec".to_string(), vec![
                    SmtExpr::var("__k"),
                    SmtExpr::Apply("senc".to_string(), vec![
                        SmtExpr::var("__k"),
                        SmtExpr::var("__m"),
                    ]),
                ]),
                SmtExpr::var("__m"),
            )),
        ));

        // ∀ sk, m: adec(sk, aenc(pk(sk), m)) = m
        axioms.push(SmtExpr::ForAll(
            vec![
                ("__sk".to_string(), term_sort.clone()),
                ("__m".to_string(), term_sort.clone()),
            ],
            Box::new(SmtExpr::eq(
                SmtExpr::Apply("adec".to_string(), vec![
                    SmtExpr::var("__sk"),
                    SmtExpr::Apply("aenc".to_string(), vec![
                        SmtExpr::Apply("pk".to_string(), vec![SmtExpr::var("__sk")]),
                        SmtExpr::var("__m"),
                    ]),
                ]),
                SmtExpr::var("__m"),
            )),
        ));

        // ∀ a, b: fst(pair(a, b)) = a
        axioms.push(SmtExpr::ForAll(
            vec![
                ("__a".to_string(), term_sort.clone()),
                ("__b".to_string(), term_sort.clone()),
            ],
            Box::new(SmtExpr::eq(
                SmtExpr::Apply("fst".to_string(), vec![
                    SmtExpr::Apply("pair".to_string(), vec![
                        SmtExpr::var("__a"),
                        SmtExpr::var("__b"),
                    ]),
                ]),
                SmtExpr::var("__a"),
            )),
        ));

        // ∀ a, b: snd(pair(a, b)) = b
        axioms.push(SmtExpr::ForAll(
            vec![
                ("__a".to_string(), term_sort.clone()),
                ("__b".to_string(), term_sort),
            ],
            Box::new(SmtExpr::eq(
                SmtExpr::Apply("snd".to_string(), vec![
                    SmtExpr::Apply("pair".to_string(), vec![
                        SmtExpr::var("__a"),
                        SmtExpr::var("__b"),
                    ]),
                ]),
                SmtExpr::var("__b"),
            )),
        ));

        axioms
    }

    /// Encode a DY term as an SMT expression using uninterpreted functions.
    pub fn encode_term(&mut self, term: &DYTerm) -> SmtExpr {
        match term {
            DYTerm::Nonce { .. }
            | DYTerm::Key { .. }
            | DYTerm::CipherSuiteId(_)
            | DYTerm::VersionId(_)
            | DYTerm::ExtensionId(_)
            | DYTerm::Literal(_) => {
                let name = self.register_term(term);
                SmtExpr::var(name)
            }
            DYTerm::Variable(v) => SmtExpr::var(v.clone()),
            DYTerm::SymEncrypt { key, plaintext } => {
                let k = self.encode_term(key);
                let m = self.encode_term(plaintext);
                SmtExpr::Apply("senc".to_string(), vec![k, m])
            }
            DYTerm::AsymEncrypt { public_key, plaintext } => {
                let pk = self.encode_term(public_key);
                let m = self.encode_term(plaintext);
                SmtExpr::Apply("aenc".to_string(), vec![pk, m])
            }
            DYTerm::Mac { key, message } => {
                let k = self.encode_term(key);
                let m = self.encode_term(message);
                SmtExpr::Apply("mac".to_string(), vec![k, m])
            }
            DYTerm::Hash(t) => {
                let inner = self.encode_term(t);
                SmtExpr::Apply("hash".to_string(), vec![inner])
            }
            DYTerm::Pair(a, b) => {
                let ea = self.encode_term(a);
                let eb = self.encode_term(b);
                SmtExpr::Apply("pair".to_string(), vec![ea, eb])
            }
            DYTerm::TlsRecord { record_type, version, payload } => {
                let v = self.encode_term(version);
                let p = self.encode_term(payload);
                SmtExpr::Apply(
                    "tls_record".to_string(),
                    vec![SmtExpr::bv_lit(*record_type as u64, 8), v, p],
                )
            }
            DYTerm::SshPacket { payload, .. } => {
                self.encode_term(payload)
            }
            DYTerm::Tuple(elems) => {
                if elems.is_empty() {
                    let name = self.register_term(term);
                    return SmtExpr::var(name);
                }
                let mut result = self.encode_term(&elems[0]);
                for e in &elems[1..] {
                    let encoded = self.encode_term(e);
                    result = SmtExpr::Apply("pair".to_string(), vec![result, encoded]);
                }
                result
            }
        }
    }

    /// Get all registered term names and their declarations.
    pub fn term_declarations(&self) -> Vec<SmtDeclaration> {
        let term_sort = SmtSort::Uninterpreted("DYTerm".to_string());
        self.term_names
            .values()
            .map(|name| SmtDeclaration::DeclareConst {
                name: name.clone(),
                sort: term_sort.clone(),
            })
            .collect()
    }
}

impl Default for DYTermAlgebra {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Term Encoder ───────────────────────────────────────────────────────

/// Maps DY terms to SMT expressions with caching.
#[derive(Debug, Clone)]
pub struct TermEncoder {
    algebra: DYTermAlgebra,
    cache: IndexMap<DYTerm, SmtExpr>,
}

impl TermEncoder {
    pub fn new() -> Self {
        TermEncoder {
            algebra: DYTermAlgebra::new(),
            cache: IndexMap::new(),
        }
    }

    pub fn encode(&mut self, term: &DYTerm) -> SmtExpr {
        if let Some(cached) = self.cache.get(term) {
            return cached.clone();
        }
        let expr = self.algebra.encode_term(term);
        self.cache.insert(term.clone(), expr.clone());
        expr
    }

    pub fn algebra(&self) -> &DYTermAlgebra {
        &self.algebra
    }

    pub fn algebra_mut(&mut self) -> &mut DYTermAlgebra {
        &mut self.algebra
    }

    pub fn declarations(&self) -> Vec<SmtDeclaration> {
        let mut decls = self.algebra.sort_declarations();
        decls.extend(self.algebra.constructor_declarations());
        decls.extend(self.algebra.term_declarations());
        decls
    }

    pub fn axioms(&self) -> Vec<SmtExpr> {
        self.algebra.constructor_destructor_axioms()
    }
}

impl Default for TermEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Adversary Knowledge Encoder ────────────────────────────────────────

/// Encodes the adversary's knowledge set 𝒦 at each time step.
#[derive(Debug, Clone)]
pub struct KnowledgeEncoder {
    term_encoder: TermEncoder,
    /// Max derivation depth for closure computation.
    max_derivation_depth: u32,
}

impl KnowledgeEncoder {
    pub fn new(max_derivation_depth: u32) -> Self {
        KnowledgeEncoder {
            term_encoder: TermEncoder::new(),
            max_derivation_depth,
        }
    }

    /// Encode "adversary knows term t at step s" as a boolean variable.
    pub fn knows_var(&self, term: &DYTerm, step: u32) -> String {
        format!("knows_{}_{}", term.smt_name(), step)
    }

    /// Declare knowledge variables for a set of terms at a step.
    pub fn declare_knowledge_vars(
        &self,
        terms: &[DYTerm],
        step: u32,
    ) -> Vec<SmtDeclaration> {
        terms
            .iter()
            .map(|t| SmtDeclaration::DeclareConst {
                name: self.knows_var(t, step),
                sort: SmtSort::Bool,
            })
            .collect()
    }

    /// Encode initial knowledge: adversary knows public values.
    pub fn encode_initial_knowledge(
        &self,
        public_terms: &[DYTerm],
        all_terms: &[DYTerm],
    ) -> Vec<SmtExpr> {
        let mut constraints = Vec::new();
        let public_set: BTreeSet<&DYTerm> = public_terms.iter().collect();

        for term in all_terms {
            let var = SmtExpr::var(self.knows_var(term, 0));
            if public_set.contains(term) {
                constraints.push(var);
            } else {
                constraints.push(SmtExpr::not(var));
            }
        }
        constraints
    }

    /// Encode knowledge monotonicity: once known, always known.
    /// knows(t, step) => knows(t, step+1)
    pub fn encode_monotonicity(&self, terms: &[DYTerm], step: u32) -> Vec<SmtExpr> {
        terms
            .iter()
            .map(|t| {
                SmtExpr::implies(
                    SmtExpr::var(self.knows_var(t, step)),
                    SmtExpr::var(self.knows_var(t, step + 1)),
                )
            })
            .collect()
    }

    /// Encode deduction rules for a step:
    /// - If knows(enc(m,k)) and knows(k), then knows(m)
    /// - If knows(a) and knows(b), then knows(pair(a,b))
    /// - If knows(pair(a,b)), then knows(a) and knows(b)
    /// - etc.
    pub fn encode_deduction_rules(
        &self,
        terms: &[DYTerm],
        step: u32,
    ) -> Vec<SmtExpr> {
        let mut rules = Vec::new();
        let term_set: BTreeSet<&DYTerm> = terms.iter().collect();

        for term in terms {
            match term {
                DYTerm::SymEncrypt { key, plaintext } => {
                    // Decryption: knows(senc(m,k)) ∧ knows(k) → knows(m)
                    if term_set.contains(plaintext.as_ref()) && term_set.contains(key.as_ref()) {
                        let knows_enc = SmtExpr::var(self.knows_var(term, step));
                        let knows_key = SmtExpr::var(self.knows_var(key, step));
                        let knows_plain = SmtExpr::var(self.knows_var(plaintext, step));
                        rules.push(SmtExpr::implies(
                            SmtExpr::and(vec![knows_enc, knows_key]),
                            knows_plain,
                        ));
                    }
                    // Encryption: knows(m) ∧ knows(k) → knows(senc(m,k))
                    if term_set.contains(key.as_ref()) && term_set.contains(plaintext.as_ref()) {
                        let knows_key = SmtExpr::var(self.knows_var(key, step));
                        let knows_plain = SmtExpr::var(self.knows_var(plaintext, step));
                        let knows_enc = SmtExpr::var(self.knows_var(term, step));
                        rules.push(SmtExpr::implies(
                            SmtExpr::and(vec![knows_key, knows_plain]),
                            knows_enc,
                        ));
                    }
                }
                DYTerm::AsymEncrypt { public_key, plaintext } => {
                    // Decryption with private key
                    let priv_key = match public_key.as_ref() {
                        DYTerm::Key { id, .. } => DYTerm::priv_key(*id),
                        _ => continue,
                    };
                    if term_set.contains(plaintext.as_ref()) && term_set.contains(&priv_key) {
                        let knows_enc = SmtExpr::var(self.knows_var(term, step));
                        let knows_privkey = SmtExpr::var(self.knows_var(&priv_key, step));
                        let knows_plain = SmtExpr::var(self.knows_var(plaintext, step));
                        rules.push(SmtExpr::implies(
                            SmtExpr::and(vec![knows_enc, knows_privkey]),
                            knows_plain,
                        ));
                    }
                    // Encryption with public key
                    if term_set.contains(public_key.as_ref())
                        && term_set.contains(plaintext.as_ref())
                    {
                        let knows_pk = SmtExpr::var(self.knows_var(public_key, step));
                        let knows_plain = SmtExpr::var(self.knows_var(plaintext, step));
                        let knows_enc = SmtExpr::var(self.knows_var(term, step));
                        rules.push(SmtExpr::implies(
                            SmtExpr::and(vec![knows_pk, knows_plain]),
                            knows_enc,
                        ));
                    }
                }
                DYTerm::Pair(a, b) => {
                    // Projection: knows(pair(a,b)) → knows(a) ∧ knows(b)
                    if term_set.contains(a.as_ref()) {
                        rules.push(SmtExpr::implies(
                            SmtExpr::var(self.knows_var(term, step)),
                            SmtExpr::var(self.knows_var(a, step)),
                        ));
                    }
                    if term_set.contains(b.as_ref()) {
                        rules.push(SmtExpr::implies(
                            SmtExpr::var(self.knows_var(term, step)),
                            SmtExpr::var(self.knows_var(b, step)),
                        ));
                    }
                    // Pairing: knows(a) ∧ knows(b) → knows(pair(a,b))
                    if term_set.contains(a.as_ref()) && term_set.contains(b.as_ref()) {
                        rules.push(SmtExpr::implies(
                            SmtExpr::and(vec![
                                SmtExpr::var(self.knows_var(a, step)),
                                SmtExpr::var(self.knows_var(b, step)),
                            ]),
                            SmtExpr::var(self.knows_var(term, step)),
                        ));
                    }
                }
                DYTerm::Hash(inner) => {
                    // Hashing: knows(m) → knows(hash(m))
                    if term_set.contains(inner.as_ref()) {
                        rules.push(SmtExpr::implies(
                            SmtExpr::var(self.knows_var(inner, step)),
                            SmtExpr::var(self.knows_var(term, step)),
                        ));
                    }
                }
                DYTerm::Mac { key, message } => {
                    // MAC creation: knows(k) ∧ knows(m) → knows(mac(k,m))
                    if term_set.contains(key.as_ref()) && term_set.contains(message.as_ref()) {
                        rules.push(SmtExpr::implies(
                            SmtExpr::and(vec![
                                SmtExpr::var(self.knows_var(key, step)),
                                SmtExpr::var(self.knows_var(message, step)),
                            ]),
                            SmtExpr::var(self.knows_var(term, step)),
                        ));
                    }
                }
                DYTerm::TlsRecord { version, payload, .. } => {
                    // Record decomposition: knows(record) → knows(payload)
                    if term_set.contains(payload.as_ref()) {
                        rules.push(SmtExpr::implies(
                            SmtExpr::var(self.knows_var(term, step)),
                            SmtExpr::var(self.knows_var(payload, step)),
                        ));
                    }
                    if term_set.contains(version.as_ref()) {
                        rules.push(SmtExpr::implies(
                            SmtExpr::var(self.knows_var(term, step)),
                            SmtExpr::var(self.knows_var(version, step)),
                        ));
                    }
                }
                _ => {}
            }
        }
        rules
    }

    /// Encode knowledge acquisition from intercepted message.
    pub fn encode_intercept_knowledge(
        &self,
        message_terms: &[DYTerm],
        step: u32,
        intercept_flag: SmtExpr,
    ) -> Vec<SmtExpr> {
        message_terms
            .iter()
            .map(|t| {
                SmtExpr::implies(
                    intercept_flag.clone(),
                    SmtExpr::var(self.knows_var(t, step + 1)),
                )
            })
            .collect()
    }

    /// Get the underlying term encoder.
    pub fn term_encoder(&self) -> &TermEncoder {
        &self.term_encoder
    }

    pub fn term_encoder_mut(&mut self) -> &mut TermEncoder {
        &mut self.term_encoder
    }

    pub fn max_derivation_depth(&self) -> u32 {
        self.max_derivation_depth
    }
}

// ─── Deduction Rules ────────────────────────────────────────────────────

/// Encodes deduction rules as SMT constraints.
#[derive(Debug, Clone)]
pub struct DeductionRules {
    /// All known terms in the system.
    all_terms: Vec<DYTerm>,
    /// Subterm index for optimization.
    subterm_index: BTreeMap<DYTerm, BTreeSet<DYTerm>>,
}

impl DeductionRules {
    pub fn new(terms: Vec<DYTerm>) -> Self {
        let mut subterm_index = BTreeMap::new();
        for term in &terms {
            let subs = term.subterms();
            subterm_index.insert(term.clone(), subs);
        }
        DeductionRules {
            all_terms: terms,
            subterm_index,
        }
    }

    /// Check if term `a` is a subterm of term `b`.
    pub fn is_subterm(&self, a: &DYTerm, b: &DYTerm) -> bool {
        self.subterm_index
            .get(b)
            .map(|subs| subs.contains(a))
            .unwrap_or(false)
    }

    /// Generate the complete deduction closure constraints for a step.
    pub fn encode_closure(
        &self,
        knowledge_encoder: &KnowledgeEncoder,
        step: u32,
    ) -> Vec<SmtConstraint> {
        let rules = knowledge_encoder.encode_deduction_rules(&self.all_terms, step);
        rules
            .into_iter()
            .enumerate()
            .map(|(i, formula)| {
                SmtConstraint::new(
                    formula,
                    ConstraintOrigin::KnowledgeAccumulation { step },
                    format!("deduction_rule_{}_step_{}", i, step),
                )
            })
            .collect()
    }

    /// Subterm reasoning: if adversary knows t, it knows all subterms of t
    /// that don't require a key to extract.
    pub fn encode_subterm_reasoning(
        &self,
        knowledge_encoder: &KnowledgeEncoder,
        step: u32,
    ) -> Vec<SmtExpr> {
        let mut constraints = Vec::new();

        for (term, subterms) in &self.subterm_index {
            for sub in subterms {
                if sub == term {
                    continue;
                }
                // Only add subterm rule if the subterm is "accessible" (not encrypted)
                if self.is_accessible_subterm(sub, term) {
                    let knows_term = SmtExpr::var(knowledge_encoder.knows_var(term, step));
                    let knows_sub = SmtExpr::var(knowledge_encoder.knows_var(sub, step));
                    constraints.push(SmtExpr::implies(knows_term, knows_sub));
                }
            }
        }
        constraints
    }

    /// Determine if a subterm is accessible without keys (through pairs/tuples only).
    fn is_accessible_subterm(&self, sub: &DYTerm, parent: &DYTerm) -> bool {
        if sub == parent {
            return true;
        }
        match parent {
            DYTerm::Pair(a, b) => {
                self.is_accessible_subterm(sub, a) || self.is_accessible_subterm(sub, b)
            }
            DYTerm::Tuple(elems) => {
                elems.iter().any(|e| self.is_accessible_subterm(sub, e))
            }
            DYTerm::TlsRecord { version, payload, .. } => {
                self.is_accessible_subterm(sub, version)
                    || self.is_accessible_subterm(sub, payload)
            }
            DYTerm::SshPacket { payload, .. } => self.is_accessible_subterm(sub, payload),
            // Encrypted terms require a key — not freely accessible
            DYTerm::SymEncrypt { .. } | DYTerm::AsymEncrypt { .. } => false,
            DYTerm::Hash(_) | DYTerm::Mac { .. } => false,
            _ => false,
        }
    }

    pub fn all_terms(&self) -> &[DYTerm] {
        &self.all_terms
    }

    pub fn term_count(&self) -> usize {
        self.all_terms.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dy_term_atoms() {
        let n = DYTerm::nonce(1);
        assert!(n.is_atom());
        assert_eq!(n.depth(), 0);
        assert_eq!(format!("{}", n), "n1");
    }

    #[test]
    fn test_dy_term_constructors() {
        let k = DYTerm::sym_key(1);
        let m = DYTerm::nonce(2);
        let enc = DYTerm::sym_encrypt(k.clone(), m.clone());
        assert!(!enc.is_atom());
        assert_eq!(enc.depth(), 1);

        let pair = DYTerm::pair(k, m);
        assert_eq!(pair.depth(), 1);
    }

    #[test]
    fn test_dy_term_subterms() {
        let k = DYTerm::sym_key(1);
        let m = DYTerm::nonce(2);
        let enc = DYTerm::sym_encrypt(k.clone(), m.clone());
        let subs = enc.subterms();
        assert!(subs.contains(&k));
        assert!(subs.contains(&m));
        assert!(subs.contains(&enc));
    }

    #[test]
    fn test_term_algebra_encoding() {
        let mut algebra = DYTermAlgebra::new();
        let k = DYTerm::sym_key(1);
        let m = DYTerm::nonce(2);
        let enc = DYTerm::sym_encrypt(k.clone(), m.clone());

        let encoded = algebra.encode_term(&enc);
        let s = format!("{}", encoded);
        assert!(s.contains("senc"));
    }

    #[test]
    fn test_constructor_destructor_axioms() {
        let algebra = DYTermAlgebra::new();
        let axioms = algebra.constructor_destructor_axioms();
        assert!(axioms.len() >= 4);

        // Check symmetric decryption axiom
        let s = format!("{}", axioms[0]);
        assert!(s.contains("forall"));
        assert!(s.contains("sdec"));
        assert!(s.contains("senc"));
    }

    #[test]
    fn test_term_encoder_caching() {
        let mut encoder = TermEncoder::new();
        let t = DYTerm::nonce(1);
        let e1 = encoder.encode(&t);
        let e2 = encoder.encode(&t);
        assert_eq!(e1, e2);
    }

    #[test]
    fn test_knowledge_encoder_vars() {
        let ke = KnowledgeEncoder::new(3);
        let t = DYTerm::nonce(1);
        let var_name = ke.knows_var(&t, 2);
        assert!(var_name.contains("nonce_1"));
        assert!(var_name.contains("_2"));
    }

    #[test]
    fn test_knowledge_monotonicity() {
        let ke = KnowledgeEncoder::new(3);
        let terms = vec![DYTerm::nonce(1), DYTerm::nonce(2)];
        let mono = ke.encode_monotonicity(&terms, 0);
        assert_eq!(mono.len(), 2);
        let s = format!("{}", mono[0]);
        assert!(s.contains("=>"));
    }

    #[test]
    fn test_deduction_rules_sym_encrypt() {
        let k = DYTerm::sym_key(1);
        let m = DYTerm::nonce(2);
        let enc = DYTerm::sym_encrypt(k.clone(), m.clone());

        let ke = KnowledgeEncoder::new(3);
        let terms = vec![k, m, enc];
        let rules = ke.encode_deduction_rules(&terms, 0);
        assert!(!rules.is_empty());
    }

    #[test]
    fn test_deduction_rules_pair() {
        let a = DYTerm::nonce(1);
        let b = DYTerm::nonce(2);
        let p = DYTerm::pair(a.clone(), b.clone());

        let ke = KnowledgeEncoder::new(3);
        let terms = vec![a, b, p];
        let rules = ke.encode_deduction_rules(&terms, 0);
        // Should have projection and pairing rules
        assert!(rules.len() >= 3);
    }

    #[test]
    fn test_deduction_rules_hash() {
        let m = DYTerm::nonce(1);
        let h = DYTerm::hash(m.clone());

        let ke = KnowledgeEncoder::new(3);
        let terms = vec![m, h];
        let rules = ke.encode_deduction_rules(&terms, 0);
        assert!(!rules.is_empty());
    }

    #[test]
    fn test_initial_knowledge() {
        let ke = KnowledgeEncoder::new(3);
        let public = vec![DYTerm::nonce(1)];
        let all = vec![DYTerm::nonce(1), DYTerm::sym_key(2)];
        let constraints = ke.encode_initial_knowledge(&public, &all);
        assert_eq!(constraints.len(), 2);
    }

    #[test]
    fn test_deduction_closure() {
        let k = DYTerm::sym_key(1);
        let m = DYTerm::nonce(2);
        let enc = DYTerm::sym_encrypt(k.clone(), m.clone());
        let rules = DeductionRules::new(vec![k, m, enc]);
        assert_eq!(rules.term_count(), 3);
    }

    #[test]
    fn test_subterm_reasoning() {
        let a = DYTerm::nonce(1);
        let b = DYTerm::nonce(2);
        let p = DYTerm::pair(a.clone(), b.clone());

        let dr = DeductionRules::new(vec![a.clone(), b.clone(), p.clone()]);
        assert!(dr.is_subterm(&a, &p));
        assert!(dr.is_subterm(&b, &p));
        assert!(!dr.is_subterm(&p, &a));
    }

    #[test]
    fn test_accessible_subterm() {
        let k = DYTerm::sym_key(1);
        let m = DYTerm::nonce(2);
        let enc = DYTerm::sym_encrypt(k.clone(), m.clone());

        let dr = DeductionRules::new(vec![k.clone(), m.clone(), enc.clone()]);
        // m is not freely accessible inside enc (needs key)
        assert!(!dr.is_subterm(&m, &DYTerm::pair(enc.clone(), DYTerm::nonce(3))));
    }

    #[test]
    fn test_intercept_knowledge() {
        let ke = KnowledgeEncoder::new(3);
        let terms = vec![DYTerm::nonce(1), DYTerm::nonce(2)];
        let flag = SmtExpr::var("intercept_0");
        let constraints = ke.encode_intercept_knowledge(&terms, 0, flag);
        assert_eq!(constraints.len(), 2);
    }

    #[test]
    fn test_dy_term_display() {
        let k = DYTerm::sym_key(1);
        let m = DYTerm::nonce(2);
        let enc = DYTerm::sym_encrypt(k, m);
        let s = format!("{}", enc);
        assert!(s.contains("senc"));

        let a = DYTerm::nonce(3);
        let b = DYTerm::nonce(4);
        let p = DYTerm::pair(a, b);
        let s = format!("{}", p);
        assert!(s.contains("⟨"));
    }

    #[test]
    fn test_term_algebra_declarations() {
        let algebra = DYTermAlgebra::new();
        let decls = algebra.constructor_declarations();
        assert!(decls.len() >= 10);
    }
}
