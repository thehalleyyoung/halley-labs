//! # negsyn-concrete
//!
//! Attack trace concretizer with CEGAR refinement loop for the NegSynth
//! protocol downgrade synthesis tool.
//!
//! Implements **ALG5: CONCRETIZE** — takes SAT models from the SMT solver and
//! concretizes them into executable byte-level attack traces.  Includes a full
//! CEGAR (CounterExample-Guided Abstraction Refinement) loop that iteratively
//! refines the abstraction until either a concrete attack is found or safety
//! is certified.
//!
//! ## Crate layout
//!
//! | Module           | Purpose |
//! |------------------|---------|
//! | `trace`          | Concrete trace representation & builder |
//! | `byte_encoding`  | Byte-level TLS/SSH protocol encoding |
//! | `concretizer`    | Core concretization (ALG5) |
//! | `refinement`     | Refinement predicate types & encoding |
//! | `validation`     | Trace validation & conformance checking |
//! | `cegar`          | CEGAR refinement loop |
//! | `certificate_gen` | Bounded-completeness certificate generation (Def D6) |

pub mod byte_encoding;
pub mod cegar;
pub mod certificate_gen;
pub mod concretizer;
pub mod refinement;
pub mod trace;
pub mod validation;

// ── Re-exports ───────────────────────────────────────────────────────────

pub use byte_encoding::{ByteEncoder, TlsRecordEncoder, TlsHandshakeEncoder, SshPacketEncoder};
pub use cegar::{CegarLoop, CegarConfig, CegarResult, CegarState, CegarStats};
pub use certificate_gen::{CertificateGenerator, CertificateBuilder, BoundedCertificate};
pub use concretizer::{Concretizer, ConcretizerConfig};
pub use refinement::{
    RefinementPredicate, PredicateEncoder, RefinementHistory, RefinementStrategy,
};
pub use trace::{ConcreteTrace, ConcreteMessage, TraceStep, TraceBuilder};
pub use validation::{
    TraceValidator, ProtocolConformance, AdversaryCapabilityCheck, ByteLevelVerifier,
    ReplaySimulator, ValidationReport,
};

// ── Compatible local types ────────────────────────────────────────────────
//
// These mirror the types from negsyn-types, negsyn-encode, and negsyn-extract
// so this crate can compile independently of upstream compilation issues.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::collections::BTreeSet;

/// SMT model returned by the solver — maps variable names to values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmtModel {
    pub assignments: BTreeMap<String, SmtValue>,
    pub is_sat: bool,
}

impl SmtModel {
    pub fn new() -> Self {
        Self {
            assignments: BTreeMap::new(),
            is_sat: true,
        }
    }

    pub fn get(&self, name: &str) -> Option<&SmtValue> {
        self.assignments.get(name)
    }

    pub fn get_bitvec(&self, name: &str) -> Option<u64> {
        match self.assignments.get(name)? {
            SmtValue::BitVec(v, _) => Some(*v),
            SmtValue::Int(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn get_bool(&self, name: &str) -> Option<bool> {
        match self.assignments.get(name)? {
            SmtValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn get_bytes(&self, name: &str) -> Option<&[u8]> {
        match self.assignments.get(name)? {
            SmtValue::Bytes(b) => Some(b),
            _ => None,
        }
    }

    pub fn insert(&mut self, name: impl Into<String>, value: SmtValue) {
        self.assignments.insert(name.into(), value);
    }

    pub fn variable_names(&self) -> impl Iterator<Item = &String> {
        self.assignments.keys()
    }
}

impl Default for SmtModel {
    fn default() -> Self {
        Self::new()
    }
}

/// A value in an SMT model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SmtValue {
    Bool(bool),
    Int(i64),
    BitVec(u64, u32),
    Bytes(Vec<u8>),
    String(String),
    Array(BTreeMap<u64, SmtValue>),
}

impl SmtValue {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            SmtValue::BitVec(v, _) => Some(*v),
            SmtValue::Int(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            SmtValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_bytes(&self) -> Option<&[u8]> {
        match self {
            SmtValue::Bytes(b) => Some(b),
            _ => None,
        }
    }

    pub fn bit_width(&self) -> Option<u32> {
        match self {
            SmtValue::BitVec(_, w) => Some(*w),
            _ => None,
        }
    }
}

/// An SMT formula / encoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmtFormula {
    pub assertions: Vec<SmtExpr>,
    pub declarations: BTreeMap<String, SmtSort>,
    pub name: String,
}

impl SmtFormula {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            assertions: Vec::new(),
            declarations: BTreeMap::new(),
            name: name.into(),
        }
    }

    pub fn add_assertion(&mut self, expr: SmtExpr) {
        self.assertions.push(expr);
    }

    pub fn declare(&mut self, name: impl Into<String>, sort: SmtSort) {
        self.declarations.insert(name.into(), sort);
    }

    pub fn assertion_count(&self) -> usize {
        self.assertions.len()
    }
}

/// SMT sort.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SmtSort {
    Bool,
    Int,
    BitVec(u32),
    Array(Box<SmtSort>, Box<SmtSort>),
}

/// SMT expression.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SmtExpr {
    BoolLit(bool),
    IntLit(i64),
    BitVecLit(u64, u32),
    Var(String),
    Not(Box<SmtExpr>),
    And(Vec<SmtExpr>),
    Or(Vec<SmtExpr>),
    Implies(Box<SmtExpr>, Box<SmtExpr>),
    Eq(Box<SmtExpr>, Box<SmtExpr>),
    Distinct(Vec<SmtExpr>),
    Ite(Box<SmtExpr>, Box<SmtExpr>, Box<SmtExpr>),
    BvAnd(Box<SmtExpr>, Box<SmtExpr>),
    BvOr(Box<SmtExpr>, Box<SmtExpr>),
    BvAdd(Box<SmtExpr>, Box<SmtExpr>),
    BvExtract(Box<SmtExpr>, u32, u32),
    BvConcat(Box<SmtExpr>, Box<SmtExpr>),
    Select(Box<SmtExpr>, Box<SmtExpr>),
    Store(Box<SmtExpr>, Box<SmtExpr>, Box<SmtExpr>),
    Le(Box<SmtExpr>, Box<SmtExpr>),
    Lt(Box<SmtExpr>, Box<SmtExpr>),
}

impl SmtExpr {
    pub fn var(name: impl Into<String>) -> Self {
        SmtExpr::Var(name.into())
    }

    pub fn and(exprs: Vec<SmtExpr>) -> Self {
        SmtExpr::And(exprs)
    }

    pub fn or(exprs: Vec<SmtExpr>) -> Self {
        SmtExpr::Or(exprs)
    }

    pub fn not(e: SmtExpr) -> Self {
        SmtExpr::Not(Box::new(e))
    }

    pub fn eq(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Eq(Box::new(a), Box::new(b))
    }

    pub fn implies(a: SmtExpr, b: SmtExpr) -> Self {
        SmtExpr::Implies(Box::new(a), Box::new(b))
    }

    pub fn bv_lit(val: u64, width: u32) -> Self {
        SmtExpr::BitVecLit(val, width)
    }

    pub fn ite(c: SmtExpr, t: SmtExpr, e: SmtExpr) -> Self {
        SmtExpr::Ite(Box::new(c), Box::new(t), Box::new(e))
    }
}

/// Adversary action within the Dolev-Yao model.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdversaryAction {
    Forward { from: String, to: String, msg_idx: usize },
    Drop { from: String, msg_idx: usize },
    Inject { to: String, payload: Vec<u8> },
    Modify { from: String, to: String, msg_idx: usize, field: String, new_value: Vec<u8> },
    Replay { original_idx: usize, to: String },
    Intercept { from: String, msg_idx: usize },
}

impl AdversaryAction {
    pub fn is_passive(&self) -> bool {
        matches!(self, AdversaryAction::Forward { .. })
    }

    pub fn target(&self) -> &str {
        match self {
            AdversaryAction::Forward { to, .. } => to,
            AdversaryAction::Drop { from, .. } => from,
            AdversaryAction::Inject { to, .. } => to,
            AdversaryAction::Modify { to, .. } => to,
            AdversaryAction::Replay { to, .. } => to,
            AdversaryAction::Intercept { from, .. } => from,
        }
    }
}

/// UNSAT proof element for certificate generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnsatProof {
    pub core: Vec<String>,
    pub resolution_steps: Vec<ResolutionStep>,
    pub is_valid: bool,
}

impl UnsatProof {
    pub fn new(core: Vec<String>) -> Self {
        Self {
            core,
            resolution_steps: Vec::new(),
            is_valid: true,
        }
    }

    pub fn core_size(&self) -> usize {
        self.core.len()
    }
}

/// A resolution step in an UNSAT proof.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionStep {
    pub clause_a: usize,
    pub clause_b: usize,
    pub pivot: String,
    pub result: usize,
}

/// Library identification for multi-library analysis.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LibraryId {
    pub name: String,
    pub version: String,
    pub protocol: String,
}

impl LibraryId {
    pub fn new(name: impl Into<String>, version: impl Into<String>, protocol: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            protocol: protocol.into(),
        }
    }
}

impl std::fmt::Display for LibraryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{} ({})", self.name, self.version, self.protocol)
    }
}

/// Errors specific to the concrete crate.
#[derive(Debug, thiserror::Error)]
pub enum ConcreteError {
    #[error("concretization failed: {0}")]
    Concretization(String),

    #[error("CEGAR loop failed: {0}")]
    Cegar(String),

    #[error("validation failed: {0}")]
    Validation(String),

    #[error("encoding error: {0}")]
    Encoding(String),

    #[error("certificate generation failed: {0}")]
    Certificate(String),

    #[error("refinement error: {0}")]
    Refinement(String),

    #[error("model extraction error: {0}")]
    ModelExtraction(String),

    #[error("protocol error: {0}")]
    Protocol(String),

    #[error("timeout after {0}ms")]
    Timeout(u64),

    #[error("max iterations ({0}) exceeded")]
    MaxIterations(usize),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type ConcreteResult<T> = std::result::Result<T, ConcreteError>;

// ── Protocol types (compatible with negsyn-types) ────────────────────────

/// Phases of a TLS/SSH handshake.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HandshakePhase {
    Initial,
    ClientHello,
    ServerHello,
    Certificate,
    KeyExchange,
    ChangeCipherSpec,
    Finished,
    ApplicationData,
    Alert,
    Renegotiation,
}

impl HandshakePhase {
    pub fn order_index(&self) -> u32 {
        match self {
            Self::Initial => 0,
            Self::ClientHello => 1,
            Self::ServerHello => 2,
            Self::Certificate => 3,
            Self::KeyExchange => 4,
            Self::ChangeCipherSpec => 5,
            Self::Finished => 6,
            Self::ApplicationData => 7,
            Self::Alert => 100,
            Self::Renegotiation => 50,
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::ApplicationData | Self::Alert)
    }
}

impl std::fmt::Display for HandshakePhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Protocol version identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ProtocolVersion {
    Ssl30,
    Tls10,
    Tls11,
    Tls12,
    Tls13,
    Ssh2,
    Dtls10,
    Dtls12,
    Unknown(u16),
}

impl ProtocolVersion {
    pub fn security_level(&self) -> u32 {
        match self {
            Self::Ssl30 => 0,
            Self::Tls10 => 1,
            Self::Tls11 => 2,
            Self::Tls12 => 3,
            Self::Tls13 => 4,
            Self::Ssh2 => 3,
            Self::Dtls10 => 1,
            Self::Dtls12 => 3,
            Self::Unknown(_) => 0,
        }
    }

    pub fn is_deprecated(&self) -> bool {
        matches!(self, Self::Ssl30 | Self::Tls10 | Self::Tls11)
    }
}

impl std::fmt::Display for ProtocolVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ssl30 => write!(f, "SSL 3.0"),
            Self::Tls10 => write!(f, "TLS 1.0"),
            Self::Tls11 => write!(f, "TLS 1.1"),
            Self::Tls12 => write!(f, "TLS 1.2"),
            Self::Tls13 => write!(f, "TLS 1.3"),
            Self::Ssh2 => write!(f, "SSH 2.0"),
            Self::Dtls10 => write!(f, "DTLS 1.0"),
            Self::Dtls12 => write!(f, "DTLS 1.2"),
            Self::Unknown(v) => write!(f, "Unknown(0x{:04x})", v),
        }
    }
}

/// Key exchange algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum KeyExchange {
    Null, Rsa, Dhe, Ecdhe, Psk, DhePsk, EcdhePsk, Srp, Kerberos,
}

/// Authentication algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AuthAlgorithm {
    Null, Rsa, Dss, Ecdsa, Psk, Anonymous, Ed25519,
}

/// Bulk encryption algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BulkEncryption {
    Null, Rc4_128, Des_40, Des_56, TripleDes, Aes128, Aes256,
    Aes128Gcm, Aes256Gcm, Chacha20Poly1305, Camellia128, Camellia256,
    Aria128, Aria256,
}

/// MAC algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MacAlgorithm {
    Null, HmacMd5, HmacSha1, HmacSha256, HmacSha384, Aead,
}

/// A cipher suite with its component algorithms.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CipherSuite {
    pub id: u16,
    pub name: String,
    pub key_exchange: KeyExchange,
    pub authentication: AuthAlgorithm,
    pub encryption: BulkEncryption,
    pub mac: MacAlgorithm,
    pub is_export: bool,
    pub is_fips_approved: bool,
}

impl CipherSuite {
    pub fn new(
        id: u16,
        name: impl Into<String>,
        kx: KeyExchange,
        auth: AuthAlgorithm,
        enc: BulkEncryption,
        mac: MacAlgorithm,
    ) -> Self {
        let is_export = matches!(enc, BulkEncryption::Des_40);
        let is_fips = matches!(
            enc,
            BulkEncryption::Aes128
                | BulkEncryption::Aes256
                | BulkEncryption::Aes128Gcm
                | BulkEncryption::Aes256Gcm
                | BulkEncryption::TripleDes
        );
        Self {
            id,
            name: name.into(),
            key_exchange: kx,
            authentication: auth,
            encryption: enc,
            mac,
            is_export,
            is_fips_approved: is_fips,
        }
    }

    pub fn null_suite() -> Self {
        Self::new(
            0x0000,
            "TLS_NULL_WITH_NULL_NULL",
            KeyExchange::Null,
            AuthAlgorithm::Null,
            BulkEncryption::Null,
            MacAlgorithm::Null,
        )
    }
}

impl std::fmt::Display for CipherSuite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0x{:04x}:{}", self.id, self.name)
    }
}

/// TLS/SSH extension.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Extension {
    pub id: u16,
    pub name: String,
    pub data: Vec<u8>,
    pub is_critical: bool,
}

impl Extension {
    pub fn new(id: u16, name: impl Into<String>, data: Vec<u8>) -> Self {
        Self {
            id,
            name: name.into(),
            data,
            is_critical: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smt_model_basics() {
        let mut model = SmtModel::new();
        model.insert("x", SmtValue::BitVec(42, 16));
        model.insert("flag", SmtValue::Bool(true));
        model.insert("data", SmtValue::Bytes(vec![1, 2, 3]));

        assert_eq!(model.get_bitvec("x"), Some(42));
        assert_eq!(model.get_bool("flag"), Some(true));
        assert_eq!(model.get_bytes("data"), Some(&[1u8, 2, 3][..]));
        assert_eq!(model.get_bitvec("missing"), None);
    }

    #[test]
    fn test_smt_value_conversions() {
        let bv = SmtValue::BitVec(0x0303, 16);
        assert_eq!(bv.as_u64(), Some(0x0303));
        assert_eq!(bv.bit_width(), Some(16));
        assert_eq!(bv.as_bool(), None);

        let b = SmtValue::Bool(false);
        assert_eq!(b.as_bool(), Some(false));
        assert_eq!(b.as_u64(), None);
    }

    #[test]
    fn test_smt_formula_builder() {
        let mut formula = SmtFormula::new("test");
        formula.declare("x", SmtSort::BitVec(16));
        formula.add_assertion(SmtExpr::eq(
            SmtExpr::var("x"),
            SmtExpr::bv_lit(42, 16),
        ));
        assert_eq!(formula.assertion_count(), 1);
        assert_eq!(formula.declarations.len(), 1);
    }

    #[test]
    fn test_adversary_action() {
        let fwd = AdversaryAction::Forward {
            from: "client".into(),
            to: "server".into(),
            msg_idx: 0,
        };
        assert!(fwd.is_passive());
        assert_eq!(fwd.target(), "server");

        let drop = AdversaryAction::Drop {
            from: "client".into(),
            msg_idx: 1,
        };
        assert!(!drop.is_passive());
    }

    #[test]
    fn test_library_id() {
        let lib = LibraryId::new("openssl", "1.1.1", "TLS");
        assert_eq!(format!("{}", lib), "openssl@1.1.1 (TLS)");
    }

    #[test]
    fn test_unsat_proof() {
        let proof = UnsatProof::new(vec!["c1".into(), "c2".into()]);
        assert_eq!(proof.core_size(), 2);
        assert!(proof.is_valid);
    }
}
