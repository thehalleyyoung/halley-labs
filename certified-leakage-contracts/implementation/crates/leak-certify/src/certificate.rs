//! Core certificate types with hash-chain integrity.
//!
//! A [`Certificate`] bundles a cryptographic identity, a set of [`Claim`]s about
//! information-leakage bounds, and supporting [`Evidence`].  Certificates are
//! content-addressed via SHA-256 so they can be linked into tamper-evident chains.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

use shared_types::FunctionId;

// ---------------------------------------------------------------------------
// CertificateHash
// ---------------------------------------------------------------------------

/// SHA-256 hash used as a content address for certificates.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CertificateHash(pub String);

impl CertificateHash {
    /// Compute a SHA-256 hash over arbitrary bytes and return it hex-encoded.
    pub fn from_bytes(data: &[u8]) -> Self {
        let digest = Sha256::digest(data);
        Self(hex::encode(digest))
    }

    /// Return the hex string representation.
    pub fn as_hex(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for CertificateHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// CertificateId
// ---------------------------------------------------------------------------

/// Unique identifier for a certificate (UUID v4).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CertificateId(pub Uuid);

impl CertificateId {
    /// Generate a fresh random identifier.
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Build from an existing UUID.
    pub fn from_uuid(id: Uuid) -> Self {
        Self(id)
    }
}

impl Default for CertificateId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for CertificateId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// CertificateKind
// ---------------------------------------------------------------------------

/// Classification of the certificate's proof scope.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CertificateKind {
    /// Single-function leakage bound.
    FunctionLevel,
    /// Composition of multiple function-level certificates.
    Composition,
    /// Whole-library / whole-binary bound.
    Library,
    /// Reduction to a known-hard problem.
    Reduction,
    /// Intermediate or auxiliary certificate.
    Auxiliary,
}

// ---------------------------------------------------------------------------
// CertificateSubject
// ---------------------------------------------------------------------------

/// The entity (function, composition, library) a certificate speaks about.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CertificateSubject {
    /// Human-readable name (e.g. function symbol or library name).
    pub name: String,
    /// Optional function identifier from the analysis.
    pub function_id: Option<FunctionId>,
    /// Optional binary / object file path.
    pub binary_path: Option<String>,
    /// SHA-256 of the subject binary for integrity.
    pub binary_hash: Option<CertificateHash>,
}

impl CertificateSubject {
    /// Create a subject for a single function.
    pub fn function(name: impl Into<String>, function_id: FunctionId) -> Self {
        Self {
            name: name.into(),
            function_id: Some(function_id),
            binary_path: None,
            binary_hash: None,
        }
    }

    /// Create a subject for a whole library.
    pub fn library(name: impl Into<String>, binary_path: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            function_id: None,
            binary_path: Some(binary_path.into()),
            binary_hash: None,
        }
    }
}

// ---------------------------------------------------------------------------
// ClaimProperty
// ---------------------------------------------------------------------------

/// A specific property asserted by a [`Claim`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ClaimProperty {
    /// Upper bound on cache-line leakage (in bits).
    CacheLeakageBound { bits: f64 },
    /// Upper bound on timing-channel leakage (in bits).
    TimingLeakageBound { bits: f64 },
    /// Constant-time guarantee (zero leakage).
    ConstantTime,
    /// Secret-independent memory access pattern.
    SecretIndependentAccess,
    /// Custom property described by a free-form string.
    Custom { description: String },
}

// ---------------------------------------------------------------------------
// Claim
// ---------------------------------------------------------------------------

/// A single verifiable assertion within a certificate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Claim {
    /// What the claim asserts.
    pub property: ClaimProperty,
    /// Free-text description for audit trails.
    pub description: String,
    /// Whether the claim has been independently checked.
    pub verified: bool,
}

impl Claim {
    /// Create an unchecked claim.
    pub fn new(property: ClaimProperty, description: impl Into<String>) -> Self {
        Self {
            property,
            description: description.into(),
            verified: false,
        }
    }

    /// Mark this claim as independently verified.
    pub fn mark_verified(&mut self) {
        self.verified = true;
    }
}

// ---------------------------------------------------------------------------
// Evidence
// ---------------------------------------------------------------------------

/// Supporting data that backs a [`Claim`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    /// Human-readable label (e.g. "fixpoint trace", "counting argument").
    pub kind: String,
    /// Serialised evidence payload (JSON or opaque bytes).
    pub payload: serde_json::Value,
    /// Hash of the evidence for integrity checking.
    pub hash: CertificateHash,
}

impl Evidence {
    /// Build evidence from a JSON-serialisable value.
    pub fn from_value(kind: impl Into<String>, value: &impl Serialize) -> Result<Self, serde_json::Error> {
        let payload = serde_json::to_value(value)?;
        let bytes = serde_json::to_vec(&payload).unwrap_or_default();
        let hash = CertificateHash::from_bytes(&bytes);
        Ok(Self {
            kind: kind.into(),
            payload,
            hash,
        })
    }

    /// Verify that the stored hash matches the payload.
    pub fn verify_integrity(&self) -> bool {
        let bytes = serde_json::to_vec(&self.payload).unwrap_or_default();
        let recomputed = CertificateHash::from_bytes(&bytes);
        recomputed == self.hash
    }
}

// ---------------------------------------------------------------------------
// Certificate
// ---------------------------------------------------------------------------

/// A self-contained, hash-linked certificate of leakage analysis results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Certificate {
    /// Unique identifier.
    pub id: CertificateId,
    /// Classification of this certificate.
    pub kind: CertificateKind,
    /// The entity the certificate speaks about.
    pub subject: CertificateSubject,
    /// Assertions made by this certificate.
    pub claims: Vec<Claim>,
    /// Supporting evidence for the claims.
    pub evidence: Vec<Evidence>,
    /// Content-addressed hash of this certificate.
    pub hash: CertificateHash,
    /// Hash of the parent certificate in the chain (`None` for roots).
    pub parent_hash: Option<CertificateHash>,
    /// Timestamp of certificate creation.
    pub created_at: DateTime<Utc>,
    /// Version string of the tool that produced this certificate.
    pub tool_version: String,
}

impl Certificate {
    /// Compute the content hash from the certificate's serialised fields
    /// (excluding the `hash` field itself).
    pub fn compute_hash(&self) -> CertificateHash {
        let mut hasher = Sha256::new();
        hasher.update(self.id.0.as_bytes());
        hasher.update(serde_json::to_vec(&self.kind).unwrap_or_default());
        hasher.update(serde_json::to_vec(&self.subject).unwrap_or_default());
        hasher.update(serde_json::to_vec(&self.claims).unwrap_or_default());
        for ev in &self.evidence {
            hasher.update(ev.hash.as_hex().as_bytes());
        }
        if let Some(ref ph) = self.parent_hash {
            hasher.update(ph.as_hex().as_bytes());
        }
        CertificateHash(hex::encode(hasher.finalize()))
    }

    /// Verify that the stored hash matches the content.
    pub fn verify_hash(&self) -> bool {
        self.compute_hash() == self.hash
    }
}
