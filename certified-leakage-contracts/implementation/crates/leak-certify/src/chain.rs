//! Certificate chains linking function → composition → library proofs.
//!
//! A [`CertificateChain`] is an ordered sequence of [`Certificate`]s whose hashes
//! form a tamper-evident linked list.  [`ChainValidator`] checks the integrity of
//! such a chain, while [`ChainBuilder`] provides a fluent API for constructing one.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::certificate::{Certificate, CertificateId, CertificateKind};

// ---------------------------------------------------------------------------
// CertificateChain
// ---------------------------------------------------------------------------

/// An ordered, hash-linked sequence of certificates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateChain {
    /// Unique identifier for the chain.
    pub id: CertificateId,
    /// Display name for the chain.
    pub name: String,
    /// Certificates in link order (index 0 is the root).
    pub certificates: Vec<Certificate>,
    /// Timestamp of chain creation.
    pub created_at: DateTime<Utc>,
}

impl CertificateChain {
    /// Number of certificates in the chain.
    pub fn len(&self) -> usize {
        self.certificates.len()
    }

    /// Whether the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.certificates.is_empty()
    }

    /// Return the root certificate, if present.
    pub fn root(&self) -> Option<&Certificate> {
        self.certificates.first()
    }

    /// Return the leaf (most recent) certificate, if present.
    pub fn leaf(&self) -> Option<&Certificate> {
        self.certificates.last()
    }

    /// Iterate over all certificates in link order.
    pub fn iter(&self) -> impl Iterator<Item = &Certificate> {
        self.certificates.iter()
    }
}

// ---------------------------------------------------------------------------
// ChainValidator
// ---------------------------------------------------------------------------

/// Result produced by [`ChainValidator::validate`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainValidationResult {
    /// Whether the entire chain is valid.
    pub valid: bool,
    /// Per-certificate issues (empty when valid).
    pub issues: Vec<ChainIssue>,
}

/// A single issue discovered during chain validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainIssue {
    /// Index of the problematic certificate in the chain.
    pub index: usize,
    /// Certificate ID involved.
    pub certificate_id: CertificateId,
    /// Human-readable description of the issue.
    pub description: String,
}

/// Independent validator for [`CertificateChain`] integrity.
#[derive(Debug, Clone)]
pub struct ChainValidator {
    /// Whether to require strict kind ordering (Function → Composition → Library).
    pub require_kind_ordering: bool,
}

impl Default for ChainValidator {
    fn default() -> Self {
        Self {
            require_kind_ordering: true,
        }
    }
}

impl ChainValidator {
    /// Create a new validator with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate hash-chain integrity and (optionally) kind ordering.
    pub fn validate(&self, chain: &CertificateChain) -> ChainValidationResult {
        let mut issues = Vec::new();

        for (i, cert) in chain.certificates.iter().enumerate() {
            // Check self-hash integrity.
            if !cert.verify_hash() {
                issues.push(ChainIssue {
                    index: i,
                    certificate_id: cert.id.clone(),
                    description: "certificate hash does not match content".into(),
                });
            }

            // Check parent-hash linkage.
            if i == 0 {
                if cert.parent_hash.is_some() {
                    issues.push(ChainIssue {
                        index: i,
                        certificate_id: cert.id.clone(),
                        description: "root certificate should not have a parent hash".into(),
                    });
                }
            } else {
                let expected_parent = &chain.certificates[i - 1].hash;
                match &cert.parent_hash {
                    Some(ph) if ph == expected_parent => {}
                    _ => {
                        issues.push(ChainIssue {
                            index: i,
                            certificate_id: cert.id.clone(),
                            description: "parent hash does not match previous certificate".into(),
                        });
                    }
                }
            }

            // Optional kind ordering check.
            if self.require_kind_ordering && i > 0 {
                let prev_kind = &chain.certificates[i - 1].kind;
                if !Self::kind_order_valid(prev_kind, &cert.kind) {
                    issues.push(ChainIssue {
                        index: i,
                        certificate_id: cert.id.clone(),
                        description: format!(
                            "kind ordering violation: {:?} followed by {:?}",
                            prev_kind, cert.kind
                        ),
                    });
                }
            }
        }

        ChainValidationResult {
            valid: issues.is_empty(),
            issues,
        }
    }

    /// Returns `true` when `next` may follow `prev` in a well-ordered chain.
    fn kind_order_valid(prev: &CertificateKind, next: &CertificateKind) -> bool {
        use CertificateKind::*;
        match (prev, next) {
            (FunctionLevel, FunctionLevel) => true,
            (FunctionLevel, Composition) => true,
            (Composition, Composition) => true,
            (Composition, Library) => true,
            (_, Auxiliary) | (Auxiliary, _) => true,
            (_, Reduction) | (Reduction, _) => true,
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// ChainBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for constructing a [`CertificateChain`].
#[derive(Debug, Clone)]
pub struct ChainBuilder {
    name: String,
    certificates: Vec<Certificate>,
}

impl ChainBuilder {
    /// Start building a new chain with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            certificates: Vec::new(),
        }
    }

    /// Append a certificate to the chain.
    pub fn add(mut self, certificate: Certificate) -> Self {
        self.certificates.push(certificate);
        self
    }

    /// Append several certificates at once.
    pub fn add_all(mut self, certs: impl IntoIterator<Item = Certificate>) -> Self {
        self.certificates.extend(certs);
        self
    }

    /// Finalise and return the chain.
    pub fn build(self) -> CertificateChain {
        CertificateChain {
            id: CertificateId::new(),
            name: self.name,
            certificates: self.certificates,
            created_at: Utc::now(),
        }
    }
}
