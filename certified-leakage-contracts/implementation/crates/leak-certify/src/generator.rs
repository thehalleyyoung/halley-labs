//! Certificate production from analysis results.
//!
//! [`CertificateGenerator`] turns raw analysis outputs into signed, hash-linked
//! [`Certificate`]s ready for chain assembly and export.

use chrono::Utc;
use serde::{Deserialize, Serialize};

use shared_types::FunctionId;

use crate::certificate::{
    Certificate, CertificateHash, CertificateId, CertificateKind,
    CertificateSubject, Claim, ClaimProperty, Evidence,
};
use crate::witness::Witness;

// ---------------------------------------------------------------------------
// CertificateGenerator
// ---------------------------------------------------------------------------

/// Configuration and factory for producing certificates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateGenerator {
    /// Version string baked into every generated certificate.
    pub tool_version: String,
    /// Default kind to assign when not overridden per-call.
    pub default_kind: CertificateKind,
}

impl Default for CertificateGenerator {
    fn default() -> Self {
        Self {
            tool_version: env!("CARGO_PKG_VERSION").to_string(),
            default_kind: CertificateKind::FunctionLevel,
        }
    }
}

impl CertificateGenerator {
    /// Create a new generator stamping the given tool version.
    pub fn new(tool_version: impl Into<String>) -> Self {
        Self {
            tool_version: tool_version.into(),
            ..Default::default()
        }
    }

    /// Generate a function-level certificate from analysis results.
    pub fn generate_function_certificate(
        &self,
        function_id: FunctionId,
        function_name: impl Into<String>,
        leakage_bits: f64,
        witness: &Witness,
        parent_hash: Option<CertificateHash>,
    ) -> Result<Certificate, serde_json::Error> {
        let subject = CertificateSubject::function(function_name, function_id);

        let claim = Claim::new(
            ClaimProperty::CacheLeakageBound { bits: leakage_bits },
            format!("cache leakage ≤ {:.4} bits", leakage_bits),
        );

        let evidence = Evidence::from_value("witness", witness)?;

        let mut cert = Certificate {
            id: CertificateId::new(),
            kind: CertificateKind::FunctionLevel,
            subject,
            claims: vec![claim],
            evidence: vec![evidence],
            hash: CertificateHash("".into()),
            parent_hash,
            created_at: Utc::now(),
            tool_version: self.tool_version.clone(),
        };
        cert.hash = cert.compute_hash();

        Ok(cert)
    }

    /// Generate a composition certificate from sub-certificates.
    pub fn generate_composition_certificate(
        &self,
        name: impl Into<String>,
        claims: Vec<Claim>,
        evidence: Vec<Evidence>,
        parent_hash: Option<CertificateHash>,
    ) -> Certificate {
        let subject = CertificateSubject::library(name, "");

        let mut cert = Certificate {
            id: CertificateId::new(),
            kind: CertificateKind::Composition,
            subject,
            claims,
            evidence,
            hash: CertificateHash("".into()),
            parent_hash,
            created_at: Utc::now(),
            tool_version: self.tool_version.clone(),
        };
        cert.hash = cert.compute_hash();

        cert
    }

    /// Generate a library-level certificate wrapping all compositions.
    pub fn generate_library_certificate(
        &self,
        library_name: impl Into<String>,
        binary_path: impl Into<String>,
        claims: Vec<Claim>,
        evidence: Vec<Evidence>,
        parent_hash: Option<CertificateHash>,
    ) -> Certificate {
        let subject = CertificateSubject::library(library_name, binary_path);

        let mut cert = Certificate {
            id: CertificateId::new(),
            kind: CertificateKind::Library,
            subject,
            claims,
            evidence,
            hash: CertificateHash("".into()),
            parent_hash,
            created_at: Utc::now(),
            tool_version: self.tool_version.clone(),
        };
        cert.hash = cert.compute_hash();

        cert
    }
}
