//! Audit trail for compliance-ready documentation.
//!
//! An [`AuditLog`] records every significant action (certificate generation,
//! verification, export) as an [`AuditEntry`].  An [`AuditTrail`] links a log
//! to a certificate chain, and [`AuditReport`] renders the trail as a
//! human-readable compliance document.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::certificate::{CertificateHash, CertificateId};
use crate::checker::CheckStatus;

// ---------------------------------------------------------------------------
// AuditEntry
// ---------------------------------------------------------------------------

/// A single auditable event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Unique entry identifier.
    pub id: Uuid,
    /// When the event occurred.
    pub timestamp: DateTime<Utc>,
    /// Category of the event.
    pub action: String,
    /// Optional certificate this event relates to.
    pub certificate_id: Option<CertificateId>,
    /// Optional check outcome.
    pub status: Option<CheckStatus>,
    /// Free-text description.
    pub description: String,
    /// Optional key-value metadata.
    pub metadata: indexmap::IndexMap<String, String>,
}

impl AuditEntry {
    /// Create a new entry for the current instant.
    pub fn new(action: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            action: action.into(),
            certificate_id: None,
            status: None,
            description: description.into(),
            metadata: indexmap::IndexMap::new(),
        }
    }

    /// Attach a certificate reference.
    pub fn with_certificate(mut self, id: CertificateId) -> Self {
        self.certificate_id = Some(id);
        self
    }

    /// Attach a check status.
    pub fn with_status(mut self, status: CheckStatus) -> Self {
        self.status = Some(status);
        self
    }

    /// Add a key-value metadata pair.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

// ---------------------------------------------------------------------------
// AuditLog
// ---------------------------------------------------------------------------

/// An append-only log of [`AuditEntry`] items.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLog {
    /// Ordered entries (oldest first).
    pub entries: Vec<AuditEntry>,
}

impl Default for AuditLog {
    fn default() -> Self {
        Self::new()
    }
}

impl AuditLog {
    /// Create an empty log.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Append an entry.
    pub fn push(&mut self, entry: AuditEntry) {
        log::debug!("audit: {} – {}", entry.action, entry.description);
        self.entries.push(entry);
    }

    /// Number of recorded entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Filter entries related to a specific certificate.
    pub fn entries_for_certificate(&self, id: &CertificateId) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.certificate_id.as_ref() == Some(id))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// AuditTrail
// ---------------------------------------------------------------------------

/// An audit log bound to a specific certificate chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditTrail {
    /// Identifier of the certificate chain this trail accompanies.
    pub chain_id: CertificateId,
    /// Hash of the chain's leaf certificate at the time the trail was sealed.
    pub chain_leaf_hash: Option<CertificateHash>,
    /// The underlying log.
    pub log: AuditLog,
    /// When the trail was finalised.
    pub sealed_at: Option<DateTime<Utc>>,
}

impl AuditTrail {
    /// Start a new trail for the given chain.
    pub fn new(chain_id: CertificateId) -> Self {
        Self {
            chain_id,
            chain_leaf_hash: None,
            log: AuditLog::new(),
            sealed_at: None,
        }
    }

    /// Record an event on this trail.
    pub fn record(&mut self, entry: AuditEntry) {
        self.log.push(entry);
    }

    /// Seal the trail, preventing further modification.
    pub fn seal(&mut self, leaf_hash: CertificateHash) {
        self.chain_leaf_hash = Some(leaf_hash);
        self.sealed_at = Some(Utc::now());
    }

    /// Whether the trail has been sealed.
    pub fn is_sealed(&self) -> bool {
        self.sealed_at.is_some()
    }
}

// ---------------------------------------------------------------------------
// AuditReport
// ---------------------------------------------------------------------------

/// A rendered, human-readable compliance report derived from an [`AuditTrail`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditReport {
    /// Report title.
    pub title: String,
    /// When the report was generated.
    pub generated_at: DateTime<Utc>,
    /// Chain identifier covered by this report.
    pub chain_id: CertificateId,
    /// Total certificates in the chain.
    pub certificate_count: usize,
    /// Number of entries that passed verification.
    pub verified_count: usize,
    /// Number of entries that failed verification.
    pub failed_count: usize,
    /// Rendered sections (Markdown or plain text).
    pub sections: Vec<ReportSection>,
}

/// A titled section inside an [`AuditReport`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSection {
    /// Section heading.
    pub heading: String,
    /// Section body (Markdown).
    pub body: String,
}

impl AuditReport {
    /// Generate a report from a sealed audit trail.
    pub fn from_trail(trail: &AuditTrail) -> Self {
        let verified_count = trail
            .log
            .entries
            .iter()
            .filter(|e| e.status == Some(CheckStatus::Verified))
            .count();
        let failed_count = trail
            .log
            .entries
            .iter()
            .filter(|e| e.status == Some(CheckStatus::Failed))
            .count();

        let mut sections = Vec::new();

        // Summary section.
        sections.push(ReportSection {
            heading: "Summary".into(),
            body: format!(
                "Chain `{}` — {} audit entries, {} verified, {} failed.",
                trail.chain_id,
                trail.log.len(),
                verified_count,
                failed_count,
            ),
        });

        // Detail section.
        let mut detail_body = String::new();
        for entry in &trail.log.entries {
            detail_body.push_str(&format!(
                "- **[{}]** {} — {}\n",
                entry.timestamp.format("%Y-%m-%dT%H:%M:%SZ"),
                entry.action,
                entry.description,
            ));
        }
        sections.push(ReportSection {
            heading: "Event Log".into(),
            body: detail_body,
        });

        Self {
            title: format!("Audit Report — chain {}", trail.chain_id),
            generated_at: Utc::now(),
            chain_id: trail.chain_id.clone(),
            certificate_count: 0, // caller should fill in from chain
            verified_count,
            failed_count,
            sections,
        }
    }

    /// Render the report as a Markdown string.
    pub fn to_markdown(&self) -> String {
        let mut md = format!("# {}\n\n", self.title);
        md.push_str(&format!(
            "Generated: {}  \nCertificates: {}  \nVerified: {}  \nFailed: {}\n\n",
            self.generated_at.format("%Y-%m-%dT%H:%M:%SZ"),
            self.certificate_count,
            self.verified_count,
            self.failed_count,
        ));
        for section in &self.sections {
            md.push_str(&format!("## {}\n\n{}\n\n", section.heading, section.body));
        }
        md
    }
}
