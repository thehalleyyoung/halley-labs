//! Certificate export: serialize certificates to JSON, human-readable text,
//! and compliance report format. Includes hash computation for integrity.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use xr_types::certificate::{
    CertificateGrade, CoverageCertificate,
};
use xr_types::{VerifierError, VerifierResult};

// ──────────────────── Certificate Exporter ────────────────────────────────

/// Exports certificates in various formats.
pub struct CertificateExporter {
    /// Whether to include individual sample data in exports.
    pub include_samples: bool,
    /// Whether to include per-element breakdown.
    pub include_element_breakdown: bool,
    /// Maximum samples to include in text reports.
    pub max_text_samples: usize,
}

impl CertificateExporter {
    /// Create a new exporter with default settings.
    pub fn new() -> Self {
        Self {
            include_samples: true,
            include_element_breakdown: true,
            max_text_samples: 20,
        }
    }

    /// Create an exporter that omits sample data (smaller output).
    pub fn compact() -> Self {
        Self {
            include_samples: false,
            include_element_breakdown: false,
            max_text_samples: 0,
        }
    }

    /// Export to JSON string.
    pub fn to_json(&self, cert: &CoverageCertificate) -> VerifierResult<String> {
        if self.include_samples {
            serde_json::to_string_pretty(cert).map_err(VerifierError::Json)
        } else {
            // Create a version without sample details
            let summary = CertificateJsonSummary::from_cert(cert);
            serde_json::to_string_pretty(&summary).map_err(VerifierError::Json)
        }
    }

    /// Export to a human-readable text report.
    pub fn to_text_report(&self, cert: &CoverageCertificate) -> String {
        let mut report = String::new();
        let hash = self.compute_hash(cert);

        // Header
        report.push_str(&format!(
            "╔══════════════════════════════════════════════════════════════╗\n"
        ));
        report.push_str(&format!(
            "║            XR AFFORDANCE COVERAGE CERTIFICATE              ║\n"
        ));
        report.push_str(&format!(
            "╚══════════════════════════════════════════════════════════════╝\n\n"
        ));

        // Certificate ID and metadata
        report.push_str(&format!("Certificate ID:   {}\n", cert.id));
        report.push_str(&format!("Scene ID:         {}\n", cert.scene_id));
        report.push_str(&format!("Protocol:         {}\n", cert.protocol_version));
        report.push_str(&format!("Timestamp:        {}\n", cert.timestamp));
        report.push_str(&format!("Hash:             {}\n", hash));
        report.push_str("\n");

        // Grade and primary metrics
        let grade_str = match cert.grade {
            CertificateGrade::Full => "★★★ FULL",
            CertificateGrade::Partial => "★★☆ PARTIAL",
            CertificateGrade::Weak => "★☆☆ WEAK",
        };
        report.push_str(&format!("Grade:            {}\n", grade_str));
        report.push_str(&format!("Coverage (κ):     {:.6}\n", cert.kappa));
        report.push_str(&format!("Confidence (1-δ): {:.4}%\n", (1.0 - cert.delta) * 100.0));
        report.push_str("\n");

        // Error bounds
        report.push_str("── Error Bounds ──────────────────────────────────────────────\n");
        report.push_str(&format!("  ε_analytical:   {:.8}\n", cert.epsilon_analytical));
        report.push_str(&format!("  ε_estimated:    {:.8}\n", cert.epsilon_estimated));
        report.push_str(&format!("  δ:              {:.8}\n", cert.delta));
        report.push_str("\n");

        // Sampling summary
        let n_samples = cert.samples.len();
        let n_pass = cert.samples.iter().filter(|s| s.is_pass()).count();
        let n_fail = n_samples - n_pass;
        report.push_str("── Sampling Summary ──────────────────────────────────────────\n");
        report.push_str(&format!("  Total samples:  {}\n", n_samples));
        report.push_str(&format!("  Passing:        {} ({:.1}%)\n", n_pass, if n_samples > 0 { n_pass as f64 / n_samples as f64 * 100.0 } else { 0.0 }));
        report.push_str(&format!("  Failing:        {} ({:.1}%)\n", n_fail, if n_samples > 0 { n_fail as f64 / n_samples as f64 * 100.0 } else { 0.0 }));
        report.push_str("\n");

        // Verified regions
        report.push_str("── Verified Regions ──────────────────────────────────────────\n");
        report.push_str(&format!("  Count:          {}\n", cert.verified_regions.len()));
        let total_verified_vol: f64 = cert.verified_regions.iter().map(|r| r.volume()).sum();
        report.push_str(&format!("  Total volume:   {:.8}\n", total_verified_vol));
        for region in &cert.verified_regions {
            report.push_str(&format!(
                "    [{:?}] {} (vol={:.6}, lin_err={:.6})\n",
                region.proof_status, region.label, region.volume(), region.linearization_error
            ));
        }
        report.push_str("\n");

        // Violations
        report.push_str("── Violations ────────────────────────────────────────────────\n");
        if cert.violations.is_empty() {
            report.push_str("  None\n");
        } else {
            report.push_str(&format!("  Count:          {}\n", cert.violations.len()));
            for (i, v) in cert.violations.iter().enumerate() {
                report.push_str(&format!(
                    "  {}. [{:?}] {} (measure={:.6})\n",
                    i + 1, v.severity, v.description, v.estimated_measure
                ));
            }
        }
        report.push_str("\n");

        // Per-element coverage
        if self.include_element_breakdown && !cert.element_coverage.is_empty() {
            report.push_str("── Per-Element Coverage ──────────────────────────────────────\n");
            let mut sorted: Vec<_> = cert.element_coverage.iter().collect();
            sorted.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());
            for (eid, cov) in &sorted {
                let bar = coverage_bar(**cov, 30);
                report.push_str(&format!("  {} {:.4} {}\n", eid, cov, bar));
            }
            report.push_str("\n");
        }

        // Sample excerpts
        if self.include_samples && !cert.samples.is_empty() {
            report.push_str("── Sample Excerpts ──────────────────────────────────────────\n");
            let show = cert.samples.len().min(self.max_text_samples);
            for (i, s) in cert.samples.iter().take(show).enumerate() {
                let verdict = if s.is_pass() { "PASS" } else { "FAIL" };
                let params: String = s
                    .body_params
                    .iter()
                    .map(|p| format!("{:.3}", p))
                    .collect::<Vec<_>>()
                    .join(", ");
                report.push_str(&format!("  {:3}. [{}] [{}]\n", i + 1, verdict, params));
            }
            if cert.samples.len() > show {
                report.push_str(&format!(
                    "  ... and {} more samples\n",
                    cert.samples.len() - show
                ));
            }
            report.push_str("\n");
        }

        // Timing
        report.push_str("── Timing ────────────────────────────────────────────────────\n");
        report.push_str(&format!("  Total time:     {:.3}s\n", cert.total_time_s));
        report.push_str("\n");

        // Metadata
        if !cert.metadata.is_empty() {
            report.push_str("── Metadata ──────────────────────────────────────────────────\n");
            let mut keys: Vec<_> = cert.metadata.keys().collect();
            keys.sort();
            for key in keys {
                report.push_str(&format!("  {}: {}\n", key, cert.metadata[key]));
            }
            report.push_str("\n");
        }

        report.push_str("══════════════════════════════════════════════════════════════\n");

        report
    }

    /// Export to a compliance document format.
    pub fn to_compliance_document(&self, cert: &CoverageCertificate) -> String {
        let hash = self.compute_hash(cert);
        let mut doc = String::new();

        doc.push_str("ACCESSIBILITY COVERAGE COMPLIANCE DOCUMENT\n");
        doc.push_str("==========================================\n\n");

        doc.push_str("1. CERTIFICATE IDENTIFICATION\n");
        doc.push_str(&format!("   Certificate ID: {}\n", cert.id));
        doc.push_str(&format!("   Scene ID:       {}\n", cert.scene_id));
        doc.push_str(&format!("   Date:           {}\n", cert.timestamp));
        doc.push_str(&format!("   Protocol:       {}\n", cert.protocol_version));
        doc.push_str(&format!("   Integrity Hash: {}\n\n", hash));

        doc.push_str("2. COMPLIANCE SUMMARY\n");
        let compliant = matches!(
            cert.grade,
            CertificateGrade::Full | CertificateGrade::Partial
        );
        doc.push_str(&format!(
            "   Status:         {}\n",
            if compliant { "COMPLIANT" } else { "NON-COMPLIANT" }
        ));
        doc.push_str(&format!("   Grade:          {:?}\n", cert.grade));
        doc.push_str(&format!("   Coverage:       {:.4} ({:.1}%)\n", cert.kappa, cert.kappa * 100.0));
        doc.push_str(&format!("   Confidence:     {:.4} ({:.1}%)\n", 1.0 - cert.delta, (1.0 - cert.delta) * 100.0));
        doc.push_str("\n");

        doc.push_str("3. STATISTICAL GUARANTEES\n");
        doc.push_str(&format!(
            "   With probability at least {:.4}, the true accessibility\n",
            1.0 - cert.delta
        ));
        doc.push_str(&format!(
            "   coverage is within ±{:.6} of the estimated {:.4}.\n",
            cert.epsilon_estimated, cert.kappa
        ));
        doc.push_str(&format!(
            "   This gives a coverage range of [{:.4}, {:.4}].\n\n",
            (cert.kappa - cert.epsilon_estimated).max(0.0),
            (cert.kappa + cert.epsilon_estimated).min(1.0),
        ));

        doc.push_str("4. EVIDENCE SUMMARY\n");
        doc.push_str(&format!("   Samples tested:    {}\n", cert.samples.len()));
        doc.push_str(&format!("   Regions verified:  {}\n", cert.verified_regions.len()));
        doc.push_str(&format!("   Violations found:  {}\n", cert.violations.len()));
        doc.push_str(&format!("   Elements covered:  {}\n", cert.element_coverage.len()));
        doc.push_str("\n");

        if !cert.element_coverage.is_empty() {
            doc.push_str("5. PER-ELEMENT COMPLIANCE\n");
            let min_element_cov = cert
                .element_coverage
                .values()
                .copied()
                .fold(f64::INFINITY, f64::min);
            doc.push_str(&format!(
                "   Minimum element coverage: {:.4}\n",
                min_element_cov
            ));
            for (eid, &cov) in &cert.element_coverage {
                let status = if cov >= 0.90 { "PASS" } else { "NEEDS REVIEW" };
                doc.push_str(&format!("   Element {}: {:.4} [{}]\n", eid, cov, status));
            }
            doc.push_str("\n");
        }

        if !cert.violations.is_empty() {
            doc.push_str("6. VIOLATION DETAILS\n");
            for (i, v) in cert.violations.iter().enumerate() {
                doc.push_str(&format!(
                    "   {}. [{:?}] {}\n      Estimated measure: {:.6}\n",
                    i + 1, v.severity, v.description, v.estimated_measure
                ));
            }
            doc.push_str("\n");
        }

        doc.push_str("---\n");
        doc.push_str(&format!(
            "Generated by xr-certificate v{}\n",
            xr_types::PROTOCOL_VERSION
        ));

        doc
    }

    /// Compute a hash for certificate integrity verification.
    ///
    /// Uses a simple deterministic hash based on certificate content.
    /// This is not cryptographically secure — for production use,
    /// replace with SHA-256 or similar.
    pub fn compute_hash(&self, cert: &CoverageCertificate) -> String {
        let mut hasher = SimpleHasher::new();

        hasher.update_str(&cert.id.to_string());
        hasher.update_str(&cert.scene_id.to_string());
        hasher.update_str(&cert.timestamp);
        hasher.update_str(&cert.protocol_version);
        hasher.update_f64(cert.kappa);
        hasher.update_f64(cert.epsilon_analytical);
        hasher.update_f64(cert.epsilon_estimated);
        hasher.update_f64(cert.delta);
        hasher.update_u64(cert.samples.len() as u64);
        hasher.update_u64(cert.verified_regions.len() as u64);
        hasher.update_u64(cert.violations.len() as u64);

        // Include sample summaries
        let n_pass = cert.samples.iter().filter(|s| s.is_pass()).count();
        hasher.update_u64(n_pass as u64);

        // Include region volumes
        for region in &cert.verified_regions {
            hasher.update_f64(region.volume());
        }

        format!("{:016x}", hasher.finish())
    }
}

impl Default for CertificateExporter {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────── JSON Summary ───────────────────────────────────────

/// Compact JSON representation without individual sample data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateJsonSummary {
    pub id: Uuid,
    pub scene_id: Uuid,
    pub timestamp: String,
    pub protocol_version: String,
    pub grade: CertificateGrade,
    pub kappa: f64,
    pub epsilon_analytical: f64,
    pub epsilon_estimated: f64,
    pub delta: f64,
    pub total_time_s: f64,
    pub num_samples: usize,
    pub num_pass: usize,
    pub num_fail: usize,
    pub num_verified_regions: usize,
    pub num_violations: usize,
    pub element_coverage: HashMap<String, f64>,
    pub metadata: HashMap<String, String>,
}

impl CertificateJsonSummary {
    /// Create from a full certificate.
    pub fn from_cert(cert: &CoverageCertificate) -> Self {
        let n_pass = cert.samples.iter().filter(|s| s.is_pass()).count();
        let element_coverage: HashMap<String, f64> = cert
            .element_coverage
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect();

        Self {
            id: cert.id,
            scene_id: cert.scene_id,
            timestamp: cert.timestamp.clone(),
            protocol_version: cert.protocol_version.clone(),
            grade: cert.grade.clone(),
            kappa: cert.kappa,
            epsilon_analytical: cert.epsilon_analytical,
            epsilon_estimated: cert.epsilon_estimated,
            delta: cert.delta,
            total_time_s: cert.total_time_s,
            num_samples: cert.samples.len(),
            num_pass: n_pass,
            num_fail: cert.samples.len() - n_pass,
            num_verified_regions: cert.verified_regions.len(),
            num_violations: cert.violations.len(),
            element_coverage,
            metadata: cert.metadata.clone(),
        }
    }
}

// ──────────────────── Simple Hasher ──────────────────────────────────────

/// Simple non-cryptographic hash for certificate integrity.
/// Uses FNV-1a algorithm.
struct SimpleHasher {
    state: u64,
}

impl SimpleHasher {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    fn new() -> Self {
        Self {
            state: Self::FNV_OFFSET,
        }
    }

    fn update_byte(&mut self, byte: u8) {
        self.state ^= byte as u64;
        self.state = self.state.wrapping_mul(Self::FNV_PRIME);
    }

    fn update_bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.update_byte(b);
        }
    }

    fn update_str(&mut self, s: &str) {
        self.update_bytes(s.as_bytes());
    }

    fn update_f64(&mut self, v: f64) {
        self.update_bytes(&v.to_bits().to_le_bytes());
    }

    fn update_u64(&mut self, v: u64) {
        self.update_bytes(&v.to_le_bytes());
    }

    fn finish(&self) -> u64 {
        self.state
    }
}

// ──────────────────── Utility ────────────────────────────────────────────

/// Generate a simple ASCII coverage bar.
fn coverage_bar(coverage: f64, width: usize) -> String {
    let filled = (coverage * width as f64).round() as usize;
    let filled = filled.min(width);
    let empty = width - filled;
    format!("[{}{}]", "█".repeat(filled), "░".repeat(empty))
}

// ────────────────────────────── Tests ──────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use xr_types::certificate::{CertificateGrade, SampleVerdict, VerifiedRegion, ViolationSurface, ViolationSeverity};
    use xr_types::ElementId;

    fn test_element() -> ElementId {
        Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap()
    }

    fn make_cert() -> CoverageCertificate {
        let eid = test_element();
        let samples: Vec<SampleVerdict> = (0..50)
            .map(|i| {
                let t = i as f64 / 50.0;
                if t < 0.9 {
                    SampleVerdict::pass(vec![1.5 + t * 0.4, 0.25 + t * 0.15, 0.35 + t * 0.15, 0.22 + t * 0.11, 0.16 + t * 0.06], eid)
                } else {
                    SampleVerdict::fail(vec![1.5 + t * 0.4, 0.25 + t * 0.15, 0.35 + t * 0.15, 0.22 + t * 0.11, 0.16 + t * 0.06], eid, "unreachable".into())
                }
            })
            .collect();

        let mut element_coverage = HashMap::new();
        element_coverage.insert(eid, 0.9);

        CoverageCertificate {
            id: Uuid::new_v4(),
            timestamp: "2024-06-15T12:00:00Z".to_string(),
            protocol_version: "0.1.0".to_string(),
            scene_id: Uuid::new_v4(),
            samples,
            verified_regions: vec![VerifiedRegion::new(
                "green_zone",
                vec![1.5, 0.25, 0.35, 0.22, 0.16],
                vec![1.7, 0.32, 0.42, 0.27, 0.19],
                eid,
            )],
            violations: vec![],
            epsilon_analytical: 0.002,
            epsilon_estimated: 0.08,
            delta: 0.05,
            kappa: 0.92,
            grade: CertificateGrade::Partial,
            total_time_s: 3.5,
            element_coverage,
            metadata: {
                let mut m = HashMap::new();
                m.insert("generator".into(), "xr-certificate".into());
                m
            },
        }
    }

    #[test]
    fn test_to_json() {
        let cert = make_cert();
        let exporter = CertificateExporter::new();
        let json = exporter.to_json(&cert).unwrap();
        assert!(json.contains("kappa"));
        assert!(json.contains("epsilon_analytical"));
        assert!(json.contains("body_params"));
    }

    #[test]
    fn test_to_json_compact() {
        let cert = make_cert();
        let exporter = CertificateExporter::compact();
        let json = exporter.to_json(&cert).unwrap();
        assert!(json.contains("kappa"));
        assert!(json.contains("num_samples"));
        assert!(!json.contains("body_params"));
    }

    #[test]
    fn test_to_text_report() {
        let cert = make_cert();
        let exporter = CertificateExporter::new();
        let text = exporter.to_text_report(&cert);

        assert!(text.contains("COVERAGE CERTIFICATE"));
        assert!(text.contains("κ"));
        assert!(text.contains("PARTIAL"));
        assert!(text.contains("Verified Regions"));
        assert!(text.contains("Sampling Summary"));
    }

    #[test]
    fn test_compliance_document() {
        let cert = make_cert();
        let exporter = CertificateExporter::new();
        let doc = exporter.to_compliance_document(&cert);

        assert!(doc.contains("COMPLIANCE"));
        assert!(doc.contains("STATISTICAL GUARANTEES"));
        assert!(doc.contains("EVIDENCE SUMMARY"));
    }

    #[test]
    fn test_compute_hash() {
        let cert = make_cert();
        let exporter = CertificateExporter::new();
        let hash = exporter.compute_hash(&cert);

        assert_eq!(hash.len(), 16); // 16 hex chars = 64 bits
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_hash_deterministic() {
        let cert = make_cert();
        let exporter = CertificateExporter::new();
        let h1 = exporter.compute_hash(&cert);
        let h2 = exporter.compute_hash(&cert);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_changes_with_content() {
        let mut cert1 = make_cert();
        let cert2 = make_cert();
        cert1.kappa = 0.5;

        let exporter = CertificateExporter::new();
        let h1 = exporter.compute_hash(&cert1);
        let h2 = exporter.compute_hash(&cert2);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_coverage_bar() {
        let bar100 = coverage_bar(1.0, 10);
        assert!(bar100.contains("██████████"));

        let bar50 = coverage_bar(0.5, 10);
        assert!(bar50.contains("█████"));

        let bar0 = coverage_bar(0.0, 10);
        assert!(bar0.contains("░░░░░░░░░░"));
    }

    #[test]
    fn test_json_summary_from_cert() {
        let cert = make_cert();
        let summary = CertificateJsonSummary::from_cert(&cert);
        assert_eq!(summary.num_samples, cert.samples.len());
        assert_eq!(summary.kappa, cert.kappa);
        assert!(summary.num_pass + summary.num_fail == summary.num_samples);
    }

    #[test]
    fn test_text_report_empty_cert() {
        let cert = CoverageCertificate {
            id: Uuid::new_v4(),
            timestamp: "2024-01-01T00:00:00Z".into(),
            protocol_version: "0.1.0".into(),
            scene_id: Uuid::new_v4(),
            samples: vec![],
            verified_regions: vec![],
            violations: vec![],
            epsilon_analytical: 0.0,
            epsilon_estimated: 0.0,
            delta: 0.05,
            kappa: 0.0,
            grade: CertificateGrade::Weak,
            total_time_s: 0.0,
            element_coverage: HashMap::new(),
            metadata: HashMap::new(),
        };

        let exporter = CertificateExporter::new();
        let text = exporter.to_text_report(&cert);
        assert!(text.contains("WEAK"));
        assert!(text.contains("None")); // no violations
    }

    #[test]
    fn test_compliance_with_violations() {
        let mut cert = make_cert();
        let mut v = ViolationSurface::new(
            "Test violation",
            test_element(),
            ViolationSeverity::Medium,
        );
        v.estimated_measure = 0.01;
        cert.violations.push(v);

        let exporter = CertificateExporter::new();
        let doc = exporter.to_compliance_document(&cert);
        assert!(doc.contains("VIOLATION DETAILS"));
        assert!(doc.contains("Test violation"));
    }

    #[test]
    fn test_exporter_default() {
        let exporter = CertificateExporter::default();
        assert!(exporter.include_samples);
        assert!(exporter.include_element_breakdown);
    }

    #[test]
    fn test_simple_hasher() {
        let mut h1 = SimpleHasher::new();
        h1.update_str("hello");
        let hash1 = h1.finish();

        let mut h2 = SimpleHasher::new();
        h2.update_str("world");
        let hash2 = h2.finish();

        assert_ne!(hash1, hash2);

        // Same input should give same hash
        let mut h3 = SimpleHasher::new();
        h3.update_str("hello");
        assert_eq!(h1.finish(), h3.finish());
    }
}
