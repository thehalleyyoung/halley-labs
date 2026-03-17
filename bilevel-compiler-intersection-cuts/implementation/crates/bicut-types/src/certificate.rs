use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt;

use crate::signature::{CqStatus, LowerLevelType};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReformulationType {
    KKT,
    StrongDuality,
    ValueFunction,
    ColumnConstraintGeneration,
    Hybrid,
}

impl fmt::Display for ReformulationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KKT => write!(f, "KKT"),
            Self::StrongDuality => write!(f, "StrongDuality"),
            Self::ValueFunction => write!(f, "ValueFunction"),
            Self::ColumnConstraintGeneration => write!(f, "CCG"),
            Self::Hybrid => write!(f, "Hybrid"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationTier {
    Syntactic,
    LpBased,
    SamplingBased,
    Formal,
}

impl fmt::Display for VerificationTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syntactic => write!(f, "Syntactic"),
            Self::LpBased => write!(f, "LP-based"),
            Self::SamplingBased => write!(f, "Sampling"),
            Self::Formal => write!(f, "Formal"),
        }
    }
}

impl VerificationTier {
    pub fn confidence(&self) -> f64 {
        match self {
            Self::Syntactic => 0.7,
            Self::LpBased => 0.9,
            Self::SamplingBased => 0.95,
            Self::Formal => 1.0,
        }
    }
    pub fn is_sufficient(&self) -> bool {
        matches!(self, Self::LpBased | Self::SamplingBased | Self::Formal)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CqVerification {
    pub status: CqStatus,
    pub tier: VerificationTier,
    pub verified_at_points: usize,
    pub failures_found: usize,
    pub details: String,
}

impl CqVerification {
    pub fn new(status: CqStatus, tier: VerificationTier) -> Self {
        Self {
            status,
            tier,
            verified_at_points: 0,
            failures_found: 0,
            details: String::new(),
        }
    }

    pub fn with_details(mut self, points: usize, failures: usize, details: &str) -> Self {
        self.verified_at_points = points;
        self.failures_found = failures;
        self.details = details.to_string();
        self
    }

    pub fn is_valid(&self) -> bool {
        self.failures_found == 0 && self.status.is_sufficient_for_kkt()
    }

    pub fn confidence(&self) -> f64 {
        if self.failures_found > 0 {
            return 0.0;
        }
        self.tier.confidence()
    }
}

impl fmt::Display for CqVerification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CQ({} via {}, {}pts, {}fails)",
            self.status, self.tier, self.verified_at_points, self.failures_found
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundednessProof {
    pub lower_level_bounded: bool,
    pub upper_level_bounded: bool,
    pub proof_method: String,
    pub bound_values: Vec<(String, f64, f64)>,
    pub details: String,
}

impl BoundednessProof {
    pub fn new(lower_bounded: bool, upper_bounded: bool, method: &str) -> Self {
        Self {
            lower_level_bounded: lower_bounded,
            upper_level_bounded: upper_bounded,
            proof_method: method.to_string(),
            bound_values: Vec::new(),
            details: String::new(),
        }
    }
    pub fn add_bound(&mut self, name: &str, lb: f64, ub: f64) {
        self.bound_values.push((name.to_string(), lb, ub));
    }
    pub fn is_fully_bounded(&self) -> bool {
        self.lower_level_bounded && self.upper_level_bounded
    }
}

impl fmt::Display for BoundednessProof {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Boundedness(lower={}, upper={}, method={})",
            self.lower_level_bounded, self.upper_level_bounded, self.proof_method
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BigMDerivation {
    pub values: HashMap<String, f64>,
    pub method: String,
    pub is_tight: bool,
    pub safety_margin: f64,
}

impl BigMDerivation {
    pub fn new(method: &str) -> Self {
        Self {
            values: HashMap::new(),
            method: method.to_string(),
            is_tight: false,
            safety_margin: 1e-4,
        }
    }

    pub fn add_value(&mut self, name: &str, value: f64) {
        self.values.insert(name.to_string(), value);
    }

    pub fn max_value(&self) -> f64 {
        self.values.values().copied().fold(0.0f64, f64::max)
    }

    pub fn num_big_m_values(&self) -> usize {
        self.values.len()
    }
}

impl fmt::Display for BigMDerivation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BigM(method={}, count={}, max={:.2})",
            self.method,
            self.values.len(),
            self.max_value()
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpotCheckResult {
    pub point_index: usize,
    pub bilevel_feasible: bool,
    pub reformulation_feasible: bool,
    pub objective_match: bool,
    pub tolerance: f64,
    pub details: String,
}

impl SpotCheckResult {
    pub fn new(idx: usize, bf: bool, rf: bool, om: bool, tol: f64) -> Self {
        Self {
            point_index: idx,
            bilevel_feasible: bf,
            reformulation_feasible: rf,
            objective_match: om,
            tolerance: tol,
            details: String::new(),
        }
    }
    pub fn is_consistent(&self) -> bool {
        self.bilevel_feasible == self.reformulation_feasible && self.objective_match
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateEntry {
    pub name: String,
    pub verified: bool,
    pub tier: VerificationTier,
    pub details: String,
    pub timestamp: String,
}

impl CertificateEntry {
    pub fn new(name: &str, verified: bool, tier: VerificationTier) -> Self {
        Self {
            name: name.to_string(),
            verified,
            tier,
            details: String::new(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }
    pub fn with_details(mut self, details: &str) -> Self {
        self.details = details.to_string();
        self
    }
}

impl fmt::Display for CertificateEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {} ({}): {}",
            if self.verified { "✓" } else { "✗" },
            self.name,
            self.tier,
            if self.details.is_empty() {
                "no details"
            } else {
                &self.details
            }
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Certificate {
    pub problem_hash: String,
    pub problem_name: String,
    pub reformulation_type: ReformulationType,
    pub lower_level_type: LowerLevelType,
    pub cq_verification: CqVerification,
    pub boundedness_proof: BoundednessProof,
    pub big_m_derivation: Option<BigMDerivation>,
    pub spot_checks: Vec<SpotCheckResult>,
    pub entries: Vec<CertificateEntry>,
    pub is_valid: bool,
    pub created_at: String,
    pub version: String,
}

impl Certificate {
    pub fn new(problem_name: &str, problem_data: &[u8], reformulation: ReformulationType) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(problem_data);
        let hash = hex::encode(hasher.finalize());
        Self {
            problem_hash: hash,
            problem_name: problem_name.to_string(),
            reformulation_type: reformulation,
            lower_level_type: LowerLevelType::LP,
            cq_verification: CqVerification::new(CqStatus::Unknown, VerificationTier::Syntactic),
            boundedness_proof: BoundednessProof::new(false, false, "none"),
            big_m_derivation: None,
            spot_checks: Vec::new(),
            entries: Vec::new(),
            is_valid: false,
            created_at: chrono::Utc::now().to_rfc3339(),
            version: "1.0.0".to_string(),
        }
    }

    pub fn add_entry(&mut self, entry: CertificateEntry) {
        if !entry.verified {
            self.is_valid = false;
        }
        self.entries.push(entry);
    }

    pub fn add_spot_check(&mut self, check: SpotCheckResult) {
        if !check.is_consistent() {
            self.is_valid = false;
        }
        self.spot_checks.push(check);
    }

    pub fn finalize(&mut self) {
        self.is_valid = self.cq_verification.is_valid()
            && self.boundedness_proof.is_fully_bounded()
            && self.entries.iter().all(|e| e.verified)
            && self.spot_checks.iter().all(|s| s.is_consistent());
    }

    pub fn num_checks(&self) -> usize {
        self.entries.len() + self.spot_checks.len()
    }

    pub fn num_passed(&self) -> usize {
        self.entries.iter().filter(|e| e.verified).count()
            + self
                .spot_checks
                .iter()
                .filter(|s| s.is_consistent())
                .count()
    }

    pub fn pass_rate(&self) -> f64 {
        let total = self.num_checks();
        if total == 0 {
            return 1.0;
        }
        self.num_passed() as f64 / total as f64
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    pub fn summary(&self) -> String {
        format!(
            "Certificate({}, {}, valid={}, {}/{} checks passed)",
            self.problem_name,
            self.reformulation_type,
            self.is_valid,
            self.num_passed(),
            self.num_checks()
        )
    }

    pub fn confidence(&self) -> f64 {
        if !self.is_valid {
            return 0.0;
        }
        let cq_conf = self.cq_verification.confidence();
        let entry_conf = if self.entries.is_empty() {
            1.0
        } else {
            self.entries
                .iter()
                .map(|e| e.tier.confidence())
                .fold(f64::MAX, f64::min)
        };
        cq_conf.min(entry_conf)
    }
}

impl fmt::Display for Certificate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Certificate for '{}' ===", self.problem_name)?;
        writeln!(f, "Hash: {}", &self.problem_hash[..16])?;
        writeln!(f, "Reformulation: {}", self.reformulation_type)?;
        writeln!(f, "CQ: {}", self.cq_verification)?;
        writeln!(f, "Boundedness: {}", self.boundedness_proof)?;
        if let Some(ref bm) = self.big_m_derivation {
            writeln!(f, "BigM: {}", bm)?;
        }
        writeln!(
            f,
            "Valid: {} ({}/{})",
            self.is_valid,
            self.num_passed(),
            self.num_checks()
        )?;
        for e in &self.entries {
            writeln!(f, "  {}", e)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_certificate_creation() {
        let cert = Certificate::new("test", b"data", ReformulationType::KKT);
        assert!(!cert.is_valid);
        assert_eq!(cert.problem_name, "test");
    }

    #[test]
    fn test_hash() {
        let c1 = Certificate::new("t", b"data1", ReformulationType::KKT);
        let c2 = Certificate::new("t", b"data2", ReformulationType::KKT);
        assert_ne!(c1.problem_hash, c2.problem_hash);
    }

    #[test]
    fn test_finalize() {
        let mut cert = Certificate::new("test", b"data", ReformulationType::StrongDuality);
        cert.cq_verification = CqVerification::new(CqStatus::LICQ, VerificationTier::LpBased);
        cert.boundedness_proof = BoundednessProof::new(true, true, "lp");
        cert.finalize();
        assert!(cert.is_valid);
    }

    #[test]
    fn test_spot_check() {
        let sc = SpotCheckResult::new(0, true, true, true, 1e-6);
        assert!(sc.is_consistent());
    }

    #[test]
    fn test_entry() {
        let e = CertificateEntry::new("cq_check", true, VerificationTier::LpBased);
        assert!(e.verified);
    }

    #[test]
    fn test_big_m() {
        let mut bm = BigMDerivation::new("lp_tightening");
        bm.add_value("comp1", 1000.0);
        bm.add_value("comp2", 500.0);
        assert!((bm.max_value() - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_json() {
        let cert = Certificate::new("test", b"data", ReformulationType::KKT);
        let json = cert.to_json();
        assert!(json.contains("test"));
    }

    #[test]
    fn test_pass_rate() {
        let mut cert = Certificate::new("t", b"d", ReformulationType::KKT);
        cert.add_entry(CertificateEntry::new(
            "a",
            true,
            VerificationTier::Syntactic,
        ));
        cert.add_entry(CertificateEntry::new(
            "b",
            false,
            VerificationTier::Syntactic,
        ));
        assert!((cert.pass_rate() - 0.5).abs() < 1e-10);
    }
}
