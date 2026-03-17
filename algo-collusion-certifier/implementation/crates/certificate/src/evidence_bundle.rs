//! Evidence bundle pipeline.
//!
//! Complete evidence packaging, verification, and serialization for
//! CollusionProof certificates.

use crate::ast::{CertificateAST, VerdictType};
use crate::checker::{ProofChecker, VerificationReport, VerificationResult};
use crate::merkle::{
    compute_data_hash, CertMerkleTree, EvidenceIntegrity, Hash, IncrementalMerkleTree,
    MerkleEvidenceItem, MerkleForest,
};
use serde::{Deserialize, Serialize};
use shared_types::OracleAccessLevel;

// ── Evidence bundle ──────────────────────────────────────────────────────────

/// A complete evidence package containing all data needed for verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertEvidenceBundle {
    pub metadata: BundleMetadata,
    pub certificate: CertificateAST,
    pub trajectory_hashes: Vec<TrajectoryHashEntry>,
    pub test_result_hashes: Vec<TestResultHashEntry>,
    pub deviation_hashes: Vec<DeviationHashEntry>,
    pub punishment_hashes: Vec<PunishmentHashEntry>,
    pub merkle_root: Hash,
    pub merkle_tree_json: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryHashEntry {
    pub ref_id: String,
    pub data_hash: Hash,
    pub segment_type: String,
    pub num_rounds: usize,
    pub num_players: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResultHashEntry {
    pub ref_id: String,
    pub test_name: String,
    pub data_hash: Hash,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviationHashEntry {
    pub ref_id: String,
    pub player: usize,
    pub data_hash: Hash,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PunishmentHashEntry {
    pub ref_id: String,
    pub player: usize,
    pub data_hash: Hash,
}

// ── Bundle metadata ──────────────────────────────────────────────────────────

/// Metadata for an evidence bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleMetadata {
    pub bundle_id: String,
    pub creation_time: String,
    pub version: String,
    pub oracle_level: OracleAccessLevel,
    pub alpha: f64,
    pub scenario: String,
    pub num_evidence_items: usize,
}

impl BundleMetadata {
    pub fn new(
        scenario: &str,
        oracle_level: OracleAccessLevel,
        alpha: f64,
    ) -> Self {
        Self {
            bundle_id: uuid::Uuid::new_v4().to_string(),
            creation_time: chrono::Utc::now().to_rfc3339(),
            version: "1.0.0".to_string(),
            oracle_level,
            alpha,
            scenario: scenario.to_string(),
            num_evidence_items: 0,
        }
    }
}

// ── Bundle builder ───────────────────────────────────────────────────────────

/// Construct evidence bundles step-by-step.
pub struct BundleBuilder {
    metadata: BundleMetadata,
    certificate: Option<CertificateAST>,
    trajectory_hashes: Vec<TrajectoryHashEntry>,
    test_result_hashes: Vec<TestResultHashEntry>,
    deviation_hashes: Vec<DeviationHashEntry>,
    punishment_hashes: Vec<PunishmentHashEntry>,
    merkle_tree: IncrementalMerkleTree,
}

impl BundleBuilder {
    pub fn new(
        scenario: &str,
        oracle_level: OracleAccessLevel,
        alpha: f64,
    ) -> Self {
        Self {
            metadata: BundleMetadata::new(scenario, oracle_level, alpha),
            certificate: None,
            trajectory_hashes: Vec::new(),
            test_result_hashes: Vec::new(),
            deviation_hashes: Vec::new(),
            punishment_hashes: Vec::new(),
            merkle_tree: IncrementalMerkleTree::new(),
        }
    }

    /// Add hashed trajectory data (not raw data).
    pub fn add_trajectory(
        &mut self,
        ref_id: &str,
        segment_type: &str,
        num_rounds: usize,
        num_players: usize,
        data: &[u8],
    ) -> &mut Self {
        let hash = compute_data_hash(data);
        self.trajectory_hashes.push(TrajectoryHashEntry {
            ref_id: ref_id.to_string(),
            data_hash: hash.clone(),
            segment_type: segment_type.to_string(),
            num_rounds,
            num_players,
        });
        self.merkle_tree
            .add_item(MerkleEvidenceItem::from_hash(ref_id, "trajectory", &hash));
        self
    }

    /// Add a test result entry.
    pub fn add_test_result(
        &mut self,
        ref_id: &str,
        test_name: &str,
        data: &[u8],
    ) -> &mut Self {
        let hash = compute_data_hash(data);
        self.test_result_hashes.push(TestResultHashEntry {
            ref_id: ref_id.to_string(),
            test_name: test_name.to_string(),
            data_hash: hash.clone(),
        });
        self.merkle_tree
            .add_item(MerkleEvidenceItem::from_hash(ref_id, "test_result", &hash));
        self
    }

    /// Add deviation analysis results.
    pub fn add_deviation_result(
        &mut self,
        ref_id: &str,
        player: usize,
        data: &[u8],
    ) -> &mut Self {
        let hash = compute_data_hash(data);
        self.deviation_hashes.push(DeviationHashEntry {
            ref_id: ref_id.to_string(),
            player,
            data_hash: hash.clone(),
        });
        self.merkle_tree.add_item(MerkleEvidenceItem::from_hash(
            ref_id, "deviation", &hash,
        ));
        self
    }

    /// Add punishment detection results.
    pub fn add_punishment_result(
        &mut self,
        ref_id: &str,
        player: usize,
        data: &[u8],
    ) -> &mut Self {
        let hash = compute_data_hash(data);
        self.punishment_hashes.push(PunishmentHashEntry {
            ref_id: ref_id.to_string(),
            player,
            data_hash: hash.clone(),
        });
        self.merkle_tree.add_item(MerkleEvidenceItem::from_hash(
            ref_id, "punishment", &hash,
        ));
        self
    }

    /// Set the certificate.
    pub fn set_certificate(&mut self, cert: CertificateAST) -> &mut Self {
        // Also add certificate hash to Merkle tree
        if let Ok(cert_json) = serde_json::to_string(&cert) {
            let hash = compute_data_hash(cert_json.as_bytes());
            self.merkle_tree.add_item(MerkleEvidenceItem::from_hash(
                "__certificate__",
                "certificate",
                &hash,
            ));
        }
        self.certificate = Some(cert);
        self
    }

    /// Build the evidence bundle, computing the Merkle root.
    pub fn build(mut self) -> Result<CertEvidenceBundle, String> {
        let certificate = self
            .certificate
            .ok_or_else(|| "Certificate is required".to_string())?;

        let merkle_root = self
            .merkle_tree
            .root_hash()
            .ok_or_else(|| "No evidence items to build Merkle tree".to_string())?;

        let tree = self.merkle_tree.finalize().clone();
        let merkle_tree_json = serde_json::to_string(&tree)
            .map_err(|e| format!("Failed to serialize Merkle tree: {}", e))?;

        let total_items = self.trajectory_hashes.len()
            + self.test_result_hashes.len()
            + self.deviation_hashes.len()
            + self.punishment_hashes.len()
            + 1; // certificate itself

        let mut metadata = self.metadata;
        metadata.num_evidence_items = total_items;

        Ok(CertEvidenceBundle {
            metadata,
            certificate,
            trajectory_hashes: self.trajectory_hashes,
            test_result_hashes: self.test_result_hashes,
            deviation_hashes: self.deviation_hashes,
            punishment_hashes: self.punishment_hashes,
            merkle_root,
            merkle_tree_json,
        })
    }
}

// ── Bundle verifier ──────────────────────────────────────────────────────────

/// Verify the integrity and validity of an evidence bundle.
pub struct BundleVerifier {
    checker: ProofChecker,
}

impl BundleVerifier {
    pub fn new() -> Self {
        Self {
            checker: ProofChecker::new(),
        }
    }

    pub fn with_checker(checker: ProofChecker) -> Self {
        Self { checker }
    }

    /// Verify the complete bundle: Merkle integrity + certificate validity.
    pub fn verify(&self, bundle: &CertEvidenceBundle) -> BundleVerificationResult {
        let mut issues = Vec::new();

        // 1. Verify Merkle tree integrity
        let merkle_tree: Result<CertMerkleTree, _> =
            serde_json::from_str(&bundle.merkle_tree_json);
        let merkle_valid = match merkle_tree {
            Ok(tree) => {
                let integrity = EvidenceIntegrity::verify_tree(&tree);
                if !integrity {
                    issues.push("Merkle tree internal integrity check failed".to_string());
                }
                let root_matches = tree
                    .root_hash()
                    .map(|r| r == bundle.merkle_root)
                    .unwrap_or(false);
                if !root_matches {
                    issues.push("Merkle root hash mismatch".to_string());
                }
                integrity && root_matches
            }
            Err(e) => {
                issues.push(format!("Failed to deserialize Merkle tree: {}", e));
                false
            }
        };

        // 2. Verify certificate
        let cert_result = self.checker.check_certificate(&bundle.certificate);
        let cert_valid = cert_result.is_valid();
        if !cert_valid {
            if let VerificationResult::Invalid(e) = &cert_result {
                issues.push(format!("Certificate verification failed: {}", e.message));
            }
        }

        // 3. Cross-check: verify certificate refs match evidence
        let cert_refs: std::collections::HashSet<String> = bundle
            .certificate
            .declared_refs()
            .into_iter()
            .collect();
        let evidence_refs: std::collections::HashSet<String> = bundle
            .trajectory_hashes
            .iter()
            .map(|h| h.ref_id.clone())
            .chain(bundle.test_result_hashes.iter().map(|h| h.ref_id.clone()))
            .chain(bundle.deviation_hashes.iter().map(|h| h.ref_id.clone()))
            .chain(bundle.punishment_hashes.iter().map(|h| h.ref_id.clone()))
            .collect();

        // Check that data-related refs in certificate have corresponding evidence
        for cert_ref in &cert_refs {
            if cert_ref.starts_with("traj_") && !evidence_refs.contains(cert_ref) {
                issues.push(format!(
                    "Certificate reference '{}' has no corresponding evidence",
                    cert_ref
                ));
            }
        }

        // 4. Verify metadata consistency
        if bundle.metadata.oracle_level != bundle.certificate.header.oracle_level {
            issues.push("Oracle level mismatch between metadata and certificate".to_string());
        }

        let verification_report = match cert_result {
            VerificationResult::Valid(report) => Some(report),
            _ => None,
        };

        BundleVerificationResult {
            merkle_valid,
            certificate_valid: cert_valid,
            cross_check_passed: issues
                .iter()
                .all(|i| !i.contains("no corresponding evidence")),
            metadata_consistent: bundle.metadata.oracle_level
                == bundle.certificate.header.oracle_level,
            overall_valid: merkle_valid && cert_valid && issues.is_empty(),
            issues,
            verification_report,
        }
    }
}

impl Default for BundleVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of verifying an evidence bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleVerificationResult {
    pub merkle_valid: bool,
    pub certificate_valid: bool,
    pub cross_check_passed: bool,
    pub metadata_consistent: bool,
    pub overall_valid: bool,
    pub issues: Vec<String>,
    pub verification_report: Option<VerificationReport>,
}

// ── Standalone verifier ──────────────────────────────────────────────────────

/// Verify a bundle without any CollusionProof code beyond the proof checker kernel.
/// Suitable for independent third-party verification.
pub struct StandaloneVerifier;

impl StandaloneVerifier {
    /// Verify a serialized evidence bundle.
    pub fn verify_from_json(json: &str) -> StandaloneVerificationResult {
        // Parse the bundle
        let bundle: Result<CertEvidenceBundle, _> = serde_json::from_str(json);
        let bundle = match bundle {
            Ok(b) => b,
            Err(e) => {
                return StandaloneVerificationResult {
                    valid: false,
                    verdict: None,
                    confidence: None,
                    message: format!("Failed to parse bundle: {}", e),
                };
            }
        };

        Self::verify_bundle(&bundle)
    }

    /// Verify an evidence bundle directly.
    pub fn verify_bundle(bundle: &CertEvidenceBundle) -> StandaloneVerificationResult {
        let checker = ProofChecker::new();
        let result = checker.check_certificate(&bundle.certificate);

        match result {
            VerificationResult::Valid(report) => StandaloneVerificationResult {
                valid: true,
                verdict: report.verdict,
                confidence: report.verdict_confidence,
                message: format!(
                    "Certificate valid: {} steps verified, verdict={:?}",
                    report.verified_steps, report.verdict
                ),
            },
            VerificationResult::Invalid(error) => StandaloneVerificationResult {
                valid: false,
                verdict: None,
                confidence: None,
                message: format!("Certificate invalid: {}", error.message),
            },
        }
    }
}

/// Result of standalone verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandaloneVerificationResult {
    pub valid: bool,
    pub verdict: Option<VerdictType>,
    pub confidence: Option<f64>,
    pub message: String,
}

// ── Bundle serializer ────────────────────────────────────────────────────────

/// Serialize evidence bundles to multiple formats.
pub struct BundleSerializer;

impl BundleSerializer {
    /// Serialize to pretty-printed JSON.
    pub fn to_json(bundle: &CertEvidenceBundle) -> Result<String, String> {
        serde_json::to_string_pretty(bundle)
            .map_err(|e| format!("JSON serialization error: {}", e))
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> Result<CertEvidenceBundle, String> {
        serde_json::from_str(json).map_err(|e| format!("JSON deserialization error: {}", e))
    }

    /// Serialize to compact binary.
    pub fn to_binary(bundle: &CertEvidenceBundle) -> Result<Vec<u8>, String> {
        bincode::serialize(bundle)
            .map_err(|e| format!("Binary serialization error: {}", e))
    }

    /// Deserialize from binary.
    pub fn from_binary(data: &[u8]) -> Result<CertEvidenceBundle, String> {
        bincode::deserialize(data)
            .map_err(|e| format!("Binary deserialization error: {}", e))
    }

    /// Compute the size of the serialized bundle in bytes.
    pub fn estimate_json_size(bundle: &CertEvidenceBundle) -> usize {
        serde_json::to_string(bundle)
            .map(|s| s.len())
            .unwrap_or(0)
    }

    /// Compute the size of the binary bundle in bytes.
    pub fn estimate_binary_size(bundle: &CertEvidenceBundle) -> usize {
        bincode::serialized_size(bundle).unwrap_or(0) as usize
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;
    use shared_types::OracleAccessLevel;

    fn make_test_bundle() -> CertEvidenceBundle {
        let mut builder =
            BundleBuilder::new("test_scenario", OracleAccessLevel::Layer0, 0.05);

        builder.add_trajectory(
            "traj_testing",
            "testing",
            500,
            2,
            b"trajectory_data_here",
        );
        builder.add_test_result("test_0", "PriceCorrelation", b"test_result_data");

        // Build a valid certificate
        let header =
            CertificateHeader::new("test_scenario", OracleAccessLevel::Layer0, 0.05);
        let mut body = CertificateBody::new();
        body.push(ProofStep::DataDeclaration(
            TrajectoryRef::new("traj_testing"),
            SegmentSpec::new("testing", 0, 500, &compute_data_hash(b"trajectory_data_here"), 2),
        ));
        body.push(ProofStep::StatisticalTest(
            TestRef::new("test_0"),
            TestType::new("PriceCorrelation", "layer0"),
            Statistic::new(3.5),
            PValueWrapper::new(0.001),
        ));
        body.push(ProofStep::Inference(
            InferenceRef::new("inf_0"),
            Rule::new("VerdictDerivation"),
            Premises::new(vec!["test_0".into()]),
            Conclusion::new("Evidence assessed"),
        ));
        body.push(ProofStep::Verdict(
            VerdictType::Collusive,
            Confidence::new(0.95),
            SupportingRefs::new(vec!["test_0".into(), "inf_0".into()]),
        ));
        let cert = CertificateAST::new(header, body);

        builder.set_certificate(cert);
        builder.build().unwrap()
    }

    #[test]
    fn test_bundle_builder() {
        let bundle = make_test_bundle();
        assert!(!bundle.merkle_root.is_empty());
        assert_eq!(bundle.trajectory_hashes.len(), 1);
        assert_eq!(bundle.test_result_hashes.len(), 1);
    }

    #[test]
    fn test_bundle_builder_no_certificate() {
        let builder =
            BundleBuilder::new("test", OracleAccessLevel::Layer0, 0.05);
        let result = builder.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_bundle_verifier() {
        let bundle = make_test_bundle();
        let verifier = BundleVerifier::new();
        let result = verifier.verify(&bundle);
        assert!(result.certificate_valid);
        assert!(result.metadata_consistent);
    }

    #[test]
    fn test_bundle_metadata() {
        let meta = BundleMetadata::new("scn", OracleAccessLevel::Layer1, 0.05);
        assert!(!meta.bundle_id.is_empty());
        assert_eq!(meta.scenario, "scn");
        assert_eq!(meta.oracle_level, OracleAccessLevel::Layer1);
    }

    #[test]
    fn test_standalone_verifier() {
        let bundle = make_test_bundle();
        let result = StandaloneVerifier::verify_bundle(&bundle);
        assert!(result.valid);
        assert_eq!(result.verdict, Some(VerdictType::Collusive));
    }

    #[test]
    fn test_standalone_verifier_from_json() {
        let bundle = make_test_bundle();
        let json = BundleSerializer::to_json(&bundle).unwrap();
        let result = StandaloneVerifier::verify_from_json(&json);
        assert!(result.valid);
    }

    #[test]
    fn test_standalone_verifier_invalid_json() {
        let result = StandaloneVerifier::verify_from_json("not valid json");
        assert!(!result.valid);
    }

    #[test]
    fn test_bundle_serializer_json_roundtrip() {
        let bundle = make_test_bundle();
        let json = BundleSerializer::to_json(&bundle).unwrap();
        let bundle2 = BundleSerializer::from_json(&json).unwrap();
        assert_eq!(bundle2.metadata.scenario, bundle.metadata.scenario);
        assert_eq!(bundle2.merkle_root, bundle.merkle_root);
    }

    #[test]
    fn test_bundle_serializer_binary_roundtrip() {
        let bundle = make_test_bundle();
        let bin = BundleSerializer::to_binary(&bundle).unwrap();
        let bundle2 = BundleSerializer::from_binary(&bin).unwrap();
        assert_eq!(bundle2.merkle_root, bundle.merkle_root);
    }

    #[test]
    fn test_bundle_size_estimation() {
        let bundle = make_test_bundle();
        let json_size = BundleSerializer::estimate_json_size(&bundle);
        let bin_size = BundleSerializer::estimate_binary_size(&bundle);
        assert!(json_size > 0);
        assert!(bin_size > 0);
        assert!(bin_size < json_size);
    }

    #[test]
    fn test_bundle_with_deviation_data() {
        let mut builder =
            BundleBuilder::new("test", OracleAccessLevel::Layer1, 0.05);
        builder.add_trajectory("traj_0", "testing", 100, 2, b"data");
        builder.add_deviation_result("dev_0", 0, b"deviation_data");

        let header = CertificateHeader::new("test", OracleAccessLevel::Layer1, 0.05);
        let mut body = CertificateBody::new();
        body.push(ProofStep::Verdict(
            VerdictType::Competitive,
            Confidence::new(0.5),
            SupportingRefs::new(vec![]),
        ));
        let cert = CertificateAST::new(header, body);
        builder.set_certificate(cert);

        let bundle = builder.build().unwrap();
        assert_eq!(bundle.deviation_hashes.len(), 1);
        assert_eq!(bundle.deviation_hashes[0].player, 0);
    }

    #[test]
    fn test_bundle_with_punishment_data() {
        let mut builder =
            BundleBuilder::new("test", OracleAccessLevel::Layer2, 0.05);
        builder.add_trajectory("traj_0", "testing", 100, 2, b"data");
        builder.add_punishment_result("pun_0", 0, b"punishment_data");

        let header = CertificateHeader::new("test", OracleAccessLevel::Layer2, 0.05);
        let mut body = CertificateBody::new();
        body.push(ProofStep::Verdict(
            VerdictType::Competitive,
            Confidence::new(0.5),
            SupportingRefs::new(vec![]),
        ));
        let cert = CertificateAST::new(header, body);
        builder.set_certificate(cert);

        let bundle = builder.build().unwrap();
        assert_eq!(bundle.punishment_hashes.len(), 1);
    }

    #[test]
    fn test_trajectory_hash_entry() {
        let entry = TrajectoryHashEntry {
            ref_id: "traj_0".to_string(),
            data_hash: compute_data_hash(b"data"),
            segment_type: "testing".to_string(),
            num_rounds: 500,
            num_players: 2,
        };
        assert_eq!(entry.num_rounds, 500);
    }

    #[test]
    fn test_bundle_verifier_oracle_mismatch() {
        let mut builder =
            BundleBuilder::new("test", OracleAccessLevel::Layer0, 0.05);
        builder.add_trajectory("traj_0", "testing", 100, 2, b"data");

        // Certificate has Layer1 but metadata has Layer0
        let header = CertificateHeader::new("test", OracleAccessLevel::Layer1, 0.05);
        let mut body = CertificateBody::new();
        body.push(ProofStep::Verdict(
            VerdictType::Competitive,
            Confidence::new(0.5),
            SupportingRefs::new(vec![]),
        ));
        let cert = CertificateAST::new(header, body);
        builder.set_certificate(cert);

        let bundle = builder.build().unwrap();
        let verifier = BundleVerifier::new();
        let result = verifier.verify(&bundle);
        assert!(!result.metadata_consistent);
    }
}
