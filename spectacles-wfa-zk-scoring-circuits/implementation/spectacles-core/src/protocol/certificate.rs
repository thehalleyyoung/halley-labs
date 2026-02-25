use std::collections::HashMap;
use std::fmt;
use serde::{Serialize, Deserialize};

// ─────────────────────────────────────────────────────────────
// CertificateError
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum CertificateError {
    InvalidFormat,
    ExpiredCertificate,
    IntegrityCheckFailed,
    MissingField(String),
    DeserializationError(String),
    ChainBroken,
}

impl fmt::Display for CertificateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CertificateError::InvalidFormat => write!(f, "InvalidFormat"),
            CertificateError::ExpiredCertificate => write!(f, "ExpiredCertificate"),
            CertificateError::IntegrityCheckFailed => write!(f, "IntegrityCheckFailed"),
            CertificateError::MissingField(s) => write!(f, "MissingField({})", s),
            CertificateError::DeserializationError(s) => write!(f, "DeserializationError({})", s),
            CertificateError::ChainBroken => write!(f, "ChainBroken"),
        }
    }
}

impl std::error::Error for CertificateError {}

// ─────────────────────────────────────────────────────────────
// CertifiedResult
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CertifiedResult {
    pub score: f64,
    pub score_field_element: u64,
    pub confidence: Option<f64>,
    pub details: HashMap<String, String>,
}

impl CertifiedResult {
    pub fn new(score: f64, score_field_element: u64) -> Self {
        Self {
            score,
            score_field_element,
            confidence: None,
            details: HashMap::new(),
        }
    }

    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = Some(confidence);
        self
    }

    pub fn with_detail(mut self, key: &str, value: &str) -> Self {
        self.details.insert(key.to_string(), value.to_string());
        self
    }
}

// ─────────────────────────────────────────────────────────────
// PSIAttestation
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PSIAttestation {
    pub contamination_bound: f64,
    pub ngram_size: usize,
    pub intersection_cardinality_bound: u64,
    pub protocol_hash: [u8; 32],
    pub threshold_satisfied: bool,
}

impl PSIAttestation {
    pub fn new(
        contamination_bound: f64,
        ngram_size: usize,
        intersection_cardinality_bound: u64,
        protocol_hash: [u8; 32],
        threshold_satisfied: bool,
    ) -> Self {
        Self {
            contamination_bound,
            ngram_size,
            intersection_cardinality_bound,
            protocol_hash,
            threshold_satisfied,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// CommitmentOpening
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommitmentOpening {
    pub name: String,
    pub commitment_hash: [u8; 32],
    pub revealed_value: Vec<u8>,
    pub randomness: Vec<u8>,
    pub scheme: String,
}

impl CommitmentOpening {
    pub fn new(
        name: &str,
        commitment_hash: [u8; 32],
        revealed_value: Vec<u8>,
        randomness: Vec<u8>,
        scheme: &str,
    ) -> Self {
        Self {
            name: name.to_string(),
            commitment_hash,
            revealed_value,
            randomness,
            scheme: scheme.to_string(),
        }
    }

    /// Verify this opening against its commitment hash using BLAKE3.
    pub fn verify(&self) -> bool {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.randomness);
        hasher.update(&self.revealed_value);
        let computed = hasher.finalize();
        computed.as_bytes() == &self.commitment_hash
    }
}

// ─────────────────────────────────────────────────────────────
// CertificateMetadata
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CertificateMetadata {
    pub proving_time_ms: u64,
    pub verification_time_ms: Option<u64>,
    pub trace_width: usize,
    pub trace_length: usize,
    pub constraint_count: usize,
    pub security_bits: u32,
    pub proof_size_bytes: usize,
}

impl Default for CertificateMetadata {
    fn default() -> Self {
        Self {
            proving_time_ms: 0,
            verification_time_ms: None,
            trace_width: 0,
            trace_length: 0,
            constraint_count: 0,
            security_bits: 128,
            proof_size_bytes: 0,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// VerificationCheck / CertificateVerification
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerificationCheck {
    pub name: String,
    pub passed: bool,
    pub details: String,
}

impl VerificationCheck {
    pub fn new(name: &str, passed: bool, details: &str) -> Self {
        Self {
            name: name.to_string(),
            passed,
            details: details.to_string(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CertificateVerification {
    pub is_valid: bool,
    pub checks: Vec<VerificationCheck>,
}

impl CertificateVerification {
    pub fn new() -> Self {
        Self {
            is_valid: true,
            checks: Vec::new(),
        }
    }

    pub fn add_check(&mut self, check: VerificationCheck) {
        if !check.passed {
            self.is_valid = false;
        }
        self.checks.push(check);
    }
}

impl Default for CertificateVerification {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────
// EvaluationCertificate
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationCertificate {
    pub version: u32,
    pub certificate_id: String,
    pub metric_name: String,
    pub metric_hash: [u8; 32],
    pub model_id: String,
    pub benchmark_id: String,
    pub evaluation_result: CertifiedResult,
    pub stark_proof_hash: [u8; 32],
    pub stark_proof: Option<Vec<u8>>,
    pub psi_attestation: Option<PSIAttestation>,
    pub commitment_openings: Vec<CommitmentOpening>,
    pub timestamp: String,
    pub valid_until: Option<String>,
    pub issuer: String,
    pub signature: Option<Vec<u8>>,
    pub metadata: CertificateMetadata,
}

impl EvaluationCertificate {
    /// Create a new evaluation certificate with the core fields filled in.
    pub fn new(
        metric: &str,
        model: &str,
        benchmark: &str,
        result: CertifiedResult,
    ) -> Self {
        let id = uuid::Uuid::new_v4().to_string();
        let metric_hash = *blake3::hash(metric.as_bytes()).as_bytes();
        let now = chrono::Utc::now().to_rfc3339();

        Self {
            version: 1,
            certificate_id: id,
            metric_name: metric.to_string(),
            metric_hash,
            model_id: model.to_string(),
            benchmark_id: benchmark.to_string(),
            evaluation_result: result,
            stark_proof_hash: [0u8; 32],
            stark_proof: None,
            psi_attestation: None,
            commitment_openings: Vec::new(),
            timestamp: now,
            valid_until: None,
            issuer: "spectacles-core".to_string(),
            signature: None,
            metadata: CertificateMetadata::default(),
        }
    }

    /// Attach a serialized STARK proof.
    pub fn with_stark_proof(&mut self, proof_bytes: Vec<u8>) -> &mut Self {
        self.stark_proof_hash = *blake3::hash(&proof_bytes).as_bytes();
        self.metadata.proof_size_bytes = proof_bytes.len();
        self.stark_proof = Some(proof_bytes);
        self
    }

    /// Attach a PSI attestation.
    pub fn with_psi_attestation(&mut self, attestation: PSIAttestation) -> &mut Self {
        self.psi_attestation = Some(attestation);
        self
    }

    /// Add a commitment opening.
    pub fn add_commitment_opening(&mut self, opening: CommitmentOpening) -> &mut Self {
        self.commitment_openings.push(opening);
        self
    }

    /// Set certificate validity duration from now.
    pub fn set_validity(&mut self, duration_hours: u64) -> &mut Self {
        let valid_until = chrono::Utc::now()
            + chrono::Duration::hours(duration_hours as i64);
        self.valid_until = Some(valid_until.to_rfc3339());
        self
    }

    /// Finalize the certificate: recompute hashes and update timestamp.
    pub fn finalize(&mut self) {
        self.timestamp = chrono::Utc::now().to_rfc3339();
        self.metric_hash = *blake3::hash(self.metric_name.as_bytes()).as_bytes();
        if let Some(ref proof) = self.stark_proof {
            self.stark_proof_hash = *blake3::hash(proof).as_bytes();
            self.metadata.proof_size_bytes = proof.len();
        }
    }

    /// Verify internal consistency of the certificate.
    pub fn verify_integrity(&self) -> CertificateVerification {
        let mut verification = CertificateVerification::new();

        // Check version
        verification.add_check(VerificationCheck::new(
            "version",
            self.version > 0,
            &format!("Version is {}", self.version),
        ));

        // Check certificate_id is non-empty
        verification.add_check(VerificationCheck::new(
            "certificate_id",
            !self.certificate_id.is_empty(),
            "Certificate ID is present",
        ));

        // Check metric_hash matches metric_name
        let expected_hash = *blake3::hash(self.metric_name.as_bytes()).as_bytes();
        verification.add_check(VerificationCheck::new(
            "metric_hash",
            self.metric_hash == expected_hash,
            "Metric hash matches metric name",
        ));

        // Check STARK proof hash if proof is present
        if let Some(ref proof) = self.stark_proof {
            let expected_proof_hash = *blake3::hash(proof).as_bytes();
            verification.add_check(VerificationCheck::new(
                "stark_proof_hash",
                self.stark_proof_hash == expected_proof_hash,
                "STARK proof hash matches proof bytes",
            ));
        }

        // Check timestamp is parseable
        let ts_ok = chrono::DateTime::parse_from_rfc3339(&self.timestamp).is_ok();
        verification.add_check(VerificationCheck::new(
            "timestamp",
            ts_ok,
            "Timestamp is valid RFC 3339",
        ));

        // Check valid_until is parseable if present
        if let Some(ref valid_until) = self.valid_until {
            let vu_ok = chrono::DateTime::parse_from_rfc3339(valid_until).is_ok();
            verification.add_check(VerificationCheck::new(
                "valid_until",
                vu_ok,
                "valid_until is valid RFC 3339",
            ));
        }

        // Check score is finite
        verification.add_check(VerificationCheck::new(
            "score_finite",
            self.evaluation_result.score.is_finite(),
            "Score is finite",
        ));

        // Check commitment openings
        for opening in &self.commitment_openings {
            verification.add_check(VerificationCheck::new(
                &format!("commitment_{}", opening.name),
                opening.verify(),
                &format!("Commitment opening '{}' is valid", opening.name),
            ));
        }

        verification
    }

    /// Check if the certificate can be verified without external resources.
    pub fn verify_offline(&self) -> bool {
        let v = self.verify_integrity();
        v.is_valid
    }

    /// Check if the certificate has expired.
    pub fn is_expired(&self) -> bool {
        if let Some(ref valid_until) = self.valid_until {
            if let Ok(expiry) = chrono::DateTime::parse_from_rfc3339(valid_until) {
                return chrono::Utc::now() > expiry;
            }
        }
        false // no expiry set => not expired
    }

    /// Check if the certificate is valid: not expired and integrity passes.
    pub fn is_valid(&self) -> bool {
        !self.is_expired() && self.verify_offline()
    }

    /// Serialize the certificate without the STARK proof (compact form).
    pub fn serialize_compact(&self) -> Vec<u8> {
        let mut compact = self.clone();
        compact.stark_proof = None;
        serde_json::to_vec(&compact).unwrap_or_default()
    }

    /// Serialize the full certificate including the proof.
    pub fn serialize_full(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    /// Deserialize a certificate from bytes.
    pub fn deserialize(bytes: &[u8]) -> Result<Self, CertificateError> {
        serde_json::from_slice(bytes)
            .map_err(|e| CertificateError::DeserializationError(e.to_string()))
    }

    /// Serialize to a JSON string.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Deserialize from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, CertificateError> {
        serde_json::from_str(json)
            .map_err(|e| CertificateError::DeserializationError(e.to_string()))
    }

    /// Produce a human-readable summary of the certificate.
    pub fn human_readable_summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("Certificate: {}", self.certificate_id));
        lines.push(format!("  Metric:    {}", self.metric_name));
        lines.push(format!("  Model:     {}", self.model_id));
        lines.push(format!("  Benchmark: {}", self.benchmark_id));
        lines.push(format!("  Score:     {:.6}", self.evaluation_result.score));
        if let Some(conf) = self.evaluation_result.confidence {
            lines.push(format!("  Confidence:{:.6}", conf));
        }
        lines.push(format!("  Timestamp: {}", self.timestamp));
        if let Some(ref vu) = self.valid_until {
            lines.push(format!("  Valid until:{}", vu));
        }
        lines.push(format!("  Issuer:    {}", self.issuer));
        lines.push(format!(
            "  Proof:     {} bytes",
            self.stark_proof.as_ref().map_or(0, |p| p.len())
        ));
        lines.push(format!(
            "  PSI:       {}",
            if self.psi_attestation.is_some() {
                "present"
            } else {
                "absent"
            }
        ));
        lines.push(format!(
            "  Openings:  {}",
            self.commitment_openings.len()
        ));
        lines.join("\n")
    }

    /// Compute a unique hash of this certificate (excluding the signature).
    pub fn certificate_hash(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"cert-hash-v1");
        hasher.update(self.certificate_id.as_bytes());
        hasher.update(self.metric_name.as_bytes());
        hasher.update(&self.metric_hash);
        hasher.update(self.model_id.as_bytes());
        hasher.update(self.benchmark_id.as_bytes());
        hasher.update(&self.evaluation_result.score.to_le_bytes());
        hasher.update(&self.evaluation_result.score_field_element.to_le_bytes());
        hasher.update(&self.stark_proof_hash);
        hasher.update(self.timestamp.as_bytes());
        hasher.update(self.issuer.as_bytes());
        *hasher.finalize().as_bytes()
    }

    /// Chain this certificate with another to form a certificate chain.
    pub fn chain_with(&self, other: &EvaluationCertificate) -> CertificateChain {
        let mut chain = CertificateChain::new();
        chain.add(self.clone());
        chain.add(other.clone());
        chain
    }
}

// ─────────────────────────────────────────────────────────────
// CertificateChain
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CertificateChain {
    pub certificates: Vec<EvaluationCertificate>,
    pub chain_hash: [u8; 32],
}

impl CertificateChain {
    pub fn new() -> Self {
        Self {
            certificates: Vec::new(),
            chain_hash: [0u8; 32],
        }
    }

    /// Add a certificate and recompute the chain hash.
    pub fn add(&mut self, cert: EvaluationCertificate) {
        self.certificates.push(cert);
        self.recompute_hash();
    }

    fn recompute_hash(&mut self) {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"cert-chain-v1");
        for cert in &self.certificates {
            hasher.update(&cert.certificate_hash());
        }
        self.chain_hash = *hasher.finalize().as_bytes();
    }

    /// Verify that every certificate in the chain is valid and the hash is
    /// consistent.
    pub fn verify_chain(&self) -> bool {
        // All individual certificates must be valid
        for cert in &self.certificates {
            if !cert.is_valid() {
                return false;
            }
        }

        // Recompute chain hash and compare
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"cert-chain-v1");
        for cert in &self.certificates {
            hasher.update(&cert.certificate_hash());
        }
        let expected = *hasher.finalize().as_bytes();
        self.chain_hash == expected
    }

    /// Produce a combined result summarising the entire chain.
    pub fn combined_result(&self) -> Option<CertifiedResult> {
        if self.certificates.is_empty() {
            return None;
        }

        let total_score: f64 = self
            .certificates
            .iter()
            .map(|c| c.evaluation_result.score)
            .sum();
        let avg_score = total_score / self.certificates.len() as f64;

        let mut details = HashMap::new();
        details.insert("chain_length".to_string(), self.certificates.len().to_string());
        for (i, cert) in self.certificates.iter().enumerate() {
            details.insert(
                format!("cert_{}_metric", i),
                cert.metric_name.clone(),
            );
            details.insert(
                format!("cert_{}_score", i),
                format!("{:.6}", cert.evaluation_result.score),
            );
        }

        Some(CertifiedResult {
            score: avg_score,
            score_field_element: (avg_score * 1_000_000.0) as u64,
            confidence: None,
            details,
        })
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    pub fn len(&self) -> usize {
        self.certificates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.certificates.is_empty()
    }
}

impl Default for CertificateChain {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────
// CertificateStore
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CertificateStore {
    certificates: HashMap<String, EvaluationCertificate>,
}

impl CertificateStore {
    pub fn new() -> Self {
        Self {
            certificates: HashMap::new(),
        }
    }

    pub fn insert(&mut self, cert: EvaluationCertificate) {
        self.certificates.insert(cert.certificate_id.clone(), cert);
    }

    pub fn get(&self, id: &str) -> Option<&EvaluationCertificate> {
        self.certificates.get(id)
    }

    pub fn list(&self) -> Vec<&EvaluationCertificate> {
        self.certificates.values().collect()
    }

    pub fn remove(&mut self, id: &str) -> Option<EvaluationCertificate> {
        self.certificates.remove(id)
    }

    pub fn find_by_model(&self, model: &str) -> Vec<&EvaluationCertificate> {
        self.certificates
            .values()
            .filter(|c| c.model_id == model)
            .collect()
    }

    pub fn find_by_metric(&self, metric: &str) -> Vec<&EvaluationCertificate> {
        self.certificates
            .values()
            .filter(|c| c.metric_name == metric)
            .collect()
    }

    pub fn export_all(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    pub fn import(bytes: &[u8]) -> Result<Self, CertificateError> {
        serde_json::from_slice(bytes)
            .map_err(|e| CertificateError::DeserializationError(e.to_string()))
    }

    pub fn len(&self) -> usize {
        self.certificates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.certificates.is_empty()
    }
}

impl Default for CertificateStore {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────
// CertificateBuilder
// ─────────────────────────────────────────────────────────────

pub struct CertificateBuilder {
    metric_name: Option<String>,
    model_id: Option<String>,
    benchmark_id: Option<String>,
    score: Option<f64>,
    score_field: Option<u64>,
    stark_proof: Option<Vec<u8>>,
    psi_attestation: Option<PSIAttestation>,
    commitment_openings: Vec<CommitmentOpening>,
    metadata: Option<CertificateMetadata>,
    validity_hours: Option<u64>,
}

impl CertificateBuilder {
    pub fn new() -> Self {
        Self {
            metric_name: None,
            model_id: None,
            benchmark_id: None,
            score: None,
            score_field: None,
            stark_proof: None,
            psi_attestation: None,
            commitment_openings: Vec::new(),
            metadata: None,
            validity_hours: None,
        }
    }

    pub fn metric(mut self, name: &str) -> Self {
        self.metric_name = Some(name.to_string());
        self
    }

    pub fn model(mut self, id: &str) -> Self {
        self.model_id = Some(id.to_string());
        self
    }

    pub fn benchmark(mut self, id: &str) -> Self {
        self.benchmark_id = Some(id.to_string());
        self
    }

    pub fn score(mut self, value: f64) -> Self {
        self.score = Some(value);
        self
    }

    pub fn score_field_element(mut self, value: u64) -> Self {
        self.score_field = Some(value);
        self
    }

    pub fn with_proof(mut self, proof: Vec<u8>) -> Self {
        self.stark_proof = Some(proof);
        self
    }

    pub fn with_psi(mut self, attestation: PSIAttestation) -> Self {
        self.psi_attestation = Some(attestation);
        self
    }

    pub fn add_commitment(mut self, opening: CommitmentOpening) -> Self {
        self.commitment_openings.push(opening);
        self
    }

    pub fn with_metadata(mut self, metadata: CertificateMetadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    pub fn valid_for_hours(mut self, hours: u64) -> Self {
        self.validity_hours = Some(hours);
        self
    }

    pub fn build(self) -> Result<EvaluationCertificate, CertificateError> {
        let metric = self.metric_name
            .ok_or_else(|| CertificateError::MissingField("metric_name".to_string()))?;
        let model = self.model_id
            .ok_or_else(|| CertificateError::MissingField("model_id".to_string()))?;
        let benchmark = self.benchmark_id
            .ok_or_else(|| CertificateError::MissingField("benchmark_id".to_string()))?;
        let score = self.score
            .ok_or_else(|| CertificateError::MissingField("score".to_string()))?;

        let score_field = self.score_field.unwrap_or((score * 1_000_000.0) as u64);
        let result = CertifiedResult::new(score, score_field);

        let mut cert = EvaluationCertificate::new(&metric, &model, &benchmark, result);

        if let Some(proof) = self.stark_proof {
            cert.with_stark_proof(proof);
        }

        if let Some(psi) = self.psi_attestation {
            cert.with_psi_attestation(psi);
        }

        for opening in self.commitment_openings {
            cert.add_commitment_opening(opening);
        }

        if let Some(metadata) = self.metadata {
            cert.metadata = metadata;
        }

        if let Some(hours) = self.validity_hours {
            cert.set_validity(hours);
        }

        cert.finalize();
        Ok(cert)
    }
}

impl Default for CertificateBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────
// CertificateRegistry
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct CertificateRegistry {
    certificates: HashMap<String, EvaluationCertificate>,
    index_by_model: HashMap<String, Vec<String>>,
    index_by_metric: HashMap<String, Vec<String>>,
    index_by_benchmark: HashMap<String, Vec<String>>,
    revoked: std::collections::HashSet<String>,
}

impl CertificateRegistry {
    pub fn new() -> Self {
        Self {
            certificates: HashMap::new(),
            index_by_model: HashMap::new(),
            index_by_metric: HashMap::new(),
            index_by_benchmark: HashMap::new(),
            revoked: std::collections::HashSet::new(),
        }
    }

    pub fn register(&mut self, cert: EvaluationCertificate) -> Result<(), CertificateError> {
        let id = cert.certificate_id.clone();
        if self.certificates.contains_key(&id) {
            return Err(CertificateError::InvalidFormat);
        }
        self.index_by_model
            .entry(cert.model_id.clone())
            .or_default()
            .push(id.clone());
        self.index_by_metric
            .entry(cert.metric_name.clone())
            .or_default()
            .push(id.clone());
        self.index_by_benchmark
            .entry(cert.benchmark_id.clone())
            .or_default()
            .push(id.clone());
        self.certificates.insert(id, cert);
        Ok(())
    }

    pub fn get(&self, id: &str) -> Option<&EvaluationCertificate> {
        self.certificates.get(id)
    }

    pub fn find_by_model(&self, model: &str) -> Vec<&EvaluationCertificate> {
        self.index_by_model
            .get(model)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.certificates.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn find_by_metric(&self, metric: &str) -> Vec<&EvaluationCertificate> {
        self.index_by_metric
            .get(metric)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.certificates.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn find_by_benchmark(&self, benchmark: &str) -> Vec<&EvaluationCertificate> {
        self.index_by_benchmark
            .get(benchmark)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.certificates.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn find_valid(&self) -> Vec<&EvaluationCertificate> {
        self.certificates
            .values()
            .filter(|c| c.is_valid() && !self.revoked.contains(&c.certificate_id))
            .collect()
    }

    pub fn find_expired(&self) -> Vec<&EvaluationCertificate> {
        self.certificates
            .values()
            .filter(|c| c.is_expired())
            .collect()
    }

    pub fn revoke(&mut self, id: &str) -> Result<(), CertificateError> {
        if !self.certificates.contains_key(id) {
            return Err(CertificateError::MissingField(format!("certificate {}", id)));
        }
        self.revoked.insert(id.to_string());
        Ok(())
    }

    pub fn is_revoked(&self, id: &str) -> bool {
        self.revoked.contains(id)
    }

    pub fn count(&self) -> usize {
        self.certificates.len()
    }

    pub fn export_all(&self) -> Vec<u8> {
        let certs: Vec<&EvaluationCertificate> = self.certificates.values().collect();
        serde_json::to_vec(&certs).unwrap_or_default()
    }

    pub fn import(&mut self, bytes: &[u8]) -> Result<usize, CertificateError> {
        let certs: Vec<EvaluationCertificate> = serde_json::from_slice(bytes)
            .map_err(|e| CertificateError::DeserializationError(e.to_string()))?;
        let count = certs.len();
        for cert in certs {
            let _ = self.register(cert);
        }
        Ok(count)
    }

    pub fn prune_expired(&mut self) -> usize {
        let expired_ids: Vec<String> = self.certificates
            .iter()
            .filter(|(_, c)| c.is_expired())
            .map(|(id, _)| id.clone())
            .collect();
        let count = expired_ids.len();
        for id in &expired_ids {
            self.certificates.remove(id);
            for ids in self.index_by_model.values_mut() {
                ids.retain(|x| x != id);
            }
            for ids in self.index_by_metric.values_mut() {
                ids.retain(|x| x != id);
            }
            for ids in self.index_by_benchmark.values_mut() {
                ids.retain(|x| x != id);
            }
        }
        count
    }
}

impl Default for CertificateRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────
// ComparisonSummary
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComparisonSummary {
    pub num_certificates: usize,
    pub metrics: Vec<String>,
    pub score_range: (f64, f64),
    pub avg_score: f64,
    pub best_model: String,
    pub worst_model: String,
    pub avg_contamination: f64,
}

// ─────────────────────────────────────────────────────────────
// CertificateComparator
// ─────────────────────────────────────────────────────────────

pub struct CertificateComparator;

impl CertificateComparator {
    pub fn compare_scores(
        a: &EvaluationCertificate,
        b: &EvaluationCertificate,
    ) -> std::cmp::Ordering {
        a.evaluation_result
            .score
            .partial_cmp(&b.evaluation_result.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }

    pub fn compare_contamination(
        a: &EvaluationCertificate,
        b: &EvaluationCertificate,
    ) -> Option<std::cmp::Ordering> {
        let a_cont = a.psi_attestation.as_ref().map(|p| p.contamination_bound);
        let b_cont = b.psi_attestation.as_ref().map(|p| p.contamination_bound);
        match (a_cont, b_cont) {
            (Some(ac), Some(bc)) => ac.partial_cmp(&bc),
            _ => None,
        }
    }

    pub fn are_compatible(
        a: &EvaluationCertificate,
        b: &EvaluationCertificate,
    ) -> bool {
        a.metric_name == b.metric_name && a.benchmark_id == b.benchmark_id
    }

    pub fn merge_results(certs: &[&EvaluationCertificate]) -> CertifiedResult {
        if certs.is_empty() {
            return CertifiedResult::new(0.0, 0);
        }
        let total: f64 = certs.iter().map(|c| c.evaluation_result.score).sum();
        let avg = total / certs.len() as f64;
        let field = (avg * 1_000_000.0) as u64;
        let mut result = CertifiedResult::new(avg, field);
        result.details.insert("merged_count".to_string(), certs.len().to_string());
        result
    }

    pub fn score_delta(
        a: &EvaluationCertificate,
        b: &EvaluationCertificate,
    ) -> f64 {
        (a.evaluation_result.score - b.evaluation_result.score).abs()
    }

    pub fn summary_comparison(certs: &[&EvaluationCertificate]) -> ComparisonSummary {
        if certs.is_empty() {
            return ComparisonSummary {
                num_certificates: 0,
                metrics: Vec::new(),
                score_range: (0.0, 0.0),
                avg_score: 0.0,
                best_model: String::new(),
                worst_model: String::new(),
                avg_contamination: 0.0,
            };
        }

        let scores: Vec<f64> = certs.iter().map(|c| c.evaluation_result.score).collect();
        let min_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let avg_score = scores.iter().sum::<f64>() / scores.len() as f64;

        let mut metrics: Vec<String> = certs.iter().map(|c| c.metric_name.clone()).collect();
        metrics.sort();
        metrics.dedup();

        let best_idx = scores.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let worst_idx = scores.iter().enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let contaminations: Vec<f64> = certs
            .iter()
            .filter_map(|c| c.psi_attestation.as_ref().map(|p| p.contamination_bound))
            .collect();
        let avg_contamination = if contaminations.is_empty() {
            0.0
        } else {
            contaminations.iter().sum::<f64>() / contaminations.len() as f64
        };

        ComparisonSummary {
            num_certificates: certs.len(),
            metrics,
            score_range: (min_score, max_score),
            avg_score,
            best_model: certs[best_idx].model_id.clone(),
            worst_model: certs[worst_idx].model_id.clone(),
            avg_contamination,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// CertificateReport
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct CertificateReport {
    certificate_id: String,
    metric_name: String,
    model_id: String,
    benchmark_id: String,
    score: f64,
    confidence: Option<f64>,
    timestamp: String,
    valid_until: Option<String>,
    issuer: String,
    proof_size: usize,
    has_psi: bool,
    contamination_bound: Option<f64>,
    num_openings: usize,
    is_valid: bool,
    chain_length: Option<usize>,
}

impl CertificateReport {
    pub fn from_certificate(cert: &EvaluationCertificate) -> Self {
        Self {
            certificate_id: cert.certificate_id.clone(),
            metric_name: cert.metric_name.clone(),
            model_id: cert.model_id.clone(),
            benchmark_id: cert.benchmark_id.clone(),
            score: cert.evaluation_result.score,
            confidence: cert.evaluation_result.confidence,
            timestamp: cert.timestamp.clone(),
            valid_until: cert.valid_until.clone(),
            issuer: cert.issuer.clone(),
            proof_size: cert.stark_proof.as_ref().map_or(0, |p| p.len()),
            has_psi: cert.psi_attestation.is_some(),
            contamination_bound: cert.psi_attestation.as_ref().map(|p| p.contamination_bound),
            num_openings: cert.commitment_openings.len(),
            is_valid: cert.is_valid(),
            chain_length: None,
        }
    }

    pub fn from_chain(chain: &CertificateChain) -> Self {
        let combined = chain.combined_result();
        let score = combined.as_ref().map_or(0.0, |r| r.score);
        let first = chain.certificates.first();
        Self {
            certificate_id: first.map_or(String::new(), |c| c.certificate_id.clone()),
            metric_name: first.map_or(String::new(), |c| c.metric_name.clone()),
            model_id: first.map_or(String::new(), |c| c.model_id.clone()),
            benchmark_id: first.map_or(String::new(), |c| c.benchmark_id.clone()),
            score,
            confidence: None,
            timestamp: first.map_or(String::new(), |c| c.timestamp.clone()),
            valid_until: first.and_then(|c| c.valid_until.clone()),
            issuer: first.map_or(String::new(), |c| c.issuer.clone()),
            proof_size: chain.certificates.iter()
                .map(|c| c.stark_proof.as_ref().map_or(0, |p| p.len())).sum(),
            has_psi: chain.certificates.iter().any(|c| c.psi_attestation.is_some()),
            contamination_bound: None,
            num_openings: chain.certificates.iter()
                .map(|c| c.commitment_openings.len()).sum(),
            is_valid: chain.verify_chain(),
            chain_length: Some(chain.len()),
        }
    }

    pub fn to_text(&self) -> String {
        let mut lines = Vec::new();
        lines.push("=== Certificate Report ===".to_string());
        lines.push(format!("ID:            {}", self.certificate_id));
        lines.push(format!("Metric:        {}", self.metric_name));
        lines.push(format!("Model:         {}", self.model_id));
        lines.push(format!("Benchmark:     {}", self.benchmark_id));
        lines.push(format!("Score:         {:.6}", self.score));
        if let Some(conf) = self.confidence {
            lines.push(format!("Confidence:    {:.6}", conf));
        }
        lines.push(format!("Timestamp:     {}", self.timestamp));
        if let Some(ref vu) = self.valid_until {
            lines.push(format!("Valid Until:    {}", vu));
        }
        lines.push(format!("Issuer:        {}", self.issuer));
        lines.push(format!("Proof Size:    {} bytes", self.proof_size));
        lines.push(format!("PSI Present:   {}", self.has_psi));
        if let Some(cb) = self.contamination_bound {
            lines.push(format!("Contamination: {:.6}", cb));
        }
        lines.push(format!("Openings:      {}", self.num_openings));
        lines.push(format!("Valid:         {}", self.is_valid));
        if let Some(cl) = self.chain_length {
            lines.push(format!("Chain Length:   {}", cl));
        }
        lines.push("==========================".to_string());
        lines.join("\n")
    }

    pub fn to_html(&self) -> String {
        let mut html = String::from("<div class=\"certificate-report\">\n");
        html.push_str("  <h2>Certificate Report</h2>\n");
        html.push_str("  <table>\n");
        html.push_str(&format!("    <tr><td>ID</td><td>{}</td></tr>\n", self.certificate_id));
        html.push_str(&format!("    <tr><td>Metric</td><td>{}</td></tr>\n", self.metric_name));
        html.push_str(&format!("    <tr><td>Model</td><td>{}</td></tr>\n", self.model_id));
        html.push_str(&format!("    <tr><td>Benchmark</td><td>{}</td></tr>\n", self.benchmark_id));
        html.push_str(&format!("    <tr><td>Score</td><td>{:.6}</td></tr>\n", self.score));
        html.push_str(&format!("    <tr><td>Valid</td><td>{}</td></tr>\n", self.is_valid));
        html.push_str("  </table>\n");
        html.push_str("</div>");
        html
    }

    pub fn to_json(&self) -> String {
        let mut map = serde_json::Map::new();
        map.insert("certificate_id".to_string(), serde_json::Value::String(self.certificate_id.clone()));
        map.insert("metric_name".to_string(), serde_json::Value::String(self.metric_name.clone()));
        map.insert("model_id".to_string(), serde_json::Value::String(self.model_id.clone()));
        map.insert("benchmark_id".to_string(), serde_json::Value::String(self.benchmark_id.clone()));
        map.insert("score".to_string(), serde_json::json!(self.score));
        map.insert("is_valid".to_string(), serde_json::Value::Bool(self.is_valid));
        map.insert("proof_size".to_string(), serde_json::json!(self.proof_size));
        map.insert("has_psi".to_string(), serde_json::Value::Bool(self.has_psi));
        map.insert("num_openings".to_string(), serde_json::json!(self.num_openings));
        serde_json::to_string_pretty(&serde_json::Value::Object(map)).unwrap_or_default()
    }

    pub fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{},{:.6},{},{},{}",
            self.certificate_id,
            self.metric_name,
            self.model_id,
            self.benchmark_id,
            self.score,
            self.proof_size,
            self.has_psi,
            self.is_valid,
        )
    }
}

// ─────────────────────────────────────────────────────────────
// ValidationRule / ValidationResult / CertificateValidator
// ─────────────────────────────────────────────────────────────

pub struct ValidationRule {
    pub name: String,
    pub description: String,
    pub check: fn(&EvaluationCertificate) -> bool,
}

#[derive(Clone, Debug)]
pub struct ValidationResult {
    pub rule_name: String,
    pub passed: bool,
    pub details: String,
}

pub struct CertificateValidator {
    validation_rules: Vec<ValidationRule>,
}

impl CertificateValidator {
    pub fn new() -> Self {
        Self {
            validation_rules: Vec::new(),
        }
    }

    pub fn add_rule(&mut self, rule: ValidationRule) -> &mut Self {
        self.validation_rules.push(rule);
        self
    }

    pub fn with_standard_rules() -> Self {
        let mut v = Self::new();
        v.add_rule(ValidationRule {
            name: "score_finite".to_string(),
            description: "Score must be finite".to_string(),
            check: |cert| cert.evaluation_result.score.is_finite(),
        });
        v.add_rule(ValidationRule {
            name: "score_range".to_string(),
            description: "Score must be between 0 and 1".to_string(),
            check: |cert| {
                let s = cert.evaluation_result.score;
                s >= 0.0 && s <= 1.0
            },
        });
        v.add_rule(ValidationRule {
            name: "has_metric".to_string(),
            description: "Metric name must not be empty".to_string(),
            check: |cert| !cert.metric_name.is_empty(),
        });
        v.add_rule(ValidationRule {
            name: "has_model".to_string(),
            description: "Model ID must not be empty".to_string(),
            check: |cert| !cert.model_id.is_empty(),
        });
        v.add_rule(ValidationRule {
            name: "has_benchmark".to_string(),
            description: "Benchmark ID must not be empty".to_string(),
            check: |cert| !cert.benchmark_id.is_empty(),
        });
        v.add_rule(ValidationRule {
            name: "not_expired".to_string(),
            description: "Certificate must not be expired".to_string(),
            check: |cert| !cert.is_expired(),
        });
        v.add_rule(ValidationRule {
            name: "integrity".to_string(),
            description: "Certificate integrity check must pass".to_string(),
            check: |cert| cert.verify_offline(),
        });
        v
    }

    pub fn validate(&self, cert: &EvaluationCertificate) -> Vec<ValidationResult> {
        self.validation_rules
            .iter()
            .map(|rule| {
                let passed = (rule.check)(cert);
                ValidationResult {
                    rule_name: rule.name.clone(),
                    passed,
                    details: if passed {
                        format!("{}: OK", rule.description)
                    } else {
                        format!("{}: FAILED", rule.description)
                    },
                }
            })
            .collect()
    }

    pub fn validate_chain(&self, chain: &CertificateChain) -> Vec<ValidationResult> {
        let mut results = Vec::new();
        for (i, cert) in chain.certificates.iter().enumerate() {
            for result in self.validate(cert) {
                results.push(ValidationResult {
                    rule_name: format!("cert[{}].{}", i, result.rule_name),
                    passed: result.passed,
                    details: result.details,
                });
            }
        }
        results
    }

    pub fn is_valid(&self, cert: &EvaluationCertificate) -> bool {
        self.validate(cert).iter().all(|r| r.passed)
    }
}

impl Default for CertificateValidator {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────
// ScoreDistribution / ContaminationSummary / CertificateAggregator
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScoreDistribution {
    pub mean: f64,
    pub median: f64,
    pub stddev: f64,
    pub min: f64,
    pub max: f64,
    pub percentiles: HashMap<u32, f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContaminationSummary {
    pub avg_contamination: f64,
    pub max_contamination: f64,
    pub num_clean: usize,
    pub num_contaminated: usize,
    pub threshold: f64,
}

pub struct CertificateAggregator {
    certificates: Vec<EvaluationCertificate>,
}

impl CertificateAggregator {
    pub fn new() -> Self {
        Self {
            certificates: Vec::new(),
        }
    }

    pub fn add(&mut self, cert: EvaluationCertificate) {
        self.certificates.push(cert);
    }

    pub fn aggregate_score(&self) -> f64 {
        if self.certificates.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.certificates.iter()
            .map(|c| c.evaluation_result.score).sum();
        sum / self.certificates.len() as f64
    }

    pub fn score_distribution(&self) -> ScoreDistribution {
        let mut scores: Vec<f64> = self.certificates.iter()
            .map(|c| c.evaluation_result.score).collect();
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = scores.len();
        if n == 0 {
            return ScoreDistribution {
                mean: 0.0,
                median: 0.0,
                stddev: 0.0,
                min: 0.0,
                max: 0.0,
                percentiles: HashMap::new(),
            };
        }

        let mean = scores.iter().sum::<f64>() / n as f64;
        let median = if n % 2 == 0 {
            (scores[n / 2 - 1] + scores[n / 2]) / 2.0
        } else {
            scores[n / 2]
        };
        let variance = scores.iter()
            .map(|s| (s - mean).powi(2)).sum::<f64>() / n as f64;
        let stddev = variance.sqrt();
        let min = scores[0];
        let max = scores[n - 1];

        let mut percentiles = HashMap::new();
        for &p in &[10u32, 25, 50, 75, 90, 95, 99] {
            let idx = ((p as f64 / 100.0) * (n as f64 - 1.0)).round() as usize;
            let idx = idx.min(n - 1);
            percentiles.insert(p, scores[idx]);
        }

        ScoreDistribution { mean, median, stddev, min, max, percentiles }
    }

    pub fn contamination_summary(&self) -> ContaminationSummary {
        let threshold = 0.05;
        let contaminations: Vec<f64> = self.certificates
            .iter()
            .filter_map(|c| c.psi_attestation.as_ref().map(|p| p.contamination_bound))
            .collect();

        if contaminations.is_empty() {
            return ContaminationSummary {
                avg_contamination: 0.0,
                max_contamination: 0.0,
                num_clean: self.certificates.len(),
                num_contaminated: 0,
                threshold,
            };
        }

        let avg = contaminations.iter().sum::<f64>() / contaminations.len() as f64;
        let max_c = contaminations.iter().cloned().fold(0.0_f64, f64::max);
        let num_contaminated = contaminations.iter().filter(|&&c| c > threshold).count();
        let num_clean = contaminations.len() - num_contaminated;

        ContaminationSummary {
            avg_contamination: avg,
            max_contamination: max_c,
            num_clean,
            num_contaminated,
            threshold,
        }
    }

    pub fn model_leaderboard(&self) -> Vec<(String, f64)> {
        let mut model_scores: HashMap<String, Vec<f64>> = HashMap::new();
        for cert in &self.certificates {
            model_scores
                .entry(cert.model_id.clone())
                .or_default()
                .push(cert.evaluation_result.score);
        }
        let mut leaderboard: Vec<(String, f64)> = model_scores
            .into_iter()
            .map(|(model, scores)| {
                let avg = scores.iter().sum::<f64>() / scores.len() as f64;
                (model, avg)
            })
            .collect();
        leaderboard.sort_by(|(_, a), (_, b)| {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        });
        leaderboard
    }

    pub fn metric_summary(&self) -> HashMap<String, f64> {
        let mut metric_scores: HashMap<String, Vec<f64>> = HashMap::new();
        for cert in &self.certificates {
            metric_scores
                .entry(cert.metric_name.clone())
                .or_default()
                .push(cert.evaluation_result.score);
        }
        metric_scores
            .into_iter()
            .map(|(metric, scores)| {
                let avg = scores.iter().sum::<f64>() / scores.len() as f64;
                (metric, avg)
            })
            .collect()
    }
}

impl Default for CertificateAggregator {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────
// CertificateSerializer
// ─────────────────────────────────────────────────────────────

fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::new();
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
    let input = input.trim_end_matches('=');
    let mut result = Vec::new();
    let chars: Vec<u8> = input.bytes().collect();

    for chunk in chars.chunks(4) {
        let vals: Vec<u32> = chunk.iter().map(|&c| {
            match c {
                b'A'..=b'Z' => (c - b'A') as u32,
                b'a'..=b'z' => (c - b'a' + 26) as u32,
                b'0'..=b'9' => (c - b'0' + 52) as u32,
                b'+' => 62,
                b'/' => 63,
                _ => 0,
            }
        }).collect();

        if vals.len() >= 2 {
            let triple = (vals[0] << 18)
                | (vals[1] << 12)
                | (if vals.len() > 2 { vals[2] } else { 0 } << 6)
                | (if vals.len() > 3 { vals[3] } else { 0 });
            result.push(((triple >> 16) & 0xFF) as u8);
            if vals.len() > 2 {
                result.push(((triple >> 8) & 0xFF) as u8);
            }
            if vals.len() > 3 {
                result.push((triple & 0xFF) as u8);
            }
        }
    }
    Ok(result)
}

pub struct CertificateSerializer;

impl CertificateSerializer {
    pub fn to_binary(cert: &EvaluationCertificate) -> Vec<u8> {
        serde_json::to_vec(cert).unwrap_or_default()
    }

    pub fn from_binary(bytes: &[u8]) -> Result<EvaluationCertificate, CertificateError> {
        serde_json::from_slice(bytes)
            .map_err(|e| CertificateError::DeserializationError(e.to_string()))
    }

    pub fn to_cbor_like(cert: &EvaluationCertificate) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"CERT");
        buf.extend_from_slice(&cert.version.to_le_bytes());
        let id_bytes = cert.certificate_id.as_bytes();
        buf.extend_from_slice(&(id_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(id_bytes);
        let metric_bytes = cert.metric_name.as_bytes();
        buf.extend_from_slice(&(metric_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(metric_bytes);
        let model_bytes = cert.model_id.as_bytes();
        buf.extend_from_slice(&(model_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(model_bytes);
        let bench_bytes = cert.benchmark_id.as_bytes();
        buf.extend_from_slice(&(bench_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(bench_bytes);
        buf.extend_from_slice(&cert.evaluation_result.score.to_le_bytes());
        buf.extend_from_slice(&cert.evaluation_result.score_field_element.to_le_bytes());
        buf.extend_from_slice(&cert.metric_hash);
        buf.extend_from_slice(&cert.stark_proof_hash);
        buf
    }

    pub fn to_base64(cert: &EvaluationCertificate) -> String {
        let bytes = Self::to_binary(cert);
        base64_encode(&bytes)
    }

    pub fn from_base64(s: &str) -> Result<EvaluationCertificate, CertificateError> {
        let bytes = base64_decode(s)
            .map_err(|e| CertificateError::DeserializationError(e))?;
        Self::from_binary(&bytes)
    }

    pub fn estimate_size(cert: &EvaluationCertificate) -> usize {
        let mut size = 0usize;
        size += 4; // version
        size += cert.certificate_id.len();
        size += cert.metric_name.len();
        size += 32; // metric_hash
        size += cert.model_id.len();
        size += cert.benchmark_id.len();
        size += 8; // score (f64)
        size += 8; // score_field_element (u64)
        size += 32; // stark_proof_hash
        if let Some(ref proof) = cert.stark_proof {
            size += proof.len();
        }
        if cert.psi_attestation.is_some() {
            size += 8 + 8 + 8 + 32 + 1; // f64 + usize + u64 + hash + bool
        }
        for opening in &cert.commitment_openings {
            size += opening.name.len() + 32 + opening.revealed_value.len()
                + opening.randomness.len() + opening.scheme.len();
        }
        size += cert.timestamp.len();
        if let Some(ref vu) = cert.valid_until {
            size += vu.len();
        }
        size += cert.issuer.len();
        if let Some(ref sig) = cert.signature {
            size += sig.len();
        }
        size += 8 + 8 + 8 + 8 + 8 + 4 + 8; // metadata fields
        size
    }
}

// ─────────────────────────────────────────────────────────────
// CertificateExporter
// ─────────────────────────────────────────────────────────────

pub struct CertificateExporter;

impl CertificateExporter {
    /// Encode the certificate as PEM: base64-wrapped JSON between header/footer.
    pub fn to_pem(cert: &EvaluationCertificate) -> String {
        let json = cert.to_json();
        let b64 = base64_encode(json.as_bytes());
        let mut pem = String::from("-----BEGIN SPECTACLES CERTIFICATE-----\n");
        for line in b64.as_bytes().chunks(64) {
            pem.push_str(std::str::from_utf8(line).unwrap_or(""));
            pem.push('\n');
        }
        pem.push_str("-----END SPECTACLES CERTIFICATE-----\n");
        pem
    }

    /// Decode a PEM-encoded certificate back to an EvaluationCertificate.
    pub fn from_pem(pem: &str) -> Result<EvaluationCertificate, CertificateError> {
        const BEGIN: &str = "-----BEGIN SPECTACLES CERTIFICATE-----";
        const END: &str = "-----END SPECTACLES CERTIFICATE-----";

        let start = pem
            .find(BEGIN)
            .ok_or(CertificateError::InvalidFormat)?;
        let end = pem
            .find(END)
            .ok_or(CertificateError::InvalidFormat)?;

        let b64_section = &pem[start + BEGIN.len()..end];
        let b64_clean: String = b64_section
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();

        let json_bytes = base64_decode(&b64_clean)
            .map_err(|e| CertificateError::DeserializationError(e))?;
        let json_str = std::str::from_utf8(&json_bytes)
            .map_err(|e| CertificateError::DeserializationError(e.to_string()))?;

        EvaluationCertificate::from_json(json_str)
    }

    /// Simple DER-like binary format: magic + version + length + JSON + blake3 hash.
    pub fn to_der(cert: &EvaluationCertificate) -> Vec<u8> {
        let json_bytes = cert.to_json().into_bytes();
        let json_hash = blake3::hash(&json_bytes);
        let json_len = json_bytes.len() as u32;

        let mut out = Vec::with_capacity(4 + 4 + 4 + json_bytes.len() + 32);
        out.extend_from_slice(b"SPEC");
        out.extend_from_slice(&cert.version.to_le_bytes());
        out.extend_from_slice(&json_len.to_le_bytes());
        out.extend_from_slice(&json_bytes);
        out.extend_from_slice(json_hash.as_bytes());
        out
    }

    /// Produce a JWK-like JSON object describing the certificate.
    pub fn to_jwk(cert: &EvaluationCertificate) -> String {
        let hash = cert.certificate_hash();
        let hex_hash: String = hash.iter().map(|b| format!("{:02x}", b)).collect();

        format!(
            "{{\"kty\":\"SPECTACLES\",\"kid\":\"{}\",\"metric\":\"{}\",\"score\":{},\"hash\":\"{}\",\"timestamp\":\"{}\"}}",
            cert.certificate_id,
            cert.metric_name,
            cert.evaluation_result.score,
            hex_hash,
            cert.timestamp,
        )
    }

    /// Decode a DER-encoded certificate, verifying magic bytes and blake3 hash.
    pub fn from_der(data: &[u8]) -> Result<EvaluationCertificate, CertificateError> {
        if data.len() < 12 {
            return Err(CertificateError::InvalidFormat);
        }
        if &data[0..4] != b"SPEC" {
            return Err(CertificateError::InvalidFormat);
        }
        let json_len =
            u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        let expected_total = 4 + 4 + 4 + json_len + 32;
        if data.len() < expected_total {
            return Err(CertificateError::InvalidFormat);
        }
        let json_bytes = &data[12..12 + json_len];
        let stored_hash = &data[12 + json_len..12 + json_len + 32];
        let computed_hash = blake3::hash(json_bytes);
        if stored_hash != computed_hash.as_bytes() {
            return Err(CertificateError::IntegrityCheckFailed);
        }
        let json_str = std::str::from_utf8(json_bytes)
            .map_err(|e| CertificateError::DeserializationError(e.to_string()))?;
        EvaluationCertificate::from_json(json_str)
    }

    /// Export certificate as a CSV row (header + data).
    pub fn to_csv_row(cert: &EvaluationCertificate) -> String {
        let hash = cert.certificate_hash();
        let hex_hash: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
        let header = "certificate_id,metric_name,model_id,benchmark_id,score,timestamp,issuer,hash";
        let row = format!(
            "{},{},{},{},{},{},{},{}",
            cert.certificate_id,
            cert.metric_name,
            cert.model_id,
            cert.benchmark_id,
            cert.evaluation_result.score,
            cert.timestamp,
            cert.issuer,
            hex_hash,
        );
        format!("{}\n{}", header, row)
    }
}

// ─────────────────────────────────────────────────────────────
// CertificateAuditLog
// ─────────────────────────────────────────────────────────────

/// Severity level for audit log entries.
#[derive(Clone, Debug, PartialEq)]
pub enum AuditSeverity {
    Info,
    Warning,
    Error,
}

impl fmt::Display for AuditSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AuditSeverity::Info => write!(f, "INFO"),
            AuditSeverity::Warning => write!(f, "WARNING"),
            AuditSeverity::Error => write!(f, "ERROR"),
        }
    }
}

/// A single audit log entry.
#[derive(Clone, Debug)]
pub struct AuditEntry {
    pub timestamp: u64,
    pub cert_id: String,
    pub action: String,
    pub severity: AuditSeverity,
    pub details: String,
}

/// Append-only audit log for certificate operations.
pub struct CertificateAuditLog {
    entries: Vec<AuditEntry>,
}

impl CertificateAuditLog {
    /// Create a new empty audit log.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Append an audit entry.
    pub fn log(
        &mut self,
        timestamp: u64,
        cert_id: &str,
        action: &str,
        severity: AuditSeverity,
        details: &str,
    ) {
        self.entries.push(AuditEntry {
            timestamp,
            cert_id: cert_id.to_string(),
            action: action.to_string(),
            severity,
            details: details.to_string(),
        });
    }

    /// Return the total number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return true if the log is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Return all entries for a given certificate id, in insertion order.
    pub fn entries_for(&self, cert_id: &str) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.cert_id == cert_id)
            .collect()
    }

    /// Return all entries with the given severity level.
    pub fn entries_by_severity(&self, severity: &AuditSeverity) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| &e.severity == severity)
            .collect()
    }

    /// Return entries within a timestamp range [from, to] inclusive.
    pub fn entries_in_range(&self, from: u64, to: u64) -> Vec<&AuditEntry> {
        self.entries
            .iter()
            .filter(|e| e.timestamp >= from && e.timestamp <= to)
            .collect()
    }

    /// Produce a human-readable summary of the log.
    pub fn summary(&self) -> String {
        let info_count = self
            .entries
            .iter()
            .filter(|e| e.severity == AuditSeverity::Info)
            .count();
        let warn_count = self
            .entries
            .iter()
            .filter(|e| e.severity == AuditSeverity::Warning)
            .count();
        let err_count = self
            .entries
            .iter()
            .filter(|e| e.severity == AuditSeverity::Error)
            .count();
        let unique_certs: std::collections::HashSet<&str> = self
            .entries
            .iter()
            .map(|e| e.cert_id.as_str())
            .collect();
        format!(
            "AuditLog: {} entries ({} info, {} warning, {} error) across {} certificates",
            self.entries.len(),
            info_count,
            warn_count,
            err_count,
            unique_certs.len(),
        )
    }

    /// Compute a blake3 digest of the entire log for tamper detection.
    pub fn digest(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"audit-log-v1");
        for entry in &self.entries {
            hasher.update(&entry.timestamp.to_le_bytes());
            hasher.update(entry.cert_id.as_bytes());
            hasher.update(entry.action.as_bytes());
            hasher.update(entry.severity.to_string().as_bytes());
            hasher.update(entry.details.as_bytes());
        }
        *hasher.finalize().as_bytes()
    }
}

// ─────────────────────────────────────────────────────────────
// CertificateTimeline
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub enum TimelineEventType {
    Created,
    Verified(bool),
    Revoked,
}

impl fmt::Display for TimelineEventType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TimelineEventType::Created => write!(f, "Created"),
            TimelineEventType::Verified(ok) => write!(f, "Verified({})", ok),
            TimelineEventType::Revoked => write!(f, "Revoked"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct TimelineEvent {
    pub cert_id: String,
    pub timestamp: u64,
    pub event_type: TimelineEventType,
}

pub struct CertificateTimeline {
    events: Vec<TimelineEvent>,
}

impl CertificateTimeline {
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    pub fn record_creation(&mut self, cert_id: &str, timestamp: u64) {
        self.events.push(TimelineEvent {
            cert_id: cert_id.to_string(),
            timestamp,
            event_type: TimelineEventType::Created,
        });
    }

    pub fn record_verification(&mut self, cert_id: &str, timestamp: u64, result: bool) {
        self.events.push(TimelineEvent {
            cert_id: cert_id.to_string(),
            timestamp,
            event_type: TimelineEventType::Verified(result),
        });
    }

    pub fn record_revocation(&mut self, cert_id: &str, timestamp: u64) {
        self.events.push(TimelineEvent {
            cert_id: cert_id.to_string(),
            timestamp,
            event_type: TimelineEventType::Revoked,
        });
    }

    /// Returns (timestamp, description) pairs for the given cert_id, sorted by timestamp.
    pub fn timeline_for(&self, cert_id: &str) -> Vec<(u64, String)> {
        let mut matching: Vec<(u64, String)> = self
            .events
            .iter()
            .filter(|e| e.cert_id == cert_id)
            .map(|e| (e.timestamp, e.event_type.to_string()))
            .collect();
        matching.sort_by_key(|(ts, _)| *ts);
        matching
    }

    /// Returns unique cert_ids that were created at or before `timestamp`
    /// and NOT revoked at or before `timestamp`.
    pub fn active_at(&self, timestamp: u64) -> Vec<String> {
        let mut created: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        let mut revoked: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        for event in &self.events {
            if event.timestamp <= timestamp {
                match &event.event_type {
                    TimelineEventType::Created => {
                        created.insert(event.cert_id.clone());
                    }
                    TimelineEventType::Revoked => {
                        revoked.insert(event.cert_id.clone());
                    }
                    _ => {}
                }
            }
        }

        let mut active: Vec<String> = created
            .difference(&revoked)
            .cloned()
            .collect();
        active.sort();
        active
    }
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cert() -> EvaluationCertificate {
        let result = CertifiedResult::new(0.95, 950000);
        EvaluationCertificate::new("accuracy", "gpt-4", "mmlu", result)
    }

    // ── Basic creation ──

    #[test]
    fn test_create_certificate() {
        let cert = make_cert();
        assert_eq!(cert.version, 1);
        assert_eq!(cert.metric_name, "accuracy");
        assert_eq!(cert.model_id, "gpt-4");
        assert_eq!(cert.benchmark_id, "mmlu");
        assert!(!cert.certificate_id.is_empty());
        assert!(!cert.timestamp.is_empty());
    }

    #[test]
    fn test_unique_ids() {
        let c1 = make_cert();
        let c2 = make_cert();
        assert_ne!(c1.certificate_id, c2.certificate_id);
    }

    // ── Builder ──

    #[test]
    fn test_builder_basic() {
        let cert = CertificateBuilder::new()
            .metric("bleu")
            .model("model-x")
            .benchmark("wmt-22")
            .score(0.42)
            .build()
            .unwrap();
        assert_eq!(cert.metric_name, "bleu");
        assert_eq!(cert.evaluation_result.score, 0.42);
    }

    #[test]
    fn test_builder_missing_score() {
        let result = CertificateBuilder::new()
            .metric("bleu")
            .model("model-x")
            .benchmark("wmt-22")
            .build();
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            CertificateError::MissingField("score".to_string())
        );
    }

    #[test]
    fn test_builder_with_proof() {
        let proof = vec![1, 2, 3, 4, 5];
        let cert = CertificateBuilder::new()
            .metric("acc")
            .model("m")
            .benchmark("b")
            .score(0.9)
            .with_proof(proof.clone())
            .build()
            .unwrap();
        assert!(cert.stark_proof.is_some());
        assert_eq!(cert.stark_proof.unwrap(), proof);
    }

    #[test]
    fn test_builder_with_psi() {
        let psi = PSIAttestation::new(0.01, 5, 100, [0u8; 32], true);
        let cert = CertificateBuilder::new()
            .metric("acc")
            .model("m")
            .benchmark("b")
            .score(0.9)
            .with_psi(psi)
            .build()
            .unwrap();
        assert!(cert.psi_attestation.is_some());
    }

    #[test]
    fn test_builder_with_validity() {
        let cert = CertificateBuilder::new()
            .metric("acc")
            .model("m")
            .benchmark("b")
            .score(0.9)
            .valid_for_hours(24)
            .build()
            .unwrap();
        assert!(cert.valid_until.is_some());
        assert!(!cert.is_expired());
    }

    // ── Integrity verification ──

    #[test]
    fn test_verify_integrity_valid() {
        let mut cert = make_cert();
        cert.finalize();
        let v = cert.verify_integrity();
        assert!(v.is_valid);
    }

    #[test]
    fn test_verify_integrity_bad_metric_hash() {
        let mut cert = make_cert();
        cert.finalize();
        cert.metric_hash = [0xFFu8; 32]; // corrupt
        let v = cert.verify_integrity();
        assert!(!v.is_valid);
    }

    #[test]
    fn test_verify_integrity_bad_proof_hash() {
        let mut cert = make_cert();
        cert.with_stark_proof(vec![1, 2, 3]);
        cert.finalize();
        cert.stark_proof_hash = [0xFFu8; 32]; // corrupt
        let v = cert.verify_integrity();
        assert!(!v.is_valid);
    }

    // ── Expiry ──

    #[test]
    fn test_not_expired_by_default() {
        let cert = make_cert();
        assert!(!cert.is_expired());
    }

    #[test]
    fn test_expired_certificate() {
        let mut cert = make_cert();
        // Set valid_until to the past
        let past = chrono::Utc::now() - chrono::Duration::hours(1);
        cert.valid_until = Some(past.to_rfc3339());
        assert!(cert.is_expired());
    }

    #[test]
    fn test_is_valid_combines_expiry_and_integrity() {
        let mut cert = make_cert();
        cert.finalize();
        assert!(cert.is_valid());

        // Expire it
        let past = chrono::Utc::now() - chrono::Duration::hours(1);
        cert.valid_until = Some(past.to_rfc3339());
        assert!(!cert.is_valid());
    }

    // ── Serialization ──

    #[test]
    fn test_serialize_deserialize_full() {
        let mut cert = make_cert();
        cert.with_stark_proof(vec![10, 20, 30]);
        cert.finalize();

        let bytes = cert.serialize_full();
        let cert2 = EvaluationCertificate::deserialize(&bytes).unwrap();
        assert_eq!(cert2.certificate_id, cert.certificate_id);
        assert!(cert2.stark_proof.is_some());
    }

    #[test]
    fn test_serialize_compact() {
        let mut cert = make_cert();
        cert.with_stark_proof(vec![10, 20, 30]);

        let compact = cert.serialize_compact();
        let cert2 = EvaluationCertificate::deserialize(&compact).unwrap();
        assert!(cert2.stark_proof.is_none());
    }

    #[test]
    fn test_json_roundtrip() {
        let cert = make_cert();
        let json = cert.to_json();
        let cert2 = EvaluationCertificate::from_json(&json).unwrap();
        assert_eq!(cert2.certificate_id, cert.certificate_id);
    }

    #[test]
    fn test_deserialize_invalid() {
        let result = EvaluationCertificate::deserialize(b"not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_json_invalid() {
        let result = EvaluationCertificate::from_json("not json");
        assert!(result.is_err());
    }

    // ── Human-readable summary ──

    #[test]
    fn test_human_readable_summary() {
        let cert = make_cert();
        let summary = cert.human_readable_summary();
        assert!(summary.contains("accuracy"));
        assert!(summary.contains("gpt-4"));
        assert!(summary.contains("mmlu"));
        assert!(summary.contains("0.950000"));
    }

    // ── Certificate hash ──

    #[test]
    fn test_certificate_hash_deterministic() {
        let cert = make_cert();
        assert_eq!(cert.certificate_hash(), cert.certificate_hash());
    }

    #[test]
    fn test_certificate_hash_different_certs() {
        let c1 = make_cert();
        let c2 = make_cert(); // different UUID
        assert_ne!(c1.certificate_hash(), c2.certificate_hash());
    }

    // ── With methods ──

    #[test]
    fn test_with_stark_proof() {
        let mut cert = make_cert();
        cert.with_stark_proof(vec![1, 2, 3, 4]);
        assert!(cert.stark_proof.is_some());
        assert_ne!(cert.stark_proof_hash, [0u8; 32]);
        assert_eq!(cert.metadata.proof_size_bytes, 4);
    }

    #[test]
    fn test_with_psi_attestation() {
        let mut cert = make_cert();
        let psi = PSIAttestation::new(0.05, 3, 50, [1u8; 32], true);
        cert.with_psi_attestation(psi);
        assert!(cert.psi_attestation.is_some());
    }

    #[test]
    fn test_add_commitment_opening() {
        let mut cert = make_cert();
        let opening = CommitmentOpening::new("test", [0u8; 32], vec![1], vec![2], "blake3");
        cert.add_commitment_opening(opening);
        assert_eq!(cert.commitment_openings.len(), 1);
    }

    #[test]
    fn test_set_validity() {
        let mut cert = make_cert();
        cert.set_validity(48);
        assert!(cert.valid_until.is_some());
    }

    // ── chain_with ──

    #[test]
    fn test_chain_with() {
        let c1 = make_cert();
        let c2 = make_cert();
        let chain = c1.chain_with(&c2);
        assert_eq!(chain.len(), 2);
    }

    // ── CertificateChain ──

    #[test]
    fn test_chain_add() {
        let mut chain = CertificateChain::new();
        assert!(chain.is_empty());
        chain.add(make_cert());
        assert_eq!(chain.len(), 1);
        assert!(!chain.is_empty());
    }

    #[test]
    fn test_chain_verify() {
        let mut chain = CertificateChain::new();
        let mut c1 = make_cert();
        c1.finalize();
        let mut c2 = make_cert();
        c2.finalize();

        chain.add(c1);
        chain.add(c2);
        assert!(chain.verify_chain());
    }

    #[test]
    fn test_chain_verify_tampered() {
        let mut chain = CertificateChain::new();
        let mut c1 = make_cert();
        c1.finalize();
        chain.add(c1);

        // Tamper with the chain hash
        chain.chain_hash = [0xFFu8; 32];
        assert!(!chain.verify_chain());
    }

    #[test]
    fn test_chain_combined_result() {
        let mut chain = CertificateChain::new();
        let r1 = CertifiedResult::new(0.80, 800000);
        let r2 = CertifiedResult::new(0.90, 900000);

        let mut c1 = EvaluationCertificate::new("acc", "m1", "b1", r1);
        c1.finalize();
        let mut c2 = EvaluationCertificate::new("acc", "m1", "b1", r2);
        c2.finalize();

        chain.add(c1);
        chain.add(c2);

        let combined = chain.combined_result().unwrap();
        assert!((combined.score - 0.85).abs() < 1e-9);
    }

    #[test]
    fn test_chain_combined_result_empty() {
        let chain = CertificateChain::new();
        assert!(chain.combined_result().is_none());
    }

    #[test]
    fn test_chain_to_json() {
        let mut chain = CertificateChain::new();
        chain.add(make_cert());
        let json = chain.to_json();
        assert!(json.contains("certificates"));
    }

    // ── CertificateStore ──

    #[test]
    fn test_store_insert_get() {
        let mut store = CertificateStore::new();
        let cert = make_cert();
        let id = cert.certificate_id.clone();
        store.insert(cert);
        assert!(store.get(&id).is_some());
    }

    #[test]
    fn test_store_list() {
        let mut store = CertificateStore::new();
        store.insert(make_cert());
        store.insert(make_cert());
        assert_eq!(store.list().len(), 2);
    }

    #[test]
    fn test_store_remove() {
        let mut store = CertificateStore::new();
        let cert = make_cert();
        let id = cert.certificate_id.clone();
        store.insert(cert);
        assert!(store.remove(&id).is_some());
        assert!(store.get(&id).is_none());
    }

    #[test]
    fn test_store_find_by_model() {
        let mut store = CertificateStore::new();
        store.insert(make_cert()); // model = "gpt-4"

        let r = CertifiedResult::new(0.5, 500000);
        store.insert(EvaluationCertificate::new("acc", "llama", "b", r));

        assert_eq!(store.find_by_model("gpt-4").len(), 1);
        assert_eq!(store.find_by_model("llama").len(), 1);
        assert_eq!(store.find_by_model("missing").len(), 0);
    }

    #[test]
    fn test_store_find_by_metric() {
        let mut store = CertificateStore::new();
        store.insert(make_cert()); // metric = "accuracy"

        let r = CertifiedResult::new(0.3, 300000);
        store.insert(EvaluationCertificate::new("bleu", "m", "b", r));

        assert_eq!(store.find_by_metric("accuracy").len(), 1);
        assert_eq!(store.find_by_metric("bleu").len(), 1);
    }

    #[test]
    fn test_store_export_import() {
        let mut store = CertificateStore::new();
        store.insert(make_cert());
        store.insert(make_cert());

        let exported = store.export_all();
        let store2 = CertificateStore::import(&exported).unwrap();
        assert_eq!(store2.len(), 2);
    }

    #[test]
    fn test_store_import_invalid() {
        let result = CertificateStore::import(b"bad");
        assert!(result.is_err());
    }

    #[test]
    fn test_store_empty() {
        let store = CertificateStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    // ── CertifiedResult ──

    #[test]
    fn test_certified_result_builder() {
        let r = CertifiedResult::new(0.9, 900000)
            .with_confidence(0.95)
            .with_detail("note", "test");
        assert_eq!(r.confidence, Some(0.95));
        assert_eq!(r.details.get("note"), Some(&"test".to_string()));
    }

    // ── CommitmentOpening ──

    #[test]
    fn test_commitment_opening_verify_valid() {
        let value = b"test data".to_vec();
        let randomness = b"random".to_vec();

        let mut hasher = blake3::Hasher::new();
        hasher.update(&randomness);
        hasher.update(&value);
        let hash = *hasher.finalize().as_bytes();

        let opening = CommitmentOpening::new("test", hash, value, randomness, "blake3");
        assert!(opening.verify());
    }

    #[test]
    fn test_commitment_opening_verify_invalid() {
        let opening = CommitmentOpening::new(
            "test",
            [0xFFu8; 32],
            b"value".to_vec(),
            b"random".to_vec(),
            "blake3",
        );
        assert!(!opening.verify());
    }

    // ── CertificateMetadata ──

    #[test]
    fn test_metadata_default() {
        let m = CertificateMetadata::default();
        assert_eq!(m.security_bits, 128);
        assert_eq!(m.proving_time_ms, 0);
    }

    // ── PSIAttestation ──

    #[test]
    fn test_psi_attestation() {
        let psi = PSIAttestation::new(0.01, 5, 100, [42u8; 32], true);
        assert!(psi.threshold_satisfied);
        assert_eq!(psi.ngram_size, 5);
    }

    // ── CertificateVerification ──

    #[test]
    fn test_verification_all_pass() {
        let mut v = CertificateVerification::new();
        v.add_check(VerificationCheck::new("a", true, "ok"));
        v.add_check(VerificationCheck::new("b", true, "ok"));
        assert!(v.is_valid);
    }

    #[test]
    fn test_verification_one_fails() {
        let mut v = CertificateVerification::new();
        v.add_check(VerificationCheck::new("a", true, "ok"));
        v.add_check(VerificationCheck::new("b", false, "bad"));
        assert!(!v.is_valid);
    }

    // ── CertificateError display ──

    #[test]
    fn test_error_display() {
        assert_eq!(format!("{}", CertificateError::InvalidFormat), "InvalidFormat");
        assert_eq!(
            format!("{}", CertificateError::MissingField("x".to_string())),
            "MissingField(x)"
        );
    }

    // ── Finalize ──

    #[test]
    fn test_finalize_updates_timestamp() {
        let mut cert = make_cert();
        let ts1 = cert.timestamp.clone();
        std::thread::sleep(std::time::Duration::from_millis(10));
        cert.finalize();
        // Timestamp should have been updated (or at least re-set)
        assert!(!cert.timestamp.is_empty());
    }

    #[test]
    fn test_finalize_updates_metric_hash() {
        let mut cert = make_cert();
        cert.metric_name = "new-metric".to_string();
        cert.finalize();
        let expected = *blake3::hash(b"new-metric").as_bytes();
        assert_eq!(cert.metric_hash, expected);
    }
    // ── CertificateBuilder extended ──

    #[test]
    fn test_builder_missing_metric() {
        let result = CertificateBuilder::new()
            .model("m")
            .benchmark("b")
            .score(0.5)
            .build();
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            CertificateError::MissingField("metric_name".to_string())
        );
    }

    #[test]
    fn test_builder_missing_model() {
        let result = CertificateBuilder::new()
            .metric("acc")
            .benchmark("b")
            .score(0.5)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_missing_benchmark() {
        let result = CertificateBuilder::new()
            .metric("acc")
            .model("m")
            .score(0.5)
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_score_field_element() {
        let cert = CertificateBuilder::new()
            .metric("acc")
            .model("m")
            .benchmark("b")
            .score(0.5)
            .score_field_element(42)
            .build()
            .unwrap();
        assert_eq!(cert.evaluation_result.score_field_element, 42);
    }

    #[test]
    fn test_builder_add_commitment() {
        let value = b"test".to_vec();
        let randomness = b"rand".to_vec();
        let mut hasher = blake3::Hasher::new();
        hasher.update(&randomness);
        hasher.update(&value);
        let hash = *hasher.finalize().as_bytes();
        let opening = CommitmentOpening::new("test", hash, value, randomness, "blake3");
        let cert = CertificateBuilder::new()
            .metric("acc")
            .model("m")
            .benchmark("b")
            .score(0.5)
            .add_commitment(opening)
            .build()
            .unwrap();
        assert_eq!(cert.commitment_openings.len(), 1);
    }

    #[test]
    fn test_builder_with_metadata() {
        let meta = CertificateMetadata {
            proving_time_ms: 100,
            verification_time_ms: Some(10),
            trace_width: 8,
            trace_length: 1024,
            constraint_count: 50,
            security_bits: 128,
            proof_size_bytes: 2048,
        };
        let cert = CertificateBuilder::new()
            .metric("acc")
            .model("m")
            .benchmark("b")
            .score(0.5)
            .with_metadata(meta)
            .build()
            .unwrap();
        assert_eq!(cert.metadata.proving_time_ms, 100);
    }

    #[test]
    fn test_builder_default() {
        let builder = CertificateBuilder::default();
        let result = builder.metric("acc").model("m").benchmark("b").score(0.5).build();
        assert!(result.is_ok());
    }

    // ── CertificateRegistry ──

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = CertificateRegistry::new();
        let cert = make_cert();
        let id = cert.certificate_id.clone();
        assert!(reg.register(cert).is_ok());
        assert!(reg.get(&id).is_some());
        assert_eq!(reg.count(), 1);
    }

    #[test]
    fn test_registry_duplicate_register() {
        let mut reg = CertificateRegistry::new();
        let cert = make_cert();
        let cert2 = cert.clone();
        assert!(reg.register(cert).is_ok());
        assert!(reg.register(cert2).is_err());
    }

    #[test]
    fn test_registry_find_by_model() {
        let mut reg = CertificateRegistry::new();
        reg.register(make_cert()).unwrap();
        let r = CertifiedResult::new(0.5, 500000);
        reg.register(EvaluationCertificate::new("acc", "llama", "b", r)).unwrap();
        assert_eq!(reg.find_by_model("gpt-4").len(), 1);
        assert_eq!(reg.find_by_model("llama").len(), 1);
        assert_eq!(reg.find_by_model("missing").len(), 0);
    }

    #[test]
    fn test_registry_find_by_metric() {
        let mut reg = CertificateRegistry::new();
        reg.register(make_cert()).unwrap();
        assert_eq!(reg.find_by_metric("accuracy").len(), 1);
        assert_eq!(reg.find_by_metric("bleu").len(), 0);
    }

    #[test]
    fn test_registry_find_by_benchmark() {
        let mut reg = CertificateRegistry::new();
        reg.register(make_cert()).unwrap();
        assert_eq!(reg.find_by_benchmark("mmlu").len(), 1);
        assert_eq!(reg.find_by_benchmark("other").len(), 0);
    }

    #[test]
    fn test_registry_find_valid() {
        let mut reg = CertificateRegistry::new();
        let mut cert = make_cert();
        cert.finalize();
        reg.register(cert).unwrap();
        assert_eq!(reg.find_valid().len(), 1);
    }

    #[test]
    fn test_registry_find_expired() {
        let mut reg = CertificateRegistry::new();
        let mut cert = make_cert();
        let past = chrono::Utc::now() - chrono::Duration::hours(1);
        cert.valid_until = Some(past.to_rfc3339());
        reg.register(cert).unwrap();
        assert_eq!(reg.find_expired().len(), 1);
    }

    #[test]
    fn test_registry_revoke() {
        let mut reg = CertificateRegistry::new();
        let cert = make_cert();
        let id = cert.certificate_id.clone();
        reg.register(cert).unwrap();
        assert!(!reg.is_revoked(&id));
        assert!(reg.revoke(&id).is_ok());
        assert!(reg.is_revoked(&id));
    }

    #[test]
    fn test_registry_revoke_missing() {
        let mut reg = CertificateRegistry::new();
        assert!(reg.revoke("nonexistent").is_err());
    }

    #[test]
    fn test_registry_export_import() {
        let mut reg = CertificateRegistry::new();
        reg.register(make_cert()).unwrap();
        reg.register(make_cert()).unwrap();
        let exported = reg.export_all();
        let mut reg2 = CertificateRegistry::new();
        let count = reg2.import(&exported).unwrap();
        assert_eq!(count, 2);
        assert_eq!(reg2.count(), 2);
    }

    #[test]
    fn test_registry_import_invalid() {
        let mut reg = CertificateRegistry::new();
        assert!(reg.import(b"bad data").is_err());
    }

    #[test]
    fn test_registry_prune_expired() {
        let mut reg = CertificateRegistry::new();
        let mut cert = make_cert();
        let past = chrono::Utc::now() - chrono::Duration::hours(1);
        cert.valid_until = Some(past.to_rfc3339());
        reg.register(cert).unwrap();
        let mut cert2 = make_cert();
        cert2.finalize();
        reg.register(cert2).unwrap();
        assert_eq!(reg.count(), 2);
        let pruned = reg.prune_expired();
        assert_eq!(pruned, 1);
        assert_eq!(reg.count(), 1);
    }

    #[test]
    fn test_registry_default() {
        let reg = CertificateRegistry::default();
        assert_eq!(reg.count(), 0);
    }

    // ── CertificateComparator ──

    #[test]
    fn test_comparator_compare_scores() {
        let r1 = CertifiedResult::new(0.8, 800000);
        let r2 = CertifiedResult::new(0.9, 900000);
        let c1 = EvaluationCertificate::new("acc", "m1", "b", r1);
        let c2 = EvaluationCertificate::new("acc", "m2", "b", r2);
        assert_eq!(
            CertificateComparator::compare_scores(&c1, &c2),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            CertificateComparator::compare_scores(&c2, &c1),
            std::cmp::Ordering::Greater
        );
    }

    #[test]
    fn test_comparator_compare_contamination() {
        let r1 = CertifiedResult::new(0.8, 800000);
        let r2 = CertifiedResult::new(0.9, 900000);
        let mut c1 = EvaluationCertificate::new("acc", "m1", "b", r1);
        let mut c2 = EvaluationCertificate::new("acc", "m2", "b", r2);
        c1.with_psi_attestation(PSIAttestation::new(0.01, 5, 100, [0u8; 32], true));
        c2.with_psi_attestation(PSIAttestation::new(0.05, 5, 100, [0u8; 32], false));
        let cmp = CertificateComparator::compare_contamination(&c1, &c2);
        assert!(cmp.is_some());
        assert_eq!(cmp.unwrap(), std::cmp::Ordering::Less);
    }

    #[test]
    fn test_comparator_compare_contamination_none() {
        let c1 = make_cert();
        let c2 = make_cert();
        assert!(CertificateComparator::compare_contamination(&c1, &c2).is_none());
    }

    #[test]
    fn test_comparator_are_compatible() {
        let r1 = CertifiedResult::new(0.8, 800000);
        let r2 = CertifiedResult::new(0.9, 900000);
        let c1 = EvaluationCertificate::new("acc", "m1", "b", r1);
        let c2 = EvaluationCertificate::new("acc", "m2", "b", r2);
        assert!(CertificateComparator::are_compatible(&c1, &c2));
        let r3 = CertifiedResult::new(0.7, 700000);
        let c3 = EvaluationCertificate::new("bleu", "m1", "b", r3);
        assert!(!CertificateComparator::are_compatible(&c1, &c3));
    }

    #[test]
    fn test_comparator_merge_results() {
        let r1 = CertifiedResult::new(0.8, 800000);
        let r2 = CertifiedResult::new(0.9, 900000);
        let c1 = EvaluationCertificate::new("acc", "m1", "b", r1);
        let c2 = EvaluationCertificate::new("acc", "m2", "b", r2);
        let merged = CertificateComparator::merge_results(&[&c1, &c2]);
        assert!((merged.score - 0.85).abs() < 1e-9);
    }

    #[test]
    fn test_comparator_merge_results_empty() {
        let merged = CertificateComparator::merge_results(&[]);
        assert_eq!(merged.score, 0.0);
    }

    #[test]
    fn test_comparator_score_delta() {
        let r1 = CertifiedResult::new(0.8, 800000);
        let r2 = CertifiedResult::new(0.9, 900000);
        let c1 = EvaluationCertificate::new("acc", "m1", "b", r1);
        let c2 = EvaluationCertificate::new("acc", "m2", "b", r2);
        assert!((CertificateComparator::score_delta(&c1, &c2) - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_comparator_summary() {
        let r1 = CertifiedResult::new(0.7, 700000);
        let r2 = CertifiedResult::new(0.9, 900000);
        let c1 = EvaluationCertificate::new("acc", "m1", "b", r1);
        let c2 = EvaluationCertificate::new("acc", "m2", "b", r2);
        let summary = CertificateComparator::summary_comparison(&[&c1, &c2]);
        assert_eq!(summary.num_certificates, 2);
        assert!((summary.avg_score - 0.8).abs() < 1e-9);
        assert_eq!(summary.best_model, "m2");
        assert_eq!(summary.worst_model, "m1");
    }

    #[test]
    fn test_comparator_summary_empty() {
        let summary = CertificateComparator::summary_comparison(&[]);
        assert_eq!(summary.num_certificates, 0);
    }

    // ── CertificateReport ──

    #[test]
    fn test_report_from_certificate() {
        let cert = make_cert();
        let report = CertificateReport::from_certificate(&cert);
        assert_eq!(report.metric_name, "accuracy");
        assert_eq!(report.model_id, "gpt-4");
        assert_eq!(report.score, 0.95);
    }

    #[test]
    fn test_report_from_chain() {
        let mut chain = CertificateChain::new();
        let mut c = make_cert();
        c.finalize();
        chain.add(c);
        let report = CertificateReport::from_chain(&chain);
        assert_eq!(report.chain_length, Some(1));
    }

    #[test]
    fn test_report_to_text() {
        let cert = make_cert();
        let report = CertificateReport::from_certificate(&cert);
        let text = report.to_text();
        assert!(text.contains("Certificate Report"));
        assert!(text.contains("accuracy"));
        assert!(text.contains("gpt-4"));
    }

    #[test]
    fn test_report_to_html() {
        let cert = make_cert();
        let report = CertificateReport::from_certificate(&cert);
        let html = report.to_html();
        assert!(html.contains("<div"));
        assert!(html.contains("accuracy"));
    }

    #[test]
    fn test_report_to_json() {
        let cert = make_cert();
        let report = CertificateReport::from_certificate(&cert);
        let json = report.to_json();
        assert!(json.contains("accuracy"));
        assert!(json.contains("gpt-4"));
    }

    #[test]
    fn test_report_to_csv_row() {
        let cert = make_cert();
        let report = CertificateReport::from_certificate(&cert);
        let csv = report.to_csv_row();
        assert!(csv.contains("accuracy"));
        assert!(csv.contains("gpt-4"));
        assert!(csv.contains(","));
    }

    // ── CertificateValidator ──

    #[test]
    fn test_validator_standard_rules_valid() {
        let mut cert = make_cert();
        cert.finalize();
        let validator = CertificateValidator::with_standard_rules();
        assert!(validator.is_valid(&cert));
    }

    #[test]
    fn test_validator_standard_rules_invalid_score() {
        let r = CertifiedResult::new(2.0, 2000000);
        let mut cert = EvaluationCertificate::new("acc", "m", "b", r);
        cert.finalize();
        let validator = CertificateValidator::with_standard_rules();
        assert!(!validator.is_valid(&cert));
    }

    #[test]
    fn test_validator_custom_rule() {
        let mut validator = CertificateValidator::new();
        validator.add_rule(ValidationRule {
            name: "custom".to_string(),
            description: "Score > 0.5".to_string(),
            check: |cert| cert.evaluation_result.score > 0.5,
        });
        let cert = make_cert();
        assert!(validator.is_valid(&cert));
    }

    #[test]
    fn test_validator_validate_returns_results() {
        let mut cert = make_cert();
        cert.finalize();
        let validator = CertificateValidator::with_standard_rules();
        let results = validator.validate(&cert);
        assert!(!results.is_empty());
        assert!(results.iter().all(|r| r.passed));
    }

    #[test]
    fn test_validator_validate_chain() {
        let mut chain = CertificateChain::new();
        let mut c = make_cert();
        c.finalize();
        chain.add(c);
        let validator = CertificateValidator::with_standard_rules();
        let results = validator.validate_chain(&chain);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_validator_default() {
        let validator = CertificateValidator::default();
        let cert = make_cert();
        let results = validator.validate(&cert);
        assert!(results.is_empty());
        assert!(validator.is_valid(&cert));
    }

    // ── CertificateAggregator ──

    #[test]
    fn test_aggregator_aggregate_score() {
        let mut agg = CertificateAggregator::new();
        let r1 = CertifiedResult::new(0.8, 800000);
        let r2 = CertifiedResult::new(0.9, 900000);
        agg.add(EvaluationCertificate::new("acc", "m1", "b", r1));
        agg.add(EvaluationCertificate::new("acc", "m2", "b", r2));
        assert!((agg.aggregate_score() - 0.85).abs() < 1e-9);
    }

    #[test]
    fn test_aggregator_empty() {
        let agg = CertificateAggregator::new();
        assert_eq!(agg.aggregate_score(), 0.0);
    }

    #[test]
    fn test_aggregator_score_distribution() {
        let mut agg = CertificateAggregator::new();
        for s in &[0.7, 0.8, 0.85, 0.9, 0.95] {
            let r = CertifiedResult::new(*s, (*s * 1_000_000.0) as u64);
            agg.add(EvaluationCertificate::new("acc", "m", "b", r));
        }
        let dist = agg.score_distribution();
        assert!((dist.min - 0.7).abs() < 1e-9);
        assert!((dist.max - 0.95).abs() < 1e-9);
        assert!((dist.median - 0.85).abs() < 1e-9);
        assert!(dist.stddev > 0.0);
        assert!(!dist.percentiles.is_empty());
    }

    #[test]
    fn test_aggregator_score_distribution_empty() {
        let agg = CertificateAggregator::new();
        let dist = agg.score_distribution();
        assert_eq!(dist.mean, 0.0);
        assert_eq!(dist.median, 0.0);
    }

    #[test]
    fn test_aggregator_contamination_summary() {
        let mut agg = CertificateAggregator::new();
        let r1 = CertifiedResult::new(0.8, 800000);
        let mut c1 = EvaluationCertificate::new("acc", "m1", "b", r1);
        c1.with_psi_attestation(PSIAttestation::new(0.01, 5, 100, [0u8; 32], true));
        agg.add(c1);
        let r2 = CertifiedResult::new(0.9, 900000);
        let mut c2 = EvaluationCertificate::new("acc", "m2", "b", r2);
        c2.with_psi_attestation(PSIAttestation::new(0.1, 5, 100, [0u8; 32], false));
        agg.add(c2);

        let summary = agg.contamination_summary();
        assert_eq!(summary.num_clean, 1);
        assert_eq!(summary.num_contaminated, 1);
        assert!((summary.max_contamination - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_aggregator_contamination_no_psi() {
        let mut agg = CertificateAggregator::new();
        agg.add(make_cert());
        let summary = agg.contamination_summary();
        assert_eq!(summary.num_clean, 1);
        assert_eq!(summary.num_contaminated, 0);
    }

    #[test]
    fn test_aggregator_model_leaderboard() {
        let mut agg = CertificateAggregator::new();
        let r1 = CertifiedResult::new(0.8, 800000);
        let r2 = CertifiedResult::new(0.9, 900000);
        agg.add(EvaluationCertificate::new("acc", "model-a", "b", r1));
        agg.add(EvaluationCertificate::new("acc", "model-b", "b", r2));
        let lb = agg.model_leaderboard();
        assert_eq!(lb.len(), 2);
        assert_eq!(lb[0].0, "model-b");
        assert_eq!(lb[1].0, "model-a");
    }

    #[test]
    fn test_aggregator_metric_summary() {
        let mut agg = CertificateAggregator::new();
        let r1 = CertifiedResult::new(0.8, 800000);
        let r2 = CertifiedResult::new(0.9, 900000);
        agg.add(EvaluationCertificate::new("acc", "m", "b", r1));
        agg.add(EvaluationCertificate::new("bleu", "m", "b", r2));
        let summary = agg.metric_summary();
        assert_eq!(summary.len(), 2);
        assert!(summary.contains_key("acc"));
        assert!(summary.contains_key("bleu"));
    }

    #[test]
    fn test_aggregator_default() {
        let agg = CertificateAggregator::default();
        assert_eq!(agg.aggregate_score(), 0.0);
    }

    // ── CertificateSerializer ──

    #[test]
    fn test_serializer_binary_roundtrip() {
        let cert = make_cert();
        let bytes = CertificateSerializer::to_binary(&cert);
        let cert2 = CertificateSerializer::from_binary(&bytes).unwrap();
        assert_eq!(cert2.certificate_id, cert.certificate_id);
    }

    #[test]
    fn test_serializer_from_binary_invalid() {
        assert!(CertificateSerializer::from_binary(b"bad").is_err());
    }

    #[test]
    fn test_serializer_cbor_like() {
        let cert = make_cert();
        let bytes = CertificateSerializer::to_cbor_like(&cert);
        assert!(bytes.starts_with(b"CERT"));
        assert!(bytes.len() > 4);
    }

    #[test]
    fn test_serializer_base64_roundtrip() {
        let cert = make_cert();
        let b64 = CertificateSerializer::to_base64(&cert);
        let cert2 = CertificateSerializer::from_base64(&b64).unwrap();
        assert_eq!(cert2.certificate_id, cert.certificate_id);
    }

    #[test]
    fn test_serializer_from_base64_invalid() {
        assert!(CertificateSerializer::from_base64("dGVzdA==").is_err());
    }

    #[test]
    fn test_serializer_estimate_size() {
        let cert = make_cert();
        let size = CertificateSerializer::estimate_size(&cert);
        assert!(size > 0);
    }

    #[test]
    fn test_serializer_estimate_size_with_proof() {
        let mut cert = make_cert();
        cert.with_stark_proof(vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let size = CertificateSerializer::estimate_size(&cert);
        let size_no_proof = CertificateSerializer::estimate_size(&make_cert());
        assert!(size > size_no_proof);
    }

    // ── ComparisonSummary ──

    #[test]
    fn test_comparison_summary_serialization() {
        let summary = ComparisonSummary {
            num_certificates: 3,
            metrics: vec!["acc".to_string()],
            score_range: (0.5, 0.9),
            avg_score: 0.7,
            best_model: "m1".to_string(),
            worst_model: "m2".to_string(),
            avg_contamination: 0.02,
        };
        let json = serde_json::to_string(&summary).unwrap();
        let deserialized: ComparisonSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.num_certificates, 3);
        assert_eq!(deserialized.best_model, "m1");
    }

    // ── ScoreDistribution ──

    #[test]
    fn test_score_distribution_serialization() {
        let mut percentiles = HashMap::new();
        percentiles.insert(50, 0.85);
        let dist = ScoreDistribution {
            mean: 0.85,
            median: 0.85,
            stddev: 0.1,
            min: 0.7,
            max: 0.95,
            percentiles,
        };
        let json = serde_json::to_string(&dist).unwrap();
        let d2: ScoreDistribution = serde_json::from_str(&json).unwrap();
        assert!((d2.mean - 0.85).abs() < 1e-9);
    }

    // ── ContaminationSummary ──

    #[test]
    fn test_contamination_summary_serialization() {
        let cs = ContaminationSummary {
            avg_contamination: 0.03,
            max_contamination: 0.1,
            num_clean: 8,
            num_contaminated: 2,
            threshold: 0.05,
        };
        let json = serde_json::to_string(&cs).unwrap();
        let cs2: ContaminationSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(cs2.num_clean, 8);
        assert_eq!(cs2.num_contaminated, 2);
    }

    // ── ValidationResult ──

    #[test]
    fn test_validation_result_fields() {
        let vr = ValidationResult {
            rule_name: "test_rule".to_string(),
            passed: true,
            details: "All good".to_string(),
        };
        assert_eq!(vr.rule_name, "test_rule");
        assert!(vr.passed);
        assert_eq!(vr.details, "All good");
    }

    // ── Base64 helpers ──

    #[test]
    fn test_base64_encode_decode_roundtrip() {
        let data = b"Hello, certificate world!";
        let encoded = base64_encode(data);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_base64_empty() {
        let encoded = base64_encode(b"");
        let decoded = base64_decode(&encoded).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_base64_single_byte() {
        let data = b"A";
        let encoded = base64_encode(data);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_base64_two_bytes() {
        let data = b"AB";
        let encoded = base64_encode(data);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    // ── CertificateRegistry revoked filtering ──

    #[test]
    fn test_registry_revoked_not_in_valid() {
        let mut reg = CertificateRegistry::new();
        let mut cert = make_cert();
        cert.finalize();
        let id = cert.certificate_id.clone();
        reg.register(cert).unwrap();
        assert_eq!(reg.find_valid().len(), 1);
        reg.revoke(&id).unwrap();
        assert_eq!(reg.find_valid().len(), 0);
    }

    // ── CertificateComparator equal scores ──

    #[test]
    fn test_comparator_equal_scores() {
        let r1 = CertifiedResult::new(0.85, 850000);
        let r2 = CertifiedResult::new(0.85, 850000);
        let c1 = EvaluationCertificate::new("acc", "m1", "b", r1);
        let c2 = EvaluationCertificate::new("acc", "m2", "b", r2);
        assert_eq!(
            CertificateComparator::compare_scores(&c1, &c2),
            std::cmp::Ordering::Equal
        );
    }

    // ── CertificateReport from empty chain ──

    #[test]
    fn test_report_from_empty_chain() {
        let chain = CertificateChain::new();
        let report = CertificateReport::from_chain(&chain);
        assert_eq!(report.chain_length, Some(0));
        assert_eq!(report.score, 0.0);
    }

    // ── CertificateValidator with expired cert ──

    #[test]
    fn test_validator_expired_cert() {
        let mut cert = make_cert();
        cert.finalize();
        let past = chrono::Utc::now() - chrono::Duration::hours(1);
        cert.valid_until = Some(past.to_rfc3339());
        let validator = CertificateValidator::with_standard_rules();
        assert!(!validator.is_valid(&cert));
    }

    // ── CertificateAggregator single cert ──

    #[test]
    fn test_aggregator_single_cert() {
        let mut agg = CertificateAggregator::new();
        agg.add(make_cert());
        assert!((agg.aggregate_score() - 0.95).abs() < 1e-9);
        let dist = agg.score_distribution();
        assert!((dist.mean - 0.95).abs() < 1e-9);
        assert!((dist.median - 0.95).abs() < 1e-9);
        assert!((dist.stddev - 0.0).abs() < 1e-9);
    }

    // ── CertificateSerializer with full cert ──

    #[test]
    fn test_serializer_full_cert_roundtrip() {
        let mut cert = make_cert();
        cert.with_stark_proof(vec![10, 20, 30, 40]);
        cert.with_psi_attestation(PSIAttestation::new(0.02, 5, 100, [1u8; 32], true));
        cert.finalize();

        let bytes = CertificateSerializer::to_binary(&cert);
        let cert2 = CertificateSerializer::from_binary(&bytes).unwrap();
        assert_eq!(cert2.certificate_id, cert.certificate_id);
        assert!(cert2.stark_proof.is_some());
        assert!(cert2.psi_attestation.is_some());
    }

    #[test]
    fn test_serializer_cbor_like_contains_fields() {
        let cert = make_cert();
        let bytes = CertificateSerializer::to_cbor_like(&cert);
        // Should contain the magic bytes
        assert_eq!(&bytes[0..4], b"CERT");
        // Should contain the metric name somewhere after the headers
        let metric_bytes = cert.metric_name.as_bytes();
        let contains_metric = bytes.windows(metric_bytes.len())
            .any(|w| w == metric_bytes);
        assert!(contains_metric);
    }

    // ── CertificateExporter tests ──

    #[test]
    fn test_exporter_pem_roundtrip() {
        let cert = make_cert();
        let pem = CertificateExporter::to_pem(&cert);
        let cert2 = CertificateExporter::from_pem(&pem).unwrap();
        assert_eq!(cert2.certificate_id, cert.certificate_id);
        assert_eq!(cert2.metric_name, cert.metric_name);
        assert!((cert2.evaluation_result.score - cert.evaluation_result.score).abs() < 1e-9);
    }

    #[test]
    fn test_exporter_pem_format() {
        let cert = make_cert();
        let pem = CertificateExporter::to_pem(&cert);
        assert!(pem.starts_with("-----BEGIN SPECTACLES CERTIFICATE-----\n"));
        assert!(pem.ends_with("-----END SPECTACLES CERTIFICATE-----\n"));
        // Check that base64 lines are at most 64 chars
        let lines: Vec<&str> = pem.lines().collect();
        for line in &lines[1..lines.len() - 1] {
            assert!(line.len() <= 64, "PEM line too long: {}", line.len());
        }
    }

    #[test]
    fn test_exporter_from_pem_invalid_header() {
        let result = CertificateExporter::from_pem("not a pem at all");
        assert!(result.is_err());
        match result {
            Err(CertificateError::InvalidFormat) => {}
            other => panic!("Expected InvalidFormat, got {:?}", other),
        }
    }

    #[test]
    fn test_exporter_from_pem_bad_base64() {
        let bad_pem = "-----BEGIN SPECTACLES CERTIFICATE-----\n!!!\n-----END SPECTACLES CERTIFICATE-----\n";
        let result = CertificateExporter::from_pem(bad_pem);
        assert!(result.is_err());
    }

    #[test]
    fn test_exporter_der_format() {
        let cert = make_cert();
        let der = CertificateExporter::to_der(&cert);
        // Check magic bytes
        assert_eq!(&der[0..4], b"SPEC");
        // Check version
        let version = u32::from_le_bytes([der[4], der[5], der[6], der[7]]);
        assert_eq!(version, cert.version);
        // Check length
        let json_len = u32::from_le_bytes([der[8], der[9], der[10], der[11]]) as usize;
        // Total should be 4 + 4 + 4 + json_len + 32
        assert_eq!(der.len(), 4 + 4 + 4 + json_len + 32);
    }

    #[test]
    fn test_exporter_der_magic_bytes() {
        let cert = make_cert();
        let der = CertificateExporter::to_der(&cert);
        assert_eq!(&der[..4], b"SPEC");
    }

    #[test]
    fn test_exporter_der_hash_integrity() {
        let cert = make_cert();
        let der = CertificateExporter::to_der(&cert);
        let json_len = u32::from_le_bytes([der[8], der[9], der[10], der[11]]) as usize;
        let json_bytes = &der[12..12 + json_len];
        let stored_hash = &der[12 + json_len..12 + json_len + 32];
        let computed_hash = blake3::hash(json_bytes);
        assert_eq!(stored_hash, computed_hash.as_bytes());
    }

    #[test]
    fn test_exporter_jwk_contains_fields() {
        let cert = make_cert();
        let jwk = CertificateExporter::to_jwk(&cert);
        assert!(jwk.contains("\"kty\":\"SPECTACLES\""));
        assert!(jwk.contains(&format!("\"kid\":\"{}\"", cert.certificate_id)));
        assert!(jwk.contains(&format!("\"metric\":\"{}\"", cert.metric_name)));
        assert!(jwk.contains(&format!("\"score\":{}", cert.evaluation_result.score)));
        assert!(jwk.contains("\"hash\":\""));
        assert!(jwk.contains("\"timestamp\":\""));
    }

    #[test]
    fn test_exporter_jwk_is_valid_json() {
        let cert = make_cert();
        let jwk = CertificateExporter::to_jwk(&cert);
        let parsed: serde_json::Value = serde_json::from_str(&jwk).unwrap();
        assert_eq!(parsed["kty"], "SPECTACLES");
    }

    // ── CertificateTimeline tests ──

    #[test]
    fn test_timeline_new() {
        let tl = CertificateTimeline::new();
        assert!(tl.timeline_for("anything").is_empty());
    }

    #[test]
    fn test_timeline_record_creation() {
        let mut tl = CertificateTimeline::new();
        tl.record_creation("cert-1", 100);
        let events = tl.timeline_for("cert-1");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, 100);
        assert_eq!(events[0].1, "Created");
    }

    #[test]
    fn test_timeline_record_verification() {
        let mut tl = CertificateTimeline::new();
        tl.record_verification("cert-1", 200, true);
        let events = tl.timeline_for("cert-1");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].1, "Verified(true)");
    }

    #[test]
    fn test_timeline_record_revocation() {
        let mut tl = CertificateTimeline::new();
        tl.record_revocation("cert-1", 300);
        let events = tl.timeline_for("cert-1");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].1, "Revoked");
    }

    #[test]
    fn test_timeline_for_sorted() {
        let mut tl = CertificateTimeline::new();
        tl.record_verification("cert-1", 300, false);
        tl.record_creation("cert-1", 100);
        tl.record_revocation("cert-1", 500);
        let events = tl.timeline_for("cert-1");
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].0, 100);
        assert_eq!(events[1].0, 300);
        assert_eq!(events[2].0, 500);
    }

    #[test]
    fn test_timeline_for_filters_by_id() {
        let mut tl = CertificateTimeline::new();
        tl.record_creation("cert-1", 100);
        tl.record_creation("cert-2", 200);
        tl.record_verification("cert-1", 300, true);
        let events = tl.timeline_for("cert-1");
        assert_eq!(events.len(), 2);
        let events2 = tl.timeline_for("cert-2");
        assert_eq!(events2.len(), 1);
    }

    #[test]
    fn test_timeline_active_at_creation_only() {
        let mut tl = CertificateTimeline::new();
        tl.record_creation("cert-1", 100);
        let active = tl.active_at(100);
        assert_eq!(active, vec!["cert-1"]);
        let active_before = tl.active_at(50);
        assert!(active_before.is_empty());
    }

    #[test]
    fn test_timeline_active_at_with_revocation() {
        let mut tl = CertificateTimeline::new();
        tl.record_creation("cert-1", 100);
        tl.record_revocation("cert-1", 300);
        let active = tl.active_at(200);
        assert_eq!(active, vec!["cert-1"]);
        let active_after = tl.active_at(300);
        assert!(active_after.is_empty());
    }

    #[test]
    fn test_timeline_active_at_multiple_certs() {
        let mut tl = CertificateTimeline::new();
        tl.record_creation("alpha", 100);
        tl.record_creation("beta", 200);
        tl.record_creation("gamma", 300);
        tl.record_revocation("beta", 250);
        let active = tl.active_at(350);
        assert_eq!(active, vec!["alpha", "gamma"]);
    }

    // ── CertificateExporter: from_der / to_csv_row tests ──

    #[test]
    fn test_exporter_der_roundtrip() {
        let cert = make_cert();
        let der = CertificateExporter::to_der(&cert);
        let cert2 = CertificateExporter::from_der(&der).unwrap();
        assert_eq!(cert2.certificate_id, cert.certificate_id);
        assert_eq!(cert2.metric_name, cert.metric_name);
        assert!((cert2.evaluation_result.score - cert.evaluation_result.score).abs() < 1e-9);
    }

    #[test]
    fn test_exporter_from_der_too_short() {
        let result = CertificateExporter::from_der(&[0u8; 5]);
        assert!(matches!(result, Err(CertificateError::InvalidFormat)));
    }

    #[test]
    fn test_exporter_from_der_bad_magic() {
        let mut der = CertificateExporter::to_der(&make_cert());
        der[0] = b'X';
        assert!(matches!(
            CertificateExporter::from_der(&der),
            Err(CertificateError::InvalidFormat)
        ));
    }

    #[test]
    fn test_exporter_from_der_tampered_hash() {
        let mut der = CertificateExporter::to_der(&make_cert());
        let last = der.len() - 1;
        der[last] ^= 0xFF;
        assert!(matches!(
            CertificateExporter::from_der(&der),
            Err(CertificateError::IntegrityCheckFailed)
        ));
    }

    #[test]
    fn test_exporter_csv_row_format() {
        let cert = make_cert();
        let csv = CertificateExporter::to_csv_row(&cert);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("certificate_id,"));
        assert!(lines[1].contains(&cert.certificate_id));
        assert!(lines[1].contains(&cert.metric_name));
    }

    #[test]
    fn test_exporter_csv_row_field_count() {
        let cert = make_cert();
        let csv = CertificateExporter::to_csv_row(&cert);
        let lines: Vec<&str> = csv.lines().collect();
        let header_fields: Vec<&str> = lines[0].split(',').collect();
        let data_fields: Vec<&str> = lines[1].split(',').collect();
        assert_eq!(header_fields.len(), data_fields.len());
    }

    // ── CertificateAuditLog tests ──

    #[test]
    fn test_audit_log_new() {
        let log = CertificateAuditLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    #[test]
    fn test_audit_log_append() {
        let mut log = CertificateAuditLog::new();
        log.log(100, "cert-1", "created", AuditSeverity::Info, "initial creation");
        assert_eq!(log.len(), 1);
        assert!(!log.is_empty());
    }

    #[test]
    fn test_audit_log_entries_for() {
        let mut log = CertificateAuditLog::new();
        log.log(100, "cert-1", "created", AuditSeverity::Info, "");
        log.log(200, "cert-2", "created", AuditSeverity::Info, "");
        log.log(300, "cert-1", "verified", AuditSeverity::Info, "passed");
        let entries = log.entries_for("cert-1");
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].action, "created");
        assert_eq!(entries[1].action, "verified");
    }

    #[test]
    fn test_audit_log_entries_by_severity() {
        let mut log = CertificateAuditLog::new();
        log.log(100, "cert-1", "created", AuditSeverity::Info, "");
        log.log(200, "cert-1", "tampered", AuditSeverity::Error, "hash mismatch");
        log.log(300, "cert-1", "expiring", AuditSeverity::Warning, "expires soon");
        assert_eq!(log.entries_by_severity(&AuditSeverity::Info).len(), 1);
        assert_eq!(log.entries_by_severity(&AuditSeverity::Error).len(), 1);
        assert_eq!(log.entries_by_severity(&AuditSeverity::Warning).len(), 1);
    }

    #[test]
    fn test_audit_log_entries_in_range() {
        let mut log = CertificateAuditLog::new();
        log.log(100, "c1", "a", AuditSeverity::Info, "");
        log.log(200, "c2", "b", AuditSeverity::Info, "");
        log.log(300, "c3", "c", AuditSeverity::Info, "");
        log.log(400, "c4", "d", AuditSeverity::Info, "");
        let range = log.entries_in_range(150, 350);
        assert_eq!(range.len(), 2);
        assert_eq!(range[0].cert_id, "c2");
        assert_eq!(range[1].cert_id, "c3");
    }

    #[test]
    fn test_audit_log_summary() {
        let mut log = CertificateAuditLog::new();
        log.log(100, "cert-1", "created", AuditSeverity::Info, "");
        log.log(200, "cert-2", "failed", AuditSeverity::Error, "bad");
        log.log(300, "cert-1", "warn", AuditSeverity::Warning, "w");
        let summary = log.summary();
        assert!(summary.contains("3 entries"));
        assert!(summary.contains("1 info"));
        assert!(summary.contains("1 warning"));
        assert!(summary.contains("1 error"));
        assert!(summary.contains("2 certificates"));
    }

    #[test]
    fn test_audit_log_digest_deterministic() {
        let mut log = CertificateAuditLog::new();
        log.log(100, "cert-1", "created", AuditSeverity::Info, "details");
        let d1 = log.digest();
        let d2 = log.digest();
        assert_eq!(d1, d2);
    }

    #[test]
    fn test_audit_log_digest_changes_on_append() {
        let mut log = CertificateAuditLog::new();
        log.log(100, "cert-1", "created", AuditSeverity::Info, "");
        let d1 = log.digest();
        log.log(200, "cert-1", "verified", AuditSeverity::Info, "");
        let d2 = log.digest();
        assert_ne!(d1, d2);
    }

    #[test]
    fn test_timeline_event_type_display() {
        assert_eq!(TimelineEventType::Created.to_string(), "Created");
        assert_eq!(TimelineEventType::Verified(true).to_string(), "Verified(true)");
        assert_eq!(TimelineEventType::Verified(false).to_string(), "Verified(false)");
        assert_eq!(TimelineEventType::Revoked.to_string(), "Revoked");
    }

    #[test]
    fn test_timeline_active_at_boundary() {
        let mut tl = CertificateTimeline::new();
        tl.record_creation("c1", 100);
        tl.record_revocation("c1", 100);
        // Created and revoked at same timestamp => revoked wins
        let active = tl.active_at(100);
        assert!(active.is_empty());
    }

}
