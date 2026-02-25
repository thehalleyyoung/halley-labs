//! Certificate generation module for CABER.
//!
//! Provides coalgebraic behavioral certificate generation, signing,
//! compression, and validation for LLM behavioral audits.

use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can occur during certificate generation.
#[derive(Debug, Clone)]
pub enum CertificateError {
    InvalidInput(String),
    SigningError(String),
    SerializationError(String),
    SizeExceeded { actual: usize, max: usize },
    ExpiredCertificate,
    InvalidPACBounds(String),
}

impl fmt::Display for CertificateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CertificateError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            CertificateError::SigningError(msg) => write!(f, "Signing error: {}", msg),
            CertificateError::SerializationError(msg) => {
                write!(f, "Serialization error: {}", msg)
            }
            CertificateError::SizeExceeded { actual, max } => {
                write!(
                    f,
                    "Certificate size {} exceeds maximum {}",
                    actual, max
                )
            }
            CertificateError::ExpiredCertificate => write!(f, "Certificate has expired"),
            CertificateError::InvalidPACBounds(msg) => {
                write!(f, "Invalid PAC bounds: {}", msg)
            }
        }
    }
}

impl std::error::Error for CertificateError {}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the certificate generator.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct GeneratorConfig {
    /// HMAC signing key (hex-encoded or arbitrary string).
    pub signing_key: String,
    /// How many hours a certificate remains valid (default: 720 = 30 days).
    pub validity_duration_hours: u64,
    /// Whether to enable compression when producing compressed certificates.
    pub compression_enabled: bool,
    /// Whether to include full automaton data in the certificate.
    pub include_automaton: bool,
    /// Whether to include witness strings in property results.
    pub include_witnesses: bool,
    /// Maximum allowed serialized certificate size in bytes.
    pub max_certificate_size_bytes: usize,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            signing_key: String::from("default-caber-signing-key"),
            validity_duration_hours: 720, // 30 days
            compression_enabled: true,
            include_automaton: true,
            include_witnesses: true,
            max_certificate_size_bytes: 10 * 1024 * 1024, // 10 MiB
        }
    }
}

// ---------------------------------------------------------------------------
// Input types
// ---------------------------------------------------------------------------

/// Serialized representation of a learned automaton.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AutomatonData {
    pub num_states: usize,
    pub num_transitions: usize,
    pub alphabet_size: usize,
    /// JSON representation of the transition function.
    pub serialized_transitions: String,
    /// Human-readable labels for each state.
    pub state_labels: Vec<String>,
}

/// Result of checking a single temporal/behavioral property.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PropertyResult {
    pub property_name: String,
    pub property_description: String,
    pub satisfied: bool,
    /// Degree of satisfaction in [0, 1].
    pub satisfaction_degree: f64,
    /// Optional witness trace demonstrating satisfaction or violation.
    pub witness: Option<String>,
    /// ISO-8601 timestamp when the check was performed.
    pub checked_at: String,
}

/// Result of computing a bisimulation distance between two models.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DistanceResult {
    pub model_a: String,
    pub model_b: String,
    pub distance: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    /// Method used to compute the distance (e.g. "on-the-fly", "partition").
    pub method: String,
}

/// PAC (Probably Approximately Correct) learning bounds.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PACBounds {
    /// Accuracy parameter – upper bound on approximation error.
    pub epsilon: f64,
    /// Confidence parameter – probability of failure.
    pub delta: f64,
    /// Number of samples used.
    pub sample_complexity: usize,
    /// Confidence level = 1 - delta.
    pub confidence: f64,
}

impl PACBounds {
    /// Returns `true` when the bounds are mathematically valid.
    pub fn is_valid(&self) -> bool {
        self.epsilon > 0.0
            && self.epsilon <= 1.0
            && self.delta > 0.0
            && self.delta <= 1.0
            && self.sample_complexity > 0
            && self.confidence > 0.0
            && self.confidence <= 1.0
            && (self.confidence - (1.0 - self.delta)).abs() < 1e-9
    }

    /// Combines two independent PAC bounds via a union bound.
    pub fn combined_with(&self, other: &PACBounds) -> PACBounds {
        let combined_epsilon = self.epsilon + other.epsilon;
        let combined_delta = self.delta + other.delta - self.delta * other.delta;
        let combined_samples = self.sample_complexity + other.sample_complexity;
        let combined_confidence = 1.0 - combined_delta;
        PACBounds {
            epsilon: combined_epsilon.min(1.0),
            delta: combined_delta.min(1.0),
            sample_complexity: combined_samples,
            confidence: combined_confidence.max(0.0),
        }
    }
}

/// Metadata about the audit session.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AuditMetadata {
    pub auditor_id: String,
    /// ISO-8601 timestamp.
    pub audit_start: String,
    /// ISO-8601 timestamp, `None` if still in progress.
    pub audit_end: Option<String>,
    pub model_version: String,
    pub framework_version: String,
    pub notes: Vec<String>,
}

/// Everything needed to generate a certificate.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CertificateInput {
    pub model_id: String,
    pub automaton_data: AutomatonData,
    pub model_check_results: Vec<PropertyResult>,
    pub bisimulation_distances: Vec<DistanceResult>,
    pub pac_bounds: PACBounds,
    pub audit_metadata: AuditMetadata,
    pub query_budget_used: usize,
    pub total_queries_available: usize,
}

// ---------------------------------------------------------------------------
// Certificate types
// ---------------------------------------------------------------------------

/// Compact summary of the learned automaton stored inside a certificate.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AutomatonSummary {
    pub num_states: usize,
    pub num_transitions: usize,
    pub alphabet_size: usize,
    pub state_labels: Vec<String>,
}

/// Composed error bound combining sampling, learning, and abstraction errors.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ComposedError {
    pub epsilon_sample: f64,
    pub delta_learn: f64,
    pub epsilon_abstraction: f64,
    pub total_epsilon: f64,
    pub total_delta: f64,
    pub description: String,
}

/// Cryptographic signature attached to a certificate.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CertificateSignature {
    pub algorithm: String,
    pub hash: String,
    pub signed_at: String,
}

impl CertificateSignature {
    /// Verify the signature against `data` and `key`.
    ///
    /// Re-computes the HMAC-like hash and compares with the stored hash.
    pub fn verify(&self, data: &str, key: &str) -> bool {
        let expected = hmac_hash(data, key);
        constant_time_eq(&self.hash, &expected)
    }
}

/// A compressed representation of a certificate.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CompressedCertificate {
    pub original_size: usize,
    pub compressed_data: String,
    pub compression_ratio: f64,
}

/// The main behavioral certificate issued by CABER.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BehavioralCertificate {
    pub id: String,
    pub model_id: String,
    /// ISO-8601 issued timestamp.
    pub issued_at: String,
    /// ISO-8601 expiry timestamp.
    pub valid_until: String,
    pub automaton_summary: AutomatonSummary,
    pub property_results: Vec<PropertyResult>,
    pub distance_results: Vec<DistanceResult>,
    pub pac_bounds: PACBounds,
    pub composed_error: ComposedError,
    pub metadata: AuditMetadata,
    pub signature: Option<CertificateSignature>,
    pub query_budget_used: usize,
}

impl BehavioralCertificate {
    /// Check whether the certificate is valid at the given ISO-8601 timestamp.
    ///
    /// Performs a lexicographic comparison on the ISO-8601 strings which works
    /// correctly for timestamps in the same timezone (UTC recommended).
    pub fn is_valid_at(&self, timestamp: &str) -> bool {
        self.issued_at.as_str() <= timestamp && timestamp <= self.valid_until.as_str()
    }

    /// Produce a human-readable one-line summary.
    pub fn summary(&self) -> String {
        let props_satisfied = self
            .property_results
            .iter()
            .filter(|p| p.satisfied)
            .count();
        let props_total = self.property_results.len();
        format!(
            "Certificate {} for model '{}': {}/{} properties satisfied, \
             ε={:.4}, δ={:.4}, {} states, issued {}",
            self.id,
            self.model_id,
            props_satisfied,
            props_total,
            self.composed_error.total_epsilon,
            self.composed_error.total_delta,
            self.automaton_summary.num_states,
            self.issued_at,
        )
    }

    /// Returns `true` when every property in the certificate is satisfied.
    pub fn all_properties_satisfied(&self) -> bool {
        self.property_results.iter().all(|p| p.satisfied)
    }
}

// ---------------------------------------------------------------------------
// HMAC-like hash helper
// ---------------------------------------------------------------------------

/// Compute a deterministic HMAC-like hash of `data` using `key`.
///
/// Uses a simple but deterministic scheme:
///   1. Derive inner-key and outer-key by XOR-ing key bytes with fixed pads.
///   2. Hash inner_key || data  →  inner_hash
///   3. Hash outer_key || inner_hash  →  final hash
///
/// The underlying hash is a 64-bit SipHash-like round function iterated over
/// the input, producing a 128-bit (16-byte) hex digest.
pub fn hmac_hash(data: &str, key: &str) -> String {
    // Derive a fixed-length key block (64 bytes).
    let key_block = derive_key_block(key);

    // Inner pad = key_block XOR 0x36
    let inner_pad: Vec<u8> = key_block.iter().map(|b| b ^ 0x36).collect();
    // Outer pad = key_block XOR 0x5c
    let outer_pad: Vec<u8> = key_block.iter().map(|b| b ^ 0x5c).collect();

    // inner_hash = H(inner_pad || data)
    let mut inner_input = inner_pad;
    inner_input.extend_from_slice(data.as_bytes());
    let inner_hash = sip_hash_bytes(&inner_input);

    // outer_hash = H(outer_pad || inner_hash)
    let mut outer_input = outer_pad;
    outer_input.extend_from_slice(&inner_hash);
    let outer_hash = sip_hash_bytes(&outer_input);

    bytes_to_hex(&outer_hash)
}

/// Derive a 64-byte key block from an arbitrary-length key.
fn derive_key_block(key: &str) -> Vec<u8> {
    let key_bytes = key.as_bytes();
    let mut block = vec![0u8; 64];
    if key_bytes.len() <= 64 {
        block[..key_bytes.len()].copy_from_slice(key_bytes);
    } else {
        // Hash the key first to fit it into 64 bytes.
        let hashed = sip_hash_bytes(key_bytes);
        block[..hashed.len()].copy_from_slice(&hashed);
    }
    block
}

/// A SipHash-inspired mixing function producing a 16-byte digest.
///
/// Not cryptographically secure – suitable for integrity checking in an
/// audit pipeline where the signing key is kept secret.
fn sip_hash_bytes(data: &[u8]) -> Vec<u8> {
    let mut v0: u64 = 0x736f6d6570736575;
    let mut v1: u64 = 0x646f72616e646f6d;
    let mut v2: u64 = 0x6c7967656e657261;
    let mut v3: u64 = 0x7465646279746573;

    let length = data.len() as u64;

    // Process 8-byte blocks.
    let blocks = data.len() / 8;
    for i in 0..blocks {
        let start = i * 8;
        let mut m: u64 = 0;
        for j in 0..8 {
            m |= (data[start + j] as u64) << (j * 8);
        }
        v3 ^= m;
        sip_round(&mut v0, &mut v1, &mut v2, &mut v3);
        sip_round(&mut v0, &mut v1, &mut v2, &mut v3);
        v0 ^= m;
    }

    // Process remaining bytes with length encoding.
    let mut last: u64 = length << 56;
    let remaining = data.len() % 8;
    let tail_start = blocks * 8;
    for i in 0..remaining {
        last |= (data[tail_start + i] as u64) << (i * 8);
    }
    v3 ^= last;
    sip_round(&mut v0, &mut v1, &mut v2, &mut v3);
    sip_round(&mut v0, &mut v1, &mut v2, &mut v3);
    v0 ^= last;

    // Finalization.
    v2 ^= 0xff;
    sip_round(&mut v0, &mut v1, &mut v2, &mut v3);
    sip_round(&mut v0, &mut v1, &mut v2, &mut v3);
    sip_round(&mut v0, &mut v1, &mut v2, &mut v3);
    sip_round(&mut v0, &mut v1, &mut v2, &mut v3);

    let h1 = v0 ^ v1 ^ v2 ^ v3;

    v1 ^= 0xee;
    sip_round(&mut v0, &mut v1, &mut v2, &mut v3);
    sip_round(&mut v0, &mut v1, &mut v2, &mut v3);
    sip_round(&mut v0, &mut v1, &mut v2, &mut v3);
    sip_round(&mut v0, &mut v1, &mut v2, &mut v3);

    let h2 = v0 ^ v1 ^ v2 ^ v3;

    let mut out = Vec::with_capacity(16);
    out.extend_from_slice(&h1.to_le_bytes());
    out.extend_from_slice(&h2.to_le_bytes());
    out
}

/// One SipHash compression round.
#[inline]
fn sip_round(v0: &mut u64, v1: &mut u64, v2: &mut u64, v3: &mut u64) {
    *v0 = v0.wrapping_add(*v1);
    *v1 = v1.rotate_left(13);
    *v1 ^= *v0;
    *v0 = v0.rotate_left(32);
    *v2 = v2.wrapping_add(*v3);
    *v3 = v3.rotate_left(16);
    *v3 ^= *v2;
    *v0 = v0.wrapping_add(*v3);
    *v3 = v3.rotate_left(21);
    *v3 ^= *v0;
    *v2 = v2.wrapping_add(*v1);
    *v1 = v1.rotate_left(17);
    *v1 ^= *v2;
    *v2 = v2.rotate_left(32);
}

/// Convert a byte slice to a lowercase hex string.
fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// Constant-time equality comparison for two hex strings.
fn constant_time_eq(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.bytes().zip(b.bytes()) {
        diff |= x ^ y;
    }
    diff == 0
}

// ---------------------------------------------------------------------------
// Timestamp helpers
// ---------------------------------------------------------------------------

/// Return the current UTC time as an ISO-8601 string.
fn now_iso8601() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format_epoch_secs(now.as_secs())
}

/// Format epoch seconds as `YYYY-MM-DDTHH:MM:SSZ`.
fn format_epoch_secs(epoch: u64) -> String {
    let secs_per_minute: u64 = 60;
    let secs_per_hour: u64 = 3600;
    let secs_per_day: u64 = 86400;

    let days = epoch / secs_per_day;
    let remaining = epoch % secs_per_day;
    let hours = remaining / secs_per_hour;
    let minutes = (remaining % secs_per_hour) / secs_per_minute;
    let seconds = remaining % secs_per_minute;

    let (year, month, day) = days_to_ymd(days);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hours, minutes, seconds
    )
}

/// Convert days since Unix epoch to (year, month, day).
fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Algorithm from Howard Hinnant's civil_from_days.
    let z = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64; // day of era [0, 146096]
    let yoe =
        (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // year of era [0, 399]
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // day of year [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };
    (y as u64, m, d)
}

/// Add `hours` to an ISO-8601 timestamp string. Falls back to string if
/// parsing fails, adding `hours * 3600` to the epoch derived from the input.
fn add_hours_to_iso(iso: &str, hours: u64) -> String {
    if let Some(epoch) = iso8601_to_epoch(iso) {
        format_epoch_secs(epoch + hours * 3600)
    } else {
        // Fallback: just append hours info (should not happen with well-formed input).
        format!("{}+{}h", iso, hours)
    }
}

/// Parse a subset of ISO-8601 (`YYYY-MM-DDTHH:MM:SSZ`) into epoch seconds.
fn iso8601_to_epoch(s: &str) -> Option<u64> {
    // Expect format: YYYY-MM-DDTHH:MM:SSZ (20 chars).
    let s = s.trim();
    if s.len() < 19 {
        return None;
    }
    let year: u64 = s.get(0..4)?.parse().ok()?;
    let month: u64 = s.get(5..7)?.parse().ok()?;
    let day: u64 = s.get(8..10)?.parse().ok()?;
    let hour: u64 = s.get(11..13)?.parse().ok()?;
    let minute: u64 = s.get(14..16)?.parse().ok()?;
    let second: u64 = s.get(17..19)?.parse().ok()?;

    // Convert to days since epoch using inverse of days_to_ymd.
    let days = ymd_to_days(year, month, day)?;
    Some(days * 86400 + hour * 3600 + minute * 60 + second)
}

/// Convert (year, month, day) to days since Unix epoch.
fn ymd_to_days(year: u64, month: u64, day: u64) -> Option<u64> {
    if month < 1 || month > 12 || day < 1 || day > 31 {
        return None;
    }
    // Howard Hinnant's days_from_civil.
    let y = if month <= 2 { year as i64 - 1 } else { year as i64 };
    let m = if month <= 2 { month + 9 } else { month - 3 };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u64;
    let doy = (153 * m + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = (era as i64) * 146097 + doe as i64 - 719468;
    if days < 0 {
        None
    } else {
        Some(days as u64)
    }
}

// ---------------------------------------------------------------------------
// UUID helper
// ---------------------------------------------------------------------------

/// Generate a v4-like UUID using basic randomness from the standard library.
fn generate_uuid() -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};

    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    let h1 = hasher.finish();

    // Second hash with a different seed.
    let mut hasher2 = DefaultHasher::new();
    (seed.wrapping_mul(6364136223846793005).wrapping_add(1)).hash(&mut hasher2);
    let h2 = hasher2.finish();

    let b1 = h1.to_le_bytes();
    let b2 = h2.to_le_bytes();
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&b1);
    bytes[8..].copy_from_slice(&b2);

    // Set version 4.
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    // Set variant 1.
    bytes[8] = (bytes[8] & 0x3f) | 0x80;

    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5],
        bytes[6], bytes[7],
        bytes[8], bytes[9],
        bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
    )
}

// ---------------------------------------------------------------------------
// CertificateGenerator
// ---------------------------------------------------------------------------

/// The main certificate generator.
///
/// Holds configuration and a log of all certificates produced during its
/// lifetime.
pub struct CertificateGenerator {
    config: GeneratorConfig,
    certificates: Vec<BehavioralCertificate>,
}

impl CertificateGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: GeneratorConfig) -> Self {
        Self {
            config,
            certificates: Vec::new(),
        }
    }

    /// Generate a behavioral certificate from the provided input.
    ///
    /// Steps:
    /// 1. Validate the input.
    /// 2. Compute the composed error bound.
    /// 3. Build the certificate struct.
    /// 4. Optionally strip witnesses based on config.
    /// 5. Sign the certificate.
    /// 6. Check serialized size against the configured maximum.
    /// 7. Store and return the certificate.
    pub fn generate_certificate(
        &mut self,
        input: CertificateInput,
    ) -> Result<BehavioralCertificate, CertificateError> {
        // 1. Validate.
        self.validate_input(&input)?;

        // 2. Compose error bounds.
        let composed_error = Self::compose_errors(
            input.pac_bounds.epsilon,
            input.pac_bounds.delta,
            self.estimate_abstraction_error(&input),
        );

        // 3. Build certificate.
        let issued_at = now_iso8601();
        let valid_until =
            add_hours_to_iso(&issued_at, self.config.validity_duration_hours);

        let automaton_summary = AutomatonSummary {
            num_states: input.automaton_data.num_states,
            num_transitions: input.automaton_data.num_transitions,
            alphabet_size: input.automaton_data.alphabet_size,
            state_labels: if self.config.include_automaton {
                input.automaton_data.state_labels.clone()
            } else {
                Vec::new()
            },
        };

        let property_results = if self.config.include_witnesses {
            input.model_check_results.clone()
        } else {
            input
                .model_check_results
                .iter()
                .map(|p| PropertyResult {
                    witness: None,
                    ..p.clone()
                })
                .collect()
        };

        let mut cert = BehavioralCertificate {
            id: generate_uuid(),
            model_id: input.model_id.clone(),
            issued_at,
            valid_until,
            automaton_summary,
            property_results,
            distance_results: input.bisimulation_distances.clone(),
            pac_bounds: input.pac_bounds.clone(),
            composed_error,
            metadata: input.audit_metadata.clone(),
            signature: None,
            query_budget_used: input.query_budget_used,
        };

        // 4. Sign.
        let signature = self.sign_certificate(&cert);
        cert.signature = Some(signature);

        // 5. Size check.
        let serialized = serde_json::to_string(&cert)
            .map_err(|e| CertificateError::SerializationError(e.to_string()))?;
        if serialized.len() > self.config.max_certificate_size_bytes {
            return Err(CertificateError::SizeExceeded {
                actual: serialized.len(),
                max: self.config.max_certificate_size_bytes,
            });
        }

        // 6. Store.
        self.certificates.push(cert.clone());

        Ok(cert)
    }

    /// Compose three independent error sources via a union bound.
    ///
    /// - `epsilon_sample`: error from finite sampling.
    /// - `delta_learn`: probability that the learning algorithm fails.
    /// - `epsilon_abstraction`: error introduced by the abstraction functor.
    ///
    /// The total epsilon is bounded by the sum of the two epsilon terms
    /// (triangle inequality on the behavioural metric), and the total delta
    /// equals `delta_learn` (the only probabilistic term here).
    pub fn compose_errors(
        epsilon_sample: f64,
        delta_learn: f64,
        epsilon_abstraction: f64,
    ) -> ComposedError {
        let total_epsilon = epsilon_sample + epsilon_abstraction;
        let total_delta = delta_learn;

        let description = format!(
            "Union bound composition: ε_total = ε_sample({:.6}) + ε_abstraction({:.6}) = {:.6}; \
             δ_total = δ_learn({:.6})",
            epsilon_sample, epsilon_abstraction, total_epsilon, delta_learn,
        );

        ComposedError {
            epsilon_sample,
            delta_learn,
            epsilon_abstraction,
            total_epsilon,
            total_delta,
            description,
        }
    }

    /// Produce an HMAC-like signature over the certificate data.
    pub fn sign_certificate(&self, cert: &BehavioralCertificate) -> CertificateSignature {
        let data = self.certificate_signing_payload(cert);
        let hash = hmac_hash(&data, &self.config.signing_key);
        CertificateSignature {
            algorithm: "CABER-HMAC-SipHash128".to_string(),
            hash,
            signed_at: now_iso8601(),
        }
    }

    /// Compress a certificate to JSON and compute size statistics.
    pub fn compress_certificate(
        &self,
        cert: &BehavioralCertificate,
    ) -> CompressedCertificate {
        // Serialize to compact JSON.
        let json = serde_json::to_string(cert).unwrap_or_default();
        let original_size = json.len();

        // Apply a simple run-length encoding on top of the JSON to achieve
        // some compression.  For real deployments this would be zstd/gzip.
        let compressed = rle_compress(&json);

        let compressed_size = compressed.len();
        let compression_ratio = if original_size > 0 {
            compressed_size as f64 / original_size as f64
        } else {
            1.0
        };

        CompressedCertificate {
            original_size,
            compressed_data: compressed,
            compression_ratio,
        }
    }

    /// Validate that the certificate input is well-formed.
    pub fn validate_input(
        &self,
        input: &CertificateInput,
    ) -> Result<(), CertificateError> {
        if input.model_id.is_empty() {
            return Err(CertificateError::InvalidInput(
                "model_id must not be empty".into(),
            ));
        }

        if input.automaton_data.num_states == 0 {
            return Err(CertificateError::InvalidInput(
                "automaton must have at least one state".into(),
            ));
        }

        if input.automaton_data.alphabet_size == 0 {
            return Err(CertificateError::InvalidInput(
                "alphabet must have at least one symbol".into(),
            ));
        }

        if !input.pac_bounds.is_valid() {
            return Err(CertificateError::InvalidPACBounds(format!(
                "ε={}, δ={}, confidence={}, samples={}",
                input.pac_bounds.epsilon,
                input.pac_bounds.delta,
                input.pac_bounds.confidence,
                input.pac_bounds.sample_complexity,
            )));
        }

        if input.query_budget_used > input.total_queries_available {
            return Err(CertificateError::InvalidInput(format!(
                "query_budget_used ({}) exceeds total_queries_available ({})",
                input.query_budget_used, input.total_queries_available,
            )));
        }

        if input.audit_metadata.auditor_id.is_empty() {
            return Err(CertificateError::InvalidInput(
                "auditor_id must not be empty".into(),
            ));
        }

        if input.audit_metadata.audit_start.is_empty() {
            return Err(CertificateError::InvalidInput(
                "audit_start must not be empty".into(),
            ));
        }

        // Validate property satisfaction degrees.
        for p in &input.model_check_results {
            if p.satisfaction_degree < 0.0 || p.satisfaction_degree > 1.0 {
                return Err(CertificateError::InvalidInput(format!(
                    "satisfaction_degree for '{}' is {}, must be in [0, 1]",
                    p.property_name, p.satisfaction_degree,
                )));
            }
        }

        // Validate distance bounds consistency.
        for d in &input.bisimulation_distances {
            if d.lower_bound > d.distance || d.distance > d.upper_bound {
                return Err(CertificateError::InvalidInput(format!(
                    "distance bounds inconsistent for ({}, {}): lb={} d={} ub={}",
                    d.model_a, d.model_b, d.lower_bound, d.distance, d.upper_bound,
                )));
            }
        }

        Ok(())
    }

    /// Return a reference to all certificates generated so far.
    pub fn certificates(&self) -> &[BehavioralCertificate] {
        &self.certificates
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// Build the canonical string payload used for signing.
    fn certificate_signing_payload(&self, cert: &BehavioralCertificate) -> String {
        // We deliberately exclude the signature field itself to avoid
        // circular dependency.
        format!(
            "{}|{}|{}|{}|{}|{}|{:.8}|{:.8}|{:.8}|{:.8}|{}",
            cert.id,
            cert.model_id,
            cert.issued_at,
            cert.valid_until,
            cert.automaton_summary.num_states,
            cert.automaton_summary.num_transitions,
            cert.pac_bounds.epsilon,
            cert.pac_bounds.delta,
            cert.composed_error.total_epsilon,
            cert.composed_error.total_delta,
            cert.query_budget_used,
        )
    }

    /// Estimate the abstraction error from the input data.
    ///
    /// Uses the maximum bisimulation distance upper bound as a conservative
    /// estimate of the abstraction error induced by the coalgebraic functor.
    fn estimate_abstraction_error(&self, input: &CertificateInput) -> f64 {
        input
            .bisimulation_distances
            .iter()
            .map(|d| d.upper_bound)
            .fold(0.0_f64, f64::max)
    }
}

// ---------------------------------------------------------------------------
// Run-length encoding (simple compression)
// ---------------------------------------------------------------------------

/// A simple run-length encoder for repeated characters in JSON output.
fn rle_compress(input: &str) -> String {
    if input.is_empty() {
        return String::new();
    }

    let mut result = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        let ch = bytes[i];
        let mut count: usize = 1;
        while i + count < bytes.len() && bytes[i + count] == ch && count < 255 {
            count += 1;
        }
        if count >= 4 {
            // Encode as <ESC><count><char>.
            result.push('\x1b');
            result.push(count as u8 as char);
            result.push(ch as char);
        } else {
            for _ in 0..count {
                result.push(ch as char);
            }
        }
        i += count;
    }
    result
}

/// Decompress an RLE-compressed string.
fn rle_decompress(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut result = String::with_capacity(bytes.len() * 2);
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i] == 0x1b && i + 2 < bytes.len() {
            let count = bytes[i + 1] as usize;
            let ch = bytes[i + 2] as char;
            for _ in 0..count {
                result.push(ch);
            }
            i += 3;
        } else {
            result.push(bytes[i] as char);
            i += 1;
        }
    }
    result
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn sample_automaton_data() -> AutomatonData {
        AutomatonData {
            num_states: 5,
            num_transitions: 12,
            alphabet_size: 3,
            serialized_transitions: r#"{"q0":{"a":"q1","b":"q2"}}"#.to_string(),
            state_labels: vec![
                "q0".into(),
                "q1".into(),
                "q2".into(),
                "q3".into(),
                "q4".into(),
            ],
        }
    }

    fn sample_property_results() -> Vec<PropertyResult> {
        vec![
            PropertyResult {
                property_name: "safety".into(),
                property_description: "No unsafe states reachable".into(),
                satisfied: true,
                satisfaction_degree: 1.0,
                witness: None,
                checked_at: "2025-01-15T10:00:00Z".into(),
            },
            PropertyResult {
                property_name: "liveness".into(),
                property_description: "Eventually reaches accepting state".into(),
                satisfied: true,
                satisfaction_degree: 0.95,
                witness: Some("q0 -> q1 -> q3".into()),
                checked_at: "2025-01-15T10:01:00Z".into(),
            },
        ]
    }

    fn sample_distance_results() -> Vec<DistanceResult> {
        vec![DistanceResult {
            model_a: "model-v1".into(),
            model_b: "model-v2".into(),
            distance: 0.05,
            lower_bound: 0.03,
            upper_bound: 0.08,
            method: "on-the-fly".into(),
        }]
    }

    fn sample_pac_bounds() -> PACBounds {
        PACBounds {
            epsilon: 0.05,
            delta: 0.01,
            sample_complexity: 10000,
            confidence: 0.99,
        }
    }

    fn sample_audit_metadata() -> AuditMetadata {
        AuditMetadata {
            auditor_id: "auditor-42".into(),
            audit_start: "2025-01-15T09:00:00Z".into(),
            audit_end: Some("2025-01-15T11:00:00Z".into()),
            model_version: "gpt-4o-2025".into(),
            framework_version: "caber-0.1.0".into(),
            notes: vec!["Initial audit".into()],
        }
    }

    fn sample_input() -> CertificateInput {
        CertificateInput {
            model_id: "test-model-1".into(),
            automaton_data: sample_automaton_data(),
            model_check_results: sample_property_results(),
            bisimulation_distances: sample_distance_results(),
            pac_bounds: sample_pac_bounds(),
            audit_metadata: sample_audit_metadata(),
            query_budget_used: 5000,
            total_queries_available: 10000,
        }
    }

    fn default_generator() -> CertificateGenerator {
        CertificateGenerator::new(GeneratorConfig::default())
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_generate_certificate_success() {
        let mut gen = default_generator();
        let input = sample_input();
        let cert = gen.generate_certificate(input).unwrap();
        assert!(!cert.id.is_empty());
        assert_eq!(cert.model_id, "test-model-1");
        assert!(cert.signature.is_some());
        assert_eq!(gen.certificates().len(), 1);
    }

    #[test]
    fn test_validate_input_empty_model_id() {
        let gen = default_generator();
        let mut input = sample_input();
        input.model_id = String::new();
        let err = gen.validate_input(&input).unwrap_err();
        match err {
            CertificateError::InvalidInput(msg) => {
                assert!(msg.contains("model_id"));
            }
            _ => panic!("Expected InvalidInput"),
        }
    }

    #[test]
    fn test_validate_input_zero_states() {
        let gen = default_generator();
        let mut input = sample_input();
        input.automaton_data.num_states = 0;
        let err = gen.validate_input(&input).unwrap_err();
        match err {
            CertificateError::InvalidInput(msg) => {
                assert!(msg.contains("state"));
            }
            _ => panic!("Expected InvalidInput"),
        }
    }

    #[test]
    fn test_validate_input_budget_exceeded() {
        let gen = default_generator();
        let mut input = sample_input();
        input.query_budget_used = 20000;
        input.total_queries_available = 10000;
        let err = gen.validate_input(&input).unwrap_err();
        match err {
            CertificateError::InvalidInput(msg) => {
                assert!(msg.contains("query_budget_used"));
            }
            _ => panic!("Expected InvalidInput"),
        }
    }

    #[test]
    fn test_validate_input_invalid_pac_bounds() {
        let gen = default_generator();
        let mut input = sample_input();
        input.pac_bounds.epsilon = -0.1;
        let err = gen.validate_input(&input).unwrap_err();
        match err {
            CertificateError::InvalidPACBounds(_) => {}
            _ => panic!("Expected InvalidPACBounds"),
        }
    }

    #[test]
    fn test_compose_errors() {
        let err = CertificateGenerator::compose_errors(0.05, 0.01, 0.03);
        assert!((err.total_epsilon - 0.08).abs() < 1e-12);
        assert!((err.total_delta - 0.01).abs() < 1e-12);
        assert_eq!(err.epsilon_sample, 0.05);
        assert_eq!(err.epsilon_abstraction, 0.03);
    }

    #[test]
    fn test_sign_and_verify() {
        let gen = default_generator();
        let mut gen2 = default_generator();
        let input = sample_input();
        let cert = gen2.generate_certificate(input).unwrap();

        let payload = gen.certificate_signing_payload(&cert);
        let sig = cert.signature.as_ref().unwrap();
        assert!(sig.verify(&payload, &gen.config.signing_key));
        assert!(!sig.verify(&payload, "wrong-key"));
        assert!(!sig.verify("tampered-data", &gen.config.signing_key));
    }

    #[test]
    fn test_compress_certificate() {
        let mut gen = default_generator();
        let input = sample_input();
        let cert = gen.generate_certificate(input).unwrap();
        let compressed = gen.compress_certificate(&cert);
        assert!(compressed.original_size > 0);
        assert!(compressed.compression_ratio > 0.0);
        assert!(compressed.compression_ratio <= 1.5);
    }

    #[test]
    fn test_certificate_is_valid_at() {
        let mut gen = default_generator();
        let input = sample_input();
        let cert = gen.generate_certificate(input).unwrap();
        // The cert should be valid at its issued_at time.
        assert!(cert.is_valid_at(&cert.issued_at));
        // Should be valid at its expiry time.
        assert!(cert.is_valid_at(&cert.valid_until));
        // Should not be valid in the distant past.
        assert!(!cert.is_valid_at("2000-01-01T00:00:00Z"));
    }

    #[test]
    fn test_all_properties_satisfied() {
        let mut gen = default_generator();
        let input = sample_input();
        let cert = gen.generate_certificate(input).unwrap();
        assert!(cert.all_properties_satisfied());
    }

    #[test]
    fn test_certificate_summary_format() {
        let mut gen = default_generator();
        let input = sample_input();
        let cert = gen.generate_certificate(input).unwrap();
        let summary = cert.summary();
        assert!(summary.contains("test-model-1"));
        assert!(summary.contains("2/2 properties satisfied"));
    }

    #[test]
    fn test_pac_bounds_combined() {
        let b1 = PACBounds {
            epsilon: 0.05,
            delta: 0.01,
            sample_complexity: 1000,
            confidence: 0.99,
        };
        let b2 = PACBounds {
            epsilon: 0.03,
            delta: 0.02,
            sample_complexity: 2000,
            confidence: 0.98,
        };
        let combined = b1.combined_with(&b2);
        assert!((combined.epsilon - 0.08).abs() < 1e-12);
        // delta combined = 0.01 + 0.02 - 0.01*0.02 = 0.0298
        assert!((combined.delta - 0.0298).abs() < 1e-12);
        assert_eq!(combined.sample_complexity, 3000);
        assert!((combined.confidence - (1.0 - 0.0298)).abs() < 1e-12);
    }

    #[test]
    fn test_hmac_hash_deterministic() {
        let h1 = hmac_hash("hello world", "secret");
        let h2 = hmac_hash("hello world", "secret");
        assert_eq!(h1, h2);
        // Different data should yield a different hash.
        let h3 = hmac_hash("hello world!", "secret");
        assert_ne!(h1, h3);
        // Different key should yield a different hash.
        let h4 = hmac_hash("hello world", "other-secret");
        assert_ne!(h1, h4);
        // Hash should be a 32-char hex string (128 bits).
        assert_eq!(h1.len(), 32);
    }

    #[test]
    fn test_size_exceeded_error() {
        let config = GeneratorConfig {
            max_certificate_size_bytes: 10, // impossibly small
            ..GeneratorConfig::default()
        };
        let mut gen = CertificateGenerator::new(config);
        let input = sample_input();
        let err = gen.generate_certificate(input).unwrap_err();
        match err {
            CertificateError::SizeExceeded { actual, max } => {
                assert!(actual > 10);
                assert_eq!(max, 10);
            }
            _ => panic!("Expected SizeExceeded"),
        }
    }

    #[test]
    fn test_rle_roundtrip() {
        let original = "aaaaaabbbbccddddddddeeee";
        let compressed = rle_compress(original);
        let decompressed = rle_decompress(&compressed);
        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_timestamp_roundtrip() {
        let epoch = 1736899200; // 2025-01-15T00:00:00Z
        let iso = format_epoch_secs(epoch);
        assert_eq!(iso, "2025-01-15T00:00:00Z");
        let back = iso8601_to_epoch(&iso).unwrap();
        assert_eq!(back, epoch);
    }

    #[test]
    fn test_add_hours() {
        let base = "2025-01-15T00:00:00Z";
        let result = add_hours_to_iso(base, 720);
        assert_eq!(result, "2025-02-14T00:00:00Z");
    }

    #[test]
    fn test_witnesses_excluded_when_config_disabled() {
        let config = GeneratorConfig {
            include_witnesses: false,
            ..GeneratorConfig::default()
        };
        let mut gen = CertificateGenerator::new(config);
        let input = sample_input();
        let cert = gen.generate_certificate(input).unwrap();
        for p in &cert.property_results {
            assert!(p.witness.is_none());
        }
    }

    #[test]
    fn test_invalid_distance_bounds() {
        let gen = default_generator();
        let mut input = sample_input();
        input.bisimulation_distances = vec![DistanceResult {
            model_a: "a".into(),
            model_b: "b".into(),
            distance: 0.5,
            lower_bound: 0.6, // lower > distance
            upper_bound: 0.8,
            method: "partition".into(),
        }];
        let err = gen.validate_input(&input).unwrap_err();
        match err {
            CertificateError::InvalidInput(msg) => {
                assert!(msg.contains("distance bounds inconsistent"));
            }
            _ => panic!("Expected InvalidInput"),
        }
    }

    #[test]
    fn test_multiple_certificates_stored() {
        let mut gen = default_generator();
        for i in 0..5 {
            let mut input = sample_input();
            input.model_id = format!("model-{}", i);
            gen.generate_certificate(input).unwrap();
        }
        assert_eq!(gen.certificates().len(), 5);
        for (i, cert) in gen.certificates().iter().enumerate() {
            assert_eq!(cert.model_id, format!("model-{}", i));
        }
    }
}
