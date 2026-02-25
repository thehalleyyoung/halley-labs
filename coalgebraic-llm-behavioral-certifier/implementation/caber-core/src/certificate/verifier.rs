// Certificate verification module for CABER.
// Provides full verification of behavioral certificates including integrity,
// expiry, PAC bounds, automaton consistency, property results, chain
// verification, and independent replication checks.

use std::collections::HashMap;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Serialization / deserialization helpers (minimal, self-contained)
// ---------------------------------------------------------------------------

/// Marker traits so the structs document their intent without pulling in serde.
/// Real serialization would use serde; here we keep the module dependency-free.
pub trait Serialize {}
pub trait Deserialize {}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct VerifierConfig {
    pub tolerance: f64,
    pub strict_mode: bool,
    pub check_expiry: bool,
    pub reference_time: Option<String>,
    pub replication_tolerance: f64,
}

impl Serialize for VerifierConfig {}
impl Deserialize for VerifierConfig {}

impl Default for VerifierConfig {
    fn default() -> Self {
        Self {
            tolerance: 0.01,
            strict_mode: false,
            check_expiry: true,
            reference_time: None,
            replication_tolerance: 0.05,
        }
    }
}

// ---------------------------------------------------------------------------
// Certificate data types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct VerifiableProperty {
    pub name: String,
    pub satisfied: bool,
    pub degree: f64,
    pub witness: Option<String>,
}

impl Serialize for VerifiableProperty {}
impl Deserialize for VerifiableProperty {}

#[derive(Clone, Debug)]
pub struct VerifiableDistance {
    pub model_a: String,
    pub model_b: String,
    pub distance: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
}

impl Serialize for VerifiableDistance {}
impl Deserialize for VerifiableDistance {}

#[derive(Clone, Debug)]
pub struct CertificateData {
    pub id: String,
    pub model_id: String,
    pub issued_at: String,
    pub valid_until: String,
    pub num_states: usize,
    pub num_transitions: usize,
    pub alphabet_size: usize,
    pub transition_data: String,
    pub property_results: Vec<VerifiableProperty>,
    pub distance_results: Vec<VerifiableDistance>,
    pub pac_epsilon: f64,
    pub pac_delta: f64,
    pub sample_complexity: usize,
    pub signature_algorithm: Option<String>,
    pub signature_hash: Option<String>,
    pub signed_at: Option<String>,
    pub metadata_json: String,
}

impl Serialize for CertificateData {}
impl Deserialize for CertificateData {}

// ---------------------------------------------------------------------------
// Verification result types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct IntegrityCheck {
    pub passed: bool,
    pub details: String,
    pub warning: Option<String>,
}

impl Serialize for IntegrityCheck {}
impl Deserialize for IntegrityCheck {}

#[derive(Clone, Debug)]
pub struct ExpiryCheck {
    pub passed: bool,
    pub details: String,
    pub warning: Option<String>,
}

impl Serialize for ExpiryCheck {}
impl Deserialize for ExpiryCheck {}

#[derive(Clone, Debug)]
pub struct PACBoundsCheck {
    pub passed: bool,
    pub details: String,
    pub warning: Option<String>,
}

impl Serialize for PACBoundsCheck {}
impl Deserialize for PACBoundsCheck {}

#[derive(Clone, Debug)]
pub struct AutomatonCheck {
    pub passed: bool,
    pub details: String,
    pub warning: Option<String>,
}

impl Serialize for AutomatonCheck {}
impl Deserialize for AutomatonCheck {}

#[derive(Clone, Debug)]
pub struct PropertyCheck {
    pub passed: bool,
    pub details: String,
    pub warning: Option<String>,
}

impl Serialize for PropertyCheck {}
impl Deserialize for PropertyCheck {}

#[derive(Clone, Debug)]
pub struct VerificationResult {
    pub overall_valid: bool,
    pub integrity: IntegrityCheck,
    pub expiry: ExpiryCheck,
    pub pac_bounds: PACBoundsCheck,
    pub automaton: AutomatonCheck,
    pub properties: PropertyCheck,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

impl Serialize for VerificationResult {}
impl Deserialize for VerificationResult {}

impl VerificationResult {
    /// Human-readable one-line summary of the verification outcome.
    pub fn summary(&self) -> String {
        if self.overall_valid {
            format!(
                "VALID – integrity={} expiry={} pac={} automaton={} props={} (warnings: {})",
                self.integrity.passed,
                self.expiry.passed,
                self.pac_bounds.passed,
                self.automaton.passed,
                self.properties.passed,
                self.warnings.len(),
            )
        } else {
            format!(
                "INVALID – errors: [{}]",
                self.errors.join("; ")
            )
        }
    }

    /// A certificate is trusted when every individual check passed and there
    /// are zero errors.
    pub fn is_trusted(&self) -> bool {
        self.overall_valid
            && self.integrity.passed
            && self.expiry.passed
            && self.pac_bounds.passed
            && self.automaton.passed
            && self.properties.passed
            && self.errors.is_empty()
    }
}

impl fmt::Display for VerificationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// ---------------------------------------------------------------------------
// Chain verification
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ChainVerificationResult {
    pub all_valid: bool,
    pub individual_results: Vec<VerificationResult>,
    pub chain_consistent: bool,
    pub temporal_order_valid: bool,
    pub model_consistent: bool,
    pub warnings: Vec<String>,
}

impl Serialize for ChainVerificationResult {}
impl Deserialize for ChainVerificationResult {}

// ---------------------------------------------------------------------------
// Replication types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ReplicatedResults {
    pub property_results: Vec<VerifiableProperty>,
    pub distance_results: Vec<VerifiableDistance>,
}

#[derive(Clone, Debug)]
pub struct PropertyMatch {
    pub name: String,
    pub original_degree: f64,
    pub replicated_degree: f64,
    pub deviation: f64,
    pub matches: bool,
}

#[derive(Clone, Debug)]
pub struct DistanceMatch {
    pub model_a: String,
    pub model_b: String,
    pub original_distance: f64,
    pub replicated_distance: f64,
    pub deviation: f64,
    pub matches: bool,
}

#[derive(Clone, Debug)]
pub struct ReplicationResult {
    pub matches: bool,
    pub property_matches: Vec<PropertyMatch>,
    pub distance_matches: Vec<DistanceMatch>,
    pub max_deviation: f64,
}

// ---------------------------------------------------------------------------
// Verification log
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct VerificationEntry {
    pub cert_id: String,
    pub timestamp: String,
    pub result: String,
    pub details: String,
}

// ---------------------------------------------------------------------------
// HMAC helper
// ---------------------------------------------------------------------------

/// Compute a simple keyed hash (HMAC-like) of `data` using `key`.
///
/// Uses a variant of the HMAC construction:
///   H((key XOR opad) || H((key XOR ipad) || data))
/// where H is a basic hash (DJB2-family 64-bit) and ipad/opad are the
/// standard 0x36/0x5c bytes.
///
/// This is intentionally self-contained – no external crate needed.
fn hmac_compute(data: &str, key: &str) -> String {
    // Pad or hash key to 64 bytes
    let key_bytes: Vec<u8> = if key.len() > 64 {
        let h = djb2_hash_bytes(key.as_bytes());
        let mut v = h.to_le_bytes().to_vec();
        v.resize(64, 0u8);
        v
    } else {
        let mut v = key.as_bytes().to_vec();
        v.resize(64, 0u8);
        v
    };

    let ipad: Vec<u8> = key_bytes.iter().map(|b| b ^ 0x36).collect();
    let opad: Vec<u8> = key_bytes.iter().map(|b| b ^ 0x5c).collect();

    // inner = H(ipad || data)
    let mut inner_input: Vec<u8> = ipad;
    inner_input.extend_from_slice(data.as_bytes());
    let inner_hash = djb2_hash_bytes(&inner_input);

    // outer = H(opad || inner_hash_bytes)
    let mut outer_input: Vec<u8> = opad;
    outer_input.extend_from_slice(&inner_hash.to_le_bytes());
    let outer_hash = djb2_hash_bytes(&outer_input);

    format!("{:016x}", outer_hash)
}

/// Verify that `expected_hash` matches the HMAC of `data` under `key`.
pub fn hmac_verify(data: &str, key: &str, expected_hash: &str) -> bool {
    let computed = hmac_compute(data, key);
    constant_time_eq(computed.as_bytes(), expected_hash.as_bytes())
}

/// Constant-time comparison to avoid timing side-channels.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// DJB2-family 64-bit hash (deterministic, non-cryptographic).
fn djb2_hash_bytes(data: &[u8]) -> u64 {
    let mut hash: u64 = 5381;
    for &b in data {
        hash = hash.wrapping_mul(33).wrapping_add(b as u64);
    }
    hash
}

// ---------------------------------------------------------------------------
// Timestamp helpers
// ---------------------------------------------------------------------------

/// Parse an ISO-8601-ish timestamp (YYYY-MM-DDTHH:MM:SSZ or similar) into
/// seconds since the Unix epoch.  Handles the subset produced by this module.
fn parse_timestamp(ts: &str) -> Option<i64> {
    let s = ts.trim().trim_end_matches('Z');
    let parts: Vec<&str> = s.split('T').collect();
    if parts.is_empty() || parts.len() > 2 {
        return None;
    }

    let date_parts: Vec<&str> = parts[0].split('-').collect();
    if date_parts.len() != 3 {
        return None;
    }
    let year: i64 = date_parts[0].parse().ok()?;
    let month: i64 = date_parts[1].parse().ok()?;
    let day: i64 = date_parts[2].parse().ok()?;

    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }

    let (hour, minute, second) = if parts.len() == 2 {
        let time_parts: Vec<&str> = parts[1].split(':').collect();
        if time_parts.len() != 3 {
            return None;
        }
        let h: i64 = time_parts[0].parse().ok()?;
        let m: i64 = time_parts[1].parse().ok()?;
        let sec: i64 = time_parts[2].parse().ok()?;
        (h, m, sec)
    } else {
        (0, 0, 0)
    };

    // Convert to approximate Unix timestamp using a simplified algorithm
    // (ignores leap seconds, good enough for certificate comparison).
    let days = days_from_civil(year, month, day);
    Some(days * 86400 + hour * 3600 + minute * 60 + second)
}

/// Days from 0000-03-01 (an epoch chosen for calendar arithmetic convenience)
/// converted to Unix epoch days. Uses the algorithms from Howard Hinnant's
/// `chrono`-compatible date library.
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u64;
    let m = month;
    let doy = if m > 2 {
        (153 * (m - 3) + 2) / 5 + day - 1
    } else {
        (153 * (m + 9) + 2) / 5 + day - 1
    };
    let doe = yoe as i64 * 365 + (yoe / 4) as i64 - (yoe / 100) as i64 + doy;
    let days_epoch = era * 146097 + doe - 719468; // shift to Unix epoch
    days_epoch
}

/// Return the current Unix timestamp in seconds.
fn current_unix_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Format a Unix timestamp back to ISO-8601.
fn format_timestamp(ts: i64) -> String {
    let secs_per_day: i64 = 86400;
    let days = ts.div_euclid(secs_per_day);
    let rem = ts.rem_euclid(secs_per_day);
    let hour = rem / 3600;
    let minute = (rem % 3600) / 60;
    let second = rem % 60;

    let (y, m, d) = civil_from_days(days);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y, m, d, hour, minute, second
    )
}

fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy as i64 - (153 * mp as i64 + 2) / 5 + 1;
    let m = if mp < 10 { mp as i64 + 3 } else { mp as i64 - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

// ---------------------------------------------------------------------------
// Transition-data helpers
// ---------------------------------------------------------------------------

/// Transition data is encoded as lines of the form:
///   src_state -> dst_state : symbol / probability
/// e.g.  "0 -> 1 : a / 0.5\n0 -> 2 : b / 0.3"
///
/// Returns a map: (src, symbol) -> Vec<(dst, prob)>.
fn parse_transition_data(
    data: &str,
) -> Result<HashMap<(usize, String), Vec<(usize, f64)>>, String> {
    let mut map: HashMap<(usize, String), Vec<(usize, f64)>> = HashMap::new();

    if data.trim().is_empty() {
        return Ok(map);
    }

    for (line_no, line) in data.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Expected format: "src -> dst : sym / prob"
        let arrow_parts: Vec<&str> = line.splitn(2, "->").collect();
        if arrow_parts.len() != 2 {
            return Err(format!(
                "line {}: expected '->' in transition '{}'",
                line_no + 1,
                line
            ));
        }
        let src: usize = arrow_parts[0]
            .trim()
            .parse()
            .map_err(|_| format!("line {}: bad source state", line_no + 1))?;

        let rest = arrow_parts[1];
        let colon_parts: Vec<&str> = rest.splitn(2, ':').collect();
        if colon_parts.len() != 2 {
            return Err(format!(
                "line {}: expected ':' after destination state",
                line_no + 1
            ));
        }
        let dst: usize = colon_parts[0]
            .trim()
            .parse()
            .map_err(|_| format!("line {}: bad destination state", line_no + 1))?;

        let slash_parts: Vec<&str> = colon_parts[1].splitn(2, '/').collect();
        if slash_parts.len() != 2 {
            return Err(format!(
                "line {}: expected '/' separating symbol and probability",
                line_no + 1
            ));
        }
        let sym = slash_parts[0].trim().to_string();
        let prob: f64 = slash_parts[1]
            .trim()
            .parse()
            .map_err(|_| format!("line {}: bad probability", line_no + 1))?;

        map.entry((src, sym)).or_default().push((dst, prob));
    }

    Ok(map)
}

/// Build a canonical string representation of a certificate for hashing.
/// Includes all semantically significant fields in deterministic order.
fn canonical_cert_string(cert: &CertificateData) -> String {
    let mut parts: Vec<String> = Vec::new();
    parts.push(format!("id={}", cert.id));
    parts.push(format!("model_id={}", cert.model_id));
    parts.push(format!("issued_at={}", cert.issued_at));
    parts.push(format!("valid_until={}", cert.valid_until));
    parts.push(format!("num_states={}", cert.num_states));
    parts.push(format!("num_transitions={}", cert.num_transitions));
    parts.push(format!("alphabet_size={}", cert.alphabet_size));
    parts.push(format!("transition_data={}", cert.transition_data));
    for (i, p) in cert.property_results.iter().enumerate() {
        parts.push(format!(
            "prop[{}]={},{},{},{}",
            i,
            p.name,
            p.satisfied,
            p.degree,
            p.witness.as_deref().unwrap_or("")
        ));
    }
    for (i, d) in cert.distance_results.iter().enumerate() {
        parts.push(format!(
            "dist[{}]={},{},{},{},{}",
            i, d.model_a, d.model_b, d.distance, d.lower_bound, d.upper_bound
        ));
    }
    parts.push(format!("pac_epsilon={}", cert.pac_epsilon));
    parts.push(format!("pac_delta={}", cert.pac_delta));
    parts.push(format!("sample_complexity={}", cert.sample_complexity));
    parts.push(format!("metadata={}", cert.metadata_json));
    parts.join("|")
}

// ---------------------------------------------------------------------------
// CertificateVerifier
// ---------------------------------------------------------------------------

pub struct CertificateVerifier {
    pub config: VerifierConfig,
    verification_log: Vec<VerificationEntry>,
    trusted_keys: Vec<String>,
}

impl CertificateVerifier {
    /// Create a new verifier with the given configuration.
    pub fn new(config: VerifierConfig) -> Self {
        Self {
            config,
            verification_log: Vec::new(),
            trusted_keys: Vec::new(),
        }
    }

    /// Register a trusted key for signature verification.
    pub fn add_trusted_key(&mut self, key: &str) {
        if !self.trusted_keys.contains(&key.to_string()) {
            self.trusted_keys.push(key.to_string());
        }
    }

    // -----------------------------------------------------------------------
    // Top-level verification
    // -----------------------------------------------------------------------

    /// Run all verification checks on a certificate and record the outcome.
    pub fn verify_certificate(&mut self, cert: &CertificateData) -> VerificationResult {
        let integrity = self.verify_integrity(cert);
        let expiry = self.verify_expiry(cert);
        let pac_bounds = self.verify_pac_bounds(cert);
        let automaton = self.verify_automaton(cert);
        let properties = self.verify_property_results(cert);

        let mut warnings: Vec<String> = Vec::new();
        let mut errors: Vec<String> = Vec::new();

        // Collect warnings from individual checks
        if let Some(ref w) = integrity.warning {
            warnings.push(format!("integrity: {}", w));
        }
        if let Some(ref w) = expiry.warning {
            warnings.push(format!("expiry: {}", w));
        }
        if let Some(ref w) = pac_bounds.warning {
            warnings.push(format!("pac_bounds: {}", w));
        }
        if let Some(ref w) = automaton.warning {
            warnings.push(format!("automaton: {}", w));
        }
        if let Some(ref w) = properties.warning {
            warnings.push(format!("properties: {}", w));
        }

        // Collect errors
        if !integrity.passed {
            errors.push(format!("Integrity check failed: {}", integrity.details));
        }
        if !expiry.passed {
            errors.push(format!("Expiry check failed: {}", expiry.details));
        }
        if !pac_bounds.passed {
            errors.push(format!("PAC bounds check failed: {}", pac_bounds.details));
        }
        if !automaton.passed {
            errors.push(format!("Automaton check failed: {}", automaton.details));
        }
        if !properties.passed {
            errors.push(format!("Property check failed: {}", properties.details));
        }

        // In non-strict mode, integrity and expiry failures are warnings, not fatal
        let overall_valid = if self.config.strict_mode {
            integrity.passed
                && expiry.passed
                && pac_bounds.passed
                && automaton.passed
                && properties.passed
        } else {
            pac_bounds.passed && automaton.passed && properties.passed
        };

        let result = VerificationResult {
            overall_valid,
            integrity,
            expiry,
            pac_bounds,
            automaton,
            properties,
            warnings,
            errors,
        };

        // Record in log
        let entry = VerificationEntry {
            cert_id: cert.id.clone(),
            timestamp: format_timestamp(current_unix_timestamp()),
            result: if result.overall_valid {
                "VALID".to_string()
            } else {
                "INVALID".to_string()
            },
            details: result.summary(),
        };
        self.verification_log.push(entry);

        result
    }

    // -----------------------------------------------------------------------
    // Integrity
    // -----------------------------------------------------------------------

    /// Verify certificate integrity by recomputing the HMAC hash from the
    /// certificate contents and comparing it against the stored signature.
    pub fn verify_integrity(&self, cert: &CertificateData) -> IntegrityCheck {
        let sig_hash = match &cert.signature_hash {
            Some(h) => h.clone(),
            None => {
                return IntegrityCheck {
                    passed: false,
                    details: "No signature hash present on certificate".to_string(),
                    warning: Some("Certificate is unsigned".to_string()),
                };
            }
        };

        let canonical = canonical_cert_string(cert);

        // Try every trusted key
        if self.trusted_keys.is_empty() {
            return IntegrityCheck {
                passed: false,
                details: "No trusted keys configured; cannot verify signature".to_string(),
                warning: Some("Add at least one trusted key to the verifier".to_string()),
            };
        }

        for key in &self.trusted_keys {
            if hmac_verify(&canonical, key, &sig_hash) {
                return IntegrityCheck {
                    passed: true,
                    details: format!(
                        "Signature verified with algorithm '{}'",
                        cert.signature_algorithm
                            .as_deref()
                            .unwrap_or("hmac-djb2")
                    ),
                    warning: None,
                };
            }
        }

        IntegrityCheck {
            passed: false,
            details: "Signature hash does not match any trusted key".to_string(),
            warning: None,
        }
    }

    // -----------------------------------------------------------------------
    // Expiry
    // -----------------------------------------------------------------------

    /// Verify that the certificate has not expired.
    pub fn verify_expiry(&self, cert: &CertificateData) -> ExpiryCheck {
        if !self.config.check_expiry {
            return ExpiryCheck {
                passed: true,
                details: "Expiry checking disabled by configuration".to_string(),
                warning: Some("Expiry checking is disabled".to_string()),
            };
        }

        let now = if let Some(ref ref_time) = self.config.reference_time {
            match parse_timestamp(ref_time) {
                Some(t) => t,
                None => {
                    return ExpiryCheck {
                        passed: false,
                        details: format!(
                            "Could not parse reference_time '{}'",
                            ref_time
                        ),
                        warning: None,
                    };
                }
            }
        } else {
            current_unix_timestamp()
        };

        let issued = match parse_timestamp(&cert.issued_at) {
            Some(t) => t,
            None => {
                return ExpiryCheck {
                    passed: false,
                    details: format!(
                        "Could not parse issued_at timestamp '{}'",
                        cert.issued_at
                    ),
                    warning: None,
                };
            }
        };

        let valid_until = match parse_timestamp(&cert.valid_until) {
            Some(t) => t,
            None => {
                return ExpiryCheck {
                    passed: false,
                    details: format!(
                        "Could not parse valid_until timestamp '{}'",
                        cert.valid_until
                    ),
                    warning: None,
                };
            }
        };

        if now < issued {
            return ExpiryCheck {
                passed: false,
                details: format!(
                    "Certificate not yet valid (issued_at {} is in the future)",
                    cert.issued_at
                ),
                warning: None,
            };
        }

        if now > valid_until {
            let expired_ago = now - valid_until;
            return ExpiryCheck {
                passed: false,
                details: format!(
                    "Certificate expired {} seconds ago (valid_until={})",
                    expired_ago, cert.valid_until
                ),
                warning: None,
            };
        }

        // Warn if expiring within 24 hours
        let remaining = valid_until - now;
        let warning = if remaining < 86400 {
            Some(format!(
                "Certificate expires in {} seconds (< 24 hours)",
                remaining
            ))
        } else {
            None
        };

        ExpiryCheck {
            passed: true,
            details: format!(
                "Certificate valid ({} seconds remaining)",
                remaining
            ),
            warning,
        }
    }

    // -----------------------------------------------------------------------
    // PAC bounds
    // -----------------------------------------------------------------------

    /// Verify PAC learning bounds:
    /// - epsilon in (0, 1)
    /// - delta in (0, 1)
    /// - sample_complexity >= ceil(1 / epsilon^2 * ln(1 / delta))
    pub fn verify_pac_bounds(&self, cert: &CertificateData) -> PACBoundsCheck {
        let eps = cert.pac_epsilon;
        let delta = cert.pac_delta;
        let n = cert.sample_complexity;

        if eps <= 0.0 || eps >= 1.0 {
            return PACBoundsCheck {
                passed: false,
                details: format!("epsilon={} is not in (0,1)", eps),
                warning: None,
            };
        }

        if delta <= 0.0 || delta >= 1.0 {
            return PACBoundsCheck {
                passed: false,
                details: format!("delta={} is not in (0,1)", delta),
                warning: None,
            };
        }

        // Required: n >= (1/eps^2) * ln(1/delta)
        let required = (1.0 / (eps * eps)) * (1.0_f64 / delta).ln();
        let required_ceil = required.ceil() as usize;

        if n < required_ceil {
            return PACBoundsCheck {
                passed: false,
                details: format!(
                    "sample_complexity {} < required {} (eps={}, delta={})",
                    n, required_ceil, eps, delta
                ),
                warning: None,
            };
        }

        // Warn if sample complexity is much larger than needed (>100x)
        let warning = if n as f64 > required * 100.0 {
            Some(format!(
                "sample_complexity {} is >100x the required {} – possibly wasteful",
                n, required_ceil
            ))
        } else {
            None
        };

        PACBoundsCheck {
            passed: true,
            details: format!(
                "PAC bounds satisfied: n={} >= required {} (eps={}, delta={})",
                n, required_ceil, eps, delta
            ),
            warning,
        }
    }

    // -----------------------------------------------------------------------
    // Automaton
    // -----------------------------------------------------------------------

    /// Verify automaton structural consistency:
    /// - num_states > 0
    /// - transitions parse correctly
    /// - all referenced states < num_states
    /// - for each (src, symbol), probabilities sum to <= 1.0
    pub fn verify_automaton(&self, cert: &CertificateData) -> AutomatonCheck {
        if cert.num_states == 0 {
            return AutomatonCheck {
                passed: false,
                details: "num_states is 0".to_string(),
                warning: None,
            };
        }

        let transitions = match parse_transition_data(&cert.transition_data) {
            Ok(t) => t,
            Err(e) => {
                return AutomatonCheck {
                    passed: false,
                    details: format!("Failed to parse transition data: {}", e),
                    warning: None,
                };
            }
        };

        let mut actual_transitions: usize = 0;
        let mut distinct_symbols: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        let mut warning: Option<String> = None;

        for ((src, sym), dests) in &transitions {
            distinct_symbols.insert(sym.clone());
            actual_transitions += dests.len();

            if *src >= cert.num_states {
                return AutomatonCheck {
                    passed: false,
                    details: format!(
                        "Source state {} >= num_states {}",
                        src, cert.num_states
                    ),
                    warning: None,
                };
            }

            let mut prob_sum: f64 = 0.0;
            for (dst, prob) in dests {
                if *dst >= cert.num_states {
                    return AutomatonCheck {
                        passed: false,
                        details: format!(
                            "Destination state {} >= num_states {}",
                            dst, cert.num_states
                        ),
                        warning: None,
                    };
                }
                if *prob < 0.0 || *prob > 1.0 {
                    return AutomatonCheck {
                        passed: false,
                        details: format!(
                            "Probability {} out of [0,1] for transition {} ->{} : {}",
                            prob, src, dst, sym
                        ),
                        warning: None,
                    };
                }
                prob_sum += prob;
            }

            if prob_sum > 1.0 + self.config.tolerance {
                return AutomatonCheck {
                    passed: false,
                    details: format!(
                        "Probabilities sum to {} (> 1.0 + tolerance {}) for state {} symbol '{}'",
                        prob_sum, self.config.tolerance, src, sym
                    ),
                    warning: None,
                };
            }

            if prob_sum > 1.0 {
                warning = Some(format!(
                    "Probability sum {} slightly exceeds 1.0 for state {} symbol '{}' (within tolerance)",
                    prob_sum, src, sym
                ));
            }
        }

        // Check declared vs actual transition count
        if actual_transitions != cert.num_transitions && cert.num_transitions > 0 {
            warning = Some(format!(
                "Declared num_transitions={} but found {} in transition_data",
                cert.num_transitions, actual_transitions
            ));
        }

        // Check declared alphabet size
        if cert.alphabet_size > 0 && distinct_symbols.len() > cert.alphabet_size {
            return AutomatonCheck {
                passed: false,
                details: format!(
                    "Found {} distinct symbols but alphabet_size is {}",
                    distinct_symbols.len(),
                    cert.alphabet_size
                ),
                warning: None,
            };
        }

        AutomatonCheck {
            passed: true,
            details: format!(
                "Automaton consistent: {} states, {} transitions, {} symbols",
                cert.num_states, actual_transitions, distinct_symbols.len()
            ),
            warning,
        }
    }

    // -----------------------------------------------------------------------
    // Property results
    // -----------------------------------------------------------------------

    /// Verify property results:
    /// - satisfaction degree in [0, 1]
    /// - if satisfied == true then degree > 0
    /// - if satisfied == false then degree should be 0 (or warn)
    pub fn verify_property_results(&self, cert: &CertificateData) -> PropertyCheck {
        if cert.property_results.is_empty() {
            return PropertyCheck {
                passed: true,
                details: "No property results to verify".to_string(),
                warning: Some("Certificate contains no property results".to_string()),
            };
        }

        let mut issues: Vec<String> = Vec::new();
        let mut warn_list: Vec<String> = Vec::new();

        for prop in &cert.property_results {
            if prop.degree < 0.0 || prop.degree > 1.0 {
                issues.push(format!(
                    "Property '{}': degree {} not in [0,1]",
                    prop.name, prop.degree
                ));
            }

            if prop.satisfied && prop.degree <= 0.0 {
                issues.push(format!(
                    "Property '{}': satisfied=true but degree={}",
                    prop.name, prop.degree
                ));
            }

            if !prop.satisfied && prop.degree > 0.0 {
                if self.config.strict_mode {
                    issues.push(format!(
                        "Property '{}': satisfied=false but degree={} (strict mode)",
                        prop.name, prop.degree
                    ));
                } else {
                    warn_list.push(format!(
                        "Property '{}': satisfied=false but degree={}",
                        prop.name, prop.degree
                    ));
                }
            }
        }

        // Also check distance results for sanity
        for dist in &cert.distance_results {
            if dist.distance < 0.0 {
                issues.push(format!(
                    "Distance ({}, {}): negative distance {}",
                    dist.model_a, dist.model_b, dist.distance
                ));
            }
            if dist.lower_bound > dist.distance + self.config.tolerance {
                issues.push(format!(
                    "Distance ({}, {}): lower_bound {} > distance {}",
                    dist.model_a, dist.model_b, dist.lower_bound, dist.distance
                ));
            }
            if dist.upper_bound < dist.distance - self.config.tolerance {
                issues.push(format!(
                    "Distance ({}, {}): upper_bound {} < distance {}",
                    dist.model_a, dist.model_b, dist.upper_bound, dist.distance
                ));
            }
        }

        if !issues.is_empty() {
            return PropertyCheck {
                passed: false,
                details: issues.join("; "),
                warning: if warn_list.is_empty() {
                    None
                } else {
                    Some(warn_list.join("; "))
                },
            };
        }

        PropertyCheck {
            passed: true,
            details: format!(
                "All {} properties and {} distances verified",
                cert.property_results.len(),
                cert.distance_results.len()
            ),
            warning: if warn_list.is_empty() {
                None
            } else {
                Some(warn_list.join("; "))
            },
        }
    }

    // -----------------------------------------------------------------------
    // Chain verification
    // -----------------------------------------------------------------------

    /// Verify a chain (sequence) of certificates:
    /// 1. Each individual certificate must be valid.
    /// 2. Temporal ordering: issued_at of cert[i] <= issued_at of cert[i+1].
    /// 3. Model consistency: all certs in the chain refer to the same model_id
    ///    (or a coherent succession).
    pub fn verify_chain(
        &mut self,
        chain: &[CertificateData],
    ) -> ChainVerificationResult {
        if chain.is_empty() {
            return ChainVerificationResult {
                all_valid: true,
                individual_results: Vec::new(),
                chain_consistent: true,
                temporal_order_valid: true,
                model_consistent: true,
                warnings: vec!["Empty certificate chain".to_string()],
            };
        }

        let mut individual_results: Vec<VerificationResult> = Vec::new();
        let mut all_valid = true;
        let mut warnings: Vec<String> = Vec::new();

        for cert in chain {
            let result = self.verify_certificate(cert);
            if !result.overall_valid {
                all_valid = false;
            }
            individual_results.push(result);
        }

        // Check temporal ordering
        let mut temporal_order_valid = true;
        for i in 0..chain.len() - 1 {
            let t_a = parse_timestamp(&chain[i].issued_at);
            let t_b = parse_timestamp(&chain[i + 1].issued_at);
            match (t_a, t_b) {
                (Some(a), Some(b)) => {
                    if a > b {
                        temporal_order_valid = false;
                        warnings.push(format!(
                            "Temporal order violation: cert '{}' issued at {} after cert '{}' issued at {}",
                            chain[i].id, chain[i].issued_at,
                            chain[i + 1].id, chain[i + 1].issued_at
                        ));
                    }
                }
                _ => {
                    temporal_order_valid = false;
                    warnings.push(format!(
                        "Could not parse timestamps for ordering check between certs '{}' and '{}'",
                        chain[i].id, chain[i + 1].id
                    ));
                }
            }
        }

        // Check model consistency – all certs should reference the same model
        let first_model = &chain[0].model_id;
        let model_consistent = chain.iter().all(|c| c.model_id == *first_model);
        if !model_consistent {
            let models: Vec<&str> = chain.iter().map(|c| c.model_id.as_str()).collect();
            warnings.push(format!(
                "Inconsistent model_ids in chain: {:?}",
                models
            ));
        }

        let chain_consistent = temporal_order_valid && model_consistent && all_valid;

        ChainVerificationResult {
            all_valid,
            individual_results,
            chain_consistent,
            temporal_order_valid,
            model_consistent,
            warnings,
        }
    }

    // -----------------------------------------------------------------------
    // Independent replication
    // -----------------------------------------------------------------------

    /// Compare a certificate's results against independently replicated
    /// results. Each property and distance is compared; deviations beyond
    /// `replication_tolerance` are flagged.
    pub fn independent_replication(
        &self,
        cert: &CertificateData,
        replicated: &ReplicatedResults,
    ) -> ReplicationResult {
        let tol = self.config.replication_tolerance;
        let mut property_matches: Vec<PropertyMatch> = Vec::new();
        let mut distance_matches: Vec<DistanceMatch> = Vec::new();
        let mut max_deviation: f64 = 0.0;
        let mut all_match = true;

        // Build a lookup from name -> replicated property
        let rep_props: HashMap<&str, &VerifiableProperty> = replicated
            .property_results
            .iter()
            .map(|p| (p.name.as_str(), p))
            .collect();

        for orig in &cert.property_results {
            if let Some(rep) = rep_props.get(orig.name.as_str()) {
                let dev = (orig.degree - rep.degree).abs();
                let matches = dev <= tol;
                if !matches {
                    all_match = false;
                }
                if dev > max_deviation {
                    max_deviation = dev;
                }
                property_matches.push(PropertyMatch {
                    name: orig.name.clone(),
                    original_degree: orig.degree,
                    replicated_degree: rep.degree,
                    deviation: dev,
                    matches,
                });
            } else {
                // Missing replicated result is a mismatch
                all_match = false;
                property_matches.push(PropertyMatch {
                    name: orig.name.clone(),
                    original_degree: orig.degree,
                    replicated_degree: f64::NAN,
                    deviation: f64::INFINITY,
                    matches: false,
                });
            }
        }

        // Distance comparison – match on (model_a, model_b) pair
        let rep_dists: HashMap<(&str, &str), &VerifiableDistance> = replicated
            .distance_results
            .iter()
            .map(|d| ((d.model_a.as_str(), d.model_b.as_str()), d))
            .collect();

        for orig in &cert.distance_results {
            let key = (orig.model_a.as_str(), orig.model_b.as_str());
            if let Some(rep) = rep_dists.get(&key) {
                let dev = (orig.distance - rep.distance).abs();
                let matches = dev <= tol;
                if !matches {
                    all_match = false;
                }
                if dev > max_deviation {
                    max_deviation = dev;
                }
                distance_matches.push(DistanceMatch {
                    model_a: orig.model_a.clone(),
                    model_b: orig.model_b.clone(),
                    original_distance: orig.distance,
                    replicated_distance: rep.distance,
                    deviation: dev,
                    matches,
                });
            } else {
                all_match = false;
                distance_matches.push(DistanceMatch {
                    model_a: orig.model_a.clone(),
                    model_b: orig.model_b.clone(),
                    original_distance: orig.distance,
                    replicated_distance: f64::NAN,
                    deviation: f64::INFINITY,
                    matches: false,
                });
            }
        }

        ReplicationResult {
            matches: all_match,
            property_matches,
            distance_matches,
            max_deviation,
        }
    }

    // -----------------------------------------------------------------------
    // Log access
    // -----------------------------------------------------------------------

    /// Return the full verification log.
    pub fn verification_log(&self) -> &[VerificationEntry] {
        &self.verification_log
    }
}

// ---------------------------------------------------------------------------
// Convenience: sign a certificate (for testing / round-trip)
// ---------------------------------------------------------------------------

/// Sign a certificate using the given key, setting signature_hash and
/// signature_algorithm fields. Returns the computed hash.
pub fn sign_certificate(cert: &mut CertificateData, key: &str) -> String {
    let canonical = canonical_cert_string(cert);
    let hash = hmac_compute(&canonical, key);
    cert.signature_algorithm = Some("hmac-djb2".to_string());
    cert.signature_hash = Some(hash.clone());
    hash
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Build a minimal valid certificate for testing purposes.
fn make_test_certificate() -> CertificateData {
    CertificateData {
        id: "test-cert-001".to_string(),
        model_id: "model-alpha".to_string(),
        issued_at: "2025-01-01T00:00:00Z".to_string(),
        valid_until: "2030-12-31T23:59:59Z".to_string(),
        num_states: 3,
        num_transitions: 4,
        alphabet_size: 2,
        transition_data: "0 -> 1 : a / 0.5\n0 -> 2 : b / 0.3\n1 -> 0 : a / 1.0\n2 -> 0 : b / 0.7"
            .to_string(),
        property_results: vec![
            VerifiableProperty {
                name: "safety".to_string(),
                satisfied: true,
                degree: 0.95,
                witness: Some("state 0".to_string()),
            },
            VerifiableProperty {
                name: "liveness".to_string(),
                satisfied: true,
                degree: 0.80,
                witness: None,
            },
        ],
        distance_results: vec![VerifiableDistance {
            model_a: "model-alpha".to_string(),
            model_b: "model-beta".to_string(),
            distance: 0.12,
            lower_bound: 0.10,
            upper_bound: 0.15,
        }],
        pac_epsilon: 0.1,
        pac_delta: 0.05,
        sample_complexity: 600, // 1/(0.01)*ln(20) ≈ 300; 600 is sufficient
        signature_algorithm: None,
        signature_hash: None,
        signed_at: None,
        metadata_json: "{}".to_string(),
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_verifier() -> CertificateVerifier {
        CertificateVerifier::new(VerifierConfig::default())
    }

    fn signed_test_cert(key: &str) -> CertificateData {
        let mut cert = make_test_certificate();
        sign_certificate(&mut cert, key);
        cert
    }

    // 1. Basic construction
    #[test]
    fn test_verifier_new() {
        let v = default_verifier();
        assert!(v.verification_log.is_empty());
        assert!(v.trusted_keys.is_empty());
        assert!((v.config.tolerance - 0.01).abs() < f64::EPSILON);
    }

    // 2. Adding trusted keys (no duplicates)
    #[test]
    fn test_add_trusted_key() {
        let mut v = default_verifier();
        v.add_trusted_key("key-1");
        v.add_trusted_key("key-1");
        v.add_trusted_key("key-2");
        assert_eq!(v.trusted_keys.len(), 2);
    }

    // 3. HMAC round-trip
    #[test]
    fn test_hmac_roundtrip() {
        let hash = hmac_compute("hello world", "secret");
        assert!(hmac_verify("hello world", "secret", &hash));
        assert!(!hmac_verify("hello world", "wrong-key", &hash));
        assert!(!hmac_verify("tampered", "secret", &hash));
    }

    // 4. Integrity check – signed certificate passes
    #[test]
    fn test_integrity_pass() {
        let mut v = default_verifier();
        v.add_trusted_key("my-key");
        let cert = signed_test_cert("my-key");
        let check = v.verify_integrity(&cert);
        assert!(check.passed);
    }

    // 5. Integrity check – wrong key fails
    #[test]
    fn test_integrity_fail_wrong_key() {
        let mut v = default_verifier();
        v.add_trusted_key("other-key");
        let cert = signed_test_cert("my-key");
        let check = v.verify_integrity(&cert);
        assert!(!check.passed);
    }

    // 6. Integrity check – unsigned certificate
    #[test]
    fn test_integrity_unsigned() {
        let v = default_verifier();
        let cert = make_test_certificate();
        let check = v.verify_integrity(&cert);
        assert!(!check.passed);
        assert!(check.warning.is_some());
    }

    // 7. Expiry check – valid certificate
    #[test]
    fn test_expiry_valid() {
        let config = VerifierConfig {
            reference_time: Some("2026-06-15T12:00:00Z".to_string()),
            ..Default::default()
        };
        let v = CertificateVerifier::new(config);
        let cert = make_test_certificate();
        let check = v.verify_expiry(&cert);
        assert!(check.passed);
    }

    // 8. Expiry check – expired certificate
    #[test]
    fn test_expiry_expired() {
        let config = VerifierConfig {
            reference_time: Some("2031-06-15T12:00:00Z".to_string()),
            ..Default::default()
        };
        let v = CertificateVerifier::new(config);
        let cert = make_test_certificate();
        let check = v.verify_expiry(&cert);
        assert!(!check.passed);
        assert!(check.details.contains("expired"));
    }

    // 9. PAC bounds – valid
    #[test]
    fn test_pac_bounds_valid() {
        let v = default_verifier();
        let cert = make_test_certificate();
        let check = v.verify_pac_bounds(&cert);
        assert!(check.passed, "PAC check should pass: {}", check.details);
    }

    // 10. PAC bounds – insufficient samples
    #[test]
    fn test_pac_bounds_insufficient() {
        let v = default_verifier();
        let mut cert = make_test_certificate();
        cert.sample_complexity = 1; // way too low
        let check = v.verify_pac_bounds(&cert);
        assert!(!check.passed);
    }

    // 11. Automaton check – valid
    #[test]
    fn test_automaton_valid() {
        let v = default_verifier();
        let cert = make_test_certificate();
        let check = v.verify_automaton(&cert);
        assert!(check.passed, "Automaton check should pass: {}", check.details);
    }

    // 12. Automaton check – zero states
    #[test]
    fn test_automaton_zero_states() {
        let v = default_verifier();
        let mut cert = make_test_certificate();
        cert.num_states = 0;
        let check = v.verify_automaton(&cert);
        assert!(!check.passed);
    }

    // 13. Property check – valid
    #[test]
    fn test_property_check_valid() {
        let v = default_verifier();
        let cert = make_test_certificate();
        let check = v.verify_property_results(&cert);
        assert!(check.passed, "Property check failed: {}", check.details);
    }

    // 14. Property check – degree out of range
    #[test]
    fn test_property_degree_out_of_range() {
        let v = default_verifier();
        let mut cert = make_test_certificate();
        cert.property_results.push(VerifiableProperty {
            name: "bad-prop".to_string(),
            satisfied: true,
            degree: 1.5,
            witness: None,
        });
        let check = v.verify_property_results(&cert);
        assert!(!check.passed);
    }

    // 15. Full certificate verification – non-strict, signed
    #[test]
    fn test_full_verify_valid() {
        let config = VerifierConfig {
            reference_time: Some("2026-01-01T00:00:00Z".to_string()),
            ..Default::default()
        };
        let mut v = CertificateVerifier::new(config);
        v.add_trusted_key("key-1");
        let cert = signed_test_cert("key-1");
        let result = v.verify_certificate(&cert);
        assert!(result.overall_valid);
        assert!(result.is_trusted());
        assert_eq!(v.verification_log().len(), 1);
    }

    // 16. Chain verification – correct order
    #[test]
    fn test_chain_verification_valid() {
        let config = VerifierConfig {
            reference_time: Some("2026-01-01T00:00:00Z".to_string()),
            ..Default::default()
        };
        let mut v = CertificateVerifier::new(config);
        v.add_trusted_key("k");

        let mut c1 = make_test_certificate();
        c1.id = "cert-1".to_string();
        c1.issued_at = "2025-01-01T00:00:00Z".to_string();
        sign_certificate(&mut c1, "k");

        let mut c2 = make_test_certificate();
        c2.id = "cert-2".to_string();
        c2.issued_at = "2025-06-01T00:00:00Z".to_string();
        sign_certificate(&mut c2, "k");

        let chain_result = v.verify_chain(&[c1, c2]);
        assert!(chain_result.all_valid);
        assert!(chain_result.temporal_order_valid);
        assert!(chain_result.model_consistent);
        assert!(chain_result.chain_consistent);
    }

    // 17. Chain verification – bad temporal order
    #[test]
    fn test_chain_temporal_order_violation() {
        let config = VerifierConfig {
            reference_time: Some("2026-01-01T00:00:00Z".to_string()),
            ..Default::default()
        };
        let mut v = CertificateVerifier::new(config);
        v.add_trusted_key("k");

        let mut c1 = make_test_certificate();
        c1.id = "cert-late".to_string();
        c1.issued_at = "2025-06-01T00:00:00Z".to_string();
        sign_certificate(&mut c1, "k");

        let mut c2 = make_test_certificate();
        c2.id = "cert-early".to_string();
        c2.issued_at = "2025-01-01T00:00:00Z".to_string();
        sign_certificate(&mut c2, "k");

        let chain_result = v.verify_chain(&[c1, c2]);
        assert!(!chain_result.temporal_order_valid);
    }

    // 18. Replication – matching results
    #[test]
    fn test_replication_match() {
        let v = default_verifier();
        let cert = make_test_certificate();
        let replicated = ReplicatedResults {
            property_results: vec![
                VerifiableProperty {
                    name: "safety".to_string(),
                    satisfied: true,
                    degree: 0.96,
                    witness: None,
                },
                VerifiableProperty {
                    name: "liveness".to_string(),
                    satisfied: true,
                    degree: 0.81,
                    witness: None,
                },
            ],
            distance_results: vec![VerifiableDistance {
                model_a: "model-alpha".to_string(),
                model_b: "model-beta".to_string(),
                distance: 0.13,
                lower_bound: 0.10,
                upper_bound: 0.15,
            }],
        };
        let rep = v.independent_replication(&cert, &replicated);
        assert!(rep.matches, "Replication should match within tolerance");
        assert!(rep.max_deviation <= 0.05);
    }

    // 19. Replication – mismatch
    #[test]
    fn test_replication_mismatch() {
        let v = default_verifier();
        let cert = make_test_certificate();
        let replicated = ReplicatedResults {
            property_results: vec![VerifiableProperty {
                name: "safety".to_string(),
                satisfied: false,
                degree: 0.10,
                witness: None,
            }],
            distance_results: vec![],
        };
        let rep = v.independent_replication(&cert, &replicated);
        assert!(!rep.matches);
        assert!(rep.max_deviation > 0.05);
    }

    // 20. Timestamp parsing
    #[test]
    fn test_timestamp_parsing() {
        let ts = parse_timestamp("2025-01-01T00:00:00Z");
        assert!(ts.is_some());
        let ts2 = parse_timestamp("2025-06-15T12:30:45Z");
        assert!(ts2.is_some());
        assert!(ts2.unwrap() > ts.unwrap());

        // Invalid
        assert!(parse_timestamp("not-a-date").is_none());
        assert!(parse_timestamp("2025-13-01T00:00:00Z").is_none());
    }

    // 21. Transition data parsing
    #[test]
    fn test_transition_parsing() {
        let data = "0 -> 1 : a / 0.5\n1 -> 0 : b / 1.0";
        let parsed = parse_transition_data(data);
        assert!(parsed.is_ok());
        let map = parsed.unwrap();
        assert_eq!(map.len(), 2);
    }

    // 22. Automaton – probabilities exceed 1
    #[test]
    fn test_automaton_prob_exceeds_one() {
        let v = default_verifier();
        let mut cert = make_test_certificate();
        cert.transition_data =
            "0 -> 1 : a / 0.6\n0 -> 2 : a / 0.6".to_string();
        let check = v.verify_automaton(&cert);
        assert!(!check.passed, "Should fail: prob sum > 1 + tolerance");
    }

    // 23. Strict mode – integrity failure causes overall failure
    #[test]
    fn test_strict_mode_integrity_failure() {
        let config = VerifierConfig {
            strict_mode: true,
            reference_time: Some("2026-01-01T00:00:00Z".to_string()),
            ..Default::default()
        };
        let mut v = CertificateVerifier::new(config);
        // No trusted keys – integrity will fail
        let cert = make_test_certificate();
        let result = v.verify_certificate(&cert);
        assert!(!result.overall_valid);
        assert!(!result.is_trusted());
    }

    // 24. Verification log records entries
    #[test]
    fn test_verification_log() {
        let config = VerifierConfig {
            reference_time: Some("2026-01-01T00:00:00Z".to_string()),
            ..Default::default()
        };
        let mut v = CertificateVerifier::new(config);
        let cert = make_test_certificate();
        let _ = v.verify_certificate(&cert);
        let _ = v.verify_certificate(&cert);
        assert_eq!(v.verification_log().len(), 2);
        assert_eq!(v.verification_log()[0].cert_id, "test-cert-001");
    }

    // 25. PAC bounds – epsilon out of range
    #[test]
    fn test_pac_epsilon_out_of_range() {
        let v = default_verifier();
        let mut cert = make_test_certificate();
        cert.pac_epsilon = 0.0;
        let check = v.verify_pac_bounds(&cert);
        assert!(!check.passed);

        cert.pac_epsilon = 1.0;
        let check2 = v.verify_pac_bounds(&cert);
        assert!(!check2.passed);
    }

    // 26. Summary and display
    #[test]
    fn test_summary_display() {
        let config = VerifierConfig {
            reference_time: Some("2026-01-01T00:00:00Z".to_string()),
            ..Default::default()
        };
        let mut v = CertificateVerifier::new(config);
        v.add_trusted_key("k");
        let cert = signed_test_cert("k");
        let result = v.verify_certificate(&cert);
        let summary = result.summary();
        assert!(summary.contains("VALID"));
        let display = format!("{}", result);
        assert!(!display.is_empty());
    }

    // 27. Chain – model inconsistency
    #[test]
    fn test_chain_model_inconsistency() {
        let config = VerifierConfig {
            reference_time: Some("2026-01-01T00:00:00Z".to_string()),
            ..Default::default()
        };
        let mut v = CertificateVerifier::new(config);

        let mut c1 = make_test_certificate();
        c1.model_id = "model-A".to_string();
        let mut c2 = make_test_certificate();
        c2.model_id = "model-B".to_string();
        c2.issued_at = "2025-06-01T00:00:00Z".to_string();

        let chain_result = v.verify_chain(&[c1, c2]);
        assert!(!chain_result.model_consistent);
    }

    // 28. Expiry disabled
    #[test]
    fn test_expiry_disabled() {
        let config = VerifierConfig {
            check_expiry: false,
            ..Default::default()
        };
        let v = CertificateVerifier::new(config);
        let mut cert = make_test_certificate();
        cert.valid_until = "2000-01-01T00:00:00Z".to_string();
        let check = v.verify_expiry(&cert);
        assert!(check.passed);
    }
}
