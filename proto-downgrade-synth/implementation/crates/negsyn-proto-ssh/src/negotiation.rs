//! SSH algorithm negotiation engine.
//!
//! Implements the RFC 4253 §7.1 algorithm selection rules:
//! - For each category, iterate through the **client's** list; the first
//!   algorithm that also appears in the **server's** list is chosen.
//! - If no common algorithm exists, the connection fails.
//!
//! Also handles strict-KEX negotiation, cross-category consistency
//! checking, and guess algorithm handling (first_kex_packet_follows).

use crate::algorithms::{
    CompressionAlgorithm, EncryptionAlgorithm, HostKeyAlgorithm,
    MacAlgorithm as SshMac, SecurityClassification,
};
use crate::extensions::StrictKex;
use crate::kex::{KexAlgorithm, KexInit};

use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Negotiated algorithms
// ---------------------------------------------------------------------------

/// The result of a successful algorithm negotiation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NegotiatedAlgorithms {
    pub kex: String,
    pub host_key: String,
    pub encryption_client_to_server: String,
    pub encryption_server_to_client: String,
    pub mac_client_to_server: String,
    pub mac_server_to_client: String,
    pub compression_client_to_server: String,
    pub compression_server_to_client: String,
}

impl NegotiatedAlgorithms {
    /// Compute the overall security classification.
    pub fn overall_security(&self) -> SecurityClassification {
        let mut min = SecurityClassification::Recommended;

        // KEX
        if let Some(alg) = KexAlgorithm::from_wire_name(&self.kex) {
            let s = alg.security();
            if s < min {
                min = s;
            }
        }

        // Host key
        if let Some(alg) = HostKeyAlgorithm::from_wire_name(&self.host_key) {
            let s = alg.security();
            if s < min {
                min = s;
            }
        }

        // Encryption
        for enc_name in [
            &self.encryption_client_to_server,
            &self.encryption_server_to_client,
        ] {
            if let Some(alg) = EncryptionAlgorithm::from_wire_name(enc_name) {
                let s = alg.security();
                if s < min {
                    min = s;
                }
            }
        }

        // MAC (only relevant for non-AEAD ciphers)
        let enc_c2s = EncryptionAlgorithm::from_wire_name(&self.encryption_client_to_server);
        let enc_s2c = EncryptionAlgorithm::from_wire_name(&self.encryption_server_to_client);

        if !enc_c2s.map(|e| e.is_aead()).unwrap_or(false) {
            if let Some(alg) = SshMac::from_wire_name(&self.mac_client_to_server) {
                let s = alg.security();
                if s < min {
                    min = s;
                }
            }
        }

        if !enc_s2c.map(|e| e.is_aead()).unwrap_or(false) {
            if let Some(alg) = SshMac::from_wire_name(&self.mac_server_to_client) {
                let s = alg.security();
                if s < min {
                    min = s;
                }
            }
        }

        // Compression
        for comp_name in [
            &self.compression_client_to_server,
            &self.compression_server_to_client,
        ] {
            if let Some(alg) = CompressionAlgorithm::from_wire_name(comp_name) {
                let s = alg.security();
                if s < min {
                    min = s;
                }
            }
        }

        min
    }

    /// Returns true if any deprecated algorithm is in use.
    pub fn has_deprecated(&self) -> bool {
        self.overall_security().is_deprecated()
    }

    /// Returns the list of weak or broken algorithms in this negotiation.
    pub fn weak_algorithms(&self) -> Vec<(String, SecurityClassification)> {
        let mut weak = Vec::new();

        let check = |name: &str, weak: &mut Vec<(String, SecurityClassification)>| {
            // Check all categories
            if let Some(alg) = KexAlgorithm::from_wire_name(name) {
                if alg.security().is_deprecated() {
                    weak.push((name.to_string(), alg.security()));
                }
                return;
            }
            if let Some(alg) = HostKeyAlgorithm::from_wire_name(name) {
                if alg.security().is_deprecated() {
                    weak.push((name.to_string(), alg.security()));
                }
                return;
            }
            if let Some(alg) = EncryptionAlgorithm::from_wire_name(name) {
                if alg.security().is_deprecated() {
                    weak.push((name.to_string(), alg.security()));
                }
                return;
            }
            if let Some(alg) = SshMac::from_wire_name(name) {
                if alg.security().is_deprecated() {
                    weak.push((name.to_string(), alg.security()));
                }
                return;
            }
            if let Some(alg) = CompressionAlgorithm::from_wire_name(name) {
                if alg.security().is_deprecated() {
                    weak.push((name.to_string(), alg.security()));
                }
            }
        };

        check(&self.kex, &mut weak);
        check(&self.host_key, &mut weak);
        check(&self.encryption_client_to_server, &mut weak);
        check(&self.encryption_server_to_client, &mut weak);
        check(&self.mac_client_to_server, &mut weak);
        check(&self.mac_server_to_client, &mut weak);
        check(&self.compression_client_to_server, &mut weak);
        check(&self.compression_server_to_client, &mut weak);

        weak
    }

    /// Check if AEAD cipher was negotiated (MAC is implicit).
    pub fn is_aead_c2s(&self) -> bool {
        EncryptionAlgorithm::from_wire_name(&self.encryption_client_to_server)
            .map(|e| e.is_aead())
            .unwrap_or(false)
    }

    pub fn is_aead_s2c(&self) -> bool {
        EncryptionAlgorithm::from_wire_name(&self.encryption_server_to_client)
            .map(|e| e.is_aead())
            .unwrap_or(false)
    }
}

impl fmt::Display for NegotiatedAlgorithms {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Negotiated algorithms:")?;
        writeln!(f, "  kex: {}", self.kex)?;
        writeln!(f, "  host_key: {}", self.host_key)?;
        writeln!(f, "  enc_c2s: {}", self.encryption_client_to_server)?;
        writeln!(f, "  enc_s2c: {}", self.encryption_server_to_client)?;
        writeln!(f, "  mac_c2s: {}", self.mac_client_to_server)?;
        writeln!(f, "  mac_s2c: {}", self.mac_server_to_client)?;
        writeln!(f, "  comp_c2s: {}", self.compression_client_to_server)?;
        writeln!(f, "  comp_s2c: {}", self.compression_server_to_client)?;
        writeln!(f, "  security: {}", self.overall_security())?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Negotiation failure
// ---------------------------------------------------------------------------

/// Detailed information about a negotiation failure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NegotiationFailure {
    pub category: String,
    pub client_offered: Vec<String>,
    pub server_offered: Vec<String>,
}

impl fmt::Display for NegotiationFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "No matching {} algorithm. Client: [{}], Server: [{}]",
            self.category,
            self.client_offered.join(", "),
            self.server_offered.join(", ")
        )
    }
}

// ---------------------------------------------------------------------------
// SshNegotiationEngine
// ---------------------------------------------------------------------------

/// SSH algorithm negotiation engine implementing RFC 4253 §7.1.
#[derive(Debug, Clone)]
pub struct SshNegotiationEngine {
    /// Whether strict-KEX is active.
    pub strict_kex: StrictKex,
    /// Whether to enforce FIPS mode (only FIPS-approved algorithms).
    pub fips_mode: bool,
    /// Minimum acceptable security classification.
    pub min_security: SecurityClassification,
}

impl SshNegotiationEngine {
    pub fn new() -> Self {
        Self {
            strict_kex: StrictKex::new(),
            fips_mode: false,
            min_security: SecurityClassification::Broken, // accept anything by default
        }
    }

    pub fn with_strict_kex(mut self, sk: StrictKex) -> Self {
        self.strict_kex = sk;
        self
    }

    pub fn with_fips_mode(mut self, fips: bool) -> Self {
        self.fips_mode = fips;
        self
    }

    pub fn with_min_security(mut self, min: SecurityClassification) -> Self {
        self.min_security = min;
        self
    }

    /// Perform full negotiation between client and server KEXINIT messages.
    ///
    /// Returns the selected algorithms or an error if no common algorithm
    /// exists in any category.
    pub fn negotiate(
        &self,
        client: &KexInit,
        server: &KexInit,
    ) -> Result<NegotiatedAlgorithms, NegotiationFailure> {
        // 1. Key exchange algorithm (filter pseudo-algorithms)
        let client_kex: Vec<&str> = client
            .real_kex_algorithms()
            .into_iter()
            .collect();
        let server_kex: Vec<&str> = server
            .real_kex_algorithms()
            .into_iter()
            .collect();
        let kex = self
            .first_match(&client_kex, &server_kex)
            .ok_or_else(|| NegotiationFailure {
                category: "kex".into(),
                client_offered: client_kex.iter().map(|s| s.to_string()).collect(),
                server_offered: server_kex.iter().map(|s| s.to_string()).collect(),
            })?;

        // 2. Server host key algorithm
        let client_hk: Vec<&str> = client
            .server_host_key_algorithms
            .iter()
            .map(|s| s.as_str())
            .collect();
        let server_hk: Vec<&str> = server
            .server_host_key_algorithms
            .iter()
            .map(|s| s.as_str())
            .collect();
        let host_key =
            self.first_match(&client_hk, &server_hk)
                .ok_or_else(|| NegotiationFailure {
                    category: "host_key".into(),
                    client_offered: client_hk.iter().map(|s| s.to_string()).collect(),
                    server_offered: server_hk.iter().map(|s| s.to_string()).collect(),
                })?;

        // 3. Encryption c→s
        let client_enc_c2s: Vec<&str> = client
            .encryption_algorithms_client_to_server
            .iter()
            .map(|s| s.as_str())
            .collect();
        let server_enc_c2s: Vec<&str> = server
            .encryption_algorithms_client_to_server
            .iter()
            .map(|s| s.as_str())
            .collect();
        let enc_c2s =
            self.first_match(&client_enc_c2s, &server_enc_c2s)
                .ok_or_else(|| NegotiationFailure {
                    category: "encryption_c2s".into(),
                    client_offered: client_enc_c2s.iter().map(|s| s.to_string()).collect(),
                    server_offered: server_enc_c2s.iter().map(|s| s.to_string()).collect(),
                })?;

        // 4. Encryption s→c
        let client_enc_s2c: Vec<&str> = client
            .encryption_algorithms_server_to_client
            .iter()
            .map(|s| s.as_str())
            .collect();
        let server_enc_s2c: Vec<&str> = server
            .encryption_algorithms_server_to_client
            .iter()
            .map(|s| s.as_str())
            .collect();
        let enc_s2c =
            self.first_match(&client_enc_s2c, &server_enc_s2c)
                .ok_or_else(|| NegotiationFailure {
                    category: "encryption_s2c".into(),
                    client_offered: client_enc_s2c.iter().map(|s| s.to_string()).collect(),
                    server_offered: server_enc_s2c.iter().map(|s| s.to_string()).collect(),
                })?;

        // 5. MAC c→s
        let client_mac_c2s: Vec<&str> = client
            .mac_algorithms_client_to_server
            .iter()
            .map(|s| s.as_str())
            .collect();
        let server_mac_c2s: Vec<&str> = server
            .mac_algorithms_client_to_server
            .iter()
            .map(|s| s.as_str())
            .collect();
        let mac_c2s =
            self.first_match(&client_mac_c2s, &server_mac_c2s)
                .ok_or_else(|| NegotiationFailure {
                    category: "mac_c2s".into(),
                    client_offered: client_mac_c2s.iter().map(|s| s.to_string()).collect(),
                    server_offered: server_mac_c2s.iter().map(|s| s.to_string()).collect(),
                })?;

        // 6. MAC s→c
        let client_mac_s2c: Vec<&str> = client
            .mac_algorithms_server_to_client
            .iter()
            .map(|s| s.as_str())
            .collect();
        let server_mac_s2c: Vec<&str> = server
            .mac_algorithms_server_to_client
            .iter()
            .map(|s| s.as_str())
            .collect();
        let mac_s2c =
            self.first_match(&client_mac_s2c, &server_mac_s2c)
                .ok_or_else(|| NegotiationFailure {
                    category: "mac_s2c".into(),
                    client_offered: client_mac_s2c.iter().map(|s| s.to_string()).collect(),
                    server_offered: server_mac_s2c.iter().map(|s| s.to_string()).collect(),
                })?;

        // 7. Compression c→s
        let client_comp_c2s: Vec<&str> = client
            .compression_algorithms_client_to_server
            .iter()
            .map(|s| s.as_str())
            .collect();
        let server_comp_c2s: Vec<&str> = server
            .compression_algorithms_client_to_server
            .iter()
            .map(|s| s.as_str())
            .collect();
        let comp_c2s =
            self.first_match(&client_comp_c2s, &server_comp_c2s)
                .ok_or_else(|| NegotiationFailure {
                    category: "compression_c2s".into(),
                    client_offered: client_comp_c2s.iter().map(|s| s.to_string()).collect(),
                    server_offered: server_comp_c2s.iter().map(|s| s.to_string()).collect(),
                })?;

        // 8. Compression s→c
        let client_comp_s2c: Vec<&str> = client
            .compression_algorithms_server_to_client
            .iter()
            .map(|s| s.as_str())
            .collect();
        let server_comp_s2c: Vec<&str> = server
            .compression_algorithms_server_to_client
            .iter()
            .map(|s| s.as_str())
            .collect();
        let comp_s2c =
            self.first_match(&client_comp_s2c, &server_comp_s2c)
                .ok_or_else(|| NegotiationFailure {
                    category: "compression_s2c".into(),
                    client_offered: client_comp_s2c.iter().map(|s| s.to_string()).collect(),
                    server_offered: server_comp_s2c.iter().map(|s| s.to_string()).collect(),
                })?;

        Ok(NegotiatedAlgorithms {
            kex: kex.to_string(),
            host_key: host_key.to_string(),
            encryption_client_to_server: enc_c2s.to_string(),
            encryption_server_to_client: enc_s2c.to_string(),
            mac_client_to_server: mac_c2s.to_string(),
            mac_server_to_client: mac_s2c.to_string(),
            compression_client_to_server: comp_c2s.to_string(),
            compression_server_to_client: comp_s2c.to_string(),
        })
    }

    /// First-match-wins algorithm selection (RFC 4253 §7.1).
    ///
    /// Iterates through the client's list; the first entry that also appears
    /// in the server's list is selected.
    fn first_match<'a>(&self, client: &[&'a str], server: &[&'a str]) -> Option<&'a str> {
        for c in client {
            if server.contains(c) {
                return Some(c);
            }
        }
        None
    }

    /// Negotiate strict-KEX mode from both KEXINITs.
    pub fn negotiate_strict_kex(
        &mut self,
        client: &KexInit,
        server: &KexInit,
    ) -> bool {
        self.strict_kex.process_client_kexinit(client);
        self.strict_kex.process_server_kexinit(server);
        self.strict_kex.active
    }

    /// Check whether the client's guess is correct (first_kex_packet_follows).
    ///
    /// The guess is correct if the first algorithm in the client's kex list
    /// matches the first algorithm in the server's kex list AND the first
    /// host key algorithm also matches.
    pub fn check_guess(
        &self,
        client: &KexInit,
        server: &KexInit,
    ) -> GuessResult {
        if !client.first_kex_packet_follows {
            return GuessResult::NoGuess;
        }

        let client_kex_first = client.real_kex_algorithms().into_iter().next();
        let server_kex_first = server.real_kex_algorithms().into_iter().next();

        let client_hk_first = client.server_host_key_algorithms.first().map(|s| s.as_str());
        let server_hk_first = server.server_host_key_algorithms.first().map(|s| s.as_str());

        let kex_match = client_kex_first == server_kex_first;
        let hk_match = client_hk_first == server_hk_first;

        if kex_match && hk_match {
            GuessResult::Correct
        } else {
            GuessResult::Wrong {
                kex_mismatch: !kex_match,
                host_key_mismatch: !hk_match,
            }
        }
    }

    /// Cross-category consistency check.
    ///
    /// Verifies that the negotiated algorithms are consistent with each other.
    /// For example, if an AEAD cipher is selected, the MAC should be implicit
    /// (or "none").
    pub fn check_consistency(
        &self,
        negotiated: &NegotiatedAlgorithms,
    ) -> Vec<ConsistencyWarning> {
        let mut warnings = Vec::new();

        // Check AEAD + explicit MAC
        if negotiated.is_aead_c2s() && negotiated.mac_client_to_server != "none" {
            // Not really an error — SSH implementations ignore the MAC for AEAD ciphers
            // But it's unusual to negotiate a non-none MAC with AEAD
            if let Some(mac) = SshMac::from_wire_name(&negotiated.mac_client_to_server) {
                if mac.security().is_deprecated() {
                    warnings.push(ConsistencyWarning {
                        category: "mac_c2s".into(),
                        message: format!(
                            "AEAD cipher {} selected but weak MAC {} also negotiated \
                             (MAC is ignored for AEAD, but presence may indicate \
                             misconfiguration)",
                            negotiated.encryption_client_to_server,
                            negotiated.mac_client_to_server,
                        ),
                    });
                }
            }
        }

        if negotiated.is_aead_s2c() && negotiated.mac_server_to_client != "none" {
            if let Some(mac) = SshMac::from_wire_name(&negotiated.mac_server_to_client) {
                if mac.security().is_deprecated() {
                    warnings.push(ConsistencyWarning {
                        category: "mac_s2c".into(),
                        message: format!(
                            "AEAD cipher {} selected but weak MAC {} also negotiated",
                            negotiated.encryption_server_to_client,
                            negotiated.mac_server_to_client,
                        ),
                    });
                }
            }
        }

        // Check asymmetric encryption (c2s vs s2c mismatch — usually fine but worth noting)
        if negotiated.encryption_client_to_server != negotiated.encryption_server_to_client {
            // Check if security levels are very different
            let sec_c2s = EncryptionAlgorithm::from_wire_name(
                &negotiated.encryption_client_to_server,
            )
            .map(|e| e.security());
            let sec_s2c = EncryptionAlgorithm::from_wire_name(
                &negotiated.encryption_server_to_client,
            )
            .map(|e| e.security());

            if let (Some(sc), Some(ss)) = (sec_c2s, sec_s2c) {
                if sc.score().abs_diff(ss.score()) >= 2 {
                    warnings.push(ConsistencyWarning {
                        category: "encryption_asymmetry".into(),
                        message: format!(
                            "Asymmetric encryption: c2s={} ({}) vs s2c={} ({})",
                            negotiated.encryption_client_to_server,
                            sc,
                            negotiated.encryption_server_to_client,
                            ss,
                        ),
                    });
                }
            }
        }

        warnings
    }

    /// Detect possible downgrade attacks by comparing what COULD have been
    /// negotiated vs what WAS negotiated.
    pub fn detect_downgrade(
        &self,
        client: &KexInit,
        server: &KexInit,
        negotiated: &NegotiatedAlgorithms,
    ) -> Vec<DowngradeOpportunity> {
        let mut opportunities = Vec::new();

        // Check if a stronger KEX algorithm was available but not selected
        let real_client_kex = client.real_kex_algorithms();
        let real_server_kex: Vec<&str> = server
            .real_kex_algorithms()
            .into_iter()
            .collect();

        let best_kex = self.first_match(
            &real_client_kex.iter().map(|s| *s).collect::<Vec<_>>(),
            &real_server_kex,
        );

        if let Some(best) = best_kex {
            if best != negotiated.kex {
                let best_sec = KexAlgorithm::from_wire_name(best)
                    .map(|a| a.security())
                    .unwrap_or(SecurityClassification::Broken);
                let neg_sec = KexAlgorithm::from_wire_name(&negotiated.kex)
                    .map(|a| a.security())
                    .unwrap_or(SecurityClassification::Broken);

                if best_sec > neg_sec {
                    opportunities.push(DowngradeOpportunity {
                        category: "kex".into(),
                        expected: best.to_string(),
                        actual: negotiated.kex.clone(),
                        expected_security: best_sec,
                        actual_security: neg_sec,
                    });
                }
            }
        }

        // Check encryption c2s
        let client_enc: Vec<&str> = client
            .encryption_algorithms_client_to_server
            .iter()
            .map(|s| s.as_str())
            .collect();
        let server_enc: Vec<&str> = server
            .encryption_algorithms_client_to_server
            .iter()
            .map(|s| s.as_str())
            .collect();

        if let Some(best_enc) = self.first_match(&client_enc, &server_enc) {
            if best_enc != negotiated.encryption_client_to_server {
                let best_sec = EncryptionAlgorithm::from_wire_name(best_enc)
                    .map(|a| a.security())
                    .unwrap_or(SecurityClassification::Broken);
                let neg_sec = EncryptionAlgorithm::from_wire_name(
                    &negotiated.encryption_client_to_server,
                )
                .map(|a| a.security())
                .unwrap_or(SecurityClassification::Broken);

                if best_sec > neg_sec {
                    opportunities.push(DowngradeOpportunity {
                        category: "encryption_c2s".into(),
                        expected: best_enc.to_string(),
                        actual: negotiated.encryption_client_to_server.clone(),
                        expected_security: best_sec,
                        actual_security: neg_sec,
                    });
                }
            }
        }

        opportunities
    }
}

impl Default for SshNegotiationEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Supporting types
// ---------------------------------------------------------------------------

/// Result of checking whether the client's KEX guess was correct.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GuessResult {
    /// Client did not send a guess.
    NoGuess,
    /// Guess matches the negotiated algorithms.
    Correct,
    /// Guess was wrong — the guessed KEX packet must be ignored.
    Wrong {
        kex_mismatch: bool,
        host_key_mismatch: bool,
    },
}

impl GuessResult {
    pub fn is_correct(&self) -> bool {
        matches!(self, Self::Correct)
    }

    pub fn requires_discard(&self) -> bool {
        matches!(self, Self::Wrong { .. })
    }
}

/// A warning from the cross-category consistency check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyWarning {
    pub category: String,
    pub message: String,
}

impl fmt::Display for ConsistencyWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.category, self.message)
    }
}

/// A detected downgrade opportunity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DowngradeOpportunity {
    pub category: String,
    pub expected: String,
    pub actual: String,
    pub expected_security: SecurityClassification,
    pub actual_security: SecurityClassification,
}

impl fmt::Display for DowngradeOpportunity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Downgrade in {}: expected '{}' ({}) but got '{}' ({})",
            self.category,
            self.expected,
            self.expected_security,
            self.actual,
            self.actual_security,
        )
    }
}

/// Full negotiation analysis result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NegotiationAnalysis {
    pub negotiated: NegotiatedAlgorithms,
    pub strict_kex_active: bool,
    pub guess_result: GuessResult,
    pub consistency_warnings: Vec<ConsistencyWarning>,
    pub downgrade_opportunities: Vec<DowngradeOpportunity>,
    pub overall_security: SecurityClassification,
}

impl NegotiationAnalysis {
    /// Perform a complete negotiation analysis.
    pub fn analyze(
        client: &KexInit,
        server: &KexInit,
    ) -> Result<Self, NegotiationFailure> {
        let mut engine = SshNegotiationEngine::new();
        let strict_kex_active = engine.negotiate_strict_kex(client, server);
        let negotiated = engine.negotiate(client, server)?;
        let guess_result = engine.check_guess(client, server);
        let consistency_warnings = engine.check_consistency(&negotiated);
        let downgrade_opportunities =
            engine.detect_downgrade(client, server, &negotiated);
        let overall_security = negotiated.overall_security();

        Ok(Self {
            negotiated,
            strict_kex_active,
            guess_result,
            consistency_warnings,
            downgrade_opportunities,
            overall_security,
        })
    }
}

impl fmt::Display for NegotiationAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.negotiated)?;
        writeln!(f, "Strict-KEX: {}", self.strict_kex_active)?;
        writeln!(f, "Guess: {:?}", self.guess_result)?;
        if !self.consistency_warnings.is_empty() {
            writeln!(f, "Warnings:")?;
            for w in &self.consistency_warnings {
                writeln!(f, "  {}", w)?;
            }
        }
        if !self.downgrade_opportunities.is_empty() {
            writeln!(f, "Downgrade opportunities:")?;
            for d in &self.downgrade_opportunities {
                writeln!(f, "  {}", d)?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kex::{default_client_kex_init, default_server_kex_init, KexInitBuilder};

    #[test]
    fn basic_negotiation() {
        let client = default_client_kex_init();
        let server = default_server_kex_init();
        let engine = SshNegotiationEngine::new();

        let result = engine.negotiate(&client, &server).unwrap();
        assert_eq!(result.kex, "curve25519-sha256");
        assert_eq!(result.host_key, "ssh-ed25519");
        assert_eq!(
            result.encryption_client_to_server,
            "chacha20-poly1305@openssh.com"
        );
        assert_eq!(
            result.mac_client_to_server,
            "hmac-sha2-512-etm@openssh.com"
        );
        assert_eq!(result.compression_client_to_server, "none");
    }

    #[test]
    fn negotiation_selects_first_client_match() {
        // Client prefers aes256-ctr, server offers aes128-ctr first but also has aes256-ctr
        let client = KexInitBuilder::new()
            .kex_algorithms(vec!["curve25519-sha256".into()])
            .server_host_key_algorithms(vec!["ssh-ed25519".into()])
            .encryption_c2s(vec!["aes256-ctr".into(), "aes128-ctr".into()])
            .encryption_s2c(vec!["aes256-ctr".into(), "aes128-ctr".into()])
            .mac_c2s(vec!["hmac-sha2-256".into()])
            .mac_s2c(vec!["hmac-sha2-256".into()])
            .build();

        let server = KexInitBuilder::new()
            .kex_algorithms(vec!["curve25519-sha256".into()])
            .server_host_key_algorithms(vec!["ssh-ed25519".into()])
            .encryption_c2s(vec!["aes128-ctr".into(), "aes256-ctr".into()])
            .encryption_s2c(vec!["aes128-ctr".into(), "aes256-ctr".into()])
            .mac_c2s(vec!["hmac-sha2-256".into()])
            .mac_s2c(vec!["hmac-sha2-256".into()])
            .build();

        let engine = SshNegotiationEngine::new();
        let result = engine.negotiate(&client, &server).unwrap();

        // Client's first choice that server also supports
        assert_eq!(result.encryption_client_to_server, "aes256-ctr");
    }

    #[test]
    fn negotiation_failure_no_common_kex() {
        let client = KexInitBuilder::new()
            .kex_algorithms(vec!["curve25519-sha256".into()])
            .server_host_key_algorithms(vec!["ssh-ed25519".into()])
            .encryption_c2s(vec!["aes256-ctr".into()])
            .encryption_s2c(vec!["aes256-ctr".into()])
            .mac_c2s(vec!["hmac-sha2-256".into()])
            .mac_s2c(vec!["hmac-sha2-256".into()])
            .build();

        let server = KexInitBuilder::new()
            .kex_algorithms(vec!["diffie-hellman-group14-sha256".into()])
            .server_host_key_algorithms(vec!["ssh-ed25519".into()])
            .encryption_c2s(vec!["aes256-ctr".into()])
            .encryption_s2c(vec!["aes256-ctr".into()])
            .mac_c2s(vec!["hmac-sha2-256".into()])
            .mac_s2c(vec!["hmac-sha2-256".into()])
            .build();

        let engine = SshNegotiationEngine::new();
        let result = engine.negotiate(&client, &server);
        assert!(result.is_err());

        let failure = result.unwrap_err();
        assert_eq!(failure.category, "kex");
    }

    #[test]
    fn negotiation_failure_no_common_encryption() {
        let client = KexInitBuilder::new()
            .kex_algorithms(vec!["curve25519-sha256".into()])
            .server_host_key_algorithms(vec!["ssh-ed25519".into()])
            .encryption_c2s(vec!["aes256-gcm@openssh.com".into()])
            .encryption_s2c(vec!["aes256-gcm@openssh.com".into()])
            .mac_c2s(vec!["hmac-sha2-256".into()])
            .mac_s2c(vec!["hmac-sha2-256".into()])
            .build();

        let server = KexInitBuilder::new()
            .kex_algorithms(vec!["curve25519-sha256".into()])
            .server_host_key_algorithms(vec!["ssh-ed25519".into()])
            .encryption_c2s(vec!["aes128-ctr".into()])
            .encryption_s2c(vec!["aes128-ctr".into()])
            .mac_c2s(vec!["hmac-sha2-256".into()])
            .mac_s2c(vec!["hmac-sha2-256".into()])
            .build();

        let engine = SshNegotiationEngine::new();
        let result = engine.negotiate(&client, &server);
        assert!(result.is_err());
    }

    #[test]
    fn strict_kex_negotiation() {
        let client = KexInitBuilder::new()
            .kex_algorithms(vec!["curve25519-sha256".into()])
            .with_strict_kex_client()
            .build();
        let server = KexInitBuilder::new()
            .kex_algorithms(vec!["curve25519-sha256".into()])
            .with_strict_kex_server()
            .build();

        let mut engine = SshNegotiationEngine::new();
        let active = engine.negotiate_strict_kex(&client, &server);
        assert!(active);
    }

    #[test]
    fn strict_kex_partial_no_activation() {
        let client = KexInitBuilder::new()
            .kex_algorithms(vec!["curve25519-sha256".into()])
            .with_strict_kex_client()
            .build();
        let server = KexInitBuilder::new()
            .kex_algorithms(vec!["curve25519-sha256".into()])
            // no strict-kex server marker
            .build();

        let mut engine = SshNegotiationEngine::new();
        let active = engine.negotiate_strict_kex(&client, &server);
        assert!(!active);
    }

    #[test]
    fn guess_correct() {
        let client = KexInitBuilder::new()
            .kex_algorithms(vec!["curve25519-sha256".into()])
            .server_host_key_algorithms(vec!["ssh-ed25519".into()])
            .first_kex_packet_follows(true)
            .build();
        let server = KexInitBuilder::new()
            .kex_algorithms(vec!["curve25519-sha256".into()])
            .server_host_key_algorithms(vec!["ssh-ed25519".into()])
            .build();

        let engine = SshNegotiationEngine::new();
        let result = engine.check_guess(&client, &server);
        assert!(result.is_correct());
    }

    #[test]
    fn guess_wrong_kex() {
        let client = KexInitBuilder::new()
            .kex_algorithms(vec![
                "curve25519-sha256".into(),
                "diffie-hellman-group14-sha256".into(),
            ])
            .server_host_key_algorithms(vec!["ssh-ed25519".into()])
            .first_kex_packet_follows(true)
            .build();
        let server = KexInitBuilder::new()
            .kex_algorithms(vec!["diffie-hellman-group14-sha256".into()])
            .server_host_key_algorithms(vec!["ssh-ed25519".into()])
            .build();

        let engine = SshNegotiationEngine::new();
        let result = engine.check_guess(&client, &server);
        assert!(result.requires_discard());
    }

    #[test]
    fn no_guess() {
        let client = default_client_kex_init();
        let server = default_server_kex_init();
        let engine = SshNegotiationEngine::new();
        let result = engine.check_guess(&client, &server);
        assert!(matches!(result, GuessResult::NoGuess));
    }

    #[test]
    fn overall_security() {
        let client = default_client_kex_init();
        let server = default_server_kex_init();
        let engine = SshNegotiationEngine::new();

        let negotiated = engine.negotiate(&client, &server).unwrap();
        let sec = negotiated.overall_security();
        assert!(sec.is_safe());
    }

    #[test]
    fn weak_algorithms_detection() {
        let negotiated = NegotiatedAlgorithms {
            kex: "diffie-hellman-group1-sha1".into(),
            host_key: "ssh-dss".into(),
            encryption_client_to_server: "aes128-cbc".into(),
            encryption_server_to_client: "aes128-cbc".into(),
            mac_client_to_server: "hmac-md5".into(),
            mac_server_to_client: "hmac-sha2-256".into(),
            compression_client_to_server: "none".into(),
            compression_server_to_client: "none".into(),
        };

        let weak = negotiated.weak_algorithms();
        assert!(weak.len() >= 3, "should have multiple weak algorithms");
        assert!(negotiated.has_deprecated());
    }

    #[test]
    fn detect_downgrade_opportunity() {
        let client = KexInitBuilder::new()
            .kex_algorithms(vec![
                "curve25519-sha256".into(),
                "diffie-hellman-group1-sha1".into(),
            ])
            .server_host_key_algorithms(vec!["ssh-ed25519".into()])
            .encryption_c2s(vec!["aes256-gcm@openssh.com".into()])
            .encryption_s2c(vec!["aes256-gcm@openssh.com".into()])
            .mac_c2s(vec!["hmac-sha2-256".into()])
            .mac_s2c(vec!["hmac-sha2-256".into()])
            .build();

        let server = KexInitBuilder::new()
            .kex_algorithms(vec![
                "curve25519-sha256".into(),
                "diffie-hellman-group1-sha1".into(),
            ])
            .server_host_key_algorithms(vec!["ssh-ed25519".into()])
            .encryption_c2s(vec!["aes256-gcm@openssh.com".into()])
            .encryption_s2c(vec!["aes256-gcm@openssh.com".into()])
            .mac_c2s(vec!["hmac-sha2-256".into()])
            .mac_s2c(vec!["hmac-sha2-256".into()])
            .build();

        // Simulate a downgrade: negotiated is group1 instead of curve25519
        let negotiated = NegotiatedAlgorithms {
            kex: "diffie-hellman-group1-sha1".into(),
            host_key: "ssh-ed25519".into(),
            encryption_client_to_server: "aes256-gcm@openssh.com".into(),
            encryption_server_to_client: "aes256-gcm@openssh.com".into(),
            mac_client_to_server: "hmac-sha2-256".into(),
            mac_server_to_client: "hmac-sha2-256".into(),
            compression_client_to_server: "none".into(),
            compression_server_to_client: "none".into(),
        };

        let engine = SshNegotiationEngine::new();
        let downgrades = engine.detect_downgrade(&client, &server, &negotiated);
        assert!(!downgrades.is_empty());
        assert_eq!(downgrades[0].category, "kex");
    }

    #[test]
    fn full_analysis() {
        let client = default_client_kex_init();
        let server = default_server_kex_init();

        let analysis = NegotiationAnalysis::analyze(&client, &server).unwrap();
        assert!(analysis.strict_kex_active);
        assert_eq!(analysis.negotiated.kex, "curve25519-sha256");
        assert!(analysis.overall_security.is_safe());
        assert!(analysis.downgrade_opportunities.is_empty());
    }

    #[test]
    fn analysis_display() {
        let client = default_client_kex_init();
        let server = default_server_kex_init();

        let analysis = NegotiationAnalysis::analyze(&client, &server).unwrap();
        let s = format!("{}", analysis);
        assert!(s.contains("curve25519-sha256"));
        assert!(s.contains("Strict-KEX"));
    }

    #[test]
    fn negotiated_aead_detection() {
        let negotiated = NegotiatedAlgorithms {
            kex: "curve25519-sha256".into(),
            host_key: "ssh-ed25519".into(),
            encryption_client_to_server: "aes256-gcm@openssh.com".into(),
            encryption_server_to_client: "chacha20-poly1305@openssh.com".into(),
            mac_client_to_server: "hmac-sha2-256".into(),
            mac_server_to_client: "hmac-sha2-256".into(),
            compression_client_to_server: "none".into(),
            compression_server_to_client: "none".into(),
        };

        assert!(negotiated.is_aead_c2s());
        assert!(negotiated.is_aead_s2c());
    }

    #[test]
    fn negotiation_failure_display() {
        let failure = NegotiationFailure {
            category: "kex".into(),
            client_offered: vec!["curve25519-sha256".into()],
            server_offered: vec!["diffie-hellman-group14-sha256".into()],
        };
        let s = format!("{}", failure);
        assert!(s.contains("kex"));
        assert!(s.contains("curve25519"));
    }

    #[test]
    fn downgrade_display() {
        let d = DowngradeOpportunity {
            category: "kex".into(),
            expected: "curve25519-sha256".into(),
            actual: "diffie-hellman-group1-sha1".into(),
            expected_security: SecurityClassification::Recommended,
            actual_security: SecurityClassification::Broken,
        };
        let s = format!("{}", d);
        assert!(s.contains("Downgrade"));
        assert!(s.contains("curve25519"));
    }
}
