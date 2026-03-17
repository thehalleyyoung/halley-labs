//! Cipher suite preference lattice implementing Axiom A2.
//!
//! Defines a partial order on cipher suites based on security strength of
//! each component (key exchange, authentication, encryption, MAC). The lattice
//! enables monotone cipher selection functions and join/meet operations.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use negsyn_types::{
    AuthAlgorithm, BulkEncryption, CipherSuite, KeyExchange, MacAlgorithm, MergeError,
    SecurityLevel as NtSecurityLevel,
};

// ---------------------------------------------------------------------------
// Component strength enums
// ---------------------------------------------------------------------------

/// Overall security level classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SecurityLevel {
    Export,
    Low,
    Medium,
    High,
    Maximum,
}

impl SecurityLevel {
    pub fn numeric(&self) -> u32 {
        match self {
            Self::Export => 0,
            Self::Low => 1,
            Self::Medium => 2,
            Self::High => 3,
            Self::Maximum => 4,
        }
    }

    pub fn from_numeric(n: u32) -> Self {
        match n {
            0 => Self::Export,
            1 => Self::Low,
            2 => Self::Medium,
            3 => Self::High,
            _ => Self::Maximum,
        }
    }

    pub fn all() -> &'static [SecurityLevel] {
        &[
            Self::Export,
            Self::Low,
            Self::Medium,
            Self::High,
            Self::Maximum,
        ]
    }
}

impl PartialOrd for SecurityLevel {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SecurityLevel {
    fn cmp(&self, other: &Self) -> Ordering {
        self.numeric().cmp(&other.numeric())
    }
}

impl fmt::Display for SecurityLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Strength of key exchange algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum KeyExchangeStrength {
    None,
    Static,
    Ephemeral,
    EphemeralEcc,
}

impl KeyExchangeStrength {
    pub fn from_algorithm(kx: &KeyExchange) -> Self {
        match kx {
            KeyExchange::NULL => Self::None,
            KeyExchange::Rsa | KeyExchange::RSA | KeyExchange::Psk | KeyExchange::PSK
            | KeyExchange::Kerberos | KeyExchange::SRP => Self::Static,
            KeyExchange::Dhe | KeyExchange::DHE | KeyExchange::DHEPSK => Self::Ephemeral,
            KeyExchange::Ecdhe | KeyExchange::ECDHE | KeyExchange::ECDHEPSK => Self::EphemeralEcc,
        }
    }

    pub fn numeric(&self) -> u32 {
        match self {
            Self::None => 0,
            Self::Static => 1,
            Self::Ephemeral => 2,
            Self::EphemeralEcc => 3,
        }
    }

    pub fn provides_forward_secrecy(&self) -> bool {
        matches!(self, Self::Ephemeral | Self::EphemeralEcc)
    }
}

/// Strength of authentication algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AuthStrength {
    None,
    Psk,
    Rsa,
    Dss,
    Ecdsa,
    Ed25519,
}

impl AuthStrength {
    pub fn from_algorithm(auth: &AuthAlgorithm) -> Self {
        match auth {
            AuthAlgorithm::NULL | AuthAlgorithm::Anon => Self::None,
            AuthAlgorithm::PSK => Self::Psk,
            AuthAlgorithm::Rsa | AuthAlgorithm::RSA => Self::Rsa,
            AuthAlgorithm::Dss | AuthAlgorithm::DSS => Self::Dss,
            AuthAlgorithm::Ecdsa | AuthAlgorithm::ECDSA => Self::Ecdsa,
            AuthAlgorithm::SHA256 | AuthAlgorithm::SHA384 => Self::Rsa,
        }
    }

    pub fn numeric(&self) -> u32 {
        match self {
            Self::None => 0,
            Self::Psk => 1,
            Self::Rsa => 2,
            Self::Dss => 2,
            Self::Ecdsa => 3,
            Self::Ed25519 => 3,
        }
    }
}

/// Strength of bulk encryption algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EncryptionStrength {
    None,
    Export,
    Weak,
    Medium,
    Strong,
    Aead,
}

impl EncryptionStrength {
    pub fn from_algorithm(enc: &BulkEncryption) -> Self {
        match enc {
            BulkEncryption::NULL => Self::None,
            BulkEncryption::DESCBC | BulkEncryption::Des => Self::Export,
            BulkEncryption::RC4_128 | BulkEncryption::Rc4 => Self::Weak,
            BulkEncryption::TripleDes | BulkEncryption::TripleDESCBC
            | BulkEncryption::Aes128 | BulkEncryption::AES128CBC | BulkEncryption::Aes128Cbc
            | BulkEncryption::AES128CCM
            | BulkEncryption::Camellia128CBC | BulkEncryption::SEED_CBC
            | BulkEncryption::ARIA128GCM => Self::Medium,
            BulkEncryption::Aes256 | BulkEncryption::AES256CBC | BulkEncryption::Aes256Cbc
            | BulkEncryption::AES256CCM
            | BulkEncryption::Camellia256CBC
            | BulkEncryption::ARIA256GCM => Self::Strong,
            BulkEncryption::Aes128Gcm | BulkEncryption::AES128GCM
            | BulkEncryption::Aes256Gcm | BulkEncryption::AES256GCM
            | BulkEncryption::ChaCha20Poly1305
            | BulkEncryption::Camellia128GCM | BulkEncryption::Camellia256GCM => Self::Aead,
        }
    }

    pub fn numeric(&self) -> u32 {
        match self {
            Self::None => 0,
            Self::Export => 1,
            Self::Weak => 2,
            Self::Medium => 3,
            Self::Strong => 4,
            Self::Aead => 5,
        }
    }

    pub fn key_bits(&self) -> u32 {
        match self {
            Self::None => 0,
            Self::Export => 40,
            Self::Weak => 64,
            Self::Medium => 128,
            Self::Strong => 256,
            Self::Aead => 256,
        }
    }
}

/// Strength of MAC algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MacStrength {
    None,
    Md5,
    Sha1,
    Sha256,
    Sha384,
    Aead,
}

impl MacStrength {
    pub fn from_algorithm(mac: &MacAlgorithm) -> Self {
        match mac {
            MacAlgorithm::NULL => Self::None,
            MacAlgorithm::MD5 => Self::Md5,
            MacAlgorithm::HmacSha1 | MacAlgorithm::SHA1 => Self::Sha1,
            MacAlgorithm::HmacSha256 | MacAlgorithm::SHA256 => Self::Sha256,
            MacAlgorithm::HmacSha384 | MacAlgorithm::SHA384 => Self::Sha384,
            MacAlgorithm::Aead | MacAlgorithm::AEAD => Self::Aead,
        }
    }

    pub fn numeric(&self) -> u32 {
        match self {
            Self::None => 0,
            Self::Md5 => 1,
            Self::Sha1 => 2,
            Self::Sha256 => 3,
            Self::Sha384 => 4,
            Self::Aead => 5,
        }
    }
}

// ---------------------------------------------------------------------------
// Composite strength profile
// ---------------------------------------------------------------------------

/// Full security profile of a cipher suite.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SecurityProfile {
    pub kx_strength: KeyExchangeStrength,
    pub auth_strength: AuthStrength,
    pub enc_strength: EncryptionStrength,
    pub mac_strength: MacStrength,
    pub overall: SecurityLevel,
    pub is_fips: bool,
}

impl SecurityProfile {
    pub fn from_cipher_suite(cs: &CipherSuite) -> Self {
        let kx = KeyExchangeStrength::from_algorithm(&cs.key_exchange);
        let auth = AuthStrength::from_algorithm(&cs.authentication);
        let enc = EncryptionStrength::from_algorithm(&cs.encryption);
        let mac = MacStrength::from_algorithm(&cs.mac);

        let min_component = kx
            .numeric()
            .min(auth.numeric().max(1)) // auth=none is only bad if kx also bad
            .min(enc.numeric())
            .min(mac.numeric().max(1));

        let overall = SecurityLevel::from_numeric(min_component.min(4));

        Self {
            kx_strength: kx,
            auth_strength: auth,
            enc_strength: enc,
            mac_strength: mac,
            overall,
            is_fips: cs.is_fips_approved,
        }
    }

    /// Composite numeric score for total ordering when partial order is insufficient.
    pub fn composite_score(&self) -> u64 {
        (self.kx_strength.numeric() as u64) * 10000
            + (self.auth_strength.numeric() as u64) * 1000
            + (self.enc_strength.numeric() as u64) * 100
            + (self.mac_strength.numeric() as u64) * 10
            + if self.is_fips { 5 } else { 0 }
    }
}

impl PartialOrd for SecurityProfile {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let kx = self.kx_strength.cmp(&other.kx_strength);
        let auth = self.auth_strength.cmp(&other.auth_strength);
        let enc = self.enc_strength.cmp(&other.enc_strength);
        let mac = self.mac_strength.cmp(&other.mac_strength);

        // Partial order: a <= b iff every component of a <= corresponding component of b
        let components = [kx, auth, enc, mac];
        let has_less = components.iter().any(|c| *c == Ordering::Less);
        let has_greater = components.iter().any(|c| *c == Ordering::Greater);

        match (has_less, has_greater) {
            (false, false) => Some(Ordering::Equal),
            (true, false) => Some(Ordering::Less),
            (false, true) => Some(Ordering::Greater),
            (true, true) => None, // incomparable
        }
    }
}

// ---------------------------------------------------------------------------
// Security lattice
// ---------------------------------------------------------------------------

/// Lattice structure on cipher suites based on security strength.
///
/// Implements the partial order from Axiom A2, where cipher suites are
/// ordered by the component-wise comparison of their security profiles.
pub struct SecurityLattice {
    profiles: IndexMap<u16, SecurityProfile>,
    suites: IndexMap<u16, CipherSuite>,
    fips_override: bool,
}

impl SecurityLattice {
    pub fn new() -> Self {
        Self {
            profiles: IndexMap::new(),
            suites: IndexMap::new(),
            fips_override: false,
        }
    }

    pub fn with_fips_override(mut self) -> Self {
        self.fips_override = true;
        self
    }

    pub fn register_suite(&mut self, suite: CipherSuite) {
        let profile = SecurityProfile::from_cipher_suite(&suite);
        self.profiles.insert(suite.iana_id, profile);
        self.suites.insert(suite.iana_id, suite);
    }

    pub fn register_suites(&mut self, suites: impl IntoIterator<Item = CipherSuite>) {
        for suite in suites {
            self.register_suite(suite);
        }
    }

    pub fn profile(&self, suite_id: u16) -> Option<&SecurityProfile> {
        self.profiles.get(&suite_id)
    }

    pub fn suite(&self, suite_id: u16) -> Option<&CipherSuite> {
        self.suites.get(&suite_id)
    }

    /// Compare two cipher suites in the security partial order.
    pub fn compare(&self, a: u16, b: u16) -> Option<Ordering> {
        let pa = self.profiles.get(&a)?;
        let pb = self.profiles.get(&b)?;

        if self.fips_override {
            match (pa.is_fips, pb.is_fips) {
                (true, false) => return Some(Ordering::Greater),
                (false, true) => return Some(Ordering::Less),
                _ => {}
            }
        }

        pa.partial_cmp(pb)
    }

    /// Compute the join (least upper bound) of two cipher suite IDs.
    /// Returns the stronger of the two if comparable, or the one with higher composite score.
    pub fn join(&self, a: u16, b: u16) -> Option<u16> {
        let pa = self.profiles.get(&a)?;
        let pb = self.profiles.get(&b)?;

        match pa.partial_cmp(pb) {
            Some(Ordering::Greater | Ordering::Equal) => Some(a),
            Some(Ordering::Less) => Some(b),
            None => {
                // Incomparable: pick the one with higher composite score
                if pa.composite_score() >= pb.composite_score() {
                    Some(a)
                } else {
                    Some(b)
                }
            }
        }
    }

    /// Compute the meet (greatest lower bound) of two cipher suite IDs.
    pub fn meet(&self, a: u16, b: u16) -> Option<u16> {
        let pa = self.profiles.get(&a)?;
        let pb = self.profiles.get(&b)?;

        match pa.partial_cmp(pb) {
            Some(Ordering::Less | Ordering::Equal) => Some(a),
            Some(Ordering::Greater) => Some(b),
            None => {
                if pa.composite_score() <= pb.composite_score() {
                    Some(a)
                } else {
                    Some(b)
                }
            }
        }
    }

    /// Get the top element (strongest cipher) among registered suites.
    pub fn top(&self) -> Option<u16> {
        self.profiles
            .iter()
            .max_by(|(_, pa), (_, pb)| {
                pa.composite_score().cmp(&pb.composite_score())
            })
            .map(|(id, _)| *id)
    }

    /// Get the bottom element (weakest cipher) among registered suites.
    pub fn bottom(&self) -> Option<u16> {
        self.profiles
            .iter()
            .min_by(|(_, pa), (_, pb)| {
                pa.composite_score().cmp(&pb.composite_score())
            })
            .map(|(id, _)| *id)
    }

    /// Check if `a` is strictly weaker than `b`.
    pub fn is_weaker(&self, a: u16, b: u16) -> bool {
        self.compare(a, b) == Some(Ordering::Less)
    }

    /// Check if `a` is at least as strong as `b`.
    pub fn is_at_least_as_strong(&self, a: u16, b: u16) -> bool {
        matches!(
            self.compare(a, b),
            Some(Ordering::Greater | Ordering::Equal)
        )
    }

    /// Check if the registered suites form a valid lattice (every pair has join and meet).
    pub fn is_valid_lattice(&self) -> bool {
        let ids: Vec<u16> = self.profiles.keys().copied().collect();
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                if self.join(ids[i], ids[j]).is_none() || self.meet(ids[i], ids[j]).is_none() {
                    return false;
                }
            }
        }
        true
    }

    /// Returns all registered suite IDs sorted by composite score descending.
    pub fn suites_by_strength(&self) -> Vec<u16> {
        let mut ids: Vec<(u16, u64)> = self
            .profiles
            .iter()
            .map(|(id, p)| (*id, p.composite_score()))
            .collect();
        ids.sort_by(|a, b| b.1.cmp(&a.1));
        ids.into_iter().map(|(id, _)| id).collect()
    }

    /// Filter suites to those meeting a minimum security level.
    pub fn filter_by_level(&self, min_level: SecurityLevel) -> Vec<u16> {
        self.profiles
            .iter()
            .filter(|(_, p)| p.overall >= min_level)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Filter to FIPS-approved suites only.
    pub fn fips_suites(&self) -> Vec<u16> {
        self.profiles
            .iter()
            .filter(|(_, p)| p.is_fips)
            .map(|(id, _)| *id)
            .collect()
    }

    pub fn suite_count(&self) -> usize {
        self.suites.len()
    }

    /// Build from a standard set of well-known cipher suites (IANA registry subset).
    pub fn from_standard_registry() -> Self {
        let mut lattice = Self::new();
        lattice.register_suites(standard_cipher_suites());
        lattice
    }
}

impl Default for SecurityLattice {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Preference lattice
// ---------------------------------------------------------------------------

/// Combines security ordering with explicit preference ordering.
///
/// While `SecurityLattice` captures the objective security partial order,
/// `PreferenceLattice` layers on server or client preference lists to
/// produce a total order for cipher selection.
pub struct PreferenceLattice {
    security: SecurityLattice,
    preference_order: Vec<u16>,
    preference_rank: HashMap<u16, usize>,
}

impl PreferenceLattice {
    pub fn new(security: SecurityLattice, preference_order: Vec<u16>) -> Self {
        let preference_rank: HashMap<u16, usize> = preference_order
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, i))
            .collect();
        Self {
            security,
            preference_order,
            preference_rank,
        }
    }

    /// Compare two suites: first by security lattice, breaking ties by preference rank.
    pub fn compare(&self, a: u16, b: u16) -> Ordering {
        match self.security.compare(a, b) {
            Some(ord) if ord != Ordering::Equal => ord,
            _ => {
                let ra = self.preference_rank.get(&a).copied().unwrap_or(usize::MAX);
                let rb = self.preference_rank.get(&b).copied().unwrap_or(usize::MAX);
                // Lower rank = higher preference = Greater
                rb.cmp(&ra)
            }
        }
    }

    /// Join: pick the more-preferred of two suites.
    pub fn join(&self, a: u16, b: u16) -> u16 {
        match self.compare(a, b) {
            Ordering::Greater | Ordering::Equal => a,
            Ordering::Less => b,
        }
    }

    /// Meet: pick the less-preferred of two suites.
    pub fn meet(&self, a: u16, b: u16) -> u16 {
        match self.compare(a, b) {
            Ordering::Less | Ordering::Equal => a,
            Ordering::Greater => b,
        }
    }

    pub fn top(&self) -> Option<u16> {
        self.preference_order.first().copied()
    }

    pub fn bottom(&self) -> Option<u16> {
        self.preference_order.last().copied()
    }

    /// Select the best cipher from offered set according to preference.
    pub fn select_best(&self, offered: &BTreeSet<u16>) -> Option<u16> {
        self.preference_order
            .iter()
            .find(|id| offered.contains(id))
            .copied()
    }

    /// Select the best cipher meeting a minimum security level.
    pub fn select_best_at_level(
        &self,
        offered: &BTreeSet<u16>,
        min_level: SecurityLevel,
    ) -> Option<u16> {
        self.preference_order
            .iter()
            .filter(|id| offered.contains(id))
            .filter(|id| {
                self.security
                    .profile(**id)
                    .map(|p| p.overall >= min_level)
                    .unwrap_or(false)
            })
            .next()
            .copied()
    }

    /// Get all suites in preference order that are in the offered set.
    pub fn filter_and_sort(&self, offered: &BTreeSet<u16>) -> Vec<u16> {
        self.preference_order
            .iter()
            .filter(|id| offered.contains(id))
            .copied()
            .collect()
    }

    pub fn security_lattice(&self) -> &SecurityLattice {
        &self.security
    }

    pub fn preference_rank(&self, suite_id: u16) -> Option<usize> {
        self.preference_rank.get(&suite_id).copied()
    }
}

// ---------------------------------------------------------------------------
// Selection function
// ---------------------------------------------------------------------------

/// Monotone cipher selection function (per Axiom A2).
///
/// Given an offered set of cipher suites and a preference lattice,
/// deterministically selects a single cipher suite such that:
/// - If offered set A ⊆ B, then select(A) ≤ select(B) in the lattice
pub struct SelectionFunction {
    lattice: PreferenceLattice,
    fips_required: bool,
    min_level: SecurityLevel,
    blocked_suites: BTreeSet<u16>,
}

impl SelectionFunction {
    pub fn new(lattice: PreferenceLattice) -> Self {
        Self {
            lattice,
            fips_required: false,
            min_level: SecurityLevel::Export,
            blocked_suites: BTreeSet::new(),
        }
    }

    pub fn with_fips_required(mut self, fips: bool) -> Self {
        self.fips_required = fips;
        self
    }

    pub fn with_min_level(mut self, level: SecurityLevel) -> Self {
        self.min_level = level;
        self
    }

    pub fn with_blocked(mut self, blocked: BTreeSet<u16>) -> Self {
        self.blocked_suites = blocked;
        self
    }

    /// Select a cipher suite from the offered set.
    pub fn select(&self, offered: &BTreeSet<u16>) -> Option<u16> {
        let filtered: BTreeSet<u16> = offered
            .iter()
            .copied()
            .filter(|id| !self.blocked_suites.contains(id))
            .filter(|id| {
                if self.fips_required {
                    self.lattice
                        .security_lattice()
                        .profile(*id)
                        .map(|p| p.is_fips)
                        .unwrap_or(false)
                } else {
                    true
                }
            })
            .collect();

        self.lattice.select_best_at_level(&filtered, self.min_level)
    }

    /// Check monotonicity: for all A ⊆ B, select(A) ≤ select(B).
    /// Tests with random subsets for practical verification.
    pub fn verify_monotonicity(&self, universe: &BTreeSet<u16>) -> bool {
        let elems: Vec<u16> = universe.iter().copied().collect();
        let n = elems.len();
        if n > 20 {
            return self.verify_monotonicity_sampled(universe, 1000);
        }

        // Exhaustive check for small sets
        let subsets = 1u64 << n;
        for mask_a in 0..subsets {
            let set_a: BTreeSet<u16> = (0..n)
                .filter(|i| mask_a & (1 << i) != 0)
                .map(|i| elems[i])
                .collect();
            let sel_a = self.select(&set_a);

            for mask_b in mask_a..subsets {
                if mask_a & mask_b != mask_a {
                    continue; // set_a is not a subset of set_b
                }
                let set_b: BTreeSet<u16> = (0..n)
                    .filter(|i| mask_b & (1 << i) != 0)
                    .map(|i| elems[i])
                    .collect();
                let sel_b = self.select(&set_b);

                match (sel_a, sel_b) {
                    (Some(a), Some(b)) => {
                        let cmp = self.lattice.compare(a, b);
                        if cmp == Ordering::Greater {
                            return false;
                        }
                    }
                    (Some(_), None) => return false, // superset should also select something
                    _ => {}
                }
            }
        }
        true
    }

    fn verify_monotonicity_sampled(&self, universe: &BTreeSet<u16>, samples: usize) -> bool {
        let elems: Vec<u16> = universe.iter().copied().collect();
        let n = elems.len();

        // Use deterministic "random" sampling based on index
        for sample_idx in 0..samples {
            let seed = sample_idx as u64;
            let mask_a = seed % (1u64 << n.min(63));
            let extra_bits = (seed.wrapping_mul(2654435761)) % (1u64 << n.min(63));
            let mask_b = mask_a | extra_bits;

            let set_a: BTreeSet<u16> = (0..n)
                .filter(|i| mask_a & (1 << (*i).min(62)) != 0)
                .map(|i| elems[i])
                .collect();

            let set_b: BTreeSet<u16> = (0..n)
                .filter(|i| mask_b & (1 << (*i).min(62)) != 0)
                .map(|i| elems[i])
                .collect();

            let sel_a = self.select(&set_a);
            let sel_b = self.select(&set_b);

            match (sel_a, sel_b) {
                (Some(a), Some(b)) => {
                    if self.lattice.compare(a, b) == Ordering::Greater {
                        return false;
                    }
                }
                (Some(_), None) => return false,
                _ => {}
            }
        }
        true
    }

    pub fn preference_lattice(&self) -> &PreferenceLattice {
        &self.lattice
    }
}

// ---------------------------------------------------------------------------
// Standard cipher suite registry
// ---------------------------------------------------------------------------

/// Build a set of well-known cipher suites modeled after the IANA TLS registry.
pub fn standard_cipher_suites() -> Vec<CipherSuite> {
    vec![
        CipherSuite::new(
            0x002F,
            "TLS_RSA_WITH_AES_128_CBC_SHA",
            KeyExchange::Rsa,
            AuthAlgorithm::Rsa,
            BulkEncryption::Aes128,
            MacAlgorithm::HmacSha1,
            NtSecurityLevel::Legacy,
        ),
        CipherSuite::new(
            0x0035,
            "TLS_RSA_WITH_AES_256_CBC_SHA",
            KeyExchange::Rsa,
            AuthAlgorithm::Rsa,
            BulkEncryption::Aes256,
            MacAlgorithm::HmacSha1,
            NtSecurityLevel::Legacy,
        ),
        CipherSuite::new(
            0x003C,
            "TLS_RSA_WITH_AES_128_CBC_SHA256",
            KeyExchange::Rsa,
            AuthAlgorithm::Rsa,
            BulkEncryption::Aes128,
            MacAlgorithm::HmacSha256,
            NtSecurityLevel::Legacy,
        ),
        CipherSuite::new(
            0x003D,
            "TLS_RSA_WITH_AES_256_CBC_SHA256",
            KeyExchange::Rsa,
            AuthAlgorithm::Rsa,
            BulkEncryption::Aes256,
            MacAlgorithm::HmacSha256,
            NtSecurityLevel::Legacy,
        ),
        CipherSuite::new(
            0x009C,
            "TLS_RSA_WITH_AES_128_GCM_SHA256",
            KeyExchange::Rsa,
            AuthAlgorithm::Rsa,
            BulkEncryption::Aes128Gcm,
            MacAlgorithm::Aead,
            NtSecurityLevel::Standard,
        ),
        CipherSuite::new(
            0x009D,
            "TLS_RSA_WITH_AES_256_GCM_SHA384",
            KeyExchange::Rsa,
            AuthAlgorithm::Rsa,
            BulkEncryption::Aes256Gcm,
            MacAlgorithm::Aead,
            NtSecurityLevel::Standard,
        ),
        CipherSuite::new(
            0xC013,
            "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA",
            KeyExchange::Ecdhe,
            AuthAlgorithm::Rsa,
            BulkEncryption::Aes128,
            MacAlgorithm::HmacSha1,
            NtSecurityLevel::Legacy,
        ),
        CipherSuite::new(
            0xC014,
            "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA",
            KeyExchange::Ecdhe,
            AuthAlgorithm::Rsa,
            BulkEncryption::Aes256,
            MacAlgorithm::HmacSha1,
            NtSecurityLevel::Legacy,
        ),
        CipherSuite::new(
            0xC027,
            "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256",
            KeyExchange::Ecdhe,
            AuthAlgorithm::Rsa,
            BulkEncryption::Aes128,
            MacAlgorithm::HmacSha256,
            NtSecurityLevel::Legacy,
        ),
        CipherSuite::new(
            0xC02F,
            "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
            KeyExchange::Ecdhe,
            AuthAlgorithm::Rsa,
            BulkEncryption::Aes128Gcm,
            MacAlgorithm::Aead,
            NtSecurityLevel::Standard,
        ),
        CipherSuite::new(
            0xC030,
            "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
            KeyExchange::Ecdhe,
            AuthAlgorithm::Rsa,
            BulkEncryption::Aes256Gcm,
            MacAlgorithm::Aead,
            NtSecurityLevel::High,
        ),
        CipherSuite::new(
            0xCCA8,
            "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256",
            KeyExchange::Ecdhe,
            AuthAlgorithm::Rsa,
            BulkEncryption::ChaCha20Poly1305,
            MacAlgorithm::Aead,
            NtSecurityLevel::High,
        ),
        CipherSuite::new(
            0xC009,
            "TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA",
            KeyExchange::Ecdhe,
            AuthAlgorithm::Ecdsa,
            BulkEncryption::Aes128,
            MacAlgorithm::HmacSha1,
            NtSecurityLevel::Legacy,
        ),
        CipherSuite::new(
            0xC02B,
            "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256",
            KeyExchange::Ecdhe,
            AuthAlgorithm::Ecdsa,
            BulkEncryption::Aes128Gcm,
            MacAlgorithm::Aead,
            NtSecurityLevel::Standard,
        ),
        CipherSuite::new(
            0xC02C,
            "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384",
            KeyExchange::Ecdhe,
            AuthAlgorithm::Ecdsa,
            BulkEncryption::Aes256Gcm,
            MacAlgorithm::Aead,
            NtSecurityLevel::High,
        ),
        CipherSuite::new(
            0x0033,
            "TLS_DHE_RSA_WITH_AES_128_CBC_SHA",
            KeyExchange::Dhe,
            AuthAlgorithm::Rsa,
            BulkEncryption::Aes128,
            MacAlgorithm::HmacSha1,
            NtSecurityLevel::Legacy,
        ),
        CipherSuite::new(
            0x0039,
            "TLS_DHE_RSA_WITH_AES_256_CBC_SHA",
            KeyExchange::Dhe,
            AuthAlgorithm::Rsa,
            BulkEncryption::Aes256,
            MacAlgorithm::HmacSha1,
            NtSecurityLevel::Legacy,
        ),
        CipherSuite::new(
            0x009E,
            "TLS_DHE_RSA_WITH_AES_128_GCM_SHA256",
            KeyExchange::Dhe,
            AuthAlgorithm::Rsa,
            BulkEncryption::Aes128Gcm,
            MacAlgorithm::Aead,
            NtSecurityLevel::Standard,
        ),
        CipherSuite::new(
            0x0003,
            "TLS_RSA_EXPORT_WITH_RC4_40_MD5",
            KeyExchange::Rsa,
            AuthAlgorithm::Rsa,
            BulkEncryption::DESCBC,
            MacAlgorithm::MD5,
            NtSecurityLevel::Broken,
        ),
        CipherSuite::new(
            0x000A,
            "TLS_RSA_WITH_3DES_EDE_CBC_SHA",
            KeyExchange::Rsa,
            AuthAlgorithm::Rsa,
            BulkEncryption::TripleDes,
            MacAlgorithm::HmacSha1,
            NtSecurityLevel::Weak,
        ),
        CipherSuite::null_suite(),
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_lattice() -> SecurityLattice {
        SecurityLattice::from_standard_registry()
    }

    #[test]
    fn test_security_level_ordering() {
        assert!(SecurityLevel::Export < SecurityLevel::Low);
        assert!(SecurityLevel::Low < SecurityLevel::Medium);
        assert!(SecurityLevel::Medium < SecurityLevel::High);
        assert!(SecurityLevel::High < SecurityLevel::Maximum);
    }

    #[test]
    fn test_kx_strength_ordering() {
        assert!(KeyExchangeStrength::None < KeyExchangeStrength::Static);
        assert!(KeyExchangeStrength::Static < KeyExchangeStrength::Ephemeral);
        assert!(KeyExchangeStrength::Ephemeral < KeyExchangeStrength::EphemeralEcc);
    }

    #[test]
    fn test_security_profile_from_suite() {
        let suite = CipherSuite::new(
            0xC02F,
            "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
            KeyExchange::Ecdhe,
            AuthAlgorithm::Rsa,
            BulkEncryption::Aes128Gcm,
            MacAlgorithm::Aead,
            NtSecurityLevel::Standard,
        );
        let profile = SecurityProfile::from_cipher_suite(&suite);
        assert_eq!(profile.kx_strength, KeyExchangeStrength::EphemeralEcc);
        assert_eq!(profile.enc_strength, EncryptionStrength::Aead);
        assert_eq!(profile.mac_strength, MacStrength::Aead);
    }

    #[test]
    fn test_security_profile_partial_order() {
        let strong = SecurityProfile {
            kx_strength: KeyExchangeStrength::EphemeralEcc,
            auth_strength: AuthStrength::Rsa,
            enc_strength: EncryptionStrength::Aead,
            mac_strength: MacStrength::Aead,
            overall: SecurityLevel::Maximum,
            is_fips: true,
        };
        let weak = SecurityProfile {
            kx_strength: KeyExchangeStrength::Static,
            auth_strength: AuthStrength::Rsa,
            enc_strength: EncryptionStrength::Medium,
            mac_strength: MacStrength::Sha1,
            overall: SecurityLevel::Medium,
            is_fips: true,
        };
        assert_eq!(weak.partial_cmp(&strong), Some(Ordering::Less));
        assert_eq!(strong.partial_cmp(&weak), Some(Ordering::Greater));
    }

    #[test]
    fn test_security_profile_incomparable() {
        // Better kx but worse encryption
        let a = SecurityProfile {
            kx_strength: KeyExchangeStrength::EphemeralEcc,
            auth_strength: AuthStrength::Rsa,
            enc_strength: EncryptionStrength::Weak,
            mac_strength: MacStrength::Sha1,
            overall: SecurityLevel::Low,
            is_fips: false,
        };
        let b = SecurityProfile {
            kx_strength: KeyExchangeStrength::Static,
            auth_strength: AuthStrength::Rsa,
            enc_strength: EncryptionStrength::Aead,
            mac_strength: MacStrength::Aead,
            overall: SecurityLevel::Medium,
            is_fips: true,
        };
        assert_eq!(a.partial_cmp(&b), None);
    }

    #[test]
    fn test_lattice_join_meet() {
        let lattice = make_test_lattice();
        // RSA_AES128 vs ECDHE_RSA_AES128_GCM
        let weak = 0x002F;
        let strong = 0xC02F;

        let joined = lattice.join(weak, strong);
        assert_eq!(joined, Some(strong));

        let met = lattice.meet(weak, strong);
        assert_eq!(met, Some(weak));
    }

    #[test]
    fn test_lattice_top_bottom() {
        let lattice = make_test_lattice();
        let top = lattice.top();
        let bottom = lattice.bottom();
        assert!(top.is_some());
        assert!(bottom.is_some());
        assert_ne!(top, bottom);
    }

    #[test]
    fn test_preference_lattice_selection() {
        let security = make_test_lattice();
        let pref = vec![0xC02F, 0xCCA8, 0xC030, 0x009C, 0x002F];
        let plat = PreferenceLattice::new(security, pref);

        let offered: BTreeSet<u16> = [0x002F, 0xC02F, 0x009C].into();
        assert_eq!(plat.select_best(&offered), Some(0xC02F));
    }

    #[test]
    fn test_selection_function_basic() {
        let security = make_test_lattice();
        let pref = vec![0xC02F, 0xCCA8, 0xC030, 0x009C, 0x002F, 0x0003];
        let plat = PreferenceLattice::new(security, pref);
        let sf = SelectionFunction::new(plat);

        let offered: BTreeSet<u16> = [0x002F, 0x0003].into();
        let selected = sf.select(&offered);
        assert!(selected.is_some());
    }

    #[test]
    fn test_selection_function_fips() {
        let security = make_test_lattice();
        let pref = vec![0xCCA8, 0xC02F, 0x002F, 0x0003];
        let plat = PreferenceLattice::new(security, pref);
        let sf = SelectionFunction::new(plat).with_fips_required(true);

        let offered: BTreeSet<u16> = [0xCCA8, 0xC02F, 0x002F, 0x0003].into();
        let selected = sf.select(&offered);
        // ChaCha20 is not FIPS, should pick AES-based
        assert!(selected.is_some());
        assert_ne!(selected, Some(0xCCA8));
    }

    #[test]
    fn test_selection_function_blocked() {
        let security = make_test_lattice();
        let pref = vec![0xC02F, 0x002F];
        let plat = PreferenceLattice::new(security, pref);
        let blocked: BTreeSet<u16> = [0xC02F].into();
        let sf = SelectionFunction::new(plat).with_blocked(blocked);

        let offered: BTreeSet<u16> = [0xC02F, 0x002F].into();
        assert_eq!(sf.select(&offered), Some(0x002F));
    }

    #[test]
    fn test_monotonicity_small() {
        let security = make_test_lattice();
        let pref = vec![0xC02F, 0x009C, 0x002F];
        let plat = PreferenceLattice::new(security, pref.clone());
        let sf = SelectionFunction::new(plat);
        let universe: BTreeSet<u16> = pref.into_iter().collect();
        assert!(sf.verify_monotonicity(&universe));
    }

    #[test]
    fn test_filter_by_level() {
        let lattice = make_test_lattice();
        let high = lattice.filter_by_level(SecurityLevel::High);
        for id in &high {
            let p = lattice.profile(*id).unwrap();
            assert!(p.overall >= SecurityLevel::High);
        }
    }

    #[test]
    fn test_fips_suites() {
        let lattice = make_test_lattice();
        let fips = lattice.fips_suites();
        for id in &fips {
            let p = lattice.profile(*id).unwrap();
            assert!(p.is_fips);
        }
    }

    #[test]
    fn test_suites_by_strength() {
        let lattice = make_test_lattice();
        let sorted = lattice.suites_by_strength();
        for w in sorted.windows(2) {
            let pa = lattice.profile(w[0]).unwrap();
            let pb = lattice.profile(w[1]).unwrap();
            assert!(pa.composite_score() >= pb.composite_score());
        }
    }
}
