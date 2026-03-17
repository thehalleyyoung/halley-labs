//! Observation function for the negotiation LTS (Definition D7).
//!
//! Maps terminal states to their observable negotiation outcomes and provides
//! observation equivalence checking used by the bisimulation computation.

use crate::{
    ConcreteValue, ExtractError, ExtractResult, HandshakePhase, LtsState, MessageLabel, NegotiationLTS,
    NegotiationOutcome, Observable, ProtocolVersion, StateId, SymbolicState, SymbolicValue,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// Observation domain (Definition D7)
// ---------------------------------------------------------------------------

/// The domain of observations in the negotiation protocol.
///
/// Each variant corresponds to a dimension of the observable outcome:
/// - CipherSelected: which cipher suite was negotiated
/// - VersionSelected: which protocol version was selected
/// - ExtensionsActive: which extensions are active
/// - Abort: the handshake was aborted (with reason)
/// - InProgress: handshake has not yet terminated
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObservationDomain {
    CipherSelected(u16),
    VersionSelected(ProtocolVersion),
    ExtensionsActive(BTreeSet<u16>),
    Abort { level: u8, description: u8 },
    InProgress,
}

impl ObservationDomain {
    /// Whether this observation represents a completed handshake.
    pub fn is_terminal(&self) -> bool {
        !matches!(self, Self::InProgress)
    }

    /// Whether this observation represents an error/abort.
    pub fn is_abort(&self) -> bool {
        matches!(self, Self::Abort { .. })
    }
}

impl fmt::Display for ObservationDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CipherSelected(c) => write!(f, "cipher(0x{:04x})", c),
            Self::VersionSelected(v) => write!(f, "version({})", v),
            Self::ExtensionsActive(exts) => {
                write!(f, "exts{{")?;
                for (i, e) in exts.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "0x{:04x}", e)?;
                }
                write!(f, "}}")
            }
            Self::Abort { level, description } => {
                write!(f, "abort({},{})", level, description)
            }
            Self::InProgress => write!(f, "in_progress"),
        }
    }
}

// ---------------------------------------------------------------------------
// Attack class scoping
// ---------------------------------------------------------------------------

/// Attack classes that are in-scope for the negotiation analysis (per D7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AttackClass {
    /// Downgrade of cipher suite selection.
    CipherDowngrade,
    /// Downgrade of protocol version.
    VersionDowngrade,
    /// Removal of extensions (e.g., stripping SCSV).
    ExtensionStripping,
    /// CCS injection attack (ChangeCipherSpec).
    CcsInjection,
}

/// Attack classes that are out-of-scope (timing, padding oracle, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutOfScopeAttack {
    TimingSideChannel,
    PaddingOracle,
    Bleichenbacher,
    CompressLeak,
    HeartbleedMemLeak,
}

/// Scope configuration controlling which attacks are analyzed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackScope {
    pub in_scope: HashSet<AttackClass>,
    pub out_of_scope: HashSet<OutOfScopeAttack>,
}

impl Default for AttackScope {
    fn default() -> Self {
        Self {
            in_scope: [
                AttackClass::CipherDowngrade,
                AttackClass::VersionDowngrade,
                AttackClass::ExtensionStripping,
                AttackClass::CcsInjection,
            ]
            .into_iter()
            .collect(),
            out_of_scope: [
                OutOfScopeAttack::TimingSideChannel,
                OutOfScopeAttack::PaddingOracle,
                OutOfScopeAttack::Bleichenbacher,
                OutOfScopeAttack::CompressLeak,
                OutOfScopeAttack::HeartbleedMemLeak,
            ]
            .into_iter()
            .collect(),
        }
    }
}

impl AttackScope {
    /// Check whether a given attack class is in scope.
    pub fn is_in_scope(&self, attack: AttackClass) -> bool {
        self.in_scope.contains(&attack)
    }

    /// Given two observables (honest and adversarial), determine which
    /// in-scope attack classes apply.
    pub fn applicable_attacks(
        &self,
        honest: &Observable,
        adversarial: &Observable,
    ) -> Vec<AttackClass> {
        let mut attacks = Vec::new();

        if self.is_in_scope(AttackClass::CipherDowngrade) {
            if let (Some(hc), Some(ac)) = (honest.selected_cipher, adversarial.selected_cipher) {
                if ac != hc {
                    attacks.push(AttackClass::CipherDowngrade);
                }
            }
        }

        if self.is_in_scope(AttackClass::VersionDowngrade) {
            if let (Some(hv), Some(av)) =
                (&honest.selected_version, &adversarial.selected_version)
            {
                if av.security_level() < hv.security_level() {
                    attacks.push(AttackClass::VersionDowngrade);
                }
            }
        }

        if self.is_in_scope(AttackClass::ExtensionStripping) {
            if honest.active_extensions != adversarial.active_extensions {
                let stripped: BTreeSet<u16> = honest
                    .active_extensions
                    .difference(&adversarial.active_extensions)
                    .copied()
                    .collect();
                if !stripped.is_empty() {
                    attacks.push(AttackClass::ExtensionStripping);
                }
            }
        }

        attacks
    }
}

// ---------------------------------------------------------------------------
// ObservationFunction
// ---------------------------------------------------------------------------

/// The observation function obs: S → O mapping LTS states to negotiation outcomes.
///
/// For terminal states (ApplicationData, Alert), produces a concrete Observable.
/// For non-terminal states, produces InProgress.
pub struct ObservationFunction {
    scope: AttackScope,
    /// Cache of computed observations.
    cache: HashMap<StateId, Observable>,
}

impl ObservationFunction {
    pub fn new() -> Self {
        Self {
            scope: AttackScope::default(),
            cache: HashMap::new(),
        }
    }

    pub fn with_scope(scope: AttackScope) -> Self {
        Self {
            scope,
            cache: HashMap::new(),
        }
    }

    /// Compute the observation for a state in the LTS.
    pub fn observe(&mut self, lts: &NegotiationLTS, state_id: StateId) -> Observable {
        if let Some(cached) = self.cache.get(&state_id) {
            return cached.clone();
        }

        let obs = match lts.get_state(state_id) {
            Some(state) => self.compute_observation(state),
            None => Observable::in_progress(HandshakePhase::Init),
        };

        self.cache.insert(state_id, obs.clone());
        obs
    }

    /// Compute observation from an LTS state.
    fn compute_observation(&self, state: &LtsState) -> Observable {
        let neg = &state.negotiation;

        if !state.is_terminal {
            return Observable::in_progress(neg.phase);
        }

        // Alert → abort
        if neg.phase == HandshakePhase::Alert {
            return Observable::aborted();
        }

        // Terminal completed state → extract observables from negotiation state.
        let cipher = neg.selected_cipher.as_ref().map(|c| c.iana_id);
        let version = neg.version.clone();
        let extensions: BTreeSet<u16> = neg.extensions.iter().map(|e| e.id).collect();

        Observable {
            selected_cipher: cipher,
            selected_version: version,
            active_extensions: extensions,
            outcome: NegotiationOutcome::Completed,
            phase: neg.phase,
        }
    }

    /// Compute observation directly from a symbolic state (for extraction).
    pub fn observe_symbolic(&self, sym: &SymbolicState) -> Observable {
        let neg = &sym.negotiation;

        if !neg.phase.is_terminal() {
            return Observable::in_progress(neg.phase);
        }

        if neg.phase == HandshakePhase::Alert {
            return Observable::aborted();
        }

        let extensions: BTreeSet<u16> = neg.extensions.iter().map(|e| e.id).collect();
        Observable {
            selected_cipher: neg.selected_cipher.as_ref().map(|c| c.iana_id),
            selected_version: neg.version.clone(),
            active_extensions: extensions,
            outcome: NegotiationOutcome::Completed,
            phase: neg.phase,
        }
    }

    /// Recompute all observations for an LTS (after structural modifications).
    pub fn recompute_all(&mut self, lts: &NegotiationLTS) {
        self.cache.clear();
        for &sid in lts.states.keys() {
            self.observe(lts, sid);
        }
    }

    /// Access the attack scope.
    pub fn scope(&self) -> &AttackScope {
        &self.scope
    }

    /// Clear the observation cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl Default for ObservationFunction {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// OutcomeExtractor
// ---------------------------------------------------------------------------

/// Extracts observable outcomes from symbolic states, resolving
/// symbolic cipher/version values where possible.
pub struct OutcomeExtractor {
    /// Known cipher suite IDs (for resolving symbolic references).
    known_ciphers: BTreeSet<u16>,
    /// Known protocol versions.
    known_versions: Vec<ProtocolVersion>,
}

impl OutcomeExtractor {
    pub fn new() -> Self {
        Self {
            known_ciphers: [
                0x0000, // NULL
                0x002f, // TLS_RSA_WITH_AES_128_CBC_SHA
                0x0033, // TLS_DHE_RSA_WITH_AES_128_CBC_SHA
                0x0035, // TLS_RSA_WITH_AES_256_CBC_SHA
                0x009c, // TLS_RSA_WITH_AES_128_GCM_SHA256
                0x009d, // TLS_RSA_WITH_AES_256_GCM_SHA384
                0xc02b, // TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256
                0xc02f, // TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
                0xcca8, // TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305
                0xcca9, // TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305
                0x1301, // TLS_AES_128_GCM_SHA256 (TLS 1.3)
                0x1302, // TLS_AES_256_GCM_SHA384 (TLS 1.3)
                0x1303, // TLS_CHACHA20_POLY1305_SHA256 (TLS 1.3)
                0x00ff, // TLS_EMPTY_RENEGOTIATION_INFO_SCSV
            ]
            .into(),
            known_versions: vec![
                ProtocolVersion::Ssl30,
                ProtocolVersion::Tls10,
                ProtocolVersion::Tls11,
                ProtocolVersion::Tls12,
                ProtocolVersion::Tls13,
            ],
        }
    }

    /// Extract the outcome from a symbolic state.
    pub fn extract(&self, state: &SymbolicState) -> Observable {
        let neg = &state.negotiation;

        if !neg.phase.is_terminal() {
            return Observable::in_progress(neg.phase);
        }

        if neg.phase == HandshakePhase::Alert {
            return Observable::aborted();
        }

        let cipher = neg.selected_cipher.as_ref().map(|c| c.iana_id);
        let version = neg.version.clone();

        let extensions: BTreeSet<u16> = neg.extensions.iter().map(|e| e.id).collect();

        Observable {
            selected_cipher: cipher,
            selected_version: version,
            active_extensions: extensions,
            outcome: NegotiationOutcome::Completed,
            phase: neg.phase,
        }
    }

    /// Try to resolve a symbolic cipher value to a concrete ID.
    pub fn resolve_cipher(&self, val: &SymbolicValue) -> Option<u16> {
        match val {
            SymbolicValue::Concrete(ConcreteValue::Int(n)) => {
                let id = *n as u16;
                Some(id)
            }
            SymbolicValue::Concrete(ConcreteValue::BitVec { value, .. }) => {
                let id = *value as u16;
                Some(id)
            }
            _ => None,
        }
    }

    /// Try to resolve a symbolic version value to a concrete ProtocolVersion.
    pub fn resolve_version(&self, val: &SymbolicValue) -> Option<ProtocolVersion> {
        match val {
            SymbolicValue::Concrete(ConcreteValue::Int(n)) => {
                let raw = *n as u16;
                match raw {
                    0x0300 => Some(ProtocolVersion::Ssl30),
                    0x0301 => Some(ProtocolVersion::Tls10),
                    0x0302 => Some(ProtocolVersion::Tls11),
                    0x0303 => Some(ProtocolVersion::Tls12),
                    0x0304 => Some(ProtocolVersion::Tls13),
                    _ => Some(ProtocolVersion::Unknown),
                }
            }
            SymbolicValue::Concrete(ConcreteValue::BitVec { value, .. }) => {
                let raw = *value as u16;
                match raw {
                    0x0300 => Some(ProtocolVersion::Ssl30),
                    0x0301 => Some(ProtocolVersion::Tls10),
                    0x0302 => Some(ProtocolVersion::Tls11),
                    0x0303 => Some(ProtocolVersion::Tls12),
                    0x0304 => Some(ProtocolVersion::Tls13),
                    _ => Some(ProtocolVersion::Unknown),
                }
            }
            _ => None,
        }
    }

    /// Register additional known cipher IDs.
    pub fn add_known_cipher(&mut self, id: u16) {
        self.known_ciphers.insert(id);
    }
}

impl Default for OutcomeExtractor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ObservationEquivalence
// ---------------------------------------------------------------------------

/// Checks observation agreement between pairs of states,
/// used by the bisimulation algorithm.
pub struct ObservationEquivalence {
    obs_fn: ObservationFunction,
}

impl ObservationEquivalence {
    pub fn new() -> Self {
        Self {
            obs_fn: ObservationFunction::new(),
        }
    }

    pub fn with_scope(scope: AttackScope) -> Self {
        Self {
            obs_fn: ObservationFunction::with_scope(scope),
        }
    }

    /// Check whether two states have the same observation in the LTS.
    pub fn are_observation_equivalent(
        &mut self,
        lts: &NegotiationLTS,
        s1: StateId,
        s2: StateId,
    ) -> bool {
        let o1 = self.obs_fn.observe(lts, s1);
        let o2 = self.obs_fn.observe(lts, s2);
        o1.agrees_with(&o2)
    }

    /// Partition states by their observation (initial partition for bisimulation).
    pub fn partition_by_observation(
        &mut self,
        lts: &NegotiationLTS,
    ) -> Vec<Vec<StateId>> {
        let mut groups: HashMap<Observable, Vec<StateId>> = HashMap::new();
        for &sid in lts.states.keys() {
            let obs = self.obs_fn.observe(lts, sid);
            groups.entry(obs).or_default().push(sid);
        }

        let mut partitions: Vec<Vec<StateId>> = groups.into_values().collect();
        // Deterministic ordering for reproducibility.
        for part in &mut partitions {
            part.sort();
        }
        partitions.sort_by(|a, b| a.first().cmp(&b.first()));
        partitions
    }

    /// Get the observation for a state.
    pub fn observe(&mut self, lts: &NegotiationLTS, state: StateId) -> Observable {
        self.obs_fn.observe(lts, state)
    }

    /// Count the number of distinct observations in an LTS.
    pub fn distinct_observation_count(&mut self, lts: &NegotiationLTS) -> usize {
        let mut obs_set: HashSet<Observable> = HashSet::new();
        for &sid in lts.states.keys() {
            let obs = self.obs_fn.observe(lts, sid);
            obs_set.insert(obs);
        }
        obs_set.len()
    }
}

impl Default for ObservationEquivalence {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// WeaknessOrder (Definition D5)
// ---------------------------------------------------------------------------

/// A weakness ordering on cipher suites: c1 ⪯_sec c2 means c1 is weaker
/// than or equal to c2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeaknessOrder {
    /// Cipher IDs ordered from weakest to strongest.
    ordering: Vec<u16>,
    /// Map from cipher ID to its strength rank (lower = weaker).
    rank: HashMap<u16, usize>,
}

impl WeaknessOrder {
    /// Create from a list of cipher IDs, weakest first.
    pub fn new(ordering: Vec<u16>) -> Self {
        let rank: HashMap<u16, usize> = ordering
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();
        Self { ordering, rank }
    }

    /// Default TLS weakness ordering.
    pub fn default_tls() -> Self {
        Self::new(vec![
            0x0000, // NULL
            0x0001, // RSA_WITH_NULL_MD5
            0x002f, // RSA_WITH_AES_128_CBC_SHA
            0x0035, // RSA_WITH_AES_256_CBC_SHA
            0x009c, // RSA_WITH_AES_128_GCM_SHA256
            0x009d, // RSA_WITH_AES_256_GCM_SHA384
            0x0033, // DHE_RSA_WITH_AES_128_CBC_SHA
            0xc02f, // ECDHE_RSA_WITH_AES_128_GCM_SHA256
            0xc02b, // ECDHE_ECDSA_WITH_AES_128_GCM_SHA256
            0xcca8, // ECDHE_RSA_WITH_CHACHA20_POLY1305
            0xcca9, // ECDHE_ECDSA_WITH_CHACHA20_POLY1305
            0x1301, // AES_128_GCM_SHA256 (TLS 1.3)
            0x1302, // AES_256_GCM_SHA384 (TLS 1.3)
            0x1303, // CHACHA20_POLY1305_SHA256 (TLS 1.3)
        ])
    }

    /// c1 ⪯_sec c2: is c1 weaker or equal to c2?
    pub fn is_weaker_or_equal(&self, c1: u16, c2: u16) -> bool {
        let r1 = self.rank.get(&c1).copied().unwrap_or(usize::MAX);
        let r2 = self.rank.get(&c2).copied().unwrap_or(usize::MAX);
        r1 <= r2
    }

    /// Is c1 strictly weaker than c2?
    pub fn is_strictly_weaker(&self, c1: u16, c2: u16) -> bool {
        let r1 = self.rank.get(&c1).copied().unwrap_or(usize::MAX);
        let r2 = self.rank.get(&c2).copied().unwrap_or(usize::MAX);
        r1 < r2
    }

    /// Get the rank of a cipher (lower = weaker).
    pub fn rank_of(&self, cipher: u16) -> Option<usize> {
        self.rank.get(&cipher).copied()
    }

    /// Check if an adversarial outcome is a downgrade from the honest outcome.
    pub fn is_downgrade(
        &self,
        honest: &Observable,
        adversarial: &Observable,
    ) -> bool {
        // Cipher downgrade check.
        if let (Some(hc), Some(ac)) = (honest.selected_cipher, adversarial.selected_cipher) {
            if self.is_strictly_weaker(ac, hc) {
                return true;
            }
        }

        // Version downgrade check.
        if let (Some(hv), Some(av)) = (&honest.selected_version, &adversarial.selected_version) {
            if av.security_level() < hv.security_level() {
                return true;
            }
        }

        // Extension stripping check.
        if honest.active_extensions.len() > adversarial.active_extensions.len() {
            let stripped: BTreeSet<u16> = honest
                .active_extensions
                .difference(&adversarial.active_extensions)
                .copied()
                .collect();
            if !stripped.is_empty() {
                return true;
            }
        }

        false
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NegotiationLTS;
    use negsyn_types::{Extension, NegotiationState};

    fn make_terminal_state(
        phase: HandshakePhase,
        cipher: Option<u16>,
        version: ProtocolVersion,
    ) -> NegotiationState {
        let mut ns = NegotiationState::new();
        ns.phase = phase;
        ns.version = Some(version);
        ns.selected_cipher = cipher.map(|id| CipherSuite::new(
            id,
            format!("TEST_0x{:04x}", id),
            negsyn_types::protocol::KeyExchange::NULL,
            negsyn_types::protocol::AuthAlgorithm::NULL,
            negsyn_types::protocol::EncryptionAlgorithm::NULL,
            negsyn_types::protocol::MacAlgorithm::NULL,
            SecurityLevel::Standard,
        ));
        ns
    }

    #[test]
    fn test_observation_domain_display() {
        let od = ObservationDomain::CipherSelected(0x002f);
        assert_eq!(format!("{}", od), "cipher(0x002f)");

        let od = ObservationDomain::VersionSelected(ProtocolVersion::Tls12);
        assert!(format!("{}", od).contains("TLS 1.2"));

        let od = ObservationDomain::Abort {
            level: 2,
            description: 40,
        };
        assert_eq!(format!("{}", od), "abort(2,40)");
    }

    #[test]
    fn test_observation_function_terminal() {
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_terminal_state(
            HandshakePhase::ApplicationData,
            Some(0x002f),
            ProtocolVersion::Tls12,
        ));

        let mut obs_fn = ObservationFunction::new();
        let obs = obs_fn.observe(&lts, s0);
        assert_eq!(obs.outcome, NegotiationOutcome::Completed);
        assert_eq!(obs.selected_cipher, Some(0x002f));
        assert_eq!(obs.selected_version, Some(ProtocolVersion::Tls12));
    }

    #[test]
    fn test_observation_function_abort() {
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_terminal_state(
            HandshakePhase::Alert,
            None,
            ProtocolVersion::Tls12,
        ));

        let mut obs_fn = ObservationFunction::new();
        let obs = obs_fn.observe(&lts, s0);
        assert_eq!(obs.outcome, NegotiationOutcome::Aborted);
    }

    #[test]
    fn test_observation_function_in_progress() {
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_terminal_state(
            HandshakePhase::ClientHello,
            None,
            ProtocolVersion::Tls12,
        ));

        let mut obs_fn = ObservationFunction::new();
        let obs = obs_fn.observe(&lts, s0);
        assert_eq!(obs.outcome, NegotiationOutcome::InProgress);
    }

    #[test]
    fn test_observation_equivalence() {
        let mut lts = NegotiationLTS::new();
        let s1 = lts.add_state(make_terminal_state(
            HandshakePhase::ApplicationData,
            Some(0x002f),
            ProtocolVersion::Tls12,
        ));
        let s2 = lts.add_state(make_terminal_state(
            HandshakePhase::ApplicationData,
            Some(0x002f),
            ProtocolVersion::Tls12,
        ));
        let s3 = lts.add_state(make_terminal_state(
            HandshakePhase::ApplicationData,
            Some(0x0035),
            ProtocolVersion::Tls12,
        ));

        let mut eq = ObservationEquivalence::new();
        assert!(eq.are_observation_equivalent(&lts, s1, s2));
        assert!(!eq.are_observation_equivalent(&lts, s1, s3));
    }

    #[test]
    fn test_partition_by_observation() {
        let mut lts = NegotiationLTS::new();
        let _s1 = lts.add_state(make_terminal_state(
            HandshakePhase::ApplicationData,
            Some(0x002f),
            ProtocolVersion::Tls12,
        ));
        let _s2 = lts.add_state(make_terminal_state(
            HandshakePhase::ApplicationData,
            Some(0x002f),
            ProtocolVersion::Tls12,
        ));
        let _s3 = lts.add_state(make_terminal_state(
            HandshakePhase::Alert,
            None,
            ProtocolVersion::Tls12,
        ));

        let mut eq = ObservationEquivalence::new();
        let parts = eq.partition_by_observation(&lts);
        // Two distinct observations: Completed(0x002f, TLS1.2) and Aborted.
        assert_eq!(parts.len(), 2);
    }

    #[test]
    fn test_weakness_order() {
        let wo = WeaknessOrder::default_tls();
        // NULL is weaker than AES_128_GCM
        assert!(wo.is_weaker_or_equal(0x0000, 0x009c));
        assert!(wo.is_strictly_weaker(0x0000, 0x009c));
        // AES_128_GCM is not weaker than NULL
        assert!(!wo.is_strictly_weaker(0x009c, 0x0000));
    }

    #[test]
    fn test_is_downgrade() {
        let wo = WeaknessOrder::default_tls();
        let honest = Observable::successful(0x009c, ProtocolVersion::Tls12, BTreeSet::new());
        let adversarial =
            Observable::successful(0x002f, ProtocolVersion::Tls12, BTreeSet::new());
        assert!(wo.is_downgrade(&honest, &adversarial));
    }

    #[test]
    fn test_is_not_downgrade() {
        let wo = WeaknessOrder::default_tls();
        let honest = Observable::successful(0x002f, ProtocolVersion::Tls12, BTreeSet::new());
        let same = Observable::successful(0x002f, ProtocolVersion::Tls12, BTreeSet::new());
        assert!(!wo.is_downgrade(&honest, &same));
    }

    #[test]
    fn test_version_downgrade() {
        let wo = WeaknessOrder::default_tls();
        let honest = Observable::successful(0x002f, ProtocolVersion::Tls12, BTreeSet::new());
        let adversarial =
            Observable::successful(0x002f, ProtocolVersion::Tls10, BTreeSet::new());
        assert!(wo.is_downgrade(&honest, &adversarial));
    }

    #[test]
    fn test_attack_scope() {
        let scope = AttackScope::default();
        assert!(scope.is_in_scope(AttackClass::CipherDowngrade));
        assert!(scope.is_in_scope(AttackClass::VersionDowngrade));
        assert!(scope.is_in_scope(AttackClass::ExtensionStripping));
        assert!(scope.is_in_scope(AttackClass::CcsInjection));

        let honest = Observable::successful(0x009c, ProtocolVersion::Tls12, BTreeSet::new());
        let adv = Observable::successful(0x002f, ProtocolVersion::Tls10, BTreeSet::new());
        let attacks = scope.applicable_attacks(&honest, &adv);
        assert!(attacks.contains(&AttackClass::CipherDowngrade));
        assert!(attacks.contains(&AttackClass::VersionDowngrade));
    }

    #[test]
    fn test_outcome_extractor() {
        let extractor = OutcomeExtractor::new();

        let neg = make_terminal_state(
            HandshakePhase::ApplicationData,
            Some(0x002f),
            ProtocolVersion::Tls12,
        );
        let mut sym = SymbolicState::new(0, 0x1000);
        sym.negotiation = neg;
        let obs = extractor.extract(&sym);
        assert_eq!(obs.selected_cipher, Some(0x002f));
        assert_eq!(obs.selected_version, Some(ProtocolVersion::Tls12));
    }

    #[test]
    fn test_outcome_extractor_resolve_cipher() {
        let extractor = OutcomeExtractor::new();
        let val = SymbolicValue::int_const(0x002f);
        assert_eq!(extractor.resolve_cipher(&val), Some(0x002f));

        let symbolic = SymbolicValue::var("cipher_var", SymSort::BitVec(16));
        assert_eq!(extractor.resolve_cipher(&symbolic), None);
    }

    #[test]
    fn test_outcome_extractor_resolve_version() {
        let extractor = OutcomeExtractor::new();
        let val = SymbolicValue::int_const(0x0303);
        assert_eq!(
            extractor.resolve_version(&val),
            Some(ProtocolVersion::Tls12)
        );
    }

    #[test]
    fn test_extension_stripping_detection() {
        let wo = WeaknessOrder::default_tls();
        let honest_exts: BTreeSet<u16> = [0xff01, 0x0017].into();
        let adv_exts: BTreeSet<u16> = [0x0017].into();
        let honest = Observable::successful(0x002f, ProtocolVersion::Tls12, honest_exts);
        let adversarial = Observable::successful(0x002f, ProtocolVersion::Tls12, adv_exts);
        assert!(wo.is_downgrade(&honest, &adversarial));
    }
}
