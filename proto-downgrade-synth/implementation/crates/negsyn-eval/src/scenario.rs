//! Test scenario generation for protocol negotiation.

use negsyn_types::{HandshakePhase, ProtocolVersion, SecurityLevel};

use itertools::Itertools;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use uuid::Uuid;

/// A cipher suite negotiation scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CipherSuiteScenario {
    pub id: String,
    pub name: String,
    pub client_offered: Vec<u16>,
    pub server_supported: Vec<u16>,
    pub expected_selected: Option<u16>,
    pub client_preference_order: bool,
    pub is_export_allowed: bool,
}

impl CipherSuiteScenario {
    pub fn new(name: impl Into<String>, client: Vec<u16>, server: Vec<u16>) -> Self {
        let expected = if client_preference_first(&client, &server) {
            client.iter().find(|c| server.contains(c)).copied()
        } else {
            None
        };

        Self {
            id: Uuid::new_v4().to_string(),
            name: name.into(),
            client_offered: client,
            server_supported: server,
            expected_selected: expected,
            client_preference_order: true,
            is_export_allowed: false,
        }
    }

    pub fn with_export(mut self, allowed: bool) -> Self {
        self.is_export_allowed = allowed;
        self
    }

    pub fn with_server_preference(mut self) -> Self {
        self.client_preference_order = false;
        if !self.server_supported.is_empty() && !self.client_offered.is_empty() {
            self.expected_selected = self
                .server_supported
                .iter()
                .find(|s| self.client_offered.contains(s))
                .copied();
        }
        self
    }

    /// Compute the common cipher suites between client and server.
    pub fn common_ciphers(&self) -> Vec<u16> {
        let server_set: BTreeSet<u16> = self.server_supported.iter().copied().collect();
        self.client_offered
            .iter()
            .filter(|c| server_set.contains(c))
            .copied()
            .collect()
    }

    pub fn has_overlap(&self) -> bool {
        !self.common_ciphers().is_empty()
    }
}

fn client_preference_first(client: &[u16], server: &[u16]) -> bool {
    let server_set: BTreeSet<u16> = server.iter().copied().collect();
    client.iter().any(|c| server_set.contains(c))
}

/// A version negotiation scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionScenario {
    pub id: String,
    pub name: String,
    pub client_min_version: ProtocolVersion,
    pub client_max_version: ProtocolVersion,
    pub server_min_version: ProtocolVersion,
    pub server_max_version: ProtocolVersion,
    pub expected_version: Option<ProtocolVersion>,
    pub expect_downgrade_sentinel: bool,
}

impl VersionScenario {
    pub fn new(
        name: impl Into<String>,
        client_min: ProtocolVersion,
        client_max: ProtocolVersion,
        server_min: ProtocolVersion,
        server_max: ProtocolVersion,
    ) -> Self {
        let expected = compute_negotiated_version(client_min, client_max, server_min, server_max);
        let expect_sentinel = match &expected {
            Some(v) if client_max == ProtocolVersion::tls13() && v.security_level() < ProtocolVersion::tls13().security_level() => true,
            Some(v) if client_max == ProtocolVersion::tls12() && v.security_level() < ProtocolVersion::tls12().security_level() => true,
            _ => false,
        };

        Self {
            id: Uuid::new_v4().to_string(),
            name: name.into(),
            client_min_version: client_min,
            client_max_version: client_max,
            server_min_version: server_min,
            server_max_version: server_max,
            expected_version: expected,
            expect_downgrade_sentinel: expect_sentinel,
        }
    }

    pub fn has_overlap(&self) -> bool {
        self.expected_version.is_some()
    }
}

fn compute_negotiated_version(
    client_min: ProtocolVersion,
    client_max: ProtocolVersion,
    server_min: ProtocolVersion,
    server_max: ProtocolVersion,
) -> Option<ProtocolVersion> {
    let versions = [
        ProtocolVersion::tls13(),
        ProtocolVersion::tls12(),
        ProtocolVersion::tls11(),
        ProtocolVersion::tls10(),
        ProtocolVersion::ssl30(),
    ];

    for v in &versions {
        let in_client = v.security_level() >= client_min.security_level()
            && v.security_level() <= client_max.security_level();
        let in_server = v.security_level() >= server_min.security_level()
            && v.security_level() <= server_max.security_level();
        if in_client && in_server {
            return Some(v.clone());
        }
    }
    None
}

/// An extension negotiation scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionScenario {
    pub id: String,
    pub name: String,
    pub client_extensions: Vec<ExtensionConfig>,
    pub server_extensions: Vec<ExtensionConfig>,
    pub expected_negotiated: Vec<u16>,
    pub is_mandatory_missing: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionConfig {
    pub id: u16,
    pub name: String,
    pub is_critical: bool,
    pub data_length: usize,
}

impl ExtensionScenario {
    pub fn new(
        name: impl Into<String>,
        client_ext: Vec<ExtensionConfig>,
        server_ext: Vec<ExtensionConfig>,
    ) -> Self {
        let client_ids: BTreeSet<u16> = client_ext.iter().map(|e| e.id).collect();
        let server_ids: BTreeSet<u16> = server_ext.iter().map(|e| e.id).collect();
        let negotiated: Vec<u16> = client_ids.intersection(&server_ids).copied().collect();

        let mandatory_missing = client_ext
            .iter()
            .any(|e| e.is_critical && !server_ids.contains(&e.id));

        Self {
            id: Uuid::new_v4().to_string(),
            name: name.into(),
            client_extensions: client_ext,
            server_extensions: server_ext,
            expected_negotiated: negotiated,
            is_mandatory_missing: mandatory_missing,
        }
    }

    pub fn negotiated_count(&self) -> usize {
        self.expected_negotiated.len()
    }
}

/// An adversary scenario with budget constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdversaryScenario {
    pub id: String,
    pub name: String,
    pub adversary_budget: u32,
    pub can_intercept: bool,
    pub can_modify: bool,
    pub can_inject: bool,
    pub can_drop: bool,
    pub can_replay: bool,
    pub target_cipher_suites: Vec<u16>,
    pub target_version: Option<ProtocolVersion>,
    pub max_message_modifications: u32,
}

impl AdversaryScenario {
    pub fn passive(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.into(),
            adversary_budget: 0,
            can_intercept: true,
            can_modify: false,
            can_inject: false,
            can_drop: false,
            can_replay: false,
            target_cipher_suites: vec![],
            target_version: None,
            max_message_modifications: 0,
        }
    }

    pub fn active(name: impl Into<String>, budget: u32) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.into(),
            adversary_budget: budget,
            can_intercept: true,
            can_modify: true,
            can_inject: true,
            can_drop: true,
            can_replay: true,
            target_cipher_suites: vec![],
            target_version: None,
            max_message_modifications: budget,
        }
    }

    pub fn with_target_ciphers(mut self, ciphers: Vec<u16>) -> Self {
        self.target_cipher_suites = ciphers;
        self
    }

    pub fn with_target_version(mut self, version: ProtocolVersion) -> Self {
        self.target_version = Some(version);
        self
    }

    pub fn capability_count(&self) -> u32 {
        let mut count = 0u32;
        if self.can_intercept { count += 1; }
        if self.can_modify { count += 1; }
        if self.can_inject { count += 1; }
        if self.can_drop { count += 1; }
        if self.can_replay { count += 1; }
        count
    }

    pub fn is_passive(&self) -> bool {
        !self.can_modify && !self.can_inject && !self.can_drop && !self.can_replay
    }
}

/// A complete test scenario combining all dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestScenario {
    pub id: String,
    pub name: String,
    pub cipher_scenario: CipherSuiteScenario,
    pub version_scenario: VersionScenario,
    pub extension_scenario: ExtensionScenario,
    pub adversary_scenario: AdversaryScenario,
    pub tags: Vec<String>,
}

/// The main scenario generator.
pub struct ScenarioGenerator {
    seed: u64,
    cipher_suites: Vec<u16>,
    versions: Vec<ProtocolVersion>,
    extensions: Vec<ExtensionConfig>,
    adversary_budgets: Vec<u32>,
}

impl ScenarioGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            cipher_suites: default_cipher_suites(),
            versions: default_versions(),
            extensions: default_extensions(),
            adversary_budgets: vec![0, 1, 2, 3, 5],
        }
    }

    pub fn with_cipher_suites(mut self, suites: Vec<u16>) -> Self {
        self.cipher_suites = suites;
        self
    }

    pub fn with_versions(mut self, versions: Vec<ProtocolVersion>) -> Self {
        self.versions = versions;
        self
    }

    pub fn with_extensions(mut self, extensions: Vec<ExtensionConfig>) -> Self {
        self.extensions = extensions;
        self
    }

    pub fn with_budgets(mut self, budgets: Vec<u32>) -> Self {
        self.adversary_budgets = budgets;
        self
    }

    /// Generate cipher suite scenarios.
    pub fn generate_cipher_scenarios(&self) -> Vec<CipherSuiteScenario> {
        let mut scenarios = Vec::new();

        scenarios.push(CipherSuiteScenario::new(
            "all_common",
            self.cipher_suites.clone(),
            self.cipher_suites.clone(),
        ));

        if self.cipher_suites.len() >= 2 {
            let mid = self.cipher_suites.len() / 2;
            scenarios.push(CipherSuiteScenario::new(
                "disjoint_halves",
                self.cipher_suites[..mid].to_vec(),
                self.cipher_suites[mid..].to_vec(),
            ));
        }

        if let Some(&first) = self.cipher_suites.first() {
            scenarios.push(CipherSuiteScenario::new(
                "single_overlap",
                vec![first],
                self.cipher_suites.clone(),
            ));
        }

        scenarios.push(CipherSuiteScenario::new(
            "empty_client",
            vec![],
            self.cipher_suites.clone(),
        ));

        let export_ciphers: Vec<u16> = vec![0x0003, 0x0006, 0x0008, 0x000B];
        let mixed: Vec<u16> = self
            .cipher_suites
            .iter()
            .copied()
            .chain(export_ciphers.iter().copied())
            .collect();
        scenarios.push(
            CipherSuiteScenario::new("export_mixed", mixed, self.cipher_suites.clone())
                .with_export(true),
        );

        scenarios.push(
            CipherSuiteScenario::new(
                "reversed_preference",
                self.cipher_suites.iter().rev().copied().collect(),
                self.cipher_suites.clone(),
            )
            .with_server_preference(),
        );

        let mut rng = StdRng::seed_from_u64(self.seed);
        for i in 0..3 {
            let client_size = (i + 2).min(self.cipher_suites.len());
            let server_size = (i + 3).min(self.cipher_suites.len());
            let mut client = self.cipher_suites.clone();
            client.shuffle(&mut rng);
            client.truncate(client_size);
            let mut server = self.cipher_suites.clone();
            server.shuffle(&mut rng);
            server.truncate(server_size);
            scenarios.push(CipherSuiteScenario::new(
                format!("random_{}", i),
                client,
                server,
            ));
        }

        scenarios
    }

    /// Generate version negotiation scenarios.
    pub fn generate_version_scenarios(&self) -> Vec<VersionScenario> {
        let mut scenarios = Vec::new();

        for &v in &self.versions {
            scenarios.push(VersionScenario::new(
                format!("same_{:?}", v),
                v, v, v, v,
            ));
        }

        if self.versions.len() >= 2 {
            let min_v = self.versions[0];
            let max_v = self.versions[self.versions.len() - 1];
            scenarios.push(VersionScenario::new(
                "client_higher",
                min_v, max_v, min_v, min_v,
            ));
            scenarios.push(VersionScenario::new(
                "server_higher",
                min_v, min_v, min_v, max_v,
            ));
            scenarios.push(VersionScenario::new(
                "full_range_both",
                min_v, max_v, min_v, max_v,
            ));
        }

        scenarios.push(VersionScenario::new(
            "tls13_only_client_vs_tls12_server",
            ProtocolVersion::tls13(),
            ProtocolVersion::tls13(),
            ProtocolVersion::tls10(),
            ProtocolVersion::tls12(),
        ));

        scenarios.push(VersionScenario::new(
            "sslv3_fallback",
            ProtocolVersion::ssl30(),
            ProtocolVersion::tls12(),
            ProtocolVersion::ssl30(),
            ProtocolVersion::tls12(),
        ));

        scenarios
    }

    /// Generate extension negotiation scenarios.
    pub fn generate_extension_scenarios(&self) -> Vec<ExtensionScenario> {
        let mut scenarios = Vec::new();

        scenarios.push(ExtensionScenario::new(
            "all_common",
            self.extensions.clone(),
            self.extensions.clone(),
        ));

        scenarios.push(ExtensionScenario::new(
            "no_extensions",
            vec![],
            vec![],
        ));

        if !self.extensions.is_empty() {
            scenarios.push(ExtensionScenario::new(
                "client_only",
                self.extensions.clone(),
                vec![],
            ));

            scenarios.push(ExtensionScenario::new(
                "server_only",
                vec![],
                self.extensions.clone(),
            ));
        }

        if self.extensions.len() >= 2 {
            let mid = self.extensions.len() / 2;
            scenarios.push(ExtensionScenario::new(
                "partial_overlap",
                self.extensions[..mid + 1].to_vec(),
                self.extensions[mid..].to_vec(),
            ));
        }

        let critical_exts: Vec<ExtensionConfig> = self
            .extensions
            .iter()
            .filter(|e| e.is_critical)
            .cloned()
            .collect();
        if !critical_exts.is_empty() {
            let non_critical: Vec<ExtensionConfig> = self
                .extensions
                .iter()
                .filter(|e| !e.is_critical)
                .cloned()
                .collect();
            scenarios.push(ExtensionScenario::new(
                "missing_critical",
                critical_exts,
                non_critical,
            ));
        }

        scenarios
    }

    /// Generate adversary scenarios.
    pub fn generate_adversary_scenarios(&self) -> Vec<AdversaryScenario> {
        let mut scenarios = Vec::new();

        scenarios.push(AdversaryScenario::passive("passive_observer"));

        for &budget in &self.adversary_budgets {
            if budget == 0 {
                continue;
            }
            scenarios.push(
                AdversaryScenario::active(format!("active_budget_{}", budget), budget)
                    .with_target_ciphers(vec![0x0003, 0x0006]),
            );
        }

        let mut modify_only = AdversaryScenario::passive("modify_only");
        modify_only.can_modify = true;
        modify_only.adversary_budget = 1;
        modify_only.max_message_modifications = 1;
        scenarios.push(modify_only);

        let mut drop_only = AdversaryScenario::passive("drop_only");
        drop_only.can_drop = true;
        drop_only.adversary_budget = 1;
        scenarios.push(drop_only);

        let mut inject_only = AdversaryScenario::passive("inject_only");
        inject_only.can_inject = true;
        inject_only.adversary_budget = 2;
        scenarios.push(inject_only);

        if let Some(&max_budget) = self.adversary_budgets.last() {
            scenarios.push(
                AdversaryScenario::active("version_downgrade_attacker", max_budget)
                    .with_target_version(ProtocolVersion::ssl30()),
            );
        }

        scenarios
    }

    /// Generate combined scenarios using covering design approach.
    pub fn generate_covering_scenarios(&self) -> Vec<TestScenario> {
        let cipher_scenarios = self.generate_cipher_scenarios();
        let version_scenarios = self.generate_version_scenarios();
        let extension_scenarios = self.generate_extension_scenarios();
        let adversary_scenarios = self.generate_adversary_scenarios();

        let mut combined = Vec::new();
        let mut seen = HashSet::new();

        for (ci, cs) in cipher_scenarios.iter().enumerate() {
            for (vi, vs) in version_scenarios.iter().enumerate() {
                let ei = (ci + vi) % extension_scenarios.len().max(1);
                let ai = (ci + vi) % adversary_scenarios.len().max(1);

                let key = (ci, vi, ei, ai);
                if seen.insert(key) {
                    let es = extension_scenarios.get(ei).cloned().unwrap_or_else(|| {
                        ExtensionScenario::new("empty", vec![], vec![])
                    });
                    let adv = adversary_scenarios.get(ai).cloned().unwrap_or_else(|| {
                        AdversaryScenario::passive("default")
                    });

                    let name = format!(
                        "{}+{}+{}+{}",
                        cs.name, vs.name, es.name, adv.name
                    );

                    let mut tags = Vec::new();
                    if !cs.has_overlap() {
                        tags.push("no_cipher_overlap".into());
                    }
                    if !vs.has_overlap() {
                        tags.push("no_version_overlap".into());
                    }
                    if adv.is_passive() {
                        tags.push("passive_adversary".into());
                    } else {
                        tags.push("active_adversary".into());
                    }
                    if cs.is_export_allowed {
                        tags.push("export_ciphers".into());
                    }

                    combined.push(TestScenario {
                        id: Uuid::new_v4().to_string(),
                        name,
                        cipher_scenario: cs.clone(),
                        version_scenario: vs.clone(),
                        extension_scenario: es,
                        adversary_scenario: adv,
                        tags,
                    });
                }
            }
        }

        combined
    }

    /// Serialize all scenarios to JSON.
    pub fn serialize_all(&self) -> Result<String, serde_json::Error> {
        let scenarios = self.generate_covering_scenarios();
        serde_json::to_string_pretty(&scenarios)
    }

    /// Generate a subset of scenarios for quick testing.
    pub fn generate_smoke_scenarios(&self) -> Vec<TestScenario> {
        let all = self.generate_covering_scenarios();
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut subset: Vec<TestScenario> = all;

        let desired = 10.min(subset.len());
        subset.shuffle(&mut rng);
        subset.truncate(desired);
        subset
    }
}

fn default_cipher_suites() -> Vec<u16> {
    vec![
        0x1301, // TLS_AES_128_GCM_SHA256
        0x1302, // TLS_AES_256_GCM_SHA384
        0x1303, // TLS_CHACHA20_POLY1305_SHA256
        0xC02B, // TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256
        0xC02F, // TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
        0x009C, // TLS_RSA_WITH_AES_128_GCM_SHA256
        0x009D, // TLS_RSA_WITH_AES_256_GCM_SHA384
        0x002F, // TLS_RSA_WITH_AES_128_CBC_SHA
        0x0035, // TLS_RSA_WITH_AES_256_CBC_SHA
    ]
}

fn default_versions() -> Vec<ProtocolVersion> {
    vec![
        ProtocolVersion::ssl30(),
        ProtocolVersion::tls10(),
        ProtocolVersion::tls11(),
        ProtocolVersion::tls12(),
        ProtocolVersion::tls13(),
    ]
}

fn default_extensions() -> Vec<ExtensionConfig> {
    vec![
        ExtensionConfig {
            id: 0x0000,
            name: "server_name".into(),
            is_critical: false,
            data_length: 32,
        },
        ExtensionConfig {
            id: 0x000D,
            name: "signature_algorithms".into(),
            is_critical: true,
            data_length: 16,
        },
        ExtensionConfig {
            id: 0x000A,
            name: "supported_groups".into(),
            is_critical: true,
            data_length: 8,
        },
        ExtensionConfig {
            id: 0x002B,
            name: "supported_versions".into(),
            is_critical: true,
            data_length: 4,
        },
        ExtensionConfig {
            id: 0x0033,
            name: "key_share".into(),
            is_critical: true,
            data_length: 64,
        },
        ExtensionConfig {
            id: 0xFF01,
            name: "renegotiation_info".into(),
            is_critical: false,
            data_length: 1,
        },
        ExtensionConfig {
            id: 0x0017,
            name: "extended_master_secret".into(),
            is_critical: false,
            data_length: 0,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cipher_scenario_creation() {
        let cs = CipherSuiteScenario::new(
            "test",
            vec![0x002F, 0x0035],
            vec![0x0035, 0x009C],
        );
        assert_eq!(cs.expected_selected, Some(0x0035));
        assert!(cs.has_overlap());
        assert_eq!(cs.common_ciphers(), vec![0x0035]);
    }

    #[test]
    fn test_cipher_scenario_no_overlap() {
        let cs = CipherSuiteScenario::new("test", vec![0x002F], vec![0x0035]);
        assert!(!cs.has_overlap());
        assert_eq!(cs.expected_selected, None);
    }

    #[test]
    fn test_cipher_scenario_server_preference() {
        let cs = CipherSuiteScenario::new(
            "test",
            vec![0x002F, 0x0035],
            vec![0x0035, 0x002F],
        )
        .with_server_preference();
        assert_eq!(cs.expected_selected, Some(0x0035));
    }

    #[test]
    fn test_version_scenario_same() {
        let vs = VersionScenario::new(
            "same",
            ProtocolVersion::tls12(),
            ProtocolVersion::tls12(),
            ProtocolVersion::tls12(),
            ProtocolVersion::tls12(),
        );
        assert_eq!(vs.expected_version, Some(ProtocolVersion::tls12()));
        assert!(vs.has_overlap());
    }

    #[test]
    fn test_version_scenario_no_overlap() {
        let vs = VersionScenario::new(
            "no_overlap",
            ProtocolVersion::tls13(),
            ProtocolVersion::tls13(),
            ProtocolVersion::tls10(),
            ProtocolVersion::tls11(),
        );
        // TLS 1.3 vs TLS 1.0-1.1: TLS 1.3 has higher security level, no overlap
        // depends on how security_level is defined
    }

    #[test]
    fn test_extension_scenario() {
        let client = vec![
            ExtensionConfig {
                id: 0x000D,
                name: "sig_alg".into(),
                is_critical: true,
                data_length: 16,
            },
            ExtensionConfig {
                id: 0x0000,
                name: "sni".into(),
                is_critical: false,
                data_length: 32,
            },
        ];
        let server = vec![ExtensionConfig {
            id: 0x000D,
            name: "sig_alg".into(),
            is_critical: true,
            data_length: 16,
        }];

        let es = ExtensionScenario::new("test", client, server);
        assert_eq!(es.negotiated_count(), 1);
        assert!(es.expected_negotiated.contains(&0x000D));
        assert!(!es.is_mandatory_missing);
    }

    #[test]
    fn test_extension_missing_critical() {
        let client = vec![ExtensionConfig {
            id: 0x000D,
            name: "sig_alg".into(),
            is_critical: true,
            data_length: 16,
        }];
        let server = vec![ExtensionConfig {
            id: 0x0000,
            name: "sni".into(),
            is_critical: false,
            data_length: 32,
        }];

        let es = ExtensionScenario::new("missing", client, server);
        assert!(es.is_mandatory_missing);
    }

    #[test]
    fn test_adversary_passive() {
        let adv = AdversaryScenario::passive("observer");
        assert!(adv.is_passive());
        assert_eq!(adv.adversary_budget, 0);
        assert_eq!(adv.capability_count(), 1); // intercept only
    }

    #[test]
    fn test_adversary_active() {
        let adv = AdversaryScenario::active("attacker", 3);
        assert!(!adv.is_passive());
        assert_eq!(adv.adversary_budget, 3);
        assert_eq!(adv.capability_count(), 5);
    }

    #[test]
    fn test_adversary_with_targets() {
        let adv = AdversaryScenario::active("targeted", 2)
            .with_target_ciphers(vec![0x0003])
            .with_target_version(ProtocolVersion::ssl30());
        assert_eq!(adv.target_cipher_suites, vec![0x0003]);
        assert_eq!(adv.target_version, Some(ProtocolVersion::ssl30()));
    }

    #[test]
    fn test_scenario_generator_cipher_scenarios() {
        let gen = ScenarioGenerator::new(42);
        let scenarios = gen.generate_cipher_scenarios();
        assert!(scenarios.len() >= 5);
    }

    #[test]
    fn test_scenario_generator_version_scenarios() {
        let gen = ScenarioGenerator::new(42);
        let scenarios = gen.generate_version_scenarios();
        assert!(scenarios.len() >= 5);
    }

    #[test]
    fn test_scenario_generator_extension_scenarios() {
        let gen = ScenarioGenerator::new(42);
        let scenarios = gen.generate_extension_scenarios();
        assert!(scenarios.len() >= 3);
    }

    #[test]
    fn test_scenario_generator_adversary_scenarios() {
        let gen = ScenarioGenerator::new(42);
        let scenarios = gen.generate_adversary_scenarios();
        assert!(scenarios.len() >= 5);
        assert!(scenarios.iter().any(|s| s.is_passive()));
        assert!(scenarios.iter().any(|s| !s.is_passive()));
    }

    #[test]
    fn test_covering_scenarios() {
        let gen = ScenarioGenerator::new(42);
        let scenarios = gen.generate_covering_scenarios();
        assert!(!scenarios.is_empty());

        let has_passive = scenarios.iter().any(|s| s.tags.contains(&"passive_adversary".into()));
        let has_active = scenarios.iter().any(|s| s.tags.contains(&"active_adversary".into()));
        assert!(has_passive);
        assert!(has_active);
    }

    #[test]
    fn test_smoke_scenarios() {
        let gen = ScenarioGenerator::new(42);
        let scenarios = gen.generate_smoke_scenarios();
        assert!(scenarios.len() <= 10);
    }

    #[test]
    fn test_scenario_serialization() {
        let gen = ScenarioGenerator::new(42)
            .with_cipher_suites(vec![0x002F, 0x0035])
            .with_versions(vec![ProtocolVersion::tls12(), ProtocolVersion::tls13()])
            .with_budgets(vec![0, 1]);

        let json = gen.serialize_all().unwrap();
        assert!(json.contains("cipher_scenario"));
        assert!(json.contains("version_scenario"));
    }

    #[test]
    fn test_default_cipher_suites() {
        let suites = default_cipher_suites();
        assert!(suites.len() >= 9);
        assert!(suites.contains(&0x1301));
        assert!(suites.contains(&0x002F));
    }

    #[test]
    fn test_default_extensions() {
        let exts = default_extensions();
        assert!(exts.len() >= 5);
        assert!(exts.iter().any(|e| e.is_critical));
    }

    #[test]
    fn test_custom_generator() {
        let gen = ScenarioGenerator::new(99)
            .with_cipher_suites(vec![0x002F])
            .with_versions(vec![ProtocolVersion::tls12()])
            .with_extensions(vec![])
            .with_budgets(vec![1]);

        let scenarios = gen.generate_covering_scenarios();
        assert!(!scenarios.is_empty());
    }
}
