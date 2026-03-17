//! UUID-based identifier types for the CollusionProof system.
//!
//! Each identifier encodes a UUID v4 and a creation timestamp, supports
//! Display, Hash, Eq, ordering by timestamp, and serialization.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Macro for identifier types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

macro_rules! define_id_type {
    ($(#[$meta:meta])* $name:ident, $prefix:expr) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Serialize, Deserialize)]
        pub struct $name {
            uuid: Uuid,
            created_at: DateTime<Utc>,
        }

        impl $name {
            /// Create a new random identifier with the current timestamp.
            pub fn new() -> Self {
                $name {
                    uuid: Uuid::new_v4(),
                    created_at: Utc::now(),
                }
            }

            /// Create from an existing UUID and timestamp.
            pub fn from_parts(uuid: Uuid, created_at: DateTime<Utc>) -> Self {
                $name { uuid, created_at }
            }

            /// Create a deterministic identifier from a string (for testing).
            pub fn from_name(name: &str) -> Self {
                let uuid = Uuid::new_v5(&Uuid::NAMESPACE_OID, name.as_bytes());
                $name {
                    uuid,
                    created_at: Utc::now(),
                }
            }

            /// Return the underlying UUID.
            pub fn uuid(&self) -> Uuid {
                self.uuid
            }

            /// Return the creation timestamp.
            pub fn created_at(&self) -> DateTime<Utc> {
                self.created_at
            }

            /// Short display string (prefix + first 8 hex chars).
            pub fn short(&self) -> String {
                let hex = self.uuid.as_simple().to_string();
                format!("{}-{}", $prefix, &hex[..8])
            }
        }

        impl Default for $name {
            fn default() -> Self { $name::new() }
        }

        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.uuid == other.uuid
            }
        }

        impl Eq for $name {}

        impl std::hash::Hash for $name {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                self.uuid.hash(state);
            }
        }

        impl PartialOrd for $name {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for $name {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.created_at.cmp(&other.created_at)
                    .then_with(|| self.uuid.cmp(&other.uuid))
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}-{}", $prefix, self.uuid)
            }
        }

        impl AsRef<Uuid> for $name {
            fn as_ref(&self) -> &Uuid {
                &self.uuid
            }
        }
    };
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Identifier types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

define_id_type!(
    /// Unique identifier for a simulation run.
    SimulationId, "sim"
);

define_id_type!(
    /// Unique identifier for a certificate.
    CertificateId, "cert"
);

define_id_type!(
    /// Unique identifier for a statistical test.
    TestId, "test"
);

define_id_type!(
    /// Unique identifier for a deviation/counterfactual scenario.
    ScenarioId, "scen"
);

define_id_type!(
    /// Unique identifier for an evidence bundle.
    BundleId, "bndl"
);

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Utility functions
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Parse a UUID string into a `Uuid`.
pub fn parse_uuid(s: &str) -> Result<Uuid, uuid::Error> {
    // Strip any prefix before the UUID (e.g., "sim-..." → "...")
    let uuid_str = if let Some(pos) = s.find('-') {
        let after = &s[pos + 1..];
        // Check if the part after the first dash looks like a UUID
        if after.len() >= 32 {
            after
        } else {
            s
        }
    } else {
        s
    };
    Uuid::parse_str(uuid_str)
}

/// Generate a batch of simulation IDs.
pub fn generate_simulation_batch(count: usize) -> Vec<SimulationId> {
    (0..count).map(|_| SimulationId::new()).collect()
}

/// Generate a deterministic set of test IDs from names.
pub fn generate_named_test_ids(names: &[&str]) -> Vec<TestId> {
    names.iter().map(|n| TestId::from_name(n)).collect()
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_id_new() {
        let id = SimulationId::new();
        let s = format!("{}", id);
        assert!(s.starts_with("sim-"));
    }

    #[test]
    fn test_certificate_id_new() {
        let id = CertificateId::new();
        let s = format!("{}", id);
        assert!(s.starts_with("cert-"));
    }

    #[test]
    fn test_id_equality() {
        let a = TestId::from_name("test-alpha");
        let b = TestId::from_name("test-alpha");
        let c = TestId::from_name("test-beta");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_id_hash() {
        use std::collections::HashSet;
        let a = TestId::from_name("x");
        let b = TestId::from_name("x");
        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_id_short() {
        let id = SimulationId::new();
        let short = id.short();
        assert!(short.starts_with("sim-"));
        assert!(short.len() < 20);
    }

    #[test]
    fn test_id_serialization() {
        let id = ScenarioId::new();
        let json = serde_json::to_string(&id).unwrap();
        let id2: ScenarioId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, id2);
    }

    #[test]
    fn test_id_ordering() {
        let a = SimulationId::new();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let b = SimulationId::new();
        assert!(a < b);
    }

    #[test]
    fn test_bundle_id() {
        let id = BundleId::new();
        assert!(format!("{}", id).starts_with("bndl-"));
    }

    #[test]
    fn test_generate_batch() {
        let batch = generate_simulation_batch(5);
        assert_eq!(batch.len(), 5);
        // All unique
        let unique: std::collections::HashSet<_> = batch.iter().collect();
        assert_eq!(unique.len(), 5);
    }

    #[test]
    fn test_named_test_ids() {
        let ids = generate_named_test_ids(&["a", "b", "c"]);
        assert_eq!(ids.len(), 3);
        assert_ne!(ids[0], ids[1]);
    }

    #[test]
    fn test_uuid_accessor() {
        let id = TestId::new();
        let _uuid: Uuid = id.uuid();
        let _ref: &Uuid = id.as_ref();
    }

    #[test]
    fn test_default() {
        let id: SimulationId = Default::default();
        assert!(!format!("{}", id).is_empty());
    }

    #[test]
    fn test_from_parts() {
        let uuid = Uuid::new_v4();
        let ts = Utc::now();
        let id = CertificateId::from_parts(uuid, ts);
        assert_eq!(id.uuid(), uuid);
        assert_eq!(id.created_at(), ts);
    }

    #[test]
    fn test_parse_uuid() {
        let id = SimulationId::new();
        let s = format!("{}", id);
        let parsed = parse_uuid(&s);
        assert!(parsed.is_ok());
    }
}
