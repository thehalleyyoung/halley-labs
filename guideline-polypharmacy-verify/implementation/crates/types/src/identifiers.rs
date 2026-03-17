//! Strongly-typed identifiers for GuardPharma entities.
//!
//! Each identifier wraps a [`Uuid`] and provides [`Display`], [`FromStr`],
//! [`Hash`], [`Eq`], [`Serialize`], and [`Deserialize`] implementations.
//! Factory functions generate fresh v4 UUIDs.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Macro for generating newtype identifiers
// ---------------------------------------------------------------------------

macro_rules! define_id {
    (
        $(#[$meta:meta])*
        $name:ident, $prefix:expr
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub struct $name(Uuid);

        impl $name {
            /// Create a new random identifier.
            pub fn new() -> Self {
                Self(Uuid::new_v4())
            }

            /// Create an identifier from an existing [`Uuid`].
            pub fn from_uuid(u: Uuid) -> Self {
                Self(u)
            }

            /// Create an identifier from raw bytes.
            pub fn from_bytes(bytes: [u8; 16]) -> Self {
                Self(Uuid::from_bytes(bytes))
            }

            /// Return the inner [`Uuid`].
            pub fn as_uuid(&self) -> &Uuid {
                &self.0
            }

            /// Return the nil (all-zeros) identifier.
            pub fn nil() -> Self {
                Self(Uuid::nil())
            }

            /// Returns `true` if this is the nil identifier.
            pub fn is_nil(&self) -> bool {
                self.0.is_nil()
            }

            /// Return the identifier as a hyphenated lowercase string.
            pub fn to_hyphenated(&self) -> String {
                self.0.to_string()
            }

            /// Prefix used in human-readable representations.
            pub fn prefix() -> &'static str {
                $prefix
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}-{}", $prefix, self.0)
            }
        }

        impl FromStr for $name {
            type Err = IdParseError;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                // Accept both "prefix-uuid" and bare "uuid" forms.
                let uuid_part = if let Some(stripped) = s.strip_prefix(&format!("{}-", $prefix)) {
                    stripped
                } else {
                    s
                };
                let u = Uuid::parse_str(uuid_part).map_err(|e| IdParseError {
                    input: s.to_string(),
                    reason: e.to_string(),
                })?;
                Ok(Self(u))
            }
        }

        impl From<Uuid> for $name {
            fn from(u: Uuid) -> Self {
                Self(u)
            }
        }

        impl From<$name> for Uuid {
            fn from(id: $name) -> Self {
                id.0
            }
        }

        impl AsRef<Uuid> for $name {
            fn as_ref(&self) -> &Uuid {
                &self.0
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Identifier types
// ---------------------------------------------------------------------------

define_id!(
    /// Unique identifier for a clinical practice guideline.
    GuidelineId, "GL"
);

define_id!(
    /// Unique identifier for a patient.
    PatientId, "PT"
);

define_id!(
    /// Unique identifier for a drug–drug interaction record.
    InteractionId, "IX"
);

define_id!(
    /// Unique identifier for a verification run.
    VerificationRunId, "VR"
);

// ---------------------------------------------------------------------------
// IdParseError
// ---------------------------------------------------------------------------

/// Error type returned when parsing an identifier string fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdParseError {
    /// The input string that failed to parse.
    pub input: String,
    /// Human-readable reason for the failure.
    pub reason: String,
}

impl fmt::Display for IdParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "failed to parse identifier from '{}': {}",
            self.input, self.reason
        )
    }
}

impl std::error::Error for IdParseError {}

// ---------------------------------------------------------------------------
// Helper: deterministic ID from a name (for testing / reproducibility)
// ---------------------------------------------------------------------------

/// Create a [`GuidelineId`] from a name by hashing it into a v5 UUID
/// (using the DNS namespace as a stable base).
pub fn guideline_id_from_name(name: &str) -> GuidelineId {
    let u = Uuid::new_v5(&Uuid::NAMESPACE_DNS, name.as_bytes());
    GuidelineId::from_uuid(u)
}

/// Create a [`PatientId`] from a medical record number string.
pub fn patient_id_from_mrn(mrn: &str) -> PatientId {
    let u = Uuid::new_v5(&Uuid::NAMESPACE_DNS, mrn.as_bytes());
    PatientId::from_uuid(u)
}

/// Create an [`InteractionId`] from two drug names (order-independent).
pub fn interaction_id_from_drugs(drug_a: &str, drug_b: &str) -> InteractionId {
    let mut parts = [drug_a, drug_b];
    parts.sort();
    let combined = format!("{}+{}", parts[0], parts[1]);
    let u = Uuid::new_v5(&Uuid::NAMESPACE_DNS, combined.as_bytes());
    InteractionId::from_uuid(u)
}

/// Create a [`VerificationRunId`] from a label and timestamp.
pub fn verification_run_id_from_label(label: &str, timestamp_epoch_secs: u64) -> VerificationRunId {
    let combined = format!("{}@{}", label, timestamp_epoch_secs);
    let u = Uuid::new_v5(&Uuid::NAMESPACE_DNS, combined.as_bytes());
    VerificationRunId::from_uuid(u)
}

// ---------------------------------------------------------------------------
// Batch generation
// ---------------------------------------------------------------------------

/// Generate `n` fresh [`GuidelineId`]s.
pub fn generate_guideline_ids(n: usize) -> Vec<GuidelineId> {
    (0..n).map(|_| GuidelineId::new()).collect()
}

/// Generate `n` fresh [`PatientId`]s.
pub fn generate_patient_ids(n: usize) -> Vec<PatientId> {
    (0..n).map(|_| PatientId::new()).collect()
}

/// Generate `n` fresh [`InteractionId`]s.
pub fn generate_interaction_ids(n: usize) -> Vec<InteractionId> {
    (0..n).map(|_| InteractionId::new()).collect()
}

/// Generate `n` fresh [`VerificationRunId`]s.
pub fn generate_verification_run_ids(n: usize) -> Vec<VerificationRunId> {
    (0..n).map(|_| VerificationRunId::new()).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guideline_id_roundtrip() {
        let id = GuidelineId::new();
        let s = id.to_string();
        let parsed: GuidelineId = s.parse().expect("should parse");
        assert_eq!(id, parsed);
    }

    #[test]
    fn test_patient_id_roundtrip() {
        let id = PatientId::new();
        let s = id.to_string();
        let parsed: PatientId = s.parse().expect("should parse");
        assert_eq!(id, parsed);
    }

    #[test]
    fn test_interaction_id_roundtrip() {
        let id = InteractionId::new();
        let s = id.to_string();
        let parsed: InteractionId = s.parse().expect("should parse");
        assert_eq!(id, parsed);
    }

    #[test]
    fn test_verification_run_id_roundtrip() {
        let id = VerificationRunId::new();
        let s = id.to_string();
        let parsed: VerificationRunId = s.parse().expect("should parse");
        assert_eq!(id, parsed);
    }

    #[test]
    fn test_bare_uuid_parse() {
        let id = GuidelineId::new();
        let bare = id.as_uuid().to_string();
        let parsed: GuidelineId = bare.parse().expect("should parse bare UUID");
        assert_eq!(id, parsed);
    }

    #[test]
    fn test_invalid_uuid_parse() {
        let result: Result<GuidelineId, _> = "not-a-uuid".parse();
        assert!(result.is_err());
    }

    #[test]
    fn test_nil_id() {
        let id = PatientId::nil();
        assert!(id.is_nil());
    }

    #[test]
    fn test_deterministic_guideline_id() {
        let a = guideline_id_from_name("beers-criteria-2023");
        let b = guideline_id_from_name("beers-criteria-2023");
        assert_eq!(a, b);
    }

    #[test]
    fn test_deterministic_interaction_id_order_independent() {
        let a = interaction_id_from_drugs("warfarin", "aspirin");
        let b = interaction_id_from_drugs("aspirin", "warfarin");
        assert_eq!(a, b);
    }

    #[test]
    fn test_deterministic_patient_id() {
        let a = patient_id_from_mrn("MRN-12345");
        let b = patient_id_from_mrn("MRN-12345");
        assert_eq!(a, b);
    }

    #[test]
    fn test_verification_run_id_from_label() {
        let a = verification_run_id_from_label("daily-run", 1700000000);
        let b = verification_run_id_from_label("daily-run", 1700000000);
        assert_eq!(a, b);

        let c = verification_run_id_from_label("daily-run", 1700000001);
        assert_ne!(a, c);
    }

    #[test]
    fn test_batch_generation() {
        let ids = generate_guideline_ids(5);
        assert_eq!(ids.len(), 5);
        // All unique
        let set: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(set.len(), 5);
    }

    #[test]
    fn test_display_prefix() {
        let id = GuidelineId::new();
        assert!(id.to_string().starts_with("GL-"));

        let id = PatientId::new();
        assert!(id.to_string().starts_with("PT-"));

        let id = InteractionId::new();
        assert!(id.to_string().starts_with("IX-"));

        let id = VerificationRunId::new();
        assert!(id.to_string().starts_with("VR-"));
    }

    #[test]
    fn test_serde_roundtrip() {
        let id = GuidelineId::new();
        let json = serde_json::to_string(&id).expect("serialize");
        let parsed: GuidelineId = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(id, parsed);
    }

    #[test]
    fn test_hash_consistency() {
        use std::collections::HashMap;
        let id = PatientId::new();
        let mut map = HashMap::new();
        map.insert(id, "patient data");
        assert_eq!(map.get(&id), Some(&"patient data"));
    }

    #[test]
    fn test_from_bytes() {
        let bytes = [1u8; 16];
        let a = GuidelineId::from_bytes(bytes);
        let b = GuidelineId::from_bytes(bytes);
        assert_eq!(a, b);
    }
}
