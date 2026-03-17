//! Serialization utilities for the CollusionProof system.
//!
//! Provides JSON and bincode helpers, versioned serialization with schema
//! evolution, checksum computation, and data integrity verification.

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Serialization format enum
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Supported serialization formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SerializationFormat {
    Json,
    JsonPretty,
    Bincode,
}

impl fmt::Display for SerializationFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SerializationFormat::Json => write!(f, "JSON"),
            SerializationFormat::JsonPretty => write!(f, "JSON(pretty)"),
            SerializationFormat::Bincode => write!(f, "bincode"),
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// JSON helpers
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Serialize a value to a compact JSON string.
pub fn to_json<T: Serialize>(value: &T) -> Result<String, String> {
    serde_json::to_string(value).map_err(|e| format!("JSON serialization failed: {}", e))
}

/// Serialize a value to a pretty-printed JSON string.
pub fn to_json_pretty<T: Serialize>(value: &T) -> Result<String, String> {
    serde_json::to_string_pretty(value).map_err(|e| format!("JSON serialization failed: {}", e))
}

/// Deserialize a value from a JSON string.
pub fn from_json<T: DeserializeOwned>(json: &str) -> Result<T, String> {
    serde_json::from_str(json).map_err(|e| format!("JSON deserialization failed: {}", e))
}

/// Serialize to JSON bytes.
pub fn to_json_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>, String> {
    serde_json::to_vec(value).map_err(|e| format!("JSON serialization failed: {}", e))
}

/// Deserialize from JSON bytes.
pub fn from_json_bytes<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, String> {
    serde_json::from_slice(bytes).map_err(|e| format!("JSON deserialization failed: {}", e))
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Bincode helpers
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Serialize a value to bincode bytes.
pub fn to_bincode<T: Serialize>(value: &T) -> Result<Vec<u8>, String> {
    bincode::serialize(value).map_err(|e| format!("bincode serialization failed: {}", e))
}

/// Deserialize a value from bincode bytes.
pub fn from_bincode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, String> {
    bincode::deserialize(bytes).map_err(|e| format!("bincode deserialization failed: {}", e))
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Generic serialization
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Serialize a value to bytes in the specified format.
pub fn serialize<T: Serialize>(value: &T, format: SerializationFormat) -> Result<Vec<u8>, String> {
    match format {
        SerializationFormat::Json => to_json_bytes(value),
        SerializationFormat::JsonPretty => to_json_pretty(value).map(|s| s.into_bytes()),
        SerializationFormat::Bincode => to_bincode(value),
    }
}

/// Deserialize a value from bytes in the specified format.
pub fn deserialize<T: DeserializeOwned>(
    bytes: &[u8],
    format: SerializationFormat,
) -> Result<T, String> {
    match format {
        SerializationFormat::Json | SerializationFormat::JsonPretty => from_json_bytes(bytes),
        SerializationFormat::Bincode => from_bincode(bytes),
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Versioned serialization
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A versioned wrapper for schema evolution support.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Versioned<T> {
    pub version: u32,
    pub schema_name: String,
    pub data: T,
    pub checksum: String,
}

impl<T: Serialize + DeserializeOwned> Versioned<T> {
    /// Create a new versioned wrapper.
    pub fn new(data: T, schema_name: impl Into<String>, version: u32) -> Result<Self, String> {
        let json = serde_json::to_string(&data)
            .map_err(|e| format!("versioned serialization failed: {}", e))?;
        let checksum = compute_sha256(json.as_bytes());
        Ok(Versioned {
            version,
            schema_name: schema_name.into(),
            data,
            checksum,
        })
    }

    /// Verify the checksum matches the current data.
    pub fn verify_checksum(&self) -> bool {
        let json = match serde_json::to_string(&self.data) {
            Ok(j) => j,
            Err(_) => return false,
        };
        let computed = compute_sha256(json.as_bytes());
        computed == self.checksum
    }

    /// Serialize the entire versioned envelope to JSON.
    pub fn to_json(&self) -> Result<String, String>
    where
        T: Serialize,
    {
        to_json_pretty(self)
    }

    /// Deserialize from JSON, verifying version and checksum.
    pub fn from_json_verified(json: &str, expected_schema: &str) -> Result<Self, String>
    where
        T: DeserializeOwned,
    {
        let versioned: Versioned<T> = from_json(json)?;
        if versioned.schema_name != expected_schema {
            return Err(format!(
                "schema mismatch: expected '{}', got '{}'",
                expected_schema, versioned.schema_name
            ));
        }
        if !versioned.verify_checksum() {
            return Err("checksum verification failed".into());
        }
        Ok(versioned)
    }
}

impl<T: fmt::Debug> fmt::Display for Versioned<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Versioned<{}>(v{}, checksum={}...)",
            self.schema_name,
            self.version,
            &self.checksum[..8]
        )
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Checksum and integrity
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Compute a SHA-256 hash of the given bytes, returned as a hex string.
pub fn compute_sha256(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

/// Compute SHA-256 of a serializable value (by first converting to JSON).
pub fn compute_content_hash<T: Serialize>(value: &T) -> Result<String, String> {
    let json = serde_json::to_string(value)
        .map_err(|e| format!("cannot hash value: {}", e))?;
    Ok(compute_sha256(json.as_bytes()))
}

/// Verify that the SHA-256 of `data` matches `expected_hash`.
pub fn verify_sha256(data: &[u8], expected_hash: &str) -> bool {
    compute_sha256(data) == expected_hash
}

/// Verify content integrity of a serializable value against an expected hash.
pub fn verify_content_integrity<T: Serialize>(value: &T, expected_hash: &str) -> Result<bool, String> {
    let hash = compute_content_hash(value)?;
    Ok(hash == expected_hash)
}

/// An integrity-checked data payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrityEnvelope<T> {
    pub data: T,
    pub sha256: String,
    pub format: SerializationFormat,
}

impl<T: Serialize + DeserializeOwned + Clone> IntegrityEnvelope<T> {
    /// Create a new envelope with computed checksum.
    pub fn seal(data: T, format: SerializationFormat) -> Result<Self, String> {
        let bytes = serialize(&data, format)?;
        let sha256 = compute_sha256(&bytes);
        Ok(IntegrityEnvelope { data, sha256, format })
    }

    /// Verify the integrity of the payload.
    pub fn verify(&self) -> bool {
        match serialize(&self.data, self.format) {
            Ok(bytes) => compute_sha256(&bytes) == self.sha256,
            Err(_) => false,
        }
    }

    /// Extract the data, checking integrity first.
    pub fn open(self) -> Result<T, String> {
        if self.verify() {
            Ok(self.data)
        } else {
            Err("integrity check failed: data has been modified".into())
        }
    }
}

impl<T: fmt::Debug> fmt::Display for IntegrityEnvelope<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Envelope({}, hash={}...)", self.format, &self.sha256[..8])
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestData {
        x: f64,
        label: String,
    }

    #[test]
    fn test_json_roundtrip() {
        let data = TestData { x: 3.14, label: "pi".into() };
        let json = to_json(&data).unwrap();
        let data2: TestData = from_json(&json).unwrap();
        assert_eq!(data, data2);
    }

    #[test]
    fn test_json_pretty() {
        let data = TestData { x: 1.0, label: "one".into() };
        let json = to_json_pretty(&data).unwrap();
        assert!(json.contains('\n'));
    }

    #[test]
    fn test_bincode_roundtrip() {
        let data = TestData { x: 2.71, label: "e".into() };
        let bytes = to_bincode(&data).unwrap();
        let data2: TestData = from_bincode(&bytes).unwrap();
        assert_eq!(data, data2);
    }

    #[test]
    fn test_generic_serialize() {
        let data = TestData { x: 42.0, label: "answer".into() };
        for fmt in [SerializationFormat::Json, SerializationFormat::JsonPretty, SerializationFormat::Bincode] {
            let bytes = serialize(&data, fmt).unwrap();
            let data2: TestData = deserialize(&bytes, fmt).unwrap();
            assert_eq!(data, data2);
        }
    }

    #[test]
    fn test_sha256() {
        let hash = compute_sha256(b"hello world");
        assert_eq!(hash.len(), 64);
        // Known SHA-256 of "hello world"
        assert_eq!(hash, "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9");
    }

    #[test]
    fn test_verify_sha256() {
        let data = b"test data";
        let hash = compute_sha256(data);
        assert!(verify_sha256(data, &hash));
        assert!(!verify_sha256(b"other data", &hash));
    }

    #[test]
    fn test_content_hash() {
        let data = TestData { x: 1.0, label: "a".into() };
        let h1 = compute_content_hash(&data).unwrap();
        let h2 = compute_content_hash(&data).unwrap();
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_content_integrity() {
        let data = TestData { x: 1.0, label: "a".into() };
        let hash = compute_content_hash(&data).unwrap();
        assert!(verify_content_integrity(&data, &hash).unwrap());
        let modified = TestData { x: 2.0, label: "a".into() };
        assert!(!verify_content_integrity(&modified, &hash).unwrap());
    }

    #[test]
    fn test_versioned_roundtrip() {
        let data = TestData { x: 1.0, label: "test".into() };
        let v = Versioned::new(data.clone(), "TestData", 1).unwrap();
        assert!(v.verify_checksum());
        let json = v.to_json().unwrap();
        let v2 = Versioned::<TestData>::from_json_verified(&json, "TestData").unwrap();
        assert_eq!(v2.data, data);
        assert_eq!(v2.version, 1);
    }

    #[test]
    fn test_versioned_schema_mismatch() {
        let data = TestData { x: 1.0, label: "test".into() };
        let v = Versioned::new(data, "TestData", 1).unwrap();
        let json = v.to_json().unwrap();
        let result = Versioned::<TestData>::from_json_verified(&json, "WrongSchema");
        assert!(result.is_err());
    }

    #[test]
    fn test_integrity_envelope() {
        let data = TestData { x: 5.0, label: "five".into() };
        let env = IntegrityEnvelope::seal(data.clone(), SerializationFormat::Json).unwrap();
        assert!(env.verify());
        let opened = env.open().unwrap();
        assert_eq!(opened, data);
    }

    #[test]
    fn test_integrity_envelope_bincode() {
        let data = TestData { x: 7.0, label: "seven".into() };
        let env = IntegrityEnvelope::seal(data.clone(), SerializationFormat::Bincode).unwrap();
        assert!(env.verify());
    }

    #[test]
    fn test_json_bytes_roundtrip() {
        let data = TestData { x: 9.0, label: "nine".into() };
        let bytes = to_json_bytes(&data).unwrap();
        let data2: TestData = from_json_bytes(&bytes).unwrap();
        assert_eq!(data, data2);
    }

    #[test]
    fn test_versioned_display() {
        let data = TestData { x: 1.0, label: "a".into() };
        let v = Versioned::new(data, "TestData", 1).unwrap();
        let s = format!("{}", v);
        assert!(s.contains("TestData") && s.contains("v1"));
    }
}
