//! Contract persistence and serialisation.
//!
//! Provides a [`ContractStore`] trait for pluggable storage backends and a
//! default JSON-file-based [`ContractDatabase`] implementation.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::contract::LeakageContract;

// ---------------------------------------------------------------------------
// Contract version
// ---------------------------------------------------------------------------

/// A versioned snapshot of a leakage contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractVersion {
    /// Monotonically increasing version number.
    pub version: u64,
    /// Content hash of the contract at this version.
    pub content_hash: String,
    /// ISO-8601 timestamp of when this version was stored.
    pub timestamp: String,
    /// Optional human-readable label (e.g., git tag).
    pub label: Option<String>,
    /// The serialised contract.
    pub contract: LeakageContract,
}

impl ContractVersion {
    /// Create a new version entry.
    pub fn new(version: u64, contract: LeakageContract) -> Self {
        let content_hash = contract.compute_hash();
        Self {
            version,
            content_hash,
            timestamp: chrono::Utc::now().to_rfc3339(),
            label: None,
            contract,
        }
    }

    /// Builder: add a label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

impl fmt::Display for ContractVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "v{} [{}] {} ({})",
            self.version,
            self.content_hash
                .get(..8)
                .unwrap_or(&self.content_hash),
            self.contract.function_name,
            self.timestamp
        )
    }
}

// ---------------------------------------------------------------------------
// Contract store trait
// ---------------------------------------------------------------------------

/// Pluggable storage backend for leakage contracts.
pub trait ContractStore {
    /// Store a contract, returning the assigned version number.
    fn store(&mut self, contract: &LeakageContract) -> Result<u64, StorageError>;

    /// Load the latest version of a contract by function name.
    fn load_latest(&self, function_name: &str) -> Result<Option<ContractVersion>, StorageError>;

    /// Load a specific version.
    fn load_version(
        &self,
        function_name: &str,
        version: u64,
    ) -> Result<Option<ContractVersion>, StorageError>;

    /// List all stored function names.
    fn list_functions(&self) -> Result<Vec<String>, StorageError>;

    /// List all versions for a given function.
    fn list_versions(&self, function_name: &str) -> Result<Vec<ContractVersion>, StorageError>;

    /// Delete all versions of a contract.
    fn delete(&mut self, function_name: &str) -> Result<(), StorageError>;
}

/// Errors from storage operations.
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialisation error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("contract not found: {name}")]
    NotFound { name: String },

    #[error("storage error: {reason}")]
    Other { reason: String },
}

// ---------------------------------------------------------------------------
// JSON-file contract database
// ---------------------------------------------------------------------------

/// In-memory contract database backed by a directory of JSON files.
#[derive(Debug, Clone)]
pub struct ContractDatabase {
    /// Root directory for contract storage.
    pub root: PathBuf,
    /// In-memory index: function_name → versions (sorted ascending).
    contracts: BTreeMap<String, Vec<ContractVersion>>,
}

impl ContractDatabase {
    /// Create a new database rooted at the given directory.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            contracts: BTreeMap::new(),
        }
    }

    /// Create an in-memory-only database (useful for testing).
    pub fn in_memory() -> Self {
        Self {
            root: PathBuf::from(":memory:"),
            contracts: BTreeMap::new(),
        }
    }

    /// Load all contracts from the root directory.
    pub fn load_from_disk(&mut self) -> Result<usize, StorageError> {
        let mut count = 0;
        if !self.root.exists() {
            return Ok(0);
        }
        for entry in std::fs::read_dir(&self.root)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                let data = std::fs::read_to_string(&path)?;
                let contract: LeakageContract = serde_json::from_str(&data)?;
                self.store(&contract)?;
                count += 1;
            }
        }
        Ok(count)
    }

    /// Persist a contract version to disk.
    pub fn persist(&self, version: &ContractVersion) -> Result<(), StorageError> {
        if self.root.to_str() == Some(":memory:") {
            return Ok(());
        }
        std::fs::create_dir_all(&self.root)?;
        let filename = format!(
            "{}_v{}.json",
            version.contract.function_name, version.version
        );
        let path = self.root.join(filename);
        let json = serde_json::to_string_pretty(&version)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Number of functions stored.
    pub fn function_count(&self) -> usize {
        self.contracts.len()
    }

    /// Total number of versions across all functions.
    pub fn total_versions(&self) -> usize {
        self.contracts.values().map(|v| v.len()).sum()
    }
}

impl ContractStore for ContractDatabase {
    fn store(&mut self, contract: &LeakageContract) -> Result<u64, StorageError> {
        let versions = self
            .contracts
            .entry(contract.function_name.clone())
            .or_default();
        let next_version = versions.last().map_or(1, |v| v.version + 1);
        let cv = ContractVersion::new(next_version, contract.clone());
        versions.push(cv);
        Ok(next_version)
    }

    fn load_latest(&self, function_name: &str) -> Result<Option<ContractVersion>, StorageError> {
        Ok(self
            .contracts
            .get(function_name)
            .and_then(|vs| vs.last().cloned()))
    }

    fn load_version(
        &self,
        function_name: &str,
        version: u64,
    ) -> Result<Option<ContractVersion>, StorageError> {
        Ok(self
            .contracts
            .get(function_name)
            .and_then(|vs| vs.iter().find(|v| v.version == version).cloned()))
    }

    fn list_functions(&self) -> Result<Vec<String>, StorageError> {
        Ok(self.contracts.keys().cloned().collect())
    }

    fn list_versions(&self, function_name: &str) -> Result<Vec<ContractVersion>, StorageError> {
        Ok(self
            .contracts
            .get(function_name)
            .cloned()
            .unwrap_or_default())
    }

    fn delete(&mut self, function_name: &str) -> Result<(), StorageError> {
        self.contracts.remove(function_name);
        Ok(())
    }
}
