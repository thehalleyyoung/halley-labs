//! Serialization and export of certificates in multiple formats.
//!
//! Supported formats include JSON (human-readable) and a compact binary encoding.
//! [`CertificateExporter`] writes certificates out, [`CertificateImporter`] reads
//! them back.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::certificate::Certificate;
use crate::chain::CertificateChain;

// ---------------------------------------------------------------------------
// CertificateFormat
// ---------------------------------------------------------------------------

/// Wire format for certificate serialisation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CertificateFormat {
    /// Pretty-printed JSON.
    Json,
    /// Compact (minified) JSON.
    JsonCompact,
    /// MessagePack-style compact binary (placeholder; uses JSON internally).
    Binary,
}

impl CertificateFormat {
    /// Canonical file extension for this format.
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::JsonCompact => "json",
            Self::Binary => "bin",
        }
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during import / export.
#[derive(Debug, Error)]
pub enum FormatError {
    #[error("serialization failed: {0}")]
    Serialize(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("unsupported format: {0:?}")]
    Unsupported(CertificateFormat),
}

// ---------------------------------------------------------------------------
// CertificateExporter
// ---------------------------------------------------------------------------

/// Writes certificates and chains to bytes or files.
#[derive(Debug, Clone)]
pub struct CertificateExporter {
    /// The output format.
    pub format: CertificateFormat,
}

impl CertificateExporter {
    /// Create an exporter for the given format.
    pub fn new(format: CertificateFormat) -> Self {
        Self { format }
    }

    /// Serialise a single certificate to bytes.
    pub fn export_certificate(&self, cert: &Certificate) -> Result<Vec<u8>, FormatError> {
        self.serialize_value(cert)
    }

    /// Serialise an entire chain to bytes.
    pub fn export_chain(&self, chain: &CertificateChain) -> Result<Vec<u8>, FormatError> {
        self.serialize_value(chain)
    }

    /// Write a certificate to a file at `path`.
    pub fn export_certificate_to_file(
        &self,
        cert: &Certificate,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), FormatError> {
        let bytes = self.export_certificate(cert)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Write a chain to a file at `path`.
    pub fn export_chain_to_file(
        &self,
        chain: &CertificateChain,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), FormatError> {
        let bytes = self.export_chain(chain)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    // Internal helper that dispatches on format.
    fn serialize_value<T: Serialize>(&self, value: &T) -> Result<Vec<u8>, FormatError> {
        match self.format {
            CertificateFormat::Json => {
                let s = serde_json::to_string_pretty(value)?;
                Ok(s.into_bytes())
            }
            CertificateFormat::JsonCompact | CertificateFormat::Binary => {
                let s = serde_json::to_string(value)?;
                Ok(s.into_bytes())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CertificateImporter
// ---------------------------------------------------------------------------

/// Reads certificates and chains from bytes or files.
#[derive(Debug, Clone)]
pub struct CertificateImporter {
    /// Expected format of the input.
    pub format: CertificateFormat,
}

impl CertificateImporter {
    /// Create an importer expecting the given format.
    pub fn new(format: CertificateFormat) -> Self {
        Self { format }
    }

    /// Deserialise a single certificate from bytes.
    pub fn import_certificate(&self, data: &[u8]) -> Result<Certificate, FormatError> {
        self.deserialize_value(data)
    }

    /// Deserialise a chain from bytes.
    pub fn import_chain(&self, data: &[u8]) -> Result<CertificateChain, FormatError> {
        self.deserialize_value(data)
    }

    /// Read a certificate from a file.
    pub fn import_certificate_from_file(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<Certificate, FormatError> {
        let data = std::fs::read(path)?;
        self.import_certificate(&data)
    }

    /// Read a chain from a file.
    pub fn import_chain_from_file(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<CertificateChain, FormatError> {
        let data = std::fs::read(path)?;
        self.import_chain(&data)
    }

    fn deserialize_value<T: for<'de> Deserialize<'de>>(&self, data: &[u8]) -> Result<T, FormatError> {
        let value = serde_json::from_slice(data)?;
        Ok(value)
    }
}
