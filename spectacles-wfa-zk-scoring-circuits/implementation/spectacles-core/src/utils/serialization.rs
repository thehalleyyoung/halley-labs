//! Serialization utilities for proofs and proof components.
//!
//! Provides compact binary serialization, size estimation, and format versioning.

use serde::{Serialize, Deserialize};
use std::io::{self, Read, Write};

/// Supported proof serialization formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProofFormat {
    /// Compact binary format (default)
    CompactBinary,
    /// JSON format (human-readable, larger)
    Json,
    /// CBOR format (compact, schema-less)
    Cbor,
}

/// Format version for forward compatibility
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FormatVersion {
    pub major: u16,
    pub minor: u16,
}

impl FormatVersion {
    pub const CURRENT: Self = Self { major: 1, minor: 0 };
    
    pub fn is_compatible(&self, other: &Self) -> bool {
        self.major == other.major
    }
}

/// Header for serialized proofs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofHeader {
    pub magic: [u8; 4],
    pub version: FormatVersion,
    pub format: ProofFormat,
    pub payload_size: u64,
    pub checksum: [u8; 32],
}

impl ProofHeader {
    pub const MAGIC: [u8; 4] = [b'S', b'P', b'C', b'T']; // "SPCT"
    
    pub fn new(format: ProofFormat, payload_size: u64, checksum: [u8; 32]) -> Self {
        Self {
            magic: Self::MAGIC,
            version: FormatVersion::CURRENT,
            format,
            payload_size,
            checksum,
        }
    }
    
    pub fn validate(&self) -> Result<(), String> {
        if self.magic != Self::MAGIC {
            return Err("Invalid magic bytes".to_string());
        }
        if !self.version.is_compatible(&FormatVersion::CURRENT) {
            return Err(format!(
                "Incompatible version: {}.{} (expected {}.x)",
                self.version.major, self.version.minor,
                FormatVersion::CURRENT.major
            ));
        }
        Ok(())
    }
    
    /// Serialize header to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(48);
        buf.extend_from_slice(&self.magic);
        buf.extend_from_slice(&self.version.major.to_le_bytes());
        buf.extend_from_slice(&self.version.minor.to_le_bytes());
        buf.push(match self.format {
            ProofFormat::CompactBinary => 0,
            ProofFormat::Json => 1,
            ProofFormat::Cbor => 2,
        });
        buf.extend_from_slice(&self.payload_size.to_le_bytes());
        buf.extend_from_slice(&self.checksum);
        buf
    }
    
    /// Deserialize header from bytes
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        if data.len() < 47 {
            return Err("Header too short".to_string());
        }
        
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&data[0..4]);
        
        let major = u16::from_le_bytes([data[4], data[5]]);
        let minor = u16::from_le_bytes([data[6], data[7]]);
        
        let format = match data[8] {
            0 => ProofFormat::CompactBinary,
            1 => ProofFormat::Json,
            2 => ProofFormat::Cbor,
            _ => return Err("Unknown format".to_string()),
        };
        
        let payload_size = u64::from_le_bytes([
            data[9], data[10], data[11], data[12],
            data[13], data[14], data[15], data[16],
        ]);
        
        let mut checksum = [0u8; 32];
        checksum.copy_from_slice(&data[17..49]);
        
        let header = Self {
            magic,
            version: FormatVersion { major, minor },
            format,
            payload_size,
            checksum,
        };
        
        header.validate()?;
        Ok(header)
    }
}

/// A compact proof representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactProof {
    pub metric_id: String,
    pub candidate_hash: [u8; 32],
    pub reference_hash: [u8; 32],
    pub score_numerator: u64,
    pub score_denominator: u64,
    pub commitment: [u8; 32],
    pub proof_data: Vec<u8>,
}

impl CompactProof {
    pub fn new(
        metric_id: String,
        candidate_hash: [u8; 32],
        reference_hash: [u8; 32],
        score_num: u64,
        score_den: u64,
        commitment: [u8; 32],
        proof_data: Vec<u8>,
    ) -> Self {
        Self {
            metric_id,
            candidate_hash,
            reference_hash,
            score_numerator: score_num,
            score_denominator: score_den,
            commitment,
            proof_data,
        }
    }
    
    /// Estimate the serialized size in bytes
    pub fn estimated_size(&self) -> usize {
        let fixed = 32 + 32 + 8 + 8 + 32; // hashes + score + commitment
        let variable = self.metric_id.len() + self.proof_data.len();
        fixed + variable + 16 // overhead
    }
}

/// Proof serializer
#[derive(Debug, Clone)]
pub struct ProofSerializer {
    format: ProofFormat,
}

impl ProofSerializer {
    pub fn new(format: ProofFormat) -> Self {
        Self { format }
    }
    
    pub fn compact_binary() -> Self {
        Self::new(ProofFormat::CompactBinary)
    }
    
    pub fn json() -> Self {
        Self::new(ProofFormat::Json)
    }
    
    /// Serialize a compact proof
    pub fn serialize(&self, proof: &CompactProof) -> Result<Vec<u8>, String> {
        match self.format {
            ProofFormat::CompactBinary => self.serialize_binary(proof),
            ProofFormat::Json => self.serialize_json(proof),
            ProofFormat::Cbor => self.serialize_binary(proof), // fallback to binary
        }
    }
    
    /// Deserialize a compact proof
    pub fn deserialize(&self, data: &[u8]) -> Result<CompactProof, String> {
        match self.format {
            ProofFormat::CompactBinary => self.deserialize_binary(data),
            ProofFormat::Json => self.deserialize_json(data),
            ProofFormat::Cbor => self.deserialize_binary(data),
        }
    }
    
    fn serialize_binary(&self, proof: &CompactProof) -> Result<Vec<u8>, String> {
        let mut buf = Vec::new();
        
        // Metric ID (length-prefixed)
        let id_bytes = proof.metric_id.as_bytes();
        buf.extend_from_slice(&(id_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(id_bytes);
        
        // Hashes
        buf.extend_from_slice(&proof.candidate_hash);
        buf.extend_from_slice(&proof.reference_hash);
        
        // Score
        buf.extend_from_slice(&proof.score_numerator.to_le_bytes());
        buf.extend_from_slice(&proof.score_denominator.to_le_bytes());
        
        // Commitment
        buf.extend_from_slice(&proof.commitment);
        
        // Proof data (length-prefixed)
        buf.extend_from_slice(&(proof.proof_data.len() as u64).to_le_bytes());
        buf.extend_from_slice(&proof.proof_data);
        
        // Compute checksum
        let checksum = blake3::hash(&buf);
        let header = ProofHeader::new(
            ProofFormat::CompactBinary,
            buf.len() as u64,
            *checksum.as_bytes(),
        );
        
        let mut result = header.to_bytes();
        result.extend_from_slice(&buf);
        
        Ok(result)
    }
    
    fn deserialize_binary(&self, data: &[u8]) -> Result<CompactProof, String> {
        if data.len() < 49 {
            return Err("Data too short for header".to_string());
        }
        
        let header = ProofHeader::from_bytes(data)?;
        let payload = &data[49..];
        
        if payload.len() < header.payload_size as usize {
            return Err("Payload too short".to_string());
        }
        
        let mut pos = 0;
        
        // Metric ID
        if pos + 4 > payload.len() { return Err("Truncated metric ID length".to_string()); }
        let id_len = u32::from_le_bytes([payload[pos], payload[pos+1], payload[pos+2], payload[pos+3]]) as usize;
        pos += 4;
        
        if pos + id_len > payload.len() { return Err("Truncated metric ID".to_string()); }
        let metric_id = String::from_utf8(payload[pos..pos+id_len].to_vec())
            .map_err(|e| format!("Invalid metric ID: {}", e))?;
        pos += id_len;
        
        // Hashes
        if pos + 64 > payload.len() { return Err("Truncated hashes".to_string()); }
        let mut candidate_hash = [0u8; 32];
        candidate_hash.copy_from_slice(&payload[pos..pos+32]);
        pos += 32;
        let mut reference_hash = [0u8; 32];
        reference_hash.copy_from_slice(&payload[pos..pos+32]);
        pos += 32;
        
        // Score
        if pos + 16 > payload.len() { return Err("Truncated score".to_string()); }
        let score_numerator = u64::from_le_bytes([
            payload[pos], payload[pos+1], payload[pos+2], payload[pos+3],
            payload[pos+4], payload[pos+5], payload[pos+6], payload[pos+7],
        ]);
        pos += 8;
        let score_denominator = u64::from_le_bytes([
            payload[pos], payload[pos+1], payload[pos+2], payload[pos+3],
            payload[pos+4], payload[pos+5], payload[pos+6], payload[pos+7],
        ]);
        pos += 8;
        
        // Commitment
        if pos + 32 > payload.len() { return Err("Truncated commitment".to_string()); }
        let mut commitment = [0u8; 32];
        commitment.copy_from_slice(&payload[pos..pos+32]);
        pos += 32;
        
        // Proof data
        if pos + 8 > payload.len() { return Err("Truncated proof data length".to_string()); }
        let proof_len = u64::from_le_bytes([
            payload[pos], payload[pos+1], payload[pos+2], payload[pos+3],
            payload[pos+4], payload[pos+5], payload[pos+6], payload[pos+7],
        ]) as usize;
        pos += 8;
        
        if pos + proof_len > payload.len() { return Err("Truncated proof data".to_string()); }
        let proof_data = payload[pos..pos+proof_len].to_vec();
        
        Ok(CompactProof {
            metric_id,
            candidate_hash,
            reference_hash,
            score_numerator,
            score_denominator,
            commitment,
            proof_data,
        })
    }
    
    fn serialize_json(&self, proof: &CompactProof) -> Result<Vec<u8>, String> {
        serde_json::to_vec(proof).map_err(|e| format!("JSON serialization failed: {}", e))
    }
    
    fn deserialize_json(&self, data: &[u8]) -> Result<CompactProof, String> {
        serde_json::from_slice(data).map_err(|e| format!("JSON deserialization failed: {}", e))
    }
}

/// Estimate proof size for a given metric configuration
pub fn estimate_proof_size(
    num_constraints: usize,
    num_wires: usize,
    security_bits: usize,
) -> usize {
    // STARK proof size estimation
    let field_element_size = 8; // Goldilocks = 64 bits
    let hash_size = 32; // BLAKE3
    
    // Trace polynomial commitments
    let trace_size = num_wires * field_element_size;
    
    // FRI proof layers
    let fri_layers = (num_constraints as f64).log2().ceil() as usize + 1;
    let fri_layer_size = hash_size * 2; // Merkle authentication paths
    let fri_total = fri_layers * fri_layer_size * security_bits / 8;
    
    // Constraint evaluation proofs
    let constraint_proof_size = num_constraints * field_element_size;
    
    // Total estimate
    trace_size + fri_total + constraint_proof_size + 256 // overhead
}

/// Compress proof data using simple run-length encoding
pub fn compress_rle(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }
    
    let mut compressed = Vec::new();
    let mut i = 0;
    
    while i < data.len() {
        let byte = data[i];
        let mut count = 1u8;
        
        while i + (count as usize) < data.len()
            && data[i + (count as usize)] == byte
            && count < 255
        {
            count += 1;
        }
        
        if count >= 3 || byte == 0xFF {
            // RLE marker: 0xFF, count, byte
            compressed.push(0xFF);
            compressed.push(count);
            compressed.push(byte);
        } else {
            for _ in 0..count {
                compressed.push(byte);
            }
        }
        
        i += count as usize;
    }
    
    compressed
}

/// Decompress RLE data
pub fn decompress_rle(data: &[u8]) -> Vec<u8> {
    let mut decompressed = Vec::new();
    let mut i = 0;
    
    while i < data.len() {
        if data[i] == 0xFF && i + 2 < data.len() {
            let count = data[i + 1];
            let byte = data[i + 2];
            for _ in 0..count {
                decompressed.push(byte);
            }
            i += 3;
        } else {
            decompressed.push(data[i]);
            i += 1;
        }
    }
    
    decompressed
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    fn sample_proof() -> CompactProof {
        CompactProof {
            metric_id: "bleu".to_string(),
            candidate_hash: [1u8; 32],
            reference_hash: [2u8; 32],
            score_numerator: 75,
            score_denominator: 100,
            commitment: [3u8; 32],
            proof_data: vec![10, 20, 30, 40, 50],
        }
    }
    
    #[test]
    fn test_binary_roundtrip() {
        let serializer = ProofSerializer::compact_binary();
        let proof = sample_proof();
        
        let bytes = serializer.serialize(&proof).unwrap();
        let decoded = serializer.deserialize(&bytes).unwrap();
        
        assert_eq!(decoded.metric_id, proof.metric_id);
        assert_eq!(decoded.candidate_hash, proof.candidate_hash);
        assert_eq!(decoded.reference_hash, proof.reference_hash);
        assert_eq!(decoded.score_numerator, proof.score_numerator);
        assert_eq!(decoded.score_denominator, proof.score_denominator);
        assert_eq!(decoded.commitment, proof.commitment);
        assert_eq!(decoded.proof_data, proof.proof_data);
    }
    
    #[test]
    fn test_json_roundtrip() {
        let serializer = ProofSerializer::json();
        let proof = sample_proof();
        
        let bytes = serializer.serialize(&proof).unwrap();
        let decoded = serializer.deserialize(&bytes).unwrap();
        
        assert_eq!(decoded.metric_id, proof.metric_id);
        assert_eq!(decoded.score_numerator, proof.score_numerator);
    }
    
    #[test]
    fn test_header_roundtrip() {
        let header = ProofHeader::new(ProofFormat::CompactBinary, 1000, [42u8; 32]);
        let bytes = header.to_bytes();
        let decoded = ProofHeader::from_bytes(&bytes).unwrap();
        
        assert_eq!(decoded.magic, ProofHeader::MAGIC);
        assert_eq!(decoded.payload_size, 1000);
        assert_eq!(decoded.format, ProofFormat::CompactBinary);
    }
    
    #[test]
    fn test_header_validation_bad_magic() {
        let mut header = ProofHeader::new(ProofFormat::CompactBinary, 0, [0u8; 32]);
        header.magic = [0, 0, 0, 0];
        assert!(header.validate().is_err());
    }
    
    #[test]
    fn test_format_version_compatibility() {
        let v1 = FormatVersion { major: 1, minor: 0 };
        let v1_1 = FormatVersion { major: 1, minor: 1 };
        let v2 = FormatVersion { major: 2, minor: 0 };
        
        assert!(v1.is_compatible(&v1_1));
        assert!(!v1.is_compatible(&v2));
    }
    
    #[test]
    fn test_proof_size_estimation() {
        let size = estimate_proof_size(1000, 500, 128);
        assert!(size > 0);
        assert!(size < 1_000_000); // Should be reasonable
    }
    
    #[test]
    fn test_compact_proof_estimated_size() {
        let proof = sample_proof();
        let est = proof.estimated_size();
        assert!(est > 100);
    }
    
    #[test]
    fn test_rle_compression() {
        let data = vec![0, 0, 0, 0, 0, 1, 2, 3, 3, 3, 3, 3];
        let compressed = compress_rle(&data);
        let decompressed = decompress_rle(&compressed);
        assert_eq!(decompressed, data);
    }
    
    #[test]
    fn test_rle_empty() {
        let compressed = compress_rle(&[]);
        assert!(compressed.is_empty());
        let decompressed = decompress_rle(&[]);
        assert!(decompressed.is_empty());
    }
    
    #[test]
    fn test_rle_no_repetition() {
        let data = vec![1, 2, 3, 4, 5];
        let compressed = compress_rle(&data);
        let decompressed = decompress_rle(&compressed);
        assert_eq!(decompressed, data);
    }
    
    #[test]
    fn test_rle_all_same() {
        let data = vec![42u8; 100];
        let compressed = compress_rle(&data);
        assert!(compressed.len() < data.len());
        let decompressed = decompress_rle(&compressed);
        assert_eq!(decompressed, data);
    }
}
