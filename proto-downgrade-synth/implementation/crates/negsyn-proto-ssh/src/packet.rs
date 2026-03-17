//! SSH binary packet layer — RFC 4253 §6.
//!
//! Wire format:
//! ```text
//!   uint32    packet_length    (not including self or mac)
//!   byte      padding_length
//!   byte[n1]  payload          (n1 = packet_length - padding_length - 1)
//!   byte[n2]  random padding   (n2 = padding_length)
//!   byte[m]   mac              (m depends on negotiated MAC)
//! ```
//!
//! The entire `packet_length + padding_length + payload + padding` block must
//! be a multiple of the cipher block size (min 8).

use crate::constants::{DEFAULT_BLOCK_SIZE, MAX_PACKET_LENGTH, MAX_PADDING, MIN_PADDING};
use crate::{SshError, SshResult};
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use bytes::{Buf, Bytes, BytesMut};
use serde::{Deserialize, Serialize};
use std::io::Cursor;

// ---------------------------------------------------------------------------
// SshPacket
// ---------------------------------------------------------------------------

/// A decoded SSH binary packet.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SshPacket {
    /// Total length of the packet body (padding_length + payload + padding).
    pub packet_length: u32,
    /// Length of the random padding.
    pub padding_length: u8,
    /// The useful payload bytes.
    pub payload: Vec<u8>,
    /// The random padding bytes.
    pub padding: Vec<u8>,
    /// The MAC tag (empty before MAC is negotiated).
    pub mac: Vec<u8>,
}

impl SshPacket {
    /// Returns the SSH message type byte (first byte of payload), if any.
    pub fn msg_type(&self) -> Option<u8> {
        self.payload.first().copied()
    }

    /// Returns the payload without the message type byte.
    pub fn msg_payload(&self) -> &[u8] {
        if self.payload.len() > 1 {
            &self.payload[1..]
        } else {
            &[]
        }
    }

    /// Total number of bytes on the wire (4-byte length prefix + body + mac).
    pub fn wire_length(&self) -> usize {
        4 + self.packet_length as usize + self.mac.len()
    }

    /// Validate invariants.
    pub fn validate(&self) -> SshResult<()> {
        if self.packet_length > MAX_PACKET_LENGTH {
            return Err(SshError::PacketTooLarge {
                length: self.packet_length,
                max: MAX_PACKET_LENGTH,
            });
        }
        if self.padding_length < MIN_PADDING {
            return Err(SshError::InvalidPadding(self.padding_length));
        }
        let expected_body =
            1u32 + self.payload.len() as u32 + self.padding_length as u32;
        if self.packet_length != expected_body {
            return Err(SshError::PacketTooSmall {
                length: self.packet_length,
            });
        }
        Ok(())
    }

    /// Serialize to wire bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.wire_length());
        buf.write_u32::<BigEndian>(self.packet_length).unwrap();
        buf.push(self.padding_length);
        buf.extend_from_slice(&self.payload);
        buf.extend_from_slice(&self.padding);
        buf.extend_from_slice(&self.mac);
        buf
    }

    /// Serialize to `Bytes`.
    pub fn to_bytes_buf(&self) -> Bytes {
        Bytes::from(self.to_bytes())
    }
}

// ---------------------------------------------------------------------------
// PacketParser
// ---------------------------------------------------------------------------

/// Parses SSH binary packets from a byte stream.
#[derive(Debug, Clone)]
pub struct PacketParser {
    /// Cipher block size — the packet body must be aligned to this.
    block_size: usize,
    /// MAC length in bytes (0 when no MAC negotiated).
    mac_length: usize,
    /// Maximum acceptable packet length (default 35 000).
    max_packet_length: u32,
}

impl PacketParser {
    pub fn new() -> Self {
        Self {
            block_size: DEFAULT_BLOCK_SIZE,
            mac_length: 0,
            max_packet_length: MAX_PACKET_LENGTH,
        }
    }

    pub fn with_block_size(mut self, bs: usize) -> Self {
        assert!(bs >= 8, "block_size must be >= 8");
        self.block_size = bs;
        self
    }

    pub fn with_mac_length(mut self, ml: usize) -> Self {
        self.mac_length = ml;
        self
    }

    pub fn with_max_packet_length(mut self, max: u32) -> Self {
        self.max_packet_length = max;
        self
    }

    /// Attempt to parse a single packet from `data`.
    ///
    /// Returns `(packet, bytes_consumed)` on success, or an error if the
    /// packet is malformed.  Returns `IncompleteData` if more bytes are needed.
    pub fn parse(&self, data: &[u8]) -> SshResult<(SshPacket, usize)> {
        if data.len() < 4 {
            return Err(SshError::IncompleteData {
                needed: 4,
                available: data.len(),
            });
        }

        let mut cursor = Cursor::new(data);
        let packet_length = cursor.read_u32::<BigEndian>().map_err(|_| {
            SshError::IncompleteData {
                needed: 4,
                available: data.len(),
            }
        })?;

        // Validate length
        if packet_length < 12 {
            return Err(SshError::PacketTooSmall {
                length: packet_length,
            });
        }
        if packet_length > self.max_packet_length {
            return Err(SshError::PacketTooLarge {
                length: packet_length,
                max: self.max_packet_length,
            });
        }

        let total_needed = 4 + packet_length as usize + self.mac_length;
        if data.len() < total_needed {
            return Err(SshError::IncompleteData {
                needed: total_needed,
                available: data.len(),
            });
        }

        let padding_length = cursor.read_u8().map_err(|_| SshError::ParseError(
            "failed to read padding_length".into(),
        ))?;

        if padding_length < MIN_PADDING || padding_length > MAX_PADDING {
            return Err(SshError::InvalidPadding(padding_length));
        }

        let payload_length =
            packet_length as usize - padding_length as usize - 1;

        let pos = cursor.position() as usize;
        let payload = data[pos..pos + payload_length].to_vec();
        let padding =
            data[pos + payload_length..pos + payload_length + padding_length as usize].to_vec();

        let mac_start = 4 + packet_length as usize;
        let mac = data[mac_start..mac_start + self.mac_length].to_vec();

        // Check alignment
        let body_len = packet_length as usize + 4;
        if body_len % self.block_size != 0 {
            // Alignment issue — this is technically a protocol violation,
            // but we still return the parsed packet with a note.
            log::warn!(
                "packet body length {} not aligned to block size {}",
                body_len,
                self.block_size,
            );
        }

        let pkt = SshPacket {
            packet_length,
            padding_length,
            payload,
            padding,
            mac,
        };

        Ok((pkt, total_needed))
    }

    /// Parse all complete packets from a buffer, returning remaining bytes.
    pub fn parse_all(&self, data: &[u8]) -> SshResult<(Vec<SshPacket>, usize)> {
        let mut packets = Vec::new();
        let mut offset = 0;

        loop {
            match self.parse(&data[offset..]) {
                Ok((pkt, consumed)) => {
                    packets.push(pkt);
                    offset += consumed;
                }
                Err(SshError::IncompleteData { .. }) => break,
                Err(e) => return Err(e),
            }
        }

        Ok((packets, offset))
    }

    /// Parse from a `BytesMut`, advancing the buffer.
    pub fn parse_buf(&self, buf: &mut BytesMut) -> SshResult<Option<SshPacket>> {
        match self.parse(buf.as_ref()) {
            Ok((pkt, consumed)) => {
                buf.advance(consumed);
                Ok(Some(pkt))
            }
            Err(SshError::IncompleteData { .. }) => Ok(None),
            Err(e) => Err(e),
        }
    }
}

impl Default for PacketParser {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PacketBuilder
// ---------------------------------------------------------------------------

/// Constructs valid SSH binary packets with correct padding.
#[derive(Debug, Clone)]
pub struct PacketBuilder {
    /// Cipher block size for padding alignment.
    block_size: usize,
    /// MAC length to reserve.
    mac_length: usize,
}

impl PacketBuilder {
    pub fn new() -> Self {
        Self {
            block_size: DEFAULT_BLOCK_SIZE,
            mac_length: 0,
        }
    }

    pub fn with_block_size(mut self, bs: usize) -> Self {
        assert!(bs >= 8);
        self.block_size = bs;
        self
    }

    pub fn with_mac_length(mut self, ml: usize) -> Self {
        self.mac_length = ml;
        self
    }

    /// Compute the padding length required for the given payload.
    ///
    /// The total of (4 + 1 + payload_len + padding_len) must be a multiple
    /// of `block_size`, and `padding_len` must be in `[4, 255]`.
    pub fn compute_padding_length(&self, payload_len: usize) -> u8 {
        // unpadded = 4 (packet_length) + 1 (padding_length byte) + payload_len
        let unpadded = 4 + 1 + payload_len;
        let remainder = unpadded % self.block_size;
        let mut pad = if remainder == 0 {
            0
        } else {
            self.block_size - remainder
        };
        if pad < MIN_PADDING as usize {
            pad += self.block_size;
        }
        // safety: pad should fit in u8 for reasonable block sizes
        pad as u8
    }

    /// Build a packet from the given payload bytes.
    ///
    /// Padding is filled with zeros (caller should overwrite with random
    /// bytes for production use).
    pub fn build(&self, payload: &[u8]) -> SshResult<SshPacket> {
        let padding_length = self.compute_padding_length(payload.len());
        let packet_length = 1u32 + payload.len() as u32 + padding_length as u32;

        if packet_length > MAX_PACKET_LENGTH {
            return Err(SshError::PacketTooLarge {
                length: packet_length,
                max: MAX_PACKET_LENGTH,
            });
        }

        let padding = vec![0u8; padding_length as usize];
        let mac = vec![0u8; self.mac_length];

        Ok(SshPacket {
            packet_length,
            padding_length,
            payload: payload.to_vec(),
            padding,
            mac,
        })
    }

    /// Build a packet with explicit random padding.
    pub fn build_with_padding(
        &self,
        payload: &[u8],
        random_padding: &[u8],
    ) -> SshResult<SshPacket> {
        let padding_length = random_padding.len() as u8;
        if padding_length < MIN_PADDING {
            return Err(SshError::InvalidPadding(padding_length));
        }

        let packet_length = 1u32 + payload.len() as u32 + padding_length as u32;
        if packet_length > MAX_PACKET_LENGTH {
            return Err(SshError::PacketTooLarge {
                length: packet_length,
                max: MAX_PACKET_LENGTH,
            });
        }

        // Verify alignment
        let total = 4 + packet_length as usize;
        if total % self.block_size != 0 {
            return Err(SshError::InvalidPadding(padding_length));
        }

        Ok(SshPacket {
            packet_length,
            padding_length,
            payload: payload.to_vec(),
            padding: random_padding.to_vec(),
            mac: vec![0u8; self.mac_length],
        })
    }

    /// Build a packet and immediately serialize to bytes.
    pub fn build_bytes(&self, payload: &[u8]) -> SshResult<Vec<u8>> {
        self.build(payload).map(|p| p.to_bytes())
    }

    /// Compute the MAC placeholder for a packet.
    ///
    /// In real use this would be HMAC(key, sequence_number || unencrypted_packet).
    /// We return zeros as a placeholder; the caller's crypto layer fills in
    /// the real tag.
    pub fn compute_mac_placeholder(
        &self,
        _sequence_number: u32,
        _packet_bytes: &[u8],
    ) -> Vec<u8> {
        vec![0u8; self.mac_length]
    }
}

impl Default for PacketBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Sequence number tracker
// ---------------------------------------------------------------------------

/// Tracks send/receive sequence numbers per RFC 4253 §6.4.
///
/// Sequence numbers wrap at 2^32 — critical for Terrapin attack modelling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceNumbers {
    pub send_seq: u32,
    pub recv_seq: u32,
    pub send_count: u64,
    pub recv_count: u64,
}

impl SequenceNumbers {
    pub fn new() -> Self {
        Self {
            send_seq: 0,
            recv_seq: 0,
            send_count: 0,
            recv_count: 0,
        }
    }

    /// Increment the send sequence number (wraps at 2^32).
    pub fn increment_send(&mut self) -> u32 {
        let current = self.send_seq;
        self.send_seq = self.send_seq.wrapping_add(1);
        self.send_count += 1;
        current
    }

    /// Increment the receive sequence number (wraps at 2^32).
    pub fn increment_recv(&mut self) -> u32 {
        let current = self.recv_seq;
        self.recv_seq = self.recv_seq.wrapping_add(1);
        self.recv_count += 1;
        current
    }

    /// Reset both sequence numbers to zero.
    /// Used after NEWKEYS in strict-KEX mode.
    pub fn reset(&mut self) {
        self.send_seq = 0;
        self.recv_seq = 0;
    }

    /// Reset only the send counter.
    pub fn reset_send(&mut self) {
        self.send_seq = 0;
    }

    /// Reset only the receive counter.
    pub fn reset_recv(&mut self) {
        self.recv_seq = 0;
    }

    /// Check if a sequence number wrap has occurred (for vulnerability detection).
    pub fn has_send_wrapped(&self) -> bool {
        self.send_count > u32::MAX as u64
    }

    pub fn has_recv_wrapped(&self) -> bool {
        self.recv_count > u32::MAX as u64
    }
}

impl Default for SequenceNumbers {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Version string
// ---------------------------------------------------------------------------

/// Parsed SSH version string: `SSH-protoversion-softwareversion SP comments CR LF`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SshVersionString {
    /// The raw version line (without trailing CR LF).
    pub raw: String,
    /// Protocol version (should be "2.0").
    pub proto_version: String,
    /// Software version identifier.
    pub software_version: String,
    /// Optional comments.
    pub comments: Option<String>,
}

impl SshVersionString {
    /// Parse an SSH version line.
    pub fn parse(line: &str) -> SshResult<Self> {
        let trimmed = line.trim_end_matches(|c| c == '\r' || c == '\n');
        if !trimmed.starts_with("SSH-") {
            return Err(SshError::InvalidVersionString(line.to_string()));
        }

        let rest = &trimmed[4..];
        let mut parts = rest.splitn(2, '-');
        let proto_version = parts
            .next()
            .ok_or_else(|| SshError::InvalidVersionString(line.to_string()))?
            .to_string();
        let remainder = parts
            .next()
            .ok_or_else(|| SshError::InvalidVersionString(line.to_string()))?;

        let (software_version, comments) = if let Some(idx) = remainder.find(' ') {
            (
                remainder[..idx].to_string(),
                Some(remainder[idx + 1..].to_string()),
            )
        } else {
            (remainder.to_string(), None)
        };

        Ok(Self {
            raw: trimmed.to_string(),
            proto_version,
            software_version,
            comments,
        })
    }

    /// Check that the protocol version is "2.0".
    pub fn is_v2(&self) -> bool {
        self.proto_version == "2.0"
    }

    /// Serialize back to wire format (with CR LF).
    pub fn to_wire(&self) -> String {
        let base = if let Some(ref c) = self.comments {
            format!("SSH-{}-{} {}", self.proto_version, self.software_version, c)
        } else {
            format!("SSH-{}-{}", self.proto_version, self.software_version)
        };
        format!("{}\r\n", base)
    }
}

impl fmt::Display for SshVersionString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.raw)
    }
}

use std::fmt;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_builder_padding_alignment() {
        let builder = PacketBuilder::new(); // block_size=8
        // payload=0 ⟹ unpadded = 4+1+0 = 5 ⟹ need 3 more, but min pad=4 → +8 → pad=11? Let's compute:
        // 5 % 8 = 5, block_size - 5 = 3, 3 < 4 → 3+8 = 11
        let pad = builder.compute_padding_length(0);
        assert!(pad >= MIN_PADDING);
        assert_eq!((4 + 1 + 0 + pad as usize) % 8, 0);
    }

    #[test]
    fn test_packet_builder_various_sizes() {
        let builder = PacketBuilder::new();
        for payload_len in 0..128 {
            let pad = builder.compute_padding_length(payload_len);
            assert!(pad >= MIN_PADDING);
            assert!(pad <= MAX_PADDING);
            let total = 4 + 1 + payload_len + pad as usize;
            assert_eq!(total % 8, 0, "misaligned for payload_len={}", payload_len);
        }
    }

    #[test]
    fn test_build_and_parse_roundtrip() {
        let builder = PacketBuilder::new();
        let parser = PacketParser::new();

        let payload = b"\x14hello world from SSH"; // 0x14 = KEXINIT
        let pkt = builder.build(payload).unwrap();
        pkt.validate().unwrap();

        let wire = pkt.to_bytes();
        let (parsed, consumed) = parser.parse(&wire).unwrap();
        assert_eq!(consumed, wire.len());
        assert_eq!(parsed.payload, payload.to_vec());
        assert_eq!(parsed.msg_type(), Some(0x14));
    }

    #[test]
    fn test_packet_too_large() {
        let builder = PacketBuilder::new();
        let big_payload = vec![0u8; 40000];
        assert!(builder.build(&big_payload).is_err());
    }

    #[test]
    fn test_packet_parser_incomplete() {
        let parser = PacketParser::new();
        let result = parser.parse(&[0, 0, 0]);
        assert!(matches!(result, Err(SshError::IncompleteData { .. })));
    }

    #[test]
    fn test_packet_parser_too_small() {
        let parser = PacketParser::new();
        // packet_length = 1 (way too small)
        let data = [0, 0, 0, 1, 4];
        let result = parser.parse(&data);
        assert!(matches!(result, Err(SshError::PacketTooSmall { .. })));
    }

    #[test]
    fn test_parse_buf() {
        let builder = PacketBuilder::new();
        let pkt = builder.build(b"\x15done").unwrap();
        let mut buf = BytesMut::from(pkt.to_bytes().as_slice());
        let parser = PacketParser::new();

        let parsed = parser.parse_buf(&mut buf).unwrap().unwrap();
        assert_eq!(parsed.payload, b"\x15done");
        assert!(buf.is_empty());
    }

    #[test]
    fn test_parse_all_multiple() {
        let builder = PacketBuilder::new();
        let parser = PacketParser::new();

        let pkt1 = builder.build(b"\x01first").unwrap();
        let pkt2 = builder.build(b"\x02second").unwrap();
        let mut wire = pkt1.to_bytes();
        wire.extend_from_slice(&pkt2.to_bytes());

        let (packets, consumed) = parser.parse_all(&wire).unwrap();
        assert_eq!(packets.len(), 2);
        assert_eq!(consumed, wire.len());
        assert_eq!(packets[0].payload, b"\x01first");
        assert_eq!(packets[1].payload, b"\x02second");
    }

    #[test]
    fn test_version_string_parse() {
        let vs =
            SshVersionString::parse("SSH-2.0-OpenSSH_8.9p1 Ubuntu-3ubuntu0.6\r\n").unwrap();
        assert!(vs.is_v2());
        assert_eq!(vs.proto_version, "2.0");
        assert_eq!(vs.software_version, "OpenSSH_8.9p1");
        assert_eq!(vs.comments.as_deref(), Some("Ubuntu-3ubuntu0.6"));
    }

    #[test]
    fn test_version_string_minimal() {
        let vs = SshVersionString::parse("SSH-2.0-NegSynth_0.1").unwrap();
        assert!(vs.is_v2());
        assert_eq!(vs.software_version, "NegSynth_0.1");
        assert!(vs.comments.is_none());
    }

    #[test]
    fn test_version_string_invalid() {
        assert!(SshVersionString::parse("HTTP/1.1 200 OK").is_err());
    }

    #[test]
    fn test_version_string_wire_roundtrip() {
        let vs = SshVersionString::parse("SSH-2.0-Test_1.0 comment").unwrap();
        let wire = vs.to_wire();
        assert_eq!(wire, "SSH-2.0-Test_1.0 comment\r\n");
        let vs2 = SshVersionString::parse(&wire).unwrap();
        assert_eq!(vs, vs2);
    }

    #[test]
    fn test_sequence_numbers() {
        let mut seq = SequenceNumbers::new();
        assert_eq!(seq.increment_send(), 0);
        assert_eq!(seq.increment_send(), 1);
        assert_eq!(seq.send_seq, 2);
        assert_eq!(seq.send_count, 2);

        seq.reset();
        assert_eq!(seq.send_seq, 0);
        assert_eq!(seq.recv_seq, 0);
    }

    #[test]
    fn test_sequence_number_wrap() {
        let mut seq = SequenceNumbers::new();
        seq.send_seq = u32::MAX;
        seq.send_count = u32::MAX as u64;
        let val = seq.increment_send();
        assert_eq!(val, u32::MAX);
        assert_eq!(seq.send_seq, 0); // wrapped
        assert!(seq.has_send_wrapped());
    }

    #[test]
    fn test_packet_with_mac() {
        let builder = PacketBuilder::new().with_mac_length(32);
        let parser = PacketParser::new().with_mac_length(32);

        let pkt = builder.build(b"\x14test").unwrap();
        assert_eq!(pkt.mac.len(), 32);

        let wire = pkt.to_bytes();
        let (parsed, _) = parser.parse(&wire).unwrap();
        assert_eq!(parsed.mac.len(), 32);
        assert_eq!(parsed.payload, b"\x14test");
    }

    #[test]
    fn test_block_size_16() {
        let builder = PacketBuilder::new().with_block_size(16);
        for payload_len in 0..128 {
            let pad = builder.compute_padding_length(payload_len);
            let total = 4 + 1 + payload_len + pad as usize;
            assert_eq!(total % 16, 0);
            assert!(pad >= MIN_PADDING);
        }
    }

    #[test]
    fn test_build_with_explicit_padding() {
        let builder = PacketBuilder::new(); // block_size=8
        // payload len=3 (b"\x14hi"), unpadded = 4+1+3 = 8.
        // 8+pad must be multiple of 8. pad=8,16,24,...  Min valid (>=4) = 8.
        let padding = vec![0xABu8; 8];
        let pkt = builder.build_with_padding(b"\x14hi", &padding).unwrap();
        assert_eq!(pkt.padding_length, 8);
        assert_eq!((4 + pkt.packet_length as usize) % 8, 0);
    }

    #[test]
    fn test_msg_payload() {
        let builder = PacketBuilder::new();
        let pkt = builder.build(b"\x14\x01\x02\x03").unwrap();
        assert_eq!(pkt.msg_type(), Some(0x14));
        assert_eq!(pkt.msg_payload(), &[0x01, 0x02, 0x03]);
    }

    #[test]
    fn test_empty_payload_msg_type() {
        let builder = PacketBuilder::new();
        let pkt = builder.build(b"").unwrap();
        assert_eq!(pkt.msg_type(), None);
        assert_eq!(pkt.msg_payload(), &[] as &[u8]);
    }
}
