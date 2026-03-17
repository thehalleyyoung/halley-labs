//! Byte-level protocol encoding for TLS and SSH messages.
//!
//! Provides [`ByteEncoder`] trait and concrete encoders for TLS record layer,
//! TLS handshake messages, and SSH packets.  Used by the concretizer to
//! produce wire-format bytes from abstract protocol fields.

use crate::{ConcreteError, ConcreteResult, CipherSuite, Extension, ProtocolVersion};
use byteorder::{BigEndian, WriteBytesExt};
use sha2::{Sha256, Digest};
use std::io::Write;

// ── Content type / handshake type constants ──────────────────────────────

/// TLS record content types.
pub mod tls_content_type {
    pub const CHANGE_CIPHER_SPEC: u8 = 20;
    pub const ALERT: u8 = 21;
    pub const HANDSHAKE: u8 = 22;
    pub const APPLICATION_DATA: u8 = 23;
}

/// TLS handshake message types.
pub mod tls_handshake_type {
    pub const CLIENT_HELLO: u8 = 1;
    pub const SERVER_HELLO: u8 = 2;
    pub const CERTIFICATE: u8 = 11;
    pub const SERVER_KEY_EXCHANGE: u8 = 12;
    pub const SERVER_HELLO_DONE: u8 = 14;
    pub const CLIENT_KEY_EXCHANGE: u8 = 16;
    pub const FINISHED: u8 = 20;
}

/// SSH message types.
pub mod ssh_msg_type {
    pub const DISCONNECT: u8 = 1;
    pub const IGNORE: u8 = 2;
    pub const SERVICE_REQUEST: u8 = 5;
    pub const SERVICE_ACCEPT: u8 = 6;
    pub const KEXINIT: u8 = 20;
    pub const NEWKEYS: u8 = 21;
    pub const KEX_DH_INIT: u8 = 30;
    pub const KEX_DH_REPLY: u8 = 31;
}

// ── ByteEncoder trait ────────────────────────────────────────────────────

/// Trait for encoding protocol elements to raw bytes.
pub trait ByteEncoder {
    /// Encode the element to bytes, appending to the provided buffer.
    fn encode(&self, buf: &mut Vec<u8>) -> ConcreteResult<()>;

    /// Encode and return a new Vec.
    fn encode_to_vec(&self) -> ConcreteResult<Vec<u8>> {
        let mut buf = Vec::new();
        self.encode(&mut buf)?;
        Ok(buf)
    }

    /// Returns the expected encoded size (may be approximate).
    fn encoded_size_hint(&self) -> usize;
}

// ── TLS version helpers ──────────────────────────────────────────────────

/// Convert a ProtocolVersion to TLS wire format (major, minor).
pub fn version_to_wire(version: ProtocolVersion) -> (u8, u8) {
    match version {
        ProtocolVersion::Ssl30 => (3, 0),
        ProtocolVersion::Tls10 => (3, 1),
        ProtocolVersion::Tls11 => (3, 2),
        ProtocolVersion::Tls12 => (3, 3),
        ProtocolVersion::Tls13 => (3, 3), // TLS 1.3 uses 3,3 on the wire
        ProtocolVersion::Dtls10 => (254, 255),
        ProtocolVersion::Dtls12 => (254, 253),
        ProtocolVersion::Ssh2 => (2, 0),
        ProtocolVersion::Unknown(v) => ((v >> 8) as u8, (v & 0xff) as u8),
    }
}

/// Convert wire format (major, minor) to ProtocolVersion.
pub fn wire_to_version(major: u8, minor: u8) -> ProtocolVersion {
    match (major, minor) {
        (3, 0) => ProtocolVersion::Ssl30,
        (3, 1) => ProtocolVersion::Tls10,
        (3, 2) => ProtocolVersion::Tls11,
        (3, 3) => ProtocolVersion::Tls12,
        (254, 255) => ProtocolVersion::Dtls10,
        (254, 253) => ProtocolVersion::Dtls12,
        (2, 0) => ProtocolVersion::Ssh2,
        _ => ProtocolVersion::Unknown(((major as u16) << 8) | minor as u16),
    }
}

// ── TLS Record Layer ─────────────────────────────────────────────────────

/// Encodes TLS record layer frames.
pub struct TlsRecordEncoder {
    pub content_type: u8,
    pub version: ProtocolVersion,
    pub payload: Vec<u8>,
}

impl TlsRecordEncoder {
    pub fn new(content_type: u8, version: ProtocolVersion, payload: Vec<u8>) -> Self {
        Self {
            content_type,
            version,
            payload,
        }
    }

    pub fn handshake(version: ProtocolVersion, payload: Vec<u8>) -> Self {
        Self::new(tls_content_type::HANDSHAKE, version, payload)
    }

    pub fn alert(version: ProtocolVersion, level: u8, desc: u8) -> Self {
        Self::new(tls_content_type::ALERT, version, vec![level, desc])
    }

    pub fn change_cipher_spec(version: ProtocolVersion) -> Self {
        Self::new(tls_content_type::CHANGE_CIPHER_SPEC, version, vec![1])
    }

    pub fn application_data(version: ProtocolVersion, data: Vec<u8>) -> Self {
        Self::new(tls_content_type::APPLICATION_DATA, version, data)
    }

    /// Fragment into multiple records if payload exceeds max_fragment_size.
    pub fn fragment(self, max_fragment_size: usize) -> Vec<TlsRecordEncoder> {
        if self.payload.len() <= max_fragment_size {
            return vec![self];
        }
        self.payload
            .chunks(max_fragment_size)
            .map(|chunk| TlsRecordEncoder::new(self.content_type, self.version, chunk.to_vec()))
            .collect()
    }
}

impl ByteEncoder for TlsRecordEncoder {
    fn encode(&self, buf: &mut Vec<u8>) -> ConcreteResult<()> {
        if self.payload.len() > 16384 + 2048 {
            return Err(ConcreteError::Encoding(format!(
                "TLS record payload too large: {} bytes",
                self.payload.len()
            )));
        }
        let (major, minor) = version_to_wire(self.version);
        buf.push(self.content_type);
        buf.push(major);
        buf.push(minor);
        let len = self.payload.len() as u16;
        buf.write_u16::<BigEndian>(len)
            .map_err(|e| ConcreteError::Encoding(e.to_string()))?;
        buf.extend_from_slice(&self.payload);
        Ok(())
    }

    fn encoded_size_hint(&self) -> usize {
        5 + self.payload.len()
    }
}

// ── TLS Handshake Message ────────────────────────────────────────────────

/// Encodes TLS handshake messages (inside a record's payload).
pub struct TlsHandshakeEncoder {
    pub handshake_type: u8,
    pub body: Vec<u8>,
}

impl TlsHandshakeEncoder {
    pub fn new(handshake_type: u8, body: Vec<u8>) -> Self {
        Self {
            handshake_type,
            body,
        }
    }
}

impl ByteEncoder for TlsHandshakeEncoder {
    fn encode(&self, buf: &mut Vec<u8>) -> ConcreteResult<()> {
        buf.push(self.handshake_type);
        let len = self.body.len() as u32;
        // Handshake length is 3 bytes (24-bit)
        buf.push(((len >> 16) & 0xff) as u8);
        buf.push(((len >> 8) & 0xff) as u8);
        buf.push((len & 0xff) as u8);
        buf.extend_from_slice(&self.body);
        Ok(())
    }

    fn encoded_size_hint(&self) -> usize {
        4 + self.body.len()
    }
}

// ── ClientHello construction ─────────────────────────────────────────────

/// Parameters for constructing a ClientHello message.
#[derive(Debug, Clone)]
pub struct ClientHelloParams {
    pub version: ProtocolVersion,
    pub random: [u8; 32],
    pub session_id: Vec<u8>,
    pub cipher_suites: Vec<u16>,
    pub compression_methods: Vec<u8>,
    pub extensions: Vec<Extension>,
}

impl ClientHelloParams {
    pub fn new(version: ProtocolVersion, cipher_suites: Vec<u16>) -> Self {
        let mut random = [0u8; 32];
        // Fill with deterministic pseudo-random for reproducibility
        for (i, byte) in random.iter_mut().enumerate() {
            *byte = (i as u8).wrapping_mul(0x37).wrapping_add(0x42);
        }
        Self {
            version,
            random,
            session_id: Vec::new(),
            cipher_suites,
            compression_methods: vec![0x00], // null compression
            extensions: Vec::new(),
        }
    }

    pub fn with_random(mut self, random: [u8; 32]) -> Self {
        self.random = random;
        self
    }

    pub fn with_session_id(mut self, session_id: Vec<u8>) -> Self {
        self.session_id = session_id;
        self
    }

    pub fn with_extensions(mut self, extensions: Vec<Extension>) -> Self {
        self.extensions = extensions;
        self
    }
}

impl ByteEncoder for ClientHelloParams {
    fn encode(&self, buf: &mut Vec<u8>) -> ConcreteResult<()> {
        let (major, minor) = version_to_wire(self.version);
        buf.push(major);
        buf.push(minor);

        // Random (32 bytes)
        buf.extend_from_slice(&self.random);

        // Session ID
        if self.session_id.len() > 32 {
            return Err(ConcreteError::Encoding("Session ID too long".into()));
        }
        buf.push(self.session_id.len() as u8);
        buf.extend_from_slice(&self.session_id);

        // Cipher suites
        let cs_len = (self.cipher_suites.len() * 2) as u16;
        buf.write_u16::<BigEndian>(cs_len)
            .map_err(|e| ConcreteError::Encoding(e.to_string()))?;
        for &cs in &self.cipher_suites {
            buf.write_u16::<BigEndian>(cs)
                .map_err(|e| ConcreteError::Encoding(e.to_string()))?;
        }

        // Compression methods
        buf.push(self.compression_methods.len() as u8);
        buf.extend_from_slice(&self.compression_methods);

        // Extensions
        if !self.extensions.is_empty() {
            let ext_bytes = encode_extensions(&self.extensions)?;
            let ext_len = ext_bytes.len() as u16;
            buf.write_u16::<BigEndian>(ext_len)
                .map_err(|e| ConcreteError::Encoding(e.to_string()))?;
            buf.extend_from_slice(&ext_bytes);
        }

        Ok(())
    }

    fn encoded_size_hint(&self) -> usize {
        2 + 32 + 1 + self.session_id.len() + 2 + self.cipher_suites.len() * 2 + 1
            + self.compression_methods.len()
            + if self.extensions.is_empty() { 0 } else { 2 + self.extensions.len() * 8 }
    }
}

/// Build a complete ClientHello as a TLS record (record + handshake + body).
pub fn build_client_hello(params: &ClientHelloParams) -> ConcreteResult<Vec<u8>> {
    let body = params.encode_to_vec()?;
    let hs = TlsHandshakeEncoder::new(tls_handshake_type::CLIENT_HELLO, body);
    let hs_bytes = hs.encode_to_vec()?;
    // Use TLS 1.0 (3,1) for the record layer version per spec recommendation
    let record_version = if params.version == ProtocolVersion::Tls13 {
        ProtocolVersion::Tls10
    } else {
        params.version
    };
    let record = TlsRecordEncoder::handshake(record_version, hs_bytes);
    record.encode_to_vec()
}

// ── ServerHello construction ─────────────────────────────────────────────

/// Parameters for constructing a ServerHello message.
#[derive(Debug, Clone)]
pub struct ServerHelloParams {
    pub version: ProtocolVersion,
    pub random: [u8; 32],
    pub session_id: Vec<u8>,
    pub cipher_suite: u16,
    pub compression_method: u8,
    pub extensions: Vec<Extension>,
}

impl ServerHelloParams {
    pub fn new(version: ProtocolVersion, cipher_suite: u16) -> Self {
        let mut random = [0u8; 32];
        for (i, byte) in random.iter_mut().enumerate() {
            *byte = (i as u8).wrapping_mul(0x53).wrapping_add(0x17);
        }
        Self {
            version,
            random,
            session_id: Vec::new(),
            cipher_suite,
            compression_method: 0x00,
            extensions: Vec::new(),
        }
    }

    pub fn with_random(mut self, random: [u8; 32]) -> Self {
        self.random = random;
        self
    }

    pub fn with_session_id(mut self, session_id: Vec<u8>) -> Self {
        self.session_id = session_id;
        self
    }

    pub fn with_extensions(mut self, extensions: Vec<Extension>) -> Self {
        self.extensions = extensions;
        self
    }
}

impl ByteEncoder for ServerHelloParams {
    fn encode(&self, buf: &mut Vec<u8>) -> ConcreteResult<()> {
        let (major, minor) = version_to_wire(self.version);
        buf.push(major);
        buf.push(minor);

        // Random (32 bytes)
        buf.extend_from_slice(&self.random);

        // Session ID
        buf.push(self.session_id.len() as u8);
        buf.extend_from_slice(&self.session_id);

        // Cipher suite (2 bytes)
        buf.write_u16::<BigEndian>(self.cipher_suite)
            .map_err(|e| ConcreteError::Encoding(e.to_string()))?;

        // Compression method
        buf.push(self.compression_method);

        // Extensions
        if !self.extensions.is_empty() {
            let ext_bytes = encode_extensions(&self.extensions)?;
            let ext_len = ext_bytes.len() as u16;
            buf.write_u16::<BigEndian>(ext_len)
                .map_err(|e| ConcreteError::Encoding(e.to_string()))?;
            buf.extend_from_slice(&ext_bytes);
        }

        Ok(())
    }

    fn encoded_size_hint(&self) -> usize {
        2 + 32 + 1 + self.session_id.len() + 2 + 1
            + if self.extensions.is_empty() { 0 } else { 2 + self.extensions.len() * 8 }
    }
}

/// Build a complete ServerHello as a TLS record.
pub fn build_server_hello(params: &ServerHelloParams) -> ConcreteResult<Vec<u8>> {
    let body = params.encode_to_vec()?;
    let hs = TlsHandshakeEncoder::new(tls_handshake_type::SERVER_HELLO, body);
    let hs_bytes = hs.encode_to_vec()?;
    let record = TlsRecordEncoder::handshake(params.version, hs_bytes);
    record.encode_to_vec()
}

// ── Extension encoding ───────────────────────────────────────────────────

/// Encode a list of TLS extensions to bytes.
pub fn encode_extensions(extensions: &[Extension]) -> ConcreteResult<Vec<u8>> {
    let mut buf = Vec::with_capacity(extensions.len() * 16);
    for ext in extensions {
        buf.write_u16::<BigEndian>(ext.id)
            .map_err(|e| ConcreteError::Encoding(e.to_string()))?;
        let data_len = ext.data.len() as u16;
        buf.write_u16::<BigEndian>(data_len)
            .map_err(|e| ConcreteError::Encoding(e.to_string()))?;
        buf.extend_from_slice(&ext.data);
    }
    Ok(buf)
}

/// Decode extensions from bytes.
pub fn decode_extensions(data: &[u8]) -> ConcreteResult<Vec<Extension>> {
    let mut extensions = Vec::new();
    let mut pos = 0;
    while pos + 4 <= data.len() {
        let ext_type = ((data[pos] as u16) << 8) | data[pos + 1] as u16;
        let ext_len = ((data[pos + 2] as u16) << 8) | data[pos + 3] as u16;
        pos += 4;
        if pos + ext_len as usize > data.len() {
            return Err(ConcreteError::Encoding(format!(
                "Extension data truncated at offset {}",
                pos
            )));
        }
        let ext_data = data[pos..pos + ext_len as usize].to_vec();
        pos += ext_len as usize;
        extensions.push(Extension::new(ext_type, format!("ext_0x{:04x}", ext_type), ext_data));
    }
    Ok(extensions)
}

// ── SSH packet encoding ──────────────────────────────────────────────────

/// Encodes SSH binary packets.
///
/// SSH packet format:
/// ```text
///   uint32    packet_length
///   byte      padding_length
///   byte[n1]  payload
///   byte[n2]  random padding
///   byte[m]   mac
/// ```
pub struct SshPacketEncoder {
    pub message_type: u8,
    pub payload: Vec<u8>,
    pub mac_length: usize,
}

impl SshPacketEncoder {
    pub fn new(message_type: u8, payload: Vec<u8>) -> Self {
        Self {
            message_type,
            payload,
            mac_length: 0,
        }
    }

    pub fn with_mac_length(mut self, mac_length: usize) -> Self {
        self.mac_length = mac_length;
        self
    }

    /// Compute minimum padding for block alignment.
    fn compute_padding_length(&self, block_size: usize) -> u8 {
        let payload_len = 1 + self.payload.len(); // type byte + payload
        let unpadded = 4 + 1 + payload_len; // packet_length(4) + padding_length(1) + payload
        let block_size = block_size.max(8);
        let remainder = unpadded % block_size;
        let padding = if remainder == 0 { block_size } else { block_size - remainder };
        let padding = if padding < 4 { padding + block_size } else { padding };
        padding.min(255) as u8
    }
}

impl ByteEncoder for SshPacketEncoder {
    fn encode(&self, buf: &mut Vec<u8>) -> ConcreteResult<()> {
        let padding_length = self.compute_padding_length(8);
        let payload_with_type_len = 1 + self.payload.len();
        let packet_length = 1 + payload_with_type_len + padding_length as usize;

        buf.write_u32::<BigEndian>(packet_length as u32)
            .map_err(|e| ConcreteError::Encoding(e.to_string()))?;
        buf.push(padding_length);
        buf.push(self.message_type);
        buf.extend_from_slice(&self.payload);

        // Padding (zero-filled for simplicity; in real use this would be random)
        for _ in 0..padding_length {
            buf.push(0x00);
        }

        // MAC placeholder (zeros)
        for _ in 0..self.mac_length {
            buf.push(0x00);
        }

        Ok(())
    }

    fn encoded_size_hint(&self) -> usize {
        let padding_length = self.compute_padding_length(8) as usize;
        4 + 1 + 1 + self.payload.len() + padding_length + self.mac_length
    }
}

// ── SSH KEX_INIT encoding ────────────────────────────────────────────────

/// SSH KEX_INIT message builder.
#[derive(Debug, Clone)]
pub struct SshKexInitParams {
    pub cookie: [u8; 16],
    pub kex_algorithms: Vec<String>,
    pub server_host_key_algorithms: Vec<String>,
    pub encryption_client_to_server: Vec<String>,
    pub encryption_server_to_client: Vec<String>,
    pub mac_client_to_server: Vec<String>,
    pub mac_server_to_client: Vec<String>,
    pub compression_client_to_server: Vec<String>,
    pub compression_server_to_client: Vec<String>,
    pub languages_client_to_server: Vec<String>,
    pub languages_server_to_client: Vec<String>,
    pub first_kex_packet_follows: bool,
}

impl SshKexInitParams {
    pub fn new() -> Self {
        let mut cookie = [0u8; 16];
        for (i, byte) in cookie.iter_mut().enumerate() {
            *byte = (i as u8).wrapping_mul(0x7b).wrapping_add(0x3d);
        }
        Self {
            cookie,
            kex_algorithms: vec!["diffie-hellman-group14-sha256".into()],
            server_host_key_algorithms: vec!["ssh-rsa".into()],
            encryption_client_to_server: vec!["aes128-ctr".into()],
            encryption_server_to_client: vec!["aes128-ctr".into()],
            mac_client_to_server: vec!["hmac-sha2-256".into()],
            mac_server_to_client: vec!["hmac-sha2-256".into()],
            compression_client_to_server: vec!["none".into()],
            compression_server_to_client: vec!["none".into()],
            languages_client_to_server: Vec::new(),
            languages_server_to_client: Vec::new(),
            first_kex_packet_follows: false,
        }
    }

    pub fn with_cookie(mut self, cookie: [u8; 16]) -> Self {
        self.cookie = cookie;
        self
    }

    pub fn with_kex_algorithms(mut self, algos: Vec<String>) -> Self {
        self.kex_algorithms = algos;
        self
    }

    pub fn with_encryption(mut self, algos: Vec<String>) -> Self {
        self.encryption_client_to_server = algos.clone();
        self.encryption_server_to_client = algos;
        self
    }

    fn encode_name_list(buf: &mut Vec<u8>, names: &[String]) -> ConcreteResult<()> {
        let joined = names.join(",");
        let len = joined.len() as u32;
        buf.write_u32::<BigEndian>(len)
            .map_err(|e| ConcreteError::Encoding(e.to_string()))?;
        buf.extend_from_slice(joined.as_bytes());
        Ok(())
    }
}

impl Default for SshKexInitParams {
    fn default() -> Self {
        Self::new()
    }
}

impl ByteEncoder for SshKexInitParams {
    fn encode(&self, buf: &mut Vec<u8>) -> ConcreteResult<()> {
        // Cookie (16 bytes)
        buf.extend_from_slice(&self.cookie);

        // 10 name-lists
        Self::encode_name_list(buf, &self.kex_algorithms)?;
        Self::encode_name_list(buf, &self.server_host_key_algorithms)?;
        Self::encode_name_list(buf, &self.encryption_client_to_server)?;
        Self::encode_name_list(buf, &self.encryption_server_to_client)?;
        Self::encode_name_list(buf, &self.mac_client_to_server)?;
        Self::encode_name_list(buf, &self.mac_server_to_client)?;
        Self::encode_name_list(buf, &self.compression_client_to_server)?;
        Self::encode_name_list(buf, &self.compression_server_to_client)?;
        Self::encode_name_list(buf, &self.languages_client_to_server)?;
        Self::encode_name_list(buf, &self.languages_server_to_client)?;

        // first_kex_packet_follows (boolean, 1 byte)
        buf.push(if self.first_kex_packet_follows { 1 } else { 0 });

        // reserved (uint32 = 0)
        buf.write_u32::<BigEndian>(0)
            .map_err(|e| ConcreteError::Encoding(e.to_string()))?;

        Ok(())
    }

    fn encoded_size_hint(&self) -> usize {
        let name_lists_size: usize = [
            &self.kex_algorithms,
            &self.server_host_key_algorithms,
            &self.encryption_client_to_server,
            &self.encryption_server_to_client,
            &self.mac_client_to_server,
            &self.mac_server_to_client,
            &self.compression_client_to_server,
            &self.compression_server_to_client,
            &self.languages_client_to_server,
            &self.languages_server_to_client,
        ]
        .iter()
        .map(|names| 4 + names.join(",").len())
        .sum();
        16 + name_lists_size + 1 + 4
    }
}

/// Build a complete SSH KEX_INIT packet.
pub fn build_ssh_kex_init(params: &SshKexInitParams) -> ConcreteResult<Vec<u8>> {
    let kex_body = params.encode_to_vec()?;
    let pkt = SshPacketEncoder::new(ssh_msg_type::KEXINIT, kex_body);
    pkt.encode_to_vec()
}

// ── Checksum / MAC helpers ───────────────────────────────────────────────

/// Compute SHA-256 hash of data.
pub fn sha256_hash(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

/// Compute HMAC-SHA256 (simplified — uses H(key || data) rather than full HMAC).
/// For actual protocol use, a proper HMAC implementation would be needed.
pub fn hmac_sha256_simple(key: &[u8], data: &[u8]) -> [u8; 32] {
    let mut padded_key = [0x36u8; 64];
    for (i, &b) in key.iter().enumerate().take(64) {
        padded_key[i] ^= b;
    }
    let mut inner = Vec::with_capacity(64 + data.len());
    inner.extend_from_slice(&padded_key);
    inner.extend_from_slice(data);
    let inner_hash = sha256_hash(&inner);

    let mut outer_key = [0x5cu8; 64];
    for (i, &b) in key.iter().enumerate().take(64) {
        outer_key[i] ^= b;
    }
    let mut outer = Vec::with_capacity(64 + 32);
    outer.extend_from_slice(&outer_key);
    outer.extend_from_slice(&inner_hash);
    sha256_hash(&outer)
}

/// Verify that a TLS record's structure is internally consistent.
pub fn verify_tls_record_structure(data: &[u8]) -> ConcreteResult<()> {
    if data.len() < 5 {
        return Err(ConcreteError::Encoding("TLS record too short".into()));
    }
    let content_type = data[0];
    if !matches!(content_type, 20..=23) {
        return Err(ConcreteError::Encoding(format!(
            "Invalid TLS content type: 0x{:02x}",
            content_type
        )));
    }
    let major = data[1];
    let minor = data[2];
    if major != 3 && major != 254 {
        return Err(ConcreteError::Encoding(format!(
            "Invalid TLS version major byte: {}",
            major
        )));
    }
    let record_len = ((data[3] as usize) << 8) | data[4] as usize;
    if record_len + 5 != data.len() {
        return Err(ConcreteError::Encoding(format!(
            "TLS record length mismatch: header says {} but have {} payload bytes",
            record_len,
            data.len() - 5
        )));
    }
    let _ = (major, minor); // version validated above
    Ok(())
}

/// Verify that a TLS handshake message structure is consistent.
pub fn verify_tls_handshake_structure(data: &[u8]) -> ConcreteResult<()> {
    if data.len() < 4 {
        return Err(ConcreteError::Encoding("TLS handshake too short".into()));
    }
    let hs_len = ((data[1] as usize) << 16) | ((data[2] as usize) << 8) | data[3] as usize;
    if hs_len + 4 != data.len() {
        return Err(ConcreteError::Encoding(format!(
            "TLS handshake length mismatch: header says {} but have {} body bytes",
            hs_len,
            data.len() - 4
        )));
    }
    Ok(())
}

/// Build a TLS Finished message (simplified — just hash of handshake messages).
pub fn build_tls_finished(
    version: ProtocolVersion,
    handshake_hash: &[u8; 32],
    is_client: bool,
) -> ConcreteResult<Vec<u8>> {
    let label = if is_client {
        b"client finished"
    } else {
        b"server finished"
    };

    // PRF output (simplified: hash(label || seed))
    let mut prf_input = Vec::with_capacity(label.len() + 32);
    prf_input.extend_from_slice(label);
    prf_input.extend_from_slice(handshake_hash);
    let verify_data = sha256_hash(&prf_input);

    // Finished message uses 12 bytes of verify_data
    let verify_data_slice = &verify_data[..12];
    let hs = TlsHandshakeEncoder::new(tls_handshake_type::FINISHED, verify_data_slice.to_vec());
    let hs_bytes = hs.encode_to_vec()?;
    let record = TlsRecordEncoder::handshake(version, hs_bytes);
    record.encode_to_vec()
}

/// Build a TLS ChangeCipherSpec message.
pub fn build_change_cipher_spec(version: ProtocolVersion) -> ConcreteResult<Vec<u8>> {
    let record = TlsRecordEncoder::change_cipher_spec(version);
    record.encode_to_vec()
}

/// Build a TLS Alert message.
pub fn build_tls_alert(
    version: ProtocolVersion,
    level: u8,
    description: u8,
) -> ConcreteResult<Vec<u8>> {
    let record = TlsRecordEncoder::alert(version, level, description);
    record.encode_to_vec()
}

/// Parse cipher suite ID from bytes at the given offset.
pub fn parse_cipher_suite_id(data: &[u8], offset: usize) -> ConcreteResult<u16> {
    if offset + 2 > data.len() {
        return Err(ConcreteError::Encoding("Not enough bytes for cipher suite ID".into()));
    }
    Ok(((data[offset] as u16) << 8) | data[offset + 1] as u16)
}

/// Encode a cipher suite ID to bytes.
pub fn encode_cipher_suite_id(id: u16) -> [u8; 2] {
    [(id >> 8) as u8, (id & 0xff) as u8]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_wire_roundtrip() {
        let versions = [
            ProtocolVersion::Ssl30,
            ProtocolVersion::Tls10,
            ProtocolVersion::Tls11,
            ProtocolVersion::Tls12,
            ProtocolVersion::Dtls10,
            ProtocolVersion::Dtls12,
        ];
        for v in versions {
            let (major, minor) = version_to_wire(v);
            let roundtrip = wire_to_version(major, minor);
            assert_eq!(v, roundtrip, "Roundtrip failed for {:?}", v);
        }
    }

    #[test]
    fn test_tls_record_encoding() {
        let record = TlsRecordEncoder::handshake(ProtocolVersion::Tls12, vec![0x01, 0x02, 0x03]);
        let bytes = record.encode_to_vec().unwrap();
        assert_eq!(bytes[0], tls_content_type::HANDSHAKE);
        assert_eq!(bytes[1], 3); // major
        assert_eq!(bytes[2], 3); // minor (TLS 1.2)
        assert_eq!(bytes[3], 0); // length high
        assert_eq!(bytes[4], 3); // length low
        assert_eq!(&bytes[5..], &[0x01, 0x02, 0x03]);
        assert_eq!(bytes.len(), record.encoded_size_hint());
    }

    #[test]
    fn test_tls_record_verify() {
        let record = TlsRecordEncoder::handshake(ProtocolVersion::Tls12, vec![0x01, 0x02]);
        let bytes = record.encode_to_vec().unwrap();
        assert!(verify_tls_record_structure(&bytes).is_ok());

        // Corrupt length
        let mut bad = bytes.clone();
        bad[4] = 99;
        assert!(verify_tls_record_structure(&bad).is_err());
    }

    #[test]
    fn test_handshake_encoding() {
        let hs = TlsHandshakeEncoder::new(tls_handshake_type::CLIENT_HELLO, vec![0xAA; 10]);
        let bytes = hs.encode_to_vec().unwrap();
        assert_eq!(bytes[0], tls_handshake_type::CLIENT_HELLO);
        // 3-byte length = 10
        assert_eq!(bytes[1], 0);
        assert_eq!(bytes[2], 0);
        assert_eq!(bytes[3], 10);
        assert_eq!(&bytes[4..], &[0xAA; 10]);
    }

    #[test]
    fn test_client_hello_construction() {
        let params = ClientHelloParams::new(ProtocolVersion::Tls12, vec![0xc02f, 0xc030, 0x009e]);
        let record_bytes = build_client_hello(&params).unwrap();
        assert!(verify_tls_record_structure(&record_bytes).is_ok());
        assert_eq!(record_bytes[0], tls_content_type::HANDSHAKE);
        // Inside the handshake, first byte should be CLIENT_HELLO
        assert_eq!(record_bytes[5], tls_handshake_type::CLIENT_HELLO);
    }

    #[test]
    fn test_server_hello_construction() {
        let params = ServerHelloParams::new(ProtocolVersion::Tls12, 0xc02f);
        let record_bytes = build_server_hello(&params).unwrap();
        assert!(verify_tls_record_structure(&record_bytes).is_ok());
        assert_eq!(record_bytes[5], tls_handshake_type::SERVER_HELLO);
    }

    #[test]
    fn test_extension_roundtrip() {
        let exts = vec![
            Extension::new(0x0000, "server_name", vec![0x01, 0x02]),
            Extension::new(0xff01, "renegotiation_info", vec![0x00]),
        ];
        let encoded = encode_extensions(&exts).unwrap();
        let decoded = decode_extensions(&encoded).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].id, 0x0000);
        assert_eq!(decoded[0].data, vec![0x01, 0x02]);
        assert_eq!(decoded[1].id, 0xff01);
    }

    #[test]
    fn test_ssh_packet_encoding() {
        let pkt = SshPacketEncoder::new(ssh_msg_type::KEXINIT, vec![0xAA; 20]);
        let bytes = pkt.encode_to_vec().unwrap();
        // First 4 bytes are packet_length (big-endian)
        let pkt_len = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        // 5th byte is padding_length
        let pad_len = bytes[4] as u32;
        // 6th byte is message type
        assert_eq!(bytes[5], ssh_msg_type::KEXINIT);
        // Total = 4 (len field) + pkt_len
        assert_eq!(bytes.len() as u32, 4 + pkt_len);
        // pkt_len = 1 (pad_len) + 1 (type) + 20 (payload) + padding
        assert_eq!(pkt_len, 1 + 1 + 20 + pad_len);
    }

    #[test]
    fn test_ssh_kex_init() {
        let params = SshKexInitParams::new();
        let bytes = build_ssh_kex_init(&params).unwrap();
        // Should be a valid SSH packet with KEXINIT type
        assert!(bytes.len() > 20);
        // After packet header (4 bytes len + 1 byte padding_len), message type
        assert_eq!(bytes[5], ssh_msg_type::KEXINIT);
    }

    #[test]
    fn test_sha256_hash() {
        let hash = sha256_hash(b"hello");
        assert_eq!(hash.len(), 32);
        let hex_str = hex::encode(hash);
        assert_eq!(
            hex_str,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn test_hmac_sha256_simple() {
        let mac = hmac_sha256_simple(b"key", b"data");
        assert_eq!(mac.len(), 32);
        // Just check it's deterministic
        let mac2 = hmac_sha256_simple(b"key", b"data");
        assert_eq!(mac, mac2);
        // Different inputs → different outputs
        let mac3 = hmac_sha256_simple(b"key2", b"data");
        assert_ne!(mac, mac3);
    }

    #[test]
    fn test_record_fragment() {
        let payload = vec![0xAA; 100];
        let record = TlsRecordEncoder::handshake(ProtocolVersion::Tls12, payload);
        let fragments = record.fragment(40);
        assert_eq!(fragments.len(), 3); // 40 + 40 + 20
        assert_eq!(fragments[0].payload.len(), 40);
        assert_eq!(fragments[1].payload.len(), 40);
        assert_eq!(fragments[2].payload.len(), 20);
    }

    #[test]
    fn test_cipher_suite_id_encoding() {
        let id = 0xc02f;
        let bytes = encode_cipher_suite_id(id);
        assert_eq!(bytes, [0xc0, 0x2f]);
        let parsed = parse_cipher_suite_id(&bytes, 0).unwrap();
        assert_eq!(parsed, id);
    }

    #[test]
    fn test_change_cipher_spec() {
        let bytes = build_change_cipher_spec(ProtocolVersion::Tls12).unwrap();
        assert_eq!(bytes[0], tls_content_type::CHANGE_CIPHER_SPEC);
        assert!(verify_tls_record_structure(&bytes).is_ok());
        assert_eq!(bytes[5], 1); // CCS byte
    }

    #[test]
    fn test_tls_alert() {
        let bytes = build_tls_alert(ProtocolVersion::Tls12, 2, 40).unwrap();
        assert_eq!(bytes[0], tls_content_type::ALERT);
        assert!(verify_tls_record_structure(&bytes).is_ok());
        assert_eq!(bytes[5], 2); // level = fatal
        assert_eq!(bytes[6], 40); // handshake_failure
    }

    #[test]
    fn test_finished_message() {
        let hash = sha256_hash(b"handshake messages");
        let bytes = build_tls_finished(ProtocolVersion::Tls12, &hash, true).unwrap();
        assert!(verify_tls_record_structure(&bytes).is_ok());
        assert_eq!(bytes[5], tls_handshake_type::FINISHED);
        // Verify data is 12 bytes, so handshake body = 12
        let hs_len = ((bytes[6] as usize) << 16) | ((bytes[7] as usize) << 8) | bytes[8] as usize;
        assert_eq!(hs_len, 12);
    }

    #[test]
    fn test_client_hello_with_extensions() {
        let ext = Extension::new(0x0017, "extended_master_secret", vec![]);
        let params = ClientHelloParams::new(ProtocolVersion::Tls12, vec![0xc02f])
            .with_extensions(vec![ext]);
        let bytes = build_client_hello(&params).unwrap();
        assert!(verify_tls_record_structure(&bytes).is_ok());
    }
}
