//! SSH protocol module for NegSynth — RFC 4253/4254 compliant parsing,
//! negotiation modelling, and vulnerability detection.
//!
//! This crate provides:
//! - Binary SSH packet parsing and construction (`packet`)
//! - Key-exchange-init (KEX_INIT) parsing and building (`kex`)
//! - Full algorithm taxonomy with security classification (`algorithms`)
//! - SSH extension handling including strict-KEX / Terrapin (`extensions`)
//! - Transport-layer finite state machine (`state_machine`)
//! - Known vulnerability patterns and detection heuristics (`vulnerabilities`)
//! - nom-based message parsers (`parser`)
//! - Algorithm negotiation engine (`negotiation`)

pub mod algorithms;
pub mod extensions;
pub mod kex;
pub mod negotiation;
pub mod packet;
pub mod parser;
pub mod state_machine;
pub mod vulnerabilities;

// Re-exports for convenience
pub use algorithms::{
    CompressionAlgorithm, EncryptionAlgorithm, HostKeyAlgorithm, MacAlgorithm as SshMacAlgorithm,
    SecurityClassification,
};
pub use extensions::{ExtInfo, SshExtension, StrictKex};
pub use kex::{KexAlgorithm, KexInit, KexInitBuilder, KexInitParser};
pub use negotiation::SshNegotiationEngine;
pub use packet::{PacketBuilder, PacketParser, SshPacket};
pub use parser::SshParser;
pub use state_machine::{SshHandshakeState, SshStateMachine};
pub use vulnerabilities::SshVulnerability;

/// SSH protocol constants from RFC 4253.
pub mod constants {
    /// Maximum packet length per RFC 4253 §6.1.
    pub const MAX_PACKET_LENGTH: u32 = 35000;

    /// Minimum padding length.
    pub const MIN_PADDING: u8 = 4;

    /// Maximum padding length.
    pub const MAX_PADDING: u8 = 255;

    /// Block size for padding alignment (no cipher).
    pub const DEFAULT_BLOCK_SIZE: usize = 8;

    // -----------------------------------------------------------------------
    // SSH message type codes (RFC 4253 §12)
    // -----------------------------------------------------------------------
    pub const SSH_MSG_DISCONNECT: u8 = 1;
    pub const SSH_MSG_IGNORE: u8 = 2;
    pub const SSH_MSG_UNIMPLEMENTED: u8 = 3;
    pub const SSH_MSG_DEBUG: u8 = 4;
    pub const SSH_MSG_SERVICE_REQUEST: u8 = 5;
    pub const SSH_MSG_SERVICE_ACCEPT: u8 = 6;
    pub const SSH_MSG_EXT_INFO: u8 = 7;
    pub const SSH_MSG_KEXINIT: u8 = 20;
    pub const SSH_MSG_NEWKEYS: u8 = 21;

    // KEX-specific (RFC 4253 §8)
    pub const SSH_MSG_KEXDH_INIT: u8 = 30;
    pub const SSH_MSG_KEXDH_REPLY: u8 = 31;

    // User-auth (RFC 4252)
    pub const SSH_MSG_USERAUTH_REQUEST: u8 = 50;
    pub const SSH_MSG_USERAUTH_FAILURE: u8 = 51;
    pub const SSH_MSG_USERAUTH_SUCCESS: u8 = 52;
    pub const SSH_MSG_USERAUTH_BANNER: u8 = 53;

    // Connection (RFC 4254)
    pub const SSH_MSG_GLOBAL_REQUEST: u8 = 80;
    pub const SSH_MSG_REQUEST_SUCCESS: u8 = 81;
    pub const SSH_MSG_REQUEST_FAILURE: u8 = 82;
    pub const SSH_MSG_CHANNEL_OPEN: u8 = 90;
    pub const SSH_MSG_CHANNEL_OPEN_CONFIRMATION: u8 = 91;
    pub const SSH_MSG_CHANNEL_OPEN_FAILURE: u8 = 92;
    pub const SSH_MSG_CHANNEL_WINDOW_ADJUST: u8 = 93;
    pub const SSH_MSG_CHANNEL_DATA: u8 = 94;
    pub const SSH_MSG_CHANNEL_EOF: u8 = 96;
    pub const SSH_MSG_CHANNEL_CLOSE: u8 = 97;
    pub const SSH_MSG_CHANNEL_REQUEST: u8 = 98;
    pub const SSH_MSG_CHANNEL_SUCCESS: u8 = 99;
    pub const SSH_MSG_CHANNEL_FAILURE: u8 = 100;

    // Disconnect reason codes
    pub const SSH_DISCONNECT_HOST_NOT_ALLOWED: u32 = 1;
    pub const SSH_DISCONNECT_PROTOCOL_ERROR: u32 = 2;
    pub const SSH_DISCONNECT_KEY_EXCHANGE_FAILED: u32 = 3;
    pub const SSH_DISCONNECT_MAC_ERROR: u32 = 5;
    pub const SSH_DISCONNECT_COMPRESSION_ERROR: u32 = 6;
    pub const SSH_DISCONNECT_SERVICE_NOT_AVAILABLE: u32 = 7;
    pub const SSH_DISCONNECT_PROTOCOL_VERSION_NOT_SUPPORTED: u32 = 8;
    pub const SSH_DISCONNECT_HOST_KEY_NOT_VERIFIABLE: u32 = 9;
    pub const SSH_DISCONNECT_CONNECTION_LOST: u32 = 10;
    pub const SSH_DISCONNECT_BY_APPLICATION: u32 = 11;
    pub const SSH_DISCONNECT_TOO_MANY_CONNECTIONS: u32 = 12;
    pub const SSH_DISCONNECT_AUTH_CANCELLED_BY_USER: u32 = 13;
    pub const SSH_DISCONNECT_NO_MORE_AUTH_METHODS: u32 = 14;
    pub const SSH_DISCONNECT_ILLEGAL_USER_NAME: u32 = 15;
}

/// Errors specific to the SSH protocol crate.
#[derive(Debug, thiserror::Error)]
pub enum SshError {
    #[error("packet too large: {length} bytes (max {max})")]
    PacketTooLarge { length: u32, max: u32 },

    #[error("packet too small: {length} bytes")]
    PacketTooSmall { length: u32 },

    #[error("invalid padding length: {0}")]
    InvalidPadding(u8),

    #[error("MAC verification failed")]
    MacVerificationFailed,

    #[error("invalid version string: {0}")]
    InvalidVersionString(String),

    #[error("unsupported protocol version: {0}")]
    UnsupportedVersion(String),

    #[error("KEX negotiation failed: no matching {category} algorithm")]
    KexNegotiationFailed { category: String },

    #[error("unexpected message type {msg_type} in state {state}")]
    UnexpectedMessage { msg_type: u8, state: String },

    #[error("invalid state transition from {from} to {to}")]
    InvalidTransition { from: String, to: String },

    #[error("sequence number overflow")]
    SequenceNumberOverflow,

    #[error("strict KEX violation: {0}")]
    StrictKexViolation(String),

    #[error("parse error: {0}")]
    ParseError(String),

    #[error("incomplete data: need {needed} bytes, have {available}")]
    IncompleteData { needed: usize, available: usize },

    #[error("algorithm not supported: {0}")]
    UnsupportedAlgorithm(String),

    #[error("extension error: {0}")]
    ExtensionError(String),

    #[error("internal error: {0}")]
    Internal(String),
}

pub type SshResult<T> = std::result::Result<T, SshError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants_msg_types() {
        assert_eq!(constants::SSH_MSG_KEXINIT, 20);
        assert_eq!(constants::SSH_MSG_NEWKEYS, 21);
        assert_eq!(constants::SSH_MSG_SERVICE_REQUEST, 5);
        assert_eq!(constants::SSH_MSG_DISCONNECT, 1);
    }

    #[test]
    fn test_error_display() {
        let e = SshError::PacketTooLarge {
            length: 40000,
            max: 35000,
        };
        assert!(e.to_string().contains("40000"));
        assert!(e.to_string().contains("35000"));
    }

    #[test]
    fn test_error_kex_negotiation() {
        let e = SshError::KexNegotiationFailed {
            category: "encryption".into(),
        };
        assert!(e.to_string().contains("encryption"));
    }
}
