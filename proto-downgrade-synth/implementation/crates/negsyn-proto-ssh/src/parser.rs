//! nom-based SSH message parsers.
//!
//! Provides combinators for parsing SSH wire-format structures including
//! version strings, name-lists, and all major SSH message types.

use crate::constants::*;
use crate::kex::KexInit;
use crate::{SshError, SshResult};
use nom::bytes::complete::take;
use nom::combinator::{map, map_res};
use nom::number::complete::{be_u32, be_u8};
use nom::sequence::{pair, tuple};
use nom::IResult;
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Low-level combinators
// ---------------------------------------------------------------------------

/// Parse a uint32 length-prefixed byte string.
fn ssh_string(input: &[u8]) -> IResult<&[u8], &[u8]> {
    let (input, len) = be_u32(input)?;
    take(len as usize)(input)
}

/// Parse a uint32 length-prefixed UTF-8 string.
fn ssh_utf8_string(input: &[u8]) -> IResult<&[u8], &str> {
    map_res(ssh_string, std::str::from_utf8)(input)
}

/// Parse a name-list (uint32 length, then comma-separated ASCII names).
fn name_list(input: &[u8]) -> IResult<&[u8], Vec<String>> {
    let (input, raw) = ssh_utf8_string(input)?;
    if raw.is_empty() {
        Ok((input, Vec::new()))
    } else {
        Ok((input, raw.split(',').map(|s| s.to_string()).collect()))
    }
}

/// Parse a boolean byte.
fn ssh_bool(input: &[u8]) -> IResult<&[u8], bool> {
    map(be_u8, |b| b != 0)(input)
}

/// Parse exactly 16 bytes (cookie).
#[allow(dead_code)]
fn cookie(input: &[u8]) -> IResult<&[u8], [u8; 16]> {
    let (input, bytes) = take(16usize)(input)?;
    let mut arr = [0u8; 16];
    arr.copy_from_slice(bytes);
    Ok((input, arr))
}

// ---------------------------------------------------------------------------
// SSH message types
// ---------------------------------------------------------------------------

/// A parsed SSH message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SshMessage {
    /// SSH_MSG_DISCONNECT (1)
    Disconnect {
        reason_code: u32,
        description: String,
        language_tag: String,
    },
    /// SSH_MSG_IGNORE (2)
    Ignore { data: Vec<u8> },
    /// SSH_MSG_UNIMPLEMENTED (3)
    Unimplemented { sequence_number: u32 },
    /// SSH_MSG_DEBUG (4)
    Debug {
        always_display: bool,
        message: String,
        language_tag: String,
    },
    /// SSH_MSG_SERVICE_REQUEST (5)
    ServiceRequest { service_name: String },
    /// SSH_MSG_SERVICE_ACCEPT (6)
    ServiceAccept { service_name: String },
    /// SSH_MSG_KEXINIT (20)
    KexInit(KexInit),
    /// SSH_MSG_NEWKEYS (21)
    NewKeys,
    /// SSH_MSG_KEXDH_INIT (30)
    KexDhInit { e: Vec<u8> },
    /// SSH_MSG_KEXDH_REPLY (31)
    KexDhReply {
        host_key: Vec<u8>,
        f: Vec<u8>,
        signature: Vec<u8>,
    },
    /// SSH_MSG_USERAUTH_REQUEST (50)
    UserauthRequest {
        username: String,
        service_name: String,
        method_name: String,
    },
    /// SSH_MSG_USERAUTH_FAILURE (51)
    UserauthFailure {
        authentications: Vec<String>,
        partial_success: bool,
    },
    /// SSH_MSG_USERAUTH_SUCCESS (52)
    UserauthSuccess,
    /// SSH_MSG_USERAUTH_BANNER (53)
    UserauthBanner {
        message: String,
        language_tag: String,
    },
    /// SSH_MSG_CHANNEL_OPEN (90)
    ChannelOpen {
        channel_type: String,
        sender_channel: u32,
        initial_window_size: u32,
        maximum_packet_size: u32,
    },
    /// SSH_MSG_CHANNEL_OPEN_CONFIRMATION (91)
    ChannelOpenConfirmation {
        recipient_channel: u32,
        sender_channel: u32,
        initial_window_size: u32,
        maximum_packet_size: u32,
    },
    /// Unknown message type
    Unknown { msg_type: u8, data: Vec<u8> },
}

impl SshMessage {
    /// Returns the message type byte.
    pub fn msg_type(&self) -> u8 {
        match self {
            Self::Disconnect { .. } => SSH_MSG_DISCONNECT,
            Self::Ignore { .. } => SSH_MSG_IGNORE,
            Self::Unimplemented { .. } => SSH_MSG_UNIMPLEMENTED,
            Self::Debug { .. } => SSH_MSG_DEBUG,
            Self::ServiceRequest { .. } => SSH_MSG_SERVICE_REQUEST,
            Self::ServiceAccept { .. } => SSH_MSG_SERVICE_ACCEPT,
            Self::KexInit(_) => SSH_MSG_KEXINIT,
            Self::NewKeys => SSH_MSG_NEWKEYS,
            Self::KexDhInit { .. } => SSH_MSG_KEXDH_INIT,
            Self::KexDhReply { .. } => SSH_MSG_KEXDH_REPLY,
            Self::UserauthRequest { .. } => SSH_MSG_USERAUTH_REQUEST,
            Self::UserauthFailure { .. } => SSH_MSG_USERAUTH_FAILURE,
            Self::UserauthSuccess => SSH_MSG_USERAUTH_SUCCESS,
            Self::UserauthBanner { .. } => SSH_MSG_USERAUTH_BANNER,
            Self::ChannelOpen { .. } => SSH_MSG_CHANNEL_OPEN,
            Self::ChannelOpenConfirmation { .. } => SSH_MSG_CHANNEL_OPEN_CONFIRMATION,
            Self::Unknown { msg_type, .. } => *msg_type,
        }
    }

    /// Returns the human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Disconnect { .. } => "SSH_MSG_DISCONNECT",
            Self::Ignore { .. } => "SSH_MSG_IGNORE",
            Self::Unimplemented { .. } => "SSH_MSG_UNIMPLEMENTED",
            Self::Debug { .. } => "SSH_MSG_DEBUG",
            Self::ServiceRequest { .. } => "SSH_MSG_SERVICE_REQUEST",
            Self::ServiceAccept { .. } => "SSH_MSG_SERVICE_ACCEPT",
            Self::KexInit(_) => "SSH_MSG_KEXINIT",
            Self::NewKeys => "SSH_MSG_NEWKEYS",
            Self::KexDhInit { .. } => "SSH_MSG_KEXDH_INIT",
            Self::KexDhReply { .. } => "SSH_MSG_KEXDH_REPLY",
            Self::UserauthRequest { .. } => "SSH_MSG_USERAUTH_REQUEST",
            Self::UserauthFailure { .. } => "SSH_MSG_USERAUTH_FAILURE",
            Self::UserauthSuccess => "SSH_MSG_USERAUTH_SUCCESS",
            Self::UserauthBanner { .. } => "SSH_MSG_USERAUTH_BANNER",
            Self::ChannelOpen { .. } => "SSH_MSG_CHANNEL_OPEN",
            Self::ChannelOpenConfirmation { .. } => "SSH_MSG_CHANNEL_OPEN_CONFIRMATION",
            Self::Unknown { .. } => "SSH_MSG_UNKNOWN",
        }
    }

    /// Returns true if this is a transport-layer message (1-19, 20-49).
    pub fn is_transport(&self) -> bool {
        let t = self.msg_type();
        (1..=49).contains(&t)
    }

    /// Returns true if this is an authentication message (50-79).
    pub fn is_auth(&self) -> bool {
        let t = self.msg_type();
        (50..=79).contains(&t)
    }

    /// Returns true if this is a channel message (80-127).
    pub fn is_channel(&self) -> bool {
        let t = self.msg_type();
        (80..=127).contains(&t)
    }
}

impl fmt::Display for SshMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.name(), self.msg_type())
    }
}

// ---------------------------------------------------------------------------
// SshParser
// ---------------------------------------------------------------------------

/// Main parser for SSH messages.
pub struct SshParser;

impl SshParser {
    /// Parse an SSH version string from a byte slice.
    ///
    /// The version string format is: `SSH-protoversion-softwareversion[ SP comments] CR LF`
    pub fn parse_version_string(input: &[u8]) -> SshResult<(crate::packet::SshVersionString, usize)> {
        // Find CR LF
        let mut end = None;
        for i in 0..input.len().saturating_sub(1) {
            if input[i] == b'\r' && input[i + 1] == b'\n' {
                end = Some(i + 2);
                break;
            }
        }
        // Also accept just LF
        if end.is_none() {
            for i in 0..input.len() {
                if input[i] == b'\n' {
                    end = Some(i + 1);
                    break;
                }
            }
        }

        let end = end.ok_or(SshError::IncompleteData {
            needed: 0,
            available: input.len(),
        })?;

        let line = std::str::from_utf8(&input[..end])
            .map_err(|e| SshError::ParseError(format!("version string utf8: {}", e)))?;

        let vs = crate::packet::SshVersionString::parse(line)?;
        Ok((vs, end))
    }

    /// Parse an SSH message from the payload of an SSH packet.
    /// The first byte is the message type.
    pub fn parse_message(payload: &[u8]) -> SshResult<SshMessage> {
        if payload.is_empty() {
            return Err(SshError::ParseError("empty payload".into()));
        }

        let msg_type = payload[0];

        match msg_type {
            SSH_MSG_DISCONNECT => Self::parse_disconnect(payload),
            SSH_MSG_IGNORE => Self::parse_ignore(payload),
            SSH_MSG_UNIMPLEMENTED => Self::parse_unimplemented(payload),
            SSH_MSG_DEBUG => Self::parse_debug(payload),
            SSH_MSG_SERVICE_REQUEST => Self::parse_service_request(payload),
            SSH_MSG_SERVICE_ACCEPT => Self::parse_service_accept(payload),
            SSH_MSG_KEXINIT => Self::parse_kexinit(payload),
            SSH_MSG_NEWKEYS => Ok(SshMessage::NewKeys),
            SSH_MSG_KEXDH_INIT => Self::parse_kexdh_init(payload),
            SSH_MSG_KEXDH_REPLY => Self::parse_kexdh_reply(payload),
            SSH_MSG_USERAUTH_REQUEST => Self::parse_userauth_request(payload),
            SSH_MSG_USERAUTH_FAILURE => Self::parse_userauth_failure(payload),
            SSH_MSG_USERAUTH_SUCCESS => Ok(SshMessage::UserauthSuccess),
            SSH_MSG_USERAUTH_BANNER => Self::parse_userauth_banner(payload),
            SSH_MSG_CHANNEL_OPEN => Self::parse_channel_open(payload),
            SSH_MSG_CHANNEL_OPEN_CONFIRMATION => Self::parse_channel_open_confirmation(payload),
            _ => Ok(SshMessage::Unknown {
                msg_type,
                data: payload[1..].to_vec(),
            }),
        }
    }

    fn parse_disconnect(payload: &[u8]) -> SshResult<SshMessage> {
        let input = &payload[1..]; // skip msg type
        let result: IResult<&[u8], (u32, &str, &str)> =
            tuple((be_u32, ssh_utf8_string, ssh_utf8_string))(input);

        match result {
            Ok((_, (reason_code, description, language_tag))) => Ok(SshMessage::Disconnect {
                reason_code,
                description: description.to_string(),
                language_tag: language_tag.to_string(),
            }),
            Err(e) => Err(SshError::ParseError(format!("DISCONNECT: {}", e))),
        }
    }

    fn parse_ignore(payload: &[u8]) -> SshResult<SshMessage> {
        let input = &payload[1..];
        let result: IResult<&[u8], &[u8]> = ssh_string(input);

        match result {
            Ok((_, data)) => Ok(SshMessage::Ignore {
                data: data.to_vec(),
            }),
            Err(_) => Ok(SshMessage::Ignore {
                data: input.to_vec(),
            }),
        }
    }

    fn parse_unimplemented(payload: &[u8]) -> SshResult<SshMessage> {
        let input = &payload[1..];
        let result: IResult<&[u8], u32> = be_u32(input);

        match result {
            Ok((_, seq)) => Ok(SshMessage::Unimplemented {
                sequence_number: seq,
            }),
            Err(e) => Err(SshError::ParseError(format!("UNIMPLEMENTED: {}", e))),
        }
    }

    fn parse_debug(payload: &[u8]) -> SshResult<SshMessage> {
        let input = &payload[1..];
        let result: IResult<&[u8], (bool, &str, &str)> =
            tuple((ssh_bool, ssh_utf8_string, ssh_utf8_string))(input);

        match result {
            Ok((_, (always_display, message, language_tag))) => Ok(SshMessage::Debug {
                always_display,
                message: message.to_string(),
                language_tag: language_tag.to_string(),
            }),
            Err(e) => Err(SshError::ParseError(format!("DEBUG: {}", e))),
        }
    }

    fn parse_service_request(payload: &[u8]) -> SshResult<SshMessage> {
        let input = &payload[1..];
        let result: IResult<&[u8], &str> = ssh_utf8_string(input);

        match result {
            Ok((_, name)) => Ok(SshMessage::ServiceRequest {
                service_name: name.to_string(),
            }),
            Err(e) => Err(SshError::ParseError(format!("SERVICE_REQUEST: {}", e))),
        }
    }

    fn parse_service_accept(payload: &[u8]) -> SshResult<SshMessage> {
        let input = &payload[1..];
        let result: IResult<&[u8], &str> = ssh_utf8_string(input);

        match result {
            Ok((_, name)) => Ok(SshMessage::ServiceAccept {
                service_name: name.to_string(),
            }),
            Err(e) => Err(SshError::ParseError(format!("SERVICE_ACCEPT: {}", e))),
        }
    }

    fn parse_kexinit(payload: &[u8]) -> SshResult<SshMessage> {
        // Delegate to KexInitParser which expects the full payload (including msg type)
        let ki = crate::kex::KexInitParser::parse(payload)?;
        Ok(SshMessage::KexInit(ki))
    }

    fn parse_kexdh_init(payload: &[u8]) -> SshResult<SshMessage> {
        let input = &payload[1..];
        let result: IResult<&[u8], &[u8]> = ssh_string(input);

        match result {
            Ok((_, e)) => Ok(SshMessage::KexDhInit { e: e.to_vec() }),
            Err(e) => Err(SshError::ParseError(format!("KEXDH_INIT: {}", e))),
        }
    }

    fn parse_kexdh_reply(payload: &[u8]) -> SshResult<SshMessage> {
        let input = &payload[1..];
        let result: IResult<&[u8], (&[u8], &[u8], &[u8])> =
            tuple((ssh_string, ssh_string, ssh_string))(input);

        match result {
            Ok((_, (host_key, f, signature))) => Ok(SshMessage::KexDhReply {
                host_key: host_key.to_vec(),
                f: f.to_vec(),
                signature: signature.to_vec(),
            }),
            Err(e) => Err(SshError::ParseError(format!("KEXDH_REPLY: {}", e))),
        }
    }

    fn parse_userauth_request(payload: &[u8]) -> SshResult<SshMessage> {
        let input = &payload[1..];
        let result: IResult<&[u8], (&str, &str, &str)> =
            tuple((ssh_utf8_string, ssh_utf8_string, ssh_utf8_string))(input);

        match result {
            Ok((_, (username, service, method))) => Ok(SshMessage::UserauthRequest {
                username: username.to_string(),
                service_name: service.to_string(),
                method_name: method.to_string(),
            }),
            Err(e) => Err(SshError::ParseError(format!("USERAUTH_REQUEST: {}", e))),
        }
    }

    fn parse_userauth_failure(payload: &[u8]) -> SshResult<SshMessage> {
        let input = &payload[1..];
        let result: IResult<&[u8], (Vec<String>, bool)> = pair(name_list, ssh_bool)(input);

        match result {
            Ok((_, (auths, partial))) => Ok(SshMessage::UserauthFailure {
                authentications: auths,
                partial_success: partial,
            }),
            Err(e) => Err(SshError::ParseError(format!("USERAUTH_FAILURE: {}", e))),
        }
    }

    fn parse_userauth_banner(payload: &[u8]) -> SshResult<SshMessage> {
        let input = &payload[1..];
        let result: IResult<&[u8], (&str, &str)> =
            pair(ssh_utf8_string, ssh_utf8_string)(input);

        match result {
            Ok((_, (message, language_tag))) => Ok(SshMessage::UserauthBanner {
                message: message.to_string(),
                language_tag: language_tag.to_string(),
            }),
            Err(e) => Err(SshError::ParseError(format!("USERAUTH_BANNER: {}", e))),
        }
    }

    fn parse_channel_open(payload: &[u8]) -> SshResult<SshMessage> {
        let input = &payload[1..];
        let result: IResult<&[u8], (&str, u32, u32, u32)> =
            tuple((ssh_utf8_string, be_u32, be_u32, be_u32))(input);

        match result {
            Ok((_, (channel_type, sender, window, max_packet))) => {
                Ok(SshMessage::ChannelOpen {
                    channel_type: channel_type.to_string(),
                    sender_channel: sender,
                    initial_window_size: window,
                    maximum_packet_size: max_packet,
                })
            }
            Err(e) => Err(SshError::ParseError(format!("CHANNEL_OPEN: {}", e))),
        }
    }

    fn parse_channel_open_confirmation(payload: &[u8]) -> SshResult<SshMessage> {
        let input = &payload[1..];
        let result: IResult<&[u8], (u32, u32, u32, u32)> =
            tuple((be_u32, be_u32, be_u32, be_u32))(input);

        match result {
            Ok((_, (recipient, sender, window, max_packet))) => {
                Ok(SshMessage::ChannelOpenConfirmation {
                    recipient_channel: recipient,
                    sender_channel: sender,
                    initial_window_size: window,
                    maximum_packet_size: max_packet,
                })
            }
            Err(e) => Err(SshError::ParseError(format!("CHANNEL_OPEN_CONFIRM: {}", e))),
        }
    }

    /// Build raw SSH string bytes (uint32 length + data).
    pub fn encode_ssh_string(data: &[u8]) -> Vec<u8> {
        let mut buf = Vec::with_capacity(4 + data.len());
        buf.extend_from_slice(&(data.len() as u32).to_be_bytes());
        buf.extend_from_slice(data);
        buf
    }

    /// Build raw SSH utf8 string bytes.
    pub fn encode_ssh_utf8_string(s: &str) -> Vec<u8> {
        Self::encode_ssh_string(s.as_bytes())
    }

    /// Build an SSH_MSG_DISCONNECT payload.
    pub fn build_disconnect(reason_code: u32, description: &str, language: &str) -> Vec<u8> {
        let mut buf = vec![SSH_MSG_DISCONNECT];
        buf.extend_from_slice(&reason_code.to_be_bytes());
        buf.extend_from_slice(&Self::encode_ssh_utf8_string(description));
        buf.extend_from_slice(&Self::encode_ssh_utf8_string(language));
        buf
    }

    /// Build an SSH_MSG_IGNORE payload.
    pub fn build_ignore(data: &[u8]) -> Vec<u8> {
        let mut buf = vec![SSH_MSG_IGNORE];
        buf.extend_from_slice(&Self::encode_ssh_string(data));
        buf
    }

    /// Build an SSH_MSG_SERVICE_REQUEST payload.
    pub fn build_service_request(service_name: &str) -> Vec<u8> {
        let mut buf = vec![SSH_MSG_SERVICE_REQUEST];
        buf.extend_from_slice(&Self::encode_ssh_utf8_string(service_name));
        buf
    }

    /// Build an SSH_MSG_SERVICE_ACCEPT payload.
    pub fn build_service_accept(service_name: &str) -> Vec<u8> {
        let mut buf = vec![SSH_MSG_SERVICE_ACCEPT];
        buf.extend_from_slice(&Self::encode_ssh_utf8_string(service_name));
        buf
    }

    /// Build an SSH_MSG_NEWKEYS payload.
    pub fn build_newkeys() -> Vec<u8> {
        vec![SSH_MSG_NEWKEYS]
    }

    /// Attempt to recover from a malformed message by extracting
    /// the message type and returning the raw data.
    pub fn recover_message(payload: &[u8]) -> SshMessage {
        if payload.is_empty() {
            return SshMessage::Unknown {
                msg_type: 0,
                data: Vec::new(),
            };
        }

        SshMessage::Unknown {
            msg_type: payload[0],
            data: payload[1..].to_vec(),
        }
    }
}

// ---------------------------------------------------------------------------
// Parsing helpers for name-lists (standalone, not nom)
// ---------------------------------------------------------------------------

/// Parse a comma-separated name-list string into a Vec of Strings.
pub fn parse_name_list_str(input: &str) -> Vec<String> {
    if input.is_empty() {
        Vec::new()
    } else {
        input.split(',').map(|s| s.trim().to_string()).collect()
    }
}

/// Encode a list of names into a comma-separated string.
pub fn encode_name_list(names: &[String]) -> String {
    names.join(",")
}

/// Encode a name-list into SSH wire format (uint32 length + data).
pub fn encode_name_list_bytes(names: &[String]) -> Vec<u8> {
    let joined = names.join(",");
    SshParser::encode_ssh_utf8_string(&joined)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_version_string() {
        let input = b"SSH-2.0-OpenSSH_8.9p1 Ubuntu\r\n";
        let (vs, consumed) = SshParser::parse_version_string(input).unwrap();
        assert!(vs.is_v2());
        assert_eq!(vs.software_version, "OpenSSH_8.9p1");
        assert_eq!(consumed, input.len());
    }

    #[test]
    fn parse_version_string_lf_only() {
        let input = b"SSH-2.0-NegSynth\n";
        let (vs, consumed) = SshParser::parse_version_string(input).unwrap();
        assert!(vs.is_v2());
        assert_eq!(vs.software_version, "NegSynth");
        assert_eq!(consumed, input.len());
    }

    #[test]
    fn parse_disconnect() {
        let payload = SshParser::build_disconnect(2, "protocol error", "en");
        let msg = SshParser::parse_message(&payload).unwrap();

        if let SshMessage::Disconnect {
            reason_code,
            description,
            language_tag,
        } = msg
        {
            assert_eq!(reason_code, 2);
            assert_eq!(description, "protocol error");
            assert_eq!(language_tag, "en");
        } else {
            panic!("expected Disconnect");
        }
    }

    #[test]
    fn parse_ignore() {
        let payload = SshParser::build_ignore(b"padding data");
        let msg = SshParser::parse_message(&payload).unwrap();

        if let SshMessage::Ignore { data } = msg {
            assert_eq!(data, b"padding data");
        } else {
            panic!("expected Ignore");
        }
    }

    #[test]
    fn parse_service_request() {
        let payload = SshParser::build_service_request("ssh-userauth");
        let msg = SshParser::parse_message(&payload).unwrap();

        if let SshMessage::ServiceRequest { service_name } = msg {
            assert_eq!(service_name, "ssh-userauth");
        } else {
            panic!("expected ServiceRequest");
        }
    }

    #[test]
    fn parse_service_accept() {
        let payload = SshParser::build_service_accept("ssh-userauth");
        let msg = SshParser::parse_message(&payload).unwrap();

        if let SshMessage::ServiceAccept { service_name } = msg {
            assert_eq!(service_name, "ssh-userauth");
        } else {
            panic!("expected ServiceAccept");
        }
    }

    #[test]
    fn parse_newkeys() {
        let payload = SshParser::build_newkeys();
        let msg = SshParser::parse_message(&payload).unwrap();
        assert!(matches!(msg, SshMessage::NewKeys));
    }

    #[test]
    fn parse_kexinit_via_parser() {
        let ki = crate::kex::default_client_kex_init();
        let bytes = ki.to_bytes();
        let msg = SshParser::parse_message(&bytes).unwrap();

        if let SshMessage::KexInit(parsed) = msg {
            assert_eq!(parsed.kex_algorithms, ki.kex_algorithms);
        } else {
            panic!("expected KexInit");
        }
    }

    #[test]
    fn parse_unknown_message() {
        let payload = vec![200, 0x01, 0x02, 0x03];
        let msg = SshParser::parse_message(&payload).unwrap();

        if let SshMessage::Unknown { msg_type, data } = msg {
            assert_eq!(msg_type, 200);
            assert_eq!(data, vec![0x01, 0x02, 0x03]);
        } else {
            panic!("expected Unknown");
        }
    }

    #[test]
    fn parse_empty_payload() {
        assert!(SshParser::parse_message(&[]).is_err());
    }

    #[test]
    fn message_type_classification() {
        let msg = SshMessage::NewKeys;
        assert!(msg.is_transport());
        assert!(!msg.is_auth());
        assert!(!msg.is_channel());

        let msg = SshMessage::UserauthSuccess;
        assert!(!msg.is_transport());
        assert!(msg.is_auth());

        let msg = SshMessage::ChannelOpen {
            channel_type: "session".into(),
            sender_channel: 0,
            initial_window_size: 0,
            maximum_packet_size: 0,
        };
        assert!(msg.is_channel());
    }

    #[test]
    fn name_list_roundtrip() {
        let names = vec!["foo".to_string(), "bar".to_string(), "baz".to_string()];
        let encoded = encode_name_list(&names);
        assert_eq!(encoded, "foo,bar,baz");
        let decoded = parse_name_list_str(&encoded);
        assert_eq!(decoded, names);
    }

    #[test]
    fn name_list_empty() {
        assert!(parse_name_list_str("").is_empty());
        assert_eq!(encode_name_list(&[]), "");
    }

    #[test]
    fn encode_ssh_string_roundtrip() {
        let data = b"hello";
        let encoded = SshParser::encode_ssh_string(data);
        let result: IResult<&[u8], &[u8]> = ssh_string(&encoded);
        let (_, parsed) = result.unwrap();
        assert_eq!(parsed, data);
    }

    #[test]
    fn message_display() {
        let msg = SshMessage::NewKeys;
        assert_eq!(format!("{}", msg), "SSH_MSG_NEWKEYS (21)");

        let msg = SshMessage::Disconnect {
            reason_code: 2,
            description: "test".into(),
            language_tag: "en".into(),
        };
        assert_eq!(format!("{}", msg), "SSH_MSG_DISCONNECT (1)");
    }

    #[test]
    fn recover_malformed() {
        let msg = SshParser::recover_message(&[42, 1, 2, 3]);
        assert_eq!(msg.msg_type(), 42);
    }

    #[test]
    fn recover_empty() {
        let msg = SshParser::recover_message(&[]);
        assert_eq!(msg.msg_type(), 0);
    }

    #[test]
    fn parse_kexdh_init() {
        let e_data = vec![0x01, 0x02, 0x03, 0x04];
        let mut payload = vec![SSH_MSG_KEXDH_INIT];
        payload.extend_from_slice(&SshParser::encode_ssh_string(&e_data));

        let msg = SshParser::parse_message(&payload).unwrap();
        if let SshMessage::KexDhInit { e } = msg {
            assert_eq!(e, e_data);
        } else {
            panic!("expected KexDhInit");
        }
    }

    #[test]
    fn parse_kexdh_reply() {
        let host_key = vec![0x01, 0x02];
        let f_data = vec![0x03, 0x04];
        let sig = vec![0x05, 0x06];

        let mut payload = vec![SSH_MSG_KEXDH_REPLY];
        payload.extend_from_slice(&SshParser::encode_ssh_string(&host_key));
        payload.extend_from_slice(&SshParser::encode_ssh_string(&f_data));
        payload.extend_from_slice(&SshParser::encode_ssh_string(&sig));

        let msg = SshParser::parse_message(&payload).unwrap();
        if let SshMessage::KexDhReply {
            host_key: hk,
            f,
            signature,
        } = msg
        {
            assert_eq!(hk, host_key);
            assert_eq!(f, f_data);
            assert_eq!(signature, sig);
        } else {
            panic!("expected KexDhReply");
        }
    }

    #[test]
    fn parse_userauth_failure() {
        let mut payload = vec![SSH_MSG_USERAUTH_FAILURE];
        // name-list: "publickey,password"
        let nl = "publickey,password";
        payload.extend_from_slice(&(nl.len() as u32).to_be_bytes());
        payload.extend_from_slice(nl.as_bytes());
        payload.push(0); // partial_success = false

        let msg = SshParser::parse_message(&payload).unwrap();
        if let SshMessage::UserauthFailure {
            authentications,
            partial_success,
        } = msg
        {
            assert_eq!(authentications, vec!["publickey", "password"]);
            assert!(!partial_success);
        } else {
            panic!("expected UserauthFailure");
        }
    }

    #[test]
    fn parse_channel_open() {
        let mut payload = vec![SSH_MSG_CHANNEL_OPEN];
        payload.extend_from_slice(&SshParser::encode_ssh_utf8_string("session"));
        payload.extend_from_slice(&0u32.to_be_bytes());
        payload.extend_from_slice(&65536u32.to_be_bytes());
        payload.extend_from_slice(&32768u32.to_be_bytes());

        let msg = SshParser::parse_message(&payload).unwrap();
        if let SshMessage::ChannelOpen {
            channel_type,
            sender_channel,
            initial_window_size,
            maximum_packet_size,
        } = msg
        {
            assert_eq!(channel_type, "session");
            assert_eq!(sender_channel, 0);
            assert_eq!(initial_window_size, 65536);
            assert_eq!(maximum_packet_size, 32768);
        } else {
            panic!("expected ChannelOpen");
        }
    }

    #[test]
    fn parse_channel_open_confirmation() {
        let mut payload = vec![SSH_MSG_CHANNEL_OPEN_CONFIRMATION];
        payload.extend_from_slice(&0u32.to_be_bytes());
        payload.extend_from_slice(&1u32.to_be_bytes());
        payload.extend_from_slice(&65536u32.to_be_bytes());
        payload.extend_from_slice(&32768u32.to_be_bytes());

        let msg = SshParser::parse_message(&payload).unwrap();
        if let SshMessage::ChannelOpenConfirmation {
            recipient_channel,
            sender_channel,
            initial_window_size,
            maximum_packet_size,
        } = msg
        {
            assert_eq!(recipient_channel, 0);
            assert_eq!(sender_channel, 1);
            assert_eq!(initial_window_size, 65536);
            assert_eq!(maximum_packet_size, 32768);
        } else {
            panic!("expected ChannelOpenConfirmation");
        }
    }

    #[test]
    fn encode_name_list_bytes_roundtrip() {
        let names = vec!["aes256-ctr".to_string(), "aes128-ctr".to_string()];
        let encoded = encode_name_list_bytes(&names);
        let result: IResult<&[u8], &str> = ssh_utf8_string(&encoded);
        let (_, parsed) = result.unwrap();
        assert_eq!(parsed, "aes256-ctr,aes128-ctr");
    }
}
