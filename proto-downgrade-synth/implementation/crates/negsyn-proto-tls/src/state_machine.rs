//! TLS handshake state machine (RFC 5246 §7.3, RFC 8446 §Appendix A).
//!
//! Models the full TLS handshake finite state machine for both TLS 1.2 and
//! TLS 1.3, including session resumption, renegotiation, and 0-RTT.

use crate::handshake::HandshakeType;
use crate::version::TlsVersion;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Handshake states
// ---------------------------------------------------------------------------

/// States in the TLS handshake state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TlsState {
    // Common states
    Start,
    Error,
    Connected,

    // TLS 1.2 client states
    ClientWaitServerHello,
    ClientWaitCertificate,
    ClientWaitServerKeyExchange,
    ClientWaitCertificateRequest,
    ClientWaitServerHelloDone,
    ClientWaitChangeCipherSpec,
    ClientWaitFinished,

    // TLS 1.2 server states
    ServerWaitClientHello,
    ServerSentServerHello,
    ServerSentCertificate,
    ServerSentServerKeyExchange,
    ServerSentCertificateRequest,
    ServerSentServerHelloDone,
    ServerWaitClientKeyExchange,
    ServerWaitCertificateVerify,
    ServerWaitChangeCipherSpec,
    ServerWaitFinished,

    // TLS 1.3 client states
    Client13WaitServerHello,
    Client13WaitEncryptedExtensions,
    Client13WaitCertificateRequest,
    Client13WaitCertificate,
    Client13WaitCertificateVerify,
    Client13WaitFinished,

    // TLS 1.3 server states
    Server13WaitClientHello,
    Server13SentServerHello,
    Server13WaitClientCertificate,
    Server13WaitClientCertificateVerify,
    Server13WaitFinished,
    Server13WaitEndOfEarlyData,

    // Session resumption
    ResumeClientWaitServerHello,
    ResumeClientWaitChangeCipherSpec,
    ResumeClientWaitFinished,
    ResumeServerWaitChangeCipherSpec,
    ResumeServerWaitFinished,

    // Renegotiation
    RenegotiationPending,
}

impl TlsState {
    /// Whether this state is a terminal (connected or error) state.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Connected | Self::Error)
    }

    /// Whether this is a client-side state.
    pub fn is_client_state(&self) -> bool {
        matches!(
            self,
            Self::Start
                | Self::ClientWaitServerHello
                | Self::ClientWaitCertificate
                | Self::ClientWaitServerKeyExchange
                | Self::ClientWaitCertificateRequest
                | Self::ClientWaitServerHelloDone
                | Self::ClientWaitChangeCipherSpec
                | Self::ClientWaitFinished
                | Self::Client13WaitServerHello
                | Self::Client13WaitEncryptedExtensions
                | Self::Client13WaitCertificateRequest
                | Self::Client13WaitCertificate
                | Self::Client13WaitCertificateVerify
                | Self::Client13WaitFinished
                | Self::ResumeClientWaitServerHello
                | Self::ResumeClientWaitChangeCipherSpec
                | Self::ResumeClientWaitFinished
                | Self::Connected
                | Self::Error
        )
    }

    /// Whether this is a server-side state.
    pub fn is_server_state(&self) -> bool {
        matches!(
            self,
            Self::ServerWaitClientHello
                | Self::ServerSentServerHello
                | Self::ServerSentCertificate
                | Self::ServerSentServerKeyExchange
                | Self::ServerSentCertificateRequest
                | Self::ServerSentServerHelloDone
                | Self::ServerWaitClientKeyExchange
                | Self::ServerWaitCertificateVerify
                | Self::ServerWaitChangeCipherSpec
                | Self::ServerWaitFinished
                | Self::Server13WaitClientHello
                | Self::Server13SentServerHello
                | Self::Server13WaitClientCertificate
                | Self::Server13WaitClientCertificateVerify
                | Self::Server13WaitFinished
                | Self::Server13WaitEndOfEarlyData
                | Self::ResumeServerWaitChangeCipherSpec
                | Self::ResumeServerWaitFinished
                | Self::Connected
                | Self::Error
        )
    }

    /// Convert to negsyn_types HandshakePhase.
    pub fn to_handshake_phase(&self) -> negsyn_types::HandshakePhase {
        match self {
            Self::Start | Self::ServerWaitClientHello | Self::Server13WaitClientHello => {
                negsyn_types::HandshakePhase::Init
            }
            Self::ClientWaitServerHello | Self::Client13WaitServerHello
            | Self::ResumeClientWaitServerHello => negsyn_types::HandshakePhase::ClientHelloSent,
            Self::ServerSentServerHello | Self::Server13SentServerHello
            | Self::ClientWaitCertificate | Self::Client13WaitCertificate
            | Self::ServerSentCertificate
            | Self::ClientWaitServerKeyExchange | Self::ServerSentServerKeyExchange
            | Self::ServerWaitClientKeyExchange
            | Self::ClientWaitCertificateRequest | Self::Client13WaitCertificateRequest
            | Self::ServerSentCertificateRequest
            | Self::ClientWaitServerHelloDone | Self::ServerSentServerHelloDone
            | Self::Client13WaitEncryptedExtensions
            | Self::Client13WaitCertificateVerify
            | Self::Server13WaitClientCertificateVerify => {
                negsyn_types::HandshakePhase::ServerHelloReceived
            }
            Self::ClientWaitChangeCipherSpec | Self::ServerWaitChangeCipherSpec
            | Self::ResumeClientWaitChangeCipherSpec | Self::ResumeServerWaitChangeCipherSpec
            | Self::ClientWaitFinished | Self::ServerWaitFinished
            | Self::Client13WaitFinished | Self::Server13WaitFinished
            | Self::ResumeClientWaitFinished | Self::ResumeServerWaitFinished
            | Self::RenegotiationPending
            | Self::ServerWaitCertificateVerify
            | Self::Server13WaitClientCertificate
            | Self::Server13WaitEndOfEarlyData => {
                negsyn_types::HandshakePhase::Negotiated
            }
            Self::Connected => negsyn_types::HandshakePhase::Done,
            Self::Error => negsyn_types::HandshakePhase::Abort,
        }
    }
}

impl fmt::Display for TlsState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ---------------------------------------------------------------------------
// Events / transitions
// ---------------------------------------------------------------------------

/// Events that trigger state transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TlsEvent {
    SendClientHello,
    RecvClientHello,
    SendServerHello,
    RecvServerHello,
    SendCertificate,
    RecvCertificate,
    SendServerKeyExchange,
    RecvServerKeyExchange,
    SendCertificateRequest,
    RecvCertificateRequest,
    SendServerHelloDone,
    RecvServerHelloDone,
    SendClientKeyExchange,
    RecvClientKeyExchange,
    SendCertificateVerify,
    RecvCertificateVerify,
    SendChangeCipherSpec,
    RecvChangeCipherSpec,
    SendFinished,
    RecvFinished,
    SendEncryptedExtensions,
    RecvEncryptedExtensions,
    SendEndOfEarlyData,
    RecvEndOfEarlyData,
    SendKeyUpdate,
    RecvKeyUpdate,
    SendAlert,
    RecvAlert,
    SendHelloRequest,
    RecvHelloRequest,
    ApplicationData,
}

impl fmt::Display for TlsEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ---------------------------------------------------------------------------
// Transition rule
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TransitionRule {
    pub from: TlsState,
    pub event: TlsEvent,
    pub to: TlsState,
    pub version: TlsVersion,
    pub description: &'static str,
}

// ---------------------------------------------------------------------------
// TLS State Machine
// ---------------------------------------------------------------------------

/// The TLS handshake state machine.
#[derive(Debug, Clone)]
pub struct TlsStateMachine {
    current: TlsState,
    version: TlsVersion,
    is_server: bool,
    is_resumption: bool,
    transitions: Vec<TransitionRule>,
    history: Vec<(TlsState, TlsEvent, TlsState)>,
    message_count: u32,
}

impl TlsStateMachine {
    /// Create a new state machine for the client role.
    pub fn new_client(version: TlsVersion) -> Self {
        let transitions = if version >= TlsVersion::TLS1_3 {
            Self::tls13_client_transitions()
        } else {
            Self::tls12_client_transitions()
        };
        Self {
            current: TlsState::Start,
            version,
            is_server: false,
            is_resumption: false,
            transitions,
            history: Vec::new(),
            message_count: 0,
        }
    }

    /// Create a new state machine for the server role.
    pub fn new_server(version: TlsVersion) -> Self {
        let transitions = if version >= TlsVersion::TLS1_3 {
            Self::tls13_server_transitions()
        } else {
            Self::tls12_server_transitions()
        };
        Self {
            current: if version >= TlsVersion::TLS1_3 {
                TlsState::Server13WaitClientHello
            } else {
                TlsState::ServerWaitClientHello
            },
            version,
            is_server: true,
            is_resumption: false,
            transitions,
            history: Vec::new(),
            message_count: 0,
        }
    }

    /// Create a state machine configured for session resumption.
    pub fn new_resumption_client(version: TlsVersion) -> Self {
        let mut sm = Self::new_client(version);
        sm.is_resumption = true;
        sm.transitions.extend(Self::resumption_client_transitions());
        sm
    }

    pub fn new_resumption_server(version: TlsVersion) -> Self {
        let mut sm = Self::new_server(version);
        sm.is_resumption = true;
        sm.transitions.extend(Self::resumption_server_transitions());
        sm
    }

    /// Process an event and transition to the next state.
    pub fn process_event(&mut self, event: TlsEvent) -> Result<TlsState, StateMachineError> {
        // Alert always transitions to Error.
        if matches!(event, TlsEvent::SendAlert | TlsEvent::RecvAlert) {
            let old = self.current;
            self.current = TlsState::Error;
            self.history.push((old, event, TlsState::Error));
            self.message_count += 1;
            return Ok(TlsState::Error);
        }

        // Find a matching transition.
        let transition = self
            .transitions
            .iter()
            .find(|t| t.from == self.current && t.event == event);

        match transition {
            Some(t) => {
                let old = self.current;
                self.current = t.to;
                self.history.push((old, event, t.to));
                self.message_count += 1;
                Ok(t.to)
            }
            None => Err(StateMachineError::InvalidTransition {
                from: self.current,
                event,
            }),
        }
    }

    /// Get the current state.
    pub fn current_state(&self) -> TlsState {
        self.current
    }

    /// Check if the handshake is complete.
    pub fn is_connected(&self) -> bool {
        self.current == TlsState::Connected
    }

    /// Check if the machine is in an error state.
    pub fn is_error(&self) -> bool {
        self.current == TlsState::Error
    }

    /// Get valid events from the current state.
    pub fn valid_events(&self) -> Vec<TlsEvent> {
        self.transitions
            .iter()
            .filter(|t| t.from == self.current)
            .map(|t| t.event)
            .collect()
    }

    /// Get the transition history.
    pub fn history(&self) -> &[(TlsState, TlsEvent, TlsState)] {
        &self.history
    }

    /// Number of messages processed.
    pub fn message_count(&self) -> u32 {
        self.message_count
    }

    /// The TLS version this FSM is configured for.
    pub fn version(&self) -> TlsVersion {
        self.version
    }

    /// Reset to the initial state.
    pub fn reset(&mut self) {
        self.current = if self.is_server {
            if self.version >= TlsVersion::TLS1_3 {
                TlsState::Server13WaitClientHello
            } else {
                TlsState::ServerWaitClientHello
            }
        } else {
            TlsState::Start
        };
        self.history.clear();
        self.message_count = 0;
    }

    /// Initiate renegotiation (TLS 1.2 only).
    pub fn initiate_renegotiation(&mut self) -> Result<(), StateMachineError> {
        if self.version >= TlsVersion::TLS1_3 {
            return Err(StateMachineError::RenegotiationNotAllowed);
        }
        if self.current != TlsState::Connected {
            return Err(StateMachineError::InvalidTransition {
                from: self.current,
                event: TlsEvent::RecvHelloRequest,
            });
        }
        self.current = TlsState::RenegotiationPending;
        self.history.push((
            TlsState::Connected,
            TlsEvent::RecvHelloRequest,
            TlsState::RenegotiationPending,
        ));
        Ok(())
    }

    // -----------------------------------------------------------------------
    // TLS 1.2 full handshake transitions (client)
    // -----------------------------------------------------------------------
    fn tls12_client_transitions() -> Vec<TransitionRule> {
        vec![
            TransitionRule {
                from: TlsState::Start,
                event: TlsEvent::SendClientHello,
                to: TlsState::ClientWaitServerHello,
                version: TlsVersion::TLS1_2,
                description: "Client sends ClientHello",
            },
            TransitionRule {
                from: TlsState::ClientWaitServerHello,
                event: TlsEvent::RecvServerHello,
                to: TlsState::ClientWaitCertificate,
                version: TlsVersion::TLS1_2,
                description: "Client receives ServerHello",
            },
            TransitionRule {
                from: TlsState::ClientWaitCertificate,
                event: TlsEvent::RecvCertificate,
                to: TlsState::ClientWaitServerKeyExchange,
                version: TlsVersion::TLS1_2,
                description: "Client receives Certificate",
            },
            TransitionRule {
                from: TlsState::ClientWaitServerKeyExchange,
                event: TlsEvent::RecvServerKeyExchange,
                to: TlsState::ClientWaitCertificateRequest,
                version: TlsVersion::TLS1_2,
                description: "Client receives ServerKeyExchange",
            },
            // Skip ServerKeyExchange (RSA key exchange).
            TransitionRule {
                from: TlsState::ClientWaitServerKeyExchange,
                event: TlsEvent::RecvServerHelloDone,
                to: TlsState::ClientWaitChangeCipherSpec,
                version: TlsVersion::TLS1_2,
                description: "Client receives ServerHelloDone (no SKE)",
            },
            TransitionRule {
                from: TlsState::ClientWaitCertificateRequest,
                event: TlsEvent::RecvCertificateRequest,
                to: TlsState::ClientWaitServerHelloDone,
                version: TlsVersion::TLS1_2,
                description: "Client receives CertificateRequest",
            },
            // Skip CertificateRequest.
            TransitionRule {
                from: TlsState::ClientWaitCertificateRequest,
                event: TlsEvent::RecvServerHelloDone,
                to: TlsState::ClientWaitChangeCipherSpec,
                version: TlsVersion::TLS1_2,
                description: "Client receives ServerHelloDone (no CertReq)",
            },
            TransitionRule {
                from: TlsState::ClientWaitServerHelloDone,
                event: TlsEvent::RecvServerHelloDone,
                to: TlsState::ClientWaitChangeCipherSpec,
                version: TlsVersion::TLS1_2,
                description: "Client receives ServerHelloDone",
            },
            TransitionRule {
                from: TlsState::ClientWaitChangeCipherSpec,
                event: TlsEvent::SendClientKeyExchange,
                to: TlsState::ClientWaitChangeCipherSpec,
                version: TlsVersion::TLS1_2,
                description: "Client sends ClientKeyExchange",
            },
            TransitionRule {
                from: TlsState::ClientWaitChangeCipherSpec,
                event: TlsEvent::SendChangeCipherSpec,
                to: TlsState::ClientWaitChangeCipherSpec,
                version: TlsVersion::TLS1_2,
                description: "Client sends ChangeCipherSpec",
            },
            TransitionRule {
                from: TlsState::ClientWaitChangeCipherSpec,
                event: TlsEvent::SendFinished,
                to: TlsState::ClientWaitChangeCipherSpec,
                version: TlsVersion::TLS1_2,
                description: "Client sends Finished",
            },
            TransitionRule {
                from: TlsState::ClientWaitChangeCipherSpec,
                event: TlsEvent::RecvChangeCipherSpec,
                to: TlsState::ClientWaitFinished,
                version: TlsVersion::TLS1_2,
                description: "Client receives ChangeCipherSpec from server",
            },
            TransitionRule {
                from: TlsState::ClientWaitFinished,
                event: TlsEvent::RecvFinished,
                to: TlsState::Connected,
                version: TlsVersion::TLS1_2,
                description: "Client receives Finished -> connected",
            },
        ]
    }

    // -----------------------------------------------------------------------
    // TLS 1.2 full handshake transitions (server)
    // -----------------------------------------------------------------------
    fn tls12_server_transitions() -> Vec<TransitionRule> {
        vec![
            TransitionRule {
                from: TlsState::ServerWaitClientHello,
                event: TlsEvent::RecvClientHello,
                to: TlsState::ServerSentServerHello,
                version: TlsVersion::TLS1_2,
                description: "Server receives ClientHello",
            },
            TransitionRule {
                from: TlsState::ServerSentServerHello,
                event: TlsEvent::SendServerHello,
                to: TlsState::ServerSentCertificate,
                version: TlsVersion::TLS1_2,
                description: "Server sends ServerHello",
            },
            TransitionRule {
                from: TlsState::ServerSentCertificate,
                event: TlsEvent::SendCertificate,
                to: TlsState::ServerSentServerKeyExchange,
                version: TlsVersion::TLS1_2,
                description: "Server sends Certificate",
            },
            TransitionRule {
                from: TlsState::ServerSentServerKeyExchange,
                event: TlsEvent::SendServerKeyExchange,
                to: TlsState::ServerSentCertificateRequest,
                version: TlsVersion::TLS1_2,
                description: "Server sends ServerKeyExchange",
            },
            // Skip ServerKeyExchange.
            TransitionRule {
                from: TlsState::ServerSentServerKeyExchange,
                event: TlsEvent::SendServerHelloDone,
                to: TlsState::ServerWaitClientKeyExchange,
                version: TlsVersion::TLS1_2,
                description: "Server sends ServerHelloDone (no SKE)",
            },
            TransitionRule {
                from: TlsState::ServerSentCertificateRequest,
                event: TlsEvent::SendCertificateRequest,
                to: TlsState::ServerSentServerHelloDone,
                version: TlsVersion::TLS1_2,
                description: "Server sends CertificateRequest",
            },
            TransitionRule {
                from: TlsState::ServerSentCertificateRequest,
                event: TlsEvent::SendServerHelloDone,
                to: TlsState::ServerWaitClientKeyExchange,
                version: TlsVersion::TLS1_2,
                description: "Server sends ServerHelloDone (no CertReq)",
            },
            TransitionRule {
                from: TlsState::ServerSentServerHelloDone,
                event: TlsEvent::SendServerHelloDone,
                to: TlsState::ServerWaitClientKeyExchange,
                version: TlsVersion::TLS1_2,
                description: "Server sends ServerHelloDone",
            },
            TransitionRule {
                from: TlsState::ServerWaitClientKeyExchange,
                event: TlsEvent::RecvClientKeyExchange,
                to: TlsState::ServerWaitChangeCipherSpec,
                version: TlsVersion::TLS1_2,
                description: "Server receives ClientKeyExchange",
            },
            TransitionRule {
                from: TlsState::ServerWaitChangeCipherSpec,
                event: TlsEvent::RecvCertificateVerify,
                to: TlsState::ServerWaitChangeCipherSpec,
                version: TlsVersion::TLS1_2,
                description: "Server receives CertificateVerify",
            },
            TransitionRule {
                from: TlsState::ServerWaitChangeCipherSpec,
                event: TlsEvent::RecvChangeCipherSpec,
                to: TlsState::ServerWaitFinished,
                version: TlsVersion::TLS1_2,
                description: "Server receives ChangeCipherSpec",
            },
            TransitionRule {
                from: TlsState::ServerWaitFinished,
                event: TlsEvent::RecvFinished,
                to: TlsState::ServerWaitFinished,
                version: TlsVersion::TLS1_2,
                description: "Server receives client Finished",
            },
            TransitionRule {
                from: TlsState::ServerWaitFinished,
                event: TlsEvent::SendChangeCipherSpec,
                to: TlsState::ServerWaitFinished,
                version: TlsVersion::TLS1_2,
                description: "Server sends ChangeCipherSpec",
            },
            TransitionRule {
                from: TlsState::ServerWaitFinished,
                event: TlsEvent::SendFinished,
                to: TlsState::Connected,
                version: TlsVersion::TLS1_2,
                description: "Server sends Finished -> connected",
            },
        ]
    }

    // -----------------------------------------------------------------------
    // TLS 1.3 handshake transitions (client)
    // -----------------------------------------------------------------------
    fn tls13_client_transitions() -> Vec<TransitionRule> {
        vec![
            TransitionRule {
                from: TlsState::Start,
                event: TlsEvent::SendClientHello,
                to: TlsState::Client13WaitServerHello,
                version: TlsVersion::TLS1_3,
                description: "Client sends ClientHello (TLS 1.3)",
            },
            TransitionRule {
                from: TlsState::Client13WaitServerHello,
                event: TlsEvent::RecvServerHello,
                to: TlsState::Client13WaitEncryptedExtensions,
                version: TlsVersion::TLS1_3,
                description: "Client receives ServerHello (TLS 1.3)",
            },
            TransitionRule {
                from: TlsState::Client13WaitEncryptedExtensions,
                event: TlsEvent::RecvEncryptedExtensions,
                to: TlsState::Client13WaitCertificateRequest,
                version: TlsVersion::TLS1_3,
                description: "Client receives EncryptedExtensions",
            },
            TransitionRule {
                from: TlsState::Client13WaitCertificateRequest,
                event: TlsEvent::RecvCertificateRequest,
                to: TlsState::Client13WaitCertificate,
                version: TlsVersion::TLS1_3,
                description: "Client receives CertificateRequest",
            },
            // Skip CertificateRequest (server doesn't request client cert).
            TransitionRule {
                from: TlsState::Client13WaitCertificateRequest,
                event: TlsEvent::RecvCertificate,
                to: TlsState::Client13WaitCertificateVerify,
                version: TlsVersion::TLS1_3,
                description: "Client receives Certificate (no CertReq)",
            },
            TransitionRule {
                from: TlsState::Client13WaitCertificate,
                event: TlsEvent::RecvCertificate,
                to: TlsState::Client13WaitCertificateVerify,
                version: TlsVersion::TLS1_3,
                description: "Client receives Certificate",
            },
            TransitionRule {
                from: TlsState::Client13WaitCertificateVerify,
                event: TlsEvent::RecvCertificateVerify,
                to: TlsState::Client13WaitFinished,
                version: TlsVersion::TLS1_3,
                description: "Client receives CertificateVerify",
            },
            TransitionRule {
                from: TlsState::Client13WaitFinished,
                event: TlsEvent::RecvFinished,
                to: TlsState::Connected,
                version: TlsVersion::TLS1_3,
                description: "Client receives server Finished -> connected",
            },
        ]
    }

    // -----------------------------------------------------------------------
    // TLS 1.3 handshake transitions (server)
    // -----------------------------------------------------------------------
    fn tls13_server_transitions() -> Vec<TransitionRule> {
        vec![
            TransitionRule {
                from: TlsState::Server13WaitClientHello,
                event: TlsEvent::RecvClientHello,
                to: TlsState::Server13SentServerHello,
                version: TlsVersion::TLS1_3,
                description: "Server receives ClientHello (TLS 1.3)",
            },
            TransitionRule {
                from: TlsState::Server13SentServerHello,
                event: TlsEvent::SendServerHello,
                to: TlsState::Server13SentServerHello,
                version: TlsVersion::TLS1_3,
                description: "Server sends ServerHello",
            },
            TransitionRule {
                from: TlsState::Server13SentServerHello,
                event: TlsEvent::SendEncryptedExtensions,
                to: TlsState::Server13SentServerHello,
                version: TlsVersion::TLS1_3,
                description: "Server sends EncryptedExtensions",
            },
            TransitionRule {
                from: TlsState::Server13SentServerHello,
                event: TlsEvent::SendCertificate,
                to: TlsState::Server13SentServerHello,
                version: TlsVersion::TLS1_3,
                description: "Server sends Certificate",
            },
            TransitionRule {
                from: TlsState::Server13SentServerHello,
                event: TlsEvent::SendCertificateVerify,
                to: TlsState::Server13SentServerHello,
                version: TlsVersion::TLS1_3,
                description: "Server sends CertificateVerify",
            },
            TransitionRule {
                from: TlsState::Server13SentServerHello,
                event: TlsEvent::SendFinished,
                to: TlsState::Server13WaitFinished,
                version: TlsVersion::TLS1_3,
                description: "Server sends Finished, waits for client",
            },
            TransitionRule {
                from: TlsState::Server13WaitEndOfEarlyData,
                event: TlsEvent::RecvEndOfEarlyData,
                to: TlsState::Server13WaitFinished,
                version: TlsVersion::TLS1_3,
                description: "Server receives EndOfEarlyData",
            },
            TransitionRule {
                from: TlsState::Server13WaitFinished,
                event: TlsEvent::RecvCertificate,
                to: TlsState::Server13WaitClientCertificateVerify,
                version: TlsVersion::TLS1_3,
                description: "Server receives client Certificate",
            },
            TransitionRule {
                from: TlsState::Server13WaitClientCertificateVerify,
                event: TlsEvent::RecvCertificateVerify,
                to: TlsState::Server13WaitFinished,
                version: TlsVersion::TLS1_3,
                description: "Server receives client CertificateVerify",
            },
            TransitionRule {
                from: TlsState::Server13WaitFinished,
                event: TlsEvent::RecvFinished,
                to: TlsState::Connected,
                version: TlsVersion::TLS1_3,
                description: "Server receives client Finished -> connected",
            },
        ]
    }

    // -----------------------------------------------------------------------
    // Resumption transitions
    // -----------------------------------------------------------------------
    fn resumption_client_transitions() -> Vec<TransitionRule> {
        vec![
            TransitionRule {
                from: TlsState::Start,
                event: TlsEvent::SendClientHello,
                to: TlsState::ResumeClientWaitServerHello,
                version: TlsVersion::TLS1_2,
                description: "Resumption: client sends ClientHello",
            },
            TransitionRule {
                from: TlsState::ResumeClientWaitServerHello,
                event: TlsEvent::RecvServerHello,
                to: TlsState::ResumeClientWaitChangeCipherSpec,
                version: TlsVersion::TLS1_2,
                description: "Resumption: client receives ServerHello",
            },
            TransitionRule {
                from: TlsState::ResumeClientWaitChangeCipherSpec,
                event: TlsEvent::RecvChangeCipherSpec,
                to: TlsState::ResumeClientWaitFinished,
                version: TlsVersion::TLS1_2,
                description: "Resumption: client receives CCS",
            },
            TransitionRule {
                from: TlsState::ResumeClientWaitFinished,
                event: TlsEvent::RecvFinished,
                to: TlsState::ResumeClientWaitFinished,
                version: TlsVersion::TLS1_2,
                description: "Resumption: client receives server Finished",
            },
            TransitionRule {
                from: TlsState::ResumeClientWaitFinished,
                event: TlsEvent::SendChangeCipherSpec,
                to: TlsState::ResumeClientWaitFinished,
                version: TlsVersion::TLS1_2,
                description: "Resumption: client sends CCS",
            },
            TransitionRule {
                from: TlsState::ResumeClientWaitFinished,
                event: TlsEvent::SendFinished,
                to: TlsState::Connected,
                version: TlsVersion::TLS1_2,
                description: "Resumption: client sends Finished -> connected",
            },
        ]
    }

    fn resumption_server_transitions() -> Vec<TransitionRule> {
        vec![
            TransitionRule {
                from: TlsState::ServerWaitClientHello,
                event: TlsEvent::RecvClientHello,
                to: TlsState::ResumeServerWaitChangeCipherSpec,
                version: TlsVersion::TLS1_2,
                description: "Resumption: server receives ClientHello",
            },
            TransitionRule {
                from: TlsState::ResumeServerWaitChangeCipherSpec,
                event: TlsEvent::SendServerHello,
                to: TlsState::ResumeServerWaitChangeCipherSpec,
                version: TlsVersion::TLS1_2,
                description: "Resumption: server sends ServerHello",
            },
            TransitionRule {
                from: TlsState::ResumeServerWaitChangeCipherSpec,
                event: TlsEvent::SendChangeCipherSpec,
                to: TlsState::ResumeServerWaitChangeCipherSpec,
                version: TlsVersion::TLS1_2,
                description: "Resumption: server sends CCS",
            },
            TransitionRule {
                from: TlsState::ResumeServerWaitChangeCipherSpec,
                event: TlsEvent::SendFinished,
                to: TlsState::ResumeServerWaitFinished,
                version: TlsVersion::TLS1_2,
                description: "Resumption: server sends Finished",
            },
            TransitionRule {
                from: TlsState::ResumeServerWaitFinished,
                event: TlsEvent::RecvChangeCipherSpec,
                to: TlsState::ResumeServerWaitFinished,
                version: TlsVersion::TLS1_2,
                description: "Resumption: server receives CCS",
            },
            TransitionRule {
                from: TlsState::ResumeServerWaitFinished,
                event: TlsEvent::RecvFinished,
                to: TlsState::Connected,
                version: TlsVersion::TLS1_2,
                description: "Resumption: server receives Finished -> connected",
            },
        ]
    }

    /// Build a negotiation LTS description from the current FSM rules.
    pub fn build_negotiation_lts(&self) -> NegotiationLts {
        let mut lts = NegotiationLts::new(self.version);
        for rule in &self.transitions {
            lts.add_transition(rule.from, rule.event, rule.to, rule.description);
        }
        lts
    }
}

// ---------------------------------------------------------------------------
// State machine errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, thiserror::Error)]
pub enum StateMachineError {
    #[error("invalid transition from {from} on event {event}")]
    InvalidTransition { from: TlsState, event: TlsEvent },

    #[error("renegotiation not allowed in TLS 1.3")]
    RenegotiationNotAllowed,

    #[error("state machine already in terminal state: {0}")]
    TerminalState(TlsState),
}

// ---------------------------------------------------------------------------
// Negotiation LTS (Labeled Transition System)
// ---------------------------------------------------------------------------

/// A labeled transition system built from TLS handshake rules.
#[derive(Debug, Clone)]
pub struct NegotiationLts {
    pub version: TlsVersion,
    pub transitions: Vec<LtsTransition>,
    pub states: Vec<TlsState>,
}

#[derive(Debug, Clone)]
pub struct LtsTransition {
    pub source: TlsState,
    pub label: TlsEvent,
    pub target: TlsState,
    pub description: &'static str,
}

impl NegotiationLts {
    pub fn new(version: TlsVersion) -> Self {
        Self {
            version,
            transitions: Vec::new(),
            states: Vec::new(),
        }
    }

    pub fn add_transition(
        &mut self,
        source: TlsState,
        label: TlsEvent,
        target: TlsState,
        description: &'static str,
    ) {
        if !self.states.contains(&source) {
            self.states.push(source);
        }
        if !self.states.contains(&target) {
            self.states.push(target);
        }
        self.transitions.push(LtsTransition {
            source,
            label,
            target,
            description,
        });
    }

    /// Get all outgoing transitions from a state.
    pub fn transitions_from(&self, state: TlsState) -> Vec<&LtsTransition> {
        self.transitions.iter().filter(|t| t.source == state).collect()
    }

    /// Get all incoming transitions to a state.
    pub fn transitions_to(&self, state: TlsState) -> Vec<&LtsTransition> {
        self.transitions.iter().filter(|t| t.target == state).collect()
    }

    /// Number of states in the LTS.
    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    /// Number of transitions in the LTS.
    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tls12_client_full_handshake() {
        let mut sm = TlsStateMachine::new_client(TlsVersion::TLS1_2);
        assert_eq!(sm.current_state(), TlsState::Start);

        sm.process_event(TlsEvent::SendClientHello).unwrap();
        assert_eq!(sm.current_state(), TlsState::ClientWaitServerHello);

        sm.process_event(TlsEvent::RecvServerHello).unwrap();
        sm.process_event(TlsEvent::RecvCertificate).unwrap();
        sm.process_event(TlsEvent::RecvServerKeyExchange).unwrap();
        sm.process_event(TlsEvent::RecvServerHelloDone).unwrap();
        sm.process_event(TlsEvent::SendClientKeyExchange).unwrap();
        sm.process_event(TlsEvent::SendChangeCipherSpec).unwrap();
        sm.process_event(TlsEvent::SendFinished).unwrap();
        sm.process_event(TlsEvent::RecvChangeCipherSpec).unwrap();
        sm.process_event(TlsEvent::RecvFinished).unwrap();

        assert!(sm.is_connected());
        assert_eq!(sm.message_count(), 10);
    }

    #[test]
    fn test_tls12_server_full_handshake() {
        let mut sm = TlsStateMachine::new_server(TlsVersion::TLS1_2);
        assert_eq!(sm.current_state(), TlsState::ServerWaitClientHello);

        sm.process_event(TlsEvent::RecvClientHello).unwrap();
        sm.process_event(TlsEvent::SendServerHello).unwrap();
        sm.process_event(TlsEvent::SendCertificate).unwrap();
        sm.process_event(TlsEvent::SendServerHelloDone).unwrap();
        sm.process_event(TlsEvent::RecvClientKeyExchange).unwrap();
        sm.process_event(TlsEvent::RecvChangeCipherSpec).unwrap();
        sm.process_event(TlsEvent::RecvFinished).unwrap();
        sm.process_event(TlsEvent::SendChangeCipherSpec).unwrap();
        sm.process_event(TlsEvent::SendFinished).unwrap();

        assert!(sm.is_connected());
    }

    #[test]
    fn test_tls13_client_handshake() {
        let mut sm = TlsStateMachine::new_client(TlsVersion::TLS1_3);

        sm.process_event(TlsEvent::SendClientHello).unwrap();
        sm.process_event(TlsEvent::RecvServerHello).unwrap();
        sm.process_event(TlsEvent::RecvEncryptedExtensions).unwrap();
        // Skip CertificateRequest.
        sm.process_event(TlsEvent::RecvCertificate).unwrap();
        sm.process_event(TlsEvent::RecvCertificateVerify).unwrap();
        sm.process_event(TlsEvent::RecvFinished).unwrap();

        assert!(sm.is_connected());
    }

    #[test]
    fn test_tls13_server_handshake() {
        let mut sm = TlsStateMachine::new_server(TlsVersion::TLS1_3);

        sm.process_event(TlsEvent::RecvClientHello).unwrap();
        sm.process_event(TlsEvent::SendServerHello).unwrap();
        sm.process_event(TlsEvent::SendEncryptedExtensions).unwrap();
        sm.process_event(TlsEvent::SendCertificate).unwrap();
        sm.process_event(TlsEvent::SendCertificateVerify).unwrap();
        sm.process_event(TlsEvent::SendFinished).unwrap();
        sm.process_event(TlsEvent::RecvFinished).unwrap();

        assert!(sm.is_connected());
    }

    #[test]
    fn test_invalid_transition() {
        let mut sm = TlsStateMachine::new_client(TlsVersion::TLS1_2);
        let result = sm.process_event(TlsEvent::RecvFinished);
        assert!(result.is_err());
    }

    #[test]
    fn test_alert_transitions_to_error() {
        let mut sm = TlsStateMachine::new_client(TlsVersion::TLS1_2);
        sm.process_event(TlsEvent::SendClientHello).unwrap();
        sm.process_event(TlsEvent::RecvAlert).unwrap();
        assert!(sm.is_error());
    }

    #[test]
    fn test_valid_events() {
        let sm = TlsStateMachine::new_client(TlsVersion::TLS1_2);
        let events = sm.valid_events();
        assert!(events.contains(&TlsEvent::SendClientHello));
        assert!(!events.contains(&TlsEvent::RecvFinished));
    }

    #[test]
    fn test_renegotiation_tls12() {
        let mut sm = TlsStateMachine::new_client(TlsVersion::TLS1_2);
        sm.process_event(TlsEvent::SendClientHello).unwrap();
        sm.process_event(TlsEvent::RecvServerHello).unwrap();
        sm.process_event(TlsEvent::RecvCertificate).unwrap();
        sm.process_event(TlsEvent::RecvServerHelloDone).unwrap();
        sm.process_event(TlsEvent::SendClientKeyExchange).unwrap();
        sm.process_event(TlsEvent::SendChangeCipherSpec).unwrap();
        sm.process_event(TlsEvent::SendFinished).unwrap();
        sm.process_event(TlsEvent::RecvChangeCipherSpec).unwrap();
        sm.process_event(TlsEvent::RecvFinished).unwrap();
        assert!(sm.is_connected());

        sm.initiate_renegotiation().unwrap();
        assert_eq!(sm.current_state(), TlsState::RenegotiationPending);
    }

    #[test]
    fn test_renegotiation_tls13_not_allowed() {
        let mut sm = TlsStateMachine::new_client(TlsVersion::TLS1_3);
        sm.process_event(TlsEvent::SendClientHello).unwrap();
        sm.process_event(TlsEvent::RecvServerHello).unwrap();
        sm.process_event(TlsEvent::RecvEncryptedExtensions).unwrap();
        sm.process_event(TlsEvent::RecvCertificate).unwrap();
        sm.process_event(TlsEvent::RecvCertificateVerify).unwrap();
        sm.process_event(TlsEvent::RecvFinished).unwrap();

        let result = sm.initiate_renegotiation();
        assert!(result.is_err());
    }

    #[test]
    fn test_build_lts() {
        let sm = TlsStateMachine::new_client(TlsVersion::TLS1_2);
        let lts = sm.build_negotiation_lts();
        assert!(lts.state_count() > 0);
        assert!(lts.transition_count() > 0);
    }

    #[test]
    fn test_lts_transitions_from() {
        let sm = TlsStateMachine::new_client(TlsVersion::TLS1_2);
        let lts = sm.build_negotiation_lts();
        let from_start = lts.transitions_from(TlsState::Start);
        assert!(!from_start.is_empty());
    }

    #[test]
    fn test_reset() {
        let mut sm = TlsStateMachine::new_client(TlsVersion::TLS1_2);
        sm.process_event(TlsEvent::SendClientHello).unwrap();
        assert_ne!(sm.current_state(), TlsState::Start);
        sm.reset();
        assert_eq!(sm.current_state(), TlsState::Start);
        assert_eq!(sm.message_count(), 0);
    }

    #[test]
    fn test_state_to_handshake_phase() {
        assert_eq!(
            TlsState::Start.to_handshake_phase(),
            negsyn_types::HandshakePhase::Init
        );
        assert_eq!(
            TlsState::Connected.to_handshake_phase(),
            negsyn_types::HandshakePhase::Done
        );
        assert_eq!(
            TlsState::Error.to_handshake_phase(),
            negsyn_types::HandshakePhase::Abort
        );
    }

    #[test]
    fn test_history() {
        let mut sm = TlsStateMachine::new_client(TlsVersion::TLS1_2);
        sm.process_event(TlsEvent::SendClientHello).unwrap();
        let hist = sm.history();
        assert_eq!(hist.len(), 1);
        assert_eq!(hist[0].0, TlsState::Start);
        assert_eq!(hist[0].1, TlsEvent::SendClientHello);
        assert_eq!(hist[0].2, TlsState::ClientWaitServerHello);
    }
}
