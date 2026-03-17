//! SSH transport-layer handshake state machine — RFC 4253.
//!
//! Models the SSH handshake as a finite state machine (FSM) with states
//! from `Initial` through `ChannelOpen`.  Transition rules enforce the
//! message ordering required by the RFC and optionally enforce strict-KEX
//! constraints.

use crate::constants::*;
use crate::extensions::{ExtensionNegotiationState, StrictKex, StrictKexPhase};
use crate::kex::{KexInit, RekeyState};
use crate::packet::SequenceNumbers;
use crate::{SshError, SshResult};
use negsyn_types::{Extension, HandshakePhase, NegotiationState, ProtocolVersion};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

// ---------------------------------------------------------------------------
// States
// ---------------------------------------------------------------------------

/// SSH handshake states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SshHandshakeState {
    /// Connection established, no data exchanged.
    Initial,
    /// Version strings have been exchanged.
    VersionExchanged,
    /// We have sent our KEXINIT.
    KexInitSent,
    /// We have received the peer's KEXINIT.
    KexInitReceived,
    /// Both sides have exchanged KEXINIT; key exchange algorithm running.
    KexInProgress,
    /// NEWKEYS sent and received — new keys are in effect.
    NewKeysExchanged,
    /// SSH_MSG_SERVICE_REQUEST sent.
    ServiceRequested,
    /// SSH_MSG_SERVICE_ACCEPT received.
    ServiceAccepted,
    /// User authentication completed successfully.
    Authenticated,
    /// A channel has been opened.
    ChannelOpen,
    /// Re-keying in progress.
    Rekeying,
    /// Disconnected or error.
    Disconnected,
}

impl SshHandshakeState {
    /// Map to the generic `HandshakePhase` used by negsyn-types.
    pub fn to_handshake_phase(self) -> HandshakePhase {
        match self {
            Self::Initial => HandshakePhase::Init,
            Self::VersionExchanged => HandshakePhase::ClientHelloSent,
            Self::KexInitSent | Self::KexInitReceived | Self::KexInProgress => {
                HandshakePhase::Negotiated
            }
            Self::NewKeysExchanged => HandshakePhase::Done,
            Self::ServiceRequested | Self::ServiceAccepted => HandshakePhase::Done,
            Self::Authenticated | Self::ChannelOpen => HandshakePhase::Done,
            Self::Rekeying => HandshakePhase::Negotiated,
            Self::Disconnected => HandshakePhase::Abort,
        }
    }

    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Disconnected)
    }

    pub fn all_states() -> &'static [Self] {
        &[
            Self::Initial,
            Self::VersionExchanged,
            Self::KexInitSent,
            Self::KexInitReceived,
            Self::KexInProgress,
            Self::NewKeysExchanged,
            Self::ServiceRequested,
            Self::ServiceAccepted,
            Self::Authenticated,
            Self::ChannelOpen,
            Self::Rekeying,
            Self::Disconnected,
        ]
    }
}

impl fmt::Display for SshHandshakeState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ---------------------------------------------------------------------------
// Transition
// ---------------------------------------------------------------------------

/// An SSH state transition triggered by a message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SshTransition {
    pub from: SshHandshakeState,
    pub to: SshHandshakeState,
    pub msg_type: u8,
    pub msg_name: String,
    pub direction: MessageDirection,
}

/// Direction of the message that triggered the transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MessageDirection {
    Send,
    Receive,
}

impl fmt::Display for MessageDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Send => write!(f, "→"),
            Self::Receive => write!(f, "←"),
        }
    }
}

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

/// SSH transport-layer finite state machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SshStateMachine {
    /// Current state.
    pub state: SshHandshakeState,
    /// Whether we are the client (true) or server (false).
    pub is_client: bool,
    /// Sequence number tracking.
    pub sequence_numbers: SequenceNumbers,
    /// Extension negotiation state.
    #[serde(skip)]
    pub extensions: ExtensionNegotiationState,
    /// Strict-KEX state.
    pub strict_kex: StrictKex,
    /// Re-key state.
    pub rekey: RekeyState,
    /// Transition history.
    pub history: Vec<SshTransition>,
    /// Whether the very first KEXINIT has been seen.
    pub first_kexinit_seen: bool,
    /// Whether both KEXINITs have been exchanged (ready for DH).
    pub both_kexinits_exchanged: bool,
    /// Whether NEWKEYS has been sent by us.
    pub newkeys_sent: bool,
    /// Whether NEWKEYS has been received from peer.
    pub newkeys_received: bool,
    /// Client KEXINIT (cached for negotiation).
    pub client_kexinit: Option<KexInit>,
    /// Server KEXINIT (cached for negotiation).
    pub server_kexinit: Option<KexInit>,
}

impl SshStateMachine {
    pub fn new(is_client: bool) -> Self {
        Self {
            state: SshHandshakeState::Initial,
            is_client,
            sequence_numbers: SequenceNumbers::new(),
            extensions: ExtensionNegotiationState::new(),
            strict_kex: StrictKex::new(),
            rekey: RekeyState::new(),
            history: Vec::new(),
            first_kexinit_seen: false,
            both_kexinits_exchanged: false,
            newkeys_sent: false,
            newkeys_received: false,
            client_kexinit: None,
            server_kexinit: None,
        }
    }

    pub fn client() -> Self {
        Self::new(true)
    }

    pub fn server() -> Self {
        Self::new(false)
    }

    /// Process version string exchange.
    pub fn process_version_exchange(&mut self) -> SshResult<()> {
        if self.state != SshHandshakeState::Initial {
            return Err(SshError::InvalidTransition {
                from: self.state.to_string(),
                to: "VersionExchanged".into(),
            });
        }
        self.record_transition(
            SshHandshakeState::VersionExchanged,
            0, // no message type for version exchange
            "VERSION_EXCHANGE",
            MessageDirection::Send,
        );
        self.state = SshHandshakeState::VersionExchanged;
        Ok(())
    }

    /// Process sending our KEXINIT.
    pub fn send_kexinit(&mut self, kex_init: &KexInit) -> SshResult<()> {
        let valid_from = matches!(
            self.state,
            SshHandshakeState::VersionExchanged
                | SshHandshakeState::KexInitReceived
                | SshHandshakeState::Authenticated
                | SshHandshakeState::ChannelOpen
        );
        if !valid_from {
            return Err(SshError::InvalidTransition {
                from: self.state.to_string(),
                to: "KexInitSent".into(),
            });
        }

        if self.is_client {
            self.client_kexinit = Some(kex_init.clone());
            self.extensions.process_client_kexinit(kex_init);
            self.strict_kex.process_client_kexinit(kex_init);
        } else {
            self.server_kexinit = Some(kex_init.clone());
            self.extensions.process_server_kexinit(kex_init);
            self.strict_kex.process_server_kexinit(kex_init);
        }

        let next = if self.state == SshHandshakeState::KexInitReceived {
            self.both_kexinits_exchanged = true;
            SshHandshakeState::KexInProgress
        } else if matches!(
            self.state,
            SshHandshakeState::Authenticated | SshHandshakeState::ChannelOpen
        ) {
            SshHandshakeState::Rekeying
        } else {
            SshHandshakeState::KexInitSent
        };

        self.first_kexinit_seen = true;
        self.sequence_numbers.increment_send();

        self.record_transition(next, SSH_MSG_KEXINIT, "KEXINIT", MessageDirection::Send);
        self.state = next;
        Ok(())
    }

    /// Process receiving peer's KEXINIT.
    pub fn receive_kexinit(&mut self, kex_init: &KexInit) -> SshResult<()> {
        // Strict-KEX validation
        if !self.first_kexinit_seen && self.strict_kex.active {
            // First message after version exchange must be KEXINIT
            self.strict_kex
                .validate_message(SSH_MSG_KEXINIT, StrictKexPhase::BeforeKexInit)?;
        }

        let valid_from = matches!(
            self.state,
            SshHandshakeState::VersionExchanged
                | SshHandshakeState::KexInitSent
                | SshHandshakeState::Authenticated
                | SshHandshakeState::ChannelOpen
                | SshHandshakeState::Rekeying
        );
        if !valid_from {
            return Err(SshError::InvalidTransition {
                from: self.state.to_string(),
                to: "KexInitReceived".into(),
            });
        }

        if self.is_client {
            self.server_kexinit = Some(kex_init.clone());
            self.extensions.process_server_kexinit(kex_init);
            self.strict_kex.process_server_kexinit(kex_init);
        } else {
            self.client_kexinit = Some(kex_init.clone());
            self.extensions.process_client_kexinit(kex_init);
            self.strict_kex.process_client_kexinit(kex_init);
        }

        let next = if self.state == SshHandshakeState::KexInitSent
            || self.state == SshHandshakeState::Rekeying
        {
            self.both_kexinits_exchanged = true;
            SshHandshakeState::KexInProgress
        } else if matches!(
            self.state,
            SshHandshakeState::Authenticated | SshHandshakeState::ChannelOpen
        ) {
            SshHandshakeState::Rekeying
        } else {
            SshHandshakeState::KexInitReceived
        };

        self.first_kexinit_seen = true;
        self.sequence_numbers.increment_recv();

        self.record_transition(
            next,
            SSH_MSG_KEXINIT,
            "KEXINIT",
            MessageDirection::Receive,
        );
        self.state = next;
        Ok(())
    }

    /// Process a KEX DH message (KEXDH_INIT, KEXDH_REPLY, etc.).
    pub fn process_kex_dh(&mut self, msg_type: u8, direction: MessageDirection) -> SshResult<()> {
        let valid = matches!(
            self.state,
            SshHandshakeState::KexInProgress | SshHandshakeState::Rekeying
        );
        if !valid {
            return Err(SshError::UnexpectedMessage {
                msg_type,
                state: self.state.to_string(),
            });
        }

        // Strict-KEX: validate message during KEX
        if self.strict_kex.active {
            self.strict_kex
                .validate_message(msg_type, StrictKexPhase::DuringKex)?;
        }

        match direction {
            MessageDirection::Send => {
                self.sequence_numbers.increment_send();
            }
            MessageDirection::Receive => {
                self.sequence_numbers.increment_recv();
            }
        }

        let name = match msg_type {
            SSH_MSG_KEXDH_INIT => "KEXDH_INIT",
            SSH_MSG_KEXDH_REPLY => "KEXDH_REPLY",
            _ => "KEX_DH_MSG",
        };

        self.record_transition(self.state, msg_type, name, direction);
        Ok(())
    }

    /// Process sending NEWKEYS.
    pub fn send_newkeys(&mut self) -> SshResult<()> {
        let valid = matches!(
            self.state,
            SshHandshakeState::KexInProgress | SshHandshakeState::Rekeying
        );
        if !valid {
            return Err(SshError::InvalidTransition {
                from: self.state.to_string(),
                to: "NewKeysExchanged (send)".into(),
            });
        }

        self.sequence_numbers.increment_send();
        self.newkeys_sent = true;

        if self.newkeys_received {
            self.complete_newkeys();
        }

        self.record_transition(
            self.state,
            SSH_MSG_NEWKEYS,
            "NEWKEYS",
            MessageDirection::Send,
        );
        Ok(())
    }

    /// Process receiving NEWKEYS.
    pub fn receive_newkeys(&mut self) -> SshResult<()> {
        let valid = matches!(
            self.state,
            SshHandshakeState::KexInProgress | SshHandshakeState::Rekeying
        );
        if !valid {
            return Err(SshError::InvalidTransition {
                from: self.state.to_string(),
                to: "NewKeysExchanged (recv)".into(),
            });
        }

        self.sequence_numbers.increment_recv();
        self.newkeys_received = true;

        if self.newkeys_sent {
            self.complete_newkeys();
        }

        self.record_transition(
            self.state,
            SSH_MSG_NEWKEYS,
            "NEWKEYS",
            MessageDirection::Receive,
        );
        Ok(())
    }

    fn complete_newkeys(&mut self) {
        // Reset sequence numbers in strict-KEX mode
        if self.strict_kex.should_reset_sequence_numbers() {
            self.sequence_numbers.reset();
            self.strict_kex.mark_initial_kex_done();
        }

        let is_rekey = self.strict_kex.initial_kex_done;
        if is_rekey {
            // Return to pre-rekey state
            self.state = SshHandshakeState::Authenticated;
            self.rekey.complete_rekey();
        } else {
            self.state = SshHandshakeState::NewKeysExchanged;
        }

        self.newkeys_sent = false;
        self.newkeys_received = false;
        self.both_kexinits_exchanged = false;
    }

    /// Process SSH_MSG_EXT_INFO (can be sent after NEWKEYS).
    pub fn process_ext_info(
        &mut self,
        ext_info: &crate::extensions::ExtInfo,
        direction: MessageDirection,
    ) -> SshResult<()> {
        let valid = matches!(
            self.state,
            SshHandshakeState::NewKeysExchanged
                | SshHandshakeState::ServiceRequested
                | SshHandshakeState::ServiceAccepted
        );
        if !valid {
            return Err(SshError::UnexpectedMessage {
                msg_type: SSH_MSG_EXT_INFO,
                state: self.state.to_string(),
            });
        }

        match direction {
            MessageDirection::Send => {
                self.sequence_numbers.increment_send();
                if self.is_client {
                    self.extensions.process_client_ext_info(ext_info);
                } else {
                    self.extensions.process_server_ext_info(ext_info);
                }
            }
            MessageDirection::Receive => {
                self.sequence_numbers.increment_recv();
                if self.is_client {
                    self.extensions.process_server_ext_info(ext_info);
                } else {
                    self.extensions.process_client_ext_info(ext_info);
                }
            }
        }

        self.record_transition(
            self.state,
            SSH_MSG_EXT_INFO,
            "EXT_INFO",
            direction,
        );
        Ok(())
    }

    /// Process SSH_MSG_SERVICE_REQUEST.
    pub fn send_service_request(&mut self) -> SshResult<()> {
        if self.state != SshHandshakeState::NewKeysExchanged {
            return Err(SshError::InvalidTransition {
                from: self.state.to_string(),
                to: "ServiceRequested".into(),
            });
        }

        self.sequence_numbers.increment_send();
        self.record_transition(
            SshHandshakeState::ServiceRequested,
            SSH_MSG_SERVICE_REQUEST,
            "SERVICE_REQUEST",
            MessageDirection::Send,
        );
        self.state = SshHandshakeState::ServiceRequested;
        Ok(())
    }

    /// Process SSH_MSG_SERVICE_ACCEPT.
    pub fn receive_service_accept(&mut self) -> SshResult<()> {
        if self.state != SshHandshakeState::ServiceRequested {
            return Err(SshError::InvalidTransition {
                from: self.state.to_string(),
                to: "ServiceAccepted".into(),
            });
        }

        self.sequence_numbers.increment_recv();
        self.record_transition(
            SshHandshakeState::ServiceAccepted,
            SSH_MSG_SERVICE_ACCEPT,
            "SERVICE_ACCEPT",
            MessageDirection::Receive,
        );
        self.state = SshHandshakeState::ServiceAccepted;
        Ok(())
    }

    /// Process SSH_MSG_USERAUTH_SUCCESS.
    pub fn process_auth_success(&mut self) -> SshResult<()> {
        let valid = matches!(
            self.state,
            SshHandshakeState::ServiceAccepted | SshHandshakeState::ServiceRequested
        );
        if !valid {
            return Err(SshError::InvalidTransition {
                from: self.state.to_string(),
                to: "Authenticated".into(),
            });
        }

        self.sequence_numbers.increment_recv();
        self.record_transition(
            SshHandshakeState::Authenticated,
            SSH_MSG_USERAUTH_SUCCESS,
            "USERAUTH_SUCCESS",
            MessageDirection::Receive,
        );
        self.state = SshHandshakeState::Authenticated;
        Ok(())
    }

    /// Process SSH_MSG_CHANNEL_OPEN_CONFIRMATION.
    pub fn process_channel_open(&mut self) -> SshResult<()> {
        if self.state != SshHandshakeState::Authenticated {
            return Err(SshError::InvalidTransition {
                from: self.state.to_string(),
                to: "ChannelOpen".into(),
            });
        }

        self.sequence_numbers.increment_recv();
        self.record_transition(
            SshHandshakeState::ChannelOpen,
            SSH_MSG_CHANNEL_OPEN_CONFIRMATION,
            "CHANNEL_OPEN_CONFIRMATION",
            MessageDirection::Receive,
        );
        self.state = SshHandshakeState::ChannelOpen;
        Ok(())
    }

    /// Process SSH_MSG_DISCONNECT.
    pub fn process_disconnect(&mut self, direction: MessageDirection) -> SshResult<()> {
        match direction {
            MessageDirection::Send => {
                self.sequence_numbers.increment_send();
            }
            MessageDirection::Receive => {
                self.sequence_numbers.increment_recv();
            }
        }

        self.record_transition(
            SshHandshakeState::Disconnected,
            SSH_MSG_DISCONNECT,
            "DISCONNECT",
            direction,
        );
        self.state = SshHandshakeState::Disconnected;
        Ok(())
    }

    /// Process an arbitrary incoming message, updating sequence numbers and
    /// performing strict-KEX validation.
    pub fn process_incoming_message(&mut self, msg_type: u8) -> SshResult<()> {
        // Strict-KEX validation
        if self.strict_kex.active && !self.strict_kex.initial_kex_done {
            let phase = match self.state {
                SshHandshakeState::VersionExchanged => StrictKexPhase::BeforeKexInit,
                SshHandshakeState::KexInitSent
                | SshHandshakeState::KexInitReceived
                | SshHandshakeState::KexInProgress => StrictKexPhase::DuringKex,
                _ => StrictKexPhase::AfterNewKeys,
            };
            self.strict_kex.validate_message(msg_type, phase)?;
        }

        self.sequence_numbers.increment_recv();
        Ok(())
    }

    /// Build a `NegotiationState` from the current FSM state.
    pub fn to_negotiation_state(&self) -> NegotiationState {
        let mut ns = NegotiationState::initial();
        ns.phase = self.state.to_handshake_phase();
        ns.version = Some(ProtocolVersion::ssh2());

        // Populate extensions
        for ext in &self.extensions.server_extensions {
            ns.extensions.push(Extension::new(
                0,
                ext.name().to_string(),
                Vec::new(),
                false,
            ));
        }

        ns
    }

    /// Check if the FSM is in a state where a re-key can be initiated.
    pub fn can_rekey(&self) -> bool {
        matches!(
            self.state,
            SshHandshakeState::Authenticated | SshHandshakeState::ChannelOpen
        )
    }

    /// Get the transition history.
    pub fn transitions(&self) -> &[SshTransition] {
        &self.history
    }

    fn record_transition(
        &mut self,
        to: SshHandshakeState,
        msg_type: u8,
        msg_name: &str,
        direction: MessageDirection,
    ) {
        self.history.push(SshTransition {
            from: self.state,
            to,
            msg_type,
            msg_name: msg_name.to_string(),
            direction,
        });
    }
}

// ---------------------------------------------------------------------------
// Labelled Transition System builder
// ---------------------------------------------------------------------------

/// An LTS state for SSH negotiation (wraps negsyn-types types).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SshNegotiationLtsState {
    pub id: u64,
    pub ssh_state: SshHandshakeState,
    pub negotiation: NegotiationState,
}

/// An LTS transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SshNegotiationLtsTransition {
    pub from_id: u64,
    pub to_id: u64,
    pub label: String,
    pub msg_type: u8,
}

/// Builds a Negotiation Labelled Transition System from SSH rules.
#[derive(Debug, Clone)]
pub struct SshNegotiationLts {
    pub states: Vec<SshNegotiationLtsState>,
    pub transitions: Vec<SshNegotiationLtsTransition>,
}

impl SshNegotiationLts {
    /// Build the LTS for the standard SSH handshake.
    pub fn build_standard() -> Self {
        let mut states = Vec::new();
        let mut transitions = Vec::new();
        let mut id = 0u64;

        // Create a state for each SshHandshakeState
        let state_ids: Vec<(SshHandshakeState, u64)> = SshHandshakeState::all_states()
            .iter()
            .map(|s| {
                let sid = id;
                id += 1;
                states.push(SshNegotiationLtsState {
                    id: sid,
                    ssh_state: *s,
                    negotiation: {
                        let mut ns = NegotiationState::initial();
                        ns.phase = s.to_handshake_phase();
                        ns.version = Some(ProtocolVersion::ssh2());
                        ns
                    },
                });
                (*s, sid)
            })
            .collect();

        let find_id = |s: SshHandshakeState| -> u64 {
            state_ids.iter().find(|(st, _)| *st == s).unwrap().1
        };

        // Standard transitions
        let standard_transitions: Vec<(SshHandshakeState, SshHandshakeState, &str, u8)> = vec![
            (
                SshHandshakeState::Initial,
                SshHandshakeState::VersionExchanged,
                "VERSION_EXCHANGE",
                0,
            ),
            (
                SshHandshakeState::VersionExchanged,
                SshHandshakeState::KexInitSent,
                "KEXINIT_SEND",
                SSH_MSG_KEXINIT,
            ),
            (
                SshHandshakeState::VersionExchanged,
                SshHandshakeState::KexInitReceived,
                "KEXINIT_RECV",
                SSH_MSG_KEXINIT,
            ),
            (
                SshHandshakeState::KexInitSent,
                SshHandshakeState::KexInProgress,
                "KEXINIT_RECV",
                SSH_MSG_KEXINIT,
            ),
            (
                SshHandshakeState::KexInitReceived,
                SshHandshakeState::KexInProgress,
                "KEXINIT_SEND",
                SSH_MSG_KEXINIT,
            ),
            (
                SshHandshakeState::KexInProgress,
                SshHandshakeState::KexInProgress,
                "KEXDH_INIT",
                SSH_MSG_KEXDH_INIT,
            ),
            (
                SshHandshakeState::KexInProgress,
                SshHandshakeState::KexInProgress,
                "KEXDH_REPLY",
                SSH_MSG_KEXDH_REPLY,
            ),
            (
                SshHandshakeState::KexInProgress,
                SshHandshakeState::NewKeysExchanged,
                "NEWKEYS",
                SSH_MSG_NEWKEYS,
            ),
            (
                SshHandshakeState::NewKeysExchanged,
                SshHandshakeState::NewKeysExchanged,
                "EXT_INFO",
                SSH_MSG_EXT_INFO,
            ),
            (
                SshHandshakeState::NewKeysExchanged,
                SshHandshakeState::ServiceRequested,
                "SERVICE_REQUEST",
                SSH_MSG_SERVICE_REQUEST,
            ),
            (
                SshHandshakeState::ServiceRequested,
                SshHandshakeState::ServiceAccepted,
                "SERVICE_ACCEPT",
                SSH_MSG_SERVICE_ACCEPT,
            ),
            (
                SshHandshakeState::ServiceAccepted,
                SshHandshakeState::Authenticated,
                "USERAUTH_SUCCESS",
                SSH_MSG_USERAUTH_SUCCESS,
            ),
            (
                SshHandshakeState::Authenticated,
                SshHandshakeState::ChannelOpen,
                "CHANNEL_OPEN_CONFIRMATION",
                SSH_MSG_CHANNEL_OPEN_CONFIRMATION,
            ),
            // Re-keying transitions
            (
                SshHandshakeState::Authenticated,
                SshHandshakeState::Rekeying,
                "REKEY_KEXINIT",
                SSH_MSG_KEXINIT,
            ),
            (
                SshHandshakeState::ChannelOpen,
                SshHandshakeState::Rekeying,
                "REKEY_KEXINIT",
                SSH_MSG_KEXINIT,
            ),
            (
                SshHandshakeState::Rekeying,
                SshHandshakeState::Rekeying,
                "KEXDH_MSG",
                SSH_MSG_KEXDH_INIT,
            ),
            (
                SshHandshakeState::Rekeying,
                SshHandshakeState::Authenticated,
                "NEWKEYS",
                SSH_MSG_NEWKEYS,
            ),
        ];

        for (from, to, label, msg_type) in standard_transitions {
            transitions.push(SshNegotiationLtsTransition {
                from_id: find_id(from),
                to_id: find_id(to),
                label: label.to_string(),
                msg_type,
            });
        }

        // Disconnect can happen from any non-terminal state
        for s in SshHandshakeState::all_states() {
            if *s != SshHandshakeState::Disconnected {
                transitions.push(SshNegotiationLtsTransition {
                    from_id: find_id(*s),
                    to_id: find_id(SshHandshakeState::Disconnected),
                    label: "DISCONNECT".to_string(),
                    msg_type: SSH_MSG_DISCONNECT,
                });
            }
        }

        Self {
            states,
            transitions,
        }
    }

    /// Returns the number of states.
    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    /// Returns the number of transitions.
    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }

    /// Get all transitions from a given state ID.
    pub fn transitions_from(&self, state_id: u64) -> Vec<&SshNegotiationLtsTransition> {
        self.transitions
            .iter()
            .filter(|t| t.from_id == state_id)
            .collect()
    }

    /// Get all transitions to a given state ID.
    pub fn transitions_to(&self, state_id: u64) -> Vec<&SshNegotiationLtsTransition> {
        self.transitions
            .iter()
            .filter(|t| t.to_id == state_id)
            .collect()
    }

    /// Check if a state is reachable from the initial state.
    pub fn is_reachable(&self, target_id: u64) -> bool {
        let initial_id = self
            .states
            .iter()
            .find(|s| s.ssh_state == SshHandshakeState::Initial)
            .map(|s| s.id);

        if let Some(start) = initial_id {
            let mut visited = BTreeSet::new();
            let mut stack = vec![start];

            while let Some(current) = stack.pop() {
                if current == target_id {
                    return true;
                }
                if visited.insert(current) {
                    for t in self.transitions_from(current) {
                        if !visited.contains(&t.to_id) {
                            stack.push(t.to_id);
                        }
                    }
                }
            }
        }
        false
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kex::KexInitBuilder;

    fn make_client_kexinit() -> KexInit {
        KexInitBuilder::new()
            .kex_algorithms(vec!["curve25519-sha256".into()])
            .server_host_key_algorithms(vec!["ssh-ed25519".into()])
            .encryption_c2s(vec!["aes256-gcm@openssh.com".into()])
            .encryption_s2c(vec!["aes256-gcm@openssh.com".into()])
            .mac_c2s(vec!["hmac-sha2-256".into()])
            .mac_s2c(vec!["hmac-sha2-256".into()])
            .with_strict_kex_client()
            .with_ext_info_c()
            .build()
    }

    fn make_server_kexinit() -> KexInit {
        KexInitBuilder::new()
            .kex_algorithms(vec!["curve25519-sha256".into()])
            .server_host_key_algorithms(vec!["ssh-ed25519".into()])
            .encryption_c2s(vec!["aes256-gcm@openssh.com".into()])
            .encryption_s2c(vec!["aes256-gcm@openssh.com".into()])
            .mac_c2s(vec!["hmac-sha2-256".into()])
            .mac_s2c(vec!["hmac-sha2-256".into()])
            .with_strict_kex_server()
            .with_ext_info_s()
            .build()
    }

    #[test]
    fn full_handshake_client() {
        let mut sm = SshStateMachine::client();
        let cki = make_client_kexinit();
        let ski = make_server_kexinit();

        sm.process_version_exchange().unwrap();
        assert_eq!(sm.state, SshHandshakeState::VersionExchanged);

        sm.send_kexinit(&cki).unwrap();
        assert_eq!(sm.state, SshHandshakeState::KexInitSent);

        sm.receive_kexinit(&ski).unwrap();
        assert_eq!(sm.state, SshHandshakeState::KexInProgress);

        // Strict-KEX should be active
        assert!(sm.strict_kex.active);

        sm.process_kex_dh(SSH_MSG_KEXDH_INIT, MessageDirection::Send)
            .unwrap();
        sm.process_kex_dh(SSH_MSG_KEXDH_REPLY, MessageDirection::Receive)
            .unwrap();

        sm.send_newkeys().unwrap();
        sm.receive_newkeys().unwrap();
        assert_eq!(sm.state, SshHandshakeState::NewKeysExchanged);

        // Sequence numbers should have been reset by strict-KEX
        assert_eq!(sm.sequence_numbers.send_seq, 0);
        assert_eq!(sm.sequence_numbers.recv_seq, 0);

        sm.send_service_request().unwrap();
        assert_eq!(sm.state, SshHandshakeState::ServiceRequested);

        sm.receive_service_accept().unwrap();
        assert_eq!(sm.state, SshHandshakeState::ServiceAccepted);

        sm.process_auth_success().unwrap();
        assert_eq!(sm.state, SshHandshakeState::Authenticated);

        sm.process_channel_open().unwrap();
        assert_eq!(sm.state, SshHandshakeState::ChannelOpen);
    }

    #[test]
    fn invalid_transition_rejected() {
        let mut sm = SshStateMachine::client();
        // Can't send KEXINIT from Initial state
        assert!(sm.send_kexinit(&make_client_kexinit()).is_err());
    }

    #[test]
    fn disconnect_from_any_state() {
        for &state in SshHandshakeState::all_states() {
            if state == SshHandshakeState::Disconnected {
                continue;
            }
            let mut sm = SshStateMachine::client();
            sm.state = state;
            sm.process_disconnect(MessageDirection::Receive).unwrap();
            assert_eq!(sm.state, SshHandshakeState::Disconnected);
        }
    }

    #[test]
    fn rekey_flow() {
        let mut sm = SshStateMachine::client();
        let cki = make_client_kexinit();
        let ski = make_server_kexinit();

        // Get to Authenticated
        sm.state = SshHandshakeState::Authenticated;
        sm.strict_kex.initial_kex_done = true;

        // Initiate rekey
        sm.send_kexinit(&cki).unwrap();
        assert_eq!(sm.state, SshHandshakeState::Rekeying);

        sm.receive_kexinit(&ski).unwrap();
        // Still Rekeying since both KI came in during rekey
        // The receive should handle appropriately

        sm.process_kex_dh(SSH_MSG_KEXDH_INIT, MessageDirection::Send)
            .unwrap();
        sm.process_kex_dh(SSH_MSG_KEXDH_REPLY, MessageDirection::Receive)
            .unwrap();

        sm.send_newkeys().unwrap();
        sm.receive_newkeys().unwrap();

        // Should return to Authenticated
        assert_eq!(sm.state, SshHandshakeState::Authenticated);
        assert_eq!(sm.rekey.rekey_count, 1);
    }

    #[test]
    fn to_negotiation_state() {
        let mut sm = SshStateMachine::client();
        sm.state = SshHandshakeState::KexInProgress;
        sm.client_kexinit = Some(make_client_kexinit());

        let ns = sm.to_negotiation_state();
        assert_eq!(ns.phase, HandshakePhase::Negotiated);
        assert_eq!(ns.version, Some(ProtocolVersion::ssh2()));
    }

    #[test]
    fn transition_history() {
        let mut sm = SshStateMachine::client();
        sm.process_version_exchange().unwrap();
        sm.send_kexinit(&make_client_kexinit()).unwrap();

        assert_eq!(sm.history.len(), 2);
        assert_eq!(sm.history[0].msg_name, "VERSION_EXCHANGE");
        assert_eq!(sm.history[1].msg_name, "KEXINIT");
    }

    #[test]
    fn lts_build() {
        let lts = SshNegotiationLts::build_standard();
        assert_eq!(lts.state_count(), SshHandshakeState::all_states().len());
        assert!(lts.transition_count() > 10);

        // All states should be reachable from Initial
        for state in &lts.states {
            if state.ssh_state != SshHandshakeState::Initial {
                assert!(
                    lts.is_reachable(state.id),
                    "{:?} should be reachable",
                    state.ssh_state
                );
            }
        }
    }

    #[test]
    fn lts_transitions_from() {
        let lts = SshNegotiationLts::build_standard();
        let initial = lts
            .states
            .iter()
            .find(|s| s.ssh_state == SshHandshakeState::Initial)
            .unwrap();
        let from_initial = lts.transitions_from(initial.id);
        // Should have VERSION_EXCHANGE + DISCONNECT
        assert!(from_initial.len() >= 2);
    }

    #[test]
    fn state_to_handshake_phase() {
        assert_eq!(
            SshHandshakeState::Initial.to_handshake_phase(),
            HandshakePhase::Init
        );
        assert_eq!(
            SshHandshakeState::KexInProgress.to_handshake_phase(),
            HandshakePhase::Negotiated
        );
        assert_eq!(
            SshHandshakeState::Authenticated.to_handshake_phase(),
            HandshakePhase::Done
        );
        assert_eq!(
            SshHandshakeState::Disconnected.to_handshake_phase(),
            HandshakePhase::Abort
        );
    }

    #[test]
    fn can_rekey_check() {
        let mut sm = SshStateMachine::client();
        sm.state = SshHandshakeState::KexInProgress;
        assert!(!sm.can_rekey());

        sm.state = SshHandshakeState::Authenticated;
        assert!(sm.can_rekey());

        sm.state = SshHandshakeState::ChannelOpen;
        assert!(sm.can_rekey());
    }

    #[test]
    fn strict_kex_blocks_ignore_during_kex() {
        let mut sm = SshStateMachine::client();
        sm.strict_kex.active = true;
        sm.state = SshHandshakeState::KexInProgress;

        // SSH_MSG_IGNORE should be blocked during KEX in strict mode
        let result = sm.process_kex_dh(SSH_MSG_IGNORE, MessageDirection::Receive);
        assert!(result.is_err());
    }
}
