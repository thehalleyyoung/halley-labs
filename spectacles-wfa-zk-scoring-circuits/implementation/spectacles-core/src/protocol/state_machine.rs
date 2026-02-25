use std::collections::HashMap;
use std::fmt;
use serde::{Serialize, Deserialize};

// ─────────────────────────────────────────────────────────────
// AbortReason
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AbortReason {
    ConstraintViolation,
    TimeoutExceeded,
    InvalidTransition,
    CommitmentMismatch,
    ProofFailed,
    ExternalAbort(String),
}

impl fmt::Display for AbortReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AbortReason::ConstraintViolation => write!(f, "ConstraintViolation"),
            AbortReason::TimeoutExceeded => write!(f, "TimeoutExceeded"),
            AbortReason::InvalidTransition => write!(f, "InvalidTransition"),
            AbortReason::CommitmentMismatch => write!(f, "CommitmentMismatch"),
            AbortReason::ProofFailed => write!(f, "ProofFailed"),
            AbortReason::ExternalAbort(s) => write!(f, "ExternalAbort({})", s),
        }
    }
}

// ─────────────────────────────────────────────────────────────
// ProtocolState
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ProtocolState {
    Initialized,
    CommitOutputs,
    RevealBenchmark,
    Evaluate,
    Prove,
    Verify,
    Certify,
    Completed,
    Aborted(AbortReason),
    TimedOut,
}

impl fmt::Display for ProtocolState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProtocolState::Initialized => write!(f, "Initialized"),
            ProtocolState::CommitOutputs => write!(f, "CommitOutputs"),
            ProtocolState::RevealBenchmark => write!(f, "RevealBenchmark"),
            ProtocolState::Evaluate => write!(f, "Evaluate"),
            ProtocolState::Prove => write!(f, "Prove"),
            ProtocolState::Verify => write!(f, "Verify"),
            ProtocolState::Certify => write!(f, "Certify"),
            ProtocolState::Completed => write!(f, "Completed"),
            ProtocolState::Aborted(reason) => write!(f, "Aborted({})", reason),
            ProtocolState::TimedOut => write!(f, "TimedOut"),
        }
    }
}

impl ProtocolState {
    /// Returns the discriminant index for ordered comparison.
    fn discriminant_index(&self) -> u8 {
        match self {
            ProtocolState::Initialized => 0,
            ProtocolState::CommitOutputs => 1,
            ProtocolState::RevealBenchmark => 2,
            ProtocolState::Evaluate => 3,
            ProtocolState::Prove => 4,
            ProtocolState::Verify => 5,
            ProtocolState::Certify => 6,
            ProtocolState::Completed => 7,
            ProtocolState::Aborted(_) => 8,
            ProtocolState::TimedOut => 9,
        }
    }

    /// Returns a string key for use in HashMaps.
    pub fn key(&self) -> String {
        match self {
            ProtocolState::Aborted(reason) => format!("Aborted({})", reason),
            other => format!("{}", other),
        }
    }
}

// ─────────────────────────────────────────────────────────────
// EventType
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum EventType {
    StateTransition,
    Timeout,
    Error,
    Commitment,
    Reveal,
    ProofGenerated,
    ProofVerified,
    CertificateIssued,
}

impl fmt::Display for EventType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EventType::StateTransition => write!(f, "StateTransition"),
            EventType::Timeout => write!(f, "Timeout"),
            EventType::Error => write!(f, "Error"),
            EventType::Commitment => write!(f, "Commitment"),
            EventType::Reveal => write!(f, "Reveal"),
            EventType::ProofGenerated => write!(f, "ProofGenerated"),
            EventType::ProofVerified => write!(f, "ProofVerified"),
            EventType::CertificateIssued => write!(f, "CertificateIssued"),
        }
    }
}

// ─────────────────────────────────────────────────────────────
// ProtocolEvent
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProtocolEvent {
    pub from_state: ProtocolState,
    pub to_state: ProtocolState,
    pub timestamp: u64,
    pub event_type: EventType,
    pub details: String,
}

impl ProtocolEvent {
    pub fn new(
        from_state: ProtocolState,
        to_state: ProtocolState,
        timestamp: u64,
        event_type: EventType,
        details: String,
    ) -> Self {
        Self {
            from_state,
            to_state,
            timestamp,
            event_type,
            details,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// StateTransitionRule
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateTransitionRule {
    pub from: ProtocolState,
    pub to: ProtocolState,
    pub condition: String,
    pub timeout_ms: u64,
}

impl StateTransitionRule {
    pub fn new(from: ProtocolState, to: ProtocolState, condition: &str, timeout_ms: u64) -> Self {
        Self {
            from,
            to,
            condition: condition.to_string(),
            timeout_ms,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// ProtocolConfig
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProtocolConfig {
    pub name: String,
    pub version: String,
    pub timeouts: HashMap<String, u64>,
    pub enable_logging: bool,
    pub max_retries: u32,
    pub require_grinding: bool,
}

impl Default for ProtocolConfig {
    fn default() -> Self {
        let mut timeouts = HashMap::new();
        timeouts.insert("Initialized".to_string(), 30_000);
        timeouts.insert("CommitOutputs".to_string(), 60_000);
        timeouts.insert("RevealBenchmark".to_string(), 60_000);
        timeouts.insert("Evaluate".to_string(), 120_000);
        timeouts.insert("Prove".to_string(), 300_000);
        timeouts.insert("Verify".to_string(), 60_000);
        timeouts.insert("Certify".to_string(), 30_000);

        Self {
            name: "spectacles-protocol".to_string(),
            version: "0.1.0".to_string(),
            timeouts,
            enable_logging: true,
            max_retries: 3,
            require_grinding: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// ProtocolError
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ProtocolError {
    InvalidTransition(ProtocolState, ProtocolState),
    Timeout(ProtocolState),
    AlreadyTerminal,
    CommitmentMismatch(String),
    SerializationError(String),
    InvalidState(String),
}

impl fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProtocolError::InvalidTransition(from, to) => {
                write!(f, "Invalid transition from {} to {}", from, to)
            }
            ProtocolError::Timeout(state) => write!(f, "Timeout in state {}", state),
            ProtocolError::AlreadyTerminal => write!(f, "Protocol is already in a terminal state"),
            ProtocolError::CommitmentMismatch(key) => {
                write!(f, "Commitment mismatch for key '{}'", key)
            }
            ProtocolError::SerializationError(msg) => {
                write!(f, "Serialization error: {}", msg)
            }
            ProtocolError::InvalidState(msg) => write!(f, "Invalid state: {}", msg),
        }
    }
}

impl std::error::Error for ProtocolError {}

// ─────────────────────────────────────────────────────────────
// Helper: current timestamp in milliseconds
// ─────────────────────────────────────────────────────────────

fn current_time_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ─────────────────────────────────────────────────────────────
// ProtocolStateMachine
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProtocolStateMachine {
    current_state: ProtocolState,
    history: Vec<ProtocolEvent>,
    transition_rules: Vec<StateTransitionRule>,
    timeouts: HashMap<String, u64>,
    start_time: u64,
    state_entry_time: u64,
    config: ProtocolConfig,
    committed_values: HashMap<String, Vec<u8>>,
    revealed_values: HashMap<String, Vec<u8>>,
}

impl ProtocolStateMachine {
    /// Create a new protocol state machine with the given configuration.
    pub fn new(config: ProtocolConfig) -> Self {
        let now = current_time_ms();
        let timeouts = config.timeouts.clone();

        let transition_rules = Self::default_transition_rules();

        Self {
            current_state: ProtocolState::Initialized,
            history: Vec::new(),
            transition_rules,
            timeouts,
            start_time: now,
            state_entry_time: now,
            config,
            committed_values: HashMap::new(),
            revealed_values: HashMap::new(),
        }
    }

    /// Build the default set of transition rules for the protocol.
    fn default_transition_rules() -> Vec<StateTransitionRule> {
        vec![
            StateTransitionRule::new(
                ProtocolState::Initialized,
                ProtocolState::CommitOutputs,
                "begin_commit",
                30_000,
            ),
            StateTransitionRule::new(
                ProtocolState::CommitOutputs,
                ProtocolState::RevealBenchmark,
                "outputs_committed",
                60_000,
            ),
            StateTransitionRule::new(
                ProtocolState::RevealBenchmark,
                ProtocolState::Evaluate,
                "benchmark_revealed",
                60_000,
            ),
            StateTransitionRule::new(
                ProtocolState::Evaluate,
                ProtocolState::Prove,
                "evaluation_complete",
                120_000,
            ),
            StateTransitionRule::new(
                ProtocolState::Prove,
                ProtocolState::Verify,
                "proof_generated",
                300_000,
            ),
            StateTransitionRule::new(
                ProtocolState::Verify,
                ProtocolState::Certify,
                "proof_verified",
                60_000,
            ),
            StateTransitionRule::new(
                ProtocolState::Certify,
                ProtocolState::Completed,
                "certificate_issued",
                30_000,
            ),
        ]
    }

    /// Returns a reference to the current protocol state.
    pub fn current_state(&self) -> &ProtocolState {
        &self.current_state
    }

    /// Attempt to transition to the target state. Returns an error if the
    /// transition is not allowed by the configured rules.
    pub fn transition_to(&mut self, target: ProtocolState) -> Result<(), ProtocolError> {
        if self.is_terminal() {
            return Err(ProtocolError::AlreadyTerminal);
        }

        // Abort transitions are always allowed from any non-terminal state
        if let ProtocolState::Aborted(_) = &target {
            let now = current_time_ms();
            let event = ProtocolEvent::new(
                self.current_state.clone(),
                target.clone(),
                now,
                EventType::StateTransition,
                format!("Abort from {}", self.current_state),
            );
            if self.config.enable_logging {
                self.history.push(event);
            }
            self.current_state = target;
            self.state_entry_time = now;
            return Ok(());
        }

        if !self.can_transition_to(&target) {
            return Err(ProtocolError::InvalidTransition(
                self.current_state.clone(),
                target,
            ));
        }

        let now = current_time_ms();
        let event_type = match &target {
            ProtocolState::CommitOutputs => EventType::Commitment,
            ProtocolState::RevealBenchmark => EventType::Reveal,
            ProtocolState::Prove => EventType::ProofGenerated,
            ProtocolState::Verify => EventType::ProofVerified,
            ProtocolState::Certify => EventType::CertificateIssued,
            ProtocolState::TimedOut => EventType::Timeout,
            _ => EventType::StateTransition,
        };

        let event = ProtocolEvent::new(
            self.current_state.clone(),
            target.clone(),
            now,
            event_type,
            format!("Transition {} -> {}", self.current_state, target),
        );

        if self.config.enable_logging {
            self.history.push(event);
        }

        self.current_state = target;
        self.state_entry_time = now;
        Ok(())
    }

    /// Check whether a transition to `target` is allowed from the current state.
    pub fn can_transition_to(&self, target: &ProtocolState) -> bool {
        if self.is_terminal() {
            return false;
        }

        // Abort is always valid from any non-terminal state
        if let ProtocolState::Aborted(_) = target {
            return true;
        }

        // TimedOut is always valid from any non-terminal state
        if *target == ProtocolState::TimedOut {
            return true;
        }

        self.transition_rules.iter().any(|rule| {
            rule.from == self.current_state && rule.to == *target
        })
    }

    /// List all valid states that can be reached from the current state.
    pub fn valid_transitions(&self) -> Vec<ProtocolState> {
        if self.is_terminal() {
            return Vec::new();
        }

        let mut targets: Vec<ProtocolState> = self
            .transition_rules
            .iter()
            .filter(|rule| rule.from == self.current_state)
            .map(|rule| rule.to.clone())
            .collect();

        // Abort is always valid
        targets.push(ProtocolState::Aborted(AbortReason::ExternalAbort(
            "user-initiated".to_string(),
        )));
        // TimedOut is always valid
        targets.push(ProtocolState::TimedOut);

        targets
    }

    /// Force-abort the protocol from any non-terminal state.
    pub fn abort(&mut self, reason: AbortReason) {
        if self.is_terminal() {
            return;
        }
        let now = current_time_ms();
        let target = ProtocolState::Aborted(reason.clone());

        let event = ProtocolEvent::new(
            self.current_state.clone(),
            target.clone(),
            now,
            EventType::Error,
            format!("Aborted: {}", reason),
        );
        if self.config.enable_logging {
            self.history.push(event);
        }
        self.current_state = target;
        self.state_entry_time = now;
    }

    /// Returns `true` if the protocol is in a terminal state (Completed or Aborted).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.current_state,
            ProtocolState::Completed | ProtocolState::Aborted(_) | ProtocolState::TimedOut
        )
    }

    /// Check whether the current state has exceeded its configured timeout.
    /// If so, automatically transitions to `TimedOut` and returns `true`.
    pub fn check_timeout(&mut self) -> bool {
        if self.is_terminal() {
            return false;
        }

        let key = self.current_state.key();
        if let Some(&timeout_ms) = self.timeouts.get(&key) {
            let elapsed = self.elapsed_in_state();
            if elapsed >= timeout_ms {
                let now = current_time_ms();
                let event = ProtocolEvent::new(
                    self.current_state.clone(),
                    ProtocolState::TimedOut,
                    now,
                    EventType::Timeout,
                    format!("Timeout after {}ms in state {}", elapsed, self.current_state),
                );
                if self.config.enable_logging {
                    self.history.push(event);
                }
                self.current_state = ProtocolState::TimedOut;
                self.state_entry_time = now;
                return true;
            }
        }
        false
    }

    /// Milliseconds elapsed since entering the current state.
    pub fn elapsed_in_state(&self) -> u64 {
        current_time_ms().saturating_sub(self.state_entry_time)
    }

    /// Total milliseconds elapsed since the protocol started.
    pub fn total_elapsed(&self) -> u64 {
        current_time_ms().saturating_sub(self.start_time)
    }

    /// Returns the full event log.
    pub fn event_log(&self) -> &[ProtocolEvent] {
        &self.history
    }

    /// Store a commitment value (the hash of the data).
    pub fn store_commitment(&mut self, key: String, value: Vec<u8>) {
        let hash = blake3::hash(&value);
        self.committed_values.insert(key.clone(), hash.as_bytes().to_vec());

        if self.config.enable_logging {
            let now = current_time_ms();
            let event = ProtocolEvent::new(
                self.current_state.clone(),
                self.current_state.clone(),
                now,
                EventType::Commitment,
                format!("Commitment stored for key '{}'", key),
            );
            self.history.push(event);
        }
    }

    /// Store a revealed value for later verification against the commitment.
    pub fn store_reveal(&mut self, key: String, value: Vec<u8>) {
        self.revealed_values.insert(key.clone(), value);

        if self.config.enable_logging {
            let now = current_time_ms();
            let event = ProtocolEvent::new(
                self.current_state.clone(),
                self.current_state.clone(),
                now,
                EventType::Reveal,
                format!("Reveal stored for key '{}'", key),
            );
            self.history.push(event);
        }
    }

    /// Verify that the revealed value for `key` matches its commitment (BLAKE3).
    pub fn verify_commitment_reveal(&self, key: &str) -> bool {
        let committed = match self.committed_values.get(key) {
            Some(c) => c,
            None => return false,
        };
        let revealed = match self.revealed_values.get(key) {
            Some(r) => r,
            None => return false,
        };
        let hash = blake3::hash(revealed);
        hash.as_bytes().as_slice() == committed.as_slice()
    }

    /// Reset the state machine to `Initialized`.
    pub fn reset(&mut self) {
        let now = current_time_ms();
        self.current_state = ProtocolState::Initialized;
        self.history.clear();
        self.committed_values.clear();
        self.revealed_values.clear();
        self.start_time = now;
        self.state_entry_time = now;
    }

    /// Serialize the entire state machine to bytes.
    pub fn serialize_state(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    /// Deserialize a state machine from bytes.
    pub fn deserialize_state(bytes: &[u8]) -> Result<Self, ProtocolError> {
        serde_json::from_slice(bytes)
            .map_err(|e| ProtocolError::SerializationError(e.to_string()))
    }

    /// Run the protocol by repeatedly calling `step_fn` to determine the next
    /// state. The step function receives the current state and returns the target
    /// state (or an error). The loop terminates when a terminal state is reached.
    pub fn run_protocol<F>(&mut self, step_fn: F) -> Result<ProtocolState, ProtocolError>
    where
        F: Fn(&ProtocolState) -> Result<ProtocolState, ProtocolError>,
    {
        let max_iterations = 100; // safety bound
        let mut iterations = 0;

        while !self.is_terminal() && iterations < max_iterations {
            if self.check_timeout() {
                return Ok(self.current_state.clone());
            }

            let next = step_fn(&self.current_state)?;
            self.transition_to(next)?;
            iterations += 1;
        }

        Ok(self.current_state.clone())
    }
}

// ─────────────────────────────────────────────────────────────
// ProtocolPhase / PhaseResult / ProtocolPhaseManager
// ─────────────────────────────────────────────────────────────

/// Describes a single phase within a multi-phase protocol execution.
#[derive(Clone, Debug)]
pub struct ProtocolPhase {
    pub name: String,
    pub required_state: ProtocolState,
    pub timeout_ms: u64,
    pub validator: fn(&PhaseResult) -> bool,
}

/// The result produced by executing a single protocol phase.
#[derive(Clone, Debug)]
pub struct PhaseResult {
    pub success: bool,
    pub duration_ms: u64,
    pub data: HashMap<String, Vec<u8>>,
    pub error: Option<String>,
}

impl PhaseResult {
    pub fn ok(duration_ms: u64) -> Self {
        Self {
            success: true,
            duration_ms,
            data: HashMap::new(),
            error: None,
        }
    }

    pub fn err(duration_ms: u64, error: String) -> Self {
        Self {
            success: false,
            duration_ms,
            data: HashMap::new(),
            error: Some(error),
        }
    }
}

/// Manages multi-phase protocol execution, tracking which phase is active and
/// accumulating results.
pub struct ProtocolPhaseManager {
    phases: Vec<ProtocolPhase>,
    current_phase_index: usize,
    phase_results: HashMap<String, PhaseResult>,
}

impl ProtocolPhaseManager {
    pub fn new(phases: Vec<ProtocolPhase>) -> Self {
        Self {
            phases,
            current_phase_index: 0,
            phase_results: HashMap::new(),
        }
    }

    /// Advance to the next phase. Fails if the current phase has not recorded a
    /// passing result or the protocol is already complete.
    pub fn advance(&mut self) -> Result<(), ProtocolError> {
        if self.is_complete() {
            return Err(ProtocolError::AlreadyTerminal);
        }

        let current_name = self.phases[self.current_phase_index].name.clone();
        if let Some(result) = self.phase_results.get(&current_name) {
            let validator = self.phases[self.current_phase_index].validator;
            if !validator(result) {
                return Err(ProtocolError::InvalidState(format!(
                    "Phase '{}' result did not pass validation",
                    current_name
                )));
            }
        } else {
            return Err(ProtocolError::InvalidState(format!(
                "Phase '{}' has no recorded result",
                current_name
            )));
        }

        self.current_phase_index += 1;
        Ok(())
    }

    pub fn current_phase(&self) -> &ProtocolPhase {
        &self.phases[self.current_phase_index.min(self.phases.len() - 1)]
    }

    pub fn is_complete(&self) -> bool {
        self.current_phase_index >= self.phases.len()
    }

    pub fn record_result(&mut self, name: String, result: PhaseResult) {
        self.phase_results.insert(name, result);
    }

    /// Returns `(phase_name, duration_ms)` pairs for every phase that has a
    /// recorded result, in the order the phases were declared.
    pub fn phase_durations(&self) -> Vec<(String, u64)> {
        self.phases
            .iter()
            .filter_map(|p| {
                self.phase_results
                    .get(&p.name)
                    .map(|r| (p.name.clone(), r.duration_ms))
            })
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────
// BackoffStrategy / RetryManager
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub enum BackoffStrategy {
    /// Wait a fixed number of milliseconds between retries.
    Fixed(u64),
    /// Increase delay linearly: `base * attempt_number`.
    Linear(u64),
    /// Increase delay exponentially: `base * multiplier^attempt`.
    Exponential(u64, f64),
    /// Use the Fibonacci sequence: 1, 1, 2, 3, 5, 8, …
    Fibonacci,
}

pub struct RetryManager {
    max_retries: u32,
    retry_count: HashMap<String, u32>,
    backoff_strategy: BackoffStrategy,
}

impl RetryManager {
    pub fn new(max_retries: u32, strategy: BackoffStrategy) -> Self {
        Self {
            max_retries,
            retry_count: HashMap::new(),
            backoff_strategy: strategy,
        }
    }

    pub fn should_retry(&self, state: &ProtocolState) -> bool {
        let key = state.key();
        let count = self.retry_count.get(&key).copied().unwrap_or(0);
        count < self.max_retries
    }

    pub fn record_attempt(&mut self, state: &ProtocolState) {
        let key = state.key();
        let count = self.retry_count.entry(key).or_insert(0);
        *count += 1;
    }

    pub fn reset(&mut self, state: &ProtocolState) {
        self.retry_count.remove(&state.key());
    }

    pub fn delay_ms(&self, state: &ProtocolState) -> u64 {
        let key = state.key();
        let attempt = self.retry_count.get(&key).copied().unwrap_or(0);
        match &self.backoff_strategy {
            BackoffStrategy::Fixed(ms) => *ms,
            BackoffStrategy::Linear(base) => base * (attempt as u64 + 1),
            BackoffStrategy::Exponential(base, multiplier) => {
                let factor = multiplier.powi(attempt as i32);
                (*base as f64 * factor) as u64
            }
            BackoffStrategy::Fibonacci => {
                fibonacci_ms(attempt)
            }
        }
    }
}

/// Returns the `n`-th Fibonacci number (in milliseconds), starting with
/// fib(0)=1, fib(1)=1.
fn fibonacci_ms(n: u32) -> u64 {
    if n == 0 || n == 1 {
        return 1;
    }
    let mut a: u64 = 1;
    let mut b: u64 = 1;
    for _ in 2..=n {
        let tmp = a + b;
        a = b;
        b = tmp;
    }
    b
}

// ─────────────────────────────────────────────────────────────
// AuditEvent / AuditSummary / ProtocolAuditor
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct AuditEvent {
    pub timestamp: u64,
    pub event_type: EventType,
    pub state_before: ProtocolState,
    pub state_after: ProtocolState,
    pub actor: String,
    pub details: String,
    pub hash: [u8; 32],
}

impl AuditEvent {
    /// Build a new audit event. The `hash` is computed over the serialised
    /// content of the event using BLAKE3.
    pub fn new(
        timestamp: u64,
        event_type: EventType,
        state_before: ProtocolState,
        state_after: ProtocolState,
        actor: String,
        details: String,
    ) -> Self {
        let mut data = Vec::new();
        data.extend_from_slice(&timestamp.to_le_bytes());
        data.extend_from_slice(format!("{}", event_type).as_bytes());
        data.extend_from_slice(format!("{}", state_before).as_bytes());
        data.extend_from_slice(format!("{}", state_after).as_bytes());
        data.extend_from_slice(actor.as_bytes());
        data.extend_from_slice(details.as_bytes());
        let hash: [u8; 32] = *blake3::hash(&data).as_bytes();

        Self {
            timestamp,
            event_type,
            state_before,
            state_after,
            actor,
            details,
            hash,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AuditSummary {
    pub total_events: usize,
    pub total_transitions: usize,
    pub successful_transitions: usize,
    pub failed_transitions: usize,
    pub duration_ms: u64,
    pub states_visited: Vec<ProtocolState>,
}

pub struct ProtocolAuditor {
    events: Vec<AuditEvent>,
    start_time: u64,
}

impl ProtocolAuditor {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            start_time: current_time_ms(),
        }
    }

    pub fn record_event(&mut self, event: AuditEvent) {
        self.events.push(event);
    }

    /// Produce a human-readable audit log string.
    pub fn generate_audit_log(&self) -> String {
        let mut log = String::from("=== PROTOCOL AUDIT LOG ===\n");
        for (i, ev) in self.events.iter().enumerate() {
            log.push_str(&format!(
                "[{}] t={} type={} {} -> {} actor={} | {}\n",
                i, ev.timestamp, ev.event_type, ev.state_before, ev.state_after, ev.actor, ev.details
            ));
        }
        log.push_str(&format!("Total events: {}\n", self.events.len()));
        log
    }

    /// Verify that the audit trail is correctly ordered (non-decreasing
    /// timestamps) and that every event's hash matches its content.
    pub fn verify_audit_trail(&self) -> bool {
        for i in 1..self.events.len() {
            if self.events[i].timestamp < self.events[i - 1].timestamp {
                return false;
            }
        }
        for ev in &self.events {
            let recomputed = AuditEvent::new(
                ev.timestamp,
                ev.event_type.clone(),
                ev.state_before.clone(),
                ev.state_after.clone(),
                ev.actor.clone(),
                ev.details.clone(),
            );
            if recomputed.hash != ev.hash {
                return false;
            }
        }
        true
    }

    /// Return references to events whose timestamp falls in `[start, end]`.
    pub fn events_in_range(&self, start: u64, end: u64) -> Vec<&AuditEvent> {
        self.events
            .iter()
            .filter(|e| e.timestamp >= start && e.timestamp <= end)
            .collect()
    }

    /// Compute aggregate statistics over the audit trail.
    pub fn summary(&self) -> AuditSummary {
        let total_events = self.events.len();
        let total_transitions = self
            .events
            .iter()
            .filter(|e| e.event_type == EventType::StateTransition)
            .count();

        let successful_transitions = self
            .events
            .iter()
            .filter(|e| {
                e.event_type == EventType::StateTransition
                    && !matches!(e.state_after, ProtocolState::Aborted(_) | ProtocolState::TimedOut)
            })
            .count();

        let failed_transitions = total_transitions - successful_transitions;

        let duration_ms = if self.events.is_empty() {
            0
        } else {
            self.events.last().unwrap().timestamp - self.events.first().unwrap().timestamp
        };

        let mut states_visited: Vec<ProtocolState> = Vec::new();
        for ev in &self.events {
            if !states_visited.iter().any(|s| s == &ev.state_before) {
                states_visited.push(ev.state_before.clone());
            }
            if !states_visited.iter().any(|s| s == &ev.state_after) {
                states_visited.push(ev.state_after.clone());
            }
        }

        AuditSummary {
            total_events,
            total_transitions,
            successful_transitions,
            failed_transitions,
            duration_ms,
            states_visited,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// ProtocolMetrics
// ─────────────────────────────────────────────────────────────

pub struct ProtocolMetrics {
    state_durations: HashMap<String, Vec<u64>>,
    transition_counts: HashMap<(String, String), usize>,
    error_counts: HashMap<String, usize>,
}

impl ProtocolMetrics {
    pub fn new() -> Self {
        Self {
            state_durations: HashMap::new(),
            transition_counts: HashMap::new(),
            error_counts: HashMap::new(),
        }
    }

    pub fn record_state_duration(&mut self, state: &ProtocolState, ms: u64) {
        self.state_durations
            .entry(state.key())
            .or_insert_with(Vec::new)
            .push(ms);
    }

    pub fn record_transition(&mut self, from: &ProtocolState, to: &ProtocolState) {
        let key = (from.key(), to.key());
        *self.transition_counts.entry(key).or_insert(0) += 1;
    }

    pub fn record_error(&mut self, error: &str) {
        *self.error_counts.entry(error.to_string()).or_insert(0) += 1;
    }

    pub fn average_duration(&self, state: &ProtocolState) -> Option<f64> {
        let key = state.key();
        self.state_durations.get(&key).and_then(|durations| {
            if durations.is_empty() {
                None
            } else {
                let sum: u64 = durations.iter().sum();
                Some(sum as f64 / durations.len() as f64)
            }
        })
    }

    pub fn total_transitions(&self) -> usize {
        self.transition_counts.values().sum()
    }

    pub fn most_common_error(&self) -> Option<&str> {
        self.error_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(name, _)| name.as_str())
    }

    /// Produce a JSON representation of the collected metrics.
    pub fn to_json(&self) -> String {
        let mut parts = Vec::new();

        // state_durations
        let mut dur_parts = Vec::new();
        for (state, durs) in &self.state_durations {
            let vals: Vec<String> = durs.iter().map(|d| d.to_string()).collect();
            dur_parts.push(format!("\"{}\":[{}]", state, vals.join(",")));
        }
        parts.push(format!("\"state_durations\":{{{}}}", dur_parts.join(",")));

        // transition_counts
        let mut tc_parts = Vec::new();
        for ((from, to), count) in &self.transition_counts {
            tc_parts.push(format!("\"{}->{}\":{}", from, to, count));
        }
        parts.push(format!("\"transition_counts\":{{{}}}", tc_parts.join(",")));

        // error_counts
        let mut ec_parts = Vec::new();
        for (err, count) in &self.error_counts {
            ec_parts.push(format!("\"{}\":{}", err, count));
        }
        parts.push(format!("\"error_counts\":{{{}}}", ec_parts.join(",")));

        format!("{{{}}}", parts.join(","))
    }
}

// ─────────────────────────────────────────────────────────────
// ProtocolCheckpoint
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProtocolCheckpoint {
    pub state: ProtocolState,
    pub history: Vec<ProtocolEvent>,
    pub committed_values: HashMap<String, Vec<u8>>,
    pub timestamp: u64,
    pub checksum: [u8; 32],
}

impl ProtocolCheckpoint {
    /// Snapshot the current state of a running state machine.
    pub fn create(machine: &ProtocolStateMachine) -> Self {
        let state = machine.current_state.clone();
        let history = machine.history.clone();
        let committed_values = machine.committed_values.clone();
        let timestamp = current_time_ms();

        let payload = Self::compute_payload(&state, &history, &committed_values, timestamp);
        let checksum: [u8; 32] = *blake3::hash(&payload).as_bytes();

        Self {
            state,
            history,
            committed_values,
            timestamp,
            checksum,
        }
    }

    /// Restore a `ProtocolStateMachine` from a previously taken checkpoint.
    pub fn restore(checkpoint: Self) -> Result<ProtocolStateMachine, ProtocolError> {
        if !checkpoint.verify_integrity() {
            return Err(ProtocolError::SerializationError(
                "Checkpoint integrity check failed".to_string(),
            ));
        }

        let config = ProtocolConfig::default();
        let timeouts = config.timeouts.clone();

        Ok(ProtocolStateMachine {
            current_state: checkpoint.state,
            history: checkpoint.history,
            transition_rules: ProtocolStateMachine::default_transition_rules(),
            timeouts,
            start_time: checkpoint.timestamp,
            state_entry_time: checkpoint.timestamp,
            config,
            committed_values: checkpoint.committed_values,
            revealed_values: HashMap::new(),
        })
    }

    /// Verify that the checksum matches the checkpoint's content.
    pub fn verify_integrity(&self) -> bool {
        let payload =
            Self::compute_payload(&self.state, &self.history, &self.committed_values, self.timestamp);
        let expected: [u8; 32] = *blake3::hash(&payload).as_bytes();
        expected == self.checksum
    }

    pub fn serialize(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    pub fn deserialize(bytes: &[u8]) -> Result<Self, ProtocolError> {
        serde_json::from_slice(bytes)
            .map_err(|e| ProtocolError::SerializationError(e.to_string()))
    }

    fn compute_payload(
        state: &ProtocolState,
        history: &[ProtocolEvent],
        committed: &HashMap<String, Vec<u8>>,
        timestamp: u64,
    ) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(format!("{}", state).as_bytes());
        data.extend_from_slice(&timestamp.to_le_bytes());
        for ev in history {
            data.extend_from_slice(format!("{}{}{}", ev.from_state, ev.to_state, ev.timestamp).as_bytes());
        }
        let mut keys: Vec<&String> = committed.keys().collect();
        keys.sort();
        for k in keys {
            data.extend_from_slice(k.as_bytes());
            data.extend_from_slice(&committed[k]);
        }
        data
    }
}

// ─────────────────────────────────────────────────────────────
// SimulationStep / SimulationResult / ProtocolSimulator
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub enum SimulationStep {
    Transition(ProtocolState),
    Commit(String, Vec<u8>),
    Reveal(String, Vec<u8>),
    Wait(u64),
    InjectError(ProtocolError),
}

#[derive(Clone, Debug)]
pub struct SimulationResult {
    pub final_state: ProtocolState,
    pub steps_completed: usize,
    pub total_time_ms: u64,
    pub errors: Vec<ProtocolError>,
    pub events: Vec<ProtocolEvent>,
}

pub struct ProtocolSimulator {
    machine: ProtocolStateMachine,
    script: Vec<SimulationStep>,
}

impl ProtocolSimulator {
    pub fn new(config: ProtocolConfig) -> Self {
        Self {
            machine: ProtocolStateMachine::new(config),
            script: Vec::new(),
        }
    }

    pub fn add_step(&mut self, step: SimulationStep) {
        self.script.push(step);
    }

    /// Execute every step in the script in order.
    pub fn run_simulation(&mut self) -> SimulationResult {
        let start = current_time_ms();
        let mut errors: Vec<ProtocolError> = Vec::new();
        let mut steps_completed: usize = 0;

        for step in self.script.clone() {
            match step {
                SimulationStep::Transition(target) => {
                    match self.machine.transition_to(target) {
                        Ok(_) => {}
                        Err(e) => errors.push(e),
                    }
                }
                SimulationStep::Commit(key, value) => {
                    self.machine.store_commitment(key, value);
                }
                SimulationStep::Reveal(key, value) => {
                    self.machine.store_reveal(key, value);
                }
                SimulationStep::Wait(_ms) => {
                    // In simulation we do not actually sleep; just account for it.
                }
                SimulationStep::InjectError(err) => {
                    errors.push(err);
                }
            }
            steps_completed += 1;
        }

        let total_time_ms = current_time_ms().saturating_sub(start);

        SimulationResult {
            final_state: self.machine.current_state().clone(),
            steps_completed,
            total_time_ms,
            errors,
            events: self.machine.event_log().to_vec(),
        }
    }

    /// Run the simulation but inject an `InvalidState` error at each of the
    /// given step indices.
    pub fn run_with_failures(&mut self, failure_points: &[usize]) -> SimulationResult {
        let start = current_time_ms();
        let mut errors: Vec<ProtocolError> = Vec::new();
        let mut steps_completed: usize = 0;

        for (i, step) in self.script.clone().iter().enumerate() {
            if failure_points.contains(&i) {
                errors.push(ProtocolError::InvalidState(format!(
                    "Injected failure at step {}",
                    i
                )));
                steps_completed += 1;
                continue;
            }

            match step {
                SimulationStep::Transition(target) => {
                    match self.machine.transition_to(target.clone()) {
                        Ok(_) => {}
                        Err(e) => errors.push(e),
                    }
                }
                SimulationStep::Commit(key, value) => {
                    self.machine.store_commitment(key.clone(), value.clone());
                }
                SimulationStep::Reveal(key, value) => {
                    self.machine.store_reveal(key.clone(), value.clone());
                }
                SimulationStep::Wait(_) => {}
                SimulationStep::InjectError(err) => {
                    errors.push(err.clone());
                }
            }
            steps_completed += 1;
        }

        let total_time_ms = current_time_ms().saturating_sub(start);

        SimulationResult {
            final_state: self.machine.current_state().clone(),
            steps_completed,
            total_time_ms,
            errors,
            events: self.machine.event_log().to_vec(),
        }
    }

    /// Run the simulation, recording expected delay values for specified states.
    /// The delays are purely logical and do not cause real waiting.
    pub fn run_with_delays(&mut self, delays: &[(ProtocolState, u64)]) -> SimulationResult {
        let start = current_time_ms();
        let mut errors: Vec<ProtocolError> = Vec::new();
        let mut steps_completed: usize = 0;
        let mut accumulated_delay: u64 = 0;

        for step in self.script.clone() {
            // Check if the current state has a configured delay.
            for (ds, dms) in delays {
                if self.machine.current_state() == ds {
                    accumulated_delay += dms;
                }
            }

            match step {
                SimulationStep::Transition(target) => {
                    match self.machine.transition_to(target) {
                        Ok(_) => {}
                        Err(e) => errors.push(e),
                    }
                }
                SimulationStep::Commit(key, value) => {
                    self.machine.store_commitment(key, value);
                }
                SimulationStep::Reveal(key, value) => {
                    self.machine.store_reveal(key, value);
                }
                SimulationStep::Wait(ms) => {
                    accumulated_delay += ms;
                }
                SimulationStep::InjectError(err) => {
                    errors.push(err);
                }
            }
            steps_completed += 1;
        }

        let wall_time = current_time_ms().saturating_sub(start);

        SimulationResult {
            final_state: self.machine.current_state().clone(),
            steps_completed,
            total_time_ms: wall_time + accumulated_delay,
            errors,
            events: self.machine.event_log().to_vec(),
        }
    }
}

// ─────────────────────────────────────────────────────────────
// StateGraph
// ─────────────────────────────────────────────────────────────

pub struct StateGraph {
    nodes: Vec<ProtocolState>,
    edges: Vec<(usize, usize, String)>,
}

impl StateGraph {
    /// Build a graph from transition rules. Each unique state becomes a node;
    /// each rule becomes a directed edge labelled with the rule's condition.
    pub fn from_rules(rules: &[StateTransitionRule]) -> Self {
        let mut nodes: Vec<ProtocolState> = Vec::new();
        let mut edges: Vec<(usize, usize, String)> = Vec::new();

        let node_index = |state: &ProtocolState, ns: &mut Vec<ProtocolState>| -> usize {
            if let Some(pos) = ns.iter().position(|n| n == state) {
                pos
            } else {
                ns.push(state.clone());
                ns.len() - 1
            }
        };

        for rule in rules {
            let from_idx = node_index(&rule.from, &mut nodes);
            let to_idx = node_index(&rule.to, &mut nodes);
            edges.push((from_idx, to_idx, rule.condition.clone()));
        }

        Self { nodes, edges }
    }

    /// All states reachable from `state` via one or more transitions (BFS).
    pub fn reachable_from(&self, state: &ProtocolState) -> Vec<ProtocolState> {
        let start = match self.nodes.iter().position(|n| n == state) {
            Some(i) => i,
            None => return Vec::new(),
        };

        let mut visited = vec![false; self.nodes.len()];
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        visited[start] = true;

        while let Some(current) = queue.pop_front() {
            for &(from, to, _) in &self.edges {
                if from == current && !visited[to] {
                    visited[to] = true;
                    queue.push_back(to);
                }
            }
        }

        visited
            .iter()
            .enumerate()
            .filter(|&(i, &v)| v && i != start)
            .map(|(i, _)| self.nodes[i].clone())
            .collect()
    }

    /// BFS shortest path returning the sequence of states from `from` to `to`
    /// (inclusive). Returns `None` if no path exists.
    pub fn shortest_path(
        &self,
        from: &ProtocolState,
        to: &ProtocolState,
    ) -> Option<Vec<ProtocolState>> {
        let start = self.nodes.iter().position(|n| n == from)?;
        let end = self.nodes.iter().position(|n| n == to)?;

        if start == end {
            return Some(vec![from.clone()]);
        }

        let n = self.nodes.len();
        let mut visited = vec![false; n];
        let mut parent: Vec<Option<usize>> = vec![None; n];
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        visited[start] = true;

        while let Some(current) = queue.pop_front() {
            if current == end {
                // reconstruct path
                let mut path = Vec::new();
                let mut c = end;
                while let Some(p) = parent[c] {
                    path.push(self.nodes[c].clone());
                    c = p;
                }
                path.push(self.nodes[start].clone());
                path.reverse();
                return Some(path);
            }

            for &(f, t, _) in &self.edges {
                if f == current && !visited[t] {
                    visited[t] = true;
                    parent[t] = Some(current);
                    queue.push_back(t);
                }
            }
        }

        None
    }

    /// A state graph is deadlock-free if every non-terminal node has at least
    /// one outgoing edge.
    pub fn is_deadlock_free(&self) -> bool {
        for (i, state) in self.nodes.iter().enumerate() {
            let is_terminal = matches!(
                state,
                ProtocolState::Completed | ProtocolState::Aborted(_) | ProtocolState::TimedOut
            );
            if is_terminal {
                continue;
            }
            let has_outgoing = self.edges.iter().any(|&(from, _, _)| from == i);
            if !has_outgoing {
                return false;
            }
        }
        true
    }

    /// Render the graph in Graphviz DOT format.
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph ProtocolStateGraph {\n");
        dot.push_str("    rankdir=LR;\n");
        for (i, state) in self.nodes.iter().enumerate() {
            let label = format!("{}", state);
            let shape = if matches!(
                state,
                ProtocolState::Completed | ProtocolState::Aborted(_) | ProtocolState::TimedOut
            ) {
                "doublecircle"
            } else {
                "circle"
            };
            dot.push_str(&format!(
                "    {} [label=\"{}\" shape={}];\n",
                i, label, shape
            ));
        }
        for (from, to, label) in &self.edges {
            dot.push_str(&format!(
                "    {} -> {} [label=\"{}\"];\n",
                from, to, label
            ));
        }
        dot.push_str("}\n");
        dot
    }
}

// ─────────────────────────────────────────────────────────────
// ProtocolRunner
// ─────────────────────────────────────────────────────────────

/// Executes a complete protocol sequence by dispatching registered handler
/// functions keyed by state.
pub struct ProtocolRunner {
    machine: ProtocolStateMachine,
    handlers: HashMap<String, Box<dyn Fn(&mut ProtocolStateMachine) -> Result<(), ProtocolError>>>,
}

impl ProtocolRunner {
    /// Create a new runner with a fresh state machine and no handlers.
    pub fn new(config: ProtocolConfig) -> Self {
        Self {
            machine: ProtocolStateMachine::new(config),
            handlers: HashMap::new(),
        }
    }

    /// Register a handler for a given state. The handler is keyed by
    /// `state.key()` so it can be looked up when the machine enters that state.
    pub fn register_handler(
        &mut self,
        state: ProtocolState,
        handler: impl Fn(&mut ProtocolStateMachine) -> Result<(), ProtocolError> + 'static,
    ) {
        self.handlers.insert(state.key(), Box::new(handler));
    }

    /// Run handlers in a loop until a terminal state is reached or 100
    /// iterations elapse. Returns the final state or the first error.
    pub fn run_to_completion(&mut self) -> Result<ProtocolState, ProtocolError> {
        let max_iterations = 100;
        for _ in 0..max_iterations {
            if self.machine.is_terminal() {
                return Ok(self.machine.current_state().clone());
            }
            let key = self.machine.current_state().key();
            match self.handlers.get(&key) {
                Some(handler) => {
                    // We need to call the handler with &mut self.machine, but
                    // the borrow checker won't allow it while `handler` borrows
                    // `self.handlers`.  Work around by extracting a raw pointer.
                    let handler_ptr = handler.as_ref() as *const _;
                    let handler_ref: &dyn Fn(&mut ProtocolStateMachine) -> Result<(), ProtocolError> =
                        unsafe { &*handler_ptr };
                    handler_ref(&mut self.machine)?;
                }
                None => {
                    return Err(ProtocolError::InvalidState(format!(
                        "No handler registered for state {}",
                        key
                    )));
                }
            }
        }
        Ok(self.machine.current_state().clone())
    }

    /// Run a single handler for the current state.
    pub fn run_single_step(&mut self) -> Result<ProtocolState, ProtocolError> {
        if self.machine.is_terminal() {
            return Ok(self.machine.current_state().clone());
        }
        let key = self.machine.current_state().key();
        match self.handlers.get(&key) {
            Some(handler) => {
                let handler_ptr = handler.as_ref() as *const _;
                let handler_ref: &dyn Fn(&mut ProtocolStateMachine) -> Result<(), ProtocolError> =
                    unsafe { &*handler_ptr };
                handler_ref(&mut self.machine)?;
            }
            None => {
                return Err(ProtocolError::InvalidState(format!(
                    "No handler registered for state {}",
                    key
                )));
            }
        }
        Ok(self.machine.current_state().clone())
    }

    /// Returns a reference to the current protocol state.
    pub fn current_state(&self) -> &ProtocolState {
        self.machine.current_state()
    }

    /// Returns the number of events recorded so far.
    pub fn event_count(&self) -> usize {
        self.machine.event_log().len()
    }
}

// ─────────────────────────────────────────────────────────────
// ProtocolTemplate
// ─────────────────────────────────────────────────────────────

/// A reusable protocol definition that can be instantiated into a live
/// state machine.
pub struct ProtocolTemplate {
    pub name: String,
    pub states: Vec<ProtocolState>,
    pub transitions: Vec<StateTransitionRule>,
    pub default_timeouts: HashMap<String, u64>,
}

impl ProtocolTemplate {
    /// Full evaluation protocol: Initialized → CommitOutputs → RevealBenchmark
    /// → Evaluate → Prove → Verify → Certify → Completed.
    pub fn evaluation_protocol() -> Self {
        let states = vec![
            ProtocolState::Initialized,
            ProtocolState::CommitOutputs,
            ProtocolState::RevealBenchmark,
            ProtocolState::Evaluate,
            ProtocolState::Prove,
            ProtocolState::Verify,
            ProtocolState::Certify,
            ProtocolState::Completed,
        ];
        let transitions = vec![
            StateTransitionRule::new(ProtocolState::Initialized, ProtocolState::CommitOutputs, "begin_commit", 30_000),
            StateTransitionRule::new(ProtocolState::CommitOutputs, ProtocolState::RevealBenchmark, "outputs_committed", 60_000),
            StateTransitionRule::new(ProtocolState::RevealBenchmark, ProtocolState::Evaluate, "benchmark_revealed", 60_000),
            StateTransitionRule::new(ProtocolState::Evaluate, ProtocolState::Prove, "evaluation_complete", 120_000),
            StateTransitionRule::new(ProtocolState::Prove, ProtocolState::Verify, "proof_generated", 300_000),
            StateTransitionRule::new(ProtocolState::Verify, ProtocolState::Certify, "proof_verified", 60_000),
            StateTransitionRule::new(ProtocolState::Certify, ProtocolState::Completed, "certificate_issued", 30_000),
        ];
        let mut timeouts = HashMap::new();
        timeouts.insert("Initialized".to_string(), 30_000);
        timeouts.insert("CommitOutputs".to_string(), 60_000);
        timeouts.insert("RevealBenchmark".to_string(), 60_000);
        timeouts.insert("Evaluate".to_string(), 120_000);
        timeouts.insert("Prove".to_string(), 300_000);
        timeouts.insert("Verify".to_string(), 60_000);
        timeouts.insert("Certify".to_string(), 30_000);
        Self {
            name: "evaluation-protocol".to_string(),
            states,
            transitions,
            default_timeouts: timeouts,
        }
    }

    /// Certification-only protocol: Initialized → Verify → Certify → Completed.
    pub fn certification_protocol() -> Self {
        let states = vec![
            ProtocolState::Initialized,
            ProtocolState::Verify,
            ProtocolState::Certify,
            ProtocolState::Completed,
        ];
        let transitions = vec![
            StateTransitionRule::new(ProtocolState::Initialized, ProtocolState::Verify, "begin_verify", 30_000),
            StateTransitionRule::new(ProtocolState::Verify, ProtocolState::Certify, "proof_verified", 60_000),
            StateTransitionRule::new(ProtocolState::Certify, ProtocolState::Completed, "certificate_issued", 30_000),
        ];
        let mut timeouts = HashMap::new();
        timeouts.insert("Initialized".to_string(), 30_000);
        timeouts.insert("Verify".to_string(), 60_000);
        timeouts.insert("Certify".to_string(), 30_000);
        Self {
            name: "certification-protocol".to_string(),
            states,
            transitions,
            default_timeouts: timeouts,
        }
    }

    /// Verification-only protocol: Initialized → Verify → Completed.
    pub fn verification_only() -> Self {
        let states = vec![
            ProtocolState::Initialized,
            ProtocolState::Verify,
            ProtocolState::Completed,
        ];
        let transitions = vec![
            StateTransitionRule::new(ProtocolState::Initialized, ProtocolState::Verify, "begin_verify", 30_000),
            StateTransitionRule::new(ProtocolState::Verify, ProtocolState::Completed, "verification_complete", 60_000),
        ];
        let mut timeouts = HashMap::new();
        timeouts.insert("Initialized".to_string(), 30_000);
        timeouts.insert("Verify".to_string(), 60_000);
        Self {
            name: "verification-only".to_string(),
            states,
            transitions,
            default_timeouts: timeouts,
        }
    }

    /// Convert this template to a `ProtocolConfig`.
    pub fn to_config(&self) -> ProtocolConfig {
        ProtocolConfig {
            name: self.name.clone(),
            version: "0.1.0".to_string(),
            timeouts: self.default_timeouts.clone(),
            enable_logging: true,
            max_retries: 3,
            require_grinding: false,
        }
    }

    /// Instantiate a live `ProtocolStateMachine` from this template.
    pub fn instantiate(&self) -> ProtocolStateMachine {
        ProtocolStateMachine::new(self.to_config())
    }
}

// ─────────────────────────────────────────────────────────────
// StateHistory
// ─────────────────────────────────────────────────────────────

/// Queryable analysis of a protocol's event history.
pub struct StateHistory {
    events: Vec<ProtocolEvent>,
}

impl StateHistory {
    /// Create from a pre-existing event vector.
    pub fn new(events: Vec<ProtocolEvent>) -> Self {
        Self { events }
    }

    /// Clone the event log from a live state machine.
    pub fn from_machine(machine: &ProtocolStateMachine) -> Self {
        Self {
            events: machine.event_log().to_vec(),
        }
    }

    /// Total time spent in `state`, calculated by summing durations between
    /// consecutive events where the machine was in that state.
    pub fn time_in_state(&self, state: &ProtocolState) -> u64 {
        let mut total: u64 = 0;
        for i in 0..self.events.len() {
            if self.events[i].to_state == *state {
                // Find the next event to compute the duration in this state.
                if let Some(next) = self.events.get(i + 1) {
                    total += next.timestamp.saturating_sub(self.events[i].timestamp);
                }
            }
        }
        total
    }

    /// Count events that transition from `from` to `to`.
    pub fn transition_count(&self, from: &ProtocolState, to: &ProtocolState) -> usize {
        self.events
            .iter()
            .filter(|e| e.from_state == *from && e.to_state == *to)
            .count()
    }

    /// Unique states that appear in the event history (both `from` and `to`
    /// fields).
    pub fn states_visited(&self) -> Vec<ProtocolState> {
        let mut seen = Vec::<ProtocolState>::new();
        for event in &self.events {
            if !seen.contains(&event.from_state) {
                seen.push(event.from_state.clone());
            }
            if !seen.contains(&event.to_state) {
                seen.push(event.to_state.clone());
            }
        }
        seen
    }

    /// Which state had the longest total residence time.
    pub fn longest_state(&self) -> Option<(ProtocolState, u64)> {
        let visited = self.states_visited();
        if visited.is_empty() {
            return None;
        }
        let mut best_state = visited[0].clone();
        let mut best_time: u64 = 0;
        for state in &visited {
            let t = self.time_in_state(state);
            if t > best_time {
                best_time = t;
                best_state = state.clone();
            }
        }
        Some((best_state, best_time))
    }

    /// Timeline of `(timestamp, to_state)` pairs in chronological order.
    pub fn timeline(&self) -> Vec<(u64, ProtocolState)> {
        self.events
            .iter()
            .map(|e| (e.timestamp, e.to_state.clone()))
            .collect()
    }

    /// Export the history as CSV text.
    pub fn to_csv(&self) -> String {
        let mut out = String::from("timestamp,from,to,event_type,details\n");
        for e in &self.events {
            out.push_str(&format!(
                "{},{},{},{},{}\n",
                e.timestamp, e.from_state, e.to_state, e.event_type, e.details
            ));
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sm() -> ProtocolStateMachine {
        ProtocolStateMachine::new(ProtocolConfig::default())
    }

    // ── Basic construction ──

    #[test]
    fn test_initial_state() {
        let sm = make_sm();
        assert_eq!(*sm.current_state(), ProtocolState::Initialized);
        assert!(!sm.is_terminal());
    }

    #[test]
    fn test_default_config() {
        let cfg = ProtocolConfig::default();
        assert_eq!(cfg.name, "spectacles-protocol");
        assert!(cfg.enable_logging);
        assert_eq!(cfg.max_retries, 3);
    }

    // ── Transition rules ──

    #[test]
    fn test_valid_first_transition() {
        let mut sm = make_sm();
        assert!(sm.can_transition_to(&ProtocolState::CommitOutputs));
        assert!(sm.transition_to(ProtocolState::CommitOutputs).is_ok());
        assert_eq!(*sm.current_state(), ProtocolState::CommitOutputs);
    }

    #[test]
    fn test_invalid_transition() {
        let mut sm = make_sm();
        // Cannot jump from Initialized to Prove
        assert!(!sm.can_transition_to(&ProtocolState::Prove));
        let result = sm.transition_to(ProtocolState::Prove);
        assert!(result.is_err());
        match result {
            Err(ProtocolError::InvalidTransition(from, to)) => {
                assert_eq!(from, ProtocolState::Initialized);
                assert_eq!(to, ProtocolState::Prove);
            }
            _ => panic!("expected InvalidTransition error"),
        }
    }

    #[test]
    fn test_full_protocol_sequence() {
        let mut sm = make_sm();

        let sequence = vec![
            ProtocolState::CommitOutputs,
            ProtocolState::RevealBenchmark,
            ProtocolState::Evaluate,
            ProtocolState::Prove,
            ProtocolState::Verify,
            ProtocolState::Certify,
            ProtocolState::Completed,
        ];

        for target in sequence {
            assert!(sm.transition_to(target.clone()).is_ok());
        }
        assert!(sm.is_terminal());
        assert_eq!(*sm.current_state(), ProtocolState::Completed);
    }

    #[test]
    fn test_valid_transitions_from_initialized() {
        let sm = make_sm();
        let valid = sm.valid_transitions();
        assert!(valid.contains(&ProtocolState::CommitOutputs));
        assert!(valid.contains(&ProtocolState::TimedOut));
        // Should contain an Aborted variant
        assert!(valid.iter().any(|s| matches!(s, ProtocolState::Aborted(_))));
    }

    // ── Abort ──

    #[test]
    fn test_abort_from_any_state() {
        let mut sm = make_sm();
        sm.transition_to(ProtocolState::CommitOutputs).unwrap();
        sm.abort(AbortReason::ConstraintViolation);
        assert!(sm.is_terminal());
        assert_eq!(
            *sm.current_state(),
            ProtocolState::Aborted(AbortReason::ConstraintViolation)
        );
    }

    #[test]
    fn test_abort_with_external_reason() {
        let mut sm = make_sm();
        sm.abort(AbortReason::ExternalAbort("test reason".to_string()));
        assert!(sm.is_terminal());
    }

    #[test]
    fn test_cannot_transition_after_abort() {
        let mut sm = make_sm();
        sm.abort(AbortReason::ProofFailed);
        let result = sm.transition_to(ProtocolState::CommitOutputs);
        assert_eq!(result, Err(ProtocolError::AlreadyTerminal));
    }

    #[test]
    fn test_cannot_transition_after_completed() {
        let mut sm = make_sm();
        sm.transition_to(ProtocolState::CommitOutputs).unwrap();
        sm.transition_to(ProtocolState::RevealBenchmark).unwrap();
        sm.transition_to(ProtocolState::Evaluate).unwrap();
        sm.transition_to(ProtocolState::Prove).unwrap();
        sm.transition_to(ProtocolState::Verify).unwrap();
        sm.transition_to(ProtocolState::Certify).unwrap();
        sm.transition_to(ProtocolState::Completed).unwrap();

        let result = sm.transition_to(ProtocolState::Initialized);
        assert_eq!(result, Err(ProtocolError::AlreadyTerminal));
    }

    // ── Abort idempotence ──

    #[test]
    fn test_abort_when_already_terminal_is_noop() {
        let mut sm = make_sm();
        sm.abort(AbortReason::ProofFailed);
        let state_before = sm.current_state().clone();
        sm.abort(AbortReason::ConstraintViolation);
        assert_eq!(*sm.current_state(), state_before);
    }

    // ── Event log ──

    #[test]
    fn test_event_log_populated() {
        let mut sm = make_sm();
        sm.transition_to(ProtocolState::CommitOutputs).unwrap();
        sm.transition_to(ProtocolState::RevealBenchmark).unwrap();
        assert!(sm.event_log().len() >= 2);
    }

    #[test]
    fn test_event_log_records_abort() {
        let mut sm = make_sm();
        sm.abort(AbortReason::TimeoutExceeded);
        let last = sm.event_log().last().unwrap();
        assert_eq!(last.event_type, EventType::Error);
    }

    // ── Commitment / Reveal ──

    #[test]
    fn test_commitment_reveal_cycle() {
        let mut sm = make_sm();
        let data = b"hello world".to_vec();
        sm.store_commitment("k1".to_string(), data.clone());
        sm.store_reveal("k1".to_string(), data);
        assert!(sm.verify_commitment_reveal("k1"));
    }

    #[test]
    fn test_commitment_reveal_mismatch() {
        let mut sm = make_sm();
        sm.store_commitment("k1".to_string(), b"one".to_vec());
        sm.store_reveal("k1".to_string(), b"two".to_vec());
        assert!(!sm.verify_commitment_reveal("k1"));
    }

    #[test]
    fn test_commitment_missing_key() {
        let sm = make_sm();
        assert!(!sm.verify_commitment_reveal("nonexistent"));
    }

    // ── Serialization ──

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let mut sm = make_sm();
        sm.transition_to(ProtocolState::CommitOutputs).unwrap();
        sm.store_commitment("x".to_string(), b"data".to_vec());

        let bytes = sm.serialize_state();
        let sm2 = ProtocolStateMachine::deserialize_state(&bytes).unwrap();
        assert_eq!(*sm2.current_state(), ProtocolState::CommitOutputs);
    }

    #[test]
    fn test_deserialize_invalid_bytes() {
        let result = ProtocolStateMachine::deserialize_state(b"not json");
        assert!(result.is_err());
    }

    // ── Reset ──

    #[test]
    fn test_reset() {
        let mut sm = make_sm();
        sm.transition_to(ProtocolState::CommitOutputs).unwrap();
        sm.store_commitment("k".to_string(), b"v".to_vec());
        sm.reset();
        assert_eq!(*sm.current_state(), ProtocolState::Initialized);
        assert!(sm.event_log().is_empty());
        assert!(!sm.verify_commitment_reveal("k"));
    }

    // ── Timeout ──

    #[test]
    fn test_check_timeout_no_timeout_immediately() {
        let mut sm = make_sm();
        // Immediately after creation the timeout should not have expired
        assert!(!sm.check_timeout());
    }

    #[test]
    fn test_timeout_on_terminal_is_false() {
        let mut sm = make_sm();
        sm.abort(AbortReason::ProofFailed);
        assert!(!sm.check_timeout());
    }

    // ── Elapsed ──

    #[test]
    fn test_elapsed_in_state() {
        let sm = make_sm();
        let elapsed = sm.elapsed_in_state();
        // Should be very small (just created)
        assert!(elapsed < 1000);
    }

    #[test]
    fn test_total_elapsed() {
        let sm = make_sm();
        let total = sm.total_elapsed();
        assert!(total < 1000);
    }

    // ── run_protocol ──

    #[test]
    fn test_run_protocol_happy_path() {
        let mut sm = make_sm();
        let result = sm.run_protocol(|state| {
            match state {
                ProtocolState::Initialized => Ok(ProtocolState::CommitOutputs),
                ProtocolState::CommitOutputs => Ok(ProtocolState::RevealBenchmark),
                ProtocolState::RevealBenchmark => Ok(ProtocolState::Evaluate),
                ProtocolState::Evaluate => Ok(ProtocolState::Prove),
                ProtocolState::Prove => Ok(ProtocolState::Verify),
                ProtocolState::Verify => Ok(ProtocolState::Certify),
                ProtocolState::Certify => Ok(ProtocolState::Completed),
                _ => Err(ProtocolError::InvalidState("unexpected".to_string())),
            }
        });
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ProtocolState::Completed);
    }

    #[test]
    fn test_run_protocol_error_propagation() {
        let mut sm = make_sm();
        let result = sm.run_protocol(|_state| {
            Err(ProtocolError::InvalidState("forced failure".to_string()))
        });
        assert!(result.is_err());
    }

    // ── Display impls ──

    #[test]
    fn test_state_display() {
        assert_eq!(format!("{}", ProtocolState::Initialized), "Initialized");
        assert_eq!(
            format!("{}", ProtocolState::Aborted(AbortReason::ProofFailed)),
            "Aborted(ProofFailed)"
        );
    }

    #[test]
    fn test_error_display() {
        let err = ProtocolError::InvalidTransition(
            ProtocolState::Initialized,
            ProtocolState::Prove,
        );
        let msg = format!("{}", err);
        assert!(msg.contains("Invalid transition"));
    }

    #[test]
    fn test_event_type_display() {
        assert_eq!(format!("{}", EventType::StateTransition), "StateTransition");
    }

    // ── ProtocolState key ──

    #[test]
    fn test_state_key() {
        assert_eq!(ProtocolState::Initialized.key(), "Initialized");
        assert_eq!(
            ProtocolState::Aborted(AbortReason::ProofFailed).key(),
            "Aborted(ProofFailed)"
        );
    }

    // ── Transition to abort via transition_to ──

    #[test]
    fn test_transition_to_abort() {
        let mut sm = make_sm();
        sm.transition_to(ProtocolState::CommitOutputs).unwrap();
        let result = sm.transition_to(ProtocolState::Aborted(AbortReason::CommitmentMismatch));
        assert!(result.is_ok());
        assert!(sm.is_terminal());
    }

    // ── Transition to TimedOut ──

    #[test]
    fn test_transition_to_timed_out() {
        let mut sm = make_sm();
        assert!(sm.can_transition_to(&ProtocolState::TimedOut));
        sm.transition_to(ProtocolState::TimedOut).unwrap();
        assert!(sm.is_terminal());
    }

    // ── Multiple commitments ──

    #[test]
    fn test_multiple_commitments() {
        let mut sm = make_sm();
        sm.store_commitment("a".to_string(), b"alpha".to_vec());
        sm.store_commitment("b".to_string(), b"beta".to_vec());
        sm.store_reveal("a".to_string(), b"alpha".to_vec());
        sm.store_reveal("b".to_string(), b"beta".to_vec());
        assert!(sm.verify_commitment_reveal("a"));
        assert!(sm.verify_commitment_reveal("b"));
    }

    // ── ProtocolEvent construction ──

    #[test]
    fn test_protocol_event_new() {
        let ev = ProtocolEvent::new(
            ProtocolState::Initialized,
            ProtocolState::CommitOutputs,
            12345,
            EventType::StateTransition,
            "test".to_string(),
        );
        assert_eq!(ev.timestamp, 12345);
        assert_eq!(ev.event_type, EventType::StateTransition);
        assert_eq!(ev.details, "test");
    }

    // ── StateTransitionRule ──

    #[test]
    fn test_state_transition_rule() {
        let rule = StateTransitionRule::new(
            ProtocolState::Initialized,
            ProtocolState::CommitOutputs,
            "begin",
            5000,
        );
        assert_eq!(rule.from, ProtocolState::Initialized);
        assert_eq!(rule.to, ProtocolState::CommitOutputs);
        assert_eq!(rule.condition, "begin");
        assert_eq!(rule.timeout_ms, 5000);
    }

    // ── Discriminant ordering ──

    #[test]
    fn test_discriminant_order() {
        assert!(ProtocolState::Initialized.discriminant_index() < ProtocolState::CommitOutputs.discriminant_index());
        assert!(ProtocolState::Certify.discriminant_index() < ProtocolState::Completed.discriminant_index());
    }

    // ── Config with custom timeouts ──

    #[test]
    fn test_custom_config_timeouts() {
        let mut cfg = ProtocolConfig::default();
        cfg.timeouts.insert("Initialized".to_string(), 1);
        let sm = ProtocolStateMachine::new(cfg);
        assert_eq!(*sm.current_state(), ProtocolState::Initialized);
    }

    // ── Logging disabled ──

    #[test]
    fn test_logging_disabled() {
        let mut cfg = ProtocolConfig::default();
        cfg.enable_logging = false;
        let mut sm = ProtocolStateMachine::new(cfg);
        sm.transition_to(ProtocolState::CommitOutputs).unwrap();
        assert!(sm.event_log().is_empty());
    }

    // ── Serialize roundtrip after full protocol ──

    #[test]
    fn test_serialize_after_full_protocol() {
        let mut sm = make_sm();
        for target in &[
            ProtocolState::CommitOutputs,
            ProtocolState::RevealBenchmark,
            ProtocolState::Evaluate,
            ProtocolState::Prove,
            ProtocolState::Verify,
            ProtocolState::Certify,
            ProtocolState::Completed,
        ] {
            sm.transition_to(target.clone()).unwrap();
        }
        let bytes = sm.serialize_state();
        let sm2 = ProtocolStateMachine::deserialize_state(&bytes).unwrap();
        assert_eq!(*sm2.current_state(), ProtocolState::Completed);
        assert_eq!(sm2.event_log().len(), sm.event_log().len());
    }

    // ═══════════════════════════════════════════════════════════
    //  ProtocolPhaseManager tests
    // ═══════════════════════════════════════════════════════════

    fn always_valid(_r: &PhaseResult) -> bool { true }
    fn never_valid(_r: &PhaseResult) -> bool { false }
    fn success_only(r: &PhaseResult) -> bool { r.success }

    fn sample_phases() -> Vec<ProtocolPhase> {
        vec![
            ProtocolPhase {
                name: "commit".to_string(),
                required_state: ProtocolState::Initialized,
                timeout_ms: 5000,
                validator: always_valid,
            },
            ProtocolPhase {
                name: "reveal".to_string(),
                required_state: ProtocolState::CommitOutputs,
                timeout_ms: 5000,
                validator: always_valid,
            },
            ProtocolPhase {
                name: "prove".to_string(),
                required_state: ProtocolState::Prove,
                timeout_ms: 10_000,
                validator: success_only,
            },
        ]
    }

    #[test]
    fn test_phase_manager_creation() {
        let mgr = ProtocolPhaseManager::new(sample_phases());
        assert_eq!(mgr.current_phase().name, "commit");
        assert!(!mgr.is_complete());
    }

    #[test]
    fn test_phase_manager_advance_happy() {
        let mut mgr = ProtocolPhaseManager::new(sample_phases());
        mgr.record_result("commit".to_string(), PhaseResult::ok(100));
        assert!(mgr.advance().is_ok());
        assert_eq!(mgr.current_phase().name, "reveal");
    }

    #[test]
    fn test_phase_manager_advance_no_result() {
        let mut mgr = ProtocolPhaseManager::new(sample_phases());
        let err = mgr.advance();
        assert!(err.is_err());
    }

    #[test]
    fn test_phase_manager_advance_failed_validation() {
        let phases = vec![ProtocolPhase {
            name: "strict".to_string(),
            required_state: ProtocolState::Initialized,
            timeout_ms: 1000,
            validator: never_valid,
        }];
        let mut mgr = ProtocolPhaseManager::new(phases);
        mgr.record_result("strict".to_string(), PhaseResult::ok(10));
        assert!(mgr.advance().is_err());
    }

    #[test]
    fn test_phase_manager_full_run() {
        let mut mgr = ProtocolPhaseManager::new(sample_phases());
        mgr.record_result("commit".to_string(), PhaseResult::ok(100));
        mgr.advance().unwrap();
        mgr.record_result("reveal".to_string(), PhaseResult::ok(200));
        mgr.advance().unwrap();
        mgr.record_result("prove".to_string(), PhaseResult::ok(300));
        mgr.advance().unwrap();
        assert!(mgr.is_complete());
    }

    #[test]
    fn test_phase_manager_advance_past_end() {
        let mut mgr = ProtocolPhaseManager::new(sample_phases());
        mgr.record_result("commit".to_string(), PhaseResult::ok(50));
        mgr.advance().unwrap();
        mgr.record_result("reveal".to_string(), PhaseResult::ok(50));
        mgr.advance().unwrap();
        mgr.record_result("prove".to_string(), PhaseResult::ok(50));
        mgr.advance().unwrap();
        assert!(mgr.advance().is_err()); // already complete
    }

    #[test]
    fn test_phase_durations() {
        let mut mgr = ProtocolPhaseManager::new(sample_phases());
        mgr.record_result("commit".to_string(), PhaseResult::ok(42));
        mgr.record_result("reveal".to_string(), PhaseResult::ok(84));
        let durs = mgr.phase_durations();
        assert_eq!(durs.len(), 2);
        assert_eq!(durs[0], ("commit".to_string(), 42));
        assert_eq!(durs[1], ("reveal".to_string(), 84));
    }

    #[test]
    fn test_phase_result_err() {
        let r = PhaseResult::err(10, "boom".to_string());
        assert!(!r.success);
        assert_eq!(r.error, Some("boom".to_string()));
    }

    #[test]
    fn test_phase_result_with_data() {
        let mut r = PhaseResult::ok(5);
        r.data.insert("key".to_string(), vec![1, 2, 3]);
        assert_eq!(r.data.get("key").unwrap(), &vec![1u8, 2, 3]);
    }

    #[test]
    fn test_phase_manager_success_only_validator_with_error_result() {
        let phases = vec![ProtocolPhase {
            name: "guarded".to_string(),
            required_state: ProtocolState::Initialized,
            timeout_ms: 1000,
            validator: success_only,
        }];
        let mut mgr = ProtocolPhaseManager::new(phases);
        mgr.record_result("guarded".to_string(), PhaseResult::err(10, "bad".to_string()));
        assert!(mgr.advance().is_err());
    }

    // ═══════════════════════════════════════════════════════════
    //  RetryManager tests
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_retry_manager_fixed() {
        let mut rm = RetryManager::new(3, BackoffStrategy::Fixed(100));
        let state = ProtocolState::Initialized;
        assert!(rm.should_retry(&state));
        assert_eq!(rm.delay_ms(&state), 100);
        rm.record_attempt(&state);
        assert_eq!(rm.delay_ms(&state), 100); // fixed stays the same
        rm.record_attempt(&state);
        rm.record_attempt(&state);
        assert!(!rm.should_retry(&state));
    }

    #[test]
    fn test_retry_manager_linear() {
        let mut rm = RetryManager::new(5, BackoffStrategy::Linear(50));
        let state = ProtocolState::CommitOutputs;
        assert_eq!(rm.delay_ms(&state), 50); // 50 * (0+1) = 50
        rm.record_attempt(&state);
        assert_eq!(rm.delay_ms(&state), 100); // 50 * (1+1) = 100
        rm.record_attempt(&state);
        assert_eq!(rm.delay_ms(&state), 150); // 50 * (2+1) = 150
    }

    #[test]
    fn test_retry_manager_exponential() {
        let mut rm = RetryManager::new(5, BackoffStrategy::Exponential(100, 2.0));
        let state = ProtocolState::Prove;
        assert_eq!(rm.delay_ms(&state), 100); // 100 * 2^0 = 100
        rm.record_attempt(&state);
        assert_eq!(rm.delay_ms(&state), 200); // 100 * 2^1 = 200
        rm.record_attempt(&state);
        assert_eq!(rm.delay_ms(&state), 400); // 100 * 2^2 = 400
    }

    #[test]
    fn test_retry_manager_fibonacci() {
        let mut rm = RetryManager::new(10, BackoffStrategy::Fibonacci);
        let state = ProtocolState::Verify;
        // fib(0)=1, fib(1)=1, fib(2)=2, fib(3)=3, fib(4)=5
        assert_eq!(rm.delay_ms(&state), 1);
        rm.record_attempt(&state);
        assert_eq!(rm.delay_ms(&state), 1);
        rm.record_attempt(&state);
        assert_eq!(rm.delay_ms(&state), 2);
        rm.record_attempt(&state);
        assert_eq!(rm.delay_ms(&state), 3);
        rm.record_attempt(&state);
        assert_eq!(rm.delay_ms(&state), 5);
    }

    #[test]
    fn test_retry_manager_reset() {
        let mut rm = RetryManager::new(2, BackoffStrategy::Fixed(10));
        let state = ProtocolState::Evaluate;
        rm.record_attempt(&state);
        rm.record_attempt(&state);
        assert!(!rm.should_retry(&state));
        rm.reset(&state);
        assert!(rm.should_retry(&state));
    }

    #[test]
    fn test_retry_manager_independent_states() {
        let mut rm = RetryManager::new(2, BackoffStrategy::Fixed(10));
        let s1 = ProtocolState::Initialized;
        let s2 = ProtocolState::CommitOutputs;
        rm.record_attempt(&s1);
        rm.record_attempt(&s1);
        assert!(!rm.should_retry(&s1));
        assert!(rm.should_retry(&s2)); // s2 is untouched
    }

    #[test]
    fn test_fibonacci_helper_values() {
        assert_eq!(fibonacci_ms(0), 1);
        assert_eq!(fibonacci_ms(1), 1);
        assert_eq!(fibonacci_ms(2), 2);
        assert_eq!(fibonacci_ms(3), 3);
        assert_eq!(fibonacci_ms(4), 5);
        assert_eq!(fibonacci_ms(5), 8);
        assert_eq!(fibonacci_ms(6), 13);
    }

    // ═══════════════════════════════════════════════════════════
    //  ProtocolAuditor tests
    // ═══════════════════════════════════════════════════════════

    fn make_audit_event(ts: u64, before: ProtocolState, after: ProtocolState) -> AuditEvent {
        AuditEvent::new(
            ts,
            EventType::StateTransition,
            before,
            after,
            "prover".to_string(),
            "transition".to_string(),
        )
    }

    #[test]
    fn test_auditor_new() {
        let a = ProtocolAuditor::new();
        assert_eq!(a.events.len(), 0);
    }

    #[test]
    fn test_auditor_record_and_log() {
        let mut a = ProtocolAuditor::new();
        a.record_event(make_audit_event(
            100,
            ProtocolState::Initialized,
            ProtocolState::CommitOutputs,
        ));
        let log = a.generate_audit_log();
        assert!(log.contains("PROTOCOL AUDIT LOG"));
        assert!(log.contains("Initialized"));
        assert!(log.contains("CommitOutputs"));
        assert!(log.contains("Total events: 1"));
    }

    #[test]
    fn test_auditor_verify_valid_trail() {
        let mut a = ProtocolAuditor::new();
        a.record_event(make_audit_event(100, ProtocolState::Initialized, ProtocolState::CommitOutputs));
        a.record_event(make_audit_event(200, ProtocolState::CommitOutputs, ProtocolState::RevealBenchmark));
        assert!(a.verify_audit_trail());
    }

    #[test]
    fn test_auditor_verify_out_of_order() {
        let mut a = ProtocolAuditor::new();
        a.record_event(make_audit_event(200, ProtocolState::Initialized, ProtocolState::CommitOutputs));
        a.record_event(make_audit_event(100, ProtocolState::CommitOutputs, ProtocolState::RevealBenchmark));
        assert!(!a.verify_audit_trail());
    }

    #[test]
    fn test_auditor_verify_tampered_hash() {
        let mut a = ProtocolAuditor::new();
        let mut ev = make_audit_event(100, ProtocolState::Initialized, ProtocolState::CommitOutputs);
        ev.hash = [0u8; 32]; // tamper
        a.record_event(ev);
        assert!(!a.verify_audit_trail());
    }

    #[test]
    fn test_auditor_events_in_range() {
        let mut a = ProtocolAuditor::new();
        a.record_event(make_audit_event(100, ProtocolState::Initialized, ProtocolState::CommitOutputs));
        a.record_event(make_audit_event(200, ProtocolState::CommitOutputs, ProtocolState::RevealBenchmark));
        a.record_event(make_audit_event(300, ProtocolState::RevealBenchmark, ProtocolState::Evaluate));
        let range = a.events_in_range(150, 250);
        assert_eq!(range.len(), 1);
        assert_eq!(range[0].timestamp, 200);
    }

    #[test]
    fn test_auditor_events_in_range_empty() {
        let a = ProtocolAuditor::new();
        assert!(a.events_in_range(0, 1000).is_empty());
    }

    #[test]
    fn test_auditor_summary() {
        let mut a = ProtocolAuditor::new();
        a.record_event(make_audit_event(100, ProtocolState::Initialized, ProtocolState::CommitOutputs));
        a.record_event(make_audit_event(200, ProtocolState::CommitOutputs, ProtocolState::RevealBenchmark));
        a.record_event(AuditEvent::new(
            300,
            EventType::StateTransition,
            ProtocolState::RevealBenchmark,
            ProtocolState::Aborted(AbortReason::ProofFailed),
            "verifier".to_string(),
            "failed".to_string(),
        ));
        let s = a.summary();
        assert_eq!(s.total_events, 3);
        assert_eq!(s.total_transitions, 3);
        assert_eq!(s.successful_transitions, 2);
        assert_eq!(s.failed_transitions, 1);
        assert_eq!(s.duration_ms, 200); // 300-100
        assert!(s.states_visited.len() >= 4);
    }

    #[test]
    fn test_auditor_summary_empty() {
        let a = ProtocolAuditor::new();
        let s = a.summary();
        assert_eq!(s.total_events, 0);
        assert_eq!(s.duration_ms, 0);
    }

    #[test]
    fn test_audit_event_hash_deterministic() {
        let e1 = AuditEvent::new(42, EventType::Commitment, ProtocolState::Initialized, ProtocolState::CommitOutputs, "a".to_string(), "d".to_string());
        let e2 = AuditEvent::new(42, EventType::Commitment, ProtocolState::Initialized, ProtocolState::CommitOutputs, "a".to_string(), "d".to_string());
        assert_eq!(e1.hash, e2.hash);
    }

    #[test]
    fn test_audit_event_hash_changes_with_content() {
        let e1 = AuditEvent::new(42, EventType::Commitment, ProtocolState::Initialized, ProtocolState::CommitOutputs, "a".to_string(), "d".to_string());
        let e2 = AuditEvent::new(43, EventType::Commitment, ProtocolState::Initialized, ProtocolState::CommitOutputs, "a".to_string(), "d".to_string());
        assert_ne!(e1.hash, e2.hash);
    }

    // ═══════════════════════════════════════════════════════════
    //  ProtocolMetrics tests
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_metrics_new() {
        let m = ProtocolMetrics::new();
        assert_eq!(m.total_transitions(), 0);
    }

    #[test]
    fn test_metrics_record_duration() {
        let mut m = ProtocolMetrics::new();
        m.record_state_duration(&ProtocolState::Prove, 100);
        m.record_state_duration(&ProtocolState::Prove, 200);
        let avg = m.average_duration(&ProtocolState::Prove);
        assert_eq!(avg, Some(150.0));
    }

    #[test]
    fn test_metrics_average_no_data() {
        let m = ProtocolMetrics::new();
        assert_eq!(m.average_duration(&ProtocolState::Initialized), None);
    }

    #[test]
    fn test_metrics_transitions() {
        let mut m = ProtocolMetrics::new();
        m.record_transition(&ProtocolState::Initialized, &ProtocolState::CommitOutputs);
        m.record_transition(&ProtocolState::Initialized, &ProtocolState::CommitOutputs);
        m.record_transition(&ProtocolState::CommitOutputs, &ProtocolState::RevealBenchmark);
        assert_eq!(m.total_transitions(), 3);
    }

    #[test]
    fn test_metrics_errors() {
        let mut m = ProtocolMetrics::new();
        m.record_error("timeout");
        m.record_error("timeout");
        m.record_error("invalid_proof");
        assert_eq!(m.most_common_error(), Some("timeout"));
    }

    #[test]
    fn test_metrics_no_errors() {
        let m = ProtocolMetrics::new();
        assert_eq!(m.most_common_error(), None);
    }

    #[test]
    fn test_metrics_to_json() {
        let mut m = ProtocolMetrics::new();
        m.record_state_duration(&ProtocolState::Initialized, 10);
        m.record_transition(&ProtocolState::Initialized, &ProtocolState::CommitOutputs);
        m.record_error("oops");
        let json = m.to_json();
        assert!(json.contains("state_durations"));
        assert!(json.contains("transition_counts"));
        assert!(json.contains("error_counts"));
        assert!(json.contains("oops"));
    }

    #[test]
    fn test_metrics_single_duration_average() {
        let mut m = ProtocolMetrics::new();
        m.record_state_duration(&ProtocolState::Verify, 42);
        assert_eq!(m.average_duration(&ProtocolState::Verify), Some(42.0));
    }

    // ═══════════════════════════════════════════════════════════
    //  ProtocolCheckpoint tests
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_checkpoint_create_and_verify() {
        let mut sm = make_sm();
        sm.transition_to(ProtocolState::CommitOutputs).unwrap();
        sm.store_commitment("x".to_string(), b"data".to_vec());
        let cp = ProtocolCheckpoint::create(&sm);
        assert!(cp.verify_integrity());
        assert_eq!(cp.state, ProtocolState::CommitOutputs);
    }

    #[test]
    fn test_checkpoint_restore() {
        let mut sm = make_sm();
        sm.transition_to(ProtocolState::CommitOutputs).unwrap();
        sm.transition_to(ProtocolState::RevealBenchmark).unwrap();
        let cp = ProtocolCheckpoint::create(&sm);
        let restored = ProtocolCheckpoint::restore(cp).unwrap();
        assert_eq!(*restored.current_state(), ProtocolState::RevealBenchmark);
    }

    #[test]
    fn test_checkpoint_tampered_fails_restore() {
        let sm = make_sm();
        let mut cp = ProtocolCheckpoint::create(&sm);
        cp.checksum = [0xFF; 32]; // tamper
        assert!(ProtocolCheckpoint::restore(cp).is_err());
    }

    #[test]
    fn test_checkpoint_serialize_deserialize() {
        let mut sm = make_sm();
        sm.transition_to(ProtocolState::CommitOutputs).unwrap();
        let cp = ProtocolCheckpoint::create(&sm);
        let bytes = cp.serialize();
        let cp2 = ProtocolCheckpoint::deserialize(&bytes).unwrap();
        assert!(cp2.verify_integrity());
        assert_eq!(cp2.state, ProtocolState::CommitOutputs);
    }

    #[test]
    fn test_checkpoint_deserialize_invalid() {
        let result = ProtocolCheckpoint::deserialize(b"garbage");
        assert!(result.is_err());
    }

    #[test]
    fn test_checkpoint_with_commitments() {
        let mut sm = make_sm();
        sm.store_commitment("a".to_string(), b"alpha".to_vec());
        sm.store_commitment("b".to_string(), b"beta".to_vec());
        let cp = ProtocolCheckpoint::create(&sm);
        assert!(cp.verify_integrity());
        assert_eq!(cp.committed_values.len(), 2);
    }

    #[test]
    fn test_checkpoint_roundtrip_preserves_history() {
        let mut sm = make_sm();
        sm.transition_to(ProtocolState::CommitOutputs).unwrap();
        sm.transition_to(ProtocolState::RevealBenchmark).unwrap();
        let history_len = sm.event_log().len();
        let cp = ProtocolCheckpoint::create(&sm);
        let bytes = cp.serialize();
        let cp2 = ProtocolCheckpoint::deserialize(&bytes).unwrap();
        assert_eq!(cp2.history.len(), history_len);
    }

    // ═══════════════════════════════════════════════════════════
    //  ProtocolSimulator tests
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_simulator_happy_path() {
        let mut sim = ProtocolSimulator::new(ProtocolConfig::default());
        sim.add_step(SimulationStep::Transition(ProtocolState::CommitOutputs));
        sim.add_step(SimulationStep::Transition(ProtocolState::RevealBenchmark));
        sim.add_step(SimulationStep::Transition(ProtocolState::Evaluate));
        sim.add_step(SimulationStep::Transition(ProtocolState::Prove));
        sim.add_step(SimulationStep::Transition(ProtocolState::Verify));
        sim.add_step(SimulationStep::Transition(ProtocolState::Certify));
        sim.add_step(SimulationStep::Transition(ProtocolState::Completed));
        let result = sim.run_simulation();
        assert_eq!(result.final_state, ProtocolState::Completed);
        assert_eq!(result.steps_completed, 7);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_simulator_with_commit_reveal() {
        let mut sim = ProtocolSimulator::new(ProtocolConfig::default());
        sim.add_step(SimulationStep::Commit("k".to_string(), b"val".to_vec()));
        sim.add_step(SimulationStep::Reveal("k".to_string(), b"val".to_vec()));
        sim.add_step(SimulationStep::Transition(ProtocolState::CommitOutputs));
        let result = sim.run_simulation();
        assert_eq!(result.final_state, ProtocolState::CommitOutputs);
        assert_eq!(result.steps_completed, 3);
    }

    #[test]
    fn test_simulator_inject_error() {
        let mut sim = ProtocolSimulator::new(ProtocolConfig::default());
        sim.add_step(SimulationStep::Transition(ProtocolState::CommitOutputs));
        sim.add_step(SimulationStep::InjectError(ProtocolError::InvalidState("boom".to_string())));
        let result = sim.run_simulation();
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.steps_completed, 2);
    }

    #[test]
    fn test_simulator_invalid_transition() {
        let mut sim = ProtocolSimulator::new(ProtocolConfig::default());
        sim.add_step(SimulationStep::Transition(ProtocolState::Prove)); // skip
        let result = sim.run_simulation();
        assert_eq!(result.errors.len(), 1);
    }

    #[test]
    fn test_simulator_with_failures() {
        let mut sim = ProtocolSimulator::new(ProtocolConfig::default());
        sim.add_step(SimulationStep::Transition(ProtocolState::CommitOutputs));
        sim.add_step(SimulationStep::Transition(ProtocolState::RevealBenchmark));
        sim.add_step(SimulationStep::Transition(ProtocolState::Evaluate));
        let result = sim.run_with_failures(&[1]); // skip step 1
        assert_eq!(result.errors.len(), 1);
        // Step 0 transitions to CommitOutputs, step 1 is skipped, step 2 tries
        // RevealBenchmark->Evaluate which should fail since we are at CommitOutputs
        // actually step 2 transitions to Evaluate from CommitOutputs which is invalid
        assert!(result.steps_completed >= 3);
    }

    #[test]
    fn test_simulator_with_delays() {
        let mut sim = ProtocolSimulator::new(ProtocolConfig::default());
        sim.add_step(SimulationStep::Transition(ProtocolState::CommitOutputs));
        sim.add_step(SimulationStep::Wait(500));
        sim.add_step(SimulationStep::Transition(ProtocolState::RevealBenchmark));
        let delays = vec![(ProtocolState::CommitOutputs, 100)];
        let result = sim.run_with_delays(&delays);
        assert_eq!(result.final_state, ProtocolState::RevealBenchmark);
        // total_time_ms includes accumulated_delay (500 from Wait + 100 from delay) plus wall
        assert!(result.total_time_ms >= 600);
    }

    #[test]
    fn test_simulator_empty_script() {
        let mut sim = ProtocolSimulator::new(ProtocolConfig::default());
        let result = sim.run_simulation();
        assert_eq!(result.final_state, ProtocolState::Initialized);
        assert_eq!(result.steps_completed, 0);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_simulator_wait_step() {
        let mut sim = ProtocolSimulator::new(ProtocolConfig::default());
        sim.add_step(SimulationStep::Wait(1000));
        let result = sim.run_simulation();
        assert_eq!(result.steps_completed, 1);
        assert_eq!(result.final_state, ProtocolState::Initialized);
    }

    // ═══════════════════════════════════════════════════════════
    //  StateGraph tests
    // ═══════════════════════════════════════════════════════════

    fn default_rules() -> Vec<StateTransitionRule> {
        ProtocolStateMachine::default_transition_rules()
    }

    #[test]
    fn test_state_graph_from_rules() {
        let g = StateGraph::from_rules(&default_rules());
        assert!(g.nodes.len() >= 8); // at least the 8 states in default rules
        assert_eq!(g.edges.len(), 7);
    }

    #[test]
    fn test_state_graph_reachable_from_init() {
        let g = StateGraph::from_rules(&default_rules());
        let reachable = g.reachable_from(&ProtocolState::Initialized);
        assert!(reachable.contains(&ProtocolState::CommitOutputs));
        assert!(reachable.contains(&ProtocolState::Completed));
    }

    #[test]
    fn test_state_graph_reachable_from_completed() {
        let g = StateGraph::from_rules(&default_rules());
        let reachable = g.reachable_from(&ProtocolState::Completed);
        assert!(reachable.is_empty()); // terminal
    }

    #[test]
    fn test_state_graph_reachable_unknown() {
        let g = StateGraph::from_rules(&default_rules());
        let reachable = g.reachable_from(&ProtocolState::Aborted(AbortReason::ProofFailed));
        assert!(reachable.is_empty()); // not in graph
    }

    #[test]
    fn test_state_graph_shortest_path_init_to_completed() {
        let g = StateGraph::from_rules(&default_rules());
        let path = g.shortest_path(&ProtocolState::Initialized, &ProtocolState::Completed);
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(*path.first().unwrap(), ProtocolState::Initialized);
        assert_eq!(*path.last().unwrap(), ProtocolState::Completed);
        assert_eq!(path.len(), 8); // all 8 states in order
    }

    #[test]
    fn test_state_graph_shortest_path_self() {
        let g = StateGraph::from_rules(&default_rules());
        let path = g.shortest_path(&ProtocolState::Initialized, &ProtocolState::Initialized);
        assert_eq!(path, Some(vec![ProtocolState::Initialized]));
    }

    #[test]
    fn test_state_graph_shortest_path_unreachable() {
        let g = StateGraph::from_rules(&default_rules());
        let path = g.shortest_path(&ProtocolState::Completed, &ProtocolState::Initialized);
        assert!(path.is_none());
    }

    #[test]
    fn test_state_graph_shortest_path_mid() {
        let g = StateGraph::from_rules(&default_rules());
        let path = g.shortest_path(&ProtocolState::Prove, &ProtocolState::Completed);
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path.len(), 4); // Prove -> Verify -> Certify -> Completed
    }

    #[test]
    fn test_state_graph_is_deadlock_free() {
        let g = StateGraph::from_rules(&default_rules());
        assert!(g.is_deadlock_free());
    }

    #[test]
    fn test_state_graph_deadlock_detected() {
        // Create a graph where a non-terminal node has no outgoing edges.
        let rules = vec![StateTransitionRule::new(
            ProtocolState::Initialized,
            ProtocolState::CommitOutputs,
            "go",
            1000,
        )];
        let g = StateGraph::from_rules(&rules);
        // CommitOutputs is non-terminal but has no outgoing edge → deadlock
        assert!(!g.is_deadlock_free());
    }

    #[test]
    fn test_state_graph_to_dot() {
        let g = StateGraph::from_rules(&default_rules());
        let dot = g.to_dot();
        assert!(dot.starts_with("digraph"));
        assert!(dot.contains("->"));
        assert!(dot.contains("Initialized"));
        assert!(dot.contains("Completed"));
        assert!(dot.contains("doublecircle")); // terminal node shape
        assert!(dot.contains("circle"));       // non-terminal node shape
    }

    #[test]
    fn test_state_graph_dot_contains_all_edges() {
        let g = StateGraph::from_rules(&default_rules());
        let dot = g.to_dot();
        // 7 rules → 7 edges → 7 occurrences of "->"
        let arrow_count = dot.matches("->").count();
        assert_eq!(arrow_count, 7);
    }

    #[test]
    fn test_state_graph_empty_rules() {
        let g = StateGraph::from_rules(&[]);
        assert!(g.nodes.is_empty());
        assert!(g.edges.is_empty());
        assert!(g.is_deadlock_free()); // vacuously true
    }

    // ═══════════════════════════════════════════════════════════
    //  Integration / cross-component tests
    // ═══════════════════════════════════════════════════════════

    #[test]
    fn test_metrics_from_simulation() {
        let mut sim = ProtocolSimulator::new(ProtocolConfig::default());
        sim.add_step(SimulationStep::Transition(ProtocolState::CommitOutputs));
        sim.add_step(SimulationStep::Transition(ProtocolState::RevealBenchmark));
        let result = sim.run_simulation();

        let mut metrics = ProtocolMetrics::new();
        for ev in &result.events {
            metrics.record_transition(&ev.from_state, &ev.to_state);
        }
        assert_eq!(metrics.total_transitions(), result.events.len());
    }

    #[test]
    fn test_auditor_from_simulation_events() {
        let mut sim = ProtocolSimulator::new(ProtocolConfig::default());
        sim.add_step(SimulationStep::Transition(ProtocolState::CommitOutputs));
        sim.add_step(SimulationStep::Transition(ProtocolState::RevealBenchmark));
        sim.add_step(SimulationStep::Transition(ProtocolState::Evaluate));
        let result = sim.run_simulation();

        let mut auditor = ProtocolAuditor::new();
        for ev in &result.events {
            auditor.record_event(AuditEvent::new(
                ev.timestamp,
                ev.event_type.clone(),
                ev.from_state.clone(),
                ev.to_state.clone(),
                "simulator".to_string(),
                ev.details.clone(),
            ));
        }
        assert!(auditor.verify_audit_trail());
        let s = auditor.summary();
        assert!(s.total_events >= 3);
    }

    #[test]
    fn test_checkpoint_roundtrip_after_simulation() {
        let mut sim = ProtocolSimulator::new(ProtocolConfig::default());
        sim.add_step(SimulationStep::Commit("key".to_string(), b"val".to_vec()));
        sim.add_step(SimulationStep::Transition(ProtocolState::CommitOutputs));
        sim.add_step(SimulationStep::Transition(ProtocolState::RevealBenchmark));
        sim.run_simulation();

        let cp = ProtocolCheckpoint::create(&sim.machine);
        assert!(cp.verify_integrity());
        let bytes = cp.serialize();
        let cp2 = ProtocolCheckpoint::deserialize(&bytes).unwrap();
        assert!(cp2.verify_integrity());
        assert_eq!(cp2.state, ProtocolState::RevealBenchmark);
    }

    #[test]
    fn test_retry_and_phase_manager_together() {
        let mut rm = RetryManager::new(3, BackoffStrategy::Exponential(50, 2.0));
        let mut mgr = ProtocolPhaseManager::new(vec![
            ProtocolPhase {
                name: "step1".to_string(),
                required_state: ProtocolState::Initialized,
                timeout_ms: 1000,
                validator: success_only,
            },
        ]);

        // Simulate retries until success
        let state = ProtocolState::Initialized;
        let mut attempt = 0;
        while rm.should_retry(&state) && attempt < 3 {
            rm.record_attempt(&state);
            attempt += 1;
            if attempt == 2 {
                mgr.record_result("step1".to_string(), PhaseResult::ok(rm.delay_ms(&state)));
            }
        }

        assert!(mgr.advance().is_ok());
        assert!(mgr.is_complete());
    }

    #[test]
    fn test_graph_and_simulator_consistency() {
        let g = StateGraph::from_rules(&default_rules());
        let path = g.shortest_path(&ProtocolState::Initialized, &ProtocolState::Completed).unwrap();

        let mut sim = ProtocolSimulator::new(ProtocolConfig::default());
        // Skip the first element (Initialized) since the machine starts there.
        for state in path.iter().skip(1) {
            sim.add_step(SimulationStep::Transition(state.clone()));
        }
        let result = sim.run_simulation();
        assert_eq!(result.final_state, ProtocolState::Completed);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_phase_durations_empty() {
        let mgr = ProtocolPhaseManager::new(sample_phases());
        assert!(mgr.phase_durations().is_empty());
    }

    #[test]
    fn test_metrics_multiple_states() {
        let mut m = ProtocolMetrics::new();
        m.record_state_duration(&ProtocolState::Initialized, 10);
        m.record_state_duration(&ProtocolState::CommitOutputs, 20);
        m.record_state_duration(&ProtocolState::Initialized, 30);
        assert_eq!(m.average_duration(&ProtocolState::Initialized), Some(20.0));
        assert_eq!(m.average_duration(&ProtocolState::CommitOutputs), Some(20.0));
    }

    #[test]
    fn test_auditor_multiple_event_types() {
        let mut a = ProtocolAuditor::new();
        a.record_event(AuditEvent::new(
            10, EventType::Commitment, ProtocolState::Initialized, ProtocolState::Initialized,
            "prover".to_string(), "commit".to_string(),
        ));
        a.record_event(AuditEvent::new(
            20, EventType::Reveal, ProtocolState::Initialized, ProtocolState::Initialized,
            "prover".to_string(), "reveal".to_string(),
        ));
        a.record_event(AuditEvent::new(
            30, EventType::StateTransition, ProtocolState::Initialized, ProtocolState::CommitOutputs,
            "prover".to_string(), "go".to_string(),
        ));
        assert!(a.verify_audit_trail());
        let s = a.summary();
        assert_eq!(s.total_events, 3);
        assert_eq!(s.total_transitions, 1);
    }

    #[test]
    fn test_checkpoint_empty_machine() {
        let sm = make_sm();
        let cp = ProtocolCheckpoint::create(&sm);
        assert!(cp.verify_integrity());
        assert!(cp.history.is_empty());
        let restored = ProtocolCheckpoint::restore(cp).unwrap();
        assert_eq!(*restored.current_state(), ProtocolState::Initialized);
    }

    #[test]
    fn test_simulator_run_with_failures_no_failures() {
        let mut sim = ProtocolSimulator::new(ProtocolConfig::default());
        sim.add_step(SimulationStep::Transition(ProtocolState::CommitOutputs));
        let result = sim.run_with_failures(&[]);
        assert!(result.errors.is_empty());
        assert_eq!(result.final_state, ProtocolState::CommitOutputs);
    }

    #[test]
    fn test_state_graph_reachable_intermediate() {
        let g = StateGraph::from_rules(&default_rules());
        let reachable = g.reachable_from(&ProtocolState::Evaluate);
        assert!(reachable.contains(&ProtocolState::Prove));
        assert!(reachable.contains(&ProtocolState::Completed));
        assert!(!reachable.contains(&ProtocolState::Initialized));
    }

    // ── ProtocolRunner tests ──

    #[test]
    fn test_runner_new() {
        let runner = ProtocolRunner::new(ProtocolConfig::default());
        assert_eq!(*runner.current_state(), ProtocolState::Initialized);
        assert_eq!(runner.event_count(), 0);
    }

    #[test]
    fn test_runner_register_handler_and_single_step() {
        let mut runner = ProtocolRunner::new(ProtocolConfig::default());
        runner.register_handler(ProtocolState::Initialized, |machine| {
            machine.transition_to(ProtocolState::CommitOutputs)
        });
        let state = runner.run_single_step().unwrap();
        assert_eq!(state, ProtocolState::CommitOutputs);
        assert!(runner.event_count() > 0);
    }

    #[test]
    fn test_runner_run_to_completion() {
        let mut runner = ProtocolRunner::new(ProtocolConfig::default());
        runner.register_handler(ProtocolState::Initialized, |m| {
            m.transition_to(ProtocolState::CommitOutputs)
        });
        runner.register_handler(ProtocolState::CommitOutputs, |m| {
            m.transition_to(ProtocolState::RevealBenchmark)
        });
        runner.register_handler(ProtocolState::RevealBenchmark, |m| {
            m.transition_to(ProtocolState::Evaluate)
        });
        runner.register_handler(ProtocolState::Evaluate, |m| {
            m.transition_to(ProtocolState::Prove)
        });
        runner.register_handler(ProtocolState::Prove, |m| {
            m.transition_to(ProtocolState::Verify)
        });
        runner.register_handler(ProtocolState::Verify, |m| {
            m.transition_to(ProtocolState::Certify)
        });
        runner.register_handler(ProtocolState::Certify, |m| {
            m.transition_to(ProtocolState::Completed)
        });
        let final_state = runner.run_to_completion().unwrap();
        assert_eq!(final_state, ProtocolState::Completed);
        assert_eq!(runner.event_count(), 7);
    }

    #[test]
    fn test_runner_missing_handler() {
        let mut runner = ProtocolRunner::new(ProtocolConfig::default());
        // No handlers registered – should error
        let result = runner.run_single_step();
        assert!(result.is_err());
    }

    #[test]
    fn test_runner_event_count_increments() {
        let mut runner = ProtocolRunner::new(ProtocolConfig::default());
        runner.register_handler(ProtocolState::Initialized, |m| {
            m.transition_to(ProtocolState::CommitOutputs)
        });
        assert_eq!(runner.event_count(), 0);
        runner.run_single_step().unwrap();
        assert_eq!(runner.event_count(), 1);
    }

    // ── ProtocolTemplate tests ──

    #[test]
    fn test_template_evaluation_protocol() {
        let tmpl = ProtocolTemplate::evaluation_protocol();
        assert_eq!(tmpl.name, "evaluation-protocol");
        assert_eq!(tmpl.states.len(), 8);
        assert_eq!(tmpl.transitions.len(), 7);
        assert!(tmpl.default_timeouts.contains_key("Prove"));
    }

    #[test]
    fn test_template_certification_protocol() {
        let tmpl = ProtocolTemplate::certification_protocol();
        assert_eq!(tmpl.name, "certification-protocol");
        assert_eq!(tmpl.states.len(), 4);
        assert_eq!(tmpl.transitions.len(), 3);
        assert!(tmpl.states.contains(&ProtocolState::Certify));
    }

    #[test]
    fn test_template_verification_only() {
        let tmpl = ProtocolTemplate::verification_only();
        assert_eq!(tmpl.name, "verification-only");
        assert_eq!(tmpl.states.len(), 3);
        assert_eq!(tmpl.transitions.len(), 2);
        assert!(!tmpl.states.contains(&ProtocolState::Certify));
    }

    #[test]
    fn test_template_to_config() {
        let tmpl = ProtocolTemplate::evaluation_protocol();
        let cfg = tmpl.to_config();
        assert_eq!(cfg.name, "evaluation-protocol");
        assert!(cfg.enable_logging);
        assert_eq!(cfg.max_retries, 3);
        assert_eq!(cfg.timeouts.len(), tmpl.default_timeouts.len());
    }

    #[test]
    fn test_template_instantiate() {
        let tmpl = ProtocolTemplate::evaluation_protocol();
        let machine = tmpl.instantiate();
        assert_eq!(*machine.current_state(), ProtocolState::Initialized);
        assert!(!machine.is_terminal());
    }

    // ── StateHistory tests ──

    fn make_sample_events() -> Vec<ProtocolEvent> {
        vec![
            ProtocolEvent::new(
                ProtocolState::Initialized,
                ProtocolState::CommitOutputs,
                1000,
                EventType::StateTransition,
                "step1".to_string(),
            ),
            ProtocolEvent::new(
                ProtocolState::CommitOutputs,
                ProtocolState::RevealBenchmark,
                1050,
                EventType::StateTransition,
                "step2".to_string(),
            ),
            ProtocolEvent::new(
                ProtocolState::RevealBenchmark,
                ProtocolState::Evaluate,
                1200,
                EventType::StateTransition,
                "step3".to_string(),
            ),
            ProtocolEvent::new(
                ProtocolState::Evaluate,
                ProtocolState::Prove,
                1500,
                EventType::StateTransition,
                "step4".to_string(),
            ),
        ]
    }

    #[test]
    fn test_state_history_new() {
        let hist = StateHistory::new(make_sample_events());
        assert_eq!(hist.events.len(), 4);
    }

    #[test]
    fn test_state_history_from_machine() {
        let mut sm = make_sm();
        sm.transition_to(ProtocolState::CommitOutputs).unwrap();
        let hist = StateHistory::from_machine(&sm);
        assert_eq!(hist.events.len(), 1);
    }

    #[test]
    fn test_state_history_time_in_state() {
        let hist = StateHistory::new(make_sample_events());
        // CommitOutputs entered at t=1000, next event at t=1050 → 50ms
        assert_eq!(hist.time_in_state(&ProtocolState::CommitOutputs), 50);
        // RevealBenchmark entered at t=1050, next event at t=1200 → 150ms
        assert_eq!(hist.time_in_state(&ProtocolState::RevealBenchmark), 150);
        // Evaluate entered at t=1200, next event at t=1500 → 300ms
        assert_eq!(hist.time_in_state(&ProtocolState::Evaluate), 300);
        // Prove is the last event – no successor, so 0ms
        assert_eq!(hist.time_in_state(&ProtocolState::Prove), 0);
    }

    #[test]
    fn test_state_history_transition_count() {
        let hist = StateHistory::new(make_sample_events());
        assert_eq!(
            hist.transition_count(&ProtocolState::Initialized, &ProtocolState::CommitOutputs),
            1
        );
        assert_eq!(
            hist.transition_count(&ProtocolState::Initialized, &ProtocolState::Prove),
            0
        );
    }

    #[test]
    fn test_state_history_states_visited() {
        let hist = StateHistory::new(make_sample_events());
        let visited = hist.states_visited();
        assert!(visited.contains(&ProtocolState::Initialized));
        assert!(visited.contains(&ProtocolState::CommitOutputs));
        assert!(visited.contains(&ProtocolState::RevealBenchmark));
        assert!(visited.contains(&ProtocolState::Evaluate));
        assert!(visited.contains(&ProtocolState::Prove));
        // Each state appears only once
        let init_count = visited.iter().filter(|s| **s == ProtocolState::Initialized).count();
        assert_eq!(init_count, 1);
    }

    #[test]
    fn test_state_history_longest_state() {
        let hist = StateHistory::new(make_sample_events());
        let (state, duration) = hist.longest_state().unwrap();
        // Evaluate has the longest duration (300ms)
        assert_eq!(state, ProtocolState::Evaluate);
        assert_eq!(duration, 300);
    }

    #[test]
    fn test_state_history_longest_state_empty() {
        let hist = StateHistory::new(vec![]);
        assert!(hist.longest_state().is_none());
    }

    #[test]
    fn test_state_history_timeline() {
        let hist = StateHistory::new(make_sample_events());
        let tl = hist.timeline();
        assert_eq!(tl.len(), 4);
        assert_eq!(tl[0], (1000, ProtocolState::CommitOutputs));
        assert_eq!(tl[3], (1500, ProtocolState::Prove));
    }

    #[test]
    fn test_state_history_to_csv() {
        let hist = StateHistory::new(make_sample_events());
        let csv = hist.to_csv();
        assert!(csv.starts_with("timestamp,from,to,event_type,details\n"));
        let lines: Vec<&str> = csv.lines().collect();
        // header + 4 data rows
        assert_eq!(lines.len(), 5);
        assert!(lines[1].contains("Initialized"));
        assert!(lines[1].contains("CommitOutputs"));
    }
}
