//! Multi-shot prompt engine for LLM-assisted litmus test generation.
//!
//! Supports 1-shot, 3-shot, and 5-shot prompting with curated examples,
//! chain-of-thought decomposition, and multiple LLM API backends.

use std::collections::HashMap;
use std::fmt;
use serde::{Serialize, Deserialize};

use crate::checker::litmus::{
    LitmusTest, Thread, Instruction, Ordering, Outcome, LitmusOutcome, RegId, Scope,
};
use crate::checker::execution::{Address, Value};

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

/// Which LLM backend to target.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LlmBackend {
    OpenAI { api_base: String },
    Anthropic { api_base: String },
    Local { endpoint: String },
}

impl Default for LlmBackend {
    fn default() -> Self {
        LlmBackend::OpenAI {
            api_base: "https://api.openai.com/v1".to_string(),
        }
    }
}

/// Top-level configuration for the prompt engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptConfig {
    pub model_name: String,
    pub temperature: f64,
    pub max_tokens: usize,
    pub num_shots: usize,
    pub backend: LlmBackend,
    pub chain_of_thought: bool,
    pub output_format: OutputFormat,
}

impl Default for PromptConfig {
    fn default() -> Self {
        Self {
            model_name: "gpt-4".to_string(),
            temperature: 0.2,
            max_tokens: 4096,
            num_shots: 3,
            backend: LlmBackend::default(),
            chain_of_thought: true,
            output_format: OutputFormat::Pseudocode,
        }
    }
}

/// Desired output format the LLM should produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputFormat {
    Pseudocode,
    CLike,
    AssemblyLike,
}

impl fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OutputFormat::Pseudocode => write!(f, "pseudocode"),
            OutputFormat::CLike => write!(f, "C-like"),
            OutputFormat::AssemblyLike => write!(f, "assembly-like"),
        }
    }
}

// ---------------------------------------------------------------------------
// Shot examples
// ---------------------------------------------------------------------------

/// A single input-output pair used for few-shot prompting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShotExample {
    pub description: String,
    pub input: String,
    pub output: String,
    pub pattern_tag: String,
}

impl ShotExample {
    pub fn new(description: &str, input: &str, output: &str, pattern_tag: &str) -> Self {
        Self {
            description: description.to_string(),
            input: input.to_string(),
            output: output.to_string(),
            pattern_tag: pattern_tag.to_string(),
        }
    }

    /// Render the example as a prompt fragment.
    pub fn render(&self) -> String {
        format!(
            "### Example ({tag})\n\
             **Description:** {desc}\n\
             **Input:** {input}\n\
             **Output:**\n```\n{output}\n```\n",
            tag = self.pattern_tag,
            desc = self.description,
            input = self.input,
            output = self.output,
        )
    }
}

// ---------------------------------------------------------------------------
// Prompt template
// ---------------------------------------------------------------------------

/// A fully-assembled prompt ready to send to the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplate {
    pub system_prompt: String,
    pub examples: Vec<ShotExample>,
    pub user_prompt: String,
    pub chain_of_thought_instructions: Option<String>,
    pub output_format_instructions: String,
}

impl PromptTemplate {
    /// Render into a single string suitable for chat-style APIs.
    pub fn render_chat_messages(&self) -> Vec<ChatMessage> {
        let mut msgs = Vec::new();

        // System message
        let mut system = self.system_prompt.clone();
        if let Some(ref cot) = self.chain_of_thought_instructions {
            system.push_str("\n\n");
            system.push_str(cot);
        }
        system.push_str("\n\n");
        system.push_str(&self.output_format_instructions);
        msgs.push(ChatMessage {
            role: ChatRole::System,
            content: system,
        });

        // Few-shot examples as user/assistant pairs
        for ex in &self.examples {
            msgs.push(ChatMessage {
                role: ChatRole::User,
                content: ex.input.clone(),
            });
            msgs.push(ChatMessage {
                role: ChatRole::Assistant,
                content: ex.output.clone(),
            });
        }

        // Actual request
        msgs.push(ChatMessage {
            role: ChatRole::User,
            content: self.user_prompt.clone(),
        });

        msgs
    }

    /// Render into a flat completion-style string.
    pub fn render_flat(&self) -> String {
        let mut buf = String::with_capacity(4096);
        buf.push_str(&self.system_prompt);
        buf.push_str("\n\n");

        if let Some(ref cot) = self.chain_of_thought_instructions {
            buf.push_str(cot);
            buf.push_str("\n\n");
        }

        buf.push_str(&self.output_format_instructions);
        buf.push_str("\n\n");

        for ex in &self.examples {
            buf.push_str(&ex.render());
            buf.push('\n');
        }

        buf.push_str("### Your Task\n");
        buf.push_str(&self.user_prompt);
        buf
    }
}

/// A chat message for chat-style LLM APIs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

/// Role in a chat conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

// ---------------------------------------------------------------------------
// Generation request / result
// ---------------------------------------------------------------------------

/// What kind of litmus test to generate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRequest {
    pub target_pattern: String,
    pub memory_model: String,
    pub constraints: Vec<GenerationConstraint>,
    pub description: Option<String>,
    pub num_threads: Option<usize>,
    pub max_instructions_per_thread: Option<usize>,
}

/// Constraints on the generated test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GenerationConstraint {
    MaxThreads(usize),
    MaxInstructions(usize),
    RequireOrdering(Ordering),
    ForbidOrdering(Ordering),
    RequireScope(String),
    RequireOutcome(LitmusOutcome),
    Custom(String),
}

impl fmt::Display for GenerationConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GenerationConstraint::MaxThreads(n) => write!(f, "at most {} threads", n),
            GenerationConstraint::MaxInstructions(n) => {
                write!(f, "at most {} instructions per thread", n)
            }
            GenerationConstraint::RequireOrdering(o) => {
                write!(f, "must use {:?} ordering", o)
            }
            GenerationConstraint::ForbidOrdering(o) => {
                write!(f, "must NOT use {:?} ordering", o)
            }
            GenerationConstraint::RequireScope(s) => {
                write!(f, "must target scope {}", s)
            }
            GenerationConstraint::RequireOutcome(o) => {
                write!(f, "expected outcome must be {:?}", o)
            }
            GenerationConstraint::Custom(s) => write!(f, "{}", s),
        }
    }
}

/// Result of a generation attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    pub test: Option<LitmusTest>,
    pub raw_response: String,
    pub confidence: f64,
    pub parse_success: bool,
    pub parse_errors: Vec<String>,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

// ---------------------------------------------------------------------------
// Parse errors
// ---------------------------------------------------------------------------

/// Errors that can occur while parsing LLM output into a `LitmusTest`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParseError {
    NoTestFound,
    MalformedThread { thread_id: usize, reason: String },
    UnknownInstruction(String),
    UnknownOrdering(String),
    InvalidAddress(String),
    InvalidValue(String),
    MissingField(String),
    InvalidOutcome(String),
    MultipleErrors(Vec<ParseError>),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::NoTestFound => write!(f, "no litmus test found in LLM output"),
            ParseError::MalformedThread { thread_id, reason } => {
                write!(f, "thread {} malformed: {}", thread_id, reason)
            }
            ParseError::UnknownInstruction(s) => write!(f, "unknown instruction: {}", s),
            ParseError::UnknownOrdering(s) => write!(f, "unknown ordering: {}", s),
            ParseError::InvalidAddress(s) => write!(f, "invalid address: {}", s),
            ParseError::InvalidValue(s) => write!(f, "invalid value: {}", s),
            ParseError::MissingField(s) => write!(f, "missing field: {}", s),
            ParseError::InvalidOutcome(s) => write!(f, "invalid outcome: {}", s),
            ParseError::MultipleErrors(errs) => {
                write!(f, "multiple errors: ")?;
                for (i, e) in errs.iter().enumerate() {
                    if i > 0 {
                        write!(f, "; ")?;
                    }
                    write!(f, "{}", e)?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for ParseError {}

// ---------------------------------------------------------------------------
// Litmus test parser
// ---------------------------------------------------------------------------

/// Parses raw LLM text output into a `LitmusTest`.
///
/// Supports pseudocode, C-like, and assembly-like formats with error recovery.
#[derive(Debug, Clone)]
pub struct LitmusTestParser {
    pub format_hint: Option<OutputFormat>,
    pub lenient: bool,
}

impl Default for LitmusTestParser {
    fn default() -> Self {
        Self {
            format_hint: None,
            lenient: true,
        }
    }
}

impl LitmusTestParser {
    pub fn new(format_hint: OutputFormat) -> Self {
        Self {
            format_hint: Some(format_hint),
            lenient: true,
        }
    }

    pub fn strict(format_hint: OutputFormat) -> Self {
        Self {
            format_hint: Some(format_hint),
            lenient: false,
        }
    }

    /// Try to parse the full response, attempting each format in turn.
    pub fn parse(&self, raw: &str) -> Result<LitmusTest, ParseError> {
        // Extract code block if present
        let text = Self::extract_code_block(raw);

        // Try the hinted format first
        if let Some(fmt) = self.format_hint {
            if let Ok(test) = self.parse_format(&text, fmt) {
                return Ok(test);
            }
        }

        // Try all formats
        for fmt in &[
            OutputFormat::Pseudocode,
            OutputFormat::CLike,
            OutputFormat::AssemblyLike,
        ] {
            if let Ok(test) = self.parse_format(&text, *fmt) {
                return Ok(test);
            }
        }

        Err(ParseError::NoTestFound)
    }

    /// Extract a fenced code block from markdown-style output.
    fn extract_code_block(raw: &str) -> String {
        let mut in_block = false;
        let mut block = String::new();

        for line in raw.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("```") {
                if in_block {
                    break;
                }
                in_block = true;
                continue;
            }
            if in_block {
                block.push_str(line);
                block.push('\n');
            }
        }

        if block.is_empty() {
            raw.to_string()
        } else {
            block
        }
    }

    /// Parse in a specific format.
    fn parse_format(&self, text: &str, fmt: OutputFormat) -> Result<LitmusTest, ParseError> {
        match fmt {
            OutputFormat::Pseudocode => self.parse_pseudocode(text),
            OutputFormat::CLike => self.parse_c_like(text),
            OutputFormat::AssemblyLike => self.parse_assembly_like(text),
        }
    }

    // ---- Pseudocode parser ------------------------------------------------

    fn parse_pseudocode(&self, text: &str) -> Result<LitmusTest, ParseError> {
        let mut name = String::from("generated_test");
        let mut threads: Vec<Thread> = Vec::new();
        let mut initial_state: HashMap<Address, Value> = HashMap::new();
        let mut expected_outcomes: Vec<(Outcome, LitmusOutcome)> = Vec::new();

        let mut current_thread: Option<(usize, Vec<Instruction>)> = None;

        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with('#') {
                continue;
            }

            // Test name
            if let Some(n) = trimmed.strip_prefix("test ").or_else(|| trimmed.strip_prefix("Test "))
            {
                name = n.trim_end_matches(':').trim().to_string();
                continue;
            }

            // Initial state: "init x = 0"
            if let Some(rest) = trimmed
                .strip_prefix("init ")
                .or_else(|| trimmed.strip_prefix("Init "))
            {
                if let Some((addr_s, val_s)) = rest.split_once('=') {
                    let addr = Self::parse_address(addr_s.trim())?;
                    let val = Self::parse_value(val_s.trim())?;
                    initial_state.insert(addr, val);
                }
                continue;
            }

            // Thread header: "Thread 0:" or "T0:"
            if let Some(tid) = Self::parse_thread_header(trimmed) {
                if let Some((id, instrs)) = current_thread.take() {
                    threads.push(Thread {
                        id,
                        instructions: instrs,
                    });
                }
                current_thread = Some((tid, Vec::new()));
                continue;
            }

            // Instructions inside a thread
            if let Some((_, ref mut instrs)) = current_thread {
                if let Ok(instr) = self.parse_instruction_pseudocode(trimmed) {
                    instrs.push(instr);
                } else if self.lenient {
                    // skip unrecognised lines
                } else {
                    return Err(ParseError::UnknownInstruction(trimmed.to_string()));
                }
                continue;
            }

            // Outcome line: "outcome: r0=1, r1=0 -> Forbidden"
            if let Some(rest) = trimmed
                .strip_prefix("outcome:")
                .or_else(|| trimmed.strip_prefix("Outcome:"))
                .or_else(|| trimmed.strip_prefix("expected:"))
            {
                if let Ok((outcome, classification)) = self.parse_outcome_line(rest.trim()) {
                    expected_outcomes.push((outcome, classification));
                }
            }
        }

        // Flush last thread
        if let Some((id, instrs)) = current_thread {
            threads.push(Thread {
                id,
                instructions: instrs,
            });
        }

        if threads.is_empty() {
            return Err(ParseError::NoTestFound);
        }

        Ok(LitmusTest {
            name,
            threads,
            initial_state,
            expected_outcomes,
        })
    }

    // ---- C-like parser ----------------------------------------------------

    fn parse_c_like(&self, text: &str) -> Result<LitmusTest, ParseError> {
        let mut name = String::from("generated_test");
        let mut threads: Vec<Thread> = Vec::new();
        let mut initial_state: HashMap<Address, Value> = HashMap::new();
        let mut expected_outcomes: Vec<(Outcome, LitmusOutcome)> = Vec::new();

        let mut current_thread: Option<(usize, Vec<Instruction>)> = None;
        let mut brace_depth: i32 = 0;

        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("//") {
                continue;
            }

            // Look for "void thread0()" style headers
            if let Some(tid) = Self::parse_c_thread_header(trimmed) {
                if let Some((id, instrs)) = current_thread.take() {
                    threads.push(Thread {
                        id,
                        instructions: instrs,
                    });
                }
                current_thread = Some((tid, Vec::new()));
                brace_depth = 0;
                continue;
            }

            if trimmed == "{" {
                brace_depth += 1;
                continue;
            }
            if trimmed == "}" {
                brace_depth -= 1;
                if brace_depth <= 0 {
                    if let Some((id, instrs)) = current_thread.take() {
                        threads.push(Thread {
                            id,
                            instructions: instrs,
                        });
                    }
                }
                continue;
            }

            // Initialisation: "x = 0;" at top-level
            if current_thread.is_none() {
                if let Some(rest) = trimmed.strip_suffix(';') {
                    if let Some((addr_s, val_s)) = rest.split_once('=') {
                        if let (Ok(a), Ok(v)) = (
                            Self::parse_address(addr_s.trim()),
                            Self::parse_value(val_s.trim()),
                        ) {
                            initial_state.insert(a, v);
                            continue;
                        }
                    }
                }
            }

            // Instructions
            if let Some((_, ref mut instrs)) = current_thread {
                let cleaned = trimmed.trim_end_matches(';');
                if let Ok(instr) = self.parse_instruction_c_like(cleaned) {
                    instrs.push(instr);
                } else if !self.lenient {
                    return Err(ParseError::UnknownInstruction(trimmed.to_string()));
                }
                continue;
            }

            // Outcome
            if let Some(rest) = trimmed
                .strip_prefix("outcome:")
                .or_else(|| trimmed.strip_prefix("expected:"))
            {
                if let Ok((outcome, classification)) = self.parse_outcome_line(rest.trim()) {
                    expected_outcomes.push((outcome, classification));
                }
            }
        }

        if threads.is_empty() {
            return Err(ParseError::NoTestFound);
        }

        Ok(LitmusTest {
            name,
            threads,
            initial_state,
            expected_outcomes,
        })
    }

    // ---- Assembly-like parser ---------------------------------------------

    fn parse_assembly_like(&self, text: &str) -> Result<LitmusTest, ParseError> {
        let mut name = String::from("generated_test");
        let mut threads: Vec<Thread> = Vec::new();
        let mut initial_state: HashMap<Address, Value> = HashMap::new();
        let mut expected_outcomes: Vec<(Outcome, LitmusOutcome)> = Vec::new();

        let mut current_thread: Option<(usize, Vec<Instruction>)> = None;

        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty()
                || trimmed.starts_with(';')
                || trimmed.starts_with("//")
                || trimmed.starts_with('#')
            {
                continue;
            }

            // Thread header
            if let Some(tid) = Self::parse_thread_header(trimmed)
                .or_else(|| Self::parse_asm_thread_header(trimmed))
            {
                if let Some((id, instrs)) = current_thread.take() {
                    threads.push(Thread {
                        id,
                        instructions: instrs,
                    });
                }
                current_thread = Some((tid, Vec::new()));
                continue;
            }

            // Init
            if let Some(rest) = trimmed.strip_prefix(".init") {
                let rest = rest.trim();
                if let Some((addr_s, val_s)) = rest.split_once('=') {
                    if let (Ok(a), Ok(v)) = (
                        Self::parse_address(addr_s.trim()),
                        Self::parse_value(val_s.trim()),
                    ) {
                        initial_state.insert(a, v);
                    }
                }
                continue;
            }

            // Instructions
            if let Some((_, ref mut instrs)) = current_thread {
                if let Ok(instr) = self.parse_instruction_asm(trimmed) {
                    instrs.push(instr);
                } else if !self.lenient {
                    return Err(ParseError::UnknownInstruction(trimmed.to_string()));
                }
                continue;
            }

            // Outcome
            if let Some(rest) = trimmed.strip_prefix("outcome:").or_else(|| trimmed.strip_prefix("exists")) {
                if let Ok((outcome, classification)) = self.parse_outcome_line(rest.trim()) {
                    expected_outcomes.push((outcome, classification));
                }
            }
        }

        if let Some((id, instrs)) = current_thread {
            threads.push(Thread {
                id,
                instructions: instrs,
            });
        }

        if threads.is_empty() {
            return Err(ParseError::NoTestFound);
        }

        Ok(LitmusTest {
            name,
            threads,
            initial_state,
            expected_outcomes,
        })
    }

    // ---- Shared helpers ---------------------------------------------------

    fn parse_thread_header(s: &str) -> Option<usize> {
        // "Thread 0:", "T0:", "thread 1:"
        let s = s.trim_end_matches(':');
        if let Some(rest) = s
            .strip_prefix("Thread ")
            .or_else(|| s.strip_prefix("thread "))
        {
            return rest.trim().parse().ok();
        }
        if let Some(rest) = s.strip_prefix('T').or_else(|| s.strip_prefix('t')) {
            if rest.chars().all(|c| c.is_ascii_digit()) {
                return rest.parse().ok();
            }
        }
        None
    }

    fn parse_c_thread_header(s: &str) -> Option<usize> {
        // "void thread0()" or "void P0()"
        let s = s.trim();
        let after = s
            .strip_prefix("void ")
            .unwrap_or(s);
        let after = after.trim_end_matches('{').trim_end_matches("()").trim();
        if let Some(rest) = after
            .strip_prefix("thread")
            .or_else(|| after.strip_prefix("Thread"))
            .or_else(|| after.strip_prefix('P'))
        {
            return rest.parse().ok();
        }
        None
    }

    fn parse_asm_thread_header(s: &str) -> Option<usize> {
        // "P0:" or ".proc 0"
        let s = s.trim_end_matches(':');
        if let Some(rest) = s.strip_prefix('P') {
            if rest.chars().all(|c| c.is_ascii_digit()) {
                return rest.parse().ok();
            }
        }
        if let Some(rest) = s.strip_prefix(".proc ") {
            return rest.trim().parse().ok();
        }
        None
    }

    fn parse_address(s: &str) -> Result<Address, ParseError> {
        let s = s.trim();
        // Named locations: x -> 0, y -> 1, z -> 2, ...
        if s.len() == 1 && s.chars().next().map_or(false, |c| c.is_ascii_lowercase()) {
            return Ok((s.as_bytes()[0] - b'a') as u64);
        }
        // Prefixed: [x], addr(0), *x
        let inner = s
            .trim_start_matches('[')
            .trim_end_matches(']')
            .trim_start_matches('*')
            .trim_start_matches("addr(")
            .trim_end_matches(')');
        if inner.len() == 1 && inner.chars().next().map_or(false, |c| c.is_ascii_lowercase()) {
            return Ok((inner.as_bytes()[0] - b'a') as u64);
        }
        // Numeric
        if let Some(hex) = inner.strip_prefix("0x") {
            return u64::from_str_radix(hex, 16)
                .map_err(|_| ParseError::InvalidAddress(s.to_string()));
        }
        inner
            .parse::<u64>()
            .map_err(|_| ParseError::InvalidAddress(s.to_string()))
    }

    fn parse_value(s: &str) -> Result<Value, ParseError> {
        let s = s.trim();
        if let Some(hex) = s.strip_prefix("0x") {
            return u64::from_str_radix(hex, 16)
                .map_err(|_| ParseError::InvalidValue(s.to_string()));
        }
        s.parse::<u64>()
            .map_err(|_| ParseError::InvalidValue(s.to_string()))
    }

    fn parse_reg(s: &str) -> Result<RegId, ParseError> {
        let s = s.trim();
        // r0, r1, ... or R0, R1, ...
        if let Some(rest) = s.strip_prefix('r').or_else(|| s.strip_prefix('R')) {
            return rest
                .parse::<usize>()
                .map_err(|_| ParseError::MissingField(format!("bad register: {}", s)));
        }
        Err(ParseError::MissingField(format!("bad register: {}", s)))
    }

    fn parse_ordering_str(s: &str) -> Result<Ordering, ParseError> {
        match s.to_lowercase().as_str() {
            "relaxed" | "rlx" => Ok(Ordering::Relaxed),
            "acquire" | "acq" => Ok(Ordering::Acquire),
            "release" | "rel" => Ok(Ordering::Release),
            "acqrel" | "acq_rel" => Ok(Ordering::AcqRel),
            "seqcst" | "seq_cst" | "sc" => Ok(Ordering::SeqCst),
            "acquire_cta" | "acq_cta" => Ok(Ordering::AcquireCTA),
            "release_cta" | "rel_cta" => Ok(Ordering::ReleaseCTA),
            "acquire_gpu" | "acq_gpu" => Ok(Ordering::AcquireGPU),
            "release_gpu" | "rel_gpu" => Ok(Ordering::ReleaseGPU),
            "acquire_system" | "acq_sys" => Ok(Ordering::AcquireSystem),
            "release_system" | "rel_sys" => Ok(Ordering::ReleaseSystem),
            _ => Err(ParseError::UnknownOrdering(s.to_string())),
        }
    }

    fn parse_scope_str(s: &str) -> Scope {
        match s.to_lowercase().as_str() {
            "cta" | "workgroup" | "block" => Scope::CTA,
            "gpu" | "device" => Scope::GPU,
            "system" | "sys" => Scope::System,
            _ => Scope::None,
        }
    }

    // ---- Instruction parsers per format -----------------------------------

    fn parse_instruction_pseudocode(&self, line: &str) -> Result<Instruction, ParseError> {
        let line = line.trim().trim_end_matches(';');
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            return Err(ParseError::UnknownInstruction(line.to_string()));
        }

        match parts[0].to_lowercase().as_str() {
            "load" | "read" | "ld" => {
                // load r0 x relaxed
                if parts.len() < 3 {
                    return Err(ParseError::UnknownInstruction(line.to_string()));
                }
                let reg = Self::parse_reg(parts[1])?;
                let addr = Self::parse_address(parts[2])?;
                let ordering = if parts.len() > 3 {
                    Self::parse_ordering_str(parts[3])?
                } else {
                    Ordering::Relaxed
                };
                Ok(Instruction::Load { reg, addr, ordering })
            }
            "store" | "write" | "st" => {
                // store x 1 release
                if parts.len() < 3 {
                    return Err(ParseError::UnknownInstruction(line.to_string()));
                }
                let addr = Self::parse_address(parts[1])?;
                let value = Self::parse_value(parts[2])?;
                let ordering = if parts.len() > 3 {
                    Self::parse_ordering_str(parts[3])?
                } else {
                    Ordering::Relaxed
                };
                Ok(Instruction::Store {
                    addr,
                    value,
                    ordering,
                })
            }
            "fence" | "barrier" | "mfence" | "dmb" | "sync" => {
                let ordering = if parts.len() > 1 {
                    Self::parse_ordering_str(parts[1])?
                } else {
                    Ordering::SeqCst
                };
                let scope = if parts.len() > 2 {
                    Self::parse_scope_str(parts[2])
                } else {
                    Scope::None
                };
                Ok(Instruction::Fence { ordering, scope })
            }
            "rmw" | "cas" | "exchange" | "xchg" => {
                // rmw r0 x 1 acq_rel
                if parts.len() < 4 {
                    return Err(ParseError::UnknownInstruction(line.to_string()));
                }
                let reg = Self::parse_reg(parts[1])?;
                let addr = Self::parse_address(parts[2])?;
                let value = Self::parse_value(parts[3])?;
                let ordering = if parts.len() > 4 {
                    Self::parse_ordering_str(parts[4])?
                } else {
                    Ordering::AcqRel
                };
                Ok(Instruction::RMW {
                    reg,
                    addr,
                    value,
                    ordering,
                })
            }
            "branch" | "br" | "jmp" | "goto" => {
                if parts.len() < 2 {
                    return Err(ParseError::UnknownInstruction(line.to_string()));
                }
                let label = parts[1]
                    .parse::<usize>()
                    .map_err(|_| ParseError::UnknownInstruction(line.to_string()))?;
                Ok(Instruction::Branch { label })
            }
            "label" => {
                if parts.len() < 2 {
                    return Err(ParseError::UnknownInstruction(line.to_string()));
                }
                let id = parts[1]
                    .trim_end_matches(':')
                    .parse::<usize>()
                    .map_err(|_| ParseError::UnknownInstruction(line.to_string()))?;
                Ok(Instruction::Label { id })
            }
            "beq" | "bne" | "branchcond" => {
                // beq r0 0 label
                if parts.len() < 4 {
                    return Err(ParseError::UnknownInstruction(line.to_string()));
                }
                let reg = Self::parse_reg(parts[1])?;
                let expected = Self::parse_value(parts[2])?;
                let label = parts[3]
                    .parse::<usize>()
                    .map_err(|_| ParseError::UnknownInstruction(line.to_string()))?;
                Ok(Instruction::BranchCond {
                    reg,
                    expected,
                    label,
                })
            }
            _ => Err(ParseError::UnknownInstruction(line.to_string())),
        }
    }

    fn parse_instruction_c_like(&self, line: &str) -> Result<Instruction, ParseError> {
        let line = line.trim().trim_end_matches(';');

        // atomic_load_explicit(&x, memory_order_acquire) -> r0
        // atomic_store_explicit(&x, 1, memory_order_release)
        // atomic_thread_fence(memory_order_seq_cst)
        // r0 = atomic_load(&x, relaxed)
        // store(&x, 1, release)

        // Simple assignment-style: r0 = load(x, acq)
        if let Some((lhs, rhs)) = line.split_once('=') {
            let lhs = lhs.trim();
            let rhs = rhs.trim();

            if let Ok(reg) = Self::parse_reg(lhs) {
                // Parse the RHS as a function call
                if let Some(call) = rhs.strip_prefix("load").or_else(|| rhs.strip_prefix("atomic_load")) {
                    let args = Self::extract_call_args(call);
                    if args.is_empty() {
                        return Err(ParseError::UnknownInstruction(line.to_string()));
                    }
                    let addr = Self::parse_address(args[0])?;
                    let ordering = if args.len() > 1 {
                        Self::parse_c_ordering(args[1])?
                    } else {
                        Ordering::Relaxed
                    };
                    return Ok(Instruction::Load { reg, addr, ordering });
                }
                if let Some(call) = rhs.strip_prefix("rmw")
                    .or_else(|| rhs.strip_prefix("cas"))
                    .or_else(|| rhs.strip_prefix("exchange"))
                {
                    let args = Self::extract_call_args(call);
                    if args.len() < 2 {
                        return Err(ParseError::UnknownInstruction(line.to_string()));
                    }
                    let addr = Self::parse_address(args[0])?;
                    let value = Self::parse_value(args[1])?;
                    let ordering = if args.len() > 2 {
                        Self::parse_c_ordering(args[2])?
                    } else {
                        Ordering::AcqRel
                    };
                    return Ok(Instruction::RMW { reg, addr, value, ordering });
                }
            }
        }

        // store(x, 1, release)
        if let Some(call) = line
            .strip_prefix("store")
            .or_else(|| line.strip_prefix("atomic_store"))
        {
            let args = Self::extract_call_args(call);
            if args.len() < 2 {
                return Err(ParseError::UnknownInstruction(line.to_string()));
            }
            let addr = Self::parse_address(args[0])?;
            let value = Self::parse_value(args[1])?;
            let ordering = if args.len() > 2 {
                Self::parse_c_ordering(args[2])?
            } else {
                Ordering::Relaxed
            };
            return Ok(Instruction::Store {
                addr,
                value,
                ordering,
            });
        }

        // fence / atomic_thread_fence
        if let Some(call) = line
            .strip_prefix("fence")
            .or_else(|| line.strip_prefix("atomic_thread_fence"))
        {
            let args = Self::extract_call_args(call);
            let ordering = if !args.is_empty() {
                Self::parse_c_ordering(args[0])?
            } else {
                Ordering::SeqCst
            };
            let scope = if args.len() > 1 {
                Self::parse_scope_str(args[1])
            } else {
                Scope::None
            };
            return Ok(Instruction::Fence { ordering, scope });
        }

        Err(ParseError::UnknownInstruction(line.to_string()))
    }

    fn parse_instruction_asm(&self, line: &str) -> Result<Instruction, ParseError> {
        let line = line.trim();
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            return Err(ParseError::UnknownInstruction(line.to_string()));
        }

        // Label: "L0:" / "0:"
        if parts[0].ends_with(':') {
            let label_s = parts[0].trim_end_matches(':');
            if let Ok(id) = label_s.trim_start_matches('L').parse::<usize>() {
                return Ok(Instruction::Label { id });
            }
        }

        let mnemonic = parts[0].to_uppercase();
        match mnemonic.as_str() {
            "LD" | "LDR" | "LW" | "LD.ACQUIRE" | "LDAR" | "LD.ACQ" => {
                if parts.len() < 3 {
                    return Err(ParseError::UnknownInstruction(line.to_string()));
                }
                let reg = Self::parse_reg(parts[1].trim_end_matches(','))?;
                let addr = Self::parse_address(parts[2].trim_end_matches(','))?;
                let ordering = Self::infer_asm_ordering(&mnemonic, true);
                Ok(Instruction::Load { reg, addr, ordering })
            }
            "ST" | "STR" | "SW" | "ST.RELEASE" | "STLR" | "ST.REL" => {
                if parts.len() < 3 {
                    return Err(ParseError::UnknownInstruction(line.to_string()));
                }
                let addr = Self::parse_address(parts[1].trim_end_matches(','))?;
                let value = Self::parse_value(parts[2].trim_end_matches(','))?;
                let ordering = Self::infer_asm_ordering(&mnemonic, false);
                Ok(Instruction::Store { addr, value, ordering })
            }
            "FENCE" | "MFENCE" | "DMB" | "DSB" | "SYNC" | "MEMBAR" => {
                let ordering = if parts.len() > 1 {
                    Self::parse_ordering_str(parts[1]).unwrap_or(Ordering::SeqCst)
                } else {
                    Ordering::SeqCst
                };
                let scope = if parts.len() > 2 {
                    Self::parse_scope_str(parts[2])
                } else {
                    Scope::None
                };
                Ok(Instruction::Fence { ordering, scope })
            }
            "XCHG" | "CAS" | "AMO" | "LDXR" | "STXR" => {
                if parts.len() < 4 {
                    return Err(ParseError::UnknownInstruction(line.to_string()));
                }
                let reg = Self::parse_reg(parts[1].trim_end_matches(','))?;
                let addr = Self::parse_address(parts[2].trim_end_matches(','))?;
                let value = Self::parse_value(parts[3].trim_end_matches(','))?;
                let ordering = if parts.len() > 4 {
                    Self::parse_ordering_str(parts[4]).unwrap_or(Ordering::AcqRel)
                } else {
                    Ordering::AcqRel
                };
                Ok(Instruction::RMW { reg, addr, value, ordering })
            }
            "B" | "BR" | "JMP" | "J" => {
                if parts.len() < 2 {
                    return Err(ParseError::UnknownInstruction(line.to_string()));
                }
                let label = parts[1]
                    .trim_start_matches('L')
                    .parse::<usize>()
                    .map_err(|_| ParseError::UnknownInstruction(line.to_string()))?;
                Ok(Instruction::Branch { label })
            }
            "BEQ" | "BNE" | "CBZ" | "CBNZ" => {
                if parts.len() < 4 {
                    return Err(ParseError::UnknownInstruction(line.to_string()));
                }
                let reg = Self::parse_reg(parts[1].trim_end_matches(','))?;
                let expected = Self::parse_value(parts[2].trim_end_matches(','))?;
                let label = parts[3]
                    .trim_start_matches('L')
                    .parse::<usize>()
                    .map_err(|_| ParseError::UnknownInstruction(line.to_string()))?;
                Ok(Instruction::BranchCond { reg, expected, label })
            }
            _ => Err(ParseError::UnknownInstruction(line.to_string())),
        }
    }

    fn infer_asm_ordering(mnemonic: &str, is_load: bool) -> Ordering {
        let upper = mnemonic.to_uppercase();
        if upper.contains("ACQUIRE") || upper.contains("ACQ") || upper == "LDAR" {
            Ordering::Acquire
        } else if upper.contains("RELEASE") || upper.contains("REL") || upper == "STLR" {
            Ordering::Release
        } else if is_load {
            Ordering::Relaxed
        } else {
            Ordering::Relaxed
        }
    }

    fn extract_call_args(call: &str) -> Vec<&str> {
        let call = call.trim();
        let inner = if call.starts_with('(') && call.ends_with(')') {
            &call[1..call.len() - 1]
        } else if let Some(start) = call.find('(') {
            let end = call.rfind(')').unwrap_or(call.len());
            &call[start + 1..end]
        } else {
            return Vec::new();
        };
        inner.split(',').map(|s| s.trim()).collect()
    }

    fn parse_c_ordering(s: &str) -> Result<Ordering, ParseError> {
        let s = s.trim();
        // Strip memory_order_ prefix if present
        let core = s
            .strip_prefix("memory_order_")
            .or_else(|| s.strip_prefix("std::memory_order::"))
            .unwrap_or(s);
        Self::parse_ordering_str(core)
    }

    fn parse_outcome_line(
        &self,
        line: &str,
    ) -> Result<(Outcome, LitmusOutcome), ParseError> {
        // Format: "r0=1, r1=0 -> Forbidden"  or  "r0=1 /\ r1=0 : Forbidden"
        let (assigns_part, class_part) = if let Some((a, c)) = line.split_once("->") {
            (a.trim(), c.trim())
        } else if let Some((a, c)) = line.split_once(':') {
            (a.trim(), c.trim())
        } else {
            (line, "Allowed")
        };

        let classification = match class_part.to_lowercase().as_str() {
            "allowed" | "observable" => LitmusOutcome::Allowed,
            "forbidden" | "not observable" => LitmusOutcome::Forbidden,
            "required" => LitmusOutcome::Required,
            _ => LitmusOutcome::Allowed,
        };

        let mut registers: HashMap<(usize, RegId), Value> = HashMap::new();
        let mut memory: HashMap<Address, Value> = HashMap::new();

        let separators = ["/\\", ",", "&&", "and", "&"];
        let mut assigns_str = assigns_part.to_string();
        for sep in &separators {
            assigns_str = assigns_str.replace(sep, "\n");
        }

        for assign in assigns_str.lines() {
            let assign = assign.trim();
            if assign.is_empty() {
                continue;
            }
            if let Some((lhs, rhs)) = assign.split_once('=') {
                let lhs = lhs.trim();
                let rhs = rhs.trim();
                let val = Self::parse_value(rhs).unwrap_or(0);

                // Thread-qualified register: 0:r0 or T0:r0
                if let Some((tid_s, reg_s)) = lhs.split_once(':') {
                    let tid_s = tid_s.trim().trim_start_matches('T').trim_start_matches('t');
                    if let (Ok(tid), Ok(reg)) =
                        (tid_s.parse::<usize>(), Self::parse_reg(reg_s.trim()))
                    {
                        registers.insert((tid, reg), val);
                        continue;
                    }
                }

                // Plain register
                if let Ok(reg) = Self::parse_reg(lhs) {
                    registers.insert((0, reg), val);
                    continue;
                }

                // Memory location
                if let Ok(addr) = Self::parse_address(lhs) {
                    memory.insert(addr, val);
                }
            }
        }

        Ok((Outcome { registers, memory }, classification))
    }
}

// ---------------------------------------------------------------------------
// Pre-built prompt library
// ---------------------------------------------------------------------------

/// Curated shot examples for common litmus test patterns.
pub struct BuiltinExamples;

impl BuiltinExamples {
    /// Store-Buffering (SB) example.
    pub fn sb() -> ShotExample {
        ShotExample::new(
            "Store-Buffering: two threads write then read different locations",
            "Generate a Store-Buffering (SB) litmus test with 2 threads and relaxed memory ordering.",
            "test SB\n\
             init x = 0\n\
             init y = 0\n\n\
             Thread 0:\n  \
               store x 1 relaxed\n  \
               load r0 y relaxed\n\n\
             Thread 1:\n  \
               store y 1 relaxed\n  \
               load r0 x relaxed\n\n\
             outcome: 0:r0=0, 1:r0=0 -> Allowed",
            "SB",
        )
    }

    /// Message-Passing (MP) example.
    pub fn mp() -> ShotExample {
        ShotExample::new(
            "Message-Passing: writer publishes data then flag; reader checks flag then data",
            "Generate a Message-Passing (MP) litmus test with release/acquire ordering.",
            "test MP\n\
             init x = 0\n\
             init y = 0\n\n\
             Thread 0:\n  \
               store x 1 relaxed\n  \
               store y 1 release\n\n\
             Thread 1:\n  \
               load r0 y acquire\n  \
               load r1 x relaxed\n\n\
             outcome: 1:r0=1, 1:r1=0 -> Forbidden",
            "MP",
        )
    }

    /// Independent-Reads-Independent-Writes (IRIW) example.
    pub fn iriw() -> ShotExample {
        ShotExample::new(
            "IRIW: four threads test multi-copy atomicity",
            "Generate an IRIW litmus test that tests multi-copy atomicity with 4 threads.",
            "test IRIW\n\
             init x = 0\n\
             init y = 0\n\n\
             Thread 0:\n  \
               store x 1 relaxed\n\n\
             Thread 1:\n  \
               store y 1 relaxed\n\n\
             Thread 2:\n  \
               load r0 x acquire\n  \
               load r1 y acquire\n\n\
             Thread 3:\n  \
               load r0 y acquire\n  \
               load r1 x acquire\n\n\
             outcome: 2:r0=1, 2:r1=0, 3:r0=1, 3:r1=0 -> Forbidden",
            "IRIW",
        )
    }

    /// Load-Buffering (LB) example.
    pub fn lb() -> ShotExample {
        ShotExample::new(
            "Load-Buffering: two threads read then write different locations",
            "Generate a Load-Buffering (LB) litmus test with relaxed ordering.",
            "test LB\n\
             init x = 0\n\
             init y = 0\n\n\
             Thread 0:\n  \
               load r0 x relaxed\n  \
               store y 1 relaxed\n\n\
             Thread 1:\n  \
               load r0 y relaxed\n  \
               store x 1 relaxed\n\n\
             outcome: 0:r0=1, 1:r0=1 -> Allowed",
            "LB",
        )
    }

    /// Write-Read-Write (WRW / 2+2W) example.
    pub fn two_plus_two_w() -> ShotExample {
        ShotExample::new(
            "2+2W: coherence test with two threads each writing two locations",
            "Generate a 2+2W coherence test to test write ordering.",
            "test 2+2W\n\
             init x = 0\n\
             init y = 0\n\n\
             Thread 0:\n  \
               store x 1 relaxed\n  \
               store y 2 relaxed\n\n\
             Thread 1:\n  \
               store y 1 relaxed\n  \
               store x 2 relaxed\n\n\
             outcome: x=1, y=1 -> Allowed",
            "2+2W",
        )
    }

    /// Return all built-in examples.
    pub fn all() -> Vec<ShotExample> {
        vec![
            Self::sb(),
            Self::mp(),
            Self::iriw(),
            Self::lb(),
            Self::two_plus_two_w(),
        ]
    }

    /// Select examples relevant to a pattern.
    pub fn for_pattern(pattern: &str) -> Vec<ShotExample> {
        let p = pattern.to_uppercase();
        let mut result = Vec::new();
        let all = Self::all();

        // Always include the exact match first
        for ex in &all {
            if ex.pattern_tag.to_uppercase() == p {
                result.push(ex.clone());
            }
        }
        // Then add related examples
        for ex in &all {
            if ex.pattern_tag.to_uppercase() != p {
                result.push(ex.clone());
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Chain-of-thought builder
// ---------------------------------------------------------------------------

/// Decomposes complex pattern requests into chain-of-thought steps.
struct ChainOfThoughtBuilder;

impl ChainOfThoughtBuilder {
    fn build_instructions(request: &GenerationRequest) -> String {
        let mut steps = Vec::new();

        steps.push(format!(
            "Step 1: Identify the key property of the {} pattern. \
             What ordering anomaly does it demonstrate?",
            request.target_pattern
        ));

        steps.push(format!(
            "Step 2: Determine the minimum number of threads needed. \
             The {} memory model may affect the required structure.",
            request.memory_model
        ));

        steps.push(
            "Step 3: For each thread, list the memory operations in order. \
             Specify the memory ordering for each operation."
                .to_string(),
        );

        steps.push(
            "Step 4: Define the initial state (all shared locations and their values).".to_string(),
        );

        steps.push(
            "Step 5: Specify the expected outcome and whether it is \
             Allowed, Forbidden, or Required under the target model."
                .to_string(),
        );

        if !request.constraints.is_empty() {
            let constraint_list: Vec<String> =
                request.constraints.iter().map(|c| format!("  - {}", c)).collect();
            steps.push(format!(
                "Step 6: Verify your test satisfies these constraints:\n{}",
                constraint_list.join("\n")
            ));
        }

        let mut out = String::from(
            "Please think through this step-by-step before writing the litmus test:\n\n",
        );
        for s in &steps {
            out.push_str(s);
            out.push_str("\n\n");
        }
        out.push_str("After reasoning through each step, output the litmus test.\n");
        out
    }
}

// ---------------------------------------------------------------------------
// Prompt engine
// ---------------------------------------------------------------------------

/// Main engine that builds prompts and parses LLM responses.
#[derive(Debug, Clone)]
pub struct LlmPromptEngine {
    pub config: PromptConfig,
    parser: LitmusTestParser,
    custom_examples: Vec<ShotExample>,
}

impl LlmPromptEngine {
    pub fn new(config: PromptConfig) -> Self {
        let parser = LitmusTestParser {
            format_hint: Some(config.output_format),
            lenient: true,
        };
        Self {
            config,
            parser,
            custom_examples: Vec::new(),
        }
    }

    /// Add a custom example to the shot library.
    pub fn add_example(&mut self, example: ShotExample) {
        self.custom_examples.push(example);
    }

    /// Build a complete prompt from a generation request.
    pub fn build_prompt(&self, request: &GenerationRequest) -> String {
        let template = self.build_template(request);
        template.render_flat()
    }

    /// Build a structured prompt template.
    pub fn build_template(&self, request: &GenerationRequest) -> PromptTemplate {
        let system_prompt = self.build_system_prompt(request);
        let examples = self.select_examples(request);
        let user_prompt = self.build_user_prompt(request);
        let cot = if self.config.chain_of_thought {
            Some(ChainOfThoughtBuilder::build_instructions(request))
        } else {
            None
        };
        let output_instructions = self.build_output_format_instructions();

        PromptTemplate {
            system_prompt,
            examples,
            user_prompt,
            chain_of_thought_instructions: cot,
            output_format_instructions: output_instructions,
        }
    }

    /// Build chat-style messages.
    pub fn build_chat_messages(&self, request: &GenerationRequest) -> Vec<ChatMessage> {
        self.build_template(request).render_chat_messages()
    }

    /// Parse an LLM response into a `LitmusTest`.
    pub fn parse_response(&self, response: &str) -> Result<LitmusTest, ParseError> {
        self.parser.parse(response)
    }

    /// Build a refinement prompt after a failed parse.
    pub fn build_refinement_prompt(
        &self,
        request: &GenerationRequest,
        previous_response: &str,
        error: &ParseError,
    ) -> String {
        let mut prompt = String::new();
        prompt.push_str("Your previous response had an error. Please fix it.\n\n");
        prompt.push_str(&format!("Error: {}\n\n", error));
        prompt.push_str("Previous response:\n```\n");
        prompt.push_str(previous_response);
        prompt.push_str("\n```\n\n");
        prompt.push_str("Please regenerate the litmus test, fixing the error above.\n");
        prompt.push_str(&self.build_output_format_instructions());
        prompt.push_str("\n\n");
        prompt.push_str(&self.build_user_prompt(request));
        prompt
    }

    /// Build a refinement prompt after a validation failure.
    pub fn build_constraint_refinement_prompt(
        &self,
        request: &GenerationRequest,
        previous_response: &str,
        violations: &[String],
    ) -> String {
        let mut prompt = String::new();
        prompt.push_str(
            "Your previous litmus test did not satisfy all constraints. Please fix it.\n\n",
        );
        prompt.push_str("Constraint violations:\n");
        for v in violations {
            prompt.push_str(&format!("  - {}\n", v));
        }
        prompt.push_str("\nPrevious response:\n```\n");
        prompt.push_str(previous_response);
        prompt.push_str("\n```\n\n");
        prompt.push_str("Please regenerate the litmus test, addressing all violations.\n");
        prompt.push_str(&self.build_user_prompt(request));
        prompt
    }

    // ---- Private helpers --------------------------------------------------

    fn build_system_prompt(&self, request: &GenerationRequest) -> String {
        format!(
            "You are an expert in concurrent programming and memory consistency models.\n\
             You generate litmus tests—small concurrent programs that test specific \
             memory ordering behaviors.\n\n\
             Target memory model: {model}\n\
             You must produce syntactically correct litmus tests in {fmt} format.\n\
             Each test has: a name, initial state, per-thread instructions, and expected outcomes.\n\
             \n\
             Rules:\n\
             - Every shared variable must appear in the initial state.\n\
             - Each instruction specifies its memory ordering explicitly.\n\
             - Outcomes list register/memory values and whether the outcome is Allowed, \
               Forbidden, or Required.\n\
             - Be precise about orderings: relaxed, acquire, release, acq_rel, seq_cst.\n\
             - For GPU models, also use scope-qualified orderings (e.g., acquire_cta).\n",
            model = request.memory_model,
            fmt = self.config.output_format,
        )
    }

    fn build_user_prompt(&self, request: &GenerationRequest) -> String {
        let mut prompt = format!(
            "Generate a {} litmus test for the {} memory model.",
            request.target_pattern, request.memory_model
        );

        if let Some(ref desc) = request.description {
            prompt.push_str(&format!("\n\nAdditional context: {}", desc));
        }

        if let Some(n) = request.num_threads {
            prompt.push_str(&format!("\nUse exactly {} threads.", n));
        }

        if let Some(n) = request.max_instructions_per_thread {
            prompt.push_str(&format!("\nAt most {} instructions per thread.", n));
        }

        if !request.constraints.is_empty() {
            prompt.push_str("\n\nConstraints:");
            for c in &request.constraints {
                prompt.push_str(&format!("\n  - {}", c));
            }
        }

        prompt
    }

    fn build_output_format_instructions(&self) -> String {
        match self.config.output_format {
            OutputFormat::Pseudocode => {
                "Output format (pseudocode):\n\
                 ```\n\
                 test <NAME>\n\
                 init <var> = <val>\n\
                 ...\n\n\
                 Thread <N>:\n  \
                   <instruction> [ordering]\n  \
                   ...\n\n\
                 outcome: <var>=<val>, ... -> Allowed|Forbidden|Required\n\
                 ```\n\
                 Instructions: load <reg> <addr> [ordering], store <addr> <val> [ordering], \
                 fence [ordering] [scope], rmw <reg> <addr> <val> [ordering]\n\
                 Orderings: relaxed, acquire, release, acq_rel, seq_cst\n"
                    .to_string()
            }
            OutputFormat::CLike => {
                "Output format (C-like):\n\
                 ```\n\
                 // <NAME>\n\
                 x = 0;\n\
                 y = 0;\n\n\
                 void thread0() {\n  \
                   r0 = load(x, acquire);\n  \
                   store(y, 1, release);\n\
                 }\n\n\
                 outcome: <var>=<val>, ... -> Allowed|Forbidden|Required\n\
                 ```\n"
                    .to_string()
            }
            OutputFormat::AssemblyLike => {
                "Output format (assembly-like):\n\
                 ```\n\
                 ; <NAME>\n\
                 .init x = 0\n\
                 .init y = 0\n\n\
                 P0:\n  \
                   LD r0, x\n  \
                   ST y, 1\n\n\
                 outcome: <var>=<val>, ... -> Allowed|Forbidden|Required\n\
                 ```\n\
                 Mnemonics: LD, ST, FENCE, XCHG; LD.ACQUIRE, ST.RELEASE for ordered ops\n"
                    .to_string()
            }
        }
    }

    fn select_examples(&self, request: &GenerationRequest) -> Vec<ShotExample> {
        let mut examples = Vec::new();

        // Custom examples first
        for ex in &self.custom_examples {
            if ex.pattern_tag.to_uppercase() == request.target_pattern.to_uppercase() {
                examples.push(ex.clone());
            }
        }

        // Then built-in, pattern-relevant
        let builtin = BuiltinExamples::for_pattern(&request.target_pattern);
        for ex in builtin {
            if examples.len() >= self.config.num_shots {
                break;
            }
            // Avoid duplicates
            if !examples.iter().any(|e| e.pattern_tag == ex.pattern_tag) {
                examples.push(ex);
            }
        }

        examples.truncate(self.config.num_shots);
        examples
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_request() -> GenerationRequest {
        GenerationRequest {
            target_pattern: "MP".to_string(),
            memory_model: "TSO".to_string(),
            constraints: vec![],
            description: None,
            num_threads: Some(2),
            max_instructions_per_thread: Some(4),
        }
    }

    // -- PromptConfig -------------------------------------------------------

    #[test]
    fn test_default_config() {
        let cfg = PromptConfig::default();
        assert_eq!(cfg.model_name, "gpt-4");
        assert_eq!(cfg.num_shots, 3);
        assert!(cfg.chain_of_thought);
    }

    // -- ShotExample --------------------------------------------------------

    #[test]
    fn test_shot_example_render() {
        let ex = BuiltinExamples::sb();
        let rendered = ex.render();
        assert!(rendered.contains("Store-Buffering"));
        assert!(rendered.contains("SB"));
    }

    // -- BuiltinExamples ----------------------------------------------------

    #[test]
    fn test_builtin_all() {
        let all = BuiltinExamples::all();
        assert_eq!(all.len(), 5);
    }

    #[test]
    fn test_builtin_for_pattern_mp() {
        let exs = BuiltinExamples::for_pattern("MP");
        assert_eq!(exs[0].pattern_tag, "MP");
    }

    #[test]
    fn test_builtin_for_pattern_unknown() {
        let exs = BuiltinExamples::for_pattern("UNKNOWN");
        // Returns all examples even if no direct match
        assert!(!exs.is_empty());
    }

    // -- Prompt building ----------------------------------------------------

    #[test]
    fn test_build_prompt_contains_pattern() {
        let engine = LlmPromptEngine::new(PromptConfig::default());
        let req = default_request();
        let prompt = engine.build_prompt(&req);
        assert!(prompt.contains("MP"));
        assert!(prompt.contains("TSO"));
    }

    #[test]
    fn test_build_prompt_with_constraints() {
        let engine = LlmPromptEngine::new(PromptConfig::default());
        let req = GenerationRequest {
            target_pattern: "SB".to_string(),
            memory_model: "ARMv8".to_string(),
            constraints: vec![
                GenerationConstraint::MaxThreads(2),
                GenerationConstraint::RequireOrdering(Ordering::Acquire),
            ],
            description: Some("Test store buffer forwarding".to_string()),
            num_threads: None,
            max_instructions_per_thread: None,
        };
        let prompt = engine.build_prompt(&req);
        assert!(prompt.contains("at most 2 threads"));
        assert!(prompt.contains("Acquire"));
    }

    #[test]
    fn test_build_prompt_chain_of_thought() {
        let mut cfg = PromptConfig::default();
        cfg.chain_of_thought = true;
        let engine = LlmPromptEngine::new(cfg);
        let prompt = engine.build_prompt(&default_request());
        assert!(prompt.contains("Step 1"));
        assert!(prompt.contains("Step 5"));
    }

    #[test]
    fn test_build_prompt_no_cot() {
        let mut cfg = PromptConfig::default();
        cfg.chain_of_thought = false;
        let engine = LlmPromptEngine::new(cfg);
        let prompt = engine.build_prompt(&default_request());
        assert!(!prompt.contains("Step 1"));
    }

    #[test]
    fn test_build_chat_messages() {
        let engine = LlmPromptEngine::new(PromptConfig::default());
        let msgs = engine.build_chat_messages(&default_request());
        // System + (num_shots * 2 user/assistant) + 1 user
        assert!(msgs.len() >= 2);
        assert_eq!(msgs[0].role, ChatRole::System);
        assert_eq!(msgs.last().unwrap().role, ChatRole::User);
    }

    #[test]
    fn test_num_shots_limits_examples() {
        let mut cfg = PromptConfig::default();
        cfg.num_shots = 1;
        let engine = LlmPromptEngine::new(cfg);
        let template = engine.build_template(&default_request());
        assert!(template.examples.len() <= 1);
    }

    #[test]
    fn test_custom_example_priority() {
        let mut cfg = PromptConfig::default();
        cfg.num_shots = 1;
        let mut engine = LlmPromptEngine::new(cfg);
        engine.add_example(ShotExample::new(
            "Custom MP",
            "custom input",
            "custom output",
            "MP",
        ));
        let template = engine.build_template(&default_request());
        assert_eq!(template.examples[0].description, "Custom MP");
    }

    // -- Parser: pseudocode -------------------------------------------------

    #[test]
    fn test_parse_pseudocode_sb() {
        let parser = LitmusTestParser::new(OutputFormat::Pseudocode);
        let input = "\
test SB
init x = 0
init y = 0

Thread 0:
  store x 1 relaxed
  load r0 y relaxed

Thread 1:
  store y 1 relaxed
  load r0 x relaxed

outcome: 0:r0=0, 1:r0=0 -> Allowed";

        let test = parser.parse(input).unwrap();
        assert_eq!(test.name, "SB");
        assert_eq!(test.threads.len(), 2);
        assert_eq!(test.threads[0].instructions.len(), 2);
        assert_eq!(test.initial_state.len(), 2);
        assert_eq!(test.expected_outcomes.len(), 1);
        assert_eq!(test.expected_outcomes[0].1, LitmusOutcome::Allowed);
    }

    #[test]
    fn test_parse_pseudocode_mp_forbidden() {
        let parser = LitmusTestParser::new(OutputFormat::Pseudocode);
        let input = "\
test MP
init x = 0
init y = 0

Thread 0:
  store x 1 relaxed
  store y 1 release

Thread 1:
  load r0 y acquire
  load r1 x relaxed

outcome: 1:r0=1, 1:r1=0 -> Forbidden";

        let test = parser.parse(input).unwrap();
        assert_eq!(test.name, "MP");
        assert_eq!(test.expected_outcomes[0].1, LitmusOutcome::Forbidden);
        // Check orderings
        match &test.threads[0].instructions[1] {
            Instruction::Store { ordering, .. } => assert_eq!(*ordering, Ordering::Release),
            _ => panic!("expected store"),
        }
        match &test.threads[1].instructions[0] {
            Instruction::Load { ordering, .. } => assert_eq!(*ordering, Ordering::Acquire),
            _ => panic!("expected load"),
        }
    }

    #[test]
    fn test_parse_pseudocode_fence() {
        let parser = LitmusTestParser::new(OutputFormat::Pseudocode);
        let input = "\
test FenceTest
init x = 0
init y = 0

Thread 0:
  store x 1 relaxed
  fence seq_cst
  load r0 y relaxed";

        let test = parser.parse(input).unwrap();
        match &test.threads[0].instructions[1] {
            Instruction::Fence { ordering, scope } => {
                assert_eq!(*ordering, Ordering::SeqCst);
                assert_eq!(*scope, Scope::None);
            }
            _ => panic!("expected fence"),
        }
    }

    #[test]
    fn test_parse_pseudocode_rmw() {
        let parser = LitmusTestParser::new(OutputFormat::Pseudocode);
        let input = "\
test RMWTest
init x = 0

Thread 0:
  rmw r0 x 1 acq_rel";

        let test = parser.parse(input).unwrap();
        match &test.threads[0].instructions[0] {
            Instruction::RMW {
                reg,
                addr,
                value,
                ordering,
            } => {
                assert_eq!(*reg, 0);
                assert_eq!(*addr, 23); // 'x' -> 23
                assert_eq!(*value, 1);
                assert_eq!(*ordering, Ordering::AcqRel);
            }
            _ => panic!("expected rmw"),
        }
    }

    #[test]
    fn test_parse_pseudocode_branch() {
        let parser = LitmusTestParser::new(OutputFormat::Pseudocode);
        let input = "\
test BranchTest
init x = 0

Thread 0:
  load r0 x relaxed
  beq r0 0 1
  store x 1 relaxed
  label 1";

        let test = parser.parse(input).unwrap();
        assert_eq!(test.threads[0].instructions.len(), 4);
        match &test.threads[0].instructions[1] {
            Instruction::BranchCond {
                reg,
                expected,
                label,
            } => {
                assert_eq!(*reg, 0);
                assert_eq!(*expected, 0);
                assert_eq!(*label, 1);
            }
            _ => panic!("expected branch cond"),
        }
    }

    // -- Parser: code block extraction --------------------------------------

    #[test]
    fn test_parse_extracts_code_block() {
        let parser = LitmusTestParser::new(OutputFormat::Pseudocode);
        let input = "\
Here is the litmus test:

```
test SB
init x = 0

Thread 0:
  store x 1 relaxed
```

That tests store buffering.";

        let test = parser.parse(input).unwrap();
        assert_eq!(test.name, "SB");
    }

    // -- Parser: C-like -----------------------------------------------------

    #[test]
    fn test_parse_c_like_basic() {
        let parser = LitmusTestParser::new(OutputFormat::CLike);
        let input = "\
void thread0() {
  r0 = load(x, acquire);
  store(y, 1, release);
}

void thread1() {
  r0 = load(y, acquire);
  r1 = load(x, relaxed);
}

outcome: 1:r0=1, 1:r1=0 -> Forbidden";

        let test = parser.parse(input).unwrap();
        assert_eq!(test.threads.len(), 2);
        assert_eq!(test.threads[0].id, 0);
        assert_eq!(test.threads[1].id, 1);
    }

    #[test]
    fn test_parse_c_like_with_init() {
        let parser = LitmusTestParser::new(OutputFormat::CLike);
        let input = "\
x = 0;
y = 0;

void thread0() {
  store(x, 1, relaxed);
}";

        let test = parser.parse(input).unwrap();
        assert_eq!(test.initial_state.len(), 2);
    }

    // -- Parser: assembly-like ----------------------------------------------

    #[test]
    fn test_parse_asm_basic() {
        let parser = LitmusTestParser::new(OutputFormat::AssemblyLike);
        let input = "\
; SB test
.init x = 0
.init y = 0

P0:
  ST x, 1
  LD r0, y

P1:
  ST y, 1
  LD r0, x

outcome: 0:r0=0, 1:r0=0 -> Allowed";

        let test = parser.parse(input).unwrap();
        assert_eq!(test.threads.len(), 2);
        assert_eq!(test.initial_state.len(), 2);
    }

    #[test]
    fn test_parse_asm_acquire_release() {
        let parser = LitmusTestParser::new(OutputFormat::AssemblyLike);
        let input = "\
P0:
  ST.RELEASE x, 1
  LDAR r0, y

P1:
  ST y, 1";

        let test = parser.parse(input).unwrap();
        match &test.threads[0].instructions[0] {
            Instruction::Store { ordering, .. } => assert_eq!(*ordering, Ordering::Release),
            _ => panic!("expected store"),
        }
        match &test.threads[0].instructions[1] {
            Instruction::Load { ordering, .. } => assert_eq!(*ordering, Ordering::Acquire),
            _ => panic!("expected load"),
        }
    }

    #[test]
    fn test_parse_asm_fence() {
        let parser = LitmusTestParser::new(OutputFormat::AssemblyLike);
        let input = "\
P0:
  ST x, 1
  DMB seq_cst
  LD r0, y";

        let test = parser.parse(input).unwrap();
        match &test.threads[0].instructions[1] {
            Instruction::Fence { ordering, .. } => assert_eq!(*ordering, Ordering::SeqCst),
            _ => panic!("expected fence"),
        }
    }

    // -- Parser: error cases ------------------------------------------------

    #[test]
    fn test_parse_empty_returns_error() {
        let parser = LitmusTestParser::default();
        assert!(parser.parse("").is_err());
    }

    #[test]
    fn test_parse_no_threads_returns_error() {
        let parser = LitmusTestParser::default();
        let input = "init x = 0\noutcome: x=0 -> Allowed";
        assert!(parser.parse(input).is_err());
    }

    #[test]
    fn test_parse_strict_rejects_bad_instruction() {
        let parser = LitmusTestParser::strict(OutputFormat::Pseudocode);
        let input = "\
Thread 0:
  store x 1 relaxed
  gobbledygook
  load r0 x relaxed";
        assert!(parser.parse(input).is_err());
    }

    #[test]
    fn test_parse_lenient_skips_bad_instruction() {
        let parser = LitmusTestParser::new(OutputFormat::Pseudocode);
        let input = "\
Thread 0:
  store x 1 relaxed
  gobbledygook
  load r0 x relaxed";
        let test = parser.parse(input).unwrap();
        assert_eq!(test.threads[0].instructions.len(), 2);
    }

    // -- Parser: outcome parsing --------------------------------------------

    #[test]
    fn test_parse_outcome_with_backslash_and() {
        let parser = LitmusTestParser::default();
        let input = "\
Thread 0:
  load r0 x relaxed

Thread 1:
  load r0 y relaxed

outcome: 0:r0=0 /\\ 1:r0=0 -> Forbidden";

        let test = parser.parse(input).unwrap();
        assert_eq!(test.expected_outcomes.len(), 1);
        let (outcome, class) = &test.expected_outcomes[0];
        assert_eq!(*class, LitmusOutcome::Forbidden);
        assert_eq!(outcome.registers.len(), 2);
    }

    // -- Parser: address mapping --------------------------------------------

    #[test]
    fn test_parse_address_named() {
        assert_eq!(LitmusTestParser::parse_address("x").unwrap(), 23);
        assert_eq!(LitmusTestParser::parse_address("a").unwrap(), 0);
        assert_eq!(LitmusTestParser::parse_address("z").unwrap(), 25);
    }

    #[test]
    fn test_parse_address_numeric() {
        assert_eq!(LitmusTestParser::parse_address("42").unwrap(), 42);
        assert_eq!(LitmusTestParser::parse_address("0x10").unwrap(), 16);
    }

    #[test]
    fn test_parse_address_bracketed() {
        assert_eq!(LitmusTestParser::parse_address("[x]").unwrap(), 23);
        assert_eq!(LitmusTestParser::parse_address("*y").unwrap(), 24);
    }

    // -- Refinement prompts -------------------------------------------------

    #[test]
    fn test_refinement_prompt_includes_error() {
        let engine = LlmPromptEngine::new(PromptConfig::default());
        let req = default_request();
        let error = ParseError::MissingField("initial_state".to_string());
        let prompt = engine.build_refinement_prompt(&req, "bad output", &error);
        assert!(prompt.contains("missing field: initial_state"));
        assert!(prompt.contains("bad output"));
    }

    #[test]
    fn test_constraint_refinement_prompt() {
        let engine = LlmPromptEngine::new(PromptConfig::default());
        let req = default_request();
        let prompt = engine.build_constraint_refinement_prompt(
            &req,
            "previous",
            &["too many threads".to_string()],
        );
        assert!(prompt.contains("too many threads"));
    }

    // -- Output format display ----------------------------------------------

    #[test]
    fn test_output_format_display() {
        assert_eq!(format!("{}", OutputFormat::Pseudocode), "pseudocode");
        assert_eq!(format!("{}", OutputFormat::CLike), "C-like");
        assert_eq!(format!("{}", OutputFormat::AssemblyLike), "assembly-like");
    }

    // -- Constraint display -------------------------------------------------

    #[test]
    fn test_constraint_display() {
        let c = GenerationConstraint::MaxThreads(4);
        assert_eq!(format!("{}", c), "at most 4 threads");

        let c = GenerationConstraint::Custom("no fences".to_string());
        assert_eq!(format!("{}", c), "no fences");
    }

    // -- ParseError display -------------------------------------------------

    #[test]
    fn test_parse_error_display() {
        let e = ParseError::NoTestFound;
        assert_eq!(format!("{}", e), "no litmus test found in LLM output");

        let e = ParseError::MultipleErrors(vec![
            ParseError::UnknownInstruction("foo".to_string()),
            ParseError::InvalidAddress("bar".to_string()),
        ]);
        let s = format!("{}", e);
        assert!(s.contains("foo"));
        assert!(s.contains("bar"));
    }

    // -- LlmBackend ---------------------------------------------------------

    #[test]
    fn test_backend_default() {
        let b = LlmBackend::default();
        match b {
            LlmBackend::OpenAI { api_base } => {
                assert!(api_base.contains("openai"));
            }
            _ => panic!("expected OpenAI default"),
        }
    }

    // -- Full round-trip: build prompt, fake response, parse ----------------

    #[test]
    fn test_round_trip_pseudocode() {
        let engine = LlmPromptEngine::new(PromptConfig::default());
        let req = default_request();
        let _prompt = engine.build_prompt(&req);

        // Simulate LLM response
        let response = "\
Here is a Message-Passing litmus test for TSO:

```
test MP_TSO
init x = 0
init y = 0

Thread 0:
  store x 1 relaxed
  store y 1 release

Thread 1:
  load r0 y acquire
  load r1 x relaxed

outcome: 1:r0=1, 1:r1=0 -> Forbidden
```";
        let test = engine.parse_response(response).unwrap();
        assert_eq!(test.name, "MP_TSO");
        assert_eq!(test.threads.len(), 2);
        assert_eq!(test.expected_outcomes[0].1, LitmusOutcome::Forbidden);
    }

    #[test]
    fn test_round_trip_c_like() {
        let mut cfg = PromptConfig::default();
        cfg.output_format = OutputFormat::CLike;
        let engine = LlmPromptEngine::new(cfg);

        let response = "\
```
x = 0;
y = 0;

void thread0() {
  store(x, 1, relaxed);
  store(y, 1, release);
}

void thread1() {
  r0 = load(y, acquire);
  r1 = load(x, relaxed);
}

outcome: 1:r0=1, 1:r1=0 -> Forbidden
```";
        let test = engine.parse_response(response).unwrap();
        assert_eq!(test.threads.len(), 2);
    }

    #[test]
    fn test_round_trip_asm() {
        let mut cfg = PromptConfig::default();
        cfg.output_format = OutputFormat::AssemblyLike;
        let engine = LlmPromptEngine::new(cfg);

        let response = "\
```
.init x = 0
.init y = 0

P0:
  ST x, 1
  ST.RELEASE y, 1

P1:
  LDAR r0, y
  LD r1, x

outcome: 1:r0=1, 1:r1=0 -> Forbidden
```";
        let test = engine.parse_response(response).unwrap();
        assert_eq!(test.threads.len(), 2);
    }
}
