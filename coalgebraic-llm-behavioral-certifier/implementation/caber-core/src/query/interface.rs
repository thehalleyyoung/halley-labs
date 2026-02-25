//! Black-box model query interface for the CABER project.
//!
//! Provides traits, types, and utilities for querying language models
//! in a coalgebraic behavioral certification pipeline. Includes a mock
//! model for deterministic testing and a response analyzer for
//! aggregating sampled completions into probability distributions.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use chrono::Utc;
use rand::Rng;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur when querying a black-box model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryError {
    /// The model returned a rate-limit response.
    RateLimited { retry_after_ms: u64 },
    /// The request was malformed or violated constraints.
    InvalidRequest(String),
    /// An internal model error occurred.
    ModelError(String),
    /// The request timed out.
    Timeout,
    /// A network-level error prevented the request from completing.
    NetworkError(String),
    /// The query budget has been exhausted.
    BudgetExhausted,
}

impl std::fmt::Display for QueryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueryError::RateLimited { retry_after_ms } => {
                write!(f, "rate limited – retry after {} ms", retry_after_ms)
            }
            QueryError::InvalidRequest(msg) => write!(f, "invalid request: {}", msg),
            QueryError::ModelError(msg) => write!(f, "model error: {}", msg),
            QueryError::Timeout => write!(f, "request timed out"),
            QueryError::NetworkError(msg) => write!(f, "network error: {}", msg),
            QueryError::BudgetExhausted => write!(f, "query budget exhausted"),
        }
    }
}

impl std::error::Error for QueryError {}

// ---------------------------------------------------------------------------
// Message types
// ---------------------------------------------------------------------------

/// Role of a participant in a chat conversation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

impl std::fmt::Display for MessageRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MessageRole::System => write!(f, "system"),
            MessageRole::User => write!(f, "user"),
            MessageRole::Assistant => write!(f, "assistant"),
        }
    }
}

/// A single message in a chat conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: MessageRole,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: &str) -> Self {
        Self {
            role: MessageRole::System,
            content: content.to_string(),
        }
    }

    pub fn user(content: &str) -> Self {
        Self {
            role: MessageRole::User,
            content: content.to_string(),
        }
    }

    pub fn assistant(content: &str) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

/// A query to be sent to a black-box language model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelQuery {
    /// Unique identifier for this query (UUID v4).
    pub id: String,
    /// Optional system-level prompt prepended to the conversation.
    pub system_prompt: Option<String>,
    /// The sequence of chat messages forming the conversation context.
    pub messages: Vec<ChatMessage>,
    /// Sampling temperature (0.0 = deterministic, higher = more random).
    pub temperature: f64,
    /// Nucleus-sampling probability mass.
    pub top_p: f64,
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Whether to request per-token log-probabilities.
    pub logprobs: bool,
    /// Number of independent completions to sample.
    pub n_completions: usize,
    /// Sequences that, when generated, cause the model to stop.
    pub stop_sequences: Vec<String>,
    /// ISO-8601 timestamp of query creation.
    pub created_at: String,
}

impl ModelQuery {
    /// Create a new query with sensible defaults.
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            system_prompt: None,
            messages: Vec::new(),
            temperature: 0.7,
            top_p: 1.0,
            max_tokens: 256,
            logprobs: false,
            n_completions: 1,
            stop_sequences: Vec::new(),
            created_at: Utc::now().to_rfc3339(),
        }
    }

    pub fn with_system_prompt(mut self, prompt: &str) -> Self {
        self.system_prompt = Some(prompt.to_string());
        self
    }

    pub fn with_message(mut self, message: ChatMessage) -> Self {
        self.messages.push(message);
        self
    }

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_top_p(mut self, top_p: f64) -> Self {
        self.top_p = top_p;
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn with_logprobs(mut self, logprobs: bool) -> Self {
        self.logprobs = logprobs;
        self
    }

    pub fn with_n_completions(mut self, n: usize) -> Self {
        self.n_completions = n;
        self
    }

    pub fn with_stop_sequence(mut self, seq: &str) -> Self {
        self.stop_sequences.push(seq.to_string());
        self
    }

    /// Total number of tokens across all messages (rough estimate).
    pub fn estimated_prompt_tokens(&self) -> usize {
        let mut total = 0usize;
        if let Some(ref sp) = self.system_prompt {
            total += estimate_tokens(sp);
        }
        for msg in &self.messages {
            total += estimate_tokens(&msg.content);
        }
        total
    }
}

impl Default for ModelQuery {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

/// Reason why the model stopped generating tokens.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    /// The model produced a natural stop token or hit a stop sequence.
    Stop,
    /// The model reached the maximum token limit.
    Length,
    /// A content filter intervened.
    ContentFilter,
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FinishReason::Stop => write!(f, "stop"),
            FinishReason::Length => write!(f, "length"),
            FinishReason::ContentFilter => write!(f, "content_filter"),
        }
    }
}

/// Log-probability information for a single generated token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenLogProb {
    /// The token string.
    pub token: String,
    /// Natural-log probability assigned by the model.
    pub logprob: f64,
    /// The top-k alternative tokens and their log-probabilities.
    pub top_logprobs: Vec<(String, f64)>,
}

/// Token-usage statistics for a single model invocation.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

impl TokenUsage {
    pub fn new(prompt_tokens: usize, completion_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

/// A single completion returned by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Completion {
    /// The generated text.
    pub content: String,
    /// Why the model stopped generating.
    pub finish_reason: FinishReason,
    /// Per-token log-probabilities, if requested.
    pub logprobs: Option<Vec<TokenLogProb>>,
}

/// The full response from a model query, potentially containing multiple
/// completions when `n_completions > 1`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResponse {
    /// The id of the query that produced this response.
    pub query_id: String,
    /// One or more sampled completions.
    pub completions: Vec<Completion>,
    /// Identifier of the model that produced the response.
    pub model_id: String,
    /// Token-usage statistics.
    pub usage: TokenUsage,
    /// Wall-clock latency of the model call in milliseconds.
    pub latency_ms: f64,
    /// ISO-8601 timestamp of response creation.
    pub created_at: String,
}

// ---------------------------------------------------------------------------
// BlackBoxModel trait
// ---------------------------------------------------------------------------

/// Trait abstracting over any language model that can be queried through a
/// request/response interface.  Implementations may wrap HTTP clients for
/// cloud APIs, local inference engines, or deterministic mocks.
pub trait BlackBoxModel: Send + Sync {
    /// Submit a query and block until a response (or error) is available.
    fn query(&self, request: &ModelQuery) -> Result<ModelResponse, QueryError>;

    /// A human-readable identifier for this model (e.g. "gpt-4").
    fn model_id(&self) -> &str;

    /// Whether the model supports returning per-token log-probabilities.
    fn supports_logprobs(&self) -> bool {
        false
    }

    /// Maximum number of tokens the model can accept as context.
    fn max_context_length(&self) -> usize {
        4096
    }
}

// ---------------------------------------------------------------------------
// MockModel
// ---------------------------------------------------------------------------

/// Weighted output entry used by `MockModel` to sample from a distribution.
#[derive(Debug, Clone)]
struct WeightedOutput {
    text: String,
    weight: f64,
}

/// A fully configurable mock model for deterministic and probabilistic
/// testing of query pipelines.
pub struct MockModel {
    id: String,
    /// Exact input → output mapping (keyed on last user message content).
    responses: Mutex<HashMap<String, String>>,
    /// Probabilistic mappings: input → weighted outputs.
    distributions: Mutex<HashMap<String, Vec<WeightedOutput>>>,
    /// Simulated latency range in milliseconds.
    latency_min_ms: f64,
    latency_max_ms: f64,
    /// Probability in [0, 1] that a query returns a `ModelError`.
    error_rate: f64,
    /// Number of queries that have been processed.
    query_count: AtomicUsize,
}

impl std::fmt::Debug for MockModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockModel")
            .field("id", &self.id)
            .field("latency_min_ms", &self.latency_min_ms)
            .field("latency_max_ms", &self.latency_max_ms)
            .field("error_rate", &self.error_rate)
            .field("query_count", &self.query_count.load(Ordering::Relaxed))
            .finish()
    }
}

impl MockModel {
    /// Create a new mock model with the given identifier.
    pub fn new(model_id: &str) -> Self {
        Self {
            id: model_id.to_string(),
            responses: Mutex::new(HashMap::new()),
            distributions: Mutex::new(HashMap::new()),
            latency_min_ms: 0.0,
            latency_max_ms: 0.0,
            error_rate: 0.0,
            query_count: AtomicUsize::new(0),
        }
    }

    /// Register a deterministic response for a given input key.
    pub fn with_response(self, input: &str, output: &str) -> Self {
        self.responses
            .lock()
            .unwrap()
            .insert(input.to_string(), output.to_string());
        self
    }

    /// Register a probabilistic response distribution for a given input key.
    /// Each entry is `(output_text, weight)` where weights need not sum to 1.
    pub fn with_distribution(self, input: &str, outputs: Vec<(&str, f64)>) -> Self {
        let weighted: Vec<WeightedOutput> = outputs
            .into_iter()
            .map(|(text, weight)| WeightedOutput {
                text: text.to_string(),
                weight,
            })
            .collect();
        self.distributions
            .lock()
            .unwrap()
            .insert(input.to_string(), weighted);
        self
    }

    /// Set the simulated latency range.
    pub fn with_latency(mut self, min_ms: f64, max_ms: f64) -> Self {
        self.latency_min_ms = min_ms;
        self.latency_max_ms = max_ms;
        self
    }

    /// Set the probability that any single query returns an error.
    pub fn with_error_rate(mut self, rate: f64) -> Self {
        self.error_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Return the total number of queries processed so far.
    pub fn query_count(&self) -> usize {
        self.query_count.load(Ordering::Relaxed)
    }

    // -- internal helpers ---------------------------------------------------

    fn last_user_content(request: &ModelQuery) -> Option<String> {
        request
            .messages
            .iter()
            .rev()
            .find(|m| m.role == MessageRole::User)
            .map(|m| m.content.clone())
    }

    fn sample_from_distribution(entries: &[WeightedOutput]) -> String {
        if entries.is_empty() {
            return String::from("(empty distribution)");
        }
        let total_weight: f64 = entries.iter().map(|e| e.weight).sum();
        if total_weight <= 0.0 {
            return entries[0].text.clone();
        }
        let mut rng = rand::thread_rng();
        let mut dart: f64 = rng.gen::<f64>() * total_weight;
        for entry in entries {
            dart -= entry.weight;
            if dart <= 0.0 {
                return entry.text.clone();
            }
        }
        entries.last().unwrap().text.clone()
    }

    fn simulate_latency(&self) -> f64 {
        if self.latency_max_ms <= self.latency_min_ms {
            return self.latency_min_ms;
        }
        let mut rng = rand::thread_rng();
        rng.gen_range(self.latency_min_ms..=self.latency_max_ms)
    }
}

impl BlackBoxModel for MockModel {
    fn query(&self, request: &ModelQuery) -> Result<ModelResponse, QueryError> {
        // 1. Increment counter.
        self.query_count.fetch_add(1, Ordering::Relaxed);

        // 2. Check error injection.
        if self.error_rate > 0.0 {
            let mut rng = rand::thread_rng();
            if rng.gen::<f64>() < self.error_rate {
                return Err(QueryError::ModelError(
                    "injected mock error".to_string(),
                ));
            }
        }

        // 3. Determine the response text.
        let key = Self::last_user_content(request).unwrap_or_default();

        let response_text = {
            // First check deterministic map.
            let responses = self.responses.lock().unwrap();
            if let Some(text) = responses.get(&key) {
                text.clone()
            } else {
                drop(responses);
                // Then check distribution map.
                let distributions = self.distributions.lock().unwrap();
                if let Some(entries) = distributions.get(&key) {
                    Self::sample_from_distribution(entries)
                } else {
                    // Default response when no mapping exists.
                    format!("Mock response to: {}", key)
                }
            }
        };

        // 4. Generate n_completions (all identical for the deterministic path,
        //    independently sampled for the distribution path).
        let n = request.n_completions.max(1);
        let mut completions = Vec::with_capacity(n);
        for i in 0..n {
            let content = if i == 0 {
                response_text.clone()
            } else {
                // For subsequent completions, re-sample if a distribution
                // exists; otherwise duplicate.
                let distributions = self.distributions.lock().unwrap();
                if let Some(entries) = distributions.get(&key) {
                    Self::sample_from_distribution(entries)
                } else {
                    response_text.clone()
                }
            };
            completions.push(Completion {
                content,
                finish_reason: FinishReason::Stop,
                logprobs: None,
            });
        }

        // 5. Simulate latency.
        let latency = self.simulate_latency();

        // 6. Token usage estimate.
        let prompt_tokens = request.estimated_prompt_tokens();
        let completion_tokens: usize = completions
            .iter()
            .map(|c| estimate_tokens(&c.content))
            .sum();

        Ok(ModelResponse {
            query_id: request.id.clone(),
            completions,
            model_id: self.id.clone(),
            usage: TokenUsage::new(prompt_tokens, completion_tokens),
            latency_ms: latency,
            created_at: Utc::now().to_rfc3339(),
        })
    }

    fn model_id(&self) -> &str {
        &self.id
    }

    fn supports_logprobs(&self) -> bool {
        false
    }

    fn max_context_length(&self) -> usize {
        8192
    }
}

// ---------------------------------------------------------------------------
// QueryBuilder
// ---------------------------------------------------------------------------

/// Ergonomic builder for constructing multi-turn `ModelQuery` instances.
pub struct QueryBuilder {
    query: ModelQuery,
}

impl QueryBuilder {
    pub fn new() -> Self {
        Self {
            query: ModelQuery::new(),
        }
    }

    /// Set the system prompt.
    pub fn system(mut self, prompt: &str) -> Self {
        self.query.system_prompt = Some(prompt.to_string());
        self.query
            .messages
            .push(ChatMessage::system(prompt));
        self
    }

    /// Append a user message.
    pub fn user(mut self, content: &str) -> Self {
        self.query.messages.push(ChatMessage::user(content));
        self
    }

    /// Append an assistant message (for few-shot examples or continued
    /// conversations).
    pub fn assistant(mut self, content: &str) -> Self {
        self.query
            .messages
            .push(ChatMessage::assistant(content));
        self
    }

    /// Set the sampling temperature.
    pub fn temperature(mut self, t: f64) -> Self {
        self.query.temperature = t;
        self
    }

    /// Set the nucleus-sampling parameter.
    pub fn top_p(mut self, p: f64) -> Self {
        self.query.top_p = p;
        self
    }

    /// Set the maximum number of tokens to generate.
    pub fn max_tokens(mut self, n: usize) -> Self {
        self.query.max_tokens = n;
        self
    }

    /// Request multiple independent completions.
    pub fn n_completions(mut self, n: usize) -> Self {
        self.query.n_completions = n;
        self
    }

    /// Request log-probabilities.
    pub fn logprobs(mut self, enabled: bool) -> Self {
        self.query.logprobs = enabled;
        self
    }

    /// Add a stop sequence.
    pub fn stop_sequence(mut self, seq: &str) -> Self {
        self.query.stop_sequences.push(seq.to_string());
        self
    }

    /// Consume the builder and return the finished `ModelQuery`.
    pub fn build(self) -> ModelQuery {
        self.query
    }
}

impl Default for QueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Token estimation
// ---------------------------------------------------------------------------

/// Rough token-count estimator: splits on whitespace and multiplies by 1.3
/// to approximate sub-word tokenization overhead.
pub fn estimate_tokens(text: &str) -> usize {
    let word_count = text.split_whitespace().count();
    ((word_count as f64) * 1.3).ceil() as usize
}

// ---------------------------------------------------------------------------
// ResponseAnalyzer
// ---------------------------------------------------------------------------

/// Utilities for aggregating and analysing collections of model responses.
pub struct ResponseAnalyzer;

impl ResponseAnalyzer {
    /// Aggregate all completions across the given responses into a
    /// `(content, probability)` distribution.  Probabilities sum to 1.0.
    pub fn extract_distribution(responses: &[ModelResponse]) -> Vec<(String, f64)> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut total = 0usize;
        for resp in responses {
            for comp in &resp.completions {
                *counts.entry(comp.content.clone()).or_insert(0) += 1;
                total += 1;
            }
        }
        if total == 0 {
            return Vec::new();
        }
        let mut dist: Vec<(String, f64)> = counts
            .into_iter()
            .map(|(text, count)| (text, count as f64 / total as f64))
            .collect();
        dist.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        dist
    }

    /// Compute the Shannon entropy (in nats) of the empirical response
    /// distribution.
    pub fn entropy_of_responses(responses: &[ModelResponse]) -> f64 {
        let dist = Self::extract_distribution(responses);
        if dist.is_empty() {
            return 0.0;
        }
        let mut h = 0.0f64;
        for (_text, p) in &dist {
            if *p > 0.0 {
                h -= p * p.ln();
            }
        }
        h
    }

    /// Return the most frequently occurring completion text, or `None` if
    /// there are no completions at all.
    pub fn most_common_response(responses: &[ModelResponse]) -> Option<String> {
        let dist = Self::extract_distribution(responses);
        dist.into_iter().next().map(|(text, _)| text)
    }

    /// Count the total number of completions across all responses.
    pub fn total_completions(responses: &[ModelResponse]) -> usize {
        responses.iter().map(|r| r.completions.len()).sum()
    }

    /// Compute the mean latency in milliseconds across all responses.
    pub fn mean_latency_ms(responses: &[ModelResponse]) -> f64 {
        if responses.is_empty() {
            return 0.0;
        }
        let sum: f64 = responses.iter().map(|r| r.latency_ms).sum();
        sum / responses.len() as f64
    }

    /// Aggregate total token usage across all responses.
    pub fn total_usage(responses: &[ModelResponse]) -> TokenUsage {
        let mut usage = TokenUsage::default();
        for r in responses {
            usage.prompt_tokens += r.usage.prompt_tokens;
            usage.completion_tokens += r.usage.completion_tokens;
            usage.total_tokens += r.usage.total_tokens;
        }
        usage
    }
}

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

/// Build a simple single-turn user query.
pub fn single_turn_query(user_message: &str) -> ModelQuery {
    QueryBuilder::new().user(user_message).build()
}

/// Build a system + user query.
pub fn system_user_query(system: &str, user_message: &str) -> ModelQuery {
    QueryBuilder::new()
        .system(system)
        .user(user_message)
        .build()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- ModelQuery ----------------------------------------------------------

    #[test]
    fn test_model_query_defaults() {
        let q = ModelQuery::new();
        assert!(q.messages.is_empty());
        assert!(q.system_prompt.is_none());
        assert!((q.temperature - 0.7).abs() < 1e-9);
        assert_eq!(q.n_completions, 1);
        assert!(!q.id.is_empty());
    }

    #[test]
    fn test_model_query_builder_methods() {
        let q = ModelQuery::new()
            .with_system_prompt("You are helpful.")
            .with_message(ChatMessage::user("Hello"))
            .with_temperature(0.3)
            .with_top_p(0.9)
            .with_max_tokens(128)
            .with_logprobs(true)
            .with_n_completions(5)
            .with_stop_sequence("\n");

        assert_eq!(q.system_prompt.as_deref(), Some("You are helpful."));
        assert_eq!(q.messages.len(), 1);
        assert!((q.temperature - 0.3).abs() < 1e-9);
        assert!((q.top_p - 0.9).abs() < 1e-9);
        assert_eq!(q.max_tokens, 128);
        assert!(q.logprobs);
        assert_eq!(q.n_completions, 5);
        assert_eq!(q.stop_sequences, vec!["\n".to_string()]);
    }

    // -- ChatMessage --------------------------------------------------------

    #[test]
    fn test_chat_message_constructors() {
        let s = ChatMessage::system("sys");
        let u = ChatMessage::user("usr");
        let a = ChatMessage::assistant("ast");
        assert_eq!(s.role, MessageRole::System);
        assert_eq!(u.role, MessageRole::User);
        assert_eq!(a.role, MessageRole::Assistant);
        assert_eq!(u.content, "usr");
    }

    // -- QueryBuilder -------------------------------------------------------

    #[test]
    fn test_query_builder_multi_turn() {
        let q = QueryBuilder::new()
            .system("Be concise.")
            .user("What is 2+2?")
            .assistant("4")
            .user("And 3+3?")
            .temperature(0.0)
            .max_tokens(64)
            .n_completions(3)
            .build();

        // system + user + assistant + user = 4 messages
        assert_eq!(q.messages.len(), 4);
        assert_eq!(q.messages[0].role, MessageRole::System);
        assert_eq!(q.messages[3].role, MessageRole::User);
        assert!((q.temperature - 0.0).abs() < 1e-9);
        assert_eq!(q.max_tokens, 64);
        assert_eq!(q.n_completions, 3);
    }

    // -- estimate_tokens ----------------------------------------------------

    #[test]
    fn test_estimate_tokens_empty() {
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn test_estimate_tokens_nonempty() {
        // "Hello world" = 2 words → ceil(2 * 1.3) = 3
        assert_eq!(estimate_tokens("Hello world"), 3);
    }

    // -- MockModel ----------------------------------------------------------

    #[test]
    fn test_mock_model_deterministic_response() {
        let model = MockModel::new("test-model")
            .with_response("ping", "pong");

        let q = single_turn_query("ping");
        let resp = model.query(&q).unwrap();
        assert_eq!(resp.completions.len(), 1);
        assert_eq!(resp.completions[0].content, "pong");
        assert_eq!(resp.model_id, "test-model");
        assert_eq!(model.query_count(), 1);
    }

    #[test]
    fn test_mock_model_default_response() {
        let model = MockModel::new("m");
        let q = single_turn_query("anything");
        let resp = model.query(&q).unwrap();
        assert!(resp.completions[0].content.contains("anything"));
    }

    #[test]
    fn test_mock_model_distribution() {
        let model = MockModel::new("dist-model")
            .with_distribution("coin", vec![("heads", 0.5), ("tails", 0.5)]);

        let q = single_turn_query("coin");
        let resp = model.query(&q).unwrap();
        let content = &resp.completions[0].content;
        assert!(content == "heads" || content == "tails");
    }

    #[test]
    fn test_mock_model_error_injection() {
        let model = MockModel::new("err").with_error_rate(1.0);
        let q = single_turn_query("hi");
        let result = model.query(&q);
        assert!(result.is_err());
        // Counter should still increment even on error.
        assert_eq!(model.query_count(), 1);
    }

    #[test]
    fn test_mock_model_multiple_completions() {
        let model = MockModel::new("m").with_response("x", "y");
        let q = ModelQuery::new()
            .with_message(ChatMessage::user("x"))
            .with_n_completions(4);
        let resp = model.query(&q).unwrap();
        assert_eq!(resp.completions.len(), 4);
        for c in &resp.completions {
            assert_eq!(c.content, "y");
        }
    }

    #[test]
    fn test_mock_model_latency() {
        let model = MockModel::new("m").with_latency(10.0, 20.0);
        let q = single_turn_query("test");
        let resp = model.query(&q).unwrap();
        assert!(resp.latency_ms >= 10.0 && resp.latency_ms <= 20.0);
    }

    // -- ResponseAnalyzer ---------------------------------------------------

    #[test]
    fn test_response_analyzer_distribution_and_entropy() {
        let model = MockModel::new("m")
            .with_response("q", "A");

        // Generate 10 identical responses.
        let mut responses = Vec::new();
        for _ in 0..10 {
            let q = single_turn_query("q");
            responses.push(model.query(&q).unwrap());
        }

        let dist = ResponseAnalyzer::extract_distribution(&responses);
        assert_eq!(dist.len(), 1);
        assert!((dist[0].1 - 1.0).abs() < 1e-9);

        // Entropy of a single-outcome distribution is 0.
        let h = ResponseAnalyzer::entropy_of_responses(&responses);
        assert!(h.abs() < 1e-9);

        let most = ResponseAnalyzer::most_common_response(&responses);
        assert_eq!(most, Some("A".to_string()));
    }

    #[test]
    fn test_response_analyzer_empty() {
        let dist = ResponseAnalyzer::extract_distribution(&[]);
        assert!(dist.is_empty());
        assert!(ResponseAnalyzer::entropy_of_responses(&[]).abs() < 1e-9);
        assert_eq!(ResponseAnalyzer::most_common_response(&[]), None);
    }

    #[test]
    fn test_response_analyzer_usage_and_latency() {
        let model = MockModel::new("m")
            .with_response("hi", "hello")
            .with_latency(5.0, 5.0);

        let mut responses = Vec::new();
        for _ in 0..3 {
            let q = single_turn_query("hi");
            responses.push(model.query(&q).unwrap());
        }

        let usage = ResponseAnalyzer::total_usage(&responses);
        assert!(usage.total_tokens > 0);
        assert_eq!(ResponseAnalyzer::total_completions(&responses), 3);

        let mean_lat = ResponseAnalyzer::mean_latency_ms(&responses);
        assert!((mean_lat - 5.0).abs() < 1e-9);
    }

    // -- Serde roundtrip ----------------------------------------------------

    #[test]
    fn test_serde_roundtrip_query() {
        let q = ModelQuery::new()
            .with_system_prompt("test")
            .with_message(ChatMessage::user("hello"))
            .with_temperature(0.5);

        let json = serde_json::to_string(&q).unwrap();
        let q2: ModelQuery = serde_json::from_str(&json).unwrap();
        assert_eq!(q.id, q2.id);
        assert_eq!(q.system_prompt, q2.system_prompt);
        assert_eq!(q.messages.len(), q2.messages.len());
        assert!((q.temperature - q2.temperature).abs() < 1e-9);
    }

    #[test]
    fn test_serde_roundtrip_response() {
        let resp = ModelResponse {
            query_id: "q1".to_string(),
            completions: vec![Completion {
                content: "hi".to_string(),
                finish_reason: FinishReason::Stop,
                logprobs: Some(vec![TokenLogProb {
                    token: "hi".to_string(),
                    logprob: -0.5,
                    top_logprobs: vec![("hi".to_string(), -0.5), ("hello".to_string(), -1.2)],
                }]),
            }],
            model_id: "m".to_string(),
            usage: TokenUsage::new(10, 5),
            latency_ms: 42.0,
            created_at: Utc::now().to_rfc3339(),
        };

        let json = serde_json::to_string(&resp).unwrap();
        let resp2: ModelResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp.query_id, resp2.query_id);
        assert_eq!(resp.completions.len(), resp2.completions.len());
        assert_eq!(resp.usage.total_tokens, 15);
    }

    // -- Convenience constructors -------------------------------------------

    #[test]
    fn test_convenience_constructors() {
        let q1 = single_turn_query("What is 1+1?");
        assert_eq!(q1.messages.len(), 1);
        assert_eq!(q1.messages[0].role, MessageRole::User);

        let q2 = system_user_query("Be brief.", "Hi");
        assert_eq!(q2.messages.len(), 2);
        assert_eq!(q2.system_prompt, Some("Be brief.".to_string()));
    }

    // -- QueryError Display -------------------------------------------------

    #[test]
    fn test_query_error_display() {
        let e = QueryError::RateLimited { retry_after_ms: 500 };
        assert!(format!("{}", e).contains("500"));

        let e2 = QueryError::BudgetExhausted;
        assert!(format!("{}", e2).contains("budget"));
    }

    // -- TokenUsage ---------------------------------------------------------

    #[test]
    fn test_token_usage_new() {
        let u = TokenUsage::new(100, 50);
        assert_eq!(u.prompt_tokens, 100);
        assert_eq!(u.completion_tokens, 50);
        assert_eq!(u.total_tokens, 150);
    }
}
