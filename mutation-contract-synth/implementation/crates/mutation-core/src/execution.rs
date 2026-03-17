//! Interpreter-based mutant execution engine.

use shared_types::{KillInfo, MutantDescriptor, MutantId, MutantStatus, SpanInfo};
use std::collections::HashMap;

/// Result of executing a single test against a mutant.
#[derive(Debug, Clone)]
pub struct TestResult {
    pub test_name: String,
    pub passed: bool,
    pub execution_time_ms: f64,
    pub error_message: Option<String>,
}

/// Aggregated result for executing all tests against a mutant.
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub mutant_id: MutantId,
    pub status: MutantStatus,
    pub test_results: Vec<TestResult>,
    pub total_time_ms: f64,
}

/// Trait for executing a mutant against a test suite.
pub trait MutantExecutor {
    fn execute(&self, mutant: &MutantDescriptor) -> ExecutionResult;
}

/// Interpreter-based execution engine.
pub struct ExecutionEngine {
    pub timeout_ms: u64,
}

impl ExecutionEngine {
    pub fn new(timeout_ms: u64) -> Self {
        Self { timeout_ms }
    }
}

impl MutantExecutor for ExecutionEngine {
    fn execute(&self, mutant: &MutantDescriptor) -> ExecutionResult {
        ExecutionResult {
            mutant_id: mutant.id.clone(),
            status: MutantStatus::Alive,
            test_results: Vec::new(),
            total_time_ms: 0.0,
        }
    }
}
