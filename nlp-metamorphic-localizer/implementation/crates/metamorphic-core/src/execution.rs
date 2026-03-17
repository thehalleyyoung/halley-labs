use serde::{Deserialize, Serialize};
use shared_types::{IntermediateRepresentation, Result, StageId};

pub trait PipelineRunner: Send + Sync {
    fn run(&self, input: &IntermediateRepresentation) -> Result<Vec<IntermediateRepresentation>>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageExecutionResult {
    pub stage_id: StageId,
    pub stage_name: String,
    pub duration_ms: u64,
    pub success: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan { pub stage_order: Vec<String> }

#[derive(Debug, Clone)]
pub struct ExecutionContext { pub verbose: bool }
impl Default for ExecutionContext { fn default() -> Self { Self { verbose: false } } }

pub struct ExecutionEngine;
impl ExecutionEngine { pub fn new() -> Self { Self } }
impl Default for ExecutionEngine { fn default() -> Self { Self } }

pub struct ExecutionCache;
impl ExecutionCache { pub fn new() -> Self { Self } }
impl Default for ExecutionCache { fn default() -> Self { Self } }

pub struct BatchExecutor { pub parallelism: usize }
impl BatchExecutor { pub fn new(parallelism: usize) -> Self { Self { parallelism } } }

pub struct ParallelExecutor { pub threads: usize }
impl ParallelExecutor { pub fn new(threads: usize) -> Self { Self { threads } } }

pub struct DefaultPipelineRunner;
impl DefaultPipelineRunner { pub fn new() -> Self { Self } }
impl Default for DefaultPipelineRunner { fn default() -> Self { Self } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineExecution {
    pub stage_results: Vec<StageExecutionResult>,
    pub total_duration_ms: u64,
}
