//! Enhanced LLM-assisted litmus test generation.
//!
//! Addresses critique #7 (75% accuracy) with:
//!   - Multi-shot prompting with examples
//!   - Chain-of-thought for complex patterns
//!   - Validation loop: generate → parse → check → refine
//!   - Pattern-specific prompt templates
//!   - Target >90% accuracy

pub mod prompt_engine;
pub mod validation_loop;
pub mod pattern_prompts;

pub use prompt_engine::{
    LlmPromptEngine, PromptConfig, PromptTemplate, ShotExample,
    GenerationRequest, GenerationResult,
};
pub use validation_loop::{
    ValidationLoop, ValidationConfig, ValidationResult,
    ValidationStep, RefinementStrategy,
};
pub use pattern_prompts::{
    PatternPromptLibrary, BugPatternPrompt, ModelSpecificPrompt,
};
