//! Benchmark suite: scaled litmus tests and real concurrent code patterns.
//!
//! Addresses critique #1 (trivially small instances) by generating tests with
//! 4-8 threads, 3-6 memory locations, producing thousands to millions of
//! execution graphs.

pub mod scaled_litmus;
pub mod real_concurrent_code;
pub mod comprehensive_runner;

pub use scaled_litmus::{
    ScaledLitmusGenerator, ScaledTestConfig, ScaledTestResult,
    ScalingDimension, PatternFamily,
};
pub use real_concurrent_code::{
    RealCodeExtractor, ConcurrentPattern, PatternSource,
    LockFreePattern, RcuPattern, KernelPattern,
};
pub use comprehensive_runner::{
    ComprehensiveBenchRunner, ComprehensiveBenchConfig,
    ComprehensiveBenchResult, OutputFormat,
};
