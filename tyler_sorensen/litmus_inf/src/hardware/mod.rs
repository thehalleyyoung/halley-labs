//! Hardware testing framework for running litmus tests on actual GPU hardware.
//!
//! This module provides tools for:
//! - Generating GPU kernel code from litmus tests (CUDA, OpenCL, Vulkan, Metal)
//! - Running litmus tests with stress-testing techniques
//! - Analyzing hardware results against model predictions

pub mod litmus_runner;
pub mod stress_testing;
pub mod kernel_gen;
pub mod result_analysis;
