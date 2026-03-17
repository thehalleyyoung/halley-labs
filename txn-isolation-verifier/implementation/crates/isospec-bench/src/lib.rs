//! # isospec-bench
//!
//! Benchmarking framework for the IsoSpec transaction isolation verification system.
//! Provides harnesses, workload generators, TPC-C/TPC-E templates, metrics collection,
//! and report generation for systematic performance evaluation.

pub mod harness;
pub mod workloads;
pub mod tpcc;
pub mod tpce;
pub mod metrics;
pub mod report;
