//! # isospec-adapter
//!
//! Database adapter layer for executing witness schedules against real database
//! engines. Manages Docker containers, connection pooling, concurrent SQL
//! execution, and result parsing.

pub mod adapter;
pub mod docker;
pub mod executor;
pub mod parser;
pub mod connection;
