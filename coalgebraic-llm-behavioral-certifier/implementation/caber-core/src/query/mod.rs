//! Query module — black-box model interface and query scheduling.
//!
//! Provides the abstraction layer for querying LLMs as black boxes,
//! along with scheduling, rate limiting, and consistency monitoring.

pub mod interface;
pub mod scheduler;
pub mod consistency;
