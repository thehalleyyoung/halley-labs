//! Engine-faithful operational semantics (LTS) for production SQL engines.
//!
//! Each module implements the `EngineModel` and `EngineState` traits from
//! `isospec_core::engine_traits`, faithfully modeling the concurrency-control
//! mechanisms of PostgreSQL 16.x, MySQL 8.0 InnoDB, and SQL Server 2022.

pub mod common;
pub mod postgresql;
pub mod mysql;
pub mod sqlserver;
