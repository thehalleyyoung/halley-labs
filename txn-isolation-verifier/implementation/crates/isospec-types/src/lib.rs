//! IsoSpec shared types crate.
//!
//! Provides foundational types for transaction modeling, isolation levels,
//! operations, predicates, and the transaction intermediate representation (IR).

pub mod transaction;
pub mod isolation;
pub mod operation;
pub mod predicate;
pub mod ir;
pub mod column;
pub mod schema;
pub mod value;
pub mod error;
pub mod config;
pub mod lock;
pub mod version;
pub mod snapshot;
pub mod dependency;
pub mod schedule;
pub mod workload;
pub mod dialect;
pub mod constraint;
pub mod identifier;

pub use transaction::*;
pub use isolation::*;
pub use operation::*;
pub use error::IsoSpecError;
