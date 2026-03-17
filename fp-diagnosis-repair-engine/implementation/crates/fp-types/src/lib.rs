//! Core IEEE 754 floating-point types, precision modes, rounding modes, error bounds,
//! and data structures for the Penumbra floating-point diagnosis and repair engine.

pub mod ieee754;
pub mod precision;
pub mod rounding;
pub mod error_bounds;
pub mod expression;
pub mod eag;
pub mod trace;
pub mod diagnosis;
pub mod repair;
pub mod fpclass;
pub mod ulp;
pub mod double_double;

pub use ieee754::*;
pub use precision::*;
pub use rounding::*;
pub use error_bounds::*;
pub use expression::*;
pub use eag::*;
pub use trace::*;
pub use diagnosis::*;
pub use repair::*;
pub use fpclass::*;
pub use ulp::*;
pub use double_double::*;
