//! CABER Core - Coalgebraic Behavioral Auditing of Foundation Models
//!
//! This crate provides the core mathematical infrastructure for behavioral
//! auditing of black-box LLMs using coalgebraic automata theory. The key
//! insight is that an LLM's behavior can be modeled as a coalgebra over
//! a behavioral functor, enabling rigorous comparison via bisimulation metrics.

pub mod coalgebra;
pub mod model_checker;
pub mod bisimulation;
pub mod certificate;
pub mod query;
pub mod utils;
pub mod learning;
pub mod abstraction;
pub mod temporal;
