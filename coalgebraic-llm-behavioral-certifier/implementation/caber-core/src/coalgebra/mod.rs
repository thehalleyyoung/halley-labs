//! Coalgebra module — core mathematical abstractions for behavioral modeling.
//!
//! This module implements the foundational coalgebraic framework:
//! - Functors (polynomial, behavioral, sub-distribution)
//! - Coalgebras (finite, probabilistic, LLM-behavioral)
//! - Semirings for weighted automata
//! - Sub-distributions with metric computations
//! - Abstraction lattices and Galois connections
//! - Bisimulation relations and distance computation
//! - Functor bandwidth and sample complexity

pub mod types;
pub mod semiring;
pub mod distribution;
pub mod functor;
pub mod coalgebra;
pub mod abstraction;
pub mod bisimulation;
pub mod bandwidth;

pub use types::*;
pub use semiring::{Semiring, StarSemiring, ProbabilitySemiring, TropicalSemiring, ViterbiSemiring, BooleanSemiring, CountingSemiring, LogSemiring};
pub use distribution::SubDistribution;
pub use functor::{Functor, BehavioralFunctor, SubDistributionFunctor};
pub use coalgebra::{CoalgebraSystem, FiniteCoalgebra, ProbabilisticCoalgebra, LLMBehavioralCoalgebra};
pub use abstraction::{AbstractionLevel, AbstractionLattice};
pub use bisimulation::{BisimulationRelation, QuantitativeBisimulation};
pub use bandwidth::FunctorBandwidth;
