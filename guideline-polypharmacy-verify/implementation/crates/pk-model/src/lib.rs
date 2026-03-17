//! # GuardPharma Pharmacokinetic Model Library
//!
//! Implements compartmental PK models, Metzler matrix operations, CYP enzyme
//! inhibition, steady-state solvers, ODE integration, population PK, and
//! drug interaction computations for the polypharmacy verification system.

pub mod compartment;
pub mod metzler;
pub mod cyp_inhibition;
pub mod cyp_overapprox;
pub mod ode_solver;
pub mod drug_database;
pub mod population;
pub mod interaction;

pub use compartment::{CompartmentModel, OneCompartmentModel, TwoCompartmentModel, ThreeCompartmentModel};
pub use metzler::{MetzlerMatrix, MetzlerInterval};
pub use cyp_inhibition::{InhibitionModel, InhibitionModelType, CypInhibitionNetwork};
pub use cyp_overapprox::{CypOverApproximator, MetzlerCheckResult, ConfidenceReport, check_metzler};
pub use ode_solver::{OdeSolver, EulerSolver, RungeKutta4Solver, AdaptiveRK45Solver, PkOdeSystem, Trajectory};
pub use drug_database::{DrugDatabase, DrugPkEntry, build_default_database};
pub use population::{PopulationModel, CovariateModel, WorstCaseParameters, BoundedParameterSpace, ParameterSampler};
pub use interaction::{InteractionNetwork, InteractionEdge, InteractionMechanism, InteractionChain, MonotonicityChecker};
