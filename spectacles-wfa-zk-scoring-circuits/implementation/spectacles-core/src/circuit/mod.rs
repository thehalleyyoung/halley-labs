// Circuit module: WFA-to-STARK circuit synthesizer
//
// Compiles Weighted Finite Automata into AIR (Algebraic Intermediate Representation)
// constraints, generates execution traces, and produces/verifies STARK proofs.

pub mod goldilocks;
pub mod air;
pub mod compiler;
pub mod trace;
pub mod stark;
pub mod fri;
pub mod merkle;
pub mod gadgets;

pub use goldilocks::{GoldilocksField, GoldilocksExt};
pub use air::{AIRConstraint, AIRTrace, AIRProgram, ConstraintType};
pub use compiler::WFACircuitCompiler;
pub use trace::ExecutionTrace;
pub use stark::{STARKProver, STARKVerifier, STARKProof};
pub use fri::FRIProtocol;
pub use merkle::MerkleTree;
pub use gadgets::*;
