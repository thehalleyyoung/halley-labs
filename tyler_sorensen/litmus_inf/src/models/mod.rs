//! Memory model specifications for LITMUS∞.
//!
//! Provides detailed, standalone memory model implementations for
//! Sequential Consistency (SC), Total Store Order (TSO),
//! Partial Store Order (PSO), and C++11 relaxed atomics.

pub mod sc;
pub mod tso;
pub mod pso;
pub mod relaxed;

pub use sc::{ScModel, ScAxiom, ScViolation, ScTestResult};
pub use tso::{TsoModel, TsoAxiom, TsoViolation, StoreBuffer, TsoTestResult};
pub use pso::{PsoModel, PsoAxiom, PsoViolation, PerAddressStoreBuffer, PsoTestResult};
pub use relaxed::{RelaxedModel, RelaxedAxiom, RelaxedViolation, ReleaseAcquireChecker, RelaxedTestResult};

pub mod opencl;
pub mod metal;

pub mod vulkan;

pub mod cuda_scoped;
pub mod vulkan_full;
pub mod opencl2;
pub mod hip;

pub use cuda_scoped::{CudaScopeHierarchy, CudaScopedModel, CudaScope, CudaAxiom, CudaViolation, CudaExecution, CudaEvent, CudaVerificationResult};
pub use vulkan_full::{VulkanMemoryModel, VulkanStorageClass, VulkanAvailVisOp, VulkanFullScope, VulkanFullAxiom, VulkanFullViolation, VulkanFullExecution, VulkanFullEvent, VulkanFullVerificationResult};
pub use opencl2::{OpenCl2Model, OpenClScope, OpenClMemoryRegion, OpenCl2Axiom, OpenCl2Violation, OpenCl2Execution, OpenCl2Event, OpenCl2VerificationResult};
pub use hip::{HipModel, HipScope, HipAxiom, HipViolation, HipExecution, HipEvent, HipVerificationResult};
