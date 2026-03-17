//! Shared types for the Certified Leakage Contracts framework.
//!
//! This crate provides the foundational type definitions used across all other crates
//! in the workspace, including address types, cache configuration, instruction representation,
//! register definitions, memory models, and error types.

pub mod address;
pub mod cache_config;
pub mod instruction;
pub mod register;
pub mod memory;
pub mod error;
pub mod cfg;
pub mod operand;
pub mod interval;
pub mod bitvector;
pub mod security_level;
pub mod program;

pub use address::{VirtualAddress, PhysicalAddress, AddressRange, CacheLine, CacheSet, CacheTag};
pub use cache_config::{CacheConfig, CacheGeometry, ReplacementPolicy, CacheLevel};
pub use instruction::{Instruction, InstructionKind, Opcode, InstructionFlags};
pub use register::{Register, RegisterFile, RegisterClass, RegisterId};
pub use memory::{MemoryAccess, MemoryRegion, MemoryPermission, MemoryMap, MemoryAccessKind};
pub use error::{AnalysisError, ErrorKind};
pub use cfg::{ControlFlowGraph, BasicBlock, CfgEdge, CfgEdgeKind, BlockId};
pub use operand::{Operand, OperandKind, ImmediateValue, MemoryOperand};
pub use interval::{Interval, IntervalBound};
pub use bitvector::{BitVector, BitWidth};
pub use security_level::{SecurityLevel, SecurityLattice};
pub use program::{Program, Function, FunctionId, CallGraph, CallSite};
