# MutSpec API Reference

## shared-types

The foundational crate providing all shared type definitions.

### Core Types

| Type | Module | Description |
|------|--------|-------------|
| `Expression` | `ast` | AST expression nodes (literals, binary ops, etc.) |
| `Statement` | `ast` | AST statement nodes (assign, if-else, return, etc.) |
| `Function` | `ast` | Function definition with parameters and body |
| `Program` | `ast` | Top-level program containing functions |
| `Formula` | `formula` | Logical formulas (And, Or, Not, Implies, Atom, etc.) |
| `Term` | `formula` | Arithmetic terms (Const, Var, Add, Sub, etc.) |
| `Predicate` | `formula` | Atomic predicate (left relation right) |
| `Relation` | `formula` | Relational operators (Eq, Ne, Lt, Le, Gt, Ge) |
| `IrExpr` | `ir` | IR expressions in SSA form |
| `IrStatement` | `ir` | IR statements (Assign, Assert, Assume, PhiAssign) |
| `BasicBlock` | `ir` | Basic block with phi nodes, statements, terminator |
| `IrFunction` | `ir` | IR function with basic blocks |
| `Terminator` | `ir` | Block terminators (Branch, ConditionalBranch, Return) |
| `SsaVar` | `ir` | SSA variable with name, version, and type |
| `Contract` | `contracts` | A synthesized contract with clauses and provenance |
| `ContractClause` | `contracts` | Requires, Ensures, or Invariant clause |
| `MutationOperator` | `operators` | Enum of all 12 mutation operators |
| `MutantDescriptor` | `operators` | Full description of a generated mutant |
| `MutantStatus` | `operators` | Status: Alive, Killed, Equivalent, Timeout, Error |
| `MutSpecConfig` | `config` | Top-level configuration loaded from TOML |
| `MutSpecError` | `errors` | Unified error type for all MutSpec operations |

## mutation-core

Mutation generation, application, and kill matrix tracking.

### Key APIs

- `generate_mutants(function: &Function, config: &MutationConfig) -> Vec<MutantDescriptor>`
- `KillMatrix::new() -> KillMatrix`
- `KillMatrix::record_kill(mutant: MutantId, test: &str, info: KillInfo)`
- `KillMatrix::mutation_score() -> f64`

## contract-synth

Contract synthesis via the lattice-walk algorithm.

### Key APIs

- `synthesize_contracts(function: &IrFunction, kill_matrix: &KillMatrix) -> Vec<Contract>`
- `LatticeWalker::new(config: &SynthesisConfig) -> LatticeWalker`
- `LatticeWalker::walk(predicates: &[Formula]) -> Formula`

## smt-solver

SMT-LIB 2.6 interface with incremental solving support.

### Key APIs

- `SmtSolver::new(config: &SmtConfig) -> Result<SmtSolver>`
- `SmtSolver::check_sat(formula: &Formula) -> Result<SatResult>`
- `SmtSolver::check_implies(antecedent: &Formula, consequent: &Formula) -> Result<bool>`
- `IncrementalContext::push() / pop()`
