# MutSpec Examples

Example programs for the MutSpec mutation-driven contract synthesis pipeline.
Each `.ms` file is a loop-free, first-order imperative program over QF-LIA
(quantifier-free linear integer arithmetic).

## Running an example

```bash
# Run from the Cargo workspace root
cd implementation

# One-shot pipeline
cargo run --release --bin mutspec -- analyze ../examples/absolute_value.ms

# Staged quickstart
cargo run --release --bin mutspec -- mutate ../examples/absolute_value.ms \
    -o mutants.json
cargo run --release --bin mutspec -- synthesize mutants.json \
    --tier 1 -o contracts.json
cargo run --release --bin mutspec -- verify contracts.json
```

## Examples

| File                    | Description                                       | Mutation operators |
|-------------------------|---------------------------------------------------|--------------------|
| `basic_arithmetic.ms`   | Clamp function with range postconditions          | ROR, AOR           |
| `array_bounds.ms`       | Array index validation and safe access            | ROR, UOI, AOR      |
| `absolute_value.ms`     | Absolute value with sign postconditions           | ROR, UOI           |
| `max_function.ms`       | Max of two integers                               | ROR                |
| `linear_search.ms`      | Unrolled linear search in bounded array           | ROR, AOR, UOI      |
| `gap_analysis_demo.ms`  | Safe division exposing a missing-precondition gap | ROR, AOR, UOI      |
| `jml_output_demo.ms`    | Bounded increment with JML annotation output      | ROR, AOR           |
| `multi_function.ms`     | Multiple related functions (min, max, clamp, mid) | ROR, AOR, UOI      |

## Expected contracts

The `expected_contracts/` directory contains JSON files describing the ground-truth
contracts for selected examples. Each file records:

- **preconditions** – constraints the caller must satisfy
- **postconditions** – guarantees the function provides
- **provenance** – which mutants were killed to discover each clause
- **tier** – lattice tier at which the contract was synthesized (1 = strongest)

## File format

MutSpec input files (`.ms`) use a minimal imperative syntax:
- Types: `int`, `int[]` (bounded arrays)
- Control flow: `if / else` (no loops—programs must be loop-free)
- Annotations: `@requires`, `@ensures`, `@mutate` directives in comments
- Arithmetic: `+`, `-`, `*` (linear only; no division or modulo)

See the [language reference](../docs/) for the full grammar.
