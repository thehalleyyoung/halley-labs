# Spectacles API Reference

API documentation for implemented public modules, types, traits, and
functions in the `spectacles-core` crate. All items documented below
are implemented and tested.

**Scope note.** This documents the Rust implementation only. The Lean 4
formalization covers semiring axioms and core compilation theorems but
does not have a verified extraction to this Rust codebase.

**What has been empirically validated:**
- Scoring module: 57,518 differential tests, 0 disagreements
- WFA module: Semiring axioms (125 unit tests), automaton evaluation (81 tests)
- Circuit module: STARK prove+verify demonstrated on real data (9 proofs, all verified)
- Proof sizes: ~36–73 KiB measured on real proofs; full metric circuit proofs are future work
- PSI module: Detects n-gram overlap only, not paraphrase memorization

## Table of Contents

- [EvalSpec Module](#evalspec-module)
  - [Types](#evalspec-types)
  - [Parser](#evalspec-parser)
  - [TypeChecker](#evalspec-typechecker)
  - [Compiler](#evalspec-compiler)
  - [Semantics](#evalspec-semantics)
  - [Builtins](#evalspec-builtins)
- [WFA Module](#wfa-module)
  - [Semiring](#wfa-semiring)
  - [Automaton](#wfa-automaton)
  - [Transducer](#wfa-transducer)
  - [Minimization](#wfa-minimization)
  - [Equivalence](#wfa-equivalence)
  - [Operations](#wfa-operations)
  - [Formal Power Series](#wfa-formal-power-series)
  - [Field Embedding](#wfa-field-embedding)
- [Circuit Module](#circuit-module)
  - [Goldilocks Field](#circuit-goldilocks)
  - [AIR Constraints](#circuit-air)
  - [Compiler](#circuit-compiler)
  - [Trace](#circuit-trace)
  - [STARK](#circuit-stark)
  - [FRI](#circuit-fri)
  - [Merkle Trees](#circuit-merkle)
  - [Gadgets](#circuit-gadgets)
- [Protocol Module](#protocol-module)
  - [State Machine](#protocol-state-machine)
  - [Commitment](#protocol-commitment)
  - [Transcript](#protocol-transcript)
  - [Certificate](#protocol-certificate)
- [PSI Module](#psi-module)
  - [N-gram](#psi-ngram)
  - [Trie](#psi-trie)
  - [OPRF](#psi-oprf)
  - [Protocol](#psi-protocol)
- [Scoring Module](#scoring-module)
  - [Tokenizer](#scoring-tokenizer)
  - [Exact Match](#scoring-exact-match)
  - [Token F1](#scoring-token-f1)
  - [BLEU](#scoring-bleu)
  - [ROUGE](#scoring-rouge)
  - [Regex Match](#scoring-regex-match)
  - [Pass@k](#scoring-pass-at-k)
  - [Differential Testing](#scoring-differential)
- [Utils Module](#utils-module)
  - [Hash](#utils-hash)
  - [Serialization](#utils-serialization)
  - [Math](#utils-math)
- [Common Workflows](#common-workflows)

---

## EvalSpec Module

**Path:** `spectacles-core/src/evalspec/`

The EvalSpec module provides a typed domain-specific language for specifying
NLP evaluation metrics. It includes a parser, type checker, compiler to WFA,
denotational semantics, and built-in metric definitions.

### EvalSpec Types

**File:** `evalspec/types.rs`

#### `Span`

Source location span for error reporting.

```rust
pub struct Span {
    pub file: String,
    pub start_line: usize,
    pub start_col: usize,
    pub end_line: usize,
    pub end_col: usize,
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(file: &str, start_line: usize, start_col: usize, end_line: usize, end_col: usize) -> Self` | Create a span with explicit positions |
| `synthetic` | `fn synthetic() -> Self` | Create a synthetic span (for compiler-generated nodes) |
| `merge` | `fn merge(&self, other: &Span) -> Span` | Merge two spans into the smallest enclosing span |

#### `Spanned<T>`

AST node with attached source location.

```rust
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(node: T, span: Span) -> Self` | Wrap a node with a span |
| `map` | `fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Spanned<U>` | Transform the inner node, preserving the span |
| `as_ref` | `fn as_ref(&self) -> Spanned<&T>` | Borrow the inner node |

#### `BaseType`

Primitive types in the EvalSpec type system.

```rust
pub enum BaseType {
    String,
    Integer,
    Float,
    Bool,
    List,
    Tuple,
    Token,
    TokenSequence,
    NGram(usize),
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `is_scalar` | `fn is_scalar(&self) -> bool` | Returns `true` for `Integer`, `Float`, `Bool` |
| `is_sequence` | `fn is_sequence(&self) -> bool` | Returns `true` for `List`, `Tuple`, `TokenSequence` |
| `nesting_depth` | `fn nesting_depth(&self) -> usize` | Nesting depth (0 for scalars, 1 for sequences) |

#### `SemiringType`

Semiring type annotations inferred by the type checker.

```rust
pub enum SemiringType {
    Counting,
    Boolean,
    Tropical,
    BoundedCounting(u64),
    Real,
    LogDomain,
    Viterbi,
    Goldilocks,
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `zero` | `fn zero(&self) -> String` | String representation of the zero element |
| `one` | `fn one(&self) -> String` | String representation of the one element |
| `add` | `fn add(&self) -> String` | String representation of the addition operation |
| `mul` | `fn mul(&self) -> String` | String representation of the multiplication operation |
| `is_commutative` | `fn is_commutative(&self) -> bool` | Whether the semiring is commutative |

#### `EvalType`

Full type representation in the EvalSpec type system.

```rust
pub enum EvalType {
    Base(BaseType),
    Semiring(SemiringType),
    Function {
        params: Vec<EvalType>,
        ret: Box<EvalType>,
    },
    Metric {
        input_type: Box<EvalType>,
        output_type: Box<EvalType>,
        semiring: SemiringType,
    },
}
```

#### `MetricType`

High-level metric classification.

```rust
pub enum MetricType {
    StringMatch,
    NGramBased,
    SequenceBased,
    Statistical,
    Custom,
}
```

**Example — Working with EvalSpec types:**

```rust
use spectacles_core::evalspec::types::*;

// Create a metric type for BLEU
let bleu_type = EvalType::Metric {
    input_type: Box::new(EvalType::Base(BaseType::TokenSequence)),
    output_type: Box::new(EvalType::Base(BaseType::Float)),
    semiring: SemiringType::Counting,
};

// Check semiring properties
let sr = SemiringType::Counting;
assert!(sr.is_commutative());
assert_eq!(sr.zero(), "0");
assert_eq!(sr.one(), "1");

// Create a spanned expression
let span = Span::new("bleu.eval", 1, 0, 1, 20);
let spanned = Spanned::new(bleu_type, span);
```

### EvalSpec Parser

**File:** `evalspec/parser.rs`

The parser transforms EvalSpec source text into a `Spanned<Expr>` AST with
source location information for error reporting.

#### `Parser`

```rust
pub struct Parser {
    // internal state
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(source: &str, file: &str) -> Self` | Create a parser for the given source |
| `parse` | `fn parse(&mut self) -> Result<Spanned<Expr>, ParseError>` | Parse the source into an AST |
| `parse_expr` | `fn parse_expr(&mut self) -> Result<Spanned<Expr>, ParseError>` | Parse a single expression |
| `parse_metric_def` | `fn parse_metric_def(&mut self) -> Result<Spanned<Expr>, ParseError>` | Parse a metric definition |

#### `ParseError`

```rust
pub struct ParseError {
    pub message: String,
    pub span: Span,
}
```

**Example — Parsing EvalSpec:**

```rust
use spectacles_core::evalspec::parser::Parser;

let source = r#"
metric bleu(candidate: TokenSequence, reference: TokenSequence) -> Float {
    ngram_precision(candidate, reference, n=4)
}
"#;

let mut parser = Parser::new(source, "bleu.eval");
let ast = parser.parse().expect("parse failed");
```

### EvalSpec TypeChecker

**File:** `evalspec/typechecker.rs`

The type checker validates EvalSpec ASTs and infers semiring requirements
for each metric definition.

#### `TypeChecker`

```rust
pub struct TypeChecker {
    // type environment
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new() -> Self` | Create a type checker with default environment |
| `check` | `fn check(&mut self, ast: &Spanned<Expr>) -> Result<EvalType, TypeError>` | Type-check an AST node |
| `infer_semiring` | `fn infer_semiring(&self, expr: &Expr) -> Result<SemiringType, TypeError>` | Infer the required semiring for an expression |
| `with_builtins` | `fn with_builtins(self) -> Self` | Load built-in metric type signatures |

#### `TypeError`

```rust
pub struct TypeError {
    pub message: String,
    pub span: Span,
    pub expected: Option<EvalType>,
    pub found: Option<EvalType>,
}
```

**Example — Type checking:**

```rust
use spectacles_core::evalspec::{parser::Parser, typechecker::TypeChecker};

let mut parser = Parser::new(source, "metric.eval");
let ast = parser.parse()?;

let mut checker = TypeChecker::new().with_builtins();
let eval_type = checker.check(&ast)?;

// The type checker infers the semiring:
if let EvalType::Metric { semiring, .. } = eval_type {
    println!("Inferred semiring: {:?}", semiring);
    // e.g., SemiringType::Counting for BLEU
}
```

### EvalSpec Compiler

**File:** `evalspec/compiler.rs`

The compiler lowers typed EvalSpec ASTs into weighted finite automata.

#### `EvalSpecCompiler`

```rust
pub struct EvalSpecCompiler {
    // compilation context
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new() -> Self` | Create a compiler with default settings |
| `compile` | `fn compile(&mut self, ast: &Spanned<Expr>) -> Result<CompiledMetric, CompileError>` | Compile an AST into a WFA |
| `compile_to_counting` | `fn compile_to_counting(&mut self, ast: &Spanned<Expr>) -> Result<WeightedFiniteAutomaton<CountingSemiring>, CompileError>` | Compile to a counting semiring WFA |
| `compile_to_boolean` | `fn compile_to_boolean(&mut self, ast: &Spanned<Expr>) -> Result<WeightedFiniteAutomaton<BooleanSemiring>, CompileError>` | Compile to a Boolean semiring WFA |

#### `CompiledMetric`

```rust
pub struct CompiledMetric {
    pub name: String,
    pub semiring: SemiringType,
    pub state_count: usize,
    pub alphabet_size: usize,
}
```

**Example — Compiling a metric to WFA:**

```rust
use spectacles_core::evalspec::compiler::EvalSpecCompiler;

let mut compiler = EvalSpecCompiler::new();
let compiled = compiler.compile(&typed_ast)?;

println!("Metric: {}", compiled.name);
println!("Semiring: {:?}", compiled.semiring);
println!("States: {}", compiled.state_count);
```

### EvalSpec Semantics

**File:** `evalspec/semantics.rs`

Denotational semantics for EvalSpec, providing ground-truth definitions
for each built-in metric. Used for correctness validation.

### EvalSpec Builtins

**File:** `evalspec/builtins.rs`

Pre-defined scoring functions available in EvalSpec without explicit
definition:

| Builtin | Semiring | Description |
|---------|----------|-------------|
| `exact_match` | Boolean | Binary string equality |
| `token_f1` | Counting | Token-level F1 score |
| `bleu` | Counting | BLEU with n-gram precision (n=1..4) |
| `rouge_n` | Counting | ROUGE-N n-gram recall |
| `rouge_l` | MaxPlus | ROUGE-L LCS F-measure |
| `regex_match` | Boolean | Regular expression matching |
| `pass_at_k` | Counting | Pass@k code generation estimator |

---

## WFA Module

**Path:** `spectacles-core/src/wfa/`

The WFA module implements a full weighted finite automaton engine parameterized
over semiring types.

### WFA Semiring

**File:** `wfa/semiring.rs`

#### `Semiring` Trait

The foundational algebraic trait for all WFA operations.

```rust
pub trait Semiring: Clone + Debug + PartialEq {
    /// Additive identity (⊕-identity)
    fn zero() -> Self;

    /// Multiplicative identity (⊗-identity)
    fn one() -> Self;

    /// Semiring addition (⊕)
    fn add(&self, other: &Self) -> Self;

    /// Semiring multiplication (⊗)
    fn mul(&self, other: &Self) -> Self;
}
```

**Laws that all implementations satisfy:**

1. (S, ⊕, 0̄) is a commutative monoid
2. (S, ⊗, 1̄) is a monoid
3. ⊗ distributes over ⊕
4. 0̄ is an annihilator for ⊗: a ⊗ 0̄ = 0̄ ⊗ a = 0̄

#### `BooleanSemiring`

```rust
pub struct BooleanSemiring(pub bool);
// (⊕, ⊗, 0̄, 1̄) = (∨, ∧, false, true)
```

Used for: exact match, regex match.

#### `CountingSemiring`

```rust
pub struct CountingSemiring(pub u64);
// (⊕, ⊗, 0̄, 1̄) = (+, ×, 0, 1)
```

Used for: BLEU, ROUGE-N, Token F1, Pass@k.

#### `BoundedCountingSemiring`

```rust
pub struct BoundedCountingSemiring {
    pub value: u64,
    pub bound: u64,
}
// Like CountingSemiring but add is capped at bound
```

Used for: bounded n-gram counting.

#### `TropicalSemiring`

```rust
pub struct TropicalSemiring(pub f64);
// (⊕, ⊗, 0̄, 1̄) = (min, +, ∞, 0)
```

Used for: shortest-path, edit distance.

#### `MaxPlusSemiring`

```rust
pub struct MaxPlusSemiring(pub f64);
// (⊕, ⊗, 0̄, 1̄) = (max, +, −∞, 0)
```

Used for: ROUGE-L (longest common subsequence).

**Example — Using semirings:**

```rust
use spectacles_core::wfa::semiring::*;

// Counting semiring
let a = CountingSemiring(3);
let b = CountingSemiring(5);
assert_eq!(a.add(&b), CountingSemiring(8));  // 3 + 5 = 8
assert_eq!(a.mul(&b), CountingSemiring(15)); // 3 × 5 = 15

// Boolean semiring
let t = BooleanSemiring(true);
let f = BooleanSemiring(false);
assert_eq!(t.add(&f), BooleanSemiring(true));   // true ∨ false = true
assert_eq!(t.mul(&f), BooleanSemiring(false));   // true ∧ false = false

// Tropical semiring
let x = TropicalSemiring(3.0);
let y = TropicalSemiring(5.0);
assert_eq!(x.add(&y), TropicalSemiring(3.0));  // min(3, 5) = 3
assert_eq!(x.mul(&y), TropicalSemiring(8.0));  // 3 + 5 = 8
```

### WFA Automaton

**File:** `wfa/automaton.rs`

#### `Symbol`

Alphabet symbol types.

```rust
pub enum Symbol {
    Char(char),       // Unicode character
    Byte(u8),         // Raw byte
    Token(String),    // String token
    Epsilon,          // ε-transition
    Wildcard,         // Matches any symbol
    Id(usize),        // Numeric identifier
}
```

#### `Alphabet`

Ordered set of symbols with bijection to indices.

```rust
pub struct Alphabet {
    // internal ordered set
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new() -> Self` | Create an empty alphabet |
| `from_chars` | `fn from_chars(chars: &[char]) -> Self` | Create from character slice |
| `from_strings` | `fn from_strings(strings: &[&str]) -> Self` | Create from string slice |
| `from_range` | `fn from_range(start: char, end: char) -> Self` | Create from character range |
| `with_epsilon` | `fn with_epsilon(self) -> Self` | Add epsilon to the alphabet |
| `insert` | `fn insert(&mut self, symbol: Symbol) -> usize` | Insert a symbol, return its index |
| `contains` | `fn contains(&self, symbol: &Symbol) -> bool` | Check if a symbol is present |
| `index_of` | `fn index_of(&self, symbol: &Symbol) -> Option<usize>` | Get the index of a symbol |
| `symbol_at` | `fn symbol_at(&self, index: usize) -> Option<&Symbol>` | Get the symbol at an index |
| `len` | `fn len(&self) -> usize` | Number of symbols |
| `union` | `fn union(&self, other: &Alphabet) -> Alphabet` | Set union of two alphabets |
| `intersection` | `fn intersection(&self, other: &Alphabet) -> Alphabet` | Set intersection |
| `index_mapping` | `fn index_mapping(&self, other: &Alphabet) -> Vec<Option<usize>>` | Map indices between alphabets |

#### `Transition<S>`

A single weighted transition.

```rust
pub struct Transition<S: Semiring> {
    pub from: usize,
    pub to: usize,
    pub symbol: Symbol,
    pub weight: S,
}
```

#### `WeightedFiniteAutomaton<S>`

The core WFA data structure.

```rust
pub struct WeightedFiniteAutomaton<S: Semiring> {
    num_states: usize,
    alphabet_size: usize,
    initial_weights: Vec<S>,
    final_weights: Vec<S>,
    transitions: Vec<Vec<Vec<S>>>,  // [state][symbol][state]
}
```

**Constructor Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(num_states: usize, alphabet: &Alphabet) -> Self` | Create WFA with zero weights |
| `from_transitions` | `fn from_transitions(states: usize, alphabet: &Alphabet, transitions: Vec<Transition<S>>) -> Self` | Create from transition list |
| `single_symbol` | `fn single_symbol(alphabet: &Alphabet, symbol: &Symbol, weight: S) -> Self` | Single-symbol recognizer |
| `epsilon_wfa` | `fn epsilon_wfa(alphabet: &Alphabet) -> Self` | Recognizes only ε |
| `empty` | `fn empty(alphabet: &Alphabet) -> Self` | Recognizes nothing |
| `universal` | `fn universal(alphabet: &Alphabet, weight: S) -> Self` | Recognizes everything |

**State Management:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_state` | `fn add_state(&mut self) -> usize` | Add a state, return its index |
| `set_state_label` | `fn set_state_label(&mut self, state: usize, label: &str)` | Set a state label |
| `state_count` | `fn state_count(&self) -> usize` | Number of states |
| `alphabet` | `fn alphabet(&self) -> &Alphabet` | Reference to the alphabet |

**Transition Management:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_transition` | `fn add_transition(&mut self, from: usize, symbol: &Symbol, to: usize, weight: S)` | Add a weighted transition |
| `transition_weight` | `fn transition_weight(&self, from: usize, symbol: &Symbol, to: usize) -> &S` | Get transition weight |
| `transitions_from` | `fn transitions_from(&self, state: usize) -> Vec<Transition<S>>` | All transitions from a state |
| `all_transitions` | `fn all_transitions(&self) -> Vec<Transition<S>>` | All transitions in the WFA |

**Weight Management:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `set_initial_weight` | `fn set_initial_weight(&mut self, state: usize, weight: S)` | Set initial weight for a state |
| `set_final_weight` | `fn set_final_weight(&mut self, state: usize, weight: S)` | Set final weight for a state |
| `initial_weights` | `fn initial_weights(&self) -> &Vec<S>` | All initial weights |
| `final_weights` | `fn final_weights(&self) -> &Vec<S>` | All final weights |

**Computation:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `compute_weight` | `fn compute_weight(&self, input: &[Symbol]) -> S` | Compute the weight of an input string |
| `forward_vectors` | `fn forward_vectors(&self, input: &[Symbol]) -> Vec<Vec<S>>` | Forward weight vectors at each position |
| `backward_vectors` | `fn backward_vectors(&self, input: &[Symbol]) -> Vec<Vec<S>>` | Backward weight vectors at each position |
| `accepts` | `fn accepts(&self, input: &[Symbol]) -> bool` | Whether the WFA accepts the input (weight ≠ zero) |
| `weight_matrix` | `fn weight_matrix(&self, symbol: &Symbol) -> Vec<Vec<S>>` | Transition matrix for a symbol |

**Operations:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `union` | `fn union(&self, other: &Self) -> Self` | WFA union (⊕ of weights) |
| `concatenation` | `fn concatenation(&self, other: &Self) -> Self` | WFA concatenation |
| `intersection` | `fn intersection(&self, other: &Self) -> Self` | WFA intersection (product construction) |
| `reverse` | `fn reverse(&self) -> Self` | Reverse all transitions |
| `kleene_star` | `fn kleene_star(&self) -> Self` | Kleene closure |
| `complement` | `fn complement(&self) -> Self` | Complement (for Boolean semiring) |

**Optimization:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `determinize` | `fn determinize(&self) -> Self` | Subset construction |
| `remove_epsilon` | `fn remove_epsilon(&self) -> Self` | ε-elimination |
| `trim` | `fn trim(&self) -> Self` | Remove unreachable/unproductive states |
| `merge_parallel_transitions` | `fn merge_parallel_transitions(&mut self)` | Merge parallel edges |
| `remove_zero_transitions` | `fn remove_zero_transitions(&mut self)` | Remove zero-weight edges |

**Analysis:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `is_deterministic` | `fn is_deterministic(&self) -> bool` | Check determinism |
| `is_acyclic` | `fn is_acyclic(&self) -> bool` | Check for cycles |
| `reachable_states` | `fn reachable_states(&self) -> Vec<usize>` | States reachable from initial |
| `productive_states` | `fn productive_states(&self) -> Vec<usize>` | States that can reach final |
| `strongly_connected_components` | `fn strongly_connected_components(&self) -> Vec<Vec<usize>>` | Tarjan's SCC |
| `stats` | `fn stats(&self) -> WfaStats` | Size and complexity statistics |
| `active_states` | `fn active_states(&self) -> Vec<usize>` | States with non-zero activity |
| `used_symbols` | `fn used_symbols(&self) -> Vec<Symbol>` | Symbols appearing in transitions |
| `in_degrees` | `fn in_degrees(&self) -> Vec<usize>` | In-degree per state |
| `out_degrees` | `fn out_degrees(&self) -> Vec<usize>` | Out-degree per state |
| `topological_order` | `fn topological_order(&self) -> Option<Vec<usize>>` | Topological sort (if acyclic) |

**Serialization:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `to_dot` | `fn to_dot(&self) -> String` | Export to Graphviz DOT format |
| `to_json` | `fn to_json(&self) -> String` | Serialize to JSON |
| `from_json` | `fn from_json(json: &str) -> Result<Self, WfaError>` | Deserialize from JSON |
| `from_regex_str` | `fn from_regex_str(regex: &str, alphabet: &Alphabet) -> Result<Self, WfaError>` | Construct from regex string |

#### `WFABuilder`

Fluent builder for constructing WFAs.

```rust
pub struct WFABuilder<S: Semiring> {
    // builder state
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(alphabet: Alphabet) -> Self` | Start building a WFA |
| `initial` | `fn initial(self, state: usize, weight: S) -> Self` | Set initial weight |
| `final_state` | `fn final_state(self, state: usize, weight: S) -> Self` | Set final weight |
| `transition` | `fn transition(self, from: usize, symbol: Symbol, to: usize, weight: S) -> Self` | Add transition |
| `label` | `fn label(self, state: usize, name: &str) -> Self` | Label a state |
| `build` | `fn build(self) -> Result<WeightedFiniteAutomaton<S>, WfaError>` | Build the WFA |

#### `WfaError`

```rust
pub enum WfaError {
    InvalidState(usize),
    DimensionMismatch { expected: usize, found: usize },
    EmptyAutomaton,
    NonDeterministic,
    InvalidSymbol(Symbol),
    SerializationError(String),
    // ... additional variants
}
```

**Example — Building and using a WFA:**

```rust
use spectacles_core::wfa::*;

// Build a WFA that counts occurrences of "ab"
let alphabet = Alphabet::from_chars(&['a', 'b']);
let wfa = WFABuilder::new(alphabet.clone())
    .initial(0, CountingSemiring::one())
    .final_state(0, CountingSemiring::one())
    .final_state(2, CountingSemiring::one())
    .transition(0, Symbol::Char('a'), 1, CountingSemiring::one())
    .transition(1, Symbol::Char('b'), 2, CountingSemiring::one())
    .transition(2, Symbol::Char('a'), 1, CountingSemiring::one())
    .build()?;

// Compute weight for input "ab"
let input = vec![Symbol::Char('a'), Symbol::Char('b')];
let weight = wfa.compute_weight(&input);
println!("Weight: {:?}", weight);

// Check structural properties
println!("States: {}", wfa.state_count());
println!("Deterministic: {}", wfa.is_deterministic());
println!("Acyclic: {}", wfa.is_acyclic());

// Export to DOT
let dot = wfa.to_dot();
println!("{}", dot);
```

### WFA Transducer

**File:** `wfa/transducer.rs`

Weighted finite-state transducers for input/output transformations.

#### `WeightedTransducer<S>`

```rust
pub struct WeightedTransducer<S: Semiring> {
    // transducer state
}
```

Extends the WFA with input/output label pairs on each transition. Used for
tokenization, normalization, and other text transformations in the pipeline.

### WFA Minimization

**File:** `wfa/minimization.rs`

Hopcroft-style minimization adapted for weighted automata.

**Key Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `minimize` | `fn minimize<S: Semiring>(wfa: &WeightedFiniteAutomaton<S>) -> WeightedFiniteAutomaton<S>` | Minimize a WFA preserving its formal power series |

The minimization algorithm:

1. Compute equivalence classes of states based on their future behavior
2. Merge equivalent states
3. Recompute transition weights for merged states
4. Produce a minimal WFA with the same recognized FPS

**Example:**

```rust
use spectacles_core::wfa::minimization::minimize;

let minimal_wfa = minimize(&large_wfa);
println!(
    "Reduced from {} to {} states",
    large_wfa.state_count(),
    minimal_wfa.state_count()
);
```

### WFA Equivalence

**File:** `wfa/equivalence.rs`

Language equivalence testing via formal power series comparison.

**Key Functions:**

| Function | Signature | Description |
|----------|-----------|-------------|
| `are_equivalent` | `fn are_equivalent<S: Semiring>(a: &WeightedFiniteAutomaton<S>, b: &WeightedFiniteAutomaton<S>) -> bool` | Test if two WFAs recognize the same FPS |
| `find_distinguishing_word` | `fn find_distinguishing_word<S: Semiring>(a: &WeightedFiniteAutomaton<S>, b: &WeightedFiniteAutomaton<S>) -> Option<Vec<Symbol>>` | Find a word where the WFAs differ |

The equivalence algorithm:

1. Minimize both WFAs
2. Check isomorphism of the minimal WFAs
3. If not isomorphic, find a distinguishing word as a counterexample

**Example:**

```rust
use spectacles_core::wfa::equivalence::*;

let equivalent = are_equivalent(&wfa1, &wfa2);
if !equivalent {
    if let Some(word) = find_distinguishing_word(&wfa1, &wfa2) {
        println!("WFAs differ on: {:?}", word);
    }
}
```

### WFA Operations

**File:** `wfa/operations.rs`

Algebraic operations on WFAs.

| Function | Signature | Description |
|----------|-----------|-------------|
| `union` | `fn union<S: Semiring>(a: &WFA<S>, b: &WFA<S>) -> WFA<S>` | L₁ ∪ L₂ |
| `concatenation` | `fn concatenation<S: Semiring>(a: &WFA<S>, b: &WFA<S>) -> WFA<S>` | L₁ · L₂ |
| `kleene_star` | `fn kleene_star<S: Semiring>(a: &WFA<S>) -> WFA<S>` | L* |
| `intersection` | `fn intersection<S: Semiring>(a: &WFA<S>, b: &WFA<S>) -> WFA<S>` | L₁ ∩ L₂ (product construction) |
| `complement` | `fn complement<S: Semiring>(a: &WFA<S>) -> WFA<S>` | Σ* \ L |
| `reverse` | `fn reverse<S: Semiring>(a: &WFA<S>) -> WFA<S>` | Reverse all transitions |

### WFA Formal Power Series

**File:** `wfa/formal_power_series.rs`

Algebraic semantics of WFAs as formal power series over the free monoid.

A formal power series S: Σ* → K maps each word to a semiring element.
The WFA recognizes the FPS defined by:

```
S(w) = α^T · M(w₁) · M(w₂) · ... · M(wₙ) · β
```

This module provides operations on FPS independently of their WFA representations.

### WFA Field Embedding

**File:** `wfa/field_embedding.rs`

Maps semiring elements into the Goldilocks prime field for circuit compilation.

| Function | Signature | Description |
|----------|-----------|-------------|
| `embed_boolean` | `fn embed_boolean(b: &BooleanSemiring) -> GoldilocksField` | false→0, true→1 |
| `embed_counting` | `fn embed_counting(c: &CountingSemiring) -> GoldilocksField` | n → n mod p |
| `embed_wfa` | `fn embed_wfa<S: Semiring>(wfa: &WFA<S>) -> WFA<GoldilocksField>` | Embed entire WFA |

The embedding is a semiring homomorphism:

```
h(a ⊕ b) = h(a) + h(b)   (mod p)
h(a ⊗ b) = h(a) × h(b)   (mod p)
```

**Example:**

```rust
use spectacles_core::wfa::field_embedding::embed_wfa;

let field_wfa = embed_wfa(&counting_wfa);
// field_wfa is now a WFA<GoldilocksField> ready for circuit compilation
```

---

## Circuit Module

**Path:** `spectacles-core/src/circuit/`

The circuit module compiles WFAs into STARK proof circuits over the Goldilocks
prime field.

### Circuit Goldilocks

**File:** `circuit/goldilocks.rs`

#### `GoldilocksField`

Prime field element with modulus p = 2⁶⁴ − 2³² + 1.

```rust
pub struct GoldilocksField(pub u64);

pub const MODULUS: u64 = 0xFFFFFFFF00000001;
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(value: u64) -> Self` | Create a field element (reduced mod p) |
| `zero` | `fn zero() -> Self` | Additive identity |
| `one` | `fn one() -> Self` | Multiplicative identity |
| `add` | `fn add(&self, other: &Self) -> Self` | Field addition |
| `sub` | `fn sub(&self, other: &Self) -> Self` | Field subtraction |
| `mul` | `fn mul(&self, other: &Self) -> Self` | Field multiplication |
| `div` | `fn div(&self, other: &Self) -> Self` | Field division (mul by inverse) |
| `inv` | `fn inv(&self) -> Self` | Multiplicative inverse (via Fermat) |
| `pow` | `fn pow(&self, exp: u64) -> Self` | Modular exponentiation |

`GoldilocksField` implements `Semiring`, so it can be used directly as
a WFA weight type.

#### `GoldilocksExt`

Quadratic extension field for FRI query domain expansion.

```rust
pub struct GoldilocksExt {
    pub re: GoldilocksField,
    pub im: GoldilocksField,
}
```

**Example:**

```rust
use spectacles_core::circuit::goldilocks::*;

let a = GoldilocksField::new(42);
let b = GoldilocksField::new(17);

let sum = a.add(&b);      // 42 + 17 = 59
let prod = a.mul(&b);     // 42 × 17 = 714
let inv_a = a.inv();       // a⁻¹ such that a × a⁻¹ = 1
let quotient = a.div(&b); // a × b⁻¹

assert_eq!(a.mul(&inv_a), GoldilocksField::one());
```

### Circuit AIR

**File:** `circuit/air.rs`

Algebraic Intermediate Representation for constraint systems.

#### `ConstraintType`

```rust
pub enum ConstraintType {
    Boundary,     // Constraints on specific rows
    Transition,   // Row-to-row constraints
    Periodic,     // Repeating constraints
    Composition,  // Cross-column constraints
}
```

#### `ColumnType`

```rust
pub enum ColumnType {
    State,      // WFA state columns
    Input,      // Input symbol columns
    Auxiliary,  // Helper columns for gadgets
    Public,     // Public input columns
}
```

#### `SymbolicExpression`

Expression tree for constraint polynomials.

```rust
pub enum SymbolicExpression {
    Constant(GoldilocksField),
    Variable(String),
    Add(Box<SymbolicExpression>, Box<SymbolicExpression>),
    Mul(Box<SymbolicExpression>, Box<SymbolicExpression>),
    Sub(Box<SymbolicExpression>, Box<SymbolicExpression>),
    Neg(Box<SymbolicExpression>),
    Pow(Box<SymbolicExpression>, u64),
    CurrentRow(usize),    // column value at current row
    NextRow(usize),       // column value at next row
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `zero` | `fn zero() -> Self` | Constant zero expression |
| `one` | `fn one() -> Self` | Constant one expression |
| `constant` | `fn constant(val: u64) -> Self` | Constant from u64 |
| `constant_field` | `fn constant_field(val: GoldilocksField) -> Self` | Constant from field element |
| `var` | `fn var(name: &str) -> Self` | Named variable |
| `cur` | `fn cur(col: usize) -> Self` | Current row reference |
| `nxt` | `fn nxt(col: usize) -> Self` | Next row reference |
| `evaluate` | `fn evaluate(&self, current: &[GoldilocksField], next: &[GoldilocksField]) -> GoldilocksField` | Evaluate with two-row window |
| `evaluate_single_row` | `fn evaluate_single_row(&self, row: &[GoldilocksField]) -> GoldilocksField` | Evaluate with single row |
| `degree` | `fn degree(&self) -> usize` | Polynomial degree |
| `simplify` | `fn simplify(&self) -> Self` | Algebraic simplification |
| `normalize` | `fn normalize(&self) -> Self` | Canonical form |
| `substitute` | `fn substitute(&self, var: &str, expr: &Self) -> Self` | Variable substitution |
| `substitute_all` | `fn substitute_all(&self, mapping: &HashMap<String, Self>) -> Self` | Multi-variable substitution |
| `is_constant` | `fn is_constant(&self) -> bool` | Check if expression is a constant |
| `constant_value` | `fn constant_value(&self) -> Option<GoldilocksField>` | Extract constant value |
| `variables_used` | `fn variables_used(&self) -> HashSet<String>` | All referenced variables |
| `node_count` | `fn node_count(&self) -> usize` | Expression tree size |
| `uses_next_row` | `fn uses_next_row(&self) -> bool` | Whether expression references NextRow |

#### `AIRConstraint`

```rust
pub struct AIRConstraint {
    pub name: String,
    pub expr: SymbolicExpression,
    pub constraint_type: ConstraintType,
    pub boundary_row: Option<usize>,
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(name: &str, expr: SymbolicExpression, ct: ConstraintType) -> Self` | Create a constraint |
| `new_periodic` | `fn new_periodic(name: &str, expr: SymbolicExpression, period: usize) -> Self` | Create periodic constraint |
| `verify_at_row` | `fn verify_at_row(&self, trace: &AIRTrace, row: usize) -> bool` | Check constraint at a row |
| `verify_boundary` | `fn verify_boundary(&self, trace: &AIRTrace) -> bool` | Check boundary constraint |
| `is_boundary` | `fn is_boundary(&self) -> bool` | Type check |
| `is_transition` | `fn is_transition(&self) -> bool` | Type check |
| `is_periodic` | `fn is_periodic(&self) -> bool` | Type check |
| `composition_degree` | `fn composition_degree(&self) -> usize` | Degree for composition |
| `apply_to_trace_window` | `fn apply_to_trace_window(&self, current: &[GoldilocksField], next: &[GoldilocksField]) -> GoldilocksField` | Evaluate on trace window |
| `referenced_columns` | `fn referenced_columns(&self) -> Vec<usize>` | Columns used by this constraint |
| `applies_at_row` | `fn applies_at_row(&self, row: usize, trace_len: usize) -> bool` | Whether constraint applies at given row |

#### `AIRProgram`

Complete constraint system.

```rust
pub struct AIRProgram {
    pub constraints: Vec<AIRConstraint>,
    pub layout: TraceLayout,
    pub public_inputs: Vec<GoldilocksField>,
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(layout: TraceLayout) -> Self` | Create an empty program |
| `add_constraint` | `fn add_constraint(&mut self, constraint: AIRConstraint)` | Add a constraint |
| `add_boundary_constraint` | `fn add_boundary_constraint(&mut self, name: &str, row: usize, expr: SymbolicExpression)` | Add boundary constraint |
| `add_transition_constraint` | `fn add_transition_constraint(&mut self, name: &str, expr: SymbolicExpression)` | Add transition constraint |
| `add_periodic_constraint` | `fn add_periodic_constraint(&mut self, name: &str, expr: SymbolicExpression, period: usize)` | Add periodic constraint |
| `add_periodic_column` | `fn add_periodic_column(&mut self, column: PeriodicColumn)` | Add a periodic column |
| `verify_trace` | `fn verify_trace(&self, trace: &AIRTrace) -> bool` | Verify all constraints against a trace |

#### `PeriodicColumn`

```rust
pub struct PeriodicColumn {
    pub name: String,
    pub values: Vec<GoldilocksField>,
    // values repeat cyclically over the trace
}
```

#### `BoundaryDescriptor`

```rust
pub struct BoundaryDescriptor {
    pub column: usize,
    pub row: usize,
    pub value: GoldilocksField,
}
```

**Example — Building an AIR program:**

```rust
use spectacles_core::circuit::air::*;
use spectacles_core::circuit::goldilocks::GoldilocksField;

// Create a trace layout with 3 state columns and 1 input column
let layout = TraceLayout::from_wfa_state_count(3, 1);
let mut program = AIRProgram::new(layout);

// Boundary: state_0 starts at 1 (initial weight)
program.add_boundary_constraint(
    "initial_state_0",
    0,
    SymbolicExpression::Sub(
        Box::new(SymbolicExpression::cur(0)),
        Box::new(SymbolicExpression::one()),
    ),
);

// Transition: state_0(next) = state_0(cur) * input(cur)
program.add_transition_constraint(
    "state_0_transition",
    SymbolicExpression::Sub(
        Box::new(SymbolicExpression::nxt(0)),
        Box::new(SymbolicExpression::Mul(
            Box::new(SymbolicExpression::cur(0)),
            Box::new(SymbolicExpression::cur(3)),
        )),
    ),
);
```

### Circuit Compiler

**File:** `circuit/compiler.rs`

#### `WFACircuitCompiler`

Compiles a WFA into an AIR program.

```rust
pub struct WFACircuitCompiler {
    // compilation state
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new() -> Self` | Create a compiler |
| `compile` | `fn compile<S: Semiring>(&mut self, wfa: &WeightedFiniteAutomaton<S>) -> Result<AIRProgram, CompileError>` | Compile WFA to AIR |
| `compile_with_gadgets` | `fn compile_with_gadgets<S: Semiring>(&mut self, wfa: &WeightedFiniteAutomaton<S>, gadgets: &[Gadget]) -> Result<AIRProgram, CompileError>` | Tier 2 compilation with gadgets |
| `estimate_constraints` | `fn estimate_constraints<S: Semiring>(&self, wfa: &WeightedFiniteAutomaton<S>) -> usize` | Estimate constraint count |

**Example:**

```rust
use spectacles_core::circuit::compiler::WFACircuitCompiler;

let mut compiler = WFACircuitCompiler::new();
let air_program = compiler.compile(&field_wfa)?;

println!("Constraints: {}", air_program.constraints.len());
println!("Columns: {}", air_program.layout.column_names().len());
```

### Circuit Trace

**File:** `circuit/trace.rs`

#### `TraceLayout`

Describes the column schema of an execution trace.

```rust
pub struct TraceLayout {
    // column descriptions
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new() -> Self` | Create an empty layout |
| `from_wfa_state_count` | `fn from_wfa_state_count(states: usize, inputs: usize) -> Self` | Create layout for a WFA |
| `add_column` | `fn add_column(&mut self, name: &str, column_type: ColumnType)` | Add a column |
| `column_index_by_name` | `fn column_index_by_name(&self, name: &str) -> Option<usize>` | Lookup column by name |
| `validate` | `fn validate(&self) -> bool` | Validate layout consistency |
| `column_names` | `fn column_names(&self) -> Vec<&str>` | All column names |
| `state_column_count` | `fn state_column_count(&self) -> usize` | Number of state columns |
| `input_column_count` | `fn input_column_count(&self) -> usize` | Number of input columns |
| `aux_column_count` | `fn aux_column_count(&self) -> usize` | Number of auxiliary columns |
| `public_column_count` | `fn public_column_count(&self) -> usize` | Number of public columns |

#### `AIRTrace`

2D table of field elements serving as the execution witness.

```rust
pub struct AIRTrace {
    // rows × columns of GoldilocksField
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(rows: usize, cols: usize) -> Self` | Create zero-filled trace |
| `new_raw` | `fn new_raw(data: Vec<Vec<GoldilocksField>>) -> Self` | Create from raw data |
| `from_rows` | `fn from_rows(rows: Vec<Vec<GoldilocksField>>) -> Self` | Create from row vectors |
| `from_columns` | `fn from_columns(cols: Vec<Vec<GoldilocksField>>) -> Self` | Create from column vectors |
| `transpose` | `fn transpose(&self) -> Self` | Transpose rows/columns |
| `get` | `fn get(&self, row: usize, col: usize) -> &GoldilocksField` | Get element |
| `set` | `fn set(&mut self, row: usize, col: usize, val: GoldilocksField)` | Set element |
| `get_row` | `fn get_row(&self, row: usize) -> &[GoldilocksField]` | Get entire row |
| `get_column` | `fn get_column(&self, col: usize) -> Vec<GoldilocksField>` | Get entire column |
| `set_row` | `fn set_row(&mut self, row: usize, values: &[GoldilocksField])` | Set entire row |
| `set_column` | `fn set_column(&mut self, col: usize, values: &[GoldilocksField])` | Set entire column |
| `push_row` | `fn push_row(&mut self, row: Vec<GoldilocksField>)` | Append a row |
| `pad_to_power_of_two` | `fn pad_to_power_of_two(&mut self)` | Pad rows to next power of 2 |
| `validate_dimensions` | `fn validate_dimensions(&self) -> bool` | Check dimensional consistency |
| `window_at` | `fn window_at(&self, row: usize) -> (&[GoldilocksField], &[GoldilocksField])` | Two-row window for transition constraints |
| `sub_trace` | `fn sub_trace(&self, rows: Range<usize>, cols: Range<usize>) -> Self` | Extract sub-trace |
| `extend_trace` | `fn extend_trace(&mut self, other: &AIRTrace)` | Append another trace |
| `is_zero` | `fn is_zero(&self) -> bool` | All elements are zero |
| `column_fingerprints` | `fn column_fingerprints(&self) -> Vec<GoldilocksField>` | Hash fingerprint per column |
| `equals` | `fn equals(&self, other: &AIRTrace) -> bool` | Element-wise comparison |

**Example:**

```rust
use spectacles_core::circuit::trace::*;
use spectacles_core::circuit::goldilocks::GoldilocksField;

let mut trace = AIRTrace::new(8, 4);
trace.set(0, 0, GoldilocksField::one());  // Initial state
trace.set(0, 3, GoldilocksField::new(5)); // First input symbol

// Pad to power-of-two length for FRI
trace.pad_to_power_of_two();

// Get a window for transition constraint checking
let (current, next) = trace.window_at(0);
```

### Circuit STARK

**File:** `circuit/stark.rs`

STARK prover and verifier.

#### `STARKProver`

```rust
pub struct STARKProver {
    // prover configuration
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(security_bits: usize) -> Self` | Create prover with security parameter |
| `prove` | `fn prove(&self, program: &AIRProgram, trace: &AIRTrace) -> Result<STARKProof, ProverError>` | Generate STARK proof |

#### `STARKVerifier`

```rust
pub struct STARKVerifier {
    // verifier configuration
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(security_bits: usize) -> Self` | Create verifier |
| `verify` | `fn verify(&self, program: &AIRProgram, proof: &STARKProof) -> Result<bool, VerifierError>` | Verify a STARK proof |

#### `STARKProof`

```rust
pub struct STARKProof {
    pub trace_commitments: Vec<[u8; 32]>,
    pub composition_commitment: [u8; 32],
    pub fri_proof: FRIProof,
    pub query_responses: Vec<QueryResponse>,
    pub public_inputs: Vec<GoldilocksField>,
}
```

**Example:**

```rust
use spectacles_core::circuit::stark::*;

let prover = STARKProver::new(128);
let proof = prover.prove(&air_program, &trace)?;

let verifier = STARKVerifier::new(128);
let valid = verifier.verify(&air_program, &proof)?;
assert!(valid);
```

### Circuit FRI

**File:** `circuit/fri.rs`

Fast Reed-Solomon Interactive oracle proofs of proximity.

#### `FRIProtocol`

```rust
pub struct FRIProtocol {
    // FRI configuration
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(blowup_factor: usize, num_queries: usize) -> Self` | Create FRI protocol |
| `commit` | `fn commit(&self, polynomial_evaluations: &[GoldilocksField]) -> FRICommitment` | Commit to evaluations |
| `prove_low_degree` | `fn prove_low_degree(&self, commitment: &FRICommitment) -> FRIProof` | Prove low degree |
| `verify_low_degree` | `fn verify_low_degree(&self, proof: &FRIProof) -> bool` | Verify low degree proof |

### Circuit Merkle

**File:** `circuit/merkle.rs`

Merkle tree commitments for trace columns. (Also available via `utils/hash.rs`.)

### Circuit Gadgets

**File:** `circuit/gadgets.rs`

Reusable arithmetic sub-circuits for Tier 2 compilation.

| Gadget | Description | Constraints |
|--------|-------------|-------------|
| `BitDecomposition` | Decompose x into k bits | O(log p) |
| `RangeCheck` | Verify 0 ≤ x < 2^k | O(k) |
| `Comparison` | Verify x ≥ y | O(log p) |
| `Division` | Compute x / y with verified remainder | O(1) + range check |
| `Maximum` | Compute max(x, y) | O(log p) |

Each gadget produces a set of `AIRConstraint`s and auxiliary columns that are
added to the `AIRProgram`.

**Example:**

```rust
use spectacles_core::circuit::gadgets::*;

// Create a comparison gadget for ROUGE-L
let comparison = Comparison::new(column_x, column_y, result_column);
let constraints = comparison.generate_constraints();
for c in constraints {
    program.add_constraint(c);
}
```

---

## Protocol Module

**Path:** `spectacles-core/src/protocol/`

The protocol module implements the commit-reveal-verify state machine,
commitment schemes, Fiat-Shamir transcripts, and evaluation certificates.

### Protocol State Machine

**File:** `protocol/state_machine.rs`

#### `ProtocolState`

```rust
pub enum ProtocolState {
    Initialized,
    CommitOutputs,
    RevealBenchmark,
    Evaluate,
    Prove,
    Verify,
    Certify,
    Completed,
    Aborted(AbortReason),
    TimedOut,
}
```

#### `AbortReason`

```rust
pub enum AbortReason {
    ConstraintViolation,
    TimeoutExceeded,
    InvalidTransition,
    CommitmentMismatch,
    ProofFailed,
    ExternalAbort,
}
```

#### `EventType`

```rust
pub enum EventType {
    StateTransition,
    Timeout,
    Error,
    Commitment,
    Reveal,
    ProofGenerated,
    ProofVerified,
    CertificateIssued,
}
```

#### `ProtocolEvent`

```rust
pub struct ProtocolEvent {
    pub from_state: ProtocolState,
    pub to_state: ProtocolState,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub event_type: EventType,
    pub details: String,
}
```

#### `ProtocolConfig`

```rust
pub struct ProtocolConfig {
    pub name: String,
    pub version: String,
    pub timeouts: HashMap<ProtocolState, Duration>,
    pub enable_logging: bool,
    pub max_retries: usize,
    pub require_grinding: bool,
}
```

#### `ProtocolStateMachine`

Core state machine managing protocol flow.

```rust
pub struct ProtocolStateMachine {
    // state, config, event log, commitments
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(config: ProtocolConfig) -> Self` | Create state machine |
| `current_state` | `fn current_state(&self) -> &ProtocolState` | Get current state |
| `transition_to` | `fn transition_to(&mut self, state: ProtocolState) -> Result<(), ProtocolError>` | Transition to new state |
| `can_transition_to` | `fn can_transition_to(&self, state: &ProtocolState) -> bool` | Check if transition is valid |
| `valid_transitions` | `fn valid_transitions(&self) -> Vec<ProtocolState>` | List valid next states |
| `abort` | `fn abort(&mut self, reason: AbortReason)` | Abort the protocol |
| `is_terminal` | `fn is_terminal(&self) -> bool` | Whether current state is terminal |
| `check_timeout` | `fn check_timeout(&self) -> bool` | Check if current state has timed out |
| `elapsed_in_state` | `fn elapsed_in_state(&self) -> Duration` | Time in current state |
| `total_elapsed` | `fn total_elapsed(&self) -> Duration` | Total protocol duration |
| `event_log` | `fn event_log(&self) -> &[ProtocolEvent]` | Full event history |
| `store_commitment` | `fn store_commitment(&mut self, id: &str, commitment: Vec<u8>)` | Store a commitment |
| `store_reveal` | `fn store_reveal(&mut self, id: &str, reveal: Vec<u8>)` | Store a reveal value |
| `verify_commitment_reveal` | `fn verify_commitment_reveal(&self, id: &str) -> bool` | Verify commitment-reveal pair |
| `reset` | `fn reset(&mut self)` | Reset to Initialized |
| `serialize_state` | `fn serialize_state(&self) -> Vec<u8>` | Serialize for checkpoint |
| `deserialize_state` | `fn deserialize_state(data: &[u8]) -> Result<Self, ProtocolError>` | Restore from checkpoint |
| `run_protocol` | `async fn run_protocol(&mut self) -> Result<(), ProtocolError>` | Run full protocol to completion |

#### `ProtocolError`

```rust
pub enum ProtocolError {
    InvalidTransition { from: ProtocolState, to: ProtocolState },
    Timeout { state: ProtocolState, elapsed: Duration },
    AlreadyTerminal,
    CommitmentMismatch { id: String },
    SerializationError(String),
    InvalidState(String),
}
```

#### `ProtocolPhaseManager`

Manages execution of named phases with timeouts.

```rust
pub struct ProtocolPhaseManager { /* ... */ }
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(phases: Vec<ProtocolPhase>) -> Self` | Create phase manager |
| `advance` | `fn advance(&mut self) -> Option<&ProtocolPhase>` | Move to next phase |
| `current_phase` | `fn current_phase(&self) -> Option<&ProtocolPhase>` | Current phase |
| `is_complete` | `fn is_complete(&self) -> bool` | All phases done |
| `record_result` | `fn record_result(&mut self, result: PhaseResult)` | Record phase outcome |
| `phase_durations` | `fn phase_durations(&self) -> Vec<(&str, Duration)>` | Timing for each phase |

#### `BackoffStrategy`

```rust
pub enum BackoffStrategy {
    Exponential { base_ms: u64, max_ms: u64 },
    Linear { step_ms: u64, max_ms: u64 },
    Fixed { delay_ms: u64 },
}
```

#### `RetryManager`

```rust
pub struct RetryManager { /* ... */ }
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(max_retries: usize, strategy: BackoffStrategy) -> Self` | Create retry manager |
| `should_retry` | `fn should_retry(&self) -> bool` | Whether retry is allowed |
| `record_attempt` | `fn record_attempt(&mut self)` | Record a retry attempt |
| `reset` | `fn reset(&mut self)` | Reset retry counter |
| `delay_ms` | `fn delay_ms(&self) -> u64` | Compute next delay |

#### Additional Types

- **`ProtocolAuditor`** — records `AuditEvent` instances, verifies audit trail integrity, provides `generate_audit_log()` and `verify_audit_trail()`.
- **`ProtocolSimulator`** — simulates protocol execution with `run_simulation()`, `run_with_failures()`, `run_with_delays()`.
- **`StateGraph`** — graph representation from `StateTransitionRule`s with `reachable_from()`, `shortest_path()`, `is_deadlock_free()`, `to_dot()`.
- **`ProtocolRunner`** — high-level orchestrator with `register_handler()`, `run_to_completion()`, `run_single_step()`.
- **`ProtocolTemplate`** — reusable definitions via `evaluation_protocol()`, `certification_protocol()`, `verification_only()`, `to_config()`, `instantiate()`.
- **`StateHistory`** — event timeline analysis.
- **`ProtocolMetrics`** — performance metrics tracking.
- **`ProtocolCheckpoint`** — serializable snapshot for recovery.

**Example — Running a protocol:**

```rust
use spectacles_core::protocol::state_machine::*;

let config = ProtocolConfig {
    name: "evaluation".into(),
    version: "1.0".into(),
    timeouts: Default::default(),
    enable_logging: true,
    max_retries: 3,
    require_grinding: false,
};

let mut sm = ProtocolStateMachine::new(config);
assert_eq!(*sm.current_state(), ProtocolState::Initialized);

// Transition through states
sm.transition_to(ProtocolState::CommitOutputs)?;
sm.store_commitment("output_hash", hash_bytes.to_vec());

sm.transition_to(ProtocolState::RevealBenchmark)?;
sm.transition_to(ProtocolState::Evaluate)?;
sm.transition_to(ProtocolState::Prove)?;
sm.transition_to(ProtocolState::Verify)?;
sm.transition_to(ProtocolState::Certify)?;
sm.transition_to(ProtocolState::Completed)?;

assert!(sm.is_terminal());
println!("Protocol completed in {:?}", sm.total_elapsed());
```

### Protocol Commitment

**File:** `protocol/commitment.rs`

#### `CommitmentScheme` Trait

```rust
pub trait CommitmentScheme {
    fn commit(&self, data: &[u8]) -> Vec<u8>;
    fn verify(&self, commitment: &[u8], data: &[u8]) -> bool;
}
```

#### Implementations

| Type | Description |
|------|-------------|
| `HashCommitment` | BLAKE3-based hash commitment (computationally binding) |
| `PedersenCommitment` | Algebraic commitment (information-theoretically hiding) |
| `VectorCommitment` | Merkle-tree-based position-binding commitment |
| `PolynomialCommitment` | Homomorphic polynomial commitment |
| `TimelockCommitment` | Time-locked commitment with delayed revelation |

**Example:**

```rust
use spectacles_core::protocol::commitment::*;

let scheme = HashCommitment::new();
let data = b"model output scores";
let commitment = scheme.commit(data);

assert!(scheme.verify(&commitment, data));
assert!(!scheme.verify(&commitment, b"different data"));
```

### Protocol Transcript

**File:** `protocol/transcript.rs`

#### `FiatShamirTranscript`

Non-interactive transcript for converting interactive proofs.

```rust
pub struct FiatShamirTranscript {
    // hash chain state
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(domain: &str) -> Self` | Create with domain separation |
| `append_message` | `fn append_message(&mut self, label: &str, data: &[u8])` | Append a labeled message |
| `challenge_bytes` | `fn challenge_bytes(&mut self, label: &str, len: usize) -> Vec<u8>` | Generate challenge bytes |
| `challenge_field` | `fn challenge_field(&mut self, label: &str) -> GoldilocksField` | Generate field element challenge |

#### `TranscriptError`

```rust
pub enum TranscriptError {
    InvalidLabel(String),
    EmptyTranscript,
    VerificationFailed,
}
```

**Example:**

```rust
use spectacles_core::protocol::transcript::FiatShamirTranscript;

let mut transcript = FiatShamirTranscript::new("spectacles-stark");
transcript.append_message("trace_commitment", &merkle_root);
transcript.append_message("public_input", &score_bytes);

let challenge = transcript.challenge_field("composition_alpha");
```

### Protocol Certificate

**File:** `protocol/certificate.rs`

#### `EvaluationCertificate`

Cryptographic attestation that a score was computed correctly.

```rust
pub struct EvaluationCertificate {
    pub metric_name: String,
    pub score: FixedPointScore,
    pub proof_hash: [u8; 32],
    pub input_commitment: [u8; 32],
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub protocol_version: String,
}
```

#### `CertificateBuilder`

Fluent builder for certificates.

```rust
pub struct CertificateBuilder { /* ... */ }
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new() -> Self` | Start building |
| `metric` | `fn metric(self, name: &str) -> Self` | Set metric name |
| `score` | `fn score(self, score: FixedPointScore) -> Self` | Set score |
| `proof_hash` | `fn proof_hash(self, hash: [u8; 32]) -> Self` | Set proof hash |
| `input_commitment` | `fn input_commitment(self, commitment: [u8; 32]) -> Self` | Set input commitment |
| `build` | `fn build(self) -> Result<EvaluationCertificate, CertificateError>` | Build certificate |

#### `CertificateChain`

Chain of certificates for multi-metric evaluations.

```rust
pub struct CertificateChain {
    pub certificates: Vec<EvaluationCertificate>,
}
```

#### `CertificateStore`

Persistent storage and retrieval of certificates.

```rust
pub struct CertificateStore { /* ... */ }
```

#### `CertificateError`

```rust
pub enum CertificateError {
    MissingField(String),
    InvalidProofHash,
    ExpiredCertificate,
    ChainVerificationFailed,
}
```

**Example:**

```rust
use spectacles_core::protocol::certificate::*;

let cert = CertificateBuilder::new()
    .metric("bleu")
    .score(FixedPointScore { numerator: 3542, denominator: 10000 })
    .proof_hash(proof_hash)
    .input_commitment(commitment)
    .build()?;

println!("Certificate: {} = {}/{}", cert.metric_name,
    cert.score.numerator, cert.score.denominator);
```

---

## PSI Module

**Path:** `spectacles-core/src/psi/`

The PSI module implements Private Set Intersection for training data
contamination detection.

### PSI N-gram

**File:** `psi/ngram.rs`

#### `GramType`

```rust
pub enum GramType {
    Character,   // Character-level n-grams
    Token,       // Token-level n-grams (after tokenization)
    Byte,        // Byte-level n-grams
}
```

#### `NGramConfig`

```rust
pub struct NGramConfig {
    pub n: usize,               // N-gram size
    pub gram_type: GramType,    // Type of n-gram
    pub min_frequency: usize,   // Minimum frequency threshold
    pub normalized: bool,       // Whether to normalize text
}
```

#### `NGram`

```rust
pub struct NGram {
    pub tokens: Vec<String>,    // The n-gram tokens
    pub frequency: usize,       // Occurrence count
}
```

#### `NGramExtractor`

```rust
pub struct NGramExtractor {
    config: NGramConfig,
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(config: NGramConfig) -> Self` | Create extractor |
| `extract` | `fn extract(&self, text: &str) -> Vec<NGram>` | Extract n-grams from text |
| `extract_with_frequency` | `fn extract_with_frequency(&self, text: &str) -> NGramFrequencyMap` | Extract with frequency counts |
| `compare` | `fn compare(&self, text_a: &str, text_b: &str) -> f64` | Compute n-gram overlap ratio |

#### Collection Types

- **`NGramSet`** — deduplicated set of n-grams
- **`NGramFrequencyMap`** — n-gram → frequency mapping
- **`NGramIndex`** — indexed collection for fast lookup

**Example:**

```rust
use spectacles_core::psi::ngram::*;

let config = NGramConfig {
    n: 3,
    gram_type: GramType::Token,
    min_frequency: 1,
    normalized: true,
};

let extractor = NGramExtractor::new(config);
let ngrams = extractor.extract("the quick brown fox jumps over the lazy dog");
println!("Extracted {} 3-grams", ngrams.len());

let overlap = extractor.compare(
    "the quick brown fox",
    "the slow brown fox",
);
println!("N-gram overlap: {:.2}%", overlap * 100.0);
```

### PSI Trie

**File:** `psi/trie.rs`

#### `NGramTrie`

Memory-efficient n-gram storage using a trie structure.

```rust
pub struct NGramTrie {
    // trie nodes
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new() -> Self` | Create empty trie |
| `insert` | `fn insert(&mut self, ngram: &[String])` | Insert an n-gram |
| `contains` | `fn contains(&self, ngram: &[String]) -> bool` | Check membership |
| `prefix_search` | `fn prefix_search(&self, prefix: &[String]) -> Vec<Vec<String>>` | Find n-grams with prefix |
| `len` | `fn len(&self) -> usize` | Number of stored n-grams |
| `intersection` | `fn intersection(&self, other: &NGramTrie) -> NGramTrie` | Set intersection |
| `merge` | `fn merge(&mut self, other: &NGramTrie)` | Merge another trie |
| `statistics` | `fn statistics(&self) -> TrieStats` | Size and depth statistics |

#### `CompactTrie`

Memory-optimized variant for large n-gram sets.

```rust
pub struct CompactTrie {
    // compressed trie representation
}
```

**Example:**

```rust
use spectacles_core::psi::trie::NGramTrie;

let mut eval_trie = NGramTrie::new();
eval_trie.insert(&["the".into(), "quick".into(), "brown".into()]);
eval_trie.insert(&["quick".into(), "brown".into(), "fox".into()]);

let mut train_trie = NGramTrie::new();
train_trie.insert(&["the".into(), "quick".into(), "brown".into()]);
train_trie.insert(&["lazy".into(), "brown".into(), "dog".into()]);

let overlap = eval_trie.intersection(&train_trie);
println!("Overlapping n-grams: {}", overlap.len()); // 1
```

### PSI OPRF

**File:** `psi/oprf.rs`

Oblivious Pseudorandom Function protocol.

#### `OPRFKey`

```rust
pub struct OPRFKey {
    // secret key material
}
```

#### `OPRFProtocol`

```rust
pub struct OPRFProtocol {
    // OPRF state
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(config: OPRFConfig) -> Self` | Create OPRF protocol |
| `keygen` | `fn keygen(&self) -> OPRFKey` | Generate OPRF key |
| `blind` | `fn blind(&self, input: &[u8]) -> (BlindedInput, BlindingFactor)` | Blind an input |
| `evaluate` | `fn evaluate(&self, key: &OPRFKey, blinded: &BlindedInput) -> BlindedOutput` | Evaluate on blinded input |
| `unblind` | `fn unblind(&self, output: &BlindedOutput, factor: &BlindingFactor) -> Vec<u8>` | Unblind the output |
| `hash_to_field` | `fn hash_to_field(&self, input: &[u8]) -> GoldilocksField` | Hash input to field element |

#### `OPRFConfig`

```rust
pub struct OPRFConfig {
    pub security_bits: usize,
    pub hash_function: String,
}
```

#### Related Types

- **`BlindedInput`** — blinded input message
- **`BlindedOutput`** — server's blinded evaluation
- **`BlindingFactor`** — client's blinding randomness
- **`OPRFProof`** — proof of correct OPRF evaluation
- **`VerifiableOPRF`** — OPRF with verifiable computation
- **`OTExtension`** — Oblivious Transfer Extension for amortized PSI

### PSI Protocol

**File:** `psi/protocol.rs`

#### `PSIMode`

```rust
pub enum PSIMode {
    Streaming,    // Process elements one at a time
    Batch,        // Process all elements at once
    Threshold,    // Only report if overlap exceeds threshold
}
```

#### `PSIPhase`

```rust
pub enum PSIPhase {
    Setup,        // Key exchange and parameter agreement
    Hashing,      // OPRF evaluation phase
    Matching,     // Intersection computation
    Verification, // Result verification
}
```

#### `PSIConfig`

```rust
pub struct PSIConfig {
    pub mode: PSIMode,
    pub ngram_config: NGramConfig,
    pub security_bits: usize,
    pub threshold: Option<f64>,
}
```

#### `PSIResult`

```rust
pub struct PSIResult {
    pub intersection_size: usize,
    pub evaluator_set_size: usize,
    pub overlap_ratio: f64,
    pub contamination_score: f64,
}
```

#### `PSIProtocol`

```rust
pub struct PSIProtocol {
    // protocol state
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(config: PSIConfig) -> Self` | Create PSI protocol |
| `setup` | `fn setup(&mut self) -> Result<(), PSIError>` | Initialize protocol |
| `run` | `fn run(&mut self, eval_text: &str, train_ngrams: &NGramTrie) -> Result<PSIResult, PSIError>` | Run full protocol |
| `current_phase` | `fn current_phase(&self) -> &PSIPhase` | Current protocol phase |
| `generate_attestation` | `fn generate_attestation(&self, result: &PSIResult) -> ContaminationAttestation` | Create attestation |

#### `ContaminationAttestation`

```rust
pub struct ContaminationAttestation {
    pub overlap_percentage: f64,
    pub ngram_config: NGramConfig,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub protocol_version: String,
}
```

#### `ContaminationMatrix`

Cross-analysis across multiple n-gram sizes.

```rust
pub struct ContaminationMatrix {
    pub results: Vec<(usize, PSIResult)>,  // (n, result) pairs
}
```

#### `PSIError` / `PSIErrorKind`

```rust
pub enum PSIErrorKind {
    SetupFailed,
    HashingFailed,
    MatchingFailed,
    VerificationFailed,
    InvalidConfig,
}
```

**Example — Running PSI contamination detection:**

```rust
use spectacles_core::psi::*;

let config = PSIConfig {
    mode: PSIMode::Batch,
    ngram_config: NGramConfig {
        n: 5,
        gram_type: GramType::Token,
        min_frequency: 1,
        normalized: true,
    },
    security_bits: 128,
    threshold: Some(0.1), // flag if >10% overlap
};

let mut protocol = PSIProtocol::new(config);
protocol.setup()?;

let result = protocol.run(
    "the quick brown fox jumps over the lazy dog",
    &training_ngram_trie,
)?;

println!("Overlap: {:.2}%", result.overlap_ratio * 100.0);
println!("Contamination score: {:.4}", result.contamination_score);

if result.overlap_ratio > 0.1 {
    println!("WARNING: Possible contamination detected!");
}

let attestation = protocol.generate_attestation(&result);
```

---

## Scoring Module

**Path:** `spectacles-core/src/scoring/`

The scoring module implements 7 NLP evaluation metrics, each with a triple
implementation pattern (reference, WFA, circuit).

### Core Scoring Types

**File:** `scoring/mod.rs`

#### `TripleMetric` Trait

```rust
pub trait TripleMetric {
    type Input;
    type Score;

    /// Reference (standard) implementation
    fn score_reference(&self, input: &Self::Input) -> Self::Score;

    /// WFA-based implementation
    fn score_automaton(&self, input: &Self::Input) -> Self::Score;

    /// Arithmetic circuit implementation
    fn score_circuit(&self, input: &Self::Input) -> Self::Score;

    /// Score and verify all three agree
    fn score_and_verify(&self, input: &Self::Input) -> Self::Score;
}
```

#### `ScoringPair`

```rust
pub struct ScoringPair {
    pub candidate: String,
    pub reference: String,
}
```

#### `MultiRefScoringPair`

```rust
pub struct MultiRefScoringPair {
    pub candidate: String,
    pub references: Vec<String>,
}
```

#### `FixedPointScore`

```rust
pub struct FixedPointScore {
    pub numerator: u64,
    pub denominator: u64,
}
```

Exact rational representation for scores, avoiding floating-point imprecision
in the Goldilocks field.

#### `ScoringCircuit`

```rust
pub struct ScoringCircuit {
    pub num_wires: usize,
    pub constraints: Vec<CircuitConstraint>,
}
```

#### `CircuitConstraint`

```rust
pub enum CircuitConstraint {
    Mul { a: usize, b: usize, c: usize },   // wire_a × wire_b = wire_c
    Add { a: usize, b: usize, c: usize },   // wire_a + wire_b = wire_c
    Eq  { a: usize, b: usize },              // wire_a = wire_b
    Const { a: usize, val: u64 },            // wire_a = val
    Bool { a: usize },                       // wire_a ∈ {0, 1}
}
```

#### `ScoringWFA<S>`

```rust
pub struct ScoringWFA<S: Semiring> {
    pub wfa: WeightedFiniteAutomaton<S>,
    pub metric_name: String,
}
```

### Scoring Tokenizer

**File:** `scoring/tokenizer.rs`

#### `Token`

```rust
pub struct Token {
    pub id: usize,
    pub value: String,
    pub span: (usize, usize),  // (start, end) byte offsets
}
```

#### `NormalizationConfig`

```rust
pub struct NormalizationConfig {
    pub case_sensitive: bool,
    pub remove_punctuation: bool,
    pub lowercase: bool,
}
```

#### `Tokenizer` Trait

```rust
pub trait Tokenizer {
    fn tokenize(&self, text: &str) -> Vec<Token>;
    fn token_type(&self) -> &str;
}
```

#### Implementations

| Type | Description |
|------|-------------|
| `WhitespaceTokenizer` | Split on whitespace boundaries |
| `WordPieceTokenizer` | BPE-approximate subword tokenization |
| `CharacterTokenizer` | Character-level tokenization |
| `NGramTokenizer` | N-gram-based tokenization |

#### Helper Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `batch_tokenize` | `fn batch_tokenize(tokenizer: &dyn Tokenizer, texts: &[&str]) -> Vec<Vec<Token>>` | Tokenize multiple texts |
| `tokens_to_ids` | `fn tokens_to_ids(tokens: &[Token]) -> Vec<usize>` | Extract token IDs |
| `token_overlap` | `fn token_overlap(a: &[Token], b: &[Token]) -> usize` | Count overlapping tokens |
| `count_token_ngrams` | `fn count_token_ngrams(tokens: &[Token], n: usize) -> HashMap<Vec<String>, usize>` | Count token n-grams |

**Example:**

```rust
use spectacles_core::scoring::tokenizer::*;

let tokenizer = WhitespaceTokenizer::new();
let tokens = tokenizer.tokenize("hello world foo bar");
assert_eq!(tokens.len(), 4);
assert_eq!(tokens[0].value, "hello");

// N-gram counting
let ngrams = count_token_ngrams(&tokens, 2);
println!("Bigrams: {:?}", ngrams);
```

### Scoring Exact Match

**File:** `scoring/exact_match.rs`

#### `ExactMatchScorer`

```rust
pub struct ExactMatchScorer;
```

Implements `TripleMetric<Input = ScoringPair, Score = FixedPointScore>`.

- **Reference**: Direct `candidate == reference` comparison
- **WFA**: Single-path DFA over `BooleanSemiring`
- **Circuit**: Boolean equality check circuit

#### `NormalizedExactMatchScorer`

```rust
pub struct NormalizedExactMatchScorer {
    pub config: NormalizationConfig,
}
```

Same as `ExactMatchScorer` but with text normalization.

#### Helper Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `exact_match_accuracy` | `fn exact_match_accuracy(pairs: &[ScoringPair]) -> f64` | Batch accuracy |
| `build_multi_string_wfa` | `fn build_multi_string_wfa(strings: &[&str]) -> WeightedFiniteAutomaton<BooleanSemiring>` | Multi-reference WFA |

**Example:**

```rust
use spectacles_core::scoring::exact_match::*;
use spectacles_core::scoring::{ScoringPair, TripleMetric};

let scorer = ExactMatchScorer;
let pair = ScoringPair {
    candidate: "hello world".into(),
    reference: "hello world".into(),
};

let score = scorer.score_reference(&pair);
assert_eq!(score.numerator, 1); // exact match
assert_eq!(score.denominator, 1);

// Differential testing verifies all 3 agree
let verified = scorer.score_and_verify(&pair);
```

### Scoring Token F1

**File:** `scoring/token_f1.rs`

#### `TokenF1Scorer`

```rust
pub struct TokenF1Scorer;
```

Implements `TripleMetric<Input = ScoringPair, Score = FixedPointScore>`.

Computes token-level precision, recall, and F1 using bag-of-words
intersection over `CountingSemiring`.

#### `MacroF1Scorer`

Macro-averaged F1 across categories.

#### `MicroF1Scorer`

Micro-averaged F1 (global TP/FP/FN aggregation).

**Example:**

```rust
use spectacles_core::scoring::token_f1::*;
use spectacles_core::scoring::{ScoringPair, TripleMetric};

let scorer = TokenF1Scorer;
let pair = ScoringPair {
    candidate: "the cat sat on the mat".into(),
    reference: "the cat is on a mat".into(),
};

let ref_score = scorer.score_reference(&pair);
let wfa_score = scorer.score_automaton(&pair);
assert_eq!(ref_score, wfa_score);
```

### Scoring BLEU

**File:** `scoring/bleu.rs`

#### `BleuConfig`

```rust
pub struct BleuConfig {
    pub max_n: usize,            // Maximum n-gram order (default: 4)
    pub smoothing_method: u8,    // Smoothing method (0=none, 1=add-1, etc.)
    pub lowercase: bool,         // Lowercase before scoring
}
```

#### `BleuScorer`

```rust
pub struct BleuScorer {
    pub config: BleuConfig,
}
```

Implements `TripleMetric<Input = ScoringPair, Score = FixedPointScore>`.

- **Reference**: Standard BLEU with modified precision and brevity penalty
- **WFA**: Product of 4 n-gram counting WFAs over `CountingSemiring`
- **Circuit**: Counting circuit with division gadget for precision ratios

#### `NgramPrecision`

```rust
pub struct NgramPrecision {
    pub n: usize,
    pub matches: u64,
    pub total: u64,
    pub precision: f64,
}
```

#### `BleuResult`

```rust
pub struct BleuResult {
    pub score: f64,
    pub brevity_penalty: f64,
    pub precisions: Vec<NgramPrecision>,
}
```

**Example:**

```rust
use spectacles_core::scoring::bleu::*;
use spectacles_core::scoring::{ScoringPair, TripleMetric};

let scorer = BleuScorer {
    config: BleuConfig {
        max_n: 4,
        smoothing_method: 1,
        lowercase: true,
    },
};

let pair = ScoringPair {
    candidate: "the cat sat on the mat".into(),
    reference: "the cat is on the mat".into(),
};

let score = scorer.score_reference(&pair);
println!("BLEU: {}/{}", score.numerator, score.denominator);

// All three implementations agree
let verified = scorer.score_and_verify(&pair);
```

### Scoring ROUGE

**File:** `scoring/rouge.rs`

#### `RougeConfig`

```rust
pub struct RougeConfig {
    pub n: usize,             // N-gram size for ROUGE-N
    pub use_stemming: bool,   // Apply stemming before comparison
    pub lowercase: bool,      // Lowercase before scoring
}
```

#### `RougeNScorer`

```rust
pub struct RougeNScorer {
    pub config: RougeConfig,
}
```

Implements `TripleMetric<Input = ScoringPair, Score = FixedPointScore>`.

N-gram recall metric over `CountingSemiring`.

#### `RougeLScorer`

```rust
pub struct RougeLScorer {
    pub config: RougeConfig,
}
```

Implements `TripleMetric<Input = ScoringPair, Score = FixedPointScore>`.

LCS-based F-measure over `MaxPlusSemiring`. This is a Tier 2 metric
requiring comparison gadgets.

#### Helper Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `simple_stem` | `fn simple_stem(word: &str) -> String` | Basic Porter-style stemming |

**Example:**

```rust
use spectacles_core::scoring::rouge::*;
use spectacles_core::scoring::{ScoringPair, TripleMetric};

// ROUGE-1
let scorer = RougeNScorer {
    config: RougeConfig { n: 1, use_stemming: false, lowercase: true },
};

let pair = ScoringPair {
    candidate: "the cat sat on the mat".into(),
    reference: "the cat is on the mat".into(),
};

let score = scorer.score_reference(&pair);
println!("ROUGE-1: {}/{}", score.numerator, score.denominator);

// ROUGE-L
let scorer_l = RougeLScorer {
    config: RougeConfig { n: 0, use_stemming: false, lowercase: true },
};
let score_l = scorer_l.score_and_verify(&pair);
```

### Scoring Regex Match

**File:** `scoring/regex_match.rs`

#### `RegexMatchScorer`

```rust
pub struct RegexMatchScorer {
    pub pattern: String,
}
```

Implements `TripleMetric<Input = ScoringPair, Score = FixedPointScore>`.

Pattern matching via compiled automata over `BooleanSemiring`.

#### `RegexCompiler`

Compilation pipeline: `RegexAst` → `Nfa` → `Dfa`.

```rust
pub struct RegexCompiler;
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `compile` | `fn compile(pattern: &str) -> Result<Dfa, RegexError>` | Compile regex to DFA |
| `to_wfa` | `fn to_wfa(dfa: &Dfa, alphabet: &Alphabet) -> WeightedFiniteAutomaton<BooleanSemiring>` | Convert DFA to WFA |

#### Related Types

- **`RegexAst`** — parsed regex syntax tree
- **`Nfa`** — nondeterministic finite automaton
- **`Dfa`** — deterministic finite automaton

#### Helper Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `regex_union` | `fn regex_union(a: &str, b: &str) -> String` | Regex alternation |
| `regex_concat` | `fn regex_concat(a: &str, b: &str) -> String` | Regex concatenation |

**Example:**

```rust
use spectacles_core::scoring::regex_match::*;
use spectacles_core::scoring::{ScoringPair, TripleMetric};

let scorer = RegexMatchScorer {
    pattern: r"^\d{4}-\d{2}-\d{2}$".into(),
};

let pair = ScoringPair {
    candidate: "2024-01-15".into(),
    reference: "".into(), // reference unused for regex match
};

let score = scorer.score_reference(&pair);
assert_eq!(score.numerator, 1); // matches date pattern
```

### Scoring Pass@k

**File:** `scoring/pass_at_k.rs`

#### `PassAtKConfig`

```rust
pub struct PassAtKConfig {
    pub k: usize,              // Number of samples
    pub total_trials: usize,   // Total number of trials
}
```

#### `PassAtKScorer`

```rust
pub struct PassAtKScorer {
    pub config: PassAtKConfig,
}
```

Implements `TripleMetric`. Unbiased estimator for code generation
correctness: probability that at least one of k samples passes.

Uses `CountingSemiring` with division gadgets (Tier 2).

#### `PassAtKResult`

```rust
pub struct PassAtKResult {
    pub pass_at_k: f64,
    pub total_correct: usize,
    pub total_trials: usize,
}
```

#### Helper Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `binomial` | `fn binomial(n: u64, k: u64) -> u64` | Binomial coefficient C(n,k) |
| `corpus_pass_at_k` | `fn corpus_pass_at_k(results: &[PassAtKResult]) -> f64` | Corpus-level pass@k |

**Example:**

```rust
use spectacles_core::scoring::pass_at_k::*;

let scorer = PassAtKScorer {
    config: PassAtKConfig { k: 10, total_trials: 100 },
};

// binomial helper
assert_eq!(binomial(10, 3), 120);
```

### Scoring Differential

**File:** `scoring/differential.rs`

#### `DifferentialTester`

Cross-validates all three implementations of a metric.

```rust
pub struct DifferentialTester<M: TripleMetric> {
    metric: M,
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(metric: M) -> Self` | Create tester for a metric |
| `test` | `fn test(&self, input: &M::Input) -> DifferentialResult<M::Score>` | Run all 3 and compare |
| `test_batch` | `fn test_batch(&self, inputs: &[M::Input]) -> AgreementReport` | Batch test |

#### `DifferentialResult<T>`

```rust
pub struct DifferentialResult<T> {
    pub reference_score: T,
    pub automaton_score: T,
    pub circuit_score: T,
    pub all_agree: bool,
}
```

#### `AgreementReport`

```rust
pub struct AgreementReport {
    pub total_tests: usize,
    pub agreements: usize,
    pub disagreements: usize,
    pub agreement_rate: f64,
    pub failing_indices: Vec<usize>,
}
```

#### Helper Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `standard_test_suite` | `fn standard_test_suite() -> Vec<ScoringPair>` | Pre-defined test pairs |
| `random_test_pairs` | `fn random_test_pairs(count: usize, seed: u64) -> Vec<ScoringPair>` | Random test generation |

**Example — Differential testing:**

```rust
use spectacles_core::scoring::differential::*;
use spectacles_core::scoring::bleu::BleuScorer;

let scorer = BleuScorer::default();
let tester = DifferentialTester::new(scorer);

// Test a single input
let pair = ScoringPair {
    candidate: "hello world".into(),
    reference: "hello there world".into(),
};
let result = tester.test(&pair);
assert!(result.all_agree, "Implementations disagree!");

// Batch test with standard suite
let report = tester.test_batch(&standard_test_suite());
println!(
    "Agreement: {}/{} ({:.1}%)",
    report.agreements, report.total_tests,
    report.agreement_rate * 100.0
);
```

---

## Utils Module

**Path:** `spectacles-core/src/utils/`

The utils module provides hashing, serialization, and mathematical utilities.

### Utils Hash

**File:** `utils/hash.rs`

#### `SpectaclesHasher`

```rust
pub struct SpectaclesHasher {
    // BLAKE3 hasher state
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new() -> Self` | Create hasher |
| `hash` | `fn hash(&self, data: &[u8]) -> [u8; 32]` | Hash data |
| `hash_pair` | `fn hash_pair(&self, a: &[u8], b: &[u8]) -> [u8; 32]` | Hash concatenation |

#### `DomainSeparatedHasher`

```rust
pub struct DomainSeparatedHasher {
    domain: String,
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(domain: &str) -> Self` | Create with domain tag |
| `hash` | `fn hash(&self, data: &[u8]) -> [u8; 32]` | Domain-separated hash |

#### `MerkleTree`

```rust
pub struct MerkleTree {
    // tree nodes
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(leaves: Vec<[u8; 32]>) -> Self` | Build from leaf hashes |
| `root` | `fn root(&self) -> [u8; 32]` | Root hash |
| `prove` | `fn prove(&self, index: usize) -> MerkleProof` | Generate inclusion proof |
| `verify` | `fn verify(root: &[u8; 32], proof: &MerkleProof, leaf: &[u8; 32]) -> bool` | Verify proof |

#### `MerkleProof`

```rust
pub struct MerkleProof {
    pub leaf_index: usize,
    pub steps: Vec<MerkleProofStep>,
}
```

#### `MerkleProofStep`

```rust
pub struct MerkleProofStep {
    pub hash: [u8; 32],
    pub is_left: bool,
}
```

#### `HashChain`

Sequential commitment chain for Fiat-Shamir.

```rust
pub struct HashChain {
    // chain state
}
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `new` | `fn new(domain: &str) -> Self` | Create chain |
| `append` | `fn append(&mut self, data: &[u8])` | Append to chain |
| `finalize` | `fn finalize(&self) -> [u8; 32]` | Finalize to hash |

#### `Commitment` / `CommitmentOpening`

```rust
pub struct Commitment {
    pub hash: [u8; 32],
}

pub struct CommitmentOpening {
    pub data: Vec<u8>,
    pub randomness: [u8; 32],
}
```

**Example:**

```rust
use spectacles_core::utils::hash::*;

// Domain-separated hashing
let hasher = DomainSeparatedHasher::new("spectacles.commitment.v1");
let hash = hasher.hash(b"my data");

// Merkle tree
let leaves: Vec<[u8; 32]> = trace_columns.iter()
    .map(|col| hasher.hash(col))
    .collect();
let tree = MerkleTree::new(leaves);
let proof = tree.prove(3);
assert!(MerkleTree::verify(&tree.root(), &proof, &leaves[3]));
```

### Utils Serialization

**File:** `utils/serialization.rs`

#### `ProofFormat`

```rust
pub enum ProofFormat {
    Json,
    Bincode,
    CompactBinary,
    HumanReadable,
}
```

#### `FormatVersion`

```rust
pub struct FormatVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}
```

#### `ProofHeader`

```rust
pub struct ProofHeader {
    pub version: FormatVersion,
    pub format: ProofFormat,
    pub metric_name: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}
```

#### `CompactProof`

```rust
pub struct CompactProof {
    pub header: ProofHeader,
    pub compressed_data: Vec<u8>,
}
```

#### `ProofSerializer`

```rust
pub struct ProofSerializer;
```

**Methods:**

| Method | Signature | Description |
|--------|-----------|-------------|
| `serialize` | `fn serialize(proof: &STARKProof, format: ProofFormat) -> Vec<u8>` | Serialize a proof |
| `deserialize` | `fn deserialize(data: &[u8]) -> Result<STARKProof, SerializationError>` | Deserialize a proof |
| `to_compact` | `fn to_compact(proof: &STARKProof) -> CompactProof` | Create compact representation |
| `from_compact` | `fn from_compact(compact: &CompactProof) -> Result<STARKProof, SerializationError>` | Restore from compact |

#### Helper Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `estimate_proof_size` | `fn estimate_proof_size(constraints: usize, wires: usize, security: usize) -> usize` | Estimate serialized proof size |
| `compress_rle` | `fn compress_rle(data: &[u8]) -> Vec<u8>` | Run-length encoding compression |
| `decompress_rle` | `fn decompress_rle(data: &[u8]) -> Vec<u8>` | Run-length encoding decompression |

**Example:**

```rust
use spectacles_core::utils::serialization::*;

// Serialize proof to JSON
let json_bytes = ProofSerializer::serialize(&proof, ProofFormat::Json);

// Deserialize
let restored = ProofSerializer::deserialize(&json_bytes)?;

// Estimate proof size
let estimated = estimate_proof_size(1024, 2048, 128);
println!("Estimated proof size: {} bytes", estimated);

// Compact representation
let compact = ProofSerializer::to_compact(&proof);
println!("Compact size: {} bytes", compact.compressed_data.len());
```

### Utils Math

**File:** `utils/math.rs`

Number theory and polynomial utilities for field arithmetic.

#### Modular Arithmetic

| Function | Signature | Description |
|----------|-----------|-------------|
| `extended_gcd` | `fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64)` | Extended GCD: returns (gcd, x, y) where ax + by = gcd |
| `mod_pow` | `fn mod_pow(base: u64, exp: u64, modulus: u64) -> u64` | Modular exponentiation |
| `mod_inv` | `fn mod_inv(a: u64, modulus: u64) -> Option<u64>` | Modular inverse |
| `mod_add` | `fn mod_add(a: u64, b: u64, modulus: u64) -> u64` | Modular addition |
| `mod_sub` | `fn mod_sub(a: u64, b: u64, modulus: u64) -> u64` | Modular subtraction |
| `mod_mul` | `fn mod_mul(a: u64, b: u64, modulus: u64) -> u64` | Modular multiplication |

#### Polynomial Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `polynomial_eval` | `fn polynomial_eval(coeffs: &[f64], x: f64) -> f64` | Evaluate polynomial (Horner's method) |
| `polynomial_eval_mod` | `fn polynomial_eval_mod(coeffs: &[u64], x: u64, modulus: u64) -> u64` | Evaluate polynomial mod p |
| `polynomial_add` | `fn polynomial_add(a: &[u64], b: &[u64]) -> Vec<u64>` | Add polynomials |
| `polynomial_mul` | `fn polynomial_mul(a: &[u64], b: &[u64]) -> Vec<u64>` | Multiply polynomials |
| `polynomial_mul_mod` | `fn polynomial_mul_mod(a: &[u64], b: &[u64], modulus: u64) -> Vec<u64>` | Multiply polynomials mod p |

#### Interpolation

| Function | Signature | Description |
|----------|-----------|-------------|
| `lagrange_interpolate` | `fn lagrange_interpolate(points: &[(f64, f64)]) -> Vec<f64>` | Lagrange interpolation → coefficients |
| `lagrange_interpolate_mod` | `fn lagrange_interpolate_mod(points: &[(u64, u64)], modulus: u64) -> Vec<u64>` | Lagrange interpolation mod p |

#### FFT / NTT

| Function | Signature | Description |
|----------|-----------|-------------|
| `fft` | `fn fft(coeffs: &[Complex<f64>]) -> Vec<Complex<f64>>` | Fast Fourier Transform |
| `ntt` | `fn ntt(coeffs: &[u64], modulus: u64, root: u64) -> Vec<u64>` | Number Theoretic Transform |
| `polynomial_mul_fft` | `fn polynomial_mul_fft(a: &[f64], b: &[f64]) -> Vec<f64>` | FFT-based polynomial multiplication |

#### Number Theory

| Function | Signature | Description |
|----------|-----------|-------------|
| `find_primitive_root` | `fn find_primitive_root(modulus: u64) -> Option<u64>` | Find primitive root of Z/pZ |
| `is_prime` | `fn is_prime(n: u64) -> bool` | Deterministic primality test |
| `is_probable_prime` | `fn is_probable_prime(n: u64, rounds: usize) -> bool` | Miller-Rabin primality test |
| `crt` | `fn crt(remainders: &[(u64, u64)]) -> Option<u64>` | Chinese Remainder Theorem |

**Example:**

```rust
use spectacles_core::utils::math::*;

// Modular arithmetic
let inv = mod_inv(42, GOLDILOCKS_MODULUS).unwrap();
assert_eq!(mod_mul(42, inv, GOLDILOCKS_MODULUS), 1);

// Polynomial evaluation
let coeffs = vec![1, 2, 3]; // 1 + 2x + 3x²
let result = polynomial_eval_mod(&coeffs, 5, GOLDILOCKS_MODULUS);

// Lagrange interpolation
let points = vec![(0, 1), (1, 4), (2, 9)];
let poly = lagrange_interpolate_mod(&points, GOLDILOCKS_MODULUS);

// NTT for fast polynomial multiplication
let root = find_primitive_root(GOLDILOCKS_MODULUS).unwrap();
let transformed = ntt(&coeffs, GOLDILOCKS_MODULUS, root);
```

---

## Common Workflows

### Workflow 1: Define → Compile → Prove → Verify

End-to-end pipeline from metric definition to verified proof.

```rust
use spectacles_core::evalspec::{parser::Parser, typechecker::TypeChecker, compiler::EvalSpecCompiler};
use spectacles_core::wfa::field_embedding::embed_wfa;
use spectacles_core::circuit::{compiler::WFACircuitCompiler, stark::*};
use spectacles_core::protocol::certificate::CertificateBuilder;
use spectacles_core::scoring::{ScoringPair, FixedPointScore};

// Step 1: Parse and type-check the metric
let source = r#"metric bleu(c: TokenSequence, r: TokenSequence) -> Float {
    ngram_precision(c, r, n=4)
}"#;
let mut parser = Parser::new(source, "bleu.eval");
let ast = parser.parse()?;

let mut checker = TypeChecker::new().with_builtins();
let eval_type = checker.check(&ast)?;

// Step 2: Compile to WFA
let mut eval_compiler = EvalSpecCompiler::new();
let compiled = eval_compiler.compile(&ast)?;

// Step 3: Embed WFA into Goldilocks field
// (In practice, the compiled metric provides a WFA handle)
// let field_wfa = embed_wfa(&counting_wfa);

// Step 4: Compile to AIR circuit
let mut circuit_compiler = WFACircuitCompiler::new();
// let air_program = circuit_compiler.compile(&field_wfa)?;

// Step 5: Generate execution trace (from actual scoring)
// let trace = generate_trace(&air_program, &input);

// Step 6: Prove
let prover = STARKProver::new(128);
// let proof = prover.prove(&air_program, &trace)?;

// Step 7: Verify
let verifier = STARKVerifier::new(128);
// let valid = verifier.verify(&air_program, &proof)?;
// assert!(valid);

// Step 8: Issue certificate
// let cert = CertificateBuilder::new()
//     .metric("bleu")
//     .score(score)
//     .proof_hash(hash_of_proof)
//     .input_commitment(input_commitment)
//     .build()?;
```

### Workflow 2: Check Metric Equivalence

Verify that two BLEU variants compute the same function.

```rust
use spectacles_core::scoring::bleu::{BleuScorer, BleuConfig};
use spectacles_core::wfa::equivalence::{are_equivalent, find_distinguishing_word};

// Create two BLEU variants
let bleu_standard = BleuScorer {
    config: BleuConfig { max_n: 4, smoothing_method: 0, lowercase: true },
};
let bleu_smoothed = BleuScorer {
    config: BleuConfig { max_n: 4, smoothing_method: 1, lowercase: true },
};

// Get their WFA representations (internal to scoring)
// let wfa1 = bleu_standard.build_wfa();
// let wfa2 = bleu_smoothed.build_wfa();

// Check equivalence
// let equiv = are_equivalent(&wfa1, &wfa2);
// if !equiv {
//     let word = find_distinguishing_word(&wfa1, &wfa2);
//     println!("Variants differ on: {:?}", word);
// }

// Or use differential testing as a practical alternative
use spectacles_core::scoring::differential::*;

let tester = DifferentialTester::new(bleu_standard);
let report = tester.test_batch(&standard_test_suite());
println!("Agreement rate: {:.1}%", report.agreement_rate * 100.0);
```

### Workflow 3: PSI Contamination Detection

Run contamination detection between evaluation text and training data.

```rust
use spectacles_core::psi::*;
use spectacles_core::psi::ngram::*;
use spectacles_core::psi::trie::NGramTrie;
use spectacles_core::psi::protocol::{PSIProtocol, PSIConfig, PSIMode};

// Step 1: Configure n-gram extraction
let ngram_config = NGramConfig {
    n: 5,
    gram_type: GramType::Token,
    min_frequency: 1,
    normalized: true,
};

// Step 2: Build training data trie
let extractor = NGramExtractor::new(ngram_config.clone());
let mut train_trie = NGramTrie::new();
let training_texts = vec![
    "the quick brown fox jumps over the lazy dog",
    "a stitch in time saves nine",
];
for text in &training_texts {
    for ngram in extractor.extract(text) {
        train_trie.insert(&ngram.tokens);
    }
}

// Step 3: Run PSI protocol
let psi_config = PSIConfig {
    mode: PSIMode::Batch,
    ngram_config,
    security_bits: 128,
    threshold: Some(0.05),
};

let mut protocol = PSIProtocol::new(psi_config);
protocol.setup().expect("PSI setup failed");

let eval_text = "the quick brown fox runs through the field";
let result = protocol.run(eval_text, &train_trie)
    .expect("PSI protocol failed");

println!("Overlap: {:.2}% ({}/{})",
    result.overlap_ratio * 100.0,
    result.intersection_size,
    result.evaluator_set_size);

// Step 4: Generate attestation
let attestation = protocol.generate_attestation(&result);
println!("Contamination attestation: {:.2}% overlap",
    attestation.overlap_percentage);
```

### Workflow 4: Differential Testing Across Three Implementations

Validate that reference, WFA, and circuit implementations agree.

```rust
use spectacles_core::scoring::*;
use spectacles_core::scoring::differential::*;
use spectacles_core::scoring::bleu::BleuScorer;
use spectacles_core::scoring::rouge::{RougeNScorer, RougeConfig};
use spectacles_core::scoring::token_f1::TokenF1Scorer;
use spectacles_core::scoring::exact_match::ExactMatchScorer;

// Test all metrics against standard suite
let test_pairs = standard_test_suite();

// BLEU
let bleu_tester = DifferentialTester::new(BleuScorer::default());
let bleu_report = bleu_tester.test_batch(&test_pairs);
println!("BLEU agreement: {:.1}%", bleu_report.agreement_rate * 100.0);

// ROUGE-1
let rouge_tester = DifferentialTester::new(RougeNScorer {
    config: RougeConfig { n: 1, use_stemming: false, lowercase: true },
});
let rouge_report = rouge_tester.test_batch(&test_pairs);
println!("ROUGE-1 agreement: {:.1}%", rouge_report.agreement_rate * 100.0);

// Token F1
let f1_tester = DifferentialTester::new(TokenF1Scorer);
let f1_report = f1_tester.test_batch(&test_pairs);
println!("Token F1 agreement: {:.1}%", f1_report.agreement_rate * 100.0);

// Exact Match
let em_tester = DifferentialTester::new(ExactMatchScorer);
let em_report = em_tester.test_batch(&test_pairs);
println!("Exact Match agreement: {:.1}%", em_report.agreement_rate * 100.0);

// Test with random pairs
let random_pairs = random_test_pairs(1000, 42);
let random_report = bleu_tester.test_batch(&random_pairs);
println!(
    "BLEU random test: {}/{} agree",
    random_report.agreements, random_report.total_tests
);

// Investigate disagreements
if !random_report.failing_indices.is_empty() {
    for idx in &random_report.failing_indices {
        let result = bleu_tester.test(&random_pairs[*idx]);
        println!("Disagreement at index {}:", idx);
        println!("  Reference: {:?}", result.reference_score);
        println!("  Automaton: {:?}", result.automaton_score);
        println!("  Circuit:   {:?}", result.circuit_score);
    }
}
```

---

## Error Types Summary

| Module | Error Type | Key Variants |
|--------|-----------|--------------|
| `wfa` | `WfaError` | `InvalidState`, `DimensionMismatch`, `EmptyAutomaton`, `NonDeterministic`, `InvalidSymbol`, `SerializationError` |
| `evalspec` | `ParseError` | `message`, `span` |
| `evalspec` | `TypeError` | `message`, `span`, `expected`, `found` |
| `circuit` | `CompileError` | Circuit compilation failures |
| `circuit` | `ProverError` | Proof generation failures |
| `circuit` | `VerifierError` | Proof verification failures |
| `protocol` | `ProtocolError` | `InvalidTransition`, `Timeout`, `AlreadyTerminal`, `CommitmentMismatch`, `SerializationError`, `InvalidState` |
| `protocol` | `CertificateError` | `MissingField`, `InvalidProofHash`, `ExpiredCertificate`, `ChainVerificationFailed` |
| `protocol` | `TranscriptError` | `InvalidLabel`, `EmptyTranscript`, `VerificationFailed` |
| `psi` | `PSIError`/`PSIErrorKind` | `SetupFailed`, `HashingFailed`, `MatchingFailed`, `VerificationFailed`, `InvalidConfig` |

---

## Constants

| Constant | Value | Location | Description |
|----------|-------|----------|-------------|
| `GOLDILOCKS_MODULUS` | `0xFFFFFFFF00000001` | `scoring/mod.rs`, `circuit/goldilocks.rs` | Goldilocks prime field modulus (2⁶⁴ − 2³² + 1) |
