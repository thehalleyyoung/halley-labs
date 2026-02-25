# EvalSpec: Formal Grammar, Typing Rules, and Denotational Semantics

> **Status**: Reference specification for the EvalSpec DSL.
> Addresses the requirement for a formal grammar, typing rules, and denotational semantics.

---

## 1. Formal Grammar (BNF)

The grammar below is derived from the recursive-descent parser in
`spectacles-core/src/evalspec/parser.rs` and the AST definitions in
`spectacles-core/src/evalspec/types.rs`.

### 1.1 Programs and Declarations

```
⟨program⟩          ::= ⟨declaration⟩*

⟨declaration⟩       ::= ⟨metric-decl⟩
                       | ⟨let-decl⟩
                       | ⟨type-decl⟩
                       | ⟨import-decl⟩
                       | ⟨test-decl⟩

⟨metric-decl⟩       ::= ⟨attributes⟩ 'metric' ⟨ident⟩ '(' ⟨params⟩ ')' ':' ⟨type⟩ '{' ⟨expr⟩ '}'

⟨let-decl⟩          ::= ⟨attributes⟩ 'let' ⟨ident⟩ ( ':' ⟨type⟩ )? '=' ⟨expr⟩ ';'

⟨type-decl⟩         ::= ⟨attributes⟩ 'type' ⟨ident⟩ '=' ⟨type⟩ ';'

⟨import-decl⟩       ::= 'import' ⟨import-items⟩ 'from' ⟨import-path⟩ ';'
                       | 'import' ⟨import-path⟩ ( 'as' ⟨ident⟩ )? ';'

⟨import-path⟩       ::= ⟨ident⟩ ( '::' ⟨ident⟩ )*

⟨import-items⟩      ::= '*'
                       | '{' ⟨import-item⟩ ( ',' ⟨import-item⟩ )* '}'

⟨import-item⟩       ::= ⟨ident⟩ ( 'as' ⟨ident⟩ )?

⟨test-decl⟩         ::= ⟨attributes⟩ 'test' ⟨string-lit⟩ '{' ⟨expr⟩ 'expect' ⟨expectation⟩ '}'

⟨expectation⟩       ::= ⟨literal⟩
                       | '≈' ⟨float-lit⟩ '±' ⟨float-lit⟩
                       | 'success'
                       | 'error' ( '(' ⟨string-lit⟩ ')' )?
```

### 1.2 Attributes

```
⟨attributes⟩       ::= ( '#' '[' ⟨attribute⟩ ']' )*

⟨attribute⟩         ::= 'doc' '(' ⟨string-lit⟩ ')'
                       | 'deprecated' '(' ⟨string-lit⟩ ')'
                       | 'test'
                       | 'inline'
                       | 'semiring' '(' ⟨semiring-type⟩ ')'
```

### 1.3 Parameters and Types

```
⟨params⟩            ::= ε | ⟨param⟩ ( ',' ⟨param⟩ )*

⟨param⟩             ::= ⟨ident⟩ ':' ⟨type⟩

⟨type⟩              ::= ⟨base-type⟩
                       | ⟨semiring-type⟩
                       | ⟨function-type⟩
                       | ⟨metric-type⟩
                       | ⟨annotated-type⟩
                       | ⟨type-var⟩
                       | '(' ')'

⟨base-type⟩         ::= 'String'
                       | 'Int'
                       | 'Float'
                       | 'Bool'
                       | 'Token'
                       | 'TokenSequence'
                       | '[' ⟨base-type⟩ ']'
                       | '(' ⟨base-type⟩ ( ',' ⟨base-type⟩ )+ ')'
                       | 'NGram' '(' ⟨int-lit⟩ ')'

⟨semiring-type⟩     ::= 'Counting'
                       | 'Boolean'
                       | 'Tropical'
                       | 'BoundedCounting' '(' ⟨int-lit⟩ ')'
                       | 'Real'
                       | 'LogDomain'
                       | 'Viterbi'
                       | 'Goldilocks'

⟨function-type⟩     ::= '(' ⟨type⟩ ( ',' ⟨type⟩ )* ')' '->' ⟨type⟩
                       | ⟨type⟩ '->' ⟨type⟩

⟨metric-type⟩       ::= 'Metric' '<' ⟨type⟩ ',' ⟨type⟩ '>'

⟨annotated-type⟩    ::= ⟨base-type⟩ '@' ⟨semiring-type⟩

⟨type-var⟩          ::= '\'' ⟨ident⟩
```

### 1.4 Expressions

The parser uses Pratt parsing with binding powers (BP) for precedence.
Productions below are ordered from lowest to highest precedence.

```
⟨expr⟩              ::= ⟨let-expr⟩
                       | ⟨if-expr⟩
                       | ⟨match-expr⟩
                       | ⟨lambda-expr⟩
                       | ⟨or-expr⟩

⟨let-expr⟩          ::= 'let' ⟨ident⟩ ( ':' ⟨type⟩ )? '=' ⟨expr⟩ 'in' ⟨expr⟩

⟨if-expr⟩           ::= 'if' ⟨expr⟩ 'then' ⟨expr⟩ 'else' ⟨expr⟩

⟨match-expr⟩        ::= 'match' ⟨expr⟩ 'with' ⟨match-arm⟩+

⟨match-arm⟩         ::= '|' ⟨pattern⟩ '=>' ⟨expr⟩

⟨lambda-expr⟩       ::= 'fn' '(' ⟨params⟩ ')' '=>' ⟨expr⟩
                       | 'fn' '(' ⟨params⟩ ')' ':' ⟨type⟩ '{' ⟨expr⟩ '}'

⟨or-expr⟩           ::= ⟨and-expr⟩ ( 'or' ⟨and-expr⟩ )*            /* BP 1 */

⟨and-expr⟩          ::= ⟨cmp-expr⟩ ( 'and' ⟨cmp-expr⟩ )*           /* BP 2 */

⟨cmp-expr⟩          ::= ⟨add-expr⟩ ( ⟨cmp-op⟩ ⟨add-expr⟩ )*       /* BP 3 */
⟨cmp-op⟩            ::= '==' | '!=' | '<' | '<=' | '>' | '>='

⟨add-expr⟩          ::= ⟨mul-expr⟩ ( ⟨add-op⟩ ⟨mul-expr⟩ )*       /* BP 4 */
⟨add-op⟩            ::= '+' | '-' | '⊕'

⟨mul-expr⟩          ::= ⟨unary-expr⟩ ( ⟨mul-op⟩ ⟨unary-expr⟩ )*   /* BP 5 */
⟨mul-op⟩            ::= '*' | '/' | '%' | '⊗'

⟨unary-expr⟩        ::= ⟨prefix-op⟩ ⟨unary-expr⟩                   /* BP 6 */
                       | ⟨power-expr⟩
⟨prefix-op⟩         ::= '-' | 'not' | '*'

⟨power-expr⟩        ::= ⟨postfix-expr⟩ ( '^' ⟨unary-expr⟩ )?      /* BP 7, right-assoc */

⟨postfix-expr⟩      ::= ⟨primary⟩ ⟨postfix⟩*
⟨postfix⟩           ::= '(' ⟨args⟩ ')'                              /* function call */
                       | '.' ⟨ident⟩                                  /* field access */
                       | '[' ⟨expr⟩ ']'                               /* index access */
                       | '.' ⟨ident⟩ '(' ⟨args⟩ ')'                  /* method call */
```

### 1.5 Primary Expressions

```
⟨primary⟩           ::= ⟨literal⟩
                       | ⟨ident⟩
                       | '(' ⟨expr⟩ ( ',' ⟨expr⟩ )* ')'             /* tuple or paren */
                       | '[' ( ⟨expr⟩ ( ',' ⟨expr⟩ )* )? ']'        /* list literal */
                       | '{' ⟨expr⟩ ( ';' ⟨expr⟩ )* '}'             /* block */
                       | ⟨aggregate-expr⟩
                       | ⟨ngram-expr⟩
                       | ⟨tokenize-expr⟩
                       | ⟨clip-expr⟩
                       | ⟨compose-expr⟩
                       | ⟨semiring-cast⟩
                       | ⟨match-pattern-expr⟩

⟨literal⟩           ::= ⟨int-lit⟩ | ⟨float-lit⟩ | ⟨string-lit⟩ | 'true' | 'false'
```

### 1.6 Domain-Specific Expressions

```
⟨aggregate-expr⟩    ::= 'aggregate' '(' ⟨agg-op⟩ ',' ⟨expr⟩ ','
                           ⟨ident⟩ '=>' ⟨expr⟩
                           ( ',' 'semiring' ':' ⟨semiring-type⟩ )? ')'
⟨agg-op⟩            ::= 'sum' | 'product' | 'min' | 'max' | 'count' | 'mean'

⟨ngram-expr⟩        ::= 'ngram' '(' ⟨int-lit⟩ ',' ⟨expr⟩ ')'

⟨tokenize-expr⟩     ::= 'tokenize' '(' ⟨expr⟩ ( ',' ⟨ident⟩ )? ')'

⟨clip-expr⟩         ::= 'clip' '(' ⟨expr⟩ ',' ⟨expr⟩ ')'

⟨compose-expr⟩      ::= 'compose' '(' ⟨expr⟩ ',' ⟨expr⟩ ')'

⟨semiring-cast⟩     ::= ⟨expr⟩ 'as' ⟨semiring-type⟩

⟨match-pattern-expr⟩ ::= 'match' ⟨expr⟩ 'with' ⟨match-mode⟩ ⟨expr⟩
⟨match-mode⟩        ::= 'regex' | 'glob' | 'exact' | 'contains'
```

### 1.7 Patterns

```
⟨pattern⟩           ::= '_'                                          /* wildcard */
                       | ⟨ident⟩                                      /* variable */
                       | ⟨literal⟩                                    /* literal */
                       | ⟨ident⟩ '(' ⟨pattern⟩ ( ',' ⟨pattern⟩ )* ')' /* constructor */
                       | '(' ⟨pattern⟩ ( ',' ⟨pattern⟩ )* ')'        /* tuple */
                       | '[' ⟨pattern⟩* ( '..' ⟨ident⟩ )? ']'        /* list + rest */
                       | ⟨pattern⟩ 'if' ⟨expr⟩                       /* guard */
```

### 1.8 Lexical Conventions

```
⟨ident⟩             ::= [a-zA-Z_][a-zA-Z0-9_]*

⟨int-lit⟩           ::= [0-9]+

⟨float-lit⟩         ::= [0-9]+ '.' [0-9]+ ( [eE] [+-]? [0-9]+ )?

⟨string-lit⟩        ::= '"' ( [^"\\] | '\\' . )* '"'

⟨comment⟩           ::= '//' [^\n]* '\n'
                       | '/*' .* '*/'
```

---

## 2. Typing Rules

We define a typing judgment **Γ ⊢ e : τ** where Γ is a typing context
(finite map from identifiers to types), e is an EvalSpec expression, and τ
is an `EvalType`.

### 2.1 Notation

| Symbol | Meaning |
|--------|---------|
| Γ | Typing context (variable → type map) |
| τ, σ | Types (elements of `EvalType`) |
| S | Semiring type |
| ⊕ | Semiring addition |
| ⊗ | Semiring multiplication |
| 0_S | Additive identity of semiring S |
| 1_S | Multiplicative identity of semiring S |
| ⟨S, ⊕, ⊗, 0_S, 1_S⟩ | Semiring structure |

Concrete semiring instances:

| Name | Carrier | ⊕ | ⊗ | 0 | 1 |
|------|---------|---|---|---|---|
| Counting | ℕ | + | × | 0 | 1 |
| Boolean | {0,1} | ∨ | ∧ | 0 | 1 |
| Tropical | ℝ ∪ {∞} | min | + | ∞ | 0 |
| BoundedCounting(k) | {0,…,k} | min(·+·, k) | × | 0 | 1 |
| Real | ℝ | + | × | 0.0 | 1.0 |
| LogDomain | ℝ ∪ {−∞} | log-sum-exp | + | −∞ | 0 |
| Viterbi | [0,1] | max | × | 0 | 1 |
| Goldilocks | 𝔽_p (p = 2⁶⁴ − 2³² + 1) | +_p | ×_p | 0 | 1 |

### 2.2 Core Typing Rules

**T-Var**
```
  x : τ ∈ Γ
  ──────────
  Γ ⊢ x : τ
```

**T-IntLit**
```
  n ∈ ℤ
  ───────────────
  Γ ⊢ n : Int
```

**T-FloatLit**
```
  r ∈ ℝ
  ──────────────────
  Γ ⊢ r : Float
```

**T-BoolLit**
```
  b ∈ {true, false}
  ──────────────────
  Γ ⊢ b : Bool
```

**T-StringLit**
```
  s is a string literal
  ──────────────────────
  Γ ⊢ s : String
```

**T-BinOp-Arith**
```
  Γ ⊢ e₁ : τ    Γ ⊢ e₂ : τ    τ ∈ {Int, Float}    op ∈ {+, -, *, /, %, ^}
  ─────────────────────────────────────────────────────────────────────────────
  Γ ⊢ e₁ op e₂ : τ
```

**T-BinOp-Semiring**
```
  Γ ⊢ e₁ : Semiring(S)    Γ ⊢ e₂ : Semiring(S)    op ∈ {⊕, ⊗}
  ────────────────────────────────────────────────────────────────
  Γ ⊢ e₁ op e₂ : Semiring(S)
```

**T-BinOp-Cmp**
```
  Γ ⊢ e₁ : τ    Γ ⊢ e₂ : τ    op ∈ {==, !=, <, <=, >, >=}
  ───────────────────────────────────────────────────────────
  Γ ⊢ e₁ op e₂ : Bool
```

**T-BinOp-Logic**
```
  Γ ⊢ e₁ : Bool    Γ ⊢ e₂ : Bool    op ∈ {and, or}
  ──────────────────────────────────────────────────
  Γ ⊢ e₁ op e₂ : Bool
```

**T-BinOp-Concat**
```
  Γ ⊢ e₁ : String    Γ ⊢ e₂ : String
  ─────────────────────────────────────
  Γ ⊢ e₁ ++ e₂ : String
```

**T-UnaryNeg**
```
  Γ ⊢ e : τ    τ ∈ {Int, Float}
  ──────────────────────────────
  Γ ⊢ -e : τ
```

**T-UnaryNot**
```
  Γ ⊢ e : Bool
  ─────────────
  Γ ⊢ not e : Bool
```

**T-UnaryStar** *(Kleene star / semiring closure)*
```
  Γ ⊢ e : Semiring(S)
  ────────────────────
  Γ ⊢ *e : Semiring(S)
```

### 2.3 Binding and Control Flow

**T-If**
```
  Γ ⊢ e₁ : Bool    Γ ⊢ e₂ : τ    Γ ⊢ e₃ : τ
  ──────────────────────────────────────────────
  Γ ⊢ if e₁ then e₂ else e₃ : τ
```

**T-Let**
```
  Γ ⊢ e₁ : σ    Γ, x : σ ⊢ e₂ : τ
  ──────────────────────────────────
  Γ ⊢ let x = e₁ in e₂ : τ
```

**T-Let-Annotated**
```
  Γ ⊢ e₁ : σ    σ ≤ σ'    Γ, x : σ' ⊢ e₂ : τ
  ──────────────────────────────────────────────
  Γ ⊢ let x : σ' = e₁ in e₂ : τ
```

**T-Lambda**
```
  Γ, x₁ : τ₁, …, xₙ : τₙ ⊢ e : σ
  ─────────────────────────────────────────────
  Γ ⊢ fn(x₁: τ₁, …, xₙ: τₙ) => e : (τ₁, …, τₙ) → σ
```

**T-Apply**
```
  Γ ⊢ f : (τ₁, …, τₙ) → σ    Γ ⊢ eᵢ : τᵢ  (for i = 1…n)
  ──────────────────────────────────────────────────────────
  Γ ⊢ f(e₁, …, eₙ) : σ
```

**T-Match**
```
  Γ ⊢ e : σ    ∀i. Γ ⊢ pᵢ : σ ⇒ Γᵢ    Γ, Γᵢ ⊢ eᵢ : τ
  ────────────────────────────────────────────────────────
  Γ ⊢ match e with | p₁ => e₁ | … | pₙ => eₙ : τ
```

**T-Block**
```
  Γ ⊢ e₁ : τ₁    …    Γ ⊢ eₙ : τₙ
  ──────────────────────────────────
  Γ ⊢ { e₁; …; eₙ } : τₙ
```

### 2.4 Compound Types

**T-List**
```
  Γ ⊢ eᵢ : τ  (for all i = 1…n)
  ──────────────────────────────
  Γ ⊢ [e₁, …, eₙ] : [τ]
```

**T-Tuple**
```
  Γ ⊢ eᵢ : τᵢ  (for i = 1…n, n ≥ 2)
  ────────────────────────────────────
  Γ ⊢ (e₁, …, eₙ) : (τ₁, …, τₙ)
```

**T-Field**
```
  Γ ⊢ e : (τ₁, …, τₙ)    1 ≤ i ≤ n
  ────────────────────────────────────
  Γ ⊢ e.i : τᵢ
```

**T-Index**
```
  Γ ⊢ e₁ : [τ]    Γ ⊢ e₂ : Int
  ──────────────────────────────
  Γ ⊢ e₁[e₂] : τ
```

### 2.5 Domain-Specific Typing Rules

**T-Aggregate**
```
  Γ ⊢ xs : [σ]    Γ, x : σ ⊢ body : Semiring(S)    op ∈ {sum, product, min, max, count}
  ────────────────────────────────────────────────────────────────────────────────────────
  Γ ⊢ aggregate(op, xs, x => body) : Semiring(S)
```

When `op = sum`:    aggregate computes ⊕_S over elements.
When `op = product`: aggregate computes ⊗_S over elements.
When `op = count`:   S is inferred as Counting.

**T-Aggregate-Count** *(special case: counting infers Counting semiring)*
```
  Γ ⊢ xs : [σ]    Γ, x : σ ⊢ body : Bool
  ─────────────────────────────────────────
  Γ ⊢ aggregate(count, xs, x => body) : Semiring(Counting)
```

**T-NGram**
```
  Γ ⊢ s : TokenSequence    n ∈ ℕ, n ≥ 1
  ────────────────────────────────────────
  Γ ⊢ ngram(n, s) : [NGram(n)]
```

Side-effect on semiring context: The presence of `ngram` in a metric body
sets the ambient semiring to **Counting**, since n-gram extraction produces
multisets counted over ℕ.

**T-Tokenize**
```
  Γ ⊢ s : String
  ──────────────────────────────────────
  Γ ⊢ tokenize(s) : TokenSequence
```

```
  Γ ⊢ s : String    t is a tokenizer identifier
  ────────────────────────────────────────────────
  Γ ⊢ tokenize(s, t) : TokenSequence
```

**T-Clip** *(produces BoundedCounting semiring)*
```
  Γ ⊢ e : Semiring(Counting)    Γ ⊢ k : Int    k > 0
  ─────────────────────────────────────────────────────
  Γ ⊢ clip(e, k) : Semiring(BoundedCounting(k))
```

**T-Compose** *(semiring must match)*
```
  Γ ⊢ f : Metric⟨α, Semiring(S)⟩    Γ ⊢ g : Metric⟨β, Semiring(S)⟩
  ────────────────────────────────────────────────────────────────────
  Γ ⊢ compose(f, g) : Metric⟨(α, β), Semiring(S)⟩
```

**T-SemiringCast**
```
  Γ ⊢ e : Semiring(S₁)    S₁ ↪ S₂  (coercion exists)
  ─────────────────────────────────────────────────────
  Γ ⊢ e as S₂ : Semiring(S₂)
```

Valid coercions (↪):
- Counting ↪ Real
- Counting ↪ BoundedCounting(k) (for any k)
- BoundedCounting(k) ↪ Counting
- Boolean ↪ Counting
- Real ↪ LogDomain
- Viterbi ↪ Real
- Any S ↪ Goldilocks (for ZK circuit encoding)

### 2.6 Metric Declaration Typing

**T-MetricDecl**
```
  Γ, x₁ : τ₁, …, xₙ : τₙ ⊢ body : σ
  ─────────────────────────────────────────────────────────────────
  Γ ⊢ metric m(x₁: τ₁, …, xₙ: τₙ) : σ { body } : Metric⟨(τ₁,…,τₙ), σ⟩
```

**Semiring Propagation Rule**:
If σ = Semiring(S) or σ contains Semiring(S) as a sub-component,
then the metric m is a *WFA-candidate* over semiring S.

---

## 3. Denotational Semantics

We define a semantic function **⟦·⟧** that maps well-typed EvalSpec terms
to their mathematical denotations. Metrics denote **formal power series**
over weighted finite automata (WFAs).

### 3.1 Semantic Domains

| EvalSpec Type | Semantic Domain |
|---|---|
| Int | ℤ |
| Float | ℝ |
| Bool | 𝔹 = {⊤, ⊥} |
| String | Σ* (strings over alphabet Σ) |
| Token | ℕ (opaque identifiers) |
| TokenSequence | ℕ* (sequences of token ids) |
| [τ] | List(⟦τ⟧) |
| (τ₁, …, τₙ) | ⟦τ₁⟧ × … × ⟦τₙ⟧ |
| τ₁ → τ₂ | ⟦τ₁⟧ → ⟦τ₂⟧ |
| Semiring(S) | Carrier set of S |
| Metric⟨τ_in, τ_out⟩ | ⟦τ_in⟧ → ⟦τ_out⟧ |
| NGram(n) | Σⁿ (strings of length n) |

### 3.2 Core Expression Semantics

Let ρ : Var → Value be an environment mapping variables to values.

**Literals**:
```
⟦n⟧ρ         = n                              (n ∈ ℤ)
⟦r⟧ρ         = r                              (r ∈ ℝ)
⟦b⟧ρ         = b                              (b ∈ 𝔹)
⟦s⟧ρ         = s                              (s ∈ Σ*)
```

**Variables and Binding**:
```
⟦x⟧ρ         = ρ(x)
⟦let x = e₁ in e₂⟧ρ  = ⟦e₂⟧(ρ[x ↦ ⟦e₁⟧ρ])
```

**Arithmetic** (where • ∈ {+, -, *, /, %, ^}):
```
⟦e₁ • e₂⟧ρ   = ⟦e₁⟧ρ  •  ⟦e₂⟧ρ
```

**Semiring operations**:
```
⟦e₁ ⊕ e₂⟧ρ   = add_S(⟦e₁⟧ρ, ⟦e₂⟧ρ)
⟦e₁ ⊗ e₂⟧ρ   = mul_S(⟦e₁⟧ρ, ⟦e₂⟧ρ)
```

**Control flow**:
```
⟦if e₁ then e₂ else e₃⟧ρ  = ⟦e₂⟧ρ   if ⟦e₁⟧ρ = ⊤
                             = ⟦e₃⟧ρ   if ⟦e₁⟧ρ = ⊥

⟦match e with | pᵢ => eᵢ⟧ρ = ⟦eⱼ⟧(ρ ∪ θⱼ)
    where j = min{i | match(⟦e⟧ρ, pᵢ) = Some(θᵢ)}
```

**Functions**:
```
⟦fn(x₁,…,xₙ) => e⟧ρ       = λ(v₁,…,vₙ). ⟦e⟧(ρ[x₁↦v₁,…,xₙ↦vₙ])
⟦f(e₁,…,eₙ)⟧ρ              = (⟦f⟧ρ)(⟦e₁⟧ρ, …, ⟦eₙ⟧ρ)
```

**Compound values**:
```
⟦[e₁, …, eₙ]⟧ρ             = [⟦e₁⟧ρ, …, ⟦eₙ⟧ρ]
⟦(e₁, …, eₙ)⟧ρ             = (⟦e₁⟧ρ, …, ⟦eₙ⟧ρ)
⟦e.i⟧ρ                      = πᵢ(⟦e⟧ρ)
⟦e₁[e₂]⟧ρ                   = ⟦e₁⟧ρ[⟦e₂⟧ρ]    (list indexing)
```

### 3.3 Domain-Specific Semantics (WFA Denotations)

The following defines the WFA semantics. A **WFA** over semiring
⟨S, ⊕, ⊗, 0_S, 1_S⟩ is a tuple 𝒜 = (Q, Σ, δ, I, F) where:
- Q is a finite set of states
- Σ is the input alphabet
- δ : Q × Σ × Q → S is the transition weight function
- I : Q → S assigns initial weights
- F : Q → S assigns final weights

The **formal power series** recognized by 𝒜 maps each string w ∈ Σ* to:

$$⟦𝒜⟧(w) = \bigoplus_{π ∈ \text{Paths}(w)} I(π_0) ⊗ δ(π_0, w_1, π_1) ⊗ … ⊗ δ(π_{n-1}, w_n, π_n) ⊗ F(π_n)$$

**Metric denotation**:
```
⟦metric m(c: String, r: String) : Semiring(S) { e }⟧
    = 𝒜_m : Σ* × Σ* → S
    where 𝒜_m(c, r) = ⟦e⟧[c ↦ c, r ↦ r]
```

When e is in the WFA-expressible fragment (§4), 𝒜_m is realized as
a concrete WFA over S.

**N-gram extraction**:
```
⟦ngram(n, s)⟧ρ = { w ∈ Σⁿ | w is a contiguous substring of ⟦s⟧ρ }
```

As a WFA over the Counting semiring:
```
𝒜_{ngram(n)} = (Q, Σ, δ, I, F)  where
    Q = {q₀, q₁, …, qₙ}
    I(q₀) = 1,  I(qᵢ) = 0  for i > 0
    F(qₙ) = 1,  F(qᵢ) = 0  for i < n
    δ(qᵢ, a, qᵢ₊₁) = 1  for all a ∈ Σ, 0 ≤ i < n
    δ(q₀, a, q₀) = 1     for all a ∈ Σ  (skip prefix)
    δ(qₙ, a, q₀) = 1     for all a ∈ Σ  (restart after match)
```

This WFA assigns to each input string the **count** of distinct n-gram
occurrences — a formal power series f : Σ* → ℕ.

**Aggregation**:
```
⟦aggregate(sum, xs, x => body)⟧ρ
    = ⊕_{v ∈ ⟦xs⟧ρ} ⟦body⟧(ρ[x ↦ v])

⟦aggregate(product, xs, x => body)⟧ρ
    = ⊗_{v ∈ ⟦xs⟧ρ} ⟦body⟧(ρ[x ↦ v])

⟦aggregate(count, xs, x => body)⟧ρ
    = |{ v ∈ ⟦xs⟧ρ | ⟦body⟧(ρ[x ↦ v]) = ⊤ }|  ∈ ℕ

⟦aggregate(min, xs, x => body)⟧ρ
    = min_{v ∈ ⟦xs⟧ρ} ⟦body⟧(ρ[x ↦ v])

⟦aggregate(max, xs, x => body)⟧ρ
    = max_{v ∈ ⟦xs⟧ρ} ⟦body⟧(ρ[x ↦ v])
```

When the body denotes a WFA, aggregation lifts to the semiring product or
union of automata:
- `sum` → WFA union (nondeterministic choice)
- `product` → WFA intersection (via tensor product of states)

**Clip** *(bounded counting)*:
```
⟦clip(e, k)⟧ρ = min(⟦e⟧ρ, k)
```

As a semiring morphism: φ_k : Counting → BoundedCounting(k) where
φ_k(n) = min(n, k). The WFA 𝒜 over Counting is transformed to
φ_k(𝒜) over BoundedCounting(k) by applying φ_k to all transition
weights, initial weights, and final weights.

**Compose** *(transducer composition)*:
```
⟦compose(f, g)⟧ρ = ⟦g⟧ρ ∘ ⟦f⟧ρ
```

When f and g are WFAs over the same semiring S, their composition is
the standard weighted transducer composition:

Given 𝒜_f = (Q_f, Σ, δ_f, I_f, F_f) and 𝒜_g = (Q_g, Σ, δ_g, I_g, F_g):
```
𝒜_{compose(f,g)} = (Q_f × Q_g, Σ, δ', I', F')  where
    I'(p, q) = I_f(p) ⊗ I_g(q)
    F'(p, q) = F_f(p) ⊗ F_g(q)
    δ'((p₁,q₁), a, (p₂,q₂)) = δ_f(p₁, a, p₂) ⊗ δ_g(q₁, a, q₂)
```

**Tokenize**:
```
⟦tokenize(s)⟧ρ       = tok_default(⟦s⟧ρ)  ∈ ℕ*
⟦tokenize(s, t)⟧ρ    = tok_t(⟦s⟧ρ)        ∈ ℕ*
```

where tok_t : Σ* → ℕ* is the tokenization function associated with
tokenizer t, mapping a string to a sequence of token identifiers.

**Semiring cast**:
```
⟦e as S₂⟧ρ = φ_{S₁→S₂}(⟦e⟧ρ)
```

where φ_{S₁→S₂} is the canonical semiring homomorphism from S₁ to S₂
(per the coercion table in §2.5, T-SemiringCast).

---

## 4. WFA-Expressible Fragment

### 4.1 Characterization Theorem

**Theorem 4.1** (WFA Expressibility).
*An EvalSpec metric* `m` *is WFA-expressible if and only if the formal
power series* ⟦m⟧ : Σ* → S *is a **rational formal power series** over
the semiring S.*

Equivalently, ⟦m⟧ is WFA-expressible iff it belongs to the rational
closure of the finite power series under the operations:
1. **Sum** (⊕): union of formal power series
2. **Product** (⊗, Cauchy/Hadamard): product of formal power series
3. **Kleene star** (*): iteration closure

**Proof sketch**.
(⇒) Every WFA recognizes a rational power series by the
Schützenberger–Kleene theorem generalized to semirings.
(⇐) The EvalSpec compiler constructs a WFA for each expression in the
rational fragment by structural induction: `ngram` builds base automata,
`aggregate(sum,·)` takes automaton union, `aggregate(product,·)` takes
automaton product, `clip` applies a semiring homomorphism, and `compose`
performs weighted transducer composition. Each operation preserves
rationality.  ∎

### 4.2 Decidability

**Theorem 4.2** (Equivalence Decidability).
*Given two WFA-expressible EvalSpec metrics* m₁, m₂ *over the same
semiring S, the equivalence* ⟦m₁⟧ = ⟦m₂⟧ *is decidable when S is a
commutative ring or a field.*

**Procedure**: Minimize both WFAs via weighted Hopcroft minimization
(or the Schützenberger algorithm for weighted automata over a field),
then check isomorphism of the canonical forms.

**Complexity**: The minimization algorithm runs in O(n² · |Σ|) time
where n = max(|Q₁|, |Q₂|), yielding a polynomial-time decision
procedure.

**Remark**. For semirings that are not fields (e.g., Counting = (ℕ, +, ×)),
equivalence is decidable via the approach of Boreale (2009) using a
linear-algebraic characterization of WFA equivalence.

### 4.3 Expressibility Boundary

The following table characterizes which common NLG evaluation metrics
fall within the WFA-expressible fragment:

| Metric / Operation | WFA-Expressible | Semiring | Notes |
|---|:---:|---|---|
| N-gram precision (individual) | ✓ | Counting | Ratio of clipped counts |
| N-gram recall (individual) | ✓ | Counting | Ratio of matched counts |
| ROUGE-N | ✓ | Counting | N-gram overlap / reference count |
| Token-level F1 | ✓ | Counting | Harmonic mean of P, R over tokens |
| Exact match | ✓ | Boolean | Equality check as boolean WFA |
| Substring containment | ✓ | Boolean | NFA for substring recognition |
| Weighted precision | ✓ | Real | Real-weighted n-gram counts |
| Clipped n-gram count | ✓ | BoundedCounting(k) | clip applied to counting WFA |
| Viterbi-best alignment | ✓ | Viterbi | Max-probability path |
| Tropical edit distance | ✓ | Tropical | Levenshtein via tropical WFA |
| **Geometric mean** | **✗** | — | Not a rational operation |
| **BLEU (corpus-level)** | **✗** | — | Geometric mean of n-gram precs |
| **BERTScore** | **✗** | — | Requires neural embedding |
| **Calibration error** | **✗** | — | Global distribution statistic |
| **Perplexity** | **✗** | — | Requires exponentiation over corpus |

### 4.4 BLEU Decomposition

BLEU illustrates the boundary between the WFA fragment and circuit
gadgets.

**Definition**. Corpus-level BLEU with maximum n-gram order N is:
```
BLEU = BP · exp( (1/N) · Σᵢ₌₁ᴺ log pᵢ )
```

where pᵢ is the modified i-gram precision and BP is the brevity penalty.

**Decomposition into WFA + Circuit**:

1. **WFA Fragment** — Each individual modified n-gram precision pₙ:
   ```
   pₙ = aggregate(sum, ngram(n, candidate), g =>
           clip(count_in(g, reference), max_count(g, reference)))
         /
         aggregate(count, ngram(n, candidate), g => true)
   ```
   This is WFA-expressible over BoundedCounting(k), since clipped
   counting and plain counting are both rational operations.

2. **Circuit Gadget** — The geometric mean and brevity penalty:
   ```
   BLEU_circuit(p₁, …, p_N, c_len, r_len) =
       BP(c_len, r_len) · (p₁ · p₂ · … · p_N)^(1/N)
   ```
   The Nth root (equivalently, exponentiation by 1/N) is **not** a
   rational operation over any semiring. It requires an arithmetic
   circuit gadget operating on the WFA outputs.

**Architecture**:
```
    ┌──────────────┐     ┌──────────────┐
    │  WFA (p₁)    │────▶│              │
    │  over        │     │   Circuit    │
    │  BndCount(k) │     │   Gadget     │
    ├──────────────┤     │              │
    │  WFA (p₂)    │────▶│  geometric   │──▶ BLEU score
    │  over        │     │  mean + BP   │
    │  BndCount(k) │     │              │
    ├──────────────┤     │              │
    │   ⋮          │────▶│              │
    └──────────────┘     └──────────────┘
```

The WFA components are verifiable via the Spectacles ZK proof system
(operating over the Goldilocks semiring). The circuit gadget is verified
by a separate arithmetic circuit proof that consumes the WFA outputs
as public inputs.

### 4.5 Formal Criterion

**Definition 4.3** (WFA Fragment Membership).
An EvalSpec expression e belongs to the WFA-expressible fragment iff
it is generated by the following restricted grammar:

```
⟨wfa-expr⟩   ::= ⟨literal⟩
               | ⟨ident⟩
               | ⟨wfa-expr⟩ ⟨arith-op⟩ ⟨wfa-expr⟩
               | ⟨wfa-expr⟩ ⟨semiring-op⟩ ⟨wfa-expr⟩
               | 'if' ⟨bool-expr⟩ 'then' ⟨wfa-expr⟩ 'else' ⟨wfa-expr⟩
               | 'let' ⟨ident⟩ '=' ⟨wfa-expr⟩ 'in' ⟨wfa-expr⟩
               | 'ngram' '(' ⟨int-lit⟩ ',' ⟨wfa-expr⟩ ')'
               | 'tokenize' '(' ⟨wfa-expr⟩ ')'
               | 'aggregate' '(' ⟨rational-op⟩ ',' ⟨wfa-expr⟩ ',' ⟨ident⟩ '=>' ⟨wfa-expr⟩ ')'
               | 'clip' '(' ⟨wfa-expr⟩ ',' ⟨wfa-expr⟩ ')'
               | 'compose' '(' ⟨wfa-expr⟩ ',' ⟨wfa-expr⟩ ')'
               | '*' ⟨wfa-expr⟩

⟨rational-op⟩ ::= 'sum' | 'product' | 'count' | 'min' | 'max'

⟨semiring-op⟩ ::= '⊕' | '⊗'
```

Expressions outside this fragment (e.g., those using `mean`, arbitrary
function application, or non-rational aggregations like geometric mean)
require **circuit gadgets** for evaluation and ZK verification.
