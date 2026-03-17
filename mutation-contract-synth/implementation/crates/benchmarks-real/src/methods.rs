//! Method specifications for benchmarking.
//!
//! Each method has:
//! - A signature and description
//! - Ground-truth pre/postconditions
//! - Simulated error predicates from mutation operators (AOR, ROR, UOI, LCR)
//!
//! The error predicates are derived by hand-applying mutation operators to the
//! method body and computing the weakest precondition of the mutant differing
//! from the original (i.e., the condition under which the mutant produces a
//! different result).

use shared_types::{Formula, MutantId, Predicate, Relation, Term};
use uuid::Uuid;

/// A method specification for benchmarking.
pub struct MethodSpec {
    pub name: String,
    pub signature: String,
    pub description: String,
    pub mutation_ops: Vec<&'static str>,
    /// Ground-truth precondition formulas as strings.
    pub ground_truth_pre: Vec<String>,
    /// Ground-truth postcondition formulas as strings.
    pub ground_truth_post: Vec<String>,
    /// Error predicates keyed by mutant ID.
    /// E(m) = condition under which mutant m differs from original.
    pub error_predicates: Vec<(MutantId, Formula)>,
}

fn mid() -> MutantId {
    MutantId(Uuid::new_v4())
}

/// Build the full set of 8 method specifications.
pub fn build_method_specs() -> Vec<MethodSpec> {
    vec![
        build_abs(),
        build_max(),
        build_min(),
        build_clamp(),
        build_signum(),
        build_linear_search(),
        build_safe_get(),
        build_safe_divide(),
    ]
}

// =========================================================================
// 1. abs(x) -> int
// =========================================================================
fn build_abs() -> MethodSpec {
    // Body: if x >= 0 then x else -x
    // ROR mutation on (x >= 0):
    //   m1: x > 0  → error when x == 0 (returns -0=0, same; but boundary matters for other types)
    //   m2: x <= 0 → error when x > 0 (returns -x instead of x)
    //   m3: x < 0  → error when x == 0 (returns -0)
    //   m4: x == 0 → error when x > 0 (returns -x) or x < 0 (returns x)
    // UOI on return x:
    //   m5: return -x (when x >= 0) → error when x > 0
    // UOI on return -x:
    //   m6: return x  (when x < 0) → error when x < 0

    let eps = vec![
        // m1: ROR x>=0 → x>0.  Error pred: x == 0 (boundary case; ret same since abs(0)=0)
        //     Actually: mutant returns -x when x==0 → -0 = 0. No real error for int.
        //     For the synthesis this is an equivalent mutant.  We include it to test skipping.
        (
            mid(),
            Formula::atom(Predicate::eq(Term::var("x"), Term::constant(0))),
        ),
        // m2: ROR x>=0 → x<=0.  Error pred: x > 0.
        (
            mid(),
            Formula::atom(Predicate::gt(Term::var("x"), Term::constant(0))),
        ),
        // m3: ROR x>=0 → x<0.   Error pred: x >= 0 (takes else-branch for x>=0).
        (
            mid(),
            Formula::atom(Predicate::ge(Term::var("x"), Term::constant(0))),
        ),
        // m4: ROR x>=0 → x==0.  Error pred: x != 0.
        (
            mid(),
            Formula::atom(Predicate::ne(Term::var("x"), Term::constant(0))),
        ),
        // m5: UOI return x → return -x (in then-branch).  Error pred: x > 0.
        (
            mid(),
            Formula::atom(Predicate::gt(Term::var("x"), Term::constant(0))),
        ),
        // m6: UOI return -x → return x (in else-branch).  Error pred: x < 0.
        (
            mid(),
            Formula::atom(Predicate::lt(Term::var("x"), Term::constant(0))),
        ),
    ];

    MethodSpec {
        name: "abs".into(),
        signature: "abs(x: int) -> int".into(),
        description: "Absolute value".into(),
        mutation_ops: vec!["ROR", "UOI"],
        ground_truth_pre: vec!["true".into()],
        ground_truth_post: vec![
            "ret >= 0".into(),
            "(x >= 0) -> ret == x".into(),
            "(x < 0) -> ret == -x".into(),
        ],
        error_predicates: eps,
    }
}

// =========================================================================
// 2. max(a, b) -> int
// =========================================================================
fn build_max() -> MethodSpec {
    // Body: if a >= b then a else b
    // ROR on (a >= b):
    //   m1: a > b  → error when a == b (returns b instead of a, same value)
    //   m2: a <= b → error when a > b
    //   m3: a < b  → error when a >= b (returns b when should return a)
    //   m4: a == b → error when a != b (specifically a > b returns b)
    //   m5: a != b → error when a == b (returns b=a, equivalent)

    let eps = vec![
        (
            mid(),
            Formula::atom(Predicate::eq(Term::var("a"), Term::var("b"))),
        ),
        (
            mid(),
            Formula::atom(Predicate::gt(Term::var("a"), Term::var("b"))),
        ),
        (
            mid(),
            Formula::atom(Predicate::ge(Term::var("a"), Term::var("b"))),
        ),
        (
            mid(),
            Formula::atom(Predicate::ne(Term::var("a"), Term::var("b"))),
        ),
        (
            mid(),
            Formula::atom(Predicate::lt(Term::var("a"), Term::var("b"))),
        ),
    ];

    MethodSpec {
        name: "max".into(),
        signature: "max(a: int, b: int) -> int".into(),
        description: "Maximum of two integers".into(),
        mutation_ops: vec!["ROR"],
        ground_truth_pre: vec!["true".into()],
        ground_truth_post: vec![
            "ret >= a".into(),
            "ret >= b".into(),
            "ret == a || ret == b".into(),
        ],
        error_predicates: eps,
    }
}

// =========================================================================
// 3. min(a, b) -> int
// =========================================================================
fn build_min() -> MethodSpec {
    // Body: if a <= b then a else b
    let eps = vec![
        (
            mid(),
            Formula::atom(Predicate::eq(Term::var("a"), Term::var("b"))),
        ),
        (
            mid(),
            Formula::atom(Predicate::lt(Term::var("a"), Term::var("b"))),
        ),
        (
            mid(),
            Formula::atom(Predicate::le(Term::var("a"), Term::var("b"))),
        ),
        (
            mid(),
            Formula::atom(Predicate::ne(Term::var("a"), Term::var("b"))),
        ),
        (
            mid(),
            Formula::atom(Predicate::gt(Term::var("a"), Term::var("b"))),
        ),
    ];

    MethodSpec {
        name: "min".into(),
        signature: "min(a: int, b: int) -> int".into(),
        description: "Minimum of two integers".into(),
        mutation_ops: vec!["ROR"],
        ground_truth_pre: vec!["true".into()],
        ground_truth_post: vec![
            "ret <= a".into(),
            "ret <= b".into(),
            "ret == a || ret == b".into(),
        ],
        error_predicates: eps,
    }
}

// =========================================================================
// 4. clamp(x, lo, hi) -> int
// =========================================================================
fn build_clamp() -> MethodSpec {
    // Body: if x < lo then lo else if x > hi then hi else x
    // ROR on x < lo:
    //   m1: x <= lo → error when x == lo (returns lo instead of x=lo, equivalent)
    //   m2: x >= lo → error when x < lo
    //   m3: x > lo  → error when x <= lo (returns lo when should return x or lo)
    // ROR on x > hi:
    //   m4: x >= hi → error when x == hi
    //   m5: x < hi  → error when x >= hi (returns x instead of hi)
    //   m6: x <= hi → error when x > hi
    // AOR on return lo → return lo + 1:
    //   m7: error when x < lo (returns lo+1 instead of lo)
    // AOR on return hi → return hi - 1:
    //   m8: error when x > hi

    let eps = vec![
        (
            mid(),
            Formula::atom(Predicate::eq(Term::var("x"), Term::var("lo"))),
        ),
        (
            mid(),
            Formula::atom(Predicate::lt(Term::var("x"), Term::var("lo"))),
        ),
        (
            mid(),
            Formula::atom(Predicate::le(Term::var("x"), Term::var("lo"))),
        ),
        (
            mid(),
            Formula::atom(Predicate::eq(Term::var("x"), Term::var("hi"))),
        ),
        (
            mid(),
            Formula::atom(Predicate::ge(Term::var("x"), Term::var("hi"))),
        ),
        (
            mid(),
            Formula::atom(Predicate::gt(Term::var("x"), Term::var("hi"))),
        ),
        (
            mid(),
            Formula::atom(Predicate::lt(Term::var("x"), Term::var("lo"))),
        ),
        (
            mid(),
            Formula::atom(Predicate::gt(Term::var("x"), Term::var("hi"))),
        ),
    ];

    MethodSpec {
        name: "clamp".into(),
        signature: "clamp(x: int, lo: int, hi: int) -> int".into(),
        description: "Clamp x to [lo, hi]".into(),
        mutation_ops: vec!["ROR", "AOR"],
        ground_truth_pre: vec!["lo <= hi".into()],
        ground_truth_post: vec![
            "ret >= lo".into(),
            "ret <= hi".into(),
            "ret == x || ret == lo || ret == hi".into(),
        ],
        error_predicates: eps,
    }
}

// =========================================================================
// 5. signum(x) -> int
// =========================================================================
fn build_signum() -> MethodSpec {
    // Body: if x > 0 then 1 else if x < 0 then -1 else 0
    // ROR on x > 0:
    //   m1: x >= 0 → error when x == 0
    //   m2: x < 0  → error when x >= 0
    //   m3: x == 0 → error when x > 0
    // ROR on x < 0:
    //   m4: x <= 0 → error when x == 0
    //   m5: x > 0  → error when x < 0  (equivalent? no, falls through differently)
    //   m6: x == 0 → error when x < 0
    // AOR on return 1 → return 0:
    //   m7: error when x > 0
    // AOR on return -1 → return 0:
    //   m8: error when x < 0

    let eps = vec![
        (
            mid(),
            Formula::atom(Predicate::eq(Term::var("x"), Term::constant(0))),
        ),
        (
            mid(),
            Formula::atom(Predicate::ge(Term::var("x"), Term::constant(0))),
        ),
        (
            mid(),
            Formula::atom(Predicate::gt(Term::var("x"), Term::constant(0))),
        ),
        (
            mid(),
            Formula::atom(Predicate::eq(Term::var("x"), Term::constant(0))),
        ),
        (
            mid(),
            Formula::atom(Predicate::lt(Term::var("x"), Term::constant(0))),
        ),
        (
            mid(),
            Formula::atom(Predicate::lt(Term::var("x"), Term::constant(0))),
        ),
        (
            mid(),
            Formula::atom(Predicate::gt(Term::var("x"), Term::constant(0))),
        ),
        (
            mid(),
            Formula::atom(Predicate::lt(Term::var("x"), Term::constant(0))),
        ),
    ];

    MethodSpec {
        name: "signum".into(),
        signature: "signum(x: int) -> int".into(),
        description: "Sign function: 1, -1, or 0".into(),
        mutation_ops: vec!["ROR", "AOR"],
        ground_truth_pre: vec!["true".into()],
        ground_truth_post: vec![
            "ret >= -1".into(),
            "ret <= 1".into(),
            "(x > 0) -> ret == 1".into(),
            "(x < 0) -> ret == -1".into(),
            "(x == 0) -> ret == 0".into(),
        ],
        error_predicates: eps,
    }
}

// =========================================================================
// 6. linear_search(arr, key) -> int
// =========================================================================
fn build_linear_search() -> MethodSpec {
    // Body: for i in 0..4: if arr[i] == key then return i; return -1
    // ROR on arr[i] == key:
    //   m1: arr[i] != key → error when arr contains key
    //   m2: arr[i] < key  → error when arr[i] == key (different type, but for int)
    //   m3: arr[i] > key  → error when arr[i] == key
    // AOR on return i → return i+1:
    //   m4: error when arr[i] == key (returns wrong index)
    // AOR on return -1 → return 0:
    //   m5: error when key not found (returns 0 instead of -1)
    // ROR on loop bound i < 4:
    //   m6: i < 3 → error when key at index 3
    //   m7: i <= 4 → error when accessing arr[4] (out of bounds)

    let eps = vec![
        // m1: arr[i] != key.  Error when any arr[i] == key (i.e., key exists).
        (
            mid(),
            Formula::or(vec![
                Formula::atom(Predicate::eq(
                    Term::array_select(Term::var("arr"), Term::constant(0)),
                    Term::var("key"),
                )),
                Formula::atom(Predicate::eq(
                    Term::array_select(Term::var("arr"), Term::constant(1)),
                    Term::var("key"),
                )),
                Formula::atom(Predicate::eq(
                    Term::array_select(Term::var("arr"), Term::constant(2)),
                    Term::var("key"),
                )),
                Formula::atom(Predicate::eq(
                    Term::array_select(Term::var("arr"), Term::constant(3)),
                    Term::var("key"),
                )),
            ]),
        ),
        // m2: arr[0] < key instead of ==.  Error when arr[0] == key.
        (
            mid(),
            Formula::atom(Predicate::eq(
                Term::array_select(Term::var("arr"), Term::constant(0)),
                Term::var("key"),
            )),
        ),
        // m3: arr[0] > key instead of ==.
        (
            mid(),
            Formula::atom(Predicate::eq(
                Term::array_select(Term::var("arr"), Term::constant(0)),
                Term::var("key"),
            )),
        ),
        // m4: return i+1 instead of i.  Error when arr[i] == key for some i.
        (
            mid(),
            Formula::or(vec![
                Formula::atom(Predicate::eq(
                    Term::array_select(Term::var("arr"), Term::constant(0)),
                    Term::var("key"),
                )),
                Formula::atom(Predicate::eq(
                    Term::array_select(Term::var("arr"), Term::constant(1)),
                    Term::var("key"),
                )),
            ]),
        ),
        // m5: return 0 instead of -1.  Error when key not in arr.
        (
            mid(),
            Formula::and(vec![
                Formula::atom(Predicate::ne(
                    Term::array_select(Term::var("arr"), Term::constant(0)),
                    Term::var("key"),
                )),
                Formula::atom(Predicate::ne(
                    Term::array_select(Term::var("arr"), Term::constant(1)),
                    Term::var("key"),
                )),
                Formula::atom(Predicate::ne(
                    Term::array_select(Term::var("arr"), Term::constant(2)),
                    Term::var("key"),
                )),
                Formula::atom(Predicate::ne(
                    Term::array_select(Term::var("arr"), Term::constant(3)),
                    Term::var("key"),
                )),
            ]),
        ),
        // m6: loop bound i < 3.  Error when key at index 3.
        (
            mid(),
            Formula::atom(Predicate::eq(
                Term::array_select(Term::var("arr"), Term::constant(3)),
                Term::var("key"),
            )),
        ),
        // m7: loop bound i <= 4 (would be OOB, but EP is access condition)
        (mid(), Formula::True),
    ];

    MethodSpec {
        name: "linear_search".into(),
        signature: "linear_search(arr: int[4], key: int) -> int".into(),
        description: "Linear search over 4-element array".into(),
        mutation_ops: vec!["ROR", "AOR"],
        ground_truth_pre: vec!["true".into()],
        ground_truth_post: vec![
            "ret >= -1".into(),
            "ret <= 3".into(),
            "(ret >= 0) -> arr[ret] == key".into(),
        ],
        error_predicates: eps,
    }
}

// =========================================================================
// 7. safe_get(arr, idx, default_val) -> int
// =========================================================================
fn build_safe_get() -> MethodSpec {
    // Body: if idx >= 0 && idx < 8 then arr[idx] else default_val
    // ROR on idx >= 0:
    //   m1: idx > 0 → error when idx == 0
    //   m2: idx < 0 → error when idx >= 0
    // ROR on idx < 8:
    //   m3: idx <= 8 → error when idx == 8
    //   m4: idx >= 8 → error when idx < 8
    // UOI on return default_val → return -default_val:
    //   m5: error when (idx < 0 || idx >= 8) && default_val != 0

    let eps = vec![
        (
            mid(),
            Formula::atom(Predicate::eq(Term::var("idx"), Term::constant(0))),
        ),
        (
            mid(),
            Formula::atom(Predicate::ge(Term::var("idx"), Term::constant(0))),
        ),
        (
            mid(),
            Formula::atom(Predicate::eq(Term::var("idx"), Term::constant(8))),
        ),
        (
            mid(),
            Formula::atom(Predicate::lt(Term::var("idx"), Term::constant(8))),
        ),
        (
            mid(),
            Formula::and(vec![
                Formula::or(vec![
                    Formula::atom(Predicate::lt(Term::var("idx"), Term::constant(0))),
                    Formula::atom(Predicate::ge(Term::var("idx"), Term::constant(8))),
                ]),
                Formula::atom(Predicate::ne(Term::var("default_val"), Term::constant(0))),
            ]),
        ),
    ];

    MethodSpec {
        name: "safe_get".into(),
        signature: "safe_get(arr: int[8], idx: int, default_val: int) -> int".into(),
        description: "Bounds-checked array access with default".into(),
        mutation_ops: vec!["ROR", "UOI"],
        ground_truth_pre: vec!["true".into()],
        ground_truth_post: vec![
            "(idx >= 0 && idx < 8) -> ret == arr[idx]".into(),
            "(idx < 0 || idx >= 8) -> ret == default_val".into(),
        ],
        error_predicates: eps,
    }
}

// =========================================================================
// 8. safe_divide(dividend, divisor) -> int
// =========================================================================
fn build_safe_divide() -> MethodSpec {
    // Body: if divisor == 0 then 0 else dividend / divisor
    // ROR on divisor == 0:
    //   m1: divisor != 0 → error when divisor == 0 (divides by zero) OR divisor != 0 (returns 0)
    //   m2: divisor < 0  → error when divisor > 0 (returns 0 instead of dividend/divisor)
    //   m3: divisor > 0  → error when divisor < 0
    //   m4: divisor >= 0 → error when divisor < 0
    // AOR on dividend / divisor → dividend * divisor:
    //   m5: error when divisor != 0 && divisor != 1 && divisor != -1 && dividend != 0
    // AOR on return 0 → return 1:
    //   m6: error when divisor == 0

    let eps = vec![
        // m1: divisor != 0 instead of ==.  Error pred: divisor != 0 (returns 0) OR divisor == 0 (does division)
        (
            mid(),
            Formula::atom(Predicate::ne(Term::var("divisor"), Term::constant(0))),
        ),
        // m2: divisor < 0.  Error pred: divisor > 0 (returns 0 instead of doing division)
        (
            mid(),
            Formula::atom(Predicate::gt(Term::var("divisor"), Term::constant(0))),
        ),
        // m3: divisor > 0.  Error pred: divisor < 0
        (
            mid(),
            Formula::atom(Predicate::lt(Term::var("divisor"), Term::constant(0))),
        ),
        // m4: divisor >= 0.  Error pred: divisor < 0
        (
            mid(),
            Formula::atom(Predicate::lt(Term::var("divisor"), Term::constant(0))),
        ),
        // m5: AOR * instead of /.  Error pred: divisor != 0 && divisor != 1 && dividend != 0
        (
            mid(),
            Formula::and(vec![
                Formula::atom(Predicate::ne(Term::var("divisor"), Term::constant(0))),
                Formula::atom(Predicate::ne(Term::var("divisor"), Term::constant(1))),
                Formula::atom(Predicate::ne(Term::var("dividend"), Term::constant(0))),
            ]),
        ),
        // m6: return 1 instead of 0.  Error pred: divisor == 0
        (
            mid(),
            Formula::atom(Predicate::eq(Term::var("divisor"), Term::constant(0))),
        ),
    ];

    MethodSpec {
        name: "safe_divide".into(),
        signature: "safe_divide(dividend: int, divisor: int) -> int".into(),
        description: "Division with zero guard".into(),
        mutation_ops: vec!["ROR", "AOR"],
        ground_truth_pre: vec!["divisor != 0".into()],
        ground_truth_post: vec![
            "(divisor != 0) -> ret == dividend / divisor".into(),
            "(divisor == 0) -> ret == 0".into(),
        ],
        error_predicates: eps,
    }
}
