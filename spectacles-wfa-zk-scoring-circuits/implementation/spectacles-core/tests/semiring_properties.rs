//! Property-based tests establishing the formal correspondence between Lean 4
//! KleeneSemiring axioms and the Rust semiring implementations.
//!
//! The Lean 4 proofs verify that KleeneSemiring instances satisfy a set of
//! algebraic axioms (commutativity, associativity, identity, annihilation,
//! distributivity, idempotency). These proptest-driven tests exercise the
//! *same* axioms against the concrete Rust types, ensuring the implementation
//! honours the formally-verified specification.

use ordered_float::OrderedFloat;
use proptest::prelude::*;

use spectacles_core::wfa::semiring::{
    BooleanSemiring, CountingSemiring, GoldilocksField, Semiring, StarSemiring, TropicalSemiring,
    GOLDILOCKS_PRIME,
};

// =========================================================================
// Strategies
// =========================================================================

fn boolean_semiring() -> impl Strategy<Value = BooleanSemiring> {
    any::<bool>().prop_map(BooleanSemiring::new)
}

fn counting_semiring() -> impl Strategy<Value = CountingSemiring> {
    // Keep values small enough that saturating_add/mul behave like true arithmetic
    // for most pairs, while still exercising larger values occasionally.
    (0u64..=1_000_000).prop_map(CountingSemiring::new)
}

fn tropical_semiring() -> impl Strategy<Value = TropicalSemiring> {
    // Use finite values only; infinity is the semiring zero and tested separately.
    // Keep magnitudes moderate so that f64 addition stays exact or nearly exact.
    (-1e6f64..1e6f64).prop_map(TropicalSemiring::new)
}

/// Approximate equality for tropical semiring: multiplication is f64 addition,
/// so associativity and distributivity may show ULP-level rounding differences.
fn tropical_approx_eq(a: &TropicalSemiring, b: &TropicalSemiring) -> bool {
    let x = a.value.into_inner();
    let y = b.value.into_inner();
    if x == y {
        return true;
    }
    // Both infinite with same sign
    if x.is_infinite() && y.is_infinite() && x.signum() == y.signum() {
        return true;
    }
    let diff = (x - y).abs();
    let scale = x.abs().max(y.abs()).max(1.0);
    diff / scale < 1e-10
}

fn goldilocks_field() -> impl Strategy<Value = GoldilocksField> {
    (0u64..GOLDILOCKS_PRIME).prop_map(|v| GoldilocksField::new(v % GOLDILOCKS_PRIME))
}

// =========================================================================
// Macro: generate a full axiom suite for a given semiring
// =========================================================================

/// Generates proptest functions for the core semiring axioms.
macro_rules! semiring_axiom_tests {
    ($mod_name:ident, $strategy:expr, $type_name:ty) => {
        mod $mod_name {
            use super::*;

            proptest! {
                /// Lean 4: `theorem add_comm (a b : S) : a ⊕ b = b ⊕ a`
                #[test]
                fn additive_commutativity(a in $strategy, b in $strategy) {
                    prop_assert_eq!(a.add(&b), b.add(&a));
                }

                /// Lean 4: `theorem add_assoc (a b c : S) : (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)`
                #[test]
                fn additive_associativity(a in $strategy, b in $strategy, c in $strategy) {
                    prop_assert_eq!(a.add(&b).add(&c), a.add(&b.add(&c)));
                }

                /// Lean 4: `theorem add_zero (a : S) : a ⊕ 0 = a`
                #[test]
                fn additive_identity(a in $strategy) {
                    let zero = <$type_name>::zero();
                    prop_assert_eq!(a.add(&zero), a.clone());
                    prop_assert_eq!(zero.add(&a), a);
                }

                /// Lean 4: `theorem mul_assoc (a b c : S) : (a ⊗ b) ⊗ c = a ⊗ (b ⊗ c)`
                #[test]
                fn multiplicative_associativity(a in $strategy, b in $strategy, c in $strategy) {
                    prop_assert_eq!(a.mul(&b).mul(&c), a.mul(&b.mul(&c)));
                }

                /// Lean 4: `theorem mul_one (a : S) : a ⊗ 1 = a` and `theorem one_mul (a : S) : 1 ⊗ a = a`
                #[test]
                fn multiplicative_identity(a in $strategy) {
                    let one = <$type_name>::one();
                    prop_assert_eq!(a.mul(&one), a.clone());
                    prop_assert_eq!(one.mul(&a), a);
                }

                /// Lean 4: `theorem mul_zero (a : S) : a ⊗ 0 = 0` and `theorem zero_mul (a : S) : 0 ⊗ a = 0`
                #[test]
                fn zero_annihilation(a in $strategy) {
                    let zero = <$type_name>::zero();
                    prop_assert_eq!(a.mul(&zero), zero.clone());
                    prop_assert_eq!(zero.mul(&a), zero);
                }

                /// Lean 4: `theorem left_distrib (a b c : S) : a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c)`
                #[test]
                fn left_distributivity(a in $strategy, b in $strategy, c in $strategy) {
                    prop_assert_eq!(
                        a.mul(&b.add(&c)),
                        a.mul(&b).add(&a.mul(&c))
                    );
                }

                /// Lean 4: `theorem right_distrib (a b c : S) : (b ⊕ c) ⊗ a = (b ⊗ a) ⊕ (c ⊗ a)`
                #[test]
                fn right_distributivity(a in $strategy, b in $strategy, c in $strategy) {
                    prop_assert_eq!(
                        b.add(&c).mul(&a),
                        b.mul(&a).add(&c.mul(&a))
                    );
                }
            }
        }
    };
}

// =========================================================================
// Instantiate axiom suites
// =========================================================================

semiring_axiom_tests!(boolean_axioms, boolean_semiring(), BooleanSemiring);
semiring_axiom_tests!(counting_axioms, counting_semiring(), CountingSemiring);
semiring_axiom_tests!(goldilocks_axioms, goldilocks_field(), GoldilocksField);

// ---------------------------------------------------------------------------
// TropicalSemiring axiom tests (separate because multiplication = f64 addition
// can exhibit ULP-level rounding, so we use approximate equality where needed)
// ---------------------------------------------------------------------------

mod tropical_axioms {
    use super::*;

    proptest! {
        /// Lean 4: `theorem add_comm (a b : S) : a ⊕ b = b ⊕ a`
        #[test]
        fn additive_commutativity(a in tropical_semiring(), b in tropical_semiring()) {
            prop_assert_eq!(a.add(&b), b.add(&a));
        }

        /// Lean 4: `theorem add_assoc (a b c : S) : (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)`
        #[test]
        fn additive_associativity(a in tropical_semiring(), b in tropical_semiring(), c in tropical_semiring()) {
            prop_assert_eq!(a.add(&b).add(&c), a.add(&b.add(&c)));
        }

        /// Lean 4: `theorem add_zero (a : S) : a ⊕ 0 = a`
        #[test]
        fn additive_identity(a in tropical_semiring()) {
            let zero = TropicalSemiring::zero();
            prop_assert_eq!(a.add(&zero), a.clone());
            prop_assert_eq!(zero.add(&a), a);
        }

        /// Lean 4: `theorem mul_assoc (a b c : S) : (a ⊗ b) ⊗ c = a ⊗ (b ⊗ c)`
        ///
        /// Uses approximate equality because tropical ⊗ is f64 addition.
        #[test]
        fn multiplicative_associativity(a in tropical_semiring(), b in tropical_semiring(), c in tropical_semiring()) {
            let lhs = a.mul(&b).mul(&c);
            let rhs = a.mul(&b.mul(&c));
            prop_assert!(tropical_approx_eq(&lhs, &rhs),
                "mul_assoc: lhs={:?} rhs={:?}", lhs, rhs);
        }

        /// Lean 4: `theorem mul_one (a : S) : a ⊗ 1 = a`
        #[test]
        fn multiplicative_identity(a in tropical_semiring()) {
            let one = TropicalSemiring::one();
            prop_assert_eq!(a.mul(&one), a.clone());
            prop_assert_eq!(one.mul(&a), a);
        }

        /// Lean 4: `theorem mul_zero (a : S) : a ⊗ 0 = 0`
        #[test]
        fn zero_annihilation(a in tropical_semiring()) {
            let zero = TropicalSemiring::zero();
            prop_assert_eq!(a.mul(&zero), zero.clone());
            prop_assert_eq!(zero.mul(&a), zero);
        }

        /// Lean 4: `theorem left_distrib (a b c : S) : a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c)`
        ///
        /// Uses approximate equality for the same f64-rounding reason.
        #[test]
        fn left_distributivity(a in tropical_semiring(), b in tropical_semiring(), c in tropical_semiring()) {
            let lhs = a.mul(&b.add(&c));
            let rhs = a.mul(&b).add(&a.mul(&c));
            prop_assert!(tropical_approx_eq(&lhs, &rhs),
                "left_distrib: lhs={:?} rhs={:?}", lhs, rhs);
        }

        /// Lean 4: `theorem right_distrib (a b c : S) : (b ⊕ c) ⊗ a = (b ⊗ a) ⊕ (c ⊗ a)`
        ///
        /// Uses approximate equality for the same f64-rounding reason.
        #[test]
        fn right_distributivity(a in tropical_semiring(), b in tropical_semiring(), c in tropical_semiring()) {
            let lhs = b.add(&c).mul(&a);
            let rhs = b.mul(&a).add(&c.mul(&a));
            prop_assert!(tropical_approx_eq(&lhs, &rhs),
                "right_distrib: lhs={:?} rhs={:?}", lhs, rhs);
        }
    }
}

// =========================================================================
// Additional axioms that only apply to specific semirings
// =========================================================================

proptest! {
    /// Lean 4: `theorem add_idem (a : BoolSemiring) : a ⊕ a = a`
    ///
    /// The Boolean semiring is idempotent under addition (OR is idempotent).
    #[test]
    fn boolean_additive_idempotency(a in boolean_semiring()) {
        prop_assert_eq!(a.add(&a), a);
    }

    /// Lean 4: `theorem star_unfold (a : S) : a* = 1 ⊕ a ⊗ a*`
    ///
    /// The Kleene star unfolding axiom for BooleanSemiring.
    #[test]
    fn boolean_star_unfold(a in boolean_semiring()) {
        let star_a = a.star();
        let rhs = BooleanSemiring::one().add(&a.mul(&star_a));
        prop_assert_eq!(star_a, rhs);
    }

    /// Lean 4: `theorem star_unfold (a : S) : a* = 1 ⊕ a ⊗ a*`
    ///
    /// The Kleene star unfolding axiom for TropicalSemiring (finite weights ≥ 0).
    #[test]
    fn tropical_star_unfold_nonneg(x in 0.0f64..1e6) {
        let a = TropicalSemiring::new(x);
        let star_a = a.star();
        let rhs = TropicalSemiring::one().add(&a.mul(&star_a));
        prop_assert_eq!(star_a, rhs);
    }

    /// Lean 4: `theorem zero_is_zero : (0 : S).is_zero = true`
    #[test]
    fn goldilocks_zero_is_zero(_dummy in 0u8..1) {
        prop_assert!(GoldilocksField::zero().is_zero());
    }

    /// Lean 4: `theorem one_is_one : (1 : S).is_one = true`
    #[test]
    fn goldilocks_one_is_one(_dummy in 0u8..1) {
        prop_assert!(GoldilocksField::one().is_one());
    }

    /// Lean 4: `theorem mul_comm (a b : GoldilocksField) : a ⊗ b = b ⊗ a`
    ///
    /// GoldilocksField is a commutative ring, so multiplication is also commutative.
    #[test]
    fn goldilocks_multiplicative_commutativity(a in goldilocks_field(), b in goldilocks_field()) {
        prop_assert_eq!(a.mul(&b), b.mul(&a));
    }
}
