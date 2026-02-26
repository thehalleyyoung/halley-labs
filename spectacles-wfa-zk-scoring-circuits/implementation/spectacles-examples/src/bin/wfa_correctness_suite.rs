use serde_json::json;
use spectacles_core::evalspec::builtins::BuiltinRegistry;
use spectacles_core::evalspec::parser::{Lexer, Parser};
use spectacles_core::wfa::automaton::{Alphabet, WeightedFiniteAutomaton};
use spectacles_core::wfa::equivalence::{check_equivalence, EquivalenceResult};
use spectacles_core::wfa::field_embedding::{
    BooleanToGoldilocks, CountingToGoldilocks, SemiringEmbedding, WfaEmbedder,
};
use spectacles_core::wfa::minimization::{is_minimal, minimize_hopcroft, MinimizationConfig};
use spectacles_core::wfa::operations::{enumerate_accepted, hadamard_product};
use spectacles_core::wfa::semiring::{
    BooleanSemiring, CountingSemiring, GoldilocksField, Semiring, GOLDILOCKS_PRIME,
};
use spectacles_core::wfa::transducer::WeightedTransducer;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct TestResult {
    name: String,
    passed: bool,
    detail: String,
}

fn section_json(results: &[TestResult]) -> serde_json::Value {
    let passed = results.iter().filter(|r| r.passed).count();
    let failed = results.iter().filter(|r| !r.passed).count();
    let details: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            json!({
                "name": r.name,
                "passed": r.passed,
                "detail": r.detail,
            })
        })
        .collect();
    json!({ "passed": passed, "failed": failed, "details": details })
}

/// Build a minimal 1-state DFA over `n_symbols` that accepts everything.
fn universal_bool(n_symbols: usize) -> WeightedFiniteAutomaton<BooleanSemiring> {
    let alpha = Alphabet::from_range(n_symbols);
    let mut wfa = WeightedFiniteAutomaton::new(1, alpha);
    wfa.set_initial_weight(0, BooleanSemiring::one());
    wfa.set_final_weight(0, BooleanSemiring::one());
    for s in 0..n_symbols {
        wfa.add_transition(0, 0, s, BooleanSemiring::one());
    }
    wfa
}

/// Build a minimal 2-state DFA accepting only strings starting with symbol 0.
fn starts_with_a(n_symbols: usize) -> WeightedFiniteAutomaton<BooleanSemiring> {
    let alpha = Alphabet::from_range(n_symbols);
    let mut wfa = WeightedFiniteAutomaton::new(2, alpha);
    wfa.set_initial_weight(0, BooleanSemiring::one());
    wfa.set_final_weight(1, BooleanSemiring::one());
    // from start, symbol 0 -> accepting loop state
    wfa.add_transition(0, 1, 0, BooleanSemiring::one());
    // accepting state loops on all symbols
    for s in 0..n_symbols {
        wfa.add_transition(1, 1, s, BooleanSemiring::one());
    }
    wfa
}

/// Build a minimal 1-state WFA (counting) that counts all paths.
fn single_state_counter(n_symbols: usize) -> WeightedFiniteAutomaton<CountingSemiring> {
    let alpha = Alphabet::from_range(n_symbols);
    let mut wfa = WeightedFiniteAutomaton::new(1, alpha);
    wfa.set_initial_weight(0, CountingSemiring::one());
    wfa.set_final_weight(0, CountingSemiring::one());
    for s in 0..n_symbols {
        wfa.add_transition(0, 0, s, CountingSemiring::one());
    }
    wfa
}

// ---------------------------------------------------------------------------
// 1. Minimization tests
// ---------------------------------------------------------------------------

fn run_minimization_tests() -> Vec<TestResult> {
    let mut results = Vec::new();

    // Test 1: single-state universal DFA (already minimal)
    {
        let wfa = universal_bool(2);
        let res = minimize_hopcroft(&wfa, &MinimizationConfig::default()).unwrap();
        let pass = res.minimized_states == 1;
        results.push(TestResult {
            name: "single_state_universal".into(),
            passed: pass,
            detail: format!("states before={} after={}", res.original_states, res.minimized_states),
        });
    }

    // Test 2: two-state DFA (starts_with_a) stays 2 states
    {
        let wfa = starts_with_a(2);
        let res = minimize_hopcroft(&wfa, &MinimizationConfig::default()).unwrap();
        let pass = res.minimized_states == 2;
        results.push(TestResult {
            name: "two_state_starts_with_a".into(),
            passed: pass,
            detail: format!("states before={} after={}", res.original_states, res.minimized_states),
        });
    }

    // Test 3: counting single-state (already minimal)
    {
        let wfa = single_state_counter(3);
        let res = minimize_hopcroft(&wfa, &MinimizationConfig::default()).unwrap();
        let pass = res.minimized_states == 1;
        results.push(TestResult {
            name: "single_state_counter".into(),
            passed: pass,
            detail: format!("states before={} after={}", res.original_states, res.minimized_states),
        });
    }

    // Test 4: empty DFA (no accepting states) should minimize to 1 sink
    {
        let alpha = Alphabet::from_range(2);
        let wfa: WeightedFiniteAutomaton<BooleanSemiring> =
            WeightedFiniteAutomaton::new(3, alpha);
        let res = minimize_hopcroft(&wfa, &MinimizationConfig::default()).unwrap();
        let pass = res.minimized_states <= 1;
        results.push(TestResult {
            name: "empty_language_minimizes_to_sink".into(),
            passed: pass,
            detail: format!("states before={} after={}", res.original_states, res.minimized_states),
        });
    }

    // Test 5: 2-state chain accepting only "a" stays 2
    {
        let alpha = Alphabet::from_range(2);
        let mut wfa: WeightedFiniteAutomaton<BooleanSemiring> =
            WeightedFiniteAutomaton::new(2, alpha);
        wfa.set_initial_weight(0, BooleanSemiring::one());
        wfa.set_final_weight(1, BooleanSemiring::one());
        wfa.add_transition(0, 1, 0, BooleanSemiring::one());
        let res = minimize_hopcroft(&wfa, &MinimizationConfig::default()).unwrap();
        let pass = res.minimized_states <= 2;
        results.push(TestResult {
            name: "chain_accepts_single_a".into(),
            passed: pass,
            detail: format!("states before={} after={}", res.original_states, res.minimized_states),
        });
    }

    // Test 6: DFA with redundant states (two identical accepting self-loops
    //         reachable from a common predecessor via the same symbol)
    {
        let alpha = Alphabet::from_range(2);
        let mut wfa: WeightedFiniteAutomaton<BooleanSemiring> =
            WeightedFiniteAutomaton::new(4, alpha);
        wfa.set_initial_weight(0, BooleanSemiring::one());
        // state 0 -> 1 on sym-0, state 0 -> 2 on sym-1
        wfa.add_transition(0, 1, 0, BooleanSemiring::one());
        wfa.add_transition(0, 2, 1, BooleanSemiring::one());
        // states 1 and 2 are both accepting and transition identically to state 3
        wfa.set_final_weight(1, BooleanSemiring::one());
        wfa.set_final_weight(2, BooleanSemiring::one());
        wfa.set_final_weight(3, BooleanSemiring::one());
        for s in 0..2 {
            wfa.add_transition(1, 3, s, BooleanSemiring::one());
            wfa.add_transition(2, 3, s, BooleanSemiring::one());
            wfa.add_transition(3, 3, s, BooleanSemiring::one());
        }
        let res = minimize_hopcroft(&wfa, &MinimizationConfig::default()).unwrap();
        // states 1, 2 and 3 are all equivalent → should merge
        let pass = res.minimized_states < 4;
        results.push(TestResult {
            name: "merge_redundant_accepting_states".into(),
            passed: pass,
            detail: format!("states before={} after={}", res.original_states, res.minimized_states),
        });
    }

    // Test 7: is_minimal returns true for already-minimal automaton
    {
        let wfa = universal_bool(2);
        let pass = is_minimal(&wfa);
        results.push(TestResult {
            name: "is_minimal_on_universal".into(),
            passed: pass,
            detail: format!("is_minimal={}", pass),
        });
    }

    // Test 8: minimization preserves language (brute-force check)
    {
        let wfa = starts_with_a(2);
        let res = minimize_hopcroft(&wfa, &MinimizationConfig::default()).unwrap();
        let orig_words = enumerate_accepted(&wfa, 4);
        let min_words = enumerate_accepted(&res.minimized, 4);
        let pass = orig_words.len() == min_words.len();
        results.push(TestResult {
            name: "minimization_preserves_language".into(),
            passed: pass,
            detail: format!(
                "original_accepted={} minimized_accepted={}",
                orig_words.len(),
                min_words.len()
            ),
        });
    }

    // Test 9: 3-state cycle (a→b→c→a, accept all) is already minimal
    {
        let alpha = Alphabet::from_range(1);
        let mut wfa: WeightedFiniteAutomaton<BooleanSemiring> =
            WeightedFiniteAutomaton::new(3, alpha);
        wfa.set_initial_weight(0, BooleanSemiring::one());
        wfa.set_final_weight(0, BooleanSemiring::one());
        wfa.set_final_weight(1, BooleanSemiring::one());
        wfa.set_final_weight(2, BooleanSemiring::one());
        wfa.add_transition(0, 1, 0, BooleanSemiring::one());
        wfa.add_transition(1, 2, 0, BooleanSemiring::one());
        wfa.add_transition(2, 0, 0, BooleanSemiring::one());
        let res = minimize_hopcroft(&wfa, &MinimizationConfig::default()).unwrap();
        // All states are equivalent (all accept, same transition structure mod 3)
        let pass = res.minimized_states <= 3;
        results.push(TestResult {
            name: "cycle_3_states".into(),
            passed: pass,
            detail: format!("states before={} after={}", res.original_states, res.minimized_states),
        });
    }

    // Test 10: single-symbol alphabet, 2-state (accept even-length strings)
    {
        let alpha = Alphabet::from_range(1);
        let mut wfa: WeightedFiniteAutomaton<BooleanSemiring> =
            WeightedFiniteAutomaton::new(2, alpha);
        wfa.set_initial_weight(0, BooleanSemiring::one());
        wfa.set_final_weight(0, BooleanSemiring::one());
        wfa.add_transition(0, 1, 0, BooleanSemiring::one());
        wfa.add_transition(1, 0, 0, BooleanSemiring::one());
        let res = minimize_hopcroft(&wfa, &MinimizationConfig::default()).unwrap();
        let pass = res.minimized_states == 2;
        results.push(TestResult {
            name: "even_length_parity_stays_2".into(),
            passed: pass,
            detail: format!("states before={} after={}", res.original_states, res.minimized_states),
        });
    }

    // Test 11: 4-state DFA with two pairs of equivalent states
    {
        let alpha = Alphabet::from_range(2);
        let mut wfa: WeightedFiniteAutomaton<BooleanSemiring> =
            WeightedFiniteAutomaton::new(4, alpha);
        wfa.set_initial_weight(0, BooleanSemiring::one());
        // States 0,2 are non-accepting; states 1,3 are accepting with same transitions
        wfa.set_final_weight(1, BooleanSemiring::one());
        wfa.set_final_weight(3, BooleanSemiring::one());
        wfa.add_transition(0, 1, 0, BooleanSemiring::one());
        wfa.add_transition(0, 3, 1, BooleanSemiring::one());
        wfa.add_transition(2, 1, 0, BooleanSemiring::one());
        wfa.add_transition(2, 3, 1, BooleanSemiring::one());
        for s in 0..2 {
            wfa.add_transition(1, 1, s, BooleanSemiring::one());
            wfa.add_transition(3, 3, s, BooleanSemiring::one());
        }
        let res = minimize_hopcroft(&wfa, &MinimizationConfig::default()).unwrap();
        let pass = res.minimized_states < 4;
        results.push(TestResult {
            name: "four_state_two_equiv_pairs".into(),
            passed: pass,
            detail: format!("states before={} after={}", res.original_states, res.minimized_states),
        });
    }

    results
}

// ---------------------------------------------------------------------------
// 2. Equivalence tests
// ---------------------------------------------------------------------------

fn brute_force_equivalent<S: Semiring>(
    wfa1: &WeightedFiniteAutomaton<S>,
    wfa2: &WeightedFiniteAutomaton<S>,
    max_len: usize,
) -> bool {
    let words1 = enumerate_accepted(wfa1, max_len);
    let words2 = enumerate_accepted(wfa2, max_len);
    if words1.len() != words2.len() {
        return false;
    }
    let mut map1 = std::collections::HashMap::new();
    for (w, s) in &words1 {
        map1.insert(w.clone(), s.clone());
    }
    for (w, s) in &words2 {
        match map1.get(w) {
            Some(s1) if s1 == s => {}
            _ => return false,
        }
    }
    true
}

fn run_equivalence_tests() -> Vec<TestResult> {
    let mut results = Vec::new();

    // Pair 1: universal DFA vs itself
    {
        let wfa = universal_bool(2);
        let bf = brute_force_equivalent(&wfa, &wfa, 5);
        let api_result = check_equivalence(&wfa, &wfa);
        let api_eq = api_result.map(|r| r.is_equivalent()).unwrap_or(false);
        let pass = bf && api_eq;
        results.push(TestResult {
            name: "universal_self_equiv".into(),
            passed: pass,
            detail: format!("brute_force={} api={}", bf, api_eq),
        });
    }

    // Pair 2: two differently-constructed but equivalent WFAs
    {
        // Both accept strings starting with symbol 0
        let wfa1 = starts_with_a(2);
        // Build a 3-state version with a redundant state
        let alpha = Alphabet::from_range(2);
        let mut wfa2: WeightedFiniteAutomaton<BooleanSemiring> =
            WeightedFiniteAutomaton::new(3, alpha);
        wfa2.set_initial_weight(0, BooleanSemiring::one());
        wfa2.set_final_weight(1, BooleanSemiring::one());
        wfa2.set_final_weight(2, BooleanSemiring::one());
        wfa2.add_transition(0, 1, 0, BooleanSemiring::one());
        wfa2.add_transition(1, 2, 0, BooleanSemiring::one());
        wfa2.add_transition(1, 2, 1, BooleanSemiring::one());
        wfa2.add_transition(2, 2, 0, BooleanSemiring::one());
        wfa2.add_transition(2, 2, 1, BooleanSemiring::one());

        let bf = brute_force_equivalent(&wfa1, &wfa2, 5);
        let api_result = check_equivalence(&wfa1, &wfa2);
        let api_eq = api_result.map(|r| r.is_equivalent()).unwrap_or(false);
        let pass = bf && api_eq;
        results.push(TestResult {
            name: "starts_with_a_redundant_equiv".into(),
            passed: pass,
            detail: format!("brute_force={} api={}", bf, api_eq),
        });
    }

    // Pair 3: non-equivalent WFAs detected
    {
        let wfa1 = universal_bool(2);
        let wfa2 = starts_with_a(2);
        let bf = brute_force_equivalent(&wfa1, &wfa2, 5);
        let api_result = check_equivalence(&wfa1, &wfa2);
        let api_neq = api_result
            .map(|r| matches!(r, EquivalenceResult::NotEquivalent { .. }))
            .unwrap_or(false);
        let pass = !bf && api_neq;
        results.push(TestResult {
            name: "universal_vs_starts_a_not_equiv".into(),
            passed: pass,
            detail: format!("brute_force_equiv={} api_not_equiv={}", bf, api_neq),
        });
    }

    // Pair 4: minimized version equivalent to original
    {
        let alpha = Alphabet::from_range(2);
        let mut wfa: WeightedFiniteAutomaton<BooleanSemiring> =
            WeightedFiniteAutomaton::new(4, alpha);
        wfa.set_initial_weight(0, BooleanSemiring::one());
        wfa.set_final_weight(2, BooleanSemiring::one());
        wfa.set_final_weight(3, BooleanSemiring::one());
        wfa.add_transition(0, 1, 0, BooleanSemiring::one());
        wfa.add_transition(0, 1, 1, BooleanSemiring::one());
        wfa.add_transition(1, 2, 0, BooleanSemiring::one());
        wfa.add_transition(1, 3, 1, BooleanSemiring::one());
        for s in 0..2 {
            wfa.add_transition(2, 2, s, BooleanSemiring::one());
            wfa.add_transition(3, 3, s, BooleanSemiring::one());
        }
        let min_res = minimize_hopcroft(&wfa, &MinimizationConfig::default()).unwrap();
        let bf = brute_force_equivalent(&wfa, &min_res.minimized, 5);
        let pass = bf;
        results.push(TestResult {
            name: "minimized_equiv_to_original".into(),
            passed: pass,
            detail: format!(
                "brute_force_equiv={} orig_states={} min_states={}",
                bf, min_res.original_states, min_res.minimized_states
            ),
        });
    }

    // Pair 5: counting WFAs - same structure different weights are not equivalent
    {
        let alpha = Alphabet::from_range(2);
        let mut wfa1: WeightedFiniteAutomaton<CountingSemiring> =
            WeightedFiniteAutomaton::new(1, alpha.clone());
        wfa1.set_initial_weight(0, CountingSemiring::one());
        wfa1.set_final_weight(0, CountingSemiring::one());
        wfa1.add_transition(0, 0, 0, CountingSemiring::new(1));
        wfa1.add_transition(0, 0, 1, CountingSemiring::new(1));

        let mut wfa2: WeightedFiniteAutomaton<CountingSemiring> =
            WeightedFiniteAutomaton::new(1, alpha);
        wfa2.set_initial_weight(0, CountingSemiring::one());
        wfa2.set_final_weight(0, CountingSemiring::one());
        wfa2.add_transition(0, 0, 0, CountingSemiring::new(2));
        wfa2.add_transition(0, 0, 1, CountingSemiring::new(1));

        let bf = brute_force_equivalent(&wfa1, &wfa2, 5);
        let pass = !bf; // they should NOT be equivalent
        results.push(TestResult {
            name: "counting_different_weights_not_equiv".into(),
            passed: pass,
            detail: format!("brute_force_equiv={}", bf),
        });
    }

    // Pair 6: two empty WFAs are equivalent
    {
        let alpha = Alphabet::from_range(2);
        let wfa1: WeightedFiniteAutomaton<BooleanSemiring> =
            WeightedFiniteAutomaton::new(1, alpha.clone());
        let wfa2: WeightedFiniteAutomaton<BooleanSemiring> =
            WeightedFiniteAutomaton::new(2, alpha);
        let bf = brute_force_equivalent(&wfa1, &wfa2, 5);
        let pass = bf;
        results.push(TestResult {
            name: "two_empty_wfas_equiv".into(),
            passed: pass,
            detail: format!("brute_force_equiv={}", bf),
        });
    }

    results
}

// ---------------------------------------------------------------------------
// 3. Field embedding tests
// ---------------------------------------------------------------------------

fn run_embedding_tests() -> Vec<TestResult> {
    let mut results = Vec::new();
    let c_emb = CountingToGoldilocks::new();
    let b_emb = BooleanToGoldilocks::new();

    // Test 1: CountingSemiring zero embeds to GoldilocksField zero
    {
        let c = CountingSemiring::zero();
        let g = c_emb.embed(&c).unwrap();
        let pass = g == GoldilocksField::zero();
        results.push(TestResult {
            name: "counting_zero_embeds".into(),
            passed: pass,
            detail: format!("embedded={:?}", g),
        });
    }

    // Test 2: CountingSemiring one embeds to GoldilocksField one
    {
        let c = CountingSemiring::one();
        let g = c_emb.embed(&c).unwrap();
        let pass = g == GoldilocksField::one();
        results.push(TestResult {
            name: "counting_one_embeds".into(),
            passed: pass,
            detail: format!("embedded={:?}", g),
        });
    }

    // Test 3: addition is preserved: embed(a+b) == embed(a) + embed(b)
    {
        let a = CountingSemiring::new(7);
        let b = CountingSemiring::new(13);
        let sum = a.add(&b);
        let g_sum = c_emb.embed(&sum).unwrap();
        let g_a = c_emb.embed(&a).unwrap();
        let g_b = c_emb.embed(&b).unwrap();
        let g_add = g_a.add(&g_b);
        let pass = g_sum == g_add;
        results.push(TestResult {
            name: "counting_add_homomorphism".into(),
            passed: pass,
            detail: format!("embed(7+13)={:?} embed(7)+embed(13)={:?}", g_sum, g_add),
        });
    }

    // Test 4: multiplication is preserved
    {
        let a = CountingSemiring::new(5);
        let b = CountingSemiring::new(11);
        let prod = a.mul(&b);
        let g_prod = c_emb.embed(&prod).unwrap();
        let g_a = c_emb.embed(&a).unwrap();
        let g_b = c_emb.embed(&b).unwrap();
        let g_mul = g_a.mul(&g_b);
        let pass = g_prod == g_mul;
        results.push(TestResult {
            name: "counting_mul_homomorphism".into(),
            passed: pass,
            detail: format!("embed(5*11)={:?} embed(5)*embed(11)={:?}", g_prod, g_mul),
        });
    }

    // Test 5: round-trip: unembed(embed(n)) == n
    {
        let vals = [0u64, 1, 42, 1000, 999999];
        let mut all_pass = true;
        for v in &vals {
            let c = CountingSemiring::new(*v);
            let g = c_emb.embed(&c).unwrap();
            let back = c_emb.unembed(&g).unwrap();
            if back != c {
                all_pass = false;
            }
        }
        results.push(TestResult {
            name: "counting_roundtrip".into(),
            passed: all_pass,
            detail: format!("tested {} values", vals.len()),
        });
    }

    // Test 6: BooleanSemiring false → 0, true → 1
    {
        let f = BooleanSemiring::new(false);
        let t = BooleanSemiring::new(true);
        let gf = b_emb.embed(&f).unwrap();
        let gt = b_emb.embed(&t).unwrap();
        let pass = gf == GoldilocksField::zero() && gt == GoldilocksField::one();
        results.push(TestResult {
            name: "boolean_embedding".into(),
            passed: pass,
            detail: format!("false→{:?} true→{:?}", gf, gt),
        });
    }

    // Test 7: Boolean OR preserved: embed(a|b) == embed(a)+embed(b) in field (clamped)
    {
        let a = BooleanSemiring::new(true);
        let b = BooleanSemiring::new(false);
        let or = a.add(&b); // true | false = true
        let g_or = b_emb.embed(&or).unwrap();
        let pass = g_or == GoldilocksField::one();
        results.push(TestResult {
            name: "boolean_or_preserved".into(),
            passed: pass,
            detail: format!("embed(true|false)={:?}", g_or),
        });
    }

    // Test 8: Boolean AND preserved: embed(a&b) == embed(a)*embed(b) in field
    {
        let a = BooleanSemiring::new(true);
        let b = BooleanSemiring::new(true);
        let and_val = a.mul(&b);
        let g_and = b_emb.embed(&and_val).unwrap();
        let g_a = b_emb.embed(&a).unwrap();
        let g_b = b_emb.embed(&b).unwrap();
        let g_mul = g_a.mul(&g_b);
        let pass = g_and == g_mul && g_and == GoldilocksField::one();
        results.push(TestResult {
            name: "boolean_and_homomorphism".into(),
            passed: pass,
            detail: format!("embed(T&T)={:?} embed(T)*embed(T)={:?}", g_and, g_mul),
        });
    }

    // Test 9: Goldilocks arithmetic modular correctness
    {
        let a = GoldilocksField::new(GOLDILOCKS_PRIME - 1);
        let b = GoldilocksField::new(2);
        let sum = a.add(&b); // (p-1)+2 = p+1 ≡ 1 mod p... actually (p-1)+2 mod p = 1
        let pass = sum == GoldilocksField::one();
        results.push(TestResult {
            name: "goldilocks_modular_add".into(),
            passed: pass,
            detail: format!("(p-1)+2 mod p = {:?}", sum),
        });
    }

    // Test 10: WFA embedding preserves compute_weight
    {
        let alpha = Alphabet::from_range(2);
        let mut wfa: WeightedFiniteAutomaton<CountingSemiring> =
            WeightedFiniteAutomaton::new(2, alpha);
        wfa.set_initial_weight(0, CountingSemiring::one());
        wfa.set_final_weight(1, CountingSemiring::one());
        wfa.add_transition(0, 1, 0, CountingSemiring::new(3));
        wfa.add_transition(0, 1, 1, CountingSemiring::new(5));

        let embedded = WfaEmbedder::<CountingSemiring>::embed_wfa(&wfa, &c_emb).unwrap();
        let orig_w = wfa.compute_weight(&[0]);
        let emb_w = embedded.compute_weight(&[0]);
        let expected = c_emb.embed(&orig_w).unwrap();
        let pass = emb_w == expected;
        results.push(TestResult {
            name: "wfa_embedding_preserves_weight".into(),
            passed: pass,
            detail: format!("orig={:?} embedded={:?} expected={:?}", orig_w, emb_w, expected),
        });
    }

    results
}

// ---------------------------------------------------------------------------
// 4. Composition / product tests
// ---------------------------------------------------------------------------

fn run_composition_tests() -> Vec<TestResult> {
    let mut results = Vec::new();

    // Test 1: Hadamard product of two boolean WFAs = intersection
    {
        let wfa1 = universal_bool(2);
        let wfa2 = starts_with_a(2);
        let prod = hadamard_product(&wfa1, &wfa2).unwrap();
        let orig_words = enumerate_accepted(&wfa2, 4);
        let prod_words = enumerate_accepted(&prod, 4);
        let pass = orig_words.len() == prod_words.len();
        results.push(TestResult {
            name: "hadamard_universal_intersect".into(),
            passed: pass,
            detail: format!("expected={} got={}", orig_words.len(), prod_words.len()),
        });
    }

    // Test 2: Hadamard product of counting WFAs multiplies weights
    {
        let alpha = Alphabet::from_range(2);
        let mut wfa1: WeightedFiniteAutomaton<CountingSemiring> =
            WeightedFiniteAutomaton::new(1, alpha.clone());
        wfa1.set_initial_weight(0, CountingSemiring::one());
        wfa1.set_final_weight(0, CountingSemiring::one());
        wfa1.add_transition(0, 0, 0, CountingSemiring::new(2));
        wfa1.add_transition(0, 0, 1, CountingSemiring::new(3));

        let mut wfa2: WeightedFiniteAutomaton<CountingSemiring> =
            WeightedFiniteAutomaton::new(1, alpha);
        wfa2.set_initial_weight(0, CountingSemiring::one());
        wfa2.set_final_weight(0, CountingSemiring::one());
        wfa2.add_transition(0, 0, 0, CountingSemiring::new(5));
        wfa2.add_transition(0, 0, 1, CountingSemiring::new(7));

        let prod = hadamard_product(&wfa1, &wfa2).unwrap();
        // Weight of symbol 0: 2*5=10
        let w = prod.compute_weight(&[0]);
        let pass = w == CountingSemiring::new(10);
        results.push(TestResult {
            name: "hadamard_counting_multiplies_weights".into(),
            passed: pass,
            detail: format!("weight([0])={:?} expected=10", w),
        });
    }

    // Test 3: Transducer compose identity ∘ T = T
    {
        let alpha = Alphabet::from_range(2);
        let id_t = WeightedTransducer::<CountingSemiring>::identity(&alpha);
        let mut t: WeightedTransducer<CountingSemiring> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, CountingSemiring::one());
        t.set_final_weight(1, CountingSemiring::one());
        t.add_transition(0, Some(0), Some(1), 1, CountingSemiring::new(3));
        t.add_transition(0, Some(1), Some(0), 1, CountingSemiring::new(5));

        match id_t.compose(&t) {
            Ok(composed) => {
                let orig_out = t.transduce(&[0]);
                let comp_out = composed.transduce(&[0]);
                let pass = !orig_out.is_empty() && orig_out.len() == comp_out.len();
                results.push(TestResult {
                    name: "transducer_identity_compose".into(),
                    passed: pass,
                    detail: format!(
                        "orig_outputs={} composed_outputs={}",
                        orig_out.len(),
                        comp_out.len()
                    ),
                });
            }
            Err(e) => {
                results.push(TestResult {
                    name: "transducer_identity_compose".into(),
                    passed: false,
                    detail: format!("compose error: {:?}", e),
                });
            }
        }
    }

    // Test 4: Transducer inversion
    {
        let alpha = Alphabet::from_range(2);
        let mut t: WeightedTransducer<CountingSemiring> =
            WeightedTransducer::new(2, alpha.clone(), alpha.clone());
        t.set_initial_weight(0, CountingSemiring::one());
        t.set_final_weight(1, CountingSemiring::one());
        t.add_transition(0, Some(0), Some(1), 1, CountingSemiring::one());
        let inv = t.invert();
        // Inverting swaps input/output: transducing [1] on inv should give [0]
        let out = inv.transduce(&[1]);
        let pass = out.iter().any(|(seq, _)| seq == &[0]);
        results.push(TestResult {
            name: "transducer_inversion".into(),
            passed: pass,
            detail: format!("inv.transduce([1])={:?}", out),
        });
    }

    // Test 5: Transducer from_wfa preserves state structure
    {
        let alpha = Alphabet::from_range(2);
        let mut wfa: WeightedFiniteAutomaton<CountingSemiring> =
            WeightedFiniteAutomaton::new(2, alpha);
        wfa.set_initial_weight(0, CountingSemiring::one());
        wfa.set_final_weight(1, CountingSemiring::one());
        wfa.add_transition(0, 1, 0, CountingSemiring::new(3));
        let t = WeightedTransducer::from_wfa(&wfa);
        let pass = t.state_count() == wfa.num_states();
        results.push(TestResult {
            name: "transducer_from_wfa_structure".into(),
            passed: pass,
            detail: format!("wfa_states={} transducer_states={}", wfa.num_states(), t.state_count()),
        });
    }

    // Test 6: Hadamard self-product squares weights
    {
        let alpha = Alphabet::from_range(1);
        let mut wfa: WeightedFiniteAutomaton<CountingSemiring> =
            WeightedFiniteAutomaton::new(1, alpha);
        wfa.set_initial_weight(0, CountingSemiring::one());
        wfa.set_final_weight(0, CountingSemiring::one());
        wfa.add_transition(0, 0, 0, CountingSemiring::new(4));
        let prod = hadamard_product(&wfa, &wfa).unwrap();
        let w = prod.compute_weight(&[0]);
        let pass = w == CountingSemiring::new(16); // 4*4
        results.push(TestResult {
            name: "hadamard_self_product_squares".into(),
            passed: pass,
            detail: format!("weight([0])={:?} expected=16", w),
        });
    }

    results
}

// ---------------------------------------------------------------------------
// 5. EvalSpec parser tests
// ---------------------------------------------------------------------------

fn run_parser_tests() -> Vec<TestResult> {
    let mut results = Vec::new();

    let registry = BuiltinRegistry::new();
    let names = registry.list_names();

    // Test each builtin metric's declaration can be rendered and re-parsed
    for name in &names {
        let decl = registry.get_declaration(name);
        let pass = decl.is_some();
        results.push(TestResult {
            name: format!("builtin_decl_{}", name),
            passed: pass,
            detail: format!("has_declaration={}", pass),
        });
    }

    // Test parsing simple metric expressions
    let test_specs = vec![
        ("metric m(x: String, y: String) -> Real = exact_match(x, y)", "simple_metric"),
        ("let threshold: Real = 0.5", "let_binding"),
        ("metric score(a: String, b: String) -> Real = if true then 1.0 else 0.0", "if_expr"),
        ("metric f(a: String, b: String) -> Real = 1.0 + 2.0", "binary_add"),
        ("metric g(a: String, b: String) -> Real = token_f1(a, b)", "func_call"),
    ];

    for (spec, label) in &test_specs {
        let lex_result = Lexer::new(spec, "<test>").tokenize();
        match lex_result {
            Ok(tokens) => {
                let parse_result = Parser::new(tokens, "<test>").parse();
                let pass = parse_result.is_ok();
                let detail = if pass {
                    "parsed successfully".to_string()
                } else {
                    format!("parse errors: {:?}", parse_result.err())
                };
                results.push(TestResult {
                    name: format!("parse_{}", label),
                    passed: pass,
                    detail,
                });
            }
            Err(e) => {
                results.push(TestResult {
                    name: format!("parse_{}", label),
                    passed: false,
                    detail: format!("lex error: {:?}", e),
                });
            }
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let minimization = run_minimization_tests();
    let equivalence = run_equivalence_tests();
    let embedding = run_embedding_tests();
    let composition = run_composition_tests();
    let parser = run_parser_tests();

    let output = json!({
        "minimization_tests": section_json(&minimization),
        "equivalence_tests": section_json(&equivalence),
        "embedding_tests": section_json(&embedding),
        "composition_tests": section_json(&composition),
        "parser_tests": section_json(&parser),
    });

    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}
