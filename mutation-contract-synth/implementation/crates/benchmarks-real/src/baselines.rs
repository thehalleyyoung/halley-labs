//! Baseline algorithms for comparison.
//!
//! 1. **Random mutation sampling**: randomly selects a subset of error predicates
//!    and conjoins their negations—no lattice ordering, no subsumption pruning.
//! 2. **Spec mining from tests**: simulates Daikon-style dynamic invariant
//!    detection by emitting simple range and equality invariants over the
//!    function signature without any mutation information.

use rand::seq::SliceRandom;
use rand::SeedableRng;
use shared_types::{Contract, ContractClause, Formula, MutantId};

use crate::lattice::DiscriminationLattice;
use crate::methods::MethodSpec;
use crate::{AlgorithmStats, ClauseKind, SynthesizedClause};

/// Baseline 1: Random mutation sampling.
///
/// Picks a random 60% subset of mutants (seeded for reproducibility),
/// conjoins their negated error predicates without any ordering or
/// subsumption check.  This isolates the contribution of the lattice-walk
/// ordering from the raw mutation-specification duality.
pub fn random_mutation_baseline(spec: &MethodSpec) -> (Vec<SynthesizedClause>, AlgorithmStats) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let n = spec.error_predicates.len();
    let sample_size = (n as f64 * 0.6).ceil() as usize;

    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(&mut rng);
    indices.truncate(sample_size);

    // Conjoin negated error predicates in random order (no lattice structure).
    let mut conjuncts: Vec<Formula> = Vec::new();
    let mut mutations_used = 0;

    for &idx in &indices {
        let (_, ep) = &spec.error_predicates[idx];
        let negated = Formula::not(ep.clone());
        // No subsumption check—just blindly add.
        conjuncts.push(negated);
        mutations_used += 1;
    }

    let clauses = if conjuncts.is_empty() {
        vec![]
    } else {
        conjuncts
            .iter()
            .map(|f| SynthesizedClause {
                kind: ClauseKind::Post,
                formula_str: format!("{}", f),
            })
            .collect()
    };

    let stats = AlgorithmStats {
        mutations_used,
        lattice_steps: 0, // no lattice walk
        entailment_checks: 0,
    };

    (clauses, stats)
}

/// Baseline 2: Spec mining from tests (Daikon-style).
///
/// Emits template-based invariants from the function signature without
/// using mutation information at all.  This simulates what a tool like
/// Daikon would produce from a modest test suite: simple range bounds
/// and non-null/non-zero checks on parameters and return value.
pub fn spec_mining_baseline(spec: &MethodSpec) -> (Vec<SynthesizedClause>, AlgorithmStats) {
    let mut clauses: Vec<SynthesizedClause> = Vec::new();

    // Parse variable names from signature for template instantiation.
    let vars = extract_param_names(&spec.signature);

    // Template 1: ret >= 0 (non-negativity) — common Daikon invariant.
    clauses.push(SynthesizedClause {
        kind: ClauseKind::Post,
        formula_str: "ret >= 0".into(),
    });

    // Template 2: for each pair of params, emit ret >= param and ret <= param.
    // Daikon generates these from observed test traces.
    for v in &vars {
        // ret >= v (Daikon's "y >= x" family)
        clauses.push(SynthesizedClause {
            kind: ClauseKind::Post,
            formula_str: format!("ret >= {}", v),
        });
        // ret <= v (sometimes correct, sometimes not)
        clauses.push(SynthesizedClause {
            kind: ClauseKind::Post,
            formula_str: format!("ret <= {}", v),
        });
    }

    // Template 3: for functions with 2 params, emit ret == a || ret == b.
    if vars.len() == 2 {
        clauses.push(SynthesizedClause {
            kind: ClauseKind::Post,
            formula_str: format!("ret == {} || ret == {}", vars[0], vars[1]),
        });
    }

    // Template 4: ret != 0 (common Daikon invariant, often wrong).
    clauses.push(SynthesizedClause {
        kind: ClauseKind::Post,
        formula_str: "ret != 0".into(),
    });

    let stats = AlgorithmStats {
        mutations_used: 0,
        lattice_steps: 0,
        entailment_checks: 0,
    };

    (clauses, stats)
}

/// Extract parameter names from a signature like "foo(a: int, b: int) -> int".
fn extract_param_names(sig: &str) -> Vec<String> {
    let mut names = Vec::new();
    if let Some(start) = sig.find('(') {
        if let Some(end) = sig.find(')') {
            let params = &sig[start + 1..end];
            for param in params.split(',') {
                let param = param.trim();
                if let Some(colon) = param.find(':') {
                    let name = param[..colon].trim().to_string();
                    if !name.is_empty() {
                        names.push(name);
                    }
                }
            }
        }
    }
    names
}
