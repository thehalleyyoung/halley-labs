//! Benchmarks for the DY+SMT constraint encoding engine.
//!
//! Measures encoding time for Dolev-Yao adversary models of varying protocol
//! depth, bitvector cipher-suite constraints, and property encoding.

use std::collections::BTreeSet;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use negsyn_encode::{
    AdversaryBudget, AdversaryEncoder, BvEncoder, DYEncoder, DyEncodeResult, EncoderConfig,
    PropertyEncoder, PropertyKind, SmtConstraint, SmtExpr, SmtFormula, SmtLib2Writer,
    UnrollingEngine, WriterConfig,
};
use negsyn_types::{
    CipherSuiteRegistry, EncodingConfig, HandshakePhase, NegotiationLTS, NegotiationState,
    ProtocolVersion, StateId,
};

/// Build a synthetic `NegotiationLTS` with `n_states` states arranged in a
/// linear chain, simulating unrolled protocol steps.
fn make_synthetic_lts(n_states: usize) -> NegotiationLTS {
    let mut lts = NegotiationLTS::new();

    let mut prev_id = None;
    for i in 0..n_states {
        let mut state = NegotiationState::initial();
        state.version = ProtocolVersion::tls12();
        state.offered_suites = (0..32).map(|j| 0x0030 + j).collect();

        let id = lts.add_state(state);
        if i == 0 {
            lts.mark_initial(id);
        }
        if let Some(prev) = prev_id {
            lts.add_transition(prev, id, format!("step_{}", i));
        }
        prev_id = Some(id);
    }

    lts
}

/// Benchmark: DY adversary encoding at varying unrolling depths.
fn bench_dy_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("dy_encoding");

    for &depth in &[4u32, 8, 16] {
        let n_states = depth as usize * 4;
        let lts = make_synthetic_lts(n_states);
        let config = EncoderConfig::default()
            .with_depth(depth)
            .with_budget(depth);

        group.bench_with_input(
            BenchmarkId::new("depth", depth),
            &depth,
            |b, _| {
                b.iter(|| {
                    let encoder = DYEncoder::new(config.clone());
                    let _result = encoder.encode_lts(black_box(&lts));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: bitvector encoding of cipher-suite selection constraints.
fn bench_bitvector_constraints(c: &mut Criterion) {
    c.bench_function("bitvector_constraints", |b| {
        b.iter(|| {
            let mut formula = SmtFormula::new(8, 4);

            // Encode cipher suite membership constraints as bitvectors
            for suite_id in 0..64u16 {
                let var = SmtExpr::var(format!("cs_{}", suite_id));
                let bound = SmtExpr::bv_lit(suite_id as u64, 16);
                let constraint = SmtExpr::eq(var, bound);
                formula.add_constraint(SmtConstraint::new(
                    constraint,
                    negsyn_encode::ConstraintOrigin::CipherSelection,
                    format!("cs_bound_{}", suite_id),
                ));
            }

            // Add exclusion constraints (no two selected simultaneously)
            for i in 0..16u16 {
                let a = SmtExpr::var(format!("selected_{}", i));
                let b = SmtExpr::var(format!("selected_{}", i + 1));
                let not_both = SmtExpr::not(SmtExpr::and(vec![a, b]));
                formula.add_constraint(SmtConstraint::new(
                    not_both,
                    negsyn_encode::ConstraintOrigin::CipherSelection,
                    format!("excl_{}_{}", i, i + 1),
                ));
            }

            black_box(formula.constraint_count());
        });
    });
}

/// Benchmark: encoding of downgrade-freedom and secrecy properties.
fn bench_property_encoding(c: &mut Criterion) {
    let lts = make_synthetic_lts(32);
    let properties = vec![
        PropertyKind::DowngradeFreedom,
        PropertyKind::VersionIntegrity,
        PropertyKind::CipherSuiteSecrecy,
    ];

    let config = EncoderConfig::default()
        .with_depth(8)
        .with_budget(4)
        .with_properties(properties);

    c.bench_function("property_encoding", |b| {
        b.iter(|| {
            let encoder = DYEncoder::new(config.clone());
            let _result = encoder.encode_lts(black_box(&lts));
        });
    });
}

criterion_group!(
    benches,
    bench_dy_encoding,
    bench_bitvector_constraints,
    bench_property_encoding,
);
criterion_main!(benches);
