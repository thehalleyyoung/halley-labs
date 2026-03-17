//! Benchmarks for the protocol-aware merge operator.
//!
//! Measures merge performance across varying cipher suite counts and compares
//! the protocol-aware merge against a naive baseline.

use std::collections::BTreeSet;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use negsyn_merge::{
    MergeConfig, MergeOperator, SecurityLattice, standard_cipher_suites,
};
use negsyn_types::{
    CipherSuite, CipherSuiteRegistry, HandshakePhase, MergeConfig as TypesMergeConfig,
    NegotiationState, ProtocolVersion, SymbolicState, SymbolicValue, PathConstraint,
};

/// Build a synthetic `NegotiationState` offering `n` cipher suites drawn from
/// the standard registry.
fn make_negotiation_state(n: usize) -> NegotiationState {
    let all = CipherSuiteRegistry::all();
    let mut state = NegotiationState::initial();
    state.offered_suites = all.iter().take(n).map(|cs| cs.id()).collect();
    state.phase = HandshakePhase::ClientHello;
    state.version = ProtocolVersion::tls12();
    state
}

/// Build a pair of synthetic `SymbolicState`s that differ in offered cipher
/// suites — the typical merge scenario during symbolic execution.
fn make_symbolic_pair(n: usize) -> (SymbolicState, SymbolicState) {
    let all = CipherSuiteRegistry::all();
    let ids_a: BTreeSet<u16> = all.iter().take(n).map(|cs| cs.id()).collect();
    let ids_b: BTreeSet<u16> = all.iter().skip(n / 4).take(n).map(|cs| cs.id()).collect();

    let mut left = SymbolicState::new(0);
    left.negotiation = make_negotiation_state(n);
    left.negotiation.offered_suites = ids_a;
    left.path_constraint = PathConstraint::new();

    let mut right = SymbolicState::new(1);
    right.negotiation = make_negotiation_state(n);
    right.negotiation.offered_suites = ids_b;
    right.path_constraint = PathConstraint::new();
    right.path_constraint.add(SymbolicValue::bool_const(true));

    (left, right)
}

/// Benchmark: merge states with varying cipher suite counts.
fn bench_merge_cipher_suites(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge_cipher_suites");

    for &n in &[16, 32, 64, 128] {
        let (left, right) = make_symbolic_pair(n);
        let config = MergeConfig {
            max_states: 8192,
            ..Default::default()
        };
        let mut operator = MergeOperator::new(config);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let _ = operator.merge_states(black_box(&left), black_box(&right));
            });
        });
    }

    group.finish();
}

/// Benchmark: protocol-aware merge vs naive (union) merge baseline.
fn bench_merge_vs_naive(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge_vs_naive");
    let n = 64;
    let (left, right) = make_symbolic_pair(n);

    // Protocol-aware merge with lattice checking
    let config = MergeConfig {
        max_states: 8192,
        ..Default::default()
    };
    let lattice = SecurityLattice::from_standard_registry();
    let mut operator = MergeOperator::new(config.clone()).with_lattice(lattice);

    group.bench_function("protocol_aware", |b| {
        b.iter(|| {
            let _ = operator.merge_states(black_box(&left), black_box(&right));
        });
    });

    // Naive merge — no lattice, minimal config
    let mut naive_operator = MergeOperator::new(MergeConfig {
        max_states: 8192,
        ..Default::default()
    });

    group.bench_function("naive_merge", |b| {
        b.iter(|| {
            let _ = naive_operator.merge_states(black_box(&left), black_box(&right));
        });
    });

    group.finish();
}

/// Benchmark: merge realistic TLS negotiation states that include cipher
/// suites, extensions, and version information.
fn bench_merge_tls_negotiation(c: &mut Criterion) {
    let mut state_a = NegotiationState::initial();
    state_a.phase = HandshakePhase::ServerHello;
    state_a.version = ProtocolVersion::tls13();
    state_a.offered_suites = CipherSuiteRegistry::all()
        .iter()
        .filter(|cs| cs.has_forward_secrecy())
        .map(|cs| cs.id())
        .collect();

    let mut state_b = NegotiationState::initial();
    state_b.phase = HandshakePhase::ServerHello;
    state_b.version = ProtocolVersion::tls12();
    state_b.offered_suites = CipherSuiteRegistry::all()
        .iter()
        .take(32)
        .map(|cs| cs.id())
        .collect();

    let mut left = SymbolicState::new(10);
    left.negotiation = state_a;

    let mut right = SymbolicState::new(11);
    right.negotiation = state_b;

    let config = MergeConfig {
        max_states: 4096,
        ..Default::default()
    };
    let mut operator = MergeOperator::new(config);

    c.bench_function("merge_tls_negotiation", |b| {
        b.iter(|| {
            let _ = operator.merge_states(black_box(&left), black_box(&right));
        });
    });
}

criterion_group!(
    benches,
    bench_merge_cipher_suites,
    bench_merge_vs_naive,
    bench_merge_tls_negotiation,
);
criterion_main!(benches);
