use criterion::{criterion_group, criterion_main, Criterion};
use litmus_infinity::checker::litmus::*;
use litmus_infinity::checker::memory_model::*;
use litmus_infinity::checker::verifier::*;
use litmus_infinity::algebraic::symmetry::{LitmusTest as AlgLitmusTest, MemoryOp, Opcode};
use litmus_infinity::algebraic::compress::StateSpaceCompressor;

fn build_checker_sb() -> litmus_infinity::checker::litmus::LitmusTest {
    let mut test = litmus_infinity::checker::litmus::LitmusTest::new("SB");
    test.set_initial(0x100, 0);
    test.set_initial(0x200, 0);
    let mut t0 = Thread::new(0);
    t0.store(0x100, 1, Ordering::Relaxed);
    t0.load(0, 0x200, Ordering::Relaxed);
    test.add_thread(t0);
    let mut t1 = Thread::new(1);
    t1.store(0x200, 1, Ordering::Relaxed);
    t1.load(0, 0x100, Ordering::Relaxed);
    test.add_thread(t1);
    test.expect(
        Outcome::new().with_reg(0, 0, 0).with_reg(1, 0, 0),
        LitmusOutcome::Forbidden,
    );
    test
}

fn build_alg_sb() -> AlgLitmusTest {
    let mut t = AlgLitmusTest::new("SB", 2, 2, 2);
    t.threads[0].push(MemoryOp { thread_id: 0, op_index: 0, opcode: Opcode::Store, address: Some(0), value: Some(1), depends_on: vec![] });
    t.threads[0].push(MemoryOp { thread_id: 0, op_index: 1, opcode: Opcode::Load,  address: Some(1), value: None,    depends_on: vec![] });
    t.threads[1].push(MemoryOp { thread_id: 1, op_index: 0, opcode: Opcode::Store, address: Some(1), value: Some(1), depends_on: vec![] });
    t.threads[1].push(MemoryOp { thread_id: 1, op_index: 1, opcode: Opcode::Load,  address: Some(0), value: None,    depends_on: vec![] });
    t
}

fn bench_verification(c: &mut Criterion) {
    let test = build_checker_sb();
    let model = BuiltinModel::SC.build();

    c.bench_function("verify_sb_sc", |b| {
        b.iter(|| {
            let mut verifier = Verifier::new(model.clone());
            verifier.verify_litmus(&test)
        })
    });

    let tso_model = BuiltinModel::TSO.build();
    c.bench_function("verify_sb_tso", |b| {
        b.iter(|| {
            let mut verifier = Verifier::new(tso_model.clone());
            verifier.verify_litmus(&test)
        })
    });
}

fn bench_compression(c: &mut Criterion) {
    let alg_test = build_alg_sb();

    c.bench_function("compress_sb", |b| {
        b.iter(|| {
            let compressor = StateSpaceCompressor::new(alg_test.clone());
            compressor.compress()
        })
    });
}

fn bench_model_build(c: &mut Criterion) {
    c.bench_function("build_model_sc", |b| {
        b.iter(|| BuiltinModel::SC.build())
    });

    c.bench_function("build_model_arm", |b| {
        b.iter(|| BuiltinModel::ARM.build())
    });
}

criterion_group!(benches, bench_verification, bench_compression, bench_model_build);
criterion_main!(benches);
