//! Benchmarks for the protocol-aware program slicer.
//!
//! Measures slicing performance on synthetic program dependency graphs of
//! increasing size, plus call-graph and dominator-tree construction.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use negsyn_slicer::{
    BasicBlock, CFG, CallGraph, CallGraphBuilder, CallSite, DominatorTree, Function,
    Instruction, InstructionId, Module, ProgramDependenceGraph, ProtocolAwareSlicer,
    SliceCriterion, SlicerConfig, Value,
};

/// Build a synthetic `Module` with a chain of `n_funcs` functions, each
/// containing a basic block with a handful of instructions.
fn make_synthetic_module(n_funcs: usize) -> Module {
    let mut module = Module::new("bench_module");

    for i in 0..n_funcs {
        let name = format!("func_{}", i);
        let mut func = Function::new(&name);
        let mut block = BasicBlock::new("entry");

        // Each block has ~4 instructions to give the slicer something to work with
        for j in 0..4 {
            block.add_instruction(Instruction::Assign {
                dst: Value::local(format!("v{}_{}", i, j)),
                src: Value::constant(j as i64),
            });
        }

        // Add a call to the next function in the chain to create cross-function deps
        if i + 1 < n_funcs {
            block.add_instruction(Instruction::Call {
                target: format!("func_{}", i + 1),
                args: vec![Value::local(format!("v{}_0", i))],
                dst: Some(Value::local(format!("ret_{}", i))),
            });
        }

        func.add_block(block);
        module.add_function(func);
    }

    module
}

/// Build a `CallGraph` from a synthetic module.
fn build_callgraph(module: &Module) -> CallGraph {
    CallGraphBuilder::new(module).build()
}

/// Benchmark: backward slicing on graphs of varying size.
fn bench_slicer_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("slicer_scaling");

    for &n in &[64, 256, 1024, 4096] {
        let module = make_synthetic_module(n);
        let call_graph = build_callgraph(&module);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            let criterion = SliceCriterion::cipher_negotiation();
            b.iter(|| {
                let mut slicer = ProtocolAwareSlicer::new(
                    black_box(&module),
                    black_box(&call_graph),
                );
                let _ = slicer.prepare();
                let _ = slicer.slice(black_box(&criterion));
            });
        });
    }

    group.finish();
}

/// Benchmark: call-graph construction including indirect-call resolution.
fn bench_callgraph_construction(c: &mut Criterion) {
    let module = make_synthetic_module(512);

    c.bench_function("callgraph_construction", |b| {
        b.iter(|| {
            let _cg = CallGraphBuilder::new(black_box(&module)).build();
        });
    });
}

/// Benchmark: dominator tree computation on CFGs of varying complexity.
fn bench_cfg_dominator_tree(c: &mut Criterion) {
    let module = make_synthetic_module(128);

    c.bench_function("cfg_dominator_tree", |b| {
        b.iter(|| {
            for func in module.functions() {
                let cfg = CFG::from_function(func);
                let _ = DominatorTree::compute(black_box(&cfg));
            }
        });
    });
}

criterion_group!(
    benches,
    bench_slicer_scaling,
    bench_callgraph_construction,
    bench_cfg_dominator_tree,
);
criterion_main!(benches);
