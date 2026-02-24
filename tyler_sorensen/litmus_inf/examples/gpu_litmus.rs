//! GPU Litmus Test Example for LITMUS∞
//!
//! Demonstrates GPU memory model verification with hierarchical scopes,
//! WebGPU litmus tests, and multi-model comparison.

use litmus_infinity::checker::execution::*;
use litmus_infinity::checker::memory_model::*;
use litmus_infinity::checker::litmus::*;
use litmus_infinity::checker::verifier::*;
use litmus_infinity::checker::webgpu::*;
use litmus_infinity::algebraic::types::*;
use litmus_infinity::algebraic::group::PermutationGroup;
use litmus_infinity::algebraic::symmetry::FullSymmetryGroup;
use litmus_infinity::algebraic::wreath::*;
use litmus_infinity::frontend::visualizer::*;
use std::collections::HashMap;

/// Build a Store Buffering (SB) litmus test with GPU orderings.
fn build_gpu_sb() -> LitmusTest {
    let mut test = LitmusTest::new("GPU-SB");
    test.set_initial(0x100, 0);
    test.set_initial(0x200, 0);

    let mut t0 = Thread::new(0);
    t0.store(0x100, 1, Ordering::ReleaseGPU);
    t0.load(0, 0x200, Ordering::AcquireGPU);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.store(0x200, 1, Ordering::ReleaseGPU);
    t1.load(0, 0x100, Ordering::AcquireGPU);
    test.add_thread(t1);

    test.expect(
        Outcome::new().with_reg(0, 0, 0).with_reg(1, 0, 0),
        LitmusOutcome::Forbidden,
    );

    test
}

/// Build a Message Passing (MP) litmus test with GPU scoped fences.
fn build_gpu_mp_fenced() -> LitmusTest {
    let mut test = LitmusTest::new("GPU-MP-fenced");
    test.set_initial(0x100, 0);
    test.set_initial(0x200, 0);

    let mut t0 = Thread::new(0);
    t0.store(0x100, 1, Ordering::Relaxed);
    t0.fence(Ordering::ReleaseGPU, litmus_infinity::checker::litmus::Scope::GPU);
    t0.store(0x200, 1, Ordering::Relaxed);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.load(0, 0x200, Ordering::Relaxed);
    t1.fence(Ordering::AcquireGPU, litmus_infinity::checker::litmus::Scope::GPU);
    t1.load(1, 0x100, Ordering::Relaxed);
    test.add_thread(t1);

    // r0=1, r1=0 should be forbidden with proper fencing
    test.expect(
        Outcome::new().with_reg(1, 0, 1).with_reg(1, 1, 0),
        LitmusOutcome::Forbidden,
    );

    test
}

/// Build IRIW with GPU scoped reads.
fn build_gpu_iriw() -> LitmusTest {
    let mut test = LitmusTest::new("GPU-IRIW");
    test.set_initial(0x100, 0);
    test.set_initial(0x200, 0);

    // Writer threads
    let mut t0 = Thread::new(0);
    t0.store(0x100, 1, Ordering::ReleaseSystem);
    test.add_thread(t0);

    let mut t1 = Thread::new(1);
    t1.store(0x200, 1, Ordering::ReleaseSystem);
    test.add_thread(t1);

    // Reader threads
    let mut t2 = Thread::new(2);
    t2.load(0, 0x100, Ordering::AcquireSystem);
    t2.load(1, 0x200, Ordering::AcquireSystem);
    test.add_thread(t2);

    let mut t3 = Thread::new(3);
    t3.load(0, 0x200, Ordering::AcquireSystem);
    t3.load(1, 0x100, Ordering::AcquireSystem);
    test.add_thread(t3);

    test
}

/// Demonstrate WebGPU litmus tests.
fn demo_webgpu_tests() {
    println!("=== WebGPU Litmus Tests ===\n");

    let mp = WebGPULitmusTest::message_passing();
    let sb = WebGPULitmusTest::store_buffering();
    let coh = WebGPULitmusTest::workgroup_coherence();

    let mp_litmus = mp.to_litmus_test();
    let sb_litmus = sb.to_litmus_test();
    let coh_litmus = coh.to_litmus_test();

    println!("  Message Passing: {} threads, {} instructions",
        mp_litmus.thread_count(), mp_litmus.total_instructions());
    println!("  Store Buffering: {} threads, {} instructions",
        sb_litmus.thread_count(), sb_litmus.total_instructions());
    println!("  Workgroup Coherence: {} threads, {} instructions",
        coh_litmus.thread_count(), coh_litmus.total_instructions());
}

/// Verify a litmus test against multiple memory models.
fn demo_multi_model_verification(test: &LitmusTest) {
    println!("\n=== Multi-Model Verification: {} ===\n", test.name);

    let models = vec![
        BuiltinModel::SC,
        BuiltinModel::TSO,
        BuiltinModel::ARM,
        BuiltinModel::PTX,
    ];

    for m in &models {
        let model = m.build();
        let mut verifier = Verifier::new(model);
        let result = verifier.verify_litmus(test);
        let stats = verifier.stats();

        println!("  {:8} | executions: {:>4} | consistent: {:>4}",
            m.name(),
            stats.executions_checked,
            stats.consistent_found,
        );
    }
}

/// Demonstrate GPU hierarchical symmetry.
fn demo_gpu_hierarchy() {
    println!("\n=== GPU Hierarchical Symmetry ===\n");

    let hier = GpuHierarchicalSymmetry::new(2, 2, 4, PermutationGroup::symmetric(2), PermutationGroup::symmetric(2), PermutationGroup::symmetric(4));
    println!("  Total threads: {}", hier.total_threads());
    println!("  Total symmetry order: {}", hier.total_order());
    println!("  {}", hier.summary());

    let factors = hier.level_factors();
    for (level, factor) in &factors {
        println!("  {:?}: factor = {:.1}", level, factor);
    }

    // Demonstrate thread ID decomposition
    println!("\n  Thread ID decomposition:");
    for i in 0..hier.total_threads().min(8) {
        let (cta, warp, thread) = hier.decompose_thread_id(i);
        println!("    Global {} -> CTA={}, Warp={}, Thread={}", i, cta, warp, thread);
    }
}

/// Demonstrate execution graph visualization.
fn demo_visualization(test: &LitmusTest) {
    println!("\n=== Execution Visualization: {} ===\n", test.name);

    let execs = test.enumerate_executions();
    println!("  Total executions: {}", execs.len());

    if let Some((graph, reg_state, mem_state)) = execs.first() {
        let vis = ExecutionVisualizer::new(
            VisualizationConfig::new().with_title(&test.name)
        );
        let ascii = vis.to_ascii(graph);
        println!("{}", ascii);
    }
}


fn demo_webgpu_model() {
    println!("\n=== WebGPU Memory Model ===\n");

    let model = WebGPUModel::new();
    let ax_model = model.axiomatic_model();
    println!("  Model: {}", ax_model.name);
    println!("  Base relations: {}", ax_model.base_relations.len());
    println!("  Derived relations: {}", ax_model.derived_relations.len());
    println!("  Constraints: {}", ax_model.constraints.len());
    println!("  Valid: {}", ax_model.validate().is_ok());

    // Model differences
    let diffs = ModelDifference::compare_gpu_models();
    println!("\n  GPU model comparisons:");
    for diff in &diffs {
        println!("    {}", diff.summary());
    }
}

/// Demonstrate wreath product for GPU.
fn demo_wreath_product() {
    println!("\n=== Wreath Product (GPU Thread Hierarchy) ===\n");

    let warp_sym = PermutationGroup::symmetric(4);   // 4 threads per warp
    let cta_sym = PermutationGroup::symmetric(2);     // 2 warps per CTA

    let wreath = WreathProduct::new(&warp_sym, &cta_sym);
    println!("  Warp symmetry order: {}", warp_sym.order());
    println!("  CTA symmetry order: {}", cta_sym.order());
    println!("  Wreath product order: {}", wreath.order());

    println!("\n  Block structure:");
    for i in 0..8 {
        println!("    Element {} -> block={}, position={}",
            i, wreath.block_of(i), wreath.position_in_block(i));
    }
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║       LITMUS∞ — GPU Litmus Test Verification Demo       ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // 1. WebGPU litmus tests
    demo_webgpu_tests();

    // 2. GPU-scoped litmus tests
    let sb = build_gpu_sb();
    let mp = build_gpu_mp_fenced();
    let iriw = build_gpu_iriw();

    // 3. Multi-model verification
    demo_multi_model_verification(&sb);
    demo_multi_model_verification(&mp);

    // 4. GPU hierarchical symmetry
    demo_gpu_hierarchy();

    // 5. Visualization
    demo_visualization(&sb);

    // 6. WebGPU model
    demo_webgpu_model();

    // 9. Wreath product
    demo_wreath_product();

    println!("\n✅ GPU litmus test demo completed successfully.");
}
