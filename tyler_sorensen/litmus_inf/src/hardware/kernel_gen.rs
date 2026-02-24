//! Generate GPU kernel source code from litmus tests.
//!
//! Provides backend-specific code generators that emit compilable GPU kernel
//! source (CUDA `.cu`, OpenCL `.cl`, Vulkan GLSL `.comp`, Metal `.metal`)
//! from `LitmusTest` definitions.

use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

use serde::{Deserialize, Serialize};

use crate::checker::litmus::{
    Instruction, LitmusOutcome, LitmusTest, Ordering, Outcome, Thread,
};
use crate::checker::execution::{Address, Value};

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Common interface for all GPU kernel generators.
pub trait KernelGenerator {
    /// Generate the complete source file for a litmus test.
    fn generate_source(&self, test: &LitmusTest) -> String;

    /// Generate just the test kernel body (no harness).
    fn generate_kernel_body(&self, test: &LitmusTest) -> String;

    /// Generate the host-side harness that launches the kernel.
    fn generate_harness(&self, test: &LitmusTest) -> String;

    /// Generate the result-checking code.
    fn generate_result_check(&self, test: &LitmusTest) -> String;
}

// ---------------------------------------------------------------------------
// Helper: ordering to backend-specific atomic qualifier
// ---------------------------------------------------------------------------

fn cuda_memory_order(ord: &Ordering) -> &'static str {
    match ord {
        Ordering::Relaxed => "cuda::memory_order_relaxed",
        Ordering::Acquire | Ordering::AcquireCTA | Ordering::AcquireGPU
        | Ordering::AcquireSystem => "cuda::memory_order_acquire",
        Ordering::Release | Ordering::ReleaseCTA | Ordering::ReleaseGPU
        | Ordering::ReleaseSystem => "cuda::memory_order_release",
        Ordering::AcqRel => "cuda::memory_order_acq_rel",
        Ordering::SeqCst => "cuda::memory_order_seq_cst",
    }
}

fn opencl_memory_order(ord: &Ordering) -> &'static str {
    match ord {
        Ordering::Relaxed => "memory_order_relaxed",
        Ordering::Acquire | Ordering::AcquireCTA | Ordering::AcquireGPU
        | Ordering::AcquireSystem => "memory_order_acquire",
        Ordering::Release | Ordering::ReleaseCTA | Ordering::ReleaseGPU
        | Ordering::ReleaseSystem => "memory_order_release",
        Ordering::AcqRel => "memory_order_acq_rel",
        Ordering::SeqCst => "memory_order_seq_cst",
    }
}

fn cuda_scope(ord: &Ordering) -> &'static str {
    match ord {
        Ordering::AcquireCTA | Ordering::ReleaseCTA => "cuda::thread_scope_block",
        Ordering::AcquireGPU | Ordering::ReleaseGPU => "cuda::thread_scope_device",
        Ordering::AcquireSystem | Ordering::ReleaseSystem => "cuda::thread_scope_system",
        _ => "cuda::thread_scope_device",
    }
}

fn opencl_scope(ord: &Ordering) -> &'static str {
    match ord {
        Ordering::AcquireCTA | Ordering::ReleaseCTA => "memory_scope_work_group",
        Ordering::AcquireGPU | Ordering::ReleaseGPU => "memory_scope_device",
        Ordering::AcquireSystem | Ordering::ReleaseSystem => "memory_scope_all_svm_devices",
        _ => "memory_scope_device",
    }
}

fn vulkan_memory_semantics(ord: &Ordering) -> &'static str {
    match ord {
        Ordering::Relaxed => "gl_SemanticsRelaxed",
        Ordering::Acquire | Ordering::AcquireCTA | Ordering::AcquireGPU
        | Ordering::AcquireSystem => "gl_SemanticsAcquire",
        Ordering::Release | Ordering::ReleaseCTA | Ordering::ReleaseGPU
        | Ordering::ReleaseSystem => "gl_SemanticsRelease",
        Ordering::AcqRel => "gl_SemanticsAcquireRelease",
        Ordering::SeqCst => "gl_SemanticsAcquireRelease | gl_SemanticsMakeAvailable | gl_SemanticsMakeVisible",
    }
}

fn metal_memory_order(ord: &Ordering) -> &'static str {
    match ord {
        Ordering::Relaxed => "memory_order_relaxed",
        Ordering::Acquire | Ordering::AcquireCTA | Ordering::AcquireGPU
        | Ordering::AcquireSystem => "memory_order_acquire",
        Ordering::Release | Ordering::ReleaseCTA | Ordering::ReleaseGPU
        | Ordering::ReleaseSystem => "memory_order_release",
        Ordering::AcqRel => "memory_order_acq_rel",
        Ordering::SeqCst => "memory_order_relaxed", // Metal lacks SeqCst
    }
}

/// Collect unique addresses used by a test.
fn collect_addresses(test: &LitmusTest) -> Vec<Address> {
    let mut addrs: Vec<Address> = test.initial_state.keys().copied().collect();
    for thread in &test.threads {
        for instr in &thread.instructions {
            let addr = match instr {
                Instruction::Load { addr, .. } => Some(*addr),
                Instruction::Store { addr, .. } => Some(*addr),
                Instruction::RMW { addr, .. } => Some(*addr),
                _ => None,
            };
            if let Some(a) = addr {
                if !addrs.contains(&a) {
                    addrs.push(a);
                }
            }
        }
    }
    addrs.sort();
    addrs
}

/// Maximum register id used by a thread.
fn max_reg(thread: &Thread) -> usize {
    let mut m = 0;
    for instr in &thread.instructions {
        match instr {
            Instruction::Load { reg, .. } | Instruction::RMW { reg, .. } => {
                if *reg > m {
                    m = *reg;
                }
            }
            _ => {}
        }
    }
    m
}

// ===========================================================================
// CUDA
// ===========================================================================

/// Generates CUDA `.cu` kernel source from litmus tests.
#[derive(Debug, Clone)]
pub struct CudaKernelGenerator {
    pub threads_per_block: u32,
    pub blocks: u32,
    pub iterations: u64,
}

impl CudaKernelGenerator {
    pub fn new(threads_per_block: u32, blocks: u32, iterations: u64) -> Self {
        Self {
            threads_per_block,
            blocks,
            iterations,
        }
    }

    fn emit_instruction(&self, instr: &Instruction, buf: &mut String) {
        match instr {
            Instruction::Load { reg, addr, ordering } => {
                writeln!(
                    buf,
                    "        r{reg} = atomicLoad(&mem[{addr}], {order}, {scope});",
                    reg = reg,
                    addr = addr,
                    order = cuda_memory_order(ordering),
                    scope = cuda_scope(ordering),
                )
                .unwrap();
            }
            Instruction::Store { addr, value, ordering } => {
                writeln!(
                    buf,
                    "        atomicStore(&mem[{addr}], {val}, {order}, {scope});",
                    addr = addr,
                    val = value,
                    order = cuda_memory_order(ordering),
                    scope = cuda_scope(ordering),
                )
                .unwrap();
            }
            Instruction::Fence { ordering, .. } => {
                writeln!(
                    buf,
                    "        __threadfence(); // fence {order}",
                    order = cuda_memory_order(ordering),
                )
                .unwrap();
            }
            Instruction::RMW { reg, addr, value, ordering } => {
                writeln!(
                    buf,
                    "        r{reg} = atomicExch(&mem[{addr}], {val}); // {order}",
                    reg = reg,
                    addr = addr,
                    val = value,
                    order = cuda_memory_order(ordering),
                )
                .unwrap();
            }
            _ => {
                writeln!(buf, "        // unsupported instruction").unwrap();
            }
        }
    }
}

impl KernelGenerator for CudaKernelGenerator {
    fn generate_source(&self, test: &LitmusTest) -> String {
        let mut src = String::with_capacity(4096);

        // Header
        writeln!(src, "// Auto-generated CUDA litmus test: {}", test.name).unwrap();
        writeln!(src, "// Threads per block: {}, Blocks: {}, Iterations: {}",
                 self.threads_per_block, self.blocks, self.iterations).unwrap();
        writeln!(src, "#include <cstdio>").unwrap();
        writeln!(src, "#include <cstdint>").unwrap();
        writeln!(src, "#include <cuda.h>").unwrap();
        writeln!(src).unwrap();

        let addrs = collect_addresses(test);
        writeln!(src, "#define NUM_LOCATIONS {}", addrs.len()).unwrap();
        writeln!(src, "#define ITERATIONS {}ULL", self.iterations).unwrap();
        writeln!(src).unwrap();

        // Atomic helpers
        writeln!(src, "__device__ int atomicLoad(volatile int* addr, int, int) {{").unwrap();
        writeln!(src, "    return *addr;").unwrap();
        writeln!(src, "}}").unwrap();
        writeln!(src).unwrap();
        writeln!(src, "__device__ void atomicStore(volatile int* addr, int val, int, int) {{").unwrap();
        writeln!(src, "    *addr = val;").unwrap();
        writeln!(src, "}}").unwrap();
        writeln!(src).unwrap();

        // Kernel
        src.push_str(&self.generate_kernel_body(test));
        src.push('\n');

        // Harness
        src.push_str(&self.generate_harness(test));
        src.push('\n');

        // Result check & main
        src.push_str(&self.generate_result_check(test));

        src
    }

    fn generate_kernel_body(&self, test: &LitmusTest) -> String {
        let mut buf = String::with_capacity(2048);
        let addrs = collect_addresses(test);

        writeln!(buf, "__global__ void litmus_test(").unwrap();
        writeln!(buf, "    volatile int* mem,").unwrap();
        writeln!(buf, "    int* results,").unwrap();
        writeln!(buf, "    int* outcome_counts,").unwrap();
        writeln!(buf, "    int num_iterations)").unwrap();
        writeln!(buf, "{{").unwrap();
        writeln!(buf, "    int tid = blockIdx.x * blockDim.x + threadIdx.x;").unwrap();
        writeln!(buf, "    int num_threads = blockDim.x * gridDim.x;").unwrap();
        writeln!(buf, "    // STRESS_INSERTION_POINT").unwrap();
        writeln!(buf).unwrap();

        // Declare registers
        for thread in &test.threads {
            let mr = max_reg(thread);
            for r in 0..=mr {
                writeln!(buf, "    int t{}_r{} = 0;", thread.id, r).unwrap();
            }
        }
        writeln!(buf).unwrap();

        writeln!(buf, "    for (int iter = 0; iter < num_iterations; iter++) {{").unwrap();

        // Initialise memory
        writeln!(buf, "        if (tid == 0) {{").unwrap();
        for &addr in &addrs {
            let val = test.initial_state.get(&addr).copied().unwrap_or(0);
            writeln!(buf, "            mem[{}] = {};", addr, val).unwrap();
        }
        writeln!(buf, "        }}").unwrap();
        writeln!(buf, "        __syncthreads();").unwrap();
        writeln!(buf).unwrap();

        // Thread bodies
        for thread in &test.threads {
            writeln!(buf, "        // Thread {}", thread.id).unwrap();
            writeln!(buf, "        if (tid % {} == {}) {{", test.threads.len(), thread.id).unwrap();
            for instr in &thread.instructions {
                self.emit_instruction(instr, &mut buf);
            }
            writeln!(buf, "        }}").unwrap();
        }

        writeln!(buf, "        __syncthreads();").unwrap();
        writeln!(buf).unwrap();

        // Record results
        writeln!(buf, "        if (tid == 0) {{").unwrap();
        let mut result_idx = 0;
        for thread in &test.threads {
            let mr = max_reg(thread);
            for r in 0..=mr {
                writeln!(
                    buf,
                    "            results[iter * {} + {}] = t{}_r{};",
                    total_result_slots(test),
                    result_idx,
                    thread.id,
                    r,
                ).unwrap();
                result_idx += 1;
            }
        }
        writeln!(buf, "        }}").unwrap();
        writeln!(buf, "        __syncthreads();").unwrap();
        writeln!(buf, "    }}").unwrap();
        writeln!(buf, "}}").unwrap();

        buf
    }

    fn generate_harness(&self, test: &LitmusTest) -> String {
        let mut buf = String::with_capacity(2048);
        let addrs = collect_addresses(test);
        let slots = total_result_slots(test);

        writeln!(buf, "void run_test(int iterations) {{").unwrap();
        writeln!(buf, "    int *d_mem, *d_results, *d_counts;").unwrap();
        writeln!(buf, "    cudaMalloc(&d_mem, sizeof(int) * {});", addrs.len()).unwrap();
        writeln!(buf, "    cudaMalloc(&d_results, sizeof(int) * iterations * {});", slots).unwrap();
        writeln!(buf, "    cudaMalloc(&d_counts, sizeof(int) * 1024);").unwrap();
        writeln!(buf, "    cudaMemset(d_counts, 0, sizeof(int) * 1024);").unwrap();
        writeln!(buf).unwrap();
        writeln!(buf, "    litmus_test<<<{}, {}>>>(d_mem, d_results, d_counts, iterations);",
                 self.blocks, self.threads_per_block).unwrap();
        writeln!(buf, "    cudaDeviceSynchronize();").unwrap();
        writeln!(buf).unwrap();
        writeln!(buf, "    int* h_results = new int[iterations * {}];", slots).unwrap();
        writeln!(buf, "    cudaMemcpy(h_results, d_results, sizeof(int) * iterations * {}, cudaMemcpyDeviceToHost);", slots).unwrap();
        writeln!(buf).unwrap();

        // Tally outcomes
        writeln!(buf, "    // Tally outcomes").unwrap();
        writeln!(buf, "    struct OutcomeKey {{").unwrap();
        for i in 0..slots {
            writeln!(buf, "        int v{};", i).unwrap();
        }
        writeln!(buf, "    }};").unwrap();
        writeln!(buf).unwrap();
        writeln!(buf, "    // Simple histogram via sorting — real code uses a hash map").unwrap();
        writeln!(buf, "    for (int i = 0; i < iterations; i++) {{").unwrap();
        write!(buf, "        printf(\"OUTCOME ").unwrap();
        let mut first = true;
        let mut idx = 0;
        for thread in &test.threads {
            let mr = max_reg(thread);
            for r in 0..=mr {
                if !first {
                    write!(buf, ",").unwrap();
                }
                write!(buf, "T{}:r{}=%d", thread.id, r).unwrap();
                first = false;
                idx += 1;
            }
        }
        write!(buf, " COUNT 1\\n\"").unwrap();
        for i in 0..idx {
            write!(buf, ", h_results[i * {} + {}]", slots, i).unwrap();
        }
        writeln!(buf, ");").unwrap();
        writeln!(buf, "    }}").unwrap();
        writeln!(buf).unwrap();

        writeln!(buf, "    delete[] h_results;").unwrap();
        writeln!(buf, "    cudaFree(d_mem);").unwrap();
        writeln!(buf, "    cudaFree(d_results);").unwrap();
        writeln!(buf, "    cudaFree(d_counts);").unwrap();
        writeln!(buf, "}}").unwrap();

        buf
    }

    fn generate_result_check(&self, test: &LitmusTest) -> String {
        let mut buf = String::with_capacity(1024);

        writeln!(buf, "int main(int argc, char** argv) {{").unwrap();
        writeln!(buf, "    int iterations = {};", self.iterations).unwrap();
        writeln!(buf, "    if (argc > 1) iterations = atoi(argv[1]);").unwrap();
        writeln!(buf, "    run_test(iterations);").unwrap();
        writeln!(buf, "    return 0;").unwrap();
        writeln!(buf, "}}").unwrap();

        buf
    }
}

// ===========================================================================
// OpenCL
// ===========================================================================

/// Generates OpenCL `.cl` kernel source from litmus tests.
#[derive(Debug, Clone)]
pub struct OpenClKernelGenerator {
    pub work_group_size: u32,
    pub num_groups: u32,
    pub iterations: u64,
}

impl OpenClKernelGenerator {
    pub fn new(work_group_size: u32, num_groups: u32, iterations: u64) -> Self {
        Self {
            work_group_size,
            num_groups,
            iterations,
        }
    }

    fn emit_instruction(&self, instr: &Instruction, buf: &mut String) {
        match instr {
            Instruction::Load { reg, addr, ordering } => {
                writeln!(
                    buf,
                    "        r{reg} = atomic_load_explicit(&mem[{addr}], {order}, {scope});",
                    reg = reg,
                    addr = addr,
                    order = opencl_memory_order(ordering),
                    scope = opencl_scope(ordering),
                )
                .unwrap();
            }
            Instruction::Store { addr, value, ordering } => {
                writeln!(
                    buf,
                    "        atomic_store_explicit(&mem[{addr}], {val}, {order}, {scope});",
                    addr = addr,
                    val = value,
                    order = opencl_memory_order(ordering),
                    scope = opencl_scope(ordering),
                )
                .unwrap();
            }
            Instruction::Fence { ordering, .. } => {
                writeln!(
                    buf,
                    "        atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE, {order}, {scope});",
                    order = opencl_memory_order(ordering),
                    scope = opencl_scope(ordering),
                )
                .unwrap();
            }
            Instruction::RMW { reg, addr, value, ordering } => {
                writeln!(
                    buf,
                    "        r{reg} = atomic_exchange_explicit(&mem[{addr}], {val}, {order}, {scope});",
                    reg = reg,
                    addr = addr,
                    val = value,
                    order = opencl_memory_order(ordering),
                    scope = opencl_scope(ordering),
                )
                .unwrap();
            }
            _ => {
                writeln!(buf, "        // unsupported instruction").unwrap();
            }
        }
    }
}

impl KernelGenerator for OpenClKernelGenerator {
    fn generate_source(&self, test: &LitmusTest) -> String {
        let mut src = String::with_capacity(4096);

        writeln!(src, "// Auto-generated OpenCL litmus test: {}", test.name).unwrap();
        writeln!(src, "// Work-group size: {}, Num groups: {}, Iterations: {}",
                 self.work_group_size, self.num_groups, self.iterations).unwrap();
        writeln!(src, "#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable").unwrap();
        writeln!(src).unwrap();

        src.push_str(&self.generate_kernel_body(test));
        src.push('\n');
        src.push_str(&self.generate_harness(test));
        src.push('\n');
        src.push_str(&self.generate_result_check(test));

        src
    }

    fn generate_kernel_body(&self, test: &LitmusTest) -> String {
        let mut buf = String::with_capacity(2048);
        let addrs = collect_addresses(test);

        writeln!(buf, "__kernel void litmus_test(").unwrap();
        writeln!(buf, "    __global volatile atomic_int* mem,").unwrap();
        writeln!(buf, "    __global int* results,").unwrap();
        writeln!(buf, "    int num_iterations)").unwrap();
        writeln!(buf, "{{").unwrap();
        writeln!(buf, "    int tid = get_global_id(0);").unwrap();
        writeln!(buf, "    // STRESS_INSERTION_POINT").unwrap();
        writeln!(buf).unwrap();

        for thread in &test.threads {
            let mr = max_reg(thread);
            for r in 0..=mr {
                writeln!(buf, "    int t{}_r{} = 0;", thread.id, r).unwrap();
            }
        }
        writeln!(buf).unwrap();

        writeln!(buf, "    for (int iter = 0; iter < num_iterations; iter++) {{").unwrap();
        writeln!(buf, "        if (tid == 0) {{").unwrap();
        for &addr in &addrs {
            let val = test.initial_state.get(&addr).copied().unwrap_or(0);
            writeln!(buf, "            atomic_store_explicit(&mem[{}], {}, memory_order_relaxed, memory_scope_device);", addr, val).unwrap();
        }
        writeln!(buf, "        }}").unwrap();
        writeln!(buf, "        barrier(CLK_GLOBAL_MEM_FENCE);").unwrap();
        writeln!(buf).unwrap();

        for thread in &test.threads {
            writeln!(buf, "        // Thread {}", thread.id).unwrap();
            writeln!(buf, "        if (tid % {} == {}) {{", test.threads.len(), thread.id).unwrap();
            for instr in &thread.instructions {
                self.emit_instruction(instr, &mut buf);
            }
            writeln!(buf, "        }}").unwrap();
        }

        writeln!(buf, "        barrier(CLK_GLOBAL_MEM_FENCE);").unwrap();

        // Record results
        let slots = total_result_slots(test);
        writeln!(buf, "        if (tid == 0) {{").unwrap();
        let mut idx = 0;
        for thread in &test.threads {
            let mr = max_reg(thread);
            for r in 0..=mr {
                writeln!(buf, "            results[iter * {} + {}] = t{}_r{};", slots, idx, thread.id, r).unwrap();
                idx += 1;
            }
        }
        writeln!(buf, "        }}").unwrap();
        writeln!(buf, "        barrier(CLK_GLOBAL_MEM_FENCE);").unwrap();
        writeln!(buf, "    }}").unwrap();
        writeln!(buf, "}}").unwrap();

        buf
    }

    fn generate_harness(&self, test: &LitmusTest) -> String {
        let mut buf = String::with_capacity(1024);
        writeln!(buf, "// Host harness for OpenCL — typically in a separate .c file.").unwrap();
        writeln!(buf, "// Use clCreateBuffer, clEnqueueNDRangeKernel, etc.").unwrap();
        writeln!(buf, "// Work-group size: {}, Num groups: {}",
                 self.work_group_size, self.num_groups).unwrap();
        buf
    }

    fn generate_result_check(&self, test: &LitmusTest) -> String {
        let mut buf = String::with_capacity(512);
        writeln!(buf, "// Result checking performed host-side after readback.").unwrap();
        writeln!(buf, "// See run_hardware_tests.py for the host driver.").unwrap();
        buf
    }
}

// ===========================================================================
// Vulkan
// ===========================================================================

/// Generates Vulkan GLSL compute shaders from litmus tests.
#[derive(Debug, Clone)]
pub struct VulkanShaderGenerator {
    pub local_size_x: u32,
    pub num_workgroups: u32,
    pub iterations: u64,
}

impl VulkanShaderGenerator {
    pub fn new(local_size_x: u32, num_workgroups: u32, iterations: u64) -> Self {
        Self {
            local_size_x,
            num_workgroups,
            iterations,
        }
    }

    fn emit_instruction(&self, instr: &Instruction, buf: &mut String) {
        match instr {
            Instruction::Load { reg, addr, ordering } => {
                writeln!(
                    buf,
                    "        r{reg} = atomicLoad(mem.data[{addr}], {sem}, gl_ScopeDevice, 0);",
                    reg = reg,
                    addr = addr,
                    sem = vulkan_memory_semantics(ordering),
                )
                .unwrap();
            }
            Instruction::Store { addr, value, ordering } => {
                writeln!(
                    buf,
                    "        atomicStore(mem.data[{addr}], {val}, {sem}, gl_ScopeDevice, 0);",
                    addr = addr,
                    val = value,
                    sem = vulkan_memory_semantics(ordering),
                )
                .unwrap();
            }
            Instruction::Fence { ordering, .. } => {
                writeln!(
                    buf,
                    "        memoryBarrier(); // {sem}",
                    sem = vulkan_memory_semantics(ordering),
                )
                .unwrap();
            }
            Instruction::RMW { reg, addr, value, ordering } => {
                writeln!(
                    buf,
                    "        r{reg} = atomicExchange(mem.data[{addr}], {val}); // {sem}",
                    reg = reg,
                    addr = addr,
                    val = value,
                    sem = vulkan_memory_semantics(ordering),
                )
                .unwrap();
            }
            _ => {
                writeln!(buf, "        // unsupported instruction").unwrap();
            }
        }
    }
}

impl KernelGenerator for VulkanShaderGenerator {
    fn generate_source(&self, test: &LitmusTest) -> String {
        let mut src = String::with_capacity(4096);

        writeln!(src, "// Auto-generated Vulkan compute shader: {}", test.name).unwrap();
        writeln!(src, "#version 450").unwrap();
        writeln!(src, "#extension GL_KHR_memory_scope_semantics : enable").unwrap();
        writeln!(src).unwrap();

        src.push_str(&self.generate_kernel_body(test));
        src.push('\n');
        src.push_str(&self.generate_harness(test));
        src.push('\n');
        src.push_str(&self.generate_result_check(test));

        src
    }

    fn generate_kernel_body(&self, test: &LitmusTest) -> String {
        let mut buf = String::with_capacity(2048);
        let addrs = collect_addresses(test);
        let slots = total_result_slots(test);

        writeln!(buf, "layout(local_size_x = {}) in;", self.local_size_x).unwrap();
        writeln!(buf).unwrap();
        writeln!(buf, "layout(set = 0, binding = 0) buffer MemBuf {{").unwrap();
        writeln!(buf, "    int data[];").unwrap();
        writeln!(buf, "}} mem;").unwrap();
        writeln!(buf).unwrap();
        writeln!(buf, "layout(set = 0, binding = 1) buffer ResultBuf {{").unwrap();
        writeln!(buf, "    int data[];").unwrap();
        writeln!(buf, "}} results;").unwrap();
        writeln!(buf).unwrap();
        writeln!(buf, "layout(push_constant) uniform PushConstants {{").unwrap();
        writeln!(buf, "    int num_iterations;").unwrap();
        writeln!(buf, "}} pc;").unwrap();
        writeln!(buf).unwrap();

        // Shared stress memory
        writeln!(buf, "shared int shared_stress[1024];").unwrap();
        writeln!(buf).unwrap();

        writeln!(buf, "void main() {{").unwrap();
        writeln!(buf, "    uint tid = gl_GlobalInvocationID.x;").unwrap();
        writeln!(buf, "    // STRESS_INSERTION_POINT").unwrap();
        writeln!(buf).unwrap();

        for thread in &test.threads {
            let mr = max_reg(thread);
            for r in 0..=mr {
                writeln!(buf, "    int t{}_r{} = 0;", thread.id, r).unwrap();
            }
        }
        writeln!(buf).unwrap();

        writeln!(buf, "    for (int iter = 0; iter < pc.num_iterations; iter++) {{").unwrap();
        writeln!(buf, "        if (tid == 0u) {{").unwrap();
        for &addr in &addrs {
            let val = test.initial_state.get(&addr).copied().unwrap_or(0);
            writeln!(buf, "            mem.data[{}] = {};", addr, val).unwrap();
        }
        writeln!(buf, "        }}").unwrap();
        writeln!(buf, "        barrier();").unwrap();
        writeln!(buf, "        memoryBarrierBuffer();").unwrap();
        writeln!(buf).unwrap();

        for thread in &test.threads {
            writeln!(buf, "        if (tid % {}u == {}u) {{", test.threads.len(), thread.id).unwrap();
            for instr in &thread.instructions {
                self.emit_instruction(instr, &mut buf);
            }
            writeln!(buf, "        }}").unwrap();
        }

        writeln!(buf, "        barrier();").unwrap();
        writeln!(buf, "        memoryBarrierBuffer();").unwrap();

        writeln!(buf, "        if (tid == 0u) {{").unwrap();
        let mut idx = 0;
        for thread in &test.threads {
            let mr = max_reg(thread);
            for r in 0..=mr {
                writeln!(buf, "            results.data[iter * {} + {}] = t{}_r{};", slots, idx, thread.id, r).unwrap();
                idx += 1;
            }
        }
        writeln!(buf, "        }}").unwrap();
        writeln!(buf, "        barrier();").unwrap();
        writeln!(buf, "    }}").unwrap();
        writeln!(buf, "}}").unwrap();

        buf
    }

    fn generate_harness(&self, test: &LitmusTest) -> String {
        let mut buf = String::new();
        writeln!(buf, "// Vulkan host harness — use VkComputePipeline with").unwrap();
        writeln!(buf, "// vkCmdDispatch({}, 1, 1)", self.num_workgroups).unwrap();
        buf
    }

    fn generate_result_check(&self, test: &LitmusTest) -> String {
        let mut buf = String::new();
        writeln!(buf, "// Result checking performed host-side after buffer readback.").unwrap();
        buf
    }
}

// ===========================================================================
// Metal
// ===========================================================================

/// Generates Metal compute shader source from litmus tests.
#[derive(Debug, Clone)]
pub struct MetalKernelGenerator {
    pub threads_per_group: u32,
    pub threadgroups: u32,
    pub iterations: u64,
}

impl MetalKernelGenerator {
    pub fn new(threads_per_group: u32, threadgroups: u32, iterations: u64) -> Self {
        Self {
            threads_per_group,
            threadgroups,
            iterations,
        }
    }

    fn emit_instruction(&self, instr: &Instruction, buf: &mut String) {
        match instr {
            Instruction::Load { reg, addr, ordering } => {
                writeln!(
                    buf,
                    "        r{reg} = atomic_load_explicit(&mem[{addr}], {order});",
                    reg = reg,
                    addr = addr,
                    order = metal_memory_order(ordering),
                )
                .unwrap();
            }
            Instruction::Store { addr, value, ordering } => {
                writeln!(
                    buf,
                    "        atomic_store_explicit(&mem[{addr}], {val}, {order});",
                    addr = addr,
                    val = value,
                    order = metal_memory_order(ordering),
                )
                .unwrap();
            }
            Instruction::Fence { ordering, .. } => {
                writeln!(
                    buf,
                    "        threadgroup_barrier(mem_flags::mem_device); // {order}",
                    order = metal_memory_order(ordering),
                )
                .unwrap();
            }
            Instruction::RMW { reg, addr, value, ordering } => {
                writeln!(
                    buf,
                    "        r{reg} = atomic_exchange_explicit(&mem[{addr}], {val}, {order});",
                    reg = reg,
                    addr = addr,
                    val = value,
                    order = metal_memory_order(ordering),
                )
                .unwrap();
            }
            _ => {
                writeln!(buf, "        // unsupported instruction").unwrap();
            }
        }
    }
}

impl KernelGenerator for MetalKernelGenerator {
    fn generate_source(&self, test: &LitmusTest) -> String {
        let mut src = String::with_capacity(4096);

        writeln!(src, "// Auto-generated Metal compute shader: {}", test.name).unwrap();
        writeln!(src, "#include <metal_stdlib>").unwrap();
        writeln!(src, "#include <metal_atomic>").unwrap();
        writeln!(src, "using namespace metal;").unwrap();
        writeln!(src).unwrap();

        src.push_str(&self.generate_kernel_body(test));
        src.push('\n');
        src.push_str(&self.generate_harness(test));
        src.push('\n');
        src.push_str(&self.generate_result_check(test));

        src
    }

    fn generate_kernel_body(&self, test: &LitmusTest) -> String {
        let mut buf = String::with_capacity(2048);
        let addrs = collect_addresses(test);
        let slots = total_result_slots(test);

        writeln!(buf, "kernel void litmus_test(").unwrap();
        writeln!(buf, "    device atomic_int* mem [[buffer(0)]],").unwrap();
        writeln!(buf, "    device int* results [[buffer(1)]],").unwrap();
        writeln!(buf, "    constant int& num_iterations [[buffer(2)]],").unwrap();
        writeln!(buf, "    uint tid [[thread_position_in_grid]])").unwrap();
        writeln!(buf, "{{").unwrap();
        writeln!(buf, "    // STRESS_INSERTION_POINT").unwrap();
        writeln!(buf).unwrap();

        for thread in &test.threads {
            let mr = max_reg(thread);
            for r in 0..=mr {
                writeln!(buf, "    int t{}_r{} = 0;", thread.id, r).unwrap();
            }
        }
        writeln!(buf).unwrap();

        writeln!(buf, "    for (int iter = 0; iter < num_iterations; iter++) {{").unwrap();
        writeln!(buf, "        if (tid == 0u) {{").unwrap();
        for &addr in &addrs {
            let val = test.initial_state.get(&addr).copied().unwrap_or(0);
            writeln!(buf, "            atomic_store_explicit(&mem[{}], {}, memory_order_relaxed);", addr, val).unwrap();
        }
        writeln!(buf, "        }}").unwrap();
        writeln!(buf, "        threadgroup_barrier(mem_flags::mem_device);").unwrap();
        writeln!(buf).unwrap();

        for thread in &test.threads {
            writeln!(buf, "        if (tid % {}u == {}u) {{", test.threads.len(), thread.id).unwrap();
            for instr in &thread.instructions {
                self.emit_instruction(instr, &mut buf);
            }
            writeln!(buf, "        }}").unwrap();
        }

        writeln!(buf, "        threadgroup_barrier(mem_flags::mem_device);").unwrap();

        writeln!(buf, "        if (tid == 0u) {{").unwrap();
        let mut idx = 0;
        for thread in &test.threads {
            let mr = max_reg(thread);
            for r in 0..=mr {
                writeln!(buf, "            results[iter * {} + {}] = t{}_r{};", slots, idx, thread.id, r).unwrap();
                idx += 1;
            }
        }
        writeln!(buf, "        }}").unwrap();
        writeln!(buf, "        threadgroup_barrier(mem_flags::mem_device);").unwrap();
        writeln!(buf, "    }}").unwrap();
        writeln!(buf, "}}").unwrap();

        buf
    }

    fn generate_harness(&self, test: &LitmusTest) -> String {
        let mut buf = String::new();
        writeln!(buf, "// Metal host harness — use MTLComputeCommandEncoder with").unwrap();
        writeln!(buf, "// dispatchThreadgroups:({}, 1, 1) threadsPerThreadgroup:({}, 1, 1)",
                 self.threadgroups, self.threads_per_group).unwrap();
        buf
    }

    fn generate_result_check(&self, test: &LitmusTest) -> String {
        let mut buf = String::new();
        writeln!(buf, "// Result checking performed host-side after buffer readback.").unwrap();
        buf
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Total number of per-iteration result slots for a test.
fn total_result_slots(test: &LitmusTest) -> usize {
    test.threads.iter().map(|t| max_reg(t) + 1).sum()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checker::litmus::{LitmusOutcome, Scope};

    fn make_mp_test() -> LitmusTest {
        let t0 = Thread {
            id: 0,
            instructions: vec![
                Instruction::Store {
                    addr: 0,
                    value: 1,
                    ordering: Ordering::Relaxed,
                },
                Instruction::Store {
                    addr: 1,
                    value: 1,
                    ordering: Ordering::Release,
                },
            ],
        };
        let t1 = Thread {
            id: 1,
            instructions: vec![
                Instruction::Load {
                    reg: 0,
                    addr: 1,
                    ordering: Ordering::Acquire,
                },
                Instruction::Load {
                    reg: 1,
                    addr: 0,
                    ordering: Ordering::Relaxed,
                },
            ],
        };

        let mut initial = HashMap::new();
        initial.insert(0u64, 0u64);
        initial.insert(1u64, 0u64);

        LitmusTest {
            name: "MP".into(),
            threads: vec![t0, t1],
            initial_state: initial,
            expected_outcomes: vec![
                (
                    Outcome {
                        registers: {
                            let mut m = HashMap::new();
                            m.insert((1, 0), 1);
                            m.insert((1, 1), 0);
                            m
                        },
                        memory: HashMap::new(),
                    },
                    LitmusOutcome::Forbidden,
                ),
            ],
        }
    }

    fn make_sb_test() -> LitmusTest {
        let t0 = Thread {
            id: 0,
            instructions: vec![
                Instruction::Store {
                    addr: 0,
                    value: 1,
                    ordering: Ordering::Relaxed,
                },
                Instruction::Load {
                    reg: 0,
                    addr: 1,
                    ordering: Ordering::Relaxed,
                },
            ],
        };
        let t1 = Thread {
            id: 1,
            instructions: vec![
                Instruction::Store {
                    addr: 1,
                    value: 1,
                    ordering: Ordering::Relaxed,
                },
                Instruction::Load {
                    reg: 0,
                    addr: 0,
                    ordering: Ordering::Relaxed,
                },
            ],
        };

        let mut initial = HashMap::new();
        initial.insert(0u64, 0u64);
        initial.insert(1u64, 0u64);

        LitmusTest {
            name: "SB".into(),
            threads: vec![t0, t1],
            initial_state: initial,
            expected_outcomes: vec![],
        }
    }

    // -- CUDA tests --

    #[test]
    fn test_cuda_generate_source_mp() {
        let gen = CudaKernelGenerator::new(256, 1, 10000);
        let src = gen.generate_source(&make_mp_test());
        assert!(src.contains("__global__"));
        assert!(src.contains("litmus_test"));
        assert!(src.contains("mem[0]"));
        assert!(src.contains("mem[1]"));
        assert!(src.contains("cudaMalloc"));
    }

    #[test]
    fn test_cuda_kernel_body_sb() {
        let gen = CudaKernelGenerator::new(256, 1, 1000);
        let body = gen.generate_kernel_body(&make_sb_test());
        assert!(body.contains("Thread 0"));
        assert!(body.contains("Thread 1"));
        assert!(body.contains("__syncthreads"));
    }

    #[test]
    fn test_cuda_result_check() {
        let gen = CudaKernelGenerator::new(256, 1, 1000);
        let rc = gen.generate_result_check(&make_mp_test());
        assert!(rc.contains("main"));
        assert!(rc.contains("run_test"));
    }

    // -- OpenCL tests --

    #[test]
    fn test_opencl_generate_source() {
        let gen = OpenClKernelGenerator::new(256, 1, 10000);
        let src = gen.generate_source(&make_mp_test());
        assert!(src.contains("__kernel"));
        assert!(src.contains("atomic_load_explicit"));
        assert!(src.contains("atomic_store_explicit"));
    }

    #[test]
    fn test_opencl_kernel_body() {
        let gen = OpenClKernelGenerator::new(64, 2, 5000);
        let body = gen.generate_kernel_body(&make_sb_test());
        assert!(body.contains("get_global_id"));
        assert!(body.contains("barrier(CLK_GLOBAL_MEM_FENCE)"));
    }

    // -- Vulkan tests --

    #[test]
    fn test_vulkan_generate_source() {
        let gen = VulkanShaderGenerator::new(256, 1, 10000);
        let src = gen.generate_source(&make_mp_test());
        assert!(src.contains("#version 450"));
        assert!(src.contains("GL_KHR_memory_scope_semantics"));
        assert!(src.contains("gl_GlobalInvocationID"));
    }

    #[test]
    fn test_vulkan_kernel_body() {
        let gen = VulkanShaderGenerator::new(64, 2, 5000);
        let body = gen.generate_kernel_body(&make_sb_test());
        assert!(body.contains("layout(local_size_x"));
        assert!(body.contains("memoryBarrierBuffer"));
    }

    // -- Metal tests --

    #[test]
    fn test_metal_generate_source() {
        let gen = MetalKernelGenerator::new(256, 1, 10000);
        let src = gen.generate_source(&make_mp_test());
        assert!(src.contains("metal_stdlib"));
        assert!(src.contains("kernel void"));
        assert!(src.contains("atomic_load_explicit"));
    }

    #[test]
    fn test_metal_kernel_body() {
        let gen = MetalKernelGenerator::new(64, 2, 5000);
        let body = gen.generate_kernel_body(&make_sb_test());
        assert!(body.contains("thread_position_in_grid"));
        assert!(body.contains("threadgroup_barrier"));
    }

    // -- General tests --

    #[test]
    fn test_collect_addresses() {
        let test = make_mp_test();
        let addrs = collect_addresses(&test);
        assert!(addrs.contains(&0));
        assert!(addrs.contains(&1));
    }

    #[test]
    fn test_max_reg() {
        let t = Thread {
            id: 0,
            instructions: vec![
                Instruction::Load { reg: 0, addr: 0, ordering: Ordering::Relaxed },
                Instruction::Load { reg: 3, addr: 1, ordering: Ordering::Relaxed },
            ],
        };
        assert_eq!(max_reg(&t), 3);
    }

    #[test]
    fn test_total_result_slots() {
        let test = make_mp_test();
        let slots = total_result_slots(&test);
        // Thread 0 has no loads (max_reg = 0 from store), Thread 1 has reg 0 and 1
        assert!(slots >= 2);
    }

    #[test]
    fn test_kernel_generator_trait_object() {
        let gen: Box<dyn KernelGenerator> = Box::new(CudaKernelGenerator::new(64, 1, 100));
        let test = make_mp_test();
        let src = gen.generate_source(&test);
        assert!(!src.is_empty());
    }

    #[test]
    fn test_fence_instruction_cuda() {
        let gen = CudaKernelGenerator::new(256, 1, 100);
        let test = LitmusTest {
            name: "fence_test".into(),
            threads: vec![Thread {
                id: 0,
                instructions: vec![
                    Instruction::Store { addr: 0, value: 1, ordering: Ordering::Relaxed },
                    Instruction::Fence { ordering: Ordering::SeqCst, scope: Scope::GPU },
                    Instruction::Store { addr: 1, value: 1, ordering: Ordering::Relaxed },
                ],
            }],
            initial_state: {
                let mut m = HashMap::new();
                m.insert(0u64, 0u64);
                m.insert(1u64, 0u64);
                m
            },
            expected_outcomes: vec![],
        };
        let src = gen.generate_source(&test);
        assert!(src.contains("__threadfence"));
    }

    #[test]
    fn test_rmw_instruction_opencl() {
        let gen = OpenClKernelGenerator::new(256, 1, 100);
        let test = LitmusTest {
            name: "rmw_test".into(),
            threads: vec![Thread {
                id: 0,
                instructions: vec![
                    Instruction::RMW { reg: 0, addr: 0, value: 1, ordering: Ordering::AcqRel },
                ],
            }],
            initial_state: {
                let mut m = HashMap::new();
                m.insert(0u64, 0u64);
                m
            },
            expected_outcomes: vec![],
        };
        let src = gen.generate_source(&test);
        assert!(src.contains("atomic_exchange_explicit"));
    }

    #[test]
    fn test_cuda_memory_order_mapping() {
        assert_eq!(cuda_memory_order(&Ordering::Relaxed), "cuda::memory_order_relaxed");
        assert_eq!(cuda_memory_order(&Ordering::Acquire), "cuda::memory_order_acquire");
        assert_eq!(cuda_memory_order(&Ordering::Release), "cuda::memory_order_release");
        assert_eq!(cuda_memory_order(&Ordering::SeqCst), "cuda::memory_order_seq_cst");
    }

    #[test]
    fn test_opencl_memory_order_mapping() {
        assert_eq!(opencl_memory_order(&Ordering::Relaxed), "memory_order_relaxed");
        assert_eq!(opencl_memory_order(&Ordering::AcqRel), "memory_order_acq_rel");
    }

    #[test]
    fn test_metal_memory_order_mapping() {
        assert_eq!(metal_memory_order(&Ordering::Acquire), "memory_order_acquire");
        assert_eq!(metal_memory_order(&Ordering::SeqCst), "memory_order_relaxed");
    }

    #[test]
    fn test_vulkan_memory_semantics_mapping() {
        assert_eq!(vulkan_memory_semantics(&Ordering::Relaxed), "gl_SemanticsRelaxed");
        assert_eq!(vulkan_memory_semantics(&Ordering::Acquire), "gl_SemanticsAcquire");
    }
}
