//! Code generation from IR — transforms the lowered audio graph into an
//! executable renderer with per-node processing functions and buffer wiring.

use crate::{
    lowering::{LoweredGraph, LoweredNode},
    scheduler::{Schedule, ScheduleStep},
    BufferKind, CgGraph, CodegenConfig, CodegenError, CodegenResult, NodeKind,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Buffer allocation plan (graph coloring)
// ---------------------------------------------------------------------------

/// Describes how audio buffers are allocated and reused across the processing
/// graph via graph coloring of lifetimes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferAllocationPlan {
    /// Total physical buffers allocated.
    pub buffer_count: usize,
    /// Block size per buffer in samples.
    pub block_size: usize,
    /// Channel count per buffer.
    pub channel_count: usize,
    /// Map from logical buffer id (per edge) to physical buffer id.
    pub logical_to_physical: HashMap<usize, usize>,
    /// Total memory in bytes.
    pub total_memory_bytes: usize,
}

impl BufferAllocationPlan {
    /// Build a plan from a schedule.
    pub fn from_schedule(schedule: &Schedule, config: &CodegenConfig) -> Self {
        // Start with identity mapping.
        let mut logical_to_physical: HashMap<usize, usize> = HashMap::new();
        for i in 0..schedule.buffer_count {
            logical_to_physical.insert(i, i);
        }

        // Compute buffer lifetimes: for each logical buffer, find the first step
        // that writes to it and the last step that reads from it.
        let mut first_write: HashMap<usize, usize> = HashMap::new();
        let mut last_read: HashMap<usize, usize> = HashMap::new();

        for (step_idx, step) in schedule.steps.iter().enumerate() {
            if let Some(buf) = step.output_buffer {
                first_write.entry(buf).or_insert(step_idx);
            }
            for &buf in &step.input_buffers {
                last_read
                    .entry(buf)
                    .and_modify(|v| *v = (*v).max(step_idx))
                    .or_insert(step_idx);
            }
        }

        // Build interference graph.
        let n = schedule.buffer_count;
        if n <= 1 {
            return Self {
                buffer_count: n,
                block_size: config.block_size,
                channel_count: config.channel_count,
                logical_to_physical,
                total_memory_bytes: n * config.block_size * config.channel_count * 4,
            };
        }

        let mut interferes = vec![vec![false; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let i_start = first_write.get(&i).copied().unwrap_or(0);
                let i_end = last_read.get(&i).copied().unwrap_or(i_start);
                let j_start = first_write.get(&j).copied().unwrap_or(0);
                let j_end = last_read.get(&j).copied().unwrap_or(j_start);
                if i_start <= j_end && j_start <= i_end {
                    interferes[i][j] = true;
                    interferes[j][i] = true;
                }
            }
        }

        // Greedy graph coloring.
        let mut color_of: Vec<usize> = vec![0; n];
        let mut max_color = 0;
        for buf in 0..n {
            let mut used: HashSet<usize> = HashSet::new();
            for other in 0..n {
                if interferes[buf][other] && other < buf {
                    used.insert(color_of[other]);
                }
            }
            let mut c = 0;
            while used.contains(&c) {
                c += 1;
            }
            color_of[buf] = c;
            if c > max_color {
                max_color = c;
            }
        }

        let physical_count = max_color + 1;
        logical_to_physical.clear();
        for (i, &c) in color_of.iter().enumerate() {
            logical_to_physical.insert(i, c);
        }

        let total_mem = physical_count * config.block_size * config.channel_count * 4;

        Self {
            buffer_count: physical_count,
            block_size: config.block_size,
            channel_count: config.channel_count,
            logical_to_physical,
            total_memory_bytes: total_mem,
        }
    }

    /// Physical buffer id for a logical buffer.
    pub fn physical_buffer(&self, logical: usize) -> usize {
        self.logical_to_physical.get(&logical).copied().unwrap_or(logical)
    }
}

// ---------------------------------------------------------------------------
// NodeCodegen — per-node code generation
// ---------------------------------------------------------------------------

/// Specialized code generation output for a single node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCodegen {
    pub node_id: u64,
    pub kind: NodeKind,
    /// Generated processing function body (pseudo-Rust).
    pub process_body: String,
    /// State fields required by this node.
    pub state_fields: Vec<StateField>,
    /// Initialization code for state fields.
    pub init_body: String,
}

/// A state field required by a processing node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateField {
    pub name: String,
    pub field_type: String,
    pub initial_value: String,
}

impl NodeCodegen {
    /// Generate code for a wavetable oscillator.
    pub fn wavetable_oscillator(node_id: u64, params: &HashMap<String, f64>) -> Self {
        let freq = params.get("frequency").copied().unwrap_or(440.0);
        let amp = params.get("amplitude").copied().unwrap_or(1.0);
        let sr = params.get("sample_rate").copied().unwrap_or(48000.0);

        let state_fields = vec![
            StateField {
                name: "phase".into(),
                field_type: "f64".into(),
                initial_value: "0.0".into(),
            },
            StateField {
                name: "wavetable".into(),
                field_type: "Vec<f64>".into(),
                initial_value: format!("generate_sine_table({})", 2048),
            },
        ];

        let process_body = format!(
            r#"let phase_inc = {freq:.6} / {sr:.1};
for i in 0..block_size {{
    let idx = (self.phase * self.wavetable.len() as f64) as usize % self.wavetable.len();
    output[i] = self.wavetable[idx] * {amp:.6};
    self.phase += phase_inc;
    if self.phase >= 1.0 {{ self.phase -= 1.0; }}
}}"#,
            freq = freq,
            sr = sr,
            amp = amp
        );

        let init_body = format!(
            "self.phase = 0.0;\nself.wavetable = generate_sine_table(2048);"
        );

        Self {
            node_id,
            kind: NodeKind::Oscillator,
            process_body,
            state_fields,
            init_body,
        }
    }

    /// Generate code for a biquad filter.
    pub fn biquad_filter(node_id: u64, params: &HashMap<String, f64>) -> Self {
        let cutoff = params.get("cutoff").copied().unwrap_or(1000.0);
        let q = params.get("q").copied().unwrap_or(0.707);
        let sr = params.get("sample_rate").copied().unwrap_or(48000.0);

        let state_fields = vec![
            StateField { name: "x1".into(), field_type: "f64".into(), initial_value: "0.0".into() },
            StateField { name: "x2".into(), field_type: "f64".into(), initial_value: "0.0".into() },
            StateField { name: "y1".into(), field_type: "f64".into(), initial_value: "0.0".into() },
            StateField { name: "y2".into(), field_type: "f64".into(), initial_value: "0.0".into() },
            StateField { name: "b0".into(), field_type: "f64".into(), initial_value: "0.0".into() },
            StateField { name: "b1".into(), field_type: "f64".into(), initial_value: "0.0".into() },
            StateField { name: "b2".into(), field_type: "f64".into(), initial_value: "0.0".into() },
            StateField { name: "a1".into(), field_type: "f64".into(), initial_value: "0.0".into() },
            StateField { name: "a2".into(), field_type: "f64".into(), initial_value: "0.0".into() },
        ];

        let w0 = 2.0 * std::f64::consts::PI * cutoff / sr;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();
        // Lowpass coefficients
        let b0 = (1.0 - cos_w0) / 2.0;
        let b1 = 1.0 - cos_w0;
        let b2 = b0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        let init_body = format!(
            "self.b0 = {:.10};\nself.b1 = {:.10};\nself.b2 = {:.10};\nself.a1 = {:.10};\nself.a2 = {:.10};\nself.x1 = 0.0;\nself.x2 = 0.0;\nself.y1 = 0.0;\nself.y2 = 0.0;",
            b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0
        );

        let process_body = format!(
            r#"for i in 0..block_size {{
    let x0 = input[i];
    let y0 = self.b0 * x0 + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2;
    self.x2 = self.x1;
    self.x1 = x0;
    self.y2 = self.y1;
    self.y1 = y0;
    output[i] = y0;
}}"#
        );

        Self {
            node_id,
            kind: NodeKind::Filter,
            process_body,
            state_fields,
            init_body,
        }
    }

    /// Generate code for an ADSR envelope state machine.
    pub fn envelope_state_machine(node_id: u64, params: &HashMap<String, f64>) -> Self {
        let attack = params.get("attack").copied().unwrap_or(0.01);
        let decay = params.get("decay").copied().unwrap_or(0.1);
        let sustain = params.get("sustain").copied().unwrap_or(0.7);
        let release = params.get("release").copied().unwrap_or(0.3);
        let sr = params.get("sample_rate").copied().unwrap_or(48000.0);

        let attack_samples = (attack * sr).max(1.0);
        let decay_samples = (decay * sr).max(1.0);
        let release_samples = (release * sr).max(1.0);

        let state_fields = vec![
            StateField { name: "stage".into(), field_type: "u8".into(), initial_value: "0".into() },
            StateField { name: "level".into(), field_type: "f64".into(), initial_value: "0.0".into() },
            StateField { name: "sample_counter".into(), field_type: "usize".into(), initial_value: "0".into() },
        ];

        let process_body = format!(
            r#"let attack_rate = 1.0 / {attack_s:.1};
let decay_rate = (1.0 - {sustain:.6}) / {decay_s:.1};
let release_rate = {sustain:.6} / {release_s:.1};
for i in 0..block_size {{
    match self.stage {{
        0 => {{ // Attack
            self.level += attack_rate;
            if self.level >= 1.0 {{
                self.level = 1.0;
                self.stage = 1;
            }}
        }}
        1 => {{ // Decay
            self.level -= decay_rate;
            if self.level <= {sustain:.6} {{
                self.level = {sustain:.6};
                self.stage = 2;
            }}
        }}
        2 => {{ // Sustain
            self.level = {sustain:.6};
        }}
        3 => {{ // Release
            self.level -= release_rate;
            if self.level <= 0.0 {{
                self.level = 0.0;
                self.stage = 4;
            }}
        }}
        _ => {{ self.level = 0.0; }}
    }}
    output[i] = input[i] * self.level;
}}"#,
            attack_s = attack_samples,
            decay_s = decay_samples,
            sustain = sustain,
            release_s = release_samples
        );

        let init_body = "self.stage = 0;\nself.level = 0.0;\nself.sample_counter = 0;".into();

        Self {
            node_id,
            kind: NodeKind::Envelope,
            process_body,
            state_fields,
            init_body,
        }
    }

    /// Generate code for a summing mixer.
    pub fn summing_mixer(node_id: u64, input_count: usize) -> Self {
        let process_body = format!(
            r#"for i in 0..block_size {{
    let mut sum = 0.0_f64;
    for ch in 0..{n} {{
        sum += inputs[ch][i];
    }}
    output[i] = sum / {n}_f64;
}}"#,
            n = input_count.max(1)
        );

        Self {
            node_id,
            kind: NodeKind::Mixer,
            process_body,
            state_fields: Vec::new(),
            init_body: String::new(),
        }
    }

    /// Generate code for a delay line.
    pub fn delay_line(node_id: u64, params: &HashMap<String, f64>) -> Self {
        let samples = params.get("samples").copied().unwrap_or(4800.0) as usize;
        let feedback = params.get("feedback").copied().unwrap_or(0.0);
        let mix = params.get("mix").copied().unwrap_or(0.5);

        let state_fields = vec![
            StateField {
                name: "buffer".into(),
                field_type: "Vec<f64>".into(),
                initial_value: format!("vec![0.0; {}]", samples),
            },
            StateField {
                name: "write_pos".into(),
                field_type: "usize".into(),
                initial_value: "0".into(),
            },
        ];

        let process_body = format!(
            r#"let delay_len = self.buffer.len();
for i in 0..block_size {{
    let read_pos = (self.write_pos + delay_len - {delay}) % delay_len;
    let delayed = self.buffer[read_pos];
    let out = input[i] * (1.0 - {mix:.6}) + delayed * {mix:.6};
    self.buffer[self.write_pos] = input[i] + delayed * {feedback:.6};
    self.write_pos = (self.write_pos + 1) % delay_len;
    output[i] = out;
}}"#,
            delay = samples,
            mix = mix,
            feedback = feedback
        );

        let init_body = format!(
            "self.buffer = vec![0.0; {}];\nself.write_pos = 0;",
            samples
        );

        Self {
            node_id,
            kind: NodeKind::Delay,
            process_body,
            state_fields,
            init_body,
        }
    }

    /// Generate code for a compressor.
    pub fn compressor(node_id: u64, params: &HashMap<String, f64>) -> Self {
        let threshold = params.get("threshold").copied().unwrap_or(-20.0);
        let ratio = params.get("ratio").copied().unwrap_or(4.0);
        let attack = params.get("attack").copied().unwrap_or(0.01);
        let release = params.get("release").copied().unwrap_or(0.1);
        let sr = params.get("sample_rate").copied().unwrap_or(48000.0);

        let attack_coeff = (-1.0 / (attack * sr)).exp();
        let release_coeff = (-1.0 / (release * sr)).exp();

        let state_fields = vec![
            StateField { name: "envelope".into(), field_type: "f64".into(), initial_value: "0.0".into() },
        ];

        let process_body = format!(
            r#"let threshold_lin = 10.0_f64.powf({threshold:.6} / 20.0);
for i in 0..block_size {{
    let abs_in = input[i].abs();
    if abs_in > self.envelope {{
        self.envelope = {att:.10} * self.envelope + (1.0 - {att:.10}) * abs_in;
    }} else {{
        self.envelope = {rel:.10} * self.envelope + (1.0 - {rel:.10}) * abs_in;
    }}
    let gain = if self.envelope > threshold_lin {{
        let over_db = 20.0 * (self.envelope / threshold_lin).log10();
        let compressed_db = over_db / {ratio:.6};
        10.0_f64.powf((compressed_db - over_db) / 20.0)
    }} else {{
        1.0
    }};
    output[i] = input[i] * gain;
}}"#,
            threshold = threshold,
            att = attack_coeff,
            rel = release_coeff,
            ratio = ratio
        );

        let init_body = "self.envelope = 0.0;".into();

        Self {
            node_id,
            kind: NodeKind::Compressor,
            process_body,
            state_fields,
            init_body,
        }
    }

    /// Generate code for a simple gain node.
    pub fn gain(node_id: u64, params: &HashMap<String, f64>) -> Self {
        let level = params.get("level").copied().unwrap_or(1.0);
        Self {
            node_id,
            kind: NodeKind::Gain,
            process_body: format!(
                "for i in 0..block_size {{\n    output[i] = input[i] * {:.10};\n}}",
                level
            ),
            state_fields: Vec::new(),
            init_body: String::new(),
        }
    }

    /// Generate code for a pan node.
    pub fn pan(node_id: u64, params: &HashMap<String, f64>) -> Self {
        let position = params.get("position").copied().unwrap_or(0.0);
        let left_gain = ((1.0 - position) * std::f64::consts::FRAC_PI_4).cos();
        let right_gain = ((1.0 + position) * std::f64::consts::FRAC_PI_4).cos();
        Self {
            node_id,
            kind: NodeKind::Pan,
            process_body: format!(
                r#"for i in 0..block_size {{
    output_left[i] = input[i] * {:.10};
    output_right[i] = input[i] * {:.10};
}}"#,
                left_gain, right_gain
            ),
            state_fields: Vec::new(),
            init_body: String::new(),
        }
    }

    /// Generate code for any node kind based on its parameters.
    pub fn for_node(node: &LoweredNode, input_count: usize) -> Self {
        match node.kind {
            NodeKind::Oscillator => Self::wavetable_oscillator(node.id, &node.resolved_params),
            NodeKind::Filter => Self::biquad_filter(node.id, &node.resolved_params),
            NodeKind::Envelope => Self::envelope_state_machine(node.id, &node.resolved_params),
            NodeKind::Mixer => Self::summing_mixer(node.id, input_count),
            NodeKind::Delay => Self::delay_line(node.id, &node.resolved_params),
            NodeKind::Compressor => Self::compressor(node.id, &node.resolved_params),
            NodeKind::Gain => Self::gain(node.id, &node.resolved_params),
            NodeKind::Pan => Self::pan(node.id, &node.resolved_params),
            _ => Self::passthrough(node.id, node.kind),
        }
    }

    fn passthrough(node_id: u64, kind: NodeKind) -> Self {
        Self {
            node_id,
            kind,
            process_body: "for i in 0..block_size {\n    output[i] = input[i];\n}".into(),
            state_fields: Vec::new(),
            init_body: String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// GeneratedRenderer
// ---------------------------------------------------------------------------

/// The fully generated renderer, containing all processing components,
/// buffer plan, and wiring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedRenderer {
    pub config: CodegenConfig,
    pub buffer_plan: BufferAllocationPlan,
    pub node_codegen: Vec<NodeCodegen>,
    pub processing_order: Vec<u64>,
    /// Map from node ID to its input buffer indices and output buffer index.
    pub wiring: HashMap<u64, NodeWiring>,
}

/// Buffer wiring for a single node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeWiring {
    pub node_id: u64,
    pub input_buffers: Vec<usize>,
    pub output_buffer: Option<usize>,
}

impl GeneratedRenderer {
    /// Process a single block using the generated renderer.
    /// `buffers` is indexed by physical buffer ID.
    /// Returns the output buffer contents.
    pub fn process_block(&self, input_data: &[f64], block_size: usize) -> Vec<f64> {
        // Simulate processing: for now, allocate buffers and pass through.
        let mut buffers: Vec<Vec<f64>> = (0..self.buffer_plan.buffer_count)
            .map(|_| vec![0.0; block_size])
            .collect();

        // Write input data to the first buffer (if any).
        if !buffers.is_empty() && !input_data.is_empty() {
            let len = input_data.len().min(block_size);
            buffers[0][..len].copy_from_slice(&input_data[..len]);
        }

        // Process each node in order.
        for &node_id in &self.processing_order {
            if let Some(wiring) = self.wiring.get(&node_id) {
                if let Some(codegen) = self.node_codegen.iter().find(|c| c.node_id == node_id) {
                    // Read from input buffers, write to output buffer.
                    let input_sum: Vec<f64> = if wiring.input_buffers.is_empty() {
                        vec![0.0; block_size]
                    } else {
                        let mut sum = vec![0.0; block_size];
                        for &buf_id in &wiring.input_buffers {
                            if buf_id < buffers.len() {
                                for i in 0..block_size {
                                    sum[i] += buffers[buf_id][i];
                                }
                            }
                        }
                        sum
                    };

                    if let Some(out_buf) = wiring.output_buffer {
                        if out_buf < buffers.len() {
                            // Simple passthrough simulation.
                            buffers[out_buf] = input_sum;
                        }
                    }
                }
            }
        }

        // Return the last buffer (output).
        buffers.last().cloned().unwrap_or_else(|| vec![0.0; block_size])
    }
}

// ---------------------------------------------------------------------------
// CodeGenerator
// ---------------------------------------------------------------------------

/// Main code generator: lowers + schedules + generates code + wires buffers.
#[derive(Debug, Clone)]
pub struct CodeGenerator {
    pub config: CodegenConfig,
}

impl CodeGenerator {
    pub fn new(config: &CodegenConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// Generate a renderer from a lowered graph and schedule.
    pub fn generate(
        &self,
        lowered: &LoweredGraph,
        schedule: &Schedule,
    ) -> CodegenResult<GeneratedRenderer> {
        let buffer_plan = BufferAllocationPlan::from_schedule(schedule, &self.config);

        // Generate code for each node.
        let mut node_codegen = Vec::new();
        for node in &lowered.nodes {
            let input_count = lowered
                .edges
                .iter()
                .filter(|e| e.dest_node == node.id)
                .count();
            let codegen = NodeCodegen::for_node(node, input_count);
            node_codegen.push(codegen);
        }

        // Build wiring from the schedule.
        let mut wiring = HashMap::new();
        for step in &schedule.steps {
            let phys_inputs: Vec<usize> = step
                .input_buffers
                .iter()
                .map(|&b| buffer_plan.physical_buffer(b))
                .collect();
            let phys_output = step.output_buffer.map(|b| buffer_plan.physical_buffer(b));
            wiring.insert(
                step.node_id,
                NodeWiring {
                    node_id: step.node_id,
                    input_buffers: phys_inputs,
                    output_buffer: phys_output,
                },
            );
        }

        let processing_order: Vec<u64> = schedule.steps.iter().map(|s| s.node_id).collect();

        Ok(GeneratedRenderer {
            config: self.config.clone(),
            buffer_plan,
            node_codegen,
            processing_order,
            wiring,
        })
    }

    /// Quick end-to-end: lower, schedule, generate.
    pub fn generate_from_graph(&self, graph: &CgGraph) -> CodegenResult<GeneratedRenderer> {
        let lowerer = crate::lowering::IrLowerer::new(&self.config);
        let lowered = lowerer.lower(graph)?;

        // Build a CgGraph-compatible schedule from the lowered graph.
        let scheduler = crate::scheduler::ExecutionScheduler::new(&self.config);

        // Create a temporary CgGraph from the lowered nodes for scheduling.
        let temp_graph = self.lowered_to_cg_graph(&lowered);
        let schedule = scheduler.schedule_sequential(&temp_graph)?;

        self.generate(&lowered, &schedule)
    }

    /// Convert a LoweredGraph back to CgGraph for scheduling.
    fn lowered_to_cg_graph(&self, lowered: &LoweredGraph) -> CgGraph {
        let nodes = lowered
            .nodes
            .iter()
            .map(|n| crate::NodeInfo {
                id: n.id,
                name: n.name.clone(),
                kind: n.kind,
                sample_rate: lowered.sample_rate,
                block_size: lowered.block_size,
                wcet_cycles: 0.0,
                parameters: n.resolved_params.clone(),
                metadata: HashMap::new(),
            })
            .collect();

        CgGraph {
            nodes,
            edges: lowered.edges.clone(),
            topological_order: lowered.topological_order.clone(),
            sample_rate: lowered.sample_rate,
            block_size: lowered.block_size,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BufferKind, CgGraphBuilder, CodegenConfig, NodeKind};
    use std::collections::HashMap;

    fn test_config() -> CodegenConfig {
        CodegenConfig::default()
    }

    fn simple_chain() -> CgGraph {
        let mut b = CgGraphBuilder::new(48000.0, 256);
        let osc = b.add_node("osc", NodeKind::Oscillator);
        let filt = b.add_node("filt", NodeKind::Filter);
        let out = b.add_node("out", NodeKind::Output);
        b.connect(osc, filt, BufferKind::Audio);
        b.connect(filt, out, BufferKind::Audio);
        b.build()
    }

    #[test]
    fn test_buffer_allocation_from_schedule() {
        let cfg = test_config();
        let scheduler = crate::scheduler::ExecutionScheduler::new(&cfg);
        let graph = simple_chain();
        let sched = scheduler.schedule_sequential(&graph).unwrap();
        let plan = BufferAllocationPlan::from_schedule(&sched, &cfg);
        assert!(plan.buffer_count > 0);
        assert!(plan.total_memory_bytes > 0);
    }

    #[test]
    fn test_wavetable_oscillator_codegen() {
        let mut params = HashMap::new();
        params.insert("frequency".into(), 440.0);
        params.insert("amplitude".into(), 0.8);
        params.insert("sample_rate".into(), 48000.0);
        let cg = NodeCodegen::wavetable_oscillator(0, &params);
        assert!(cg.process_body.contains("phase_inc"));
        assert!(!cg.state_fields.is_empty());
    }

    #[test]
    fn test_biquad_filter_codegen() {
        let mut params = HashMap::new();
        params.insert("cutoff".into(), 1000.0);
        params.insert("q".into(), 0.707);
        params.insert("sample_rate".into(), 48000.0);
        let cg = NodeCodegen::biquad_filter(1, &params);
        assert!(cg.process_body.contains("b0"));
        assert_eq!(cg.state_fields.len(), 9);
    }

    #[test]
    fn test_envelope_codegen() {
        let mut params = HashMap::new();
        params.insert("attack".into(), 0.01);
        params.insert("decay".into(), 0.1);
        params.insert("sustain".into(), 0.7);
        params.insert("release".into(), 0.3);
        params.insert("sample_rate".into(), 48000.0);
        let cg = NodeCodegen::envelope_state_machine(2, &params);
        assert!(cg.process_body.contains("Attack"));
        assert_eq!(cg.state_fields.len(), 3);
    }

    #[test]
    fn test_summing_mixer_codegen() {
        let cg = NodeCodegen::summing_mixer(3, 4);
        assert!(cg.process_body.contains("sum"));
    }

    #[test]
    fn test_delay_line_codegen() {
        let mut params = HashMap::new();
        params.insert("samples".into(), 4800.0);
        params.insert("feedback".into(), 0.3);
        params.insert("mix".into(), 0.5);
        let cg = NodeCodegen::delay_line(4, &params);
        assert!(cg.process_body.contains("delayed"));
        assert_eq!(cg.state_fields.len(), 2);
    }

    #[test]
    fn test_compressor_codegen() {
        let mut params = HashMap::new();
        params.insert("threshold".into(), -20.0);
        params.insert("ratio".into(), 4.0);
        params.insert("attack".into(), 0.01);
        params.insert("release".into(), 0.1);
        params.insert("sample_rate".into(), 48000.0);
        let cg = NodeCodegen::compressor(5, &params);
        assert!(cg.process_body.contains("threshold_lin"));
    }

    #[test]
    fn test_gain_codegen() {
        let mut params = HashMap::new();
        params.insert("level".into(), 0.5);
        let cg = NodeCodegen::gain(6, &params);
        assert!(cg.process_body.contains("0.5"));
    }

    #[test]
    fn test_code_generator_end_to_end() {
        let cfg = test_config();
        let gen = CodeGenerator::new(&cfg);
        let graph = simple_chain();
        let renderer = gen.generate_from_graph(&graph).unwrap();
        assert!(!renderer.node_codegen.is_empty());
        assert!(!renderer.processing_order.is_empty());
    }

    #[test]
    fn test_generated_renderer_process_block() {
        let cfg = test_config();
        let gen = CodeGenerator::new(&cfg);
        let graph = simple_chain();
        let renderer = gen.generate_from_graph(&graph).unwrap();
        let input: Vec<f64> = (0..256).map(|i| (i as f64 * 0.01).sin()).collect();
        let output = renderer.process_block(&input, 256);
        assert_eq!(output.len(), 256);
    }

    #[test]
    fn test_buffer_plan_physical_mapping() {
        let cfg = test_config();
        let scheduler = crate::scheduler::ExecutionScheduler::new(&cfg);
        let graph = simple_chain();
        let sched = scheduler.schedule_sequential(&graph).unwrap();
        let plan = BufferAllocationPlan::from_schedule(&sched, &cfg);
        // All logical buffers should map to valid physical buffers.
        for i in 0..sched.buffer_count {
            assert!(plan.physical_buffer(i) < plan.buffer_count);
        }
    }

    #[test]
    fn test_pan_codegen() {
        let mut params = HashMap::new();
        params.insert("position".into(), -0.5);
        let cg = NodeCodegen::pan(7, &params);
        assert!(cg.process_body.contains("output_left"));
        assert!(cg.process_body.contains("output_right"));
    }
}
