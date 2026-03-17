//! Code emission — generates compilable Rust source, WAV output code, and
//! in-process inline renderers from a `GeneratedRenderer`.

use crate::{
    codegen::{GeneratedRenderer, NodeCodegen, StateField},
    CodegenConfig, CodegenError, CodegenResult, NodeKind,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// EmittedCode
// ---------------------------------------------------------------------------

/// Generated source code with compilation instructions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmittedCode {
    /// Generated Rust source code.
    pub source: String,
    /// Suggested file name.
    pub filename: String,
    /// Crate dependencies required.
    pub dependencies: Vec<String>,
    /// Compilation command.
    pub compile_command: String,
    /// Warnings or notes.
    pub notes: Vec<String>,
}

impl EmittedCode {
    /// Total lines of generated code.
    pub fn line_count(&self) -> usize {
        self.source.lines().count()
    }

    /// Whether the source is non-empty.
    pub fn is_valid(&self) -> bool {
        !self.source.is_empty()
    }
}

// ---------------------------------------------------------------------------
// RustEmitter
// ---------------------------------------------------------------------------

/// Emits compilable Rust source for a `GeneratedRenderer`.
#[derive(Debug, Clone)]
pub struct RustEmitter {
    pub config: CodegenConfig,
}

impl RustEmitter {
    pub fn new(config: &CodegenConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// Emit a complete Rust source file for the renderer.
    pub fn emit(&self, renderer: &GeneratedRenderer) -> CodegenResult<EmittedCode> {
        let mut source = String::new();

        // Header
        source.push_str("//! Auto-generated audio renderer.\n");
        source.push_str("//! Do not edit — regenerate from the audio graph.\n\n");

        // Imports
        source.push_str("#![allow(unused_variables, dead_code, clippy::excessive_precision)]\n\n");

        // Helper functions
        source.push_str(&self.emit_helpers());
        source.push_str("\n");

        // Emit state structs for each node.
        for codegen in &renderer.node_codegen {
            source.push_str(&self.emit_node_struct(codegen));
            source.push_str("\n");
        }

        // Emit the main renderer struct.
        source.push_str(&self.emit_renderer_struct(renderer));
        source.push_str("\n");

        // Emit the impl block.
        source.push_str(&self.emit_renderer_impl(renderer));
        source.push_str("\n");

        Ok(EmittedCode {
            source,
            filename: "renderer.rs".into(),
            dependencies: vec!["std".into()],
            compile_command: "rustc --edition 2021 -O renderer.rs".into(),
            notes: vec![format!(
                "Generated for {} at {}Hz, block_size={}",
                self.config.architecture, self.config.sample_rate, self.config.block_size
            )],
        })
    }

    fn emit_helpers(&self) -> String {
        let mut s = String::new();
        s.push_str("/// Generate a sine wavetable.\n");
        s.push_str("fn generate_sine_table(size: usize) -> Vec<f64> {\n");
        s.push_str("    (0..size)\n");
        s.push_str("        .map(|i| (2.0 * std::f64::consts::PI * i as f64 / size as f64).sin())\n");
        s.push_str("        .collect()\n");
        s.push_str("}\n\n");
        s
    }

    fn emit_node_struct(&self, codegen: &NodeCodegen) -> String {
        let struct_name = format!("Node{}State", codegen.node_id);
        let mut s = String::new();

        if codegen.state_fields.is_empty() {
            s.push_str(&format!("struct {} {{}}\n", struct_name));
            return s;
        }

        s.push_str(&format!("struct {} {{\n", struct_name));
        for field in &codegen.state_fields {
            s.push_str(&format!("    {}: {},\n", field.name, field.field_type));
        }
        s.push_str("}\n\n");

        // Default impl
        s.push_str(&format!("impl Default for {} {{\n", struct_name));
        s.push_str("    fn default() -> Self {\n");
        s.push_str("        Self {\n");
        for field in &codegen.state_fields {
            s.push_str(&format!("            {}: {},\n", field.name, field.initial_value));
        }
        s.push_str("        }\n");
        s.push_str("    }\n");
        s.push_str("}\n");

        s
    }

    fn emit_renderer_struct(&self, renderer: &GeneratedRenderer) -> String {
        let mut s = String::new();
        s.push_str("pub struct AudioRenderer {\n");
        s.push_str(&format!(
            "    buffers: Vec<Vec<f64>>,  // {} physical buffers\n",
            renderer.buffer_plan.buffer_count
        ));

        for codegen in &renderer.node_codegen {
            if !codegen.state_fields.is_empty() {
                s.push_str(&format!(
                    "    node_{}_state: Node{}State,\n",
                    codegen.node_id, codegen.node_id
                ));
            }
        }

        s.push_str(&format!("    block_size: usize,  // {}\n", self.config.block_size));
        s.push_str("}\n");
        s
    }

    fn emit_renderer_impl(&self, renderer: &GeneratedRenderer) -> String {
        let mut s = String::new();
        s.push_str("impl AudioRenderer {\n");

        // Constructor
        s.push_str("    pub fn new() -> Self {\n");
        s.push_str("        Self {\n");
        s.push_str(&format!(
            "            buffers: vec![vec![0.0; {}]; {}],\n",
            self.config.block_size, renderer.buffer_plan.buffer_count
        ));
        for codegen in &renderer.node_codegen {
            if !codegen.state_fields.is_empty() {
                s.push_str(&format!(
                    "            node_{}_state: Node{}State::default(),\n",
                    codegen.node_id, codegen.node_id
                ));
            }
        }
        s.push_str(&format!("            block_size: {},\n", self.config.block_size));
        s.push_str("        }\n");
        s.push_str("    }\n\n");

        // Process method
        s.push_str("    pub fn process(&mut self) {\n");
        s.push_str(&format!("        let block_size = {};\n", self.config.block_size));

        for &node_id in &renderer.processing_order {
            if let Some(codegen) = renderer.node_codegen.iter().find(|c| c.node_id == node_id) {
                s.push_str(&format!("\n        // Node {}: {:?}\n", node_id, codegen.kind));

                if let Some(wiring) = renderer.wiring.get(&node_id) {
                    // Read inputs
                    if !wiring.input_buffers.is_empty() {
                        s.push_str(&format!(
                            "        let input = &self.buffers[{}].clone();\n",
                            wiring.input_buffers[0]
                        ));
                    }

                    // Process
                    for line in codegen.process_body.lines() {
                        s.push_str(&format!("        {}\n", line));
                    }
                }
            }
        }

        s.push_str("    }\n");

        // Reset method
        s.push_str("\n    pub fn reset(&mut self) {\n");
        s.push_str("        for buf in &mut self.buffers {\n");
        s.push_str("            buf.fill(0.0);\n");
        s.push_str("        }\n");
        for codegen in &renderer.node_codegen {
            if !codegen.state_fields.is_empty() {
                s.push_str(&format!(
                    "        self.node_{}_state = Node{}State::default();\n",
                    codegen.node_id, codegen.node_id
                ));
            }
        }
        s.push_str("    }\n");

        s.push_str("}\n");
        s
    }
}

// ---------------------------------------------------------------------------
// WavEmitter
// ---------------------------------------------------------------------------

/// Emits code that renders audio and writes a WAV file.
#[derive(Debug, Clone)]
pub struct WavEmitter {
    pub config: CodegenConfig,
    pub duration_seconds: f64,
    pub output_path: String,
}

impl WavEmitter {
    pub fn new(config: &CodegenConfig, duration_seconds: f64, output_path: &str) -> Self {
        Self {
            config: config.clone(),
            duration_seconds,
            output_path: output_path.to_string(),
        }
    }

    /// Emit Rust source that renders and writes a WAV file.
    pub fn emit(&self, renderer: &GeneratedRenderer) -> CodegenResult<EmittedCode> {
        let rust_emitter = RustEmitter::new(&self.config);
        let mut base = rust_emitter.emit(renderer)?;

        let total_blocks =
            (self.duration_seconds * self.config.sample_rate / self.config.block_size as f64)
                .ceil() as usize;

        let mut main_fn = String::new();
        main_fn.push_str("\nfn main() {\n");
        main_fn.push_str("    let mut renderer = AudioRenderer::new();\n");
        main_fn.push_str(&format!("    let total_blocks = {};\n", total_blocks));
        main_fn.push_str(&format!("    let block_size = {};\n", self.config.block_size));
        main_fn.push_str(&format!(
            "    let sample_rate = {:.0};\n",
            self.config.sample_rate
        ));
        main_fn.push_str("    let mut all_samples: Vec<f64> = Vec::new();\n\n");
        main_fn.push_str("    for _ in 0..total_blocks {\n");
        main_fn.push_str("        renderer.process();\n");
        main_fn.push_str("        // Collect output from last buffer\n");
        main_fn.push_str("        all_samples.extend_from_slice(&renderer.buffers.last().unwrap());\n");
        main_fn.push_str("    }\n\n");
        main_fn.push_str(&format!(
            "    write_wav(\"{}\", &all_samples, sample_rate as u32);\n",
            self.output_path
        ));
        main_fn.push_str("}\n\n");

        // WAV writer function
        main_fn.push_str(&self.emit_wav_writer());

        base.source.push_str(&main_fn);
        base.filename = "render_wav.rs".into();
        base.compile_command = "rustc --edition 2021 -O render_wav.rs -o render_wav".into();
        base.notes
            .push(format!("Renders {:.1}s to {}", self.duration_seconds, self.output_path));

        Ok(base)
    }

    fn emit_wav_writer(&self) -> String {
        let mut s = String::new();
        s.push_str("fn write_wav(path: &str, samples: &[f64], sample_rate: u32) {\n");
        s.push_str("    use std::io::Write;\n");
        s.push_str("    let mut file = std::fs::File::create(path).expect(\"Cannot create WAV file\");\n");
        s.push_str("    let num_samples = samples.len() as u32;\n");
        s.push_str("    let bits_per_sample: u16 = 16;\n");
        s.push_str("    let num_channels: u16 = 1;\n");
        s.push_str("    let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;\n");
        s.push_str("    let block_align = num_channels * bits_per_sample / 8;\n");
        s.push_str("    let data_size = num_samples * num_channels as u32 * bits_per_sample as u32 / 8;\n");
        s.push_str("    let file_size = 36 + data_size;\n\n");
        s.push_str("    // RIFF header\n");
        s.push_str("    file.write_all(b\"RIFF\").unwrap();\n");
        s.push_str("    file.write_all(&file_size.to_le_bytes()).unwrap();\n");
        s.push_str("    file.write_all(b\"WAVE\").unwrap();\n");
        s.push_str("    // fmt chunk\n");
        s.push_str("    file.write_all(b\"fmt \").unwrap();\n");
        s.push_str("    file.write_all(&16u32.to_le_bytes()).unwrap();\n");
        s.push_str("    file.write_all(&1u16.to_le_bytes()).unwrap();\n");
        s.push_str("    file.write_all(&num_channels.to_le_bytes()).unwrap();\n");
        s.push_str("    file.write_all(&sample_rate.to_le_bytes()).unwrap();\n");
        s.push_str("    file.write_all(&byte_rate.to_le_bytes()).unwrap();\n");
        s.push_str("    file.write_all(&block_align.to_le_bytes()).unwrap();\n");
        s.push_str("    file.write_all(&bits_per_sample.to_le_bytes()).unwrap();\n");
        s.push_str("    // data chunk\n");
        s.push_str("    file.write_all(b\"data\").unwrap();\n");
        s.push_str("    file.write_all(&data_size.to_le_bytes()).unwrap();\n");
        s.push_str("    for &s in samples {\n");
        s.push_str("        let clamped = s.max(-1.0).min(1.0);\n");
        s.push_str("        let i16_val = (clamped * 32767.0) as i16;\n");
        s.push_str("        file.write_all(&i16_val.to_le_bytes()).unwrap();\n");
        s.push_str("    }\n");
        s.push_str("}\n");
        s
    }
}

// ---------------------------------------------------------------------------
// InlineRenderer
// ---------------------------------------------------------------------------

/// An in-process renderer that can be executed directly without compilation.
#[derive(Debug, Clone)]
pub struct InlineRenderer {
    pub config: CodegenConfig,
    renderer: GeneratedRenderer,
    /// Internal buffers.
    buffers: Vec<Vec<f64>>,
    /// Per-node state: map from node_id to a state vector.
    node_states: HashMap<u64, Vec<f64>>,
}

impl InlineRenderer {
    /// Create from a GeneratedRenderer.
    pub fn new(renderer: GeneratedRenderer) -> Self {
        let buf_count = renderer.buffer_plan.buffer_count.max(1);
        let block_size = renderer.config.block_size;
        let buffers = vec![vec![0.0; block_size]; buf_count];

        let mut node_states = HashMap::new();
        for codegen in &renderer.node_codegen {
            // Initialize state for stateful nodes.
            let state_size = codegen.state_fields.len();
            node_states.insert(codegen.node_id, vec![0.0; state_size.max(1)]);
        }

        Self {
            config: renderer.config.clone(),
            renderer,
            buffers,
            node_states,
        }
    }

    /// Process a single block, returning the output buffer.
    pub fn process(&mut self, input: &[f64]) -> Vec<f64> {
        let bs = self.config.block_size;

        // Clear buffers.
        for buf in &mut self.buffers {
            buf.fill(0.0);
        }

        // Write input to first buffer.
        if !self.buffers.is_empty() {
            let len = input.len().min(bs);
            self.buffers[0][..len].copy_from_slice(&input[..len]);
        }

        // Process each node.
        let processing_order = self.renderer.processing_order.clone();
        for &node_id in &processing_order {
            // Clone wiring data to avoid overlapping borrows.
            let wiring_data = self.renderer.wiring.get(&node_id).cloned();
            if let Some(wiring) = wiring_data {
                // Sum inputs.
                let mut input_sum = vec![0.0; bs];
                for &buf_id in &wiring.input_buffers {
                    if buf_id < self.buffers.len() {
                        for i in 0..bs {
                            input_sum[i] += self.buffers[buf_id][i];
                        }
                    }
                }

                // Find the node's codegen to determine kind.
                let kind = self
                    .renderer
                    .node_codegen
                    .iter()
                    .find(|c| c.node_id == node_id)
                    .map(|c| c.kind);

                // Execute inline processing.
                let output = self.process_node_inline(node_id, kind, &input_sum);

                // Write to output buffer.
                if let Some(out_buf) = wiring.output_buffer {
                    if out_buf < self.buffers.len() {
                        self.buffers[out_buf] = output;
                    }
                }
            }
        }

        self.buffers.last().cloned().unwrap_or_else(|| vec![0.0; bs])
    }

    /// Inline processing for common node types.
    fn process_node_inline(
        &mut self,
        node_id: u64,
        kind: Option<NodeKind>,
        input: &[f64],
    ) -> Vec<f64> {
        let bs = input.len();
        let mut output = vec![0.0; bs];

        let state = self.node_states.entry(node_id).or_insert_with(|| vec![0.0; 4]);

        match kind {
            Some(NodeKind::Oscillator) => {
                // Simple sine oscillator using state[0] as phase.
                let freq = 440.0;
                let phase_inc = freq / self.config.sample_rate;
                for i in 0..bs {
                    output[i] = (state[0] * 2.0 * std::f64::consts::PI).sin();
                    state[0] += phase_inc;
                    if state[0] >= 1.0 {
                        state[0] -= 1.0;
                    }
                }
            }
            Some(NodeKind::Filter) => {
                // Simple one-pole lowpass.
                let alpha = 0.1;
                for i in 0..bs {
                    state[0] = state[0] + alpha * (input[i] - state[0]);
                    output[i] = state[0];
                }
            }
            Some(NodeKind::Gain) => {
                let level = 1.0;
                for i in 0..bs {
                    output[i] = input[i] * level;
                }
            }
            _ => {
                // Passthrough.
                output.copy_from_slice(input);
            }
        }

        output
    }

    /// Reset all state.
    pub fn reset(&mut self) {
        for buf in &mut self.buffers {
            buf.fill(0.0);
        }
        for state in self.node_states.values_mut() {
            state.fill(0.0);
        }
    }

    /// Render `duration_seconds` of audio, returning all samples.
    pub fn render(&mut self, duration_seconds: f64) -> Vec<f64> {
        let total_samples =
            (duration_seconds * self.config.sample_rate).ceil() as usize;
        let blocks = (total_samples + self.config.block_size - 1) / self.config.block_size;
        let silence = vec![0.0; self.config.block_size];
        let mut all = Vec::with_capacity(total_samples);

        for _ in 0..blocks {
            let out = self.process(&silence);
            all.extend_from_slice(&out);
        }

        all.truncate(total_samples);
        all
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BufferKind, CgGraphBuilder, CodegenConfig, NodeKind};

    fn test_config() -> CodegenConfig {
        CodegenConfig::default()
    }

    fn simple_renderer() -> GeneratedRenderer {
        let cfg = test_config();
        let gen = crate::codegen::CodeGenerator::new(&cfg);
        let mut b = CgGraphBuilder::new(48000.0, 256);
        let osc = b.add_node("osc", NodeKind::Oscillator);
        let filt = b.add_node("filt", NodeKind::Filter);
        let out = b.add_node("out", NodeKind::Output);
        b.connect(osc, filt, BufferKind::Audio);
        b.connect(filt, out, BufferKind::Audio);
        let graph = b.build();
        gen.generate_from_graph(&graph).unwrap()
    }

    #[test]
    fn test_rust_emitter_produces_code() {
        let cfg = test_config();
        let emitter = RustEmitter::new(&cfg);
        let renderer = simple_renderer();
        let code = emitter.emit(&renderer).unwrap();
        assert!(code.is_valid());
        assert!(code.source.contains("AudioRenderer"));
        assert!(code.source.contains("process"));
    }

    #[test]
    fn test_rust_emitter_contains_structs() {
        let cfg = test_config();
        let emitter = RustEmitter::new(&cfg);
        let renderer = simple_renderer();
        let code = emitter.emit(&renderer).unwrap();
        assert!(code.source.contains("struct"));
    }

    #[test]
    fn test_rust_emitter_line_count() {
        let cfg = test_config();
        let emitter = RustEmitter::new(&cfg);
        let renderer = simple_renderer();
        let code = emitter.emit(&renderer).unwrap();
        assert!(code.line_count() > 10);
    }

    #[test]
    fn test_wav_emitter_produces_main() {
        let cfg = test_config();
        let emitter = WavEmitter::new(&cfg, 1.0, "output.wav");
        let renderer = simple_renderer();
        let code = emitter.emit(&renderer).unwrap();
        assert!(code.source.contains("fn main()"));
        assert!(code.source.contains("RIFF"));
    }

    #[test]
    fn test_wav_emitter_filename() {
        let cfg = test_config();
        let emitter = WavEmitter::new(&cfg, 1.0, "test.wav");
        let renderer = simple_renderer();
        let code = emitter.emit(&renderer).unwrap();
        assert_eq!(code.filename, "render_wav.rs");
    }

    #[test]
    fn test_inline_renderer_creation() {
        let renderer = simple_renderer();
        let inline = InlineRenderer::new(renderer);
        assert!(!inline.buffers.is_empty());
    }

    #[test]
    fn test_inline_renderer_process() {
        let renderer = simple_renderer();
        let mut inline = InlineRenderer::new(renderer);
        let input = vec![0.0; 256];
        let output = inline.process(&input);
        assert_eq!(output.len(), 256);
    }

    #[test]
    fn test_inline_renderer_render() {
        let renderer = simple_renderer();
        let mut inline = InlineRenderer::new(renderer);
        let samples = inline.render(0.01); // 10ms
        assert!(!samples.is_empty());
        let expected = (0.01 * 48000.0).ceil() as usize;
        assert_eq!(samples.len(), expected);
    }

    #[test]
    fn test_inline_renderer_reset() {
        let renderer = simple_renderer();
        let mut inline = InlineRenderer::new(renderer);
        let input = vec![1.0; 256];
        let _ = inline.process(&input);
        inline.reset();
        // After reset, state should be zeroed.
        for state in inline.node_states.values() {
            assert!(state.iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn test_emitted_code_metadata() {
        let cfg = test_config();
        let emitter = RustEmitter::new(&cfg);
        let renderer = simple_renderer();
        let code = emitter.emit(&renderer).unwrap();
        assert!(!code.dependencies.is_empty());
        assert!(!code.compile_command.is_empty());
    }

    #[test]
    fn test_wav_emitter_duration_note() {
        let cfg = test_config();
        let emitter = WavEmitter::new(&cfg, 2.5, "out.wav");
        let renderer = simple_renderer();
        let code = emitter.emit(&renderer).unwrap();
        assert!(code.notes.iter().any(|n| n.contains("2.5s")));
    }
}
