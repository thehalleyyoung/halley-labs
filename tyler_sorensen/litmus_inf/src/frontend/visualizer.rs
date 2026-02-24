//! Execution graph visualization.
//!
//! Generates visual representations of execution graphs in multiple formats:
//! DOT (Graphviz), ASCII art, LaTeX/TikZ, and interactive HTML (D3.js spec).
//! Also supports side-by-side comparison of two executions.

use std::collections::HashMap;
use std::fmt;

use crate::checker::execution::{
    Event, EventId, ExecutionGraph, BitMatrix, OpType, Scope,
};

// ═══════════════════════════════════════════════════════════════════════
// Color scheme for relations
// ═══════════════════════════════════════════════════════════════════════

/// Color configuration for graph visualization.
#[derive(Debug, Clone)]
pub struct ColorScheme {
    pub po_color: String,
    pub rf_color: String,
    pub co_color: String,
    pub fr_color: String,
    pub hb_color: String,
    pub read_color: String,
    pub write_color: String,
    pub fence_color: String,
    pub rmw_color: String,
    pub background: String,
    pub text_color: String,
}

impl ColorScheme {
    pub fn default_scheme() -> Self {
        Self {
            po_color: "#000000".to_string(),
            rf_color: "#FF0000".to_string(),
            co_color: "#0000FF".to_string(),
            fr_color: "#008000".to_string(),
            hb_color: "#FF8C00".to_string(),
            read_color: "#E8F5E9".to_string(),
            write_color: "#FFEBEE".to_string(),
            fence_color: "#E3F2FD".to_string(),
            rmw_color: "#FFF3E0".to_string(),
            background: "#FFFFFF".to_string(),
            text_color: "#000000".to_string(),
        }
    }

    pub fn dark_scheme() -> Self {
        Self {
            po_color: "#CCCCCC".to_string(),
            rf_color: "#FF6B6B".to_string(),
            co_color: "#4ECDC4".to_string(),
            fr_color: "#95E1D3".to_string(),
            hb_color: "#F38181".to_string(),
            read_color: "#2D4059".to_string(),
            write_color: "#4A1942".to_string(),
            fence_color: "#1B3A4B".to_string(),
            rmw_color: "#3E2723".to_string(),
            background: "#1E1E1E".to_string(),
            text_color: "#EEEEEE".to_string(),
        }
    }

    pub fn event_color(&self, op: OpType) -> &str {
        match op {
            OpType::Read => &self.read_color,
            OpType::Write => &self.write_color,
            OpType::Fence => &self.fence_color,
            OpType::RMW => &self.rmw_color,
        }
    }
}

impl Default for ColorScheme {
    fn default() -> Self {
        Self::default_scheme()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// VisualizationConfig
// ═══════════════════════════════════════════════════════════════════════

/// Configuration for graph visualization.
#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    /// Color scheme.
    pub colors: ColorScheme,
    /// Whether to show program order edges.
    pub show_po: bool,
    /// Whether to show reads-from edges.
    pub show_rf: bool,
    /// Whether to show coherence edges.
    pub show_co: bool,
    /// Whether to show from-reads edges.
    pub show_fr: bool,
    /// Whether to show extra named relations.
    pub show_extra: bool,
    /// Whether to group events by thread.
    pub group_by_thread: bool,
    /// Title for the visualization.
    pub title: String,
    /// Maximum number of events to render (for large graphs).
    pub max_events: usize,
}

impl VisualizationConfig {
    pub fn new() -> Self {
        Self {
            colors: ColorScheme::default(),
            show_po: true,
            show_rf: true,
            show_co: true,
            show_fr: true,
            show_extra: false,
            group_by_thread: true,
            title: String::new(),
            max_events: 100,
        }
    }

    pub fn with_title(mut self, title: &str) -> Self {
        self.title = title.to_string();
        self
    }

    pub fn minimal() -> Self {
        Self {
            colors: ColorScheme::default(),
            show_po: true,
            show_rf: true,
            show_co: false,
            show_fr: false,
            show_extra: false,
            group_by_thread: true,
            title: String::new(),
            max_events: 50,
        }
    }
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// ExecutionVisualizer — main API
// ═══════════════════════════════════════════════════════════════════════

/// Generates visual representations of execution graphs.
pub struct ExecutionVisualizer {
    pub config: VisualizationConfig,
}

impl ExecutionVisualizer {
    pub fn new(config: VisualizationConfig) -> Self {
        Self { config }
    }

    pub fn with_default_config() -> Self {
        Self::new(VisualizationConfig::new())
    }

    // ─── DOT output ───────────────────────────────────────────────

    /// Generate a DOT (Graphviz) representation.
    pub fn to_dot(&self, graph: &ExecutionGraph) -> String {
        let mut dot = String::new();
        dot.push_str("digraph execution {\n");
        dot.push_str("  rankdir=TB;\n");
        dot.push_str(&format!(
            "  bgcolor=\"{}\";\n",
            self.config.colors.background
        ));
        dot.push_str(&format!(
            "  node [fontcolor=\"{}\"];\n",
            self.config.colors.text_color
        ));

        if !self.config.title.is_empty() {
            dot.push_str(&format!(
                "  labelloc=t;\n  label=\"{}\";\n",
                self.config.title,
            ));
        }

        // Group events by thread
        if self.config.group_by_thread {
            let mut thread_events: HashMap<usize, Vec<&Event>> = HashMap::new();
            for event in &graph.events {
                thread_events.entry(event.thread).or_default().push(event);
            }

            let mut threads: Vec<usize> = thread_events.keys().copied().collect();
            threads.sort();

            for tid in threads {
                let events = &thread_events[&tid];
                dot.push_str(&format!(
                    "  subgraph cluster_t{} {{\n    label=\"Thread {}\";\n    style=dashed;\n",
                    tid, tid,
                ));
                for event in events {
                    dot.push_str(&self.dot_node(event));
                }
                dot.push_str("  }\n");
            }
        } else {
            for event in &graph.events {
                dot.push_str(&self.dot_node(event));
            }
        }

        // Edges
        if self.config.show_po {
            dot.push_str(&graph.po.dot_edges("po", &self.config.colors.po_color));
        }
        if self.config.show_rf {
            dot.push_str(&graph.rf.dot_edges("rf", &self.config.colors.rf_color));
        }
        if self.config.show_co {
            dot.push_str(&graph.co.dot_edges("co", &self.config.colors.co_color));
        }
        if self.config.show_fr {
            dot.push_str(&graph.fr.dot_edges("fr", &self.config.colors.fr_color));
        }

        if self.config.show_extra {
            let extra_colors = ["#9C27B0", "#795548", "#607D8B", "#FF5722"];
            for (i, rel) in graph.extra.iter().enumerate() {
                let color = extra_colors[i % extra_colors.len()];
                dot.push_str(&rel.matrix.dot_edges(&rel.name, color));
            }
        }

        dot.push_str("}\n");
        dot
    }

    fn dot_node(&self, event: &Event) -> String {
        let color = self.config.colors.event_color(event.op_type);
        let label = event.label();
        format!(
            "    e{} [label=\"e{}:{}\" shape=box style=filled fillcolor=\"{}\"];\n",
            event.id, event.id, label, color,
        )
    }

    // ─── ASCII art ────────────────────────────────────────────────

    /// Generate ASCII art for small executions.
    pub fn to_ascii(&self, graph: &ExecutionGraph) -> String {
        let mut output = String::new();
        let n = graph.events.len();

        if n == 0 {
            return "(empty execution)\n".to_string();
        }

        // Group events by thread
        let mut thread_events: HashMap<usize, Vec<&Event>> = HashMap::new();
        for event in &graph.events {
            thread_events.entry(event.thread).or_default().push(event);
        }

        let mut threads: Vec<usize> = thread_events.keys().copied().collect();
        threads.sort();

        if !self.config.title.is_empty() {
            output.push_str(&format!("=== {} ===\n", self.config.title));
        }

        // Header
        let col_width = 16;
        output.push_str(&format!("{:>5} |", ""));
        for &tid in &threads {
            output.push_str(&format!(" {:^width$}|", format!("T{}", tid), width = col_width));
        }
        output.push('\n');
        output.push_str(&format!(
            "{:>5}-+{}+\n",
            "-----",
            format!("-{}-+", "-".repeat(col_width)).repeat(threads.len()),
        ));

        // Find max number of events in any thread
        let max_events = thread_events.values().map(|v| v.len()).max().unwrap_or(0);

        // Rows
        for row in 0..max_events {
            output.push_str(&format!("{:>5} |", row));
            for &tid in &threads {
                if let Some(events) = thread_events.get(&tid) {
                    if let Some(event) = events.get(row) {
                        let label = event.label();
                        let truncated = if label.len() > col_width {
                            format!("{}…", &label[..col_width - 1])
                        } else {
                            label
                        };
                        output.push_str(&format!(
                            " {:^width$}|",
                            truncated,
                            width = col_width,
                        ));
                    } else {
                        output.push_str(&format!(" {:^width$}|", "", width = col_width));
                    }
                } else {
                    output.push_str(&format!(" {:^width$}|", "", width = col_width));
                }
            }
            output.push('\n');
        }

        // Relations summary
        output.push('\n');
        if self.config.show_rf {
            let rf_edges = graph.rf.edges();
            if !rf_edges.is_empty() {
                output.push_str("rf: ");
                let edge_strs: Vec<String> =
                    rf_edges.iter().map(|(i, j)| format!("e{}→e{}", i, j)).collect();
                output.push_str(&edge_strs.join(", "));
                output.push('\n');
            }
        }
        if self.config.show_co {
            let co_edges = graph.co.edges();
            if !co_edges.is_empty() {
                output.push_str("co: ");
                let edge_strs: Vec<String> =
                    co_edges.iter().map(|(i, j)| format!("e{}→e{}", i, j)).collect();
                output.push_str(&edge_strs.join(", "));
                output.push('\n');
            }
        }
        if self.config.show_fr {
            let fr_edges = graph.fr.edges();
            if !fr_edges.is_empty() {
                output.push_str("fr: ");
                let edge_strs: Vec<String> =
                    fr_edges.iter().map(|(i, j)| format!("e{}→e{}", i, j)).collect();
                output.push_str(&edge_strs.join(", "));
                output.push('\n');
            }
        }

        output
    }

    // ─── LaTeX/TikZ output ────────────────────────────────────────

    /// Generate LaTeX/TikZ code for paper figures.
    pub fn to_latex(&self, graph: &ExecutionGraph) -> String {
        let mut tex = String::new();
        tex.push_str("\\begin{tikzpicture}[\n");
        tex.push_str("  event/.style={rectangle, draw, minimum width=2cm, minimum height=0.7cm},\n");
        tex.push_str("  po/.style={->, thick},\n");
        tex.push_str("  rf/.style={->, red, thick, dashed},\n");
        tex.push_str("  co/.style={->, blue, thick, dotted},\n");
        tex.push_str("  fr/.style={->, green!60!black, thick, dashdotted},\n");
        tex.push_str("]\n\n");

        // Group by thread
        let mut thread_events: HashMap<usize, Vec<&Event>> = HashMap::new();
        for event in &graph.events {
            thread_events.entry(event.thread).or_default().push(event);
        }

        let mut threads: Vec<usize> = thread_events.keys().copied().collect();
        threads.sort();

        // Place events
        let x_spacing = 4.0;
        let y_spacing = 1.5;

        for (col, &tid) in threads.iter().enumerate() {
            let x = col as f64 * x_spacing;
            if let Some(events) = thread_events.get(&tid) {
                // Thread label
                tex.push_str(&format!(
                    "  \\node at ({}, 1) {{\\textbf{{Thread {}}}}};\n",
                    x, tid,
                ));
                for (row, event) in events.iter().enumerate() {
                    let y = -(row as f64) * y_spacing;
                    let label = self.latex_escape(&event.label());
                    let fill = match event.op_type {
                        OpType::Read => "green!10",
                        OpType::Write => "red!10",
                        OpType::Fence => "blue!10",
                        OpType::RMW => "orange!10",
                    };
                    tex.push_str(&format!(
                        "  \\node[event, fill={}] (e{}) at ({}, {}) {{$\\mathtt{{{}}}$}};\n",
                        fill, event.id, x, y, label,
                    ));
                }
            }
        }

        tex.push('\n');

        // Program order edges
        if self.config.show_po {
            for (i, j) in graph.po.edges() {
                if graph.events[i].thread == graph.events[j].thread {
                    tex.push_str(&format!(
                        "  \\draw[po] (e{}) -- (e{}) node[midway, left] {{\\scriptsize po}};\n",
                        i, j,
                    ));
                }
            }
        }

        // RF edges
        if self.config.show_rf {
            for (i, j) in graph.rf.edges() {
                tex.push_str(&format!(
                    "  \\draw[rf] (e{}) -- (e{}) node[midway, above] {{\\scriptsize rf}};\n",
                    i, j,
                ));
            }
        }

        // CO edges
        if self.config.show_co {
            for (i, j) in graph.co.edges() {
                tex.push_str(&format!(
                    "  \\draw[co] (e{}) -- (e{}) node[midway, right] {{\\scriptsize co}};\n",
                    i, j,
                ));
            }
        }

        // FR edges
        if self.config.show_fr {
            for (i, j) in graph.fr.edges() {
                tex.push_str(&format!(
                    "  \\draw[fr] (e{}) -- (e{}) node[midway, below] {{\\scriptsize fr}};\n",
                    i, j,
                ));
            }
        }

        tex.push_str("\\end{tikzpicture}\n");
        tex
    }

    fn latex_escape(&self, s: &str) -> String {
        s.replace('_', "\\_")
            .replace('#', "\\#")
            .replace('&', "\\&")
            .replace('{', "\\{")
            .replace('}', "\\}")
    }

    // ─── HTML/D3.js output ────────────────────────────────────────

    /// Generate interactive HTML with D3.js graph specification.
    pub fn to_html(&self, graph: &ExecutionGraph) -> String {
        let nodes = self.build_d3_nodes(graph);
        let links = self.build_d3_links(graph);

        let mut html = String::new();
        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("  <meta charset=\"utf-8\">\n");
        html.push_str(&format!(
            "  <title>{}</title>\n",
            if self.config.title.is_empty() {
                "Execution Graph"
            } else {
                &self.config.title
            },
        ));
        html.push_str("  <script src=\"https://d3js.org/d3.v7.min.js\"></script>\n");
        html.push_str("  <style>\n");
        html.push_str(&format!(
            "    body {{ background: {}; color: {}; font-family: monospace; }}\n",
            self.config.colors.background, self.config.colors.text_color,
        ));
        html.push_str("    .node rect { stroke: #333; stroke-width: 1.5; }\n");
        html.push_str("    .link { fill: none; stroke-width: 2; }\n");
        html.push_str("    .link-label { font-size: 10px; }\n");
        html.push_str("    svg { border: 1px solid #ccc; }\n");
        html.push_str("  </style>\n");
        html.push_str("</head>\n<body>\n");
        html.push_str(&format!(
            "  <h2>{}</h2>\n",
            if self.config.title.is_empty() {
                "Execution Graph"
            } else {
                &self.config.title
            },
        ));
        html.push_str("  <div id=\"graph\"></div>\n");
        html.push_str("  <script>\n");
        html.push_str(&format!("    const nodes = {};\n", nodes));
        html.push_str(&format!("    const links = {};\n", links));
        html.push_str("    const width = 800, height = 600;\n");
        html.push_str("    const svg = d3.select('#graph').append('svg')\n");
        html.push_str("      .attr('width', width).attr('height', height);\n\n");
        html.push_str("    // Arrow markers\n");
        html.push_str("    svg.append('defs').selectAll('marker')\n");
        html.push_str("      .data(['po', 'rf', 'co', 'fr'])\n");
        html.push_str("      .join('marker')\n");
        html.push_str("        .attr('id', d => 'arrow-' + d)\n");
        html.push_str("        .attr('viewBox', '0 -5 10 10')\n");
        html.push_str("        .attr('refX', 20).attr('refY', 0)\n");
        html.push_str("        .attr('markerWidth', 6).attr('markerHeight', 6)\n");
        html.push_str("        .attr('orient', 'auto')\n");
        html.push_str("      .append('path')\n");
        html.push_str("        .attr('d', 'M0,-5L10,0L0,5');\n\n");
        html.push_str("    const simulation = d3.forceSimulation(nodes)\n");
        html.push_str("      .force('link', d3.forceLink(links).id(d => d.id).distance(100))\n");
        html.push_str("      .force('charge', d3.forceManyBody().strength(-200))\n");
        html.push_str("      .force('center', d3.forceCenter(width/2, height/2));\n\n");
        html.push_str("    const link = svg.append('g').selectAll('line')\n");
        html.push_str("      .data(links).join('line')\n");
        html.push_str("      .attr('class', 'link')\n");
        html.push_str("      .attr('stroke', d => d.color)\n");
        html.push_str("      .attr('marker-end', d => 'url(#arrow-' + d.type + ')');\n\n");
        html.push_str("    const node = svg.append('g').selectAll('g')\n");
        html.push_str("      .data(nodes).join('g').attr('class', 'node');\n\n");
        html.push_str("    node.append('rect')\n");
        html.push_str("      .attr('width', 60).attr('height', 25)\n");
        html.push_str("      .attr('x', -30).attr('y', -12)\n");
        html.push_str("      .attr('rx', 4).attr('fill', d => d.color);\n\n");
        html.push_str("    node.append('text')\n");
        html.push_str("      .attr('text-anchor', 'middle').attr('dy', 4)\n");
        html.push_str("      .text(d => d.label).style('font-size', '10px');\n\n");
        html.push_str("    node.append('title').text(d => d.tooltip);\n\n");
        html.push_str("    simulation.on('tick', () => {\n");
        html.push_str("      link.attr('x1', d => d.source.x).attr('y1', d => d.source.y)\n");
        html.push_str("          .attr('x2', d => d.target.x).attr('y2', d => d.target.y);\n");
        html.push_str("      node.attr('transform', d => `translate(${d.x},${d.y})`);\n");
        html.push_str("    });\n");
        html.push_str("  </script>\n");
        html.push_str("</body>\n</html>\n");
        html
    }

    fn build_d3_nodes(&self, graph: &ExecutionGraph) -> String {
        let mut nodes = String::from("[");
        for (i, event) in graph.events.iter().enumerate() {
            if i > 0 {
                nodes.push(',');
            }
            let color = self.config.colors.event_color(event.op_type);
            let label = event.label();
            let tooltip = format!("e{}: T{} {} addr={:#x} val={}", event.id, event.thread, event.op_type, event.address, event.value);
            nodes.push_str(&format!(
                "{{\"id\":\"e{}\",\"label\":\"e{}:{}\",\"color\":\"{}\",\"thread\":{},\"tooltip\":\"{}\"}}",
                event.id, event.id, label, color, event.thread, tooltip,
            ));
        }
        nodes.push(']');
        nodes
    }

    fn build_d3_links(&self, graph: &ExecutionGraph) -> String {
        let mut links = String::from("[");
        let mut first = true;

        let mut add_edges = |matrix: &BitMatrix, rel_type: &str, color: &str| {
            for (i, j) in matrix.edges() {
                if !first {
                    links.push(',');
                }
                first = false;
                links.push_str(&format!(
                    "{{\"source\":\"e{}\",\"target\":\"e{}\",\"type\":\"{}\",\"color\":\"{}\"}}",
                    i, j, rel_type, color,
                ));
            }
        };

        if self.config.show_po {
            add_edges(&graph.po, "po", &self.config.colors.po_color);
        }
        if self.config.show_rf {
            add_edges(&graph.rf, "rf", &self.config.colors.rf_color);
        }
        if self.config.show_co {
            add_edges(&graph.co, "co", &self.config.colors.co_color);
        }
        if self.config.show_fr {
            add_edges(&graph.fr, "fr", &self.config.colors.fr_color);
        }

        links.push(']');
        links
    }

    // ─── Comparison visualization ─────────────────────────────────

    /// Generate a side-by-side comparison of two execution graphs.
    pub fn compare_dot(
        &self,
        graph_a: &ExecutionGraph,
        graph_b: &ExecutionGraph,
        label_a: &str,
        label_b: &str,
    ) -> String {
        let mut dot = String::new();
        dot.push_str("digraph comparison {\n");
        dot.push_str("  rankdir=TB;\n");
        dot.push_str("  compound=true;\n\n");

        // Subgraph A
        dot.push_str(&format!("  subgraph cluster_a {{\n    label=\"{}\";\n", label_a));
        dot.push_str("    style=solid;\n    color=blue;\n");
        for event in &graph_a.events {
            let color = self.config.colors.event_color(event.op_type);
            dot.push_str(&format!(
                "    a_e{} [label=\"e{}:{}\" shape=box style=filled fillcolor=\"{}\"];\n",
                event.id, event.id, event.label(), color,
            ));
        }
        if self.config.show_po {
            for (i, j) in graph_a.po.edges() {
                dot.push_str(&format!(
                    "    a_e{} -> a_e{} [label=\"po\" color=\"{}\"];\n",
                    i, j, self.config.colors.po_color,
                ));
            }
        }
        if self.config.show_rf {
            for (i, j) in graph_a.rf.edges() {
                dot.push_str(&format!(
                    "    a_e{} -> a_e{} [label=\"rf\" color=\"{}\" style=dashed];\n",
                    i, j, self.config.colors.rf_color,
                ));
            }
        }
        dot.push_str("  }\n\n");

        // Subgraph B
        dot.push_str(&format!("  subgraph cluster_b {{\n    label=\"{}\";\n", label_b));
        dot.push_str("    style=solid;\n    color=red;\n");
        for event in &graph_b.events {
            let color = self.config.colors.event_color(event.op_type);
            dot.push_str(&format!(
                "    b_e{} [label=\"e{}:{}\" shape=box style=filled fillcolor=\"{}\"];\n",
                event.id, event.id, event.label(), color,
            ));
        }
        if self.config.show_po {
            for (i, j) in graph_b.po.edges() {
                dot.push_str(&format!(
                    "    b_e{} -> b_e{} [label=\"po\" color=\"{}\"];\n",
                    i, j, self.config.colors.po_color,
                ));
            }
        }
        if self.config.show_rf {
            for (i, j) in graph_b.rf.edges() {
                dot.push_str(&format!(
                    "    b_e{} -> b_e{} [label=\"rf\" color=\"{}\" style=dashed];\n",
                    i, j, self.config.colors.rf_color,
                ));
            }
        }
        dot.push_str("  }\n");

        dot.push_str("}\n");
        dot
    }

    /// Compare two executions in ASCII.
    pub fn compare_ascii(
        &self,
        graph_a: &ExecutionGraph,
        graph_b: &ExecutionGraph,
        label_a: &str,
        label_b: &str,
    ) -> String {
        let mut output = String::new();
        output.push_str(&format!("╔══ {} ", label_a));
        output.push_str(&"═".repeat(40 - label_a.len().min(38)));
        output.push_str("╗\n");
        for line in self.to_ascii(graph_a).lines() {
            output.push_str(&format!("║ {:width$} ║\n", line, width = 42));
        }
        output.push_str("╠══ ");
        output.push_str(label_b);
        output.push(' ');
        output.push_str(&"═".repeat(40 - label_b.len().min(38)));
        output.push_str("╣\n");
        for line in self.to_ascii(graph_b).lines() {
            output.push_str(&format!("║ {:width$} ║\n", line, width = 42));
        }
        output.push_str("╚");
        output.push_str(&"═".repeat(44));
        output.push_str("╝\n");
        output
    }
}

impl fmt::Display for ExecutionVisualizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ExecutionVisualizer(title={})", self.config.title)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_graph() -> ExecutionGraph {
        let events = vec![
            Event::new(0, 0, OpType::Write, 0x100, 1).with_po_index(0),
            Event::new(1, 0, OpType::Write, 0x200, 1).with_po_index(1),
            Event::new(2, 1, OpType::Read, 0x100, 1).with_po_index(0),
            Event::new(3, 1, OpType::Read, 0x200, 1).with_po_index(1),
        ];
        let mut graph = ExecutionGraph::new(events);
        graph.rf.set(0, 2, true); // W(x,1) → R(x,1)
        graph.rf.set(1, 3, true); // W(y,1) → R(y,1)
        graph.co = BitMatrix::new(4);
        graph.fr = BitMatrix::new(4);
        graph
    }

    fn make_empty_graph() -> ExecutionGraph {
        ExecutionGraph::new(vec![])
    }

    #[test]
    fn test_color_scheme_default() {
        let cs = ColorScheme::default_scheme();
        assert!(cs.po_color.starts_with('#'));
        assert!(cs.rf_color.starts_with('#'));
    }

    #[test]
    fn test_color_scheme_dark() {
        let cs = ColorScheme::dark_scheme();
        assert!(cs.background.starts_with('#'));
    }

    #[test]
    fn test_color_scheme_event_color() {
        let cs = ColorScheme::default();
        assert_eq!(cs.event_color(OpType::Read), &cs.read_color);
        assert_eq!(cs.event_color(OpType::Write), &cs.write_color);
        assert_eq!(cs.event_color(OpType::Fence), &cs.fence_color);
        assert_eq!(cs.event_color(OpType::RMW), &cs.rmw_color);
    }

    #[test]
    fn test_visualization_config_default() {
        let config = VisualizationConfig::new();
        assert!(config.show_po);
        assert!(config.show_rf);
        assert!(config.show_co);
        assert!(config.show_fr);
        assert!(config.group_by_thread);
    }

    #[test]
    fn test_visualization_config_minimal() {
        let config = VisualizationConfig::minimal();
        assert!(config.show_po);
        assert!(config.show_rf);
        assert!(!config.show_co);
        assert!(!config.show_fr);
    }

    // --- DOT output tests ---

    #[test]
    fn test_dot_output_simple() {
        let graph = make_simple_graph();
        let viz = ExecutionVisualizer::with_default_config();
        let dot = viz.to_dot(&graph);

        assert!(dot.starts_with("digraph"));
        assert!(dot.contains("cluster_t0"));
        assert!(dot.contains("cluster_t1"));
        assert!(dot.contains("e0"));
        assert!(dot.contains("rf"));
    }

    #[test]
    fn test_dot_output_with_title() {
        let graph = make_simple_graph();
        let config = VisualizationConfig::new().with_title("Test Execution");
        let viz = ExecutionVisualizer::new(config);
        let dot = viz.to_dot(&graph);
        assert!(dot.contains("Test Execution"));
    }

    #[test]
    fn test_dot_output_no_grouping() {
        let graph = make_simple_graph();
        let mut config = VisualizationConfig::new();
        config.group_by_thread = false;
        let viz = ExecutionVisualizer::new(config);
        let dot = viz.to_dot(&graph);
        assert!(!dot.contains("cluster_t0"));
    }

    #[test]
    fn test_dot_empty_graph() {
        let graph = make_empty_graph();
        let viz = ExecutionVisualizer::with_default_config();
        let dot = viz.to_dot(&graph);
        assert!(dot.contains("digraph"));
    }

    // --- ASCII output tests ---

    #[test]
    fn test_ascii_output() {
        let graph = make_simple_graph();
        let viz = ExecutionVisualizer::with_default_config();
        let ascii = viz.to_ascii(&graph);
        assert!(ascii.contains("T0"));
        assert!(ascii.contains("T1"));
        assert!(ascii.contains("rf:"));
    }

    #[test]
    fn test_ascii_empty() {
        let graph = make_empty_graph();
        let viz = ExecutionVisualizer::with_default_config();
        let ascii = viz.to_ascii(&graph);
        assert!(ascii.contains("empty"));
    }

    #[test]
    fn test_ascii_with_title() {
        let graph = make_simple_graph();
        let config = VisualizationConfig::new().with_title("MP Test");
        let viz = ExecutionVisualizer::new(config);
        let ascii = viz.to_ascii(&graph);
        assert!(ascii.contains("MP Test"));
    }

    // --- LaTeX output tests ---

    #[test]
    fn test_latex_output() {
        let graph = make_simple_graph();
        let viz = ExecutionVisualizer::with_default_config();
        let tex = viz.to_latex(&graph);
        assert!(tex.contains("\\begin{tikzpicture}"));
        assert!(tex.contains("\\end{tikzpicture}"));
        assert!(tex.contains("Thread 0"));
        assert!(tex.contains("Thread 1"));
        assert!(tex.contains("rf"));
    }

    #[test]
    fn test_latex_escape() {
        let viz = ExecutionVisualizer::with_default_config();
        assert_eq!(viz.latex_escape("a_b"), "a\\_b");
        assert_eq!(viz.latex_escape("a#b"), "a\\#b");
        assert_eq!(viz.latex_escape("a&b"), "a\\&b");
    }

    // --- HTML output tests ---

    #[test]
    fn test_html_output() {
        let graph = make_simple_graph();
        let config = VisualizationConfig::new().with_title("Test");
        let viz = ExecutionVisualizer::new(config);
        let html = viz.to_html(&graph);

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("d3.v7"));
        assert!(html.contains("const nodes"));
        assert!(html.contains("const links"));
        assert!(html.contains("Test"));
    }

    #[test]
    fn test_html_nodes_json() {
        let graph = make_simple_graph();
        let viz = ExecutionVisualizer::with_default_config();
        let nodes = viz.build_d3_nodes(&graph);
        assert!(nodes.starts_with('['));
        assert!(nodes.ends_with(']'));
        assert!(nodes.contains("\"id\":\"e0\""));
    }

    #[test]
    fn test_html_links_json() {
        let graph = make_simple_graph();
        let viz = ExecutionVisualizer::with_default_config();
        let links = viz.build_d3_links(&graph);
        assert!(links.starts_with('['));
        assert!(links.ends_with(']'));
        assert!(links.contains("\"type\":\"rf\""));
    }

    // --- Comparison tests ---

    #[test]
    fn test_compare_dot() {
        let graph_a = make_simple_graph();
        let graph_b = make_simple_graph();
        let viz = ExecutionVisualizer::with_default_config();
        let dot = viz.compare_dot(&graph_a, &graph_b, "Exec A", "Exec B");

        assert!(dot.contains("cluster_a"));
        assert!(dot.contains("cluster_b"));
        assert!(dot.contains("Exec A"));
        assert!(dot.contains("Exec B"));
        assert!(dot.contains("a_e0"));
        assert!(dot.contains("b_e0"));
    }

    #[test]
    fn test_compare_ascii() {
        let graph_a = make_simple_graph();
        let graph_b = make_simple_graph();
        let viz = ExecutionVisualizer::with_default_config();
        let ascii = viz.compare_ascii(&graph_a, &graph_b, "A", "B");

        assert!(ascii.contains("A"));
        assert!(ascii.contains("B"));
        assert!(ascii.contains("╔"));
        assert!(ascii.contains("╝"));
    }

    // --- Display test ---

    #[test]
    fn test_visualizer_display() {
        let config = VisualizationConfig::new().with_title("test");
        let viz = ExecutionVisualizer::new(config);
        let s = format!("{}", viz);
        assert!(s.contains("test"));
    }
}
