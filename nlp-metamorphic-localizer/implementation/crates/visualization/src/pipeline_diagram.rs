//! Pipeline diagram generation for visualizing NLP pipeline structure with fault annotations.
//!
//! Produces [`PipelineDiagram`] data structures representing each stage as a node,
//! data-flow as directed edges, and fault markers overlaid on suspicious or faulty stages.
//! Output can be rendered to SVG-like structures or ASCII art.

use serde::{Deserialize, Serialize};
use shared_types::StageId;
use std::fmt;

use crate::color_scale::Color;

// ---------------------------------------------------------------------------
// Fault severity
// ---------------------------------------------------------------------------

/// Severity level for a fault annotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FaultSeverity {
    /// No fault detected.
    None,
    /// Mild anomaly.
    Low,
    /// Suspicious — warrants investigation.
    Medium,
    /// Strong evidence of fault.
    High,
    /// Definite fault.
    Critical,
}

impl FaultSeverity {
    /// Map severity to a representative color.
    pub fn color(&self) -> Color {
        match self {
            FaultSeverity::None => Color::new(200, 200, 200),     // light gray
            FaultSeverity::Low => Color::new(144, 238, 144),      // light green
            FaultSeverity::Medium => Color::new(255, 215, 0),     // gold / yellow
            FaultSeverity::High => Color::new(255, 99, 71),       // tomato
            FaultSeverity::Critical => Color::new(220, 20, 60),   // crimson
        }
    }

    /// ASCII symbol for terminal rendering.
    pub fn symbol(&self) -> &'static str {
        match self {
            FaultSeverity::None => "○",
            FaultSeverity::Low => "◔",
            FaultSeverity::Medium => "◑",
            FaultSeverity::High => "◕",
            FaultSeverity::Critical => "●",
        }
    }
}

impl fmt::Display for FaultSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            FaultSeverity::None => "none",
            FaultSeverity::Low => "low",
            FaultSeverity::Medium => "medium",
            FaultSeverity::High => "high",
            FaultSeverity::Critical => "critical",
        };
        write!(f, "{}", s)
    }
}

// ---------------------------------------------------------------------------
// FaultMarker
// ---------------------------------------------------------------------------

/// A fault annotation attached to a pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultMarker {
    /// Severity of the fault.
    pub severity: FaultSeverity,
    /// Suspiciousness score (0–1).
    pub score: f64,
    /// Human-readable description.
    pub description: String,
    /// Metric name that produced this score.
    pub metric: String,
}

impl FaultMarker {
    /// Create a new fault marker.
    pub fn new(severity: FaultSeverity, score: f64, description: &str, metric: &str) -> Self {
        Self {
            severity,
            score: score.clamp(0.0, 1.0),
            description: description.into(),
            metric: metric.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// DiagramNode
// ---------------------------------------------------------------------------

/// A node in the pipeline diagram, representing one processing stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagramNode {
    /// Unique stage identifier.
    pub stage_id: StageId,
    /// Display label.
    pub label: String,
    /// Order / position index in the pipeline (0-based).
    pub order: usize,
    /// Background fill color.
    pub fill_color: Color,
    /// Border / stroke color.
    pub border_color: Color,
    /// Width in abstract units.
    pub width: f64,
    /// Height in abstract units.
    pub height: f64,
    /// X position (populated by layout).
    pub x: f64,
    /// Y position (populated by layout).
    pub y: f64,
    /// Fault markers attached to this node.
    pub fault_markers: Vec<FaultMarker>,
}

impl DiagramNode {
    /// Create a new diagram node with default geometry.
    pub fn new(stage_id: StageId, label: &str, order: usize) -> Self {
        Self {
            stage_id,
            label: label.into(),
            order,
            fill_color: Color::new(200, 200, 200),
            border_color: Color::new(100, 100, 100),
            width: 120.0,
            height: 60.0,
            x: 0.0,
            y: 0.0,
            fault_markers: Vec::new(),
        }
    }

    /// Attach a fault marker and update node colors accordingly.
    pub fn add_fault_marker(&mut self, marker: FaultMarker) {
        let color = marker.severity.color();
        if marker.severity as u8 > self.max_severity() as u8 {
            self.fill_color = color;
            self.border_color = Color::new(
                color.r.saturating_sub(40),
                color.g.saturating_sub(40),
                color.b.saturating_sub(40),
            );
        }
        self.fault_markers.push(marker);
    }

    /// The worst (highest) fault severity on this node.
    pub fn max_severity(&self) -> FaultSeverity {
        self.fault_markers
            .iter()
            .map(|m| m.severity)
            .max_by_key(|s| *s as u8)
            .unwrap_or(FaultSeverity::None)
    }

    /// Highest suspiciousness score among markers.
    pub fn max_score(&self) -> f64 {
        self.fault_markers
            .iter()
            .map(|m| m.score)
            .fold(0.0_f64, f64::max)
    }

    /// Center point (x + w/2, y + h/2).
    pub fn center(&self) -> (f64, f64) {
        (self.x + self.width / 2.0, self.y + self.height / 2.0)
    }
}

// ---------------------------------------------------------------------------
// DiagramEdge
// ---------------------------------------------------------------------------

/// A directed edge representing data flow between two pipeline stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagramEdge {
    /// Source stage.
    pub from: StageId,
    /// Target stage.
    pub to: StageId,
    /// Optional label (e.g., data type flowing along the edge).
    pub label: Option<String>,
    /// Stroke color.
    pub color: Color,
    /// Stroke width.
    pub stroke_width: f64,
    /// Whether this edge carries the critical path.
    pub is_critical: bool,
}

impl DiagramEdge {
    /// Create a plain data-flow edge.
    pub fn new(from: StageId, to: StageId) -> Self {
        Self {
            from,
            to,
            label: None,
            color: Color::new(100, 100, 100),
            stroke_width: 2.0,
            is_critical: false,
        }
    }

    /// Create a labeled edge.
    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Mark this edge as part of the critical fault path.
    pub fn mark_critical(mut self) -> Self {
        self.is_critical = true;
        self.color = Color::new(220, 20, 60);
        self.stroke_width = 3.0;
        self
    }
}

// ---------------------------------------------------------------------------
// DiagramLayout
// ---------------------------------------------------------------------------

/// Layout strategy for the pipeline diagram.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagramLayout {
    /// Left-to-right horizontal flow.
    Horizontal,
    /// Top-to-bottom vertical flow.
    Vertical,
}

impl Default for DiagramLayout {
    fn default() -> Self {
        Self::Horizontal
    }
}

// ---------------------------------------------------------------------------
// PipelineDiagram
// ---------------------------------------------------------------------------

/// Complete pipeline diagram data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineDiagram {
    /// Title of the diagram.
    pub title: String,
    /// Pipeline stage nodes.
    pub nodes: Vec<DiagramNode>,
    /// Data-flow edges.
    pub edges: Vec<DiagramEdge>,
    /// Layout strategy.
    pub layout: DiagramLayout,
    /// Horizontal gap between nodes.
    pub h_gap: f64,
    /// Vertical gap between nodes.
    pub v_gap: f64,
    /// Total width after layout.
    pub total_width: f64,
    /// Total height after layout.
    pub total_height: f64,
}

impl PipelineDiagram {
    /// Create a new empty diagram.
    pub fn new(title: &str, layout: DiagramLayout) -> Self {
        Self {
            title: title.into(),
            nodes: Vec::new(),
            edges: Vec::new(),
            layout,
            h_gap: 40.0,
            v_gap: 30.0,
            total_width: 0.0,
            total_height: 0.0,
        }
    }

    /// Add a stage node.
    pub fn add_node(&mut self, node: DiagramNode) {
        self.nodes.push(node);
    }

    /// Add a data-flow edge.
    pub fn add_edge(&mut self, edge: DiagramEdge) {
        self.edges.push(edge);
    }

    /// Build a linear pipeline from stage ids and labels.
    pub fn linear_pipeline(
        stages: &[(StageId, String)],
        layout: DiagramLayout,
    ) -> Self {
        let title = "NLP Pipeline Diagram";
        let mut diagram = Self::new(title, layout);

        for (i, (sid, label)) in stages.iter().enumerate() {
            diagram.add_node(DiagramNode::new(sid.clone(), label, i));
        }
        for i in 0..stages.len().saturating_sub(1) {
            diagram.add_edge(DiagramEdge::new(
                stages[i].0.clone(),
                stages[i + 1].0.clone(),
            ));
        }
        diagram.compute_layout();
        diagram
    }

    /// Compute positions for all nodes based on the layout strategy.
    pub fn compute_layout(&mut self) {
        self.nodes.sort_by_key(|n| n.order);

        match self.layout {
            DiagramLayout::Horizontal => {
                let mut x = self.h_gap;
                let y = self.v_gap;
                for node in &mut self.nodes {
                    node.x = x;
                    node.y = y;
                    x += node.width + self.h_gap;
                }
                self.total_width = x;
                self.total_height = self
                    .nodes
                    .iter()
                    .map(|n| n.y + n.height)
                    .fold(0.0_f64, f64::max)
                    + self.v_gap;
            }
            DiagramLayout::Vertical => {
                let x = self.h_gap;
                let mut y = self.v_gap;
                for node in &mut self.nodes {
                    node.x = x;
                    node.y = y;
                    y += node.height + self.v_gap;
                }
                self.total_width = self
                    .nodes
                    .iter()
                    .map(|n| n.x + n.width)
                    .fold(0.0_f64, f64::max)
                    + self.h_gap;
                self.total_height = y;
            }
        }
    }

    /// Look up a node by stage id.
    pub fn node_by_id(&self, id: &StageId) -> Option<&DiagramNode> {
        self.nodes.iter().find(|n| &n.stage_id == id)
    }

    /// Look up a mutable node by stage id.
    pub fn node_by_id_mut(&mut self, id: &StageId) -> Option<&mut DiagramNode> {
        self.nodes.iter_mut().find(|n| &n.stage_id == id)
    }

    /// Return nodes sorted by descending suspiciousness score.
    pub fn nodes_by_suspiciousness(&self) -> Vec<&DiagramNode> {
        let mut v: Vec<_> = self.nodes.iter().collect();
        v.sort_by(|a, b| b.max_score().partial_cmp(&a.max_score()).unwrap());
        v
    }

    /// Return all faulty nodes (severity ≥ High).
    pub fn faulty_nodes(&self) -> Vec<&DiagramNode> {
        self.nodes
            .iter()
            .filter(|n| n.max_severity() as u8 >= FaultSeverity::High as u8)
            .collect()
    }

    /// Generate an ASCII art representation of the pipeline.
    pub fn to_ascii(&self) -> String {
        let mut out = String::new();
        out.push_str(&self.title);
        out.push('\n');
        out.push_str(&"─".repeat(self.title.len()));
        out.push('\n');

        let sorted: Vec<&DiagramNode> = {
            let mut v: Vec<_> = self.nodes.iter().collect();
            v.sort_by_key(|n| n.order);
            v
        };

        match self.layout {
            DiagramLayout::Horizontal => {
                // Top border
                for node in &sorted {
                    out.push_str(&format!("┌{:─^width$}┐", "", width = node.label.len() + 2));
                    if node.order < sorted.len() - 1 {
                        out.push_str("──");
                    }
                }
                out.push('\n');
                // Label row
                for node in &sorted {
                    let sev = node.max_severity();
                    out.push_str(&format!(
                        "│ {}{} │",
                        node.label,
                        if sev != FaultSeverity::None {
                            format!("{}", sev.symbol())
                        } else {
                            " ".into()
                        }
                    ));
                    if node.order < sorted.len() - 1 {
                        out.push_str("→ ");
                    }
                }
                out.push('\n');
                // Bottom border
                for node in &sorted {
                    out.push_str(&format!("└{:─^width$}┘", "", width = node.label.len() + 2));
                    if node.order < sorted.len() - 1 {
                        out.push_str("──");
                    }
                }
                out.push('\n');
            }
            DiagramLayout::Vertical => {
                for node in &sorted {
                    let sev = node.max_severity();
                    let sym = if sev != FaultSeverity::None {
                        sev.symbol()
                    } else {
                        " "
                    };
                    out.push_str(&format!(
                        "┌{:─^width$}┐\n",
                        "",
                        width = node.label.len() + 4
                    ));
                    out.push_str(&format!(
                        "│ {} {} │\n",
                        node.label, sym
                    ));
                    out.push_str(&format!(
                        "└{:─^width$}┘\n",
                        "",
                        width = node.label.len() + 4
                    ));
                    if node.order < sorted.len() - 1 {
                        let pad = (node.label.len() + 6) / 2;
                        out.push_str(&format!("{:>pad$}\n", "│", pad = pad));
                        out.push_str(&format!("{:>pad$}\n", "▼", pad = pad));
                    }
                }
            }
        }
        out
    }

    /// Export diagram metadata as JSON.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "title": self.title,
            "layout": format!("{:?}", self.layout),
            "nodes": self.nodes.iter().map(|n| serde_json::json!({
                "id": n.stage_id.to_string(),
                "label": n.label,
                "order": n.order,
                "x": n.x,
                "y": n.y,
                "width": n.width,
                "height": n.height,
                "fill": n.fill_color.to_hex(),
                "severity": n.max_severity().to_string(),
                "score": n.max_score(),
            })).collect::<Vec<_>>(),
            "edges": self.edges.iter().map(|e| serde_json::json!({
                "from": e.from.to_string(),
                "to": e.to.to_string(),
                "label": e.label,
                "critical": e.is_critical,
            })).collect::<Vec<_>>(),
        })
    }
}

impl fmt::Display for PipelineDiagram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_ascii())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_stages() -> Vec<(StageId, String)> {
        vec![
            (StageId::new("tokenizer"), "Tokenizer".into()),
            (StageId::new("pos"), "POS Tagger".into()),
            (StageId::new("parser"), "Parser".into()),
        ]
    }

    #[test]
    fn linear_pipeline_creates_nodes() {
        let d = PipelineDiagram::linear_pipeline(&sample_stages(), DiagramLayout::Horizontal);
        assert_eq!(d.nodes.len(), 3);
        assert_eq!(d.edges.len(), 2);
    }

    #[test]
    fn layout_assigns_positions() {
        let d = PipelineDiagram::linear_pipeline(&sample_stages(), DiagramLayout::Horizontal);
        for node in &d.nodes {
            assert!(node.x > 0.0 || node.order == 0);
        }
        assert!(d.total_width > 0.0);
    }

    #[test]
    fn vertical_layout() {
        let d = PipelineDiagram::linear_pipeline(&sample_stages(), DiagramLayout::Vertical);
        assert!(d.total_height > d.total_width);
    }

    #[test]
    fn add_fault_marker_updates_color() {
        let mut d = PipelineDiagram::linear_pipeline(&sample_stages(), DiagramLayout::Horizontal);
        let sid = StageId::new("parser");
        if let Some(node) = d.node_by_id_mut(&sid) {
            node.add_fault_marker(FaultMarker::new(
                FaultSeverity::High,
                0.85,
                "High differential",
                "ochiai",
            ));
        }
        let node = d.node_by_id(&sid).unwrap();
        assert_eq!(node.max_severity(), FaultSeverity::High);
        assert!((node.max_score() - 0.85).abs() < 1e-9);
    }

    #[test]
    fn faulty_nodes_filter() {
        let mut d = PipelineDiagram::linear_pipeline(&sample_stages(), DiagramLayout::Horizontal);
        let sid = StageId::new("parser");
        d.node_by_id_mut(&sid).unwrap().add_fault_marker(FaultMarker::new(
            FaultSeverity::Critical,
            0.95,
            "Critical fault",
            "dstar",
        ));
        assert_eq!(d.faulty_nodes().len(), 1);
    }

    #[test]
    fn ascii_horizontal() {
        let d = PipelineDiagram::linear_pipeline(&sample_stages(), DiagramLayout::Horizontal);
        let ascii = d.to_ascii();
        assert!(ascii.contains("Tokenizer"));
        assert!(ascii.contains("→"));
    }

    #[test]
    fn ascii_vertical() {
        let d = PipelineDiagram::linear_pipeline(&sample_stages(), DiagramLayout::Vertical);
        let ascii = d.to_ascii();
        assert!(ascii.contains("▼"));
    }

    #[test]
    fn node_by_id_missing() {
        let d = PipelineDiagram::linear_pipeline(&sample_stages(), DiagramLayout::Horizontal);
        assert!(d.node_by_id(&StageId::new("nonexistent")).is_none());
    }

    #[test]
    fn to_json_structure() {
        let d = PipelineDiagram::linear_pipeline(&sample_stages(), DiagramLayout::Horizontal);
        let j = d.to_json();
        assert_eq!(j["nodes"].as_array().unwrap().len(), 3);
        assert_eq!(j["edges"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn edge_mark_critical() {
        let e = DiagramEdge::new(StageId::new("a"), StageId::new("b")).mark_critical();
        assert!(e.is_critical);
        assert!(e.stroke_width > 2.0);
    }
}
