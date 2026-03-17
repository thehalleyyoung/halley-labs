//! Safety map: visual representation of the safety envelope.
//!
//! Renders deployment states as ASCII art, Graphviz DOT, Mermaid diagrams,
//! or structured JSON, with colour-coded safety annotations.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use safestep_types::StateId;

use crate::{DeploymentState, SafetyEnvelope, Version};

// ---------------------------------------------------------------------------
// EnvelopeMembership
// ---------------------------------------------------------------------------

/// Classifies a state's relationship to the safety envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EnvelopeMembership {
    Inside,
    Outside,
    PNR,
    Boundary,
}

impl std::fmt::Display for EnvelopeMembership {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Inside => write!(f, "Inside"),
            Self::Outside => write!(f, "Outside"),
            Self::PNR => write!(f, "PNR"),
            Self::Boundary => write!(f, "Boundary"),
        }
    }
}

impl EnvelopeMembership {
    pub fn symbol(&self) -> &'static str {
        match self {
            Self::Inside => "●",
            Self::Outside => "○",
            Self::PNR => "◆",
            Self::Boundary => "◇",
        }
    }

    pub fn ansi_color(&self) -> &'static str {
        match self {
            Self::Inside => "\x1b[32m",   // green
            Self::Outside => "\x1b[31m",  // red
            Self::PNR => "\x1b[91m",      // bright red
            Self::Boundary => "\x1b[33m", // yellow
        }
    }
}

// ---------------------------------------------------------------------------
// StateEntry
// ---------------------------------------------------------------------------

/// Information about a single state in the safety map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateEntry {
    pub state_id: StateId,
    pub label: String,
    pub membership: EnvelopeMembership,
    pub forward_reachable: bool,
    pub backward_reachable: bool,
    pub risk_score: f64,
    pub annotations: Vec<String>,
    pub service_versions: Vec<(String, String)>,
}

impl StateEntry {
    pub fn new(state_id: StateId, membership: EnvelopeMembership) -> Self {
        Self {
            label: state_id.as_str().to_string(),
            state_id,
            membership,
            forward_reachable: true,
            backward_reachable: true,
            risk_score: 0.0,
            annotations: Vec::new(),
            service_versions: Vec::new(),
        }
    }

    pub fn with_label(mut self, label: &str) -> Self {
        self.label = label.to_string();
        self
    }

    pub fn with_risk(mut self, score: f64) -> Self {
        self.risk_score = score.clamp(0.0, 1.0);
        self
    }

    pub fn with_reachability(mut self, forward: bool, backward: bool) -> Self {
        self.forward_reachable = forward;
        self.backward_reachable = backward;
        self
    }

    pub fn with_annotation(mut self, note: &str) -> Self {
        self.annotations.push(note.to_string());
        self
    }

    pub fn with_versions(mut self, versions: Vec<(String, String)>) -> Self {
        self.service_versions = versions;
        self
    }

    pub fn is_safe(&self) -> bool {
        self.membership == EnvelopeMembership::Inside
    }

    pub fn is_dangerous(&self) -> bool {
        matches!(
            self.membership,
            EnvelopeMembership::PNR | EnvelopeMembership::Outside
        )
    }
}

// ---------------------------------------------------------------------------
// SafetyMap
// ---------------------------------------------------------------------------

/// A collection of state entries plus transitions, representing the
/// safety envelope visually.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyMap {
    pub states: Vec<StateEntry>,
    pub transitions: Vec<(StateId, StateId, String)>,
    pub title: String,
}

impl SafetyMap {
    pub fn new(title: &str) -> Self {
        Self {
            states: Vec::new(),
            transitions: Vec::new(),
            title: title.to_string(),
        }
    }

    pub fn add_state(&mut self, entry: StateEntry) {
        self.states.push(entry);
    }

    pub fn add_transition(&mut self, from: StateId, to: StateId, label: &str) {
        self.transitions.push((from, to, label.to_string()));
    }

    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }

    pub fn get_state(&self, id: &StateId) -> Option<&StateEntry> {
        self.states.iter().find(|s| s.state_id == *id)
    }

    pub fn pnr_states(&self) -> Vec<&StateEntry> {
        self.states
            .iter()
            .filter(|s| s.membership == EnvelopeMembership::PNR)
            .collect()
    }

    pub fn safe_states(&self) -> Vec<&StateEntry> {
        self.states
            .iter()
            .filter(|s| s.membership == EnvelopeMembership::Inside)
            .collect()
    }

    pub fn max_risk(&self) -> f64 {
        self.states
            .iter()
            .map(|s| s.risk_score)
            .fold(0.0f64, f64::max)
    }

    /// Build a safety map from a plan's states and an envelope.
    pub fn from_envelope(
        envelope: &SafetyEnvelope,
        states: &[DeploymentState],
    ) -> Self {
        let mut map = SafetyMap::new("Safety Envelope Map");

        let pnr_set: std::collections::HashSet<String> = envelope
            .pnr_states
            .iter()
            .map(|s| s.as_str().to_string())
            .collect();
        let boundary_set: std::collections::HashSet<String> = envelope
            .boundary_states
            .iter()
            .map(|s| s.as_str().to_string())
            .collect();
        let safe_set: std::collections::HashSet<String> = envelope
            .safe_states
            .iter()
            .map(|s| s.as_str().to_string())
            .collect();

        for ds in states {
            let sid = ds.id.as_str();
            let membership = if pnr_set.contains(sid) {
                EnvelopeMembership::PNR
            } else if boundary_set.contains(sid) {
                EnvelopeMembership::Boundary
            } else if safe_set.contains(sid) {
                EnvelopeMembership::Inside
            } else {
                EnvelopeMembership::Outside
            };

            let risk = match membership {
                EnvelopeMembership::Inside => 0.1,
                EnvelopeMembership::Boundary => 0.5,
                EnvelopeMembership::PNR => 0.9,
                EnvelopeMembership::Outside => 1.0,
            };

            let versions: Vec<(String, String)> = ds
                .service_versions
                .iter()
                .map(|(svc, ver)| (svc.as_str().to_string(), ver.to_string()))
                .collect();

            let entry = StateEntry::new(ds.id.clone(), membership)
                .with_risk(risk)
                .with_versions(versions);
            map.add_state(entry);
        }

        // Compute reachability from the transition graph
        for (from, to) in &envelope.transitions {
            map.add_transition(from.clone(), to.clone(), "deploy");
        }

        // Forward reachability from first state
        if !map.states.is_empty() {
            let forward = Self::compute_reachable(&map.transitions, true);
            let backward = Self::compute_reachable(&map.transitions, false);
            let first_id = map.states[0].state_id.as_str().to_string();
            let last_id = map
                .states
                .last()
                .map(|s| s.state_id.as_str().to_string())
                .unwrap_or_default();
            for entry in &mut map.states {
                let sid = entry.state_id.as_str().to_string();
                entry.forward_reachable = forward
                    .get(&first_id)
                    .map(|s| s.contains(&sid))
                    .unwrap_or(sid == first_id);
                entry.backward_reachable = backward
                    .get(&last_id)
                    .map(|s| s.contains(&sid))
                    .unwrap_or(sid == last_id);
            }
        }

        map
    }

    fn compute_reachable(
        transitions: &[(StateId, StateId, String)],
        forward: bool,
    ) -> HashMap<String, std::collections::HashSet<String>> {
        let mut adj: HashMap<String, Vec<String>> = HashMap::new();
        for (from, to, _) in transitions {
            let (src, dst) = if forward {
                (from.as_str().to_string(), to.as_str().to_string())
            } else {
                (to.as_str().to_string(), from.as_str().to_string())
            };
            adj.entry(src).or_default().push(dst);
        }

        let mut result: HashMap<String, std::collections::HashSet<String>> = HashMap::new();
        for start in adj.keys() {
            let mut visited = std::collections::HashSet::new();
            let mut stack = vec![start.clone()];
            while let Some(node) = stack.pop() {
                if visited.insert(node.clone()) {
                    if let Some(neighbours) = adj.get(&node) {
                        for n in neighbours {
                            stack.push(n.clone());
                        }
                    }
                }
            }
            result.insert(start.clone(), visited);
        }
        result
    }
}

// ---------------------------------------------------------------------------
// SafetyMapRenderer (facade)
// ---------------------------------------------------------------------------

/// Top-level renderer that delegates to format-specific renderers.
pub struct SafetyMapRenderer;

impl SafetyMapRenderer {
    pub fn to_text(map: &SafetyMap) -> String {
        AsciiRenderer::render(map)
    }

    pub fn to_json(map: &SafetyMap) -> serde_json::Value {
        serde_json::to_value(map).unwrap_or(serde_json::Value::Null)
    }

    pub fn to_dot(map: &SafetyMap) -> String {
        DotRenderer::render(map)
    }

    pub fn to_mermaid(map: &SafetyMap) -> String {
        MermaidRenderer::render(map)
    }
}

// ---------------------------------------------------------------------------
// AsciiRenderer
// ---------------------------------------------------------------------------

/// Renders a safety map as ASCII art with box-drawing characters.
pub struct AsciiRenderer;

impl AsciiRenderer {
    pub fn render(map: &SafetyMap) -> String {
        let mut out = String::new();

        // Title
        let title_len = map.title.len() + 4;
        out.push_str(&format!("┌{}┐\n", "─".repeat(title_len)));
        out.push_str(&format!("│  {}  │\n", map.title));
        out.push_str(&format!("└{}┘\n\n", "─".repeat(title_len)));

        // Legend
        out.push_str("Legend:\n");
        out.push_str(&format!(
            "  {} Inside (safe)   {} Boundary   {} PNR   {} Outside\n\n",
            EnvelopeMembership::Inside.symbol(),
            EnvelopeMembership::Boundary.symbol(),
            EnvelopeMembership::PNR.symbol(),
            EnvelopeMembership::Outside.symbol(),
        ));

        // Timeline
        if !map.states.is_empty() {
            out.push_str("Timeline:\n");
            let col_width = 18;
            let separator = "─".repeat(col_width);

            // Header
            out.push_str("┌");
            for i in 0..map.states.len() {
                out.push_str(&separator);
                if i < map.states.len() - 1 {
                    out.push_str("┬");
                }
            }
            out.push_str("┐\n");

            // State names
            out.push_str("│");
            for entry in &map.states {
                let label = Self::truncate(&entry.label, col_width - 2);
                let padding = col_width - label.len() - 2;
                out.push_str(&format!(" {}{} │", label, " ".repeat(padding)));
            }
            out.push('\n');

            // Membership row
            out.push_str("│");
            for entry in &map.states {
                let sym = entry.membership.symbol();
                let status = format!("{} {}", sym, entry.membership);
                let padding = col_width.saturating_sub(status.len() + 2);
                out.push_str(&format!(" {}{} │", status, " ".repeat(padding)));
            }
            out.push('\n');

            // Risk row
            out.push_str("│");
            for entry in &map.states {
                let risk_str = format!("risk: {:.2}", entry.risk_score);
                let padding = col_width.saturating_sub(risk_str.len() + 2);
                out.push_str(&format!(" {}{} │", risk_str, " ".repeat(padding)));
            }
            out.push('\n');

            // Reachability row
            out.push_str("│");
            for entry in &map.states {
                let fwd = if entry.forward_reachable { "→" } else { "✗" };
                let bwd = if entry.backward_reachable { "←" } else { "✗" };
                let reach = format!("fwd:{} bwd:{}", fwd, bwd);
                let padding = col_width.saturating_sub(reach.len() + 2);
                out.push_str(&format!(" {}{} │", reach, " ".repeat(padding)));
            }
            out.push('\n');

            // Footer
            out.push_str("└");
            for i in 0..map.states.len() {
                out.push_str(&separator);
                if i < map.states.len() - 1 {
                    out.push_str("┴");
                }
            }
            out.push_str("┘\n");

            // Transition arrows
            if !map.transitions.is_empty() {
                out.push_str("\nTransitions:\n");
                for (from, to, label) in &map.transitions {
                    let from_entry = map.get_state(from);
                    let to_entry = map.get_state(to);
                    let from_sym = from_entry
                        .map(|e| e.membership.symbol())
                        .unwrap_or("?");
                    let to_sym = to_entry
                        .map(|e| e.membership.symbol())
                        .unwrap_or("?");
                    out.push_str(&format!(
                        "  {} {} ──[{}]──> {} {}\n",
                        from_sym,
                        from.as_str(),
                        label,
                        to_sym,
                        to.as_str()
                    ));
                }
            }
        }

        // Summary
        out.push_str(&format!(
            "\nSummary: {} states, {} transitions, max risk: {:.2}\n",
            map.state_count(),
            map.transition_count(),
            map.max_risk()
        ));

        let pnr_count = map.pnr_states().len();
        if pnr_count > 0 {
            out.push_str(&format!(
                "WARNING: {} PNR state(s) detected!\n",
                pnr_count
            ));
        }

        out
    }

    fn truncate(s: &str, max: usize) -> String {
        if s.len() <= max {
            s.to_string()
        } else {
            format!("{}…", &s[..max - 1])
        }
    }

    /// Render with ANSI colour codes.
    pub fn render_colored(map: &SafetyMap) -> String {
        let reset = "\x1b[0m";
        let mut out = String::new();

        out.push_str(&format!("\x1b[1m{}\x1b[0m\n\n", map.title));

        for entry in &map.states {
            let color = entry.membership.ansi_color();
            let sym = entry.membership.symbol();
            out.push_str(&format!(
                "{}{} {} [{}] risk={:.2}{}",
                color,
                sym,
                entry.label,
                entry.membership,
                entry.risk_score,
                reset,
            ));
            if !entry.annotations.is_empty() {
                out.push_str(&format!(" ({})", entry.annotations.join("; ")));
            }
            out.push('\n');
        }

        for (from, to, label) in &map.transitions {
            out.push_str(&format!(
                "  {} ──[{}]──> {}\n",
                from.as_str(),
                label,
                to.as_str()
            ));
        }

        out
    }
}

// ---------------------------------------------------------------------------
// DotRenderer
// ---------------------------------------------------------------------------

/// Renders a safety map as a Graphviz DOT graph.
pub struct DotRenderer;

impl DotRenderer {
    pub fn render(map: &SafetyMap) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "digraph \"{}\" {{\n",
            Self::escape_dot(&map.title)
        ));
        out.push_str("  rankdir=LR;\n");
        out.push_str("  node [shape=box, style=filled, fontname=\"Helvetica\"];\n");
        out.push_str("  edge [fontname=\"Helvetica\", fontsize=10];\n\n");

        // Group states by membership for subgraphs
        let mut groups: HashMap<EnvelopeMembership, Vec<&StateEntry>> = HashMap::new();
        for entry in &map.states {
            groups.entry(entry.membership).or_default().push(entry);
        }

        for (membership, entries) in &groups {
            let cluster_name = format!("{:?}", membership).to_lowercase();
            out.push_str(&format!(
                "  subgraph cluster_{} {{\n",
                cluster_name
            ));
            out.push_str(&format!(
                "    label=\"{}\";\n",
                membership
            ));
            out.push_str(&format!(
                "    style=filled;\n    color=\"{}\";\n",
                Self::cluster_color(*membership)
            ));

            for entry in entries {
                let node_id = Self::node_id(&entry.state_id);
                let color = Self::node_fill_color(entry.membership);
                let font_color = Self::font_color(entry.membership);
                let mut label_parts = vec![entry.label.clone()];
                label_parts.push(format!("risk: {:.2}", entry.risk_score));
                for (svc, ver) in &entry.service_versions {
                    label_parts.push(format!("{}: {}", svc, ver));
                }
                let label = label_parts.join("\\n");
                out.push_str(&format!(
                    "    {} [label=\"{}\", fillcolor=\"{}\", fontcolor=\"{}\"];\n",
                    node_id, label, color, font_color
                ));
            }

            out.push_str("  }\n\n");
        }

        // Edges
        for (from, to, label) in &map.transitions {
            let from_id = Self::node_id(from);
            let to_id = Self::node_id(to);
            let edge_color = Self::edge_color(map, from, to);
            out.push_str(&format!(
                "  {} -> {} [label=\"{}\", color=\"{}\"];\n",
                from_id,
                to_id,
                Self::escape_dot(label),
                edge_color
            ));
        }

        out.push_str("}\n");
        out
    }

    fn node_id(state_id: &StateId) -> String {
        let s = state_id.as_str().replace(['-', '.', '/', ' '], "_");
        format!("s_{}", s)
    }

    fn escape_dot(s: &str) -> String {
        s.replace('\"', "\\\"")
    }

    fn node_fill_color(m: EnvelopeMembership) -> &'static str {
        match m {
            EnvelopeMembership::Inside => "#90EE90",   // light green
            EnvelopeMembership::Boundary => "#FFD700",  // gold
            EnvelopeMembership::PNR => "#FF6347",       // tomato
            EnvelopeMembership::Outside => "#FF4500",    // orange red
        }
    }

    fn font_color(m: EnvelopeMembership) -> &'static str {
        match m {
            EnvelopeMembership::Inside | EnvelopeMembership::Boundary => "#000000",
            EnvelopeMembership::PNR | EnvelopeMembership::Outside => "#FFFFFF",
        }
    }

    fn cluster_color(m: EnvelopeMembership) -> &'static str {
        match m {
            EnvelopeMembership::Inside => "#E8F5E9",
            EnvelopeMembership::Boundary => "#FFFDE7",
            EnvelopeMembership::PNR => "#FFEBEE",
            EnvelopeMembership::Outside => "#FBE9E7",
        }
    }

    fn edge_color(map: &SafetyMap, _from: &StateId, to: &StateId) -> &'static str {
        match map.get_state(to).map(|e| e.membership) {
            Some(EnvelopeMembership::Inside) => "#2E7D32",
            Some(EnvelopeMembership::Boundary) => "#F57F17",
            Some(EnvelopeMembership::PNR) => "#C62828",
            Some(EnvelopeMembership::Outside) => "#B71C1C",
            None => "#000000",
        }
    }
}

// ---------------------------------------------------------------------------
// MermaidRenderer
// ---------------------------------------------------------------------------

/// Renders a safety map as a Mermaid diagram.
pub struct MermaidRenderer;

impl MermaidRenderer {
    pub fn render(map: &SafetyMap) -> String {
        let mut out = String::new();
        out.push_str(&format!("---\ntitle: {}\n---\n", map.title));
        out.push_str("stateDiagram-v2\n");

        // State definitions
        for entry in &map.states {
            let id = Self::mermaid_id(&entry.state_id);
            let label = if entry.service_versions.is_empty() {
                format!("{} ({})", entry.label, entry.membership)
            } else {
                let vers: Vec<String> = entry
                    .service_versions
                    .iter()
                    .map(|(s, v)| format!("{}: {}", s, v))
                    .collect();
                format!(
                    "{} ({})\\n{}",
                    entry.label,
                    entry.membership,
                    vers.join("\\n")
                )
            };
            out.push_str(&format!("    {} : {}\n", id, label));
        }

        out.push('\n');

        // Style classes
        out.push_str("    classDef safe fill:#90EE90,stroke:#2E7D32,color:#000\n");
        out.push_str("    classDef boundary fill:#FFD700,stroke:#F57F17,color:#000\n");
        out.push_str("    classDef pnr fill:#FF6347,stroke:#C62828,color:#FFF\n");
        out.push_str("    classDef outside fill:#FF4500,stroke:#B71C1C,color:#FFF\n\n");

        // Apply styles
        for entry in &map.states {
            let id = Self::mermaid_id(&entry.state_id);
            let class = match entry.membership {
                EnvelopeMembership::Inside => "safe",
                EnvelopeMembership::Boundary => "boundary",
                EnvelopeMembership::PNR => "pnr",
                EnvelopeMembership::Outside => "outside",
            };
            out.push_str(&format!("    class {} {}\n", id, class));
        }

        out.push('\n');

        // Transitions
        for (from, to, label) in &map.transitions {
            let from_id = Self::mermaid_id(from);
            let to_id = Self::mermaid_id(to);
            if label.is_empty() {
                out.push_str(&format!("    {} --> {}\n", from_id, to_id));
            } else {
                out.push_str(&format!(
                    "    {} --> {} : {}\n",
                    from_id, to_id, label
                ));
            }
        }

        out
    }

    /// Render as a Mermaid flowchart (alternative style).
    pub fn render_flowchart(map: &SafetyMap) -> String {
        let mut out = String::new();
        out.push_str("flowchart LR\n");

        for entry in &map.states {
            let id = Self::mermaid_id(&entry.state_id);
            let shape = match entry.membership {
                EnvelopeMembership::Inside => format!("{}[{}]", id, entry.label),
                EnvelopeMembership::Boundary => format!("{}{{{}}}",  id, entry.label),
                EnvelopeMembership::PNR => format!("{}(({}))", id, entry.label),
                EnvelopeMembership::Outside => format!("{}>{}]", id, entry.label),
            };
            out.push_str(&format!("    {}\n", shape));
        }

        for (from, to, label) in &map.transitions {
            let from_id = Self::mermaid_id(from);
            let to_id = Self::mermaid_id(to);
            if label.is_empty() {
                out.push_str(&format!("    {} --> {}\n", from_id, to_id));
            } else {
                out.push_str(&format!(
                    "    {} -->|{}| {}\n",
                    from_id, label, to_id
                ));
            }
        }

        // Style classes
        out.push_str("\n    classDef safe fill:#90EE90,stroke:#2E7D32\n");
        out.push_str("    classDef boundary fill:#FFD700,stroke:#F57F17\n");
        out.push_str("    classDef pnr fill:#FF6347,stroke:#C62828,color:#FFF\n");
        out.push_str("    classDef outside fill:#FF4500,stroke:#B71C1C,color:#FFF\n");

        for entry in &map.states {
            let id = Self::mermaid_id(&entry.state_id);
            let class = match entry.membership {
                EnvelopeMembership::Inside => "safe",
                EnvelopeMembership::Boundary => "boundary",
                EnvelopeMembership::PNR => "pnr",
                EnvelopeMembership::Outside => "outside",
            };
            out.push_str(&format!("    class {} {}\n", id, class));
        }

        out
    }

    fn mermaid_id(state_id: &StateId) -> String {
        state_id.as_str().replace(['-', '.', '/', ' '], "_")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use safestep_types::{EnvelopeId, StateId};

    fn sample_map() -> SafetyMap {
        let mut map = SafetyMap::new("Test Deployment");
        map.add_state(
            StateEntry::new(StateId::new("s1"), EnvelopeMembership::Inside)
                .with_label("Initial")
                .with_risk(0.1)
                .with_versions(vec![("api".into(), "1.0.0".into())]),
        );
        map.add_state(
            StateEntry::new(StateId::new("s2"), EnvelopeMembership::Boundary)
                .with_label("Migrating")
                .with_risk(0.5)
                .with_annotation("health check pending"),
        );
        map.add_state(
            StateEntry::new(StateId::new("s3"), EnvelopeMembership::PNR)
                .with_label("PNR-state")
                .with_risk(0.9),
        );
        map.add_state(
            StateEntry::new(StateId::new("s4"), EnvelopeMembership::Inside)
                .with_label("Final")
                .with_risk(0.05),
        );
        map.add_transition(StateId::new("s1"), StateId::new("s2"), "deploy api v2");
        map.add_transition(StateId::new("s2"), StateId::new("s3"), "deploy web v2");
        map.add_transition(StateId::new("s3"), StateId::new("s4"), "deploy db v2");
        map
    }

    #[test]
    fn test_envelope_membership_display() {
        assert_eq!(EnvelopeMembership::Inside.to_string(), "Inside");
        assert_eq!(EnvelopeMembership::PNR.to_string(), "PNR");
    }

    #[test]
    fn test_envelope_membership_symbol() {
        assert_eq!(EnvelopeMembership::Inside.symbol(), "●");
        assert_eq!(EnvelopeMembership::Outside.symbol(), "○");
    }

    #[test]
    fn test_state_entry_builder() {
        let entry = StateEntry::new(StateId::new("s1"), EnvelopeMembership::Inside)
            .with_label("Start")
            .with_risk(0.2)
            .with_reachability(true, false)
            .with_annotation("note1");

        assert_eq!(entry.label, "Start");
        assert!((entry.risk_score - 0.2).abs() < 1e-10);
        assert!(entry.forward_reachable);
        assert!(!entry.backward_reachable);
        assert_eq!(entry.annotations, vec!["note1".to_string()]);
        assert!(entry.is_safe());
        assert!(!entry.is_dangerous());
    }

    #[test]
    fn test_state_entry_risk_clamping() {
        let e = StateEntry::new(StateId::new("s1"), EnvelopeMembership::Inside)
            .with_risk(1.5);
        assert_eq!(e.risk_score, 1.0);
        let e2 = StateEntry::new(StateId::new("s2"), EnvelopeMembership::Inside)
            .with_risk(-0.5);
        assert_eq!(e2.risk_score, 0.0);
    }

    #[test]
    fn test_state_entry_dangerous() {
        let pnr = StateEntry::new(StateId::new("s1"), EnvelopeMembership::PNR);
        assert!(pnr.is_dangerous());
        let outside = StateEntry::new(StateId::new("s2"), EnvelopeMembership::Outside);
        assert!(outside.is_dangerous());
        let boundary = StateEntry::new(StateId::new("s3"), EnvelopeMembership::Boundary);
        assert!(!boundary.is_dangerous());
    }

    #[test]
    fn test_safety_map_basics() {
        let map = sample_map();
        assert_eq!(map.state_count(), 4);
        assert_eq!(map.transition_count(), 3);
        assert!(map.get_state(&StateId::new("s1")).is_some());
        assert!(map.get_state(&StateId::new("s99")).is_none());
        assert_eq!(map.pnr_states().len(), 1);
        assert_eq!(map.safe_states().len(), 2);
        assert!((map.max_risk() - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_safety_map_from_envelope() {
        let mut envelope = SafetyEnvelope::new(EnvelopeId::new("e1"));
        envelope.safe_states.push(StateId::new("s1"));
        envelope.boundary_states.push(StateId::new("s2"));
        envelope.pnr_states.push(StateId::new("s3"));
        envelope.transitions.push((StateId::new("s1"), StateId::new("s2")));
        envelope.transitions.push((StateId::new("s2"), StateId::new("s3")));

        let states = vec![
            DeploymentState::new(StateId::new("s1"))
                .with_service(safestep_types::ServiceId::new("api"), Version::new(1, 0, 0)),
            DeploymentState::new(StateId::new("s2"))
                .with_service(safestep_types::ServiceId::new("api"), Version::new(2, 0, 0)),
            DeploymentState::new(StateId::new("s3"))
                .with_service(safestep_types::ServiceId::new("api"), Version::new(3, 0, 0)),
        ];

        let map = SafetyMap::from_envelope(&envelope, &states);
        assert_eq!(map.state_count(), 3);
        assert_eq!(map.transition_count(), 2);

        let s1 = map.get_state(&StateId::new("s1")).unwrap();
        assert_eq!(s1.membership, EnvelopeMembership::Inside);

        let s3 = map.get_state(&StateId::new("s3")).unwrap();
        assert_eq!(s3.membership, EnvelopeMembership::PNR);
    }

    #[test]
    fn test_ascii_render() {
        let map = sample_map();
        let text = AsciiRenderer::render(&map);
        assert!(text.contains("Test Deployment"));
        assert!(text.contains("Legend:"));
        assert!(text.contains("Timeline:"));
        assert!(text.contains("PNR"));
        assert!(text.contains("risk:"));
        assert!(text.contains("Summary:"));
        assert!(text.contains("4 states"));
    }

    #[test]
    fn test_ascii_render_colored() {
        let map = sample_map();
        let colored = AsciiRenderer::render_colored(&map);
        assert!(colored.contains("\x1b[32m")); // green
        assert!(colored.contains("\x1b[33m")); // yellow
        assert!(colored.contains("\x1b[91m")); // bright red
    }

    #[test]
    fn test_ascii_render_empty() {
        let map = SafetyMap::new("Empty");
        let text = AsciiRenderer::render(&map);
        assert!(text.contains("Empty"));
        assert!(text.contains("0 states"));
    }

    #[test]
    fn test_dot_render() {
        let map = sample_map();
        let dot = DotRenderer::render(&map);
        assert!(dot.starts_with("digraph"));
        assert!(dot.contains("rankdir=LR"));
        assert!(dot.contains("fillcolor="));
        assert!(dot.contains("->"));
        assert!(dot.ends_with("}\n"));
    }

    #[test]
    fn test_dot_node_coloring() {
        let dot = DotRenderer::render(&sample_map());
        assert!(dot.contains("#90EE90")); // green for Inside
        assert!(dot.contains("#FFD700")); // gold for Boundary
        assert!(dot.contains("#FF6347")); // tomato for PNR
    }

    #[test]
    fn test_dot_escape() {
        let mut map = SafetyMap::new("Test \"quoted\"");
        map.add_state(StateEntry::new(StateId::new("s1"), EnvelopeMembership::Inside));
        let dot = DotRenderer::render(&map);
        assert!(dot.contains("Test \\\"quoted\\\""));
    }

    #[test]
    fn test_mermaid_render() {
        let map = sample_map();
        let mermaid = MermaidRenderer::render(&map);
        assert!(mermaid.contains("stateDiagram-v2"));
        assert!(mermaid.contains("classDef safe"));
        assert!(mermaid.contains("classDef pnr"));
        assert!(mermaid.contains("-->"));
    }

    #[test]
    fn test_mermaid_flowchart() {
        let map = sample_map();
        let fc = MermaidRenderer::render_flowchart(&map);
        assert!(fc.contains("flowchart LR"));
        assert!(fc.contains("-->"));
        assert!(fc.contains("classDef safe"));
    }

    #[test]
    fn test_safety_map_renderer_facade() {
        let map = sample_map();
        let text = SafetyMapRenderer::to_text(&map);
        assert!(!text.is_empty());
        let json = SafetyMapRenderer::to_json(&map);
        assert!(json.is_object());
        let dot = SafetyMapRenderer::to_dot(&map);
        assert!(dot.contains("digraph"));
        let mermaid = SafetyMapRenderer::to_mermaid(&map);
        assert!(mermaid.contains("stateDiagram"));
    }

    #[test]
    fn test_mermaid_transition_labels() {
        let mut map = SafetyMap::new("Test");
        map.add_state(StateEntry::new(StateId::new("a"), EnvelopeMembership::Inside));
        map.add_state(StateEntry::new(StateId::new("b"), EnvelopeMembership::Inside));
        map.add_transition(StateId::new("a"), StateId::new("b"), "upgrade");
        let mermaid = MermaidRenderer::render(&map);
        assert!(mermaid.contains("upgrade"));
    }

    #[test]
    fn test_mermaid_empty_label() {
        let mut map = SafetyMap::new("Test");
        map.add_state(StateEntry::new(StateId::new("a"), EnvelopeMembership::Inside));
        map.add_state(StateEntry::new(StateId::new("b"), EnvelopeMembership::Inside));
        map.add_transition(StateId::new("a"), StateId::new("b"), "");
        let mermaid = MermaidRenderer::render(&map);
        assert!(mermaid.contains("a --> b"));
    }

    #[test]
    fn test_state_entry_with_versions() {
        let entry = StateEntry::new(StateId::new("s1"), EnvelopeMembership::Inside)
            .with_versions(vec![
                ("api".into(), "1.0.0".into()),
                ("web".into(), "2.0.0".into()),
            ]);
        assert_eq!(entry.service_versions.len(), 2);
    }

    #[test]
    fn test_dot_subgraph_structure() {
        let map = sample_map();
        let dot = DotRenderer::render(&map);
        assert!(dot.contains("subgraph cluster_"));
        assert!(dot.contains("label="));
    }

    #[test]
    fn test_safety_map_max_risk_empty() {
        let map = SafetyMap::new("empty");
        assert_eq!(map.max_risk(), 0.0);
    }
}
