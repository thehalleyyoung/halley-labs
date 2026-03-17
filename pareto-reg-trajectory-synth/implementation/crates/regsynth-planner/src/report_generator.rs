use serde::{Deserialize, Serialize};

use crate::roadmap::ComplianceRoadmap;
use crate::cost_estimator::CostSummary;
use crate::milestone_tracker::MilestoneTracker;

// ─── Report Format ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportFormat {
    Json,
    PlainText,
    Markdown,
}

impl std::fmt::Display for ReportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Json => write!(f, "JSON"),
            Self::PlainText => write!(f, "Plain Text"),
            Self::Markdown => write!(f, "Markdown"),
        }
    }
}

// ─── Report Section ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSection {
    pub title: String,
    pub content: String,
    pub subsections: Vec<ReportSection>,
}

impl ReportSection {
    pub fn new(title: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            content: content.into(),
            subsections: Vec::new(),
        }
    }

    pub fn with_subsection(mut self, sub: ReportSection) -> Self {
        self.subsections.push(sub);
        self
    }

    fn render_plain(&self, indent: usize) -> String {
        let prefix = " ".repeat(indent);
        let mut out = format!("{}--- {} ---\n{}{}\n\n", prefix, self.title, prefix, self.content);
        for sub in &self.subsections {
            out.push_str(&sub.render_plain(indent + 2));
        }
        out
    }

    fn render_markdown(&self, level: usize) -> String {
        let hashes = "#".repeat(level.min(6));
        let mut out = format!("{} {}\n\n{}\n\n", hashes, self.title, self.content);
        for sub in &self.subsections {
            out.push_str(&sub.render_markdown(level + 1));
        }
        out
    }
}

// ─── Report ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Report {
    pub title: String,
    pub format: ReportFormat,
    pub sections: Vec<ReportSection>,
    pub generated_at: String,
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,
}

impl Report {
    pub fn new(title: impl Into<String>, format: ReportFormat) -> Self {
        Self {
            title: title.into(),
            format,
            sections: Vec::new(),
            generated_at: chrono::Utc::now().to_rfc3339(),
            metadata: std::collections::HashMap::new(),
        }
    }

    pub fn add_section(&mut self, section: ReportSection) {
        self.sections.push(section);
    }

    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Render report to plain text.
    pub fn to_plain_text(&self) -> String {
        let mut out = format!("=== {} ===\nGenerated: {}\n", self.title, self.generated_at);
        if !self.metadata.is_empty() {
            for (k, v) in &self.metadata {
                out.push_str(&format!("{}: {}\n", k, v));
            }
        }
        out.push('\n');
        for section in &self.sections {
            out.push_str(&section.render_plain(0));
        }
        out
    }

    /// Render report to Markdown.
    pub fn to_markdown(&self) -> String {
        let mut out = format!("# {}\n\n*Generated: {}*\n\n", self.title, self.generated_at);
        if !self.metadata.is_empty() {
            out.push_str("| Key | Value |\n|-----|-------|\n");
            for (k, v) in &self.metadata {
                out.push_str(&format!("| {} | {} |\n", k, v));
            }
            out.push('\n');
        }
        for section in &self.sections {
            out.push_str(&section.render_markdown(2));
        }
        out
    }

    /// Render report to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Render in the report's configured format.
    pub fn render(&self) -> String {
        match self.format {
            ReportFormat::Json => self.to_json().unwrap_or_else(|e| format!("Error: {}", e)),
            ReportFormat::PlainText => self.to_plain_text(),
            ReportFormat::Markdown => self.to_markdown(),
        }
    }
}

// ─── Report Generator ───────────────────────────────────────────────────────

/// Generates compliance reports from roadmap, cost, and milestone data.
pub struct ReportGenerator {
    pub format: ReportFormat,
}

impl ReportGenerator {
    pub fn new(format: ReportFormat) -> Self {
        Self { format }
    }

    /// Generate a report from a compliance roadmap.
    pub fn generate(&self, roadmap: &ComplianceRoadmap) -> Report {
        let mut report = Report::new(
            format!("Compliance Roadmap: {}", roadmap.name),
            self.format,
        );
        report.add_metadata("roadmap_id", &roadmap.id);
        report.add_metadata("phases", roadmap.phases.len().to_string());

        let total_tasks: usize = roadmap.phases.iter().map(|p| p.tasks.len()).sum();
        let total_cost: f64 = roadmap.phases.iter().map(|p| p.total_cost()).sum();
        let total_effort: f64 = roadmap.phases.iter().map(|p| p.total_effort()).sum();

        // Executive summary
        report.add_section(ReportSection::new(
            "Executive Summary",
            format!(
                "This roadmap comprises {} phases with {} tasks. \
                 Total estimated effort: {:.0} days. Total estimated cost: ${:.2}.",
                roadmap.phases.len(), total_tasks, total_effort, total_cost,
            ),
        ));

        // Per-phase details
        for phase in &roadmap.phases {
            let task_lines: Vec<String> = phase.tasks.iter()
                .map(|t| format!(
                    "  - {} ({:.0}d, ${:.0}) [{}]",
                    t.name, t.effort_days, t.cost_estimate, t.status
                ))
                .collect();

            let content = format!(
                "Duration: {} days | Effort: {:.0}d | Cost: ${:.2} | Completion: {:.0}%\n\nTasks ({}):\n{}",
                phase.duration_days(),
                phase.total_effort(),
                phase.total_cost(),
                phase.completion_percentage(),
                phase.tasks.len(),
                task_lines.join("\n"),
            );

            report.add_section(ReportSection::new(&phase.name, content));
        }

        report
    }

    /// Generate a report that includes cost summary data.
    pub fn generate_with_costs(&self, roadmap: &ComplianceRoadmap, cost_summary: &CostSummary) -> Report {
        let mut report = self.generate(roadmap);

        let cost_section = ReportSection::new(
            "Cost Analysis",
            format!(
                "Expected total: ${:.2} ({})\n\
                 Optimistic: ${:.2} | Pessimistic: ${:.2}\n\
                 Std deviation: ${:.2}\n\
                 90% confidence interval: ${:.2} – ${:.2}",
                cost_summary.total_expected,
                cost_summary.currency,
                cost_summary.total_optimistic,
                cost_summary.total_pessimistic,
                cost_summary.total_std_deviation,
                cost_summary.confidence_90_low,
                cost_summary.confidence_90_high,
            ),
        );

        report.add_section(cost_section);

        if !cost_summary.by_category.is_empty() {
            let breakdown_lines: Vec<String> = cost_summary.by_category.iter()
                .map(|(cat, amount)| format!("  - {}: ${:.2}", cat, amount))
                .collect();

            report.add_section(ReportSection::new(
                "Cost Breakdown by Category",
                breakdown_lines.join("\n"),
            ));
        }

        report
    }

    /// Generate a report that includes milestone tracking data.
    pub fn generate_with_milestones(
        &self,
        roadmap: &ComplianceRoadmap,
        tracker: &MilestoneTracker,
        as_of: chrono::NaiveDate,
    ) -> Report {
        let mut report = self.generate(roadmap);
        let status = tracker.status_report(as_of);

        let milestone_lines: Vec<String> = status.milestones.iter()
            .map(|m| format!(
                "  - {} [{}] deadline: {} ({:?})",
                m.name, m.id, m.deadline, m.status,
            ))
            .collect();

        report.add_section(ReportSection::new(
            "Milestone Status",
            format!(
                "As of: {}\nOn track: {} | At risk: {} | Missed: {}\n\n{}",
                status.report_date,
                status.on_track_count,
                status.at_risk_count,
                status.missed_count,
                milestone_lines.join("\n"),
            ),
        ));

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roadmap::{ComplianceRoadmap, RoadmapPhase, RoadmapTask};
    use chrono::NaiveDate;

    fn sample_roadmap() -> ComplianceRoadmap {
        let mut roadmap = ComplianceRoadmap::new("Test Roadmap");
        let start = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
        let end = NaiveDate::from_ymd_opt(2025, 3, 31).unwrap();
        let mut phase = RoadmapPhase::new("p1", "Phase 1", start, end);
        phase.add_task(RoadmapTask::new("t1", "Task One").with_effort(10.0).with_cost(5000.0));
        phase.add_task(RoadmapTask::new("t2", "Task Two").with_effort(20.0).with_cost(10000.0));
        roadmap.add_phase(phase);
        roadmap
    }

    #[test]
    fn test_generate_plain_text() {
        let gen = ReportGenerator::new(ReportFormat::PlainText);
        let report = gen.generate(&sample_roadmap());
        let text = report.to_plain_text();
        assert!(text.contains("Test Roadmap"));
        assert!(text.contains("Phase 1"));
        assert!(text.contains("Task One"));
    }

    #[test]
    fn test_generate_markdown() {
        let gen = ReportGenerator::new(ReportFormat::Markdown);
        let report = gen.generate(&sample_roadmap());
        let md = report.to_markdown();
        assert!(md.contains("# Compliance Roadmap"));
        assert!(md.contains("## Phase 1"));
    }

    #[test]
    fn test_generate_json() {
        let gen = ReportGenerator::new(ReportFormat::Json);
        let report = gen.generate(&sample_roadmap());
        let json = report.to_json().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.get("title").is_some());
        assert!(parsed.get("sections").unwrap().as_array().unwrap().len() >= 2);
    }

    #[test]
    fn test_render_delegates() {
        let gen = ReportGenerator::new(ReportFormat::PlainText);
        let report = gen.generate(&sample_roadmap());
        let rendered = report.render();
        assert!(rendered.contains("Test Roadmap"));
    }

    #[test]
    fn test_report_section_nesting() {
        let section = ReportSection::new("Outer", "outer content")
            .with_subsection(ReportSection::new("Inner", "inner content"));
        let text = section.render_plain(0);
        assert!(text.contains("Outer"));
        assert!(text.contains("Inner"));
    }

    #[test]
    fn test_report_metadata() {
        let mut report = Report::new("Test", ReportFormat::PlainText);
        report.add_metadata("version", "1.0");
        let text = report.to_plain_text();
        assert!(text.contains("version: 1.0"));
    }

    #[test]
    fn test_generate_with_costs() {
        let gen = ReportGenerator::new(ReportFormat::PlainText);
        let summary = CostSummary {
            total_expected: 50000.0,
            total_optimistic: 35000.0,
            total_pessimistic: 80000.0,
            total_std_deviation: 7500.0,
            confidence_90_low: 37662.5,
            confidence_90_high: 62337.5,
            by_category: [("personnel".into(), 30000.0), ("tooling".into(), 20000.0)].into(),
            currency: "USD".into(),
        };
        let report = gen.generate_with_costs(&sample_roadmap(), &summary);
        let text = report.to_plain_text();
        assert!(text.contains("Cost Analysis"));
        assert!(text.contains("50000.00"));
    }

    #[test]
    fn test_format_display() {
        assert_eq!(ReportFormat::Json.to_string(), "JSON");
        assert_eq!(ReportFormat::Markdown.to_string(), "Markdown");
    }
}
