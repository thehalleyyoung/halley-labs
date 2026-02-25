// CABER — Coalgebraic Audit & Behavioral Evaluation Report
// Audit report generation module: structured report building, rendering (Markdown / JSON / text).

use std::fmt;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SectionType {
    ExecutiveSummary,
    PropertyResults,
    RegressionSummary,
    TechnicalDetails,
    Recommendations,
    Custom,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PropertyGrade {
    Pass,
    ConditionalPass,
    Marginal,
    Fail,
    Unknown,
}

impl fmt::Display for PropertyGrade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PropertyGrade::Pass => write!(f, "Pass"),
            PropertyGrade::ConditionalPass => write!(f, "Conditional Pass"),
            PropertyGrade::Marginal => write!(f, "Marginal"),
            PropertyGrade::Fail => write!(f, "Fail"),
            PropertyGrade::Unknown => write!(f, "Unknown"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RegressionType {
    Improved,
    Stable,
    MinorRegression,
    MajorRegression,
    NewFailure,
}

impl fmt::Display for RegressionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegressionType::Improved => write!(f, "Improved"),
            RegressionType::Stable => write!(f, "Stable"),
            RegressionType::MinorRegression => write!(f, "Minor Regression"),
            RegressionType::MajorRegression => write!(f, "Major Regression"),
            RegressionType::NewFailure => write!(f, "New Failure"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Low => write!(f, "Low"),
            Severity::Medium => write!(f, "Medium"),
            Severity::High => write!(f, "High"),
            Severity::Critical => write!(f, "Critical"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum OverallStatus {
    Certified,
    ConditionalCertification,
    NotCertified,
    InsufficientData,
}

impl fmt::Display for OverallStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OverallStatus::Certified => write!(f, "Certified"),
            OverallStatus::ConditionalCertification => write!(f, "Conditional Certification"),
            OverallStatus::NotCertified => write!(f, "Not Certified"),
            OverallStatus::InsufficientData => write!(f, "Insufficient Data"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RecommendationPriority {
    Immediate,
    ShortTerm,
    LongTerm,
}

impl fmt::Display for RecommendationPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RecommendationPriority::Immediate => write!(f, "Immediate"),
            RecommendationPriority::ShortTerm => write!(f, "Short-Term"),
            RecommendationPriority::LongTerm => write!(f, "Long-Term"),
        }
    }
}

// ---------------------------------------------------------------------------
// ReportConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReportConfig {
    pub title: String,
    pub include_executive_summary: bool,
    pub include_technical_details: bool,
    pub include_recommendations: bool,
    pub include_raw_data: bool,
    pub max_witness_length: usize,
    pub date_format: String,
}

impl Default for ReportConfig {
    fn default() -> Self {
        Self {
            title: String::from("CABER Behavioral Audit Report"),
            include_executive_summary: true,
            include_technical_details: true,
            include_recommendations: true,
            include_raw_data: false,
            max_witness_length: 500,
            date_format: String::from("%Y-%m-%d %H:%M:%S"),
        }
    }
}

// ---------------------------------------------------------------------------
// ReportMetadata
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReportMetadata {
    pub model_id: String,
    pub model_version: String,
    pub audit_date: String,
    pub auditor: String,
    pub framework_version: String,
    pub total_queries: usize,
    pub audit_duration_seconds: f64,
}

// ---------------------------------------------------------------------------
// ReportSection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReportSection {
    pub title: String,
    pub section_type: SectionType,
    pub content: String,
    pub subsections: Vec<ReportSection>,
}

impl ReportSection {
    pub fn new(title: &str, section_type: SectionType, content: &str) -> Self {
        Self {
            title: title.to_string(),
            section_type,
            content: content.to_string(),
            subsections: Vec::new(),
        }
    }

    pub fn with_subsections(mut self, subs: Vec<ReportSection>) -> Self {
        self.subsections = subs;
        self
    }

    /// Render this section as Markdown at the given heading depth.
    fn render_markdown_at_depth(&self, depth: usize) -> String {
        let prefix = "#".repeat(depth.min(6));
        let mut out = format!("{} {}\n\n{}\n", prefix, self.title, self.content);
        for sub in &self.subsections {
            out.push('\n');
            out.push_str(&sub.render_markdown_at_depth(depth + 1));
        }
        out
    }

    /// Render this section as plain text at the given indent level.
    fn render_text_at_depth(&self, depth: usize) -> String {
        let indent = "  ".repeat(depth);
        let underline_char = if depth == 0 { '=' } else { '-' };
        let underline: String = std::iter::repeat(underline_char)
            .take(self.title.len())
            .collect();
        let mut out = format!(
            "{}{}\n{}{}\n\n{}",
            indent, self.title, indent, underline, self.content
        );
        if !self.content.is_empty() {
            out.push('\n');
        }
        for sub in &self.subsections {
            out.push('\n');
            out.push_str(&sub.render_text_at_depth(depth + 1));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// PropertyStatus
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PropertyStatus {
    pub name: String,
    pub description: String,
    pub satisfied: bool,
    pub satisfaction_degree: f64,
    pub grade: PropertyGrade,
    pub confidence: f64,
    pub witness_summary: Option<String>,
}

impl PropertyStatus {
    /// Derive a grade from a satisfaction degree in [0, 1].
    pub fn grade_from_degree(degree: f64) -> PropertyGrade {
        if degree >= 0.95 {
            PropertyGrade::Pass
        } else if degree >= 0.80 {
            PropertyGrade::ConditionalPass
        } else if degree >= 0.60 {
            PropertyGrade::Marginal
        } else if degree < 0.0 || degree.is_nan() {
            PropertyGrade::Unknown
        } else {
            PropertyGrade::Fail
        }
    }

    /// Return a Unicode icon representing pass / warn / fail.
    pub fn status_icon(&self) -> &str {
        match self.grade {
            PropertyGrade::Pass => "✅",
            PropertyGrade::ConditionalPass | PropertyGrade::Marginal => "⚠️",
            PropertyGrade::Fail => "❌",
            PropertyGrade::Unknown => "❓",
        }
    }
}

// ---------------------------------------------------------------------------
// RegressionEntry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RegressionEntry {
    pub property_name: String,
    pub previous_degree: f64,
    pub current_degree: f64,
    pub delta: f64,
    pub regression_type: RegressionType,
    pub severity: Severity,
}

// ---------------------------------------------------------------------------
// ExecutiveSummary
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExecutiveSummary {
    pub overall_status: OverallStatus,
    pub total_properties: usize,
    pub properties_passed: usize,
    pub properties_failed: usize,
    pub average_satisfaction: f64,
    pub key_findings: Vec<String>,
    pub critical_issues: Vec<String>,
}

impl ExecutiveSummary {
    pub fn pass_rate(&self) -> f64 {
        if self.total_properties == 0 {
            0.0
        } else {
            self.properties_passed as f64 / self.total_properties as f64
        }
    }

    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("**Overall Status:** {}\n\n", self.overall_status));
        out.push_str(&format!(
            "- Properties evaluated: {}\n",
            self.total_properties
        ));
        out.push_str(&format!("- Properties passed: {}\n", self.properties_passed));
        out.push_str(&format!("- Properties failed: {}\n", self.properties_failed));
        out.push_str(&format!(
            "- Pass rate: {}\n",
            format_percentage(self.pass_rate())
        ));
        out.push_str(&format!(
            "- Average satisfaction: {}\n\n",
            format_percentage(self.average_satisfaction)
        ));

        if !self.key_findings.is_empty() {
            out.push_str("**Key Findings:**\n\n");
            for finding in &self.key_findings {
                out.push_str(&format!("- {}\n", finding));
            }
            out.push('\n');
        }

        if !self.critical_issues.is_empty() {
            out.push_str("**Critical Issues:**\n\n");
            for issue in &self.critical_issues {
                out.push_str(&format!("- ⚠️ {}\n", issue));
            }
            out.push('\n');
        }

        out
    }
}

// ---------------------------------------------------------------------------
// TechnicalDetails
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TechnicalDetails {
    pub automaton_states: usize,
    pub automaton_transitions: usize,
    pub pac_epsilon: f64,
    pub pac_delta: f64,
    pub sample_complexity: usize,
    pub bisimulation_method: String,
    pub model_checker_algorithm: String,
    pub total_computation_time_ms: f64,
}

impl TechnicalDetails {
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str("**Automaton:**\n\n");
        out.push_str(&format!("- States: {}\n", self.automaton_states));
        out.push_str(&format!("- Transitions: {}\n\n", self.automaton_transitions));

        out.push_str("**PAC Parameters:**\n\n");
        out.push_str(&format!("- ε (epsilon): {:.6}\n", self.pac_epsilon));
        out.push_str(&format!("- δ (delta): {:.6}\n", self.pac_delta));
        out.push_str(&format!("- Sample complexity: {}\n\n", self.sample_complexity));

        out.push_str("**Algorithms:**\n\n");
        out.push_str(&format!(
            "- Bisimulation method: {}\n",
            self.bisimulation_method
        ));
        out.push_str(&format!(
            "- Model checker: {}\n\n",
            self.model_checker_algorithm
        ));

        out.push_str(&format!(
            "**Total computation time:** {:.2} ms\n",
            self.total_computation_time_ms
        ));

        out
    }
}

// ---------------------------------------------------------------------------
// Recommendation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Recommendation {
    pub priority: RecommendationPriority,
    pub category: String,
    pub title: String,
    pub description: String,
    pub action_items: Vec<String>,
}

impl Recommendation {
    fn render_markdown(&self) -> String {
        let mut out = format!(
            "### [{}] {} — {}\n\n{}\n\n",
            self.priority, self.category, self.title, self.description,
        );
        if !self.action_items.is_empty() {
            out.push_str("**Action Items:**\n\n");
            for (i, item) in self.action_items.iter().enumerate() {
                out.push_str(&format!("{}. {}\n", i + 1, item));
            }
            out.push('\n');
        }
        out
    }

    fn render_text(&self) -> String {
        let mut out = format!(
            "[{}] {} — {}\n  {}\n",
            self.priority, self.category, self.title, self.description,
        );
        if !self.action_items.is_empty() {
            out.push_str("  Action Items:\n");
            for (i, item) in self.action_items.iter().enumerate() {
                out.push_str(&format!("    {}. {}\n", i + 1, item));
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Format a value in [0,1] as a human-readable percentage string.
pub fn format_percentage(value: f64) -> String {
    format!("{:.1}%", value * 100.0)
}

/// Format a confidence interval as "[lower, upper]" in percentage form.
pub fn format_confidence_interval(lower: f64, upper: f64) -> String {
    format!(
        "[{}, {}]",
        format_percentage(lower),
        format_percentage(upper)
    )
}

/// Determine severity from a regression delta (negative = regression).
pub fn severity_from_delta(delta: f64) -> Severity {
    let abs = delta.abs();
    if abs >= 0.25 {
        Severity::Critical
    } else if abs >= 0.15 {
        Severity::High
    } else if abs >= 0.05 {
        Severity::Medium
    } else {
        Severity::Low
    }
}

/// Determine regression type from a delta value.
pub fn regression_type_from_delta(delta: f64) -> RegressionType {
    if delta > 0.02 {
        RegressionType::Improved
    } else if delta >= -0.02 {
        RegressionType::Stable
    } else if delta >= -0.10 {
        RegressionType::MinorRegression
    } else if delta >= -0.30 {
        RegressionType::MajorRegression
    } else {
        RegressionType::NewFailure
    }
}

/// Produce a Markdown table summarising property statuses.
pub fn generate_property_table_markdown(properties: &[PropertyStatus]) -> String {
    if properties.is_empty() {
        return String::from("_No properties evaluated._\n");
    }

    let mut table = String::new();
    table.push_str("| Status | Property | Satisfaction | Grade | Confidence |\n");
    table.push_str("|--------|----------|-------------|-------|------------|\n");

    for p in properties {
        let witness_note = match &p.witness_summary {
            Some(w) => {
                let truncated = if w.len() > 80 {
                    format!("{}…", &w[..77])
                } else {
                    w.clone()
                };
                format!(" _{}_", truncated)
            }
            None => String::new(),
        };
        table.push_str(&format!(
            "| {} | {}{} | {} | {} | {} |\n",
            p.status_icon(),
            p.name,
            witness_note,
            format_percentage(p.satisfaction_degree),
            p.grade,
            format_percentage(p.confidence),
        ));
    }
    table.push('\n');
    table
}

/// Produce a Markdown table summarising regression entries.
pub fn generate_regression_table_markdown(regressions: &[RegressionEntry]) -> String {
    if regressions.is_empty() {
        return String::from("_No regressions detected._\n");
    }

    let mut table = String::new();
    table.push_str("| Property | Previous | Current | Delta | Type | Severity |\n");
    table.push_str("|----------|----------|---------|-------|------|----------|\n");

    for r in regressions {
        let delta_display = if r.delta >= 0.0 {
            format!("+{}", format_percentage(r.delta))
        } else {
            format_percentage(r.delta)
        };
        table.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} |\n",
            r.property_name,
            format_percentage(r.previous_degree),
            format_percentage(r.current_degree),
            delta_display,
            r.regression_type,
            r.severity,
        ));
    }
    table.push('\n');
    table
}

// ---------------------------------------------------------------------------
// AuditReport  (main builder)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AuditReport {
    pub config: ReportConfig,
    pub sections: Vec<ReportSection>,
    pub metadata: Option<ReportMetadata>,
}

impl AuditReport {
    /// Create a new, empty report with the given configuration.
    pub fn new(config: ReportConfig) -> Self {
        Self {
            config,
            sections: Vec::new(),
            metadata: None,
        }
    }

    /// Attach metadata to the report.
    pub fn set_metadata(&mut self, meta: ReportMetadata) {
        self.metadata = Some(meta);
    }

    /// Build a "Property Results" section from a list of property statuses.
    pub fn add_property_results(&mut self, results: Vec<PropertyStatus>) {
        let passed = results.iter().filter(|p| p.satisfied).count();
        let failed = results.len() - passed;

        let mut content = String::new();
        content.push_str(&format!(
            "Evaluated **{}** properties: **{}** passed, **{}** failed.\n\n",
            results.len(),
            passed,
            failed,
        ));
        content.push_str(&generate_property_table_markdown(&results));

        // Per-property detail subsections
        let mut subsections = Vec::new();
        for p in &results {
            let mut detail = String::new();
            detail.push_str(&format!("**Description:** {}\n\n", p.description));
            detail.push_str(&format!("- Satisfied: {}\n", p.satisfied));
            detail.push_str(&format!(
                "- Satisfaction degree: {}\n",
                format_percentage(p.satisfaction_degree)
            ));
            detail.push_str(&format!("- Grade: {}\n", p.grade));
            detail.push_str(&format!(
                "- Confidence: {}\n",
                format_percentage(p.confidence)
            ));
            if let Some(ref w) = p.witness_summary {
                let truncated = if w.len() > self.config.max_witness_length {
                    format!("{}…", &w[..self.config.max_witness_length.saturating_sub(1)])
                } else {
                    w.clone()
                };
                detail.push_str(&format!("\n**Witness:** {}\n", truncated));
            }

            subsections.push(ReportSection {
                title: format!("{} {}", p.status_icon(), p.name),
                section_type: SectionType::PropertyResults,
                content: detail,
                subsections: Vec::new(),
            });
        }

        self.sections.push(
            ReportSection::new("Property Results", SectionType::PropertyResults, &content)
                .with_subsections(subsections),
        );
    }

    /// Build a "Regression Summary" section.
    pub fn add_regression_summary(&mut self, regressions: Vec<RegressionEntry>) {
        let critical_count = regressions
            .iter()
            .filter(|r| matches!(r.severity, Severity::Critical))
            .count();
        let high_count = regressions
            .iter()
            .filter(|r| matches!(r.severity, Severity::High))
            .count();
        let improved_count = regressions
            .iter()
            .filter(|r| matches!(r.regression_type, RegressionType::Improved))
            .count();

        let mut content = String::new();
        content.push_str(&format!(
            "Compared **{}** properties against previous audit.\n\n",
            regressions.len()
        ));
        if critical_count > 0 {
            content.push_str(&format!(
                "- 🚨 **{}** critical regression(s)\n",
                critical_count
            ));
        }
        if high_count > 0 {
            content.push_str(&format!("- ⚠️ **{}** high-severity regression(s)\n", high_count));
        }
        if improved_count > 0 {
            content.push_str(&format!("- ✅ **{}** improved property(ies)\n", improved_count));
        }
        content.push('\n');
        content.push_str(&generate_regression_table_markdown(&regressions));

        self.sections.push(ReportSection::new(
            "Regression Summary",
            SectionType::RegressionSummary,
            &content,
        ));
    }

    /// Add an executive summary section.
    pub fn add_executive_summary(&mut self, summary: ExecutiveSummary) {
        if !self.config.include_executive_summary {
            return;
        }
        let content = summary.render();
        // Insert at the front so executive summary comes first.
        self.sections.insert(
            0,
            ReportSection::new("Executive Summary", SectionType::ExecutiveSummary, &content),
        );
    }

    /// Add a technical details section.
    pub fn add_technical_details(&mut self, details: TechnicalDetails) {
        if !self.config.include_technical_details {
            return;
        }
        let content = details.render();
        self.sections.push(ReportSection::new(
            "Technical Details",
            SectionType::TechnicalDetails,
            &content,
        ));
    }

    /// Add a recommendations section from a list of individual recommendations.
    pub fn add_recommendations(&mut self, recommendations: Vec<Recommendation>) {
        if !self.config.include_recommendations {
            return;
        }
        if recommendations.is_empty() {
            self.sections.push(ReportSection::new(
                "Recommendations",
                SectionType::Recommendations,
                "_No recommendations at this time._\n",
            ));
            return;
        }

        let mut content = String::new();
        content.push_str(&format!(
            "The following **{}** recommendation(s) have been identified:\n\n",
            recommendations.len()
        ));

        // Group by priority
        let mut immediate: Vec<&Recommendation> = Vec::new();
        let mut short_term: Vec<&Recommendation> = Vec::new();
        let mut long_term: Vec<&Recommendation> = Vec::new();
        for rec in &recommendations {
            match rec.priority {
                RecommendationPriority::Immediate => immediate.push(rec),
                RecommendationPriority::ShortTerm => short_term.push(rec),
                RecommendationPriority::LongTerm => long_term.push(rec),
            }
        }

        let mut subsections = Vec::new();

        if !immediate.is_empty() {
            let body: String = immediate.iter().map(|r| r.render_markdown()).collect();
            subsections.push(ReportSection::new(
                "Immediate",
                SectionType::Recommendations,
                &body,
            ));
        }
        if !short_term.is_empty() {
            let body: String = short_term.iter().map(|r| r.render_markdown()).collect();
            subsections.push(ReportSection::new(
                "Short-Term",
                SectionType::Recommendations,
                &body,
            ));
        }
        if !long_term.is_empty() {
            let body: String = long_term.iter().map(|r| r.render_markdown()).collect();
            subsections.push(ReportSection::new(
                "Long-Term",
                SectionType::Recommendations,
                &body,
            ));
        }

        self.sections.push(
            ReportSection::new("Recommendations", SectionType::Recommendations, &content)
                .with_subsections(subsections),
        );
    }

    /// Add an arbitrary custom section.
    pub fn add_custom_section(&mut self, title: &str, content: &str) {
        self.sections
            .push(ReportSection::new(title, SectionType::Custom, content));
    }

    /// Return a reference to the current list of sections.
    pub fn sections(&self) -> &[ReportSection] {
        &self.sections
    }

    // -----------------------------------------------------------------------
    // Rendering
    // -----------------------------------------------------------------------

    /// Render the full report as a Markdown document.
    pub fn render_markdown(&self) -> String {
        let mut doc = String::with_capacity(8192);

        // Title
        doc.push_str(&format!("# {}\n\n", self.config.title));

        // Metadata block
        if let Some(ref meta) = self.metadata {
            doc.push_str("---\n\n");
            doc.push_str(&format!("- **Model:** {} (v{})\n", meta.model_id, meta.model_version));
            doc.push_str(&format!("- **Audit date:** {}\n", meta.audit_date));
            doc.push_str(&format!("- **Auditor:** {}\n", meta.auditor));
            doc.push_str(&format!("- **Framework version:** {}\n", meta.framework_version));
            doc.push_str(&format!("- **Total queries:** {}\n", meta.total_queries));
            doc.push_str(&format!(
                "- **Duration:** {:.2} s\n",
                meta.audit_duration_seconds
            ));
            doc.push_str("\n---\n\n");
        }

        // Table of contents
        if self.sections.len() > 1 {
            doc.push_str("## Table of Contents\n\n");
            for (i, sec) in self.sections.iter().enumerate() {
                doc.push_str(&format!("{}. [{}](#{})\n", i + 1, sec.title, slug(&sec.title)));
            }
            doc.push_str("\n---\n\n");
        }

        // Sections
        for sec in &self.sections {
            doc.push_str(&sec.render_markdown_at_depth(2));
            doc.push('\n');
        }

        // Footer
        doc.push_str("---\n\n");
        doc.push_str("_Report generated by CABER — Coalgebraic Audit & Behavioral Evaluation Reporter._\n");

        doc
    }

    /// Render the report as a JSON string (pretty-printed) suitable for
    /// consumption by dashboards and visualisation tools.
    pub fn render_json(&self) -> String {
        // Build a serializable view that contains everything.
        #[derive(serde::Serialize)]
        struct JsonReport<'a> {
            title: &'a str,
            metadata: &'a Option<ReportMetadata>,
            sections: &'a [ReportSection],
        }

        let jr = JsonReport {
            title: &self.config.title,
            metadata: &self.metadata,
            sections: &self.sections,
        };

        serde_json::to_string_pretty(&jr).unwrap_or_else(|e| {
            format!("{{\"error\": \"serialization failed: {}\"}}", e)
        })
    }

    /// Render the report as plain text (no Markdown formatting).
    pub fn render_text(&self) -> String {
        let mut doc = String::with_capacity(8192);

        let title_border: String = std::iter::repeat('=')
            .take(self.config.title.len())
            .collect();
        doc.push_str(&format!("{}\n{}\n\n", self.config.title, title_border));

        if let Some(ref meta) = self.metadata {
            doc.push_str(&format!("Model:             {} (v{})\n", meta.model_id, meta.model_version));
            doc.push_str(&format!("Audit date:        {}\n", meta.audit_date));
            doc.push_str(&format!("Auditor:           {}\n", meta.auditor));
            doc.push_str(&format!("Framework version: {}\n", meta.framework_version));
            doc.push_str(&format!("Total queries:     {}\n", meta.total_queries));
            doc.push_str(&format!(
                "Duration:          {:.2} s\n",
                meta.audit_duration_seconds
            ));
            doc.push('\n');
        }

        for sec in &self.sections {
            doc.push_str(&sec.render_text_at_depth(0));
            doc.push('\n');
        }

        doc.push_str("Report generated by CABER.\n");
        doc
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Create a URL-safe slug from a section title for Markdown anchor links.
fn slug(title: &str) -> String {
    title
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Fixture helpers --------------------------------------------------

    fn sample_config() -> ReportConfig {
        ReportConfig {
            title: "Test Report".into(),
            ..Default::default()
        }
    }

    fn sample_metadata() -> ReportMetadata {
        ReportMetadata {
            model_id: "llm-abc".into(),
            model_version: "1.2.3".into(),
            audit_date: "2025-01-15".into(),
            auditor: "tester".into(),
            framework_version: "0.1.0".into(),
            total_queries: 500,
            audit_duration_seconds: 42.5,
        }
    }

    fn sample_properties() -> Vec<PropertyStatus> {
        vec![
            PropertyStatus {
                name: "Safety".into(),
                description: "No harmful outputs".into(),
                satisfied: true,
                satisfaction_degree: 0.97,
                grade: PropertyGrade::Pass,
                confidence: 0.95,
                witness_summary: None,
            },
            PropertyStatus {
                name: "Fairness".into(),
                description: "Equal treatment across groups".into(),
                satisfied: false,
                satisfaction_degree: 0.55,
                grade: PropertyGrade::Fail,
                confidence: 0.88,
                witness_summary: Some("Bias detected in group B".into()),
            },
        ]
    }

    fn sample_regressions() -> Vec<RegressionEntry> {
        vec![
            RegressionEntry {
                property_name: "Safety".into(),
                previous_degree: 0.90,
                current_degree: 0.97,
                delta: 0.07,
                regression_type: RegressionType::Improved,
                severity: Severity::Low,
            },
            RegressionEntry {
                property_name: "Fairness".into(),
                previous_degree: 0.80,
                current_degree: 0.55,
                delta: -0.25,
                regression_type: RegressionType::MajorRegression,
                severity: Severity::Critical,
            },
        ]
    }

    fn sample_executive_summary() -> ExecutiveSummary {
        ExecutiveSummary {
            overall_status: OverallStatus::ConditionalCertification,
            total_properties: 5,
            properties_passed: 4,
            properties_failed: 1,
            average_satisfaction: 0.85,
            key_findings: vec!["Model mostly safe".into()],
            critical_issues: vec!["Fairness regression".into()],
        }
    }

    fn sample_technical_details() -> TechnicalDetails {
        TechnicalDetails {
            automaton_states: 128,
            automaton_transitions: 512,
            pac_epsilon: 0.05,
            pac_delta: 0.01,
            sample_complexity: 2000,
            bisimulation_method: "partition-refinement".into(),
            model_checker_algorithm: "PCTL".into(),
            total_computation_time_ms: 12345.67,
        }
    }

    fn sample_recommendations() -> Vec<Recommendation> {
        vec![
            Recommendation {
                priority: RecommendationPriority::Immediate,
                category: "Fairness".into(),
                title: "Address bias in group B".into(),
                description: "Retrain with balanced data".into(),
                action_items: vec!["Collect more data".into(), "Retrain model".into()],
            },
            Recommendation {
                priority: RecommendationPriority::LongTerm,
                category: "Monitoring".into(),
                title: "Continuous monitoring".into(),
                description: "Set up automated audits".into(),
                action_items: vec!["Deploy pipeline".into()],
            },
        ]
    }

    // -- Tests -----------------------------------------------------------

    #[test]
    fn test_report_config_default() {
        let cfg = ReportConfig::default();
        assert!(cfg.include_executive_summary);
        assert!(cfg.include_technical_details);
        assert!(cfg.include_recommendations);
        assert!(!cfg.include_raw_data);
        assert_eq!(cfg.max_witness_length, 500);
    }

    #[test]
    fn test_new_report_is_empty() {
        let r = AuditReport::new(sample_config());
        assert!(r.sections().is_empty());
        assert!(r.metadata.is_none());
    }

    #[test]
    fn test_set_metadata() {
        let mut r = AuditReport::new(sample_config());
        r.set_metadata(sample_metadata());
        let meta = r.metadata.as_ref().unwrap();
        assert_eq!(meta.model_id, "llm-abc");
        assert_eq!(meta.total_queries, 500);
    }

    #[test]
    fn test_add_property_results_creates_section() {
        let mut r = AuditReport::new(sample_config());
        r.add_property_results(sample_properties());
        assert_eq!(r.sections().len(), 1);
        assert_eq!(r.sections()[0].section_type, SectionType::PropertyResults);
        assert_eq!(r.sections()[0].subsections.len(), 2);
    }

    #[test]
    fn test_add_regression_summary_creates_section() {
        let mut r = AuditReport::new(sample_config());
        r.add_regression_summary(sample_regressions());
        assert_eq!(r.sections().len(), 1);
        assert_eq!(
            r.sections()[0].section_type,
            SectionType::RegressionSummary
        );
        assert!(r.sections()[0].content.contains("critical"));
    }

    #[test]
    fn test_executive_summary_pass_rate() {
        let es = sample_executive_summary();
        let rate = es.pass_rate();
        assert!((rate - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_executive_summary_zero_total() {
        let es = ExecutiveSummary {
            overall_status: OverallStatus::InsufficientData,
            total_properties: 0,
            properties_passed: 0,
            properties_failed: 0,
            average_satisfaction: 0.0,
            key_findings: vec![],
            critical_issues: vec![],
        };
        assert_eq!(es.pass_rate(), 0.0);
    }

    #[test]
    fn test_render_markdown_contains_title_and_sections() {
        let mut r = AuditReport::new(sample_config());
        r.set_metadata(sample_metadata());
        r.add_executive_summary(sample_executive_summary());
        r.add_property_results(sample_properties());
        r.add_recommendations(sample_recommendations());

        let md = r.render_markdown();
        assert!(md.contains("# Test Report"));
        assert!(md.contains("Executive Summary"));
        assert!(md.contains("Property Results"));
        assert!(md.contains("Recommendations"));
        assert!(md.contains("llm-abc"));
    }

    #[test]
    fn test_render_json_is_valid() {
        let mut r = AuditReport::new(sample_config());
        r.set_metadata(sample_metadata());
        r.add_property_results(sample_properties());

        let json_str = r.render_json();
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed["title"], "Test Report");
        assert!(parsed["sections"].is_array());
    }

    #[test]
    fn test_render_text_contains_metadata() {
        let mut r = AuditReport::new(sample_config());
        r.set_metadata(sample_metadata());
        r.add_custom_section("Notes", "Some custom notes here.");

        let txt = r.render_text();
        assert!(txt.contains("Test Report"));
        assert!(txt.contains("llm-abc"));
        assert!(txt.contains("Some custom notes here."));
    }

    #[test]
    fn test_property_grade_from_degree() {
        assert_eq!(PropertyStatus::grade_from_degree(1.0), PropertyGrade::Pass);
        assert_eq!(
            PropertyStatus::grade_from_degree(0.85),
            PropertyGrade::ConditionalPass
        );
        assert_eq!(
            PropertyStatus::grade_from_degree(0.65),
            PropertyGrade::Marginal
        );
        assert_eq!(PropertyStatus::grade_from_degree(0.3), PropertyGrade::Fail);
        assert_eq!(
            PropertyStatus::grade_from_degree(f64::NAN),
            PropertyGrade::Unknown
        );
        assert_eq!(
            PropertyStatus::grade_from_degree(-1.0),
            PropertyGrade::Unknown
        );
    }

    #[test]
    fn test_helper_functions() {
        assert_eq!(format_percentage(0.856), "85.6%");
        assert_eq!(
            format_confidence_interval(0.80, 0.95),
            "[80.0%, 95.0%]"
        );
        assert!(matches!(severity_from_delta(-0.30), Severity::Critical));
        assert!(matches!(severity_from_delta(-0.06), Severity::Medium));
        assert!(matches!(severity_from_delta(-0.01), Severity::Low));
        assert!(matches!(
            regression_type_from_delta(0.10),
            RegressionType::Improved
        ));
        assert!(matches!(
            regression_type_from_delta(0.0),
            RegressionType::Stable
        ));
        assert!(matches!(
            regression_type_from_delta(-0.05),
            RegressionType::MinorRegression
        ));
        assert!(matches!(
            regression_type_from_delta(-0.20),
            RegressionType::MajorRegression
        ));
        assert!(matches!(
            regression_type_from_delta(-0.50),
            RegressionType::NewFailure
        ));
    }

    #[test]
    fn test_custom_section_and_disabled_flags() {
        let mut cfg = sample_config();
        cfg.include_executive_summary = false;
        cfg.include_technical_details = false;
        cfg.include_recommendations = false;

        let mut r = AuditReport::new(cfg);
        // These should be silently skipped due to config flags
        r.add_executive_summary(sample_executive_summary());
        r.add_technical_details(sample_technical_details());
        r.add_recommendations(sample_recommendations());
        assert_eq!(r.sections().len(), 0);

        r.add_custom_section("Custom", "Hello world");
        assert_eq!(r.sections().len(), 1);
        assert_eq!(r.sections()[0].section_type, SectionType::Custom);
        assert_eq!(r.sections()[0].content, "Hello world");
    }

    #[test]
    fn test_generate_property_table_empty() {
        let table = generate_property_table_markdown(&[]);
        assert!(table.contains("No properties evaluated"));
    }

    #[test]
    fn test_generate_regression_table_has_rows() {
        let table = generate_regression_table_markdown(&sample_regressions());
        assert!(table.contains("Safety"));
        assert!(table.contains("Fairness"));
        assert!(table.contains("Delta"));
    }

    #[test]
    fn test_technical_details_render() {
        let td = sample_technical_details();
        let rendered = td.render();
        assert!(rendered.contains("128"));
        assert!(rendered.contains("partition-refinement"));
        assert!(rendered.contains("12345.67"));
    }

    #[test]
    fn test_status_icon() {
        let pass = PropertyStatus {
            name: "p".into(),
            description: "d".into(),
            satisfied: true,
            satisfaction_degree: 1.0,
            grade: PropertyGrade::Pass,
            confidence: 1.0,
            witness_summary: None,
        };
        assert_eq!(pass.status_icon(), "✅");

        let fail = PropertyStatus {
            grade: PropertyGrade::Fail,
            ..pass.clone()
        };
        assert_eq!(fail.status_icon(), "❌");

        let warn = PropertyStatus {
            grade: PropertyGrade::ConditionalPass,
            ..pass.clone()
        };
        assert_eq!(warn.status_icon(), "⚠️");
    }

    #[test]
    fn test_full_report_round_trip() {
        let mut r = AuditReport::new(sample_config());
        r.set_metadata(sample_metadata());
        r.add_executive_summary(sample_executive_summary());
        r.add_property_results(sample_properties());
        r.add_regression_summary(sample_regressions());
        r.add_technical_details(sample_technical_details());
        r.add_recommendations(sample_recommendations());
        r.add_custom_section("Appendix", "Extra data here.");

        // Verify section count (exec summary + properties + regression + tech + recommendations + custom)
        assert_eq!(r.sections().len(), 6);

        // All three renderers should produce non-empty output
        let md = r.render_markdown();
        let json = r.render_json();
        let txt = r.render_text();

        assert!(md.len() > 500);
        assert!(json.len() > 500);
        assert!(txt.len() > 200);

        // JSON should be parseable
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["sections"].as_array().unwrap().len(), 6);
    }
}
