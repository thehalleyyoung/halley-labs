//! Human-readable explanations and counterfactual reasoning for repairs.
//!
//! Produces structured explanations that help operators understand *why* a
//! repair is proposed and *what would happen* without it.

use serde::{Deserialize, Serialize};

use super::synthesizer::RiskyPathInfo;
use super::{RepairAction, RepairActionType, RepairPlan};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A complete explanation for a repair plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Explanation {
    pub summary: String,
    pub details: Vec<ExplanationDetail>,
    pub counterfactuals: Vec<Counterfactual>,
    pub recommendations: Vec<String>,
}

/// Explanation for a single parameter change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationDetail {
    pub change: String,
    pub reason: String,
    pub impact: String,
    pub confidence: f64,
}

/// A counterfactual scenario: what would happen with vs. without the repair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Counterfactual {
    pub scenario: String,
    pub original_behavior: String,
    pub repaired_behavior: String,
    pub key_difference: String,
}

// ---------------------------------------------------------------------------
// ExplanationGenerator
// ---------------------------------------------------------------------------

/// Generates human-readable explanations for repair plans.
#[derive(Debug, Clone, Default)]
pub struct ExplanationGenerator;

impl ExplanationGenerator {
    pub fn new() -> Self {
        Self
    }

    /// Generate a full explanation for a repair plan given the risks it
    /// addresses.
    pub fn generate(
        &self,
        repair: &RepairPlan,
        risks: &[RiskyPathInfo],
    ) -> Explanation {
        let mut details = Vec::new();
        let mut counterfactuals = Vec::new();
        let mut recommendations = Vec::new();

        for action in &repair.actions {
            let detail = match &action.action_type {
                RepairActionType::ReduceRetries { from, to } => {
                    let edge_str = action
                        .edge
                        .as_ref()
                        .map(|(s, t)| format!("{}->{}", s, t))
                        .unwrap_or_else(|| action.service.clone());
                    self.explain_retry_reduction(&edge_str, *from, *to)
                }
                RepairActionType::AdjustTimeout { from_ms, to_ms } => {
                    let edge_str = action
                        .edge
                        .as_ref()
                        .map(|(s, t)| format!("{}->{}", s, t))
                        .unwrap_or_else(|| action.service.clone());
                    self.explain_timeout_adjustment(&edge_str, *from_ms, *to_ms)
                }
                RepairActionType::AddCircuitBreaker { threshold } => ExplanationDetail {
                    change: format!(
                        "Add circuit breaker on {} (threshold={})",
                        action.service, threshold
                    ),
                    reason: "Prevents cascading failures by stopping requests when error rate is high.".into(),
                    impact: format!(
                        "Requests to {} will be rejected after {} consecutive failures.",
                        action.service, threshold
                    ),
                    confidence: 0.85,
                },
                RepairActionType::AddRateLimit { rps } => ExplanationDetail {
                    change: format!(
                        "Add rate limit on {} ({:.0} req/s)",
                        action.service, rps
                    ),
                    reason: "Limits request throughput to prevent overload during cascade.".into(),
                    impact: format!(
                        "Traffic to {} capped at {:.0} requests per second.",
                        action.service, rps
                    ),
                    confidence: 0.80,
                },
                RepairActionType::IncreaseCapacity { from, to } => ExplanationDetail {
                    change: format!(
                        "Increase capacity of {} from {:.0} to {:.0}",
                        action.service, from, to
                    ),
                    reason: "Additional capacity absorbs retry-amplified load.".into(),
                    impact: format!(
                        "{} can handle {:.0}% more load after scaling.",
                        action.service,
                        ((to - from) / from.max(1.0)) * 100.0
                    ),
                    confidence: 0.75,
                },
            };
            details.push(detail);
        }

        // Generate counterfactuals for each risk.
        for risk in risks {
            let cf = self.generate_counterfactual(risk, repair);
            counterfactuals.push(cf);
        }

        // Recommendations.
        if repair.actions.iter().any(|a| matches!(a.action_type, RepairActionType::ReduceRetries { .. })) {
            recommendations.push(
                "Monitor error rates after reducing retries; compensate with circuit breakers if needed.".into(),
            );
        }
        if repair.actions.iter().any(|a| matches!(a.action_type, RepairActionType::AdjustTimeout { .. })) {
            recommendations.push(
                "Verify downstream SLAs still hold with adjusted timeouts.".into(),
            );
        }
        if repair.total_deviation > 5.0 {
            recommendations.push(
                "This repair involves significant parameter changes; consider staged rollout.".into(),
            );
        }
        if recommendations.is_empty() {
            recommendations.push("No additional recommendations.".into());
        }

        let n_amp = risks.iter().filter(|r| r.has_amplification_violation()).count();
        let n_to = risks.iter().filter(|r| r.has_timeout_violation()).count();
        let summary = format!(
            "Repair plan with {} action(s) addressing {} amplification and {} timeout violation(s). \
             Total deviation: {:.2}.",
            repair.actions.len(),
            n_amp,
            n_to,
            repair.total_deviation,
        );

        Explanation {
            summary,
            details,
            counterfactuals,
            recommendations,
        }
    }

    /// Explain a retry-count reduction on a specific edge.
    pub fn explain_retry_reduction(
        &self,
        edge: &str,
        old: u32,
        new: u32,
    ) -> ExplanationDetail {
        let reduction = old.saturating_sub(new);
        let pct = if old > 0 {
            (reduction as f64 / old as f64) * 100.0
        } else {
            0.0
        };
        ExplanationDetail {
            change: format!("Reduce retries on {} from {} to {}", edge, old, new),
            reason: format!(
                "Each retry multiplies downstream load; reducing by {} ({:.0}%) lowers the \
                 retry-amplification factor on this edge from {} to {}.",
                reduction, pct, 1 + old, 1 + new
            ),
            impact: format!(
                "Under failure, downstream load from this edge drops by {:.0}%.",
                pct
            ),
            confidence: if reduction <= 2 { 0.95 } else { 0.85 },
        }
    }

    /// Explain a timeout adjustment on a specific edge.
    pub fn explain_timeout_adjustment(
        &self,
        edge: &str,
        old: u64,
        new: u64,
    ) -> ExplanationDetail {
        let direction = if new > old { "Increased" } else { "Decreased" };
        let diff = (new as i64 - old as i64).unsigned_abs();
        ExplanationDetail {
            change: format!(
                "{} timeout on {} from {}ms to {}ms",
                direction, edge, old, new
            ),
            reason: format!(
                "The cumulative timeout chain through this edge contributes to \
                 deadline violations. Adjusting by {}ms brings the path within budget.",
                diff
            ),
            impact: format!(
                "End-to-end latency budget on paths through {} changes by {}ms.",
                edge, diff
            ),
            confidence: 0.90,
        }
    }

    /// Generate a counterfactual for a single risk.
    pub fn generate_counterfactual(
        &self,
        risk: &RiskyPathInfo,
        repair: &RepairPlan,
    ) -> Counterfactual {
        let path_str = risk.path.join(" → ");
        let scenario = format!("Path: {}", path_str);

        let original_behavior = if risk.has_amplification_violation() {
            format!(
                "Retry amplification of {:.1}x exceeds threshold {:.1}x; a single failure \
                 could amplify into a cascade overwhelming downstream services.",
                risk.amplification, risk.threshold
            )
        } else if risk.has_timeout_violation() {
            format!(
                "Cumulative timeout of {}ms exceeds deadline of {}ms; requests may \
                 time out before completing the full call chain.",
                risk.timeout_ms, risk.deadline_ms
            )
        } else {
            "No violation detected on this path.".to_string()
        };

        let relevant_changes: Vec<String> = repair
            .actions
            .iter()
            .filter(|a| {
                if let Some((ref s, ref t)) = a.edge {
                    risk.path.windows(2).any(|w| w[0] == *s && w[1] == *t)
                } else {
                    false
                }
            })
            .map(|a| a.description.clone())
            .collect();

        let repaired_behavior = if relevant_changes.is_empty() {
            "No changes target this path directly.".to_string()
        } else {
            format!(
                "After applying [{}], the path metrics are brought within safe thresholds.",
                relevant_changes.join("; ")
            )
        };

        let key_difference = if risk.has_amplification_violation() {
            format!(
                "Amplification reduced from {:.1}x to within {:.1}x threshold.",
                risk.amplification, risk.threshold
            )
        } else if risk.has_timeout_violation() {
            format!(
                "Timeout reduced from {}ms to within {}ms deadline.",
                risk.timeout_ms, risk.deadline_ms
            )
        } else {
            "Path was already safe.".to_string()
        };

        Counterfactual {
            scenario,
            original_behavior,
            repaired_behavior,
            key_difference,
        }
    }

    /// Render an explanation as Markdown.
    pub fn format_markdown(&self, explanation: &Explanation) -> String {
        let mut md = String::new();
        md.push_str("# Repair Explanation\n\n");
        md.push_str(&format!("**Summary:** {}\n\n", explanation.summary));

        md.push_str("## Changes\n\n");
        for (i, detail) in explanation.details.iter().enumerate() {
            md.push_str(&format!("### {}. {}\n\n", i + 1, detail.change));
            md.push_str(&format!("- **Reason:** {}\n", detail.reason));
            md.push_str(&format!("- **Impact:** {}\n", detail.impact));
            md.push_str(&format!("- **Confidence:** {:.0}%\n\n", detail.confidence * 100.0));
        }

        if !explanation.counterfactuals.is_empty() {
            md.push_str("## Counterfactuals\n\n");
            for cf in &explanation.counterfactuals {
                md.push_str(&format!("### {}\n\n", cf.scenario));
                md.push_str(&format!("- **Without repair:** {}\n", cf.original_behavior));
                md.push_str(&format!("- **With repair:** {}\n", cf.repaired_behavior));
                md.push_str(&format!("- **Key difference:** {}\n\n", cf.key_difference));
            }
        }

        if !explanation.recommendations.is_empty() {
            md.push_str("## Recommendations\n\n");
            for rec in &explanation.recommendations {
                md.push_str(&format!("- {}\n", rec));
            }
        }

        md
    }

    /// Render an explanation as plain text.
    pub fn format_plain(&self, explanation: &Explanation) -> String {
        let mut text = String::new();
        text.push_str("REPAIR EXPLANATION\n");
        text.push_str(&"=".repeat(60));
        text.push('\n');
        text.push_str(&format!("Summary: {}\n\n", explanation.summary));

        text.push_str("CHANGES:\n");
        for (i, detail) in explanation.details.iter().enumerate() {
            text.push_str(&format!("  {}. {}\n", i + 1, detail.change));
            text.push_str(&format!("     Reason: {}\n", detail.reason));
            text.push_str(&format!("     Impact: {}\n", detail.impact));
            text.push_str(&format!("     Confidence: {:.0}%\n\n", detail.confidence * 100.0));
        }

        if !explanation.counterfactuals.is_empty() {
            text.push_str("COUNTERFACTUALS:\n");
            for cf in &explanation.counterfactuals {
                text.push_str(&format!("  Scenario: {}\n", cf.scenario));
                text.push_str(&format!("    Without repair: {}\n", cf.original_behavior));
                text.push_str(&format!("    With repair: {}\n", cf.repaired_behavior));
                text.push_str(&format!("    Key difference: {}\n\n", cf.key_difference));
            }
        }

        if !explanation.recommendations.is_empty() {
            text.push_str("RECOMMENDATIONS:\n");
            for rec in &explanation.recommendations {
                text.push_str(&format!("  - {}\n", rec));
            }
        }

        text
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_repair() -> RepairPlan {
        let mut plan = RepairPlan::default();
        plan.feasible = true;
        plan.add_action(RepairAction::reduce_retries("gateway", "auth", 5, 2));
        plan.add_action(RepairAction::adjust_timeout("auth", "db", 5000, 3000));
        plan
    }

    fn sample_risks() -> Vec<RiskyPathInfo> {
        vec![
            RiskyPathInfo {
                path: vec!["gateway".into(), "auth".into(), "db".into()],
                amplification: 24.0,
                timeout_ms: 8000,
                threshold: 10.0,
                deadline_ms: 6000,
            },
        ]
    }

    #[test]
    fn test_generate_explanation_structure() {
        let gen = ExplanationGenerator::new();
        let expl = gen.generate(&sample_repair(), &sample_risks());
        assert!(!expl.summary.is_empty());
        assert_eq!(expl.details.len(), 2);
        assert_eq!(expl.counterfactuals.len(), 1);
        assert!(!expl.recommendations.is_empty());
    }

    #[test]
    fn test_explain_retry_reduction() {
        let gen = ExplanationGenerator::new();
        let detail = gen.explain_retry_reduction("A->B", 5, 2);
        assert!(detail.change.contains("A->B"));
        assert!(detail.change.contains("5"));
        assert!(detail.change.contains("2"));
        assert!(detail.confidence > 0.0 && detail.confidence <= 1.0);
    }

    #[test]
    fn test_explain_timeout_adjustment_decrease() {
        let gen = ExplanationGenerator::new();
        let detail = gen.explain_timeout_adjustment("X->Y", 5000, 3000);
        assert!(detail.change.contains("Decreased"));
        assert!(detail.change.contains("5000"));
        assert!(detail.change.contains("3000"));
    }

    #[test]
    fn test_explain_timeout_adjustment_increase() {
        let gen = ExplanationGenerator::new();
        let detail = gen.explain_timeout_adjustment("X->Y", 1000, 3000);
        assert!(detail.change.contains("Increased"));
    }

    #[test]
    fn test_generate_counterfactual_amplification() {
        let gen = ExplanationGenerator::new();
        let risk = &sample_risks()[0];
        let cf = gen.generate_counterfactual(risk, &sample_repair());
        assert!(cf.scenario.contains("gateway"));
        assert!(cf.original_behavior.contains("amplification"));
        assert!(cf.key_difference.contains("Amplification"));
    }

    #[test]
    fn test_format_markdown_contains_headers() {
        let gen = ExplanationGenerator::new();
        let expl = gen.generate(&sample_repair(), &sample_risks());
        let md = gen.format_markdown(&expl);
        assert!(md.contains("# Repair Explanation"));
        assert!(md.contains("## Changes"));
        assert!(md.contains("## Counterfactuals"));
        assert!(md.contains("## Recommendations"));
    }

    #[test]
    fn test_format_plain_contains_sections() {
        let gen = ExplanationGenerator::new();
        let expl = gen.generate(&sample_repair(), &sample_risks());
        let text = gen.format_plain(&expl);
        assert!(text.contains("REPAIR EXPLANATION"));
        assert!(text.contains("CHANGES:"));
        assert!(text.contains("COUNTERFACTUALS:"));
        assert!(text.contains("RECOMMENDATIONS:"));
    }

    #[test]
    fn test_empty_plan_explanation() {
        let gen = ExplanationGenerator::new();
        let plan = RepairPlan::default();
        let expl = gen.generate(&plan, &[]);
        assert!(expl.details.is_empty());
        assert!(expl.counterfactuals.is_empty());
        assert!(!expl.recommendations.is_empty());
    }

    #[test]
    fn test_confidence_range() {
        let gen = ExplanationGenerator::new();
        let expl = gen.generate(&sample_repair(), &sample_risks());
        for detail in &expl.details {
            assert!(detail.confidence > 0.0 && detail.confidence <= 1.0);
        }
    }

    #[test]
    fn test_counterfactual_timeout_violation() {
        let gen = ExplanationGenerator::new();
        let risk = RiskyPathInfo {
            path: vec!["A".into(), "B".into()],
            amplification: 1.0,
            timeout_ms: 10000,
            threshold: 100.0,
            deadline_ms: 5000,
        };
        let cf = gen.generate_counterfactual(&risk, &RepairPlan::default());
        assert!(cf.original_behavior.contains("timeout"));
        assert!(cf.key_difference.contains("Timeout"));
    }
}
