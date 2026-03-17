use anyhow::{Context, Result};
use std::path::PathBuf;

use regsynth_pareto::ComplianceStrategy;
use regsynth_pareto::ParetoFrontier;
use regsynth_planner::*;

use crate::config::AppConfig;
use crate::output::{OutputFormatter, PhaseDisplayEntry, TaskDisplayEntry};

/// Intermediate representation for frontier data loaded from file.
#[derive(Debug, serde::Deserialize)]
struct FrontierInput {
    frontier: ParetoFrontier<ComplianceStrategy>,
}

/// Run the planning command: generate a remediation roadmap from a Pareto strategy.
pub fn run(
    config: &AppConfig,
    formatter: &OutputFormatter,
    input: &PathBuf,
    strategy_index: usize,
    max_parallel: usize,
    start_date_str: Option<&str>,
) -> Result<()> {
    formatter.status("Generating remediation roadmap...");

    let content = std::fs::read_to_string(input)
        .with_context(|| format!("Failed to read {}", input.display()))?;

    let frontier_input: FrontierInput = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse Pareto frontier from {}", input.display()))?;

    let frontier = &frontier_input.frontier;

    let strategies: Vec<&ComplianceStrategy> = frontier.points();
    if strategies.is_empty() {
        anyhow::bail!("Pareto frontier is empty; no strategies to plan from");
    }

    if strategy_index >= strategies.len() {
        anyhow::bail!(
            "Strategy index {} is out of range (frontier has {} strategies)",
            strategy_index,
            strategies.len()
        );
    }

    let strategy = strategies[strategy_index];
    formatter.status(&format!("  Selected strategy: {} (index {})", strategy.name, strategy_index));
    formatter.status(&format!(
        "  Obligations: {} satisfied, {} waived",
        strategy.obligation_entries.len(),
        strategy.waived_obligations.len()
    ));
    formatter.status(&format!("  Total cost: ${:.2}", strategy.total_cost.amount));

    let start_date = parse_start_date(start_date_str)?;

    let roadmap = build_roadmap(strategy, &config.planner, max_parallel, start_date)?;

    // Display roadmap
    let total_days = roadmap.phases.iter().map(|p| p.duration_days()).sum::<i64>() as u32;

    let phase_entries: Vec<PhaseDisplayEntry> = roadmap
        .phases
        .iter()
        .map(|phase| PhaseDisplayEntry {
            name: phase.name.clone(),
            task_count: phase.tasks.len(),
            duration_days: phase.duration_days(),
            tasks: phase
                .tasks
                .iter()
                .map(|t| TaskDisplayEntry {
                    name: t.name.clone(),
                    status: t.status.to_string(),
                    effort_days: t.effort_days,
                    cost: t.cost_estimate,
                })
                .collect(),
        })
        .collect();

    formatter.write_roadmap(&phase_entries, total_days, roadmap.total_cost())?;

    // Gantt chart
    let gantt = roadmap.gantt_data();
    if !gantt.is_empty() {
        formatter.status("\nGantt Chart:");
        let headers = vec!["Task", "Phase", "Start Day", "Duration", "Critical"];
        let rows: Vec<Vec<String>> = gantt
            .iter()
            .map(|g| {
                vec![
                    g.task_name.clone(),
                    g.phase_name.clone(),
                    format!("{}", g.start_day),
                    format!("{:.0}", g.duration_days),
                    if g.is_critical { "★" } else { " " }.into(),
                ]
            })
            .collect();
        formatter.write_table(&headers, &rows)?;
    }

    // Write full output
    let output = serde_json::json!({
        "strategy": strategy.name,
        "strategy_index": strategy_index,
        "roadmap": roadmap,
        "total_days": total_days,
        "total_cost": roadmap.total_cost(),
        "phases": roadmap.phases.len(),
        "tasks": roadmap.phases.iter().map(|p| p.tasks.len()).sum::<usize>(),
    });
    formatter.write_value(&output)?;

    Ok(())
}

/// Build a roadmap from a selected compliance strategy.
fn build_roadmap(
    strategy: &regsynth_pareto::ComplianceStrategy,
    planner_config: &crate::config::PlannerConfig,
    _max_parallel: usize,
    start_date: chrono::NaiveDate,
) -> Result<ComplianceRoadmap> {
    let mut roadmap = ComplianceRoadmap::new(format!("Roadmap for {}", strategy.name));

    let obligations = &strategy.obligation_entries;
    let total = obligations.len();

    if total == 0 {
        return Ok(roadmap);
    }

    // Split obligations into phases based on effort/priority
    let phase_size = (total / 3).max(1);
    let phase_specs: Vec<(&str, &str, usize, usize)> = vec![
        ("phase-1", "Immediate Compliance", 0, phase_size.min(total)),
        ("phase-2", "Secondary Compliance", phase_size.min(total), (2 * phase_size).min(total)),
        ("phase-3", "Long-term Compliance", (2 * phase_size).min(total), total),
    ];

    let mut current_date = start_date;
    for (phase_id, phase_name, start_idx, end_idx) in phase_specs {
        if start_idx >= end_idx {
            continue;
        }

        let phase_obligations = &obligations[start_idx..end_idx];
        let phase_duration = phase_obligations.len() as i64 * planner_config.default_effort_days as i64;
        let phase_end = current_date + chrono::Duration::days(phase_duration);

        let mut phase = RoadmapPhase::new(phase_id, phase_name, current_date, phase_end);

        for (i, entry) in phase_obligations.iter().enumerate() {
            let task_cost = entry
                .estimated_cost
                .as_ref()
                .map(|c| c.amount)
                .unwrap_or(planner_config.default_effort_days * 100.0);

            let task = RoadmapTask::new(
                format!("{}-task-{}", phase_id, i),
                entry.name.clone(),
            )
            .with_effort(planner_config.default_effort_days)
            .with_cost(task_cost)
            .with_obligation(entry.obligation_id.0.to_string())
            .with_description(format!("Implement compliance for {}", entry.name));

            phase.add_task(task);
        }

        current_date = phase_end;
        roadmap.add_phase(phase);
    }

    Ok(roadmap)
}

fn parse_start_date(date_str: Option<&str>) -> Result<chrono::NaiveDate> {
    match date_str {
        Some(s) => chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d")
            .with_context(|| format!("Invalid date format '{}', expected YYYY-MM-DD", s)),
        None => Ok(chrono::Utc::now().date_naive()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use regsynth_types::*;

    #[test]
    fn test_parse_start_date_valid() {
        let d = parse_start_date(Some("2025-01-15")).unwrap();
        assert_eq!(d.to_string(), "2025-01-15");
    }

    #[test]
    fn test_parse_start_date_none() {
        let d = parse_start_date(None).unwrap();
        assert!(d.to_string().starts_with("20")); // Starts with 20xx
    }

    #[test]
    fn test_parse_start_date_invalid() {
        assert!(parse_start_date(Some("not-a-date")).is_err());
    }

    #[test]
    fn test_build_roadmap_empty() {
        let strategy = regsynth_pareto::ComplianceStrategy::new("empty", vec![]);
        let config = crate::config::PlannerConfig::default();
        let date = chrono::NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
        let roadmap = build_roadmap(&strategy, &config, 4, date).unwrap();
        assert!(roadmap.phases.is_empty());
    }

    #[test]
    fn test_build_roadmap_with_entries() {
        let entries = vec![
            regsynth_pareto::ObligationEntry {
                obligation_id: Id::new(),
                name: "obl-1".into(),
                estimated_cost: Some(Cost { amount: 1000.0, currency: "USD".into() }),
            },
            regsynth_pareto::ObligationEntry {
                obligation_id: Id::new(),
                name: "obl-2".into(),
                estimated_cost: Some(Cost { amount: 2000.0, currency: "USD".into() }),
            },
        ];
        let strategy = regsynth_pareto::ComplianceStrategy::new("test", entries);
        let config = crate::config::PlannerConfig::default();
        let date = chrono::NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
        let roadmap = build_roadmap(&strategy, &config, 4, date).unwrap();
        assert!(!roadmap.phases.is_empty());
        let task_count: usize = roadmap.phases.iter().map(|p| p.tasks.len()).sum();
        assert_eq!(task_count, 2);
    }
}
