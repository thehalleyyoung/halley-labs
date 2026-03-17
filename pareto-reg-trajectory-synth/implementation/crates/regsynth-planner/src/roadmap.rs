use std::collections::{HashMap, HashSet, VecDeque};

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::PlannerError;
use crate::PlannerResult;

// ─── Task Status ────────────────────────────────────────────────────────────

/// Status of a roadmap task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskStatus {
    NotStarted,
    InProgress,
    Completed,
    Blocked,
    Deferred,
}

impl std::fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotStarted => write!(f, "Not Started"),
            Self::InProgress => write!(f, "In Progress"),
            Self::Completed => write!(f, "Completed"),
            Self::Blocked => write!(f, "Blocked"),
            Self::Deferred => write!(f, "Deferred"),
        }
    }
}

// ─── Roadmap Task ───────────────────────────────────────────────────────────

/// A single task within a roadmap phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoadmapTask {
    pub id: String,
    pub name: String,
    pub description: String,
    pub obligation_id: Option<String>,
    pub effort_days: f64,
    pub cost_estimate: f64,
    pub assigned_resources: Vec<String>,
    pub dependencies: Vec<String>,
    pub status: TaskStatus,
    pub start_date: Option<NaiveDate>,
    pub end_date: Option<NaiveDate>,
    pub priority: u32,
}

impl RoadmapTask {
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            obligation_id: None,
            effort_days: 0.0,
            cost_estimate: 0.0,
            assigned_resources: Vec::new(),
            dependencies: Vec::new(),
            status: TaskStatus::NotStarted,
            start_date: None,
            end_date: None,
            priority: 0,
        }
    }

    pub fn with_effort(mut self, days: f64) -> Self {
        self.effort_days = days;
        self
    }

    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost_estimate = cost;
        self
    }

    pub fn with_obligation(mut self, obligation_id: impl Into<String>) -> Self {
        self.obligation_id = Some(obligation_id.into());
        self
    }

    pub fn with_dependency(mut self, dep_id: impl Into<String>) -> Self {
        self.dependencies.push(dep_id.into());
        self
    }

    pub fn with_resource(mut self, resource: impl Into<String>) -> Self {
        self.assigned_resources.push(resource.into());
        self
    }

    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn with_dates(mut self, start: NaiveDate, end: NaiveDate) -> Self {
        self.start_date = Some(start);
        self.end_date = Some(end);
        self
    }

    /// Duration in calendar days (from dates or effort estimate).
    pub fn duration_days(&self) -> f64 {
        match (self.start_date, self.end_date) {
            (Some(s), Some(e)) => (e - s).num_days() as f64,
            _ => self.effort_days,
        }
    }
}

// ─── Gantt Entry ────────────────────────────────────────────────────────────

/// A single entry in a Gantt chart representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GanttEntry {
    pub task_id: String,
    pub task_name: String,
    pub phase_name: String,
    pub start_day: i64,
    pub duration_days: f64,
    pub dependencies: Vec<String>,
    pub resource: String,
    pub is_critical: bool,
}

// ─── Roadmap Phase ──────────────────────────────────────────────────────────

/// A phase within a compliance roadmap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoadmapPhase {
    pub id: String,
    pub name: String,
    pub description: String,
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
    pub tasks: Vec<RoadmapTask>,
    pub milestones: Vec<String>,
    pub budget: f64,
}

impl RoadmapPhase {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        start: NaiveDate,
        end: NaiveDate,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            start_date: start,
            end_date: end,
            tasks: Vec::new(),
            milestones: Vec::new(),
            budget: 0.0,
        }
    }

    pub fn add_task(&mut self, task: RoadmapTask) {
        self.tasks.push(task);
    }

    pub fn add_milestone(&mut self, milestone: impl Into<String>) {
        self.milestones.push(milestone.into());
    }

    pub fn duration_days(&self) -> i64 {
        (self.end_date - self.start_date).num_days()
    }

    pub fn total_effort(&self) -> f64 {
        self.tasks.iter().map(|t| t.effort_days).sum()
    }

    pub fn total_cost(&self) -> f64 {
        self.tasks.iter().map(|t| t.cost_estimate).sum()
    }

    pub fn completion_percentage(&self) -> f64 {
        if self.tasks.is_empty() {
            return 0.0;
        }
        let completed = self.tasks.iter()
            .filter(|t| t.status == TaskStatus::Completed)
            .count();
        (completed as f64 / self.tasks.len() as f64) * 100.0
    }
}

// ─── Compliance Roadmap ─────────────────────────────────────────────────────

/// Complete compliance implementation roadmap with phased plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceRoadmap {
    pub id: String,
    pub name: String,
    pub description: String,
    pub created_at: NaiveDate,
    pub phases: Vec<RoadmapPhase>,
    pub strategy_id: Option<String>,
}

impl ComplianceRoadmap {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.into(),
            description: String::new(),
            created_at: chrono::Utc::now().date_naive(),
            phases: Vec::new(),
            strategy_id: None,
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn with_strategy(mut self, strategy_id: impl Into<String>) -> Self {
        self.strategy_id = Some(strategy_id.into());
        self
    }

    /// Add a phase to the roadmap, maintaining chronological order.
    pub fn add_phase(&mut self, phase: RoadmapPhase) {
        self.phases.push(phase);
        self.phases.sort_by_key(|p| p.start_date);
    }

    /// Add a task to a specific phase by phase ID.
    pub fn add_task(&mut self, phase_id: &str, task: RoadmapTask) -> PlannerResult<()> {
        let phase = self.phases.iter_mut()
            .find(|p| p.id == phase_id)
            .ok_or_else(|| PlannerError::PhaseNotFound(phase_id.to_string()))?;
        phase.add_task(task);
        Ok(())
    }

    /// Validate the roadmap: no circular dependencies, phases don't overlap negatively,
    /// resource feasibility.
    pub fn validate(&self) -> PlannerResult<Vec<String>> {
        let mut warnings = Vec::new();

        // Collect all task IDs
        let all_task_ids: HashSet<String> = self.phases.iter()
            .flat_map(|p| p.tasks.iter())
            .map(|t| t.id.clone())
            .collect();

        // Check for dangling dependencies
        for phase in &self.phases {
            for task in &phase.tasks {
                for dep in &task.dependencies {
                    if !all_task_ids.contains(dep) {
                        warnings.push(format!(
                            "Task '{}' depends on unknown task '{}'", task.id, dep
                        ));
                    }
                }
            }
        }

        // Detect circular dependencies using Kahn's algorithm
        let all_tasks: Vec<&RoadmapTask> = self.phases.iter()
            .flat_map(|p| p.tasks.iter())
            .collect();

        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        for task in &all_tasks {
            in_degree.entry(&task.id).or_insert(0);
            adj.entry(&task.id).or_default();
        }
        for task in &all_tasks {
            for dep in &task.dependencies {
                if all_task_ids.contains(dep) {
                    adj.entry(dep.as_str()).or_default().push(&task.id);
                    *in_degree.entry(&task.id).or_insert(0) += 1;
                }
            }
        }

        let mut queue: VecDeque<&str> = in_degree.iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut visited = 0usize;
        while let Some(node) = queue.pop_front() {
            visited += 1;
            if let Some(neighbors) = adj.get(node) {
                for &neighbor in neighbors {
                    if let Some(deg) = in_degree.get_mut(neighbor) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }

        if visited < all_task_ids.len() {
            return Err(PlannerError::CircularDependency(
                "Circular dependency detected among roadmap tasks".to_string()
            ));
        }

        // Check phase date consistency
        for (i, phase) in self.phases.iter().enumerate() {
            if phase.start_date > phase.end_date {
                warnings.push(format!(
                    "Phase '{}' has start_date after end_date", phase.name
                ));
            }
            if i > 0 && phase.start_date < self.phases[i - 1].end_date {
                warnings.push(format!(
                    "Phase '{}' overlaps with previous phase '{}'",
                    phase.name, self.phases[i - 1].name
                ));
            }
        }

        // Check budget feasibility per phase
        for phase in &self.phases {
            if phase.budget > 0.0 && phase.total_cost() > phase.budget {
                warnings.push(format!(
                    "Phase '{}' cost ({:.0}) exceeds budget ({:.0})",
                    phase.name, phase.total_cost(), phase.budget
                ));
            }
        }

        Ok(warnings)
    }

    /// Total cost across all phases.
    pub fn total_cost(&self) -> f64 {
        self.phases.iter().map(|p| p.total_cost()).sum()
    }

    /// Total effort in person-days across all phases.
    pub fn total_effort(&self) -> f64 {
        self.phases.iter().map(|p| p.total_effort()).sum()
    }

    /// Overall start date (earliest phase start).
    pub fn start_date(&self) -> Option<NaiveDate> {
        self.phases.first().map(|p| p.start_date)
    }

    /// Overall end date (latest phase end).
    pub fn end_date(&self) -> Option<NaiveDate> {
        self.phases.last().map(|p| p.end_date)
    }

    /// Compute the critical path: longest chain of dependent tasks.
    pub fn critical_path(&self) -> Vec<String> {
        let all_tasks: HashMap<String, &RoadmapTask> = self.phases.iter()
            .flat_map(|p| p.tasks.iter())
            .map(|t| (t.id.clone(), t))
            .collect();

        // Build adjacency for forward pass
        let mut successors: HashMap<&str, Vec<&str>> = HashMap::new();
        let mut predecessors: HashMap<&str, Vec<&str>> = HashMap::new();
        for task in all_tasks.values() {
            successors.entry(&task.id).or_default();
            predecessors.entry(&task.id).or_default();
        }
        for task in all_tasks.values() {
            for dep in &task.dependencies {
                if all_tasks.contains_key(dep) {
                    successors.entry(dep.as_str()).or_default().push(&task.id);
                    predecessors.entry(&task.id).or_default().push(dep.as_str());
                }
            }
        }

        // Forward pass: compute earliest start
        let topo = self.topological_order(&all_tasks);
        let mut earliest_finish: HashMap<&str, f64> = HashMap::new();
        for task_id in &topo {
            let task = all_tasks[task_id.as_str()];
            let es = predecessors.get(task_id.as_str())
                .map(|preds| preds.iter()
                    .filter_map(|p| earliest_finish.get(p))
                    .cloned()
                    .fold(0.0f64, f64::max))
                .unwrap_or(0.0);
            earliest_finish.insert(task_id, es + task.duration_days());
        }

        // Find project duration
        let project_duration = earliest_finish.values().cloned().fold(0.0f64, f64::max);

        // Backward pass: compute latest finish
        let mut latest_finish: HashMap<&str, f64> = HashMap::new();
        for task_id in topo.iter().rev() {
            let task = all_tasks[task_id.as_str()];
            let lf = successors.get(task_id.as_str())
                .map(|succs| succs.iter()
                    .filter_map(|s| latest_finish.get(s).map(|&lf| lf - all_tasks[*s].duration_days()))
                    .fold(project_duration, f64::min))
                .unwrap_or(project_duration);
            latest_finish.insert(task_id, lf);
        }

        // Critical tasks: slack == 0
        let mut critical: Vec<String> = Vec::new();
        for task_id in &topo {
            let ef = earliest_finish[task_id.as_str()];
            let lf = latest_finish[task_id.as_str()];
            let slack = lf - ef;
            if slack.abs() < 0.001 {
                critical.push(task_id.clone());
            }
        }
        critical
    }

    /// Generate Gantt chart data.
    pub fn gantt_data(&self) -> Vec<GanttEntry> {
        let project_start = self.start_date().unwrap_or(chrono::Utc::now().date_naive());
        let critical_set: HashSet<String> = self.critical_path().into_iter().collect();

        let mut entries = Vec::new();
        for phase in &self.phases {
            for task in &phase.tasks {
                let start_day = task.start_date
                    .map(|d| (d - project_start).num_days())
                    .unwrap_or_else(|| (phase.start_date - project_start).num_days());

                entries.push(GanttEntry {
                    task_id: task.id.clone(),
                    task_name: task.name.clone(),
                    phase_name: phase.name.clone(),
                    start_day,
                    duration_days: task.duration_days(),
                    dependencies: task.dependencies.clone(),
                    resource: task.assigned_resources.first()
                        .cloned()
                        .unwrap_or_else(|| "unassigned".to_string()),
                    is_critical: critical_set.contains(&task.id),
                });
            }
        }
        entries
    }

    /// Serialize the roadmap to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize a roadmap from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Overall completion percentage.
    pub fn completion_percentage(&self) -> f64 {
        let total_tasks: usize = self.phases.iter().map(|p| p.tasks.len()).sum();
        if total_tasks == 0 {
            return 0.0;
        }
        let completed: usize = self.phases.iter()
            .flat_map(|p| p.tasks.iter())
            .filter(|t| t.status == TaskStatus::Completed)
            .count();
        (completed as f64 / total_tasks as f64) * 100.0
    }

    /// Helper: topological sort of tasks.
    fn topological_order(&self, all_tasks: &HashMap<String, &RoadmapTask>) -> Vec<String> {
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();

        for id in all_tasks.keys() {
            in_degree.entry(id.as_str()).or_insert(0);
            adj.entry(id.as_str()).or_default();
        }
        for task in all_tasks.values() {
            for dep in &task.dependencies {
                if all_tasks.contains_key(dep.as_str()) {
                    adj.entry(dep.as_str()).or_default().push(&task.id);
                    *in_degree.entry(task.id.as_str()).or_insert(0) += 1;
                }
            }
        }

        let mut queue: VecDeque<&str> = in_degree.iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&id, _)| id)
            .collect();
        let mut order = Vec::new();

        while let Some(node) = queue.pop_front() {
            order.push(node.to_string());
            if let Some(neighbors) = adj.get(node) {
                for &neighbor in neighbors {
                    if let Some(deg) = in_degree.get_mut(neighbor) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }
        order
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    #[test]
    fn test_task_builder() {
        let task = RoadmapTask::new("t1", "Risk Assessment")
            .with_effort(30.0)
            .with_cost(50000.0)
            .with_obligation("obl-1")
            .with_resource("compliance-team")
            .with_priority(1);

        assert_eq!(task.id, "t1");
        assert_eq!(task.effort_days, 30.0);
        assert_eq!(task.cost_estimate, 50000.0);
        assert_eq!(task.obligation_id, Some("obl-1".to_string()));
        assert_eq!(task.assigned_resources, vec!["compliance-team"]);
        assert_eq!(task.priority, 1);
    }

    #[test]
    fn test_phase_completion() {
        let mut phase = RoadmapPhase::new("p1", "Phase 1", date(2025, 1, 1), date(2025, 6, 30));
        phase.add_task(RoadmapTask { status: TaskStatus::Completed, ..RoadmapTask::new("t1", "T1") });
        phase.add_task(RoadmapTask { status: TaskStatus::Completed, ..RoadmapTask::new("t2", "T2") });
        phase.add_task(RoadmapTask::new("t3", "T3"));

        assert!((phase.completion_percentage() - 66.666).abs() < 1.0);
    }

    #[test]
    fn test_roadmap_add_phase_sorted() {
        let mut roadmap = ComplianceRoadmap::new("Test Roadmap");
        roadmap.add_phase(RoadmapPhase::new("p2", "Phase 2", date(2025, 7, 1), date(2025, 12, 31)));
        roadmap.add_phase(RoadmapPhase::new("p1", "Phase 1", date(2025, 1, 1), date(2025, 6, 30)));

        assert_eq!(roadmap.phases[0].id, "p1");
        assert_eq!(roadmap.phases[1].id, "p2");
    }

    #[test]
    fn test_validate_no_circular_deps() {
        let mut roadmap = ComplianceRoadmap::new("Test");
        let mut phase = RoadmapPhase::new("p1", "P1", date(2025, 1, 1), date(2025, 12, 31));
        phase.add_task(RoadmapTask::new("t1", "Task A").with_effort(10.0));
        phase.add_task(RoadmapTask::new("t2", "Task B").with_dependency("t1").with_effort(5.0));
        phase.add_task(RoadmapTask::new("t3", "Task C").with_dependency("t2").with_effort(8.0));
        roadmap.add_phase(phase);

        let warnings = roadmap.validate().unwrap();
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_validate_circular_deps() {
        let mut roadmap = ComplianceRoadmap::new("Test");
        let mut phase = RoadmapPhase::new("p1", "P1", date(2025, 1, 1), date(2025, 12, 31));
        phase.add_task(RoadmapTask::new("t1", "A").with_dependency("t3"));
        phase.add_task(RoadmapTask::new("t2", "B").with_dependency("t1"));
        phase.add_task(RoadmapTask::new("t3", "C").with_dependency("t2"));
        roadmap.add_phase(phase);

        assert!(roadmap.validate().is_err());
    }

    #[test]
    fn test_total_cost() {
        let mut roadmap = ComplianceRoadmap::new("Test");
        let mut p1 = RoadmapPhase::new("p1", "P1", date(2025, 1, 1), date(2025, 6, 30));
        p1.add_task(RoadmapTask::new("t1", "T1").with_cost(10000.0));
        p1.add_task(RoadmapTask::new("t2", "T2").with_cost(20000.0));
        let mut p2 = RoadmapPhase::new("p2", "P2", date(2025, 7, 1), date(2025, 12, 31));
        p2.add_task(RoadmapTask::new("t3", "T3").with_cost(30000.0));
        roadmap.add_phase(p1);
        roadmap.add_phase(p2);

        assert_eq!(roadmap.total_cost(), 60000.0);
    }

    #[test]
    fn test_critical_path() {
        let mut roadmap = ComplianceRoadmap::new("Test");
        let mut phase = RoadmapPhase::new("p1", "P1", date(2025, 1, 1), date(2025, 12, 31));
        // Chain: t1 (10d) -> t2 (20d) -> t4 (5d)
        // Branch: t1 (10d) -> t3 (3d)
        // Critical path should be t1 -> t2 -> t4 (35 days)
        phase.add_task(RoadmapTask::new("t1", "Start").with_effort(10.0));
        phase.add_task(RoadmapTask::new("t2", "Middle").with_effort(20.0).with_dependency("t1"));
        phase.add_task(RoadmapTask::new("t3", "Side").with_effort(3.0).with_dependency("t1"));
        phase.add_task(RoadmapTask::new("t4", "End").with_effort(5.0).with_dependency("t2"));
        roadmap.add_phase(phase);

        let cp = roadmap.critical_path();
        assert!(cp.contains(&"t1".to_string()));
        assert!(cp.contains(&"t2".to_string()));
        assert!(cp.contains(&"t4".to_string()));
        assert!(!cp.contains(&"t3".to_string()));
    }

    #[test]
    fn test_gantt_data() {
        let mut roadmap = ComplianceRoadmap::new("Test");
        let mut phase = RoadmapPhase::new("p1", "P1", date(2025, 1, 1), date(2025, 6, 30));
        phase.add_task(
            RoadmapTask::new("t1", "Task 1")
                .with_effort(10.0)
                .with_dates(date(2025, 1, 1), date(2025, 1, 11))
                .with_resource("team-a")
        );
        roadmap.add_phase(phase);

        let gantt = roadmap.gantt_data();
        assert_eq!(gantt.len(), 1);
        assert_eq!(gantt[0].task_id, "t1");
        assert_eq!(gantt[0].start_day, 0);
        assert_eq!(gantt[0].duration_days, 10.0);
        assert_eq!(gantt[0].resource, "team-a");
    }

    #[test]
    fn test_json_roundtrip() {
        let mut roadmap = ComplianceRoadmap::new("Roundtrip Test");
        let mut phase = RoadmapPhase::new("p1", "P1", date(2025, 1, 1), date(2025, 6, 30));
        phase.add_task(RoadmapTask::new("t1", "Task 1").with_effort(10.0));
        roadmap.add_phase(phase);

        let json = roadmap.to_json().unwrap();
        let restored = ComplianceRoadmap::from_json(&json).unwrap();
        assert_eq!(restored.name, "Roundtrip Test");
        assert_eq!(restored.phases.len(), 1);
        assert_eq!(restored.phases[0].tasks.len(), 1);
    }

    #[test]
    fn test_budget_warning() {
        let mut roadmap = ComplianceRoadmap::new("Budget Test");
        let mut phase = RoadmapPhase::new("p1", "P1", date(2025, 1, 1), date(2025, 6, 30));
        phase.budget = 10000.0;
        phase.add_task(RoadmapTask::new("t1", "Expensive").with_cost(50000.0));
        roadmap.add_phase(phase);

        let warnings = roadmap.validate().unwrap();
        assert!(warnings.iter().any(|w| w.contains("exceeds budget")));
    }
}
