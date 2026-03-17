use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

use crate::PlannerError;
use crate::PlannerResult;

// ─── Scheduler Configuration ────────────────────────────────────────────────

/// Configuration for the RCPSP scheduler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Maximum number of scheduling iterations before giving up.
    pub max_iterations: usize,
    /// Weight for earliest-deadline-first priority (0.0–1.0).
    pub deadline_weight: f64,
    /// Weight for critical-path priority (0.0–1.0).
    pub critical_path_weight: f64,
    /// Allow preemption of lower-priority tasks.
    pub allow_preemption: bool,
    /// Working days per week (default 5).
    pub working_days_per_week: u32,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10_000,
            deadline_weight: 0.6,
            critical_path_weight: 0.4,
            allow_preemption: false,
            working_days_per_week: 5,
        }
    }
}

// ─── Resource Capacity ──────────────────────────────────────────────────────

/// Resource availability per time period.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceCapacity {
    pub resource_id: String,
    pub capacity_per_day: f64,
}

// ─── Task Input ─────────────────────────────────────────────────────────────

/// A task to be scheduled.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskInput {
    pub id: String,
    pub name: String,
    pub duration_days: u32,
    pub resource_requirements: HashMap<String, f64>,
    pub dependencies: Vec<String>,
    pub deadline: Option<NaiveDate>,
    pub earliest_start: Option<NaiveDate>,
    pub priority: u32,
}

impl TaskInput {
    pub fn new(id: impl Into<String>, name: impl Into<String>, duration: u32) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            duration_days: duration,
            resource_requirements: HashMap::new(),
            dependencies: Vec::new(),
            deadline: None,
            earliest_start: None,
            priority: 0,
        }
    }

    pub fn with_resource(mut self, resource_id: impl Into<String>, amount: f64) -> Self {
        self.resource_requirements.insert(resource_id.into(), amount);
        self
    }

    pub fn with_dependency(mut self, dep: impl Into<String>) -> Self {
        self.dependencies.push(dep.into());
        self
    }

    pub fn with_deadline(mut self, deadline: NaiveDate) -> Self {
        self.deadline = Some(deadline);
        self
    }

    pub fn with_earliest_start(mut self, date: NaiveDate) -> Self {
        self.earliest_start = Some(date);
        self
    }
}

// ─── Scheduled Task ─────────────────────────────────────────────────────────

/// A task with its computed schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledTask {
    pub task_id: String,
    pub task_name: String,
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
    pub start_day: i32,
    pub duration_days: u32,
    pub resource_assignments: HashMap<String, f64>,
    pub slack: i32,
    pub is_critical: bool,
    pub deadline_met: bool,
}

// ─── Schedule ───────────────────────────────────────────────────────────────

/// Complete schedule produced by the scheduler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schedule {
    pub project_start: NaiveDate,
    pub project_end: NaiveDate,
    pub makespan_days: i32,
    pub tasks: Vec<ScheduledTask>,
    pub feasible: bool,
    pub deadline_violations: Vec<String>,
    pub resource_utilization: HashMap<String, f64>,
}

impl Schedule {
    pub fn critical_path_tasks(&self) -> Vec<&ScheduledTask> {
        self.tasks.iter().filter(|t| t.is_critical).collect()
    }

    pub fn tasks_in_period(&self, start: NaiveDate, end: NaiveDate) -> Vec<&ScheduledTask> {
        self.tasks.iter()
            .filter(|t| t.start_date <= end && t.end_date >= start)
            .collect()
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

// ─── Priority queue entry ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct PriorityEntry {
    task_id: String,
    priority_score: f64,
}

impl PartialEq for PriorityEntry {
    fn eq(&self, other: &Self) -> bool {
        self.task_id == other.task_id
    }
}

impl Eq for PriorityEntry {}

impl PartialOrd for PriorityEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority_score.partial_cmp(&other.priority_score)
            .unwrap_or(Ordering::Equal)
            .reverse() // higher score = higher priority = dequeue first
    }
}

// ─── Scheduler ──────────────────────────────────────────────────────────────

/// RCPSP-based schedule optimizer using priority-rule heuristic.
#[derive(Debug, Clone)]
pub struct Scheduler {
    pub config: SchedulerConfig,
    pub resources: Vec<ResourceCapacity>,
    pub project_start: NaiveDate,
}

impl Scheduler {
    pub fn new(project_start: NaiveDate) -> Self {
        Self {
            config: SchedulerConfig::default(),
            resources: Vec::new(),
            project_start,
        }
    }

    pub fn with_config(mut self, config: SchedulerConfig) -> Self {
        self.config = config;
        self
    }

    pub fn add_resource(&mut self, id: impl Into<String>, capacity_per_day: f64) {
        self.resources.push(ResourceCapacity {
            resource_id: id.into(),
            capacity_per_day,
        });
    }

    /// Schedule tasks using the RCPSP priority-rule heuristic.
    ///
    /// Algorithm:
    /// 1. Compute longest-path values (critical path lengths) for each task
    /// 2. Assign priority scores: weighted combination of deadline urgency + critical path length
    /// 3. Serial schedule generation scheme: schedule tasks one at a time in priority order
    /// 4. For each task, find the earliest feasible start time respecting dependencies and resources
    pub fn schedule(&self, tasks: &[TaskInput]) -> PlannerResult<Schedule> {
        if tasks.is_empty() {
            return Ok(Schedule {
                project_start: self.project_start,
                project_end: self.project_start,
                makespan_days: 0,
                tasks: Vec::new(),
                feasible: true,
                deadline_violations: Vec::new(),
                resource_utilization: HashMap::new(),
            });
        }

        let task_map: HashMap<&str, &TaskInput> = tasks.iter()
            .map(|t| (t.id.as_str(), t))
            .collect();

        // Validate dependencies exist
        for task in tasks {
            for dep in &task.dependencies {
                if !task_map.contains_key(dep.as_str()) {
                    return Err(PlannerError::TaskNotFound(format!(
                        "Task '{}' depends on unknown task '{}'", task.id, dep
                    )));
                }
            }
        }

        // Detect cycles
        if self.has_cycle(tasks, &task_map) {
            return Err(PlannerError::CircularDependency(
                "Circular dependency in task graph".to_string()
            ));
        }

        // Step 1: Compute longest-path values via reverse topological order
        let longest_path = self.compute_longest_paths(tasks, &task_map);

        // Step 2: Compute priority scores
        let priorities = self.compute_priorities(tasks, &longest_path);

        // Step 3: Serial schedule generation scheme (SSGS)
        let max_day = tasks.iter()
            .map(|t| t.duration_days)
            .sum::<u32>() as usize + 1;

        // Resource usage tracking: resource_id -> day -> used_amount
        let mut resource_usage: HashMap<String, Vec<f64>> = HashMap::new();
        for res in &self.resources {
            resource_usage.insert(res.resource_id.clone(), vec![0.0; max_day + 1]);
        }

        // Build priority queue
        let mut pq = BinaryHeap::new();
        for (task_id, score) in &priorities {
            pq.push(PriorityEntry {
                task_id: task_id.clone(),
                priority_score: *score,
            });
        }

        // Track scheduled tasks
        let mut scheduled: HashMap<String, (i32, i32)> = HashMap::new(); // id -> (start_day, end_day)
        let mut scheduling_order: Vec<String> = Vec::new();

        // Pop from queue, but we need dependency-aware ordering.
        // Re-sort by: all dependencies scheduled first, then by priority.
        let topo_order = self.topological_sort(tasks, &task_map);
        let mut topo_priority: Vec<(String, f64)> = topo_order.iter()
            .map(|id| (id.clone(), priorities.get(id).copied().unwrap_or(0.0)))
            .collect();
        // Within the topological layers, sort by priority
        topo_priority.sort_by(|a, b| {
            let a_depth = self.task_depth(&a.0, &task_map, &mut HashMap::new());
            let b_depth = self.task_depth(&b.0, &task_map, &mut HashMap::new());
            a_depth.cmp(&b_depth).then_with(|| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal))
        });

        for (task_id, _) in &topo_priority {
            let task = task_map[task_id.as_str()];

            // Earliest start from dependencies
            let dep_end = task.dependencies.iter()
                .filter_map(|d| scheduled.get(d).map(|&(_, e)| e))
                .max()
                .unwrap_or(0);

            // Earliest start from constraint
            let earliest_from_constraint = task.earliest_start
                .map(|d| (d - self.project_start).num_days() as i32)
                .unwrap_or(0);

            let min_start = dep_end.max(earliest_from_constraint).max(0);

            // Find earliest feasible start day considering resource constraints
            let start_day = self.find_feasible_start(
                task, min_start, max_day as i32, &resource_usage
            );

            let end_day = start_day + task.duration_days as i32;

            // Update resource usage
            for (res_id, &amount) in &task.resource_requirements {
                if let Some(usage) = resource_usage.get_mut(res_id) {
                    for day in start_day..end_day {
                        if (day as usize) < usage.len() {
                            usage[day as usize] += amount;
                        }
                    }
                }
            }

            scheduled.insert(task_id.clone(), (start_day, end_day));
            scheduling_order.push(task_id.clone());
        }

        // Step 4: Build result
        let makespan = scheduled.values()
            .map(|&(_, e)| e)
            .max()
            .unwrap_or(0);

        // Compute critical path (longest path in scheduled graph)
        let mut latest_finish: HashMap<&str, i32> = HashMap::new();
        for task_id in topo_order.iter().rev() {
            let task = task_map[task_id.as_str()];
            let successors: Vec<&str> = tasks.iter()
                .filter(|t| t.dependencies.contains(task_id))
                .map(|t| t.id.as_str())
                .collect();

            let lf = if successors.is_empty() {
                makespan
            } else {
                successors.iter()
                    .filter_map(|s| scheduled.get(*s).map(|&(start, _)| start))
                    .min()
                    .unwrap_or(makespan)
            };
            latest_finish.insert(task_id, lf);
        }

        let mut deadline_violations = Vec::new();
        let mut scheduled_tasks: Vec<ScheduledTask> = Vec::new();

        for task in tasks {
            let (start_day, end_day) = scheduled[&task.id];
            let start_date = self.project_start + chrono::Duration::days(start_day as i64);
            let end_date = self.project_start + chrono::Duration::days(end_day as i64);

            let lf = latest_finish.get(task.id.as_str()).copied().unwrap_or(makespan);
            let slack = lf - end_day;
            let is_critical = slack == 0;

            let deadline_met = task.deadline
                .map(|d| end_date <= d)
                .unwrap_or(true);

            if !deadline_met {
                deadline_violations.push(format!(
                    "Task '{}' finishes on {} but deadline is {}",
                    task.name,
                    end_date,
                    task.deadline.unwrap()
                ));
            }

            scheduled_tasks.push(ScheduledTask {
                task_id: task.id.clone(),
                task_name: task.name.clone(),
                start_date,
                end_date,
                start_day,
                duration_days: task.duration_days,
                resource_assignments: task.resource_requirements.clone(),
                slack,
                is_critical,
                deadline_met,
            });
        }

        // Compute resource utilization
        let mut utilization = HashMap::new();
        for res in &self.resources {
            if let Some(usage) = resource_usage.get(&res.resource_id) {
                let total_used: f64 = usage.iter().take(makespan as usize).sum();
                let total_capacity = res.capacity_per_day * makespan as f64;
                if total_capacity > 0.0 {
                    utilization.insert(
                        res.resource_id.clone(),
                        total_used / total_capacity,
                    );
                }
            }
        }

        let project_end = self.project_start + chrono::Duration::days(makespan as i64);
        let feasible = deadline_violations.is_empty();

        Ok(Schedule {
            project_start: self.project_start,
            project_end,
            makespan_days: makespan,
            tasks: scheduled_tasks,
            feasible,
            deadline_violations,
            resource_utilization: utilization,
        })
    }

    /// Check feasibility without full scheduling.
    pub fn check_feasibility(&self, tasks: &[TaskInput]) -> PlannerResult<Vec<String>> {
        let schedule = self.schedule(tasks)?;
        let mut issues = Vec::new();
        issues.extend(schedule.deadline_violations);

        // Check resource overloading
        for (res_id, &util) in &schedule.resource_utilization {
            if util > 1.0 {
                issues.push(format!(
                    "Resource '{}' overloaded at {:.0}% utilization", res_id, util * 100.0
                ));
            }
        }
        Ok(issues)
    }

    // ─── Internal helpers ───────────────────────────────────────────────────

    fn compute_longest_paths(
        &self,
        tasks: &[TaskInput],
        task_map: &HashMap<&str, &TaskInput>,
    ) -> HashMap<String, u32> {
        let mut longest: HashMap<String, u32> = HashMap::new();
        let mut memo: HashMap<String, u32> = HashMap::new();

        for task in tasks {
            let val = self.longest_path_from(&task.id, task_map, &mut memo);
            longest.insert(task.id.clone(), val);
        }
        longest
    }

    fn longest_path_from(
        &self,
        task_id: &str,
        task_map: &HashMap<&str, &TaskInput>,
        memo: &mut HashMap<String, u32>,
    ) -> u32 {
        if let Some(&val) = memo.get(task_id) {
            return val;
        }
        let task = match task_map.get(task_id) {
            Some(t) => t,
            None => return 0,
        };

        // Find successors (tasks that depend on this one)
        let successor_max: u32 = task_map.values()
            .filter(|t| t.dependencies.contains(&task_id.to_string()))
            .map(|t| self.longest_path_from(&t.id, task_map, memo))
            .max()
            .unwrap_or(0);

        let val = task.duration_days + successor_max;
        memo.insert(task_id.to_string(), val);
        val
    }

    fn compute_priorities(
        &self,
        tasks: &[TaskInput],
        longest_path: &HashMap<String, u32>,
    ) -> HashMap<String, f64> {
        let max_lp = longest_path.values().copied().max().unwrap_or(1) as f64;
        let mut priorities = HashMap::new();

        for task in tasks {
            let lp_score = longest_path.get(&task.id).copied().unwrap_or(0) as f64 / max_lp;

            let deadline_score = task.deadline
                .map(|d| {
                    let days_until = (d - self.project_start).num_days() as f64;
                    if days_until <= 0.0 { 1.0 } else { 1.0 / days_until }
                })
                .unwrap_or(0.0);

            let score = self.config.critical_path_weight * lp_score
                + self.config.deadline_weight * deadline_score
                + (task.priority as f64) * 0.01;

            priorities.insert(task.id.clone(), score);
        }
        priorities
    }

    fn find_feasible_start(
        &self,
        task: &TaskInput,
        min_start: i32,
        max_day: i32,
        resource_usage: &HashMap<String, Vec<f64>>,
    ) -> i32 {
        let resource_caps: HashMap<&str, f64> = self.resources.iter()
            .map(|r| (r.resource_id.as_str(), r.capacity_per_day))
            .collect();

        for start in min_start..max_day {
            let end = start + task.duration_days as i32;
            let mut feasible = true;

            for (res_id, &required) in &task.resource_requirements {
                if let Some(usage) = resource_usage.get(res_id) {
                    let cap = resource_caps.get(res_id.as_str()).copied().unwrap_or(f64::INFINITY);
                    for day in start..end {
                        if (day as usize) < usage.len() {
                            if usage[day as usize] + required > cap {
                                feasible = false;
                                break;
                            }
                        }
                    }
                }
                if !feasible { break; }
            }

            if feasible {
                return start;
            }
        }
        // Fallback: return min_start (may be infeasible)
        min_start
    }

    fn has_cycle(&self, tasks: &[TaskInput], task_map: &HashMap<&str, &TaskInput>) -> bool {
        let mut visited = HashSet::new();
        let mut in_stack = HashSet::new();

        for task in tasks {
            if self.dfs_cycle(&task.id, task_map, &mut visited, &mut in_stack) {
                return true;
            }
        }
        false
    }

    fn dfs_cycle(
        &self,
        task_id: &str,
        task_map: &HashMap<&str, &TaskInput>,
        visited: &mut HashSet<String>,
        in_stack: &mut HashSet<String>,
    ) -> bool {
        if in_stack.contains(task_id) {
            return true;
        }
        if visited.contains(task_id) {
            return false;
        }
        visited.insert(task_id.to_string());
        in_stack.insert(task_id.to_string());

        if let Some(task) = task_map.get(task_id) {
            for dep in &task.dependencies {
                if self.dfs_cycle(dep, task_map, visited, in_stack) {
                    return true;
                }
            }
        }

        in_stack.remove(task_id);
        false
    }

    fn topological_sort(
        &self,
        tasks: &[TaskInput],
        task_map: &HashMap<&str, &TaskInput>,
    ) -> Vec<String> {
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();

        for task in tasks {
            in_degree.entry(&task.id).or_insert(0);
            adj.entry(&task.id).or_default();
        }
        for task in tasks {
            for dep in &task.dependencies {
                if task_map.contains_key(dep.as_str()) {
                    adj.entry(dep.as_str()).or_default().push(&task.id);
                    *in_degree.entry(&task.id).or_insert(0) += 1;
                }
            }
        }

        let mut queue: std::collections::VecDeque<&str> = in_degree.iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&id, _)| id)
            .collect();
        let mut order = Vec::new();

        while let Some(node) = queue.pop_front() {
            order.push(node.to_string());
            if let Some(neighbors) = adj.get(node) {
                for &n in neighbors {
                    if let Some(deg) = in_degree.get_mut(n) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(n);
                        }
                    }
                }
            }
        }
        order
    }

    fn task_depth(
        &self,
        task_id: &str,
        task_map: &HashMap<&str, &TaskInput>,
        memo: &mut HashMap<String, usize>,
    ) -> usize {
        if let Some(&d) = memo.get(task_id) {
            return d;
        }
        let task = match task_map.get(task_id) {
            Some(t) => t,
            None => return 0,
        };
        let depth = task.dependencies.iter()
            .map(|d| self.task_depth(d, task_map, memo) + 1)
            .max()
            .unwrap_or(0);
        memo.insert(task_id.to_string(), depth);
        depth
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap()
    }

    #[test]
    fn test_empty_schedule() {
        let scheduler = Scheduler::new(date(2025, 1, 1));
        let schedule = scheduler.schedule(&[]).unwrap();
        assert!(schedule.feasible);
        assert_eq!(schedule.makespan_days, 0);
    }

    #[test]
    fn test_single_task() {
        let scheduler = Scheduler::new(date(2025, 1, 1));
        let tasks = vec![TaskInput::new("t1", "Task 1", 10)];
        let schedule = scheduler.schedule(&tasks).unwrap();

        assert_eq!(schedule.tasks.len(), 1);
        assert_eq!(schedule.tasks[0].start_day, 0);
        assert_eq!(schedule.tasks[0].duration_days, 10);
        assert_eq!(schedule.makespan_days, 10);
    }

    #[test]
    fn test_dependent_tasks() {
        let scheduler = Scheduler::new(date(2025, 1, 1));
        let tasks = vec![
            TaskInput::new("t1", "Task 1", 10),
            TaskInput::new("t2", "Task 2", 5).with_dependency("t1"),
            TaskInput::new("t3", "Task 3", 8).with_dependency("t2"),
        ];
        let schedule = scheduler.schedule(&tasks).unwrap();

        let t1 = schedule.tasks.iter().find(|t| t.task_id == "t1").unwrap();
        let t2 = schedule.tasks.iter().find(|t| t.task_id == "t2").unwrap();
        let t3 = schedule.tasks.iter().find(|t| t.task_id == "t3").unwrap();

        assert!(t2.start_day >= t1.start_day + t1.duration_days as i32);
        assert!(t3.start_day >= t2.start_day + t2.duration_days as i32);
        assert_eq!(schedule.makespan_days, 23);
    }

    #[test]
    fn test_parallel_tasks() {
        let mut scheduler = Scheduler::new(date(2025, 1, 1));
        scheduler.add_resource("team", 2.0);

        let tasks = vec![
            TaskInput::new("t1", "Task 1", 10).with_resource("team", 1.0),
            TaskInput::new("t2", "Task 2", 10).with_resource("team", 1.0),
        ];
        let schedule = scheduler.schedule(&tasks).unwrap();

        // Both tasks can run in parallel (total resource need = 2.0 = capacity)
        assert_eq!(schedule.makespan_days, 10);
    }

    #[test]
    fn test_resource_contention() {
        let mut scheduler = Scheduler::new(date(2025, 1, 1));
        scheduler.add_resource("team", 1.0);

        let tasks = vec![
            TaskInput::new("t1", "Task 1", 10).with_resource("team", 1.0),
            TaskInput::new("t2", "Task 2", 10).with_resource("team", 1.0),
        ];
        let schedule = scheduler.schedule(&tasks).unwrap();

        // Tasks must be sequential due to resource constraint
        assert_eq!(schedule.makespan_days, 20);
    }

    #[test]
    fn test_deadline_violation_detected() {
        let scheduler = Scheduler::new(date(2025, 1, 1));
        let tasks = vec![
            TaskInput::new("t1", "Task 1", 30)
                .with_deadline(date(2025, 1, 15)),
        ];
        let schedule = scheduler.schedule(&tasks).unwrap();

        assert!(!schedule.feasible);
        assert!(!schedule.deadline_violations.is_empty());
        assert!(!schedule.tasks[0].deadline_met);
    }

    #[test]
    fn test_deadline_met() {
        let scheduler = Scheduler::new(date(2025, 1, 1));
        let tasks = vec![
            TaskInput::new("t1", "Task 1", 10)
                .with_deadline(date(2025, 2, 1)),
        ];
        let schedule = scheduler.schedule(&tasks).unwrap();

        assert!(schedule.feasible);
        assert!(schedule.tasks[0].deadline_met);
    }

    #[test]
    fn test_circular_dependency_error() {
        let scheduler = Scheduler::new(date(2025, 1, 1));
        let tasks = vec![
            TaskInput::new("t1", "A", 5).with_dependency("t3"),
            TaskInput::new("t2", "B", 5).with_dependency("t1"),
            TaskInput::new("t3", "C", 5).with_dependency("t2"),
        ];
        assert!(scheduler.schedule(&tasks).is_err());
    }

    #[test]
    fn test_earliest_start_constraint() {
        let scheduler = Scheduler::new(date(2025, 1, 1));
        let tasks = vec![
            TaskInput::new("t1", "Task 1", 10)
                .with_earliest_start(date(2025, 2, 1)),
        ];
        let schedule = scheduler.schedule(&tasks).unwrap();

        assert_eq!(schedule.tasks[0].start_date, date(2025, 2, 1));
    }

    #[test]
    fn test_critical_path_identification() {
        let scheduler = Scheduler::new(date(2025, 1, 1));
        let tasks = vec![
            TaskInput::new("t1", "Start", 10),
            TaskInput::new("t2", "Long Path", 20).with_dependency("t1"),
            TaskInput::new("t3", "Short Path", 3).with_dependency("t1"),
            TaskInput::new("t4", "End", 5).with_dependency("t2"),
        ];
        let schedule = scheduler.schedule(&tasks).unwrap();

        let critical: Vec<&str> = schedule.tasks.iter()
            .filter(|t| t.is_critical)
            .map(|t| t.task_id.as_str())
            .collect();
        assert!(critical.contains(&"t1"));
        assert!(critical.contains(&"t2"));
        assert!(critical.contains(&"t4"));
    }

    #[test]
    fn test_feasibility_check() {
        let scheduler = Scheduler::new(date(2025, 1, 1));
        let tasks = vec![
            TaskInput::new("t1", "Task 1", 10)
                .with_deadline(date(2025, 1, 5)),
        ];
        let issues = scheduler.check_feasibility(&tasks).unwrap();
        assert!(!issues.is_empty());
    }
}
