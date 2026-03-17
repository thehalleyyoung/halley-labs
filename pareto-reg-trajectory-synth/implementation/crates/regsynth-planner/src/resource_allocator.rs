use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::PlannerError;
use crate::roadmap::RoadmapTask;

// ─── Resource Types ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    Personnel,
    Budget,
    Infrastructure,
    Legal,
}

impl std::fmt::Display for ResourceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Personnel => write!(f, "Personnel"),
            Self::Budget => write!(f, "Budget"),
            Self::Infrastructure => write!(f, "Infrastructure"),
            Self::Legal => write!(f, "Legal"),
        }
    }
}

// ─── Resource Pool ──────────────────────────────────────────────────────────

/// A pool of resources available for allocation across periods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePool {
    pub name: String,
    pub resource_type: ResourceType,
    pub capacity: f64,
    pub unit: String,
    /// Per-period capacity overrides (period index → capacity).
    /// If a period is not present, `capacity` is used.
    #[serde(default)]
    pub period_capacities: HashMap<u32, f64>,
}

impl ResourcePool {
    pub fn new(name: impl Into<String>, resource_type: ResourceType, capacity: f64, unit: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            resource_type,
            capacity,
            unit: unit.into(),
            period_capacities: HashMap::new(),
        }
    }

    /// Return the effective capacity for a given period.
    pub fn capacity_for_period(&self, period: u32) -> f64 {
        self.period_capacities.get(&period).copied().unwrap_or(self.capacity)
    }

    pub fn with_period_capacity(mut self, period: u32, cap: f64) -> Self {
        self.period_capacities.insert(period, cap);
        self
    }
}

// ─── Resource Demand ────────────────────────────────────────────────────────

/// Demand from a single task on a specific resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceDemand {
    pub task_id: String,
    pub resource_name: String,
    pub amount: f64,
}

// ─── Allocation Entry & Result ──────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationEntry {
    pub task_id: String,
    pub resource_name: String,
    pub amount: f64,
    pub period: u32,
}

/// Summary of utilization for a single resource across all periods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    pub resource_name: String,
    pub peak_utilization: f64,
    pub average_utilization: f64,
    pub overloaded_periods: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Allocation {
    pub entries: Vec<AllocationEntry>,
    pub total_utilization: f64,
    pub per_resource: Vec<ResourceUtilization>,
    pub num_periods: u32,
}

// ─── Resource Allocator ─────────────────────────────────────────────────────

pub struct ResourceAllocator {
    pub pools: Vec<ResourcePool>,
    /// Explicit demand overrides. If empty, demands are estimated from task effort.
    demands: Vec<ResourceDemand>,
}

impl ResourceAllocator {
    pub fn new(pools: Vec<ResourcePool>) -> Self {
        Self { pools, demands: Vec::new() }
    }

    pub fn add_demand(&mut self, demand: ResourceDemand) {
        self.demands.push(demand);
    }

    /// Allocate resources to tasks across multiple periods.
    ///
    /// Tasks are assigned to periods in order. Each period can only hold tasks whose
    /// combined demand does not exceed the pool capacity for that period. When a period
    /// is full, the allocator advances to the next period (multi-period levelling).
    pub fn allocate(&self, tasks: &[RoadmapTask]) -> Result<Allocation, PlannerError> {
        if tasks.is_empty() {
            return Ok(Allocation {
                entries: Vec::new(),
                total_utilization: 0.0,
                per_resource: self.pools.iter().map(|p| ResourceUtilization {
                    resource_name: p.name.clone(),
                    peak_utilization: 0.0,
                    average_utilization: 0.0,
                    overloaded_periods: Vec::new(),
                }).collect(),
                num_periods: 0,
            });
        }

        let demand_map = self.build_demand_map(tasks);

        // period → resource_name → consumed
        let mut period_usage: HashMap<u32, HashMap<String, f64>> = HashMap::new();
        let mut entries = Vec::new();
        let mut task_periods: HashMap<String, u32> = HashMap::new();

        for task in tasks {
            let task_demands = self.demands_for_task(&task.id, task, &demand_map);

            // Find the earliest period where all demands fit
            let mut period = 0u32;
            loop {
                let fits = task_demands.iter().all(|(res_name, amount)| {
                    let cap = self.pool_capacity(res_name, period);
                    let used = period_usage
                        .get(&period)
                        .and_then(|m| m.get(res_name.as_str()))
                        .copied()
                        .unwrap_or(0.0);
                    used + amount <= cap + f64::EPSILON
                });

                if fits {
                    break;
                }

                period += 1;
                if period > 1000 {
                    return Err(PlannerError::ResourceInfeasible(format!(
                        "Cannot fit task '{}' in any period within 1000 periods",
                        task.id
                    )));
                }
            }

            // Record allocations
            for (res_name, amount) in &task_demands {
                *period_usage
                    .entry(period)
                    .or_default()
                    .entry(res_name.clone())
                    .or_insert(0.0) += amount;

                entries.push(AllocationEntry {
                    task_id: task.id.clone(),
                    resource_name: res_name.to_string(),
                    amount: *amount,
                    period,
                });
            }

            task_periods.insert(task.id.clone(), period);
        }

        let num_periods = period_usage.keys().copied().max().map(|p| p + 1).unwrap_or(0);
        let per_resource = self.compute_utilization(&period_usage, num_periods);

        let total_cap: f64 = self.pools.iter().map(|p| p.capacity).sum::<f64>().max(1.0);
        let total_used: f64 = entries.iter().map(|e| e.amount).sum();
        let total_utilization = total_used / (total_cap * num_periods.max(1) as f64);

        Ok(Allocation {
            entries,
            total_utilization,
            per_resource,
            num_periods,
        })
    }

    /// Load-balance tasks across periods using round-robin assignment.
    /// Unlike `allocate`, this does not pack greedily but distributes evenly.
    pub fn allocate_balanced(&self, tasks: &[RoadmapTask], max_periods: u32) -> Result<Allocation, PlannerError> {
        if tasks.is_empty() || max_periods == 0 {
            return self.allocate(tasks);
        }

        let demand_map = self.build_demand_map(tasks);
        let mut period_usage: HashMap<u32, HashMap<String, f64>> = HashMap::new();
        let mut entries = Vec::new();

        // Sort tasks by effort descending (largest-first bin packing heuristic)
        let mut sorted_indices: Vec<usize> = (0..tasks.len()).collect();
        sorted_indices.sort_by(|&a, &b| {
            tasks[b].effort_days.partial_cmp(&tasks[a].effort_days).unwrap_or(std::cmp::Ordering::Equal)
        });

        for idx in sorted_indices {
            let task = &tasks[idx];
            let task_demands = self.demands_for_task(&task.id, task, &demand_map);

            // Find the period with the least total load that still has capacity
            let mut best_period = 0u32;
            let mut best_load = f64::MAX;

            for period in 0..max_periods {
                let current_load: f64 = period_usage
                    .get(&period)
                    .map(|m| m.values().sum())
                    .unwrap_or(0.0);

                let fits = task_demands.iter().all(|(res_name, amount)| {
                    let cap = self.pool_capacity(res_name, period);
                    let used = period_usage
                        .get(&period)
                        .and_then(|m| m.get(res_name.as_str()))
                        .copied()
                        .unwrap_or(0.0);
                    used + amount <= cap + f64::EPSILON
                });

                if fits && current_load < best_load {
                    best_load = current_load;
                    best_period = period;
                }
            }

            for (res_name, amount) in &task_demands {
                *period_usage
                    .entry(best_period)
                    .or_default()
                    .entry(res_name.clone())
                    .or_insert(0.0) += amount;

                entries.push(AllocationEntry {
                    task_id: task.id.clone(),
                    resource_name: res_name.to_string(),
                    amount: *amount,
                    period: best_period,
                });
            }
        }

        let num_periods = period_usage.keys().copied().max().map(|p| p + 1).unwrap_or(0);
        let per_resource = self.compute_utilization(&period_usage, num_periods);
        let total_cap: f64 = self.pools.iter().map(|p| p.capacity).sum::<f64>().max(1.0);
        let total_used: f64 = entries.iter().map(|e| e.amount).sum();
        let total_utilization = total_used / (total_cap * num_periods.max(1) as f64);

        Ok(Allocation {
            entries,
            total_utilization,
            per_resource,
            num_periods,
        })
    }

    // ── Internals ───────────────────────────────────────────────────────────

    fn build_demand_map(&self, _tasks: &[RoadmapTask]) -> HashMap<String, Vec<ResourceDemand>> {
        let mut map: HashMap<String, Vec<ResourceDemand>> = HashMap::new();
        for d in &self.demands {
            map.entry(d.task_id.clone()).or_default().push(d.clone());
        }
        map
    }

    /// Get demands for a task. Uses explicit demands if registered, otherwise
    /// estimates proportional demand across all pools based on effort.
    fn demands_for_task(
        &self,
        task_id: &str,
        task: &RoadmapTask,
        demand_map: &HashMap<String, Vec<ResourceDemand>>,
    ) -> Vec<(String, f64)> {
        if let Some(explicit) = demand_map.get(task_id) {
            explicit.iter().map(|d| (d.resource_name.clone(), d.amount)).collect()
        } else {
            // Estimate: task uses a fraction of each pool proportional to its effort
            let effort_fraction = (task.effort_days / 30.0).clamp(0.01, 1.0);
            self.pools
                .iter()
                .map(|p| (p.name.clone(), p.capacity * effort_fraction * 0.25))
                .collect()
        }
    }

    fn pool_capacity(&self, resource_name: &str, period: u32) -> f64 {
        self.pools
            .iter()
            .find(|p| p.name == resource_name)
            .map(|p| p.capacity_for_period(period))
            .unwrap_or(0.0)
    }

    fn compute_utilization(
        &self,
        period_usage: &HashMap<u32, HashMap<String, f64>>,
        num_periods: u32,
    ) -> Vec<ResourceUtilization> {
        self.pools
            .iter()
            .map(|pool| {
                let mut peak = 0.0f64;
                let mut total = 0.0f64;
                let mut overloaded = Vec::new();

                for period in 0..num_periods {
                    let used = period_usage
                        .get(&period)
                        .and_then(|m| m.get(pool.name.as_str()))
                        .copied()
                        .unwrap_or(0.0);
                    let cap = pool.capacity_for_period(period);
                    let util = if cap > 0.0 { used / cap } else { 0.0 };

                    peak = peak.max(util);
                    total += util;

                    if used > cap + f64::EPSILON {
                        overloaded.push(period);
                    }
                }

                let avg = if num_periods > 0 { total / num_periods as f64 } else { 0.0 };

                ResourceUtilization {
                    resource_name: pool.name.clone(),
                    peak_utilization: peak,
                    average_utilization: avg,
                    overloaded_periods: overloaded,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::roadmap::{RoadmapTask, TaskStatus};

    fn make_pool(name: &str, cap: f64) -> ResourcePool {
        ResourcePool::new(name, ResourceType::Personnel, cap, "FTE")
    }

    fn make_task(id: &str, effort: f64) -> RoadmapTask {
        RoadmapTask::new(id, id).with_effort(effort)
    }

    #[test]
    fn test_allocate_empty() {
        let allocator = ResourceAllocator::new(vec![make_pool("eng", 10.0)]);
        let alloc = allocator.allocate(&[]).unwrap();
        assert_eq!(alloc.entries.len(), 0);
        assert_eq!(alloc.num_periods, 0);
    }

    #[test]
    fn test_allocate_single_task() {
        let allocator = ResourceAllocator::new(vec![make_pool("eng", 10.0)]);
        let tasks = vec![make_task("t1", 5.0)];
        let alloc = allocator.allocate(&tasks).unwrap();
        assert!(!alloc.entries.is_empty());
        assert!(alloc.entries.iter().all(|e| e.task_id == "t1"));
    }

    #[test]
    fn test_allocate_multiple_tasks_fit_one_period() {
        let allocator = ResourceAllocator::new(vec![make_pool("eng", 100.0)]);
        let tasks = vec![make_task("t1", 1.0), make_task("t2", 1.0)];
        let alloc = allocator.allocate(&tasks).unwrap();
        // Both small tasks should fit in period 0
        assert!(alloc.entries.iter().all(|e| e.period == 0));
    }

    #[test]
    fn test_allocate_spills_to_next_period() {
        let mut allocator = ResourceAllocator::new(vec![make_pool("eng", 5.0)]);
        allocator.add_demand(ResourceDemand { task_id: "t1".into(), resource_name: "eng".into(), amount: 4.0 });
        allocator.add_demand(ResourceDemand { task_id: "t2".into(), resource_name: "eng".into(), amount: 4.0 });

        let tasks = vec![make_task("t1", 10.0), make_task("t2", 10.0)];
        let alloc = allocator.allocate(&tasks).unwrap();
        // t1 in period 0, t2 must spill to period 1 since 4+4 > 5
        let t2_period = alloc.entries.iter().find(|e| e.task_id == "t2").unwrap().period;
        assert_eq!(t2_period, 1);
    }

    #[test]
    fn test_balanced_allocation() {
        let allocator = ResourceAllocator::new(vec![make_pool("eng", 100.0)]);
        let tasks = vec![
            make_task("t1", 10.0),
            make_task("t2", 10.0),
            make_task("t3", 10.0),
            make_task("t4", 10.0),
        ];
        let alloc = allocator.allocate_balanced(&tasks, 2).unwrap();
        // Tasks should be distributed across 2 periods
        let p0_count = alloc.entries.iter().filter(|e| e.period == 0).map(|e| &e.task_id).collect::<std::collections::HashSet<_>>().len();
        let p1_count = alloc.entries.iter().filter(|e| e.period == 1).map(|e| &e.task_id).collect::<std::collections::HashSet<_>>().len();
        assert!(p0_count > 0 && p1_count > 0, "Expected tasks in both periods");
    }

    #[test]
    fn test_per_period_capacity() {
        let pool = make_pool("eng", 10.0).with_period_capacity(1, 20.0);
        assert_eq!(pool.capacity_for_period(0), 10.0);
        assert_eq!(pool.capacity_for_period(1), 20.0);
        assert_eq!(pool.capacity_for_period(2), 10.0);
    }

    #[test]
    fn test_utilization_reporting() {
        let allocator = ResourceAllocator::new(vec![make_pool("eng", 10.0)]);
        let tasks = vec![make_task("t1", 5.0)];
        let alloc = allocator.allocate(&tasks).unwrap();
        assert_eq!(alloc.per_resource.len(), 1);
        assert!(alloc.per_resource[0].peak_utilization >= 0.0);
    }

    #[test]
    fn test_resource_type_display() {
        assert_eq!(ResourceType::Personnel.to_string(), "Personnel");
        assert_eq!(ResourceType::Legal.to_string(), "Legal");
    }
}
