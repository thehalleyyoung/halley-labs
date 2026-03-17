use crate::constraint_evolution::ConstraintEvolution;
use crate::ObligationId;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// An incremental update between two regulatory states with tracked evolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalUpdate {
    pub from_state: String,
    pub to_state: String,
    pub evolution: ConstraintEvolution,
}

impl IncrementalUpdate {
    pub fn new(
        from: impl Into<String>,
        to: impl Into<String>,
        evolution: ConstraintEvolution,
    ) -> Self {
        Self {
            from_state: from.into(),
            to_state: to.into(),
            evolution,
        }
    }

    /// Returns the set of all obligation IDs affected by this update.
    pub fn affected_obligations(&self) -> BTreeSet<ObligationId> {
        let mut affected = BTreeSet::new();
        for id in &self.evolution.added {
            affected.insert(id.clone());
        }
        for id in &self.evolution.removed {
            affected.insert(id.clone());
        }
        for id in &self.evolution.modified {
            affected.insert(id.clone());
        }
        affected
    }

    /// An update is trivial if no obligations changed.
    pub fn is_trivial(&self) -> bool {
        self.evolution.is_empty()
    }
}

/// Returns the subset of `constraints` whose string representation
/// contains any of the affected obligation IDs from the update.
pub fn compute_affected_constraints(
    update: &IncrementalUpdate,
    constraints: &[String],
) -> Vec<String> {
    let affected = update.affected_obligations();
    constraints
        .iter()
        .filter(|c| affected.iter().any(|obl| c.contains(obl.as_str())))
        .cloned()
        .collect()
}

/// Returns the IDs of Pareto points that reference any affected obligation.
pub fn invalidated_pareto_points(
    update: &IncrementalUpdate,
    points: &[(String, Vec<f64>)],
) -> Vec<String> {
    let affected = update.affected_obligations();
    points
        .iter()
        .filter(|(id, _)| affected.iter().any(|obl| id.contains(obl.as_str())))
        .map(|(id, _)| id.clone())
        .collect()
}

/// Returns the minimal set of obligations that need recomputation.
pub fn minimal_recomputation_scope(
    update: &IncrementalUpdate,
    all_obligations: &[String],
) -> Vec<String> {
    let affected = update.affected_obligations();
    all_obligations
        .iter()
        .filter(|o| affected.contains(o.as_str()))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_update(added: &[&str], removed: &[&str], modified: &[&str]) -> IncrementalUpdate {
        IncrementalUpdate::new(
            "state-before",
            "state-after",
            ConstraintEvolution {
                added: added.iter().map(|s| s.to_string()).collect(),
                removed: removed.iter().map(|s| s.to_string()).collect(),
                modified: modified.iter().map(|s| s.to_string()).collect(),
            },
        )
    }

    #[test]
    fn test_affected_obligations() {
        let u = make_update(&["a", "b"], &["c"], &["d"]);
        let affected = u.affected_obligations();
        assert_eq!(affected.len(), 4);
        assert!(affected.contains("a"));
        assert!(affected.contains("b"));
        assert!(affected.contains("c"));
        assert!(affected.contains("d"));
    }

    #[test]
    fn test_is_trivial() {
        let trivial = make_update(&[], &[], &[]);
        assert!(trivial.is_trivial());

        let non_trivial = make_update(&["x"], &[], &[]);
        assert!(!non_trivial.is_trivial());
    }

    #[test]
    fn test_compute_affected_constraints() {
        let u = make_update(&["obl-A"], &["obl-B"], &[]);
        let constraints = vec![
            "constraint-for-obl-A-transparency".to_string(),
            "constraint-for-obl-B-reporting".to_string(),
            "constraint-for-obl-C-governance".to_string(),
        ];
        let affected = compute_affected_constraints(&u, &constraints);
        assert_eq!(affected.len(), 2);
        assert!(affected.contains(&"constraint-for-obl-A-transparency".to_string()));
        assert!(affected.contains(&"constraint-for-obl-B-reporting".to_string()));
    }

    #[test]
    fn test_compute_affected_constraints_empty() {
        let u = make_update(&[], &[], &[]);
        let constraints = vec!["c1".to_string(), "c2".to_string()];
        let affected = compute_affected_constraints(&u, &constraints);
        assert!(affected.is_empty());
    }

    #[test]
    fn test_invalidated_pareto_points() {
        let u = make_update(&["obl-X"], &[], &["obl-Y"]);
        let points = vec![
            ("point-obl-X-1".to_string(), vec![1.0, 2.0]),
            ("point-obl-Y-2".to_string(), vec![3.0, 4.0]),
            ("point-obl-Z-3".to_string(), vec![5.0, 6.0]),
        ];
        let invalidated = invalidated_pareto_points(&u, &points);
        assert_eq!(invalidated.len(), 2);
        assert!(invalidated.contains(&"point-obl-X-1".to_string()));
        assert!(invalidated.contains(&"point-obl-Y-2".to_string()));
    }

    #[test]
    fn test_invalidated_pareto_points_none() {
        let u = make_update(&["obl-A"], &[], &[]);
        let points = vec![
            ("point-obl-X".to_string(), vec![1.0]),
        ];
        let invalidated = invalidated_pareto_points(&u, &points);
        assert!(invalidated.is_empty());
    }

    #[test]
    fn test_minimal_recomputation_scope() {
        let u = make_update(&["a", "b"], &["c"], &[]);
        let all = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
            "e".to_string(),
        ];
        let scope = minimal_recomputation_scope(&u, &all);
        assert_eq!(scope.len(), 3);
        assert!(scope.contains(&"a".to_string()));
        assert!(scope.contains(&"b".to_string()));
        assert!(scope.contains(&"c".to_string()));
        assert!(!scope.contains(&"d".to_string()));
    }

    #[test]
    fn test_minimal_recomputation_scope_empty_update() {
        let u = make_update(&[], &[], &[]);
        let all = vec!["a".to_string(), "b".to_string()];
        let scope = minimal_recomputation_scope(&u, &all);
        assert!(scope.is_empty());
    }

    #[test]
    fn test_incremental_update_creation() {
        let u = make_update(&["x"], &["y"], &["z"]);
        assert_eq!(u.from_state, "state-before");
        assert_eq!(u.to_state, "state-after");
        assert_eq!(u.evolution.total_changes(), 3);
    }
}
