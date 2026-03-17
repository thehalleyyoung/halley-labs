//! Anomaly pattern catalog for the Adya isolation anomaly taxonomy.
//!
//! Provides a registry of known anomaly patterns (G0–G2) and matching logic
//! to identify which patterns a given dependency cycle corresponds to.

use std::collections::HashMap;
use isospec_types::isolation::{AnomalyClass, IsolationLevel};
use isospec_types::dependency::DependencyType;
use isospec_types::config::EngineKind;
use crate::classifier::CycleInfo;

/// Describes how closely a cycle matched an anomaly pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MatchQuality {
    Exact,
    Rotated,
    Subsequence,
    Partial,
}

impl MatchQuality {
    pub fn as_str(&self) -> &'static str {
        match self {
            MatchQuality::Exact => "exact",
            MatchQuality::Rotated => "rotated",
            MatchQuality::Subsequence => "subsequence",
            MatchQuality::Partial => "partial",
        }
    }

    pub fn is_exact(&self) -> bool {
        matches!(self, MatchQuality::Exact)
    }
}

/// A named anomaly pattern from the formal isolation literature.
#[derive(Debug, Clone)]
pub struct AnomalyPattern {
    pub id: String,
    pub name: String,
    pub anomaly_class: AnomalyClass,
    pub description: String,
    pub edge_pattern: Vec<DependencyType>,
    pub min_txns: usize,
    pub references: Vec<String>,
}

impl AnomalyPattern {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        anomaly_class: AnomalyClass,
        description: impl Into<String>,
        edge_pattern: Vec<DependencyType>,
        min_txns: usize,
        references: Vec<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            anomaly_class,
            description: description.into(),
            edge_pattern,
            min_txns,
            references,
        }
    }

    /// Check whether `cycle`'s edge type sequence matches this pattern,
    /// allowing arbitrary rotation of the cycle (since cycles have no fixed
    /// starting point).
    pub fn matches_cycle(&self, cycle: &CycleInfo) -> bool {
        let edge_types: Vec<DependencyType> =
            cycle.edges.iter().map(|e| e.dep_type).collect();
        if edge_types.len() != self.edge_pattern.len() {
            return false;
        }
        rotate_matches(&self.edge_pattern, &edge_types).is_some()
    }

    /// Returns `true` if every edge in `self.edge_pattern` also appears (with
    /// at least the same multiplicity) in `other.edge_pattern`, meaning this
    /// pattern is a sub-pattern of `other`.
    pub fn is_subset_of(&self, other: &AnomalyPattern) -> bool {
        let mut remaining: Vec<DependencyType> = other.edge_pattern.clone();
        for dep in &self.edge_pattern {
            if let Some(pos) = remaining.iter().position(|d| d == dep) {
                remaining.remove(pos);
            } else {
                return false;
            }
        }
        true
    }
}

/// A successful match of an [`AnomalyPattern`] against a [`CycleInfo`].
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub pattern: AnomalyPattern,
    pub cycle: CycleInfo,
    pub match_quality: MatchQuality,
    pub rotation_offset: usize,
}

/// Check whether `edges` equals `pattern` at some rotation.
/// Returns `Some(offset)` giving the rotation amount, or `None`.
pub fn rotate_matches(
    pattern: &[DependencyType],
    edges: &[DependencyType],
) -> Option<usize> {
    let n = pattern.len();
    if n == 0 || edges.len() != n {
        return None;
    }
    for offset in 0..n {
        let mut ok = true;
        for i in 0..n {
            if edges[(i + offset) % n] != pattern[i] {
                ok = false;
                break;
            }
        }
        if ok {
            return Some(offset);
        }
    }
    None
}

/// Return `true` if `pattern` appears as a contiguous sub-sequence inside
/// `edges` (which may be longer).
fn is_contiguous_subsequence(
    pattern: &[DependencyType],
    edges: &[DependencyType],
) -> bool {
    if pattern.is_empty() {
        return true;
    }
    if pattern.len() > edges.len() {
        return false;
    }
    let limit = edges.len() - pattern.len() + 1;
    for start in 0..limit {
        if edges[start..start + pattern.len()] == *pattern {
            return true;
        }
    }
    false
}

/// Count how many elements from `pattern` appear in `edges` (respecting
/// multiplicity).  Returns the fraction matched (0.0–1.0).
fn partial_overlap(pattern: &[DependencyType], edges: &[DependencyType]) -> f64 {
    if pattern.is_empty() {
        return 1.0;
    }
    let mut remaining: Vec<DependencyType> = edges.to_vec();
    let mut hits: usize = 0;
    for dep in pattern {
        if let Some(pos) = remaining.iter().position(|d| d == dep) {
            remaining.remove(pos);
            hits += 1;
        }
    }
    hits as f64 / pattern.len() as f64
}

/// Registry of known anomaly patterns with lookup and matching helpers.
pub struct AnomalyCatalog {
    patterns: Vec<AnomalyPattern>,
    by_class: HashMap<AnomalyClass, Vec<usize>>,
    by_name: HashMap<String, usize>,
}

impl AnomalyCatalog {
    /// Create an empty catalog.
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            by_class: HashMap::new(),
            by_name: HashMap::new(),
        }
    }

    /// Create a catalog pre-populated with all standard Adya/Berenson
    /// anomaly patterns.
    pub fn with_standard_patterns() -> Self {
        let mut cat = Self::new();
        cat.register(AnomalyPattern::new(
            "g0-dirty-write", "dirty-write", AnomalyClass::G0,
            "Two concurrent txns write the same item (ww cycle).",
            vec![DependencyType::WriteWrite], 2,
            vec!["Adya 1999".into()],
        ));
        cat.register(AnomalyPattern::new(
            "g1a-aborted-read", "aborted-read", AnomalyClass::G1a,
            "A txn reads a value written by an aborted txn.",
            vec![DependencyType::WriteRead], 2,
            vec!["Adya 1999".into()],
        ));
        cat.register(AnomalyPattern::new(
            "g1b-intermediate-read", "intermediate-read", AnomalyClass::G1b,
            "A txn reads an intermediate version later overwritten.",
            vec![DependencyType::WriteRead], 2,
            vec!["Adya 1999".into()],
        ));
        cat.register(AnomalyPattern::new(
            "g1c-circular-info-flow", "circular-info-flow", AnomalyClass::G1c,
            "Two committed txns each read the other's write.",
            vec![DependencyType::WriteRead, DependencyType::WriteRead], 2,
            vec!["Adya 1999".into()],
        ));
        cat.register(AnomalyPattern::new(
            "g2item-lost-update-rw-ww", "lost-update", AnomalyClass::G2Item,
            "A read-write anti-dep followed by write-write loses an update.",
            vec![DependencyType::ReadWrite, DependencyType::WriteWrite], 2,
            vec!["Berenson 1995".into()],
        ));
        cat.register(AnomalyPattern::new(
            "g2item-write-skew", "write-skew", AnomalyClass::G2Item,
            "Two txns each read an item the other writes (rw+rw cycle).",
            vec![DependencyType::ReadWrite, DependencyType::ReadWrite], 2,
            vec!["Berenson 1995".into(), "Adya 1999".into()],
        ));
        cat.register(AnomalyPattern::new(
            "g2item-read-skew", "read-skew", AnomalyClass::G2Item,
            "A txn sees inconsistent values due to concurrent write.",
            vec![DependencyType::WriteRead, DependencyType::ReadWrite], 2,
            vec!["Berenson 1995".into()],
        ));
        cat.register(AnomalyPattern::new(
            "g2-phantom-read", "phantom-read", AnomalyClass::G2,
            "Predicate-based read invalidated by concurrent insert/delete.",
            vec![DependencyType::PredicateReadWrite, DependencyType::WriteRead], 2,
            vec!["Adya 1999".into(), "Berenson 1995".into()],
        ));
        cat.register(AnomalyPattern::new(
            "g2-predicate-write-skew", "predicate-write-skew", AnomalyClass::G2,
            "Two txns each predicate-read a set modified by the other.",
            vec![DependencyType::PredicateReadWrite, DependencyType::PredicateReadWrite], 2,
            vec!["Adya 1999".into()],
        ));
        cat
    }

    /// Register a new pattern. Returns `&mut Self` for chaining.
    pub fn register(&mut self, pattern: AnomalyPattern) -> &mut Self {
        let idx = self.patterns.len();
        self.by_class
            .entry(pattern.anomaly_class)
            .or_default()
            .push(idx);
        self.by_name.insert(pattern.name.clone(), idx);
        self.patterns.push(pattern);
        self
    }

    /// Look up a pattern by its display name.
    pub fn get_by_name(&self, name: &str) -> Option<&AnomalyPattern> {
        self.by_name.get(name).map(|&i| &self.patterns[i])
    }

    /// Return all patterns belonging to the given anomaly class.
    pub fn get_by_class(&self, class: AnomalyClass) -> Vec<&AnomalyPattern> {
        self.by_class
            .get(&class)
            .map(|indices| indices.iter().map(|&i| &self.patterns[i]).collect())
            .unwrap_or_default()
    }

    /// Borrow the full list of registered patterns.
    pub fn all_patterns(&self) -> &[AnomalyPattern] {
        &self.patterns
    }

    /// Number of registered patterns.
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    /// Find every pattern that matches `cycle`, ordered by match quality
    /// (best first).
    pub fn match_cycle(&self, cycle: &CycleInfo) -> Vec<PatternMatch> {
        let edge_types: Vec<DependencyType> =
            cycle.edges.iter().map(|e| e.dep_type).collect();

        let mut matches: Vec<PatternMatch> = Vec::new();

        for pattern in &self.patterns {
            if pattern.edge_pattern.is_empty() {
                continue;
            }
            // 1) Exact / rotated match (same length)
            if edge_types.len() == pattern.edge_pattern.len() {
                if let Some(offset) =
                    rotate_matches(&pattern.edge_pattern, &edge_types)
                {
                    let quality = if offset == 0 {
                        MatchQuality::Exact
                    } else {
                        MatchQuality::Rotated
                    };
                    matches.push(PatternMatch {
                        pattern: pattern.clone(),
                        cycle: cycle.clone(),
                        match_quality: quality,
                        rotation_offset: offset,
                    });
                    continue;
                }
            }
            // 2) Subsequence (cycle is longer than pattern)
            if edge_types.len() > pattern.edge_pattern.len()
                && is_contiguous_subsequence(&pattern.edge_pattern, &edge_types)
            {
                matches.push(PatternMatch {
                    pattern: pattern.clone(),
                    cycle: cycle.clone(),
                    match_quality: MatchQuality::Subsequence,
                    rotation_offset: 0,
                });
                continue;
            }
            // 3) Partial overlap (> 50 % of pattern edges present)
            let overlap = partial_overlap(&pattern.edge_pattern, &edge_types);
            if overlap > 0.5 {
                matches.push(PatternMatch {
                    pattern: pattern.clone(),
                    cycle: cycle.clone(),
                    match_quality: MatchQuality::Partial,
                    rotation_offset: 0,
                });
            }
        }

        // Sort: Exact < Rotated < Subsequence < Partial  (derives Ord)
        matches.sort_by_key(|m| m.match_quality);
        matches
    }

    /// Return the single best match for `cycle`, if any.
    pub fn best_match(&self, cycle: &CycleInfo) -> Option<PatternMatch> {
        let mut matches = self.match_cycle(cycle);
        if matches.is_empty() {
            None
        } else {
            Some(matches.remove(0))
        }
    }

    /// Remove the pattern with the given name. Returns `true` if it existed.
    pub fn remove_pattern(&mut self, name: &str) -> bool {
        let idx = match self.by_name.remove(name) {
            Some(i) => i,
            None => return false,
        };
        let removed = self.patterns.remove(idx);
        // Remove from by_class
        if let Some(indices) = self.by_class.get_mut(&removed.anomaly_class) {
            indices.retain(|&i| i != idx);
            // Shift indices above the removed one
            for v in indices.iter_mut() {
                if *v > idx {
                    *v -= 1;
                }
            }
        }
        // Rebuild by_name because all indices >= idx shifted down
        self.by_name.clear();
        for (i, p) in self.patterns.iter().enumerate() {
            self.by_name.insert(p.name.clone(), i);
        }
        // Rebuild by_class for correctness after removal
        self.by_class.clear();
        for (i, p) in self.patterns.iter().enumerate() {
            self.by_class.entry(p.anomaly_class).or_default().push(i);
        }
        true
    }
}

impl Default for AnomalyCatalog {
    fn default() -> Self {
        Self::new()
    }
}

/// Describes which anomaly patterns a particular engine + isolation level
/// combination is expected to prevent or allow.
#[derive(Debug, Clone)]
pub struct EnginePatternExpectation {
    pub engine: EngineKind,
    pub isolation: IsolationLevel,
    pub expected_prevented: Vec<String>,
    pub expected_possible: Vec<String>,
}

impl EnginePatternExpectation {
    pub fn new(
        engine: EngineKind,
        isolation: IsolationLevel,
        expected_prevented: Vec<String>,
        expected_possible: Vec<String>,
    ) -> Self {
        Self {
            engine,
            isolation,
            expected_prevented,
            expected_possible,
        }
    }

    /// Returns `true` if `pattern_name` is in the prevented list.
    pub fn is_prevented(&self, pattern_name: &str) -> bool {
        self.expected_prevented.iter().any(|n| n == pattern_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use isospec_types::identifier::TransactionId;
    use isospec_types::dependency::Dependency;

    fn make_cycle(dep_types: &[DependencyType]) -> CycleInfo {
        let n = dep_types.len();
        let nodes: Vec<TransactionId> = (0..n).map(|i| TransactionId::new(i as u64)).collect();
        let edges: Vec<Dependency> = dep_types.iter().enumerate().map(|(i, &dt)| {
            Dependency::new(
                TransactionId::new(i as u64),
                TransactionId::new(((i + 1) % n) as u64),
                dt,
            )
        }).collect();
        CycleInfo { nodes, edges }
    }

    #[test]
    fn test_register_custom_pattern() {
        let mut catalog = AnomalyCatalog::new();
        assert_eq!(catalog.pattern_count(), 0);
        catalog.register(AnomalyPattern::new(
            "custom-1", "custom-pattern", AnomalyClass::G0,
            "A custom test pattern",
            vec![DependencyType::WriteWrite, DependencyType::WriteRead],
            2, vec!["Test 2024".into()],
        ));
        assert_eq!(catalog.pattern_count(), 1);
        let p = catalog.get_by_name("custom-pattern").unwrap();
        assert_eq!(p.anomaly_class, AnomalyClass::G0);
        assert_eq!(p.edge_pattern.len(), 2);
    }

    #[test]
    fn test_standard_catalog() {
        let catalog = AnomalyCatalog::with_standard_patterns();
        assert_eq!(catalog.pattern_count(), 9);
        assert!(catalog.get_by_name("dirty-write").is_some());
        assert!(catalog.get_by_name("write-skew").is_some());
        assert!(catalog.get_by_name("phantom-read").is_some());
        assert_eq!(catalog.get_by_class(AnomalyClass::G2).len(), 2);
    }

    #[test]
    fn test_exact_match() {
        let catalog = AnomalyCatalog::with_standard_patterns();
        let cycle = make_cycle(&[DependencyType::ReadWrite, DependencyType::ReadWrite]);
        let matches = catalog.match_cycle(&cycle);
        assert!(!matches.is_empty());
        assert!(matches[0].match_quality.is_exact());
        assert_eq!(matches[0].pattern.name, "write-skew");
    }

    #[test]
    fn test_rotated_match() {
        let catalog = AnomalyCatalog::with_standard_patterns();
        // lost-update = [RW, WW]; supply [WW, RW] → rotated by 1
        let cycle = make_cycle(&[DependencyType::WriteWrite, DependencyType::ReadWrite]);
        let best = catalog.best_match(&cycle).unwrap();
        assert_eq!(best.pattern.name, "lost-update");
        assert_eq!(best.match_quality, MatchQuality::Rotated);
        assert_eq!(best.rotation_offset, 1);
    }

    #[test]
    fn test_no_match_for_unrelated_cycle() {
        let catalog = AnomalyCatalog::with_standard_patterns();
        let cycle = make_cycle(&[
            DependencyType::WriteWrite, DependencyType::ReadWrite,
            DependencyType::WriteRead, DependencyType::PredicateWriteRead,
        ]);
        let matches = catalog.match_cycle(&cycle);
        for m in &matches {
            assert_ne!(m.match_quality, MatchQuality::Exact);
            assert_ne!(m.match_quality, MatchQuality::Rotated);
        }
    }

    #[test]
    fn test_best_match_selection() {
        let mut catalog = AnomalyCatalog::new();
        catalog.register(AnomalyPattern::new(
            "p1", "pattern-a", AnomalyClass::G0,
            "desc", vec![DependencyType::WriteWrite], 2, vec![],
        ));
        catalog.register(AnomalyPattern::new(
            "p2", "pattern-b", AnomalyClass::G1c,
            "desc",
            vec![DependencyType::WriteRead, DependencyType::WriteRead],
            2, vec![],
        ));
        // Cycle that matches pattern-b exactly
        let cycle = make_cycle(&[DependencyType::WriteRead, DependencyType::WriteRead]);
        let best = catalog.best_match(&cycle).unwrap();
        assert_eq!(best.pattern.name, "pattern-b");
        assert!(best.match_quality.is_exact());
    }

    #[test]
    fn test_rotate_matches_fn() {
        let pat = [DependencyType::ReadWrite, DependencyType::WriteWrite];
        assert_eq!(rotate_matches(&pat, &[DependencyType::WriteWrite, DependencyType::ReadWrite]), Some(1));
        assert_eq!(rotate_matches(&pat, &[DependencyType::ReadWrite, DependencyType::WriteWrite]), Some(0));
        assert_eq!(rotate_matches(&pat, &[DependencyType::WriteRead, DependencyType::WriteRead]), None);
    }

    #[test]
    fn test_engine_pattern_expectation() {
        let exp = EnginePatternExpectation::new(
            EngineKind::PostgreSQL, IsolationLevel::Serializable,
            vec!["dirty-write".into(), "aborted-read".into(),
                 "write-skew".into(), "phantom-read".into()],
            vec![],
        );
        assert!(exp.is_prevented("dirty-write"));
        assert!(exp.is_prevented("write-skew"));
        assert!(!exp.is_prevented("lost-update"));
    }

    #[test]
    fn test_remove_pattern() {
        let mut catalog = AnomalyCatalog::with_standard_patterns();
        let before = catalog.pattern_count();
        assert!(catalog.get_by_name("dirty-write").is_some());
        assert!(catalog.remove_pattern("dirty-write"));
        assert_eq!(catalog.pattern_count(), before - 1);
        assert!(catalog.get_by_name("dirty-write").is_none());
        assert!(catalog.get_by_name("write-skew").is_some());
        assert!(!catalog.remove_pattern("nonexistent"));
    }

    #[test]
    fn test_match_quality_ordering() {
        assert!(MatchQuality::Exact < MatchQuality::Rotated);
        assert!(MatchQuality::Rotated < MatchQuality::Subsequence);
        assert!(MatchQuality::Subsequence < MatchQuality::Partial);
    }

    #[test]
    fn test_is_subset_of() {
        let small = AnomalyPattern::new(
            "s", "small", AnomalyClass::G0, "",
            vec![DependencyType::WriteWrite], 2, vec![],
        );
        let big = AnomalyPattern::new(
            "b", "big", AnomalyClass::G0, "",
            vec![DependencyType::WriteWrite, DependencyType::ReadWrite], 2, vec![],
        );
        assert!(small.is_subset_of(&big));
        assert!(!big.is_subset_of(&small));
    }

    #[test]
    fn test_subsequence_match() {
        let mut catalog = AnomalyCatalog::new();
        catalog.register(AnomalyPattern::new(
            "sub", "sub-pat", AnomalyClass::G0,
            "single WW", vec![DependencyType::WriteWrite], 2, vec![],
        ));
        let cycle = make_cycle(&[
            DependencyType::WriteWrite, DependencyType::ReadWrite, DependencyType::WriteRead,
        ]);
        let matches = catalog.match_cycle(&cycle);
        assert!(!matches.is_empty());
        assert_eq!(matches[0].match_quality, MatchQuality::Subsequence);
    }

    #[test]
    fn test_partial_overlap() {
        let frac = partial_overlap(
            &[DependencyType::ReadWrite, DependencyType::WriteWrite],
            &[DependencyType::ReadWrite, DependencyType::WriteRead],
        );
        assert!((frac - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_get_by_class_empty() {
        assert!(AnomalyCatalog::new().get_by_class(AnomalyClass::G0).is_empty());
    }
}
