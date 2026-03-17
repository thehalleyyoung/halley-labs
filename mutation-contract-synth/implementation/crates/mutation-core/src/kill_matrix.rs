//! Kill matrix: 2-D sparse matrix of mutants × tests with kill/survive/timeout
//! results.  Provides row/column operations, statistics, serialisation, merging,
//! filtering, projection, minimal killing test set computation, and PIT XML
//! import.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// CellResult
// ---------------------------------------------------------------------------

/// Outcome of running one test against one mutant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CellResult {
    /// The test killed the mutant (output differs or assertion fires).
    Kill,
    /// The mutant survived (test passed).
    Survive,
    /// The test timed out on this mutant.
    Timeout,
    /// The test had an error on this mutant.
    Error,
    /// Not yet executed.
    Unknown,
}

impl CellResult {
    pub fn is_kill(self) -> bool {
        self == CellResult::Kill
    }

    pub fn is_survive(self) -> bool {
        self == CellResult::Survive
    }

    pub fn is_timeout(self) -> bool {
        self == CellResult::Timeout
    }
}

impl fmt::Display for CellResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CellResult::Kill => write!(f, "K"),
            CellResult::Survive => write!(f, "S"),
            CellResult::Timeout => write!(f, "T"),
            CellResult::Error => write!(f, "E"),
            CellResult::Unknown => write!(f, "?"),
        }
    }
}

// ---------------------------------------------------------------------------
// KillMatrix
// ---------------------------------------------------------------------------

/// Sparse 2-D matrix of mutants × tests.
///
/// Internally stores only non-`Unknown` cells.  The matrix is indexed by string
/// keys for both mutant IDs and test names.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KillMatrix {
    /// Ordered set of mutant IDs.
    mutant_ids: Vec<String>,
    /// Ordered set of test names.
    test_names: Vec<String>,
    /// Sparse storage: (mutant_index, test_index) → CellResult.
    cells: HashMap<(usize, usize), CellResult>,
    /// Mutant index lookup.
    mutant_index: HashMap<String, usize>,
    /// Test index lookup.
    test_index: HashMap<String, usize>,
    /// Mutants marked equivalent.
    equivalent: HashSet<usize>,
}

impl KillMatrix {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a matrix pre-sized for the given mutant and test counts.
    pub fn with_capacity(mutant_count: usize, test_count: usize) -> Self {
        Self {
            mutant_ids: Vec::with_capacity(mutant_count),
            test_names: Vec::with_capacity(test_count),
            cells: HashMap::with_capacity(mutant_count * test_count / 4), // sparse
            mutant_index: HashMap::with_capacity(mutant_count),
            test_index: HashMap::with_capacity(test_count),
            equivalent: HashSet::new(),
        }
    }

    // ----- Registration -----

    pub fn add_mutant(&mut self, id: impl Into<String>) -> usize {
        let id = id.into();
        if let Some(&idx) = self.mutant_index.get(&id) {
            return idx;
        }
        let idx = self.mutant_ids.len();
        self.mutant_index.insert(id.clone(), idx);
        self.mutant_ids.push(id);
        idx
    }

    pub fn add_test(&mut self, name: impl Into<String>) -> usize {
        let name = name.into();
        if let Some(&idx) = self.test_index.get(&name) {
            return idx;
        }
        let idx = self.test_names.len();
        self.test_index.insert(name.clone(), idx);
        self.test_names.push(name);
        idx
    }

    // ----- Cell access -----

    pub fn set(&mut self, mutant: &str, test: &str, result: CellResult) {
        let mi = self.add_mutant(mutant);
        let ti = self.add_test(test);
        self.cells.insert((mi, ti), result);
    }

    pub fn get(&self, mutant: &str, test: &str) -> CellResult {
        let mi = self.mutant_index.get(mutant);
        let ti = self.test_index.get(test);
        match (mi, ti) {
            (Some(&m), Some(&t)) => self
                .cells
                .get(&(m, t))
                .copied()
                .unwrap_or(CellResult::Unknown),
            _ => CellResult::Unknown,
        }
    }

    pub fn record_kill(&mut self, mutant: &str, test: &str) {
        self.set(mutant, test, CellResult::Kill);
    }

    pub fn record_survive(&mut self, mutant: &str, test: &str) {
        self.set(mutant, test, CellResult::Survive);
    }

    pub fn record_timeout(&mut self, mutant: &str, test: &str) {
        self.set(mutant, test, CellResult::Timeout);
    }

    pub fn mark_equivalent(&mut self, mutant: &str) {
        if let Some(&idx) = self.mutant_index.get(mutant) {
            self.equivalent.insert(idx);
        }
    }

    // ----- Row operations (per mutant) -----

    /// Which tests kill mutant `m`?
    pub fn killing_tests(&self, mutant: &str) -> Vec<String> {
        let mi = match self.mutant_index.get(mutant) {
            Some(&i) => i,
            None => return Vec::new(),
        };
        self.test_names
            .iter()
            .enumerate()
            .filter(|(ti, _)| self.cells.get(&(mi, *ti)).copied() == Some(CellResult::Kill))
            .map(|(_, name)| name.clone())
            .collect()
    }

    /// Is mutant `m` killed by at least one test?
    pub fn is_killed(&self, mutant: &str) -> bool {
        let mi = match self.mutant_index.get(mutant) {
            Some(&i) => i,
            None => return false,
        };
        (0..self.test_names.len())
            .any(|ti| self.cells.get(&(mi, ti)).copied() == Some(CellResult::Kill))
    }

    /// Row vector for a mutant (results for each test).
    pub fn mutant_row(&self, mutant: &str) -> Vec<(String, CellResult)> {
        let mi = match self.mutant_index.get(mutant) {
            Some(&i) => i,
            None => return Vec::new(),
        };
        self.test_names
            .iter()
            .enumerate()
            .map(|(ti, name)| {
                let r = self
                    .cells
                    .get(&(mi, ti))
                    .copied()
                    .unwrap_or(CellResult::Unknown);
                (name.clone(), r)
            })
            .collect()
    }

    // ----- Column operations (per test) -----

    /// Which mutants does test `t` kill?
    pub fn mutants_killed_by(&self, test: &str) -> Vec<String> {
        let ti = match self.test_index.get(test) {
            Some(&i) => i,
            None => return Vec::new(),
        };
        self.mutant_ids
            .iter()
            .enumerate()
            .filter(|(mi, _)| self.cells.get(&(*mi, ti)).copied() == Some(CellResult::Kill))
            .map(|(_, id)| id.clone())
            .collect()
    }

    /// Column vector for a test.
    pub fn test_column(&self, test: &str) -> Vec<(String, CellResult)> {
        let ti = match self.test_index.get(test) {
            Some(&i) => i,
            None => return Vec::new(),
        };
        self.mutant_ids
            .iter()
            .enumerate()
            .map(|(mi, id)| {
                let r = self
                    .cells
                    .get(&(mi, ti))
                    .copied()
                    .unwrap_or(CellResult::Unknown);
                (id.clone(), r)
            })
            .collect()
    }

    /// Number of mutants killed by test `t`.
    pub fn test_kill_count(&self, test: &str) -> usize {
        self.mutants_killed_by(test).len()
    }

    // ----- Dimensions -----

    pub fn num_mutants(&self) -> usize {
        self.mutant_ids.len()
    }

    pub fn num_tests(&self) -> usize {
        self.test_names.len()
    }

    pub fn num_cells(&self) -> usize {
        self.cells.len()
    }

    pub fn mutant_ids(&self) -> &[String] {
        &self.mutant_ids
    }

    pub fn test_names(&self) -> &[String] {
        &self.test_names
    }

    pub fn is_equivalent(&self, mutant: &str) -> bool {
        self.mutant_index
            .get(mutant)
            .map(|&i| self.equivalent.contains(&i))
            .unwrap_or(false)
    }

    // ----- Statistics -----

    /// Killed mutants (non-equivalent).
    pub fn killed_mutants(&self) -> Vec<String> {
        self.mutant_ids
            .iter()
            .enumerate()
            .filter(|(mi, _)| {
                !self.equivalent.contains(mi)
                    && (0..self.test_names.len())
                        .any(|ti| self.cells.get(&(*mi, ti)).copied() == Some(CellResult::Kill))
            })
            .map(|(_, id)| id.clone())
            .collect()
    }

    /// Survived mutants (non-equivalent, not killed).
    pub fn survived_mutants(&self) -> Vec<String> {
        self.mutant_ids
            .iter()
            .enumerate()
            .filter(|(mi, _)| {
                !self.equivalent.contains(mi)
                    && !(0..self.test_names.len())
                        .any(|ti| self.cells.get(&(*mi, ti)).copied() == Some(CellResult::Kill))
            })
            .map(|(_, id)| id.clone())
            .collect()
    }

    /// Mutation score: killed / (total − equivalent).
    pub fn mutation_score(&self) -> f64 {
        let total = self.mutant_ids.len();
        let equiv = self.equivalent.len();
        let denom = total.saturating_sub(equiv);
        if denom == 0 {
            return 1.0;
        }
        self.killed_mutants().len() as f64 / denom as f64
    }

    /// Per-test effectiveness (how many mutants each test kills).
    pub fn test_effectiveness(&self) -> Vec<(String, usize)> {
        self.test_names
            .iter()
            .map(|name| (name.clone(), self.test_kill_count(name)))
            .collect()
    }

    /// Per-test effectiveness sorted descending.
    pub fn test_effectiveness_sorted(&self) -> Vec<(String, usize)> {
        let mut eff = self.test_effectiveness();
        eff.sort_by(|a, b| b.1.cmp(&a.1));
        eff
    }

    // ----- Minimal killing test set (greedy set cover) -----

    /// Compute a minimal set of tests that together kill all killable mutants.
    /// Uses a greedy set-cover heuristic.
    pub fn minimal_killing_set(&self) -> Vec<String> {
        let mut uncovered: HashSet<usize> = HashSet::new();
        for (mi, _) in self.mutant_ids.iter().enumerate() {
            if !self.equivalent.contains(&mi) {
                let killed = (0..self.test_names.len())
                    .any(|ti| self.cells.get(&(mi, ti)).copied() == Some(CellResult::Kill));
                if killed {
                    uncovered.insert(mi);
                }
            }
        }

        let mut selected = Vec::new();
        let mut remaining_tests: HashSet<usize> = (0..self.test_names.len()).collect();

        while !uncovered.is_empty() && !remaining_tests.is_empty() {
            // Pick the test that covers the most uncovered mutants.
            let best_test = remaining_tests
                .iter()
                .max_by_key(|&&ti| {
                    uncovered
                        .iter()
                        .filter(|&&mi| self.cells.get(&(mi, ti)).copied() == Some(CellResult::Kill))
                        .count()
                })
                .copied();

            match best_test {
                Some(ti) => {
                    remaining_tests.remove(&ti);
                    let newly_covered: Vec<usize> = uncovered
                        .iter()
                        .filter(|&&mi| self.cells.get(&(mi, ti)).copied() == Some(CellResult::Kill))
                        .copied()
                        .collect();
                    if newly_covered.is_empty() {
                        break; // no more progress
                    }
                    for mi in &newly_covered {
                        uncovered.remove(mi);
                    }
                    selected.push(self.test_names[ti].clone());
                }
                None => break,
            }
        }

        selected
    }

    // ----- Merging -----

    /// Merge another kill matrix into this one.
    pub fn merge(&mut self, other: &KillMatrix) {
        for (id, _mi) in &other.mutant_index {
            self.add_mutant(id.clone());
        }
        for (name, _ti) in &other.test_index {
            self.add_test(name.clone());
        }
        for (&(omi, oti), &result) in &other.cells {
            let mid = &other.mutant_ids[omi];
            let tname = &other.test_names[oti];
            self.set(mid, tname, result);
        }
        for &ei in &other.equivalent {
            if ei < other.mutant_ids.len() {
                self.mark_equivalent(&other.mutant_ids[ei]);
            }
        }
    }

    // ----- Filtering & projection -----

    /// Create a new matrix containing only the specified mutants.
    pub fn filter_mutants(&self, keep: &HashSet<String>) -> KillMatrix {
        let mut new = KillMatrix::new();
        for id in &self.mutant_ids {
            if keep.contains(id) {
                new.add_mutant(id.clone());
            }
        }
        for name in &self.test_names {
            new.add_test(name.clone());
        }
        for (&(mi, ti), &result) in &self.cells {
            let mid = &self.mutant_ids[mi];
            if keep.contains(mid) {
                let tname = &self.test_names[ti];
                new.set(mid, tname, result);
            }
        }
        new
    }

    /// Create a new matrix containing only the specified tests.
    pub fn filter_tests(&self, keep: &HashSet<String>) -> KillMatrix {
        let mut new = KillMatrix::new();
        for id in &self.mutant_ids {
            new.add_mutant(id.clone());
        }
        for name in &self.test_names {
            if keep.contains(name) {
                new.add_test(name.clone());
            }
        }
        for (&(mi, ti), &result) in &self.cells {
            let tname = &self.test_names[ti];
            if keep.contains(tname) {
                let mid = &self.mutant_ids[mi];
                new.set(mid, tname, result);
            }
        }
        new
    }

    // ----- Serialisation: CSV -----

    /// Export to CSV format.
    pub fn to_csv(&self) -> String {
        let mut out = String::new();
        out.push_str("mutant");
        for name in &self.test_names {
            out.push(',');
            out.push_str(name);
        }
        out.push('\n');
        for (mi, id) in self.mutant_ids.iter().enumerate() {
            out.push_str(id);
            for ti in 0..self.test_names.len() {
                out.push(',');
                let r = self
                    .cells
                    .get(&(mi, ti))
                    .copied()
                    .unwrap_or(CellResult::Unknown);
                out.push_str(&format!("{}", r));
            }
            out.push('\n');
        }
        out
    }

    /// Import from CSV format.
    pub fn from_csv(csv: &str) -> Result<Self, String> {
        let mut lines = csv.lines();
        let header = lines.next().ok_or("empty CSV")?;
        let cols: Vec<&str> = header.split(',').collect();
        if cols.is_empty() || cols[0] != "mutant" {
            return Err("expected 'mutant' as first column".into());
        }
        let test_names: Vec<String> = cols[1..].iter().map(|s| s.to_string()).collect();

        let mut matrix = KillMatrix::new();
        for name in &test_names {
            matrix.add_test(name.clone());
        }

        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let vals: Vec<&str> = line.split(',').collect();
            if vals.is_empty() {
                continue;
            }
            let mutant_id = vals[0].to_string();
            matrix.add_mutant(mutant_id.clone());
            for (i, val) in vals[1..].iter().enumerate() {
                if i < test_names.len() {
                    let result = match val.trim() {
                        "K" => CellResult::Kill,
                        "S" => CellResult::Survive,
                        "T" => CellResult::Timeout,
                        "E" => CellResult::Error,
                        _ => CellResult::Unknown,
                    };
                    if result != CellResult::Unknown {
                        matrix.set(&mutant_id, &test_names[i], result);
                    }
                }
            }
        }
        Ok(matrix)
    }

    // ----- Serialisation: JSON -----

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    // ----- PIT XML import -----

    /// Parse a simplified PIT-style XML mutation report.
    ///
    /// Expected format (simplified):
    /// ```xml
    /// <mutations>
    ///   <mutation detected="true" status="KILLED">
    ///     <mutatedClass>com.Foo</mutatedClass>
    ///     <mutatedMethod>bar</mutatedMethod>
    ///     <mutator>AOR</mutator>
    ///     <killingTest>testBar</killingTest>
    ///     <description>...</description>
    ///   </mutation>
    /// </mutations>
    /// ```
    pub fn from_pit_xml(xml: &str) -> Result<Self, String> {
        let mut matrix = KillMatrix::new();
        let mut mutant_counter = 0usize;

        // Simple line-by-line parser (no external XML dep).
        let mut in_mutation = false;
        let mut detected = false;
        let mut mutant_id = String::new();
        let mut killing_test = String::new();

        for line in xml.lines() {
            let trimmed = line.trim();

            if trimmed.starts_with("<mutation") {
                in_mutation = true;
                detected = trimmed.contains("detected=\"true\"");
                mutant_counter += 1;
                mutant_id = format!("pit-m{}", mutant_counter);
                killing_test.clear();
            } else if trimmed == "</mutation>" {
                if in_mutation {
                    matrix.add_mutant(&mutant_id);
                    if detected && !killing_test.is_empty() {
                        matrix.record_kill(&mutant_id, &killing_test);
                    } else if !detected {
                        // Survived – add a synthetic survive entry with a sentinel test
                        matrix.add_test("__all_tests");
                        matrix.record_survive(&mutant_id, "__all_tests");
                    }
                }
                in_mutation = false;
            } else if in_mutation {
                if let Some(val) = extract_xml_value(trimmed, "killingTest") {
                    killing_test = val;
                    matrix.add_test(&killing_test);
                }
            }
        }

        Ok(matrix)
    }

    // ----- Kill equivalence classes -----

    /// Group mutants by their killing test sets.
    pub fn kill_equivalence_classes(&self) -> Vec<(Vec<String>, Vec<String>)> {
        let mut classes: HashMap<Vec<String>, Vec<String>> = HashMap::new();
        for id in &self.mutant_ids {
            let mut tests = self.killing_tests(id);
            tests.sort();
            classes.entry(tests).or_default().push(id.clone());
        }
        classes.into_iter().collect()
    }

    /// Dominator ordering: killed mutants sorted by number of killing tests (ascending).
    pub fn dominator_ordering(&self) -> Vec<(String, usize)> {
        let mut entries: Vec<(String, usize)> = self
            .killed_mutants()
            .into_iter()
            .map(|id| {
                let count = self.killing_tests(&id).len();
                (id, count)
            })
            .collect();
        entries.sort_by_key(|(_, c)| *c);
        entries
    }
}

impl fmt::Display for KillMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "KillMatrix: {} mutants × {} tests ({} cells)",
            self.num_mutants(),
            self.num_tests(),
            self.num_cells()
        )?;
        writeln!(f, "  Killed:     {}", self.killed_mutants().len())?;
        writeln!(f, "  Survived:   {}", self.survived_mutants().len())?;
        writeln!(f, "  Equivalent: {}", self.equivalent.len())?;
        writeln!(f, "  Score:      {:.1}%", self.mutation_score() * 100.0)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn extract_xml_value(line: &str, tag: &str) -> Option<String> {
    let open = format!("<{}>", tag);
    let close = format!("</{}>", tag);
    if let Some(start) = line.find(&open) {
        let value_start = start + open.len();
        if let Some(end) = line.find(&close) {
            return Some(line[value_start..end].to_string());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_matrix() -> KillMatrix {
        let mut m = KillMatrix::new();
        m.add_mutant("m1");
        m.add_mutant("m2");
        m.add_mutant("m3");
        m.add_test("t1");
        m.add_test("t2");
        m.add_test("t3");
        m.record_kill("m1", "t1");
        m.record_kill("m1", "t2");
        m.record_kill("m2", "t1");
        m.record_survive("m3", "t1");
        m.record_survive("m3", "t2");
        m.record_survive("m3", "t3");
        m
    }

    #[test]
    fn test_basic_operations() {
        let m = make_matrix();
        assert_eq!(m.num_mutants(), 3);
        assert_eq!(m.num_tests(), 3);
        assert!(m.is_killed("m1"));
        assert!(m.is_killed("m2"));
        assert!(!m.is_killed("m3"));
    }

    #[test]
    fn test_killing_tests() {
        let m = make_matrix();
        let kills = m.killing_tests("m1");
        assert!(kills.contains(&"t1".to_string()));
        assert!(kills.contains(&"t2".to_string()));
        assert_eq!(kills.len(), 2);
    }

    #[test]
    fn test_mutants_killed_by() {
        let m = make_matrix();
        let killed = m.mutants_killed_by("t1");
        assert!(killed.contains(&"m1".to_string()));
        assert!(killed.contains(&"m2".to_string()));
    }

    #[test]
    fn test_mutation_score() {
        let m = make_matrix();
        // 2 killed / 3 total = 0.666...
        let score = m.mutation_score();
        assert!((score - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_mutation_score_with_equivalent() {
        let mut m = make_matrix();
        m.mark_equivalent("m3");
        // 2 killed / (3 - 1 equiv) = 1.0
        let score = m.mutation_score();
        assert!((score - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_killed_survived() {
        let m = make_matrix();
        assert_eq!(m.killed_mutants().len(), 2);
        assert_eq!(m.survived_mutants().len(), 1);
    }

    #[test]
    fn test_cell_access() {
        let m = make_matrix();
        assert_eq!(m.get("m1", "t1"), CellResult::Kill);
        assert_eq!(m.get("m3", "t1"), CellResult::Survive);
        assert_eq!(m.get("m2", "t3"), CellResult::Unknown);
        assert_eq!(m.get("nonexistent", "t1"), CellResult::Unknown);
    }

    #[test]
    fn test_mutant_row() {
        let m = make_matrix();
        let row = m.mutant_row("m1");
        assert_eq!(row.len(), 3);
        assert!(row.iter().any(|(n, r)| n == "t1" && *r == CellResult::Kill));
    }

    #[test]
    fn test_test_column() {
        let m = make_matrix();
        let col = m.test_column("t1");
        assert_eq!(col.len(), 3);
    }

    #[test]
    fn test_test_effectiveness() {
        let m = make_matrix();
        let eff = m.test_effectiveness();
        let t1_eff = eff.iter().find(|(n, _)| n == "t1").unwrap();
        assert_eq!(t1_eff.1, 2);
    }

    #[test]
    fn test_minimal_killing_set() {
        let m = make_matrix();
        let mks = m.minimal_killing_set();
        // t1 kills both m1 and m2, so it should suffice
        assert_eq!(mks.len(), 1);
        assert_eq!(mks[0], "t1");
    }

    #[test]
    fn test_minimal_killing_set_multiple() {
        let mut m = KillMatrix::new();
        m.record_kill("m1", "t1");
        m.record_kill("m2", "t2");
        m.record_kill("m3", "t3");
        let mks = m.minimal_killing_set();
        // Each test kills exactly one mutant, need all 3
        assert_eq!(mks.len(), 3);
    }

    #[test]
    fn test_merge() {
        let mut m1 = KillMatrix::new();
        m1.record_kill("m1", "t1");

        let mut m2 = KillMatrix::new();
        m2.record_kill("m2", "t2");

        m1.merge(&m2);
        assert_eq!(m1.num_mutants(), 2);
        assert_eq!(m1.num_tests(), 2);
        assert!(m1.is_killed("m1"));
        assert!(m1.is_killed("m2"));
    }

    #[test]
    fn test_filter_mutants() {
        let m = make_matrix();
        let keep: HashSet<String> = ["m1".into(), "m2".into()].into();
        let filtered = m.filter_mutants(&keep);
        assert_eq!(filtered.num_mutants(), 2);
        assert!(filtered.is_killed("m1"));
    }

    #[test]
    fn test_filter_tests() {
        let m = make_matrix();
        let keep: HashSet<String> = ["t1".into()].into();
        let filtered = m.filter_tests(&keep);
        assert_eq!(filtered.num_tests(), 1);
        assert!(filtered.is_killed("m1"));
    }

    #[test]
    fn test_csv_roundtrip() {
        let m = make_matrix();
        let csv = m.to_csv();
        let restored = KillMatrix::from_csv(&csv).unwrap();
        assert_eq!(restored.num_mutants(), m.num_mutants());
        assert_eq!(restored.num_tests(), m.num_tests());
        assert_eq!(restored.get("m1", "t1"), CellResult::Kill);
        assert_eq!(restored.get("m3", "t1"), CellResult::Survive);
    }

    #[test]
    fn test_json_roundtrip() {
        let m = make_matrix();
        let json = m.to_json().unwrap();
        let restored = KillMatrix::from_json(&json).unwrap();
        assert_eq!(restored.num_mutants(), m.num_mutants());
    }

    #[test]
    fn test_pit_xml_import() {
        let xml = r#"<mutations>
  <mutation detected="true" status="KILLED">
    <mutatedClass>Foo</mutatedClass>
    <mutatedMethod>bar</mutatedMethod>
    <mutator>AOR</mutator>
    <killingTest>testBar</killingTest>
  </mutation>
  <mutation detected="false" status="SURVIVED">
    <mutatedClass>Foo</mutatedClass>
    <mutatedMethod>baz</mutatedMethod>
    <mutator>ROR</mutator>
  </mutation>
</mutations>"#;
        let m = KillMatrix::from_pit_xml(xml).unwrap();
        assert_eq!(m.num_mutants(), 2);
        assert!(m.is_killed("pit-m1"));
        assert!(!m.is_killed("pit-m2"));
    }

    #[test]
    fn test_kill_equivalence_classes() {
        let m = make_matrix();
        let classes = m.kill_equivalence_classes();
        assert!(!classes.is_empty());
    }

    #[test]
    fn test_dominator_ordering() {
        let m = make_matrix();
        let dom = m.dominator_ordering();
        // m2 killed by 1 test, m1 killed by 2
        assert!(!dom.is_empty());
        assert_eq!(dom[0].0, "m2");
        assert_eq!(dom[0].1, 1);
    }

    #[test]
    fn test_display() {
        let m = make_matrix();
        let s = format!("{}", m);
        assert!(s.contains("KillMatrix"));
        assert!(s.contains("Score"));
    }

    #[test]
    fn test_with_capacity() {
        let m = KillMatrix::with_capacity(100, 50);
        assert_eq!(m.num_mutants(), 0);
        assert_eq!(m.num_tests(), 0);
    }

    #[test]
    fn test_empty_matrix() {
        let m = KillMatrix::new();
        assert!((m.mutation_score() - 1.0).abs() < 0.01);
        assert!(m.killed_mutants().is_empty());
        assert!(m.survived_mutants().is_empty());
        assert!(m.minimal_killing_set().is_empty());
    }

    #[test]
    fn test_timeout_cells() {
        let mut m = KillMatrix::new();
        m.record_timeout("m1", "t1");
        assert_eq!(m.get("m1", "t1"), CellResult::Timeout);
        assert!(!m.is_killed("m1"));
    }

    #[test]
    fn test_cell_result_display() {
        assert_eq!(format!("{}", CellResult::Kill), "K");
        assert_eq!(format!("{}", CellResult::Survive), "S");
        assert_eq!(format!("{}", CellResult::Timeout), "T");
        assert_eq!(format!("{}", CellResult::Error), "E");
        assert_eq!(format!("{}", CellResult::Unknown), "?");
    }

    #[test]
    fn test_add_mutant_idempotent() {
        let mut m = KillMatrix::new();
        let i1 = m.add_mutant("m1");
        let i2 = m.add_mutant("m1");
        assert_eq!(i1, i2);
        assert_eq!(m.num_mutants(), 1);
    }

    #[test]
    fn test_add_test_idempotent() {
        let mut m = KillMatrix::new();
        let i1 = m.add_test("t1");
        let i2 = m.add_test("t1");
        assert_eq!(i1, i2);
        assert_eq!(m.num_tests(), 1);
    }

    #[test]
    fn test_csv_empty() {
        let m = KillMatrix::new();
        let csv = m.to_csv();
        assert!(csv.starts_with("mutant"));
    }

    #[test]
    fn test_extract_xml_value() {
        assert_eq!(
            extract_xml_value("<killingTest>testFoo</killingTest>", "killingTest"),
            Some("testFoo".into())
        );
        assert_eq!(extract_xml_value("<other>val</other>", "killingTest"), None);
    }

    #[test]
    fn test_test_effectiveness_sorted() {
        let m = make_matrix();
        let sorted = m.test_effectiveness_sorted();
        // t1 kills 2, should be first
        assert_eq!(sorted[0].0, "t1");
        assert_eq!(sorted[0].1, 2);
    }
}
