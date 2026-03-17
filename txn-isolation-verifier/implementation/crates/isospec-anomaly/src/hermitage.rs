//! Hermitage test case catalog for transaction isolation anomaly verification.
//!
//! Encodes the standard Hermitage test suite (<https://github.com/ept/hermitage>)
//! as structured data so that expected anomaly behaviour can be queried
//! per engine and isolation level.

use std::collections::HashMap;
use std::fmt;

use isospec_types::config::EngineKind;
use isospec_types::isolation::{AnomalyClass, IsolationLevel};

/// A single SQL-level action inside a Hermitage test step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepAction {
    Begin,
    Read { table: String, key: String },
    Write { table: String, key: String, value: String },
    Insert { table: String, key: String, value: String },
    Delete { table: String, key: String },
    Commit,
    Abort,
    SelectWhere { table: String, condition: String },
    SelectCount { table: String, condition: String },
}

impl fmt::Display for StepAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Begin => write!(f, "BEGIN"),
            Self::Read { table, key } => write!(f, "READ {table}.{key}"),
            Self::Write { table, key, value } => write!(f, "WRITE {table}.{key} = {value}"),
            Self::Insert { table, key, value } => write!(f, "INSERT {table}.{key} = {value}"),
            Self::Delete { table, key } => write!(f, "DELETE {table}.{key}"),
            Self::Commit => write!(f, "COMMIT"),
            Self::Abort => write!(f, "ABORT"),
            Self::SelectWhere { table, condition } => {
                write!(f, "SELECT * FROM {table} WHERE {condition}")
            }
            Self::SelectCount { table, condition } => {
                write!(f, "SELECT COUNT(*) FROM {table} WHERE {condition}")
            }
        }
    }
}

/// One interleaved step executed by a particular transaction.
#[derive(Debug, Clone)]
pub struct HermitageStep {
    pub txn: u8,
    pub action: StepAction,
    pub description: String,
}

impl HermitageStep {
    pub fn new(txn: u8, action: StepAction, desc: impl Into<String>) -> Self {
        Self { txn, action, description: desc.into() }
    }
}

/// Whether a Hermitage test anomaly is allowed or prevented.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpectedBehavior {
    Allowed,
    Prevented,
    EngineSpecific(String),
}

impl ExpectedBehavior {
    pub fn is_allowed(&self) -> bool { matches!(self, Self::Allowed) }
    pub fn is_prevented(&self) -> bool { matches!(self, Self::Prevented) }
}

impl fmt::Display for ExpectedBehavior {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Allowed => write!(f, "Allowed"),
            Self::Prevented => write!(f, "Prevented"),
            Self::EngineSpecific(s) => write!(f, "EngineSpecific({s})"),
        }
    }
}

/// Expected outcome for a single (engine, level) pair.
#[derive(Debug, Clone)]
pub struct EngineOutcome {
    pub engine: EngineKind,
    pub level: IsolationLevel,
    pub behavior: ExpectedBehavior,
}

impl EngineOutcome {
    pub fn new(engine: EngineKind, level: IsolationLevel, behavior: ExpectedBehavior) -> Self {
        Self { engine, level, behavior }
    }
}

/// A single Hermitage test case with expected outcomes per engine × level.
#[derive(Debug, Clone)]
pub struct HermitageTestCase {
    pub id: String,
    pub name: String,
    pub anomaly_class: AnomalyClass,
    pub description: String,
    pub steps: Vec<HermitageStep>,
    pub expected_outcomes: Vec<EngineOutcome>,
}

impl HermitageTestCase {
    pub fn new(
        id: impl Into<String>, name: impl Into<String>, anomaly_class: AnomalyClass,
        description: impl Into<String>, steps: Vec<HermitageStep>,
        expected_outcomes: Vec<EngineOutcome>,
    ) -> Self {
        Self {
            id: id.into(), name: name.into(), anomaly_class,
            description: description.into(), steps, expected_outcomes,
        }
    }

    /// Look up the expected behaviour for a specific engine and isolation level.
    pub fn expected_for(&self, engine: EngineKind, level: IsolationLevel) -> Option<&ExpectedBehavior> {
        self.expected_outcomes.iter()
            .find(|o| o.engine == engine && o.level == level)
            .map(|o| &o.behavior)
    }
}

/// One cell in the full engine × level × test expected-outcome matrix.
#[derive(Debug, Clone)]
pub struct MatrixEntry {
    pub test_id: String,
    pub engine: EngineKind,
    pub level: IsolationLevel,
    pub behavior: ExpectedBehavior,
}

/// The complete catalogue of Hermitage tests, indexed for efficient lookup.
#[derive(Debug, Clone)]
pub struct HermitageCatalog {
    test_cases: Vec<HermitageTestCase>,
    by_anomaly: HashMap<AnomalyClass, Vec<usize>>,
    by_id: HashMap<String, usize>,
}

impl HermitageCatalog {
    /// Build the catalogue with every standard Hermitage test.
    pub fn new() -> Self {
        let test_cases = build_standard_tests();
        let mut by_anomaly: HashMap<AnomalyClass, Vec<usize>> = HashMap::new();
        let mut by_id: HashMap<String, usize> = HashMap::new();
        for (idx, tc) in test_cases.iter().enumerate() {
            by_anomaly.entry(tc.anomaly_class).or_default().push(idx);
            by_id.insert(tc.id.clone(), idx);
        }
        Self { test_cases, by_anomaly, by_id }
    }

    pub fn get_test(&self, id: &str) -> Option<&HermitageTestCase> {
        self.by_id.get(id).map(|&idx| &self.test_cases[idx])
    }

    pub fn tests_for_anomaly(&self, class: AnomalyClass) -> Vec<&HermitageTestCase> {
        self.by_anomaly.get(&class)
            .map(|idxs| idxs.iter().map(|&i| &self.test_cases[i]).collect())
            .unwrap_or_default()
    }

    pub fn all_tests(&self) -> &[HermitageTestCase] { &self.test_cases }
    pub fn test_count(&self) -> usize { self.test_cases.len() }

    /// Produce the full engine × level × test expected-outcome matrix.
    pub fn expected_matrix(&self) -> Vec<MatrixEntry> {
        let mut entries = Vec::new();
        for tc in &self.test_cases {
            for o in &tc.expected_outcomes {
                entries.push(MatrixEntry {
                    test_id: tc.id.clone(), engine: o.engine,
                    level: o.level, behavior: o.behavior.clone(),
                });
            }
        }
        entries
    }

    /// Tests whose anomaly is *prevented* for the given engine + level.
    pub fn tests_prevented_at(&self, engine: EngineKind, level: IsolationLevel) -> Vec<&HermitageTestCase> {
        self.test_cases.iter()
            .filter(|tc| tc.expected_for(engine, level).map_or(false, |b| b.is_prevented()))
            .collect()
    }

    /// Tests whose anomaly is *allowed* for the given engine + level.
    pub fn tests_allowed_at(&self, engine: EngineKind, level: IsolationLevel) -> Vec<&HermitageTestCase> {
        self.test_cases.iter()
            .filter(|tc| tc.expected_for(engine, level).map_or(false, |b| b.is_allowed()))
            .collect()
    }
}

impl Default for HermitageCatalog {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn oc(engine: EngineKind, level: IsolationLevel, behavior: ExpectedBehavior) -> EngineOutcome {
    EngineOutcome::new(engine, level, behavior)
}

/// Levels at index >= `threshold_idx` are Prevented; below are Allowed.
fn by_threshold(engine: EngineKind, levels: &[IsolationLevel], threshold_idx: usize) -> Vec<EngineOutcome> {
    levels.iter().enumerate().map(|(i, &lvl)| {
        let b = if i >= threshold_idx { ExpectedBehavior::Prevented } else { ExpectedBehavior::Allowed };
        oc(engine, lvl, b)
    }).collect()
}

fn pg() -> Vec<IsolationLevel> {
    use IsolationLevel::*;
    vec![ReadUncommitted, ReadCommitted, RepeatableRead, Serializable]
}
fn my() -> Vec<IsolationLevel> {
    use IsolationLevel::*;
    vec![ReadUncommitted, ReadCommitted, RepeatableRead, Serializable]
}
fn ss() -> Vec<IsolationLevel> {
    use IsolationLevel::*;
    vec![ReadUncommitted, ReadCommitted, RepeatableRead, Snapshot, Serializable]
}
fn s(v: &str) -> String { v.to_owned() }

fn step(txn: u8, action: StepAction, desc: &str) -> HermitageStep {
    HermitageStep::new(txn, action, desc)
}
fn w(key: &str, val: &str) -> StepAction {
    StepAction::Write { table: s("test"), key: s(key), value: s(val) }
}
fn r(key: &str) -> StepAction {
    StepAction::Read { table: s("test"), key: s(key) }
}

// ---------------------------------------------------------------------------
// Standard test builders
// ---------------------------------------------------------------------------

fn build_standard_tests() -> Vec<HermitageTestCase> {
    vec![
        build_g0(), build_g1a(), build_g1b(), build_g1c(),
        build_g2item_lost_update(), build_g2item_write_skew(), build_g2_phantom(),
    ]
}

fn build_g0() -> HermitageTestCase {
    let steps = vec![
        step(1, StepAction::Begin, "T1 begins"), step(2, StepAction::Begin, "T2 begins"),
        step(1, w("x", "1"), "T1 writes x=1"), step(2, w("x", "2"), "T2 writes x=2"),
        step(1, w("y", "1"), "T1 writes y=1"), step(2, w("y", "2"), "T2 writes y=2"),
        step(1, StepAction::Commit, "T1 commits"), step(2, StepAction::Commit, "T2 commits"),
    ];
    let mut out = by_threshold(EngineKind::PostgreSQL, &pg(), 0);
    out.extend(by_threshold(EngineKind::MySQL, &my(), 0));
    out.extend(by_threshold(EngineKind::SqlServer, &ss(), 0));
    HermitageTestCase::new("g0-dirty-write", "Dirty Write (G0)", AnomalyClass::G0,
        "Two transactions concurrently write to x and y. A dirty-write anomaly \
         would result in x and y having values written by different transactions.",
        steps, out)
}

fn build_g1a() -> HermitageTestCase {
    let steps = vec![
        step(1, StepAction::Begin, "T1 begins"), step(2, StepAction::Begin, "T2 begins"),
        step(1, w("x", "1"), "T1 writes x=1"), step(2, r("x"), "T2 reads x"),
        step(1, StepAction::Abort, "T1 aborts"), step(2, r("x"), "T2 reads x again"),
        step(2, StepAction::Commit, "T2 commits"),
    ];
    let mut out = by_threshold(EngineKind::PostgreSQL, &pg(), 1);
    out.extend(by_threshold(EngineKind::MySQL, &my(), 1));
    out.extend(by_threshold(EngineKind::SqlServer, &ss(), 1));
    HermitageTestCase::new("g1a-aborted-read", "Aborted Read (G1a)", AnomalyClass::G1a,
        "T1 writes x, T2 reads x, T1 aborts. If T2 saw the uncommitted \
         write, this is an aborted-read anomaly.", steps, out)
}

fn build_g1b() -> HermitageTestCase {
    let steps = vec![
        step(1, StepAction::Begin, "T1 begins"), step(2, StepAction::Begin, "T2 begins"),
        step(1, w("x", "1"), "T1 writes x=1"), step(2, r("x"), "T2 reads x"),
        step(1, w("x", "2"), "T1 overwrites x=2"), step(1, StepAction::Commit, "T1 commits"),
        step(2, r("x"), "T2 reads x again"), step(2, StepAction::Commit, "T2 commits"),
    ];
    // PG: RU allowed, RC+ prevented. MySQL: all prevented. SS: RU allowed, RC+ prevented.
    let mut out = by_threshold(EngineKind::PostgreSQL, &pg(), 1);
    out.extend(by_threshold(EngineKind::MySQL, &my(), 0));
    out.extend(by_threshold(EngineKind::SqlServer, &ss(), 1));
    HermitageTestCase::new("g1b-intermediate-read", "Intermediate Read (G1b)", AnomalyClass::G1b,
        "T1 writes x=1, T2 reads x, T1 overwrites x=2 and commits. If T2 \
         observed the intermediate value, this is an intermediate-read anomaly.",
        steps, out)
}

fn build_g1c() -> HermitageTestCase {
    let steps = vec![
        step(1, StepAction::Begin, "T1 begins"), step(2, StepAction::Begin, "T2 begins"),
        step(1, w("x", "1"), "T1 writes x=1"), step(2, w("y", "1"), "T2 writes y=1"),
        step(1, r("y"), "T1 reads y"), step(2, r("x"), "T2 reads x"),
        step(1, StepAction::Commit, "T1 commits"), step(2, StepAction::Commit, "T2 commits"),
    ];
    let mut out = by_threshold(EngineKind::PostgreSQL, &pg(), 2);
    out.extend(by_threshold(EngineKind::MySQL, &my(), 2));
    out.extend(by_threshold(EngineKind::SqlServer, &ss(), 2));
    HermitageTestCase::new("g1c-circular-info-flow", "Circular Information Flow (G1c)",
        AnomalyClass::G1c,
        "T1 writes x, T2 writes y, T1 reads y, T2 reads x — both commit. \
         Cycle through write-read edges constitutes circular information flow.",
        steps, out)
}

fn build_g2item_lost_update() -> HermitageTestCase {
    let steps = vec![
        step(1, StepAction::Begin, "T1 begins"), step(2, StepAction::Begin, "T2 begins"),
        step(1, r("x"), "T1 reads x"), step(2, r("x"), "T2 reads x"),
        step(1, w("x", "x+1"), "T1 writes x=x+1"),
        step(2, w("x", "x+1"), "T2 writes x=x+1"),
        step(1, StepAction::Commit, "T1 commits"), step(2, StepAction::Commit, "T2 commits"),
    ];
    let mut out = by_threshold(EngineKind::PostgreSQL, &pg(), 2);
    out.extend(by_threshold(EngineKind::MySQL, &my(), 2));
    out.extend(by_threshold(EngineKind::SqlServer, &ss(), 2));
    HermitageTestCase::new("g2item-lost-update", "Lost Update (G2-item)", AnomalyClass::G2Item,
        "Both transactions read x, then write x=x+1 and commit. One increment \
         is lost, violating item-level anti-dependency constraints.", steps, out)
}

fn build_g2item_write_skew() -> HermitageTestCase {
    use EngineKind::*; use ExpectedBehavior::*; use IsolationLevel::*;
    let steps = vec![
        step(1, StepAction::Begin, "T1 begins"), step(2, StepAction::Begin, "T2 begins"),
        step(1, r("x"), "T1 reads x"), step(1, r("y"), "T1 reads y"),
        step(2, r("x"), "T2 reads x"), step(2, r("y"), "T2 reads y"),
        step(1, w("y", "f(x)"), "T1 writes y based on x"),
        step(2, w("x", "g(y)"), "T2 writes x based on y"),
        step(1, StepAction::Commit, "T1 commits"), step(2, StepAction::Commit, "T2 commits"),
    ];
    // PG: RU/RC/RR allowed, Serializable prevented (SSI).
    // MySQL: RU/RC allowed, RR/Serializable prevented (lock-based).
    // SS: RU/RC/RR/Snapshot allowed, Serializable prevented.
    let out = vec![
        oc(PostgreSQL, ReadUncommitted, Allowed), oc(PostgreSQL, ReadCommitted, Allowed),
        oc(PostgreSQL, RepeatableRead, Allowed), oc(PostgreSQL, Serializable, Prevented),
        oc(MySQL, ReadUncommitted, Allowed), oc(MySQL, ReadCommitted, Allowed),
        oc(MySQL, RepeatableRead, Prevented), oc(MySQL, Serializable, Prevented),
        oc(SqlServer, ReadUncommitted, Allowed), oc(SqlServer, ReadCommitted, Allowed),
        oc(SqlServer, RepeatableRead, Allowed), oc(SqlServer, Snapshot, Allowed),
        oc(SqlServer, Serializable, Prevented),
    ];
    HermitageTestCase::new("g2item-write-skew", "Write Skew (G2-item)", AnomalyClass::G2Item,
        "T1 reads x,y and writes y=f(x). T2 reads x,y and writes x=g(y). Both \
         commit. Anti-dependency cycle constitutes write skew.", steps, out)
}

fn build_g2_phantom() -> HermitageTestCase {
    let steps = vec![
        step(1, StepAction::Begin, "T1 begins"), step(2, StepAction::Begin, "T2 begins"),
        step(1, StepAction::SelectWhere { table: s("test"), condition: s("value > 10") },
             "T1 selects rows WHERE value > 10"),
        step(2, StepAction::Insert { table: s("test"), key: s("z"), value: s("15") },
             "T2 inserts z=15 matching predicate"),
        step(2, StepAction::Commit, "T2 commits"),
        step(1, StepAction::SelectWhere { table: s("test"), condition: s("value > 10") },
             "T1 re-selects WHERE value > 10"),
        step(1, StepAction::Commit, "T1 commits"),
    ];
    let mut out = by_threshold(EngineKind::PostgreSQL, &pg(), 2);
    out.extend(by_threshold(EngineKind::MySQL, &my(), 2));
    out.extend(by_threshold(EngineKind::SqlServer, &ss(), 2));
    HermitageTestCase::new("g2-phantom-read", "Phantom Read (G2)", AnomalyClass::G2,
        "T1 selects rows matching a predicate, T2 inserts a matching row and \
         commits, T1 re-selects. Phantom visibility is a G2 anomaly.", steps, out)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_construction_and_count() {
        let cat = HermitageCatalog::new();
        assert_eq!(cat.test_count(), 7);
        assert_eq!(cat.all_tests().len(), 7);
    }

    #[test]
    fn lookup_by_id() {
        let cat = HermitageCatalog::new();
        let tc = cat.get_test("g0-dirty-write").expect("g0 should exist");
        assert_eq!(tc.anomaly_class, AnomalyClass::G0);
        assert!(tc.name.contains("Dirty Write"));
        assert!(cat.get_test("nonexistent").is_none());
    }

    #[test]
    fn tests_for_anomaly_class() {
        let cat = HermitageCatalog::new();
        let g2item = cat.tests_for_anomaly(AnomalyClass::G2Item);
        assert_eq!(g2item.len(), 2, "lost-update and write-skew");
        let g0 = cat.tests_for_anomaly(AnomalyClass::G0);
        assert_eq!(g0.len(), 1);
        assert_eq!(g0[0].id, "g0-dirty-write");
    }

    #[test]
    fn expected_outcomes_g1a_postgresql() {
        let cat = HermitageCatalog::new();
        let tc = cat.get_test("g1a-aborted-read").unwrap();
        assert!(tc.expected_for(EngineKind::PostgreSQL, IsolationLevel::ReadUncommitted).unwrap().is_allowed());
        assert!(tc.expected_for(EngineKind::PostgreSQL, IsolationLevel::ReadCommitted).unwrap().is_prevented());
        assert!(tc.expected_for(EngineKind::PostgreSQL, IsolationLevel::Serializable).unwrap().is_prevented());
    }

    #[test]
    fn expected_outcomes_g1b_mysql_all_prevented() {
        let cat = HermitageCatalog::new();
        let tc = cat.get_test("g1b-intermediate-read").unwrap();
        for lvl in my() {
            assert!(tc.expected_for(EngineKind::MySQL, lvl).unwrap().is_prevented(),
                "MySQL {lvl:?} should prevent G1b");
        }
    }

    #[test]
    fn write_skew_pg_rr_allowed_serializable_prevented() {
        let cat = HermitageCatalog::new();
        let tc = cat.get_test("g2item-write-skew").unwrap();
        assert!(tc.expected_for(EngineKind::PostgreSQL, IsolationLevel::RepeatableRead).unwrap().is_allowed());
        assert!(tc.expected_for(EngineKind::PostgreSQL, IsolationLevel::Serializable).unwrap().is_prevented());
    }

    #[test]
    fn phantom_sqlserver_outcomes() {
        let cat = HermitageCatalog::new();
        let tc = cat.get_test("g2-phantom-read").unwrap();
        assert!(tc.expected_for(EngineKind::SqlServer, IsolationLevel::ReadCommitted).unwrap().is_allowed());
        assert!(tc.expected_for(EngineKind::SqlServer, IsolationLevel::RepeatableRead).unwrap().is_prevented());
        assert!(tc.expected_for(EngineKind::SqlServer, IsolationLevel::Snapshot).unwrap().is_prevented());
    }

    #[test]
    fn matrix_generation() {
        let cat = HermitageCatalog::new();
        let matrix = cat.expected_matrix();
        assert!(!matrix.is_empty());
        for entry in &matrix {
            assert!(cat.get_test(&entry.test_id).is_some(),
                "unknown test_id: {}", entry.test_id);
        }
    }

    #[test]
    fn prevented_and_allowed_filtering() {
        let cat = HermitageCatalog::new();
        let prevented = cat.tests_prevented_at(EngineKind::PostgreSQL, IsolationLevel::Serializable);
        assert_eq!(prevented.len(), cat.test_count(),
            "PG Serializable should prevent all Hermitage anomalies");
        let allowed_ru = cat.tests_allowed_at(EngineKind::PostgreSQL, IsolationLevel::ReadUncommitted);
        assert!(!allowed_ru.is_empty(), "PG RU should allow some anomalies");
        assert!(!allowed_ru.iter().any(|tc| tc.anomaly_class == AnomalyClass::G0),
            "G0 should not be allowed at PG RU");
    }

    #[test]
    fn step_action_display() {
        assert_eq!(format!("{}", w("x", "42")), "WRITE test.x = 42");
        assert_eq!(format!("{}", StepAction::Begin), "BEGIN");
        assert_eq!(format!("{}", StepAction::Commit), "COMMIT");
        assert_eq!(format!("{}", StepAction::Abort), "ABORT");
    }

    #[test]
    fn expected_behavior_methods() {
        assert!(ExpectedBehavior::Allowed.is_allowed());
        assert!(!ExpectedBehavior::Allowed.is_prevented());
        assert!(ExpectedBehavior::Prevented.is_prevented());
        assert!(!ExpectedBehavior::Prevented.is_allowed());
        let es = ExpectedBehavior::EngineSpecific(s("blocked then proceeds"));
        assert!(!es.is_allowed());
        assert!(!es.is_prevented());
    }

    #[test]
    fn catalog_default_trait() {
        let cat = HermitageCatalog::default();
        assert_eq!(cat.test_count(), 7);
    }

    #[test]
    fn all_tests_have_nonempty_steps_and_outcomes() {
        let cat = HermitageCatalog::new();
        for tc in cat.all_tests() {
            assert!(!tc.steps.is_empty(), "test {} has no steps", tc.id);
            assert!(!tc.expected_outcomes.is_empty(), "test {} has no outcomes", tc.id);
            assert!(!tc.description.is_empty(), "test {} has no description", tc.id);
        }
    }

    #[test]
    fn g0_prevented_everywhere() {
        let cat = HermitageCatalog::new();
        let tc = cat.get_test("g0-dirty-write").unwrap();
        for o in &tc.expected_outcomes {
            assert!(o.behavior.is_prevented(), "G0 should be prevented on {:?} {:?}", o.engine, o.level);
        }
    }
}
