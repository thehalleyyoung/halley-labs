//! Isolation level definitions and relationships.
use serde::{Deserialize, Serialize};
use std::fmt;

/// Standard SQL isolation levels plus engine-specific variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IsolationLevel {
    ReadUncommitted,
    ReadCommitted,
    RepeatableRead,
    Serializable,
    /// SQL Server specific: snapshot isolation (RCSI mode)
    Snapshot,
    /// PostgreSQL: Read Committed with statement-level snapshots
    PgReadCommitted,
    /// MySQL InnoDB: Repeatable Read with gap locking
    MySqlRepeatableRead,
    /// SQL Server: Read Committed Snapshot Isolation
    SqlServerRCSI,
}

impl IsolationLevel {
    /// Returns the standard SQL level this maps to (if engine-specific).
    pub fn standard_level(self) -> IsolationLevel {
        match self {
            Self::PgReadCommitted => Self::ReadCommitted,
            Self::MySqlRepeatableRead => Self::RepeatableRead,
            Self::SqlServerRCSI => Self::ReadCommitted,
            other => other,
        }
    }

    /// Returns true if this level provides at least the guarantees of `other`.
    pub fn at_least(self, other: IsolationLevel) -> bool {
        self.strength() >= other.strength()
    }

    /// Returns a numeric strength ranking for total-order comparison.
    /// Note: Snapshot is not directly comparable with RepeatableRead in SQL Server.
    pub fn strength(self) -> u8 {
        match self.standard_level() {
            Self::ReadUncommitted => 0,
            Self::ReadCommitted => 1,
            Self::Snapshot | Self::SqlServerRCSI => 2,
            Self::RepeatableRead | Self::MySqlRepeatableRead => 3,
            Self::Serializable => 4,
            _ => 1,
        }
    }

    /// Whether snapshot isolation is comparable with repeatable read.
    /// In SQL Server, SNAPSHOT and RR form a DAG (incomparable).
    pub fn is_comparable_with(self, other: IsolationLevel) -> bool {
        let a = self.standard_level();
        let b = other.standard_level();
        if (a == Self::Snapshot && b == Self::RepeatableRead)
            || (a == Self::RepeatableRead && b == Self::Snapshot)
        {
            return false;
        }
        true
    }

    /// Returns all standard isolation levels.
    pub fn all_standard() -> Vec<IsolationLevel> {
        vec![
            Self::ReadUncommitted,
            Self::ReadCommitted,
            Self::RepeatableRead,
            Self::Serializable,
        ]
    }

    /// Returns all levels including engine-specific ones.
    pub fn all_levels() -> Vec<IsolationLevel> {
        vec![
            Self::ReadUncommitted,
            Self::ReadCommitted,
            Self::RepeatableRead,
            Self::Serializable,
            Self::Snapshot,
            Self::PgReadCommitted,
            Self::MySqlRepeatableRead,
            Self::SqlServerRCSI,
        ]
    }

    /// Returns the anomalies prevented at this level per Adya's formalization.
    pub fn prevented_anomalies(self) -> Vec<AnomalyClass> {
        match self.standard_level() {
            Self::ReadUncommitted => vec![],
            Self::ReadCommitted => vec![AnomalyClass::G0, AnomalyClass::G1a, AnomalyClass::G1b, AnomalyClass::G1c],
            Self::RepeatableRead | Self::MySqlRepeatableRead => vec![
                AnomalyClass::G0, AnomalyClass::G1a, AnomalyClass::G1b,
                AnomalyClass::G1c, AnomalyClass::G2Item,
            ],
            Self::Snapshot | Self::SqlServerRCSI => vec![
                AnomalyClass::G0, AnomalyClass::G1a, AnomalyClass::G1b,
                AnomalyClass::G1c, AnomalyClass::G2Item,
            ],
            Self::Serializable => AnomalyClass::all(),
            _ => vec![],
        }
    }

    /// Returns the anomalies that may occur at this level.
    pub fn possible_anomalies(self) -> Vec<AnomalyClass> {
        let prevented = self.prevented_anomalies();
        AnomalyClass::all()
            .into_iter()
            .filter(|a| !prevented.contains(a))
            .collect()
    }

    /// Returns the cost of this isolation level (higher = more expensive).
    pub fn cost(self) -> u32 {
        match self.standard_level() {
            Self::ReadUncommitted => 1,
            Self::ReadCommitted => 2,
            Self::Snapshot | Self::SqlServerRCSI => 4,
            Self::RepeatableRead | Self::MySqlRepeatableRead => 6,
            Self::Serializable => 10,
            _ => 2,
        }
    }

    pub fn from_str_loose(s: &str) -> Option<Self> {
        let lower = s.to_lowercase().replace(['-', '_', ' '], "");
        match lower.as_str() {
            "readuncommitted" | "ru" => Some(Self::ReadUncommitted),
            "readcommitted" | "rc" => Some(Self::ReadCommitted),
            "repeatableread" | "rr" => Some(Self::RepeatableRead),
            "serializable" | "ser" => Some(Self::Serializable),
            "snapshot" | "si" => Some(Self::Snapshot),
            "pgreadcommitted" | "pgrc" => Some(Self::PgReadCommitted),
            "mysqlrepeatableread" | "mysqlrr" => Some(Self::MySqlRepeatableRead),
            "rcsi" | "sqlserverrcsi" => Some(Self::SqlServerRCSI),
            _ => None,
        }
    }
}

impl fmt::Display for IsolationLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReadUncommitted => write!(f, "READ UNCOMMITTED"),
            Self::ReadCommitted => write!(f, "READ COMMITTED"),
            Self::RepeatableRead => write!(f, "REPEATABLE READ"),
            Self::Serializable => write!(f, "SERIALIZABLE"),
            Self::Snapshot => write!(f, "SNAPSHOT"),
            Self::PgReadCommitted => write!(f, "PG READ COMMITTED"),
            Self::MySqlRepeatableRead => write!(f, "MYSQL REPEATABLE READ"),
            Self::SqlServerRCSI => write!(f, "RCSI"),
        }
    }
}

/// Adya anomaly classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AnomalyClass {
    /// Dirty write (write-write cycle on uncommitted data)
    G0,
    /// Aborted reads
    G1a,
    /// Intermediate reads
    G1b,
    /// Circular information flow
    G1c,
    /// Item anti-dependency cycles
    G2Item,
    /// Predicate anti-dependency cycles (phantom)
    G2,
}

impl AnomalyClass {
    pub fn all() -> Vec<Self> {
        vec![Self::G0, Self::G1a, Self::G1b, Self::G1c, Self::G2Item, Self::G2]
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::G0 => "G0 (Dirty Write)",
            Self::G1a => "G1a (Aborted Read)",
            Self::G1b => "G1b (Intermediate Read)",
            Self::G1c => "G1c (Circular Information Flow)",
            Self::G2Item => "G2-item (Item Anti-Dependency)",
            Self::G2 => "G2 (Predicate Anti-Dependency)",
        }
    }

    pub fn severity(self) -> AnomalySeverity {
        match self {
            Self::G0 => AnomalySeverity::Critical,
            Self::G1a | Self::G1b => AnomalySeverity::High,
            Self::G1c => AnomalySeverity::Medium,
            Self::G2Item => AnomalySeverity::Medium,
            Self::G2 => AnomalySeverity::Low,
        }
    }

    pub fn min_transactions(self) -> usize {
        match self {
            Self::G0 | Self::G1a | Self::G1b | Self::G1c | Self::G2Item => 2,
            Self::G2 => 3,
        }
    }
}

impl fmt::Display for AnomalyClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Severity levels for anomalies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// An isolation level DAG for engine-specific ordering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsolationDag {
    pub edges: Vec<(IsolationLevel, IsolationLevel)>,
}

impl IsolationDag {
    pub fn standard() -> Self {
        Self {
            edges: vec![
                (IsolationLevel::ReadUncommitted, IsolationLevel::ReadCommitted),
                (IsolationLevel::ReadCommitted, IsolationLevel::RepeatableRead),
                (IsolationLevel::RepeatableRead, IsolationLevel::Serializable),
            ],
        }
    }

    pub fn sql_server() -> Self {
        Self {
            edges: vec![
                (IsolationLevel::ReadUncommitted, IsolationLevel::ReadCommitted),
                (IsolationLevel::ReadCommitted, IsolationLevel::RepeatableRead),
                (IsolationLevel::ReadCommitted, IsolationLevel::Snapshot),
                (IsolationLevel::RepeatableRead, IsolationLevel::Serializable),
                (IsolationLevel::Snapshot, IsolationLevel::Serializable),
            ],
        }
    }

    pub fn is_stronger(&self, a: IsolationLevel, b: IsolationLevel) -> bool {
        if a == b {
            return false;
        }
        let mut visited = std::collections::HashSet::new();
        let mut stack = vec![b];
        while let Some(current) = stack.pop() {
            if current == a {
                return true;
            }
            if visited.insert(current) {
                for (from, to) in &self.edges {
                    if *to == current {
                        stack.push(*from);
                    }
                }
            }
        }
        false
    }

    pub fn comparable(&self, a: IsolationLevel, b: IsolationLevel) -> bool {
        a == b || self.is_stronger(a, b) || self.is_stronger(b, a)
    }

    pub fn levels_between(&self, low: IsolationLevel, high: IsolationLevel) -> Vec<IsolationLevel> {
        let mut result = Vec::new();
        for level in IsolationLevel::all_standard() {
            if (level == low || self.is_stronger(level, low))
                && (level == high || self.is_stronger(high, level))
            {
                result.push(level);
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isolation_level_strength() {
        assert!(IsolationLevel::Serializable.at_least(IsolationLevel::ReadCommitted));
        assert!(!IsolationLevel::ReadCommitted.at_least(IsolationLevel::Serializable));
    }

    #[test]
    fn test_prevented_anomalies() {
        let ser = IsolationLevel::Serializable.prevented_anomalies();
        assert_eq!(ser.len(), AnomalyClass::all().len());

        let ru = IsolationLevel::ReadUncommitted.prevented_anomalies();
        assert!(ru.is_empty());
    }

    #[test]
    fn test_anomaly_class_severity() {
        assert!(AnomalyClass::G0.severity() > AnomalyClass::G2.severity());
    }

    #[test]
    fn test_isolation_dag() {
        let dag = IsolationDag::standard();
        assert!(dag.is_stronger(IsolationLevel::Serializable, IsolationLevel::ReadCommitted));
        assert!(!dag.is_stronger(IsolationLevel::ReadCommitted, IsolationLevel::Serializable));
    }

    #[test]
    fn test_sql_server_dag_incomparable() {
        let dag = IsolationDag::sql_server();
        assert!(!dag.is_stronger(IsolationLevel::Snapshot, IsolationLevel::RepeatableRead));
        assert!(!dag.is_stronger(IsolationLevel::RepeatableRead, IsolationLevel::Snapshot));
    }

    #[test]
    fn test_from_str_loose() {
        assert_eq!(IsolationLevel::from_str_loose("serializable"), Some(IsolationLevel::Serializable));
        assert_eq!(IsolationLevel::from_str_loose("read-committed"), Some(IsolationLevel::ReadCommitted));
        assert_eq!(IsolationLevel::from_str_loose("RC"), Some(IsolationLevel::ReadCommitted));
        assert_eq!(IsolationLevel::from_str_loose("nonsense"), None);
    }

    #[test]
    fn test_isolation_cost_ordering() {
        assert!(IsolationLevel::Serializable.cost() > IsolationLevel::ReadCommitted.cost());
        assert!(IsolationLevel::RepeatableRead.cost() > IsolationLevel::Snapshot.cost());
    }

    #[test]
    fn test_min_transactions() {
        assert_eq!(AnomalyClass::G0.min_transactions(), 2);
        assert_eq!(AnomalyClass::G2.min_transactions(), 3);
    }
}
