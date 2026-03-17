//! Configuration types.
use serde::{Deserialize, Serialize};
use crate::isolation::IsolationLevel;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    pub max_transactions: usize,
    pub max_operations_per_txn: usize,
    pub bound_k: usize,
    pub smt_timeout_seconds: u64,
    pub enable_predicate_analysis: bool,
    pub enable_witness_synthesis: bool,
    pub enable_minimization: bool,
    pub target_anomalies: Vec<crate::isolation::AnomalyClass>,
    pub engine_config: EngineConfig,
    pub output_format: OutputFormat,
    pub verbosity: Verbosity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub postgresql: PostgresConfig,
    pub mysql: MySqlConfig,
    pub sqlserver: SqlServerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostgresConfig {
    pub version: String,
    pub ssi_enabled: bool,
    pub siread_lock_memory_mb: usize,
    pub read_only_optimization: bool,
    pub granularity_escalation_threshold: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MySqlConfig {
    pub version: String,
    pub innodb_gap_lock_mode: GapLockMode,
    pub index_hint: Option<String>,
    pub over_approximate_indexes: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SqlServerConfig {
    pub version: String,
    pub rcsi_enabled: bool,
    pub lock_escalation_threshold: usize,
    pub use_pessimistic_mode: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GapLockMode {
    NextKey,
    Gap,
    RecordOnly,
    InsertIntention,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    Json,
    Text,
    Sql,
    Dot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Verbosity {
    Quiet,
    Normal,
    Verbose,
    Debug,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            max_transactions: 5,
            max_operations_per_txn: 10,
            bound_k: 3,
            smt_timeout_seconds: 120,
            enable_predicate_analysis: true,
            enable_witness_synthesis: true,
            enable_minimization: true,
            target_anomalies: crate::isolation::AnomalyClass::all(),
            engine_config: EngineConfig::default(),
            output_format: OutputFormat::Json,
            verbosity: Verbosity::Normal,
        }
    }
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            postgresql: PostgresConfig::default(),
            mysql: MySqlConfig::default(),
            sqlserver: SqlServerConfig::default(),
        }
    }
}

impl Default for PostgresConfig {
    fn default() -> Self {
        Self {
            version: "16.0".into(),
            ssi_enabled: true,
            siread_lock_memory_mb: 256,
            read_only_optimization: true,
            granularity_escalation_threshold: 10000,
        }
    }
}

impl Default for MySqlConfig {
    fn default() -> Self {
        Self {
            version: "8.0".into(),
            innodb_gap_lock_mode: GapLockMode::NextKey,
            index_hint: None,
            over_approximate_indexes: true,
        }
    }
}

impl Default for SqlServerConfig {
    fn default() -> Self {
        Self {
            version: "2022".into(),
            rcsi_enabled: false,
            lock_escalation_threshold: 5000,
            use_pessimistic_mode: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortabilityConfig {
    pub source_engine: EngineKind,
    pub source_isolation: IsolationLevel,
    pub target_engine: EngineKind,
    pub target_isolation: IsolationLevel,
    pub generate_witnesses: bool,
    pub max_witness_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EngineKind {
    PostgreSQL,
    MySQL,
    SqlServer,
}

impl std::fmt::Display for EngineKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PostgreSQL => write!(f, "PostgreSQL"),
            Self::MySQL => write!(f, "MySQL"),
            Self::SqlServer => write!(f, "SQL Server"),
        }
    }
}

impl EngineKind {
    pub fn all() -> Vec<Self> {
        vec![Self::PostgreSQL, Self::MySQL, Self::SqlServer]
    }
    pub fn default_isolation_levels(self) -> Vec<IsolationLevel> {
        match self {
            Self::PostgreSQL => vec![
                IsolationLevel::ReadUncommitted, IsolationLevel::ReadCommitted,
                IsolationLevel::RepeatableRead, IsolationLevel::Serializable,
            ],
            Self::MySQL => vec![
                IsolationLevel::ReadUncommitted, IsolationLevel::ReadCommitted,
                IsolationLevel::RepeatableRead, IsolationLevel::Serializable,
            ],
            Self::SqlServer => vec![
                IsolationLevel::ReadUncommitted, IsolationLevel::ReadCommitted,
                IsolationLevel::RepeatableRead, IsolationLevel::Serializable,
                IsolationLevel::Snapshot,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_default_config() {
        let config = AnalysisConfig::default();
        assert_eq!(config.bound_k, 3);
        assert_eq!(config.max_transactions, 5);
        assert!(config.enable_predicate_analysis);
    }
    #[test]
    fn test_engine_kind_display() {
        assert_eq!(format!("{}", EngineKind::PostgreSQL), "PostgreSQL");
    }
    #[test]
    fn test_engine_isolation_levels() {
        let pg = EngineKind::PostgreSQL.default_isolation_levels();
        assert_eq!(pg.len(), 4);
        let ss = EngineKind::SqlServer.default_isolation_levels();
        assert_eq!(ss.len(), 5);
    }
}
