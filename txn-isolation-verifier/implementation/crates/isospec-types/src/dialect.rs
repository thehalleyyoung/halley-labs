//! SQL dialect definitions.
use serde::{Deserialize, Serialize};
use crate::config::EngineKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SqlDialect {
    PostgreSQL,
    MySQL,
    TransactSql,
    Ansi,
}

impl SqlDialect {
    pub fn from_engine(engine: EngineKind) -> Self {
        match engine {
            EngineKind::PostgreSQL => Self::PostgreSQL,
            EngineKind::MySQL => Self::MySQL,
            EngineKind::SqlServer => Self::TransactSql,
        }
    }
    pub fn begin_statement(self) -> &'static str {
        match self {
            Self::PostgreSQL => "BEGIN",
            Self::MySQL => "START TRANSACTION",
            Self::TransactSql => "BEGIN TRANSACTION",
            Self::Ansi => "START TRANSACTION",
        }
    }
    pub fn set_isolation(self, level: crate::isolation::IsolationLevel) -> String {
        match self {
            Self::TransactSql => format!("SET TRANSACTION ISOLATION LEVEL {}", level),
            _ => format!("SET TRANSACTION ISOLATION LEVEL {}", level),
        }
    }
    pub fn for_update_clause(self) -> &'static str {
        match self {
            Self::PostgreSQL => "FOR UPDATE",
            Self::MySQL => "FOR UPDATE",
            Self::TransactSql => "WITH (UPDLOCK, HOLDLOCK)",
            Self::Ansi => "FOR UPDATE",
        }
    }
    pub fn for_share_clause(self) -> &'static str {
        match self {
            Self::PostgreSQL => "FOR SHARE",
            Self::MySQL => "FOR SHARE",
            Self::TransactSql => "WITH (HOLDLOCK)",
            Self::Ansi => "FOR SHARE",
        }
    }
    pub fn advisory_lock(self, lock_id: i64) -> String {
        match self {
            Self::PostgreSQL => format!("SELECT pg_advisory_lock({})", lock_id),
            Self::MySQL => format!("SELECT GET_LOCK('lock_{}', 10)", lock_id),
            Self::TransactSql => format!("EXEC sp_getapplock @Resource='lock_{}', @LockMode='Exclusive'", lock_id),
            Self::Ansi => format!("-- advisory lock {} (not supported)", lock_id),
        }
    }
    pub fn advisory_unlock(self, lock_id: i64) -> String {
        match self {
            Self::PostgreSQL => format!("SELECT pg_advisory_unlock({})", lock_id),
            Self::MySQL => format!("SELECT RELEASE_LOCK('lock_{}')", lock_id),
            Self::TransactSql => format!("EXEC sp_releaseapplock @Resource='lock_{}'", lock_id),
            Self::Ansi => format!("-- advisory unlock {} (not supported)", lock_id),
        }
    }
    pub fn string_concat(self) -> &'static str {
        match self {
            Self::PostgreSQL | Self::Ansi => "||",
            Self::MySQL => "CONCAT",
            Self::TransactSql => "+",
        }
    }
}

impl std::fmt::Display for SqlDialect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PostgreSQL => write!(f, "PostgreSQL"),
            Self::MySQL => write!(f, "MySQL"),
            Self::TransactSql => write!(f, "T-SQL"),
            Self::Ansi => write!(f, "ANSI SQL"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_dialect_from_engine() {
        assert_eq!(SqlDialect::from_engine(EngineKind::PostgreSQL), SqlDialect::PostgreSQL);
        assert_eq!(SqlDialect::from_engine(EngineKind::MySQL), SqlDialect::MySQL);
    }
    #[test]
    fn test_begin_statement() {
        assert_eq!(SqlDialect::PostgreSQL.begin_statement(), "BEGIN");
        assert_eq!(SqlDialect::MySQL.begin_statement(), "START TRANSACTION");
    }
    #[test]
    fn test_advisory_lock() {
        let pg = SqlDialect::PostgreSQL.advisory_lock(42);
        assert!(pg.contains("pg_advisory_lock"));
    }
}
