//! Dialect-specific SQL generation for witness scripts.
//!
//! Generates CREATE TABLE, transaction scripts, and DML statements
//! for PostgreSQL, MySQL, and SQL Server.

use std::fmt;

use isospec_types::error::{IsoSpecError, IsoSpecResult};
use isospec_types::identifier::ItemId;
use isospec_types::isolation::IsolationLevel;
use isospec_types::operation::OpKind;
use isospec_types::schedule::{Schedule, ScheduleStep};
use isospec_types::column::{ColumnDef, DataType};
use isospec_types::schema::TableSchema;
use isospec_types::value::Value;

// ---------------------------------------------------------------------------
// SqlDialect enum
// ---------------------------------------------------------------------------

/// Supported SQL dialects for code generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetDialect {
    PostgreSql,
    MySql,
    SqlServer,
}

impl fmt::Display for TargetDialect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TargetDialect::PostgreSql => write!(f, "PostgreSQL"),
            TargetDialect::MySql => write!(f, "MySQL"),
            TargetDialect::SqlServer => write!(f, "SQL Server"),
        }
    }
}

// ---------------------------------------------------------------------------
// SqlGenerator trait
// ---------------------------------------------------------------------------

/// Trait for generating dialect-specific SQL.
pub trait SqlGenerator: Send + Sync {
    fn dialect(&self) -> TargetDialect;

    /// Generate a CREATE TABLE statement.
    fn create_table(&self, table: &str, columns: &[ColumnSpec]) -> String;

    /// Generate a DROP TABLE statement.
    fn drop_table(&self, table: &str) -> String;

    /// Generate a BEGIN TRANSACTION statement.
    fn begin_transaction(&self, isolation: &IsolationLevel) -> Vec<String>;

    /// Generate a COMMIT statement.
    fn commit(&self) -> String;

    /// Generate a ROLLBACK statement.
    fn rollback(&self) -> String;

    /// Generate a SELECT statement.
    fn select(&self, table: &str, key_col: &str, val_col: &str, key: i64) -> String;

    /// Generate a SELECT ... FOR UPDATE statement.
    fn select_for_update(&self, table: &str, key_col: &str, val_col: &str, key: i64) -> String;

    /// Generate an UPDATE statement.
    fn update(&self, table: &str, key_col: &str, val_col: &str, key: i64, value: &str) -> String;

    /// Generate an INSERT statement.
    fn insert(&self, table: &str, key_col: &str, val_col: &str, key: i64, value: &str) -> String;

    /// Generate a DELETE statement.
    fn delete(&self, table: &str, key_col: &str, key: i64) -> String;

    /// Generate a predicate-based SELECT.
    fn predicate_select(&self, table: &str, val_col: &str, predicate: &str) -> String;

    /// Generate a predicate-based UPDATE.
    fn predicate_update(
        &self,
        table: &str,
        val_col: &str,
        set_expr: &str,
        predicate: &str,
    ) -> String;

    /// Generate an advisory lock acquisition.
    fn advisory_lock(&self, lock_id: i64) -> String;

    /// Generate an advisory lock release.
    fn advisory_unlock(&self, lock_id: i64) -> String;

    /// Generate a sleep/delay statement (in seconds).
    fn sleep(&self, seconds: f64) -> String;

    /// Map a DataType to this dialect's type name.
    fn type_name(&self, dt: &DataType) -> String;
}

/// Column specification for CREATE TABLE.
#[derive(Debug, Clone)]
pub struct ColumnSpec {
    pub name: String,
    pub data_type: DataType,
    pub primary_key: bool,
    pub not_null: bool,
    pub default_value: Option<String>,
}

impl ColumnSpec {
    pub fn new(name: &str, data_type: DataType) -> Self {
        Self {
            name: name.to_string(),
            data_type,
            primary_key: false,
            not_null: false,
            default_value: None,
        }
    }

    pub fn primary_key(mut self) -> Self {
        self.primary_key = true;
        self.not_null = true;
        self
    }

    pub fn not_null(mut self) -> Self {
        self.not_null = true;
        self
    }

    pub fn with_default(mut self, val: &str) -> Self {
        self.default_value = Some(val.to_string());
        self
    }
}

// ---------------------------------------------------------------------------
// PostgreSqlGenerator
// ---------------------------------------------------------------------------

pub struct PostgreSqlGenerator;

impl SqlGenerator for PostgreSqlGenerator {
    fn dialect(&self) -> TargetDialect {
        TargetDialect::PostgreSql
    }

    fn create_table(&self, table: &str, columns: &[ColumnSpec]) -> String {
        let col_defs: Vec<String> = columns
            .iter()
            .map(|c| {
                let mut parts = vec![
                    c.name.clone(),
                    self.type_name(&c.data_type),
                ];
                if c.primary_key {
                    parts.push("PRIMARY KEY".to_string());
                }
                if c.not_null && !c.primary_key {
                    parts.push("NOT NULL".to_string());
                }
                if let Some(ref def) = c.default_value {
                    parts.push(format!("DEFAULT {}", def));
                }
                parts.join(" ")
            })
            .collect();
        format!("CREATE TABLE {} ({});", table, col_defs.join(", "))
    }

    fn drop_table(&self, table: &str) -> String {
        format!("DROP TABLE IF EXISTS {} CASCADE;", table)
    }

    fn begin_transaction(&self, isolation: &IsolationLevel) -> Vec<String> {
        let iso_str = pg_isolation_str(isolation);
        vec![
            format!("BEGIN TRANSACTION ISOLATION LEVEL {};", iso_str),
        ]
    }

    fn commit(&self) -> String {
        "COMMIT;".to_string()
    }

    fn rollback(&self) -> String {
        "ROLLBACK;".to_string()
    }

    fn select(&self, table: &str, key_col: &str, val_col: &str, key: i64) -> String {
        format!(
            "SELECT {} FROM {} WHERE {} = {};",
            val_col, table, key_col, key
        )
    }

    fn select_for_update(&self, table: &str, key_col: &str, val_col: &str, key: i64) -> String {
        format!(
            "SELECT {} FROM {} WHERE {} = {} FOR UPDATE;",
            val_col, table, key_col, key
        )
    }

    fn update(&self, table: &str, key_col: &str, val_col: &str, key: i64, value: &str) -> String {
        format!(
            "UPDATE {} SET {} = {} WHERE {} = {};",
            table, val_col, value, key_col, key
        )
    }

    fn insert(&self, table: &str, key_col: &str, val_col: &str, key: i64, value: &str) -> String {
        format!(
            "INSERT INTO {} ({}, {}) VALUES ({}, {});",
            table, key_col, val_col, key, value
        )
    }

    fn delete(&self, table: &str, key_col: &str, key: i64) -> String {
        format!("DELETE FROM {} WHERE {} = {};", table, key_col, key)
    }

    fn predicate_select(&self, table: &str, val_col: &str, predicate: &str) -> String {
        format!("SELECT {} FROM {} WHERE {};", val_col, table, predicate)
    }

    fn predicate_update(
        &self,
        table: &str,
        val_col: &str,
        set_expr: &str,
        predicate: &str,
    ) -> String {
        format!(
            "UPDATE {} SET {} = {} WHERE {};",
            table, val_col, set_expr, predicate
        )
    }

    fn advisory_lock(&self, lock_id: i64) -> String {
        format!("SELECT pg_advisory_lock({});", lock_id)
    }

    fn advisory_unlock(&self, lock_id: i64) -> String {
        format!("SELECT pg_advisory_unlock({});", lock_id)
    }

    fn sleep(&self, seconds: f64) -> String {
        format!("SELECT pg_sleep({});", seconds)
    }

    fn type_name(&self, dt: &DataType) -> String {
        match dt {
            DataType::Boolean => "BOOLEAN".to_string(),
            DataType::Integer => "INTEGER".to_string(),
            DataType::BigInt => "BIGINT".to_string(),
            DataType::SmallInt => "SMALLINT".to_string(),
            DataType::Float => "DOUBLE PRECISION".to_string(),
            DataType::Decimal { precision, scale } => format!("DECIMAL({}, {})", precision, scale),
            DataType::Varchar(len) => format!("VARCHAR({})", len),
            DataType::Char(len) => format!("CHAR({})", len),
            DataType::Text => "TEXT".to_string(),
            DataType::Timestamp => "TIMESTAMP".to_string(),
            DataType::Date => "DATE".to_string(),
            DataType::Time => "TIME".to_string(),
            DataType::Blob => "BYTEA".to_string(),
            DataType::Uuid => "UUID".to_string(),
            _ => "TEXT".to_string(),
        }
    }
}

fn pg_isolation_str(level: &IsolationLevel) -> &'static str {
    match level {
        IsolationLevel::ReadUncommitted => "READ UNCOMMITTED",
        IsolationLevel::ReadCommitted => "READ COMMITTED",
        IsolationLevel::RepeatableRead => "REPEATABLE READ",
        IsolationLevel::Serializable => "SERIALIZABLE",
        _ => "SERIALIZABLE",
    }
}

// ---------------------------------------------------------------------------
// MySqlGenerator
// ---------------------------------------------------------------------------

pub struct MySqlGenerator;

impl SqlGenerator for MySqlGenerator {
    fn dialect(&self) -> TargetDialect {
        TargetDialect::MySql
    }

    fn create_table(&self, table: &str, columns: &[ColumnSpec]) -> String {
        let col_defs: Vec<String> = columns
            .iter()
            .map(|c| {
                let mut parts = vec![
                    format!("`{}`", c.name),
                    self.type_name(&c.data_type),
                ];
                if c.primary_key {
                    parts.push("PRIMARY KEY".to_string());
                }
                if c.not_null && !c.primary_key {
                    parts.push("NOT NULL".to_string());
                }
                if let Some(ref def) = c.default_value {
                    parts.push(format!("DEFAULT {}", def));
                }
                parts.join(" ")
            })
            .collect();
        format!(
            "CREATE TABLE `{}` ({}) ENGINE=InnoDB;",
            table,
            col_defs.join(", ")
        )
    }

    fn drop_table(&self, table: &str) -> String {
        format!("DROP TABLE IF EXISTS `{}`;", table)
    }

    fn begin_transaction(&self, isolation: &IsolationLevel) -> Vec<String> {
        let iso_str = mysql_isolation_str(isolation);
        vec![
            format!(
                "SET TRANSACTION ISOLATION LEVEL {};",
                iso_str
            ),
            "START TRANSACTION;".to_string(),
        ]
    }

    fn commit(&self) -> String {
        "COMMIT;".to_string()
    }

    fn rollback(&self) -> String {
        "ROLLBACK;".to_string()
    }

    fn select(&self, table: &str, key_col: &str, val_col: &str, key: i64) -> String {
        format!(
            "SELECT `{}` FROM `{}` WHERE `{}` = {};",
            val_col, table, key_col, key
        )
    }

    fn select_for_update(&self, table: &str, key_col: &str, val_col: &str, key: i64) -> String {
        format!(
            "SELECT `{}` FROM `{}` WHERE `{}` = {} FOR UPDATE;",
            val_col, table, key_col, key
        )
    }

    fn update(&self, table: &str, key_col: &str, val_col: &str, key: i64, value: &str) -> String {
        format!(
            "UPDATE `{}` SET `{}` = {} WHERE `{}` = {};",
            table, val_col, value, key_col, key
        )
    }

    fn insert(&self, table: &str, key_col: &str, val_col: &str, key: i64, value: &str) -> String {
        format!(
            "INSERT INTO `{}` (`{}`, `{}`) VALUES ({}, {});",
            table, key_col, val_col, key, value
        )
    }

    fn delete(&self, table: &str, key_col: &str, key: i64) -> String {
        format!("DELETE FROM `{}` WHERE `{}` = {};", table, key_col, key)
    }

    fn predicate_select(&self, table: &str, val_col: &str, predicate: &str) -> String {
        format!(
            "SELECT `{}` FROM `{}` WHERE {};",
            val_col, table, predicate
        )
    }

    fn predicate_update(
        &self,
        table: &str,
        val_col: &str,
        set_expr: &str,
        predicate: &str,
    ) -> String {
        format!(
            "UPDATE `{}` SET `{}` = {} WHERE {};",
            table, val_col, set_expr, predicate
        )
    }

    fn advisory_lock(&self, lock_id: i64) -> String {
        format!("SELECT GET_LOCK('isospec_{}', 30);", lock_id)
    }

    fn advisory_unlock(&self, lock_id: i64) -> String {
        format!("SELECT RELEASE_LOCK('isospec_{}');", lock_id)
    }

    fn sleep(&self, seconds: f64) -> String {
        format!("SELECT SLEEP({});", seconds)
    }

    fn type_name(&self, dt: &DataType) -> String {
        match dt {
            DataType::Boolean => "TINYINT(1)".to_string(),
            DataType::Integer => "INT".to_string(),
            DataType::BigInt => "BIGINT".to_string(),
            DataType::SmallInt => "SMALLINT".to_string(),
            DataType::Float => "DOUBLE".to_string(),
            DataType::Decimal { precision, scale } => format!("DECIMAL({}, {})", precision, scale),
            DataType::Varchar(len) => format!("VARCHAR({})", len),
            DataType::Char(len) => format!("CHAR({})", len),
            DataType::Text => "TEXT".to_string(),
            DataType::Timestamp => "DATETIME(6)".to_string(),
            DataType::Date => "DATE".to_string(),
            DataType::Time => "TIME".to_string(),
            DataType::Blob => "LONGBLOB".to_string(),
            DataType::Uuid => "CHAR(36)".to_string(),
            _ => "TEXT".to_string(),
        }
    }
}

fn mysql_isolation_str(level: &IsolationLevel) -> &'static str {
    match level {
        IsolationLevel::ReadUncommitted => "READ UNCOMMITTED",
        IsolationLevel::ReadCommitted => "READ COMMITTED",
        IsolationLevel::RepeatableRead => "REPEATABLE READ",
        IsolationLevel::Serializable => "SERIALIZABLE",
        _ => "SERIALIZABLE",
    }
}

// ---------------------------------------------------------------------------
// SqlServerGenerator
// ---------------------------------------------------------------------------

pub struct SqlServerGenerator;

impl SqlGenerator for SqlServerGenerator {
    fn dialect(&self) -> TargetDialect {
        TargetDialect::SqlServer
    }

    fn create_table(&self, table: &str, columns: &[ColumnSpec]) -> String {
        let col_defs: Vec<String> = columns
            .iter()
            .map(|c| {
                let mut parts = vec![
                    format!("[{}]", c.name),
                    self.type_name(&c.data_type),
                ];
                if c.primary_key {
                    parts.push("PRIMARY KEY".to_string());
                }
                if c.not_null && !c.primary_key {
                    parts.push("NOT NULL".to_string());
                }
                if let Some(ref def) = c.default_value {
                    parts.push(format!("DEFAULT {}", def));
                }
                parts.join(" ")
            })
            .collect();
        format!("CREATE TABLE [{}] ({});", table, col_defs.join(", "))
    }

    fn drop_table(&self, table: &str) -> String {
        format!(
            "IF OBJECT_ID('{}', 'U') IS NOT NULL DROP TABLE [{}];",
            table, table
        )
    }

    fn begin_transaction(&self, isolation: &IsolationLevel) -> Vec<String> {
        let iso_str = tsql_isolation_str(isolation);
        vec![
            format!("SET TRANSACTION ISOLATION LEVEL {};", iso_str),
            "BEGIN TRANSACTION;".to_string(),
        ]
    }

    fn commit(&self) -> String {
        "COMMIT TRANSACTION;".to_string()
    }

    fn rollback(&self) -> String {
        "ROLLBACK TRANSACTION;".to_string()
    }

    fn select(&self, table: &str, key_col: &str, val_col: &str, key: i64) -> String {
        format!(
            "SELECT [{}] FROM [{}] WHERE [{}] = {};",
            val_col, table, key_col, key
        )
    }

    fn select_for_update(&self, table: &str, key_col: &str, val_col: &str, key: i64) -> String {
        format!(
            "SELECT [{}] FROM [{}] WITH (UPDLOCK, ROWLOCK) WHERE [{}] = {};",
            val_col, table, key_col, key
        )
    }

    fn update(&self, table: &str, key_col: &str, val_col: &str, key: i64, value: &str) -> String {
        format!(
            "UPDATE [{}] SET [{}] = {} WHERE [{}] = {};",
            table, val_col, value, key_col, key
        )
    }

    fn insert(&self, table: &str, key_col: &str, val_col: &str, key: i64, value: &str) -> String {
        format!(
            "INSERT INTO [{}] ([{}], [{}]) VALUES ({}, {});",
            table, key_col, val_col, key, value
        )
    }

    fn delete(&self, table: &str, key_col: &str, key: i64) -> String {
        format!("DELETE FROM [{}] WHERE [{}] = {};", table, key_col, key)
    }

    fn predicate_select(&self, table: &str, val_col: &str, predicate: &str) -> String {
        format!(
            "SELECT [{}] FROM [{}] WHERE {};",
            val_col, table, predicate
        )
    }

    fn predicate_update(
        &self,
        table: &str,
        val_col: &str,
        set_expr: &str,
        predicate: &str,
    ) -> String {
        format!(
            "UPDATE [{}] SET [{}] = {} WHERE {};",
            table, val_col, set_expr, predicate
        )
    }

    fn advisory_lock(&self, lock_id: i64) -> String {
        format!(
            "EXEC sp_getapplock @Resource = 'isospec_{}', @LockMode = 'Exclusive', @LockTimeout = 30000;",
            lock_id
        )
    }

    fn advisory_unlock(&self, lock_id: i64) -> String {
        format!(
            "EXEC sp_releaseapplock @Resource = 'isospec_{}';",
            lock_id
        )
    }

    fn sleep(&self, seconds: f64) -> String {
        let ms = (seconds * 1000.0) as u64;
        let secs = ms / 1000;
        let remainder = ms % 1000;
        format!("WAITFOR DELAY '00:00:{:02}.{:03}';", secs, remainder)
    }

    fn type_name(&self, dt: &DataType) -> String {
        match dt {
            DataType::Boolean => "BIT".to_string(),
            DataType::Integer => "INT".to_string(),
            DataType::BigInt => "BIGINT".to_string(),
            DataType::SmallInt => "SMALLINT".to_string(),
            DataType::Float => "FLOAT".to_string(),
            DataType::Decimal { precision, scale } => format!("DECIMAL({}, {})", precision, scale),
            DataType::Varchar(len) => format!("NVARCHAR({})", len),
            DataType::Char(len) => format!("NCHAR({})", len),
            DataType::Text => "NVARCHAR(MAX)".to_string(),
            DataType::Timestamp => "DATETIME2".to_string(),
            DataType::Date => "DATE".to_string(),
            DataType::Time => "TIME".to_string(),
            DataType::Blob => "VARBINARY(MAX)".to_string(),
            DataType::Uuid => "UNIQUEIDENTIFIER".to_string(),
            _ => "NVARCHAR(MAX)".to_string(),
        }
    }
}

fn tsql_isolation_str(level: &IsolationLevel) -> &'static str {
    match level {
        IsolationLevel::ReadUncommitted => "READ UNCOMMITTED",
        IsolationLevel::ReadCommitted => "READ COMMITTED",
        IsolationLevel::RepeatableRead => "REPEATABLE READ",
        IsolationLevel::Serializable => "SERIALIZABLE",
        _ => "SERIALIZABLE",
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Create a SqlGenerator for a given dialect.
pub fn generator_for(dialect: TargetDialect) -> Box<dyn SqlGenerator> {
    match dialect {
        TargetDialect::PostgreSql => Box::new(PostgreSqlGenerator),
        TargetDialect::MySql => Box::new(MySqlGenerator),
        TargetDialect::SqlServer => Box::new(SqlServerGenerator),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn witness_columns() -> Vec<ColumnSpec> {
        vec![
            ColumnSpec::new("id", DataType::Integer).primary_key(),
            ColumnSpec::new("val", DataType::Integer).not_null(),
        ]
    }

    #[test]
    fn test_pg_create_table() {
        let gen = PostgreSqlGenerator;
        let sql = gen.create_table("test_data", &witness_columns());
        assert!(sql.contains("CREATE TABLE test_data"));
        assert!(sql.contains("PRIMARY KEY"));
    }

    #[test]
    fn test_pg_crud() {
        let gen = PostgreSqlGenerator;
        assert!(gen.select("t", "id", "val", 1).contains("SELECT val"));
        assert!(gen.update("t", "id", "val", 1, "42").contains("UPDATE t SET"));
        assert!(gen.insert("t", "id", "val", 1, "42").contains("INSERT INTO"));
        assert!(gen.delete("t", "id", 1).contains("DELETE FROM"));
    }

    #[test]
    fn test_pg_transaction() {
        let gen = PostgreSqlGenerator;
        let stmts = gen.begin_transaction(&IsolationLevel::RepeatableRead);
        assert!(stmts[0].contains("REPEATABLE READ"));
        assert_eq!(gen.commit(), "COMMIT;");
        assert_eq!(gen.rollback(), "ROLLBACK;");
    }

    #[test]
    fn test_pg_advisory_lock() {
        let gen = PostgreSqlGenerator;
        assert!(gen.advisory_lock(42).contains("pg_advisory_lock(42)"));
        assert!(gen.advisory_unlock(42).contains("pg_advisory_unlock(42)"));
    }

    #[test]
    fn test_mysql_create_table() {
        let gen = MySqlGenerator;
        let sql = gen.create_table("test_data", &witness_columns());
        assert!(sql.contains("ENGINE=InnoDB"));
        assert!(sql.contains("`id`"));
    }

    #[test]
    fn test_mysql_transaction() {
        let gen = MySqlGenerator;
        let stmts = gen.begin_transaction(&IsolationLevel::ReadCommitted);
        assert!(stmts[0].contains("READ COMMITTED"));
        assert!(stmts[1].contains("START TRANSACTION"));
    }

    #[test]
    fn test_mysql_advisory_lock() {
        let gen = MySqlGenerator;
        assert!(gen.advisory_lock(7).contains("GET_LOCK"));
        assert!(gen.advisory_unlock(7).contains("RELEASE_LOCK"));
    }

    #[test]
    fn test_sqlserver_create_table() {
        let gen = SqlServerGenerator;
        let sql = gen.create_table("test_data", &witness_columns());
        assert!(sql.contains("[id]"));
        assert!(sql.contains("[val]"));
    }

    #[test]
    fn test_sqlserver_select_for_update() {
        let gen = SqlServerGenerator;
        let sql = gen.select_for_update("t", "id", "val", 1);
        assert!(sql.contains("UPDLOCK"));
        assert!(sql.contains("ROWLOCK"));
    }

    #[test]
    fn test_sqlserver_sleep() {
        let gen = SqlServerGenerator;
        let sql = gen.sleep(1.5);
        assert!(sql.contains("WAITFOR DELAY"));
    }

    #[test]
    fn test_sqlserver_advisory_lock() {
        let gen = SqlServerGenerator;
        assert!(gen.advisory_lock(1).contains("sp_getapplock"));
        assert!(gen.advisory_unlock(1).contains("sp_releaseapplock"));
    }

    #[test]
    fn test_generator_factory() {
        let pg = generator_for(TargetDialect::PostgreSql);
        assert_eq!(pg.dialect(), TargetDialect::PostgreSql);
        let mysql = generator_for(TargetDialect::MySql);
        assert_eq!(mysql.dialect(), TargetDialect::MySql);
        let mssql = generator_for(TargetDialect::SqlServer);
        assert_eq!(mssql.dialect(), TargetDialect::SqlServer);
    }

    #[test]
    fn test_type_mapping_pg() {
        let gen = PostgreSqlGenerator;
        assert_eq!(gen.type_name(&DataType::Boolean), "BOOLEAN");
        assert_eq!(gen.type_name(&DataType::Blob), "BYTEA");
        assert_eq!(gen.type_name(&DataType::Json), "JSONB");
    }

    #[test]
    fn test_type_mapping_mysql() {
        let gen = MySqlGenerator;
        assert_eq!(gen.type_name(&DataType::Boolean), "TINYINT(1)");
        assert_eq!(gen.type_name(&DataType::Timestamp), "DATETIME(6)");
        assert_eq!(gen.type_name(&DataType::Uuid), "CHAR(36)");
    }

    #[test]
    fn test_type_mapping_sqlserver() {
        let gen = SqlServerGenerator;
        assert_eq!(gen.type_name(&DataType::Boolean), "BIT");
        assert_eq!(gen.type_name(&DataType::Timestamp), "DATETIME2");
        assert_eq!(gen.type_name(&DataType::Text), "NVARCHAR(MAX)");
    }

    #[test]
    fn test_column_spec_builder() {
        let col = ColumnSpec::new("age", DataType::Integer)
            .not_null()
            .with_default("0");
        assert!(col.not_null);
        assert_eq!(col.default_value, Some("0".to_string()));
    }
}
