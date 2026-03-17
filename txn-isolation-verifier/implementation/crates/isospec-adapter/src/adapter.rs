//! Database adapter for connecting to real database engines.
//!
//! Provides a trait-based abstraction for executing SQL against PostgreSQL,
//! MySQL, and SQL Server, with connection pooling and result collection.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use isospec_types::config::EngineKind;
use isospec_types::error::{IsoSpecError, IsoSpecResult};
use isospec_types::value::Value;

// ---------------------------------------------------------------------------
// AdapterConfig
// ---------------------------------------------------------------------------

/// Configuration for a database adapter connection.
#[derive(Debug, Clone)]
pub struct AdapterConfig {
    pub engine: EngineKind,
    pub host: String,
    pub port: u16,
    pub database: String,
    pub username: String,
    pub password: String,
    pub pool_size: usize,
    pub connect_timeout: Duration,
    pub statement_timeout: Duration,
    pub ssl_mode: SslMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SslMode {
    Disable,
    Prefer,
    Require,
}

impl Default for SslMode {
    fn default() -> Self {
        SslMode::Prefer
    }
}

impl Default for AdapterConfig {
    fn default() -> Self {
        Self {
            engine: EngineKind::PostgreSQL,
            host: "localhost".to_string(),
            port: 5432,
            database: "isospec_test".to_string(),
            username: "postgres".to_string(),
            password: String::new(),
            pool_size: 4,
            connect_timeout: Duration::from_secs(10),
            statement_timeout: Duration::from_secs(30),
            ssl_mode: SslMode::Disable,
        }
    }
}

impl AdapterConfig {
    pub fn postgres() -> Self {
        Self {
            engine: EngineKind::PostgreSQL,
            port: 5432,
            username: "postgres".to_string(),
            ..Default::default()
        }
    }

    pub fn mysql() -> Self {
        Self {
            engine: EngineKind::MySQL,
            port: 3306,
            username: "root".to_string(),
            ..Default::default()
        }
    }

    pub fn sqlserver() -> Self {
        Self {
            engine: EngineKind::SqlServer,
            port: 1433,
            username: "sa".to_string(),
            ..Default::default()
        }
    }

    pub fn with_host(mut self, host: &str) -> Self {
        self.host = host.to_string();
        self
    }

    pub fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    pub fn with_database(mut self, db: &str) -> Self {
        self.database = db.to_string();
        self
    }

    pub fn with_credentials(mut self, user: &str, pass: &str) -> Self {
        self.username = user.to_string();
        self.password = pass.to_string();
        self
    }

    pub fn with_pool_size(mut self, size: usize) -> Self {
        self.pool_size = size;
        self
    }
}

// ---------------------------------------------------------------------------
// QueryResult
// ---------------------------------------------------------------------------

/// Result of executing a SQL query.
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Column names.
    pub columns: Vec<String>,
    /// Row data.
    pub rows: Vec<Vec<Value>>,
    /// Number of rows affected (for DML).
    pub rows_affected: u64,
    /// Execution time.
    pub elapsed: Duration,
    /// Engine-specific messages or notices.
    pub messages: Vec<String>,
}

impl QueryResult {
    pub fn empty() -> Self {
        Self {
            columns: Vec::new(),
            rows: Vec::new(),
            rows_affected: 0,
            elapsed: Duration::ZERO,
            messages: Vec::new(),
        }
    }

    pub fn with_rows(columns: Vec<String>, rows: Vec<Vec<Value>>) -> Self {
        Self {
            columns,
            rows,
            rows_affected: 0,
            elapsed: Duration::ZERO,
            messages: Vec::new(),
        }
    }

    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn get_value(&self, row: usize, col: usize) -> Option<&Value> {
        self.rows.get(row).and_then(|r| r.get(col))
    }

    pub fn get_by_name(&self, row: usize, col_name: &str) -> Option<&Value> {
        let col_idx = self.columns.iter().position(|c| c == col_name)?;
        self.get_value(row, col_idx)
    }

    /// Convert to a simple map for single-row results.
    pub fn as_map(&self) -> Option<HashMap<String, Value>> {
        if self.rows.is_empty() {
            return None;
        }
        let mut map = HashMap::new();
        for (idx, col) in self.columns.iter().enumerate() {
            if let Some(val) = self.rows[0].get(idx) {
                map.insert(col.clone(), val.clone());
            }
        }
        Some(map)
    }
}

impl fmt::Display for QueryResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QueryResult({} rows, {} cols, {} affected, {:.3}s)",
            self.row_count(),
            self.columns.len(),
            self.rows_affected,
            self.elapsed.as_secs_f64(),
        )
    }
}

// ---------------------------------------------------------------------------
// DatabaseAdapter trait
// ---------------------------------------------------------------------------

/// Trait for database engine adapters.
pub trait DatabaseAdapter: Send + Sync {
    /// Get the engine kind.
    fn engine(&self) -> EngineKind;

    /// Execute a SQL statement and return results.
    fn execute(&self, sql: &str) -> IsoSpecResult<QueryResult>;

    /// Execute a batch of SQL statements.
    fn execute_batch(&self, statements: &[String]) -> IsoSpecResult<Vec<QueryResult>>;

    /// Test connectivity.
    fn ping(&self) -> IsoSpecResult<bool>;

    /// Get the server version string.
    fn server_version(&self) -> IsoSpecResult<String>;

    /// Close all connections.
    fn close(&self) -> IsoSpecResult<()>;
}

// ---------------------------------------------------------------------------
// ConnectionPool abstraction
// ---------------------------------------------------------------------------

/// A simple connection pool that manages multiple adapter instances.
#[derive(Debug)]
pub struct ConnectionPool {
    config: AdapterConfig,
    /// Available connection indices.
    available: Arc<Mutex<Vec<usize>>>,
    /// Total pool capacity.
    capacity: usize,
}

impl ConnectionPool {
    pub fn new(config: AdapterConfig) -> Self {
        let capacity = config.pool_size;
        let available: Vec<usize> = (0..capacity).collect();
        Self {
            config,
            available: Arc::new(Mutex::new(available)),
            capacity,
        }
    }

    /// Acquire a connection slot from the pool.
    pub fn acquire(&self) -> IsoSpecResult<PooledConnection> {
        let mut available = self
            .available
            .lock()
            .map_err(|e| IsoSpecError::SmtSolver { msg: format!("pool lock error: {}", e) })?;
        if let Some(idx) = available.pop() {
            Ok(PooledConnection {
                index: idx,
                pool: Arc::clone(&self.available),
                config: self.config.clone(),
            })
        } else {
            Err(IsoSpecError::SmtSolver {
                msg: "connection pool exhausted".to_string(),
            })
        }
    }

    pub fn available_count(&self) -> usize {
        self.available.lock().map(|a| a.len()).unwrap_or(0)
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

/// A pooled connection that returns its slot to the pool on drop.
#[derive(Debug)]
pub struct PooledConnection {
    pub index: usize,
    pool: Arc<Mutex<Vec<usize>>>,
    pub config: AdapterConfig,
}

impl Drop for PooledConnection {
    fn drop(&mut self) {
        if let Ok(mut available) = self.pool.lock() {
            available.push(self.index);
        }
    }
}

// ---------------------------------------------------------------------------
// MockAdapter for testing
// ---------------------------------------------------------------------------

/// A mock database adapter for testing without a real database.
pub struct MockAdapter {
    engine: EngineKind,
    /// Pre-configured results for queries matching specific patterns.
    responses: Arc<Mutex<Vec<(String, QueryResult)>>>,
    /// Recorded executed SQL statements.
    executed: Arc<Mutex<Vec<String>>>,
    /// Whether ping should succeed.
    ping_ok: bool,
}

impl MockAdapter {
    pub fn new(engine: EngineKind) -> Self {
        Self {
            engine,
            responses: Arc::new(Mutex::new(Vec::new())),
            executed: Arc::new(Mutex::new(Vec::new())),
            ping_ok: true,
        }
    }

    /// Add a canned response for SQL matching a pattern.
    pub fn add_response(&self, pattern: &str, result: QueryResult) {
        if let Ok(mut responses) = self.responses.lock() {
            responses.push((pattern.to_string(), result));
        }
    }

    /// Get all executed SQL statements.
    pub fn executed_sql(&self) -> Vec<String> {
        self.executed.lock().map(|e| e.clone()).unwrap_or_default()
    }

    pub fn set_ping_ok(&mut self, ok: bool) {
        self.ping_ok = ok;
    }
}

impl DatabaseAdapter for MockAdapter {
    fn engine(&self) -> EngineKind {
        self.engine
    }

    fn execute(&self, sql: &str) -> IsoSpecResult<QueryResult> {
        if let Ok(mut executed) = self.executed.lock() {
            executed.push(sql.to_string());
        }

        if let Ok(responses) = self.responses.lock() {
            for (pattern, result) in responses.iter() {
                if sql.contains(pattern.as_str()) {
                    return Ok(result.clone());
                }
            }
        }

        Ok(QueryResult::empty())
    }

    fn execute_batch(&self, statements: &[String]) -> IsoSpecResult<Vec<QueryResult>> {
        let mut results = Vec::new();
        for stmt in statements {
            results.push(self.execute(stmt)?);
        }
        Ok(results)
    }

    fn ping(&self) -> IsoSpecResult<bool> {
        Ok(self.ping_ok)
    }

    fn server_version(&self) -> IsoSpecResult<String> {
        match self.engine {
            EngineKind::PostgreSQL => Ok("PostgreSQL 15.4 (mock)".to_string()),
            EngineKind::MySQL => Ok("8.0.34 (mock)".to_string()),
            EngineKind::SqlServer => Ok("16.0.1000.6 (mock)".to_string()),
        }
    }

    fn close(&self) -> IsoSpecResult<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AdapterFactory
// ---------------------------------------------------------------------------

/// Create a mock adapter for a given engine kind.
pub fn create_mock_adapter(engine: EngineKind) -> MockAdapter {
    MockAdapter::new(engine)
}

/// Get the default port for an engine kind.
pub fn default_port(engine: &EngineKind) -> u16 {
    match engine {
        EngineKind::PostgreSQL => 5432,
        EngineKind::MySQL => 3306,
        EngineKind::SqlServer => 1433,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_config_defaults() {
        let config = AdapterConfig::default();
        assert_eq!(config.host, "localhost");
        assert_eq!(config.port, 5432);
        assert_eq!(config.pool_size, 4);
    }

    #[test]
    fn test_adapter_config_postgres() {
        let config = AdapterConfig::postgres();
        assert_eq!(config.engine, EngineKind::PostgreSQL);
        assert_eq!(config.port, 5432);
    }

    #[test]
    fn test_adapter_config_mysql() {
        let config = AdapterConfig::mysql();
        assert_eq!(config.engine, EngineKind::MySQL);
        assert_eq!(config.port, 3306);
    }

    #[test]
    fn test_adapter_config_builder() {
        let config = AdapterConfig::postgres()
            .with_host("db.example.com")
            .with_port(5433)
            .with_database("mydb")
            .with_credentials("user", "pass")
            .with_pool_size(8);
        assert_eq!(config.host, "db.example.com");
        assert_eq!(config.port, 5433);
        assert_eq!(config.database, "mydb");
        assert_eq!(config.username, "user");
        assert_eq!(config.pool_size, 8);
    }

    #[test]
    fn test_query_result_empty() {
        let qr = QueryResult::empty();
        assert!(qr.is_empty());
        assert_eq!(qr.row_count(), 0);
    }

    #[test]
    fn test_query_result_with_rows() {
        let qr = QueryResult::with_rows(
            vec!["id".into(), "val".into()],
            vec![
                vec![Value::Integer(1), Value::Integer(42)],
                vec![Value::Integer(2), Value::Integer(99)],
            ],
        );
        assert_eq!(qr.row_count(), 2);
        assert_eq!(qr.get_value(0, 1), Some(&Value::Integer(42)));
        assert_eq!(qr.get_by_name(1, "val"), Some(&Value::Integer(99)));
    }

    #[test]
    fn test_query_result_as_map() {
        let qr = QueryResult::with_rows(
            vec!["name".into()],
            vec![vec![Value::Text("hello".into())]],
        );
        let map = qr.as_map().unwrap();
        assert_eq!(map.get("name"), Some(&Value::Text("hello".into())));
    }

    #[test]
    fn test_mock_adapter_execute() {
        let adapter = MockAdapter::new(EngineKind::PostgreSQL);
        adapter.add_response(
            "SELECT",
            QueryResult::with_rows(
                vec!["val".into()],
                vec![vec![Value::Integer(42)]],
            ),
        );
        let result = adapter.execute("SELECT val FROM t WHERE id = 1").unwrap();
        assert_eq!(result.row_count(), 1);
    }

    #[test]
    fn test_mock_adapter_records_sql() {
        let adapter = MockAdapter::new(EngineKind::MySQL);
        adapter.execute("INSERT INTO t VALUES (1, 2)").unwrap();
        adapter.execute("UPDATE t SET val = 3 WHERE id = 1").unwrap();
        let executed = adapter.executed_sql();
        assert_eq!(executed.len(), 2);
    }

    #[test]
    fn test_mock_adapter_batch() {
        let adapter = MockAdapter::new(EngineKind::PostgreSQL);
        let stmts = vec!["SELECT 1;".into(), "SELECT 2;".into()];
        let results = adapter.execute_batch(&stmts).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_mock_adapter_ping() {
        let adapter = MockAdapter::new(EngineKind::PostgreSQL);
        assert!(adapter.ping().unwrap());
    }

    #[test]
    fn test_connection_pool() {
        let config = AdapterConfig::default().with_pool_size(2);
        let pool = ConnectionPool::new(config);
        assert_eq!(pool.capacity(), 2);
        assert_eq!(pool.available_count(), 2);

        let conn1 = pool.acquire().unwrap();
        assert_eq!(pool.available_count(), 1);

        let conn2 = pool.acquire().unwrap();
        assert_eq!(pool.available_count(), 0);

        assert!(pool.acquire().is_err());

        drop(conn1);
        assert_eq!(pool.available_count(), 1);

        drop(conn2);
        assert_eq!(pool.available_count(), 2);
    }

    #[test]
    fn test_default_port() {
        assert_eq!(default_port(&EngineKind::PostgreSQL), 5432);
        assert_eq!(default_port(&EngineKind::MySQL), 3306);
        assert_eq!(default_port(&EngineKind::SqlServer), 1433);
    }

    #[test]
    fn test_query_result_display() {
        let qr = QueryResult::with_rows(
            vec!["x".into()],
            vec![vec![Value::Integer(1)]],
        );
        let display = format!("{}", qr);
        assert!(display.contains("1 rows"));
    }
}
