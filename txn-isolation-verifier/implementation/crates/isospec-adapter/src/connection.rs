//! Connection management for database engines.
//!
//! Provides connection configuration, connection string builders, and
//! connection pooling abstractions for PostgreSQL, MySQL, and SQL Server.

use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

use isospec_types::config::EngineKind;
use isospec_types::error::{IsoSpecError, IsoSpecResult};

// ---------------------------------------------------------------------------
// ConnectionConfig
// ---------------------------------------------------------------------------

/// Connection configuration for a single database engine.
#[derive(Debug, Clone)]
pub struct ConnectionConfig {
    pub engine: EngineKind,
    pub host: String,
    pub port: u16,
    pub database: String,
    pub username: String,
    pub password: String,
    pub ssl_enabled: bool,
    pub connect_timeout: Duration,
    pub statement_timeout: Duration,
    /// Extra connection parameters.
    pub params: HashMap<String, String>,
}

impl ConnectionConfig {
    pub fn new(engine: EngineKind) -> Self {
        let (default_port, default_user) = match engine {
            EngineKind::PostgreSQL => (5432, "postgres"),
            EngineKind::MySQL => (3306, "root"),
            EngineKind::SqlServer => (1433, "sa"),
        };
        Self {
            engine,
            host: "localhost".to_string(),
            port: default_port,
            database: "isospec_test".to_string(),
            username: default_user.to_string(),
            password: String::new(),
            ssl_enabled: false,
            connect_timeout: Duration::from_secs(10),
            statement_timeout: Duration::from_secs(30),
            params: HashMap::new(),
        }
    }

    pub fn postgres() -> Self {
        Self::new(EngineKind::PostgreSQL)
    }

    pub fn mysql() -> Self {
        Self::new(EngineKind::MySQL)
    }

    pub fn sqlserver() -> Self {
        Self::new(EngineKind::SqlServer)
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

    pub fn with_ssl(mut self, enabled: bool) -> Self {
        self.ssl_enabled = enabled;
        self
    }

    pub fn with_connect_timeout(mut self, timeout: Duration) -> Self {
        self.connect_timeout = timeout;
        self
    }

    pub fn with_statement_timeout(mut self, timeout: Duration) -> Self {
        self.statement_timeout = timeout;
        self
    }

    pub fn with_param(mut self, key: &str, value: &str) -> Self {
        self.params.insert(key.to_string(), value.to_string());
        self
    }
}

// ---------------------------------------------------------------------------
// Connection string builders
// ---------------------------------------------------------------------------

/// Builds connection strings for different database engines.
pub struct ConnectionStringBuilder;

impl ConnectionStringBuilder {
    /// Build a connection string for the given config.
    pub fn build(config: &ConnectionConfig) -> String {
        match config.engine {
            EngineKind::PostgreSQL => Self::build_postgres(config),
            EngineKind::MySQL => Self::build_mysql(config),
            EngineKind::SqlServer => Self::build_sqlserver(config),
        }
    }

    /// Build a PostgreSQL connection string (libpq format).
    fn build_postgres(config: &ConnectionConfig) -> String {
        let mut parts = Vec::new();
        parts.push(format!("host={}", config.host));
        parts.push(format!("port={}", config.port));
        parts.push(format!("dbname={}", config.database));
        parts.push(format!("user={}", config.username));
        if !config.password.is_empty() {
            parts.push(format!("password={}", config.password));
        }
        if config.ssl_enabled {
            parts.push("sslmode=require".to_string());
        } else {
            parts.push("sslmode=disable".to_string());
        }
        parts.push(format!(
            "connect_timeout={}",
            config.connect_timeout.as_secs()
        ));

        // Statement timeout in milliseconds
        let stmt_ms = config.statement_timeout.as_millis();
        parts.push(format!("options=-c statement_timeout={}", stmt_ms));

        for (key, value) in &config.params {
            parts.push(format!("{}={}", key, value));
        }

        parts.join(" ")
    }

    /// Build a PostgreSQL connection URI.
    pub fn build_postgres_uri(config: &ConnectionConfig) -> String {
        let auth = if config.password.is_empty() {
            config.username.clone()
        } else {
            format!("{}:{}", config.username, config.password)
        };
        let ssl_param = if config.ssl_enabled {
            "sslmode=require"
        } else {
            "sslmode=disable"
        };
        format!(
            "postgresql://{}@{}:{}/{}?{}",
            auth, config.host, config.port, config.database, ssl_param
        )
    }

    /// Build a MySQL connection string (DSN format).
    fn build_mysql(config: &ConnectionConfig) -> String {
        let mut parts = Vec::new();
        parts.push(format!("{}:{}", config.host, config.port));
        parts.push(format!("user={}", config.username));
        if !config.password.is_empty() {
            parts.push(format!("password={}", config.password));
        }
        parts.push(format!("database={}", config.database));
        parts.push(format!(
            "timeout={}",
            config.connect_timeout.as_secs()
        ));
        if !config.ssl_enabled {
            parts.push("ssl-mode=DISABLED".to_string());
        }
        for (key, value) in &config.params {
            parts.push(format!("{}={}", key, value));
        }
        parts.join(";")
    }

    /// Build a MySQL JDBC-style URL.
    pub fn build_mysql_url(config: &ConnectionConfig) -> String {
        let mut params = Vec::new();
        params.push(format!(
            "connectTimeout={}",
            config.connect_timeout.as_millis()
        ));
        if !config.ssl_enabled {
            params.push("useSSL=false".to_string());
        }
        for (key, value) in &config.params {
            params.push(format!("{}={}", key, value));
        }
        format!(
            "jdbc:mysql://{}:{}/{}?{}",
            config.host,
            config.port,
            config.database,
            params.join("&")
        )
    }

    /// Build a SQL Server connection string (ADO.NET/ODBC format).
    fn build_sqlserver(config: &ConnectionConfig) -> String {
        let mut parts = Vec::new();
        parts.push(format!("Server={},{}", config.host, config.port));
        parts.push(format!("Database={}", config.database));
        parts.push(format!("User Id={}", config.username));
        if !config.password.is_empty() {
            parts.push(format!("Password={}", config.password));
        }
        parts.push(format!(
            "Connection Timeout={}",
            config.connect_timeout.as_secs()
        ));
        if config.ssl_enabled {
            parts.push("Encrypt=true".to_string());
            parts.push("TrustServerCertificate=true".to_string());
        } else {
            parts.push("Encrypt=false".to_string());
        }
        for (key, value) in &config.params {
            parts.push(format!("{}={}", key, value));
        }
        parts.join(";")
    }

    /// Build a SQL Server JDBC-style URL.
    pub fn build_sqlserver_url(config: &ConnectionConfig) -> String {
        let mut params = Vec::new();
        params.push(format!("databaseName={}", config.database));
        params.push(format!("user={}", config.username));
        if !config.password.is_empty() {
            params.push(format!("password={}", config.password));
        }
        if !config.ssl_enabled {
            params.push("encrypt=false".to_string());
        }
        for (key, value) in &config.params {
            params.push(format!("{}={}", key, value));
        }
        format!(
            "jdbc:sqlserver://{}:{};{}",
            config.host,
            config.port,
            params.join(";")
        )
    }

    /// Build a CLI command for the engine's command-line client.
    pub fn build_cli_command(config: &ConnectionConfig) -> Vec<String> {
        match config.engine {
            EngineKind::PostgreSQL => {
                let mut cmd = vec!["psql".to_string()];
                cmd.push("-h".to_string());
                cmd.push(config.host.clone());
                cmd.push("-p".to_string());
                cmd.push(config.port.to_string());
                cmd.push("-U".to_string());
                cmd.push(config.username.clone());
                cmd.push("-d".to_string());
                cmd.push(config.database.clone());
                cmd
            }
            EngineKind::MySQL => {
                let mut cmd = vec!["mysql".to_string()];
                cmd.push("-h".to_string());
                cmd.push(config.host.clone());
                cmd.push("-P".to_string());
                cmd.push(config.port.to_string());
                cmd.push("-u".to_string());
                cmd.push(config.username.clone());
                if !config.password.is_empty() {
                    cmd.push(format!("-p{}", config.password));
                }
                cmd.push(config.database.clone());
                cmd
            }
            EngineKind::SqlServer => {
                let mut cmd = vec!["sqlcmd".to_string()];
                cmd.push("-S".to_string());
                cmd.push(format!("{},{}", config.host, config.port));
                cmd.push("-U".to_string());
                cmd.push(config.username.clone());
                if !config.password.is_empty() {
                    cmd.push("-P".to_string());
                    cmd.push(config.password.clone());
                }
                cmd.push("-d".to_string());
                cmd.push(config.database.clone());
                cmd
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PoolConfig
// ---------------------------------------------------------------------------

/// Configuration for a connection pool.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Minimum number of idle connections.
    pub min_idle: usize,
    /// Maximum number of connections.
    pub max_size: usize,
    /// Maximum time to wait for a connection from the pool.
    pub acquire_timeout: Duration,
    /// Maximum idle time before a connection is closed.
    pub idle_timeout: Duration,
    /// Maximum lifetime of a connection.
    pub max_lifetime: Duration,
    /// Whether to validate connections on checkout.
    pub test_on_checkout: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            min_idle: 1,
            max_size: 4,
            acquire_timeout: Duration::from_secs(10),
            idle_timeout: Duration::from_secs(300),
            max_lifetime: Duration::from_secs(1800),
            test_on_checkout: true,
        }
    }
}

impl PoolConfig {
    pub fn with_max_size(mut self, size: usize) -> Self {
        self.max_size = size;
        self
    }

    pub fn with_min_idle(mut self, min: usize) -> Self {
        self.min_idle = min;
        self
    }

    pub fn with_acquire_timeout(mut self, timeout: Duration) -> Self {
        self.acquire_timeout = timeout;
        self
    }
}

// ---------------------------------------------------------------------------
// PoolState
// ---------------------------------------------------------------------------

/// Snapshot of a connection pool's current state.
#[derive(Debug, Clone)]
pub struct PoolState {
    pub total_connections: usize,
    pub idle_connections: usize,
    pub active_connections: usize,
    pub waiting_requests: usize,
    pub total_acquired: u64,
    pub total_released: u64,
    pub total_timeouts: u64,
}

impl PoolState {
    pub fn new(max_size: usize) -> Self {
        Self {
            total_connections: max_size,
            idle_connections: max_size,
            active_connections: 0,
            waiting_requests: 0,
            total_acquired: 0,
            total_released: 0,
            total_timeouts: 0,
        }
    }

    pub fn utilization(&self) -> f64 {
        if self.total_connections == 0 {
            0.0
        } else {
            self.active_connections as f64 / self.total_connections as f64
        }
    }
}

impl fmt::Display for PoolState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Pool: {}/{} active, {} idle, {} waiting ({:.0}% util)",
            self.active_connections,
            self.total_connections,
            self.idle_connections,
            self.waiting_requests,
            self.utilization() * 100.0,
        )
    }
}

// ---------------------------------------------------------------------------
// MultiEngineConnectionManager
// ---------------------------------------------------------------------------

/// Manages connections to multiple database engines simultaneously.
pub struct MultiEngineConnectionManager {
    configs: HashMap<EngineKind, ConnectionConfig>,
}

impl MultiEngineConnectionManager {
    pub fn new() -> Self {
        Self {
            configs: HashMap::new(),
        }
    }

    pub fn add_engine(&mut self, config: ConnectionConfig) {
        self.configs.insert(config.engine, config);
    }

    pub fn connection_string(&self, engine: &EngineKind) -> Option<String> {
        self.configs
            .get(engine)
            .map(|c| ConnectionStringBuilder::build(c))
    }

    pub fn cli_command(&self, engine: &EngineKind) -> Option<Vec<String>> {
        self.configs
            .get(engine)
            .map(|c| ConnectionStringBuilder::build_cli_command(c))
    }

    pub fn engines(&self) -> Vec<EngineKind> {
        self.configs.keys().copied().collect()
    }

    pub fn config_for(&self, engine: &EngineKind) -> Option<&ConnectionConfig> {
        self.configs.get(engine)
    }

    pub fn has_engine(&self, engine: &EngineKind) -> bool {
        self.configs.contains_key(engine)
    }
}

impl Default for MultiEngineConnectionManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection_config_postgres() {
        let config = ConnectionConfig::postgres();
        assert_eq!(config.engine, EngineKind::PostgreSQL);
        assert_eq!(config.port, 5432);
        assert_eq!(config.username, "postgres");
    }

    #[test]
    fn test_connection_config_mysql() {
        let config = ConnectionConfig::mysql();
        assert_eq!(config.engine, EngineKind::MySQL);
        assert_eq!(config.port, 3306);
    }

    #[test]
    fn test_connection_config_sqlserver() {
        let config = ConnectionConfig::sqlserver();
        assert_eq!(config.engine, EngineKind::SqlServer);
        assert_eq!(config.port, 1433);
    }

    #[test]
    fn test_connection_config_builder() {
        let config = ConnectionConfig::postgres()
            .with_host("db.example.com")
            .with_port(5433)
            .with_database("mydb")
            .with_credentials("user", "pass")
            .with_ssl(true)
            .with_param("application_name", "isospec");
        assert_eq!(config.host, "db.example.com");
        assert_eq!(config.port, 5433);
        assert!(config.ssl_enabled);
        assert_eq!(config.params.get("application_name"), Some(&"isospec".to_string()));
    }

    #[test]
    fn test_postgres_connection_string() {
        let config = ConnectionConfig::postgres()
            .with_credentials("user", "pass");
        let cs = ConnectionStringBuilder::build(&config);
        assert!(cs.contains("host=localhost"));
        assert!(cs.contains("port=5432"));
        assert!(cs.contains("user=user"));
        assert!(cs.contains("password=pass"));
        assert!(cs.contains("sslmode=disable"));
    }

    #[test]
    fn test_postgres_uri() {
        let config = ConnectionConfig::postgres()
            .with_credentials("user", "pass");
        let uri = ConnectionStringBuilder::build_postgres_uri(&config);
        assert!(uri.starts_with("postgresql://"));
        assert!(uri.contains("user:pass@"));
    }

    #[test]
    fn test_mysql_connection_string() {
        let config = ConnectionConfig::mysql()
            .with_credentials("root", "secret");
        let cs = ConnectionStringBuilder::build(&config);
        assert!(cs.contains("localhost:3306"));
        assert!(cs.contains("user=root"));
    }

    #[test]
    fn test_mysql_url() {
        let config = ConnectionConfig::mysql();
        let url = ConnectionStringBuilder::build_mysql_url(&config);
        assert!(url.starts_with("jdbc:mysql://"));
    }

    #[test]
    fn test_sqlserver_connection_string() {
        let config = ConnectionConfig::sqlserver()
            .with_credentials("sa", "MyPass1!");
        let cs = ConnectionStringBuilder::build(&config);
        assert!(cs.contains("Server=localhost,1433"));
        assert!(cs.contains("User Id=sa"));
    }

    #[test]
    fn test_sqlserver_url() {
        let config = ConnectionConfig::sqlserver();
        let url = ConnectionStringBuilder::build_sqlserver_url(&config);
        assert!(url.starts_with("jdbc:sqlserver://"));
    }

    #[test]
    fn test_cli_command_postgres() {
        let config = ConnectionConfig::postgres();
        let cmd = ConnectionStringBuilder::build_cli_command(&config);
        assert_eq!(cmd[0], "psql");
        assert!(cmd.contains(&"-h".to_string()));
    }

    #[test]
    fn test_cli_command_mysql() {
        let config = ConnectionConfig::mysql();
        let cmd = ConnectionStringBuilder::build_cli_command(&config);
        assert_eq!(cmd[0], "mysql");
    }

    #[test]
    fn test_cli_command_sqlserver() {
        let config = ConnectionConfig::sqlserver();
        let cmd = ConnectionStringBuilder::build_cli_command(&config);
        assert_eq!(cmd[0], "sqlcmd");
    }

    #[test]
    fn test_pool_config_defaults() {
        let pc = PoolConfig::default();
        assert_eq!(pc.min_idle, 1);
        assert_eq!(pc.max_size, 4);
        assert!(pc.test_on_checkout);
    }

    #[test]
    fn test_pool_state() {
        let mut state = PoolState::new(4);
        assert_eq!(state.utilization(), 0.0);
        state.active_connections = 2;
        state.idle_connections = 2;
        assert!((state.utilization() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_pool_state_display() {
        let state = PoolState::new(4);
        let display = format!("{}", state);
        assert!(display.contains("0/4 active"));
    }

    #[test]
    fn test_multi_engine_manager() {
        let mut mgr = MultiEngineConnectionManager::new();
        mgr.add_engine(ConnectionConfig::postgres());
        mgr.add_engine(ConnectionConfig::mysql());

        assert!(mgr.has_engine(&EngineKind::PostgreSQL));
        assert!(mgr.has_engine(&EngineKind::MySQL));
        assert!(!mgr.has_engine(&EngineKind::SqlServer));
        assert_eq!(mgr.engines().len(), 2);

        let pg_cs = mgr.connection_string(&EngineKind::PostgreSQL);
        assert!(pg_cs.is_some());
        assert!(pg_cs.unwrap().contains("host=localhost"));
    }

    #[test]
    fn test_multi_engine_cli() {
        let mut mgr = MultiEngineConnectionManager::new();
        mgr.add_engine(ConnectionConfig::postgres());
        let cmd = mgr.cli_command(&EngineKind::PostgreSQL).unwrap();
        assert_eq!(cmd[0], "psql");
    }
}
