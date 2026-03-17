//! Docker container management for database engine instances.
//!
//! Manages the lifecycle of PostgreSQL, MySQL, and SQL Server containers
//! for integration testing of witness schedules.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

use isospec_types::config::EngineKind;
use isospec_types::error::{IsoSpecError, IsoSpecResult};

// ---------------------------------------------------------------------------
// ContainerConfig
// ---------------------------------------------------------------------------

/// Configuration for a Docker container.
#[derive(Debug, Clone)]
pub struct ContainerConfig {
    /// Docker image name.
    pub image: String,
    /// Image tag.
    pub tag: String,
    /// Container name prefix.
    pub name_prefix: String,
    /// Port mapping: host_port -> container_port.
    pub port_mappings: Vec<(u16, u16)>,
    /// Environment variables.
    pub env_vars: HashMap<String, String>,
    /// Volumes: host_path -> container_path.
    pub volumes: HashMap<String, String>,
    /// Health check command.
    pub health_check_cmd: Option<String>,
    /// Maximum time to wait for container to become healthy.
    pub startup_timeout: Duration,
    /// Interval between health checks.
    pub health_check_interval: Duration,
    /// Memory limit in bytes (0 = unlimited).
    pub memory_limit: u64,
}

impl ContainerConfig {
    /// Standard configuration for PostgreSQL.
    pub fn postgres(host_port: u16) -> Self {
        let mut env = HashMap::new();
        env.insert("POSTGRES_PASSWORD".to_string(), "isospec_test".to_string());
        env.insert("POSTGRES_DB".to_string(), "isospec_test".to_string());
        Self {
            image: "postgres".to_string(),
            tag: "15".to_string(),
            name_prefix: "isospec-pg".to_string(),
            port_mappings: vec![(host_port, 5432)],
            env_vars: env,
            volumes: HashMap::new(),
            health_check_cmd: Some(
                "pg_isready -U postgres -d isospec_test".to_string(),
            ),
            startup_timeout: Duration::from_secs(30),
            health_check_interval: Duration::from_secs(2),
            memory_limit: 0,
        }
    }

    /// Standard configuration for MySQL.
    pub fn mysql(host_port: u16) -> Self {
        let mut env = HashMap::new();
        env.insert("MYSQL_ROOT_PASSWORD".to_string(), "isospec_test".to_string());
        env.insert("MYSQL_DATABASE".to_string(), "isospec_test".to_string());
        Self {
            image: "mysql".to_string(),
            tag: "8.0".to_string(),
            name_prefix: "isospec-mysql".to_string(),
            port_mappings: vec![(host_port, 3306)],
            env_vars: env,
            volumes: HashMap::new(),
            health_check_cmd: Some(
                "mysqladmin ping -h localhost -u root -pisospec_test".to_string(),
            ),
            startup_timeout: Duration::from_secs(60),
            health_check_interval: Duration::from_secs(3),
            memory_limit: 0,
        }
    }

    /// Standard configuration for SQL Server.
    pub fn sqlserver(host_port: u16) -> Self {
        let mut env = HashMap::new();
        env.insert("ACCEPT_EULA".to_string(), "Y".to_string());
        env.insert("SA_PASSWORD".to_string(), "IsoSpec_Test1!".to_string());
        env.insert("MSSQL_PID".to_string(), "Developer".to_string());
        Self {
            image: "mcr.microsoft.com/mssql/server".to_string(),
            tag: "2022-latest".to_string(),
            name_prefix: "isospec-mssql".to_string(),
            port_mappings: vec![(host_port, 1433)],
            env_vars: env,
            volumes: HashMap::new(),
            health_check_cmd: Some(
                "/opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P 'IsoSpec_Test1!' -Q 'SELECT 1'"
                    .to_string(),
            ),
            startup_timeout: Duration::from_secs(60),
            health_check_interval: Duration::from_secs(3),
            memory_limit: 0,
        }
    }

    /// Set a custom image tag.
    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tag = tag.to_string();
        self
    }

    pub fn with_env(mut self, key: &str, value: &str) -> Self {
        self.env_vars.insert(key.to_string(), value.to_string());
        self
    }

    pub fn with_volume(mut self, host: &str, container: &str) -> Self {
        self.volumes
            .insert(host.to_string(), container.to_string());
        self
    }

    pub fn with_memory_limit(mut self, bytes: u64) -> Self {
        self.memory_limit = bytes;
        self
    }

    /// Full image reference (image:tag).
    pub fn image_ref(&self) -> String {
        format!("{}:{}", self.image, self.tag)
    }

    /// Generate a unique container name.
    pub fn container_name(&self) -> String {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs()
            % 100000;
        format!("{}-{}", self.name_prefix, ts)
    }
}

// ---------------------------------------------------------------------------
// ContainerState
// ---------------------------------------------------------------------------

/// Current state of a managed container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContainerState {
    NotCreated,
    Created,
    Running,
    Healthy,
    Stopped,
    Failed(String),
}

impl fmt::Display for ContainerState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContainerState::NotCreated => write!(f, "not_created"),
            ContainerState::Created => write!(f, "created"),
            ContainerState::Running => write!(f, "running"),
            ContainerState::Healthy => write!(f, "healthy"),
            ContainerState::Stopped => write!(f, "stopped"),
            ContainerState::Failed(msg) => write!(f, "failed: {}", msg),
        }
    }
}

// ---------------------------------------------------------------------------
// ContainerInfo
// ---------------------------------------------------------------------------

/// Information about a running container.
#[derive(Debug, Clone)]
pub struct ContainerInfo {
    pub container_id: String,
    pub name: String,
    pub image: String,
    pub state: ContainerState,
    pub host_port: u16,
    pub engine: EngineKind,
    pub started_at: Option<Instant>,
}

impl ContainerInfo {
    pub fn is_healthy(&self) -> bool {
        self.state == ContainerState::Healthy
    }

    pub fn uptime(&self) -> Duration {
        self.started_at
            .map(|s| s.elapsed())
            .unwrap_or(Duration::ZERO)
    }
}

// ---------------------------------------------------------------------------
// DockerCommand builder
// ---------------------------------------------------------------------------

/// Builds docker CLI commands.
pub struct DockerCommand;

impl DockerCommand {
    /// Build a `docker run` command string.
    pub fn run(config: &ContainerConfig) -> String {
        let mut parts = Vec::new();
        parts.push("docker run -d".to_string());
        parts.push(format!("--name {}", config.container_name()));

        for (host, container) in &config.port_mappings {
            parts.push(format!("-p {}:{}", host, container));
        }

        for (key, value) in &config.env_vars {
            parts.push(format!("-e {}={}", key, value));
        }

        for (host_path, container_path) in &config.volumes {
            parts.push(format!("-v {}:{}", host_path, container_path));
        }

        if config.memory_limit > 0 {
            parts.push(format!("--memory {}", config.memory_limit));
        }

        parts.push(config.image_ref());
        parts.join(" ")
    }

    /// Build a `docker stop` command.
    pub fn stop(container_id: &str) -> String {
        format!("docker stop {}", container_id)
    }

    /// Build a `docker rm` command.
    pub fn remove(container_id: &str) -> String {
        format!("docker rm -f {}", container_id)
    }

    /// Build a `docker exec` command for health checking.
    pub fn health_check(container_id: &str, cmd: &str) -> String {
        format!("docker exec {} {}", container_id, cmd)
    }

    /// Build a `docker logs` command.
    pub fn logs(container_id: &str, tail: usize) -> String {
        format!("docker logs --tail {} {}", tail, container_id)
    }

    /// Build a `docker inspect` command for state.
    pub fn inspect_state(container_id: &str) -> String {
        format!(
            "docker inspect --format '{{{{.State.Status}}}}' {}",
            container_id
        )
    }

    /// Build a `docker pull` command.
    pub fn pull(image_ref: &str) -> String {
        format!("docker pull {}", image_ref)
    }
}

// ---------------------------------------------------------------------------
// DockerManager
// ---------------------------------------------------------------------------

/// Manages Docker containers for database engine instances.
pub struct DockerManager {
    /// Active containers keyed by engine kind.
    containers: HashMap<EngineKind, ContainerInfo>,
    /// Port allocation tracker.
    next_port: u16,
    /// Base port for automatic allocation.
    base_port: u16,
}

impl DockerManager {
    pub fn new() -> Self {
        Self {
            containers: HashMap::new(),
            next_port: 15432,
            base_port: 15432,
        }
    }

    pub fn with_base_port(mut self, port: u16) -> Self {
        self.base_port = port;
        self.next_port = port;
        self
    }

    /// Allocate the next available port.
    fn allocate_port(&mut self) -> u16 {
        let port = self.next_port;
        self.next_port += 1;
        port
    }

    /// Get a container configuration for an engine kind with auto-allocated port.
    pub fn config_for(&mut self, engine: EngineKind) -> ContainerConfig {
        let port = self.allocate_port();
        match engine {
            EngineKind::PostgreSQL => ContainerConfig::postgres(port),
            EngineKind::MySQL => ContainerConfig::mysql(port),
            EngineKind::SqlServer => ContainerConfig::sqlserver(port),
        }
    }

    /// Generate the docker run command for an engine.
    pub fn start_command(&mut self, engine: EngineKind) -> String {
        let config = self.config_for(engine);
        let name = config.container_name();
        let host_port = config.port_mappings.first().map(|(h, _)| *h).unwrap_or(0);
        let cmd = DockerCommand::run(&config);

        self.containers.insert(
            engine,
            ContainerInfo {
                container_id: String::new(), // Will be set after docker returns
                name,
                image: config.image_ref(),
                state: ContainerState::Created,
                host_port,
                engine,
                started_at: Some(Instant::now()),
            },
        );

        cmd
    }

    /// Record a container ID after starting.
    pub fn set_container_id(&mut self, engine: EngineKind, id: &str) {
        if let Some(info) = self.containers.get_mut(&engine) {
            info.container_id = id.trim().to_string();
            info.state = ContainerState::Running;
        }
    }

    /// Mark a container as healthy.
    pub fn mark_healthy(&mut self, engine: EngineKind) {
        if let Some(info) = self.containers.get_mut(&engine) {
            info.state = ContainerState::Healthy;
        }
    }

    /// Mark a container as failed.
    pub fn mark_failed(&mut self, engine: EngineKind, reason: &str) {
        if let Some(info) = self.containers.get_mut(&engine) {
            info.state = ContainerState::Failed(reason.to_string());
        }
    }

    /// Generate the stop command for an engine's container.
    pub fn stop_command(&self, engine: &EngineKind) -> Option<String> {
        self.containers
            .get(engine)
            .filter(|info| !info.container_id.is_empty())
            .map(|info| DockerCommand::stop(&info.container_id))
    }

    /// Generate the remove command for an engine's container.
    pub fn remove_command(&self, engine: &EngineKind) -> Option<String> {
        self.containers
            .get(engine)
            .filter(|info| !info.container_id.is_empty())
            .map(|info| DockerCommand::remove(&info.container_id))
    }

    /// Generate stop commands for all containers.
    pub fn stop_all_commands(&self) -> Vec<String> {
        self.containers
            .values()
            .filter(|info| !info.container_id.is_empty())
            .map(|info| DockerCommand::remove(&info.container_id))
            .collect()
    }

    /// Get info about a running container.
    pub fn container_info(&self, engine: &EngineKind) -> Option<&ContainerInfo> {
        self.containers.get(engine)
    }

    /// Get the host port for a running engine container.
    pub fn host_port(&self, engine: &EngineKind) -> Option<u16> {
        self.containers.get(engine).map(|i| i.host_port)
    }

    /// List all managed containers.
    pub fn list_containers(&self) -> Vec<&ContainerInfo> {
        self.containers.values().collect()
    }

    /// Check if any containers are running.
    pub fn has_running_containers(&self) -> bool {
        self.containers.values().any(|info| {
            matches!(
                info.state,
                ContainerState::Running | ContainerState::Healthy
            )
        })
    }
}

impl Default for DockerManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// HealthChecker
// ---------------------------------------------------------------------------

/// Generates health check commands and interprets results.
pub struct HealthChecker;

impl HealthChecker {
    /// Generate a health check command for a container.
    pub fn check_command(container_id: &str, engine: &EngineKind) -> String {
        let cmd = match engine {
            EngineKind::PostgreSQL => "pg_isready -U postgres",
            EngineKind::MySQL => "mysqladmin ping -u root -pisospec_test --silent",
            EngineKind::SqlServer => {
                "/opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P 'IsoSpec_Test1!' -Q 'SELECT 1' -b"
            }
        };
        DockerCommand::health_check(container_id, cmd)
    }

    /// Interpret a health check exit code.
    pub fn is_healthy(exit_code: i32) -> bool {
        exit_code == 0
    }

    /// Generate a sequence of health check attempts with delays.
    pub fn check_sequence(
        container_id: &str,
        engine: &EngineKind,
        max_attempts: usize,
        interval_secs: u64,
    ) -> Vec<(String, Duration)> {
        let cmd = Self::check_command(container_id, engine);
        (0..max_attempts)
            .map(|i| {
                let delay = Duration::from_secs(interval_secs * (i as u64));
                (cmd.clone(), delay)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_container_config_postgres() {
        let config = ContainerConfig::postgres(15432);
        assert_eq!(config.image, "postgres");
        assert_eq!(config.tag, "15");
        assert_eq!(config.port_mappings, vec![(15432, 5432)]);
        assert!(config.env_vars.contains_key("POSTGRES_PASSWORD"));
    }

    #[test]
    fn test_container_config_mysql() {
        let config = ContainerConfig::mysql(13306);
        assert_eq!(config.image, "mysql");
        assert_eq!(config.port_mappings, vec![(13306, 3306)]);
    }

    #[test]
    fn test_container_config_sqlserver() {
        let config = ContainerConfig::sqlserver(11433);
        assert!(config.image.contains("mssql"));
        assert!(config.env_vars.contains_key("ACCEPT_EULA"));
    }

    #[test]
    fn test_container_config_builder() {
        let config = ContainerConfig::postgres(5432)
            .with_tag("16")
            .with_env("PGDATA", "/custom")
            .with_memory_limit(1_073_741_824);
        assert_eq!(config.tag, "16");
        assert_eq!(config.memory_limit, 1_073_741_824);
        assert_eq!(config.image_ref(), "postgres:16");
    }

    #[test]
    fn test_docker_command_run() {
        let config = ContainerConfig::postgres(15432);
        let cmd = DockerCommand::run(&config);
        assert!(cmd.starts_with("docker run -d"));
        assert!(cmd.contains("-p 15432:5432"));
        assert!(cmd.contains("postgres:15"));
    }

    #[test]
    fn test_docker_command_stop() {
        let cmd = DockerCommand::stop("abc123");
        assert_eq!(cmd, "docker stop abc123");
    }

    #[test]
    fn test_docker_command_remove() {
        let cmd = DockerCommand::remove("abc123");
        assert_eq!(cmd, "docker rm -f abc123");
    }

    #[test]
    fn test_docker_manager_start() {
        let mut mgr = DockerManager::new();
        let cmd = mgr.start_command(EngineKind::PostgreSQL);
        assert!(cmd.contains("docker run"));
        assert!(mgr.container_info(&EngineKind::PostgreSQL).is_some());
    }

    #[test]
    fn test_docker_manager_port_allocation() {
        let mut mgr = DockerManager::new().with_base_port(20000);
        let _ = mgr.start_command(EngineKind::PostgreSQL);
        let _ = mgr.start_command(EngineKind::MySQL);
        let pg_port = mgr.host_port(&EngineKind::PostgreSQL).unwrap();
        let my_port = mgr.host_port(&EngineKind::MySQL).unwrap();
        assert_eq!(pg_port, 20000);
        assert_eq!(my_port, 20001);
    }

    #[test]
    fn test_docker_manager_lifecycle() {
        let mut mgr = DockerManager::new();
        mgr.start_command(EngineKind::PostgreSQL);
        mgr.set_container_id(EngineKind::PostgreSQL, "abc123");
        let info = mgr.container_info(&EngineKind::PostgreSQL).unwrap();
        assert_eq!(info.state, ContainerState::Running);

        mgr.mark_healthy(EngineKind::PostgreSQL);
        let info = mgr.container_info(&EngineKind::PostgreSQL).unwrap();
        assert!(info.is_healthy());
    }

    #[test]
    fn test_docker_manager_stop_all() {
        let mut mgr = DockerManager::new();
        mgr.start_command(EngineKind::PostgreSQL);
        mgr.set_container_id(EngineKind::PostgreSQL, "pg1");
        mgr.start_command(EngineKind::MySQL);
        mgr.set_container_id(EngineKind::MySQL, "my1");

        let cmds = mgr.stop_all_commands();
        assert_eq!(cmds.len(), 2);
    }

    #[test]
    fn test_health_checker() {
        let cmd = HealthChecker::check_command("abc", &EngineKind::PostgreSQL);
        assert!(cmd.contains("pg_isready"));
        assert!(HealthChecker::is_healthy(0));
        assert!(!HealthChecker::is_healthy(1));
    }

    #[test]
    fn test_health_check_sequence() {
        let seq = HealthChecker::check_sequence("abc", &EngineKind::MySQL, 3, 2);
        assert_eq!(seq.len(), 3);
        assert_eq!(seq[0].1, Duration::from_secs(0));
        assert_eq!(seq[1].1, Duration::from_secs(2));
    }

    #[test]
    fn test_container_state_display() {
        assert_eq!(format!("{}", ContainerState::Healthy), "healthy");
        assert_eq!(format!("{}", ContainerState::Running), "running");
        let failed = ContainerState::Failed("oom".into());
        assert!(format!("{}", failed).contains("oom"));
    }
}
