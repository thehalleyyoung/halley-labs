// Service descriptor types for the SafeStep deployment planner.

use std::fmt;
use std::str::FromStr;

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use crate::error::{Result, SafeStepError};
use crate::identifiers::ServiceId;
use crate::version::{Version, VersionSet};

/// Full descriptor for a Kubernetes service managed by SafeStep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceDescriptor {
    pub id: ServiceId,
    pub name: String,
    pub namespace: String,
    pub version_set: VersionSet,
    pub current_version: Option<Version>,
    pub target_version: Option<Version>,
    pub replicas: ReplicaConfig,
    pub resources: ResourceRequirements,
    pub ports: Vec<ServicePort>,
    pub health_check: Option<HealthCheck>,
    pub labels: indexmap::IndexMap<String, String>,
    pub annotations: indexmap::IndexMap<String, String>,
    pub dependencies: Vec<ServiceId>,
    pub config: ServiceConfig,
}

impl ServiceDescriptor {
    pub fn new(name: impl Into<String>, namespace: impl Into<String>) -> Self {
        let name = name.into();
        let namespace = namespace.into();
        let id = ServiceId::new(format!("{}/{}", namespace, name));
        Self {
            id,
            name: name.clone(),
            namespace,
            version_set: VersionSet::new(&name),
            current_version: None,
            target_version: None,
            replicas: ReplicaConfig::default(),
            resources: ResourceRequirements::default(),
            ports: Vec::new(),
            health_check: None,
            labels: indexmap::IndexMap::new(),
            annotations: indexmap::IndexMap::new(),
            dependencies: Vec::new(),
            config: ServiceConfig::default(),
        }
    }

    pub fn with_versions(mut self, versions: VersionSet) -> Self {
        self.version_set = versions;
        self
    }

    pub fn with_current_version(mut self, version: Version) -> Self {
        self.current_version = Some(version);
        self
    }

    pub fn with_target_version(mut self, version: Version) -> Self {
        self.target_version = Some(version);
        self
    }

    pub fn with_replicas(mut self, replicas: ReplicaConfig) -> Self {
        self.replicas = replicas;
        self
    }

    pub fn with_resources(mut self, resources: ResourceRequirements) -> Self {
        self.resources = resources;
        self
    }

    pub fn with_port(mut self, port: ServicePort) -> Self {
        self.ports.push(port);
        self
    }

    pub fn with_health_check(mut self, hc: HealthCheck) -> Self {
        self.health_check = Some(hc);
        self
    }

    pub fn with_dependency(mut self, dep: ServiceId) -> Self {
        self.dependencies.push(dep);
        self
    }

    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    pub fn with_config(mut self, config: ServiceConfig) -> Self {
        self.config = config;
        self
    }

    pub fn qualified_name(&self) -> String {
        format!("{}/{}", self.namespace, self.name)
    }

    pub fn needs_upgrade(&self) -> bool {
        match (&self.current_version, &self.target_version) {
            (Some(curr), Some(target)) => curr != target,
            _ => false,
        }
    }

    /// Total resource requirements accounting for replica count.
    pub fn total_resources(&self) -> ResourceRequirements {
        let count = self.replicas.desired as f64;
        ResourceRequirements {
            cpu_request: self.resources.cpu_request.map(|v| ResourceQuantity {
                millicores: (v.millicores as f64 * count) as u64,
            }),
            cpu_limit: self.resources.cpu_limit.map(|v| ResourceQuantity {
                millicores: (v.millicores as f64 * count) as u64,
            }),
            memory_request: self.resources.memory_request.map(|v| ResourceQuantity {
                millicores: (v.millicores as f64 * count) as u64,
            }),
            memory_limit: self.resources.memory_limit.map(|v| ResourceQuantity {
                millicores: (v.millicores as f64 * count) as u64,
            }),
            storage: self.resources.storage.map(|v| ResourceQuantity {
                millicores: (v.millicores as f64 * count) as u64,
            }),
            custom: self
                .resources
                .custom
                .iter()
                .map(|(k, v)| {
                    (
                        k.clone(),
                        ResourceQuantity {
                            millicores: (v.millicores as f64 * count) as u64,
                        },
                    )
                })
                .collect(),
        }
    }

    /// Validate this descriptor for internal consistency.
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(SafeStepError::config("service name cannot be empty"));
        }
        if self.namespace.is_empty() {
            return Err(SafeStepError::config("service namespace cannot be empty"));
        }
        self.replicas.validate()?;
        if let (Some(req), Some(lim)) = (&self.resources.cpu_request, &self.resources.cpu_limit) {
            if req.millicores > lim.millicores {
                return Err(SafeStepError::config(
                    "CPU request cannot exceed CPU limit",
                ));
            }
        }
        if let (Some(req), Some(lim)) = (
            &self.resources.memory_request,
            &self.resources.memory_limit,
        ) {
            if req.millicores > lim.millicores {
                return Err(SafeStepError::config(
                    "memory request cannot exceed memory limit",
                ));
            }
        }
        Ok(())
    }
}

impl fmt::Display for ServiceDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.namespace, self.name)?;
        if let Some(v) = &self.current_version {
            write!(f, " (v{})", v)?;
        }
        if let Some(t) = &self.target_version {
            write!(f, " -> v{}", t)?;
        }
        Ok(())
    }
}

// ─── ReplicaConfig ──────────────────────────────────────────────────────

/// Replica configuration for a service.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ReplicaConfig {
    pub desired: u32,
    pub min_available: u32,
    pub max_surge: u32,
    pub max_unavailable: u32,
}

impl ReplicaConfig {
    pub fn new(desired: u32) -> Self {
        Self {
            desired,
            min_available: desired.saturating_sub(1).max(1),
            max_surge: 1,
            max_unavailable: 1,
        }
    }

    pub fn with_min_available(mut self, min: u32) -> Self {
        self.min_available = min;
        self
    }

    pub fn with_max_surge(mut self, max_surge: u32) -> Self {
        self.max_surge = max_surge;
        self
    }

    pub fn with_max_unavailable(mut self, max_unavail: u32) -> Self {
        self.max_unavailable = max_unavail;
        self
    }

    /// Maximum total replicas during a rolling update.
    pub fn max_total(&self) -> u32 {
        self.desired + self.max_surge
    }

    /// Minimum healthy replicas at any point during update.
    pub fn min_healthy(&self) -> u32 {
        self.desired.saturating_sub(self.max_unavailable)
    }

    /// During a canary deployment with (old_count, new_count), check validity.
    pub fn is_valid_canary_split(&self, old_count: u32, new_count: u32) -> bool {
        let total = old_count + new_count;
        total <= self.max_total() && total >= self.min_healthy()
    }

    /// Collapse replica state to (old_count, new_count) pairs — replica symmetry.
    pub fn symmetric_configurations(&self) -> Vec<(u32, u32)> {
        let max = self.max_total();
        let min = self.min_healthy();
        let mut configs = Vec::new();
        for total in min..=max {
            for new_count in 0..=total {
                let old_count = total - new_count;
                if old_count <= self.desired && new_count <= self.desired {
                    configs.push((old_count, new_count));
                }
            }
        }
        configs
    }

    pub fn validate(&self) -> Result<()> {
        if self.desired == 0 {
            return Err(SafeStepError::config("desired replicas cannot be 0"));
        }
        if self.min_available > self.desired {
            return Err(SafeStepError::config(
                "min_available cannot exceed desired replicas",
            ));
        }
        Ok(())
    }
}

impl Default for ReplicaConfig {
    fn default() -> Self {
        Self::new(1)
    }
}

impl fmt::Display for ReplicaConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "replicas={} (min_avail={}, max_surge={}, max_unavail={})",
            self.desired, self.min_available, self.max_surge, self.max_unavailable
        )
    }
}

// ─── ResourceQuantity ────────────────────────────────────────────────────

/// A parsed Kubernetes resource quantity. Internally stored as milliunits.
/// For CPU: millicores (1000m = 1 CPU). For memory: bytes stored in millicores field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ResourceQuantity {
    /// Milliunit value. For CPU, these are millicores. For memory, these are bytes.
    pub millicores: u64,
}

impl ResourceQuantity {
    pub fn from_millicores(m: u64) -> Self {
        Self { millicores: m }
    }

    pub fn from_cores(c: f64) -> Self {
        Self {
            millicores: (c * 1000.0) as u64,
        }
    }

    pub fn from_bytes(b: u64) -> Self {
        Self { millicores: b }
    }

    pub fn zero() -> Self {
        Self { millicores: 0 }
    }

    pub fn is_zero(&self) -> bool {
        self.millicores == 0
    }

    /// Parse a Kubernetes resource quantity string.
    /// CPU: "500m", "1", "0.5", "2.5"
    /// Memory: "128Mi", "1Gi", "500M", "1G", "1024Ki", "1048576"
    pub fn parse(input: &str) -> Result<Self> {
        let input = input.trim();
        if input.is_empty() {
            return Err(SafeStepError::version_parse(
                "empty resource quantity",
                input,
            ));
        }

        // CPU millicores
        if let Some(rest) = input.strip_suffix('m') {
            let val: u64 = rest
                .parse()
                .map_err(|_| SafeStepError::version_parse("invalid millicores", input))?;
            return Ok(Self::from_millicores(val));
        }

        // Binary memory units
        if let Some(rest) = input.strip_suffix("Gi") {
            let val: f64 = rest
                .parse()
                .map_err(|_| SafeStepError::version_parse("invalid Gi value", input))?;
            return Ok(Self::from_bytes((val * 1024.0 * 1024.0 * 1024.0) as u64));
        }
        if let Some(rest) = input.strip_suffix("Mi") {
            let val: f64 = rest
                .parse()
                .map_err(|_| SafeStepError::version_parse("invalid Mi value", input))?;
            return Ok(Self::from_bytes((val * 1024.0 * 1024.0) as u64));
        }
        if let Some(rest) = input.strip_suffix("Ki") {
            let val: f64 = rest
                .parse()
                .map_err(|_| SafeStepError::version_parse("invalid Ki value", input))?;
            return Ok(Self::from_bytes((val * 1024.0) as u64));
        }
        if let Some(rest) = input.strip_suffix("Ti") {
            let val: f64 = rest
                .parse()
                .map_err(|_| SafeStepError::version_parse("invalid Ti value", input))?;
            return Ok(Self::from_bytes(
                (val * 1024.0 * 1024.0 * 1024.0 * 1024.0) as u64,
            ));
        }

        // SI memory units
        if let Some(rest) = input.strip_suffix('G') {
            let val: f64 = rest
                .parse()
                .map_err(|_| SafeStepError::version_parse("invalid G value", input))?;
            return Ok(Self::from_bytes((val * 1_000_000_000.0) as u64));
        }
        if let Some(rest) = input.strip_suffix('M') {
            let val: f64 = rest
                .parse()
                .map_err(|_| SafeStepError::version_parse("invalid M value", input))?;
            return Ok(Self::from_bytes((val * 1_000_000.0) as u64));
        }
        if let Some(rest) = input.strip_suffix('K') {
            let val: f64 = rest
                .parse()
                .map_err(|_| SafeStepError::version_parse("invalid K value", input))?;
            return Ok(Self::from_bytes((val * 1_000.0) as u64));
        }
        if let Some(rest) = input.strip_suffix('T') {
            let val: f64 = rest
                .parse()
                .map_err(|_| SafeStepError::version_parse("invalid T value", input))?;
            return Ok(Self::from_bytes((val * 1_000_000_000_000.0) as u64));
        }

        // Plain number: could be CPU cores or bytes
        if let Ok(val) = input.parse::<f64>() {
            if val.fract() != 0.0 || input.contains('.') {
                return Ok(Self::from_cores(val));
            }
            return Ok(Self { millicores: val as u64 });
        }

        Err(SafeStepError::version_parse(
            format!("unrecognized resource quantity format: {:?}", input),
            input,
        ))
    }

    pub fn as_cores(&self) -> f64 {
        self.millicores as f64 / 1000.0
    }

    pub fn as_gi(&self) -> f64 {
        self.millicores as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    pub fn as_mi(&self) -> f64 {
        self.millicores as f64 / (1024.0 * 1024.0)
    }

    pub fn saturating_add(&self, other: &Self) -> Self {
        Self {
            millicores: self.millicores.saturating_add(other.millicores),
        }
    }

    pub fn saturating_sub(&self, other: &Self) -> Self {
        Self {
            millicores: self.millicores.saturating_sub(other.millicores),
        }
    }

    pub fn checked_mul(&self, factor: u64) -> Option<Self> {
        self.millicores
            .checked_mul(factor)
            .map(|m| Self { millicores: m })
    }
}

impl fmt::Display for ResourceQuantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.millicores)
    }
}

impl FromStr for ResourceQuantity {
    type Err = SafeStepError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        ResourceQuantity::parse(s)
    }
}

// ─── ResourceRequirements ────────────────────────────────────────────────

/// Resource requirements for a single replica of a service.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub cpu_request: Option<ResourceQuantity>,
    pub cpu_limit: Option<ResourceQuantity>,
    pub memory_request: Option<ResourceQuantity>,
    pub memory_limit: Option<ResourceQuantity>,
    pub storage: Option<ResourceQuantity>,
    pub custom: indexmap::IndexMap<String, ResourceQuantity>,
}

impl ResourceRequirements {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_cpu(mut self, request: ResourceQuantity, limit: ResourceQuantity) -> Self {
        self.cpu_request = Some(request);
        self.cpu_limit = Some(limit);
        self
    }

    pub fn with_memory(mut self, request: ResourceQuantity, limit: ResourceQuantity) -> Self {
        self.memory_request = Some(request);
        self.memory_limit = Some(limit);
        self
    }

    pub fn with_storage(mut self, storage: ResourceQuantity) -> Self {
        self.storage = Some(storage);
        self
    }

    pub fn with_custom(
        mut self,
        name: impl Into<String>,
        quantity: ResourceQuantity,
    ) -> Self {
        self.custom.insert(name.into(), quantity);
        self
    }

    /// Check if this requirement fits within the given limits.
    pub fn fits_within(&self, limits: &ResourceRequirements) -> bool {
        let check = |req: &Option<ResourceQuantity>, lim: &Option<ResourceQuantity>| -> bool {
            match (req, lim) {
                (Some(r), Some(l)) => r.millicores <= l.millicores,
                (Some(_), None) => false,
                _ => true,
            }
        };
        check(&self.cpu_request, &limits.cpu_limit)
            && check(&self.memory_request, &limits.memory_limit)
            && check(&self.storage, &limits.storage)
    }

    /// Sum two resource requirements.
    pub fn add(&self, other: &ResourceRequirements) -> ResourceRequirements {
        let add_opt = |a: &Option<ResourceQuantity>, b: &Option<ResourceQuantity>| -> Option<ResourceQuantity> {
            match (a, b) {
                (Some(x), Some(y)) => Some(x.saturating_add(y)),
                (Some(x), None) | (None, Some(x)) => Some(*x),
                (None, None) => None,
            }
        };
        let mut custom = self.custom.clone();
        for (k, v) in &other.custom {
            let entry = custom.entry(k.clone()).or_insert(ResourceQuantity::zero());
            *entry = entry.saturating_add(v);
        }
        ResourceRequirements {
            cpu_request: add_opt(&self.cpu_request, &other.cpu_request),
            cpu_limit: add_opt(&self.cpu_limit, &other.cpu_limit),
            memory_request: add_opt(&self.memory_request, &other.memory_request),
            memory_limit: add_opt(&self.memory_limit, &other.memory_limit),
            storage: add_opt(&self.storage, &other.storage),
            custom,
        }
    }

    /// Utilization ratio against limits (0.0 to 1.0+).
    pub fn utilization(&self) -> OrderedFloat<f64> {
        let ratios = [
            self.cpu_request
                .as_ref()
                .and_then(|r| self.cpu_limit.as_ref().map(|l| r.millicores as f64 / l.millicores.max(1) as f64)),
            self.memory_request
                .as_ref()
                .and_then(|r| self.memory_limit.as_ref().map(|l| r.millicores as f64 / l.millicores.max(1) as f64)),
        ];
        let valid: Vec<f64> = ratios.iter().filter_map(|r| *r).collect();
        if valid.is_empty() {
            OrderedFloat(0.0)
        } else {
            OrderedFloat(valid.iter().sum::<f64>() / valid.len() as f64)
        }
    }
}

impl fmt::Display for ResourceRequirements {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if let Some(cpu) = &self.cpu_request {
            parts.push(format!("cpu_req={}", cpu));
        }
        if let Some(cpu) = &self.cpu_limit {
            parts.push(format!("cpu_lim={}", cpu));
        }
        if let Some(mem) = &self.memory_request {
            parts.push(format!("mem_req={}", mem));
        }
        if let Some(mem) = &self.memory_limit {
            parts.push(format!("mem_lim={}", mem));
        }
        if parts.is_empty() {
            write!(f, "<no resources>")
        } else {
            write!(f, "{}", parts.join(", "))
        }
    }
}

// ─── ServicePort ─────────────────────────────────────────────────────────

/// Port definition for a service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServicePort {
    pub name: String,
    pub port: u16,
    pub target_port: u16,
    pub protocol: PortProtocol,
}

impl ServicePort {
    pub fn new(name: impl Into<String>, port: u16) -> Self {
        Self {
            name: name.into(),
            port,
            target_port: port,
            protocol: PortProtocol::TCP,
        }
    }

    pub fn with_target_port(mut self, target: u16) -> Self {
        self.target_port = target;
        self
    }

    pub fn with_protocol(mut self, protocol: PortProtocol) -> Self {
        self.protocol = protocol;
        self
    }
}

impl fmt::Display for ServicePort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{}->{}/{}",
            self.name, self.port, self.target_port, self.protocol
        )
    }
}

/// Network protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PortProtocol {
    TCP,
    UDP,
    SCTP,
}

impl fmt::Display for PortProtocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TCP => write!(f, "TCP"),
            Self::UDP => write!(f, "UDP"),
            Self::SCTP => write!(f, "SCTP"),
        }
    }
}

// ─── HealthCheck ─────────────────────────────────────────────────────────

/// Health check (probe) configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    pub liveness: Option<Probe>,
    pub readiness: Option<Probe>,
    pub startup: Option<Probe>,
}

impl HealthCheck {
    pub fn new() -> Self {
        Self {
            liveness: None,
            readiness: None,
            startup: None,
        }
    }

    pub fn with_liveness(mut self, probe: Probe) -> Self {
        self.liveness = Some(probe);
        self
    }

    pub fn with_readiness(mut self, probe: Probe) -> Self {
        self.readiness = Some(probe);
        self
    }

    pub fn with_startup(mut self, probe: Probe) -> Self {
        self.startup = Some(probe);
        self
    }

    /// Estimate maximum time for a pod to become ready.
    pub fn max_ready_time_secs(&self) -> u32 {
        let startup_time = self
            .startup
            .as_ref()
            .map(|p| p.initial_delay_secs + p.failure_threshold * p.period_secs)
            .unwrap_or(0);
        let readiness_time = self
            .readiness
            .as_ref()
            .map(|p| p.initial_delay_secs + p.failure_threshold * p.period_secs)
            .unwrap_or(30);
        startup_time + readiness_time
    }
}

impl Default for HealthCheck {
    fn default() -> Self {
        Self::new()
    }
}

/// A single probe definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Probe {
    pub action: ProbeAction,
    pub initial_delay_secs: u32,
    pub period_secs: u32,
    pub timeout_secs: u32,
    pub success_threshold: u32,
    pub failure_threshold: u32,
}

impl Probe {
    pub fn http(path: impl Into<String>, port: u16) -> Self {
        Self {
            action: ProbeAction::HttpGet {
                path: path.into(),
                port,
            },
            initial_delay_secs: 0,
            period_secs: 10,
            timeout_secs: 1,
            success_threshold: 1,
            failure_threshold: 3,
        }
    }

    pub fn tcp(port: u16) -> Self {
        Self {
            action: ProbeAction::TcpSocket { port },
            initial_delay_secs: 0,
            period_secs: 10,
            timeout_secs: 1,
            success_threshold: 1,
            failure_threshold: 3,
        }
    }

    pub fn exec(command: Vec<String>) -> Self {
        Self {
            action: ProbeAction::Exec { command },
            initial_delay_secs: 0,
            period_secs: 10,
            timeout_secs: 1,
            success_threshold: 1,
            failure_threshold: 3,
        }
    }

    pub fn with_initial_delay(mut self, secs: u32) -> Self {
        self.initial_delay_secs = secs;
        self
    }

    pub fn with_period(mut self, secs: u32) -> Self {
        self.period_secs = secs;
        self
    }

    pub fn with_timeout(mut self, secs: u32) -> Self {
        self.timeout_secs = secs;
        self
    }

    pub fn with_failure_threshold(mut self, threshold: u32) -> Self {
        self.failure_threshold = threshold;
        self
    }
}

/// Action for a health check probe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProbeAction {
    HttpGet { path: String, port: u16 },
    TcpSocket { port: u16 },
    Exec { command: Vec<String> },
}

// ─── ServiceConfig ───────────────────────────────────────────────────────

/// Runtime configuration for a service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    pub rollout_strategy: RolloutStrategy,
    pub rollback_enabled: bool,
    pub canary_weight: Option<f64>,
    pub max_rollout_duration_secs: u64,
    pub env_vars: indexmap::IndexMap<String, String>,
    pub config_maps: Vec<String>,
    pub secrets: Vec<String>,
    pub image_pull_policy: ImagePullPolicy,
    pub priority_class: Option<String>,
    pub tolerations: Vec<Toleration>,
}

impl ServiceConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_strategy(mut self, strategy: RolloutStrategy) -> Self {
        self.rollout_strategy = strategy;
        self
    }

    pub fn with_rollback(mut self, enabled: bool) -> Self {
        self.rollback_enabled = enabled;
        self
    }

    pub fn with_canary_weight(mut self, weight: f64) -> Self {
        self.canary_weight = Some(weight);
        self
    }

    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env_vars.insert(key.into(), value.into());
        self
    }

    pub fn with_config_map(mut self, name: impl Into<String>) -> Self {
        self.config_maps.push(name.into());
        self
    }

    pub fn with_max_duration(mut self, secs: u64) -> Self {
        self.max_rollout_duration_secs = secs;
        self
    }
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            rollout_strategy: RolloutStrategy::RollingUpdate,
            rollback_enabled: true,
            canary_weight: None,
            max_rollout_duration_secs: 600,
            env_vars: indexmap::IndexMap::new(),
            config_maps: Vec::new(),
            secrets: Vec::new(),
            image_pull_policy: ImagePullPolicy::IfNotPresent,
            priority_class: None,
            tolerations: Vec::new(),
        }
    }
}

/// Rollout strategy for a service.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RolloutStrategy {
    RollingUpdate,
    Recreate,
    Canary,
    BlueGreen,
}

impl fmt::Display for RolloutStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RollingUpdate => write!(f, "RollingUpdate"),
            Self::Recreate => write!(f, "Recreate"),
            Self::Canary => write!(f, "Canary"),
            Self::BlueGreen => write!(f, "BlueGreen"),
        }
    }
}

/// Image pull policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ImagePullPolicy {
    Always,
    IfNotPresent,
    Never,
}

impl fmt::Display for ImagePullPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Always => write!(f, "Always"),
            Self::IfNotPresent => write!(f, "IfNotPresent"),
            Self::Never => write!(f, "Never"),
        }
    }
}

/// Toleration for pod scheduling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Toleration {
    pub key: String,
    pub operator: TolerationOperator,
    pub value: Option<String>,
    pub effect: Option<TaintEffect>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TolerationOperator {
    Exists,
    Equal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaintEffect {
    NoSchedule,
    PreferNoSchedule,
    NoExecute,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_descriptor_new() {
        let svc = ServiceDescriptor::new("api-server", "production");
        assert_eq!(svc.name, "api-server");
        assert_eq!(svc.namespace, "production");
        assert_eq!(svc.qualified_name(), "production/api-server");
    }

    #[test]
    fn test_service_needs_upgrade() {
        let svc = ServiceDescriptor::new("svc", "ns")
            .with_current_version(Version::new(1, 0, 0))
            .with_target_version(Version::new(2, 0, 0));
        assert!(svc.needs_upgrade());

        let svc2 = ServiceDescriptor::new("svc", "ns")
            .with_current_version(Version::new(1, 0, 0))
            .with_target_version(Version::new(1, 0, 0));
        assert!(!svc2.needs_upgrade());
    }

    #[test]
    fn test_service_validate_ok() {
        let svc = ServiceDescriptor::new("svc", "ns");
        assert!(svc.validate().is_ok());
    }

    #[test]
    fn test_service_validate_empty_name() {
        let svc = ServiceDescriptor::new("", "ns");
        assert!(svc.validate().is_err());
    }

    #[test]
    fn test_replica_config() {
        let rc = ReplicaConfig::new(3)
            .with_max_surge(2)
            .with_max_unavailable(1);
        assert_eq!(rc.max_total(), 5);
        assert_eq!(rc.min_healthy(), 2);
    }

    #[test]
    fn test_replica_symmetric_configurations() {
        let rc = ReplicaConfig::new(3)
            .with_max_surge(1)
            .with_max_unavailable(1);
        let configs = rc.symmetric_configurations();
        assert!(!configs.is_empty());
        for (old, new) in &configs {
            assert!(*old + *new >= rc.min_healthy());
            assert!(*old + *new <= rc.max_total());
        }
    }

    #[test]
    fn test_replica_canary_split() {
        let rc = ReplicaConfig::new(3)
            .with_max_surge(1)
            .with_max_unavailable(1);
        assert!(rc.is_valid_canary_split(2, 1));
        assert!(rc.is_valid_canary_split(3, 1));
        assert!(!rc.is_valid_canary_split(3, 2)); // 5 > max_total=4
    }

    #[test]
    fn test_replica_validate() {
        assert!(ReplicaConfig::new(3).validate().is_ok());
        let rc = ReplicaConfig {
            desired: 0,
            min_available: 0,
            max_surge: 1,
            max_unavailable: 1,
        };
        assert!(rc.validate().is_err());
    }

    #[test]
    fn test_resource_quantity_parse_millicores() {
        let q = ResourceQuantity::parse("500m").unwrap();
        assert_eq!(q.millicores, 500);
        assert!((q.as_cores() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_resource_quantity_parse_gi() {
        let q = ResourceQuantity::parse("2Gi").unwrap();
        assert_eq!(q.millicores, 2 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_resource_quantity_parse_mi() {
        let q = ResourceQuantity::parse("256Mi").unwrap();
        assert_eq!(q.millicores, 256 * 1024 * 1024);
    }

    #[test]
    fn test_resource_quantity_parse_si() {
        let q = ResourceQuantity::parse("1G").unwrap();
        assert_eq!(q.millicores, 1_000_000_000);
    }

    #[test]
    fn test_resource_quantity_parse_float_cpu() {
        let q = ResourceQuantity::parse("0.5").unwrap();
        assert_eq!(q.millicores, 500);
    }

    #[test]
    fn test_resource_quantity_arithmetic() {
        let a = ResourceQuantity::from_millicores(100);
        let b = ResourceQuantity::from_millicores(200);
        assert_eq!(a.saturating_add(&b).millicores, 300);
        assert_eq!(b.saturating_sub(&a).millicores, 100);
        assert_eq!(a.saturating_sub(&b).millicores, 0);
    }

    #[test]
    fn test_resource_requirements_fits() {
        let req = ResourceRequirements::new()
            .with_cpu(
                ResourceQuantity::from_millicores(250),
                ResourceQuantity::from_millicores(500),
            );
        let limits = ResourceRequirements::new()
            .with_cpu(
                ResourceQuantity::from_millicores(1000),
                ResourceQuantity::from_millicores(1000),
            );
        assert!(req.fits_within(&limits));
    }

    #[test]
    fn test_resource_requirements_add() {
        let a = ResourceRequirements::new()
            .with_cpu(
                ResourceQuantity::from_millicores(100),
                ResourceQuantity::from_millicores(200),
            );
        let b = ResourceRequirements::new()
            .with_cpu(
                ResourceQuantity::from_millicores(150),
                ResourceQuantity::from_millicores(300),
            );
        let sum = a.add(&b);
        assert_eq!(sum.cpu_request.unwrap().millicores, 250);
        assert_eq!(sum.cpu_limit.unwrap().millicores, 500);
    }

    #[test]
    fn test_service_port() {
        let port = ServicePort::new("http", 80)
            .with_target_port(8080)
            .with_protocol(PortProtocol::TCP);
        assert_eq!(port.port, 80);
        assert_eq!(port.target_port, 8080);
        let s = port.to_string();
        assert!(s.contains("80"));
        assert!(s.contains("8080"));
    }

    #[test]
    fn test_health_check() {
        let hc = HealthCheck::new()
            .with_liveness(Probe::http("/healthz", 8080).with_period(15))
            .with_readiness(
                Probe::http("/ready", 8080)
                    .with_initial_delay(5)
                    .with_period(10),
            );
        assert!(hc.liveness.is_some());
        assert!(hc.readiness.is_some());
        assert!(hc.max_ready_time_secs() > 0);
    }

    #[test]
    fn test_service_config_default() {
        let cfg = ServiceConfig::default();
        assert_eq!(cfg.rollout_strategy, RolloutStrategy::RollingUpdate);
        assert!(cfg.rollback_enabled);
    }

    #[test]
    fn test_service_config_builder() {
        let cfg = ServiceConfig::new()
            .with_strategy(RolloutStrategy::Canary)
            .with_canary_weight(0.1)
            .with_env("VERSION", "v2")
            .with_max_duration(300);
        assert_eq!(cfg.rollout_strategy, RolloutStrategy::Canary);
        assert_eq!(cfg.canary_weight, Some(0.1));
        assert_eq!(cfg.env_vars.get("VERSION").unwrap(), "v2");
    }

    #[test]
    fn test_service_descriptor_display() {
        let svc = ServiceDescriptor::new("api", "prod")
            .with_current_version(Version::new(1, 2, 0))
            .with_target_version(Version::new(2, 0, 0));
        let s = svc.to_string();
        assert!(s.contains("prod/api"));
        assert!(s.contains("1.2.0"));
        assert!(s.contains("2.0.0"));
    }

    #[test]
    fn test_rollout_strategy_display() {
        assert_eq!(RolloutStrategy::RollingUpdate.to_string(), "RollingUpdate");
        assert_eq!(RolloutStrategy::Canary.to_string(), "Canary");
    }

    #[test]
    fn test_probe_constructors() {
        let http = Probe::http("/health", 8080);
        assert!(matches!(http.action, ProbeAction::HttpGet { .. }));

        let tcp = Probe::tcp(5432);
        assert!(matches!(tcp.action, ProbeAction::TcpSocket { port: 5432 }));

        let exec = Probe::exec(vec!["cat".into(), "/tmp/healthy".into()]);
        assert!(matches!(exec.action, ProbeAction::Exec { .. }));
    }

    #[test]
    fn test_resource_quantity_zero() {
        let z = ResourceQuantity::zero();
        assert!(z.is_zero());
        assert_eq!(z.millicores, 0);
    }

    #[test]
    fn test_resource_quantity_from_str() {
        let q: ResourceQuantity = "500m".parse().unwrap();
        assert_eq!(q.millicores, 500);
    }

    #[test]
    fn test_total_resources() {
        let svc = ServiceDescriptor::new("svc", "ns")
            .with_replicas(ReplicaConfig::new(3))
            .with_resources(
                ResourceRequirements::new().with_cpu(
                    ResourceQuantity::from_millicores(100),
                    ResourceQuantity::from_millicores(200),
                ),
            );
        let total = svc.total_resources();
        assert_eq!(total.cpu_request.unwrap().millicores, 300);
        assert_eq!(total.cpu_limit.unwrap().millicores, 600);
    }

    #[test]
    fn test_service_descriptor_serialization() {
        let svc = ServiceDescriptor::new("api", "default");
        let json = serde_json::to_string(&svc).unwrap();
        let parsed: ServiceDescriptor = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.name, "api");
    }
}
