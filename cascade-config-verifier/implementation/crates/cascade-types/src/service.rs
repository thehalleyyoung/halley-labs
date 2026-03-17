use serde::{Deserialize, Serialize};

use std::collections::BTreeMap;
use std::fmt;

// ---------------------------------------------------------------------------
// ServiceId
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ServiceId(pub String);

impl ServiceId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Display for ServiceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for ServiceId {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

impl From<String> for ServiceId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl AsRef<str> for ServiceId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

// ---------------------------------------------------------------------------
// ServiceName / ServiceNamespace
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ServiceName(pub String);

impl ServiceName {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Validate that the name follows Kubernetes naming conventions:
    /// lowercase alphanumeric + hyphens, max 63 chars, starts/ends with alnum.
    pub fn validate(&self) -> Result<(), String> {
        let s = &self.0;
        if s.is_empty() {
            return Err("service name must not be empty".into());
        }
        if s.len() > 63 {
            return Err(format!("service name too long: {} > 63", s.len()));
        }
        if !s.starts_with(|c: char| c.is_ascii_lowercase() || c.is_ascii_digit()) {
            return Err("must start with lowercase letter or digit".into());
        }
        if !s.ends_with(|c: char| c.is_ascii_lowercase() || c.is_ascii_digit()) {
            return Err("must end with lowercase letter or digit".into());
        }
        for c in s.chars() {
            if !(c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-') {
                return Err(format!("invalid character '{c}' in service name"));
            }
        }
        Ok(())
    }
}

impl fmt::Display for ServiceName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for ServiceName {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ServiceNamespace(pub String);

impl ServiceNamespace {
    pub fn new(ns: impl Into<String>) -> Self {
        Self(ns.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn is_default(&self) -> bool {
        self.0 == "default"
    }
}

impl Default for ServiceNamespace {
    fn default() -> Self {
        Self("default".to_owned())
    }
}

impl fmt::Display for ServiceNamespace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for ServiceNamespace {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

// ---------------------------------------------------------------------------
// ServicePort
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ServicePort {
    pub name: Option<String>,
    pub port: u16,
    pub target_port: Option<u16>,
    pub protocol: Protocol,
}

impl ServicePort {
    pub fn new(port: u16, protocol: Protocol) -> Self {
        Self {
            name: None,
            port,
            target_port: None,
            protocol,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_target_port(mut self, tp: u16) -> Self {
        self.target_port = Some(tp);
        self
    }

    pub fn effective_target_port(&self) -> u16 {
        self.target_port.unwrap_or(self.port)
    }
}

impl fmt::Display for ServicePort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref name) = self.name {
            write!(f, "{name}:")?;
        }
        write!(f, "{}/{}", self.port, self.protocol)?;
        if let Some(tp) = self.target_port {
            write!(f, " -> {tp}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Protocol
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Protocol {
    HTTP,
    HTTPS,
    #[serde(rename = "gRPC")]
    GRPC,
    TCP,
}

impl Protocol {
    pub fn default_port(self) -> u16 {
        match self {
            Protocol::HTTP => 80,
            Protocol::HTTPS => 443,
            Protocol::GRPC => 50051,
            Protocol::TCP => 0,
        }
    }

    pub fn is_secure(self) -> bool {
        matches!(self, Protocol::HTTPS)
    }

    pub fn supports_retries(self) -> bool {
        matches!(self, Protocol::HTTP | Protocol::HTTPS | Protocol::GRPC)
    }
}

impl fmt::Display for Protocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Protocol::HTTP => write!(f, "HTTP"),
            Protocol::HTTPS => write!(f, "HTTPS"),
            Protocol::GRPC => write!(f, "gRPC"),
            Protocol::TCP => write!(f, "TCP"),
        }
    }
}

impl Default for Protocol {
    fn default() -> Self {
        Protocol::HTTP
    }
}

impl std::str::FromStr for Protocol {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "HTTP" => Ok(Protocol::HTTP),
            "HTTPS" => Ok(Protocol::HTTPS),
            "GRPC" => Ok(Protocol::GRPC),
            "TCP" => Ok(Protocol::TCP),
            _ => Err(format!("unknown protocol: {s}")),
        }
    }
}

// ---------------------------------------------------------------------------
// ServiceEndpoint
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ServiceEndpoint {
    pub host: String,
    pub port: u16,
    pub protocol: Protocol,
}

impl ServiceEndpoint {
    pub fn new(host: impl Into<String>, port: u16, protocol: Protocol) -> Self {
        Self {
            host: host.into(),
            port,
            protocol,
        }
    }

    pub fn authority(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    pub fn url(&self) -> String {
        let scheme = match self.protocol {
            Protocol::HTTP => "http",
            Protocol::HTTPS => "https",
            Protocol::GRPC => "grpc",
            Protocol::TCP => "tcp",
        };
        format!("{scheme}://{}:{}", self.host, self.port)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.host.is_empty() {
            return Err("host must not be empty".into());
        }
        if self.port == 0 {
            return Err("port must not be zero".into());
        }
        Ok(())
    }
}

impl fmt::Display for ServiceEndpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.url())
    }
}

// ---------------------------------------------------------------------------
// ServiceMetadata
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceMetadata {
    pub name: ServiceName,
    pub namespace: ServiceNamespace,
    pub labels: BTreeMap<String, String>,
    pub annotations: BTreeMap<String, String>,
    pub version: Option<String>,
    pub creation_timestamp: Option<chrono::DateTime<chrono::Utc>>,
}

impl ServiceMetadata {
    pub fn new(name: impl Into<String>, namespace: impl Into<String>) -> Self {
        Self {
            name: ServiceName::new(name),
            namespace: ServiceNamespace::new(namespace),
            labels: BTreeMap::new(),
            annotations: BTreeMap::new(),
            version: None,
            creation_timestamp: None,
        }
    }

    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    pub fn with_annotation(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.annotations.insert(key.into(), value.into());
        self
    }

    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    pub fn with_creation_timestamp(mut self, ts: chrono::DateTime<chrono::Utc>) -> Self {
        self.creation_timestamp = Some(ts);
        self
    }

    pub fn qualified_name(&self) -> String {
        format!("{}/{}", self.namespace, self.name)
    }

    pub fn has_label(&self, key: &str) -> bool {
        self.labels.contains_key(key)
    }

    pub fn label_value(&self, key: &str) -> Option<&str> {
        self.labels.get(key).map(|s| s.as_str())
    }
}

impl fmt::Display for ServiceMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.qualified_name())
    }
}

// ---------------------------------------------------------------------------
// ServiceType
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ServiceType {
    ClusterIP,
    NodePort,
    LoadBalancer,
    ExternalName,
}

impl ServiceType {
    pub fn exposes_externally(self) -> bool {
        matches!(self, ServiceType::NodePort | ServiceType::LoadBalancer)
    }

    pub fn requires_cluster_ip(self) -> bool {
        !matches!(self, ServiceType::ExternalName)
    }
}

impl Default for ServiceType {
    fn default() -> Self {
        ServiceType::ClusterIP
    }
}

impl fmt::Display for ServiceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ServiceType::ClusterIP => write!(f, "ClusterIP"),
            ServiceType::NodePort => write!(f, "NodePort"),
            ServiceType::LoadBalancer => write!(f, "LoadBalancer"),
            ServiceType::ExternalName => write!(f, "ExternalName"),
        }
    }
}

// ---------------------------------------------------------------------------
// ServiceSpec
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServiceSpec {
    pub endpoints: Vec<ServiceEndpoint>,
    pub ports: Vec<ServicePort>,
    pub selector: BTreeMap<String, String>,
    pub service_type: ServiceType,
}

impl ServiceSpec {
    pub fn new(service_type: ServiceType) -> Self {
        Self {
            endpoints: Vec::new(),
            ports: Vec::new(),
            selector: BTreeMap::new(),
            service_type,
        }
    }

    pub fn with_endpoint(mut self, ep: ServiceEndpoint) -> Self {
        self.endpoints.push(ep);
        self
    }

    pub fn with_port(mut self, port: ServicePort) -> Self {
        self.ports.push(port);
        self
    }

    pub fn with_selector(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.selector.insert(key.into(), value.into());
        self
    }

    pub fn primary_port(&self) -> Option<&ServicePort> {
        self.ports.first()
    }

    pub fn port_by_name(&self, name: &str) -> Option<&ServicePort> {
        self.ports.iter().find(|p| p.name.as_deref() == Some(name))
    }

    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.ports.is_empty() {
            errors.push("service spec must have at least one port".into());
        }
        for ep in &self.endpoints {
            if let Err(e) = ep.validate() {
                errors.push(format!("invalid endpoint: {e}"));
            }
        }
        let mut seen_ports = std::collections::HashSet::new();
        for p in &self.ports {
            if !seen_ports.insert(p.port) {
                errors.push(format!("duplicate port number: {}", p.port));
            }
        }
        if self.service_type == ServiceType::ExternalName && !self.selector.is_empty() {
            errors.push("ExternalName services must not have selectors".into());
        }
        errors
    }
}

impl Default for ServiceSpec {
    fn default() -> Self {
        Self::new(ServiceType::ClusterIP)
    }
}

// ---------------------------------------------------------------------------
// ServiceHealth
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ServiceHealth {
    Healthy,
    Degraded,
    Unavailable,
}

impl ServiceHealth {
    pub fn is_healthy(self) -> bool {
        matches!(self, ServiceHealth::Healthy)
    }

    pub fn is_available(self) -> bool {
        !matches!(self, ServiceHealth::Unavailable)
    }

    pub fn severity_score(self) -> u8 {
        match self {
            ServiceHealth::Healthy => 0,
            ServiceHealth::Degraded => 1,
            ServiceHealth::Unavailable => 2,
        }
    }

    pub fn worst(self, other: ServiceHealth) -> ServiceHealth {
        if self.severity_score() >= other.severity_score() {
            self
        } else {
            other
        }
    }
}

impl Default for ServiceHealth {
    fn default() -> Self {
        ServiceHealth::Healthy
    }
}

impl fmt::Display for ServiceHealth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ServiceHealth::Healthy => write!(f, "Healthy"),
            ServiceHealth::Degraded => write!(f, "Degraded"),
            ServiceHealth::Unavailable => write!(f, "Unavailable"),
        }
    }
}

// ---------------------------------------------------------------------------
// ServiceState
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServiceState {
    pub health: ServiceHealth,
    pub current_load: f64,
    pub capacity: f64,
    pub error_rate: f64,
    pub latency_p99: f64,
}

impl ServiceState {
    pub fn new(capacity: f64) -> Self {
        Self {
            health: ServiceHealth::Healthy,
            current_load: 0.0,
            capacity,
            error_rate: 0.0,
            latency_p99: 0.0,
        }
    }

    pub fn utilization(&self) -> f64 {
        if self.capacity <= 0.0 {
            return f64::INFINITY;
        }
        self.current_load / self.capacity
    }

    pub fn headroom(&self) -> f64 {
        (self.capacity - self.current_load).max(0.0)
    }

    pub fn is_overloaded(&self) -> bool {
        self.current_load > self.capacity
    }

    pub fn derive_health(&self) -> ServiceHealth {
        if self.is_overloaded() || self.error_rate > 0.5 {
            ServiceHealth::Unavailable
        } else if self.utilization() > 0.8 || self.error_rate > 0.1 {
            ServiceHealth::Degraded
        } else {
            ServiceHealth::Healthy
        }
    }

    pub fn apply_load(&mut self, additional: f64) {
        self.current_load += additional;
        self.health = self.derive_health();
    }

    pub fn reset(&mut self) {
        self.current_load = 0.0;
        self.error_rate = 0.0;
        self.latency_p99 = 0.0;
        self.health = ServiceHealth::Healthy;
    }
}

impl Default for ServiceState {
    fn default() -> Self {
        Self::new(100.0)
    }
}

impl fmt::Display for ServiceState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{} load={:.1}/{:.1} err={:.2}% p99={:.1}ms]",
            self.health,
            self.current_load,
            self.capacity,
            self.error_rate * 100.0,
            self.latency_p99
        )
    }
}

// ---------------------------------------------------------------------------
// Service
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Service {
    pub id: ServiceId,
    pub metadata: ServiceMetadata,
    pub spec: ServiceSpec,
    pub state: ServiceState,
}

impl Service {
    pub fn new(id: impl Into<String>, name: impl Into<String>, namespace: impl Into<String>) -> Self {
        let id_str: String = id.into();
        Self {
            id: ServiceId::new(id_str),
            metadata: ServiceMetadata::new(name, namespace),
            spec: ServiceSpec::default(),
            state: ServiceState::default(),
        }
    }

    pub fn with_spec(mut self, spec: ServiceSpec) -> Self {
        self.spec = spec;
        self
    }

    pub fn with_state(mut self, state: ServiceState) -> Self {
        self.state = state;
        self
    }

    pub fn with_capacity(mut self, capacity: f64) -> Self {
        self.state = ServiceState::new(capacity);
        self
    }

    pub fn qualified_name(&self) -> String {
        self.metadata.qualified_name()
    }

    pub fn is_healthy(&self) -> bool {
        self.state.health.is_healthy()
    }

    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.id.is_empty() {
            errors.push("service id must not be empty".into());
        }
        if let Err(e) = self.metadata.name.validate() {
            errors.push(format!("invalid service name: {e}"));
        }
        errors.extend(self.spec.validate());
        if self.state.capacity < 0.0 {
            errors.push("capacity must be non-negative".into());
        }
        errors
    }
}

impl fmt::Display for Service {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Service({} {} {})",
            self.id, self.metadata, self.state
        )
    }
}

// ---------------------------------------------------------------------------
// ServiceBuilder
// ---------------------------------------------------------------------------

pub struct ServiceBuilder {
    id: String,
    name: String,
    namespace: String,
    labels: BTreeMap<String, String>,
    annotations: BTreeMap<String, String>,
    version: Option<String>,
    endpoints: Vec<ServiceEndpoint>,
    ports: Vec<ServicePort>,
    selector: BTreeMap<String, String>,
    service_type: ServiceType,
    capacity: f64,
}

impl ServiceBuilder {
    pub fn new(id: impl Into<String>) -> Self {
        let id_str: String = id.into();
        Self {
            id: id_str.clone(),
            name: id_str,
            namespace: "default".into(),
            labels: BTreeMap::new(),
            annotations: BTreeMap::new(),
            version: None,
            endpoints: Vec::new(),
            ports: Vec::new(),
            selector: BTreeMap::new(),
            service_type: ServiceType::ClusterIP,
            capacity: 100.0,
        }
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn namespace(mut self, ns: impl Into<String>) -> Self {
        self.namespace = ns.into();
        self
    }

    pub fn label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    pub fn annotation(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.annotations.insert(key.into(), value.into());
        self
    }

    pub fn version(mut self, v: impl Into<String>) -> Self {
        self.version = Some(v.into());
        self
    }

    pub fn endpoint(mut self, ep: ServiceEndpoint) -> Self {
        self.endpoints.push(ep);
        self
    }

    pub fn port(mut self, p: ServicePort) -> Self {
        self.ports.push(p);
        self
    }

    pub fn selector(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.selector.insert(key.into(), value.into());
        self
    }

    pub fn service_type(mut self, st: ServiceType) -> Self {
        self.service_type = st;
        self
    }

    pub fn capacity(mut self, cap: f64) -> Self {
        self.capacity = cap;
        self
    }

    pub fn build(self) -> Service {
        let metadata = ServiceMetadata {
            name: ServiceName::new(self.name),
            namespace: ServiceNamespace::new(self.namespace),
            labels: self.labels,
            annotations: self.annotations,
            version: self.version,
            creation_timestamp: None,
        };
        let spec = ServiceSpec {
            endpoints: self.endpoints,
            ports: self.ports,
            selector: self.selector,
            service_type: self.service_type,
        };
        Service {
            id: ServiceId::new(self.id),
            metadata,
            spec,
            state: ServiceState::new(self.capacity),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_id_display_and_eq() {
        let a = ServiceId::new("svc-a");
        let b = ServiceId::from("svc-a");
        assert_eq!(a, b);
        assert_eq!(a.to_string(), "svc-a");
    }

    #[test]
    fn test_service_id_ord() {
        let a = ServiceId::new("alpha");
        let b = ServiceId::new("beta");
        assert!(a < b);
    }

    #[test]
    fn test_service_name_validation_ok() {
        assert!(ServiceName::new("my-service-1").validate().is_ok());
    }

    #[test]
    fn test_service_name_validation_empty() {
        assert!(ServiceName::new("").validate().is_err());
    }

    #[test]
    fn test_service_name_validation_bad_char() {
        assert!(ServiceName::new("my_Service").validate().is_err());
    }

    #[test]
    fn test_service_name_validation_too_long() {
        let long = "a".repeat(64);
        assert!(ServiceName::new(long).validate().is_err());
    }

    #[test]
    fn test_service_namespace_default() {
        let ns = ServiceNamespace::default();
        assert!(ns.is_default());
        assert_eq!(ns.to_string(), "default");
    }

    #[test]
    fn test_service_port_display() {
        let p = ServicePort::new(8080, Protocol::HTTP)
            .with_name("http")
            .with_target_port(80);
        let s = p.to_string();
        assert!(s.contains("http:") && s.contains("8080") && s.contains("80"));
    }

    #[test]
    fn test_protocol_default_port() {
        assert_eq!(Protocol::HTTP.default_port(), 80);
        assert_eq!(Protocol::HTTPS.default_port(), 443);
        assert_eq!(Protocol::GRPC.default_port(), 50051);
    }

    #[test]
    fn test_protocol_from_str() {
        assert_eq!("http".parse::<Protocol>().unwrap(), Protocol::HTTP);
        assert_eq!("GRPC".parse::<Protocol>().unwrap(), Protocol::GRPC);
        assert!("ftp".parse::<Protocol>().is_err());
    }

    #[test]
    fn test_endpoint_url_and_validate() {
        let ep = ServiceEndpoint::new("localhost", 8080, Protocol::HTTP);
        assert_eq!(ep.url(), "http://localhost:8080");
        assert!(ep.validate().is_ok());

        let bad = ServiceEndpoint::new("", 0, Protocol::TCP);
        assert!(bad.validate().is_err());
    }

    #[test]
    fn test_service_metadata_qualified_name() {
        let m = ServiceMetadata::new("gateway", "prod")
            .with_label("app", "gw")
            .with_version("v2");
        assert_eq!(m.qualified_name(), "prod/gateway");
        assert_eq!(m.label_value("app"), Some("gw"));
        assert!(m.has_label("app"));
    }

    #[test]
    fn test_service_type_external() {
        assert!(!ServiceType::ClusterIP.exposes_externally());
        assert!(ServiceType::LoadBalancer.exposes_externally());
        assert!(!ServiceType::ExternalName.requires_cluster_ip());
    }

    #[test]
    fn test_service_spec_validate() {
        let spec = ServiceSpec::new(ServiceType::ClusterIP);
        let errs = spec.validate();
        assert!(!errs.is_empty()); // no ports

        let spec2 = ServiceSpec::new(ServiceType::ClusterIP)
            .with_port(ServicePort::new(80, Protocol::HTTP));
        assert!(spec2.validate().is_empty());
    }

    #[test]
    fn test_service_health_worst() {
        assert_eq!(
            ServiceHealth::Healthy.worst(ServiceHealth::Degraded),
            ServiceHealth::Degraded
        );
        assert_eq!(
            ServiceHealth::Unavailable.worst(ServiceHealth::Healthy),
            ServiceHealth::Unavailable
        );
    }

    #[test]
    fn test_service_state_utilization() {
        let s = ServiceState {
            health: ServiceHealth::Healthy,
            current_load: 50.0,
            capacity: 100.0,
            error_rate: 0.0,
            latency_p99: 10.0,
        };
        assert!((s.utilization() - 0.5).abs() < 1e-9);
        assert!((s.headroom() - 50.0).abs() < 1e-9);
        assert!(!s.is_overloaded());
    }

    #[test]
    fn test_service_state_derive_health() {
        let mut s = ServiceState::new(100.0);
        s.current_load = 85.0;
        assert_eq!(s.derive_health(), ServiceHealth::Degraded);
        s.current_load = 110.0;
        assert_eq!(s.derive_health(), ServiceHealth::Unavailable);
    }

    #[test]
    fn test_service_state_apply_load() {
        let mut s = ServiceState::new(100.0);
        s.apply_load(120.0);
        assert!(s.is_overloaded());
        assert_eq!(s.health, ServiceHealth::Unavailable);
    }

    #[test]
    fn test_service_builder() {
        let svc = ServiceBuilder::new("gw")
            .name("gateway")
            .namespace("prod")
            .label("app", "gw")
            .version("v1")
            .port(ServicePort::new(80, Protocol::HTTP))
            .capacity(200.0)
            .build();
        assert_eq!(svc.id.as_str(), "gw");
        assert_eq!(svc.metadata.name.as_str(), "gateway");
        assert_eq!(svc.state.capacity, 200.0);
    }

    #[test]
    fn test_service_validate_ok() {
        let svc = ServiceBuilder::new("gw")
            .name("gateway")
            .port(ServicePort::new(80, Protocol::HTTP))
            .build();
        assert!(svc.validate().is_empty());
    }

    #[test]
    fn test_service_validate_empty_id() {
        let svc = ServiceBuilder::new("")
            .name("gateway")
            .port(ServicePort::new(80, Protocol::HTTP))
            .build();
        let errs = svc.validate();
        assert!(errs.iter().any(|e| e.contains("id")));
    }

    #[test]
    fn test_service_display() {
        let svc = ServiceBuilder::new("gw").name("gateway").build();
        let s = svc.to_string();
        assert!(s.contains("gw") && s.contains("default/gateway"));
    }

    #[test]
    fn test_service_serialization_roundtrip() {
        let svc = ServiceBuilder::new("svc-1")
            .name("service-1")
            .namespace("ns")
            .port(ServicePort::new(8080, Protocol::GRPC))
            .capacity(500.0)
            .build();
        let json = serde_json::to_string(&svc).unwrap();
        let back: Service = serde_json::from_str(&json).unwrap();
        assert_eq!(svc.id, back.id);
        assert_eq!(svc.metadata.name, back.metadata.name);
    }

    #[test]
    fn test_protocol_secure() {
        assert!(Protocol::HTTPS.is_secure());
        assert!(!Protocol::HTTP.is_secure());
    }

    #[test]
    fn test_effective_target_port() {
        let p = ServicePort::new(80, Protocol::HTTP);
        assert_eq!(p.effective_target_port(), 80);
        let p2 = p.with_target_port(8080);
        assert_eq!(p2.effective_target_port(), 8080);
    }
}
