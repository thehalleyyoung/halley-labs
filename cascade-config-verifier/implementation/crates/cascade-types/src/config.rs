use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Kubernetes types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KubernetesConfig {
    pub api_version: String,
    pub kind: String,
    pub metadata: KubeMetadata,
    pub deployment: Option<DeploymentSpec>,
    pub service: Option<KubeServiceSpec>,
    pub ingress: Option<IngressSpec>,
}

impl KubernetesConfig {
    pub fn builder() -> KubernetesConfigBuilder {
        KubernetesConfigBuilder::default()
    }

    pub fn validate(&self) -> Vec<ConfigValidationError> {
        let mut errors = Vec::new();
        if self.metadata.name.is_empty() {
            errors.push(ConfigValidationError {
                file: String::new(),
                source_type: "Kubernetes".into(),
                message: "metadata.name must not be empty".into(),
            });
        }
        if self.api_version.is_empty() {
            errors.push(ConfigValidationError {
                file: String::new(),
                source_type: "Kubernetes".into(),
                message: "api_version must not be empty".into(),
            });
        }
        if let Some(ref dep) = self.deployment {
            if dep.replicas == 0 {
                errors.push(ConfigValidationError {
                    file: String::new(),
                    source_type: "Kubernetes".into(),
                    message: "deployment.replicas must be > 0".into(),
                });
            }
            if dep.container_image.is_empty() {
                errors.push(ConfigValidationError {
                    file: String::new(),
                    source_type: "Kubernetes".into(),
                    message: "deployment.container_image must not be empty".into(),
                });
            }
        }
        errors
    }
}

#[derive(Debug, Clone, Default)]
pub struct KubernetesConfigBuilder {
    api_version: String,
    kind: String,
    metadata: Option<KubeMetadata>,
    deployment: Option<DeploymentSpec>,
    service: Option<KubeServiceSpec>,
    ingress: Option<IngressSpec>,
}

impl KubernetesConfigBuilder {
    pub fn api_version(mut self, v: impl Into<String>) -> Self {
        self.api_version = v.into();
        self
    }
    pub fn kind(mut self, v: impl Into<String>) -> Self {
        self.kind = v.into();
        self
    }
    pub fn metadata(mut self, v: KubeMetadata) -> Self {
        self.metadata = Some(v);
        self
    }
    pub fn deployment(mut self, v: DeploymentSpec) -> Self {
        self.deployment = Some(v);
        self
    }
    pub fn service(mut self, v: KubeServiceSpec) -> Self {
        self.service = Some(v);
        self
    }
    pub fn ingress(mut self, v: IngressSpec) -> Self {
        self.ingress = Some(v);
        self
    }
    pub fn build(self) -> KubernetesConfig {
        KubernetesConfig {
            api_version: self.api_version,
            kind: self.kind,
            metadata: self.metadata.unwrap_or_else(|| KubeMetadata {
                name: String::new(),
                namespace: None,
                labels: BTreeMap::new(),
                annotations: BTreeMap::new(),
            }),
            deployment: self.deployment,
            service: self.service,
            ingress: self.ingress,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KubeMetadata {
    pub name: String,
    pub namespace: Option<String>,
    pub labels: BTreeMap<String, String>,
    pub annotations: BTreeMap<String, String>,
}

impl KubeMetadata {
    pub fn qualified_name(&self) -> String {
        match &self.namespace {
            Some(ns) => format!("{}/{}", ns, self.name),
            None => self.name.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DeploymentSpec {
    pub replicas: u32,
    pub selector: BTreeMap<String, String>,
    pub container_image: String,
    pub resource_limits: Option<ResourceLimits>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub cpu: Option<String>,
    pub memory: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KubeServiceSpec {
    pub service_type: String,
    pub ports: Vec<KubePortSpec>,
    pub selector: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KubePortSpec {
    pub name: Option<String>,
    pub port: u16,
    pub target_port: u16,
    pub protocol: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IngressSpec {
    pub rules: Vec<IngressRule>,
    pub tls: Vec<IngressTls>,
}

impl Default for IngressSpec {
    fn default() -> Self {
        Self {
            rules: Vec::new(),
            tls: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IngressRule {
    pub host: String,
    pub paths: Vec<IngressPath>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IngressPath {
    pub path: String,
    pub backend_service: String,
    pub backend_port: u16,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IngressTls {
    pub hosts: Vec<String>,
    pub secret_name: String,
}

// ---------------------------------------------------------------------------
// Istio types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IstioConfig {
    pub virtual_services: Vec<VirtualService>,
    pub destination_rules: Vec<DestinationRule>,
    pub gateways: Vec<IstioGateway>,
}

impl IstioConfig {
    pub fn builder() -> IstioConfigBuilder {
        IstioConfigBuilder::default()
    }

    pub fn validate(&self) -> Vec<ConfigValidationError> {
        let mut errors = Vec::new();
        for vs in &self.virtual_services {
            if vs.name.is_empty() {
                errors.push(ConfigValidationError {
                    file: String::new(),
                    source_type: "Istio".into(),
                    message: "virtual_service.name must not be empty".into(),
                });
            }
            if vs.hosts.is_empty() {
                errors.push(ConfigValidationError {
                    file: String::new(),
                    source_type: "Istio".into(),
                    message: format!(
                        "virtual_service '{}' must have at least one host",
                        vs.name
                    ),
                });
            }
            for route in &vs.http_routes {
                let total_weight: u32 = route.destinations.iter().map(|d| d.weight).sum();
                if !route.destinations.is_empty() && total_weight != 100 {
                    errors.push(ConfigValidationError {
                        file: String::new(),
                        source_type: "Istio".into(),
                        message: format!(
                            "route destination weights must sum to 100, got {}",
                            total_weight
                        ),
                    });
                }
            }
        }
        for dr in &self.destination_rules {
            if dr.host.is_empty() {
                errors.push(ConfigValidationError {
                    file: String::new(),
                    source_type: "Istio".into(),
                    message: format!(
                        "destination_rule '{}' must have a host",
                        dr.name
                    ),
                });
            }
        }
        errors
    }
}

#[derive(Debug, Clone, Default)]
pub struct IstioConfigBuilder {
    virtual_services: Vec<VirtualService>,
    destination_rules: Vec<DestinationRule>,
    gateways: Vec<IstioGateway>,
}

impl IstioConfigBuilder {
    pub fn virtual_service(mut self, vs: VirtualService) -> Self {
        self.virtual_services.push(vs);
        self
    }
    pub fn destination_rule(mut self, dr: DestinationRule) -> Self {
        self.destination_rules.push(dr);
        self
    }
    pub fn gateway(mut self, gw: IstioGateway) -> Self {
        self.gateways.push(gw);
        self
    }
    pub fn build(self) -> IstioConfig {
        IstioConfig {
            virtual_services: self.virtual_services,
            destination_rules: self.destination_rules,
            gateways: self.gateways,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VirtualService {
    pub name: String,
    pub hosts: Vec<String>,
    pub http_routes: Vec<HttpRoute>,
    pub retries: Option<IstioRetryPolicy>,
    pub timeout: Option<String>,
}

impl VirtualService {
    pub fn builder() -> VirtualServiceBuilder {
        VirtualServiceBuilder::default()
    }
}

#[derive(Debug, Clone, Default)]
pub struct VirtualServiceBuilder {
    name: String,
    hosts: Vec<String>,
    http_routes: Vec<HttpRoute>,
    retries: Option<IstioRetryPolicy>,
    timeout: Option<String>,
}

impl VirtualServiceBuilder {
    pub fn name(mut self, v: impl Into<String>) -> Self {
        self.name = v.into();
        self
    }
    pub fn host(mut self, v: impl Into<String>) -> Self {
        self.hosts.push(v.into());
        self
    }
    pub fn http_route(mut self, r: HttpRoute) -> Self {
        self.http_routes.push(r);
        self
    }
    pub fn retries(mut self, r: IstioRetryPolicy) -> Self {
        self.retries = Some(r);
        self
    }
    pub fn timeout(mut self, t: impl Into<String>) -> Self {
        self.timeout = Some(t.into());
        self
    }
    pub fn build(self) -> VirtualService {
        VirtualService {
            name: self.name,
            hosts: self.hosts,
            http_routes: self.http_routes,
            retries: self.retries,
            timeout: self.timeout,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HttpRoute {
    pub match_conditions: Vec<HttpMatchRequest>,
    pub destinations: Vec<RouteDestination>,
    pub timeout: Option<String>,
    pub retries: Option<IstioRetryPolicy>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HttpMatchRequest {
    pub uri_prefix: Option<String>,
    pub headers: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteDestination {
    pub host: String,
    pub port: Option<u16>,
    pub weight: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IstioRetryPolicy {
    pub attempts: u32,
    pub per_try_timeout: String,
    pub retry_on: String,
}

impl IstioRetryPolicy {
    pub fn new(attempts: u32, per_try_timeout: impl Into<String>, retry_on: impl Into<String>) -> Self {
        Self {
            attempts,
            per_try_timeout: per_try_timeout.into(),
            retry_on: retry_on.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DestinationRule {
    pub name: String,
    pub host: String,
    pub traffic_policy: Option<TrafficPolicy>,
    pub subsets: Vec<Subset>,
}

impl DestinationRule {
    pub fn builder() -> DestinationRuleBuilder {
        DestinationRuleBuilder::default()
    }
}

#[derive(Debug, Clone, Default)]
pub struct DestinationRuleBuilder {
    name: String,
    host: String,
    traffic_policy: Option<TrafficPolicy>,
    subsets: Vec<Subset>,
}

impl DestinationRuleBuilder {
    pub fn name(mut self, v: impl Into<String>) -> Self {
        self.name = v.into();
        self
    }
    pub fn host(mut self, v: impl Into<String>) -> Self {
        self.host = v.into();
        self
    }
    pub fn traffic_policy(mut self, tp: TrafficPolicy) -> Self {
        self.traffic_policy = Some(tp);
        self
    }
    pub fn subset(mut self, s: Subset) -> Self {
        self.subsets.push(s);
        self
    }
    pub fn build(self) -> DestinationRule {
        DestinationRule {
            name: self.name,
            host: self.host,
            traffic_policy: self.traffic_policy,
            subsets: self.subsets,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrafficPolicy {
    pub connection_pool: Option<ConnectionPool>,
    pub outlier_detection: Option<OutlierDetection>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConnectionPool {
    pub tcp_max_connections: Option<u32>,
    pub http_max_pending: Option<u32>,
    pub http_max_requests: Option<u32>,
    pub http_max_retries: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutlierDetection {
    pub consecutive_errors: u32,
    pub interval: String,
    pub base_ejection_time: String,
    pub max_ejection_percent: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Subset {
    pub name: String,
    pub labels: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IstioGateway {
    pub name: String,
    pub servers: Vec<GatewayServer>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GatewayServer {
    pub port: GatewayPort,
    pub hosts: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GatewayPort {
    pub number: u16,
    pub name: String,
    pub protocol: String,
}

// ---------------------------------------------------------------------------
// Envoy types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvoyConfig {
    pub clusters: Vec<EnvoyCluster>,
    pub listeners: Vec<EnvoyListener>,
    pub routes: Vec<EnvoyRoute>,
}

impl EnvoyConfig {
    pub fn builder() -> EnvoyConfigBuilder {
        EnvoyConfigBuilder::default()
    }

    pub fn validate(&self) -> Vec<ConfigValidationError> {
        let mut errors = Vec::new();
        for cluster in &self.clusters {
            if cluster.name.is_empty() {
                errors.push(ConfigValidationError {
                    file: String::new(),
                    source_type: "Envoy".into(),
                    message: "cluster.name must not be empty".into(),
                });
            }
            if cluster.endpoints.is_empty() {
                errors.push(ConfigValidationError {
                    file: String::new(),
                    source_type: "Envoy".into(),
                    message: format!(
                        "cluster '{}' must have at least one endpoint",
                        cluster.name
                    ),
                });
            }
        }
        for listener in &self.listeners {
            if listener.address.is_empty() {
                errors.push(ConfigValidationError {
                    file: String::new(),
                    source_type: "Envoy".into(),
                    message: format!(
                        "listener '{}' must have an address",
                        listener.name
                    ),
                });
            }
        }
        errors
    }
}

#[derive(Debug, Clone, Default)]
pub struct EnvoyConfigBuilder {
    clusters: Vec<EnvoyCluster>,
    listeners: Vec<EnvoyListener>,
    routes: Vec<EnvoyRoute>,
}

impl EnvoyConfigBuilder {
    pub fn cluster(mut self, c: EnvoyCluster) -> Self {
        self.clusters.push(c);
        self
    }
    pub fn listener(mut self, l: EnvoyListener) -> Self {
        self.listeners.push(l);
        self
    }
    pub fn route(mut self, r: EnvoyRoute) -> Self {
        self.routes.push(r);
        self
    }
    pub fn build(self) -> EnvoyConfig {
        EnvoyConfig {
            clusters: self.clusters,
            listeners: self.listeners,
            routes: self.routes,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvoyCluster {
    pub name: String,
    pub connect_timeout: String,
    pub endpoints: Vec<EnvoyEndpoint>,
    pub circuit_breakers: Option<EnvoyCircuitBreaker>,
    pub outlier_detection: Option<EnvoyOutlierDetection>,
}

impl EnvoyCluster {
    pub fn builder() -> EnvoyClusterBuilder {
        EnvoyClusterBuilder::default()
    }
}

#[derive(Debug, Clone, Default)]
pub struct EnvoyClusterBuilder {
    name: String,
    connect_timeout: String,
    endpoints: Vec<EnvoyEndpoint>,
    circuit_breakers: Option<EnvoyCircuitBreaker>,
    outlier_detection: Option<EnvoyOutlierDetection>,
}

impl EnvoyClusterBuilder {
    pub fn name(mut self, v: impl Into<String>) -> Self {
        self.name = v.into();
        self
    }
    pub fn connect_timeout(mut self, v: impl Into<String>) -> Self {
        self.connect_timeout = v.into();
        self
    }
    pub fn endpoint(mut self, e: EnvoyEndpoint) -> Self {
        self.endpoints.push(e);
        self
    }
    pub fn circuit_breakers(mut self, cb: EnvoyCircuitBreaker) -> Self {
        self.circuit_breakers = Some(cb);
        self
    }
    pub fn outlier_detection(mut self, od: EnvoyOutlierDetection) -> Self {
        self.outlier_detection = Some(od);
        self
    }
    pub fn build(self) -> EnvoyCluster {
        EnvoyCluster {
            name: self.name,
            connect_timeout: self.connect_timeout,
            endpoints: self.endpoints,
            circuit_breakers: self.circuit_breakers,
            outlier_detection: self.outlier_detection,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvoyEndpoint {
    pub address: String,
    pub port: u16,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvoyCircuitBreaker {
    pub max_connections: u32,
    pub max_pending_requests: u32,
    pub max_requests: u32,
    pub max_retries: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvoyOutlierDetection {
    pub consecutive_5xx: u32,
    pub interval: String,
    pub base_ejection_time: String,
    pub max_ejection_percent: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvoyListener {
    pub name: String,
    pub address: String,
    pub port: u16,
    pub filter_chains: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvoyRoute {
    pub name: String,
    pub virtual_hosts: Vec<EnvoyVirtualHost>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvoyVirtualHost {
    pub name: String,
    pub domains: Vec<String>,
    pub routes: Vec<EnvoyRouteEntry>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvoyRouteEntry {
    pub prefix: String,
    pub cluster: String,
    pub timeout: Option<String>,
    pub retry_policy: Option<EnvoyRetryPolicy>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnvoyRetryPolicy {
    pub retry_on: String,
    pub num_retries: u32,
    pub per_try_timeout: String,
}

// ---------------------------------------------------------------------------
// ConfigSource enum
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ConfigSource {
    Kubernetes(KubernetesConfig),
    Istio(IstioConfig),
    Envoy(EnvoyConfig),
    Raw { format: String, content: String },
}

impl ConfigSource {
    pub fn source_type(&self) -> &str {
        match self {
            ConfigSource::Kubernetes(_) => "Kubernetes",
            ConfigSource::Istio(_) => "Istio",
            ConfigSource::Envoy(_) => "Envoy",
            ConfigSource::Raw { .. } => "Raw",
        }
    }

    pub fn validate(&self) -> Vec<ConfigValidationError> {
        match self {
            ConfigSource::Kubernetes(k) => k.validate(),
            ConfigSource::Istio(i) => i.validate(),
            ConfigSource::Envoy(e) => e.validate(),
            ConfigSource::Raw { format, content } => {
                let mut errors = Vec::new();
                if format.is_empty() {
                    errors.push(ConfigValidationError {
                        file: String::new(),
                        source_type: "Raw".into(),
                        message: "format must not be empty".into(),
                    });
                }
                if content.is_empty() {
                    errors.push(ConfigValidationError {
                        file: String::new(),
                        source_type: "Raw".into(),
                        message: "content must not be empty".into(),
                    });
                }
                errors
            }
        }
    }
}

impl fmt::Display for ConfigSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigSource::Kubernetes(k) => {
                write!(f, "Kubernetes({})", k.metadata.qualified_name())
            }
            ConfigSource::Istio(i) => {
                write!(
                    f,
                    "Istio({} virtual_services, {} destination_rules)",
                    i.virtual_services.len(),
                    i.destination_rules.len()
                )
            }
            ConfigSource::Envoy(e) => {
                write!(
                    f,
                    "Envoy({} clusters, {} listeners)",
                    e.clusters.len(),
                    e.listeners.len()
                )
            }
            ConfigSource::Raw { format, .. } => write!(f, "Raw({})", format),
        }
    }
}

// ---------------------------------------------------------------------------
// ConfigManifest
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfigManifest {
    pub sources: Vec<ConfigSource>,
    pub file_paths: Vec<String>,
}

impl Default for ConfigManifest {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigManifest {
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
            file_paths: Vec::new(),
        }
    }

    pub fn add(&mut self, source: ConfigSource) {
        self.sources.push(source);
    }

    pub fn len(&self) -> usize {
        self.sources.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    pub fn kubernetes_configs(&self) -> Vec<&KubernetesConfig> {
        self.sources
            .iter()
            .filter_map(|s| match s {
                ConfigSource::Kubernetes(k) => Some(k),
                _ => None,
            })
            .collect()
    }

    pub fn istio_configs(&self) -> Vec<&IstioConfig> {
        self.sources
            .iter()
            .filter_map(|s| match s {
                ConfigSource::Istio(i) => Some(i),
                _ => None,
            })
            .collect()
    }

    pub fn envoy_configs(&self) -> Vec<&EnvoyConfig> {
        self.sources
            .iter()
            .filter_map(|s| match s {
                ConfigSource::Envoy(e) => Some(e),
                _ => None,
            })
            .collect()
    }

    pub fn validate_all(&self) -> Vec<ConfigValidationError> {
        self.sources.iter().flat_map(|s| s.validate()).collect()
    }
}

// ---------------------------------------------------------------------------
// Validation / warning helpers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfigValidationError {
    pub file: String,
    pub source_type: String,
    pub message: String,
}

impl fmt::Display for ConfigValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.file.is_empty() {
            write!(f, "[{}] {}", self.source_type, self.message)
        } else {
            write!(f, "[{}] {}: {}", self.source_type, self.file, self.message)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfigWarning {
    pub file: String,
    pub source_type: String,
    pub message: String,
    pub suggestion: Option<String>,
}

impl ConfigWarning {
    pub fn builder() -> ConfigWarningBuilder {
        ConfigWarningBuilder::default()
    }
}

impl fmt::Display for ConfigWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WARN [{}] {}", self.source_type, self.message)?;
        if let Some(ref sug) = self.suggestion {
            write!(f, " (suggestion: {})", sug)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct ConfigWarningBuilder {
    file: String,
    source_type: String,
    message: String,
    suggestion: Option<String>,
}

impl ConfigWarningBuilder {
    pub fn file(mut self, v: impl Into<String>) -> Self {
        self.file = v.into();
        self
    }
    pub fn source_type(mut self, v: impl Into<String>) -> Self {
        self.source_type = v.into();
        self
    }
    pub fn message(mut self, v: impl Into<String>) -> Self {
        self.message = v.into();
        self
    }
    pub fn suggestion(mut self, v: impl Into<String>) -> Self {
        self.suggestion = Some(v.into());
        self
    }
    pub fn build(self) -> ConfigWarning {
        ConfigWarning {
            file: self.file,
            source_type: self.source_type,
            message: self.message,
            suggestion: self.suggestion,
        }
    }
}

// ---------------------------------------------------------------------------
// Helm types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HelmValues {
    pub chart_name: String,
    pub chart_version: Option<String>,
    pub values: serde_json::Value,
    pub overrides: Vec<HelmOverride>,
}

impl HelmValues {
    pub fn builder() -> HelmValuesBuilder {
        HelmValuesBuilder::default()
    }

    /// Retrieve a value at a dotted path (e.g. "a.b.c").
    pub fn get_value(&self, path: &str) -> Option<&serde_json::Value> {
        let parts: Vec<&str> = path.split('.').collect();
        let mut current = &self.values;
        for part in parts {
            match current.get(part) {
                Some(v) => current = v,
                None => return None,
            }
        }
        Some(current)
    }

    /// Return a new `serde_json::Value` with all overrides applied in order.
    pub fn resolve(&self) -> serde_json::Value {
        let mut resolved = self.values.clone();
        for ov in &self.overrides {
            set_nested_value(&mut resolved, &ov.key, ov.value.clone());
        }
        resolved
    }
}

#[derive(Debug, Clone, Default)]
pub struct HelmValuesBuilder {
    chart_name: String,
    chart_version: Option<String>,
    values: serde_json::Value,
    overrides: Vec<HelmOverride>,
}

impl HelmValuesBuilder {
    pub fn chart_name(mut self, v: impl Into<String>) -> Self {
        self.chart_name = v.into();
        self
    }
    pub fn chart_version(mut self, v: impl Into<String>) -> Self {
        self.chart_version = Some(v.into());
        self
    }
    pub fn values(mut self, v: serde_json::Value) -> Self {
        self.values = v;
        self
    }
    pub fn override_value(mut self, ov: HelmOverride) -> Self {
        self.overrides.push(ov);
        self
    }
    pub fn build(self) -> HelmValues {
        HelmValues {
            chart_name: self.chart_name,
            chart_version: self.chart_version,
            values: self.values,
            overrides: self.overrides,
        }
    }
}

/// Set a value inside a nested `serde_json::Value` map at a dotted path,
/// creating intermediate objects as needed.
pub fn set_nested_value(root: &mut serde_json::Value, path: &str, value: serde_json::Value) {
    let parts: Vec<&str> = path.split('.').collect();
    let mut current = root;
    for (i, part) in parts.iter().enumerate() {
        if i == parts.len() - 1 {
            current[*part] = value;
            return;
        }
        if !current[*part].is_object() {
            current[*part] = serde_json::json!({});
        }
        current = &mut current[*part];
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HelmOverride {
    pub key: String,
    pub value: serde_json::Value,
    pub source: String,
}

// ---------------------------------------------------------------------------
// Kustomize types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KustomizeOverlay {
    pub name: String,
    pub resources: Vec<String>,
    pub patches: Vec<KustomizePatch>,
    pub config_map_generators: Vec<String>,
}

impl KustomizeOverlay {
    pub fn builder() -> KustomizeOverlayBuilder {
        KustomizeOverlayBuilder::default()
    }
}

#[derive(Debug, Clone, Default)]
pub struct KustomizeOverlayBuilder {
    name: String,
    resources: Vec<String>,
    patches: Vec<KustomizePatch>,
    config_map_generators: Vec<String>,
}

impl KustomizeOverlayBuilder {
    pub fn name(mut self, v: impl Into<String>) -> Self {
        self.name = v.into();
        self
    }
    pub fn resource(mut self, v: impl Into<String>) -> Self {
        self.resources.push(v.into());
        self
    }
    pub fn patch(mut self, p: KustomizePatch) -> Self {
        self.patches.push(p);
        self
    }
    pub fn config_map_generator(mut self, v: impl Into<String>) -> Self {
        self.config_map_generators.push(v.into());
        self
    }
    pub fn build(self) -> KustomizeOverlay {
        KustomizeOverlay {
            name: self.name,
            resources: self.resources,
            patches: self.patches,
            config_map_generators: self.config_map_generators,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KustomizePatch {
    pub target_kind: String,
    pub target_name: String,
    pub patch: String,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample_kube_metadata() -> KubeMetadata {
        KubeMetadata {
            name: "my-svc".into(),
            namespace: Some("production".into()),
            labels: BTreeMap::from([("app".into(), "web".into())]),
            annotations: BTreeMap::new(),
        }
    }

    fn sample_deployment() -> DeploymentSpec {
        DeploymentSpec {
            replicas: 3,
            selector: BTreeMap::from([("app".into(), "web".into())]),
            container_image: "nginx:latest".into(),
            resource_limits: Some(ResourceLimits {
                cpu: Some("500m".into()),
                memory: Some("256Mi".into()),
            }),
        }
    }

    // 1. KubernetesConfig creation & validation (valid)
    #[test]
    fn test_kubernetes_config_valid() {
        let cfg = KubernetesConfig::builder()
            .api_version("apps/v1")
            .kind("Deployment")
            .metadata(sample_kube_metadata())
            .deployment(sample_deployment())
            .build();

        assert_eq!(cfg.api_version, "apps/v1");
        assert!(cfg.validate().is_empty());
    }

    // 2. KubernetesConfig validation catches errors
    #[test]
    fn test_kubernetes_config_invalid() {
        let cfg = KubernetesConfig::builder()
            .api_version("")
            .kind("Deployment")
            .deployment(DeploymentSpec {
                replicas: 0,
                selector: BTreeMap::new(),
                container_image: "".into(),
                resource_limits: None,
            })
            .build();

        let errs = cfg.validate();
        assert!(errs.len() >= 3);
        let msgs: Vec<&str> = errs.iter().map(|e| e.message.as_str()).collect();
        assert!(msgs.iter().any(|m| m.contains("metadata.name")));
        assert!(msgs.iter().any(|m| m.contains("api_version")));
        assert!(msgs.iter().any(|m| m.contains("replicas")));
    }

    // 3. KubeMetadata qualified_name
    #[test]
    fn test_kube_metadata_qualified_name() {
        let meta = sample_kube_metadata();
        assert_eq!(meta.qualified_name(), "production/my-svc");

        let no_ns = KubeMetadata {
            name: "solo".into(),
            namespace: None,
            labels: BTreeMap::new(),
            annotations: BTreeMap::new(),
        };
        assert_eq!(no_ns.qualified_name(), "solo");
    }

    // 4. Istio config validation – valid
    #[test]
    fn test_istio_config_valid() {
        let vs = VirtualService::builder()
            .name("reviews")
            .host("reviews.default.svc.cluster.local")
            .http_route(HttpRoute {
                match_conditions: vec![],
                destinations: vec![RouteDestination {
                    host: "reviews".into(),
                    port: Some(9080),
                    weight: 100,
                }],
                timeout: None,
                retries: None,
            })
            .build();

        let cfg = IstioConfig::builder().virtual_service(vs).build();
        assert!(cfg.validate().is_empty());
    }

    // 5. Istio config validation – weight mismatch
    #[test]
    fn test_istio_config_invalid_weights() {
        let vs = VirtualService::builder()
            .name("reviews")
            .host("reviews")
            .http_route(HttpRoute {
                match_conditions: vec![],
                destinations: vec![
                    RouteDestination { host: "v1".into(), port: None, weight: 50 },
                    RouteDestination { host: "v2".into(), port: None, weight: 30 },
                ],
                timeout: None,
                retries: None,
            })
            .build();

        let cfg = IstioConfig::builder().virtual_service(vs).build();
        let errs = cfg.validate();
        assert!(!errs.is_empty());
        assert!(errs[0].message.contains("weights must sum to 100"));
    }

    // 6. Envoy validation
    #[test]
    fn test_envoy_config_valid() {
        let cluster = EnvoyCluster::builder()
            .name("service_a")
            .connect_timeout("0.25s")
            .endpoint(EnvoyEndpoint { address: "10.0.0.1".into(), port: 8080 })
            .build();

        let cfg = EnvoyConfig::builder().cluster(cluster).build();
        assert!(cfg.validate().is_empty());
    }

    // 7. Envoy validation – empty endpoints
    #[test]
    fn test_envoy_config_invalid() {
        let cluster = EnvoyCluster::builder()
            .name("bad_cluster")
            .connect_timeout("1s")
            .build();

        let cfg = EnvoyConfig::builder().cluster(cluster).build();
        let errs = cfg.validate();
        assert!(!errs.is_empty());
        assert!(errs[0].message.contains("at least one endpoint"));
    }

    // 8. ConfigSource type string
    #[test]
    fn test_config_source_type() {
        let k = ConfigSource::Kubernetes(KubernetesConfig::builder().api_version("v1").kind("Service").metadata(sample_kube_metadata()).build());
        assert_eq!(k.source_type(), "Kubernetes");

        let raw = ConfigSource::Raw { format: "yaml".into(), content: "foo: bar".into() };
        assert_eq!(raw.source_type(), "Raw");
    }

    // 9. ConfigManifest filtering
    #[test]
    fn test_config_manifest_filtering() {
        let mut m = ConfigManifest::new();
        assert!(m.is_empty());

        m.add(ConfigSource::Kubernetes(KubernetesConfig::builder().api_version("v1").kind("Svc").metadata(sample_kube_metadata()).build()));
        m.add(ConfigSource::Istio(IstioConfig::builder().build()));
        m.add(ConfigSource::Envoy(EnvoyConfig::builder().build()));

        assert_eq!(m.len(), 3);
        assert_eq!(m.kubernetes_configs().len(), 1);
        assert_eq!(m.istio_configs().len(), 1);
        assert_eq!(m.envoy_configs().len(), 1);
    }

    // 10. validate_all aggregation
    #[test]
    fn test_validate_all() {
        let mut m = ConfigManifest::new();
        // invalid kubernetes (empty name)
        m.add(ConfigSource::Kubernetes(KubernetesConfig::builder().api_version("").kind("x").build()));
        // invalid raw
        m.add(ConfigSource::Raw { format: "".into(), content: "".into() });

        let errs = m.validate_all();
        assert!(errs.len() >= 3);
    }

    // 11. ConfigWarning display
    #[test]
    fn test_config_warning_display() {
        let w = ConfigWarning::builder()
            .source_type("Istio")
            .message("no retries configured")
            .suggestion("add retry policy")
            .build();

        let s = format!("{}", w);
        assert!(s.contains("WARN [Istio]"));
        assert!(s.contains("suggestion: add retry policy"));
    }

    // 12. HelmValues get_value
    #[test]
    fn test_helm_get_value() {
        let hv = HelmValues::builder()
            .chart_name("my-chart")
            .values(json!({
                "replicaCount": 3,
                "image": { "repository": "nginx", "tag": "latest" }
            }))
            .build();

        assert_eq!(hv.get_value("replicaCount"), Some(&json!(3)));
        assert_eq!(hv.get_value("image.repository"), Some(&json!("nginx")));
        assert!(hv.get_value("nonexistent.path").is_none());
    }

    // 13. HelmValues resolve with overrides
    #[test]
    fn test_helm_resolve() {
        let hv = HelmValues::builder()
            .chart_name("my-chart")
            .values(json!({ "image": { "tag": "latest" }, "replicas": 1 }))
            .override_value(HelmOverride {
                key: "image.tag".into(),
                value: json!("v2.0.0"),
                source: "cli".into(),
            })
            .override_value(HelmOverride {
                key: "replicas".into(),
                value: json!(5),
                source: "values-prod.yaml".into(),
            })
            .build();

        let resolved = hv.resolve();
        assert_eq!(resolved["image"]["tag"], json!("v2.0.0"));
        assert_eq!(resolved["replicas"], json!(5));
    }

    // 14. KustomizeOverlay builder
    #[test]
    fn test_kustomize_overlay() {
        let overlay = KustomizeOverlay::builder()
            .name("production")
            .resource("deployment.yaml")
            .resource("service.yaml")
            .patch(KustomizePatch {
                target_kind: "Deployment".into(),
                target_name: "web".into(),
                patch: r#"[{"op":"replace","path":"/spec/replicas","value":5}]"#.into(),
            })
            .config_map_generator("app-config")
            .build();

        assert_eq!(overlay.name, "production");
        assert_eq!(overlay.resources.len(), 2);
        assert_eq!(overlay.patches.len(), 1);
        assert_eq!(overlay.config_map_generators.len(), 1);
    }

    // 15. Serialization roundtrip
    #[test]
    fn test_serialization_roundtrip() {
        let cfg = KubernetesConfig::builder()
            .api_version("v1")
            .kind("Service")
            .metadata(sample_kube_metadata())
            .build();

        let json_str = serde_json::to_string(&cfg).unwrap();
        let deserialized: KubernetesConfig = serde_json::from_str(&json_str).unwrap();
        assert_eq!(cfg, deserialized);
    }

    // 16. IstioRetryPolicy constructor
    #[test]
    fn test_istio_retry_policy_new() {
        let policy = IstioRetryPolicy::new(3, "2s", "5xx,reset");
        assert_eq!(policy.attempts, 3);
        assert_eq!(policy.per_try_timeout, "2s");
        assert_eq!(policy.retry_on, "5xx,reset");
    }

    // 17. ConfigSource Display
    #[test]
    fn test_config_source_display() {
        let src = ConfigSource::Envoy(EnvoyConfig {
            clusters: vec![],
            listeners: vec![],
            routes: vec![],
        });
        let s = format!("{}", src);
        assert!(s.contains("Envoy(0 clusters"));
    }

    // 18. IngressSpec default
    #[test]
    fn test_ingress_spec_default() {
        let spec = IngressSpec::default();
        assert!(spec.rules.is_empty());
        assert!(spec.tls.is_empty());
    }

    // 19. set_nested_value creates intermediate objects
    #[test]
    fn test_set_nested_value() {
        let mut root = json!({});
        set_nested_value(&mut root, "a.b.c", json!(42));
        assert_eq!(root["a"]["b"]["c"], json!(42));
    }
}
