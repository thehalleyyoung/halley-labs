//! Istio configuration parsing and policy extraction.
//!
//! This module handles parsing Istio networking resources from YAML:
//! VirtualService, DestinationRule, Gateway, and ServiceEntry.
//! It also provides policy merging, precedence resolution, and conversion
//! to the unified [`RetryPolicy`] / [`TimeoutPolicy`] types.

use anyhow::{bail, Context, Result};
use indexmap::IndexMap;
use log::{debug, warn};
use serde::{Deserialize, Serialize};

use crate::{ObjectMeta, RetryPolicy, TimeoutPolicy};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Istio VirtualService resource.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VirtualService {
    pub metadata: ObjectMeta,
    pub hosts: Vec<String>,
    #[serde(default)]
    pub gateways: Vec<String>,
    #[serde(default)]
    pub http_routes: Vec<HttpRoute>,
    #[serde(default)]
    pub tcp_routes: Vec<TcpRoute>,
    #[serde(default)]
    pub tls_routes: Vec<TlsRoute>,
    #[serde(default)]
    pub export_to: Vec<String>,
}

/// A single HTTP route entry inside a VirtualService.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HttpRoute {
    pub name: Option<String>,
    #[serde(default)]
    pub match_conditions: Vec<HttpMatchRequest>,
    #[serde(default)]
    pub route: Vec<HttpRouteDestination>,
    pub retries: Option<HttpRetryPolicy>,
    pub timeout: Option<String>,
    pub fault: Option<FaultInjection>,
    pub mirror: Option<Destination>,
    pub headers: Option<HeaderOperations>,
    pub rewrite: Option<HttpRewrite>,
}

/// Match criteria for an HTTP route.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HttpMatchRequest {
    pub uri: Option<StringMatch>,
    #[serde(default)]
    pub headers: IndexMap<String, StringMatch>,
    pub method: Option<StringMatch>,
    pub authority: Option<StringMatch>,
    pub port: Option<u16>,
    #[serde(default)]
    pub source_labels: IndexMap<String, String>,
}

/// String match strategy used across Istio matching.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum StringMatch {
    Exact(String),
    Prefix(String),
    Regex(String),
}

/// A weighted route destination in an HTTP route.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HttpRouteDestination {
    pub destination: Destination,
    #[serde(default = "default_weight")]
    pub weight: u32,
    pub headers: Option<HeaderOperations>,
}

fn default_weight() -> u32 {
    100
}

/// A target destination (host + optional port/subset).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Destination {
    pub host: String,
    pub port: Option<PortSelector>,
    pub subset: Option<String>,
}

/// Port selector by number.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PortSelector {
    pub number: u16,
}

/// Istio HTTP retry policy (as declared in VirtualService).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HttpRetryPolicy {
    pub attempts: u32,
    #[serde(default = "default_per_try_timeout")]
    pub per_try_timeout: String,
    #[serde(default = "default_retry_on")]
    pub retry_on: String,
    #[serde(default)]
    pub retry_remote_localities: bool,
}

fn default_per_try_timeout() -> String {
    "2s".to_string()
}

fn default_retry_on() -> String {
    "connect-failure,refused-stream,unavailable,cancelled,retriable-status-codes".to_string()
}

/// Fault injection configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FaultInjection {
    pub delay: Option<FaultDelay>,
    pub abort: Option<FaultAbort>,
}

/// Fixed-delay fault injection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FaultDelay {
    pub percentage: f64,
    pub fixed_delay: String,
}

/// Abort fault injection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FaultAbort {
    pub percentage: f64,
    pub http_status: u16,
}

/// Header manipulation operations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct HeaderOperations {
    #[serde(default)]
    pub set: IndexMap<String, String>,
    #[serde(default)]
    pub add: IndexMap<String, String>,
    #[serde(default)]
    pub remove: Vec<String>,
}

/// HTTP rewrite configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HttpRewrite {
    pub uri: Option<String>,
    pub authority: Option<String>,
}

// ---- TCP Route types ------------------------------------------------------

/// A TCP route entry inside a VirtualService.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TcpRoute {
    #[serde(default)]
    pub match_conditions: Vec<TcpMatchRequest>,
    #[serde(default)]
    pub route: Vec<TcpRouteDestination>,
}

/// Match criteria for a TCP route.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TcpMatchRequest {
    #[serde(default)]
    pub destination_subnets: Vec<String>,
    pub port: Option<u16>,
    #[serde(default)]
    pub source_labels: IndexMap<String, String>,
}

/// A weighted route destination in a TCP route.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TcpRouteDestination {
    pub destination: Destination,
    #[serde(default = "default_weight")]
    pub weight: u32,
}

// ---- TLS Route types ------------------------------------------------------

/// A TLS route entry inside a VirtualService.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TlsRoute {
    #[serde(default)]
    pub match_conditions: Vec<TlsMatchRequest>,
    #[serde(default)]
    pub route: Vec<TcpRouteDestination>,
}

/// Match criteria for a TLS route.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TlsMatchRequest {
    #[serde(default)]
    pub sni_hosts: Vec<String>,
    pub port: Option<u16>,
    #[serde(default)]
    pub source_labels: IndexMap<String, String>,
}

// ---- DestinationRule types ------------------------------------------------

/// Istio DestinationRule resource.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DestinationRule {
    pub metadata: ObjectMeta,
    pub host: String,
    pub traffic_policy: Option<TrafficPolicy>,
    #[serde(default)]
    pub subsets: Vec<Subset>,
    #[serde(default)]
    pub export_to: Vec<String>,
}

/// Traffic policy (connection pool, LB, outlier detection, TLS).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct TrafficPolicy {
    pub connection_pool: Option<ConnectionPool>,
    pub load_balancer: Option<LoadBalancerSettings>,
    pub outlier_detection: Option<OutlierDetection>,
    pub tls: Option<TlsSettings>,
    #[serde(default)]
    pub port_level_settings: Vec<PortTrafficPolicy>,
}

/// Connection pool settings (TCP + HTTP).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConnectionPool {
    pub tcp: Option<TcpSettings>,
    pub http: Option<HttpSettings>,
}

/// TCP connection pool settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TcpSettings {
    #[serde(default = "default_max_connections")]
    pub max_connections: u32,
    #[serde(default = "default_connect_timeout")]
    pub connect_timeout: String,
    pub tcp_keepalive: Option<TcpKeepalive>,
}

fn default_max_connections() -> u32 {
    1024
}

fn default_connect_timeout() -> String {
    "10s".to_string()
}

/// TCP keepalive settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TcpKeepalive {
    pub probes: u32,
    pub time: String,
    pub interval: String,
}

/// HTTP connection pool settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HttpSettings {
    #[serde(default = "default_h2_upgrade_policy")]
    pub h2_upgrade_policy: String,
    #[serde(default = "default_max_requests")]
    pub max_requests_per_connection: u32,
    #[serde(default = "default_max_retries_http")]
    pub max_retries: u32,
    #[serde(default = "default_idle_timeout")]
    pub idle_timeout: String,
}

fn default_h2_upgrade_policy() -> String {
    "DEFAULT".to_string()
}

fn default_max_requests() -> u32 {
    0
}

fn default_max_retries_http() -> u32 {
    3
}

fn default_idle_timeout() -> String {
    "1h".to_string()
}

/// Load balancer settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoadBalancerSettings {
    pub simple: Option<String>,
    pub consistent_hash: Option<ConsistentHashLB>,
}

/// Consistent hash load balancing configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConsistentHashLB {
    pub http_header_name: Option<String>,
    #[serde(default = "default_ring_size")]
    pub minimum_ring_size: u64,
}

fn default_ring_size() -> u64 {
    1024
}

/// Outlier detection (circuit breaker) settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OutlierDetection {
    #[serde(default = "default_consecutive_errors")]
    pub consecutive_errors: u32,
    #[serde(default = "default_interval")]
    pub interval: String,
    #[serde(default = "default_base_ejection_time")]
    pub base_ejection_time: String,
    #[serde(default = "default_max_ejection_percent")]
    pub max_ejection_percent: u32,
    #[serde(default)]
    pub min_health_percent: u32,
}

fn default_consecutive_errors() -> u32 {
    5
}

fn default_interval() -> String {
    "10s".to_string()
}

fn default_base_ejection_time() -> String {
    "30s".to_string()
}

fn default_max_ejection_percent() -> u32 {
    10
}

/// TLS settings on a traffic policy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TlsSettings {
    pub mode: String,
    pub client_certificate: Option<String>,
    pub private_key: Option<String>,
    pub ca_certificates: Option<String>,
}

/// Per-port traffic policy override.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PortTrafficPolicy {
    pub port: PortSelector,
    pub traffic_policy: Option<Box<TrafficPolicy>>,
}

/// A named subset of a DestinationRule.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Subset {
    pub name: String,
    #[serde(default)]
    pub labels: IndexMap<String, String>,
    pub traffic_policy: Option<TrafficPolicy>,
}

// ---- Gateway types --------------------------------------------------------

/// Istio Gateway resource.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Gateway {
    pub metadata: ObjectMeta,
    #[serde(default)]
    pub servers: Vec<Server>,
    #[serde(default)]
    pub selector: IndexMap<String, String>,
}

/// A server entry inside a Gateway.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Server {
    pub port: GatewayPort,
    pub hosts: Vec<String>,
    pub tls: Option<GatewayTls>,
}

/// Gateway port declaration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GatewayPort {
    pub number: u16,
    pub name: String,
    pub protocol: String,
}

/// TLS configuration on a gateway server.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GatewayTls {
    pub mode: String,
    pub credential_name: Option<String>,
}

// ---- ServiceEntry types ---------------------------------------------------

/// Istio ServiceEntry resource.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ServiceEntry {
    pub metadata: ObjectMeta,
    pub hosts: Vec<String>,
    #[serde(default)]
    pub ports: Vec<ServiceEntryPort>,
    #[serde(default = "default_location")]
    pub location: String,
    #[serde(default = "default_resolution")]
    pub resolution: String,
    #[serde(default)]
    pub endpoints: Vec<ServiceEntryEndpoint>,
}

fn default_location() -> String {
    "MESH_EXTERNAL".to_string()
}

fn default_resolution() -> String {
    "DNS".to_string()
}

/// Port declaration in a ServiceEntry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ServiceEntryPort {
    pub number: u16,
    pub name: String,
    pub protocol: String,
}

/// Endpoint declaration in a ServiceEntry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ServiceEntryEndpoint {
    pub address: String,
    #[serde(default)]
    pub ports: IndexMap<String, u16>,
    #[serde(default)]
    pub labels: IndexMap<String, String>,
}

// ---- Enum wrapper ---------------------------------------------------------

/// Discriminated union over all supported Istio resource kinds.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IstioConfig {
    VirtualService(VirtualService),
    DestinationRule(DestinationRule),
    Gateway(Gateway),
    ServiceEntry(ServiceEntry),
}

// ---------------------------------------------------------------------------
// IstioParser
// ---------------------------------------------------------------------------

/// Main parser entry point for Istio configuration resources.
#[derive(Debug, Clone, Default)]
pub struct IstioParser;

impl IstioParser {
    pub fn new() -> Self {
        Self
    }

    // ---- Top-level parsers ------------------------------------------------

    /// Parse a YAML document as a VirtualService.
    pub fn parse_virtual_service(yaml: &str) -> Result<VirtualService> {
        let doc: serde_yaml::Value =
            serde_yaml::from_str(yaml).context("Failed to parse YAML for VirtualService")?;

        let metadata = parse_metadata(&doc)?;

        let spec = doc
            .get("spec")
            .context("VirtualService missing 'spec' field")?;

        let hosts = parse_string_array(spec.get("hosts").unwrap_or(&serde_yaml::Value::Null));
        let gateways =
            parse_string_array(spec.get("gateways").unwrap_or(&serde_yaml::Value::Null));
        let export_to =
            parse_string_array(spec.get("exportTo").unwrap_or(&serde_yaml::Value::Null));

        let http_routes = if let Some(http_val) = spec.get("http") {
            parse_http_routes(http_val)?
        } else {
            Vec::new()
        };

        let tcp_routes = if let Some(tcp_val) = spec.get("tcp") {
            parse_tcp_routes(tcp_val)?
        } else {
            Vec::new()
        };

        let tls_routes = if let Some(tls_val) = spec.get("tls") {
            parse_tls_routes(tls_val)?
        } else {
            Vec::new()
        };

        Ok(VirtualService {
            metadata,
            hosts,
            gateways,
            http_routes,
            tcp_routes,
            tls_routes,
            export_to,
        })
    }

    /// Parse a YAML document as a DestinationRule.
    pub fn parse_destination_rule(yaml: &str) -> Result<DestinationRule> {
        let doc: serde_yaml::Value =
            serde_yaml::from_str(yaml).context("Failed to parse YAML for DestinationRule")?;

        let metadata = parse_metadata(&doc)?;

        let spec = doc
            .get("spec")
            .context("DestinationRule missing 'spec' field")?;

        let host = spec
            .get("host")
            .and_then(|v| v.as_str())
            .context("DestinationRule spec missing 'host'")?
            .to_string();

        let traffic_policy = if let Some(tp_val) = spec.get("trafficPolicy") {
            Some(parse_traffic_policy(tp_val)?)
        } else {
            None
        };

        let subsets = if let Some(arr) = spec.get("subsets").and_then(|v| v.as_sequence()) {
            arr.iter().map(parse_subset).collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };

        let export_to =
            parse_string_array(spec.get("exportTo").unwrap_or(&serde_yaml::Value::Null));

        Ok(DestinationRule {
            metadata,
            host,
            traffic_policy,
            subsets,
            export_to,
        })
    }

    /// Parse a YAML document as a Gateway.
    pub fn parse_gateway(yaml: &str) -> Result<Gateway> {
        let doc: serde_yaml::Value =
            serde_yaml::from_str(yaml).context("Failed to parse YAML for Gateway")?;

        let metadata = parse_metadata(&doc)?;

        let spec = doc.get("spec").context("Gateway missing 'spec' field")?;

        let selector = if let Some(sel) = spec.get("selector") {
            parse_string_map(sel)
        } else {
            IndexMap::new()
        };

        let servers = if let Some(arr) = spec.get("servers").and_then(|v| v.as_sequence()) {
            arr.iter().map(parse_server).collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };

        Ok(Gateway {
            metadata,
            servers,
            selector,
        })
    }

    /// Parse a YAML document as a ServiceEntry.
    pub fn parse_service_entry(yaml: &str) -> Result<ServiceEntry> {
        let doc: serde_yaml::Value =
            serde_yaml::from_str(yaml).context("Failed to parse YAML for ServiceEntry")?;

        let metadata = parse_metadata(&doc)?;

        let spec = doc
            .get("spec")
            .context("ServiceEntry missing 'spec' field")?;

        let hosts = parse_string_array(spec.get("hosts").unwrap_or(&serde_yaml::Value::Null));

        let location = spec
            .get("location")
            .and_then(|v| v.as_str())
            .unwrap_or("MESH_EXTERNAL")
            .to_string();

        let resolution = spec
            .get("resolution")
            .and_then(|v| v.as_str())
            .unwrap_or("DNS")
            .to_string();

        let ports = if let Some(arr) = spec.get("ports").and_then(|v| v.as_sequence()) {
            arr.iter()
                .map(parse_service_entry_port)
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };

        let endpoints = if let Some(arr) = spec.get("endpoints").and_then(|v| v.as_sequence()) {
            arr.iter()
                .map(parse_service_entry_endpoint)
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };

        Ok(ServiceEntry {
            metadata,
            hosts,
            ports,
            location,
            resolution,
            endpoints,
        })
    }

    /// Auto-detect kind and parse accordingly.
    pub fn parse_istio_config(yaml: &str) -> Result<IstioConfig> {
        let doc: serde_yaml::Value =
            serde_yaml::from_str(yaml).context("Failed to parse YAML for IstioConfig")?;

        let kind = doc
            .get("kind")
            .and_then(|v| v.as_str())
            .context("Missing 'kind' field in YAML document")?;

        debug!("Parsing Istio resource of kind: {}", kind);

        match kind {
            "VirtualService" => {
                Ok(IstioConfig::VirtualService(Self::parse_virtual_service(yaml)?))
            }
            "DestinationRule" => Ok(IstioConfig::DestinationRule(
                Self::parse_destination_rule(yaml)?,
            )),
            "Gateway" => Ok(IstioConfig::Gateway(Self::parse_gateway(yaml)?)),
            "ServiceEntry" => Ok(IstioConfig::ServiceEntry(Self::parse_service_entry(yaml)?)),
            other => bail!("Unsupported Istio resource kind: {}", other),
        }
    }

    // ---- Policy functions -------------------------------------------------

    /// Merge multiple VirtualServices that target the same host. Later entries
    /// take precedence for metadata; routes from all entries are combined.
    pub fn merge_virtual_services(services: &[VirtualService]) -> Result<VirtualService> {
        if services.is_empty() {
            bail!("Cannot merge zero VirtualServices");
        }

        let mut merged = services[0].clone();

        for vs in &services[1..] {
            // Later metadata wins
            merged.metadata = vs.metadata.clone();

            // Combine hosts (dedup)
            for h in &vs.hosts {
                if !merged.hosts.contains(h) {
                    merged.hosts.push(h.clone());
                }
            }

            // Combine gateways (dedup)
            for g in &vs.gateways {
                if !merged.gateways.contains(g) {
                    merged.gateways.push(g.clone());
                }
            }

            // Prepend later routes so they take higher precedence
            let mut new_http = vs.http_routes.clone();
            new_http.extend(merged.http_routes.drain(..));
            merged.http_routes = new_http;

            let mut new_tcp = vs.tcp_routes.clone();
            new_tcp.extend(merged.tcp_routes.drain(..));
            merged.tcp_routes = new_tcp;

            let mut new_tls = vs.tls_routes.clone();
            new_tls.extend(merged.tls_routes.drain(..));
            merged.tls_routes = new_tls;

            // Last export_to wins
            if !vs.export_to.is_empty() {
                merged.export_to = vs.export_to.clone();
            }
        }

        debug!(
            "Merged {} VirtualServices; total http routes: {}",
            services.len(),
            merged.http_routes.len()
        );

        Ok(merged)
    }

    /// Merge multiple DestinationRules for the same host. Later entries
    /// override traffic policy; subsets are combined (last wins per name).
    pub fn merge_destination_rules(rules: &[DestinationRule]) -> Result<DestinationRule> {
        if rules.is_empty() {
            bail!("Cannot merge zero DestinationRules");
        }

        let mut merged = rules[0].clone();

        for dr in &rules[1..] {
            merged.metadata = dr.metadata.clone();
            merged.host = dr.host.clone();

            // Later traffic policy wins entirely
            if dr.traffic_policy.is_some() {
                merged.traffic_policy = dr.traffic_policy.clone();
            }

            // Merge subsets: later definitions for the same name win
            for subset in &dr.subsets {
                if let Some(pos) = merged.subsets.iter().position(|s| s.name == subset.name) {
                    merged.subsets[pos] = subset.clone();
                } else {
                    merged.subsets.push(subset.clone());
                }
            }

            if !dr.export_to.is_empty() {
                merged.export_to = dr.export_to.clone();
            }
        }

        debug!(
            "Merged {} DestinationRules; total subsets: {}",
            rules.len(),
            merged.subsets.len()
        );

        Ok(merged)
    }

    /// Convert an Istio HttpRetryPolicy to the unified [`RetryPolicy`].
    pub fn extract_retry_policy_from_istio(route: &HttpRoute) -> Option<RetryPolicy> {
        let istio_retry = route.retries.as_ref()?;

        let per_try_timeout_ms =
            parse_duration_to_ms(&istio_retry.per_try_timeout).unwrap_or(2000);

        let retry_on: Vec<String> = istio_retry
            .retry_on
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let backoff_base_ms = std::cmp::max(25, per_try_timeout_ms / 10);
        let backoff_max_ms = per_try_timeout_ms;

        Some(RetryPolicy {
            max_retries: istio_retry.attempts,
            per_try_timeout_ms,
            retry_on,
            backoff_base_ms,
            backoff_max_ms,
        })
    }

    /// Extract a unified [`TimeoutPolicy`] from an HttpRoute's timeout and
    /// any associated traffic policy connection settings.
    pub fn extract_timeout_policy_from_istio(route: &HttpRoute) -> Option<TimeoutPolicy> {
        let timeout_str = route.timeout.as_ref()?;
        let request_timeout_ms = parse_duration_to_ms(timeout_str).unwrap_or(15000);

        // Idle timeout defaults to 5× the request timeout, capped at 5 min.
        let idle_timeout_ms = std::cmp::min(request_timeout_ms * 5, 300_000);

        // Connect timeout defaults to request_timeout / 3, at least 1 s.
        let connect_timeout_ms = std::cmp::max(1000, request_timeout_ms / 3);

        Some(TimeoutPolicy {
            request_timeout_ms,
            idle_timeout_ms,
            connect_timeout_ms,
        })
    }

    /// Resolve traffic policy precedence: workload-level > namespace-level >
    /// mesh-level. Fields present in a higher-precedence layer override
    /// those from lower layers.
    pub fn resolve_policy_precedence(
        workload: Option<&TrafficPolicy>,
        namespace: Option<&TrafficPolicy>,
        mesh: Option<&TrafficPolicy>,
    ) -> TrafficPolicy {
        let base = mesh.cloned().unwrap_or_default();
        let ns = namespace.cloned().unwrap_or_default();
        let wl = workload.cloned().unwrap_or_default();

        // Start from mesh, overlay namespace, then workload.
        let after_ns = merge_traffic_policies(&base, &ns);
        merge_traffic_policies(&after_ns, &wl)
    }
}

// ---------------------------------------------------------------------------
// Helper / free functions
// ---------------------------------------------------------------------------

/// Parse the `metadata` section from a Kubernetes-style YAML document.
pub fn parse_metadata(val: &serde_yaml::Value) -> Result<ObjectMeta> {
    let meta = val
        .get("metadata")
        .context("YAML document missing 'metadata' field")?;

    let name = meta
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let namespace = meta
        .get("namespace")
        .and_then(|v| v.as_str())
        .unwrap_or("default")
        .to_string();

    let uid = meta
        .get("uid")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let resource_version = meta
        .get("resourceVersion")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let labels = if let Some(lbl) = meta.get("labels") {
        parse_string_map(lbl)
    } else {
        IndexMap::new()
    };

    let annotations = if let Some(ann) = meta.get("annotations") {
        parse_string_map(ann)
    } else {
        IndexMap::new()
    };

    Ok(ObjectMeta {
        name,
        namespace,
        labels,
        annotations,
        uid,
        resource_version,
    })
}

/// Detect which kind of StringMatch a YAML value represents.
pub fn parse_string_match(val: &serde_yaml::Value) -> Option<StringMatch> {
    if let Some(exact) = val.get("exact").and_then(|v| v.as_str()) {
        return Some(StringMatch::Exact(exact.to_string()));
    }
    if let Some(prefix) = val.get("prefix").and_then(|v| v.as_str()) {
        return Some(StringMatch::Prefix(prefix.to_string()));
    }
    if let Some(regex) = val.get("regex").and_then(|v| v.as_str()) {
        return Some(StringMatch::Regex(regex.to_string()));
    }
    None
}

/// Parse a duration string such as `"5s"`, `"100ms"`, `"1m"`, `"0.5s"`,
/// `"1h"`, or `"1.5m"` into milliseconds.
pub fn parse_duration_to_ms(duration: &str) -> Result<u64> {
    let s = duration.trim();
    if s.is_empty() {
        bail!("Empty duration string");
    }

    // Try hours
    if let Some(num_str) = s.strip_suffix('h') {
        let val: f64 = num_str
            .parse()
            .with_context(|| format!("Invalid hour duration: {}", s))?;
        return Ok((val * 3_600_000.0) as u64);
    }

    // Try minutes (but not "ms")
    if s.ends_with('m') && !s.ends_with("ms") {
        let num_str = s.strip_suffix('m').unwrap();
        let val: f64 = num_str
            .parse()
            .with_context(|| format!("Invalid minute duration: {}", s))?;
        return Ok((val * 60_000.0) as u64);
    }

    // Try milliseconds
    if let Some(num_str) = s.strip_suffix("ms") {
        let val: f64 = num_str
            .parse()
            .with_context(|| format!("Invalid millisecond duration: {}", s))?;
        return Ok(val as u64);
    }

    // Try seconds
    if let Some(num_str) = s.strip_suffix('s') {
        let val: f64 = num_str
            .parse()
            .with_context(|| format!("Invalid second duration: {}", s))?;
        return Ok((val * 1000.0) as u64);
    }

    // Bare number -> treat as seconds for convenience
    if let Ok(val) = s.parse::<f64>() {
        warn!(
            "Duration '{}' has no unit suffix; assuming seconds",
            duration
        );
        return Ok((val * 1000.0) as u64);
    }

    bail!("Unrecognised duration format: {}", duration);
}

/// Parse the `http` array from a VirtualService spec.
pub fn parse_http_routes(val: &serde_yaml::Value) -> Result<Vec<HttpRoute>> {
    let arr = val
        .as_sequence()
        .context("Expected 'http' to be an array")?;

    arr.iter().map(parse_single_http_route).collect()
}

/// Parse a single HTTP route entry.
fn parse_single_http_route(val: &serde_yaml::Value) -> Result<HttpRoute> {
    let name = val.get("name").and_then(|v| v.as_str()).map(String::from);

    let match_conditions =
        if let Some(matches) = val.get("match").and_then(|v| v.as_sequence()) {
            matches
                .iter()
                .map(parse_http_match)
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };

    let route = if let Some(routes) = val.get("route").and_then(|v| v.as_sequence()) {
        routes
            .iter()
            .map(parse_http_route_destination)
            .collect::<Result<Vec<_>>>()?
    } else {
        Vec::new()
    };

    let retries = if let Some(r) = val.get("retries") {
        Some(parse_http_retry_policy(r)?)
    } else {
        None
    };

    let timeout = val
        .get("timeout")
        .and_then(|v| v.as_str())
        .map(String::from);

    let fault = if let Some(f) = val.get("fault") {
        Some(parse_fault_injection(f)?)
    } else {
        None
    };

    let mirror = if let Some(m) = val.get("mirror") {
        Some(parse_destination(m)?)
    } else {
        None
    };

    let headers = if let Some(h) = val.get("headers") {
        Some(parse_header_operations(h))
    } else {
        None
    };

    let rewrite = if let Some(rw) = val.get("rewrite") {
        Some(HttpRewrite {
            uri: rw.get("uri").and_then(|v| v.as_str()).map(String::from),
            authority: rw
                .get("authority")
                .and_then(|v| v.as_str())
                .map(String::from),
        })
    } else {
        None
    };

    Ok(HttpRoute {
        name,
        match_conditions,
        route,
        retries,
        timeout,
        fault,
        mirror,
        headers,
        rewrite,
    })
}

/// Parse a single HTTP match request from a YAML value.
fn parse_http_match(val: &serde_yaml::Value) -> Result<HttpMatchRequest> {
    let uri = val.get("uri").and_then(|v| parse_string_match(v));
    let method = val.get("method").and_then(|v| parse_string_match(v));
    let authority = val.get("authority").and_then(|v| parse_string_match(v));

    let port = val.get("port").and_then(|v| v.as_u64()).map(|p| p as u16);

    let headers = if let Some(hdr_map) = val.get("headers").and_then(|v| v.as_mapping()) {
        let mut map = IndexMap::new();
        for (k, v) in hdr_map {
            if let Some(key) = k.as_str() {
                if let Some(sm) = parse_string_match(v) {
                    map.insert(key.to_string(), sm);
                }
            }
        }
        map
    } else {
        IndexMap::new()
    };

    let source_labels = if let Some(sl) = val.get("sourceLabels") {
        parse_string_map(sl)
    } else {
        IndexMap::new()
    };

    Ok(HttpMatchRequest {
        uri,
        headers,
        method,
        authority,
        port,
        source_labels,
    })
}

/// Parse an HTTP route destination.
fn parse_http_route_destination(val: &serde_yaml::Value) -> Result<HttpRouteDestination> {
    let destination = val
        .get("destination")
        .context("Route entry missing 'destination'")?;
    let dest = parse_destination(destination)?;

    let weight = val
        .get("weight")
        .and_then(|v| v.as_u64())
        .unwrap_or(100) as u32;

    let headers = val.get("headers").map(|h| parse_header_operations(h));

    Ok(HttpRouteDestination {
        destination: dest,
        weight,
        headers,
    })
}

/// Parse a Destination value (host, port, subset).
fn parse_destination(val: &serde_yaml::Value) -> Result<Destination> {
    let host = val
        .get("host")
        .and_then(|v| v.as_str())
        .context("Destination missing 'host'")?
        .to_string();

    let port = if let Some(p) = val.get("port") {
        let number = p
            .get("number")
            .and_then(|v| v.as_u64())
            .context("Port missing 'number'")?;
        Some(PortSelector {
            number: number as u16,
        })
    } else {
        None
    };

    let subset = val
        .get("subset")
        .and_then(|v| v.as_str())
        .map(String::from);

    Ok(Destination {
        host,
        port,
        subset,
    })
}

/// Parse an Istio HttpRetryPolicy from YAML.
fn parse_http_retry_policy(val: &serde_yaml::Value) -> Result<HttpRetryPolicy> {
    let attempts = val
        .get("attempts")
        .and_then(|v| v.as_u64())
        .unwrap_or(2) as u32;

    let per_try_timeout = val
        .get("perTryTimeout")
        .and_then(|v| v.as_str())
        .unwrap_or("2s")
        .to_string();

    let retry_on = val
        .get("retryOn")
        .and_then(|v| v.as_str())
        .unwrap_or("connect-failure,refused-stream,unavailable,cancelled,retriable-status-codes")
        .to_string();

    let retry_remote_localities = val
        .get("retryRemoteLocalities")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    Ok(HttpRetryPolicy {
        attempts,
        per_try_timeout,
        retry_on,
        retry_remote_localities,
    })
}

/// Parse a FaultInjection block.
fn parse_fault_injection(val: &serde_yaml::Value) -> Result<FaultInjection> {
    let delay = if let Some(d) = val.get("delay") {
        let percentage = extract_percentage(d);
        let fixed_delay = d
            .get("fixedDelay")
            .and_then(|v| v.as_str())
            .unwrap_or("0s")
            .to_string();
        Some(FaultDelay {
            percentage,
            fixed_delay,
        })
    } else {
        None
    };

    let abort = if let Some(a) = val.get("abort") {
        let percentage = extract_percentage(a);
        let http_status = a
            .get("httpStatus")
            .and_then(|v| v.as_u64())
            .unwrap_or(500) as u16;
        Some(FaultAbort {
            percentage,
            http_status,
        })
    } else {
        None
    };

    Ok(FaultInjection { delay, abort })
}

/// Extract a percentage value, handling both `percentage.value` and direct `percentage` forms.
fn extract_percentage(val: &serde_yaml::Value) -> f64 {
    if let Some(pct) = val.get("percentage") {
        if let Some(v) = pct.get("value") {
            return v.as_f64().unwrap_or(0.0);
        }
        return pct.as_f64().unwrap_or(0.0);
    }
    // Some shorthand forms use `percent` directly.
    val.get("percent")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
}

/// Parse HeaderOperations from YAML.
fn parse_header_operations(val: &serde_yaml::Value) -> HeaderOperations {
    let request = val.get("request").unwrap_or(val);

    let set = if let Some(s) = request.get("set") {
        parse_string_map(s)
    } else {
        IndexMap::new()
    };

    let add = if let Some(a) = request.get("add") {
        parse_string_map(a)
    } else {
        IndexMap::new()
    };

    let remove = parse_string_array(request.get("remove").unwrap_or(&serde_yaml::Value::Null));

    HeaderOperations { set, add, remove }
}

/// Parse TCP routes array.
fn parse_tcp_routes(val: &serde_yaml::Value) -> Result<Vec<TcpRoute>> {
    let arr = val
        .as_sequence()
        .context("Expected 'tcp' to be an array")?;
    arr.iter().map(parse_single_tcp_route).collect()
}

fn parse_single_tcp_route(val: &serde_yaml::Value) -> Result<TcpRoute> {
    let match_conditions =
        if let Some(matches) = val.get("match").and_then(|v| v.as_sequence()) {
            matches
                .iter()
                .map(parse_tcp_match)
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };

    let route = if let Some(routes) = val.get("route").and_then(|v| v.as_sequence()) {
        routes
            .iter()
            .map(parse_tcp_route_destination)
            .collect::<Result<Vec<_>>>()?
    } else {
        Vec::new()
    };

    Ok(TcpRoute {
        match_conditions,
        route,
    })
}

fn parse_tcp_match(val: &serde_yaml::Value) -> Result<TcpMatchRequest> {
    let destination_subnets = parse_string_array(
        val.get("destinationSubnets")
            .unwrap_or(&serde_yaml::Value::Null),
    );
    let port = val.get("port").and_then(|v| v.as_u64()).map(|p| p as u16);
    let source_labels = if let Some(sl) = val.get("sourceLabels") {
        parse_string_map(sl)
    } else {
        IndexMap::new()
    };

    Ok(TcpMatchRequest {
        destination_subnets,
        port,
        source_labels,
    })
}

fn parse_tcp_route_destination(val: &serde_yaml::Value) -> Result<TcpRouteDestination> {
    let destination = val
        .get("destination")
        .context("TCP route entry missing 'destination'")?;
    let dest = parse_destination(destination)?;
    let weight = val
        .get("weight")
        .and_then(|v| v.as_u64())
        .unwrap_or(100) as u32;

    Ok(TcpRouteDestination {
        destination: dest,
        weight,
    })
}

/// Parse TLS routes array.
fn parse_tls_routes(val: &serde_yaml::Value) -> Result<Vec<TlsRoute>> {
    let arr = val
        .as_sequence()
        .context("Expected 'tls' to be an array")?;
    arr.iter().map(parse_single_tls_route).collect()
}

fn parse_single_tls_route(val: &serde_yaml::Value) -> Result<TlsRoute> {
    let match_conditions =
        if let Some(matches) = val.get("match").and_then(|v| v.as_sequence()) {
            matches
                .iter()
                .map(parse_tls_match)
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };

    let route = if let Some(routes) = val.get("route").and_then(|v| v.as_sequence()) {
        routes
            .iter()
            .map(parse_tcp_route_destination)
            .collect::<Result<Vec<_>>>()?
    } else {
        Vec::new()
    };

    Ok(TlsRoute {
        match_conditions,
        route,
    })
}

fn parse_tls_match(val: &serde_yaml::Value) -> Result<TlsMatchRequest> {
    let sni_hosts =
        parse_string_array(val.get("sniHosts").unwrap_or(&serde_yaml::Value::Null));
    let port = val.get("port").and_then(|v| v.as_u64()).map(|p| p as u16);
    let source_labels = if let Some(sl) = val.get("sourceLabels") {
        parse_string_map(sl)
    } else {
        IndexMap::new()
    };

    Ok(TlsMatchRequest {
        sni_hosts,
        port,
        source_labels,
    })
}

// ---- DestinationRule helpers -----------------------------------------------

/// Parse a TrafficPolicy from a YAML value tree.
pub fn parse_traffic_policy(val: &serde_yaml::Value) -> Result<TrafficPolicy> {
    let connection_pool = if let Some(cp) = val.get("connectionPool") {
        Some(parse_connection_pool(cp)?)
    } else {
        None
    };

    let load_balancer = if let Some(lb) = val.get("loadBalancer") {
        Some(parse_load_balancer(lb)?)
    } else {
        None
    };

    let outlier_detection = if let Some(od) = val.get("outlierDetection") {
        Some(parse_outlier_detection(od)?)
    } else {
        None
    };

    let tls = if let Some(t) = val.get("tls") {
        Some(parse_tls_settings(t)?)
    } else {
        None
    };

    let port_level_settings =
        if let Some(arr) = val.get("portLevelSettings").and_then(|v| v.as_sequence()) {
            arr.iter()
                .map(parse_port_traffic_policy)
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };

    Ok(TrafficPolicy {
        connection_pool,
        load_balancer,
        outlier_detection,
        tls,
        port_level_settings,
    })
}

fn parse_connection_pool(val: &serde_yaml::Value) -> Result<ConnectionPool> {
    let tcp = if let Some(t) = val.get("tcp") {
        Some(parse_tcp_settings(t)?)
    } else {
        None
    };

    let http = if let Some(h) = val.get("http") {
        Some(parse_http_settings(h)?)
    } else {
        None
    };

    Ok(ConnectionPool { tcp, http })
}

fn parse_tcp_settings(val: &serde_yaml::Value) -> Result<TcpSettings> {
    let max_connections = val
        .get("maxConnections")
        .and_then(|v| v.as_u64())
        .unwrap_or(1024) as u32;

    let connect_timeout = val
        .get("connectTimeout")
        .and_then(|v| v.as_str())
        .unwrap_or("10s")
        .to_string();

    let tcp_keepalive = if let Some(ka) = val.get("tcpKeepalive") {
        Some(TcpKeepalive {
            probes: ka
                .get("probes")
                .and_then(|v| v.as_u64())
                .unwrap_or(3) as u32,
            time: ka
                .get("time")
                .and_then(|v| v.as_str())
                .unwrap_or("7200s")
                .to_string(),
            interval: ka
                .get("interval")
                .and_then(|v| v.as_str())
                .unwrap_or("75s")
                .to_string(),
        })
    } else {
        None
    };

    Ok(TcpSettings {
        max_connections,
        connect_timeout,
        tcp_keepalive,
    })
}

fn parse_http_settings(val: &serde_yaml::Value) -> Result<HttpSettings> {
    let h2_upgrade_policy = val
        .get("h2UpgradePolicy")
        .and_then(|v| v.as_str())
        .unwrap_or("DEFAULT")
        .to_string();

    let max_requests_per_connection = val
        .get("maxRequestsPerConnection")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;

    let max_retries = val
        .get("maxRetries")
        .and_then(|v| v.as_u64())
        .unwrap_or(3) as u32;

    let idle_timeout = val
        .get("idleTimeout")
        .and_then(|v| v.as_str())
        .unwrap_or("1h")
        .to_string();

    Ok(HttpSettings {
        h2_upgrade_policy,
        max_requests_per_connection,
        max_retries,
        idle_timeout,
    })
}

fn parse_load_balancer(val: &serde_yaml::Value) -> Result<LoadBalancerSettings> {
    let simple = val
        .get("simple")
        .and_then(|v| v.as_str())
        .map(String::from);

    let consistent_hash = if let Some(ch) = val.get("consistentHash") {
        Some(ConsistentHashLB {
            http_header_name: ch
                .get("httpHeaderName")
                .and_then(|v| v.as_str())
                .map(String::from),
            minimum_ring_size: ch
                .get("minimumRingSize")
                .and_then(|v| v.as_u64())
                .unwrap_or(1024),
        })
    } else {
        None
    };

    Ok(LoadBalancerSettings {
        simple,
        consistent_hash,
    })
}

fn parse_outlier_detection(val: &serde_yaml::Value) -> Result<OutlierDetection> {
    Ok(OutlierDetection {
        consecutive_errors: val
            .get("consecutiveErrors")
            .or_else(|| val.get("consecutive5xxErrors"))
            .and_then(|v| v.as_u64())
            .unwrap_or(5) as u32,
        interval: val
            .get("interval")
            .and_then(|v| v.as_str())
            .unwrap_or("10s")
            .to_string(),
        base_ejection_time: val
            .get("baseEjectionTime")
            .and_then(|v| v.as_str())
            .unwrap_or("30s")
            .to_string(),
        max_ejection_percent: val
            .get("maxEjectionPercent")
            .and_then(|v| v.as_u64())
            .unwrap_or(10) as u32,
        min_health_percent: val
            .get("minHealthPercent")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32,
    })
}

fn parse_tls_settings(val: &serde_yaml::Value) -> Result<TlsSettings> {
    let mode = val
        .get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or("DISABLE")
        .to_string();

    let client_certificate = val
        .get("clientCertificate")
        .and_then(|v| v.as_str())
        .map(String::from);

    let private_key = val
        .get("privateKey")
        .and_then(|v| v.as_str())
        .map(String::from);

    let ca_certificates = val
        .get("caCertificates")
        .and_then(|v| v.as_str())
        .map(String::from);

    Ok(TlsSettings {
        mode,
        client_certificate,
        private_key,
        ca_certificates,
    })
}

fn parse_port_traffic_policy(val: &serde_yaml::Value) -> Result<PortTrafficPolicy> {
    let port_val = val
        .get("port")
        .context("portLevelSettings entry missing 'port'")?;
    let number = port_val
        .get("number")
        .and_then(|v| v.as_u64())
        .context("Port missing 'number'")?;
    let port = PortSelector {
        number: number as u16,
    };

    let traffic_policy = if let Some(tp) = val.get("trafficPolicy") {
        Some(Box::new(parse_traffic_policy(tp)?))
    } else {
        // If there is no nested "trafficPolicy" key, try to parse the
        // remaining fields at this level as inline policy fields.
        let has_any_policy_field = val.get("connectionPool").is_some()
            || val.get("loadBalancer").is_some()
            || val.get("outlierDetection").is_some()
            || val.get("tls").is_some();
        if has_any_policy_field {
            Some(Box::new(parse_traffic_policy(val)?))
        } else {
            None
        }
    };

    Ok(PortTrafficPolicy {
        port,
        traffic_policy,
    })
}

fn parse_subset(val: &serde_yaml::Value) -> Result<Subset> {
    let name = val
        .get("name")
        .and_then(|v| v.as_str())
        .context("Subset missing 'name'")?
        .to_string();

    let labels = if let Some(l) = val.get("labels") {
        parse_string_map(l)
    } else {
        IndexMap::new()
    };

    let traffic_policy = if let Some(tp) = val.get("trafficPolicy") {
        Some(parse_traffic_policy(tp)?)
    } else {
        None
    };

    Ok(Subset {
        name,
        labels,
        traffic_policy,
    })
}

// ---- Gateway helpers -------------------------------------------------------

fn parse_server(val: &serde_yaml::Value) -> Result<Server> {
    let port_val = val.get("port").context("Server missing 'port'")?;
    let port = GatewayPort {
        number: port_val
            .get("number")
            .and_then(|v| v.as_u64())
            .context("Server port missing 'number'")? as u16,
        name: port_val
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        protocol: port_val
            .get("protocol")
            .and_then(|v| v.as_str())
            .unwrap_or("HTTP")
            .to_string(),
    };

    let hosts = parse_string_array(val.get("hosts").unwrap_or(&serde_yaml::Value::Null));

    let tls = if let Some(t) = val.get("tls") {
        Some(GatewayTls {
            mode: t
                .get("mode")
                .and_then(|v| v.as_str())
                .unwrap_or("PASSTHROUGH")
                .to_string(),
            credential_name: t
                .get("credentialName")
                .and_then(|v| v.as_str())
                .map(String::from),
        })
    } else {
        None
    };

    Ok(Server { port, hosts, tls })
}

// ---- ServiceEntry helpers --------------------------------------------------

fn parse_service_entry_port(val: &serde_yaml::Value) -> Result<ServiceEntryPort> {
    Ok(ServiceEntryPort {
        number: val
            .get("number")
            .and_then(|v| v.as_u64())
            .context("ServiceEntry port missing 'number'")? as u16,
        name: val
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        protocol: val
            .get("protocol")
            .and_then(|v| v.as_str())
            .unwrap_or("TCP")
            .to_string(),
    })
}

fn parse_service_entry_endpoint(val: &serde_yaml::Value) -> Result<ServiceEntryEndpoint> {
    let address = val
        .get("address")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let ports = if let Some(p_map) = val.get("ports").and_then(|v| v.as_mapping()) {
        let mut map = IndexMap::new();
        for (k, v) in p_map {
            if let (Some(key), Some(port)) = (k.as_str(), v.as_u64()) {
                map.insert(key.to_string(), port as u16);
            }
        }
        map
    } else {
        IndexMap::new()
    };

    let labels = if let Some(l) = val.get("labels") {
        parse_string_map(l)
    } else {
        IndexMap::new()
    };

    Ok(ServiceEntryEndpoint {
        address,
        ports,
        labels,
    })
}

// ---- Generic YAML helpers --------------------------------------------------

/// Extract a `Vec<String>` from a YAML sequence of scalars.
fn parse_string_array(val: &serde_yaml::Value) -> Vec<String> {
    val.as_sequence()
        .map(|seq| {
            seq.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// Extract an `IndexMap<String, String>` from a YAML mapping.
fn parse_string_map(val: &serde_yaml::Value) -> IndexMap<String, String> {
    val.as_mapping()
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| {
                    let key = k.as_str()?;
                    let value = v.as_str().or_else(|| {
                        // Handle numeric/bool values by converting to string
                        v.as_i64()
                            .map(|_| ()) // we just need to know it's numeric
                            .or_else(|| v.as_bool().map(|_| ()))
                            .and(None)
                    });
                    // Fall back to serde_yaml's Display for non-string scalars
                    let val_str = match value {
                        Some(s) => s.to_string(),
                        None => {
                            if let Some(n) = v.as_i64() {
                                n.to_string()
                            } else if let Some(b) = v.as_bool() {
                                b.to_string()
                            } else if let Some(f) = v.as_f64() {
                                f.to_string()
                            } else {
                                return None;
                            }
                        }
                    };
                    Some((key.to_string(), val_str))
                })
                .collect()
        })
        .unwrap_or_default()
}

// ---- Policy merge helper ---------------------------------------------------

/// Merge two TrafficPolicy values; fields in `override_policy` take precedence.
fn merge_traffic_policies(base: &TrafficPolicy, override_policy: &TrafficPolicy) -> TrafficPolicy {
    TrafficPolicy {
        connection_pool: override_policy
            .connection_pool
            .clone()
            .or_else(|| base.connection_pool.clone()),
        load_balancer: override_policy
            .load_balancer
            .clone()
            .or_else(|| base.load_balancer.clone()),
        outlier_detection: override_policy
            .outlier_detection
            .clone()
            .or_else(|| base.outlier_detection.clone()),
        tls: override_policy.tls.clone().or_else(|| base.tls.clone()),
        port_level_settings: if override_policy.port_level_settings.is_empty() {
            base.port_level_settings.clone()
        } else {
            override_policy.port_level_settings.clone()
        },
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- VirtualService tests -----------------------------------------------

    #[test]
    fn test_parse_virtual_service_basic() {
        let yaml = r#"
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: reviews-vs
  namespace: bookinfo
spec:
  hosts:
    - reviews
  http:
    - route:
        - destination:
            host: reviews
            port:
              number: 9080
"#;
        let vs = IstioParser::parse_virtual_service(yaml).unwrap();
        assert_eq!(vs.metadata.name, "reviews-vs");
        assert_eq!(vs.metadata.namespace, "bookinfo");
        assert_eq!(vs.hosts, vec!["reviews"]);
        assert_eq!(vs.http_routes.len(), 1);
        assert_eq!(vs.http_routes[0].route[0].destination.host, "reviews");
        assert_eq!(
            vs.http_routes[0].route[0].destination.port.as_ref().unwrap().number,
            9080
        );
    }

    #[test]
    fn test_parse_virtual_service_with_retries() {
        let yaml = r#"
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: retry-vs
  namespace: default
spec:
  hosts:
    - ratings
  http:
    - route:
        - destination:
            host: ratings
      retries:
        attempts: 3
        perTryTimeout: 2s
        retryOn: "gateway-error,connect-failure"
"#;
        let vs = IstioParser::parse_virtual_service(yaml).unwrap();
        let route = &vs.http_routes[0];
        let retries = route.retries.as_ref().unwrap();
        assert_eq!(retries.attempts, 3);
        assert_eq!(retries.per_try_timeout, "2s");
        assert_eq!(retries.retry_on, "gateway-error,connect-failure");
    }

    #[test]
    fn test_parse_virtual_service_with_fault_injection() {
        let yaml = r#"
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: fault-vs
  namespace: default
spec:
  hosts:
    - ratings
  http:
    - fault:
        delay:
          percentage:
            value: 10.0
          fixedDelay: 5s
        abort:
          percentage:
            value: 5.0
          httpStatus: 503
      route:
        - destination:
            host: ratings
"#;
        let vs = IstioParser::parse_virtual_service(yaml).unwrap();
        let fault = vs.http_routes[0].fault.as_ref().unwrap();
        let delay = fault.delay.as_ref().unwrap();
        assert!((delay.percentage - 10.0).abs() < f64::EPSILON);
        assert_eq!(delay.fixed_delay, "5s");
        let abort = fault.abort.as_ref().unwrap();
        assert!((abort.percentage - 5.0).abs() < f64::EPSILON);
        assert_eq!(abort.http_status, 503);
    }

    #[test]
    fn test_parse_virtual_service_weighted_routing() {
        let yaml = r#"
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: weighted-vs
  namespace: default
spec:
  hosts:
    - reviews
  http:
    - route:
        - destination:
            host: reviews
            subset: v1
          weight: 80
        - destination:
            host: reviews
            subset: v2
          weight: 20
"#;
        let vs = IstioParser::parse_virtual_service(yaml).unwrap();
        let route = &vs.http_routes[0];
        assert_eq!(route.route.len(), 2);
        assert_eq!(route.route[0].weight, 80);
        assert_eq!(route.route[0].destination.subset.as_deref(), Some("v1"));
        assert_eq!(route.route[1].weight, 20);
        assert_eq!(route.route[1].destination.subset.as_deref(), Some("v2"));
    }

    // ---- DestinationRule tests -----------------------------------------------

    #[test]
    fn test_parse_destination_rule_basic() {
        let yaml = r#"
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: reviews-dr
  namespace: bookinfo
spec:
  host: reviews
  subsets:
    - name: v1
      labels:
        version: v1
    - name: v2
      labels:
        version: v2
"#;
        let dr = IstioParser::parse_destination_rule(yaml).unwrap();
        assert_eq!(dr.metadata.name, "reviews-dr");
        assert_eq!(dr.host, "reviews");
        assert_eq!(dr.subsets.len(), 2);
        assert_eq!(dr.subsets[0].name, "v1");
        assert_eq!(dr.subsets[0].labels.get("version").unwrap(), "v1");
        assert_eq!(dr.subsets[1].name, "v2");
    }

    #[test]
    fn test_parse_destination_rule_with_outlier_detection() {
        let yaml = r#"
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: od-dr
  namespace: default
spec:
  host: httpbin
  trafficPolicy:
    outlierDetection:
      consecutiveErrors: 7
      interval: 5s
      baseEjectionTime: 15s
      maxEjectionPercent: 50
      minHealthPercent: 30
"#;
        let dr = IstioParser::parse_destination_rule(yaml).unwrap();
        let od = dr
            .traffic_policy
            .as_ref()
            .unwrap()
            .outlier_detection
            .as_ref()
            .unwrap();
        assert_eq!(od.consecutive_errors, 7);
        assert_eq!(od.interval, "5s");
        assert_eq!(od.base_ejection_time, "15s");
        assert_eq!(od.max_ejection_percent, 50);
        assert_eq!(od.min_health_percent, 30);
    }

    #[test]
    fn test_parse_destination_rule_with_connection_pool() {
        let yaml = r#"
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: pool-dr
  namespace: default
spec:
  host: httpbin
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
        connectTimeout: 30ms
      http:
        h2UpgradePolicy: UPGRADE
        maxRequestsPerConnection: 10
        maxRetries: 5
        idleTimeout: 30s
"#;
        let dr = IstioParser::parse_destination_rule(yaml).unwrap();
        let cp = dr
            .traffic_policy
            .as_ref()
            .unwrap()
            .connection_pool
            .as_ref()
            .unwrap();
        let tcp = cp.tcp.as_ref().unwrap();
        assert_eq!(tcp.max_connections, 100);
        assert_eq!(tcp.connect_timeout, "30ms");
        let http = cp.http.as_ref().unwrap();
        assert_eq!(http.h2_upgrade_policy, "UPGRADE");
        assert_eq!(http.max_requests_per_connection, 10);
        assert_eq!(http.max_retries, 5);
        assert_eq!(http.idle_timeout, "30s");
    }

    // ---- Gateway test -------------------------------------------------------

    #[test]
    fn test_parse_gateway() {
        let yaml = r#"
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: bookinfo-gw
  namespace: bookinfo
spec:
  selector:
    istio: ingressgateway
  servers:
    - port:
        number: 443
        name: https
        protocol: HTTPS
      hosts:
        - "*.bookinfo.com"
      tls:
        mode: SIMPLE
        credentialName: bookinfo-cert
    - port:
        number: 80
        name: http
        protocol: HTTP
      hosts:
        - "bookinfo.com"
"#;
        let gw = IstioParser::parse_gateway(yaml).unwrap();
        assert_eq!(gw.metadata.name, "bookinfo-gw");
        assert_eq!(gw.selector.get("istio").unwrap(), "ingressgateway");
        assert_eq!(gw.servers.len(), 2);

        let https = &gw.servers[0];
        assert_eq!(https.port.number, 443);
        assert_eq!(https.port.protocol, "HTTPS");
        let tls = https.tls.as_ref().unwrap();
        assert_eq!(tls.mode, "SIMPLE");
        assert_eq!(tls.credential_name.as_deref(), Some("bookinfo-cert"));

        let http = &gw.servers[1];
        assert_eq!(http.port.number, 80);
        assert!(http.tls.is_none());
    }

    // ---- ServiceEntry test --------------------------------------------------

    #[test]
    fn test_parse_service_entry() {
        let yaml = r#"
apiVersion: networking.istio.io/v1beta1
kind: ServiceEntry
metadata:
  name: external-api
  namespace: default
spec:
  hosts:
    - api.external.com
  location: MESH_EXTERNAL
  ports:
    - number: 443
      name: https
      protocol: HTTPS
  resolution: DNS
  endpoints:
    - address: 203.0.113.10
      ports:
        https: 443
      labels:
        region: us-east
"#;
        let se = IstioParser::parse_service_entry(yaml).unwrap();
        assert_eq!(se.metadata.name, "external-api");
        assert_eq!(se.hosts, vec!["api.external.com"]);
        assert_eq!(se.location, "MESH_EXTERNAL");
        assert_eq!(se.resolution, "DNS");
        assert_eq!(se.ports.len(), 1);
        assert_eq!(se.ports[0].number, 443);
        assert_eq!(se.endpoints.len(), 1);
        assert_eq!(se.endpoints[0].address, "203.0.113.10");
        assert_eq!(*se.endpoints[0].ports.get("https").unwrap(), 443u16);
        assert_eq!(
            se.endpoints[0].labels.get("region").unwrap(),
            "us-east"
        );
    }

    // ---- Merge tests --------------------------------------------------------

    #[test]
    fn test_merge_virtual_services() {
        let yaml1 = r#"
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: vs-a
  namespace: default
spec:
  hosts:
    - reviews
  http:
    - name: route-a
      route:
        - destination:
            host: reviews
            subset: v1
"#;
        let yaml2 = r#"
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: vs-b
  namespace: default
spec:
  hosts:
    - reviews
  http:
    - name: route-b
      route:
        - destination:
            host: reviews
            subset: v2
"#;
        let vs1 = IstioParser::parse_virtual_service(yaml1).unwrap();
        let vs2 = IstioParser::parse_virtual_service(yaml2).unwrap();

        let merged = IstioParser::merge_virtual_services(&[vs1, vs2]).unwrap();
        // Later metadata wins
        assert_eq!(merged.metadata.name, "vs-b");
        // Routes from both are present, later routes first
        assert_eq!(merged.http_routes.len(), 2);
        assert_eq!(merged.http_routes[0].name.as_deref(), Some("route-b"));
        assert_eq!(merged.http_routes[1].name.as_deref(), Some("route-a"));
    }

    #[test]
    fn test_merge_destination_rules() {
        let yaml1 = r#"
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: dr-a
  namespace: default
spec:
  host: reviews
  subsets:
    - name: v1
      labels:
        version: v1
"#;
        let yaml2 = r#"
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: dr-b
  namespace: default
spec:
  host: reviews
  subsets:
    - name: v1
      labels:
        version: v1-canary
    - name: v2
      labels:
        version: v2
"#;
        let dr1 = IstioParser::parse_destination_rule(yaml1).unwrap();
        let dr2 = IstioParser::parse_destination_rule(yaml2).unwrap();

        let merged = IstioParser::merge_destination_rules(&[dr1, dr2]).unwrap();
        assert_eq!(merged.metadata.name, "dr-b");
        // v1 is overridden, v2 is added
        assert_eq!(merged.subsets.len(), 2);
        assert_eq!(
            merged.subsets[0].labels.get("version").unwrap(),
            "v1-canary"
        );
        assert_eq!(merged.subsets[1].name, "v2");
    }

    // ---- Policy extraction tests -------------------------------------------

    #[test]
    fn test_extract_retry_policy() {
        let route = HttpRoute {
            name: Some("retry-test".to_string()),
            match_conditions: vec![],
            route: vec![],
            retries: Some(HttpRetryPolicy {
                attempts: 5,
                per_try_timeout: "3s".to_string(),
                retry_on: "5xx,reset".to_string(),
                retry_remote_localities: true,
            }),
            timeout: None,
            fault: None,
            mirror: None,
            headers: None,
            rewrite: None,
        };

        let policy = IstioParser::extract_retry_policy_from_istio(&route).unwrap();
        assert_eq!(policy.max_retries, 5);
        assert_eq!(policy.per_try_timeout_ms, 3000);
        assert_eq!(policy.retry_on, vec!["5xx", "reset"]);
        assert_eq!(policy.backoff_base_ms, 300); // 3000 / 10
        assert_eq!(policy.backoff_max_ms, 3000);
    }

    #[test]
    fn test_extract_timeout_policy() {
        let route = HttpRoute {
            name: None,
            match_conditions: vec![],
            route: vec![],
            retries: None,
            timeout: Some("10s".to_string()),
            fault: None,
            mirror: None,
            headers: None,
            rewrite: None,
        };

        let policy = IstioParser::extract_timeout_policy_from_istio(&route).unwrap();
        assert_eq!(policy.request_timeout_ms, 10_000);
        assert_eq!(policy.idle_timeout_ms, 50_000);
        assert_eq!(policy.connect_timeout_ms, 3333);
    }

    // ---- Duration parsing test -----------------------------------------------

    #[test]
    fn test_parse_duration() {
        assert_eq!(parse_duration_to_ms("5s").unwrap(), 5000);
        assert_eq!(parse_duration_to_ms("100ms").unwrap(), 100);
        assert_eq!(parse_duration_to_ms("1m").unwrap(), 60_000);
        assert_eq!(parse_duration_to_ms("0.5s").unwrap(), 500);
        assert_eq!(parse_duration_to_ms("1h").unwrap(), 3_600_000);
        assert_eq!(parse_duration_to_ms("1.5m").unwrap(), 90_000);
        assert!(parse_duration_to_ms("").is_err());
    }

    // ---- Precedence test -----------------------------------------------------

    #[test]
    fn test_resolve_policy_precedence() {
        let mesh = TrafficPolicy {
            outlier_detection: Some(OutlierDetection {
                consecutive_errors: 5,
                interval: "10s".to_string(),
                base_ejection_time: "30s".to_string(),
                max_ejection_percent: 10,
                min_health_percent: 0,
            }),
            load_balancer: Some(LoadBalancerSettings {
                simple: Some("ROUND_ROBIN".to_string()),
                consistent_hash: None,
            }),
            ..Default::default()
        };

        let namespace = TrafficPolicy {
            outlier_detection: Some(OutlierDetection {
                consecutive_errors: 3,
                interval: "5s".to_string(),
                base_ejection_time: "15s".to_string(),
                max_ejection_percent: 20,
                min_health_percent: 50,
            }),
            ..Default::default()
        };

        let workload = TrafficPolicy {
            tls: Some(TlsSettings {
                mode: "ISTIO_MUTUAL".to_string(),
                client_certificate: None,
                private_key: None,
                ca_certificates: None,
            }),
            ..Default::default()
        };

        let resolved =
            IstioParser::resolve_policy_precedence(Some(&workload), Some(&namespace), Some(&mesh));

        // TLS from workload
        assert_eq!(resolved.tls.as_ref().unwrap().mode, "ISTIO_MUTUAL");
        // Outlier from namespace (overrides mesh)
        assert_eq!(
            resolved
                .outlier_detection
                .as_ref()
                .unwrap()
                .consecutive_errors,
            3
        );
        // Load balancer from mesh (not overridden)
        assert_eq!(
            resolved.load_balancer.as_ref().unwrap().simple.as_deref(),
            Some("ROUND_ROBIN")
        );
    }

    // ---- StringMatch parsing test -------------------------------------------

    #[test]
    fn test_parse_string_match() {
        let exact_val: serde_yaml::Value =
            serde_yaml::from_str(r#"{ exact: "/api/v1" }"#).unwrap();
        assert_eq!(
            parse_string_match(&exact_val),
            Some(StringMatch::Exact("/api/v1".to_string()))
        );

        let prefix_val: serde_yaml::Value =
            serde_yaml::from_str(r#"{ prefix: "/api/" }"#).unwrap();
        assert_eq!(
            parse_string_match(&prefix_val),
            Some(StringMatch::Prefix("/api/".to_string()))
        );

        let regex_val: serde_yaml::Value =
            serde_yaml::from_str(r#"{ regex: "^/api/v[0-9]+" }"#).unwrap();
        assert_eq!(
            parse_string_match(&regex_val),
            Some(StringMatch::Regex("^/api/v[0-9]+".to_string()))
        );

        let empty_val: serde_yaml::Value = serde_yaml::from_str("{}").unwrap();
        assert_eq!(parse_string_match(&empty_val), None);
    }
}
