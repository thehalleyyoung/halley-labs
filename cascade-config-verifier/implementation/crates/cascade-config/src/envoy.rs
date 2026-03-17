//! Envoy xDS configuration parsing.
//!
//! Parses Envoy Cluster, Listener, and Route configuration from JSON
//! (as produced by `envoy config_dump` or xDS gRPC responses) into
//! strongly-typed Rust structs.  Also provides conversion helpers that
//! map Envoy-specific retry / timeout policies to the unified
//! [`crate::RetryPolicy`] and [`crate::TimeoutPolicy`] types used
//! across the cascade-config crate.

use anyhow::{Context, Result};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::{RetryPolicy, TimeoutPolicy};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Envoy Cluster (CDS) representation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnvoyCluster {
    pub name: String,
    #[serde(default)]
    pub cluster_type: String,
    #[serde(default)]
    pub connect_timeout: String,
    #[serde(default)]
    pub lb_policy: String,
    pub load_assignment: Option<ClusterLoadAssignment>,
    pub circuit_breakers: Option<CircuitBreakers>,
    pub outlier_detection: Option<EnvoyOutlierDetection>,
    #[serde(default)]
    pub health_checks: Vec<HealthCheck>,
    pub transport_socket: Option<TransportSocket>,
}

/// Cluster load assignment (EDS endpoints).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ClusterLoadAssignment {
    pub cluster_name: String,
    #[serde(default)]
    pub endpoints: Vec<LocalityLbEndpoints>,
}

/// A set of endpoints scoped to a locality.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LocalityLbEndpoints {
    pub locality: Option<Locality>,
    #[serde(default)]
    pub lb_endpoints: Vec<LbEndpoint>,
    #[serde(default)]
    pub priority: u32,
}

/// Locality metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Locality {
    #[serde(default)]
    pub region: String,
    #[serde(default)]
    pub zone: String,
    #[serde(default)]
    pub sub_zone: String,
}

/// A single load-balanced endpoint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LbEndpoint {
    pub address: String,
    pub port: u16,
    #[serde(default)]
    pub health_status: String,
    #[serde(default)]
    pub metadata: IndexMap<String, String>,
}

/// Circuit-breaker thresholds.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CircuitBreakers {
    #[serde(default)]
    pub thresholds: Vec<Threshold>,
}

/// A single priority-level threshold.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Threshold {
    #[serde(default)]
    pub priority: String,
    #[serde(default = "default_max_connections")]
    pub max_connections: u32,
    #[serde(default = "default_max_pending")]
    pub max_pending_requests: u32,
    #[serde(default = "default_max_requests")]
    pub max_requests: u32,
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    #[serde(default)]
    pub track_remaining: bool,
}

fn default_max_connections() -> u32 {
    1024
}
fn default_max_pending() -> u32 {
    1024
}
fn default_max_requests() -> u32 {
    1024
}
fn default_max_retries() -> u32 {
    3
}

/// Outlier detection settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnvoyOutlierDetection {
    #[serde(default = "default_consecutive_5xx")]
    pub consecutive_5xx: u32,
    #[serde(default = "default_consecutive_gw_failure")]
    pub consecutive_gateway_failure: u32,
    #[serde(default = "default_interval")]
    pub interval: String,
    #[serde(default = "default_base_ejection_time")]
    pub base_ejection_time: String,
    #[serde(default = "default_max_ejection_percent")]
    pub max_ejection_percent: u32,
    #[serde(default = "default_enforcing_consecutive_5xx")]
    pub enforcing_consecutive_5xx: u32,
    #[serde(default)]
    pub enforcing_success_rate: u32,
    #[serde(default = "default_sr_min_hosts")]
    pub success_rate_minimum_hosts: u32,
    #[serde(default = "default_sr_request_volume")]
    pub success_rate_request_volume: u32,
    #[serde(default = "default_sr_stdev_factor")]
    pub success_rate_stdev_factor: u32,
}

fn default_consecutive_5xx() -> u32 {
    5
}
fn default_consecutive_gw_failure() -> u32 {
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
fn default_enforcing_consecutive_5xx() -> u32 {
    100
}
fn default_sr_min_hosts() -> u32 {
    5
}
fn default_sr_request_volume() -> u32 {
    100
}
fn default_sr_stdev_factor() -> u32 {
    1900
}

/// Health check configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HealthCheck {
    #[serde(default = "default_hc_timeout")]
    pub timeout: String,
    #[serde(default = "default_hc_interval")]
    pub interval: String,
    #[serde(default = "default_hc_unhealthy_threshold")]
    pub unhealthy_threshold: u32,
    #[serde(default = "default_hc_healthy_threshold")]
    pub healthy_threshold: u32,
    pub health_checker: HealthChecker,
}

fn default_hc_timeout() -> String {
    "5s".to_string()
}
fn default_hc_interval() -> String {
    "10s".to_string()
}
fn default_hc_unhealthy_threshold() -> u32 {
    3
}
fn default_hc_healthy_threshold() -> u32 {
    3
}

/// Discriminated union of health-check strategies.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum HealthChecker {
    HttpHealthCheck { path: String, host: String },
    TcpHealthCheck { send: Option<String> },
    GrpcHealthCheck { service_name: String },
}

/// Envoy transport socket configuration (TLS).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TransportSocket {
    pub name: String,
    pub tls_context: Option<TlsContext>,
}

/// Upstream/downstream TLS context.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TlsContext {
    pub common_tls_context: CommonTlsContext,
    pub sni: Option<String>,
}

/// Shared TLS settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CommonTlsContext {
    #[serde(default)]
    pub alpn_protocols: Vec<String>,
    #[serde(default)]
    pub tls_certificates: Vec<TlsCertificate>,
}

/// A TLS certificate pair.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TlsCertificate {
    #[serde(default)]
    pub certificate_chain: String,
    #[serde(default)]
    pub private_key: String,
}

// ---------------------------------------------------------------------------
// Listener types
// ---------------------------------------------------------------------------

/// Envoy Listener (LDS) representation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnvoyListener {
    pub name: String,
    pub address: ListenerAddress,
    #[serde(default)]
    pub filter_chains: Vec<FilterChain>,
    #[serde(default)]
    pub listener_filters: Vec<ListenerFilter>,
}

/// Listener bind address wrapper.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ListenerAddress {
    pub socket_address: SocketAddress,
}

/// IP + port.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SocketAddress {
    pub address: String,
    pub port_value: u16,
}

/// A single filter chain within a listener.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FilterChain {
    pub filter_chain_match: Option<FilterChainMatch>,
    #[serde(default)]
    pub filters: Vec<Filter>,
    pub transport_socket: Option<TransportSocket>,
}

/// Match criteria selecting a filter chain.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FilterChainMatch {
    #[serde(default)]
    pub server_names: Vec<String>,
    pub transport_protocol: Option<String>,
    #[serde(default)]
    pub application_protocols: Vec<String>,
}

/// A network or HTTP filter.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Filter {
    pub name: String,
    pub typed_config: Option<serde_json::Value>,
}

/// Listener-level filter (e.g. TLS inspector).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ListenerFilter {
    pub name: String,
    pub typed_config: Option<serde_json::Value>,
}

/// HTTP-level filter inside the HTTP connection manager.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HttpFilter {
    pub name: String,
    #[serde(default = "default_json_null")]
    pub config: serde_json::Value,
}

fn default_json_null() -> serde_json::Value {
    serde_json::Value::Null
}

// ---------------------------------------------------------------------------
// Route types
// ---------------------------------------------------------------------------

/// Envoy Route Configuration (RDS).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RouteConfiguration {
    pub name: String,
    #[serde(default)]
    pub virtual_hosts: Vec<VirtualHost>,
}

/// A virtual host with domain matching and routes.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VirtualHost {
    pub name: String,
    #[serde(default)]
    pub domains: Vec<String>,
    #[serde(default)]
    pub routes: Vec<Route>,
    pub retry_policy: Option<EnvoyRetryPolicy>,
    #[serde(default)]
    pub request_headers_to_add: Vec<HeaderValueOption>,
}

/// Header value to add to requests.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HeaderValueOption {
    pub header_name: String,
    pub header_value: String,
    #[serde(default = "default_true")]
    pub append: bool,
}

fn default_true() -> bool {
    true
}

/// A single route entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Route {
    pub name: Option<String>,
    pub match_pattern: RouteMatch,
    pub route_action: Option<RouteAction>,
    pub direct_response: Option<DirectResponseAction>,
}

/// Match criteria for a route.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RouteMatch {
    pub prefix: Option<String>,
    pub path: Option<String>,
    pub safe_regex: Option<String>,
    #[serde(default)]
    pub headers: Vec<HeaderMatcher>,
}

/// Header-based route match.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HeaderMatcher {
    pub name: String,
    pub exact_match: Option<String>,
    pub prefix_match: Option<String>,
    pub present_match: Option<bool>,
}

/// Routing action (forward to cluster).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RouteAction {
    #[serde(default)]
    pub cluster: String,
    pub timeout: Option<String>,
    pub retry_policy: Option<EnvoyRetryPolicy>,
    pub host_rewrite_literal: Option<String>,
    pub prefix_rewrite: Option<String>,
    pub weighted_clusters: Option<WeightedClusters>,
}

/// Weighted cluster routing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WeightedClusters {
    #[serde(default)]
    pub clusters: Vec<WeightedClusterEntry>,
    #[serde(default = "default_total_weight")]
    pub total_weight: u32,
}

fn default_total_weight() -> u32 {
    100
}

/// A single weighted cluster entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WeightedClusterEntry {
    pub name: String,
    pub weight: u32,
}

/// Direct response (no upstream).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DirectResponseAction {
    pub status: u32,
    pub body: Option<String>,
}

/// Envoy retry policy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnvoyRetryPolicy {
    #[serde(default)]
    pub retry_on: String,
    #[serde(default)]
    pub num_retries: u32,
    pub per_try_timeout: Option<String>,
    pub retry_back_off: Option<RetryBackOff>,
    #[serde(default)]
    pub retry_host_predicate: Vec<RetryHostPredicate>,
    #[serde(default)]
    pub host_selection_retry_max_attempts: u32,
}

/// Back-off parameters for retries.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RetryBackOff {
    pub base_interval: String,
    pub max_interval: Option<String>,
}

/// Retry host predicate entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RetryHostPredicate {
    pub name: String,
}

// ---------------------------------------------------------------------------
// Envoy resource enum (for xDS parsing)
// ---------------------------------------------------------------------------

/// A tagged Envoy xDS resource.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EnvoyResource {
    Cluster(EnvoyCluster),
    Listener(EnvoyListener),
    RouteConfig(RouteConfiguration),
}

// ---------------------------------------------------------------------------
// EnvoyParser
// ---------------------------------------------------------------------------

/// Stateless parser that converts Envoy JSON configuration into typed structs.
#[derive(Debug, Clone, Default)]
pub struct EnvoyParser;

impl EnvoyParser {
    pub fn new() -> Self {
        Self
    }

    // -- top-level parse functions ------------------------------------------

    /// Parse an Envoy Cluster from a JSON value.
    pub fn parse_cluster(&self, config: &serde_json::Value) -> Result<EnvoyCluster> {
        let name = json_str(config, "name").context("cluster missing 'name'")?;
        let cluster_type = json_str(config, "type")
            .or_else(|_| json_str(config, "cluster_type"))
            .unwrap_or_else(|_| "EDS".to_string());
        let connect_timeout = json_str(config, "connect_timeout").unwrap_or_else(|_| "5s".to_string());
        let lb_policy = json_str(config, "lb_policy").unwrap_or_else(|_| "ROUND_ROBIN".to_string());

        let load_assignment = config.get("load_assignment").and_then(|la| {
            let cluster_name = la
                .get("cluster_name")
                .and_then(|v| v.as_str())
                .unwrap_or(&name)
                .to_string();
            let endpoints = la
                .get("endpoints")
                .and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|ep| parse_locality_lb_endpoints(ep)).collect())
                .unwrap_or_default();
            Some(ClusterLoadAssignment {
                cluster_name,
                endpoints,
            })
        });

        let circuit_breakers = config.get("circuit_breakers").and_then(|v| parse_circuit_breakers(v));
        let outlier_detection = config.get("outlier_detection").and_then(|v| parse_outlier_detection(v));

        let health_checks = config
            .get("health_checks")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|hc| parse_health_check(hc)).collect())
            .unwrap_or_default();

        let transport_socket = config.get("transport_socket").and_then(|v| parse_transport_socket(v));

        Ok(EnvoyCluster {
            name,
            cluster_type,
            connect_timeout,
            lb_policy,
            load_assignment,
            circuit_breakers,
            outlier_detection,
            health_checks,
            transport_socket,
        })
    }

    /// Parse an Envoy Listener from a JSON value.
    pub fn parse_listener(&self, config: &serde_json::Value) -> Result<EnvoyListener> {
        let name = json_str(config, "name").context("listener missing 'name'")?;

        let address = config
            .get("address")
            .context("listener missing 'address'")?;
        let socket_address_val = address
            .get("socket_address")
            .context("address missing 'socket_address'")?;
        let addr = json_str(socket_address_val, "address").unwrap_or_else(|_| "0.0.0.0".to_string());
        let port_value = socket_address_val
            .get("port_value")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u16;

        let filter_chains = parse_filter_chains(config)?;
        let listener_filters = parse_listener_filters(config);

        Ok(EnvoyListener {
            name,
            address: ListenerAddress {
                socket_address: SocketAddress {
                    address: addr,
                    port_value,
                },
            },
            filter_chains,
            listener_filters,
        })
    }

    /// Parse an Envoy RouteConfiguration from a JSON value.
    pub fn parse_route_config(&self, config: &serde_json::Value) -> Result<RouteConfiguration> {
        let name = json_str(config, "name").unwrap_or_default();
        let virtual_hosts = parse_virtual_hosts(config)?;
        Ok(RouteConfiguration {
            name,
            virtual_hosts,
        })
    }

    /// Parse a full xDS config dump, returning all recognised resources.
    pub fn parse_xds_config(&self, config: &serde_json::Value) -> Result<Vec<EnvoyResource>> {
        let mut resources = Vec::new();

        // Static / dynamic clusters
        for key in &[
            "static_clusters",
            "dynamic_active_clusters",
            "clusters",
        ] {
            if let Some(arr) = config.get(*key).and_then(|v| v.as_array()) {
                for entry in arr {
                    // Entries may be wrapped: { "cluster": { ... } }
                    let cluster_val = entry.get("cluster").unwrap_or(entry);
                    if let Ok(cluster) = self.parse_cluster(cluster_val) {
                        resources.push(EnvoyResource::Cluster(cluster));
                    }
                }
            }
        }

        // Static / dynamic listeners
        for key in &[
            "static_listeners",
            "dynamic_listeners",
            "listeners",
        ] {
            if let Some(arr) = config.get(*key).and_then(|v| v.as_array()) {
                for entry in arr {
                    let listener_val = entry
                        .get("active_state")
                        .and_then(|s| s.get("listener"))
                        .or_else(|| entry.get("listener"))
                        .unwrap_or(entry);
                    if let Ok(listener) = self.parse_listener(listener_val) {
                        resources.push(EnvoyResource::Listener(listener));
                    }
                }
            }
        }

        // Static / dynamic route configs
        for key in &[
            "static_route_configs",
            "dynamic_route_configs",
            "route_configs",
        ] {
            if let Some(arr) = config.get(*key).and_then(|v| v.as_array()) {
                for entry in arr {
                    let rc_val = entry.get("route_config").unwrap_or(entry);
                    if let Ok(rc) = self.parse_route_config(rc_val) {
                        resources.push(EnvoyResource::RouteConfig(rc));
                    }
                }
            }
        }

        // Handle the top-level `configs` array used by `config_dump`.
        if let Some(configs) = config.get("configs").and_then(|v| v.as_array()) {
            for cfg in configs {
                let sub = self.parse_xds_config(cfg)?;
                resources.extend(sub);
            }
        }

        Ok(resources)
    }

    // -- conversion helpers -------------------------------------------------

    /// Convert an [`EnvoyRetryPolicy`] into the unified [`RetryPolicy`].
    pub fn extract_retry_policy_from_envoy(policy: &EnvoyRetryPolicy) -> RetryPolicy {
        let retry_on: Vec<String> = policy
            .retry_on
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let per_try_timeout_ms = policy
            .per_try_timeout
            .as_deref()
            .map(parse_envoy_duration_to_ms)
            .unwrap_or(1000);

        let (backoff_base_ms, backoff_max_ms) = match &policy.retry_back_off {
            Some(bo) => {
                let base = parse_envoy_duration_to_ms(&bo.base_interval);
                let max = bo
                    .max_interval
                    .as_deref()
                    .map(parse_envoy_duration_to_ms)
                    .unwrap_or(base * 10);
                (base, max)
            }
            None => (25, 250),
        };

        RetryPolicy {
            max_retries: policy.num_retries,
            per_try_timeout_ms,
            retry_on,
            backoff_base_ms,
            backoff_max_ms,
        }
    }

    /// Derive a [`TimeoutPolicy`] from a [`RouteAction`].
    pub fn extract_timeout_policy_from_envoy(action: &RouteAction) -> TimeoutPolicy {
        let request_timeout_ms = action
            .timeout
            .as_deref()
            .map(parse_envoy_duration_to_ms)
            .unwrap_or(15000);

        // Envoy does not embed idle/connect timeouts in RouteAction directly;
        // we fall back to sensible defaults.
        TimeoutPolicy {
            request_timeout_ms,
            idle_timeout_ms: 300_000,
            connect_timeout_ms: 5000,
        }
    }
}

// ---------------------------------------------------------------------------
// Standalone helper functions
// ---------------------------------------------------------------------------

/// Parse Envoy circuit-breaker config into typed struct.
pub fn parse_circuit_breakers(val: &serde_json::Value) -> Option<CircuitBreakers> {
    let thresholds_arr = val.get("thresholds").and_then(|v| v.as_array())?;
    let thresholds: Vec<Threshold> = thresholds_arr
        .iter()
        .map(|t| {
            Threshold {
                priority: t
                    .get("priority")
                    .and_then(|v| v.as_str())
                    .unwrap_or("DEFAULT")
                    .to_string(),
                max_connections: json_u32(t, "max_connections").unwrap_or(1024),
                max_pending_requests: json_u32(t, "max_pending_requests").unwrap_or(1024),
                max_requests: json_u32(t, "max_requests").unwrap_or(1024),
                max_retries: json_u32(t, "max_retries").unwrap_or(3),
                track_remaining: t
                    .get("track_remaining")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
            }
        })
        .collect();
    Some(CircuitBreakers { thresholds })
}

/// Parse Envoy outlier-detection config.
pub fn parse_outlier_detection(val: &serde_json::Value) -> Option<EnvoyOutlierDetection> {
    // If the value is null or not an object return None.
    if val.is_null() || !val.is_object() {
        return None;
    }
    Some(EnvoyOutlierDetection {
        consecutive_5xx: json_u32(val, "consecutive_5xx").unwrap_or(5),
        consecutive_gateway_failure: json_u32(val, "consecutive_gateway_failure").unwrap_or(5),
        interval: json_str(val, "interval").unwrap_or_else(|_| "10s".to_string()),
        base_ejection_time: json_str(val, "base_ejection_time").unwrap_or_else(|_| "30s".to_string()),
        max_ejection_percent: json_u32(val, "max_ejection_percent").unwrap_or(10),
        enforcing_consecutive_5xx: json_u32(val, "enforcing_consecutive_5xx").unwrap_or(100),
        enforcing_success_rate: json_u32(val, "enforcing_success_rate").unwrap_or(0),
        success_rate_minimum_hosts: json_u32(val, "success_rate_minimum_hosts").unwrap_or(5),
        success_rate_request_volume: json_u32(val, "success_rate_request_volume").unwrap_or(100),
        success_rate_stdev_factor: json_u32(val, "success_rate_stdev_factor").unwrap_or(1900),
    })
}

/// Parse the `filter_chains` array inside a listener config.
pub fn parse_filter_chains(config: &serde_json::Value) -> Result<Vec<FilterChain>> {
    let arr = match config.get("filter_chains").and_then(|v| v.as_array()) {
        Some(a) => a,
        None => return Ok(Vec::new()),
    };

    let mut chains = Vec::with_capacity(arr.len());
    for fc in arr {
        let filter_chain_match = fc.get("filter_chain_match").map(|m| {
            FilterChainMatch {
                server_names: m
                    .get("server_names")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect())
                    .unwrap_or_default(),
                transport_protocol: m.get("transport_protocol").and_then(|v| v.as_str()).map(String::from),
                application_protocols: m
                    .get("application_protocols")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect())
                    .unwrap_or_default(),
            }
        });

        let filters = fc
            .get("filters")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .map(|f| Filter {
                        name: f.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                        typed_config: f.get("typed_config").cloned(),
                    })
                    .collect()
            })
            .unwrap_or_default();

        let transport_socket = fc.get("transport_socket").and_then(|v| parse_transport_socket(v));

        chains.push(FilterChain {
            filter_chain_match,
            filters,
            transport_socket,
        });
    }
    Ok(chains)
}

/// Parse virtual hosts from a route configuration JSON value.
pub fn parse_virtual_hosts(config: &serde_json::Value) -> Result<Vec<VirtualHost>> {
    let arr = match config.get("virtual_hosts").and_then(|v| v.as_array()) {
        Some(a) => a,
        None => return Ok(Vec::new()),
    };

    let mut hosts = Vec::with_capacity(arr.len());
    for vh in arr {
        let name = json_str(vh, "name").unwrap_or_default();
        let domains = vh
            .get("domains")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect())
            .unwrap_or_default();

        let routes = parse_routes(vh)?;

        let retry_policy = vh
            .get("retry_policy")
            .and_then(|rp| parse_envoy_retry_policy(rp));

        let request_headers_to_add = vh
            .get("request_headers_to_add")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|h| {
                        // Two common forms: { "header": { "key": .., "value": .. }, "append": .. }
                        // or flat { "header_name": .., "header_value": .., "append": .. }
                        let (hname, hval) = if let Some(inner) = h.get("header") {
                            (
                                inner.get("key").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                                inner.get("value").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                            )
                        } else {
                            (
                                h.get("header_name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                                h.get("header_value").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                            )
                        };
                        if hname.is_empty() {
                            return None;
                        }
                        Some(HeaderValueOption {
                            header_name: hname,
                            header_value: hval,
                            append: h.get("append").and_then(|v| v.as_bool()).unwrap_or(true),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        hosts.push(VirtualHost {
            name,
            domains,
            routes,
            retry_policy,
            request_headers_to_add,
        });
    }
    Ok(hosts)
}

/// Parse the `routes` array inside a virtual host JSON value.
pub fn parse_routes(vh: &serde_json::Value) -> Result<Vec<Route>> {
    let arr = match vh.get("routes").and_then(|v| v.as_array()) {
        Some(a) => a,
        None => return Ok(Vec::new()),
    };

    let mut routes = Vec::with_capacity(arr.len());
    for r in arr {
        let name = r.get("name").and_then(|v| v.as_str()).map(String::from);

        // Match
        let m = r.get("match").unwrap_or(&serde_json::Value::Null);
        let match_pattern = RouteMatch {
            prefix: m.get("prefix").and_then(|v| v.as_str()).map(String::from),
            path: m.get("path").and_then(|v| v.as_str()).map(String::from),
            safe_regex: m
                .get("safe_regex")
                .and_then(|v| v.get("regex"))
                .and_then(|v| v.as_str())
                .map(String::from),
            headers: m
                .get("headers")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .map(|hm| HeaderMatcher {
                            name: hm.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                            exact_match: hm.get("exact_match").and_then(|v| v.as_str()).map(String::from),
                            prefix_match: hm.get("prefix_match").and_then(|v| v.as_str()).map(String::from),
                            present_match: hm.get("present_match").and_then(|v| v.as_bool()),
                        })
                        .collect()
                })
                .unwrap_or_default(),
        };

        // Route action
        let route_action = r.get("route").map(|ra| {
            let cluster = json_str(ra, "cluster").unwrap_or_default();
            let timeout = ra.get("timeout").and_then(|v| v.as_str()).map(String::from);
            let retry_policy = ra.get("retry_policy").and_then(|rp| parse_envoy_retry_policy(rp));
            let host_rewrite_literal = ra
                .get("host_rewrite_literal")
                .and_then(|v| v.as_str())
                .map(String::from);
            let prefix_rewrite = ra.get("prefix_rewrite").and_then(|v| v.as_str()).map(String::from);
            let weighted_clusters = ra.get("weighted_clusters").map(|wc| {
                let clusters = wc
                    .get("clusters")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .map(|c| WeightedClusterEntry {
                                name: json_str(c, "name").unwrap_or_default(),
                                weight: json_u32(c, "weight").unwrap_or(0),
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                let total_weight = json_u32(wc, "total_weight").unwrap_or(100);
                WeightedClusters {
                    clusters,
                    total_weight,
                }
            });

            RouteAction {
                cluster,
                timeout,
                retry_policy,
                host_rewrite_literal,
                prefix_rewrite,
                weighted_clusters,
            }
        });

        // Direct response
        let direct_response = r.get("direct_response").map(|dr| {
            DirectResponseAction {
                status: json_u32(dr, "status").unwrap_or(200),
                body: dr
                    .get("body")
                    .and_then(|b| b.get("inline_string").and_then(|v| v.as_str()))
                    .or_else(|| dr.get("body").and_then(|b| b.as_str()))
                    .map(String::from),
            }
        });

        routes.push(Route {
            name,
            match_pattern,
            route_action,
            direct_response,
        });
    }
    Ok(routes)
}

/// Parse an Envoy duration string (`"5s"`, `"100ms"`, `"0.5s"`, `"1.5s"`,
/// `"500us"`) into whole milliseconds.
pub fn parse_envoy_duration_to_ms(duration: &str) -> u64 {
    let s = duration.trim();
    if s.is_empty() {
        return 0;
    }

    // Try milliseconds first: "100ms"
    if let Some(num) = s.strip_suffix("ms") {
        if let Ok(v) = num.trim().parse::<f64>() {
            return v as u64;
        }
    }

    // Microseconds: "500us"
    if let Some(num) = s.strip_suffix("us") {
        if let Ok(v) = num.trim().parse::<f64>() {
            return (v / 1000.0) as u64;
        }
    }

    // Minutes: "1m"
    if let Some(num) = s.strip_suffix('m') {
        // Guard against "ms" already handled above (won't reach here).
        if let Ok(v) = num.trim().parse::<f64>() {
            return (v * 60_000.0) as u64;
        }
    }

    // Seconds: "5s" or "0.5s"
    if let Some(num) = s.strip_suffix('s') {
        if let Ok(v) = num.trim().parse::<f64>() {
            return (v * 1000.0) as u64;
        }
    }

    // Bare number – assume seconds.
    if let Ok(v) = s.parse::<f64>() {
        return (v * 1000.0) as u64;
    }

    0
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn json_str(val: &serde_json::Value, key: &str) -> Result<String> {
    val.get(key)
        .and_then(|v| v.as_str())
        .map(String::from)
        .ok_or_else(|| anyhow::anyhow!("missing or non-string field '{}'", key))
}

fn json_u32(val: &serde_json::Value, key: &str) -> Option<u32> {
    val.get(key).and_then(|v| {
        v.as_u64()
            .map(|n| n as u32)
            .or_else(|| v.as_str().and_then(|s| s.parse::<u32>().ok()))
    })
}

fn parse_locality_lb_endpoints(val: &serde_json::Value) -> Option<LocalityLbEndpoints> {
    let locality = val.get("locality").map(|l| Locality {
        region: l.get("region").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        zone: l.get("zone").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        sub_zone: l.get("sub_zone").and_then(|v| v.as_str()).unwrap_or("").to_string(),
    });

    let lb_endpoints = val
        .get("lb_endpoints")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|ep| {
                    let endpoint = ep.get("endpoint")?;
                    let sa = endpoint.get("address")?.get("socket_address")?;
                    let address = sa.get("address").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let port = sa.get("port_value").and_then(|v| v.as_u64()).unwrap_or(0) as u16;

                    let health_status = ep
                        .get("health_status")
                        .and_then(|v| v.as_str())
                        .unwrap_or("HEALTHY")
                        .to_string();

                    let metadata = ep
                        .get("metadata")
                        .and_then(|m| m.get("filter_metadata"))
                        .and_then(|fm| fm.as_object())
                        .map(|obj| {
                            let mut map = IndexMap::new();
                            for (k, v) in obj {
                                if let Some(s) = v.as_str() {
                                    map.insert(k.clone(), s.to_string());
                                } else {
                                    map.insert(k.clone(), v.to_string());
                                }
                            }
                            map
                        })
                        .unwrap_or_default();

                    Some(LbEndpoint {
                        address,
                        port,
                        health_status,
                        metadata,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let priority = val.get("priority").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

    Some(LocalityLbEndpoints {
        locality,
        lb_endpoints,
        priority,
    })
}

fn parse_health_check(val: &serde_json::Value) -> Option<HealthCheck> {
    let timeout = json_str(val, "timeout").unwrap_or_else(|_| "5s".to_string());
    let interval = json_str(val, "interval").unwrap_or_else(|_| "10s".to_string());
    let unhealthy_threshold = json_u32(val, "unhealthy_threshold").unwrap_or(3);
    let healthy_threshold = json_u32(val, "healthy_threshold").unwrap_or(3);

    let health_checker = if let Some(http) = val.get("http_health_check") {
        HealthChecker::HttpHealthCheck {
            path: json_str(http, "path").unwrap_or_else(|_| "/healthz".to_string()),
            host: json_str(http, "host").unwrap_or_default(),
        }
    } else if let Some(tcp) = val.get("tcp_health_check") {
        HealthChecker::TcpHealthCheck {
            send: tcp
                .get("send")
                .and_then(|s| s.get("text"))
                .and_then(|v| v.as_str())
                .map(String::from),
        }
    } else if let Some(grpc) = val.get("grpc_health_check") {
        HealthChecker::GrpcHealthCheck {
            service_name: json_str(grpc, "service_name").unwrap_or_default(),
        }
    } else {
        // Default to TCP if none recognised.
        HealthChecker::TcpHealthCheck { send: None }
    };

    Some(HealthCheck {
        timeout,
        interval,
        unhealthy_threshold,
        healthy_threshold,
        health_checker,
    })
}

fn parse_transport_socket(val: &serde_json::Value) -> Option<TransportSocket> {
    let name = val.get("name").and_then(|v| v.as_str()).unwrap_or("envoy.transport_sockets.tls").to_string();
    let tls_context = val
        .get("typed_config")
        .and_then(|tc| {
            let common = tc
                .get("common_tls_context")
                .map(|ctc| {
                    let alpn = ctc
                        .get("alpn_protocols")
                        .and_then(|v| v.as_array())
                        .map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect())
                        .unwrap_or_default();
                    let certs = ctc
                        .get("tls_certificates")
                        .and_then(|v| v.as_array())
                        .map(|a| {
                            a.iter()
                                .map(|c| TlsCertificate {
                                    certificate_chain: c
                                        .get("certificate_chain")
                                        .and_then(|v| v.get("filename").and_then(|f| f.as_str()))
                                        .or_else(|| c.get("certificate_chain").and_then(|v| v.as_str()))
                                        .unwrap_or("")
                                        .to_string(),
                                    private_key: c
                                        .get("private_key")
                                        .and_then(|v| v.get("filename").and_then(|f| f.as_str()))
                                        .or_else(|| c.get("private_key").and_then(|v| v.as_str()))
                                        .unwrap_or("")
                                        .to_string(),
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    CommonTlsContext {
                        alpn_protocols: alpn,
                        tls_certificates: certs,
                    }
                })
                .unwrap_or_else(|| CommonTlsContext {
                    alpn_protocols: Vec::new(),
                    tls_certificates: Vec::new(),
                });

            let sni = tc.get("sni").and_then(|v| v.as_str()).map(String::from);

            Some(TlsContext {
                common_tls_context: common,
                sni,
            })
        });

    Some(TransportSocket {
        name,
        tls_context,
    })
}

fn parse_envoy_retry_policy(val: &serde_json::Value) -> Option<EnvoyRetryPolicy> {
    if val.is_null() || !val.is_object() {
        return None;
    }
    let retry_on = json_str(val, "retry_on").unwrap_or_default();
    let num_retries = json_u32(val, "num_retries").unwrap_or(1);
    let per_try_timeout = val.get("per_try_timeout").and_then(|v| v.as_str()).map(String::from);
    let retry_back_off = val.get("retry_back_off").map(|bo| {
        RetryBackOff {
            base_interval: json_str(bo, "base_interval").unwrap_or_else(|_| "25ms".to_string()),
            max_interval: bo.get("max_interval").and_then(|v| v.as_str()).map(String::from),
        }
    });
    let retry_host_predicate = val
        .get("retry_host_predicate")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|p| {
                    p.get("name").and_then(|v| v.as_str()).map(|n| RetryHostPredicate {
                        name: n.to_string(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let host_selection_retry_max_attempts =
        json_u32(val, "host_selection_retry_max_attempts").unwrap_or(0);

    Some(EnvoyRetryPolicy {
        retry_on,
        num_retries,
        per_try_timeout,
        retry_back_off,
        retry_host_predicate,
        host_selection_retry_max_attempts,
    })
}

fn parse_listener_filters(config: &serde_json::Value) -> Vec<ListenerFilter> {
    config
        .get("listener_filters")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .map(|f| ListenerFilter {
                    name: f.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    typed_config: f.get("typed_config").cloned(),
                })
                .collect()
        })
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parser() -> EnvoyParser {
        EnvoyParser::new()
    }

    #[test]
    fn test_parse_cluster_basic() {
        let cfg = json!({
            "name": "my-cluster",
            "type": "STRICT_DNS",
            "connect_timeout": "1s",
            "lb_policy": "ROUND_ROBIN"
        });
        let cluster = parser().parse_cluster(&cfg).unwrap();
        assert_eq!(cluster.name, "my-cluster");
        assert_eq!(cluster.cluster_type, "STRICT_DNS");
        assert_eq!(cluster.connect_timeout, "1s");
        assert_eq!(cluster.lb_policy, "ROUND_ROBIN");
        assert!(cluster.load_assignment.is_none());
        assert!(cluster.circuit_breakers.is_none());
        assert!(cluster.outlier_detection.is_none());
        assert!(cluster.health_checks.is_empty());
    }

    #[test]
    fn test_parse_cluster_with_circuit_breakers() {
        let cfg = json!({
            "name": "cb-cluster",
            "type": "EDS",
            "connect_timeout": "5s",
            "lb_policy": "LEAST_REQUEST",
            "circuit_breakers": {
                "thresholds": [
                    {
                        "priority": "DEFAULT",
                        "max_connections": 512,
                        "max_pending_requests": 256,
                        "max_requests": 1024,
                        "max_retries": 5,
                        "track_remaining": true
                    },
                    {
                        "priority": "HIGH",
                        "max_connections": 2048,
                        "max_pending_requests": 1024,
                        "max_requests": 4096,
                        "max_retries": 10,
                        "track_remaining": false
                    }
                ]
            }
        });
        let cluster = parser().parse_cluster(&cfg).unwrap();
        let cb = cluster.circuit_breakers.unwrap();
        assert_eq!(cb.thresholds.len(), 2);
        assert_eq!(cb.thresholds[0].priority, "DEFAULT");
        assert_eq!(cb.thresholds[0].max_connections, 512);
        assert_eq!(cb.thresholds[0].max_pending_requests, 256);
        assert!(cb.thresholds[0].track_remaining);
        assert_eq!(cb.thresholds[1].priority, "HIGH");
        assert_eq!(cb.thresholds[1].max_retries, 10);
    }

    #[test]
    fn test_parse_cluster_with_outlier_detection() {
        let cfg = json!({
            "name": "od-cluster",
            "type": "EDS",
            "connect_timeout": "5s",
            "lb_policy": "ROUND_ROBIN",
            "outlier_detection": {
                "consecutive_5xx": 3,
                "consecutive_gateway_failure": 2,
                "interval": "5s",
                "base_ejection_time": "15s",
                "max_ejection_percent": 50,
                "enforcing_consecutive_5xx": 80,
                "enforcing_success_rate": 100,
                "success_rate_minimum_hosts": 3,
                "success_rate_request_volume": 50,
                "success_rate_stdev_factor": 1500
            }
        });
        let cluster = parser().parse_cluster(&cfg).unwrap();
        let od = cluster.outlier_detection.unwrap();
        assert_eq!(od.consecutive_5xx, 3);
        assert_eq!(od.consecutive_gateway_failure, 2);
        assert_eq!(od.interval, "5s");
        assert_eq!(od.base_ejection_time, "15s");
        assert_eq!(od.max_ejection_percent, 50);
        assert_eq!(od.enforcing_consecutive_5xx, 80);
        assert_eq!(od.enforcing_success_rate, 100);
        assert_eq!(od.success_rate_stdev_factor, 1500);
    }

    #[test]
    fn test_parse_listener_basic() {
        let cfg = json!({
            "name": "http-listener",
            "address": {
                "socket_address": {
                    "address": "0.0.0.0",
                    "port_value": 8080
                }
            },
            "filter_chains": [
                {
                    "filters": [
                        {
                            "name": "envoy.filters.network.http_connection_manager",
                            "typed_config": { "@type": "type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager" }
                        }
                    ]
                }
            ],
            "listener_filters": [
                { "name": "envoy.filters.listener.tls_inspector" }
            ]
        });
        let listener = parser().parse_listener(&cfg).unwrap();
        assert_eq!(listener.name, "http-listener");
        assert_eq!(listener.address.socket_address.address, "0.0.0.0");
        assert_eq!(listener.address.socket_address.port_value, 8080);
        assert_eq!(listener.filter_chains.len(), 1);
        assert_eq!(listener.filter_chains[0].filters.len(), 1);
        assert_eq!(
            listener.filter_chains[0].filters[0].name,
            "envoy.filters.network.http_connection_manager"
        );
        assert_eq!(listener.listener_filters.len(), 1);
        assert_eq!(listener.listener_filters[0].name, "envoy.filters.listener.tls_inspector");
    }

    #[test]
    fn test_parse_route_config() {
        let cfg = json!({
            "name": "local_route",
            "virtual_hosts": [
                {
                    "name": "backend",
                    "domains": ["*"],
                    "routes": [
                        {
                            "match": { "prefix": "/" },
                            "route": { "cluster": "backend-cluster", "timeout": "30s" }
                        }
                    ]
                }
            ]
        });
        let rc = parser().parse_route_config(&cfg).unwrap();
        assert_eq!(rc.name, "local_route");
        assert_eq!(rc.virtual_hosts.len(), 1);
        assert_eq!(rc.virtual_hosts[0].name, "backend");
        assert_eq!(rc.virtual_hosts[0].domains, vec!["*"]);
        assert_eq!(rc.virtual_hosts[0].routes.len(), 1);
        let route = &rc.virtual_hosts[0].routes[0];
        assert_eq!(route.match_pattern.prefix.as_deref(), Some("/"));
        let action = route.route_action.as_ref().unwrap();
        assert_eq!(action.cluster, "backend-cluster");
        assert_eq!(action.timeout.as_deref(), Some("30s"));
    }

    #[test]
    fn test_parse_route_with_retry() {
        let cfg = json!({
            "name": "retry_route",
            "virtual_hosts": [{
                "name": "svc",
                "domains": ["svc.example.com"],
                "routes": [{
                    "match": { "prefix": "/api" },
                    "route": {
                        "cluster": "svc-cluster",
                        "timeout": "10s",
                        "retry_policy": {
                            "retry_on": "5xx,connect-failure",
                            "num_retries": 3,
                            "per_try_timeout": "2s",
                            "retry_back_off": {
                                "base_interval": "100ms",
                                "max_interval": "1s"
                            },
                            "retry_host_predicate": [
                                { "name": "envoy.retry_host_predicates.previous_hosts" }
                            ],
                            "host_selection_retry_max_attempts": 5
                        }
                    }
                }]
            }]
        });
        let rc = parser().parse_route_config(&cfg).unwrap();
        let action = rc.virtual_hosts[0].routes[0].route_action.as_ref().unwrap();
        let rp = action.retry_policy.as_ref().unwrap();
        assert_eq!(rp.retry_on, "5xx,connect-failure");
        assert_eq!(rp.num_retries, 3);
        assert_eq!(rp.per_try_timeout.as_deref(), Some("2s"));
        let bo = rp.retry_back_off.as_ref().unwrap();
        assert_eq!(bo.base_interval, "100ms");
        assert_eq!(bo.max_interval.as_deref(), Some("1s"));
        assert_eq!(rp.retry_host_predicate.len(), 1);
        assert_eq!(rp.retry_host_predicate[0].name, "envoy.retry_host_predicates.previous_hosts");
        assert_eq!(rp.host_selection_retry_max_attempts, 5);
    }

    #[test]
    fn test_parse_weighted_clusters() {
        let cfg = json!({
            "name": "canary_route",
            "virtual_hosts": [{
                "name": "canary",
                "domains": ["*"],
                "routes": [{
                    "match": { "prefix": "/" },
                    "route": {
                        "weighted_clusters": {
                            "clusters": [
                                { "name": "v1", "weight": 90 },
                                { "name": "v2", "weight": 10 }
                            ],
                            "total_weight": 100
                        }
                    }
                }]
            }]
        });
        let rc = parser().parse_route_config(&cfg).unwrap();
        let action = rc.virtual_hosts[0].routes[0].route_action.as_ref().unwrap();
        let wc = action.weighted_clusters.as_ref().unwrap();
        assert_eq!(wc.total_weight, 100);
        assert_eq!(wc.clusters.len(), 2);
        assert_eq!(wc.clusters[0].name, "v1");
        assert_eq!(wc.clusters[0].weight, 90);
        assert_eq!(wc.clusters[1].name, "v2");
        assert_eq!(wc.clusters[1].weight, 10);
    }

    #[test]
    fn test_parse_xds_config() {
        let cfg = json!({
            "static_clusters": [
                { "cluster": { "name": "c1", "type": "STATIC", "connect_timeout": "1s", "lb_policy": "ROUND_ROBIN" } }
            ],
            "static_listeners": [
                { "listener": {
                    "name": "l1",
                    "address": { "socket_address": { "address": "0.0.0.0", "port_value": 80 } }
                }}
            ],
            "static_route_configs": [
                { "route_config": { "name": "r1", "virtual_hosts": [] } }
            ]
        });
        let resources = parser().parse_xds_config(&cfg).unwrap();
        assert_eq!(resources.len(), 3);
        assert!(matches!(&resources[0], EnvoyResource::Cluster(c) if c.name == "c1"));
        assert!(matches!(&resources[1], EnvoyResource::Listener(l) if l.name == "l1"));
        assert!(matches!(&resources[2], EnvoyResource::RouteConfig(r) if r.name == "r1"));
    }

    #[test]
    fn test_extract_retry_policy() {
        let envoy_rp = EnvoyRetryPolicy {
            retry_on: "5xx,connect-failure,reset".to_string(),
            num_retries: 4,
            per_try_timeout: Some("3s".to_string()),
            retry_back_off: Some(RetryBackOff {
                base_interval: "200ms".to_string(),
                max_interval: Some("2s".to_string()),
            }),
            retry_host_predicate: vec![],
            host_selection_retry_max_attempts: 0,
        };
        let unified = EnvoyParser::extract_retry_policy_from_envoy(&envoy_rp);
        assert_eq!(unified.max_retries, 4);
        assert_eq!(unified.per_try_timeout_ms, 3000);
        assert_eq!(unified.retry_on, vec!["5xx", "connect-failure", "reset"]);
        assert_eq!(unified.backoff_base_ms, 200);
        assert_eq!(unified.backoff_max_ms, 2000);
    }

    #[test]
    fn test_extract_timeout_policy() {
        let action = RouteAction {
            cluster: "svc".to_string(),
            timeout: Some("15s".to_string()),
            retry_policy: None,
            host_rewrite_literal: None,
            prefix_rewrite: None,
            weighted_clusters: None,
        };
        let tp = EnvoyParser::extract_timeout_policy_from_envoy(&action);
        assert_eq!(tp.request_timeout_ms, 15000);
        assert_eq!(tp.idle_timeout_ms, 300_000);
        assert_eq!(tp.connect_timeout_ms, 5000);

        // Default when no timeout specified
        let action2 = RouteAction {
            cluster: "svc".to_string(),
            timeout: None,
            retry_policy: None,
            host_rewrite_literal: None,
            prefix_rewrite: None,
            weighted_clusters: None,
        };
        let tp2 = EnvoyParser::extract_timeout_policy_from_envoy(&action2);
        assert_eq!(tp2.request_timeout_ms, 15000);
    }

    #[test]
    fn test_parse_health_check() {
        let hc_val = json!({
            "timeout": "3s",
            "interval": "5s",
            "unhealthy_threshold": 2,
            "healthy_threshold": 1,
            "http_health_check": {
                "path": "/ready",
                "host": "backend.local"
            }
        });
        let hc = parse_health_check(&hc_val).unwrap();
        assert_eq!(hc.timeout, "3s");
        assert_eq!(hc.interval, "5s");
        assert_eq!(hc.unhealthy_threshold, 2);
        assert_eq!(hc.healthy_threshold, 1);
        assert!(matches!(
            &hc.health_checker,
            HealthChecker::HttpHealthCheck { path, host }
            if path == "/ready" && host == "backend.local"
        ));

        // gRPC variant
        let grpc_val = json!({
            "timeout": "2s",
            "interval": "10s",
            "grpc_health_check": { "service_name": "my.service" }
        });
        let grpc_hc = parse_health_check(&grpc_val).unwrap();
        assert!(matches!(
            &grpc_hc.health_checker,
            HealthChecker::GrpcHealthCheck { service_name } if service_name == "my.service"
        ));

        // TCP variant
        let tcp_val = json!({
            "timeout": "1s",
            "interval": "5s",
            "tcp_health_check": { "send": { "text": "000000FF" } }
        });
        let tcp_hc = parse_health_check(&tcp_val).unwrap();
        assert!(matches!(
            &tcp_hc.health_checker,
            HealthChecker::TcpHealthCheck { send } if send.as_deref() == Some("000000FF")
        ));
    }

    #[test]
    fn test_parse_cluster_with_endpoints() {
        let cfg = json!({
            "name": "eds-cluster",
            "type": "EDS",
            "connect_timeout": "5s",
            "lb_policy": "ROUND_ROBIN",
            "load_assignment": {
                "cluster_name": "eds-cluster",
                "endpoints": [
                    {
                        "locality": { "region": "us-east-1", "zone": "us-east-1a", "sub_zone": "" },
                        "lb_endpoints": [
                            {
                                "endpoint": {
                                    "address": { "socket_address": { "address": "10.0.0.1", "port_value": 8080 } }
                                },
                                "health_status": "HEALTHY"
                            },
                            {
                                "endpoint": {
                                    "address": { "socket_address": { "address": "10.0.0.2", "port_value": 8080 } }
                                },
                                "health_status": "UNHEALTHY"
                            }
                        ],
                        "priority": 0
                    },
                    {
                        "locality": { "region": "us-west-2", "zone": "us-west-2a" },
                        "lb_endpoints": [
                            {
                                "endpoint": {
                                    "address": { "socket_address": { "address": "10.1.0.1", "port_value": 8080 } }
                                },
                                "health_status": "HEALTHY"
                            }
                        ],
                        "priority": 1
                    }
                ]
            }
        });
        let cluster = parser().parse_cluster(&cfg).unwrap();
        let la = cluster.load_assignment.unwrap();
        assert_eq!(la.cluster_name, "eds-cluster");
        assert_eq!(la.endpoints.len(), 2);

        let ep0 = &la.endpoints[0];
        assert_eq!(ep0.locality.as_ref().unwrap().region, "us-east-1");
        assert_eq!(ep0.locality.as_ref().unwrap().zone, "us-east-1a");
        assert_eq!(ep0.lb_endpoints.len(), 2);
        assert_eq!(ep0.lb_endpoints[0].address, "10.0.0.1");
        assert_eq!(ep0.lb_endpoints[0].port, 8080);
        assert_eq!(ep0.lb_endpoints[0].health_status, "HEALTHY");
        assert_eq!(ep0.lb_endpoints[1].health_status, "UNHEALTHY");
        assert_eq!(ep0.priority, 0);

        let ep1 = &la.endpoints[1];
        assert_eq!(ep1.locality.as_ref().unwrap().region, "us-west-2");
        assert_eq!(ep1.lb_endpoints.len(), 1);
        assert_eq!(ep1.lb_endpoints[0].address, "10.1.0.1");
        assert_eq!(ep1.priority, 1);
    }

    #[test]
    fn test_parse_envoy_duration_to_ms() {
        assert_eq!(parse_envoy_duration_to_ms("5s"), 5000);
        assert_eq!(parse_envoy_duration_to_ms("100ms"), 100);
        assert_eq!(parse_envoy_duration_to_ms("0.5s"), 500);
        assert_eq!(parse_envoy_duration_to_ms("1.5s"), 1500);
        assert_eq!(parse_envoy_duration_to_ms("500us"), 0); // rounds down
        assert_eq!(parse_envoy_duration_to_ms("5000us"), 5);
        assert_eq!(parse_envoy_duration_to_ms("1m"), 60000);
        assert_eq!(parse_envoy_duration_to_ms(""), 0);
        assert_eq!(parse_envoy_duration_to_ms("10"), 10000); // bare number => seconds
    }

    #[test]
    fn test_parse_direct_response_route() {
        let cfg = json!({
            "name": "direct_route",
            "virtual_hosts": [{
                "name": "dr",
                "domains": ["*"],
                "routes": [{
                    "match": { "prefix": "/healthz" },
                    "direct_response": { "status": 200, "body": { "inline_string": "OK" } }
                }]
            }]
        });
        let rc = parser().parse_route_config(&cfg).unwrap();
        let route = &rc.virtual_hosts[0].routes[0];
        assert!(route.route_action.is_none());
        let dr = route.direct_response.as_ref().unwrap();
        assert_eq!(dr.status, 200);
        assert_eq!(dr.body.as_deref(), Some("OK"));
    }

    #[test]
    fn test_parse_filter_chain_match() {
        let cfg = json!({
            "name": "tls-listener",
            "address": {
                "socket_address": { "address": "0.0.0.0", "port_value": 443 }
            },
            "filter_chains": [{
                "filter_chain_match": {
                    "server_names": ["example.com", "www.example.com"],
                    "transport_protocol": "tls",
                    "application_protocols": ["h2", "http/1.1"]
                },
                "filters": [
                    { "name": "envoy.filters.network.http_connection_manager" }
                ]
            }],
            "listener_filters": [
                { "name": "envoy.filters.listener.tls_inspector" }
            ]
        });
        let listener = parser().parse_listener(&cfg).unwrap();
        assert_eq!(listener.address.socket_address.port_value, 443);
        let fcm = listener.filter_chains[0].filter_chain_match.as_ref().unwrap();
        assert_eq!(fcm.server_names, vec!["example.com", "www.example.com"]);
        assert_eq!(fcm.transport_protocol.as_deref(), Some("tls"));
        assert_eq!(fcm.application_protocols, vec!["h2", "http/1.1"]);
    }

    #[test]
    fn test_parse_header_matchers() {
        let cfg = json!({
            "name": "header_route",
            "virtual_hosts": [{
                "name": "hdr",
                "domains": ["*"],
                "routes": [{
                    "match": {
                        "prefix": "/",
                        "headers": [
                            { "name": ":method", "exact_match": "GET" },
                            { "name": "x-debug", "present_match": true },
                            { "name": "x-team", "prefix_match": "platform-" }
                        ]
                    },
                    "route": { "cluster": "hdr-cluster" }
                }]
            }]
        });
        let rc = parser().parse_route_config(&cfg).unwrap();
        let headers = &rc.virtual_hosts[0].routes[0].match_pattern.headers;
        assert_eq!(headers.len(), 3);
        assert_eq!(headers[0].name, ":method");
        assert_eq!(headers[0].exact_match.as_deref(), Some("GET"));
        assert_eq!(headers[1].name, "x-debug");
        assert_eq!(headers[1].present_match, Some(true));
        assert_eq!(headers[2].name, "x-team");
        assert_eq!(headers[2].prefix_match.as_deref(), Some("platform-"));
    }
}
