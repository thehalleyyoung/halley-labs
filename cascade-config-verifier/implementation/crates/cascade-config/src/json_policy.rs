//! JSON service mesh policy parsing.
//!
//! This module handles parsing JSON-format service mesh policy documents
//! that describe services, their dependencies, and resilience configuration
//! (retry, timeout, circuit-breaker).  The parsed representation can be
//! converted to the unified [`RetryPolicy`] / [`TimeoutPolicy`] types used
//! throughout the cascade-verify pipeline.

use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use crate::{ObjectMeta, RetryPolicy, ServiceId, TimeoutPolicy};

// ---------------------------------------------------------------------------
// JSON policy document types
// ---------------------------------------------------------------------------

/// Top-level service mesh policy document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ServiceMeshPolicy {
    /// API version string, expected to be `"cascade-verify/v1"`.
    pub api_version: String,
    /// Resource kind, expected to be `"ServiceMeshPolicy"`.
    pub kind: String,
    /// Standard Kubernetes-style metadata.
    pub metadata: PolicyMetadata,
    /// The policy specification.
    pub spec: PolicySpec,
}

/// Metadata block for a service mesh policy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PolicyMetadata {
    pub name: String,
    #[serde(default)]
    pub namespace: String,
    #[serde(default)]
    pub labels: indexmap::IndexMap<String, String>,
    #[serde(default)]
    pub annotations: indexmap::IndexMap<String, String>,
}

/// The specification section of a service mesh policy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PolicySpec {
    #[serde(default)]
    pub services: Vec<ServiceEntry>,
    #[serde(default)]
    pub dependencies: Vec<DependencyEntry>,
    #[serde(default)]
    pub global_defaults: Option<GlobalDefaults>,
}

/// A single service described in the policy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ServiceEntry {
    pub name: String,
    #[serde(default)]
    pub namespace: String,
    #[serde(default)]
    pub capacity: Option<u64>,
    #[serde(default)]
    pub baseline_load: Option<u64>,
    #[serde(default)]
    pub replicas: Option<u32>,
}

/// A dependency edge between two services, with optional resilience configs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DependencyEntry {
    pub from: String,
    pub to: String,
    #[serde(default)]
    pub retry: Option<JsonRetryConfig>,
    #[serde(default)]
    pub timeout: Option<JsonTimeoutConfig>,
    #[serde(default)]
    pub circuit_breaker: Option<JsonCircuitBreakerConfig>,
}

/// Retry configuration in JSON policy format.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JsonRetryConfig {
    #[serde(default)]
    pub max_retries: Option<u32>,
    #[serde(default)]
    pub per_try_timeout_ms: Option<u64>,
    #[serde(default)]
    pub retry_on: Option<Vec<String>>,
    #[serde(default)]
    pub backoff_base_ms: Option<u64>,
    #[serde(default)]
    pub backoff_max_ms: Option<u64>,
}

/// Timeout configuration in JSON policy format.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JsonTimeoutConfig {
    #[serde(default)]
    pub request_timeout_ms: Option<u64>,
    #[serde(default)]
    pub idle_timeout_ms: Option<u64>,
    #[serde(default)]
    pub connect_timeout_ms: Option<u64>,
}

/// Circuit-breaker configuration in JSON policy format.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JsonCircuitBreakerConfig {
    #[serde(default)]
    pub max_connections: Option<u32>,
    #[serde(default)]
    pub max_pending_requests: Option<u32>,
    #[serde(default)]
    pub consecutive_errors: Option<u32>,
}

/// Global default configs that apply when a dependency omits its own.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GlobalDefaults {
    #[serde(default)]
    pub retry: Option<JsonRetryConfig>,
    #[serde(default)]
    pub timeout: Option<JsonTimeoutConfig>,
    #[serde(default)]
    pub circuit_breaker: Option<JsonCircuitBreakerConfig>,
}

// ---------------------------------------------------------------------------
// Conversions to unified types
// ---------------------------------------------------------------------------

impl JsonRetryConfig {
    /// Convert to the unified [`RetryPolicy`], filling unset fields from
    /// `defaults` (the global default retry config, if any).
    pub fn to_retry_policy(&self, defaults: Option<&JsonRetryConfig>) -> RetryPolicy {
        let base = RetryPolicy::default();
        let def = |f: fn(&JsonRetryConfig) -> Option<u32>| -> Option<u32> {
            defaults.and_then(f)
        };
        let def64 = |f: fn(&JsonRetryConfig) -> Option<u64>| -> Option<u64> {
            defaults.and_then(f)
        };
        RetryPolicy {
            max_retries: self
                .max_retries
                .or_else(|| def(|d| d.max_retries))
                .unwrap_or(base.max_retries),
            per_try_timeout_ms: self
                .per_try_timeout_ms
                .or_else(|| def64(|d| d.per_try_timeout_ms))
                .unwrap_or(base.per_try_timeout_ms),
            retry_on: self
                .retry_on
                .clone()
                .or_else(|| defaults.and_then(|d| d.retry_on.clone()))
                .unwrap_or(base.retry_on),
            backoff_base_ms: self
                .backoff_base_ms
                .or_else(|| def64(|d| d.backoff_base_ms))
                .unwrap_or(base.backoff_base_ms),
            backoff_max_ms: self
                .backoff_max_ms
                .or_else(|| def64(|d| d.backoff_max_ms))
                .unwrap_or(base.backoff_max_ms),
        }
    }
}

impl JsonTimeoutConfig {
    /// Convert to the unified [`TimeoutPolicy`], filling unset fields from
    /// `defaults`.
    pub fn to_timeout_policy(&self, defaults: Option<&JsonTimeoutConfig>) -> TimeoutPolicy {
        let base = TimeoutPolicy::default();
        let def = |f: fn(&JsonTimeoutConfig) -> Option<u64>| -> Option<u64> {
            defaults.and_then(f)
        };
        TimeoutPolicy {
            request_timeout_ms: self
                .request_timeout_ms
                .or_else(|| def(|d| d.request_timeout_ms))
                .unwrap_or(base.request_timeout_ms),
            idle_timeout_ms: self
                .idle_timeout_ms
                .or_else(|| def(|d| d.idle_timeout_ms))
                .unwrap_or(base.idle_timeout_ms),
            connect_timeout_ms: self
                .connect_timeout_ms
                .or_else(|| def(|d| d.connect_timeout_ms))
                .unwrap_or(base.connect_timeout_ms),
        }
    }
}

impl ServiceEntry {
    /// Convert to a [`ServiceId`].
    pub fn to_service_id(&self) -> ServiceId {
        ServiceId::new(&self.name, &self.namespace)
    }
}

impl PolicyMetadata {
    /// Convert to the shared [`ObjectMeta`].
    pub fn to_object_meta(&self) -> ObjectMeta {
        ObjectMeta {
            name: self.name.clone(),
            namespace: self.namespace.clone(),
            labels: self.labels.clone(),
            annotations: self.annotations.clone(),
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Parser for JSON service mesh policy documents.
#[derive(Debug, Clone, Default)]
pub struct JsonPolicyParser;

impl JsonPolicyParser {
    pub fn new() -> Self {
        Self
    }

    /// Parse a JSON string into a [`ServiceMeshPolicy`].
    pub fn parse(&self, json: &str) -> Result<ServiceMeshPolicy> {
        parse_json_policy(json)
    }

    /// Parse a JSON file into a [`ServiceMeshPolicy`].
    pub fn parse_file(&self, path: &Path) -> Result<ServiceMeshPolicy> {
        parse_json_policy_file(path)
    }
}

/// Parse a JSON string into a [`ServiceMeshPolicy`].
pub fn parse_json_policy(json: &str) -> Result<ServiceMeshPolicy> {
    let policy: ServiceMeshPolicy =
        serde_json::from_str(json).context("failed to parse JSON service mesh policy")?;

    validate_policy(&policy)?;

    Ok(policy)
}

/// Read and parse a JSON file into a [`ServiceMeshPolicy`].
pub fn parse_json_policy_file(path: &Path) -> Result<ServiceMeshPolicy> {
    let contents =
        std::fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    parse_json_policy(&contents)
        .with_context(|| format!("while parsing {}", path.display()))
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

fn validate_policy(policy: &ServiceMeshPolicy) -> Result<()> {
    if policy.api_version.is_empty() {
        bail!("apiVersion must not be empty");
    }
    if policy.kind != "ServiceMeshPolicy" {
        bail!(
            "expected kind \"ServiceMeshPolicy\", got \"{}\"",
            policy.kind
        );
    }
    if policy.metadata.name.is_empty() {
        bail!("metadata.name must not be empty");
    }

    // Validate dependency references point to declared services (if services
    // are listed).
    if !policy.spec.services.is_empty() {
        let names: std::collections::HashSet<&str> = policy
            .spec
            .services
            .iter()
            .map(|s| s.name.as_str())
            .collect();
        for dep in &policy.spec.dependencies {
            if !names.contains(dep.from.as_str()) {
                bail!(
                    "dependency references unknown source service \"{}\"",
                    dep.from
                );
            }
            if !names.contains(dep.to.as_str()) {
                bail!(
                    "dependency references unknown target service \"{}\"",
                    dep.to
                );
            }
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers --

    fn minimal_policy_json() -> String {
        serde_json::json!({
            "apiVersion": "cascade-verify/v1",
            "kind": "ServiceMeshPolicy",
            "metadata": { "name": "test-policy", "namespace": "default" },
            "spec": {
                "services": [],
                "dependencies": []
            }
        })
        .to_string()
    }

    fn full_policy_json() -> String {
        serde_json::json!({
            "apiVersion": "cascade-verify/v1",
            "kind": "ServiceMeshPolicy",
            "metadata": {
                "name": "production-mesh",
                "namespace": "prod",
                "labels": { "env": "production" },
                "annotations": { "team": "platform" }
            },
            "spec": {
                "services": [
                    {
                        "name": "api-gateway",
                        "namespace": "prod",
                        "capacity": 5000,
                        "baseline_load": 1000,
                        "replicas": 3
                    },
                    {
                        "name": "user-service",
                        "namespace": "prod",
                        "capacity": 2000,
                        "baseline_load": 500,
                        "replicas": 2
                    },
                    {
                        "name": "order-service",
                        "namespace": "prod",
                        "capacity": 1000,
                        "baseline_load": 200,
                        "replicas": 3
                    }
                ],
                "dependencies": [
                    {
                        "from": "api-gateway",
                        "to": "user-service",
                        "retry": {
                            "max_retries": 3,
                            "per_try_timeout_ms": 5000,
                            "retry_on": ["5xx", "reset"]
                        },
                        "timeout": {
                            "request_timeout_ms": 15000,
                            "connect_timeout_ms": 1000
                        },
                        "circuit_breaker": {
                            "max_connections": 100,
                            "max_pending_requests": 50,
                            "consecutive_errors": 5
                        }
                    },
                    {
                        "from": "api-gateway",
                        "to": "order-service",
                        "retry": {
                            "max_retries": 2,
                            "per_try_timeout_ms": 3000
                        },
                        "timeout": {
                            "request_timeout_ms": 10000
                        }
                    }
                ],
                "global_defaults": {
                    "retry": {
                        "max_retries": 2,
                        "per_try_timeout_ms": 1000
                    },
                    "timeout": {
                        "request_timeout_ms": 10000
                    }
                }
            }
        })
        .to_string()
    }

    // -- parse_json_policy ---------------------------------------------------

    #[test]
    fn test_parse_minimal_policy() {
        let policy = parse_json_policy(&minimal_policy_json()).unwrap();
        assert_eq!(policy.api_version, "cascade-verify/v1");
        assert_eq!(policy.kind, "ServiceMeshPolicy");
        assert_eq!(policy.metadata.name, "test-policy");
        assert!(policy.spec.services.is_empty());
        assert!(policy.spec.dependencies.is_empty());
    }

    #[test]
    fn test_parse_full_policy() {
        let policy = parse_json_policy(&full_policy_json()).unwrap();
        assert_eq!(policy.metadata.name, "production-mesh");
        assert_eq!(policy.metadata.namespace, "prod");
        assert_eq!(policy.spec.services.len(), 3);
        assert_eq!(policy.spec.dependencies.len(), 2);
        assert!(policy.spec.global_defaults.is_some());
    }

    #[test]
    fn test_parse_services() {
        let policy = parse_json_policy(&full_policy_json()).unwrap();
        let svc = &policy.spec.services[0];
        assert_eq!(svc.name, "api-gateway");
        assert_eq!(svc.namespace, "prod");
        assert_eq!(svc.capacity, Some(5000));
        assert_eq!(svc.baseline_load, Some(1000));
        assert_eq!(svc.replicas, Some(3));
    }

    #[test]
    fn test_parse_dependency_retry() {
        let policy = parse_json_policy(&full_policy_json()).unwrap();
        let dep = &policy.spec.dependencies[0];
        assert_eq!(dep.from, "api-gateway");
        assert_eq!(dep.to, "user-service");

        let retry = dep.retry.as_ref().unwrap();
        assert_eq!(retry.max_retries, Some(3));
        assert_eq!(retry.per_try_timeout_ms, Some(5000));
        assert_eq!(
            retry.retry_on,
            Some(vec!["5xx".to_string(), "reset".to_string()])
        );
    }

    #[test]
    fn test_parse_dependency_timeout() {
        let policy = parse_json_policy(&full_policy_json()).unwrap();
        let dep = &policy.spec.dependencies[0];
        let timeout = dep.timeout.as_ref().unwrap();
        assert_eq!(timeout.request_timeout_ms, Some(15000));
        assert_eq!(timeout.connect_timeout_ms, Some(1000));
        assert_eq!(timeout.idle_timeout_ms, None);
    }

    #[test]
    fn test_parse_dependency_circuit_breaker() {
        let policy = parse_json_policy(&full_policy_json()).unwrap();
        let dep = &policy.spec.dependencies[0];
        let cb = dep.circuit_breaker.as_ref().unwrap();
        assert_eq!(cb.max_connections, Some(100));
        assert_eq!(cb.max_pending_requests, Some(50));
        assert_eq!(cb.consecutive_errors, Some(5));
    }

    #[test]
    fn test_parse_global_defaults() {
        let policy = parse_json_policy(&full_policy_json()).unwrap();
        let defaults = policy.spec.global_defaults.as_ref().unwrap();
        let retry = defaults.retry.as_ref().unwrap();
        assert_eq!(retry.max_retries, Some(2));
        assert_eq!(retry.per_try_timeout_ms, Some(1000));
        let timeout = defaults.timeout.as_ref().unwrap();
        assert_eq!(timeout.request_timeout_ms, Some(10000));
    }

    // -- validation ----------------------------------------------------------

    #[test]
    fn test_reject_empty_api_version() {
        let json = serde_json::json!({
            "apiVersion": "",
            "kind": "ServiceMeshPolicy",
            "metadata": { "name": "x" },
            "spec": { "services": [], "dependencies": [] }
        })
        .to_string();
        let err = parse_json_policy(&json).unwrap_err();
        assert!(err.to_string().contains("apiVersion"));
    }

    #[test]
    fn test_reject_wrong_kind() {
        let json = serde_json::json!({
            "apiVersion": "cascade-verify/v1",
            "kind": "WrongKind",
            "metadata": { "name": "x" },
            "spec": { "services": [], "dependencies": [] }
        })
        .to_string();
        let err = parse_json_policy(&json).unwrap_err();
        assert!(err.to_string().contains("ServiceMeshPolicy"));
    }

    #[test]
    fn test_reject_empty_name() {
        let json = serde_json::json!({
            "apiVersion": "cascade-verify/v1",
            "kind": "ServiceMeshPolicy",
            "metadata": { "name": "" },
            "spec": { "services": [], "dependencies": [] }
        })
        .to_string();
        let err = parse_json_policy(&json).unwrap_err();
        assert!(err.to_string().contains("name"));
    }

    #[test]
    fn test_reject_unknown_dependency_source() {
        let json = serde_json::json!({
            "apiVersion": "cascade-verify/v1",
            "kind": "ServiceMeshPolicy",
            "metadata": { "name": "p" },
            "spec": {
                "services": [{ "name": "a" }],
                "dependencies": [{ "from": "missing", "to": "a" }]
            }
        })
        .to_string();
        let err = parse_json_policy(&json).unwrap_err();
        assert!(err.to_string().contains("missing"));
    }

    #[test]
    fn test_reject_unknown_dependency_target() {
        let json = serde_json::json!({
            "apiVersion": "cascade-verify/v1",
            "kind": "ServiceMeshPolicy",
            "metadata": { "name": "p" },
            "spec": {
                "services": [{ "name": "a" }],
                "dependencies": [{ "from": "a", "to": "missing" }]
            }
        })
        .to_string();
        let err = parse_json_policy(&json).unwrap_err();
        assert!(err.to_string().contains("missing"));
    }

    #[test]
    fn test_reject_invalid_json() {
        let err = parse_json_policy("not json").unwrap_err();
        assert!(err.to_string().contains("failed to parse"));
    }

    // -- conversion to unified types -----------------------------------------

    #[test]
    fn test_retry_config_to_policy_all_set() {
        let cfg = JsonRetryConfig {
            max_retries: Some(5),
            per_try_timeout_ms: Some(2000),
            retry_on: Some(vec!["5xx".into(), "reset".into()]),
            backoff_base_ms: Some(100),
            backoff_max_ms: Some(1000),
        };
        let policy = cfg.to_retry_policy(None);
        assert_eq!(policy.max_retries, 5);
        assert_eq!(policy.per_try_timeout_ms, 2000);
        assert_eq!(policy.retry_on, vec!["5xx".to_string(), "reset".to_string()]);
        assert_eq!(policy.backoff_base_ms, 100);
        assert_eq!(policy.backoff_max_ms, 1000);
    }

    #[test]
    fn test_retry_config_to_policy_with_defaults() {
        let cfg = JsonRetryConfig {
            max_retries: Some(3),
            per_try_timeout_ms: None,
            retry_on: None,
            backoff_base_ms: None,
            backoff_max_ms: None,
        };
        let defaults = JsonRetryConfig {
            max_retries: Some(1),
            per_try_timeout_ms: Some(500),
            retry_on: Some(vec!["gateway-error".into()]),
            backoff_base_ms: Some(50),
            backoff_max_ms: Some(200),
        };
        let policy = cfg.to_retry_policy(Some(&defaults));
        // Explicit value wins
        assert_eq!(policy.max_retries, 3);
        // Defaults fill in
        assert_eq!(policy.per_try_timeout_ms, 500);
        assert_eq!(policy.retry_on, vec!["gateway-error".to_string()]);
        assert_eq!(policy.backoff_base_ms, 50);
        assert_eq!(policy.backoff_max_ms, 200);
    }

    #[test]
    fn test_retry_config_to_policy_no_values_no_defaults() {
        let cfg = JsonRetryConfig {
            max_retries: None,
            per_try_timeout_ms: None,
            retry_on: None,
            backoff_base_ms: None,
            backoff_max_ms: None,
        };
        let policy = cfg.to_retry_policy(None);
        let base = RetryPolicy::default();
        assert_eq!(policy, base);
    }

    #[test]
    fn test_timeout_config_to_policy_all_set() {
        let cfg = JsonTimeoutConfig {
            request_timeout_ms: Some(30000),
            idle_timeout_ms: Some(60000),
            connect_timeout_ms: Some(3000),
        };
        let policy = cfg.to_timeout_policy(None);
        assert_eq!(policy.request_timeout_ms, 30000);
        assert_eq!(policy.idle_timeout_ms, 60000);
        assert_eq!(policy.connect_timeout_ms, 3000);
    }

    #[test]
    fn test_timeout_config_to_policy_with_defaults() {
        let cfg = JsonTimeoutConfig {
            request_timeout_ms: Some(20000),
            idle_timeout_ms: None,
            connect_timeout_ms: None,
        };
        let defaults = JsonTimeoutConfig {
            request_timeout_ms: Some(10000),
            idle_timeout_ms: Some(120000),
            connect_timeout_ms: Some(2000),
        };
        let policy = cfg.to_timeout_policy(Some(&defaults));
        assert_eq!(policy.request_timeout_ms, 20000);
        assert_eq!(policy.idle_timeout_ms, 120000);
        assert_eq!(policy.connect_timeout_ms, 2000);
    }

    #[test]
    fn test_timeout_config_to_policy_no_values_no_defaults() {
        let cfg = JsonTimeoutConfig {
            request_timeout_ms: None,
            idle_timeout_ms: None,
            connect_timeout_ms: None,
        };
        let policy = cfg.to_timeout_policy(None);
        let base = TimeoutPolicy::default();
        assert_eq!(policy, base);
    }

    #[test]
    fn test_service_entry_to_service_id() {
        let entry = ServiceEntry {
            name: "api-gateway".into(),
            namespace: "prod".into(),
            capacity: Some(5000),
            baseline_load: Some(1000),
            replicas: Some(3),
        };
        let id = entry.to_service_id();
        assert_eq!(id.name, "api-gateway");
        assert_eq!(id.namespace, "prod");
    }

    #[test]
    fn test_policy_metadata_to_object_meta() {
        let pm = PolicyMetadata {
            name: "mesh-policy".into(),
            namespace: "prod".into(),
            labels: {
                let mut m = indexmap::IndexMap::new();
                m.insert("env".into(), "production".into());
                m
            },
            annotations: indexmap::IndexMap::new(),
        };
        let meta = pm.to_object_meta();
        assert_eq!(meta.name, "mesh-policy");
        assert_eq!(meta.namespace, "prod");
        assert_eq!(meta.labels.get("env"), Some(&"production".to_string()));
        assert!(meta.uid.is_empty());
    }

    // -- JsonPolicyParser struct ----------------------------------------------

    #[test]
    fn test_parser_struct_parse() {
        let parser = JsonPolicyParser::new();
        let policy = parser.parse(&minimal_policy_json()).unwrap();
        assert_eq!(policy.metadata.name, "test-policy");
    }

    #[test]
    fn test_parser_struct_parse_file_missing() {
        let parser = JsonPolicyParser::new();
        let err = parser
            .parse_file(Path::new("/nonexistent/path.json"))
            .unwrap_err();
        assert!(err.to_string().contains("failed to read"));
    }

    // -- serde round-trip ----------------------------------------------------

    #[test]
    fn test_serde_roundtrip() {
        let original = parse_json_policy(&full_policy_json()).unwrap();
        let serialized = serde_json::to_string(&original).unwrap();
        let deserialized: ServiceMeshPolicy = serde_json::from_str(&serialized).unwrap();
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_circuit_breaker_serde() {
        let cb = JsonCircuitBreakerConfig {
            max_connections: Some(100),
            max_pending_requests: Some(50),
            consecutive_errors: Some(5),
        };
        let json = serde_json::to_string(&cb).unwrap();
        let deser: JsonCircuitBreakerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cb, deser);
    }

    #[test]
    fn test_dependency_without_optional_configs() {
        let json = serde_json::json!({
            "apiVersion": "cascade-verify/v1",
            "kind": "ServiceMeshPolicy",
            "metadata": { "name": "sparse" },
            "spec": {
                "services": [{ "name": "a" }, { "name": "b" }],
                "dependencies": [{ "from": "a", "to": "b" }]
            }
        })
        .to_string();
        let policy = parse_json_policy(&json).unwrap();
        let dep = &policy.spec.dependencies[0];
        assert!(dep.retry.is_none());
        assert!(dep.timeout.is_none());
        assert!(dep.circuit_breaker.is_none());
    }

    #[test]
    fn test_service_entry_defaults() {
        let json = serde_json::json!({
            "apiVersion": "cascade-verify/v1",
            "kind": "ServiceMeshPolicy",
            "metadata": { "name": "defaults" },
            "spec": {
                "services": [{ "name": "svc" }],
                "dependencies": []
            }
        })
        .to_string();
        let policy = parse_json_policy(&json).unwrap();
        let svc = &policy.spec.services[0];
        assert_eq!(svc.namespace, "");
        assert_eq!(svc.capacity, None);
        assert_eq!(svc.baseline_load, None);
        assert_eq!(svc.replicas, None);
    }

    // -- parse_json_policy_file with temp file --------------------------------

    #[test]
    fn test_parse_json_policy_file_roundtrip() {
        let dir = std::env::temp_dir().join("cascade_test_json_policy");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test_policy.json");
        std::fs::write(&path, full_policy_json()).unwrap();

        let policy = parse_json_policy_file(&path).unwrap();
        assert_eq!(policy.metadata.name, "production-mesh");
        assert_eq!(policy.spec.services.len(), 3);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }
}
