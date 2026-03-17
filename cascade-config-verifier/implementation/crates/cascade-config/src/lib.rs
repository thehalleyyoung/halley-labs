//! # cascade-config
//!
//! Configuration parsing and processing for the CascadeVerify project.
//!
//! This crate handles parsing Kubernetes, Istio, and Envoy configuration files,
//! Helm template expansion, Kustomize overlay processing, cross-resource reference
//! resolution, and configuration validation.
//!
//! ## Modules
//!
//! - [`kubernetes`] – Kubernetes resource parsing (Deployments, Services, Ingress, ConfigMaps)
//! - [`istio`] – Istio configuration parsing (VirtualService, DestinationRule, Gateway, ServiceEntry)
//! - [`envoy`] – Envoy xDS configuration parsing (Clusters, Listeners, Routes)
//! - [`helm`] – Helm chart template expansion with Go template support
//! - [`kustomize`] – Kustomize overlay and patch processing
//! - [`reference_resolver`] – Cross-resource reference resolution and dependency inference
//! - [`validator`] – Configuration validation and consistency checking

pub mod envoy;
pub mod helm;
pub mod istio;
pub mod json_policy;
pub mod kubernetes;
pub mod kustomize;
pub mod reference_resolver;
pub mod validator;

// Re-export primary types for ergonomic access.

pub use envoy::{
    CircuitBreakers, EnvoyCluster, EnvoyListener, EnvoyParser, EnvoyRetryPolicy, FilterChain,
    HttpFilter, Route, RouteAction, RouteConfiguration, Threshold, VirtualHost,
};
pub use helm::{
    ChartMetadata, GoTemplateEngine, HelmProcessor, HelmValues, ReleaseInfo, TemplateContext,
};
pub use json_policy::{
    DependencyEntry, GlobalDefaults, JsonCircuitBreakerConfig, JsonPolicyParser, JsonRetryConfig,
    JsonTimeoutConfig, PolicyMetadata, PolicySpec, ServiceEntry, ServiceMeshPolicy,
    parse_json_policy, parse_json_policy_file,
};
pub use istio::{
    ConnectionPool, DestinationRule, Gateway, HttpRetryPolicy, HttpRoute, IstioParser,
    OutlierDetection, Subset, TrafficPolicy, VirtualService,
};
pub use kubernetes::{
    ConfigMap, ContainerSpec, Deployment, DeploymentSpec, Ingress, KubeService,
    KubernetesParser, KubernetesResource, ResourceRequirements,
};
pub use kustomize::{JsonPatchOp, Kustomization, KustomizeLayer, KustomizeProcessor};
pub use reference_resolver::{
    CrossNamespaceResolver, DefaultValueInferrer, ReferenceResolver, ResolvedRefs, SubsetRouting,
};
pub use validator::{ConfigValidator, ValidationIssue, ValidatorConfig};

/// Convenience type alias for Results using `anyhow::Error`.
pub type Result<T> = anyhow::Result<T>;

/// Shared metadata found on all Kubernetes-style resources.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct ObjectMeta {
    pub name: String,
    #[serde(default)]
    pub namespace: String,
    #[serde(default)]
    pub labels: indexmap::IndexMap<String, String>,
    #[serde(default)]
    pub annotations: indexmap::IndexMap<String, String>,
    #[serde(default)]
    pub uid: String,
    #[serde(default)]
    pub resource_version: String,
}

/// Identifies a service uniquely within the mesh.
#[derive(Debug, Clone, Hash, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ServiceId {
    pub name: String,
    pub namespace: String,
}

impl std::fmt::Display for ServiceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.namespace, self.name)
    }
}

impl ServiceId {
    pub fn new(name: impl Into<String>, namespace: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            namespace: namespace.into(),
        }
    }
}

/// Represents a concrete service endpoint address.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct ServiceEndpoint {
    pub address: String,
    pub port: u16,
    pub protocol: String,
}

/// A unified retry policy suitable for cross-layer analysis.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub per_try_timeout_ms: u64,
    pub retry_on: Vec<String>,
    pub backoff_base_ms: u64,
    pub backoff_max_ms: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 2,
            per_try_timeout_ms: 1000,
            retry_on: vec!["5xx".to_string()],
            backoff_base_ms: 25,
            backoff_max_ms: 250,
        }
    }
}

/// A unified timeout policy suitable for cross-layer analysis.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct TimeoutPolicy {
    pub request_timeout_ms: u64,
    pub idle_timeout_ms: u64,
    pub connect_timeout_ms: u64,
}

impl Default for TimeoutPolicy {
    fn default() -> Self {
        Self {
            request_timeout_ms: 15000,
            idle_timeout_ms: 300000,
            connect_timeout_ms: 5000,
        }
    }
}

/// Severity levels for validation findings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub enum Severity {
    Info,
    Warning,
    Error,
    Critical,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Info => write!(f, "INFO"),
            Severity::Warning => write!(f, "WARNING"),
            Severity::Error => write!(f, "ERROR"),
            Severity::Critical => write!(f, "CRITICAL"),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- ObjectMeta ---------------------------------------------------------

    #[test]
    fn test_object_meta_default() {
        let meta = ObjectMeta::default();
        assert!(meta.name.is_empty());
        assert!(meta.namespace.is_empty());
        assert!(meta.labels.is_empty());
        assert!(meta.annotations.is_empty());
        assert!(meta.uid.is_empty());
        assert!(meta.resource_version.is_empty());
    }

    #[test]
    fn test_object_meta_serde_minimal() {
        let json = r#"{"name":"my-svc","namespace":"prod"}"#;
        let meta: ObjectMeta = serde_json::from_str(json).unwrap();
        assert_eq!(meta.name, "my-svc");
        assert_eq!(meta.namespace, "prod");
        assert!(meta.labels.is_empty());
    }

    #[test]
    fn test_object_meta_serde_full() {
        let meta = ObjectMeta {
            name: "api-server".into(),
            namespace: "production".into(),
            labels: {
                let mut m = indexmap::IndexMap::new();
                m.insert("app".into(), "api".into());
                m.insert("version".into(), "v2".into());
                m
            },
            annotations: {
                let mut m = indexmap::IndexMap::new();
                m.insert("owner".into(), "team-platform".into());
                m
            },
            uid: "abc-123".into(),
            resource_version: "42".into(),
        };
        let json = serde_json::to_string(&meta).unwrap();
        let deser: ObjectMeta = serde_json::from_str(&json).unwrap();
        assert_eq!(deser, meta);
    }

    #[test]
    fn test_object_meta_yaml_roundtrip() {
        let meta = ObjectMeta {
            name: "test-svc".into(),
            namespace: "default".into(),
            ..Default::default()
        };
        let yaml = serde_yaml::to_string(&meta).unwrap();
        let deser: ObjectMeta = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(deser.name, "test-svc");
        assert_eq!(deser.namespace, "default");
    }

    #[test]
    fn test_object_meta_equality() {
        let a = ObjectMeta {
            name: "x".into(),
            namespace: "y".into(),
            ..Default::default()
        };
        let b = ObjectMeta {
            name: "x".into(),
            namespace: "y".into(),
            ..Default::default()
        };
        assert_eq!(a, b);
    }

    // -- ServiceId ----------------------------------------------------------

    #[test]
    fn test_service_id_new() {
        let sid = ServiceId::new("api", "default");
        assert_eq!(sid.name, "api");
        assert_eq!(sid.namespace, "default");
    }

    #[test]
    fn test_service_id_display() {
        let sid = ServiceId::new("gateway", "prod");
        assert_eq!(format!("{sid}"), "prod/gateway");
    }

    #[test]
    fn test_service_id_hash_eq() {
        use std::collections::HashSet;
        let a = ServiceId::new("x", "ns");
        let b = ServiceId::new("x", "ns");
        let c = ServiceId::new("y", "ns");
        let mut set = HashSet::new();
        set.insert(a.clone());
        assert!(set.contains(&b));
        assert!(!set.contains(&c));
    }

    #[test]
    fn test_service_id_serde() {
        let sid = ServiceId::new("my-svc", "kube-system");
        let json = serde_json::to_string(&sid).unwrap();
        let deser: ServiceId = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.name, "my-svc");
        assert_eq!(deser.namespace, "kube-system");
    }

    #[test]
    fn test_service_id_from_strings() {
        let sid = ServiceId::new(String::from("svc"), String::from("ns"));
        assert_eq!(sid.name, "svc");
    }

    // -- ServiceEndpoint ----------------------------------------------------

    #[test]
    fn test_service_endpoint_new() {
        let ep = ServiceEndpoint {
            address: "10.0.0.1".into(),
            port: 8080,
            protocol: "http".into(),
        };
        assert_eq!(ep.address, "10.0.0.1");
        assert_eq!(ep.port, 8080);
        assert_eq!(ep.protocol, "http");
    }

    #[test]
    fn test_service_endpoint_serde() {
        let ep = ServiceEndpoint {
            address: "10.0.0.1".into(),
            port: 443,
            protocol: "https".into(),
        };
        let json = serde_json::to_string(&ep).unwrap();
        let deser: ServiceEndpoint = serde_json::from_str(&json).unwrap();
        assert_eq!(deser, ep);
    }

    #[test]
    fn test_service_endpoint_equality() {
        let a = ServiceEndpoint {
            address: "1.2.3.4".into(),
            port: 80,
            protocol: "http".into(),
        };
        let b = ServiceEndpoint {
            address: "1.2.3.4".into(),
            port: 80,
            protocol: "http".into(),
        };
        assert_eq!(a, b);
    }

    // -- RetryPolicy --------------------------------------------------------

    #[test]
    fn test_retry_policy_default() {
        let rp = RetryPolicy::default();
        assert_eq!(rp.max_retries, 2);
        assert_eq!(rp.per_try_timeout_ms, 1000);
        assert_eq!(rp.retry_on, vec!["5xx".to_string()]);
        assert_eq!(rp.backoff_base_ms, 25);
        assert_eq!(rp.backoff_max_ms, 250);
    }

    #[test]
    fn test_retry_policy_serde() {
        let rp = RetryPolicy {
            max_retries: 5,
            per_try_timeout_ms: 2000,
            retry_on: vec!["5xx".into(), "reset".into()],
            backoff_base_ms: 50,
            backoff_max_ms: 500,
        };
        let json = serde_json::to_string(&rp).unwrap();
        let deser: RetryPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(deser, rp);
    }

    #[test]
    fn test_retry_policy_yaml() {
        let rp = RetryPolicy::default();
        let yaml = serde_yaml::to_string(&rp).unwrap();
        assert!(yaml.contains("max_retries"));
        let deser: RetryPolicy = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(deser.max_retries, 2);
    }

    // -- TimeoutPolicy ------------------------------------------------------

    #[test]
    fn test_timeout_policy_default() {
        let tp = TimeoutPolicy::default();
        assert_eq!(tp.request_timeout_ms, 15000);
        assert_eq!(tp.idle_timeout_ms, 300000);
        assert_eq!(tp.connect_timeout_ms, 5000);
    }

    #[test]
    fn test_timeout_policy_serde() {
        let tp = TimeoutPolicy {
            request_timeout_ms: 30000,
            idle_timeout_ms: 60000,
            connect_timeout_ms: 3000,
        };
        let json = serde_json::to_string(&tp).unwrap();
        let deser: TimeoutPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(deser, tp);
    }

    #[test]
    fn test_timeout_policy_yaml() {
        let tp = TimeoutPolicy::default();
        let yaml = serde_yaml::to_string(&tp).unwrap();
        let deser: TimeoutPolicy = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(deser.request_timeout_ms, 15000);
    }

    // -- Severity -----------------------------------------------------------

    #[test]
    fn test_severity_display() {
        assert_eq!(format!("{}", Severity::Info), "INFO");
        assert_eq!(format!("{}", Severity::Warning), "WARNING");
        assert_eq!(format!("{}", Severity::Error), "ERROR");
        assert_eq!(format!("{}", Severity::Critical), "CRITICAL");
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Critical > Severity::Error);
        assert!(Severity::Error > Severity::Warning);
        assert!(Severity::Warning > Severity::Info);
    }

    #[test]
    fn test_severity_serde() {
        let s = Severity::Error;
        let json = serde_json::to_string(&s).unwrap();
        let deser: Severity = serde_json::from_str(&json).unwrap();
        assert_eq!(deser, Severity::Error);
    }

    #[test]
    fn test_severity_copy() {
        let a = Severity::Warning;
        let b = a;
        assert_eq!(a, b);
    }

    // -- Integration / cross-type tests -------------------------------------

    #[test]
    fn test_meta_with_labels_lookup() {
        let mut meta = ObjectMeta::default();
        meta.name = "frontend".into();
        meta.labels.insert("app".into(), "web".into());
        meta.labels.insert("tier".into(), "frontend".into());

        assert_eq!(meta.labels.get("app"), Some(&"web".to_string()));
        assert_eq!(meta.labels.get("tier"), Some(&"frontend".to_string()));
        assert_eq!(meta.labels.get("missing"), None);
    }

    #[test]
    fn test_meta_annotations_iteration() {
        let mut meta = ObjectMeta::default();
        meta.annotations.insert("key1".into(), "val1".into());
        meta.annotations.insert("key2".into(), "val2".into());

        let keys: Vec<&String> = meta.annotations.keys().collect();
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn test_config_types_from_yaml_string() {
        let yaml = r#"
name: my-service
namespace: production
labels:
  app: my-app
  version: v1
annotations: {}
uid: ""
resource_version: ""
"#;
        let meta: ObjectMeta = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(meta.name, "my-service");
        assert_eq!(meta.namespace, "production");
        assert_eq!(meta.labels.get("app"), Some(&"my-app".to_string()));
    }

    #[test]
    fn test_retry_and_timeout_policy_together() {
        let rp = RetryPolicy {
            max_retries: 3,
            per_try_timeout_ms: 1000,
            retry_on: vec!["5xx".into()],
            backoff_base_ms: 100,
            backoff_max_ms: 1000,
        };
        let tp = TimeoutPolicy {
            request_timeout_ms: 5000,
            idle_timeout_ms: 30000,
            connect_timeout_ms: 1000,
        };
        // Worst case: (1 + max_retries) * per_try_timeout
        let worst_case = (1 + rp.max_retries as u64) * rp.per_try_timeout_ms;
        // Should not exceed request timeout for well-configured system
        assert!(
            worst_case <= tp.request_timeout_ms,
            "worst case {worst_case}ms exceeds timeout {}ms",
            tp.request_timeout_ms
        );
    }

    #[test]
    fn test_result_type_alias() {
        fn returns_ok() -> Result<u32> {
            Ok(42)
        }
        assert_eq!(returns_ok().unwrap(), 42);
    }
}
