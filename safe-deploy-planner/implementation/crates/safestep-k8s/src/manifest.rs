//! Kubernetes manifest parsing, representation, and validation.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use safestep_types::SafeStepError;

pub type Result<T> = std::result::Result<T, SafeStepError>;

// ---------------------------------------------------------------------------
// Core manifest types
// ---------------------------------------------------------------------------

/// A parsed Kubernetes manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KubernetesManifest {
    pub api_version: String,
    pub kind: String,
    pub metadata: ManifestMetadata,
    pub spec: Option<Value>,
    /// The full raw YAML value for round-tripping.
    #[serde(skip)]
    pub raw: Option<Value>,
}

impl KubernetesManifest {
    /// Parse a single YAML document into a manifest.
    pub fn parse(yaml: &str) -> Result<Vec<Self>> {
        Self::parse_multi_doc(yaml)
    }

    /// Parse a multi-document YAML string (separated by `---`).
    pub fn parse_multi_doc(yaml: &str) -> Result<Vec<Self>> {
        let mut manifests = Vec::new();
        for doc in split_yaml_docs(yaml) {
            let trimmed = doc.trim();
            if trimmed.is_empty() {
                continue;
            }
            let value: Value = serde_yaml::from_str(trimmed).map_err(|e| {
                SafeStepError::K8sError {
                    message: format!("YAML parse error: {e}"),
                    resource: None,
                    namespace: None,
                    context: None,
                }
            })?;
            if value.is_null() {
                continue;
            }
            let manifest = Self::from_value(value)?;
            manifests.push(manifest);
        }
        Ok(manifests)
    }

    /// Build a manifest from a serde_json::Value.
    pub fn from_value(value: Value) -> Result<Self> {
        let api_version = value
            .get("apiVersion")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let kind = value
            .get("kind")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let metadata = parse_metadata(value.get("metadata"))?;
        let spec = value.get("spec").cloned();
        Ok(Self {
            api_version,
            kind,
            metadata,
            spec,
            raw: Some(value),
        })
    }

    /// Serialize this manifest to a YAML string.
    pub fn to_yaml(&self) -> Result<String> {
        let value = self.to_value();
        serde_yaml::to_string(&value).map_err(|e| SafeStepError::K8sError {
            message: format!("YAML serialize error: {e}"),
            resource: Some(self.metadata.name.clone()),
            namespace: self.metadata.namespace.clone(),
            context: None,
        })
    }

    /// Convert to a serde_json Value.
    pub fn to_value(&self) -> Value {
        if let Some(raw) = &self.raw {
            return raw.clone();
        }
        let mut map = serde_json::Map::new();
        map.insert("apiVersion".into(), Value::String(self.api_version.clone()));
        map.insert("kind".into(), Value::String(self.kind.clone()));
        map.insert("metadata".into(), metadata_to_value(&self.metadata));
        if let Some(spec) = &self.spec {
            map.insert("spec".into(), spec.clone());
        }
        Value::Object(map)
    }

    /// Returns true if this manifest is a workload (Deployment, StatefulSet, DaemonSet, Job).
    pub fn is_workload(&self) -> bool {
        matches!(
            self.kind.as_str(),
            "Deployment" | "StatefulSet" | "DaemonSet" | "Job" | "CronJob"
        )
    }

    /// Try to parse the spec as a DeploymentSpec.
    pub fn as_deployment_spec(&self) -> Option<DeploymentSpec> {
        if self.kind != "Deployment" {
            return None;
        }
        self.spec.as_ref().and_then(|s| parse_deployment_spec(s).ok())
    }

    /// Try to parse the spec as a StatefulSetSpec.
    pub fn as_statefulset_spec(&self) -> Option<StatefulSetSpec> {
        if self.kind != "StatefulSet" {
            return None;
        }
        self.spec.as_ref().and_then(|s| parse_statefulset_spec(s).ok())
    }

    /// Try to parse the spec as a DaemonSetSpec.
    pub fn as_daemonset_spec(&self) -> Option<DaemonSetSpec> {
        if self.kind != "DaemonSet" {
            return None;
        }
        self.spec.as_ref().and_then(|s| parse_daemonset_spec(s).ok())
    }

    /// Try to parse the spec as a ServiceManifest.
    pub fn as_service(&self) -> Option<ServiceManifest> {
        if self.kind != "Service" {
            return None;
        }
        self.spec.as_ref().and_then(|s| parse_service_spec(s).ok())
    }
}

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

/// Metadata section of a Kubernetes manifest.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ManifestMetadata {
    pub name: String,
    pub namespace: Option<String>,
    pub labels: HashMap<String, String>,
    pub annotations: HashMap<String, String>,
}

fn parse_metadata(value: Option<&Value>) -> Result<ManifestMetadata> {
    let Some(v) = value else {
        return Ok(ManifestMetadata::default());
    };
    let name = v.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
    let namespace = v.get("namespace").and_then(|n| n.as_str()).map(String::from);
    let labels = parse_string_map(v.get("labels"));
    let annotations = parse_string_map(v.get("annotations"));
    Ok(ManifestMetadata { name, namespace, labels, annotations })
}

fn metadata_to_value(m: &ManifestMetadata) -> Value {
    let mut map = serde_json::Map::new();
    map.insert("name".into(), Value::String(m.name.clone()));
    if let Some(ns) = &m.namespace {
        map.insert("namespace".into(), Value::String(ns.clone()));
    }
    if !m.labels.is_empty() {
        map.insert("labels".into(), string_map_to_value(&m.labels));
    }
    if !m.annotations.is_empty() {
        map.insert("annotations".into(), string_map_to_value(&m.annotations));
    }
    Value::Object(map)
}

fn parse_string_map(value: Option<&Value>) -> HashMap<String, String> {
    let mut result = HashMap::new();
    if let Some(Value::Object(obj)) = value {
        for (k, v) in obj {
            if let Some(s) = v.as_str() {
                result.insert(k.clone(), s.to_string());
            } else {
                result.insert(k.clone(), v.to_string());
            }
        }
    }
    result
}

fn string_map_to_value(m: &HashMap<String, String>) -> Value {
    let obj: serde_json::Map<String, Value> = m
        .iter()
        .map(|(k, v)| (k.clone(), Value::String(v.clone())))
        .collect();
    Value::Object(obj)
}

// ---------------------------------------------------------------------------
// Deployment
// ---------------------------------------------------------------------------

/// Parsed spec for a Kubernetes Deployment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentSpec {
    pub replicas: u32,
    pub selector: LabelSelector,
    pub template: PodTemplateSpec,
    pub strategy: DeploymentStrategy,
}

/// Deployment update strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeploymentStrategy {
    RollingUpdate {
        max_surge: IntOrString,
        max_unavailable: IntOrString,
    },
    Recreate,
}

impl Default for DeploymentStrategy {
    fn default() -> Self {
        DeploymentStrategy::RollingUpdate {
            max_surge: IntOrString::String("25%".into()),
            max_unavailable: IntOrString::String("25%".into()),
        }
    }
}

/// Either an integer or a string (used for maxSurge/maxUnavailable).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IntOrString {
    Int(i64),
    String(String),
}

impl fmt::Display for IntOrString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntOrString::Int(i) => write!(f, "{i}"),
            IntOrString::String(s) => write!(f, "{s}"),
        }
    }
}

fn parse_int_or_string(v: &Value) -> IntOrString {
    if let Some(i) = v.as_i64() {
        IntOrString::Int(i)
    } else if let Some(s) = v.as_str() {
        IntOrString::String(s.to_string())
    } else {
        IntOrString::String(v.to_string())
    }
}

fn parse_deployment_spec(spec: &Value) -> Result<DeploymentSpec> {
    let replicas = spec
        .get("replicas")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as u32;
    let selector = parse_label_selector(spec.get("selector"))?;
    let template = parse_pod_template(spec.get("template"))?;
    let strategy = parse_deployment_strategy(spec.get("strategy"));
    Ok(DeploymentSpec { replicas, selector, template, strategy })
}

fn parse_deployment_strategy(v: Option<&Value>) -> DeploymentStrategy {
    let Some(v) = v else { return DeploymentStrategy::default() };
    let type_str = v.get("type").and_then(|t| t.as_str()).unwrap_or("RollingUpdate");
    match type_str {
        "Recreate" => DeploymentStrategy::Recreate,
        _ => {
            let ru = v.get("rollingUpdate");
            let max_surge = ru
                .and_then(|r| r.get("maxSurge"))
                .map(parse_int_or_string)
                .unwrap_or(IntOrString::String("25%".into()));
            let max_unavailable = ru
                .and_then(|r| r.get("maxUnavailable"))
                .map(parse_int_or_string)
                .unwrap_or(IntOrString::String("25%".into()));
            DeploymentStrategy::RollingUpdate { max_surge, max_unavailable }
        }
    }
}

// ---------------------------------------------------------------------------
// StatefulSet
// ---------------------------------------------------------------------------

/// Parsed spec for a Kubernetes StatefulSet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatefulSetSpec {
    pub replicas: u32,
    pub service_name: String,
    pub selector: LabelSelector,
    pub template: PodTemplateSpec,
    pub volume_claim_templates: Vec<Value>,
    pub update_strategy: StatefulSetUpdateStrategy,
}

/// StatefulSet update strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatefulSetUpdateStrategy {
    RollingUpdate { partition: u32 },
    OnDelete,
}

impl Default for StatefulSetUpdateStrategy {
    fn default() -> Self {
        StatefulSetUpdateStrategy::RollingUpdate { partition: 0 }
    }
}

fn parse_statefulset_spec(spec: &Value) -> Result<StatefulSetSpec> {
    let replicas = spec.get("replicas").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
    let service_name = spec
        .get("serviceName")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let selector = parse_label_selector(spec.get("selector"))?;
    let template = parse_pod_template(spec.get("template"))?;
    let volume_claim_templates = spec
        .get("volumeClaimTemplates")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let update_strategy = parse_statefulset_update_strategy(spec.get("updateStrategy"));
    Ok(StatefulSetSpec {
        replicas,
        service_name,
        selector,
        template,
        volume_claim_templates,
        update_strategy,
    })
}

fn parse_statefulset_update_strategy(v: Option<&Value>) -> StatefulSetUpdateStrategy {
    let Some(v) = v else { return StatefulSetUpdateStrategy::default() };
    let type_str = v.get("type").and_then(|t| t.as_str()).unwrap_or("RollingUpdate");
    match type_str {
        "OnDelete" => StatefulSetUpdateStrategy::OnDelete,
        _ => {
            let partition = v
                .get("rollingUpdate")
                .and_then(|r| r.get("partition"))
                .and_then(|p| p.as_u64())
                .unwrap_or(0) as u32;
            StatefulSetUpdateStrategy::RollingUpdate { partition }
        }
    }
}

// ---------------------------------------------------------------------------
// DaemonSet
// ---------------------------------------------------------------------------

/// Parsed spec for a Kubernetes DaemonSet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonSetSpec {
    pub selector: LabelSelector,
    pub template: PodTemplateSpec,
    pub update_strategy: DaemonSetUpdateStrategy,
}

/// DaemonSet update strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DaemonSetUpdateStrategy {
    RollingUpdate { max_unavailable: IntOrString },
    OnDelete,
}

impl Default for DaemonSetUpdateStrategy {
    fn default() -> Self {
        DaemonSetUpdateStrategy::RollingUpdate {
            max_unavailable: IntOrString::Int(1),
        }
    }
}

fn parse_daemonset_spec(spec: &Value) -> Result<DaemonSetSpec> {
    let selector = parse_label_selector(spec.get("selector"))?;
    let template = parse_pod_template(spec.get("template"))?;
    let update_strategy = parse_daemonset_update_strategy(spec.get("updateStrategy"));
    Ok(DaemonSetSpec { selector, template, update_strategy })
}

fn parse_daemonset_update_strategy(v: Option<&Value>) -> DaemonSetUpdateStrategy {
    let Some(v) = v else { return DaemonSetUpdateStrategy::default() };
    let type_str = v.get("type").and_then(|t| t.as_str()).unwrap_or("RollingUpdate");
    match type_str {
        "OnDelete" => DaemonSetUpdateStrategy::OnDelete,
        _ => {
            let max_unavailable = v
                .get("rollingUpdate")
                .and_then(|r| r.get("maxUnavailable"))
                .map(parse_int_or_string)
                .unwrap_or(IntOrString::Int(1));
            DaemonSetUpdateStrategy::RollingUpdate { max_unavailable }
        }
    }
}

// ---------------------------------------------------------------------------
// Label selector
// ---------------------------------------------------------------------------

/// Kubernetes label selector.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LabelSelector {
    pub match_labels: HashMap<String, String>,
    pub match_expressions: Vec<LabelSelectorRequirement>,
}

/// A single requirement in a label selector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelSelectorRequirement {
    pub key: String,
    pub operator: String,
    pub values: Vec<String>,
}

fn parse_label_selector(v: Option<&Value>) -> Result<LabelSelector> {
    let Some(v) = v else { return Ok(LabelSelector::default()) };
    let match_labels = parse_string_map(v.get("matchLabels"));
    let match_expressions = v
        .get("matchExpressions")
        .and_then(|arr| arr.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    let key = item.get("key")?.as_str()?.to_string();
                    let operator = item.get("operator")?.as_str()?.to_string();
                    let values = item
                        .get("values")
                        .and_then(|v| v.as_array())
                        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
                        .unwrap_or_default();
                    Some(LabelSelectorRequirement { key, operator, values })
                })
                .collect()
        })
        .unwrap_or_default();
    Ok(LabelSelector { match_labels, match_expressions })
}

// ---------------------------------------------------------------------------
// Pod template / Pod spec
// ---------------------------------------------------------------------------

/// Pod template specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodTemplateSpec {
    pub metadata: ManifestMetadata,
    pub spec: PodSpec,
}

fn parse_pod_template(v: Option<&Value>) -> Result<PodTemplateSpec> {
    let Some(v) = v else {
        return Ok(PodTemplateSpec {
            metadata: ManifestMetadata::default(),
            spec: PodSpec::default(),
        });
    };
    let metadata = parse_metadata(v.get("metadata"))?;
    let spec = parse_pod_spec(v.get("spec"))?;
    Ok(PodTemplateSpec { metadata, spec })
}

/// Pod specification.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PodSpec {
    pub containers: Vec<ContainerSpec>,
    pub init_containers: Vec<ContainerSpec>,
    pub volumes: Vec<Value>,
    pub node_selector: HashMap<String, String>,
    pub affinity: Option<Value>,
    pub tolerations: Vec<Value>,
    pub service_account_name: Option<String>,
    pub restart_policy: Option<String>,
}

fn parse_pod_spec(v: Option<&Value>) -> Result<PodSpec> {
    let Some(v) = v else { return Ok(PodSpec::default()) };
    let containers = v
        .get("containers")
        .and_then(|c| c.as_array())
        .map(|arr| arr.iter().filter_map(|c| parse_container(c).ok()).collect())
        .unwrap_or_default();
    let init_containers = v
        .get("initContainers")
        .and_then(|c| c.as_array())
        .map(|arr| arr.iter().filter_map(|c| parse_container(c).ok()).collect())
        .unwrap_or_default();
    let volumes = v
        .get("volumes")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let node_selector = parse_string_map(v.get("nodeSelector"));
    let affinity = v.get("affinity").cloned();
    let tolerations = v
        .get("tolerations")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let service_account_name = v.get("serviceAccountName").and_then(|s| s.as_str()).map(String::from);
    let restart_policy = v.get("restartPolicy").and_then(|s| s.as_str()).map(String::from);
    Ok(PodSpec {
        containers,
        init_containers,
        volumes,
        node_selector,
        affinity,
        tolerations,
        service_account_name,
        restart_policy,
    })
}

// ---------------------------------------------------------------------------
// Container spec
// ---------------------------------------------------------------------------

/// Container specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerSpec {
    pub name: String,
    pub image: String,
    pub ports: Vec<ContainerPort>,
    pub env: Vec<EnvVar>,
    pub resources: Option<ResourceSpec>,
    pub liveness_probe: Option<Probe>,
    pub readiness_probe: Option<Probe>,
    pub startup_probe: Option<Probe>,
    pub image_pull_policy: Option<String>,
    pub command: Vec<String>,
    pub args: Vec<String>,
    pub volume_mounts: Vec<Value>,
}

/// A container port definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerPort {
    pub name: Option<String>,
    pub container_port: u16,
    pub protocol: String,
}

/// An environment variable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvVar {
    pub name: String,
    pub value: Option<String>,
    pub value_from: Option<Value>,
}

/// A health-check probe.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Probe {
    pub http_get: Option<HttpGetAction>,
    pub tcp_socket: Option<TcpSocketAction>,
    pub exec_action: Option<ExecAction>,
    pub initial_delay_seconds: u32,
    pub period_seconds: u32,
    pub timeout_seconds: u32,
    pub success_threshold: u32,
    pub failure_threshold: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpGetAction {
    pub path: String,
    pub port: IntOrString,
    pub scheme: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TcpSocketAction {
    pub port: IntOrString,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecAction {
    pub command: Vec<String>,
}

fn parse_container(v: &Value) -> Result<ContainerSpec> {
    let name = v.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
    let image = v.get("image").and_then(|n| n.as_str()).unwrap_or("").to_string();
    let ports = v
        .get("ports")
        .and_then(|p| p.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|p| {
                    let container_port = p.get("containerPort")?.as_u64()? as u16;
                    let port_name = p.get("name").and_then(|n| n.as_str()).map(String::from);
                    let protocol = p
                        .get("protocol")
                        .and_then(|pr| pr.as_str())
                        .unwrap_or("TCP")
                        .to_string();
                    Some(ContainerPort { name: port_name, container_port, protocol })
                })
                .collect()
        })
        .unwrap_or_default();
    let env = v
        .get("env")
        .and_then(|e| e.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|e| {
                    let env_name = e.get("name")?.as_str()?.to_string();
                    let value = e.get("value").and_then(|v| v.as_str()).map(String::from);
                    let value_from = e.get("valueFrom").cloned();
                    Some(EnvVar { name: env_name, value, value_from })
                })
                .collect()
        })
        .unwrap_or_default();
    let resources = v.get("resources").and_then(|r| parse_resource_spec(r).ok());
    let liveness_probe = v.get("livenessProbe").and_then(|p| parse_probe(p));
    let readiness_probe = v.get("readinessProbe").and_then(|p| parse_probe(p));
    let startup_probe = v.get("startupProbe").and_then(|p| parse_probe(p));
    let image_pull_policy = v.get("imagePullPolicy").and_then(|s| s.as_str()).map(String::from);
    let command = parse_string_array(v.get("command"));
    let args = parse_string_array(v.get("args"));
    let volume_mounts = v
        .get("volumeMounts")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    Ok(ContainerSpec {
        name,
        image,
        ports,
        env,
        resources,
        liveness_probe,
        readiness_probe,
        startup_probe,
        image_pull_policy,
        command,
        args,
        volume_mounts,
    })
}

fn parse_string_array(v: Option<&Value>) -> Vec<String> {
    v.and_then(|a| a.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

fn parse_probe(v: &Value) -> Option<Probe> {
    let http_get = v.get("httpGet").map(|h| HttpGetAction {
        path: h.get("path").and_then(|p| p.as_str()).unwrap_or("/").to_string(),
        port: h
            .get("port")
            .map(parse_int_or_string)
            .unwrap_or(IntOrString::Int(80)),
        scheme: h.get("scheme").and_then(|s| s.as_str()).unwrap_or("HTTP").to_string(),
    });
    let tcp_socket = v.get("tcpSocket").map(|t| TcpSocketAction {
        port: t
            .get("port")
            .map(parse_int_or_string)
            .unwrap_or(IntOrString::Int(80)),
    });
    let exec_action = v.get("exec").map(|e| ExecAction {
        command: parse_string_array(e.get("command")),
    });
    let initial_delay_seconds = v.get("initialDelaySeconds").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
    let period_seconds = v.get("periodSeconds").and_then(|v| v.as_u64()).unwrap_or(10) as u32;
    let timeout_seconds = v.get("timeoutSeconds").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
    let success_threshold = v.get("successThreshold").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
    let failure_threshold = v.get("failureThreshold").and_then(|v| v.as_u64()).unwrap_or(3) as u32;
    Some(Probe {
        http_get,
        tcp_socket,
        exec_action,
        initial_delay_seconds,
        period_seconds,
        timeout_seconds,
        success_threshold,
        failure_threshold,
    })
}

// ---------------------------------------------------------------------------
// Resource spec / quantities
// ---------------------------------------------------------------------------

/// Resource requests and limits for a container.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceSpec {
    pub cpu_request: Option<ResourceQuantity>,
    pub cpu_limit: Option<ResourceQuantity>,
    pub memory_request: Option<ResourceQuantity>,
    pub memory_limit: Option<ResourceQuantity>,
}

/// A parsed Kubernetes resource quantity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceQuantity {
    /// The raw string (e.g. "500m", "1Gi").
    pub raw: String,
    /// Numeric value in base units (CPU cores or bytes).
    pub value: f64,
}

impl ResourceQuantity {
    /// Parse a CPU quantity string into cores (e.g. "500m" -> 0.5, "2" -> 2.0).
    pub fn parse_cpu(s: &str) -> Result<Self> {
        let s = s.trim();
        let value = if s.ends_with('m') {
            let millis: f64 = s[..s.len() - 1].parse().map_err(|_| SafeStepError::K8sError {
                message: format!("Invalid CPU quantity: {s}"),
                resource: None,
                namespace: None,
                context: None,
            })?;
            millis / 1000.0
        } else {
            s.parse::<f64>().map_err(|_| SafeStepError::K8sError {
                message: format!("Invalid CPU quantity: {s}"),
                resource: None,
                namespace: None,
                context: None,
            })?
        };
        Ok(ResourceQuantity { raw: s.to_string(), value })
    }

    /// Parse a memory quantity string into bytes (e.g. "100Mi" -> 104857600).
    pub fn parse_memory(s: &str) -> Result<Self> {
        let s = s.trim();
        let (num_str, multiplier) = if s.ends_with("Ki") {
            (&s[..s.len() - 2], 1024.0)
        } else if s.ends_with("Mi") {
            (&s[..s.len() - 2], 1024.0 * 1024.0)
        } else if s.ends_with("Gi") {
            (&s[..s.len() - 2], 1024.0 * 1024.0 * 1024.0)
        } else if s.ends_with("Ti") {
            (&s[..s.len() - 2], 1024.0 * 1024.0 * 1024.0 * 1024.0)
        } else if s.ends_with('k') || s.ends_with('K') {
            (&s[..s.len() - 1], 1000.0)
        } else if s.ends_with('M') {
            (&s[..s.len() - 1], 1_000_000.0)
        } else if s.ends_with('G') {
            (&s[..s.len() - 1], 1_000_000_000.0)
        } else if s.ends_with('T') {
            (&s[..s.len() - 1], 1_000_000_000_000.0)
        } else if s.ends_with('E') {
            (&s[..s.len() - 1], 1e18)
        } else if s.ends_with("Ei") {
            (&s[..s.len() - 2], (1024.0_f64).powi(6))
        } else if s.ends_with("Pi") {
            (&s[..s.len() - 2], (1024.0_f64).powi(5))
        } else {
            (s, 1.0)
        };
        let num: f64 = num_str.parse().map_err(|_| SafeStepError::K8sError {
            message: format!("Invalid memory quantity: {s}"),
            resource: None,
            namespace: None,
            context: None,
        })?;
        Ok(ResourceQuantity {
            raw: s.to_string(),
            value: num * multiplier,
        })
    }

    /// Format as human-readable CPU string.
    pub fn format_cpu(&self) -> String {
        if self.value < 1.0 {
            format!("{}m", (self.value * 1000.0).round() as i64)
        } else {
            format!("{}", self.value)
        }
    }

    /// Format as human-readable memory string.
    pub fn format_memory(&self) -> String {
        let bytes = self.value;
        if bytes >= 1024.0 * 1024.0 * 1024.0 {
            format!("{:.1}Gi", bytes / (1024.0 * 1024.0 * 1024.0))
        } else if bytes >= 1024.0 * 1024.0 {
            format!("{:.0}Mi", bytes / (1024.0 * 1024.0))
        } else if bytes >= 1024.0 {
            format!("{:.0}Ki", bytes / 1024.0)
        } else {
            format!("{bytes}")
        }
    }
}

impl fmt::Display for ResourceQuantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.raw)
    }
}

fn parse_resource_spec(v: &Value) -> Result<ResourceSpec> {
    let requests = v.get("requests");
    let limits = v.get("limits");
    let cpu_request = requests
        .and_then(|r| r.get("cpu"))
        .and_then(|c| c.as_str())
        .and_then(|s| ResourceQuantity::parse_cpu(s).ok());
    let cpu_limit = limits
        .and_then(|r| r.get("cpu"))
        .and_then(|c| c.as_str())
        .and_then(|s| ResourceQuantity::parse_cpu(s).ok());
    let memory_request = requests
        .and_then(|r| r.get("memory"))
        .and_then(|c| c.as_str())
        .and_then(|s| ResourceQuantity::parse_memory(s).ok());
    let memory_limit = limits
        .and_then(|r| r.get("memory"))
        .and_then(|c| c.as_str())
        .and_then(|s| ResourceQuantity::parse_memory(s).ok());
    Ok(ResourceSpec { cpu_request, cpu_limit, memory_request, memory_limit })
}

// ---------------------------------------------------------------------------
// Service
// ---------------------------------------------------------------------------

/// Parsed spec for a Kubernetes Service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceManifest {
    pub type_: ServiceType,
    pub ports: Vec<ServicePort>,
    pub selector: HashMap<String, String>,
    pub cluster_ip: Option<String>,
}

/// Kubernetes Service type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ServiceType {
    ClusterIP,
    NodePort,
    LoadBalancer,
    ExternalName,
}

impl Default for ServiceType {
    fn default() -> Self {
        ServiceType::ClusterIP
    }
}

/// A service port definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServicePort {
    pub name: Option<String>,
    pub port: u16,
    pub target_port: IntOrString,
    pub protocol: String,
    pub node_port: Option<u16>,
}

fn parse_service_spec(spec: &Value) -> Result<ServiceManifest> {
    let type_str = spec.get("type").and_then(|t| t.as_str()).unwrap_or("ClusterIP");
    let type_ = match type_str {
        "NodePort" => ServiceType::NodePort,
        "LoadBalancer" => ServiceType::LoadBalancer,
        "ExternalName" => ServiceType::ExternalName,
        _ => ServiceType::ClusterIP,
    };
    let ports = spec
        .get("ports")
        .and_then(|p| p.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|p| {
                    let port = p.get("port")?.as_u64()? as u16;
                    let target_port = p
                        .get("targetPort")
                        .map(parse_int_or_string)
                        .unwrap_or(IntOrString::Int(port as i64));
                    let name = p.get("name").and_then(|n| n.as_str()).map(String::from);
                    let protocol = p.get("protocol").and_then(|pr| pr.as_str()).unwrap_or("TCP").to_string();
                    let node_port = p.get("nodePort").and_then(|n| n.as_u64()).map(|n| n as u16);
                    Some(ServicePort { name, port, target_port, protocol, node_port })
                })
                .collect()
        })
        .unwrap_or_default();
    let selector = parse_string_map(spec.get("selector"));
    let cluster_ip = spec.get("clusterIP").and_then(|s| s.as_str()).map(String::from);
    Ok(ServiceManifest { type_, ports, selector, cluster_ip })
}

// ---------------------------------------------------------------------------
// PodDisruptionBudget
// ---------------------------------------------------------------------------

/// Parsed PodDisruptionBudget spec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodDisruptionBudget {
    pub min_available: Option<IntOrString>,
    pub max_unavailable: Option<IntOrString>,
    pub selector: LabelSelector,
}

impl PodDisruptionBudget {
    /// Parse a PDB from a manifest value.
    pub fn from_manifest(manifest: &KubernetesManifest) -> Option<Self> {
        if manifest.kind != "PodDisruptionBudget" {
            return None;
        }
        let spec = manifest.spec.as_ref()?;
        let min_available = spec.get("minAvailable").map(parse_int_or_string);
        let max_unavailable = spec.get("maxUnavailable").map(parse_int_or_string);
        let selector = parse_label_selector(spec.get("selector")).ok()?;
        Some(PodDisruptionBudget { min_available, max_unavailable, selector })
    }
}

// ---------------------------------------------------------------------------
// Manifest validator
// ---------------------------------------------------------------------------

/// Validation error found in a manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    pub path: String,
    pub message: String,
    pub severity: ValidationSeverity,
}

/// Severity of a validation finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationSeverity {
    Error,
    Warning,
    Info,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:?}] {}: {}", self.severity, self.path, self.message)
    }
}

/// Validates Kubernetes manifest structure and common mistakes.
pub struct ManifestValidator {
    pub strict_mode: bool,
    pub required_labels: Vec<String>,
    pub required_annotations: Vec<String>,
}

impl Default for ManifestValidator {
    fn default() -> Self {
        Self {
            strict_mode: false,
            required_labels: Vec::new(),
            required_annotations: Vec::new(),
        }
    }
}

impl ManifestValidator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn strict() -> Self {
        Self {
            strict_mode: true,
            required_labels: vec!["app.kubernetes.io/name".into(), "app.kubernetes.io/version".into()],
            required_annotations: Vec::new(),
        }
    }

    /// Validate a manifest and return all findings.
    pub fn validate(&self, manifest: &KubernetesManifest) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        self.validate_basic(manifest, &mut errors);
        self.validate_metadata(manifest, &mut errors);
        match manifest.kind.as_str() {
            "Deployment" => self.validate_deployment(manifest, &mut errors),
            "StatefulSet" => self.validate_statefulset(manifest, &mut errors),
            "Service" => self.validate_service(manifest, &mut errors),
            _ => {}
        }
        if self.strict_mode {
            self.validate_strict(manifest, &mut errors);
        }
        errors
    }

    fn validate_basic(&self, manifest: &KubernetesManifest, errors: &mut Vec<ValidationError>) {
        if manifest.api_version.is_empty() {
            errors.push(ValidationError {
                path: "apiVersion".into(),
                message: "apiVersion is required".into(),
                severity: ValidationSeverity::Error,
            });
        }
        if manifest.kind.is_empty() {
            errors.push(ValidationError {
                path: "kind".into(),
                message: "kind is required".into(),
                severity: ValidationSeverity::Error,
            });
        }
        if manifest.metadata.name.is_empty() {
            errors.push(ValidationError {
                path: "metadata.name".into(),
                message: "metadata.name is required".into(),
                severity: ValidationSeverity::Error,
            });
        }
        // Validate name is a valid DNS subdomain name
        if !manifest.metadata.name.is_empty() && !is_valid_dns_name(&manifest.metadata.name) {
            errors.push(ValidationError {
                path: "metadata.name".into(),
                message: format!(
                    "name '{}' is not a valid DNS subdomain name",
                    manifest.metadata.name
                ),
                severity: ValidationSeverity::Error,
            });
        }
    }

    fn validate_metadata(&self, manifest: &KubernetesManifest, errors: &mut Vec<ValidationError>) {
        for required_label in &self.required_labels {
            if !manifest.metadata.labels.contains_key(required_label) {
                errors.push(ValidationError {
                    path: format!("metadata.labels.{required_label}"),
                    message: format!("required label '{required_label}' is missing"),
                    severity: ValidationSeverity::Warning,
                });
            }
        }
        for required_annotation in &self.required_annotations {
            if !manifest.metadata.annotations.contains_key(required_annotation) {
                errors.push(ValidationError {
                    path: format!("metadata.annotations.{required_annotation}"),
                    message: format!("required annotation '{required_annotation}' is missing"),
                    severity: ValidationSeverity::Warning,
                });
            }
        }
        // Validate label values
        for (k, v) in &manifest.metadata.labels {
            if v.len() > 63 {
                errors.push(ValidationError {
                    path: format!("metadata.labels.{k}"),
                    message: format!("label value exceeds 63 characters (len={})", v.len()),
                    severity: ValidationSeverity::Error,
                });
            }
        }
    }

    fn validate_deployment(&self, manifest: &KubernetesManifest, errors: &mut Vec<ValidationError>) {
        if let Some(spec) = &manifest.spec {
            let replicas = spec.get("replicas").and_then(|v| v.as_u64()).unwrap_or(1);
            if replicas == 0 {
                errors.push(ValidationError {
                    path: "spec.replicas".into(),
                    message: "replicas is 0; deployment will have no pods".into(),
                    severity: ValidationSeverity::Warning,
                });
            }
            if let Some(template) = spec.get("template").and_then(|t| t.get("spec")) {
                self.validate_pod_spec(template, "spec.template.spec", errors);
            }
            // Check selector matches template labels
            if let (Some(sel), Some(tpl_labels)) = (
                spec.get("selector")
                    .and_then(|s| s.get("matchLabels"))
                    .and_then(|m| m.as_object()),
                spec.get("template")
                    .and_then(|t| t.get("metadata"))
                    .and_then(|m| m.get("labels"))
                    .and_then(|l| l.as_object()),
            ) {
                for (k, v) in sel {
                    match tpl_labels.get(k) {
                        None => {
                            errors.push(ValidationError {
                                path: format!("spec.selector.matchLabels.{k}"),
                                message: format!("selector label '{k}' not found in pod template labels"),
                                severity: ValidationSeverity::Error,
                            });
                        }
                        Some(tv) if tv != v => {
                            errors.push(ValidationError {
                                path: format!("spec.selector.matchLabels.{k}"),
                                message: format!(
                                    "selector label '{k}' value mismatch: selector={}, template={}",
                                    v, tv
                                ),
                                severity: ValidationSeverity::Error,
                            });
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    fn validate_statefulset(&self, manifest: &KubernetesManifest, errors: &mut Vec<ValidationError>) {
        if let Some(spec) = &manifest.spec {
            if spec.get("serviceName").and_then(|v| v.as_str()).unwrap_or("").is_empty() {
                errors.push(ValidationError {
                    path: "spec.serviceName".into(),
                    message: "serviceName is required for StatefulSet".into(),
                    severity: ValidationSeverity::Error,
                });
            }
            if let Some(template) = spec.get("template").and_then(|t| t.get("spec")) {
                self.validate_pod_spec(template, "spec.template.spec", errors);
            }
        }
    }

    fn validate_service(&self, manifest: &KubernetesManifest, errors: &mut Vec<ValidationError>) {
        if let Some(spec) = &manifest.spec {
            let ports = spec.get("ports").and_then(|p| p.as_array());
            if ports.map(|p| p.is_empty()).unwrap_or(true) {
                errors.push(ValidationError {
                    path: "spec.ports".into(),
                    message: "service must define at least one port".into(),
                    severity: ValidationSeverity::Error,
                });
            }
            if let Some(arr) = ports {
                for (i, p) in arr.iter().enumerate() {
                    if let Some(port) = p.get("port").and_then(|v| v.as_u64()) {
                        if port > 65535 {
                            errors.push(ValidationError {
                                path: format!("spec.ports[{i}].port"),
                                message: format!("port {port} is out of range (1-65535)"),
                                severity: ValidationSeverity::Error,
                            });
                        }
                    }
                }
            }
        }
    }

    fn validate_pod_spec(&self, spec: &Value, prefix: &str, errors: &mut Vec<ValidationError>) {
        if let Some(containers) = spec.get("containers").and_then(|c| c.as_array()) {
            if containers.is_empty() {
                errors.push(ValidationError {
                    path: format!("{prefix}.containers"),
                    message: "at least one container is required".into(),
                    severity: ValidationSeverity::Error,
                });
            }
            for (i, c) in containers.iter().enumerate() {
                let path = format!("{prefix}.containers[{i}]");
                if c.get("name").and_then(|n| n.as_str()).unwrap_or("").is_empty() {
                    errors.push(ValidationError {
                        path: format!("{path}.name"),
                        message: "container name is required".into(),
                        severity: ValidationSeverity::Error,
                    });
                }
                if c.get("image").and_then(|n| n.as_str()).unwrap_or("").is_empty() {
                    errors.push(ValidationError {
                        path: format!("{path}.image"),
                        message: "container image is required".into(),
                        severity: ValidationSeverity::Warning,
                    });
                }
                // Warn if no resource limits
                if self.strict_mode && c.get("resources").is_none() {
                    errors.push(ValidationError {
                        path: format!("{path}.resources"),
                        message: "container has no resource requests/limits".into(),
                        severity: ValidationSeverity::Warning,
                    });
                }
            }
        }
    }

    fn validate_strict(&self, manifest: &KubernetesManifest, errors: &mut Vec<ValidationError>) {
        // In strict mode, workloads must have resource limits
        if manifest.is_workload() {
            if let Some(spec) = &manifest.spec {
                if let Some(containers) = spec
                    .get("template")
                    .and_then(|t| t.get("spec"))
                    .and_then(|s| s.get("containers"))
                    .and_then(|c| c.as_array())
                {
                    for (i, c) in containers.iter().enumerate() {
                        if c.get("resources").and_then(|r| r.get("limits")).is_none() {
                            errors.push(ValidationError {
                                path: format!("spec.template.spec.containers[{i}].resources.limits"),
                                message: "resource limits are required in strict mode".into(),
                                severity: ValidationSeverity::Error,
                            });
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Split a YAML string into individual documents on `---` boundaries.
fn split_yaml_docs(yaml: &str) -> Vec<&str> {
    let mut docs = Vec::new();
    let mut start = 0;
    for (i, line) in yaml.lines().enumerate() {
        if line.trim() == "---" {
            let byte_offset = yaml
                .lines()
                .take(i)
                .map(|l| l.len() + 1) // +1 for newline
                .sum::<usize>();
            if byte_offset > start {
                docs.push(&yaml[start..byte_offset]);
            }
            start = byte_offset + line.len() + 1;
        }
    }
    if start < yaml.len() {
        docs.push(&yaml[start..]);
    }
    if docs.is_empty() && !yaml.trim().is_empty() {
        docs.push(yaml);
    }
    docs
}

fn is_valid_dns_name(name: &str) -> bool {
    if name.is_empty() || name.len() > 253 {
        return false;
    }
    let re = regex::Regex::new(r"^[a-z0-9]([a-z0-9\-\.]*[a-z0-9])?$").unwrap();
    re.is_match(name)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const DEPLOYMENT_YAML: &str = r#"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  namespace: default
  labels:
    app: nginx
    app.kubernetes.io/name: nginx
    app.kubernetes.io/version: "1.21"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.21.0
        ports:
        - containerPort: 80
        resources:
          requests:
            cpu: "250m"
            memory: "128Mi"
          limits:
            cpu: "500m"
            memory: "256Mi"
        livenessProbe:
          httpGet:
            path: /healthz
            port: 80
          initialDelaySeconds: 10
          periodSeconds: 5
        readinessProbe:
          httpGet:
            path: /ready
            port: 80
          initialDelaySeconds: 5
          periodSeconds: 3
        env:
        - name: ENV
          value: production
        - name: DB_HOST
          value: postgres.default.svc.cluster.local
"#;

    const MULTI_DOC_YAML: &str = r#"---
apiVersion: v1
kind: Service
metadata:
  name: nginx-service
spec:
  type: ClusterIP
  selector:
    app: nginx
  ports:
  - port: 80
    targetPort: 80
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
  namespace: default
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:latest
"#;

    const STATEFULSET_YAML: &str = r#"
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: db
spec:
  serviceName: postgres-headless
  replicas: 3
  selector:
    matchLabels:
      app: postgres
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      partition: 1
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15.2
        ports:
        - containerPort: 5432
        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
"#;

    const DAEMONSET_YAML: &str = r#"
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluentd
  namespace: logging
spec:
  selector:
    matchLabels:
      app: fluentd
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
  template:
    metadata:
      labels:
        app: fluentd
    spec:
      tolerations:
      - key: node-role.kubernetes.io/master
        effect: NoSchedule
      containers:
      - name: fluentd
        image: fluentd:v1.16
        resources:
          requests:
            cpu: "100m"
            memory: "200Mi"
          limits:
            cpu: "500m"
            memory: "500Mi"
"#;

    const PDB_YAML: &str = r#"
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: nginx-pdb
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: nginx
"#;

    #[test]
    fn test_parse_deployment() {
        let manifests = KubernetesManifest::parse(DEPLOYMENT_YAML).unwrap();
        assert_eq!(manifests.len(), 1);
        let m = &manifests[0];
        assert_eq!(m.api_version, "apps/v1");
        assert_eq!(m.kind, "Deployment");
        assert_eq!(m.metadata.name, "nginx-deployment");
        assert_eq!(m.metadata.namespace.as_deref(), Some("default"));
        assert_eq!(m.metadata.labels.get("app").unwrap(), "nginx");

        let spec = m.as_deployment_spec().unwrap();
        assert_eq!(spec.replicas, 3);
        assert_eq!(spec.selector.match_labels.get("app").unwrap(), "nginx");
        match &spec.strategy {
            DeploymentStrategy::RollingUpdate { max_surge, max_unavailable } => {
                assert!(matches!(max_surge, IntOrString::Int(1)));
                assert!(matches!(max_unavailable, IntOrString::Int(0)));
            }
            _ => panic!("Expected RollingUpdate strategy"),
        }
        let container = &spec.template.spec.containers[0];
        assert_eq!(container.name, "nginx");
        assert_eq!(container.image, "nginx:1.21.0");
        assert_eq!(container.ports[0].container_port, 80);
        let res = container.resources.as_ref().unwrap();
        assert!((res.cpu_request.as_ref().unwrap().value - 0.25).abs() < 0.001);
        assert!((res.cpu_limit.as_ref().unwrap().value - 0.5).abs() < 0.001);
        assert!((res.memory_request.as_ref().unwrap().value - 128.0 * 1024.0 * 1024.0).abs() < 1.0);
    }

    #[test]
    fn test_parse_multi_doc() {
        let manifests = KubernetesManifest::parse_multi_doc(MULTI_DOC_YAML).unwrap();
        assert_eq!(manifests.len(), 2);
        assert_eq!(manifests[0].kind, "Service");
        assert_eq!(manifests[1].kind, "Deployment");
    }

    #[test]
    fn test_parse_statefulset() {
        let manifests = KubernetesManifest::parse(STATEFULSET_YAML).unwrap();
        let m = &manifests[0];
        assert_eq!(m.kind, "StatefulSet");
        let spec = m.as_statefulset_spec().unwrap();
        assert_eq!(spec.replicas, 3);
        assert_eq!(spec.service_name, "postgres-headless");
        assert!(!spec.volume_claim_templates.is_empty());
        match &spec.update_strategy {
            StatefulSetUpdateStrategy::RollingUpdate { partition } => assert_eq!(*partition, 1),
            _ => panic!("Expected RollingUpdate"),
        }
    }

    #[test]
    fn test_parse_daemonset() {
        let manifests = KubernetesManifest::parse(DAEMONSET_YAML).unwrap();
        let m = &manifests[0];
        assert_eq!(m.kind, "DaemonSet");
        let spec = m.as_daemonset_spec().unwrap();
        assert_eq!(spec.selector.match_labels.get("app").unwrap(), "fluentd");
        match &spec.update_strategy {
            DaemonSetUpdateStrategy::RollingUpdate { max_unavailable } => {
                assert!(matches!(max_unavailable, IntOrString::Int(1)));
            }
            _ => panic!("Expected RollingUpdate"),
        }
    }

    #[test]
    fn test_parse_service() {
        let manifests = KubernetesManifest::parse_multi_doc(MULTI_DOC_YAML).unwrap();
        let svc = manifests[0].as_service().unwrap();
        assert_eq!(svc.type_, ServiceType::ClusterIP);
        assert_eq!(svc.ports.len(), 1);
        assert_eq!(svc.ports[0].port, 80);
        assert_eq!(svc.selector.get("app").unwrap(), "nginx");
    }

    #[test]
    fn test_parse_pdb() {
        let manifests = KubernetesManifest::parse(PDB_YAML).unwrap();
        let pdb = PodDisruptionBudget::from_manifest(&manifests[0]).unwrap();
        assert!(matches!(&pdb.min_available, Some(IntOrString::Int(2))));
        assert!(pdb.max_unavailable.is_none());
        assert_eq!(pdb.selector.match_labels.get("app").unwrap(), "nginx");
    }

    #[test]
    fn test_resource_quantity_cpu() {
        let q = ResourceQuantity::parse_cpu("500m").unwrap();
        assert!((q.value - 0.5).abs() < 0.001);
        assert_eq!(q.format_cpu(), "500m");

        let q2 = ResourceQuantity::parse_cpu("2").unwrap();
        assert!((q2.value - 2.0).abs() < 0.001);

        let q3 = ResourceQuantity::parse_cpu("100m").unwrap();
        assert!((q3.value - 0.1).abs() < 0.001);
    }

    #[test]
    fn test_resource_quantity_memory() {
        let q = ResourceQuantity::parse_memory("128Mi").unwrap();
        assert!((q.value - 128.0 * 1024.0 * 1024.0).abs() < 1.0);

        let q2 = ResourceQuantity::parse_memory("1Gi").unwrap();
        assert!((q2.value - 1024.0 * 1024.0 * 1024.0).abs() < 1.0);

        let q3 = ResourceQuantity::parse_memory("500Ki").unwrap();
        assert!((q3.value - 500.0 * 1024.0).abs() < 1.0);

        let q4 = ResourceQuantity::parse_memory("1000").unwrap();
        assert!((q4.value - 1000.0).abs() < 0.001);

        assert_eq!(q2.format_memory(), "1.0Gi");
        assert_eq!(q.format_memory(), "128Mi");
    }

    #[test]
    fn test_manifest_validator_basic() {
        let manifest = KubernetesManifest {
            api_version: String::new(),
            kind: String::new(),
            metadata: ManifestMetadata::default(),
            spec: None,
            raw: None,
        };
        let validator = ManifestValidator::new();
        let errors = validator.validate(&manifest);
        assert!(errors.iter().any(|e| e.path == "apiVersion"));
        assert!(errors.iter().any(|e| e.path == "kind"));
        assert!(errors.iter().any(|e| e.path == "metadata.name"));
    }

    #[test]
    fn test_manifest_validator_deployment() {
        let manifests = KubernetesManifest::parse(DEPLOYMENT_YAML).unwrap();
        let validator = ManifestValidator::new();
        let errors = validator.validate(&manifests[0]);
        // Valid deployment should have no errors
        assert!(errors.iter().all(|e| e.severity != ValidationSeverity::Error));
    }

    #[test]
    fn test_manifest_validator_strict() {
        let manifests = KubernetesManifest::parse(DEPLOYMENT_YAML).unwrap();
        let validator = ManifestValidator::strict();
        let errors = validator.validate(&manifests[0]);
        // The deployment has resource limits, so strict mode should pass for those
        let resource_errors: Vec<_> = errors
            .iter()
            .filter(|e| e.path.contains("resources.limits"))
            .collect();
        assert!(resource_errors.is_empty());
    }

    #[test]
    fn test_manifest_to_yaml_roundtrip() {
        let manifests = KubernetesManifest::parse(DEPLOYMENT_YAML).unwrap();
        let yaml_str = manifests[0].to_yaml().unwrap();
        let reparsed = KubernetesManifest::parse(&yaml_str).unwrap();
        assert_eq!(reparsed[0].kind, "Deployment");
        assert_eq!(reparsed[0].metadata.name, "nginx-deployment");
    }

    #[test]
    fn test_is_workload() {
        let manifests = KubernetesManifest::parse_multi_doc(MULTI_DOC_YAML).unwrap();
        assert!(!manifests[0].is_workload()); // Service
        assert!(manifests[1].is_workload());  // Deployment
    }

    #[test]
    fn test_env_parsing() {
        let manifests = KubernetesManifest::parse(DEPLOYMENT_YAML).unwrap();
        let spec = manifests[0].as_deployment_spec().unwrap();
        let env = &spec.template.spec.containers[0].env;
        assert_eq!(env.len(), 2);
        assert_eq!(env[0].name, "ENV");
        assert_eq!(env[0].value.as_deref(), Some("production"));
        assert_eq!(env[1].name, "DB_HOST");
        assert_eq!(env[1].value.as_deref(), Some("postgres.default.svc.cluster.local"));
    }

    #[test]
    fn test_probes_parsing() {
        let manifests = KubernetesManifest::parse(DEPLOYMENT_YAML).unwrap();
        let spec = manifests[0].as_deployment_spec().unwrap();
        let container = &spec.template.spec.containers[0];
        let liveness = container.liveness_probe.as_ref().unwrap();
        assert_eq!(liveness.initial_delay_seconds, 10);
        assert_eq!(liveness.period_seconds, 5);
        let http = liveness.http_get.as_ref().unwrap();
        assert_eq!(http.path, "/healthz");
    }

    #[test]
    fn test_empty_yaml() {
        let manifests = KubernetesManifest::parse("").unwrap();
        assert!(manifests.is_empty());
    }

    #[test]
    fn test_invalid_yaml() {
        let result = KubernetesManifest::parse("{{invalid yaml}}");
        assert!(result.is_err());
    }

    #[test]
    fn test_split_yaml_docs() {
        let docs = split_yaml_docs("---\na: 1\n---\nb: 2\n");
        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn test_validator_selector_mismatch() {
        let yaml = r#"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test
spec:
  replicas: 1
  selector:
    matchLabels:
      app: test
  template:
    metadata:
      labels:
        app: wrong
    spec:
      containers:
      - name: test
        image: test:latest
"#;
        let manifests = KubernetesManifest::parse(yaml).unwrap();
        let validator = ManifestValidator::new();
        let errors = validator.validate(&manifests[0]);
        assert!(errors.iter().any(|e| e.message.contains("mismatch")));
    }
}
