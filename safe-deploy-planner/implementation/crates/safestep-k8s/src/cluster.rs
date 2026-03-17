//! Kubernetes cluster model: nodes, capacity, and state snapshots.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that may arise when working with cluster models.
#[derive(Debug, Error, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClusterError {
    #[error("insufficient capacity: {0}")]
    InsufficientCapacity(String),
    #[error("node not ready: {0}")]
    NodeNotReady(String),
    #[error("scheduling failed: {0}")]
    SchedulingFailed(String),
}

// ---------------------------------------------------------------------------
// TaintEffect / Taint
// ---------------------------------------------------------------------------

/// Kubernetes taint effects.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaintEffect {
    NoSchedule,
    PreferNoSchedule,
    NoExecute,
}

/// A taint applied to a Kubernetes node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Taint {
    pub key: String,
    pub value: Option<String>,
    pub effect: TaintEffect,
}

// ---------------------------------------------------------------------------
// TolerationOperator / Toleration
// ---------------------------------------------------------------------------

/// Operator used in a toleration rule.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TolerationOperator {
    Equal,
    Exists,
}

/// A toleration that allows a pod to be scheduled on a tainted node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Toleration {
    pub key: Option<String>,
    pub operator: TolerationOperator,
    pub value: Option<String>,
    pub effect: Option<TaintEffect>,
}

impl Toleration {
    /// Returns `true` if this toleration matches the given taint.
    ///
    /// Matching rules:
    /// - If `key` is `None`, the toleration matches **any** taint key.
    /// - If `operator` is `Exists`, the value is ignored; only the key and
    ///   effect must match.
    /// - If `operator` is `Equal`, the key **and** value must match.
    /// - If `effect` is `None`, the toleration matches **any** effect.
    pub fn tolerates(&self, taint: &Taint) -> bool {
        // Effect check: None matches everything.
        if let Some(ref eff) = self.effect {
            if *eff != taint.effect {
                return false;
            }
        }

        // Key check: None matches everything.
        match &self.key {
            None => true,
            Some(k) => {
                if k != &taint.key {
                    return false;
                }
                match self.operator {
                    TolerationOperator::Exists => true,
                    TolerationOperator::Equal => self.value == taint.value,
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NodeCondition
// ---------------------------------------------------------------------------

/// A reported condition on a Kubernetes node (e.g. Ready, DiskPressure).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeCondition {
    pub type_name: String,
    pub status: String,
    pub reason: String,
}

// ---------------------------------------------------------------------------
// ResourceCapacity
// ---------------------------------------------------------------------------

/// Describes the resource capacity (or usage) of a node or request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceCapacity {
    pub cpu_millicores: u64,
    pub memory_bytes: u64,
    pub storage_bytes: u64,
    pub pods: u64,
    pub custom: HashMap<String, u64>,
}

impl ResourceCapacity {
    /// A capacity with all fields set to zero.
    pub fn zero() -> Self {
        Self {
            cpu_millicores: 0,
            memory_bytes: 0,
            storage_bytes: 0,
            pods: 0,
            custom: HashMap::new(),
        }
    }

    /// Construct a capacity with explicit values and an empty custom map.
    pub fn new(cpu_millicores: u64, memory_bytes: u64, storage_bytes: u64, pods: u64) -> Self {
        Self {
            cpu_millicores,
            memory_bytes,
            storage_bytes,
            pods,
            custom: HashMap::new(),
        }
    }

    /// Add `other` into `self` (field-wise).
    pub fn add(&mut self, other: &Self) {
        self.cpu_millicores += other.cpu_millicores;
        self.memory_bytes += other.memory_bytes;
        self.storage_bytes += other.storage_bytes;
        self.pods += other.pods;
        for (k, v) in &other.custom {
            *self.custom.entry(k.clone()).or_insert(0) += v;
        }
    }

    /// Subtract `other` from `self` using saturating arithmetic.
    pub fn subtract(&mut self, other: &Self) {
        self.cpu_millicores = self.cpu_millicores.saturating_sub(other.cpu_millicores);
        self.memory_bytes = self.memory_bytes.saturating_sub(other.memory_bytes);
        self.storage_bytes = self.storage_bytes.saturating_sub(other.storage_bytes);
        self.pods = self.pods.saturating_sub(other.pods);
        for (k, v) in &other.custom {
            if let Some(entry) = self.custom.get_mut(k) {
                *entry = entry.saturating_sub(*v);
            }
        }
    }

    /// Returns `true` if every field of `self` is >= the corresponding field
    /// of `requirement`.
    pub fn fits(&self, requirement: &Self) -> bool {
        if self.cpu_millicores < requirement.cpu_millicores {
            return false;
        }
        if self.memory_bytes < requirement.memory_bytes {
            return false;
        }
        if self.storage_bytes < requirement.storage_bytes {
            return false;
        }
        if self.pods < requirement.pods {
            return false;
        }
        for (k, v) in &requirement.custom {
            let have = self.custom.get(k).copied().unwrap_or(0);
            if have < *v {
                return false;
            }
        }
        true
    }

    /// CPU utilization ratio: `used.cpu_millicores / self.cpu_millicores`.
    /// Returns 0.0 when capacity is zero.
    pub fn utilization_ratio(&self, used: &Self) -> f64 {
        if self.cpu_millicores == 0 {
            return 0.0;
        }
        used.cpu_millicores as f64 / self.cpu_millicores as f64
    }

    /// Parse Kubernetes-style resource strings.
    ///
    /// CPU: `"100m"` → 100 millicores, `"4"` → 4000 millicores.
    /// Memory: `"1Gi"` → 1 GiB, `"512Mi"` → 512 MiB, `"1024Ki"` → 1024 KiB,
    ///         plain number is bytes.
    pub fn from_kubernetes_units(cpu_str: &str, mem_str: &str) -> Self {
        let cpu = parse_cpu(cpu_str);
        let mem = parse_memory(mem_str);
        Self {
            cpu_millicores: cpu,
            memory_bytes: mem,
            storage_bytes: 0,
            pods: 0,
            custom: HashMap::new(),
        }
    }
}

impl fmt::Display for ResourceCapacity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mem_gib = self.memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        write!(
            f,
            "cpu: {}m, mem: {:.1}GiB, pods: {}",
            self.cpu_millicores, mem_gib, self.pods
        )
    }
}

// ---- helpers for Kubernetes unit parsing ----------------------------------

fn parse_cpu(s: &str) -> u64 {
    let s = s.trim();
    if let Some(stripped) = s.strip_suffix('m') {
        stripped.parse::<u64>().unwrap_or(0)
    } else if let Ok(cores) = s.parse::<f64>() {
        (cores * 1000.0) as u64
    } else {
        0
    }
}

fn parse_memory(s: &str) -> u64 {
    let s = s.trim();
    if let Some(stripped) = s.strip_suffix("Gi") {
        stripped
            .parse::<u64>()
            .map(|v| v * 1024 * 1024 * 1024)
            .unwrap_or(0)
    } else if let Some(stripped) = s.strip_suffix("Mi") {
        stripped
            .parse::<u64>()
            .map(|v| v * 1024 * 1024)
            .unwrap_or(0)
    } else if let Some(stripped) = s.strip_suffix("Ki") {
        stripped.parse::<u64>().map(|v| v * 1024).unwrap_or(0)
    } else if let Some(stripped) = s.strip_suffix('G') {
        stripped
            .parse::<u64>()
            .map(|v| v * 1_000_000_000)
            .unwrap_or(0)
    } else if let Some(stripped) = s.strip_suffix('M') {
        stripped
            .parse::<u64>()
            .map(|v| v * 1_000_000)
            .unwrap_or(0)
    } else if let Some(stripped) = s.strip_suffix('K') {
        stripped.parse::<u64>().map(|v| v * 1_000).unwrap_or(0)
    } else {
        s.parse::<u64>().unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// ResourceRequirements
// ---------------------------------------------------------------------------

/// Wraps a request / limit pair of resources (like a Kubernetes container
/// resource spec).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub requests: ResourceCapacity,
    pub limits: ResourceCapacity,
}

impl ResourceRequirements {
    pub fn new(requests: ResourceCapacity, limits: ResourceCapacity) -> Self {
        Self { requests, limits }
    }

    /// Limits must be >= requests for every field.
    pub fn is_valid(&self) -> bool {
        self.limits.cpu_millicores >= self.requests.cpu_millicores
            && self.limits.memory_bytes >= self.requests.memory_bytes
            && self.limits.storage_bytes >= self.requests.storage_bytes
            && self.limits.pods >= self.requests.pods
    }
}

// ---------------------------------------------------------------------------
// NodeInfo
// ---------------------------------------------------------------------------

/// Information about a single Kubernetes node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeInfo {
    pub name: String,
    pub labels: HashMap<String, String>,
    pub taints: Vec<Taint>,
    pub capacity: ResourceCapacity,
    pub allocatable: ResourceCapacity,
    pub conditions: Vec<NodeCondition>,
}

impl NodeInfo {
    /// A node is ready if it has a condition of type `"Ready"` with status
    /// `"True"`.
    pub fn is_ready(&self) -> bool {
        self.conditions
            .iter()
            .any(|c| c.type_name == "Ready" && c.status == "True")
    }

    /// A node is schedulable when it is ready **and** carries no `NoSchedule`
    /// taints (taints with `PreferNoSchedule` or `NoExecute` are allowed for
    /// scheduling purposes).
    pub fn is_schedulable(&self) -> bool {
        if !self.is_ready() {
            return false;
        }
        !self
            .taints
            .iter()
            .any(|t| t.effect == TaintEffect::NoSchedule)
    }

    /// Returns `true` when every key-value pair in `selector` is present in
    /// the node's labels.
    pub fn matches_selector(&self, selector: &HashMap<String, String>) -> bool {
        selector
            .iter()
            .all(|(k, v)| self.labels.get(k).map_or(false, |lv| lv == v))
    }
}

// ---------------------------------------------------------------------------
// RunningService
// ---------------------------------------------------------------------------

/// A service observed running in the cluster.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunningService {
    pub name: String,
    pub namespace: String,
    pub image: String,
    pub replicas: u32,
    pub version_tag: Option<String>,
}

// ---------------------------------------------------------------------------
// ClusterSnapshot
// ---------------------------------------------------------------------------

/// A point-in-time snapshot of the services running in a cluster.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClusterSnapshot {
    pub services: Vec<RunningService>,
    pub timestamp: String,
}

impl ClusterSnapshot {
    /// Build a snapshot from a slice of JSON manifests.
    ///
    /// Each manifest is expected to be a Kubernetes-style JSON object with
    /// optional fields:
    /// - `metadata.name`, `metadata.namespace`
    /// - `spec.replicas`
    /// - `spec.template.spec.containers[0].image`
    ///
    /// Missing fields are filled with defaults.
    pub fn from_manifests(manifests: &[serde_json::Value]) -> Self {
        let mut services = Vec::new();
        for m in manifests {
            let name = m
                .pointer("/metadata/name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();

            let namespace = m
                .pointer("/metadata/namespace")
                .and_then(|v| v.as_str())
                .unwrap_or("default")
                .to_string();

            let replicas = m
                .pointer("/spec/replicas")
                .and_then(|v| v.as_u64())
                .unwrap_or(1) as u32;

            let image = m
                .pointer("/spec/template/spec/containers/0/image")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown:latest")
                .to_string();

            let version_tag = extract_image_tag(&image);

            services.push(RunningService {
                name,
                namespace,
                image,
                replicas,
                version_tag,
            });
        }

        Self {
            services,
            timestamp: chrono_like_now(),
        }
    }

    pub fn service_count(&self) -> usize {
        self.services.len()
    }

    pub fn find_service(&self, name: &str) -> Option<&RunningService> {
        self.services.iter().find(|s| s.name == name)
    }
}

/// Extract the tag portion of a Docker image reference (after the last `:`).
fn extract_image_tag(image: &str) -> Option<String> {
    // Ignore digest references (contain '@').
    if image.contains('@') {
        return None;
    }
    // The tag is the part after the last ':' — but only if the colon is after
    // any '/' (to avoid confusing a registry port with a tag).
    if let Some(slash_pos) = image.rfind('/') {
        let after_slash = &image[slash_pos..];
        if let Some(colon_pos) = after_slash.rfind(':') {
            let tag = &after_slash[colon_pos + 1..];
            if !tag.is_empty() {
                return Some(tag.to_string());
            }
        }
    } else if let Some(colon_pos) = image.rfind(':') {
        let tag = &image[colon_pos + 1..];
        if !tag.is_empty() {
            return Some(tag.to_string());
        }
    }
    None
}

/// Simple timestamp (no external chrono dependency).
fn chrono_like_now() -> String {
    // Provide a deterministic placeholder; real implementations would use
    // `chrono::Utc::now()` or `std::time::SystemTime`.
    String::from("1970-01-01T00:00:00Z")
}

// ---------------------------------------------------------------------------
// ClusterModel
// ---------------------------------------------------------------------------

/// High-level model of a Kubernetes cluster.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClusterModel {
    pub nodes: Vec<NodeInfo>,
    pub namespaces: Vec<String>,
    pub total_capacity: ResourceCapacity,
}

impl ClusterModel {
    /// Create a new cluster model from a list of nodes.
    ///
    /// `total_capacity` is computed as the sum of all node `allocatable`
    /// resources.  `namespaces` starts empty.
    pub fn new(nodes: Vec<NodeInfo>) -> Self {
        let mut total = ResourceCapacity::zero();
        for n in &nodes {
            total.add(&n.allocatable);
        }
        Self {
            nodes,
            namespaces: Vec::new(),
            total_capacity: total,
        }
    }

    /// Sum of `allocatable` across **ready** nodes only.
    pub fn available_capacity(&self) -> ResourceCapacity {
        let mut cap = ResourceCapacity::zero();
        for n in &self.nodes {
            if n.is_ready() {
                cap.add(&n.allocatable);
            }
        }
        cap
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of nodes that are both ready and schedulable.
    pub fn ready_node_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_schedulable()).count()
    }

    /// Returns `true` if at least one **schedulable** node can fit the
    /// resource requests.
    pub fn can_schedule(&self, requirements: &ResourceRequirements) -> bool {
        self.nodes.iter().any(|n| {
            n.is_schedulable() && n.allocatable.fits(&requirements.requests)
        })
    }

    /// Return references to all schedulable nodes whose allocatable resources
    /// fit the given requests.
    pub fn find_suitable_nodes(&self, req: &ResourceRequirements) -> Vec<&NodeInfo> {
        self.nodes
            .iter()
            .filter(|n| n.is_schedulable() && n.allocatable.fits(&req.requests))
            .collect()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers ------------------------------------------------------------

    fn ready_condition() -> NodeCondition {
        NodeCondition {
            type_name: "Ready".into(),
            status: "True".into(),
            reason: "KubeletReady".into(),
        }
    }

    fn not_ready_condition() -> NodeCondition {
        NodeCondition {
            type_name: "Ready".into(),
            status: "False".into(),
            reason: "KubeletNotReady".into(),
        }
    }

    fn basic_node(name: &str, cpu: u64, mem: u64, ready: bool) -> NodeInfo {
        let cond = if ready {
            ready_condition()
        } else {
            not_ready_condition()
        };
        NodeInfo {
            name: name.into(),
            labels: HashMap::new(),
            taints: Vec::new(),
            capacity: ResourceCapacity::new(cpu, mem, 100_000_000_000, 110),
            allocatable: ResourceCapacity::new(cpu, mem, 100_000_000_000, 110),
            conditions: vec![cond],
        }
    }

    fn labeled_node(name: &str, labels: HashMap<String, String>) -> NodeInfo {
        NodeInfo {
            name: name.into(),
            labels,
            taints: Vec::new(),
            capacity: ResourceCapacity::new(4000, 8_000_000_000, 0, 110),
            allocatable: ResourceCapacity::new(4000, 8_000_000_000, 0, 110),
            conditions: vec![ready_condition()],
        }
    }

    fn tainted_node(name: &str, taints: Vec<Taint>) -> NodeInfo {
        NodeInfo {
            name: name.into(),
            labels: HashMap::new(),
            taints,
            capacity: ResourceCapacity::new(4000, 8_000_000_000, 0, 110),
            allocatable: ResourceCapacity::new(4000, 8_000_000_000, 0, 110),
            conditions: vec![ready_condition()],
        }
    }

    // -- ClusterError -------------------------------------------------------

    #[test]
    fn cluster_error_display() {
        let e = ClusterError::InsufficientCapacity("cpu".into());
        assert_eq!(e.to_string(), "insufficient capacity: cpu");

        let e2 = ClusterError::NodeNotReady("node-1".into());
        assert_eq!(e2.to_string(), "node not ready: node-1");

        let e3 = ClusterError::SchedulingFailed("no fit".into());
        assert_eq!(e3.to_string(), "scheduling failed: no fit");
    }

    #[test]
    fn cluster_error_serde_roundtrip() {
        let e = ClusterError::InsufficientCapacity("mem".into());
        let json = serde_json::to_string(&e).unwrap();
        let back: ClusterError = serde_json::from_str(&json).unwrap();
        assert_eq!(e, back);
    }

    // -- ResourceCapacity ---------------------------------------------------

    #[test]
    fn resource_capacity_zero() {
        let z = ResourceCapacity::zero();
        assert_eq!(z.cpu_millicores, 0);
        assert_eq!(z.memory_bytes, 0);
        assert_eq!(z.storage_bytes, 0);
        assert_eq!(z.pods, 0);
        assert!(z.custom.is_empty());
    }

    #[test]
    fn resource_capacity_add() {
        let mut a = ResourceCapacity::new(1000, 2000, 3000, 10);
        let b = ResourceCapacity::new(500, 600, 700, 5);
        a.add(&b);
        assert_eq!(a.cpu_millicores, 1500);
        assert_eq!(a.memory_bytes, 2600);
        assert_eq!(a.storage_bytes, 3700);
        assert_eq!(a.pods, 15);
    }

    #[test]
    fn resource_capacity_add_custom() {
        let mut a = ResourceCapacity::zero();
        a.custom.insert("gpu".into(), 2);
        let mut b = ResourceCapacity::zero();
        b.custom.insert("gpu".into(), 3);
        b.custom.insert("fpga".into(), 1);
        a.add(&b);
        assert_eq!(a.custom["gpu"], 5);
        assert_eq!(a.custom["fpga"], 1);
    }

    #[test]
    fn resource_capacity_subtract_saturating() {
        let mut a = ResourceCapacity::new(100, 200, 300, 5);
        let b = ResourceCapacity::new(150, 200, 100, 10);
        a.subtract(&b);
        assert_eq!(a.cpu_millicores, 0); // saturated
        assert_eq!(a.memory_bytes, 0);
        assert_eq!(a.storage_bytes, 200);
        assert_eq!(a.pods, 0); // saturated
    }

    #[test]
    fn resource_capacity_subtract_custom_saturating() {
        let mut a = ResourceCapacity::zero();
        a.custom.insert("gpu".into(), 2);
        let mut b = ResourceCapacity::zero();
        b.custom.insert("gpu".into(), 5);
        a.subtract(&b);
        assert_eq!(a.custom["gpu"], 0);
    }

    #[test]
    fn resource_capacity_fits() {
        let cap = ResourceCapacity::new(4000, 8_000_000_000, 100_000, 110);
        let req = ResourceCapacity::new(2000, 4_000_000_000, 50_000, 50);
        assert!(cap.fits(&req));
    }

    #[test]
    fn resource_capacity_does_not_fit() {
        let cap = ResourceCapacity::new(1000, 1_000_000_000, 100, 10);
        let req = ResourceCapacity::new(2000, 500_000_000, 50, 5);
        assert!(!cap.fits(&req));
    }

    #[test]
    fn resource_capacity_fits_custom() {
        let mut cap = ResourceCapacity::new(4000, 8_000_000_000, 0, 110);
        cap.custom.insert("gpu".into(), 4);
        let mut req = ResourceCapacity::new(1000, 1_000_000_000, 0, 1);
        req.custom.insert("gpu".into(), 2);
        assert!(cap.fits(&req));
        req.custom.insert("gpu".into(), 8);
        assert!(!cap.fits(&req));
    }

    #[test]
    fn resource_capacity_utilization_ratio() {
        let cap = ResourceCapacity::new(4000, 0, 0, 0);
        let used = ResourceCapacity::new(2000, 0, 0, 0);
        let ratio = cap.utilization_ratio(&used);
        assert!((ratio - 0.5).abs() < 1e-9);
    }

    #[test]
    fn resource_capacity_utilization_zero_capacity() {
        let cap = ResourceCapacity::zero();
        let used = ResourceCapacity::new(100, 0, 0, 0);
        assert_eq!(cap.utilization_ratio(&used), 0.0);
    }

    #[test]
    fn resource_capacity_display() {
        let cap = ResourceCapacity::new(
            4000,
            8 * 1024 * 1024 * 1024, // 8 GiB
            0,
            110,
        );
        let s = format!("{}", cap);
        assert!(s.contains("cpu: 4000m"));
        assert!(s.contains("8.0GiB"));
        assert!(s.contains("pods: 110"));
    }

    // -- Kubernetes unit parsing -------------------------------------------

    #[test]
    fn parse_cpu_millicores() {
        let cap = ResourceCapacity::from_kubernetes_units("250m", "0");
        assert_eq!(cap.cpu_millicores, 250);
    }

    #[test]
    fn parse_cpu_whole_cores() {
        let cap = ResourceCapacity::from_kubernetes_units("4", "0");
        assert_eq!(cap.cpu_millicores, 4000);
    }

    #[test]
    fn parse_cpu_fractional_cores() {
        let cap = ResourceCapacity::from_kubernetes_units("1.5", "0");
        assert_eq!(cap.cpu_millicores, 1500);
    }

    #[test]
    fn parse_memory_gi() {
        let cap = ResourceCapacity::from_kubernetes_units("0", "2Gi");
        assert_eq!(cap.memory_bytes, 2 * 1024 * 1024 * 1024);
    }

    #[test]
    fn parse_memory_mi() {
        let cap = ResourceCapacity::from_kubernetes_units("0", "512Mi");
        assert_eq!(cap.memory_bytes, 512 * 1024 * 1024);
    }

    #[test]
    fn parse_memory_ki() {
        let cap = ResourceCapacity::from_kubernetes_units("0", "1024Ki");
        assert_eq!(cap.memory_bytes, 1024 * 1024);
    }

    #[test]
    fn parse_memory_plain_bytes() {
        let cap = ResourceCapacity::from_kubernetes_units("0", "65536");
        assert_eq!(cap.memory_bytes, 65536);
    }

    #[test]
    fn parse_memory_decimal_g() {
        let cap = ResourceCapacity::from_kubernetes_units("0", "1G");
        assert_eq!(cap.memory_bytes, 1_000_000_000);
    }

    // -- Toleration --------------------------------------------------------

    #[test]
    fn toleration_exact_match() {
        let taint = Taint {
            key: "dedicated".into(),
            value: Some("gpu".into()),
            effect: TaintEffect::NoSchedule,
        };
        let tol = Toleration {
            key: Some("dedicated".into()),
            operator: TolerationOperator::Equal,
            value: Some("gpu".into()),
            effect: Some(TaintEffect::NoSchedule),
        };
        assert!(tol.tolerates(&taint));
    }

    #[test]
    fn toleration_value_mismatch() {
        let taint = Taint {
            key: "dedicated".into(),
            value: Some("gpu".into()),
            effect: TaintEffect::NoSchedule,
        };
        let tol = Toleration {
            key: Some("dedicated".into()),
            operator: TolerationOperator::Equal,
            value: Some("cpu".into()),
            effect: Some(TaintEffect::NoSchedule),
        };
        assert!(!tol.tolerates(&taint));
    }

    #[test]
    fn toleration_exists_operator_ignores_value() {
        let taint = Taint {
            key: "dedicated".into(),
            value: Some("gpu".into()),
            effect: TaintEffect::NoSchedule,
        };
        let tol = Toleration {
            key: Some("dedicated".into()),
            operator: TolerationOperator::Exists,
            value: None,
            effect: Some(TaintEffect::NoSchedule),
        };
        assert!(tol.tolerates(&taint));
    }

    #[test]
    fn toleration_wildcard_key() {
        let taint = Taint {
            key: "anything".into(),
            value: None,
            effect: TaintEffect::NoExecute,
        };
        let tol = Toleration {
            key: None,
            operator: TolerationOperator::Exists,
            value: None,
            effect: Some(TaintEffect::NoExecute),
        };
        assert!(tol.tolerates(&taint));
    }

    #[test]
    fn toleration_wildcard_effect() {
        let taint = Taint {
            key: "zone".into(),
            value: Some("us-east".into()),
            effect: TaintEffect::PreferNoSchedule,
        };
        let tol = Toleration {
            key: Some("zone".into()),
            operator: TolerationOperator::Equal,
            value: Some("us-east".into()),
            effect: None, // matches any effect
        };
        assert!(tol.tolerates(&taint));
    }

    #[test]
    fn toleration_effect_mismatch() {
        let taint = Taint {
            key: "zone".into(),
            value: None,
            effect: TaintEffect::NoExecute,
        };
        let tol = Toleration {
            key: Some("zone".into()),
            operator: TolerationOperator::Exists,
            value: None,
            effect: Some(TaintEffect::NoSchedule),
        };
        assert!(!tol.tolerates(&taint));
    }

    // -- NodeInfo -----------------------------------------------------------

    #[test]
    fn node_is_ready_true() {
        let node = basic_node("n1", 1000, 1_000_000, true);
        assert!(node.is_ready());
    }

    #[test]
    fn node_is_ready_false() {
        let node = basic_node("n1", 1000, 1_000_000, false);
        assert!(!node.is_ready());
    }

    #[test]
    fn node_is_schedulable_no_taints() {
        let node = basic_node("n1", 1000, 1_000_000, true);
        assert!(node.is_schedulable());
    }

    #[test]
    fn node_not_schedulable_when_not_ready() {
        let node = basic_node("n1", 1000, 1_000_000, false);
        assert!(!node.is_schedulable());
    }

    #[test]
    fn node_not_schedulable_with_noschedule_taint() {
        let node = tainted_node(
            "n1",
            vec![Taint {
                key: "dedicated".into(),
                value: Some("special".into()),
                effect: TaintEffect::NoSchedule,
            }],
        );
        assert!(!node.is_schedulable());
    }

    #[test]
    fn node_schedulable_with_prefer_no_schedule_taint() {
        let node = tainted_node(
            "n1",
            vec![Taint {
                key: "zone".into(),
                value: None,
                effect: TaintEffect::PreferNoSchedule,
            }],
        );
        assert!(node.is_schedulable());
    }

    #[test]
    fn node_matches_selector_empty() {
        let node = basic_node("n1", 1000, 1_000_000, true);
        assert!(node.matches_selector(&HashMap::new()));
    }

    #[test]
    fn node_matches_selector_present() {
        let mut labels = HashMap::new();
        labels.insert("zone".into(), "us-east-1".into());
        labels.insert("tier".into(), "frontend".into());
        let node = labeled_node("n1", labels);

        let mut sel = HashMap::new();
        sel.insert("zone".into(), "us-east-1".into());
        assert!(node.matches_selector(&sel));
    }

    #[test]
    fn node_does_not_match_selector_missing_key() {
        let node = basic_node("n1", 1000, 1_000_000, true);
        let mut sel = HashMap::new();
        sel.insert("zone".into(), "us-east-1".into());
        assert!(!node.matches_selector(&sel));
    }

    #[test]
    fn node_does_not_match_selector_wrong_value() {
        let mut labels = HashMap::new();
        labels.insert("zone".into(), "us-west-2".into());
        let node = labeled_node("n1", labels);
        let mut sel = HashMap::new();
        sel.insert("zone".into(), "us-east-1".into());
        assert!(!node.matches_selector(&sel));
    }

    // -- ClusterModel -------------------------------------------------------

    #[test]
    fn cluster_model_total_capacity() {
        let nodes = vec![
            basic_node("n1", 4000, 8_000_000_000, true),
            basic_node("n2", 4000, 8_000_000_000, true),
        ];
        let model = ClusterModel::new(nodes);
        assert_eq!(model.total_capacity.cpu_millicores, 8000);
        assert_eq!(model.total_capacity.memory_bytes, 16_000_000_000);
    }

    #[test]
    fn cluster_model_node_count() {
        let model = ClusterModel::new(vec![
            basic_node("a", 1000, 1000, true),
            basic_node("b", 1000, 1000, false),
            basic_node("c", 1000, 1000, true),
        ]);
        assert_eq!(model.node_count(), 3);
    }

    #[test]
    fn cluster_model_ready_node_count() {
        let model = ClusterModel::new(vec![
            basic_node("a", 1000, 1000, true),
            basic_node("b", 1000, 1000, false),
            basic_node("c", 1000, 1000, true),
        ]);
        assert_eq!(model.ready_node_count(), 2);
    }

    #[test]
    fn cluster_model_available_capacity_excludes_not_ready() {
        let nodes = vec![
            basic_node("n1", 4000, 8_000_000_000, true),
            basic_node("n2", 2000, 4_000_000_000, false),
        ];
        let model = ClusterModel::new(nodes);
        let avail = model.available_capacity();
        assert_eq!(avail.cpu_millicores, 4000);
        assert_eq!(avail.memory_bytes, 8_000_000_000);
    }

    #[test]
    fn cluster_model_can_schedule() {
        let model = ClusterModel::new(vec![basic_node("n1", 4000, 8_000_000_000, true)]);
        let req = ResourceRequirements::new(
            ResourceCapacity::new(2000, 4_000_000_000, 0, 1),
            ResourceCapacity::new(4000, 8_000_000_000, 0, 1),
        );
        assert!(model.can_schedule(&req));
    }

    #[test]
    fn cluster_model_cannot_schedule_too_large() {
        let model = ClusterModel::new(vec![basic_node("n1", 1000, 1_000_000_000, true)]);
        let req = ResourceRequirements::new(
            ResourceCapacity::new(2000, 500_000_000, 0, 1),
            ResourceCapacity::new(2000, 500_000_000, 0, 1),
        );
        assert!(!model.can_schedule(&req));
    }

    #[test]
    fn cluster_model_find_suitable_nodes() {
        let model = ClusterModel::new(vec![
            basic_node("big", 8000, 16_000_000_000, true),
            basic_node("small", 1000, 1_000_000_000, true),
            basic_node("off", 8000, 16_000_000_000, false),
        ]);
        let req = ResourceRequirements::new(
            ResourceCapacity::new(4000, 8_000_000_000, 0, 1),
            ResourceCapacity::new(8000, 16_000_000_000, 0, 1),
        );
        let suitable = model.find_suitable_nodes(&req);
        assert_eq!(suitable.len(), 1);
        assert_eq!(suitable[0].name, "big");
    }

    // -- ResourceRequirements -----------------------------------------------

    #[test]
    fn resource_requirements_valid() {
        let rr = ResourceRequirements::new(
            ResourceCapacity::new(100, 200, 300, 1),
            ResourceCapacity::new(200, 400, 600, 2),
        );
        assert!(rr.is_valid());
    }

    #[test]
    fn resource_requirements_invalid_cpu() {
        let rr = ResourceRequirements::new(
            ResourceCapacity::new(500, 200, 300, 1),
            ResourceCapacity::new(200, 400, 600, 2),
        );
        assert!(!rr.is_valid());
    }

    #[test]
    fn resource_requirements_equal_is_valid() {
        let cap = ResourceCapacity::new(100, 200, 300, 1);
        let rr = ResourceRequirements::new(cap.clone(), cap);
        assert!(rr.is_valid());
    }

    // -- ClusterSnapshot ----------------------------------------------------

    #[test]
    fn snapshot_from_manifests_basic() {
        let manifest = serde_json::json!({
            "metadata": { "name": "web", "namespace": "prod" },
            "spec": {
                "replicas": 3,
                "template": {
                    "spec": {
                        "containers": [
                            { "image": "myrepo/web:v1.2.3" }
                        ]
                    }
                }
            }
        });
        let snap = ClusterSnapshot::from_manifests(&[manifest]);
        assert_eq!(snap.service_count(), 1);
        let svc = snap.find_service("web").unwrap();
        assert_eq!(svc.namespace, "prod");
        assert_eq!(svc.replicas, 3);
        assert_eq!(svc.image, "myrepo/web:v1.2.3");
        assert_eq!(svc.version_tag.as_deref(), Some("v1.2.3"));
    }

    #[test]
    fn snapshot_from_manifests_defaults() {
        let manifest = serde_json::json!({});
        let snap = ClusterSnapshot::from_manifests(&[manifest]);
        let svc = &snap.services[0];
        assert_eq!(svc.name, "unknown");
        assert_eq!(svc.namespace, "default");
        assert_eq!(svc.replicas, 1);
    }

    #[test]
    fn snapshot_find_service_missing() {
        let snap = ClusterSnapshot::from_manifests(&[]);
        assert!(snap.find_service("nope").is_none());
    }

    #[test]
    fn snapshot_multiple_services() {
        let m1 = serde_json::json!({
            "metadata": { "name": "api" },
            "spec": { "replicas": 2, "template": { "spec": { "containers": [{ "image": "api:latest" }] } } }
        });
        let m2 = serde_json::json!({
            "metadata": { "name": "worker" },
            "spec": { "replicas": 5, "template": { "spec": { "containers": [{ "image": "worker:v2" }] } } }
        });
        let snap = ClusterSnapshot::from_manifests(&[m1, m2]);
        assert_eq!(snap.service_count(), 2);
        assert_eq!(snap.find_service("api").unwrap().replicas, 2);
        assert_eq!(
            snap.find_service("worker").unwrap().version_tag.as_deref(),
            Some("v2")
        );
    }

    // -- Serde round-trips --------------------------------------------------

    #[test]
    fn resource_capacity_serde_roundtrip() {
        let mut cap = ResourceCapacity::new(4000, 8_000_000_000, 500_000, 110);
        cap.custom.insert("gpu".into(), 4);
        let json = serde_json::to_string(&cap).unwrap();
        let back: ResourceCapacity = serde_json::from_str(&json).unwrap();
        assert_eq!(cap, back);
    }

    #[test]
    fn node_info_serde_roundtrip() {
        let node = basic_node("n1", 4000, 8_000_000_000, true);
        let json = serde_json::to_string(&node).unwrap();
        let back: NodeInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(node, back);
    }

    #[test]
    fn cluster_model_serde_roundtrip() {
        let model = ClusterModel::new(vec![
            basic_node("a", 1000, 1000, true),
            basic_node("b", 2000, 2000, false),
        ]);
        let json = serde_json::to_string(&model).unwrap();
        let back: ClusterModel = serde_json::from_str(&json).unwrap();
        assert_eq!(model, back);
    }

    #[test]
    fn taint_effect_serde_roundtrip() {
        for eff in &[
            TaintEffect::NoSchedule,
            TaintEffect::PreferNoSchedule,
            TaintEffect::NoExecute,
        ] {
            let json = serde_json::to_string(eff).unwrap();
            let back: TaintEffect = serde_json::from_str(&json).unwrap();
            assert_eq!(*eff, back);
        }
    }

    #[test]
    fn cluster_snapshot_serde_roundtrip() {
        let snap = ClusterSnapshot {
            services: vec![RunningService {
                name: "svc".into(),
                namespace: "ns".into(),
                image: "img:v1".into(),
                replicas: 3,
                version_tag: Some("v1".into()),
            }],
            timestamp: "2024-01-01T00:00:00Z".into(),
        };
        let json = serde_json::to_string(&snap).unwrap();
        let back: ClusterSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(snap, back);
    }

    // -- extract_image_tag edge cases --------------------------------------

    #[test]
    fn image_tag_simple() {
        assert_eq!(extract_image_tag("nginx:1.25"), Some("1.25".into()));
    }

    #[test]
    fn image_tag_with_registry() {
        assert_eq!(
            extract_image_tag("registry.io:5000/app:v2"),
            Some("v2".into())
        );
    }

    #[test]
    fn image_tag_no_tag() {
        assert_eq!(extract_image_tag("nginx"), None);
    }

    #[test]
    fn image_tag_digest_returns_none() {
        assert_eq!(
            extract_image_tag("nginx@sha256:abcdef1234567890"),
            None
        );
    }
}
