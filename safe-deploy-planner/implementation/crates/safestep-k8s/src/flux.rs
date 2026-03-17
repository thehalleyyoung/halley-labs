//! Flux CD resource generation from deployment plans.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use safestep_types::SafeStepError;

use crate::{DeploymentPlan, DeploymentStep};

pub type Result<T> = std::result::Result<T, SafeStepError>;

// ---------------------------------------------------------------------------
// Flux HelmRelease
// ---------------------------------------------------------------------------

/// Flux HelmRelease Custom Resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxHelmRelease {
    pub api_version: String,
    pub kind: String,
    pub metadata: FluxMetadata,
    pub spec: HelmReleaseSpec,
}

/// Flux resource metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxMetadata {
    pub name: String,
    pub namespace: String,
    pub labels: HashMap<String, String>,
    pub annotations: HashMap<String, String>,
}

/// HelmRelease spec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelmReleaseSpec {
    pub chart: HelmChartTemplate,
    pub interval: String,
    pub values: Option<Value>,
    pub depends_on: Vec<FluxDependency>,
    pub timeout: Option<String>,
    pub install: Option<HelmInstallSpec>,
    pub upgrade: Option<HelmUpgradeSpec>,
    pub rollback: Option<HelmRollbackSpec>,
    pub target_namespace: Option<String>,
    pub service_account_name: Option<String>,
    pub suspend: bool,
    pub max_history: Option<u32>,
}

/// Helm chart template reference in a HelmRelease.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelmChartTemplate {
    pub spec: ChartTemplateSpec,
}

/// Chart template spec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartTemplateSpec {
    pub chart: String,
    pub version: Option<String>,
    pub source_ref: SourceRef,
    pub interval: Option<String>,
    pub reconcile_strategy: Option<String>,
}

/// Reference to a Flux source (HelmRepository, GitRepository, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceRef {
    pub kind: String,
    pub name: String,
    pub namespace: Option<String>,
}

/// A dependency reference for ordering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxDependency {
    pub name: String,
    pub namespace: Option<String>,
}

/// Helm install configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelmInstallSpec {
    pub create_namespace: bool,
    pub remediation: Option<Remediation>,
}

/// Helm upgrade configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelmUpgradeSpec {
    pub remediation: Option<Remediation>,
    pub clean_up_on_fail: bool,
    pub preserve_values: bool,
    pub force: bool,
}

/// Remediation strategy for install/upgrade failures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Remediation {
    pub retries: u32,
    pub strategy: Option<String>,
}

/// Helm rollback configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelmRollbackSpec {
    pub clean_up_on_fail: bool,
    pub timeout: Option<String>,
    pub disable_hooks: bool,
    pub force: bool,
    pub recreate: bool,
}

impl FluxHelmRelease {
    pub fn new(name: &str, namespace: &str, chart: &str, source_name: &str) -> Self {
        Self {
            api_version: "helm.toolkit.fluxcd.io/v2beta2".into(),
            kind: "HelmRelease".into(),
            metadata: FluxMetadata {
                name: name.to_string(),
                namespace: namespace.to_string(),
                labels: HashMap::new(),
                annotations: HashMap::new(),
            },
            spec: HelmReleaseSpec {
                chart: HelmChartTemplate {
                    spec: ChartTemplateSpec {
                        chart: chart.to_string(),
                        version: None,
                        source_ref: SourceRef {
                            kind: "HelmRepository".into(),
                            name: source_name.to_string(),
                            namespace: None,
                        },
                        interval: None,
                        reconcile_strategy: None,
                    },
                },
                interval: "5m".into(),
                values: None,
                depends_on: Vec::new(),
                timeout: Some("10m".into()),
                install: Some(HelmInstallSpec {
                    create_namespace: true,
                    remediation: Some(Remediation {
                        retries: 3,
                        strategy: None,
                    }),
                }),
                upgrade: Some(HelmUpgradeSpec {
                    remediation: Some(Remediation {
                        retries: 3,
                        strategy: Some("rollback".into()),
                    }),
                    clean_up_on_fail: true,
                    preserve_values: false,
                    force: false,
                }),
                rollback: None,
                target_namespace: None,
                service_account_name: None,
                suspend: false,
                max_history: Some(5),
            },
        }
    }

    /// Serialize to YAML string.
    pub fn to_yaml(&self) -> Result<String> {
        let value = self.to_value();
        serde_yaml::to_string(&value).map_err(|e| SafeStepError::K8sError {
            message: format!("Failed to serialize HelmRelease: {e}"),
            resource: Some(self.metadata.name.clone()),
            namespace: Some(self.metadata.namespace.clone()),
            context: None,
        })
    }

    /// Convert to a serde_json Value.
    pub fn to_value(&self) -> Value {
        let mut metadata = serde_json::Map::new();
        metadata.insert("name".into(), Value::String(self.metadata.name.clone()));
        metadata.insert("namespace".into(), Value::String(self.metadata.namespace.clone()));
        if !self.metadata.labels.is_empty() {
            metadata.insert("labels".into(), serde_json::to_value(&self.metadata.labels).unwrap_or_default());
        }
        if !self.metadata.annotations.is_empty() {
            metadata.insert("annotations".into(), serde_json::to_value(&self.metadata.annotations).unwrap_or_default());
        }

        let mut chart_spec = serde_json::Map::new();
        chart_spec.insert("chart".into(), Value::String(self.spec.chart.spec.chart.clone()));
        if let Some(v) = &self.spec.chart.spec.version {
            chart_spec.insert("version".into(), Value::String(v.clone()));
        }
        chart_spec.insert("sourceRef".into(), serde_json::json!({
            "kind": self.spec.chart.spec.source_ref.kind,
            "name": self.spec.chart.spec.source_ref.name,
        }));

        let mut spec = serde_json::Map::new();
        spec.insert("chart".into(), serde_json::json!({"spec": Value::Object(chart_spec)}));
        spec.insert("interval".into(), Value::String(self.spec.interval.clone()));

        if let Some(values) = &self.spec.values {
            spec.insert("values".into(), values.clone());
        }

        if !self.spec.depends_on.is_empty() {
            let deps: Vec<Value> = self.spec.depends_on.iter().map(|d| {
                let mut dep = serde_json::Map::new();
                dep.insert("name".into(), Value::String(d.name.clone()));
                if let Some(ns) = &d.namespace {
                    dep.insert("namespace".into(), Value::String(ns.clone()));
                }
                Value::Object(dep)
            }).collect();
            spec.insert("dependsOn".into(), Value::Array(deps));
        }

        if let Some(timeout) = &self.spec.timeout {
            spec.insert("timeout".into(), Value::String(timeout.clone()));
        }

        if let Some(install) = &self.spec.install {
            let mut inst = serde_json::Map::new();
            inst.insert("createNamespace".into(), Value::Bool(install.create_namespace));
            if let Some(rem) = &install.remediation {
                inst.insert("remediation".into(), serde_json::json!({"retries": rem.retries}));
            }
            spec.insert("install".into(), Value::Object(inst));
        }

        if let Some(upgrade) = &self.spec.upgrade {
            let mut upg = serde_json::Map::new();
            upg.insert("cleanupOnFail".into(), Value::Bool(upgrade.clean_up_on_fail));
            if let Some(rem) = &upgrade.remediation {
                let mut rem_map = serde_json::Map::new();
                rem_map.insert("retries".into(), serde_json::json!(rem.retries));
                if let Some(strategy) = &rem.strategy {
                    rem_map.insert("strategy".into(), Value::String(strategy.clone()));
                }
                upg.insert("remediation".into(), Value::Object(rem_map));
            }
            spec.insert("upgrade".into(), Value::Object(upg));
        }

        if self.spec.suspend {
            spec.insert("suspend".into(), Value::Bool(true));
        }

        if let Some(max) = self.spec.max_history {
            spec.insert("maxHistory".into(), serde_json::json!(max));
        }

        serde_json::json!({
            "apiVersion": self.api_version,
            "kind": self.kind,
            "metadata": Value::Object(metadata),
            "spec": Value::Object(spec),
        })
    }

    /// Add a dependency.
    pub fn add_dependency(&mut self, name: &str, namespace: Option<&str>) {
        self.spec.depends_on.push(FluxDependency {
            name: name.to_string(),
            namespace: namespace.map(String::from),
        });
    }
}

// ---------------------------------------------------------------------------
// Flux Kustomization
// ---------------------------------------------------------------------------

/// Flux Kustomization Custom Resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxKustomization {
    pub api_version: String,
    pub kind: String,
    pub metadata: FluxMetadata,
    pub spec: KustomizationSpec,
}

/// Kustomization spec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KustomizationSpec {
    pub source_ref: SourceRef,
    pub path: String,
    pub interval: String,
    pub prune: bool,
    pub depends_on: Vec<FluxDependency>,
    pub health_checks: Vec<FluxHealthCheck>,
    pub timeout: Option<String>,
    pub target_namespace: Option<String>,
    pub suspend: bool,
    pub retry_interval: Option<String>,
    pub patches: Vec<Value>,
    pub post_build: Option<PostBuild>,
}

/// Post-build variable substitution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostBuild {
    pub substitute: HashMap<String, String>,
    pub substitute_from: Vec<SubstituteFrom>,
}

/// Source for variable substitution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstituteFrom {
    pub kind: String,
    pub name: String,
}

impl FluxKustomization {
    pub fn new(name: &str, namespace: &str, source_name: &str, path: &str) -> Self {
        Self {
            api_version: "kustomize.toolkit.fluxcd.io/v1".into(),
            kind: "Kustomization".into(),
            metadata: FluxMetadata {
                name: name.to_string(),
                namespace: namespace.to_string(),
                labels: HashMap::new(),
                annotations: HashMap::new(),
            },
            spec: KustomizationSpec {
                source_ref: SourceRef {
                    kind: "GitRepository".into(),
                    name: source_name.to_string(),
                    namespace: None,
                },
                path: path.to_string(),
                interval: "5m".into(),
                prune: true,
                depends_on: Vec::new(),
                health_checks: Vec::new(),
                timeout: Some("5m".into()),
                target_namespace: None,
                suspend: false,
                retry_interval: None,
                patches: Vec::new(),
                post_build: None,
            },
        }
    }

    /// Serialize to YAML string.
    pub fn to_yaml(&self) -> Result<String> {
        let value = self.to_value();
        serde_yaml::to_string(&value).map_err(|e| SafeStepError::K8sError {
            message: format!("Failed to serialize Flux Kustomization: {e}"),
            resource: Some(self.metadata.name.clone()),
            namespace: Some(self.metadata.namespace.clone()),
            context: None,
        })
    }

    /// Convert to a serde_json Value.
    pub fn to_value(&self) -> Value {
        let mut metadata = serde_json::Map::new();
        metadata.insert("name".into(), Value::String(self.metadata.name.clone()));
        metadata.insert("namespace".into(), Value::String(self.metadata.namespace.clone()));
        if !self.metadata.labels.is_empty() {
            metadata.insert("labels".into(), serde_json::to_value(&self.metadata.labels).unwrap_or_default());
        }
        if !self.metadata.annotations.is_empty() {
            metadata.insert("annotations".into(), serde_json::to_value(&self.metadata.annotations).unwrap_or_default());
        }

        let mut spec = serde_json::Map::new();
        spec.insert("sourceRef".into(), serde_json::json!({
            "kind": self.spec.source_ref.kind,
            "name": self.spec.source_ref.name,
        }));
        spec.insert("path".into(), Value::String(self.spec.path.clone()));
        spec.insert("interval".into(), Value::String(self.spec.interval.clone()));
        spec.insert("prune".into(), Value::Bool(self.spec.prune));

        if !self.spec.depends_on.is_empty() {
            let deps: Vec<Value> = self.spec.depends_on.iter().map(|d| {
                let mut dep = serde_json::Map::new();
                dep.insert("name".into(), Value::String(d.name.clone()));
                if let Some(ns) = &d.namespace {
                    dep.insert("namespace".into(), Value::String(ns.clone()));
                }
                Value::Object(dep)
            }).collect();
            spec.insert("dependsOn".into(), Value::Array(deps));
        }

        if !self.spec.health_checks.is_empty() {
            let checks: Vec<Value> = self.spec.health_checks.iter().map(|hc| {
                serde_json::json!({
                    "apiVersion": hc.api_version,
                    "kind": hc.kind,
                    "name": hc.name,
                    "namespace": hc.namespace,
                })
            }).collect();
            spec.insert("healthChecks".into(), Value::Array(checks));
        }

        if let Some(timeout) = &self.spec.timeout {
            spec.insert("timeout".into(), Value::String(timeout.clone()));
        }

        if self.spec.suspend {
            spec.insert("suspend".into(), Value::Bool(true));
        }

        serde_json::json!({
            "apiVersion": self.api_version,
            "kind": self.kind,
            "metadata": Value::Object(metadata),
            "spec": Value::Object(spec),
        })
    }

    /// Add a dependency.
    pub fn add_dependency(&mut self, name: &str, namespace: Option<&str>) {
        self.spec.depends_on.push(FluxDependency {
            name: name.to_string(),
            namespace: namespace.map(String::from),
        });
    }

    /// Add a health check.
    pub fn add_health_check(&mut self, kind: &str, name: &str, namespace: &str, timeout: u64) {
        self.spec.health_checks.push(FluxHealthCheck {
            api_version: match kind {
                "Deployment" | "StatefulSet" | "DaemonSet" => "apps/v1".into(),
                "Service" => "v1".into(),
                _ => "apps/v1".into(),
            },
            kind: kind.to_string(),
            name: name.to_string(),
            namespace: namespace.to_string(),
            timeout_seconds: timeout,
        });
    }
}

// ---------------------------------------------------------------------------
// Flux resource union
// ---------------------------------------------------------------------------

/// Union of all Flux resource types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FluxResource {
    HelmRelease(FluxHelmRelease),
    Kustomization(FluxKustomization),
}

impl FluxResource {
    pub fn name(&self) -> &str {
        match self {
            FluxResource::HelmRelease(hr) => &hr.metadata.name,
            FluxResource::Kustomization(k) => &k.metadata.name,
        }
    }

    pub fn namespace(&self) -> &str {
        match self {
            FluxResource::HelmRelease(hr) => &hr.metadata.namespace,
            FluxResource::Kustomization(k) => &k.metadata.namespace,
        }
    }

    pub fn to_yaml(&self) -> Result<String> {
        match self {
            FluxResource::HelmRelease(hr) => hr.to_yaml(),
            FluxResource::Kustomization(k) => k.to_yaml(),
        }
    }

    pub fn to_value(&self) -> Value {
        match self {
            FluxResource::HelmRelease(hr) => hr.to_value(),
            FluxResource::Kustomization(k) => k.to_value(),
        }
    }
}

// ---------------------------------------------------------------------------
// Flux health check
// ---------------------------------------------------------------------------

/// Health check configuration for Flux resources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FluxHealthCheck {
    pub api_version: String,
    pub kind: String,
    pub name: String,
    pub namespace: String,
    pub timeout_seconds: u64,
}

impl FluxHealthCheck {
    pub fn for_deployment(name: &str, namespace: &str, timeout: u64) -> Self {
        Self {
            api_version: "apps/v1".into(),
            kind: "Deployment".into(),
            name: name.to_string(),
            namespace: namespace.to_string(),
            timeout_seconds: timeout,
        }
    }

    pub fn for_statefulset(name: &str, namespace: &str, timeout: u64) -> Self {
        Self {
            api_version: "apps/v1".into(),
            kind: "StatefulSet".into(),
            name: name.to_string(),
            namespace: namespace.to_string(),
            timeout_seconds: timeout,
        }
    }

    pub fn for_service(name: &str, namespace: &str, timeout: u64) -> Self {
        Self {
            api_version: "v1".into(),
            kind: "Service".into(),
            name: name.to_string(),
            namespace: namespace.to_string(),
            timeout_seconds: timeout,
        }
    }
}

// ---------------------------------------------------------------------------
// Flux output generator
// ---------------------------------------------------------------------------

/// Generates Flux-compatible output from a deployment plan.
pub struct FluxOutput {
    pub source_name: String,
    pub source_kind: FluxSourceKind,
    pub base_path: String,
    pub target_namespace: Option<String>,
    pub interval: String,
    pub prune: bool,
}

/// Kind of Flux source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FluxSourceKind {
    GitRepository,
    HelmRepository,
    Bucket,
    OCIRepository,
}

impl std::fmt::Display for FluxSourceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FluxSourceKind::GitRepository => write!(f, "GitRepository"),
            FluxSourceKind::HelmRepository => write!(f, "HelmRepository"),
            FluxSourceKind::Bucket => write!(f, "Bucket"),
            FluxSourceKind::OCIRepository => write!(f, "OCIRepository"),
        }
    }
}

impl Default for FluxOutput {
    fn default() -> Self {
        Self {
            source_name: "flux-system".into(),
            source_kind: FluxSourceKind::GitRepository,
            base_path: ".".into(),
            target_namespace: None,
            interval: "5m".into(),
            prune: true,
        }
    }
}

impl FluxOutput {
    pub fn new(source_name: &str) -> Self {
        Self {
            source_name: source_name.to_string(),
            ..Default::default()
        }
    }

    /// Generate Flux resources from a deployment plan.
    pub fn from_plan(&self, plan: &DeploymentPlan) -> Vec<FluxResource> {
        let mut resources = Vec::new();

        // Group steps by order to determine dependencies
        let mut steps_by_order: HashMap<u32, Vec<&DeploymentStep>> = HashMap::new();
        for step in &plan.steps {
            steps_by_order.entry(step.order).or_default().push(step);
        }

        let mut orders: Vec<u32> = steps_by_order.keys().copied().collect();
        orders.sort();

        let mut prev_order_names: Vec<String> = Vec::new();

        for order in &orders {
            let steps = steps_by_order.get(order).unwrap();
            let mut current_order_names = Vec::new();

            for step in steps {
                let mut kustomization = FluxKustomization::new(
                    &step.service_name,
                    &step.namespace,
                    &self.source_name,
                    &format!("{}/{}", self.base_path, step.service_name),
                );
                kustomization.spec.interval = self.interval.clone();
                kustomization.spec.prune = self.prune;

                if let Some(ns) = &self.target_namespace {
                    kustomization.spec.target_namespace = Some(ns.clone());
                }

                // Add dependencies from previous wave
                for dep_name in &prev_order_names {
                    kustomization.add_dependency(dep_name, None);
                }

                // Add explicit dependencies from the step
                for dep in &step.depends_on {
                    if !prev_order_names.contains(dep) {
                        kustomization.add_dependency(dep, None);
                    }
                }

                // Add health checks
                if let Some(hc) = &step.health_check {
                    kustomization.add_health_check(
                        &hc.kind,
                        &hc.name,
                        &hc.namespace,
                        hc.timeout_seconds,
                    );
                }

                // Add labels
                kustomization.metadata.labels.insert(
                    "safestep.io/managed-by".into(),
                    "safestep".into(),
                );
                kustomization.metadata.labels.insert(
                    "safestep.io/plan".into(),
                    plan.name.clone(),
                );
                kustomization.metadata.annotations.insert(
                    "safestep.io/order".into(),
                    order.to_string(),
                );

                current_order_names.push(step.service_name.clone());
                resources.push(FluxResource::Kustomization(kustomization));
            }

            prev_order_names = current_order_names;
        }

        resources
    }

    /// Generate a multi-doc YAML string for all resources.
    pub fn to_yaml(&self, plan: &DeploymentPlan) -> Result<String> {
        let resources = self.from_plan(plan);
        let mut docs = Vec::new();
        for r in &resources {
            docs.push(r.to_yaml()?);
        }
        Ok(docs.join("---\n"))
    }

    /// Generate Flux HelmReleases instead of Kustomizations.
    pub fn from_plan_as_helm(
        &self,
        plan: &DeploymentPlan,
        chart_configs: &HashMap<String, HelmReleaseConfig>,
    ) -> Vec<FluxResource> {
        let mut resources = Vec::new();
        let mut steps_by_order: HashMap<u32, Vec<&DeploymentStep>> = HashMap::new();
        for step in &plan.steps {
            steps_by_order.entry(step.order).or_default().push(step);
        }

        let mut orders: Vec<u32> = steps_by_order.keys().copied().collect();
        orders.sort();
        let mut prev_names: Vec<String> = Vec::new();

        for order in &orders {
            let steps = steps_by_order.get(order).unwrap();
            let mut current_names = Vec::new();

            for step in steps {
                let config = chart_configs.get(&step.service_name);
                let chart_name = config
                    .map(|c| c.chart.clone())
                    .unwrap_or_else(|| step.service_name.clone());
                let source_name = config
                    .map(|c| c.source_name.clone())
                    .unwrap_or_else(|| self.source_name.clone());

                let mut hr = FluxHelmRelease::new(
                    &step.service_name,
                    &step.namespace,
                    &chart_name,
                    &source_name,
                );
                hr.spec.interval = self.interval.clone();

                if let Some(cfg) = config {
                    hr.spec.chart.spec.version = cfg.version.clone();
                    hr.spec.values = cfg.values.clone();
                }

                for dep in &prev_names {
                    hr.add_dependency(dep, None);
                }
                for dep in &step.depends_on {
                    if !prev_names.contains(dep) {
                        hr.add_dependency(dep, None);
                    }
                }

                hr.metadata.labels.insert("safestep.io/managed-by".into(), "safestep".into());
                hr.metadata.labels.insert("safestep.io/plan".into(), plan.name.clone());

                current_names.push(step.service_name.clone());
                resources.push(FluxResource::HelmRelease(hr));
            }

            prev_names = current_names;
        }

        resources
    }
}

/// Configuration for generating a HelmRelease from a plan step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelmReleaseConfig {
    pub chart: String,
    pub source_name: String,
    pub version: Option<String>,
    pub values: Option<Value>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DeploymentPlan, DeploymentStep};

    fn sample_plan() -> DeploymentPlan {
        DeploymentPlan {
            name: "release-v2".into(),
            namespace: "flux-system".into(),
            steps: vec![
                DeploymentStep {
                    order: 0,
                    service_name: "database".into(),
                    namespace: "production".into(),
                    action: DeploymentAction::Apply,
                    manifests: Vec::new(),
                    health_check: Some(HealthCheckDef {
                        kind: "StatefulSet".into(),
                        name: "database".into(),
                        namespace: "production".into(),
                        timeout_seconds: 300,
                        interval_seconds: 10,
                    }),
                    depends_on: Vec::new(),
                },
                DeploymentStep {
                    order: 1,
                    service_name: "api".into(),
                    namespace: "production".into(),
                    action: DeploymentAction::Apply,
                    manifests: Vec::new(),
                    health_check: Some(HealthCheckDef {
                        kind: "Deployment".into(),
                        name: "api".into(),
                        namespace: "production".into(),
                        timeout_seconds: 120,
                        interval_seconds: 5,
                    }),
                    depends_on: vec!["database".into()],
                },
                DeploymentStep {
                    order: 2,
                    service_name: "frontend".into(),
                    namespace: "production".into(),
                    action: DeploymentAction::Apply,
                    manifests: Vec::new(),
                    health_check: None,
                    depends_on: vec!["api".into()],
                },
            ],
        }
    }

    #[test]
    fn test_flux_helm_release_new() {
        let hr = FluxHelmRelease::new("nginx", "default", "nginx", "bitnami");
        assert_eq!(hr.api_version, "helm.toolkit.fluxcd.io/v2beta2");
        assert_eq!(hr.kind, "HelmRelease");
        assert_eq!(hr.metadata.name, "nginx");
        assert_eq!(hr.spec.chart.spec.chart, "nginx");
        assert_eq!(hr.spec.chart.spec.source_ref.name, "bitnami");
    }

    #[test]
    fn test_flux_helm_release_to_yaml() {
        let hr = FluxHelmRelease::new("redis", "cache", "redis", "bitnami");
        let yaml = hr.to_yaml().unwrap();
        assert!(yaml.contains("HelmRelease"));
        assert!(yaml.contains("redis"));
        assert!(yaml.contains("helm.toolkit.fluxcd.io"));
    }

    #[test]
    fn test_flux_helm_release_dependencies() {
        let mut hr = FluxHelmRelease::new("api", "prod", "api-chart", "local");
        hr.add_dependency("database", Some("prod"));
        hr.add_dependency("redis", None);
        assert_eq!(hr.spec.depends_on.len(), 2);
        assert_eq!(hr.spec.depends_on[0].name, "database");
        assert_eq!(hr.spec.depends_on[0].namespace.as_deref(), Some("prod"));
        assert_eq!(hr.spec.depends_on[1].name, "redis");
    }

    #[test]
    fn test_flux_kustomization_new() {
        let k = FluxKustomization::new("infra", "flux-system", "flux-system", "./infrastructure");
        assert_eq!(k.api_version, "kustomize.toolkit.fluxcd.io/v1");
        assert_eq!(k.kind, "Kustomization");
        assert_eq!(k.metadata.name, "infra");
        assert_eq!(k.spec.path, "./infrastructure");
        assert!(k.spec.prune);
    }

    #[test]
    fn test_flux_kustomization_to_yaml() {
        let k = FluxKustomization::new("apps", "flux-system", "flux-system", "./apps");
        let yaml = k.to_yaml().unwrap();
        assert!(yaml.contains("Kustomization"));
        assert!(yaml.contains("kustomize.toolkit.fluxcd.io"));
        assert!(yaml.contains("apps"));
    }

    #[test]
    fn test_flux_kustomization_health_checks() {
        let mut k = FluxKustomization::new("test", "ns", "src", "./path");
        k.add_health_check("Deployment", "web", "prod", 300);
        k.add_health_check("StatefulSet", "db", "prod", 600);
        assert_eq!(k.spec.health_checks.len(), 2);
        assert_eq!(k.spec.health_checks[0].kind, "Deployment");
        assert_eq!(k.spec.health_checks[1].kind, "StatefulSet");

        let value = k.to_value();
        let checks = value["spec"]["healthChecks"].as_array().unwrap();
        assert_eq!(checks.len(), 2);
    }

    #[test]
    fn test_flux_output_from_plan() {
        let output = FluxOutput::new("flux-system");
        let plan = sample_plan();
        let resources = output.from_plan(&plan);

        assert_eq!(resources.len(), 3);

        // First resource has no dependencies
        if let FluxResource::Kustomization(k) = &resources[0] {
            assert_eq!(k.metadata.name, "database");
            assert!(k.spec.depends_on.is_empty());
            assert_eq!(k.spec.health_checks.len(), 1);
        } else {
            panic!("Expected Kustomization");
        }

        // Second resource depends on first
        if let FluxResource::Kustomization(k) = &resources[1] {
            assert_eq!(k.metadata.name, "api");
            assert!(k.spec.depends_on.iter().any(|d| d.name == "database"));
        } else {
            panic!("Expected Kustomization");
        }

        // Third resource depends on second
        if let FluxResource::Kustomization(k) = &resources[2] {
            assert_eq!(k.metadata.name, "frontend");
            assert!(k.spec.depends_on.iter().any(|d| d.name == "api"));
        } else {
            panic!("Expected Kustomization");
        }
    }

    #[test]
    fn test_flux_output_to_yaml() {
        let output = FluxOutput::new("flux-system");
        let plan = sample_plan();
        let yaml = output.to_yaml(&plan).unwrap();
        assert!(yaml.contains("database"));
        assert!(yaml.contains("api"));
        assert!(yaml.contains("frontend"));
        assert!(yaml.contains("kustomize.toolkit.fluxcd.io"));
    }

    #[test]
    fn test_flux_output_as_helm() {
        let output = FluxOutput::new("bitnami");
        let plan = DeploymentPlan {
            name: "helm-deploy".into(),
            namespace: "flux-system".into(),
            steps: vec![DeploymentStep {
                order: 0,
                service_name: "redis".into(),
                namespace: "cache".into(),
                action: DeploymentAction::Apply,
                manifests: Vec::new(),
                health_check: None,
                depends_on: Vec::new(),
            }],
        };
        let mut configs = HashMap::new();
        configs.insert(
            "redis".into(),
            HelmReleaseConfig {
                chart: "redis".into(),
                source_name: "bitnami".into(),
                version: Some("17.0.0".into()),
                values: Some(serde_json::json!({"auth": {"enabled": false}})),
            },
        );
        let resources = output.from_plan_as_helm(&plan, &configs);
        assert_eq!(resources.len(), 1);
        if let FluxResource::HelmRelease(hr) = &resources[0] {
            assert_eq!(hr.metadata.name, "redis");
            assert_eq!(hr.spec.chart.spec.version.as_deref(), Some("17.0.0"));
            assert!(hr.spec.values.is_some());
        } else {
            panic!("Expected HelmRelease");
        }
    }

    #[test]
    fn test_flux_resource_enum() {
        let hr = FluxHelmRelease::new("test", "ns", "chart", "src");
        let resource = FluxResource::HelmRelease(hr);
        assert_eq!(resource.name(), "test");
        assert_eq!(resource.namespace(), "ns");
        let yaml = resource.to_yaml().unwrap();
        assert!(yaml.contains("HelmRelease"));
    }

    #[test]
    fn test_flux_health_check_constructors() {
        let hc = FluxHealthCheck::for_deployment("web", "prod", 120);
        assert_eq!(hc.kind, "Deployment");
        assert_eq!(hc.api_version, "apps/v1");

        let hc2 = FluxHealthCheck::for_statefulset("db", "prod", 300);
        assert_eq!(hc2.kind, "StatefulSet");

        let hc3 = FluxHealthCheck::for_service("svc", "prod", 60);
        assert_eq!(hc3.kind, "Service");
        assert_eq!(hc3.api_version, "v1");
    }

    #[test]
    fn test_flux_source_kind_display() {
        assert_eq!(FluxSourceKind::GitRepository.to_string(), "GitRepository");
        assert_eq!(FluxSourceKind::HelmRepository.to_string(), "HelmRepository");
        assert_eq!(FluxSourceKind::Bucket.to_string(), "Bucket");
    }

    #[test]
    fn test_helm_release_with_values() {
        let mut hr = FluxHelmRelease::new("app", "prod", "my-chart", "my-repo");
        hr.spec.values = Some(serde_json::json!({
            "replicaCount": 3,
            "image": {"tag": "v2.0"}
        }));
        let value = hr.to_value();
        assert_eq!(value["spec"]["values"]["replicaCount"], 3);
    }

    #[test]
    fn test_multiple_deps_same_wave() {
        let plan = DeploymentPlan {
            name: "test".into(),
            namespace: "flux-system".into(),
            steps: vec![
                DeploymentStep {
                    order: 0,
                    service_name: "a".into(),
                    namespace: "prod".into(),
                    action: DeploymentAction::Apply,
                    manifests: Vec::new(),
                    health_check: None,
                    depends_on: Vec::new(),
                },
                DeploymentStep {
                    order: 0,
                    service_name: "b".into(),
                    namespace: "prod".into(),
                    action: DeploymentAction::Apply,
                    manifests: Vec::new(),
                    health_check: None,
                    depends_on: Vec::new(),
                },
                DeploymentStep {
                    order: 1,
                    service_name: "c".into(),
                    namespace: "prod".into(),
                    action: DeploymentAction::Apply,
                    manifests: Vec::new(),
                    health_check: None,
                    depends_on: Vec::new(),
                },
            ],
        };
        let output = FluxOutput::new("src");
        let resources = output.from_plan(&plan);
        assert_eq!(resources.len(), 3);
        // "c" should depend on both "a" and "b"
        if let FluxResource::Kustomization(k) = &resources[2] {
            assert_eq!(k.spec.depends_on.len(), 2);
            let dep_names: Vec<&str> = k.spec.depends_on.iter().map(|d| d.name.as_str()).collect();
            assert!(dep_names.contains(&"a"));
            assert!(dep_names.contains(&"b"));
        }
    }
}
