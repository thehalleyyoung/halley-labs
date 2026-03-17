//! ArgoCD Application manifest generation from deployment plans.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use safestep_types::SafeStepError;

use crate::DeploymentPlan;

pub type Result<T> = std::result::Result<T, SafeStepError>;

// ---------------------------------------------------------------------------
// ArgoCD Application
// ---------------------------------------------------------------------------

/// An ArgoCD Application manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgoCdApplication {
    pub api_version: String,
    pub kind: String,
    pub metadata: ArgoCdMetadata,
    pub spec: ArgoCdAppSpec,
}

/// Metadata for an ArgoCD Application.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgoCdMetadata {
    pub name: String,
    pub namespace: String,
    pub labels: HashMap<String, String>,
    pub annotations: HashMap<String, String>,
    pub finalizers: Vec<String>,
}

/// Spec of an ArgoCD Application.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgoCdAppSpec {
    pub project: String,
    pub source: ArgoCdSource,
    pub destination: ArgoCdDestination,
    pub sync_policy: Option<ArgoCdSyncPolicy>,
    pub ignore_differences: Vec<IgnoreDifference>,
    pub info: Vec<AppInfo>,
}

/// ArgoCD application source definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgoCdSource {
    pub repo_url: String,
    pub path: Option<String>,
    pub target_revision: String,
    pub helm: Option<ArgoCdHelm>,
    pub kustomize: Option<ArgoCdKustomize>,
}

/// Helm-specific source settings for ArgoCD.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArgoCdHelm {
    pub value_files: Vec<String>,
    pub values: Option<String>,
    pub parameters: Vec<HelmParameter>,
    pub release_name: Option<String>,
}

/// A Helm parameter override.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelmParameter {
    pub name: String,
    pub value: String,
    pub force_string: bool,
}

/// Kustomize-specific source settings for ArgoCD.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArgoCdKustomize {
    pub name_prefix: Option<String>,
    pub name_suffix: Option<String>,
    pub images: Vec<String>,
    pub common_labels: HashMap<String, String>,
    pub common_annotations: HashMap<String, String>,
}

/// ArgoCD application destination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgoCdDestination {
    pub server: String,
    pub namespace: String,
}

/// Sync policy configuration for ArgoCD.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgoCdSyncPolicy {
    pub automated: Option<AutomatedSync>,
    pub sync_options: Vec<String>,
    pub retry: Option<RetryPolicy>,
}

impl Default for ArgoCdSyncPolicy {
    fn default() -> Self {
        Self {
            automated: None,
            sync_options: Vec::new(),
            retry: None,
        }
    }
}

/// Automated sync settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomatedSync {
    pub prune: bool,
    pub self_heal: bool,
    pub allow_empty: bool,
}

/// Retry policy for sync operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub limit: u32,
    pub backoff_duration: String,
    pub backoff_factor: u32,
    pub backoff_max_duration: String,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            limit: 5,
            backoff_duration: "5s".into(),
            backoff_factor: 2,
            backoff_max_duration: "3m".into(),
        }
    }
}

/// Difference to ignore during sync.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IgnoreDifference {
    pub group: Option<String>,
    pub kind: String,
    pub json_pointers: Vec<String>,
}

/// App metadata info entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppInfo {
    pub name: String,
    pub value: String,
}

impl ArgoCdApplication {
    /// Create a new ArgoCD Application with defaults.
    pub fn new(name: &str, namespace: &str, repo_url: &str, path: &str) -> Self {
        Self {
            api_version: "argoproj.io/v1alpha1".into(),
            kind: "Application".into(),
            metadata: ArgoCdMetadata {
                name: name.to_string(),
                namespace: namespace.to_string(),
                labels: HashMap::new(),
                annotations: HashMap::new(),
                finalizers: vec!["resources-finalizer.argocd.argoproj.io".into()],
            },
            spec: ArgoCdAppSpec {
                project: "default".into(),
                source: ArgoCdSource {
                    repo_url: repo_url.to_string(),
                    path: Some(path.to_string()),
                    target_revision: "HEAD".into(),
                    helm: None,
                    kustomize: None,
                },
                destination: ArgoCdDestination {
                    server: "https://kubernetes.default.svc".into(),
                    namespace: namespace.to_string(),
                },
                sync_policy: Some(ArgoCdSyncPolicy::default()),
                ignore_differences: Vec::new(),
                info: Vec::new(),
            },
        }
    }

    /// Serialize to YAML string.
    pub fn to_yaml(&self) -> Result<String> {
        let value = self.to_value();
        serde_yaml::to_string(&value).map_err(|e| SafeStepError::K8sError {
            message: format!("Failed to serialize ArgoCD Application: {e}"),
            resource: Some(self.metadata.name.clone()),
            namespace: Some(self.metadata.namespace.clone()),
            context: None,
        })
    }

    /// Convert to a serde_json Value.
    pub fn to_value(&self) -> Value {
        let mut metadata = serde_json::Map::new();
        metadata.insert("name".into(), Value::String(self.metadata.name.clone()));
        metadata.insert(
            "namespace".into(),
            Value::String(self.metadata.namespace.clone()),
        );
        if !self.metadata.labels.is_empty() {
            metadata.insert(
                "labels".into(),
                serde_json::to_value(&self.metadata.labels).unwrap_or_default(),
            );
        }
        if !self.metadata.annotations.is_empty() {
            metadata.insert(
                "annotations".into(),
                serde_json::to_value(&self.metadata.annotations).unwrap_or_default(),
            );
        }
        if !self.metadata.finalizers.is_empty() {
            metadata.insert(
                "finalizers".into(),
                serde_json::to_value(&self.metadata.finalizers).unwrap_or_default(),
            );
        }

        let mut source = serde_json::Map::new();
        source.insert("repoURL".into(), Value::String(self.spec.source.repo_url.clone()));
        if let Some(path) = &self.spec.source.path {
            source.insert("path".into(), Value::String(path.clone()));
        }
        source.insert(
            "targetRevision".into(),
            Value::String(self.spec.source.target_revision.clone()),
        );
        if let Some(helm) = &self.spec.source.helm {
            let mut helm_map = serde_json::Map::new();
            if !helm.value_files.is_empty() {
                helm_map.insert(
                    "valueFiles".into(),
                    serde_json::to_value(&helm.value_files).unwrap_or_default(),
                );
            }
            if let Some(values) = &helm.values {
                helm_map.insert("values".into(), Value::String(values.clone()));
            }
            if !helm.parameters.is_empty() {
                let params: Vec<Value> = helm.parameters
                    .iter()
                    .map(|p| {
                        serde_json::json!({
                            "name": p.name,
                            "value": p.value,
                            "forceString": p.force_string,
                        })
                    })
                    .collect();
                helm_map.insert("parameters".into(), Value::Array(params));
            }
            if let Some(name) = &helm.release_name {
                helm_map.insert("releaseName".into(), Value::String(name.clone()));
            }
            source.insert("helm".into(), Value::Object(helm_map));
        }

        let mut destination = serde_json::Map::new();
        destination.insert(
            "server".into(),
            Value::String(self.spec.destination.server.clone()),
        );
        destination.insert(
            "namespace".into(),
            Value::String(self.spec.destination.namespace.clone()),
        );

        let mut spec = serde_json::Map::new();
        spec.insert("project".into(), Value::String(self.spec.project.clone()));
        spec.insert("source".into(), Value::Object(source));
        spec.insert("destination".into(), Value::Object(destination));

        if let Some(sync_policy) = &self.spec.sync_policy {
            let mut sp = serde_json::Map::new();
            if let Some(automated) = &sync_policy.automated {
                sp.insert("automated".into(), serde_json::json!({
                    "prune": automated.prune,
                    "selfHeal": automated.self_heal,
                    "allowEmpty": automated.allow_empty,
                }));
            }
            if !sync_policy.sync_options.is_empty() {
                sp.insert(
                    "syncOptions".into(),
                    serde_json::to_value(&sync_policy.sync_options).unwrap_or_default(),
                );
            }
            if let Some(retry) = &sync_policy.retry {
                sp.insert("retry".into(), serde_json::json!({
                    "limit": retry.limit,
                    "backoff": {
                        "duration": retry.backoff_duration,
                        "factor": retry.backoff_factor,
                        "maxDuration": retry.backoff_max_duration,
                    }
                }));
            }
            spec.insert("syncPolicy".into(), Value::Object(sp));
        }

        serde_json::json!({
            "apiVersion": self.api_version,
            "kind": self.kind,
            "metadata": Value::Object(metadata),
            "spec": Value::Object(spec),
        })
    }

    /// Set the sync wave annotation for ordered deployment.
    pub fn set_sync_wave(&mut self, wave: i32) {
        self.metadata
            .annotations
            .insert("argocd.argoproj.io/sync-wave".into(), wave.to_string());
    }

    /// Add a hook annotation (PreSync, Sync, PostSync, SyncFail).
    pub fn set_hook(&mut self, hook_type: &str) {
        self.metadata
            .annotations
            .insert("argocd.argoproj.io/hook".into(), hook_type.to_string());
    }

    /// Enable automated sync with prune and self-heal.
    pub fn enable_auto_sync(&mut self, prune: bool, self_heal: bool) {
        self.spec.sync_policy = Some(ArgoCdSyncPolicy {
            automated: Some(AutomatedSync {
                prune,
                self_heal,
                allow_empty: false,
            }),
            sync_options: vec![
                "CreateNamespace=true".into(),
                "PrunePropagationPolicy=foreground".into(),
            ],
            retry: Some(RetryPolicy::default()),
        });
    }
}

// ---------------------------------------------------------------------------
// Sync wave
// ---------------------------------------------------------------------------

/// Represents an ArgoCD sync wave containing resources deployed together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncWave {
    pub wave_number: i32,
    pub resources: Vec<SyncWaveResource>,
    pub pre_sync_hooks: Vec<SyncHook>,
    pub post_sync_hooks: Vec<SyncHook>,
}

/// A resource within a sync wave.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncWaveResource {
    pub name: String,
    pub kind: String,
    pub namespace: String,
}

/// A sync hook (pre/post).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncHook {
    pub name: String,
    pub hook_type: HookType,
    pub delete_policy: HookDeletePolicy,
}

/// ArgoCD hook types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HookType {
    PreSync,
    Sync,
    PostSync,
    SyncFail,
    Skip,
}

impl std::fmt::Display for HookType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HookType::PreSync => write!(f, "PreSync"),
            HookType::Sync => write!(f, "Sync"),
            HookType::PostSync => write!(f, "PostSync"),
            HookType::SyncFail => write!(f, "SyncFail"),
            HookType::Skip => write!(f, "Skip"),
        }
    }
}

/// Delete policy for hooks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HookDeletePolicy {
    HookSucceeded,
    HookFailed,
    BeforeHookCreation,
}

// ---------------------------------------------------------------------------
// ArgoCD health check
// ---------------------------------------------------------------------------

/// Health check configuration between sync waves.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgoCdHealthCheck {
    pub type_: HealthStatus,
    pub timeout_seconds: u64,
    pub interval_seconds: u64,
    pub resource_kind: String,
    pub resource_name: String,
    pub resource_namespace: String,
}

/// ArgoCD health status for a resource.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Progressing,
    Degraded,
    Suspended,
    Missing,
    Unknown,
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealthStatus::Healthy => write!(f, "Healthy"),
            HealthStatus::Progressing => write!(f, "Progressing"),
            HealthStatus::Degraded => write!(f, "Degraded"),
            HealthStatus::Suspended => write!(f, "Suspended"),
            HealthStatus::Missing => write!(f, "Missing"),
            HealthStatus::Unknown => write!(f, "Unknown"),
        }
    }
}

impl ArgoCdHealthCheck {
    pub fn for_deployment(name: &str, namespace: &str, timeout: u64) -> Self {
        Self {
            type_: HealthStatus::Healthy,
            timeout_seconds: timeout,
            interval_seconds: 10,
            resource_kind: "Deployment".into(),
            resource_name: name.to_string(),
            resource_namespace: namespace.to_string(),
        }
    }

    pub fn for_statefulset(name: &str, namespace: &str, timeout: u64) -> Self {
        Self {
            type_: HealthStatus::Healthy,
            timeout_seconds: timeout,
            interval_seconds: 10,
            resource_kind: "StatefulSet".into(),
            resource_name: name.to_string(),
            resource_namespace: namespace.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// ArgoCD output generator
// ---------------------------------------------------------------------------

/// Generates ArgoCD-compatible output from a deployment plan.
pub struct ArgoCdOutput {
    pub repo_url: String,
    pub base_path: String,
    pub target_revision: String,
    pub cluster_server: String,
    pub project: String,
    pub auto_sync: bool,
    pub self_heal: bool,
    pub prune: bool,
}

impl Default for ArgoCdOutput {
    fn default() -> Self {
        Self {
            repo_url: String::new(),
            base_path: ".".into(),
            target_revision: "HEAD".into(),
            cluster_server: "https://kubernetes.default.svc".into(),
            project: "default".into(),
            auto_sync: false,
            self_heal: false,
            prune: false,
        }
    }
}

impl ArgoCdOutput {
    pub fn new(repo_url: &str) -> Self {
        Self {
            repo_url: repo_url.to_string(),
            ..Default::default()
        }
    }

    /// Generate ArgoCD Applications from a deployment plan.
    pub fn from_plan(&self, plan: &DeploymentPlan) -> Vec<ArgoCdApplication> {
        let waves = self.compute_sync_waves(plan);
        let mut applications = Vec::new();

        for wave in &waves {
            for resource in &wave.resources {
                let mut app = ArgoCdApplication::new(
                    &resource.name,
                    &resource.namespace,
                    &self.repo_url,
                    &self.base_path,
                );
                app.spec.project = self.project.clone();
                app.spec.source.target_revision = self.target_revision.clone();
                app.spec.destination.server = self.cluster_server.clone();
                app.spec.destination.namespace = resource.namespace.clone();
                app.set_sync_wave(wave.wave_number);

                // Set deployment plan annotations
                app.metadata.annotations.insert(
                    "safestep.io/plan-name".into(),
                    plan.name.clone(),
                );
                app.metadata.annotations.insert(
                    "safestep.io/wave".into(),
                    wave.wave_number.to_string(),
                );
                app.metadata.labels.insert(
                    "safestep.io/managed-by".into(),
                    "safestep".into(),
                );

                if self.auto_sync {
                    app.enable_auto_sync(self.prune, self.self_heal);
                }

                applications.push(app);
            }
        }

        applications
    }

    /// Compute sync waves from plan steps (each order becomes a wave).
    pub fn compute_sync_waves(&self, plan: &DeploymentPlan) -> Vec<SyncWave> {
        let mut wave_map: HashMap<u32, Vec<SyncWaveResource>> = HashMap::new();
        let mut health_checks_map: HashMap<u32, Vec<ArgoCdHealthCheck>> = HashMap::new();

        for step in &plan.steps {
            wave_map
                .entry(step.order)
                .or_default()
                .push(SyncWaveResource {
                    name: step.service_name.clone(),
                    kind: "Application".into(),
                    namespace: step.namespace.clone(),
                });

            if let Some(hc) = &step.health_check {
                health_checks_map
                    .entry(step.order)
                    .or_default()
                    .push(ArgoCdHealthCheck {
                        type_: HealthStatus::Healthy,
                        timeout_seconds: hc.timeout_seconds,
                        interval_seconds: hc.interval_seconds,
                        resource_kind: hc.kind.clone(),
                        resource_name: hc.name.clone(),
                        resource_namespace: hc.namespace.clone(),
                    });
            }
        }

        let mut orders: Vec<u32> = wave_map.keys().copied().collect();
        orders.sort();

        orders
            .into_iter()
            .map(|order| {
                let resources = wave_map.remove(&order).unwrap_or_default();
                // Create post-sync hooks for health checks
                let post_sync_hooks = health_checks_map
                    .get(&order)
                    .map(|checks| {
                        checks
                            .iter()
                            .map(|hc| SyncHook {
                                name: format!("health-check-{}", hc.resource_name),
                                hook_type: HookType::PostSync,
                                delete_policy: HookDeletePolicy::HookSucceeded,
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                SyncWave {
                    wave_number: order as i32,
                    resources,
                    pre_sync_hooks: Vec::new(),
                    post_sync_hooks,
                }
            })
            .collect()
    }

    /// Generate a multi-doc YAML string for all applications.
    pub fn to_yaml(&self, plan: &DeploymentPlan) -> Result<String> {
        let apps = self.from_plan(plan);
        let mut docs = Vec::new();
        for app in &apps {
            docs.push(app.to_yaml()?);
        }
        Ok(docs.join("---\n"))
    }

    /// Generate ArgoCD ApplicationSet for a plan (umbrella application).
    pub fn generate_app_set(&self, plan: &DeploymentPlan) -> Value {
        let generators: Vec<Value> = plan
            .steps
            .iter()
            .map(|step| {
                serde_json::json!({
                    "list": {
                        "elements": [{
                            "service": step.service_name,
                            "namespace": step.namespace,
                            "wave": step.order.to_string(),
                        }]
                    }
                })
            })
            .collect();

        serde_json::json!({
            "apiVersion": "argoproj.io/v1alpha1",
            "kind": "ApplicationSet",
            "metadata": {
                "name": format!("{}-set", plan.name),
                "namespace": plan.namespace,
            },
            "spec": {
                "generators": generators,
                "template": {
                    "metadata": {
                        "name": "{{service}}",
                        "annotations": {
                            "argocd.argoproj.io/sync-wave": "{{wave}}"
                        }
                    },
                    "spec": {
                        "project": self.project,
                        "source": {
                            "repoURL": self.repo_url,
                            "path": self.base_path,
                            "targetRevision": self.target_revision,
                        },
                        "destination": {
                            "server": self.cluster_server,
                            "namespace": "{{namespace}}"
                        }
                    }
                }
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DeploymentAction, DeploymentPlan, DeploymentStep, HealthCheckDef};

    fn sample_plan() -> DeploymentPlan {
        DeploymentPlan {
            name: "release-v2".into(),
            namespace: "argocd".into(),
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
    fn test_argocd_application_new() {
        let app = ArgoCdApplication::new(
            "my-app",
            "production",
            "https://github.com/org/repo.git",
            "k8s/overlays/prod",
        );
        assert_eq!(app.api_version, "argoproj.io/v1alpha1");
        assert_eq!(app.kind, "Application");
        assert_eq!(app.metadata.name, "my-app");
        assert_eq!(app.spec.destination.namespace, "production");
        assert_eq!(
            app.spec.source.repo_url,
            "https://github.com/org/repo.git"
        );
    }

    #[test]
    fn test_argocd_application_to_yaml() {
        let app = ArgoCdApplication::new(
            "test-app",
            "default",
            "https://github.com/org/repo.git",
            "charts/myapp",
        );
        let yaml = app.to_yaml().unwrap();
        assert!(yaml.contains("apiVersion"));
        assert!(yaml.contains("argoproj.io/v1alpha1"));
        assert!(yaml.contains("test-app"));
    }

    #[test]
    fn test_sync_wave_annotation() {
        let mut app = ArgoCdApplication::new("test", "ns", "url", "path");
        app.set_sync_wave(3);
        assert_eq!(
            app.metadata.annotations.get("argocd.argoproj.io/sync-wave").unwrap(),
            "3"
        );
    }

    #[test]
    fn test_hook_annotation() {
        let mut app = ArgoCdApplication::new("test", "ns", "url", "path");
        app.set_hook("PreSync");
        assert_eq!(
            app.metadata.annotations.get("argocd.argoproj.io/hook").unwrap(),
            "PreSync"
        );
    }

    #[test]
    fn test_auto_sync_enable() {
        let mut app = ArgoCdApplication::new("test", "ns", "url", "path");
        app.enable_auto_sync(true, true);
        let sync = app.spec.sync_policy.as_ref().unwrap();
        let auto = sync.automated.as_ref().unwrap();
        assert!(auto.prune);
        assert!(auto.self_heal);
        assert!(sync.retry.is_some());
    }

    #[test]
    fn test_argocd_output_from_plan() {
        let output = ArgoCdOutput {
            repo_url: "https://github.com/org/repo.git".into(),
            base_path: "k8s/prod".into(),
            auto_sync: true,
            self_heal: true,
            prune: true,
            ..Default::default()
        };
        let plan = sample_plan();
        let apps = output.from_plan(&plan);

        assert_eq!(apps.len(), 3);

        // Check sync waves
        assert_eq!(
            apps[0].metadata.annotations.get("argocd.argoproj.io/sync-wave").unwrap(),
            "0"
        );
        assert_eq!(
            apps[1].metadata.annotations.get("argocd.argoproj.io/sync-wave").unwrap(),
            "1"
        );
        assert_eq!(
            apps[2].metadata.annotations.get("argocd.argoproj.io/sync-wave").unwrap(),
            "2"
        );

        // Check plan annotations
        assert_eq!(
            apps[0].metadata.annotations.get("safestep.io/plan-name").unwrap(),
            "release-v2"
        );

        // Check auto-sync enabled
        assert!(apps[0].spec.sync_policy.as_ref().unwrap().automated.is_some());
    }

    #[test]
    fn test_compute_sync_waves() {
        let output = ArgoCdOutput::default();
        let plan = sample_plan();
        let waves = output.compute_sync_waves(&plan);

        assert_eq!(waves.len(), 3);
        assert_eq!(waves[0].wave_number, 0);
        assert_eq!(waves[0].resources.len(), 1);
        assert_eq!(waves[0].resources[0].name, "database");
        assert_eq!(waves[0].post_sync_hooks.len(), 1);

        assert_eq!(waves[1].wave_number, 1);
        assert_eq!(waves[1].resources[0].name, "api");
        assert_eq!(waves[1].post_sync_hooks.len(), 1);

        assert_eq!(waves[2].wave_number, 2);
        assert_eq!(waves[2].resources[0].name, "frontend");
        assert!(waves[2].post_sync_hooks.is_empty());
    }

    #[test]
    fn test_argocd_output_to_yaml() {
        let output = ArgoCdOutput::new("https://github.com/org/repo.git");
        let plan = sample_plan();
        let yaml = output.to_yaml(&plan).unwrap();
        assert!(yaml.contains("database"));
        assert!(yaml.contains("api"));
        assert!(yaml.contains("frontend"));
        assert!(yaml.contains("argoproj.io/v1alpha1"));
    }

    #[test]
    fn test_generate_app_set() {
        let output = ArgoCdOutput::new("https://github.com/org/repo.git");
        let plan = sample_plan();
        let app_set = output.generate_app_set(&plan);
        assert_eq!(app_set["kind"], "ApplicationSet");
        assert_eq!(app_set["metadata"]["name"], "release-v2-set");
    }

    #[test]
    fn test_health_check_constructors() {
        let hc = ArgoCdHealthCheck::for_deployment("nginx", "prod", 300);
        assert_eq!(hc.resource_kind, "Deployment");
        assert_eq!(hc.resource_name, "nginx");
        assert_eq!(hc.timeout_seconds, 300);

        let hc2 = ArgoCdHealthCheck::for_statefulset("pg", "db", 600);
        assert_eq!(hc2.resource_kind, "StatefulSet");
    }

    #[test]
    fn test_retry_policy_default() {
        let retry = RetryPolicy::default();
        assert_eq!(retry.limit, 5);
        assert_eq!(retry.backoff_duration, "5s");
        assert_eq!(retry.backoff_factor, 2);
    }

    #[test]
    fn test_health_status_display() {
        assert_eq!(HealthStatus::Healthy.to_string(), "Healthy");
        assert_eq!(HealthStatus::Degraded.to_string(), "Degraded");
        assert_eq!(HealthStatus::Progressing.to_string(), "Progressing");
    }

    #[test]
    fn test_hook_type_display() {
        assert_eq!(HookType::PreSync.to_string(), "PreSync");
        assert_eq!(HookType::PostSync.to_string(), "PostSync");
    }

    #[test]
    fn test_application_with_helm_source() {
        let mut app = ArgoCdApplication::new("test", "ns", "url", "charts/app");
        app.spec.source.helm = Some(ArgoCdHelm {
            value_files: vec!["values-prod.yaml".into()],
            values: None,
            parameters: vec![HelmParameter {
                name: "image.tag".into(),
                value: "v2.0".into(),
                force_string: true,
            }],
            release_name: Some("my-release".into()),
        });
        let value = app.to_value();
        assert!(value["spec"]["source"]["helm"]["valueFiles"]
            .as_array()
            .unwrap()
            .contains(&Value::String("values-prod.yaml".into())));
    }

    #[test]
    fn test_same_wave_multiple_resources() {
        let plan = DeploymentPlan {
            name: "test".into(),
            namespace: "argocd".into(),
            steps: vec![
                DeploymentStep {
                    order: 0,
                    service_name: "svc-a".into(),
                    namespace: "prod".into(),
                    action: DeploymentAction::Apply,
                    manifests: Vec::new(),
                    health_check: None,
                    depends_on: Vec::new(),
                },
                DeploymentStep {
                    order: 0,
                    service_name: "svc-b".into(),
                    namespace: "prod".into(),
                    action: DeploymentAction::Apply,
                    manifests: Vec::new(),
                    health_check: None,
                    depends_on: Vec::new(),
                },
            ],
        };
        let output = ArgoCdOutput::default();
        let waves = output.compute_sync_waves(&plan);
        assert_eq!(waves.len(), 1);
        assert_eq!(waves[0].resources.len(), 2);
    }
}
