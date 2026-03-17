//! Live Kubernetes cluster integration via kube-rs.
//!
//! This module provides a [`KubeClient`] that connects to a running Kubernetes
//! cluster and can:
//! - list Deployments, StatefulSets, and DaemonSets and extract version info,
//! - build a [`ClusterStateSnapshot`] of currently deployed service versions,
//! - apply SafeStep deployment plans (manifests in dependency order),
//! - watch rollout status until completion or timeout.
//!
//! Everything in this module is gated behind `#[cfg(feature = "kube-api")]`.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::time::Duration;

use k8s_openapi::api::apps::v1::{
    DaemonSet, DaemonSetStatus, Deployment, DeploymentStatus,
    StatefulSet, StatefulSetStatus,
};
use k8s_openapi::api::core::v1::Namespace;
use kube::api::{Api, ListParams, Patch, PatchParams};
use kube::{Client, Config, ResourceExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::time;
use tracing::{debug, info, warn};

use safestep_types::SafeStepError;

use crate::{DeploymentAction, DeploymentPlan, DeploymentStep, HealthCheckDef};

pub type Result<T> = std::result::Result<T, SafeStepError>;

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

fn k8s_err(message: impl Into<String>) -> SafeStepError {
    SafeStepError::K8sError {
        message: message.into(),
        resource: None,
        namespace: None,
        context: None,
    }
}

fn k8s_resource_err(
    message: impl Into<String>,
    resource: impl Into<String>,
    namespace: Option<String>,
) -> SafeStepError {
    SafeStepError::K8sError {
        message: message.into(),
        resource: Some(resource.into()),
        namespace,
        context: None,
    }
}

// ---------------------------------------------------------------------------
// Version extraction helpers
// ---------------------------------------------------------------------------

/// Extract a version tag from a container image string (e.g. `nginx:1.21` → `1.21`).
fn extract_image_tag(image: &str) -> Option<String> {
    // Strip digest first.
    let before_digest = image.split('@').next().unwrap_or(image);
    // Find the tag after the last colon that is *not* part of a registry port.
    if let Some(colon_idx) = before_digest.rfind(':') {
        let candidate = &before_digest[colon_idx + 1..];
        // If the candidate contains a `/` it is a registry:port/repo form, not a tag.
        if candidate.contains('/') {
            None
        } else if candidate.is_empty() {
            None
        } else {
            Some(candidate.to_string())
        }
    } else {
        None
    }
}

/// Extract the first container image from a pod template spec JSON value.
#[allow(dead_code)]
fn first_container_image(spec: &Value) -> Option<String> {
    spec.pointer("/template/spec/containers")
        .and_then(|cs| cs.as_array())
        .and_then(|arr| arr.first())
        .and_then(|c| c.get("image"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

// ---------------------------------------------------------------------------
// WorkloadKind
// ---------------------------------------------------------------------------

/// The kind of Kubernetes workload resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkloadKind {
    Deployment,
    StatefulSet,
    DaemonSet,
}

impl fmt::Display for WorkloadKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Deployment => write!(f, "Deployment"),
            Self::StatefulSet => write!(f, "StatefulSet"),
            Self::DaemonSet => write!(f, "DaemonSet"),
        }
    }
}

// ---------------------------------------------------------------------------
// WorkloadInfo
// ---------------------------------------------------------------------------

/// Summarised information about a discovered workload resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadInfo {
    pub name: String,
    pub namespace: String,
    pub kind: WorkloadKind,
    pub image: Option<String>,
    pub version_tag: Option<String>,
    pub replicas: u32,
    pub ready_replicas: u32,
    pub labels: HashMap<String, String>,
}

impl WorkloadInfo {
    /// Returns `true` when all desired replicas are ready.
    pub fn is_fully_ready(&self) -> bool {
        self.replicas > 0 && self.ready_replicas >= self.replicas
    }
}

// ---------------------------------------------------------------------------
// RolloutStatus
// ---------------------------------------------------------------------------

/// The observed status of a workload rollout.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RolloutPhase {
    Progressing,
    Complete,
    Failed,
    Unknown,
}

/// Rollout status details for a single workload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutStatus {
    pub name: String,
    pub namespace: String,
    pub kind: WorkloadKind,
    pub phase: RolloutPhase,
    pub ready_replicas: u32,
    pub desired_replicas: u32,
    pub message: Option<String>,
}

impl RolloutStatus {
    pub fn is_complete(&self) -> bool {
        self.phase == RolloutPhase::Complete
    }
}

// ---------------------------------------------------------------------------
// StepResult
// ---------------------------------------------------------------------------

/// Result of applying a single deployment step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    pub step_order: u32,
    pub service_name: String,
    pub namespace: String,
    pub action: String,
    pub success: bool,
    pub message: String,
    pub applied_manifests: usize,
}

// ---------------------------------------------------------------------------
// PlanExecutionResult
// ---------------------------------------------------------------------------

/// Aggregated result after executing a full deployment plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanExecutionResult {
    pub plan_name: String,
    pub steps_total: usize,
    pub steps_succeeded: usize,
    pub steps_failed: usize,
    pub step_results: Vec<StepResult>,
}

impl PlanExecutionResult {
    pub fn is_success(&self) -> bool {
        self.steps_failed == 0
    }
}

// ---------------------------------------------------------------------------
// ClusterStateSnapshot
// ---------------------------------------------------------------------------

/// A point-in-time snapshot of all workloads across requested namespaces.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClusterStateSnapshot {
    pub workloads: Vec<WorkloadInfo>,
    pub timestamp: String,
    pub namespaces: Vec<String>,
}

impl ClusterStateSnapshot {
    /// Look up a workload by name and namespace.
    pub fn find_workload(&self, name: &str, namespace: &str) -> Option<&WorkloadInfo> {
        self.workloads
            .iter()
            .find(|w| w.name == name && w.namespace == namespace)
    }

    /// Return all workloads in the given namespace.
    pub fn workloads_in_namespace(&self, namespace: &str) -> Vec<&WorkloadInfo> {
        self.workloads
            .iter()
            .filter(|w| w.namespace == namespace)
            .collect()
    }

    /// Build a mapping of `(namespace, name) → version_tag`.
    pub fn version_map(&self) -> HashMap<(String, String), Option<String>> {
        self.workloads
            .iter()
            .map(|w| {
                (
                    (w.namespace.clone(), w.name.clone()),
                    w.version_tag.clone(),
                )
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// KubeClient
// ---------------------------------------------------------------------------

/// A wrapper around the kube-rs [`Client`] that provides SafeStep-specific
/// operations against a live Kubernetes cluster.
pub struct KubeClient {
    client: Client,
}

impl KubeClient {
    // -- construction -------------------------------------------------------

    /// Create a [`KubeClient`] from the default kubeconfig / in-cluster
    /// service-account.
    pub async fn from_env() -> Result<Self> {
        let client = Client::try_default()
            .await
            .map_err(|e| k8s_err(format!("failed to create kube client: {e}")))?;
        info!("KubeClient connected to cluster");
        Ok(Self { client })
    }

    /// Create a [`KubeClient`] from an explicit [`Config`].
    pub async fn from_config(config: Config) -> Result<Self> {
        let client = Client::try_from(config)
            .map_err(|e| k8s_err(format!("failed to create kube client from config: {e}")))?;
        Ok(Self { client })
    }

    /// Obtain a reference to the underlying kube-rs [`Client`].
    pub fn inner(&self) -> &Client {
        &self.client
    }

    // -- namespace helpers --------------------------------------------------

    /// List all namespace names in the cluster.
    pub async fn list_namespaces(&self) -> Result<Vec<String>> {
        let ns_api: Api<Namespace> = Api::all(self.client.clone());
        let ns_list = ns_api
            .list(&ListParams::default())
            .await
            .map_err(|e| k8s_err(format!("failed to list namespaces: {e}")))?;
        Ok(ns_list.items.iter().map(|n| n.name_any()).collect())
    }

    // -- list workloads -----------------------------------------------------

    /// List Deployments in `namespace`.
    pub async fn list_deployments(&self, namespace: &str) -> Result<Vec<WorkloadInfo>> {
        let api: Api<Deployment> = Api::namespaced(self.client.clone(), namespace);
        let list = api
            .list(&ListParams::default())
            .await
            .map_err(|e| k8s_resource_err(
                format!("failed to list deployments: {e}"),
                "Deployment",
                Some(namespace.to_string()),
            ))?;

        Ok(list.items.into_iter().map(|d| deployment_to_info(d, namespace)).collect())
    }

    /// List StatefulSets in `namespace`.
    pub async fn list_statefulsets(&self, namespace: &str) -> Result<Vec<WorkloadInfo>> {
        let api: Api<StatefulSet> = Api::namespaced(self.client.clone(), namespace);
        let list = api
            .list(&ListParams::default())
            .await
            .map_err(|e| k8s_resource_err(
                format!("failed to list statefulsets: {e}"),
                "StatefulSet",
                Some(namespace.to_string()),
            ))?;

        Ok(list.items.into_iter().map(|s| statefulset_to_info(s, namespace)).collect())
    }

    /// List DaemonSets in `namespace`.
    pub async fn list_daemonsets(&self, namespace: &str) -> Result<Vec<WorkloadInfo>> {
        let api: Api<DaemonSet> = Api::namespaced(self.client.clone(), namespace);
        let list = api
            .list(&ListParams::default())
            .await
            .map_err(|e| k8s_resource_err(
                format!("failed to list daemonsets: {e}"),
                "DaemonSet",
                Some(namespace.to_string()),
            ))?;

        Ok(list.items.into_iter().map(|d| daemonset_to_info(d, namespace)).collect())
    }

    /// List all workload kinds (Deployment, StatefulSet, DaemonSet) in `namespace`.
    pub async fn list_all_workloads(&self, namespace: &str) -> Result<Vec<WorkloadInfo>> {
        let (deploys, sts, ds) = tokio::try_join!(
            self.list_deployments(namespace),
            self.list_statefulsets(namespace),
            self.list_daemonsets(namespace),
        )?;
        let mut all = Vec::with_capacity(deploys.len() + sts.len() + ds.len());
        all.extend(deploys);
        all.extend(sts);
        all.extend(ds);
        Ok(all)
    }

    // -- rollout status -----------------------------------------------------

    /// Get the rollout status for a Deployment.
    pub async fn deployment_rollout_status(
        &self,
        name: &str,
        namespace: &str,
    ) -> Result<RolloutStatus> {
        let api: Api<Deployment> = Api::namespaced(self.client.clone(), namespace);
        let deploy = api.get(name).await.map_err(|e| {
            k8s_resource_err(
                format!("failed to get deployment `{name}`: {e}"),
                name,
                Some(namespace.to_string()),
            )
        })?;
        Ok(rollout_status_from_deployment(&deploy, namespace))
    }

    /// Get the rollout status for a StatefulSet.
    pub async fn statefulset_rollout_status(
        &self,
        name: &str,
        namespace: &str,
    ) -> Result<RolloutStatus> {
        let api: Api<StatefulSet> = Api::namespaced(self.client.clone(), namespace);
        let sts = api.get(name).await.map_err(|e| {
            k8s_resource_err(
                format!("failed to get statefulset `{name}`: {e}"),
                name,
                Some(namespace.to_string()),
            )
        })?;
        Ok(rollout_status_from_statefulset(&sts, namespace))
    }

    /// Get the rollout status for a DaemonSet.
    pub async fn daemonset_rollout_status(
        &self,
        name: &str,
        namespace: &str,
    ) -> Result<RolloutStatus> {
        let api: Api<DaemonSet> = Api::namespaced(self.client.clone(), namespace);
        let ds = api.get(name).await.map_err(|e| {
            k8s_resource_err(
                format!("failed to get daemonset `{name}`: {e}"),
                name,
                Some(namespace.to_string()),
            )
        })?;
        Ok(rollout_status_from_daemonset(&ds, namespace))
    }

    /// Poll a Deployment until it is fully rolled out or `timeout` elapses.
    pub async fn wait_for_deployment_rollout(
        &self,
        name: &str,
        namespace: &str,
        timeout: Duration,
        poll_interval: Duration,
    ) -> Result<RolloutStatus> {
        let deadline = time::Instant::now() + timeout;
        loop {
            let status = self.deployment_rollout_status(name, namespace).await?;
            if status.is_complete() {
                return Ok(status);
            }
            if status.phase == RolloutPhase::Failed {
                return Ok(status);
            }
            if time::Instant::now() >= deadline {
                return Err(k8s_resource_err(
                    format!(
                        "timeout waiting for deployment `{name}` rollout ({}s)",
                        timeout.as_secs()
                    ),
                    name,
                    Some(namespace.to_string()),
                ));
            }
            debug!(
                "deployment {}/{} rollout: {}/{} ready – polling again in {}s",
                namespace,
                name,
                status.ready_replicas,
                status.desired_replicas,
                poll_interval.as_secs()
            );
            time::sleep(poll_interval).await;
        }
    }

    /// Poll a StatefulSet until it is fully rolled out or `timeout` elapses.
    pub async fn wait_for_statefulset_rollout(
        &self,
        name: &str,
        namespace: &str,
        timeout: Duration,
        poll_interval: Duration,
    ) -> Result<RolloutStatus> {
        let deadline = time::Instant::now() + timeout;
        loop {
            let status = self.statefulset_rollout_status(name, namespace).await?;
            if status.is_complete() {
                return Ok(status);
            }
            if status.phase == RolloutPhase::Failed {
                return Ok(status);
            }
            if time::Instant::now() >= deadline {
                return Err(k8s_resource_err(
                    format!(
                        "timeout waiting for statefulset `{name}` rollout ({}s)",
                        timeout.as_secs()
                    ),
                    name,
                    Some(namespace.to_string()),
                ));
            }
            debug!(
                "statefulset {}/{} rollout: {}/{} ready – polling again in {}s",
                namespace,
                name,
                status.ready_replicas,
                status.desired_replicas,
                poll_interval.as_secs()
            );
            time::sleep(poll_interval).await;
        }
    }

    /// Poll a DaemonSet until it is fully rolled out or `timeout` elapses.
    pub async fn wait_for_daemonset_rollout(
        &self,
        name: &str,
        namespace: &str,
        timeout: Duration,
        poll_interval: Duration,
    ) -> Result<RolloutStatus> {
        let deadline = time::Instant::now() + timeout;
        loop {
            let status = self.daemonset_rollout_status(name, namespace).await?;
            if status.is_complete() {
                return Ok(status);
            }
            if status.phase == RolloutPhase::Failed {
                return Ok(status);
            }
            if time::Instant::now() >= deadline {
                return Err(k8s_resource_err(
                    format!(
                        "timeout waiting for daemonset `{name}` rollout ({}s)",
                        timeout.as_secs()
                    ),
                    name,
                    Some(namespace.to_string()),
                ));
            }
            debug!(
                "daemonset {}/{} rollout: {}/{} ready – polling again in {}s",
                namespace,
                name,
                status.ready_replicas,
                status.desired_replicas,
                poll_interval.as_secs()
            );
            time::sleep(poll_interval).await;
        }
    }
}

// ---------------------------------------------------------------------------
// ClusterStateReader
// ---------------------------------------------------------------------------

/// Reads the current state of workloads across one or more namespaces and
/// produces a [`ClusterStateSnapshot`].
pub struct ClusterStateReader<'a> {
    client: &'a KubeClient,
    namespaces: Vec<String>,
    label_selector: Option<String>,
}

impl<'a> ClusterStateReader<'a> {
    /// Create a reader that will scan the given namespaces. If `namespaces` is
    /// empty, the reader will discover all namespaces in the cluster.
    pub fn new(client: &'a KubeClient, namespaces: Vec<String>) -> Self {
        Self {
            client,
            namespaces,
            label_selector: None,
        }
    }

    /// Apply an optional label selector to filter workloads.
    pub fn with_label_selector(mut self, selector: impl Into<String>) -> Self {
        self.label_selector = Some(selector.into());
        self
    }

    /// Build a [`ClusterStateSnapshot`] by querying the cluster.
    pub async fn snapshot(&self) -> Result<ClusterStateSnapshot> {
        let namespaces = if self.namespaces.is_empty() {
            self.client.list_namespaces().await?
        } else {
            self.namespaces.clone()
        };

        let mut all_workloads: Vec<WorkloadInfo> = Vec::new();
        for ns in &namespaces {
            let mut workloads = self.client.list_all_workloads(ns).await?;

            // Filter by label selector if provided.
            if let Some(ref sel) = self.label_selector {
                workloads = filter_workloads_by_selector(workloads, sel);
            }

            all_workloads.extend(workloads);
        }

        info!(
            "ClusterStateReader: discovered {} workloads across {} namespaces",
            all_workloads.len(),
            namespaces.len()
        );

        let timestamp = chrono::Utc::now().to_rfc3339();
        Ok(ClusterStateSnapshot {
            workloads: all_workloads,
            timestamp,
            namespaces,
        })
    }

    /// Build a snapshot and return a version map keyed by `(namespace, name)`.
    pub async fn version_map(&self) -> Result<HashMap<(String, String), Option<String>>> {
        let snap = self.snapshot().await?;
        Ok(snap.version_map())
    }
}

// ---------------------------------------------------------------------------
// ManifestApplier
// ---------------------------------------------------------------------------

/// Applies a SafeStep [`DeploymentPlan`] to a live cluster.
///
/// Manifests are applied step-by-step in dependency order using server-side
/// apply. After each step an optional health check polls for rollout
/// completion.
pub struct ManifestApplier<'a> {
    client: &'a KubeClient,
    dry_run: bool,
    default_timeout: Duration,
    poll_interval: Duration,
    field_manager: String,
}

impl<'a> ManifestApplier<'a> {
    pub fn new(client: &'a KubeClient) -> Self {
        Self {
            client,
            dry_run: false,
            default_timeout: Duration::from_secs(300),
            poll_interval: Duration::from_secs(5),
            field_manager: "safestep".to_string(),
        }
    }

    /// Enable dry-run mode: manifests will be validated by the API server but
    /// not persisted.
    pub fn dry_run(mut self, enabled: bool) -> Self {
        self.dry_run = enabled;
        self
    }

    /// Set the default timeout used when waiting for rollout health checks.
    pub fn default_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = timeout;
        self
    }

    /// Set the poll interval for rollout status checks.
    pub fn poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    /// Override the field-manager name used in server-side apply.
    pub fn field_manager(mut self, name: impl Into<String>) -> Self {
        self.field_manager = name.into();
        self
    }

    /// Execute a full [`DeploymentPlan`].
    pub async fn apply_plan(&self, plan: &DeploymentPlan) -> Result<PlanExecutionResult> {
        info!("applying deployment plan `{}` ({} steps)", plan.name, plan.steps.len());

        let mut step_results = Vec::with_capacity(plan.steps.len());
        let mut failed = 0usize;

        for step in &plan.steps {
            let result = self.apply_step(step).await;
            match result {
                Ok(sr) => {
                    if !sr.success {
                        failed += 1;
                    }
                    step_results.push(sr);
                }
                Err(e) => {
                    failed += 1;
                    step_results.push(StepResult {
                        step_order: step.order,
                        service_name: step.service_name.clone(),
                        namespace: step.namespace.clone(),
                        action: format!("{:?}", step.action),
                        success: false,
                        message: format!("step failed: {e}"),
                        applied_manifests: 0,
                    });
                    warn!(
                        "step {} ({}) failed: {e} – continuing with remaining steps",
                        step.order, step.service_name
                    );
                }
            }
        }

        let succeeded = step_results.len() - failed;
        Ok(PlanExecutionResult {
            plan_name: plan.name.clone(),
            steps_total: plan.steps.len(),
            steps_succeeded: succeeded,
            steps_failed: failed,
            step_results,
        })
    }

    /// Apply a single deployment step.
    async fn apply_step(&self, step: &DeploymentStep) -> Result<StepResult> {
        info!(
            "step {}: {:?} {}/{}",
            step.order, step.action, step.namespace, step.service_name
        );

        let applied = match step.action {
            DeploymentAction::Apply => self.apply_manifests(step).await?,
            DeploymentAction::Patch => self.patch_manifests(step).await?,
            DeploymentAction::Delete => self.delete_manifests(step).await?,
            DeploymentAction::Restart => self.restart_workload(step).await?,
        };

        // Optional health check.
        if let Some(ref hc) = step.health_check {
            self.run_health_check(hc).await?;
        }

        Ok(StepResult {
            step_order: step.order,
            service_name: step.service_name.clone(),
            namespace: step.namespace.clone(),
            action: format!("{:?}", step.action),
            success: true,
            message: "applied successfully".to_string(),
            applied_manifests: applied,
        })
    }

    /// Apply manifests in a step via server-side apply.
    async fn apply_manifests(&self, step: &DeploymentStep) -> Result<usize> {
        let mut count = 0usize;
        for manifest_yaml in &step.manifests {
            let value: Value = serde_yaml::from_str(manifest_yaml).map_err(|e| {
                k8s_resource_err(
                    format!("invalid manifest YAML: {e}"),
                    &step.service_name,
                    Some(step.namespace.clone()),
                )
            })?;
            self.server_side_apply(&step.namespace, &value).await?;
            count += 1;
        }
        Ok(count)
    }

    /// Patch manifests in a step (same mechanism as apply for server-side apply).
    async fn patch_manifests(&self, step: &DeploymentStep) -> Result<usize> {
        // Server-side apply is inherently a patch operation.
        self.apply_manifests(step).await
    }

    /// Delete resources referenced in a step's manifests.
    async fn delete_manifests(&self, step: &DeploymentStep) -> Result<usize> {
        let mut count = 0usize;
        for manifest_yaml in &step.manifests {
            let value: Value = serde_yaml::from_str(manifest_yaml).map_err(|e| {
                k8s_resource_err(
                    format!("invalid manifest YAML for delete: {e}"),
                    &step.service_name,
                    Some(step.namespace.clone()),
                )
            })?;
            self.delete_resource(&step.namespace, &value).await?;
            count += 1;
        }
        Ok(count)
    }

    /// Trigger a rollout restart by patching the pod template annotation.
    async fn restart_workload(&self, step: &DeploymentStep) -> Result<usize> {
        let now = chrono::Utc::now().to_rfc3339();
        let restart_patch = serde_json::json!({
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "safestep.io/restartedAt": now
                        }
                    }
                }
            }
        });

        let pp = PatchParams::apply(&self.field_manager);

        let api: Api<Deployment> =
            Api::namespaced(self.client.inner().clone(), &step.namespace);

        api.patch(&step.service_name, &pp, &Patch::Merge(&restart_patch))
            .await
            .map_err(|e| {
                k8s_resource_err(
                    format!("restart patch failed: {e}"),
                    &step.service_name,
                    Some(step.namespace.clone()),
                )
            })?;

        info!("triggered rollout restart for {}/{}", step.namespace, step.service_name);
        Ok(1)
    }

    /// Execute a server-side apply for an arbitrary manifest value.
    async fn server_side_apply(&self, namespace: &str, value: &Value) -> Result<()> {
        let kind = value
            .get("kind")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown");
        let name = value
            .pointer("/metadata/name")
            .and_then(|v| v.as_str())
            .unwrap_or("unnamed");

        debug!("server-side apply: {kind}/{name} in namespace {namespace}");

        let mut pp = PatchParams::apply(&self.field_manager);
        pp.force = true;
        if self.dry_run {
            pp.dry_run = true;
        }

        // Route to the correct typed API based on `kind`.
        match kind {
            "Deployment" => {
                let api: Api<Deployment> =
                    Api::namespaced(self.client.inner().clone(), namespace);
                api.patch(name, &pp, &Patch::Apply(value))
                    .await
                    .map_err(|e| k8s_resource_err(format!("apply failed: {e}"), name, Some(namespace.to_string())))?;
            }
            "StatefulSet" => {
                let api: Api<StatefulSet> =
                    Api::namespaced(self.client.inner().clone(), namespace);
                api.patch(name, &pp, &Patch::Apply(value))
                    .await
                    .map_err(|e| k8s_resource_err(format!("apply failed: {e}"), name, Some(namespace.to_string())))?;
            }
            "DaemonSet" => {
                let api: Api<DaemonSet> =
                    Api::namespaced(self.client.inner().clone(), namespace);
                api.patch(name, &pp, &Patch::Apply(value))
                    .await
                    .map_err(|e| k8s_resource_err(format!("apply failed: {e}"), name, Some(namespace.to_string())))?;
            }
            _ => {
                // For any other resource kind, use the dynamic API.
                self.dynamic_apply(namespace, name, value, &pp).await?;
            }
        }

        Ok(())
    }

    /// Dynamic server-side apply for arbitrary resource types.
    async fn dynamic_apply(
        &self,
        namespace: &str,
        name: &str,
        value: &Value,
        pp: &PatchParams,
    ) -> Result<()> {
        let api_version = value
            .get("apiVersion")
            .and_then(|v| v.as_str())
            .unwrap_or("v1");
        let kind = value
            .get("kind")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown");

        let (group, version) = parse_api_version(api_version);
        let plural = guess_plural(kind);

        let gvk = kube::discovery::ApiResource {
            group: group.to_string(),
            version: version.to_string(),
            api_version: api_version.to_string(),
            kind: kind.to_string(),
            plural: plural.clone(),
        };

        let api: Api<kube::api::DynamicObject> = if namespace.is_empty() {
            Api::all_with(self.client.inner().clone(), &gvk)
        } else {
            Api::namespaced_with(self.client.inner().clone(), namespace, &gvk)
        };

        api.patch(name, pp, &Patch::Apply(value))
            .await
            .map_err(|e| {
                k8s_resource_err(
                    format!("dynamic apply for {kind}/{name} failed: {e}"),
                    name,
                    Some(namespace.to_string()),
                )
            })?;

        Ok(())
    }

    /// Delete a single resource based on its manifest value.
    async fn delete_resource(&self, namespace: &str, value: &Value) -> Result<()> {
        let kind = value
            .get("kind")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown");
        let name = value
            .pointer("/metadata/name")
            .and_then(|v| v.as_str())
            .unwrap_or("unnamed");

        debug!("delete: {kind}/{name} in namespace {namespace}");

        let dp = kube::api::DeleteParams::default();

        match kind {
            "Deployment" => {
                let api: Api<Deployment> =
                    Api::namespaced(self.client.inner().clone(), namespace);
                api.delete(name, &dp)
                    .await
                    .map_err(|e| k8s_resource_err(format!("delete failed: {e}"), name, Some(namespace.to_string())))?;
            }
            "StatefulSet" => {
                let api: Api<StatefulSet> =
                    Api::namespaced(self.client.inner().clone(), namespace);
                api.delete(name, &dp)
                    .await
                    .map_err(|e| k8s_resource_err(format!("delete failed: {e}"), name, Some(namespace.to_string())))?;
            }
            "DaemonSet" => {
                let api: Api<DaemonSet> =
                    Api::namespaced(self.client.inner().clone(), namespace);
                api.delete(name, &dp)
                    .await
                    .map_err(|e| k8s_resource_err(format!("delete failed: {e}"), name, Some(namespace.to_string())))?;
            }
            _ => {
                self.dynamic_delete(namespace, name, value).await?;
            }
        }

        Ok(())
    }

    /// Dynamic delete for arbitrary resource types.
    async fn dynamic_delete(
        &self,
        namespace: &str,
        name: &str,
        value: &Value,
    ) -> Result<()> {
        let api_version = value
            .get("apiVersion")
            .and_then(|v| v.as_str())
            .unwrap_or("v1");
        let kind = value
            .get("kind")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown");

        let (group, version) = parse_api_version(api_version);
        let plural = guess_plural(kind);

        let gvk = kube::discovery::ApiResource {
            group: group.to_string(),
            version: version.to_string(),
            api_version: api_version.to_string(),
            kind: kind.to_string(),
            plural: plural.clone(),
        };

        let api: Api<kube::api::DynamicObject> = if namespace.is_empty() {
            Api::all_with(self.client.inner().clone(), &gvk)
        } else {
            Api::namespaced_with(self.client.inner().clone(), namespace, &gvk)
        };

        let dp = kube::api::DeleteParams::default();
        api.delete(name, &dp)
            .await
            .map_err(|e| {
                k8s_resource_err(
                    format!("dynamic delete for {kind}/{name} failed: {e}"),
                    name,
                    Some(namespace.to_string()),
                )
            })?;

        Ok(())
    }

    /// Run a health check by polling rollout status until the workload is
    /// ready or the configured timeout elapses.
    async fn run_health_check(&self, hc: &HealthCheckDef) -> Result<()> {
        let timeout = Duration::from_secs(hc.timeout_seconds);
        let interval = Duration::from_secs(hc.interval_seconds.max(1));

        info!(
            "health check: waiting for {}/{} ({}) – timeout {}s",
            hc.namespace, hc.name, hc.kind, hc.timeout_seconds
        );

        match hc.kind.as_str() {
            "Deployment" => {
                let status = self
                    .client
                    .wait_for_deployment_rollout(&hc.name, &hc.namespace, timeout, interval)
                    .await?;
                if !status.is_complete() {
                    return Err(k8s_resource_err(
                        format!("health check failed: {}", status.message.unwrap_or_default()),
                        &hc.name,
                        Some(hc.namespace.clone()),
                    ));
                }
            }
            "StatefulSet" => {
                let status = self
                    .client
                    .wait_for_statefulset_rollout(&hc.name, &hc.namespace, timeout, interval)
                    .await?;
                if !status.is_complete() {
                    return Err(k8s_resource_err(
                        format!("health check failed: {}", status.message.unwrap_or_default()),
                        &hc.name,
                        Some(hc.namespace.clone()),
                    ));
                }
            }
            "DaemonSet" => {
                let status = self
                    .client
                    .wait_for_daemonset_rollout(&hc.name, &hc.namespace, timeout, interval)
                    .await?;
                if !status.is_complete() {
                    return Err(k8s_resource_err(
                        format!("health check failed: {}", status.message.unwrap_or_default()),
                        &hc.name,
                        Some(hc.namespace.clone()),
                    ));
                }
            }
            other => {
                warn!("unsupported health check kind `{other}` – skipping");
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Workload → WorkloadInfo converters
// ---------------------------------------------------------------------------

fn btree_to_hash(bt: BTreeMap<String, String>) -> HashMap<String, String> {
    bt.into_iter().collect()
}

fn deployment_to_info(deploy: Deployment, namespace: &str) -> WorkloadInfo {
    let name = deploy.name_any();
    let labels = btree_to_hash(deploy.metadata.labels.clone().unwrap_or_default());
    let spec = deploy.spec.as_ref();
    let replicas = spec.and_then(|s| s.replicas).unwrap_or(1) as u32;

    let image = spec
        .and_then(|s| {
            s.template
                .spec
                .as_ref()
                .and_then(|ps| ps.containers.first())
                .map(|c| c.image.clone().unwrap_or_default())
        });

    let version_tag = image.as_deref().and_then(extract_image_tag);

    let status: Option<&DeploymentStatus> = deploy.status.as_ref();
    let ready = status.and_then(|s| s.ready_replicas).unwrap_or(0) as u32;

    WorkloadInfo {
        name,
        namespace: namespace.to_string(),
        kind: WorkloadKind::Deployment,
        image,
        version_tag,
        replicas,
        ready_replicas: ready,
        labels,
    }
}

fn statefulset_to_info(sts: StatefulSet, namespace: &str) -> WorkloadInfo {
    let name = sts.name_any();
    let labels = btree_to_hash(sts.metadata.labels.clone().unwrap_or_default());
    let spec = sts.spec.as_ref();
    let replicas = spec.and_then(|s| s.replicas).unwrap_or(1) as u32;

    let image = spec
        .and_then(|s| {
            s.template
                .spec
                .as_ref()
                .and_then(|ps| ps.containers.first())
                .map(|c| c.image.clone().unwrap_or_default())
        });

    let version_tag = image.as_deref().and_then(extract_image_tag);

    let status: Option<&StatefulSetStatus> = sts.status.as_ref();
    let ready = status.and_then(|s| s.ready_replicas).unwrap_or(0) as u32;

    WorkloadInfo {
        name,
        namespace: namespace.to_string(),
        kind: WorkloadKind::StatefulSet,
        image,
        version_tag,
        replicas,
        ready_replicas: ready,
        labels,
    }
}

fn daemonset_to_info(ds: DaemonSet, namespace: &str) -> WorkloadInfo {
    let name = ds.name_any();
    let labels = btree_to_hash(ds.metadata.labels.clone().unwrap_or_default());

    let image = ds.spec.as_ref().and_then(|s| {
        s.template
            .spec
            .as_ref()
            .and_then(|ps| ps.containers.first())
            .map(|c| c.image.clone().unwrap_or_default())
    });

    let version_tag = image.as_deref().and_then(extract_image_tag);

    let status: Option<&DaemonSetStatus> = ds.status.as_ref();
    let desired = status.map(|s| s.desired_number_scheduled).unwrap_or(0) as u32;
    let ready = status.map(|s| s.number_ready).unwrap_or(0) as u32;

    WorkloadInfo {
        name,
        namespace: namespace.to_string(),
        kind: WorkloadKind::DaemonSet,
        image,
        version_tag,
        replicas: desired,
        ready_replicas: ready,
        labels,
    }
}

// ---------------------------------------------------------------------------
// Rollout status extractors
// ---------------------------------------------------------------------------

fn rollout_status_from_deployment(deploy: &Deployment, namespace: &str) -> RolloutStatus {
    let name = deploy.name_any();
    let spec = deploy.spec.as_ref();
    let desired = spec.and_then(|s| s.replicas).unwrap_or(1) as u32;

    let status = deploy.status.as_ref();
    let ready = status.and_then(|s| s.ready_replicas).unwrap_or(0) as u32;
    let updated = status.and_then(|s| s.updated_replicas).unwrap_or(0) as u32;
    let available = status.and_then(|s| s.available_replicas).unwrap_or(0) as u32;

    let phase = if ready >= desired && updated >= desired && available >= desired {
        RolloutPhase::Complete
    } else {
        // Check conditions for failure.
        let failed = status
            .map(|s| {
                s.conditions.as_ref().map_or(false, |conds| {
                    conds.iter().any(|c| {
                        c.type_ == "Progressing"
                            && c.status == "False"
                            && c.reason.as_deref() == Some("ProgressDeadlineExceeded")
                    })
                })
            })
            .unwrap_or(false);
        if failed {
            RolloutPhase::Failed
        } else {
            RolloutPhase::Progressing
        }
    };

    let message = status.and_then(|s| {
        s.conditions
            .as_ref()
            .and_then(|conds| conds.last().map(|c| c.message.clone().unwrap_or_default()))
    });

    RolloutStatus {
        name,
        namespace: namespace.to_string(),
        kind: WorkloadKind::Deployment,
        phase,
        ready_replicas: ready,
        desired_replicas: desired,
        message,
    }
}

fn rollout_status_from_statefulset(sts: &StatefulSet, namespace: &str) -> RolloutStatus {
    let name = sts.name_any();
    let spec = sts.spec.as_ref();
    let desired = spec.and_then(|s| s.replicas).unwrap_or(1) as u32;

    let status = sts.status.as_ref();
    let ready = status.and_then(|s| s.ready_replicas).unwrap_or(0) as u32;
    let updated = status.and_then(|s| s.updated_replicas).unwrap_or(0) as u32;
    let current = status.map(|s| s.current_replicas.unwrap_or(0)).unwrap_or(0) as u32;

    let phase = if ready >= desired && updated >= desired {
        RolloutPhase::Complete
    } else {
        RolloutPhase::Progressing
    };

    RolloutStatus {
        name,
        namespace: namespace.to_string(),
        kind: WorkloadKind::StatefulSet,
        phase,
        ready_replicas: ready,
        desired_replicas: desired,
        message: Some(format!("current={current}, updated={updated}, ready={ready}")),
    }
}

fn rollout_status_from_daemonset(ds: &DaemonSet, namespace: &str) -> RolloutStatus {
    let name = ds.name_any();
    let status = ds.status.as_ref();
    let desired = status.map(|s| s.desired_number_scheduled).unwrap_or(0) as u32;
    let ready = status.map(|s| s.number_ready).unwrap_or(0) as u32;
    let updated = status.and_then(|s| s.updated_number_scheduled).unwrap_or(0) as u32;

    let phase = if ready >= desired && updated >= desired && desired > 0 {
        RolloutPhase::Complete
    } else {
        RolloutPhase::Progressing
    };

    RolloutStatus {
        name,
        namespace: namespace.to_string(),
        kind: WorkloadKind::DaemonSet,
        phase,
        ready_replicas: ready,
        desired_replicas: desired,
        message: Some(format!("desired={desired}, updated={updated}, ready={ready}")),
    }
}

// ---------------------------------------------------------------------------
// Label-selector filtering
// ---------------------------------------------------------------------------

/// Naive label selector filter: supports comma-separated `key=value` pairs.
fn filter_workloads_by_selector(
    workloads: Vec<WorkloadInfo>,
    selector: &str,
) -> Vec<WorkloadInfo> {
    let pairs: Vec<(&str, &str)> = selector
        .split(',')
        .filter_map(|part| {
            let mut kv = part.splitn(2, '=');
            match (kv.next(), kv.next()) {
                (Some(k), Some(v)) => Some((k.trim(), v.trim())),
                _ => None,
            }
        })
        .collect();

    if pairs.is_empty() {
        return workloads;
    }

    workloads
        .into_iter()
        .filter(|w| {
            pairs.iter().all(|(k, v)| {
                w.labels.get(*k).map_or(false, |lv| lv == v)
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// apiVersion / plural helpers
// ---------------------------------------------------------------------------

/// Split `apps/v1` into `("apps", "v1")` or `v1` into `("", "v1")`.
fn parse_api_version(api_version: &str) -> (&str, &str) {
    if let Some(idx) = api_version.rfind('/') {
        (&api_version[..idx], &api_version[idx + 1..])
    } else {
        ("", api_version)
    }
}

/// Guess the plural resource name from a Kind string.
///
/// Handles common irregular plurals; defaults to lowercasing and appending `s`.
fn guess_plural(kind: &str) -> String {
    match kind {
        "Ingress" => "ingresses".to_string(),
        "NetworkPolicy" => "networkpolicies".to_string(),
        "Endpoints" => "endpoints".to_string(),
        _ => {
            let lower = kind.to_ascii_lowercase();
            if lower.ends_with('s') {
                lower
            } else {
                format!("{lower}s")
            }
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
    fn test_extract_image_tag_basic() {
        assert_eq!(extract_image_tag("nginx:1.21"), Some("1.21".to_string()));
        assert_eq!(extract_image_tag("nginx:latest"), Some("latest".to_string()));
        assert_eq!(extract_image_tag("nginx"), None);
    }

    #[test]
    fn test_extract_image_tag_with_registry() {
        assert_eq!(
            extract_image_tag("registry.io/repo/app:v2.3.1"),
            Some("v2.3.1".to_string())
        );
        assert_eq!(
            extract_image_tag("localhost:5000/myapp:0.1.0"),
            Some("0.1.0".to_string())
        );
    }

    #[test]
    fn test_extract_image_tag_with_digest() {
        assert_eq!(
            extract_image_tag("nginx:1.21@sha256:abcdef"),
            Some("1.21".to_string())
        );
        assert_eq!(extract_image_tag("nginx@sha256:abcdef"), None);
    }

    #[test]
    fn test_parse_api_version() {
        assert_eq!(parse_api_version("apps/v1"), ("apps", "v1"));
        assert_eq!(parse_api_version("v1"), ("", "v1"));
        assert_eq!(
            parse_api_version("networking.k8s.io/v1"),
            ("networking.k8s.io", "v1")
        );
    }

    #[test]
    fn test_guess_plural() {
        assert_eq!(guess_plural("Deployment"), "deployments");
        assert_eq!(guess_plural("Service"), "services");
        assert_eq!(guess_plural("Ingress"), "ingresses");
        assert_eq!(guess_plural("NetworkPolicy"), "networkpolicies");
        assert_eq!(guess_plural("Endpoints"), "endpoints");
    }

    #[test]
    fn test_cluster_state_snapshot_find() {
        let snap = ClusterStateSnapshot {
            workloads: vec![
                WorkloadInfo {
                    name: "api".to_string(),
                    namespace: "prod".to_string(),
                    kind: WorkloadKind::Deployment,
                    image: Some("myapp:1.0".to_string()),
                    version_tag: Some("1.0".to_string()),
                    replicas: 3,
                    ready_replicas: 3,
                    labels: HashMap::new(),
                },
                WorkloadInfo {
                    name: "db".to_string(),
                    namespace: "prod".to_string(),
                    kind: WorkloadKind::StatefulSet,
                    image: Some("postgres:15".to_string()),
                    version_tag: Some("15".to_string()),
                    replicas: 1,
                    ready_replicas: 1,
                    labels: HashMap::new(),
                },
            ],
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            namespaces: vec!["prod".to_string()],
        };

        assert!(snap.find_workload("api", "prod").is_some());
        assert!(snap.find_workload("api", "staging").is_none());
        assert_eq!(snap.workloads_in_namespace("prod").len(), 2);
        assert_eq!(snap.version_map().len(), 2);
    }

    #[test]
    fn test_workload_info_is_fully_ready() {
        let ready = WorkloadInfo {
            name: "app".to_string(),
            namespace: "default".to_string(),
            kind: WorkloadKind::Deployment,
            image: None,
            version_tag: None,
            replicas: 3,
            ready_replicas: 3,
            labels: HashMap::new(),
        };
        assert!(ready.is_fully_ready());

        let not_ready = WorkloadInfo {
            replicas: 3,
            ready_replicas: 2,
            ..ready.clone()
        };
        assert!(!not_ready.is_fully_ready());

        let zero = WorkloadInfo {
            replicas: 0,
            ready_replicas: 0,
            ..ready
        };
        assert!(!zero.is_fully_ready());
    }

    #[test]
    fn test_filter_workloads_by_selector() {
        let mut labels = HashMap::new();
        labels.insert("app".to_string(), "web".to_string());
        labels.insert("env".to_string(), "prod".to_string());

        let workloads = vec![
            WorkloadInfo {
                name: "web".to_string(),
                namespace: "default".to_string(),
                kind: WorkloadKind::Deployment,
                image: None,
                version_tag: None,
                replicas: 1,
                ready_replicas: 1,
                labels: labels.clone(),
            },
            WorkloadInfo {
                name: "worker".to_string(),
                namespace: "default".to_string(),
                kind: WorkloadKind::Deployment,
                image: None,
                version_tag: None,
                replicas: 1,
                ready_replicas: 1,
                labels: {
                    let mut m = HashMap::new();
                    m.insert("app".to_string(), "worker".to_string());
                    m
                },
            },
        ];

        let filtered = filter_workloads_by_selector(workloads.clone(), "app=web");
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].name, "web");

        let filtered = filter_workloads_by_selector(workloads.clone(), "app=web,env=prod");
        assert_eq!(filtered.len(), 1);

        let filtered = filter_workloads_by_selector(workloads.clone(), "");
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_plan_execution_result() {
        let result = PlanExecutionResult {
            plan_name: "test".to_string(),
            steps_total: 3,
            steps_succeeded: 3,
            steps_failed: 0,
            step_results: vec![],
        };
        assert!(result.is_success());

        let failed = PlanExecutionResult {
            steps_failed: 1,
            steps_succeeded: 2,
            ..result
        };
        assert!(!failed.is_success());
    }

    #[test]
    fn test_rollout_status_is_complete() {
        let complete = RolloutStatus {
            name: "app".to_string(),
            namespace: "default".to_string(),
            kind: WorkloadKind::Deployment,
            phase: RolloutPhase::Complete,
            ready_replicas: 3,
            desired_replicas: 3,
            message: None,
        };
        assert!(complete.is_complete());

        let progressing = RolloutStatus {
            phase: RolloutPhase::Progressing,
            ..complete
        };
        assert!(!progressing.is_complete());
    }
}
