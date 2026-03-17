//! safestep-k8s: Kubernetes integration for the SafeStep verified deployment planner.
//!
//! This crate provides parsing of Helm charts, Kustomize overlays, raw K8s manifests,
//! and generation of ArgoCD/Flux-compatible output for GitOps deployment.

pub mod manifest;
pub mod helm;
pub mod kustomize;
pub mod resource_extraction;
pub mod argocd;
pub mod flux;
pub mod namespace;
pub mod image;
pub mod cluster;
pub mod compose;
#[cfg(feature = "kube-api")]
pub mod kube_api;

pub use manifest::*;
pub use helm::{
    HelmChart, ChartMetadata, HelmTemplate, ChartDependency,
    HelmRenderer, ValuesResolver, HelmChartLoader,
};
pub use kustomize::{
    Kustomization, KustomizePatch, StrategicMergePatch, JsonPatchOperation,
    KustomizeResolver, ImageOverride, ConfigMapGenerator,
};
pub use resource_extraction::{
    ResourceExtractor, VersionExtractor, DependencyExtractor, ResourceAggregator,
    ServiceDescriptor, ClusterResourceModel,
};
pub use argocd::{
    ArgoCdApplication, ArgoCdOutput, SyncWave, ArgoCdSyncPolicy, ArgoCdHealthCheck,
};
pub use flux::{
    FluxHelmRelease, FluxKustomization, FluxOutput, FluxResource, FluxHealthCheck,
};
pub use namespace::{NamespaceResolver, NamespaceFilter, CrossNamespaceRef};
pub use image::{ContainerImage, ImagePolicy, ImageVersionMapper, RegistryInfo};
pub use cluster::{ClusterModel, NodeInfo, ResourceCapacity, ClusterSnapshot};
pub use compose::{
    ComposeFile, ComposeService, ComposeParser, ComposeVersionExtractor,
    ComposeFormatVersion, ComposeHealthCheck, ComposeDependency, DependencyCondition,
    ComposePort, ComposeVolume, ComposeResourceLimits, ComposeNetworkRef,
    ComposeNetworkDef, ComposeVolumeDef, RestartPolicy,
};

/// A deployment plan produced by the solver, consumed by ArgoCD/Flux output generators.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DeploymentPlan {
    pub name: String,
    pub namespace: String,
    pub steps: Vec<DeploymentStep>,
}

/// A single step within a deployment plan.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DeploymentStep {
    pub order: u32,
    pub service_name: String,
    pub namespace: String,
    pub action: DeploymentAction,
    pub manifests: Vec<String>,
    pub health_check: Option<HealthCheckDef>,
    pub depends_on: Vec<String>,
}

/// What action to perform in a deployment step.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum DeploymentAction {
    Apply,
    Delete,
    Patch,
    Restart,
}

/// Health check definition attached to a step.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HealthCheckDef {
    pub kind: String,
    pub name: String,
    pub namespace: String,
    pub timeout_seconds: u64,
    pub interval_seconds: u64,
}
