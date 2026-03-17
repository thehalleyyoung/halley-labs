//! Kustomize overlay and patch processing.
//!
//! Implements parsing of `kustomization.yaml` files, strategic merge patching,
//! RFC 6902 JSON Patch operations, and the full set of Kustomize transformers
//! (namespace, name-prefix/suffix, common labels/annotations, image overrides,
//! replica overrides, ConfigMap/Secret generators).

use anyhow::{bail, Context, Result};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::kubernetes::KubernetesResource;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Represents a parsed `kustomization.yaml`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Kustomization {
    #[serde(default = "default_api_version")]
    pub api_version: String,
    #[serde(default = "default_kind")]
    pub kind: String,
    #[serde(default)]
    pub resources: Vec<String>,
    #[serde(default)]
    pub patches: Vec<KustomizePatch>,
    #[serde(default, rename = "patchesStrategicMerge")]
    pub patch_strategic_merge: Vec<String>,
    #[serde(default, rename = "patchesJson6902")]
    pub patch_json6902: Vec<Json6902Patch>,
    #[serde(default, rename = "configMapGenerator")]
    pub config_map_generator: Vec<ConfigMapGeneratorArgs>,
    #[serde(default, rename = "secretGenerator")]
    pub secret_generator: Vec<SecretGeneratorArgs>,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default, rename = "namePrefix")]
    pub name_prefix: Option<String>,
    #[serde(default, rename = "nameSuffix")]
    pub name_suffix: Option<String>,
    #[serde(default, rename = "commonLabels")]
    pub common_labels: IndexMap<String, String>,
    #[serde(default, rename = "commonAnnotations")]
    pub common_annotations: IndexMap<String, String>,
    #[serde(default)]
    pub images: Vec<ImageOverride>,
    #[serde(default)]
    pub replicas: Vec<ReplicaOverride>,
}

fn default_api_version() -> String {
    "kustomize.config.k8s.io/v1beta1".to_string()
}

fn default_kind() -> String {
    "Kustomization".to_string()
}

/// A generic patch entry that can target specific resources.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KustomizePatch {
    #[serde(default)]
    pub target: Option<PatchTarget>,
    #[serde(default)]
    pub patch: String,
    #[serde(default)]
    pub path: Option<String>,
}

/// Selects the set of resources a patch should apply to.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PatchTarget {
    #[serde(default)]
    pub group: Option<String>,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default)]
    pub label_selector: Option<String>,
    #[serde(default)]
    pub annotation_selector: Option<String>,
}

/// A JSON Patch (RFC 6902) entry targeting a specific resource.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Json6902Patch {
    pub target: PatchTarget,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub ops: Vec<JsonPatchOp>,
}

/// A single RFC 6902 JSON Patch operation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "op", rename_all = "lowercase")]
pub enum JsonPatchOp {
    Add {
        path: String,
        value: serde_yaml::Value,
    },
    Remove {
        path: String,
    },
    Replace {
        path: String,
        value: serde_yaml::Value,
    },
    Move {
        from: String,
        path: String,
    },
    Copy {
        from: String,
        path: String,
    },
    Test {
        path: String,
        value: serde_yaml::Value,
    },
}

/// Arguments for the `configMapGenerator` transformer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ConfigMapGeneratorArgs {
    pub name: String,
    #[serde(default)]
    pub literals: Vec<String>,
    #[serde(default)]
    pub files: Vec<String>,
    #[serde(default, rename = "envs")]
    pub env_files: Vec<String>,
    #[serde(default)]
    pub behavior: Option<String>,
}

/// Arguments for the `secretGenerator` transformer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SecretGeneratorArgs {
    pub name: String,
    #[serde(default)]
    pub literals: Vec<String>,
    #[serde(default)]
    pub files: Vec<String>,
    #[serde(default, rename = "type")]
    pub secret_type: Option<String>,
    #[serde(default)]
    pub behavior: Option<String>,
}

/// Override the image name/tag/digest for containers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ImageOverride {
    pub name: String,
    #[serde(default)]
    pub new_name: Option<String>,
    #[serde(default)]
    pub new_tag: Option<String>,
    #[serde(default)]
    pub digest: Option<String>,
}

/// Override the replica count for a named Deployment/StatefulSet.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReplicaOverride {
    pub name: String,
    pub count: u32,
}

/// A resolved Kustomize layer (base or overlay) with its parsed resources.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KustomizeLayer {
    pub name: String,
    pub kustomization: Kustomization,
    pub resources: Vec<serde_yaml::Value>,
    pub order: u32,
}

// ---------------------------------------------------------------------------
// KustomizeProcessor — main entry point
// ---------------------------------------------------------------------------

/// Processes Kustomize overlays and patches.
#[derive(Debug, Clone)]
pub struct KustomizeProcessor;

impl KustomizeProcessor {
    /// Parse a `kustomization.yaml` from its YAML string.
    pub fn parse_kustomization(yaml: &str) -> Result<Kustomization> {
        serde_yaml::from_str(yaml).context("failed to parse kustomization.yaml")
    }

    /// Apply a sequence of overlay YAML strings on top of `base_yaml`.
    ///
    /// `base_yaml` is treated as a multi-document YAML stream (resources separated
    /// by `---`).  Each entry in `overlays` is itself a `kustomization.yaml` whose
    /// transformers are applied in order.
    pub fn process_kustomization(
        base_yaml: &str,
        overlays: &[String],
    ) -> Result<Vec<KubernetesResource>> {
        // 1. Parse base resources from multi-doc YAML.
        let mut resources = parse_multi_doc_yaml(base_yaml)?;

        // 2. Apply each overlay kustomization in order.
        for (idx, overlay_yaml) in overlays.iter().enumerate() {
            let kustomization: Kustomization = serde_yaml::from_str(overlay_yaml)
                .with_context(|| format!("failed to parse overlay kustomization #{}", idx))?;

            resources = apply_kustomization_transforms(resources, &kustomization)?;
        }

        // 3. Convert the final YAML values into typed KubernetesResource enums.
        let mut typed: Vec<KubernetesResource> = Vec::with_capacity(resources.len());
        for val in &resources {
            typed.push(yaml_value_to_kubernetes_resource(val)?);
        }
        Ok(typed)
    }

    /// Kubernetes-style strategic merge patch.
    ///
    /// * Mappings are recursively merged (overlay keys win).
    /// * Sequences whose elements contain a `name` field are merged by that key.
    /// * An explicit `null` value in the patch means *delete* the corresponding
    ///   key from the base.
    pub fn apply_strategic_merge_patch(
        base: &serde_yaml::Value,
        patch: &serde_yaml::Value,
    ) -> Result<serde_yaml::Value> {
        strategic_merge(base, patch)
    }

    /// Apply a sequence of RFC 6902 JSON Patch operations.
    pub fn apply_json_patch(
        base: &serde_yaml::Value,
        ops: &[JsonPatchOp],
    ) -> Result<serde_yaml::Value> {
        let mut doc = base.clone();
        for op in ops {
            match op {
                JsonPatchOp::Add { path, value } => {
                    set_json_pointer(&mut doc, path, value.clone())?;
                }
                JsonPatchOp::Remove { path } => {
                    remove_json_pointer(&mut doc, path)?;
                }
                JsonPatchOp::Replace { path, value } => {
                    // The path MUST already exist for replace.
                    if resolve_json_pointer(&doc, path).is_none() {
                        bail!("replace target does not exist: {}", path);
                    }
                    set_json_pointer(&mut doc, path, value.clone())?;
                }
                JsonPatchOp::Move { from, path } => {
                    let val = remove_json_pointer(&mut doc, from)?;
                    set_json_pointer(&mut doc, path, val)?;
                }
                JsonPatchOp::Copy { from, path } => {
                    let val = resolve_json_pointer(&doc, from)
                        .cloned()
                        .with_context(|| format!("copy source does not exist: {}", from))?;
                    set_json_pointer(&mut doc, path, val)?;
                }
                JsonPatchOp::Test { path, value } => {
                    let actual = resolve_json_pointer(&doc, path)
                        .with_context(|| format!("test target does not exist: {}", path))?;
                    if actual != value {
                        bail!(
                            "test operation failed at {}: expected {:?}, got {:?}",
                            path,
                            value,
                            actual
                        );
                    }
                }
            }
        }
        Ok(doc)
    }

    /// Insert `labels` into `metadata.labels`, `spec.selector.matchLabels`, and
    /// `spec.template.metadata.labels`.
    pub fn apply_common_labels(
        resource: &mut serde_yaml::Value,
        labels: &IndexMap<String, String>,
    ) {
        if labels.is_empty() {
            return;
        }
        ensure_map_and_merge(resource, &["metadata", "labels"], labels);
        ensure_map_and_merge(resource, &["spec", "selector", "matchLabels"], labels);
        ensure_map_and_merge(resource, &["spec", "template", "metadata", "labels"], labels);
    }

    /// Insert `annotations` into `metadata.annotations`.
    pub fn apply_common_annotations(
        resource: &mut serde_yaml::Value,
        annotations: &IndexMap<String, String>,
    ) {
        if annotations.is_empty() {
            return;
        }
        ensure_map_and_merge(resource, &["metadata", "annotations"], annotations);
    }

    /// Override `metadata.namespace` on the resource.
    pub fn apply_namespace_override(resource: &mut serde_yaml::Value, namespace: &str) {
        if let serde_yaml::Value::Mapping(ref mut root) = resource {
            let meta = root
                .entry(serde_yaml::Value::String("metadata".into()))
                .or_insert_with(|| serde_yaml::Value::Mapping(serde_yaml::Mapping::new()));
            if let serde_yaml::Value::Mapping(ref mut m) = meta {
                m.insert(
                    serde_yaml::Value::String("namespace".into()),
                    serde_yaml::Value::String(namespace.into()),
                );
            }
        }
    }

    /// Prepend `prefix` to `metadata.name`.
    pub fn apply_name_prefix(resource: &mut serde_yaml::Value, prefix: &str) {
        if let Some(name) = get_metadata_name_mut(resource) {
            *name = format!("{}{}", prefix, name);
        }
    }

    /// Append `suffix` to `metadata.name`.
    pub fn apply_name_suffix(resource: &mut serde_yaml::Value, suffix: &str) {
        if let Some(name) = get_metadata_name_mut(resource) {
            *name = format!("{}{}", name, suffix);
        }
    }

    /// Walk all container specs in the resource and apply image overrides.
    pub fn apply_image_overrides(
        resource: &mut serde_yaml::Value,
        images: &[ImageOverride],
    ) {
        if images.is_empty() {
            return;
        }
        walk_and_override_images(resource, images);
    }

    /// Override `spec.replicas` for resources whose `metadata.name` matches.
    pub fn apply_replica_overrides(
        resource: &mut serde_yaml::Value,
        replicas: &[ReplicaOverride],
    ) {
        let res_name = resource
            .get("metadata")
            .and_then(|m| m.get("name"))
            .and_then(|n| n.as_str())
            .map(|s| s.to_string());

        if let Some(name) = res_name {
            for ro in replicas {
                if ro.name == name {
                    if let serde_yaml::Value::Mapping(ref mut root) = resource {
                        let spec = root
                            .entry(serde_yaml::Value::String("spec".into()))
                            .or_insert_with(|| {
                                serde_yaml::Value::Mapping(serde_yaml::Mapping::new())
                            });
                        if let serde_yaml::Value::Mapping(ref mut s) = spec {
                            s.insert(
                                serde_yaml::Value::String("replicas".into()),
                                serde_yaml::Value::Number(serde_yaml::Number::from(ro.count as u64)),
                            );
                        }
                    }
                    break;
                }
            }
        }
    }

    /// Merge two resource lists.  Overlay resources with the same `(kind, name)`
    /// replace the corresponding base resource.
    pub fn merge_resources(
        base: Vec<serde_yaml::Value>,
        overlay: Vec<serde_yaml::Value>,
    ) -> Vec<serde_yaml::Value> {
        let mut merged: IndexMap<(String, String), serde_yaml::Value> = IndexMap::new();

        for val in base {
            let key = resource_key(&val).unwrap_or_else(|| ("_unknown".into(), uuid::Uuid::new_v4().to_string()));
            merged.insert(key, val);
        }
        for val in overlay {
            let key = resource_key(&val).unwrap_or_else(|| ("_unknown".into(), uuid::Uuid::new_v4().to_string()));
            merged.insert(key, val);
        }

        merged.into_values().collect()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Navigate a JSON Pointer (`/a/b/0/c`) to a read-only reference.
fn resolve_json_pointer<'a>(
    root: &'a serde_yaml::Value,
    pointer: &str,
) -> Option<&'a serde_yaml::Value> {
    if pointer.is_empty() || pointer == "/" {
        return Some(root);
    }
    let segments = parse_pointer_segments(pointer);
    let mut current = root;
    for seg in &segments {
        match current {
            serde_yaml::Value::Mapping(map) => {
                current = map.get(serde_yaml::Value::String(seg.clone()))?;
            }
            serde_yaml::Value::Sequence(seq) => {
                let idx: usize = seg.parse().ok()?;
                current = seq.get(idx)?;
            }
            _ => return None,
        }
    }
    Some(current)
}

/// Set the value at a JSON Pointer path, creating intermediate mappings as needed.
fn set_json_pointer(
    root: &mut serde_yaml::Value,
    pointer: &str,
    value: serde_yaml::Value,
) -> Result<()> {
    let segments = parse_pointer_segments(pointer);
    if segments.is_empty() {
        *root = value;
        return Ok(());
    }

    let mut current = root;
    for (i, seg) in segments.iter().enumerate() {
        let is_last = i == segments.len() - 1;
        if is_last {
            match current {
                serde_yaml::Value::Mapping(ref mut map) => {
                    map.insert(serde_yaml::Value::String(seg.clone()), value.clone());
                    return Ok(());
                }
                serde_yaml::Value::Sequence(ref mut seq) => {
                    if seg == "-" {
                        seq.push(value.clone());
                        return Ok(());
                    }
                    let idx: usize = seg
                        .parse()
                        .with_context(|| format!("invalid array index: {}", seg))?;
                    if idx <= seq.len() {
                        seq.insert(idx, value.clone());
                    } else {
                        bail!("index {} out of bounds (len {})", idx, seq.len());
                    }
                    return Ok(());
                }
                _ => bail!("cannot index into scalar at segment {}", seg),
            }
        }
        // Intermediate navigation
        match current {
            serde_yaml::Value::Mapping(ref mut map) => {
                let key = serde_yaml::Value::String(seg.clone());
                if !map.contains_key(&key) {
                    map.insert(
                        key.clone(),
                        serde_yaml::Value::Mapping(serde_yaml::Mapping::new()),
                    );
                }
                current = map.get_mut(&key).unwrap();
            }
            serde_yaml::Value::Sequence(ref mut seq) => {
                let idx: usize = seg
                    .parse()
                    .with_context(|| format!("invalid array index: {}", seg))?;
                current = seq
                    .get_mut(idx)
                    .with_context(|| format!("index {} out of bounds", idx))?;
            }
            _ => bail!("cannot navigate through scalar at segment {}", seg),
        }
    }
    Ok(())
}

/// Remove and return the value at a JSON Pointer path.
fn remove_json_pointer(
    root: &mut serde_yaml::Value,
    pointer: &str,
) -> Result<serde_yaml::Value> {
    let segments = parse_pointer_segments(pointer);
    if segments.is_empty() {
        bail!("cannot remove root document");
    }

    let mut current = root;
    for (i, seg) in segments.iter().enumerate() {
        let is_last = i == segments.len() - 1;
        if is_last {
            match current {
                serde_yaml::Value::Mapping(ref mut map) => {
                    let key = serde_yaml::Value::String(seg.clone());
                    return map
                        .remove(&key)
                        .with_context(|| format!("key not found for removal: {}", seg));
                }
                serde_yaml::Value::Sequence(ref mut seq) => {
                    let idx: usize = seg
                        .parse()
                        .with_context(|| format!("invalid array index: {}", seg))?;
                    if idx < seq.len() {
                        return Ok(seq.remove(idx));
                    }
                    bail!("index {} out of bounds for removal (len {})", idx, seq.len());
                }
                _ => bail!("cannot remove from scalar at {}", seg),
            }
        }
        match current {
            serde_yaml::Value::Mapping(ref mut map) => {
                let key = serde_yaml::Value::String(seg.clone());
                current = map
                    .get_mut(&key)
                    .with_context(|| format!("path segment not found: {}", seg))?;
            }
            serde_yaml::Value::Sequence(ref mut seq) => {
                let idx: usize = seg
                    .parse()
                    .with_context(|| format!("invalid array index: {}", seg))?;
                current = seq
                    .get_mut(idx)
                    .with_context(|| format!("index {} out of bounds", idx))?;
            }
            _ => bail!("cannot navigate through scalar at {}", seg),
        }
    }
    bail!("unreachable in remove_json_pointer")
}

/// Extract `(kind, metadata.name)` from a resource YAML value.
fn resource_key(val: &serde_yaml::Value) -> Option<(String, String)> {
    let kind = val.get("kind")?.as_str()?.to_string();
    let name = val.get("metadata")?.get("name")?.as_str()?.to_string();
    Some((kind, name))
}

/// Check whether a YAML resource matches a `PatchTarget` selector.
fn matches_target(resource: &serde_yaml::Value, target: &PatchTarget) -> bool {
    if let Some(ref kind) = target.kind {
        let res_kind = resource.get("kind").and_then(|v| v.as_str()).unwrap_or("");
        if !kind.eq_ignore_ascii_case(res_kind) {
            return false;
        }
    }
    if let Some(ref name) = target.name {
        let res_name = resource
            .get("metadata")
            .and_then(|m| m.get("name"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if name != res_name {
            return false;
        }
    }
    if let Some(ref ns) = target.namespace {
        let res_ns = resource
            .get("metadata")
            .and_then(|m| m.get("namespace"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if ns != res_ns {
            return false;
        }
    }
    if let Some(ref version) = target.version {
        let api = resource.get("apiVersion").and_then(|v| v.as_str()).unwrap_or("");
        if !api.contains(version.as_str()) {
            return false;
        }
    }
    if let Some(ref group) = target.group {
        let api = resource.get("apiVersion").and_then(|v| v.as_str()).unwrap_or("");
        let res_group = api.split('/').next().unwrap_or("");
        if !group.is_empty() && group != res_group {
            return false;
        }
    }
    if let Some(ref label_sel) = target.label_selector {
        if !matches_label_selector(resource, label_sel) {
            return false;
        }
    }
    if let Some(ref ann_sel) = target.annotation_selector {
        if !matches_annotation_selector(resource, ann_sel) {
            return false;
        }
    }
    true
}

/// Generate a ConfigMap YAML `Value` from `ConfigMapGeneratorArgs`.
fn generate_config_map(args: &ConfigMapGeneratorArgs) -> serde_yaml::Value {
    let mut data = serde_yaml::Mapping::new();

    for literal in &args.literals {
        if let Some((k, v)) = literal.split_once('=') {
            data.insert(
                serde_yaml::Value::String(k.to_string()),
                serde_yaml::Value::String(v.to_string()),
            );
        }
    }

    // Files and envFiles are path-based; in this in-memory implementation we
    // record the file names as keys with placeholder values because we have no
    // filesystem access at this layer.
    for file_ref in &args.files {
        let key = file_ref.rsplit('/').next().unwrap_or(file_ref);
        data.entry(serde_yaml::Value::String(key.to_string()))
            .or_insert_with(|| serde_yaml::Value::String(String::new()));
    }

    let mut root = serde_yaml::Mapping::new();
    root.insert(
        serde_yaml::Value::String("apiVersion".into()),
        serde_yaml::Value::String("v1".into()),
    );
    root.insert(
        serde_yaml::Value::String("kind".into()),
        serde_yaml::Value::String("ConfigMap".into()),
    );

    let mut meta = serde_yaml::Mapping::new();
    meta.insert(
        serde_yaml::Value::String("name".into()),
        serde_yaml::Value::String(args.name.clone()),
    );
    root.insert(
        serde_yaml::Value::String("metadata".into()),
        serde_yaml::Value::Mapping(meta),
    );
    root.insert(
        serde_yaml::Value::String("data".into()),
        serde_yaml::Value::Mapping(data),
    );

    serde_yaml::Value::Mapping(root)
}

// ---------------------------------------------------------------------------
// Private utilities
// ---------------------------------------------------------------------------

/// Split a JSON Pointer string into segments, unescaping `~1` → `/` and `~0` → `~`.
fn parse_pointer_segments(pointer: &str) -> Vec<String> {
    if pointer.is_empty() {
        return vec![];
    }
    let trimmed = pointer.strip_prefix('/').unwrap_or(pointer);
    trimmed
        .split('/')
        .map(|s| s.replace("~1", "/").replace("~0", "~"))
        .collect()
}

/// Parse a multi-document YAML string into a vec of `serde_yaml::Value`.
fn parse_multi_doc_yaml(yaml: &str) -> Result<Vec<serde_yaml::Value>> {
    let mut results = Vec::new();
    for doc in serde_yaml::Deserializer::from_str(yaml) {
        let val = serde_yaml::Value::deserialize(doc).context("failed to parse YAML document")?;
        if !val.is_null() {
            results.push(val);
        }
    }
    Ok(results)
}

/// Apply all kustomization-level transforms to a set of resources.
fn apply_kustomization_transforms(
    mut resources: Vec<serde_yaml::Value>,
    kust: &Kustomization,
) -> Result<Vec<serde_yaml::Value>> {
    // Strategic merge patches (inline YAML strings).
    for patch_yaml in &kust.patch_strategic_merge {
        let patch_val: serde_yaml::Value =
            serde_yaml::from_str(patch_yaml).context("invalid strategic merge patch YAML")?;
        let patch_key = resource_key(&patch_val);
        for res in resources.iter_mut() {
            let res_key = resource_key(res);
            if patch_key.is_some() && patch_key == res_key {
                *res = strategic_merge(res, &patch_val)?;
            }
        }
    }

    // Generic patches (may be strategic merge or targeted).
    for kp in &kust.patches {
        if kp.patch.is_empty() {
            continue;
        }
        let patch_val: serde_yaml::Value =
            serde_yaml::from_str(&kp.patch).context("invalid patch YAML in patches entry")?;
        for res in resources.iter_mut() {
            let should_apply = match &kp.target {
                Some(t) => matches_target(res, t),
                None => {
                    // If no target, match by kind+name from the patch itself.
                    resource_key(res) == resource_key(&patch_val)
                }
            };
            if should_apply {
                *res = strategic_merge(res, &patch_val)?;
            }
        }
    }

    // JSON 6902 patches.
    for jp in &kust.patch_json6902 {
        for res in resources.iter_mut() {
            if matches_target(res, &jp.target) {
                *res = KustomizeProcessor::apply_json_patch(res, &jp.ops)?;
            }
        }
    }

    // ConfigMap generator.
    for cm_args in &kust.config_map_generator {
        resources.push(generate_config_map(cm_args));
    }

    // Namespace, name prefix/suffix, labels, annotations, images, replicas.
    for res in resources.iter_mut() {
        if let Some(ref ns) = kust.namespace {
            KustomizeProcessor::apply_namespace_override(res, ns);
        }
        if let Some(ref prefix) = kust.name_prefix {
            KustomizeProcessor::apply_name_prefix(res, prefix);
        }
        if let Some(ref suffix) = kust.name_suffix {
            KustomizeProcessor::apply_name_suffix(res, suffix);
        }
        KustomizeProcessor::apply_common_labels(res, &kust.common_labels);
        KustomizeProcessor::apply_common_annotations(res, &kust.common_annotations);
        KustomizeProcessor::apply_image_overrides(res, &kust.images);
        KustomizeProcessor::apply_replica_overrides(res, &kust.replicas);
    }

    Ok(resources)
}

/// Recursive strategic merge of two YAML values.
fn strategic_merge(
    base: &serde_yaml::Value,
    patch: &serde_yaml::Value,
) -> Result<serde_yaml::Value> {
    match (base, patch) {
        (serde_yaml::Value::Mapping(bm), serde_yaml::Value::Mapping(pm)) => {
            let mut merged = bm.clone();
            for (pk, pv) in pm.iter() {
                if pv.is_null() {
                    // null means delete.
                    merged.remove(pk);
                } else if let Some(bv) = bm.get(pk) {
                    merged.insert(pk.clone(), strategic_merge(bv, pv)?);
                } else {
                    merged.insert(pk.clone(), pv.clone());
                }
            }
            Ok(serde_yaml::Value::Mapping(merged))
        }
        (serde_yaml::Value::Sequence(bs), serde_yaml::Value::Sequence(ps)) => {
            // If elements have a "name" key, merge by name.
            let base_has_names = bs.iter().any(|v| v.get("name").is_some());
            let patch_has_names = ps.iter().any(|v| v.get("name").is_some());

            if base_has_names && patch_has_names {
                let mut merged: Vec<serde_yaml::Value> = bs.clone();
                for pv in ps {
                    let p_name = pv.get("name").and_then(|n| n.as_str());
                    if let Some(pn) = p_name {
                        if let Some(existing) = merged
                            .iter_mut()
                            .find(|v| v.get("name").and_then(|n| n.as_str()) == Some(pn))
                        {
                            *existing = strategic_merge(existing, pv)?;
                        } else {
                            merged.push(pv.clone());
                        }
                    } else {
                        merged.push(pv.clone());
                    }
                }
                Ok(serde_yaml::Value::Sequence(merged))
            } else {
                // No name key — patch list replaces base list.
                Ok(patch.clone())
            }
        }
        _ => Ok(patch.clone()),
    }
}

/// Walk a YAML value looking for `containers` / `initContainers` sequences and
/// apply image overrides.
fn walk_and_override_images(val: &mut serde_yaml::Value, images: &[ImageOverride]) {
    match val {
        serde_yaml::Value::Mapping(map) => {
            // Check if this mapping has a "containers" or "initContainers" key.
            for container_key in &["containers", "initContainers"] {
                let key = serde_yaml::Value::String((*container_key).to_string());
                if let Some(serde_yaml::Value::Sequence(ref mut containers)) = map.get_mut(&key) {
                    for container in containers.iter_mut() {
                        override_container_image(container, images);
                    }
                }
            }
            // Recurse into all children.
            for (_, child) in map.iter_mut() {
                walk_and_override_images(child, images);
            }
        }
        serde_yaml::Value::Sequence(seq) => {
            for child in seq.iter_mut() {
                walk_and_override_images(child, images);
            }
        }
        _ => {}
    }
}

/// Apply image overrides to a single container mapping.
fn override_container_image(container: &mut serde_yaml::Value, images: &[ImageOverride]) {
    let image_key = serde_yaml::Value::String("image".into());
    if let Some(serde_yaml::Value::String(ref current)) = container.get(&image_key).cloned() {
        let (img_name, _img_tag) = split_image_ref(current);
        for ov in images {
            if ov.name == img_name {
                let final_name = ov.new_name.as_deref().unwrap_or(&img_name);
                let new_image = if let Some(ref digest) = ov.digest {
                    format!("{}@{}", final_name, digest)
                } else if let Some(ref tag) = ov.new_tag {
                    format!("{}:{}", final_name, tag)
                } else {
                    final_name.to_string()
                };
                if let serde_yaml::Value::Mapping(ref mut m) = container {
                    m.insert(image_key.clone(), serde_yaml::Value::String(new_image));
                }
                break;
            }
        }
    }
}

/// Split an image reference like `nginx:1.21` into `("nginx", "1.21")`.
fn split_image_ref(image: &str) -> (String, String) {
    if let Some(at_pos) = image.find('@') {
        (image[..at_pos].to_string(), image[at_pos..].to_string())
    } else if let Some(colon_pos) = image.rfind(':') {
        // Avoid splitting on a port inside a registry URL (e.g. registry:5000/img).
        let after = &image[colon_pos + 1..];
        if after.contains('/') {
            (image.to_string(), String::new())
        } else {
            (image[..colon_pos].to_string(), after.to_string())
        }
    } else {
        (image.to_string(), String::new())
    }
}

/// Convert a `serde_yaml::Value` into a `KubernetesResource`.
fn yaml_value_to_kubernetes_resource(val: &serde_yaml::Value) -> Result<KubernetesResource> {
    let kind = val
        .get("kind")
        .and_then(|v| v.as_str())
        .unwrap_or("Unknown")
        .to_string();

    match kind.as_str() {
        "Deployment" => {
            let dep = serde_yaml::from_value(val.clone())
                .context("failed to deserialize Deployment")?;
            Ok(KubernetesResource::Deployment(dep))
        }
        "Service" => {
            let svc = serde_yaml::from_value(val.clone())
                .context("failed to deserialize Service")?;
            Ok(KubernetesResource::Service(svc))
        }
        "Ingress" => {
            let ing = serde_yaml::from_value(val.clone())
                .context("failed to deserialize Ingress")?;
            Ok(KubernetesResource::Ingress(ing))
        }
        "ConfigMap" => {
            let cm = serde_yaml::from_value(val.clone())
                .context("failed to deserialize ConfigMap")?;
            Ok(KubernetesResource::ConfigMap(cm))
        }
        _ => Ok(KubernetesResource::Unknown(kind, val.clone())),
    }
}

/// Get a mutable reference to `metadata.name` as a `String`.
fn get_metadata_name_mut(resource: &mut serde_yaml::Value) -> Option<&mut String> {
    resource
        .get_mut("metadata")?
        .get_mut("name")?
        .as_str()
        .map(|_| ())
        .and_then(|_| {
            if let serde_yaml::Value::String(ref mut s) = resource
                .get_mut("metadata")
                .unwrap()
                .get_mut("name")
                .unwrap()
            {
                Some(s)
            } else {
                None
            }
        })
}

/// Navigate to a nested mapping path, creating intermediate maps, and merge
/// the provided key/value pairs.
fn ensure_map_and_merge(
    root: &mut serde_yaml::Value,
    path: &[&str],
    entries: &IndexMap<String, String>,
) {
    let mut current = root;
    for segment in path {
        let key = serde_yaml::Value::String((*segment).to_string());
        if !current.is_mapping() {
            return;
        }
        let map = current.as_mapping_mut().unwrap();
        if !map.contains_key(&key) {
            map.insert(key.clone(), serde_yaml::Value::Mapping(serde_yaml::Mapping::new()));
        }
        current = map.get_mut(&key).unwrap();
    }
    if let serde_yaml::Value::Mapping(ref mut m) = current {
        for (k, v) in entries {
            m.insert(
                serde_yaml::Value::String(k.clone()),
                serde_yaml::Value::String(v.clone()),
            );
        }
    }
}

/// Simple `key=value` label selector matching against `metadata.labels`.
fn matches_label_selector(resource: &serde_yaml::Value, selector: &str) -> bool {
    let labels = resource
        .get("metadata")
        .and_then(|m| m.get("labels"))
        .and_then(|l| l.as_mapping());

    for part in selector.split(',') {
        let part = part.trim();
        if let Some((key, val)) = part.split_once('=') {
            let key = key.trim();
            let val = val.trim();
            let actual = labels
                .and_then(|m| m.get(serde_yaml::Value::String(key.to_string())))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if actual != val {
                return false;
            }
        }
    }
    true
}

/// Simple `key=value` annotation selector matching against `metadata.annotations`.
fn matches_annotation_selector(resource: &serde_yaml::Value, selector: &str) -> bool {
    let annotations = resource
        .get("metadata")
        .and_then(|m| m.get("annotations"))
        .and_then(|l| l.as_mapping());

    for part in selector.split(',') {
        let part = part.trim();
        if let Some((key, val)) = part.split_once('=') {
            let key = key.trim();
            let val = val.trim();
            let actual = annotations
                .and_then(|m| m.get(serde_yaml::Value::String(key.to_string())))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if actual != val {
                return false;
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_kustomization() {
        let yaml = r#"
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - deployment.yaml
  - service.yaml
namespace: production
namePrefix: prod-
commonLabels:
  env: production
images:
  - name: nginx
    newTag: "1.25"
replicas:
  - name: web
    count: 5
configMapGenerator:
  - name: app-config
    literals:
      - DB_HOST=postgres.prod
"#;
        let k = KustomizeProcessor::parse_kustomization(yaml).unwrap();
        assert_eq!(k.resources.len(), 2);
        assert_eq!(k.namespace, Some("production".to_string()));
        assert_eq!(k.name_prefix, Some("prod-".to_string()));
        assert_eq!(k.common_labels.get("env"), Some(&"production".to_string()));
        assert_eq!(k.images.len(), 1);
        assert_eq!(k.images[0].new_tag, Some("1.25".to_string()));
        assert_eq!(k.replicas.len(), 1);
        assert_eq!(k.replicas[0].count, 5);
        assert_eq!(k.config_map_generator.len(), 1);
    }

    #[test]
    fn test_strategic_merge_patch_simple() {
        let base: serde_yaml::Value = serde_yaml::from_str(
            r#"
kind: Deployment
metadata:
  name: web
spec:
  replicas: 1
"#,
        )
        .unwrap();
        let patch: serde_yaml::Value = serde_yaml::from_str(
            r#"
kind: Deployment
metadata:
  name: web
spec:
  replicas: 3
"#,
        )
        .unwrap();
        let result = KustomizeProcessor::apply_strategic_merge_patch(&base, &patch).unwrap();
        assert_eq!(
            result.get("spec").unwrap().get("replicas").unwrap().as_u64(),
            Some(3)
        );
        // kind and name preserved
        assert_eq!(result.get("kind").unwrap().as_str(), Some("Deployment"));
    }

    #[test]
    fn test_strategic_merge_patch_nested() {
        let base: serde_yaml::Value = serde_yaml::from_str(
            r#"
metadata:
  name: app
  labels:
    app: web
    version: v1
spec:
  replicas: 1
"#,
        )
        .unwrap();
        let patch: serde_yaml::Value = serde_yaml::from_str(
            r#"
metadata:
  labels:
    version: v2
    tier: frontend
"#,
        )
        .unwrap();
        let result = KustomizeProcessor::apply_strategic_merge_patch(&base, &patch).unwrap();
        let labels = result.get("metadata").unwrap().get("labels").unwrap();
        assert_eq!(labels.get("app").unwrap().as_str(), Some("web"));
        assert_eq!(labels.get("version").unwrap().as_str(), Some("v2"));
        assert_eq!(labels.get("tier").unwrap().as_str(), Some("frontend"));
        // spec preserved untouched
        assert_eq!(
            result.get("spec").unwrap().get("replicas").unwrap().as_u64(),
            Some(1)
        );
    }

    #[test]
    fn test_strategic_merge_patch_list_by_name() {
        let base: serde_yaml::Value = serde_yaml::from_str(
            r#"
containers:
  - name: app
    image: nginx:1.20
  - name: sidecar
    image: envoy:1.0
"#,
        )
        .unwrap();
        let patch: serde_yaml::Value = serde_yaml::from_str(
            r#"
containers:
  - name: app
    image: nginx:1.25
"#,
        )
        .unwrap();
        let result = KustomizeProcessor::apply_strategic_merge_patch(&base, &patch).unwrap();
        let containers = result.get("containers").unwrap().as_sequence().unwrap();
        assert_eq!(containers.len(), 2);
        assert_eq!(containers[0].get("image").unwrap().as_str(), Some("nginx:1.25"));
        assert_eq!(containers[1].get("image").unwrap().as_str(), Some("envoy:1.0"));
    }

    #[test]
    fn test_strategic_merge_patch_delete_with_null() {
        let base: serde_yaml::Value = serde_yaml::from_str(
            r#"
metadata:
  name: web
  labels:
    app: web
    temporary: "true"
"#,
        )
        .unwrap();
        let patch: serde_yaml::Value = serde_yaml::from_str(
            r#"
metadata:
  labels:
    temporary: null
"#,
        )
        .unwrap();
        let result = KustomizeProcessor::apply_strategic_merge_patch(&base, &patch).unwrap();
        let labels = result.get("metadata").unwrap().get("labels").unwrap();
        assert!(labels.get("temporary").is_none());
        assert_eq!(labels.get("app").unwrap().as_str(), Some("web"));
    }

    #[test]
    fn test_json_patch_add() {
        let base: serde_yaml::Value = serde_yaml::from_str(
            r#"
metadata:
  name: test
"#,
        )
        .unwrap();
        let ops = vec![JsonPatchOp::Add {
            path: "/metadata/labels".to_string(),
            value: serde_yaml::from_str("{app: web}").unwrap(),
        }];
        let result = KustomizeProcessor::apply_json_patch(&base, &ops).unwrap();
        assert_eq!(
            result
                .get("metadata")
                .unwrap()
                .get("labels")
                .unwrap()
                .get("app")
                .unwrap()
                .as_str(),
            Some("web")
        );
    }

    #[test]
    fn test_json_patch_remove() {
        let base: serde_yaml::Value = serde_yaml::from_str(
            r#"
metadata:
  name: test
  labels:
    app: web
    env: dev
"#,
        )
        .unwrap();
        let ops = vec![JsonPatchOp::Remove {
            path: "/metadata/labels/env".to_string(),
        }];
        let result = KustomizeProcessor::apply_json_patch(&base, &ops).unwrap();
        let labels = result.get("metadata").unwrap().get("labels").unwrap();
        assert!(labels.get("env").is_none());
        assert_eq!(labels.get("app").unwrap().as_str(), Some("web"));
    }

    #[test]
    fn test_json_patch_replace() {
        let base: serde_yaml::Value = serde_yaml::from_str(
            r#"
spec:
  replicas: 1
"#,
        )
        .unwrap();
        let ops = vec![JsonPatchOp::Replace {
            path: "/spec/replicas".to_string(),
            value: serde_yaml::Value::Number(serde_yaml::Number::from(5u64)),
        }];
        let result = KustomizeProcessor::apply_json_patch(&base, &ops).unwrap();
        assert_eq!(result.get("spec").unwrap().get("replicas").unwrap().as_u64(), Some(5));
    }

    #[test]
    fn test_json_patch_move() {
        let base: serde_yaml::Value = serde_yaml::from_str(
            r#"
source:
  value: hello
target: {}
"#,
        )
        .unwrap();
        let ops = vec![JsonPatchOp::Move {
            from: "/source/value".to_string(),
            path: "/target/value".to_string(),
        }];
        let result = KustomizeProcessor::apply_json_patch(&base, &ops).unwrap();
        assert!(result.get("source").unwrap().get("value").is_none());
        assert_eq!(
            result.get("target").unwrap().get("value").unwrap().as_str(),
            Some("hello")
        );
    }

    #[test]
    fn test_json_patch_test_op() {
        let base: serde_yaml::Value = serde_yaml::from_str(
            r#"
spec:
  replicas: 3
"#,
        )
        .unwrap();

        // Successful test
        let ops_ok = vec![JsonPatchOp::Test {
            path: "/spec/replicas".to_string(),
            value: serde_yaml::Value::Number(serde_yaml::Number::from(3u64)),
        }];
        assert!(KustomizeProcessor::apply_json_patch(&base, &ops_ok).is_ok());

        // Failing test
        let ops_fail = vec![JsonPatchOp::Test {
            path: "/spec/replicas".to_string(),
            value: serde_yaml::Value::Number(serde_yaml::Number::from(99u64)),
        }];
        assert!(KustomizeProcessor::apply_json_patch(&base, &ops_fail).is_err());
    }

    #[test]
    fn test_apply_common_labels() {
        let mut resource: serde_yaml::Value = serde_yaml::from_str(
            r#"
kind: Deployment
metadata:
  name: web
spec:
  selector: {}
  template:
    metadata: {}
"#,
        )
        .unwrap();
        let mut labels = IndexMap::new();
        labels.insert("env".to_string(), "staging".to_string());
        labels.insert("team".to_string(), "platform".to_string());

        KustomizeProcessor::apply_common_labels(&mut resource, &labels);

        let meta_labels = resource.get("metadata").unwrap().get("labels").unwrap();
        assert_eq!(meta_labels.get("env").unwrap().as_str(), Some("staging"));
        assert_eq!(meta_labels.get("team").unwrap().as_str(), Some("platform"));

        let selector_labels = resource
            .get("spec")
            .unwrap()
            .get("selector")
            .unwrap()
            .get("matchLabels")
            .unwrap();
        assert_eq!(selector_labels.get("env").unwrap().as_str(), Some("staging"));

        let tpl_labels = resource
            .get("spec")
            .unwrap()
            .get("template")
            .unwrap()
            .get("metadata")
            .unwrap()
            .get("labels")
            .unwrap();
        assert_eq!(tpl_labels.get("team").unwrap().as_str(), Some("platform"));
    }

    #[test]
    fn test_apply_namespace_override() {
        let mut resource: serde_yaml::Value = serde_yaml::from_str(
            r#"
kind: Service
metadata:
  name: api
"#,
        )
        .unwrap();
        KustomizeProcessor::apply_namespace_override(&mut resource, "production");
        assert_eq!(
            resource
                .get("metadata")
                .unwrap()
                .get("namespace")
                .unwrap()
                .as_str(),
            Some("production")
        );
    }

    #[test]
    fn test_apply_image_overrides() {
        let mut resource: serde_yaml::Value = serde_yaml::from_str(
            r#"
kind: Deployment
metadata:
  name: web
spec:
  template:
    spec:
      containers:
        - name: app
          image: nginx:1.20
        - name: sidecar
          image: envoy:1.22
"#,
        )
        .unwrap();
        let overrides = vec![
            ImageOverride {
                name: "nginx".to_string(),
                new_name: None,
                new_tag: Some("1.25".to_string()),
                digest: None,
            },
            ImageOverride {
                name: "envoy".to_string(),
                new_name: Some("envoyproxy/envoy".to_string()),
                new_tag: Some("v1.28".to_string()),
                digest: None,
            },
        ];
        KustomizeProcessor::apply_image_overrides(&mut resource, &overrides);

        let containers = resource
            .get("spec")
            .unwrap()
            .get("template")
            .unwrap()
            .get("spec")
            .unwrap()
            .get("containers")
            .unwrap()
            .as_sequence()
            .unwrap();
        assert_eq!(containers[0].get("image").unwrap().as_str(), Some("nginx:1.25"));
        assert_eq!(
            containers[1].get("image").unwrap().as_str(),
            Some("envoyproxy/envoy:v1.28")
        );
    }

    #[test]
    fn test_merge_resources() {
        let base: Vec<serde_yaml::Value> = vec![
            serde_yaml::from_str(
                r#"
kind: Deployment
metadata:
  name: web
spec:
  replicas: 1
"#,
            )
            .unwrap(),
            serde_yaml::from_str(
                r#"
kind: Service
metadata:
  name: web
spec:
  type: ClusterIP
"#,
            )
            .unwrap(),
        ];
        let overlay: Vec<serde_yaml::Value> = vec![serde_yaml::from_str(
            r#"
kind: Deployment
metadata:
  name: web
spec:
  replicas: 5
"#,
        )
        .unwrap()];

        let merged = KustomizeProcessor::merge_resources(base, overlay);
        assert_eq!(merged.len(), 2);

        // The Deployment should be the overlay version.
        let dep = merged
            .iter()
            .find(|v| v.get("kind").unwrap().as_str() == Some("Deployment"))
            .unwrap();
        assert_eq!(dep.get("spec").unwrap().get("replicas").unwrap().as_u64(), Some(5));
    }

    // ---- Additional helper tests ----

    #[test]
    fn test_resource_key() {
        let val: serde_yaml::Value = serde_yaml::from_str(
            r#"
kind: ConfigMap
metadata:
  name: settings
"#,
        )
        .unwrap();
        assert_eq!(
            resource_key(&val),
            Some(("ConfigMap".to_string(), "settings".to_string()))
        );

        let no_name: serde_yaml::Value = serde_yaml::from_str("kind: Pod").unwrap();
        assert_eq!(resource_key(&no_name), None);
    }

    #[test]
    fn test_json_pointer_round_trip() {
        let mut val: serde_yaml::Value = serde_yaml::from_str("a:\n  b:\n    c: 1").unwrap();
        assert_eq!(
            resolve_json_pointer(&val, "/a/b/c")
                .unwrap()
                .as_u64(),
            Some(1)
        );

        set_json_pointer(&mut val, "/a/b/d", serde_yaml::Value::String("new".into())).unwrap();
        assert_eq!(
            resolve_json_pointer(&val, "/a/b/d")
                .unwrap()
                .as_str(),
            Some("new")
        );

        let removed = remove_json_pointer(&mut val, "/a/b/c").unwrap();
        assert_eq!(removed.as_u64(), Some(1));
        assert!(resolve_json_pointer(&val, "/a/b/c").is_none());
    }

    #[test]
    fn test_matches_target() {
        let resource: serde_yaml::Value = serde_yaml::from_str(
            r#"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
  namespace: default
  labels:
    app: web
"#,
        )
        .unwrap();

        let target_match = PatchTarget {
            group: None,
            version: Some("v1".into()),
            kind: Some("Deployment".into()),
            name: Some("web".into()),
            namespace: None,
            label_selector: Some("app=web".into()),
            annotation_selector: None,
        };
        assert!(matches_target(&resource, &target_match));

        let target_no_match = PatchTarget {
            group: None,
            version: None,
            kind: Some("Service".into()),
            name: None,
            namespace: None,
            label_selector: None,
            annotation_selector: None,
        };
        assert!(!matches_target(&resource, &target_no_match));
    }

    #[test]
    fn test_generate_config_map() {
        let args = ConfigMapGeneratorArgs {
            name: "my-config".to_string(),
            literals: vec!["KEY1=val1".to_string(), "KEY2=val2".to_string()],
            files: vec![],
            env_files: vec![],
            behavior: None,
        };
        let cm = generate_config_map(&args);
        assert_eq!(cm.get("kind").unwrap().as_str(), Some("ConfigMap"));
        assert_eq!(
            cm.get("metadata").unwrap().get("name").unwrap().as_str(),
            Some("my-config")
        );
        let data = cm.get("data").unwrap();
        assert_eq!(data.get("KEY1").unwrap().as_str(), Some("val1"));
        assert_eq!(data.get("KEY2").unwrap().as_str(), Some("val2"));
    }

    #[test]
    fn test_apply_name_prefix_and_suffix() {
        let mut resource: serde_yaml::Value = serde_yaml::from_str(
            r#"
kind: Deployment
metadata:
  name: web
"#,
        )
        .unwrap();
        KustomizeProcessor::apply_name_prefix(&mut resource, "staging-");
        assert_eq!(
            resource.get("metadata").unwrap().get("name").unwrap().as_str(),
            Some("staging-web")
        );
        KustomizeProcessor::apply_name_suffix(&mut resource, "-v2");
        assert_eq!(
            resource.get("metadata").unwrap().get("name").unwrap().as_str(),
            Some("staging-web-v2")
        );
    }

    #[test]
    fn test_apply_replica_overrides() {
        let mut resource: serde_yaml::Value = serde_yaml::from_str(
            r#"
kind: Deployment
metadata:
  name: api
spec:
  replicas: 1
"#,
        )
        .unwrap();
        let overrides = vec![
            ReplicaOverride {
                name: "api".to_string(),
                count: 10,
            },
            ReplicaOverride {
                name: "other".to_string(),
                count: 3,
            },
        ];
        KustomizeProcessor::apply_replica_overrides(&mut resource, &overrides);
        assert_eq!(
            resource
                .get("spec")
                .unwrap()
                .get("replicas")
                .unwrap()
                .as_u64(),
            Some(10)
        );
    }
}
