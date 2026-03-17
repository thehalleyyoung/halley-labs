//! Kustomize overlay parsing and resolution.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use safestep_types::SafeStepError;

use crate::manifest::{KubernetesManifest, ManifestMetadata};

pub type Result<T> = std::result::Result<T, SafeStepError>;

// ---------------------------------------------------------------------------
// Core kustomization types
// ---------------------------------------------------------------------------

/// A parsed kustomization.yaml.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Kustomization {
    pub resources: Vec<String>,
    pub patches: Vec<KustomizePatch>,
    pub config_map_generator: Vec<ConfigMapGenerator>,
    pub secret_generator: Vec<SecretGenerator>,
    pub namespace: Option<String>,
    pub name_prefix: Option<String>,
    pub name_suffix: Option<String>,
    pub common_labels: HashMap<String, String>,
    pub common_annotations: HashMap<String, String>,
    pub images: Vec<ImageOverride>,
    pub bases: Vec<String>,
    pub generators: Vec<String>,
    pub transformers: Vec<String>,
    #[serde(default)]
    pub crds: Vec<String>,
    #[serde(default)]
    pub configurations: Vec<String>,
}

impl Kustomization {
    /// Parse a kustomization.yaml string.
    pub fn parse(yaml: &str) -> Result<Self> {
        let value: Value = serde_yaml::from_str(yaml).map_err(|e| SafeStepError::K8sError {
            message: format!("Failed to parse kustomization.yaml: {e}"),
            resource: None,
            namespace: None,
            context: None,
        })?;
        Self::from_value(&value)
    }

    /// Build from a serde_json::Value.
    pub fn from_value(v: &Value) -> Result<Self> {
        let resources = parse_string_array(v.get("resources"));
        let bases = parse_string_array(v.get("bases"));
        let namespace = v.get("namespace").and_then(|n| n.as_str()).map(String::from);
        let name_prefix = v.get("namePrefix").and_then(|n| n.as_str()).map(String::from);
        let name_suffix = v.get("nameSuffix").and_then(|n| n.as_str()).map(String::from);
        let common_labels = parse_string_map(v.get("commonLabels"));
        let common_annotations = parse_string_map(v.get("commonAnnotations"));

        let images = v
            .get("images")
            .and_then(|i| i.as_array())
            .map(|arr| arr.iter().filter_map(|i| ImageOverride::from_value(i)).collect())
            .unwrap_or_default();

        let patches = parse_patches(v)?;
        let config_map_generator = parse_config_map_generators(v);
        let secret_generator = parse_secret_generators(v);
        let generators = parse_string_array(v.get("generators"));
        let transformers = parse_string_array(v.get("transformers"));
        let crds = parse_string_array(v.get("crds"));
        let configurations = parse_string_array(v.get("configurations"));

        Ok(Self {
            resources,
            patches,
            config_map_generator,
            secret_generator,
            namespace,
            name_prefix,
            name_suffix,
            common_labels,
            common_annotations,
            images,
            bases,
            generators,
            transformers,
            crds,
            configurations,
        })
    }
}

// ---------------------------------------------------------------------------
// Patch types
// ---------------------------------------------------------------------------

/// A Kustomize patch specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KustomizePatch {
    StrategicMerge(String),
    JsonPatch {
        target: PatchTarget,
        operations: Vec<JsonPatchOperation>,
    },
    Inline {
        target: Option<PatchTarget>,
        patch: String,
    },
}

/// Identifies a target resource for a patch.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PatchTarget {
    pub group: Option<String>,
    pub version: Option<String>,
    pub kind: Option<String>,
    pub name: Option<String>,
    pub namespace: Option<String>,
    pub label_selector: Option<String>,
    pub annotation_selector: Option<String>,
}

impl PatchTarget {
    /// Check if a manifest matches this target.
    pub fn matches(&self, manifest: &KubernetesManifest) -> bool {
        if let Some(kind) = &self.kind {
            if kind != &manifest.kind {
                return false;
            }
        }
        if let Some(name) = &self.name {
            if name != &manifest.metadata.name {
                return false;
            }
        }
        if let Some(ns) = &self.namespace {
            if manifest.metadata.namespace.as_deref() != Some(ns.as_str()) {
                return false;
            }
        }
        if let Some(group) = &self.group {
            let api_group = manifest
                .api_version
                .split('/')
                .next()
                .unwrap_or("");
            if group != api_group && !(group.is_empty() && !manifest.api_version.contains('/')) {
                return false;
            }
        }
        if let Some(version) = &self.version {
            let api_version = if manifest.api_version.contains('/') {
                manifest.api_version.split('/').nth(1).unwrap_or("")
            } else {
                &manifest.api_version
            };
            if version != api_version {
                return false;
            }
        }
        true
    }
}

fn parse_patches(v: &Value) -> Result<Vec<KustomizePatch>> {
    let mut patches = Vec::new();

    // patchesStrategicMerge (deprecated but still used)
    if let Some(arr) = v.get("patchesStrategicMerge").and_then(|a| a.as_array()) {
        for item in arr {
            if let Some(s) = item.as_str() {
                patches.push(KustomizePatch::StrategicMerge(s.to_string()));
            }
        }
    }

    // patchesJson6902 (deprecated but still used)
    if let Some(arr) = v.get("patchesJson6902").and_then(|a| a.as_array()) {
        for item in arr {
            let target = parse_patch_target(item.get("target"));
            let ops = item
                .get("patch")
                .and_then(|p| p.as_str())
                .and_then(|s| parse_json_patch_string(s).ok())
                .unwrap_or_default();
            patches.push(KustomizePatch::JsonPatch {
                target,
                operations: ops,
            });
        }
    }

    // patches (unified field)
    if let Some(arr) = v.get("patches").and_then(|a| a.as_array()) {
        for item in arr {
            if let Some(s) = item.as_str() {
                patches.push(KustomizePatch::StrategicMerge(s.to_string()));
            } else if let Some(obj) = item.as_object() {
                let target = parse_patch_target(obj.get("target").map(|v| v));
                if let Some(patch_str) = obj.get("patch").and_then(|p| p.as_str()) {
                    patches.push(KustomizePatch::Inline {
                        target: Some(target),
                        patch: patch_str.to_string(),
                    });
                } else if let Some(path_str) = obj.get("path").and_then(|p| p.as_str()) {
                    patches.push(KustomizePatch::Inline {
                        target: Some(target),
                        patch: path_str.to_string(),
                    });
                }
            }
        }
    }

    Ok(patches)
}

fn parse_patch_target(v: Option<&Value>) -> PatchTarget {
    let Some(v) = v else { return PatchTarget::default() };
    PatchTarget {
        group: v.get("group").and_then(|s| s.as_str()).map(String::from),
        version: v.get("version").and_then(|s| s.as_str()).map(String::from),
        kind: v.get("kind").and_then(|s| s.as_str()).map(String::from),
        name: v.get("name").and_then(|s| s.as_str()).map(String::from),
        namespace: v.get("namespace").and_then(|s| s.as_str()).map(String::from),
        label_selector: v.get("labelSelector").and_then(|s| s.as_str()).map(String::from),
        annotation_selector: v
            .get("annotationSelector")
            .and_then(|s| s.as_str())
            .map(String::from),
    }
}

// ---------------------------------------------------------------------------
// Strategic merge patch
// ---------------------------------------------------------------------------

/// Kubernetes strategic merge patch implementation.
pub struct StrategicMergePatch;

impl StrategicMergePatch {
    /// Apply a strategic merge patch to a base document.
    pub fn apply(base: &Value, patch: &Value) -> Result<Value> {
        strategic_merge(base, patch)
    }
}

fn strategic_merge(base: &Value, patch: &Value) -> Result<Value> {
    match (base, patch) {
        (Value::Object(base_map), Value::Object(patch_map)) => {
            let mut result = base_map.clone();

            for (key, patch_val) in patch_map {
                // Check for $patch directive
                if key == "$patch" {
                    if let Some(directive) = patch_val.as_str() {
                        match directive {
                            "delete" => return Ok(Value::Null),
                            "replace" => {
                                let mut replacement = patch_map.clone();
                                replacement.remove("$patch");
                                return Ok(Value::Object(replacement));
                            }
                            _ => {}
                        }
                    }
                    continue;
                }

                if let Some(base_val) = result.get(key) {
                    // Check if the patch value contains $patch: delete for this field
                    if patch_val.is_null() {
                        result.remove(key);
                    } else if let (Value::Array(base_arr), Value::Array(patch_arr)) =
                        (base_val, patch_val)
                    {
                        // Strategic merge for arrays: use merge key if present
                        let merged = strategic_merge_arrays(base_arr, patch_arr)?;
                        result.insert(key.clone(), Value::Array(merged));
                    } else if base_val.is_object() && patch_val.is_object() {
                        let merged = strategic_merge(base_val, patch_val)?;
                        result.insert(key.clone(), merged);
                    } else {
                        result.insert(key.clone(), patch_val.clone());
                    }
                } else {
                    result.insert(key.clone(), patch_val.clone());
                }
            }

            Ok(Value::Object(result))
        }
        (_, patch) => Ok(patch.clone()),
    }
}

fn strategic_merge_arrays(base: &[Value], patch: &[Value]) -> Result<Vec<Value>> {
    // Check if items have a merge key (commonly "name" for containers)
    let merge_key = detect_merge_key(base).or_else(|| detect_merge_key(patch));

    match merge_key {
        Some(key) => {
            let mut result: Vec<Value> = base.to_vec();
            for patch_item in patch {
                if let Some(patch_key_val) = patch_item.get(&key) {
                    // Check for $patch: delete on individual items
                    if patch_item.get("$patch").and_then(|v| v.as_str()) == Some("delete") {
                        result.retain(|item| item.get(&key) != Some(patch_key_val));
                        continue;
                    }
                    // Find matching base item and merge
                    let mut found = false;
                    for base_item in result.iter_mut() {
                        if base_item.get(&key) == Some(patch_key_val) {
                            *base_item = strategic_merge(base_item, patch_item)?;
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        result.push(patch_item.clone());
                    }
                } else {
                    result.push(patch_item.clone());
                }
            }
            Ok(result)
        }
        None => {
            // No merge key: replace the entire array
            Ok(patch.to_vec())
        }
    }
}

fn detect_merge_key(items: &[Value]) -> Option<String> {
    // Common Kubernetes merge keys
    let candidates = ["name", "containerPort", "mountPath", "ip"];
    for key in &candidates {
        if items
            .iter()
            .all(|item| item.get(*key).is_some() || item.get("$patch").is_some())
        {
            return Some(key.to_string());
        }
    }
    None
}

// ---------------------------------------------------------------------------
// JSON Patch (RFC 6902)
// ---------------------------------------------------------------------------

/// A single JSON Patch operation (RFC 6902).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JsonPatchOperation {
    Add { path: String, value: Value },
    Remove { path: String },
    Replace { path: String, value: Value },
    Move { from: String, path: String },
    Copy { from: String, path: String },
    Test { path: String, value: Value },
}

impl JsonPatchOperation {
    /// Apply this operation to a JSON document.
    pub fn apply(&self, doc: &mut Value) -> Result<()> {
        match self {
            JsonPatchOperation::Add { path, value } => {
                json_patch_add(doc, path, value.clone())
            }
            JsonPatchOperation::Remove { path } => {
                json_patch_remove(doc, path)
            }
            JsonPatchOperation::Replace { path, value } => {
                json_patch_replace(doc, path, value.clone())
            }
            JsonPatchOperation::Move { from, path } => {
                let val = json_patch_get(doc, from)?;
                json_patch_remove(doc, from)?;
                json_patch_add(doc, path, val)
            }
            JsonPatchOperation::Copy { from, path } => {
                let val = json_patch_get(doc, from)?;
                json_patch_add(doc, path, val)
            }
            JsonPatchOperation::Test { path, value } => {
                let actual = json_patch_get(doc, path)?;
                if actual != *value {
                    return Err(SafeStepError::K8sError {
                        message: format!(
                            "JSON Patch test failed at {path}: expected {value}, got {actual}"
                        ),
                        resource: None,
                        namespace: None,
                        context: None,
                    });
                }
                Ok(())
            }
        }
    }

    /// Apply a sequence of operations to a document.
    pub fn apply_all(ops: &[JsonPatchOperation], doc: &mut Value) -> Result<()> {
        for op in ops {
            op.apply(doc)?;
        }
        Ok(())
    }
}

fn parse_json_path(path: &str) -> Vec<String> {
    if path.is_empty() || path == "/" {
        return Vec::new();
    }
    path.trim_start_matches('/')
        .split('/')
        .map(|s| s.replace("~1", "/").replace("~0", "~"))
        .collect()
}

fn json_patch_get(doc: &Value, path: &str) -> Result<Value> {
    let segments = parse_json_path(path);
    let mut current = doc;
    for seg in &segments {
        current = match current {
            Value::Object(map) => map.get(seg.as_str()).ok_or_else(|| SafeStepError::K8sError {
                message: format!("Path not found: {path} (missing key: {seg})"),
                resource: None,
                namespace: None,
                context: None,
            })?,
            Value::Array(arr) => {
                let idx: usize = seg.parse().map_err(|_| SafeStepError::K8sError {
                    message: format!("Invalid array index: {seg} in path {path}"),
                    resource: None,
                    namespace: None,
                    context: None,
                })?;
                arr.get(idx).ok_or_else(|| SafeStepError::K8sError {
                    message: format!("Array index {idx} out of bounds at {path}"),
                    resource: None,
                    namespace: None,
                    context: None,
                })?
            }
            _ => {
                return Err(SafeStepError::K8sError {
                    message: format!("Cannot traverse non-container at {path}"),
                    resource: None,
                    namespace: None,
                    context: None,
                });
            }
        };
    }
    Ok(current.clone())
}

fn json_patch_add(doc: &mut Value, path: &str, value: Value) -> Result<()> {
    if path.is_empty() || path == "/" {
        *doc = value;
        return Ok(());
    }

    let segments = parse_json_path(path);
    let (parent_segments, last) = segments.split_at(segments.len() - 1);
    let last_key = &last[0];

    let mut current = doc;
    for seg in parent_segments {
        current = match current {
            Value::Object(map) => map
                .entry(seg.clone())
                .or_insert(Value::Object(serde_json::Map::new())),
            Value::Array(arr) => {
                let idx: usize = seg.parse().map_err(|_| SafeStepError::K8sError {
                    message: format!("Invalid array index in add path: {path}"),
                    resource: None,
                    namespace: None,
                    context: None,
                })?;
                arr.get_mut(idx).ok_or_else(|| SafeStepError::K8sError {
                    message: format!("Array index {idx} out of bounds"),
                    resource: None,
                    namespace: None,
                    context: None,
                })?
            }
            _ => {
                return Err(SafeStepError::K8sError {
                    message: format!("Cannot add to non-container at {path}"),
                    resource: None,
                    namespace: None,
                    context: None,
                });
            }
        };
    }

    match current {
        Value::Object(map) => {
            map.insert(last_key.clone(), value);
            Ok(())
        }
        Value::Array(arr) => {
            if last_key == "-" {
                arr.push(value);
            } else {
                let idx: usize = last_key.parse().map_err(|_| SafeStepError::K8sError {
                    message: format!("Invalid array index: {last_key}"),
                    resource: None,
                    namespace: None,
                    context: None,
                })?;
                if idx > arr.len() {
                    return Err(SafeStepError::K8sError {
                        message: format!("Array index {idx} out of bounds (len={})", arr.len()),
                        resource: None,
                        namespace: None,
                        context: None,
                    });
                }
                arr.insert(idx, value);
            }
            Ok(())
        }
        _ => Err(SafeStepError::K8sError {
            message: format!("Cannot add to non-container at {path}"),
            resource: None,
            namespace: None,
            context: None,
        }),
    }
}

fn json_patch_remove(doc: &mut Value, path: &str) -> Result<()> {
    let segments = parse_json_path(path);
    if segments.is_empty() {
        return Err(SafeStepError::K8sError {
            message: "Cannot remove root document".into(),
            resource: None,
            namespace: None,
            context: None,
        });
    }

    let (parent_segments, last) = segments.split_at(segments.len() - 1);
    let last_key = &last[0];

    let mut current = doc;
    for seg in parent_segments {
        current = match current {
            Value::Object(map) => map.get_mut(seg.as_str()).ok_or_else(|| SafeStepError::K8sError {
                message: format!("Path not found during remove: {path}"),
                resource: None,
                namespace: None,
                context: None,
            })?,
            Value::Array(arr) => {
                let idx: usize = seg.parse().map_err(|_| SafeStepError::K8sError {
                    message: format!("Invalid array index: {seg}"),
                    resource: None,
                    namespace: None,
                    context: None,
                })?;
                arr.get_mut(idx).ok_or_else(|| SafeStepError::K8sError {
                    message: format!("Array index {idx} out of bounds"),
                    resource: None,
                    namespace: None,
                    context: None,
                })?
            }
            _ => {
                return Err(SafeStepError::K8sError {
                    message: format!("Cannot traverse at {path}"),
                    resource: None,
                    namespace: None,
                    context: None,
                });
            }
        };
    }

    match current {
        Value::Object(map) => {
            map.remove(last_key.as_str()).ok_or_else(|| SafeStepError::K8sError {
                message: format!("Key '{last_key}' not found for removal"),
                resource: None,
                namespace: None,
                context: None,
            })?;
            Ok(())
        }
        Value::Array(arr) => {
            let idx: usize = last_key.parse().map_err(|_| SafeStepError::K8sError {
                message: format!("Invalid array index: {last_key}"),
                resource: None,
                namespace: None,
                context: None,
            })?;
            if idx >= arr.len() {
                return Err(SafeStepError::K8sError {
                    message: format!("Array index {idx} out of bounds"),
                    resource: None,
                    namespace: None,
                    context: None,
                });
            }
            arr.remove(idx);
            Ok(())
        }
        _ => Err(SafeStepError::K8sError {
            message: format!("Cannot remove from non-container at {path}"),
            resource: None,
            namespace: None,
            context: None,
        }),
    }
}

fn json_patch_replace(doc: &mut Value, path: &str, value: Value) -> Result<()> {
    // Verify path exists, then set
    let _ = json_patch_get(doc, path)?;
    json_patch_remove(doc, path)?;
    json_patch_add(doc, path, value)
}

fn parse_json_patch_string(s: &str) -> Result<Vec<JsonPatchOperation>> {
    let arr: Vec<Value> = serde_yaml::from_str(s).map_err(|e| SafeStepError::K8sError {
        message: format!("Failed to parse JSON patch: {e}"),
        resource: None,
        namespace: None,
        context: None,
    })?;
    let mut ops = Vec::new();
    for item in &arr {
        let op_str = item.get("op").and_then(|o| o.as_str()).unwrap_or("");
        let path = item
            .get("path")
            .and_then(|p| p.as_str())
            .unwrap_or("")
            .to_string();
        let value = item.get("value").cloned().unwrap_or(Value::Null);
        let from = item
            .get("from")
            .and_then(|f| f.as_str())
            .unwrap_or("")
            .to_string();

        let op = match op_str {
            "add" => JsonPatchOperation::Add { path, value },
            "remove" => JsonPatchOperation::Remove { path },
            "replace" => JsonPatchOperation::Replace { path, value },
            "move" => JsonPatchOperation::Move { from, path },
            "copy" => JsonPatchOperation::Copy { from, path },
            "test" => JsonPatchOperation::Test { path, value },
            _ => continue,
        };
        ops.push(op);
    }
    Ok(ops)
}

// ---------------------------------------------------------------------------
// Kustomize resolver
// ---------------------------------------------------------------------------

/// Resolves a full kustomization, applying all transformations.
pub struct KustomizeResolver;

impl KustomizeResolver {
    /// Resolve a kustomization from a parsed Kustomization and its resource manifests.
    pub fn resolve(
        kustomization: &Kustomization,
        resource_manifests: &[KubernetesManifest],
    ) -> Result<Vec<KubernetesManifest>> {
        let mut manifests = resource_manifests.to_vec();

        // Apply namespace override
        if let Some(ns) = &kustomization.namespace {
            for m in &mut manifests {
                m.metadata.namespace = Some(ns.clone());
                if let Some(raw) = &mut m.raw {
                    if let Some(meta) = raw.get_mut("metadata") {
                        if let Value::Object(map) = meta {
                            map.insert("namespace".into(), Value::String(ns.clone()));
                        }
                    }
                }
            }
        }

        // Apply name prefix/suffix
        if let Some(prefix) = &kustomization.name_prefix {
            for m in &mut manifests {
                m.metadata.name = format!("{prefix}{}", m.metadata.name);
                update_raw_name(&mut m.raw, &m.metadata.name);
            }
        }
        if let Some(suffix) = &kustomization.name_suffix {
            for m in &mut manifests {
                m.metadata.name = format!("{}{suffix}", m.metadata.name);
                update_raw_name(&mut m.raw, &m.metadata.name);
            }
        }

        // Apply common labels
        if !kustomization.common_labels.is_empty() {
            for m in &mut manifests {
                for (k, v) in &kustomization.common_labels {
                    m.metadata.labels.insert(k.clone(), v.clone());
                }
                inject_common_labels_into_raw(&mut m.raw, &kustomization.common_labels);
            }
        }

        // Apply common annotations
        if !kustomization.common_annotations.is_empty() {
            for m in &mut manifests {
                for (k, v) in &kustomization.common_annotations {
                    m.metadata.annotations.insert(k.clone(), v.clone());
                }
                inject_common_annotations_into_raw(&mut m.raw, &kustomization.common_annotations);
            }
        }

        // Apply image overrides
        for img_override in &kustomization.images {
            for m in &mut manifests {
                img_override.apply(m);
            }
        }

        // Apply patches
        for patch in &kustomization.patches {
            match patch {
                KustomizePatch::StrategicMerge(patch_yaml) => {
                    let patch_val: Value =
                        serde_yaml::from_str(patch_yaml).map_err(|e| SafeStepError::K8sError {
                            message: format!("Failed to parse strategic merge patch: {e}"),
                            resource: None,
                            namespace: None,
                            context: None,
                        })?;
                    let patch_kind = patch_val.get("kind").and_then(|k| k.as_str());
                    let patch_name = patch_val
                        .get("metadata")
                        .and_then(|m| m.get("name"))
                        .and_then(|n| n.as_str());

                    for m in &mut manifests {
                        let kind_match = patch_kind.map(|k| k == m.kind).unwrap_or(true);
                        let name_match =
                            patch_name.map(|n| n == m.metadata.name).unwrap_or(true);
                        if kind_match && name_match {
                            let base_val = m.to_value();
                            let merged = StrategicMergePatch::apply(&base_val, &patch_val)?;
                            *m = KubernetesManifest::from_value(merged)?;
                        }
                    }
                }
                KustomizePatch::JsonPatch { target, operations } => {
                    for m in &mut manifests {
                        if target.matches(m) {
                            let mut val = m.to_value();
                            JsonPatchOperation::apply_all(operations, &mut val)?;
                            *m = KubernetesManifest::from_value(val)?;
                        }
                    }
                }
                KustomizePatch::Inline { target, patch: patch_str } => {
                    let patch_val: Value = serde_yaml::from_str(patch_str).unwrap_or(Value::Null);
                    if !patch_val.is_null() {
                        for m in &mut manifests {
                            let should_apply = target
                                .as_ref()
                                .map(|t| t.matches(m))
                                .unwrap_or(true);
                            if should_apply {
                                let base_val = m.to_value();
                                let merged = StrategicMergePatch::apply(&base_val, &patch_val)?;
                                *m = KubernetesManifest::from_value(merged)?;
                            }
                        }
                    }
                }
            }
        }

        // Add generated ConfigMaps
        for gen in &kustomization.config_map_generator {
            let cm = gen.generate()?;
            manifests.push(cm);
        }

        Ok(manifests)
    }
}

fn update_raw_name(raw: &mut Option<Value>, new_name: &str) {
    if let Some(raw) = raw {
        if let Some(meta) = raw.get_mut("metadata") {
            if let Value::Object(map) = meta {
                map.insert("name".into(), Value::String(new_name.to_string()));
            }
        }
    }
}

fn inject_common_labels_into_raw(raw: &mut Option<Value>, labels: &HashMap<String, String>) {
    if let Some(raw) = raw {
        if let Some(meta) = raw.get_mut("metadata") {
            if let Value::Object(meta_map) = meta {
                let labels_val = meta_map
                    .entry("labels")
                    .or_insert(Value::Object(serde_json::Map::new()));
                if let Value::Object(labels_map) = labels_val {
                    for (k, v) in labels {
                        labels_map.insert(k.clone(), Value::String(v.clone()));
                    }
                }
            }
        }
    }
}

fn inject_common_annotations_into_raw(
    raw: &mut Option<Value>,
    annotations: &HashMap<String, String>,
) {
    if let Some(raw) = raw {
        if let Some(meta) = raw.get_mut("metadata") {
            if let Value::Object(meta_map) = meta {
                let ann_val = meta_map
                    .entry("annotations")
                    .or_insert(Value::Object(serde_json::Map::new()));
                if let Value::Object(ann_map) = ann_val {
                    for (k, v) in annotations {
                        ann_map.insert(k.clone(), Value::String(v.clone()));
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Image override
// ---------------------------------------------------------------------------

/// Image tag/name override used in kustomization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageOverride {
    pub name: String,
    pub new_name: Option<String>,
    pub new_tag: Option<String>,
    pub digest: Option<String>,
}

impl ImageOverride {
    pub fn from_value(v: &Value) -> Option<Self> {
        let name = v.get("name")?.as_str()?.to_string();
        let new_name = v.get("newName").and_then(|n| n.as_str()).map(String::from);
        let new_tag = v.get("newTag").and_then(|n| n.as_str()).map(String::from);
        let digest = v.get("digest").and_then(|d| d.as_str()).map(String::from);
        Some(Self { name, new_name, new_tag, digest })
    }

    /// Apply this image override to all container images in a manifest.
    pub fn apply(&self, manifest: &mut KubernetesManifest) {
        if let Some(raw) = &mut manifest.raw {
            self.apply_to_value(raw);
        }
        // Also update the spec if present
        if let Some(spec) = &mut manifest.spec {
            self.apply_to_value(spec);
        }
    }

    fn apply_to_value(&self, value: &mut Value) {
        match value {
            Value::String(s) => {
                if self.matches_image(s) {
                    *s = self.replace_image(s);
                }
            }
            Value::Object(map) => {
                // Check for "image" fields directly
                if let Some(Value::String(img)) = map.get("image") {
                    if self.matches_image(img) {
                        let new_img = self.replace_image(img);
                        map.insert("image".into(), Value::String(new_img));
                    }
                }
                // Recurse into all values
                for v in map.values_mut() {
                    self.apply_to_value(v);
                }
            }
            Value::Array(arr) => {
                for v in arr.iter_mut() {
                    self.apply_to_value(v);
                }
            }
            _ => {}
        }
    }

    fn matches_image(&self, image: &str) -> bool {
        let img_name = image.split(':').next().unwrap_or(image);
        let img_name = img_name.split('@').next().unwrap_or(img_name);
        img_name == self.name || img_name.ends_with(&format!("/{}", self.name))
    }

    fn replace_image(&self, image: &str) -> String {
        let base_name = if let Some(new_name) = &self.new_name {
            new_name.clone()
        } else {
            image.split(':').next().unwrap_or(image).split('@').next().unwrap_or(image).to_string()
        };

        if let Some(digest) = &self.digest {
            format!("{base_name}@{digest}")
        } else if let Some(tag) = &self.new_tag {
            format!("{base_name}:{tag}")
        } else {
            base_name
        }
    }
}

// ---------------------------------------------------------------------------
// ConfigMap / Secret generators
// ---------------------------------------------------------------------------

/// Generates ConfigMaps from literals, files, or env files.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConfigMapGenerator {
    pub name: String,
    pub namespace: Option<String>,
    pub from_literals: Vec<KeyValue>,
    pub from_files: Vec<String>,
    pub from_env: Vec<String>,
    pub options: GeneratorOptions,
}

/// Key-value pair for literal data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyValue {
    pub key: String,
    pub value: String,
}

/// Generator options (shared by ConfigMap and Secret generators).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GeneratorOptions {
    pub disable_name_suffix_hash: bool,
    pub labels: HashMap<String, String>,
    pub annotations: HashMap<String, String>,
}

impl ConfigMapGenerator {
    /// Generate a ConfigMap manifest from this generator specification.
    pub fn generate(&self) -> Result<KubernetesManifest> {
        let mut data = serde_json::Map::new();

        for kv in &self.from_literals {
            data.insert(kv.key.clone(), Value::String(kv.value.clone()));
        }

        // from_files and from_env are path-based; in this context we store the reference
        for file_ref in &self.from_files {
            let key = std::path::Path::new(file_ref)
                .file_name()
                .map(|f| f.to_string_lossy().to_string())
                .unwrap_or_else(|| file_ref.clone());
            data.insert(key, Value::String(format!("# content from {file_ref}")));
        }

        for env_file in &self.from_env {
            data.insert(
                env_file.clone(),
                Value::String(format!("# env content from {env_file}")),
            );
        }

        let name = if self.options.disable_name_suffix_hash {
            self.name.clone()
        } else {
            let hash = compute_data_hash(&data);
            format!("{}-{}", self.name, &hash[..8])
        };

        let labels = self.options.labels.clone();
        let annotations = self.options.annotations.clone();

        let metadata = ManifestMetadata {
            name: name.clone(),
            namespace: self.namespace.clone(),
            labels,
            annotations,
        };

        let spec_value = serde_json::json!({
            "data": Value::Object(data)
        });

        // Build full manifest value
        let mut manifest_map = serde_json::Map::new();
        manifest_map.insert("apiVersion".into(), Value::String("v1".into()));
        manifest_map.insert("kind".into(), Value::String("ConfigMap".into()));
        manifest_map.insert("metadata".into(), serde_json::json!({
            "name": name,
            "namespace": metadata.namespace,
            "labels": metadata.labels,
            "annotations": metadata.annotations,
        }));
        manifest_map.insert("data".into(), spec_value["data"].clone());

        Ok(KubernetesManifest {
            api_version: "v1".into(),
            kind: "ConfigMap".into(),
            metadata,
            spec: None,
            raw: Some(Value::Object(manifest_map)),
        })
    }
}

/// Secret generator (similar to ConfigMap but for Secrets).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SecretGenerator {
    pub name: String,
    pub namespace: Option<String>,
    pub type_: Option<String>,
    pub from_literals: Vec<KeyValue>,
    pub from_files: Vec<String>,
    pub options: GeneratorOptions,
}

fn compute_data_hash(data: &serde_json::Map<String, Value>) -> String {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    // Sort keys for deterministic hashing
    let mut keys: Vec<&String> = data.keys().collect();
    keys.sort();
    for k in keys {
        hasher.update(k.as_bytes());
        if let Some(Value::String(v)) = data.get(k) {
            hasher.update(v.as_bytes());
        }
    }
    hex::encode(hasher.finalize())
}

fn parse_config_map_generators(v: &Value) -> Vec<ConfigMapGenerator> {
    v.get("configMapGenerator")
        .and_then(|arr| arr.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    let name = item.get("name")?.as_str()?.to_string();
                    let namespace = item.get("namespace").and_then(|n| n.as_str()).map(String::from);
                    let from_literals = item
                        .get("literals")
                        .and_then(|l| l.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|l| {
                                    let s = l.as_str()?;
                                    let (key, value) = s.split_once('=')?;
                                    Some(KeyValue {
                                        key: key.to_string(),
                                        value: value.to_string(),
                                    })
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    let from_files = parse_string_array(item.get("files"));
                    let from_env = parse_string_array(item.get("envs"));
                    let options = parse_generator_options(item);
                    Some(ConfigMapGenerator {
                        name,
                        namespace,
                        from_literals,
                        from_files,
                        from_env,
                        options,
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn parse_secret_generators(v: &Value) -> Vec<SecretGenerator> {
    v.get("secretGenerator")
        .and_then(|arr| arr.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    let name = item.get("name")?.as_str()?.to_string();
                    let namespace = item.get("namespace").and_then(|n| n.as_str()).map(String::from);
                    let type_ = item.get("type").and_then(|t| t.as_str()).map(String::from);
                    let from_literals = item
                        .get("literals")
                        .and_then(|l| l.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|l| {
                                    let s = l.as_str()?;
                                    let (key, value) = s.split_once('=')?;
                                    Some(KeyValue {
                                        key: key.to_string(),
                                        value: value.to_string(),
                                    })
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    let from_files = parse_string_array(item.get("files"));
                    let options = parse_generator_options(item);
                    Some(SecretGenerator {
                        name,
                        namespace,
                        type_,
                        from_literals,
                        from_files,
                        options,
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn parse_generator_options(item: &Value) -> GeneratorOptions {
    let disable_name_suffix_hash = item
        .get("options")
        .and_then(|o| o.get("disableNameSuffixHash"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let labels = parse_string_map(
        item.get("options").and_then(|o| o.get("labels")),
    );
    let annotations = parse_string_map(
        item.get("options").and_then(|o| o.get("annotations")),
    );
    GeneratorOptions { disable_name_suffix_hash, labels, annotations }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_string_array(v: Option<&Value>) -> Vec<String> {
    v.and_then(|a| a.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default()
}

fn parse_string_map(v: Option<&Value>) -> HashMap<String, String> {
    let mut result = HashMap::new();
    if let Some(Value::Object(obj)) = v {
        for (k, v) in obj {
            if let Some(s) = v.as_str() {
                result.insert(k.clone(), s.to_string());
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const KUSTOMIZATION_YAML: &str = r#"
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: production
namePrefix: prod-
commonLabels:
  environment: production
  team: platform
commonAnnotations:
  managed-by: kustomize
resources:
  - deployment.yaml
  - service.yaml
images:
  - name: nginx
    newTag: "1.22.0"
  - name: redis
    newName: my-registry.io/redis
    newTag: "7.0"
configMapGenerator:
  - name: app-config
    literals:
      - KEY1=value1
      - KEY2=value2
patches:
  - target:
      kind: Deployment
      name: nginx
    patch: |
      apiVersion: apps/v1
      kind: Deployment
      metadata:
        name: nginx
      spec:
        replicas: 5
"#;

    #[test]
    fn test_parse_kustomization() {
        let kustomization = Kustomization::parse(KUSTOMIZATION_YAML).unwrap();
        assert_eq!(kustomization.namespace.as_deref(), Some("production"));
        assert_eq!(kustomization.name_prefix.as_deref(), Some("prod-"));
        assert_eq!(kustomization.resources, vec!["deployment.yaml", "service.yaml"]);
        assert_eq!(
            kustomization.common_labels.get("environment").unwrap(),
            "production"
        );
        assert_eq!(kustomization.images.len(), 2);
        assert_eq!(kustomization.images[0].name, "nginx");
        assert_eq!(kustomization.images[0].new_tag.as_deref(), Some("1.22.0"));
        assert_eq!(kustomization.config_map_generator.len(), 1);
        assert_eq!(kustomization.config_map_generator[0].name, "app-config");
    }

    #[test]
    fn test_strategic_merge_patch_basic() {
        let base = serde_json::json!({
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "test"},
            "spec": {
                "replicas": 1,
                "template": {
                    "spec": {
                        "containers": [
                            {"name": "app", "image": "app:1.0", "ports": [{"containerPort": 80}]}
                        ]
                    }
                }
            }
        });
        let patch = serde_json::json!({
            "spec": {
                "replicas": 3
            }
        });
        let result = StrategicMergePatch::apply(&base, &patch).unwrap();
        assert_eq!(result["spec"]["replicas"], 3);
        // Original fields preserved
        assert_eq!(result["apiVersion"], "apps/v1");
    }

    #[test]
    fn test_strategic_merge_patch_nested() {
        let base = serde_json::json!({
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {"name": "app", "image": "app:1.0", "resources": {"limits": {"cpu": "100m"}}}
                        ]
                    }
                }
            }
        });
        let patch = serde_json::json!({
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {"name": "app", "image": "app:2.0"}
                        ]
                    }
                }
            }
        });
        let result = StrategicMergePatch::apply(&base, &patch).unwrap();
        let containers = result["spec"]["template"]["spec"]["containers"]
            .as_array()
            .unwrap();
        assert_eq!(containers.len(), 1);
        assert_eq!(containers[0]["image"], "app:2.0");
        // Resources preserved from base due to strategic merge
        assert_eq!(containers[0]["resources"]["limits"]["cpu"], "100m");
    }

    #[test]
    fn test_strategic_merge_patch_delete_directive() {
        let base = serde_json::json!({
            "a": 1,
            "b": {"c": 2, "d": 3}
        });
        let patch = serde_json::json!({
            "b": {"$patch": "delete"}
        });
        let result = StrategicMergePatch::apply(&base, &patch).unwrap();
        assert_eq!(result["a"], 1);
        assert!(result["b"].is_null());
    }

    #[test]
    fn test_strategic_merge_patch_replace_directive() {
        let base = serde_json::json!({
            "spec": {
                "containers": [
                    {"name": "a", "image": "a:1"},
                    {"name": "b", "image": "b:1"}
                ]
            }
        });
        let patch = serde_json::json!({
            "spec": {
                "$patch": "replace",
                "containers": [{"name": "c", "image": "c:1"}]
            }
        });
        let result = StrategicMergePatch::apply(&base, &patch).unwrap();
        let containers = result["spec"]["containers"].as_array().unwrap();
        assert_eq!(containers.len(), 1);
        assert_eq!(containers[0]["name"], "c");
    }

    #[test]
    fn test_json_patch_add() {
        let mut doc = serde_json::json!({"a": 1});
        let op = JsonPatchOperation::Add {
            path: "/b".into(),
            value: serde_json::json!(2),
        };
        op.apply(&mut doc).unwrap();
        assert_eq!(doc["b"], 2);
    }

    #[test]
    fn test_json_patch_remove() {
        let mut doc = serde_json::json!({"a": 1, "b": 2});
        let op = JsonPatchOperation::Remove { path: "/b".into() };
        op.apply(&mut doc).unwrap();
        assert!(doc.get("b").is_none());
    }

    #[test]
    fn test_json_patch_replace() {
        let mut doc = serde_json::json!({"a": 1});
        let op = JsonPatchOperation::Replace {
            path: "/a".into(),
            value: serde_json::json!(99),
        };
        op.apply(&mut doc).unwrap();
        assert_eq!(doc["a"], 99);
    }

    #[test]
    fn test_json_patch_move() {
        let mut doc = serde_json::json!({"a": 1, "b": 2});
        let op = JsonPatchOperation::Move {
            from: "/a".into(),
            path: "/c".into(),
        };
        op.apply(&mut doc).unwrap();
        assert!(doc.get("a").is_none());
        assert_eq!(doc["c"], 1);
    }

    #[test]
    fn test_json_patch_copy() {
        let mut doc = serde_json::json!({"a": 1});
        let op = JsonPatchOperation::Copy {
            from: "/a".into(),
            path: "/b".into(),
        };
        op.apply(&mut doc).unwrap();
        assert_eq!(doc["a"], 1);
        assert_eq!(doc["b"], 1);
    }

    #[test]
    fn test_json_patch_test_pass() {
        let mut doc = serde_json::json!({"a": 1});
        let op = JsonPatchOperation::Test {
            path: "/a".into(),
            value: serde_json::json!(1),
        };
        assert!(op.apply(&mut doc).is_ok());
    }

    #[test]
    fn test_json_patch_test_fail() {
        let mut doc = serde_json::json!({"a": 1});
        let op = JsonPatchOperation::Test {
            path: "/a".into(),
            value: serde_json::json!(2),
        };
        assert!(op.apply(&mut doc).is_err());
    }

    #[test]
    fn test_json_patch_array() {
        let mut doc = serde_json::json!({"arr": [1, 2, 3]});
        let op = JsonPatchOperation::Add {
            path: "/arr/-".into(),
            value: serde_json::json!(4),
        };
        op.apply(&mut doc).unwrap();
        assert_eq!(doc["arr"], serde_json::json!([1, 2, 3, 4]));

        let op2 = JsonPatchOperation::Remove {
            path: "/arr/0".into(),
        };
        op2.apply(&mut doc).unwrap();
        assert_eq!(doc["arr"], serde_json::json!([2, 3, 4]));
    }

    #[test]
    fn test_json_patch_apply_all() {
        let mut doc = serde_json::json!({"a": 1, "b": 2});
        let ops = vec![
            JsonPatchOperation::Replace {
                path: "/a".into(),
                value: serde_json::json!(10),
            },
            JsonPatchOperation::Add {
                path: "/c".into(),
                value: serde_json::json!(3),
            },
        ];
        JsonPatchOperation::apply_all(&ops, &mut doc).unwrap();
        assert_eq!(doc["a"], 10);
        assert_eq!(doc["c"], 3);
    }

    #[test]
    fn test_image_override_replace_tag() {
        let ov = ImageOverride {
            name: "nginx".into(),
            new_name: None,
            new_tag: Some("1.22.0".into()),
            digest: None,
        };
        assert!(ov.matches_image("nginx:1.21.0"));
        assert!(ov.matches_image("docker.io/nginx:latest"));
        assert!(!ov.matches_image("redis:latest"));
        assert_eq!(ov.replace_image("nginx:1.21.0"), "nginx:1.22.0");
    }

    #[test]
    fn test_image_override_replace_name_and_tag() {
        let ov = ImageOverride {
            name: "redis".into(),
            new_name: Some("my-registry.io/redis".into()),
            new_tag: Some("7.0".into()),
            digest: None,
        };
        assert_eq!(ov.replace_image("redis:6.0"), "my-registry.io/redis:7.0");
    }

    #[test]
    fn test_image_override_digest() {
        let ov = ImageOverride {
            name: "nginx".into(),
            new_name: None,
            new_tag: None,
            digest: Some("sha256:abc123".into()),
        };
        assert_eq!(ov.replace_image("nginx:1.21.0"), "nginx@sha256:abc123");
    }

    #[test]
    fn test_configmap_generator() {
        let gen = ConfigMapGenerator {
            name: "my-config".into(),
            namespace: Some("default".into()),
            from_literals: vec![
                KeyValue { key: "KEY1".into(), value: "value1".into() },
                KeyValue { key: "KEY2".into(), value: "value2".into() },
            ],
            from_files: Vec::new(),
            from_env: Vec::new(),
            options: GeneratorOptions::default(),
        };
        let manifest = gen.generate().unwrap();
        assert_eq!(manifest.kind, "ConfigMap");
        assert!(manifest.metadata.name.starts_with("my-config-"));
    }

    #[test]
    fn test_configmap_generator_no_hash() {
        let gen = ConfigMapGenerator {
            name: "my-config".into(),
            namespace: None,
            from_literals: vec![KeyValue { key: "A".into(), value: "B".into() }],
            from_files: Vec::new(),
            from_env: Vec::new(),
            options: GeneratorOptions {
                disable_name_suffix_hash: true,
                ..Default::default()
            },
        };
        let manifest = gen.generate().unwrap();
        assert_eq!(manifest.metadata.name, "my-config");
    }

    #[test]
    fn test_kustomize_resolver_namespace() {
        let kust = Kustomization {
            namespace: Some("production".into()),
            ..Default::default()
        };
        let manifests = vec![KubernetesManifest {
            api_version: "apps/v1".into(),
            kind: "Deployment".into(),
            metadata: ManifestMetadata {
                name: "test".into(),
                namespace: None,
                ..Default::default()
            },
            spec: None,
            raw: Some(serde_json::json!({
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {"name": "test"}
            })),
        }];
        let result = KustomizeResolver::resolve(&kust, &manifests).unwrap();
        assert_eq!(result[0].metadata.namespace.as_deref(), Some("production"));
    }

    #[test]
    fn test_kustomize_resolver_prefix_suffix() {
        let kust = Kustomization {
            name_prefix: Some("prod-".into()),
            name_suffix: Some("-v2".into()),
            ..Default::default()
        };
        let manifests = vec![KubernetesManifest {
            api_version: "v1".into(),
            kind: "Service".into(),
            metadata: ManifestMetadata {
                name: "web".into(),
                ..Default::default()
            },
            spec: None,
            raw: Some(serde_json::json!({
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {"name": "web"}
            })),
        }];
        let result = KustomizeResolver::resolve(&kust, &manifests).unwrap();
        assert_eq!(result[0].metadata.name, "prod-web-v2");
    }

    #[test]
    fn test_kustomize_resolver_common_labels() {
        let mut labels = HashMap::new();
        labels.insert("env".into(), "staging".into());
        let kust = Kustomization {
            common_labels: labels,
            ..Default::default()
        };
        let manifests = vec![KubernetesManifest {
            api_version: "v1".into(),
            kind: "ConfigMap".into(),
            metadata: ManifestMetadata {
                name: "cfg".into(),
                ..Default::default()
            },
            spec: None,
            raw: Some(serde_json::json!({
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {"name": "cfg"}
            })),
        }];
        let result = KustomizeResolver::resolve(&kust, &manifests).unwrap();
        assert_eq!(result[0].metadata.labels.get("env").unwrap(), "staging");
    }

    #[test]
    fn test_patch_target_matches() {
        let target = PatchTarget {
            kind: Some("Deployment".into()),
            name: Some("nginx".into()),
            ..Default::default()
        };
        let manifest = KubernetesManifest {
            api_version: "apps/v1".into(),
            kind: "Deployment".into(),
            metadata: ManifestMetadata {
                name: "nginx".into(),
                ..Default::default()
            },
            spec: None,
            raw: None,
        };
        assert!(target.matches(&manifest));

        let other = KubernetesManifest {
            api_version: "apps/v1".into(),
            kind: "Deployment".into(),
            metadata: ManifestMetadata {
                name: "redis".into(),
                ..Default::default()
            },
            spec: None,
            raw: None,
        };
        assert!(!target.matches(&other));
    }

    #[test]
    fn test_parse_json_patch_string() {
        let patch_str = r#"
- op: add
  path: /metadata/labels/env
  value: production
- op: replace
  path: /spec/replicas
  value: 5
"#;
        let ops = parse_json_patch_string(patch_str).unwrap();
        assert_eq!(ops.len(), 2);
        match &ops[0] {
            JsonPatchOperation::Add { path, value } => {
                assert_eq!(path, "/metadata/labels/env");
                assert_eq!(*value, Value::String("production".into()));
            }
            _ => panic!("Expected Add"),
        }
    }

    #[test]
    fn test_kustomize_resolver_image_override() {
        let kust = Kustomization {
            images: vec![ImageOverride {
                name: "nginx".into(),
                new_name: None,
                new_tag: Some("1.22.0".into()),
                digest: None,
            }],
            ..Default::default()
        };
        let manifests = vec![KubernetesManifest {
            api_version: "apps/v1".into(),
            kind: "Deployment".into(),
            metadata: ManifestMetadata {
                name: "web".into(),
                ..Default::default()
            },
            spec: Some(serde_json::json!({
                "template": {
                    "spec": {
                        "containers": [{"name": "web", "image": "nginx:1.21.0"}]
                    }
                }
            })),
            raw: Some(serde_json::json!({
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {"name": "web"},
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [{"name": "web", "image": "nginx:1.21.0"}]
                        }
                    }
                }
            })),
        }];
        let result = KustomizeResolver::resolve(&kust, &manifests).unwrap();
        let raw = result[0].raw.as_ref().unwrap();
        let image = raw["spec"]["template"]["spec"]["containers"][0]["image"]
            .as_str()
            .unwrap();
        assert_eq!(image, "nginx:1.22.0");
    }
}
