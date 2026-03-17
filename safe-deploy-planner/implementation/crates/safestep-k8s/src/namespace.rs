//! Namespace resolution and filtering for Kubernetes resources.

use std::collections::{HashMap, HashSet};
use std::fmt;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Error, Serialize, Deserialize, PartialEq, Eq)]
pub enum NamespaceError {
    #[error("cross-namespace reference denied: {source_ns} -> {target_ns} for {kind}/{name}")]
    CrossNamespaceDenied {
        source_ns: String,
        target_ns: String,
        kind: String,
        name: String,
    },
    #[error("namespace excluded by filter: {namespace}")]
    NamespaceExcluded { namespace: String },
    #[error("invalid namespace name: {name} — {reason}")]
    InvalidName { name: String, reason: String },
    #[error("missing required field: {field}")]
    MissingField { field: String },
}

pub type Result<T> = std::result::Result<T, NamespaceError>;

// ---------------------------------------------------------------------------
// validate_namespace_name
// ---------------------------------------------------------------------------

/// Validates a Kubernetes namespace name according to DNS label rules:
/// non-empty, at most 63 characters, lowercase alphanumeric and hyphens only,
/// must not start or end with a hyphen.
pub fn validate_namespace_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(NamespaceError::InvalidName {
            name: name.to_string(),
            reason: "must not be empty".into(),
        });
    }
    if name.len() > 63 {
        return Err(NamespaceError::InvalidName {
            name: name.to_string(),
            reason: format!("length {} exceeds maximum of 63", name.len()),
        });
    }
    if name.starts_with('-') || name.ends_with('-') {
        return Err(NamespaceError::InvalidName {
            name: name.to_string(),
            reason: "must not start or end with a hyphen".into(),
        });
    }
    for ch in name.chars() {
        if !(ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '-') {
            return Err(NamespaceError::InvalidName {
                name: name.to_string(),
                reason: format!("invalid character '{ch}'; only lowercase letters, digits, and hyphens allowed"),
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// NamespaceResolver
// ---------------------------------------------------------------------------

/// Resolves the target namespace for a Kubernetes resource.
///
/// Resolution order:
/// 1. Exact key `"kind/name"` (kind lowercased).
/// 2. Wildcard key `"kind/*"`.
/// 3. `default_namespace`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamespaceResolver {
    pub default_namespace: String,
    pub overrides: HashMap<String, String>,
}

impl NamespaceResolver {
    pub fn with_default(ns: &str) -> Self {
        Self {
            default_namespace: ns.to_string(),
            overrides: HashMap::new(),
        }
    }

    pub fn new(default: &str, overrides: HashMap<String, String>) -> Self {
        Self {
            default_namespace: default.to_string(),
            overrides,
        }
    }

    /// Adds an override mapping `"kind/name"` → `namespace`.
    /// The kind is lowercased for consistent lookup.
    pub fn add_override(&mut self, resource_kind: &str, resource_name: &str, namespace: &str) {
        let key = format!("{}/{}", resource_kind.to_ascii_lowercase(), resource_name);
        self.overrides.insert(key, namespace.to_string());
    }

    /// Resolves the namespace for `kind/name`:
    /// exact match → wildcard `kind/*` → default.
    pub fn resolve(&self, resource_kind: &str, resource_name: &str) -> String {
        let kind_lower = resource_kind.to_ascii_lowercase();
        let exact_key = format!("{kind_lower}/{resource_name}");
        if let Some(ns) = self.overrides.get(&exact_key) {
            return ns.clone();
        }
        let wildcard_key = format!("{kind_lower}/*");
        if let Some(ns) = self.overrides.get(&wildcard_key) {
            return ns.clone();
        }
        self.default_namespace.clone()
    }

    pub fn override_count(&self) -> usize {
        self.overrides.len()
    }

    pub fn clear_overrides(&mut self) {
        self.overrides.clear();
    }

    /// Returns the override keys in sorted order.
    pub fn overridden_keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.overrides.keys().cloned().collect();
        keys.sort();
        keys
    }
}

// ---------------------------------------------------------------------------
// NamespaceFilter
// ---------------------------------------------------------------------------

/// Filters namespaces by inclusion / exclusion lists.
///
/// * Exclude takes precedence over include.
/// * Empty `include` means "allow all (not excluded)".
/// * Supports simple glob-like patterns: a leading or trailing `*`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamespaceFilter {
    pub include: Vec<String>,
    pub exclude: Vec<String>,
}

/// Returns `true` if `value` matches `pattern` (supports leading/trailing `*`).
fn glob_match(pattern: &str, value: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if let Some(suffix) = pattern.strip_prefix('*') {
        return value.ends_with(suffix);
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        return value.starts_with(prefix);
    }
    pattern == value
}

impl NamespaceFilter {
    /// Allow every namespace (empty include and exclude).
    pub fn allow_all() -> Self {
        Self {
            include: Vec::new(),
            exclude: Vec::new(),
        }
    }

    /// Allow exactly one namespace.
    pub fn single(ns: &str) -> Self {
        Self {
            include: vec![ns.to_string()],
            exclude: Vec::new(),
        }
    }

    pub fn new(include: Vec<String>, exclude: Vec<String>) -> Self {
        Self { include, exclude }
    }

    /// Returns `true` when the given namespace is allowed by this filter.
    pub fn matches(&self, namespace: &str) -> bool {
        // Exclude takes precedence.
        for pat in &self.exclude {
            if glob_match(pat, namespace) {
                return false;
            }
        }
        // Empty include means "all".
        if self.include.is_empty() {
            return true;
        }
        for pat in &self.include {
            if glob_match(pat, namespace) {
                return true;
            }
        }
        false
    }

    pub fn is_allow_all(&self) -> bool {
        self.include.is_empty() && self.exclude.is_empty()
    }

    pub fn include_count(&self) -> usize {
        self.include.len()
    }

    pub fn exclude_count(&self) -> usize {
        self.exclude.len()
    }

    pub fn add_include(&mut self, ns: &str) {
        self.include.push(ns.to_string());
    }

    pub fn add_exclude(&mut self, ns: &str) {
        self.exclude.push(ns.to_string());
    }

    /// Keeps only the namespaces that pass this filter.
    pub fn filter_namespaces<'a>(&self, namespaces: &'a [String]) -> Vec<&'a String> {
        namespaces.iter().filter(|ns| self.matches(ns)).collect()
    }
}

// ---------------------------------------------------------------------------
// CrossNamespaceRef
// ---------------------------------------------------------------------------

/// A reference from one namespace to a resource in another namespace.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CrossNamespaceRef {
    pub source_ns: String,
    pub target_ns: String,
    pub resource_kind: String,
    pub resource_name: String,
}

impl CrossNamespaceRef {
    pub fn new(source_ns: &str, target_ns: &str, kind: &str, name: &str) -> Self {
        Self {
            source_ns: source_ns.to_string(),
            target_ns: target_ns.to_string(),
            resource_kind: kind.to_string(),
            resource_name: name.to_string(),
        }
    }

    pub fn is_same_namespace(&self) -> bool {
        self.source_ns == self.target_ns
    }

    /// Validates that all fields are non-empty and namespace names are valid DNS labels.
    pub fn validate(&self) -> Result<()> {
        if self.source_ns.is_empty() {
            return Err(NamespaceError::MissingField {
                field: "source_ns".into(),
            });
        }
        if self.target_ns.is_empty() {
            return Err(NamespaceError::MissingField {
                field: "target_ns".into(),
            });
        }
        if self.resource_kind.is_empty() {
            return Err(NamespaceError::MissingField {
                field: "resource_kind".into(),
            });
        }
        if self.resource_name.is_empty() {
            return Err(NamespaceError::MissingField {
                field: "resource_name".into(),
            });
        }
        validate_namespace_name(&self.source_ns)?;
        validate_namespace_name(&self.target_ns)?;
        Ok(())
    }

    pub fn description(&self) -> String {
        format!(
            "{}/{} ({} -> {})",
            self.resource_kind, self.resource_name, self.source_ns, self.target_ns
        )
    }
}

impl fmt::Display for CrossNamespaceRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}/{} ({} -> {})",
            self.resource_kind, self.resource_name, self.source_ns, self.target_ns
        )
    }
}

// ---------------------------------------------------------------------------
// NamespacePolicy
// ---------------------------------------------------------------------------

/// Governs which cross-namespace references are permitted.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NamespacePolicy {
    /// No cross-namespace references allowed.
    Strict,
    /// All cross-namespace references allowed.
    Permissive,
    /// Only explicitly listed `(from_pattern, to_pattern)` pairs allowed.
    /// Patterns support `*` (any) and trailing `*` prefix globs (e.g. `prod-*`).
    Selective(Vec<(String, String)>),
}

impl NamespacePolicy {
    /// Returns `true` when traffic from `from_ns` to `to_ns` is allowed.
    /// Same-namespace references are always allowed.
    pub fn allows(&self, from_ns: &str, to_ns: &str) -> bool {
        if from_ns == to_ns {
            return true;
        }
        match self {
            Self::Strict => false,
            Self::Permissive => true,
            Self::Selective(pairs) => pairs
                .iter()
                .any(|(fp, tp)| glob_match(fp, from_ns) && glob_match(tp, to_ns)),
        }
    }

    /// Validates a `CrossNamespaceRef` against this policy.
    pub fn validate_ref(&self, cross_ref: &CrossNamespaceRef) -> Result<()> {
        if self.allows(&cross_ref.source_ns, &cross_ref.target_ns) {
            Ok(())
        } else {
            Err(NamespaceError::CrossNamespaceDenied {
                source_ns: cross_ref.source_ns.clone(),
                target_ns: cross_ref.target_ns.clone(),
                kind: cross_ref.resource_kind.clone(),
                name: cross_ref.resource_name.clone(),
            })
        }
    }

    pub fn description(&self) -> String {
        match self {
            Self::Strict => "strict: no cross-namespace references".into(),
            Self::Permissive => "permissive: all cross-namespace references allowed".into(),
            Self::Selective(pairs) => {
                format!("selective: {} allowed pair(s)", pairs.len())
            }
        }
    }

    /// Number of explicit pairs (0 for Strict and Permissive).
    pub fn pair_count(&self) -> usize {
        match self {
            Self::Strict | Self::Permissive => 0,
            Self::Selective(pairs) => pairs.len(),
        }
    }

    /// Checks every reference, returning a `Vec` of errors for denied ones.
    pub fn check_all(&self, refs: &[CrossNamespaceRef]) -> Vec<NamespaceError> {
        refs.iter()
            .filter_map(|r| self.validate_ref(r).err())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Free-standing helpers
// ---------------------------------------------------------------------------

/// Collects every unique namespace mentioned (source or target) in the refs.
pub fn collect_namespaces(refs: &[CrossNamespaceRef]) -> HashSet<String> {
    let mut set = HashSet::new();
    for r in refs {
        set.insert(r.source_ns.clone());
        set.insert(r.target_ns.clone());
    }
    set
}

/// Groups references by their source namespace.
pub fn group_by_source<'a>(refs: &'a [CrossNamespaceRef]) -> HashMap<String, Vec<&'a CrossNamespaceRef>> {
    let mut map: HashMap<String, Vec<&'a CrossNamespaceRef>> = HashMap::new();
    for r in refs {
        map.entry(r.source_ns.clone()).or_default().push(r);
    }
    map
}

/// Groups references by their target namespace.
pub fn group_by_target<'a>(refs: &'a [CrossNamespaceRef]) -> HashMap<String, Vec<&'a CrossNamespaceRef>> {
    let mut map: HashMap<String, Vec<&'a CrossNamespaceRef>> = HashMap::new();
    for r in refs {
        map.entry(r.target_ns.clone()).or_default().push(r);
    }
    map
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- validate_namespace_name -------------------------------------------

    #[test]
    fn valid_namespace_names() {
        assert!(validate_namespace_name("default").is_ok());
        assert!(validate_namespace_name("kube-system").is_ok());
        assert!(validate_namespace_name("ns1").is_ok());
        assert!(validate_namespace_name("a").is_ok());
        assert!(validate_namespace_name("a-b-c").is_ok());
        let max = "a".repeat(63);
        assert!(validate_namespace_name(&max).is_ok());
    }

    #[test]
    fn invalid_namespace_empty() {
        let err = validate_namespace_name("").unwrap_err();
        assert!(matches!(err, NamespaceError::InvalidName { .. }));
    }

    #[test]
    fn invalid_namespace_too_long() {
        let long = "a".repeat(64);
        let err = validate_namespace_name(&long).unwrap_err();
        match &err {
            NamespaceError::InvalidName { reason, .. } => assert!(reason.contains("64")),
            _ => panic!("expected InvalidName"),
        }
    }

    #[test]
    fn invalid_namespace_leading_hyphen() {
        assert!(validate_namespace_name("-abc").is_err());
    }

    #[test]
    fn invalid_namespace_trailing_hyphen() {
        assert!(validate_namespace_name("abc-").is_err());
    }

    #[test]
    fn invalid_namespace_uppercase() {
        assert!(validate_namespace_name("Default").is_err());
    }

    #[test]
    fn invalid_namespace_underscore() {
        assert!(validate_namespace_name("my_ns").is_err());
    }

    #[test]
    fn invalid_namespace_dot() {
        assert!(validate_namespace_name("my.ns").is_err());
    }

    // --- NamespaceResolver -------------------------------------------------

    #[test]
    fn resolver_default_only() {
        let r = NamespaceResolver::with_default("production");
        assert_eq!(r.resolve("Deployment", "web"), "production");
        assert_eq!(r.override_count(), 0);
    }

    #[test]
    fn resolver_exact_override() {
        let mut r = NamespaceResolver::with_default("default");
        r.add_override("Deployment", "web", "frontend");
        assert_eq!(r.resolve("Deployment", "web"), "frontend");
        assert_eq!(r.resolve("Deployment", "api"), "default");
    }

    #[test]
    fn resolver_kind_case_insensitive() {
        let mut r = NamespaceResolver::with_default("default");
        r.add_override("deployment", "web", "frontend");
        assert_eq!(r.resolve("Deployment", "web"), "frontend");
        assert_eq!(r.resolve("DEPLOYMENT", "web"), "frontend");
    }

    #[test]
    fn resolver_wildcard_override() {
        let mut r = NamespaceResolver::with_default("default");
        r.add_override("Service", "*", "services-ns");
        assert_eq!(r.resolve("Service", "anything"), "services-ns");
        assert_eq!(r.resolve("Service", "other"), "services-ns");
        assert_eq!(r.resolve("Deployment", "x"), "default");
    }

    #[test]
    fn resolver_exact_beats_wildcard() {
        let mut r = NamespaceResolver::with_default("default");
        r.add_override("Service", "*", "services-ns");
        r.add_override("Service", "special", "special-ns");
        assert_eq!(r.resolve("Service", "special"), "special-ns");
        assert_eq!(r.resolve("Service", "other"), "services-ns");
    }

    #[test]
    fn resolver_new_with_overrides() {
        let mut overrides = HashMap::new();
        overrides.insert("deployment/web".into(), "frontend".into());
        let r = NamespaceResolver::new("default", overrides);
        assert_eq!(r.resolve("Deployment", "web"), "frontend");
        assert_eq!(r.override_count(), 1);
    }

    #[test]
    fn resolver_clear_overrides() {
        let mut r = NamespaceResolver::with_default("default");
        r.add_override("Deployment", "web", "frontend");
        assert_eq!(r.override_count(), 1);
        r.clear_overrides();
        assert_eq!(r.override_count(), 0);
        assert_eq!(r.resolve("Deployment", "web"), "default");
    }

    #[test]
    fn resolver_overridden_keys_sorted() {
        let mut r = NamespaceResolver::with_default("default");
        r.add_override("Service", "z-svc", "ns1");
        r.add_override("Deployment", "a-dep", "ns2");
        r.add_override("ConfigMap", "mid", "ns3");
        let keys = r.overridden_keys();
        assert_eq!(keys, vec!["configmap/mid", "deployment/a-dep", "service/z-svc"]);
    }

    // --- NamespaceFilter ---------------------------------------------------

    #[test]
    fn filter_allow_all() {
        let f = NamespaceFilter::allow_all();
        assert!(f.matches("anything"));
        assert!(f.matches("kube-system"));
        assert!(f.is_allow_all());
    }

    #[test]
    fn filter_single() {
        let f = NamespaceFilter::single("production");
        assert!(f.matches("production"));
        assert!(!f.matches("staging"));
        assert!(!f.is_allow_all());
        assert_eq!(f.include_count(), 1);
        assert_eq!(f.exclude_count(), 0);
    }

    #[test]
    fn filter_exclude_precedence() {
        let f = NamespaceFilter::new(
            vec!["production".into()],
            vec!["production".into()],
        );
        assert!(!f.matches("production"));
    }

    #[test]
    fn filter_glob_suffix() {
        let f = NamespaceFilter::new(vec!["prod-*".into()], vec![]);
        assert!(f.matches("prod-us"));
        assert!(f.matches("prod-eu"));
        assert!(!f.matches("staging"));
    }

    #[test]
    fn filter_glob_prefix() {
        let f = NamespaceFilter::new(vec!["*-system".into()], vec![]);
        assert!(f.matches("kube-system"));
        assert!(f.matches("my-system"));
        assert!(!f.matches("default"));
    }

    #[test]
    fn filter_exclude_glob() {
        let f = NamespaceFilter::new(vec![], vec!["kube-*".into()]);
        assert!(!f.matches("kube-system"));
        assert!(!f.matches("kube-public"));
        assert!(f.matches("default"));
    }

    #[test]
    fn filter_add_include_and_exclude() {
        let mut f = NamespaceFilter::allow_all();
        f.add_include("ns1");
        f.add_exclude("ns2");
        assert!(f.matches("ns1"));
        assert!(!f.matches("ns2"));
        assert!(!f.matches("ns3")); // include is now non-empty, ns3 not in it
    }

    #[test]
    fn filter_namespaces_vec() {
        let f = NamespaceFilter::new(
            vec!["prod-*".into(), "staging".into()],
            vec!["prod-secret".into()],
        );
        let nss: Vec<String> = vec![
            "prod-us".into(),
            "prod-secret".into(),
            "staging".into(),
            "dev".into(),
        ];
        let filtered = f.filter_namespaces(&nss);
        assert_eq!(filtered, vec![&"prod-us".to_string(), &"staging".to_string()]);
    }

    #[test]
    fn filter_star_pattern_matches_everything() {
        let f = NamespaceFilter::new(vec!["*".into()], vec![]);
        assert!(f.matches("anything"));
        assert!(f.matches(""));
    }

    // --- CrossNamespaceRef -------------------------------------------------

    #[test]
    fn cross_ref_same_ns() {
        let r = CrossNamespaceRef::new("default", "default", "Service", "web");
        assert!(r.is_same_namespace());
    }

    #[test]
    fn cross_ref_different_ns() {
        let r = CrossNamespaceRef::new("frontend", "backend", "Service", "api");
        assert!(!r.is_same_namespace());
    }

    #[test]
    fn cross_ref_validate_ok() {
        let r = CrossNamespaceRef::new("ns1", "ns2", "Service", "api");
        assert!(r.validate().is_ok());
    }

    #[test]
    fn cross_ref_validate_missing_source() {
        let r = CrossNamespaceRef::new("", "ns2", "Service", "api");
        let err = r.validate().unwrap_err();
        assert!(matches!(err, NamespaceError::MissingField { .. }));
    }

    #[test]
    fn cross_ref_validate_missing_target() {
        let r = CrossNamespaceRef::new("ns1", "", "Service", "api");
        assert!(r.validate().is_err());
    }

    #[test]
    fn cross_ref_validate_missing_kind() {
        let r = CrossNamespaceRef::new("ns1", "ns2", "", "api");
        assert!(r.validate().is_err());
    }

    #[test]
    fn cross_ref_validate_missing_name() {
        let r = CrossNamespaceRef::new("ns1", "ns2", "Service", "");
        assert!(r.validate().is_err());
    }

    #[test]
    fn cross_ref_validate_bad_ns_name() {
        let r = CrossNamespaceRef::new("INVALID", "ns2", "Service", "api");
        let err = r.validate().unwrap_err();
        assert!(matches!(err, NamespaceError::InvalidName { .. }));
    }

    #[test]
    fn cross_ref_description_and_display() {
        let r = CrossNamespaceRef::new("src", "tgt", "Service", "api");
        let desc = r.description();
        assert_eq!(desc, "Service/api (src -> tgt)");
        assert_eq!(format!("{r}"), desc);
    }

    // --- NamespacePolicy ---------------------------------------------------

    #[test]
    fn policy_strict_same_ns() {
        let p = NamespacePolicy::Strict;
        assert!(p.allows("ns1", "ns1"));
    }

    #[test]
    fn policy_strict_cross_ns() {
        let p = NamespacePolicy::Strict;
        assert!(!p.allows("ns1", "ns2"));
    }

    #[test]
    fn policy_permissive() {
        let p = NamespacePolicy::Permissive;
        assert!(p.allows("ns1", "ns2"));
        assert!(p.allows("a", "b"));
    }

    #[test]
    fn policy_selective_exact() {
        let p = NamespacePolicy::Selective(vec![
            ("frontend".into(), "backend".into()),
        ]);
        assert!(p.allows("frontend", "backend"));
        assert!(!p.allows("backend", "frontend"));
        assert!(p.allows("frontend", "frontend")); // same-ns always ok
    }

    #[test]
    fn policy_selective_wildcard_star() {
        let p = NamespacePolicy::Selective(vec![
            ("*".into(), "shared".into()),
        ]);
        assert!(p.allows("anything", "shared"));
        assert!(!p.allows("anything", "other"));
    }

    #[test]
    fn policy_selective_prefix_glob() {
        let p = NamespacePolicy::Selective(vec![
            ("prod-*".into(), "monitoring".into()),
        ]);
        assert!(p.allows("prod-us", "monitoring"));
        assert!(p.allows("prod-eu", "monitoring"));
        assert!(!p.allows("staging", "monitoring"));
    }

    #[test]
    fn policy_validate_ref_ok() {
        let p = NamespacePolicy::Permissive;
        let r = CrossNamespaceRef::new("a", "b", "Service", "api");
        assert!(p.validate_ref(&r).is_ok());
    }

    #[test]
    fn policy_validate_ref_denied() {
        let p = NamespacePolicy::Strict;
        let r = CrossNamespaceRef::new("a", "b", "Service", "api");
        let err = p.validate_ref(&r).unwrap_err();
        assert!(matches!(err, NamespaceError::CrossNamespaceDenied { .. }));
    }

    #[test]
    fn policy_description() {
        assert_eq!(
            NamespacePolicy::Strict.description(),
            "strict: no cross-namespace references"
        );
        assert_eq!(
            NamespacePolicy::Permissive.description(),
            "permissive: all cross-namespace references allowed"
        );
        let sel = NamespacePolicy::Selective(vec![("a".into(), "b".into())]);
        assert!(sel.description().contains("1 allowed pair"));
    }

    #[test]
    fn policy_pair_count() {
        assert_eq!(NamespacePolicy::Strict.pair_count(), 0);
        assert_eq!(NamespacePolicy::Permissive.pair_count(), 0);
        let sel = NamespacePolicy::Selective(vec![
            ("a".into(), "b".into()),
            ("c".into(), "d".into()),
        ]);
        assert_eq!(sel.pair_count(), 2);
    }

    #[test]
    fn policy_check_all_collects_errors() {
        let p = NamespacePolicy::Selective(vec![
            ("frontend".into(), "backend".into()),
        ]);
        let refs = vec![
            CrossNamespaceRef::new("frontend", "backend", "Service", "api"),   // ok
            CrossNamespaceRef::new("frontend", "frontend", "Service", "web"),  // same-ns ok
            CrossNamespaceRef::new("random", "backend", "Service", "rpc"),     // denied
            CrossNamespaceRef::new("frontend", "secrets", "Secret", "token"),  // denied
        ];
        let errors = p.check_all(&refs);
        assert_eq!(errors.len(), 2);
        for e in &errors {
            assert!(matches!(e, NamespaceError::CrossNamespaceDenied { .. }));
        }
    }

    #[test]
    fn policy_check_all_empty() {
        let p = NamespacePolicy::Strict;
        assert!(p.check_all(&[]).is_empty());
    }

    // --- collect_namespaces ------------------------------------------------

    #[test]
    fn collect_namespaces_deduplicates() {
        let refs = vec![
            CrossNamespaceRef::new("a", "b", "Service", "s1"),
            CrossNamespaceRef::new("b", "c", "Service", "s2"),
            CrossNamespaceRef::new("a", "c", "Service", "s3"),
        ];
        let ns = collect_namespaces(&refs);
        assert_eq!(ns.len(), 3);
        assert!(ns.contains("a"));
        assert!(ns.contains("b"));
        assert!(ns.contains("c"));
    }

    #[test]
    fn collect_namespaces_empty() {
        assert!(collect_namespaces(&[]).is_empty());
    }

    // --- group_by_source ---------------------------------------------------

    #[test]
    fn group_by_source_basic() {
        let refs = vec![
            CrossNamespaceRef::new("a", "b", "Service", "s1"),
            CrossNamespaceRef::new("a", "c", "Service", "s2"),
            CrossNamespaceRef::new("b", "c", "Service", "s3"),
        ];
        let grouped = group_by_source(&refs);
        assert_eq!(grouped.len(), 2);
        assert_eq!(grouped["a"].len(), 2);
        assert_eq!(grouped["b"].len(), 1);
    }

    #[test]
    fn group_by_source_empty() {
        assert!(group_by_source(&[]).is_empty());
    }

    // --- group_by_target ---------------------------------------------------

    #[test]
    fn group_by_target_basic() {
        let refs = vec![
            CrossNamespaceRef::new("a", "shared", "Service", "s1"),
            CrossNamespaceRef::new("b", "shared", "Service", "s2"),
            CrossNamespaceRef::new("c", "other", "Service", "s3"),
        ];
        let grouped = group_by_target(&refs);
        assert_eq!(grouped.len(), 2);
        assert_eq!(grouped["shared"].len(), 2);
        assert_eq!(grouped["other"].len(), 1);
    }

    #[test]
    fn group_by_target_empty() {
        assert!(group_by_target(&[]).is_empty());
    }

    // --- Serde round-trips -------------------------------------------------

    #[test]
    fn serde_roundtrip_resolver() {
        let mut r = NamespaceResolver::with_default("prod");
        r.add_override("Service", "web", "frontend");
        let json = serde_json::to_string(&r).unwrap();
        let r2: NamespaceResolver = serde_json::from_str(&json).unwrap();
        assert_eq!(r2.default_namespace, "prod");
        assert_eq!(r2.resolve("Service", "web"), "frontend");
    }

    #[test]
    fn serde_roundtrip_filter() {
        let f = NamespaceFilter::new(
            vec!["prod-*".into()],
            vec!["prod-secret".into()],
        );
        let json = serde_json::to_string(&f).unwrap();
        let f2: NamespaceFilter = serde_json::from_str(&json).unwrap();
        assert!(f2.matches("prod-us"));
        assert!(!f2.matches("prod-secret"));
    }

    #[test]
    fn serde_roundtrip_cross_ref() {
        let r = CrossNamespaceRef::new("a", "b", "Service", "web");
        let json = serde_json::to_string(&r).unwrap();
        let r2: CrossNamespaceRef = serde_json::from_str(&json).unwrap();
        assert_eq!(r, r2);
    }

    #[test]
    fn serde_roundtrip_policy() {
        let p = NamespacePolicy::Selective(vec![("a".into(), "b".into())]);
        let json = serde_json::to_string(&p).unwrap();
        let p2: NamespacePolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(p, p2);
    }

    // --- NamespaceError Display --------------------------------------------

    #[test]
    fn error_display_cross_namespace_denied() {
        let e = NamespaceError::CrossNamespaceDenied {
            source_ns: "a".into(),
            target_ns: "b".into(),
            kind: "Service".into(),
            name: "web".into(),
        };
        let msg = format!("{e}");
        assert!(msg.contains("a -> b"));
        assert!(msg.contains("Service/web"));
    }

    #[test]
    fn error_display_namespace_excluded() {
        let e = NamespaceError::NamespaceExcluded {
            namespace: "kube-system".into(),
        };
        assert!(format!("{e}").contains("kube-system"));
    }

    #[test]
    fn error_display_invalid_name() {
        let e = NamespaceError::InvalidName {
            name: "BAD".into(),
            reason: "uppercase".into(),
        };
        let msg = format!("{e}");
        assert!(msg.contains("BAD"));
        assert!(msg.contains("uppercase"));
    }

    #[test]
    fn error_display_missing_field() {
        let e = NamespaceError::MissingField {
            field: "source_ns".into(),
        };
        assert!(format!("{e}").contains("source_ns"));
    }

    // --- edge-case / integration tests -------------------------------------

    #[test]
    fn resolver_multiple_overrides_for_same_kind() {
        let mut r = NamespaceResolver::with_default("default");
        r.add_override("Deployment", "web", "frontend");
        r.add_override("Deployment", "api", "backend");
        r.add_override("Deployment", "*", "catch-all");
        assert_eq!(r.resolve("Deployment", "web"), "frontend");
        assert_eq!(r.resolve("Deployment", "api"), "backend");
        assert_eq!(r.resolve("Deployment", "unknown"), "catch-all");
        assert_eq!(r.override_count(), 3);
    }

    #[test]
    fn filter_include_multiple_globs() {
        let f = NamespaceFilter::new(
            vec!["prod-*".into(), "staging-*".into()],
            vec![],
        );
        assert!(f.matches("prod-us"));
        assert!(f.matches("staging-eu"));
        assert!(!f.matches("dev"));
    }

    #[test]
    fn policy_selective_multiple_pairs() {
        let p = NamespacePolicy::Selective(vec![
            ("frontend".into(), "backend".into()),
            ("backend".into(), "database".into()),
        ]);
        assert!(p.allows("frontend", "backend"));
        assert!(p.allows("backend", "database"));
        assert!(!p.allows("frontend", "database"));
    }

    #[test]
    fn cross_ref_validate_rejects_bad_target_ns() {
        let r = CrossNamespaceRef::new("good", "BAD-NS", "Service", "api");
        assert!(r.validate().is_err());
    }

    #[test]
    fn group_helpers_with_single_ref() {
        let refs = vec![CrossNamespaceRef::new("a", "b", "Service", "web")];
        let by_src = group_by_source(&refs);
        let by_tgt = group_by_target(&refs);
        assert_eq!(by_src.len(), 1);
        assert_eq!(by_tgt.len(), 1);
        assert_eq!(by_src["a"][0].resource_name, "web");
        assert_eq!(by_tgt["b"][0].resource_name, "web");
    }

    #[test]
    fn filter_empty_include_non_empty_exclude() {
        let f = NamespaceFilter::new(vec![], vec!["secret".into()]);
        assert!(f.matches("default"));
        assert!(f.matches("production"));
        assert!(!f.matches("secret"));
        assert!(!f.is_allow_all());
    }

    #[test]
    fn policy_check_all_strict_all_cross_ns() {
        let p = NamespacePolicy::Strict;
        let refs = vec![
            CrossNamespaceRef::new("a", "b", "Service", "s1"),
            CrossNamespaceRef::new("c", "d", "Service", "s2"),
        ];
        let errs = p.check_all(&refs);
        assert_eq!(errs.len(), 2);
    }

    #[test]
    fn policy_check_all_permissive_no_errors() {
        let p = NamespacePolicy::Permissive;
        let refs = vec![
            CrossNamespaceRef::new("a", "b", "Service", "s1"),
            CrossNamespaceRef::new("c", "d", "Service", "s2"),
        ];
        assert!(p.check_all(&refs).is_empty());
    }
}
