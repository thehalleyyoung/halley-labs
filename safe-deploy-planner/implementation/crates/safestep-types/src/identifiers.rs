// Type-safe identifiers for the SafeStep deployment planner.

use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;

/// A generic typed identifier. The phantom type parameter ensures that IDs for
/// different entity types are not accidentally interchanged.
#[derive(Serialize, Deserialize)]
#[serde(transparent)]
pub struct Id<T> {
    value: String,
    #[serde(skip)]
    _marker: PhantomData<fn() -> T>,
}

impl<T> Id<T> {
    /// Create a new identifier from a string value.
    pub fn new(value: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            _marker: PhantomData,
        }
    }

    /// Generate a new random UUID-based identifier.
    pub fn generate() -> Self {
        Self {
            value: Uuid::new_v4().to_string(),
            _marker: PhantomData,
        }
    }

    /// Generate a content-addressed identifier from the given bytes.
    pub fn from_content(content: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(content);
        let hash = hasher.finalize();
        Self {
            value: hex::encode(&hash[..16]),
            _marker: PhantomData,
        }
    }

    /// Generate from a string by hashing it.
    pub fn from_name(name: &str) -> Self {
        Self::from_content(name.as_bytes())
    }

    /// Return the raw string value.
    pub fn as_str(&self) -> &str {
        &self.value
    }

    /// Consume and return the inner string.
    pub fn into_inner(self) -> String {
        self.value
    }

    /// Check if this is a UUID-format identifier.
    pub fn is_uuid(&self) -> bool {
        Uuid::parse_str(&self.value).is_ok()
    }

    /// Check if this is a content-addressed (hex) identifier.
    pub fn is_content_addressed(&self) -> bool {
        self.value.len() == 32 && self.value.chars().all(|c| c.is_ascii_hexdigit())
    }

    /// Validate that the identifier matches expected format constraints.
    pub fn validate(&self) -> crate::error::Result<()> {
        if self.value.is_empty() {
            return Err(crate::error::SafeStepError::invalid_id(
                "identifier cannot be empty",
                &self.value,
            ));
        }
        if self.value.len() > 256 {
            return Err(crate::error::SafeStepError::invalid_id(
                "identifier too long (max 256 characters)",
                &self.value,
            ));
        }
        Ok(())
    }

    /// Cast this identifier to a different type. Use with caution.
    pub fn cast<U>(self) -> Id<U> {
        Id {
            value: self.value,
            _marker: PhantomData,
        }
    }
}

impl<T> Clone for Id<T> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            _marker: PhantomData,
        }
    }
}

impl<T> fmt::Debug for Id<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Id({})", self.value)
    }
}

impl<T> fmt::Display for Id<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl<T> PartialEq for Id<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T> Eq for Id<T> {}

impl<T> Hash for Id<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl<T> PartialOrd for Id<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for Id<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value.cmp(&other.value)
    }
}

impl<T> FromStr for Id<T> {
    type Err = crate::error::SafeStepError;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let id = Self::new(s);
        id.validate()?;
        Ok(id)
    }
}

impl<T> From<String> for Id<T> {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl<T> From<&str> for Id<T> {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl<T> AsRef<str> for Id<T> {
    fn as_ref(&self) -> &str {
        &self.value
    }
}

// ─── Domain-specific marker types ──────────────────────────────────────────

/// Marker type for service identifiers.
#[derive(Debug, Clone, Copy)]
pub struct ServiceTag;

/// Marker type for constraint identifiers.
#[derive(Debug, Clone, Copy)]
pub struct ConstraintTag;

/// Marker type for plan identifiers.
#[derive(Debug, Clone, Copy)]
pub struct PlanTag;

/// Marker type for step identifiers.
#[derive(Debug, Clone, Copy)]
pub struct StepTag;

/// Marker type for envelope identifiers.
#[derive(Debug, Clone, Copy)]
pub struct EnvelopeTag;

/// Marker type for state identifiers.
#[derive(Debug, Clone, Copy)]
pub struct StateTag;

/// Marker type for metric identifiers.
#[derive(Debug, Clone, Copy)]
pub struct MetricTag;

/// Unique identifier for a service.
pub type ServiceId = Id<ServiceTag>;

/// Unique identifier for a constraint.
pub type ConstraintId = Id<ConstraintTag>;

/// Unique identifier for a deployment plan.
pub type PlanId = Id<PlanTag>;

/// Unique identifier for a deployment step.
pub type StepId = Id<StepTag>;

/// Unique identifier for a safety envelope.
pub type EnvelopeId = Id<EnvelopeTag>;

/// Unique identifier for a cluster state.
pub type StateId = Id<StateTag>;

/// Unique identifier for a metric.
pub type MetricId = Id<MetricTag>;

// ─── ID collection helper ──────────────────────────────────────────────────

/// A set of identifiers for batch operations.
#[derive(Debug, Serialize, Deserialize)]
pub struct IdSet<T> {
    ids: Vec<Id<T>>,
}

impl<T> Clone for IdSet<T> {
    fn clone(&self) -> Self {
        Self {
            ids: self.ids.clone(),
        }
    }
}

impl<T> IdSet<T> {
    pub fn new() -> Self {
        Self { ids: Vec::new() }
    }

    pub fn from_vec(ids: Vec<Id<T>>) -> Self {
        Self { ids }
    }

    pub fn push(&mut self, id: Id<T>) {
        if !self.contains(&id) {
            self.ids.push(id);
        }
    }

    pub fn contains(&self, id: &Id<T>) -> bool {
        self.ids.iter().any(|i| i.value == id.value)
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Id<T>> {
        self.ids.iter()
    }

    pub fn into_vec(self) -> Vec<Id<T>> {
        self.ids
    }

    pub fn intersection(&self, other: &IdSet<T>) -> IdSet<T> {
        let ids = self
            .ids
            .iter()
            .filter(|id| other.contains(id))
            .cloned()
            .collect();
        IdSet { ids }
    }

    pub fn union(&self, other: &IdSet<T>) -> IdSet<T> {
        let mut result = self.clone();
        for id in &other.ids {
            if !result.contains(id) {
                result.ids.push(id.clone());
            }
        }
        result
    }

    pub fn difference(&self, other: &IdSet<T>) -> IdSet<T> {
        let ids = self
            .ids
            .iter()
            .filter(|id| !other.contains(id))
            .cloned()
            .collect();
        IdSet { ids }
    }
}

impl<T> Default for IdSet<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> IntoIterator for IdSet<T> {
    type Item = Id<T>;
    type IntoIter = std::vec::IntoIter<Id<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.ids.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a IdSet<T> {
    type Item = &'a Id<T>;
    type IntoIter = std::slice::Iter<'a, Id<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.ids.iter()
    }
}

impl<T> FromIterator<Id<T>> for IdSet<T> {
    fn from_iter<I: IntoIterator<Item = Id<T>>>(iter: I) -> Self {
        let mut set = IdSet::new();
        for id in iter {
            set.push(id);
        }
        set
    }
}

/// A mapping from identifiers to values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdMap<T, V> {
    entries: Vec<(Id<T>, V)>,
}

impl<T, V> IdMap<T, V> {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn insert(&mut self, id: Id<T>, value: V) -> Option<V> {
        for entry in &mut self.entries {
            if entry.0.value == id.value {
                let old = std::mem::replace(&mut entry.1, value);
                return Some(old);
            }
        }
        self.entries.push((id, value));
        None
    }

    pub fn get(&self, id: &Id<T>) -> Option<&V> {
        self.entries
            .iter()
            .find(|(k, _)| k.value == id.value)
            .map(|(_, v)| v)
    }

    pub fn get_mut(&mut self, id: &Id<T>) -> Option<&mut V> {
        self.entries
            .iter_mut()
            .find(|(k, _)| k.value == id.value)
            .map(|(_, v)| v)
    }

    pub fn remove(&mut self, id: &Id<T>) -> Option<V> {
        if let Some(pos) = self.entries.iter().position(|(k, _)| k.value == id.value) {
            Some(self.entries.remove(pos).1)
        } else {
            None
        }
    }

    pub fn contains_key(&self, id: &Id<T>) -> bool {
        self.entries.iter().any(|(k, _)| k.value == id.value)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn keys(&self) -> impl Iterator<Item = &Id<T>> {
        self.entries.iter().map(|(k, _)| k)
    }

    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.entries.iter().map(|(_, v)| v)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Id<T>, &V)> {
        self.entries.iter().map(|(k, v)| (k, v))
    }
}

impl<T, V> Default for IdMap<T, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, V> FromIterator<(Id<T>, V)> for IdMap<T, V> {
    fn from_iter<I: IntoIterator<Item = (Id<T>, V)>>(iter: I) -> Self {
        let mut map = IdMap::new();
        for (k, v) in iter {
            map.insert(k, v);
        }
        map
    }
}

/// Helper to create a namespaced identifier: "namespace/name".
pub fn namespaced_id<T>(namespace: &str, name: &str) -> Id<T> {
    Id::new(format!("{}/{}", namespace, name))
}

/// Helper to split a namespaced identifier.
pub fn split_namespaced<T>(id: &Id<T>) -> Option<(&str, &str)> {
    id.as_str().split_once('/')
}

/// Validate that a string is a valid DNS-compatible name (used for K8s identifiers).
pub fn validate_dns_name(name: &str) -> crate::error::Result<()> {
    if name.is_empty() {
        return Err(crate::error::SafeStepError::invalid_id(
            "name cannot be empty",
            name,
        ));
    }
    if name.len() > 253 {
        return Err(crate::error::SafeStepError::invalid_id(
            "name too long for DNS (max 253)",
            name,
        ));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '.')
    {
        return Err(crate::error::SafeStepError::invalid_id(
            "name must contain only lowercase alphanumeric, '-', or '.'",
            name,
        ));
    }
    if !name.chars().next().map_or(false, |c| c.is_ascii_alphanumeric()) {
        return Err(crate::error::SafeStepError::invalid_id(
            "name must start with alphanumeric character",
            name,
        ));
    }
    if !name.chars().last().map_or(false, |c| c.is_ascii_alphanumeric()) {
        return Err(crate::error::SafeStepError::invalid_id(
            "name must end with alphanumeric character",
            name,
        ));
    }
    Ok(())
}

/// Validate that a string is a valid Kubernetes label value.
pub fn validate_label_value(value: &str) -> crate::error::Result<()> {
    if value.len() > 63 {
        return Err(crate::error::SafeStepError::invalid_id(
            "label value too long (max 63)",
            value,
        ));
    }
    if value.is_empty() {
        return Ok(());
    }
    if !value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
    {
        return Err(crate::error::SafeStepError::invalid_id(
            "label value must contain only alphanumeric, '-', '_', or '.'",
            value,
        ));
    }
    if !value.chars().next().map_or(false, |c| c.is_ascii_alphanumeric()) {
        return Err(crate::error::SafeStepError::invalid_id(
            "label value must start with alphanumeric character",
            value,
        ));
    }
    if !value.chars().last().map_or(false, |c| c.is_ascii_alphanumeric()) {
        return Err(crate::error::SafeStepError::invalid_id(
            "label value must end with alphanumeric character",
            value,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_id_new() {
        let id: ServiceId = Id::new("my-service");
        assert_eq!(id.as_str(), "my-service");
    }

    #[test]
    fn test_id_generate() {
        let id: ServiceId = Id::generate();
        assert!(id.is_uuid());
        assert!(!id.is_content_addressed());
    }

    #[test]
    fn test_id_from_content() {
        let id1: ServiceId = Id::from_content(b"hello");
        let id2: ServiceId = Id::from_content(b"hello");
        let id3: ServiceId = Id::from_content(b"world");
        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
        assert!(id1.is_content_addressed());
    }

    #[test]
    fn test_id_from_name() {
        let id: ServiceId = Id::from_name("my-service");
        assert!(id.is_content_addressed());
    }

    #[test]
    fn test_id_equality() {
        let a: ServiceId = Id::new("abc");
        let b: ServiceId = Id::new("abc");
        let c: ServiceId = Id::new("xyz");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_id_ordering() {
        let a: ServiceId = Id::new("alpha");
        let b: ServiceId = Id::new("beta");
        assert!(a < b);
    }

    #[test]
    fn test_id_validate_empty() {
        let id: ServiceId = Id::new("");
        assert!(id.validate().is_err());
    }

    #[test]
    fn test_id_validate_too_long() {
        let long = "x".repeat(300);
        let id: ServiceId = Id::new(long);
        assert!(id.validate().is_err());
    }

    #[test]
    fn test_id_validate_ok() {
        let id: ServiceId = Id::new("my-service-123");
        assert!(id.validate().is_ok());
    }

    #[test]
    fn test_id_from_str() {
        let id: ServiceId = "my-svc".parse().unwrap();
        assert_eq!(id.as_str(), "my-svc");
    }

    #[test]
    fn test_id_from_str_empty() {
        let r: std::result::Result<ServiceId, _> = "".parse();
        assert!(r.is_err());
    }

    #[test]
    fn test_id_cast() {
        let svc: ServiceId = Id::new("svc-1");
        let constraint: ConstraintId = svc.cast();
        assert_eq!(constraint.as_str(), "svc-1");
    }

    #[test]
    fn test_id_display() {
        let id: PlanId = Id::new("plan-001");
        assert_eq!(format!("{}", id), "plan-001");
    }

    #[test]
    fn test_id_serialization() {
        let id: ServiceId = Id::new("svc-x");
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "\"svc-x\"");
        let parsed: ServiceId = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, id);
    }

    #[test]
    fn test_id_set() {
        let mut set: IdSet<ServiceTag> = IdSet::new();
        set.push(Id::new("a"));
        set.push(Id::new("b"));
        set.push(Id::new("a")); // duplicate
        assert_eq!(set.len(), 2);
        assert!(set.contains(&Id::new("a")));
        assert!(!set.contains(&Id::new("c")));
    }

    #[test]
    fn test_id_set_operations() {
        let s1: IdSet<ServiceTag> =
            vec![Id::new("a"), Id::new("b"), Id::new("c")].into_iter().collect();
        let s2: IdSet<ServiceTag> =
            vec![Id::new("b"), Id::new("c"), Id::new("d")].into_iter().collect();

        let inter = s1.intersection(&s2);
        assert_eq!(inter.len(), 2);

        let union = s1.union(&s2);
        assert_eq!(union.len(), 4);

        let diff = s1.difference(&s2);
        assert_eq!(diff.len(), 1);
        assert!(diff.contains(&Id::new("a")));
    }

    #[test]
    fn test_id_map() {
        let mut map: IdMap<ServiceTag, u32> = IdMap::new();
        assert!(map.is_empty());

        map.insert(Id::new("svc-a"), 10);
        map.insert(Id::new("svc-b"), 20);
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&Id::new("svc-a")), Some(&10));
        assert_eq!(map.get(&Id::new("svc-c")), None);

        let old = map.insert(Id::new("svc-a"), 30);
        assert_eq!(old, Some(10));
        assert_eq!(map.get(&Id::new("svc-a")), Some(&30));

        let removed = map.remove(&Id::new("svc-b"));
        assert_eq!(removed, Some(20));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_namespaced_id() {
        let id: ServiceId = namespaced_id("default", "nginx");
        assert_eq!(id.as_str(), "default/nginx");

        let (ns, name) = split_namespaced(&id).unwrap();
        assert_eq!(ns, "default");
        assert_eq!(name, "nginx");
    }

    #[test]
    fn test_split_namespaced_none() {
        let id: ServiceId = Id::new("no-namespace");
        assert!(split_namespaced(&id).is_none());
    }

    #[test]
    fn test_validate_dns_name_valid() {
        assert!(validate_dns_name("my-service").is_ok());
        assert!(validate_dns_name("a").is_ok());
        assert!(validate_dns_name("service-123.example").is_ok());
    }

    #[test]
    fn test_validate_dns_name_invalid() {
        assert!(validate_dns_name("").is_err());
        assert!(validate_dns_name("-starts-with-dash").is_err());
        assert!(validate_dns_name("ends-with-dash-").is_err());
        assert!(validate_dns_name("UPPERCASE").is_err());
        assert!(validate_dns_name("has space").is_err());
    }

    #[test]
    fn test_validate_label_value() {
        assert!(validate_label_value("").is_ok());
        assert!(validate_label_value("valid-value").is_ok());
        assert!(validate_label_value("v1.2.3").is_ok());
        assert!(validate_label_value("has space").is_err());
    }

    #[test]
    fn test_id_into_inner() {
        let id: ServiceId = Id::new("test");
        let s = id.into_inner();
        assert_eq!(s, "test");
    }

    #[test]
    fn test_id_map_iter() {
        let mut map: IdMap<ServiceTag, u32> = IdMap::new();
        map.insert(Id::new("a"), 1);
        map.insert(Id::new("b"), 2);
        let keys: Vec<_> = map.keys().collect();
        assert_eq!(keys.len(), 2);
        let values: Vec<_> = map.values().collect();
        assert_eq!(values.len(), 2);
    }

    #[test]
    fn test_id_set_into_iter() {
        let set: IdSet<ServiceTag> = vec![Id::new("x"), Id::new("y")].into_iter().collect();
        let v: Vec<_> = set.into_iter().collect();
        assert_eq!(v.len(), 2);
    }
}
