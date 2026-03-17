//! Container image reference parsing, policies, and version mapping.

use std::cmp::Ordering;
use std::fmt;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur when working with container images.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ImageError {
    #[error("invalid image reference: {0}")]
    InvalidReference(String),

    #[error("policy violation for image `{image}`: {reason}")]
    PolicyViolation { image: String, reason: String },

    #[error("failed to parse version from tag: {0}")]
    VersionParseFailed(String),
}

type Result<T> = std::result::Result<T, ImageError>;

// ---------------------------------------------------------------------------
// ContainerImage
// ---------------------------------------------------------------------------

/// A parsed container image reference such as `registry.io/repo/name:tag@sha256:abc`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContainerImage {
    /// Optional registry hostname (e.g. `docker.io`, `ghcr.io`, `localhost:5000`).
    pub registry: Option<String>,
    /// Repository path (e.g. `library/nginx`, `myorg/myapp`).
    pub repository: String,
    /// Optional tag (e.g. `1.21`, `latest`).
    pub tag: Option<String>,
    /// Optional digest (e.g. `sha256:abcdef0123456789...`).
    pub digest: Option<String>,
}

impl ContainerImage {
    /// Parse an image reference string into a [`ContainerImage`].
    ///
    /// Accepted formats:
    /// - `nginx`
    /// - `nginx:1.21`
    /// - `library/nginx:latest`
    /// - `registry.io/repo/name:tag`
    /// - `registry.io/repo/name@sha256:abc...`
    /// - `registry.io/repo/name:tag@sha256:abc...`
    /// - `localhost:5000/app:v1` (port in registry)
    pub fn parse(image_str: &str) -> Result<Self> {
        let s = image_str.trim();
        if s.is_empty() {
            return Err(ImageError::InvalidReference(
                "empty image reference".to_string(),
            ));
        }

        // Split off digest first (everything after `@`).
        let (before_digest, digest) = if let Some(idx) = s.find('@') {
            let d = &s[idx + 1..];
            if d.is_empty() {
                return Err(ImageError::InvalidReference(format!(
                    "empty digest in `{s}`"
                )));
            }
            (&s[..idx], Some(d.to_string()))
        } else {
            (s, None)
        };

        // Split off tag. The tag is the part after the *last* colon that is not
        // part of a registry port. We first separate the "name" portion
        // (registry + repository) from the tag.
        let (name_part, tag) = split_name_and_tag(before_digest);

        if name_part.is_empty() {
            return Err(ImageError::InvalidReference(format!(
                "empty repository in `{s}`"
            )));
        }

        // Decide whether the first path component is a registry.
        let components: Vec<&str> = name_part.split('/').collect();
        let (registry, repository) = if components.len() == 1 {
            // Single component like `nginx` — no registry.
            (None, components[0].to_string())
        } else {
            let first = components[0];
            if looks_like_registry(first) {
                (
                    Some(first.to_string()),
                    components[1..].join("/"),
                )
            } else {
                // e.g. `library/nginx`
                (None, name_part.to_string())
            }
        };

        if repository.is_empty() {
            return Err(ImageError::InvalidReference(format!(
                "empty repository in `{s}`"
            )));
        }

        Ok(ContainerImage {
            registry,
            repository,
            tag,
            digest,
        })
    }

    /// Returns `true` if a tag is present.
    pub fn has_tag(&self) -> bool {
        self.tag.is_some()
    }

    /// Returns `true` if a digest is present.
    pub fn has_digest(&self) -> bool {
        self.digest.is_some()
    }

    /// Full reference string (same as `Display`).
    pub fn full_name(&self) -> String {
        self.to_string()
    }

    /// Returns the tag if present, otherwise `"latest"`.
    pub fn effective_tag(&self) -> &str {
        self.tag.as_deref().unwrap_or("latest")
    }

    /// The image name without tag or digest — `registry/repository`.
    pub fn image_name(&self) -> String {
        match &self.registry {
            Some(r) => format!("{r}/{}", self.repository),
            None => self.repository.clone(),
        }
    }

    /// The last path component of the repository (e.g. `nginx` for
    /// `library/nginx`).
    pub fn short_name(&self) -> &str {
        self.repository
            .rsplit('/')
            .next()
            .unwrap_or(&self.repository)
    }
}

impl fmt::Display for ContainerImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref r) = self.registry {
            write!(f, "{r}/")?;
        }
        write!(f, "{}", self.repository)?;
        if let Some(ref t) = self.tag {
            write!(f, ":{t}")?;
        }
        if let Some(ref d) = self.digest {
            write!(f, "@{d}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers for parsing
// ---------------------------------------------------------------------------

/// Returns `true` if `component` looks like a registry hostname.
///
/// Heuristic: contains a `.` or `:`, or equals `localhost`.
fn looks_like_registry(component: &str) -> bool {
    component.contains('.') || component.contains(':') || component == "localhost"
}

/// Split a name portion (no digest) into (name, optional tag).
///
/// We need to be careful with registries that include a port
/// (e.g. `localhost:5000/app:v1`). The tag is the portion after the *last*
/// colon that appears after the *last* slash.
fn split_name_and_tag(name: &str) -> (&str, Option<String>) {
    // Find the last `/` — if there is one, the tag colon must be after it.
    let search_start = name.rfind('/').map_or(0, |i| i + 1);
    let tail = &name[search_start..];
    if let Some(colon_in_tail) = tail.rfind(':') {
        let colon_pos = search_start + colon_in_tail;
        let tag_str = &name[colon_pos + 1..];
        if tag_str.is_empty() {
            (name, None)
        } else {
            (&name[..colon_pos], Some(tag_str.to_string()))
        }
    } else {
        (name, None)
    }
}

// ---------------------------------------------------------------------------
// ImagePolicy
// ---------------------------------------------------------------------------

/// A policy that governs which container images are acceptable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagePolicy {
    /// Only images from these registries are allowed. Empty means all allowed.
    pub allowed_registries: Vec<String>,
    /// Regex pattern that tags must match (if set).
    pub tag_pattern: Option<String>,
    /// Whether a digest is required on every image.
    pub require_digest: bool,
    /// Tags that are explicitly blocked (e.g. `"latest"`).
    pub blocked_tags: Vec<String>,
    /// Whether a tag is required on every image.
    pub require_tag: bool,
}

impl ImagePolicy {
    /// A permissive policy that allows everything.
    pub fn permissive() -> Self {
        Self {
            allowed_registries: Vec::new(),
            tag_pattern: None,
            require_digest: false,
            blocked_tags: Vec::new(),
            require_tag: false,
        }
    }

    /// A strict policy that requires specific registries, a digest, and blocks
    /// the `latest` tag.
    pub fn strict(registries: Vec<String>) -> Self {
        Self {
            allowed_registries: registries,
            tag_pattern: None,
            require_digest: true,
            blocked_tags: vec!["latest".to_string()],
            require_tag: true,
        }
    }

    /// Returns `true` if the image is allowed under this policy (no
    /// violations).
    pub fn is_allowed(&self, image: &ContainerImage) -> bool {
        self.validate(image).is_empty()
    }

    /// Validate an image against this policy and return a list of human-
    /// readable violation messages. An empty vec means the image is acceptable.
    pub fn validate(&self, image: &ContainerImage) -> Vec<String> {
        let mut violations: Vec<String> = Vec::new();

        // Registry check
        if !self.allowed_registries.is_empty() {
            let reg = image.registry.as_deref().unwrap_or("");
            if !self.allowed_registries.iter().any(|r| r == reg) {
                violations.push(format!(
                    "registry `{reg}` is not in allowed list: {:?}",
                    self.allowed_registries,
                ));
            }
        }

        // Require digest
        if self.require_digest && !image.has_digest() {
            violations.push("image must have a digest".to_string());
        }

        // Require tag
        if self.require_tag && !image.has_tag() {
            violations.push("image must have a tag".to_string());
        }

        // Blocked tags
        if let Some(ref tag) = image.tag {
            if self.blocked_tags.iter().any(|b| b == tag) {
                violations.push(format!("tag `{tag}` is blocked"));
            }
        }

        // Tag pattern
        if let Some(ref pattern) = self.tag_pattern {
            if let Some(ref tag) = image.tag {
                if let Ok(re) = regex::Regex::new(pattern) {
                    if !re.is_match(tag) {
                        violations.push(format!(
                            "tag `{tag}` does not match required pattern `{pattern}`",
                        ));
                    }
                }
            }
        }

        violations
    }
}

// ---------------------------------------------------------------------------
// TagType
// ---------------------------------------------------------------------------

/// Classification of a container image tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TagType {
    /// Semantic version (e.g. `1.2.3`, `v2.0.0-alpine`).
    Semver,
    /// The `latest` tag.
    Latest,
    /// A Git SHA (e.g. `sha-abc1234` or a hex string of 7-40 chars).
    GitSha,
    /// Anything else.
    Other,
}

impl fmt::Display for TagType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TagType::Semver => write!(f, "semver"),
            TagType::Latest => write!(f, "latest"),
            TagType::GitSha => write!(f, "git-sha"),
            TagType::Other => write!(f, "other"),
        }
    }
}

// ---------------------------------------------------------------------------
// ImageVersionMapper
// ---------------------------------------------------------------------------

/// Utility for extracting, comparing, and classifying image tags.
pub struct ImageVersionMapper;

impl ImageVersionMapper {
    /// Attempt to extract a `(major, minor, patch)` triple from a tag.
    ///
    /// Strips a leading `v` or `V`, then parses numeric components separated
    /// by `.`. Supports:
    /// - `1.2.3` → `(1, 2, 3)`
    /// - `v1.2.3-alpine` → `(1, 2, 3)` (suffix after first non-numeric ignored)
    /// - `1.2` → `(1, 2, 0)`
    /// - `1` → `(1, 0, 0)`
    /// - `latest`, `sha-abc` → `None`
    pub fn extract_version(tag: &str) -> Option<(u64, u64, u64)> {
        let s = tag.strip_prefix('v').or_else(|| tag.strip_prefix('V')).unwrap_or(tag);

        if s.is_empty() {
            return None;
        }

        // The first character must be a digit.
        if !s.starts_with(|c: char| c.is_ascii_digit()) {
            return None;
        }

        // Take the numeric-dot prefix (also allow `-` separated suffixes like
        // `1.2.3-alpine`).
        let numeric_prefix: String = s
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect();

        let parts: Vec<&str> = numeric_prefix.split('.').collect();
        if parts.is_empty() || parts.len() > 3 {
            return None;
        }

        let major: u64 = parts[0].parse().ok()?;
        let minor: u64 = if parts.len() > 1 {
            parts[1].parse().ok()?
        } else {
            0
        };
        let patch: u64 = if parts.len() > 2 {
            parts[2].parse().ok()?
        } else {
            0
        };

        Some((major, minor, patch))
    }

    /// Compare two tags by semantic version. Tags that parse as semver are
    /// ordered by version; non-semver tags are ordered lexicographically after
    /// all semver tags.
    pub fn compare_tags(a: &str, b: &str) -> Ordering {
        let va = Self::extract_version(a);
        let vb = Self::extract_version(b);
        match (va, vb) {
            (Some(va), Some(vb)) => va.cmp(&vb),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => a.cmp(b),
        }
    }

    /// Returns `true` if the tag can be parsed as a semantic version.
    pub fn is_semver_tag(tag: &str) -> bool {
        Self::extract_version(tag).is_some()
    }

    /// Returns `true` if the tag is `latest` (case-insensitive).
    pub fn is_latest(tag: &str) -> bool {
        tag.eq_ignore_ascii_case("latest")
    }

    /// Returns `true` if the tag looks like a Git SHA.
    ///
    /// Matches:
    /// - `sha-` prefix followed by hex characters
    /// - A hex string of 7–40 characters
    pub fn is_git_sha(tag: &str) -> bool {
        if let Some(hex_part) = tag.strip_prefix("sha-") {
            return !hex_part.is_empty() && hex_part.chars().all(|c| c.is_ascii_hexdigit());
        }
        let len = tag.len();
        (7..=40).contains(&len) && tag.chars().all(|c| c.is_ascii_hexdigit())
    }

    /// Classify a tag into a [`TagType`].
    pub fn classify_tag(tag: &str) -> TagType {
        if Self::is_latest(tag) {
            TagType::Latest
        } else if Self::is_semver_tag(tag) {
            TagType::Semver
        } else if Self::is_git_sha(tag) {
            TagType::GitSha
        } else {
            TagType::Other
        }
    }

    /// Sort a slice of tags in ascending semver order. Non-semver tags are
    /// sorted lexicographically and placed after semver tags.
    pub fn sort_tags(tags: &mut [String]) {
        tags.sort_by(|a, b| Self::compare_tags(a, b));
    }
}

// ---------------------------------------------------------------------------
// RegistryInfo
// ---------------------------------------------------------------------------

/// Metadata about a container registry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegistryInfo {
    /// Registry URL or hostname (e.g. `docker.io`, `ghcr.io`).
    pub url: String,
    /// Whether this is a public registry that requires no authentication.
    pub is_public: bool,
    /// Name of the Kubernetes secret holding credentials, if any.
    pub credentials_secret: Option<String>,
}

impl RegistryInfo {
    /// Create info for a public registry.
    pub fn public(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            is_public: true,
            credentials_secret: None,
        }
    }

    /// Create info for a private registry that requires credentials.
    pub fn private(url: impl Into<String>, secret: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            is_public: false,
            credentials_secret: Some(secret.into()),
        }
    }

    /// Returns `true` if authentication is needed (not public or has a
    /// credentials secret).
    pub fn requires_auth(&self) -> bool {
        !self.is_public || self.credentials_secret.is_some()
    }

    /// Returns `true` if this registry matches the given hostname string.
    ///
    /// Comparison is case-insensitive and also matches if the input is a
    /// prefix of the URL.
    pub fn matches(&self, registry: &str) -> bool {
        let a = self.url.to_lowercase();
        let b = registry.to_lowercase();
        a == b || a.starts_with(&b) || b.starts_with(&a)
    }
}

// ---------------------------------------------------------------------------
// UpdateType & ImageUpdatePlan
// ---------------------------------------------------------------------------

/// The kind of change between two image references.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UpdateType {
    /// Only the tag changed.
    TagChange,
    /// Only the digest changed (or was added/removed).
    DigestChange,
    /// The registry is different.
    RegistryMigration,
    /// The two references are equivalent.
    NoChange,
}

impl fmt::Display for UpdateType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UpdateType::TagChange => write!(f, "tag change"),
            UpdateType::DigestChange => write!(f, "digest change"),
            UpdateType::RegistryMigration => write!(f, "registry migration"),
            UpdateType::NoChange => write!(f, "no change"),
        }
    }
}

/// A plan describing how one container image should be updated to another.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageUpdatePlan {
    /// The image currently in use.
    pub current: ContainerImage,
    /// The desired target image.
    pub target: ContainerImage,
    /// Classification of the update.
    pub update_type: UpdateType,
}

impl ImageUpdatePlan {
    /// Create a plan that classifies the update between `current` and
    /// `target`.
    pub fn plan(current: ContainerImage, target: ContainerImage) -> Self {
        let update_type = classify_update(&current, &target);
        Self {
            current,
            target,
            update_type,
        }
    }

    /// Human-readable summary of the planned update.
    pub fn summary(&self) -> String {
        match self.update_type {
            UpdateType::NoChange => format!(
                "no change required for {}",
                self.current.full_name(),
            ),
            UpdateType::TagChange => format!(
                "update {} from tag `{}` to `{}`",
                self.current.image_name(),
                self.current.effective_tag(),
                self.target.effective_tag(),
            ),
            UpdateType::DigestChange => format!(
                "update digest for {} from `{}` to `{}`",
                self.current.image_name(),
                self.current.digest.as_deref().unwrap_or("none"),
                self.target.digest.as_deref().unwrap_or("none"),
            ),
            UpdateType::RegistryMigration => format!(
                "migrate {} from registry `{}` to `{}`",
                self.current.repository,
                self.current.registry.as_deref().unwrap_or("(default)"),
                self.target.registry.as_deref().unwrap_or("(default)"),
            ),
        }
    }
}

fn classify_update(current: &ContainerImage, target: &ContainerImage) -> UpdateType {
    if current == target {
        return UpdateType::NoChange;
    }
    if current.registry != target.registry {
        return UpdateType::RegistryMigration;
    }
    if current.tag != target.tag {
        return UpdateType::TagChange;
    }
    if current.digest != target.digest {
        return UpdateType::DigestChange;
    }
    // Repository differs but that is essentially a different image; classify
    // as a tag change to keep things simple.
    UpdateType::TagChange
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ContainerImage::parse ------------------------------------------

    #[test]
    fn parse_simple_name() {
        let img = ContainerImage::parse("nginx").unwrap();
        assert_eq!(img.registry, None);
        assert_eq!(img.repository, "nginx");
        assert_eq!(img.tag, None);
        assert_eq!(img.digest, None);
    }

    #[test]
    fn parse_name_with_tag() {
        let img = ContainerImage::parse("nginx:1.21").unwrap();
        assert_eq!(img.repository, "nginx");
        assert_eq!(img.tag.as_deref(), Some("1.21"));
    }

    #[test]
    fn parse_library_path() {
        let img = ContainerImage::parse("library/nginx:latest").unwrap();
        assert_eq!(img.registry, None);
        assert_eq!(img.repository, "library/nginx");
        assert_eq!(img.tag.as_deref(), Some("latest"));
    }

    #[test]
    fn parse_full_registry() {
        let img = ContainerImage::parse("registry.io/repo/name:tag").unwrap();
        assert_eq!(img.registry.as_deref(), Some("registry.io"));
        assert_eq!(img.repository, "repo/name");
        assert_eq!(img.tag.as_deref(), Some("tag"));
    }

    #[test]
    fn parse_digest_only() {
        let img = ContainerImage::parse("registry.io/repo/name@sha256:abcdef").unwrap();
        assert_eq!(img.registry.as_deref(), Some("registry.io"));
        assert_eq!(img.repository, "repo/name");
        assert_eq!(img.tag, None);
        assert_eq!(img.digest.as_deref(), Some("sha256:abcdef"));
    }

    #[test]
    fn parse_tag_and_digest() {
        let img =
            ContainerImage::parse("registry.io/repo/name:v1.0@sha256:abcdef").unwrap();
        assert_eq!(img.tag.as_deref(), Some("v1.0"));
        assert_eq!(img.digest.as_deref(), Some("sha256:abcdef"));
    }

    #[test]
    fn parse_localhost_with_port() {
        let img = ContainerImage::parse("localhost:5000/app:v1").unwrap();
        assert_eq!(img.registry.as_deref(), Some("localhost:5000"));
        assert_eq!(img.repository, "app");
        assert_eq!(img.tag.as_deref(), Some("v1"));
    }

    #[test]
    fn parse_ghcr() {
        let img = ContainerImage::parse("ghcr.io/owner/repo:sha-abc123").unwrap();
        assert_eq!(img.registry.as_deref(), Some("ghcr.io"));
        assert_eq!(img.repository, "owner/repo");
        assert_eq!(img.tag.as_deref(), Some("sha-abc123"));
    }

    #[test]
    fn parse_empty_errors() {
        assert!(ContainerImage::parse("").is_err());
        assert!(ContainerImage::parse("   ").is_err());
    }

    #[test]
    fn parse_empty_digest_errors() {
        assert!(ContainerImage::parse("nginx@").is_err());
    }

    #[test]
    fn parse_deep_path() {
        let img = ContainerImage::parse("my.registry.io/a/b/c/d:1.0").unwrap();
        assert_eq!(img.registry.as_deref(), Some("my.registry.io"));
        assert_eq!(img.repository, "a/b/c/d");
        assert_eq!(img.tag.as_deref(), Some("1.0"));
    }

    // ---- ContainerImage helpers ------------------------------------------

    #[test]
    fn has_tag_and_digest() {
        let img = ContainerImage::parse("nginx:1.21").unwrap();
        assert!(img.has_tag());
        assert!(!img.has_digest());

        let img2 = ContainerImage::parse("nginx@sha256:abc").unwrap();
        assert!(!img2.has_tag());
        assert!(img2.has_digest());
    }

    #[test]
    fn effective_tag_defaults_to_latest() {
        let img = ContainerImage::parse("nginx").unwrap();
        assert_eq!(img.effective_tag(), "latest");
        let img2 = ContainerImage::parse("nginx:alpine").unwrap();
        assert_eq!(img2.effective_tag(), "alpine");
    }

    #[test]
    fn full_name_equals_display() {
        let img = ContainerImage::parse("ghcr.io/org/app:v2.0@sha256:abc").unwrap();
        assert_eq!(img.full_name(), img.to_string());
        assert_eq!(img.full_name(), "ghcr.io/org/app:v2.0@sha256:abc");
    }

    #[test]
    fn image_name_excludes_tag_digest() {
        let img = ContainerImage::parse("ghcr.io/org/app:v2.0@sha256:abc").unwrap();
        assert_eq!(img.image_name(), "ghcr.io/org/app");
    }

    #[test]
    fn short_name_extracts_last_component() {
        let img = ContainerImage::parse("ghcr.io/org/deep/path/myapp:v1").unwrap();
        assert_eq!(img.short_name(), "myapp");
        let img2 = ContainerImage::parse("nginx").unwrap();
        assert_eq!(img2.short_name(), "nginx");
    }

    // ---- ImagePolicy ----------------------------------------------------

    #[test]
    fn permissive_allows_everything() {
        let policy = ImagePolicy::permissive();
        let img = ContainerImage::parse("anything:latest").unwrap();
        assert!(policy.is_allowed(&img));
    }

    #[test]
    fn strict_requires_registry() {
        let policy = ImagePolicy::strict(vec!["ghcr.io".to_string()]);
        let img = ContainerImage::parse("nginx:1.21").unwrap();
        let violations = policy.validate(&img);
        assert!(violations.iter().any(|v| v.contains("registry")));
    }

    #[test]
    fn strict_blocks_latest() {
        let policy = ImagePolicy::strict(vec!["ghcr.io".to_string()]);
        let img = ContainerImage::parse("ghcr.io/app:latest@sha256:abc").unwrap();
        let violations = policy.validate(&img);
        assert!(violations.iter().any(|v| v.contains("blocked")));
    }

    #[test]
    fn strict_requires_digest() {
        let policy = ImagePolicy::strict(vec!["ghcr.io".to_string()]);
        let img = ContainerImage::parse("ghcr.io/app:v1.0").unwrap();
        let violations = policy.validate(&img);
        assert!(violations.iter().any(|v| v.contains("digest")));
    }

    #[test]
    fn strict_valid_image() {
        let policy = ImagePolicy::strict(vec!["ghcr.io".to_string()]);
        let img =
            ContainerImage::parse("ghcr.io/app:v1.0@sha256:abcdef0123456789").unwrap();
        assert!(policy.is_allowed(&img));
    }

    #[test]
    fn policy_tag_pattern() {
        let mut policy = ImagePolicy::permissive();
        policy.tag_pattern = Some(r"^v\d+\.\d+\.\d+$".to_string());
        let good = ContainerImage::parse("app:v1.2.3").unwrap();
        assert!(policy.is_allowed(&good));
        let bad = ContainerImage::parse("app:latest").unwrap();
        assert!(!policy.is_allowed(&bad));
    }

    #[test]
    fn policy_require_tag() {
        let mut policy = ImagePolicy::permissive();
        policy.require_tag = true;
        let no_tag = ContainerImage::parse("nginx").unwrap();
        assert!(!policy.is_allowed(&no_tag));
        let with_tag = ContainerImage::parse("nginx:1.0").unwrap();
        assert!(policy.is_allowed(&with_tag));
    }

    #[test]
    fn policy_multiple_violations() {
        let policy = ImagePolicy::strict(vec!["acr.io".to_string()]);
        let img = ContainerImage::parse("nginx:latest").unwrap();
        let violations = policy.validate(&img);
        // Should have: wrong registry, blocked tag, missing digest
        assert!(violations.len() >= 3, "expected >=3 violations, got: {violations:?}");
    }

    // ---- ImageVersionMapper ---------------------------------------------

    #[test]
    fn extract_version_basic() {
        assert_eq!(
            ImageVersionMapper::extract_version("1.2.3"),
            Some((1, 2, 3))
        );
    }

    #[test]
    fn extract_version_v_prefix() {
        assert_eq!(
            ImageVersionMapper::extract_version("v1.2.3"),
            Some((1, 2, 3))
        );
    }

    #[test]
    fn extract_version_two_parts() {
        assert_eq!(
            ImageVersionMapper::extract_version("1.2"),
            Some((1, 2, 0))
        );
    }

    #[test]
    fn extract_version_one_part() {
        assert_eq!(ImageVersionMapper::extract_version("3"), Some((3, 0, 0)));
    }

    #[test]
    fn extract_version_with_suffix() {
        assert_eq!(
            ImageVersionMapper::extract_version("1.2.3-alpine"),
            Some((1, 2, 3))
        );
    }

    #[test]
    fn extract_version_latest_returns_none() {
        assert_eq!(ImageVersionMapper::extract_version("latest"), None);
    }

    #[test]
    fn extract_version_sha_returns_none() {
        assert_eq!(ImageVersionMapper::extract_version("sha-abc123"), None);
    }

    #[test]
    fn extract_version_uppercase_v() {
        assert_eq!(
            ImageVersionMapper::extract_version("V2.0.1"),
            Some((2, 0, 1))
        );
    }

    #[test]
    fn compare_tags_semver() {
        assert_eq!(
            ImageVersionMapper::compare_tags("1.0.0", "2.0.0"),
            Ordering::Less
        );
        assert_eq!(
            ImageVersionMapper::compare_tags("1.2.3", "1.2.3"),
            Ordering::Equal
        );
        assert_eq!(
            ImageVersionMapper::compare_tags("v3.0.0", "v2.9.9"),
            Ordering::Greater
        );
    }

    #[test]
    fn compare_tags_semver_before_non_semver() {
        assert_eq!(
            ImageVersionMapper::compare_tags("1.0.0", "latest"),
            Ordering::Less
        );
        assert_eq!(
            ImageVersionMapper::compare_tags("latest", "1.0.0"),
            Ordering::Greater
        );
    }

    #[test]
    fn compare_tags_non_semver_lexicographic() {
        assert_eq!(
            ImageVersionMapper::compare_tags("alpha", "beta"),
            Ordering::Less
        );
    }

    #[test]
    fn is_semver_tag_cases() {
        assert!(ImageVersionMapper::is_semver_tag("1.2.3"));
        assert!(ImageVersionMapper::is_semver_tag("v0.1.0"));
        assert!(!ImageVersionMapper::is_semver_tag("latest"));
        assert!(!ImageVersionMapper::is_semver_tag("sha-abc"));
    }

    #[test]
    fn is_latest_cases() {
        assert!(ImageVersionMapper::is_latest("latest"));
        assert!(ImageVersionMapper::is_latest("LATEST"));
        assert!(!ImageVersionMapper::is_latest("v1.0"));
    }

    #[test]
    fn is_git_sha_prefix() {
        assert!(ImageVersionMapper::is_git_sha("sha-abc1234"));
        assert!(ImageVersionMapper::is_git_sha("sha-0123456789abcdef"));
        assert!(!ImageVersionMapper::is_git_sha("sha-")); // empty hex
    }

    #[test]
    fn is_git_sha_hex_string() {
        assert!(ImageVersionMapper::is_git_sha("abcdef0")); // 7 chars
        assert!(ImageVersionMapper::is_git_sha("abcdef0123456789abcdef0123456789abcdef01")); // 40
        assert!(!ImageVersionMapper::is_git_sha("abcde")); // too short (5)
        assert!(!ImageVersionMapper::is_git_sha("xyz1234")); // non-hex
    }

    #[test]
    fn classify_tag_cases() {
        assert_eq!(
            ImageVersionMapper::classify_tag("v1.2.3"),
            TagType::Semver
        );
        assert_eq!(
            ImageVersionMapper::classify_tag("latest"),
            TagType::Latest
        );
        assert_eq!(
            ImageVersionMapper::classify_tag("sha-abc1234"),
            TagType::GitSha
        );
        assert_eq!(
            ImageVersionMapper::classify_tag("alpine"),
            TagType::Other
        );
    }

    #[test]
    fn sort_tags_ascending() {
        let mut tags = vec![
            "v2.0.0".to_string(),
            "latest".to_string(),
            "v1.0.0".to_string(),
            "v1.5.0".to_string(),
            "sha-abc".to_string(),
        ];
        ImageVersionMapper::sort_tags(&mut tags);
        assert_eq!(tags[0], "v1.0.0");
        assert_eq!(tags[1], "v1.5.0");
        assert_eq!(tags[2], "v2.0.0");
        // Non-semver sorted lexicographically after
        assert_eq!(tags[3], "latest");
        assert_eq!(tags[4], "sha-abc");
    }

    // ---- RegistryInfo ---------------------------------------------------

    #[test]
    fn registry_public() {
        let info = RegistryInfo::public("docker.io");
        assert!(info.is_public);
        assert!(info.credentials_secret.is_none());
        assert!(!info.requires_auth());
    }

    #[test]
    fn registry_private() {
        let info = RegistryInfo::private("ghcr.io", "ghcr-secret");
        assert!(!info.is_public);
        assert_eq!(info.credentials_secret.as_deref(), Some("ghcr-secret"));
        assert!(info.requires_auth());
    }

    #[test]
    fn registry_matches() {
        let info = RegistryInfo::public("ghcr.io");
        assert!(info.matches("ghcr.io"));
        assert!(info.matches("GHCR.IO"));
        assert!(!info.matches("docker.io"));
    }

    #[test]
    fn registry_matches_prefix() {
        let info = RegistryInfo::public("my.registry.io/v2");
        assert!(info.matches("my.registry.io"));
    }

    // ---- ImageUpdatePlan ------------------------------------------------

    #[test]
    fn plan_no_change() {
        let a = ContainerImage::parse("ghcr.io/app:v1.0").unwrap();
        let b = ContainerImage::parse("ghcr.io/app:v1.0").unwrap();
        let plan = ImageUpdatePlan::plan(a, b);
        assert_eq!(plan.update_type, UpdateType::NoChange);
        assert!(plan.summary().contains("no change"));
    }

    #[test]
    fn plan_tag_change() {
        let a = ContainerImage::parse("ghcr.io/app:v1.0").unwrap();
        let b = ContainerImage::parse("ghcr.io/app:v2.0").unwrap();
        let plan = ImageUpdatePlan::plan(a, b);
        assert_eq!(plan.update_type, UpdateType::TagChange);
        assert!(plan.summary().contains("v1.0"));
        assert!(plan.summary().contains("v2.0"));
    }

    #[test]
    fn plan_digest_change() {
        let a = ContainerImage::parse("ghcr.io/app:v1.0@sha256:aaa").unwrap();
        let b = ContainerImage::parse("ghcr.io/app:v1.0@sha256:bbb").unwrap();
        let plan = ImageUpdatePlan::plan(a, b);
        assert_eq!(plan.update_type, UpdateType::DigestChange);
        assert!(plan.summary().contains("digest"));
    }

    #[test]
    fn plan_registry_migration() {
        let a = ContainerImage::parse("docker.io/app:v1.0").unwrap();
        let b = ContainerImage::parse("ghcr.io/app:v1.0").unwrap();
        let plan = ImageUpdatePlan::plan(a, b);
        assert_eq!(plan.update_type, UpdateType::RegistryMigration);
        assert!(plan.summary().contains("migrate"));
    }

    // ---- Serialization --------------------------------------------------

    #[test]
    fn container_image_roundtrip_json() {
        let img = ContainerImage::parse("ghcr.io/org/app:v1.0@sha256:abc").unwrap();
        let json = serde_json::to_string(&img).unwrap();
        let back: ContainerImage = serde_json::from_str(&json).unwrap();
        assert_eq!(img, back);
    }

    #[test]
    fn image_policy_roundtrip_json() {
        let policy = ImagePolicy::strict(vec!["ghcr.io".to_string()]);
        let json = serde_json::to_string(&policy).unwrap();
        let back: ImagePolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(back.allowed_registries, vec!["ghcr.io"]);
        assert!(back.require_digest);
    }

    #[test]
    fn registry_info_roundtrip_json() {
        let info = RegistryInfo::private("ecr.aws", "ecr-creds");
        let json = serde_json::to_string(&info).unwrap();
        let back: RegistryInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(info, back);
    }

    #[test]
    fn update_plan_roundtrip_json() {
        let a = ContainerImage::parse("docker.io/app:v1").unwrap();
        let b = ContainerImage::parse("ghcr.io/app:v2").unwrap();
        let plan = ImageUpdatePlan::plan(a, b);
        let json = serde_json::to_string(&plan).unwrap();
        let back: ImageUpdatePlan = serde_json::from_str(&json).unwrap();
        assert_eq!(plan, back);
    }

    #[test]
    fn tag_type_display() {
        assert_eq!(TagType::Semver.to_string(), "semver");
        assert_eq!(TagType::Latest.to_string(), "latest");
        assert_eq!(TagType::GitSha.to_string(), "git-sha");
        assert_eq!(TagType::Other.to_string(), "other");
    }

    #[test]
    fn update_type_display() {
        assert_eq!(UpdateType::TagChange.to_string(), "tag change");
        assert_eq!(UpdateType::NoChange.to_string(), "no change");
    }

    // ---- Edge-case parsing tests -----------------------------------------

    #[test]
    fn parse_registry_with_port_and_deep_path() {
        let img = ContainerImage::parse("myhost:8080/a/b/c:latest").unwrap();
        assert_eq!(img.registry.as_deref(), Some("myhost:8080"));
        assert_eq!(img.repository, "a/b/c");
        assert_eq!(img.tag.as_deref(), Some("latest"));
    }

    #[test]
    fn parse_localhost_no_port() {
        let img = ContainerImage::parse("localhost/myapp:v1").unwrap();
        assert_eq!(img.registry.as_deref(), Some("localhost"));
        assert_eq!(img.repository, "myapp");
    }

    #[test]
    fn display_no_tag_no_digest() {
        let img = ContainerImage::parse("library/nginx").unwrap();
        assert_eq!(img.to_string(), "library/nginx");
    }

    #[test]
    fn display_full() {
        let img =
            ContainerImage::parse("ghcr.io/org/app:v2.0@sha256:deadbeef").unwrap();
        assert_eq!(img.to_string(), "ghcr.io/org/app:v2.0@sha256:deadbeef");
    }

    #[test]
    fn extract_version_empty_tag() {
        assert_eq!(ImageVersionMapper::extract_version(""), None);
        assert_eq!(ImageVersionMapper::extract_version("v"), None);
    }

    #[test]
    fn sort_tags_all_non_semver() {
        let mut tags = vec!["gamma".to_string(), "alpha".to_string(), "beta".to_string()];
        ImageVersionMapper::sort_tags(&mut tags);
        assert_eq!(tags, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn sort_tags_all_semver() {
        let mut tags = vec![
            "3.0.0".to_string(),
            "1.0.0".to_string(),
            "2.0.0".to_string(),
        ];
        ImageVersionMapper::sort_tags(&mut tags);
        assert_eq!(tags, vec!["1.0.0", "2.0.0", "3.0.0"]);
    }

    #[test]
    fn plan_add_digest() {
        let a = ContainerImage::parse("ghcr.io/app:v1.0").unwrap();
        let b = ContainerImage::parse("ghcr.io/app:v1.0@sha256:abc").unwrap();
        let plan = ImageUpdatePlan::plan(a, b);
        assert_eq!(plan.update_type, UpdateType::DigestChange);
    }

    #[test]
    fn classify_hex_exactly_7() {
        assert!(ImageVersionMapper::is_git_sha("a1b2c3d"));
    }

    #[test]
    fn classify_hex_6_not_sha() {
        assert!(!ImageVersionMapper::is_git_sha("a1b2c3"));
    }
}
