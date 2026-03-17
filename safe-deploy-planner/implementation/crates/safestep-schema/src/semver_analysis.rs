//! Semantic versioning analysis for compatibility hints.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::str::FromStr;

/// A parsed semantic version.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SemVer {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
    pub pre_release: Option<String>,
    pub build_metadata: Option<String>,
}

impl SemVer {
    pub fn new(major: u64, minor: u64, patch: u64) -> Self {
        Self {
            major,
            minor,
            patch,
            pre_release: None,
            build_metadata: None,
        }
    }

    pub fn with_pre_release(mut self, pre: &str) -> Self {
        self.pre_release = Some(pre.to_string());
        self
    }

    pub fn with_build(mut self, build: &str) -> Self {
        self.build_metadata = Some(build.to_string());
        self
    }

    pub fn is_pre_release(&self) -> bool {
        self.pre_release.is_some()
    }

    pub fn is_stable(&self) -> bool {
        self.major > 0 && self.pre_release.is_none()
    }
}

impl fmt::Display for SemVer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if let Some(ref pre) = self.pre_release {
            write!(f, "-{}", pre)?;
        }
        if let Some(ref build) = self.build_metadata {
            write!(f, "+{}", build)?;
        }
        Ok(())
    }
}

impl PartialOrd for SemVer {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SemVer {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.major
            .cmp(&other.major)
            .then(self.minor.cmp(&other.minor))
            .then(self.patch.cmp(&other.patch))
            .then_with(|| match (&self.pre_release, &other.pre_release) {
                (None, None) => std::cmp::Ordering::Equal,
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (Some(a), Some(b)) => a.cmp(b),
            })
    }
}

impl FromStr for SemVer {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim().trim_start_matches('v');
        let (version_part, build_metadata) = if let Some(idx) = s.find('+') {
            (&s[..idx], Some(s[idx + 1..].to_string()))
        } else {
            (s, None)
        };

        let (version_core, pre_release) = if let Some(idx) = version_part.find('-') {
            (&version_part[..idx], Some(version_part[idx + 1..].to_string()))
        } else {
            (version_part, None)
        };

        let parts: Vec<&str> = version_core.split('.').collect();
        if parts.len() < 2 || parts.len() > 3 {
            return Err(format!("Invalid semver: {}", s));
        }

        let major = parts[0]
            .parse::<u64>()
            .map_err(|e| format!("Invalid major: {}", e))?;
        let minor = parts[1]
            .parse::<u64>()
            .map_err(|e| format!("Invalid minor: {}", e))?;
        let patch = if parts.len() == 3 {
            parts[2]
                .parse::<u64>()
                .map_err(|e| format!("Invalid patch: {}", e))?
        } else {
            0
        };

        Ok(SemVer {
            major,
            minor,
            patch,
            pre_release,
            build_metadata,
        })
    }
}

/// Compatibility hint derived from semver comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CompatibilityHint {
    LikelyBreaking,
    LikelyBackwardCompatible,
    LikelyFullyCompatible,
    Identical,
    Unknown,
}

impl fmt::Display for CompatibilityHint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LikelyBreaking => write!(f, "likely_breaking"),
            Self::LikelyBackwardCompatible => write!(f, "likely_backward_compatible"),
            Self::LikelyFullyCompatible => write!(f, "likely_fully_compatible"),
            Self::Identical => write!(f, "identical"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Analyzes semantic versioning for compatibility hints.
#[derive(Debug, Clone)]
pub struct SemverAnalyzer;

impl SemverAnalyzer {
    pub fn new() -> Self {
        Self
    }

    /// Compare two versions and return a compatibility hint.
    pub fn analyze(&self, old: &SemVer, new: &SemVer) -> CompatibilityHint {
        if old == new {
            return CompatibilityHint::Identical;
        }

        if new.major != old.major {
            // Major bump (or downgrade) -> likely breaking
            if new.major > old.major {
                return CompatibilityHint::LikelyBreaking;
            }
            // Downgrade of major version
            return CompatibilityHint::LikelyBreaking;
        }

        if new.minor != old.minor {
            return CompatibilityHint::LikelyBackwardCompatible;
        }

        if new.patch != old.patch {
            return CompatibilityHint::LikelyFullyCompatible;
        }

        // Only pre-release or build metadata changed
        if new.pre_release != old.pre_release {
            return CompatibilityHint::Unknown;
        }

        CompatibilityHint::Identical
    }

    /// Parse two version strings and analyze.
    pub fn analyze_versions(&self, old: &str, new: &str) -> Result<CompatibilityHint, String> {
        let old_v = old.parse::<SemVer>()?;
        let new_v = new.parse::<SemVer>()?;
        Ok(self.analyze(&old_v, &new_v))
    }

    /// Check if a version bump is major.
    pub fn is_major_bump(&self, old: &SemVer, new: &SemVer) -> bool {
        new.major > old.major
    }

    /// Check if a version bump is minor (same major).
    pub fn is_minor_bump(&self, old: &SemVer, new: &SemVer) -> bool {
        old.major == new.major && new.minor > old.minor
    }

    /// Check if a version bump is patch (same major+minor).
    pub fn is_patch_bump(&self, old: &SemVer, new: &SemVer) -> bool {
        old.major == new.major && old.minor == new.minor && new.patch > old.patch
    }

    /// Confidence for the hint based on semver.
    pub fn confidence(&self, old: &SemVer, new: &SemVer) -> f64 {
        match self.analyze(old, new) {
            CompatibilityHint::Identical => 0.99,
            CompatibilityHint::LikelyFullyCompatible => 0.85,
            CompatibilityHint::LikelyBackwardCompatible => 0.70,
            CompatibilityHint::LikelyBreaking => 0.60,
            CompatibilityHint::Unknown => 0.40,
        }
    }
}

impl Default for SemverAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Version range specification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VersionRange {
    /// Exact match: =1.2.3
    Exact(SemVer),
    /// Caret: ^1.2.3 (compatible with 1.x.x)
    Caret(SemVer),
    /// Tilde: ~1.2.3 (compatible with 1.2.x)
    Tilde(SemVer),
    /// Greater-or-equal: >=1.2.3
    Gte(SemVer),
    /// Less-than: <2.0.0
    Lt(SemVer),
    /// Range: >=1.2.3, <2.0.0
    Range { gte: SemVer, lt: SemVer },
    /// Wildcard: 1.2.* or *
    Wildcard { major: Option<u64>, minor: Option<u64> },
    /// Any version
    Any,
}

impl fmt::Display for VersionRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact(v) => write!(f, "={}", v),
            Self::Caret(v) => write!(f, "^{}", v),
            Self::Tilde(v) => write!(f, "~{}", v),
            Self::Gte(v) => write!(f, ">={}", v),
            Self::Lt(v) => write!(f, "<{}", v),
            Self::Range { gte, lt } => write!(f, ">={}, <{}", gte, lt),
            Self::Wildcard { major: Some(maj), minor: Some(min) } => {
                write!(f, "{}.{}.*", maj, min)
            }
            Self::Wildcard { major: Some(maj), minor: None } => write!(f, "{}.*", maj),
            Self::Wildcard { .. } => write!(f, "*"),
            Self::Any => write!(f, "*"),
        }
    }
}

/// Analyzes dependency version ranges.
#[derive(Debug, Clone)]
pub struct VersionRangeAnalyzer;

impl VersionRangeAnalyzer {
    pub fn new() -> Self {
        Self
    }

    /// Parse a version range specification string.
    pub fn parse_range(&self, spec: &str) -> Result<VersionRange, String> {
        let spec = spec.trim();

        if spec == "*" {
            return Ok(VersionRange::Any);
        }

        // Check for wildcard patterns like 1.* or 1.2.*
        if spec.contains('*') && !spec.starts_with('^') && !spec.starts_with('~') {
            return self.parse_wildcard(spec);
        }

        // Caret range
        if let Some(rest) = spec.strip_prefix('^') {
            let v = rest.parse::<SemVer>()?;
            return Ok(VersionRange::Caret(v));
        }

        // Tilde range
        if let Some(rest) = spec.strip_prefix('~') {
            let v = rest.parse::<SemVer>()?;
            return Ok(VersionRange::Tilde(v));
        }

        // Exact match
        if let Some(rest) = spec.strip_prefix('=') {
            let v = rest.trim().parse::<SemVer>()?;
            return Ok(VersionRange::Exact(v));
        }

        // Combined range: >=X, <Y
        if spec.contains(',') {
            let parts: Vec<&str> = spec.split(',').map(|s| s.trim()).collect();
            if parts.len() == 2 {
                let gte = parts[0]
                    .strip_prefix(">=")
                    .ok_or_else(|| format!("Expected >= in range: {}", parts[0]))?
                    .trim()
                    .parse::<SemVer>()?;
                let lt = parts[1]
                    .strip_prefix('<')
                    .ok_or_else(|| format!("Expected < in range: {}", parts[1]))?
                    .trim()
                    .parse::<SemVer>()?;
                return Ok(VersionRange::Range { gte, lt });
            }
        }

        // >= only
        if let Some(rest) = spec.strip_prefix(">=") {
            let v = rest.trim().parse::<SemVer>()?;
            return Ok(VersionRange::Gte(v));
        }

        // < only
        if let Some(rest) = spec.strip_prefix('<') {
            let v = rest.trim().parse::<SemVer>()?;
            return Ok(VersionRange::Lt(v));
        }

        // Bare version -> exact
        if let Ok(v) = spec.parse::<SemVer>() {
            return Ok(VersionRange::Exact(v));
        }

        Err(format!("Unrecognized version range spec: {}", spec))
    }

    fn parse_wildcard(&self, spec: &str) -> Result<VersionRange, String> {
        let parts: Vec<&str> = spec.split('.').collect();
        match parts.len() {
            1 => Ok(VersionRange::Wildcard {
                major: None,
                minor: None,
            }),
            2 => {
                let major = parts[0]
                    .parse::<u64>()
                    .map_err(|e| format!("Invalid major in wildcard: {}", e))?;
                Ok(VersionRange::Wildcard {
                    major: Some(major),
                    minor: None,
                })
            }
            3 => {
                let major = parts[0]
                    .parse::<u64>()
                    .map_err(|e| format!("Invalid major in wildcard: {}", e))?;
                let minor_str = parts[1];
                if minor_str == "*" {
                    Ok(VersionRange::Wildcard {
                        major: Some(major),
                        minor: None,
                    })
                } else {
                    let minor = minor_str
                        .parse::<u64>()
                        .map_err(|e| format!("Invalid minor in wildcard: {}", e))?;
                    Ok(VersionRange::Wildcard {
                        major: Some(major),
                        minor: Some(minor),
                    })
                }
            }
            _ => Err(format!("Invalid wildcard pattern: {}", spec)),
        }
    }

    /// Check if a version satisfies a range.
    pub fn is_compatible(&self, version: &SemVer, range: &VersionRange) -> bool {
        match range {
            VersionRange::Any => true,
            VersionRange::Exact(v) => {
                version.major == v.major
                    && version.minor == v.minor
                    && version.patch == v.patch
            }
            VersionRange::Caret(v) => {
                if v.major == 0 {
                    if v.minor == 0 {
                        // ^0.0.x means exact patch
                        version.major == 0
                            && version.minor == 0
                            && version.patch == v.patch
                    } else {
                        // ^0.y.z means >=0.y.z, <0.(y+1).0
                        version.major == 0
                            && version.minor == v.minor
                            && version.patch >= v.patch
                    }
                } else {
                    // ^x.y.z means >=x.y.z, <(x+1).0.0
                    version.major == v.major
                        && (version.minor > v.minor
                            || (version.minor == v.minor && version.patch >= v.patch))
                }
            }
            VersionRange::Tilde(v) => {
                // ~x.y.z means >=x.y.z, <x.(y+1).0
                version.major == v.major
                    && version.minor == v.minor
                    && version.patch >= v.patch
            }
            VersionRange::Gte(v) => version >= v,
            VersionRange::Lt(v) => version < v,
            VersionRange::Range { gte, lt } => version >= gte && version < lt,
            VersionRange::Wildcard { major, minor } => match (major, minor) {
                (None, _) => true,
                (Some(maj), None) => version.major == *maj,
                (Some(maj), Some(min)) => {
                    version.major == *maj && version.minor == *min
                }
            },
        }
    }

    /// Find the most permissive range that satisfies all given versions.
    pub fn covering_range(&self, versions: &[SemVer]) -> Option<VersionRange> {
        if versions.is_empty() {
            return None;
        }
        let min_v = versions.iter().min().unwrap().clone();
        let max_v = versions.iter().max().unwrap().clone();

        if min_v == max_v {
            return Some(VersionRange::Exact(min_v));
        }

        if min_v.major == max_v.major {
            return Some(VersionRange::Caret(min_v));
        }

        Some(VersionRange::Range {
            gte: min_v,
            lt: SemVer::new(max_v.major + 1, 0, 0),
        })
    }
}

impl Default for VersionRangeAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Tracks API deprecation information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeprecationEntry {
    pub api: String,
    pub deprecated_since: SemVer,
    pub removal_version: Option<SemVer>,
    pub replacement: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DeprecationTracker {
    entries: HashMap<String, Vec<DeprecationEntry>>,
}

impl DeprecationTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that an API was deprecated.
    pub fn deprecate(
        &mut self,
        api: &str,
        since: SemVer,
        removal: Option<SemVer>,
        replacement: Option<&str>,
        message: &str,
    ) {
        let entry = DeprecationEntry {
            api: api.to_string(),
            deprecated_since: since,
            removal_version: removal,
            replacement: replacement.map(|s| s.to_string()),
            message: message.to_string(),
        };
        self.entries
            .entry(api.to_string())
            .or_default()
            .push(entry);
    }

    /// Get the version at which an API was deprecated.
    pub fn deprecated_since(&self, api: &str, version: &SemVer) -> Option<&SemVer> {
        self.entries.get(api).and_then(|entries| {
            entries
                .iter()
                .filter(|e| &e.deprecated_since <= version)
                .map(|e| &e.deprecated_since)
                .min()
        })
    }

    /// Get the planned removal version for an API.
    pub fn removal_version(&self, api: &str) -> Option<&SemVer> {
        self.entries.get(api).and_then(|entries| {
            entries
                .iter()
                .filter_map(|e| e.removal_version.as_ref())
                .max()
        })
    }

    /// Check if an API is deprecated at a given version.
    pub fn is_deprecated(&self, api: &str, at_version: &SemVer) -> bool {
        self.entries.get(api).map_or(false, |entries| {
            entries.iter().any(|e| &e.deprecated_since <= at_version)
        })
    }

    /// Check if an API has been removed at a given version.
    pub fn is_removed(&self, api: &str, at_version: &SemVer) -> bool {
        self.entries.get(api).map_or(false, |entries| {
            entries
                .iter()
                .any(|e| e.removal_version.as_ref().map_or(false, |r| r <= at_version))
        })
    }

    /// Get the replacement suggestion for a deprecated API.
    pub fn replacement(&self, api: &str) -> Option<&str> {
        self.entries.get(api).and_then(|entries| {
            entries
                .iter()
                .rev()
                .find_map(|e| e.replacement.as_deref())
        })
    }

    /// List all deprecated APIs at a given version.
    pub fn deprecated_at(&self, version: &SemVer) -> Vec<&DeprecationEntry> {
        self.entries
            .values()
            .flatten()
            .filter(|e| {
                &e.deprecated_since <= version
                    && e.removal_version
                        .as_ref()
                        .map_or(true, |r| version < r)
            })
            .collect()
    }

    /// List all removed APIs at a given version.
    pub fn removed_at(&self, version: &SemVer) -> Vec<&DeprecationEntry> {
        self.entries
            .values()
            .flatten()
            .filter(|e| e.removal_version.as_ref().map_or(false, |r| r <= version))
            .collect()
    }

    /// Count total deprecation entries.
    pub fn len(&self) -> usize {
        self.entries.values().map(|v| v.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Generate a deprecation report for a version range.
    pub fn report(&self, from: &SemVer, to: &SemVer) -> DeprecationReport {
        let newly_deprecated: Vec<_> = self
            .entries
            .values()
            .flatten()
            .filter(|e| &e.deprecated_since > from && &e.deprecated_since <= to)
            .cloned()
            .collect();

        let newly_removed: Vec<_> = self
            .entries
            .values()
            .flatten()
            .filter(|e| {
                e.removal_version
                    .as_ref()
                    .map_or(false, |r| r > from && r <= to)
            })
            .cloned()
            .collect();

        let still_deprecated: Vec<_> = self
            .entries
            .values()
            .flatten()
            .filter(|e| {
                &e.deprecated_since <= to
                    && e.removal_version.as_ref().map_or(true, |r| to < r)
            })
            .cloned()
            .collect();

        DeprecationReport {
            from: from.clone(),
            to: to.clone(),
            newly_deprecated,
            newly_removed,
            still_deprecated,
        }
    }
}

/// Report of deprecation changes between two versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeprecationReport {
    pub from: SemVer,
    pub to: SemVer,
    pub newly_deprecated: Vec<DeprecationEntry>,
    pub newly_removed: Vec<DeprecationEntry>,
    pub still_deprecated: Vec<DeprecationEntry>,
}

impl DeprecationReport {
    pub fn has_breaking_removals(&self) -> bool {
        !self.newly_removed.is_empty()
    }

    pub fn summary(&self) -> String {
        format!(
            "{} -> {}: {} newly deprecated, {} removed, {} still deprecated",
            self.from,
            self.to,
            self.newly_deprecated.len(),
            self.newly_removed.len(),
            self.still_deprecated.len(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semver_parse_basic() {
        let v: SemVer = "1.2.3".parse().unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
        assert!(v.pre_release.is_none());
    }

    #[test]
    fn test_semver_parse_with_v() {
        let v: SemVer = "v2.0.1".parse().unwrap();
        assert_eq!(v.major, 2);
    }

    #[test]
    fn test_semver_parse_pre_release() {
        let v: SemVer = "1.0.0-alpha.1".parse().unwrap();
        assert_eq!(v.pre_release.as_deref(), Some("alpha.1"));
        assert!(v.is_pre_release());
        assert!(!v.is_stable());
    }

    #[test]
    fn test_semver_parse_build() {
        let v: SemVer = "1.0.0+build.123".parse().unwrap();
        assert_eq!(v.build_metadata.as_deref(), Some("build.123"));
    }

    #[test]
    fn test_semver_parse_two_parts() {
        let v: SemVer = "1.2".parse().unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 0);
    }

    #[test]
    fn test_semver_ordering() {
        let v1: SemVer = "1.0.0".parse().unwrap();
        let v2: SemVer = "1.1.0".parse().unwrap();
        let v3: SemVer = "2.0.0".parse().unwrap();
        assert!(v1 < v2);
        assert!(v2 < v3);
    }

    #[test]
    fn test_semver_display() {
        let v = SemVer::new(1, 2, 3).with_pre_release("beta").with_build("001");
        assert_eq!(format!("{}", v), "1.2.3-beta+001");
    }

    #[test]
    fn test_analyzer_major_bump() {
        let a = SemverAnalyzer::new();
        let old = SemVer::new(1, 0, 0);
        let new = SemVer::new(2, 0, 0);
        assert_eq!(a.analyze(&old, &new), CompatibilityHint::LikelyBreaking);
        assert!(a.is_major_bump(&old, &new));
    }

    #[test]
    fn test_analyzer_minor_bump() {
        let a = SemverAnalyzer::new();
        let old = SemVer::new(1, 0, 0);
        let new = SemVer::new(1, 1, 0);
        assert_eq!(
            a.analyze(&old, &new),
            CompatibilityHint::LikelyBackwardCompatible
        );
        assert!(a.is_minor_bump(&old, &new));
    }

    #[test]
    fn test_analyzer_patch_bump() {
        let a = SemverAnalyzer::new();
        let old = SemVer::new(1, 0, 0);
        let new = SemVer::new(1, 0, 1);
        assert_eq!(
            a.analyze(&old, &new),
            CompatibilityHint::LikelyFullyCompatible
        );
        assert!(a.is_patch_bump(&old, &new));
    }

    #[test]
    fn test_analyzer_identical() {
        let a = SemverAnalyzer::new();
        let v = SemVer::new(1, 0, 0);
        assert_eq!(a.analyze(&v, &v), CompatibilityHint::Identical);
    }

    #[test]
    fn test_analyzer_confidence() {
        let a = SemverAnalyzer::new();
        let v1 = SemVer::new(1, 0, 0);
        let v2 = SemVer::new(1, 0, 1);
        assert!(a.confidence(&v1, &v2) > 0.8);
    }

    #[test]
    fn test_analyze_versions() {
        let a = SemverAnalyzer::new();
        let hint = a.analyze_versions("1.0.0", "2.0.0").unwrap();
        assert_eq!(hint, CompatibilityHint::LikelyBreaking);
    }

    #[test]
    fn test_version_range_exact() {
        let vra = VersionRangeAnalyzer::new();
        let range = vra.parse_range("=1.2.3").unwrap();
        assert!(vra.is_compatible(&SemVer::new(1, 2, 3), &range));
        assert!(!vra.is_compatible(&SemVer::new(1, 2, 4), &range));
    }

    #[test]
    fn test_version_range_caret() {
        let vra = VersionRangeAnalyzer::new();
        let range = vra.parse_range("^1.2.3").unwrap();
        assert!(vra.is_compatible(&SemVer::new(1, 2, 3), &range));
        assert!(vra.is_compatible(&SemVer::new(1, 3, 0), &range));
        assert!(vra.is_compatible(&SemVer::new(1, 9, 9), &range));
        assert!(!vra.is_compatible(&SemVer::new(2, 0, 0), &range));
        assert!(!vra.is_compatible(&SemVer::new(1, 2, 2), &range));
    }

    #[test]
    fn test_version_range_caret_zero() {
        let vra = VersionRangeAnalyzer::new();
        let range = vra.parse_range("^0.2.3").unwrap();
        assert!(vra.is_compatible(&SemVer::new(0, 2, 3), &range));
        assert!(vra.is_compatible(&SemVer::new(0, 2, 9), &range));
        assert!(!vra.is_compatible(&SemVer::new(0, 3, 0), &range));
    }

    #[test]
    fn test_version_range_tilde() {
        let vra = VersionRangeAnalyzer::new();
        let range = vra.parse_range("~1.2.3").unwrap();
        assert!(vra.is_compatible(&SemVer::new(1, 2, 3), &range));
        assert!(vra.is_compatible(&SemVer::new(1, 2, 9), &range));
        assert!(!vra.is_compatible(&SemVer::new(1, 3, 0), &range));
    }

    #[test]
    fn test_version_range_gte() {
        let vra = VersionRangeAnalyzer::new();
        let range = vra.parse_range(">=1.2.3").unwrap();
        assert!(vra.is_compatible(&SemVer::new(1, 2, 3), &range));
        assert!(vra.is_compatible(&SemVer::new(2, 0, 0), &range));
        assert!(!vra.is_compatible(&SemVer::new(1, 2, 2), &range));
    }

    #[test]
    fn test_version_range_lt() {
        let vra = VersionRangeAnalyzer::new();
        let range = vra.parse_range("<2.0.0").unwrap();
        assert!(vra.is_compatible(&SemVer::new(1, 9, 9), &range));
        assert!(!vra.is_compatible(&SemVer::new(2, 0, 0), &range));
    }

    #[test]
    fn test_version_range_combined() {
        let vra = VersionRangeAnalyzer::new();
        let range = vra.parse_range(">=1.0.0, <2.0.0").unwrap();
        assert!(vra.is_compatible(&SemVer::new(1, 5, 0), &range));
        assert!(!vra.is_compatible(&SemVer::new(2, 0, 0), &range));
        assert!(!vra.is_compatible(&SemVer::new(0, 9, 0), &range));
    }

    #[test]
    fn test_version_range_wildcard() {
        let vra = VersionRangeAnalyzer::new();
        let range = vra.parse_range("1.2.*").unwrap();
        assert!(vra.is_compatible(&SemVer::new(1, 2, 0), &range));
        assert!(vra.is_compatible(&SemVer::new(1, 2, 99), &range));
        assert!(!vra.is_compatible(&SemVer::new(1, 3, 0), &range));
    }

    #[test]
    fn test_version_range_any() {
        let vra = VersionRangeAnalyzer::new();
        let range = vra.parse_range("*").unwrap();
        assert!(vra.is_compatible(&SemVer::new(99, 99, 99), &range));
    }

    #[test]
    fn test_covering_range() {
        let vra = VersionRangeAnalyzer::new();
        let versions = vec![SemVer::new(1, 0, 0), SemVer::new(1, 3, 0)];
        let range = vra.covering_range(&versions).unwrap();
        assert!(matches!(range, VersionRange::Caret(_)));
    }

    #[test]
    fn test_deprecation_tracker() {
        let mut dt = DeprecationTracker::new();
        dt.deprecate(
            "/api/v1/users",
            SemVer::new(1, 5, 0),
            Some(SemVer::new(2, 0, 0)),
            Some("/api/v2/users"),
            "Use v2 endpoint instead",
        );

        assert!(dt.is_deprecated("/api/v1/users", &SemVer::new(1, 6, 0)));
        assert!(!dt.is_deprecated("/api/v1/users", &SemVer::new(1, 4, 0)));
        assert!(dt.is_removed("/api/v1/users", &SemVer::new(2, 0, 0)));
        assert!(!dt.is_removed("/api/v1/users", &SemVer::new(1, 9, 0)));
    }

    #[test]
    fn test_deprecation_replacement() {
        let mut dt = DeprecationTracker::new();
        dt.deprecate(
            "/old",
            SemVer::new(1, 0, 0),
            None,
            Some("/new"),
            "moved",
        );
        assert_eq!(dt.replacement("/old"), Some("/new"));
        assert_eq!(dt.replacement("/unknown"), None);
    }

    #[test]
    fn test_deprecation_report() {
        let mut dt = DeprecationTracker::new();
        dt.deprecate(
            "/api/a",
            SemVer::new(1, 2, 0),
            Some(SemVer::new(2, 0, 0)),
            None,
            "deprecated",
        );
        dt.deprecate("/api/b", SemVer::new(1, 5, 0), None, None, "deprecated");

        let report = dt.report(&SemVer::new(1, 0, 0), &SemVer::new(1, 3, 0));
        assert_eq!(report.newly_deprecated.len(), 1);
        assert_eq!(report.newly_removed.len(), 0);
    }

    #[test]
    fn test_deprecation_report_with_removal() {
        let mut dt = DeprecationTracker::new();
        dt.deprecate(
            "/api/old",
            SemVer::new(1, 0, 0),
            Some(SemVer::new(2, 0, 0)),
            None,
            "old api",
        );

        let report = dt.report(&SemVer::new(1, 9, 0), &SemVer::new(2, 0, 0));
        assert!(report.has_breaking_removals());
        assert_eq!(report.newly_removed.len(), 1);
    }

    #[test]
    fn test_deprecated_at() {
        let mut dt = DeprecationTracker::new();
        dt.deprecate("/a", SemVer::new(1, 0, 0), Some(SemVer::new(3, 0, 0)), None, "a");
        dt.deprecate("/b", SemVer::new(2, 0, 0), None, None, "b");

        let at_1_5 = dt.deprecated_at(&SemVer::new(1, 5, 0));
        assert_eq!(at_1_5.len(), 1);

        let at_2_5 = dt.deprecated_at(&SemVer::new(2, 5, 0));
        assert_eq!(at_2_5.len(), 2);
    }
}
