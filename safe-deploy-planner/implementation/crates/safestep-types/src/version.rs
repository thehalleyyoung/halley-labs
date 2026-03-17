// Semantic version types for the SafeStep deployment planner.

use std::cmp::Ordering;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::error::{Result, SafeStepError};

/// A semantic version with major, minor, patch, optional pre-release and build metadata.
#[derive(Clone, Eq, Serialize, Deserialize)]
pub struct Version {
    pub major: u64,
    pub minor: u64,
    pub patch: u64,
    pub pre_release: Option<PreRelease>,
    pub build_metadata: Option<String>,
}

impl Version {
    pub fn new(major: u64, minor: u64, patch: u64) -> Self {
        Self {
            major,
            minor,
            patch,
            pre_release: None,
            build_metadata: None,
        }
    }

    pub fn with_pre_release(mut self, pre: impl Into<String>) -> Self {
        self.pre_release = Some(PreRelease::new(pre));
        self
    }

    pub fn with_build(mut self, build: impl Into<String>) -> Self {
        self.build_metadata = Some(build.into());
        self
    }

    /// Returns true if this is a pre-release version.
    pub fn is_pre_release(&self) -> bool {
        self.pre_release.is_some()
    }

    /// Returns true if this version is a stable release (no pre-release tag).
    pub fn is_stable(&self) -> bool {
        self.pre_release.is_none()
    }

    /// Returns true if this is a major version 0 (unstable API).
    pub fn is_initial_development(&self) -> bool {
        self.major == 0
    }

    /// Returns the next major version.
    pub fn next_major(&self) -> Self {
        Self::new(self.major + 1, 0, 0)
    }

    /// Returns the next minor version.
    pub fn next_minor(&self) -> Self {
        Self::new(self.major, self.minor + 1, 0)
    }

    /// Returns the next patch version.
    pub fn next_patch(&self) -> Self {
        Self::new(self.major, self.minor, self.patch + 1)
    }

    /// Check if this version is API-compatible with another (same major, >= minor).
    pub fn is_api_compatible_with(&self, other: &Version) -> bool {
        if self.major == 0 && other.major == 0 {
            self.minor == other.minor
        } else {
            self.major == other.major
        }
    }

    /// Numeric tuple for compact comparison.
    pub fn as_tuple(&self) -> (u64, u64, u64) {
        (self.major, self.minor, self.patch)
    }

    /// Parse from a string like "1.2.3", "v1.2.3", "1.2.3-beta.1+build.42".
    pub fn parse(input: &str) -> Result<Self> {
        SemverParser::parse(input)
    }
}

impl PartialEq for Version {
    fn eq(&self, other: &Self) -> bool {
        self.major == other.major
            && self.minor == other.minor
            && self.patch == other.patch
            && self.pre_release == other.pre_release
    }
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Version {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.major.cmp(&other.major) {
            Ordering::Equal => {}
            o => return o,
        }
        match self.minor.cmp(&other.minor) {
            Ordering::Equal => {}
            o => return o,
        }
        match self.patch.cmp(&other.patch) {
            Ordering::Equal => {}
            o => return o,
        }
        match (&self.pre_release, &other.pre_release) {
            (None, None) => Ordering::Equal,
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (Some(a), Some(b)) => a.cmp(b),
        }
    }
}

impl std::hash::Hash for Version {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.major.hash(state);
        self.minor.hash(state);
        self.patch.hash(state);
        self.pre_release.hash(state);
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if let Some(pre) = &self.pre_release {
            write!(f, "-{}", pre)?;
        }
        if let Some(build) = &self.build_metadata {
            write!(f, "+{}", build)?;
        }
        Ok(())
    }
}

impl fmt::Debug for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Version({})", self)
    }
}

impl FromStr for Version {
    type Err = SafeStepError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Version::parse(s)
    }
}

// ─── Pre-release identifiers ────────────────────────────────────────────

/// A pre-release identifier is a dot-separated series of identifiers.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PreRelease {
    identifiers: SmallVec<[PreReleaseId; 4]>,
}

/// A single pre-release identifier component.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PreReleaseId {
    Numeric(u64),
    AlphaNumeric(String),
}

impl PreRelease {
    pub fn new(s: impl Into<String>) -> Self {
        let s = s.into();
        let identifiers = s
            .split('.')
            .map(|part| {
                if let Ok(n) = part.parse::<u64>() {
                    PreReleaseId::Numeric(n)
                } else {
                    PreReleaseId::AlphaNumeric(part.to_string())
                }
            })
            .collect();
        Self { identifiers }
    }

    pub fn identifiers(&self) -> &[PreReleaseId] {
        &self.identifiers
    }
}

impl PartialOrd for PreRelease {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PreRelease {
    fn cmp(&self, other: &Self) -> Ordering {
        for (a, b) in self.identifiers.iter().zip(other.identifiers.iter()) {
            let ord = match (a, b) {
                (PreReleaseId::Numeric(x), PreReleaseId::Numeric(y)) => x.cmp(y),
                (PreReleaseId::AlphaNumeric(x), PreReleaseId::AlphaNumeric(y)) => x.cmp(y),
                (PreReleaseId::Numeric(_), PreReleaseId::AlphaNumeric(_)) => Ordering::Less,
                (PreReleaseId::AlphaNumeric(_), PreReleaseId::Numeric(_)) => Ordering::Greater,
            };
            if ord != Ordering::Equal {
                return ord;
            }
        }
        self.identifiers.len().cmp(&other.identifiers.len())
    }
}

impl fmt::Display for PreRelease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, id) in self.identifiers.iter().enumerate() {
            if i > 0 {
                write!(f, ".")?;
            }
            match id {
                PreReleaseId::Numeric(n) => write!(f, "{}", n)?,
                PreReleaseId::AlphaNumeric(s) => write!(f, "{}", s)?,
            }
        }
        Ok(())
    }
}

impl fmt::Debug for PreRelease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PreRelease({})", self)
    }
}

// ─── SemverParser ────────────────────────────────────────────────────────

/// Parser for semver strings.
pub struct SemverParser;

impl SemverParser {
    pub fn parse(input: &str) -> Result<Version> {
        let input = input.trim();
        let input = input.strip_prefix('v').unwrap_or(input);

        if input.is_empty() {
            return Err(SafeStepError::version_parse("empty version string", input));
        }

        // Split off build metadata
        let (version_pre, build_metadata) = match input.split_once('+') {
            Some((vp, bm)) => (vp, Some(bm.to_string())),
            None => (input, None),
        };

        // Split off pre-release
        let (version_part, pre_release) = match version_pre.split_once('-') {
            Some((vp, pr)) => (vp, Some(PreRelease::new(pr))),
            None => (version_pre, None),
        };

        let parts: Vec<&str> = version_part.split('.').collect();
        if parts.len() < 2 || parts.len() > 3 {
            return Err(SafeStepError::version_parse(
                format!(
                    "expected 2 or 3 numeric components, found {}",
                    parts.len()
                ),
                input,
            ));
        }

        let major = parts[0].parse::<u64>().map_err(|_| {
            SafeStepError::version_parse(
                format!("invalid major version: {:?}", parts[0]),
                input,
            )
        })?;
        let minor = parts[1].parse::<u64>().map_err(|_| {
            SafeStepError::version_parse(
                format!("invalid minor version: {:?}", parts[1]),
                input,
            )
        })?;
        let patch = if parts.len() == 3 {
            parts[2].parse::<u64>().map_err(|_| {
                SafeStepError::version_parse(
                    format!("invalid patch version: {:?}", parts[2]),
                    input,
                )
            })?
        } else {
            0
        };

        Ok(Version {
            major,
            minor,
            patch,
            pre_release,
            build_metadata,
        })
    }

    /// Parse a version requirement string like ">=1.2.0, <2.0.0" or "^1.2".
    pub fn parse_requirement(input: &str) -> Result<VersionReq> {
        let input = input.trim();
        if input == "*" {
            return Ok(VersionReq::Any);
        }
        if let Some(rest) = input.strip_prefix('^') {
            let v = Self::parse(rest)?;
            return Ok(VersionReq::Caret(v));
        }
        if let Some(rest) = input.strip_prefix('~') {
            let v = Self::parse(rest)?;
            return Ok(VersionReq::Tilde(v));
        }
        if input.contains(',') {
            let parts: Vec<&str> = input.split(',').collect();
            let mut comparators = Vec::new();
            for part in parts {
                comparators.push(Self::parse_comparator(part.trim())?);
            }
            return Ok(VersionReq::Range(comparators));
        }
        if let Some(rest) = input.strip_prefix(">=") {
            let v = Self::parse(rest.trim())?;
            return Ok(VersionReq::Range(vec![Comparator::GreaterEqual(v)]));
        }
        if let Some(rest) = input.strip_prefix("<=") {
            let v = Self::parse(rest.trim())?;
            return Ok(VersionReq::Range(vec![Comparator::LessEqual(v)]));
        }
        if let Some(rest) = input.strip_prefix('>') {
            let v = Self::parse(rest.trim())?;
            return Ok(VersionReq::Range(vec![Comparator::Greater(v)]));
        }
        if let Some(rest) = input.strip_prefix('<') {
            let v = Self::parse(rest.trim())?;
            return Ok(VersionReq::Range(vec![Comparator::Less(v)]));
        }
        if let Some(rest) = input.strip_prefix('=') {
            let v = Self::parse(rest.trim())?;
            return Ok(VersionReq::Exact(v));
        }
        let v = Self::parse(input)?;
        Ok(VersionReq::Exact(v))
    }

    fn parse_comparator(input: &str) -> Result<Comparator> {
        let input = input.trim();
        if let Some(rest) = input.strip_prefix(">=") {
            Ok(Comparator::GreaterEqual(Self::parse(rest.trim())?))
        } else if let Some(rest) = input.strip_prefix("<=") {
            Ok(Comparator::LessEqual(Self::parse(rest.trim())?))
        } else if let Some(rest) = input.strip_prefix('>') {
            Ok(Comparator::Greater(Self::parse(rest.trim())?))
        } else if let Some(rest) = input.strip_prefix('<') {
            Ok(Comparator::Less(Self::parse(rest.trim())?))
        } else if let Some(rest) = input.strip_prefix('=') {
            Ok(Comparator::Equal(Self::parse(rest.trim())?))
        } else {
            Ok(Comparator::Equal(Self::parse(input)?))
        }
    }
}

// ─── Version requirements ────────────────────────────────────────────────

/// A comparator for version matching.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Comparator {
    Equal(Version),
    Greater(Version),
    GreaterEqual(Version),
    Less(Version),
    LessEqual(Version),
}

impl Comparator {
    pub fn matches(&self, version: &Version) -> bool {
        match self {
            Self::Equal(v) => version == v,
            Self::Greater(v) => version > v,
            Self::GreaterEqual(v) => version >= v,
            Self::Less(v) => version < v,
            Self::LessEqual(v) => version <= v,
        }
    }
}

impl fmt::Display for Comparator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Equal(v) => write!(f, "={}", v),
            Self::Greater(v) => write!(f, ">{}", v),
            Self::GreaterEqual(v) => write!(f, ">={}", v),
            Self::Less(v) => write!(f, "<{}", v),
            Self::LessEqual(v) => write!(f, "<={}", v),
        }
    }
}

/// A version requirement (range specification).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VersionReq {
    /// Matches any version.
    Any,
    /// Exact version match.
    Exact(Version),
    /// Caret requirement (^1.2.3): compatible updates.
    Caret(Version),
    /// Tilde requirement (~1.2.3): patch-level updates.
    Tilde(Version),
    /// Range of comparators (all must be satisfied).
    Range(Vec<Comparator>),
}

impl VersionReq {
    pub fn matches(&self, version: &Version) -> bool {
        match self {
            Self::Any => true,
            Self::Exact(v) => version == v,
            Self::Caret(v) => {
                if v.major != 0 {
                    version.major == v.major && version >= v
                } else if v.minor != 0 {
                    version.major == 0 && version.minor == v.minor && version >= v
                } else {
                    version.major == 0 && version.minor == 0 && version.patch == v.patch
                }
            }
            Self::Tilde(v) => {
                version.major == v.major && version.minor == v.minor && version >= v
            }
            Self::Range(comparators) => comparators.iter().all(|c| c.matches(version)),
        }
    }

    pub fn parse(input: &str) -> Result<Self> {
        SemverParser::parse_requirement(input)
    }
}

impl fmt::Display for VersionReq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Any => write!(f, "*"),
            Self::Exact(v) => write!(f, "={}", v),
            Self::Caret(v) => write!(f, "^{}", v),
            Self::Tilde(v) => write!(f, "~{}", v),
            Self::Range(comparators) => {
                for (i, c) in comparators.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", c)?;
                }
                Ok(())
            }
        }
    }
}

impl FromStr for VersionReq {
    type Err = SafeStepError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        VersionReq::parse(s)
    }
}

// ─── VersionIndex ────────────────────────────────────────────────────────

/// Compact index into a VersionSet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct VersionIndex(pub u32);

impl VersionIndex {
    pub fn new(index: u32) -> Self {
        Self(index)
    }

    pub fn as_usize(&self) -> usize {
        self.0 as usize
    }

    /// Successor index (next version).
    pub fn next(&self) -> Self {
        Self(self.0 + 1)
    }

    /// Predecessor index if > 0.
    pub fn prev(&self) -> Option<Self> {
        if self.0 > 0 {
            Some(Self(self.0 - 1))
        } else {
            None
        }
    }
}

impl fmt::Display for VersionIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v#{}", self.0)
    }
}

impl From<u32> for VersionIndex {
    fn from(v: u32) -> Self {
        Self(v)
    }
}

impl From<usize> for VersionIndex {
    fn from(v: usize) -> Self {
        Self(v as u32)
    }
}

// ─── VersionRange ────────────────────────────────────────────────────────

/// A contiguous range [lo, hi] in a VersionSet, represented by indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VersionRange {
    pub lo: VersionIndex,
    pub hi: VersionIndex,
}

impl VersionRange {
    pub fn new(lo: VersionIndex, hi: VersionIndex) -> Self {
        debug_assert!(lo.0 <= hi.0, "lo must be <= hi");
        Self { lo, hi }
    }

    pub fn single(idx: VersionIndex) -> Self {
        Self { lo: idx, hi: idx }
    }

    pub fn contains(&self, idx: VersionIndex) -> bool {
        idx.0 >= self.lo.0 && idx.0 <= self.hi.0
    }

    pub fn len(&self) -> u32 {
        self.hi.0 - self.lo.0 + 1
    }

    pub fn is_empty(&self) -> bool {
        false // always has at least lo..=hi
    }

    pub fn iter(&self) -> impl Iterator<Item = VersionIndex> {
        (self.lo.0..=self.hi.0).map(VersionIndex)
    }

    pub fn overlaps(&self, other: &VersionRange) -> bool {
        self.lo.0 <= other.hi.0 && other.lo.0 <= self.hi.0
    }

    /// Return the intersection of two ranges, or None if they don't overlap.
    pub fn intersection(&self, other: &VersionRange) -> Option<VersionRange> {
        let lo = self.lo.0.max(other.lo.0);
        let hi = self.hi.0.min(other.hi.0);
        if lo <= hi {
            Some(VersionRange::new(VersionIndex(lo), VersionIndex(hi)))
        } else {
            None
        }
    }

    /// Return the union bounding range.
    pub fn bounding_union(&self, other: &VersionRange) -> VersionRange {
        VersionRange::new(
            VersionIndex(self.lo.0.min(other.lo.0)),
            VersionIndex(self.hi.0.max(other.hi.0)),
        )
    }
}

impl fmt::Display for VersionRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.lo == self.hi {
            write!(f, "[{}]", self.lo)
        } else {
            write!(f, "[{}, {}]", self.lo, self.hi)
        }
    }
}

// ─── VersionSet ──────────────────────────────────────────────────────────

/// An ordered set of versions for a single service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionSet {
    versions: Vec<Version>,
    service_name: String,
}

impl VersionSet {
    /// Create a new empty version set for a service.
    pub fn new(service_name: impl Into<String>) -> Self {
        Self {
            versions: Vec::new(),
            service_name: service_name.into(),
        }
    }

    /// Create from a pre-sorted list of versions.
    pub fn from_sorted(service_name: impl Into<String>, versions: Vec<Version>) -> Self {
        Self {
            versions,
            service_name: service_name.into(),
        }
    }

    /// Insert a version, maintaining sort order. Returns the index.
    pub fn insert(&mut self, version: Version) -> VersionIndex {
        match self.versions.binary_search(&version) {
            Ok(idx) => VersionIndex(idx as u32),
            Err(idx) => {
                self.versions.insert(idx, version);
                VersionIndex(idx as u32)
            }
        }
    }

    /// Get the version at the given index.
    pub fn get(&self, idx: VersionIndex) -> Option<&Version> {
        self.versions.get(idx.as_usize())
    }

    /// Look up the index of a version.
    pub fn index_of(&self, version: &Version) -> Option<VersionIndex> {
        self.versions
            .binary_search(version)
            .ok()
            .map(|i| VersionIndex(i as u32))
    }

    /// Number of versions.
    pub fn len(&self) -> usize {
        self.versions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.versions.is_empty()
    }

    /// The full range [0, len-1].
    pub fn full_range(&self) -> Option<VersionRange> {
        if self.versions.is_empty() {
            None
        } else {
            Some(VersionRange::new(
                VersionIndex(0),
                VersionIndex((self.versions.len() - 1) as u32),
            ))
        }
    }

    /// The lowest version.
    pub fn min_version(&self) -> Option<&Version> {
        self.versions.first()
    }

    /// The highest version.
    pub fn max_version(&self) -> Option<&Version> {
        self.versions.last()
    }

    pub fn service_name(&self) -> &str {
        &self.service_name
    }

    /// Iterate over (index, version) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (VersionIndex, &Version)> {
        self.versions
            .iter()
            .enumerate()
            .map(|(i, v)| (VersionIndex(i as u32), v))
    }

    /// Return versions matching a requirement.
    pub fn matching(&self, req: &VersionReq) -> Vec<VersionIndex> {
        self.versions
            .iter()
            .enumerate()
            .filter(|(_, v)| req.matches(v))
            .map(|(i, _)| VersionIndex(i as u32))
            .collect()
    }

    /// Find the range of indices matching a requirement (if contiguous).
    pub fn matching_range(&self, req: &VersionReq) -> Option<VersionRange> {
        let matches = self.matching(req);
        if matches.is_empty() {
            return None;
        }
        // Check if contiguous
        let first = matches[0].0;
        let last = matches[matches.len() - 1].0;
        if (last - first + 1) as usize == matches.len() {
            Some(VersionRange::new(
                VersionIndex(first),
                VersionIndex(last),
            ))
        } else {
            None
        }
    }

    /// Filter versions by a predicate, returning indices.
    pub fn filter<F: Fn(&Version) -> bool>(&self, pred: F) -> Vec<VersionIndex> {
        self.versions
            .iter()
            .enumerate()
            .filter(|(_, v)| pred(v))
            .map(|(i, _)| VersionIndex(i as u32))
            .collect()
    }

    /// Return the subset of versions in a given range.
    pub fn versions_in_range(&self, range: &VersionRange) -> Vec<&Version> {
        (range.lo.0..=range.hi.0)
            .filter_map(|i| self.versions.get(i as usize))
            .collect()
    }

    /// Check if an upgrade from idx_from to idx_to is a monotone (non-downgrading) transition.
    pub fn is_monotone_upgrade(&self, idx_from: VersionIndex, idx_to: VersionIndex) -> bool {
        idx_to.0 >= idx_from.0
    }

    /// Compute number of bits needed to encode a version index.
    pub fn bits_needed(&self) -> u32 {
        if self.versions.is_empty() {
            0
        } else {
            let n = self.versions.len() as u64;
            64 - n.leading_zeros()
        }
    }
}

impl fmt::Display for VersionSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[", self.service_name)?;
        for (i, v) in self.versions.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", v)?;
        }
        write!(f, "]")
    }
}

// ─── VersionDiff ─────────────────────────────────────────────────────────

/// Describes the type of change between two versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VersionDiffKind {
    Major,
    Minor,
    Patch,
    PreRelease,
    None,
}

impl VersionDiffKind {
    /// Compute the diff kind between two versions.
    pub fn between(from: &Version, to: &Version) -> Self {
        if from.major != to.major {
            Self::Major
        } else if from.minor != to.minor {
            Self::Minor
        } else if from.patch != to.patch {
            Self::Patch
        } else if from.pre_release != to.pre_release {
            Self::PreRelease
        } else {
            Self::None
        }
    }

    pub fn risk_score(&self) -> f64 {
        match self {
            Self::Major => 1.0,
            Self::Minor => 0.5,
            Self::Patch => 0.1,
            Self::PreRelease => 0.3,
            Self::None => 0.0,
        }
    }
}

impl fmt::Display for VersionDiffKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Major => write!(f, "major"),
            Self::Minor => write!(f, "minor"),
            Self::Patch => write!(f, "patch"),
            Self::PreRelease => write!(f, "pre-release"),
            Self::None => write!(f, "none"),
        }
    }
}

/// Full diff information between two versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionDiff {
    pub from: Version,
    pub to: Version,
    pub kind: VersionDiffKind,
    pub is_upgrade: bool,
    pub is_downgrade: bool,
}

impl VersionDiff {
    pub fn compute(from: &Version, to: &Version) -> Self {
        Self {
            from: from.clone(),
            to: to.clone(),
            kind: VersionDiffKind::between(from, to),
            is_upgrade: to > from,
            is_downgrade: to < from,
        }
    }
}

impl fmt::Display for VersionDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} -> {} ({})", self.from, self.to, self.kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_new() {
        let v = Version::new(1, 2, 3);
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
        assert!(v.is_stable());
        assert!(!v.is_pre_release());
    }

    #[test]
    fn test_version_display() {
        assert_eq!(Version::new(1, 2, 3).to_string(), "1.2.3");
        assert_eq!(
            Version::new(1, 0, 0)
                .with_pre_release("alpha.1")
                .to_string(),
            "1.0.0-alpha.1"
        );
        assert_eq!(
            Version::new(1, 0, 0).with_build("20240101").to_string(),
            "1.0.0+20240101"
        );
    }

    #[test]
    fn test_version_parse_simple() {
        let v = Version::parse("1.2.3").unwrap();
        assert_eq!(v, Version::new(1, 2, 3));
    }

    #[test]
    fn test_version_parse_with_v() {
        let v = Version::parse("v2.0.1").unwrap();
        assert_eq!(v, Version::new(2, 0, 1));
    }

    #[test]
    fn test_version_parse_two_part() {
        let v = Version::parse("1.5").unwrap();
        assert_eq!(v, Version::new(1, 5, 0));
    }

    #[test]
    fn test_version_parse_pre_release() {
        let v = Version::parse("1.0.0-beta.2").unwrap();
        assert_eq!(v.major, 1);
        assert!(v.is_pre_release());
        assert_eq!(v.pre_release.as_ref().unwrap().to_string(), "beta.2");
    }

    #[test]
    fn test_version_parse_build() {
        let v = Version::parse("1.0.0+build.42").unwrap();
        assert_eq!(v.build_metadata.as_deref(), Some("build.42"));
    }

    #[test]
    fn test_version_parse_full() {
        let v = Version::parse("1.2.3-rc.1+linux").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
        assert!(v.is_pre_release());
    }

    #[test]
    fn test_version_parse_errors() {
        assert!(Version::parse("").is_err());
        assert!(Version::parse("abc").is_err());
        assert!(Version::parse("1").is_err());
    }

    #[test]
    fn test_version_ordering() {
        assert!(Version::new(1, 0, 0) < Version::new(2, 0, 0));
        assert!(Version::new(1, 0, 0) < Version::new(1, 1, 0));
        assert!(Version::new(1, 0, 0) < Version::new(1, 0, 1));
        assert!(
            Version::new(1, 0, 0).with_pre_release("alpha") < Version::new(1, 0, 0)
        );
        assert!(
            Version::new(1, 0, 0).with_pre_release("alpha")
                < Version::new(1, 0, 0).with_pre_release("beta")
        );
    }

    #[test]
    fn test_version_equality_ignores_build() {
        let a = Version::new(1, 0, 0).with_build("a");
        let b = Version::new(1, 0, 0).with_build("b");
        assert_eq!(a, b);
    }

    #[test]
    fn test_version_api_compatible() {
        let v1 = Version::new(1, 2, 0);
        let v2 = Version::new(1, 5, 0);
        let v3 = Version::new(2, 0, 0);
        assert!(v1.is_api_compatible_with(&v2));
        assert!(!v1.is_api_compatible_with(&v3));
    }

    #[test]
    fn test_version_next() {
        let v = Version::new(1, 2, 3);
        assert_eq!(v.next_major(), Version::new(2, 0, 0));
        assert_eq!(v.next_minor(), Version::new(1, 3, 0));
        assert_eq!(v.next_patch(), Version::new(1, 2, 4));
    }

    #[test]
    fn test_version_req_any() {
        let req = VersionReq::Any;
        assert!(req.matches(&Version::new(0, 0, 1)));
        assert!(req.matches(&Version::new(99, 99, 99)));
    }

    #[test]
    fn test_version_req_exact() {
        let req = VersionReq::Exact(Version::new(1, 2, 3));
        assert!(req.matches(&Version::new(1, 2, 3)));
        assert!(!req.matches(&Version::new(1, 2, 4)));
    }

    #[test]
    fn test_version_req_caret() {
        let req = VersionReq::Caret(Version::new(1, 2, 0));
        assert!(req.matches(&Version::new(1, 2, 0)));
        assert!(req.matches(&Version::new(1, 9, 0)));
        assert!(!req.matches(&Version::new(2, 0, 0)));
        assert!(!req.matches(&Version::new(1, 1, 0)));
    }

    #[test]
    fn test_version_req_tilde() {
        let req = VersionReq::Tilde(Version::new(1, 2, 0));
        assert!(req.matches(&Version::new(1, 2, 0)));
        assert!(req.matches(&Version::new(1, 2, 9)));
        assert!(!req.matches(&Version::new(1, 3, 0)));
    }

    #[test]
    fn test_version_req_range() {
        let req = VersionReq::Range(vec![
            Comparator::GreaterEqual(Version::new(1, 0, 0)),
            Comparator::Less(Version::new(2, 0, 0)),
        ]);
        assert!(req.matches(&Version::new(1, 0, 0)));
        assert!(req.matches(&Version::new(1, 9, 9)));
        assert!(!req.matches(&Version::new(2, 0, 0)));
        assert!(!req.matches(&Version::new(0, 9, 9)));
    }

    #[test]
    fn test_version_req_parse_caret() {
        let req = VersionReq::parse("^1.2.0").unwrap();
        assert!(matches!(req, VersionReq::Caret(_)));
    }

    #[test]
    fn test_version_req_parse_tilde() {
        let req = VersionReq::parse("~1.2.0").unwrap();
        assert!(matches!(req, VersionReq::Tilde(_)));
    }

    #[test]
    fn test_version_req_parse_any() {
        let req = VersionReq::parse("*").unwrap();
        assert!(matches!(req, VersionReq::Any));
    }

    #[test]
    fn test_version_req_parse_range() {
        let req = VersionReq::parse(">=1.0.0, <2.0.0").unwrap();
        assert!(matches!(req, VersionReq::Range(_)));
    }

    #[test]
    fn test_version_index() {
        let idx = VersionIndex::new(5);
        assert_eq!(idx.as_usize(), 5);
        assert_eq!(idx.next(), VersionIndex(6));
        assert_eq!(idx.prev(), Some(VersionIndex(4)));
        assert_eq!(VersionIndex(0).prev(), None);
    }

    #[test]
    fn test_version_range() {
        let r = VersionRange::new(VersionIndex(2), VersionIndex(5));
        assert!(r.contains(VersionIndex(2)));
        assert!(r.contains(VersionIndex(3)));
        assert!(r.contains(VersionIndex(5)));
        assert!(!r.contains(VersionIndex(1)));
        assert!(!r.contains(VersionIndex(6)));
        assert_eq!(r.len(), 4);
    }

    #[test]
    fn test_version_range_overlap() {
        let a = VersionRange::new(VersionIndex(1), VersionIndex(5));
        let b = VersionRange::new(VersionIndex(3), VersionIndex(8));
        let c = VersionRange::new(VersionIndex(6), VersionIndex(8));
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_version_range_intersection() {
        let a = VersionRange::new(VersionIndex(1), VersionIndex(5));
        let b = VersionRange::new(VersionIndex(3), VersionIndex(8));
        let int = a.intersection(&b).unwrap();
        assert_eq!(int.lo, VersionIndex(3));
        assert_eq!(int.hi, VersionIndex(5));
    }

    #[test]
    fn test_version_range_no_intersection() {
        let a = VersionRange::new(VersionIndex(1), VersionIndex(3));
        let b = VersionRange::new(VersionIndex(5), VersionIndex(8));
        assert!(a.intersection(&b).is_none());
    }

    #[test]
    fn test_version_set() {
        let mut vs = VersionSet::new("api-server");
        let i1 = vs.insert(Version::new(1, 0, 0));
        let i2 = vs.insert(Version::new(2, 0, 0));
        let i3 = vs.insert(Version::new(1, 5, 0));

        assert_eq!(vs.len(), 3);
        assert_eq!(i1, VersionIndex(0));
        assert_eq!(i2, VersionIndex(1)); // index at time of insert (before 1.5.0 shifts it)
        assert_eq!(i3, VersionIndex(1)); // 1.5.0 inserted at index 1, shifting 2.0.0 to index 2

        assert_eq!(vs.get(VersionIndex(0)), Some(&Version::new(1, 0, 0)));
        assert_eq!(vs.get(VersionIndex(1)), Some(&Version::new(1, 5, 0)));
        assert_eq!(vs.get(VersionIndex(2)), Some(&Version::new(2, 0, 0)));
    }

    #[test]
    fn test_version_set_index_of() {
        let mut vs = VersionSet::new("svc");
        vs.insert(Version::new(1, 0, 0));
        vs.insert(Version::new(2, 0, 0));
        assert_eq!(vs.index_of(&Version::new(1, 0, 0)), Some(VersionIndex(0)));
        assert_eq!(vs.index_of(&Version::new(3, 0, 0)), None);
    }

    #[test]
    fn test_version_set_matching() {
        let mut vs = VersionSet::new("svc");
        vs.insert(Version::new(1, 0, 0));
        vs.insert(Version::new(1, 1, 0));
        vs.insert(Version::new(1, 2, 0));
        vs.insert(Version::new(2, 0, 0));

        let req = VersionReq::Caret(Version::new(1, 0, 0));
        let matches = vs.matching(&req);
        assert_eq!(matches.len(), 3);
    }

    #[test]
    fn test_version_set_matching_range() {
        let mut vs = VersionSet::new("svc");
        vs.insert(Version::new(1, 0, 0));
        vs.insert(Version::new(1, 1, 0));
        vs.insert(Version::new(1, 2, 0));
        vs.insert(Version::new(2, 0, 0));

        let req = VersionReq::Caret(Version::new(1, 0, 0));
        let range = vs.matching_range(&req).unwrap();
        assert_eq!(range.lo, VersionIndex(0));
        assert_eq!(range.hi, VersionIndex(2));
    }

    #[test]
    fn test_version_set_bits_needed() {
        let mut vs = VersionSet::new("svc");
        assert_eq!(vs.bits_needed(), 0);
        vs.insert(Version::new(1, 0, 0));
        assert_eq!(vs.bits_needed(), 1);
        vs.insert(Version::new(2, 0, 0));
        assert_eq!(vs.bits_needed(), 2);
        vs.insert(Version::new(3, 0, 0));
        vs.insert(Version::new(4, 0, 0));
        assert_eq!(vs.bits_needed(), 3);
    }

    #[test]
    fn test_version_set_monotone() {
        let vs = VersionSet::new("svc");
        assert!(vs.is_monotone_upgrade(VersionIndex(0), VersionIndex(1)));
        assert!(vs.is_monotone_upgrade(VersionIndex(0), VersionIndex(0)));
        assert!(!vs.is_monotone_upgrade(VersionIndex(1), VersionIndex(0)));
    }

    #[test]
    fn test_version_diff() {
        let a = Version::new(1, 2, 3);
        let b = Version::new(2, 0, 0);
        let diff = VersionDiff::compute(&a, &b);
        assert_eq!(diff.kind, VersionDiffKind::Major);
        assert!(diff.is_upgrade);
        assert!(!diff.is_downgrade);
    }

    #[test]
    fn test_version_diff_downgrade() {
        let a = Version::new(2, 0, 0);
        let b = Version::new(1, 5, 0);
        let diff = VersionDiff::compute(&a, &b);
        assert!(diff.is_downgrade);
        assert!(!diff.is_upgrade);
    }

    #[test]
    fn test_version_diff_kind_risk() {
        assert!(VersionDiffKind::Major.risk_score() > VersionDiffKind::Minor.risk_score());
        assert!(VersionDiffKind::Minor.risk_score() > VersionDiffKind::Patch.risk_score());
        assert_eq!(VersionDiffKind::None.risk_score(), 0.0);
    }

    #[test]
    fn test_version_serialization() {
        let v = Version::new(1, 2, 3);
        let json = serde_json::to_string(&v).unwrap();
        let parsed: Version = serde_json::from_str(&json).unwrap();
        assert_eq!(v, parsed);
    }

    #[test]
    fn test_prerelease_ordering() {
        let a = PreRelease::new("alpha");
        let b = PreRelease::new("beta");
        let c = PreRelease::new("1");
        assert!(a < b);
        assert!(c < a); // numeric < alpha
    }

    #[test]
    fn test_version_req_display() {
        let req = VersionReq::Caret(Version::new(1, 2, 0));
        assert_eq!(req.to_string(), "^1.2.0");
        let req = VersionReq::Any;
        assert_eq!(req.to_string(), "*");
    }

    #[test]
    fn test_version_set_display() {
        let mut vs = VersionSet::new("svc");
        vs.insert(Version::new(1, 0, 0));
        vs.insert(Version::new(2, 0, 0));
        let s = vs.to_string();
        assert!(s.contains("svc["));
        assert!(s.contains("1.0.0"));
        assert!(s.contains("2.0.0"));
    }

    #[test]
    fn test_version_range_iter() {
        let r = VersionRange::new(VersionIndex(2), VersionIndex(4));
        let indices: Vec<_> = r.iter().collect();
        assert_eq!(indices, vec![VersionIndex(2), VersionIndex(3), VersionIndex(4)]);
    }

    #[test]
    fn test_version_set_full_range() {
        let mut vs = VersionSet::new("svc");
        assert!(vs.full_range().is_none());
        vs.insert(Version::new(1, 0, 0));
        vs.insert(Version::new(2, 0, 0));
        let r = vs.full_range().unwrap();
        assert_eq!(r.lo, VersionIndex(0));
        assert_eq!(r.hi, VersionIndex(1));
    }

    #[test]
    fn test_version_initial_development() {
        assert!(Version::new(0, 1, 0).is_initial_development());
        assert!(!Version::new(1, 0, 0).is_initial_development());
    }

    #[test]
    fn test_version_from_str() {
        let v: Version = "3.2.1".parse().unwrap();
        assert_eq!(v, Version::new(3, 2, 1));
    }

    #[test]
    fn test_version_range_single() {
        let r = VersionRange::single(VersionIndex(5));
        assert_eq!(r.len(), 1);
        assert!(r.contains(VersionIndex(5)));
        assert!(!r.contains(VersionIndex(4)));
    }

    #[test]
    fn test_version_range_bounding_union() {
        let a = VersionRange::new(VersionIndex(1), VersionIndex(3));
        let b = VersionRange::new(VersionIndex(5), VersionIndex(8));
        let u = a.bounding_union(&b);
        assert_eq!(u.lo, VersionIndex(1));
        assert_eq!(u.hi, VersionIndex(8));
    }
}
