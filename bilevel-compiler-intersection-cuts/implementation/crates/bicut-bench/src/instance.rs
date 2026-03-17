//! Benchmark instance management: loading, metadata, filtering, and set operations.

use bicut_types::{BilevelProblem, SparseMatrix, DEFAULT_TOLERANCE};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Difficulty classification for benchmark instances.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DifficultyClass {
    /// Solvable by most methods in under 10 seconds.
    Easy,
    /// Requires moderate computation (10–300 s).
    Medium,
    /// Requires significant computation (300–3600 s).
    Hard,
    /// No known optimal solution; used as stress tests.
    Open,
}

impl DifficultyClass {
    /// Classify based on total variable count and constraint count.
    pub fn from_dimensions(total_vars: usize, total_constraints: usize) -> Self {
        let size = total_vars + total_constraints;
        if size <= 50 {
            DifficultyClass::Easy
        } else if size <= 200 {
            DifficultyClass::Medium
        } else if size <= 1000 {
            DifficultyClass::Hard
        } else {
            DifficultyClass::Open
        }
    }

    /// All difficulty classes in order.
    pub fn all() -> &'static [DifficultyClass] {
        &[
            DifficultyClass::Easy,
            DifficultyClass::Medium,
            DifficultyClass::Hard,
            DifficultyClass::Open,
        ]
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            DifficultyClass::Easy => "Easy",
            DifficultyClass::Medium => "Medium",
            DifficultyClass::Hard => "Hard",
            DifficultyClass::Open => "Open",
        }
    }
}

impl std::fmt::Display for DifficultyClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Type of bilevel optimization instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InstanceType {
    /// Bilevel linear program (continuous upper and lower).
    BLP,
    /// Mixed-integer bilevel linear program (integer upper variables).
    MIBLP,
    /// Integer bilevel linear program (all variables integer).
    IBLP,
    /// Bilevel program with integer lower-level variables.
    BPIL,
    /// Knapsack interdiction problem.
    KnapsackInterdiction,
    /// Network interdiction problem.
    NetworkInterdiction,
    /// Generic / unclassified.
    Generic,
}

impl InstanceType {
    /// Determine type from problem characteristics.
    pub fn classify(
        has_integer_upper: bool,
        has_integer_lower: bool,
        is_interdiction: bool,
    ) -> Self {
        if is_interdiction {
            return InstanceType::KnapsackInterdiction;
        }
        match (has_integer_upper, has_integer_lower) {
            (false, false) => InstanceType::BLP,
            (true, false) => InstanceType::MIBLP,
            (true, true) => InstanceType::IBLP,
            (false, true) => InstanceType::BPIL,
        }
    }

    /// All instance types.
    pub fn all() -> &'static [InstanceType] {
        &[
            InstanceType::BLP,
            InstanceType::MIBLP,
            InstanceType::IBLP,
            InstanceType::BPIL,
            InstanceType::KnapsackInterdiction,
            InstanceType::NetworkInterdiction,
            InstanceType::Generic,
        ]
    }

    /// Short label for tables.
    pub fn short_label(&self) -> &'static str {
        match self {
            InstanceType::BLP => "BLP",
            InstanceType::MIBLP => "MIBLP",
            InstanceType::IBLP => "IBLP",
            InstanceType::BPIL => "BPIL",
            InstanceType::KnapsackInterdiction => "KI",
            InstanceType::NetworkInterdiction => "NI",
            InstanceType::Generic => "GEN",
        }
    }
}

impl std::fmt::Display for InstanceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.short_label())
    }
}

// ---------------------------------------------------------------------------
// Instance metadata
// ---------------------------------------------------------------------------

/// Metadata about a benchmark instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceMetadata {
    /// Instance name (unique identifier).
    pub name: String,
    /// Source file path, if loaded from disk.
    pub source_path: Option<PathBuf>,
    /// Number of upper-level variables.
    pub num_upper_vars: usize,
    /// Number of lower-level variables.
    pub num_lower_vars: usize,
    /// Number of upper-level constraints.
    pub num_upper_constraints: usize,
    /// Number of lower-level constraints.
    pub num_lower_constraints: usize,
    /// Number of non-zero entries in the lower constraint matrix.
    pub lower_nnz: usize,
    /// Number of non-zero entries in the linking matrix.
    pub linking_nnz: usize,
    /// Difficulty classification.
    pub difficulty: DifficultyClass,
    /// Problem type.
    pub instance_type: InstanceType,
    /// Known optimal objective value, if available.
    pub known_optimal: Option<f64>,
    /// Reference to original source (paper, library, etc.).
    pub source_reference: Option<String>,
    /// Arbitrary tags for grouping.
    pub tags: Vec<String>,
}

impl InstanceMetadata {
    /// Create metadata from a bilevel problem with a given name.
    pub fn from_problem(name: &str, problem: &BilevelProblem) -> Self {
        let total_vars = problem.num_upper_vars + problem.num_lower_vars;
        let total_cons = problem.num_upper_constraints + problem.num_lower_constraints;
        let difficulty = DifficultyClass::from_dimensions(total_vars, total_cons);
        InstanceMetadata {
            name: name.to_string(),
            source_path: None,
            num_upper_vars: problem.num_upper_vars,
            num_lower_vars: problem.num_lower_vars,
            num_upper_constraints: problem.num_upper_constraints,
            num_lower_constraints: problem.num_lower_constraints,
            lower_nnz: problem.lower_a.entries.len(),
            linking_nnz: problem.lower_linking_b.entries.len(),
            difficulty,
            instance_type: InstanceType::BLP,
            known_optimal: None,
            source_reference: None,
            tags: Vec::new(),
        }
    }

    /// Total number of variables.
    pub fn total_vars(&self) -> usize {
        self.num_upper_vars + self.num_lower_vars
    }

    /// Total number of constraints.
    pub fn total_constraints(&self) -> usize {
        self.num_upper_constraints + self.num_lower_constraints
    }

    /// Total number of non-zeros across all matrices.
    pub fn total_nnz(&self) -> usize {
        self.lower_nnz + self.linking_nnz
    }

    /// Density of the lower-level constraint matrix.
    pub fn lower_density(&self) -> f64 {
        let capacity = self.num_lower_vars * self.num_lower_constraints;
        if capacity == 0 {
            0.0
        } else {
            self.lower_nnz as f64 / capacity as f64
        }
    }

    /// Whether the instance has linking constraints.
    pub fn has_linking(&self) -> bool {
        self.linking_nnz > 0
    }

    /// Whether the instance has a known optimal value.
    pub fn has_known_optimal(&self) -> bool {
        self.known_optimal.is_some()
    }
}

// ---------------------------------------------------------------------------
// BenchmarkInstance
// ---------------------------------------------------------------------------

/// A single benchmark instance combining a bilevel problem with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkInstance {
    /// Metadata describing the instance.
    pub metadata: InstanceMetadata,
    /// The bilevel optimization problem.
    pub problem: BilevelProblem,
}

impl BenchmarkInstance {
    /// Create a new benchmark instance.
    pub fn new(name: &str, problem: BilevelProblem) -> Self {
        let metadata = InstanceMetadata::from_problem(name, &problem);
        BenchmarkInstance { metadata, problem }
    }

    /// Create with explicit metadata.
    pub fn with_metadata(metadata: InstanceMetadata, problem: BilevelProblem) -> Self {
        BenchmarkInstance { metadata, problem }
    }

    /// Instance name.
    pub fn name(&self) -> &str {
        &self.metadata.name
    }

    /// Difficulty class.
    pub fn difficulty(&self) -> DifficultyClass {
        self.metadata.difficulty
    }

    /// Instance type.
    pub fn instance_type(&self) -> InstanceType {
        self.metadata.instance_type
    }

    /// Set the known optimal value.
    pub fn set_known_optimal(&mut self, val: f64) {
        self.metadata.known_optimal = Some(val);
    }

    /// Set the instance type.
    pub fn set_instance_type(&mut self, itype: InstanceType) {
        self.metadata.instance_type = itype;
    }

    /// Add a tag.
    pub fn add_tag(&mut self, tag: &str) {
        if !self.metadata.tags.contains(&tag.to_string()) {
            self.metadata.tags.push(tag.to_string());
        }
    }

    /// Check whether the instance matches a filter.
    pub fn matches_filter(&self, filter: &InstanceFilter) -> bool {
        filter.matches(&self.metadata)
    }

    /// Validate dimensional consistency of the problem.
    pub fn validate(&self) -> Result<(), String> {
        let p = &self.problem;
        if p.upper_obj_c_x.len() != p.num_upper_vars {
            return Err(format!(
                "upper_obj_c_x length {} != num_upper_vars {}",
                p.upper_obj_c_x.len(),
                p.num_upper_vars
            ));
        }
        if p.upper_obj_c_y.len() != p.num_lower_vars {
            return Err(format!(
                "upper_obj_c_y length {} != num_lower_vars {}",
                p.upper_obj_c_y.len(),
                p.num_lower_vars
            ));
        }
        if p.lower_obj_c.len() != p.num_lower_vars {
            return Err(format!(
                "lower_obj_c length {} != num_lower_vars {}",
                p.lower_obj_c.len(),
                p.num_lower_vars
            ));
        }
        if p.lower_b.len() != p.num_lower_constraints {
            return Err(format!(
                "lower_b length {} != num_lower_constraints {}",
                p.lower_b.len(),
                p.num_lower_constraints
            ));
        }
        if p.lower_a.rows != p.num_lower_constraints || p.lower_a.cols != p.num_lower_vars {
            return Err("lower_a dimension mismatch".to_string());
        }
        Ok(())
    }

    /// Compute a hash-like fingerprint of the problem dimensions.
    pub fn fingerprint(&self) -> u64 {
        let p = &self.problem;
        let mut h: u64 = 0xcbf29ce484222325;
        for &dim in &[
            p.num_upper_vars,
            p.num_lower_vars,
            p.num_upper_constraints,
            p.num_lower_constraints,
            p.lower_a.entries.len(),
        ] {
            h ^= dim as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        for &v in &p.lower_obj_c {
            let bits = v.to_bits();
            h ^= bits;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    /// Load an instance from a JSON file.
    pub fn load_json(path: &Path) -> Result<Self, crate::BenchError> {
        let content = std::fs::read_to_string(path).map_err(crate::BenchError::Io)?;
        let mut inst: BenchmarkInstance =
            serde_json::from_str(&content).map_err(|e| crate::BenchError::Parse(e.to_string()))?;
        inst.metadata.source_path = Some(path.to_path_buf());
        Ok(inst)
    }

    /// Save an instance to a JSON file.
    pub fn save_json(&self, path: &Path) -> Result<(), crate::BenchError> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| crate::BenchError::Serialization(e.to_string()))?;
        std::fs::write(path, content).map_err(crate::BenchError::Io)
    }

    /// Load all JSON instances from a directory.
    pub fn load_directory(dir: &Path) -> Result<Vec<Self>, crate::BenchError> {
        let mut instances = Vec::new();
        if !dir.is_dir() {
            return Err(crate::BenchError::InstanceNotFound(
                dir.display().to_string(),
            ));
        }
        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .map_err(crate::BenchError::Io)?
            .filter_map(|e| e.ok())
            .collect();
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                match Self::load_json(&path) {
                    Ok(inst) => instances.push(inst),
                    Err(e) => {
                        log::warn!("Skipping {}: {}", path.display(), e);
                    }
                }
            }
        }
        Ok(instances)
    }
}

// ---------------------------------------------------------------------------
// InstanceFilter
// ---------------------------------------------------------------------------

/// Filter predicate for selecting benchmark instances.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InstanceFilter {
    /// If set, only include instances with names containing this substring.
    pub name_contains: Option<String>,
    /// If set, only include instances of these types.
    pub instance_types: Option<Vec<InstanceType>>,
    /// If set, only include instances of these difficulties.
    pub difficulties: Option<Vec<DifficultyClass>>,
    /// Minimum total number of variables.
    pub min_vars: Option<usize>,
    /// Maximum total number of variables.
    pub max_vars: Option<usize>,
    /// Minimum total number of constraints.
    pub min_constraints: Option<usize>,
    /// Maximum total number of constraints.
    pub max_constraints: Option<usize>,
    /// Only include instances with a known optimal value.
    pub require_known_optimal: bool,
    /// Only include instances whose tags include ALL of these.
    pub required_tags: Vec<String>,
    /// Exclude instances whose names contain any of these substrings.
    pub exclude_names: Vec<String>,
    /// Minimum lower-level density.
    pub min_density: Option<f64>,
    /// Maximum lower-level density.
    pub max_density: Option<f64>,
}

impl InstanceFilter {
    /// Create an empty filter (matches everything).
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: filter by name substring.
    pub fn with_name_contains(mut self, s: &str) -> Self {
        self.name_contains = Some(s.to_string());
        self
    }

    /// Builder: filter by instance type.
    pub fn with_types(mut self, types: Vec<InstanceType>) -> Self {
        self.instance_types = Some(types);
        self
    }

    /// Builder: filter by difficulty.
    pub fn with_difficulties(mut self, diffs: Vec<DifficultyClass>) -> Self {
        self.difficulties = Some(diffs);
        self
    }

    /// Builder: set minimum total variables.
    pub fn with_min_vars(mut self, n: usize) -> Self {
        self.min_vars = Some(n);
        self
    }

    /// Builder: set maximum total variables.
    pub fn with_max_vars(mut self, n: usize) -> Self {
        self.max_vars = Some(n);
        self
    }

    /// Builder: set minimum total constraints.
    pub fn with_min_constraints(mut self, n: usize) -> Self {
        self.min_constraints = Some(n);
        self
    }

    /// Builder: set maximum total constraints.
    pub fn with_max_constraints(mut self, n: usize) -> Self {
        self.max_constraints = Some(n);
        self
    }

    /// Builder: require known optimal.
    pub fn with_require_known_optimal(mut self) -> Self {
        self.require_known_optimal = true;
        self
    }

    /// Builder: require tags.
    pub fn with_required_tags(mut self, tags: Vec<String>) -> Self {
        self.required_tags = tags;
        self
    }

    /// Builder: exclude names containing substring.
    pub fn with_exclude_name(mut self, s: &str) -> Self {
        self.exclude_names.push(s.to_string());
        self
    }

    /// Builder: density range.
    pub fn with_density_range(mut self, min: f64, max: f64) -> Self {
        self.min_density = Some(min);
        self.max_density = Some(max);
        self
    }

    /// Test whether metadata matches this filter.
    pub fn matches(&self, meta: &InstanceMetadata) -> bool {
        if let Some(ref sub) = self.name_contains {
            if !meta.name.contains(sub.as_str()) {
                return false;
            }
        }
        if let Some(ref types) = self.instance_types {
            if !types.contains(&meta.instance_type) {
                return false;
            }
        }
        if let Some(ref diffs) = self.difficulties {
            if !diffs.contains(&meta.difficulty) {
                return false;
            }
        }
        let tv = meta.total_vars();
        if let Some(min) = self.min_vars {
            if tv < min {
                return false;
            }
        }
        if let Some(max) = self.max_vars {
            if tv > max {
                return false;
            }
        }
        let tc = meta.total_constraints();
        if let Some(min) = self.min_constraints {
            if tc < min {
                return false;
            }
        }
        if let Some(max) = self.max_constraints {
            if tc > max {
                return false;
            }
        }
        if self.require_known_optimal && meta.known_optimal.is_none() {
            return false;
        }
        for tag in &self.required_tags {
            if !meta.tags.contains(tag) {
                return false;
            }
        }
        for excl in &self.exclude_names {
            if meta.name.contains(excl.as_str()) {
                return false;
            }
        }
        let density = meta.lower_density();
        if let Some(min) = self.min_density {
            if density < min - DEFAULT_TOLERANCE {
                return false;
            }
        }
        if let Some(max) = self.max_density {
            if density > max + DEFAULT_TOLERANCE {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// InstanceSet
// ---------------------------------------------------------------------------

/// A named collection of benchmark instances with set operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceSet {
    /// Name of this instance set.
    pub name: String,
    /// Instances in insertion order.
    pub instances: Vec<BenchmarkInstance>,
}

impl InstanceSet {
    /// Create an empty set.
    pub fn new(name: &str) -> Self {
        InstanceSet {
            name: name.to_string(),
            instances: Vec::new(),
        }
    }

    /// Create from a vector of instances.
    pub fn from_instances(name: &str, instances: Vec<BenchmarkInstance>) -> Self {
        InstanceSet {
            name: name.to_string(),
            instances,
        }
    }

    /// Number of instances.
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Add an instance.
    pub fn push(&mut self, instance: BenchmarkInstance) {
        self.instances.push(instance);
    }

    /// Iterate over instances.
    pub fn iter(&self) -> std::slice::Iter<'_, BenchmarkInstance> {
        self.instances.iter()
    }

    /// Get a mutable iterator.
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, BenchmarkInstance> {
        self.instances.iter_mut()
    }

    /// Get instance by name.
    pub fn get_by_name(&self, name: &str) -> Option<&BenchmarkInstance> {
        self.instances.iter().find(|i| i.name() == name)
    }

    /// Filter instances by predicate, returning a new set.
    pub fn filter(&self, filter: &InstanceFilter) -> Self {
        let filtered = self
            .instances
            .iter()
            .filter(|i| filter.matches(&i.metadata))
            .cloned()
            .collect();
        InstanceSet {
            name: format!("{}_filtered", self.name),
            instances: filtered,
        }
    }

    /// Filter instances with a closure.
    pub fn filter_with<F>(&self, pred: F) -> Self
    where
        F: Fn(&BenchmarkInstance) -> bool,
    {
        let filtered = self.instances.iter().filter(|i| pred(i)).cloned().collect();
        InstanceSet {
            name: format!("{}_custom_filtered", self.name),
            instances: filtered,
        }
    }

    /// Group instances by difficulty class.
    pub fn group_by_difficulty(&self) -> HashMap<DifficultyClass, InstanceSet> {
        let mut groups: HashMap<DifficultyClass, Vec<BenchmarkInstance>> = HashMap::new();
        for inst in &self.instances {
            groups
                .entry(inst.difficulty())
                .or_default()
                .push(inst.clone());
        }
        groups
            .into_iter()
            .map(|(d, insts)| {
                (
                    d,
                    InstanceSet::from_instances(&format!("{}_{}", self.name, d.label()), insts),
                )
            })
            .collect()
    }

    /// Group instances by type.
    pub fn group_by_type(&self) -> HashMap<InstanceType, InstanceSet> {
        let mut groups: HashMap<InstanceType, Vec<BenchmarkInstance>> = HashMap::new();
        for inst in &self.instances {
            groups
                .entry(inst.instance_type())
                .or_default()
                .push(inst.clone());
        }
        groups
            .into_iter()
            .map(|(t, insts)| {
                (
                    t,
                    InstanceSet::from_instances(
                        &format!("{}_{}", self.name, t.short_label()),
                        insts,
                    ),
                )
            })
            .collect()
    }

    /// Sort instances by total variable count.
    pub fn sort_by_size(&mut self) {
        self.instances.sort_by_key(|i| i.metadata.total_vars());
    }

    /// Sort instances by name.
    pub fn sort_by_name(&mut self) {
        self.instances.sort_by(|a, b| a.name().cmp(b.name()));
    }

    /// Set intersection: instances whose names appear in both sets.
    pub fn intersect(&self, other: &InstanceSet) -> InstanceSet {
        let other_names: HashSet<&str> = other.instances.iter().map(|i| i.name()).collect();
        let insts = self
            .instances
            .iter()
            .filter(|i| other_names.contains(i.name()))
            .cloned()
            .collect();
        InstanceSet::from_instances(&format!("{}_intersect_{}", self.name, other.name), insts)
    }

    /// Set union: all instances from both sets, preferring self on name clash.
    pub fn union(&self, other: &InstanceSet) -> InstanceSet {
        let mut seen: HashSet<String> = HashSet::new();
        let mut result = Vec::new();
        for inst in &self.instances {
            seen.insert(inst.name().to_string());
            result.push(inst.clone());
        }
        for inst in &other.instances {
            if seen.insert(inst.name().to_string()) {
                result.push(inst.clone());
            }
        }
        InstanceSet::from_instances(&format!("{}_union_{}", self.name, other.name), result)
    }

    /// Set difference: instances in self but not in other.
    pub fn difference(&self, other: &InstanceSet) -> InstanceSet {
        let other_names: HashSet<&str> = other.instances.iter().map(|i| i.name()).collect();
        let insts = self
            .instances
            .iter()
            .filter(|i| !other_names.contains(i.name()))
            .cloned()
            .collect();
        InstanceSet::from_instances(&format!("{}_minus_{}", self.name, other.name), insts)
    }

    /// Collect instance names.
    pub fn names(&self) -> Vec<&str> {
        self.instances.iter().map(|i| i.name()).collect()
    }

    /// Summary statistics about this instance set.
    pub fn summary(&self) -> InstanceSetSummary {
        let count = self.instances.len();
        if count == 0 {
            return InstanceSetSummary {
                count: 0,
                min_vars: 0,
                max_vars: 0,
                mean_vars: 0.0,
                min_constraints: 0,
                max_constraints: 0,
                mean_constraints: 0.0,
                difficulty_counts: HashMap::new(),
                type_counts: HashMap::new(),
                known_optimal_count: 0,
            };
        }
        let vars: Vec<usize> = self
            .instances
            .iter()
            .map(|i| i.metadata.total_vars())
            .collect();
        let cons: Vec<usize> = self
            .instances
            .iter()
            .map(|i| i.metadata.total_constraints())
            .collect();
        let mut difficulty_counts: HashMap<DifficultyClass, usize> = HashMap::new();
        let mut type_counts: HashMap<InstanceType, usize> = HashMap::new();
        let mut known_count = 0usize;
        for inst in &self.instances {
            *difficulty_counts.entry(inst.difficulty()).or_default() += 1;
            *type_counts.entry(inst.instance_type()).or_default() += 1;
            if inst.metadata.has_known_optimal() {
                known_count += 1;
            }
        }
        InstanceSetSummary {
            count,
            min_vars: *vars.iter().min().unwrap_or(&0),
            max_vars: *vars.iter().max().unwrap_or(&0),
            mean_vars: vars.iter().sum::<usize>() as f64 / count as f64,
            min_constraints: *cons.iter().min().unwrap_or(&0),
            max_constraints: *cons.iter().max().unwrap_or(&0),
            mean_constraints: cons.iter().sum::<usize>() as f64 / count as f64,
            difficulty_counts,
            type_counts,
            known_optimal_count: known_count,
        }
    }

    /// Split the set into training and testing subsets.
    /// `train_fraction` is in [0, 1].
    pub fn train_test_split(&self, train_fraction: f64) -> (InstanceSet, InstanceSet) {
        let n = self.instances.len();
        let train_count = ((n as f64) * train_fraction.clamp(0.0, 1.0)).round() as usize;
        let train = InstanceSet::from_instances(
            &format!("{}_train", self.name),
            self.instances[..train_count].to_vec(),
        );
        let test = InstanceSet::from_instances(
            &format!("{}_test", self.name),
            self.instances[train_count..].to_vec(),
        );
        (train, test)
    }

    /// Take the first `n` instances.
    pub fn take(&self, n: usize) -> InstanceSet {
        let count = n.min(self.instances.len());
        InstanceSet::from_instances(
            &format!("{}_first{}", self.name, n),
            self.instances[..count].to_vec(),
        )
    }

    /// Sample `n` instances uniformly at random.
    pub fn sample(&self, n: usize, seed: u64) -> InstanceSet {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let count = n.min(self.instances.len());
        let mut indices: Vec<usize> = (0..self.instances.len()).collect();
        indices.shuffle(&mut rng);
        let sampled: Vec<BenchmarkInstance> = indices[..count]
            .iter()
            .map(|&i| self.instances[i].clone())
            .collect();
        InstanceSet::from_instances(&format!("{}_sample{}", self.name, n), sampled)
    }
}

/// Summary statistics for an instance set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceSetSummary {
    pub count: usize,
    pub min_vars: usize,
    pub max_vars: usize,
    pub mean_vars: f64,
    pub min_constraints: usize,
    pub max_constraints: usize,
    pub mean_constraints: f64,
    pub difficulty_counts: HashMap<DifficultyClass, usize>,
    pub type_counts: HashMap<InstanceType, usize>,
    pub known_optimal_count: usize,
}

impl std::fmt::Display for InstanceSetSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Instances: {}", self.count)?;
        writeln!(
            f,
            "Variables: min={}, max={}, mean={:.1}",
            self.min_vars, self.max_vars, self.mean_vars
        )?;
        writeln!(
            f,
            "Constraints: min={}, max={}, mean={:.1}",
            self.min_constraints, self.max_constraints, self.mean_constraints
        )?;
        writeln!(f, "Known optimal: {}", self.known_optimal_count)?;
        for d in DifficultyClass::all() {
            if let Some(&c) = self.difficulty_counts.get(d) {
                writeln!(f, "  {}: {}", d.label(), c)?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers for building small test instances
// ---------------------------------------------------------------------------

/// Create a trivial bilevel problem for testing.
pub fn make_trivial_instance(
    name: &str,
    n_upper: usize,
    n_lower: usize,
    m_lower: usize,
) -> BenchmarkInstance {
    let problem = BilevelProblem {
        upper_obj_c_x: vec![1.0; n_upper],
        upper_obj_c_y: vec![1.0; n_lower],
        lower_obj_c: vec![1.0; n_lower],
        lower_a: SparseMatrix::new(m_lower, n_lower),
        lower_b: vec![1.0; m_lower],
        lower_linking_b: SparseMatrix::new(m_lower, n_upper),
        upper_constraints_a: SparseMatrix::new(0, n_upper + n_lower),
        upper_constraints_b: vec![],
        num_upper_vars: n_upper,
        num_lower_vars: n_lower,
        num_lower_constraints: m_lower,
        num_upper_constraints: 0,
    };
    BenchmarkInstance::new(name, problem)
}

/// Create a set of trivially sized instances for quick tests.
pub fn make_test_instance_set() -> InstanceSet {
    let mut set = InstanceSet::new("test_set");
    for i in 0..5 {
        let n = 2 + i;
        let inst = make_trivial_instance(&format!("test_{}", i), n, n, n);
        set.push(inst);
    }
    set
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_instance(name: &str, n: usize) -> BenchmarkInstance {
        make_trivial_instance(name, n, n, n)
    }

    #[test]
    fn test_difficulty_from_dimensions() {
        assert_eq!(
            DifficultyClass::from_dimensions(5, 5),
            DifficultyClass::Easy
        );
        assert_eq!(
            DifficultyClass::from_dimensions(50, 60),
            DifficultyClass::Medium
        );
        assert_eq!(
            DifficultyClass::from_dimensions(200, 200),
            DifficultyClass::Hard
        );
        assert_eq!(
            DifficultyClass::from_dimensions(600, 600),
            DifficultyClass::Open
        );
    }

    #[test]
    fn test_instance_type_classify() {
        assert_eq!(
            InstanceType::classify(false, false, false),
            InstanceType::BLP
        );
        assert_eq!(
            InstanceType::classify(true, false, false),
            InstanceType::MIBLP
        );
        assert_eq!(
            InstanceType::classify(true, true, false),
            InstanceType::IBLP
        );
        assert_eq!(
            InstanceType::classify(false, false, true),
            InstanceType::KnapsackInterdiction
        );
    }

    #[test]
    fn test_metadata_from_problem() {
        let inst = sample_instance("foo", 3);
        assert_eq!(inst.metadata.name, "foo");
        assert_eq!(inst.metadata.num_upper_vars, 3);
        assert_eq!(inst.metadata.total_vars(), 6);
    }

    #[test]
    fn test_filter_by_name() {
        let filter = InstanceFilter::new().with_name_contains("bar");
        let inst_yes = sample_instance("foobar", 2);
        let inst_no = sample_instance("baz", 2);
        assert!(filter.matches(&inst_yes.metadata));
        assert!(!filter.matches(&inst_no.metadata));
    }

    #[test]
    fn test_filter_by_difficulty() {
        let filter = InstanceFilter::new().with_difficulties(vec![DifficultyClass::Easy]);
        let easy = sample_instance("small", 2);
        assert!(filter.matches(&easy.metadata));
    }

    #[test]
    fn test_filter_by_vars() {
        let filter = InstanceFilter::new().with_min_vars(10);
        let small = sample_instance("small", 2);
        let big = sample_instance("big", 10);
        assert!(!filter.matches(&small.metadata));
        assert!(filter.matches(&big.metadata));
    }

    #[test]
    fn test_instance_set_intersect() {
        let mut a = InstanceSet::new("a");
        a.push(sample_instance("x", 2));
        a.push(sample_instance("y", 2));
        let mut b = InstanceSet::new("b");
        b.push(sample_instance("y", 2));
        b.push(sample_instance("z", 2));
        let inter = a.intersect(&b);
        assert_eq!(inter.len(), 1);
        assert_eq!(inter.instances[0].name(), "y");
    }

    #[test]
    fn test_instance_set_union() {
        let mut a = InstanceSet::new("a");
        a.push(sample_instance("x", 2));
        let mut b = InstanceSet::new("b");
        b.push(sample_instance("y", 2));
        let u = a.union(&b);
        assert_eq!(u.len(), 2);
    }

    #[test]
    fn test_instance_set_difference() {
        let mut a = InstanceSet::new("a");
        a.push(sample_instance("x", 2));
        a.push(sample_instance("y", 2));
        let mut b = InstanceSet::new("b");
        b.push(sample_instance("y", 2));
        let d = a.difference(&b);
        assert_eq!(d.len(), 1);
        assert_eq!(d.instances[0].name(), "x");
    }

    #[test]
    fn test_instance_set_summary() {
        let set = make_test_instance_set();
        let s = set.summary();
        assert_eq!(s.count, 5);
        assert!(s.mean_vars > 0.0);
    }

    #[test]
    fn test_train_test_split() {
        let set = make_test_instance_set();
        let (train, test) = set.train_test_split(0.6);
        assert_eq!(train.len() + test.len(), set.len());
    }
}
