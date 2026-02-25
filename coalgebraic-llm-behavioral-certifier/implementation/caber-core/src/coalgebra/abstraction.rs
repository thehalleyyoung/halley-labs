//! Abstraction lattices and Galois connections for CEGAR-based refinement.
//!
//! The abstraction framework manages the triple α = (k, n, ε):
//! - k: output alphabet size (number of semantic clusters)
//! - n: maximum input word length (context depth)
//! - ε: approximation tolerance
//!
//! The abstraction lattice organizes these triples with a natural ordering
//! (k₁, n₁, ε₁) ≤ (k₂, n₂, ε₂) iff k₁ ≤ k₂, n₁ ≤ n₂, ε₁ ≥ ε₂.
//! Moving up in the lattice means finer abstraction (more precise but more expensive).

use std::collections::{BTreeMap, HashMap, HashSet, BTreeSet};
use std::fmt;
use std::hash::Hash;

use nalgebra::DMatrix;
use ordered_float::OrderedFloat;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

use super::distribution::{self, SubDistribution};
use super::functor::{BehavioralFunctor, SimpleBehavioralValue};
use super::types::*;
use super::coalgebra::{FiniteCoalgebra, CoalgebraSystem, LLMBehavioralCoalgebra};

// ---------------------------------------------------------------------------
// AbstractionLevel
// ---------------------------------------------------------------------------

/// An abstraction level α = (k, n, ε) specifying the granularity of
/// behavioral analysis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AbstractionLevel {
    /// Output alphabet size (number of semantic clusters).
    pub k: usize,
    /// Maximum input word length (context depth).
    pub n: usize,
    /// Approximation tolerance.
    pub epsilon: f64,
}

impl AbstractionLevel {
    pub fn new(k: usize, n: usize, epsilon: f64) -> Self {
        Self { k, n, epsilon }
    }

    /// The coarsest abstraction level.
    pub fn coarsest() -> Self {
        Self {
            k: 2,
            n: 1,
            epsilon: 1.0,
        }
    }

    /// A fine abstraction level.
    pub fn fine(k: usize, n: usize) -> Self {
        Self {
            k,
            n,
            epsilon: 0.001,
        }
    }

    /// Check if this level is coarser than another.
    pub fn is_coarser_than(&self, other: &Self) -> bool {
        self.k <= other.k && self.n <= other.n && self.epsilon >= other.epsilon
    }

    /// Check if this level is finer than another.
    pub fn is_finer_than(&self, other: &Self) -> bool {
        other.is_coarser_than(self)
    }

    /// Refine the abstraction level by increasing k.
    pub fn refine_k(&self, factor: usize) -> Self {
        Self {
            k: self.k * factor,
            n: self.n,
            epsilon: self.epsilon,
        }
    }

    /// Refine the abstraction level by increasing n.
    pub fn refine_n(&self, increase: usize) -> Self {
        Self {
            k: self.k,
            n: self.n + increase,
            epsilon: self.epsilon,
        }
    }

    /// Refine the abstraction level by decreasing epsilon.
    pub fn refine_epsilon(&self, factor: f64) -> Self {
        Self {
            k: self.k,
            n: self.n,
            epsilon: self.epsilon * factor,
        }
    }

    /// Refine all dimensions.
    pub fn refine_all(&self, k_factor: usize, n_increase: usize, eps_factor: f64) -> Self {
        Self {
            k: self.k * k_factor,
            n: self.n + n_increase,
            epsilon: self.epsilon * eps_factor,
        }
    }

    /// Coarsen the abstraction level.
    pub fn coarsen_k(&self, factor: usize) -> Self {
        Self {
            k: (self.k / factor).max(2),
            n: self.n,
            epsilon: self.epsilon,
        }
    }

    /// Estimated computational cost (proportional to state space size).
    pub fn estimated_cost(&self) -> f64 {
        let alphabet_factor = self.k as f64;
        let depth_factor = (alphabet_factor.powi(self.n as i32)) as f64;
        let precision_factor = 1.0 / self.epsilon;
        alphabet_factor * depth_factor * precision_factor
    }

    /// The tuple representation.
    pub fn as_tuple(&self) -> (usize, usize, f64) {
        (self.k, self.n, self.epsilon)
    }
}

impl fmt::Display for AbstractionLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "α({}, {}, {:.4})", self.k, self.n, self.epsilon)
    }
}

impl Eq for AbstractionLevel {}

impl PartialOrd for AbstractionLevel {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.is_coarser_than(other) && other.is_coarser_than(self) {
            Some(std::cmp::Ordering::Equal)
        } else if self.is_coarser_than(other) {
            Some(std::cmp::Ordering::Less)
        } else if other.is_coarser_than(self) {
            Some(std::cmp::Ordering::Greater)
        } else {
            None // Incomparable
        }
    }
}

// ---------------------------------------------------------------------------
// AbstractionLattice
// ---------------------------------------------------------------------------

/// The lattice of abstraction levels with meet and join operations.
#[derive(Debug, Clone)]
pub struct AbstractionLattice {
    pub levels: Vec<AbstractionLevel>,
    pub current: usize,
}

impl AbstractionLattice {
    /// Create a lattice with a range of abstraction levels.
    pub fn new(levels: Vec<AbstractionLevel>) -> Self {
        Self {
            levels,
            current: 0,
        }
    }

    /// Build a standard lattice with geometric progression.
    pub fn standard(
        k_range: (usize, usize, usize),  // (min, max, step_factor)
        n_range: (usize, usize),          // (min, max)
        eps_range: (f64, f64, f64),       // (max_eps, min_eps, factor)
    ) -> Self {
        let mut levels = Vec::new();
        let mut k = k_range.0;
        while k <= k_range.1 {
            for n in n_range.0..=n_range.1 {
                let mut eps = eps_range.0;
                while eps >= eps_range.1 {
                    levels.push(AbstractionLevel::new(k, n, eps));
                    eps *= eps_range.2;
                }
            }
            k *= k_range.2;
        }
        // Sort from coarsest to finest
        levels.sort_by(|a, b| a.estimated_cost().partial_cmp(&b.estimated_cost()).unwrap());
        Self { levels, current: 0 }
    }

    /// Get the current abstraction level.
    pub fn current_level(&self) -> &AbstractionLevel {
        &self.levels[self.current]
    }

    /// Move to the next finer level.
    pub fn refine(&mut self) -> Option<&AbstractionLevel> {
        if self.current + 1 < self.levels.len() {
            self.current += 1;
            Some(&self.levels[self.current])
        } else {
            None
        }
    }

    /// Move to the next coarser level.
    pub fn coarsen(&mut self) -> Option<&AbstractionLevel> {
        if self.current > 0 {
            self.current -= 1;
            Some(&self.levels[self.current])
        } else {
            None
        }
    }

    /// Find the coarsest level that is finer than the given level.
    pub fn find_refinement(&self, level: &AbstractionLevel) -> Option<&AbstractionLevel> {
        self.levels
            .iter()
            .filter(|l| l.is_finer_than(level) && l != &level)
            .min_by(|a, b| {
                a.estimated_cost()
                    .partial_cmp(&b.estimated_cost())
                    .unwrap()
            })
    }

    /// Find the finest level that is coarser than the given level.
    pub fn find_coarsening(&self, level: &AbstractionLevel) -> Option<&AbstractionLevel> {
        self.levels
            .iter()
            .filter(|l| l.is_coarser_than(level) && l != &level)
            .max_by(|a, b| {
                a.estimated_cost()
                    .partial_cmp(&b.estimated_cost())
                    .unwrap()
            })
    }

    /// Compute the meet (greatest lower bound) of two levels.
    pub fn meet(a: &AbstractionLevel, b: &AbstractionLevel) -> AbstractionLevel {
        AbstractionLevel {
            k: a.k.min(b.k),
            n: a.n.min(b.n),
            epsilon: a.epsilon.max(b.epsilon),
        }
    }

    /// Compute the join (least upper bound) of two levels.
    pub fn join(a: &AbstractionLevel, b: &AbstractionLevel) -> AbstractionLevel {
        AbstractionLevel {
            k: a.k.max(b.k),
            n: a.n.max(b.n),
            epsilon: a.epsilon.min(b.epsilon),
        }
    }

    /// Get the number of levels.
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Check if the lattice is well-formed (all pairs are comparable).
    pub fn is_total_order(&self) -> bool {
        for i in 0..self.levels.len() {
            for j in (i + 1)..self.levels.len() {
                if self.levels[i].partial_cmp(&self.levels[j]).is_none() {
                    return false;
                }
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// AlphabetAbstraction
// ---------------------------------------------------------------------------

/// Alphabet abstraction: clustering output symbols into semantic equivalence classes.
/// Each cluster becomes a single symbol in the abstract alphabet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphabetAbstraction {
    /// Mapping from output symbols to cluster IDs.
    pub symbol_to_cluster: HashMap<String, ClusterId>,
    /// Cluster representatives (centroids).
    pub cluster_centers: Vec<Embedding>,
    /// All symbols in each cluster.
    pub clusters: Vec<Vec<String>>,
    /// Number of clusters.
    pub num_clusters: usize,
}

impl AlphabetAbstraction {
    /// Create from an explicit clustering.
    pub fn from_clusters(clusters: Vec<Vec<String>>) -> Self {
        let num_clusters = clusters.len();
        let mut symbol_to_cluster = HashMap::new();
        for (i, cluster) in clusters.iter().enumerate() {
            for sym in cluster {
                symbol_to_cluster.insert(sym.clone(), ClusterId(i));
            }
        }
        Self {
            symbol_to_cluster,
            cluster_centers: Vec::new(),
            clusters,
            num_clusters,
        }
    }

    /// Build from embeddings using k-means clustering.
    pub fn from_embeddings(
        symbols: &[String],
        embeddings: &[Embedding],
        k: usize,
        max_iterations: usize,
    ) -> Self {
        assert_eq!(symbols.len(), embeddings.len());
        let n = symbols.len();
        if n == 0 || k == 0 {
            return Self::from_clusters(Vec::new());
        }

        let k = k.min(n);
        let dim = embeddings[0].dim();

        // Initialize centers with k-means++ like strategy
        let mut rng = rand::thread_rng();
        let mut centers: Vec<Embedding> = Vec::with_capacity(k);

        // Pick first center randomly
        let first = rng.gen_range(0..n);
        centers.push(embeddings[first].clone());

        // Pick remaining centers proportional to distance from nearest center
        for _ in 1..k {
            let mut distances: Vec<f64> = embeddings
                .iter()
                .map(|e| {
                    centers
                        .iter()
                        .map(|c| e.euclidean_distance(c))
                        .fold(f64::INFINITY, f64::min)
                })
                .collect();

            let total: f64 = distances.iter().sum();
            if total <= 1e-15 {
                // All points are at centers, just pick randomly
                let idx = rng.gen_range(0..n);
                centers.push(embeddings[idx].clone());
                continue;
            }

            // Sample proportional to squared distance
            let u: f64 = rng.gen_range(0.0..total);
            let mut cum = 0.0;
            let mut chosen = 0;
            for (i, d) in distances.iter().enumerate() {
                cum += d;
                if cum >= u {
                    chosen = i;
                    break;
                }
            }
            centers.push(embeddings[chosen].clone());
        }

        // K-means iterations
        let mut assignments = vec![0usize; n];
        for _ in 0..max_iterations {
            // Assignment step
            let mut changed = false;
            for i in 0..n {
                let mut best = 0;
                let mut best_dist = f64::INFINITY;
                for (j, center) in centers.iter().enumerate() {
                    let d = embeddings[i].euclidean_distance(center);
                    if d < best_dist {
                        best_dist = d;
                        best = j;
                    }
                }
                if assignments[i] != best {
                    assignments[i] = best;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update step
            for j in 0..k {
                let cluster_embeddings: Vec<&Embedding> = (0..n)
                    .filter(|&i| assignments[i] == j)
                    .map(|i| &embeddings[i])
                    .collect();

                if let Some(centroid) =
                    Embedding::centroid(&cluster_embeddings.iter().map(|&e| e.clone()).collect::<Vec<_>>())
                {
                    centers[j] = centroid;
                }
            }
        }

        // Build clusters
        let mut clusters: Vec<Vec<String>> = vec![Vec::new(); k];
        let mut symbol_to_cluster = HashMap::new();
        for (i, sym) in symbols.iter().enumerate() {
            let cluster = assignments[i];
            clusters[cluster].push(sym.clone());
            symbol_to_cluster.insert(sym.clone(), ClusterId(cluster));
        }

        // Remove empty clusters
        let non_empty: Vec<(usize, Vec<String>)> = clusters
            .into_iter()
            .enumerate()
            .filter(|(_, c)| !c.is_empty())
            .collect();

        let mut remapped_clusters = Vec::new();
        let mut remap: HashMap<usize, usize> = HashMap::new();
        for (new_id, (old_id, cluster)) in non_empty.into_iter().enumerate() {
            remap.insert(old_id, new_id);
            remapped_clusters.push(cluster);
        }

        // Remap symbol_to_cluster
        for v in symbol_to_cluster.values_mut() {
            if let Some(&new_id) = remap.get(&v.0) {
                *v = ClusterId(new_id);
            }
        }

        let remapped_centers: Vec<Embedding> = remap
            .iter()
            .map(|(&old, &new)| {
                if old < centers.len() {
                    centers[old].clone()
                } else {
                    Embedding::zeros(dim)
                }
            })
            .collect();

        let num_clusters = remapped_clusters.len();

        Self {
            symbol_to_cluster,
            cluster_centers: remapped_centers,
            clusters: remapped_clusters,
            num_clusters,
        }
    }

    /// Get the cluster ID for a symbol.
    pub fn cluster_of(&self, symbol: &str) -> Option<&ClusterId> {
        self.symbol_to_cluster.get(symbol)
    }

    /// Map an output symbol to its abstract version.
    pub fn abstract_symbol(&self, symbol: &OutputSymbol) -> OutputSymbol {
        match self.symbol_to_cluster.get(&symbol.value) {
            Some(cluster) => OutputSymbol::with_cluster(
                format!("C{}", cluster.0),
                cluster.clone(),
            ),
            None => symbol.clone(),
        }
    }

    /// Compute the cluster coherence for each cluster using average pairwise distance.
    pub fn cluster_coherence(&self, embeddings: &HashMap<String, Embedding>) -> Vec<f64> {
        let mut coherences = Vec::new();
        for cluster in &self.clusters {
            if cluster.len() <= 1 {
                coherences.push(1.0);
                continue;
            }
            let embs: Vec<&Embedding> = cluster
                .iter()
                .filter_map(|s| embeddings.get(s))
                .collect();

            let mut total_dist = 0.0;
            let mut count = 0;
            for i in 0..embs.len() {
                for j in (i + 1)..embs.len() {
                    total_dist += embs[i].cosine_distance(embs[j]);
                    count += 1;
                }
            }
            let avg_dist = if count > 0 {
                total_dist / count as f64
            } else {
                0.0
            };
            coherences.push(1.0 - avg_dist);
        }
        coherences
    }

    /// Compute the separation between clusters.
    pub fn cluster_separation(&self, embeddings: &HashMap<String, Embedding>) -> f64 {
        if self.num_clusters <= 1 {
            return 1.0;
        }

        let centroids: Vec<Embedding> = self
            .clusters
            .iter()
            .filter_map(|cluster| {
                let embs: Vec<Embedding> = cluster
                    .iter()
                    .filter_map(|s| embeddings.get(s).cloned())
                    .collect();
                Embedding::centroid(&embs)
            })
            .collect();

        let mut min_sep = f64::INFINITY;
        for i in 0..centroids.len() {
            for j in (i + 1)..centroids.len() {
                let d = centroids[i].euclidean_distance(&centroids[j]);
                min_sep = min_sep.min(d);
            }
        }
        min_sep
    }

    /// Split a cluster into two based on the farthest pair.
    pub fn split_cluster(
        &self,
        cluster_id: usize,
        embeddings: &HashMap<String, Embedding>,
    ) -> Self {
        if cluster_id >= self.clusters.len() || self.clusters[cluster_id].len() <= 1 {
            return self.clone();
        }

        let cluster = &self.clusters[cluster_id];
        let embs: Vec<(String, Embedding)> = cluster
            .iter()
            .filter_map(|s| embeddings.get(s).map(|e| (s.clone(), e.clone())))
            .collect();

        if embs.len() <= 1 {
            return self.clone();
        }

        // Find farthest pair
        let mut max_dist = 0.0;
        let mut seed_a = 0;
        let mut seed_b = 1;
        for i in 0..embs.len() {
            for j in (i + 1)..embs.len() {
                let d = embs[i].1.euclidean_distance(&embs[j].1);
                if d > max_dist {
                    max_dist = d;
                    seed_a = i;
                    seed_b = j;
                }
            }
        }

        // Assign each point to nearer seed
        let mut cluster_a = Vec::new();
        let mut cluster_b = Vec::new();
        for (i, (sym, emb)) in embs.iter().enumerate() {
            let da = emb.euclidean_distance(&embs[seed_a].1);
            let db = emb.euclidean_distance(&embs[seed_b].1);
            if da <= db {
                cluster_a.push(sym.clone());
            } else {
                cluster_b.push(sym.clone());
            }
        }

        // Build new clusters list
        let mut new_clusters = Vec::new();
        for (i, c) in self.clusters.iter().enumerate() {
            if i == cluster_id {
                if !cluster_a.is_empty() {
                    new_clusters.push(cluster_a.clone());
                }
                if !cluster_b.is_empty() {
                    new_clusters.push(cluster_b.clone());
                }
            } else {
                new_clusters.push(c.clone());
            }
        }

        Self::from_clusters(new_clusters)
    }

    /// Merge two clusters.
    pub fn merge_clusters(&self, id1: usize, id2: usize) -> Self {
        if id1 >= self.clusters.len() || id2 >= self.clusters.len() || id1 == id2 {
            return self.clone();
        }

        let (lo, hi) = if id1 < id2 { (id1, id2) } else { (id2, id1) };

        let mut new_clusters = Vec::new();
        let mut merged = self.clusters[lo].clone();
        merged.extend(self.clusters[hi].iter().cloned());

        for (i, c) in self.clusters.iter().enumerate() {
            if i == lo {
                new_clusters.push(merged.clone());
            } else if i != hi {
                new_clusters.push(c.clone());
            }
        }

        Self::from_clusters(new_clusters)
    }

    /// Compute the silhouette score for the clustering.
    pub fn silhouette_score(&self, embeddings: &HashMap<String, Embedding>) -> f64 {
        let mut total_score = 0.0;
        let mut count = 0;

        for (ci, cluster) in self.clusters.iter().enumerate() {
            for sym in cluster {
                if let Some(emb) = embeddings.get(sym) {
                    // a(i): average distance to same-cluster points
                    let same_cluster_embs: Vec<&Embedding> = cluster
                        .iter()
                        .filter(|s| *s != sym)
                        .filter_map(|s| embeddings.get(s))
                        .collect();

                    let a = if same_cluster_embs.is_empty() {
                        0.0
                    } else {
                        same_cluster_embs
                            .iter()
                            .map(|e| emb.euclidean_distance(e))
                            .sum::<f64>()
                            / same_cluster_embs.len() as f64
                    };

                    // b(i): minimum average distance to other clusters
                    let mut b = f64::INFINITY;
                    for (cj, other_cluster) in self.clusters.iter().enumerate() {
                        if cj == ci {
                            continue;
                        }
                        let other_embs: Vec<&Embedding> = other_cluster
                            .iter()
                            .filter_map(|s| embeddings.get(s))
                            .collect();
                        if !other_embs.is_empty() {
                            let avg_dist = other_embs
                                .iter()
                                .map(|e| emb.euclidean_distance(e))
                                .sum::<f64>()
                                / other_embs.len() as f64;
                            b = b.min(avg_dist);
                        }
                    }

                    if b.is_infinite() {
                        continue;
                    }

                    let s = if a.max(b) > 1e-15 {
                        (b - a) / a.max(b)
                    } else {
                        0.0
                    };

                    total_score += s;
                    count += 1;
                }
            }
        }

        if count > 0 {
            total_score / count as f64
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// BehavioralPreClustering
// ---------------------------------------------------------------------------

/// Response-based clustering: group states by their output behavior.
#[derive(Debug, Clone)]
pub struct BehavioralPreClustering {
    pub state_clusters: HashMap<StateId, usize>,
    pub num_clusters: usize,
    pub cluster_representatives: Vec<StateId>,
}

impl BehavioralPreClustering {
    /// Cluster states based on their output distributions.
    pub fn cluster_by_output(
        coalgebra: &FiniteCoalgebra,
        num_clusters: usize,
        tolerance: f64,
    ) -> Self {
        let states = coalgebra.states();
        let n = states.len();
        if n == 0 {
            return Self {
                state_clusters: HashMap::new(),
                num_clusters: 0,
                cluster_representatives: Vec::new(),
            };
        }

        let words = coalgebra.input_words();

        // Compute pairwise behavioral distances
        let mut distances = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let mut max_tv = 0.0f64;
                for w in &words {
                    let d1 = coalgebra.output_distribution(&states[i], w);
                    let d2 = coalgebra.output_distribution(&states[j], w);
                    max_tv = max_tv.max(d1.total_variation(&d2));
                }
                distances[i][j] = max_tv;
                distances[j][i] = max_tv;
            }
        }

        // Agglomerative clustering (single-linkage)
        let mut cluster_assignment = (0..n).collect::<Vec<_>>();
        let mut active_clusters: HashSet<usize> = (0..n).collect();

        while active_clusters.len() > num_clusters {
            // Find closest pair of clusters
            let mut best_dist = f64::INFINITY;
            let mut best_i = 0;
            let mut best_j = 0;

            let active: Vec<usize> = active_clusters.iter().copied().collect();
            for idx_a in 0..active.len() {
                for idx_b in (idx_a + 1)..active.len() {
                    let ca = active[idx_a];
                    let cb = active[idx_b];

                    // Single-linkage: minimum distance between clusters
                    let mut min_dist = f64::INFINITY;
                    for i in 0..n {
                        if cluster_assignment[i] != ca {
                            continue;
                        }
                        for j in 0..n {
                            if cluster_assignment[j] != cb {
                                continue;
                            }
                            min_dist = min_dist.min(distances[i][j]);
                        }
                    }

                    if min_dist < best_dist {
                        best_dist = min_dist;
                        best_i = ca;
                        best_j = cb;
                    }
                }
            }

            // Merge best_j into best_i
            for i in 0..n {
                if cluster_assignment[i] == best_j {
                    cluster_assignment[i] = best_i;
                }
            }
            active_clusters.remove(&best_j);
        }

        // Remap cluster IDs to be contiguous
        let unique_clusters: Vec<usize> = active_clusters.iter().copied().collect();
        let mut remap: HashMap<usize, usize> = HashMap::new();
        for (new_id, &old_id) in unique_clusters.iter().enumerate() {
            remap.insert(old_id, new_id);
        }

        let mut state_clusters = HashMap::new();
        let mut cluster_representatives = vec![StateId::new(""); unique_clusters.len()];

        for (i, state) in states.iter().enumerate() {
            let old_cluster = cluster_assignment[i];
            let new_cluster = remap[&old_cluster];
            state_clusters.insert(state.clone(), new_cluster);
            cluster_representatives[new_cluster] = state.clone();
        }

        Self {
            state_clusters,
            num_clusters: unique_clusters.len(),
            cluster_representatives,
        }
    }

    /// Get the cluster for a state.
    pub fn cluster_of(&self, state: &StateId) -> Option<usize> {
        self.state_clusters.get(state).copied()
    }

    /// Get all states in a cluster.
    pub fn states_in_cluster(&self, cluster: usize) -> Vec<&StateId> {
        self.state_clusters
            .iter()
            .filter(|(_, &c)| c == cluster)
            .map(|(s, _)| s)
            .collect()
    }

    /// Compute the behavioral distance between two clusters.
    pub fn cluster_distance(
        &self,
        c1: usize,
        c2: usize,
        coalgebra: &FiniteCoalgebra,
    ) -> f64 {
        let states1 = self.states_in_cluster(c1);
        let states2 = self.states_in_cluster(c2);

        if states1.is_empty() || states2.is_empty() {
            return 1.0;
        }

        let words = coalgebra.input_words();
        let mut max_dist = 0.0f64;

        for s1 in &states1 {
            for s2 in &states2 {
                for w in &words {
                    let d1 = coalgebra.output_distribution(s1, w);
                    let d2 = coalgebra.output_distribution(s2, w);
                    max_dist = max_dist.max(d1.total_variation(&d2));
                }
            }
        }

        max_dist
    }
}

// ---------------------------------------------------------------------------
// SemanticClustering
// ---------------------------------------------------------------------------

/// Embedding-based clustering with interpolation within clusters.
#[derive(Debug, Clone)]
pub struct SemanticClustering {
    pub embeddings: HashMap<String, Embedding>,
    pub alphabet_abstraction: AlphabetAbstraction,
}

impl SemanticClustering {
    pub fn new(embeddings: HashMap<String, Embedding>, k: usize) -> Self {
        let symbols: Vec<String> = embeddings.keys().cloned().collect();
        let embs: Vec<Embedding> = symbols.iter().map(|s| embeddings[s].clone()).collect();

        let alphabet_abstraction =
            AlphabetAbstraction::from_embeddings(&symbols, &embs, k, 100);

        Self {
            embeddings,
            alphabet_abstraction,
        }
    }

    /// Interpolate an embedding within its cluster.
    pub fn interpolate(&self, symbol: &str, t: f64) -> Option<Embedding> {
        let emb = self.embeddings.get(symbol)?;
        let cluster_id = self.alphabet_abstraction.cluster_of(symbol)?;

        if cluster_id.0 >= self.alphabet_abstraction.cluster_centers.len() {
            return Some(emb.clone());
        }

        let center = &self.alphabet_abstraction.cluster_centers[cluster_id.0];
        let t = t.clamp(0.0, 1.0);

        // Interpolate between embedding and centroid
        Some(Embedding::new(
            emb.values
                .iter()
                .zip(center.values.iter())
                .map(|(&a, &b)| a * (1.0 - t) + b * t)
                .collect(),
        ))
    }

    /// Find the nearest symbol in a given cluster.
    pub fn nearest_in_cluster(&self, query: &Embedding, cluster_id: usize) -> Option<String> {
        let cluster = self.alphabet_abstraction.clusters.get(cluster_id)?;
        cluster
            .iter()
            .filter_map(|s| {
                self.embeddings
                    .get(s)
                    .map(|e| (s.clone(), query.euclidean_distance(e)))
            })
            .min_by(|(_, d1), (_, d2)| d1.partial_cmp(d2).unwrap())
            .map(|(s, _)| s)
    }

    /// Compute inter-cluster distances.
    pub fn inter_cluster_distances(&self) -> Vec<Vec<f64>> {
        let k = self.alphabet_abstraction.num_clusters;
        let mut dists = vec![vec![0.0; k]; k];

        for i in 0..k {
            for j in (i + 1)..k {
                if i < self.alphabet_abstraction.cluster_centers.len()
                    && j < self.alphabet_abstraction.cluster_centers.len()
                {
                    let d = self.alphabet_abstraction.cluster_centers[i]
                        .euclidean_distance(&self.alphabet_abstraction.cluster_centers[j]);
                    dists[i][j] = d;
                    dists[j][i] = d;
                }
            }
        }

        dists
    }
}

// ---------------------------------------------------------------------------
// AbstractionMap
// ---------------------------------------------------------------------------

/// Maps between different abstraction levels.
#[derive(Debug, Clone)]
pub struct AbstractionMap {
    pub source: AbstractionLevel,
    pub target: AbstractionLevel,
    pub state_map: HashMap<StateId, StateId>,
    pub output_map: HashMap<OutputSymbol, OutputSymbol>,
}

impl AbstractionMap {
    pub fn new(source: AbstractionLevel, target: AbstractionLevel) -> Self {
        Self {
            source,
            target,
            state_map: HashMap::new(),
            output_map: HashMap::new(),
        }
    }

    /// Apply the abstraction map to a state.
    pub fn abstract_state(&self, state: &StateId) -> StateId {
        self.state_map
            .get(state)
            .cloned()
            .unwrap_or_else(|| state.clone())
    }

    /// Apply the abstraction map to an output symbol.
    pub fn abstract_output(&self, output: &OutputSymbol) -> OutputSymbol {
        self.output_map
            .get(output)
            .cloned()
            .unwrap_or_else(|| output.clone())
    }

    /// Apply the abstraction map to a coalgebra, producing a coarser coalgebra.
    pub fn abstract_coalgebra(&self, coalgebra: &FiniteCoalgebra) -> FiniteCoalgebra {
        let abstract_states: Vec<StateId> = coalgebra
            .states()
            .iter()
            .map(|s| self.abstract_state(s))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        let abstract_initial: Vec<StateId> = coalgebra
            .initial_states()
            .iter()
            .map(|s| self.abstract_state(s))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        let bf = coalgebra.functor().clone();
        let mut abstract_coalgebra =
            FiniteCoalgebra::new("abstract", abstract_states, abstract_initial, bf);

        for state in coalgebra.states() {
            let bv = coalgebra.structure_map(&state);
            let abstract_source = self.abstract_state(&state);

            for (input, entries) in &bv.transitions {
                for (output, target, prob) in entries {
                    let abstract_output = self.abstract_output(output);
                    let abstract_target = self.abstract_state(target);
                    abstract_coalgebra.add_transition(
                        abstract_source.clone(),
                        input.clone(),
                        abstract_output,
                        abstract_target,
                        *prob,
                    );
                }
            }
        }

        abstract_coalgebra
    }

    /// Compose two abstraction maps.
    pub fn compose(&self, other: &AbstractionMap) -> AbstractionMap {
        let mut composed_state = HashMap::new();
        for (s, t) in &self.state_map {
            let final_target = other
                .state_map
                .get(t)
                .cloned()
                .unwrap_or_else(|| t.clone());
            composed_state.insert(s.clone(), final_target);
        }

        let mut composed_output = HashMap::new();
        for (o, t) in &self.output_map {
            let final_target = other
                .output_map
                .get(t)
                .cloned()
                .unwrap_or_else(|| t.clone());
            composed_output.insert(o.clone(), final_target);
        }

        AbstractionMap {
            source: self.source.clone(),
            target: other.target.clone(),
            state_map: composed_state,
            output_map: composed_output,
        }
    }
}

// ---------------------------------------------------------------------------
// GaloisConnection
// ---------------------------------------------------------------------------

/// A Galois connection (α, γ) between concrete and abstract coalgebras.
/// α: concrete → abstract (abstraction) and γ: abstract → concrete (concretization).
#[derive(Debug, Clone)]
pub struct GaloisConnection {
    pub abstraction_map: AbstractionMap,
    pub concretization_info: ConcretizationInfo,
}

/// Information needed for concretization (inverse of abstraction).
#[derive(Debug, Clone)]
pub struct ConcretizationInfo {
    /// For each abstract state, the set of concrete states it represents.
    pub preimages: HashMap<StateId, Vec<StateId>>,
    /// For each abstract output, the concrete outputs it represents.
    pub output_preimages: HashMap<OutputSymbol, Vec<OutputSymbol>>,
}

impl GaloisConnection {
    pub fn new(
        abstraction_map: AbstractionMap,
        concretization_info: ConcretizationInfo,
    ) -> Self {
        Self {
            abstraction_map,
            concretization_info,
        }
    }

    /// Build a Galois connection from a partition-based abstraction.
    pub fn from_partition(
        partition: &[Vec<StateId>],
        output_clustering: &AlphabetAbstraction,
        source_level: AbstractionLevel,
        target_level: AbstractionLevel,
    ) -> Self {
        let mut state_map = HashMap::new();
        let mut preimages = HashMap::new();

        for (i, class) in partition.iter().enumerate() {
            let abstract_state = StateId::indexed("abs", i);
            preimages.insert(abstract_state.clone(), class.clone());
            for concrete_state in class {
                state_map.insert(concrete_state.clone(), abstract_state.clone());
            }
        }

        let mut output_map = HashMap::new();
        let mut output_preimages = HashMap::new();

        for (i, cluster) in output_clustering.clusters.iter().enumerate() {
            let abstract_output = OutputSymbol::with_cluster(
                format!("C{}", i),
                ClusterId(i),
            );
            let concrete_outputs: Vec<OutputSymbol> = cluster
                .iter()
                .map(|s| OutputSymbol::new(s.as_str()))
                .collect();
            output_preimages.insert(abstract_output.clone(), concrete_outputs.clone());
            for o in &concrete_outputs {
                output_map.insert(o.clone(), abstract_output.clone());
            }
        }

        let abstraction_map = AbstractionMap {
            source: source_level,
            target: target_level,
            state_map,
            output_map,
        };

        let concretization_info = ConcretizationInfo {
            preimages,
            output_preimages,
        };

        Self::new(abstraction_map, concretization_info)
    }

    /// Check the Galois connection property: α(γ(a)) = a for all abstract a.
    pub fn validate_upper_closure(&self) -> bool {
        for (abstract_state, concrete_states) in &self.concretization_info.preimages {
            for cs in concrete_states {
                let mapped = self.abstraction_map.abstract_state(cs);
                if mapped != *abstract_state {
                    return false;
                }
            }
        }
        true
    }

    /// Get the abstraction error bound.
    pub fn abstraction_error(
        &self,
        concrete: &FiniteCoalgebra,
    ) -> f64 {
        let mut max_error = 0.0f64;

        for (abstract_state, concrete_states) in &self.concretization_info.preimages {
            // The error is the maximum behavioral distance within each class
            for i in 0..concrete_states.len() {
                for j in (i + 1)..concrete_states.len() {
                    let s1 = &concrete_states[i];
                    let s2 = &concrete_states[j];

                    for w in concrete.input_words() {
                        let d1 = concrete.output_distribution(s1, &w);
                        let d2 = concrete.output_distribution(s2, &w);
                        let tv = d1.total_variation(&d2);
                        max_error = max_error.max(tv);
                    }
                }
            }
        }

        max_error
    }
}

// ---------------------------------------------------------------------------
// CEGAR loop state management
// ---------------------------------------------------------------------------

/// State of the Counterexample-Guided Abstraction Refinement (CEGAR) loop.
#[derive(Debug, Clone)]
pub struct CEGARState {
    pub current_level: AbstractionLevel,
    pub iteration: usize,
    pub history: Vec<CEGARIteration>,
    pub status: CEGARStatus,
    pub config: AbstractionConfig,
}

/// A single CEGAR iteration record.
#[derive(Debug, Clone)]
pub struct CEGARIteration {
    pub level: AbstractionLevel,
    pub num_states: usize,
    pub bisimulation_distance: f64,
    pub abstraction_error: f64,
    pub counterexample: Option<Vec<Word>>,
    pub refinement_applied: Option<RefinementAction>,
}

/// What refinement was applied.
#[derive(Debug, Clone)]
pub enum RefinementAction {
    SplitCluster { cluster_id: usize },
    IncreaseDepth { new_depth: usize },
    DecreaseEpsilon { new_epsilon: f64 },
    Combined(Vec<RefinementAction>),
}

/// Status of the CEGAR loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CEGARStatus {
    Running,
    Converged,
    MaxIterationsReached,
    InsufficientSamples,
    Failed(String),
}

impl CEGARState {
    pub fn new(initial_level: AbstractionLevel, config: AbstractionConfig) -> Self {
        Self {
            current_level: initial_level,
            iteration: 0,
            history: Vec::new(),
            status: CEGARStatus::Running,
            config,
        }
    }

    /// Record an iteration result and decide on next action.
    pub fn step(
        &mut self,
        num_states: usize,
        bisimulation_distance: f64,
        abstraction_error: f64,
        counterexample: Option<Vec<Word>>,
    ) -> CEGARStatus {
        self.iteration += 1;

        let refinement = if let Some(ref cex) = counterexample {
            if abstraction_error > self.config.split_threshold {
                Some(RefinementAction::SplitCluster { cluster_id: 0 })
            } else {
                Some(RefinementAction::IncreaseDepth {
                    new_depth: self.current_level.n + 1,
                })
            }
        } else {
            None
        };

        self.history.push(CEGARIteration {
            level: self.current_level.clone(),
            num_states,
            bisimulation_distance,
            abstraction_error,
            counterexample: counterexample.clone(),
            refinement_applied: refinement.clone(),
        });

        // Decide status
        if counterexample.is_none() && abstraction_error <= self.current_level.epsilon {
            self.status = CEGARStatus::Converged;
        } else if self.iteration >= self.config.max_refinement_steps {
            self.status = CEGARStatus::MaxIterationsReached;
        } else if let Some(ref action) = refinement {
            // Apply refinement
            match action {
                RefinementAction::SplitCluster { .. } => {
                    self.current_level = self.current_level.refine_k(2);
                }
                RefinementAction::IncreaseDepth { new_depth } => {
                    self.current_level = self.current_level.refine_n(1);
                }
                RefinementAction::DecreaseEpsilon { new_epsilon } => {
                    self.current_level = self.current_level.refine_epsilon(0.5);
                }
                RefinementAction::Combined(actions) => {
                    // Apply all
                    self.current_level = self.current_level.refine_all(2, 1, 0.5);
                }
            }
            self.status = CEGARStatus::Running;
        }

        self.status.clone()
    }

    /// Check if the CEGAR loop is still running.
    pub fn is_running(&self) -> bool {
        self.status == CEGARStatus::Running
    }

    /// Get the convergence rate (ratio of successive abstraction errors).
    pub fn convergence_rate(&self) -> Option<f64> {
        if self.history.len() < 2 {
            return None;
        }
        let last = &self.history[self.history.len() - 1];
        let prev = &self.history[self.history.len() - 2];
        if prev.abstraction_error > 1e-15 {
            Some(last.abstraction_error / prev.abstraction_error)
        } else {
            None
        }
    }

    /// Get the total number of queries made.
    pub fn total_states_explored(&self) -> usize {
        self.history.iter().map(|h| h.num_states).sum()
    }
}

// ---------------------------------------------------------------------------
// Abstraction validation (KS test for cluster coherence)
// ---------------------------------------------------------------------------

/// Validate an abstraction by testing whether states within each cluster
/// have statistically similar behavior.
pub fn validate_abstraction(
    coalgebra: &FiniteCoalgebra,
    clustering: &BehavioralPreClustering,
    alpha: f64,
) -> AbstractionValidationResult {
    let mut results = Vec::new();

    for cluster_id in 0..clustering.num_clusters {
        let states = clustering.states_in_cluster(cluster_id);
        if states.len() <= 1 {
            results.push(ClusterValidation {
                cluster_id,
                num_states: states.len(),
                max_tv_distance: 0.0,
                ks_p_value: 1.0,
                is_coherent: true,
            });
            continue;
        }

        let words = coalgebra.input_words();
        let mut max_tv = 0.0f64;

        for i in 0..states.len() {
            for j in (i + 1)..states.len() {
                for w in &words {
                    let d1 = coalgebra.output_distribution(states[i], w);
                    let d2 = coalgebra.output_distribution(states[j], w);
                    let tv = d1.total_variation(&d2);
                    max_tv = max_tv.max(tv);
                }
            }
        }

        let is_coherent = max_tv <= alpha;

        results.push(ClusterValidation {
            cluster_id,
            num_states: states.len(),
            max_tv_distance: max_tv,
            ks_p_value: 1.0 - max_tv, // Simplified
            is_coherent,
        });
    }

    let all_coherent = results.iter().all(|r| r.is_coherent);
    let max_incoherence = results
        .iter()
        .map(|r| r.max_tv_distance)
        .fold(0.0f64, f64::max);

    AbstractionValidationResult {
        cluster_results: results,
        overall_coherent: all_coherent,
        max_incoherence,
    }
}

/// Result of abstraction validation.
#[derive(Debug, Clone)]
pub struct AbstractionValidationResult {
    pub cluster_results: Vec<ClusterValidation>,
    pub overall_coherent: bool,
    pub max_incoherence: f64,
}

/// Validation result for a single cluster.
#[derive(Debug, Clone)]
pub struct ClusterValidation {
    pub cluster_id: usize,
    pub num_states: usize,
    pub max_tv_distance: f64,
    pub ks_p_value: f64,
    pub is_coherent: bool,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_functor() -> BehavioralFunctor {
        BehavioralFunctor::new(
            vec![Symbol::new("a")],
            vec![OutputSymbol::new("x"), OutputSymbol::new("y")],
            1,
        )
    }

    fn make_test_coalgebra() -> FiniteCoalgebra {
        let bf = make_test_functor();
        let s0 = StateId::new("s0");
        let s1 = StateId::new("s1");
        let mut c = FiniteCoalgebra::new("test", vec![s0.clone(), s1.clone()], vec![s0.clone()], bf);
        let input = Word::from_str_slice(&["a"]);
        c.add_transition(s0.clone(), input.clone(), OutputSymbol::new("x"), s1.clone(), 0.7);
        c.add_transition(s0.clone(), input.clone(), OutputSymbol::new("y"), s0.clone(), 0.3);
        c.add_transition(s1.clone(), input.clone(), OutputSymbol::new("x"), s0.clone(), 1.0);
        c
    }

    #[test]
    fn test_abstraction_level_ordering() {
        let a = AbstractionLevel::new(2, 1, 0.1);
        let b = AbstractionLevel::new(4, 2, 0.05);
        assert!(a.is_coarser_than(&b));
        assert!(b.is_finer_than(&a));
        assert!(!b.is_coarser_than(&a));
    }

    #[test]
    fn test_abstraction_level_incomparable() {
        let a = AbstractionLevel::new(4, 1, 0.1);
        let b = AbstractionLevel::new(2, 2, 0.1);
        // k: 4 > 2, but n: 1 < 2 → incomparable
        assert!(!a.is_coarser_than(&b));
        assert!(!b.is_coarser_than(&a));
        assert!(a.partial_cmp(&b).is_none());
    }

    #[test]
    fn test_abstraction_level_refinement() {
        let a = AbstractionLevel::new(4, 2, 0.1);
        let b = a.refine_k(2);
        assert_eq!(b.k, 8);
        let c = a.refine_n(1);
        assert_eq!(c.n, 3);
        let d = a.refine_epsilon(0.5);
        assert!((d.epsilon - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_abstraction_level_cost() {
        let a = AbstractionLevel::new(2, 1, 0.1);
        let b = AbstractionLevel::new(4, 2, 0.05);
        assert!(a.estimated_cost() < b.estimated_cost());
    }

    #[test]
    fn test_abstraction_lattice_meet_join() {
        let a = AbstractionLevel::new(4, 2, 0.1);
        let b = AbstractionLevel::new(2, 3, 0.05);

        let m = AbstractionLattice::meet(&a, &b);
        assert_eq!(m.k, 2);
        assert_eq!(m.n, 2);
        assert!((m.epsilon - 0.1).abs() < 1e-10);

        let j = AbstractionLattice::join(&a, &b);
        assert_eq!(j.k, 4);
        assert_eq!(j.n, 3);
        assert!((j.epsilon - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_abstraction_lattice_standard() {
        let lattice = AbstractionLattice::standard(
            (2, 8, 2),
            (1, 3),
            (0.5, 0.01, 0.1),
        );
        assert!(lattice.num_levels() > 0);
    }

    #[test]
    fn test_abstraction_lattice_refine() {
        let mut lattice = AbstractionLattice::standard(
            (2, 8, 2),
            (1, 3),
            (0.5, 0.01, 0.1),
        );
        let initial = lattice.current_level().clone();
        let refined = lattice.refine();
        assert!(refined.is_some());
    }

    #[test]
    fn test_alphabet_abstraction() {
        let clusters = vec![
            vec!["hello".to_string(), "hi".to_string()],
            vec!["bye".to_string(), "goodbye".to_string()],
        ];
        let abs = AlphabetAbstraction::from_clusters(clusters);
        assert_eq!(abs.num_clusters, 2);
        assert_eq!(abs.cluster_of("hello"), Some(&ClusterId(0)));
        assert_eq!(abs.cluster_of("bye"), Some(&ClusterId(1)));
    }

    #[test]
    fn test_alphabet_abstraction_merge() {
        let clusters = vec![
            vec!["a".to_string()],
            vec!["b".to_string()],
            vec!["c".to_string()],
        ];
        let abs = AlphabetAbstraction::from_clusters(clusters);
        let merged = abs.merge_clusters(0, 2);
        assert_eq!(merged.num_clusters, 2);
    }

    #[test]
    fn test_behavioral_preclustering() {
        let coalgebra = make_test_coalgebra();
        let clustering = BehavioralPreClustering::cluster_by_output(&coalgebra, 2, 0.1);
        assert!(clustering.num_clusters <= 2);
    }

    #[test]
    fn test_abstraction_map() {
        let coalgebra = make_test_coalgebra();
        let source = AbstractionLevel::new(2, 1, 0.1);
        let target = AbstractionLevel::new(1, 1, 0.2);

        let mut map = AbstractionMap::new(source, target);
        map.state_map.insert(StateId::new("s0"), StateId::new("q0"));
        map.state_map.insert(StateId::new("s1"), StateId::new("q0"));

        let abstract_coalgebra = map.abstract_coalgebra(&coalgebra);
        assert!(abstract_coalgebra.num_states() <= coalgebra.num_states());
    }

    #[test]
    fn test_galois_connection() {
        let partition = vec![
            vec![StateId::new("s0")],
            vec![StateId::new("s1")],
        ];
        let output_clusters = AlphabetAbstraction::from_clusters(vec![
            vec!["x".to_string(), "y".to_string()],
        ]);

        let gc = GaloisConnection::from_partition(
            &partition,
            &output_clusters,
            AbstractionLevel::new(2, 1, 0.1),
            AbstractionLevel::new(1, 1, 0.2),
        );

        assert!(gc.validate_upper_closure());
    }

    #[test]
    fn test_cegar_state() {
        let config = AbstractionConfig::default();
        let mut cegar = CEGARState::new(AbstractionLevel::coarsest(), config);
        assert!(cegar.is_running());

        let status = cegar.step(10, 0.5, 0.2, Some(vec![Word::from_str_slice(&["a"])]));
        assert_eq!(status, CEGARStatus::Running);
        assert_eq!(cegar.iteration, 1);
    }

    #[test]
    fn test_cegar_convergence() {
        let config = AbstractionConfig::default();
        let mut cegar = CEGARState::new(AbstractionLevel::new(4, 2, 0.3), config);

        // Converge when no counterexample and error is small
        let status = cegar.step(10, 0.01, 0.01, None);
        assert_eq!(status, CEGARStatus::Converged);
    }

    #[test]
    fn test_validate_abstraction() {
        let coalgebra = make_test_coalgebra();
        let clustering = BehavioralPreClustering::cluster_by_output(&coalgebra, 2, 0.1);
        let result = validate_abstraction(&coalgebra, &clustering, 0.5);
        assert!(!result.cluster_results.is_empty());
    }

    #[test]
    fn test_abstraction_error() {
        let coalgebra = make_test_coalgebra();
        let partition = vec![
            vec![StateId::new("s0"), StateId::new("s1")],
        ];
        let output_clusters = AlphabetAbstraction::from_clusters(vec![
            vec!["x".to_string(), "y".to_string()],
        ]);

        let gc = GaloisConnection::from_partition(
            &partition,
            &output_clusters,
            AbstractionLevel::new(2, 1, 0.1),
            AbstractionLevel::new(1, 1, 0.5),
        );

        let error = gc.abstraction_error(&coalgebra);
        assert!(error >= 0.0);
    }

    #[test]
    fn test_alphabet_from_embeddings() {
        let symbols = vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()];
        let embeddings = vec![
            Embedding::new(vec![1.0, 0.0]),
            Embedding::new(vec![1.1, 0.1]),
            Embedding::new(vec![0.0, 1.0]),
            Embedding::new(vec![0.1, 1.1]),
        ];
        let abs = AlphabetAbstraction::from_embeddings(&symbols, &embeddings, 2, 50);
        assert_eq!(abs.num_clusters, 2);
        // a and b should be in the same cluster, c and d in another
        let ca = abs.cluster_of("a").unwrap();
        let cb = abs.cluster_of("b").unwrap();
        assert_eq!(ca, cb);
    }

    #[test]
    fn test_semantic_clustering() {
        let mut embeddings = HashMap::new();
        embeddings.insert("a".to_string(), Embedding::new(vec![1.0, 0.0]));
        embeddings.insert("b".to_string(), Embedding::new(vec![0.0, 1.0]));

        let clustering = SemanticClustering::new(embeddings, 2);
        assert_eq!(clustering.alphabet_abstraction.num_clusters, 2);
    }

    #[test]
    fn test_cluster_coherence() {
        let clusters = vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["c".to_string()],
        ];
        let abs = AlphabetAbstraction::from_clusters(clusters);

        let mut embeddings = HashMap::new();
        embeddings.insert("a".to_string(), Embedding::new(vec![1.0, 0.0]));
        embeddings.insert("b".to_string(), Embedding::new(vec![0.9, 0.1]));
        embeddings.insert("c".to_string(), Embedding::new(vec![0.0, 1.0]));

        let coherences = abs.cluster_coherence(&embeddings);
        assert_eq!(coherences.len(), 2);
        assert!(coherences[0] > 0.5); // a and b are close
    }

    #[test]
    fn test_cegar_convergence_rate() {
        let config = AbstractionConfig::default();
        let mut cegar = CEGARState::new(AbstractionLevel::coarsest(), config);

        cegar.step(10, 0.5, 0.5, Some(vec![Word::from_str_slice(&["a"])]));
        cegar.step(20, 0.3, 0.2, Some(vec![Word::from_str_slice(&["b"])]));

        let rate = cegar.convergence_rate();
        assert!(rate.is_some());
        assert!(rate.unwrap() < 1.0); // Error decreased
    }
}
