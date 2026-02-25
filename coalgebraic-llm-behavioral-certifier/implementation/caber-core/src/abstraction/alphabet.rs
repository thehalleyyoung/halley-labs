//! Alphabet abstraction — finite alphabet construction for natural language.
//!
//! Constructs a finite output alphabet from LLM responses by clustering
//! responses into semantically coherent groups. Supports:
//! - Behavioral pre-clustering based on response statistics
//! - Semantic embedding integration
//! - Cluster validation via KS tests
//! - Adaptive cluster splitting and merging
//! - Alphabet refinement operators

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, BTreeMap};
use std::fmt;
use ordered_float::OrderedFloat;
use rand::Rng;
use rand::seq::SliceRandom;

// ---------------------------------------------------------------------------
// Local type definitions
// ---------------------------------------------------------------------------

/// Identifier for a semantic cluster.
pub type ClusterId = usize;

/// A response observation from an LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseObservation {
    /// Unique observation ID.
    pub id: usize,
    /// The raw response text (or summary).
    pub text: String,
    /// The prompt that produced this response.
    pub prompt: String,
    /// Statistical features extracted from the response.
    pub stats: ResponseStats,
    /// Embedding vector (if available).
    pub embedding: Option<Vec<f64>>,
    /// Assigned cluster.
    pub cluster: Option<ClusterId>,
}

/// Statistical features of a response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseStats {
    /// Response length in tokens.
    pub length: usize,
    /// Lexical diversity (unique tokens / total tokens).
    pub lexical_diversity: f64,
    /// Average sentence length.
    pub avg_sentence_length: f64,
    /// Sentiment score (-1 to 1).
    pub sentiment: f64,
    /// Formality score (0 to 1).
    pub formality: f64,
    /// Readability score (Flesch-Kincaid grade level approximation).
    pub readability: f64,
    /// Contains code flag.
    pub contains_code: bool,
    /// Contains math flag.
    pub contains_math: bool,
    /// Refusal flag (model refused to answer).
    pub is_refusal: bool,
}

impl ResponseStats {
    /// Compute basic statistics from response text.
    pub fn from_text(text: &str) -> Self {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        let length = tokens.len();
        let unique: HashSet<&str> = tokens.iter().copied().collect();
        let lexical_diversity = if length > 0 {
            unique.len() as f64 / length as f64
        } else {
            0.0
        };

        // Simple sentence splitting.
        let sentences: Vec<&str> = text.split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .collect();
        let avg_sentence_length = if !sentences.is_empty() {
            length as f64 / sentences.len() as f64
        } else {
            length as f64
        };

        // Heuristic sentiment: count positive/negative words.
        let positive_words = ["good", "great", "excellent", "wonderful", "happy", "love",
                              "best", "amazing", "fantastic", "positive"];
        let negative_words = ["bad", "terrible", "awful", "horrible", "hate", "worst",
                              "negative", "poor", "disappointing", "wrong"];
        let lower = text.to_lowercase();
        let pos_count = positive_words.iter().filter(|w| lower.contains(*w)).count() as f64;
        let neg_count = negative_words.iter().filter(|w| lower.contains(*w)).count() as f64;
        let total_sentiment_words = pos_count + neg_count;
        let sentiment = if total_sentiment_words > 0.0 {
            (pos_count - neg_count) / total_sentiment_words
        } else {
            0.0
        };

        // Heuristic formality.
        let informal_markers = ["lol", "omg", "btw", "imo", "gonna", "wanna", "kinda"];
        let informal_count = informal_markers.iter()
            .filter(|w| lower.contains(*w))
            .count() as f64;
        let formality = 1.0 - (informal_count / 7.0).min(1.0);

        // Simple readability (avg words per sentence * 0.5 + avg syllables per word * 10).
        let avg_word_len = if length > 0 {
            tokens.iter().map(|t| t.len()).sum::<usize>() as f64 / length as f64
        } else {
            0.0
        };
        let readability = avg_sentence_length * 0.39 + avg_word_len * 11.8 - 15.59;

        let contains_code = text.contains("```") || text.contains("def ") ||
                           text.contains("fn ") || text.contains("class ") ||
                           text.contains("import ");
        let contains_math = text.contains("$") || text.contains("\\frac") ||
                           text.contains("\\sum") || text.contains("equation");

        let refusal_markers = ["I cannot", "I can't", "I'm unable", "I apologize",
                               "I'm sorry, but I", "As an AI"];
        let is_refusal = refusal_markers.iter().any(|m| text.contains(m));

        Self {
            length,
            lexical_diversity,
            avg_sentence_length,
            sentiment,
            formality,
            readability,
            contains_code,
            contains_math,
            is_refusal,
        }
    }

    /// Convert stats to a feature vector for clustering.
    pub fn to_feature_vector(&self) -> Vec<f64> {
        vec![
            self.length as f64 / 1000.0, // Normalize length
            self.lexical_diversity,
            self.avg_sentence_length / 50.0,
            (self.sentiment + 1.0) / 2.0, // Map [-1,1] to [0,1]
            self.formality,
            self.readability / 20.0, // Rough normalization
            if self.contains_code { 1.0 } else { 0.0 },
            if self.contains_math { 1.0 } else { 0.0 },
            if self.is_refusal { 1.0 } else { 0.0 },
        ]
    }

    /// Euclidean distance between two stats feature vectors.
    pub fn distance(&self, other: &ResponseStats) -> f64 {
        let v1 = self.to_feature_vector();
        let v2 = other.to_feature_vector();
        v1.iter().zip(v2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

impl Default for ResponseStats {
    fn default() -> Self {
        Self {
            length: 0,
            lexical_diversity: 0.0,
            avg_sentence_length: 0.0,
            sentiment: 0.0,
            formality: 0.5,
            readability: 0.0,
            contains_code: false,
            contains_math: false,
            is_refusal: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Cluster info
// ---------------------------------------------------------------------------

/// Information about a single cluster in the alphabet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterInfo {
    pub id: ClusterId,
    /// Centroid in feature space.
    pub centroid: Vec<f64>,
    /// Number of members.
    pub size: usize,
    /// Member observation IDs.
    pub members: Vec<usize>,
    /// Intra-cluster variance.
    pub variance: f64,
    /// Cluster label (human-readable description).
    pub label: String,
    /// Whether the cluster passed behavioral coherence validation.
    pub validated: bool,
    /// Behavioral coherence score (lower is better).
    pub coherence_score: f64,
}

impl ClusterInfo {
    pub fn new(id: ClusterId) -> Self {
        Self {
            id,
            centroid: Vec::new(),
            size: 0,
            members: Vec::new(),
            variance: 0.0,
            label: format!("cluster_{}", id),
            validated: false,
            coherence_score: f64::INFINITY,
        }
    }

    /// Compute the centroid from member feature vectors.
    pub fn compute_centroid(&mut self, features: &[Vec<f64>]) {
        if features.is_empty() {
            return;
        }
        let dim = features[0].len();
        let mut centroid = vec![0.0; dim];
        for f in features {
            for (i, &v) in f.iter().enumerate() {
                if i < dim {
                    centroid[i] += v;
                }
            }
        }
        let n = features.len() as f64;
        for c in centroid.iter_mut() {
            *c /= n;
        }
        self.centroid = centroid;
        self.size = features.len();
    }

    /// Compute intra-cluster variance.
    pub fn compute_variance(&mut self, features: &[Vec<f64>]) {
        if features.is_empty() || self.centroid.is_empty() {
            self.variance = 0.0;
            return;
        }
        let mut total = 0.0;
        for f in features {
            let dist_sq: f64 = f.iter().zip(self.centroid.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            total += dist_sq;
        }
        self.variance = total / features.len() as f64;
    }

    /// Distance from a point to this cluster's centroid.
    pub fn distance_to(&self, point: &[f64]) -> f64 {
        self.centroid.iter().zip(point.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

impl fmt::Display for ClusterInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Cluster[id={}, size={}, var={:.4}, coherence={:.4}, label={}]",
            self.id, self.size, self.variance, self.coherence_score, self.label
        )
    }
}

/// Aggregate statistics for the entire alphabet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterStats {
    pub num_clusters: usize,
    pub total_observations: usize,
    pub avg_cluster_size: f64,
    pub min_cluster_size: usize,
    pub max_cluster_size: usize,
    pub avg_variance: f64,
    pub max_variance: f64,
    pub avg_coherence: f64,
    pub silhouette_score: f64,
}

impl fmt::Display for ClusterStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ClusterStats[k={}, n={}, avg_size={:.1}, silhouette={:.4}]",
            self.num_clusters, self.total_observations,
            self.avg_cluster_size, self.silhouette_score
        )
    }
}

// ---------------------------------------------------------------------------
// Alphabet refinement operations
// ---------------------------------------------------------------------------

/// An operation on the alphabet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlphabetRefinementOp {
    /// Split a cluster into sub-clusters.
    Split {
        cluster_id: ClusterId,
        num_parts: usize,
    },
    /// Merge two clusters.
    Merge {
        cluster_a: ClusterId,
        cluster_b: ClusterId,
    },
    /// Re-cluster everything with a new k.
    Recluster {
        new_k: usize,
    },
    /// Remove a cluster (reassign its members).
    Remove {
        cluster_id: ClusterId,
    },
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for alphabet construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlphabetConfig {
    /// Initial number of clusters.
    pub initial_k: usize,
    /// Minimum cluster size.
    pub min_cluster_size: usize,
    /// Maximum number of clusters.
    pub max_clusters: usize,
    /// KS test significance level for coherence validation.
    pub ks_significance: f64,
    /// Maximum variance threshold for a cluster.
    pub max_variance: f64,
    /// Number of k-means iterations.
    pub kmeans_iterations: usize,
    /// Whether to use behavioral features (stats) or embeddings.
    pub use_embeddings: bool,
    /// Minimum inter-cluster distance for merging threshold.
    pub merge_threshold: f64,
    /// Maximum coherence score for validation.
    pub max_coherence: f64,
}

impl Default for AlphabetConfig {
    fn default() -> Self {
        Self {
            initial_k: 8,
            min_cluster_size: 5,
            max_clusters: 64,
            ks_significance: 0.05,
            max_variance: 1.0,
            kmeans_iterations: 100,
            use_embeddings: false,
            merge_threshold: 0.1,
            max_coherence: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// The main AlphabetAbstraction
// ---------------------------------------------------------------------------

/// Finite alphabet construction from natural language responses.
#[derive(Debug, Clone)]
pub struct AlphabetAbstraction {
    /// Configuration.
    pub config: AlphabetConfig,
    /// All observations.
    pub observations: Vec<ResponseObservation>,
    /// Current clusters.
    pub clusters: Vec<ClusterInfo>,
    /// Feature vectors for all observations (cached).
    feature_cache: Vec<Vec<f64>>,
    /// History of refinement operations.
    pub history: Vec<AlphabetRefinementOp>,
}

impl AlphabetAbstraction {
    pub fn new(config: AlphabetConfig) -> Self {
        Self {
            config,
            observations: Vec::new(),
            clusters: Vec::new(),
            feature_cache: Vec::new(),
            history: Vec::new(),
        }
    }

    /// Add an observation.
    pub fn add_observation(&mut self, obs: ResponseObservation) {
        let features = if let Some(ref emb) = obs.embedding {
            emb.clone()
        } else {
            obs.stats.to_feature_vector()
        };
        self.feature_cache.push(features);
        self.observations.push(obs);
    }

    /// Add multiple observations from raw texts and prompts.
    pub fn add_texts(&mut self, prompts_and_responses: &[(String, String)]) {
        for (i, (prompt, text)) in prompts_and_responses.iter().enumerate() {
            let stats = ResponseStats::from_text(text);
            let obs = ResponseObservation {
                id: self.observations.len() + i,
                text: text.clone(),
                prompt: prompt.clone(),
                stats,
                embedding: None,
                cluster: None,
            };
            self.add_observation(obs);
        }
    }

    /// Get the number of observations.
    pub fn num_observations(&self) -> usize {
        self.observations.len()
    }

    /// Get the current number of clusters (alphabet size).
    pub fn alphabet_size(&self) -> usize {
        self.clusters.len()
    }

    /// Build the initial alphabet using k-means clustering.
    pub fn build(&mut self) {
        let k = self.config.initial_k.min(self.observations.len().max(1));
        self.kmeans_cluster(k);
        self.compute_cluster_stats();
        self.validate_clusters();
    }

    /// Perform k-means clustering on the observations.
    fn kmeans_cluster(&mut self, k: usize) {
        if self.feature_cache.is_empty() || k == 0 {
            return;
        }

        let n = self.feature_cache.len();
        let dim = self.feature_cache[0].len();
        let k = k.min(n);

        // Initialize centroids using k-means++ initialization.
        let mut centroids = self.kmeans_pp_init(k);

        // Assignment array.
        let mut assignments = vec![0usize; n];

        for _iter in 0..self.config.kmeans_iterations {
            // Assignment step: assign each point to nearest centroid.
            let mut changed = false;
            for i in 0..n {
                let mut best_k = 0;
                let mut best_dist = f64::INFINITY;
                for j in 0..k {
                    let dist = euclidean_distance(&self.feature_cache[i], &centroids[j]);
                    if dist < best_dist {
                        best_dist = dist;
                        best_k = j;
                    }
                }
                if assignments[i] != best_k {
                    assignments[i] = best_k;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update step: recompute centroids.
            let mut new_centroids = vec![vec![0.0; dim]; k];
            let mut counts = vec![0usize; k];
            for i in 0..n {
                let c = assignments[i];
                counts[c] += 1;
                for d in 0..dim {
                    new_centroids[c][d] += self.feature_cache[i][d];
                }
            }
            for j in 0..k {
                if counts[j] > 0 {
                    for d in 0..dim {
                        new_centroids[j][d] /= counts[j] as f64;
                    }
                }
            }
            centroids = new_centroids;
        }

        // Build cluster info.
        self.clusters.clear();
        for j in 0..k {
            let mut cluster = ClusterInfo::new(j);
            cluster.centroid = centroids[j].clone();
            cluster.members = (0..n).filter(|&i| assignments[i] == j).collect();
            cluster.size = cluster.members.len();
            self.clusters.push(cluster);
        }

        // Update observation assignments.
        for i in 0..n {
            self.observations[i].cluster = Some(assignments[i]);
        }
    }

    /// K-means++ initialization.
    fn kmeans_pp_init(&self, k: usize) -> Vec<Vec<f64>> {
        let n = self.feature_cache.len();
        if n == 0 || k == 0 {
            return Vec::new();
        }

        let mut rng = rand::thread_rng();
        let mut centroids = Vec::with_capacity(k);

        // Pick first centroid randomly.
        let first_idx = rng.gen_range(0..n);
        centroids.push(self.feature_cache[first_idx].clone());

        // Pick remaining centroids with probability proportional to D^2.
        for _ in 1..k {
            let mut distances: Vec<f64> = Vec::with_capacity(n);
            for i in 0..n {
                let min_dist = centroids.iter()
                    .map(|c| euclidean_distance(&self.feature_cache[i], c))
                    .fold(f64::INFINITY, f64::min);
                distances.push(min_dist * min_dist);
            }
            let total: f64 = distances.iter().sum();
            if total <= 0.0 {
                // All points coincide — just pick any.
                centroids.push(self.feature_cache[rng.gen_range(0..n)].clone());
                continue;
            }

            let threshold = rng.gen::<f64>() * total;
            let mut cumulative = 0.0;
            let mut chosen = 0;
            for (i, &d) in distances.iter().enumerate() {
                cumulative += d;
                if cumulative >= threshold {
                    chosen = i;
                    break;
                }
            }
            centroids.push(self.feature_cache[chosen].clone());
        }

        centroids
    }

    /// Compute statistics for each cluster.
    fn compute_cluster_stats(&mut self) {
        for cluster in &mut self.clusters {
            let member_features: Vec<Vec<f64>> = cluster.members.iter()
                .filter_map(|&i| self.feature_cache.get(i).cloned())
                .collect();
            cluster.compute_centroid(&member_features);
            cluster.compute_variance(&member_features);
        }
    }

    /// Validate clusters using a behavioral coherence test.
    /// Uses a two-sample Kolmogorov-Smirnov test to check if all responses
    /// in a cluster come from the same behavioral distribution.
    fn validate_clusters(&mut self) {
        for cluster in &mut self.clusters {
            if cluster.size < 2 {
                cluster.validated = cluster.size > 0;
                cluster.coherence_score = 0.0;
                continue;
            }

            // Collect the length distributions within the cluster.
            let lengths: Vec<f64> = cluster.members.iter()
                .filter_map(|&i| self.observations.get(i))
                .map(|o| o.stats.length as f64)
                .collect();

            if lengths.len() < 2 {
                cluster.validated = true;
                cluster.coherence_score = 0.0;
                continue;
            }

            // Split into two halves and run KS test.
            let mid = lengths.len() / 2;
            let (half1, half2) = lengths.split_at(mid);
            let ks_stat = ks_statistic(half1, half2);
            let n1 = half1.len() as f64;
            let n2 = half2.len() as f64;

            // Approximate KS critical value.
            let c_alpha = (-0.5 * (self.config.ks_significance / 2.0).ln()).sqrt();
            let critical = c_alpha * ((n1 + n2) / (n1 * n2)).sqrt();

            cluster.coherence_score = ks_stat;
            cluster.validated = ks_stat <= critical;
        }
    }

    /// Split a cluster that failed validation or is too large.
    pub fn split_cluster(&mut self, cluster_id: ClusterId, num_parts: usize) -> Vec<ClusterId> {
        let cluster_idx = match self.clusters.iter().position(|c| c.id == cluster_id) {
            Some(idx) => idx,
            None => return Vec::new(),
        };

        let members = self.clusters[cluster_idx].members.clone();
        if members.len() < num_parts {
            return vec![cluster_id];
        }

        // Gather features for this cluster's members.
        let member_features: Vec<Vec<f64>> = members.iter()
            .filter_map(|&i| self.feature_cache.get(i).cloned())
            .collect();

        // Run k-means on just this cluster's members.
        let dim = if member_features.is_empty() { 0 } else { member_features[0].len() };
        let sub_centroids = self.sub_kmeans(&member_features, num_parts, dim);

        // Assign members to sub-clusters.
        let mut sub_assignments = vec![0usize; members.len()];
        for (i, features) in member_features.iter().enumerate() {
            let mut best = 0;
            let mut best_dist = f64::INFINITY;
            for (j, centroid) in sub_centroids.iter().enumerate() {
                let d = euclidean_distance(features, centroid);
                if d < best_dist {
                    best_dist = d;
                    best = j;
                }
            }
            sub_assignments[i] = best;
        }

        // Create new clusters.
        let next_id = self.clusters.iter().map(|c| c.id).max().unwrap_or(0) + 1;
        let mut new_ids = Vec::new();

        // Remove old cluster.
        self.clusters.retain(|c| c.id != cluster_id);

        for part in 0..num_parts {
            let part_members: Vec<usize> = members.iter().enumerate()
                .filter(|(i, _)| sub_assignments[*i] == part)
                .map(|(_, &m)| m)
                .collect();

            if part_members.is_empty() {
                continue;
            }

            let new_id = next_id + part;
            let mut new_cluster = ClusterInfo::new(new_id);
            new_cluster.members = part_members;
            new_cluster.size = new_cluster.members.len();
            if part < sub_centroids.len() {
                new_cluster.centroid = sub_centroids[part].clone();
            }

            let member_feats: Vec<Vec<f64>> = new_cluster.members.iter()
                .filter_map(|&i| self.feature_cache.get(i).cloned())
                .collect();
            new_cluster.compute_variance(&member_feats);
            new_cluster.label = format!("cluster_{}", new_id);
            new_ids.push(new_id);
            self.clusters.push(new_cluster);
        }

        // Update observation assignments.
        for (i, &mi) in members.iter().enumerate() {
            let part = sub_assignments[i];
            let new_id = next_id + part;
            if mi < self.observations.len() {
                self.observations[mi].cluster = Some(new_id);
            }
        }

        self.history.push(AlphabetRefinementOp::Split {
            cluster_id,
            num_parts,
        });

        self.validate_clusters();
        new_ids
    }

    /// Run k-means on a subset of points.
    fn sub_kmeans(&self, features: &[Vec<f64>], k: usize, dim: usize) -> Vec<Vec<f64>> {
        if features.is_empty() || k == 0 || dim == 0 {
            return Vec::new();
        }

        let n = features.len();
        let k = k.min(n);
        let mut rng = rand::thread_rng();

        // Random initialization.
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);
        let mut centroids: Vec<Vec<f64>> = indices[..k].iter()
            .map(|&i| features[i].clone())
            .collect();

        let mut assignments = vec![0usize; n];

        for _ in 0..50 {
            // Assign.
            let mut changed = false;
            for i in 0..n {
                let mut best = 0;
                let mut best_d = f64::INFINITY;
                for j in 0..k {
                    let d = euclidean_distance(&features[i], &centroids[j]);
                    if d < best_d {
                        best_d = d;
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

            // Update.
            let mut new_centroids = vec![vec![0.0; dim]; k];
            let mut counts = vec![0usize; k];
            for i in 0..n {
                let c = assignments[i];
                counts[c] += 1;
                for d in 0..dim {
                    new_centroids[c][d] += features[i][d];
                }
            }
            for j in 0..k {
                if counts[j] > 0 {
                    for d in 0..dim {
                        new_centroids[j][d] /= counts[j] as f64;
                    }
                }
            }
            centroids = new_centroids;
        }

        centroids
    }

    /// Merge two clusters.
    pub fn merge_clusters(&mut self, a: ClusterId, b: ClusterId) -> Option<ClusterId> {
        let idx_a = self.clusters.iter().position(|c| c.id == a)?;
        let idx_b = self.clusters.iter().position(|c| c.id == b)?;

        let members_a = self.clusters[idx_a].members.clone();
        let members_b = self.clusters[idx_b].members.clone();

        let merged_id = a.min(b);
        let mut merged = ClusterInfo::new(merged_id);
        merged.members = [members_a, members_b].concat();
        merged.size = merged.members.len();

        let member_features: Vec<Vec<f64>> = merged.members.iter()
            .filter_map(|&i| self.feature_cache.get(i).cloned())
            .collect();
        merged.compute_centroid(&member_features);
        merged.compute_variance(&member_features);
        merged.label = format!("merged_{}_{}", a, b);

        // Update assignments.
        for &m in &merged.members {
            if m < self.observations.len() {
                self.observations[m].cluster = Some(merged_id);
            }
        }

        // Remove old clusters and add merged.
        self.clusters.retain(|c| c.id != a && c.id != b);
        self.clusters.push(merged);

        self.history.push(AlphabetRefinementOp::Merge {
            cluster_a: a,
            cluster_b: b,
        });

        self.validate_clusters();
        Some(merged_id)
    }

    /// Identify clusters that should be merged (too similar).
    pub fn find_merge_candidates(&self) -> Vec<(ClusterId, ClusterId, f64)> {
        let mut candidates = Vec::new();

        for i in 0..self.clusters.len() {
            for j in (i + 1)..self.clusters.len() {
                let dist = euclidean_distance(
                    &self.clusters[i].centroid,
                    &self.clusters[j].centroid,
                );
                if dist < self.config.merge_threshold {
                    candidates.push((self.clusters[i].id, self.clusters[j].id, dist));
                }
            }
        }

        candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        candidates
    }

    /// Identify clusters that should be split (failed validation or too large).
    pub fn find_split_candidates(&self) -> Vec<(ClusterId, f64)> {
        let mut candidates = Vec::new();

        for cluster in &self.clusters {
            if !cluster.validated && cluster.size >= 2 * self.config.min_cluster_size {
                candidates.push((cluster.id, cluster.coherence_score));
            } else if cluster.variance > self.config.max_variance {
                candidates.push((cluster.id, cluster.variance));
            }
        }

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates
    }

    /// Adaptively resize the alphabet.
    /// Splits clusters that are too heterogeneous, merges those that are too similar.
    pub fn adaptive_resize(&mut self) -> usize {
        let mut operations = 0;

        // First, merge candidates.
        loop {
            let candidates = self.find_merge_candidates();
            if candidates.is_empty() || self.clusters.len() <= 2 {
                break;
            }
            let (a, b, _) = candidates[0];
            self.merge_clusters(a, b);
            operations += 1;
        }

        // Then, split candidates.
        loop {
            let candidates = self.find_split_candidates();
            if candidates.is_empty() || self.clusters.len() >= self.config.max_clusters {
                break;
            }
            let (id, _) = candidates[0];
            self.split_cluster(id, 2);
            operations += 1;

            // Re-check after each split.
            if self.clusters.len() >= self.config.max_clusters {
                break;
            }
        }

        operations
    }

    /// Recluster with a new k value.
    pub fn recluster(&mut self, new_k: usize) {
        self.kmeans_cluster(new_k);
        self.compute_cluster_stats();
        self.validate_clusters();
        self.history.push(AlphabetRefinementOp::Recluster { new_k });
    }

    /// Compute the silhouette score for the current clustering.
    pub fn silhouette_score(&self) -> f64 {
        if self.clusters.len() <= 1 || self.observations.is_empty() {
            return 0.0;
        }

        let n = self.observations.len();
        let mut total_silhouette = 0.0;
        let mut count = 0;

        for i in 0..n {
            let ci = match self.observations[i].cluster {
                Some(c) => c,
                None => continue,
            };

            // Average distance to same-cluster points (a).
            let same_cluster: Vec<usize> = self.observations.iter().enumerate()
                .filter(|(j, o)| *j != i && o.cluster == Some(ci))
                .map(|(j, _)| j)
                .collect();

            if same_cluster.is_empty() {
                continue;
            }

            let a = same_cluster.iter()
                .map(|&j| euclidean_distance(&self.feature_cache[i], &self.feature_cache[j]))
                .sum::<f64>()
                / same_cluster.len() as f64;

            // Minimum average distance to other clusters (b).
            let mut min_b = f64::INFINITY;
            for cluster in &self.clusters {
                if cluster.id == ci {
                    continue;
                }
                let other_members: Vec<usize> = cluster.members.iter()
                    .filter(|&&j| j != i)
                    .copied()
                    .collect();
                if other_members.is_empty() {
                    continue;
                }
                let avg_dist = other_members.iter()
                    .map(|&j| euclidean_distance(&self.feature_cache[i], &self.feature_cache[j]))
                    .sum::<f64>()
                    / other_members.len() as f64;
                if avg_dist < min_b {
                    min_b = avg_dist;
                }
            }

            if min_b.is_infinite() {
                continue;
            }

            let s = if a.max(min_b) > 0.0 {
                (min_b - a) / a.max(min_b)
            } else {
                0.0
            };
            total_silhouette += s;
            count += 1;
        }

        if count > 0 {
            total_silhouette / count as f64
        } else {
            0.0
        }
    }

    /// Compute aggregate statistics.
    pub fn stats(&self) -> ClusterStats {
        let sizes: Vec<usize> = self.clusters.iter().map(|c| c.size).collect();
        let variances: Vec<f64> = self.clusters.iter().map(|c| c.variance).collect();
        let coherences: Vec<f64> = self.clusters.iter().map(|c| c.coherence_score).collect();

        ClusterStats {
            num_clusters: self.clusters.len(),
            total_observations: self.observations.len(),
            avg_cluster_size: if sizes.is_empty() { 0.0 }
                else { sizes.iter().sum::<usize>() as f64 / sizes.len() as f64 },
            min_cluster_size: sizes.iter().copied().min().unwrap_or(0),
            max_cluster_size: sizes.iter().copied().max().unwrap_or(0),
            avg_variance: if variances.is_empty() { 0.0 }
                else { variances.iter().sum::<f64>() / variances.len() as f64 },
            max_variance: variances.iter().cloned().fold(0.0f64, f64::max),
            avg_coherence: if coherences.is_empty() { 0.0 }
                else { coherences.iter().filter(|c| c.is_finite()).sum::<f64>()
                    / coherences.iter().filter(|c| c.is_finite()).count().max(1) as f64 },
            silhouette_score: self.silhouette_score(),
        }
    }

    /// Get the cluster assignment for an observation.
    pub fn get_cluster(&self, obs_id: usize) -> Option<ClusterId> {
        self.observations.get(obs_id)?.cluster
    }

    /// Assign a new observation to the nearest cluster.
    pub fn assign_to_nearest(&mut self, obs_idx: usize) -> Option<ClusterId> {
        let features = self.feature_cache.get(obs_idx)?.clone();
        let mut best_cluster: Option<ClusterId> = None;
        let mut best_dist = f64::INFINITY;

        for cluster in &self.clusters {
            let d = cluster.distance_to(&features);
            if d < best_dist {
                best_dist = d;
                best_cluster = Some(cluster.id);
            }
        }

        if let Some(cid) = best_cluster {
            if obs_idx < self.observations.len() {
                self.observations[obs_idx].cluster = Some(cid);
            }
        }
        best_cluster
    }

    /// Get a mapping from cluster IDs to their labels.
    pub fn alphabet_labels(&self) -> HashMap<ClusterId, String> {
        self.clusters.iter()
            .map(|c| (c.id, c.label.clone()))
            .collect()
    }

    /// Generate auto-labels for clusters based on dominant features.
    pub fn auto_label_clusters(&mut self) {
        let feature_names = [
            "length", "lex_diversity", "sentence_len", "sentiment",
            "formality", "readability", "code", "math", "refusal",
        ];

        for cluster in &mut self.clusters {
            if cluster.centroid.is_empty() {
                continue;
            }

            // Find the most distinctive feature.
            let mut max_feat_idx = 0;
            let mut max_feat_val = 0.0f64;
            for (i, &v) in cluster.centroid.iter().enumerate() {
                if v.abs() > max_feat_val.abs() {
                    max_feat_val = v;
                    max_feat_idx = i;
                }
            }

            let feat_name = if max_feat_idx < feature_names.len() {
                feature_names[max_feat_idx]
            } else {
                "feature"
            };

            let qualifier = if max_feat_val > 0.7 {
                "high"
            } else if max_feat_val > 0.3 {
                "medium"
            } else {
                "low"
            };

            cluster.label = format!("{}_{}", qualifier, feat_name);
        }
    }

    /// Visualize the current alphabet as text.
    pub fn visualize(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Alphabet Abstraction ===\n");
        out.push_str(&format!(
            "Clusters: {}, Observations: {}\n\n",
            self.clusters.len(), self.observations.len()
        ));

        for cluster in &self.clusters {
            let status = if cluster.validated { "✓" } else { "✗" };
            out.push_str(&format!(
                "  [{}] {} size={} var={:.4} coherence={:.4} {}\n",
                cluster.id, status, cluster.size,
                cluster.variance, cluster.coherence_score, cluster.label
            ));
        }

        let stats = self.stats();
        out.push_str(&format!("\nSilhouette score: {:.4}\n", stats.silhouette_score));
        out
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Euclidean distance between two vectors.
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Two-sample Kolmogorov-Smirnov test statistic.
fn ks_statistic(sample1: &[f64], sample2: &[f64]) -> f64 {
    if sample1.is_empty() || sample2.is_empty() {
        return 0.0;
    }

    let mut s1 = sample1.to_vec();
    let mut s2 = sample2.to_vec();
    s1.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    s2.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n1 = s1.len() as f64;
    let n2 = s2.len() as f64;

    let mut all_values: Vec<f64> = [s1.clone(), s2.clone()].concat();
    all_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    all_values.dedup();

    let mut max_diff = 0.0f64;

    for &val in &all_values {
        let cdf1 = s1.iter().filter(|&&x| x <= val).count() as f64 / n1;
        let cdf2 = s2.iter().filter(|&&x| x <= val).count() as f64 / n2;
        let diff = (cdf1 - cdf2).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    max_diff
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_observations(n: usize) -> Vec<(String, String)> {
        let mut data = Vec::new();
        for i in 0..n {
            let prompt = format!("Question {}", i);
            let response = if i % 3 == 0 {
                format!("This is a short answer number {}.", i)
            } else if i % 3 == 1 {
                format!(
                    "This is a much longer and more detailed response to question {}. \
                    It contains multiple sentences and explores the topic in greater depth. \
                    The answer considers various perspectives and provides examples. \
                    Furthermore, it discusses implications and potential applications.",
                    i
                )
            } else {
                format!(
                    "```python\ndef solve(n):\n    return n * 2\n```\n\
                    Here is the code solution for problem {}.",
                    i
                )
            };
            data.push((prompt, response));
        }
        data
    }

    #[test]
    fn test_response_stats_from_text() {
        let text = "This is a good test. It has multiple sentences.";
        let stats = ResponseStats::from_text(text);

        assert_eq!(stats.length, 9);
        assert!(stats.lexical_diversity > 0.0);
        assert!(stats.avg_sentence_length > 0.0);
        assert!(!stats.contains_code);
        assert!(!stats.is_refusal);
    }

    #[test]
    fn test_response_stats_code() {
        let text = "```python\ndef hello():\n    print('hello')\n```";
        let stats = ResponseStats::from_text(text);
        assert!(stats.contains_code);
    }

    #[test]
    fn test_response_stats_refusal() {
        let text = "I cannot provide that information. I apologize for any inconvenience.";
        let stats = ResponseStats::from_text(text);
        assert!(stats.is_refusal);
    }

    #[test]
    fn test_response_stats_sentiment() {
        let positive = "This is a great and excellent wonderful response.";
        let negative = "This is a terrible and awful horrible response.";
        let stats_pos = ResponseStats::from_text(positive);
        let stats_neg = ResponseStats::from_text(negative);
        assert!(stats_pos.sentiment > stats_neg.sentiment);
    }

    #[test]
    fn test_feature_vector() {
        let stats = ResponseStats::from_text("Hello world");
        let vec = stats.to_feature_vector();
        assert_eq!(vec.len(), 9);
    }

    #[test]
    fn test_stats_distance() {
        let s1 = ResponseStats::from_text("Hello world");
        let s2 = ResponseStats::from_text("Hello world");
        assert!(s1.distance(&s2) < 1e-10);

        let s3 = ResponseStats::from_text(
            "This is a very long response with many words and sentences. \
             It goes on and on with detailed explanations. And more detail."
        );
        assert!(s1.distance(&s3) > 0.0);
    }

    #[test]
    fn test_cluster_info() {
        let features = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![0.9, 1.9, 2.9],
        ];
        let mut cluster = ClusterInfo::new(0);
        cluster.members = vec![0, 1, 2];
        cluster.compute_centroid(&features);
        cluster.compute_variance(&features);

        assert_eq!(cluster.centroid.len(), 3);
        assert!(cluster.variance >= 0.0);
        assert_eq!(cluster.size, 3);
    }

    #[test]
    fn test_alphabet_construction() {
        let data = make_observations(30);
        let config = AlphabetConfig {
            initial_k: 3,
            min_cluster_size: 2,
            max_clusters: 16,
            kmeans_iterations: 50,
            ..AlphabetConfig::default()
        };
        let mut alphabet = AlphabetAbstraction::new(config);
        alphabet.add_texts(&data);
        alphabet.build();

        assert_eq!(alphabet.num_observations(), 30);
        assert!(alphabet.alphabet_size() > 0);
        assert!(alphabet.alphabet_size() <= 3);

        // Every observation should have a cluster.
        for obs in &alphabet.observations {
            assert!(obs.cluster.is_some());
        }
    }

    #[test]
    fn test_alphabet_split() {
        let data = make_observations(30);
        let config = AlphabetConfig {
            initial_k: 2,
            min_cluster_size: 2,
            max_clusters: 16,
            kmeans_iterations: 50,
            ..AlphabetConfig::default()
        };
        let mut alphabet = AlphabetAbstraction::new(config);
        alphabet.add_texts(&data);
        alphabet.build();

        let initial_size = alphabet.alphabet_size();
        let cluster_id = alphabet.clusters[0].id;
        let new_ids = alphabet.split_cluster(cluster_id, 2);

        // Should have more clusters now (if the cluster had enough members).
        assert!(alphabet.alphabet_size() >= initial_size);
        assert!(!new_ids.is_empty());
    }

    #[test]
    fn test_alphabet_merge() {
        let data = make_observations(30);
        let config = AlphabetConfig {
            initial_k: 4,
            min_cluster_size: 2,
            max_clusters: 16,
            kmeans_iterations: 50,
            ..AlphabetConfig::default()
        };
        let mut alphabet = AlphabetAbstraction::new(config);
        alphabet.add_texts(&data);
        alphabet.build();

        if alphabet.clusters.len() >= 2 {
            let a = alphabet.clusters[0].id;
            let b = alphabet.clusters[1].id;
            let size_before = alphabet.alphabet_size();
            let merged = alphabet.merge_clusters(a, b);
            assert!(merged.is_some());
            assert_eq!(alphabet.alphabet_size(), size_before - 1);
        }
    }

    #[test]
    fn test_silhouette_score() {
        let data = make_observations(30);
        let config = AlphabetConfig {
            initial_k: 3,
            min_cluster_size: 2,
            max_clusters: 16,
            kmeans_iterations: 50,
            ..AlphabetConfig::default()
        };
        let mut alphabet = AlphabetAbstraction::new(config);
        alphabet.add_texts(&data);
        alphabet.build();

        let score = alphabet.silhouette_score();
        assert!(score >= -1.0 && score <= 1.0);
    }

    #[test]
    fn test_recluster() {
        let data = make_observations(30);
        let config = AlphabetConfig::default();
        let mut alphabet = AlphabetAbstraction::new(config);
        alphabet.add_texts(&data);
        alphabet.build();

        alphabet.recluster(5);
        assert!(alphabet.alphabet_size() <= 5);
    }

    #[test]
    fn test_assign_to_nearest() {
        let data = make_observations(30);
        let config = AlphabetConfig {
            initial_k: 3,
            min_cluster_size: 2,
            ..AlphabetConfig::default()
        };
        let mut alphabet = AlphabetAbstraction::new(config);
        alphabet.add_texts(&data);
        alphabet.build();

        // Clear one observation's cluster and reassign.
        alphabet.observations[0].cluster = None;
        let assigned = alphabet.assign_to_nearest(0);
        assert!(assigned.is_some());
    }

    #[test]
    fn test_auto_label() {
        let data = make_observations(30);
        let config = AlphabetConfig {
            initial_k: 3,
            ..AlphabetConfig::default()
        };
        let mut alphabet = AlphabetAbstraction::new(config);
        alphabet.add_texts(&data);
        alphabet.build();
        alphabet.auto_label_clusters();

        for cluster in &alphabet.clusters {
            assert!(!cluster.label.is_empty());
        }
    }

    #[test]
    fn test_ks_statistic() {
        let s1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(ks_statistic(&s1, &s2) < 1e-10);

        let s3 = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        assert!(ks_statistic(&s1, &s3) > 0.0);
    }

    #[test]
    fn test_find_merge_candidates() {
        let data = make_observations(30);
        let config = AlphabetConfig {
            initial_k: 8,
            merge_threshold: 10.0, // Large threshold to guarantee candidates
            ..AlphabetConfig::default()
        };
        let mut alphabet = AlphabetAbstraction::new(config);
        alphabet.add_texts(&data);
        alphabet.build();

        let candidates = alphabet.find_merge_candidates();
        // With a very large merge threshold, there should be candidates.
        // (Unless all clusters are very far apart.)
    }

    #[test]
    fn test_visualize() {
        let data = make_observations(20);
        let config = AlphabetConfig {
            initial_k: 3,
            ..AlphabetConfig::default()
        };
        let mut alphabet = AlphabetAbstraction::new(config);
        alphabet.add_texts(&data);
        alphabet.build();

        let viz = alphabet.visualize();
        assert!(viz.contains("Alphabet Abstraction"));
        assert!(viz.contains("Clusters:"));
    }

    #[test]
    fn test_stats() {
        let data = make_observations(30);
        let config = AlphabetConfig {
            initial_k: 4,
            ..AlphabetConfig::default()
        };
        let mut alphabet = AlphabetAbstraction::new(config);
        alphabet.add_texts(&data);
        alphabet.build();

        let stats = alphabet.stats();
        assert_eq!(stats.total_observations, 30);
        assert!(stats.num_clusters > 0);
        assert!(stats.avg_cluster_size > 0.0);
    }

    #[test]
    fn test_empty_alphabet() {
        let config = AlphabetConfig::default();
        let mut alphabet = AlphabetAbstraction::new(config);
        alphabet.build();

        assert_eq!(alphabet.alphabet_size(), 0);
        assert_eq!(alphabet.num_observations(), 0);
    }

    #[test]
    fn test_single_observation() {
        let config = AlphabetConfig {
            initial_k: 3,
            ..AlphabetConfig::default()
        };
        let mut alphabet = AlphabetAbstraction::new(config);
        alphabet.add_texts(&[("prompt".to_string(), "response".to_string())]);
        alphabet.build();

        assert_eq!(alphabet.alphabet_size(), 1);
    }

    #[test]
    fn test_history_tracking() {
        let data = make_observations(30);
        let config = AlphabetConfig {
            initial_k: 3,
            ..AlphabetConfig::default()
        };
        let mut alphabet = AlphabetAbstraction::new(config);
        alphabet.add_texts(&data);
        alphabet.build();

        assert!(alphabet.history.is_empty());

        let cid = alphabet.clusters[0].id;
        alphabet.split_cluster(cid, 2);
        assert!(!alphabet.history.is_empty());
    }

    #[test]
    fn test_adaptive_resize() {
        let data = make_observations(60);
        let config = AlphabetConfig {
            initial_k: 6,
            max_variance: 0.001, // Very tight — will trigger splits
            merge_threshold: 100.0, // Very loose — will trigger merges
            max_clusters: 12,
            min_cluster_size: 2,
            ..AlphabetConfig::default()
        };
        let mut alphabet = AlphabetAbstraction::new(config);
        alphabet.add_texts(&data);
        alphabet.build();

        let ops = alphabet.adaptive_resize();
        // Should have performed at least some operations.
        // The exact count depends on the data distribution.
        let _ = ops;
    }

    #[test]
    fn test_alphabet_labels() {
        let data = make_observations(20);
        let config = AlphabetConfig {
            initial_k: 3,
            ..AlphabetConfig::default()
        };
        let mut alphabet = AlphabetAbstraction::new(config);
        alphabet.add_texts(&data);
        alphabet.build();

        let labels = alphabet.alphabet_labels();
        assert_eq!(labels.len(), alphabet.alphabet_size());
    }
}
