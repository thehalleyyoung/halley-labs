//! Partition quality metrics for evaluating graph and hypergraph clusterings.
//!
//! Provides normalized cut, ratio cut, conductance, modularity, hypergraph
//! crossing fraction, adjusted Rand index, and normalized mutual information.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use spectral_types::graph::{Graph, Hypergraph};
use spectral_types::partition::Partition;

use crate::error::{Result, SpectralCoreError};

// ---------------------------------------------------------------------------
// PartitionQuality
// ---------------------------------------------------------------------------

/// Comprehensive partition quality metrics.
///
/// Graph-based metrics are set to `NaN` when only two partitions are compared
/// (no graph available). Likewise, ARI and NMI are set to `NaN` when only a
/// single partition is evaluated against a graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionQuality {
    pub normalized_cut: f64,
    pub ratio_cut: f64,
    pub conductance: f64,
    pub modularity: f64,
    pub hypergraph_crossing_fraction: f64,
    pub adjusted_rand_index: f64,
    pub normalized_mutual_information: f64,
    pub balance_ratio: f64,
}

// ---------------------------------------------------------------------------
// High-level entry points
// ---------------------------------------------------------------------------

/// Compute all graph-based quality metrics for a partition.
///
/// ARI and NMI are set to `NaN` because no ground-truth partition is provided.
pub fn compute_partition_quality(graph: &Graph, partition: &Partition) -> Result<PartitionQuality> {
    if graph.num_vertices == 0 {
        return Err(SpectralCoreError::empty_input("graph has no vertices"));
    }
    if partition.assignment.len() != graph.num_vertices {
        return Err(SpectralCoreError::clustering(format!(
            "partition size {} does not match graph vertices {}",
            partition.assignment.len(),
            graph.num_vertices
        )));
    }

    Ok(PartitionQuality {
        normalized_cut: normalized_cut(graph, partition),
        ratio_cut: ratio_cut(graph, partition),
        conductance: conductance(graph, partition),
        modularity: modularity(graph, partition),
        hypergraph_crossing_fraction: f64::NAN,
        adjusted_rand_index: f64::NAN,
        normalized_mutual_information: f64::NAN,
        balance_ratio: partition.balance_ratio(),
    })
}

/// Compare two partitions using ARI and NMI.
///
/// Graph-based metrics are set to `NaN` because no graph is provided.
pub fn compare_partitions(
    partition_a: &Partition,
    partition_b: &Partition,
) -> Result<PartitionQuality> {
    if partition_a.assignment.is_empty() || partition_b.assignment.is_empty() {
        return Err(SpectralCoreError::empty_input("partition is empty"));
    }
    if partition_a.assignment.len() != partition_b.assignment.len() {
        return Err(SpectralCoreError::clustering(format!(
            "partition sizes differ: {} vs {}",
            partition_a.assignment.len(),
            partition_b.assignment.len()
        )));
    }

    let ari = adjusted_rand_index(&partition_a.assignment, &partition_b.assignment);
    let nmi = normalized_mutual_information(&partition_a.assignment, &partition_b.assignment);

    Ok(PartitionQuality {
        normalized_cut: f64::NAN,
        ratio_cut: f64::NAN,
        conductance: f64::NAN,
        modularity: f64::NAN,
        hypergraph_crossing_fraction: f64::NAN,
        adjusted_rand_index: ari,
        normalized_mutual_information: nmi,
        balance_ratio: f64::NAN,
    })
}

// ---------------------------------------------------------------------------
// Block-level helpers
// ---------------------------------------------------------------------------

/// Compute the cut weight for every block in the partition.
///
/// `result[k]` is the total weight of edges with exactly one endpoint in block `k`.
pub fn compute_block_cuts(graph: &Graph, partition: &Partition) -> Vec<f64> {
    let k = partition.num_blocks;
    let mut cuts = vec![0.0_f64; k];

    for u in 0..graph.num_vertices {
        let cu = partition.assignment[u];
        for &(v, w) in &graph.adjacency[u] {
            if partition.assignment[v] != cu {
                cuts[cu] += w;
            }
        }
    }

    // For crossing edge (u,v) with u in S_a and v in S_b: vertex u
    // contributes w to cuts[a] and vertex v contributes w to cuts[b].
    // Each block sees each crossing edge exactly once, so no halving.
    cuts
}

/// Compute the volume (sum of weighted degrees) for every block.
pub fn compute_block_volumes(graph: &Graph, partition: &Partition) -> Vec<f64> {
    let k = partition.num_blocks;
    let mut volumes = vec![0.0_f64; k];

    for v in 0..graph.num_vertices {
        volumes[partition.assignment[v]] += graph.weighted_degree(v);
    }

    volumes
}

// ---------------------------------------------------------------------------
// Normalized cut
// ---------------------------------------------------------------------------

/// NCut = Σ_k cut(S_k, V\\S_k) / vol(S_k).
///
/// Blocks with zero volume are skipped.
pub fn normalized_cut(graph: &Graph, partition: &Partition) -> f64 {
    let cuts = compute_block_cuts(graph, partition);
    let vols = compute_block_volumes(graph, partition);

    let mut ncut = 0.0_f64;
    for (cut, vol) in cuts.iter().zip(vols.iter()) {
        if *vol > 0.0 {
            ncut += cut / vol;
        }
    }
    ncut
}

// ---------------------------------------------------------------------------
// Ratio cut
// ---------------------------------------------------------------------------

/// RCut = Σ_k cut(S_k, V\\S_k) / |S_k|.
///
/// Blocks with zero size are skipped.
pub fn ratio_cut(graph: &Graph, partition: &Partition) -> f64 {
    let cuts = compute_block_cuts(graph, partition);
    let sizes = partition.block_sizes();

    let mut rcut = 0.0_f64;
    for (cut, sz) in cuts.iter().zip(sizes.iter()) {
        if *sz > 0 {
            rcut += cut / (*sz as f64);
        }
    }
    rcut
}

// ---------------------------------------------------------------------------
// Conductance
// ---------------------------------------------------------------------------

/// Minimum conductance over all non-trivial blocks.
///
/// φ(S_k) = cut(S_k, V\\S_k) / min(vol(S_k), vol(V\\S_k)).
///
/// Returns `f64::INFINITY` if every block has zero volume (degenerate case).
pub fn conductance(graph: &Graph, partition: &Partition) -> f64 {
    let cuts = compute_block_cuts(graph, partition);
    let vols = compute_block_volumes(graph, partition);
    let total_vol: f64 = vols.iter().sum();

    let mut min_cond = f64::INFINITY;

    for (cut, vol) in cuts.iter().zip(vols.iter()) {
        let complement_vol = total_vol - vol;
        let denom = vol.min(complement_vol);
        if denom > 0.0 {
            let c = cut / denom;
            if c < min_cond {
                min_cond = c;
            }
        }
    }
    min_cond
}

// ---------------------------------------------------------------------------
// Modularity
// ---------------------------------------------------------------------------

/// Newman-Girvan modularity.
///
/// Q = (1 / 2m) Σ_{ij} [A_{ij} − d_i d_j / (2m)] δ(c_i, c_j)
///
/// Returns 0.0 when total weight m = 0.
pub fn modularity(graph: &Graph, partition: &Partition) -> f64 {
    let m = graph.total_weight();
    if m == 0.0 {
        return 0.0;
    }
    let two_m = 2.0 * m;

    // Accumulate internal edge weight and degree-product sum per block.
    let k = partition.num_blocks;
    let mut internal_weight = vec![0.0_f64; k];
    let mut block_degree_sum = vec![0.0_f64; k];

    for v in 0..graph.num_vertices {
        let cv = partition.assignment[v];
        block_degree_sum[cv] += graph.weighted_degree(v);

        for &(u, w) in &graph.adjacency[v] {
            if partition.assignment[u] == cv {
                internal_weight[cv] += w; // counted once per directed edge
            }
        }
    }

    let mut q = 0.0_f64;
    for b in 0..k {
        // internal_weight[b] double-counts each undirected edge (u,v) with
        // u < v as (u->v) and (v->u), giving 2 * actual weight.
        // The formula wants Σ A_{ij} over all (i,j) pairs in the block, which
        // with the symmetric adjacency list is exactly internal_weight[b].
        let l_b = internal_weight[b]; // = 2 * actual internal weight
        let d_b = block_degree_sum[b];
        q += l_b / two_m - (d_b / two_m).powi(2);
    }
    q
}

// ---------------------------------------------------------------------------
// Hypergraph crossing fraction
// ---------------------------------------------------------------------------

/// Fraction of hyperedges whose vertices span more than one partition block.
///
/// Returns 0.0 when the hypergraph has no edges.
pub fn hypergraph_crossing_fraction(hypergraph: &Hypergraph, partition: &Partition) -> f64 {
    let num_edges = hypergraph.edges.len();
    if num_edges == 0 {
        return 0.0;
    }
    let crossing = hypergraph.crossing_edges(&partition.assignment);
    crossing as f64 / num_edges as f64
}

// ---------------------------------------------------------------------------
// Contingency table helper
// ---------------------------------------------------------------------------

/// Build a contingency table for two label vectors.
///
/// Returns `(table, n_a, n_b)` where `table[(i,j)]` is the count of items in
/// cluster `i` of `a` and cluster `j` of `b`, `n_a` is the number of distinct
/// labels in `a`, and `n_b` in `b`.
fn contingency_table(a: &[usize], b: &[usize]) -> (Vec<Vec<usize>>, usize, usize) {
    assert_eq!(a.len(), b.len());

    // Re-map labels to contiguous 0..n_a, 0..n_b.
    let remap = |labels: &[usize]| -> (Vec<usize>, usize) {
        let mut map: HashMap<usize, usize> = HashMap::new();
        let mut next = 0usize;
        let mapped: Vec<usize> = labels
            .iter()
            .map(|&l| {
                *map.entry(l).or_insert_with(|| {
                    let id = next;
                    next += 1;
                    id
                })
            })
            .collect();
        (mapped, next)
    };

    let (a_mapped, n_a) = remap(a);
    let (b_mapped, n_b) = remap(b);

    let mut table = vec![vec![0usize; n_b]; n_a];
    for (&ai, &bi) in a_mapped.iter().zip(b_mapped.iter()) {
        table[ai][bi] += 1;
    }

    (table, n_a, n_b)
}

// ---------------------------------------------------------------------------
// Adjusted Rand Index
// ---------------------------------------------------------------------------

/// Adjusted Rand Index between two clusterings.
///
/// ARI = (RI − E[RI]) / (max(RI) − E[RI]).
///
/// Handles degenerate cases (all elements in one cluster) by returning 1.0
/// when the two partitions are identical singletons, and 0.0 otherwise.
pub fn adjusted_rand_index(a: &[usize], b: &[usize]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    if n == 0 {
        return 1.0;
    }

    let (table, n_a, n_b) = contingency_table(a, b);

    // Row sums (a_i) and column sums (b_j).
    let row_sums: Vec<usize> = (0..n_a).map(|i| table[i].iter().sum()).collect();
    let col_sums: Vec<usize> = (0..n_b)
        .map(|j| (0..n_a).map(|i| table[i][j]).sum())
        .collect();

    let choose2 = |x: usize| -> i64 {
        if x < 2 {
            0
        } else {
            (x as i64) * (x as i64 - 1) / 2
        }
    };

    let sum_nij: i64 = table
        .iter()
        .flat_map(|row| row.iter())
        .map(|&v| choose2(v))
        .sum();
    let sum_ai: i64 = row_sums.iter().map(|&v| choose2(v)).sum();
    let sum_bj: i64 = col_sums.iter().map(|&v| choose2(v)).sum();
    let c_n = choose2(n);

    if c_n == 0 {
        return 1.0;
    }

    let expected = (sum_ai as f64) * (sum_bj as f64) / (c_n as f64);
    let max_index = 0.5 * (sum_ai as f64 + sum_bj as f64);
    let denom = max_index - expected;

    if denom.abs() < 1e-15 {
        // Both partitions assign everything to one cluster.
        if (sum_nij as f64 - expected).abs() < 1e-15 {
            return 1.0;
        }
        return 0.0;
    }

    (sum_nij as f64 - expected) / denom
}

// ---------------------------------------------------------------------------
// Normalized Mutual Information
// ---------------------------------------------------------------------------

/// Normalized Mutual Information between two clusterings.
///
/// NMI = 2 * MI(a, b) / (H(a) + H(b)).
///
/// Returns 1.0 when both partitions assign everything to a single cluster
/// (zero entropy edge case).
pub fn normalized_mutual_information(a: &[usize], b: &[usize]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    if n == 0 {
        return 1.0;
    }
    let n_f = n as f64;

    let (table, n_a, n_b) = contingency_table(a, b);

    let row_sums: Vec<usize> = (0..n_a).map(|i| table[i].iter().sum()).collect();
    let col_sums: Vec<usize> = (0..n_b)
        .map(|j| (0..n_a).map(|i| table[i][j]).sum())
        .collect();

    // Entropy H(X) = -Σ p_i log(p_i).
    let entropy = |sums: &[usize]| -> f64 {
        let mut h = 0.0_f64;
        for &s in sums {
            if s > 0 {
                let p = s as f64 / n_f;
                h -= p * p.ln();
            }
        }
        h
    };

    let h_a = entropy(&row_sums);
    let h_b = entropy(&col_sums);

    if h_a + h_b < 1e-15 {
        // Both partitions are trivial (single cluster) → identical.
        return 1.0;
    }

    // MI(A, B) = Σ_{ij} (n_ij / n) log(n * n_ij / (a_i * b_j)).
    let mut mi = 0.0_f64;
    for i in 0..n_a {
        for j in 0..n_b {
            let nij = table[i][j];
            if nij > 0 {
                let p_ij = nij as f64 / n_f;
                let p_i = row_sums[i] as f64 / n_f;
                let p_j = col_sums[j] as f64 / n_f;
                mi += p_ij * (p_ij / (p_i * p_j)).ln();
            }
        }
    }

    2.0 * mi / (h_a + h_b)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use spectral_types::graph::{Graph, Hypergraph};
    use spectral_types::partition::Partition;

    /// Build a small bipartite-like graph:
    /// Block 0: {0,1,2} fully connected (weight 1 each)
    /// Block 1: {3,4,5} fully connected (weight 1 each)
    /// One bridge edge: 2-3 (weight 1)
    fn bipartite_graph() -> (Graph, Partition) {
        let mut g = Graph::new(6);
        // Block 0 internal edges
        g.add_edge(0, 1, 1.0);
        g.add_edge(0, 2, 1.0);
        g.add_edge(1, 2, 1.0);
        // Block 1 internal edges
        g.add_edge(3, 4, 1.0);
        g.add_edge(3, 5, 1.0);
        g.add_edge(4, 5, 1.0);
        // Bridge
        g.add_edge(2, 3, 1.0);

        let p = Partition::new(vec![0, 0, 0, 1, 1, 1]);
        (g, p)
    }

    #[test]
    fn test_normalized_cut_bipartite() {
        let (g, p) = bipartite_graph();
        let nc = normalized_cut(&g, &p);
        // cut(S0) = 1, vol(S0) = 2+2+3 = 7 (vertex 2 has degree 3)
        // cut(S1) = 1, vol(S1) = 2+2+3 = 7
        // NCut = 1/7 + 1/7 = 2/7 ≈ 0.2857
        assert!((nc - 2.0 / 7.0).abs() < 1e-9, "ncut = {nc}");
    }

    #[test]
    fn test_ratio_cut() {
        let (g, p) = bipartite_graph();
        let rc = ratio_cut(&g, &p);
        // cut(S0)=1, |S0|=3; cut(S1)=1, |S1|=3
        // RCut = 1/3 + 1/3 = 2/3
        assert!((rc - 2.0 / 3.0).abs() < 1e-9, "rcut = {rc}");
    }

    #[test]
    fn test_conductance_well_separated() {
        let (g, p) = bipartite_graph();
        let c = conductance(&g, &p);
        // cut(S0)=1, vol(S0)=7, vol(S1)=7 → φ(S0)=1/7
        // cut(S1)=1 → φ(S1)=1/7
        // min = 1/7
        assert!((c - 1.0 / 7.0).abs() < 1e-9, "conductance = {c}");
    }

    #[test]
    fn test_modularity_clear_communities() {
        let (g, p) = bipartite_graph();
        let q = modularity(&g, &p);
        // Q should be positive for well-separated communities
        assert!(q > 0.0, "modularity should be positive, got {q}");
        // Known value: m=7, two_m=14
        // Block 0: internal edges 3 (counted 6 in adj), d_sum = 7
        //   contribution = 6/14 - (7/14)^2 = 3/7 - 1/4 = 5/28
        // Block 1: same → 5/28
        // Q = 10/28 = 5/14 ≈ 0.3571
        assert!((q - 5.0 / 14.0).abs() < 1e-9, "modularity = {q}");
    }

    #[test]
    fn test_modularity_random_assignment() {
        // Fully connected graph (K4) with random assignment
        let mut g = Graph::new(4);
        for i in 0..4 {
            for j in (i + 1)..4 {
                g.add_edge(i, j, 1.0);
            }
        }
        // Assign each vertex to its own block — modularity should be ≤ 0
        let p = Partition::new(vec![0, 1, 2, 3]);
        let q = modularity(&g, &p);
        // For singleton clusters on a complete graph:
        // no internal edges, so each block contributes -d_i^2/(2m)^2
        assert!(q <= 0.0, "expected non-positive modularity, got {q}");
    }

    #[test]
    fn test_ari_identical_partitions() {
        let a = vec![0, 0, 1, 1, 2, 2];
        let b = vec![0, 0, 1, 1, 2, 2];
        let ari = adjusted_rand_index(&a, &b);
        assert!((ari - 1.0).abs() < 1e-9, "ARI of identical = {ari}");
    }

    #[test]
    fn test_ari_independent_partitions() {
        // One clustering: [0,0,0,0,1,1,1,1]
        // Other clustering: [0,1,0,1,0,1,0,1]  (orthogonal)
        let a = vec![0, 0, 0, 0, 1, 1, 1, 1];
        let b = vec![0, 1, 0, 1, 0, 1, 0, 1];
        let ari = adjusted_rand_index(&a, &b);
        // ARI should be close to 0 for independent partitions
        assert!(ari.abs() < 0.2, "ARI of independent = {ari}");
    }

    #[test]
    fn test_nmi_identical_partitions() {
        let a = vec![0, 0, 1, 1, 2, 2];
        let b = vec![0, 0, 1, 1, 2, 2];
        let nmi = normalized_mutual_information(&a, &b);
        assert!((nmi - 1.0).abs() < 1e-9, "NMI of identical = {nmi}");
    }

    #[test]
    fn test_nmi_single_vs_multi() {
        // If one partition puts everything in one block, NMI should be 0
        // (H of the single-block partition is 0, so 2*MI / (0 + H_b) requires
        // special handling — our impl returns 1.0 when both are trivial,
        // but when only one is trivial MI = 0 so NMI = 0).
        let a = vec![0, 0, 0, 0]; // single block
        let b = vec![0, 0, 1, 1]; // two blocks
        let nmi = normalized_mutual_information(&a, &b);
        assert!(
            nmi.abs() < 1e-9,
            "NMI single vs multi should be 0, got {nmi}"
        );
    }

    #[test]
    fn test_hypergraph_crossing_fraction() {
        let mut hg = Hypergraph::new(6);
        // Edge fully in block 0
        hg.add_edge(vec![0, 1, 2], 1.0);
        // Edge fully in block 1
        hg.add_edge(vec![3, 4, 5], 1.0);
        // Crossing edge
        hg.add_edge(vec![2, 3], 1.0);

        let p = Partition::new(vec![0, 0, 0, 1, 1, 1]);
        let frac = hypergraph_crossing_fraction(&hg, &p);
        // 1 out of 3 edges crosses
        assert!((frac - 1.0 / 3.0).abs() < 1e-9, "crossing frac = {frac}");
    }

    #[test]
    fn test_balance_ratio() {
        let (g, p) = bipartite_graph();
        let quality = compute_partition_quality(&g, &p).unwrap();
        // Perfectly balanced 3-3 split → balance_ratio = 3/3 = 1.0
        assert!(
            (quality.balance_ratio - 1.0).abs() < 1e-9,
            "balance = {}",
            quality.balance_ratio
        );
    }

    #[test]
    fn test_block_cuts_and_volumes() {
        let (g, p) = bipartite_graph();
        let cuts = compute_block_cuts(&g, &p);
        let vols = compute_block_volumes(&g, &p);

        assert_eq!(cuts.len(), 2);
        assert_eq!(vols.len(), 2);

        // One bridge edge of weight 1 between blocks
        assert!((cuts[0] - 1.0).abs() < 1e-9, "cut[0] = {}", cuts[0]);
        assert!((cuts[1] - 1.0).abs() < 1e-9, "cut[1] = {}", cuts[1]);

        // vol(S0) = deg(0)+deg(1)+deg(2) = 2+2+3 = 7
        assert!((vols[0] - 7.0).abs() < 1e-9, "vol[0] = {}", vols[0]);
        assert!((vols[1] - 7.0).abs() < 1e-9, "vol[1] = {}", vols[1]);
    }
}
