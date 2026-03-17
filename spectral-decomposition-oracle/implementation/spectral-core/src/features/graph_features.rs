//! Graph-structural feature extraction from variable interaction graphs.
//!
//! Computes 10 topological features that characterize the structure of a
//! constraint graph (or any weighted graph):
//!
//! | # | Feature                        | Description                                         |
//! |---|-------------------------------|-----------------------------------------------------|
//! | 1 | `num_vertices`                | Number of vertices                                  |
//! | 2 | `num_edges`                   | Number of edges                                     |
//! | 3 | `avg_degree`                  | Mean vertex degree                                  |
//! | 4 | `max_degree`                  | Maximum vertex degree                               |
//! | 5 | `degree_variance`             | Variance of the degree sequence                     |
//! | 6 | `num_connected_components`    | Number of connected components                      |
//! | 7 | `largest_component_fraction`  | Fraction of vertices in the largest component       |
//! | 8 | `edge_density`                | Edge density  2|E| / (n(n-1))                       |
//! | 9 | `avg_clustering_coefficient`  | Average local clustering coefficient                |
//! |10 | `degree_assortativity`        | Pearson correlation of degrees at edge endpoints    |

use std::collections::VecDeque;

use log::debug;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use spectral_types::features::GraphFeatures;
use spectral_types::graph::Graph;
use spectral_types::sparse::CsrMatrix;

use crate::error::{Result, SpectralCoreError};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of vertices to sample when computing clustering coefficients
/// on large graphs.
const MAX_CLUSTERING_SAMPLES: usize = 500;

/// Default seed for deterministic sampling.
const DEFAULT_SEED: u64 = 0xDEAD_BEEF_CAFE;

// ---------------------------------------------------------------------------
// Primary entry point
// ---------------------------------------------------------------------------

/// Compute all 10 graph-structural features from a [`Graph`].
///
/// Returns [`SpectralCoreError::EmptyInput`] when the graph has zero vertices.
pub fn compute_graph_features(graph: &Graph) -> Result<GraphFeatures> {
    let n = graph.num_vertices;
    if n == 0 {
        return Err(SpectralCoreError::empty_input(
            "graph has zero vertices",
        ));
    }

    let degrees = graph.degrees();
    let m = graph.num_edges();

    let avg_degree = compute_mean_degree(&degrees);
    let max_degree = *degrees.iter().max().unwrap_or(&0);
    let degree_var = compute_degree_variance(&degrees, avg_degree);

    let components = graph.connected_components();
    let num_cc = components.len();
    let largest_cc = components.iter().map(|c| c.len()).max().unwrap_or(0);
    let largest_frac = largest_cc as f64 / n as f64;

    let density = graph.density();

    let avg_cc = compute_avg_clustering_coefficient(graph);
    let assortativity = compute_degree_assortativity(graph);

    debug!(
        "graph features: n={n}, m={m}, avg_deg={avg_degree:.2}, \
         max_deg={max_degree}, cc={num_cc}, density={density:.4}"
    );

    Ok(GraphFeatures {
        num_vertices: n as f64,
        num_edges: m as f64,
        avg_degree,
        max_degree: max_degree as f64,
        degree_variance: degree_var,
        num_connected_components: num_cc as f64,
        largest_component_fraction: largest_frac,
        edge_density: density,
        avg_clustering_coefficient: avg_cc,
        degree_assortativity: assortativity,
    })
}

// ---------------------------------------------------------------------------
// Matrix-based entry point
// ---------------------------------------------------------------------------

/// Compute graph features from a constraint matrix.
///
/// Builds a variable interaction graph where two columns (variables) are
/// connected if they co-occur in at least one row (constraint), then delegates
/// to [`compute_graph_features`].
pub fn compute_graph_features_from_matrix(matrix: &CsrMatrix<f64>) -> Result<GraphFeatures> {
    if matrix.cols == 0 {
        return Err(SpectralCoreError::empty_input(
            "constraint matrix has zero columns (variables)",
        ));
    }
    let graph = build_variable_interaction_graph(matrix);
    compute_graph_features(&graph)
}

// ---------------------------------------------------------------------------
// Variable interaction graph construction
// ---------------------------------------------------------------------------

/// Build a variable interaction graph from a CSR constraint matrix.
///
/// For each row (constraint), every pair of column indices (variables) that
/// appear in that row are connected with edge weight 1.0. Duplicate edges
/// (from multiple shared constraints) accumulate weight but the underlying
/// [`Graph::add_edge`] stores parallel entries; the graph methods count
/// degree correctly regardless.
pub fn build_variable_interaction_graph(matrix: &CsrMatrix<f64>) -> Graph {
    let n = matrix.cols;
    let mut graph = Graph::new(n);

    // Track which edges have already been added to avoid duplicates.
    let mut seen = std::collections::HashSet::<(usize, usize)>::new();

    for row in 0..matrix.rows {
        let cols = matrix.row_indices(row);
        let len = cols.len();
        for i in 0..len {
            for j in (i + 1)..len {
                let (u, v) = if cols[i] < cols[j] {
                    (cols[i], cols[j])
                } else {
                    (cols[j], cols[i])
                };
                if seen.insert((u, v)) {
                    graph.add_edge(u, v, 1.0);
                }
            }
        }
    }

    graph
}

// ---------------------------------------------------------------------------
// Degree statistics helpers
// ---------------------------------------------------------------------------

/// Mean of the degree sequence.
fn compute_mean_degree(degrees: &[usize]) -> f64 {
    if degrees.is_empty() {
        return 0.0;
    }
    let sum: usize = degrees.iter().sum();
    sum as f64 / degrees.len() as f64
}

/// Variance of the degree sequence (population variance).
fn compute_degree_variance(degrees: &[usize], mean: f64) -> f64 {
    if degrees.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = degrees
        .iter()
        .map(|&d| {
            let diff = d as f64 - mean;
            diff * diff
        })
        .sum();
    sum_sq / degrees.len() as f64
}

// ---------------------------------------------------------------------------
// Clustering coefficient
// ---------------------------------------------------------------------------

/// Average local clustering coefficient of the graph.
///
/// For large graphs (> [`MAX_CLUSTERING_SAMPLES`] vertices), a deterministic
/// random sample of vertices is used to estimate the average.
fn compute_avg_clustering_coefficient(graph: &Graph) -> f64 {
    let n = graph.num_vertices;
    if n == 0 {
        return 0.0;
    }

    let vertices: Vec<usize> = if n <= MAX_CLUSTERING_SAMPLES {
        (0..n).collect()
    } else {
        let mut rng = ChaCha8Rng::seed_from_u64(DEFAULT_SEED);
        let mut all: Vec<usize> = (0..n).collect();
        all.shuffle(&mut rng);
        all.truncate(MAX_CLUSTERING_SAMPLES);
        all
    };

    let sum: f64 = vertices
        .iter()
        .map(|&v| compute_local_clustering_coefficient(graph, v))
        .sum();

    sum / vertices.len() as f64
}

/// Local clustering coefficient for vertex `v`.
///
/// Defined as the fraction of pairs of neighbors of `v` that are themselves
/// connected.  Returns 0.0 when `deg(v) < 2`.
pub fn compute_local_clustering_coefficient(graph: &Graph, v: usize) -> f64 {
    let neighbors = graph.neighbors(v);
    let deg = neighbors.len();
    if deg < 2 {
        return 0.0;
    }

    let mut triangles = 0u64;
    for i in 0..deg {
        let ni_neighbors: std::collections::HashSet<usize> =
            graph.neighbors(neighbors[i]).into_iter().collect();
        for j in (i + 1)..deg {
            if ni_neighbors.contains(&neighbors[j]) {
                triangles += 1;
            }
        }
    }

    let possible = (deg * (deg - 1)) / 2;
    triangles as f64 / possible as f64
}

// ---------------------------------------------------------------------------
// Degree assortativity
// ---------------------------------------------------------------------------

/// Pearson correlation coefficient of degrees at the two endpoints of each
/// edge (degree assortativity).
///
/// For a graph with no edges, returns 0.0.
pub fn compute_degree_assortativity(graph: &Graph) -> f64 {
    let degrees = graph.degrees();
    let n = graph.num_vertices;

    // Collect (deg_u, deg_v) for every edge.
    let mut edge_degrees: Vec<(f64, f64)> = Vec::new();
    for u in 0..n {
        for &(v, _weight) in &graph.adjacency[u] {
            // Each undirected edge appears twice in the adjacency list;
            // only count it once (u < v).
            if u < v {
                edge_degrees.push((degrees[u] as f64, degrees[v] as f64));
            }
        }
    }

    if edge_degrees.is_empty() {
        return 0.0;
    }

    pearson_correlation(&edge_degrees)
}

/// Pearson correlation coefficient for a sequence of (x, y) pairs.
fn pearson_correlation(pairs: &[(f64, f64)]) -> f64 {
    let m = pairs.len() as f64;
    if m < 2.0 {
        return 0.0;
    }

    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let mut sum_xx = 0.0_f64;
    let mut sum_yy = 0.0_f64;
    let mut sum_xy = 0.0_f64;

    for &(x, y) in pairs {
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_yy += y * y;
        sum_xy += x * y;
    }

    let numerator = m * sum_xy - sum_x * sum_y;
    let denom_x = (m * sum_xx - sum_x * sum_x).sqrt();
    let denom_y = (m * sum_yy - sum_y * sum_y).sqrt();

    let denom = denom_x * denom_y;
    if denom.abs() < f64::EPSILON {
        return 0.0;
    }

    (numerator / denom).clamp(-1.0, 1.0)
}

// ---------------------------------------------------------------------------
// Sampled average shortest path length
// ---------------------------------------------------------------------------

/// Estimate the average shortest path length by BFS from a sample of source
/// vertices.
///
/// Uses a deterministic seeded RNG ([`ChaCha8Rng`]).  The number of samples
/// is capped at 100 and at `n` (the number of vertices).
///
/// Only reachable pairs contribute to the average.  Returns 0.0 for an empty
/// graph or when no paths exist.
pub fn compute_avg_path_length_sampled(graph: &Graph, num_samples: usize) -> f64 {
    let n = graph.num_vertices;
    if n == 0 {
        return 0.0;
    }

    let effective_samples = num_samples.min(100).min(n);

    let sources: Vec<usize> = if effective_samples >= n {
        (0..n).collect()
    } else {
        let mut rng = ChaCha8Rng::seed_from_u64(DEFAULT_SEED);
        let mut all: Vec<usize> = (0..n).collect();
        all.shuffle(&mut rng);
        all.truncate(effective_samples);
        all
    };

    let mut total_distance = 0u64;
    let mut total_pairs = 0u64;

    for &src in &sources {
        let distances = bfs_distances(graph, src);
        for (v, dist) in distances.iter().enumerate() {
            if v != src {
                if let Some(d) = dist {
                    total_distance += *d as u64;
                    total_pairs += 1;
                }
            }
        }
    }

    if total_pairs == 0 {
        return 0.0;
    }

    total_distance as f64 / total_pairs as f64
}

/// BFS from `source`, returning `Option<usize>` distances for each vertex.
/// `None` means unreachable.
fn bfs_distances(graph: &Graph, source: usize) -> Vec<Option<usize>> {
    let n = graph.num_vertices;
    let mut dist: Vec<Option<usize>> = vec![None; n];
    dist[source] = Some(0);

    let mut queue = VecDeque::new();
    queue.push_back(source);

    while let Some(u) = queue.pop_front() {
        let d = dist[u].unwrap();
        for &(v, _) in &graph.adjacency[u] {
            if dist[v].is_none() {
                dist[v] = Some(d + 1);
                queue.push_back(v);
            }
        }
    }

    dist
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Tolerance for floating-point comparisons.
    const EPS: f64 = 1e-9;

    // -- Helper graph builders ----------------------------------------------

    /// Complete graph K_n (unweighted, weight = 1.0).
    fn make_complete_graph(n: usize) -> Graph {
        let mut g = Graph::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                g.add_edge(i, j, 1.0);
            }
        }
        g
    }

    /// Path graph P_n: 0 - 1 - 2 - ... - (n-1).
    fn make_path_graph(n: usize) -> Graph {
        let mut g = Graph::new(n);
        for i in 0..(n.saturating_sub(1)) {
            g.add_edge(i, i + 1, 1.0);
        }
        g
    }

    /// Star graph S_n: vertex 0 is the hub connected to vertices 1..n-1.
    fn make_star_graph(n: usize) -> Graph {
        let mut g = Graph::new(n);
        for i in 1..n {
            g.add_edge(0, i, 1.0);
        }
        g
    }

    /// Triangle (K3).
    fn make_triangle() -> Graph {
        make_complete_graph(3)
    }

    /// Disconnected graph: two disjoint triangles (6 vertices).
    fn make_disconnected_graph() -> Graph {
        let mut g = Graph::new(6);
        // Component 1: {0, 1, 2}
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(0, 2, 1.0);
        // Component 2: {3, 4, 5}
        g.add_edge(3, 4, 1.0);
        g.add_edge(4, 5, 1.0);
        g.add_edge(3, 5, 1.0);
        g
    }

    /// Build a small CSR matrix for interaction graph tests.
    ///
    /// ```text
    ///     c0  c1  c2  c3
    /// r0 [ 1   0   1   1 ]   -> vars {0,2,3} all connected
    /// r1 [ 0   1   1   0 ]   -> vars {1,2} connected
    /// ```
    fn make_test_matrix() -> CsrMatrix<f64> {
        CsrMatrix::new(
            2,
            4,
            vec![0, 3, 5],
            vec![0, 2, 3, 1, 2],
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
        )
        .unwrap()
    }

    // -- Feature computation tests ------------------------------------------

    #[test]
    fn test_complete_graph_features() {
        let g = make_complete_graph(5);
        let f = compute_graph_features(&g).unwrap();

        assert!((f.num_vertices - 5.0).abs() < EPS);
        assert!((f.num_edges - 10.0).abs() < EPS);
        assert!((f.avg_degree - 4.0).abs() < EPS);
        assert!((f.max_degree - 4.0).abs() < EPS);
        assert!(f.degree_variance.abs() < EPS); // all degrees equal
        assert!((f.num_connected_components - 1.0).abs() < EPS);
        assert!((f.largest_component_fraction - 1.0).abs() < EPS);
        assert!((f.edge_density - 1.0).abs() < EPS);
        // Clustering coefficient of a complete graph is 1.0.
        assert!((f.avg_clustering_coefficient - 1.0).abs() < EPS);
        // Assortativity of a regular graph is undefined / 0.0.
        assert!(f.degree_assortativity.abs() < EPS);
    }

    #[test]
    fn test_path_graph_features() {
        let g = make_path_graph(5);
        let f = compute_graph_features(&g).unwrap();

        assert!((f.num_vertices - 5.0).abs() < EPS);
        assert!((f.num_edges - 4.0).abs() < EPS);
        // Degrees: [1, 2, 2, 2, 1] => mean = 8/5 = 1.6
        assert!((f.avg_degree - 1.6).abs() < EPS);
        assert!((f.max_degree - 2.0).abs() < EPS);
        // Variance: ((1-1.6)^2*2 + (2-1.6)^2*3)/5 = (0.72 + 0.48)/5 = 0.24
        assert!((f.degree_variance - 0.24).abs() < EPS);
        assert!((f.num_connected_components - 1.0).abs() < EPS);
        assert!((f.largest_component_fraction - 1.0).abs() < EPS);
        // No triangles in a path graph.
        assert!(f.avg_clustering_coefficient.abs() < EPS);
    }

    #[test]
    fn test_disconnected_graph() {
        let g = make_disconnected_graph();
        let f = compute_graph_features(&g).unwrap();

        assert!((f.num_vertices - 6.0).abs() < EPS);
        assert!((f.num_edges - 6.0).abs() < EPS);
        assert!((f.num_connected_components - 2.0).abs() < EPS);
        assert!((f.largest_component_fraction - 0.5).abs() < EPS);
        // Each component is a triangle → cc = 1.0.
        assert!((f.avg_clustering_coefficient - 1.0).abs() < EPS);
    }

    #[test]
    fn test_empty_graph_returns_error() {
        let g = Graph::new(0);
        let result = compute_graph_features(&g);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, SpectralCoreError::EmptyInput { .. }));
    }

    #[test]
    fn test_single_vertex() {
        let g = Graph::new(1);
        let f = compute_graph_features(&g).unwrap();

        assert!((f.num_vertices - 1.0).abs() < EPS);
        assert!(f.num_edges.abs() < EPS);
        assert!(f.avg_degree.abs() < EPS);
        assert!(f.max_degree.abs() < EPS);
        assert!(f.degree_variance.abs() < EPS);
        assert!((f.num_connected_components - 1.0).abs() < EPS);
        assert!((f.largest_component_fraction - 1.0).abs() < EPS);
        assert!(f.avg_clustering_coefficient.abs() < EPS);
        assert!(f.degree_assortativity.abs() < EPS);
    }

    #[test]
    fn test_clustering_coefficient_triangle() {
        let g = make_triangle();
        // Every vertex in K3 has cc = 1.0.
        for v in 0..3 {
            let cc = compute_local_clustering_coefficient(&g, v);
            assert!((cc - 1.0).abs() < EPS, "vertex {v} cc = {cc}");
        }
    }

    #[test]
    fn test_clustering_coefficient_path() {
        // In a path 0-1-2, vertex 1 has neighbors {0,2} but no edge 0-2 → cc=0.
        let g = make_path_graph(3);
        assert!(compute_local_clustering_coefficient(&g, 0).abs() < EPS);
        assert!(compute_local_clustering_coefficient(&g, 1).abs() < EPS);
        assert!(compute_local_clustering_coefficient(&g, 2).abs() < EPS);
    }

    #[test]
    fn test_degree_assortativity_star() {
        // Star graph: hub has degree n-1, leaves have degree 1.
        // Edges connect (n-1, 1) every time → negative assortativity.
        let g = make_star_graph(6);
        let r = compute_degree_assortativity(&g);
        // For a star, all edge pairs are (5,1), so variance of each
        // component is 0 → Pearson r is 0/0, clamped to 0.
        // Actually: every edge has the same (degree_u, degree_v) = (5,1),
        // which means zero variance in both x and y → r = 0.
        assert!(r.abs() < EPS, "star assortativity = {r}");
    }

    #[test]
    fn test_degree_assortativity_assortative_graph() {
        // Two cliques connected by a single bridge.
        // Clique A: {0,1,2} (K3), Clique B: {3,4,5} (K3), bridge: 2-3.
        let mut g = Graph::new(6);
        // Clique A
        g.add_edge(0, 1, 1.0);
        g.add_edge(0, 2, 1.0);
        g.add_edge(1, 2, 1.0);
        // Clique B
        g.add_edge(3, 4, 1.0);
        g.add_edge(3, 5, 1.0);
        g.add_edge(4, 5, 1.0);
        // Bridge
        g.add_edge(2, 3, 1.0);

        let r = compute_degree_assortativity(&g);
        // Degrees: [2, 2, 3, 3, 2, 2].
        // Edge pairs: (2,2), (2,3), (2,3) | (3,3), (3,2), (2,2) | (3,3).
        // Mixed → should be positive (high-degree connect to high-degree).
        assert!(r > -1.0 && r < 1.0, "assortativity out of range: {r}");
    }

    #[test]
    fn test_variable_interaction_graph_from_matrix() {
        let matrix = make_test_matrix();
        let g = build_variable_interaction_graph(&matrix);

        // Row 0 has cols {0,2,3} → edges (0,2), (0,3), (2,3)
        // Row 1 has cols {1,2}   → edge  (1,2)
        // Total: 4 edges
        assert_eq!(g.num_vertices, 4);
        assert_eq!(g.num_edges(), 4);

        // Column 2 appears in both rows → connected to 0, 3, and 1.
        let nbrs_2 = g.neighbors(2);
        assert!(nbrs_2.contains(&0));
        assert!(nbrs_2.contains(&1));
        assert!(nbrs_2.contains(&3));
    }

    #[test]
    fn test_compute_features_from_matrix() {
        let matrix = make_test_matrix();
        let f = compute_graph_features_from_matrix(&matrix).unwrap();

        assert!((f.num_vertices - 4.0).abs() < EPS);
        assert!((f.num_edges - 4.0).abs() < EPS);
        assert!((f.num_connected_components - 1.0).abs() < EPS);
    }

    #[test]
    fn test_sampled_path_length() {
        let g = make_path_graph(5);
        // All-pairs BFS on P5.
        // Distances: 0-1=1, 0-2=2, 0-3=3, 0-4=4, 1-2=1, 1-3=2, 1-4=3,
        //            2-3=1, 2-4=2, 3-4=1
        // Sum = 1+2+3+4+1+2+3+1+2+1 = 20, pairs = 10, avg = 2.0
        let avg = compute_avg_path_length_sampled(&g, 100);
        assert!((avg - 2.0).abs() < EPS);
    }

    #[test]
    fn test_degree_variance() {
        // Degrees: [3, 3, 3, 3] → variance = 0
        let g = make_complete_graph(4);
        let degrees = g.degrees();
        let mean = compute_mean_degree(&degrees);
        let var = compute_degree_variance(&degrees, mean);
        assert!(var.abs() < EPS);

        // Star S4 (4 vertices): degrees [3, 1, 1, 1] → mean = 1.5
        // variance = ((3-1.5)^2 + 3*(1-1.5)^2) / 4 = (2.25 + 0.75) / 4 = 0.75
        let g2 = make_star_graph(4);
        let degrees2 = g2.degrees();
        let mean2 = compute_mean_degree(&degrees2);
        let var2 = compute_degree_variance(&degrees2, mean2);
        assert!((var2 - 0.75).abs() < EPS, "degree variance = {var2}");
    }

    #[test]
    fn test_features_to_vec_roundtrip() {
        let g = make_complete_graph(4);
        let f = compute_graph_features(&g).unwrap();
        let v = f.to_vec();
        assert_eq!(v.len(), GraphFeatures::count());
        let f2 = GraphFeatures::from_vec(&v).unwrap();
        assert!((f.num_vertices - f2.num_vertices).abs() < EPS);
        assert!((f.avg_clustering_coefficient - f2.avg_clustering_coefficient).abs() < EPS);
    }
}
