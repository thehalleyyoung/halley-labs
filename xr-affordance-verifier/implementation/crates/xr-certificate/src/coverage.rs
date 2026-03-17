//! Coverage analysis: spatial coverage maps, frontier identification,
//! and heatmap generation over the body-parameter space.
//!
//! The `CoverageAnalyzer` computes grid-based coverage maps that track
//! which cells in the parameter space have been sampled and whether they
//! are accessible. It identifies frontier cells (boundary between
//! accessible and inaccessible) and computes uncovered fractions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use xr_types::certificate::SampleVerdict;
use xr_types::certificate::VerifiedRegion;
use xr_types::{ElementId, NUM_BODY_PARAMS};

// ──────────────────────── Coverage Cell ────────────────────────────────────

/// A single cell in the coverage grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageCell {
    /// Multi-dimensional grid index.
    pub grid_index: [usize; NUM_BODY_PARAMS],
    /// Lower bounds of this cell.
    pub lower: [f64; NUM_BODY_PARAMS],
    /// Upper bounds of this cell.
    pub upper: [f64; NUM_BODY_PARAMS],
    /// Number of samples in this cell.
    pub sample_count: usize,
    /// Number of passing samples.
    pub pass_count: usize,
    /// Number of failing samples.
    pub fail_count: usize,
    /// Whether this cell is covered by a verified region.
    pub verified: bool,
    /// Interpolated coverage estimate [0, 1].
    pub coverage_estimate: f64,
    /// Per-element coverage within this cell.
    pub element_coverage: HashMap<Uuid, (usize, usize)>,
}

impl CoverageCell {
    /// Create a new empty cell.
    pub fn new(
        grid_index: [usize; NUM_BODY_PARAMS],
        lower: [f64; NUM_BODY_PARAMS],
        upper: [f64; NUM_BODY_PARAMS],
    ) -> Self {
        Self {
            grid_index,
            lower,
            upper,
            sample_count: 0,
            pass_count: 0,
            fail_count: 0,
            verified: false,
            coverage_estimate: 0.0,
            element_coverage: HashMap::new(),
        }
    }

    /// Center point of this cell.
    pub fn center(&self) -> [f64; NUM_BODY_PARAMS] {
        let mut c = [0.0; NUM_BODY_PARAMS];
        for i in 0..NUM_BODY_PARAMS {
            c[i] = (self.lower[i] + self.upper[i]) * 0.5;
        }
        c
    }

    /// Volume of this cell.
    pub fn volume(&self) -> f64 {
        self.lower
            .iter()
            .zip(self.upper.iter())
            .map(|(lo, hi)| (hi - lo).max(0.0))
            .product()
    }

    /// Observed pass rate.
    pub fn pass_rate(&self) -> f64 {
        if self.sample_count == 0 {
            return 0.0;
        }
        self.pass_count as f64 / self.sample_count as f64
    }

    /// Whether this cell is on the frontier (has both pass and fail).
    pub fn is_frontier(&self) -> bool {
        self.pass_count > 0 && self.fail_count > 0
    }

    /// Whether this cell is fully covered (either verified or all-pass).
    pub fn is_fully_covered(&self) -> bool {
        self.verified || (self.sample_count > 0 && self.fail_count == 0)
    }

    /// Record a sample result for a specific element.
    pub fn record_sample(&mut self, element_id: Uuid, passed: bool) {
        self.sample_count += 1;
        if passed {
            self.pass_count += 1;
        } else {
            self.fail_count += 1;
        }
        let entry = self.element_coverage.entry(element_id).or_insert((0, 0));
        entry.0 += 1;
        if passed {
            entry.1 += 1;
        }
    }

    /// Check if a point falls within this cell.
    pub fn contains(&self, point: &[f64]) -> bool {
        if point.len() < NUM_BODY_PARAMS {
            return false;
        }
        for i in 0..NUM_BODY_PARAMS {
            if point[i] < self.lower[i] || point[i] > self.upper[i] {
                return false;
            }
        }
        true
    }
}

// ──────────────────────── Coverage Map ─────────────────────────────────────

/// Grid-based coverage map over the body-parameter space.
///
/// Divides the 5-dimensional parameter space into a regular grid
/// and tracks sample verdicts per cell. Supports interpolation
/// for continuous coverage estimates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageMap {
    /// All cells in the grid, indexed by flat index.
    pub cells: Vec<CoverageCell>,
    /// Number of cells per dimension.
    pub resolution: usize,
    /// Lower bounds of the parameter space.
    pub lower: [f64; NUM_BODY_PARAMS],
    /// Upper bounds of the parameter space.
    pub upper: [f64; NUM_BODY_PARAMS],
    /// Total volume of the parameter space.
    pub total_volume: f64,
}

impl CoverageMap {
    /// Create a new coverage map with the given resolution per dimension.
    pub fn new(
        lower: [f64; NUM_BODY_PARAMS],
        upper: [f64; NUM_BODY_PARAMS],
        resolution: usize,
    ) -> Self {
        let total = resolution.pow(NUM_BODY_PARAMS as u32);
        let steps: Vec<f64> = (0..NUM_BODY_PARAMS)
            .map(|d| (upper[d] - lower[d]) / resolution as f64)
            .collect();

        let mut cells = Vec::with_capacity(total);
        for flat_idx in 0..total {
            let mut grid = [0usize; NUM_BODY_PARAMS];
            let mut remainder = flat_idx;
            for d in (0..NUM_BODY_PARAMS).rev() {
                grid[d] = remainder % resolution;
                remainder /= resolution;
            }

            let mut lo = [0.0; NUM_BODY_PARAMS];
            let mut hi = [0.0; NUM_BODY_PARAMS];
            for d in 0..NUM_BODY_PARAMS {
                lo[d] = lower[d] + grid[d] as f64 * steps[d];
                hi[d] = lower[d] + (grid[d] + 1) as f64 * steps[d];
            }

            cells.push(CoverageCell::new(grid, lo, hi));
        }

        let total_volume = (0..NUM_BODY_PARAMS)
            .map(|d| (upper[d] - lower[d]).max(0.0))
            .product();

        Self {
            cells,
            resolution,
            lower,
            upper,
            total_volume,
        }
    }

    /// Total number of cells.
    pub fn total_cells(&self) -> usize {
        self.cells.len()
    }

    /// Find the cell index containing a point.
    pub fn find_cell(&self, point: &[f64]) -> Option<usize> {
        if point.len() < NUM_BODY_PARAMS {
            return None;
        }
        let r = self.resolution;
        let mut flat = 0;
        for d in 0..NUM_BODY_PARAMS {
            let range = self.upper[d] - self.lower[d];
            if range <= 0.0 {
                return None;
            }
            let normalized = (point[d] - self.lower[d]) / range;
            let idx = (normalized * r as f64).floor() as usize;
            let idx = idx.min(r - 1);
            flat = flat * r + idx;
        }
        if flat < self.cells.len() {
            Some(flat)
        } else {
            None
        }
    }

    /// Record a sample verdict into the appropriate cell.
    pub fn record_sample(&mut self, body_params: &[f64], element_id: Uuid, passed: bool) {
        if let Some(idx) = self.find_cell(body_params) {
            self.cells[idx].record_sample(element_id, passed);
        }
    }

    /// Mark cells covered by a verified region.
    pub fn mark_verified_region(&mut self, region: &VerifiedRegion) {
        for cell in &mut self.cells {
            if cell_overlaps_region(cell, region) {
                cell.verified = true;
            }
        }
    }

    /// Compute coverage estimates for all cells using neighbor interpolation.
    pub fn interpolate_coverage(&mut self) {
        let r = self.resolution;
        let n = self.cells.len();

        // First pass: set estimates from direct samples
        for cell in &mut self.cells {
            if cell.verified {
                cell.coverage_estimate = 1.0;
            } else if cell.sample_count > 0 {
                cell.coverage_estimate = cell.pass_rate();
            }
        }

        // Second pass: interpolate from neighbors for unsampled cells
        let estimates: Vec<f64> = self.cells.iter().map(|c| c.coverage_estimate).collect();
        let counts: Vec<usize> = self.cells.iter().map(|c| c.sample_count).collect();

        for flat in 0..n {
            if counts[flat] > 0 || self.cells[flat].verified {
                continue;
            }
            let grid = self.flat_to_grid(flat);
            let mut sum = 0.0;
            let mut weight = 0.0;

            for d in 0..NUM_BODY_PARAMS {
                if grid[d] > 0 {
                    let mut nb = grid;
                    nb[d] -= 1;
                    let nb_flat = self.grid_to_flat(&nb);
                    if counts[nb_flat] > 0 || self.cells[nb_flat].verified {
                        let w = 1.0;
                        sum += w * estimates[nb_flat];
                        weight += w;
                    }
                }
                if grid[d] + 1 < r {
                    let mut nb = grid;
                    nb[d] += 1;
                    let nb_flat = self.grid_to_flat(&nb);
                    if counts[nb_flat] > 0 || self.cells[nb_flat].verified {
                        let w = 1.0;
                        sum += w * estimates[nb_flat];
                        weight += w;
                    }
                }
            }

            if weight > 0.0 {
                self.cells[flat].coverage_estimate = sum / weight;
            }
        }
    }

    /// Compute overall coverage (volume-weighted average of cell coverage).
    pub fn overall_coverage(&self) -> f64 {
        if self.total_volume <= 0.0 || self.cells.is_empty() {
            return 0.0;
        }
        let total_cell_volume: f64 = self.cells.iter().map(|c| c.volume()).sum();
        if total_cell_volume <= 0.0 {
            return 0.0;
        }
        self.cells
            .iter()
            .map(|c| c.coverage_estimate * c.volume())
            .sum::<f64>()
            / total_cell_volume
    }

    /// Compute the fraction of cells that are uncovered (pass_rate < threshold).
    pub fn uncovered_fraction(&self, threshold: f64) -> f64 {
        let total = self.cells.len() as f64;
        if total <= 0.0 {
            return 1.0;
        }
        let uncovered = self
            .cells
            .iter()
            .filter(|c| !c.verified && c.coverage_estimate < threshold)
            .count();
        uncovered as f64 / total
    }

    /// Compute uncovered fraction accounting for verified regions.
    pub fn compute_uncovered_fraction(
        &self,
        verified_regions: &[VerifiedRegion],
    ) -> f64 {
        if self.cells.is_empty() {
            return 1.0;
        }
        let total_vol: f64 = self.cells.iter().map(|c| c.volume()).sum();
        if total_vol <= 0.0 {
            return 1.0;
        }

        let uncovered_vol: f64 = self
            .cells
            .iter()
            .filter(|c| {
                if c.verified {
                    return false;
                }
                // Check if any verified region covers this cell
                let covered = verified_regions.iter().any(|r| cell_overlaps_region(c, r));
                if covered {
                    return false;
                }
                c.sample_count == 0 || c.pass_rate() < 0.5
            })
            .map(|c| c.volume())
            .sum();

        uncovered_vol / total_vol
    }

    /// Convert flat index to grid indices.
    fn flat_to_grid(&self, flat: usize) -> [usize; NUM_BODY_PARAMS] {
        let r = self.resolution;
        let mut grid = [0usize; NUM_BODY_PARAMS];
        let mut remainder = flat;
        for d in (0..NUM_BODY_PARAMS).rev() {
            grid[d] = remainder % r;
            remainder /= r;
        }
        grid
    }

    /// Convert grid indices to flat index.
    fn grid_to_flat(&self, grid: &[usize; NUM_BODY_PARAMS]) -> usize {
        let r = self.resolution;
        let mut flat = 0;
        for d in 0..NUM_BODY_PARAMS {
            flat = flat * r + grid[d];
        }
        flat
    }

    /// Get cells with minimum per-element coverage below threshold.
    pub fn low_coverage_cells(&self, threshold: f64) -> Vec<usize> {
        self.cells
            .iter()
            .enumerate()
            .filter(|(_, c)| !c.verified && c.coverage_estimate < threshold)
            .map(|(i, _)| i)
            .collect()
    }

    /// Number of sampled cells.
    pub fn sampled_cells(&self) -> usize {
        self.cells
            .iter()
            .filter(|c| c.sample_count > 0 || c.verified)
            .count()
    }

    /// Generate heatmap data for a 2D slice of the parameter space.
    ///
    /// Projects coverage onto two selected dimensions, averaging over others.
    pub fn heatmap_2d(
        &self,
        dim_x: usize,
        dim_y: usize,
    ) -> HeatmapData {
        assert!(dim_x < NUM_BODY_PARAMS && dim_y < NUM_BODY_PARAMS);
        assert_ne!(dim_x, dim_y);

        let r = self.resolution;
        let mut grid = vec![vec![HeatmapPixel::default(); r]; r];
        let mut counts = vec![vec![0usize; r]; r];

        for cell in &self.cells {
            let ix = cell.grid_index[dim_x];
            let iy = cell.grid_index[dim_y];
            grid[iy][ix].coverage += cell.coverage_estimate;
            grid[iy][ix].sample_count += cell.sample_count;
            grid[iy][ix].pass_count += cell.pass_count;
            grid[iy][ix].fail_count += cell.fail_count;
            grid[iy][ix].verified |= cell.verified;
            counts[iy][ix] += 1;
        }

        // Average over collapsed dimensions
        for y in 0..r {
            for x in 0..r {
                if counts[y][x] > 0 {
                    grid[y][x].coverage /= counts[y][x] as f64;
                }
            }
        }

        HeatmapData {
            dim_x,
            dim_y,
            resolution: r,
            x_range: (self.lower[dim_x], self.upper[dim_x]),
            y_range: (self.lower[dim_y], self.upper[dim_y]),
            pixels: grid,
        }
    }
}

/// Check if a cell overlaps with a verified region.
fn cell_overlaps_region(cell: &CoverageCell, region: &VerifiedRegion) -> bool {
    if region.lower.len() < NUM_BODY_PARAMS || region.upper.len() < NUM_BODY_PARAMS {
        return false;
    }
    for d in 0..NUM_BODY_PARAMS {
        if cell.upper[d] <= region.lower[d] || region.upper[d] <= cell.lower[d] {
            return false;
        }
    }
    true
}

// ──────────────────────── Heatmap Data ────────────────────────────────────

/// Heatmap data for 2D visualization of coverage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatmapData {
    pub dim_x: usize,
    pub dim_y: usize,
    pub resolution: usize,
    pub x_range: (f64, f64),
    pub y_range: (f64, f64),
    pub pixels: Vec<Vec<HeatmapPixel>>,
}

/// A single pixel in the heatmap.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HeatmapPixel {
    pub coverage: f64,
    pub sample_count: usize,
    pub pass_count: usize,
    pub fail_count: usize,
    pub verified: bool,
}

// ──────────────────────── Frontier Cell ────────────────────────────────────

/// A cell on the accessibility frontier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrontierCell {
    pub cell_index: usize,
    pub grid_index: [usize; NUM_BODY_PARAMS],
    pub center: [f64; NUM_BODY_PARAMS],
    pub pass_rate: f64,
    pub sample_count: usize,
    pub gradient_magnitude: f64,
}

// ──────────────────────── Coverage Analyzer ────────────────────────────────

/// Analyzes coverage across the body-parameter space.
///
/// Builds a coverage map from sample verdicts, identifies frontier cells,
/// and computes coverage metrics.
pub struct CoverageAnalyzer {
    /// Resolution per dimension for the coverage grid.
    pub resolution: usize,
}

impl CoverageAnalyzer {
    /// Create a new analyzer with the given grid resolution.
    pub fn new(resolution: usize) -> Self {
        Self {
            resolution: resolution.max(2),
        }
    }

    /// Build a coverage map from sample verdicts.
    pub fn compute_coverage_map(
        &self,
        samples: &[SampleVerdict],
        lower: [f64; NUM_BODY_PARAMS],
        upper: [f64; NUM_BODY_PARAMS],
    ) -> CoverageMap {
        let mut map = CoverageMap::new(lower, upper, self.resolution);

        for sample in samples {
            let passed = sample.is_pass();
            map.record_sample(&sample.body_params, sample.element_id, passed);
        }

        map.interpolate_coverage();
        map
    }

    /// Identify frontier cells in the coverage map.
    ///
    /// Frontier cells are those where the coverage gradient is steep,
    /// indicating a transition between accessible and inaccessible regions.
    pub fn identify_frontier(&self, coverage_map: &CoverageMap) -> Vec<FrontierCell> {
        let _r = coverage_map.resolution;
        let mut frontiers = Vec::new();

        for (idx, cell) in coverage_map.cells.iter().enumerate() {
            if cell.sample_count == 0 && !cell.verified {
                continue;
            }

            let grid = coverage_map.flat_to_grid(idx);
            let gradient = self.compute_gradient(coverage_map, &grid);

            // A cell is on the frontier if:
            // 1. It has both pass and fail samples, OR
            // 2. Its coverage gradient is above threshold
            let is_frontier = cell.is_frontier() || gradient > 0.3;

            if is_frontier {
                frontiers.push(FrontierCell {
                    cell_index: idx,
                    grid_index: grid,
                    center: cell.center(),
                    pass_rate: cell.pass_rate(),
                    sample_count: cell.sample_count,
                    gradient_magnitude: gradient,
                });
            }
        }

        // Sort by gradient magnitude (descending)
        frontiers.sort_by(|a, b| {
            b.gradient_magnitude
                .partial_cmp(&a.gradient_magnitude)
                .unwrap()
        });

        frontiers
    }

    /// Compute the magnitude of the coverage gradient at a cell.
    fn compute_gradient(
        &self,
        map: &CoverageMap,
        grid: &[usize; NUM_BODY_PARAMS],
    ) -> f64 {
        let r = map.resolution;
        let center_flat = map.grid_to_flat(grid);
        let center_cov = map.cells[center_flat].coverage_estimate;
        let mut grad_sq = 0.0;

        for d in 0..NUM_BODY_PARAMS {
            let range = map.upper[d] - map.lower[d];
            let step = range / r as f64;
            if step <= 0.0 {
                continue;
            }

            let cov_plus = if grid[d] + 1 < r {
                let mut nb = *grid;
                nb[d] += 1;
                map.cells[map.grid_to_flat(&nb)].coverage_estimate
            } else {
                center_cov
            };

            let cov_minus = if grid[d] > 0 {
                let mut nb = *grid;
                nb[d] -= 1;
                map.cells[map.grid_to_flat(&nb)].coverage_estimate
            } else {
                center_cov
            };

            let partial = (cov_plus - cov_minus) / (2.0 * step);
            grad_sq += partial * partial;
        }

        grad_sq.sqrt()
    }

    /// Compute the uncovered fraction of the parameter space.
    pub fn compute_uncovered_fraction(
        &self,
        coverage_map: &CoverageMap,
        verified_regions: &[VerifiedRegion],
    ) -> f64 {
        coverage_map.compute_uncovered_fraction(verified_regions)
    }

    /// Get per-element coverage statistics.
    pub fn per_element_coverage(
        &self,
        samples: &[SampleVerdict],
    ) -> HashMap<ElementId, ElementCoverageStats> {
        let mut stats: HashMap<ElementId, ElementCoverageStats> = HashMap::new();

        for sample in samples {
            let entry = stats
                .entry(sample.element_id)
                .or_insert_with(|| ElementCoverageStats {
                    element_id: sample.element_id,
                    total_samples: 0,
                    pass_count: 0,
                    fail_count: 0,
                    coverage_rate: 0.0,
                    min_param: [f64::INFINITY; NUM_BODY_PARAMS],
                    max_param: [f64::NEG_INFINITY; NUM_BODY_PARAMS],
                });

            entry.total_samples += 1;
            if sample.is_pass() {
                entry.pass_count += 1;
            } else {
                entry.fail_count += 1;
            }

            for (d, &v) in sample.body_params.iter().enumerate() {
                if d < NUM_BODY_PARAMS {
                    entry.min_param[d] = entry.min_param[d].min(v);
                    entry.max_param[d] = entry.max_param[d].max(v);
                }
            }
        }

        // Compute coverage rates
        for entry in stats.values_mut() {
            if entry.total_samples > 0 {
                entry.coverage_rate = entry.pass_count as f64 / entry.total_samples as f64;
            }
        }

        stats
    }

    /// Generate a coverage summary.
    pub fn coverage_summary(
        &self,
        map: &CoverageMap,
        frontier: &[FrontierCell],
    ) -> CoverageSummary {
        let total_cells = map.total_cells();
        let sampled = map.sampled_cells();
        let verified = map.cells.iter().filter(|c| c.verified).count();

        CoverageSummary {
            total_cells,
            sampled_cells: sampled,
            verified_cells: verified,
            frontier_cells: frontier.len(),
            overall_coverage: map.overall_coverage(),
            uncovered_fraction: map.uncovered_fraction(0.5),
            resolution: map.resolution,
        }
    }
}

/// Per-element coverage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElementCoverageStats {
    pub element_id: ElementId,
    pub total_samples: usize,
    pub pass_count: usize,
    pub fail_count: usize,
    pub coverage_rate: f64,
    pub min_param: [f64; NUM_BODY_PARAMS],
    pub max_param: [f64; NUM_BODY_PARAMS],
}

/// Summary of coverage analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageSummary {
    pub total_cells: usize,
    pub sampled_cells: usize,
    pub verified_cells: usize,
    pub frontier_cells: usize,
    pub overall_coverage: f64,
    pub uncovered_fraction: f64,
    pub resolution: usize,
}

impl std::fmt::Display for CoverageSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Coverage: {:.4} | Cells: {}/{} sampled, {} verified, {} frontier | \
             Uncovered: {:.4}",
            self.overall_coverage,
            self.sampled_cells,
            self.total_cells,
            self.verified_cells,
            self.frontier_cells,
            self.uncovered_fraction,
        )
    }
}

// ────────────────────────────── Tests ──────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use xr_types::certificate::SampleVerdict;

    fn test_bounds() -> ([f64; 5], [f64; 5]) {
        ([0.0; 5], [1.0; 5])
    }

    fn test_element() -> Uuid {
        Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap()
    }

    #[test]
    fn test_coverage_cell_basic() {
        let cell = CoverageCell::new([0; 5], [0.0; 5], [1.0; 5]);
        assert_eq!(cell.volume(), 1.0);
        assert_eq!(cell.center(), [0.5; 5]);
        assert_eq!(cell.pass_rate(), 0.0);
        assert!(!cell.is_frontier());
    }

    #[test]
    fn test_coverage_cell_recording() {
        let mut cell = CoverageCell::new([0; 5], [0.0; 5], [1.0; 5]);
        let eid = test_element();
        cell.record_sample(eid, true);
        cell.record_sample(eid, false);
        assert_eq!(cell.sample_count, 2);
        assert_eq!(cell.pass_count, 1);
        assert!(cell.is_frontier());
        assert!((cell.pass_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_coverage_map_creation() {
        let (lower, upper) = test_bounds();
        let map = CoverageMap::new(lower, upper, 3);
        assert_eq!(map.total_cells(), 243); // 3^5
        assert!((map.total_volume - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_coverage_map_find_cell() {
        let (lower, upper) = test_bounds();
        let map = CoverageMap::new(lower, upper, 3);

        let idx = map.find_cell(&[0.1, 0.1, 0.1, 0.1, 0.1]);
        assert!(idx.is_some());
        assert_eq!(idx.unwrap(), 0);

        let idx2 = map.find_cell(&[0.5, 0.5, 0.5, 0.5, 0.5]);
        assert!(idx2.is_some());
    }

    #[test]
    fn test_coverage_map_record() {
        let (lower, upper) = test_bounds();
        let mut map = CoverageMap::new(lower, upper, 3);
        let eid = test_element();
        map.record_sample(&[0.1, 0.1, 0.1, 0.1, 0.1], eid, true);
        assert_eq!(map.cells[0].sample_count, 1);
        assert_eq!(map.cells[0].pass_count, 1);
    }

    #[test]
    fn test_coverage_analyzer() {
        let (lower, upper) = test_bounds();
        let analyzer = CoverageAnalyzer::new(3);
        let eid = test_element();

        let samples: Vec<SampleVerdict> = (0..20)
            .map(|i| {
                let t = i as f64 / 20.0;
                SampleVerdict::pass(vec![t, t, t, t, t], eid)
            })
            .collect();

        let map = analyzer.compute_coverage_map(&samples, lower, upper);
        let coverage = map.overall_coverage();
        assert!(coverage > 0.0);
    }

    #[test]
    fn test_frontier_identification() {
        let (lower, upper) = test_bounds();
        let mut map = CoverageMap::new(lower, upper, 3);
        let eid = test_element();

        // Make one cell frontier
        map.cells[0].record_sample(eid, true);
        map.cells[0].record_sample(eid, false);

        let analyzer = CoverageAnalyzer::new(3);
        let frontier = analyzer.identify_frontier(&map);
        assert!(!frontier.is_empty());
    }

    #[test]
    fn test_heatmap_generation() {
        let (lower, upper) = test_bounds();
        let mut map = CoverageMap::new(lower, upper, 3);
        let eid = test_element();
        map.record_sample(&[0.1, 0.1, 0.1, 0.1, 0.1], eid, true);
        map.interpolate_coverage();

        let heatmap = map.heatmap_2d(0, 1);
        assert_eq!(heatmap.resolution, 3);
        assert_eq!(heatmap.dim_x, 0);
        assert_eq!(heatmap.dim_y, 1);
        assert_eq!(heatmap.pixels.len(), 3);
    }

    #[test]
    fn test_verified_region_marking() {
        let (lower, upper) = test_bounds();
        let mut map = CoverageMap::new(lower, upper, 3);
        let eid = test_element();
        let region = VerifiedRegion::new(
            "test",
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.5, 0.5, 0.5, 0.5, 0.5],
            eid,
        );
        map.mark_verified_region(&region);
        let verified = map.cells.iter().filter(|c| c.verified).count();
        assert!(verified > 0);
    }

    #[test]
    fn test_per_element_coverage() {
        let analyzer = CoverageAnalyzer::new(3);
        let e1 = Uuid::new_v4();
        let e2 = Uuid::new_v4();
        let samples = vec![
            SampleVerdict::pass(vec![0.5; 5], e1),
            SampleVerdict::fail(vec![0.3; 5], e1, "test".into()),
            SampleVerdict::pass(vec![0.5; 5], e2),
        ];
        let stats = analyzer.per_element_coverage(&samples);
        assert_eq!(stats.len(), 2);
        assert!((stats[&e1].coverage_rate - 0.5).abs() < 1e-10);
        assert!((stats[&e2].coverage_rate - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_coverage_summary() {
        let (lower, upper) = test_bounds();
        let map = CoverageMap::new(lower, upper, 2);
        let analyzer = CoverageAnalyzer::new(2);
        let summary = analyzer.coverage_summary(&map, &[]);
        assert_eq!(summary.total_cells, 32);
        assert_eq!(summary.sampled_cells, 0);
    }

    #[test]
    fn test_uncovered_fraction() {
        let (lower, upper) = test_bounds();
        let map = CoverageMap::new(lower, upper, 2);
        let fraction = map.uncovered_fraction(0.5);
        assert!((fraction - 1.0).abs() < 1e-10); // all uncovered initially
    }

    #[test]
    fn test_low_coverage_cells() {
        let (lower, upper) = test_bounds();
        let map = CoverageMap::new(lower, upper, 2);
        let low = map.low_coverage_cells(0.5);
        assert_eq!(low.len(), 32); // all cells are below threshold
    }

    #[test]
    fn test_cell_contains() {
        let cell = CoverageCell::new([0; 5], [0.0; 5], [0.5; 5]);
        assert!(cell.contains(&[0.1, 0.2, 0.3, 0.4, 0.1]));
        assert!(!cell.contains(&[0.6, 0.2, 0.3, 0.4, 0.1]));
    }

    #[test]
    fn test_interpolation() {
        let (lower, upper) = test_bounds();
        let mut map = CoverageMap::new(lower, upper, 3);
        let eid = test_element();

        // Sample one cell as fully passing
        map.cells[0].record_sample(eid, true);
        map.cells[0].record_sample(eid, true);
        map.interpolate_coverage();

        // The sampled cell should have coverage 1.0
        assert!((map.cells[0].coverage_estimate - 1.0).abs() < 1e-10);
    }
}
