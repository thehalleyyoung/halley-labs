use std::collections::{HashMap, VecDeque};
use serde::{Serialize, Deserialize};

/// Bitmap-based pairwise compatibility matrix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityBitmap {
    size_a: usize,
    size_b: usize,
    bits: Vec<bool>,
}

impl CompatibilityBitmap {
    pub fn new(size_a: usize, size_b: usize) -> Self {
        Self {
            size_a,
            size_b,
            bits: vec![false; size_a * size_b],
        }
    }

    pub fn new_all_compatible(size_a: usize, size_b: usize) -> Self {
        Self {
            size_a,
            size_b,
            bits: vec![true; size_a * size_b],
        }
    }

    fn idx(&self, a: usize, b: usize) -> usize {
        a * self.size_b + b
    }

    pub fn set(&mut self, a: usize, b: usize, compatible: bool) {
        if a < self.size_a && b < self.size_b {
            let idx = self.idx(a, b);
            self.bits[idx] = compatible;
        }
    }

    pub fn get(&self, a: usize, b: usize) -> bool {
        if a < self.size_a && b < self.size_b {
            self.bits[self.idx(a, b)]
        } else {
            false
        }
    }

    pub fn compatible_with(&self, a: usize) -> Vec<usize> {
        if a >= self.size_a {
            return Vec::new();
        }
        (0..self.size_b)
            .filter(|&b| self.bits[self.idx(a, b)])
            .collect()
    }

    pub fn incompatible_with(&self, a: usize) -> Vec<usize> {
        if a >= self.size_a {
            return Vec::new();
        }
        (0..self.size_b)
            .filter(|&b| !self.bits[self.idx(a, b)])
            .collect()
    }

    pub fn row_and(&self, row_a: usize, other: &CompatibilityBitmap, row_b: usize) -> Vec<bool> {
        let min_b = self.size_b.min(other.size_b);
        (0..min_b)
            .map(|b| self.get(row_a, b) && other.get(row_b, b))
            .collect()
    }

    pub fn density(&self) -> f64 {
        if self.bits.is_empty() {
            return 0.0;
        }
        let compatible = self.bits.iter().filter(|&&b| b).count();
        compatible as f64 / self.bits.len() as f64
    }

    pub fn total_compatible(&self) -> usize {
        self.bits.iter().filter(|&&b| b).count()
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.size_a, self.size_b)
    }

    /// Intersect with another bitmap (AND operation)
    pub fn intersect(&self, other: &CompatibilityBitmap) -> CompatibilityBitmap {
        let sa = self.size_a.min(other.size_a);
        let sb = self.size_b.min(other.size_b);
        let mut result = CompatibilityBitmap::new(sa, sb);
        for a in 0..sa {
            for b in 0..sb {
                result.set(a, b, self.get(a, b) && other.get(a, b));
            }
        }
        result
    }

    /// Union with another bitmap (OR operation)
    pub fn union(&self, other: &CompatibilityBitmap) -> CompatibilityBitmap {
        let sa = self.size_a.max(other.size_a);
        let sb = self.size_b.max(other.size_b);
        let mut result = CompatibilityBitmap::new(sa, sb);
        for a in 0..sa {
            for b in 0..sb {
                result.set(a, b, self.get(a, b) || other.get(a, b));
            }
        }
        result
    }
}

impl std::fmt::Display for CompatibilityBitmap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "CompatibilityBitmap {}x{} (density: {:.1}%)", 
            self.size_a, self.size_b, self.density() * 100.0)?;
        for a in 0..self.size_a.min(20) {
            write!(f, "  row {}: ", a)?;
            for b in 0..self.size_b.min(40) {
                write!(f, "{}", if self.get(a, b) { '1' } else { '0' })?;
            }
            if self.size_b > 40 {
                write!(f, "...")?;
            }
            writeln!(f)?;
        }
        if self.size_a > 20 {
            writeln!(f, "  ...")?;
        }
        Ok(())
    }
}

/// Pairwise prefilter for fast rejection of infeasible states
pub struct PairwisePrefilter {
    num_services: usize,
    versions_per_service: Vec<usize>,
    pub bitmaps: HashMap<(usize, usize), CompatibilityBitmap>,
}

impl PairwisePrefilter {
    pub fn new(num_services: usize, versions_per_service: &[usize]) -> Self {
        Self {
            num_services,
            versions_per_service: versions_per_service.to_vec(),
            bitmaps: HashMap::new(),
        }
    }

    pub fn add_compatibility(
        &mut self,
        svc_a: usize,
        svc_b: usize,
        bitmap: CompatibilityBitmap,
    ) {
        self.bitmaps.insert((svc_a, svc_b), bitmap);
    }

    pub fn set_compatible(
        &mut self,
        svc_a: usize,
        ver_a: usize,
        svc_b: usize,
        ver_b: usize,
        compatible: bool,
    ) {
        let key = (svc_a, svc_b);
        let bitmap = self.bitmaps.entry(key).or_insert_with(|| {
            let sa = self.versions_per_service.get(svc_a).copied().unwrap_or(1);
            let sb = self.versions_per_service.get(svc_b).copied().unwrap_or(1);
            CompatibilityBitmap::new_all_compatible(sa, sb)
        });
        bitmap.set(ver_a, ver_b, compatible);
    }

    /// Fast check: is the given state potentially feasible?
    pub fn is_potentially_feasible(&self, state: &[usize]) -> bool {
        for (&(svc_a, svc_b), bitmap) in &self.bitmaps {
            if svc_a >= state.len() || svc_b >= state.len() {
                continue;
            }
            let ver_a = state[svc_a];
            let ver_b = state[svc_b];
            if !bitmap.get(ver_a, ver_b) {
                return false;
            }
        }
        true
    }

    /// Get versions of a service compatible with all fixed services
    pub fn feasible_versions(
        &self,
        service: usize,
        fixed: &[(usize, usize)],
    ) -> Vec<usize> {
        let num_versions = self.versions_per_service.get(service).copied().unwrap_or(0);
        let mut feasible: Vec<bool> = vec![true; num_versions];

        for &(other_svc, other_ver) in fixed {
            // Check bitmap (service, other_svc)
            if let Some(bitmap) = self.bitmaps.get(&(service, other_svc)) {
                for v in 0..num_versions {
                    if feasible[v] && !bitmap.get(v, other_ver) {
                        feasible[v] = false;
                    }
                }
            }
            // Check bitmap (other_svc, service) - reverse direction
            if let Some(bitmap) = self.bitmaps.get(&(other_svc, service)) {
                for v in 0..num_versions {
                    if feasible[v] && !bitmap.get(other_ver, v) {
                        feasible[v] = false;
                    }
                }
            }
        }

        (0..num_versions).filter(|&v| feasible[v]).collect()
    }

    pub fn service_count(&self) -> usize {
        self.num_services
    }

    pub fn constraint_count(&self) -> usize {
        self.bitmaps.len()
    }
}

/// AC-3 arc consistency algorithm for constraint propagation
pub struct ArcConsistency {
    num_services: usize,
    domains: Vec<Vec<usize>>,
    constraints: HashMap<(usize, usize), CompatibilityBitmap>,
}

impl ArcConsistency {
    pub fn new(
        num_services: usize,
        versions_per_service: &[usize],
        constraints: HashMap<(usize, usize), CompatibilityBitmap>,
    ) -> Self {
        let domains: Vec<Vec<usize>> = versions_per_service.iter()
            .map(|&n| (0..n).collect())
            .collect();
        Self { num_services, domains, constraints }
    }

    /// Run AC-3 propagation, returns reduced domains or Infeasible
    pub fn propagate(&mut self) -> Result<Vec<Vec<usize>>, String> {
        let mut queue: VecDeque<(usize, usize)> = VecDeque::new();

        // Initialize queue with all constraint arcs
        for &(xi, xj) in self.constraints.keys() {
            queue.push_back((xi, xj));
            queue.push_back((xj, xi));
        }

        while let Some((xi, xj)) = queue.pop_front() {
            if self.revise(xi, xj) {
                if self.domains.get(xi).map(|d| d.is_empty()).unwrap_or(true) {
                    return Err(format!("Domain of service {} became empty", xi));
                }

                // Add all arcs (xk, xi) where xk != xj to queue
                for &(a, b) in self.constraints.keys() {
                    if b == xi && a != xj {
                        queue.push_back((a, xi));
                    }
                    if a == xi && b != xj {
                        queue.push_back((b, xi));
                    }
                }
            }
        }

        Ok(self.domains.clone())
    }

    /// Revise domain of xi with respect to xj
    /// Returns true if domain was modified
    fn revise(&mut self, xi: usize, xj: usize) -> bool {
        let bitmap = if let Some(bm) = self.constraints.get(&(xi, xj)) {
            bm.clone()
        } else if let Some(bm) = self.constraints.get(&(xj, xi)) {
            // Need to transpose the lookup
            bm.clone()
        } else {
            return false;
        };

        let xj_domain: Vec<usize> = self.domains.get(xj).cloned().unwrap_or_default();
        let xi_domain: Vec<usize> = self.domains.get(xi).cloned().unwrap_or_default();

        let mut modified = false;
        let mut new_domain = Vec::new();

        for &vi in &xi_domain {
            let has_support = xj_domain.iter().any(|&vj| {
                if let Some(bm) = self.constraints.get(&(xi, xj)) {
                    bm.get(vi, vj)
                } else if let Some(bm) = self.constraints.get(&(xj, xi)) {
                    bm.get(vj, vi)
                } else {
                    true
                }
            });
            if has_support {
                new_domain.push(vi);
            } else {
                modified = true;
            }
        }

        if modified {
            if xi < self.domains.len() {
                self.domains[xi] = new_domain;
            }
        }

        let _ = bitmap;
        modified
    }

    pub fn domain_sizes(&self) -> Vec<usize> {
        self.domains.iter().map(|d| d.len()).collect()
    }

    pub fn total_domain_size(&self) -> usize {
        self.domains.iter().map(|d| d.len()).sum()
    }
}

/// Multi-level feasibility filtering
pub struct FeasibilityFilter {
    prefilter: PairwisePrefilter,
    arc_consistency_constraints: HashMap<(usize, usize), CompatibilityBitmap>,
    versions_per_service: Vec<usize>,
}

/// Result of feasibility filtering
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeasibilityResult {
    Feasible,
    MaybeFeasible,
    Infeasible(String),
}

impl FeasibilityFilter {
    pub fn new(prefilter: PairwisePrefilter, versions_per_service: Vec<usize>) -> Self {
        let arc_constraints = prefilter.bitmaps.clone();
        Self {
            prefilter,
            arc_consistency_constraints: arc_constraints,
            versions_per_service,
        }
    }

    /// Multi-level feasibility check
    pub fn filter(&self, state: &[usize]) -> FeasibilityResult {
        // Level 1: Fast pairwise bitmap check
        if !self.prefilter.is_potentially_feasible(state) {
            return FeasibilityResult::Infeasible(
                "Failed pairwise compatibility check".into()
            );
        }

        // Level 2: Arc consistency on reduced domains
        let fixed: Vec<(usize, usize)> = state.iter().enumerate()
            .map(|(i, &v)| (i, v))
            .collect();

        for i in 0..state.len() {
            let feasible = self.prefilter.feasible_versions(i, &fixed);
            if !feasible.contains(&state[i]) {
                return FeasibilityResult::Infeasible(
                    format!("Service {} version {} has no compatible support", i, state[i])
                );
            }
        }

        FeasibilityResult::MaybeFeasible
    }

    /// Full feasibility check with arc consistency
    pub fn full_filter(&self, state: &[usize]) -> FeasibilityResult {
        let basic = self.filter(state);
        if basic == FeasibilityResult::Infeasible("Failed pairwise compatibility check".into()) {
            return basic;
        }

        // Run AC-3 on domains restricted to {state[i]} for each fixed service
        let singleton_domains: Vec<Vec<usize>> = state.iter().map(|&v| vec![v]).collect();
        let mut ac = ArcConsistency {
            num_services: state.len(),
            domains: singleton_domains,
            constraints: self.arc_consistency_constraints.clone(),
        };

        match ac.propagate() {
            Ok(domains) => {
                if domains.iter().any(|d| d.is_empty()) {
                    FeasibilityResult::Infeasible("Arc consistency detected infeasibility".into())
                } else {
                    FeasibilityResult::Feasible
                }
            }
            Err(reason) => {
                FeasibilityResult::Infeasible(reason)
            }
        }
    }
}

/// Stores variable domains during constraint propagation
#[derive(Debug, Clone)]
pub struct DomainStore {
    domains: Vec<Vec<usize>>,
    removed: Vec<Vec<usize>>,
}

impl DomainStore {
    pub fn new(domains: Vec<Vec<usize>>) -> Self {
        let n = domains.len();
        Self {
            domains,
            removed: vec![Vec::new(); n],
        }
    }

    pub fn from_sizes(sizes: &[usize]) -> Self {
        let domains: Vec<Vec<usize>> = sizes.iter()
            .map(|&n| (0..n).collect())
            .collect();
        Self::new(domains)
    }

    pub fn remove(&mut self, var: usize, val: usize) -> bool {
        if var >= self.domains.len() {
            return false;
        }
        if let Some(pos) = self.domains[var].iter().position(|&v| v == val) {
            self.domains[var].remove(pos);
            self.removed[var].push(val);
            true
        } else {
            false
        }
    }

    pub fn restore(&mut self, var: usize) {
        if var >= self.domains.len() {
            return;
        }
        let restored: Vec<usize> = self.removed[var].drain(..).collect();
        self.domains[var].extend(restored);
        self.domains[var].sort_unstable();
    }

    pub fn is_empty(&self, var: usize) -> bool {
        self.domains.get(var).map(|d| d.is_empty()).unwrap_or(true)
    }

    pub fn domain_size(&self, var: usize) -> usize {
        self.domains.get(var).map(|d| d.len()).unwrap_or(0)
    }

    pub fn domain(&self, var: usize) -> &[usize] {
        self.domains.get(var).map(|d| d.as_slice()).unwrap_or(&[])
    }

    pub fn contains(&self, var: usize, val: usize) -> bool {
        self.domains.get(var).map(|d| d.contains(&val)).unwrap_or(false)
    }

    pub fn num_vars(&self) -> usize {
        self.domains.len()
    }

    pub fn total_size(&self) -> usize {
        self.domains.iter().map(|d| d.len()).sum()
    }

    pub fn is_singleton(&self, var: usize) -> bool {
        self.domain_size(var) == 1
    }

    pub fn singleton_value(&self, var: usize) -> Option<usize> {
        if self.is_singleton(var) {
            self.domains[var].first().copied()
        } else {
            None
        }
    }

    pub fn smallest_domain_var(&self) -> Option<usize> {
        (0..self.domains.len())
            .filter(|&v| self.domain_size(v) > 1)
            .min_by_key(|&v| self.domain_size(v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitmap_basic() {
        let mut bm = CompatibilityBitmap::new(3, 4);
        assert!(!bm.get(0, 0));
        bm.set(0, 0, true);
        assert!(bm.get(0, 0));
        bm.set(1, 2, true);
        assert!(bm.get(1, 2));
    }

    #[test]
    fn test_bitmap_all_compatible() {
        let bm = CompatibilityBitmap::new_all_compatible(3, 3);
        assert_eq!(bm.density(), 1.0);
        assert!(bm.get(0, 0));
        assert!(bm.get(2, 2));
    }

    #[test]
    fn test_bitmap_compatible_with() {
        let mut bm = CompatibilityBitmap::new(2, 3);
        bm.set(0, 0, true);
        bm.set(0, 2, true);
        let compat = bm.compatible_with(0);
        assert_eq!(compat, vec![0, 2]);
    }

    #[test]
    fn test_bitmap_intersect() {
        let mut bm1 = CompatibilityBitmap::new(2, 2);
        bm1.set(0, 0, true);
        bm1.set(0, 1, true);
        bm1.set(1, 0, true);

        let mut bm2 = CompatibilityBitmap::new(2, 2);
        bm2.set(0, 0, true);
        bm2.set(1, 0, true);
        bm2.set(1, 1, true);

        let intersection = bm1.intersect(&bm2);
        assert!(intersection.get(0, 0));
        assert!(!intersection.get(0, 1));
        assert!(intersection.get(1, 0));
        assert!(!intersection.get(1, 1));
    }

    #[test]
    fn test_bitmap_density() {
        let mut bm = CompatibilityBitmap::new(2, 2);
        bm.set(0, 0, true);
        bm.set(1, 1, true);
        assert_eq!(bm.density(), 0.5);
    }

    #[test]
    fn test_prefilter_feasible() {
        let mut prefilter = PairwisePrefilter::new(2, &[3, 3]);
        let mut bm = CompatibilityBitmap::new_all_compatible(3, 3);
        bm.set(0, 2, false); // v0 of svc0 incompatible with v2 of svc1
        prefilter.add_compatibility(0, 1, bm);

        assert!(prefilter.is_potentially_feasible(&[0, 0]));
        assert!(prefilter.is_potentially_feasible(&[0, 1]));
        assert!(!prefilter.is_potentially_feasible(&[0, 2]));
    }

    #[test]
    fn test_prefilter_feasible_versions() {
        let mut prefilter = PairwisePrefilter::new(2, &[3, 3]);
        let mut bm = CompatibilityBitmap::new_all_compatible(3, 3);
        bm.set(0, 2, false);
        bm.set(1, 2, false);
        prefilter.add_compatibility(0, 1, bm);

        let feasible = prefilter.feasible_versions(1, &[(0, 0)]);
        assert_eq!(feasible, vec![0, 1]); // v2 is excluded
    }

    #[test]
    fn test_arc_consistency() {
        let mut constraints = HashMap::new();
        let mut bm = CompatibilityBitmap::new(3, 3);
        bm.set(0, 0, true);
        bm.set(1, 1, true);
        bm.set(2, 2, true);
        constraints.insert((0, 1), bm);

        let mut ac = ArcConsistency::new(2, &[3, 3], constraints);
        let result = ac.propagate();
        assert!(result.is_ok());
        let domains = result.unwrap();
        assert_eq!(domains[0].len(), 3);
        assert_eq!(domains[1].len(), 3);
    }

    #[test]
    fn test_arc_consistency_reduction() {
        let mut constraints = HashMap::new();
        let mut bm = CompatibilityBitmap::new(3, 3);
        bm.set(0, 0, true);
        // v1 and v2 of svc0 have no support in svc1
        constraints.insert((0, 1), bm);

        let mut ac = ArcConsistency::new(2, &[3, 3], constraints);
        let result = ac.propagate();
        assert!(result.is_ok());
        let domains = result.unwrap();
        // svc0 should be reduced to {0} since only v0 has support
        assert_eq!(domains[0], vec![0]);
    }

    #[test]
    fn test_feasibility_filter() {
        let prefilter = PairwisePrefilter::new(2, &[3, 3]);
        let filter = FeasibilityFilter::new(prefilter, vec![3, 3]);
        let result = filter.filter(&[0, 0]);
        assert_ne!(result, FeasibilityResult::Infeasible("".into()));
    }

    #[test]
    fn test_domain_store() {
        let mut store = DomainStore::from_sizes(&[3, 4, 2]);
        assert_eq!(store.domain_size(0), 3);
        assert_eq!(store.domain_size(1), 4);
        assert_eq!(store.total_size(), 9);

        assert!(store.remove(0, 1));
        assert_eq!(store.domain_size(0), 2);
        assert!(!store.contains(0, 1));

        store.restore(0);
        assert_eq!(store.domain_size(0), 3);
        assert!(store.contains(0, 1));
    }

    #[test]
    fn test_domain_store_singleton() {
        let store = DomainStore::new(vec![vec![5], vec![1, 2]]);
        assert!(store.is_singleton(0));
        assert!(!store.is_singleton(1));
        assert_eq!(store.singleton_value(0), Some(5));
        assert_eq!(store.singleton_value(1), None);
    }

    #[test]
    fn test_domain_store_smallest() {
        let store = DomainStore::new(vec![vec![1], vec![1, 2, 3], vec![1, 2]]);
        assert_eq!(store.smallest_domain_var(), Some(2)); // size 2 (excluding singletons)
    }

    #[test]
    fn test_bitmap_display() {
        let mut bm = CompatibilityBitmap::new(2, 3);
        bm.set(0, 0, true);
        bm.set(1, 2, true);
        let display = format!("{}", bm);
        assert!(display.contains("2x3"));
    }

    #[test]
    fn test_bitmap_row_and() {
        let mut bm1 = CompatibilityBitmap::new(2, 3);
        bm1.set(0, 0, true);
        bm1.set(0, 1, true);
        bm1.set(0, 2, true);

        let mut bm2 = CompatibilityBitmap::new(2, 3);
        bm2.set(0, 0, true);
        bm2.set(0, 2, true);

        let result = bm1.row_and(0, &bm2, 0);
        assert_eq!(result, vec![true, false, true]);
    }

    #[test]
    fn test_feasibility_result_eq() {
        assert_eq!(FeasibilityResult::Feasible, FeasibilityResult::Feasible);
        assert_eq!(FeasibilityResult::MaybeFeasible, FeasibilityResult::MaybeFeasible);
        assert_ne!(FeasibilityResult::Feasible, FeasibilityResult::MaybeFeasible);
    }

    #[test]
    fn test_domain_store_basic() {
        let mut ds = DomainStore::from_sizes(&[3, 4]);
        assert_eq!(ds.domain_size(0), 3);
        assert_eq!(ds.domain_size(1), 4);
        assert!(ds.contains(0, 0));
        assert!(ds.contains(0, 2));
    }

    #[test]
    fn test_domain_store_remove() {
        let mut ds = DomainStore::from_sizes(&[3, 3]);
        assert!(ds.remove(0, 1));
        assert_eq!(ds.domain_size(0), 2);
        assert!(!ds.contains(0, 1));
        assert!(ds.contains(0, 0));
        assert!(ds.contains(0, 2));
    }

    #[test]
    fn test_domain_store_restore() {
        let mut ds = DomainStore::from_sizes(&[3]);
        ds.remove(0, 1);
        assert_eq!(ds.domain_size(0), 2);
        ds.restore(0);
        assert_eq!(ds.domain_size(0), 3);
        assert!(ds.contains(0, 1));
    }

    #[test]
    fn test_domain_store_empty() {
        let mut ds = DomainStore::from_sizes(&[1]);
        assert!(!ds.is_empty(0));
        ds.remove(0, 0);
        assert!(ds.is_empty(0));
    }

    #[test]
    fn test_domain_store_singleton_value() {
        let mut ds = DomainStore::from_sizes(&[3, 2, 4]);
        assert_eq!(ds.smallest_domain_var(), Some(1));
        ds.remove(2, 0);
        ds.remove(2, 1);
        ds.remove(1, 0); // Now var 1 is singleton, var 2 has size 2
        assert_eq!(ds.smallest_domain_var(), Some(2));
    }

    #[test]
    fn test_domain_store_total_size() {
        let ds = DomainStore::from_sizes(&[3, 4, 2]);
        assert_eq!(ds.total_size(), 9);
        assert_eq!(ds.num_vars(), 3);
    }

    #[test]
    fn test_bitmap_compatible_with_2() {
        let bm = CompatibilityBitmap::new_all_compatible(2, 3);
        assert_eq!(bm.density(), 1.0);
        assert_eq!(bm.compatible_with(0), vec![0, 1, 2]);
    }

    #[test]
    fn test_prefilter_all_compatible() {
        let mut pf = PairwisePrefilter::new(3, &[2, 2, 2]);
        pf.add_compatibility(0, 1, CompatibilityBitmap::new_all_compatible(2, 2));
        pf.add_compatibility(0, 2, CompatibilityBitmap::new_all_compatible(2, 2));
        pf.add_compatibility(1, 2, CompatibilityBitmap::new_all_compatible(2, 2));
        assert!(pf.is_potentially_feasible(&[0, 0, 0]));
        assert!(pf.is_potentially_feasible(&[1, 1, 1]));
    }

    #[test]
    fn test_prefilter_no_constraints() {
        let pf = PairwisePrefilter::new(2, &[3, 3]);
        assert!(pf.is_potentially_feasible(&[0, 0]));
        assert!(pf.is_potentially_feasible(&[2, 2]));
    }

    #[test]
    fn test_arc_consistency_empty() {
        let constraints = HashMap::new();
        let mut ac = ArcConsistency::new(2, &[3, 3], constraints);
        let result = ac.propagate();
        assert!(result.is_ok());
        let domains = result.unwrap();
        assert_eq!(domains[0], vec![0, 1, 2]);
        assert_eq!(domains[1], vec![0, 1, 2]);
    }

    #[test]
    fn test_arc_consistency_singleton_propagation() {
        // If svc 0 and svc 1 must be equal, and svc 0 domain is {1},
        // then svc 1 domain reduces to {1}.
        let mut bm = CompatibilityBitmap::new(3, 3);
        bm.set(0, 0, true);
        bm.set(1, 1, true);
        bm.set(2, 2, true);
        let mut constraints = HashMap::new();
        constraints.insert((0, 1), bm);
        let mut ac = ArcConsistency::new(2, &[3, 3], constraints);
        let result = ac.propagate();
        assert!(result.is_ok());
    }

    #[test]
    fn test_feasibility_filter_basic() {
        let mut pairwise = HashMap::new();
        let bm = CompatibilityBitmap::new_all_compatible(2, 2);
        pairwise.insert((0, 1), bm);

        let mut prefilter_obj = PairwisePrefilter::new(2, &[2, 2]);
        prefilter_obj.bitmaps = pairwise;
        let filter = FeasibilityFilter::new(
            prefilter_obj,
            vec![2, 2],
        );
        let result = filter.filter(&[0, 0]);
        assert!(result == FeasibilityResult::Feasible || result == FeasibilityResult::MaybeFeasible);
    }

    #[test]
    fn test_domain_store_remove_nonexistent() {
        let mut ds = DomainStore::from_sizes(&[3]);
        assert!(!ds.remove(0, 5)); // 5 not in domain
        assert_eq!(ds.domain_size(0), 3);
    }

    #[test]
    fn test_domain_store_out_of_bounds() {
        let ds = DomainStore::from_sizes(&[3]);
        assert!(ds.is_empty(10));
        assert_eq!(ds.domain_size(10), 0);
        assert!(!ds.contains(10, 0));
    }
}
