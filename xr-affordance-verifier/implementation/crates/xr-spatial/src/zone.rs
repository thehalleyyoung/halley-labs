//! Spatial zones for workspace classification.

use crate::interval::Interval;
use serde::{Deserialize, Serialize};
use xr_types::BoundingBox;

/// A spatial zone with classification metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Zone {
    pub name: String,
    pub bounds: BoundingBox,
    pub zone_type: ZoneType,
    pub comfort_score: f64,
}

/// Classification of zone types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ZoneType {
    /// Primary comfortable interaction zone.
    Comfort,
    /// Reachable but less comfortable.
    Extended,
    /// At the limit of reach.
    Marginal,
    /// Beyond reach for most users.
    Unreachable,
    /// Tracking volume boundary.
    TrackingBoundary,
}

impl Zone {
    /// Create a comfort zone centered at a position with given extents.
    pub fn comfort(name: impl Into<String>, center: [f64; 3], extents: [f64; 3]) -> Self {
        Self {
            name: name.into(),
            bounds: BoundingBox::from_center_extents(center, extents),
            zone_type: ZoneType::Comfort,
            comfort_score: 1.0,
        }
    }

    /// Create an extended reach zone.
    pub fn extended(name: impl Into<String>, bounds: BoundingBox) -> Self {
        Self {
            name: name.into(),
            bounds,
            zone_type: ZoneType::Extended,
            comfort_score: 0.6,
        }
    }

    /// Create standard ergonomic zones for a given shoulder height.
    pub fn standard_zones(shoulder_height: f64, arm_reach: f64) -> Vec<Zone> {
        let comfort_radius = arm_reach * 0.6;
        let extended_radius = arm_reach * 0.85;

        vec![
            Zone {
                name: "Primary Comfort Zone".into(),
                bounds: BoundingBox::from_center_extents(
                    [0.0, shoulder_height, -comfort_radius * 0.5],
                    [comfort_radius, comfort_radius * 0.7, comfort_radius],
                ),
                zone_type: ZoneType::Comfort,
                comfort_score: 1.0,
            },
            Zone {
                name: "Extended Reach Zone".into(),
                bounds: BoundingBox::from_center_extents(
                    [0.0, shoulder_height, -extended_radius * 0.5],
                    [extended_radius, extended_radius * 0.8, extended_radius],
                ),
                zone_type: ZoneType::Extended,
                comfort_score: 0.6,
            },
            Zone {
                name: "Marginal Reach Zone".into(),
                bounds: BoundingBox::from_center_extents(
                    [0.0, shoulder_height, -arm_reach * 0.5],
                    [arm_reach, arm_reach * 0.9, arm_reach],
                ),
                zone_type: ZoneType::Marginal,
                comfort_score: 0.3,
            },
        ]
    }

    /// Check if a point is inside this zone.
    pub fn contains_point(&self, point: [f64; 3]) -> bool {
        self.bounds.contains_point(&point)
    }

    /// Check if a bounding box overlaps this zone.
    pub fn overlaps(&self, bbox: &BoundingBox) -> bool {
        self.bounds.intersects(bbox)
    }
}

impl std::fmt::Display for Zone {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({:?}, comfort={:.1})", self.name, self.zone_type, self.comfort_score)
    }
}

/// Classification of zone accessibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ZoneClassification {
    Accessible,
    Inaccessible,
    Uncertain,
}

/// Partition of the parameter space into zones.
#[derive(Debug, Clone)]
pub struct ZonePartition {
    pub zones: Vec<(crate::interval::IntervalVector, ZoneClassification)>,
}

impl ZonePartition {
    pub fn new() -> Self { Self { zones: Vec::new() } }
    pub fn add_zone(&mut self, region: crate::interval::IntervalVector, class: ZoneClassification) {
        self.zones.push((region, class));
    }
    pub fn len(&self) -> usize { self.zones.len() }
    pub fn is_empty(&self) -> bool { self.zones.is_empty() }
    pub fn accessible_count(&self) -> usize {
        self.zones.iter().filter(|(_, c)| *c == ZoneClassification::Accessible).count()
    }
    pub fn uncertain_count(&self) -> usize {
        self.zones.iter().filter(|(_, c)| *c == ZoneClassification::Uncertain).count()
    }
}

impl Default for ZonePartition { fn default() -> Self { Self::new() } }

/// Tree-structured zone partition for hierarchical refinement.
#[derive(Debug, Clone)]
pub struct ZoneTree {
    pub root: ZoneTreeNode,
}

#[derive(Debug, Clone)]
pub struct ZoneTreeNode {
    pub region: crate::interval::IntervalVector,
    pub classification: Option<ZoneClassification>,
    pub children: Vec<ZoneTreeNode>,
}

impl ZoneTree {
    pub fn new(root_region: crate::interval::IntervalVector) -> Self {
        Self { root: ZoneTreeNode { region: root_region, classification: None, children: Vec::new() } }
    }
    pub fn leaf_count(&self) -> usize { self.root.leaf_count() }
    pub fn to_partition(&self) -> ZonePartition {
        let mut p = ZonePartition::new();
        self.root.collect_leaves(&mut p);
        p
    }
}

impl ZoneTreeNode {
    pub fn is_leaf(&self) -> bool { self.children.is_empty() }
    pub fn leaf_count(&self) -> usize {
        if self.is_leaf() { 1 } else { self.children.iter().map(|c| c.leaf_count()).sum() }
    }
    fn collect_leaves(&self, p: &mut ZonePartition) {
        if self.is_leaf() {
            if let Some(c) = self.classification { p.add_zone(self.region.clone(), c); }
        } else {
            for child in &self.children { child.collect_leaves(p); }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_zones() {
        let zones = Zone::standard_zones(1.4, 0.7);
        assert_eq!(zones.len(), 3);
        assert_eq!(zones[0].zone_type, ZoneType::Comfort);
        assert_eq!(zones[1].zone_type, ZoneType::Extended);
        assert_eq!(zones[2].zone_type, ZoneType::Marginal);
    }

    #[test]
    fn test_contains() {
        let z = Zone::comfort("test", [0.0, 1.0, 0.0], [0.5, 0.5, 0.5]);
        assert!(z.contains_point([0.0, 1.0, 0.0]));
        assert!(!z.contains_point([5.0, 5.0, 5.0]));
    }
}
