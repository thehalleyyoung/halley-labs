use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single node in the version lattice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionNode {
    pub id: String,
    pub parent: Option<String>,
    pub label: String,
    pub jurisdiction: String,
    pub version_number: u32,
    pub effective_date: Option<NaiveDate>,
}

impl VersionNode {
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            parent: None,
            label: label.into(),
            jurisdiction: String::new(),
            version_number: 1,
            effective_date: None,
        }
    }

    pub fn with_parent(mut self, parent: impl Into<String>) -> Self {
        self.parent = Some(parent.into());
        self
    }

    pub fn with_jurisdiction(mut self, j: impl Into<String>) -> Self {
        self.jurisdiction = j.into();
        self
    }

    pub fn with_version(mut self, v: u32) -> Self {
        self.version_number = v;
        self
    }

    pub fn with_effective_date(mut self, d: NaiveDate) -> Self {
        self.effective_date = Some(d);
        self
    }
}

/// Partial-order comparison between two version nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VersionOrdering {
    /// `a` is an ancestor of `b`.
    Ancestor,
    /// `b` is an ancestor of `a`.
    Descendant,
    /// Neither is an ancestor of the other.
    Incomparable,
    /// Same node.
    Equal,
}

/// A lattice of regulatory version nodes supporting join (LUB) and
/// meet (GLB) operations over a partial order defined by the parent
/// relation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionLattice {
    pub nodes: HashMap<String, VersionNode>,
}

impl VersionLattice {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    /// Insert a version node.
    pub fn add_node(&mut self, node: VersionNode) {
        self.nodes.insert(node.id.clone(), node);
    }

    /// Convenience alias used by the old API.
    pub fn add_version(&mut self, node: VersionNode) {
        self.add_node(node);
    }

    /// Retrieve a node by id.
    pub fn get(&self, id: &str) -> Option<&VersionNode> {
        self.nodes.get(id)
    }

    /// Return the latest version for a jurisdiction (highest version_number).
    pub fn get_latest(&self, jurisdiction: &str) -> Option<&VersionNode> {
        self.nodes
            .values()
            .filter(|n| n.jurisdiction == jurisdiction)
            .max_by_key(|n| n.version_number)
    }

    // ------------------------------------------------------------------
    // Ancestor queries
    // ------------------------------------------------------------------

    /// Collect the ordered chain of ancestors of `id` (parent, grandparent, …).
    pub fn ancestors(&self, id: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut cur = id.to_string();
        let mut visited = std::collections::HashSet::new();
        while let Some(node) = self.nodes.get(&cur) {
            if let Some(ref p) = node.parent {
                if !visited.insert(p.clone()) {
                    break; // guard against cycles
                }
                result.push(p.clone());
                cur = p.clone();
            } else {
                break;
            }
        }
        result
    }

    /// Return true if `ancestor_id` is a (proper) ancestor of `descendant_id`.
    pub fn is_ancestor_of(&self, ancestor_id: &str, descendant_id: &str) -> bool {
        if ancestor_id == descendant_id {
            return false;
        }
        self.ancestors(descendant_id).contains(&ancestor_id.to_string())
    }

    /// Compare two nodes in the partial order.
    pub fn compare(&self, a: &str, b: &str) -> VersionOrdering {
        if a == b {
            return VersionOrdering::Equal;
        }
        if self.is_ancestor_of(a, b) {
            return VersionOrdering::Ancestor;
        }
        if self.is_ancestor_of(b, a) {
            return VersionOrdering::Descendant;
        }
        VersionOrdering::Incomparable
    }

    // ------------------------------------------------------------------
    // Lattice operations
    // ------------------------------------------------------------------

    /// Meet (GLB): the deepest common ancestor of `a` and `b`.
    /// Returns `None` if the two nodes share no common ancestor and
    /// are not the same node.
    pub fn meet(&self, a: &str, b: &str) -> Option<String> {
        if a == b {
            return Some(a.to_string());
        }
        let anc_a = {
            let mut v = vec![a.to_string()];
            v.extend(self.ancestors(a));
            v
        };
        let anc_b: std::collections::HashSet<String> = {
            let mut s = std::collections::HashSet::new();
            s.insert(b.to_string());
            for x in self.ancestors(b) {
                s.insert(x);
            }
            s
        };
        // Walk `a`'s ancestor chain (root-ward) and return the first one
        // that also appears in `b`'s ancestor set.
        for id in &anc_a {
            if anc_b.contains(id) {
                return Some(id.clone());
            }
        }
        None
    }

    /// Join (LUB): the least version that is a descendant (or equal)
    /// of both `a` and `b`.
    ///
    /// In a tree-shaped history this only exists when one is an ancestor
    /// of the other (in which case the descendant is the join).
    /// For true lattice structure (merge commits), we look for the node
    /// with the smallest version_number that has both `a` and `b` as
    /// ancestors (or equals).
    pub fn join(&self, a: &str, b: &str) -> Option<String> {
        if a == b {
            return Some(a.to_string());
        }
        // Fast path: tree-shaped
        if self.is_ancestor_of(a, b) {
            return Some(b.to_string());
        }
        if self.is_ancestor_of(b, a) {
            return Some(a.to_string());
        }

        // General case: find the node with lowest version_number
        // that has *both* a and b as ancestors or equals.
        let mut candidates: Vec<&VersionNode> = self
            .nodes
            .values()
            .filter(|n| {
                let n_id = n.id.as_str();
                (n_id == a || self.is_ancestor_of(a, n_id))
                    && (n_id == b || self.is_ancestor_of(b, n_id))
            })
            .collect();
        candidates.sort_by_key(|n| n.version_number);
        candidates.first().map(|n| n.id.clone())
    }

    /// All node ids, sorted by version_number ascending.
    pub fn all_versions_sorted(&self) -> Vec<String> {
        let mut entries: Vec<_> = self.nodes.values().collect();
        entries.sort_by_key(|n| n.version_number);
        entries.iter().map(|n| n.id.clone()).collect()
    }

    /// Return the number of nodes in the lattice.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Return all root nodes (nodes with no parent).
    pub fn roots(&self) -> Vec<String> {
        self.nodes
            .values()
            .filter(|n| n.parent.is_none())
            .map(|n| n.id.clone())
            .collect()
    }

    /// Return all leaf nodes (no other node lists them as parent).
    pub fn leaves(&self) -> Vec<String> {
        let parents: std::collections::HashSet<&str> = self
            .nodes
            .values()
            .filter_map(|n| n.parent.as_deref())
            .collect();
        self.nodes
            .keys()
            .filter(|id| !parents.contains(id.as_str()))
            .cloned()
            .collect()
    }

    /// Depth of a node (distance to root).
    pub fn depth(&self, id: &str) -> usize {
        self.ancestors(id).len()
    }
}

impl Default for VersionLattice {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_lattice() -> VersionLattice {
        let mut lat = VersionLattice::new();
        lat.add_node(
            VersionNode::new("v1", "Initial")
                .with_jurisdiction("EU")
                .with_version(1),
        );
        lat.add_node(
            VersionNode::new("v2", "Amendment 1")
                .with_parent("v1")
                .with_jurisdiction("EU")
                .with_version(2),
        );
        lat.add_node(
            VersionNode::new("v3", "Amendment 2")
                .with_parent("v2")
                .with_jurisdiction("EU")
                .with_version(3),
        );
        lat.add_node(
            VersionNode::new("v2b", "Fork")
                .with_parent("v1")
                .with_jurisdiction("EU")
                .with_version(4),
        );
        lat
    }

    #[test]
    fn test_get_latest() {
        let lat = sample_lattice();
        let latest = lat.get_latest("EU").unwrap();
        assert_eq!(latest.id, "v2b"); // version_number 4
    }

    #[test]
    fn test_ancestors() {
        let lat = sample_lattice();
        let anc = lat.ancestors("v3");
        assert_eq!(anc, vec!["v2".to_string(), "v1".to_string()]);
    }

    #[test]
    fn test_is_ancestor_of() {
        let lat = sample_lattice();
        assert!(lat.is_ancestor_of("v1", "v3"));
        assert!(lat.is_ancestor_of("v2", "v3"));
        assert!(!lat.is_ancestor_of("v3", "v1"));
        assert!(!lat.is_ancestor_of("v2b", "v3"));
    }

    #[test]
    fn test_compare() {
        let lat = sample_lattice();
        assert_eq!(lat.compare("v1", "v3"), VersionOrdering::Ancestor);
        assert_eq!(lat.compare("v3", "v1"), VersionOrdering::Descendant);
        assert_eq!(lat.compare("v3", "v2b"), VersionOrdering::Incomparable);
        assert_eq!(lat.compare("v2", "v2"), VersionOrdering::Equal);
    }

    #[test]
    fn test_meet() {
        let lat = sample_lattice();
        assert_eq!(lat.meet("v3", "v2b"), Some("v1".into()));
        assert_eq!(lat.meet("v3", "v2"), Some("v2".into()));
        assert_eq!(lat.meet("v1", "v1"), Some("v1".into()));
    }

    #[test]
    fn test_join_tree() {
        let lat = sample_lattice();
        // v1 ancestor of v3 → join is v3
        assert_eq!(lat.join("v1", "v3"), Some("v3".into()));
        assert_eq!(lat.join("v2", "v3"), Some("v3".into()));
    }

    #[test]
    fn test_roots_and_leaves() {
        let lat = sample_lattice();
        let roots = lat.roots();
        assert_eq!(roots.len(), 1);
        assert!(roots.contains(&"v1".to_string()));

        let leaves = lat.leaves();
        assert_eq!(leaves.len(), 2);
        assert!(leaves.contains(&"v3".to_string()));
        assert!(leaves.contains(&"v2b".to_string()));
    }

    #[test]
    fn test_depth() {
        let lat = sample_lattice();
        assert_eq!(lat.depth("v1"), 0);
        assert_eq!(lat.depth("v2"), 1);
        assert_eq!(lat.depth("v3"), 2);
    }

    #[test]
    fn test_all_versions_sorted() {
        let lat = sample_lattice();
        let sorted = lat.all_versions_sorted();
        assert_eq!(sorted[0], "v1");
    }
}
