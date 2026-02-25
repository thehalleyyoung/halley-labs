use std::collections::HashMap;
use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// TrieError
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum TrieError {
    SerializationError(String),
    InvalidData(String),
    CapacityExceeded(String),
}

impl std::fmt::Display for TrieError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrieError::SerializationError(s) => write!(f, "serialization error: {}", s),
            TrieError::InvalidData(s) => write!(f, "invalid data: {}", s),
            TrieError::CapacityExceeded(s) => write!(f, "capacity exceeded: {}", s),
        }
    }
}

impl std::error::Error for TrieError {}

// ---------------------------------------------------------------------------
// TrieNode
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct TrieNode {
    pub children: HashMap<u8, Box<TrieNode>>,
    pub is_terminal: bool,
    pub count: usize,
    pub depth: usize,
    pub hash: Option<u64>,
}

impl TrieNode {
    pub fn new(depth: usize) -> Self {
        Self {
            children: HashMap::new(),
            is_terminal: false,
            count: 0,
            depth,
            hash: None,
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    pub fn has_child(&self, byte: u8) -> bool {
        self.children.contains_key(&byte)
    }
}

// ---------------------------------------------------------------------------
// TrieStats
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrieStats {
    pub node_count: usize,
    pub leaf_count: usize,
    pub internal_node_count: usize,
    pub average_depth: f64,
    pub max_depth: usize,
    pub average_fanout: f64,
    pub depth_distribution: Vec<usize>,
    pub memory_size_estimate: usize,
}

// ---------------------------------------------------------------------------
// CompactTrieNode (for Patricia trie compaction)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct CompactTrieNode {
    pub label: Vec<u8>,
    pub children: HashMap<u8, Box<CompactTrieNode>>,
    pub is_terminal: bool,
    pub count: usize,
}

impl CompactTrieNode {
    pub fn new(label: Vec<u8>) -> Self {
        Self { label, children: HashMap::new(), is_terminal: false, count: 0 }
    }
}

// ---------------------------------------------------------------------------
// TrieIterator (DFS, stack-based)
// ---------------------------------------------------------------------------

pub struct TrieIterator<'a> {
    // Stack holds (node_ref, current_path).
    stack: Vec<(&'a TrieNode, Vec<u8>)>,
}

impl<'a> TrieIterator<'a> {
    fn new(root: &'a TrieNode) -> Self {
        let mut stack = Vec::new();
        stack.push((root, Vec::new()));
        Self { stack }
    }
}

impl<'a> Iterator for TrieIterator<'a> {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, path)) = self.stack.pop() {
            // Push children in sorted order (reversed so smallest pops first).
            let mut child_keys: Vec<u8> = node.children.keys().cloned().collect();
            child_keys.sort_unstable();
            for &k in child_keys.iter().rev() {
                let child = &node.children[&k];
                let mut child_path = path.clone();
                child_path.push(k);
                self.stack.push((child, child_path));
            }
            if node.is_terminal {
                return Some(path);
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// NGramTrie
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct NGramTrie {
    pub root: TrieNode,
    pub size: usize,
    pub total_nodes: usize,
    pub max_depth: usize,
}

impl NGramTrie {
    pub fn new() -> Self {
        Self {
            root: TrieNode::new(0),
            size: 0,
            total_nodes: 1,
            max_depth: 0,
        }
    }

    pub fn from_ngrams(ngrams: &[Vec<u8>]) -> Self {
        let mut trie = Self::new();
        for ng in ngrams {
            trie.insert(ng);
        }
        trie
    }

    /// Build a trie from an NGramSet by converting each fingerprint to its
    /// little-endian byte representation.
    pub fn from_ngram_set(set: &super::ngram::NGramSet) -> Self {
        let mut trie = Self::new();
        for &fingerprint in set.fingerprints() {
            trie.insert(&fingerprint.to_le_bytes());
        }
        trie
    }

    pub fn insert(&mut self, key: &[u8]) {
        let mut node = &mut self.root;
        node.count += 1;

        for (i, &byte) in key.iter().enumerate() {
            let depth = i + 1;
            node = node.children
                .entry(byte)
                .or_insert_with(|| {
                    self.total_nodes += 1;
                    Box::new(TrieNode::new(depth))
                });
            node.count += 1;
        }

        if !node.is_terminal {
            node.is_terminal = true;
            self.size += 1;
            if key.len() > self.max_depth {
                self.max_depth = key.len();
            }
        }
    }

    pub fn contains(&self, key: &[u8]) -> bool {
        let mut node = &self.root;
        for &byte in key {
            match node.children.get(&byte) {
                Some(child) => node = child,
                None => return false,
            }
        }
        node.is_terminal
    }

    /// Return all keys that share the given prefix.
    pub fn prefix_search(&self, prefix: &[u8]) -> Vec<Vec<u8>> {
        let mut node = &self.root;
        for &byte in prefix {
            match node.children.get(&byte) {
                Some(child) => node = child,
                None => return Vec::new(),
            }
        }
        // Collect all keys under this subtree.
        let mut results = Vec::new();
        let mut stack: Vec<(&TrieNode, Vec<u8>)> = vec![(node, prefix.to_vec())];
        while let Some((n, path)) = stack.pop() {
            if n.is_terminal {
                results.push(path.clone());
            }
            let mut child_keys: Vec<u8> = n.children.keys().cloned().collect();
            child_keys.sort_unstable();
            for &k in child_keys.iter().rev() {
                let child = &n.children[&k];
                let mut cp = path.clone();
                cp.push(k);
                stack.push((child, cp));
            }
        }
        results
    }

    /// Count the number of keys with the given prefix.
    pub fn prefix_count(&self, prefix: &[u8]) -> usize {
        let mut node = &self.root;
        for &byte in prefix {
            match node.children.get(&byte) {
                Some(child) => node = child,
                None => return 0,
            }
        }
        // The count stored at this node equals the number of terminal
        // descendants (plus this node itself if terminal).
        self.count_terminals(node)
    }

    fn count_terminals(&self, node: &TrieNode) -> usize {
        let mut count = if node.is_terminal { 1 } else { 0 };
        for child in node.children.values() {
            count += self.count_terminals(child);
        }
        count
    }

    /// Remove a key from the trie. Returns true if the key existed.
    pub fn remove(&mut self, key: &[u8]) -> bool {
        if Self::remove_recursive(&mut self.root, key, 0) {
            self.size -= 1;
            true
        } else {
            false
        }
    }

    fn remove_recursive(node: &mut TrieNode, key: &[u8], depth: usize) -> bool {
        if depth == key.len() {
            if !node.is_terminal {
                return false;
            }
            node.is_terminal = false;
            node.count = node.count.saturating_sub(1);
            return true;
        }
        let byte = key[depth];
        let should_delete_child = {
            if let Some(child) = node.children.get_mut(&byte) {
                if Self::remove_recursive(child, key, depth + 1) {
                    child.count = child.count.saturating_sub(1);
                    node.count = node.count.saturating_sub(1);
                    child.is_leaf() && !child.is_terminal
                } else {
                    return false;
                }
            } else {
                return false;
            }
        };
        if should_delete_child {
            node.children.remove(&byte);
        }
        true
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn iter(&self) -> TrieIterator<'_> {
        TrieIterator::new(&self.root)
    }

    pub fn keys(&self) -> Vec<Vec<u8>> {
        self.iter().collect()
    }

    /// Length of the longest common prefix between two byte slices.
    pub fn common_prefix_length(&self, a: &[u8], b: &[u8]) -> usize {
        a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
    }

    // -------------------------------------------------------------------
    // Compaction (Patricia trie transformation)
    // -------------------------------------------------------------------

    /// Merge chains of single-child non-terminal nodes, producing a compact
    /// Patricia trie stored separately.
    pub fn compact(&self) -> CompactTrieNode {
        Self::compact_node(&self.root, Vec::new())
    }

    fn compact_node(node: &TrieNode, label: Vec<u8>) -> CompactTrieNode {
        // If the node has exactly one child and is not terminal, extend the label.
        if node.children.len() == 1 && !node.is_terminal {
            let (&byte, child) = node.children.iter().next().unwrap();
            let mut extended = label;
            extended.push(byte);
            return Self::compact_node(child, extended);
        }
        let mut compact = CompactTrieNode::new(label);
        compact.is_terminal = node.is_terminal;
        compact.count = node.count;
        for (&byte, child) in &node.children {
            let child_compact = Self::compact_node(child, vec![byte]);
            let first = if child_compact.label.is_empty() { byte } else { child_compact.label[0] };
            compact.children.insert(first, Box::new(child_compact));
        }
        compact
    }

    /// Ratio of nodes saved by compaction.
    pub fn compaction_ratio(&self) -> f64 {
        let compact = self.compact();
        let compact_count = Self::count_compact_nodes(&compact);
        if self.total_nodes == 0 {
            return 1.0;
        }
        1.0 - (compact_count as f64 / self.total_nodes as f64)
    }

    fn count_compact_nodes(node: &CompactTrieNode) -> usize {
        let mut count = 1;
        for child in node.children.values() {
            count += Self::count_compact_nodes(child);
        }
        count
    }

    // -------------------------------------------------------------------
    // Trie set operations
    // -------------------------------------------------------------------

    /// Intersection: return a trie containing only keys present in both.
    pub fn intersect(&self, other: &NGramTrie) -> NGramTrie {
        let mut result = NGramTrie::new();
        for key in self.iter() {
            if other.contains(&key) {
                result.insert(&key);
            }
        }
        result
    }

    pub fn intersection_cardinality(&self, other: &NGramTrie) -> usize {
        let mut count = 0;
        for key in self.iter() {
            if other.contains(&key) {
                count += 1;
            }
        }
        count
    }

    pub fn union(&self, other: &NGramTrie) -> NGramTrie {
        let mut result = NGramTrie::new();
        for key in self.iter() {
            result.insert(&key);
        }
        for key in other.iter() {
            result.insert(&key);
        }
        result
    }

    pub fn difference(&self, other: &NGramTrie) -> NGramTrie {
        let mut result = NGramTrie::new();
        for key in self.iter() {
            if !other.contains(&key) {
                result.insert(&key);
            }
        }
        result
    }

    // -------------------------------------------------------------------
    // Statistics
    // -------------------------------------------------------------------

    pub fn statistics(&self) -> TrieStats {
        let mut node_count = 0usize;
        let mut leaf_count = 0usize;
        let mut internal_count = 0usize;
        let mut depth_sum = 0usize;
        let mut max_d = 0usize;
        let mut fanout_sum = 0usize;
        let mut depth_dist: Vec<usize> = Vec::new();

        // BFS
        let mut queue: Vec<&TrieNode> = vec![&self.root];
        while let Some(node) = queue.pop() {
            node_count += 1;
            let d = node.depth;
            if d > max_d {
                max_d = d;
            }
            depth_sum += d;
            while depth_dist.len() <= d {
                depth_dist.push(0);
            }
            depth_dist[d] += 1;
            if node.is_leaf() {
                leaf_count += 1;
            } else {
                internal_count += 1;
                fanout_sum += node.child_count();
            }
            for child in node.children.values() {
                queue.push(child);
            }
        }

        let average_depth = if node_count > 0 { depth_sum as f64 / node_count as f64 } else { 0.0 };
        let average_fanout = if internal_count > 0 { fanout_sum as f64 / internal_count as f64 } else { 0.0 };
        // Rough memory estimate: per node ~ 64 bytes (HashMap overhead + fields).
        let memory_size_estimate = node_count * 64;

        TrieStats {
            node_count,
            leaf_count,
            internal_node_count: internal_count,
            average_depth,
            max_depth: max_d,
            average_fanout,
            depth_distribution: depth_dist,
            memory_size_estimate,
        }
    }

    pub fn depth_histogram(&self) -> Vec<usize> {
        self.statistics().depth_distribution
    }

    /// Distribution of child counts across all nodes.
    pub fn fanout_histogram(&self) -> Vec<usize> {
        let mut hist: Vec<usize> = Vec::new();
        let mut stack: Vec<&TrieNode> = vec![&self.root];
        while let Some(node) = stack.pop() {
            let fc = node.child_count();
            while hist.len() <= fc {
                hist.push(0);
            }
            hist[fc] += 1;
            for child in node.children.values() {
                stack.push(child);
            }
        }
        hist
    }

    /// Fraction of total key bytes saved by prefix sharing compared to storing
    /// every key independently.
    pub fn prefix_sharing_ratio(&self) -> f64 {
        let keys = self.keys();
        if keys.is_empty() {
            return 0.0;
        }
        let total_key_bytes: usize = keys.iter().map(|k| k.len()).sum();
        // total_nodes - 1 == edges == total bytes stored in the trie
        let trie_bytes = if self.total_nodes > 0 { self.total_nodes - 1 } else { 0 };
        if total_key_bytes == 0 {
            return 0.0;
        }
        1.0 - (trie_bytes as f64 / total_key_bytes as f64)
    }

    // -------------------------------------------------------------------
    // Serialization
    // -------------------------------------------------------------------

    /// Compact binary serialization.
    ///
    /// Format per node (recursive):
    ///   [1 byte flags: bit0 = is_terminal]
    ///   [2 bytes child_count (little-endian u16)]
    ///   for each child (sorted by key):
    ///     [1 byte key]
    ///     [recursive child serialization]
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        Self::serialize_node(&self.root, &mut buf);
        buf
    }

    fn serialize_node(node: &TrieNode, buf: &mut Vec<u8>) {
        let flags: u8 = if node.is_terminal { 1 } else { 0 };
        buf.push(flags);
        let child_count = node.children.len() as u16;
        buf.extend_from_slice(&child_count.to_le_bytes());
        let mut child_keys: Vec<u8> = node.children.keys().cloned().collect();
        child_keys.sort_unstable();
        for k in child_keys {
            buf.push(k);
            Self::serialize_node(&node.children[&k], buf);
        }
    }

    pub fn deserialize(bytes: &[u8]) -> Result<Self, TrieError> {
        let mut pos = 0;
        let root = Self::deserialize_node(bytes, &mut pos, 0)?;
        // Recount size and total_nodes.
        let mut trie = NGramTrie { root, size: 0, total_nodes: 0, max_depth: 0 };
        trie.recount();
        Ok(trie)
    }

    fn deserialize_node(bytes: &[u8], pos: &mut usize, depth: usize) -> Result<TrieNode, TrieError> {
        if *pos >= bytes.len() {
            return Err(TrieError::InvalidData("unexpected end of data".into()));
        }
        let flags = bytes[*pos];
        *pos += 1;

        if *pos + 1 >= bytes.len() {
            return Err(TrieError::InvalidData("missing child count".into()));
        }
        let child_count = u16::from_le_bytes([bytes[*pos], bytes[*pos + 1]]) as usize;
        *pos += 2;

        let mut node = TrieNode::new(depth);
        node.is_terminal = flags & 1 != 0;

        for _ in 0..child_count {
            if *pos >= bytes.len() {
                return Err(TrieError::InvalidData("missing child key".into()));
            }
            let key = bytes[*pos];
            *pos += 1;
            let child = Self::deserialize_node(bytes, pos, depth + 1)?;
            node.children.insert(key, Box::new(child));
        }
        Ok(node)
    }

    fn recount(&mut self) {
        let (size, total, max_d) = Self::recount_node(&self.root, 0);
        self.size = size;
        self.total_nodes = total;
        self.max_depth = max_d;
    }

    fn recount_node(node: &TrieNode, depth: usize) -> (usize, usize, usize) {
        let mut size = if node.is_terminal { 1 } else { 0 };
        let mut total = 1;
        let mut max_d = depth;
        for child in node.children.values() {
            let (s, t, d) = Self::recount_node(child, depth + 1);
            size += s;
            total += t;
            if d > max_d {
                max_d = d;
            }
        }
        (size, total, max_d)
    }

    /// Generate a Graphviz DOT representation for debugging / visualisation.
    pub fn to_dot(&self) -> String {
        let mut out = String::from("digraph Trie {\n  node [shape=circle];\n");
        let mut id_counter = 0usize;
        Self::dot_node(&self.root, &mut out, &mut id_counter, None, None);
        out.push_str("}\n");
        out
    }

    fn dot_node(
        node: &TrieNode,
        out: &mut String,
        id_counter: &mut usize,
        parent_id: Option<usize>,
        edge_label: Option<u8>,
    ) {
        let my_id = *id_counter;
        *id_counter += 1;
        let shape = if node.is_terminal { "doublecircle" } else { "circle" };
        out.push_str(&format!("  n{} [shape={}, label=\"{}\"];\n", my_id, shape, my_id));
        if let (Some(pid), Some(label)) = (parent_id, edge_label) {
            let ch = if label.is_ascii_graphic() { label as char } else { '.' };
            out.push_str(&format!("  n{} -> n{} [label=\"{}\"];\n", pid, my_id, ch));
        }
        let mut child_keys: Vec<u8> = node.children.keys().cloned().collect();
        child_keys.sort_unstable();
        for k in child_keys {
            Self::dot_node(&node.children[&k], out, id_counter, Some(my_id), Some(k));
        }
    }

    // -------------------------------------------------------------------
    // Communication complexity estimation
    // -------------------------------------------------------------------

    /// Estimate bytes transferred for a PSI protocol between two tries.
    pub fn estimated_communication_bytes(&self, other: &NGramTrie) -> usize {
        // Rough model: each party sends OPRF-encoded fingerprints.
        // 32 bytes per OPRF output, plus protocol overhead.
        let n = self.size;
        let m = other.size;
        let overhead_per_element = 32; // OPRF output size
        let protocol_overhead = 256; // setup messages
        (n + m) * overhead_per_element + protocol_overhead
    }

    /// Estimate number of communication rounds.
    pub fn estimated_psi_rounds(&self, _other: &NGramTrie) -> usize {
        // OPRF-based PSI: 2 rounds (blind + response)
        // Trie optimisation may add 1 round for structure exchange
        3
    }

    /// Hash of the trie structure (for protocol integrity checks).
    pub fn trie_structure_hash(&self) -> [u8; 32] {
        let serialized = self.serialize();
        let h = blake3::hash(&serialized);
        *h.as_bytes()
    }
}

impl Default for NGramTrie {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// BatchTrie
// ---------------------------------------------------------------------------

/// Handle multiple tries for multi-document PSI.
pub struct BatchTrie {
    pub tries: Vec<NGramTrie>,
}

impl BatchTrie {
    pub fn new() -> Self {
        Self { tries: Vec::new() }
    }

    /// Add a document by building an NGramTrie from its text using the given
    /// n-gram config.
    pub fn add_document(&mut self, text: &str, config: super::ngram::NGramConfig) {
        let set = super::ngram::NGramSet::from_text(text, config);
        let trie = NGramTrie::from_ngram_set(&set);
        self.tries.push(trie);
    }

    /// Build a single trie that is the union of all document tries.
    pub fn combined_trie(&self) -> NGramTrie {
        let mut combined = NGramTrie::new();
        for trie in &self.tries {
            for key in trie.iter() {
                combined.insert(&key);
            }
        }
        combined
    }

    /// Compute pairwise intersection cardinalities.
    pub fn pairwise_intersections(&self) -> Vec<((usize, usize), usize)> {
        let mut results = Vec::new();
        for i in 0..self.tries.len() {
            for j in (i + 1)..self.tries.len() {
                let card = self.tries[i].intersection_cardinality(&self.tries[j]);
                results.push(((i, j), card));
            }
        }
        results
    }
}

impl Default for BatchTrie {
    fn default() -> Self {
        Self::new()
    }
}


// ---------------------------------------------------------------------------
// CompactTrieNode — additional methods
// ---------------------------------------------------------------------------

impl CompactTrieNode {
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    /// Total label length including this node's label.
    pub fn label_len(&self) -> usize {
        self.label.len()
    }

    /// Recursively count nodes in this subtree.
    pub fn subtree_size(&self) -> usize {
        let mut count = 1;
        for child in self.children.values() {
            count += child.subtree_size();
        }
        count
    }

    /// Recursively count terminal nodes in this subtree.
    pub fn terminal_count(&self) -> usize {
        let mut count = if self.is_terminal { 1 } else { 0 };
        for child in self.children.values() {
            count += child.terminal_count();
        }
        count
    }
}

// ---------------------------------------------------------------------------
// CompactTrieIterator — DFS iterator over a CompactTrieNode tree
// ---------------------------------------------------------------------------

pub struct CompactTrieIterator<'a> {
    stack: Vec<(&'a CompactTrieNode, Vec<u8>)>,
}

impl<'a> CompactTrieIterator<'a> {
    pub fn new(root: &'a CompactTrieNode) -> Self {
        let mut stack = Vec::new();
        stack.push((root, root.label.clone()));
        Self { stack }
    }
}

impl<'a> Iterator for CompactTrieIterator<'a> {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, path)) = self.stack.pop() {
            let mut child_keys: Vec<u8> = node.children.keys().cloned().collect();
            child_keys.sort_unstable();
            for &k in child_keys.iter().rev() {
                let child = &node.children[&k];
                let mut child_path = path.clone();
                child_path.extend_from_slice(&child.label);
                self.stack.push((child, child_path));
            }
            if node.is_terminal {
                return Some(path);
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// CompactTrie — compressed (Patricia) trie wrapper
// ---------------------------------------------------------------------------

/// A compressed trie built by merging chains of single-child non-terminal
/// nodes from a standard `NGramTrie`.
pub struct CompactTrie {
    pub root: CompactTrieNode,
    pub size: usize,
}

impl CompactTrie {
    /// Build a `CompactTrie` from an existing `NGramTrie`.
    pub fn from_trie(trie: &NGramTrie) -> Self {
        let root = trie.compact();
        let size = root.terminal_count();
        Self { root, size }
    }

    /// Check whether a key exists in the compact trie.
    pub fn contains(&self, key: &[u8]) -> bool {
        Self::contains_in(&self.root, key, 0)
    }

    fn contains_in(node: &CompactTrieNode, key: &[u8], offset: usize) -> bool {
        // Skip the label portion of this node
        let label = &node.label;
        if !label.is_empty() {
            // The root might have an empty label; others have their edge label
            // Verify the label matches the key at the current offset
            let start = if offset == 0 { 0 } else { offset };
            if start + label.len() > key.len() {
                // Label is longer than remaining key - could still match if key ends here
                return false;
            }
            // For the root node the label may be empty, skip check
        }

        // Use a simpler iterative approach
        Self::lookup_iterative(&node, key)
    }

    fn lookup_iterative(root: &CompactTrieNode, key: &[u8]) -> bool {
        let mut node = root;
        let mut pos = 0usize;

        // Skip root label
        if !node.label.is_empty() {
            for &b in &node.label {
                if pos >= key.len() || key[pos] != b {
                    return false;
                }
                pos += 1;
            }
        }

        while pos < key.len() {
            let byte = key[pos];
            match node.children.get(&byte) {
                Some(child) => {
                    // Verify child label matches
                    for &b in &child.label {
                        if pos >= key.len() || key[pos] != b {
                            return false;
                        }
                        pos += 1;
                    }
                    node = child;
                }
                None => return false,
            }
        }
        node.is_terminal
    }

    /// Insert a key into the compact trie.
    pub fn insert(&mut self, key: &[u8]) {
        if Self::insert_into(&mut self.root, key, 0) {
            self.size += 1;
        }
    }

    fn insert_into(node: &mut CompactTrieNode, key: &[u8], pos: usize) -> bool {
        if pos >= key.len() {
            if node.is_terminal {
                return false;
            }
            node.is_terminal = true;
            node.count += 1;
            return true;
        }

        let byte = key[pos];
        if node.children.contains_key(&byte) {
            let child = node.children.get_mut(&byte).unwrap();
            // Check how much of the child's label matches
            let label = child.label.clone();
            let mut match_len = 0;
            let mut kpos = pos;
            for &lb in &label {
                if kpos < key.len() && key[kpos] == lb {
                    match_len += 1;
                    kpos += 1;
                } else {
                    break;
                }
            }

            if match_len == label.len() {
                // Full match of label; recurse
                return Self::insert_into(child, key, kpos);
            } else {
                // Partial match; need to split
                let mut new_child = CompactTrieNode::new(label[..match_len].to_vec());
                let mut old_remainder = child.as_ref().clone();
                old_remainder.label = label[match_len..].to_vec();
                let old_first = old_remainder.label[0];
                new_child.children.insert(old_first, Box::new(old_remainder));

                if kpos < key.len() {
                    let remaining_key = key[kpos..].to_vec();
                    let first_byte = remaining_key[0];
                    let mut leaf = CompactTrieNode::new(remaining_key);
                    leaf.is_terminal = true;
                    leaf.count = 1;
                    new_child.children.insert(first_byte, Box::new(leaf));
                } else {
                    new_child.is_terminal = true;
                    new_child.count += 1;
                }

                node.children.insert(byte, Box::new(new_child));
                return true;
            }
        } else {
            // No matching child; create a new leaf
            let remaining = key[pos..].to_vec();
            let first_byte = remaining[0];
            let mut leaf = CompactTrieNode::new(remaining);
            leaf.is_terminal = true;
            leaf.count = 1;
            node.children.insert(first_byte, Box::new(leaf));
            return true;
        }
    }

    /// Total number of nodes in the compact trie.
    pub fn node_count(&self) -> usize {
        self.root.subtree_size()
    }

    /// Compression ratio compared to the original trie: 1 - (compact_nodes / original_nodes).
    pub fn compression_ratio(&self, original: &NGramTrie) -> f64 {
        if original.total_nodes == 0 {
            return 0.0;
        }
        1.0 - (self.node_count() as f64 / original.total_nodes as f64)
    }

    /// Iterate over all keys in the compact trie.
    pub fn iter(&self) -> CompactTrieIterator<'_> {
        CompactTrieIterator::new(&self.root)
    }

    /// Estimate memory usage in bytes.
    pub fn memory_estimate(&self) -> usize {
        Self::estimate_node_memory(&self.root)
    }

    fn estimate_node_memory(node: &CompactTrieNode) -> usize {
        // Base: struct overhead + label vec + hashmap overhead
        let base = std::mem::size_of::<CompactTrieNode>() + node.label.len();
        let children_overhead = node.children.len() * (std::mem::size_of::<u8>() + std::mem::size_of::<Box<CompactTrieNode>>() + 32);
        let mut total = base + children_overhead;
        for child in node.children.values() {
            total += Self::estimate_node_memory(child);
        }
        total
    }

    /// Number of terminal (key-bearing) nodes.
    pub fn terminal_count(&self) -> usize {
        self.root.terminal_count()
    }

    /// Collect all keys.
    pub fn keys(&self) -> Vec<Vec<u8>> {
        self.iter().collect()
    }

    /// Check if the trie is empty.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Number of stored keys.
    pub fn len(&self) -> usize {
        self.size
    }
}

// ---------------------------------------------------------------------------
// TrieStatistics — richer statistics with summary / JSON output
// ---------------------------------------------------------------------------

/// Extended trie statistics with pretty-printing support.
pub struct TrieStatistics {
    pub node_count: usize,
    pub leaf_count: usize,
    pub internal_node_count: usize,
    pub average_depth: f64,
    pub max_depth: usize,
    pub average_fanout: f64,
    pub depth_distribution: Vec<usize>,
    pub memory_size_estimate: usize,
    pub key_count: usize,
    pub total_key_bytes: usize,
}

impl TrieStatistics {
    /// Compute statistics for a given trie.
    pub fn from_trie(trie: &NGramTrie) -> Self {
        let stats = trie.statistics();
        let keys = trie.keys();
        let total_key_bytes: usize = keys.iter().map(|k| k.len()).sum();
        Self {
            node_count: stats.node_count,
            leaf_count: stats.leaf_count,
            internal_node_count: stats.internal_node_count,
            average_depth: stats.average_depth,
            max_depth: stats.max_depth,
            average_fanout: stats.average_fanout,
            depth_distribution: stats.depth_distribution,
            memory_size_estimate: stats.memory_size_estimate,
            key_count: trie.len(),
            total_key_bytes,
        }
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "TrieStatistics:\n\
             \x20 keys: {}\n\
             \x20 nodes: {} (internal: {}, leaves: {})\n\
             \x20 avg depth: {:.2}, max depth: {}\n\
             \x20 avg fanout: {:.2}\n\
             \x20 total key bytes: {}\n\
             \x20 est. memory: {} bytes",
            self.key_count,
            self.node_count,
            self.internal_node_count,
            self.leaf_count,
            self.average_depth,
            self.max_depth,
            self.average_fanout,
            self.total_key_bytes,
            self.memory_size_estimate,
        )
    }

    /// JSON representation.
    pub fn to_json(&self) -> String {
        let depth_dist_str: String = self
            .depth_distribution
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(",");
        format!(
            concat!(
                "{{",
                "\"node_count\":{},",
                "\"leaf_count\":{},",
                "\"internal_node_count\":{},",
                "\"average_depth\":{:.4},",
                "\"max_depth\":{},",
                "\"average_fanout\":{:.4},",
                "\"depth_distribution\":[{}],",
                "\"memory_size_estimate\":{},",
                "\"key_count\":{},",
                "\"total_key_bytes\":{}",
                "}}"
            ),
            self.node_count,
            self.leaf_count,
            self.internal_node_count,
            self.average_depth,
            self.max_depth,
            self.average_fanout,
            depth_dist_str,
            self.memory_size_estimate,
            self.key_count,
            self.total_key_bytes,
        )
    }

    /// Branching factor distribution as (fanout, count) pairs.
    pub fn fanout_distribution(&self) -> Vec<(usize, usize)> {
        let mut result = Vec::new();
        for (fanout, &count) in self.depth_distribution.iter().enumerate() {
            if count > 0 {
                result.push((fanout, count));
            }
        }
        result
    }

    /// Average key length.
    pub fn average_key_length(&self) -> f64 {
        if self.key_count == 0 {
            0.0
        } else {
            self.total_key_bytes as f64 / self.key_count as f64
        }
    }

    /// Density: ratio of terminal nodes to total nodes.
    pub fn density(&self) -> f64 {
        if self.node_count == 0 {
            0.0
        } else {
            self.key_count as f64 / self.node_count as f64
        }
    }
}

// ---------------------------------------------------------------------------
// TrieDifference — structural comparison of two tries
// ---------------------------------------------------------------------------

/// Utilities for computing the structural difference between two tries.
pub struct TrieDifference;

impl TrieDifference {
    /// Keys present in `new_trie` but not in `old_trie`.
    pub fn added_keys(old: &NGramTrie, new: &NGramTrie) -> Vec<Vec<u8>> {
        let mut result = Vec::new();
        for key in new.iter() {
            if !old.contains(&key) {
                result.push(key);
            }
        }
        result
    }

    /// Keys present in `old_trie` but not in `new_trie`.
    pub fn removed_keys(old: &NGramTrie, new: &NGramTrie) -> Vec<Vec<u8>> {
        let mut result = Vec::new();
        for key in old.iter() {
            if !new.contains(&key) {
                result.push(key);
            }
        }
        result
    }

    /// Keys that are in exactly one of the two tries (symmetric difference).
    pub fn symmetric_difference(a: &NGramTrie, b: &NGramTrie) -> Vec<Vec<u8>> {
        let mut result = Vec::new();
        for key in a.iter() {
            if !b.contains(&key) {
                result.push(key);
            }
        }
        for key in b.iter() {
            if !a.contains(&key) {
                result.push(key);
            }
        }
        result
    }

    /// Jaccard similarity between the key sets of two tries.
    pub fn structural_similarity(a: &NGramTrie, b: &NGramTrie) -> f64 {
        let inter = a.intersection_cardinality(b);
        let union_size = a.len() + b.len() - inter;
        if union_size == 0 {
            0.0
        } else {
            inter as f64 / union_size as f64
        }
    }

    /// Number of keys added.
    pub fn added_count(old: &NGramTrie, new: &NGramTrie) -> usize {
        Self::added_keys(old, new).len()
    }

    /// Number of keys removed.
    pub fn removed_count(old: &NGramTrie, new: &NGramTrie) -> usize {
        Self::removed_keys(old, new).len()
    }

    /// Compute a difference summary.
    pub fn diff_summary(old: &NGramTrie, new: &NGramTrie) -> String {
        let added = Self::added_count(old, new);
        let removed = Self::removed_count(old, new);
        let sim = Self::structural_similarity(old, new);
        format!(
            "TrieDiff: +{} added, -{} removed, similarity={:.4}",
            added, removed, sim
        )
    }

    /// Edit distance between two tries (number of insertions + deletions needed).
    pub fn edit_distance(a: &NGramTrie, b: &NGramTrie) -> usize {
        let mut diff = 0;
        for key in a.iter() {
            if !b.contains(&key) {
                diff += 1;
            }
        }
        for key in b.iter() {
            if !a.contains(&key) {
                diff += 1;
            }
        }
        diff
    }

    /// Containment of a in b: |A ∩ B| / |A|.
    pub fn containment(a: &NGramTrie, b: &NGramTrie) -> f64 {
        if a.is_empty() {
            return 0.0;
        }
        a.intersection_cardinality(b) as f64 / a.len() as f64
    }
}

// ---------------------------------------------------------------------------
// NamedBatchTrie — batch of tries with names
// ---------------------------------------------------------------------------

/// An enhanced batch trie that associates a name with each trie, enabling
/// richer pairwise comparisons and unique-key discovery.
pub struct NamedBatchTrie {
    pub tries: Vec<NGramTrie>,
    pub names: Vec<String>,
}

impl NamedBatchTrie {
    pub fn new() -> Self {
        Self {
            tries: Vec::new(),
            names: Vec::new(),
        }
    }

    /// Add a trie with a name.
    pub fn add(&mut self, name: &str, trie: NGramTrie) {
        self.names.push(name.to_string());
        self.tries.push(trie);
    }

    /// Build a combined trie that is the union of all tries.
    pub fn combined_trie(&self) -> NGramTrie {
        let mut combined = NGramTrie::new();
        for trie in &self.tries {
            for key in trie.iter() {
                combined.insert(&key);
            }
        }
        combined
    }

    /// Pairwise intersection cardinalities with names.
    pub fn pairwise_intersections(&self) -> Vec<((String, String), usize)> {
        let mut results = Vec::new();
        for i in 0..self.tries.len() {
            for j in (i + 1)..self.tries.len() {
                let card = self.tries[i].intersection_cardinality(&self.tries[j]);
                results.push(((self.names[i].clone(), self.names[j].clone()), card));
            }
        }
        results
    }

    /// Pairwise Jaccard similarities with names.
    pub fn pairwise_jaccard(&self) -> Vec<((String, String), f64)> {
        let mut results = Vec::new();
        for i in 0..self.tries.len() {
            for j in (i + 1)..self.tries.len() {
                let inter = self.tries[i].intersection_cardinality(&self.tries[j]);
                let union = self.tries[i].len() + self.tries[j].len() - inter;
                let jaccard = if union == 0 { 0.0 } else { inter as f64 / union as f64 };
                results.push(((self.names[i].clone(), self.names[j].clone()), jaccard));
            }
        }
        results
    }

    /// Find keys that exist only in the trie at the given index.
    pub fn find_unique_keys(&self, index: usize) -> Vec<Vec<u8>> {
        if index >= self.tries.len() {
            return Vec::new();
        }
        let mut result = Vec::new();
        for key in self.tries[index].iter() {
            let mut unique = true;
            for (i, trie) in self.tries.iter().enumerate() {
                if i != index && trie.contains(&key) {
                    unique = false;
                    break;
                }
            }
            if unique {
                result.push(key);
            }
        }
        result
    }

    /// Number of tries in the batch.
    pub fn len(&self) -> usize {
        self.tries.len()
    }

    /// Whether the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.tries.is_empty()
    }

    /// Get the name of a trie by index.
    pub fn name_at(&self, index: usize) -> Option<&str> {
        self.names.get(index).map(|s| s.as_str())
    }

    /// Total number of keys across all tries (with duplicates).
    pub fn total_keys(&self) -> usize {
        self.tries.iter().map(|t| t.len()).sum()
    }

    /// Number of keys in the combined (union) trie.
    pub fn unique_keys_count(&self) -> usize {
        self.combined_trie().len()
    }

    /// Overlap matrix as a 2D vector of Jaccard similarities.
    pub fn overlap_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.tries.len();
        let mut matrix = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            matrix[i][i] = 1.0;
            for j in (i + 1)..n {
                let inter = self.tries[i].intersection_cardinality(&self.tries[j]);
                let union = self.tries[i].len() + self.tries[j].len() - inter;
                let jaccard = if union == 0 { 0.0 } else { inter as f64 / union as f64 };
                matrix[i][j] = jaccard;
                matrix[j][i] = jaccard;
            }
        }
        matrix
    }

    /// Summary of the batch.
    pub fn summary(&self) -> String {
        let total = self.total_keys();
        let unique = self.unique_keys_count();
        format!(
            "NamedBatchTrie: {} tries, {} total keys, {} unique keys",
            self.tries.len(),
            total,
            unique,
        )
    }
}

impl Default for NamedBatchTrie {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// TrieSerializer — serialization utilities
// ---------------------------------------------------------------------------

/// Extended serialization utilities for tries.
pub struct TrieSerializer;

impl TrieSerializer {
    /// Serialize a trie to compact binary format (delegates to NGramTrie::serialize).
    pub fn serialize_binary(trie: &NGramTrie) -> Vec<u8> {
        trie.serialize()
    }

    /// Deserialize a trie from compact binary format.
    pub fn deserialize_binary(bytes: &[u8]) -> Result<NGramTrie, TrieError> {
        NGramTrie::deserialize(bytes)
    }

    /// Serialize a trie to a JSON string representation.
    pub fn serialize_json(trie: &NGramTrie) -> String {
        let mut out = String::from("{");
        Self::json_node(&trie.root, &mut out);
        out.push('}');
        out
    }

    fn json_node(node: &TrieNode, out: &mut String) {
        out.push_str(&format!(
            "\"terminal\":{},\"count\":{},\"children\":{{",
            node.is_terminal, node.count
        ));
        let mut child_keys: Vec<u8> = node.children.keys().cloned().collect();
        child_keys.sort_unstable();
        for (i, &k) in child_keys.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let ch = if k.is_ascii_graphic() {
                format!("{}", k as char)
            } else {
                format!("0x{:02x}", k)
            };
            out.push_str(&format!("\"{}\":{{", ch));
            Self::json_node(&node.children[&k], out);
            out.push('}');
        }
        out.push('}');
    }

    /// Estimate the serialized binary size without actually serializing.
    pub fn estimate_serialized_size(trie: &NGramTrie) -> usize {
        Self::estimate_node_size(&trie.root)
    }

    fn estimate_node_size(node: &TrieNode) -> usize {
        // 1 byte flags + 2 bytes child count + 1 byte per child key + recursive
        let mut size = 3;
        for child in node.children.values() {
            size += 1 + Self::estimate_node_size(child);
        }
        size
    }

    /// Verify that serialization round-trips correctly.
    pub fn verify_roundtrip(trie: &NGramTrie) -> bool {
        let bytes = Self::serialize_binary(trie);
        match Self::deserialize_binary(&bytes) {
            Ok(recovered) => {
                if recovered.len() != trie.len() {
                    return false;
                }
                for key in trie.iter() {
                    if !recovered.contains(&key) {
                        return false;
                    }
                }
                true
            }
            Err(_) => false,
        }
    }

    /// Serialize to a hex string (useful for debugging).
    pub fn serialize_hex(trie: &NGramTrie) -> String {
        let bytes = Self::serialize_binary(trie);
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }

    /// Deserialize from a hex string.
    pub fn deserialize_hex(hex: &str) -> Result<NGramTrie, TrieError> {
        let mut bytes = Vec::new();
        let chars: Vec<char> = hex.chars().collect();
        if chars.len() % 2 != 0 {
            return Err(TrieError::InvalidData("odd-length hex string".into()));
        }
        for i in (0..chars.len()).step_by(2) {
            let byte = u8::from_str_radix(&hex[i..i + 2], 16)
                .map_err(|e| TrieError::InvalidData(format!("invalid hex: {}", e)))?;
            bytes.push(byte);
        }
        Self::deserialize_binary(&bytes)
    }

    /// Serialize only the keys (newline-separated, hex-encoded).
    pub fn serialize_keys(trie: &NGramTrie) -> String {
        trie.keys()
            .iter()
            .map(|k| k.iter().map(|b| format!("{:02x}", b)).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Compute a checksum of the serialized trie (Blake3 hash).
    pub fn checksum(trie: &NGramTrie) -> [u8; 32] {
        let bytes = Self::serialize_binary(trie);
        *blake3::hash(&bytes).as_bytes()
    }
}

// ---------------------------------------------------------------------------
// TrieVisualizer — trie visualization utilities
// ---------------------------------------------------------------------------

/// Utilities for visualizing tries in different formats.
pub struct TrieVisualizer;

impl TrieVisualizer {
    /// Generate Graphviz DOT representation (delegates to NGramTrie::to_dot).
    pub fn to_dot(trie: &NGramTrie) -> String {
        trie.to_dot()
    }

    /// Generate an ASCII tree representation up to `max_depth`.
    pub fn to_ascii(trie: &NGramTrie, max_depth: usize) -> String {
        let mut out = String::new();
        out.push_str("(root)\n");
        Self::ascii_node(&trie.root, &mut out, "", true, 0, max_depth);
        out
    }

    fn ascii_node(
        node: &TrieNode,
        out: &mut String,
        prefix: &str,
        _is_last: bool,
        depth: usize,
        max_depth: usize,
    ) {
        if depth > max_depth {
            return;
        }
        let mut child_keys: Vec<u8> = node.children.keys().cloned().collect();
        child_keys.sort_unstable();
        let count = child_keys.len();
        for (i, &k) in child_keys.iter().enumerate() {
            let child = &node.children[&k];
            let is_child_last = i == count - 1;
            let connector = if is_child_last { "└── " } else { "├── " };
            let ch = if k.is_ascii_graphic() {
                format!("{}", k as char)
            } else {
                format!("0x{:02x}", k)
            };
            let terminal_marker = if child.is_terminal { " *" } else { "" };
            out.push_str(&format!("{}{}{}{}\n", prefix, connector, ch, terminal_marker));
            let new_prefix = format!(
                "{}{}",
                prefix,
                if is_child_last { "    " } else { "│   " }
            );
            Self::ascii_node(child, out, &new_prefix, is_child_last, depth + 1, max_depth);
        }
    }

    /// Generate a JSON tree representation.
    pub fn to_json_tree(trie: &NGramTrie) -> String {
        let mut out = String::new();
        Self::json_tree_node(&trie.root, &mut out);
        out
    }

    fn json_tree_node(node: &TrieNode, out: &mut String) {
        out.push('{');
        out.push_str(&format!(
            "\"terminal\":{},\"count\":{}",
            node.is_terminal, node.count
        ));
        if !node.children.is_empty() {
            out.push_str(",\"children\":{");
            let mut child_keys: Vec<u8> = node.children.keys().cloned().collect();
            child_keys.sort_unstable();
            for (i, &k) in child_keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                let ch = if k.is_ascii_graphic() {
                    format!("{}", k as char)
                } else {
                    format!("0x{:02x}", k)
                };
                out.push_str(&format!("\"{}\":", ch));
                Self::json_tree_node(&node.children[&k], out);
            }
            out.push('}');
        }
        out.push('}');
    }

    /// Render a compact one-line representation showing key count per depth level.
    pub fn depth_bar_chart(trie: &NGramTrie) -> String {
        let stats = trie.statistics();
        let mut lines = Vec::new();
        for (depth, &count) in stats.depth_distribution.iter().enumerate() {
            let bar: String = std::iter::repeat('#').take(count.min(50)).collect();
            lines.push(format!("d{:>3}: {} ({})", depth, bar, count));
        }
        lines.join("\n")
    }

    /// Render the trie as indented key listing.
    pub fn key_listing(trie: &NGramTrie) -> String {
        let mut keys = trie.keys();
        keys.sort();
        keys.iter()
            .map(|k| {
                let display: String = k.iter().map(|&b| {
                    if b.is_ascii_graphic() || b == b' ' {
                        (b as char).to_string()
                    } else {
                        format!("\\x{:02x}", b)
                    }
                }).collect();
                format!("  {}", display)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Render DOT for the compact (Patricia) version of the trie.
    pub fn compact_to_dot(trie: &NGramTrie) -> String {
        let compact = trie.compact();
        let mut out = String::from("digraph CompactTrie {\n  node [shape=box];\n");
        let mut id = 0usize;
        Self::compact_dot_node(&compact, &mut out, &mut id, None);
        out.push_str("}\n");
        out
    }

    fn compact_dot_node(
        node: &CompactTrieNode,
        out: &mut String,
        id_counter: &mut usize,
        parent_id: Option<usize>,
    ) {
        let my_id = *id_counter;
        *id_counter += 1;
        let label_str: String = node.label.iter().map(|&b| {
            if b.is_ascii_graphic() { (b as char).to_string() } else { format!("\\x{:02x}", b) }
        }).collect();
        let shape = if node.is_terminal { "doublecircle" } else { "box" };
        out.push_str(&format!(
            "  n{} [shape={}, label=\"{}\"];\n",
            my_id, shape, label_str
        ));
        if let Some(pid) = parent_id {
            out.push_str(&format!("  n{} -> n{};\n", pid, my_id));
        }
        let mut child_keys: Vec<u8> = node.children.keys().cloned().collect();
        child_keys.sort_unstable();
        for k in child_keys {
            Self::compact_dot_node(&node.children[&k], out, id_counter, Some(my_id));
        }
    }
}

// ---------------------------------------------------------------------------
// TrieWalker — advanced trie traversal utilities
// ---------------------------------------------------------------------------

/// Static methods for traversing tries in various orders.
pub struct TrieWalker;

impl TrieWalker {
    /// Depth-first traversal returning (key, depth) pairs.
    pub fn dfs(trie: &NGramTrie) -> Vec<(Vec<u8>, usize)> {
        let mut result = Vec::new();
        let mut stack: Vec<(&TrieNode, Vec<u8>, usize)> = vec![(&trie.root, Vec::new(), 0)];
        while let Some((node, path, depth)) = stack.pop() {
            if node.is_terminal {
                result.push((path.clone(), depth));
            }
            let mut child_keys: Vec<u8> = node.children.keys().cloned().collect();
            child_keys.sort_unstable();
            for &k in child_keys.iter().rev() {
                let child = &node.children[&k];
                let mut cp = path.clone();
                cp.push(k);
                stack.push((child, cp, depth + 1));
            }
        }
        result
    }

    /// Breadth-first traversal returning (key, depth) pairs.
    pub fn bfs(trie: &NGramTrie) -> Vec<(Vec<u8>, usize)> {
        let mut result = Vec::new();
        let mut queue: std::collections::VecDeque<(&TrieNode, Vec<u8>, usize)> =
            std::collections::VecDeque::new();
        queue.push_back((&trie.root, Vec::new(), 0));
        while let Some((node, path, depth)) = queue.pop_front() {
            if node.is_terminal {
                result.push((path.clone(), depth));
            }
            let mut child_keys: Vec<u8> = node.children.keys().cloned().collect();
            child_keys.sort_unstable();
            for &k in &child_keys {
                let child = &node.children[&k];
                let mut cp = path.clone();
                cp.push(k);
                queue.push_back((child, cp, depth + 1));
            }
        }
        result
    }

    /// Level-order traversal: return keys grouped by depth level.
    pub fn level_order(trie: &NGramTrie) -> Vec<Vec<Vec<u8>>> {
        let bfs_result = Self::bfs(trie);
        if bfs_result.is_empty() {
            return Vec::new();
        }
        let max_depth = bfs_result.iter().map(|(_, d)| *d).max().unwrap_or(0);
        let mut levels = vec![Vec::new(); max_depth + 1];
        for (key, depth) in bfs_result {
            levels[depth].push(key);
        }
        levels
    }

    /// Collect all leaf keys (keys with no extensions in the trie).
    pub fn leaves(trie: &NGramTrie) -> Vec<Vec<u8>> {
        let mut result = Vec::new();
        Self::collect_leaves(&trie.root, &mut Vec::new(), &mut result);
        result
    }

    fn collect_leaves(node: &TrieNode, path: &mut Vec<u8>, result: &mut Vec<Vec<u8>>) {
        if node.is_terminal && node.children.is_empty() {
            result.push(path.clone());
        }
        let mut child_keys: Vec<u8> = node.children.keys().cloned().collect();
        child_keys.sort_unstable();
        for &k in &child_keys {
            path.push(k);
            Self::collect_leaves(&node.children[&k], path, result);
            path.pop();
        }
    }

    /// Count internal (non-leaf) nodes.
    pub fn internal_nodes(trie: &NGramTrie) -> usize {
        Self::count_internal(&trie.root)
    }

    fn count_internal(node: &TrieNode) -> usize {
        if node.children.is_empty() {
            return 0;
        }
        let mut count = 1;
        for child in node.children.values() {
            count += Self::count_internal(child);
        }
        count
    }

    /// Find the longest key in the trie.
    pub fn longest_key(trie: &NGramTrie) -> Vec<u8> {
        let mut longest = Vec::new();
        Self::find_longest(&trie.root, &mut Vec::new(), &mut longest);
        longest
    }

    fn find_longest(node: &TrieNode, path: &mut Vec<u8>, longest: &mut Vec<u8>) {
        if node.is_terminal && path.len() > longest.len() {
            *longest = path.clone();
        }
        let mut child_keys: Vec<u8> = node.children.keys().cloned().collect();
        child_keys.sort_unstable();
        for &k in &child_keys {
            path.push(k);
            Self::find_longest(&node.children[&k], path, longest);
            path.pop();
        }
    }

    /// Find the shortest key in the trie.
    pub fn shortest_key(trie: &NGramTrie) -> Vec<u8> {
        if trie.is_empty() {
            return Vec::new();
        }
        let mut shortest: Option<Vec<u8>> = None;
        Self::find_shortest(&trie.root, &mut Vec::new(), &mut shortest);
        shortest.unwrap_or_default()
    }

    fn find_shortest(node: &TrieNode, path: &mut Vec<u8>, shortest: &mut Option<Vec<u8>>) {
        if node.is_terminal {
            if shortest.is_none() || path.len() < shortest.as_ref().unwrap().len() {
                *shortest = Some(path.clone());
            }
        }
        let mut child_keys: Vec<u8> = node.children.keys().cloned().collect();
        child_keys.sort_unstable();
        for &k in &child_keys {
            path.push(k);
            Self::find_shortest(&node.children[&k], path, shortest);
            path.pop();
        }
    }

    /// Count nodes at each depth level.
    pub fn nodes_at_depth(trie: &NGramTrie, target_depth: usize) -> usize {
        Self::count_at_depth(&trie.root, 0, target_depth)
    }

    fn count_at_depth(node: &TrieNode, current: usize, target: usize) -> usize {
        if current == target {
            return 1;
        }
        let mut count = 0;
        for child in node.children.values() {
            count += Self::count_at_depth(child, current + 1, target);
        }
        count
    }

    /// Find all keys of a specific length.
    pub fn keys_of_length(trie: &NGramTrie, length: usize) -> Vec<Vec<u8>> {
        let mut result = Vec::new();
        Self::collect_keys_of_length(&trie.root, &mut Vec::new(), length, &mut result);
        result
    }

    fn collect_keys_of_length(
        node: &TrieNode,
        path: &mut Vec<u8>,
        target_len: usize,
        result: &mut Vec<Vec<u8>>,
    ) {
        if path.len() == target_len {
            if node.is_terminal {
                result.push(path.clone());
            }
            return;
        }
        if path.len() > target_len {
            return;
        }
        let mut child_keys: Vec<u8> = node.children.keys().cloned().collect();
        child_keys.sort_unstable();
        for &k in &child_keys {
            path.push(k);
            Self::collect_keys_of_length(&node.children[&k], path, target_len, result);
            path.pop();
        }
    }

    /// Total path length (sum of depths of all terminal nodes).
    pub fn total_path_length(trie: &NGramTrie) -> usize {
        let dfs = Self::dfs(trie);
        dfs.iter().map(|(_, d)| *d).sum()
    }

    /// Average key depth.
    pub fn average_key_depth(trie: &NGramTrie) -> f64 {
        let dfs = Self::dfs(trie);
        if dfs.is_empty() {
            return 0.0;
        }
        let total: usize = dfs.iter().map(|(_, d)| *d).sum();
        total as f64 / dfs.len() as f64
    }

    /// Check if any key in the trie is a prefix of another key.
    pub fn has_prefix_keys(trie: &NGramTrie) -> bool {
        Self::check_prefix_keys(&trie.root, &mut Vec::new())
    }

    fn check_prefix_keys(node: &TrieNode, _path: &mut Vec<u8>) -> bool {
        if node.is_terminal && !node.children.is_empty() {
            return true;
        }
        for (_, child) in &node.children {
            if Self::check_prefix_keys(child, _path) {
                return true;
            }
        }
        false
    }

    /// Depth of a specific key (or None if not found).
    pub fn key_depth(trie: &NGramTrie, key: &[u8]) -> Option<usize> {
        let mut node = &trie.root;
        for &byte in key {
            match node.children.get(&byte) {
                Some(child) => node = child,
                None => return None,
            }
        }
        if node.is_terminal {
            Some(key.len())
        } else {
            None
        }
    }

    /// Collect all keys that match a given predicate.
    pub fn filter_keys<F>(trie: &NGramTrie, predicate: F) -> Vec<Vec<u8>>
    where
        F: Fn(&[u8]) -> bool,
    {
        trie.keys().into_iter().filter(|k| predicate(k)).collect()
    }

    /// Find all keys starting with a common prefix and return them sorted.
    pub fn sorted_prefix_search(trie: &NGramTrie, prefix: &[u8]) -> Vec<Vec<u8>> {
        let mut keys = trie.prefix_search(prefix);
        keys.sort();
        keys
    }
}

// ---------------------------------------------------------------------------
// TrieMerger — utilities for combining tries
// ---------------------------------------------------------------------------

/// Utilities for merging and combining tries.
pub struct TrieMerger;

impl TrieMerger {
    /// Merge two tries into one (union of all keys).
    pub fn merge(a: &NGramTrie, b: &NGramTrie) -> NGramTrie {
        a.union(b)
    }

    /// Merge multiple tries into one.
    pub fn merge_all(tries: &[&NGramTrie]) -> NGramTrie {
        let mut result = NGramTrie::new();
        for trie in tries {
            for key in trie.iter() {
                result.insert(&key);
            }
        }
        result
    }

    /// Intersect multiple tries.
    pub fn intersect_all(tries: &[&NGramTrie]) -> NGramTrie {
        if tries.is_empty() {
            return NGramTrie::new();
        }
        let mut result = tries[0].clone();
        for trie in &tries[1..] {
            result = result.intersect(trie);
        }
        result
    }

    /// Compute the "core" keys that appear in at least `threshold` fraction of the tries.
    pub fn core_keys(tries: &[&NGramTrie], threshold: f64) -> NGramTrie {
        let min_count = (tries.len() as f64 * threshold).ceil() as usize;
        let mut key_counts: HashMap<Vec<u8>, usize> = HashMap::new();
        for trie in tries {
            for key in trie.iter() {
                *key_counts.entry(key).or_insert(0) += 1;
            }
        }
        let mut result = NGramTrie::new();
        for (key, count) in key_counts {
            if count >= min_count {
                result.insert(&key);
            }
        }
        result
    }

    /// Compute the "unique" keys that appear in only one trie.
    pub fn unique_keys(tries: &[&NGramTrie]) -> NGramTrie {
        let mut key_counts: HashMap<Vec<u8>, usize> = HashMap::new();
        for trie in tries {
            for key in trie.iter() {
                *key_counts.entry(key).or_insert(0) += 1;
            }
        }
        let mut result = NGramTrie::new();
        for (key, count) in key_counts {
            if count == 1 {
                result.insert(&key);
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// TrieComparator — comprehensive trie comparison
// ---------------------------------------------------------------------------

/// High-level comparison of two tries producing a detailed report.
pub struct TrieComparator;

impl TrieComparator {
    /// Generate a full comparison report.
    pub fn compare(a: &NGramTrie, b: &NGramTrie) -> TrieComparisonReport {
        let a_keys = a.len();
        let b_keys = b.len();
        let intersection = a.intersection_cardinality(b);
        let union = a_keys + b_keys - intersection;
        let jaccard = if union == 0 { 0.0 } else { intersection as f64 / union as f64 };
        let containment_ab = if a_keys == 0 { 0.0 } else { intersection as f64 / a_keys as f64 };
        let containment_ba = if b_keys == 0 { 0.0 } else { intersection as f64 / b_keys as f64 };
        let stats_a = TrieStatistics::from_trie(a);
        let stats_b = TrieStatistics::from_trie(b);

        TrieComparisonReport {
            keys_a: a_keys,
            keys_b: b_keys,
            intersection_size: intersection,
            union_size: union,
            jaccard,
            containment_a_in_b: containment_ab,
            containment_b_in_a: containment_ba,
            nodes_a: stats_a.node_count,
            nodes_b: stats_b.node_count,
            max_depth_a: stats_a.max_depth,
            max_depth_b: stats_b.max_depth,
        }
    }
}

/// Result of comparing two tries.
#[derive(Clone, Debug)]
pub struct TrieComparisonReport {
    pub keys_a: usize,
    pub keys_b: usize,
    pub intersection_size: usize,
    pub union_size: usize,
    pub jaccard: f64,
    pub containment_a_in_b: f64,
    pub containment_b_in_a: f64,
    pub nodes_a: usize,
    pub nodes_b: usize,
    pub max_depth_a: usize,
    pub max_depth_b: usize,
}

impl TrieComparisonReport {
    pub fn summary(&self) -> String {
        format!(
            "TrieComparison: |A|={} keys ({} nodes), |B|={} keys ({} nodes), \
             Jaccard={:.4}, |A∩B|={}, |A∪B|={}",
            self.keys_a, self.nodes_a,
            self.keys_b, self.nodes_b,
            self.jaccard,
            self.intersection_size,
            self.union_size,
        )
    }

    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"keys_a\":{},",
                "\"keys_b\":{},",
                "\"intersection_size\":{},",
                "\"union_size\":{},",
                "\"jaccard\":{:.6},",
                "\"containment_a_in_b\":{:.6},",
                "\"containment_b_in_a\":{:.6},",
                "\"nodes_a\":{},",
                "\"nodes_b\":{},",
                "\"max_depth_a\":{},",
                "\"max_depth_b\":{}",
                "}}"
            ),
            self.keys_a,
            self.keys_b,
            self.intersection_size,
            self.union_size,
            self.jaccard,
            self.containment_a_in_b,
            self.containment_b_in_a,
            self.nodes_a,
            self.nodes_b,
            self.max_depth_a,
            self.max_depth_b,
        )
    }
}

// ---------------------------------------------------------------------------
// TrieCheckpoint / TrieCheckpointDiff / TrieCheckpointer
// ---------------------------------------------------------------------------

/// A snapshot of a trie at a point in time.
#[derive(Clone, Debug)]
pub struct TrieCheckpoint {
    pub serialized_data: Vec<u8>,
    pub node_count: usize,
    pub timestamp: u64,
}

/// Describes the difference between two trie checkpoints.
#[derive(Clone, Debug)]
pub struct TrieCheckpointDiff {
    pub added_keys: Vec<Vec<u8>>,
    pub removed_keys: Vec<Vec<u8>>,
    pub num_changes: usize,
}

impl TrieCheckpoint {
    /// Return the size in bytes of the serialized data.
    pub fn data_size(&self) -> usize {
        self.serialized_data.len()
    }

    /// Check whether this checkpoint was created before `other`.
    pub fn is_older_than(&self, other: &TrieCheckpoint) -> bool {
        self.timestamp < other.timestamp
    }
}

impl TrieCheckpointDiff {
    /// True when both tries had identical key sets.
    pub fn is_empty(&self) -> bool {
        self.num_changes == 0
    }

    /// Return a human-readable summary of the diff.
    pub fn summary(&self) -> String {
        format!(
            "TrieCheckpointDiff: {} added, {} removed, {} total changes",
            self.added_keys.len(),
            self.removed_keys.len(),
            self.num_changes,
        )
    }
}

/// Utility for snapshotting and restoring tries.
pub struct TrieCheckpointer;

impl TrieCheckpointer {
    /// Serialize the trie into a checkpoint with the current timestamp.
    pub fn checkpoint(trie: &NGramTrie) -> TrieCheckpoint {
        let serialized_data = trie.serialize();
        let node_count = trie.total_nodes;
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        TrieCheckpoint {
            serialized_data,
            node_count,
            timestamp,
        }
    }

    /// Restore a trie from a checkpoint.
    pub fn restore(checkpoint: &TrieCheckpoint) -> NGramTrie {
        NGramTrie::deserialize(&checkpoint.serialized_data)
            .unwrap_or_else(|_| NGramTrie::new())
    }

    /// Compute the diff between two checkpoints by restoring both and
    /// comparing their key sets.
    pub fn diff_checkpoints(a: &TrieCheckpoint, b: &TrieCheckpoint) -> TrieCheckpointDiff {
        let trie_a = Self::restore(a);
        let trie_b = Self::restore(b);
        let keys_a: std::collections::HashSet<Vec<u8>> = trie_a.keys().into_iter().collect();
        let keys_b: std::collections::HashSet<Vec<u8>> = trie_b.keys().into_iter().collect();

        let mut added_keys: Vec<Vec<u8>> = keys_b.difference(&keys_a).cloned().collect();
        let mut removed_keys: Vec<Vec<u8>> = keys_a.difference(&keys_b).cloned().collect();
        added_keys.sort();
        removed_keys.sort();
        let num_changes = added_keys.len() + removed_keys.len();
        TrieCheckpointDiff {
            added_keys,
            removed_keys,
            num_changes,
        }
    }
}

// ---------------------------------------------------------------------------
// TrieIndex — secondary indexes on a trie
// ---------------------------------------------------------------------------

/// Build secondary indexes over trie keys.
pub struct TrieIndex;

impl TrieIndex {
    /// Group all keys by their length.
    pub fn build_depth_index(trie: &NGramTrie) -> HashMap<usize, Vec<Vec<u8>>> {
        let mut index: HashMap<usize, Vec<Vec<u8>>> = HashMap::new();
        for key in trie.iter() {
            index.entry(key.len()).or_default().push(key);
        }
        index
    }

    /// Group keys by the fanout (child count) at their terminal node.
    pub fn build_fanout_index(trie: &NGramTrie) -> HashMap<usize, Vec<Vec<u8>>> {
        let mut index: HashMap<usize, Vec<Vec<u8>>> = HashMap::new();
        for key in trie.iter() {
            let fanout = Self::terminal_fanout(trie, &key);
            index.entry(fanout).or_default().push(key);
        }
        index
    }

    /// Return the child count of the node reached after walking `key`.
    fn terminal_fanout(trie: &NGramTrie, key: &[u8]) -> usize {
        let mut node = &trie.root;
        for &byte in key {
            match node.children.get(&byte) {
                Some(child) => node = child,
                None => return 0,
            }
        }
        node.child_count()
    }

    /// Return all keys whose length is exactly `depth`.
    pub fn keys_at_depth(trie: &NGramTrie, depth: usize) -> Vec<Vec<u8>> {
        trie.iter().filter(|k| k.len() == depth).collect()
    }

    /// Count the number of keys that start with `prefix`.
    pub fn subtree_size(trie: &NGramTrie, prefix: &[u8]) -> usize {
        trie.prefix_count(prefix)
    }

    /// Build an index mapping each distinct prefix of length `len` to the
    /// keys that share that prefix.
    pub fn build_prefix_index(trie: &NGramTrie, len: usize) -> HashMap<Vec<u8>, Vec<Vec<u8>>> {
        let mut index: HashMap<Vec<u8>, Vec<Vec<u8>>> = HashMap::new();
        for key in trie.iter() {
            let prefix = if key.len() >= len {
                key[..len].to_vec()
            } else {
                key.clone()
            };
            index.entry(prefix).or_default().push(key);
        }
        index
    }

    /// Return the number of distinct key lengths present in the trie.
    pub fn depth_count(trie: &NGramTrie) -> usize {
        let mut depths: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for key in trie.iter() {
            depths.insert(key.len());
        }
        depths.len()
    }

    /// Return the shortest key(s) in the trie.
    pub fn shortest_keys(trie: &NGramTrie) -> Vec<Vec<u8>> {
        let keys = trie.keys();
        if keys.is_empty() {
            return Vec::new();
        }
        let min_len = keys.iter().map(|k| k.len()).min().unwrap_or(0);
        keys.into_iter().filter(|k| k.len() == min_len).collect()
    }

    /// Return the longest key(s) in the trie.
    pub fn longest_keys(trie: &NGramTrie) -> Vec<Vec<u8>> {
        let keys = trie.keys();
        if keys.is_empty() {
            return Vec::new();
        }
        let max_len = keys.iter().map(|k| k.len()).max().unwrap_or(0);
        keys.into_iter().filter(|k| k.len() == max_len).collect()
    }
}

// ---------------------------------------------------------------------------
// TriePartitioner — partition a trie for parallel processing
// ---------------------------------------------------------------------------

/// Split and merge tries for parallel workloads.
pub struct TriePartitioner;

impl TriePartitioner {
    /// Split the trie into `num_parts` partitions based on the first byte of
    /// each key.  Keys are distributed across partitions by dividing the byte
    /// range [0, 256) into roughly equal sub-ranges.
    pub fn partition(trie: &NGramTrie, num_parts: usize) -> Vec<NGramTrie> {
        if num_parts == 0 {
            return Vec::new();
        }
        let keys = trie.keys();
        if keys.is_empty() || num_parts == 1 {
            let mut single = NGramTrie::new();
            for key in &keys {
                single.insert(key);
            }
            return vec![single];
        }

        let range_size = (256 + num_parts - 1) / num_parts; // ceil division
        let mut parts: Vec<NGramTrie> = (0..num_parts).map(|_| NGramTrie::new()).collect();

        for key in &keys {
            let first_byte = if key.is_empty() { 0u8 } else { key[0] };
            let idx = (first_byte as usize) / range_size;
            let idx = idx.min(num_parts - 1);
            parts[idx].insert(key);
        }
        parts
    }

    /// Merge a slice of partitions back into a single trie (union).
    pub fn merge_partitions(parts: &[NGramTrie]) -> NGramTrie {
        let mut merged = NGramTrie::new();
        for part in parts {
            for key in part.iter() {
                merged.insert(&key);
            }
        }
        merged
    }

    /// Create partitions where each has at most `max_per_part` keys.
    pub fn balanced_partition(trie: &NGramTrie, max_per_part: usize) -> Vec<NGramTrie> {
        let keys = trie.keys();
        if keys.is_empty() {
            return vec![NGramTrie::new()];
        }
        let cap = if max_per_part == 0 { 1 } else { max_per_part };
        let mut parts: Vec<NGramTrie> = Vec::new();
        let mut current = NGramTrie::new();
        let mut count = 0usize;

        for key in &keys {
            current.insert(key);
            count += 1;
            if count >= cap {
                parts.push(current);
                current = NGramTrie::new();
                count = 0;
            }
        }
        if count > 0 || parts.is_empty() {
            parts.push(current);
        }
        parts
    }

    /// Partition by key depth: all keys of the same length go into the same
    /// partition.
    pub fn partition_by_depth(trie: &NGramTrie) -> Vec<NGramTrie> {
        let depth_idx = TrieIndex::build_depth_index(trie);
        let mut depths: Vec<usize> = depth_idx.keys().cloned().collect();
        depths.sort();
        depths
            .into_iter()
            .map(|d| {
                let keys = &depth_idx[&d];
                let mut part = NGramTrie::new();
                for key in keys {
                    part.insert(key);
                }
                part
            })
            .collect()
    }

    /// Return the number of non-empty partitions produced by `partition`.
    pub fn count_nonempty_partitions(trie: &NGramTrie, num_parts: usize) -> usize {
        Self::partition(trie, num_parts)
            .iter()
            .filter(|p| !p.is_empty())
            .count()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Basic insert / contains --

    #[test]
    fn test_insert_and_contains() {
        let mut trie = NGramTrie::new();
        trie.insert(b"hello");
        trie.insert(b"help");
        trie.insert(b"world");

        assert!(trie.contains(b"hello"));
        assert!(trie.contains(b"help"));
        assert!(trie.contains(b"world"));
        assert!(!trie.contains(b"hel"));
        assert!(!trie.contains(b"helloo"));
        assert_eq!(trie.len(), 3);
    }

    #[test]
    fn test_empty_trie() {
        let trie = NGramTrie::new();
        assert!(trie.is_empty());
        assert!(!trie.contains(b"x"));
    }

    #[test]
    fn test_insert_empty_key() {
        let mut trie = NGramTrie::new();
        trie.insert(b"");
        assert!(trie.contains(b""));
        assert_eq!(trie.len(), 1);
    }

    #[test]
    fn test_duplicate_insert() {
        let mut trie = NGramTrie::new();
        trie.insert(b"abc");
        trie.insert(b"abc");
        assert_eq!(trie.len(), 1);
    }

    // -- Remove --

    #[test]
    fn test_remove() {
        let mut trie = NGramTrie::new();
        trie.insert(b"abc");
        trie.insert(b"abd");
        assert!(trie.remove(b"abc"));
        assert!(!trie.contains(b"abc"));
        assert!(trie.contains(b"abd"));
        assert_eq!(trie.len(), 1);
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut trie = NGramTrie::new();
        trie.insert(b"abc");
        assert!(!trie.remove(b"xyz"));
        assert_eq!(trie.len(), 1);
    }

    // -- Prefix search --

    #[test]
    fn test_prefix_search() {
        let mut trie = NGramTrie::new();
        trie.insert(b"hello");
        trie.insert(b"help");
        trie.insert(b"heap");
        trie.insert(b"world");

        let results = trie.prefix_search(b"hel");
        assert_eq!(results.len(), 2);
        assert!(results.contains(&b"hello".to_vec()));
        assert!(results.contains(&b"help".to_vec()));
    }

    #[test]
    fn test_prefix_count() {
        let mut trie = NGramTrie::new();
        trie.insert(b"hello");
        trie.insert(b"help");
        trie.insert(b"heap");
        assert_eq!(trie.prefix_count(b"hel"), 2);
        assert_eq!(trie.prefix_count(b"he"), 3);
        assert_eq!(trie.prefix_count(b"z"), 0);
    }

    // -- Iterator / keys --

    #[test]
    fn test_iter() {
        let mut trie = NGramTrie::new();
        trie.insert(b"abc");
        trie.insert(b"abd");
        trie.insert(b"xyz");
        let keys = trie.keys();
        assert_eq!(keys.len(), 3);
        assert!(keys.contains(&b"abc".to_vec()));
        assert!(keys.contains(&b"abd".to_vec()));
        assert!(keys.contains(&b"xyz".to_vec()));
    }

    // -- from_ngrams --

    #[test]
    fn test_from_ngrams() {
        let ngrams: Vec<Vec<u8>> = vec![
            b"cat".to_vec(),
            b"car".to_vec(),
            b"card".to_vec(),
        ];
        let trie = NGramTrie::from_ngrams(&ngrams);
        assert_eq!(trie.len(), 3);
        assert!(trie.contains(b"cat"));
        assert!(trie.contains(b"car"));
        assert!(trie.contains(b"card"));
    }

    // -- Set operations --

    #[test]
    fn test_intersect() {
        let t1 = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec(), b"ghi".to_vec()]);
        let t2 = NGramTrie::from_ngrams(&[b"def".to_vec(), b"ghi".to_vec(), b"jkl".to_vec()]);
        let inter = t1.intersect(&t2);
        assert_eq!(inter.len(), 2);
        assert!(inter.contains(b"def"));
        assert!(inter.contains(b"ghi"));
    }

    #[test]
    fn test_intersection_cardinality() {
        let t1 = NGramTrie::from_ngrams(&[b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]);
        let t2 = NGramTrie::from_ngrams(&[b"b".to_vec(), b"c".to_vec(), b"d".to_vec()]);
        assert_eq!(t1.intersection_cardinality(&t2), 2);
    }

    #[test]
    fn test_union() {
        let t1 = NGramTrie::from_ngrams(&[b"a".to_vec(), b"b".to_vec()]);
        let t2 = NGramTrie::from_ngrams(&[b"b".to_vec(), b"c".to_vec()]);
        let uni = t1.union(&t2);
        assert_eq!(uni.len(), 3);
    }

    #[test]
    fn test_difference() {
        let t1 = NGramTrie::from_ngrams(&[b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]);
        let t2 = NGramTrie::from_ngrams(&[b"b".to_vec()]);
        let diff = t1.difference(&t2);
        assert_eq!(diff.len(), 2);
        assert!(diff.contains(b"a"));
        assert!(diff.contains(b"c"));
    }

    // -- Statistics --

    #[test]
    fn test_statistics() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"abd".to_vec(), b"xyz".to_vec()]);
        let stats = trie.statistics();
        assert!(stats.node_count > 0);
        assert_eq!(stats.leaf_count + stats.internal_node_count, stats.node_count);
        assert!(stats.max_depth <= 3);
    }

    #[test]
    fn test_depth_histogram() {
        let trie = NGramTrie::from_ngrams(&[b"ab".to_vec(), b"cd".to_vec()]);
        let hist = trie.depth_histogram();
        assert!(!hist.is_empty());
    }

    #[test]
    fn test_fanout_histogram() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"abd".to_vec()]);
        let hist = trie.fanout_histogram();
        assert!(!hist.is_empty());
    }

    #[test]
    fn test_prefix_sharing_ratio() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"abd".to_vec(), b"abe".to_vec()]);
        let ratio = trie.prefix_sharing_ratio();
        assert!(ratio > 0.0, "Expected positive prefix sharing, got {}", ratio);
    }

    // -- Serialization --

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let mut trie = NGramTrie::new();
        trie.insert(b"hello");
        trie.insert(b"help");
        trie.insert(b"world");

        let bytes = trie.serialize();
        let recovered = NGramTrie::deserialize(&bytes).unwrap();

        assert_eq!(recovered.len(), 3);
        assert!(recovered.contains(b"hello"));
        assert!(recovered.contains(b"help"));
        assert!(recovered.contains(b"world"));
    }

    #[test]
    fn test_serialize_empty() {
        let trie = NGramTrie::new();
        let bytes = trie.serialize();
        let recovered = NGramTrie::deserialize(&bytes).unwrap();
        assert!(recovered.is_empty());
    }

    #[test]
    fn test_deserialize_invalid() {
        let result = NGramTrie::deserialize(&[]);
        assert!(result.is_err());
    }

    // -- DOT output --

    #[test]
    fn test_to_dot() {
        let trie = NGramTrie::from_ngrams(&[b"ab".to_vec(), b"ac".to_vec()]);
        let dot = trie.to_dot();
        assert!(dot.contains("digraph"));
        assert!(dot.contains("->"));
    }

    // -- Compaction --

    #[test]
    fn test_compact() {
        let trie = NGramTrie::from_ngrams(&[b"abcdef".to_vec()]);
        let compact = trie.compact();
        // Single key: should compact into very few nodes.
        let compact_count = NGramTrie::count_compact_nodes(&compact);
        assert!(compact_count <= 2, "Expected at most 2 compact nodes, got {}", compact_count);
    }

    #[test]
    fn test_compaction_ratio() {
        let trie = NGramTrie::from_ngrams(&[b"abcdef".to_vec()]);
        let ratio = trie.compaction_ratio();
        assert!(ratio > 0.0);
    }

    // -- Communication complexity estimation --

    #[test]
    fn test_estimated_communication_bytes() {
        let t1 = NGramTrie::from_ngrams(&[b"a".to_vec(); 100]);
        let t2 = NGramTrie::from_ngrams(&[b"b".to_vec(); 200]);
        let bytes = t1.estimated_communication_bytes(&t2);
        assert!(bytes > 0);
    }

    #[test]
    fn test_estimated_psi_rounds() {
        let t1 = NGramTrie::new();
        let t2 = NGramTrie::new();
        assert_eq!(t1.estimated_psi_rounds(&t2), 3);
    }

    #[test]
    fn test_trie_structure_hash_deterministic() {
        let t1 = NGramTrie::from_ngrams(&[b"hello".to_vec(), b"world".to_vec()]);
        let t2 = NGramTrie::from_ngrams(&[b"hello".to_vec(), b"world".to_vec()]);
        assert_eq!(t1.trie_structure_hash(), t2.trie_structure_hash());
    }

    // -- Common prefix length --

    #[test]
    fn test_common_prefix_length() {
        let trie = NGramTrie::new();
        assert_eq!(trie.common_prefix_length(b"abcdef", b"abcxyz"), 3);
        assert_eq!(trie.common_prefix_length(b"abc", b"xyz"), 0);
        assert_eq!(trie.common_prefix_length(b"abc", b"abc"), 3);
    }

    // -- BatchTrie --

    #[test]
    fn test_batch_trie() {
        let config = super::super::ngram::NGramConfig::char_ngrams(3);
        let mut batch = BatchTrie::new();
        batch.add_document("hello world", config.clone());
        batch.add_document("world peace", config);
        assert_eq!(batch.tries.len(), 2);

        let combined = batch.combined_trie();
        assert!(combined.len() > 0);

        let pairwise = batch.pairwise_intersections();
        assert_eq!(pairwise.len(), 1); // 1 pair for 2 documents
    }

    // -- from_ngram_set --

    #[test]
    fn test_from_ngram_set() {
        let config = super::super::ngram::NGramConfig::char_ngrams(3);
        let set = super::super::ngram::NGramSet::from_text("abcdefgh", config);
        let trie = NGramTrie::from_ngram_set(&set);
        assert_eq!(trie.len(), set.len());
    }

    // =====================================================================
    // CompactTrieNode additional tests
    // =====================================================================

    #[test]
    fn test_compact_trie_node_is_leaf() {
        let node = CompactTrieNode::new(vec![1, 2, 3]);
        assert!(node.is_leaf());
        assert_eq!(node.child_count(), 0);
    }

    #[test]
    fn test_compact_trie_node_label_len() {
        let node = CompactTrieNode::new(vec![1, 2, 3]);
        assert_eq!(node.label_len(), 3);
    }

    #[test]
    fn test_compact_trie_node_subtree_size() {
        let mut node = CompactTrieNode::new(vec![1]);
        let child = CompactTrieNode::new(vec![2]);
        node.children.insert(2, Box::new(child));
        assert_eq!(node.subtree_size(), 2);
    }

    #[test]
    fn test_compact_trie_node_terminal_count() {
        let mut node = CompactTrieNode::new(vec![1]);
        node.is_terminal = true;
        let mut child = CompactTrieNode::new(vec![2]);
        child.is_terminal = true;
        node.children.insert(2, Box::new(child));
        assert_eq!(node.terminal_count(), 2);
    }

    // =====================================================================
    // CompactTrie tests
    // =====================================================================

    #[test]
    fn test_compact_trie_from_trie() {
        let trie = NGramTrie::from_ngrams(&[b"hello".to_vec(), b"help".to_vec(), b"world".to_vec()]);
        let compact = CompactTrie::from_trie(&trie);
        assert_eq!(compact.len(), 3);
    }

    #[test]
    fn test_compact_trie_node_count() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec()]);
        let compact = CompactTrie::from_trie(&trie);
        assert!(compact.node_count() <= trie.total_nodes);
    }

    #[test]
    fn test_compact_trie_compression_ratio() {
        let trie = NGramTrie::from_ngrams(&[b"abcdef".to_vec()]);
        let compact = CompactTrie::from_trie(&trie);
        let ratio = compact.compression_ratio(&trie);
        assert!(ratio > 0.0, "Expected positive compression ratio, got {}", ratio);
    }

    #[test]
    fn test_compact_trie_memory_estimate() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"abd".to_vec()]);
        let compact = CompactTrie::from_trie(&trie);
        assert!(compact.memory_estimate() > 0);
    }

    #[test]
    fn test_compact_trie_empty() {
        let trie = NGramTrie::new();
        let compact = CompactTrie::from_trie(&trie);
        assert!(compact.is_empty());
        assert_eq!(compact.len(), 0);
    }

    #[test]
    fn test_compact_trie_insert() {
        let trie = NGramTrie::new();
        let mut compact = CompactTrie::from_trie(&trie);
        compact.insert(b"hello");
        compact.insert(b"help");
        assert_eq!(compact.len(), 2);
    }

    // =====================================================================
    // CompactTrieIterator tests
    // =====================================================================

    #[test]
    fn test_compact_trie_iterator() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"abd".to_vec()]);
        let compact = trie.compact();
        let iter = CompactTrieIterator::new(&compact);
        let keys: Vec<Vec<u8>> = iter.collect();
        assert_eq!(keys.len(), 2);
    }

    // =====================================================================
    // TrieStatistics tests
    // =====================================================================

    #[test]
    fn test_trie_statistics_from_trie() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"abd".to_vec(), b"xyz".to_vec()]);
        let stats = TrieStatistics::from_trie(&trie);
        assert_eq!(stats.key_count, 3);
        assert!(stats.node_count > 0);
    }

    #[test]
    fn test_trie_statistics_summary() {
        let trie = NGramTrie::from_ngrams(&[b"ab".to_vec(), b"cd".to_vec()]);
        let stats = TrieStatistics::from_trie(&trie);
        let summary = stats.summary();
        assert!(summary.contains("keys:"));
        assert!(summary.contains("nodes:"));
    }

    #[test]
    fn test_trie_statistics_to_json() {
        let trie = NGramTrie::from_ngrams(&[b"ab".to_vec()]);
        let stats = TrieStatistics::from_trie(&trie);
        let json = stats.to_json();
        assert!(json.contains("\"node_count\""));
        assert!(json.contains("\"key_count\""));
    }

    #[test]
    fn test_trie_statistics_average_key_length() {
        let trie = NGramTrie::from_ngrams(&[b"ab".to_vec(), b"cdef".to_vec()]);
        let stats = TrieStatistics::from_trie(&trie);
        let avg = stats.average_key_length();
        assert!((avg - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_trie_statistics_density() {
        let trie = NGramTrie::from_ngrams(&[b"ab".to_vec()]);
        let stats = TrieStatistics::from_trie(&trie);
        let d = stats.density();
        assert!(d > 0.0 && d <= 1.0);
    }

    // =====================================================================
    // TrieDifference tests
    // =====================================================================

    #[test]
    fn test_trie_diff_added_keys() {
        let old = NGramTrie::from_ngrams(&[b"abc".to_vec()]);
        let new = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec()]);
        let added = TrieDifference::added_keys(&old, &new);
        assert_eq!(added.len(), 1);
        assert_eq!(added[0], b"def".to_vec());
    }

    #[test]
    fn test_trie_diff_removed_keys() {
        let old = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec()]);
        let new = NGramTrie::from_ngrams(&[b"abc".to_vec()]);
        let removed = TrieDifference::removed_keys(&old, &new);
        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0], b"def".to_vec());
    }

    #[test]
    fn test_trie_diff_symmetric_difference() {
        let a = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec()]);
        let b = NGramTrie::from_ngrams(&[b"def".to_vec(), b"ghi".to_vec()]);
        let sd = TrieDifference::symmetric_difference(&a, &b);
        assert_eq!(sd.len(), 2);
    }

    #[test]
    fn test_trie_diff_structural_similarity_identical() {
        let a = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec()]);
        let b = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec()]);
        let sim = TrieDifference::structural_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_trie_diff_structural_similarity_disjoint() {
        let a = NGramTrie::from_ngrams(&[b"abc".to_vec()]);
        let b = NGramTrie::from_ngrams(&[b"xyz".to_vec()]);
        let sim = TrieDifference::structural_similarity(&a, &b);
        assert!((sim - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_trie_diff_edit_distance() {
        let a = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec()]);
        let b = NGramTrie::from_ngrams(&[b"def".to_vec(), b"ghi".to_vec()]);
        let dist = TrieDifference::edit_distance(&a, &b);
        assert_eq!(dist, 2); // abc removed, ghi added
    }

    #[test]
    fn test_trie_diff_containment() {
        let a = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec()]);
        let b = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec(), b"ghi".to_vec()]);
        let c = TrieDifference::containment(&a, &b);
        assert!((c - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_trie_diff_summary() {
        let a = NGramTrie::from_ngrams(&[b"abc".to_vec()]);
        let b = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec()]);
        let s = TrieDifference::diff_summary(&a, &b);
        assert!(s.contains("added"));
        assert!(s.contains("removed"));
    }

    // =====================================================================
    // NamedBatchTrie tests
    // =====================================================================

    #[test]
    fn test_named_batch_trie_new() {
        let batch = NamedBatchTrie::new();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
    }

    #[test]
    fn test_named_batch_trie_add() {
        let mut batch = NamedBatchTrie::new();
        let t1 = NGramTrie::from_ngrams(&[b"abc".to_vec()]);
        let t2 = NGramTrie::from_ngrams(&[b"def".to_vec()]);
        batch.add("first", t1);
        batch.add("second", t2);
        assert_eq!(batch.len(), 2);
        assert_eq!(batch.name_at(0), Some("first"));
        assert_eq!(batch.name_at(1), Some("second"));
    }

    #[test]
    fn test_named_batch_trie_combined() {
        let mut batch = NamedBatchTrie::new();
        batch.add("a", NGramTrie::from_ngrams(&[b"abc".to_vec()]));
        batch.add("b", NGramTrie::from_ngrams(&[b"def".to_vec()]));
        let combined = batch.combined_trie();
        assert_eq!(combined.len(), 2);
    }

    #[test]
    fn test_named_batch_trie_pairwise_intersections() {
        let mut batch = NamedBatchTrie::new();
        batch.add("a", NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec()]));
        batch.add("b", NGramTrie::from_ngrams(&[b"def".to_vec(), b"ghi".to_vec()]));
        let ints = batch.pairwise_intersections();
        assert_eq!(ints.len(), 1);
        assert_eq!(ints[0].1, 1); // "def" is the intersection
    }

    #[test]
    fn test_named_batch_trie_pairwise_jaccard() {
        let mut batch = NamedBatchTrie::new();
        batch.add("a", NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec()]));
        batch.add("b", NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec()]));
        let jaccards = batch.pairwise_jaccard();
        assert_eq!(jaccards.len(), 1);
        assert!((jaccards[0].1 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_named_batch_trie_find_unique_keys() {
        let mut batch = NamedBatchTrie::new();
        batch.add("a", NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec()]));
        batch.add("b", NGramTrie::from_ngrams(&[b"def".to_vec(), b"ghi".to_vec()]));
        let unique_a = batch.find_unique_keys(0);
        assert_eq!(unique_a.len(), 1);
        assert_eq!(unique_a[0], b"abc".to_vec());
    }

    #[test]
    fn test_named_batch_trie_total_keys() {
        let mut batch = NamedBatchTrie::new();
        batch.add("a", NGramTrie::from_ngrams(&[b"abc".to_vec()]));
        batch.add("b", NGramTrie::from_ngrams(&[b"def".to_vec(), b"ghi".to_vec()]));
        assert_eq!(batch.total_keys(), 3);
    }

    #[test]
    fn test_named_batch_trie_overlap_matrix() {
        let mut batch = NamedBatchTrie::new();
        batch.add("a", NGramTrie::from_ngrams(&[b"abc".to_vec()]));
        batch.add("b", NGramTrie::from_ngrams(&[b"abc".to_vec()]));
        let matrix = batch.overlap_matrix();
        assert_eq!(matrix.len(), 2);
        assert!((matrix[0][1] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_named_batch_trie_summary() {
        let mut batch = NamedBatchTrie::new();
        batch.add("a", NGramTrie::from_ngrams(&[b"abc".to_vec()]));
        let s = batch.summary();
        assert!(s.contains("1 tries"));
    }

    // =====================================================================
    // TrieSerializer tests
    // =====================================================================

    #[test]
    fn test_serializer_binary_roundtrip() {
        let trie = NGramTrie::from_ngrams(&[b"hello".to_vec(), b"world".to_vec()]);
        let bytes = TrieSerializer::serialize_binary(&trie);
        let recovered = TrieSerializer::deserialize_binary(&bytes).unwrap();
        assert_eq!(recovered.len(), 2);
        assert!(recovered.contains(b"hello"));
        assert!(recovered.contains(b"world"));
    }

    #[test]
    fn test_serializer_json() {
        let trie = NGramTrie::from_ngrams(&[b"ab".to_vec()]);
        let json = TrieSerializer::serialize_json(&trie);
        assert!(json.contains("\"terminal\""));
        assert!(json.contains("\"children\""));
    }

    #[test]
    fn test_serializer_estimate_size() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec()]);
        let estimated = TrieSerializer::estimate_serialized_size(&trie);
        let actual = TrieSerializer::serialize_binary(&trie).len();
        assert_eq!(estimated, actual);
    }

    #[test]
    fn test_serializer_verify_roundtrip() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec()]);
        assert!(TrieSerializer::verify_roundtrip(&trie));
    }

    #[test]
    fn test_serializer_hex_roundtrip() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec()]);
        let hex = TrieSerializer::serialize_hex(&trie);
        let recovered = TrieSerializer::deserialize_hex(&hex).unwrap();
        assert_eq!(recovered.len(), 1);
        assert!(recovered.contains(b"abc"));
    }

    #[test]
    fn test_serializer_keys() {
        let trie = NGramTrie::from_ngrams(&[b"ab".to_vec(), b"cd".to_vec()]);
        let keys_str = TrieSerializer::serialize_keys(&trie);
        assert!(!keys_str.is_empty());
        assert!(keys_str.contains('\n') || trie.len() == 1);
    }

    #[test]
    fn test_serializer_checksum_deterministic() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec()]);
        let c1 = TrieSerializer::checksum(&trie);
        let c2 = TrieSerializer::checksum(&trie);
        assert_eq!(c1, c2);
    }

    // =====================================================================
    // TrieVisualizer tests
    // =====================================================================

    #[test]
    fn test_visualizer_to_dot() {
        let trie = NGramTrie::from_ngrams(&[b"ab".to_vec(), b"ac".to_vec()]);
        let dot = TrieVisualizer::to_dot(&trie);
        assert!(dot.contains("digraph"));
    }

    #[test]
    fn test_visualizer_to_ascii() {
        let trie = NGramTrie::from_ngrams(&[b"ab".to_vec(), b"ac".to_vec()]);
        let ascii = TrieVisualizer::to_ascii(&trie, 5);
        assert!(ascii.contains("(root)"));
        assert!(!ascii.is_empty());
    }

    #[test]
    fn test_visualizer_to_json_tree() {
        let trie = NGramTrie::from_ngrams(&[b"ab".to_vec()]);
        let json = TrieVisualizer::to_json_tree(&trie);
        assert!(json.contains("\"terminal\""));
    }

    #[test]
    fn test_visualizer_depth_bar_chart() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"abd".to_vec()]);
        let chart = TrieVisualizer::depth_bar_chart(&trie);
        assert!(!chart.is_empty());
        assert!(chart.contains("d"));
    }

    #[test]
    fn test_visualizer_key_listing() {
        let trie = NGramTrie::from_ngrams(&[b"ab".to_vec(), b"cd".to_vec()]);
        let listing = TrieVisualizer::key_listing(&trie);
        assert!(!listing.is_empty());
    }

    #[test]
    fn test_visualizer_compact_to_dot() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"abd".to_vec()]);
        let dot = TrieVisualizer::compact_to_dot(&trie);
        assert!(dot.contains("digraph"));
    }

    // =====================================================================
    // TrieWalker tests
    // =====================================================================

    #[test]
    fn test_walker_dfs() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"abd".to_vec(), b"xyz".to_vec()]);
        let result = TrieWalker::dfs(&trie);
        assert_eq!(result.len(), 3);
        for (key, depth) in &result {
            assert_eq!(*depth, key.len());
        }
    }

    #[test]
    fn test_walker_bfs() {
        let trie = NGramTrie::from_ngrams(&[b"a".to_vec(), b"ab".to_vec(), b"abc".to_vec()]);
        let result = TrieWalker::bfs(&trie);
        assert_eq!(result.len(), 3);
        // BFS should return shorter keys first
        assert!(result[0].1 <= result[1].1);
    }

    #[test]
    fn test_walker_level_order() {
        let trie = NGramTrie::from_ngrams(&[b"a".to_vec(), b"ab".to_vec(), b"abc".to_vec()]);
        let levels = TrieWalker::level_order(&trie);
        assert!(!levels.is_empty());
        assert!(levels[1].len() >= 1); // "a" at depth 1
    }

    #[test]
    fn test_walker_leaves() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"abd".to_vec()]);
        let leaves = TrieWalker::leaves(&trie);
        assert_eq!(leaves.len(), 2);
    }

    #[test]
    fn test_walker_internal_nodes() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"abd".to_vec()]);
        let internal = TrieWalker::internal_nodes(&trie);
        assert!(internal > 0);
    }

    #[test]
    fn test_walker_longest_key() {
        let trie = NGramTrie::from_ngrams(&[b"a".to_vec(), b"ab".to_vec(), b"abcdef".to_vec()]);
        let longest = TrieWalker::longest_key(&trie);
        assert_eq!(longest, b"abcdef".to_vec());
    }

    #[test]
    fn test_walker_shortest_key() {
        let trie = NGramTrie::from_ngrams(&[b"a".to_vec(), b"ab".to_vec(), b"abcdef".to_vec()]);
        let shortest = TrieWalker::shortest_key(&trie);
        assert_eq!(shortest, b"a".to_vec());
    }

    #[test]
    fn test_walker_shortest_key_empty() {
        let trie = NGramTrie::new();
        let shortest = TrieWalker::shortest_key(&trie);
        assert!(shortest.is_empty());
    }

    #[test]
    fn test_walker_nodes_at_depth() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"abd".to_vec()]);
        let at_0 = TrieWalker::nodes_at_depth(&trie, 0);
        assert_eq!(at_0, 1); // root
        let at_1 = TrieWalker::nodes_at_depth(&trie, 1);
        assert_eq!(at_1, 1); // 'a'
    }

    #[test]
    fn test_walker_keys_of_length() {
        let trie = NGramTrie::from_ngrams(&[
            b"a".to_vec(),
            b"ab".to_vec(),
            b"abc".to_vec(),
            b"xy".to_vec(),
        ]);
        let keys_2 = TrieWalker::keys_of_length(&trie, 2);
        assert_eq!(keys_2.len(), 2); // "ab" and "xy"
    }

    #[test]
    fn test_walker_total_path_length() {
        let trie = NGramTrie::from_ngrams(&[b"ab".to_vec(), b"cd".to_vec()]);
        let tpl = TrieWalker::total_path_length(&trie);
        assert_eq!(tpl, 4); // 2 + 2
    }

    #[test]
    fn test_walker_average_key_depth() {
        let trie = NGramTrie::from_ngrams(&[b"ab".to_vec(), b"cd".to_vec()]);
        let avg = TrieWalker::average_key_depth(&trie);
        assert!((avg - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_walker_has_prefix_keys() {
        let trie = NGramTrie::from_ngrams(&[b"ab".to_vec(), b"abc".to_vec()]);
        assert!(TrieWalker::has_prefix_keys(&trie));
    }

    #[test]
    fn test_walker_no_prefix_keys() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec()]);
        assert!(!TrieWalker::has_prefix_keys(&trie));
    }

    #[test]
    fn test_walker_key_depth() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec()]);
        assert_eq!(TrieWalker::key_depth(&trie, b"abc"), Some(3));
        assert_eq!(TrieWalker::key_depth(&trie, b"ab"), None);
        assert_eq!(TrieWalker::key_depth(&trie, b"xyz"), None);
    }

    #[test]
    fn test_walker_filter_keys() {
        let trie = NGramTrie::from_ngrams(&[
            b"ab".to_vec(),
            b"abc".to_vec(),
            b"xyz".to_vec(),
        ]);
        let filtered = TrieWalker::filter_keys(&trie, |k| k.len() > 2);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0], b"abc".to_vec());
    }

    #[test]
    fn test_walker_sorted_prefix_search() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"abd".to_vec(), b"xyz".to_vec()]);
        let results = TrieWalker::sorted_prefix_search(&trie, b"ab");
        assert_eq!(results.len(), 2);
        assert!(results[0] <= results[1]);
    }

    // =====================================================================
    // TrieMerger tests
    // =====================================================================

    #[test]
    fn test_merger_merge() {
        let a = NGramTrie::from_ngrams(&[b"abc".to_vec()]);
        let b = NGramTrie::from_ngrams(&[b"def".to_vec()]);
        let merged = TrieMerger::merge(&a, &b);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn test_merger_merge_all() {
        let a = NGramTrie::from_ngrams(&[b"a".to_vec()]);
        let b = NGramTrie::from_ngrams(&[b"b".to_vec()]);
        let c = NGramTrie::from_ngrams(&[b"c".to_vec()]);
        let merged = TrieMerger::merge_all(&[&a, &b, &c]);
        assert_eq!(merged.len(), 3);
    }

    #[test]
    fn test_merger_intersect_all() {
        let a = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec()]);
        let b = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"ghi".to_vec()]);
        let c = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"jkl".to_vec()]);
        let inter = TrieMerger::intersect_all(&[&a, &b, &c]);
        assert_eq!(inter.len(), 1);
        assert!(inter.contains(b"abc"));
    }

    #[test]
    fn test_merger_core_keys() {
        let a = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec()]);
        let b = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"ghi".to_vec()]);
        let c = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"jkl".to_vec()]);
        let core = TrieMerger::core_keys(&[&a, &b, &c], 1.0);
        assert_eq!(core.len(), 1);
        assert!(core.contains(b"abc"));
    }

    #[test]
    fn test_merger_unique_keys() {
        let a = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec()]);
        let b = NGramTrie::from_ngrams(&[b"def".to_vec(), b"ghi".to_vec()]);
        let unique = TrieMerger::unique_keys(&[&a, &b]);
        assert_eq!(unique.len(), 2); // abc, ghi
    }

    // =====================================================================
    // TrieComparator tests
    // =====================================================================

    #[test]
    fn test_comparator_identical() {
        let a = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec()]);
        let b = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec()]);
        let report = TrieComparator::compare(&a, &b);
        assert!((report.jaccard - 1.0).abs() < 1e-9);
        assert_eq!(report.intersection_size, 2);
    }

    #[test]
    fn test_comparator_disjoint() {
        let a = NGramTrie::from_ngrams(&[b"abc".to_vec()]);
        let b = NGramTrie::from_ngrams(&[b"xyz".to_vec()]);
        let report = TrieComparator::compare(&a, &b);
        assert!((report.jaccard - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_comparator_summary() {
        let a = NGramTrie::from_ngrams(&[b"abc".to_vec()]);
        let b = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec()]);
        let report = TrieComparator::compare(&a, &b);
        let s = report.summary();
        assert!(s.contains("Jaccard"));
    }

    #[test]
    fn test_comparator_to_json() {
        let a = NGramTrie::from_ngrams(&[b"abc".to_vec()]);
        let b = NGramTrie::from_ngrams(&[b"def".to_vec()]);
        let report = TrieComparator::compare(&a, &b);
        let json = report.to_json();
        assert!(json.contains("\"jaccard\""));
        assert!(json.contains("\"keys_a\""));
    }

    // =====================================================================
    // TrieCheckpointer tests
    // =====================================================================

    #[test]
    fn test_checkpoint_roundtrip() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec(), b"ghi".to_vec()]);
        let cp = TrieCheckpointer::checkpoint(&trie);
        let restored = TrieCheckpointer::restore(&cp);
        assert_eq!(restored.len(), 3);
        assert!(restored.contains(b"abc"));
        assert!(restored.contains(b"def"));
        assert!(restored.contains(b"ghi"));
    }

    #[test]
    fn test_checkpoint_node_count() {
        let trie = NGramTrie::from_ngrams(&[b"ab".to_vec(), b"ac".to_vec()]);
        let cp = TrieCheckpointer::checkpoint(&trie);
        assert_eq!(cp.node_count, trie.total_nodes);
    }

    #[test]
    fn test_checkpoint_timestamp_nonzero() {
        let trie = NGramTrie::from_ngrams(&[b"x".to_vec()]);
        let cp = TrieCheckpointer::checkpoint(&trie);
        assert!(cp.timestamp > 0);
    }

    #[test]
    fn test_checkpoint_empty_trie() {
        let trie = NGramTrie::new();
        let cp = TrieCheckpointer::checkpoint(&trie);
        let restored = TrieCheckpointer::restore(&cp);
        assert_eq!(restored.len(), 0);
    }

    #[test]
    fn test_diff_checkpoints_added() {
        let t1 = NGramTrie::from_ngrams(&[b"abc".to_vec()]);
        let t2 = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec()]);
        let cp1 = TrieCheckpointer::checkpoint(&t1);
        let cp2 = TrieCheckpointer::checkpoint(&t2);
        let diff = TrieCheckpointer::diff_checkpoints(&cp1, &cp2);
        assert_eq!(diff.added_keys, vec![b"def".to_vec()]);
        assert!(diff.removed_keys.is_empty());
        assert_eq!(diff.num_changes, 1);
    }

    #[test]
    fn test_diff_checkpoints_removed() {
        let t1 = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec()]);
        let t2 = NGramTrie::from_ngrams(&[b"abc".to_vec()]);
        let cp1 = TrieCheckpointer::checkpoint(&t1);
        let cp2 = TrieCheckpointer::checkpoint(&t2);
        let diff = TrieCheckpointer::diff_checkpoints(&cp1, &cp2);
        assert!(diff.added_keys.is_empty());
        assert_eq!(diff.removed_keys, vec![b"def".to_vec()]);
        assert_eq!(diff.num_changes, 1);
    }

    #[test]
    fn test_diff_checkpoints_no_change() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec()]);
        let cp = TrieCheckpointer::checkpoint(&trie);
        let diff = TrieCheckpointer::diff_checkpoints(&cp, &cp);
        assert_eq!(diff.num_changes, 0);
    }

    // =====================================================================
    // TrieIndex tests
    // =====================================================================

    #[test]
    fn test_depth_index() {
        let trie = NGramTrie::from_ngrams(&[
            b"a".to_vec(), b"ab".to_vec(), b"abc".to_vec(), b"xy".to_vec(),
        ]);
        let idx = TrieIndex::build_depth_index(&trie);
        assert_eq!(idx.get(&1).map(|v| v.len()).unwrap_or(0), 1); // "a"
        assert_eq!(idx.get(&2).map(|v| v.len()).unwrap_or(0), 2); // "ab", "xy"
        assert_eq!(idx.get(&3).map(|v| v.len()).unwrap_or(0), 1); // "abc"
    }

    #[test]
    fn test_fanout_index() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"abd".to_vec()]);
        let idx = TrieIndex::build_fanout_index(&trie);
        // Terminal nodes for "abc" and "abd" are leaves => fanout 0
        assert!(idx.contains_key(&0));
        assert_eq!(idx.get(&0).map(|v| v.len()).unwrap_or(0), 2);
    }

    #[test]
    fn test_keys_at_depth() {
        let trie = NGramTrie::from_ngrams(&[
            b"a".to_vec(), b"ab".to_vec(), b"abc".to_vec(),
        ]);
        let depth2 = TrieIndex::keys_at_depth(&trie, 2);
        assert_eq!(depth2.len(), 1);
        assert_eq!(depth2[0], b"ab".to_vec());
    }

    #[test]
    fn test_keys_at_depth_none() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec()]);
        let depth10 = TrieIndex::keys_at_depth(&trie, 10);
        assert!(depth10.is_empty());
    }

    #[test]
    fn test_subtree_size() {
        let trie = NGramTrie::from_ngrams(&[
            b"abc".to_vec(), b"abd".to_vec(), b"xyz".to_vec(),
        ]);
        assert_eq!(TrieIndex::subtree_size(&trie, b"ab"), 2);
        assert_eq!(TrieIndex::subtree_size(&trie, b"x"), 1);
        assert_eq!(TrieIndex::subtree_size(&trie, b"zz"), 0);
    }

    #[test]
    fn test_subtree_size_full() {
        let trie = NGramTrie::from_ngrams(&[b"a".to_vec(), b"b".to_vec()]);
        assert_eq!(TrieIndex::subtree_size(&trie, b""), 2);
    }

    // =====================================================================
    // TriePartitioner tests
    // =====================================================================

    #[test]
    fn test_partition_roundtrip() {
        let trie = NGramTrie::from_ngrams(&[
            b"abc".to_vec(), b"def".to_vec(), b"ghi".to_vec(), b"xyz".to_vec(),
        ]);
        let parts = TriePartitioner::partition(&trie, 2);
        let merged = TriePartitioner::merge_partitions(&parts);
        assert_eq!(merged.len(), trie.len());
        for key in trie.iter() {
            assert!(merged.contains(&key));
        }
    }

    #[test]
    fn test_partition_single() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec()]);
        let parts = TriePartitioner::partition(&trie, 1);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].len(), 2);
    }

    #[test]
    fn test_partition_zero() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec()]);
        let parts = TriePartitioner::partition(&trie, 0);
        assert!(parts.is_empty());
    }

    #[test]
    fn test_partition_more_parts_than_keys() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec()]);
        let parts = TriePartitioner::partition(&trie, 10);
        let merged = TriePartitioner::merge_partitions(&parts);
        assert_eq!(merged.len(), 1);
        assert!(merged.contains(b"abc"));
    }

    #[test]
    fn test_partition_empty_trie() {
        let trie = NGramTrie::new();
        let parts = TriePartitioner::partition(&trie, 3);
        let merged = TriePartitioner::merge_partitions(&parts);
        assert_eq!(merged.len(), 0);
    }

    #[test]
    fn test_merge_partitions_empty_list() {
        let merged = TriePartitioner::merge_partitions(&[]);
        assert_eq!(merged.len(), 0);
    }

    #[test]
    fn test_balanced_partition_basic() {
        let trie = NGramTrie::from_ngrams(&[
            b"a".to_vec(), b"b".to_vec(), b"c".to_vec(), b"d".to_vec(), b"e".to_vec(),
        ]);
        let parts = TriePartitioner::balanced_partition(&trie, 2);
        let total: usize = parts.iter().map(|p| p.len()).sum();
        assert_eq!(total, 5);
        // Each partition should have at most 2 keys
        for part in &parts {
            assert!(part.len() <= 2);
        }
    }

    #[test]
    fn test_balanced_partition_single_key() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec()]);
        let parts = TriePartitioner::balanced_partition(&trie, 10);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].len(), 1);
    }

    #[test]
    fn test_balanced_partition_empty() {
        let trie = NGramTrie::new();
        let parts = TriePartitioner::balanced_partition(&trie, 5);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].len(), 0);
    }

    #[test]
    fn test_balanced_partition_max_zero() {
        let trie = NGramTrie::from_ngrams(&[b"a".to_vec(), b"b".to_vec()]);
        let parts = TriePartitioner::balanced_partition(&trie, 0);
        // max_per_part=0 is treated as 1
        let total: usize = parts.iter().map(|p| p.len()).sum();
        assert_eq!(total, 2);
    }

    #[test]
    fn test_balanced_partition_roundtrip() {
        let trie = NGramTrie::from_ngrams(&[
            b"abc".to_vec(), b"def".to_vec(), b"ghi".to_vec(),
        ]);
        let parts = TriePartitioner::balanced_partition(&trie, 2);
        let merged = TriePartitioner::merge_partitions(&parts);
        assert_eq!(merged.len(), trie.len());
        for key in trie.iter() {
            assert!(merged.contains(&key));
        }
    }

    #[test]
    fn test_partition_preserves_all_keys() {
        let keys: Vec<Vec<u8>> = (0u8..=20).map(|b| vec![b, b + 1, b + 2]).collect();
        let trie = NGramTrie::from_ngrams(&keys);
        let parts = TriePartitioner::partition(&trie, 4);
        let merged = TriePartitioner::merge_partitions(&parts);
        assert_eq!(merged.len(), trie.len());
    }

    #[test]
    fn test_diff_checkpoints_both_directions() {
        let t1 = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec()]);
        let t2 = NGramTrie::from_ngrams(&[b"def".to_vec(), b"ghi".to_vec()]);
        let cp1 = TrieCheckpointer::checkpoint(&t1);
        let cp2 = TrieCheckpointer::checkpoint(&t2);
        let diff = TrieCheckpointer::diff_checkpoints(&cp1, &cp2);
        assert_eq!(diff.added_keys, vec![b"ghi".to_vec()]);
        assert_eq!(diff.removed_keys, vec![b"abc".to_vec()]);
        assert_eq!(diff.num_changes, 2);
    }

    // =====================================================================
    // TrieCheckpoint / TrieCheckpointDiff method tests
    // =====================================================================

    #[test]
    fn test_checkpoint_data_size() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec()]);
        let cp = TrieCheckpointer::checkpoint(&trie);
        assert!(cp.data_size() > 0);
    }

    #[test]
    fn test_checkpoint_is_older_than() {
        let t1 = NGramTrie::from_ngrams(&[b"a".to_vec()]);
        let cp1 = TrieCheckpointer::checkpoint(&t1);
        // same timestamp in practice; just verify the method works
        let cp2 = TrieCheckpoint {
            serialized_data: cp1.serialized_data.clone(),
            node_count: cp1.node_count,
            timestamp: cp1.timestamp + 100,
        };
        assert!(cp1.is_older_than(&cp2));
        assert!(!cp2.is_older_than(&cp1));
    }

    #[test]
    fn test_checkpoint_diff_is_empty() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec()]);
        let cp = TrieCheckpointer::checkpoint(&trie);
        let diff = TrieCheckpointer::diff_checkpoints(&cp, &cp);
        assert!(diff.is_empty());
    }

    #[test]
    fn test_checkpoint_diff_summary() {
        let t1 = NGramTrie::from_ngrams(&[b"abc".to_vec()]);
        let t2 = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec()]);
        let cp1 = TrieCheckpointer::checkpoint(&t1);
        let cp2 = TrieCheckpointer::checkpoint(&t2);
        let diff = TrieCheckpointer::diff_checkpoints(&cp1, &cp2);
        let s = diff.summary();
        assert!(s.contains("1 added"));
        assert!(s.contains("0 removed"));
    }

    // =====================================================================
    // TrieIndex extra method tests
    // =====================================================================

    #[test]
    fn test_build_prefix_index() {
        let trie = NGramTrie::from_ngrams(&[
            b"abc".to_vec(), b"abd".to_vec(), b"xyz".to_vec(),
        ]);
        let idx = TrieIndex::build_prefix_index(&trie, 2);
        assert_eq!(idx.get(&b"ab".to_vec()).map(|v| v.len()).unwrap_or(0), 2);
        assert_eq!(idx.get(&b"xy".to_vec()).map(|v| v.len()).unwrap_or(0), 1);
    }

    #[test]
    fn test_depth_count() {
        let trie = NGramTrie::from_ngrams(&[
            b"a".to_vec(), b"ab".to_vec(), b"abc".to_vec(),
        ]);
        assert_eq!(TrieIndex::depth_count(&trie), 3);
    }

    #[test]
    fn test_depth_count_uniform() {
        let trie = NGramTrie::from_ngrams(&[
            b"abc".to_vec(), b"def".to_vec(), b"ghi".to_vec(),
        ]);
        assert_eq!(TrieIndex::depth_count(&trie), 1);
    }

    #[test]
    fn test_shortest_keys() {
        let trie = NGramTrie::from_ngrams(&[
            b"a".to_vec(), b"ab".to_vec(), b"abc".to_vec(),
        ]);
        let shortest = TrieIndex::shortest_keys(&trie);
        assert_eq!(shortest.len(), 1);
        assert_eq!(shortest[0], b"a".to_vec());
    }

    #[test]
    fn test_longest_keys() {
        let trie = NGramTrie::from_ngrams(&[
            b"a".to_vec(), b"ab".to_vec(), b"abc".to_vec(),
        ]);
        let longest = TrieIndex::longest_keys(&trie);
        assert_eq!(longest.len(), 1);
        assert_eq!(longest[0], b"abc".to_vec());
    }

    #[test]
    fn test_shortest_longest_empty() {
        let trie = NGramTrie::new();
        assert!(TrieIndex::shortest_keys(&trie).is_empty());
        assert!(TrieIndex::longest_keys(&trie).is_empty());
    }

    #[test]
    fn test_fanout_index_nonleaf_terminal() {
        // Create a trie where "ab" is terminal but also has child "abc"
        let trie = NGramTrie::from_ngrams(&[b"ab".to_vec(), b"abc".to_vec()]);
        let idx = TrieIndex::build_fanout_index(&trie);
        // "ab" terminal node has 1 child ('c'), so fanout=1
        assert!(idx.get(&1).unwrap().contains(&b"ab".to_vec()));
        // "abc" terminal node is a leaf => fanout=0
        assert!(idx.get(&0).unwrap().contains(&b"abc".to_vec()));
    }

    // =====================================================================
    // TriePartitioner extra tests
    // =====================================================================

    #[test]
    fn test_partition_by_depth() {
        let trie = NGramTrie::from_ngrams(&[
            b"a".to_vec(), b"ab".to_vec(), b"abc".to_vec(), b"xy".to_vec(),
        ]);
        let parts = TriePartitioner::partition_by_depth(&trie);
        assert_eq!(parts.len(), 3); // depths 1, 2, 3
        let total: usize = parts.iter().map(|p| p.len()).sum();
        assert_eq!(total, 4);
    }

    #[test]
    fn test_partition_by_depth_single_depth() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec()]);
        let parts = TriePartitioner::partition_by_depth(&trie);
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].len(), 2);
    }

    #[test]
    fn test_count_nonempty_partitions() {
        let trie = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec()]);
        let ne = TriePartitioner::count_nonempty_partitions(&trie, 5);
        assert!(ne >= 1 && ne <= 5);
    }

    #[test]
    fn test_partition_large_spread() {
        // Keys whose first bytes span a wide range
        let keys = vec![
            vec![0u8, 1, 2],
            vec![64u8, 65, 66],
            vec![128u8, 129, 130],
            vec![200u8, 201, 202],
        ];
        let trie = NGramTrie::from_ngrams(&keys);
        let parts = TriePartitioner::partition(&trie, 4);
        let merged = TriePartitioner::merge_partitions(&parts);
        assert_eq!(merged.len(), 4);
        for key in &keys {
            assert!(merged.contains(key));
        }
    }

    #[test]
    fn test_balanced_partition_exact_multiple() {
        let keys: Vec<Vec<u8>> = (0u8..6).map(|b| vec![b]).collect();
        let trie = NGramTrie::from_ngrams(&keys);
        let parts = TriePartitioner::balanced_partition(&trie, 3);
        // 6 keys / 3 per part = 2 partitions
        assert_eq!(parts.len(), 2);
        for part in &parts {
            assert_eq!(part.len(), 3);
        }
    }


}
