//! ID generation and management for SoniType compiler entities.
//!
//! Provides strongly-typed identifiers for nodes, streams, mappings, and graphs,
//! backed by atomic counters for thread-safe generation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

// ---------------------------------------------------------------------------
// Atomic counter shared across all ID generators
// ---------------------------------------------------------------------------

static GLOBAL_COUNTER: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    GLOBAL_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Reset the global counter (useful for deterministic tests).
pub fn reset_id_counter() {
    GLOBAL_COUNTER.store(1, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// Macro to reduce boilerplate for ID newtypes
// ---------------------------------------------------------------------------

macro_rules! define_id {
    ($(#[$meta:meta])* $name:ident, $prefix:expr) => {
        $(#[$meta])*
        #[derive(
            Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        pub struct $name(pub u64);

        impl $name {
            /// Create an ID with a specific numeric value.
            pub fn new(val: u64) -> Self {
                Self(val)
            }

            /// Generate a fresh, globally unique ID.
            pub fn generate() -> Self {
                Self(next_id())
            }

            /// Return the raw `u64` value.
            pub fn value(self) -> u64 {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}-{}", $prefix, self.0)
            }
        }

        impl From<u64> for $name {
            fn from(v: u64) -> Self {
                Self(v)
            }
        }

        impl From<$name> for u64 {
            fn from(id: $name) -> u64 {
                id.0
            }
        }
    };
}

define_id!(
    /// Unique identifier for an audio processing node in the render graph.
    NodeId,
    "node"
);

define_id!(
    /// Unique identifier for an audio stream (a continuous sonification voice).
    StreamId,
    "stream"
);

define_id!(
    /// Unique identifier for a data-to-audio mapping specification.
    MappingId,
    "map"
);

define_id!(
    /// Unique identifier for an audio render graph.
    GraphId,
    "graph"
);

// ---------------------------------------------------------------------------
// ID Registry
// ---------------------------------------------------------------------------

/// A thread-safe registry that associates IDs of type `K` with values of type `V`.
#[derive(Debug)]
pub struct IdRegistry<K: std::hash::Hash + Eq + Copy, V> {
    inner: Arc<RwLock<HashMap<K, V>>>,
}

impl<K: std::hash::Hash + Eq + Copy, V> Clone for IdRegistry<K, V> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<K: std::hash::Hash + Eq + Copy, V> Default for IdRegistry<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: std::hash::Hash + Eq + Copy, V> IdRegistry<K, V> {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a value under the given key. Returns the previous value if any.
    pub fn register(&self, key: K, value: V) -> Option<V> {
        self.inner.write().unwrap().insert(key, value)
    }

    /// Remove a value from the registry, returning it if present.
    pub fn unregister(&self, key: &K) -> Option<V> {
        self.inner.write().unwrap().remove(key)
    }

    /// Check whether a key is present.
    pub fn contains(&self, key: &K) -> bool {
        self.inner.read().unwrap().contains_key(key)
    }

    /// Number of entries in the registry.
    pub fn len(&self) -> usize {
        self.inner.read().unwrap().len()
    }

    /// Returns `true` if registry is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.read().unwrap().is_empty()
    }

    /// Return all registered keys.
    pub fn keys(&self) -> Vec<K> {
        self.inner.read().unwrap().keys().copied().collect()
    }
}

impl<K: std::hash::Hash + Eq + Copy, V: Clone> IdRegistry<K, V> {
    /// Look up a value by key, returning a clone.
    pub fn lookup(&self, key: &K) -> Option<V> {
        self.inner.read().unwrap().get(key).cloned()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_id_display() {
        let id = NodeId::new(42);
        assert_eq!(format!("{id}"), "node-42");
    }

    #[test]
    fn stream_id_display() {
        let id = StreamId::new(7);
        assert_eq!(format!("{id}"), "stream-7");
    }

    #[test]
    fn mapping_id_display() {
        let id = MappingId::new(99);
        assert_eq!(format!("{id}"), "map-99");
    }

    #[test]
    fn graph_id_display() {
        let id = GraphId::new(1);
        assert_eq!(format!("{id}"), "graph-1");
    }

    #[test]
    fn generate_produces_unique_ids() {
        let a = NodeId::generate();
        let b = NodeId::generate();
        let c = NodeId::generate();
        assert_ne!(a, b);
        assert_ne!(b, c);
    }

    #[test]
    fn id_ordering() {
        let a = StreamId::new(1);
        let b = StreamId::new(5);
        assert!(a < b);
    }

    #[test]
    fn id_hash_works_in_hashmap() {
        let mut map = HashMap::new();
        let id = MappingId::new(10);
        map.insert(id, "hello");
        assert_eq!(map.get(&id), Some(&"hello"));
    }

    #[test]
    fn id_from_u64_roundtrip() {
        let val: u64 = 123;
        let id = GraphId::from(val);
        let back: u64 = id.into();
        assert_eq!(val, back);
    }

    #[test]
    fn registry_basic_operations() {
        let reg: IdRegistry<NodeId, String> = IdRegistry::new();
        assert!(reg.is_empty());

        let id = NodeId::new(1);
        reg.register(id, "osc".to_string());
        assert_eq!(reg.len(), 1);
        assert!(reg.contains(&id));
        assert_eq!(reg.lookup(&id), Some("osc".to_string()));

        reg.unregister(&id);
        assert!(reg.is_empty());
    }

    #[test]
    fn registry_overwrite() {
        let reg: IdRegistry<StreamId, i32> = IdRegistry::new();
        let id = StreamId::new(5);
        reg.register(id, 10);
        let prev = reg.register(id, 20);
        assert_eq!(prev, Some(10));
        assert_eq!(reg.lookup(&id), Some(20));
    }

    #[test]
    fn registry_keys() {
        let reg: IdRegistry<MappingId, &str> = IdRegistry::new();
        let a = MappingId::new(1);
        let b = MappingId::new(2);
        reg.register(a, "x");
        reg.register(b, "y");
        let mut keys = reg.keys();
        keys.sort();
        assert_eq!(keys, vec![a, b]);
    }

    #[test]
    fn serde_roundtrip() {
        let id = NodeId::new(42);
        let json = serde_json::to_string(&id).unwrap();
        let back: NodeId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, back);
    }
}
