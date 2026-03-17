//! Multi-version concurrency control (MVCC) version store.
use serde::{Deserialize, Serialize};
use crate::identifier::*;
use crate::value::Value;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionEntry {
    pub version_id: VersionId,
    pub item_id: ItemId,
    pub table_id: TableId,
    pub created_by: TransactionId,
    pub deleted_by: Option<TransactionId>,
    pub created_at: u64,
    pub deleted_at: Option<u64>,
    pub value: Value,
    pub prev_version: Option<VersionId>,
}

#[derive(Debug, Clone, Default)]
pub struct VersionStore {
    versions: HashMap<VersionId, VersionEntry>,
    item_versions: HashMap<(TableId, ItemId), Vec<VersionId>>,
    next_version_id: u64,
}

impl VersionStore {
    pub fn new() -> Self { Self::default() }

    pub fn create_version(&mut self, item_id: ItemId, table_id: TableId,
                          created_by: TransactionId, created_at: u64, value: Value) -> VersionId {
        let vid = VersionId::new(self.next_version_id);
        self.next_version_id += 1;
        let prev = self.item_versions.get(&(table_id, item_id))
            .and_then(|vs| vs.last().copied());
        self.versions.insert(vid, VersionEntry {
            version_id: vid, item_id, table_id, created_by,
            deleted_by: None, created_at, deleted_at: None,
            value, prev_version: prev,
        });
        self.item_versions.entry((table_id, item_id)).or_default().push(vid);
        vid
    }

    pub fn delete_version(&mut self, item_id: ItemId, table_id: TableId,
                          deleted_by: TransactionId, deleted_at: u64) -> Option<VersionId> {
        let versions = self.item_versions.get(&(table_id, item_id))?;
        let latest = *versions.last()?;
        if let Some(entry) = self.versions.get_mut(&latest) {
            entry.deleted_by = Some(deleted_by);
            entry.deleted_at = Some(deleted_at);
        }
        Some(latest)
    }

    pub fn visible_version(&self, item_id: ItemId, table_id: TableId,
                           snapshot_time: u64, committed: &[TransactionId]) -> Option<&VersionEntry> {
        let versions = self.item_versions.get(&(table_id, item_id))?;
        for vid in versions.iter().rev() {
            if let Some(entry) = self.versions.get(vid) {
                if entry.created_at <= snapshot_time && committed.contains(&entry.created_by) {
                    if entry.deleted_at.map_or(true, |dt| dt > snapshot_time) {
                        return Some(entry);
                    }
                }
            }
        }
        None
    }

    pub fn all_versions(&self, item_id: ItemId, table_id: TableId) -> Vec<&VersionEntry> {
        self.item_versions.get(&(table_id, item_id))
            .map(|vs| vs.iter().filter_map(|v| self.versions.get(v)).collect())
            .unwrap_or_default()
    }

    pub fn get_version(&self, vid: VersionId) -> Option<&VersionEntry> {
        self.versions.get(&vid)
    }

    pub fn version_count(&self) -> usize { self.versions.len() }

    pub fn gc_versions(&mut self, oldest_active_snapshot: u64) -> usize {
        let mut removed = 0;
        let to_remove: Vec<VersionId> = self.versions.iter()
            .filter(|(_, v)| v.deleted_at.map_or(false, |dt| dt < oldest_active_snapshot))
            .map(|(id, _)| *id)
            .collect();
        for vid in &to_remove {
            if let Some(entry) = self.versions.remove(vid) {
                if let Some(versions) = self.item_versions.get_mut(&(entry.table_id, entry.item_id)) {
                    versions.retain(|v| v != vid);
                }
                removed += 1;
            }
        }
        removed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_version_store_basic() {
        let mut vs = VersionStore::new();
        let vid = vs.create_version(ItemId::new(1), TableId::new(0),
            TransactionId::new(1), 10, Value::Integer(100));
        assert_eq!(vs.version_count(), 1);
        let entry = vs.get_version(vid).unwrap();
        assert_eq!(entry.value, Value::Integer(100));
    }
    #[test]
    fn test_version_visibility() {
        let mut vs = VersionStore::new();
        let t1 = TransactionId::new(1);
        let t2 = TransactionId::new(2);
        vs.create_version(ItemId::new(1), TableId::new(0), t1, 10, Value::Integer(100));
        vs.create_version(ItemId::new(1), TableId::new(0), t2, 20, Value::Integer(200));
        let visible = vs.visible_version(ItemId::new(1), TableId::new(0), 15, &[t1]);
        assert_eq!(visible.unwrap().value, Value::Integer(100));
        let visible2 = vs.visible_version(ItemId::new(1), TableId::new(0), 25, &[t1, t2]);
        assert_eq!(visible2.unwrap().value, Value::Integer(200));
    }
    #[test]
    fn test_version_deletion() {
        let mut vs = VersionStore::new();
        vs.create_version(ItemId::new(1), TableId::new(0),
            TransactionId::new(1), 10, Value::Integer(100));
        vs.delete_version(ItemId::new(1), TableId::new(0), TransactionId::new(2), 20);
        let visible = vs.visible_version(ItemId::new(1), TableId::new(0), 25,
            &[TransactionId::new(1), TransactionId::new(2)]);
        assert!(visible.is_none());
    }
    #[test]
    fn test_gc() {
        let mut vs = VersionStore::new();
        vs.create_version(ItemId::new(1), TableId::new(0),
            TransactionId::new(1), 10, Value::Integer(100));
        vs.delete_version(ItemId::new(1), TableId::new(0), TransactionId::new(2), 20);
        let removed = vs.gc_versions(30);
        assert_eq!(removed, 1);
        assert_eq!(vs.version_count(), 0);
    }
}
