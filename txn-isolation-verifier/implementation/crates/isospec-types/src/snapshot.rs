//! Snapshot types for MVCC.
use serde::{Deserialize, Serialize};
use crate::identifier::TransactionId;
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub snapshot_time: u64,
    pub txn_id: TransactionId,
    pub active_txns: HashSet<TransactionId>,
    pub committed_txns: HashSet<TransactionId>,
    pub min_active_txn: Option<TransactionId>,
}

impl Snapshot {
    pub fn new(snapshot_time: u64, txn_id: TransactionId) -> Self {
        Self {
            snapshot_time,
            txn_id,
            active_txns: HashSet::new(),
            committed_txns: HashSet::new(),
            min_active_txn: None,
        }
    }

    pub fn is_visible(&self, created_by: TransactionId, created_at: u64) -> bool {
        if created_by == self.txn_id { return true; }
        if self.active_txns.contains(&created_by) { return false; }
        created_at <= self.snapshot_time && self.committed_txns.contains(&created_by)
    }

    pub fn is_concurrent(&self, other_txn: TransactionId) -> bool {
        self.active_txns.contains(&other_txn)
    }

    pub fn add_committed(&mut self, txn_id: TransactionId) {
        self.committed_txns.insert(txn_id);
        self.active_txns.remove(&txn_id);
    }

    pub fn committed_list(&self) -> Vec<TransactionId> {
        self.committed_txns.iter().copied().collect()
    }
}

#[derive(Debug, Clone, Default)]
pub struct SnapshotManager {
    snapshots: std::collections::HashMap<TransactionId, Snapshot>,
    global_timestamp: u64,
}

impl SnapshotManager {
    pub fn new() -> Self { Self::default() }

    pub fn take_snapshot(&mut self, txn_id: TransactionId, active: HashSet<TransactionId>,
                         committed: HashSet<TransactionId>) -> Snapshot {
        self.global_timestamp += 1;
        let snap = Snapshot {
            snapshot_time: self.global_timestamp,
            txn_id,
            active_txns: active,
            committed_txns: committed,
            min_active_txn: None,
        };
        self.snapshots.insert(txn_id, snap.clone());
        snap
    }

    pub fn get_snapshot(&self, txn_id: TransactionId) -> Option<&Snapshot> {
        self.snapshots.get(&txn_id)
    }

    pub fn remove_snapshot(&mut self, txn_id: TransactionId) {
        self.snapshots.remove(&txn_id);
    }

    pub fn current_timestamp(&self) -> u64 { self.global_timestamp }
    pub fn active_snapshot_count(&self) -> usize { self.snapshots.len() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_snapshot_visibility() {
        let mut snap = Snapshot::new(10, TransactionId::new(1));
        snap.committed_txns.insert(TransactionId::new(0));
        snap.active_txns.insert(TransactionId::new(2));
        assert!(snap.is_visible(TransactionId::new(0), 5));
        assert!(!snap.is_visible(TransactionId::new(2), 5));
        assert!(snap.is_visible(TransactionId::new(1), 0));
    }
    #[test]
    fn test_snapshot_manager() {
        let mut mgr = SnapshotManager::new();
        let snap = mgr.take_snapshot(TransactionId::new(1), HashSet::new(), HashSet::new());
        assert_eq!(snap.snapshot_time, 1);
        assert_eq!(mgr.active_snapshot_count(), 1);
        mgr.remove_snapshot(TransactionId::new(1));
        assert_eq!(mgr.active_snapshot_count(), 0);
    }
}
