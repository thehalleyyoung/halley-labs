//! Lock types and lock table structures.
use serde::{Deserialize, Serialize};
use crate::identifier::*;
use crate::operation::LockMode;
use crate::predicate::Predicate;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockEntry {
    pub id: LockId,
    pub txn_id: TransactionId,
    pub table_id: TableId,
    pub item_id: Option<ItemId>,
    pub predicate: Option<Predicate>,
    pub mode: LockMode,
    pub granted: bool,
    pub timestamp: u64,
    pub granularity: LockGranularity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LockGranularity {
    Row,
    Page,
    Table,
    Key,
    KeyRange,
    Predicate,
}

#[derive(Debug, Clone, Default)]
pub struct LockTable {
    pub locks: Vec<LockEntry>,
    pub wait_for: HashMap<TransactionId, Vec<TransactionId>>,
    next_lock_id: u64,
}

impl LockTable {
    pub fn new() -> Self { Self::default() }

    pub fn acquire(&mut self, txn_id: TransactionId, table_id: TableId, item_id: Option<ItemId>,
                   mode: LockMode, granularity: LockGranularity, timestamp: u64) -> LockResult {
        let conflicts: Vec<LockId> = self.locks.iter()
            .filter(|l| l.granted && l.txn_id != txn_id && l.table_id == table_id
                && l.item_id == item_id && !l.mode.is_compatible_with(mode))
            .map(|l| l.id)
            .collect();

        if conflicts.is_empty() {
            let id = LockId::new(self.next_lock_id);
            self.next_lock_id += 1;
            self.locks.push(LockEntry {
                id, txn_id, table_id, item_id, predicate: None,
                mode, granted: true, timestamp, granularity,
            });
            LockResult::Granted(id)
        } else {
            let holders: Vec<TransactionId> = self.locks.iter()
                .filter(|l| conflicts.contains(&l.id))
                .map(|l| l.txn_id)
                .collect();
            self.wait_for.entry(txn_id).or_default().extend(holders.iter());
            let id = LockId::new(self.next_lock_id);
            self.next_lock_id += 1;
            self.locks.push(LockEntry {
                id, txn_id, table_id, item_id, predicate: None,
                mode, granted: false, timestamp, granularity,
            });
            LockResult::Waiting { lock_id: id, blocked_by: holders }
        }
    }

    pub fn release_all(&mut self, txn_id: TransactionId) -> Vec<LockId> {
        let mut released = Vec::new();
        self.locks.retain(|l| {
            if l.txn_id == txn_id {
                released.push(l.id);
                false
            } else {
                true
            }
        });
        self.wait_for.remove(&txn_id);
        for waiters in self.wait_for.values_mut() {
            waiters.retain(|t| *t != txn_id);
        }
        self.try_grant_waiting();
        released
    }

    fn try_grant_waiting(&mut self) {
        let waiting: Vec<usize> = self.locks.iter().enumerate()
            .filter(|(_, l)| !l.granted)
            .map(|(i, _)| i)
            .collect();
        for idx in waiting {
            let lock = &self.locks[idx];
            let can_grant = !self.locks.iter().any(|other| {
                other.granted && other.txn_id != lock.txn_id
                    && other.table_id == lock.table_id && other.item_id == lock.item_id
                    && !other.mode.is_compatible_with(lock.mode)
            });
            if can_grant {
                self.locks[idx].granted = true;
            }
        }
    }

    pub fn detect_deadlock(&self) -> Option<Vec<TransactionId>> {
        for start in self.wait_for.keys() {
            let mut visited = vec![*start];
            let mut stack = vec![*start];
            while let Some(current) = stack.pop() {
                if let Some(waiting_for) = self.wait_for.get(&current) {
                    for target in waiting_for {
                        if *target == *start {
                            return Some(visited.clone());
                        }
                        if !visited.contains(target) {
                            visited.push(*target);
                            stack.push(*target);
                        }
                    }
                }
            }
        }
        None
    }

    pub fn held_by(&self, txn_id: TransactionId) -> Vec<&LockEntry> {
        self.locks.iter().filter(|l| l.txn_id == txn_id && l.granted).collect()
    }

    pub fn lock_count(&self) -> usize { self.locks.len() }
    pub fn granted_count(&self) -> usize { self.locks.iter().filter(|l| l.granted).count() }
    pub fn waiting_count(&self) -> usize { self.locks.iter().filter(|l| !l.granted).count() }
}

#[derive(Debug, Clone)]
pub enum LockResult {
    Granted(LockId),
    Waiting { lock_id: LockId, blocked_by: Vec<TransactionId> },
    Denied { reason: String },
}

impl LockResult {
    pub fn is_granted(&self) -> bool { matches!(self, Self::Granted(_)) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_lock_acquire_release() {
        let mut lt = LockTable::new();
        let r = lt.acquire(TransactionId::new(1), TableId::new(0), Some(ItemId::new(1)),
            LockMode::Shared, LockGranularity::Row, 0);
        assert!(r.is_granted());
        assert_eq!(lt.lock_count(), 1);
        lt.release_all(TransactionId::new(1));
        assert_eq!(lt.lock_count(), 0);
    }
    #[test]
    fn test_lock_conflict() {
        let mut lt = LockTable::new();
        lt.acquire(TransactionId::new(1), TableId::new(0), Some(ItemId::new(1)),
            LockMode::Exclusive, LockGranularity::Row, 0);
        let r = lt.acquire(TransactionId::new(2), TableId::new(0), Some(ItemId::new(1)),
            LockMode::Shared, LockGranularity::Row, 1);
        assert!(!r.is_granted());
        assert_eq!(lt.waiting_count(), 1);
    }
    #[test]
    fn test_deadlock_detection() {
        let mut lt = LockTable::new();
        lt.wait_for.insert(TransactionId::new(1), vec![TransactionId::new(2)]);
        lt.wait_for.insert(TransactionId::new(2), vec![TransactionId::new(1)]);
        assert!(lt.detect_deadlock().is_some());
    }
    #[test]
    fn test_shared_lock_compatible() {
        let mut lt = LockTable::new();
        lt.acquire(TransactionId::new(1), TableId::new(0), Some(ItemId::new(1)),
            LockMode::Shared, LockGranularity::Row, 0);
        let r = lt.acquire(TransactionId::new(2), TableId::new(0), Some(ItemId::new(1)),
            LockMode::Shared, LockGranularity::Row, 1);
        assert!(r.is_granted());
    }
}
