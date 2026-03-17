use std::collections::{HashMap, HashSet};
use isospec_types::identifier::*;
use isospec_types::isolation::*;
use isospec_types::value::*;
use isospec_types::predicate::Predicate;

// ---------------------------------------------------------------------------
// HistoryEvent
// ---------------------------------------------------------------------------

/// A single logical event in a transaction history.
#[derive(Debug, Clone)]
pub enum HistoryEvent {
    Read { txn: TransactionId, item: ItemId, table: TableId, value: Option<Value> },
    Write { txn: TransactionId, item: ItemId, table: TableId, old_value: Option<Value>, new_value: Value },
    Insert { txn: TransactionId, item: ItemId, table: TableId, values: Vec<(String, Value)> },
    Delete { txn: TransactionId, item: ItemId, table: TableId },
    PredicateRead { txn: TransactionId, table: TableId, predicate: Predicate, matched_items: Vec<ItemId> },
    Commit { txn: TransactionId, timestamp: u64 },
    Abort { txn: TransactionId, reason: Option<String> },
}

impl HistoryEvent {
    /// Returns the transaction that produced this event.
    pub fn txn_id(&self) -> TransactionId {
        match self {
            Self::Read { txn, .. } | Self::Write { txn, .. } | Self::Insert { txn, .. }
            | Self::Delete { txn, .. } | Self::PredicateRead { txn, .. }
            | Self::Commit { txn, .. } | Self::Abort { txn, .. } => *txn,
        }
    }

    /// True for Read, Write, Insert, Delete, PredicateRead.
    pub fn is_data_event(&self) -> bool {
        matches!(self, Self::Read { .. } | Self::Write { .. } | Self::Insert { .. }
            | Self::Delete { .. } | Self::PredicateRead { .. })
    }

    /// True for Commit or Abort.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Commit { .. } | Self::Abort { .. })
    }

    /// True for Write, Insert, or Delete.
    pub fn is_write_event(&self) -> bool {
        matches!(self, Self::Write { .. } | Self::Insert { .. } | Self::Delete { .. })
    }

    /// True for Read or PredicateRead.
    pub fn is_read_event(&self) -> bool {
        matches!(self, Self::Read { .. } | Self::PredicateRead { .. })
    }

    /// Returns the item id if the event targets a single item.
    pub fn item_id(&self) -> Option<ItemId> {
        match self {
            Self::Read { item, .. } | Self::Write { item, .. }
            | Self::Insert { item, .. } | Self::Delete { item, .. } => Some(*item),
            _ => None,
        }
    }

    /// Returns the table id if the event references a table.
    pub fn table_id(&self) -> Option<TableId> {
        match self {
            Self::Read { table, .. } | Self::Write { table, .. } | Self::Insert { table, .. }
            | Self::Delete { table, .. } | Self::PredicateRead { table, .. } => Some(*table),
            _ => None,
        }
    }

    /// Short human-readable label for the event kind.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Read { .. } => "read",
            Self::Write { .. } => "write",
            Self::Insert { .. } => "insert",
            Self::Delete { .. } => "delete",
            Self::PredicateRead { .. } => "predicate_read",
            Self::Commit { .. } => "commit",
            Self::Abort { .. } => "abort",
        }
    }
}

// ---------------------------------------------------------------------------
// TransactionStatus
// ---------------------------------------------------------------------------

/// Lifecycle status of a transaction inside a history.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransactionStatus {
    Active,
    Committed,
    Aborted,
}

// ---------------------------------------------------------------------------
// TransactionInfo
// ---------------------------------------------------------------------------

/// Per-transaction metadata inside a `TransactionHistory`.
#[derive(Debug, Clone)]
pub struct TransactionInfo {
    pub txn_id: TransactionId,
    pub status: TransactionStatus,
    pub isolation_level: IsolationLevel,
    /// Indices into `TransactionHistory::events`.
    pub events: Vec<usize>,
    pub begin_timestamp: Option<u64>,
    pub end_timestamp: Option<u64>,
}

impl TransactionInfo {
    fn new(txn_id: TransactionId, isolation_level: IsolationLevel) -> Self {
        Self {
            txn_id,
            status: TransactionStatus::Active,
            isolation_level,
            events: Vec::new(),
            begin_timestamp: None,
            end_timestamp: None,
        }
    }
}

// ---------------------------------------------------------------------------
// TransactionHistory
// ---------------------------------------------------------------------------

/// A complete, ordered history of transaction events.
#[derive(Debug, Clone)]
pub struct TransactionHistory {
    pub(crate) events: Vec<HistoryEvent>,
    pub(crate) transactions: HashMap<TransactionId, TransactionInfo>,
}

impl TransactionHistory {
    /// Create an empty history.
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            transactions: HashMap::new(),
        }
    }

    /// Append an event, registering new transactions with `default_isolation`.
    pub fn add_event(&mut self, event: HistoryEvent, default_isolation: IsolationLevel) {
        let txn = event.txn_id();
        let idx = self.events.len();

        let info = self
            .transactions
            .entry(txn)
            .or_insert_with(|| TransactionInfo::new(txn, default_isolation));

        if info.begin_timestamp.is_none() {
            if let HistoryEvent::Commit { timestamp, .. } = &event {
                info.begin_timestamp = Some(*timestamp);
            }
        }

        match &event {
            HistoryEvent::Commit { timestamp, .. } => {
                info.status = TransactionStatus::Committed;
                info.end_timestamp = Some(*timestamp);
            }
            HistoryEvent::Abort { .. } => {
                info.status = TransactionStatus::Aborted;
                info.end_timestamp = Some(idx as u64);
            }
            _ => {}
        }

        info.events.push(idx);
        self.events.push(event);
    }

    /// Convenience: add event using `Serializable` as default isolation.
    pub fn add_event_default(&mut self, event: HistoryEvent) {
        self.add_event(event, IsolationLevel::Serializable);
    }

    /// Number of distinct transactions recorded.
    pub fn transaction_count(&self) -> usize {
        self.transactions.len()
    }

    /// Total number of events in the history.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Look up metadata for a transaction.
    pub fn get_transaction(&self, txn: TransactionId) -> Option<&TransactionInfo> {
        self.transactions.get(&txn)
    }

    /// Ids of all committed transactions (deterministic order).
    pub fn committed_transactions(&self) -> Vec<TransactionId> {
        self.txns_with_status(TransactionStatus::Committed)
    }

    /// Ids of all aborted transactions (deterministic order).
    pub fn aborted_transactions(&self) -> Vec<TransactionId> {
        self.txns_with_status(TransactionStatus::Aborted)
    }

    /// Ids of all still-active transactions (deterministic order).
    pub fn active_transactions(&self) -> Vec<TransactionId> {
        self.txns_with_status(TransactionStatus::Active)
    }

    fn txns_with_status(&self, status: TransactionStatus) -> Vec<TransactionId> {
        let mut ids: Vec<_> = self.transactions.values()
            .filter(|t| t.status == status).map(|t| t.txn_id).collect();
        ids.sort();
        ids
    }

    /// Return references to every event belonging to `txn`.
    pub fn events_for_txn(&self, txn: TransactionId) -> Vec<&HistoryEvent> {
        match self.transactions.get(&txn) {
            Some(info) => info.events.iter().filter_map(|&i| self.events.get(i)).collect(),
            None => Vec::new(),
        }
    }

    /// Return only the read events for `txn`.
    pub fn reads_for_txn(&self, txn: TransactionId) -> Vec<&HistoryEvent> {
        self.events_for_txn(txn)
            .into_iter()
            .filter(|e| e.is_read_event())
            .collect()
    }

    /// Return only the write events for `txn`.
    pub fn writes_for_txn(&self, txn: TransactionId) -> Vec<&HistoryEvent> {
        self.events_for_txn(txn)
            .into_iter()
            .filter(|e| e.is_write_event())
            .collect()
    }

    /// Collect the set of items read by `txn`.
    pub fn items_read_by(&self, txn: TransactionId) -> HashSet<ItemId> {
        let mut items = HashSet::new();
        for ev in self.events_for_txn(txn) {
            match ev {
                HistoryEvent::Read { item, .. } => {
                    items.insert(*item);
                }
                HistoryEvent::PredicateRead { matched_items, .. } => {
                    for id in matched_items {
                        items.insert(*id);
                    }
                }
                _ => {}
            }
        }
        items
    }

    /// Collect the set of items written (write/insert/delete) by `txn`.
    pub fn items_written_by(&self, txn: TransactionId) -> HashSet<ItemId> {
        let mut items = HashSet::new();
        for ev in self.events_for_txn(txn) {
            if let Some(id) = ev.item_id() {
                if ev.is_write_event() {
                    items.insert(id);
                }
            }
        }
        items
    }

    /// Build a new history containing only committed transaction events.
    pub fn committed_projection(&self) -> TransactionHistory {
        let committed: HashSet<TransactionId> =
            self.committed_transactions().into_iter().collect();

        let mut proj = TransactionHistory::new();
        for ev in &self.events {
            if committed.contains(&ev.txn_id()) {
                let txn = ev.txn_id();
                let iso = self
                    .transactions
                    .get(&txn)
                    .map(|t| t.isolation_level)
                    .unwrap_or(IsolationLevel::Serializable);
                proj.add_event(ev.clone(), iso);
            }
        }
        proj
    }

    /// Validate structural well-formedness. Returns `Ok(())` when valid, or
    /// `Err(diagnostics)` listing problems found.
    pub fn is_well_formed(&self) -> Result<(), Vec<String>> {
        let mut errors: Vec<String> = Vec::new();

        for (idx, ev) in self.events.iter().enumerate() {
            if !self.transactions.contains_key(&ev.txn_id()) {
                errors.push(format!("Event {} (txn {}) has no transaction entry", idx, ev.txn_id()));
            }
        }

        for (txn_id, info) in &self.transactions {
            for &idx in &info.events {
                if idx >= self.events.len() {
                    errors.push(format!("Transaction {} references out-of-range index {}", txn_id, idx));
                }
            }

            let mut terminal_count = 0usize;
            let mut seen_terminal = false;
            for &idx in &info.events {
                if idx >= self.events.len() { continue; }
                let ev = &self.events[idx];
                if ev.is_terminal() {
                    terminal_count += 1;
                    seen_terminal = true;
                } else if seen_terminal && ev.is_data_event() {
                    errors.push(format!(
                        "Transaction {} has data event at index {} after a terminal event", txn_id, idx
                    ));
                }
            }
            if terminal_count > 1 {
                errors.push(format!(
                    "Transaction {} has {} terminal events (expected at most 1)", txn_id, terminal_count
                ));
            }
            if info.status == TransactionStatus::Active && terminal_count > 0 {
                errors.push(format!("Transaction {} is Active but has a terminal event", txn_id));
            }
            if info.status == TransactionStatus::Committed && terminal_count == 0 {
                errors.push(format!("Transaction {} is Committed but has no Commit event", txn_id));
            }
            if info.status == TransactionStatus::Aborted && terminal_count == 0 {
                errors.push(format!("Transaction {} is Aborted but has no Abort event", txn_id));
            }
        }

        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }

    /// All events as a slice.
    pub fn events(&self) -> &[HistoryEvent] {
        &self.events
    }

    /// Create a `HistorySlice` covering events in `[start, end)`.
    pub fn slice(&self, start: usize, end: usize) -> HistorySlice<'_> {
        let clamped_end = end.min(self.events.len());
        let clamped_start = start.min(clamped_end);
        HistorySlice {
            events: &self.events[clamped_start..clamped_end],
            offset: clamped_start,
        }
    }

    /// Collect all distinct table ids referenced in the history.
    pub fn referenced_tables(&self) -> HashSet<TableId> {
        self.events.iter().filter_map(|e| e.table_id()).collect()
    }

    /// Collect all distinct item ids referenced in the history.
    pub fn referenced_items(&self) -> HashSet<ItemId> {
        let mut items = HashSet::new();
        for ev in &self.events {
            if let Some(id) = ev.item_id() {
                items.insert(id);
            }
            if let HistoryEvent::PredicateRead { matched_items, .. } = ev {
                for id in matched_items {
                    items.insert(*id);
                }
            }
        }
        items
    }
}

impl Default for TransactionHistory {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// HistorySlice
// ---------------------------------------------------------------------------

/// An immutable, borrowed view into a contiguous range of history events.
#[derive(Debug, Clone, Copy)]
pub struct HistorySlice<'a> {
    events: &'a [HistoryEvent],
    /// The index of the first event in the parent history.
    offset: usize,
}

impl<'a> HistorySlice<'a> {
    /// Number of events in this slice.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// True when the slice contains no events.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Offset of the first element relative to the owning history.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Iterate over the events.
    pub fn iter(&self) -> impl Iterator<Item = &'a HistoryEvent> {
        self.events.iter()
    }

    /// Get an event by its position within the slice.
    pub fn get(&self, index: usize) -> Option<&'a HistoryEvent> {
        self.events.get(index)
    }

    /// Return only the data events (non-terminal).
    pub fn data_events(&self) -> Vec<&'a HistoryEvent> {
        self.events.iter().filter(|e| e.is_data_event()).collect()
    }

    /// Return only the terminal events.
    pub fn terminal_events(&self) -> Vec<&'a HistoryEvent> {
        self.events.iter().filter(|e| e.is_terminal()).collect()
    }

    /// Collect distinct transaction ids that appear in this slice.
    pub fn transaction_ids(&self) -> Vec<TransactionId> {
        let mut seen = HashSet::new();
        let mut ids = Vec::new();
        for ev in self.events {
            let txn = ev.txn_id();
            if seen.insert(txn) {
                ids.push(txn);
            }
        }
        ids
    }

    /// Narrow the slice further by returning events for a single txn.
    pub fn filter_txn(&self, txn: TransactionId) -> Vec<&'a HistoryEvent> {
        self.events.iter().filter(|e| e.txn_id() == txn).collect()
    }

    /// Create a sub-slice (indices relative to *this* slice).
    pub fn sub_slice(&self, start: usize, end: usize) -> HistorySlice<'a> {
        let clamped_end = end.min(self.events.len());
        let clamped_start = start.min(clamped_end);
        HistorySlice {
            events: &self.events[clamped_start..clamped_end],
            offset: self.offset + clamped_start,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn t(n: u64) -> TransactionId { TransactionId::new(n) }
    fn i(n: u64) -> ItemId { ItemId::new(n) }
    fn tb(n: u64) -> TableId { TableId::new(n) }

    fn write_ev(txn: TransactionId, item: ItemId, table: TableId, val: i64) -> HistoryEvent {
        HistoryEvent::Write { txn, item, table, old_value: None, new_value: Value::Integer(val) }
    }

    #[test]
    fn empty_history() {
        let h = TransactionHistory::new();
        assert_eq!(h.transaction_count(), 0);
        assert_eq!(h.event_count(), 0);
        assert!(h.committed_transactions().is_empty());
        assert!(h.is_well_formed().is_ok());
    }

    #[test]
    fn add_events_and_commit() {
        let mut h = TransactionHistory::new();
        h.add_event_default(write_ev(t(1), i(100), tb(10), 42));
        h.add_event_default(HistoryEvent::Commit { txn: t(1), timestamp: 1000 });
        assert_eq!(h.transaction_count(), 1);
        assert_eq!(h.event_count(), 2);
        assert_eq!(h.committed_transactions(), vec![t(1)]);
        assert!(h.is_well_formed().is_ok());
    }

    #[test]
    fn items_read_and_written() {
        let mut h = TransactionHistory::new();
        h.add_event_default(HistoryEvent::Read { txn: t(1), item: i(1), table: tb(10), value: Some(Value::Integer(10)) });
        h.add_event_default(write_ev(t(1), i(2), tb(10), 20));
        h.add_event_default(HistoryEvent::Insert { txn: t(1), item: i(3), table: tb(10), values: vec![("c".into(), Value::Integer(30))] });
        h.add_event_default(HistoryEvent::Commit { txn: t(1), timestamp: 100 });

        assert!(h.items_read_by(t(1)).contains(&i(1)));
        assert!(!h.items_read_by(t(1)).contains(&i(2)));
        assert!(h.items_written_by(t(1)).contains(&i(2)));
        assert!(h.items_written_by(t(1)).contains(&i(3)));
    }

    #[test]
    fn committed_projection_excludes_aborted() {
        let mut h = TransactionHistory::new();
        h.add_event_default(write_ev(t(1), i(100), tb(10), 1));
        h.add_event_default(write_ev(t(2), i(100), tb(10), 2));
        h.add_event_default(HistoryEvent::Commit { txn: t(1), timestamp: 100 });
        h.add_event_default(HistoryEvent::Abort { txn: t(2), reason: Some("conflict".into()) });
        let proj = h.committed_projection();
        assert_eq!(proj.transaction_count(), 1);
        assert_eq!(proj.committed_transactions(), vec![t(1)]);
        assert_eq!(proj.event_count(), 2);
    }

    #[test]
    fn well_formedness_multiple_terminals() {
        let mut h = TransactionHistory::new();
        h.add_event_default(HistoryEvent::Commit { txn: t(1), timestamp: 1 });
        let idx = h.events.len();
        h.events.push(HistoryEvent::Abort { txn: t(1), reason: None });
        h.transactions.get_mut(&t(1)).unwrap().events.push(idx);
        let errs = h.is_well_formed().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("terminal events")));
    }

    #[test]
    fn well_formedness_data_after_terminal() {
        let mut h = TransactionHistory::new();
        h.add_event_default(HistoryEvent::Commit { txn: t(1), timestamp: 1 });
        let idx = h.events.len();
        h.events.push(HistoryEvent::Read { txn: t(1), item: i(1), table: tb(10), value: None });
        h.transactions.get_mut(&t(1)).unwrap().events.push(idx);
        let errs = h.is_well_formed().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("after a terminal")));
    }

    #[test]
    fn transaction_status_tracking() {
        let mut h = TransactionHistory::new();
        h.add_event_default(HistoryEvent::Read { txn: t(1), item: i(1), table: tb(10), value: None });
        h.add_event_default(HistoryEvent::Commit { txn: t(2), timestamp: 1 });
        h.add_event_default(HistoryEvent::Abort { txn: t(3), reason: None });
        assert_eq!(h.get_transaction(t(1)).unwrap().status, TransactionStatus::Active);
        assert_eq!(h.get_transaction(t(2)).unwrap().status, TransactionStatus::Committed);
        assert_eq!(h.get_transaction(t(3)).unwrap().status, TransactionStatus::Aborted);
    }

    #[test]
    fn history_slice_and_sub_slice() {
        let mut h = TransactionHistory::new();
        for x in 0..5 { h.add_event_default(write_ev(t(1), i(x), tb(10), x as i64)); }
        h.add_event_default(write_ev(t(2), i(99), tb(10), 99));
        let slice = h.slice(1, 4);
        assert_eq!(slice.len(), 3);
        assert_eq!(slice.offset(), 1);
        assert_eq!(slice.transaction_ids(), vec![t(1)]);
        assert_eq!(slice.sub_slice(0, 2).len(), 2);
    }

    #[test]
    fn predicate_read_items_collected() {
        let mut h = TransactionHistory::new();
        h.add_event_default(HistoryEvent::PredicateRead {
            txn: t(1), table: tb(10), predicate: Predicate::True, matched_items: vec![i(1), i(2)],
        });
        h.add_event_default(HistoryEvent::Commit { txn: t(1), timestamp: 1 });
        let read = h.items_read_by(t(1));
        assert!(read.contains(&i(1)) && read.contains(&i(2)));
    }

    #[test]
    fn event_labels_and_predicates() {
        let r = HistoryEvent::Read { txn: t(1), item: i(1), table: tb(1), value: None };
        assert_eq!(r.label(), "read");
        assert!(r.is_data_event() && r.is_read_event() && !r.is_write_event());
        let c = HistoryEvent::Commit { txn: t(1), timestamp: 0 };
        assert!(c.is_terminal() && !c.is_data_event());
        let d = HistoryEvent::Delete { txn: t(2), item: i(5), table: tb(3) };
        assert!(d.is_write_event() && d.is_data_event());
        assert_eq!(d.item_id(), Some(i(5)));
    }

    #[test]
    fn referenced_tables_and_items() {
        let mut h = TransactionHistory::new();
        h.add_event_default(HistoryEvent::Read { txn: t(1), item: i(1), table: tb(10), value: None });
        h.add_event_default(write_ev(t(1), i(2), tb(20), 0));
        h.add_event_default(HistoryEvent::Commit { txn: t(1), timestamp: 1 });
        assert_eq!(h.referenced_tables().len(), 2);
        assert_eq!(h.referenced_items().len(), 2);
    }
}
