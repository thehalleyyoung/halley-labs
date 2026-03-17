//! Execution trace recording, serialization, diffing, and statistics.
use std::collections::HashMap;
use isospec_types::identifier::{TransactionId, ItemId, TableId};
use isospec_types::isolation::IsolationLevel;
use isospec_types::value::Value;
use isospec_types::error::{IsoSpecError, IsoSpecResult};

#[derive(Debug, Clone, PartialEq)]
pub enum TraceEventKind {
    BeginTransaction { txn: TransactionId, isolation: IsolationLevel },
    CommitTransaction { txn: TransactionId },
    AbortTransaction { txn: TransactionId, reason: Option<String> },
    ReadItem { txn: TransactionId, table: TableId, item: ItemId, value: Option<Value> },
    WriteItem { txn: TransactionId, table: TableId, item: ItemId, old_value: Option<Value>, new_value: Value },
    InsertItem { txn: TransactionId, table: TableId, item: ItemId },
    DeleteItem { txn: TransactionId, table: TableId, item: ItemId },
    LockAcquired { txn: TransactionId, table: TableId, item: Option<ItemId>, mode: String },
    LockReleased { txn: TransactionId, table: TableId, item: Option<ItemId> },
    Checkpoint { label: String },
}

impl TraceEventKind {
    pub fn txn_id(&self) -> Option<TransactionId> {
        match self {
            Self::BeginTransaction { txn, .. } | Self::CommitTransaction { txn }
            | Self::AbortTransaction { txn, .. } | Self::ReadItem { txn, .. }
            | Self::WriteItem { txn, .. } | Self::InsertItem { txn, .. }
            | Self::DeleteItem { txn, .. } | Self::LockAcquired { txn, .. }
            | Self::LockReleased { txn, .. } => Some(*txn),
            Self::Checkpoint { .. } => None,
        }
    }

    pub fn is_data_event(&self) -> bool {
        matches!(self, Self::ReadItem { .. } | Self::WriteItem { .. }
            | Self::InsertItem { .. } | Self::DeleteItem { .. })
    }

    pub fn description(&self) -> String {
        match self {
            Self::BeginTransaction { txn, isolation } => format!("BEGIN {} at {:?}", txn, isolation),
            Self::CommitTransaction { txn } => format!("COMMIT {}", txn),
            Self::AbortTransaction { txn, reason } => match reason {
                Some(r) => format!("ABORT {} ({})", txn, r),
                None => format!("ABORT {}", txn),
            },
            Self::ReadItem { txn, table, item, value } => {
                let v = value.as_ref().map(|v| format!("{}", v)).unwrap_or_else(|| "NULL".into());
                format!("READ {}: {}.{} = {}", txn, table, item, v)
            }
            Self::WriteItem { txn, table, item, new_value, .. } => format!("WRITE {}: {}.{} = {}", txn, table, item, new_value),
            Self::InsertItem { txn, table, item } => format!("INSERT {}: {}.{}", txn, table, item),
            Self::DeleteItem { txn, table, item } => format!("DELETE {}: {}.{}", txn, table, item),
            Self::LockAcquired { txn, table, item, mode } => {
                let tgt = item.map(|i| format!("{}.{}", table, i)).unwrap_or_else(|| format!("{}", table));
                format!("LOCK {} on {} ({})", txn, tgt, mode)
            }
            Self::LockReleased { txn, table, item } => {
                let tgt = item.map(|i| format!("{}.{}", table, i)).unwrap_or_else(|| format!("{}", table));
                format!("UNLOCK {} on {}", txn, tgt)
            }
            Self::Checkpoint { label } => format!("CHECKPOINT: {}", label),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TraceEvent {
    pub kind: TraceEventKind,
    pub timestamp_ns: u64,
    pub sequence: u64,
    pub metadata: HashMap<String, String>,
}

impl TraceEvent {
    pub fn new(kind: TraceEventKind, timestamp_ns: u64, sequence: u64) -> Self {
        Self { kind, timestamp_ns, sequence, metadata: HashMap::new() }
    }
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TraceFilter {
    AllEvents, TransactionOnly(TransactionId), DataEventsOnly, ExcludeLocks, Custom(String),
}

impl TraceFilter {
    fn matches(&self, kind: &TraceEventKind) -> bool {
        match self {
            Self::AllEvents => true,
            Self::TransactionOnly(txn) => kind.txn_id().map_or(false, |t| t == *txn),
            Self::DataEventsOnly => kind.is_data_event(),
            Self::ExcludeLocks => !matches!(kind, TraceEventKind::LockAcquired { .. } | TraceEventKind::LockReleased { .. }),
            Self::Custom(_) => true,
        }
    }
}

pub struct TraceRecorder {
    events: Vec<TraceEvent>,
    start_time_ns: u64,
    next_sequence: u64,
    is_recording: bool,
    filters: Vec<TraceFilter>,
}

impl TraceRecorder {
    pub fn new() -> Self {
        Self { events: Vec::new(), start_time_ns: 0, next_sequence: 0, is_recording: false, filters: Vec::new() }
    }
    pub fn start(&mut self) { self.start_time_ns = Self::now_ns(); self.is_recording = true; }
    pub fn stop(&mut self) { self.is_recording = false; }
    pub fn is_recording(&self) -> bool { self.is_recording }
    pub fn add_filter(&mut self, filter: TraceFilter) { self.filters.push(filter); }

    pub fn record(&mut self, kind: TraceEventKind) {
        if !self.is_recording || !self.passes_filters(&kind) { return; }
        let ts = Self::now_ns().saturating_sub(self.start_time_ns);
        let seq = self.next_sequence;
        self.next_sequence += 1;
        self.events.push(TraceEvent::new(kind, ts, seq));
    }

    pub fn record_with_metadata(&mut self, kind: TraceEventKind, metadata: HashMap<String, String>) {
        if !self.is_recording || !self.passes_filters(&kind) { return; }
        let ts = Self::now_ns().saturating_sub(self.start_time_ns);
        let seq = self.next_sequence;
        self.next_sequence += 1;
        let mut event = TraceEvent::new(kind, ts, seq);
        event.metadata = metadata;
        self.events.push(event);
    }

    pub fn events(&self) -> &[TraceEvent] { &self.events }
    pub fn event_count(&self) -> usize { self.events.len() }
    pub fn events_for_txn(&self, txn: TransactionId) -> Vec<&TraceEvent> {
        self.events.iter().filter(|e| e.kind.txn_id() == Some(txn)).collect()
    }
    pub fn clear(&mut self) { self.events.clear(); self.next_sequence = 0; }
    pub fn elapsed_ns(&self) -> u64 {
        if self.start_time_ns == 0 { 0 } else { Self::now_ns().saturating_sub(self.start_time_ns) }
    }
    fn passes_filters(&self, kind: &TraceEventKind) -> bool {
        self.filters.is_empty() || self.filters.iter().all(|f| f.matches(kind))
    }
    fn now_ns() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos() as u64
    }
}

impl Default for TraceRecorder { fn default() -> Self { Self::new() } }

// --- Serialization helpers (manual JSON, no serde derive on our types) ---
fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for ch in s.chars() {
        match ch {
            '"' => o.push_str("\\\""), '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"), '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn iso_str(l: IsolationLevel) -> &'static str {
    match l {
        IsolationLevel::ReadUncommitted => "RU", IsolationLevel::ReadCommitted => "RC",
        IsolationLevel::RepeatableRead => "RR", IsolationLevel::Serializable => "SR",
        IsolationLevel::Snapshot => "SI", IsolationLevel::PgReadCommitted => "PgRC",
        IsolationLevel::MySqlRepeatableRead => "MyRR", IsolationLevel::SqlServerRCSI => "RCSI",
    }
}

fn parse_iso(s: &str) -> Option<IsolationLevel> {
    match s {
        "RU" => Some(IsolationLevel::ReadUncommitted), "RC" => Some(IsolationLevel::ReadCommitted),
        "RR" => Some(IsolationLevel::RepeatableRead), "SR" => Some(IsolationLevel::Serializable),
        "SI" => Some(IsolationLevel::Snapshot), "PgRC" => Some(IsolationLevel::PgReadCommitted),
        "MyRR" => Some(IsolationLevel::MySqlRepeatableRead), "RCSI" => Some(IsolationLevel::SqlServerRCSI),
        _ => None,
    }
}

fn val_json(v: &Value) -> String {
    match v {
        Value::Null => "null".into(),
        Value::Boolean(b) => format!("{}", b),
        Value::Integer(i) => format!("{{\"t\":\"I\",\"v\":{}}}", i),
        Value::Float(f) => format!("{{\"t\":\"F\",\"v\":{}}}", f),
        Value::Text(s) => format!("{{\"t\":\"T\",\"v\":{}}}", esc(s)),
        Value::Bytes(b) => {
            let hex: String = b.iter().map(|x| format!("{:02x}", x)).collect();
            format!("{{\"t\":\"B\",\"v\":{}}}", esc(&hex))
        }
        Value::Timestamp(ts) => format!("{{\"t\":\"Ts\",\"v\":{}}}", ts),
        Value::Array(a) => {
            let items: Vec<String> = a.iter().map(val_json).collect();
            format!("{{\"t\":\"A\",\"v\":[{}]}}", items.join(","))
        }
    }
}

fn parse_val(v: &serde_json::Value) -> Option<Value> {
    if v.is_null() { return Some(Value::Null); }
    let o = v.as_object()?;
    match o.get("t")?.as_str()? {
        "I" => Some(Value::Integer(o.get("v")?.as_i64()?)),
        "F" => Some(Value::Float(o.get("v")?.as_f64()?)),
        "T" => Some(Value::Text(o.get("v")?.as_str()?.into())),
        "B" => {
            let h = o.get("v")?.as_str()?;
            Some(Value::Bytes((0..h.len()).step_by(2).filter_map(|i| u8::from_str_radix(&h[i..i+2], 16).ok()).collect()))
        }
        "Ts" => Some(Value::Timestamp(o.get("v")?.as_i64()?)),
        "A" => { let a: Option<Vec<_>> = o.get("v")?.as_array()?.iter().map(parse_val).collect(); Some(Value::Array(a?)) }
        _ => None,
    }
}

fn opt_vj(v: &Option<Value>) -> String { v.as_ref().map(val_json).unwrap_or_else(|| "null".into()) }
fn opt_item(i: &Option<ItemId>) -> String { i.map(|x| x.as_u64().to_string()).unwrap_or_else(|| "null".into()) }

fn kind_json(k: &TraceEventKind) -> String {
    match k {
        TraceEventKind::BeginTransaction { txn, isolation } =>
            format!("{{\"k\":\"B\",\"t\":{},\"i\":{}}}", txn.as_u64(), esc(iso_str(*isolation))),
        TraceEventKind::CommitTransaction { txn } => format!("{{\"k\":\"C\",\"t\":{}}}", txn.as_u64()),
        TraceEventKind::AbortTransaction { txn, reason } => {
            let r = reason.as_ref().map(|s| esc(s)).unwrap_or_else(|| "null".into());
            format!("{{\"k\":\"A\",\"t\":{},\"r\":{}}}", txn.as_u64(), r)
        }
        TraceEventKind::ReadItem { txn, table, item, value } =>
            format!("{{\"k\":\"R\",\"t\":{},\"tb\":{},\"i\":{},\"v\":{}}}", txn.as_u64(), table.as_u64(), item.as_u64(), opt_vj(value)),
        TraceEventKind::WriteItem { txn, table, item, old_value, new_value } =>
            format!("{{\"k\":\"W\",\"t\":{},\"tb\":{},\"i\":{},\"o\":{},\"n\":{}}}", txn.as_u64(), table.as_u64(), item.as_u64(), opt_vj(old_value), val_json(new_value)),
        TraceEventKind::InsertItem { txn, table, item } =>
            format!("{{\"k\":\"Ins\",\"t\":{},\"tb\":{},\"i\":{}}}", txn.as_u64(), table.as_u64(), item.as_u64()),
        TraceEventKind::DeleteItem { txn, table, item } =>
            format!("{{\"k\":\"Del\",\"t\":{},\"tb\":{},\"i\":{}}}", txn.as_u64(), table.as_u64(), item.as_u64()),
        TraceEventKind::LockAcquired { txn, table, item, mode } =>
            format!("{{\"k\":\"LA\",\"t\":{},\"tb\":{},\"i\":{},\"m\":{}}}", txn.as_u64(), table.as_u64(), opt_item(item), esc(mode)),
        TraceEventKind::LockReleased { txn, table, item } =>
            format!("{{\"k\":\"LR\",\"t\":{},\"tb\":{},\"i\":{}}}", txn.as_u64(), table.as_u64(), opt_item(item)),
        TraceEventKind::Checkpoint { label } => format!("{{\"k\":\"Ck\",\"l\":{}}}", esc(label)),
    }
}

fn meta_json(m: &HashMap<String, String>) -> String {
    if m.is_empty() { return "{}".into(); }
    let e: Vec<String> = m.iter().map(|(k, v)| format!("{}:{}", esc(k), esc(v))).collect();
    format!("{{{}}}", e.join(","))
}

pub fn to_json(events: &[TraceEvent]) -> IsoSpecResult<String> {
    let parts: Vec<String> = events.iter().map(|ev| {
        format!("{{\"kind\":{},\"ts\":{},\"seq\":{},\"meta\":{}}}", kind_json(&ev.kind), ev.timestamp_ns, ev.sequence, meta_json(&ev.metadata))
    }).collect();
    Ok(format!("[{}]", parts.join(",")))
}

fn pk(o: &serde_json::Value) -> Option<TraceEventKind> {
    let k = o.get("k")?.as_str()?;
    match k {
        "B" => { let txn = TransactionId::new(o.get("t")?.as_u64()?); Some(TraceEventKind::BeginTransaction { txn, isolation: parse_iso(o.get("i")?.as_str()?)? }) }
        "C" => Some(TraceEventKind::CommitTransaction { txn: TransactionId::new(o.get("t")?.as_u64()?) }),
        "A" => { let txn = TransactionId::new(o.get("t")?.as_u64()?); let r = o.get("r").and_then(|v| v.as_str().map(String::from)); Some(TraceEventKind::AbortTransaction { txn, reason: r }) }
        "R" => { Some(TraceEventKind::ReadItem { txn: TransactionId::new(o.get("t")?.as_u64()?), table: TableId::new(o.get("tb")?.as_u64()?), item: ItemId::new(o.get("i")?.as_u64()?), value: o.get("v").and_then(parse_val) }) }
        "W" => { Some(TraceEventKind::WriteItem { txn: TransactionId::new(o.get("t")?.as_u64()?), table: TableId::new(o.get("tb")?.as_u64()?), item: ItemId::new(o.get("i")?.as_u64()?), old_value: o.get("o").and_then(parse_val), new_value: parse_val(o.get("n")?)? }) }
        "Ins" => Some(TraceEventKind::InsertItem { txn: TransactionId::new(o.get("t")?.as_u64()?), table: TableId::new(o.get("tb")?.as_u64()?), item: ItemId::new(o.get("i")?.as_u64()?) }),
        "Del" => Some(TraceEventKind::DeleteItem { txn: TransactionId::new(o.get("t")?.as_u64()?), table: TableId::new(o.get("tb")?.as_u64()?), item: ItemId::new(o.get("i")?.as_u64()?) }),
        "LA" => { Some(TraceEventKind::LockAcquired { txn: TransactionId::new(o.get("t")?.as_u64()?), table: TableId::new(o.get("tb")?.as_u64()?), item: o.get("i").and_then(|v| v.as_u64()).map(ItemId::new), mode: o.get("m")?.as_str()?.into() }) }
        "LR" => { Some(TraceEventKind::LockReleased { txn: TransactionId::new(o.get("t")?.as_u64()?), table: TableId::new(o.get("tb")?.as_u64()?), item: o.get("i").and_then(|v| v.as_u64()).map(ItemId::new) }) }
        "Ck" => Some(TraceEventKind::Checkpoint { label: o.get("l")?.as_str()?.into() }),
        _ => None,
    }
}

pub fn from_json(json: &str) -> IsoSpecResult<Vec<TraceEvent>> {
    let parsed: serde_json::Value = serde_json::from_str(json).map_err(IsoSpecError::Json)?;
    let arr = parsed.as_array().ok_or_else(|| IsoSpecError::Internal("trace JSON must be an array".into()))?;
    let mut events = Vec::with_capacity(arr.len());
    for entry in arr {
        let kind = pk(entry.get("kind").ok_or_else(|| IsoSpecError::Internal("missing kind".into()))?)
            .ok_or_else(|| IsoSpecError::Internal(format!("bad trace kind: {}", entry)))?;
        let ts = entry.get("ts").and_then(|v| v.as_u64()).unwrap_or(0);
        let seq = entry.get("seq").and_then(|v| v.as_u64()).unwrap_or(0);
        let mut meta = HashMap::new();
        if let Some(obj) = entry.get("meta").and_then(|v| v.as_object()) {
            for (k, v) in obj { if let Some(s) = v.as_str() { meta.insert(k.clone(), s.into()); } }
        }
        let mut ev = TraceEvent::new(kind, ts, seq);
        ev.metadata = meta;
        events.push(ev);
    }
    Ok(events)
}

pub fn to_csv(events: &[TraceEvent]) -> String {
    let mut buf = String::from("sequence,timestamp_ns,kind,txn,description\n");
    for ev in events {
        let txn_str = ev.kind.txn_id().map(|t| t.as_u64().to_string()).unwrap_or_default();
        let tag = match &ev.kind {
            TraceEventKind::BeginTransaction { .. } => "Begin", TraceEventKind::CommitTransaction { .. } => "Commit",
            TraceEventKind::AbortTransaction { .. } => "Abort", TraceEventKind::ReadItem { .. } => "Read",
            TraceEventKind::WriteItem { .. } => "Write", TraceEventKind::InsertItem { .. } => "Insert",
            TraceEventKind::DeleteItem { .. } => "Delete", TraceEventKind::LockAcquired { .. } => "LockAcquired",
            TraceEventKind::LockReleased { .. } => "LockReleased", TraceEventKind::Checkpoint { .. } => "Checkpoint",
        };
        buf.push_str(&format!("{},{},{},{},{}\n", ev.sequence, ev.timestamp_ns, tag, txn_str, ev.kind.description().replace(',', ";")));
    }
    buf
}

// --- TraceDiff ---
#[derive(Debug, Clone)]
pub struct TraceDiff {
    pub only_in_first: Vec<usize>,
    pub only_in_second: Vec<usize>,
    pub value_mismatches: Vec<(usize, usize)>,
    pub ordering_differences: Vec<(usize, usize)>,
}

impl TraceDiff {
    pub fn new() -> Self {
        Self { only_in_first: Vec::new(), only_in_second: Vec::new(), value_mismatches: Vec::new(), ordering_differences: Vec::new() }
    }

    pub fn compute(first: &[TraceEvent], second: &[TraceEvent]) -> Self {
        let mut diff = Self::new();
        let mut matched: Vec<bool> = vec![false; second.len()];
        let mut last_j: Option<usize> = None;
        for (i, e1) in first.iter().enumerate() {
            let d1 = e1.kind.description();
            let mut found = false;
            for (j, e2) in second.iter().enumerate() {
                if matched[j] { continue; }
                if d1 == e2.kind.description() {
                    matched[j] = true;
                    if e1.kind != e2.kind { diff.value_mismatches.push((i, j)); }
                    if let Some(pj) = last_j { if j < pj { diff.ordering_differences.push((i, j)); } }
                    last_j = Some(j);
                    found = true;
                    break;
                }
            }
            if !found { diff.only_in_first.push(i); }
        }
        for (j, m) in matched.iter().enumerate() { if !m { diff.only_in_second.push(j); } }
        diff
    }

    pub fn is_equivalent(&self) -> bool {
        self.only_in_first.is_empty() && self.only_in_second.is_empty() && self.value_mismatches.is_empty()
    }

    pub fn summary(&self) -> String {
        if self.is_equivalent() && self.ordering_differences.is_empty() { return "Traces are identical".into(); }
        let mut p = Vec::new();
        if !self.only_in_first.is_empty() { p.push(format!("{} events only in first", self.only_in_first.len())); }
        if !self.only_in_second.is_empty() { p.push(format!("{} events only in second", self.only_in_second.len())); }
        if !self.value_mismatches.is_empty() { p.push(format!("{} value mismatches", self.value_mismatches.len())); }
        if !self.ordering_differences.is_empty() { p.push(format!("{} ordering differences", self.ordering_differences.len())); }
        p.join(", ")
    }
}

impl Default for TraceDiff { fn default() -> Self { Self::new() } }

// --- TraceStatistics ---
#[derive(Debug, Clone)]
pub struct TraceStatistics {
    pub total_events: usize,
    pub txn_count: usize,
    pub read_count: usize,
    pub write_count: usize,
    pub commit_count: usize,
    pub abort_count: usize,
    pub avg_txn_length: f64,
    pub max_txn_length: usize,
}

impl TraceStatistics {
    pub fn compute(events: &[TraceEvent]) -> Self {
        let (mut rc, mut wc, mut cc, mut ac) = (0usize, 0usize, 0usize, 0usize);
        let mut txn_ev: HashMap<u64, usize> = HashMap::new();
        for ev in events {
            if let Some(t) = ev.kind.txn_id() { *txn_ev.entry(t.as_u64()).or_insert(0) += 1; }
            match &ev.kind {
                TraceEventKind::ReadItem { .. } => rc += 1, TraceEventKind::WriteItem { .. } => wc += 1,
                TraceEventKind::CommitTransaction { .. } => cc += 1, TraceEventKind::AbortTransaction { .. } => ac += 1,
                _ => {}
            }
        }
        let tc = txn_ev.len();
        let mx = txn_ev.values().copied().max().unwrap_or(0);
        let avg = if tc > 0 { txn_ev.values().sum::<usize>() as f64 / tc as f64 } else { 0.0 };
        Self { total_events: events.len(), txn_count: tc, read_count: rc, write_count: wc, commit_count: cc, abort_count: ac, avg_txn_length: avg, max_txn_length: mx }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn txn(id: u64) -> TransactionId { TransactionId::new(id) }
    fn tbl(id: u64) -> TableId { TableId::new(id) }
    fn itm(id: u64) -> ItemId { ItemId::new(id) }

    fn sample() -> Vec<TraceEvent> {
        vec![
            TraceEvent::new(TraceEventKind::BeginTransaction { txn: txn(1), isolation: IsolationLevel::ReadCommitted }, 100, 0),
            TraceEvent::new(TraceEventKind::ReadItem { txn: txn(1), table: tbl(1), item: itm(10), value: Some(Value::Integer(42)) }, 200, 1),
            TraceEvent::new(TraceEventKind::WriteItem { txn: txn(1), table: tbl(1), item: itm(10), old_value: Some(Value::Integer(42)), new_value: Value::Integer(99) }, 300, 2),
            TraceEvent::new(TraceEventKind::CommitTransaction { txn: txn(1) }, 400, 3),
        ]
    }

    #[test]
    fn test_event_kind_methods() {
        assert_eq!(TraceEventKind::BeginTransaction { txn: txn(5), isolation: IsolationLevel::Serializable }.txn_id(), Some(txn(5)));
        assert_eq!(TraceEventKind::Checkpoint { label: "x".into() }.txn_id(), None);
        assert!(TraceEventKind::ReadItem { txn: txn(1), table: tbl(1), item: itm(1), value: None }.is_data_event());
        assert!(TraceEventKind::WriteItem { txn: txn(1), table: tbl(1), item: itm(1), old_value: None, new_value: Value::Null }.is_data_event());
        assert!(!TraceEventKind::CommitTransaction { txn: txn(1) }.is_data_event());
        assert!(!TraceEventKind::LockAcquired { txn: txn(1), table: tbl(1), item: None, mode: "S".into() }.is_data_event());
        let d = TraceEventKind::AbortTransaction { txn: txn(3), reason: Some("conflict".into()) }.description();
        assert!(d.contains("ABORT") && d.contains("conflict"));
    }

    #[test]
    fn test_recorder_basic() {
        let mut rec = TraceRecorder::new();
        rec.record(TraceEventKind::Checkpoint { label: "ignored".into() });
        assert_eq!(rec.event_count(), 0);
        rec.start();
        assert!(rec.is_recording());
        rec.record(TraceEventKind::BeginTransaction { txn: txn(1), isolation: IsolationLevel::ReadCommitted });
        rec.record(TraceEventKind::CommitTransaction { txn: txn(1) });
        assert_eq!(rec.event_count(), 2);
        assert_eq!(rec.events()[0].sequence, 0);
        assert_eq!(rec.events()[1].sequence, 1);
    }

    #[test]
    fn test_stop_start() {
        let mut rec = TraceRecorder::new();
        rec.start();
        rec.record(TraceEventKind::Checkpoint { label: "a".into() });
        rec.stop();
        rec.record(TraceEventKind::Checkpoint { label: "b".into() });
        assert_eq!(rec.event_count(), 1);
        rec.start();
        rec.record(TraceEventKind::Checkpoint { label: "c".into() });
        assert_eq!(rec.event_count(), 2);
    }

    #[test]
    fn test_filter_data_and_locks() {
        let mut rec = TraceRecorder::new();
        rec.add_filter(TraceFilter::DataEventsOnly);
        rec.start();
        rec.record(TraceEventKind::BeginTransaction { txn: txn(1), isolation: IsolationLevel::Snapshot });
        rec.record(TraceEventKind::ReadItem { txn: txn(1), table: tbl(1), item: itm(1), value: None });
        rec.record(TraceEventKind::CommitTransaction { txn: txn(1) });
        assert_eq!(rec.event_count(), 1);

        let mut rec2 = TraceRecorder::new();
        rec2.add_filter(TraceFilter::ExcludeLocks);
        rec2.start();
        rec2.record(TraceEventKind::LockAcquired { txn: txn(1), table: tbl(1), item: None, mode: "X".into() });
        rec2.record(TraceEventKind::ReadItem { txn: txn(1), table: tbl(1), item: itm(1), value: None });
        rec2.record(TraceEventKind::LockReleased { txn: txn(1), table: tbl(1), item: None });
        assert_eq!(rec2.event_count(), 1);
    }

    #[test]
    fn test_filter_txn_only() {
        let mut rec = TraceRecorder::new();
        rec.add_filter(TraceFilter::TransactionOnly(txn(2)));
        rec.start();
        rec.record(TraceEventKind::ReadItem { txn: txn(1), table: tbl(1), item: itm(1), value: None });
        rec.record(TraceEventKind::ReadItem { txn: txn(2), table: tbl(1), item: itm(2), value: None });
        rec.record(TraceEventKind::CommitTransaction { txn: txn(2) });
        assert_eq!(rec.event_count(), 2);
        assert!(rec.events().iter().all(|e| e.kind.txn_id() == Some(txn(2))));
    }

    #[test]
    fn test_events_for_txn() {
        let mut rec = TraceRecorder::new();
        rec.start();
        rec.record(TraceEventKind::BeginTransaction { txn: txn(1), isolation: IsolationLevel::ReadCommitted });
        rec.record(TraceEventKind::BeginTransaction { txn: txn(2), isolation: IsolationLevel::ReadCommitted });
        rec.record(TraceEventKind::ReadItem { txn: txn(1), table: tbl(1), item: itm(1), value: None });
        rec.record(TraceEventKind::CommitTransaction { txn: txn(2) });
        assert_eq!(rec.events_for_txn(txn(1)).len(), 2);
        assert_eq!(rec.events_for_txn(txn(2)).len(), 2);
        assert_eq!(rec.events_for_txn(txn(99)).len(), 0);
    }

    #[test]
    fn test_clear_and_metadata() {
        let mut rec = TraceRecorder::new();
        rec.start();
        rec.record(TraceEventKind::Checkpoint { label: "a".into() });
        assert_eq!(rec.event_count(), 1);
        rec.clear();
        assert_eq!(rec.event_count(), 0);

        let ev = TraceEvent::new(TraceEventKind::Checkpoint { label: "x".into() }, 0, 0).with_metadata("key", "val");
        assert_eq!(ev.metadata.get("key").unwrap(), "val");
    }

    #[test]
    fn test_json_roundtrip() {
        let events = sample();
        let json = to_json(&events).unwrap();
        let parsed = from_json(&json).unwrap();
        assert_eq!(parsed.len(), events.len());
        for (a, b) in events.iter().zip(parsed.iter()) {
            assert_eq!(a.kind, b.kind);
            assert_eq!(a.sequence, b.sequence);
            assert_eq!(a.timestamp_ns, b.timestamp_ns);
        }
    }

    #[test]
    fn test_json_roundtrip_all_kinds() {
        let events = vec![
            TraceEvent::new(TraceEventKind::BeginTransaction { txn: txn(1), isolation: IsolationLevel::Snapshot }, 0, 0),
            TraceEvent::new(TraceEventKind::AbortTransaction { txn: txn(1), reason: Some("timeout".into()) }, 0, 1),
            TraceEvent::new(TraceEventKind::InsertItem { txn: txn(2), table: tbl(3), item: itm(4) }, 0, 2),
            TraceEvent::new(TraceEventKind::DeleteItem { txn: txn(2), table: tbl(3), item: itm(4) }, 0, 3),
            TraceEvent::new(TraceEventKind::LockAcquired { txn: txn(2), table: tbl(3), item: Some(itm(4)), mode: "IX".into() }, 0, 4),
            TraceEvent::new(TraceEventKind::LockReleased { txn: txn(2), table: tbl(3), item: None }, 0, 5),
            TraceEvent::new(TraceEventKind::Checkpoint { label: "mid".into() }, 0, 6),
        ];
        let json = to_json(&events).unwrap();
        let parsed = from_json(&json).unwrap();
        assert_eq!(parsed.len(), events.len());
        for (a, b) in events.iter().zip(parsed.iter()) { assert_eq!(a.kind, b.kind); }
        // CSV smoke test
        let csv = to_csv(&events);
        assert!(csv.starts_with("sequence,"));
        assert_eq!(csv.lines().count(), events.len() + 1);
    }

    #[test]
    fn test_diff_identical_and_different() {
        let events = sample();
        let diff = TraceDiff::compute(&events, &events);
        assert!(diff.is_equivalent());
        assert_eq!(diff.summary(), "Traces are identical");
        let second = vec![
            TraceEvent::new(TraceEventKind::BeginTransaction { txn: txn(1), isolation: IsolationLevel::ReadCommitted }, 100, 0),
            TraceEvent::new(TraceEventKind::CommitTransaction { txn: txn(1) }, 400, 1),
        ];
        let diff2 = TraceDiff::compute(&events, &second);
        assert!(!diff2.is_equivalent());
        assert_eq!(diff2.only_in_first.len(), 2);
        assert!(diff2.summary().contains("only in first"));
    }

    #[test]
    fn test_statistics() {
        let stats = TraceStatistics::compute(&sample());
        assert_eq!(stats.total_events, 4);
        assert_eq!(stats.txn_count, 1);
        assert_eq!(stats.read_count, 1);
        assert_eq!(stats.write_count, 1);
        assert_eq!(stats.commit_count, 1);
        assert_eq!(stats.abort_count, 0);
        assert!((stats.avg_txn_length - 4.0).abs() < f64::EPSILON);
        assert_eq!(stats.max_txn_length, 4);
    }

    #[test]
    fn test_statistics_multi_txn() {
        let events = vec![
            TraceEvent::new(TraceEventKind::BeginTransaction { txn: txn(1), isolation: IsolationLevel::ReadCommitted }, 0, 0),
            TraceEvent::new(TraceEventKind::ReadItem { txn: txn(1), table: tbl(1), item: itm(1), value: None }, 0, 1),
            TraceEvent::new(TraceEventKind::CommitTransaction { txn: txn(1) }, 0, 2),
            TraceEvent::new(TraceEventKind::BeginTransaction { txn: txn(2), isolation: IsolationLevel::Serializable }, 0, 3),
            TraceEvent::new(TraceEventKind::AbortTransaction { txn: txn(2), reason: None }, 0, 4),
        ];
        let stats = TraceStatistics::compute(&events);
        assert_eq!(stats.txn_count, 2);
        assert_eq!(stats.commit_count, 1);
        assert_eq!(stats.abort_count, 1);
        assert_eq!(stats.max_txn_length, 3);
    }
}
