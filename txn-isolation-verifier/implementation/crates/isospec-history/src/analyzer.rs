use std::collections::{HashMap, HashSet, VecDeque};
use isospec_types::identifier::*;
use isospec_types::isolation::*;
use isospec_types::dependency::*;
use isospec_types::value::*;
use isospec_types::predicate::Predicate;
use isospec_types::error::*;
use crate::history::*;

/// Maps (reader_txn, item) → writer_txn.#[derive(Debug, Clone)]
pub struct ReadFromRelation {
    relations: HashMap<(TransactionId, ItemId), TransactionId>,
}

impl ReadFromRelation {
    pub fn new() -> Self { Self { relations: HashMap::new() } }

    pub fn add(&mut self, reader: TransactionId, item: ItemId, writer: TransactionId) {
        self.relations.insert((reader, item), writer);
    }

    pub fn writer_for(&self, reader: TransactionId, item: ItemId) -> Option<TransactionId> {
        self.relations.get(&(reader, item)).copied()
    }

    pub fn all_pairs(&self) -> Vec<(TransactionId, ItemId, TransactionId)> {
        let mut v: Vec<_> = self.relations.iter()
            .map(|(&(r, i), &w)| (r, i, w)).collect();
        v.sort_by_key(|(r, i, w)| (r.as_u64(), i.as_u64(), w.as_u64()));
        v
    }

    pub fn len(&self) -> usize { self.relations.len() }
    pub fn is_empty(&self) -> bool { self.relations.is_empty() }
}

impl Default for ReadFromRelation { fn default() -> Self { Self::new() } }

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AntiDependency {
    pub reader: TransactionId,
    pub overwriter: TransactionId,
    pub item: ItemId,
}

#[derive(Debug, Clone)]pub struct AntiDependencySet { pub deps: Vec<AntiDependency> }

impl AntiDependencySet {
    pub fn new() -> Self { Self { deps: Vec::new() } }

    pub fn add(&mut self, reader: TransactionId, overwriter: TransactionId, item: ItemId) {
        self.deps.push(AntiDependency { reader, overwriter, item });
    }

    pub fn len(&self) -> usize { self.deps.len() }
    pub fn is_empty(&self) -> bool { self.deps.is_empty() }

    pub fn involves(&self, txn: TransactionId) -> Vec<&AntiDependency> {
        self.deps.iter().filter(|d| d.reader == txn || d.overwriter == txn).collect()
    }
}

impl Default for AntiDependencySet { fn default() -> Self { Self::new() } }

#[derive(Debug, Clone)]
pub struct DependencyCycle {
    pub nodes: Vec<TransactionId>,
    pub edges: Vec<Dependency>,
}

impl DependencyCycle {
    pub fn len(&self) -> usize { self.nodes.len() }
    pub fn is_empty(&self) -> bool { self.nodes.is_empty() }
    pub fn contains_txn(&self, txn: TransactionId) -> bool { self.nodes.contains(&txn) }

    pub fn edge_types(&self) -> Vec<DependencyType> {
        self.edges.iter().map(|e| e.dep_type).collect()
    }

    pub fn has_anti_dependency(&self) -> bool {
        self.edges.iter().any(|e| e.dep_type.is_anti_dependency())
    }
}

#[derive(Debug, Clone)]
pub struct DependencyGraph {
    nodes: HashSet<TransactionId>,
    edges: Vec<Dependency>,
    adjacency: HashMap<TransactionId, Vec<usize>>,
}

impl DependencyGraph {
    pub fn new() -> Self {
        Self { nodes: HashSet::new(), edges: Vec::new(), adjacency: HashMap::new() }
    }

    pub fn add_node(&mut self, txn: TransactionId) { self.nodes.insert(txn); }

    pub fn add_edge(&mut self, dep: Dependency) {
        self.nodes.insert(dep.from_txn);
        self.nodes.insert(dep.to_txn);
        let idx = self.edges.len();
        self.adjacency.entry(dep.from_txn).or_default().push(idx);
        self.edges.push(dep);
    }

    pub fn edges_from(&self, txn: TransactionId) -> Vec<&Dependency> {
        self.adjacency.get(&txn)
            .map(|ix| ix.iter().filter_map(|&i| self.edges.get(i)).collect())
            .unwrap_or_default()
    }

    pub fn edges_to(&self, txn: TransactionId) -> Vec<&Dependency> {
        self.edges.iter().filter(|e| e.to_txn == txn).collect()
    }

    pub fn edge_count(&self) -> usize { self.edges.len() }
    pub fn node_count(&self) -> usize { self.nodes.len() }
    pub fn all_edges(&self) -> &[Dependency] { &self.edges }
    pub fn nodes(&self) -> &HashSet<TransactionId> { &self.nodes }

    /// Cycle detection via DFS colouring (0=white, 1=grey, 2=black).
    pub fn has_cycle(&self) -> bool {
        let mut colour: HashMap<TransactionId, u8> =
            self.nodes.iter().map(|&n| (n, 0u8)).collect();
        self.nodes.iter().any(|&s| colour[&s] == 0 && self.dfs_has_cycle(s, &mut colour))
    }

    fn dfs_has_cycle(&self, node: TransactionId, colour: &mut HashMap<TransactionId, u8>) -> bool {
        colour.insert(node, 1);
        for dep in self.edges_from(node) {
            match colour.get(&dep.to_txn).copied().unwrap_or(0) {
                1 => return true,
                0 if self.dfs_has_cycle(dep.to_txn, colour) => return true,
                _ => {}
            }
        }
        colour.insert(node, 2);
        false
    }

    /// Enumerate elementary cycles (bounded depth DFS from each node).
    pub fn find_cycles(&self) -> Vec<DependencyCycle> {
        let mut cycles = Vec::new();
        let mut sorted: Vec<_> = self.nodes.iter().copied().collect();
        sorted.sort();

        for &start in &sorted {
            let mut stack: Vec<(TransactionId, Vec<TransactionId>, Vec<Dependency>)> =
                vec![(start, vec![start], Vec::new())];

            while let Some((cur, path, path_edges)) = stack.pop() {
                for dep in self.edges_from(cur) {
                    let next = dep.to_txn;
                    if next == start && path.len() >= 2 {
                        let mut ce = path_edges.clone();
                        ce.push(dep.clone());
                        cycles.push(DependencyCycle { nodes: path.clone(), edges: ce });
                    } else if next > start && !path.contains(&next) && path.len() < 8 {
                        let mut np = path.clone();
                        np.push(next);
                        let mut ne = path_edges.clone();
                        ne.push(dep.clone());
                        stack.push((next, np, ne));
                    }
                }
            }
        }
        cycles
    }
}

impl Default for DependencyGraph { fn default() -> Self { Self::new() } }

#[derive(Debug, Clone)]
pub struct ConflictPattern {
    pub pattern_name: String,
    pub involved_txns: Vec<TransactionId>,
    pub deps: Vec<Dependency>,
    pub description: String,
}

pub struct HistoryAnalyzer<'h> {
    history: &'h TransactionHistory,
}

impl<'h> HistoryAnalyzer<'h> {
    pub fn new(history: &'h TransactionHistory) -> Self { Self { history } }

    /// For every committed Read, find the latest committed write preceding it.
    pub fn compute_read_from(&self) -> ReadFromRelation {
        let committed: HashSet<TransactionId> =
            self.history.committed_transactions().into_iter().collect();
        let mut rf = ReadFromRelation::new();
        for ev in self.history.events() {
            if let HistoryEvent::Read { txn, item, .. } = ev {
                if committed.contains(txn) {
                    if let Some(w) = self.find_latest_writer(*item, *txn, &committed) {
                        rf.add(*txn, *item, w);
                    }
                }
            }
        }
        rf
    }

    fn find_latest_writer(&self, item: ItemId, reader: TransactionId,
                          committed: &HashSet<TransactionId>) -> Option<TransactionId> {
        let mut latest: Option<TransactionId> = None;
        for ev in self.history.events() {
            match ev {
                HistoryEvent::Read { txn, item: ri, .. } if *txn == reader && *ri == item => {
                    return latest;
                }
                HistoryEvent::Write { txn, item: wi, .. }
                | HistoryEvent::Insert { txn, item: wi, .. }
                | HistoryEvent::Delete { txn, item: wi, .. }
                    if *wi == item && committed.contains(txn) => { latest = Some(*txn); }
                _ => {}
            }
        }
        None
    }

    /// WW deps between committed txns writing the same item.
    pub fn compute_write_write_deps(&self) -> Vec<Dependency> {
        let committed: HashSet<TransactionId> =
            self.history.committed_transactions().into_iter().collect();
        let mut item_writers: HashMap<ItemId, Vec<TransactionId>> = HashMap::new();
        for ev in self.history.events() {
            if ev.is_write_event() {
                if let Some(item) = ev.item_id() {
                    let txn = ev.txn_id();
                    if committed.contains(&txn) {
                        item_writers.entry(item).or_default().push(txn);
                    }
                }
            }
        }
        let mut deps = Vec::new();
        for (item, writers) in &item_writers {
            for i in 0..writers.len() {
                for j in (i + 1)..writers.len() {
                    if writers[i] != writers[j] {
                        let tbl = self.table_for_item(*item).unwrap_or_else(TableId::zero);
                        deps.push(Dependency::new(writers[i], writers[j],
                            DependencyType::WriteWrite).with_item(*item, tbl));
                    }
                }
            }
        }
        dedup_deps(&mut deps);
        deps
    }

    /// WR deps derived from the read-from relation.
    pub fn compute_write_read_deps(&self) -> Vec<Dependency> {
        let rf = self.compute_read_from();
        let mut deps = Vec::new();
        for (reader, item, writer) in rf.all_pairs() {
            if writer != reader {
                let tbl = self.table_for_item(item).unwrap_or_else(TableId::zero);
                deps.push(Dependency::new(writer, reader, DependencyType::WriteRead)
                    .with_item(item, tbl));
            }
        }
        dedup_deps(&mut deps);
        deps
    }

    /// RW (anti) deps: reader → overwriter.
    pub fn compute_read_write_deps(&self) -> Vec<Dependency> {
        let committed: HashSet<TransactionId> =
            self.history.committed_transactions().into_iter().collect();
        let rf = self.compute_read_from();
        let mut deps = Vec::new();
        for (reader, item, writer) in rf.all_pairs() {
            for ow in self.find_overwriters(item, writer, &committed) {
                if ow != reader {
                    let tbl = self.table_for_item(item).unwrap_or_else(TableId::zero);
                    deps.push(Dependency::new(reader, ow, DependencyType::ReadWrite)
                        .with_item(item, tbl));
                }
            }
        }
        dedup_deps(&mut deps);
        deps
    }

    fn find_overwriters(&self, item: ItemId, writer: TransactionId,
                        committed: &HashSet<TransactionId>) -> Vec<TransactionId> {
        let mut past = false;
        let mut result = Vec::new();
        for ev in self.history.events() {
            match ev {
                HistoryEvent::Write { txn, item: wi, .. }
                | HistoryEvent::Insert { txn, item: wi, .. }
                | HistoryEvent::Delete { txn, item: wi, .. }
                    if *wi == item && committed.contains(txn) =>
                {
                    if past && *txn != writer && !result.contains(txn) { result.push(*txn); }
                    if *txn == writer { past = true; }
                }
                _ => {}
            }
        }
        result
    }

    /// Predicate-level deps from predicate reads vs insert/delete mutations.
    pub fn compute_predicate_deps(&self) -> Vec<Dependency> {
        let committed: HashSet<TransactionId> =
            self.history.committed_transactions().into_iter().collect();
        let mut deps = Vec::new();

        let pred_reads: Vec<(TransactionId, TableId)> = self.history.events().iter()
            .filter_map(|ev| match ev {
                HistoryEvent::PredicateRead { txn, table, .. } if committed.contains(txn) =>
                    Some((*txn, *table)),
                _ => None,
            }).collect();

        let mutations: Vec<(TransactionId, TableId, ItemId)> = self.history.events().iter()
            .filter_map(|ev| match ev {
                HistoryEvent::Insert { txn, table, item, .. }
                | HistoryEvent::Delete { txn, table, item }
                    if committed.contains(txn) => Some((*txn, *table, *item)),
                _ => None,
            }).collect();

        for &(pr_txn, pr_table) in &pred_reads {
            for &(mut_txn, mut_table, mut_item) in &mutations {
                if pr_txn != mut_txn && pr_table == mut_table {
                    deps.push(Dependency::new(pr_txn, mut_txn,
                        DependencyType::PredicateReadWrite).with_item(mut_item, mut_table));
                }
            }
        }
        dedup_deps(&mut deps);
        deps
    }

    /// Build the Direct Serialization Graph from all dependency types.
    pub fn build_dsg(&self) -> DependencyGraph {
        let mut g = DependencyGraph::new();
        for txn in self.history.committed_transactions() { g.add_node(txn); }
        for dep in self.compute_write_write_deps() { g.add_edge(dep); }
        for dep in self.compute_write_read_deps() { g.add_edge(dep); }
        for dep in self.compute_read_write_deps() { g.add_edge(dep); }
        for dep in self.compute_predicate_deps() { g.add_edge(dep); }
        g
    }

    /// Identify well-known conflict patterns from the DSG and history.
    pub fn identify_conflict_patterns(&self) -> Vec<ConflictPattern> {
        let dsg = self.build_dsg();
        let mut patterns = Vec::new();

        for cycle in dsg.find_cycles() {
            let types = cycle.edge_types();
            if types.iter().all(|t| *t == DependencyType::WriteWrite) {
                patterns.push(ConflictPattern {
                    pattern_name: "G0-DirtyWrite".into(),
                    involved_txns: cycle.nodes.clone(), deps: cycle.edges.clone(),
                    description: format!("Dirty-write cycle among {} txns", cycle.len()),
                });
            }
            if types.iter().all(|t| matches!(t, DependencyType::WriteWrite | DependencyType::WriteRead))
                && types.iter().any(|t| *t == DependencyType::WriteRead)
            {
                patterns.push(ConflictPattern {
                    pattern_name: "G1c-CircularInformationFlow".into(),
                    involved_txns: cycle.nodes.clone(), deps: cycle.edges.clone(),
                    description: format!("Circular info flow among {} txns", cycle.len()),
                });
            }
            if cycle.has_anti_dependency() && !types.iter().any(|t| t.is_predicate_level()) {
                patterns.push(ConflictPattern {
                    pattern_name: "G2-item".into(),
                    involved_txns: cycle.nodes.clone(), deps: cycle.edges.clone(),
                    description: format!("Item anti-dep cycle among {} txns", cycle.len()),
                });
            }
            if types.iter().any(|t| t.is_predicate_level()) {
                patterns.push(ConflictPattern {
                    pattern_name: "G2-Predicate".into(),
                    involved_txns: cycle.nodes.clone(), deps: cycle.edges.clone(),
                    description: format!("Predicate anti-dep cycle among {} txns", cycle.len()),
                });
            }
        }

        // Lost update: two txns each read-then-write the same item.
        let cl: Vec<TransactionId> = self.history.committed_transactions();
        for i in 0..cl.len() {
            for j in (i + 1)..cl.len() {
                let (ti, tj) = (cl[i], cl[j]);
                let ri = self.history.items_read_by(ti);
                let wi = self.history.items_written_by(ti);
                let rj = self.history.items_read_by(tj);
                let wj = self.history.items_written_by(tj);
                for &item in ri.intersection(&wi) {
                    if rj.contains(&item) && wj.contains(&item) {
                        let tbl = self.table_for_item(item).unwrap_or_else(TableId::zero);
                        patterns.push(ConflictPattern {
                            pattern_name: "LostUpdate".into(),
                            involved_txns: vec![ti, tj],
                            deps: vec![
                                Dependency::new(ti, tj, DependencyType::ReadWrite).with_item(item, tbl),
                                Dependency::new(tj, ti, DependencyType::ReadWrite).with_item(item, tbl),
                            ],
                            description: format!("Lost update on item {} between T{} and T{}",
                                item.as_u64(), ti.as_u64(), tj.as_u64()),
                        });
                    }
                }
            }
        }
        patterns
    }

    fn table_for_item(&self, item: ItemId) -> Option<TableId> {
        self.history.events().iter()
            .find(|ev| ev.item_id() == Some(item))
            .and_then(|ev| ev.table_id())
    }
}

fn dedup_deps(deps: &mut Vec<Dependency>) {
    let mut seen = HashSet::new();
    deps.retain(|d| {
        seen.insert((d.from_txn.as_u64(), d.to_txn.as_u64(), d.dep_type,
                      d.item_id.map(|i| i.as_u64())))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tid(n: u64) -> TransactionId { TransactionId::new(n) }
    fn iid(n: u64) -> ItemId { ItemId::new(n) }
    fn tbl(n: u64) -> TableId { TableId::new(n) }

    fn wr_history() -> TransactionHistory {
        let mut h = TransactionHistory::new();
        let (t, x) = (tbl(1), iid(10));
        h.add_event_default(HistoryEvent::Write { txn: tid(1), item: x, table: t,
            old_value: None, new_value: Value::Integer(1) });
        h.add_event_default(HistoryEvent::Read { txn: tid(2), item: x, table: t,
            value: Some(Value::Integer(1)) });
        h.add_event_default(HistoryEvent::Commit { txn: tid(1), timestamp: 10 });
        h.add_event_default(HistoryEvent::Commit { txn: tid(2), timestamp: 20 });
        h
    }

    fn ww_history() -> TransactionHistory {
        let mut h = TransactionHistory::new();
        let (t, x) = (tbl(1), iid(10));
        h.add_event_default(HistoryEvent::Write { txn: tid(1), item: x, table: t,
            old_value: None, new_value: Value::Integer(1) });
        h.add_event_default(HistoryEvent::Write { txn: tid(2), item: x, table: t,
            old_value: Some(Value::Integer(1)), new_value: Value::Integer(2) });
        h.add_event_default(HistoryEvent::Commit { txn: tid(1), timestamp: 10 });
        h.add_event_default(HistoryEvent::Commit { txn: tid(2), timestamp: 20 });
        h
    }

    #[test]
    fn test_read_from_simple() {
        let a = HistoryAnalyzer::new(&wr_history());
        let rf = a.compute_read_from();
        assert_eq!(rf.writer_for(tid(2), iid(10)), Some(tid(1)));
        assert_eq!(rf.len(), 1);
    }

    #[test]
    fn test_read_from_empty() {
        let h = TransactionHistory::new();
        assert!(HistoryAnalyzer::new(&h).compute_read_from().is_empty());
    }

    #[test]
    fn test_ww_deps() {
        let ww = HistoryAnalyzer::new(&ww_history()).compute_write_write_deps();
        assert_eq!(ww.len(), 1);
        assert_eq!(ww[0].from_txn, tid(1));
        assert_eq!(ww[0].to_txn, tid(2));
        assert_eq!(ww[0].dep_type, DependencyType::WriteWrite);
    }

    #[test]
    fn test_wr_deps() {
        let wr = HistoryAnalyzer::new(&wr_history()).compute_write_read_deps();
        assert_eq!(wr.len(), 1);
        assert_eq!(wr[0].dep_type, DependencyType::WriteRead);
    }

    #[test]
    fn test_rw_deps() {
        let mut h = TransactionHistory::new();
        let (t, x) = (tbl(1), iid(10));
        h.add_event_default(HistoryEvent::Write { txn: tid(1), item: x, table: t,
            old_value: None, new_value: Value::Integer(1) });
        h.add_event_default(HistoryEvent::Read { txn: tid(2), item: x, table: t,
            value: Some(Value::Integer(1)) });
        h.add_event_default(HistoryEvent::Write { txn: tid(3), item: x, table: t,
            old_value: Some(Value::Integer(1)), new_value: Value::Integer(2) });
        h.add_event_default(HistoryEvent::Commit { txn: tid(1), timestamp: 10 });
        h.add_event_default(HistoryEvent::Commit { txn: tid(2), timestamp: 20 });
        h.add_event_default(HistoryEvent::Commit { txn: tid(3), timestamp: 30 });

        let rw = HistoryAnalyzer::new(&h).compute_read_write_deps();
        assert!(rw.iter().any(|d| d.from_txn == tid(2) && d.to_txn == tid(3)
            && d.dep_type == DependencyType::ReadWrite));
    }

    #[test]
    fn test_dsg_no_cycle() {
        let dsg = HistoryAnalyzer::new(&wr_history()).build_dsg();
        assert_eq!(dsg.node_count(), 2);
        assert!(!dsg.has_cycle());
    }

    #[test]
    fn test_cycle_detection() {
        let mut g = DependencyGraph::new();
        g.add_edge(Dependency::new(tid(1), tid(2), DependencyType::WriteRead));
        g.add_edge(Dependency::new(tid(2), tid(1), DependencyType::ReadWrite));
        assert!(g.has_cycle());
        let cycles = g.find_cycles();
        assert!(!cycles.is_empty());
        assert!(cycles[0].contains_txn(tid(1)));
        assert!(cycles[0].has_anti_dependency());
    }

    #[test]
    fn test_no_cycle_linear() {
        let mut g = DependencyGraph::new();
        g.add_edge(Dependency::new(tid(1), tid(2), DependencyType::WriteWrite));
        g.add_edge(Dependency::new(tid(2), tid(3), DependencyType::WriteRead));
        assert!(!g.has_cycle());
        assert!(g.find_cycles().is_empty());
    }

    #[test]
    fn test_lost_update_pattern() {
        let mut h = TransactionHistory::new();
        let (t, x) = (tbl(1), iid(10));
        h.add_event_default(HistoryEvent::Write { txn: tid(0), item: x, table: t,
            old_value: None, new_value: Value::Integer(0) });
        h.add_event_default(HistoryEvent::Commit { txn: tid(0), timestamp: 1 });
        h.add_event_default(HistoryEvent::Read { txn: tid(1), item: x, table: t,
            value: Some(Value::Integer(0)) });
        h.add_event_default(HistoryEvent::Read { txn: tid(2), item: x, table: t,
            value: Some(Value::Integer(0)) });
        h.add_event_default(HistoryEvent::Write { txn: tid(1), item: x, table: t,
            old_value: Some(Value::Integer(0)), new_value: Value::Integer(1) });
        h.add_event_default(HistoryEvent::Write { txn: tid(2), item: x, table: t,
            old_value: Some(Value::Integer(0)), new_value: Value::Integer(2) });
        h.add_event_default(HistoryEvent::Commit { txn: tid(1), timestamp: 10 });
        h.add_event_default(HistoryEvent::Commit { txn: tid(2), timestamp: 20 });

        let pats = HistoryAnalyzer::new(&h).identify_conflict_patterns();
        assert!(pats.iter().any(|p| p.pattern_name == "LostUpdate"));
    }

    #[test]
    fn test_anti_dep_set() {
        let mut s = AntiDependencySet::new();
        s.add(tid(1), tid(2), iid(10));
        s.add(tid(3), tid(2), iid(20));
        assert_eq!(s.len(), 2);
        assert_eq!(s.involves(tid(2)).len(), 2);
        assert!(s.involves(tid(99)).is_empty());
    }

    #[test]
    fn test_cycle_edge_types() {
        let c = DependencyCycle {
            nodes: vec![tid(1), tid(2)],
            edges: vec![
                Dependency::new(tid(1), tid(2), DependencyType::WriteRead),
                Dependency::new(tid(2), tid(1), DependencyType::ReadWrite),
            ],
        };
        assert_eq!(c.len(), 2);
        assert!(c.has_anti_dependency());
        assert!(c.edge_types().contains(&DependencyType::WriteRead));
    }

    #[test]
    fn test_edges_to() {
        let mut g = DependencyGraph::new();
        g.add_edge(Dependency::new(tid(1), tid(3), DependencyType::WriteRead));
        g.add_edge(Dependency::new(tid(2), tid(3), DependencyType::WriteWrite));
        assert_eq!(g.edges_to(tid(3)).len(), 2);
        assert_eq!(g.edges_to(tid(1)).len(), 0);
    }
}
