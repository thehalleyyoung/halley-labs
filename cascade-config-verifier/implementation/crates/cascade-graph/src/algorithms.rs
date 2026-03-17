//! Graph algorithms: SCC, centrality, shortest paths, dominators,
//! articulation points, bridges, minimum cut, and more.

use crate::rtig::RtigGraph;
use cascade_types::policy::ResiliencePolicy;
use cascade_types::service::ServiceId;
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

// ── Structs ─────────────────────────────────────────────────────────

/// Transitive closure of the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitiveClosure {
    pub reachable: HashMap<ServiceId, HashSet<ServiceId>>,
}

/// Dominator tree rooted at a given entry node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DominatorTree {
    pub idom: HashMap<ServiceId, ServiceId>,
}

/// Minimum edge-cut between source and sink.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinCut {
    pub cut_edges: Vec<(ServiceId, ServiceId)>,
    pub cut_value: usize,
}

// ── Helpers ─────────────────────────────────────────────────────────

fn bfs_distances(graph: &RtigGraph, source: &ServiceId) -> HashMap<ServiceId, usize> {
    let mut dist = HashMap::new();
    let mut queue = VecDeque::new();
    dist.insert(source.clone(), 0);
    queue.push_back(source.clone());
    while let Some(cur) = queue.pop_front() {
        let d = dist[&cur];
        for next in graph.get_successors(&cur) {
            if !dist.contains_key(&next) {
                dist.insert(next.clone(), d + 1);
                queue.push_back(next);
            }
        }
    }
    dist
}

/// Treat the digraph as undirected for algorithms that need it.
fn undirected_neighbors(graph: &RtigGraph, id: &ServiceId) -> Vec<ServiceId> {
    let mut set: HashSet<ServiceId> = HashSet::new();
    for s in graph.get_successors(id) {
        set.insert(s);
    }
    for p in graph.get_predecessors(id) {
        set.insert(p);
    }
    set.into_iter().collect()
}

// ── Tarjan SCC ──────────────────────────────────────────────────────

struct TarjanState {
    index_counter: usize,
    stack: Vec<ServiceId>,
    on_stack: HashSet<ServiceId>,
    index: HashMap<ServiceId, usize>,
    lowlink: HashMap<ServiceId, usize>,
    result: Vec<Vec<ServiceId>>,
}

pub fn tarjan_scc(graph: &RtigGraph) -> Vec<Vec<ServiceId>> {
    let mut state = TarjanState {
        index_counter: 0,
        stack: Vec::new(),
        on_stack: HashSet::new(),
        index: HashMap::new(),
        lowlink: HashMap::new(),
        result: Vec::new(),
    };

    for id in graph.services() {
        if !state.index.contains_key(&id) {
            strongconnect(graph, &id, &mut state);
        }
    }
    state.result
}

fn strongconnect(graph: &RtigGraph, v: &ServiceId, state: &mut TarjanState) {
    let idx = state.index_counter;
    state.index.insert(v.clone(), idx);
    state.lowlink.insert(v.clone(), idx);
    state.index_counter += 1;
    state.stack.push(v.clone());
    state.on_stack.insert(v.clone());

    for w in graph.get_successors(v) {
        if !state.index.contains_key(&w) {
            strongconnect(graph, &w, state);
            let wl = state.lowlink[&w];
            let vl = state.lowlink[v];
            state.lowlink.insert(v.clone(), vl.min(wl));
        } else if state.on_stack.contains(&w) {
            let wi = state.index[&w];
            let vl = state.lowlink[v];
            state.lowlink.insert(v.clone(), vl.min(wi));
        }
    }

    if state.lowlink[v] == state.index[v] {
        let mut component = Vec::new();
        loop {
            let w = state.stack.pop().unwrap();
            state.on_stack.remove(&w);
            component.push(w.clone());
            if w == *v {
                break;
            }
        }
        state.result.push(component);
    }
}

pub fn strongly_connected_components(graph: &RtigGraph) -> Vec<Vec<ServiceId>> {
    tarjan_scc(graph)
}

// ── Condensation graph ─────────────────────────────────────────────

pub fn condensation_graph(graph: &RtigGraph) -> RtigGraph {
    let sccs = tarjan_scc(graph);
    let mut g = RtigGraph::new();
    let mut svc_to_scc: HashMap<ServiceId, ServiceId> = HashMap::new();

    for (i, group) in sccs.iter().enumerate() {
        let rep = ServiceId::new(format!("scc-{}", i));
        g.add_service(rep.clone());
        for s in group {
            svc_to_scc.insert(s.clone(), rep.clone());
        }
    }

    let mut seen: HashSet<(ServiceId, ServiceId)> = HashSet::new();
    for e in graph.inner.edge_references() {
        let src = graph.inner.node_weight(e.source()).unwrap();
        let tgt = graph.inner.node_weight(e.target()).unwrap();
        let s = svc_to_scc.get(src).cloned().unwrap_or_else(|| src.clone());
        let t = svc_to_scc.get(tgt).cloned().unwrap_or_else(|| tgt.clone());
        if s != t && seen.insert((s.clone(), t.clone())) {
            g.add_dependency(&s, &t, ResiliencePolicy::empty());
        }
    }
    g
}

// ── Diameter / Radius ───────────────────────────────────────────────

pub fn graph_diameter(graph: &RtigGraph) -> usize {
    graph.compute_diameter()
}

pub fn graph_radius(graph: &RtigGraph) -> usize {
    let ids = graph.services();
    if ids.is_empty() {
        return 0;
    }
    let mut min_eccentricity = usize::MAX;
    for id in &ids {
        let dist = bfs_distances(graph, id);
        let ecc = dist.values().copied().max().unwrap_or(0);
        if ecc > 0 {
            min_eccentricity = min_eccentricity.min(ecc);
        }
    }
    if min_eccentricity == usize::MAX {
        0
    } else {
        min_eccentricity
    }
}

// ── Shortest path DAG ──────────────────────────────────────────────

pub fn shortest_path_dag(graph: &RtigGraph, source: &ServiceId) -> HashMap<ServiceId, Vec<ServiceId>> {
    let dist = bfs_distances(graph, source);
    let mut parent: HashMap<ServiceId, Vec<ServiceId>> = HashMap::new();

    for (node, &d) in &dist {
        if d == 0 {
            continue;
        }
        for pred in graph.get_predecessors(node) {
            if dist.get(&pred).copied() == Some(d - 1) {
                parent.entry(node.clone()).or_default().push(pred);
            }
        }
    }
    parent
}

// ── All-pairs shortest paths (Floyd-Warshall) ──────────────────────

pub fn all_pairs_shortest_paths(graph: &RtigGraph) -> HashMap<(ServiceId, ServiceId), usize> {
    let ids = graph.services();
    let n = ids.len();
    let idx_of: HashMap<&ServiceId, usize> = ids.iter().enumerate().map(|(i, s)| (s, i)).collect();
    let inf = usize::MAX / 2;
    let mut dist = vec![vec![inf; n]; n];

    for i in 0..n {
        dist[i][i] = 0;
    }
    for e in graph.inner.edge_references() {
        let s = graph.inner.node_weight(e.source()).unwrap();
        let t = graph.inner.node_weight(e.target()).unwrap();
        if let (Some(&u), Some(&v)) = (idx_of.get(s), idx_of.get(t)) {
            dist[u][v] = 1;
        }
    }
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                let through_k = dist[i][k].saturating_add(dist[k][j]);
                if through_k < dist[i][j] {
                    dist[i][j] = through_k;
                }
            }
        }
    }
    let mut result = HashMap::new();
    for i in 0..n {
        for j in 0..n {
            if dist[i][j] < inf {
                result.insert((ids[i].clone(), ids[j].clone()), dist[i][j]);
            }
        }
    }
    result
}

// ── Transitive closure ─────────────────────────────────────────────

pub fn transitive_closure(graph: &RtigGraph) -> TransitiveClosure {
    let ids = graph.services();
    let mut reachable: HashMap<ServiceId, HashSet<ServiceId>> = HashMap::new();
    for id in &ids {
        let dist = bfs_distances(graph, id);
        let set: HashSet<ServiceId> = dist.keys().cloned().collect();
        reachable.insert(id.clone(), set);
    }
    TransitiveClosure { reachable }
}

// ── Dominator tree ─────────────────────────────────────────────────

pub fn dominator_tree(graph: &RtigGraph, root: &ServiceId) -> DominatorTree {
    let sorted = graph.topological_sort().unwrap_or_else(|| graph.services());

    let mut idom: HashMap<ServiceId, ServiceId> = HashMap::new();
    idom.insert(root.clone(), root.clone());

    let pos: HashMap<ServiceId, usize> = sorted.iter().enumerate().map(|(i, s)| (s.clone(), i)).collect();

    let mut changed = true;
    while changed {
        changed = false;
        for node in &sorted {
            if node == root {
                continue;
            }
            let preds = graph.get_predecessors(node);
            let processed: Vec<&ServiceId> = preds.iter().filter(|p| idom.contains_key(*p)).collect();
            if processed.is_empty() {
                continue;
            }

            let mut new_idom = processed[0].clone();
            for p in &processed[1..] {
                new_idom = intersect_doms(&idom, &pos, &new_idom, p);
            }
            if idom.get(node) != Some(&new_idom) {
                idom.insert(node.clone(), new_idom);
                changed = true;
            }
        }
    }
    idom.remove(root);
    DominatorTree { idom }
}

fn intersect_doms(
    idom: &HashMap<ServiceId, ServiceId>,
    pos: &HashMap<ServiceId, usize>,
    a: &ServiceId,
    b: &ServiceId,
) -> ServiceId {
    let mut a = a.clone();
    let mut b = b.clone();
    let mut pa = pos.get(&a).copied().unwrap_or(0);
    let mut pb = pos.get(&b).copied().unwrap_or(0);

    while pa != pb {
        while pa > pb {
            a = idom.get(&a).cloned().unwrap_or_else(|| a.clone());
            pa = pos.get(&a).copied().unwrap_or(0);
        }
        while pb > pa {
            b = idom.get(&b).cloned().unwrap_or_else(|| b.clone());
            pb = pos.get(&b).copied().unwrap_or(0);
        }
    }
    a
}

// ── Articulation points ────────────────────────────────────────────

pub fn articulation_points(graph: &RtigGraph) -> Vec<ServiceId> {
    let ids = graph.services();
    if ids.len() <= 2 {
        return vec![];
    }

    let mut disc: HashMap<ServiceId, usize> = HashMap::new();
    let mut low: HashMap<ServiceId, usize> = HashMap::new();
    let mut parent: HashMap<ServiceId, Option<ServiceId>> = HashMap::new();
    let mut ap: HashSet<ServiceId> = HashSet::new();
    let mut timer = 0usize;

    for id in &ids {
        if !disc.contains_key(id) {
            parent.insert(id.clone(), None);
            dfs_ap(graph, id, &mut disc, &mut low, &mut parent, &mut ap, &mut timer);
        }
    }
    ap.into_iter().collect()
}

fn dfs_ap(
    graph: &RtigGraph,
    u: &ServiceId,
    disc: &mut HashMap<ServiceId, usize>,
    low: &mut HashMap<ServiceId, usize>,
    parent: &mut HashMap<ServiceId, Option<ServiceId>>,
    ap: &mut HashSet<ServiceId>,
    timer: &mut usize,
) {
    disc.insert(u.clone(), *timer);
    low.insert(u.clone(), *timer);
    *timer += 1;
    let mut child_count = 0usize;

    let neighbors = undirected_neighbors(graph, u);

    for v in &neighbors {
        if !disc.contains_key(v) {
            child_count += 1;
            parent.insert(v.clone(), Some(u.clone()));
            dfs_ap(graph, v, disc, low, parent, ap, timer);

            let low_u = low[u];
            let low_v = low[v];
            low.insert(u.clone(), low_u.min(low_v));

            let is_root = parent.get(u).map(|p| p.is_none()).unwrap_or(true);
            if is_root && child_count > 1 {
                ap.insert(u.clone());
            }
            if !is_root && low_v >= disc[u] {
                ap.insert(u.clone());
            }
        } else if parent.get(u).and_then(|p| p.as_ref()) != Some(v) {
            let low_u = low[u];
            let disc_v = disc[v];
            low.insert(u.clone(), low_u.min(disc_v));
        }
    }
}

// ── Bridges ────────────────────────────────────────────────────────

pub fn bridges(graph: &RtigGraph) -> Vec<(ServiceId, ServiceId)> {
    let ids = graph.services();
    let mut disc: HashMap<ServiceId, usize> = HashMap::new();
    let mut low: HashMap<ServiceId, usize> = HashMap::new();
    let mut parent: HashMap<ServiceId, Option<ServiceId>> = HashMap::new();
    let mut result: Vec<(ServiceId, ServiceId)> = Vec::new();
    let mut timer = 0usize;

    for id in &ids {
        if !disc.contains_key(id) {
            parent.insert(id.clone(), None);
            dfs_bridge(graph, id, &mut disc, &mut low, &mut parent, &mut result, &mut timer);
        }
    }
    result
}

fn dfs_bridge(
    graph: &RtigGraph,
    u: &ServiceId,
    disc: &mut HashMap<ServiceId, usize>,
    low: &mut HashMap<ServiceId, usize>,
    parent: &mut HashMap<ServiceId, Option<ServiceId>>,
    bridges: &mut Vec<(ServiceId, ServiceId)>,
    timer: &mut usize,
) {
    disc.insert(u.clone(), *timer);
    low.insert(u.clone(), *timer);
    *timer += 1;

    let neighbors = undirected_neighbors(graph, u);

    for v in &neighbors {
        if !disc.contains_key(v) {
            parent.insert(v.clone(), Some(u.clone()));
            dfs_bridge(graph, v, disc, low, parent, bridges, timer);

            let low_u = low[u];
            let low_v = low[v];
            low.insert(u.clone(), low_u.min(low_v));

            if low_v > disc[u] {
                bridges.push((u.clone(), v.clone()));
            }
        } else if parent.get(u).and_then(|p| p.as_ref()) != Some(v) {
            let low_u = low[u];
            let disc_v = disc[v];
            low.insert(u.clone(), low_u.min(disc_v));
        }
    }
}

// ── Betweenness centrality (Brandes) ───────────────────────────────

pub fn betweenness_centrality(graph: &RtigGraph) -> HashMap<ServiceId, f64> {
    let ids = graph.services();
    let mut centrality: HashMap<ServiceId, f64> = ids.iter().map(|id| (id.clone(), 0.0)).collect();

    for s in &ids {
        let (sigma, pred, dist) = bfs_brandes(graph, s);
        let mut delta: HashMap<ServiceId, f64> = ids.iter().map(|id| (id.clone(), 0.0)).collect();

        let mut order: Vec<ServiceId> = dist.keys().cloned().collect();
        order.sort_by_key(|n| std::cmp::Reverse(dist.get(n).copied().unwrap_or(0)));

        for w in &order {
            if let Some(preds) = pred.get(w) {
                for v in preds {
                    let sv = sigma.get(v).copied().unwrap_or(1.0);
                    let sw = sigma.get(w).copied().unwrap_or(1.0);
                    if sw > 0.0 {
                        let d = delta.get(w).copied().unwrap_or(0.0);
                        *delta.entry(v.clone()).or_default() += (sv / sw) * (1.0 + d);
                    }
                }
            }
            if w != s {
                *centrality.entry(w.clone()).or_default() += delta.get(w).copied().unwrap_or(0.0);
            }
        }
    }

    let n = ids.len() as f64;
    if n > 2.0 {
        let norm = (n - 1.0) * (n - 2.0);
        for val in centrality.values_mut() {
            *val /= norm;
        }
    }
    centrality
}

fn bfs_brandes(
    graph: &RtigGraph,
    source: &ServiceId,
) -> (
    HashMap<ServiceId, f64>,
    HashMap<ServiceId, Vec<ServiceId>>,
    HashMap<ServiceId, usize>,
) {
    let mut sigma: HashMap<ServiceId, f64> = HashMap::new();
    let mut dist: HashMap<ServiceId, usize> = HashMap::new();
    let mut pred: HashMap<ServiceId, Vec<ServiceId>> = HashMap::new();
    let mut queue = VecDeque::new();

    sigma.insert(source.clone(), 1.0);
    dist.insert(source.clone(), 0);
    queue.push_back(source.clone());

    while let Some(v) = queue.pop_front() {
        let dv = dist[&v];
        for w in graph.get_successors(&v) {
            if !dist.contains_key(&w) {
                dist.insert(w.clone(), dv + 1);
                queue.push_back(w.clone());
            }
            if dist[&w] == dv + 1 {
                let sv = sigma.get(&v).copied().unwrap_or(0.0);
                *sigma.entry(w.clone()).or_default() += sv;
                pred.entry(w).or_default().push(v.clone());
            }
        }
    }
    (sigma, pred, dist)
}

// ── Minimum cut (Edmonds-Karp) ─────────────────────────────────────

pub fn minimum_cut(graph: &RtigGraph, source: &ServiceId, sink: &ServiceId) -> MinCut {
    let ids = graph.services();
    let mut residual: HashMap<(ServiceId, ServiceId), i32> = HashMap::new();

    for e in graph.inner.edge_references() {
        let s = graph.inner.node_weight(e.source()).unwrap().clone();
        let t = graph.inner.node_weight(e.target()).unwrap().clone();
        *residual.entry((s, t)).or_default() += 1;
    }

    loop {
        let mut parent: HashMap<ServiceId, ServiceId> = HashMap::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(source.clone());
        queue.push_back(source.clone());

        while let Some(u) = queue.pop_front() {
            if u == *sink {
                break;
            }
            for v in &ids {
                let cap = residual.get(&(u.clone(), v.clone())).copied().unwrap_or(0);
                if cap > 0 && !visited.contains(v) {
                    visited.insert(v.clone());
                    parent.insert(v.clone(), u.clone());
                    queue.push_back(v.clone());
                }
            }
        }

        if !parent.contains_key(sink) {
            break;
        }

        let mut path_flow = i32::MAX;
        let mut v = sink.clone();
        while let Some(u) = parent.get(&v) {
            let cap = residual.get(&(u.clone(), v.clone())).copied().unwrap_or(0);
            path_flow = path_flow.min(cap);
            v = u.clone();
        }

        v = sink.clone();
        while let Some(u) = parent.get(&v) {
            *residual.entry((u.clone(), v.clone())).or_default() -= path_flow;
            *residual.entry((v.clone(), u.clone())).or_default() += path_flow;
            v = u.clone();
        }
    }

    let mut reachable = HashSet::new();
    let mut queue = VecDeque::new();
    reachable.insert(source.clone());
    queue.push_back(source.clone());
    while let Some(u) = queue.pop_front() {
        for v in &ids {
            let cap = residual.get(&(u.clone(), v.clone())).copied().unwrap_or(0);
            if cap > 0 && !reachable.contains(v) {
                reachable.insert(v.clone());
                queue.push_back(v.clone());
            }
        }
    }

    let mut cut_edges = Vec::new();
    for e in graph.inner.edge_references() {
        let s = graph.inner.node_weight(e.source()).unwrap();
        let t = graph.inner.node_weight(e.target()).unwrap();
        if reachable.contains(s) && !reachable.contains(t) {
            cut_edges.push((s.clone(), t.clone()));
        }
    }
    let cut_value = cut_edges.len();
    MinCut { cut_edges, cut_value }
}

// ── K-core decomposition ───────────────────────────────────────────

pub fn k_core_decomposition(graph: &RtigGraph) -> HashMap<ServiceId, usize> {
    let ids = graph.services();
    let mut degree: HashMap<ServiceId, usize> = ids
        .iter()
        .map(|id| {
            let d = graph.get_predecessors(id).len() + graph.get_successors(id).len();
            (id.clone(), d)
        })
        .collect();

    let mut core: HashMap<ServiceId, usize> = HashMap::new();
    let mut remaining: HashSet<ServiceId> = ids.into_iter().collect();

    let mut k = 0usize;
    while !remaining.is_empty() {
        let min_deg = remaining.iter().map(|v| degree[v]).min().unwrap_or(0);
        k = k.max(min_deg);

        let to_remove: Vec<ServiceId> = remaining
            .iter()
            .filter(|v| degree[v] <= k)
            .cloned()
            .collect();

        for v in &to_remove {
            core.insert(v.clone(), k);
            remaining.remove(v);
            for n in undirected_neighbors(graph, v) {
                if let Some(d) = degree.get_mut(&n) {
                    *d = d.saturating_sub(1);
                }
            }
        }
    }
    core
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rtig::{build_chain, build_diamond, RtigGraph};
    use cascade_types::policy::ResiliencePolicy;
    use cascade_types::service::ServiceId;

    fn sid(s: &str) -> ServiceId {
        ServiceId::new(s)
    }

    fn chain() -> RtigGraph {
        build_chain(&["a", "b", "c"], 1)
    }

    #[test]
    fn scc_dag() {
        let sccs = tarjan_scc(&chain());
        assert_eq!(sccs.len(), 3);
    }

    #[test]
    fn scc_with_cycle() {
        let mut g = RtigGraph::new();
        for n in &["a", "b", "c"] {
            g.add_service(sid(n));
        }
        let p = ResiliencePolicy::empty();
        g.add_dependency(&sid("a"), &sid("b"), p.clone());
        g.add_dependency(&sid("b"), &sid("c"), p.clone());
        g.add_dependency(&sid("c"), &sid("a"), p);
        let sccs = tarjan_scc(&g);
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0].len(), 3);
    }

    #[test]
    fn test_condensation() {
        let mut g = RtigGraph::new();
        for n in &["a", "b", "c", "d"] {
            g.add_service(sid(n));
        }
        let p = ResiliencePolicy::empty();
        g.add_dependency(&sid("a"), &sid("b"), p.clone());
        g.add_dependency(&sid("b"), &sid("a"), p.clone());
        g.add_dependency(&sid("b"), &sid("c"), p.clone());
        g.add_dependency(&sid("c"), &sid("d"), p);
        let cond = condensation_graph(&g);
        assert!(cond.is_dag());
    }

    #[test]
    fn transitive_closure_chain() {
        let tc = transitive_closure(&chain());
        let reachable_a = &tc.reachable[&sid("a")];
        assert!(reachable_a.contains(&sid("c")));
    }

    #[test]
    fn dominator_tree_chain() {
        let dt = dominator_tree(&chain(), &sid("a"));
        assert_eq!(dt.idom.get(&sid("b")), Some(&sid("a")));
    }

    #[test]
    fn test_diameter() {
        let g = build_chain(&["a", "b", "c", "d"], 1);
        assert_eq!(graph_diameter(&g), 3);
    }

    #[test]
    fn test_radius() {
        let g = build_chain(&["a", "b", "c"], 1);
        let r = graph_radius(&g);
        assert!(r >= 1);
    }

    #[test]
    fn test_shortest_path_dag() {
        let g = build_chain(&["a", "b", "c"], 1);
        let spd = shortest_path_dag(&g, &sid("a"));
        assert!(spd.contains_key(&sid("b")));
    }

    #[test]
    fn test_all_pairs_shortest() {
        let g = build_chain(&["a", "b", "c"], 1);
        let apsp = all_pairs_shortest_paths(&g);
        assert_eq!(apsp[&(sid("a"), sid("c"))], 2);
    }

    #[test]
    fn betweenness_chain() {
        let bc = betweenness_centrality(&chain());
        assert!(bc[&sid("b")] >= bc[&sid("a")]);
    }

    #[test]
    fn test_bridges_chain() {
        let g = build_chain(&["a", "b", "c"], 1);
        let br = bridges(&g);
        assert!(!br.is_empty());
    }

    #[test]
    fn test_minimum_cut() {
        let g = build_chain(&["a", "b", "c"], 1);
        let mc = minimum_cut(&g, &sid("a"), &sid("c"));
        assert!(mc.cut_value >= 1);
    }

    #[test]
    fn test_k_core() {
        let g = build_chain(&["a", "b", "c", "d"], 1);
        let kc = k_core_decomposition(&g);
        assert_eq!(kc.len(), 4);
    }

    #[test]
    fn test_diamond_scc() {
        let g = build_diamond(1);
        let sccs = tarjan_scc(&g);
        assert_eq!(sccs.len(), 4); // DAG => each node its own SCC
    }

    #[test]
    fn test_empty_graph() {
        let g = RtigGraph::new();
        assert_eq!(tarjan_scc(&g).len(), 0);
        assert_eq!(graph_diameter(&g), 0);
    }
}
