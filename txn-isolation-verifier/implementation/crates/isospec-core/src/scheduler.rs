// Schedule generation and enumeration.
// ScheduleEnumerator, serial schedule generation, random schedule sampling.

use std::collections::VecDeque;

use isospec_types::identifier::{
    OperationId, ScheduleStepId, TransactionId,
};
use isospec_types::isolation::IsolationLevel;
use isospec_types::operation::{OpKind, Operation};
use isospec_types::schedule::{Schedule, ScheduleMetadata, ScheduleStep};
use isospec_types::config::EngineKind;
use isospec_types::value::Value;

// ---------------------------------------------------------------------------
// Transaction operation list helper
// ---------------------------------------------------------------------------

/// Lightweight representation of one transaction's operations for schedule
/// enumeration.
#[derive(Debug, Clone)]
pub struct TransactionOps {
    pub txn_id: TransactionId,
    pub ops: Vec<Operation>,
}

impl TransactionOps {
    pub fn new(txn_id: TransactionId, ops: Vec<Operation>) -> Self {
        Self { txn_id, ops }
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

// ---------------------------------------------------------------------------
// ScheduleEnumerator
// ---------------------------------------------------------------------------

/// Enumerates all possible interleavings of operations from multiple
/// transactions, optionally bounded by `k` (max number of context switches).
#[derive(Debug)]
pub struct ScheduleEnumerator {
    transactions: Vec<TransactionOps>,
    bound_k: Option<usize>,
    generated: u64,
}

impl ScheduleEnumerator {
    /// Create a new enumerator for the given transactions.
    pub fn new(transactions: Vec<TransactionOps>) -> Self {
        Self {
            transactions,
            bound_k: None,
            generated: 0,
        }
    }

    /// Create a bounded enumerator. `bound_k` limits the maximum number
    /// of context switches (i.e., changes of executing transaction) in
    /// generated schedules.
    pub fn bounded(transactions: Vec<TransactionOps>, bound_k: usize) -> Self {
        Self {
            transactions,
            bound_k: Some(bound_k),
            generated: 0,
        }
    }

    /// Total number of operations across all transactions.
    pub fn total_operations(&self) -> usize {
        self.transactions.iter().map(|t| t.len()).sum()
    }

    /// Number of transactions.
    pub fn transaction_count(&self) -> usize {
        self.transactions.len()
    }

    /// Number of schedules generated so far.
    pub fn generated_count(&self) -> u64 {
        self.generated
    }

    /// Generate all serial schedules (one per permutation of transactions).
    pub fn serial_schedules(&mut self) -> Vec<Schedule> {
        let n = self.transactions.len();
        let perms = permutations(n);
        let mut result = Vec::with_capacity(perms.len());

        for perm in &perms {
            let schedule = self.build_serial_schedule(perm);
            result.push(schedule);
        }
        self.generated += result.len() as u64;
        result
    }

    /// Generate a single serial schedule in the given transaction order.
    pub fn serial_schedule_in_order(&mut self, order: &[usize]) -> Schedule {
        self.generated += 1;
        self.build_serial_schedule(order)
    }

    fn build_serial_schedule(&self, order: &[usize]) -> Schedule {
        let mut schedule = Schedule {
            steps: Vec::new(),
            transaction_order: order
                .iter()
                .map(|&i| self.transactions[i].txn_id)
                .collect(),
            metadata: ScheduleMetadata::default(),
        };
        for &idx in order {
            let txn = &self.transactions[idx];
            for op in &txn.ops {
                schedule.add_step(txn.txn_id, op.clone());
            }
        }
        schedule
    }

    /// Enumerate all interleavings up to the configured bound using
    /// iterative depth-first exploration of the schedule search space.
    /// WARNING: this can be exponential in size; use with small inputs.
    pub fn enumerate_all(&mut self) -> Vec<Schedule> {
        let n = self.transactions.len();
        if n == 0 {
            return vec![];
        }
        let mut results = Vec::new();
        // cursors[i] = index of next op to schedule from transactions[i]
        let initial_cursors: Vec<usize> = vec![0; n];
        let total = self.total_operations();

        // DFS stack: (cursors, partial_steps, context_switches, last_txn_idx)
        let mut stack: Vec<(Vec<usize>, Vec<ScheduleStep>, usize, Option<usize>)> = Vec::new();
        stack.push((initial_cursors, Vec::with_capacity(total), 0, None));

        while let Some((cursors, steps, switches, last_txn)) = stack.pop() {
            if steps.len() == total {
                let txn_order = self.extract_order(&steps);
                let mut schedule = Schedule {
                    steps,
                    transaction_order: txn_order,
                    metadata: ScheduleMetadata::default(),
                };
                results.push(schedule);
                self.generated += 1;
                // Bail if we've generated too many (safety valve)
                if results.len() >= 100_000 {
                    break;
                }
                continue;
            }

            for i in 0..n {
                if cursors[i] >= self.transactions[i].len() {
                    continue;
                }
                let is_switch = last_txn.map_or(false, |lt| lt != i);
                let new_switches = switches + if is_switch { 1 } else { 0 };

                if let Some(bound) = self.bound_k {
                    if new_switches > bound {
                        continue;
                    }
                }

                let mut new_cursors = cursors.clone();
                let op = &self.transactions[i].ops[cursors[i]];
                new_cursors[i] += 1;

                let mut new_steps = steps.clone();
                new_steps.push(ScheduleStep {
                    id: ScheduleStepId::new(new_steps.len() as u64),
                    txn_id: self.transactions[i].txn_id,
                    operation: op.clone(),
                    position: new_steps.len(),
                });

                stack.push((new_cursors, new_steps, new_switches, Some(i)));
            }
        }

        results
    }

    fn extract_order(&self, steps: &[ScheduleStep]) -> Vec<TransactionId> {
        let mut seen = std::collections::HashSet::new();
        let mut order = Vec::new();
        for step in steps {
            if seen.insert(step.txn_id) {
                order.push(step.txn_id);
            }
        }
        order
    }
}

// ---------------------------------------------------------------------------
// Random schedule sampling
// ---------------------------------------------------------------------------

/// Simple deterministic PRNG (xorshift64) for reproducible sampling.
#[derive(Debug, Clone)]
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }
}

/// Samples random interleavings.
#[derive(Debug)]
pub struct RandomScheduleSampler {
    transactions: Vec<TransactionOps>,
    rng: Xorshift64,
    generated: u64,
}

impl RandomScheduleSampler {
    pub fn new(transactions: Vec<TransactionOps>, seed: u64) -> Self {
        Self {
            transactions,
            rng: Xorshift64::new(seed),
            generated: 0,
        }
    }

    /// Sample one random interleaving.
    pub fn sample_one(&mut self) -> Schedule {
        let n = self.transactions.len();
        let total: usize = self.transactions.iter().map(|t| t.len()).sum();
        let mut cursors = vec![0usize; n];
        let mut steps = Vec::with_capacity(total);

        while steps.len() < total {
            // Find transactions that still have operations
            let eligible: Vec<usize> = (0..n)
                .filter(|&i| cursors[i] < self.transactions[i].len())
                .collect();
            if eligible.is_empty() {
                break;
            }
            let chosen_idx = eligible[self.rng.next_usize(eligible.len())];
            let op = &self.transactions[chosen_idx].ops[cursors[chosen_idx]];
            steps.push(ScheduleStep {
                id: ScheduleStepId::new(steps.len() as u64),
                txn_id: self.transactions[chosen_idx].txn_id,
                operation: op.clone(),
                position: steps.len(),
            });
            cursors[chosen_idx] += 1;
        }

        let mut seen = std::collections::HashSet::new();
        let txn_order: Vec<_> = steps
            .iter()
            .filter_map(|s| if seen.insert(s.txn_id) { Some(s.txn_id) } else { None })
            .collect();

        self.generated += 1;
        Schedule {
            steps,
            transaction_order: txn_order,
            metadata: ScheduleMetadata::default(),
        }
    }

    /// Sample `n` random interleavings.
    pub fn sample(&mut self, n: usize) -> Vec<Schedule> {
        (0..n).map(|_| self.sample_one()).collect()
    }

    pub fn generated_count(&self) -> u64 {
        self.generated
    }
}

// ---------------------------------------------------------------------------
// Utility: generate all permutations of 0..n
// ---------------------------------------------------------------------------

fn permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![vec![]];
    }
    let mut result = Vec::new();
    let mut current: Vec<usize> = (0..n).collect();
    permute_heap(&mut current, n, &mut result);
    result
}

fn permute_heap(arr: &mut Vec<usize>, k: usize, result: &mut Vec<Vec<usize>>) {
    if k == 1 {
        result.push(arr.clone());
        return;
    }
    permute_heap(arr, k - 1, result);
    for i in 0..(k - 1) {
        if k % 2 == 0 {
            arr.swap(i, k - 1);
        } else {
            arr.swap(0, k - 1);
        }
        permute_heap(arr, k - 1, result);
    }
}

// ---------------------------------------------------------------------------
// Schedule analysis utilities
// ---------------------------------------------------------------------------

/// Count the number of context switches in a schedule.
pub fn context_switches(schedule: &Schedule) -> usize {
    let mut switches = 0;
    for i in 1..schedule.steps.len() {
        if schedule.steps[i].txn_id != schedule.steps[i - 1].txn_id {
            switches += 1;
        }
    }
    switches
}

/// Check whether a schedule is serial (zero context switches within
/// committed transaction boundaries – each transaction's ops are contiguous).
pub fn is_serial(schedule: &Schedule) -> bool {
    context_switches(schedule) == 0
        || {
            let mut seen_finished = std::collections::HashSet::new();
            let mut current_txn: Option<TransactionId> = None;
            for step in &schedule.steps {
                if Some(step.txn_id) != current_txn {
                    if seen_finished.contains(&step.txn_id) {
                        return false;
                    }
                    if let Some(prev) = current_txn {
                        seen_finished.insert(prev);
                    }
                    current_txn = Some(step.txn_id);
                }
            }
            true
        }
}

/// Count distinct transaction ids in a schedule.
pub fn distinct_transactions(schedule: &Schedule) -> usize {
    let ids: std::collections::HashSet<_> = schedule.steps.iter().map(|s| s.txn_id).collect();
    ids.len()
}

/// Compute the concurrency degree (max overlapping transactions at any point).
pub fn concurrency_degree(schedule: &Schedule) -> usize {
    let mut active: std::collections::HashSet<TransactionId> = std::collections::HashSet::new();
    let mut max_concurrent = 0usize;

    for step in &schedule.steps {
        match &step.operation.kind {
            OpKind::Begin(_) => {
                active.insert(step.txn_id);
            }
            OpKind::Commit(_) | OpKind::Abort(_) => {
                active.remove(&step.txn_id);
            }
            _ => {
                active.insert(step.txn_id);
            }
        }
        max_concurrent = max_concurrent.max(active.len());
    }
    max_concurrent
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use isospec_types::identifier::{OperationId, TableId, ItemId};

    fn make_txn(txn_id: u64, num_ops: usize) -> TransactionOps {
        let tid = TransactionId::new(txn_id);
        let tbl = TableId::new(0);
        let ops: Vec<Operation> = (0..num_ops)
            .map(|i| {
                Operation::read(
                    OperationId::new(txn_id * 100 + i as u64),
                    tid,
                    tbl,
                    ItemId::new(i as u64),
                    0,
                )
            })
            .collect();
        TransactionOps::new(tid, ops)
    }

    #[test]
    fn test_serial_schedules_two_txns() {
        let txns = vec![make_txn(1, 2), make_txn(2, 2)];
        let mut enumerator = ScheduleEnumerator::new(txns);
        let serials = enumerator.serial_schedules();
        // 2 transactions → 2! = 2 serial schedules
        assert_eq!(serials.len(), 2);
        for s in &serials {
            assert_eq!(s.steps.len(), 4);
        }
    }

    #[test]
    fn test_serial_schedules_three_txns() {
        let txns = vec![make_txn(1, 1), make_txn(2, 1), make_txn(3, 1)];
        let mut enumerator = ScheduleEnumerator::new(txns);
        let serials = enumerator.serial_schedules();
        assert_eq!(serials.len(), 6); // 3!
    }

    #[test]
    fn test_bounded_enumeration() {
        let txns = vec![make_txn(1, 2), make_txn(2, 2)];
        // bound_k = 0 → only serial (no switches)
        let mut enumerator = ScheduleEnumerator::bounded(txns.clone(), 0);
        let schedules = enumerator.enumerate_all();
        // With 0 switches allowed, only serial orderings: T1;T2 and T2;T1
        assert_eq!(schedules.len(), 2);

        // bound_k = 1 → allow 1 context switch
        let mut enumerator1 = ScheduleEnumerator::bounded(txns.clone(), 1);
        let schedules1 = enumerator1.enumerate_all();
        assert!(schedules1.len() > 2);
    }

    #[test]
    fn test_enumerate_all_small() {
        let txns = vec![make_txn(1, 1), make_txn(2, 1)];
        let mut enumerator = ScheduleEnumerator::new(txns);
        let all = enumerator.enumerate_all();
        // 2 ops from 2 txns: (2!)/(1!*1!) = 2 interleavings
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_random_sampler() {
        let txns = vec![make_txn(1, 3), make_txn(2, 3)];
        let mut sampler = RandomScheduleSampler::new(txns, 42);
        let samples = sampler.sample(10);
        assert_eq!(samples.len(), 10);
        assert_eq!(sampler.generated_count(), 10);
        for s in &samples {
            assert_eq!(s.steps.len(), 6);
        }
    }

    #[test]
    fn test_context_switches() {
        let txns = vec![make_txn(1, 2), make_txn(2, 2)];
        let mut enumerator = ScheduleEnumerator::new(txns);
        let serials = enumerator.serial_schedules();
        // Serial schedules have 1 context switch (switch once between txns)
        for s in &serials {
            assert_eq!(context_switches(s), 1);
        }
    }

    #[test]
    fn test_is_serial() {
        let txns = vec![make_txn(1, 2), make_txn(2, 2)];
        let mut enumerator = ScheduleEnumerator::new(txns);
        let serials = enumerator.serial_schedules();
        for s in &serials {
            assert!(is_serial(s));
        }
    }

    #[test]
    fn test_permutations_count() {
        assert_eq!(permutations(0).len(), 1);
        assert_eq!(permutations(1).len(), 1);
        assert_eq!(permutations(2).len(), 2);
        assert_eq!(permutations(3).len(), 6);
        assert_eq!(permutations(4).len(), 24);
    }

    #[test]
    fn test_distinct_transactions() {
        let txns = vec![make_txn(1, 2), make_txn(2, 2), make_txn(3, 2)];
        let mut enumerator = ScheduleEnumerator::new(txns);
        let serials = enumerator.serial_schedules();
        for s in &serials {
            assert_eq!(distinct_transactions(s), 3);
        }
    }
}
