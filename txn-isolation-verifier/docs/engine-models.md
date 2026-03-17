# Engine Model Specifications

## PostgreSQL 16.x — Serializable Snapshot Isolation (SSI)

### Concurrency Control Mechanism
PostgreSQL implements SERIALIZABLE through SSI, which extends Snapshot
Isolation with runtime detection of serialization anomalies.

### State Components
- **SIREAD locks**: Predicate-level read locks tracking which tuples/ranges
  each transaction has read
- **Write locks**: Standard exclusive locks on modified tuples
- **rw-dependency graph**: Tracks anti-dependencies between concurrent
  transactions
- **Commit order**: Total order of committed transactions

### Dangerous Structure Detection
SSI detects the pattern: T₁ →rw T₂ →rw T₃ where T₁ and T₃ are committed.
The pivot transaction T₂ is aborted to prevent serialization anomalies.

### Read-Only Optimization
Read-only transactions with no outgoing rw-edges in the dependency graph
can safely commit without dangerous-structure checking, reducing false
abort rates.

---

## MySQL 8.0 — InnoDB Gap Locking

### Concurrency Control Mechanism
InnoDB uses a combination of record locks, gap locks, and next-key locks
on B-tree index entries to prevent phantoms and maintain REPEATABLE READ.

### Lock Types
- **Record lock**: Locks a specific index record
- **Gap lock**: Locks the gap before an index record (prevents inserts)
- **Next-key lock**: Record lock + gap lock on the preceding gap

### Index Dependence
The critical subtlety: lock ranges depend on which index the query optimizer
selects. The same `SELECT ... WHERE` query acquires different locks with
different indexes, creating index-dependent isolation guarantees.

### IsoSpec Over-Approximation
IsoSpec models the union of lock ranges across all possible index choices.
This is sound (no missed anomalies) but conservative (may report false
positives when a specific index choice would prevent an anomaly).

---

## SQL Server 2022 — Dual-Mode Concurrency

### Pessimistic Mode (Lock-Based)
Key-range locks: RangeS-S, RangeS-U, RangeI-N, RangeX-X providing
range-level isolation through lock escalation hierarchies.

### Optimistic Mode (Row Versioning)
Row versions stored in tempdb with conflict detection at commit time.
Selected by `ALTER DATABASE SET READ_COMMITTED_SNAPSHOT ON`.

### Interaction Semantics
The two modes interact when transactions at different isolation levels
execute concurrently, creating a complex mixed-mode interaction space.
