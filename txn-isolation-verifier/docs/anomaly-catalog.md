# Anomaly Catalog

Complete taxonomy of transaction isolation anomalies detected by IsoSpec,
based on Adya's Direct Serialization Graph formalism.

## G0 — Write Cycles

**Definition:** The DSG contains a cycle consisting entirely of ww
(write-write) edges.

**Example:** T1 writes x=1, T2 writes x=2, T1 writes y=1, T2 writes y=2,
creating a cycle T1 →ww T2 →ww T1.

**Prevented by:** All isolation levels including READ UNCOMMITTED.

## G1a — Aborted Reads (Dirty Reads)

**Definition:** A committed transaction reads a value written by an
aborted transaction.

**Example:** T1 writes x=1, T2 reads x=1, T1 aborts. T2 has read a
value that "never existed."

**Prevented by:** READ COMMITTED and above.

## G1b — Intermediate Reads

**Definition:** A committed transaction reads an intermediate value
written by another transaction (not its final committed value).

**Example:** T1 writes x=1, T2 reads x=1, T1 writes x=2, T1 commits.
T2 read an intermediate state of T1.

**Prevented by:** READ COMMITTED and above.

## G1c — Circular Information Flow

**Definition:** The DSG contains a cycle consisting of ww and wr edges
among committed transactions.

**Example:** T1 writes x, T2 reads x (wr edge), T2 writes y, T1 reads
y (wr edge), creating circular information flow.

**Prevented by:** READ COMMITTED and above.

## G2-item — Item Anti-Dependency Cycles

**Definition:** The DSG contains a cycle with at least one item-level rw
(anti-dependency) edge among committed transactions.

**Example (Write Skew):** T1 reads x and y, T2 reads x and y, T1 writes
x, T2 writes y. The rw edges T1→T2 (T1 read y, T2 wrote y) and T2→T1
(T2 read x, T1 wrote x) form a cycle.

**Prevented by:** REPEATABLE READ and above.

## G2 — Predicate Anti-Dependency Cycles

**Definition:** The DSG contains a cycle with at least one predicate-level
rw edge. This captures phantom anomalies where new tuples inserted by one
transaction fall within the predicate range of another's scan.

**Example (Phantom):** T1 reads all rows WHERE status='active' (finding 5),
T2 inserts a new row with status='active', T2 commits. T1's predicate
read is now stale.

**Prevented by:** SERIALIZABLE only.

## G-SIa, G-SIb — Snapshot Isolation Anomalies

**G-SIa:** Write skew under Snapshot Isolation where concurrent
transactions read overlapping data and make disjoint writes.

**G-SIb:** A transaction observes a state that is not consistent with
any serial ordering due to snapshot timing.

**Prevented by:** True SERIALIZABLE (not SI).
