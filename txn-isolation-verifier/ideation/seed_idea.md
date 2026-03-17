Community: area-086-data-management-and-databases

Idea: A static analysis engine that ingests SQL transaction programs and a target isolation level (e.g., READ COMMITTED, SNAPSHOT), uses SMT-based symbolic schedule enumeration to exhaustively detect concurrency anomalies (write skew, lost updates, phantom reads, read skew) across all possible interleavings, and synthesizes minimal reproducing test harnesses with concrete data and step-by-step interleaving witnesses.
