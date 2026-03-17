// jml_output_demo.ms — Bounded increment with JML annotation output
// Demonstrates: JML (Java Modeling Language) annotation generation.
// The synthesized contract maps directly to JML requires/ensures clauses.

// @mutate: ROR, AOR
// @output_format: jml
function bounded_increment(val: int, lo: int, hi: int) -> int {
    if val < lo {
        return lo;
    }
    if val >= hi {
        return hi;
    }
    return val + 1;
}

// --- Expected synthesized contract ---
// @requires: lo <= hi
// @requires: val >= lo
// @requires: val <= hi
// @ensures:  ret >= lo
// @ensures:  ret <= hi
// @ensures:  ret >= val
// @ensures:  ret <= val + 1

// --- Expected JML output ---
// /*@ requires lo <= hi;
//   @ requires val >= lo;
//   @ requires val <= hi;
//   @ ensures \result >= lo;
//   @ ensures \result <= hi;
//   @ ensures \result >= val;
//   @ ensures \result <= val + 1;
//   @*/
// public static int bounded_increment(int val, int lo, int hi) { ... }

// --- Mutation targets ---
// M1  ROR  val < lo   →  val <= lo   (boundary: val == lo → should return lo+1, returns lo)
// M2  ROR  val < lo   →  val > lo    (negation: below-range values pass through)
// M3  ROR  val >= hi  →  val > hi    (boundary: val == hi → should return hi, returns hi+1)
// M4  ROR  val >= hi  →  val < hi    (negation: above-range values get incremented)
// M5  AOR  val + 1    →  val + 0     (increment lost; val returned unchanged)
// M6  AOR  val + 1    →  val - 1     (decrement instead of increment)

// --- Test vectors ---
// bounded_increment(5, 0, 10)   == 6
// bounded_increment(0, 0, 10)   == 1
// bounded_increment(9, 0, 10)   == 10
// bounded_increment(10, 0, 10)  == 10  (at upper bound: clamped)
// bounded_increment(-1, 0, 10)  == 0   (below lower bound: clamped)
// bounded_increment(5, 5, 5)    == 5   (degenerate range)
