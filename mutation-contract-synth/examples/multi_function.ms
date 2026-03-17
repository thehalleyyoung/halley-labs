// multi_function.ms — Multiple related functions (inter-procedural)
// Demonstrates: contract synthesis across multiple functions that share
// a logical relationship. The synthesizer should discover contracts that
// are consistent across min, max, clamp_pair, and midpoint.

// @mutate: ROR
function min(a: int, b: int) -> int {
    if a <= b {
        return a;
    }
    return b;
}

// --- Expected synthesized contract for min ---
// @requires: true
// @ensures:  ret <= a
// @ensures:  ret <= b
// @ensures:  ret == a || ret == b

// @mutate: ROR
function max(a: int, b: int) -> int {
    if a >= b {
        return a;
    }
    return b;
}

// --- Expected synthesized contract for max ---
// @requires: true
// @ensures:  ret >= a
// @ensures:  ret >= b
// @ensures:  ret == a || ret == b

// @mutate: ROR, AOR
function clamp_pair(x: int, lo: int, hi: int) -> int {
    // Uses min/max logic inline (loop-free requirement).
    // clamp_pair(x, lo, hi) == max(lo, min(x, hi))
    if x < lo {
        return lo;
    }
    if x > hi {
        return hi;
    }
    return x;
}

// --- Expected synthesized contract for clamp_pair ---
// @requires: lo <= hi
// @ensures:  ret >= lo
// @ensures:  ret <= hi
// @ensures:  ret == x || ret == lo || ret == hi

// @mutate: ROR, AOR, UOI
function midpoint(a: int, b: int) -> int {
    // Integer midpoint avoiding overflow: a + (b - a) / 2
    // Unrolled for QF-LIA: uses conditional halving.
    if a == b {
        return a;
    }
    if a < b {
        // diff = b - a; half = diff / 2 (integer division toward zero)
        if b - a >= 4 {
            return a + 2;
        }
        if b - a >= 2 {
            return a + 1;
        }
        // diff == 1: midpoint rounds down to a
        return a;
    }
    // a > b: symmetric
    if a - b >= 4 {
        return b + 2;
    }
    if a - b >= 2 {
        return b + 1;
    }
    return b;
}

// --- Expected synthesized contract for midpoint ---
// @requires: true
// @ensures:  (a <= b) -> (ret >= a && ret <= b)
// @ensures:  (a > b)  -> (ret >= b && ret <= a)
// @ensures:  ret == a || ret == b || (ret > a && ret < b) || (ret > b && ret < a)

// --- Inter-procedural consistency ---
// The following relationships should hold across function contracts:
//   min(a, b) <= midpoint(a, b) <= max(a, b)
//   clamp_pair(x, min(a,b), max(a,b)) == clamp_pair(x, a, b) when a <= b

// --- Mutation targets (selected) ---
// M1   ROR  a <= b  →  a < b   [min] (boundary: a==b should return a)
// M2   ROR  a >= b  →  a > b   [max] (boundary: a==b should return a)
// M3   ROR  x < lo  →  x <= lo [clamp_pair] (boundary: x==lo returns lo not x)
// M4   ROR  x > hi  →  x >= hi [clamp_pair] (boundary: x==hi returns hi not x)
// M5   AOR  return lo  →  return lo + 1 [clamp_pair] (off-by-one)
// M6   AOR  return hi  →  return hi - 1 [clamp_pair] (off-by-one)
// M7   ROR  a == b  →  a != b  [midpoint] (equal case lost)
// M8   ROR  a < b   →  a <= b  [midpoint] (overlaps with == case)
// M9   AOR  b - a >= 4  →  b - a >= 3  [midpoint] (threshold shift)
// M10  AOR  a + 2   →  a + 3   [midpoint] (off-by-one midpoint)
// M11  AOR  a + 1   →  a + 0   [midpoint] (increment lost)
// M12  UOI  a       →  -a      [midpoint] (sign flip)

// --- Test vectors ---
// min(3, 5) == 3
// min(5, 3) == 3
// min(4, 4) == 4
// max(3, 5) == 5
// max(5, 3) == 5
// max(4, 4) == 4
// clamp_pair(5, 0, 10)  == 5
// clamp_pair(-1, 0, 10) == 0
// clamp_pair(15, 0, 10) == 10
// clamp_pair(0, 0, 10)  == 0
// midpoint(0, 10) == 5   (not exact; unrolled approx returns a+2)
// midpoint(3, 3)  == 3
// midpoint(1, 2)  == 1
// midpoint(10, 0) == 2   (symmetric: b + 2)
// midpoint(-5, 5) == -3  (a + 2 = -5 + 2 = -3)
