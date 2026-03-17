// basic_arithmetic.ms — Clamp function
// Demonstrates: ROR and AOR mutation operators on arithmetic comparisons.
// The synthesizer should discover range-bounding pre/postconditions.

// @mutate: ROR, AOR
function clamp(x: int, lo: int, hi: int) -> int {
    if x < lo {
        return lo;
    }
    if x > hi {
        return hi;
    }
    return x;
}

// --- Expected synthesized contract ---
// @requires: lo <= hi
// @ensures:  ret >= lo
// @ensures:  ret <= hi
// @ensures:  (ret == x) || (ret == lo) || (ret == hi)

// --- Mutation targets ---
// M1  ROR  x < lo   →  x <= lo    (killed by: x == lo → ret should be x, not lo)
// M2  ROR  x < lo   →  x > lo     (killed by: x < lo → ret must be lo)
// M3  ROR  x > hi   →  x >= hi    (killed by: x == hi → ret should be x, not hi)
// M4  ROR  x > hi   →  x < hi     (killed by: x > hi → ret must be hi)
// M5  AOR  return lo →  return lo+1 (killed by: ret >= lo postcondition)
// M6  AOR  return hi →  return hi-1 (killed by: ret <= hi postcondition)

// --- Test vectors (for mutation score validation) ---
// clamp(5, 0, 10)  == 5
// clamp(-3, 0, 10) == 0
// clamp(15, 0, 10) == 10
// clamp(0, 0, 10)  == 0   (boundary)
// clamp(10, 0, 10) == 10  (boundary)
