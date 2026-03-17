// absolute_value.ms — Absolute value
// Demonstrates: ROR and UOI mutation operators on a simple branching function.

// @mutate: ROR, UOI
function abs(x: int) -> int {
    if x < 0 {
        return -x;
    }
    return x;
}

// --- Expected synthesized contract ---
// @requires: true
// @ensures:  ret >= 0
// @ensures:  (x >= 0) -> ret == x
// @ensures:  (x < 0)  -> ret == -x

// --- Mutation targets ---
// M1  ROR  x < 0  →  x <= 0   (killed by: abs(0) should be 0, not -0 ≡ 0 ... survives; tighten)
// M2  ROR  x < 0  →  x > 0    (killed by: abs(-1) should be 1, returns -1)
// M3  ROR  x < 0  →  x == 0   (killed by: abs(-5) should be 5, returns -5)
// M4  UOI  -x     →  x        (killed by: abs(-3) should be 3, returns -3)
// M5  UOI  x      →  -x       (in else branch; killed by: abs(3) should be 3, returns -3)

// --- Test vectors ---
// abs(5)  == 5
// abs(-5) == 5
// abs(0)  == 0
