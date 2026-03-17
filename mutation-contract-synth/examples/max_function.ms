// max_function.ms — Maximum of two integers
// Demonstrates: ROR mutations on a single relational comparison.

// @mutate: ROR
function max(a: int, b: int) -> int {
    if a >= b {
        return a;
    }
    return b;
}

// --- Expected synthesized contract ---
// @requires: true
// @ensures:  ret >= a
// @ensures:  ret >= b
// @ensures:  ret == a || ret == b

// --- Mutation targets ---
// M1  ROR  a >= b  →  a > b    (killed by: a==b → should return a, but falls through to b)
// M2  ROR  a >= b  →  a <= b   (killed by: a>b → should return a, returns b)
// M3  ROR  a >= b  →  a == b   (killed by: a>b → should return a, returns b)
// M4  ROR  a >= b  →  a != b   (killed by: a==b → falls through, returns b instead of a)
// M5  ROR  a >= b  →  a < b    (killed by: a>b → returns b instead of a)

// --- Test vectors ---
// max(3, 5) == 5
// max(5, 3) == 5
// max(4, 4) == 4
// max(-1, -3) == -1
// max(0, 0) == 0
