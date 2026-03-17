// array_bounds.ms — Bounded-array safe access
// Demonstrates: index validation with ROR, UOI, and AOR mutation operators.
// Programs operate over bounded arrays (length known at synthesis time).

// @mutate: ROR, UOI, AOR
// @array_bound: 8
function safe_get(arr: int[8], idx: int, default_val: int) -> int {
    if idx < 0 {
        return default_val;
    }
    if idx >= 8 {
        return default_val;
    }
    return arr[idx];
}

// --- Expected synthesized contract ---
// @requires: true  (no precondition needed; function is total)
// @ensures:  (idx >= 0 && idx < 8) -> ret == arr[idx]
// @ensures:  (idx < 0 || idx >= 8) -> ret == default_val

// --- Mutation targets ---
// M1  ROR  idx < 0   →  idx <= 0    (boundary: idx==0 is valid)
// M2  ROR  idx < 0   →  idx > 0     (negation: negative indices accepted)
// M3  ROR  idx >= 8  →  idx > 8     (boundary: idx==8 is out of bounds)
// M4  ROR  idx >= 8  →  idx >= 7    (shrinks valid range)
// M5  UOI  idx       →  -idx        (in first guard: sign flip)
// M6  AOR  8         →  8 + 1       (bound shifted to 9)
// M7  AOR  8         →  8 - 1       (bound shifted to 7)

// @mutate: ROR, AOR
// @array_bound: 8
function safe_set(arr: int[8], idx: int, val: int) -> int[8] {
    if idx < 0 {
        return arr;
    }
    if idx >= 8 {
        return arr;
    }
    arr[idx] = val;
    return arr;
}

// --- Expected synthesized contract ---
// @requires: true
// @ensures:  (idx >= 0 && idx < 8) -> ret[idx] == val
// @ensures:  (idx < 0 || idx >= 8) -> ret == arr

// --- Test vectors ---
// safe_get([1,2,3,4,5,6,7,8], 0, -1) == 1
// safe_get([1,2,3,4,5,6,7,8], 7, -1) == 8
// safe_get([1,2,3,4,5,6,7,8], -1, -1) == -1
// safe_get([1,2,3,4,5,6,7,8], 8, -1)  == -1
