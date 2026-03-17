// linear_search.ms — Unrolled linear search in a bounded array
// Demonstrates: ROR, AOR, and UOI operators on an unrolled loop.
// The loop is manually unrolled to keep the program loop-free (QF-LIA).

// @mutate: ROR, AOR, UOI
// @array_bound: 4
function linear_search(arr: int[4], key: int) -> int {
    // Unrolled search over 4 elements; returns index or -1 if not found.

    if arr[0] == key {
        return 0;
    }
    if arr[1] == key {
        return 1;
    }
    if arr[2] == key {
        return 2;
    }
    if arr[3] == key {
        return 3;
    }
    return -1;
}

// --- Expected synthesized contract ---
// @requires: true
// @ensures:  ret >= -1
// @ensures:  ret <= 3
// @ensures:  (ret >= 0) -> arr[ret] == key
// @ensures:  (ret == -1) -> (arr[0] != key && arr[1] != key
//                            && arr[2] != key && arr[3] != key)

// --- Mutation targets ---
// M1   ROR  arr[0] == key  →  arr[0] != key   (negation; false positive index 0)
// M2   ROR  arr[0] == key  →  arr[0] < key    (wrong comparator)
// M3   AOR  return 0       →  return 1         (off-by-one index)
// M4   AOR  return 1       →  return 0         (swapped indices)
// M5   AOR  return -1      →  return 0         (not-found sentinel lost)
// M6   UOI  key            →  -key             (sign flip on search key)
// M7   ROR  arr[2] == key  →  arr[2] >= key    (weakened equality)

// --- Test vectors ---
// linear_search([10, 20, 30, 40], 30) == 2
// linear_search([10, 20, 30, 40], 10) == 0
// linear_search([10, 20, 30, 40], 40) == 3
// linear_search([10, 20, 30, 40], 99) == -1
// linear_search([5, 5, 5, 5], 5) == 0  (first occurrence)
