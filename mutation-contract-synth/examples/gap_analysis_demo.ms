// gap_analysis_demo.ms — Safe integer division with gap analysis
// Demonstrates: gap analysis where mutation testing reveals a missing
// precondition that dynamic testing alone might miss.
// The subtle bug: when divisor == 0, the function returns 0 silently
// instead of signaling an error. Mutation analysis on the guard
// exposes that the contract must require divisor != 0.

// @mutate: ROR, AOR, UOI
function safe_divide(dividend: int, divisor: int) -> int {
    if divisor == 0 {
        return 0;
    }
    if dividend >= 0 {
        // positive dividend: truncate toward zero
        if divisor > 0 {
            // --- unrolled division for small quotients (QF-LIA) ---
            if dividend >= divisor + divisor + divisor {
                return 3;
            }
            if dividend >= divisor + divisor {
                return 2;
            }
            if dividend >= divisor {
                return 1;
            }
            return 0;
        }
        // divisor < 0, dividend >= 0
        if dividend >= -divisor + -divisor + -divisor {
            return -3;
        }
        if dividend >= -divisor + -divisor {
            return -2;
        }
        if dividend >= -divisor {
            return -1;
        }
        return 0;
    }
    // dividend < 0 — symmetric case
    if divisor > 0 {
        if -dividend >= divisor + divisor + divisor {
            return -3;
        }
        if -dividend >= divisor + divisor {
            return -2;
        }
        if -dividend >= divisor {
            return -1;
        }
        return 0;
    }
    // both negative
    if -dividend >= -divisor + -divisor + -divisor {
        return 3;
    }
    if -dividend >= -divisor + -divisor {
        return 2;
    }
    if -dividend >= -divisor {
        return 1;
    }
    return 0;
}

// --- Expected synthesized contract ---
// @requires: divisor != 0
// @ensures:  (divisor > 0 && dividend >= 0) -> ret >= 0
// @ensures:  (divisor > 0 && dividend < 0)  -> ret <= 0
// @ensures:  ret >= -3
// @ensures:  ret <= 3

// --- Gap analysis ---
// The function silently returns 0 when divisor == 0. Dynamic testing
// with random inputs rarely generates divisor == 0, so this gap is
// not detected. Mutation analysis on the guard `divisor == 0` (ROR →
// `divisor != 0`, `divisor > 0`, `divisor < 0`) reveals that the
// contract must include the precondition `divisor != 0` to kill all
// mutants at tier 1.

// --- Mutation targets ---
// M1  ROR  divisor == 0  →  divisor != 0   (inverts guard; all inputs hit division)
// M2  ROR  divisor == 0  →  divisor > 0    (allows negative divisor through guard)
// M3  ROR  divisor == 0  →  divisor < 0    (allows positive divisor through guard)
// M4  ROR  dividend >= 0 →  dividend > 0   (boundary: dividend == 0 misclassified)
// M5  AOR  divisor + divisor → divisor - divisor  (threshold collapses)
// M6  UOI  -divisor      →  divisor        (sign flip on negated divisor)
// M7  AOR  return 3      →  return 2       (off-by-one quotient)
// M8  AOR  return -3     →  return -2      (off-by-one quotient)

// --- Test vectors ---
// safe_divide(10, 3)   == 3
// safe_divide(6, 3)    == 2
// safe_divide(2, 3)    == 0
// safe_divide(-10, 3)  == -3
// safe_divide(10, -3)  == -3
// safe_divide(-10, -3) == 3
// safe_divide(0, 5)    == 0
// safe_divide(5, 0)    == 0   (gap: silently returns 0)
