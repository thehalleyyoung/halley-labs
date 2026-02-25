# PSI Security Analysis: Semi-Honest Model Under Commit-Then-Execute

> **Status**: Security analysis addressing the critique that semi-honest PSI contradicts the adversarial threat model.

---

## 1. Threat Model

Spectacles operates in a **two-party setting**:

- **Model Provider (P)**: holds the training corpus n-gram set $C_\text{train}$
- **Evaluator (E)**: holds the benchmark n-gram set $C_\text{bench}$

**Goal**: Compute $|C_\text{train} \cap C_\text{bench}|$ (or determine whether it exceeds threshold $\tau$) without revealing either set to the other party.

**Adversary model**: The provider P is potentially **malicious** — it may deviate from the protocol to hide contamination. The evaluator E is assumed semi-honest (it has no incentive to hide contamination).

---

## 2. The Critique

The semi-honest PSI assumption contradicts the adversarial threat model:

> A malicious provider could deviate from the PSI protocol to reduce the reported intersection cardinality, thereby hiding contamination.

Specific attack vectors under a malicious model:
1. **Input substitution**: P uses a sanitized n-gram set $C' \subset C_\text{train}$ with contaminating n-grams removed.
2. **Selective abort**: P aborts the protocol when the overlap would exceed $\tau$.
3. **OPRF deviation**: P evaluates the OPRF incorrectly on selected inputs to suppress matches.

---

## 3. Commit-Then-Execute Defense

Spectacles addresses these attacks through a **commitment-binding protocol** that transforms the problem from requiring malicious-model PSI to one where semi-honest PSI suffices.

### 3.1 Protocol Structure

```
Phase 1: COMMITMENT
  P computes: h = BLAKE3(sort(C_train) || nonce || timestamp)
  P publishes: (h, timestamp) to public bulletin / E

Phase 2: VERIFICATION
  E verifies: timestamp is before any PSI message exchange
  E stores: h for post-protocol verification

Phase 3: PSI EXECUTION (semi-honest)
  Standard OPRF-based PSI runs between P and E
  P's inputs must be consistent with the committed set

Phase 4: POST-PROTOCOL CHECK
  P reveals C_train to E (or to an auditor)
  E verifies: BLAKE3(sort(C_train) || nonce || timestamp) == h
  If verification fails: certificate is invalidated
```

### 3.2 Implementation

The commitment binding is implemented in `spectacles-core/src/psi/protocol.rs`:

- **`CommitmentBinding`** struct: `commit()`, `verify()`, `commit_ngram_set()`, `verify_ngram_set()`
- **`InputVerifier`** struct: `pre_protocol_check()` validates both temporal ordering and commitment consistency
- Uses BLAKE3 with random nonce for collision-resistant binding

---

## 4. Attack Vector Analysis

### 4.1 Input Substitution

**Attack**: P commits to $C_\text{train}$ but uses $C' \neq C_\text{train}$ in the PSI.

**Defense**: The commitment $h = H(\text{sort}(C_\text{train}) \| r)$ is binding: finding $C' \neq C_\text{train}$ such that $H(\text{sort}(C') \| r') = h$ requires breaking collision resistance of BLAKE3 (128-bit security).

**Formal argument**: Under the collision-resistance assumption on $H$:
$$\Pr[\text{Adv finds } C' \neq C_\text{train} : H(C' \| r') = H(C_\text{train} \| r)] \leq \text{negl}(\lambda)$$

**Verification**: Post-protocol, E (or an auditor) verifies that P's revealed set matches the commitment. The PSI result is only valid if this check passes.

### 4.2 Selective Abort

**Attack**: P aborts the protocol when the overlap exceeds $\tau$.

**Defense**: The protocol requires **completion** for a valid certificate:
- An aborted protocol produces **no attestation**
- The absence of a certificate is itself informative (the evaluator knows P refused)
- Certificate validity requires both the STARK proof AND the PSI result

**Practical consideration**: Selective abort is a **denial of service**, not a false attestation. The evaluator learns that P was unwilling to complete the protocol, which is a meaningful signal. No false claim of low contamination is possible.

### 4.3 OPRF Deviation

**Attack**: P evaluates the OPRF incorrectly on selected inputs to suppress matches.

**Defense**: OPRF deviation is detectable because:
1. The OPRF key $k$ is committed before the protocol begins
2. Spot-check verification: E can include known test elements in the PSI set; incorrect OPRF evaluation on these elements reveals deviation
3. For a verifiable OPRF (VOPRF), the server provides a proof of correct evaluation for each element

**Residual risk**: Without VOPRF, a sophisticated provider could selectively corrupt a small fraction of OPRF evaluations. This is acknowledged as a limitation (§5).

---

## 5. Limitations (Honestly Stated)

1. **No UC security**: The protocol does not achieve universally composable (UC) security. A full UC analysis with ideal functionality specification is future work.

2. **Collusion**: A malicious provider colluding with the benchmark maintainer is **outside the threat model**. If both parties cooperate to produce a false certificate, commitment binding does not help.

3. **OPRF without verifiability**: Without a verifiable OPRF (VOPRF), the provider could corrupt a small fraction of evaluations. Using a VOPRF (e.g., based on Dodis-Yampolskiy VRF) would close this gap at the cost of additional computational overhead.

4. **Commitment revelation**: The post-protocol verification step requires P to reveal $C_\text{train}$ (or a hash chain allowing incremental verification). This partially undermines the privacy of the PSI, since E eventually learns the full training set. A zero-knowledge proof of commitment consistency would address this but adds protocol complexity.

5. **Timing attacks**: The timestamp-based temporal ordering can be spoofed if P and E do not use a trusted time source. In practice, blockchain-based timestamping or a trusted third-party timestamp service mitigates this.

---

## 6. Summary

| Attack | Defense | Assumption | Status |
|--------|---------|------------|--------|
| Input substitution | Commitment binding (BLAKE3) | Collision resistance | ✅ Implemented & tested |
| Selective abort | Completion required for valid certificate | — | ✅ Implemented |
| OPRF deviation | Spot-check + VOPRF (optional) | VOPRF correctness | ⚠️ Partial (VOPRF not implemented) |
| Collusion | Out of threat model | — | ⚠️ Acknowledged |

**Conclusion**: The commit-then-execute framework reduces the security requirement from malicious-model PSI to semi-honest PSI by ensuring that (1) inputs are committed before the protocol runs, (2) the protocol must complete for a valid certificate, and (3) commitment consistency is verifiable post-protocol. This does not provide full malicious security but neutralizes the primary attack vectors identified in the critique.

---

## 7. Test Coverage

The following tests in `spectacles-core/src/psi/protocol.rs` verify the commitment binding mechanism:

| Test | Verifies |
|------|----------|
| `test_commitment_binding_roundtrip` | Correct commitment/verification cycle |
| `test_commitment_binding_rejects_different_set` | Different set fails verification |
| `test_commitment_binding_rejects_subset` | Subset of committed set fails |
| `test_commitment_binding_rejects_superset` | Superset fails |
| `test_commitment_binding_order_independent` | Set ordering doesn't affect commitment |
| `test_input_verifier_pre_protocol_check_valid` | Full pre-protocol check passes |
| `test_input_verifier_pre_protocol_check_temporal_violation` | Out-of-order timestamps detected |

Additionally, `implementation/tests/test_semiring_axioms.py` provides independent Python verification of the mathematical foundations.
