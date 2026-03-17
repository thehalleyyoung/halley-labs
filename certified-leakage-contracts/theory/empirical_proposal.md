# Empirical Evaluation Plan: Certified Leakage Contracts

**Paper:** Certified Leakage Contracts — compositional abstract interpretation for
speculative cache side channels in x86-64 cryptographic binaries.

**Core artefact:** Reduced product domain $D_{\text{spec}} \otimes D_{\text{cache}} \otimes D_{\text{quant}}$
with reduction operator $\rho$ producing per-function quantitative leakage contracts
$(\tau_f, B_f)$ where $\tau_f$ is a cache-state transformer and $B_f : \text{CacheState} \to \mathbb{R}_{\geq 0}$
is a leakage bound in bits.

**Constraint:** All experiments run on a single laptop CPU (no cluster), fully automated,
zero human annotation.  Reproducible via a single `make eval` invocation.

---

## 1  Research Questions

We pose seven research questions.  Each states a **falsification criterion** — a concrete
outcome that, if observed, refutes the claim — a **metric** linking the question to a
measurable quantity, and a **theorem connection** tying the empirical test to the paper's
formal results.

### RQ1 — Soundness

> Does the abstract analysis never undercount true leakage?

| Aspect | Detail |
|--------|--------|
| **Metric** | Number of false negatives: functions where the abstract bound $B_f(s)$ is strictly less than the exhaustively measured channel capacity $C_f(s)$. |
| **Falsification** | Any $B_f(s) < C_f(s)$ on a small-key ($\leq 16$-bit) exhaustive enumeration constitutes a soundness violation.  Target: **0 false negatives** across the entire benchmark suite. |
| **Theorem connection** | Soundness Theorem (A.1): $\gamma(D_{\text{spec}} \otimes D_{\text{cache}} \otimes D_{\text{quant}}) \supseteq \text{Collect}_{\text{spec}}$, which implies $B_f(s) \geq C_f(s)$ for every reachable cache state $s$.  A single counter-example invalidates the proof or its implementation. |
| **Protocol** | For every benchmark function with key size $\leq 16$ bits, enumerate all $2^k$ keys, simulate the concrete speculative + cache semantics, compute the exact channel matrix, derive $C_f(s)$ via min-entropy, and check $B_f(s) \geq C_f(s)$. |

### RQ2 — Precision

> Are abstract bounds within a small constant factor of the true leakage?

| Aspect | Detail |
|--------|--------|
| **Metric** | Tightness ratio $r_f = B_f(s) \;/\; C_f(s)$.  Report distribution (median, 90th percentile, max) separately for non-speculative and speculative analysis modes. |
| **Falsification** | Non-speculative: $r_f > 3$ on median across non-speculative benchmarks.  Speculative: $r_f > 10$ on median across speculative benchmarks.  Either exceeding its threshold falsifies the precision claim. |
| **Theorem connection** | Precision is not guaranteed by soundness alone; it reflects the quality of $\rho$ and the abstract domain widening operators (Lemma D.2.3 — taint-restricted counting).  Loose bounds indicate widening fires too eagerly or $\rho$ fails to prune dead speculative paths. |
| **Protocol** | Same exhaustive enumeration as RQ1 for small keys.  For large keys ($\geq 128$ bits), approximate $C_f$ via Monte Carlo sampling ($10^6$ traces) with Kraskov–Stögbauer–Grassberger (KSG) mutual-information estimator; report 95% confidence intervals on $r_f$. |

### RQ3 — Value of the Reduction Operator $\rho$

> Does the reduced product measurably tighten bounds over the direct (non-reduced) product?

| Aspect | Detail |
|--------|--------|
| **Metric** | Reduction ratio $\rho\text{-ratio}_f = B_f^{\text{direct}} \;/\; B_f^{\text{reduced}}$ per function, where $B_f^{\text{direct}}$ uses the direct product $D_{\text{spec}} \times D_{\text{cache}} \times D_{\text{quant}}$ without inter-domain constraint propagation, and $B_f^{\text{reduced}}$ uses the full reduced product with $\rho$. |
| **Falsification** | $\rho\text{-ratio}_f \leq 1.0$ on $\geq 50\%$ of benchmarks with non-zero leakage (i.e., $\rho$ provides no improvement or actively hurts).  A ratio of exactly 1.0 on all benchmarks is also a falsification — it means the reduction adds engineering cost for zero analytical benefit. |
| **Theorem connection** | Theorem A.3 (reduction operator soundness): $\rho$ is monotone, terminates in $O(|\text{sets}| \times |\text{ways}|)$ iterations, and $\rho(\vec{d}) \sqsubseteq \vec{d}$ for all $\vec{d}$.  The empirical question is whether $\rho(\vec{d}) \sqsubset \vec{d}$ strictly on realistic programs. |
| **Protocol** | Run full analysis in two modes (direct product; reduced product) on all benchmarks.  Pair-wise compare $B_f$ values.  Report distribution of $\rho\text{-ratio}_f$ and the fraction of functions where $\rho\text{-ratio}_f > 1.0$. |

### RQ4 — Compositional vs. Monolithic Precision

> Do compositional bounds remain within 2× of monolithic whole-program bounds?

| Aspect | Detail |
|--------|--------|
| **Metric** | Composition overhead ratio $\kappa = B^{\text{comp}} \;/\; B^{\text{mono}}$ for a multi-function program, where $B^{\text{comp}} = B_f(s) + B_g(\tau_f(s))$ (sequential composition rule) and $B^{\text{mono}}$ is the monolithic whole-program analysis. |
| **Falsification** | $\kappa > 2.0$ on any of the composition benchmarks (AES full, ChaCha20 full, Curve25519 scalar multiplication).  A failure on $\geq 2$ of these benchmarks is fatal. |
| **Theorem connection** | Composition Theorem (E.2.1): $B_{f;g}(s) = B_f(s) + B_g(\tau_f(s))$ under the cache-state independence condition.  The 2× slack accounts for imprecision in $\tau_f$ propagation, not a theoretical deficiency. |
| **Protocol** | For each composition benchmark: (a) analyze constituent functions separately, compose bounds using the composition rule; (b) analyze the whole program monolithically; (c) compute $\kappa$.  At least three benchmarks: AES-128 (10 rounds composed), ChaCha20 (20 quarter-round pairs), Curve25519 (255 ladder steps, sampled at 10 points). |

### RQ5 — CVE Regression Detection

> Does the tool detect all three known CVE-related leakage changes?

| Aspect | Detail |
|--------|--------|
| **Metric** | Per-function leakage delta $\Delta_f = B_f^{\text{vuln}} - B_f^{\text{patched}}$, computed by analyzing the pre-patch (vulnerable) and post-patch (fixed) binaries.  Regression detected iff $\Delta_f > 0$ for at least one function in the vulnerable binary and $\Delta_f \approx 0$ (within noise floor $\epsilon = 0.01$ bits) for the patched binary. |
| **Falsification** | Failure to detect $\Delta_f > 0$ for any of the three CVEs, or reporting $\Delta_f > \epsilon$ on a patched binary (false regression alarm on a known-good fix).  Target: **3/3 CVEs detected with 0 false alarms on patched versions.** |
| **Theorem connection** | Regression detection uses the contract diff $\Delta_f$ which is derived from the soundness of $B_f$ (Theorem A.1).  If $B_f$ is sound, $\Delta_f > 0$ implies a genuine increase in observable leakage.  This is the "killer app" framing from the problem statement — robust to modeling imprecision because it compares two analyses, not an analysis to an absolute threshold. |
| **CVEs** | |

| CVE | Library | Vulnerability | Leaky Function(s) |
|-----|---------|--------------|-------------------|
| CVE-2018-0734 | OpenSSL | DSA timing leak in `BN_mod_inverse` | `bn_mod_inverse`, `BN_div` |
| CVE-2018-0735 | OpenSSL | ECDSA timing leak in scalar multiplication | `ec_GFp_simple_mul`, `BN_mod_exp` |
| CVE-2022-4304 | OpenSSL | RSA timing oracle in PKCS#1 v1.5 padding check | `RSA_padding_check_PKCS1_type_2` |

| **Protocol** | Obtain pre-patch and post-patch commits from the OpenSSL git history.  Build both versions with GCC-13 at -O2.  Run the analysis on the identified functions.  Report $\Delta_f$ for each. |

### RQ6 — Scalability

> Can the tool analyze a full cryptographic library within practical time budgets?

| Aspect | Detail |
|--------|--------|
| **Metric** | Wall-clock time per function $t_f$, reported as a histogram and as percentile statistics.  Also: total library analysis time $T_{\text{lib}}$. |
| **Falsification** | Fewer than 90% of functions completing in $< 5$ minutes, or total library analysis exceeding 90 minutes on a laptop (defined: 8-core, 16 GB RAM, no GPU).  Either outcome falsifies the scalability claim. |
| **Theorem connection** | Scalability follows from the polynomial complexity of the abstract domain: $O(|\text{sets}| \times |\text{ways}|^2 \times W)$ per basic block, where $W \leq 50$ is the speculative window bound.  The empirical test validates that theoretical polynomial complexity translates to practical wall-clock performance. |
| **Protocol** | Analyze all exported functions in BoringSSL `crypto/` (≈600 functions), OpenSSL `libcrypto` (≈2000 functions), and libsodium (≈200 functions).  Record $t_f$ per function.  Report histogram, percentiles (50th, 90th, 95th, 99th, max), and $T_{\text{lib}}$.  Exclude non-crypto utility functions (string ops, memory alloc) by filtering to functions that touch key-material memory regions. |

### RQ7 — Speculative Leakage Discovery

> Does speculative analysis detect leakage that non-speculative analysis misses?

| Aspect | Detail |
|--------|--------|
| **Metric** | Speculative uplift $\sigma_f = B_f^{\text{spec}} - B_f^{\text{non-spec}}$ per function.  Report (a) the number of functions with $\sigma_f > 0$ and $B_f^{\text{non-spec}} = 0$ (speculation-only leaks), and (b) the distribution of $\sigma_f$ across all functions. |
| **Falsification** | $\sigma_f = 0$ for all functions across the Spectector suite and Kocher PoCs (i.e., the speculative domain adds nothing).  Also falsified if the speculative domain produces $\sigma_f > 0$ on programs with verified constant-time non-speculative execution where speculation is blocked by `lfence` barriers (false uplift). |
| **Theorem connection** | Theorem D.3.1 (speculative channel capacity bound): $B_f^{\text{spec}} \geq B_f^{\text{non-spec}}$ always, with strict inequality when speculative paths reach secret-dependent cache lines that architectural paths do not.  The empirical test confirms this strict inequality manifests in practice. |
| **Protocol** | Run each benchmark in two modes: (a) $D_{\text{cache}} \otimes D_{\text{quant}}$ only (non-speculative), and (b) full $D_{\text{spec}} \otimes D_{\text{cache}} \otimes D_{\text{quant}}$ (speculative).  Focus on (i) Spectector's 14 Spectre-PHT gadgets, (ii) Kocher's 15 PoC variants, (iii) crypto functions known to have speculative leaks.  Negative control: `lfence`-hardened versions of the same functions should have $\sigma_f = 0$. |

---

## 2  Benchmark Suite

### 2.1  Crypto Primitives

| Primitive | Library | Source | Key Size | Notes |
|-----------|---------|--------|----------|-------|
| AES-128 T-table | OpenSSL | `crypto/aes/aes_core.c` | 128 bits | Known cache-timing leak; canonical benchmark |
| AES-128-NI | OpenSSL, BoringSSL | `aesni-x86_64.pl` | 128 bits | Hardware AES; expected 0 leakage |
| ChaCha20 | libsodium, BoringSSL | `crypto_stream/chacha20` | 256 bits | Constant-time by design |
| Curve25519 | libsodium | `crypto_scalarmult/curve25519` | 256 bits | Montgomery ladder; conditional swap |
| RSA-2048 CRT | OpenSSL | `crypto/rsa/rsa_ossl.c` | 2048 bits | Modular exponentiation; blinding |
| Ed25519 sign | libsodium | `crypto_sign/ed25519` | 256 bits | Variable-time scalar mult in some builds |
| Poly1305 | libsodium, BoringSSL | `crypto_onetimeauth/poly1305` | 256 bits | Constant-time accumulator |
| SHA-256 | OpenSSL | `crypto/sha/sha256.c` | — | No key-dependent branches (negative control) |

**Compilation matrix:**

| Compiler | Versions | Optimization Levels |
|----------|----------|-------------------|
| GCC | 13.x | `-O0`, `-O2`, `-O3` |
| Clang/LLVM | 17.x | `-O0`, `-O2`, `-O3` |

This yields $8 \text{ primitives} \times 2 \text{ compilers} \times 3 \text{ opt levels} = 48$ binary variants
per library, though not every primitive is available in every library.  We estimate
**~100 distinct binary artifacts** after deduplication.

### 2.2  CVE Regression Binaries

| CVE | Vulnerable Commit | Patched Commit | Library | Build |
|-----|-------------------|----------------|---------|-------|
| CVE-2018-0734 | OpenSSL `< 1.1.0j` | OpenSSL `1.1.0j` | OpenSSL | GCC-13 `-O2` |
| CVE-2018-0735 | OpenSSL `< 1.1.1a` | OpenSSL `1.1.1a` | OpenSSL | GCC-13 `-O2` |
| CVE-2022-4304 | OpenSSL `< 3.0.8` | OpenSSL `3.0.8` | OpenSSL | GCC-13 `-O2` |

Each CVE produces two binaries (vulnerable, patched) = **6 additional binaries**.

### 2.3  Spectre Gadgets

| Suite | Source | # Gadgets | Spectre Variant |
|-------|--------|-----------|-----------------|
| Spectector | Guarnieri et al. 2020 | 14 | PHT (v1) |
| Kocher PoCs | Paul Kocher 2018 | 15 | PHT (v1), variants |
| MSVC STL gadgets | Cauligi et al. 2020 | 5 | PHT (v1) |

Total: **34 speculative gadgets**.  Each compiled at `-O0` and `-O2` with both compilers
= **up to 136 gadget binaries** (many are single-function, so compilation is fast).

### 2.4  Composition Benchmarks

Programs decomposed into constituent functions for compositional analysis:

| Program | Decomposition | # Components |
|---------|--------------|-------------|
| AES-128 encryption | 10 rounds (AddRoundKey, SubBytes, ShiftRows, MixColumns) × 10, plus key expansion | 11 sub-functions |
| ChaCha20 block | 20 quarter-round pairs, plus initialization + finalization | 22 sub-functions |
| Curve25519 scalar mult | 255 ladder steps (each: cswap, fe_mul, fe_sq, fe_add, fe_sub) | 5 unique sub-functions, 255 compositions |
| RSA-2048 CRT (modexp only) | Square-and-multiply chain decomposed per window | ≥32 sub-functions |

For each, we compute both $B^{\text{comp}}$ (compositional) and $B^{\text{mono}}$ (monolithic).

### 2.5  Independence Stress Tests

AES T-table with **related subkeys**: use the AES key schedule to derive round keys from
a single master key.  The key schedule introduces algebraic relations between round keys
that may violate the independence condition required by Theorem E.2.1.  This is a stress
test for the composition rule's robustness — if $\kappa$ degrades significantly under
related subkeys but not under independent random keys, we have identified a real-world
limitation of the independence assumption.

### 2.6  Compiler-Broken Constant-Time

| Library | Function | Behavior |
|---------|----------|----------|
| libsodium `crypto_secretbox_xsalsa20poly1305` | Under Clang-17 `-O3` | Compiler may introduce branch on secret |
| libsodium `crypto_verify_32` | Under GCC-13 `-O3` | Short-circuit optimization breaks CT |
| OpenSSL `CRYPTO_memcmp` | Under `-O2` with LTO | Link-time optimization can inline and branch |

These serve as positive controls: the tool **must** report non-zero leakage on
compiler-broken builds and zero (or near-zero) leakage on `-O0` builds of the same source.

### 2.7  Ground Truth Computation

| Key size regime | Method | Output |
|----------------|--------|--------|
| $k \leq 16$ bits | **Exhaustive enumeration**: simulate all $2^k$ keys through a concrete speculative + LRU cache model.  Construct the full channel matrix $P(O \mid K)$.  Compute exact min-entropy leakage $\mathcal{L}_\infty = \log_2 \sum_o \max_k P(o \mid k)$. | Exact $C_f(s)$ |
| $k > 16$ bits | **Monte Carlo**: sample $10^6$ key–trace pairs.  Estimate mutual information $I(K; O)$ via KSG estimator (Kraskov et al. 2004).  Report point estimate $\hat{C}_f$ and 95% bootstrap confidence interval.  Also compute a histogram-based upper bound $\hat{C}_f^+$ using the plug-in min-entropy estimator with Miller–Madow bias correction. | $\hat{C}_f \pm \delta$ |

For large-key ground truth we do **not** claim exact values; we claim only that
$B_f(s) \geq \hat{C}_f^+$ with high confidence (soundness check) and
$B_f(s) / \hat{C}_f \leq$ threshold (precision check).

---

## 3  Baselines

### 3.1  CacheAudit (Doychev et al., CCS 2013; TISSEC 2015)

| Property | Detail |
|----------|--------|
| **Capability** | Quantitative cache side-channel bounds via abstract interpretation on x86-32 binaries (non-speculative, monolithic). |
| **Comparison axes** | Precision ($r_f$ head-to-head on shared benchmarks), scalability ($t_f$), and qualitative coverage (x86-32 vs x86-64, speculation, composition). |
| **Limitation for comparison** | CacheAudit supports only x86-32 and a fixed LRU model.  We cross-compile shared benchmarks (AES T-table, sorting networks) to both x86-32 (for CacheAudit) and x86-64 (for our tool) at the same optimization level and compare $r_f$ on the overlapping function set.  Differences in ISA may introduce confounds; we note this as a threat to validity. |
| **Expected outcome** | Comparable or tighter $r_f$ on non-speculative benchmarks (due to $\rho$); strictly better on speculative benchmarks (CacheAudit has no speculative capability). |

### 3.2  Spectector (Guarnieri et al., S&P 2020)

| Property | Detail |
|----------|--------|
| **Capability** | Speculative non-interference checking via symbolic execution.  Boolean verdict: "secure" or "insecure" (with witness trace). |
| **Comparison axes** | Detection sensitivity on Spectre gadgets (true positive rate), scalability ($t_f$), and information granularity (boolean vs. quantitative). |
| **Limitation for comparison** | Spectector gives no quantitative bound.  We compare only detection: does our $B_f^{\text{spec}} > 0$ agree with Spectector's "insecure" verdict, and $B_f^{\text{spec}} = 0$ with "secure"?  Disagreements are manually triaged. |
| **Expected outcome** | Agreement on Spectector's own benchmark suite.  Our tool additionally provides a quantitative bound (e.g., "leaks 3.2 bits under Spectre-PHT") where Spectector says only "insecure". |

### 3.3  Binsec/Rel (Daniel et al., S&P 2020)

| Property | Detail |
|----------|--------|
| **Capability** | Binary-level relational symbolic execution for constant-time checking.  Boolean verdict. |
| **Comparison axes** | Detection agreement, scalability, and information granularity. |
| **Limitation for comparison** | Boolean output only, no speculation modeling.  We check agreement: our $B_f^{\text{non-spec}} = 0 \Leftrightarrow$ Binsec/Rel says "constant-time".  Disagreements classified as (a) false negative in Binsec/Rel (our tool finds non-zero leakage Binsec/Rel misses — unlikely since both are sound), or (b) precision difference (Binsec/Rel says "not CT" due to benign but non-constant memory access patterns where our bound is very small). |
| **Expected outcome** | Full agreement on CT/non-CT classification; our tool additionally quantifies how much leakage exists in non-CT functions. |

### 3.4  cachegrind Differential (Informal Baseline)

| Property | Detail |
|----------|--------|
| **Capability** | Valgrind's cache simulator.  Computes cache hit/miss counts for a concrete execution.  Differential: compare counts across two executions with different keys. |
| **Comparison axes** | Regression detection sensitivity, false positive rate, formal guarantees. |
| **What we catch that cachegrind misses** | (a) **Speculative leaks** — cachegrind simulates architectural execution only, missing Spectre-induced cache effects. (b) **Formal soundness** — cachegrind's differential is heuristic; a single pair of executions may miss leaks that manifest only for specific key pairs.  Our analysis covers all keys by construction. (c) **Quantitative delta** — cachegrind reports cache-miss count differences, not information-theoretic leakage in bits.  A large miss-count delta may correspond to tiny leakage (many misses, same pattern) or vice versa. (d) **Input coverage** — cachegrind requires choosing specific test inputs; our analysis is input-independent. |
| **Protocol** | For each CVE regression binary, run cachegrind with 1000 random key pairs, compute the coefficient of variation of cache-miss counts.  Compare detection sensitivity: does high CV correlate with our $\Delta_f > 0$?  Report cases where cachegrind fails to flag a leak that our tool detects (expected: speculative leaks, subtle timing differences washed out by noise). |
| **Expected outcome** | cachegrind detects gross leaks (AES T-table) but misses speculative leaks and subtle leaks with low miss-count variance.  Our tool detects all of them with formal guarantees. |

### 3.5  Naïve Composition Baseline

| Property | Detail |
|----------|--------|
| **Capability** | Sum per-function bounds $\sum_i B_{f_i}$ without propagating cache-state transformers $\tau_{f_i}$. |
| **Comparison** | Precision: $B^{\text{naive}} / B^{\text{comp}}$ measures the value of cache-state-aware composition.  The naïve baseline represents the trivially sound but imprecise composition strategy. |
| **Expected outcome** | $B^{\text{naive}} / B^{\text{comp}} \geq 2\times$ on most composition benchmarks, demonstrating that cache-state propagation is essential for practical precision. |

---

## 4  Ablation Studies

Each ablation disables one component of the framework and measures the impact on
precision and soundness.  All ablations preserve soundness by construction (disabling a
refinement can only increase the bound); we verify this empirically.

### 4.1  $\rho$ Ablation (Reduced Product → Direct Product)

| Configuration | Domain | Reduction |
|--------------|--------|-----------|
| **Full** (default) | $D_{\text{spec}} \otimes D_{\text{cache}} \otimes D_{\text{quant}}$ | $\rho = \rho_{\text{cache} \leftarrow \text{spec}} \circ \rho_{\text{quant} \leftarrow \text{cache}}$ |
| **Ablated** | $D_{\text{spec}} \times D_{\text{cache}} \times D_{\text{quant}}$ | $\rho = \text{id}$ (no inter-domain propagation) |

**Measured:** $\rho\text{-ratio}_f$ distribution (same as RQ3).
**Hypothesis:** $\rho$ improves bounds by $\geq 1.5\times$ on programs with dead speculative
paths or untainted cache lines (AES-NI, ChaCha20, `lfence`-hardened code).

### 4.2  Speculation Ablation ($D_{\text{spec}}$ Removal)

| Configuration | Domain |
|--------------|--------|
| **Full** (default) | $D_{\text{spec}} \otimes D_{\text{cache}} \otimes D_{\text{quant}}$ |
| **Ablated** | $D_{\text{cache}} \otimes D_{\text{quant}}$ (architectural paths only) |

**Measured:** Speculative uplift $\sigma_f$ (same as RQ7), and analysis time reduction.
**Hypothesis:** Spectre gadgets and unprotected crypto functions show $\sigma_f > 0$.
Constant-time code with `lfence` barriers shows $\sigma_f = 0$.  Speculation analysis
adds $\leq 2\times$ runtime overhead (bounded by speculative window size $W$).

### 4.3  Taint Ablation (Taint Filtering Removal)

| Configuration | Domain |
|--------------|--------|
| **Full** (default) | $D_{\text{quant}}$ counts only tainted cache lines |
| **Ablated** | $D_{\text{quant}}$ counts all cache lines (no taint filtering) |

**Measured:** Taint tightening ratio $\text{taint-ratio}_f = B_f^{\text{no-taint}} / B_f^{\text{taint}}$.
**Hypothesis:** Programs with many public-data cache accesses (SHA-256 on public input,
AES with known plaintext) show large taint-ratio ($\geq 5\times$), confirming that taint
filtering is essential for precision.  Programs where all cache accesses are
secret-dependent (T-table AES with secret plaintext) show taint-ratio $\approx 1$.

### 4.4  Composition Ablation (Compositional → Monolithic)

| Configuration | Analysis Mode |
|--------------|--------------|
| **Compositional** (default) | Per-function contracts composed via $B_{f;g}(s) = B_f(s) + B_g(\tau_f(s))$ |
| **Monolithic** | Whole-program analysis without function boundaries |

**Measured:** Composition overhead $\kappa$ (same as RQ4), and scalability comparison
(compositional should be faster on large programs due to sub-problem decomposition and
potential caching of function contracts).
**Hypothesis:** $\kappa \leq 2$ and compositional analysis is $\geq 3\times$ faster on
programs with $\geq 10$ composed functions (AES, ChaCha20, Curve25519).

---

## 5  Metrics Table

| # | Metric | Symbol | Target | How Measured | RQ |
|---|--------|--------|--------|-------------|-----|
| M1 | Soundness (false negatives) | $\text{FN}$ | $= 0$ | Exhaustive enumeration on $k \leq 16$ bits: count instances where $B_f(s) < C_f(s)$ | RQ1 |
| M2 | Precision, non-speculative | $r_f^{\text{ns}}$ | median $\leq 3\times$ | $r_f = B_f(s) / C_f(s)$, non-speculative mode, exhaustive or Monte Carlo | RQ2 |
| M3 | Precision, speculative | $r_f^{\text{sp}}$ | median $\leq 10\times$ | Same as M2 but in speculative mode | RQ2 |
| M4 | CVE regression sensitivity | $\text{CVE}_{\text{det}}$ | $3/3$ detected | $\Delta_f > 0$ on vulnerable binary, $\Delta_f \leq \epsilon$ on patched binary, per CVE | RQ5 |
| M5 | CVE false alarm rate | $\text{CVE}_{\text{FA}}$ | $0/3$ false alarms | $\Delta_f > \epsilon$ on any patched binary = false alarm | RQ5 |
| M6 | Composition precision | $\kappa$ | $\leq 2\times$ | $B^{\text{comp}} / B^{\text{mono}}$ on $\geq 3$ composition benchmarks | RQ4 |
| M7 | Per-function time (90th %) | $t_{90}$ | $< 5$ min | Wall-clock per-function analysis time, 90th percentile | RQ6 |
| M8 | Total library time | $T_{\text{lib}}$ | $< 90$ min | Sum of per-function times for full library | RQ6 |
| M9 | $\rho$ improvement | $\rho\text{-ratio}$ | $> 1.0$ on majority | $B_f^{\text{direct}} / B_f^{\text{reduced}}$; fraction with ratio $> 1.0$ | RQ3 |
| M10 | Speculative uplift | $\sigma_f$ | $> 0$ on Spectre gadgets | $B_f^{\text{spec}} - B_f^{\text{non-spec}}$; count functions with $\sigma_f > 0$ | RQ7 |
| M11 | Taint tightening | $\text{taint-ratio}$ | $> 1$ when public data present | $B_f^{\text{no-taint}} / B_f^{\text{taint}}$ | Ablation 4.3 |
| M12 | Composition speedup | $T^{\text{comp}} / T^{\text{mono}}$ | $\leq 1.0$ on large programs | Wall-clock comparison | Ablation 4.4 |
| M13 | Naïve vs aware composition | $B^{\text{naive}} / B^{\text{comp}}$ | $\geq 2\times$ | Naïve sum vs cache-state-aware composition | Baseline 3.5 |

---

## 6  Threats to Validity

### 6.1  Internal Validity

| Threat | Mitigation |
|--------|-----------|
| **Ground-truth simulator bugs.** Our exhaustive enumeration and Monte Carlo ground truth rely on a concrete cache simulator.  A bug in the simulator could produce incorrect $C_f(s)$, leading to false soundness violations or inflated precision. | (a) Cross-validate the simulator against cachegrind on non-speculative traces.  (b) Implement the simulator independently in two languages (Python for reference, Rust for speed) and compare outputs on 100 random traces per benchmark.  (c) Unit-test the LRU eviction logic exhaustively for cache configurations $\leq 4$ ways. |
| **Speculative semantics modeling.** Our speculative simulator uses a simplified model (bounded PHT misspeculation, no BTB/STL/RSB).  The ground truth may not reflect real hardware speculative behavior. | (a) Validate speculative path enumeration against Intel Pin traces on real hardware (where available) for the Kocher PoC suite.  (b) Clearly document which speculative variants are modeled and which are out of scope.  (c) Report results with and without speculation separately. |
| **Compiler version sensitivity.** Different compiler patch versions (e.g., GCC 13.1 vs 13.2) may produce different binaries, affecting reproducibility. | Pin exact compiler versions in a Docker/Nix build environment.  Publish the Dockerfile and all build scripts. |
| **Non-deterministic analysis.** If the abstract fixpoint computation uses any randomization (e.g., widening thresholds), results may vary across runs. | The analysis is fully deterministic.  Verify by running each benchmark twice and checking bitwise-identical output.  Report single-run results. |

### 6.2  External Validity

| Threat | Mitigation |
|--------|-----------|
| **Benchmark representativeness.** We evaluate only cryptographic code.  Results may not generalize to non-crypto programs with different memory-access patterns. | (a) Acknowledge this limitation explicitly — the tool targets crypto binaries.  (b) Include two non-crypto benchmarks (sorting network from CacheAudit, binary search) as out-of-domain sanity checks to characterize degradation.  (c) Future work: extend to media codecs and ML inference kernels. |
| **Library version scope.** We test specific versions of OpenSSL, BoringSSL, and libsodium.  Other libraries or versions may behave differently. | Pin library versions.  Include at least two versions per library (one older, one recent) to check stability. |
| **ISA coverage.** We support only x86-64.  ARM, RISC-V, and other architectures with different cache geometries and speculative behaviors are out of scope. | Acknowledge explicitly.  The abstract domain is parameterized by cache geometry; porting to ARM requires only a new binary lifter adapter, not a new domain. |

### 6.3  Construct Validity

| Threat | Mitigation |
|--------|-----------|
| **LRU vs PLRU gap.** Real Intel/AMD caches use pseudo-LRU (tree-PLRU or adaptive), not exact LRU.  Our abstract domain models exact LRU.  Bounds derived under LRU may be unsound under PLRU. | (a) Characterize the gap empirically: run the concrete PLRU simulator alongside the LRU simulator on all benchmarks and report the fraction of cases where LRU and PLRU disagree on the number of observable cache sets.  (b) Report the maximum observed divergence in bits of leakage between LRU and PLRU ground truth.  (c) Provide a conservative correction factor: if LRU analysis says $B_f$ bits, multiply by the empirically-observed PLRU/LRU ratio $\alpha$ to obtain a PLRU-adjusted bound. (d) Defer exact PLRU abstract domain to v2; characterize the gap as an open problem. |
| **Min-entropy vs Shannon entropy.** We bound min-entropy leakage, which is the operationally meaningful metric (one-try guessing probability).  Some prior work uses Shannon entropy, making direct numerical comparison misleading. | Report all comparisons in consistent units (min-entropy bits).  When comparing to CacheAudit (which reports both), use the min-entropy variant.  Include a supplementary table converting our bounds to Shannon entropy for readers familiar with that convention. |
| **Channel model simplification.** We model a noise-free, deterministic cache side channel.  Real hardware introduces noise (TLB, prefetcher, coherence traffic) that reduces effective leakage. | Our bounds are upper bounds on a noise-free model, hence they are also upper bounds on the noisy channel.  This is conservative (sound) but may inflate precision ratios.  We report this as a known source of overestimation. |

### 6.4  Statistical Validity

| Threat | Mitigation |
|--------|-----------|
| **Monte Carlo variance for large keys.** The KSG estimator has known bias for high-dimensional distributions.  With $10^6$ samples, the estimator may under- or overestimate $C_f$ for functions with many observable classes. | (a) Report 95% bootstrap confidence intervals on all Monte Carlo estimates.  (b) Use the plug-in min-entropy estimator with Miller–Madow bias correction as an independent check.  (c) For the soundness check (RQ1), use only the exhaustive small-key results where ground truth is exact.  Monte Carlo is used only for the precision check (RQ2) on large keys, where approximate ground truth is acceptable. |
| **Multiple comparisons.** With $\sim$100 benchmark variants × 7 metrics, the probability of at least one spurious result is high under naïve $p < 0.05$ testing. | We do not perform hypothesis testing with $p$-values.  All metrics are deterministic ratios or exact counts.  For the Monte Carlo estimates, we report confidence intervals rather than accept/reject decisions.  Where we state "target met," it is based on the point estimate lying within the target range, with the confidence interval reported alongside. |

---

## 7  Presentation Strategy

### 7.1  Money Plot: Per-Function Tightness Ratio Scatter

The primary figure is a **scatter plot** with one point per (function, compilation variant) pair:

- **x-axis:** Exact or estimated channel capacity $C_f(s)$ (bits), log scale.
- **y-axis:** Tightness ratio $r_f = B_f(s) / C_f(s)$, log scale.
- **Color:** Non-speculative (blue) vs. speculative (orange).
- **Shape:** Library (circle = OpenSSL, triangle = BoringSSL, square = libsodium).
- **Horizontal lines:** $r_f = 1$ (perfect tightness), $r_f = 3$ (non-spec target),
  $r_f = 10$ (spec target).
- **Annotation:** Label outliers (functions with $r_f >$ target).

This single figure communicates soundness ($r_f \geq 1$ always), precision (most points
below the target line), and the speculative-vs-non-speculative gap.

### 7.2  Key Tables

**Table 1: Comparison vs. Baselines**

| Benchmark | Our Tool ($B_f$) | CacheAudit | Spectector | Binsec/Rel | cachegrind (CV) | True ($C_f$) |
|-----------|-------------------|------------|------------|------------|-----------------|-------------|
| AES T-table | X.X bits | Y.Y bits | insecure | not CT | high | Z.Z bits |
| AES-NI | 0.0 bits | — | secure | CT | low | 0.0 bits |
| ... | ... | ... | ... | ... | ... | ... |

Columns marked "—" where the baseline does not support the benchmark (e.g., CacheAudit
on x86-64 only benchmarks).

**Table 2: CVE Regression Detection**

| CVE | $B_f^{\text{vuln}}$ (bits) | $B_f^{\text{patched}}$ (bits) | $\Delta_f$ (bits) | Detected? | cachegrind CV $\Delta$ |
|-----|---------------------------|-------------------------------|-------------------|-----------|----------------------|
| CVE-2018-0734 | ... | ... | ... | ✓/✗ | ... |
| CVE-2018-0735 | ... | ... | ... | ✓/✗ | ... |
| CVE-2022-4304 | ... | ... | ... | ✓/✗ | ... |

**Table 3: Ablation Results**

| Ablation | Metric | Full | Ablated | Ratio |
|----------|--------|------|---------|-------|
| $\rho$ removal | Median $r_f$ | ... | ... | ... |
| Speculation removal | # spec-only leaks detected | ... | ... | ... |
| Taint removal | Median $r_f$ | ... | ... | ... |
| Composition removal | Median $\kappa$ | — | — | ... |

**Table 4: Scalability**

| Library | # Functions | $t_{50}$ | $t_{90}$ | $t_{99}$ | $t_{\max}$ | $T_{\text{lib}}$ |
|---------|------------|----------|----------|----------|-----------|-----------------|
| BoringSSL crypto/ | ~600 | ... | ... | ... | ... | ... |
| OpenSSL libcrypto | ~2000 | ... | ... | ... | ... | ... |
| libsodium | ~200 | ... | ... | ... | ... | ... |

### 7.3  Additional Figures

| Figure | Content | Purpose |
|--------|---------|---------|
| **Fig 2** | Wall-clock CDF: fraction of functions completing within time $t$, one curve per library. | RQ6 scalability.  Highlight the 90% / 5-min threshold. |
| **Fig 3** | $\rho$-ratio histogram across all benchmarks.  Vertical line at 1.0. | RQ3 $\rho$ value.  Area to the right of 1.0 is the fraction where $\rho$ helps. |
| **Fig 4** | Speculative uplift $\sigma_f$ bar chart for Spectector + Kocher gadgets, grouped by gadget type. | RQ7.  Shows which Spectre variants produce the most leakage. |
| **Fig 5** | Composition: $B^{\text{comp}}$ vs $B^{\text{mono}}$ paired bar chart for each composition benchmark.  Overlay $\kappa$ values. | RQ4.  Direct visual comparison of compositional vs monolithic bounds. |
| **Fig 6** | Compiler-broken CT: leakage at `-O0` vs `-O2` vs `-O3` for each known-vulnerable function. | Section 2.6.  Demonstrates detection of compiler-introduced leakage. |

### 7.4  Reproducibility Package

All results are generated by a single automated pipeline:

```
make benchmarks    # Downloads/builds all libraries and compiles all variants
make groundtruth   # Exhaustive enumeration + Monte Carlo ground truth
make analyze       # Runs our tool on all benchmarks in all configurations
make baselines     # Runs CacheAudit, Spectector, Binsec/Rel, cachegrind
make tables        # Generates all tables and figures as LaTeX + PDF
make eval          # All of the above, end-to-end
```

**Artifact contents:**
- Docker/Nix environment with pinned compilers, libraries, and tool versions
- All source for benchmarks, ground-truth computation, analysis, and plotting
- Raw CSV output from every experiment
- Generated LaTeX tables and PDF figures
- Total estimated runtime on a laptop (8-core, 16 GB): $\leq 8$ hours for `make eval`

---

## Appendix A: Experimental Protocol Detail

### A.1  Exhaustive Enumeration Protocol

For a function $f$ with key size $k \leq 16$ bits and cache configuration
$(S \text{ sets}, W \text{ ways}, L \text{ line size})$:

1. Initialize a concrete LRU cache state $s_0$ (all-invalid or all-occupied with known
   public data, depending on the calling context).
2. For each key $\kappa \in \{0, 1\}^k$:
   a. Reset cache to $s_0$.
   b. Execute $f(\kappa)$ on the concrete speculative semantics (architectural + transient
      paths up to window $W = 50$ μops).
   c. Record the final cache state (which lines are present in which set/way).
   d. Compute the observable $o_\kappa$ = the sequence of cache hits/misses (or,
      equivalently, the final cache state projected to observability).
3. Construct the channel matrix $P(O = o \mid K = \kappa) = \mathbb{1}[o_\kappa = o]$
   (deterministic channel).
4. Compute min-entropy leakage:
   $\mathcal{L}_\infty = \log_2 \sum_{o \in \mathcal{O}} \max_{\kappa} P(o \mid \kappa)$
   For a deterministic channel, this simplifies to $\mathcal{L}_\infty = \log_2 |\mathcal{O}|$
   where $\mathcal{O}$ is the set of distinct observables.
5. Record $C_f(s_0) = \mathcal{L}_\infty$.

### A.2  Monte Carlo Protocol

For a function $f$ with key size $k > 16$ bits:

1. Sample $N = 10^6$ keys $\kappa_1, \ldots, \kappa_N$ uniformly at random from $\{0,1\}^k$.
2. For each $\kappa_i$, execute the concrete simulator and record observable $o_i$.
3. Estimate mutual information $\hat{I}(K; O)$ using:
   a. **KSG estimator** (for continuous approximation of discrete MI).
   b. **Plug-in estimator** with Miller–Madow bias correction as a cross-check.
4. Compute 95% bootstrap confidence interval using $B = 1000$ bootstrap resamples.
5. Report $\hat{C}_f = \hat{I}$ as the point estimate, $[\hat{C}_f^-, \hat{C}_f^+]$ as
   the confidence interval.

### A.3  Deterministic Reproducibility Check

For every benchmark, run the analysis twice and verify:
- Identical output (bitwise comparison of result JSON).
- Identical wall-clock times within 10% tolerance (to catch non-deterministic scheduling
  effects without requiring exact timing reproducibility).

---

## Appendix B: Benchmark Acquisition and Build

### B.1  Library Sources

| Library | Version | Source |
|---------|---------|--------|
| OpenSSL | 3.1.x (latest stable at eval time) | `https://github.com/openssl/openssl` |
| BoringSSL | HEAD at eval time (pinned commit) | `https://boringssl.googlesource.com/boringssl` |
| libsodium | 1.0.19 (latest stable) | `https://github.com/jedisct1/libsodium` |

### B.2  CVE Commit Hashes

| CVE | Vulnerable | Patched | Notes |
|-----|-----------|---------|-------|
| CVE-2018-0734 | `openssl/openssl@` tag `OpenSSL_1_1_0i` | tag `OpenSSL_1_1_0j` | DSA timing |
| CVE-2018-0735 | `openssl/openssl@` tag `OpenSSL_1_1_1` | tag `OpenSSL_1_1_1a` | ECDSA timing |
| CVE-2022-4304 | `openssl/openssl@` tag `openssl-3.0.7` | tag `openssl-3.0.8` | RSA padding oracle |

### B.3  Build Matrix Script (Pseudocode)

```bash
for lib in openssl boringssl libsodium; do
  for compiler in gcc-13 clang-17; do
    for opt in O0 O2 O3; do
      build_dir="build/${lib}/${compiler}/${opt}"
      CC=${compiler} CFLAGS="-${opt} -g -fno-omit-frame-pointer" \
        ./configure --prefix="${build_dir}" && make -j$(nproc) && make install
    done
  done
done
```

The `-g` flag ensures debug symbols are present for function boundary identification.
The `-fno-omit-frame-pointer` flag ensures reliable stack unwinding for call-graph
reconstruction.
