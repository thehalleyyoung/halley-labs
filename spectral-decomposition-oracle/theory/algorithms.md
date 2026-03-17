# Algorithm Design: Spectral Decomposition Oracle

**Role:** Algorithm Designer — Theory Stage  
**Status:** Implements/validates L3, L3-C, T2, F1, F2 from ideation  
**Target:** INFORMS JoC computational study (Amendment E binding)

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $A \in \mathbb{R}^{m \times n}$ | Constraint matrix (presolved, sparse) |
| $\text{nnz}$ | Number of nonzeros in $A$ |
| $d_e$ | Degree of hyperedge $e$ (number of variables in constraint $e$) |
| $d_{\max}$ | $\max_e d_e$ |
| $L_H \in \mathbb{R}^{n \times n}$ | Hypergraph Laplacian (variable-node graph) |
| $\lambda_1 \leq \lambda_2 \leq \cdots$ | Eigenvalues of $L_H$ |
| $v_1, v_2, \ldots$ | Corresponding eigenvectors |
| $\gamma_2 = \lambda_2$ | Spectral gap (algebraic connectivity) |
| $k$ | Target number of blocks |
| $P = \{B_1, \ldots, B_k\}$ | Partition of variable set into $k$ blocks |
| $E_{\text{cross}}(P)$ | Hyperedges crossing block boundaries |
| $y^*$ | Optimal dual vector of monolithic LP relaxation |

---

## Algorithm 1: Hypergraph Laplacian Construction

### Purpose
Constructs a sparse symmetric positive semidefinite Laplacian $L_H$ from the
constraint matrix $A$, encoding the coupling structure of the MIP as a
weighted graph on variables.

**Validates:** Proposition F1 (permutation invariance), Proposition F2
(scaling sensitivity bound).

### 1.1 Preprocessing: Equilibration

```
ALGORITHM Equilibrate(A, method)
────────────────────────────────────────────────────────────
Input:  A ∈ ℝ^{m×n} sparse constraint matrix
        method ∈ {ruiz, geometric, scip_native}
Output: Ã = D_r · A · D_c  (equilibrated matrix)
        D_r ∈ ℝ^{m×m}, D_c ∈ ℝ^{n×n} diagonal scaling matrices

1.  if method = ruiz then
2.      D_r ← I_m,  D_c ← I_n
3.      for t = 1, 2, …, T_max do          ▷ T_max = 20
4.          for i = 1, …, m do
5.              r_i ← 1 / √(max_j |A_{ij}|)   ▷ row ℓ_∞ scaling
6.              if r_i = ∞ then r_i ← 1        ▷ zero-row guard
7.          for j = 1, …, n do
8.              c_j ← 1 / √(max_i |A_{ij}|)   ▷ column ℓ_∞ scaling
9.              if c_j = ∞ then c_j ← 1        ▷ zero-column guard
10.         D_r ← diag(r) · D_r
11.         D_c ← D_c · diag(c)
12.         A ← diag(r) · A · diag(c)
13.         if max(|max_j |A_{ij}|| - 1) < ε then break   ▷ ε = 1e-6
14.     Ã ← A
15. else if method = geometric then
16.     for i = 1, …, m do
17.         r_i ← 1 / (∏_{j: A_{ij}≠0} |A_{ij}|)^{1/nnz_i}
18.     for j = 1, …, n do
19.         c_j ← 1 / (∏_{i: A_{ij}≠0} |r_i · A_{ij}|)^{1/nnz_j}
20.     D_r ← diag(r),  D_c ← diag(c)
21.     Ã ← D_r · A · D_c
22. else if method = scip_native then
23.     Ã, D_r, D_c ← extract from SCIP internal scaling
24. return Ã, D_r, D_c
────────────────────────────────────────────────────────────
```

**Complexity:** $O(T_{\max} \cdot \text{nnz})$ for Ruiz; $O(\text{nnz})$ for geometric/SCIP.  
**Space:** $O(m + n)$ for diagonal matrices.

**Numerical note:** Ruiz converges geometrically; $T_{\max} = 20$ suffices for
$\kappa(D) < 10$. Geometric-mean is a single-pass alternative. For instances
where $\kappa(A) > 10^{10}$ (big-M formulations), equilibration reduces
$\kappa(\tilde{A})$ to $O(10^{2\text{--}4})$ typically.

### 1.2 Hyperedge Weight Computation

Each constraint $i$ defines a hyperedge $e_i$ over the variables
$S_i = \{j : A_{ij} \neq 0\}$ with degree $d_i = |S_i|$.

```
ALGORITHM ComputeHyperedgeWeights(Ã)
────────────────────────────────────────────────────────────
Input:  Ã ∈ ℝ^{m×n} equilibrated sparse matrix
Output: w ∈ ℝ^m  hyperedge weights

1.  for i = 1, …, m do
2.      w_i ← ‖Ã_{i,:}‖₂² / d_i         ▷ mean squared coefficient
3.      if d_i = 0 then w_i ← 0           ▷ empty constraint (post-presolve artifact)
4.  return w
────────────────────────────────────────────────────────────
```

**Rationale:** Normalizing by $d_i$ ensures dense constraints do not dominate
purely by variable count. The $\ell_2^2$ norm captures coefficient magnitude.

**Complexity:** $O(\text{nnz})$.

### 1.3 Clique-Expansion Laplacian ($d_{\max} \leq 200$)

For each hyperedge $e_i$ with support $S_i$, create a weighted clique on
$S_i$ with edge weight $w_i / (d_i - 1)$ per pair.

```
ALGORITHM CliqueExpansionLaplacian(Ã, w)
────────────────────────────────────────────────────────────
Input:  Ã ∈ ℝ^{m×n}, w ∈ ℝ^m
Output: L_H ∈ ℝ^{n×n} sparse symmetric Laplacian

1.  W ← empty n×n sparse accumulator (COO or DOK format)
2.  for i = 1, …, m do
3.      S_i ← {j : Ã_{ij} ≠ 0}
4.      d_i ← |S_i|
5.      if d_i ≤ 1 then continue          ▷ singleton/empty → no edges
6.      α_i ← w_i / (d_i - 1)            ▷ clique normalization
7.      for (j, j') ∈ S_i × S_i, j < j' do
8.          W[j, j'] ← W[j, j'] + α_i    ▷ off-diagonal accumulation
9.  ▷ Symmetrize and form Laplacian
10. W ← W + W^T                           ▷ now W[j,j'] = W[j',j] = sum of α_i
11. D ← diag(W · 1_n)                     ▷ degree matrix
12. L_H ← D - W
13. ▷ Handle zero-degree vertices
14. for j = 1, …, n do
15.     if D[j,j] = 0 then
16.         L_H[j,j] ← ε_reg             ▷ ε_reg = 1e-12, regularization
17. return L_H
────────────────────────────────────────────────────────────
```

**Complexity:**
- **Time:** $O\bigl(\sum_{i=1}^{m} d_i^2\bigr) = O(\text{nnz} \cdot d_{\max})$ worst case.  
  Typical: $O(\text{nnz} \cdot \bar{d})$ where $\bar{d}$ is mean degree.
- **Space:** $O\bigl(\min\bigl(\sum_i d_i^2,\; n^2\bigr)\bigr)$ for the weight matrix;
  $O(n + \text{nnz}(L_H))$ for the CSR Laplacian.

**Numerical considerations:**
- When $d_{\max} = 200$: up to $\binom{200}{2} = 19{,}900$ edges per constraint.
  For $m = 10{,}000$ constraints, worst case $\sim 2 \times 10^8$ entries — manageable
  in CSR with 64-bit floats (~3 GB).
- The $(d_i - 1)$ normalization ensures each hyperedge contributes total weight
  $w_i \cdot d_i / 2$ to the graph, preserving the Cheeger relationship.
- Zero-degree regularization avoids singular Laplacian while preserving the
  near-null space structure (perturbation $< 10^{-12}$).

### 1.4 Incidence-Matrix Laplacian ($d_{\max} > 200$)

Avoids quadratic blowup using the Bolla (1993) / Zhou et al. (2006) formulation.

```
ALGORITHM IncidenceMatrixLaplacian(Ã, w)
────────────────────────────────────────────────────────────
Input:  Ã ∈ ℝ^{m×n}, w ∈ ℝ^m
Output: L_I ∈ ℝ^{n×n} sparse symmetric Laplacian

1.  ▷ Build incidence matrix H ∈ ℝ^{n×m} (variables × constraints)
2.  H ← sparse matrix where H[j, i] = 1 if Ã_{ij} ≠ 0
3.  W_e ← diag(w)                          ▷ m×m diagonal hyperedge weights
4.  D_e ← diag(d_1, …, d_m)               ▷ m×m diagonal hyperedge degrees
5.  ▷ Normalized incidence product
6.  ▷ L_I = H · W_e · D_e^{-1} · H^T − D_v   (unnormalized)
7.  ▷ More efficiently: L_I via the identity L_I = D_v − H W_e D_e^{-1} H^T
8.  ▷   where D_v = diag(∑_i w_i · H[j,i] / d_i)
9.
10. ▷ Compute vertex degree matrix
11. D_v ← zero vector of length n
12. for i = 1, …, m do
13.     if d_i = 0 then continue
14.     α_i ← w_i / d_i
15.     for j ∈ S_i do
16.         D_v[j] ← D_v[j] + α_i
17.
18. ▷ Form Laplacian via implicit product
19. ▷ For eigensolve, store L_I as a LinearOperator that computes:
20. ▷   L_I · x = D_v · x − H · (W_e · D_e^{-1} · (H^T · x))
21. ▷ This avoids forming the n×n matrix explicitly.
22.
23. ▷ For small n (< 50,000): form explicitly
24. if n < 50000 then
25.     Θ ← sparse(diag(w ./ d))           ▷ m×m: Θ_ii = w_i/d_i
26.     B ← H · Θ · H^T                    ▷ n×n sparse product
27.     L_I ← diag(D_v) − B
28.     ▷ Regularize zero-degree vertices
29.     for j = 1, …, n do
30.         if D_v[j] = 0 then L_I[j,j] ← ε_reg
31.     return L_I  (as sparse matrix)
32. else
33.     return LinearOperator(L_I)          ▷ matrix-free for LOBPCG
────────────────────────────────────────────────────────────
```

**Complexity:**
- **Time:** $O(\text{nnz})$ to build $H$ and compute $D_v$.
  Explicit product $H \Theta H^T$: $O(\text{nnz} \cdot \bar{d}_{\text{col}})$ where
  $\bar{d}_{\text{col}}$ is mean column degree (typically $\ll d_{\max}$).
  Matrix-free matvec: $O(\text{nnz})$ per iteration.
- **Space:** $O(\text{nnz})$ for $H$; $O(n + m)$ for diagonals.
  Explicit $L_I$: $O(\text{nnz}(L_I)) \leq O(n \cdot \bar{d}_{\text{col}}^2)$.
  Matrix-free: $O(\text{nnz} + n + m)$.

**Numerical note:** The incidence-matrix Laplacian has the same null space
(constant vector) as the clique expansion but different non-trivial spectrum.
Validation on the $d_{\max} \leq 200$ overlap set (Spearman $\rho > 0.85$
required) ensures features are comparable.

### 1.5 Dispatcher

```
ALGORITHM ConstructLaplacian(A_raw, method)
────────────────────────────────────────────────────────────
Input:  A_raw ∈ ℝ^{m×n} raw constraint matrix (post-presolve)
        method ∈ {ruiz, geometric, scip_native}
Output: L_H (sparse or LinearOperator), metadata dict

1.  Ã, D_r, D_c ← Equilibrate(A_raw, method)
2.  w ← ComputeHyperedgeWeights(Ã)
3.  d_max ← max_{i=1}^{m} |{j : Ã_{ij} ≠ 0}|
4.  if d_max ≤ 200 then
5.      L_H ← CliqueExpansionLaplacian(Ã, w)
6.      variant ← "clique"
7.  else
8.      L_H ← IncidenceMatrixLaplacian(Ã, w)
9.      variant ← "incidence"
10. metadata ← {variant, d_max, nnz(L_H), κ_est(D_r), κ_est(D_c)}
11. return L_H, metadata
────────────────────────────────────────────────────────────
```

**End-to-end complexity for Algorithm 1:**

| Regime | Time | Space |
|--------|------|-------|
| $d_{\max} \leq 200$ | $O(\text{nnz} \cdot d_{\max})$ | $O(\text{nnz} \cdot d_{\max})$ |
| $d_{\max} > 200$, explicit | $O(\text{nnz} \cdot \bar{d}_{\text{col}})$ | $O(\text{nnz}(L_I))$ |
| $d_{\max} > 200$, matrix-free | $O(\text{nnz})$ setup; $O(\text{nnz})$/matvec | $O(\text{nnz} + n + m)$ |

**Implementation language:** Python (scipy.sparse COO→CSR pipeline).
Hot inner loop (line 7 of clique expansion) may need Cython or numba JIT
if profiling shows >50% time there on large instances.

---

## Algorithm 2: Spectral Feature Extraction

### Purpose
Computes the 8 spectral features from $L_H$ via partial eigendecomposition
(bottom $k+1$ eigenpairs) with a robust fallback chain.

**Validates:** Features used in T2, L3-sp; implements definitions from §3.

### 2.1 Eigensolve with Fallback Chain

```
ALGORITHM RobustEigensolve(L_H, k, cache_key)
────────────────────────────────────────────────────────────
Input:  L_H ∈ ℝ^{n×n} (sparse or LinearOperator), PSD Laplacian
        k: number of blocks (compute k+1 eigenpairs)
        cache_key: (instance_id, variant, k) for HDF5 cache
Output: Λ = (λ_1, …, λ_{k+1}), V ∈ ℝ^{n×(k+1)}
        status ∈ {converged, fallback_lobpcg, fallback_shift, failed}

1.  ▷ Check HDF5 cache
2.  if cache_key exists in eigendecomposition cache then
3.      return cached (Λ, V), status = "cached"
4.
5.  n ← dimension of L_H
6.  tol ← 1e-8
7.  maxiter ← min(n, 3000)
8.
9.  ▷ ----- PRIMARY: ARPACK shift-invert -----
10. try
11.     σ ← -1e-6                           ▷ shift near zero
12.     Λ, V ← scipy.sparse.linalg.eigsh(
13.                 L_H, k=k+1, sigma=σ,
14.                 which='LM',              ▷ largest of (L_H - σI)^{-1} = smallest of L_H
15.                 tol=tol, maxiter=maxiter)
16.     ▷ Sort by eigenvalue
17.     idx ← argsort(Λ)
18.     Λ ← Λ[idx], V ← V[:, idx]
19.     ▷ Validate: λ_1 should be ≈ 0
20.     if |λ_1| > 1e-4 then
21.         warn("Smallest eigenvalue not near zero: λ_1 =", λ_1)
22.     ▷ Check residuals
23.     for i = 1, …, k+1 do
24.         r_i ← ‖L_H · V[:,i] − Λ[i] · V[:,i]‖ / max(|Λ[i]|, 1)
25.         if r_i > 1e-6 then
26.             warn("Poor residual for eigenpair", i, ": r =", r_i)
27.     status ← "converged"
28.     store (Λ, V) in cache at cache_key
29.     return Λ, V, status
30. catch ArpackNoConvergence or ArpackError:
31.     pass                                 ▷ fall through
32.
33. ▷ ----- FALLBACK 1: LOBPCG -----
34. try
35.     X_init ← random n×(k+1) matrix, orthogonalized
36.     ▷ Jacobi preconditioner: M = diag(L_H)^{-1}, clamped
37.     diag_L ← diagonal of L_H
38.     M_inv ← 1 / max(diag_L, 1e-10)
39.     Λ, V ← scipy.sparse.linalg.lobpcg(
40.                 L_H, X_init, M=diag(M_inv),
41.                 tol=tol, maxiter=500, largest=False)
42.     idx ← argsort(Λ)
43.     Λ ← Λ[idx], V ← V[:, idx]
44.     status ← "fallback_lobpcg"
45.     store (Λ, V) in cache at cache_key
46.     return Λ, V, status
47. catch Exception:
48.     pass
49.
50. ▷ ----- FALLBACK 2: Explicit shift -----
51. ▷ Add small diagonal shift to improve conditioning
52. try
53.     L_shifted ← L_H + 1e-4 · I_n
54.     Λ_s, V ← scipy.sparse.linalg.eigsh(
55.                   L_shifted, k=k+1, which='SM',
56.                   tol=tol*10, maxiter=maxiter*2)
57.     Λ ← Λ_s − 1e-4                     ▷ undo shift
58.     idx ← argsort(Λ)
59.     Λ ← Λ[idx], V ← V[:, idx]
60.     status ← "fallback_shift"
61.     store (Λ, V) in cache at cache_key
62.     return Λ, V, status
63. catch Exception:
64.     pass
65.
66. ▷ ----- ALL FAILED -----
67. Λ ← [NaN] × (k+1)
68. V ← NaN matrix of shape n×(k+1)
69. status ← "failed"
70. return Λ, V, status
────────────────────────────────────────────────────────────
```

**Complexity:**
- **ARPACK shift-invert:** Each Lanczos iteration requires one sparse
  factorization solve (amortized via LU factored once) + one matvec.
  - LU factorization of $(L_H - \sigma I)$: $O(\text{nnz}(L_H) \cdot \text{fill})$
    where fill depends on sparsity structure; typically $O(n^{1.2\text{--}1.5})$
    for sparse graph Laplacians.
  - Per iteration: $O(n)$ for triangular solve + $O(\text{nnz}(L_H))$ for matvec.
  - Total: $O(T_{\text{Lanczos}} \cdot (n + \text{nnz}(L_H)) + \text{LU cost})$.
  - $T_{\text{Lanczos}}$ typically 50–300 for $k \leq 20$.
- **LOBPCG:** $O(T_{\text{iter}} \cdot k \cdot \text{nnz}(L_H))$ with preconditioner.
  No factorization needed; works with LinearOperator.
- **Space:** $O(n \cdot k)$ for eigenvectors + $O(\text{nnz}(L_H))$ for matrix
  (or $O(\text{nnz}(A))$ for matrix-free).

**Typical wall time:** 2–15 seconds for $n \leq 50{,}000$, $k = 10$;
up to 30 seconds for $n = 200{,}000$ (ARPACK).

### 2.2 Feature Computation

```
ALGORITHM ExtractSpectralFeatures(Λ, V, L_H, k)
────────────────────────────────────────────────────────────
Input:  Λ = (λ_1, …, λ_{k+1}) eigenvalues (sorted ascending)
        V ∈ ℝ^{n×(k+1)} eigenvectors
        L_H ∈ ℝ^{n×n} Laplacian
        k: number of blocks
Output: f ∈ ℝ^8 spectral feature vector

▷ Handle eigensolve failure
1.  if any(isnan(Λ)) then
2.      return [NaN] × 8

▷ ---- Feature 1: Spectral gap γ₂ ----
3.  γ₂ ← max(λ_2, 0)                      ▷ clamp numerical noise

▷ ---- Feature 6: Coupling energy δ² ----
▷ δ² = ‖L_H − L_block‖_F² where L_block is block-diagonal
▷ approximation from spectral partition.
▷ Compute after partition (Algorithm 3); for now use proxy:
4.  δ²_proxy ← ∑_{i=2}^{k+1} λ_i         ▷ sum of bottom non-trivial eigenvalues
▷ (Refined δ² computed in Algorithm 3 after clustering)

▷ ---- Feature 2: Spectral gap ratio δ²/γ² ----
5.  if γ₂ > ε_gap then                     ▷ ε_gap = 1e-12
6.      ratio ← δ²_proxy / γ₂²
7.  else
8.      ratio ← +∞                          ▷ disconnected or near-disconnected

▷ ---- Feature 3: Eigenvalue decay rate ----
▷ Fit exponential: λ_i ≈ α · exp(β · i) for i = 2, …, k+1
9.  log_λ ← log(max(λ_2, …, λ_{k+1}, ε_floor))   ▷ ε_floor = 1e-15
10. β ← linear_regression_slope(x=[2,…,k+1], y=log_λ)
11. decay_rate ← β                          ▷ negative = fast decay

▷ ---- Feature 4: Fiedler vector localization entropy ----
12. v₂ ← V[:, 1]                           ▷ Fiedler vector (index 1 = 2nd eigenvector)
13. p ← v₂² / ‖v₂‖²₂                      ▷ probability distribution
14. p ← max(p, ε_floor)                    ▷ avoid log(0)
15. H_loc ← −∑_j p_j · log(p_j)           ▷ Shannon entropy
▷ Normalize to [0, 1]:
16. H_loc ← H_loc / log(n)                 ▷ max entropy = log(n)

▷ ---- Feature 5: Algebraic connectivity ratio γ₂/γ_k ----
17. γ_k ← max(λ_{k+1}, ε_gap)
18. acr ← γ₂ / γ_k                        ▷ ∈ (0, 1]

▷ ---- Feature 8: Effective spectral dimension ----
19. λ_med ← median(Λ[1:])                  ▷ exclude λ_1 ≈ 0
20. eff_dim ← |{i : λ_i < λ_med / 10}|    ▷ count near-zero eigenvalues

▷ ---- Features 6, 7 require partition (deferred to Algorithm 3) ----
▷ Pack available features, mark 6 and 7 as provisional
21. f ← [γ₂, ratio, decay_rate, H_loc, acr, δ²_proxy, NaN, eff_dim]
22. return f
────────────────────────────────────────────────────────────
```

**Complexity:** $O(n \cdot k)$ for all 8 features (dominated by eigenvector operations).

**Numerical stability:**
- Eigenvalue clamping at 0 (line 3) handles numerical noise from
  Lanczos producing $\lambda_2 = -10^{-14}$.
- Entropy floor (line 14) prevents $-\infty$ contributions.
- Ratio guard (line 5) handles disconnected graphs ($\gamma_2 = 0$).
- XGBoost classifiers downstream handle $+\infty$ and NaN natively.

### 2.3 Complete Feature Extraction Pipeline

```
ALGORITHM SpectralFeatureExtraction(A_raw, k, eq_method)
────────────────────────────────────────────────────────────
Input:  A_raw ∈ ℝ^{m×n}, k (blocks), eq_method
Output: f ∈ ℝ^8 complete feature vector, metadata

1.  L_H, meta ← ConstructLaplacian(A_raw, eq_method)     ▷ Algorithm 1
2.  cache_key ← (meta.instance_id, meta.variant, k)
3.  Λ, V, eigstatus ← RobustEigensolve(L_H, k, cache_key)  ▷ Alg 2.1
4.  f_partial ← ExtractSpectralFeatures(Λ, V, L_H, k)      ▷ Alg 2.2
5.  ▷ Complete features 6 and 7 via partition
6.  P, labels ← SpectralPartition(V[:, 1:k], k)             ▷ Alg 3.1
7.  δ² ← ComputeCouplingEnergy(L_H, labels)                 ▷ Alg 3.2
8.  sep ← ComputeSeparabilityIndex(V[:, 1:k], labels)       ▷ Alg 3.3
9.  f_partial[5] ← δ²                      ▷ overwrite proxy with exact
10. f_partial[6] ← sep                     ▷ fill in separability
11. meta.eigstatus ← eigstatus
12. return f_partial, meta
────────────────────────────────────────────────────────────
```

**End-to-end time complexity for Algorithm 2:**

$$O\bigl(\underbrace{\text{nnz} \cdot d_{\max}}_{\text{Laplacian}} + \underbrace{T_{\text{Lanczos}} \cdot (\text{nnz}(L_H) + n)}_{\text{eigensolve}} + \underbrace{n \cdot k}_{\text{features}}\bigr)$$

**Bottleneck:** Eigensolve dominates for large $n$; Laplacian construction
dominates for large $d_{\max}$.

**Implementation language:** Python (scipy, numpy). The eigensolve calls
ARPACK (Fortran) and LOBPCG (C/Python) under the hood — no need to
reimplement.

---

## Algorithm 3: Spectral Partition Recovery

### Purpose
Recovers a $k$-block partition of variables from the bottom eigenvectors,
computes crossing weight (the L3 bound), and fills in features 6–7.

**Implements:** L3 partition-to-bound bridge. **Validates:** L3-sp spectral
partition quality.

### 3.1 Spectral Clustering

```
ALGORITHM SpectralPartition(V_k, k)
────────────────────────────────────────────────────────────
Input:  V_k ∈ ℝ^{n×k} bottom k eigenvectors (excluding trivial λ_1)
Output: labels ∈ {1, …, k}^n, partition P = {B_1, …, B_k}

1.  ▷ Row-normalize for rotation invariance
2.  for j = 1, …, n do
3.      r_j ← ‖V_k[j, :]‖₂
4.      if r_j > ε then
5.          V_k[j, :] ← V_k[j, :] / r_j
6.      else
7.          V_k[j, :] ← 0                  ▷ zero-degree vertex → assign later
8.
9.  ▷ k-means clustering on normalized rows
10. labels, centroids ← KMeans(V_k, k,
11.     n_init=10,                          ▷ 10 random restarts
12.     max_iter=300,
13.     algorithm='lloyd')                  ▷ Lloyd's algorithm
14.
15. ▷ Assign zero-degree vertices to nearest centroid
16. for j with r_j ≤ ε do
17.     labels[j] ← argmin_{c} ‖centroids[c]‖  ▷ assign to smallest block
18.
19. ▷ Construct partition
20. for b = 1, …, k do
21.     B_b ← {j : labels[j] = b}
22. P ← {B_1, …, B_k}
23. return labels, P
────────────────────────────────────────────────────────────
```

**Complexity:**
- Row normalization: $O(n \cdot k)$.
- k-means: $O(n \cdot k \cdot T_{\text{kmeans}} \cdot n_{\text{init}})$.
  With $k \leq 20$, $T_{\text{kmeans}} \leq 300$, $n_{\text{init}} = 10$:
  $O(60{,}000 \cdot n)$.
- **Space:** $O(n \cdot k)$.

**Alternative (rotation-invariant):** For improved robustness, cluster on the
Gram matrix $G = V_k V_k^T$ (entries $G_{jj'} = \langle V_k[j,:], V_k[j',:]\rangle$).
This is $O(n^2 k)$ to form and requires kernel k-means or spectral embedding,
so we only use it as a diagnostic (compare Silhouette scores).

### 3.2 Coupling Energy (Feature 6)

```
ALGORITHM ComputeCouplingEnergy(L_H, labels, k)
────────────────────────────────────────────────────────────
Input:  L_H ∈ ℝ^{n×n} Laplacian, labels ∈ {1,…,k}^n
Output: δ² (coupling energy)

1.  ▷ L_block = block-diagonal restriction of L_H to partition
2.  δ² ← 0
3.  for each nonzero entry L_H[j, j'] with j ≠ j' do
4.      if labels[j] ≠ labels[j'] then
5.          δ² ← δ² + L_H[j, j']²         ▷ off-block-diagonal entries
6.  ▷ Account for diagonal corrections
7.  for j = 1, …, n do
8.      D_block_j ← ∑_{j': labels[j']=labels[j]} |L_H[j, j']|  (j'≠j)
9.      δ² ← δ² + (L_H[j,j] − D_block_j)²
10. return δ²
────────────────────────────────────────────────────────────
```

**Complexity:** $O(\text{nnz}(L_H))$.

**Simpler formulation:** $\delta^2 = \|L_H - L_{\text{block}}\|_F^2$ where
$L_{\text{block}}$ is formed by zeroing all entries $L_H[j,j']$ where
$\text{labels}[j] \neq \text{labels}[j']$ and adjusting the diagonal.
Implemented as a single sparse-matrix pass.

### 3.3 Block Separability Index (Feature 7)

```
ALGORITHM ComputeSeparabilityIndex(V_k, labels)
────────────────────────────────────────────────────────────
Input:  V_k ∈ ℝ^{n×k}, labels ∈ {1,…,k}^n
Output: sep ∈ [-1, 1] (Silhouette score)

1.  ▷ Use Gram-matrix distances for rotation invariance
2.  ▷ d(j, j') = ‖V_k[j,:] − V_k[j',:]‖₂ (Euclidean in spectral embedding)
3.  ▷ But full pairwise is O(n²). Use sampled Silhouette:
4.  if n > 10000 then
5.      S ← random sample of 10000 indices
6.  else
7.      S ← {1, …, n}
8.
9.  ▷ Compute Silhouette on sample
10. for j ∈ S do
11.     a_j ← mean distance to same-cluster points
12.     b_j ← min over other clusters of mean distance to that cluster
13.     s_j ← (b_j − a_j) / max(a_j, b_j)
14. sep ← mean(s_j for j ∈ S)
15. return sep
────────────────────────────────────────────────────────────
```

**Complexity:** $O(|S|^2 \cdot k)$ for sampled Silhouette; $O(10^8 \cdot k)$
worst case with $|S| = 10{,}000$.

**Implementation note:** Use `sklearn.metrics.silhouette_score` with
`sample_size=10000` for efficiency.

### 3.4 Crossing Weight Computation (L3 Bound)

This is the core connection to Lemma L3: the crossing weight IS the computable
upper bound on LP relaxation gap degradation.

```
ALGORITHM ComputeCrossingWeight(A, labels_var, y_star)
────────────────────────────────────────────────────────────
Input:  A ∈ ℝ^{m×n} constraint matrix
        labels_var ∈ {1,…,k}^n variable-to-block assignment
        y_star ∈ ℝ^m optimal dual vector (from monolithic LP)
Output: cw (crossing weight = L3 bound)
        E_cross (set of crossing constraint indices)

1.  cw ← 0
2.  E_cross ← ∅
3.  for i = 1, …, m do
4.      S_i ← {j : A_{ij} ≠ 0}            ▷ support of constraint i
5.      blocks_i ← {labels_var[j] : j ∈ S_i}
6.      if |blocks_i| > 1 then              ▷ constraint crosses blocks
7.          E_cross ← E_cross ∪ {i}
8.          n_blocks_i ← |blocks_i|
9.          cw ← cw + |y*_i| · (n_blocks_i − 1)
10. return cw, E_cross
────────────────────────────────────────────────────────────
```

**Complexity:** $O(\text{nnz}(A))$ — single pass over $A$ in CSR format.

**Space:** $O(m)$ for $E_{\text{cross}}$.

**Connection to L3:** The output `cw` satisfies
$$z_{LP} - z_D(P) \leq \text{cw} = \sum_{i \in E_{\text{cross}}} |y^*_i| \cdot (n_{b,i} - 1)$$
where $n_{b,i}$ is the number of blocks spanned by constraint $i$.

**Obtaining $y^*$:** Solve the LP relaxation via SCIP (`SCIPlpSolveAndEval`
or PySCIPOpt `model.getLPSolInfo()`). Cost: one LP solve, typically fast
relative to MIP.

---

## Algorithm 4: Method-Aware Partition Refinement

### Purpose
Greedy local search to reduce the crossing weight (L3 bound) by reassigning
variables between blocks. Activated only in C-lite mode.

**Implements:** Proposition F3 (refinement convergence — monotone decrease,
convergence in $\leq m$ iterations).

### 4.1 Greedy Local Search

```
ALGORITHM RefinePartition(A, labels, y_star, method, max_iter)
────────────────────────────────────────────────────────────
Input:  A ∈ ℝ^{m×n}, labels ∈ {1,…,k}^n
        y_star ∈ ℝ^m LP dual vector
        method ∈ {benders, dw}
        max_iter: iteration cap (default 10)
Output: labels_refined ∈ {1,…,k}^n

1.  labels_cur ← copy(labels)
2.  cw_cur, _ ← ComputeCrossingWeight(A, labels_cur, y_star)
3.
4.  for iter = 1, …, max_iter do
5.      improved ← false
6.      ▷ Randomized variable ordering for tie-breaking
7.      perm ← random_permutation(1, …, n)
8.      for j ∈ perm do
9.          ▷ Compute marginal gain of moving j to each block
10.         best_block ← labels_cur[j]
11.         best_delta ← 0
12.         ▷ Constraints involving variable j
13.         C_j ← {i : A_{ij} ≠ 0}
14.
15.         for b = 1, …, k do
16.             if b = labels_cur[j] then continue
17.             ▷ Compute change in crossing weight if j moves to block b
18.             delta ← 0
19.             for i ∈ C_j do
20.                 S_i ← {j' : A_{ij'} ≠ 0}
21.                 ▷ Count blocks before and after move
22.                 blocks_before ← {labels_cur[j'] : j' ∈ S_i}
23.                 labels_temp ← labels_cur; labels_temp[j] ← b
24.                 blocks_after ← {labels_temp[j'] : j' ∈ S_i}
25.                 delta ← delta + |y*_i| · (|blocks_after| − |blocks_before|)
26.             if delta < best_delta then
27.                 best_delta ← delta
28.                 best_block ← b
29.
30.         ▷ Move if strictly improving
31.         if best_delta < −ε_improve then  ▷ ε_improve = 1e-10
32.             labels_cur[j] ← best_block
33.             cw_cur ← cw_cur + best_delta
34.             improved ← true
35.
36.     if not improved then break           ▷ local optimum reached
37.
38. return labels_cur
────────────────────────────────────────────────────────────
```

**Complexity per iteration:**
- For each variable $j$: examine $|C_j|$ constraints, each with up to
  $d_{\max}$ variables, across $k$ candidate blocks.
- Per variable: $O(k \cdot |C_j| \cdot d_{\max})$.
- Per iteration: $O(n \cdot k \cdot \bar{c} \cdot d_{\max})$ where $\bar{c}$
  is mean constraints-per-variable.
- Simplified: $O(\text{nnz} \cdot k \cdot d_{\max})$ per iteration.

**With $\text{max\_iter} = 10$:**
$$O(10 \cdot \text{nnz} \cdot k \cdot d_{\max})$$

**Convergence guarantee (F3):** The crossing weight is bounded below by 0 and
strictly decreases each iteration (when `improved = true`). Since each
iteration moves at least one variable, and there are only $k^n$ possible
partitions (finite), the algorithm converges. The practical bound of
$\text{max\_iter} = 10$ is a budget constraint, not a convergence issue.

**Space:** $O(n + m)$ for labels and temporary structures.

### 4.2 Method-Aware Weighting

For Benders vs. DW, the crossing-weight objective differs:

```
ALGORITHM MethodAwareCrossingWeight(A, labels, duals, method)
────────────────────────────────────────────────────────────
Input:  A, labels, method ∈ {benders, dw}
        duals: method-specific dual information
Output: cw (method-specific crossing weight)

1.  if method = benders then
2.      ▷ L3-C (Benders): weight by reduced costs of coupling variables
3.      cw ← 0
4.      for j = 1, …, n do
5.          C_j ← {i : A_{ij} ≠ 0}
6.          blocks_j ← {labels[i'] : i' ∈ some constraint touching j}
7.          if |blocks_j| > 1 then          ▷ j is a coupling variable
8.              cw ← cw + |r_j| · (|blocks_j| − 1)
9.      return cw
10.
11. else if method = dw then
12.     ▷ L3-C (DW): weight by linking-constraint duals
13.     cw ← 0
14.     for i = 1, …, m do
15.         S_i ← support of constraint i
16.         blocks_i ← {labels[j] : j ∈ S_i}
17.         if |blocks_i| > 1 then          ▷ i is a linking constraint
18.             cw ← cw + |μ_i| · (|blocks_i| − 1)
19.     return cw
────────────────────────────────────────────────────────────
```

**Implementation language:** Python. The inner loop of Algorithm 4.1 is the
hot path; if profiling shows >10s per iteration, port the delta computation
to Cython.

---

## Algorithm 5: Decomposition Selection Oracle

### Purpose
End-to-end prediction: given a MIP instance, decide {Benders, DW, none} and
optionally inject a partition.

**Implements:** The trained ML classifiers from §6 evaluation design.

### 5.1 Futility Check

```
ALGORITHM FutilityCheck(f, γ_thresh, model_futility)
────────────────────────────────────────────────────────────
Input:  f ∈ ℝ^8 spectral features
        γ_thresh: learned threshold for spectral gap
        model_futility: trained binary classifier
Output: futile ∈ {true, false}, confidence ∈ [0, 1]

1.  ▷ Fast heuristic pre-screen
2.  if f.γ₂ = NaN then                     ▷ eigensolve failed
3.      return futile=true, confidence=0.5  ▷ conservative default
4.
5.  if f.γ₂ < γ_thresh then                ▷ near-disconnected → futile
6.      return futile=true, confidence=0.9
7.
8.  ▷ Full classifier prediction
9.  p_futile ← model_futility.predict_proba(f)
10. if p_futile > 0.5 then
11.     return futile=true, confidence=p_futile
12. else
13.     return futile=false, confidence=1 − p_futile
────────────────────────────────────────────────────────────
```

**Note on γ_thresh:** This is NOT derived from T2 (whose constant is vacuous).
It is learned from cross-validated census data with asymmetric loss
(Section 6, futility predictor design).

### 5.2 Method Selection

```
ALGORITHM SelectDecompositionMethod(f_spec, f_synt, models, config)
────────────────────────────────────────────────────────────
Input:  f_spec ∈ ℝ^8 spectral features
        f_synt ∈ ℝ^25 syntactic features
        models: {rf, xgb, logreg} trained classifiers
        config: {use_spectral, use_syntactic, ensemble_method}
Output: method ∈ {benders, dw, none}
        confidence ∈ [0, 1]

1.  ▷ Assemble feature vector based on configuration
2.  if config.use_spectral and config.use_syntactic then
3.      x ← concat(f_spec, f_synt)         ▷ COMB-ALL: 33 features
4.  else if config.use_spectral then
5.      x ← f_spec                          ▷ SPEC-8: 8 features
6.  else
7.      x ← f_synt                          ▷ SYNT-25: 25 features
8.
9.  ▷ Futility check
10. futile, conf_f ← FutilityCheck(f_spec, γ_thresh, model_futility)
11. if futile and conf_f > 0.8 then
12.     return "none", conf_f
13.
14. ▷ Method prediction (ensemble or single model)
15. if config.ensemble_method = "vote" then
16.     votes ← {}
17.     for model ∈ {models.rf, models.xgb, models.logreg} do
18.         pred ← model.predict(x)
19.         prob ← model.predict_proba(x)
20.         votes[pred] ← votes.get(pred, 0) + max(prob)
21.     method ← argmax(votes)
22.     confidence ← votes[method] / sum(votes.values())
23. else
24.     ▷ Single best model (determined by nested CV)
25.     method ← models.best.predict(x)
26.     confidence ← max(models.best.predict_proba(x))
27.
28. return method, confidence
────────────────────────────────────────────────────────────
```

**Complexity:** $O(1)$ for prediction — fixed-size feature vector through
pre-trained model. RF with 100 trees: $O(100 \cdot \text{depth}) \approx O(2000)$.
XGBoost similar. Negligible relative to spectral computation.

### 5.3 Partition Dispatch (C-lite only)

```
ALGORITHM DispatchDecomposition(instance, method, labels, config)
────────────────────────────────────────────────────────────
Input:  instance: SCIP model object
        method ∈ {benders, dw}
        labels ∈ {1,…,k}^n variable-to-block assignment
        config: solver configuration
Output: result: {dual_bound, primal_bound, time, gap}

1.  if method = "dw" then
2.      ▷ Serialize partition to GCG .dec format
3.      dec_path ← write_dec_file(labels, instance)
4.      ▷ Launch GCG with .dec file
5.      result ← run_gcg(instance.mps_path, dec_path,
6.                        time_limit=config.time_limit)
7.
8.  else if method = "benders" then
9.      ▷ Map variable partition to Benders structure:
10.     ▷ Block 1 = master variables, Blocks 2…k = subproblem variables
11.     master_vars ← {j : labels[j] = 1}
12.     sub_vars ← {j : labels[j] ≠ 1}
13.     ▷ Invoke SCIP Benders via PySCIPOpt
14.     result ← run_scip_benders(instance, master_vars, sub_vars,
15.                                time_limit=config.time_limit)
16.
17. return result
────────────────────────────────────────────────────────────
```

**Implementation note:** GCG .dec format specifies block membership of
constraints (not variables); the variable-to-block mapping must be transposed
via the constraint structure. SCIP Benders requires `SCIPcreateBendersDefault`
with explicit variable partition.

---

## Algorithm 6: Census Pipeline

### Purpose
Embarrassingly parallel infrastructure for the MIPLIB 2017 decomposition
census. Per-instance: parse → presolve → extract features → decompose → log.

### 6.1 Instance Processing

```
ALGORITHM ProcessInstance(mps_path, k_range, config)
────────────────────────────────────────────────────────────
Input:  mps_path: path to MPS file
        k_range: [2, 5, 10, 20] candidate block counts
        config: {eq_methods, time_limits, output_dir}
Output: record: complete feature + decomposition result dict

1.  record ← {}
2.  record.instance_id ← basename(mps_path)
3.  record.start_time ← now()
4.
5.  ▷ ---- Phase 1: Parse and Presolve ----
6.  model ← SCIPModel()
7.  model.readProblem(mps_path)
8.  model.presolve()
9.  A ← extractConstraintMatrix(model)     ▷ PySCIPOpt sparse extraction
10. m, n ← A.shape
11. record.m, record.n, record.nnz ← m, n, nnz(A)
12. if m = 0 or n = 0 then
13.     record.status ← "trivial_after_presolve"
14.     return record
15.
16. ▷ ---- Phase 2: Syntactic Features ----
17. record.syntactic ← ExtractSyntacticFeatures(A, model)  ▷ 25 features
18.
19. ▷ ---- Phase 3: Spectral Features (budget: 30s) ----
20. for eq_method ∈ config.eq_methods do
21.     for k ∈ k_range do
22.         t_start ← now()
23.         try with timeout(30s):
24.             f, meta ← SpectralFeatureExtraction(A, k, eq_method)
25.             record.spectral[eq_method][k] ← f
26.             record.spectral_meta[eq_method][k] ← meta
27.         catch TimeoutError:
28.             record.spectral[eq_method][k] ← [NaN] × 8
29.             record.spectral_meta[eq_method][k] ← {status: "timeout"}
30.         record.spectral_time[eq_method][k] ← now() − t_start
31.
32. ▷ ---- Phase 4: LP Relaxation (for y*) ----
33. try with timeout(300s):
34.     model_lp ← copy(model)
35.     model_lp.setParam("limits/solutions", 0)  ▷ LP only
36.     model_lp.optimize()
37.     y_star ← getDualSolution(model_lp)
38.     record.lp_bound ← model_lp.getObjVal()
39.     record.lp_status ← "solved"
40. catch TimeoutError:
41.     y_star ← None
42.     record.lp_status ← "timeout"
43.
44. ▷ ---- Phase 5: Crossing Weights (if LP solved) ----
45. if y_star is not None then
46.     for k ∈ k_range do
47.         ▷ Use best equilibration method's partition
48.         labels ← record.best_partition[k]
49.         if labels is not None then
50.             cw, E_cross ← ComputeCrossingWeight(A, labels, y_star)
51.             record.crossing_weight[k] ← cw
52.             record.n_crossing[k] ← |E_cross|
53.
54. ▷ ---- Phase 6: Decomposition Evaluation ----
55. for time_limit ∈ [60, 300, 900, 3600] do
56.     ▷ Monolithic baseline
57.     try with timeout(time_limit + 60):
58.         result_mono ← run_scip(mps_path, time_limit)
59.         record.monolithic[time_limit] ← result_mono
60.     catch: record.monolithic[time_limit] ← {status: "error"}
61.
62.     ▷ GCG (DW reference)
63.     try with timeout(time_limit + 60):
64.         result_gcg ← run_gcg(mps_path, time_limit)
65.         record.gcg[time_limit] ← result_gcg
66.     catch: record.gcg[time_limit] ← {status: "error"}
67.
68.     ▷ SCIP Benders
69.     try with timeout(time_limit + 60):
70.         result_bend ← run_scip_benders_default(mps_path, time_limit)
71.         record.benders[time_limit] ← result_bend
72.     catch: record.benders[time_limit] ← {status: "error"}
73.
74. ▷ ---- Phase 7: Label Assignment ----
75. for time_limit ∈ [60, 300, 900, 3600] do
76.     best ← argmax over {mono, gcg, benders} of dual_bound
77.     if best = mono or all decomposition failed then
78.         record.label[time_limit] ← "none"
79.     else if best = gcg then
80.         record.label[time_limit] ← "dw"
81.     else
82.         record.label[time_limit] ← "benders"
83.
84. record.end_time ← now()
85. return record
────────────────────────────────────────────────────────────
```

**Per-instance time budget:**
- Spectral: $\leq 30$s per (method, k) pair × 3 methods × 4 k-values = 6 min max.
- LP relaxation: $\leq 5$ min.
- Decomposition: 4 time limits × 3 methods × (limit + 60s overhead) ≈ 5.5 hours max.
- **Dominant cost:** Decomposition evaluation (Phase 6).

### 6.2 Parallel Census Orchestration

```
ALGORITHM RunCensus(instance_list, tier, n_workers, config)
────────────────────────────────────────────────────────────
Input:  instance_list: paths to MPS files
        tier ∈ {pilot_50, paper_500, full_1065}
        n_workers: parallelism level
        config: census configuration
Output: census.parquet (complete census dataset)

1.  ▷ Stratified instance selection
2.  if tier = pilot_50 then
3.      instances ← stratified_sample(instance_list, 50,
4.          strata=[structure_type × size_bin])
5.  else if tier = paper_500 then
6.      instances ← stratified_sample(instance_list, 500,
7.          strata=[5_types × 5_sizes × ~20_per_cell])
8.  else
9.      instances ← instance_list           ▷ all 1065
10.
11. ▷ Initialize job queue (SQLite-backed, idempotent)
12. db ← SQLiteJobQueue("census_jobs.db")
13. for inst ∈ instances do
14.     db.enqueue(inst, status="pending")
15.
16. ▷ Parallel execution
17. with ProcessPool(n_workers) as pool:
18.     while db.has_pending() do
19.         batch ← db.dequeue(n_workers)
20.         results ← pool.map(ProcessInstance, batch, config)
21.         for result ∈ results do
22.             db.mark_complete(result.instance_id, result)
23.             ▷ Incremental checkpoint
24.             if db.completed_count() % 50 = 0 then
25.                 export_checkpoint(db, config.output_dir)
26.
27. ▷ Final export
28. census ← db.export_all()
29. census.to_parquet(config.output_dir / "census.parquet")
30. ▷ Compute aggregate statistics
31. compute_label_stability(census)         ▷ check <20% flip across cutoffs
32. compute_coverage_report(census)         ▷ target ≥80% valid
33. return census
────────────────────────────────────────────────────────────
```

**Scaling:**
- **Pilot (50 instances):** ~50 × 6 hours = 300 CPU-hours ≈ 12.5 hours on 24 cores.
- **Paper (500 instances):** ~500 × 6 hours = 3000 CPU-hours ≈ 5.2 days on 24 cores.
- **Full (1065 instances):** ~1065 × 6 hours = 6390 CPU-hours ≈ 11.1 days on 24 cores.
- **Spectral-only pass** (no decomposition): 1065 × 6 min = ~107 CPU-hours ≈ 4.5 hours on 24 cores.

**Idempotency:** SQLite job queue ensures crashed/timed-out instances can be
restarted without re-running completed ones.

---

## Complexity Summary

### Per-Instance End-to-End Pipeline

| Stage | Time | Space | Bottleneck |
|-------|------|-------|------------|
| Parse MPS | $O(|file|)$ | $O(\text{nnz})$ | I/O |
| Presolve (SCIP) | $O(m \cdot n)$ worst | $O(\text{nnz})$ | SCIP internal |
| Equilibration | $O(T_{\max} \cdot \text{nnz})$ | $O(m + n)$ | — |
| Laplacian (clique) | $O(\text{nnz} \cdot d_{\max})$ | $O(\text{nnz} \cdot d_{\max})$ | **Memory for dense constraints** |
| Laplacian (incidence) | $O(\text{nnz})$ | $O(\text{nnz})$ | — |
| Eigensolve (ARPACK) | $O(T_L \cdot \text{nnz}(L_H))$ | $O(n \cdot k)$ | **CPU for large $n$** |
| Feature extraction | $O(n \cdot k)$ | $O(k)$ | — |
| Spectral clustering | $O(n \cdot k^2 \cdot T_k)$ | $O(n \cdot k)$ | — |
| Crossing weight | $O(\text{nnz})$ | $O(m)$ | — |
| LP relaxation | $O(\text{LP solve})$ | $O(\text{nnz})$ | LP can be slow |
| Partition refinement | $O(10 \cdot \text{nnz} \cdot k \cdot d_{\max})$ | $O(n + m)$ | C-lite only |
| ML prediction | $O(1)$ | $O(1)$ | — |
| Decomposition solve | $O(\text{MIP time limit})$ | $O(\text{solver})$ | **Dominant by far** |

### What Dominates

1. **Census mode:** Decomposition evaluation (Phase 6) dominates by 2–3 orders
   of magnitude. The spectral analysis is <1% of total time.

2. **Prediction-only mode** (trained model, no decomposition):
   The eigensolve dominates: $O(T_L \cdot \text{nnz}(L_H))$ where
   $T_L \approx 50\text{--}300$. For the clique-expansion path, Laplacian
   construction may dominate if $d_{\max} \gg 1$.

3. **Scaling with instance size:**

| Parameter | Effect on spectral pipeline |
|-----------|-----------------------------|
| $m$ (constraints) | Hyperedges; linear in $\text{nnz}$ for Laplacian, linear for crossing weight |
| $n$ (variables) | Laplacian dimension; eigensolve is $O(n)$ per iteration |
| $\text{nnz}$ | Dominates everything; all operations are $O(\text{nnz})$ or $O(\text{nnz} \cdot d_{\max})$ |
| $d_{\max}$ | Determines Laplacian path; quadratic blowup if clique expansion |
| $k$ | Block count; linear factor in eigensolve iterations, quadratic in k-means |

### Scaling Regimes (Typical MIPLIB)

| Instance class | $m$ | $n$ | $\text{nnz}$ | $d_{\max}$ | Spectral time |
|---------------|-----|-----|---------------|-------------|--------------|
| Small (set cover) | 500 | 2000 | 10K | 20 | <1s |
| Medium (scheduling) | 5000 | 20K | 200K | 50 | 2–5s |
| Large (network) | 50K | 100K | 500K | 10 | 5–15s |
| Very large (supply chain) | 200K | 500K | 5M | 200 | 15–30s |
| Extreme (big-M) | 100K | 300K | 2M | 2000 | 10–30s (incidence) |

---

## Numerical Stability Analysis

### Critical Failure Modes and Mitigations

| Failure Mode | Probability | Detection | Mitigation |
|-------------|-------------|-----------|------------|
| $\kappa(A) > 10^{10}$ (big-M) | ~35% MIPLIB | Condition estimate during equilibration | Equilibration reduces to $O(10^{2\text{--}4})$; if still bad, flag in metadata |
| ARPACK no convergence | ~10% | Exception catch | LOBPCG fallback → explicit shift → NaN |
| $\gamma_2 = 0$ (disconnected) | ~5% | λ₂ check | Report as trivially decomposable; connected-component partition |
| Near-zero eigenvalue cluster | ~15% | $|\lambda_{k+1} - \lambda_k| < \epsilon$ | Warn; spectral clustering unreliable; increase $k$ or flag |
| Numerical cancellation in $L_H$ | ~5% | $\|L_H \cdot \mathbf{1}\| > \epsilon$ | Recompute with higher precision or regularize |
| k-means stuck in local minimum | ~10% | High variance across restarts | 10 restarts; report variance; use k-means++ init |

### Precision Requirements

- **Laplacian construction:** Double precision (64-bit) throughout. The
  Laplacian must satisfy $L_H \mathbf{1} = 0$ to machine precision.
  Verification: $\|L_H \mathbf{1}\|_\infty < n \cdot \epsilon_{\text{mach}}$.

- **Eigensolve:** Residual tolerance $10^{-8}$. For shift-invert, the
  factorization $(L_H - \sigma I) = LU$ must be computed in double precision.
  The shift $\sigma = -10^{-6}$ avoids the singularity at $\lambda_1 = 0$.

- **Feature computation:** Entropy and ratio features use log/division —
  guard against $\log(0)$ and division by zero with floor values.

---

## Implementation Language Recommendations

| Component | Language | Rationale |
|-----------|----------|-----------|
| Laplacian construction | Python (scipy.sparse) | Sparse matrix operations well-optimized in scipy |
| Inner loop of clique expansion | Cython/numba | $O(d^2)$ loop per constraint; JIT gives 10–50× |
| Eigensolve | Python (scipy→ARPACK/LOBPCG) | Calls Fortran/C under the hood |
| Feature extraction | Python (numpy) | Vectorized; no hot loops |
| k-means | Python (scikit-learn) | C-backed; well-optimized |
| Crossing weight | Python (sparse iteration) | Single pass; fast enough |
| Partition refinement | Python + Cython | Inner delta computation may need JIT |
| ML classifiers | Python (scikit-learn, xgboost) | Standard ML stack |
| Census orchestration | Python (multiprocessing) | Job queue + parallel map |
| SCIP/GCG interface | Python (PySCIPOpt) | Established bindings |
| .dec file serialization | Python | Simple text format |

**Overall:** Python throughout, with Cython/numba JIT for two identified hot
loops (clique expansion inner loop, refinement delta computation). No C++
needed — all performance-critical numerical code lives in compiled libraries
(ARPACK, BLAS, scikit-learn internals).

---

## Theorem/Lemma Connection Map

| Algorithm | Implements/Validates |
|-----------|---------------------|
| Alg 1 (Laplacian) | Definition of $L_H$; Prop F1 (permutation invariance); Prop F2 (scaling sensitivity) |
| Alg 2 (Features) | 8 feature definitions from §3; T2 motivational predictor $\delta^2/\gamma^2$ |
| Alg 3 (Partition) | Lemma L3 (partition-to-bound bridge); L3-sp (spectral partition quality) |
| Alg 4 (Refinement) | L3-C Benders/DW specializations; Prop F3 (convergence) |
| Alg 5 (Oracle) | Empirical validation of T2 correlation; futility prediction |
| Alg 6 (Census) | End-to-end validation infrastructure; produces labels for G1/G3 gates |

---

## Pseudocode Conventions

All pseudocode uses 1-based indexing. Sparse matrices are stored in CSR
(Compressed Sparse Row) format unless noted. The notation $\|x\|_p$ denotes
the $\ell_p$ norm. `NaN` propagation follows IEEE 754 semantics. All
algorithms are deterministic given a fixed random seed (for k-means restarts
and census sampling).
