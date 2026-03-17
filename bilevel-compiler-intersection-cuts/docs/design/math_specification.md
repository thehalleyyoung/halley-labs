# Mathematical Specification: A Solver-Agnostic Bilevel Optimization Compiler

## Preamble

This document enumerates every load-bearing mathematical result required to build a compiler that accepts high-level bilevel optimization specifications and emits valid, solver-targeted single-level reformulations. Each result is graded for novelty and difficulty.

**Notation conventions.** We write a generic bilevel program as:

$$
\min_{x \in X} \; F(x, y) \quad \text{s.t.} \quad G(x, y) \leq 0, \quad y \in S(x)
$$

where $S(x) = \arg\min_{y \in Y(x)} \{ f(x, y) \}$ is the lower-level solution set, $X \subseteq \mathbb{R}^{n_x}$, $Y(x) \subseteq \mathbb{R}^{n_y}$ is the lower-level feasible set parameterized by $x$, $F$ is the upper-level objective, and $f$ is the lower-level objective. We use "optimistic" formulation throughout unless stated otherwise (the leader selects the best $y$ from $S(x)$).

---

## M1. Reformulation Correctness Theory

### M1.1 KKT Reformulation Exactness

**Formal statement.** Consider the bilevel program where the lower level is:

$$
\min_{y} \; f(x,y) \quad \text{s.t.} \quad g_j(x,y) \leq 0, \; j = 1, \ldots, m
$$

with $f, g_j$ continuously differentiable. The *KKT reformulation* replaces $y \in S(x)$ with the system:

$$
\nabla_y f(x,y) + \sum_{j=1}^{m} \lambda_j \nabla_y g_j(x,y) = 0, \quad \lambda_j \geq 0, \quad \lambda_j g_j(x,y) = 0, \quad g_j(x,y) \leq 0
$$

**Theorem (KKT Exactness).** The KKT reformulation is exact (i.e., every global optimum of the bilevel program corresponds to a feasible point of the KKT-reformulated single-level program, and vice versa) if and only if:

1. The lower-level problem is convex in $y$ for every fixed $x$ (i.e., $f(x, \cdot)$ is convex and $Y(x)$ is a convex set), AND
2. A constraint qualification (CQ) holds at every lower-level optimal $y$ for every feasible $x$. Sufficient CQs include:
   - **LICQ** (Linear Independence CQ): The gradients $\{\nabla_y g_j(x,y) : j \in \mathcal{A}(x,y)\}$ are linearly independent, where $\mathcal{A}(x,y) = \{j : g_j(x,y) = 0\}$.
   - **MFCQ** (Mangasarian-Fromovitz CQ): There exists $d \in \mathbb{R}^{n_y}$ such that $\nabla_y g_j(x,y)^\top d < 0$ for all $j \in \mathcal{A}(x,y)$.
   - **Slater's condition** (for convex $g_j$): There exists $\hat{y}$ with $g_j(x,\hat{y}) < 0$ for all $j$.

Without convexity, KKT conditions are necessary but not sufficient for optimality, and the reformulation is a *relaxation* (potentially strict).

**Why it's load-bearing.** The compiler must decide when the KKT reformulation is *sound* (preserves all bilevel optima) and *complete* (introduces no spurious optima). Applying KKT to a nonconvex lower level silently produces a wrong answer. The compiler's type checker must verify these conditions statically.

**Novelty assessment.** Known. Convexity + CQ sufficiency is classical (Dempe 2002, Bard 1998). The specific taxonomy of CQs and their relative strength is textbook material.

**Difficulty grade.** A — Direct application of known results.

---

### M1.2 Strong Duality Reformulation

**Formal statement.** When the lower level is a convex program with strong duality, the bilevel program can be reformulated by replacing $y \in S(x)$ with the primal-dual system:

$$
f(x,y) \leq d(x, \lambda), \quad g(x,y) \leq 0, \quad \lambda \geq 0
$$

where $d(x, \lambda)$ is the dual objective. Under strong duality (zero duality gap), this is equivalent to:

$$
f(x,y) = d(x, \lambda), \quad g(x,y) \leq 0, \quad \lambda \geq 0
$$

**Theorem (Strong Duality Reformulation Correctness).** Let the lower level be convex with strong duality holding for all feasible $x$. Then the single-level reformulation obtained by replacing the lower-level optimality condition with $f(x,y) = d(x,\lambda)$ plus primal-dual feasibility is exact.

*Special case (LP lower level):* When $f(x,y) = c(x)^\top y$ and $Y(x) = \{y : A(x)y \leq b(x), \, y \geq 0\}$, strong duality always holds (when the LP is feasible and bounded), and the reformulation introduces the dual constraint $A(x)^\top \lambda \geq c(x)$ with $f(x,y) = b(x)^\top \lambda$.

**Why it's load-bearing.** For LP and conic lower levels, the strong-duality reformulation avoids complementarity constraints entirely, often producing a problem that is more tractable for branch-and-bound solvers. The compiler must know when to prefer this over KKT.

**Novelty assessment.** Known. LP duality reformulation is standard (Fortuny-Amat & McCarl 1981). Conic extensions are known (Ben-Tal & Nemirovski 2001).

**Difficulty grade.** A

---

### M1.3 Value Function Reformulation

**Formal statement.** The *value function reformulation* replaces $y \in S(x)$ with:

$$
f(x,y) \leq \varphi(x), \quad y \in Y(x)
$$

where $\varphi(x) = \min_{y \in Y(x)} f(x,y)$ is the *optimal value function* of the lower level.

**Theorem (Value Function Equivalence — Outrata 1990, Ye & Zhu 1995).** The value function reformulation is *always* exact: $(x^*, y^*)$ solves the bilevel program if and only if it solves:

$$
\min_{x,y} \; F(x,y) \quad \text{s.t.} \quad G(x,y) \leq 0, \quad f(x,y) \leq \varphi(x), \quad y \in Y(x)
$$

**Theorem (Regularity for Value Function Approach — Dempe & Zemkoho 2012).** To derive KKT-type optimality conditions for the value-function-reformulated problem, one needs regularity of the constraint $f(x,y) - \varphi(x) \leq 0$. This requires *partial calmness* of the bilevel program at the solution, or alternatively, inner semicontinuity of the lower-level solution map $S(\cdot)$.

**Key difficulty.** $\varphi(x)$ is generally nonsmooth and nonconvex even when $f$ and $Y$ are smooth and convex. Computing $\varphi(x)$ requires solving the lower-level problem, creating a circular dependency for direct use. The compiler must either:
- Represent $\varphi$ implicitly (via cutting planes / column generation), or
- Compute bounds on $\varphi$ (via relaxation of the lower level).

**Why it's load-bearing.** The value function reformulation is the *only* reformulation that is always exact, making it the fallback for nonconvex and non-smooth lower levels. Without it, the compiler has no correct path for general problems.

**Novelty assessment.** Known in principle. Computational realization as a compiler pass (cutting-plane generation of $\varphi$-constraints) is partially known (Mitsos et al. 2008, Kleniati & Adjiman 2014).

**Difficulty grade.** B — The theory is known; encoding it as a compiler transformation with correct termination criteria requires nontrivial engineering-mathematics.

---

### M1.4 Correctness for Mixed-Integer Lower Levels

**Formal statement.** When the lower level contains integer variables, $y = (y^c, y^d)$ with $y^d \in \mathbb{Z}^{n_d}$:

$$
S(x) = \arg\min_{y^c, y^d} \{ f(x, y^c, y^d) : g(x, y^c, y^d) \leq 0, \; y^d \in \mathbb{Z}^{n_d} \}
$$

**Fact.** KKT conditions do not apply (the feasible set is non-convex and has empty interior in the integer components). Strong duality does not hold in general.

**Theorem (Enumeration-Based Reformulation — Moore & Bard 1990, DeNegre 2011).** If the lower-level integer variables are bounded ($y^d \in \{0,1\}^{n_d}$ or finitely many values), the bilevel program can be reformulated by enumerating feasible integer assignments:

$$
\min_{x, y^c, y^d} \; F(x, y^c, y^d) \quad \text{s.t.} \quad \forall \hat{y}^d \in \mathcal{D}: \; f(x, y^c, y^d) \leq f(x, \hat{y}^c(\hat{y}^d), \hat{y}^d)
$$

where $\hat{y}^c(\hat{y}^d)$ solves the continuous part given $\hat{y}^d$. This is correct but exponentially large.

**Theorem (Bilevel with Integer Lower Level via Disjunctive Cuts — Fischetti et al. 2017, Tahernejad et al. 2020).** For bilevel mixed-integer programs where the lower level is a mixed-integer linear program, intersection cuts and disjunctive programming can yield valid reformulations without full enumeration.

**Theorem (Impossibility Boundary).** When the lower level is a general (non-bounded) integer program, no finite reformulation to a single-level program exists in general. The value function $\varphi(x)$ may be discontinuous, and the bilevel feasible set may be neither closed nor connected.

**Why it's load-bearing.** Many practical bilevel problems (network interdiction, defender-attacker) have integer lower levels. Without this theory, the compiler cannot handle them or must silently reject them. The compiler needs a decision tree: bounded integers → enumeration/disjunctive; unbounded → value function with branch-and-bound; otherwise → reject with diagnostic.

**Novelty assessment.** Partially known. Individual reformulations are published. The *systematic classification* of which reformulation applies to which integer structure, and the impossibility boundaries, requires synthesizing results from multiple papers.

**Difficulty grade.** B

---

### M1.5 Pessimistic vs. Optimistic Formulations

**Formal statement.** The *pessimistic* bilevel program is:

$$
\min_{x \in X} \max_{y \in S(x)} F(x,y) \quad \text{s.t.} \quad G(x,y) \leq 0
$$

**Theorem (Wiesemann et al. 2013).** The pessimistic bilevel program with convex lower level can be reformulated as a semi-infinite program. Under compactness of $Y(x)$ and continuity conditions, it admits a finite reformulation via robust optimization techniques.

**Theorem (Non-equivalence).** In general, the optimistic and pessimistic bilevel optima differ: $F^{opt} \leq F^{pess}$. They coincide if and only if $S(x)$ is a singleton for all feasible $x$ (unique lower-level solution).

**Why it's load-bearing.** The compiler must track which formulation the user intends. Using the wrong one silently changes the problem semantics. The type system must tag problems as optimistic/pessimistic and select reformulations accordingly.

**Novelty assessment.** Known.

**Difficulty grade.** A

---

## M2. Reformulation Selection Theory

### M2.1 Reformulation Strength Ordering

**Formal statement.** Given a bilevel program $P$ and two reformulations $R_1, R_2$ that are both valid (exact), define $R_1 \succeq R_2$ ("$R_1$ is at least as strong as $R_2$") if the continuous relaxation of $R_1$ is contained in the continuous relaxation of $R_2$:

$$
\text{relax}(R_1) \subseteq \text{relax}(R_2)
$$

**Theorem (Big-M vs. SOS1 Complementarity — Pineda & Morales 2019, Kleinert et al. 2020).** For KKT reformulation with complementarity $\lambda_j g_j = 0$:

- *Big-M linearization* ($\lambda_j \leq M(1 - z_j)$, $-g_j \leq M z_j$, $z_j \in \{0,1\}$) gives a relaxation whose tightness depends on $M$.
- *SOS1 formulation* ($(\lambda_j, -g_j) \in \text{SOS1}$) avoids big-M but requires SOS-capable solvers.
- *Fortuny-Amat regularization* is equivalent to big-M.

The continuous relaxation of big-M with $M \to \infty$ is strictly weaker than the SOS1 relaxation.

**Theorem (Strong Duality vs. KKT for LP Lower Levels — Kleinert & Schmidt 2021).** For bilevel programs with LP lower level, the strong-duality reformulation (primal objective = dual objective) produces a tighter continuous relaxation than the KKT reformulation with big-M complementarity, for any finite $M$.

**Why it's load-bearing.** The compiler's optimization pass must choose among reformulations. Without a formal strength ordering, it cannot reason about which reformulation will yield better solver performance (tighter relaxation → fewer branch-and-bound nodes → faster solve).

**Novelty assessment.** Partially known. Pairwise comparisons exist for specific reformulation pairs. A *complete lattice* ordering all standard reformulations for a given problem class is new.

**Difficulty grade.** B — Extending pairwise results to a comprehensive partial order requires careful analysis of each combination.

---

### M2.2 Structure-Dependent Reformulation Selection

**Formal statement.** Define a *problem signature* $\sigma(P)$ as a tuple encoding:
- Lower-level convexity class (linear / convex quadratic / convex / nonconvex)
- Variable types (continuous / integer / mixed)
- Constraint qualification status (LICQ / MFCQ / Slater / none verifiable)
- Coupling structure (upper-level variables in lower-level objective only / constraints only / both)
- Dimension parameters $(n_x, n_y, m)$

**Theorem (Selection Correctness — New).** There exists a function $\rho: \Sigma \to 2^{\mathcal{R}}$ mapping problem signatures to the set of *valid* (exact) reformulations. For each $\sigma \in \Sigma$:

$$
\rho(\sigma) = \{ R \in \mathcal{R} : R \text{ is exact for all problems with signature } \sigma \}
$$

This function can be computed from the conditions in M1.1–M1.5. Specifically:
- $\sigma$ has convex lower level + CQ ⟹ KKT $\in \rho(\sigma)$
- $\sigma$ has convex lower level + strong duality ⟹ Duality $\in \rho(\sigma)$
- $\sigma$ has bounded integer lower level ⟹ Enumeration $\in \rho(\sigma)$
- Always: ValueFunction $\in \rho(\sigma)$ (but computationally expensive)

**Why it's load-bearing.** This is the *decision procedure* at the heart of the compiler. Without it, the compiler cannot automatically choose a reformulation.

**Novelty assessment.** New as a formal, complete decision procedure. Individual pieces are known; the synthesis into a computable function on signatures is novel.

**Difficulty grade.** B

---

### M2.3 Complexity of Reformulation Selection

**Formal statement.** Given a bilevel problem $P$ and a set of target solvers $\mathcal{S}$, the *optimal reformulation selection problem* is:

$$
\min_{R \in \rho(\sigma(P)), \; s \in \mathcal{S}} \; \text{cost}(R, s, P)
$$

where $\text{cost}(R, s, P)$ is the estimated computational cost of solving reformulation $R$ of $P$ on solver $s$.

**Conjecture (NP-hardness of optimal selection).** Even when restricted to bilevel LPs with a fixed set of three reformulations (KKT+big-M, KKT+SOS1, strong duality) and two solvers (MIP solver, NLP solver), the optimal reformulation selection problem is NP-hard, by reduction from the minimum-cost constraint encoding problem.

**Pragmatic result.** In practice, heuristic selection based on $\sigma(P)$ using the strength ordering from M2.1 is polynomial and empirically effective.

**Why it's load-bearing.** If optimal selection is intractable, the compiler must use heuristics. The compiler's architecture must be designed around polynomial-time selection rules, not exact optimization of reformulation choice.

**Novelty assessment.** New. The complexity of reformulation *selection* (as opposed to reformulation *solving*) has not been studied.

**Difficulty grade.** C — Requires a novel reduction. The conjecture is plausible but unproven.

---

### M2.4 Approximation Bounds for Inexact Reformulations

**Formal statement.** When no exact reformulation is tractable (e.g., nonconvex lower level without bounded integers), the compiler may apply an *inexact* reformulation $\tilde{R}$ that relaxes the lower-level optimality condition.

**Definition.** A reformulation $\tilde{R}$ is an *$\epsilon$-approximate reformulation* of bilevel program $P$ if for every feasible $(x, y)$ of $\tilde{R}$:

$$
f(x,y) \leq \varphi(x) + \epsilon
$$

i.e., $y$ is $\epsilon$-optimal for the lower level.

**Theorem (Dempe & Franke 2016, extended).** If $F$ is $L_F$-Lipschitz in $y$ and the lower-level solution map $S(x)$ is $\kappa$-Lipschitz in the value function sense, then an $\epsilon$-approximate reformulation yields an upper-level objective value within $L_F \cdot \kappa \cdot \epsilon$ of the true bilevel optimum, under appropriate compactness assumptions.

**Corollary (Regularization bounds).** Tikhonov regularization of the lower level (adding $\frac{\delta}{2}\|y\|^2$) produces an $\epsilon(\delta)$-approximate reformulation where $\epsilon(\delta) \to 0$ as $\delta \to 0$, with $\epsilon(\delta) = O(\delta)$ for strongly convex lower levels.

**Why it's load-bearing.** The compiler needs to report solution quality guarantees even when it cannot produce an exact reformulation. Without these bounds, the user has no way to know if the compiler's output is meaningful.

**Novelty assessment.** Partially known. Individual bounds exist. The *compositional* error bound through a chain of reformulation passes (see M4.3) is new.

**Difficulty grade.** B

---

## M3. Bilevel Complexity Results

### M3.1 Baseline Complexity Results (Known)

**Theorem (Jeroslow 1985).** Deciding whether the optimal value of a bilevel linear program is at most $k$ is NP-hard (even when both levels are linear programs with continuous variables).

**Theorem (Hansen, Jaumard & Savard 1992).** Bilevel LP is $\Sigma^p_2$-hard (i.e., NP-hard even with an NP oracle).

**Theorem (Lodi, Ralphs & Woeginger 2014).** Bilevel linear programming with integer lower-level variables is $\Sigma^p_2$-hard, and the problem is $\Sigma^p_2$-complete under standard encodings.

**Theorem (Caprara et al. 2016).** Bilevel LP with integer variables in both levels is $\Sigma^p_2$-complete. Bilevel integer programming where only the upper level has integer variables is NP-complete.

**Why it's load-bearing.** These results set the *floor* for solver expectations: no compiler can make bilevel problems tractable in general. The compiler's promise is structural exploitation, not complexity collapse.

**Novelty assessment.** Known.

**Difficulty grade.** A — citing known results.

---

### M3.2 Complexity of Compiler Passes

**Formal statement.** Define the following compiler decision problems:

1. **REFORMULATION-VALIDITY**: Given a bilevel program $P$ and a reformulation $R$, is $R$ exact for $P$?
2. **CQ-VERIFICATION**: Given a bilevel program $P$, does LICQ (or MFCQ) hold at all lower-level optimal points for all feasible $x$?
3. **CONVEXITY-DETECTION**: Given the lower-level problem parameterized by $x$, is it convex in $y$ for all feasible $x$?

**Theorem (CQ-VERIFICATION is co-NP-hard — New).** Deciding whether LICQ holds everywhere is co-NP-hard, by reduction from checking linear independence of a parametric matrix family.

**Theorem (CONVEXITY-DETECTION — known, Ahmad et al.).** Deciding whether a polynomial function is convex is NP-hard in general (checking whether the Hessian is PSD everywhere). For quadratic functions, it reduces to checking PSD-ness of a single matrix (polynomial time). For structured representations (sum-of-squares, DCP rules), convexity can be verified in polynomial time.

**Theorem (REFORMULATION-VALIDITY).** In general, REFORMULATION-VALIDITY is $\Pi^p_2$-hard (complementary to the bilevel problem itself). However, *sufficient conditions* for validity (from M1.1–M1.4) are checkable in polynomial time given a DCP-tagged problem representation.

**Why it's load-bearing.** The compiler's static analysis passes must verify conditions in polynomial time. If verification is intractable, the compiler must use sufficient (conservative) conditions and document incompleteness.

**Novelty assessment.** Partially new. The formalization of compiler passes as decision problems and their complexity classification is new. Individual complexity results are known or follow from known techniques.

**Difficulty grade.** B

---

### M3.3 Complexity of the Reformulation Output

**Formal statement.** Given a bilevel program $P$ of encoding size $|P|$, the reformulated single-level program $R(P)$ may be exponentially larger.

**Theorem (Size Blowup Bounds).** 
- KKT reformulation: $|R_{KKT}(P)| = O(|P| + n_y \cdot m)$ — polynomial blowup (adds $m$ multipliers + $m$ complementarity constraints).
- Strong duality reformulation: $|R_{SD}(P)| = O(|P|)$ — constant factor blowup for LP lower levels.
- Enumeration for binary lower level: $|R_{enum}(P)| = O(|P| \cdot 2^{n_d})$ — exponential in number of integer variables.
- Disjunctive reformulation: $|R_{disj}(P)| = O(|P| \cdot 2^{n_d})$ worst case, but lazy generation avoids full materialization.

**Why it's load-bearing.** The compiler must predict output size to decide feasibility of reformulation. An exponential blowup may make a theoretically valid reformulation practically useless.

**Novelty assessment.** Known individually; systematized compilation is partially new.

**Difficulty grade.** A

---

## M4. Convergence and Approximation Guarantees

### M4.1 Regularization Convergence

**Formal statement.** Consider the *Tikhonov-regularized* bilevel program:

$$
\min_{x} F(x, y_\delta(x)) \quad \text{where} \quad y_\delta(x) = \arg\min_{y \in Y(x)} \left\{ f(x,y) + \frac{\delta}{2}\|y\|^2 \right\}
$$

**Theorem (Convergence — Loridan & Morgan 1996).** Under:
1. Compactness of $X$ and uniform boundedness of $Y(x)$,
2. Continuity of $F, f, g$,
3. Uniqueness of $y_\delta(x)$ for each $\delta > 0$ (guaranteed by strong convexity of the regularized lower level when $f$ is convex),

any accumulation point of solutions $(x_\delta, y_\delta)$ as $\delta \to 0^+$ is an optimistic bilevel optimal solution.

**Convergence rate.** For the case where $f$ is strongly convex with parameter $\mu$:

$$
\|y_\delta(x) - y_0(x)\| \leq \frac{\delta \cdot \|y_0(x)\|}{\mu}
$$

For merely convex $f$, convergence is guaranteed but without a rate in general.

**Why it's load-bearing.** The compiler must decide regularization strength $\delta$ and know when the regularized solution is a good approximation. A convergence rate allows the compiler to set $\delta$ to achieve a target accuracy.

**Novelty assessment.** Known. Rate analysis for specific structures is known (Dempe & Mehlitz 2023).

**Difficulty grade.** A

---

### M4.2 Penalty Method Convergence

**Formal statement.** The *penalty reformulation* replaces $y \in S(x)$ with $f(x,y) - \varphi(x) \leq 0$ enforced via penalty:

$$
\min_{x,y} \; F(x,y) + \pi \cdot \max(0, f(x,y) - \varphi(x)) \quad \text{s.t.} \quad G(x,y) \leq 0, \; y \in Y(x)
$$

**Theorem (Exact Penalization — Ye & Zhu 2010).** Under *partial calmness* of the bilevel program at a solution $(x^*, y^*)$, there exists a finite $\bar{\pi}$ such that for all $\pi \geq \bar{\pi}$, the penalty reformulation is exact (the penalized problem has the same optimal solution as the bilevel program).

**Definition (Partial Calmness).** The bilevel program is *partially calm* at $(x^*, y^*)$ if there exist $\epsilon > 0$ and $\kappa > 0$ such that for all $(x,y)$ feasible with $\|(x,y) - (x^*,y^*)\| < \epsilon$:

$$
F(x,y) - F(x^*,y^*) + \kappa |f(x,y) - \varphi(x)| \geq 0
$$

**Theorem (Partial Calmness for LP bilevel — Henrion, Surowiec 2011).** Bilevel linear programs are always partially calm. Thus exact penalization applies to all bilevel LPs.

**Why it's load-bearing.** Penalty methods are a primary implementation strategy. The compiler must know when exact penalization applies (guaranteeing correctness for finite $\pi$) versus when it only gives asymptotic convergence.

**Novelty assessment.** Known.

**Difficulty grade.** A

---

### M4.3 Error Propagation Through Reformulation Chains

**Formal statement.** The compiler may apply a *sequence* of reformulation passes $R_1, R_2, \ldots, R_k$, each introducing approximation error $\epsilon_i$. Define the *chain error*:

$$
\epsilon_{\text{chain}} = \text{error}(R_k \circ R_{k-1} \circ \cdots \circ R_1(P))
$$

**Theorem (Compositional Error Bound — New).** Suppose each pass $R_i$ is an $\epsilon_i$-approximate reformulation (in the sense of M2.4), and the objective $F$ is $L_F$-Lipschitz. If the lower-level-to-upper-level error amplification factor for pass $i$ is $\kappa_i$, then:

$$
|F^* - F^*_{\text{chain}}| \leq L_F \cdot \prod_{i=1}^{k} (1 + \kappa_i) \cdot \sum_{i=1}^{k} \frac{\epsilon_i}{\prod_{j=1}^{i}(1 + \kappa_j)}
$$

Under simplifying assumptions ($\kappa_i = \kappa$ for all $i$):

$$
|F^* - F^*_{\text{chain}}| \leq L_F \cdot \frac{(1+\kappa)^k - 1}{\kappa} \cdot \max_i \epsilon_i
$$

**Key implication.** Error amplification is potentially exponential in the number of passes $k$. The compiler must minimize the chain length and track accumulated error.

**Why it's load-bearing.** The compiler's multi-pass architecture means errors compose. Without this bound, the compiler cannot guarantee solution quality for multi-step reformulations.

**Novelty assessment.** New. Compositional error analysis for reformulation pipelines has not appeared in the bilevel optimization literature.

**Difficulty grade.** C — The bound itself follows from Lipschitz analysis, but proving tightness and identifying conditions where exponential blowup is avoidable requires new ideas.

---

### M4.4 Convergence of Cutting Plane Methods for Value Function

**Formal statement.** For the value function reformulation, generate outer approximations of $\varphi(x)$ iteratively:

At iteration $t$, solve:

$$
\min_{x,y} F(x,y) \quad \text{s.t.} \quad G(x,y) \leq 0, \quad f(x,y) \leq \ell_t(x), \quad y \in Y(x)
$$

where $\ell_t(x) = \max_{i \leq t} \{ \alpha_i + \beta_i^\top x \}$ is a piecewise-linear under-approximation of $\varphi(x)$, updated by solving the lower level at the current $x_t$ to get a new cut.

**Theorem (Finite Convergence for LP — Mitsos 2010).** When both levels are linear, the cutting-plane procedure converges in finitely many iterations (bounded by the number of bases of the lower-level LP).

**Theorem (Convergence for Convex Lower Level).** When the lower level is convex with bounded feasible set, the cutting-plane procedure converges to a global optimum under standard regularity. The convergence rate is $O(1/t)$ in objective value for Lipschitz $\varphi$.

**Why it's load-bearing.** This provides the algorithmic implementation of the value function reformulation (M1.3) and guarantees termination.

**Novelty assessment.** Known for LP. Extension to general convex with rate is partially known (Mitsos et al. 2008).

**Difficulty grade.** B

---

## M5. Type System / Intermediate Representation Theory

### M5.1 Formal Grammar for Bilevel Optimization Problems

**Formal statement.** Define the grammar $\mathcal{G}_{BOP}$ for bilevel optimization problems:

```
Program       ::= min Objective s.t. Constraints, LowerLevel
LowerLevel    ::= min Objective s.t. Constraints
Objective     ::= Expression
Constraints   ::= Constraint (',' Constraint)*
Constraint    ::= Expression '<=' Expression
                |  Expression '==' Expression
                |  Variable 'in' Domain
Expression    ::= Constant
                |  Variable
                |  Expression BinOp Expression
                |  UnaryOp Expression
                |  FunctionCall '(' Expression (',' Expression)* ')'
BinOp         ::= '+' | '-' | '*' | '/'
UnaryOp       ::= '-' | 'transpose'
Domain        ::= 'Reals' | 'Integers' | 'Binary' | Interval
                |  'PSD_Cone' | 'SOC'
Variable      ::= UpperVar | LowerVar | SharedParam
```

**Theorem (Expressiveness).** $\mathcal{G}_{BOP}$ can express:
- All bilevel linear programs
- Bilevel convex quadratic programs
- Bilevel mixed-integer programs (with `Domain` including `Integers`)
- Bilevel conic programs (with `Domain` including cone constraints)
- Multi-level programs (by nesting `LowerLevel`)

$\mathcal{G}_{BOP}$ cannot express:
- Bilevel programs with equilibrium constraints (MPEC/EPEC) without extension
- Stochastic bilevel programs (requires additional probability space constructs)

**Why it's load-bearing.** The grammar defines what the compiler can accept. Incompleteness here means classes of problems are silently excluded.

**Novelty assessment.** Partially new. Modeling languages (GAMS, Pyomo) have implicit grammars. A formal grammar specifically for bilevel problems with explicit variable scoping (upper vs. lower) is new.

**Difficulty grade.** B — Defining the grammar is straightforward; proving expressiveness and decidability of parsing is nontrivial.

---

### M5.2 Type System for Problem Structure

**Formal statement.** Define a type system $\mathcal{T}$ over the AST produced by $\mathcal{G}_{BOP}$. Types annotate expressions and subproblems with structural properties:

```
ExprType  ::= Affine(vars) | Convex(vars) | Concave(vars) | Quadratic(sign, vars) 
            | General(vars)
VarType   ::= Continuous | Integer | Binary
ProbType  ::= (obj: ExprType, constrs: [ExprType], vars: [(name, VarType, level)])
CQStatus  ::= LICQ | MFCQ | Slater | None
```

**Typing rules (DCP-style — Grant & Boyd 2006, extended):**

1. **Affine composition:** $\text{Affine} \circ \text{Affine} = \text{Affine}$
2. **Convexity preservation:** $\text{Convex} + \text{Convex} = \text{Convex}$; $\alpha \cdot \text{Convex} = \text{Convex}$ for $\alpha \geq 0$
3. **Composition rule:** $h(\text{Affine})$ is $\text{Convex}$ if $h$ is convex; $h(\text{Convex})$ is $\text{Convex}$ if $h$ is convex and nondecreasing
4. **Integer contamination:** Any expression involving an Integer-typed variable cannot be typed as Convex (with respect to that variable)

**Theorem (Type Inference is Decidable).** Type inference under the DCP-extended rules is decidable in $O(|AST|)$ time — linear in the size of the abstract syntax tree.

**Theorem (Conservatism).** The type system is *sound* (every expression typed Convex is truly convex) but *incomplete* (some convex expressions cannot be recognized as such by DCP rules). Specifically, the type system fails to recognize:
- Convexity arising from domain restrictions (e.g., $x \log x$ is convex on $x > 0$ but requires knowing the domain)
- Hidden convexity from variable substitutions

**Why it's load-bearing.** The type system is how the compiler *automatically detects* problem structure without user annotation. Its soundness ensures correct reformulations; its incompleteness determines what the compiler can handle without hints.

**Novelty assessment.** Partially new. DCP is known (CVX/CVXPY). Extending DCP to bilevel programs with variable scoping (upper vs. lower level types, coupling analysis) is new.

**Difficulty grade.** B

---

### M5.3 Soundness Theorem for the Compiler

**Formal statement.** The central correctness theorem.

**Theorem (Compiler Soundness — New).** Let $P$ be a bilevel program in $\mathcal{G}_{BOP}$. Let $\sigma = \text{typecheck}(P)$ be the inferred signature. Let $R = \text{select}(\sigma)$ be the selected reformulation. Let $Q = R(P)$ be the single-level output. Then:

1. **Soundness (relaxation):** Every optimal solution of $P$ maps to a feasible point of $Q$ with the same or better objective value.
2. **Completeness (for exact reformulations):** If $R \in \rho_{\text{exact}}(\sigma)$, then every optimal solution of $Q$ maps back to an optimal solution of $P$.
3. **Bounded error (for approximate reformulations):** If $R \in \rho_{\epsilon}(\sigma)$, then $|F^*_P - F^*_Q| \leq \epsilon(\sigma)$, where $\epsilon(\sigma)$ is computable from the type information.

**Proof structure.** By case analysis on $\sigma$:
- If lower level types as Convex + CQ holds → KKT or duality reformulation. Correctness by M1.1/M1.2.
- If lower level types as Linear → strong duality reformulation. Correctness by M1.2.
- If lower level has Integer variables, bounded → enumeration/disjunctive. Correctness by M1.4.
- Otherwise → value function reformulation. Correctness by M1.3.

Each case's correctness follows from the corresponding theorem in M1, given that the type system's soundness (M5.2) guarantees the structural prerequisites.

**Why it's load-bearing.** This is the *main theorem* of the entire artifact. It is the formal guarantee that the compiler does what it claims.

**Novelty assessment.** New as a unified statement. It synthesizes known reformulation correctness results under a type-theoretic framework. The novelty is in the *architecture* (type system → selection → correctness), not in any individual reformulation result.

**Difficulty grade.** B — The proof is modular, but ensuring all cases are covered and no gaps exist requires careful formalization.

---

## M6. Solver Interface Abstraction

### M6.1 Solver Capability Model

**Formal statement.** Define a *solver capability signature* as a tuple:

$$
\text{SolverCap} = (\mathcal{O}, \mathcal{C}, \mathcal{V}, \mathcal{F})
$$

where:
- $\mathcal{O}$ = set of objective types the solver accepts (linear, quadratic, general nonlinear)
- $\mathcal{C}$ = set of constraint types (linear, quadratic, conic, complementarity, SOS1/SOS2, indicator)
- $\mathcal{V}$ = variable domains supported (continuous, integer, binary, semi-continuous)
- $\mathcal{F}$ = additional features (callbacks, lazy constraints, warm-starting, multi-objective)

**Examples:**
- Gurobi: $(\{\text{lin}, \text{quad}\}, \{\text{lin}, \text{quad}, \text{SOS1}, \text{SOS2}, \text{indicator}\}, \{\text{cont}, \text{int}, \text{bin}\}, \{\text{lazy}, \text{callback}, \text{warm}\})$
- IPOPT: $(\{\text{general-NLP}\}, \{\text{general-NLP}\}, \{\text{cont}\}, \{\text{warm}\})$
- SCIP: $(\{\text{lin}, \text{quad}\}, \{\text{lin}, \text{quad}, \text{SOS1}, \text{indicator}, \text{complementarity}\}, \{\text{cont}, \text{int}, \text{bin}\}, \{\text{lazy}, \text{callback}\})$

**Why it's load-bearing.** The compiler must match reformulation outputs to solver capabilities. Emitting complementarity constraints to a solver that doesn't support them is a compilation failure.

**Novelty assessment.** Partially new. Solver documentation implicitly specifies capabilities. Formalizing them as a structured signature for compiler use is novel.

**Difficulty grade.** A

---

### M6.2 Compilability and Completeness

**Formal statement.** A bilevel problem $P$ is *compilable* to solver $s$ if there exists a valid reformulation $R \in \rho(\sigma(P))$ such that $R(P) \in \text{Accepts}(s)$, where $\text{Accepts}(s)$ is the set of single-level programs that solver $s$ can accept.

**Theorem (Compilability Decision — New).** Given a problem signature $\sigma$ and a solver capability $\text{SolverCap}(s)$, deciding whether there exists a valid reformulation targeting $s$ is in polynomial time.

*Proof sketch.* The set $\rho(\sigma)$ of valid reformulations is finite and enumerable from $\sigma$ (see M2.2). For each $R \in \rho(\sigma)$, the output type $\tau(R(\sigma))$ can be computed in constant time (each reformulation has known output type). Checking $\tau(R(\sigma)) \subseteq \text{SolverCap}(s)$ is a set inclusion check. The total is $O(|\rho(\sigma)| \cdot |\text{SolverCap}|)$.

**Theorem (Solver Completeness — New).** A solver $s$ is *bilevel-complete for signature class* $\Sigma' \subseteq \Sigma$ if for every $\sigma \in \Sigma'$, $P$ is compilable to $s$. We provide the following classification:

| Signature class | Minimal solver requirements |
|---|---|
| Bilevel LP | MIP solver (for big-M) or NLP solver with complementarity |
| Bilevel convex QP | MIQCP solver or NLP solver |
| Bilevel MIP (bounded integer lower) | MIP solver with lazy constraint callbacks |
| General bilevel nonlinear | Global NLP solver + callback mechanism |

**Why it's load-bearing.** The compiler must report at compile time whether a given problem can be solved with available solvers, and if not, which solver capabilities are missing.

**Novelty assessment.** New. The formalization of compilability as a decision problem and the completeness classification are novel contributions.

**Difficulty grade.** B

---

### M6.3 Reformulation-to-Solver API Translation

**Formal statement.** For each reformulation $R$ and solver $s$, define a *translation function* $T_{R,s}: \text{IR} \to \text{SolverAPI}(s)$ that maps the compiler's intermediate representation to solver-specific API calls.

**Theorem (Translation Correctness).** $T_{R,s}$ is correct if for all bilevel programs $P$:

$$
\text{SolverOutput}(T_{R,s}(R(P))) \text{ is optimal for } R(P)
$$

assuming the solver is exact (finds global optima). This reduces to verifying that $T_{R,s}$ faithfully encodes all constraints and objectives of $R(P)$ in the solver's input format.

**Non-trivial translation issues:**
1. **Big-M calibration:** Translation to MIP requires choosing $M$ values. $M$ too small → cuts off feasible points (incorrect). $M$ too large → weak relaxation (slow). The compiler must either compute valid $M$ from bounds or use indicator constraints where available.
2. **Complementarity encoding:** $\lambda_j g_j = 0$ can be encoded as SOS1 pair, big-M pair, or indicator constraint, depending on solver support.
3. **Bilinear terms:** Products $\lambda_j g_j(x,y)$ in KKT reformulation require McCormick relaxation for MIP solvers or direct NLP handling.

**Why it's load-bearing.** Incorrect translation silently produces wrong answers. Big-M miscalibration is a well-documented source of errors.

**Novelty assessment.** Partially known. Individual translation tricks are known. Systematic treatment as a compiler pass with correctness proofs is partially new.

**Difficulty grade.** B

---

## M7. Additional Load-Bearing Mathematics

### M7.1 Multi-Level Extension

**Formal statement.** A *multi-level* ($k$-level) optimization program has the structure:

$$
\min_{x_1} F_1(x_1, x_2, \ldots, x_k) \quad \text{s.t.} \quad x_2 \in S_2(x_1), \; x_3 \in S_3(x_1, x_2), \; \ldots
$$

where $S_i(x_1, \ldots, x_{i-1}) = \arg\min_{x_i} \{f_i(\ldots) : x_{i+1} \in S_{i+1}(\ldots)\}$ for $i < k$, and the deepest level has no nested optimization.

**Theorem (Reduction to Bilevel — Known).** Any $k$-level program can be reformulated as a bilevel program by iterative application of single-level reformulation from the bottom up. If each level $i$ admits reformulation $R_i$ with error $\epsilon_i$, the total error is bounded by M4.3's chain bound with $k-1$ passes.

**Theorem (Complexity — Known).** $k$-level linear programming is $\Sigma^p_k$-hard (complete for the $k$-th level of the polynomial hierarchy).

**Why it's load-bearing.** The grammar (M5.1) supports multi-level via nesting. The compiler must handle $k > 2$ by recursive application of bilevel reformulation.

**Novelty assessment.** Known.

**Difficulty grade.** A (for $k$-level to bilevel reduction); B (for tracking error through the recursion).

---

### M7.2 Sensitivity Analysis for the Compiled Output

**Formal statement.** After reformulation, the user may want sensitivity information: how does the bilevel optimum change with respect to problem parameters $p$?

**Theorem (Bilevel Sensitivity — Fiacco 1983, Dempe 2002).** Under LICQ at the lower level and strong second-order conditions at both levels, the bilevel optimal value function $V(p)$ is locally Lipschitz in $p$, and directional derivatives can be computed by solving an auxiliary linear program.

**Theorem (Sensitivity Through Reformulation).** If reformulation $R$ preserves the local structure (i.e., the active set does not change in a neighborhood of the parameter), then sensitivity information for $R(P)$ correctly represents sensitivity of the original bilevel program $P$.

**Why it's load-bearing.** Many applications (e.g., Stackelberg pricing) need sensitivity analysis. The compiler must either preserve sensitivity through reformulation or warn when it cannot.

**Novelty assessment.** Known individually. The *preservation of sensitivity through reformulation* statement is a straightforward consequence but has not been formalized as a compiler requirement.

**Difficulty grade.** B

---

### M7.3 Warm-Starting Across Reformulations

**Formal statement.** Given a solution $(x^*, y^*, \lambda^*)$ to the KKT-reformulated problem, can it be used to warm-start a solve of the strong-duality reformulation of a perturbed problem?

**Theorem (Solution Portability — New).** Let $R_1, R_2$ be two exact reformulations of bilevel program $P$, with solution maps $\phi_{R_1 \to P}$ and $\phi_{P \to R_2}$ (mapping solutions between the reformulated and original problems). If both maps are computable and Lipschitz continuous, then warm-starting $R_2$ with $\phi_{P \to R_2}(\phi_{R_1 \to P}(z_1^*))$ produces a feasible starting point for $R_2$ that is within Lipschitz distance of optimal.

**Practical implication.** KKT solutions carry multipliers that can initialize dual variables in the strong-duality reformulation. This enables efficient re-solving when the compiler switches reformulation strategies online.

**Why it's load-bearing.** Real-world use involves solving sequences of related bilevel problems. Without warm-starting theory, the compiler treats each solve independently, losing potentially orders-of-magnitude speedups.

**Novelty assessment.** New. Solution portability across different reformulations has not been studied.

**Difficulty grade.** C — Establishing Lipschitz properties of the composed maps in general requires new analysis.

---

### M7.4 Valid Big-M Computation

**Formal statement.** For the big-M linearization of complementarity constraints $\lambda_j g_j(x,y) = 0$:

$$
\lambda_j \leq M_j^{\lambda}(1 - z_j), \quad -g_j(x,y) \leq M_j^{g} z_j, \quad z_j \in \{0,1\}
$$

**Theorem (Smallest Valid Big-M).** The smallest valid values are:

$$
M_j^{\lambda} = \max_{(x,y,\lambda) \in \mathcal{F}_{KKT}} \lambda_j, \quad M_j^{g} = \max_{(x,y,\lambda) \in \mathcal{F}_{KKT}} (-g_j(x,y))
$$

where $\mathcal{F}_{KKT}$ is the feasible set of the KKT-reformulated problem. Computing these exactly is itself an optimization problem (and generally NP-hard for nonlinear problems).

**Theorem (Bound Propagation for LP case — Fortuny-Amat & McCarl 1981, extended).** For bilevel LP, valid $M$ values can be computed by solving auxiliary LPs (maximizing each variable or expression over the feasible set). These auxiliary LPs are polynomial-time solvable.

**Theorem (Interval Arithmetic Bounds).** For general nonlinear problems, interval arithmetic applied to the constraint functions over variable bounds yields valid (though possibly loose) $M$ values in $O(|AST|)$ time.

**Why it's load-bearing.** Invalid (too small) $M$ values produce incorrect reformulations. The compiler must guarantee validity while keeping $M$ as tight as possible.

**Novelty assessment.** Known techniques; their integration into a compiler with automatic bound propagation is partially new.

**Difficulty grade.** B

---

## Summary Table

| ID | Result | Novel? | Difficulty | Load-Bearing For |
|---|---|---|---|---|
| M1.1 | KKT exactness conditions | Known | A | Deciding when KKT reformulation is valid |
| M1.2 | Strong duality reformulation | Known | A | LP/conic lower-level reformulation |
| M1.3 | Value function reformulation | Known | B | Universal fallback reformulation |
| M1.4 | Integer lower-level reformulations | Partial | B | MIP bilevel problems |
| M1.5 | Optimistic vs. pessimistic | Known | A | Problem semantics |
| M2.1 | Reformulation strength ordering | Partial | B | Choosing best reformulation |
| M2.2 | Structure-dependent selection | New | B | Core compiler decision procedure |
| M2.3 | Complexity of selection | New | C | Compiler architecture (heuristic vs. exact) |
| M2.4 | Approximation bounds | Partial | B | Solution quality guarantees |
| M3.1 | Baseline bilevel complexity | Known | A | Expectation management |
| M3.2 | Compiler pass complexity | Partial | B | Static analysis feasibility |
| M3.3 | Reformulation size blowup | Known | A | Output feasibility |
| M4.1 | Regularization convergence | Known | A | Regularization parameter selection |
| M4.2 | Penalty method convergence | Known | A | Exact penalization applicability |
| M4.3 | Chain error propagation | New | C | Multi-pass correctness |
| M4.4 | Cutting plane convergence | Partial | B | Value function implementation |
| M5.1 | Formal grammar | Partial | B | Compiler input specification |
| M5.2 | Type system for structure | Partial | B | Automatic structure detection |
| M5.3 | Compiler soundness theorem | New | B | Main correctness guarantee |
| M6.1 | Solver capability model | Partial | A | Solver targeting |
| M6.2 | Compilability decision | New | B | Compile-time diagnostics |
| M6.3 | Translation correctness | Partial | B | Code generation |
| M7.1 | Multi-level extension | Known | A/B | Nested problems |
| M7.2 | Sensitivity preservation | Known | B | Post-optimality analysis |
| M7.3 | Warm-starting across reformulations | New | C | Repeated solving performance |
| M7.4 | Valid big-M computation | Partial | B | Correct complementarity encoding |

---

## Novelty Summary

**Genuinely new contributions required (6):**
- M2.2: Formal selection function from signatures to valid reformulations
- M2.3: Complexity of reformulation selection as a decision problem
- M4.3: Compositional error bounds through reformulation chains
- M5.3: Compiler soundness theorem (unified over all reformulations)
- M6.2: Compilability as a formal decision problem with completeness classification
- M7.3: Solution portability / warm-starting across different reformulations

**Difficulty-C results requiring genuinely new ideas (3):**
- M2.3: Reduction proving NP-hardness of optimal reformulation selection
- M4.3: Tight compositional error bounds (especially conditions avoiding exponential blowup)
- M7.3: Lipschitz analysis of composed solution maps across reformulation types

**Primary risk.** The difficulty-C items are not guaranteed to resolve favorably. M2.3 might turn out to be polynomial (weakening the motivation for heuristics) or harder than NP-hard. M4.3's exponential blowup may be inherent for some reformulation chains. M7.3's Lipschitz properties may fail for degenerate problems. The compiler design should be robust to any of these outcomes.
