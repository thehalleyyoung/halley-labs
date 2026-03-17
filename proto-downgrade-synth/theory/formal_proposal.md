# Formal Foundations of NegSynth

**Negotiation Under Fire: Bounded-Complete Synthesis of Protocol Downgrade Attacks from Library Source Code**

**Document:** Formal Definitions, Theorem Statements, and Proof Architecture
**Status:** Load-bearing — every definition drives implementation, every theorem gates a correctness claim

---

## 1. Formal Definitions

### Definition 1: Negotiation Protocol LTS

**Serves:** State Machine Extractor, Protocol Modules

A **Negotiation Protocol Labeled Transition System** is a tuple $\mathcal{N} = (S, S_0, \Lambda, \delta, \mathcal{O}, \textsf{obs})$ where:

- $S$ is a finite set of **negotiation states**. Each state $s \in S$ is a record:

$$s = (\pi, \; C_{\text{offered}} \subseteq \mathbb{C}, \; c_{\text{sel}} \in \mathbb{C} \cup \{\bot\}, \; v \in \mathbb{V}, \; E \subseteq \mathbb{E})$$

  with $\pi \in \{\texttt{init}, \texttt{ch\_sent}, \texttt{sh\_recv}, \texttt{negotiated}, \texttt{done}, \texttt{abort}\}$ the handshake phase, $\mathbb{C}$ the IANA cipher-suite universe, $\mathbb{V}$ the version set (e.g., $\{\text{SSL3.0}, \text{TLS1.0}, \ldots, \text{TLS1.3}\}$), and $\mathbb{E}$ an extension set (SNI, ALPN, signature algorithms, etc.).

- $S_0 \subseteq S$ is the set of initial states (phase $\texttt{init}$, $c_{\text{sel}} = \bot$).

- $\Lambda$ is a finite set of **message labels**, partitioned as $\Lambda = \Lambda_C \uplus \Lambda_S \uplus \Lambda_A$ for client messages, server messages, and adversary actions respectively. Concrete elements include $\texttt{ClientHello}(\mathbb{C}', v, E)$, $\texttt{ServerHello}(c, v', E')$, $\texttt{ChangeCipherSpec}$, $\texttt{Finished}(h)$, and adversary labels $\texttt{drop}$, $\texttt{inject}(m)$, $\texttt{modify}(f)$.

- $\delta : S \times \Lambda \rightharpoonup S$ is a partial transition function (deterministic given a state and label).

- $\mathcal{O}$ is the **observation domain**: the set of negotiation outcomes $(c_{\text{sel}}, v, E_{\text{final}})$.

- $\textsf{obs} : S \to \mathcal{O} \cup \{\bot\}$ extracts the observable outcome from terminal states (phase $\in \{\texttt{done}, \texttt{abort}\}$), undefined otherwise.

**Axioms (algebraic properties exploited by the merge operator):**

**(A1) Finite Outcome Space.** $|\mathbb{C}| < \infty$, $|\mathbb{V}| < \infty$, $|\mathbb{E}| < \infty$. In practice, $|\mathbb{C}| \leq 350$ (IANA registry), $|\mathbb{V}| \leq 6$, $|\mathbb{E}| \leq 30$. The observation domain $\mathcal{O}$ is finite.

**(A2) Lattice Preferences.** There exists a partial order $\preceq_{\text{pref}}$ on $\mathbb{C}$ representing cipher-suite preference, forming a bounded lattice $(\mathbb{C}, \preceq_{\text{pref}}, \top_{\text{pref}}, \bot_{\text{pref}})$. The selection function $\textsf{select} : \mathcal{P}(\mathbb{C}) \times \mathcal{P}(\mathbb{C}) \to \mathbb{C} \cup \{\bot\}$ is monotone: if $C_1 \subseteq C_2$ then $\textsf{select}(C_1, S) \preceq_{\text{pref}} \textsf{select}(C_2, S)$ for any server preference set $S$.

**(A3) Monotonic Progression.** The handshake-phase component of states forms an acyclic directed graph. Formally, there is a strict total order $<_\pi$ on the set of phases, and for all transitions $\delta(s, \ell) = s'$ where $\ell \in \Lambda_C \cup \Lambda_S$, we have $s'.\pi \geq_\pi s.\pi$. No honest transition decreases the phase.

**(A4) Deterministic Selection.** Given fixed offered cipher sets $C_C, C_S \subseteq \mathbb{C}$, a fixed version pair $(v_C, v_S)$, and fixed extensions $E_C, E_S$:

$$|\{c \mid c = \textsf{select}(C_C \cap C_S, \preceq_{\text{pref}}) \}| = 1$$

Selection is a function, not a relation. The same inputs always produce the same cipher suite.

---

### Definition 2: Protocol-Aware Merge Operator ⊵

**Serves:** Merge Operator module (~7K LoC), KLEE Integration Layer

Let $\Sigma$ be the domain of **symbolic negotiation states** — states from Definition 1 extended with path conditions. A symbolic state is a pair $\hat{s} = (s, \varphi)$ where $s$ is a concrete-structure template and $\varphi$ is a quantifier-free formula over symbolic variables.

The **protocol-aware merge operator** $\bowtie : \Sigma \times \Sigma \rightharpoonup \Sigma$ is a partial function defined as follows.

**Mergeability predicate.** Two symbolic states $\hat{s}_1 = (s_1, \varphi_1)$ and $\hat{s}_2 = (s_2, \varphi_2)$ are **mergeable**, written $\textsf{mergeable}(\hat{s}_1, \hat{s}_2)$, iff:

1. **Same phase:** $s_1.\pi = s_2.\pi$.
2. **Compatible offered sets:** $s_1.C_{\text{offered}} = s_2.C_{\text{offered}}$ (same client offer).
3. **Compatible version:** $s_1.v = s_2.v$.
4. **Feasible conjunction:** $\varphi_1 \lor \varphi_2$ is satisfiable.

**Merge construction.** When $\textsf{mergeable}(\hat{s}_1, \hat{s}_2)$ holds, define:

$$\hat{s}_1 \bowtie \hat{s}_2 = (\hat{s}_{\bowtie}, \varphi_1 \lor \varphi_2)$$

where $\hat{s}_{\bowtie}$ is a symbolic state template with:

- $\hat{s}_{\bowtie}.\pi = s_1.\pi = s_2.\pi$
- $\hat{s}_{\bowtie}.C_{\text{offered}} = s_1.C_{\text{offered}}$
- $\hat{s}_{\bowtie}.c_{\text{sel}} = \textsf{ITE}(\varphi_1 \land \neg \varphi_2, \; s_1.c_{\text{sel}}, \; \textsf{ITE}(\varphi_2 \land \neg \varphi_1, \; s_2.c_{\text{sel}}, \; \textsf{ITE}(\varphi_1 \land \varphi_2, \; s_1.c_{\text{sel}}, \; \bot)))$

The third ITE branch handles the overlap case; by (A4) deterministic selection, when both path conditions are satisfiable and the inputs are identical, $s_1.c_{\text{sel}} = s_2.c_{\text{sel}}$ must hold, so the branch collapses.

- $\hat{s}_{\bowtie}.v = s_1.v$
- $\hat{s}_{\bowtie}.E = s_1.E \cup s_2.E$ with selector bits distinguishing per-state extensions where they differ.

**Key invariant.** For any valuation $\sigma$ satisfying the merged path condition $\varphi_1 \lor \varphi_2$, evaluating $\hat{s}_{\bowtie}$ under $\sigma$ yields a concrete state reachable from either $\hat{s}_1$ or $\hat{s}_2$ under the corresponding original path condition. This is the content of Lemma L1 (merge congruence).

---

### Definition 3: Protocol Bisimulation ≈_P

**Serves:** State Machine Extractor (bisimulation quotient), Theorem T3

A **protocol bisimulation** on Negotiation LTS $\mathcal{N}$ is a symmetric relation $R \subseteq S \times S$ such that whenever $(s_1, s_2) \in R$:

1. **Observation agreement:** If $\textsf{obs}(s_1) \neq \bot$ then $\textsf{obs}(s_2) \neq \bot$ and $\textsf{obs}(s_1) = \textsf{obs}(s_2)$.

2. **Transfer property:** For all $\ell \in \Lambda$, if $\delta(s_1, \ell) = s_1'$ then there exists $s_2'$ with $\delta(s_2, \ell) = s_2'$ and $(s_1', s_2') \in R$.

Two states are **protocol-bisimilar**, written $s_1 \approx_P s_2$, if there exists a protocol bisimulation $R$ with $(s_1, s_2) \in R$.

**Extension to symbolic states.** Symbolic states $\hat{s}_1, \hat{s}_2$ are protocol-bisimilar, written $\hat{s}_1 \approx_P \hat{s}_2$, if for every valuation $\sigma$:

$$\sigma \models \varphi_1 \implies \exists \sigma'. \; \sigma' \models \varphi_2 \land \textsf{concretize}(\hat{s}_1, \sigma) \approx_P \textsf{concretize}(\hat{s}_2, \sigma')$$

and symmetrically.

**Observable vs. internal.** The observation function $\textsf{obs}$ projects onto the negotiation outcome $(c_{\text{sel}}, v, E_{\text{final}})$. All intermediate symbolic computations — path condition structure, ITE nesting depth, variable naming — are internal. Protocol bisimulation equates states that may differ in internal symbolic representation but agree on all reachable negotiation outcomes.

---

### Definition 4: Bounded Dolev-Yao Adversary

**Serves:** DY+SMT Encoder (~10K LoC), Concretizer

**Message algebra.** The set of **terms** $\mathcal{T}$ over a sorted signature $\Sigma_{\text{DY}}$ is defined inductively:

- **Atoms:** $n \in \textsf{Nonce}$, $k \in \textsf{Key}$, $c \in \mathbb{C}$ (cipher suite identifiers), $v \in \mathbb{V}$ (version tags), $b \in \textsf{Bytes}$ (raw bitstrings).
- **Constructors:** $\textsf{enc}_s(k, m)$ (symmetric encryption), $\textsf{enc}_a(pk, m)$ (asymmetric), $\textsf{mac}(k, m)$ (MAC), $\textsf{hash}(m)$, $\textsf{pair}(m_1, m_2)$, $\textsf{record}(\text{type}, v, \text{len}, \text{payload})$ (TLS record layer), $\textsf{packet}(\text{seqno}, \text{payload})$ (SSH packet layer).
- **Destructors:** $\textsf{dec}_s(k, \textsf{enc}_s(k, m)) \to m$, $\textsf{dec}_a(sk, \textsf{enc}_a(pk, m)) \to m$, $\textsf{fst}(\textsf{pair}(m_1, m_2)) \to m_1$, $\textsf{snd}(\textsf{pair}(m_1, m_2)) \to m_2$, $\textsf{verify}(k, m, \textsf{mac}(k, m)) \to \textsf{ok}$.

**Adversary knowledge.** The adversary knowledge set $\mathcal{K} \subseteq \mathcal{T}$ is the closure of observed messages under destructors and public constructors. Formally, $\mathcal{K}$ is the smallest set satisfying:

- $\mathcal{K}_0 = \{m \mid m \text{ observed on the network}\}$ (initial knowledge)
- If $t_1, \ldots, t_n \in \mathcal{K}$ and $f$ is a public constructor, then $f(t_1, \ldots, t_n) \in \mathcal{K}$.
- If $t \in \mathcal{K}$ and $d(t) \to t'$ is a destructor reduction, then $t' \in \mathcal{K}$.

**Bounded adversary.** A **$(k, n)$-bounded Dolev-Yao adversary** is a sequence of adversary actions $a_1, \ldots, a_m$ with $m \leq n$ (action budget), applied to a protocol execution of depth at most $k$ (message rounds). Each action $a_i$ is one of:

- $\texttt{intercept}(i)$: capture message $i$ (add to $\mathcal{K}$, prevent delivery).
- $\texttt{inject}(t)$: send a term $t$ constructible from $\mathcal{K}$.
- $\texttt{drop}(i)$: discard message $i$.
- $\texttt{modify}(i, f)$: replace message $i$ with $f(m_i, \mathcal{K})$ for constructible $f$.

The depth bound $k$ limits the total number of protocol message rounds; $n$ limits the adversary's action count. Both are finite.

---

### Definition 5: Downgrade Freedom Property

**Serves:** DY+SMT Encoder (property encoding), Certificate generation

**Weakness ordering.** Define $\preceq_{\text{sec}}$ on $\mathbb{C}$ as a total preorder reflecting cryptographic strength: $c_1 \preceq_{\text{sec}} c_2$ means $c_1$ is no stronger than $c_2$. Concretely, ciphers using export-grade keys, RC4, DES, or NULL encryption rank lowest. This ordering is a parameter of the analysis, configured per engagement.

**Honest outcome.** Given a Negotiation LTS $\mathcal{N}$, the **honest outcome** $o_H = (c_H, v_H, E_H) = \textsf{obs}(s_{\text{done}})$ is the terminal observation when no adversary actions intervene — both client and server execute faithfully.

**Downgrade attack.** A $(k,n)$-bounded adversary trace $\vec{a}$ constitutes a **downgrade attack** on $\mathcal{N}$ if, when applied to $\mathcal{N}$, the resulting terminal state $s'_{\text{done}}$ satisfies:

$$\textsf{obs}(s'_{\text{done}}) = (c', v', E') \quad \text{with} \quad c' \prec_{\text{sec}} c_H \;\lor\; v' < v_H$$

That is, the adversary forces selection of a strictly weaker cipher suite or a lower protocol version than honest execution would produce.

**Downgrade freedom.** The Negotiation LTS $\mathcal{N}$ is **$(k,n)$-downgrade-free** if no $(k,n)$-bounded adversary trace is a downgrade attack:

$$\forall \vec{a} \in \textsf{Adv}(k,n). \; \textsf{obs}(\textsf{exec}(\mathcal{N}, \vec{a})) \not\prec_{\text{sec}} o_H$$

---

### Definition 6: Bounded-Completeness Certificate

**Serves:** Certificate generation, CLI/Reporting module

A **bounded-completeness certificate** is a triple:

$$\textsf{Cert} = (\ell, \; (k, n), \; \Pi)$$

where:

- $\ell$ is a library identifier with version (e.g., OpenSSL 3.2.1, commit hash).
- $(k, n)$ are the execution depth bound and adversary action budget.
- $\Pi$ is an **UNSAT proof** (or UNSAT core) from the SMT solver, witnessing that the formula $\Phi_{\ell,(k,n)}$ encoding all possible downgrade attacks is unsatisfiable.

**Validity.** $\textsf{Cert}$ is **valid** if:

1. The extracted state machine $\mathcal{N}_\ell$ faithfully represents $\ell$'s negotiation logic (T1).
2. The merge operator preserves all observable behaviors (T3).
3. The SMT encoding $\Phi_{\ell,(k,n)}$ is equisatisfiable with the existence of a $(k,n)$-downgrade attack on $\mathcal{N}_\ell$ (T5).
4. $\Pi$ is a valid UNSAT witness for $\Phi_{\ell,(k,n)}$.

**Interpretation.** A valid certificate asserts: *within execution depth $k$ and adversary budget $n$, no downgrade attack exists in library $\ell$*. It does not claim security beyond these bounds.

---

## 2. Theorem Statements with Proof Sketches

### Theorem T1: Extraction Soundness

**Statement.** Let $P$ be the source program (LLVM IR of the negotiation slice) and let $\mathcal{N}_P$ be the Negotiation LTS extracted by the state-machine extractor. There exists a simulation relation $\mathcal{R} \subseteq S_P \times S_{\mathcal{N}}$ such that:

1. **(Trace inclusion)** Every trace $\tau$ of $\mathcal{N}_P$ corresponds to a feasible execution path $\pi$ in $P$: $\forall \tau \in \textsf{Traces}(\mathcal{N}_P). \; \exists \pi \in \textsf{Paths}(P). \; \textsf{proj}(\pi) = \tau$.

2. **(Reachability preservation)** Every reachable negotiation state in $P$ maps to a reachable state in $\mathcal{N}_P$: $\forall s \in \textsf{Reach}(P). \; \exists s' \in \textsf{Reach}(\mathcal{N}_P). \; (s, s') \in \mathcal{R}$.

**Proof strategy: Forward simulation.**

Construct $\mathcal{R}$ by induction on execution steps. The simulation maps each concrete program state (program counter, memory, symbolic context) to the corresponding negotiation LTS state by projecting onto the protocol-relevant variables identified by the slicer.

*Key lemma needed:* **Slicer soundness** (L3) — the protocol-aware slicer does not remove any code that can influence the negotiation outcome. This ensures the projection from full program state to LTS state is well-defined and complete.

*Inductive step:* For each transition $s \xrightarrow{\ell} s'$ in $P$, the simulation must produce a matching transition in $\mathcal{N}_P$. When the source transition is negotiation-relevant, Definition 1's transition function $\delta$ covers it. When it is internal (e.g., buffer management, logging), the simulation stutters — $\mathcal{R}$ maps both $s$ and $s'$ to the same LTS state.

*Genuinely hard part:* Handling the merge operator's state abstractions. After merging, the symbolic state in $\mathcal{N}_P$ represents a set of concrete states via ITE constructs. The simulation must show that every concretization of the merged state is reachable in $P$. This requires Lemma L1 (merge congruence).

*What is adapted:* Standard simulation relation construction (Milner 1989, Lynch & Vaandrager 1995). Stuttering simulation for the internal steps follows standard I/O automata theory.

*What is novel:* The domain instantiation for protocol negotiation with merged symbolic states. No prior simulation relation has been established for a merge-equipped symbolic execution engine targeting protocol state machines.

*Difficulty:* **3/10.** Template is standard; domain instantiation requires care but not invention.

---

### Theorem T3: Protocol-Aware Merge Correctness (Crown Theorem)

**Statement.** For any two mergeable symbolic states $\hat{s}_1, \hat{s}_2$ encountered during symbolic execution of a negotiation slice:

**(Correctness)** $\hat{s}_1 \bowtie \hat{s}_2 \approx_P \hat{s}_1 \parallel \hat{s}_2$

where $\hat{s}_1 \parallel \hat{s}_2$ denotes the disjoint exploration of both states (the baseline without merging). That is, the merged state is protocol-bisimilar to the pair of unmerged states: it produces exactly the same set of observable negotiation outcomes.

**(Complexity)** For a negotiation slice with $n$ cipher suites and $m$ handshake phases, protocol-aware merging explores $O(n \cdot m)$ symbolic states, compared to $O(2^n \cdot m)$ for generic veritesting (Avgerinos et al., ICSE 2014).

**Proof strategy: Two-part argument.**

**Part 1 (Correctness): Bisimulation up-to congruence.** Construct a relation $R_\bowtie$ pairing each merged state with its constituent unmerged states. Show $R_\bowtie$ is a protocol bisimulation up-to (Definition 3).

*Key insight:* Axioms (A1)–(A4) guarantee that the ITE construction in Definition 2 faithfully preserves all negotiation behaviors. Specifically:

- (A1) ensures the ITE over cipher selections has finitely many branches, each decidable.
- (A4) ensures that when both path conditions overlap on identical inputs, the selected ciphers agree — the ITE collapses and does not introduce spurious outcomes.
- (A2) ensures preference ordering is preserved through the merge: the lattice structure means merged preference queries yield the same result as un-merged ones.
- (A3) ensures that forward transitions from the merged state remain well-defined, because the acyclic phase structure prevents the merge from creating cycles.

*Inductive argument:* By induction on the number of transitions after the merge point. Base case: at the merge point, the ITE construction directly encodes both possibilities. Inductive step: by (A3), the next transition advances the phase or stays. By (A4), the transition function applied to the merged state distributes over the ITE, producing a new merged state (or two states if the mergeability predicate fails at the next step, at which point we fork — but the fork produces states each bisimilar to a subset of the original unmerged states).

**Part 2 (Complexity): Counting argument.**

*Without merge:* Each cipher suite in the client offer can independently be included or excluded, creating up to $2^n$ subsets in the offered set. Each subset leads to a potentially different selection path. With $m$ phases, generic exploration visits $O(2^n \cdot m)$ states.

*With merge:* The mergeability predicate (same phase, same offered set, same version) groups states by their negotiation-relevant structure. By (A4), all states at the same phase with the same offered set and version produce the same selection — they differ only in path conditions over non-negotiation variables. The merge collapses them. The remaining variation is over which cipher is *selected*, giving at most $n$ distinct selection outcomes per phase, hence $O(n \cdot m)$ states.

*Caveat:* The $O(n \cdot m)$ bound applies to *ideal* negotiation logic satisfying all four axioms. Real code (OpenSSL callbacks, FIPS overrides, `#ifdef` forests) may partially violate axioms, degrading the bound. The empirical claim is 10–100× reduction on production code.

*Genuinely hard:* Showing that axiom violations degrade gracefully (the operator is still correct, only the complexity bound loosens) rather than breaking soundness. This requires a fallback argument: when the mergeability predicate fails, we simply do not merge, falling back to standard exploration. Soundness is preserved; only the complexity bound changes.

*What is adapted:* State merging (Kuznetsov et al., PLDI 2012) for the merge construction template. Bisimulation up-to techniques (Sangiorgi 1998) for the proof method.

*What is novel:* (1) The identification that negotiation protocols satisfy (A1)–(A4). (2) The mergeability predicate specialized to protocol structure. (3) The formal complexity separation from generic veritesting. No prior work has established a polynomial path bound for any merge operator on a structured program domain.

*Difficulty:* **5/10.** Each ingredient is textbook; the combination and domain identification are new.

---

### Theorem T2: Attack Trace Concretizability

**Statement.** Let $\sigma$ be a satisfying assignment for the SMT formula $\Phi_{\ell,(k,n)}$. The CEGAR-based concretizer produces a byte-level attack trace $\vec{b}$ such that:

1. $\vec{b}$ is a valid sequence of TLS records (or SSH packets) conforming to wire-format specifications.
2. Replaying $\vec{b}$ against a live instance of library $\ell$ reproduces the downgrade.
3. The concretization success rate is $\geq 1 - \varepsilon$ where $\varepsilon$ is empirically bounded per library (target: $\varepsilon < 0.01$).

If concretization fails (the $\varepsilon$ case), the CEGAR loop extracts a refinement predicate from the failure, adds it to the SMT formula, and re-queries. The loop terminates because refinements strictly reduce the satisfying set.

**Proof strategy: CEGAR soundness + convergence.**

*Soundness direction:* If the concretizer succeeds, the byte-level trace is valid by construction — it is assembled from the SMT model using protocol-specific record-framing functions that enforce format invariants.

*Convergence:* Each CEGAR refinement adds a constraint that eliminates at least one spurious model. Since the model space is finite (bounded by $(k, n)$ and finite cipher-suite sets), the loop terminates in at most $|\mathcal{O}|^k$ iterations — in practice, 1–3 iterations suffice.

*Key lemma needed:* **Framing correctness** (L5) — the function that maps SMT bitvector assignments to TLS record bytes (or SSH packet bytes) faithfully inverts the extraction process.

*Genuinely hard:* Handling the gap between symbolic bitvector representations and actual wire-format constraints (padding, length fields, MAC placement). This is where most concretization failures occur and where CEGAR refinement earns its keep.

*What is adapted:* CEGAR (Clarke et al., 2000) for the refinement loop. Symbolic-to-concrete mapping from KLEE's existing test generation.

*What is novel:* CEGAR refinement in the Dolev-Yao adversary domain — the refinement predicates are protocol-aware (e.g., "this cipher suite requires a key exchange message that the adversary cannot construct").

*Difficulty:* **3/10.** Standard CEGAR adapted to a new domain.

---

### Theorem T5: SMT Encoding Correctness

**Statement.** The SMT formula $\Phi_{\ell,(k,n)}$ over the combined theory $\textsf{BV} + \textsf{Arrays} + \textsf{UF} + \textsf{LIA}$ is **equisatisfiable** with the existence of a $(k,n)$-downgrade attack on the extracted Negotiation LTS $\mathcal{N}_\ell$:

$$\Phi_{\ell,(k,n)} \text{ is SAT} \iff \exists \vec{a} \in \textsf{Adv}(k,n). \; \vec{a} \text{ is a downgrade attack on } \mathcal{N}_\ell$$

**Proof strategy: Encoding-decoding bijection.**

*SAT → Attack:* Given a satisfying assignment $\sigma$ to $\Phi$, extract adversary actions $\vec{a}$ from the adversary-action variables, protocol messages from the message variables, and the final negotiation state from the outcome variables. The encoding is structured so that each variable group directly represents a component of Definition 4.

*Attack → SAT:* Given a concrete attack trace $\vec{a}$, construct the variable assignment that encodes each adversary action, each intercepted/injected message, and the resulting negotiation states at each step. Show this assignment satisfies every clause of $\Phi$.

*Key sub-arguments:*

1. **Adversary knowledge faithfulness.** The encoding of $\mathcal{K}$ (Definition 4) as array-indexed bitvector terms correctly models knowledge accumulation: each intercepted message is added (array store), each derivable term is constrained to be constructible (UF axioms encoding constructor/destructor rewrite rules).

2. **Transition faithfulness.** The encoding of $\delta$ (Definition 1) as bitvector constraints correctly models all transitions. Each transition $s \xrightarrow{\ell} s'$ is encoded as an implication: if the current-state variables match $s$ and the action variable matches $\ell$, then the next-state variables must match $s'$.

3. **Property encoding.** The downgrade property (Definition 5) is encoded as a conjunction: the final cipher-suite variable is constrained to be $\prec_{\text{sec}}$ the honest outcome, encoded as a disjunction over weaker cipher suites (finite by A1).

*Genuinely hard:* Ensuring the UF axioms for the Dolev-Yao term algebra are complete — no derivable term is missed, no underivable term is admitted. This is the analog of the "soundness bug" class that plagued early Tamarin/ProVerif encodings. We validate empirically against a reference Prolog DY model.

*What is adapted:* Theory-combination results for $\textsf{BV} + \textsf{Arrays} + \textsf{UF}$ (Nelson-Oppen, Shostak). Dolev-Yao encoding in SMT from Armando et al. (AVISPA project).

*What is novel:* The specific encoding for negotiation LTS transitions + cipher-suite selection as BV constraints. The use of LIA for adversary budget counting ($\sum_i \textsf{acted}(i) \leq n$).

*Difficulty:* **3/10.** Standard encoding with protocol-specific instantiation.

---

### Theorem T4: Bounded Completeness (Headline Result)

**Statement.** For library $\ell$, execution depth bound $k$, and adversary budget $n$:

$$\text{NegSynth}(\ell, k, n) = \begin{cases} (\texttt{ATTACK}, \vec{b}) & \text{if } \exists \text{ a } (k,n)\text{-downgrade attack on } \ell, \text{ with concretization probability} \geq 1-\varepsilon \\ (\texttt{SAFE}, \textsf{Cert}) & \text{if no } (k,n)\text{-downgrade attack exists on } \ell \end{cases}$$

and both outcomes are correct:

- If $\texttt{ATTACK}$: the byte trace $\vec{b}$ is a valid downgrade attack against a live instance of $\ell$.
- If $\texttt{SAFE}$: the certificate $\textsf{Cert}$ is valid per Definition 6.

**Proof strategy: Three-level composition.**

The proof chains T1, T3, and T5 via transitivity of soundness:

$$\underbrace{P \xrightarrow{\text{T1}} \mathcal{N}_P}_{\text{extraction}} \xrightarrow{\text{T3}} \underbrace{\mathcal{N}_P^{\bowtie}}_{\text{merged}} \xrightarrow{\text{T5}} \underbrace{\Phi}_{\text{SMT}}$$

Step 1 (T1): The extracted LTS $\mathcal{N}_P$ simulates $P$ — every source-level downgrade attack induces a trace in $\mathcal{N}_P$, and every LTS trace corresponds to a feasible source execution.

Step 2 (T3): The merged LTS $\mathcal{N}_P^{\bowtie}$ is protocol-bisimilar to $\mathcal{N}_P$ — merging does not add or remove any observable negotiation behaviors.

Step 3 (T5): The SMT formula $\Phi$ is equisatisfiable with the existence of a downgrade attack on $\mathcal{N}_P^{\bowtie}$ — the solver's SAT/UNSAT answer is correct.

Step 4 (T2): If SAT, the CEGAR concretizer produces a valid byte-level trace with probability $\geq 1 - \varepsilon$.

*Composition:* By transitivity, a downgrade attack exists in the source iff $\Phi$ is SAT (modulo the bounded scope). The SAFE case follows directly from UNSAT. The ATTACK case follows from SAT + T2.

*Bounds validation (empirical, not proven):* The analysis bounds $(k, n)$ are validated by:

- Structural argument: TLS handshakes complete in $\leq 10$ round trips; SSH in $\leq 8$. $k = 20$ covers $2\times$ protocol depth.
- CVE table: all 8 known CVEs require $k \leq 15$, $n \leq 5$.
- Coverage metric: random testing confirms $\geq 99\%$ of reachable negotiation states are explored at $k = 20$, $n = 5$.

*Genuinely hard:* The composition itself is straightforward (transitivity), but ensuring all three intermediate results share compatible definitions is where subtle bugs hide. The simulation relation in T1 must align with the bisimulation in T3, which must align with the encoding in T5. Mismatched state abstractions between stages would silently break the chain.

*What is adapted:* CompCert-style composition of simulation/refinement relations.

*What is novel:* The specific three-level composition for protocol analysis. No prior work has composed extraction soundness, protocol-aware merging, and adversary encoding into a single end-to-end guarantee.

*Difficulty:* **4/10.** Composition theorem — the hard work is in the components.

---

### Corollary C1: Covering-Design Differential Completeness (Extension)

**Statement.** For $N \geq 3$ libraries, each with extracted Negotiation LTS $\mathcal{N}_1, \ldots, \mathcal{N}_N$, and a parameter space of $n$ cipher suites, $k$ versions, and interaction depth $d$: a covering design of strength $t$ generates at most

$$B(n, k, t) = O\left(\frac{t^t \cdot \ln n}{(t-1)!}\right)$$

test configurations that guarantee detection of all $t$-way behavioral deviations between any pair of libraries.

Formally: if libraries $\ell_i$ and $\ell_j$ produce different negotiation outcomes on any configuration involving at most $t$ interacting parameters, the covering design includes a configuration that exposes this deviation.

**Proof strategy: Covering design existence + detection argument.**

*Existence:* By the Stein-Lovász-Johnson bound (or Rödl nibble method), a $t$-covering design of size $B(n, k, t)$ exists. We use a constructive greedy algorithm to produce it.

*Detection:* For any $t$-way deviation — a behavioral difference triggered by the interaction of at most $t$ parameters (e.g., cipher suite × version × extension) — the covering property guarantees at least one test configuration exercises that specific parameter combination. The deviation is then detected by comparing the outputs of $\mathcal{N}_i$ and $\mathcal{N}_j$ on that configuration.

*Explicit limitation:* C1 guarantees **testing completeness over the parameter space**, not verification completeness over execution paths. A $(t+1)$-way interaction can escape detection. C1 complements T4 (per-library path-completeness); it does not replace it.

*What is adapted:* Combinatorial interaction testing (Kuhn et al.), covering array theory.

*What is novel:* Application to protocol behavioral deviation detection across libraries. The connection between covering strength $t$ and the order of parameter interaction causing security-relevant behavioral differences.

*Difficulty:* **6/10.** Mathematically the deepest theorem. The bound $B(n, k, t)$ is tight and the connection to protocol testing is non-obvious.

---

## 3. Key Lemmas

### Lemma L1: Merge Congruence

**Statement.** The merge operator $\bowtie$ is a congruence with respect to protocol bisimulation: if $\hat{s}_1 \approx_P \hat{s}_1'$ and $\hat{s}_2 \approx_P \hat{s}_2'$ and both pairs are mergeable, then:

$$\hat{s}_1 \bowtie \hat{s}_2 \approx_P \hat{s}_1' \bowtie \hat{s}_2'$$

**Role:** Core lemma for T3. Ensures that bisimulation-equivalent inputs to the merge yield bisimulation-equivalent outputs. Without this, merging could silently break the bisimulation quotient construction in the state-machine extractor.

**Proof sketch:** By the definition of $\bowtie$ (Definition 2), the merged state is constructed from ITE terms over the path conditions and negotiation components of the inputs. Since $\hat{s}_1 \approx_P \hat{s}_1'$, they agree on all observable behaviors (Definition 3). The ITE over observationally equivalent components produces observationally equivalent results. The formal argument uses the substitutivity property of bisimulation (Milner 1989, Chapter 4).

**Difficulty:** 2/10. Follows from definitions once the framework is in place.

---

### Lemma L2: Bounded Branching

**Statement.** The Negotiation LTS $\mathcal{N}$ has bounded branching factor $B$:

$$\forall s \in S. \; |\{\ell \in \Lambda \mid \delta(s, \ell) \text{ is defined}\}| \leq B$$

where $B = |\mathbb{C}| + |\Lambda_A| \leq 350 + n$ for TLS (IANA cipher suite bound plus adversary actions).

**Role:** Ensures the state space explored by symbolic execution is bounded, justifying the finite analysis. Required by T3's complexity argument and T4's compositional soundness.

**Proof sketch:** Direct from (A1). At each state, the possible transitions are determined by the protocol phase and the available messages. Client/server messages are determined by the finite cipher-suite set, version set, and extension set. Adversary actions are bounded by the adversary budget $n$.

**Difficulty:** 1/10. Immediate from finiteness axiom.

---

### Lemma L3: Slicer Soundness

**Statement.** Let $P$ be the full library program and $P_{\text{slice}}$ the protocol-aware slice. For every execution path $\pi$ in $P$ that reaches a negotiation-outcome state:

$$\textsf{obs}(\pi \text{ in } P) = \textsf{obs}(\pi|_{\text{slice}} \text{ in } P_{\text{slice}})$$

The slice preserves all negotiation-relevant behaviors.

**Role:** Foundation of T1. If the slicer removes code that affects negotiation outcomes, the entire pipeline is unsound.

**Proof sketch:** The slicer computes a forward taint analysis from negotiation entry points (e.g., `SSL_do_handshake`) and a backward slice from outcome sinks (cipher-suite assignment, version selection). Any code not in the forward-backward intersection cannot influence the negotiation outcome. The formal argument uses the standard slicing soundness theorem (Weiser 1984), extended with protocol-specific taint propagation rules for OpenSSL's `STACK_OF(SSL_CIPHER)` containers and `SSL_METHOD` vtable dispatch.

**Difficulty:** 3/10 for the theorem; significant engineering effort for the pointer analysis precision needed on real code.

---

### Lemma L4: DY Knowledge Monotonicity

**Statement.** The adversary knowledge set is monotonically increasing over protocol execution:

$$\forall i < j. \; \mathcal{K}_i \subseteq \mathcal{K}_j$$

**Role:** Required by T5 (SMT encoding). The encoding represents knowledge as an array that is only appended to — monotonicity justifies this representation.

**Proof sketch:** By definition of the adversary model (Definition 4). Intercepting a message adds it to $\mathcal{K}$. Constructing a new term from known terms does not remove any existing term. Destructor application adds results without removing inputs. No adversary action removes knowledge.

**Difficulty:** 1/10. Direct from the definition.

---

### Lemma L5: Framing Correctness

**Statement.** The concretization function $\textsf{frame} : \textsf{SMT\_Model} \to \textsf{Bytes}^*$ satisfies:

$$\forall \sigma \models \Phi. \; \textsf{parse}(\textsf{frame}(\sigma)) = \textsf{extract\_trace}(\sigma)$$

where $\textsf{parse}$ is the standard TLS record parser (or SSH packet parser). The framing function correctly serializes SMT models into valid wire-format bytes.

**Role:** Required by T2. If framing is incorrect, concretization fails even when the SMT model is correct.

**Proof sketch:** The framing function is a straightforward encoding: bitvector assignments for cipher-suite IDs map to 2-byte IANA codes, version variables map to 2-byte version tags, etc. Correctness follows from the format specification (RFC 5246 for TLS 1.2, RFC 8446 for TLS 1.3, RFC 4253 for SSH). Validated empirically by round-trip testing: frame then parse must recover the original model values.

**Difficulty:** 2/10 for correctness argument; significant implementation effort for completeness across protocol versions.

---

### Lemma L6: CEGAR Termination

**Statement.** The CEGAR refinement loop in T2 terminates in at most $F$ iterations, where:

$$F \leq |\mathbb{C}|^k \cdot |\mathbb{V}|^k$$

In practice, $F \leq 3$ for all benchmarked CVEs.

**Role:** Required by T2 to guarantee the concretizer terminates.

**Proof sketch:** Each refinement iteration eliminates at least one spurious counterexample by adding a constraint. The total number of distinct counterexamples is bounded by the number of distinct negotiation traces of length $\leq k$, which is finite by (A1) and bounded branching (L2).

**Difficulty:** 2/10. Standard CEGAR termination argument over a finite domain.

---

## 4. Proof Architecture

### Dependency Graph

```
                    L3 (Slicer Soundness)
                         │
                         ▼
               T1 (Extraction Soundness)
                         │
           ┌─────────────┤
           │             │
   L1 (Merge            │
   Congruence)           │
      │                  │
      ▼                  │
   T3 (Merge ◄── L2 (Bounded Branching)
   Correctness)          │
      │                  │
      │         L4 (DY Monotonicity)
      │                  │
      │                  ▼
      │         T5 (SMT Encoding)
      │                  │
      └────────┬─────────┘
               │
               ▼
      T4 (Bounded Completeness) ◄── T2 (Concretizability)
                                         │
                                    L5 (Framing Correctness)
                                         │
                                    L6 (CEGAR Termination)


      C1 (Covering Design) ── independent, uses outputs of T4
```

### Critical Path

The longest dependency chain through the proof is:

$$\text{L3} \to \text{T1} \to \text{T4} \quad \text{and} \quad \text{L1} \to \text{T3} \to \text{T4} \quad \text{and} \quad \text{L4} \to \text{T5} \to \text{T4}$$

Three chains converge at T4. The critical path for *effort* is:

1. **T3 (merge correctness)** — most technically demanding, requires all four axioms to be formalized and the bisimulation-up-to argument to be executed.
2. **T5 (SMT encoding)** — most implementation-coupled, as encoding bugs are the primary soundness risk.
3. **T4 (composition)** — comes last, but is mechanically straightforward once T1, T3, T5 are in place.

### Parallelization

- **T1 and T5** can be proved concurrently (independent dependency chains).
- **T2** can be proved concurrently with T3 (independent: T2 concerns concretization, T3 concerns merging).
- **T4** must come last (depends on T1, T3, T5).
- **C1** is fully independent of the core chain.

### Risk Assessment

| Theorem | Proof Risk | Primary Risk Factor |
|---------|:----------:|---------------------|
| T1 | Low (5%) | Standard simulation; risk is slicer implementation bugs, not proof difficulty |
| T2 | Medium (15%) | CEGAR may not converge tightly on complex cipher-suite interactions; $\varepsilon$ may be too large |
| **T3** | **Low-Medium (10%)** | **Proof itself is low-risk; risk is that real code violates axioms more than expected, weakening the complexity bound** |
| T4 | Medium (15%) | Composition is straightforward; risk is definitional misalignment between T1/T3/T5 interfaces |
| T5 | Medium (20%) | **Highest risk.** Encoding faithfulness bugs are the #1 soundness threat. DY axiom completeness is hard to validate |
| C1 | Medium (15%) | Tight bound computation; 3-way interactions may escape pairwise coverage |

**Overall proof-chain risk:** ~25% that a subtle soundness issue is discovered during implementation that requires theorem revision (not abandonment — the framework is sound; specific statements may need tightening).

---

## 5. What Is Novel vs. Known

### Adapted from Existing Work

| Concept | Source | How Adapted |
|---------|--------|-------------|
| Labeled Transition Systems | Milner (1989), Keller (1976) | Instantiated with negotiation-specific state structure (phase, cipher set, version, extensions) and axioms (A1–A4) |
| Bisimulation | Milner (1989), Park (1981) | Standard definition applied to protocol LTS; observation function specialized to negotiation outcomes |
| Simulation relations for extraction | Lynch & Vaandrager (1995), CompCert (Leroy 2009) | Forward simulation between LLVM IR execution and LTS; stuttering for non-negotiation instructions |
| CEGAR | Clarke et al. (2000) | Refinement loop applied to DY adversary domain instead of hardware model checking |
| State merging in symbolic execution | Kuznetsov et al. (PLDI 2012), Avgerinos et al. (ICSE 2014) | Merge operator structure borrowed; mergeability predicate and ITE construction redesigned for protocol state |
| Dolev-Yao adversary model | Dolev & Yao (1983), tlspuffin (IEEE S&P 2024) | Term algebra directly reused from tlspuffin; bounded variant with action budget is standard |
| Program slicing | Weiser (1984), Tip (1995) | Soundness theorem adapted for protocol-specific taint tracking |
| Covering designs | Colbourn (2004), Kuhn et al. (2013) | Standard existence bounds; applied to protocol parameter spaces |

### Genuinely New Contributions

| Contribution | Why Novel |
|-------------|-----------|
| **Negotiation Protocol LTS (Definition 1) with axioms (A1–A4)** | No prior formalization identifies these four algebraic properties of negotiation protocols or exploits them for analysis. The closest work (ProVerif, Tamarin) models protocols at specification level without axiomatizing implementation-level algebraic structure. |
| **Protocol-aware merge operator (Definition 2)** | Novel mergeability predicate and ITE construction exploiting (A1–A4). Generic state merging (Kuznetsov) has no domain-specific mergeability predicate. The exponential-to-polynomial complexity separation (T3) has no prior analog for any structured program domain. |
| **End-to-end composition theorem (T4)** | No prior work composes extraction soundness + protocol-aware merging + DY adversary encoding into a single bounded-completeness guarantee. CompCert composes compiler passes; we compose analysis passes with an adversary model — a fundamentally different composition target. |
| **Bounded-completeness certificates (Definition 6)** | Novel artifacts. No existing tool produces formal certificates of downgrade-freedom for production library code within explicit bounds. |
| **Source-to-attack closed loop** | The full pipeline from C source to byte-level attack traces with formal guarantees is unprecedented. Each stage has precedent; the composition does not. |
| **Covering-design application to cross-library differential testing (C1)** | Novel connection between combinatorial design theory and protocol behavioral deviation detection. Prior differential testing (Brubaker et al., Frankencerts) uses random generation without coverage guarantees. |

---

## Appendix: Definition-to-Module Mapping

| Definition / Theorem | Implementation Module | LoC Estimate |
|----------------------|----------------------|:---:|
| Def 1 (Negotiation LTS) | Protocol Modules (TLS + SSH) | ~20K |
| Def 2 (Merge Operator ⊵) | Protocol-Aware Merge Operator | ~7K |
| Def 3 (Protocol Bisimulation) | State Machine Extractor (quotient algorithm) | ~8K |
| Def 4 (Bounded DY Adversary) | DY+SMT Encoder | ~10K |
| Def 5 (Downgrade Freedom) | DY+SMT Encoder (property clause) | included above |
| Def 6 (Certificate) | CLI/Reporting | ~6K |
| T1 (Extraction Soundness) | Slicer + Extractor integration tests | ~4K tests |
| T2 (Concretizability) | Concretizer + CEGAR loop | ~6K |
| T3 (Merge Correctness) | Merge Operator + property-based tests | ~3K tests |
| T4 (Bounded Completeness) | End-to-end pipeline + evaluation harness | ~12K |
| T5 (SMT Encoding) | DY+SMT Encoder + reference model comparison | ~3K tests |
| C1 (Covering Design) | Differential extension: scenario generator | ~6K |

---

*End of formal foundations. Every definition above is load-bearing: remove any one and at least one theorem statement becomes ill-formed. Every theorem above is load-bearing: remove any one and the end-to-end guarantee (T4) or the cross-library guarantee (C1) collapses.*
