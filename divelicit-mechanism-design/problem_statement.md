# Problem Statement: Diverse LLM Elicitation via Measure Transport

## The Problem

Given access to n LLM agents (or n samples from a single LLM), select a subset of k responses
that jointly maximize a diversity-quality objective. Formally: given a response space Y, a quality
function q: Y → [0,1], and a diversity functional D: 2^Y → R, find S* ⊆ {y₁,...,yₙ} with |S*| = k
maximizing λ·D(S*) + (1-λ)·Σ_{y∈S*} q(y), subject to incentive compatibility constraints ensuring
agents report truthfully rather than strategically.

## Why Existing Approaches Fail

**Temperature / top-k / nucleus sampling** provides no diversity guarantee. High temperature increases
entropy but does not prevent mode collapse—multiple samples often paraphrase the same high-probability
response. There is no formal coverage guarantee over the response space.

**DPP-based methods (DivElicit)** improve diversity using determinantal point processes with fixed
RBF or cosine kernels and "be different" prompting. However, they suffer from three critical limitations:

1. **Fixed kernels miss manifold structure.** A single RBF bandwidth cannot capture the varying local
   geometry of the response embedding space. Responses about "quantum physics" and "cooking recipes"
   require different notions of similarity, but a fixed kernel treats all regions identically.

2. **No differentiable diversity objective for steering.** DPP's log-det objective is submodular
   (enabling greedy approximation) but combinatorial—it cannot provide gradients for adapting the
   kernel or steering generation. This prevents end-to-end optimization of the diversity pipeline.

3. **No finite-sample coverage guarantees.** No existing method can certify that the selected k
   responses cover at least (1-δ) of the reachable response space to within ε precision. Without
   such certificates, users cannot know whether important response modes were missed.

4. **No principled quality-diversity tradeoff.** The λ parameter in DivElicit is set heuristically.
   There is no characterization of the Pareto frontier between quality and diversity, and no guidance
   on where the optimal operating point lies for a given application.

## The DivFlow Solution

DivFlow resolves these gaps through three interconnected innovations:

**Sinkhorn divergence as diversity objective.** We replace DPP's log-det with Sinkhorn divergence
S_ε(μ_S, μ_ref) between the empirical measure of selected responses and a reference measure (e.g.,
uniform on the response space). Sinkhorn divergence is differentiable, unbiased (unlike regularized OT),
metrizes weak convergence on compact spaces, and admits efficient O(n² / ε²) computation.

**Adaptive kernel learning.** Instead of a fixed kernel, DivFlow learns the repulsion kernel online
from observed responses. The kernel bandwidth adapts to local density: tighter in dense regions
(stronger repulsion) and broader in sparse regions (encouraging exploration). This is formalized as
minimizing a kernel alignment objective between the learned kernel and the ideal diversity kernel.

**Coverage certification.** Using ε-covering numbers and Chernoff concentration bounds, DivFlow
provides finite-sample certificates: "with probability ≥ 1-δ, the selected responses ε-cover at
least fraction γ of the reachable response space." This transforms diversity from a heuristic
aspiration into a certifiable property.

The result is a mechanism that is incentive-compatible (agents report truthfully), diversity-optimal
(Sinkhorn divergence is the right objective), adaptive (kernels learn from data), and certified
(coverage guarantees hold with high probability).
