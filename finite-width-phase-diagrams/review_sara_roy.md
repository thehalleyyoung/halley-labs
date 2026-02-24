# Review by Sara Roy (machine_learning_formal_verification)

## Project: PhaseKit — Finite-Width Phase Diagrams for Neural Network Initialization

**Reviewer Expertise:** ML for practitioners, developer tooling, deep learning systems, usability.

---

### Summary

PhaseKit aims to be a practical tool for neural network initialization, computing phase diagrams and recommending critical initializations before training. The problem framing is important—initialization failures waste enormous compute. However, the tool has significant usability gaps that prevent it from being practically useful.

### Strengths

The 100% binary trainability prediction across 27 runs is the most compelling practical result. The code quality is solid: clean dataclass APIs (ArchitectureSpec, MFReport, PhaseClassification), 55 passing tests, and clear module separation. The grounding.json traceability is a model for reproducible research.

### Weaknesses

**1. The tool only supports fully connected networks.** The entire theory is for MLPs. The limitations section acknowledges this. In 2024, almost no practitioner uses plain MLPs for serious tasks. Without support for convolutions, attention, or normalization layers, practical impact is near zero.

**2. The ResNet extension is oversimplified.** It uses a fixed scalar α, assumes pre-activation blocks, and ignores batch/layer normalization. Real ResNets have varying block structures, bottleneck layers, and learned scaling. The 11 unit tests verify mathematical consistency of the simplified model, not relevance to actual ResNets.

**3. The trainability experiment is underpowered.** 27 runs with 5-layer ReLU MLPs on synthetic data is far too small. Real initialization failures involve learning rate interactions, batch size effects, and architecture-specific issues. The 100% accuracy likely reflects testing only extreme cases where failure is obvious.

**4. No framework integration.** For a tool paper, the absence of a PyTorch/JAX integration workflow is critical. pytorch_integration.py exists but the paper shows no user journey: "Here is my model, here is the command, here is the output." A tool paper must demonstrate the developer experience.

**5. Edge-of-chaos values are already known.** The critical σ_w values (ReLU: √2, tanh: 1.010) are established. The value-add is finite-width corrections, but these only matter below width ~256—widths practitioners rarely use for production.

**6. Calibration diagnostics solve a non-problem.** No practitioner needs "82% probability of ordered phase." They need: "Will this train? If not, what should I change?" The tool should output actionable recommendations, not posterior probabilities.

### Grounding Assessment

The init-comparison experiment is grounded to exp_v3_init_comparison.json (5 seeds × 2 widths × 2 depths). However, it tests on synthetic regression only. MNIST experiments exist in the data directory but are not prominently featured. If the tool works on MNIST, that should be the headline result.

### Path to Best Paper

Build a PyTorch integration taking a model definition and returning initialization recommendations. Demonstrate on standard benchmarks (CIFAR-10, MNIST) with standard architectures (ResNet-18, small transformer). Show PhaseKit-recommended init outperforms PyTorch defaults. Replace posteriors with actionable output. Add batch/layer norm support.

### Score: 4/10 — Good research code, but not yet a tool practitioners would use.
