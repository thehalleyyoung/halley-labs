# Verification Gate Report: proto-downgrade-synth

**Project:** Negotiation Under Fire: Bounded-Complete Synthesis of Protocol Downgrade Attacks from Library Source Code
**Slug:** `proto-downgrade-synth`
**Stage:** Verification (post-theory)
**Date:** 2026-03-08
**Method:** Claude Code Agent Teams — 3 expert agents (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with adversarial cross-critique, lead synthesis, and verification signoff
**Prior Inputs:** 3 independent evaluations (Skeptic 5.2/10, Mathematician 5.5/10, Community Expert 6.0/10)

---

## Executive Summary

**Composite: 5.2/10** (V6/D6/BP4/L7/F5). **CONDITIONAL CONTINUE** — 2-1, Skeptic dissents ABANDON at 5.0.

Six independent expert evaluations (3 prior + 3 new) converge on the same qualitative story: real problem, novel pipeline composition, shallow mathematics (depth 3.5–4.5/10), zero implementation, and punishing compound probability (P(full vision) = 15–19%). The proposal addresses a genuine, persistent security gap — protocol downgrade attacks continue to appear in modern code (Terrapin 2023, wolfSSL 2024–2025) — and no existing tool bridges the specification-implementation gap for negotiation analysis. The end-to-end pipeline (C source → LLVM IR → protocol-aware slice → KLEE with merge → state machine → DY+SMT → attack traces or certificates) has no prior analog.

However, the proposal fails all three of the user's criteria at the 7/10 threshold: Value (6), Difficulty (6), Best-Paper (4). The crown theorem T3 is a domain instantiation of Kuznetsov state merging + Milner bisimulation at depth 4/10 — a correct and useful observation, not a hard proof. The certificates depend on an unproved slicer assumption (A0) and carry 7 scope exclusions. P(best-paper) = 3%. The CONTINUE recommendation rests entirely on the kill-gate structure: the first 6 weeks (~1.5 person-months) resolve ~73% of uncertainty, limiting downside if gates fail.

---

## 1. Team Composition and Process

| Role | Mandate | Composite | Verdict |
|------|---------|:---------:|---------|
| **Independent Auditor** | Evidence-based scoring, challenge testing | 5.4 | CONDITIONAL CONTINUE (binary: ABANDON) |
| **Fail-Fast Skeptic** | Aggressively reject under-supported claims | 5.0 | **ABANDON** |
| **Scavenging Synthesizer** | Salvage value from partial failure | 5.5 | CONDITIONAL CONTINUE |

**Process:** Independent proposals (parallel) → adversarial cross-critiques (4 explicit challenges) → synthesis of strongest elements → verification signoff.

---

## 2. Prior Evaluation Synthesis

Three independent evaluations preceded this team:

| Evaluator | Composite | Value | Difficulty | Best-Paper | Verdict |
|-----------|:---------:|:-----:|:----------:|:----------:|---------|
| Skeptic (prior) | 5.2 | 6 | 6 | 5 | COND. CONTINUE (2-1) |
| Mathematician (prior) | 5.5 | 6 | 6 | 4 | COND. CONTINUE (2-1) |
| Community Expert (prior) | 6.0 | 6.5 | 7 | 5.5 | COND. CONTINUE |

All six evaluations (3 prior + 3 new) place the composite in the 5.0–6.0 range. The disagreements are narrow — they concern whether marginal scores justify continuation (Synthesizer) or termination (Skeptic). This convergence across independent evaluations with different analytical frames provides HIGH confidence in the scoring.

---

## 3. Adversarial Cross-Critique Resolutions

### Challenge 1: Skeptic → Synthesizer (CONTINUE vs. ABANDON)
**Winner: Synthesizer.** The kill-gate structure converts an 18-month commitment into a sequence of binary decisions. The first 6 weeks cost ~1.5 person-months and resolve ~73% of uncertainty. The Skeptic's ABANDON treats the project as monolithic, ignoring the option value of cheap early gates. However, the Synthesizer's position holds ONLY if kill gates are enforced with zero tolerance for partial passes.

### Challenge 2: Skeptic → Auditor (Value 5 vs. 6)
**Winner: Split.** The Skeptic correctly identifies that OpenSSL 3.x's provider architecture breaks the slicer (F4 is real). The Auditor correctly argues SSH + legacy TLS + WolfSSL + BoringSSL carry sufficient value. **Value holds at 6** — the OpenSSL 3.x gap becomes a binding disclosure requirement, not a fatal flaw.

### Challenge 3: Auditor → Synthesizer (P(pub) inflation)
**Winner: Auditor, narrowly.** P(≥1 pub) adjusted from Synthesizer's 60–65% down to **55–60%**. Zero empirical validation weakens even fallback papers, but the fallback cascade has genuinely lower bars than the full system.

### Challenge 4: All three on Best-Paper
**Winner: All three correct.** The deflation from self-assessed 7 to 4 is fully justified. T3 at depth 4/10, certificates conditional on A0, O(n) applies only to cipher-selection subroutine, no new CVE expected. Only a 0-day discovery (~3% probability) reaches BP 7.

---

## 4. Unified Scoring

| Dimension | Score | Key Evidence |
|-----------|:-----:|-------------|
| **Extreme Value** | **6** | Proven-lethal problem (Terrapin 2023, wolfSSL 2024-2025). No competing tool closes spec-implementation gap. Narrow audience (~50 maintainers). 7 scope exclusions qualify the certificate. TLS 1.3 narrows TLS-specific surface. OpenSSL 3.x provider architecture unanalyzable (F4). |
| **Genuine Difficulty** | **6** | ~50K novel LoC, KLEE integration is genuine systems pain. But every subsystem has a published template (Kuznetsov, Milner, Clarke, AVISPA, Nelson-Oppen). Integration-hard, not algorithmically-hard. |
| **Best-Paper Potential** | **4** | T3 is domain instantiation at depth 4/10. T4 is composition at depth 3/10. All techniques borrowed. O(n) claim applies to cipher-selection subroutine only. Certificates conditional on unproved A0. No new CVE expected. P(best-paper) = 3%. |
| **Laptop-CPU Feasibility** | **7** | Architecture naturally CPU-bound. SMT solving sequential. Slicing reduces code by ~95%. 32GB RAM sufficient. Z3 timeout (40%) is feasibility risk, not architecture constraint. Zero humans needed at runtime. |
| **Overall Feasibility** | **5** | P(full vision) = 15–19%. P(MVP) ≈ 55%. P(any pub) ≈ 55–60%. Kill gates bound downside to 2–4 months. Correlated risk cascade (KLEE→slicer→Z3). impl_loc = 0 — everything is projection. |
| **Composite** | **5.2** | Weighted: (6×0.25)+(6×0.15)+(4×0.20)+(7×0.15)+(5×0.25) = 5.5; adjusted −0.3 for zero-implementation risk and correlated gates |

### User's Three Pillars

| Criterion | Score | Threshold | Status |
|-----------|:-----:|:---------:|:------:|
| (a) Extreme obvious value | 6 | 7 | ❌ BELOW |
| (b) Genuinely difficult as software | 6 | 7 | ❌ BELOW |
| (c) Real best-paper potential | 4 | 7 | ❌ FAIL |

---

## 5. Probability Estimates

| Metric | Estimate | Basis |
|--------|:--------:|-------|
| P(best-paper at top-4 venue) | **3%** | T3 depth 4/10; all techniques borrowed; 50% chance of no new CVE |
| P(accepted at top-4 venue, full vision) | **25–30%** | Requires full pipeline + certificates + 7+ CVEs |
| P(accepted at top-4 venue, MVP) | **35–45%** | MVP = 2 libraries, 4–5 CVEs, merge + attack synthesis |
| P(any peer-reviewed publication) | **55–60%** | Includes fallback cascade: money-plot paper, slicer tool, workshops |
| P(full vision delivered, 18 months) | **15–19%** | Compound of 5 correlated risks |
| P(MVP delivered) | **55%** | Drops certificates and differential extension |
| P(ABANDON at kill gate) | **30%** | R1 (KLEE bitcode, 30%) is dominant early risk |

### Risk-Adjusted Expected Value

EV = P(full paper)×7.5 + P(fallback pub only)×4.0 + P(no pub)×1.0
   = 0.27×7.5 + 0.31×4.0 + 0.42×1.0 = **3.7/10**

Marginal but positive when weighted against the ~6-week option cost to first major decision point.

---

## 6. Fatal Flaws

| # | Flaw | Severity | Status |
|---|------|----------|--------|
| F1 | Slicer soundness (A0) is assumed, not proved — certificates are only as strong as this assumption | SERIOUS | Mitigated: A0 stated explicitly, validated empirically. Certificates say "under A0." |
| F2 | Z3 tractability on DY+SMT encoding is unvalidated | SERIOUS | Kill gate G1.5 (week 5): hand-encoded micro-prototype tests tractability |
| F3 | O(n) applies to cipher-selection subroutine only, not full negotiation (~1,350 real paths) | SERIOUS | Mitigated: honest reframing in paper; report both idealized and realistic numbers |
| F4 | OpenSSL 3.x provider architecture breaks slicer | SERIOUS | Scope to OpenSSL 1.x + BoringSSL + WolfSSL + libssh2; disclose limitation |
| NF1 | Real OpenSSL code violates A1–A4 (callbacks, FIPS, renegotiation) | MODERATE | Per-region property checker with graceful fallback |
| NF2 | Correlated risk cascade (KLEE→slicer→Z3) means P(full) may be 12–15% | MODERATE | Kill gates provide early termination |

**Verdict: 0 independently fatal flaws. 4 SERIOUS issues, all mitigated or scoped.**

---

## 7. The Diamond (What Survives Partial Failure)

The **protocol-aware merge operator (T3)** — identifying that negotiation protocols satisfy four algebraic properties (A1–A4) enabling exponential-to-linear path reduction — has standalone publication value even if the full pipeline never ships. The merge operator is rated 8/10 for salvage value, independently publishable at ASE/ISSTA/ACSAC.

The **protocol-aware slicer** is independently useful for any C library security audit. The **DY+SMT encoding framework** is reusable by any symbolic execution tool wanting adversary modeling.

**Minimum viable contribution:** Merge operator formalization + OpenSSL-only KLEE validation + 3–4 CVE recoveries + money-plot graph. ~15–20K LoC. ~4 months. P(publish) ≈ 65–70% at CCS short paper or ASE/ISSTA.

---

## 8. Binding Conditions for CONTINUE

| # | Condition | Deadline | Consequence |
|---|-----------|----------|-------------|
| BC-1 | G0: KLEE symbolically executes `ssl3_choose_cipher` on OpenSSL 1.1.x bitcode | Week 2 | **KILL** if fails |
| BC-2 | G1: Protocol-aware slicer produces ≤15K lines with ≥90% negotiation coverage | Week 4 | **KILL** if >15K lines |
| BC-3 | G1.5: Z3 micro-prototype — hand-encoded FREAK DY model returns SAT in <10 minutes | Week 5 | **KILL** SMT path; pivot to Fallback A |
| BC-4 | G2: Merge operator achieves ≥10× path reduction vs. vanilla KLEE on OpenSSL | Week 6 | **KILL** full pipeline if <5×; pivot to Fallback A |
| BC-5 | Reframe "bounded completeness" as "bounded model checking under Assumption A0" | Immediate | Framing requirement — no kill gate |
| BC-6 | Target CCS/USENIX Security as primary venue, not S&P | Immediate | Expectation management |
| BC-7 | MVP-first architecture: all design decisions optimize for 2-library, 4–5 CVE scope | Immediate | Scope requirement |
| BC-8 | Zero tolerance for partial gate passes — "almost passes" = FAIL | All gates | Discipline requirement |
| BC-9 | Fix `theory_bytes=0` pipeline measurement bug | Before next stage | Pipeline instrumentation |

---

## 9. Team Signoff

| Expert | Score | Verdict | Key Argument |
|--------|:-----:|---------|-------------|
| Independent Auditor | 5.4 | CONDITIONAL CONTINUE | 7 unsupported claims demand early validation; kill gates make CONTINUE defensible; binary choice would be ABANDON |
| Fail-Fast Skeptic | 5.0 | **ABANDON** | All 3 criteria fail 7/10 threshold; certificates are "asterisk factory"; OpenSSL 3.x gap is dispositive; P(best-paper)=2–3% |
| Scavenging Synthesizer | 5.5 | CONDITIONAL CONTINUE | Merge operator (8/10 salvage) is the diamond; first 6 weeks resolve 80% of uncertainty; P(≥1 pub)=60–65%; MVP-first |

**Final disposition: CONDITIONAL CONTINUE (2-1, Skeptic dissents ABANDON)**

---

## 10. What Would Change the Verdict

**To UNCONDITIONAL CONTINUE:** All of G0–G2 pass by week 6 with strong results (merge shows 50×+ speedup, slicer produces <5K-line slices, Z3 solves FREAK in <1 minute). This would raise composite to ~6.5–7.0.

**To ABANDON:** Any kill gate fails. OR: a competing tool (tlspuffin v2) publishes source-level bounded protocol analysis before this reaches MVP. OR: kill-gate discipline breaks (partial passes tolerated).

---

## 11. Ranking

```json
{
  "rankings": [
    {
      "proposal_id": "proposal_00",
      "score": 5.2,
      "verdict": "CONTINUE",
      "reason": "Composite 5.2/10 (V6/D6/BP4/L7/F5). CONDITIONAL CONTINUE (2-1, Skeptic dissents ABANDON). Fails all three user criteria at 7/10 threshold (Value 6, Difficulty 6, Best-Paper 4), but kill-gate structure limits downside to ~6 weeks before major decision. P(best-paper)=3%. P(any-pub)=55-60%. P(full-vision)=15-19%. Merge operator has standalone salvage value (8/10). 9 binding conditions including 5 kill gates (G0-G3 + Z3 micro-prototype). theory_bytes=0 is measurement bug (329K bytes exist). MVP-first architecture targeting CCS/USENIX Security.",
      "scavenge_from": []
    }
  ]
}
```

---

*Verification gate report produced by Claude Code Agent Teams: 3 independent expert agents (Independent Auditor, Fail-Fast Skeptic, Scavenging Synthesizer) with adversarial cross-critique, lead synthesis, and verification signoff. Skeptic dissents ABANDON at 5.0. Composite: 5.2/10. 9 binding conditions. 5 kill gates.*
