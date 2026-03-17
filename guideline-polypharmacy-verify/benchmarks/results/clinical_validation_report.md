# GuardPharma Clinical Validation Report

**Date:** 2026-03-10 18:56 UTC
**Patients:** 500
**Ground-truth interaction pairs evaluated:** 319
**Interactions in knowledge base:** 127 (121 unique pairs)
**Drugs in catalogue:** 110

## Ground-Truth Sources

| Source | Description |
|--------|-------------|
| DrugBank | 2,700+ approved drugs with severity-classified interactions |
| SIDER | 1,430 drugs, 5,868 side effects from package inserts |
| TWOSIDES | 645 drugs, 1,318 drug-drug interaction side effects (data-mined) |
| WHO EML | Essential Medicines List drug interactions |
| Beers 2023 | AGS criteria: 40+ drugs to avoid in elderly |
| STOPP/START v3 | 80+ screening rules for older persons |
| Case Reports | Published case reports of serious ADRs |

## Primary Metrics

| Metric | Value |
|--------|-------|
| **Sensitivity (Recall)** | 0.0000 |
| **Specificity** | 0.9979 |
| **PPV (Precision)** | 0.0000 |
| **NPV** | 0.9747 |
| **F1 Score** | 0.0000 |
| **Major Interaction Sensitivity** | 0.0000 |

## Confusion Matrix

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actually Positive** | TP = 0 | FN = 319 |
| **Actually Negative** | FP = 26 | TN = 12301 |

## Severity Breakdown

| Severity | True Positives | False Negatives | Sensitivity |
|----------|---------------|-----------------|-------------|
| Major | 0 | 191 | 0.000 |
| Moderate | 0 | 128 | 0.000 |

## Detection by Ground-Truth Source

| Source | True Positives | False Negatives | Sensitivity |
|--------|---------------|-----------------|-------------|
| CaseReport | 0 | 105 | 0.000 |
| DrugBank | 0 | 69 | 0.000 |
| SIDER | 0 | 19 | 0.000 |
| STOPP | 0 | 101 | 0.000 |
| TWOSIDES | 0 | 25 | 0.000 |

## Comparison with Clinical Decision Support Systems

| System | Sensitivity | PPV | F1 |
|--------|------------|-----|-----|
| **GuardPharma** | **0.000** | **0.000** | **0.000** |
| Lexicomp (published) | 0.768 | 0.787 | 0.777 |
| Micromedex (published) | 0.722 | 0.790 | 0.755 |

*Published CDS rates from Roblek et al. (2015) Eur J Clin Pharmacol 71(2):131-142 and GuardPharma Experiment 1.*

## Performance

- **Total time:** 14.5s
- **Avg per patient:** 28.9ms
- **Errors:** 117
- **Timeouts:** 0
