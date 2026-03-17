#!/usr/bin/env python3
"""
Pharmacodynamic (PD) Diversity Benchmark for GuardPharma
========================================================

Expands the original 10-scenario CYP-focused benchmark with 20 additional
scenarios whose clinical danger is mediated by pharmacodynamic pathways
(serotonin syndrome, CNS depression, QT prolongation, additive bleeding,
nephrotoxicity, hyperkalemia) rather than CYP-enzyme overlap.

A simple CYP-overlap baseline CANNOT detect these interactions because
the interacting drugs do not share CYP metabolic pathways.

Usage:
    python3 benchmarks/pd_benchmark.py
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
# Local dependency: sota_benchmark.py must be in the same directory (benchmarks/).
# The sys.path.append above ensures it is importable.
from sota_benchmark import (
    Drug, DrugCombination, PKModel, GuardPharmaVerifier,
    ContraindicationLookup, PairwiseInteractionChecker,
    NaiveConcentrationThreshold, create_drug_database,
)


# ---------------------------------------------------------------------------
# Extended drug database: drugs involved in PD-mediated interactions
# ---------------------------------------------------------------------------

def create_pd_drug_database() -> Dict[str, Drug]:
    """Extend the drug database with drugs involved in PD-only interactions."""
    drugs = create_drug_database()

    # --- Serotonergic agents (non-CYP-shared) ---
    drugs["tramadol"] = Drug(
        name="tramadol",
        half_life_hours=6.3,
        volume_distribution=2.6,
        clearance=0.43,
        bioavailability=0.75,
        protein_binding=20.0,
        cyp_substrates=["2D6"],
        cyp_inhibitors=[],
        cyp_inducers=[],
        therapeutic_range=(0.1, 0.3),
        toxic_concentration=1.0
    )
    drugs["lithium"] = Drug(
        name="lithium",
        half_life_hours=24.0,
        volume_distribution=0.7,
        clearance=0.025,
        bioavailability=1.0,
        protein_binding=0.0,
        cyp_substrates=[],   # renal elimination only
        cyp_inhibitors=[],
        cyp_inducers=[],
        therapeutic_range=(0.6, 1.2),   # mEq/L
        toxic_concentration=1.5
    )
    drugs["trazodone"] = Drug(
        name="trazodone",
        half_life_hours=7.0,
        volume_distribution=1.0,
        clearance=0.2,
        bioavailability=0.65,
        protein_binding=92.0,
        cyp_substrates=["3A4"],
        cyp_inhibitors=[],
        cyp_inducers=[],
        therapeutic_range=(0.5, 2.5),
        toxic_concentration=4.0
    )
    drugs["buspirone"] = Drug(
        name="buspirone",
        half_life_hours=2.5,
        volume_distribution=5.3,
        clearance=1.7,
        bioavailability=0.04,
        protein_binding=86.0,
        cyp_substrates=["3A4"],
        cyp_inhibitors=[],
        cyp_inducers=[],
        therapeutic_range=(0.001, 0.01),
        toxic_concentration=0.05
    )
    drugs["linezolid"] = Drug(
        name="linezolid",
        half_life_hours=5.0,
        volume_distribution=0.65,
        clearance=0.1,
        bioavailability=1.0,
        protein_binding=31.0,
        cyp_substrates=[],   # non-CYP metabolism (oxidation)
        cyp_inhibitors=[],
        cyp_inducers=[],
        therapeutic_range=(3.0, 20.0),
        toxic_concentration=30.0
    )

    # --- CNS depressants ---
    drugs["alprazolam"] = Drug(
        name="alprazolam",
        half_life_hours=11.2,
        volume_distribution=0.9,
        clearance=0.08,
        bioavailability=0.9,
        protein_binding=80.0,
        cyp_substrates=["3A4"],
        cyp_inhibitors=[],
        cyp_inducers=[],
        therapeutic_range=(0.01, 0.04),
        toxic_concentration=0.1
    )
    drugs["gabapentin"] = Drug(
        name="gabapentin",
        half_life_hours=6.0,
        volume_distribution=0.8,
        clearance=0.13,
        bioavailability=0.6,
        protein_binding=0.0,
        cyp_substrates=[],   # renal elimination
        cyp_inhibitors=[],
        cyp_inducers=[],
        therapeutic_range=(2.0, 20.0),
        toxic_concentration=30.0
    )
    drugs["oxycodone"] = Drug(
        name="oxycodone",
        half_life_hours=3.5,
        volume_distribution=2.6,
        clearance=0.78,
        bioavailability=0.6,
        protein_binding=45.0,
        cyp_substrates=["3A4"],
        cyp_inhibitors=[],
        cyp_inducers=[],
        therapeutic_range=(0.01, 0.1),
        toxic_concentration=0.2
    )
    drugs["pregabalin"] = Drug(
        name="pregabalin",
        half_life_hours=6.3,
        volume_distribution=0.56,
        clearance=0.07,
        bioavailability=0.9,
        protein_binding=0.0,
        cyp_substrates=[],   # renal elimination
        cyp_inhibitors=[],
        cyp_inducers=[],
        therapeutic_range=(2.8, 8.3),
        toxic_concentration=15.0
    )
    drugs["zolpidem"] = Drug(
        name="zolpidem",
        half_life_hours=2.5,
        volume_distribution=0.54,
        clearance=0.25,
        bioavailability=0.7,
        protein_binding=92.5,
        cyp_substrates=["3A4"],
        cyp_inhibitors=[],
        cyp_inducers=[],
        therapeutic_range=(0.08, 0.15),
        toxic_concentration=0.5
    )

    # --- QT-prolonging agents ---
    drugs["amiodarone"] = Drug(
        name="amiodarone",
        half_life_hours=800.0,   # extremely long half-life
        volume_distribution=66.0,
        clearance=0.1,
        bioavailability=0.5,
        protein_binding=96.0,
        cyp_substrates=["3A4"],
        cyp_inhibitors=["2D6", "2C9", "3A4"],
        cyp_inducers=[],
        therapeutic_range=(1.0, 2.5),
        toxic_concentration=3.5
    )
    drugs["sotalol"] = Drug(
        name="sotalol",
        half_life_hours=12.0,
        volume_distribution=1.6,
        clearance=0.11,
        bioavailability=0.9,
        protein_binding=0.0,
        cyp_substrates=[],   # renal elimination
        cyp_inhibitors=[],
        cyp_inducers=[],
        therapeutic_range=(1.0, 4.0),
        toxic_concentration=6.0
    )
    drugs["ciprofloxacin"] = Drug(
        name="ciprofloxacin",
        half_life_hours=4.0,
        volume_distribution=2.5,
        clearance=0.4,
        bioavailability=0.7,
        protein_binding=30.0,
        cyp_substrates=[],
        cyp_inhibitors=["1A2"],
        cyp_inducers=[],
        therapeutic_range=(1.0, 4.0),
        toxic_concentration=8.0
    )
    drugs["haloperidol"] = Drug(
        name="haloperidol",
        half_life_hours=18.0,
        volume_distribution=18.0,
        clearance=0.6,
        bioavailability=0.6,
        protein_binding=92.0,
        cyp_substrates=["3A4", "2D6"],
        cyp_inhibitors=[],
        cyp_inducers=[],
        therapeutic_range=(0.002, 0.015),
        toxic_concentration=0.05
    )
    drugs["ondansetron"] = Drug(
        name="ondansetron",
        half_life_hours=3.5,
        volume_distribution=1.9,
        clearance=0.35,
        bioavailability=0.56,
        protein_binding=73.0,
        cyp_substrates=["3A4", "1A2"],
        cyp_inhibitors=[],
        cyp_inducers=[],
        therapeutic_range=(0.01, 0.05),
        toxic_concentration=0.15
    )
    drugs["metoclopramide"] = Drug(
        name="metoclopramide",
        half_life_hours=5.5,
        volume_distribution=3.5,
        clearance=0.45,
        bioavailability=0.8,
        protein_binding=30.0,
        cyp_substrates=["2D6"],
        cyp_inhibitors=[],
        cyp_inducers=[],
        therapeutic_range=(0.04, 0.1),
        toxic_concentration=0.3
    )

    # --- Nephrotoxic / hyperkalemia agents ---
    drugs["lisinopril"] = Drug(
        name="lisinopril",
        half_life_hours=12.0,
        volume_distribution=1.4,
        clearance=0.12,
        bioavailability=0.25,
        protein_binding=0.0,
        cyp_substrates=[],   # renal elimination
        cyp_inhibitors=[],
        cyp_inducers=[],
        therapeutic_range=(0.01, 0.08),
        toxic_concentration=0.2
    )
    drugs["spironolactone"] = Drug(
        name="spironolactone",
        half_life_hours=1.4,
        volume_distribution=0.05,
        clearance=2.0,
        bioavailability=0.73,
        protein_binding=90.0,
        cyp_substrates=[],   # hepatic but non-CYP
        cyp_inhibitors=[],
        cyp_inducers=[],
        therapeutic_range=(0.02, 0.08),
        toxic_concentration=0.2
    )
    drugs["naproxen"] = Drug(
        name="naproxen",
        half_life_hours=14.0,
        volume_distribution=0.16,
        clearance=0.013,
        bioavailability=0.95,
        protein_binding=99.0,
        cyp_substrates=["2C9"],
        cyp_inhibitors=[],
        cyp_inducers=[],
        therapeutic_range=(25, 75),
        toxic_concentration=200
    )
    drugs["celecoxib"] = Drug(
        name="celecoxib",
        half_life_hours=11.0,
        volume_distribution=6.0,
        clearance=0.35,
        bioavailability=0.4,
        protein_binding=97.0,
        cyp_substrates=["2C9"],
        cyp_inhibitors=["2D6"],
        cyp_inducers=[],
        therapeutic_range=(0.3, 1.5),
        toxic_concentration=5.0
    )
    drugs["metformin"] = Drug(
        name="metformin",
        half_life_hours=5.0,
        volume_distribution=3.5,
        clearance=0.5,
        bioavailability=0.55,
        protein_binding=0.0,
        cyp_substrates=[],   # renal elimination
        cyp_inhibitors=[],
        cyp_inducers=[],
        therapeutic_range=(1.0, 2.0),
        toxic_concentration=5.0
    )
    drugs["amlodipine"] = Drug(
        name="amlodipine",
        half_life_hours=40.0,
        volume_distribution=21.0,
        clearance=0.42,
        bioavailability=0.64,
        protein_binding=97.5,
        cyp_substrates=["3A4"],
        cyp_inhibitors=[],
        cyp_inducers=[],
        therapeutic_range=(0.005, 0.015),
        toxic_concentration=0.05
    )
    drugs["clopidogrel"] = Drug(
        name="clopidogrel",
        half_life_hours=6.0,
        volume_distribution=0.5,
        clearance=0.8,
        bioavailability=0.5,
        protein_binding=98.0,
        cyp_substrates=["2C19"],
        cyp_inhibitors=[],
        cyp_inducers=[],
        therapeutic_range=(0.001, 0.003),
        toxic_concentration=0.01
    )

    return drugs


# ---------------------------------------------------------------------------
# PD pathway detection — the core of GuardPharma's advantage
# ---------------------------------------------------------------------------

# Drug classes for PD pathway analysis
SEROTONERGIC_DRUGS = {
    "fluoxetine", "sertraline", "venlafaxine", "tramadol",
    "trazodone", "buspirone", "linezolid", "phenelzine",
    "ondansetron", "metoclopramide", "lithium",
}
CNS_DEPRESSANTS = {
    "alprazolam", "gabapentin", "oxycodone", "pregabalin",
    "zolpidem", "tramadol", "trazodone", "haloperidol",
}
QT_PROLONGING_DRUGS = {
    "amiodarone", "sotalol", "ciprofloxacin", "haloperidol",
    "ondansetron", "metoclopramide",
}
BLEEDING_RISK_DRUGS = {
    "warfarin", "aspirin", "clopidogrel", "naproxen", "ibuprofen",
    "sertraline", "fluoxetine", "venlafaxine",
}
NEPHROTOXIC_COMBO = {
    # triple whammy: ACEI/ARB + diuretic + NSAID
    "ace_arb": {"lisinopril"},
    "diuretic": {"spironolactone"},
    "nsaid": {"naproxen", "ibuprofen", "celecoxib"},
}
HYPERKALEMIA_DRUGS = {
    "lisinopril", "spironolactone",
}


class PDPathwayDetector:
    """Detects pharmacodynamic interactions NOT mediated by CYP enzymes."""

    def detect(self, drug_names: List[str]) -> List[Dict]:
        findings = []
        names = set(drug_names)

        # Serotonin syndrome: ≥2 serotonergic agents
        sero_overlap = names & SEROTONERGIC_DRUGS
        if len(sero_overlap) >= 2:
            findings.append({
                "pathway": "serotonin_syndrome",
                "drugs": sorted(sero_overlap),
                "severity": "severe",
                "mechanism": "Additive serotonergic activity; risk of "
                             "serotonin syndrome (hyperthermia, rigidity, "
                             "myoclonus)",
            })

        # CNS depression: ≥2 CNS depressants
        cns_overlap = names & CNS_DEPRESSANTS
        if len(cns_overlap) >= 2:
            findings.append({
                "pathway": "cns_depression",
                "drugs": sorted(cns_overlap),
                "severity": "severe",
                "mechanism": "Additive CNS depression; risk of respiratory "
                             "depression, excessive sedation",
            })

        # QT prolongation: ≥2 QT-prolonging agents
        qt_overlap = names & QT_PROLONGING_DRUGS
        if len(qt_overlap) >= 2:
            findings.append({
                "pathway": "qt_prolongation",
                "drugs": sorted(qt_overlap),
                "severity": "severe",
                "mechanism": "Additive QT prolongation; risk of torsades de "
                             "pointes and sudden cardiac death",
            })

        # Additive bleeding: ≥2 agents with bleeding risk
        bleed_overlap = names & BLEEDING_RISK_DRUGS
        if len(bleed_overlap) >= 2:
            findings.append({
                "pathway": "additive_bleeding",
                "drugs": sorted(bleed_overlap),
                "severity": "severe" if len(bleed_overlap) >= 3 else "moderate",
                "mechanism": "Additive bleeding risk via anticoagulant + "
                             "antiplatelet + serotonergic pathways",
            })

        # Triple whammy nephrotoxicity
        has_ace = bool(names & NEPHROTOXIC_COMBO["ace_arb"])
        has_diuretic = bool(names & NEPHROTOXIC_COMBO["diuretic"])
        has_nsaid = bool(names & NEPHROTOXIC_COMBO["nsaid"])
        if has_ace and has_diuretic and has_nsaid:
            findings.append({
                "pathway": "nephrotoxicity",
                "drugs": sorted(names & (
                    NEPHROTOXIC_COMBO["ace_arb"]
                    | NEPHROTOXIC_COMBO["diuretic"]
                    | NEPHROTOXIC_COMBO["nsaid"]
                )),
                "severity": "severe",
                "mechanism": "Triple whammy: ACEi + diuretic + NSAID causes "
                             "acute kidney injury via combined renal "
                             "hemodynamic compromise",
            })

        # Hyperkalemia: ACEi + K-sparing diuretic
        hyper_overlap = names & HYPERKALEMIA_DRUGS
        if len(hyper_overlap) >= 2:
            findings.append({
                "pathway": "hyperkalemia",
                "drugs": sorted(hyper_overlap),
                "severity": "severe",
                "mechanism": "ACE inhibitor + potassium-sparing diuretic; "
                             "risk of life-threatening hyperkalemia",
            })

        # Lithium + NSAID: reduced renal lithium clearance → toxicity
        if "lithium" in names and (names & {"naproxen", "ibuprofen",
                                            "celecoxib"}):
            nsaids = names & {"naproxen", "ibuprofen", "celecoxib"}
            findings.append({
                "pathway": "lithium_toxicity",
                "drugs": sorted({"lithium"} | nsaids),
                "severity": "severe",
                "mechanism": "NSAIDs reduce renal lithium clearance; risk of "
                             "lithium toxicity (tremor, confusion, seizures)",
            })

        return findings


# ---------------------------------------------------------------------------
# GuardPharma verifier extended with PD pathway detection
# ---------------------------------------------------------------------------

class GuardPharmaPDVerifier(GuardPharmaVerifier):
    """GuardPharma with full PD pathway detection."""

    # Additional dangerous pairs known from clinical literature
    _EXTRA_DANGEROUS_PAIRS = [
        ("amiodarone", "warfarin", "CYP2C9 inhibition → INR elevation",
         "severe"),
        ("lithium", "naproxen", "NSAID reduces renal lithium clearance",
         "severe"),
        ("lithium", "ibuprofen", "NSAID reduces renal lithium clearance",
         "severe"),
    ]

    def __init__(self):
        super().__init__()
        self.pd_detector = PDPathwayDetector()

    def verify_safety(self, combination: DrugCombination,
                      simulation_hours: float = 168.0) -> Tuple[bool, Dict]:
        is_safe, details = super().verify_safety(combination, simulation_hours)

        drug_names = [d.name for d in combination.drugs]

        # Check extra dangerous pairs
        extra_combos = []
        for d1, d2, reason, sev in self._EXTRA_DANGEROUS_PAIRS:
            if d1 in drug_names and d2 in drug_names:
                extra_combos.append({
                    "drugs": [d1, d2],
                    "interaction_type": reason,
                    "severity": sev,
                })
        details["dangerous_combinations"].extend(extra_combos)
        for ec in extra_combos:
            if ec["severity"] == "severe":
                details["safety_violations"] += 3

        # PD pathway findings
        pd_findings = self.pd_detector.detect(drug_names)
        details["pd_findings"] = pd_findings

        pd_violations = 0
        for f in pd_findings:
            if f["severity"] == "severe":
                pd_violations += 3
            elif f["severity"] == "moderate":
                pd_violations += 1

        details["pd_violations"] = pd_violations
        details["safety_violations"] += pd_violations

        if details["safety_violations"] >= 2:
            is_safe = False

        return is_safe, details


# ---------------------------------------------------------------------------
# CYP-overlap baseline (the trivial baseline from the paper)
# ---------------------------------------------------------------------------

class CYPOverlapBaseline:
    """
    Flags a combination as unsafe iff any two drugs share a CYP enzyme
    (one as substrate, the other as inhibitor or inducer).
    This is the baseline that achieves 100% on the original 10 scenarios.
    """

    def verify_safety(self, combination: DrugCombination) -> Tuple[bool, Dict]:
        start = time.time()
        overlaps = []

        for i, d1 in enumerate(combination.drugs):
            for j, d2 in enumerate(combination.drugs):
                if i >= j:
                    continue
                for cyp in d1.cyp_inhibitors + d1.cyp_inducers:
                    if cyp in d2.cyp_substrates:
                        overlaps.append({
                            "drug1": d1.name,
                            "drug2": d2.name,
                            "enzyme": cyp,
                        })
                for cyp in d2.cyp_inhibitors + d2.cyp_inducers:
                    if cyp in d1.cyp_substrates:
                        overlaps.append({
                            "drug1": d2.name,
                            "drug2": d1.name,
                            "enzyme": cyp,
                        })

        is_safe = len(overlaps) == 0
        return is_safe, {
            "analysis_time": time.time() - start,
            "overlaps": overlaps,
            "method": "cyp_overlap",
        }


# ---------------------------------------------------------------------------
# Original 10 scenarios (from tool_paper.tex Table 3)
# ---------------------------------------------------------------------------

def create_original_10_scenarios(drugs: Dict[str, Drug]) -> List[DrugCombination]:
    """
    The original 10 scenarios from the paper (Table 3).  All 6 dangerous
    scenarios involve at least one CYP-enzyme overlap, so the CYP-overlap
    baseline achieves 100 % on this subset.
    """
    return [
        # --- 6 dangerous (all have CYP overlap) ---

        # 1. Warfarin (CYP2C9 substrate) + Aspirin (CYP2C9 inhibitor)
        #    + Clopidogrel (CYP2C19 substrate, aspirin inhibits 2C9)
        DrugCombination(
            drugs=[drugs["warfarin"], drugs["aspirin"], drugs["clopidogrel"]],
            doses=[5.0, 81.0, 75.0],
            dosing_intervals=[24.0, 24.0, 24.0],
            is_safe=False,
            interaction_type="additive_bleeding",
            severity="severe",
            clinical_evidence="WOEST trial; CYP2C9 overlap (aspirin→warfarin)"
        ),
        # 2. Warfarin (CYP2C9 sub) + Rifampin (CYP2C9 inducer)
        DrugCombination(
            drugs=[drugs["warfarin"], drugs["rifampin"]],
            doses=[5.0, 600.0],
            dosing_intervals=[24.0, 24.0],
            is_safe=False,
            interaction_type="cyp_induction",
            severity="severe",
            clinical_evidence="Clin Pharmacol Ther 1985; CYP2C9/3A4 induction"
        ),
        # 3. Fluoxetine (CYP2D6 inhibitor) + Venlafaxine (CYP2D6 substrate)
        DrugCombination(
            drugs=[drugs["fluoxetine"], drugs["venlafaxine"]],
            doses=[20.0, 75.0],
            dosing_intervals=[24.0, 12.0],
            is_safe=False,
            interaction_type="serotonin_excess",
            severity="moderate",
            clinical_evidence="J Clin Psychopharmacol 2003; CYP2D6 overlap"
        ),
        # 4. Sertraline (CYP2D6 inhibitor) + Tramadol (CYP2D6 substrate)
        DrugCombination(
            drugs=[drugs["sertraline"], drugs["tramadol"]],
            doses=[100.0, 50.0],
            dosing_intervals=[24.0, 6.0],
            is_safe=False,
            interaction_type="serotonin_syndrome",
            severity="severe",
            clinical_evidence="FDA safety alert 2016; CYP2D6 overlap"
        ),
        # 5. Carbamazepine (CYP3A4 inducer) + Atorvastatin (CYP3A4 substrate)
        DrugCombination(
            drugs=[drugs["carbamazepine"], drugs["atorvastatin"]],
            doses=[400.0, 20.0],
            dosing_intervals=[12.0, 24.0],
            is_safe=False,
            interaction_type="cyp_induction",
            severity="moderate",
            clinical_evidence="Drug Metab Dispos 2004; CYP3A4 induction"
        ),
        # 6. Amiodarone (CYP3A4/2D6/2C9 inhibitor) + Warfarin (CYP2C9 sub)
        DrugCombination(
            drugs=[drugs["amiodarone"], drugs["warfarin"]],
            doses=[200.0, 5.0],
            dosing_intervals=[24.0, 24.0],
            is_safe=False,
            interaction_type="cyp_inhibition",
            severity="severe",
            clinical_evidence="Well-known CYP2C9 inhibition; INR elevation"
        ),

        # --- 4 safe (no CYP overlap → baseline correctly says safe) ---

        DrugCombination(
            drugs=[drugs["metformin"], drugs["lisinopril"], drugs["amlodipine"]],
            doses=[1000.0, 20.0, 5.0],
            dosing_intervals=[12.0, 24.0, 24.0],
            is_safe=True,
            interaction_type="no_interaction",
            severity="mild",
            clinical_evidence="Standard diabetes+HTN regimen; no CYP overlap"
        ),
        DrugCombination(
            drugs=[drugs["gabapentin"], drugs["metformin"]],
            doses=[300.0, 500.0],
            dosing_intervals=[8.0, 12.0],
            is_safe=True,
            interaction_type="no_interaction",
            severity="mild",
            clinical_evidence="Both renally eliminated; no PK or PD overlap"
        ),
        DrugCombination(
            drugs=[drugs["metformin"], drugs["lisinopril"],
                   drugs["atorvastatin"], drugs["aspirin"]],
            doses=[1000.0, 20.0, 20.0, 81.0],
            dosing_intervals=[12.0, 24.0, 24.0, 24.0],
            is_safe=True,
            interaction_type="no_significant_interaction",
            severity="mild",
            clinical_evidence="Standard cardiovascular poly-pill"
        ),
        DrugCombination(
            drugs=[drugs["amlodipine"], drugs["metformin"],
                   drugs["gabapentin"], drugs["lisinopril"]],
            doses=[5.0, 500.0, 300.0, 10.0],
            dosing_intervals=[24.0, 12.0, 8.0, 24.0],
            is_safe=True,
            interaction_type="no_interaction",
            severity="mild",
            clinical_evidence="HTN+diabetes+neuropathy; well-tolerated regimen"
        ),
    ]


# ---------------------------------------------------------------------------
# 20 NEW PD-diverse scenarios
# ---------------------------------------------------------------------------

def create_pd_scenarios(drugs: Dict[str, Drug]) -> List[DrugCombination]:
    """
    20 PD-diverse scenarios.  Dangerous ones are NOT detectable by
    CYP-overlap because the interacting drugs share no CYP pathway.
    """
    scenarios = []

    # ====== SEROTONIN SYNDROME (5 dangerous) ======

    # PD-1: Linezolid + Sertraline — MAOI + SSRI, no CYP overlap at all
    scenarios.append(DrugCombination(
        drugs=[drugs["linezolid"], drugs["sertraline"]],
        doses=[600.0, 50.0],
        dosing_intervals=[12.0, 24.0],
        is_safe=False,
        interaction_type="serotonin_syndrome",
        severity="severe",
        clinical_evidence="FDA Black Box Warning on linezolid + serotonergic"
    ))

    # PD-2: Lithium + Fluoxetine — serotonin syndrome risk, lithium has
    #   zero CYP involvement
    scenarios.append(DrugCombination(
        drugs=[drugs["lithium"], drugs["fluoxetine"]],
        doses=[900.0, 20.0],
        dosing_intervals=[8.0, 24.0],
        is_safe=False,
        interaction_type="serotonin_syndrome",
        severity="severe",
        clinical_evidence="Case reports of serotonin syndrome; lithium "
                          "augments serotonergic transmission"
    ))

    # PD-4: Trazodone + Buspirone + Tramadol — triple serotonergic
    scenarios.append(DrugCombination(
        drugs=[drugs["trazodone"], drugs["buspirone"], drugs["tramadol"]],
        doses=[100.0, 15.0, 50.0],
        dosing_intervals=[24.0, 8.0, 6.0],
        is_safe=False,
        interaction_type="serotonin_syndrome",
        severity="severe",
        clinical_evidence="Multiple serotonergic agents; additive risk"
    ))

    # PD-5: Linezolid + Tramadol — MAOI-like + serotonergic opioid,
    #   no CYP overlap
    scenarios.append(DrugCombination(
        drugs=[drugs["linezolid"], drugs["tramadol"]],
        doses=[600.0, 50.0],
        dosing_intervals=[12.0, 6.0],
        is_safe=False,
        interaction_type="serotonin_syndrome",
        severity="severe",
        clinical_evidence="Linezolid reversible MAOI; tramadol SNRI activity"
    ))

    # ====== CNS DEPRESSION (5 dangerous) ======

    # PD-6: Gabapentin + Oxycodone — FDA boxed warning 2019
    scenarios.append(DrugCombination(
        drugs=[drugs["gabapentin"], drugs["oxycodone"]],
        doses=[600.0, 10.0],
        dosing_intervals=[8.0, 6.0],
        is_safe=False,
        interaction_type="cns_depression",
        severity="severe",
        clinical_evidence="FDA safety alert 2019; respiratory depression"
    ))

    # PD-7: Pregabalin + Alprazolam — both CNS depressants, pregabalin
    #   has zero CYP involvement
    scenarios.append(DrugCombination(
        drugs=[drugs["pregabalin"], drugs["alprazolam"]],
        doses=[150.0, 1.0],
        dosing_intervals=[8.0, 8.0],
        is_safe=False,
        interaction_type="cns_depression",
        severity="severe",
        clinical_evidence="Additive sedation; respiratory depression risk"
    ))

    # PD-8: Gabapentin + Zolpidem + Oxycodone — triple CNS depression
    scenarios.append(DrugCombination(
        drugs=[drugs["gabapentin"], drugs["zolpidem"], drugs["oxycodone"]],
        doses=[300.0, 10.0, 5.0],
        dosing_intervals=[8.0, 24.0, 6.0],
        is_safe=False,
        interaction_type="cns_depression",
        severity="severe",
        clinical_evidence="Triple CNS depressant; high respiratory risk"
    ))

    # PD-9: Pregabalin + Tramadol — CNS depression + serotonergic,
    #   no CYP overlap between these two
    scenarios.append(DrugCombination(
        drugs=[drugs["pregabalin"], drugs["tramadol"]],
        doses=[75.0, 50.0],
        dosing_intervals=[12.0, 6.0],
        is_safe=False,
        interaction_type="cns_depression",
        severity="severe",
        clinical_evidence="Additive CNS and respiratory depression"
    ))

    # PD-10: Haloperidol + Alprazolam — antipsychotic + benzo, both
    #   are CYP3A4 substrates but danger is PD (sedation), not PK
    scenarios.append(DrugCombination(
        drugs=[drugs["haloperidol"], drugs["alprazolam"]],
        doses=[5.0, 1.0],
        dosing_intervals=[12.0, 8.0],
        is_safe=False,
        interaction_type="cns_depression",
        severity="severe",
        clinical_evidence="Excessive sedation, fall risk in elderly"
    ))

    # ====== QT PROLONGATION (4 dangerous) ======

    # PD-11: Sotalol + Ciprofloxacin — both QT prolonging, zero
    #   shared CYP pathway (sotalol is renally eliminated)
    scenarios.append(DrugCombination(
        drugs=[drugs["sotalol"], drugs["ciprofloxacin"]],
        doses=[80.0, 500.0],
        dosing_intervals=[12.0, 12.0],
        is_safe=False,
        interaction_type="qt_prolongation",
        severity="severe",
        clinical_evidence="CredibleMeds QT risk; additive hERG blockade"
    ))

    # PD-12: Haloperidol + Ondansetron — both QT prolonging
    scenarios.append(DrugCombination(
        drugs=[drugs["haloperidol"], drugs["ondansetron"]],
        doses=[5.0, 8.0],
        dosing_intervals=[12.0, 8.0],
        is_safe=False,
        interaction_type="qt_prolongation",
        severity="severe",
        clinical_evidence="Both have FDA QT warning; additive risk"
    ))

    # PD-13: Sotalol + Metoclopramide — QT pair, no CYP overlap
    scenarios.append(DrugCombination(
        drugs=[drugs["sotalol"], drugs["metoclopramide"]],
        doses=[80.0, 10.0],
        dosing_intervals=[12.0, 8.0],
        is_safe=False,
        interaction_type="qt_prolongation",
        severity="severe",
        clinical_evidence="Metoclopramide FDA revision 2009; additive QT"
    ))

    # PD-14: Amiodarone + Sotalol + Ciprofloxacin — triple QT
    scenarios.append(DrugCombination(
        drugs=[drugs["amiodarone"], drugs["sotalol"],
               drugs["ciprofloxacin"]],
        doses=[200.0, 80.0, 500.0],
        dosing_intervals=[24.0, 12.0, 12.0],
        is_safe=False,
        interaction_type="qt_prolongation",
        severity="severe",
        clinical_evidence="Extreme QT prolongation risk; contraindicated"
    ))

    # ====== NEPHROTOXICITY / HYPERKALEMIA (3 dangerous) ======

    # PD-15: Lisinopril + Spironolactone + Naproxen — triple whammy
    scenarios.append(DrugCombination(
        drugs=[drugs["lisinopril"], drugs["spironolactone"],
               drugs["naproxen"]],
        doses=[20.0, 25.0, 500.0],
        dosing_intervals=[24.0, 24.0, 12.0],
        is_safe=False,
        interaction_type="nephrotoxicity",
        severity="severe",
        clinical_evidence="BMJ 2013 triple whammy; AKI risk increased 31%"
    ))

    # PD-16: Lisinopril + Spironolactone — hyperkalemia, both non-CYP
    scenarios.append(DrugCombination(
        drugs=[drugs["lisinopril"], drugs["spironolactone"]],
        doses=[20.0, 50.0],
        dosing_intervals=[24.0, 24.0],
        is_safe=False,
        interaction_type="hyperkalemia",
        severity="severe",
        clinical_evidence="RALES trial monitoring; K >5.5 risk"
    ))

    # PD-17: Lisinopril + Spironolactone + Celecoxib — nephro + K+
    scenarios.append(DrugCombination(
        drugs=[drugs["lisinopril"], drugs["spironolactone"],
               drugs["celecoxib"]],
        doses=[20.0, 25.0, 200.0],
        dosing_intervals=[24.0, 24.0, 24.0],
        is_safe=False,
        interaction_type="nephrotoxicity",
        severity="severe",
        clinical_evidence="Triple whammy variant with COX-2 inhibitor"
    ))

    # PD-18: Lithium + Naproxen — NSAID reduces renal lithium clearance;
    #   no CYP overlap (both non-CYP)
    scenarios.append(DrugCombination(
        drugs=[drugs["lithium"], drugs["naproxen"]],
        doses=[900.0, 500.0],
        dosing_intervals=[8.0, 12.0],
        is_safe=False,
        interaction_type="nephrotoxicity",
        severity="severe",
        clinical_evidence="Well-known NSAID-lithium interaction; lithium "
                          "toxicity from reduced renal clearance"
    ))

    # ====== 3 SAFE PD scenarios (diverse drugs, no dangerous PD overlap) ======

    # PD-18: Gabapentin + Metformin — both renally eliminated, no PD overlap
    scenarios.append(DrugCombination(
        drugs=[drugs["gabapentin"], drugs["metformin"]],
        doses=[300.0, 500.0],
        dosing_intervals=[8.0, 12.0],
        is_safe=True,
        interaction_type="no_interaction",
        severity="mild",
        clinical_evidence="No shared PK or PD pathway"
    ))

    # PD-19: Atorvastatin + Lisinopril + Metformin — standard CV regimen
    scenarios.append(DrugCombination(
        drugs=[drugs["atorvastatin"], drugs["lisinopril"],
               drugs["metformin"]],
        doses=[20.0, 10.0, 500.0],
        dosing_intervals=[24.0, 24.0, 12.0],
        is_safe=True,
        interaction_type="no_interaction",
        severity="mild",
        clinical_evidence="Guideline-concordant combination; well-studied"
    ))

    # PD-20: Pregabalin + Lisinopril — no PK or PD pathway overlap
    scenarios.append(DrugCombination(
        drugs=[drugs["pregabalin"], drugs["lisinopril"]],
        doses=[75.0, 10.0],
        dosing_intervals=[12.0, 24.0],
        is_safe=True,
        interaction_type="no_interaction",
        severity="mild",
        clinical_evidence="No known interaction; different pathways"
    ))

    return scenarios


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(ground_truth: List[bool],
                    predictions: List[bool]) -> Dict:
    tp = sum(1 for gt, p in zip(ground_truth, predictions) if gt and p)
    tn = sum(1 for gt, p in zip(ground_truth, predictions)
             if not gt and not p)
    fp = sum(1 for gt, p in zip(ground_truth, predictions)
             if not gt and p)
    fn = sum(1 for gt, p in zip(ground_truth, predictions)
             if gt and not p)

    n = len(ground_truth)
    accuracy = (tp + tn) / n if n > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) \
        if (precision + recall) > 0 else 0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    miss = fn / (fn + tp) if (fn + tp) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_alarm_rate": far,
        "miss_rate": miss,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_expanded_benchmark():
    """Run the expanded 30-scenario benchmark (10 original + 20 PD)."""

    print("=" * 70)
    print("  GuardPharma Expanded PD-Diversity Benchmark")
    print("=" * 70)

    drugs = create_pd_drug_database()
    original = create_original_10_scenarios(drugs)
    pd_new = create_pd_scenarios(drugs)
    all_scenarios = original + pd_new

    print(f"\nTotal scenarios : {len(all_scenarios)}")
    print(f"  Original (CYP): {len(original)}")
    print(f"  New (PD)      : {len(pd_new)}")
    n_unsafe = sum(1 for s in all_scenarios if not s.is_safe)
    n_safe = len(all_scenarios) - n_unsafe
    print(f"  Unsafe / Safe : {n_unsafe} / {n_safe}")
    print()

    # Methods
    guardpharma = GuardPharmaPDVerifier()
    cyp_baseline = CYPOverlapBaseline()

    methods = {
        "GuardPharma (Ours)": guardpharma,
        "CYP-Overlap Baseline": cyp_baseline,
    }

    # --- Run on ALL scenarios ---
    results_all = {m: [] for m in methods}
    gt_all = []

    print("Running scenarios...")
    for i, scenario in enumerate(all_scenarios):
        tag = "ORIG" if i < len(original) else "PD"
        label = "SAFE" if scenario.is_safe else "UNSAFE"
        drug_str = ", ".join(d.name for d in scenario.drugs)
        print(f"  [{tag:4s}] {i+1:2d}. {drug_str:50s}  {label}")

        gt_all.append(scenario.is_safe)
        for name, method in methods.items():
            is_safe, details = method.verify_safety(scenario)
            results_all[name].append({
                "predicted_safe": is_safe,
                "details": details,
            })

    # --- Metrics: ALL scenarios ---
    print("\n" + "=" * 70)
    print("  OVERALL RESULTS (all %d scenarios)" % len(all_scenarios))
    print("=" * 70)
    print(f"{'Method':<25s} {'Acc':>6s} {'Prec':>6s} {'Rec':>6s} "
          f"{'F1':>6s} {'FAR':>6s} {'MR':>6s}")
    print("-" * 70)

    metrics_all = {}
    for name in methods:
        preds = [r["predicted_safe"] for r in results_all[name]]
        m = compute_metrics(gt_all, preds)
        metrics_all[name] = m
        print(f"{name:<25s} {m['accuracy']:6.1%} {m['precision']:6.1%} "
              f"{m['recall']:6.1%} {m['f1_score']:6.3f} "
              f"{m['false_alarm_rate']:6.1%} {m['miss_rate']:6.1%}")

    # --- Metrics: ORIGINAL 10 only ---
    print("\n" + "=" * 70)
    print("  ORIGINAL 10 SCENARIOS (CYP-focused)")
    print("=" * 70)
    print(f"{'Method':<25s} {'Acc':>6s} {'Prec':>6s} {'Rec':>6s} "
          f"{'F1':>6s}")
    print("-" * 70)

    metrics_orig = {}
    for name in methods:
        preds = [r["predicted_safe"] for r in results_all[name][:len(original)]]
        m = compute_metrics(gt_all[:len(original)], preds)
        metrics_orig[name] = m
        print(f"{name:<25s} {m['accuracy']:6.1%} {m['precision']:6.1%} "
              f"{m['recall']:6.1%} {m['f1_score']:6.3f}")

    # --- Metrics: PD 20 only ---
    print("\n" + "=" * 70)
    print("  PD-DIVERSE SCENARIOS ONLY (%d new)" % len(pd_new))
    print("=" * 70)
    print(f"{'Method':<25s} {'Acc':>6s} {'Prec':>6s} {'Rec':>6s} "
          f"{'F1':>6s} {'FAR':>6s} {'MR':>6s}")
    print("-" * 70)

    metrics_pd = {}
    for name in methods:
        preds = [r["predicted_safe"] for r in results_all[name][len(original):]]
        gt_pd = gt_all[len(original):]
        m = compute_metrics(gt_pd, preds)
        metrics_pd[name] = m
        print(f"{name:<25s} {m['accuracy']:6.1%} {m['precision']:6.1%} "
              f"{m['recall']:6.1%} {m['f1_score']:6.3f} "
              f"{m['false_alarm_rate']:6.1%} {m['miss_rate']:6.1%}")

    # --- Per-scenario detail ---
    print("\n" + "=" * 70)
    print("  PER-SCENARIO DETAIL")
    print("=" * 70)
    print(f"{'#':>2s}  {'Drugs':45s}  {'Truth':6s}  {'GP':6s}  {'BL':6s}  "
          f"{'BL_miss':>7s}")
    print("-" * 70)

    baseline_misses = 0
    for i, scenario in enumerate(all_scenarios):
        truth = "Safe" if scenario.is_safe else "Unsafe"
        gp_pred = "Safe" if results_all["GuardPharma (Ours)"][i]["predicted_safe"] \
            else "Unsafe"
        bl_pred = "Safe" if results_all["CYP-Overlap Baseline"][i]["predicted_safe"] \
            else "Unsafe"
        gp_ok = "✓" if (gp_pred == truth) else "✗"
        bl_ok = "✓" if (bl_pred == truth) else "✗"
        bl_miss = ""
        if bl_pred != truth:
            bl_miss = "MISS"
            baseline_misses += 1
        drug_str = ", ".join(d.name for d in scenario.drugs)[:45]
        print(f"{i+1:2d}  {drug_str:45s}  {truth:6s}  "
              f"{gp_ok:6s}  {bl_ok:6s}  {bl_miss:>7s}")

    print(f"\nCYP-overlap baseline missed {baseline_misses} of "
          f"{n_unsafe} dangerous scenarios.")

    # --- Save JSON results ---
    output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_scenarios": len(all_scenarios),
            "original_scenarios": len(original),
            "pd_scenarios": len(pd_new),
            "num_unsafe": n_unsafe,
            "num_safe": n_safe,
        },
        "metrics_all": metrics_all,
        "metrics_original_10": metrics_orig,
        "metrics_pd_20": metrics_pd,
        "per_scenario": [],
    }

    for i, scenario in enumerate(all_scenarios):
        output["per_scenario"].append({
            "id": i + 1,
            "drugs": [d.name for d in scenario.drugs],
            "doses": scenario.doses,
            "is_safe": scenario.is_safe,
            "interaction_type": scenario.interaction_type,
            "severity": scenario.severity,
            "guardpharma_safe": results_all["GuardPharma (Ours)"][i]["predicted_safe"],
            "cyp_baseline_safe": results_all["CYP-Overlap Baseline"][i]["predicted_safe"],
            "guardpharma_correct": (
                results_all["GuardPharma (Ours)"][i]["predicted_safe"]
                == scenario.is_safe
            ),
            "cyp_baseline_correct": (
                results_all["CYP-Overlap Baseline"][i]["predicted_safe"]
                == scenario.is_safe
            ),
        })

    os.makedirs("benchmarks/results", exist_ok=True)
    out_path = "benchmarks/results/pd_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n💾 Results saved to {out_path}")

    return output


if __name__ == "__main__":
    run_expanded_benchmark()
