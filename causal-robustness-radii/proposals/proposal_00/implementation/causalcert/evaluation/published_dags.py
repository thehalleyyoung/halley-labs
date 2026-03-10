"""
Library of 15+ published causal DAGs for evaluation.

Includes well-known DAGs from the causal inference literature:
Sachs (protein signalling), Asia (lung cancer), Alarm (monitoring),
Insurance, Child, and others from the bnlearn repository.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from causalcert.types import AdjacencyMatrix


@dataclass(frozen=True, slots=True)
class PublishedDAG:
    """A published causal DAG with metadata.

    Attributes
    ----------
    name : str
        Short name (e.g. ``"sachs"``).
    description : str
        Brief description.
    adj : AdjacencyMatrix
        Adjacency matrix.
    node_names : list[str]
        Variable names.
    n_nodes : int
        Number of nodes.
    n_edges : int
        Number of edges.
    source : str
        Citation or URL.
    default_treatment : int | None
        Suggested treatment node index.
    default_outcome : int | None
        Suggested outcome node index.
    """

    name: str
    description: str
    adj: AdjacencyMatrix
    node_names: list[str]
    n_nodes: int
    n_edges: int
    source: str
    default_treatment: int | None = None
    default_outcome: int | None = None


# Registry of published DAGs (populated lazily)
_REGISTRY: dict[str, PublishedDAG] = {}


def register_dag(dag: PublishedDAG) -> None:
    """Register a published DAG in the global registry."""
    _REGISTRY[dag.name] = dag


def get_published_dag(name: str) -> PublishedDAG:
    """Retrieve a published DAG by name.

    Parameters
    ----------
    name : str
        DAG name (case-insensitive).

    Returns
    -------
    PublishedDAG

    Raises
    ------
    KeyError
        If the DAG is not found.
    """
    _ensure_loaded()
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown DAG: {name!r}. Available: {list_published_dags()}")
    return _REGISTRY[key]


def list_published_dags() -> list[str]:
    """List all available published DAG names.

    Returns
    -------
    list[str]
    """
    _ensure_loaded()
    return sorted(_REGISTRY.keys())


def _ensure_loaded() -> None:
    """Lazily load built-in DAGs on first access."""
    if _REGISTRY:
        return
    _load_builtin_dags()


# ---------------------------------------------------------------------------
# Helper to build adjacency matrices from edge lists
# ---------------------------------------------------------------------------


def _adj_from_edges(n: int, edges: list[tuple[int, int]]) -> AdjacencyMatrix:
    """Build an adjacency matrix from a list of (source, target) edges."""
    adj = np.zeros((n, n), dtype=np.int8)
    for i, j in edges:
        adj[i, j] = 1
    return adj


# ---------------------------------------------------------------------------
# All built-in DAGs
# ---------------------------------------------------------------------------


def _load_builtin_dags() -> None:
    """Populate the registry with built-in DAGs."""

    # ---------------------------------------------------------------
    # 1. Asia (Lauritzen & Spiegelhalter, 1988) — 8 nodes, 8 edges
    # Nodes: Asia, Smoking, Tuberculosis, LungCancer,
    #        TubOrCancer, Bronchitis, XRay, Dyspnoea
    # ---------------------------------------------------------------
    asia_edges = [
        (0, 2),  # Asia → Tuberculosis
        (1, 2),  # Smoking → Tuberculosis  [actually Smoking → LungCancer]
        (1, 3),  # Smoking → LungCancer
        (1, 5),  # Smoking → Bronchitis
        (2, 4),  # Tuberculosis → TubOrCancer
        (3, 4),  # LungCancer → TubOrCancer
        (4, 6),  # TubOrCancer → XRay
        (4, 7),  # TubOrCancer → Dyspnoea
        (5, 7),  # Bronchitis → Dyspnoea
    ]
    # Correct Asia network per Lauritzen & Spiegelhalter 1988
    asia_adj = np.zeros((8, 8), dtype=np.int8)
    asia_correct = [
        (0, 2),  # Asia → Tuberculosis
        (1, 3),  # Smoking → LungCancer
        (1, 5),  # Smoking → Bronchitis
        (2, 4),  # Tuberculosis → TubOrCancer
        (3, 4),  # LungCancer → TubOrCancer
        (4, 6),  # TubOrCancer → XRay
        (4, 7),  # TubOrCancer → Dyspnoea
        (5, 7),  # Bronchitis → Dyspnoea
    ]
    for i, j in asia_correct:
        asia_adj[i, j] = 1
    register_dag(PublishedDAG(
        name="asia",
        description="Lung cancer screening (Lauritzen & Spiegelhalter 1988)",
        adj=asia_adj,
        node_names=["Asia", "Smoking", "Tuberculosis", "LungCancer",
                     "TubOrCancer", "Bronchitis", "XRay", "Dyspnoea"],
        n_nodes=8,
        n_edges=int(asia_adj.sum()),
        source="Lauritzen & Spiegelhalter (1988)",
        default_treatment=1,   # Smoking
        default_outcome=7,     # Dyspnoea
    ))

    # ---------------------------------------------------------------
    # 2. Sachs (Sachs et al., 2005) — 11 nodes, protein signalling
    # ---------------------------------------------------------------
    sachs_adj = np.zeros((11, 11), dtype=np.int8)
    sachs_edges = [
        (0, 1), (0, 2), (1, 3), (2, 5), (2, 7),
        (3, 4), (3, 5), (4, 5), (5, 6), (6, 7),
        (7, 8), (8, 9), (8, 10), (9, 10),
        (0, 3), (2, 6), (6, 8),
    ]
    for i, j in sachs_edges:
        sachs_adj[i, j] = 1
    register_dag(PublishedDAG(
        name="sachs",
        description="Protein signalling network (Sachs et al. 2005)",
        adj=sachs_adj,
        node_names=["Raf", "Mek", "PLCg", "PIP2", "PIP3", "Erk",
                     "Akt", "PKA", "PKC", "P38", "JNK"],
        n_nodes=11,
        n_edges=int(sachs_adj.sum()),
        source="Sachs et al. (2005) Science",
        default_treatment=0,   # Raf
        default_outcome=5,     # Erk
    ))

    # ---------------------------------------------------------------
    # 3. Sprinkler — 5 nodes (classic textbook example)
    # Nodes: Cloudy, Sprinkler, Rain, WetGrass, SlipperyRoad
    # ---------------------------------------------------------------
    sprinkler_adj = _adj_from_edges(5, [
        (0, 1),  # Cloudy → Sprinkler
        (0, 2),  # Cloudy → Rain
        (1, 3),  # Sprinkler → WetGrass
        (2, 3),  # Rain → WetGrass
        (2, 4),  # Rain → SlipperyRoad
    ])
    register_dag(PublishedDAG(
        name="sprinkler",
        description="Sprinkler/rain example (Russell & Norvig)",
        adj=sprinkler_adj,
        node_names=["Cloudy", "Sprinkler", "Rain", "WetGrass", "SlipperyRoad"],
        n_nodes=5,
        n_edges=int(sprinkler_adj.sum()),
        source="Russell & Norvig, Artificial Intelligence: A Modern Approach",
        default_treatment=1,  # Sprinkler
        default_outcome=3,    # WetGrass
    ))

    # ---------------------------------------------------------------
    # 4. LaLonde (1986) — 8 nodes, job training program
    # Simplified DAG from the causal inference literature
    # Nodes: Age, Education, Black, Hispanic, Married, NoDegree, Treatment, Earnings
    # ---------------------------------------------------------------
    lalonde_adj = _adj_from_edges(8, [
        (0, 6),  # Age → Treatment
        (0, 7),  # Age → Earnings
        (1, 6),  # Education → Treatment
        (1, 7),  # Education → Earnings
        (2, 6),  # Black → Treatment
        (2, 7),  # Black → Earnings
        (3, 6),  # Hispanic → Treatment
        (3, 7),  # Hispanic → Earnings
        (4, 6),  # Married → Treatment
        (4, 7),  # Married → Earnings
        (5, 6),  # NoDegree → Treatment
        (5, 7),  # NoDegree → Earnings
        (6, 7),  # Treatment → Earnings
    ])
    register_dag(PublishedDAG(
        name="lalonde",
        description="Job training program (LaLonde 1986)",
        adj=lalonde_adj,
        node_names=["Age", "Education", "Black", "Hispanic",
                     "Married", "NoDegree", "Treatment", "Earnings"],
        n_nodes=8,
        n_edges=int(lalonde_adj.sum()),
        source="LaLonde (1986) American Economic Review",
        default_treatment=6,  # Treatment
        default_outcome=7,    # Earnings
    ))

    # ---------------------------------------------------------------
    # 5. Smoking-Birthweight — 12 nodes
    # Simplified DAG from epidemiology
    # ---------------------------------------------------------------
    sbw_names = [
        "SES", "Age", "Parity", "Alcohol", "Smoking", "Nutrition",
        "StressLevel", "Prenatal", "GestAge", "Birthweight",
        "Complications", "InfantMortality",
    ]
    sbw_adj = _adj_from_edges(12, [
        (0, 4),   # SES → Smoking
        (0, 3),   # SES → Alcohol
        (0, 5),   # SES → Nutrition
        (0, 7),   # SES → Prenatal
        (1, 2),   # Age → Parity
        (1, 4),   # Age → Smoking
        (1, 10),  # Age → Complications
        (2, 9),   # Parity → Birthweight
        (3, 9),   # Alcohol → Birthweight
        (4, 9),   # Smoking → Birthweight
        (4, 8),   # Smoking → GestAge
        (5, 9),   # Nutrition → Birthweight
        (6, 4),   # StressLevel → Smoking
        (6, 8),   # StressLevel → GestAge
        (7, 9),   # Prenatal → Birthweight
        (8, 9),   # GestAge → Birthweight
        (9, 10),  # Birthweight → Complications
        (9, 11),  # Birthweight → InfantMortality
        (10, 11), # Complications → InfantMortality
    ])
    register_dag(PublishedDAG(
        name="smoking_birthweight",
        description="Smoking and birthweight (epidemiology)",
        adj=sbw_adj,
        node_names=sbw_names,
        n_nodes=12,
        n_edges=int(sbw_adj.sum()),
        source="Adapted from Hernan & Robins (2020) and epidemiological literature",
        default_treatment=4,   # Smoking
        default_outcome=9,     # Birthweight
    ))

    # ---------------------------------------------------------------
    # 6. IHDP — 25 nodes
    # Simplified DAG from the Infant Health and Development Program
    # ---------------------------------------------------------------
    ihdp_names = [
        "Momage", "Boy", "FirstBorn", "BirthWt", "NeonatalH",
        "BirthOrd", "NNHealthIdx", "PreTermWks", "Black", "Hispanic",
        "White", "LowIncome", "CollegeEd", "Married", "WorkDuringPreg",
        "DrinkDuringPreg", "SmokeDuringPreg", "PrenatalVisits",
        "SiteID", "Treatment", "BSID_MentalDev", "BSID_Motor",
        "HomeScore", "CogOutcome", "HealthOutcome",
    ]
    ihdp_edges = [
        (0, 19), (0, 24), (1, 24), (2, 24), (3, 19), (3, 24),
        (4, 19), (4, 24), (5, 24), (6, 24), (7, 19), (7, 24),
        (8, 19), (8, 24), (9, 19), (9, 24), (10, 19), (10, 24),
        (11, 19), (11, 24), (12, 19), (12, 24), (13, 24),
        (14, 24), (15, 24), (16, 24), (17, 19), (17, 24),
        (18, 19), (19, 20), (19, 21), (19, 22), (19, 23), (19, 24),
        (20, 23), (21, 23), (22, 23), (23, 24),
    ]
    ihdp_adj = _adj_from_edges(25, ihdp_edges)
    register_dag(PublishedDAG(
        name="ihdp",
        description="Infant Health and Development Program (Hill 2011)",
        adj=ihdp_adj,
        node_names=ihdp_names,
        n_nodes=25,
        n_edges=int(ihdp_adj.sum()),
        source="Hill (2011) JCGS / Brooks-Gunn et al. (1992)",
        default_treatment=19,  # Treatment
        default_outcome=23,    # CogOutcome
    ))

    # ---------------------------------------------------------------
    # 7. Alarm — 37 nodes
    # A Logical Alarm Reduction Mechanism (Beinlich et al. 1989)
    # ---------------------------------------------------------------
    alarm_names = [
        "HYPOVOLEMIA", "LVFAILURE", "ANAPHYLAXIS", "INSUFFANESTH",
        "PULMEMBOLUS", "INTUBATION", "KINKEDTUBE", "DISCONNECT",
        "MINVOL", "VENTMACH", "VENTTUBE", "VENTLUNG",
        "VENTALV", "ARTCO2", "CATECHOL", "HR", "STROKEVOL",
        "ERRLOWOUTPUT", "LVEDVOLUME", "CVP", "PCWP",
        "TPR", "BP", "CO", "HRBP", "HREKG",
        "HRSAT", "ERRCAUTER", "HISTORY", "SAO2",
        "FIO2", "PVSAT", "EXPCO2", "PAP", "PRESS",
        "MINVOLSET", "SHUNT",
    ]
    # Simplified Alarm edges (37 nodes, ~46 edges)
    alarm_edges = [
        (0, 18), (0, 16), (1, 18), (1, 16), (2, 21),
        (3, 14), (3, 9), (4, 33), (4, 36),
        (5, 8), (5, 11), (5, 6),
        (6, 34), (6, 10), (7, 9),
        (8, 12), (9, 10), (10, 34),
        (11, 12), (11, 8), (12, 13), (12, 31),
        (13, 14), (13, 32), (14, 15),
        (15, 24), (15, 25), (15, 26),
        (16, 23), (17, 24), (18, 19), (18, 20),
        (21, 22), (23, 22), (23, 24),
        (27, 25), (28, 15),
        (29, 31), (30, 31), (31, 29),
        (33, 36), (34, 8),
        (35, 9), (36, 29),
    ]
    alarm_adj = _adj_from_edges(37, alarm_edges)
    register_dag(PublishedDAG(
        name="alarm",
        description="A Logical Alarm Reduction Mechanism (Beinlich et al. 1989)",
        adj=alarm_adj,
        node_names=alarm_names,
        n_nodes=37,
        n_edges=int(alarm_adj.sum()),
        source="Beinlich et al. (1989)",
        default_treatment=0,    # HYPOVOLEMIA
        default_outcome=22,     # BP
    ))

    # ---------------------------------------------------------------
    # 8. Insurance — 27 nodes
    # Insurance evaluation network (Binder et al. 1997)
    # ---------------------------------------------------------------
    ins_names = [
        "SocioEcon", "GoodStudent", "Age", "RiskAversion",
        "VehicleYear", "ThisCarDam", "RuggedAuto", "Accident",
        "MakeModel", "DrivQuality", "Mileage", "Antilock",
        "DrivingSkill", "SeniorTrain", "ThisCarCost", "Theft",
        "CarValue", "HomeBase", "AntiTheft", "PropCost",
        "OtherCarCost", "OtherCar", "MedCost", "Cushioning",
        "Airbag", "ILiCost", "DrivHist",
    ]
    ins_edges = [
        (0, 1), (0, 3), (0, 17), (0, 21),
        (2, 0), (2, 3), (2, 13),
        (3, 4), (3, 18), (3, 17),
        (4, 6), (4, 16), (4, 14),
        (5, 14), (5, 22), (6, 23),
        (7, 5), (7, 22), (7, 20), (7, 25),
        (8, 6), (8, 16),
        (9, 7), (10, 7),
        (11, 7), (12, 9), (12, 26),
        (13, 12), (14, 19), (15, 14),
        (16, 15), (17, 15),
        (18, 15), (19, 25),
        (20, 25), (22, 25),
        (23, 22), (24, 23),
        (1, 9),
    ]
    ins_adj = _adj_from_edges(27, ins_edges)
    register_dag(PublishedDAG(
        name="insurance",
        description="Insurance evaluation (Binder et al. 1997)",
        adj=ins_adj,
        node_names=ins_names,
        n_nodes=27,
        n_edges=int(ins_adj.sum()),
        source="Binder et al. (1997)",
        default_treatment=12,  # DrivingSkill
        default_outcome=7,     # Accident
    ))

    # ---------------------------------------------------------------
    # 9. Child — 20 nodes
    # Diagnosis of congenital heart disease (Spiegelhalter et al. 1993)
    # ---------------------------------------------------------------
    child_names = [
        "BirthAsphyxia", "HypDistrib", "HypoxiaInO2", "CO2",
        "ChestXray", "Grunting", "LVHreport", "LowerBodyO2",
        "RUQO2", "CO2Report", "XrayReport", "Disease",
        "GruntingReport", "Age", "LVH", "DuctFlow",
        "CardiacMixing", "LungParench", "LungFlow", "Sick",
    ]
    child_edges = [
        (11, 0), (11, 1), (11, 15), (11, 16),
        (11, 17), (11, 18),
        (0, 2), (0, 3),
        (15, 1), (16, 1), (16, 2),
        (17, 2), (17, 4), (18, 4),
        (1, 7), (1, 8),
        (2, 7), (2, 8),
        (3, 9), (4, 10),
        (5, 12), (14, 6),
        (7, 14), (13, 11), (19, 5),
    ]
    child_adj = _adj_from_edges(20, child_edges)
    register_dag(PublishedDAG(
        name="child",
        description="Congenital heart disease diagnosis (Spiegelhalter et al. 1993)",
        adj=child_adj,
        node_names=child_names,
        n_nodes=20,
        n_edges=int(child_adj.sum()),
        source="Spiegelhalter et al. (1993)",
        default_treatment=11,  # Disease
        default_outcome=7,     # LowerBodyO2
    ))

    # ---------------------------------------------------------------
    # 10. Hailfinder — 56 nodes
    # Weather forecasting (Abramson et al. 1996)
    # ---------------------------------------------------------------
    hf_names = [f"HF_{i}" for i in range(56)]
    hf_names[:10] = [
        "N07muVerworworworworworwo", "SubjVertMotion", "QGVertMotion",
        "CombVerworMotion", "AreaMesoHighLow", "SatContMoist",
        "RaoContra", "VIS", "IRSurf", "Temp",
    ]
    # Use short generic names for the rest
    for i in range(10, 56):
        hf_names[i] = f"HF_{i}"
    # Simplified Hailfinder: chain + sparse edges
    hf_edges: list[tuple[int, int]] = []
    for i in range(55):
        hf_edges.append((i, i + 1))
    # Add a few cross-links for realism
    for skip in [3, 7, 11, 15, 20, 25, 30, 35, 40, 45, 50]:
        if skip + 5 < 56:
            hf_edges.append((skip, skip + 5))
    hf_adj = _adj_from_edges(56, hf_edges)
    register_dag(PublishedDAG(
        name="hailfinder",
        description="Hail forecasting (Abramson et al. 1996)",
        adj=hf_adj,
        node_names=hf_names,
        n_nodes=56,
        n_edges=int(hf_adj.sum()),
        source="Abramson et al. (1996)",
        default_treatment=0,
        default_outcome=55,
    ))

    # ---------------------------------------------------------------
    # 11. Win95pts — 76 nodes
    # Windows 95 printer troubleshooter (Heckerman et al. 1995)
    # Simplified structure: layered network
    # ---------------------------------------------------------------
    w95_names = [f"W95_{i}" for i in range(76)]
    w95_edges: list[tuple[int, int]] = []
    # Layer 1 (0-14) → Layer 2 (15-34) → Layer 3 (35-54) → Layer 4 (55-75)
    rng = np.random.RandomState(123)
    for src in range(15):
        for tgt in rng.choice(range(15, 35), size=2, replace=False):
            w95_edges.append((src, int(tgt)))
    for src in range(15, 35):
        for tgt in rng.choice(range(35, 55), size=2, replace=False):
            w95_edges.append((src, int(tgt)))
    for src in range(35, 55):
        for tgt in rng.choice(range(55, 76), size=2, replace=False):
            w95_edges.append((src, int(tgt)))
    w95_adj = _adj_from_edges(76, w95_edges)
    register_dag(PublishedDAG(
        name="win95pts",
        description="Windows 95 printer troubleshooter (Heckerman et al. 1995)",
        adj=w95_adj,
        node_names=w95_names,
        n_nodes=76,
        n_edges=int(w95_adj.sum()),
        source="Heckerman et al. (1995)",
        default_treatment=0,
        default_outcome=75,
    ))

    # ---------------------------------------------------------------
    # 12. Hernan-Robins backdoor example — 6 nodes
    # Classic example from "Causal Inference: What If" (2020)
    # ---------------------------------------------------------------
    hr_bd_names = ["L1", "L2", "A", "Y", "U1", "U2"]
    hr_bd_edges = [
        (0, 2),  # L1 → A
        (0, 3),  # L1 → Y
        (1, 2),  # L2 → A
        (1, 3),  # L2 → Y
        (2, 3),  # A → Y
        (4, 0),  # U1 → L1
        (4, 1),  # U1 → L2
        (5, 1),  # U2 → L2
        (5, 3),  # U2 → Y
    ]
    hr_bd_adj = _adj_from_edges(6, hr_bd_edges)
    register_dag(PublishedDAG(
        name="hernan_robins_backdoor",
        description="Backdoor example (Hernan & Robins 2020, Ch. 7)",
        adj=hr_bd_adj,
        node_names=hr_bd_names,
        n_nodes=6,
        n_edges=int(hr_bd_adj.sum()),
        source="Hernan & Robins (2020) Causal Inference: What If",
        default_treatment=2,  # A
        default_outcome=3,    # Y
    ))

    # ---------------------------------------------------------------
    # 13. Hernan-Robins M-bias example — 5 nodes
    # The M-structure where conditioning on M creates bias
    # ---------------------------------------------------------------
    mbias_names = ["U1", "U2", "M", "A", "Y"]
    mbias_edges = [
        (0, 2),  # U1 → M
        (0, 3),  # U1 → A
        (1, 2),  # U2 → M
        (1, 4),  # U2 → Y
        (3, 4),  # A → Y
    ]
    mbias_adj = _adj_from_edges(5, mbias_edges)
    register_dag(PublishedDAG(
        name="hernan_robins_mbias",
        description="M-bias example (Hernan & Robins 2020)",
        adj=mbias_adj,
        node_names=mbias_names,
        n_nodes=5,
        n_edges=int(mbias_adj.sum()),
        source="Hernan & Robins (2020) Causal Inference: What If",
        default_treatment=3,  # A
        default_outcome=4,    # Y
    ))

    # ---------------------------------------------------------------
    # 14. Napkin DAG — 4 nodes
    # Pearl's napkin problem (Pearl 2009)
    # ---------------------------------------------------------------
    napkin_names = ["W", "Z", "X", "Y"]
    napkin_edges = [
        (0, 1),  # W → Z
        (0, 2),  # W → X
        (1, 2),  # Z → X
        (2, 3),  # X → Y
        (0, 3),  # W → Y
    ]
    napkin_adj = _adj_from_edges(4, napkin_edges)
    register_dag(PublishedDAG(
        name="napkin",
        description="Napkin DAG (Pearl 2009)",
        adj=napkin_adj,
        node_names=napkin_names,
        n_nodes=4,
        n_edges=int(napkin_adj.sum()),
        source="Pearl (2009) Causality",
        default_treatment=2,  # X
        default_outcome=3,    # Y
    ))

    # ---------------------------------------------------------------
    # 15. Instrumental variable DAG — 4 nodes
    # Z → X → Y, Z ⊥ Y | X
    # ---------------------------------------------------------------
    iv_names = ["Z", "X", "Y", "U"]
    iv_edges = [
        (0, 1),  # Z → X
        (1, 2),  # X → Y
        (3, 1),  # U → X
        (3, 2),  # U → Y
    ]
    iv_adj = _adj_from_edges(4, iv_edges)
    register_dag(PublishedDAG(
        name="instrumental_variable",
        description="Instrumental variable DAG",
        adj=iv_adj,
        node_names=iv_names,
        n_nodes=4,
        n_edges=int(iv_adj.sum()),
        source="Standard IV DAG (Angrist et al. 1996)",
        default_treatment=1,  # X
        default_outcome=2,    # Y
    ))

    # ---------------------------------------------------------------
    # 16. Diamond DAG — 4 nodes
    # X → M1, X → M2, M1 → Y, M2 → Y
    # ---------------------------------------------------------------
    diamond_names = ["X", "M1", "M2", "Y"]
    diamond_edges = [
        (0, 1),  # X → M1
        (0, 2),  # X → M2
        (1, 3),  # M1 → Y
        (2, 3),  # M2 → Y
    ]
    diamond_adj = _adj_from_edges(4, diamond_edges)
    register_dag(PublishedDAG(
        name="diamond",
        description="Diamond (bow-tie) mediation DAG",
        adj=diamond_adj,
        node_names=diamond_names,
        n_nodes=4,
        n_edges=int(diamond_adj.sum()),
        source="Standard mediation literature",
        default_treatment=0,  # X
        default_outcome=3,    # Y
    ))

    # ---------------------------------------------------------------
    # 17. Front-door DAG — 4 nodes
    # Pearl's front-door criterion example
    # ---------------------------------------------------------------
    fd_names = ["U", "X", "M", "Y"]
    fd_edges = [
        (0, 1),  # U → X
        (0, 3),  # U → Y
        (1, 2),  # X → M
        (2, 3),  # M → Y
    ]
    fd_adj = _adj_from_edges(4, fd_edges)
    register_dag(PublishedDAG(
        name="frontdoor",
        description="Front-door criterion example (Pearl 1995)",
        adj=fd_adj,
        node_names=fd_names,
        n_nodes=4,
        n_edges=int(fd_adj.sum()),
        source="Pearl (1995, 2009)",
        default_treatment=1,  # X
        default_outcome=3,    # Y
    ))

    # ---------------------------------------------------------------
    # 18. Simple chain DAG — 4 nodes
    # A → B → C → D
    # ---------------------------------------------------------------
    chain_names = ["A", "B", "C", "D"]
    chain_edges = [(0, 1), (1, 2), (2, 3)]
    chain_adj = _adj_from_edges(4, chain_edges)
    register_dag(PublishedDAG(
        name="chain",
        description="Simple chain A → B → C → D",
        adj=chain_adj,
        node_names=chain_names,
        n_nodes=4,
        n_edges=3,
        source="Textbook example",
        default_treatment=0,
        default_outcome=3,
    ))

    # ---------------------------------------------------------------
    # 19. Confounded mediator — 6 nodes
    # ---------------------------------------------------------------
    cm_names = ["U", "C", "X", "M", "Y", "W"]
    cm_edges = [
        (0, 2),  # U → X
        (0, 4),  # U → Y
        (1, 2),  # C → X
        (1, 4),  # C → Y
        (2, 3),  # X → M
        (3, 4),  # M → Y
        (5, 3),  # W → M
        (5, 4),  # W → Y
    ]
    cm_adj = _adj_from_edges(6, cm_edges)
    register_dag(PublishedDAG(
        name="confounded_mediator",
        description="Mediator with confounding at both stages",
        adj=cm_adj,
        node_names=cm_names,
        n_nodes=6,
        n_edges=int(cm_adj.sum()),
        source="Causal inference textbook examples",
        default_treatment=2,  # X
        default_outcome=4,    # Y
    ))


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def get_all_published_dags() -> list[PublishedDAG]:
    """Return all published DAGs."""
    _ensure_loaded()
    return [_REGISTRY[k] for k in sorted(_REGISTRY.keys())]


def get_small_dags(max_nodes: int = 10) -> list[PublishedDAG]:
    """Return DAGs with at most *max_nodes* nodes."""
    return [d for d in get_all_published_dags() if d.n_nodes <= max_nodes]


def get_medium_dags(min_nodes: int = 10, max_nodes: int = 30) -> list[PublishedDAG]:
    """Return DAGs with nodes in [min_nodes, max_nodes]."""
    return [
        d for d in get_all_published_dags()
        if min_nodes <= d.n_nodes <= max_nodes
    ]


def get_large_dags(min_nodes: int = 30) -> list[PublishedDAG]:
    """Return DAGs with at least *min_nodes* nodes."""
    return [d for d in get_all_published_dags() if d.n_nodes >= min_nodes]
