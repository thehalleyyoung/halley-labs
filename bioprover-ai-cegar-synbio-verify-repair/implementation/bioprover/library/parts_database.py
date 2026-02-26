"""Curated biological parts database for BioProver.

Provides a searchable, in-memory database of well-characterised genetic parts
(promoters, RBSs, CDSs, terminators) drawn from the iGEM Registry and primary
literature.  Each part carries transfer-function parameters (Hill kinetics),
uncertainty ranges, organism compatibility, and external database references.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class PartType(Enum):
    """Classification of genetic parts."""

    PROMOTER = auto()
    RBS = auto()
    CDS = auto()
    TERMINATOR = auto()


class Organism(Enum):
    """Host organisms for part characterisation."""

    E_COLI = "Escherichia coli"
    B_SUBTILIS = "Bacillus subtilis"
    S_CEREVISIAE = "Saccharomyces cerevisiae"
    MAMMALIAN = "Mammalian (generic)"


# ---------------------------------------------------------------------------
# Supporting dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PartReference:
    """External database cross-reference for a biological part.

    Attributes:
        database:   Registry name (e.g. ``"iGEM"``, ``"SynBioHub"``).
        identifier: Accession / part name within the registry.
        url:        Optional direct URL to the entry.
    """

    database: str
    identifier: str
    url: Optional[str] = None

    def __repr__(self) -> str:
        return f"{self.database}:{self.identifier}"


@dataclass
class HillParameters:
    """Transfer-function parameters under Hill kinetics.

    Models activation as  Vmax · x^n / (K^n + x^n)
    or repression as      Vmax · K^n / (K^n + x^n).

    Attributes:
        Vmax:       Maximum expression rate (nM/min or AU/min).
        K:          Half-activation / half-repression concentration (nM).
        n:          Hill coefficient (cooperativity).
        Vmax_range: (low, high) uncertainty interval for Vmax.
        K_range:    (low, high) uncertainty interval for K.
        n_range:    (low, high) uncertainty interval for n.
    """

    Vmax: float
    K: float
    n: float = 2.0
    Vmax_range: Tuple[float, float] = (0.0, 0.0)
    K_range: Tuple[float, float] = (0.0, 0.0)
    n_range: Tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if self.Vmax_range == (0.0, 0.0):
            self.Vmax_range = (self.Vmax * 0.8, self.Vmax * 1.2)
        if self.K_range == (0.0, 0.0):
            self.K_range = (self.K * 0.5, self.K * 2.0)
        if self.n_range == (0.0, 0.0):
            self.n_range = (max(1.0, self.n - 0.5), self.n + 0.5)


@dataclass
class PartCharacterisation:
    """Characterised performance metrics for a biological part.

    Attributes:
        rpu:              Relative promoter units (promoters only).
        translation_rate: Translation initiation rate (RBSs only), AU.
        degradation_rate: Protein degradation rate constant (1/min).
        hill:             Hill-kinetics parameters (regulated promoters).
        measurement_conditions: Free-text experimental conditions.
    """

    rpu: Optional[float] = None
    translation_rate: Optional[float] = None
    degradation_rate: Optional[float] = None
    hill: Optional[HillParameters] = None
    measurement_conditions: str = ""


# ---------------------------------------------------------------------------
# BiologicalPart
# ---------------------------------------------------------------------------

@dataclass
class BiologicalPart:
    """A single characterised genetic part.

    Attributes:
        name:           Short identifier (e.g. ``"pTet"``).
        part_type:      Functional classification.
        description:    Human-readable summary.
        sequence:       DNA sequence (may be empty).
        organisms:      Set of organisms where the part is characterised.
        characterisation: Performance metrics.
        references:     External database links.
        family:         Optional family name grouping related parts.
        metadata:       Arbitrary extra information.
    """

    name: str
    part_type: PartType
    description: str = ""
    sequence: str = ""
    organisms: Set[Organism] = field(default_factory=lambda: {Organism.E_COLI})
    characterisation: PartCharacterisation = field(
        default_factory=PartCharacterisation
    )
    references: List[PartReference] = field(default_factory=list)
    family: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- convenience --------------------------------------------------------

    def has_hill_parameters(self) -> bool:
        """Return ``True`` if Hill transfer-function parameters are present."""
        return self.characterisation.hill is not None

    def compatible_with(self, organism: Organism) -> bool:
        """Check whether the part has been characterised for *organism*."""
        return organism in self.organisms

    def add_reference(
        self, database: str, identifier: str, url: Optional[str] = None
    ) -> None:
        """Append an external database reference."""
        self.references.append(PartReference(database, identifier, url))

    def copy(self) -> BiologicalPart:
        """Return a deep copy of this part."""
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# PartsDatabase
# ---------------------------------------------------------------------------

class PartsDatabase:
    """In-memory database of characterised biological parts.

    The database ships with a built-in library of ~20 well-characterised
    parts drawn from the iGEM Registry and primary literature.  Users may
    add, remove, and query parts by type, organism, parameter range, or
    family.
    """

    def __init__(self, *, load_builtins: bool = True) -> None:
        self._parts: Dict[str, BiologicalPart] = {}
        if load_builtins:
            self._load_builtin_parts()

    # -- CRUD ---------------------------------------------------------------

    def add_part(self, part: BiologicalPart) -> None:
        """Register a part.  Raises ``ValueError`` on name collision."""
        if part.name in self._parts:
            raise ValueError(f"Part '{part.name}' already in database")
        self._parts[part.name] = part

    def get_part(self, name: str) -> Optional[BiologicalPart]:
        """Retrieve a part by name (``None`` if absent)."""
        return self._parts.get(name)

    def remove_part(self, name: str) -> None:
        """Remove a part by name.  No-op if absent."""
        self._parts.pop(name, None)

    @property
    def all_parts(self) -> List[BiologicalPart]:
        """Return a list of all registered parts."""
        return list(self._parts.values())

    @property
    def part_names(self) -> List[str]:
        """Sorted list of all part names."""
        return sorted(self._parts)

    def __len__(self) -> int:
        return len(self._parts)

    def __contains__(self, name: str) -> bool:
        return name in self._parts

    # -- queries ------------------------------------------------------------

    def query_by_type(self, part_type: PartType) -> List[BiologicalPart]:
        """Return all parts matching the given *part_type*."""
        return [p for p in self._parts.values() if p.part_type == part_type]

    def query_by_organism(self, organism: Organism) -> List[BiologicalPart]:
        """Return parts characterised for *organism*."""
        return [p for p in self._parts.values() if organism in p.organisms]

    def query_by_family(self, family: str) -> List[BiologicalPart]:
        """Return all parts belonging to *family*."""
        return [p for p in self._parts.values() if p.family == family]

    def query_promoters_by_strength(
        self, min_rpu: float = 0.0, max_rpu: float = float("inf")
    ) -> List[BiologicalPart]:
        """Return promoters whose RPU falls in [*min_rpu*, *max_rpu*]."""
        results: List[BiologicalPart] = []
        for p in self._parts.values():
            if p.part_type != PartType.PROMOTER:
                continue
            rpu = p.characterisation.rpu
            if rpu is not None and min_rpu <= rpu <= max_rpu:
                results.append(p)
        return sorted(results, key=lambda x: x.characterisation.rpu or 0)

    def query_by_hill_K(
        self, min_K: float = 0.0, max_K: float = float("inf")
    ) -> List[BiologicalPart]:
        """Return parts whose Hill K lies in [*min_K*, *max_K*]."""
        results: List[BiologicalPart] = []
        for p in self._parts.values():
            h = p.characterisation.hill
            if h is not None and min_K <= h.K <= max_K:
                results.append(p)
        return results

    # -- compatibility checking ---------------------------------------------

    def check_compatibility(
        self, part_names: Sequence[str], organism: Organism
    ) -> Dict[str, bool]:
        """Check whether each named part is compatible with *organism*.

        Returns a dict mapping part name to compatibility flag.  Parts not
        found in the database are marked ``False``.
        """
        result: Dict[str, bool] = {}
        for name in part_names:
            part = self._parts.get(name)
            result[name] = part.compatible_with(organism) if part else False
        return result

    def compatible_set(
        self, organism: Organism, part_types: Optional[Sequence[PartType]] = None
    ) -> List[BiologicalPart]:
        """Return all parts compatible with *organism*, optionally filtered
        by *part_types*."""
        results: List[BiologicalPart] = []
        for p in self._parts.values():
            if organism not in p.organisms:
                continue
            if part_types and p.part_type not in part_types:
                continue
            results.append(p)
        return results

    # -- parameter lookup ---------------------------------------------------

    def get_hill_parameters(self, name: str) -> Optional[HillParameters]:
        """Return Hill parameters for a named part, or ``None``."""
        part = self._parts.get(name)
        if part is None:
            return None
        return part.characterisation.hill

    def get_parameter_with_uncertainty(
        self, name: str, param: str
    ) -> Optional[Tuple[float, Tuple[float, float]]]:
        """Return ``(nominal, (lo, hi))`` for a Hill parameter of part *name*.

        *param* is one of ``"Vmax"``, ``"K"``, ``"n"``.
        """
        hill = self.get_hill_parameters(name)
        if hill is None:
            return None
        if param == "Vmax":
            return (hill.Vmax, hill.Vmax_range)
        if param == "K":
            return (hill.K, hill.K_range)
        if param == "n":
            return (hill.n, hill.n_range)
        return None

    # -- family helpers -----------------------------------------------------

    def families(self) -> Dict[str, List[str]]:
        """Return a dict mapping family name to member part names."""
        fam: Dict[str, List[str]] = {}
        for p in self._parts.values():
            if p.family:
                fam.setdefault(p.family, []).append(p.name)
        return fam

    # -- built-in library ---------------------------------------------------

    def _load_builtin_parts(self) -> None:  # noqa: C901 (long but flat)
        """Populate the database with ~20 well-characterised parts."""

        # ---- Promoters (repressible / inducible) --------------------------

        self._parts["pTet"] = BiologicalPart(
            name="pTet",
            part_type=PartType.PROMOTER,
            description="TetR-repressible promoter; induced by aTc",
            organisms={Organism.E_COLI},
            characterisation=PartCharacterisation(
                rpu=1.0,
                hill=HillParameters(
                    Vmax=5.0, K=10.0, n=2.0,
                    Vmax_range=(4.0, 6.0), K_range=(5.0, 20.0),
                    n_range=(1.5, 2.5),
                ),
                measurement_conditions="E. coli MG1655, 37°C, LB",
            ),
            references=[
                PartReference("iGEM", "BBa_R0040",
                              "http://parts.igem.org/Part:BBa_R0040"),
            ],
            family="tet_promoters",
        )

        self._parts["pLac"] = BiologicalPart(
            name="pLac",
            part_type=PartType.PROMOTER,
            description="LacI-repressible promoter; induced by IPTG",
            organisms={Organism.E_COLI},
            characterisation=PartCharacterisation(
                rpu=2.0,
                hill=HillParameters(
                    Vmax=8.0, K=15.0, n=2.0,
                    Vmax_range=(6.0, 10.0), K_range=(8.0, 30.0),
                    n_range=(1.5, 2.5),
                ),
                measurement_conditions="E. coli MG1655, 37°C, LB",
            ),
            references=[
                PartReference("iGEM", "BBa_R0010",
                              "http://parts.igem.org/Part:BBa_R0010"),
            ],
            family="lac_promoters",
        )

        self._parts["pBad"] = BiologicalPart(
            name="pBad",
            part_type=PartType.PROMOTER,
            description="AraC-regulated promoter; induced by arabinose",
            organisms={Organism.E_COLI},
            characterisation=PartCharacterisation(
                rpu=3.5,
                hill=HillParameters(
                    Vmax=10.0, K=20.0, n=1.5,
                    Vmax_range=(8.0, 12.0), K_range=(10.0, 40.0),
                    n_range=(1.0, 2.0),
                ),
                measurement_conditions="E. coli MG1655, 37°C, LB",
            ),
            references=[
                PartReference("iGEM", "BBa_I0500",
                              "http://parts.igem.org/Part:BBa_I0500"),
            ],
            family="ara_promoters",
        )

        self._parts["pLux"] = BiologicalPart(
            name="pLux",
            part_type=PartType.PROMOTER,
            description="LuxR-activated promoter; quorum-sensing input",
            organisms={Organism.E_COLI},
            characterisation=PartCharacterisation(
                rpu=1.5,
                hill=HillParameters(
                    Vmax=6.0, K=12.0, n=2.0,
                    Vmax_range=(4.5, 7.5), K_range=(6.0, 25.0),
                    n_range=(1.5, 2.5),
                ),
            ),
            references=[
                PartReference("iGEM", "BBa_R0062",
                              "http://parts.igem.org/Part:BBa_R0062"),
            ],
            family="lux_promoters",
        )

        self._parts["pLambdaCI"] = BiologicalPart(
            name="pLambdaCI",
            part_type=PartType.PROMOTER,
            description="Lambda-CI repressible promoter (pR)",
            organisms={Organism.E_COLI},
            characterisation=PartCharacterisation(
                rpu=1.2,
                hill=HillParameters(
                    Vmax=5.5, K=11.0, n=2.5,
                    Vmax_range=(4.0, 7.0), K_range=(6.0, 22.0),
                    n_range=(2.0, 3.0),
                ),
            ),
            references=[
                PartReference("iGEM", "BBa_R0051",
                              "http://parts.igem.org/Part:BBa_R0051"),
            ],
            family="lambda_promoters",
        )

        self._parts["pTac"] = BiologicalPart(
            name="pTac",
            part_type=PartType.PROMOTER,
            description="Hybrid trp/lac promoter; IPTG-inducible, strong",
            organisms={Organism.E_COLI},
            characterisation=PartCharacterisation(
                rpu=4.0,
                hill=HillParameters(
                    Vmax=12.0, K=18.0, n=2.0,
                    Vmax_range=(9.0, 15.0), K_range=(10.0, 30.0),
                    n_range=(1.5, 2.5),
                ),
            ),
            references=[
                PartReference("iGEM", "BBa_K180000"),
            ],
            family="lac_promoters",
        )

        # ---- Constitutive promoters (Anderson collection, Cello) ----------

        self._parts["J23100"] = BiologicalPart(
            name="J23100",
            part_type=PartType.PROMOTER,
            description="Anderson promoter, strongest in family (RPU ≈ 1.0 ref)",
            organisms={Organism.E_COLI},
            characterisation=PartCharacterisation(rpu=1.0),
            references=[
                PartReference("iGEM", "BBa_J23100",
                              "http://parts.igem.org/Part:BBa_J23100"),
            ],
            family="anderson_promoters",
        )

        self._parts["J23101"] = BiologicalPart(
            name="J23101",
            part_type=PartType.PROMOTER,
            description="Anderson promoter, strong (RPU ≈ 0.70)",
            organisms={Organism.E_COLI},
            characterisation=PartCharacterisation(rpu=0.70),
            references=[
                PartReference("iGEM", "BBa_J23101"),
            ],
            family="anderson_promoters",
        )

        self._parts["J23106"] = BiologicalPart(
            name="J23106",
            part_type=PartType.PROMOTER,
            description="Anderson promoter, medium (RPU ≈ 0.47)",
            organisms={Organism.E_COLI},
            characterisation=PartCharacterisation(rpu=0.47),
            references=[
                PartReference("iGEM", "BBa_J23106"),
            ],
            family="anderson_promoters",
        )

        self._parts["J23116"] = BiologicalPart(
            name="J23116",
            part_type=PartType.PROMOTER,
            description="Anderson promoter, weak (RPU ≈ 0.16)",
            organisms={Organism.E_COLI},
            characterisation=PartCharacterisation(rpu=0.16),
            references=[
                PartReference("iGEM", "BBa_J23116"),
            ],
            family="anderson_promoters",
        )

        # ---- RBS ----------------------------------------------------------

        self._parts["B0034"] = BiologicalPart(
            name="B0034",
            part_type=PartType.RBS,
            description="Strong RBS (Elowitz & Leibler, 2000)",
            organisms={Organism.E_COLI},
            characterisation=PartCharacterisation(translation_rate=1.0),
            references=[
                PartReference("iGEM", "BBa_B0034",
                              "http://parts.igem.org/Part:BBa_B0034"),
            ],
            family="standard_rbs",
        )

        self._parts["B0032"] = BiologicalPart(
            name="B0032",
            part_type=PartType.RBS,
            description="Medium-strength RBS",
            organisms={Organism.E_COLI},
            characterisation=PartCharacterisation(translation_rate=0.3),
            references=[
                PartReference("iGEM", "BBa_B0032"),
            ],
            family="standard_rbs",
        )

        self._parts["B0031"] = BiologicalPart(
            name="B0031",
            part_type=PartType.RBS,
            description="Weak RBS",
            organisms={Organism.E_COLI},
            characterisation=PartCharacterisation(translation_rate=0.07),
            references=[
                PartReference("iGEM", "BBa_B0031"),
            ],
            family="standard_rbs",
        )

        # ---- Coding sequences (repressors / reporters) --------------------

        self._parts["TetR"] = BiologicalPart(
            name="TetR",
            part_type=PartType.CDS,
            description="TetR repressor protein CDS",
            organisms={Organism.E_COLI},
            characterisation=PartCharacterisation(
                degradation_rate=0.0231,
                measurement_conditions="E. coli, 37°C; t½ ≈ 30 min",
            ),
            references=[
                PartReference("iGEM", "BBa_C0040",
                              "http://parts.igem.org/Part:BBa_C0040"),
            ],
            family="tet_repressors",
        )

        self._parts["LacI"] = BiologicalPart(
            name="LacI",
            part_type=PartType.CDS,
            description="LacI repressor protein CDS",
            organisms={Organism.E_COLI},
            characterisation=PartCharacterisation(
                degradation_rate=0.0116,
                measurement_conditions="E. coli, 37°C; t½ ≈ 60 min",
            ),
            references=[
                PartReference("iGEM", "BBa_C0012",
                              "http://parts.igem.org/Part:BBa_C0012"),
            ],
            family="lac_repressors",
        )

        self._parts["LambdaCI"] = BiologicalPart(
            name="LambdaCI",
            part_type=PartType.CDS,
            description="Lambda phage CI repressor CDS",
            organisms={Organism.E_COLI},
            characterisation=PartCharacterisation(
                degradation_rate=0.0173,
                measurement_conditions="E. coli, 37°C; t½ ≈ 40 min",
            ),
            references=[
                PartReference("iGEM", "BBa_C0051"),
            ],
            family="lambda_repressors",
        )

        self._parts["GFP"] = BiologicalPart(
            name="GFP",
            part_type=PartType.CDS,
            description="Green fluorescent protein reporter",
            organisms={Organism.E_COLI, Organism.S_CEREVISIAE},
            characterisation=PartCharacterisation(
                degradation_rate=0.0077,
                measurement_conditions="Stable; t½ ≈ 90 min (untagged)",
            ),
            references=[
                PartReference("iGEM", "BBa_E0040",
                              "http://parts.igem.org/Part:BBa_E0040"),
            ],
            family="reporters",
        )

        self._parts["mCherry"] = BiologicalPart(
            name="mCherry",
            part_type=PartType.CDS,
            description="Red fluorescent protein reporter",
            organisms={Organism.E_COLI, Organism.S_CEREVISIAE},
            characterisation=PartCharacterisation(
                degradation_rate=0.0077,
            ),
            references=[
                PartReference("iGEM", "BBa_J06504"),
            ],
            family="reporters",
        )

        # ---- Terminators --------------------------------------------------

        self._parts["B0015"] = BiologicalPart(
            name="B0015",
            part_type=PartType.TERMINATOR,
            description="Double terminator (B0010 + B0012); >99% efficiency",
            organisms={Organism.E_COLI},
            references=[
                PartReference("iGEM", "BBa_B0015",
                              "http://parts.igem.org/Part:BBa_B0015"),
            ],
            family="standard_terminators",
        )

        self._parts["B0010"] = BiologicalPart(
            name="B0010",
            part_type=PartType.TERMINATOR,
            description="T1 from E. coli rrnB; ~95% efficiency",
            organisms={Organism.E_COLI},
            references=[
                PartReference("iGEM", "BBa_B0010"),
            ],
            family="standard_terminators",
        )

    # -- Cello gate library import ------------------------------------------

    def load_cello_gates(self) -> None:
        """Import a minimal set of Cello logic-gate parts (BBa NOR gates).

        Adds NOT/NOR gate response functions from the Cello 2.0 library
        (Nielsen et al., Science 2016).
        """
        cello_gates = [
            ("P1_PhlF", 2.8, 9.0, 2.5, "PhlF-based NOT gate"),
            ("P2_SrpR", 3.0, 8.0, 2.2, "SrpR-based NOT gate"),
            ("P3_BM3R1", 2.5, 7.5, 2.0, "BM3R1-based NOT gate"),
            ("P4_HlyIIR", 3.5, 12.0, 1.8, "HlyIIR-based NOT gate"),
            ("P5_BetI", 2.2, 6.5, 2.3, "BetI-based NOT gate"),
            ("P6_AmeR", 2.6, 10.0, 2.1, "AmeR-based NOT gate"),
        ]
        for name, vmax, k, n, desc in cello_gates:
            if name not in self._parts:
                self._parts[name] = BiologicalPart(
                    name=name,
                    part_type=PartType.PROMOTER,
                    description=f"Cello gate: {desc}",
                    organisms={Organism.E_COLI},
                    characterisation=PartCharacterisation(
                        hill=HillParameters(Vmax=vmax, K=k, n=n),
                        measurement_conditions="Cello 2.0 characterisation, E. coli",
                    ),
                    references=[
                        PartReference("Cello", name),
                    ],
                    family="cello_gates",
                )

    # -- summary / pretty-print ---------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of database contents."""
        counts: Dict[str, int] = {}
        for p in self._parts.values():
            key = p.part_type.name.lower()
            counts[key] = counts.get(key, 0) + 1
        lines = [f"PartsDatabase: {len(self._parts)} parts total"]
        for k, v in sorted(counts.items()):
            lines.append(f"  {k}: {v}")
        fam = self.families()
        if fam:
            lines.append(f"  families: {', '.join(sorted(fam))}")
        return "\n".join(lines)
