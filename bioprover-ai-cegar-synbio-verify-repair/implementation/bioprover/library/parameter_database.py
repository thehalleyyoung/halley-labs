"""Literature-sourced biological parameter database for BioProver.

Provides a curated, in-memory database of ~50 common biological parameters
(transcription rates, translation rates, degradation rates, Hill coefficients,
promoter strengths, etc.) with full provenance: DOI, authors, year,
organism, measurement method, and uncertainty quantification.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------

class MeasurementMethod(Enum):
    """How the parameter was measured."""

    FLUORESCENCE = auto()
    WESTERN_BLOT = auto()
    QPCR = auto()
    MASS_SPEC = auto()
    FLOW_CYTOMETRY = auto()
    PLATE_READER = auto()
    MICROSCOPY = auto()
    MODEL_FIT = auto()
    LITERATURE_ESTIMATE = auto()


@dataclass(frozen=True)
class LiteratureReference:
    """Citation for a parameter measurement.

    Attributes:
        doi:     Digital Object Identifier (e.g. ``"10.1038/403335a0"``).
        authors: Author list (abbreviated).
        title:   Paper title.
        year:    Publication year.
        journal: Journal name.
    """

    doi: str
    authors: str
    title: str = ""
    year: int = 0
    journal: str = ""

    def __str__(self) -> str:
        return f"{self.authors} ({self.year}) doi:{self.doi}"


# ---------------------------------------------------------------------------
# ParameterRecord
# ---------------------------------------------------------------------------

@dataclass
class ParameterRecord:
    """A single literature-sourced parameter value with full provenance.

    Attributes:
        name:        Parameter name / identifier.
        value:       Nominal (central) value.
        units:       Physical units string.
        organism:    Host organism for the measurement.
        method:      Measurement methodology.
        reference:   Primary literature citation.
        uncertainty_low:  Lower bound of reported range.
        uncertainty_high: Upper bound of reported range.
        std_dev:     Standard deviation (if reported).
        description: Human-readable description of the parameter.
        conditions:  Experimental conditions string.
        tags:        Searchable keyword tags.
    """

    name: str
    value: float
    units: str = ""
    organism: str = "E. coli"
    method: MeasurementMethod = MeasurementMethod.LITERATURE_ESTIMATE
    reference: Optional[LiteratureReference] = None
    uncertainty_low: Optional[float] = None
    uncertainty_high: Optional[float] = None
    std_dev: Optional[float] = None
    description: str = ""
    conditions: str = "37°C, LB medium"
    tags: List[str] = field(default_factory=list)

    @property
    def has_uncertainty(self) -> bool:
        """Return ``True`` if uncertainty information is available."""
        return (
            (self.uncertainty_low is not None and
             self.uncertainty_high is not None) or
            self.std_dev is not None
        )

    @property
    def range(self) -> Optional[Tuple[float, float]]:
        """Return ``(low, high)`` uncertainty interval, or ``None``."""
        if self.uncertainty_low is not None and self.uncertainty_high is not None:
            return (self.uncertainty_low, self.uncertainty_high)
        if self.std_dev is not None:
            return (self.value - 2 * self.std_dev,
                    self.value + 2 * self.std_dev)
        return None

    @property
    def coefficient_of_variation(self) -> Optional[float]:
        """Return CV (std_dev / value) if std_dev is known."""
        if self.std_dev is not None and self.value != 0:
            return abs(self.std_dev / self.value)
        return None


# ---------------------------------------------------------------------------
# ParameterDB
# ---------------------------------------------------------------------------

class ParameterDB:
    """In-memory database of literature-sourced biological parameters.

    Ships with ~50 built-in parameters covering *E. coli* transcription,
    translation, degradation, promoter strengths, RBS efficiencies,
    Hill coefficients, and common kinetic constants.
    """

    def __init__(self, *, load_builtins: bool = True) -> None:
        self._records: List[ParameterRecord] = []
        self._index: Dict[str, List[int]] = {}  # name -> list of indices
        if load_builtins:
            self._load_builtins()

    # -- CRUD ---------------------------------------------------------------

    def add(self, record: ParameterRecord) -> None:
        """Add a parameter record to the database."""
        idx = len(self._records)
        self._records.append(record)
        self._index.setdefault(record.name, []).append(idx)

    @property
    def all_records(self) -> List[ParameterRecord]:
        """Return all records."""
        return list(self._records)

    @property
    def parameter_names(self) -> List[str]:
        """Return sorted list of unique parameter names."""
        return sorted(self._index)

    def __len__(self) -> int:
        return len(self._records)

    # -- queries ------------------------------------------------------------

    def query_by_name(self, name: str) -> List[ParameterRecord]:
        """Return all records matching *name* (case-insensitive substring)."""
        name_lower = name.lower()
        return [r for r in self._records if name_lower in r.name.lower()]

    def query_by_organism(self, organism: str) -> List[ParameterRecord]:
        """Return records for the specified organism."""
        org_lower = organism.lower()
        return [r for r in self._records if org_lower in r.organism.lower()]

    def query_by_tag(self, tag: str) -> List[ParameterRecord]:
        """Return records containing the given tag."""
        tag_lower = tag.lower()
        return [
            r for r in self._records
            if any(tag_lower in t.lower() for t in r.tags)
        ]

    def query_by_range(
        self, name: str, low: float, high: float
    ) -> List[ParameterRecord]:
        """Return records whose *value* for *name* falls in [low, high]."""
        name_lower = name.lower()
        return [
            r for r in self._records
            if name_lower in r.name.lower() and low <= r.value <= high
        ]

    def get_best_estimate(self, name: str, organism: str = "E. coli") -> Optional[ParameterRecord]:
        """Return the single best-matching record for *name* and *organism*.

        Prefers records with the most specific organism match and an
        attached literature reference.
        """
        candidates = [
            r for r in self._records
            if r.name.lower() == name.lower()
            and organism.lower() in r.organism.lower()
        ]
        if not candidates:
            candidates = self.query_by_name(name)
        if not candidates:
            return None
        # Prefer records with a reference
        with_ref = [r for r in candidates if r.reference is not None]
        if with_ref:
            return with_ref[0]
        return candidates[0]

    # -- statistical summary ------------------------------------------------

    def summary_statistics(self, name: str) -> Dict[str, Any]:
        """Compute summary statistics across all records for *name*.

        Returns a dict with keys ``count``, ``mean``, ``median``,
        ``stdev``, ``min``, ``max``, ``units``.
        """
        records = self.query_by_name(name)
        if not records:
            return {"count": 0}
        values = [r.value for r in records]
        result: Dict[str, Any] = {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "units": records[0].units,
        }
        if len(values) >= 2:
            result["stdev"] = statistics.stdev(values)
        return result

    def parameter_distribution(self, name: str) -> List[float]:
        """Return a list of all reported values for *name*."""
        return [r.value for r in self.query_by_name(name)]

    # -- built-in database --------------------------------------------------

    def _load_builtins(self) -> None:  # noqa: C901
        """Populate with ~50 common biological parameters from literature."""

        _bionumbers = LiteratureReference(
            doi="10.1093/nar/gkp889",
            authors="Milo et al.",
            title="BioNumbers—the database of key numbers in molecular "
                  "and cell biology",
            year=2010,
            journal="Nucleic Acids Res",
        )
        _elowitz = LiteratureReference(
            doi="10.1038/35002125",
            authors="Elowitz & Leibler",
            title="A synthetic oscillatory network of transcriptional regulators",
            year=2000,
            journal="Nature",
        )
        _gardner = LiteratureReference(
            doi="10.1038/35002131",
            authors="Gardner, Cantor & Collins",
            title="Construction of a genetic toggle switch in E. coli",
            year=2000,
            journal="Nature",
        )
        _nielsen = LiteratureReference(
            doi="10.1126/science.aac7341",
            authors="Nielsen et al.",
            title="Genetic circuit design automation",
            year=2016,
            journal="Science",
        )
        _bremer = LiteratureReference(
            doi="10.1128/ecosalplus.5.2.3",
            authors="Bremer & Dennis",
            title="Modulation of chemical composition and other parameters "
                  "of the cell at different growth rates",
            year=2008,
            journal="EcoSal Plus",
        )

        # ---- E. coli transcription / translation --------------------------

        self.add(ParameterRecord(
            name="transcription_rate_ecoli",
            value=0.167, units="mRNA/s",
            organism="E. coli",
            description="Average mRNA transcription rate (10 nt/s, ~600 nt gene)",
            reference=_bremer,
            uncertainty_low=0.1, uncertainty_high=0.25,
            tags=["transcription", "mRNA"],
        ))
        self.add(ParameterRecord(
            name="translation_rate_ecoli",
            value=0.033, units="protein/s/mRNA",
            organism="E. coli",
            description="Average translation rate (~20 aa/s, ~300 aa protein)",
            reference=_bremer,
            uncertainty_low=0.02, uncertainty_high=0.05,
            tags=["translation", "protein"],
        ))
        self.add(ParameterRecord(
            name="mrna_half_life_ecoli",
            value=5.0, units="min",
            organism="E. coli",
            description="Median mRNA half-life in E. coli",
            reference=_bionumbers,
            uncertainty_low=2.0, uncertainty_high=10.0,
            tags=["degradation", "mRNA", "half_life"],
        ))
        self.add(ParameterRecord(
            name="mrna_degradation_rate",
            value=0.139, units="1/min",
            organism="E. coli",
            description="mRNA degradation rate (ln2 / 5 min)",
            reference=_bionumbers,
            uncertainty_low=0.069, uncertainty_high=0.347,
            tags=["degradation", "mRNA"],
        ))
        self.add(ParameterRecord(
            name="protein_half_life_ecoli",
            value=60.0, units="min",
            organism="E. coli",
            description="Median protein half-life (dilution-dominated)",
            reference=_bionumbers,
            uncertainty_low=20.0, uncertainty_high=180.0,
            tags=["degradation", "protein", "half_life"],
        ))
        self.add(ParameterRecord(
            name="protein_degradation_rate",
            value=0.0116, units="1/min",
            organism="E. coli",
            description="Protein degradation rate (ln2 / 60 min)",
            reference=_bionumbers,
            uncertainty_low=0.0039, uncertainty_high=0.0347,
            tags=["degradation", "protein"],
        ))
        self.add(ParameterRecord(
            name="cell_division_time_ecoli",
            value=30.0, units="min",
            organism="E. coli",
            description="Doubling time in rich media at 37°C",
            reference=_bremer,
            uncertainty_low=20.0, uncertainty_high=60.0,
            tags=["growth", "division"],
        ))
        self.add(ParameterRecord(
            name="dilution_rate",
            value=0.0231, units="1/min",
            organism="E. coli",
            description="Dilution rate (ln2 / 30 min doubling)",
            reference=_bremer,
            uncertainty_low=0.0116, uncertainty_high=0.0347,
            tags=["growth", "dilution"],
        ))

        # ---- Promoter strengths -------------------------------------------

        promoter_data = [
            ("pTet_strength", 5.0, "nM/min", 4.0, 6.0,
             "TetR-repressible promoter maximal rate", _elowitz),
            ("pLac_strength", 8.0, "nM/min", 6.0, 10.0,
             "LacI-repressible promoter maximal rate", _gardner),
            ("pBad_strength", 10.0, "nM/min", 8.0, 12.0,
             "AraC-regulated promoter maximal rate", _bionumbers),
            ("pLux_strength", 6.0, "nM/min", 4.5, 7.5,
             "LuxR-activated promoter maximal rate", _bionumbers),
            ("pLambdaCI_strength", 5.5, "nM/min", 4.0, 7.0,
             "Lambda CI-repressible promoter maximal rate", _elowitz),
            ("pTac_strength", 12.0, "nM/min", 9.0, 15.0,
             "Hybrid trc/lac strong promoter", _bionumbers),
            ("J23100_strength", 1.0, "RPU", 0.9, 1.1,
             "Anderson promoter BBa_J23100 (reference)", _nielsen),
            ("J23101_strength", 0.70, "RPU", 0.6, 0.8,
             "Anderson promoter BBa_J23101", _nielsen),
            ("J23106_strength", 0.47, "RPU", 0.4, 0.55,
             "Anderson promoter BBa_J23106", _nielsen),
            ("J23116_strength", 0.16, "RPU", 0.12, 0.20,
             "Anderson promoter BBa_J23116", _nielsen),
        ]
        for name, val, units, lo, hi, desc, ref in promoter_data:
            self.add(ParameterRecord(
                name=name, value=val, units=units,
                description=desc, reference=ref,
                uncertainty_low=lo, uncertainty_high=hi,
                tags=["promoter", "strength"],
            ))

        # ---- RBS efficiencies (relative) ----------------------------------

        rbs_data = [
            ("B0034_efficiency", 1.0, "AU", 0.8, 1.2,
             "Strong RBS BBa_B0034"),
            ("B0032_efficiency", 0.3, "AU", 0.2, 0.4,
             "Medium RBS BBa_B0032"),
            ("B0031_efficiency", 0.07, "AU", 0.04, 0.10,
             "Weak RBS BBa_B0031"),
        ]
        for name, val, units, lo, hi, desc in rbs_data:
            self.add(ParameterRecord(
                name=name, value=val, units=units,
                description=desc, reference=_nielsen,
                uncertainty_low=lo, uncertainty_high=hi,
                tags=["RBS", "translation"],
            ))

        # ---- Protein degradation rates ------------------------------------

        deg_data = [
            ("TetR_degradation", 0.0231, "1/min", 0.015, 0.035,
             "TetR protein degradation (ssrA-tagged, t½ ≈ 30 min)", _elowitz),
            ("LacI_degradation", 0.0116, "1/min", 0.008, 0.015,
             "LacI protein degradation (t½ ≈ 60 min)", _gardner),
            ("CI_degradation", 0.0173, "1/min", 0.012, 0.025,
             "Lambda CI protein degradation (t½ ≈ 40 min)", _elowitz),
            ("GFP_degradation", 0.0077, "1/min", 0.005, 0.012,
             "GFP degradation (untagged, t½ ≈ 90 min)", _bionumbers),
            ("GFP_ssrA_degradation", 0.0693, "1/min", 0.05, 0.10,
             "GFP-ssrA (ClpXP-targeted, t½ ≈ 10 min)", _elowitz),
            ("mCherry_degradation", 0.0077, "1/min", 0.005, 0.012,
             "mCherry degradation (untagged)", _bionumbers),
        ]
        for name, val, units, lo, hi, desc, ref in deg_data:
            self.add(ParameterRecord(
                name=name, value=val, units=units,
                description=desc, reference=ref,
                uncertainty_low=lo, uncertainty_high=hi,
                tags=["degradation", "protein"],
            ))

        # ---- Hill coefficients for common interactions --------------------

        hill_data = [
            ("TetR_hill_n", 2.0, "", 1.5, 2.5,
             "Hill coefficient for TetR repression", _elowitz),
            ("TetR_hill_K", 10.0, "nM", 5.0, 20.0,
             "Half-repression concentration for TetR", _elowitz),
            ("LacI_hill_n", 2.0, "", 1.5, 2.5,
             "Hill coefficient for LacI repression", _gardner),
            ("LacI_hill_K", 15.0, "nM", 8.0, 30.0,
             "Half-repression concentration for LacI", _gardner),
            ("CI_hill_n", 2.5, "", 2.0, 3.0,
             "Hill coefficient for CI repression", _elowitz),
            ("CI_hill_K", 11.0, "nM", 6.0, 22.0,
             "Half-repression concentration for CI", _elowitz),
            ("AraC_hill_n", 1.5, "", 1.0, 2.0,
             "Hill coefficient for AraC activation", _bionumbers),
            ("AraC_hill_K", 20.0, "nM", 10.0, 40.0,
             "Half-activation concentration for AraC", _bionumbers),
            ("LuxR_hill_n", 2.0, "", 1.5, 2.5,
             "Hill coefficient for LuxR activation", _bionumbers),
            ("LuxR_hill_K", 12.0, "nM", 6.0, 25.0,
             "Half-activation concentration for LuxR", _bionumbers),
        ]
        for name, val, units, lo, hi, desc, ref in hill_data:
            self.add(ParameterRecord(
                name=name, value=val, units=units,
                description=desc, reference=ref,
                uncertainty_low=lo, uncertainty_high=hi,
                tags=["Hill", "kinetics"],
            ))

        # ---- Cello NOR gate response functions ----------------------------

        cello_data = [
            ("PhlF_ymax", 2.8, "RPU", 2.2, 3.4, "PhlF gate max output"),
            ("PhlF_ymin", 0.003, "RPU", 0.001, 0.005, "PhlF gate min output"),
            ("PhlF_K", 9.0, "RPU", 6.0, 12.0, "PhlF gate half-repression"),
            ("PhlF_n", 2.5, "", 2.0, 3.0, "PhlF gate Hill coefficient"),
            ("SrpR_ymax", 3.0, "RPU", 2.4, 3.6, "SrpR gate max output"),
            ("SrpR_K", 8.0, "RPU", 5.0, 11.0, "SrpR gate half-repression"),
            ("SrpR_n", 2.2, "", 1.8, 2.6, "SrpR gate Hill coefficient"),
            ("BM3R1_ymax", 2.5, "RPU", 2.0, 3.0, "BM3R1 gate max output"),
            ("BM3R1_K", 7.5, "RPU", 5.0, 10.0, "BM3R1 gate half-repression"),
            ("BM3R1_n", 2.0, "", 1.5, 2.5, "BM3R1 gate Hill coefficient"),
        ]
        for name, val, units, lo, hi, desc in cello_data:
            self.add(ParameterRecord(
                name=name, value=val, units=units,
                description=desc, reference=_nielsen,
                uncertainty_low=lo, uncertainty_high=hi,
                tags=["Cello", "gate", "Hill"],
            ))

        # ---- Miscellaneous constants --------------------------------------

        self.add(ParameterRecord(
            name="avogadro_number",
            value=6.022e23, units="1/mol",
            organism="universal",
            description="Avogadro's number",
            tags=["constant"],
        ))
        self.add(ParameterRecord(
            name="ecoli_cell_volume",
            value=1.0e-15, units="L",
            organism="E. coli",
            description="Typical E. coli cell volume (~1 fL)",
            reference=_bionumbers,
            uncertainty_low=0.5e-15, uncertainty_high=2.0e-15,
            tags=["cell", "volume"],
        ))
        self.add(ParameterRecord(
            name="nM_per_molecule_ecoli",
            value=1.66, units="nM",
            organism="E. coli",
            description="Concentration of one molecule per E. coli cell "
                        "(1 / (V_cell * N_A) in nM)",
            reference=_bionumbers,
            uncertainty_low=0.83, uncertainty_high=3.32,
            tags=["conversion", "concentration"],
        ))
        self.add(ParameterRecord(
            name="maturation_time_GFP",
            value=10.0, units="min",
            organism="E. coli",
            description="GFP chromophore maturation time",
            reference=_bionumbers,
            uncertainty_low=5.0, uncertainty_high=30.0,
            tags=["maturation", "reporter", "GFP"],
        ))
        self.add(ParameterRecord(
            name="rnap_concentration",
            value=3000.0, units="nM",
            organism="E. coli",
            description="RNA polymerase concentration (~2000-5000 per cell)",
            reference=_bremer,
            uncertainty_low=2000.0, uncertainty_high=5000.0,
            tags=["transcription", "RNAP"],
        ))
        self.add(ParameterRecord(
            name="ribosome_concentration",
            value=25000.0, units="nM",
            organism="E. coli",
            description="Ribosome concentration (~10k-70k per cell)",
            reference=_bremer,
            uncertainty_low=10000.0, uncertainty_high=70000.0,
            tags=["translation", "ribosome"],
        ))

    # -- summary / pretty-print ---------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of database contents."""
        lines = [f"ParameterDB: {len(self._records)} records"]
        tag_counts: Dict[str, int] = {}
        for r in self._records:
            for t in r.tags:
                tag_counts[t] = tag_counts.get(t, 0) + 1
        for tag, cnt in sorted(tag_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  [{tag}]: {cnt}")
        return "\n".join(lines)
