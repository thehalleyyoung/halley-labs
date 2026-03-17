"""Jurisdiction module: regulatory framework database and cross-jurisdiction analysis.

Contains encodings of 10 regulatory frameworks and conflict detection.
"""

from regsynth_py.jurisdiction.database import JurisdictionDB
from regsynth_py.jurisdiction.cross_jurisdiction import CrossJurisdictionMapper
from regsynth_py.jurisdiction.conflict_detector import ConflictDetector
from regsynth_py.jurisdiction.eu_ai_act import EUAIAct
from regsynth_py.jurisdiction.nist_rmf import NISTRiskManagementFramework
from regsynth_py.jurisdiction.iso_42001 import ISO42001
from regsynth_py.jurisdiction.china_genai import ChinaGenAIMeasures, ChinaGenAI
from regsynth_py.jurisdiction.gdpr_ai import GDPRAIProvisions, GDPR_AI

__all__ = [
    "JurisdictionDB", "CrossJurisdictionMapper", "ConflictDetector",
    "EUAIAct", "NISTRiskManagementFramework", "ISO42001",
    "ChinaGenAIMeasures", "ChinaGenAI",
    "GDPRAIProvisions", "GDPR_AI",
]
