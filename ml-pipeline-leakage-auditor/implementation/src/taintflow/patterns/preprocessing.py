"""
taintflow.patterns.preprocessing – Preprocessing leakage pattern detectors.

Detects leakage from scaling, imputation, encoding, normalization, and
outlier handling operations applied before or across the train/test split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from taintflow.core.types import OpType, Origin, Severity


@dataclass
class PreprocessingLeakagePattern:
    """A matched preprocessing leakage pattern.

    Attributes:
        pattern_name: Identifier for the pattern type.
        operation: The preprocessing operation that caused leakage.
        source_location: File and line number where the operation occurs.
        severity: Estimated severity of the leakage.
        description: Human-readable explanation.
        remediation: Suggested fix.
    """

    pattern_name: str = ""
    operation: str = ""
    source_location: str = ""
    severity: Severity = Severity.WARNING
    description: str = ""
    remediation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "pattern_name": self.pattern_name,
            "operation": self.operation,
            "source_location": self.source_location,
            "severity": self.severity.name,
            "description": self.description,
            "remediation": self.remediation,
        }


class PreprocessingLeakageDetector:
    """Aggregate detector for all preprocessing leakage patterns.

    Combines scaling, imputation, encoding, normalization, and outlier
    sub-detectors into a single scan.

    Args:
        detectors: Optional list of sub-detectors to use.
    """

    def __init__(
        self,
        detectors: Optional[Sequence["_BasePreprocessingDetector"]] = None,
    ) -> None:
        self._detectors: list[_BasePreprocessingDetector] = list(detectors or [
            ScalingLeakageDetector(),
            ImputationLeakageDetector(),
            EncodingLeakageDetector(),
            NormalizationLeakageDetector(),
            OutlierLeakageDetector(),
        ])

    def detect(
        self, operations: Sequence[Dict[str, Any]]
    ) -> list[PreprocessingLeakagePattern]:
        """Run all sub-detectors on the given operations.

        Args:
            operations: Sequence of operation descriptors from the PI-DAG.

        Returns:
            List of matched preprocessing leakage patterns.
        """
        patterns: list[PreprocessingLeakagePattern] = []
        for detector in self._detectors:
            patterns.extend(detector.detect(operations))
        return patterns


class _BasePreprocessingDetector:
    """Base class for preprocessing leakage sub-detectors."""

    def detect(
        self, operations: Sequence[Dict[str, Any]]
    ) -> list[PreprocessingLeakagePattern]:
        """Detect leakage patterns in the given operations."""
        return []


class ScalingLeakageDetector(_BasePreprocessingDetector):
    """Detects leakage from scaling operations (StandardScaler, MinMaxScaler, etc.)
    applied before the train/test split."""

    def detect(
        self, operations: Sequence[Dict[str, Any]]
    ) -> list[PreprocessingLeakagePattern]:
        patterns: list[PreprocessingLeakagePattern] = []
        for op in operations:
            op_type = op.get("op_type", "")
            if op_type in ("STANDARD_SCALER", "MINMAX_SCALER", "ROBUST_SCALER", "SCALING"):
                if op.get("before_split", False):
                    patterns.append(PreprocessingLeakagePattern(
                        pattern_name="scaling_before_split",
                        operation=op_type,
                        source_location=op.get("source_location", ""),
                        severity=Severity.CRITICAL,
                        description=f"{op_type} fitted on full dataset before train/test split.",
                        remediation="Move scaling inside a sklearn Pipeline or apply after split.",
                    ))
        return patterns


class ImputationLeakageDetector(_BasePreprocessingDetector):
    """Detects leakage from imputation operations applied before split."""

    def detect(
        self, operations: Sequence[Dict[str, Any]]
    ) -> list[PreprocessingLeakagePattern]:
        patterns: list[PreprocessingLeakagePattern] = []
        for op in operations:
            op_type = op.get("op_type", "")
            if op_type in ("IMPUTER", "IMPUTATION", "KNN_IMPUTER", "FILLNA"):
                if op.get("before_split", False):
                    patterns.append(PreprocessingLeakagePattern(
                        pattern_name="imputation_before_split",
                        operation=op_type,
                        source_location=op.get("source_location", ""),
                        severity=Severity.CRITICAL,
                        description=f"{op_type} fitted on full dataset before split.",
                        remediation="Use sklearn Pipeline to ensure imputation uses only training data.",
                    ))
        return patterns


class EncodingLeakageDetector(_BasePreprocessingDetector):
    """Detects leakage from encoding operations applied before split."""

    def detect(
        self, operations: Sequence[Dict[str, Any]]
    ) -> list[PreprocessingLeakagePattern]:
        patterns: list[PreprocessingLeakagePattern] = []
        for op in operations:
            op_type = op.get("op_type", "")
            if op_type in ("ONEHOT_ENCODER", "LABEL_ENCODER", "ORDINAL_ENCODER",
                           "TARGET_ENCODER", "ENCODING"):
                if op.get("before_split", False):
                    patterns.append(PreprocessingLeakagePattern(
                        pattern_name="encoding_before_split",
                        operation=op_type,
                        source_location=op.get("source_location", ""),
                        severity=Severity.WARNING,
                        description=f"{op_type} fitted on full dataset before split.",
                        remediation="Fit encoder on training data only.",
                    ))
        return patterns


class NormalizationLeakageDetector(_BasePreprocessingDetector):
    """Detects leakage from normalization applied before split."""

    def detect(
        self, operations: Sequence[Dict[str, Any]]
    ) -> list[PreprocessingLeakagePattern]:
        patterns: list[PreprocessingLeakagePattern] = []
        for op in operations:
            op_type = op.get("op_type", "")
            if op_type in ("NORMALIZER", "NORMALIZATION"):
                if op.get("before_split", False):
                    patterns.append(PreprocessingLeakagePattern(
                        pattern_name="normalization_before_split",
                        operation=op_type,
                        source_location=op.get("source_location", ""),
                        severity=Severity.WARNING,
                        description=f"{op_type} applied on full dataset before split.",
                        remediation="Apply normalization within a Pipeline after splitting.",
                    ))
        return patterns


class OutlierLeakageDetector(_BasePreprocessingDetector):
    """Detects leakage from outlier removal/clipping using full-dataset statistics."""

    def detect(
        self, operations: Sequence[Dict[str, Any]]
    ) -> list[PreprocessingLeakagePattern]:
        patterns: list[PreprocessingLeakagePattern] = []
        for op in operations:
            op_type = op.get("op_type", "")
            if op_type in ("CLIP", "FILTER") and op.get("uses_global_stats", False):
                patterns.append(PreprocessingLeakagePattern(
                    pattern_name="outlier_removal_global_stats",
                    operation=op_type,
                    source_location=op.get("source_location", ""),
                    severity=Severity.NEGLIGIBLE,
                    description="Outlier thresholds computed on full dataset.",
                    remediation="Compute outlier thresholds on training data only.",
                ))
        return patterns


class PatternMatcher:
    """Utility for matching operation sequences against known leakage patterns.

    Args:
        patterns: List of pattern templates to match against.
    """

    def __init__(self, patterns: Optional[List[Dict[str, Any]]] = None) -> None:
        self._patterns = patterns or []

    def match(self, operations: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """Match operations against registered patterns.

        Returns:
            List of matched pattern descriptors.
        """
        matches: list[Dict[str, Any]] = []
        for pattern in self._patterns:
            required_ops = pattern.get("required_ops", [])
            found = all(
                any(op.get("op_type") == req for op in operations)
                for req in required_ops
            )
            if found:
                matches.append(pattern)
        return matches


class LeakagePatternLibrary:
    """Registry of all known leakage patterns.

    Provides a unified interface for querying and filtering patterns
    by category, severity, or operation type.
    """

    def __init__(self) -> None:
        self._detectors: list[PreprocessingLeakageDetector] = [
            PreprocessingLeakageDetector(),
        ]

    def scan(
        self, operations: Sequence[Dict[str, Any]]
    ) -> list[PreprocessingLeakagePattern]:
        """Scan operations with all registered pattern detectors.

        Returns:
            Aggregated list of all matched patterns.
        """
        results: list[PreprocessingLeakagePattern] = []
        for detector in self._detectors:
            results.extend(detector.detect(operations))
        return results

    def list_patterns(self) -> list[str]:
        """Return names of all registered pattern types."""
        return [
            "scaling_before_split",
            "imputation_before_split",
            "encoding_before_split",
            "normalization_before_split",
            "outlier_removal_global_stats",
        ]


__all__ = [
    "EncodingLeakageDetector",
    "ImputationLeakageDetector",
    "LeakagePatternLibrary",
    "NormalizationLeakageDetector",
    "OutlierLeakageDetector",
    "PatternMatcher",
    "PreprocessingLeakageDetector",
    "PreprocessingLeakagePattern",
    "ScalingLeakageDetector",
]
