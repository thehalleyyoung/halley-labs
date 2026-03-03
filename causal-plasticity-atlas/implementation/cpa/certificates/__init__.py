"""
Causal-Plasticity Atlas — Robustness Certificate Module.

Implements Algorithm 5 (ALG5): Robustness Certificate Generation.
Provides certificates of mechanism stability via stability selection,
parametric bootstrap, and upper confidence bound analysis.

Extended with:
    - Fisher information Lipschitz bounds (lipschitz.py)
    - Boltzmann stability analysis (boltzmann.py)
    - Adaptive Certificate Tightening / ACT / Algorithm 7 (adaptive_tightening.py)

Main classes:
    CertificateGenerator     — Full ALG5 certificate generation pipeline.
    CertificateValidator     — Certificate assumption validation.
    CertificateReport        — Human-readable certificate reports.
    StabilitySelectionEngine — Generic stability selection framework.
    BootstrapEngine          — Parametric and nonparametric bootstrap.
    FisherInformationBound   — Lipschitz bounds via Fisher information.
    MechanismStabilityBound  — Aggregate Fisher stability certificates.
    BoltzmannStabilityAnalyzer — Boltzmann posterior over DAGs.
    BoltzmannCertificate     — Certificates from Boltzmann posteriors.
    AdaptiveCertificateTightener — ACT (Algorithm 7).
    SequentialCertificate    — Online sequential certificates.
"""

from __future__ import annotations

from cpa.certificates.robustness import (
    CertificateGenerator,
    CertificateValidator,
    CertificateReport,
    RobustnessCertificate,
    CertificateConfig,
)
from cpa.certificates.stability import (
    BootstrapEngine,
    StabilitySelectionEngine,
)
from cpa.certificates.lipschitz import (
    FisherInformationBound,
    MechanismStabilityBound,
)
from cpa.certificates.boltzmann import (
    BoltzmannStabilityAnalyzer,
    BoltzmannCertificate,
)
from cpa.certificates.adaptive_tightening import (
    AdaptiveCertificateTightener,
    SequentialCertificate,
)

__all__ = [
    "CertificateGenerator",
    "CertificateValidator",
    "CertificateReport",
    "RobustnessCertificate",
    "CertificateConfig",
    "StabilitySelectionEngine",
    "BootstrapEngine",
    "FisherInformationBound",
    "MechanismStabilityBound",
    "BoltzmannStabilityAnalyzer",
    "BoltzmannCertificate",
    "AdaptiveCertificateTightener",
    "SequentialCertificate",
]
