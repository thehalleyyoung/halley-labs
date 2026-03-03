"""Abstract base class for statistical certificates."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Certificate(ABC):
    """A statistical certificate quantifying reliability of a causal structure.

    Certificates provide a scalar value in [0, 1] that measures how
    confident we are in a particular structural claim (edge, path, etc.),
    together with an associated confidence level.
    """

    @property
    @abstractmethod
    def value(self) -> float:
        """Certificate strength in [0, 1].

        Higher values indicate stronger evidence for the structural claim.
        """

    @property
    @abstractmethod
    def confidence(self) -> float:
        """Confidence level associated with this certificate (e.g. 0.95)."""

    def is_certified(self, threshold: float = 0.5) -> bool:
        """Return whether the certificate exceeds *threshold*.

        Parameters
        ----------
        threshold:
            Minimum certificate value required for certification.

        Returns
        -------
        bool
            ``True`` if :pyattr:`value` >= *threshold*.
        """
        return self.value >= threshold

    @abstractmethod
    def combine(self, other: Certificate) -> Certificate:
        """Combine this certificate with *other* to produce a joint certificate.

        The semantics of combination depend on the concrete subclass (e.g.
        taking the minimum for a path, or averaging for an ensemble).

        Parameters
        ----------
        other:
            Another certificate to combine with.

        Returns
        -------
        Certificate
            A new certificate representing the combined evidence.
        """
