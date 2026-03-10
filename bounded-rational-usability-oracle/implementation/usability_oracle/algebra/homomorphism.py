"""
usability_oracle.algebra.homomorphism — Cost algebra homomorphisms.

Provides structure-preserving maps between cost algebras, enabling:

* **Abstraction**: map concrete 4-tuple costs to simpler models.
* **Refinement**: reconstruct detailed costs from abstract summaries.
* **Galois connections**: pairs of adjoint abstraction/concretisation maps.
* **Quotient construction**: factor out irrelevant cost dimensions.
* **Kernel / image**: compute what a homomorphism preserves and collapses.
* **Soundness proofs**: verify that abstractions preserve compositional properties.

Mathematical Structure
----------------------
A **homomorphism** ``h : (A, ⊕, ⊗) → (B, ⊕', ⊗')`` between cost algebras
satisfies:

.. math::

    h(a ⊕ b) = h(a) ⊕' h(b)  \\quad\\text{and}\\quad  h(a ⊗ b) = h(a) ⊗' h(b)

and ``h(0_A) = 0_B`` (preserves identity).

Application
~~~~~~~~~~~
* Relating Fitts' law (motor cost) to Hick's law (decision cost) models.
* Projecting the full 4-tuple to ``(μ, σ²)`` for simplified analysis.
* Proving that a simplified model soundly approximates the full model.

References
----------
* Birkhoff, *Lattice Theory*, AMS, 1967, Ch. VI.
* Cousot & Cousot, *Systematic Design of Program Analysis Frameworks*,
  POPL 1979.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from usability_oracle.algebra.models import CostElement
from usability_oracle.algebra.sequential import SequentialComposer
from usability_oracle.algebra.parallel import ParallelComposer

# ---------------------------------------------------------------------------
# CostAlgebraHomomorphism
# ---------------------------------------------------------------------------


@dataclass
class CostAlgebraHomomorphism:
    r"""A structure-preserving map between cost algebras.

    Parameters
    ----------
    map_fn : callable
        The mapping function ``h : CostElement → CostElement``.
    name : str
        Human-readable name.
    inverse_fn : callable | None
        Optional inverse (for isomorphisms).
    """

    map_fn: Callable[[CostElement], CostElement]
    name: str = ""
    inverse_fn: Optional[Callable[[CostElement], CostElement]] = None

    def apply(self, element: CostElement) -> CostElement:
        """Apply the homomorphism: ``h(element)``."""
        return self.map_fn(element)

    def apply_many(self, elements: Sequence[CostElement]) -> List[CostElement]:
        """Apply to a sequence."""
        return [self.map_fn(e) for e in elements]

    def compose_with(self, other: "CostAlgebraHomomorphism") -> "CostAlgebraHomomorphism":
        """Compose homomorphisms: ``(h₂ ∘ h₁)(x) = h₂(h₁(x))``."""
        def composed(x: CostElement) -> CostElement:
            return other.map_fn(self.map_fn(x))

        inv = None
        if self.inverse_fn is not None and other.inverse_fn is not None:
            def inv_composed(y: CostElement) -> CostElement:
                return self.inverse_fn(other.inverse_fn(y))  # type: ignore
            inv = inv_composed

        return CostAlgebraHomomorphism(
            map_fn=composed,
            name=f"{other.name}∘{self.name}",
            inverse_fn=inv,
        )

    @property
    def is_invertible(self) -> bool:
        return self.inverse_fn is not None

    def inverse(self) -> "CostAlgebraHomomorphism":
        """Return the inverse homomorphism (if it exists)."""
        if self.inverse_fn is None:
            raise ValueError(f"Homomorphism {self.name!r} is not invertible.")
        return CostAlgebraHomomorphism(
            map_fn=self.inverse_fn,
            name=f"{self.name}⁻¹",
            inverse_fn=self.map_fn,
        )

    def __repr__(self) -> str:
        return f"Hom({self.name})"


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_sequential_homomorphism(
    h: CostAlgebraHomomorphism,
    samples: Sequence[Tuple[CostElement, CostElement]],
    coupling: float = 0.0,
    tol: float = 1e-6,
) -> bool:
    r"""Verify: ``h(a ⊕ b) ≈ h(a) ⊕' h(b)`` for sampled pairs.

    Parameters
    ----------
    h : CostAlgebraHomomorphism
        The homomorphism to verify.
    samples : sequence of (CostElement, CostElement)
        Pairs to test.
    coupling : float
        Coupling for sequential composition.
    tol : float
        Tolerance.

    Returns
    -------
    bool
        True if the homomorphism property holds for all samples.
    """
    seq = SequentialComposer()
    for a, b in samples:
        lhs = h.apply(seq.compose(a, b, coupling=coupling))
        rhs = seq.compose(h.apply(a), h.apply(b), coupling=coupling)
        if abs(lhs.mu - rhs.mu) > tol or abs(lhs.sigma_sq - rhs.sigma_sq) > tol:
            return False
    return True


def verify_parallel_homomorphism(
    h: CostAlgebraHomomorphism,
    samples: Sequence[Tuple[CostElement, CostElement]],
    interference: float = 0.0,
    tol: float = 1e-6,
) -> bool:
    r"""Verify: ``h(a ⊗ b) ≈ h(a) ⊗' h(b)`` for sampled pairs."""
    par = ParallelComposer()
    for a, b in samples:
        lhs = h.apply(par.compose(a, b, interference=interference))
        rhs = par.compose(h.apply(a), h.apply(b), interference=interference)
        if abs(lhs.mu - rhs.mu) > tol or abs(lhs.sigma_sq - rhs.sigma_sq) > tol:
            return False
    return True


def verify_identity_preservation(
    h: CostAlgebraHomomorphism,
    tol: float = 1e-10,
) -> bool:
    """Verify: ``h(0) = 0`` (identity preservation)."""
    z = CostElement.zero()
    hz = h.apply(z)
    return abs(hz.mu) < tol and abs(hz.sigma_sq) < tol and abs(hz.lambda_) < tol


# ---------------------------------------------------------------------------
# Standard homomorphisms
# ---------------------------------------------------------------------------


def mean_projection() -> CostAlgebraHomomorphism:
    r"""Project to mean cost only: ``(μ, σ², κ, λ) ↦ (μ, 0, 0, 0)``.

    This is a homomorphism when composition is independent (coupling = 0)
    since ``μ_{a⊕b} = μ_a + μ_b``.
    """
    def proj(c: CostElement) -> CostElement:
        return CostElement(mu=c.mu, sigma_sq=0.0, kappa=0.0, lambda_=0.0)

    return CostAlgebraHomomorphism(map_fn=proj, name="π_μ")


def moment_projection() -> CostAlgebraHomomorphism:
    r"""Project to first two moments: ``(μ, σ², κ, λ) ↦ (μ, σ², 0, 0)``."""
    def proj(c: CostElement) -> CostElement:
        return CostElement(mu=c.mu, sigma_sq=c.sigma_sq, kappa=0.0, lambda_=0.0)

    return CostAlgebraHomomorphism(map_fn=proj, name="π_{μ,σ²}")


def scale_homomorphism(factor: float) -> CostAlgebraHomomorphism:
    r"""Scalar multiplication: ``(μ, σ², κ, λ) ↦ (c·μ, c²·σ², κ, λ)``.

    This is a homomorphism for any ``c > 0``.
    """
    def scale(x: CostElement) -> CostElement:
        return CostElement(
            mu=factor * x.mu,
            sigma_sq=factor * factor * x.sigma_sq,
            kappa=x.kappa,
            lambda_=x.lambda_,
        )

    inv_factor = 1.0 / factor if factor != 0 else 0.0

    def inv_scale(y: CostElement) -> CostElement:
        return CostElement(
            mu=inv_factor * y.mu,
            sigma_sq=inv_factor * inv_factor * y.sigma_sq,
            kappa=y.kappa,
            lambda_=y.lambda_,
        )

    return CostAlgebraHomomorphism(
        map_fn=scale, name=f"×{factor}", inverse_fn=inv_scale
    )


def log_transform() -> CostAlgebraHomomorphism:
    r"""Log-transform of mean cost: ``(μ, σ², κ, λ) ↦ (log(1+μ), σ²/(1+μ)², κ, λ)``.

    Maps to a logarithmic cost scale (useful for Weber-Fechner perceptual models).
    Not a strict homomorphism but approximately preserves composition for
    costs of similar magnitude (via log-additivity).
    """
    def log_map(c: CostElement) -> CostElement:
        lmu = math.log1p(c.mu)
        scale = 1.0 / (1.0 + c.mu) if c.mu >= 0 else 1.0
        return CostElement(
            mu=lmu,
            sigma_sq=c.sigma_sq * scale * scale,
            kappa=c.kappa,
            lambda_=c.lambda_,
        )

    def exp_map(c: CostElement) -> CostElement:
        emu = math.expm1(c.mu)
        scale = (1.0 + emu)
        return CostElement(
            mu=emu,
            sigma_sq=c.sigma_sq * scale * scale,
            kappa=c.kappa,
            lambda_=c.lambda_,
        )

    return CostAlgebraHomomorphism(map_fn=log_map, name="log", inverse_fn=exp_map)


def bits_to_seconds(
    base_rate: float = 0.15,
) -> CostAlgebraHomomorphism:
    r"""Convert information-theoretic cost (bits) to time cost (seconds).

    Uses the Hick-Hyman law: ``T = a + b · H`` where ``H`` is in bits
    and ``b ≈ 0.15 s/bit``.

    Parameters
    ----------
    base_rate : float
        Seconds per bit (default 0.15 s/bit, typical for choice RT).
    """
    return scale_homomorphism(base_rate)


# ---------------------------------------------------------------------------
# Galois connections between cost algebras
# ---------------------------------------------------------------------------


@dataclass
class CostGaloisConnection:
    r"""A Galois connection ``(α, γ)`` between cost algebras.

    ``α`` (abstraction) is a homomorphism from concrete to abstract.
    ``γ`` (concretisation) is a homomorphism from abstract to concrete.

    The pair satisfies: ``∀ c, a:  α(c) ⊑ a  ⟺  c ⊑ γ(a)``.

    Parameters
    ----------
    abstraction : CostAlgebraHomomorphism
        The abstraction map ``α``.
    concretisation : CostAlgebraHomomorphism
        The concretisation map ``γ``.
    name : str
        Human-readable name.
    """

    abstraction: CostAlgebraHomomorphism
    concretisation: CostAlgebraHomomorphism
    name: str = ""

    def abstract(self, c: CostElement) -> CostElement:
        return self.abstraction.apply(c)

    def concretise(self, a: CostElement) -> CostElement:
        return self.concretisation.apply(a)

    def verify_soundness(
        self,
        concrete_samples: Sequence[CostElement],
        tol: float = 1e-10,
    ) -> bool:
        r"""Verify: ``c ⊑ γ(α(c))`` for all samples (extensive property)."""
        from usability_oracle.algebra.lattice import cost_leq
        for c in concrete_samples:
            ac = self.abstraction.apply(c)
            gac = self.concretisation.apply(ac)
            if not cost_leq(c, gac, tol):
                return False
        return True

    def verify_reductive(
        self,
        abstract_samples: Sequence[CostElement],
        tol: float = 1e-10,
    ) -> bool:
        r"""Verify: ``α(γ(a)) ⊑ a`` for all samples (reductive property)."""
        from usability_oracle.algebra.lattice import cost_leq
        for a in abstract_samples:
            ga = self.concretisation.apply(a)
            aga = self.abstraction.apply(ga)
            if not cost_leq(aga, a, tol):
                return False
        return True


def moment_galois_connection() -> CostGaloisConnection:
    r"""Galois connection projecting to ``(μ, σ²)``.

    * ``α(μ, σ², κ, λ) = (μ · (1 + λ), σ², 0, 0)`` — absorb tail risk into mean.
    * ``γ(μ', σ², 0, 0) = (μ', σ², 0, 0)`` — embed as full cost (conservative).
    """
    def alpha(c: CostElement) -> CostElement:
        return CostElement(
            mu=c.mu * (1.0 + c.lambda_),
            sigma_sq=c.sigma_sq,
            kappa=0.0,
            lambda_=0.0,
        )

    def gamma(a: CostElement) -> CostElement:
        return CostElement(
            mu=a.mu,
            sigma_sq=a.sigma_sq,
            kappa=0.0,
            lambda_=0.0,
        )

    return CostGaloisConnection(
        abstraction=CostAlgebraHomomorphism(map_fn=alpha, name="α_{μσ²}"),
        concretisation=CostAlgebraHomomorphism(map_fn=gamma, name="γ_{μσ²}"),
        name="moment_projection",
    )


# ---------------------------------------------------------------------------
# Quotient algebra
# ---------------------------------------------------------------------------


@dataclass
class QuotientAlgebra:
    r"""A quotient of the cost algebra by an equivalence relation.

    The equivalence relation is defined by a projection ``π`` that maps
    equivalent elements to the same representative.

    Parameters
    ----------
    projection : CostAlgebraHomomorphism
        The projection defining the quotient: ``a ~ b ⟺ π(a) = π(b)``.
    name : str
        Human-readable name.
    """

    projection: CostAlgebraHomomorphism
    name: str = ""

    def canonical(self, element: CostElement) -> CostElement:
        """Map to the canonical representative of the equivalence class."""
        return self.projection.apply(element)

    def equivalent(self, a: CostElement, b: CostElement, tol: float = 1e-10) -> bool:
        """Test if ``a`` and ``b`` are in the same equivalence class."""
        pa = self.canonical(a)
        pb = self.canonical(b)
        return (
            abs(pa.mu - pb.mu) < tol
            and abs(pa.sigma_sq - pb.sigma_sq) < tol
            and abs(pa.kappa - pb.kappa) < tol
            and abs(pa.lambda_ - pb.lambda_) < tol
        )

    def partition(
        self, elements: Sequence[CostElement], tol: float = 1e-10
    ) -> List[List[CostElement]]:
        """Partition elements into equivalence classes."""
        classes: List[Tuple[CostElement, List[CostElement]]] = []
        for e in elements:
            pe = self.canonical(e)
            found = False
            for rep, members in classes:
                if (abs(rep.mu - pe.mu) < tol
                        and abs(rep.sigma_sq - pe.sigma_sq) < tol
                        and abs(rep.kappa - pe.kappa) < tol
                        and abs(rep.lambda_ - pe.lambda_) < tol):
                    members.append(e)
                    found = True
                    break
            if not found:
                classes.append((pe, [e]))
        return [members for _, members in classes]


def mean_quotient() -> QuotientAlgebra:
    """Quotient by mean cost: elements with the same μ are identified."""
    return QuotientAlgebra(projection=mean_projection(), name="quotient_μ")


def moment_quotient() -> QuotientAlgebra:
    """Quotient by first two moments: same (μ, σ²) → same class."""
    return QuotientAlgebra(projection=moment_projection(), name="quotient_{μ,σ²}")


# ---------------------------------------------------------------------------
# Kernel and image
# ---------------------------------------------------------------------------


def kernel(
    h: CostAlgebraHomomorphism,
    elements: Sequence[CostElement],
    tol: float = 1e-10,
) -> List[CostElement]:
    r"""Compute the kernel: ``ker(h) = {x : h(x) ≈ 0}``."""
    z = CostElement.zero()
    return [
        e for e in elements
        if (abs(h.apply(e).mu - z.mu) < tol
            and abs(h.apply(e).sigma_sq - z.sigma_sq) < tol
            and abs(h.apply(e).lambda_ - z.lambda_) < tol)
    ]


def image(
    h: CostAlgebraHomomorphism,
    elements: Sequence[CostElement],
) -> List[CostElement]:
    r"""Compute the image: ``im(h) = {h(x) : x ∈ elements}``."""
    return [h.apply(e) for e in elements]


def image_unique(
    h: CostAlgebraHomomorphism,
    elements: Sequence[CostElement],
    tol: float = 1e-10,
) -> List[CostElement]:
    r"""Compute the image with deduplication."""
    seen: List[CostElement] = []
    for e in elements:
        he = h.apply(e)
        is_dup = False
        for s in seen:
            if (abs(he.mu - s.mu) < tol
                    and abs(he.sigma_sq - s.sigma_sq) < tol
                    and abs(he.kappa - s.kappa) < tol
                    and abs(he.lambda_ - s.lambda_) < tol):
                is_dup = True
                break
        if not is_dup:
            seen.append(he)
    return seen


# ---------------------------------------------------------------------------
# Soundness of abstraction
# ---------------------------------------------------------------------------


def prove_abstraction_soundness(
    h: CostAlgebraHomomorphism,
    samples: Sequence[Tuple[CostElement, CostElement]],
    coupling: float = 0.0,
    interference: float = 0.0,
    tol: float = 1e-6,
) -> Dict[str, bool]:
    r"""Verify soundness of a cost abstraction across both composition operators.

    Checks:
    1. Sequential homomorphism property.
    2. Parallel homomorphism property.
    3. Identity preservation.
    4. Monotonicity: ``a ⊑ b ⟹ h(a) ⊑ h(b)`` (tested on samples).

    Returns
    -------
    dict[str, bool]
        Mapping from property name to verification result.
    """
    from usability_oracle.algebra.lattice import cost_leq

    results: Dict[str, bool] = {}

    results["identity_preservation"] = verify_identity_preservation(h, tol)
    results["sequential_homomorphism"] = verify_sequential_homomorphism(
        h, samples, coupling=coupling, tol=tol
    )
    results["parallel_homomorphism"] = verify_parallel_homomorphism(
        h, samples, interference=interference, tol=tol
    )

    # Monotonicity check on comparable pairs
    monotone = True
    for a, b in samples:
        if cost_leq(a, b, tol):
            if not cost_leq(h.apply(a), h.apply(b), tol):
                monotone = False
                break
    results["monotonicity"] = monotone

    return results
