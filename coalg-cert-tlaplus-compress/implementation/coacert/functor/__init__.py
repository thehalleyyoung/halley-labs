"""
Coalgebraic functor engine for CoaCert-TLA.

Provides F-coalgebra representation, polynomial functor algebra,
stuttering closure monad, T-Fair coherence checking, morphism
computation, and behavioral equivalence.

The base functor is F(X) = P(AP) × P(X)^Act × Fair(X).
"""

from .coalgebra import (
    CoalgebraState,
    FunctorValue,
    FCoalgebra,
    CoalgebraMorphismCheck,
    QuotientCoalgebra,
    SubCoalgebra,
    ProductCoalgebra,
)
from .polynomial import (
    FunctorComponent,
    PowersetFunctor,
    ExponentialFunctor,
    ProductFunctor,
    CoproductFunctor,
    ConstantFunctor,
    FairnessFunctor,
    CompositeFunctor,
    NaturalTransformation,
    KripkeFairnessFunctor,
)
from .stutter import (
    StutterMonad,
    StutterPath,
    StutterEquivalenceClass,
    StutterClosedTransition,
)
from .tfair_coherence import (
    TFairCoherenceChecker,
    CoherenceWitness,
    CoherenceViolation,
    CategoricalCoherenceDiagram,
)
from .morphism import (
    MorphismFinder,
    CoalgebraMorphism,
    MorphismComposition,
)
from .behavioral_equiv import (
    BehavioralEquivalence,
    PartitionRefinement,
    EquivalenceClass,
)

__all__ = [
    "CoalgebraState",
    "FunctorValue",
    "FCoalgebra",
    "CoalgebraMorphismCheck",
    "QuotientCoalgebra",
    "SubCoalgebra",
    "ProductCoalgebra",
    "FunctorComponent",
    "PowersetFunctor",
    "ExponentialFunctor",
    "ProductFunctor",
    "CoproductFunctor",
    "ConstantFunctor",
    "FairnessFunctor",
    "CompositeFunctor",
    "NaturalTransformation",
    "KripkeFairnessFunctor",
    "StutterMonad",
    "StutterPath",
    "StutterEquivalenceClass",
    "StutterClosedTransition",
    "TFairCoherenceChecker",
    "CoherenceWitness",
    "CoherenceViolation",
    "CategoricalCoherenceDiagram",
    "MorphismFinder",
    "CoalgebraMorphism",
    "MorphismComposition",
    "BehavioralEquivalence",
    "PartitionRefinement",
    "EquivalenceClass",
]
