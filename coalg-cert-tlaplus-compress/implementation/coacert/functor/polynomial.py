"""
Polynomial functor algebra for CoaCert-TLA.

Polynomial functors are built from constant, identity, powerset,
exponential, product, coproduct, fairness, and composite combinators.

The Kripke-fairness functor for CoaCert is:
  F(X) = P(AP) × P(X)^Act × Fair(X)
where Fair(X) = (P(X) × P(X))^n for n acceptance pairs.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

logger = logging.getLogger(__name__)

X = TypeVar("X")
Y = TypeVar("Y")


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class FunctorComponent(ABC):
    """Abstract base class for polynomial functor components.

    A functor F maps:
      - objects (sets)   X  ↦ F(X)
      - morphisms (functions) f: X→Y  ↦ F(f): F(X)→F(Y)
    """

    @abstractmethod
    def apply(self, carrier: FrozenSet[str]) -> Any:
        """Apply the functor to a set (object mapping)."""
        ...

    @abstractmethod
    def fmap(
        self, f: Mapping[str, str], value: Any, source: FrozenSet[str]
    ) -> Any:
        """Apply the functorial action on morphisms: F(f)(value)."""
        ...

    @abstractmethod
    def signature(self) -> str:
        """Human-readable representation of this functor component."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.signature()})"

    def preserves_surjections(
        self,
        source: FrozenSet[str],
        target: FrozenSet[str],
        f: Mapping[str, str],
    ) -> bool:
        """Check whether F preserves surjectivity of f.

        f: source ↠ target is surjective. We check if
        F(f): F(source) → F(target) is also surjective.
        For polynomial functors this always holds (they are accessible).
        This method can be overridden for non-standard functors.
        """
        f_target_range = frozenset(f[s] for s in source if s in f)
        if f_target_range != target:
            return False  # f is not surjective
        return True  # polynomial functors preserve surjections


# ---------------------------------------------------------------------------
# Constant functor
# ---------------------------------------------------------------------------

class ConstantFunctor(FunctorComponent):
    """Constant functor K_A(X) = A for a fixed set A.

    Used for atomic propositions: K_{AP}(X) = AP regardless of X.
    """

    def __init__(self, constant_set: FrozenSet[str], label: str = "K"):
        self.constant_set = constant_set
        self.label = label

    def apply(self, carrier: FrozenSet[str]) -> FrozenSet[str]:
        return self.constant_set

    def fmap(
        self, f: Mapping[str, str], value: Any, source: FrozenSet[str]
    ) -> Any:
        return value  # constant functor acts as identity on morphisms

    def signature(self) -> str:
        elems = ", ".join(sorted(self.constant_set)[:5])
        if len(self.constant_set) > 5:
            elems += ", ..."
        return f"{self.label}{{{elems}}}"

    def element_count(self) -> int:
        return len(self.constant_set)


# ---------------------------------------------------------------------------
# Powerset functor
# ---------------------------------------------------------------------------

class PowersetFunctor(FunctorComponent):
    """Powerset functor P(X) = {S ⊆ X : S finite}.

    The finite powerset functor. fmap sends subsets through a function:
      P(f)(S) = {f(x) : x ∈ S}
    """

    def __init__(self, label: str = "P"):
        self.label = label

    def apply(self, carrier: FrozenSet[str]) -> FrozenSet[FrozenSet[str]]:
        """Return the powerset of carrier (only feasible for small sets)."""
        elements = list(carrier)
        n = len(elements)
        if n > 20:
            logger.warning(
                "Powerset of %d elements is very large; returning lazily", n
            )
            return frozenset()  # placeholder for large sets

        result: Set[FrozenSet[str]] = set()
        for mask in range(1 << n):
            subset = frozenset(elements[i] for i in range(n) if mask & (1 << i))
            result.add(subset)
        return frozenset(result)

    def fmap(
        self, f: Mapping[str, str], value: Any, source: FrozenSet[str]
    ) -> FrozenSet[str]:
        """P(f)(S) = {f(x) : x ∈ S}."""
        if not isinstance(value, (set, frozenset)):
            raise TypeError(f"Expected set, got {type(value)}")
        return frozenset(f.get(x, x) for x in value)

    def signature(self) -> str:
        return f"{self.label}(X)"

    def contains(self, subset: FrozenSet[str], carrier: FrozenSet[str]) -> bool:
        return subset <= carrier

    def image(
        self, f: Mapping[str, str], subsets: FrozenSet[FrozenSet[str]]
    ) -> FrozenSet[FrozenSet[str]]:
        """Apply P(f) to a collection of subsets."""
        return frozenset(
            frozenset(f.get(x, x) for x in s) for s in subsets
        )

    def preimage(
        self,
        f: Mapping[str, str],
        target_subset: FrozenSet[str],
        source: FrozenSet[str],
    ) -> FrozenSet[str]:
        """Compute f^{-1}(target_subset)."""
        return frozenset(s for s in source if f.get(s, s) in target_subset)


# ---------------------------------------------------------------------------
# Exponential functor
# ---------------------------------------------------------------------------

class ExponentialFunctor(FunctorComponent):
    """Exponential functor X^A for a fixed exponent set A.

    Elements of X^A are functions a ↦ x(a) for a ∈ A, x(a) ∈ X.
    In our setting with nondeterminism we use P(X)^A instead, represented as
    dictionaries {a: set_of_x}.
    """

    def __init__(self, exponent: FrozenSet[str], label: str = "Exp"):
        self.exponent = exponent
        self.label = label

    def apply(self, carrier: FrozenSet[str]) -> str:
        """Description of the applied functor (full enumeration is impractical)."""
        return f"P({carrier})^{self.exponent}"

    def fmap(
        self,
        f: Mapping[str, str],
        value: Dict[str, FrozenSet[str]],
        source: FrozenSet[str],
    ) -> Dict[str, FrozenSet[str]]:
        """(P(X)^A → P(Y)^A) given f: X → Y.

        For each action a, apply P(f) to the successor set.
        """
        result: Dict[str, FrozenSet[str]] = {}
        for a in self.exponent:
            s_set = value.get(a, frozenset())
            result[a] = frozenset(f.get(x, x) for x in s_set)
        return result

    def signature(self) -> str:
        acts = ", ".join(sorted(self.exponent)[:5])
        if len(self.exponent) > 5:
            acts += ", ..."
        return f"P(X)^{{{acts}}}"

    def is_total(self, value: Dict[str, FrozenSet[str]]) -> bool:
        """Check if the function is defined for all actions."""
        return all(a in value for a in self.exponent)

    def restrict(
        self,
        value: Dict[str, FrozenSet[str]],
        allowed: FrozenSet[str],
    ) -> Dict[str, FrozenSet[str]]:
        """Restrict successor sets to allowed states."""
        return {
            a: targets & allowed for a, targets in value.items()
        }

    def merge(
        self,
        v1: Dict[str, FrozenSet[str]],
        v2: Dict[str, FrozenSet[str]],
    ) -> Dict[str, FrozenSet[str]]:
        """Union of successor sets for each action (nondeterministic merge)."""
        result: Dict[str, FrozenSet[str]] = {}
        all_actions = set(v1.keys()) | set(v2.keys())
        for a in all_actions:
            s1 = v1.get(a, frozenset())
            s2 = v2.get(a, frozenset())
            result[a] = s1 | s2
        return result


# ---------------------------------------------------------------------------
# Product functor
# ---------------------------------------------------------------------------

class ProductFunctor(FunctorComponent):
    """Product functor (F × G)(X) = F(X) × G(X).

    Elements are pairs (f_val, g_val).
    """

    def __init__(self, left: FunctorComponent, right: FunctorComponent):
        self.left = left
        self.right = right

    def apply(self, carrier: FrozenSet[str]) -> Tuple[Any, Any]:
        return (self.left.apply(carrier), self.right.apply(carrier))

    def fmap(
        self,
        f: Mapping[str, str],
        value: Tuple[Any, Any],
        source: FrozenSet[str],
    ) -> Tuple[Any, Any]:
        left_val, right_val = value
        return (
            self.left.fmap(f, left_val, source),
            self.right.fmap(f, right_val, source),
        )

    def signature(self) -> str:
        return f"({self.left.signature()} × {self.right.signature()})"

    def project_left(self, value: Tuple[Any, Any]) -> Any:
        return value[0]

    def project_right(self, value: Tuple[Any, Any]) -> Any:
        return value[1]

    def pair(self, left_val: Any, right_val: Any) -> Tuple[Any, Any]:
        return (left_val, right_val)


# ---------------------------------------------------------------------------
# Coproduct functor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoproductValue:
    """Tagged value in a coproduct F + G."""
    tag: str  # "left" or "right"
    value: Any


class CoproductFunctor(FunctorComponent):
    """Coproduct functor (F + G)(X) = F(X) ⊔ G(X).

    Elements are tagged values: Left(f_val) or Right(g_val).
    """

    def __init__(self, left: FunctorComponent, right: FunctorComponent):
        self.left = left
        self.right = right

    def apply(self, carrier: FrozenSet[str]) -> Tuple[Any, Any]:
        return (self.left.apply(carrier), self.right.apply(carrier))

    def fmap(
        self,
        f: Mapping[str, str],
        value: CoproductValue,
        source: FrozenSet[str],
    ) -> CoproductValue:
        if value.tag == "left":
            return CoproductValue("left", self.left.fmap(f, value.value, source))
        else:
            return CoproductValue("right", self.right.fmap(f, value.value, source))

    def signature(self) -> str:
        return f"({self.left.signature()} + {self.right.signature()})"

    def inject_left(self, val: Any) -> CoproductValue:
        return CoproductValue("left", val)

    def inject_right(self, val: Any) -> CoproductValue:
        return CoproductValue("right", val)

    def case(
        self,
        value: CoproductValue,
        on_left: Callable[[Any], Any],
        on_right: Callable[[Any], Any],
    ) -> Any:
        if value.tag == "left":
            return on_left(value.value)
        return on_right(value.value)


# ---------------------------------------------------------------------------
# Fairness functor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FairnessValue:
    """Value in Fair(X) = (P(X) × P(X))^n.

    A list of acceptance pairs, each pair being (B_i, G_i) ⊆ X.
    """

    pairs: Tuple[Tuple[FrozenSet[str], FrozenSet[str]], ...]

    def pair_count(self) -> int:
        return len(self.pairs)

    def get_pair(self, index: int) -> Tuple[FrozenSet[str], FrozenSet[str]]:
        return self.pairs[index]

    def b_set(self, index: int) -> FrozenSet[str]:
        return self.pairs[index][0]

    def g_set(self, index: int) -> FrozenSet[str]:
        return self.pairs[index][1]


class FairnessFunctor(FunctorComponent):
    """Fairness functor Fair(X) = (P(X) × P(X))^n for n acceptance pairs.

    Each element is a tuple of (B_i, G_i) subsets of X.
    fmap applies the underlying function to each subset in each pair.
    """

    def __init__(self, num_pairs: int, label: str = "Fair"):
        self.num_pairs = num_pairs
        self.label = label

    def apply(self, carrier: FrozenSet[str]) -> str:
        return f"(P({carrier}) × P({carrier}))^{self.num_pairs}"

    def fmap(
        self,
        f: Mapping[str, str],
        value: FairnessValue,
        source: FrozenSet[str],
    ) -> FairnessValue:
        """Fair(f)((B_1,G_1),...,(B_n,G_n)) = (f(B_1),f(G_1)),...,(f(B_n),f(G_n))."""
        new_pairs: List[Tuple[FrozenSet[str], FrozenSet[str]]] = []
        for b_set, g_set in value.pairs:
            new_b = frozenset(f.get(x, x) for x in b_set)
            new_g = frozenset(f.get(x, x) for x in g_set)
            new_pairs.append((new_b, new_g))
        return FairnessValue(pairs=tuple(new_pairs))

    def signature(self) -> str:
        return f"{self.label}_{self.num_pairs}(X)"

    def empty_value(self) -> FairnessValue:
        return FairnessValue(
            pairs=tuple((frozenset(), frozenset()) for _ in range(self.num_pairs))
        )

    def from_sets(
        self, pairs: List[Tuple[Set[str], Set[str]]]
    ) -> FairnessValue:
        if len(pairs) != self.num_pairs:
            raise ValueError(
                f"Expected {self.num_pairs} pairs, got {len(pairs)}"
            )
        return FairnessValue(
            pairs=tuple(
                (frozenset(b), frozenset(g)) for b, g in pairs
            )
        )

    def restrict(
        self, value: FairnessValue, allowed: FrozenSet[str]
    ) -> FairnessValue:
        new_pairs = []
        for b_set, g_set in value.pairs:
            new_pairs.append((b_set & allowed, g_set & allowed))
        return FairnessValue(pairs=tuple(new_pairs))

    def merge(self, v1: FairnessValue, v2: FairnessValue) -> FairnessValue:
        """Union merge of corresponding pairs."""
        if v1.pair_count() != v2.pair_count():
            raise ValueError("Cannot merge fairness values with different pair counts")
        new_pairs = []
        for (b1, g1), (b2, g2) in zip(v1.pairs, v2.pairs):
            new_pairs.append((b1 | b2, g1 | g2))
        return FairnessValue(pairs=tuple(new_pairs))

    def preserves_acceptance(
        self,
        f: Mapping[str, str],
        value: FairnessValue,
        source: FrozenSet[str],
    ) -> bool:
        """Check that mapping f preserves the acceptance structure.

        For Streett: if a state is in B_i, its image must be in f(B_i),
        and similarly for G_i.
        """
        mapped = self.fmap(f, value, source)
        for i in range(self.num_pairs):
            orig_b, orig_g = value.pairs[i]
            map_b, map_g = mapped.pairs[i]
            for s in orig_b:
                if f.get(s, s) not in map_b:
                    return False
            for s in orig_g:
                if f.get(s, s) not in map_g:
                    return False
        return True


# ---------------------------------------------------------------------------
# Composite functor
# ---------------------------------------------------------------------------

class CompositeFunctor(FunctorComponent):
    """Composite functor (F ∘ G)(X) = F(G(X)).

    Composition of two functor components.
    """

    def __init__(self, outer: FunctorComponent, inner: FunctorComponent):
        self.outer = outer
        self.inner = inner

    def apply(self, carrier: FrozenSet[str]) -> Any:
        inner_result = self.inner.apply(carrier)
        if isinstance(inner_result, frozenset):
            return self.outer.apply(inner_result)
        return f"{self.outer.signature()}({inner_result})"

    def fmap(
        self,
        f: Mapping[str, str],
        value: Any,
        source: FrozenSet[str],
    ) -> Any:
        """(F ∘ G)(f) = F(G(f)).

        This is tricky in general; we apply the inner fmap first,
        then the outer fmap on the result.
        """
        inner_mapped = self.inner.fmap(f, value, source)
        target = frozenset(f.get(s, s) for s in source)
        return self.outer.fmap(f, inner_mapped, target)

    def signature(self) -> str:
        return f"{self.outer.signature()} ∘ {self.inner.signature()}"


# ---------------------------------------------------------------------------
# Natural transformation
# ---------------------------------------------------------------------------

class NaturalTransformation:
    """A natural transformation α: F ⇒ G.

    For each set X, α_X : F(X) → G(X) such that the naturality square
    commutes: G(f) ∘ α_X = α_Y ∘ F(f) for all f: X → Y.
    """

    def __init__(
        self,
        source: FunctorComponent,
        target: FunctorComponent,
        components: Optional[Dict[str, Callable]] = None,
        name: str = "α",
    ):
        self.source = source
        self.target = target
        self._components: Dict[str, Callable] = components or {}
        self.name = name
        self._generic_component: Optional[Callable] = None

    def set_generic_component(self, component: Callable[[Any], Any]) -> None:
        """Set a component that works for any carrier set."""
        self._generic_component = component

    def apply(self, carrier_name: str, value: Any) -> Any:
        """Apply α_X to a value in F(X)."""
        if carrier_name in self._components:
            return self._components[carrier_name](value)
        if self._generic_component is not None:
            return self._generic_component(value)
        raise KeyError(
            f"No component defined for carrier '{carrier_name}'"
        )

    def check_naturality(
        self,
        f: Mapping[str, str],
        source_carrier: FrozenSet[str],
        target_carrier: FrozenSet[str],
        test_values: List[Any],
    ) -> bool:
        """Verify the naturality condition for specific test values.

        Check: G(f) ∘ α_X = α_Y ∘ F(f) for each test value.
        """
        src_name = str(sorted(source_carrier))
        tgt_name = str(sorted(target_carrier))

        for val in test_values:
            # Path 1: α_Y(F(f)(val))
            ff_val = self.source.fmap(f, val, source_carrier)
            path1 = self.apply(tgt_name, ff_val)

            # Path 2: G(f)(α_X(val))
            alpha_val = self.apply(src_name, val)
            path2 = self.target.fmap(f, alpha_val, source_carrier)

            if path1 != path2:
                logger.warning(
                    "Naturality violation for value %s: α_Y∘F(f) = %s ≠ G(f)∘α_X = %s",
                    val,
                    path1,
                    path2,
                )
                return False

        return True

    def compose(self, other: "NaturalTransformation") -> "NaturalTransformation":
        """Vertical composition: (β ∘ α)_X = β_X ∘ α_X."""
        composed = NaturalTransformation(
            source=self.source,
            target=other.target,
            name=f"{other.name}∘{self.name}",
        )
        if self._generic_component and other._generic_component:
            alpha_gen = self._generic_component
            beta_gen = other._generic_component
            composed.set_generic_component(lambda v: beta_gen(alpha_gen(v)))
        return composed

    def __repr__(self) -> str:
        return (
            f"NaturalTransformation({self.name}: "
            f"{self.source.signature()} ⇒ {self.target.signature()})"
        )


# ---------------------------------------------------------------------------
# Kripke-fairness functor (the specific F for CoaCert)
# ---------------------------------------------------------------------------

class KripkeFairnessFunctor(FunctorComponent):
    """The specific functor F(X) = P(AP) × P(X)^Act × Fair(X).

    Combines ConstantFunctor, ExponentialFunctor (with powerset),
    and FairnessFunctor into the product used by CoaCert-TLA.
    """

    def __init__(
        self,
        atomic_propositions: FrozenSet[str],
        actions: FrozenSet[str],
        num_fairness_pairs: int,
    ):
        self.ap = atomic_propositions
        self.actions = actions
        self.num_fairness_pairs = num_fairness_pairs

        self._const = ConstantFunctor(atomic_propositions, label="AP")
        self._exp = ExponentialFunctor(actions, label="Succ")
        self._fair = FairnessFunctor(num_fairness_pairs, label="Fair")

        self._product = ProductFunctor(
            ProductFunctor(self._const, self._exp),
            self._fair,
        )

    def apply(self, carrier: FrozenSet[str]) -> str:
        return (
            f"P({sorted(self.ap)}) × P({carrier})^{sorted(self.actions)} "
            f"× Fair_{self.num_fairness_pairs}({carrier})"
        )

    def fmap(
        self,
        f: Mapping[str, str],
        value: Tuple[FrozenSet[str], Dict[str, FrozenSet[str]], FairnessValue],
        source: FrozenSet[str],
    ) -> Tuple[FrozenSet[str], Dict[str, FrozenSet[str]], FairnessValue]:
        """F(f)((props, succ, fair)) = (props, succ∘f, Fair(f)(fair))."""
        props, succ, fair = value
        new_props = props  # constant component
        new_succ = self._exp.fmap(f, succ, source)
        new_fair = self._fair.fmap(f, fair, source)
        return (new_props, new_succ, new_fair)

    def signature(self) -> str:
        return f"P(AP) × P(X)^Act × Fair_{self.num_fairness_pairs}(X)"

    def make_value(
        self,
        propositions: FrozenSet[str],
        successors: Dict[str, FrozenSet[str]],
        fairness: FairnessValue,
    ) -> Tuple[FrozenSet[str], Dict[str, FrozenSet[str]], FairnessValue]:
        return (propositions, successors, fairness)

    def decompose(
        self, value: Tuple[FrozenSet[str], Dict[str, FrozenSet[str]], FairnessValue]
    ) -> Tuple[FrozenSet[str], Dict[str, FrozenSet[str]], FairnessValue]:
        return value

    @property
    def constant_component(self) -> ConstantFunctor:
        return self._const

    @property
    def exponential_component(self) -> ExponentialFunctor:
        return self._exp

    @property
    def fairness_component(self) -> FairnessFunctor:
        return self._fair

    def check_value_well_formed(
        self,
        value: Tuple[FrozenSet[str], Dict[str, FrozenSet[str]], FairnessValue],
        carrier: FrozenSet[str],
    ) -> List[str]:
        """Validate that a functor value is well-formed w.r.t. a carrier."""
        errors: List[str] = []
        props, succ, fair = value

        if not props <= self.ap:
            errors.append(
                f"Propositions {props - self.ap} not in AP"
            )

        for act, targets in succ.items():
            if act not in self.actions:
                errors.append(f"Unknown action '{act}'")
            outside = targets - carrier
            if outside:
                errors.append(
                    f"Successor targets {outside} not in carrier for action '{act}'"
                )

        if fair.pair_count() != self.num_fairness_pairs:
            errors.append(
                f"Expected {self.num_fairness_pairs} fairness pairs, "
                f"got {fair.pair_count()}"
            )
        else:
            for i in range(self.num_fairness_pairs):
                b, g = fair.get_pair(i)
                outside_b = b - carrier
                outside_g = g - carrier
                if outside_b:
                    errors.append(f"B_{i} contains states {outside_b} not in carrier")
                if outside_g:
                    errors.append(f"G_{i} contains states {outside_g} not in carrier")

        return errors
