"""Transform registry mapping sklearn/pandas operations to information-theoretic properties.

Every sklearn transformer and pandas method that can appear in an ML pipeline
is catalogued here with its *information-theoretic* signature: whether it is
row-independent or requires the full dataset, whether it is invertible or
lossy, and a capacity model string that downstream analysis can use to compute
tight leakage bounds.

The registry is the single source of truth consumed by the DAG builder and the
capacity analyser.
"""

from __future__ import annotations

import copy
import enum
import logging
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from taintflow.core.types import OpType, Origin, Severity

if TYPE_CHECKING:
    from taintflow.dag.pidag import PIDAG

logger = logging.getLogger(__name__)

__all__ = [
    "TransformCategory",
    "TransformSignature",
    "TransformProperty",
    "TransformRegistryEntry",
    "TransformRegistry",
    "default_registry",
]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TransformCategory(enum.Enum):
    """High-level category for a pipeline transform."""

    SCALING = "scaling"
    IMPUTATION = "imputation"
    ENCODING = "encoding"
    FEATURE_SELECTION = "feature_selection"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    AGGREGATION = "aggregation"
    FILTERING = "filtering"
    CUSTOM = "custom"
    IDENTITY = "identity"
    COMPOSITION = "composition"

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_str(cls, text: str) -> "TransformCategory":
        """Case-insensitive lookup by value or name."""
        key = text.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == key or member.name.lower() == key:
                return member
        raise ValueError(f"Unknown TransformCategory: {text!r}")

    def is_lossy(self) -> bool:
        """Return *True* when the category is inherently lossy."""
        return self in {
            TransformCategory.AGGREGATION,
            TransformCategory.FILTERING,
            TransformCategory.FEATURE_SELECTION,
            TransformCategory.DIMENSIONALITY_REDUCTION,
        }

    def may_leak(self) -> bool:
        """Heuristic: categories whose *fit* step can channel test info."""
        return self in {
            TransformCategory.SCALING,
            TransformCategory.IMPUTATION,
            TransformCategory.ENCODING,
            TransformCategory.FEATURE_SELECTION,
            TransformCategory.DIMENSIONALITY_REDUCTION,
            TransformCategory.AGGREGATION,
        }


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransformSignature:
    """Structural signature of a single pipeline transform.

    Parameters
    ----------
    name:
        Human-readable canonical name (e.g. ``"StandardScaler"``).
    category:
        Which :class:`TransformCategory` this transform belongs to.
    input_arity:
        Expected number of input arrays (1 for most sklearn transforms,
        2 for operations like merge).
    output_arity:
        Number of output arrays produced.
    is_fitted:
        Whether the transform has a *fit* step that computes statistics
        from data.
    fitted_params:
        Names of the parameters stored after fitting (e.g.
        ``["mean_", "var_"]`` for ``StandardScaler``).
    statistic_type:
        ``"sufficient"`` when the fitted parameters are a sufficient
        statistic of the training data, ``"non-sufficient"`` otherwise.
    data_dependency:
        ``"row-independent"`` – each output row depends only on its
        corresponding input row.
        ``"row-dependent"`` – each output row may depend on a *window*
        of other rows.
        ``"all-rows"`` – fitting or transforming requires the complete
        dataset.
    """

    name: str
    category: TransformCategory
    input_arity: int = 1
    output_arity: int = 1
    is_fitted: bool = True
    fitted_params: Tuple[str, ...] = ()
    statistic_type: str = "sufficient"
    data_dependency: str = "all-rows"

    # ------------------------------------------------------------------

    def __post_init__(self) -> None:  # pragma: no cover – validation only
        if self.data_dependency not in ("row-independent", "row-dependent", "all-rows"):
            raise ValueError(
                f"Invalid data_dependency {self.data_dependency!r} for {self.name}"
            )
        if self.statistic_type not in ("sufficient", "non-sufficient"):
            raise ValueError(
                f"Invalid statistic_type {self.statistic_type!r} for {self.name}"
            )

    @property
    def is_stateless(self) -> bool:
        """True when the transform needs no fitting at all."""
        return not self.is_fitted

    @property
    def is_row_independent(self) -> bool:
        return self.data_dependency == "row-independent"


@dataclass(frozen=True)
class TransformProperty:
    """Information-theoretic properties attached to a transform.

    These are the *mathematical* invariants that the capacity analyser
    relies on.

    Parameters
    ----------
    is_monotone:
        The mapping preserves the partial order on information content.
    is_invertible:
        There exists an inverse mapping (no information is lost).
    preserves_independence:
        Statistically independent inputs remain independent after
        transformation.
    max_info_gain_fn:
        Callable signature string for the upper bound on mutual
        information gained through this transform, e.g.
        ``"log2(n_features)"`` or ``"0"`` for identity.
    min_info_loss_fn:
        Callable signature string for the lower bound on information
        lost, e.g. ``"n_features - n_components"`` for PCA.
    is_pointwise:
        The transform operates element-wise (no cross-row or
        cross-column dependencies during *transform*).
    requires_all_data:
        Whether *fit* requires all rows simultaneously (as opposed to
        streaming / mini-batch).
    fit_uses_target:
        If *True* the *fit* step consumes ``y`` and therefore has
        a direct channel from the target variable.
    """

    is_monotone: bool = True
    is_invertible: bool = False
    preserves_independence: bool = True
    max_info_gain_fn: str = "0"
    min_info_loss_fn: str = "0"
    is_pointwise: bool = False
    requires_all_data: bool = False
    fit_uses_target: bool = False

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def is_lossless(self) -> bool:
        return self.is_invertible

    @property
    def is_supervised_fit(self) -> bool:
        return self.fit_uses_target

    def worst_case_capacity_description(self) -> str:
        """Human-readable worst-case capacity note."""
        parts: list[str] = []
        if self.fit_uses_target:
            parts.append("supervised-fit (target leak risk)")
        if self.requires_all_data:
            parts.append("global-fit (all-rows dependency)")
        if not self.is_invertible:
            parts.append("lossy")
        if not parts:
            return "benign"
        return "; ".join(parts)


@dataclass(frozen=True)
class TransformRegistryEntry:
    """Complete catalogue entry for one transform.

    Bundles the structural signature with the information-theoretic
    properties and book-keeping metadata.
    """

    signature: TransformSignature
    properties: TransformProperty
    sklearn_class_name: str = ""
    pandas_method_name: str = ""
    capacity_model: str = "identity"
    notes: str = ""

    @property
    def name(self) -> str:
        return self.signature.name

    @property
    def category(self) -> TransformCategory:
        return self.signature.category

    def summary(self) -> str:
        """One-liner for logging and reports."""
        src = self.sklearn_class_name or self.pandas_method_name or "custom"
        return (
            f"{self.signature.name} ({src}) – "
            f"capacity_model={self.capacity_model}, "
            f"dep={self.signature.data_dependency}"
        )


# ---------------------------------------------------------------------------
# Registry implementation
# ---------------------------------------------------------------------------


class TransformRegistry:
    """Central catalogue of transforms and their properties.

    The registry supports look-up by *sklearn class name* (e.g.
    ``"sklearn.preprocessing.StandardScaler"``), by *pandas method name*
    (e.g. ``"DataFrame.fillna"``), or by canonical *name*.
    """

    def __init__(self) -> None:
        self._by_name: Dict[str, TransformRegistryEntry] = {}
        self._by_sklearn: Dict[str, str] = {}
        self._by_pandas: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, entry: TransformRegistryEntry, *, overwrite: bool = False) -> None:
        """Add *entry* to the registry.

        Raises :class:`ValueError` if the name is already registered
        and *overwrite* is ``False``.
        """
        name = entry.signature.name
        if name in self._by_name and not overwrite:
            raise ValueError(f"Transform {name!r} is already registered")
        self._by_name[name] = entry
        if entry.sklearn_class_name:
            self._by_sklearn[entry.sklearn_class_name] = name
        if entry.pandas_method_name:
            self._by_pandas[entry.pandas_method_name] = name
        logger.debug("Registered transform %s", name)

    def register_many(
        self,
        entries: Iterable[TransformRegistryEntry],
        *,
        overwrite: bool = False,
    ) -> None:
        for entry in entries:
            self.register(entry, overwrite=overwrite)

    # ------------------------------------------------------------------
    # Look-up
    # ------------------------------------------------------------------

    def lookup(self, name: str) -> Optional[TransformRegistryEntry]:
        """Look up by canonical name."""
        return self._by_name.get(name)

    def lookup_by_sklearn_class(self, class_name: str) -> Optional[TransformRegistryEntry]:
        """Look up by fully-qualified sklearn class name."""
        canon = self._by_sklearn.get(class_name)
        if canon is None:
            return None
        return self._by_name.get(canon)

    def lookup_by_pandas_method(self, method_name: str) -> Optional[TransformRegistryEntry]:
        """Look up by pandas method identifier (e.g. ``"DataFrame.fillna"``)."""
        canon = self._by_pandas.get(method_name)
        if canon is None:
            return None
        return self._by_name.get(canon)

    def get_properties(self, name: str) -> Optional[TransformProperty]:
        """Shorthand: return *just* the properties for a named transform."""
        entry = self.lookup(name)
        return entry.properties if entry is not None else None

    def get_capacity_model(self, name: str) -> Optional[str]:
        """Return the capacity-model string for *name*, or ``None``."""
        entry = self.lookup(name)
        return entry.capacity_model if entry is not None else None

    # ------------------------------------------------------------------
    # Enumeration / filtering
    # ------------------------------------------------------------------

    def list_all(self) -> List[TransformRegistryEntry]:
        """Return every entry, sorted by canonical name."""
        return sorted(self._by_name.values(), key=lambda e: e.signature.name)

    def filter_by_category(
        self, category: TransformCategory
    ) -> List[TransformRegistryEntry]:
        """Return entries matching *category*."""
        return [
            e
            for e in self._by_name.values()
            if e.signature.category is category
        ]

    def filter_by_data_dependency(self, dep: str) -> List[TransformRegistryEntry]:
        """Return entries whose ``data_dependency`` equals *dep*."""
        return [
            e
            for e in self._by_name.values()
            if e.signature.data_dependency == dep
        ]

    def supervised_transforms(self) -> List[TransformRegistryEntry]:
        """Return transforms whose fit step uses the target variable."""
        return [
            e
            for e in self._by_name.values()
            if e.properties.fit_uses_target
        ]

    def leaky_transforms(self) -> List[TransformRegistryEntry]:
        """Return transforms that *may* leak information across splits."""
        return [
            e
            for e in self._by_name.values()
            if e.signature.category.may_leak() and e.signature.is_fitted
        ]

    def names(self) -> FrozenSet[str]:
        return frozenset(self._by_name.keys())

    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __iter__(self):
        return iter(self._by_name.values())

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_transform(self, name: str) -> List[str]:
        """Return a list of validation warnings for the entry *name*.

        An empty list means the entry is consistent.
        """
        entry = self.lookup(name)
        if entry is None:
            return [f"Transform {name!r} not found in registry"]

        warnings: list[str] = []
        sig = entry.signature
        props = entry.properties

        if sig.is_fitted and not sig.fitted_params:
            warnings.append(
                f"{name}: is_fitted=True but fitted_params is empty"
            )
        if props.is_invertible and props.min_info_loss_fn != "0":
            warnings.append(
                f"{name}: is_invertible=True but min_info_loss_fn != '0'"
            )
        if props.is_pointwise and sig.data_dependency != "row-independent":
            warnings.append(
                f"{name}: is_pointwise=True but data_dependency "
                f"is {sig.data_dependency!r}"
            )
        if not sig.is_fitted and props.requires_all_data:
            warnings.append(
                f"{name}: not fitted but requires_all_data=True"
            )
        if not sig.is_fitted and props.fit_uses_target:
            warnings.append(
                f"{name}: not fitted but fit_uses_target=True"
            )
        if sig.category is TransformCategory.IDENTITY and sig.is_fitted:
            warnings.append(
                f"{name}: IDENTITY transforms should not require fitting"
            )
        return warnings

    def validate_all(self) -> Dict[str, List[str]]:
        """Validate every registered entry.  Returns a mapping from name
        to the list of warnings (entries with no warnings are omitted).
        """
        result: Dict[str, List[str]] = {}
        for name in self._by_name:
            w = self.validate_transform(name)
            if w:
                result[name] = w
        return result

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Dict]:
        """Serialise the whole registry to a plain dict."""
        out: Dict[str, Dict] = {}
        for name, entry in self._by_name.items():
            out[name] = {
                "sklearn_class_name": entry.sklearn_class_name,
                "pandas_method_name": entry.pandas_method_name,
                "capacity_model": entry.capacity_model,
                "category": entry.signature.category.value,
                "data_dependency": entry.signature.data_dependency,
                "is_fitted": entry.signature.is_fitted,
                "fit_uses_target": entry.properties.fit_uses_target,
            }
        return out

    def copy(self) -> "TransformRegistry":
        """Return a deep copy of this registry."""
        new = TransformRegistry()
        new._by_name = copy.deepcopy(self._by_name)
        new._by_sklearn = dict(self._by_sklearn)
        new._by_pandas = dict(self._by_pandas)
        return new


# ===================================================================
# Pre-populated entries – sklearn
# ===================================================================

def _sklearn_entry(
    name: str,
    class_name: str,
    category: TransformCategory,
    *,
    fitted_params: Tuple[str, ...] = (),
    statistic_type: str = "sufficient",
    data_dependency: str = "all-rows",
    is_fitted: bool = True,
    is_monotone: bool = True,
    is_invertible: bool = False,
    preserves_independence: bool = True,
    max_info_gain_fn: str = "0",
    min_info_loss_fn: str = "0",
    is_pointwise: bool = False,
    requires_all_data: bool = True,
    fit_uses_target: bool = False,
    capacity_model: str = "identity",
    notes: str = "",
) -> TransformRegistryEntry:
    """Helper to build a sklearn-backed registry entry concisely."""
    sig = TransformSignature(
        name=name,
        category=category,
        input_arity=1,
        output_arity=1,
        is_fitted=is_fitted,
        fitted_params=fitted_params,
        statistic_type=statistic_type,
        data_dependency=data_dependency,
    )
    props = TransformProperty(
        is_monotone=is_monotone,
        is_invertible=is_invertible,
        preserves_independence=preserves_independence,
        max_info_gain_fn=max_info_gain_fn,
        min_info_loss_fn=min_info_loss_fn,
        is_pointwise=is_pointwise,
        requires_all_data=requires_all_data,
        fit_uses_target=fit_uses_target,
    )
    return TransformRegistryEntry(
        signature=sig,
        properties=props,
        sklearn_class_name=class_name,
        capacity_model=capacity_model,
        notes=notes,
    )


_SKLEARN_ENTRIES: Tuple[TransformRegistryEntry, ...] = (
    # ------ Scaling ------
    _sklearn_entry(
        "StandardScaler",
        "sklearn.preprocessing.StandardScaler",
        TransformCategory.SCALING,
        fitted_params=("mean_", "var_", "scale_", "n_samples_seen_"),
        is_invertible=True,
        is_pointwise=True,
        capacity_model="linear_shift_scale",
        notes="z = (x - mean) / std; invertible, row-independent transform.",
    ),
    _sklearn_entry(
        "MinMaxScaler",
        "sklearn.preprocessing.MinMaxScaler",
        TransformCategory.SCALING,
        fitted_params=("data_min_", "data_max_", "data_range_", "scale_", "min_"),
        is_invertible=True,
        is_pointwise=True,
        capacity_model="linear_shift_scale",
        notes="Scales to [feature_range]. Invertible affine map.",
    ),
    _sklearn_entry(
        "RobustScaler",
        "sklearn.preprocessing.RobustScaler",
        TransformCategory.SCALING,
        fitted_params=("center_", "scale_"),
        is_invertible=True,
        is_pointwise=True,
        capacity_model="linear_shift_scale",
        notes="Uses median and IQR. Fitted params are non-sufficient statistics.",
        statistic_type="non-sufficient",
    ),
    _sklearn_entry(
        "MaxAbsScaler",
        "sklearn.preprocessing.MaxAbsScaler",
        TransformCategory.SCALING,
        fitted_params=("max_abs_", "scale_", "n_samples_seen_"),
        is_invertible=True,
        is_pointwise=True,
        capacity_model="linear_shift_scale",
        notes="Scales by max absolute value per feature.",
    ),
    _sklearn_entry(
        "Normalizer",
        "sklearn.preprocessing.Normalizer",
        TransformCategory.SCALING,
        is_fitted=False,
        fitted_params=(),
        data_dependency="row-independent",
        is_invertible=False,
        is_pointwise=False,
        requires_all_data=False,
        capacity_model="projection",
        notes="Row-wise L1/L2/max normalisation. Stateless but not invertible.",
    ),
    _sklearn_entry(
        "Binarizer",
        "sklearn.preprocessing.Binarizer",
        TransformCategory.ENCODING,
        is_fitted=False,
        fitted_params=(),
        data_dependency="row-independent",
        is_monotone=True,
        is_invertible=False,
        is_pointwise=True,
        requires_all_data=False,
        capacity_model="threshold",
        min_info_loss_fn="H(X) - 1",
        notes="Threshold binarisation. Massive information loss.",
    ),
    _sklearn_entry(
        "QuantileTransformer",
        "sklearn.preprocessing.QuantileTransformer",
        TransformCategory.SCALING,
        fitted_params=("quantiles_", "references_"),
        statistic_type="non-sufficient",
        is_monotone=True,
        is_invertible=True,
        capacity_model="quantile_mapping",
        notes="Non-parametric monotone mapping via quantiles.",
    ),
    _sklearn_entry(
        "PowerTransformer",
        "sklearn.preprocessing.PowerTransformer",
        TransformCategory.SCALING,
        fitted_params=("lambdas_",),
        is_monotone=True,
        is_invertible=True,
        is_pointwise=True,
        capacity_model="power_transform",
        notes="Box-Cox or Yeo-Johnson power transform.",
    ),
    # ------ Imputation ------
    _sklearn_entry(
        "SimpleImputer",
        "sklearn.impute.SimpleImputer",
        TransformCategory.IMPUTATION,
        fitted_params=("statistics_",),
        is_pointwise=True,
        capacity_model="imputation_mean",
        notes="Replaces missing values with column mean/median/mode/constant.",
    ),
    _sklearn_entry(
        "KNNImputer",
        "sklearn.impute.KNNImputer",
        TransformCategory.IMPUTATION,
        fitted_params=("_fit_X",),
        statistic_type="non-sufficient",
        data_dependency="all-rows",
        is_pointwise=False,
        capacity_model="knn_imputation",
        notes="Uses k-nearest-neighbours; stores full training data.",
    ),
    _sklearn_entry(
        "IterativeImputer",
        "sklearn.impute.IterativeImputer",
        TransformCategory.IMPUTATION,
        fitted_params=("initial_imputer_", "imputation_sequence_"),
        statistic_type="non-sufficient",
        data_dependency="all-rows",
        capacity_model="iterative_imputation",
        notes="MICE-style iterative imputation. High leakage potential.",
    ),
    _sklearn_entry(
        "MissingIndicator",
        "sklearn.impute.MissingIndicator",
        TransformCategory.IMPUTATION,
        fitted_params=("features_",),
        is_pointwise=True,
        is_invertible=False,
        capacity_model="indicator",
        min_info_loss_fn="H(X) - n_features",
        notes="Binary indicator of missingness. Highly lossy.",
    ),
    # ------ Encoding ------
    _sklearn_entry(
        "OrdinalEncoder",
        "sklearn.preprocessing.OrdinalEncoder",
        TransformCategory.ENCODING,
        fitted_params=("categories_",),
        is_invertible=True,
        is_pointwise=True,
        capacity_model="ordinal_map",
        notes="Deterministic integer mapping per category.",
    ),
    _sklearn_entry(
        "OneHotEncoder",
        "sklearn.preprocessing.OneHotEncoder",
        TransformCategory.ENCODING,
        fitted_params=("categories_", "drop_idx_"),
        is_invertible=True,
        is_pointwise=True,
        max_info_gain_fn="0",
        capacity_model="one_hot",
        notes="Lossless one-hot expansion.",
    ),
    _sklearn_entry(
        "LabelEncoder",
        "sklearn.preprocessing.LabelEncoder",
        TransformCategory.ENCODING,
        fitted_params=("classes_",),
        is_invertible=True,
        is_pointwise=True,
        capacity_model="ordinal_map",
        notes="Target-label integer mapping.",
    ),
    _sklearn_entry(
        "TargetEncoder",
        "sklearn.preprocessing.TargetEncoder",
        TransformCategory.ENCODING,
        fitted_params=("encodings_", "classes_"),
        fit_uses_target=True,
        is_invertible=False,
        capacity_model="target_encoding",
        max_info_gain_fn="log2(n_categories)",
        notes="Encodes categories using target mean. Supervised – high leak risk.",
    ),
    _sklearn_entry(
        "LabelBinarizer",
        "sklearn.preprocessing.LabelBinarizer",
        TransformCategory.ENCODING,
        fitted_params=("classes_", "y_type_"),
        is_invertible=True,
        is_pointwise=True,
        capacity_model="one_hot",
        notes="Binarises labels (1-of-K). Lossless.",
    ),
    _sklearn_entry(
        "KBinsDiscretizer",
        "sklearn.preprocessing.KBinsDiscretizer",
        TransformCategory.ENCODING,
        fitted_params=("bin_edges_", "n_bins_"),
        is_invertible=False,
        is_pointwise=True,
        capacity_model="binning",
        min_info_loss_fn="H(X) - log2(n_bins)",
        notes="Discretises continuous features into bins.",
    ),
    # ------ Feature selection ------
    _sklearn_entry(
        "SelectKBest",
        "sklearn.feature_selection.SelectKBest",
        TransformCategory.FEATURE_SELECTION,
        fitted_params=("scores_", "pvalues_"),
        fit_uses_target=True,
        is_invertible=False,
        capacity_model="feature_mask",
        min_info_loss_fn="n_features - k",
        notes="Univariate feature selection using score function + y.",
    ),
    _sklearn_entry(
        "VarianceThreshold",
        "sklearn.feature_selection.VarianceThreshold",
        TransformCategory.FEATURE_SELECTION,
        fitted_params=("variances_",),
        is_invertible=False,
        capacity_model="feature_mask",
        min_info_loss_fn="n_removed_features",
        notes="Removes low-variance features. Unsupervised.",
    ),
    _sklearn_entry(
        "SelectFromModel",
        "sklearn.feature_selection.SelectFromModel",
        TransformCategory.FEATURE_SELECTION,
        fitted_params=("estimator_", "threshold_"),
        fit_uses_target=True,
        is_invertible=False,
        capacity_model="model_selection",
        min_info_loss_fn="n_features - n_selected",
        notes="Wraps an estimator with feature_importances_. Supervised.",
    ),
    _sklearn_entry(
        "RFE",
        "sklearn.feature_selection.RFE",
        TransformCategory.FEATURE_SELECTION,
        fitted_params=("support_", "ranking_", "estimator_"),
        fit_uses_target=True,
        is_invertible=False,
        capacity_model="recursive_elimination",
        min_info_loss_fn="n_features - n_features_to_select",
        notes="Recursive feature elimination. Supervised, iterative.",
    ),
    # ------ Dimensionality reduction ------
    _sklearn_entry(
        "PCA",
        "sklearn.decomposition.PCA",
        TransformCategory.DIMENSIONALITY_REDUCTION,
        fitted_params=("components_", "explained_variance_", "mean_", "n_components_"),
        is_invertible=False,
        capacity_model="pca_projection",
        min_info_loss_fn="sum(explained_variance[k:])",
        notes="Linear projection onto principal components.",
    ),
    _sklearn_entry(
        "TruncatedSVD",
        "sklearn.decomposition.TruncatedSVD",
        TransformCategory.DIMENSIONALITY_REDUCTION,
        fitted_params=("components_", "explained_variance_"),
        is_invertible=False,
        capacity_model="svd_projection",
        min_info_loss_fn="sum(singular_values[k:])",
        notes="SVD without centring; suited for sparse data.",
    ),
    _sklearn_entry(
        "FastICA",
        "sklearn.decomposition.FastICA",
        TransformCategory.DIMENSIONALITY_REDUCTION,
        fitted_params=("components_", "mixing_", "mean_"),
        is_invertible=True,
        preserves_independence=False,
        capacity_model="ica_unmixing",
        notes="Independent component analysis. Invertible linear unmixing.",
    ),
    _sklearn_entry(
        "NMF",
        "sklearn.decomposition.NMF",
        TransformCategory.DIMENSIONALITY_REDUCTION,
        fitted_params=("components_", "n_components_", "reconstruction_err_"),
        is_invertible=False,
        capacity_model="nmf_factorisation",
        min_info_loss_fn="reconstruction_err",
        notes="Non-negative matrix factorisation.",
    ),
    _sklearn_entry(
        "LatentDirichletAllocation",
        "sklearn.decomposition.LatentDirichletAllocation",
        TransformCategory.DIMENSIONALITY_REDUCTION,
        fitted_params=("components_", "exp_dirichlet_component_"),
        is_invertible=False,
        capacity_model="lda_topic",
        min_info_loss_fn="perplexity_bound",
        notes="Probabilistic topic model. Non-sufficient statistics.",
        statistic_type="non-sufficient",
    ),
    _sklearn_entry(
        "FactorAnalysis",
        "sklearn.decomposition.FactorAnalysis",
        TransformCategory.DIMENSIONALITY_REDUCTION,
        fitted_params=("components_", "noise_variance_", "mean_"),
        is_invertible=False,
        capacity_model="factor_analysis",
        notes="Gaussian latent factor model.",
    ),
    _sklearn_entry(
        "IncrementalPCA",
        "sklearn.decomposition.IncrementalPCA",
        TransformCategory.DIMENSIONALITY_REDUCTION,
        fitted_params=("components_", "explained_variance_", "mean_", "var_"),
        is_invertible=False,
        capacity_model="pca_projection",
        min_info_loss_fn="sum(explained_variance[k:])",
        notes="Mini-batch PCA. Same capacity model as PCA.",
    ),
    # ------ Polynomial / interaction features ------
    _sklearn_entry(
        "PolynomialFeatures",
        "sklearn.preprocessing.PolynomialFeatures",
        TransformCategory.ENCODING,
        is_fitted=False,
        fitted_params=(),
        data_dependency="row-independent",
        is_invertible=False,
        is_pointwise=True,
        requires_all_data=False,
        max_info_gain_fn="0",
        capacity_model="polynomial_expansion",
        notes="Generates polynomial and interaction features. Stateless.",
    ),
    # ------ Function transformer ------
    _sklearn_entry(
        "FunctionTransformer",
        "sklearn.preprocessing.FunctionTransformer",
        TransformCategory.CUSTOM,
        is_fitted=False,
        fitted_params=(),
        data_dependency="row-independent",
        requires_all_data=False,
        capacity_model="custom_function",
        notes="User-supplied function. Properties unknown – conservative model.",
    ),
    # ------ Pipeline / composition ------
    _sklearn_entry(
        "Pipeline",
        "sklearn.pipeline.Pipeline",
        TransformCategory.COMPOSITION,
        fitted_params=(),
        is_fitted=True,
        capacity_model="sequential_composition",
        notes="Sequential pipeline. Capacity = DPI chain of steps.",
    ),
    _sklearn_entry(
        "ColumnTransformer",
        "sklearn.compose.ColumnTransformer",
        TransformCategory.COMPOSITION,
        fitted_params=("transformers_",),
        capacity_model="parallel_composition",
        notes="Parallel per-column transforms; capacity = max over branches.",
    ),
    _sklearn_entry(
        "FeatureUnion",
        "sklearn.pipeline.FeatureUnion",
        TransformCategory.COMPOSITION,
        fitted_params=("transformer_list",),
        capacity_model="parallel_composition",
        notes="Concatenates outputs of multiple transformers.",
    ),
)


# ===================================================================
# Pre-populated entries – pandas
# ===================================================================

def _pandas_entry(
    name: str,
    method_name: str,
    category: TransformCategory,
    *,
    input_arity: int = 1,
    output_arity: int = 1,
    is_fitted: bool = False,
    fitted_params: Tuple[str, ...] = (),
    statistic_type: str = "sufficient",
    data_dependency: str = "row-independent",
    is_monotone: bool = True,
    is_invertible: bool = False,
    preserves_independence: bool = True,
    max_info_gain_fn: str = "0",
    min_info_loss_fn: str = "0",
    is_pointwise: bool = False,
    requires_all_data: bool = False,
    fit_uses_target: bool = False,
    capacity_model: str = "identity",
    notes: str = "",
) -> TransformRegistryEntry:
    sig = TransformSignature(
        name=name,
        category=category,
        input_arity=input_arity,
        output_arity=output_arity,
        is_fitted=is_fitted,
        fitted_params=fitted_params,
        statistic_type=statistic_type,
        data_dependency=data_dependency,
    )
    props = TransformProperty(
        is_monotone=is_monotone,
        is_invertible=is_invertible,
        preserves_independence=preserves_independence,
        max_info_gain_fn=max_info_gain_fn,
        min_info_loss_fn=min_info_loss_fn,
        is_pointwise=is_pointwise,
        requires_all_data=requires_all_data,
        fit_uses_target=fit_uses_target,
    )
    return TransformRegistryEntry(
        signature=sig,
        properties=props,
        pandas_method_name=method_name,
        capacity_model=capacity_model,
        notes=notes,
    )


_PANDAS_ENTRIES: Tuple[TransformRegistryEntry, ...] = (
    _pandas_entry(
        "pandas_fillna",
        "DataFrame.fillna",
        TransformCategory.IMPUTATION,
        is_pointwise=True,
        capacity_model="constant_fill",
        notes="Fills NaN. If value is scalar → pointwise; if method='ffill' → row-dependent.",
    ),
    _pandas_entry(
        "pandas_dropna",
        "DataFrame.dropna",
        TransformCategory.FILTERING,
        is_invertible=False,
        min_info_loss_fn="n_dropped_rows * n_features",
        capacity_model="row_filter",
        notes="Drops rows/columns with NaN. Information loss proportional to dropped rows.",
    ),
    _pandas_entry(
        "pandas_merge",
        "DataFrame.merge",
        TransformCategory.AGGREGATION,
        input_arity=2,
        data_dependency="all-rows",
        requires_all_data=True,
        capacity_model="relational_join",
        notes="SQL-style join. Can multiply or reduce rows. Major leak vector.",
    ),
    _pandas_entry(
        "pandas_concat",
        "pandas.concat",
        TransformCategory.AGGREGATION,
        input_arity=2,
        data_dependency="row-independent",
        is_invertible=True,
        capacity_model="concatenation",
        notes="Vertical/horizontal concatenation. Lossless.",
    ),
    _pandas_entry(
        "pandas_groupby_transform",
        "DataFrameGroupBy.transform",
        TransformCategory.AGGREGATION,
        data_dependency="all-rows",
        requires_all_data=True,
        capacity_model="group_broadcast",
        max_info_gain_fn="log2(n_groups)",
        notes="Broadcasts group-level statistic back to each row. Classic leak vector.",
    ),
    _pandas_entry(
        "pandas_groupby_agg",
        "DataFrameGroupBy.agg",
        TransformCategory.AGGREGATION,
        data_dependency="all-rows",
        requires_all_data=True,
        is_invertible=False,
        capacity_model="group_aggregation",
        min_info_loss_fn="n_rows - n_groups",
        notes="Reduces to one row per group. Lossy.",
    ),
    _pandas_entry(
        "pandas_apply",
        "DataFrame.apply",
        TransformCategory.CUSTOM,
        data_dependency="row-dependent",
        capacity_model="custom_function",
        notes="Arbitrary user function. Conservative model.",
    ),
    _pandas_entry(
        "pandas_map",
        "Series.map",
        TransformCategory.CUSTOM,
        data_dependency="row-independent",
        is_pointwise=True,
        capacity_model="pointwise_map",
        notes="Element-wise mapping via dict or function.",
    ),
    _pandas_entry(
        "pandas_rolling_mean",
        "Rolling.mean",
        TransformCategory.AGGREGATION,
        data_dependency="row-dependent",
        requires_all_data=False,
        capacity_model="windowed_aggregation",
        max_info_gain_fn="log2(window_size)",
        notes="Windowed mean. Each output depends on a window of rows.",
    ),
    _pandas_entry(
        "pandas_rolling_std",
        "Rolling.std",
        TransformCategory.AGGREGATION,
        data_dependency="row-dependent",
        requires_all_data=False,
        capacity_model="windowed_aggregation",
        max_info_gain_fn="log2(window_size)",
        notes="Windowed standard deviation.",
    ),
    _pandas_entry(
        "pandas_rolling_sum",
        "Rolling.sum",
        TransformCategory.AGGREGATION,
        data_dependency="row-dependent",
        requires_all_data=False,
        capacity_model="windowed_aggregation",
        max_info_gain_fn="log2(window_size)",
        notes="Windowed sum.",
    ),
    _pandas_entry(
        "pandas_expanding_mean",
        "Expanding.mean",
        TransformCategory.AGGREGATION,
        data_dependency="all-rows",
        requires_all_data=True,
        capacity_model="expanding_aggregation",
        notes="Expanding-window mean. Depends on all preceding rows.",
    ),
    _pandas_entry(
        "pandas_ewm_mean",
        "ExponentialMovingWindow.mean",
        TransformCategory.AGGREGATION,
        data_dependency="all-rows",
        requires_all_data=True,
        capacity_model="ewm_aggregation",
        notes="Exponentially-weighted moving mean.",
    ),
    _pandas_entry(
        "pandas_pivot_table",
        "DataFrame.pivot_table",
        TransformCategory.AGGREGATION,
        data_dependency="all-rows",
        requires_all_data=True,
        is_invertible=False,
        capacity_model="pivot_aggregation",
        notes="Pivot with aggregation function. Lossy reshape.",
    ),
    _pandas_entry(
        "pandas_get_dummies",
        "pandas.get_dummies",
        TransformCategory.ENCODING,
        data_dependency="all-rows",
        is_invertible=True,
        capacity_model="one_hot",
        requires_all_data=True,
        notes="One-hot encoding. Lossless but column set depends on data.",
    ),
    _pandas_entry(
        "pandas_cut",
        "pandas.cut",
        TransformCategory.ENCODING,
        data_dependency="all-rows",
        is_invertible=False,
        capacity_model="binning",
        min_info_loss_fn="H(X) - log2(n_bins)",
        notes="Bin continuous values into discrete intervals.",
    ),
    _pandas_entry(
        "pandas_qcut",
        "pandas.qcut",
        TransformCategory.ENCODING,
        data_dependency="all-rows",
        requires_all_data=True,
        is_invertible=False,
        capacity_model="quantile_binning",
        min_info_loss_fn="H(X) - log2(n_quantiles)",
        notes="Quantile-based binning. Bin edges depend on full data.",
    ),
    _pandas_entry(
        "pandas_replace",
        "DataFrame.replace",
        TransformCategory.IMPUTATION,
        data_dependency="row-independent",
        is_pointwise=True,
        capacity_model="pointwise_map",
        notes="Value replacement. Pointwise when mapping is explicit.",
    ),
    _pandas_entry(
        "pandas_astype",
        "DataFrame.astype",
        TransformCategory.ENCODING,
        data_dependency="row-independent",
        is_pointwise=True,
        capacity_model="type_cast",
        notes="Type casting. May be lossy (float→int) or lossless.",
    ),
    _pandas_entry(
        "pandas_sort_values",
        "DataFrame.sort_values",
        TransformCategory.IDENTITY,
        data_dependency="all-rows",
        is_invertible=True,
        requires_all_data=True,
        capacity_model="permutation",
        notes="Row permutation. Lossless but order depends on all data.",
    ),
    _pandas_entry(
        "pandas_drop_duplicates",
        "DataFrame.drop_duplicates",
        TransformCategory.FILTERING,
        data_dependency="all-rows",
        is_invertible=False,
        requires_all_data=True,
        capacity_model="row_filter",
        min_info_loss_fn="n_duplicates * n_features",
        notes="Removes duplicate rows. Information loss = dropped rows.",
    ),
    _pandas_entry(
        "pandas_sample",
        "DataFrame.sample",
        TransformCategory.FILTERING,
        data_dependency="all-rows",
        is_invertible=False,
        requires_all_data=True,
        capacity_model="row_filter",
        notes="Random subsampling. Non-deterministic row removal.",
    ),
    _pandas_entry(
        "pandas_interpolate",
        "DataFrame.interpolate",
        TransformCategory.IMPUTATION,
        data_dependency="row-dependent",
        is_invertible=False,
        capacity_model="interpolation",
        notes="Fills NaN by interpolation between neighbouring values.",
    ),
    _pandas_entry(
        "pandas_clip",
        "DataFrame.clip",
        TransformCategory.SCALING,
        data_dependency="row-independent",
        is_pointwise=True,
        is_invertible=False,
        capacity_model="clipping",
        min_info_loss_fn="H(X_clipped | X)",
        notes="Clips values to [lower, upper]. Lossy at boundaries.",
    ),
    _pandas_entry(
        "pandas_rank",
        "DataFrame.rank",
        TransformCategory.ENCODING,
        data_dependency="all-rows",
        requires_all_data=True,
        is_invertible=False,
        capacity_model="rank_transform",
        notes="Replaces values with ranks. Depends on all rows.",
    ),
    _pandas_entry(
        "pandas_diff",
        "DataFrame.diff",
        TransformCategory.AGGREGATION,
        data_dependency="row-dependent",
        is_invertible=True,
        capacity_model="difference_operator",
        notes="First difference. Invertible given initial value.",
    ),
    _pandas_entry(
        "pandas_pct_change",
        "DataFrame.pct_change",
        TransformCategory.AGGREGATION,
        data_dependency="row-dependent",
        is_invertible=False,
        capacity_model="ratio_operator",
        notes="Percentage change between consecutive rows.",
    ),
    _pandas_entry(
        "pandas_cumsum",
        "DataFrame.cumsum",
        TransformCategory.AGGREGATION,
        data_dependency="row-dependent",
        is_invertible=True,
        capacity_model="cumulative_sum",
        notes="Cumulative sum. Invertible via diff.",
    ),
    _pandas_entry(
        "pandas_shift",
        "DataFrame.shift",
        TransformCategory.AGGREGATION,
        data_dependency="row-dependent",
        is_invertible=True,
        capacity_model="lag_operator",
        notes="Shifts rows by n periods. Lossless modulo boundary NaN.",
    ),
)


# ===================================================================
# Module-level default registry
# ===================================================================


def _build_default_registry() -> TransformRegistry:
    """Construct and return the default pre-populated registry."""
    reg = TransformRegistry()
    reg.register_many(_SKLEARN_ENTRIES)
    reg.register_many(_PANDAS_ENTRIES)
    return reg


def default_registry() -> TransformRegistry:
    """Return a *copy* of the default registry so callers can extend it
    without affecting the canonical set.
    """
    return _DEFAULT_REGISTRY.copy()


_DEFAULT_REGISTRY: TransformRegistry = _build_default_registry()
