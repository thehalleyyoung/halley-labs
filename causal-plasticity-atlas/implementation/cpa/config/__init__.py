"""CPA configuration subpackage.

Extended configuration management including experiment setup,
hyperparameter spaces, and component registry for plug-in architecture.

Modules
-------
experiment
    Experiment configuration and management.
hyperparameters
    Hyperparameter spaces and tuning.
registry
    Component registry for plug-in architecture.
"""

from cpa.config.experiment import ExperimentConfig, ExperimentManager, ExperimentRunner, ResultsTracker
from cpa.config.hyperparameters import (
    HyperparameterSpace,
    HyperparameterConfig,
    ParameterRange,
    GridSearch,
    RandomSearch,
    BayesianOptimizer,
)
from cpa.config.registry import (
    ComponentRegistry,
    PluginManager,
    registry,
    register_ci_test,
    register_score,
    register_learner,
    register_baseline,
)

__all__ = [
    # experiment.py
    "ExperimentConfig",
    "ExperimentManager",
    "ExperimentRunner",
    "ResultsTracker",
    # hyperparameters.py
    "HyperparameterSpace",
    "HyperparameterConfig",
    "ParameterRange",
    "GridSearch",
    "RandomSearch",
    "BayesianOptimizer",
    # registry.py
    "ComponentRegistry",
    "PluginManager",
    "registry",
    "register_ci_test",
    "register_score",
    "register_learner",
    "register_baseline",
]
