"""Component registry for plug-in architecture.

A simple registry that maps (category, name) pairs to component
classes, enabling a plug-in style extension mechanism.

Classes
-------
ComponentRegistry
    Central registry with category-based namespacing.
PluginManager
    Auto-discovery and dynamic loading of plugin modules.

Decorator helpers
-----------------
register_ci_test, register_score, register_learner, register_baseline
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import pkgutil
from typing import Any, Callable, Dict, List, Optional, Type


# ---------------------------------------------------------------------------
# Expected interfaces for validation
# ---------------------------------------------------------------------------

_REQUIRED_METHODS: Dict[str, List[str]] = {
    "ci_test": ["test"],
    "score": ["local_score"],
    "learner": ["fit"],
    "baseline": ["run"],
}


# ---------------------------------------------------------------------------
# Component Registry
# ---------------------------------------------------------------------------

class ComponentRegistry:
    """Registry mapping named components to their implementing classes.

    Components are organised into categories so that different
    subsystems (scorers, aligners, detectors, …) can each maintain
    an independent namespace.
    """

    _registries: Dict[str, "ComponentRegistry"] = {}

    def __init__(self) -> None:
        self._registry: Dict[str, Dict[str, Any]] = {}

    # -- Core API --

    def register(
        self,
        name: str,
        component: Any,
        category: str = "default",
    ) -> None:
        """Register *component* under *name* in *category*."""
        self._validate_component(component, category)
        self._registry.setdefault(category, {})[name] = component

    def get(self, name: str, category: str = "default") -> Any:
        """Retrieve the component registered as *name* in *category*.

        Raises KeyError if not found.
        """
        cat = self._registry.get(category)
        if cat is None or name not in cat:
            raise KeyError(
                f"Component '{name}' not found in category '{category}'. "
                f"Available: {list(cat.keys()) if cat else []}"
            )
        return cat[name]

    def list_components(self, category: Optional[str] = None) -> List[str]:
        """List registered component names, optionally filtered by *category*."""
        if category is not None:
            return list(self._registry.get(category, {}).keys())
        names: List[str] = []
        for cat in self._registry.values():
            names.extend(cat.keys())
        return names

    def has(self, name: str, category: str = "default") -> bool:
        """Return ``True`` if *name* is registered in *category*."""
        return name in self._registry.get(category, {})

    def unregister(self, name: str, category: str = "default") -> None:
        """Remove *name* from *category*."""
        cat = self._registry.get(category)
        if cat is not None and name in cat:
            del cat[name]

    def categories(self) -> List[str]:
        """Return all registered category names."""
        return list(self._registry.keys())

    # -- Validation --

    @staticmethod
    def _validate_component(component: Any, category: str) -> None:
        """Validate that *component* has the required interface for *category*.

        Only enforced for categories listed in _REQUIRED_METHODS.
        """
        required = _REQUIRED_METHODS.get(category)
        if required is None:
            return
        for method_name in required:
            if not (hasattr(component, method_name) and callable(getattr(component, method_name, None))):
                # Also accept if it's a class with the method
                if isinstance(component, type):
                    if not hasattr(component, method_name):
                        raise TypeError(
                            f"Component for category '{category}' must have a "
                            f"'{method_name}' method."
                        )
                # For non-class objects we just warn via TypeError
                elif not hasattr(component, method_name):
                    raise TypeError(
                        f"Component for category '{category}' must have a "
                        f"'{method_name}' method."
                    )


# Module-level singleton instance
registry = ComponentRegistry()


# ---------------------------------------------------------------------------
# Decorator helpers
# ---------------------------------------------------------------------------

def register_ci_test(name: str) -> Callable[[Type], Type]:
    """Decorator to register a CI test class."""
    def decorator(cls: Type) -> Type:
        registry.register(name, cls, category="ci_test")
        return cls
    return decorator


def register_score(name: str) -> Callable[[Type], Type]:
    """Decorator to register a scoring function / class."""
    def decorator(cls: Type) -> Type:
        registry.register(name, cls, category="score")
        return cls
    return decorator


def register_learner(name: str) -> Callable[[Type], Type]:
    """Decorator to register a structure learner."""
    def decorator(cls: Type) -> Type:
        registry.register(name, cls, category="learner")
        return cls
    return decorator


def register_baseline(name: str) -> Callable[[Type], Type]:
    """Decorator to register a baseline algorithm."""
    def decorator(cls: Type) -> Type:
        registry.register(name, cls, category="baseline")
        return cls
    return decorator


# ---------------------------------------------------------------------------
# Plugin Manager
# ---------------------------------------------------------------------------

class PluginManager:
    """Auto-discover and load plugin modules.

    Parameters
    ----------
    registry : ComponentRegistry
        The registry into which discovered plugins will register themselves.
    """

    def __init__(self, plugin_registry: Optional[ComponentRegistry] = None) -> None:
        self.registry = plugin_registry or registry
        self._loaded_modules: List[str] = []

    def discover_plugins(self, package_path: str) -> List[str]:
        """Auto-discover and import all Python modules under *package_path*.

        Returns list of imported module names.
        """
        loaded: List[str] = []
        if os.path.isdir(package_path):
            for finder, name, ispkg in pkgutil.iter_modules([package_path]):
                try:
                    mod = self._import_module(os.path.join(package_path, name + ".py"))
                    if mod is not None:
                        loaded.append(name)
                except Exception:
                    pass
        self._loaded_modules.extend(loaded)
        return loaded

    def load_plugin(self, module_path: str) -> Optional[str]:
        """Load a single plugin module from *module_path*.

        Returns the module name on success, None on failure.
        """
        mod = self._import_module(module_path)
        if mod is not None:
            name = getattr(mod, "__name__", module_path)
            self._loaded_modules.append(name)
            return name
        return None

    @staticmethod
    def _import_module(path: str) -> Any:
        """Dynamically import a module from a file path."""
        if not os.path.isfile(path):
            return None
        module_name = os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return mod
