"""Specification registry for the CoaCert-TLA spec library.

Provides a central catalogue of built-in TLA-lite specifications.
Specs can be looked up by name, iterated over, and instantiated with
specific parameter configurations.  Benchmark presets (small / medium /
large) are provided for each specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Type

from ..parser.ast_nodes import Module, Property


# ---------------------------------------------------------------------------
# Spec protocol — any spec class must implement this
# ---------------------------------------------------------------------------

class SpecProtocol(Protocol):
    """Structural interface every spec builder must satisfy."""

    def get_spec(self) -> Module: ...
    def get_properties(self) -> List[Property]: ...
    def get_config(self, **kwargs: Any) -> Dict[str, Any]: ...
    def validate(self) -> List[str]: ...

    @staticmethod
    def supported_configurations() -> List[Dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Spec metadata
# ---------------------------------------------------------------------------

@dataclass
class SpecMetadata:
    """Descriptive metadata attached to a registered specification."""

    name: str
    description: str
    module_path: str  # e.g. "coacert.specs.two_phase_commit"
    spec_class_name: str
    parameter_names: List[str] = field(default_factory=list)
    parameter_ranges: Dict[str, List[Any]] = field(default_factory=dict)
    expected_state_space: Dict[str, Dict[str, int]] = field(
        default_factory=dict
    )
    tags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Benchmark preset
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkPreset:
    """A named configuration preset for a specification."""

    name: str  # "small", "medium", "large"
    spec_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    timeout_seconds: int = 300


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class SpecRegistry:
    """Central registry of TLA-lite specification builders.

    Usage::

        registry = SpecRegistry.default()
        spec_cls = registry.get("TwoPhaseCommit")
        module = spec_cls(n_participants=5).get_spec()
    """

    def __init__(self) -> None:
        self._specs: Dict[str, _RegistryEntry] = {}

    # -- Registration --------------------------------------------------

    def register(self, name: str, spec_class: type, *,
                 metadata: Optional[SpecMetadata] = None,
                 presets: Optional[List[BenchmarkPreset]] = None) -> None:
        """Register a specification builder class under *name*."""
        if name in self._specs:
            raise ValueError(f"Spec '{name}' already registered")
        self._specs[name] = _RegistryEntry(
            name=name,
            spec_class=spec_class,
            metadata=metadata or SpecMetadata(
                name=name,
                description="",
                module_path="",
                spec_class_name=spec_class.__name__,
            ),
            presets=presets or [],
        )

    # -- Lookup --------------------------------------------------------

    def get(self, name: str) -> type:
        """Return the spec class registered under *name*."""
        entry = self._specs.get(name)
        if entry is None:
            raise KeyError(
                f"Unknown spec '{name}'. Available: {self.list_names()}"
            )
        return entry.spec_class

    def get_metadata(self, name: str) -> SpecMetadata:
        """Return the metadata for the spec registered under *name*."""
        return self._specs[name].metadata

    def get_presets(self, name: str) -> List[BenchmarkPreset]:
        """Return benchmark presets for *name*."""
        return list(self._specs[name].presets)

    def get_preset(self, spec_name: str,
                   preset_name: str) -> BenchmarkPreset:
        """Return a specific preset by spec and preset names."""
        for p in self.get_presets(spec_name):
            if p.name == preset_name:
                return p
        raise KeyError(
            f"No preset '{preset_name}' for spec '{spec_name}'"
        )

    # -- Enumeration ---------------------------------------------------

    def list_names(self) -> List[str]:
        """Return sorted list of all registered spec names."""
        return sorted(self._specs.keys())

    def list_all(self) -> List[SpecMetadata]:
        """Return metadata for every registered spec."""
        return [e.metadata for e in self._specs.values()]

    def list_by_tag(self, tag: str) -> List[SpecMetadata]:
        """Return specs whose tags include *tag*."""
        return [e.metadata for e in self._specs.values()
                if tag in e.metadata.tags]

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    # -- Instantiation helpers -----------------------------------------

    def instantiate(self, name: str,
                    **kwargs: Any) -> Any:
        """Instantiate a spec builder with the given parameters."""
        cls = self.get(name)
        return cls(**kwargs)

    def instantiate_preset(self, spec_name: str,
                           preset_name: str) -> Any:
        """Instantiate a spec builder from a named preset."""
        preset = self.get_preset(spec_name, preset_name)
        cls = self.get(spec_name)
        return cls(**preset.parameters)

    # -- Validation ----------------------------------------------------

    def validate_spec(self, name: str, **kwargs: Any) -> List[str]:
        """Build a spec with given params and validate it."""
        instance = self.instantiate(name, **kwargs)
        return instance.validate()

    def validate_all(self) -> Dict[str, List[str]]:
        """Validate every spec using its small preset (if available)."""
        results: Dict[str, List[str]] = {}
        for name in self.list_names():
            presets = self.get_presets(name)
            small = [p for p in presets if p.name == "small"]
            if small:
                params = small[0].parameters
            else:
                params = {}
            try:
                errors = self.validate_spec(name, **params)
                results[name] = errors
            except Exception as exc:
                results[name] = [f"Exception: {exc}"]
        return results

    # -- Factory -------------------------------------------------------

    @classmethod
    def default(cls) -> "SpecRegistry":
        """Create a registry pre-populated with all built-in specs."""
        registry = cls()
        _register_builtins(registry)
        return registry


# ---------------------------------------------------------------------------
# Internal entry type
# ---------------------------------------------------------------------------

@dataclass
class _RegistryEntry:
    name: str
    spec_class: type
    metadata: SpecMetadata
    presets: List[BenchmarkPreset] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Built-in registration
# ---------------------------------------------------------------------------

def _register_builtins(registry: SpecRegistry) -> None:
    """Register all built-in spec classes."""
    from .two_phase_commit import TwoPhaseCommitSpec
    from .leader_election import LeaderElectionSpec
    from .peterson import PetersonSpec
    from .paxos import PaxosSpec
    from .asymmetric_token_ring import AsymmetricTokenRingSpec

    # -- Two-Phase Commit ---------------------------------------------
    registry.register(
        "TwoPhaseCommit",
        TwoPhaseCommitSpec,
        metadata=SpecMetadata(
            name="TwoPhaseCommit",
            description=(
                "Two-Phase Commit protocol with N resource managers. "
                "Models the classic TM/RM coordination for atomic commits."
            ),
            module_path="coacert.specs.two_phase_commit",
            spec_class_name="TwoPhaseCommitSpec",
            parameter_names=["n_participants"],
            parameter_ranges={"n_participants": [3, 5, 7]},
            tags=["distributed", "consensus", "classic"],
        ),
        presets=[
            BenchmarkPreset("small", "TwoPhaseCommit",
                            {"n_participants": 3},
                            "3-RM two-phase commit", 30),
            BenchmarkPreset("medium", "TwoPhaseCommit",
                            {"n_participants": 5},
                            "5-RM two-phase commit", 120),
            BenchmarkPreset("large", "TwoPhaseCommit",
                            {"n_participants": 7},
                            "7-RM two-phase commit", 600),
        ],
    )

    # -- Leader Election ----------------------------------------------
    registry.register(
        "LeaderElection",
        LeaderElectionSpec,
        metadata=SpecMetadata(
            name="LeaderElection",
            description=(
                "Ring-based leader election (simplified Chang-Roberts). "
                "Each node forwards IDs around a unidirectional ring."
            ),
            module_path="coacert.specs.leader_election",
            spec_class_name="LeaderElectionSpec",
            parameter_names=["n_nodes"],
            parameter_ranges={"n_nodes": [3, 5]},
            tags=["distributed", "election", "ring"],
        ),
        presets=[
            BenchmarkPreset("small", "LeaderElection",
                            {"n_nodes": 3},
                            "3-node ring election", 30),
            BenchmarkPreset("medium", "LeaderElection",
                            {"n_nodes": 5},
                            "5-node ring election", 300),
        ],
    )

    # -- Peterson's Mutual Exclusion ----------------------------------
    registry.register(
        "Peterson",
        PetersonSpec,
        metadata=SpecMetadata(
            name="Peterson",
            description=(
                "Peterson's mutual exclusion. Classic 2-process version "
                "and generalised filter lock for N>=3."
            ),
            module_path="coacert.specs.peterson",
            spec_class_name="PetersonSpec",
            parameter_names=["n_processes"],
            parameter_ranges={"n_processes": [2, 3]},
            tags=["concurrency", "mutual-exclusion", "classic"],
        ),
        presets=[
            BenchmarkPreset("small", "Peterson",
                            {"n_processes": 2},
                            "2-process classic Peterson", 10),
            BenchmarkPreset("medium", "Peterson",
                            {"n_processes": 3},
                            "3-process filter lock", 120),
        ],
    )

    # -- Paxos --------------------------------------------------------
    registry.register(
        "Paxos",
        PaxosSpec,
        metadata=SpecMetadata(
            name="Paxos",
            description=(
                "Single-decree Paxos (Synod). Models Phase 1a/1b/2a/2b "
                "and Decide actions with quorum intersection."
            ),
            module_path="coacert.specs.paxos",
            spec_class_name="PaxosSpec",
            parameter_names=["n_acceptors", "n_values", "max_ballot"],
            parameter_ranges={
                "n_acceptors": [3, 5],
                "n_values": [2],
                "max_ballot": [2, 3],
            },
            tags=["distributed", "consensus", "paxos"],
        ),
        presets=[
            BenchmarkPreset("small", "Paxos",
                            {"n_acceptors": 3, "n_values": 2,
                             "max_ballot": 2},
                            "3-acceptor Paxos, 2 ballots", 60),
            BenchmarkPreset("medium", "Paxos",
                            {"n_acceptors": 3, "n_values": 2,
                             "max_ballot": 3},
                            "3-acceptor Paxos, 4 ballots", 300),
            BenchmarkPreset("large", "Paxos",
                            {"n_acceptors": 5, "n_values": 2,
                             "max_ballot": 3},
                            "5-acceptor Paxos", 1800),
        ],
    )

    # -- Asymmetric Token Ring ----------------------------------------
    registry.register(
        "AsymmetricTokenRing",
        AsymmetricTokenRingSpec,
        metadata=SpecMetadata(
            name="AsymmetricTokenRing",
            description=(
                "Token-passing ring with asymmetric roles. Node 0 is "
                "a distinguished initiator with priority requests. "
                "No participant symmetry."
            ),
            module_path="coacert.specs.asymmetric_token_ring",
            spec_class_name="AsymmetricTokenRingSpec",
            parameter_names=["n_nodes"],
            parameter_ranges={"n_nodes": [3, 4, 5, 6]},
            tags=["distributed", "asymmetric", "token-ring"],
        ),
        presets=[
            BenchmarkPreset("small", "AsymmetricTokenRing",
                            {"n_nodes": 3},
                            "3-node asymmetric token ring", 30),
            BenchmarkPreset("medium", "AsymmetricTokenRing",
                            {"n_nodes": 4},
                            "4-node asymmetric token ring", 60),
            BenchmarkPreset("large", "AsymmetricTokenRing",
                            {"n_nodes": 5},
                            "5-node asymmetric token ring", 300),
        ],
    )
