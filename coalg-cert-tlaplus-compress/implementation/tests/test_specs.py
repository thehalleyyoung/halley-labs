"""Comprehensive tests for coacert.specs – spec builders, registry, and utilities."""

import pytest

from coacert.parser.ast_nodes import (
    Module,
    VariableDecl,
    ConstantDecl,
    OperatorDef,
    InvariantProperty,
    SafetyProperty,
    LivenessProperty,
    TemporalProperty,
    Identifier,
    PrimedIdentifier,
    IntLiteral,
    BoolLiteral,
    OperatorApplication,
    SetEnumeration,
    QuantifiedExpr,
    IfThenElse,
    AlwaysExpr,
    EventuallyExpr,
)
from coacert.specs.two_phase_commit import TwoPhaseCommitSpec
from coacert.specs.leader_election import LeaderElectionSpec
from coacert.specs.peterson import PetersonSpec
from coacert.specs.paxos import PaxosSpec
from coacert.specs.spec_registry import (
    SpecRegistry,
    SpecMetadata,
    BenchmarkPreset,
)
from coacert.specs.spec_utils import (
    ModuleBuilder,
    ident,
    primed,
    int_lit,
    bool_lit,
    make_eq,
    make_neq,
    make_land,
    make_lor,
    make_lnot,
    make_conjunction,
    make_disjunction,
    make_set_enum,
    make_forall_single,
    make_exists_single,
    make_unchanged,
    make_always,
    make_eventually,
    make_invariant_property,
    make_safety_property,
    make_liveness_property,
    make_temporal_property,
    make_variable_decl,
    make_constant_decl,
    make_ite,
)


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_registry():
    return SpecRegistry.default()


# ---------------------------------------------------------------------------
#  Two-Phase Commit
# ---------------------------------------------------------------------------


class TestTwoPhaseCommitSpec:
    def test_construct_default(self):
        spec = TwoPhaseCommitSpec()
        assert spec is not None

    def test_construct_n3(self):
        spec = TwoPhaseCommitSpec(n_participants=3)
        module = spec.get_spec()
        assert isinstance(module, Module)

    def test_module_name(self):
        module = TwoPhaseCommitSpec(n_participants=3).get_spec()
        assert module.name == "TwoPhaseCommit"

    def test_variables_present(self):
        module = TwoPhaseCommitSpec(n_participants=3).get_spec()
        var_names = {n for v in module.variables for n in v.names}
        for expected in ("rmState", "tmState", "tmPrepared", "msgs"):
            assert expected in var_names, f"Missing variable {expected}"

    def test_definitions_nonempty(self):
        module = TwoPhaseCommitSpec(n_participants=3).get_spec()
        assert len(module.definitions) > 0

    def test_get_properties_nonempty(self):
        props = TwoPhaseCommitSpec(n_participants=3).get_properties()
        assert len(props) > 0

    def test_get_config_keys(self):
        cfg = TwoPhaseCommitSpec(n_participants=3).get_config()
        assert isinstance(cfg, dict)
        assert "n_participants" in cfg or len(cfg) > 0

    def test_validate_no_errors(self):
        errors = TwoPhaseCommitSpec(n_participants=3).validate()
        assert errors == []

    @pytest.mark.parametrize("n", [2, 5])
    def test_various_sizes(self, n):
        spec = TwoPhaseCommitSpec(n_participants=n)
        module = spec.get_spec()
        assert isinstance(module, Module)
        assert spec.validate() == []

    def test_supported_configurations(self):
        cfgs = TwoPhaseCommitSpec.supported_configurations()
        assert isinstance(cfgs, list)
        assert len(cfgs) > 0

    def test_estimate_states(self):
        est = TwoPhaseCommitSpec._estimate_states(3)
        assert isinstance(est, dict)
        assert "upper_bound" in est
        assert est["upper_bound"] > 0


# ---------------------------------------------------------------------------
#  Leader Election
# ---------------------------------------------------------------------------


class TestLeaderElectionSpec:
    def test_construct_default(self):
        spec = LeaderElectionSpec()
        assert spec is not None

    def test_module_name(self):
        module = LeaderElectionSpec(n_nodes=3).get_spec()
        assert module.name == "LeaderElection"

    def test_variables_present(self):
        module = LeaderElectionSpec(n_nodes=3).get_spec()
        var_names = {n for v in module.variables for n in v.names}
        for expected in ("leader", "inbox", "active", "id"):
            assert expected in var_names

    def test_get_properties_nonempty(self):
        props = LeaderElectionSpec(n_nodes=3).get_properties()
        assert len(props) > 0

    def test_validate_no_errors(self):
        assert LeaderElectionSpec(n_nodes=3).validate() == []

    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_various_sizes(self, n):
        spec = LeaderElectionSpec(n_nodes=n)
        module = spec.get_spec()
        assert isinstance(module, Module)
        assert spec.validate() == []

    def test_get_config(self):
        cfg = LeaderElectionSpec(n_nodes=4).get_config()
        assert isinstance(cfg, dict)

    def test_supported_configurations(self):
        cfgs = LeaderElectionSpec.supported_configurations()
        assert isinstance(cfgs, list)
        assert len(cfgs) > 0

    def test_estimate_states(self):
        est = LeaderElectionSpec._estimate_states(3)
        assert isinstance(est, dict)
        assert est["upper_bound"] > 0


# ---------------------------------------------------------------------------
#  Peterson
# ---------------------------------------------------------------------------


class TestPetersonSpec:
    def test_construct_classic(self):
        spec = PetersonSpec(n_processes=2)
        module = spec.get_spec()
        assert isinstance(module, Module)

    def test_classic_module_name(self):
        module = PetersonSpec(n_processes=2).get_spec()
        assert module.name == "Peterson"

    def test_classic_variables(self):
        module = PetersonSpec(n_processes=2).get_spec()
        var_names = {n for v in module.variables for n in v.names}
        for expected in ("pc", "flag", "turn"):
            assert expected in var_names

    def test_filter_lock_module_name(self):
        module = PetersonSpec(n_processes=3).get_spec()
        assert module.name == "PetersonFilterLock"

    def test_filter_lock_variables(self):
        module = PetersonSpec(n_processes=3).get_spec()
        var_names = {n for v in module.variables for n in v.names}
        for expected in ("pc", "level", "last_to_enter"):
            assert expected in var_names

    def test_get_properties_nonempty(self):
        props = PetersonSpec(n_processes=2).get_properties()
        assert len(props) > 0

    def test_validate_no_errors(self):
        assert PetersonSpec(n_processes=2).validate() == []

    def test_validate_n3(self):
        assert PetersonSpec(n_processes=3).validate() == []

    def test_get_config(self):
        cfg = PetersonSpec(n_processes=2).get_config()
        assert isinstance(cfg, dict)

    def test_supported_configurations(self):
        cfgs = PetersonSpec.supported_configurations()
        assert isinstance(cfgs, list)
        assert len(cfgs) > 0

    def test_estimate_states(self):
        est = PetersonSpec._estimate_states(2)
        assert isinstance(est, dict)
        assert est["upper_bound"] > 0


# ---------------------------------------------------------------------------
#  Paxos
# ---------------------------------------------------------------------------


class TestPaxosSpec:
    def test_construct_default(self):
        spec = PaxosSpec()
        assert spec is not None

    def test_module_name(self):
        module = PaxosSpec(n_acceptors=3, n_values=2, max_ballot=3).get_spec()
        assert module.name == "Paxos"

    def test_variables_present(self):
        module = PaxosSpec(n_acceptors=3, n_values=2, max_ballot=3).get_spec()
        var_names = {n for v in module.variables for n in v.names}
        for expected in ("maxBal", "maxVBal", "maxVal", "msgs", "decided"):
            assert expected in var_names

    def test_definitions_nonempty(self):
        module = PaxosSpec(n_acceptors=3, n_values=2, max_ballot=3).get_spec()
        assert len(module.definitions) > 0

    def test_get_properties_nonempty(self):
        props = PaxosSpec(n_acceptors=3, n_values=2, max_ballot=3).get_properties()
        assert len(props) > 0

    def test_validate_no_errors(self):
        assert PaxosSpec(n_acceptors=3, n_values=2, max_ballot=3).validate() == []

    def test_get_config(self):
        cfg = PaxosSpec(n_acceptors=3, n_values=2, max_ballot=3).get_config()
        assert isinstance(cfg, dict)

    def test_supported_configurations(self):
        cfgs = PaxosSpec.supported_configurations()
        assert isinstance(cfgs, list)
        assert len(cfgs) > 0

    def test_estimate_states(self):
        est = PaxosSpec._estimate_states(3)
        assert isinstance(est, dict)
        assert est["upper_bound"] > 0


# ---------------------------------------------------------------------------
#  Spec Registry
# ---------------------------------------------------------------------------

BUILTIN_SPEC_NAMES = ["TwoPhaseCommit", "LeaderElection", "Peterson", "Paxos"]


class TestSpecRegistry:
    def test_default_registry_nonempty(self, default_registry):
        assert len(default_registry) >= 4

    def test_default_has_all_builtins(self, default_registry):
        for name in BUILTIN_SPEC_NAMES:
            assert name in default_registry

    def test_list_names(self, default_registry):
        names = default_registry.list_names()
        assert isinstance(names, list)
        for name in BUILTIN_SPEC_NAMES:
            assert name in names

    def test_list_all_returns_metadata(self, default_registry):
        all_meta = default_registry.list_all()
        assert isinstance(all_meta, list)
        assert len(all_meta) >= 4
        for m in all_meta:
            assert isinstance(m, SpecMetadata)

    def test_get_returns_class(self, default_registry):
        cls = default_registry.get("TwoPhaseCommit")
        assert cls is TwoPhaseCommitSpec

    def test_get_metadata(self, default_registry):
        meta = default_registry.get_metadata("TwoPhaseCommit")
        assert isinstance(meta, SpecMetadata)
        assert meta.name == "TwoPhaseCommit"
        assert "n_participants" in meta.parameter_names

    def test_get_presets(self, default_registry):
        presets = default_registry.get_presets("TwoPhaseCommit")
        assert isinstance(presets, list)
        assert len(presets) > 0
        assert all(isinstance(p, BenchmarkPreset) for p in presets)

    def test_get_preset(self, default_registry):
        preset = default_registry.get_preset("TwoPhaseCommit", "small")
        assert isinstance(preset, BenchmarkPreset)
        assert preset.name == "small"

    def test_get_preset_not_found(self, default_registry):
        with pytest.raises(KeyError):
            default_registry.get_preset("TwoPhaseCommit", "nonexistent")

    def test_instantiate(self, default_registry):
        instance = default_registry.instantiate("TwoPhaseCommit", n_participants=3)
        module = instance.get_spec()
        assert isinstance(module, Module)

    def test_instantiate_preset(self, default_registry):
        instance = default_registry.instantiate_preset("TwoPhaseCommit", "small")
        module = instance.get_spec()
        assert isinstance(module, Module)

    def test_validate_spec(self, default_registry):
        errors = default_registry.validate_spec("TwoPhaseCommit", n_participants=3)
        assert errors == []

    def test_validate_all(self, default_registry):
        results = default_registry.validate_all()
        assert isinstance(results, dict)
        for name, errors in results.items():
            assert errors == [], f"Spec {name} has validation errors: {errors}"

    def test_contains(self, default_registry):
        assert "TwoPhaseCommit" in default_registry
        assert "NonExistentSpec" not in default_registry

    def test_len(self, default_registry):
        assert len(default_registry) >= 4

    def test_list_by_tag_distributed(self, default_registry):
        distributed = default_registry.list_by_tag("distributed")
        assert len(distributed) >= 2
        names = {m.name for m in distributed}
        assert "TwoPhaseCommit" in names
        assert "LeaderElection" in names

    def test_list_by_tag_concurrency(self, default_registry):
        conc = default_registry.list_by_tag("concurrency")
        assert len(conc) >= 1
        names = {m.name for m in conc}
        assert "Peterson" in names

    def test_list_by_tag_consensus(self, default_registry):
        cons = default_registry.list_by_tag("consensus")
        assert len(cons) >= 1
        names = {m.name for m in cons}
        assert "Paxos" in names

    def test_custom_registration(self):
        registry = SpecRegistry()
        registry.register(
            "Custom",
            TwoPhaseCommitSpec,
            metadata=SpecMetadata(
                name="Custom",
                description="A custom spec",
                module_path="coacert.specs.two_phase_commit",
                spec_class_name="TwoPhaseCommitSpec",
                parameter_names=["n_participants"],
                parameter_ranges={"n_participants": [2, 3]},
                expected_state_space={},
                tags=["custom"],
            ),
        )
        assert "Custom" in registry
        assert len(registry) == 1
        cls = registry.get("Custom")
        assert cls is TwoPhaseCommitSpec

    def test_get_unknown_spec_raises(self, default_registry):
        with pytest.raises(KeyError):
            default_registry.get("DoesNotExist")


# ---------------------------------------------------------------------------
#  Module Builder
# ---------------------------------------------------------------------------


class TestModuleBuilder:
    def test_build_empty_module(self):
        module = ModuleBuilder("Empty").build()
        assert isinstance(module, Module)
        assert module.name == "Empty"

    def test_add_extends(self):
        module = ModuleBuilder("M").add_extends("Integers", "Sequences").build()
        assert "Integers" in module.extends
        assert "Sequences" in module.extends

    def test_add_constants(self):
        module = ModuleBuilder("M").add_constants("N", "Values").build()
        names = {n for c in module.constants for n in c.names}
        assert "N" in names
        assert "Values" in names

    def test_add_variables(self):
        module = ModuleBuilder("M").add_variables("x", "y", "z").build()
        names = {n for v in module.variables for n in v.names}
        assert names == {"x", "y", "z"}

    def test_add_definition(self):
        module = (
            ModuleBuilder("M")
            .add_definition("Init", bool_lit(True))
            .build()
        )
        assert len(module.definitions) >= 1
        def_names = {d.name for d in module.definitions}
        assert "Init" in def_names

    def test_add_definition_with_params(self):
        module = (
            ModuleBuilder("M")
            .add_definition("F", int_lit(42), params=["x"])
            .build()
        )
        assert len(module.definitions) >= 1

    def test_add_assumption(self):
        module = (
            ModuleBuilder("M")
            .add_assumption(make_eq(ident("N"), int_lit(3)))
            .build()
        )
        assert len(module.assumptions) >= 1

    def test_add_invariant(self):
        module = (
            ModuleBuilder("M")
            .add_invariant("Inv1", bool_lit(True))
            .build()
        )
        assert len(module.properties) >= 1

    def test_add_safety(self):
        module = (
            ModuleBuilder("M")
            .add_safety("Safe1", bool_lit(True))
            .build()
        )
        assert len(module.properties) >= 1

    def test_add_liveness(self):
        module = (
            ModuleBuilder("M")
            .add_liveness("Live1", make_eventually(bool_lit(True)))
            .build()
        )
        assert len(module.properties) >= 1

    def test_add_property_directly(self):
        prop = make_invariant_property("MyInv", bool_lit(True))
        module = ModuleBuilder("M").add_property(prop).build()
        assert len(module.properties) == 1

    def test_fluent_chaining(self):
        module = (
            ModuleBuilder("Full")
            .add_extends("Integers")
            .add_constants("N")
            .add_variables("x", "y")
            .add_definition("Init", bool_lit(True))
            .add_invariant("TypeOK", bool_lit(True))
            .build()
        )
        assert module.name == "Full"
        assert len(module.extends) >= 1
        assert len(module.constants) >= 1
        all_vars = {n for v in module.variables for n in v.names}
        assert "x" in all_vars
        assert "y" in all_vars
        assert len(module.definitions) >= 1
        assert len(module.properties) >= 1


# ---------------------------------------------------------------------------
#  Spec Utils – helper functions
# ---------------------------------------------------------------------------


class TestSpecUtils:
    def test_ident(self):
        node = ident("x")
        assert isinstance(node, Identifier)
        assert node.name == "x"

    def test_primed(self):
        node = primed("x")
        assert isinstance(node, PrimedIdentifier)
        assert node.name == "x"

    def test_int_lit(self):
        node = int_lit(42)
        assert isinstance(node, IntLiteral)
        assert node.value == 42

    def test_bool_lit_true(self):
        node = bool_lit(True)
        assert isinstance(node, BoolLiteral)
        assert node.value is True

    def test_bool_lit_false(self):
        node = bool_lit(False)
        assert isinstance(node, BoolLiteral)
        assert node.value is False

    def test_make_eq(self):
        node = make_eq(ident("x"), int_lit(1))
        assert isinstance(node, OperatorApplication)

    def test_make_neq(self):
        node = make_neq(ident("x"), int_lit(1))
        assert isinstance(node, OperatorApplication)

    def test_make_land(self):
        node = make_land(bool_lit(True), bool_lit(False))
        assert isinstance(node, OperatorApplication)

    def test_make_lor(self):
        node = make_lor(bool_lit(True), bool_lit(False))
        assert isinstance(node, OperatorApplication)

    def test_make_lnot(self):
        node = make_lnot(bool_lit(True))
        assert isinstance(node, OperatorApplication)

    def test_make_set_enum(self):
        node = make_set_enum(int_lit(1), int_lit(2), int_lit(3))
        assert isinstance(node, SetEnumeration)

    def test_make_forall_single(self):
        node = make_forall_single("x", ident("S"), bool_lit(True))
        assert isinstance(node, QuantifiedExpr)

    def test_make_exists_single(self):
        node = make_exists_single("x", ident("S"), bool_lit(True))
        assert isinstance(node, QuantifiedExpr)

    def test_make_conjunction_multiple(self):
        node = make_conjunction([bool_lit(True), bool_lit(False), bool_lit(True)])
        # Should reduce to nested /\ applications or a single expr
        assert node is not None

    def test_make_conjunction_single(self):
        single = bool_lit(True)
        result = make_conjunction([single])
        assert result is not None

    def test_make_disjunction(self):
        node = make_disjunction([bool_lit(True), bool_lit(False)])
        assert node is not None

    def test_make_unchanged(self):
        node = make_unchanged("x", "y")
        assert node is not None

    def test_make_always(self):
        node = make_always(bool_lit(True))
        assert isinstance(node, AlwaysExpr)

    def test_make_eventually(self):
        node = make_eventually(bool_lit(True))
        assert isinstance(node, EventuallyExpr)

    def test_make_ite(self):
        node = make_ite(bool_lit(True), int_lit(1), int_lit(0))
        assert isinstance(node, IfThenElse)

    def test_make_invariant_property(self):
        prop = make_invariant_property("Inv", bool_lit(True))
        assert isinstance(prop, InvariantProperty)
        assert prop.name == "Inv"

    def test_make_safety_property(self):
        prop = make_safety_property("Safe", bool_lit(True))
        assert isinstance(prop, SafetyProperty)

    def test_make_liveness_property(self):
        prop = make_liveness_property("Live", make_eventually(bool_lit(True)))
        assert isinstance(prop, LivenessProperty)

    def test_make_temporal_property(self):
        prop = make_temporal_property("Temp", make_always(bool_lit(True)))
        assert isinstance(prop, TemporalProperty)

    def test_make_variable_decl(self):
        decl = make_variable_decl("a", "b")
        assert isinstance(decl, VariableDecl)
        assert "a" in decl.names
        assert "b" in decl.names

    def test_make_constant_decl(self):
        decl = make_constant_decl("N", "M")
        assert isinstance(decl, ConstantDecl)
        assert "N" in decl.names
        assert "M" in decl.names


# ---------------------------------------------------------------------------
#  Cross-spec validation
# ---------------------------------------------------------------------------


ALL_SPECS = [
    ("TwoPhaseCommit", TwoPhaseCommitSpec, {}),
    ("LeaderElection", LeaderElectionSpec, {}),
    ("Peterson", PetersonSpec, {}),
    ("Paxos", PaxosSpec, {}),
]


class TestSpecValidation:
    @pytest.mark.parametrize("name,cls,kwargs", ALL_SPECS, ids=[s[0] for s in ALL_SPECS])
    def test_validate_no_errors(self, name, cls, kwargs):
        spec = cls(**kwargs)
        errors = spec.validate()
        assert errors == [], f"{name} validation errors: {errors}"

    @pytest.mark.parametrize("name,cls,kwargs", ALL_SPECS, ids=[s[0] for s in ALL_SPECS])
    def test_spec_has_variables(self, name, cls, kwargs):
        module = cls(**kwargs).get_spec()
        assert len(module.variables) > 0, f"{name} has no variables"

    @pytest.mark.parametrize("name,cls,kwargs", ALL_SPECS, ids=[s[0] for s in ALL_SPECS])
    def test_spec_has_definitions(self, name, cls, kwargs):
        module = cls(**kwargs).get_spec()
        assert len(module.definitions) > 0, f"{name} has no definitions"

    @pytest.mark.parametrize("name,cls,kwargs", ALL_SPECS, ids=[s[0] for s in ALL_SPECS])
    def test_spec_has_properties(self, name, cls, kwargs):
        props = cls(**kwargs).get_properties()
        assert len(props) > 0, f"{name} has no properties"

    @pytest.mark.parametrize("name,cls,kwargs", ALL_SPECS, ids=[s[0] for s in ALL_SPECS])
    def test_spec_module_name_nonempty(self, name, cls, kwargs):
        module = cls(**kwargs).get_spec()
        assert module.name, f"{name} has empty module name"

    @pytest.mark.parametrize("name,cls,kwargs", ALL_SPECS, ids=[s[0] for s in ALL_SPECS])
    def test_get_config_returns_dict(self, name, cls, kwargs):
        cfg = cls(**kwargs).get_config()
        assert isinstance(cfg, dict), f"{name}.get_config() did not return dict"


# ---------------------------------------------------------------------------
#  Supported Configurations
# ---------------------------------------------------------------------------


class TestSupportedConfigurations:
    @pytest.mark.parametrize(
        "cls,expected_param",
        [
            (TwoPhaseCommitSpec, "n_participants"),
            (LeaderElectionSpec, "n_nodes"),
            (PetersonSpec, "n_processes"),
            (PaxosSpec, "n_acceptors"),
        ],
        ids=["TwoPhaseCommit", "LeaderElection", "Peterson", "Paxos"],
    )
    def test_nonempty(self, cls, expected_param):
        cfgs = cls.supported_configurations()
        assert isinstance(cfgs, list)
        assert len(cfgs) > 0

    @pytest.mark.parametrize(
        "cls,expected_param",
        [
            (TwoPhaseCommitSpec, "n_participants"),
            (LeaderElectionSpec, "n_nodes"),
            (PetersonSpec, "n_processes"),
            (PaxosSpec, "n_acceptors"),
        ],
        ids=["TwoPhaseCommit", "LeaderElection", "Peterson", "Paxos"],
    )
    def test_has_expected_parameter(self, cls, expected_param):
        cfgs = cls.supported_configurations()
        for cfg in cfgs:
            assert expected_param in cfg, (
                f"Configuration {cfg} missing expected parameter {expected_param}"
            )

    def test_two_phase_commit_configs_valid(self):
        for cfg in TwoPhaseCommitSpec.supported_configurations():
            spec = TwoPhaseCommitSpec(n_participants=cfg["n_participants"])
            assert spec.validate() == []

    def test_leader_election_configs_valid(self):
        for cfg in LeaderElectionSpec.supported_configurations():
            spec = LeaderElectionSpec(n_nodes=cfg["n_nodes"])
            assert spec.validate() == []

    def test_peterson_configs_valid(self):
        for cfg in PetersonSpec.supported_configurations():
            spec = PetersonSpec(n_processes=cfg["n_processes"])
            assert spec.validate() == []

    def test_paxos_configs_valid(self):
        for cfg in PaxosSpec.supported_configurations():
            params = {k: cfg[k] for k in ("n_acceptors", "n_values", "max_ballot") if k in cfg}
            spec = PaxosSpec(**params)
            assert spec.validate() == []
