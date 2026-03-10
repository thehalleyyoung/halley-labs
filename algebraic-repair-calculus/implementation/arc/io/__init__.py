"""
``arc.io`` — Serialization and deserialization for ARC.

Re-exports JSON, YAML, and schema utilities::

    from arc.io import PipelineSpec, YAMLPipelineSpec, load_pipeline
"""

from arc.io.json_format import (
    ARCJSONEncoder,
    CURRENT_SPEC_VERSION,
    SUPPORTED_SPEC_VERSIONS,
    DeltaSerializer,
    PipelineSpec,
    RepairPlanSerializer,
    dumps,
    loads,
)
from arc.io.yaml_format import (
    YAMLPipelineSpec,
    from_template,
    get_template_yaml,
    list_templates,
)
from arc.io.schema import (
    DELTA_SPEC_SCHEMA_V1,
    EXAMPLE_DELTA_SPEC,
    EXAMPLE_PIPELINE_SPEC,
    EXAMPLE_REPAIR_PLAN,
    PIPELINE_SPEC_SCHEMA_V1,
    REPAIR_PLAN_SCHEMA_V1,
    SpecMigrator,
    get_delta_schema,
    get_pipeline_schema,
    get_repair_plan_schema,
)
from arc.io.dbt_loader import (
    DbtModel,
    DbtProject,
    dbt_model_to_schema,
    dbt_project_to_pipeline,
    load_dbt_project,
)
from arc.io.migration_parser import (
    MigrationInfo,
    MigrationOp,
    load_migration_directory,
    migration_to_schema_delta,
    parse_django_migration,
)

__all__ = [
    # json_format
    "PipelineSpec",
    "DeltaSerializer",
    "RepairPlanSerializer",
    "ARCJSONEncoder",
    "CURRENT_SPEC_VERSION",
    "SUPPORTED_SPEC_VERSIONS",
    "dumps",
    "loads",
    # yaml_format
    "YAMLPipelineSpec",
    "from_template",
    "list_templates",
    "get_template_yaml",
    # schema
    "PIPELINE_SPEC_SCHEMA_V1",
    "DELTA_SPEC_SCHEMA_V1",
    "REPAIR_PLAN_SCHEMA_V1",
    "get_pipeline_schema",
    "get_delta_schema",
    "get_repair_plan_schema",
    "EXAMPLE_PIPELINE_SPEC",
    "EXAMPLE_DELTA_SPEC",
    "EXAMPLE_REPAIR_PLAN",
    "SpecMigrator",
    # dbt_loader
    "DbtModel",
    "DbtProject",
    "dbt_model_to_schema",
    "dbt_project_to_pipeline",
    "load_dbt_project",
    # migration_parser
    "MigrationInfo",
    "MigrationOp",
    "load_migration_directory",
    "migration_to_schema_delta",
    "parse_django_migration",
]
