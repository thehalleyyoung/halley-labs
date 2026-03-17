"""Configuration loading and validation for RegSynth CLI."""

from dataclasses import dataclass, field
import json
import os


@dataclass
class Config:
    """RegSynth configuration."""
    project_name: str = "regsynth-project"
    jurisdictions: list = field(default_factory=list)
    output_dir: str = "./output"
    output_format: str = "html"
    log_level: str = "info"
    max_strategies: int = 100
    timeout: float = 300.0
    budget: float = None
    risk_tolerance: float = 0.1
    objectives: list = field(default_factory=lambda: ["cost", "coverage", "risk"])
    benchmark_seed: int = 42
    benchmark_sizes: list = field(default_factory=lambda: [10, 25, 50, 100])
    solver: str = "greedy"
    visualization: dict = field(default_factory=lambda: {"width": 800, "height": 600, "format": "svg"})


def get_default_config() -> Config:
    """Return a Config with all default values."""
    return Config()


def config_to_dict(config: Config) -> dict:
    """Convert a Config to a plain dict."""
    return {
        "project_name": config.project_name,
        "jurisdictions": list(config.jurisdictions),
        "output_dir": config.output_dir,
        "output_format": config.output_format,
        "log_level": config.log_level,
        "max_strategies": config.max_strategies,
        "timeout": config.timeout,
        "budget": config.budget,
        "risk_tolerance": config.risk_tolerance,
        "objectives": list(config.objectives),
        "benchmark_seed": config.benchmark_seed,
        "benchmark_sizes": list(config.benchmark_sizes),
        "solver": config.solver,
        "visualization": dict(config.visualization),
    }


def config_from_dict(data: dict) -> Config:
    """Create a Config from a dict, using defaults for missing keys."""
    defaults = config_to_dict(get_default_config())
    merged = {**defaults, **data}
    return Config(
        project_name=merged["project_name"],
        jurisdictions=merged["jurisdictions"],
        output_dir=merged["output_dir"],
        output_format=merged["output_format"],
        log_level=merged["log_level"],
        max_strategies=merged["max_strategies"],
        timeout=merged["timeout"],
        budget=merged.get("budget"),
        risk_tolerance=merged["risk_tolerance"],
        objectives=merged["objectives"],
        benchmark_seed=merged["benchmark_seed"],
        benchmark_sizes=merged["benchmark_sizes"],
        solver=merged["solver"],
        visualization=merged["visualization"],
    )


def load_config(filepath: str = None) -> Config:
    """Load configuration from file or discover it in the current directory.

    Search order when filepath is None:
      regsynth.json, regsynth.toml, .regsynth.json
    """
    if filepath is not None:
        return _load_file(filepath)

    search_names = ["regsynth.json", "regsynth.toml", ".regsynth.json"]
    for name in search_names:
        if os.path.isfile(name):
            return _load_file(name)
    return get_default_config()


def _load_file(filepath: str) -> Config:
    """Load a config from a specific file."""
    if not os.path.isfile(filepath):
        return get_default_config()

    with open(filepath, "r", encoding="utf-8") as fh:
        text = fh.read()

    if filepath.endswith(".toml"):
        data = _parse_simple_toml(text)
    else:
        data = json.loads(text)
    return config_from_dict(data)


def _parse_simple_toml(text: str) -> dict:
    """Minimal TOML parser sufficient for flat key=value configs."""
    data = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("["):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if value.startswith('"') and value.endswith('"'):
            data[key] = value[1:-1]
        elif value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if not inner:
                data[key] = []
            else:
                items = [v.strip().strip('"').strip("'") for v in inner.split(",")]
                data[key] = items
        elif value.lower() in ("true", "false"):
            data[key] = value.lower() == "true"
        elif "." in value:
            try:
                data[key] = float(value)
            except ValueError:
                data[key] = value
        else:
            try:
                data[key] = int(value)
            except ValueError:
                data[key] = value
    return data


def save_config(config: Config, filepath: str):
    """Save configuration to a JSON file."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(config_to_dict(config), fh, indent=2)


def validate_config(config: Config) -> list:
    """Validate a Config and return a list of error messages (empty if valid)."""
    errors = []
    valid_formats = {"html", "text", "json", "csv"}
    if config.output_format not in valid_formats:
        errors.append(f"Invalid output_format '{config.output_format}'; must be one of {valid_formats}")
    valid_solvers = {"greedy", "exact", "random", "nsga2"}
    if config.solver not in valid_solvers:
        errors.append(f"Invalid solver '{config.solver}'; must be one of {valid_solvers}")
    if config.timeout <= 0:
        errors.append("timeout must be positive")
    if config.max_strategies < 1:
        errors.append("max_strategies must be at least 1")
    if config.risk_tolerance < 0 or config.risk_tolerance > 1:
        errors.append("risk_tolerance must be between 0 and 1")
    if config.budget is not None and config.budget < 0:
        errors.append("budget must be non-negative")
    valid_objectives = {"cost", "coverage", "risk", "timeline"}
    for obj in config.objectives:
        if obj not in valid_objectives:
            errors.append(f"Invalid objective '{obj}'; must be one of {valid_objectives}")
    if not config.benchmark_sizes:
        errors.append("benchmark_sizes must not be empty")
    return errors


def merge_configs(base: Config, override: dict) -> Config:
    """Merge an override dict into a base Config."""
    base_dict = config_to_dict(base)
    for key, value in override.items():
        if value is not None:
            base_dict[key] = value
    return config_from_dict(base_dict)


def config_from_args(args) -> Config:
    """Create a Config from an argparse Namespace, loading file config first."""
    file_cfg = load_config(getattr(args, "config", None))
    overrides = {}
    if hasattr(args, "output") and args.output:
        overrides["output_dir"] = os.path.dirname(args.output) or "./output"
    if hasattr(args, "format") and args.format:
        overrides["output_format"] = args.format
    if hasattr(args, "frameworks") and args.frameworks:
        overrides["jurisdictions"] = args.frameworks
    if hasattr(args, "timeout") and args.timeout is not None:
        overrides["timeout"] = args.timeout
    if hasattr(args, "budget") and args.budget is not None:
        overrides["budget"] = args.budget
    if hasattr(args, "solver") and args.solver:
        overrides["solver"] = args.solver
    if hasattr(args, "seed") and args.seed is not None:
        overrides["benchmark_seed"] = args.seed
    if hasattr(args, "verbose") and args.verbose:
        overrides["log_level"] = "debug"
    if hasattr(args, "quiet") and args.quiet:
        overrides["log_level"] = "error"
    return merge_configs(file_cfg, overrides)
