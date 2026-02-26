"""
Serialization module for the CausalBound financial systemic risk pipeline.

Provides serializers for financial network graphs, structural causal models,
and LP solution bounds with format detection, validation, checksums, and
compression support.
"""

import csv
import gzip
import hashlib
import io
import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np


class SerializationError(Exception):
    """Raised when serialization or deserialization fails."""

    pass


@dataclass
class ValidationResult:
    """Result of a schema/data validation pass."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        if not self.is_valid:
            msg = "; ".join(self.errors)
            raise SerializationError(f"Validation failed: {msg}")


# ---------------------------------------------------------------------------
# Common utilities
# ---------------------------------------------------------------------------

def _ensure_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy/set types to JSON-compatible Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(list(obj), key=str)
    if isinstance(obj, dict):
        return {str(k): _ensure_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_ensure_json_serializable(item) for item in obj]
        return converted
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _compute_checksum(data: bytes) -> str:
    """Return hex SHA-256 checksum for *data* bytes."""
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _compress(data: bytes, path: Union[str, Path]) -> None:
    """Write *data* to *path* with gzip compression."""
    path = Path(path)
    with gzip.open(path, "wb") as fh:
        fh.write(data)


def _decompress(path: Union[str, Path]) -> bytes:
    """Read and decompress a gzip file, returning raw bytes."""
    path = Path(path)
    with gzip.open(path, "rb") as fh:
        return fh.read()


def _detect_format(path: Union[str, Path]) -> str:
    """Auto-detect serialization format from file extension."""
    p = Path(path)
    suffixes = p.suffixes
    if len(suffixes) >= 2 and suffixes[-2] == ".json" and suffixes[-1] == ".gz":
        return "json.gz"
    ext = p.suffix.lower()
    mapping = {
        ".json": "json",
        ".pkl": "pickle",
        ".pickle": "pickle",
        ".gz": "json.gz",
        ".csv": "csv",
        ".dot": "dot",
        ".gv": "dot",
    }
    if ext in mapping:
        return mapping[ext]
    raise SerializationError(f"Cannot detect format from extension '{ext}' of {p}")


def _write_json(obj: Any, path: Path) -> None:
    """Write a JSON-serializable object to *path*."""
    raw = json.dumps(obj, indent=2, sort_keys=False, default=str)
    path.write_text(raw, encoding="utf-8")


def _read_json(path: Path) -> Any:
    """Read JSON from *path*."""
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def _stamp() -> str:
    """ISO-8601 UTC timestamp string."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# NetworkSerializer
# ---------------------------------------------------------------------------

class NetworkSerializer:
    """Save / load financial network graphs (networkx DiGraph)."""

    VERSION = "1.0"

    # -- save ---------------------------------------------------------------

    def save(
        self,
        graph: nx.DiGraph,
        path: Union[str, Path],
        fmt: str = "json",
    ) -> Path:
        """Persist *graph* to *path* in the requested format.

        Supported formats: ``json``, ``pickle``, ``json.gz``.
        Returns the resolved output path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "json":
            payload = self._graph_to_dict(graph)
            _write_json(payload, path)
        elif fmt == "pickle":
            with open(path, "wb") as fh:
                pickle.dump(graph, fh, protocol=pickle.HIGHEST_PROTOCOL)
        elif fmt in ("json.gz", "jsonz"):
            payload = self._graph_to_dict(graph)
            raw = json.dumps(payload, indent=2, default=str).encode("utf-8")
            _compress(raw, path)
        else:
            raise SerializationError(f"Unsupported save format: {fmt}")
        return path

    # -- load ---------------------------------------------------------------

    def load(
        self,
        path: Union[str, Path],
        fmt: Optional[str] = None,
    ) -> nx.DiGraph:
        """Load a graph from *path*.  Format is auto-detected when *fmt* is ``None``."""
        path = Path(path)
        if not path.exists():
            raise SerializationError(f"File not found: {path}")

        fmt = fmt or _detect_format(path)

        if fmt == "json":
            data = _read_json(path)
            vr = self.validate(data)
            vr.raise_if_invalid()
            return self._dict_to_graph(data)
        elif fmt == "pickle":
            with open(path, "rb") as fh:
                graph = pickle.load(fh)
            if not isinstance(graph, nx.DiGraph):
                graph = nx.DiGraph(graph)
            return graph
        elif fmt in ("json.gz", "jsonz"):
            raw = _decompress(path)
            data = json.loads(raw.decode("utf-8"))
            vr = self.validate(data)
            vr.raise_if_invalid()
            return self._dict_to_graph(data)
        else:
            raise SerializationError(f"Unsupported load format: {fmt}")

    # -- validate -----------------------------------------------------------

    def validate(self, data: Any) -> ValidationResult:
        """Check that *data* (dict) has the required node-link fields."""
        errors: List[str] = []
        warnings: List[str] = []

        if not isinstance(data, dict):
            return ValidationResult(False, ["Top-level data must be a dict"])

        graph_data = data.get("graph", data)

        if "nodes" not in graph_data:
            errors.append("Missing required field 'nodes'")
        else:
            nodes = graph_data["nodes"]
            if not isinstance(nodes, list):
                errors.append("'nodes' must be a list")
            else:
                ids_seen: set = set()
                for idx, node in enumerate(nodes):
                    if not isinstance(node, dict):
                        errors.append(f"Node at index {idx} is not a dict")
                        continue
                    nid = node.get("id")
                    if nid is None:
                        errors.append(f"Node at index {idx} missing 'id'")
                    elif nid in ids_seen:
                        warnings.append(f"Duplicate node id '{nid}'")
                    else:
                        ids_seen.add(nid)

        link_key = "links" if "links" in graph_data else "edges"
        if link_key not in graph_data:
            errors.append("Missing required field 'links' or 'edges'")
        else:
            links = graph_data[link_key]
            if not isinstance(links, list):
                errors.append(f"'{link_key}' must be a list")
            else:
                for idx, edge in enumerate(links):
                    if not isinstance(edge, dict):
                        errors.append(f"Edge at index {idx} is not a dict")
                        continue
                    if "source" not in edge:
                        errors.append(f"Edge at index {idx} missing 'source'")
                    if "target" not in edge:
                        errors.append(f"Edge at index {idx} missing 'target'")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)

    # -- adjacency CSV ------------------------------------------------------

    def to_adjacency_csv(self, graph: nx.DiGraph, path: Union[str, Path]) -> Path:
        """Export weighted adjacency matrix as CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        nodes = sorted(graph.nodes())
        node_idx = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)
        matrix = np.zeros((n, n), dtype=float)

        for u, v, d in graph.edges(data=True):
            w = d.get("weight", 1.0)
            matrix[node_idx[u], node_idx[v]] = float(w)

        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            header = [""] + [str(nd) for nd in nodes]
            writer.writerow(header)
            for i, nd in enumerate(nodes):
                row = [str(nd)] + [f"{matrix[i, j]:.8g}" for j in range(n)]
                writer.writerow(row)
        return path

    def from_adjacency_csv(self, path: Union[str, Path]) -> nx.DiGraph:
        """Load a weighted DiGraph from an adjacency-matrix CSV."""
        path = Path(path)
        if not path.exists():
            raise SerializationError(f"File not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            col_nodes = header[1:]

            graph = nx.DiGraph()
            graph.add_nodes_from(col_nodes)

            for row in reader:
                src = row[0]
                for j, val_str in enumerate(row[1:], start=0):
                    val = float(val_str)
                    if val != 0.0:
                        graph.add_edge(src, col_nodes[j], weight=val)
        return graph

    # -- internal helpers ---------------------------------------------------

    def _graph_to_dict(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """Convert a networkx DiGraph to a JSON-friendly dict."""
        nodes_list: List[Dict[str, Any]] = []
        for nid, attrs in graph.nodes(data=True):
            entry: Dict[str, Any] = {"id": nid}
            for k, v in attrs.items():
                entry[k] = _ensure_json_serializable(v)
            nodes_list.append(entry)

        links_list: List[Dict[str, Any]] = []
        for u, v, attrs in graph.edges(data=True):
            entry = {
                "source": _ensure_json_serializable(u),
                "target": _ensure_json_serializable(v),
            }
            for k, val in attrs.items():
                entry[k] = _ensure_json_serializable(val)
            links_list.append(entry)

        return {
            "meta": {
                "serializer": "NetworkSerializer",
                "version": self.VERSION,
                "timestamp": _stamp(),
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "is_directed": graph.is_directed(),
            },
            "graph": {
                "directed": True,
                "multigraph": False,
                "nodes": nodes_list,
                "links": links_list,
            },
        }

    def _dict_to_graph(self, data: Dict[str, Any]) -> nx.DiGraph:
        """Rebuild a DiGraph from node-link dict."""
        graph_data = data.get("graph", data)
        g = nx.DiGraph()

        for node in graph_data.get("nodes", []):
            nid = node.pop("id")
            g.add_node(nid, **node)
            node["id"] = nid  # restore for re-use

        link_key = "links" if "links" in graph_data else "edges"
        for edge in graph_data.get(link_key, []):
            src = edge.get("source")
            tgt = edge.get("target")
            attrs = {k: v for k, v in edge.items() if k not in ("source", "target")}
            g.add_edge(src, tgt, **attrs)

        return g


# ---------------------------------------------------------------------------
# SCMSerializer
# ---------------------------------------------------------------------------

class SCMSerializer:
    """Save / load structural causal model specifications."""

    VERSION = "1.0"

    # -- save ---------------------------------------------------------------

    def save(
        self,
        scm_spec: Dict[str, Any],
        path: Union[str, Path],
        fmt: str = "json",
    ) -> Path:
        """Persist an SCM specification dict to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = self._prepare_payload(scm_spec)

        if fmt == "json":
            _write_json(payload, path)
        elif fmt == "pickle":
            with open(path, "wb") as fh:
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        elif fmt in ("json.gz", "jsonz"):
            raw = json.dumps(payload, indent=2, default=str).encode("utf-8")
            _compress(raw, path)
        else:
            raise SerializationError(f"Unsupported save format: {fmt}")
        return path

    # -- load ---------------------------------------------------------------

    def load(
        self,
        path: Union[str, Path],
        fmt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load an SCM spec from *path* with schema validation."""
        path = Path(path)
        if not path.exists():
            raise SerializationError(f"File not found: {path}")

        fmt = fmt or _detect_format(path)

        if fmt == "json":
            data = _read_json(path)
        elif fmt == "pickle":
            with open(path, "rb") as fh:
                data = pickle.load(fh)
        elif fmt in ("json.gz", "jsonz"):
            raw = _decompress(path)
            data = json.loads(raw.decode("utf-8"))
        else:
            raise SerializationError(f"Unsupported load format: {fmt}")

        vr = self.validate(data)
        vr.raise_if_invalid()
        return data

    # -- validate -----------------------------------------------------------

    def validate(self, data: Any) -> ValidationResult:
        """Validate SCM schema: DAG acyclicity, cardinalities, edge endpoints."""
        errors: List[str] = []
        warnings: List[str] = []

        if not isinstance(data, dict):
            return ValidationResult(False, ["Top-level data must be a dict"])

        scm = data.get("scm", data)

        # --- nodes ---
        nodes = scm.get("nodes")
        if nodes is None:
            errors.append("Missing 'nodes' field")
            return ValidationResult(False, errors, warnings)

        if not isinstance(nodes, list):
            errors.append("'nodes' must be a list")
            return ValidationResult(False, errors, warnings)

        node_ids: set = set()
        for idx, node in enumerate(nodes):
            if not isinstance(node, dict):
                errors.append(f"Node at index {idx} is not a dict")
                continue
            nid = node.get("id")
            if nid is None:
                errors.append(f"Node at index {idx} missing 'id'")
                continue
            if nid in node_ids:
                errors.append(f"Duplicate node id '{nid}'")
            node_ids.add(nid)

            card = node.get("cardinality")
            if card is not None:
                if not isinstance(card, (int, float)):
                    errors.append(f"Node '{nid}': cardinality must be numeric")
                elif card <= 0:
                    errors.append(f"Node '{nid}': cardinality must be > 0, got {card}")

        # --- edges ---
        edges = scm.get("edges")
        if edges is None:
            errors.append("Missing 'edges' field")
            return ValidationResult(False, errors, warnings)

        if not isinstance(edges, list):
            errors.append("'edges' must be a list")
            return ValidationResult(False, errors, warnings)

        valid_edge_types = {"causal", "confounding", "selection", "exposure", "mediator"}
        adj: Dict[str, List[str]] = {nid: [] for nid in node_ids}

        for idx, edge in enumerate(edges):
            if not isinstance(edge, dict):
                errors.append(f"Edge at index {idx} is not a dict")
                continue
            src = edge.get("source")
            tgt = edge.get("target")
            if src is None:
                errors.append(f"Edge at index {idx} missing 'source'")
            elif src not in node_ids:
                errors.append(f"Edge at index {idx}: source '{src}' not in nodes")
            if tgt is None:
                errors.append(f"Edge at index {idx} missing 'target'")
            elif tgt not in node_ids:
                errors.append(f"Edge at index {idx}: target '{tgt}' not in nodes")

            etype = edge.get("type")
            if etype is not None and etype not in valid_edge_types:
                warnings.append(
                    f"Edge {idx} has non-standard type '{etype}'; "
                    f"expected one of {sorted(valid_edge_types)}"
                )

            if src in node_ids and tgt is not None:
                adj.setdefault(src, []).append(tgt)

        # --- DAG acyclicity check via DFS ---
        if not errors:
            WHITE, GREY, BLACK = 0, 1, 2
            colour: Dict[str, int] = {nid: WHITE for nid in node_ids}
            cycle_found = False

            def _dfs(u: str) -> bool:
                nonlocal cycle_found
                colour[u] = GREY
                for v in adj.get(u, []):
                    if colour.get(v) == GREY:
                        cycle_found = True
                        return True
                    if colour.get(v) == WHITE:
                        if _dfs(v):
                            return True
                colour[u] = BLACK
                return False

            for nid in node_ids:
                if colour[nid] == WHITE:
                    if _dfs(nid):
                        break
            if cycle_found:
                errors.append("SCM graph contains a cycle; must be a DAG")

        # --- structural equations metadata ---
        se = scm.get("structural_equations")
        if se is not None:
            if not isinstance(se, dict):
                warnings.append("'structural_equations' should be a dict mapping node -> equation info")
            else:
                for nid, eq_info in se.items():
                    if nid not in node_ids:
                        warnings.append(
                            f"Structural equation defined for unknown node '{nid}'"
                        )

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)

    # -- DOT export ---------------------------------------------------------

    def to_dot(self, scm_spec: Dict[str, Any], path: Union[str, Path]) -> Path:
        """Export SCM specification as DOT (Graphviz) format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        scm = scm_spec.get("scm", scm_spec)
        nodes = scm.get("nodes", [])
        edges = scm.get("edges", [])

        lines: List[str] = ["digraph SCM {", "  rankdir=TB;"]

        edge_style_map = {
            "causal": 'style="solid"',
            "confounding": 'style="dashed" dir="both"',
            "selection": 'style="dotted"',
            "exposure": 'style="bold"',
            "mediator": 'style="solid" color="blue"',
        }

        for node in nodes:
            nid = node.get("id", "?")
            label_parts = [str(nid)]
            card = node.get("cardinality")
            if card is not None:
                label_parts.append(f"|{card}|")
            ntype = node.get("type")
            if ntype is not None:
                label_parts.append(f"({ntype})")
            label = " ".join(label_parts)
            shape = "ellipse"
            if ntype == "latent":
                shape = "diamond"
            elif ntype == "intervention":
                shape = "box"
            lines.append(f'  "{nid}" [label="{label}" shape={shape}];')

        for edge in edges:
            src = edge.get("source", "?")
            tgt = edge.get("target", "?")
            etype = edge.get("type", "causal")
            attrs = edge_style_map.get(etype, 'style="solid"')
            weight = edge.get("weight")
            if weight is not None:
                attrs += f' label="{weight}"'
            lines.append(f'  "{src}" -> "{tgt}" [{attrs}];')

        lines.append("}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    # -- internal helpers ---------------------------------------------------

    def _prepare_payload(self, scm_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Wrap raw SCM spec with metadata and ensure JSON compatibility."""
        scm = scm_spec.get("scm", scm_spec)
        serializable_scm = _ensure_json_serializable(scm)

        return {
            "meta": {
                "serializer": "SCMSerializer",
                "version": self.VERSION,
                "timestamp": _stamp(),
                "num_nodes": len(scm.get("nodes", [])),
                "num_edges": len(scm.get("edges", [])),
            },
            "scm": serializable_scm,
        }


# ---------------------------------------------------------------------------
# BoundSerializer
# ---------------------------------------------------------------------------

class BoundSerializer:
    """Save / load LP solution bounds and dual certificates."""

    VERSION = "1.0"

    # -- save ---------------------------------------------------------------

    def save(
        self,
        result: Dict[str, Any],
        path: Union[str, Path],
        fmt: str = "json",
    ) -> Path:
        """Persist a solver result dict to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = self._prepare_payload(result)

        if fmt == "json":
            raw_bytes = json.dumps(payload, indent=2, default=str).encode("utf-8")
            checksum = _compute_checksum(raw_bytes)
            payload["meta"]["checksum"] = checksum
            _write_json(payload, path)
        elif fmt == "pickle":
            with open(path, "wb") as fh:
                pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        elif fmt in ("json.gz", "jsonz"):
            raw_bytes = json.dumps(payload, indent=2, default=str).encode("utf-8")
            checksum = _compute_checksum(raw_bytes)
            payload["meta"]["checksum"] = checksum
            raw_bytes = json.dumps(payload, indent=2, default=str).encode("utf-8")
            _compress(raw_bytes, path)
        else:
            raise SerializationError(f"Unsupported save format: {fmt}")
        return path

    # -- load ---------------------------------------------------------------

    def load(
        self,
        path: Union[str, Path],
        fmt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load solver results from *path* with validation."""
        path = Path(path)
        if not path.exists():
            raise SerializationError(f"File not found: {path}")

        fmt = fmt or _detect_format(path)

        if fmt == "json":
            data = _read_json(path)
        elif fmt == "pickle":
            with open(path, "rb") as fh:
                data = pickle.load(fh)
        elif fmt in ("json.gz", "jsonz"):
            raw = _decompress(path)
            data = json.loads(raw.decode("utf-8"))
        else:
            raise SerializationError(f"Unsupported load format: {fmt}")

        # Version compatibility check
        meta = data.get("meta", {})
        file_version = meta.get("version", "0.0")
        if file_version != self.VERSION:
            major_file = file_version.split(".")[0]
            major_cur = self.VERSION.split(".")[0]
            if major_file != major_cur:
                raise SerializationError(
                    f"Incompatible version: file has {file_version}, "
                    f"serializer expects {self.VERSION}"
                )

        vr = self.validate(data)
        vr.raise_if_invalid()
        return data

    # -- validate -----------------------------------------------------------

    def validate(self, data: Any) -> ValidationResult:
        """Validate bounds and certificate structure."""
        errors: List[str] = []
        warnings: List[str] = []

        if not isinstance(data, dict):
            return ValidationResult(False, ["Top-level data must be a dict"])

        result = data.get("result", data)

        # --- bounds ---
        bounds = result.get("bounds")
        if bounds is None:
            errors.append("Missing 'bounds' field")
        elif not isinstance(bounds, dict):
            errors.append("'bounds' must be a dict")
        else:
            lower = bounds.get("lower")
            upper = bounds.get("upper")
            if lower is None:
                errors.append("bounds missing 'lower'")
            if upper is None:
                errors.append("bounds missing 'upper'")
            if lower is not None and upper is not None:
                try:
                    lo = float(lower)
                    hi = float(upper)
                    if lo > hi + 1e-9:
                        errors.append(
                            f"Lower bound ({lo}) exceeds upper bound ({hi})"
                        )
                    if lo == hi:
                        warnings.append("Lower and upper bounds are equal (point estimate)")
                except (TypeError, ValueError) as exc:
                    errors.append(f"Bounds not numeric: {exc}")

        # --- certificate ---
        cert = result.get("certificate")
        if cert is not None:
            if not isinstance(cert, dict):
                errors.append("'certificate' must be a dict")
            else:
                dual = cert.get("dual_values")
                if dual is not None and not isinstance(dual, (list, dict)):
                    errors.append("certificate 'dual_values' must be a list or dict")
                status = cert.get("status")
                if status is not None:
                    valid_statuses = {
                        "optimal", "feasible", "infeasible",
                        "unbounded", "timeout", "error",
                    }
                    if status not in valid_statuses:
                        warnings.append(
                            f"Non-standard certificate status '{status}'; "
                            f"expected one of {sorted(valid_statuses)}"
                        )

        # --- diagnostics ---
        diag = result.get("diagnostics")
        if diag is not None:
            if not isinstance(diag, dict):
                warnings.append("'diagnostics' should be a dict")
            else:
                solve_time = diag.get("solve_time_seconds")
                if solve_time is not None:
                    try:
                        st = float(solve_time)
                        if st < 0:
                            warnings.append("Negative solve time recorded")
                    except (TypeError, ValueError):
                        warnings.append("solve_time_seconds is not numeric")

                iterations = diag.get("iterations")
                if iterations is not None:
                    try:
                        it = int(iterations)
                        if it < 0:
                            warnings.append("Negative iteration count")
                    except (TypeError, ValueError):
                        warnings.append("iterations is not an integer")

        # --- solver config ---
        config = result.get("solver_config")
        if config is not None and not isinstance(config, dict):
            warnings.append("'solver_config' should be a dict")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)

    # -- batch save / load --------------------------------------------------

    def save_batch(
        self,
        results: Sequence[Dict[str, Any]],
        path: Union[str, Path],
    ) -> Path:
        """Save multiple solver results in a single file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        entries: List[Dict[str, Any]] = []
        for idx, res in enumerate(results):
            payload = self._prepare_payload(res)
            payload["meta"]["batch_index"] = idx
            entries.append(payload)

        batch_payload = {
            "meta": {
                "serializer": "BoundSerializer",
                "version": self.VERSION,
                "timestamp": _stamp(),
                "batch_size": len(entries),
            },
            "results": entries,
        }

        raw_bytes = json.dumps(batch_payload, indent=2, default=str).encode("utf-8")
        checksum = _compute_checksum(raw_bytes)
        batch_payload["meta"]["checksum"] = checksum
        raw_bytes = json.dumps(batch_payload, indent=2, default=str).encode("utf-8")

        # Use gzip if the extension suggests it
        if str(path).endswith(".gz"):
            _compress(raw_bytes, path)
        else:
            path.write_bytes(raw_bytes)
        return path

    def load_batch(
        self,
        path: Union[str, Path],
    ) -> List[Dict[str, Any]]:
        """Load a batch of solver results."""
        path = Path(path)
        if not path.exists():
            raise SerializationError(f"File not found: {path}")

        if str(path).endswith(".gz"):
            raw = _decompress(path)
            data = json.loads(raw.decode("utf-8"))
        else:
            data = _read_json(path)

        if not isinstance(data, dict):
            raise SerializationError("Batch file top-level must be a dict")

        results_raw = data.get("results")
        if results_raw is None:
            raise SerializationError("Batch file missing 'results' key")
        if not isinstance(results_raw, list):
            raise SerializationError("'results' must be a list")

        validated: List[Dict[str, Any]] = []
        for idx, entry in enumerate(results_raw):
            vr = self.validate(entry)
            if not vr.is_valid:
                raise SerializationError(
                    f"Batch entry {idx} validation failed: "
                    + "; ".join(vr.errors)
                )
            validated.append(entry)

        return validated

    # -- internal helpers ---------------------------------------------------

    def _prepare_payload(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Wrap a solver result dict with metadata."""
        result_body = result.get("result", result)
        serializable = _ensure_json_serializable(result_body)

        bounds = result_body.get("bounds", {})
        lower = bounds.get("lower")
        upper = bounds.get("upper")
        gap = None
        if lower is not None and upper is not None:
            try:
                gap = float(upper) - float(lower)
            except (TypeError, ValueError):
                pass

        return {
            "meta": {
                "serializer": "BoundSerializer",
                "version": self.VERSION,
                "timestamp": _stamp(),
                "bound_gap": gap,
            },
            "result": serializable,
        }
