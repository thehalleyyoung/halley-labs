"""Source location tracking for the TLA-lite parser.

Provides SourceLocation for tracking positions in source files and
SourceMap for associating AST nodes with their source locations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import weakref


@dataclass(frozen=True)
class SourceLocation:
    """Represents a span in a source file.

    Attributes:
        file: The source file path (or '<string>' for inline parsing).
        line: 1-based starting line number.
        column: 1-based starting column number.
        end_line: 1-based ending line number.
        end_column: 1-based ending column number (exclusive).
        offset: 0-based byte offset from start of file.
        length: Length in bytes of the span.
    """
    file: str = "<string>"
    line: int = 0
    column: int = 0
    end_line: int = 0
    end_column: int = 0
    offset: int = 0
    length: int = 0

    def __str__(self) -> str:
        if self.file and self.file != "<string>":
            return f"{self.file}:{self.line}:{self.column}"
        return f"{self.line}:{self.column}"

    def __repr__(self) -> str:
        return (
            f"SourceLocation(file={self.file!r}, "
            f"line={self.line}, col={self.column}, "
            f"end_line={self.end_line}, end_col={self.end_column})"
        )

    def merge(self, other: SourceLocation) -> SourceLocation:
        """Create a new location spanning from this location to `other`."""
        if other.line == 0 and other.column == 0:
            return self
        if self.line == 0 and self.column == 0:
            return other
        start_line = min(self.line, other.line)
        start_col = self.column if self.line <= other.line else other.column
        if self.line == other.line:
            start_col = min(self.column, other.column)
        end_line = max(self.end_line, other.end_line)
        if self.end_line == other.end_line:
            end_col = max(self.end_column, other.end_column)
        elif self.end_line > other.end_line:
            end_col = self.end_column
        else:
            end_col = other.end_column
        start_offset = min(self.offset, other.offset)
        end_offset = max(self.offset + self.length, other.offset + other.length)
        return SourceLocation(
            file=self.file,
            line=start_line,
            column=start_col,
            end_line=end_line,
            end_column=end_col,
            offset=start_offset,
            length=end_offset - start_offset,
        )

    def contains(self, line: int, column: int) -> bool:
        """Check whether a position falls within this location span."""
        if line < self.line or line > self.end_line:
            return False
        if line == self.line and column < self.column:
            return False
        if line == self.end_line and column >= self.end_column:
            return False
        return True

    @staticmethod
    def unknown() -> SourceLocation:
        """Return a sentinel representing an unknown source location."""
        return SourceLocation()


UNKNOWN_LOCATION = SourceLocation.unknown()


@dataclass
class SourceRange:
    """A contiguous range of text within a single file."""
    start: Tuple[int, int]  # (line, column)
    end: Tuple[int, int]    # (line, column)
    file: str = "<string>"

    def to_source_location(self, offset: int = 0, length: int = 0) -> SourceLocation:
        return SourceLocation(
            file=self.file,
            line=self.start[0],
            column=self.start[1],
            end_line=self.end[0],
            end_column=self.end[1],
            offset=offset,
            length=length,
        )


class SourceMap:
    """Bidirectional mapping between AST nodes and their source locations.

    Uses node id() as keys so that any object can be mapped without
    requiring it to be hashable.
    """

    def __init__(self, file: str = "<string>") -> None:
        self.file = file
        self._node_to_loc: Dict[int, SourceLocation] = {}
        self._loc_to_nodes: Dict[SourceLocation, List[int]] = {}
        self._node_refs: Dict[int, Any] = {}
        self._source_lines: List[str] = []

    def set_source(self, source: str) -> None:
        """Store the original source text for snippet extraction."""
        self._source_lines = source.splitlines(keepends=True)

    def register(self, node: Any, location: SourceLocation) -> None:
        """Associate an AST node with a source location."""
        nid = id(node)
        self._node_to_loc[nid] = location
        self._node_refs[nid] = node
        self._loc_to_nodes.setdefault(location, []).append(nid)

    def lookup(self, node: Any) -> Optional[SourceLocation]:
        """Return the source location for a node, or None."""
        return self._node_to_loc.get(id(node))

    def lookup_required(self, node: Any) -> SourceLocation:
        """Return the source location for a node; raise if missing."""
        loc = self.lookup(node)
        if loc is None:
            raise KeyError(f"No source location registered for {node!r}")
        return loc

    def nodes_at(self, line: int, column: int) -> List[Any]:
        """Return all nodes whose source location contains the position."""
        results: List[Any] = []
        for nid, loc in self._node_to_loc.items():
            if loc.contains(line, column):
                ref = self._node_refs.get(nid)
                if ref is not None:
                    results.append(ref)
        return results

    def get_snippet(self, location: SourceLocation, context_lines: int = 0) -> str:
        """Extract the source text corresponding to a location."""
        if not self._source_lines:
            return ""
        start = max(0, location.line - 1 - context_lines)
        end = min(len(self._source_lines), location.end_line + context_lines)
        lines = self._source_lines[start:end]
        result_parts: List[str] = []
        for i, raw_line in enumerate(lines, start=start + 1):
            marker = ">>>" if location.line <= i <= location.end_line else "   "
            result_parts.append(f"{marker} {i:4d} | {raw_line.rstrip()}")
        return "\n".join(result_parts)

    def format_error(
        self, location: SourceLocation, message: str, context_lines: int = 1
    ) -> str:
        """Format an error message with source context."""
        header = f"{location}: error: {message}"
        snippet = self.get_snippet(location, context_lines)
        if snippet:
            return f"{header}\n{snippet}"
        return header

    def all_locations(self) -> List[SourceLocation]:
        """Return all registered source locations."""
        return list(self._node_to_loc.values())

    def merge_maps(self, other: SourceMap) -> None:
        """Merge another source map into this one."""
        self._node_to_loc.update(other._node_to_loc)
        self._node_refs.update(other._node_refs)
        for loc, nids in other._loc_to_nodes.items():
            self._loc_to_nodes.setdefault(loc, []).extend(nids)

    def clear(self) -> None:
        """Remove all mappings."""
        self._node_to_loc.clear()
        self._loc_to_nodes.clear()
        self._node_refs.clear()

    def __len__(self) -> int:
        return len(self._node_to_loc)

    def __contains__(self, node: Any) -> bool:
        return id(node) in self._node_to_loc
