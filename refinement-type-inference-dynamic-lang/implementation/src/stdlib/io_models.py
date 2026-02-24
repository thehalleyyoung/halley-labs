from __future__ import annotations

"""
Refinement type models for I/O operations in dynamically-typed languages.

Tracks resource state transitions (open → reading → closed), preconditions
for valid operations, postconditions, exception conditions, and encoding /
mode refinements across file, network, subprocess, serialization, database,
and other I/O subsystems.
"""

import enum
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)


# ---------------------------------------------------------------------------
# Local type definitions (no cross-module imports)
# ---------------------------------------------------------------------------

class RefinementSort(enum.Enum):
    INT = "int"
    FLOAT = "float"
    STR = "str"
    BYTES = "bytes"
    BOOL = "bool"
    NONE = "none"
    LIST = "list"
    TUPLE = "tuple"
    DICT = "dict"
    SET = "set"
    ANY = "any"
    UNION = "union"
    OBJECT = "object"
    CALLABLE = "callable"
    ITERATOR = "iterator"
    FILE = "file"
    SOCKET = "socket"
    CONNECTION = "connection"
    CURSOR = "cursor"
    STREAM = "stream"
    PATH = "path"
    PROCESS = "process"


@dataclass(frozen=True)
class RefinementPredicate:
    """A single refinement predicate."""
    variable: str
    operator: str
    operand: Any
    symbolic_operand: Optional[str] = None

    def negate(self) -> RefinementPredicate:
        neg_map = {
            ">": "<=", ">=": "<", "<": ">=", "<=": ">",
            "==": "!=", "!=": "==", "in": "not_in", "not_in": "in",
            "is": "is_not", "is_not": "is",
        }
        return RefinementPredicate(
            variable=self.variable,
            operator=neg_map.get(self.operator, self.operator),
            operand=self.operand,
            symbolic_operand=self.symbolic_operand,
        )


@dataclass
class RefinementType:
    """A base type refined by a conjunction of predicates."""
    base: RefinementSort
    predicates: List[RefinementPredicate] = field(default_factory=list)
    nullable: bool = False

    def add_predicate(self, pred: RefinementPredicate) -> RefinementType:
        return RefinementType(
            base=self.base,
            predicates=self.predicates + [pred],
            nullable=self.nullable,
        )


@dataclass
class OperationResult:
    """Result of modelling a single I/O operation."""
    preconditions: List[RefinementPredicate] = field(default_factory=list)
    postconditions: List[RefinementPredicate] = field(default_factory=list)
    return_type: Optional[RefinementType] = None
    exceptions: List[Tuple[str, List[RefinementPredicate]]] = field(default_factory=list)
    modifies_receiver: bool = False
    state_transition: Optional[Tuple[str, str]] = None  # (from_state, to_state)


# ====================================================================
# IOStateModel — tracks resource lifecycle
# ====================================================================

class ResourceState(enum.Enum):
    UNINITIALIZED = "uninitialized"
    OPEN = "open"
    READING = "reading"
    WRITING = "writing"
    CLOSED = "closed"
    ERROR = "error"


@dataclass
class IOStateModel:
    """
    Tracks file handle / resource states through their lifecycle.

    Valid transitions:
        UNINITIALIZED → OPEN → {READING, WRITING} → CLOSED
        Any state → ERROR
        READING / WRITING → OPEN (after flush / seek)
    """
    state: ResourceState = ResourceState.UNINITIALIZED
    valid_transitions: Dict[ResourceState, FrozenSet[ResourceState]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.valid_transitions:
            self.valid_transitions = {
                ResourceState.UNINITIALIZED: frozenset({ResourceState.OPEN, ResourceState.ERROR}),
                ResourceState.OPEN: frozenset({
                    ResourceState.READING, ResourceState.WRITING,
                    ResourceState.CLOSED, ResourceState.ERROR,
                }),
                ResourceState.READING: frozenset({
                    ResourceState.OPEN, ResourceState.CLOSED, ResourceState.ERROR,
                }),
                ResourceState.WRITING: frozenset({
                    ResourceState.OPEN, ResourceState.CLOSED, ResourceState.ERROR,
                }),
                ResourceState.CLOSED: frozenset({ResourceState.ERROR}),
                ResourceState.ERROR: frozenset(),
            }

    def can_transition(self, target: ResourceState) -> bool:
        allowed = self.valid_transitions.get(self.state, frozenset())
        return target in allowed

    def transition(self, target: ResourceState) -> IOStateModel:
        if not self.can_transition(target):
            raise ValueError(f"Invalid transition from {self.state} to {target}")
        return IOStateModel(state=target, valid_transitions=self.valid_transitions)

    def is_open(self) -> bool:
        return self.state in {ResourceState.OPEN, ResourceState.READING, ResourceState.WRITING}

    def is_closed(self) -> bool:
        return self.state == ResourceState.CLOSED


# ====================================================================
# FileModel
# ====================================================================

class FileMode(enum.Enum):
    READ = "r"
    WRITE = "w"
    APPEND = "a"
    READ_BINARY = "rb"
    WRITE_BINARY = "wb"
    APPEND_BINARY = "ab"
    READ_WRITE = "r+"
    WRITE_READ = "w+"
    APPEND_READ = "a+"
    READ_WRITE_BINARY = "r+b"
    WRITE_READ_BINARY = "w+b"
    APPEND_READ_BINARY = "a+b"
    EXCLUSIVE_WRITE = "x"
    EXCLUSIVE_WRITE_BINARY = "xb"


_TEXT_MODES = {
    FileMode.READ, FileMode.WRITE, FileMode.APPEND,
    FileMode.READ_WRITE, FileMode.WRITE_READ, FileMode.APPEND_READ,
    FileMode.EXCLUSIVE_WRITE,
}

_BINARY_MODES = {
    FileMode.READ_BINARY, FileMode.WRITE_BINARY, FileMode.APPEND_BINARY,
    FileMode.READ_WRITE_BINARY, FileMode.WRITE_READ_BINARY,
    FileMode.APPEND_READ_BINARY, FileMode.EXCLUSIVE_WRITE_BINARY,
}

_READABLE_MODES = {
    FileMode.READ, FileMode.READ_BINARY,
    FileMode.READ_WRITE, FileMode.WRITE_READ, FileMode.APPEND_READ,
    FileMode.READ_WRITE_BINARY, FileMode.WRITE_READ_BINARY,
    FileMode.APPEND_READ_BINARY,
}

_WRITABLE_MODES = {
    FileMode.WRITE, FileMode.WRITE_BINARY,
    FileMode.APPEND, FileMode.APPEND_BINARY,
    FileMode.READ_WRITE, FileMode.WRITE_READ, FileMode.APPEND_READ,
    FileMode.READ_WRITE_BINARY, FileMode.WRITE_READ_BINARY,
    FileMode.APPEND_READ_BINARY,
    FileMode.EXCLUSIVE_WRITE, FileMode.EXCLUSIVE_WRITE_BINARY,
}


@dataclass
class FileModel:
    """
    Refinement model for Python file objects (``open()`` return value).

    Tracks: mode, encoding, state, text vs binary, readability, writability.
    """
    io_state: IOStateModel = field(default_factory=lambda: IOStateModel(state=ResourceState.UNINITIALIZED))
    mode: Optional[FileMode] = None
    encoding: Optional[str] = None
    path_refinement: Optional[str] = None

    @property
    def is_text(self) -> bool:
        return self.mode in _TEXT_MODES if self.mode else True

    @property
    def is_binary(self) -> bool:
        return self.mode in _BINARY_MODES if self.mode else False

    @property
    def is_readable(self) -> bool:
        return self.mode in _READABLE_MODES if self.mode else False

    @property
    def is_writable(self) -> bool:
        return self.mode in _WRITABLE_MODES if self.mode else False

    def _open_precond(self) -> List[RefinementPredicate]:
        return [RefinementPredicate("self.state", "==", "open")]

    def _closed_exc(self) -> Tuple[str, List[RefinementPredicate]]:
        return ("ValueError", [RefinementPredicate("self.state", "==", "closed")])

    # -- open() (class method / factory) -------------------------------------

    @staticmethod
    def model_open(
        mode_str: str = "r",
        encoding: Optional[str] = None,
    ) -> OperationResult:
        try:
            mode = FileMode(mode_str)
        except ValueError:
            mode = FileMode.READ
        preds = [
            RefinementPredicate("result.state", "==", "open"),
            RefinementPredicate("result.mode", "==", mode_str),
        ]
        if encoding:
            preds.append(RefinementPredicate("result.encoding", "==", encoding))
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.FILE, predicates=preds),
            state_transition=("uninitialized", "open"),
            exceptions=[
                ("FileNotFoundError", [RefinementPredicate("mode", "==", "r")]),
                ("FileExistsError", [RefinementPredicate("mode", "==", "x")]),
                ("PermissionError", []),
                ("IsADirectoryError", []),
                ("OSError", []),
            ],
        )

    # -- read ----------------------------------------------------------------

    def model_read(self, size: Optional[int] = None) -> OperationResult:
        preconds = self._open_precond()
        if not self.is_readable:
            preconds.append(RefinementPredicate("self.readable()", "==", True))
        preds: List[RefinementPredicate] = []
        if self.is_text:
            base = RefinementSort.STR
        else:
            base = RefinementSort.BYTES
        if size is not None and size >= 0:
            preds.append(RefinementPredicate("len(result)", "<=", size))
        preds.append(RefinementPredicate("len(result)", ">=", 0))
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=base, predicates=preds),
            state_transition=("open", "reading"),
            exceptions=[
                self._closed_exc(),
                ("UnsupportedOperation", [RefinementPredicate("self.readable()", "==", False)]),
            ],
        )

    # -- readline ------------------------------------------------------------

    def model_readline(self, size: Optional[int] = None) -> OperationResult:
        preconds = self._open_precond()
        base = RefinementSort.STR if self.is_text else RefinementSort.BYTES
        preds = [RefinementPredicate("len(result)", ">=", 0)]
        if size is not None and size >= 0:
            preds.append(RefinementPredicate("len(result)", "<=", size))
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=base, predicates=preds),
            state_transition=("open", "reading"),
            exceptions=[self._closed_exc()],
        )

    # -- readlines -----------------------------------------------------------

    def model_readlines(self, hint: Optional[int] = None) -> OperationResult:
        preconds = self._open_precond()
        preds = [RefinementPredicate("len(result)", ">=", 0)]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.LIST, predicates=preds),
            state_transition=("open", "reading"),
            exceptions=[self._closed_exc()],
        )

    # -- write ---------------------------------------------------------------

    def model_write(self, data_type: RefinementType) -> OperationResult:
        preconds = self._open_precond()
        if not self.is_writable:
            preconds.append(RefinementPredicate("self.writable()", "==", True))
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(
                base=RefinementSort.INT,
                predicates=[RefinementPredicate("result", ">=", 0)],
            ),
            modifies_receiver=True,
            state_transition=("open", "writing"),
            exceptions=[
                self._closed_exc(),
                ("UnsupportedOperation", [RefinementPredicate("self.writable()", "==", False)]),
                ("TypeError", [RefinementPredicate("type(data)", "!=", "expected")]),
            ],
        )

    # -- writelines ----------------------------------------------------------

    def model_writelines(self, lines_type: RefinementType) -> OperationResult:
        preconds = self._open_precond()
        if not self.is_writable:
            preconds.append(RefinementPredicate("self.writable()", "==", True))
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            state_transition=("open", "writing"),
            exceptions=[self._closed_exc()],
        )

    # -- close ---------------------------------------------------------------

    def model_close(self) -> OperationResult:
        return OperationResult(
            postconditions=[RefinementPredicate("self.state", "==", "closed")],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            state_transition=("open", "closed"),
        )

    # -- seek ----------------------------------------------------------------

    def model_seek(self, offset: int = 0, whence: int = 0) -> OperationResult:
        preconds = self._open_precond()
        preds = [RefinementPredicate("result", ">=", 0)]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.INT, predicates=preds),
            modifies_receiver=True,
            exceptions=[
                self._closed_exc(),
                ("UnsupportedOperation", [RefinementPredicate("self.seekable()", "==", False)]),
            ],
        )

    # -- tell ----------------------------------------------------------------

    def model_tell(self) -> OperationResult:
        preconds = self._open_precond()
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(
                base=RefinementSort.INT,
                predicates=[RefinementPredicate("result", ">=", 0)],
            ),
            exceptions=[self._closed_exc()],
        )

    # -- flush ---------------------------------------------------------------

    def model_flush(self) -> OperationResult:
        preconds = self._open_precond()
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            exceptions=[self._closed_exc()],
        )

    # -- fileno --------------------------------------------------------------

    def model_fileno(self) -> OperationResult:
        preconds = self._open_precond()
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(
                base=RefinementSort.INT,
                predicates=[RefinementPredicate("result", ">=", 0)],
            ),
            exceptions=[self._closed_exc(), ("OSError", [])],
        )

    # -- truncate ------------------------------------------------------------

    def model_truncate(self, size: Optional[int] = None) -> OperationResult:
        preconds = self._open_precond()
        if not self.is_writable:
            preconds.append(RefinementPredicate("self.writable()", "==", True))
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(
                base=RefinementSort.INT,
                predicates=[RefinementPredicate("result", ">=", 0)],
            ),
            modifies_receiver=True,
            exceptions=[
                self._closed_exc(),
                ("UnsupportedOperation", [RefinementPredicate("self.writable()", "==", False)]),
            ],
        )

    # -- readable / writable / seekable --------------------------------------

    def model_readable(self) -> OperationResult:
        preds: List[RefinementPredicate] = []
        if self.mode is not None:
            preds.append(RefinementPredicate("result", "==", self.is_readable))
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL, predicates=preds))

    def model_writable(self) -> OperationResult:
        preds: List[RefinementPredicate] = []
        if self.mode is not None:
            preds.append(RefinementPredicate("result", "==", self.is_writable))
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL, predicates=preds))

    def model_seekable(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL))

    # -- context manager protocol --------------------------------------------

    def model_enter(self) -> OperationResult:
        return OperationResult(
            preconditions=self._open_precond(),
            postconditions=[RefinementPredicate("self.state", "==", "open")],
            return_type=RefinementType(base=RefinementSort.FILE),
        )

    def model_exit(self, exc_type: Any = None, exc_val: Any = None, exc_tb: Any = None) -> OperationResult:
        return OperationResult(
            postconditions=[RefinementPredicate("self.state", "==", "closed")],
            return_type=RefinementType(base=RefinementSort.BOOL),
            modifies_receiver=True,
            state_transition=("open", "closed"),
        )

    # -- __iter__ / __next__ -------------------------------------------------

    def model_iter(self) -> OperationResult:
        return OperationResult(
            preconditions=self._open_precond(),
            return_type=RefinementType(base=RefinementSort.ITERATOR),
            exceptions=[self._closed_exc()],
        )

    def model_next(self) -> OperationResult:
        preconds = self._open_precond()
        base = RefinementSort.STR if self.is_text else RefinementSort.BYTES
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=base),
            exceptions=[
                self._closed_exc(),
                ("StopIteration", [RefinementPredicate("eof", "==", True)]),
            ],
        )


# ====================================================================
# StreamModel
# ====================================================================

class StreamKind(enum.Enum):
    STDIN = "stdin"
    STDOUT = "stdout"
    STDERR = "stderr"
    TEXT_IO_WRAPPER = "TextIOWrapper"
    BUFFERED_READER = "BufferedReader"
    BUFFERED_WRITER = "BufferedWriter"
    BYTES_IO = "BytesIO"
    STRING_IO = "StringIO"


@dataclass
class StreamModel:
    """
    Refinement model for stream objects (stdin, stdout, stderr, BytesIO,
    StringIO, and buffered streams).
    """
    kind: StreamKind = StreamKind.STDOUT
    io_state: IOStateModel = field(default_factory=lambda: IOStateModel(state=ResourceState.OPEN))

    @property
    def is_text(self) -> bool:
        return self.kind in {
            StreamKind.STDIN, StreamKind.STDOUT, StreamKind.STDERR,
            StreamKind.TEXT_IO_WRAPPER, StreamKind.STRING_IO,
        }

    @property
    def is_binary(self) -> bool:
        return self.kind in {
            StreamKind.BUFFERED_READER, StreamKind.BUFFERED_WRITER,
            StreamKind.BYTES_IO,
        }

    @property
    def is_readable(self) -> bool:
        return self.kind in {
            StreamKind.STDIN, StreamKind.BUFFERED_READER,
            StreamKind.BYTES_IO, StreamKind.STRING_IO,
            StreamKind.TEXT_IO_WRAPPER,
        }

    @property
    def is_writable(self) -> bool:
        return self.kind in {
            StreamKind.STDOUT, StreamKind.STDERR,
            StreamKind.BUFFERED_WRITER, StreamKind.BYTES_IO,
            StreamKind.STRING_IO, StreamKind.TEXT_IO_WRAPPER,
        }

    def _open_precond(self) -> List[RefinementPredicate]:
        return [RefinementPredicate("self.closed", "==", False)]

    def model_read(self, size: Optional[int] = None) -> OperationResult:
        base = RefinementSort.STR if self.is_text else RefinementSort.BYTES
        preds = [RefinementPredicate("len(result)", ">=", 0)]
        if size is not None:
            preds.append(RefinementPredicate("len(result)", "<=", size))
        return OperationResult(
            preconditions=self._open_precond(),
            return_type=RefinementType(base=base, predicates=preds),
            exceptions=[("ValueError", [RefinementPredicate("self.closed", "==", True)])],
        )

    def model_write(self, data_type: RefinementType) -> OperationResult:
        return OperationResult(
            preconditions=self._open_precond(),
            return_type=RefinementType(
                base=RefinementSort.INT,
                predicates=[RefinementPredicate("result", ">=", 0)],
            ),
            modifies_receiver=True,
            exceptions=[("ValueError", [RefinementPredicate("self.closed", "==", True)])],
        )

    def model_readline(self, size: Optional[int] = None) -> OperationResult:
        base = RefinementSort.STR if self.is_text else RefinementSort.BYTES
        preds = [RefinementPredicate("len(result)", ">=", 0)]
        return OperationResult(
            preconditions=self._open_precond(),
            return_type=RefinementType(base=base, predicates=preds),
            exceptions=[("ValueError", [RefinementPredicate("self.closed", "==", True)])],
        )

    def model_readlines(self) -> OperationResult:
        return OperationResult(
            preconditions=self._open_precond(),
            return_type=RefinementType(base=RefinementSort.LIST, predicates=[RefinementPredicate("len(result)", ">=", 0)]),
            exceptions=[("ValueError", [RefinementPredicate("self.closed", "==", True)])],
        )

    def model_writelines(self, lines_type: RefinementType) -> OperationResult:
        return OperationResult(
            preconditions=self._open_precond(),
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            exceptions=[("ValueError", [RefinementPredicate("self.closed", "==", True)])],
        )

    def model_close(self) -> OperationResult:
        return OperationResult(
            postconditions=[RefinementPredicate("self.closed", "==", True)],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            state_transition=("open", "closed"),
        )

    def model_seek(self, offset: int = 0, whence: int = 0) -> OperationResult:
        return OperationResult(
            preconditions=self._open_precond(),
            return_type=RefinementType(base=RefinementSort.INT, predicates=[RefinementPredicate("result", ">=", 0)]),
            modifies_receiver=True,
        )

    def model_tell(self) -> OperationResult:
        return OperationResult(
            preconditions=self._open_precond(),
            return_type=RefinementType(base=RefinementSort.INT, predicates=[RefinementPredicate("result", ">=", 0)]),
        )

    def model_flush(self) -> OperationResult:
        return OperationResult(
            preconditions=self._open_precond(),
            return_type=RefinementType(base=RefinementSort.NONE),
        )

    def model_getvalue(self) -> OperationResult:
        """BytesIO.getvalue() / StringIO.getvalue()"""
        if self.kind == StreamKind.BYTES_IO:
            base = RefinementSort.BYTES
        else:
            base = RefinementSort.STR
        return OperationResult(
            return_type=RefinementType(base=base, predicates=[RefinementPredicate("len(result)", ">=", 0)]),
        )

    def model_truncate(self, size: Optional[int] = None) -> OperationResult:
        return OperationResult(
            preconditions=self._open_precond(),
            return_type=RefinementType(base=RefinementSort.INT, predicates=[RefinementPredicate("result", ">=", 0)]),
            modifies_receiver=True,
        )

    def model_enter(self) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.STREAM),
        )

    def model_exit(self) -> OperationResult:
        return OperationResult(
            postconditions=[RefinementPredicate("self.closed", "==", True)],
            return_type=RefinementType(base=RefinementSort.BOOL),
            modifies_receiver=True,
        )


# ====================================================================
# PathIOModel
# ====================================================================

@dataclass
class PathIOModel:
    """
    Refinement model for ``pathlib.Path`` I/O operations.
    """

    def model_read_text(self, encoding: str = "utf-8") -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.STR, predicates=[RefinementPredicate("len(result)", ">=", 0)]),
            exceptions=[
                ("FileNotFoundError", [RefinementPredicate("self.exists()", "==", False)]),
                ("PermissionError", []),
                ("IsADirectoryError", [RefinementPredicate("self.is_dir()", "==", True)]),
                ("UnicodeDecodeError", []),
            ],
        )

    def model_write_text(self, data: str, encoding: str = "utf-8") -> OperationResult:
        return OperationResult(
            return_type=RefinementType(
                base=RefinementSort.INT,
                predicates=[RefinementPredicate("result", ">=", 0)],
            ),
            exceptions=[
                ("PermissionError", []),
                ("IsADirectoryError", [RefinementPredicate("self.is_dir()", "==", True)]),
                ("OSError", []),
            ],
        )

    def model_read_bytes(self) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.BYTES, predicates=[RefinementPredicate("len(result)", ">=", 0)]),
            exceptions=[
                ("FileNotFoundError", [RefinementPredicate("self.exists()", "==", False)]),
                ("PermissionError", []),
                ("IsADirectoryError", [RefinementPredicate("self.is_dir()", "==", True)]),
            ],
        )

    def model_write_bytes(self, data: bytes) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(
                base=RefinementSort.INT,
                predicates=[RefinementPredicate("result", ">=", 0)],
            ),
            exceptions=[("PermissionError", []), ("OSError", [])],
        )

    def model_open(self, mode: str = "r", encoding: Optional[str] = None) -> OperationResult:
        return FileModel.model_open(mode_str=mode, encoding=encoding)

    def model_exists(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL))

    def model_is_file(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL))

    def model_is_dir(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL))

    def model_is_symlink(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL))

    def model_stat(self) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.OBJECT),
            exceptions=[
                ("FileNotFoundError", [RefinementPredicate("self.exists()", "==", False)]),
                ("OSError", []),
            ],
        )

    def model_mkdir(self, parents: bool = False, exist_ok: bool = False) -> OperationResult:
        excs: List[Tuple[str, List[RefinementPredicate]]] = [("PermissionError", [])]
        if not exist_ok:
            excs.append(("FileExistsError", [RefinementPredicate("self.exists()", "==", True)]))
        if not parents:
            excs.append(("FileNotFoundError", [RefinementPredicate("self.parent.exists()", "==", False)]))
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.NONE),
            exceptions=excs,
        )

    def model_unlink(self, missing_ok: bool = False) -> OperationResult:
        excs: List[Tuple[str, List[RefinementPredicate]]] = []
        if not missing_ok:
            excs.append(("FileNotFoundError", [RefinementPredicate("self.exists()", "==", False)]))
        excs.append(("PermissionError", []))
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.NONE),
            exceptions=excs,
        )

    def model_rename(self, target: str) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.PATH),
            exceptions=[
                ("FileNotFoundError", [RefinementPredicate("self.exists()", "==", False)]),
                ("OSError", []),
            ],
        )

    def model_resolve(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.PATH))

    def model_iterdir(self) -> OperationResult:
        return OperationResult(
            preconditions=[RefinementPredicate("self.is_dir()", "==", True)],
            return_type=RefinementType(base=RefinementSort.ITERATOR),
            exceptions=[
                ("NotADirectoryError", [RefinementPredicate("self.is_dir()", "==", False)]),
                ("PermissionError", []),
            ],
        )

    def model_glob(self, pattern: str) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.ITERATOR))

    def model_rglob(self, pattern: str) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.ITERATOR))


# ====================================================================
# NetworkIOModel
# ====================================================================

class SocketState(enum.Enum):
    CREATED = "created"
    BOUND = "bound"
    LISTENING = "listening"
    CONNECTED = "connected"
    CLOSED = "closed"


@dataclass
class NetworkIOModel:
    """
    Refinement model for socket operations, urllib, and http.client.
    """
    socket_state: SocketState = SocketState.CREATED
    address_family: Optional[str] = None
    socket_type: Optional[str] = None

    def _socket_open_precond(self) -> List[RefinementPredicate]:
        return [RefinementPredicate("self.state", "!=", "closed")]

    # -- socket operations ---------------------------------------------------

    def model_socket_create(self, family: str = "AF_INET", type_: str = "SOCK_STREAM") -> OperationResult:
        preds = [
            RefinementPredicate("result.state", "==", "created"),
            RefinementPredicate("result.family", "==", family),
            RefinementPredicate("result.type", "==", type_),
        ]
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.SOCKET, predicates=preds),
            exceptions=[("OSError", [])],
        )

    def model_bind(self, address: Tuple[str, int]) -> OperationResult:
        preconds = [RefinementPredicate("self.state", "==", "created")]
        return OperationResult(
            preconditions=preconds,
            postconditions=[RefinementPredicate("self.state", "==", "bound")],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            state_transition=("created", "bound"),
            exceptions=[
                ("OSError", [RefinementPredicate("address_in_use", "==", True)]),
                ("PermissionError", []),
            ],
        )

    def model_listen(self, backlog: int = 5) -> OperationResult:
        preconds = [RefinementPredicate("self.state", "==", "bound")]
        return OperationResult(
            preconditions=preconds,
            postconditions=[RefinementPredicate("self.state", "==", "listening")],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            state_transition=("bound", "listening"),
            exceptions=[("OSError", [])],
        )

    def model_accept(self) -> OperationResult:
        preconds = [RefinementPredicate("self.state", "==", "listening")]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.TUPLE),
            exceptions=[("OSError", []), ("BlockingIOError", [])],
        )

    def model_connect(self, address: Tuple[str, int]) -> OperationResult:
        preconds = [RefinementPredicate("self.state", "in", ["created", "bound"])]
        return OperationResult(
            preconditions=preconds,
            postconditions=[RefinementPredicate("self.state", "==", "connected")],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            state_transition=("created", "connected"),
            exceptions=[
                ("ConnectionRefusedError", []),
                ("TimeoutError", []),
                ("OSError", []),
            ],
        )

    def model_send(self, data: bytes, flags: int = 0) -> OperationResult:
        preconds = [RefinementPredicate("self.state", "==", "connected")]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(
                base=RefinementSort.INT,
                predicates=[
                    RefinementPredicate("result", ">=", 0),
                    RefinementPredicate("result", "<=", None, symbolic_operand="len(data)"),
                ],
            ),
            exceptions=[
                ("BrokenPipeError", []),
                ("ConnectionResetError", []),
                ("OSError", []),
            ],
        )

    def model_recv(self, bufsize: int, flags: int = 0) -> OperationResult:
        preconds = [RefinementPredicate("self.state", "==", "connected")]
        preds = [
            RefinementPredicate("len(result)", ">=", 0),
            RefinementPredicate("len(result)", "<=", bufsize),
        ]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.BYTES, predicates=preds),
            exceptions=[
                ("ConnectionResetError", []),
                ("TimeoutError", []),
                ("OSError", []),
            ],
        )

    def model_sendall(self, data: bytes) -> OperationResult:
        preconds = [RefinementPredicate("self.state", "==", "connected")]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.NONE),
            exceptions=[
                ("BrokenPipeError", []),
                ("ConnectionResetError", []),
                ("OSError", []),
            ],
        )

    def model_close(self) -> OperationResult:
        return OperationResult(
            postconditions=[RefinementPredicate("self.state", "==", "closed")],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            state_transition=("connected", "closed"),
        )

    def model_shutdown(self, how: int) -> OperationResult:
        return OperationResult(
            preconditions=self._socket_open_precond(),
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            exceptions=[("OSError", [])],
        )

    def model_setsockopt(self, level: int, optname: int, value: Any) -> OperationResult:
        return OperationResult(
            preconditions=self._socket_open_precond(),
            return_type=RefinementType(base=RefinementSort.NONE),
            exceptions=[("OSError", [])],
        )

    def model_getsockopt(self, level: int, optname: int) -> OperationResult:
        return OperationResult(
            preconditions=self._socket_open_precond(),
            return_type=RefinementType(base=RefinementSort.INT),
            exceptions=[("OSError", [])],
        )

    def model_settimeout(self, timeout: Optional[float]) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.NONE),
        )

    def model_gettimeout(self) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.FLOAT, nullable=True),
        )

    def model_getpeername(self) -> OperationResult:
        preconds = [RefinementPredicate("self.state", "==", "connected")]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.TUPLE),
            exceptions=[("OSError", [RefinementPredicate("self.state", "!=", "connected")])],
        )

    def model_getsockname(self) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.TUPLE),
            exceptions=[("OSError", [])],
        )

    # -- context manager -----------------------------------------------------

    def model_enter(self) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.SOCKET))

    def model_exit(self) -> OperationResult:
        return OperationResult(
            postconditions=[RefinementPredicate("self.state", "==", "closed")],
            return_type=RefinementType(base=RefinementSort.BOOL),
            modifies_receiver=True,
        )

    # -- urllib / http.client helpers -----------------------------------------

    @staticmethod
    def model_urlopen(url: str, timeout: Optional[float] = None) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.STREAM),
            exceptions=[
                ("URLError", []),
                ("HTTPError", []),
                ("TimeoutError", []),
            ],
        )

    @staticmethod
    def model_http_request(method: str, url: str) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.NONE),
            exceptions=[
                ("HTTPException", []),
                ("ConnectionError", []),
            ],
        )

    @staticmethod
    def model_http_getresponse() -> OperationResult:
        preds = [
            RefinementPredicate("result.status", ">=", 100),
            RefinementPredicate("result.status", "<=", 599),
        ]
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.OBJECT, predicates=preds),
            exceptions=[("HTTPException", [])],
        )


# ====================================================================
# SubprocessIOModel
# ====================================================================

@dataclass
class SubprocessIOModel:
    """
    Refinement model for ``subprocess`` module.
    """
    returncode: Optional[int] = None
    stdin_pipe: bool = False
    stdout_pipe: bool = False
    stderr_pipe: bool = False

    @staticmethod
    def model_run(
        args: Any,
        capture_output: bool = False,
        check: bool = False,
        timeout: Optional[float] = None,
    ) -> OperationResult:
        preds = [RefinementPredicate("result.returncode", ">=", -128)]
        if check:
            preds.append(RefinementPredicate("result.returncode", "==", 0))
        excs: List[Tuple[str, List[RefinementPredicate]]] = [
            ("FileNotFoundError", [RefinementPredicate("executable_exists", "==", False)]),
            ("PermissionError", []),
            ("OSError", []),
        ]
        if check:
            excs.append(("CalledProcessError", [RefinementPredicate("result.returncode", "!=", 0)]))
        if timeout is not None:
            excs.append(("TimeoutExpired", []))
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.OBJECT, predicates=preds),
            exceptions=excs,
        )

    @staticmethod
    def model_popen(
        args: Any,
        stdin: Optional[int] = None,
        stdout: Optional[int] = None,
        stderr: Optional[int] = None,
    ) -> OperationResult:
        preds = [RefinementPredicate("result.pid", ">", 0)]
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.PROCESS, predicates=preds),
            exceptions=[
                ("FileNotFoundError", []),
                ("PermissionError", []),
                ("OSError", []),
            ],
        )

    @staticmethod
    def model_popen_communicate(input_data: Optional[bytes] = None, timeout: Optional[float] = None) -> OperationResult:
        excs: List[Tuple[str, List[RefinementPredicate]]] = []
        if timeout is not None:
            excs.append(("TimeoutExpired", []))
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.TUPLE),
            exceptions=excs,
        )

    @staticmethod
    def model_popen_wait(timeout: Optional[float] = None) -> OperationResult:
        excs: List[Tuple[str, List[RefinementPredicate]]] = []
        if timeout is not None:
            excs.append(("TimeoutExpired", []))
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.INT),
            exceptions=excs,
        )

    @staticmethod
    def model_popen_poll() -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.INT, nullable=True),
        )

    @staticmethod
    def model_popen_kill() -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.NONE),
            exceptions=[("OSError", [])],
        )

    @staticmethod
    def model_popen_terminate() -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.NONE),
            exceptions=[("OSError", [])],
        )

    @staticmethod
    def model_popen_send_signal(signal: int) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.NONE),
            exceptions=[("OSError", []), ("ProcessLookupError", [])],
        )

    @staticmethod
    def model_check_output(args: Any, timeout: Optional[float] = None) -> OperationResult:
        excs: List[Tuple[str, List[RefinementPredicate]]] = [
            ("CalledProcessError", [RefinementPredicate("returncode", "!=", 0)]),
            ("FileNotFoundError", []),
        ]
        if timeout is not None:
            excs.append(("TimeoutExpired", []))
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.BYTES),
            exceptions=excs,
        )

    @staticmethod
    def model_check_call(args: Any, timeout: Optional[float] = None) -> OperationResult:
        excs: List[Tuple[str, List[RefinementPredicate]]] = [
            ("CalledProcessError", [RefinementPredicate("returncode", "!=", 0)]),
            ("FileNotFoundError", []),
        ]
        if timeout is not None:
            excs.append(("TimeoutExpired", []))
        return OperationResult(
            return_type=RefinementType(
                base=RefinementSort.INT,
                predicates=[RefinementPredicate("result", "==", 0)],
            ),
            exceptions=excs,
        )


# ====================================================================
# SerializationIOModel
# ====================================================================

@dataclass
class SerializationIOModel:
    """
    Refinement model for serialization modules: pickle, shelve, csv, xml.etree.
    """

    # -- pickle --------------------------------------------------------------

    @staticmethod
    def model_pickle_dump(obj: Any, file_model: Optional[FileModel] = None) -> OperationResult:
        preconds: List[RefinementPredicate] = []
        if file_model is not None:
            preconds.append(RefinementPredicate("file.state", "==", "open"))
            preconds.append(RefinementPredicate("file.writable()", "==", True))
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.NONE),
            exceptions=[
                ("PicklingError", []),
                ("TypeError", []),
            ],
        )

    @staticmethod
    def model_pickle_dumps(obj: Any) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.BYTES),
            exceptions=[("PicklingError", []), ("TypeError", [])],
        )

    @staticmethod
    def model_pickle_load(file_model: Optional[FileModel] = None) -> OperationResult:
        preconds: List[RefinementPredicate] = []
        if file_model is not None:
            preconds.append(RefinementPredicate("file.state", "==", "open"))
            preconds.append(RefinementPredicate("file.readable()", "==", True))
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.ANY),
            exceptions=[
                ("UnpicklingError", []),
                ("EOFError", []),
                ("ModuleNotFoundError", []),
            ],
        )

    @staticmethod
    def model_pickle_loads(data: bytes) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.ANY),
            exceptions=[("UnpicklingError", []), ("EOFError", [])],
        )

    # -- shelve --------------------------------------------------------------

    @staticmethod
    def model_shelve_open(filename: str) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.OBJECT),
            exceptions=[("OSError", []), ("dbm.error", [])],
        )

    # -- csv -----------------------------------------------------------------

    @staticmethod
    def model_csv_reader(file_model: Optional[FileModel] = None) -> OperationResult:
        preconds: List[RefinementPredicate] = []
        if file_model is not None:
            preconds.append(RefinementPredicate("file.state", "==", "open"))
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.ITERATOR),
        )

    @staticmethod
    def model_csv_writer(file_model: Optional[FileModel] = None) -> OperationResult:
        preconds: List[RefinementPredicate] = []
        if file_model is not None:
            preconds.append(RefinementPredicate("file.state", "==", "open"))
            preconds.append(RefinementPredicate("file.writable()", "==", True))
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.OBJECT),
        )

    @staticmethod
    def model_csv_writerow(row: Any) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.NONE),
            exceptions=[("Error", [])],
        )

    @staticmethod
    def model_csv_writerows(rows: Any) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.NONE),
            exceptions=[("Error", [])],
        )

    @staticmethod
    def model_csv_dictreader(file_model: Optional[FileModel] = None) -> OperationResult:
        preconds: List[RefinementPredicate] = []
        if file_model is not None:
            preconds.append(RefinementPredicate("file.state", "==", "open"))
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.ITERATOR),
        )

    @staticmethod
    def model_csv_dictwriter(file_model: Optional[FileModel] = None, fieldnames: Optional[List[str]] = None) -> OperationResult:
        preconds: List[RefinementPredicate] = []
        if file_model is not None:
            preconds.append(RefinementPredicate("file.state", "==", "open"))
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.OBJECT),
        )

    # -- xml.etree -----------------------------------------------------------

    @staticmethod
    def model_etree_parse(source: str) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.OBJECT),
            exceptions=[
                ("ParseError", []),
                ("FileNotFoundError", []),
            ],
        )

    @staticmethod
    def model_etree_fromstring(text: str) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.OBJECT),
            exceptions=[("ParseError", [])],
        )

    @staticmethod
    def model_etree_tostring(element: Any, encoding: Optional[str] = None) -> OperationResult:
        base = RefinementSort.BYTES if encoding is None else RefinementSort.STR
        return OperationResult(return_type=RefinementType(base=base))

    @staticmethod
    def model_etree_find(element: Any, path: str) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.OBJECT, nullable=True))

    @staticmethod
    def model_etree_findall(element: Any, path: str) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(
                base=RefinementSort.LIST,
                predicates=[RefinementPredicate("len(result)", ">=", 0)],
            ),
        )


# ====================================================================
# DatabaseIOModel
# ====================================================================

class CursorState(enum.Enum):
    CREATED = "created"
    EXECUTED = "executed"
    FETCHING = "fetching"
    CLOSED = "closed"


class ConnectionState(enum.Enum):
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    IN_TRANSACTION = "in_transaction"
    CLOSED = "closed"


@dataclass
class DatabaseIOModel:
    """
    Refinement model for ``sqlite3`` connection and cursor.
    """
    conn_state: ConnectionState = ConnectionState.DISCONNECTED
    cursor_state: CursorState = CursorState.CREATED

    # -- connection ----------------------------------------------------------

    @staticmethod
    def model_connect(database: str) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(
                base=RefinementSort.CONNECTION,
                predicates=[RefinementPredicate("result.state", "==", "connected")],
            ),
            exceptions=[
                ("OperationalError", []),
                ("DatabaseError", []),
            ],
        )

    def model_conn_close(self) -> OperationResult:
        return OperationResult(
            postconditions=[RefinementPredicate("self.state", "==", "closed")],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            state_transition=("connected", "closed"),
        )

    def model_conn_commit(self) -> OperationResult:
        preconds = [RefinementPredicate("self.state", "in", ["connected", "in_transaction"])]
        return OperationResult(
            preconditions=preconds,
            postconditions=[RefinementPredicate("self.state", "==", "connected")],
            return_type=RefinementType(base=RefinementSort.NONE),
            exceptions=[("OperationalError", [])],
        )

    def model_conn_rollback(self) -> OperationResult:
        preconds = [RefinementPredicate("self.state", "in", ["connected", "in_transaction"])]
        return OperationResult(
            preconditions=preconds,
            postconditions=[RefinementPredicate("self.state", "==", "connected")],
            return_type=RefinementType(base=RefinementSort.NONE),
            exceptions=[("OperationalError", [])],
        )

    def model_conn_cursor(self) -> OperationResult:
        preconds = [RefinementPredicate("self.state", "!=", "closed")]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(
                base=RefinementSort.CURSOR,
                predicates=[RefinementPredicate("result.state", "==", "created")],
            ),
        )

    def model_conn_execute(self, sql: str, parameters: Optional[Any] = None) -> OperationResult:
        preconds = [RefinementPredicate("self.state", "!=", "closed")]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.CURSOR),
            exceptions=[
                ("OperationalError", []),
                ("IntegrityError", []),
                ("ProgrammingError", []),
            ],
        )

    def model_conn_executemany(self, sql: str, seq_of_params: Any) -> OperationResult:
        preconds = [RefinementPredicate("self.state", "!=", "closed")]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.CURSOR),
            exceptions=[
                ("OperationalError", []),
                ("IntegrityError", []),
                ("ProgrammingError", []),
            ],
        )

    def model_conn_executescript(self, sql_script: str) -> OperationResult:
        preconds = [RefinementPredicate("self.state", "!=", "closed")]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.CURSOR),
            exceptions=[
                ("OperationalError", []),
                ("ProgrammingError", []),
            ],
        )

    # -- context manager -----------------------------------------------------

    def model_conn_enter(self) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.CONNECTION),
        )

    def model_conn_exit(self, exc_type: Any = None) -> OperationResult:
        posts: List[RefinementPredicate] = []
        if exc_type is None:
            posts.append(RefinementPredicate("committed", "==", True))
        else:
            posts.append(RefinementPredicate("rolled_back", "==", True))
        return OperationResult(
            postconditions=posts,
            return_type=RefinementType(base=RefinementSort.BOOL),
        )

    # -- cursor operations ---------------------------------------------------

    def model_cursor_execute(self, sql: str, parameters: Optional[Any] = None) -> OperationResult:
        preconds = [RefinementPredicate("cursor.state", "!=", "closed")]
        return OperationResult(
            preconditions=preconds,
            postconditions=[RefinementPredicate("cursor.state", "==", "executed")],
            return_type=RefinementType(base=RefinementSort.CURSOR),
            modifies_receiver=True,
            state_transition=("created", "executed"),
            exceptions=[
                ("OperationalError", []),
                ("IntegrityError", []),
                ("ProgrammingError", []),
            ],
        )

    def model_cursor_executemany(self, sql: str, seq_of_params: Any) -> OperationResult:
        preconds = [RefinementPredicate("cursor.state", "!=", "closed")]
        return OperationResult(
            preconditions=preconds,
            postconditions=[RefinementPredicate("cursor.state", "==", "executed")],
            return_type=RefinementType(base=RefinementSort.CURSOR),
            modifies_receiver=True,
            exceptions=[("OperationalError", []), ("IntegrityError", [])],
        )

    def model_cursor_fetchone(self) -> OperationResult:
        preconds = [RefinementPredicate("cursor.state", "==", "executed")]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.TUPLE, nullable=True),
            state_transition=("executed", "fetching"),
        )

    def model_cursor_fetchmany(self, size: Optional[int] = None) -> OperationResult:
        preconds = [RefinementPredicate("cursor.state", "==", "executed")]
        preds = [RefinementPredicate("len(result)", ">=", 0)]
        if size is not None:
            preds.append(RefinementPredicate("len(result)", "<=", size))
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.LIST, predicates=preds),
        )

    def model_cursor_fetchall(self) -> OperationResult:
        preconds = [RefinementPredicate("cursor.state", "==", "executed")]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(
                base=RefinementSort.LIST,
                predicates=[RefinementPredicate("len(result)", ">=", 0)],
            ),
        )

    def model_cursor_close(self) -> OperationResult:
        return OperationResult(
            postconditions=[RefinementPredicate("cursor.state", "==", "closed")],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            state_transition=("executed", "closed"),
        )

    def model_cursor_description(self) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.TUPLE, nullable=True),
        )

    def model_cursor_rowcount(self) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(
                base=RefinementSort.INT,
                predicates=[RefinementPredicate("result", ">=", -1)],
            ),
        )

    def model_cursor_lastrowid(self) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.INT, nullable=True),
        )

    def model_cursor_iter(self) -> OperationResult:
        preconds = [RefinementPredicate("cursor.state", "==", "executed")]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.ITERATOR),
        )


# ====================================================================
# TempFileModel
# ====================================================================

@dataclass
class TempFileModel:
    """
    Refinement model for ``tempfile`` module.
    """

    @staticmethod
    def model_named_temporary_file(
        mode: str = "w+b",
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
        dir: Optional[str] = None,
        delete: bool = True,
    ) -> OperationResult:
        preds = [
            RefinementPredicate("result.name", "!=", None),
            RefinementPredicate("result.state", "==", "open"),
        ]
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.FILE, predicates=preds),
            exceptions=[("OSError", []), ("PermissionError", [])],
        )

    @staticmethod
    def model_temporary_file(
        mode: str = "w+b",
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
        dir: Optional[str] = None,
    ) -> OperationResult:
        preds = [RefinementPredicate("result.state", "==", "open")]
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.FILE, predicates=preds),
            exceptions=[("OSError", [])],
        )

    @staticmethod
    def model_mkstemp(
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
        dir: Optional[str] = None,
    ) -> OperationResult:
        """Returns (fd, name) tuple."""
        preds = [
            RefinementPredicate("result[0]", ">=", 0),
            RefinementPredicate("len(result[1])", ">", 0),
        ]
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.TUPLE, predicates=preds),
            exceptions=[("OSError", []), ("PermissionError", [])],
        )

    @staticmethod
    def model_mkdtemp(
        suffix: Optional[str] = None,
        prefix: Optional[str] = None,
        dir: Optional[str] = None,
    ) -> OperationResult:
        preds = [RefinementPredicate("len(result)", ">", 0)]
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.STR, predicates=preds),
            exceptions=[("OSError", []), ("PermissionError", [])],
        )

    @staticmethod
    def model_gettempdir() -> OperationResult:
        preds = [RefinementPredicate("len(result)", ">", 0)]
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.STR, predicates=preds),
        )

    @staticmethod
    def model_temporary_directory() -> OperationResult:
        """tempfile.TemporaryDirectory context manager."""
        preds = [RefinementPredicate("len(result.name)", ">", 0)]
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.OBJECT, predicates=preds),
            exceptions=[("OSError", [])],
        )

    @staticmethod
    def model_spooled_temporary_file(max_size: int = 0) -> OperationResult:
        preds = [RefinementPredicate("result.state", "==", "open")]
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.FILE, predicates=preds),
            exceptions=[("OSError", [])],
        )


# ====================================================================
# LoggingModel
# ====================================================================

class LogLevel(enum.Enum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    NOTSET = 0


@dataclass
class LoggingModel:
    """
    Refinement model for the ``logging`` module.
    """
    level: LogLevel = LogLevel.WARNING

    @staticmethod
    def model_getLogger(name: Optional[str] = None) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.OBJECT),
        )

    def model_debug(self, msg: str, *args: Any) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.NONE),
        )

    def model_info(self, msg: str, *args: Any) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.NONE),
        )

    def model_warning(self, msg: str, *args: Any) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.NONE),
        )

    def model_error(self, msg: str, *args: Any) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.NONE),
        )

    def model_critical(self, msg: str, *args: Any) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.NONE),
        )

    def model_exception(self, msg: str, *args: Any) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.NONE),
        )

    @staticmethod
    def model_basicConfig(**kwargs: Any) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.NONE),
        )

    def model_setLevel(self, level: int) -> OperationResult:
        return OperationResult(
            postconditions=[RefinementPredicate("self.level", "==", level)],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
        )

    def model_addHandler(self, handler: Any) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
        )

    def model_removeHandler(self, handler: Any) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
        )

    @staticmethod
    def model_FileHandler(filename: str, mode: str = "a") -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.OBJECT),
            exceptions=[("OSError", []), ("PermissionError", [])],
        )

    @staticmethod
    def model_StreamHandler(stream: Optional[Any] = None) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.OBJECT),
        )

    @staticmethod
    def model_RotatingFileHandler(
        filename: str,
        maxBytes: int = 0,
        backupCount: int = 0,
    ) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.OBJECT),
            exceptions=[("OSError", [])],
        )

    @staticmethod
    def model_Formatter(fmt: Optional[str] = None, datefmt: Optional[str] = None) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.OBJECT),
        )

    def model_isEnabledFor(self, level: int) -> OperationResult:
        preds: List[RefinementPredicate] = []
        if self.level.value <= level:
            preds.append(RefinementPredicate("result", "==", True))
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.BOOL, predicates=preds),
        )


# ====================================================================
# OSIOModel
# ====================================================================

@dataclass
class OSIOModel:
    """
    Refinement model for ``os`` low-level I/O functions.
    """

    @staticmethod
    def model_os_open(path: str, flags: int, mode: int = 0o777) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(
                base=RefinementSort.INT,
                predicates=[RefinementPredicate("result", ">=", 0)],
            ),
            exceptions=[
                ("FileNotFoundError", []),
                ("PermissionError", []),
                ("OSError", []),
            ],
        )

    @staticmethod
    def model_os_read(fd: int, n: int) -> OperationResult:
        preconds = [
            RefinementPredicate("fd", ">=", 0),
            RefinementPredicate("n", ">=", 0),
        ]
        preds = [
            RefinementPredicate("len(result)", ">=", 0),
            RefinementPredicate("len(result)", "<=", n),
        ]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.BYTES, predicates=preds),
            exceptions=[("OSError", [])],
        )

    @staticmethod
    def model_os_write(fd: int, data: bytes) -> OperationResult:
        preconds = [RefinementPredicate("fd", ">=", 0)]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(
                base=RefinementSort.INT,
                predicates=[
                    RefinementPredicate("result", ">=", 0),
                    RefinementPredicate("result", "<=", None, symbolic_operand="len(data)"),
                ],
            ),
            exceptions=[("OSError", [])],
        )

    @staticmethod
    def model_os_close(fd: int) -> OperationResult:
        preconds = [RefinementPredicate("fd", ">=", 0)]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.NONE),
            exceptions=[("OSError", [])],
        )

    @staticmethod
    def model_os_pipe() -> OperationResult:
        return OperationResult(
            return_type=RefinementType(
                base=RefinementSort.TUPLE,
                predicates=[
                    RefinementPredicate("result[0]", ">=", 0),
                    RefinementPredicate("result[1]", ">=", 0),
                ],
            ),
            exceptions=[("OSError", [])],
        )

    @staticmethod
    def model_os_dup(fd: int) -> OperationResult:
        preconds = [RefinementPredicate("fd", ">=", 0)]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(
                base=RefinementSort.INT,
                predicates=[RefinementPredicate("result", ">=", 0)],
            ),
            exceptions=[("OSError", [])],
        )

    @staticmethod
    def model_os_dup2(fd: int, fd2: int) -> OperationResult:
        preconds = [
            RefinementPredicate("fd", ">=", 0),
            RefinementPredicate("fd2", ">=", 0),
        ]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(
                base=RefinementSort.INT,
                predicates=[RefinementPredicate("result", "==", None, symbolic_operand="fd2")],
            ),
            exceptions=[("OSError", [])],
        )

    @staticmethod
    def model_os_lseek(fd: int, pos: int, how: int) -> OperationResult:
        preconds = [RefinementPredicate("fd", ">=", 0)]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(
                base=RefinementSort.INT,
                predicates=[RefinementPredicate("result", ">=", 0)],
            ),
            exceptions=[("OSError", [])],
        )

    @staticmethod
    def model_os_fstat(fd: int) -> OperationResult:
        preconds = [RefinementPredicate("fd", ">=", 0)]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.OBJECT),
            exceptions=[("OSError", [])],
        )

    @staticmethod
    def model_os_ftruncate(fd: int, length: int) -> OperationResult:
        preconds = [
            RefinementPredicate("fd", ">=", 0),
            RefinementPredicate("length", ">=", 0),
        ]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.NONE),
            exceptions=[("OSError", [])],
        )

    @staticmethod
    def model_os_fdopen(fd: int, mode: str = "r") -> OperationResult:
        preconds = [RefinementPredicate("fd", ">=", 0)]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.FILE),
            exceptions=[("OSError", [])],
        )

    @staticmethod
    def model_os_isatty(fd: int) -> OperationResult:
        preconds = [RefinementPredicate("fd", ">=", 0)]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.BOOL),
            exceptions=[("OSError", [])],
        )


# ====================================================================
# CompressIOModel
# ====================================================================

@dataclass
class CompressIOModel:
    """
    Refinement model for compression modules: gzip, bz2, lzma, zipfile, tarfile.
    """

    # -- gzip ----------------------------------------------------------------

    @staticmethod
    def model_gzip_open(filename: str, mode: str = "rb") -> OperationResult:
        preds = [RefinementPredicate("result.state", "==", "open")]
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.FILE, predicates=preds),
            exceptions=[("OSError", []), ("gzip.BadGzipFile", [])],
        )

    @staticmethod
    def model_gzip_compress(data: bytes, compresslevel: int = 9) -> OperationResult:
        preconds = [
            RefinementPredicate("compresslevel", ">=", 0),
            RefinementPredicate("compresslevel", "<=", 9),
        ]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.BYTES, predicates=[RefinementPredicate("len(result)", ">=", 0)]),
        )

    @staticmethod
    def model_gzip_decompress(data: bytes) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.BYTES),
            exceptions=[("gzip.BadGzipFile", []), ("OSError", [])],
        )

    # -- bz2 -----------------------------------------------------------------

    @staticmethod
    def model_bz2_open(filename: str, mode: str = "rb") -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.FILE),
            exceptions=[("OSError", []), ("ValueError", [])],
        )

    @staticmethod
    def model_bz2_compress(data: bytes) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.BYTES),
        )

    @staticmethod
    def model_bz2_decompress(data: bytes) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.BYTES),
            exceptions=[("OSError", []), ("ValueError", [])],
        )

    # -- lzma ----------------------------------------------------------------

    @staticmethod
    def model_lzma_open(filename: str, mode: str = "rb") -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.FILE),
            exceptions=[("OSError", []), ("LZMAError", [])],
        )

    @staticmethod
    def model_lzma_compress(data: bytes) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.BYTES),
        )

    @staticmethod
    def model_lzma_decompress(data: bytes) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.BYTES),
            exceptions=[("LZMAError", [])],
        )

    # -- zipfile -------------------------------------------------------------

    @staticmethod
    def model_zipfile_open(file: str, mode: str = "r") -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.OBJECT),
            exceptions=[
                ("BadZipFile", []),
                ("FileNotFoundError", []),
                ("PermissionError", []),
            ],
        )

    @staticmethod
    def model_zipfile_read(name: str) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.BYTES),
            exceptions=[("KeyError", [RefinementPredicate("name", "not_in", None, symbolic_operand="self.namelist()")])],
        )

    @staticmethod
    def model_zipfile_write(filename: str, arcname: Optional[str] = None) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            exceptions=[("ValueError", []), ("OSError", [])],
        )

    @staticmethod
    def model_zipfile_extractall(path: Optional[str] = None) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.NONE),
            exceptions=[("BadZipFile", []), ("OSError", [])],
        )

    @staticmethod
    def model_zipfile_namelist() -> OperationResult:
        return OperationResult(
            return_type=RefinementType(
                base=RefinementSort.LIST,
                predicates=[RefinementPredicate("len(result)", ">=", 0)],
            ),
        )

    @staticmethod
    def model_zipfile_close() -> OperationResult:
        return OperationResult(
            postconditions=[RefinementPredicate("self.state", "==", "closed")],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
        )

    # -- tarfile -------------------------------------------------------------

    @staticmethod
    def model_tarfile_open(name: Optional[str] = None, mode: str = "r") -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.OBJECT),
            exceptions=[
                ("TarError", []),
                ("FileNotFoundError", []),
                ("PermissionError", []),
            ],
        )

    @staticmethod
    def model_tarfile_extractall(path: str = ".") -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.NONE),
            exceptions=[("TarError", []), ("OSError", [])],
        )

    @staticmethod
    def model_tarfile_add(name: str, arcname: Optional[str] = None) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
            exceptions=[("OSError", [])],
        )

    @staticmethod
    def model_tarfile_getnames() -> OperationResult:
        return OperationResult(
            return_type=RefinementType(
                base=RefinementSort.LIST,
                predicates=[RefinementPredicate("len(result)", ">=", 0)],
            ),
        )

    @staticmethod
    def model_tarfile_getmembers() -> OperationResult:
        return OperationResult(
            return_type=RefinementType(
                base=RefinementSort.LIST,
                predicates=[RefinementPredicate("len(result)", ">=", 0)],
            ),
        )

    @staticmethod
    def model_tarfile_close() -> OperationResult:
        return OperationResult(
            postconditions=[RefinementPredicate("self.state", "==", "closed")],
            return_type=RefinementType(base=RefinementSort.NONE),
            modifies_receiver=True,
        )


# ====================================================================
# ConfigIOModel
# ====================================================================

@dataclass
class ConfigIOModel:
    """
    Refinement model for configuration/serialization modules:
    configparser, json, toml, yaml.
    """

    # -- json ----------------------------------------------------------------

    @staticmethod
    def model_json_dumps(obj: Any, indent: Optional[int] = None) -> OperationResult:
        preds = [RefinementPredicate("len(result)", ">", 0)]
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.STR, predicates=preds),
            exceptions=[
                ("TypeError", []),
                ("ValueError", []),
            ],
        )

    @staticmethod
    def model_json_loads(s: str) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.ANY),
            exceptions=[("JSONDecodeError", [])],
        )

    @staticmethod
    def model_json_dump(obj: Any, fp: Any) -> OperationResult:
        preconds = [
            RefinementPredicate("fp.state", "==", "open"),
            RefinementPredicate("fp.writable()", "==", True),
        ]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.NONE),
            exceptions=[("TypeError", []), ("ValueError", [])],
        )

    @staticmethod
    def model_json_load(fp: Any) -> OperationResult:
        preconds = [
            RefinementPredicate("fp.state", "==", "open"),
            RefinementPredicate("fp.readable()", "==", True),
        ]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.ANY),
            exceptions=[("JSONDecodeError", [])],
        )

    # -- configparser --------------------------------------------------------

    @staticmethod
    def model_configparser_init() -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.OBJECT),
        )

    @staticmethod
    def model_configparser_read(filenames: Any) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(
                base=RefinementSort.LIST,
                predicates=[RefinementPredicate("len(result)", ">=", 0)],
            ),
        )

    @staticmethod
    def model_configparser_read_string(string: str) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.NONE),
            exceptions=[("MissingSectionHeaderError", []), ("ParsingError", [])],
        )

    @staticmethod
    def model_configparser_get(section: str, option: str) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.STR),
            exceptions=[
                ("NoSectionError", [RefinementPredicate("section", "not_in", None, symbolic_operand="self.sections()")]),
                ("NoOptionError", []),
            ],
        )

    @staticmethod
    def model_configparser_getint(section: str, option: str) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.INT),
            exceptions=[
                ("NoSectionError", []),
                ("NoOptionError", []),
                ("ValueError", []),
            ],
        )

    @staticmethod
    def model_configparser_getfloat(section: str, option: str) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.FLOAT),
            exceptions=[
                ("NoSectionError", []),
                ("NoOptionError", []),
                ("ValueError", []),
            ],
        )

    @staticmethod
    def model_configparser_getboolean(section: str, option: str) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.BOOL),
            exceptions=[
                ("NoSectionError", []),
                ("NoOptionError", []),
                ("ValueError", []),
            ],
        )

    @staticmethod
    def model_configparser_sections() -> OperationResult:
        return OperationResult(
            return_type=RefinementType(
                base=RefinementSort.LIST,
                predicates=[RefinementPredicate("len(result)", ">=", 0)],
            ),
        )

    @staticmethod
    def model_configparser_has_section(section: str) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL))

    @staticmethod
    def model_configparser_has_option(section: str, option: str) -> OperationResult:
        return OperationResult(return_type=RefinementType(base=RefinementSort.BOOL))

    @staticmethod
    def model_configparser_write(fp: Any) -> OperationResult:
        preconds = [
            RefinementPredicate("fp.state", "==", "open"),
            RefinementPredicate("fp.writable()", "==", True),
        ]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.NONE),
        )

    # -- toml ----------------------------------------------------------------

    @staticmethod
    def model_toml_loads(s: str) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.DICT),
            exceptions=[("TOMLDecodeError", [])],
        )

    @staticmethod
    def model_toml_load(fp: Any) -> OperationResult:
        preconds = [RefinementPredicate("fp.state", "==", "open")]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.DICT),
            exceptions=[("TOMLDecodeError", [])],
        )

    @staticmethod
    def model_toml_dumps(data: dict) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.STR),
            exceptions=[("TypeError", [])],
        )

    @staticmethod
    def model_toml_dump(data: dict, fp: Any) -> OperationResult:
        preconds = [
            RefinementPredicate("fp.state", "==", "open"),
            RefinementPredicate("fp.writable()", "==", True),
        ]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(base=RefinementSort.STR),
            exceptions=[("TypeError", [])],
        )

    # -- yaml ----------------------------------------------------------------

    @staticmethod
    def model_yaml_safe_load(stream: Any) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.ANY),
            exceptions=[("YAMLError", [])],
        )

    @staticmethod
    def model_yaml_safe_dump(data: Any, stream: Optional[Any] = None) -> OperationResult:
        if stream is None:
            return OperationResult(return_type=RefinementType(base=RefinementSort.STR))
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.NONE),
            exceptions=[("YAMLError", [])],
        )

    @staticmethod
    def model_yaml_safe_load_all(stream: Any) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.ITERATOR),
            exceptions=[("YAMLError", [])],
        )

    @staticmethod
    def model_yaml_safe_dump_all(documents: Any, stream: Optional[Any] = None) -> OperationResult:
        if stream is None:
            return OperationResult(return_type=RefinementType(base=RefinementSort.STR))
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.NONE),
            exceptions=[("YAMLError", [])],
        )


# ====================================================================
# SignalModel
# ====================================================================

@dataclass
class SignalModel:
    """
    Refinement model for the ``signal`` module.
    """

    @staticmethod
    def model_signal(signum: int, handler: Any) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.CALLABLE, nullable=True),
            exceptions=[
                ("OSError", []),
                ("ValueError", [RefinementPredicate("signum", "<", 0)]),
            ],
        )

    @staticmethod
    def model_getsignal(signum: int) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.ANY, nullable=True),
            exceptions=[("ValueError", [RefinementPredicate("signum", "<", 0)])],
        )

    @staticmethod
    def model_raise_signal(signum: int) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.NONE),
            exceptions=[("OSError", [])],
        )

    @staticmethod
    def model_alarm(time: int) -> OperationResult:
        preconds = [RefinementPredicate("time", ">=", 0)]
        return OperationResult(
            preconditions=preconds,
            return_type=RefinementType(
                base=RefinementSort.INT,
                predicates=[RefinementPredicate("result", ">=", 0)],
            ),
        )

    @staticmethod
    def model_pause() -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.NONE),
        )

    @staticmethod
    def model_siginterrupt(signum: int, flag: bool) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.NONE),
            exceptions=[("OSError", []), ("ValueError", [])],
        )

    @staticmethod
    def model_set_wakeup_fd(fd: int) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(
                base=RefinementSort.INT,
                predicates=[RefinementPredicate("result", ">=", -1)],
            ),
            exceptions=[("ValueError", [])],
        )

    @staticmethod
    def model_sigwait(sigset: Any) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.INT),
        )

    @staticmethod
    def model_pthread_sigmask(how: int, mask: Any) -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.SET),
            exceptions=[("OSError", [])],
        )

    @staticmethod
    def model_valid_signals() -> OperationResult:
        return OperationResult(
            return_type=RefinementType(base=RefinementSort.SET),
        )


# ====================================================================
# IO model registry
# ====================================================================

IO_MODEL_REGISTRY: Dict[str, type] = {
    "IOState": IOStateModel,
    "File": FileModel,
    "Stream": StreamModel,
    "PathIO": PathIOModel,
    "NetworkIO": NetworkIOModel,
    "SubprocessIO": SubprocessIOModel,
    "SerializationIO": SerializationIOModel,
    "DatabaseIO": DatabaseIOModel,
    "TempFile": TempFileModel,
    "Logging": LoggingModel,
    "OSIO": OSIOModel,
    "CompressIO": CompressIOModel,
    "ConfigIO": ConfigIOModel,
    "Signal": SignalModel,
}


def get_io_model(name: str) -> Optional[type]:
    """Look up an I/O refinement model class by name."""
    return IO_MODEL_REGISTRY.get(name)
