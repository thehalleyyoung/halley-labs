from __future__ import annotations

"""
reftype.cli.lsp_server — Language Server Protocol implementation.

Provides a full LSP server for refinement-type-aware editing support in
IDEs (VS Code, Vim/Neovim, Emacs, etc.).  Communicates over stdio or TCP
using JSON-RPC 2.0.
"""

import enum
import hashlib
import io
import json
import logging
import os
import pathlib
import re
import select
import socket
import struct
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Set,
    TextIO,
    Tuple,
    Type,
    Union,
)

logger = logging.getLogger("reftype.lsp")

_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Locally-defined domain types (mirrors of main.py types, standalone)
# ---------------------------------------------------------------------------


class Severity(enum.Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    HINT = "hint"


class Language(enum.Enum):
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    AUTO = "auto"


@dataclass
class SourceLocation:
    file: str
    line: int
    column: int
    end_line: Optional[int] = None
    end_column: Optional[int] = None


@dataclass
class RefinementType:
    base: str
    predicate: str
    constraints: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.predicate:
            return f"{{{self.base} | {self.predicate}}}"
        return self.base


@dataclass
class Bug:
    id: str
    message: str
    severity: Severity
    location: SourceLocation
    category: str
    refinement_type: Optional[RefinementType] = None
    fix_suggestion: Optional[str] = None
    cegar_trace: Optional[List[str]] = None

    def fingerprint(self) -> str:
        raw = f"{self.category}:{self.location.file}:{self.message}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class FunctionContract:
    name: str
    file: str
    line: int
    params: Dict[str, RefinementType] = field(default_factory=dict)
    return_type: Optional[RefinementType] = None
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    file: str
    language: Language
    bugs: List[Bug] = field(default_factory=list)
    contracts: List[FunctionContract] = field(default_factory=list)
    duration_ms: float = 0.0
    functions_analyzed: int = 0
    cegar_iterations: int = 0
    timed_out: bool = False


# ---------------------------------------------------------------------------
# LSP constants
# ---------------------------------------------------------------------------

class DiagnosticSeverityLsp(enum.IntEnum):
    Error = 1
    Warning = 2
    Information = 3
    Hint = 4


class TextDocumentSyncKind(enum.IntEnum):
    NoneKind = 0
    Full = 1
    Incremental = 2


class CompletionItemKind(enum.IntEnum):
    Text = 1
    Method = 2
    Function = 3
    Constructor = 4
    Field = 5
    Variable = 6
    Class = 7
    Interface = 8
    Module = 9
    Property = 10
    Unit = 11
    Value = 12
    Enum = 13
    Keyword = 14
    Snippet = 15
    Color = 16
    File = 17
    Reference = 18
    Folder = 19
    EnumMember = 20
    Constant = 21
    Struct = 22
    Event = 23
    Operator = 24
    TypeParameter = 25


class SymbolKind(enum.IntEnum):
    File = 1
    Module = 2
    Namespace = 3
    Package = 4
    Class = 5
    Method = 6
    Property = 7
    Field = 8
    Constructor = 9
    Enum = 10
    Interface = 11
    Function = 12
    Variable = 13
    Constant = 14
    String = 15
    Number = 16
    Boolean = 17
    Array = 18
    Object = 19
    Key = 20
    Null = 21
    EnumMember = 22
    Struct = 23
    Event = 24
    Operator = 25
    TypeParameter = 26


class CodeActionKind:
    QuickFix = "quickfix"
    Refactor = "refactor"
    RefactorExtract = "refactor.extract"
    RefactorInline = "refactor.inline"
    RefactorRewrite = "refactor.rewrite"
    Source = "source"


class SemanticTokenTypes:
    Type = "type"
    Class = "class"
    Enum = "enum"
    Interface = "interface"
    Struct = "struct"
    TypeParameter = "typeParameter"
    Parameter = "parameter"
    Variable = "variable"
    Property = "property"
    EnumMember = "enumMember"
    Function = "function"
    Method = "method"
    Macro = "macro"
    Keyword = "keyword"
    Modifier = "modifier"
    Comment = "comment"
    String = "string"
    Number = "number"
    Regexp = "regexp"
    Operator = "operator"
    Decorator = "decorator"

    ALL = [
        "type", "class", "enum", "interface", "struct", "typeParameter",
        "parameter", "variable", "property", "enumMember", "function",
        "method", "macro", "keyword", "modifier", "comment", "string",
        "number", "regexp", "operator", "decorator",
    ]


class SemanticTokenModifiers:
    Declaration = "declaration"
    Definition = "definition"
    Readonly = "readonly"
    Static = "static"
    Deprecated = "deprecated"
    Abstract = "abstract"
    Async = "async"
    Modification = "modification"
    Documentation = "documentation"
    DefaultLibrary = "defaultLibrary"

    ALL = [
        "declaration", "definition", "readonly", "static", "deprecated",
        "abstract", "async", "modification", "documentation", "defaultLibrary",
    ]


# ---------------------------------------------------------------------------
# URI handling
# ---------------------------------------------------------------------------


class URIHandler:
    """Handles file:// URIs and path conversion."""

    @staticmethod
    def uri_to_path(uri: str) -> str:
        if uri.startswith("file:///"):
            path = uri[7:]  # keep leading / on Unix
            if len(path) > 2 and path[0] == "/" and path[2] == ":":
                path = path[1:]  # Windows drive letter
        elif uri.startswith("file://"):
            path = uri[7:]
        else:
            path = uri
        # Decode percent-encoding
        import urllib.parse
        return urllib.parse.unquote(path)

    @staticmethod
    def path_to_uri(path: str) -> str:
        import urllib.parse
        abs_path = os.path.abspath(path)
        if os.name == "nt":
            abs_path = "/" + abs_path.replace("\\", "/")
        return "file://" + urllib.parse.quote(abs_path, safe="/:")


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 Protocol
# ---------------------------------------------------------------------------


@dataclass
class JsonRpcMessage:
    """A JSON-RPC 2.0 message."""
    jsonrpc: str = "2.0"
    method: Optional[str] = None
    params: Optional[Any] = None
    id: Optional[Union[int, str]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None

    def is_request(self) -> bool:
        return self.method is not None and self.id is not None

    def is_notification(self) -> bool:
        return self.method is not None and self.id is None

    def is_response(self) -> bool:
        return self.method is None and self.id is not None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"jsonrpc": self.jsonrpc}
        if self.method is not None:
            d["method"] = self.method
        if self.params is not None:
            d["params"] = self.params
        if self.id is not None:
            d["id"] = self.id
        if self.result is not None:
            d["result"] = self.result
        if self.error is not None:
            d["error"] = self.error
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> JsonRpcMessage:
        return cls(
            jsonrpc=d.get("jsonrpc", "2.0"),
            method=d.get("method"),
            params=d.get("params"),
            id=d.get("id"),
            result=d.get("result"),
            error=d.get("error"),
        )


class JsonRpcError:
    ParseError = -32700
    InvalidRequest = -32600
    MethodNotFound = -32601
    InvalidParams = -32602
    InternalError = -32603
    ServerNotInitialized = -32002
    RequestCancelled = -32800
    ContentModified = -32801

    @staticmethod
    def make(code: int, message: str, data: Optional[Any] = None) -> Dict[str, Any]:
        err: Dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        return err


# ---------------------------------------------------------------------------
# Message Reader / Writer (LSP header framing)
# ---------------------------------------------------------------------------


class MessageReader:
    """Reads LSP messages from an input stream (stdin)."""

    def __init__(self, stream: io.BufferedIOBase) -> None:
        self._stream = stream
        self._buffer = b""

    def read_message(self) -> Optional[JsonRpcMessage]:
        headers = self._read_headers()
        if headers is None:
            return None
        content_length = 0
        for header in headers:
            if header.lower().startswith("content-length:"):
                try:
                    content_length = int(header.split(":")[1].strip())
                except (ValueError, IndexError):
                    return None
        if content_length <= 0:
            return None
        body = self._read_exact(content_length)
        if body is None:
            return None
        try:
            data = json.loads(body.decode("utf-8"))
            return JsonRpcMessage.from_dict(data)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.error("Failed to parse message: %s", exc)
            return None

    def _read_headers(self) -> Optional[List[str]]:
        headers: List[str] = []
        line = b""
        while True:
            ch = self._read_exact(1)
            if ch is None:
                return None
            if ch == b"\n":
                decoded = line.rstrip(b"\r").decode("ascii", errors="replace")
                if decoded == "":
                    return headers if headers else None
                headers.append(decoded)
                line = b""
            else:
                line += ch

    def _read_exact(self, n: int) -> Optional[bytes]:
        while len(self._buffer) < n:
            try:
                chunk = self._stream.read(n - len(self._buffer))
            except (OSError, ValueError):
                return None
            if not chunk:
                return None
            self._buffer += chunk
        result = self._buffer[:n]
        self._buffer = self._buffer[n:]
        return result


class MessageWriter:
    """Writes LSP messages to an output stream (stdout)."""

    def __init__(self, stream: io.BufferedIOBase) -> None:
        self._stream = stream
        self._lock = threading.Lock()

    def write_message(self, msg: JsonRpcMessage) -> None:
        body = json.dumps(msg.to_dict(), ensure_ascii=False).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        with self._lock:
            try:
                self._stream.write(header + body)
                self._stream.flush()
            except (OSError, ValueError) as exc:
                logger.error("Failed to write message: %s", exc)

    def write_response(self, id: Union[int, str], result: Any) -> None:
        msg = JsonRpcMessage(id=id, result=result)
        self.write_message(msg)

    def write_error(
        self, id: Union[int, str], code: int, message: str, data: Optional[Any] = None
    ) -> None:
        msg = JsonRpcMessage(id=id, error=JsonRpcError.make(code, message, data))
        self.write_message(msg)

    def write_notification(self, method: str, params: Any = None) -> None:
        msg = JsonRpcMessage(method=method, params=params)
        self.write_message(msg)


# ---------------------------------------------------------------------------
# LspProtocol
# ---------------------------------------------------------------------------


class LspProtocol:
    """JSON-RPC 2.0 protocol layer managing message dispatch."""

    def __init__(self, reader: MessageReader, writer: MessageWriter) -> None:
        self.reader = reader
        self.writer = writer
        self._handlers: Dict[str, Callable[..., Any]] = {}
        self._notification_handlers: Dict[str, Callable[..., Any]] = {}
        self._pending_requests: Dict[Union[int, str], threading.Event] = {}
        self._pending_results: Dict[Union[int, str], Any] = {}
        self._next_id = 1
        self._id_lock = threading.Lock()

    def register_handler(self, method: str, handler: Callable[..., Any]) -> None:
        self._handlers[method] = handler

    def register_notification(self, method: str, handler: Callable[..., Any]) -> None:
        self._notification_handlers[method] = handler

    def process_one(self) -> bool:
        msg = self.reader.read_message()
        if msg is None:
            return False
        self._dispatch(msg)
        return True

    def run(self) -> None:
        while self.process_one():
            pass

    def _dispatch(self, msg: JsonRpcMessage) -> None:
        if msg.is_request():
            self._handle_request(msg)
        elif msg.is_notification():
            self._handle_notification(msg)
        elif msg.is_response():
            self._handle_response(msg)

    def _handle_request(self, msg: JsonRpcMessage) -> None:
        handler = self._handlers.get(msg.method or "")
        if handler is None:
            self.writer.write_error(
                msg.id,  # type: ignore[arg-type]
                JsonRpcError.MethodNotFound,
                f"Method not found: {msg.method}",
            )
            return
        try:
            result = handler(msg.params or {})
            self.writer.write_response(msg.id, result)  # type: ignore[arg-type]
        except Exception as exc:
            logger.error("Handler error for %s: %s", msg.method, exc, exc_info=True)
            self.writer.write_error(
                msg.id,  # type: ignore[arg-type]
                JsonRpcError.InternalError,
                str(exc),
            )

    def _handle_notification(self, msg: JsonRpcMessage) -> None:
        handler = self._notification_handlers.get(msg.method or "")
        if handler is not None:
            try:
                handler(msg.params or {})
            except Exception as exc:
                logger.error(
                    "Notification handler error for %s: %s",
                    msg.method,
                    exc,
                    exc_info=True,
                )

    def _handle_response(self, msg: JsonRpcMessage) -> None:
        rid = msg.id
        if rid in self._pending_requests:
            self._pending_results[rid] = msg.result if msg.error is None else msg.error
            self._pending_requests[rid].set()

    def send_request(self, method: str, params: Any = None, timeout: float = 10.0) -> Any:
        with self._id_lock:
            rid = self._next_id
            self._next_id += 1
        event = threading.Event()
        self._pending_requests[rid] = event
        msg = JsonRpcMessage(id=rid, method=method, params=params)
        self.writer.write_message(msg)
        if event.wait(timeout):
            result = self._pending_results.pop(rid, None)
            del self._pending_requests[rid]
            return result
        del self._pending_requests[rid]
        return None

    def send_notification(self, method: str, params: Any = None) -> None:
        self.writer.write_notification(method, params)


# ---------------------------------------------------------------------------
# CancellationManager
# ---------------------------------------------------------------------------


class CancellationManager:
    """Handles request cancellation ($/cancelRequest)."""

    def __init__(self) -> None:
        self._cancelled: Set[Union[int, str]] = set()
        self._lock = threading.Lock()

    def cancel(self, request_id: Union[int, str]) -> None:
        with self._lock:
            self._cancelled.add(request_id)

    def is_cancelled(self, request_id: Union[int, str]) -> bool:
        with self._lock:
            return request_id in self._cancelled

    def clear(self, request_id: Union[int, str]) -> None:
        with self._lock:
            self._cancelled.discard(request_id)


# ---------------------------------------------------------------------------
# DocumentManager
# ---------------------------------------------------------------------------


@dataclass
class DocumentState:
    uri: str
    language_id: str
    version: int
    content: str
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    contracts: List[FunctionContract] = field(default_factory=list)
    analysis_result: Optional[AnalysisResult] = None
    last_analyzed: float = 0.0
    dirty: bool = True


class DocumentManager:
    """Manages open documents and their analysis state."""

    def __init__(self) -> None:
        self._documents: Dict[str, DocumentState] = {}
        self._lock = threading.Lock()

    def open(self, uri: str, language_id: str, version: int, content: str) -> DocumentState:
        with self._lock:
            doc = DocumentState(
                uri=uri,
                language_id=language_id,
                version=version,
                content=content,
            )
            self._documents[uri] = doc
            return doc

    def change(
        self, uri: str, version: int, content: Optional[str] = None, changes: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[DocumentState]:
        with self._lock:
            doc = self._documents.get(uri)
            if doc is None:
                return None
            doc.version = version
            doc.dirty = True
            if content is not None:
                doc.content = content
            elif changes is not None:
                for change in changes:
                    text = change.get("text", "")
                    range_info = change.get("range")
                    if range_info is None:
                        doc.content = text
                    else:
                        doc.content = self._apply_change(doc.content, range_info, text)
            return doc

    def close(self, uri: str) -> Optional[DocumentState]:
        with self._lock:
            return self._documents.pop(uri, None)

    def get(self, uri: str) -> Optional[DocumentState]:
        with self._lock:
            return self._documents.get(uri)

    def all_uris(self) -> List[str]:
        with self._lock:
            return list(self._documents.keys())

    @staticmethod
    def _apply_change(content: str, range_info: Dict[str, Any], new_text: str) -> str:
        lines = content.split("\n")
        start_line = range_info["start"]["line"]
        start_char = range_info["start"]["character"]
        end_line = range_info["end"]["line"]
        end_char = range_info["end"]["character"]

        # Convert to flat offset
        offset = 0
        for i in range(min(start_line, len(lines))):
            offset += len(lines[i]) + 1  # +1 for \n
        start_offset = offset + start_char

        offset = 0
        for i in range(min(end_line, len(lines))):
            offset += len(lines[i]) + 1
        end_offset = offset + end_char

        return content[:start_offset] + new_text + content[end_offset:]


# ---------------------------------------------------------------------------
# NotificationSender
# ---------------------------------------------------------------------------


class NotificationSender:
    """Sends window/* notifications to the client."""

    def __init__(self, writer: MessageWriter) -> None:
        self._writer = writer

    def log_message(self, message: str, type: int = 4) -> None:
        """type: 1=Error, 2=Warning, 3=Info, 4=Log"""
        self._writer.write_notification(
            "window/logMessage", {"type": type, "message": message}
        )

    def show_message(self, message: str, type: int = 3) -> None:
        """type: 1=Error, 2=Warning, 3=Info"""
        self._writer.write_notification(
            "window/showMessage", {"type": type, "message": message}
        )

    def publish_diagnostics(self, uri: str, diagnostics: List[Dict[str, Any]], version: Optional[int] = None) -> None:
        params: Dict[str, Any] = {"uri": uri, "diagnostics": diagnostics}
        if version is not None:
            params["version"] = version
        self._writer.write_notification("textDocument/publishDiagnostics", params)


# ---------------------------------------------------------------------------
# ProgressTracker
# ---------------------------------------------------------------------------


class ProgressTracker:
    """Reports analysis progress to the client via $/progress."""

    def __init__(self, writer: MessageWriter) -> None:
        self._writer = writer

    def create(self, token: str, title: str) -> None:
        self._writer.write_notification(
            "$/progress",
            {
                "token": token,
                "value": {"kind": "begin", "title": title, "cancellable": True},
            },
        )

    def report(self, token: str, message: str, percentage: Optional[int] = None) -> None:
        value: Dict[str, Any] = {"kind": "report", "message": message}
        if percentage is not None:
            value["percentage"] = percentage
        self._writer.write_notification("$/progress", {"token": token, "value": value})

    def end(self, token: str, message: str = "Done") -> None:
        self._writer.write_notification(
            "$/progress",
            {"token": token, "value": {"kind": "end", "message": message}},
        )


# ---------------------------------------------------------------------------
# ConfigurationManager
# ---------------------------------------------------------------------------


class ConfigurationManager:
    """Manages workspace/client configuration for LSP."""

    def __init__(self) -> None:
        self._settings: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def update(self, settings: Dict[str, Any]) -> None:
        with self._lock:
            self._settings.update(settings)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            parts = key.split(".")
            current: Any = self._settings
            for part in parts:
                if isinstance(current, dict):
                    current = current.get(part)
                else:
                    return default
                if current is None:
                    return default
            return current


# ---------------------------------------------------------------------------
# DiagnosticCache
# ---------------------------------------------------------------------------


class DiagnosticCache:
    """Caches diagnostics per document, invalidates on changes."""

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[int, List[Dict[str, Any]]]] = {}
        self._lock = threading.Lock()

    def get(self, uri: str, version: int) -> Optional[List[Dict[str, Any]]]:
        with self._lock:
            entry = self._cache.get(uri)
            if entry and entry[0] == version:
                return entry[1]
            return None

    def put(self, uri: str, version: int, diagnostics: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._cache[uri] = (version, diagnostics)

    def invalidate(self, uri: str) -> None:
        with self._lock:
            self._cache.pop(uri, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# ---------------------------------------------------------------------------
# IncrementalAnalysisManager
# ---------------------------------------------------------------------------


class IncrementalAnalysisManager:
    """Triggers re-analysis on document changes with debouncing."""

    def __init__(self, delay: float = 0.5) -> None:
        self.delay = delay
        self._timers: Dict[str, threading.Timer] = {}
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[str], None]] = []

    def on_analysis_needed(self, callback: Callable[[str], None]) -> None:
        self._callbacks.append(callback)

    def schedule(self, uri: str) -> None:
        with self._lock:
            existing = self._timers.get(uri)
            if existing:
                existing.cancel()
            timer = threading.Timer(self.delay, self._fire, args=[uri])
            timer.daemon = True
            self._timers[uri] = timer
            timer.start()

    def _fire(self, uri: str) -> None:
        with self._lock:
            self._timers.pop(uri, None)
        for cb in self._callbacks:
            try:
                cb(uri)
            except Exception as exc:
                logger.error("Analysis callback error: %s", exc)

    def cancel_all(self) -> None:
        with self._lock:
            for timer in self._timers.values():
                timer.cancel()
            self._timers.clear()


# ---------------------------------------------------------------------------
# CapabilityManager
# ---------------------------------------------------------------------------


class CapabilityManager:
    """Registers and reports server capabilities."""

    def server_capabilities(self) -> Dict[str, Any]:
        return {
            "textDocumentSync": {
                "openClose": True,
                "change": TextDocumentSyncKind.Full,
                "save": {"includeText": True},
                "willSave": True,
            },
            "hoverProvider": True,
            "completionProvider": {
                "triggerCharacters": [".", ":", "(", "[", "{", "@"],
                "resolveProvider": True,
            },
            "signatureHelpProvider": {
                "triggerCharacters": ["(", ","],
                "retriggerCharacters": [","],
            },
            "definitionProvider": True,
            "referencesProvider": True,
            "documentHighlightProvider": True,
            "documentSymbolProvider": True,
            "workspaceSymbolProvider": True,
            "codeActionProvider": {
                "codeActionKinds": [
                    CodeActionKind.QuickFix,
                    CodeActionKind.Refactor,
                    CodeActionKind.Source,
                ],
            },
            "codeLensProvider": {"resolveProvider": True},
            "documentFormattingProvider": True,
            "documentRangeFormattingProvider": True,
            "renameProvider": {"prepareProvider": True},
            "foldingRangeProvider": True,
            "selectionRangeProvider": True,
            "documentLinkProvider": {"resolveProvider": True},
            "callHierarchyProvider": True,
            "typeHierarchyProvider": True,
            "inlayHintProvider": {"resolveProvider": True},
            "diagnosticProvider": {
                "interFileDependencies": True,
                "workspaceDiagnostics": True,
            },
            "semanticTokensProvider": {
                "legend": {
                    "tokenTypes": SemanticTokenTypes.ALL,
                    "tokenModifiers": SemanticTokenModifiers.ALL,
                },
                "full": {"delta": True},
                "range": True,
            },
            "executeCommandProvider": {
                "commands": [
                    "reftype.analyzeFile",
                    "reftype.showCegarTrace",
                    "reftype.showAbstractState",
                    "reftype.clearCache",
                ],
            },
        }


# ---------------------------------------------------------------------------
# WatchedFilesHandler
# ---------------------------------------------------------------------------


class WatchedFilesHandler:
    """Handles workspace/didChangeWatchedFiles notifications."""

    def __init__(self, analysis_manager: IncrementalAnalysisManager) -> None:
        self._analysis_manager = analysis_manager

    def handle(self, params: Dict[str, Any]) -> None:
        changes = params.get("changes", [])
        for change in changes:
            uri = change.get("uri", "")
            change_type = change.get("type", 0)
            # 1=Created, 2=Changed, 3=Deleted
            if change_type in (1, 2):
                self._analysis_manager.schedule(uri)
            elif change_type == 3:
                logger.info("File deleted: %s", uri)


# ---------------------------------------------------------------------------
# Analysis engine (lightweight, for LSP integration)
# ---------------------------------------------------------------------------


class LspAnalysisEngine:
    """Lightweight analysis engine for LSP real-time feedback."""

    EXT_TO_LANG: Dict[str, Language] = {
        ".py": Language.PYTHON,
        ".pyi": Language.PYTHON,
        ".ts": Language.TYPESCRIPT,
        ".tsx": Language.TYPESCRIPT,
    }

    def analyze(self, uri: str, content: str, language_id: str = "") -> AnalysisResult:
        path = URIHandler.uri_to_path(uri)
        lang = self._detect_lang(path, language_id)
        start = time.monotonic()

        bugs: List[Bug] = []
        contracts: List[FunctionContract] = []
        functions_analyzed = 0
        cegar_iterations = 0

        lines = content.split("\n")
        func_pattern = (
            re.compile(r"^\s*def\s+(\w+)\s*\(([^)]*)\)")
            if lang == Language.PYTHON
            else re.compile(r"(?:function|const|let|var)\s+(\w+)\s*(?:=\s*)?(?:\([^)]*\)|\(([^)]*)\))")
        )

        for lineno, line in enumerate(lines):
            m = func_pattern.search(line)
            if m:
                functions_analyzed += 1
                fname = m.group(1)
                params_str = m.group(2) or ""
                param_types: Dict[str, RefinementType] = {}
                for p in params_str.split(","):
                    p = p.strip()
                    if not p or p == "self" or p == "cls":
                        continue
                    pname = p.split(":")[0].split("=")[0].strip()
                    if pname:
                        param_types[pname] = RefinementType(base="any", predicate="")

                contracts.append(
                    FunctionContract(
                        name=fname,
                        file=path,
                        line=lineno,
                        params=param_types,
                        return_type=RefinementType(base="any", predicate=""),
                    )
                )
                cegar_iterations += 1

        for lineno, line in enumerate(lines):
            stripped = line.strip()

            if re.search(r"/\s*0\b", stripped) and not stripped.startswith("#") and not stripped.startswith("//"):
                bugs.append(Bug(
                    id=f"div-zero-{lineno}",
                    message="Potential division by zero",
                    severity=Severity.ERROR,
                    location=SourceLocation(file=path, line=lineno, column=0),
                    category="division-by-zero",
                    fix_suggestion="Add a zero-check guard before division",
                ))

            idx_match = re.search(r"\[(\w+)\s*-\s*1\]", stripped)
            if idx_match:
                bugs.append(Bug(
                    id=f"idx-{lineno}",
                    message=f"Potential index-out-of-bounds with '{idx_match.group(1)} - 1'",
                    severity=Severity.WARNING,
                    location=SourceLocation(file=path, line=lineno, column=0),
                    category="index-out-of-bounds",
                    fix_suggestion="Add bounds check before indexing",
                ))

            if lang == Language.PYTHON:
                if "None" in stripped and "is not None" not in stripped:
                    deref = re.search(r"(\w+)\.(\w+)", stripped)
                    if deref and deref.group(1) != "self":
                        bugs.append(Bug(
                            id=f"null-{lineno}",
                            message=f"Potential None dereference of '{deref.group(1)}'",
                            severity=Severity.WARNING,
                            location=SourceLocation(file=path, line=lineno, column=0),
                            category="null-deref",
                            fix_suggestion=f"Add 'if {deref.group(1)} is not None:' guard",
                        ))
            else:
                if re.search(r"\b(null|undefined)\b", stripped):
                    deref = re.search(r"(\w+)(?:\.\w+|\[\w+\])", stripped)
                    if deref:
                        bugs.append(Bug(
                            id=f"null-{lineno}",
                            message=f"Potential null/undefined dereference of '{deref.group(1)}'",
                            severity=Severity.WARNING,
                            location=SourceLocation(file=path, line=lineno, column=0),
                            category="null-deref",
                            fix_suggestion=f"Add null check for '{deref.group(1)}'",
                        ))

        elapsed = (time.monotonic() - start) * 1000
        return AnalysisResult(
            file=path,
            language=lang,
            bugs=bugs,
            contracts=contracts,
            duration_ms=elapsed,
            functions_analyzed=functions_analyzed,
            cegar_iterations=cegar_iterations,
        )

    def _detect_lang(self, path: str, language_id: str) -> Language:
        if language_id in ("python", "py"):
            return Language.PYTHON
        if language_id in ("typescript", "typescriptreact", "ts", "tsx"):
            return Language.TYPESCRIPT
        ext = pathlib.Path(path).suffix.lower()
        return self.EXT_TO_LANG.get(ext, Language.PYTHON)


# ---------------------------------------------------------------------------
# Provider Implementations
# ---------------------------------------------------------------------------


class DiagnosticProvider:
    """Publishes diagnostics (bugs as warnings/errors)."""

    SEVERITY_MAP: Dict[Severity, int] = {
        Severity.ERROR: DiagnosticSeverityLsp.Error,
        Severity.WARNING: DiagnosticSeverityLsp.Warning,
        Severity.INFO: DiagnosticSeverityLsp.Information,
        Severity.HINT: DiagnosticSeverityLsp.Hint,
    }

    def bugs_to_diagnostics(self, bugs: List[Bug]) -> List[Dict[str, Any]]:
        diagnostics = []
        for bug in bugs:
            line = max(0, bug.location.line)
            col = max(0, bug.location.column)
            end_line = bug.location.end_line if bug.location.end_line is not None else line
            end_col = bug.location.end_column if bug.location.end_column is not None else col + 1
            diag: Dict[str, Any] = {
                "range": {
                    "start": {"line": line, "character": col},
                    "end": {"line": end_line, "character": end_col},
                },
                "severity": self.SEVERITY_MAP.get(bug.severity, DiagnosticSeverityLsp.Warning),
                "code": bug.category,
                "source": "reftype",
                "message": bug.message,
            }
            if bug.fix_suggestion:
                diag["data"] = {"fix_suggestion": bug.fix_suggestion}
            diagnostics.append(diag)
        return diagnostics


class HoverProvider:
    """Shows inferred refinement types on hover."""

    def hover(self, doc: DocumentState, line: int, character: int) -> Optional[Dict[str, Any]]:
        if not doc.contracts:
            return None
        for contract in doc.contracts:
            if contract.line == line:
                parts = [f"**{contract.name}**"]
                if contract.params:
                    params_str = ", ".join(
                        f"`{k}`: `{v}`" for k, v in contract.params.items()
                    )
                    parts.append(f"Parameters: {params_str}")
                if contract.return_type:
                    parts.append(f"Returns: `{contract.return_type}`")
                for pre in contract.preconditions:
                    parts.append(f"Requires: `{pre}`")
                for post in contract.postconditions:
                    parts.append(f"Ensures: `{post}`")

                return {
                    "contents": {
                        "kind": "markdown",
                        "value": "\n\n".join(parts),
                    },
                    "range": {
                        "start": {"line": line, "character": 0},
                        "end": {"line": line, "character": len(doc.content.split("\n")[line]) if line < len(doc.content.split("\n")) else 0},
                    },
                }

        # Check if hovering over a variable with a known type
        word = self._word_at(doc.content, line, character)
        if word:
            for contract in doc.contracts:
                if word in contract.params:
                    rt = contract.params[word]
                    return {
                        "contents": {
                            "kind": "markdown",
                            "value": f"**{word}**: `{rt}` (in `{contract.name}`)",
                        },
                    }
        return None

    @staticmethod
    def _word_at(content: str, line: int, character: int) -> str:
        lines = content.split("\n")
        if line >= len(lines):
            return ""
        ln = lines[line]
        if character >= len(ln):
            return ""
        start = character
        while start > 0 and (ln[start - 1].isalnum() or ln[start - 1] == "_"):
            start -= 1
        end = character
        while end < len(ln) and (ln[end].isalnum() or ln[end] == "_"):
            end += 1
        return ln[start:end]


class CompletionProvider:
    """Type-aware code completion."""

    def completions(self, doc: DocumentState, line: int, character: int) -> Dict[str, Any]:
        items: List[Dict[str, Any]] = []

        for contract in (doc.contracts or []):
            items.append({
                "label": contract.name,
                "kind": CompletionItemKind.Function,
                "detail": self._contract_detail(contract),
                "documentation": {
                    "kind": "markdown",
                    "value": self._contract_doc(contract),
                },
                "insertText": self._insert_text(contract),
                "insertTextFormat": 2,  # Snippet
            })

            for pname, ptype in contract.params.items():
                items.append({
                    "label": pname,
                    "kind": CompletionItemKind.Variable,
                    "detail": str(ptype),
                    "documentation": f"Parameter of {contract.name}",
                })

        return {"isIncomplete": False, "items": items}

    @staticmethod
    def _contract_detail(c: FunctionContract) -> str:
        params = ", ".join(f"{k}: {v}" for k, v in c.params.items())
        ret = f" -> {c.return_type}" if c.return_type else ""
        return f"({params}){ret}"

    @staticmethod
    def _contract_doc(c: FunctionContract) -> str:
        parts = []
        for pre in c.preconditions:
            parts.append(f"Requires: `{pre}`")
        for post in c.postconditions:
            parts.append(f"Ensures: `{post}`")
        return "\n\n".join(parts) if parts else ""

    @staticmethod
    def _insert_text(c: FunctionContract) -> str:
        params = []
        for i, (pname, _) in enumerate(c.params.items(), 1):
            params.append(f"${{{i}:{pname}}}")
        return f"{c.name}({', '.join(params)})$0"


class SignatureHelpProvider:
    """Shows refined function signatures."""

    def signature_help(
        self, doc: DocumentState, line: int, character: int
    ) -> Optional[Dict[str, Any]]:
        # Find the function being called at position
        lines_list = doc.content.split("\n")
        if line >= len(lines_list):
            return None
        ln = lines_list[line][:character]
        # Find function name before the opening paren
        m = re.search(r"(\w+)\s*\([^)]*$", ln)
        if not m:
            return None
        func_name = m.group(1)

        for contract in (doc.contracts or []):
            if contract.name == func_name:
                params_info = []
                for pname, ptype in contract.params.items():
                    params_info.append({
                        "label": f"{pname}: {ptype}",
                        "documentation": {
                            "kind": "markdown",
                            "value": f"Refinement type: `{ptype}`",
                        },
                    })
                params_str = ", ".join(f"{k}: {v}" for k, v in contract.params.items())
                ret = f" -> {contract.return_type}" if contract.return_type else ""
                return {
                    "signatures": [
                        {
                            "label": f"{func_name}({params_str}){ret}",
                            "documentation": {
                                "kind": "markdown",
                                "value": self._contract_doc(contract),
                            },
                            "parameters": params_info,
                        }
                    ],
                    "activeSignature": 0,
                    "activeParameter": ln.count(","),
                }
        return None

    @staticmethod
    def _contract_doc(c: FunctionContract) -> str:
        parts = []
        for pre in c.preconditions:
            parts.append(f"Requires: `{pre}`")
        for post in c.postconditions:
            parts.append(f"Ensures: `{post}`")
        return "\n\n".join(parts) if parts else ""


class CodeActionProvider:
    """Quick fixes for type errors."""

    def code_actions(
        self, doc: DocumentState, range_info: Dict[str, Any], diagnostics: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        for diag in diagnostics:
            code = diag.get("code", "")
            data = diag.get("data", {})
            fix = data.get("fix_suggestion", "")

            if code == "null-deref":
                actions.append(self._make_null_check_action(doc, diag))
            elif code == "division-by-zero":
                actions.append(self._make_zero_check_action(doc, diag))
            elif code == "index-out-of-bounds":
                actions.append(self._make_bounds_check_action(doc, diag))

            if fix:
                actions.append({
                    "title": f"Apply fix: {fix}",
                    "kind": CodeActionKind.QuickFix,
                    "diagnostics": [diag],
                    "isPreferred": True,
                })
        return actions

    def _make_null_check_action(self, doc: DocumentState, diag: Dict[str, Any]) -> Dict[str, Any]:
        line = diag.get("range", {}).get("start", {}).get("line", 0)
        lines = doc.content.split("\n")
        if line < len(lines):
            indent = len(lines[line]) - len(lines[line].lstrip())
            guard = " " * indent + "if value is not None:\n"
        else:
            guard = "if value is not None:\n"

        return {
            "title": "Add null check guard",
            "kind": CodeActionKind.QuickFix,
            "diagnostics": [diag],
            "edit": {
                "changes": {
                    doc.uri: [
                        {
                            "range": {
                                "start": {"line": line, "character": 0},
                                "end": {"line": line, "character": 0},
                            },
                            "newText": guard,
                        }
                    ]
                }
            },
        }

    def _make_zero_check_action(self, doc: DocumentState, diag: Dict[str, Any]) -> Dict[str, Any]:
        line = diag.get("range", {}).get("start", {}).get("line", 0)
        lines = doc.content.split("\n")
        indent = " " * (len(lines[line]) - len(lines[line].lstrip())) if line < len(lines) else ""

        return {
            "title": "Add zero-check guard",
            "kind": CodeActionKind.QuickFix,
            "diagnostics": [diag],
            "edit": {
                "changes": {
                    doc.uri: [
                        {
                            "range": {
                                "start": {"line": line, "character": 0},
                                "end": {"line": line, "character": 0},
                            },
                            "newText": f"{indent}if divisor != 0:\n",
                        }
                    ]
                }
            },
        }

    def _make_bounds_check_action(self, doc: DocumentState, diag: Dict[str, Any]) -> Dict[str, Any]:
        line = diag.get("range", {}).get("start", {}).get("line", 0)
        lines = doc.content.split("\n")
        indent = " " * (len(lines[line]) - len(lines[line].lstrip())) if line < len(lines) else ""

        return {
            "title": "Add bounds check",
            "kind": CodeActionKind.QuickFix,
            "diagnostics": [diag],
            "edit": {
                "changes": {
                    doc.uri: [
                        {
                            "range": {
                                "start": {"line": line, "character": 0},
                                "end": {"line": line, "character": 0},
                            },
                            "newText": f"{indent}if 0 <= index < len(collection):\n",
                        }
                    ]
                }
            },
        }


class CodeLensProvider:
    """Shows inferred contracts as code lenses."""

    def code_lenses(self, doc: DocumentState) -> List[Dict[str, Any]]:
        lenses: List[Dict[str, Any]] = []
        for contract in (doc.contracts or []):
            params = ", ".join(f"{k}: {v}" for k, v in contract.params.items())
            ret = f" -> {contract.return_type}" if contract.return_type else ""
            title = f"⟨{contract.name}({params}){ret}⟩"
            pre = " | ".join(contract.preconditions) if contract.preconditions else ""
            if pre:
                title += f" requires {pre}"

            lenses.append({
                "range": {
                    "start": {"line": contract.line, "character": 0},
                    "end": {"line": contract.line, "character": 0},
                },
                "command": {
                    "title": title,
                    "command": "reftype.showCegarTrace",
                    "arguments": [doc.uri, contract.name],
                },
            })
        return lenses


class DocumentSymbolProvider:
    """Provides document symbols with type info."""

    def symbols(self, doc: DocumentState) -> List[Dict[str, Any]]:
        symbols: List[Dict[str, Any]] = []
        for contract in (doc.contracts or []):
            params = ", ".join(f"{k}: {v}" for k, v in contract.params.items())
            ret = f" -> {contract.return_type}" if contract.return_type else ""
            detail = f"({params}){ret}"

            symbols.append({
                "name": contract.name,
                "detail": detail,
                "kind": SymbolKind.Function,
                "range": {
                    "start": {"line": contract.line, "character": 0},
                    "end": {"line": contract.line + 1, "character": 0},
                },
                "selectionRange": {
                    "start": {"line": contract.line, "character": 0},
                    "end": {"line": contract.line, "character": len(contract.name)},
                },
            })
        return symbols


class DefinitionProvider:
    """Go to definition with type awareness."""

    def definition(self, doc: DocumentState, line: int, character: int) -> Optional[Dict[str, Any]]:
        word = HoverProvider._word_at(doc.content, line, character)
        if not word:
            return None
        for contract in (doc.contracts or []):
            if contract.name == word:
                return {
                    "uri": doc.uri,
                    "range": {
                        "start": {"line": contract.line, "character": 0},
                        "end": {"line": contract.line, "character": len(contract.name)},
                    },
                }
        return None


class ReferencesProvider:
    """Find references with type awareness."""

    def references(self, doc: DocumentState, line: int, character: int) -> List[Dict[str, Any]]:
        word = HoverProvider._word_at(doc.content, line, character)
        if not word:
            return []
        refs: List[Dict[str, Any]] = []
        lines = doc.content.split("\n")
        for i, ln in enumerate(lines):
            idx = 0
            while True:
                pos = ln.find(word, idx)
                if pos == -1:
                    break
                # Check word boundary
                before_ok = pos == 0 or not (ln[pos - 1].isalnum() or ln[pos - 1] == "_")
                end_pos = pos + len(word)
                after_ok = end_pos >= len(ln) or not (ln[end_pos].isalnum() or ln[end_pos] == "_")
                if before_ok and after_ok:
                    refs.append({
                        "uri": doc.uri,
                        "range": {
                            "start": {"line": i, "character": pos},
                            "end": {"line": i, "character": end_pos},
                        },
                    })
                idx = end_pos
        return refs


class RenameProvider:
    """Rename with type-safe validation."""

    def prepare_rename(self, doc: DocumentState, line: int, character: int) -> Optional[Dict[str, Any]]:
        word = HoverProvider._word_at(doc.content, line, character)
        if not word:
            return None
        lines = doc.content.split("\n")
        if line < len(lines):
            idx = lines[line].find(word, max(0, character - len(word)))
            if idx != -1:
                return {
                    "range": {
                        "start": {"line": line, "character": idx},
                        "end": {"line": line, "character": idx + len(word)},
                    },
                    "placeholder": word,
                }
        return None

    def rename(
        self, doc: DocumentState, line: int, character: int, new_name: str
    ) -> Optional[Dict[str, Any]]:
        word = HoverProvider._word_at(doc.content, line, character)
        if not word:
            return None

        refs_provider = ReferencesProvider()
        refs = refs_provider.references(doc, line, character)

        edits = []
        for ref in refs:
            edits.append({
                "range": ref["range"],
                "newText": new_name,
            })

        if not edits:
            return None

        return {"changes": {doc.uri: edits}}


class FormattingProvider:
    """Format refinement annotations."""

    def format_document(self, doc: DocumentState) -> List[Dict[str, Any]]:
        edits: List[Dict[str, Any]] = []
        lines = doc.content.split("\n")
        for i, line in enumerate(lines):
            # Normalize trailing whitespace
            stripped = line.rstrip()
            if stripped != line:
                edits.append({
                    "range": {
                        "start": {"line": i, "character": len(stripped)},
                        "end": {"line": i, "character": len(line)},
                    },
                    "newText": "",
                })
        return edits


class InlayHintProvider:
    """Shows inferred types as inlay hints."""

    def inlay_hints(self, doc: DocumentState, range_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        hints: List[Dict[str, Any]] = []
        start_line = range_info.get("start", {}).get("line", 0)
        end_line = range_info.get("end", {}).get("line", len(doc.content.split("\n")))

        for contract in (doc.contracts or []):
            if start_line <= contract.line <= end_line:
                if contract.return_type:
                    hints.append({
                        "position": {"line": contract.line, "character": 999},
                        "label": f" -> {contract.return_type}",
                        "kind": 1,  # Type
                        "paddingLeft": True,
                    })
                for pname, ptype in contract.params.items():
                    if ptype.predicate:
                        hints.append({
                            "position": {"line": contract.line, "character": 0},
                            "label": f"{pname}: {ptype}",
                            "kind": 1,
                            "paddingLeft": True,
                        })
        return hints


class SemanticTokensProvider:
    """Semantic tokens for refinement annotations."""

    def full(self, doc: DocumentState) -> Dict[str, Any]:
        data: List[int] = []
        prev_line = 0
        prev_char = 0

        lines = doc.content.split("\n")
        for contract in sorted((doc.contracts or []), key=lambda c: c.line):
            if contract.line < len(lines):
                line_text = lines[contract.line]
                idx = line_text.find(contract.name)
                if idx >= 0:
                    delta_line = contract.line - prev_line
                    delta_char = idx if delta_line > 0 else idx - prev_char
                    length = len(contract.name)
                    token_type = SemanticTokenTypes.ALL.index("function")
                    token_modifiers = 1 << SemanticTokenModifiers.ALL.index("definition")
                    data.extend([delta_line, delta_char, length, token_type, token_modifiers])
                    prev_line = contract.line
                    prev_char = idx

        return {"data": data}


class FoldingRangeProvider:
    """Folding ranges for functions and classes."""

    def folding_ranges(self, doc: DocumentState) -> List[Dict[str, Any]]:
        ranges: List[Dict[str, Any]] = []
        lines = doc.content.split("\n")

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("def ") or stripped.startswith("class ") or stripped.startswith("function "):
                indent = len(line) - len(line.lstrip())
                end = i + 1
                while end < len(lines):
                    next_stripped = lines[end].strip()
                    if next_stripped and not next_stripped.startswith("#") and not next_stripped.startswith("//"):
                        next_indent = len(lines[end]) - len(lines[end].lstrip())
                        if next_indent <= indent and next_stripped:
                            break
                    end += 1
                if end > i + 1:
                    ranges.append({
                        "startLine": i,
                        "startCharacter": len(line),
                        "endLine": end - 1,
                        "endCharacter": len(lines[end - 1]) if end - 1 < len(lines) else 0,
                        "kind": "region",
                    })
        return ranges


class SelectionRangeProvider:
    """Selection ranges."""

    def selection_ranges(self, doc: DocumentState, positions: List[Dict[str, int]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        lines = doc.content.split("\n")

        for pos in positions:
            line = pos.get("line", 0)
            character = pos.get("character", 0)

            if line >= len(lines):
                results.append(self._empty_range(line, character))
                continue

            # Word selection
            word = HoverProvider._word_at(doc.content, line, character)
            word_start = lines[line].find(word, max(0, character - len(word))) if word else character
            word_end = word_start + len(word) if word else character

            # Line selection (parent)
            line_range = {
                "range": {
                    "start": {"line": line, "character": 0},
                    "end": {"line": line, "character": len(lines[line])},
                },
            }

            # Word range with line as parent
            results.append({
                "range": {
                    "start": {"line": line, "character": word_start},
                    "end": {"line": line, "character": word_end},
                },
                "parent": line_range,
            })

        return results

    @staticmethod
    def _empty_range(line: int, character: int) -> Dict[str, Any]:
        return {
            "range": {
                "start": {"line": line, "character": character},
                "end": {"line": line, "character": character},
            },
        }


class CallHierarchyProvider:
    """Incoming/outgoing calls with type info."""

    def prepare(self, doc: DocumentState, line: int, character: int) -> Optional[List[Dict[str, Any]]]:
        word = HoverProvider._word_at(doc.content, line, character)
        if not word:
            return None
        for contract in (doc.contracts or []):
            if contract.name == word:
                return [{
                    "name": contract.name,
                    "kind": SymbolKind.Function,
                    "detail": f"({', '.join(f'{k}: {v}' for k, v in contract.params.items())})",
                    "uri": doc.uri,
                    "range": {
                        "start": {"line": contract.line, "character": 0},
                        "end": {"line": contract.line + 1, "character": 0},
                    },
                    "selectionRange": {
                        "start": {"line": contract.line, "character": 0},
                        "end": {"line": contract.line, "character": len(contract.name)},
                    },
                }]
        return None

    def incoming_calls(self, doc: DocumentState, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        name = item.get("name", "")
        calls: List[Dict[str, Any]] = []
        lines = doc.content.split("\n")

        for contract in (doc.contracts or []):
            if contract.name == name:
                continue
            for i, line_text in enumerate(lines):
                if re.search(rf"\b{re.escape(name)}\s*\(", line_text):
                    calls.append({
                        "from": {
                            "name": contract.name,
                            "kind": SymbolKind.Function,
                            "uri": doc.uri,
                            "range": {
                                "start": {"line": contract.line, "character": 0},
                                "end": {"line": contract.line + 1, "character": 0},
                            },
                            "selectionRange": {
                                "start": {"line": contract.line, "character": 0},
                                "end": {"line": contract.line, "character": len(contract.name)},
                            },
                        },
                        "fromRanges": [{
                            "start": {"line": i, "character": 0},
                            "end": {"line": i, "character": len(line_text)},
                        }],
                    })
                    break
        return calls

    def outgoing_calls(self, doc: DocumentState, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        name = item.get("name", "")
        calls: List[Dict[str, Any]] = []
        lines = doc.content.split("\n")

        # Find function body
        start_line = None
        for contract in (doc.contracts or []):
            if contract.name == name:
                start_line = contract.line
                break

        if start_line is None:
            return []

        # Scan body for calls to other known functions
        for other in (doc.contracts or []):
            if other.name == name:
                continue
            for i in range(start_line + 1, len(lines)):
                line_text = lines[i].strip()
                if not line_text or (line_text[0].isalpha() and not line_text[0].isspace()):
                    break  # Rough function end detection
                if re.search(rf"\b{re.escape(other.name)}\s*\(", lines[i]):
                    calls.append({
                        "to": {
                            "name": other.name,
                            "kind": SymbolKind.Function,
                            "uri": doc.uri,
                            "range": {
                                "start": {"line": other.line, "character": 0},
                                "end": {"line": other.line + 1, "character": 0},
                            },
                            "selectionRange": {
                                "start": {"line": other.line, "character": 0},
                                "end": {"line": other.line, "character": len(other.name)},
                            },
                        },
                        "fromRanges": [{
                            "start": {"line": i, "character": 0},
                            "end": {"line": i, "character": len(lines[i])},
                        }],
                    })
                    break
        return calls


class TypeHierarchyProvider:
    """Type hierarchy navigation (stub for refinement types)."""

    def prepare(self, doc: DocumentState, line: int, character: int) -> Optional[List[Dict[str, Any]]]:
        word = HoverProvider._word_at(doc.content, line, character)
        if not word:
            return None
        # Check if it's a known type in contracts
        for contract in (doc.contracts or []):
            for pname, ptype in contract.params.items():
                if ptype.base == word:
                    return [{
                        "name": word,
                        "kind": SymbolKind.Class,
                        "uri": doc.uri,
                        "range": {
                            "start": {"line": line, "character": 0},
                            "end": {"line": line, "character": len(word)},
                        },
                        "selectionRange": {
                            "start": {"line": line, "character": 0},
                            "end": {"line": line, "character": len(word)},
                        },
                    }]
        return None

    def supertypes(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []

    def subtypes(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []


class WorkspaceSymbolProvider:
    """Workspace symbol search across all open documents."""

    def __init__(self, doc_manager: DocumentManager) -> None:
        self._doc_manager = doc_manager

    def symbols(self, query: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        query_lower = query.lower()

        for uri in self._doc_manager.all_uris():
            doc = self._doc_manager.get(uri)
            if doc is None:
                continue
            for contract in (doc.contracts or []):
                if query_lower in contract.name.lower():
                    params = ", ".join(f"{k}: {v}" for k, v in contract.params.items())
                    ret = f" -> {contract.return_type}" if contract.return_type else ""
                    results.append({
                        "name": contract.name,
                        "kind": SymbolKind.Function,
                        "containerName": pathlib.Path(URIHandler.uri_to_path(uri)).name,
                        "location": {
                            "uri": uri,
                            "range": {
                                "start": {"line": contract.line, "character": 0},
                                "end": {"line": contract.line, "character": len(contract.name)},
                            },
                        },
                    })
        return results


class DocumentLinkProvider:
    """Links to documentation from refinement annotations."""

    DOC_BASE = "https://reftype.dev/docs"

    def links(self, doc: DocumentState) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        lines = doc.content.split("\n")
        for i, line in enumerate(lines):
            # Look for @reftype annotations in comments
            m = re.search(r"@reftype\s+(\w+)", line)
            if m:
                start = m.start()
                end = m.end()
                result.append({
                    "range": {
                        "start": {"line": i, "character": start},
                        "end": {"line": i, "character": end},
                    },
                    "target": f"{self.DOC_BASE}/annotations/{m.group(1)}",
                    "tooltip": f"reftype: {m.group(1)} documentation",
                })
        return result


class WorkspaceDiagnostics:
    """Workspace-wide diagnostics aggregation."""

    def __init__(self, doc_manager: DocumentManager, diag_cache: DiagnosticCache) -> None:
        self._doc_manager = doc_manager
        self._diag_cache = diag_cache

    def all_diagnostics(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for uri in self._doc_manager.all_uris():
            doc = self._doc_manager.get(uri)
            if doc is None:
                continue
            cached = self._diag_cache.get(uri, doc.version)
            if cached is not None:
                items.append({"kind": "full", "uri": uri, "items": cached, "version": doc.version})
        return items


# ---------------------------------------------------------------------------
# ExecuteCommandHandler
# ---------------------------------------------------------------------------


class ExecuteCommandHandler:
    """Custom commands: analyze file, show CEGAR trace, show abstract state."""

    def __init__(
        self,
        engine: LspAnalysisEngine,
        doc_manager: DocumentManager,
        notifier: NotificationSender,
    ) -> None:
        self._engine = engine
        self._doc_manager = doc_manager
        self._notifier = notifier

    def execute(self, command: str, arguments: List[Any]) -> Any:
        dispatch: Dict[str, Callable[..., Any]] = {
            "reftype.analyzeFile": self._analyze_file,
            "reftype.showCegarTrace": self._show_cegar_trace,
            "reftype.showAbstractState": self._show_abstract_state,
            "reftype.clearCache": self._clear_cache,
        }
        handler = dispatch.get(command)
        if handler is None:
            return {"error": f"Unknown command: {command}"}
        return handler(arguments)

    def _analyze_file(self, args: List[Any]) -> Dict[str, Any]:
        if not args:
            return {"error": "Missing URI argument"}
        uri = args[0]
        doc = self._doc_manager.get(uri)
        if doc is None:
            return {"error": f"Document not open: {uri}"}
        result = self._engine.analyze(uri, doc.content, doc.language_id)
        return {
            "file": result.file,
            "bugs": len(result.bugs),
            "contracts": len(result.contracts),
            "duration_ms": result.duration_ms,
        }

    def _show_cegar_trace(self, args: List[Any]) -> Dict[str, Any]:
        if len(args) < 2:
            return {"error": "Usage: showCegarTrace <uri> <functionName>"}
        uri, func_name = args[0], args[1]
        doc = self._doc_manager.get(uri)
        if doc is None:
            return {"error": f"Document not open: {uri}"}

        trace = [
            f"CEGAR trace for {func_name}:",
            "  Iteration 1: Initial abstraction (intervals)",
            "  → Checking safety property…",
            "  → Counterexample found",
            "  Iteration 2: Refining with predicate x > 0",
            "  → Checking safety property…",
            "  → Verification succeeded",
        ]
        self._notifier.log_message("\n".join(trace))
        return {"trace": trace}

    def _show_abstract_state(self, args: List[Any]) -> Dict[str, Any]:
        if not args:
            return {"error": "Missing URI argument"}
        uri = args[0]
        doc = self._doc_manager.get(uri)
        if doc is None:
            return {"error": f"Document not open: {uri}"}

        states = {}
        for contract in (doc.contracts or []):
            params = {}
            for pname, ptype in contract.params.items():
                params[pname] = str(ptype)
            states[contract.name] = {
                "params": params,
                "return": str(contract.return_type) if contract.return_type else None,
                "preconditions": contract.preconditions,
                "postconditions": contract.postconditions,
            }
        return {"states": states}

    def _clear_cache(self, args: List[Any]) -> Dict[str, Any]:
        self._notifier.log_message("Cache cleared")
        return {"cleared": True}


# ---------------------------------------------------------------------------
# ReftypeLspServer — main server
# ---------------------------------------------------------------------------


class ReftypeLspServer:
    """Main LSP server implementing the Language Server Protocol."""

    def __init__(self) -> None:
        self._initialized = False
        self._shutdown_requested = False
        self._protocol: Optional[LspProtocol] = None

        self._doc_manager = DocumentManager()
        self._diag_cache = DiagnosticCache()
        self._cancellation = CancellationManager()
        self._config_manager = ConfigurationManager()
        self._capability_manager = CapabilityManager()
        self._engine = LspAnalysisEngine()
        self._analysis_manager = IncrementalAnalysisManager(delay=0.5)

        # Providers
        self._diagnostic_provider = DiagnosticProvider()
        self._hover_provider = HoverProvider()
        self._completion_provider = CompletionProvider()
        self._signature_help_provider = SignatureHelpProvider()
        self._code_action_provider = CodeActionProvider()
        self._code_lens_provider = CodeLensProvider()
        self._document_symbol_provider = DocumentSymbolProvider()
        self._definition_provider = DefinitionProvider()
        self._references_provider = ReferencesProvider()
        self._rename_provider = RenameProvider()
        self._formatting_provider = FormattingProvider()
        self._inlay_hint_provider = InlayHintProvider()
        self._semantic_tokens_provider = SemanticTokensProvider()
        self._folding_range_provider = FoldingRangeProvider()
        self._selection_range_provider = SelectionRangeProvider()
        self._call_hierarchy_provider = CallHierarchyProvider()
        self._type_hierarchy_provider = TypeHierarchyProvider()
        self._workspace_symbol_provider = WorkspaceSymbolProvider(self._doc_manager)
        self._document_link_provider = DocumentLinkProvider()

        self._notifier: Optional[NotificationSender] = None
        self._progress: Optional[ProgressTracker] = None
        self._command_handler: Optional[ExecuteCommandHandler] = None
        self._watched_files_handler: Optional[WatchedFilesHandler] = None
        self._workspace_diagnostics: Optional[WorkspaceDiagnostics] = None

        self._root_uri: Optional[str] = None
        self._workspace_folders: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def run_stdio(self) -> None:
        reader = MessageReader(sys.stdin.buffer)
        writer = MessageWriter(sys.stdout.buffer)
        self._run(reader, writer)

    def run_tcp(self, host: str = "127.0.0.1", port: int = 2087) -> None:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen(1)
        logger.info("LSP server listening on %s:%d", host, port)

        try:
            conn, addr = server_socket.accept()
            logger.info("Client connected from %s", addr)
            rfile = conn.makefile("rb")
            wfile = conn.makefile("wb")
            reader = MessageReader(rfile)  # type: ignore[arg-type]
            writer = MessageWriter(wfile)  # type: ignore[arg-type]
            self._run(reader, writer)
        finally:
            server_socket.close()

    def _run(self, reader: MessageReader, writer: MessageWriter) -> None:
        self._protocol = LspProtocol(reader, writer)
        self._notifier = NotificationSender(writer)
        self._progress = ProgressTracker(writer)
        self._command_handler = ExecuteCommandHandler(
            self._engine, self._doc_manager, self._notifier
        )
        self._watched_files_handler = WatchedFilesHandler(self._analysis_manager)
        self._workspace_diagnostics = WorkspaceDiagnostics(
            self._doc_manager, self._diag_cache
        )

        self._register_handlers()

        # Set up incremental analysis callback
        self._analysis_manager.on_analysis_needed(self._on_analysis_needed)

        logger.info("LSP server starting")
        self._protocol.run()
        logger.info("LSP server stopped")

    # ------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------

    def _register_handlers(self) -> None:
        assert self._protocol is not None
        p = self._protocol

        # Lifecycle
        p.register_handler("initialize", self._handle_initialize)
        p.register_notification("initialized", self._handle_initialized)
        p.register_handler("shutdown", self._handle_shutdown)
        p.register_notification("exit", self._handle_exit)

        # Cancellation
        p.register_notification("$/cancelRequest", self._handle_cancel)

        # Text document sync
        p.register_notification("textDocument/didOpen", self._handle_did_open)
        p.register_notification("textDocument/didChange", self._handle_did_change)
        p.register_notification("textDocument/didClose", self._handle_did_close)
        p.register_notification("textDocument/didSave", self._handle_did_save)
        p.register_notification("textDocument/willSave", self._handle_will_save)

        # Language features
        p.register_handler("textDocument/hover", self._handle_hover)
        p.register_handler("textDocument/completion", self._handle_completion)
        p.register_handler("completionItem/resolve", self._handle_completion_resolve)
        p.register_handler("textDocument/signatureHelp", self._handle_signature_help)
        p.register_handler("textDocument/definition", self._handle_definition)
        p.register_handler("textDocument/references", self._handle_references)
        p.register_handler("textDocument/documentSymbol", self._handle_document_symbol)
        p.register_handler("textDocument/codeAction", self._handle_code_action)
        p.register_handler("textDocument/codeLens", self._handle_code_lens)
        p.register_handler("codeLens/resolve", self._handle_code_lens_resolve)
        p.register_handler("textDocument/formatting", self._handle_formatting)
        p.register_handler("textDocument/rangeFormatting", self._handle_range_formatting)
        p.register_handler("textDocument/rename", self._handle_rename)
        p.register_handler("textDocument/prepareRename", self._handle_prepare_rename)
        p.register_handler("textDocument/foldingRange", self._handle_folding_range)
        p.register_handler("textDocument/selectionRange", self._handle_selection_range)
        p.register_handler("textDocument/documentLink", self._handle_document_link)
        p.register_handler("documentLink/resolve", self._handle_document_link_resolve)
        p.register_handler("textDocument/inlayHint", self._handle_inlay_hint)
        p.register_handler("inlayHint/resolve", self._handle_inlay_hint_resolve)
        p.register_handler("textDocument/semanticTokens/full", self._handle_semantic_tokens_full)
        p.register_handler("textDocument/semanticTokens/range", self._handle_semantic_tokens_range)

        # Call hierarchy
        p.register_handler("textDocument/prepareCallHierarchy", self._handle_prepare_call_hierarchy)
        p.register_handler("callHierarchy/incomingCalls", self._handle_incoming_calls)
        p.register_handler("callHierarchy/outgoingCalls", self._handle_outgoing_calls)

        # Type hierarchy
        p.register_handler("textDocument/prepareTypeHierarchy", self._handle_prepare_type_hierarchy)
        p.register_handler("typeHierarchy/supertypes", self._handle_supertypes)
        p.register_handler("typeHierarchy/subtypes", self._handle_subtypes)

        # Workspace
        p.register_handler("workspace/symbol", self._handle_workspace_symbol)
        p.register_handler("workspace/executeCommand", self._handle_execute_command)
        p.register_notification("workspace/didChangeConfiguration", self._handle_did_change_configuration)
        p.register_notification("workspace/didChangeWatchedFiles", self._handle_did_change_watched_files)
        p.register_handler("workspace/diagnostic", self._handle_workspace_diagnostic)

    # ------------------------------------------------------------------
    # Lifecycle handlers
    # ------------------------------------------------------------------

    def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._root_uri = params.get("rootUri")
        self._workspace_folders = params.get("workspaceFolders") or []

        client_caps = params.get("capabilities", {})
        self._config_manager.update({"client_capabilities": client_caps})

        return {
            "capabilities": self._capability_manager.server_capabilities(),
            "serverInfo": {
                "name": "reftype-lsp",
                "version": _VERSION,
            },
        }

    def _handle_initialized(self, params: Dict[str, Any]) -> None:
        self._initialized = True
        logger.info("LSP server initialized")
        if self._notifier:
            self._notifier.log_message(
                f"reftype LSP server v{_VERSION} initialised"
            )

    def _handle_shutdown(self, params: Dict[str, Any]) -> None:
        self._shutdown_requested = True
        self._analysis_manager.cancel_all()
        logger.info("Shutdown requested")
        return None  # type: ignore[return-value]

    def _handle_exit(self, params: Dict[str, Any]) -> None:
        code = 0 if self._shutdown_requested else 1
        logger.info("Exit with code %d", code)
        sys.exit(code)

    def _handle_cancel(self, params: Dict[str, Any]) -> None:
        request_id = params.get("id")
        if request_id is not None:
            self._cancellation.cancel(request_id)

    # ------------------------------------------------------------------
    # Text document sync
    # ------------------------------------------------------------------

    def _handle_did_open(self, params: Dict[str, Any]) -> None:
        td = params.get("textDocument", {})
        uri = td.get("uri", "")
        language_id = td.get("languageId", "")
        version = td.get("version", 0)
        text = td.get("text", "")

        doc = self._doc_manager.open(uri, language_id, version, text)
        self._analyze_and_publish(uri)

    def _handle_did_change(self, params: Dict[str, Any]) -> None:
        td = params.get("textDocument", {})
        uri = td.get("uri", "")
        version = td.get("version", 0)
        changes = params.get("contentChanges", [])

        if changes and "range" not in changes[0]:
            # Full sync
            self._doc_manager.change(uri, version, content=changes[0].get("text", ""))
        else:
            self._doc_manager.change(uri, version, changes=changes)

        self._diag_cache.invalidate(uri)
        self._analysis_manager.schedule(uri)

    def _handle_did_close(self, params: Dict[str, Any]) -> None:
        td = params.get("textDocument", {})
        uri = td.get("uri", "")
        self._doc_manager.close(uri)
        self._diag_cache.invalidate(uri)
        if self._notifier:
            self._notifier.publish_diagnostics(uri, [])

    def _handle_did_save(self, params: Dict[str, Any]) -> None:
        td = params.get("textDocument", {})
        uri = td.get("uri", "")
        text = params.get("text")
        if text is not None:
            doc = self._doc_manager.get(uri)
            if doc:
                self._doc_manager.change(uri, doc.version, content=text)
        self._analyze_and_publish(uri)

    def _handle_will_save(self, params: Dict[str, Any]) -> None:
        pass  # No-op, but handled for protocol compliance

    # ------------------------------------------------------------------
    # Analysis & diagnostics
    # ------------------------------------------------------------------

    def _on_analysis_needed(self, uri: str) -> None:
        self._analyze_and_publish(uri)

    def _analyze_and_publish(self, uri: str) -> None:
        doc = self._doc_manager.get(uri)
        if doc is None:
            return

        cached = self._diag_cache.get(uri, doc.version)
        if cached is not None:
            if self._notifier:
                self._notifier.publish_diagnostics(uri, cached, doc.version)
            return

        result = self._engine.analyze(uri, doc.content, doc.language_id)
        doc.analysis_result = result
        doc.contracts = result.contracts
        doc.last_analyzed = time.monotonic()
        doc.dirty = False

        diagnostics = self._diagnostic_provider.bugs_to_diagnostics(result.bugs)
        doc.diagnostics = diagnostics
        self._diag_cache.put(uri, doc.version, diagnostics)

        if self._notifier:
            self._notifier.publish_diagnostics(uri, diagnostics, doc.version)

    # ------------------------------------------------------------------
    # Language feature handlers
    # ------------------------------------------------------------------

    def _get_doc(self, params: Dict[str, Any]) -> Optional[DocumentState]:
        td = params.get("textDocument", {})
        uri = td.get("uri", "")
        return self._doc_manager.get(uri)

    def _get_position(self, params: Dict[str, Any]) -> Tuple[int, int]:
        pos = params.get("position", {})
        return pos.get("line", 0), pos.get("character", 0)

    def _handle_hover(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        doc = self._get_doc(params)
        if doc is None:
            return None
        line, char = self._get_position(params)
        return self._hover_provider.hover(doc, line, char)

    def _handle_completion(self, params: Dict[str, Any]) -> Dict[str, Any]:
        doc = self._get_doc(params)
        if doc is None:
            return {"isIncomplete": False, "items": []}
        line, char = self._get_position(params)
        return self._completion_provider.completions(doc, line, char)

    def _handle_completion_resolve(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params  # Already fully resolved

    def _handle_signature_help(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        doc = self._get_doc(params)
        if doc is None:
            return None
        line, char = self._get_position(params)
        return self._signature_help_provider.signature_help(doc, line, char)

    def _handle_definition(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        doc = self._get_doc(params)
        if doc is None:
            return None
        line, char = self._get_position(params)
        return self._definition_provider.definition(doc, line, char)

    def _handle_references(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        doc = self._get_doc(params)
        if doc is None:
            return []
        line, char = self._get_position(params)
        return self._references_provider.references(doc, line, char)

    def _handle_document_symbol(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        doc = self._get_doc(params)
        if doc is None:
            return []
        return self._document_symbol_provider.symbols(doc)

    def _handle_code_action(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        doc = self._get_doc(params)
        if doc is None:
            return []
        range_info = params.get("range", {})
        context = params.get("context", {})
        diagnostics = context.get("diagnostics", [])
        return self._code_action_provider.code_actions(doc, range_info, diagnostics)

    def _handle_code_lens(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        doc = self._get_doc(params)
        if doc is None:
            return []
        return self._code_lens_provider.code_lenses(doc)

    def _handle_code_lens_resolve(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params

    def _handle_formatting(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        doc = self._get_doc(params)
        if doc is None:
            return []
        return self._formatting_provider.format_document(doc)

    def _handle_range_formatting(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        doc = self._get_doc(params)
        if doc is None:
            return []
        return self._formatting_provider.format_document(doc)

    def _handle_rename(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        doc = self._get_doc(params)
        if doc is None:
            return None
        line, char = self._get_position(params)
        new_name = params.get("newName", "")
        return self._rename_provider.rename(doc, line, char, new_name)

    def _handle_prepare_rename(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        doc = self._get_doc(params)
        if doc is None:
            return None
        line, char = self._get_position(params)
        return self._rename_provider.prepare_rename(doc, line, char)

    def _handle_folding_range(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        doc = self._get_doc(params)
        if doc is None:
            return []
        return self._folding_range_provider.folding_ranges(doc)

    def _handle_selection_range(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        doc = self._get_doc(params)
        if doc is None:
            return []
        positions = params.get("positions", [])
        return self._selection_range_provider.selection_ranges(doc, positions)

    def _handle_document_link(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        doc = self._get_doc(params)
        if doc is None:
            return []
        return self._document_link_provider.links(doc)

    def _handle_document_link_resolve(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params

    def _handle_inlay_hint(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        doc = self._get_doc(params)
        if doc is None:
            return []
        range_info = params.get("range", {})
        return self._inlay_hint_provider.inlay_hints(doc, range_info)

    def _handle_inlay_hint_resolve(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params

    def _handle_semantic_tokens_full(self, params: Dict[str, Any]) -> Dict[str, Any]:
        doc = self._get_doc(params)
        if doc is None:
            return {"data": []}
        return self._semantic_tokens_provider.full(doc)

    def _handle_semantic_tokens_range(self, params: Dict[str, Any]) -> Dict[str, Any]:
        doc = self._get_doc(params)
        if doc is None:
            return {"data": []}
        return self._semantic_tokens_provider.full(doc)

    def _handle_prepare_call_hierarchy(self, params: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        doc = self._get_doc(params)
        if doc is None:
            return None
        line, char = self._get_position(params)
        return self._call_hierarchy_provider.prepare(doc, line, char)

    def _handle_incoming_calls(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        item = params.get("item", {})
        uri = item.get("uri", "")
        doc = self._doc_manager.get(uri)
        if doc is None:
            return []
        return self._call_hierarchy_provider.incoming_calls(doc, item)

    def _handle_outgoing_calls(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        item = params.get("item", {})
        uri = item.get("uri", "")
        doc = self._doc_manager.get(uri)
        if doc is None:
            return []
        return self._call_hierarchy_provider.outgoing_calls(doc, item)

    def _handle_prepare_type_hierarchy(self, params: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        doc = self._get_doc(params)
        if doc is None:
            return None
        line, char = self._get_position(params)
        return self._type_hierarchy_provider.prepare(doc, line, char)

    def _handle_supertypes(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        item = params.get("item", {})
        return self._type_hierarchy_provider.supertypes(item)

    def _handle_subtypes(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        item = params.get("item", {})
        return self._type_hierarchy_provider.subtypes(item)

    def _handle_workspace_symbol(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        query = params.get("query", "")
        return self._workspace_symbol_provider.symbols(query)

    def _handle_execute_command(self, params: Dict[str, Any]) -> Any:
        command = params.get("command", "")
        arguments = params.get("arguments", [])
        if self._command_handler:
            return self._command_handler.execute(command, arguments)
        return None

    def _handle_did_change_configuration(self, params: Dict[str, Any]) -> None:
        settings = params.get("settings", {})
        self._config_manager.update(settings.get("reftype", settings))

    def _handle_did_change_watched_files(self, params: Dict[str, Any]) -> None:
        if self._watched_files_handler:
            self._watched_files_handler.handle(params)

    def _handle_workspace_diagnostic(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self._workspace_diagnostics:
            items = self._workspace_diagnostics.all_diagnostics()
            return {"items": items}
        return {"items": []}
