"""
CABER Model Client Interface
=============================

Coalgebraic Behavioral Auditing framework — unified async interface for
querying large-language-model APIs (OpenAI, Anthropic, HuggingFace) with
retry logic, rate-limit awareness, streaming, batch concurrency, token
estimation, and cost projection.

Provides:
    * Pydantic data-models for requests / responses
    * Abstract ``ModelClient`` with retry, batching, streaming
    * Concrete clients: OpenAI, Anthropic, HuggingFace, Mock
    * Factory helper ``create_client`` and ``estimate_cost``
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import re
import string
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import httpx
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT_SECONDS: float = 120.0
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_MAX_TOKENS: int = 4096
DEFAULT_TEMPERATURE: float = 0.7
OPENAI_BASE_URL: str = "https://api.openai.com"
ANTHROPIC_BASE_URL: str = "https://api.anthropic.com"
HUGGINGFACE_BASE_URL: str = "https://api-inference.huggingface.co"

RATE_LIMIT_HEADER_REMAINING_REQUESTS = "x-ratelimit-remaining-requests"
RATE_LIMIT_HEADER_REMAINING_TOKENS = "x-ratelimit-remaining-tokens"
RATE_LIMIT_HEADER_RESET = "x-ratelimit-reset-requests"

ANTHROPIC_VERSION = "2023-06-01"

# Cost table: model -> (input_cost_per_1k, output_cost_per_1k)
MODEL_COST_TABLE: Dict[str, Tuple[float, float]] = {
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-4": (0.03, 0.06),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "claude-3-opus-20240229": (0.015, 0.075),
    "claude-3-sonnet-20240229": (0.003, 0.015),
    "claude-3-haiku-20240307": (0.00025, 0.00125),
    "claude-3-5-sonnet-20240620": (0.003, 0.015),
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
}

# Punctuation set used by the improved tokeniser
_PUNCTUATION_RE = re.compile(r"[\s" + re.escape(string.punctuation) + r"]+")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FinishReason(str, Enum):
    """Normalised finish-reason across providers."""

    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALLS = "tool_calls"
    ERROR = "error"
    UNKNOWN = "unknown"

    @classmethod
    def from_openai(cls, value: Optional[str]) -> "FinishReason":
        """Map an OpenAI finish_reason string to the enum."""
        mapping = {
            "stop": cls.STOP,
            "length": cls.LENGTH,
            "content_filter": cls.CONTENT_FILTER,
            "tool_calls": cls.TOOL_CALLS,
        }
        if value is None:
            return cls.UNKNOWN
        return mapping.get(value, cls.UNKNOWN)

    @classmethod
    def from_anthropic(cls, value: Optional[str]) -> "FinishReason":
        """Map an Anthropic stop_reason string to the enum."""
        mapping = {
            "end_turn": cls.STOP,
            "max_tokens": cls.LENGTH,
            "stop_sequence": cls.STOP,
        }
        if value is None:
            return cls.UNKNOWN
        return mapping.get(value, cls.UNKNOWN)


class MessageRole(str, Enum):
    """Chat message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"


# ---------------------------------------------------------------------------
# Pydantic data models
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Configuration for a model client instance."""

    model_name: str = Field(
        ..., description="Model identifier, e.g. 'gpt-4o' or 'claude-3-opus-20240229'."
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )
    max_tokens: int = Field(
        default=DEFAULT_MAX_TOKENS,
        gt=0,
        description="Maximum tokens to generate.",
    )
    api_key: str = Field(default="", description="API key / bearer token.")
    base_url: str = Field(
        default="",
        description="Base URL for the API. Provider-specific default used if empty.",
    )
    timeout: float = Field(
        default=DEFAULT_TIMEOUT_SECONDS,
        gt=0,
        description="HTTP request timeout in seconds.",
    )
    max_retries: int = Field(
        default=DEFAULT_MAX_RETRIES,
        ge=0,
        description="Maximum retry attempts on transient failures.",
    )
    extra_headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional HTTP headers to include in every request.",
    )
    extra_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional body parameters forwarded to the API.",
    )
    top_p: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Nucleus sampling parameter."
    )
    frequency_penalty: Optional[float] = Field(
        default=None, ge=-2.0, le=2.0, description="Frequency penalty."
    )
    presence_penalty: Optional[float] = Field(
        default=None, ge=-2.0, le=2.0, description="Presence penalty."
    )
    stop_sequences: Optional[List[str]] = Field(
        default=None, description="Stop sequences."
    )
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility.")

    @field_validator("api_key")
    @classmethod
    def _strip_key(cls, v: str) -> str:
        return v.strip()

    @field_validator("base_url")
    @classmethod
    def _strip_url(cls, v: str) -> str:
        return v.rstrip("/").strip()


class Message(BaseModel):
    """A single chat message."""

    role: MessageRole = Field(..., description="Role of the message author.")
    content: str = Field(..., description="Text content of the message.")
    name: Optional[str] = Field(
        default=None,
        description="Optional participant name for multi-agent conversations.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata attached to this message.",
    )

    def to_openai_dict(self) -> Dict[str, Any]:
        """Serialise to the dict format expected by the OpenAI chat API."""
        d: Dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name is not None:
            d["name"] = self.name
        return d

    def to_anthropic_dict(self) -> Dict[str, Any]:
        """Serialise for the Anthropic messages API (no system role)."""
        return {"role": self.role.value, "content": self.content}

    def token_estimate(self) -> int:
        """Rough whitespace-based token count for the message content."""
        words = self.content.split()
        return max(1, int(math.ceil(len(words) / 0.75)))


class Conversation(BaseModel):
    """Ordered sequence of chat messages plus optional system prompt."""

    messages: List[Message] = Field(default_factory=list, description="Message history.")
    system_prompt: Optional[str] = Field(
        default=None, description="System-level instruction prepended to messages."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Conversation-level metadata (e.g. audit trail ids).",
    )

    # -- convenience helpers --------------------------------------------------

    def add_user(self, content: str, *, name: Optional[str] = None) -> "Conversation":
        """Append a user message and return self for chaining."""
        self.messages.append(Message(role=MessageRole.USER, content=content, name=name))
        return self

    def add_assistant(self, content: str) -> "Conversation":
        """Append an assistant message and return self for chaining."""
        self.messages.append(Message(role=MessageRole.ASSISTANT, content=content))
        return self

    def add_system(self, content: str) -> "Conversation":
        """Set the system prompt and return self for chaining."""
        self.system_prompt = content
        return self

    def last_user_message(self) -> Optional[str]:
        """Return the content of the most recent user message, or *None*."""
        for msg in reversed(self.messages):
            if msg.role == MessageRole.USER:
                return msg.content
        return None

    def last_assistant_message(self) -> Optional[str]:
        """Return the content of the most recent assistant message, or *None*."""
        for msg in reversed(self.messages):
            if msg.role == MessageRole.ASSISTANT:
                return msg.content
        return None

    def to_openai_messages(self) -> List[Dict[str, Any]]:
        """Build the ``messages`` list for the OpenAI chat completion API."""
        out: List[Dict[str, Any]] = []
        if self.system_prompt:
            out.append({"role": "system", "content": self.system_prompt})
        for m in self.messages:
            out.append(m.to_openai_dict())
        return out

    def to_anthropic_messages(self) -> List[Dict[str, Any]]:
        """Build the ``messages`` list for the Anthropic messages API.

        System prompt is handled separately in the Anthropic payload.
        """
        return [
            m.to_anthropic_dict()
            for m in self.messages
            if m.role != MessageRole.SYSTEM
        ]

    def message_count(self) -> int:
        """Total number of messages (excludes system prompt)."""
        return len(self.messages)

    def total_content_length(self) -> int:
        """Aggregate character count across all messages and system prompt."""
        total = sum(len(m.content) for m in self.messages)
        if self.system_prompt:
            total += len(self.system_prompt)
        return total

    def copy_with_messages(self, messages: List[Message]) -> "Conversation":
        """Create a shallow copy with a new message list."""
        return Conversation(
            messages=messages,
            system_prompt=self.system_prompt,
            metadata=dict(self.metadata),
        )

    def truncate(self, max_messages: int) -> "Conversation":
        """Return a copy keeping only the last *max_messages* messages."""
        kept = self.messages[-max_messages:] if max_messages < len(self.messages) else list(self.messages)
        return self.copy_with_messages(kept)


class TokenUsage(BaseModel):
    """Token consumption summary."""

    prompt_tokens: int = Field(default=0, ge=0, description="Tokens in the prompt.")
    completion_tokens: int = Field(
        default=0, ge=0, description="Tokens in the completion."
    )
    total_tokens: int = Field(
        default=0, ge=0, description="Total tokens (prompt + completion)."
    )

    @model_validator(mode="after")
    def _compute_total(self) -> "TokenUsage":
        if self.total_tokens == 0 and (self.prompt_tokens or self.completion_tokens):
            self.total_tokens = self.prompt_tokens + self.completion_tokens
        return self

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Merge two usage records."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class ModelResponse(BaseModel):
    """Normalised response from any LLM provider."""

    content: str = Field(..., description="Generated text content.")
    finish_reason: FinishReason = Field(
        default=FinishReason.UNKNOWN, description="Why generation stopped."
    )
    token_usage: TokenUsage = Field(
        default_factory=TokenUsage, description="Token consumption."
    )
    latency_ms: float = Field(
        default=0.0, ge=0.0, description="End-to-end request latency in milliseconds."
    )
    model: str = Field(default="", description="Model id that actually served the request.")
    raw_response: Optional[Dict[str, Any]] = Field(
        default=None, description="Provider-specific raw JSON body."
    )
    request_id: Optional[str] = Field(
        default=None, description="Provider-assigned request id for tracing."
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the response object was created.",
    )

    def is_complete(self) -> bool:
        """Return *True* if generation ended normally (STOP)."""
        return self.finish_reason == FinishReason.STOP

    def is_truncated(self) -> bool:
        """Return *True* if generation hit the token limit."""
        return self.finish_reason == FinishReason.LENGTH


class StreamChunk(BaseModel):
    """A single chunk from a streaming response."""

    delta: str = Field(default="", description="Incremental text content.")
    finish_reason: Optional[FinishReason] = Field(
        default=None, description="Set on the final chunk."
    )
    index: int = Field(default=0, ge=0, description="Choice index (multi-choice).")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of this chunk.",
    )

    def is_final(self) -> bool:
        """Return *True* if this chunk signals the end of the stream."""
        return self.finish_reason is not None


class RateLimitInfo(BaseModel):
    """Parsed rate-limit metadata from response headers."""

    requests_remaining: Optional[int] = Field(
        default=None, description="Remaining requests in the current window."
    )
    tokens_remaining: Optional[int] = Field(
        default=None, description="Remaining tokens in the current window."
    )
    reset_at: Optional[datetime] = Field(
        default=None, description="UTC datetime when the window resets."
    )

    def is_exhausted(self) -> bool:
        """Return *True* when either bucket is known to be empty."""
        if self.requests_remaining is not None and self.requests_remaining <= 0:
            return True
        if self.tokens_remaining is not None and self.tokens_remaining <= 0:
            return True
        return False

    def seconds_until_reset(self) -> float:
        """Seconds until the rate-limit window resets, or 0.0 if unknown."""
        if self.reset_at is None:
            return 0.0
        now = datetime.now(timezone.utc)
        delta = (self.reset_at - now).total_seconds()
        return max(0.0, delta)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ModelClientError(Exception):
    """Base exception for all model-client errors."""

    def __init__(self, message: str, *, status_code: Optional[int] = None, raw: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.status_code = status_code
        self.raw = raw


class AuthenticationError(ModelClientError):
    """Invalid or missing API key."""


class RateLimitError(ModelClientError):
    """The provider returned a 429 rate-limit response."""

    def __init__(
        self,
        message: str,
        *,
        retry_after: Optional[float] = None,
        rate_limit_info: Optional[RateLimitInfo] = None,
        status_code: int = 429,
        raw: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, status_code=status_code, raw=raw)
        self.retry_after = retry_after
        self.rate_limit_info = rate_limit_info


class ContentFilterError(ModelClientError):
    """The response was blocked by the provider's content filter."""


class InvalidRequestError(ModelClientError):
    """The request payload was malformed or semantically invalid."""


class ServerError(ModelClientError):
    """The provider returned a 5xx server error."""


class StreamInterruptedError(ModelClientError):
    """The SSE stream was interrupted before completion."""


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------

_RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


def _is_retryable_status(status_code: int) -> bool:
    """Return *True* for HTTP status codes that warrant a retry."""
    return status_code in _RETRYABLE_STATUS_CODES


def _classify_http_error(status_code: int, body: Dict[str, Any], headers: httpx.Headers) -> ModelClientError:
    """Create an appropriate exception subclass from an HTTP error response."""
    msg = body.get("error", {}).get("message", "") if isinstance(body.get("error"), dict) else str(body.get("error", ""))
    if not msg:
        msg = body.get("message", f"HTTP {status_code}")

    if status_code == 401:
        return AuthenticationError(msg, status_code=status_code, raw=body)
    if status_code == 429:
        retry_after_raw = headers.get("retry-after")
        retry_after = float(retry_after_raw) if retry_after_raw else None
        return RateLimitError(msg, retry_after=retry_after, status_code=status_code, raw=body)
    if status_code == 400:
        return InvalidRequestError(msg, status_code=status_code, raw=body)
    if status_code >= 500:
        return ServerError(msg, status_code=status_code, raw=body)
    return ModelClientError(msg, status_code=status_code, raw=body)


def _parse_retry_after(headers: httpx.Headers) -> Optional[float]:
    """Extract a ``Retry-After`` value in seconds from response headers."""
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        pass
    # RFC 7231 HTTP-date
    try:
        dt = datetime.strptime(raw, "%a, %d %b %Y %H:%M:%S %Z")
        dt = dt.replace(tzinfo=timezone.utc)
        delta = (dt - datetime.now(timezone.utc)).total_seconds()
        return max(0.0, delta)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Abstract base: ModelClient
# ---------------------------------------------------------------------------


class ModelClient(ABC):
    """Abstract asynchronous client for querying an LLM.

    Subclasses must implement:
        * ``_build_headers``
        * ``query``
        * ``stream``
    """

    def __init__(self, config: ModelConfig) -> None:
        """Initialise the client with a :class:`ModelConfig`.

        Creates an ``httpx.AsyncClient`` configured with the timeout and
        any extra headers specified in *config*.
        """
        self.config = config
        self._closed = False
        self._total_requests: int = 0
        self._total_tokens_used: int = 0
        self._cumulative_usage = TokenUsage()
        self._last_rate_limit: Optional[RateLimitInfo] = None
        transport = httpx.AsyncHTTPTransport(retries=0)
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout, connect=30.0),
            transport=transport,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )

    # -- abstract interface ---------------------------------------------------

    @abstractmethod
    async def query(self, conversation: Conversation) -> ModelResponse:
        """Send a chat completion request and return the full response.

        Implementations must handle serialisation to the provider's format
        and parse the response into a :class:`ModelResponse`.
        """
        ...

    @abstractmethod
    async def stream(self, conversation: Conversation) -> AsyncGenerator[StreamChunk, None]:
        """Send a streaming request and yield :class:`StreamChunk` objects.

        The final chunk should have a non-``None`` *finish_reason*.
        """
        ...  # pragma: no cover
        # yield is needed to make this a generator in the abstract signature
        if False:  # type: ignore[unreachable]
            yield StreamChunk()

    @abstractmethod
    def _build_headers(self) -> Dict[str, str]:
        """Return provider-specific HTTP headers (incl. auth)."""
        ...

    # -- retry wrapper --------------------------------------------------------

    async def query_with_retry(self, conversation: Conversation) -> ModelResponse:
        """Query with exponential-backoff retry on transient failures.

        Uses ``config.max_retries`` for the retry budget.  On a 429 response
        the method sleeps for the ``Retry-After`` duration (or a capped
        exponential backoff) before retrying.
        """
        last_error: Optional[Exception] = None
        for attempt in range(1 + self.config.max_retries):
            try:
                response = await self.query(conversation)
                return response
            except RateLimitError as exc:
                last_error = exc
                wait = exc.retry_after if exc.retry_after is not None else min(2 ** attempt + random.random(), 60.0)
                logger.warning(
                    "Rate-limited (attempt %d/%d). Sleeping %.1fs.",
                    attempt + 1,
                    1 + self.config.max_retries,
                    wait,
                )
                await asyncio.sleep(wait)
            except ServerError as exc:
                last_error = exc
                wait = min(2 ** attempt + random.random(), 30.0)
                logger.warning(
                    "Server error %s (attempt %d/%d). Retrying in %.1fs.",
                    exc.status_code,
                    attempt + 1,
                    1 + self.config.max_retries,
                    wait,
                )
                await asyncio.sleep(wait)
            except httpx.TransportError as exc:
                last_error = exc
                wait = min(2 ** attempt + random.random(), 30.0)
                logger.warning(
                    "Transport error (attempt %d/%d): %s. Retrying in %.1fs.",
                    attempt + 1,
                    1 + self.config.max_retries,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)
            except (AuthenticationError, InvalidRequestError, ContentFilterError):
                raise  # non-retryable

        assert last_error is not None
        raise last_error

    # -- token counting -------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Estimate token count with a simple whitespace heuristic.

        Splits on whitespace and divides by 0.75 (i.e. assumes ~0.75 words
        per token, which over-counts slightly as a safety margin).
        """
        if not text:
            return 0
        words = text.split()
        return max(1, int(math.ceil(len(words) / 0.75)))

    def count_conversation_tokens(self, conversation: Conversation) -> int:
        """Estimate total token count for a conversation.

        Sums the token estimates for every message and the system prompt.
        Adds a small per-message overhead (4 tokens) to account for role
        markers and delimiters used by most providers.
        """
        total = 0
        if conversation.system_prompt:
            total += self.count_tokens(conversation.system_prompt) + 4
        for msg in conversation.messages:
            total += self.count_tokens(msg.content) + 4
        return total

    # -- batch ----------------------------------------------------------------

    async def batch_query(
        self,
        conversations: List[Conversation],
        max_concurrent: int = 5,
    ) -> List[ModelResponse]:
        """Query multiple conversations concurrently.

        Uses an ``asyncio.Semaphore`` to cap the number of in-flight
        requests to *max_concurrent*.  Returns results in the same order
        as the input list.
        """
        if not conversations:
            return []

        semaphore = asyncio.Semaphore(max_concurrent)
        results: List[Optional[ModelResponse]] = [None] * len(conversations)
        errors: List[Optional[Exception]] = [None] * len(conversations)

        async def _run(idx: int, conv: Conversation) -> None:
            async with semaphore:
                try:
                    results[idx] = await self.query_with_retry(conv)
                except Exception as exc:  # noqa: BLE001
                    errors[idx] = exc
                    logger.error("Batch item %d failed: %s", idx, exc)

        tasks = [asyncio.create_task(_run(i, c)) for i, c in enumerate(conversations)]
        await asyncio.gather(*tasks, return_exceptions=False)

        # Raise the first error encountered, if any
        for idx, err in enumerate(errors):
            if err is not None:
                logger.error(
                    "batch_query: %d/%d conversations failed. Raising first error.",
                    sum(1 for e in errors if e is not None),
                    len(conversations),
                )
                raise err

        return [r for r in results if r is not None]

    # -- rate-limit parsing ---------------------------------------------------

    def _parse_rate_limits(self, headers: Union[httpx.Headers, Dict[str, str]]) -> Optional[RateLimitInfo]:
        """Parse standard rate-limit headers into a :class:`RateLimitInfo`.

        Looks for ``x-ratelimit-remaining-requests``,
        ``x-ratelimit-remaining-tokens``, and
        ``x-ratelimit-reset-requests``.  Returns *None* when no
        rate-limit headers are present.
        """
        raw_remaining_req = headers.get(RATE_LIMIT_HEADER_REMAINING_REQUESTS)
        raw_remaining_tok = headers.get(RATE_LIMIT_HEADER_REMAINING_TOKENS)
        raw_reset = headers.get(RATE_LIMIT_HEADER_RESET)

        if raw_remaining_req is None and raw_remaining_tok is None and raw_reset is None:
            return None

        remaining_req: Optional[int] = None
        remaining_tok: Optional[int] = None
        reset_at: Optional[datetime] = None

        if raw_remaining_req is not None:
            try:
                remaining_req = int(raw_remaining_req)
            except ValueError:
                pass

        if raw_remaining_tok is not None:
            try:
                remaining_tok = int(raw_remaining_tok)
            except ValueError:
                pass

        if raw_reset is not None:
            try:
                # Try epoch seconds first
                reset_at = datetime.fromtimestamp(float(raw_reset), tz=timezone.utc)
            except (ValueError, OSError):
                # Try ISO format
                try:
                    reset_at = datetime.fromisoformat(raw_reset.replace("Z", "+00:00"))
                except ValueError:
                    pass

        info = RateLimitInfo(
            requests_remaining=remaining_req,
            tokens_remaining=remaining_tok,
            reset_at=reset_at,
        )
        self._last_rate_limit = info
        return info

    # -- lifecycle ------------------------------------------------------------

    @property
    def is_closed(self) -> bool:
        """Whether the underlying HTTP client has been closed."""
        return self._closed

    async def close(self) -> None:
        """Close the underlying HTTP client and release resources."""
        if not self._closed:
            await self._http.aclose()
            self._closed = True
            logger.debug("ModelClient closed (total requests: %d)", self._total_requests)

    async def __aenter__(self) -> "ModelClient":
        """Enter the async context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the async context manager, closing the client."""
        await self.close()

    # -- internal helpers -----------------------------------------------------

    def _record_usage(self, usage: TokenUsage) -> None:
        """Track cumulative token usage across all requests."""
        self._total_requests += 1
        self._total_tokens_used += usage.total_tokens
        self._cumulative_usage = self._cumulative_usage + usage

    def _effective_base_url(self, default: str) -> str:
        """Return the configured base URL or the provider default."""
        return self.config.base_url if self.config.base_url else default

    def _merge_headers(self) -> Dict[str, str]:
        """Combine provider headers with user-supplied extra headers."""
        h = self._build_headers()
        h.update(self.config.extra_headers)
        return h

    # -- stats ----------------------------------------------------------------

    @property
    def total_requests(self) -> int:
        """Number of successful queries executed by this client."""
        return self._total_requests

    @property
    def cumulative_usage(self) -> TokenUsage:
        """Aggregate token usage across all successful queries."""
        return self._cumulative_usage

    @property
    def last_rate_limit(self) -> Optional[RateLimitInfo]:
        """Most recently observed rate-limit info."""
        return self._last_rate_limit


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------


class OpenAIClient(ModelClient):
    """Async client for the OpenAI Chat Completions API.

    Defaults to ``gpt-4o`` and ``https://api.openai.com``.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        base_url: str = "",
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        **extra: Any,
    ) -> None:
        """Create an OpenAI client.

        Parameters
        ----------
        api_key:
            OpenAI API key.
        model:
            Model identifier.
        temperature:
            Sampling temperature.
        max_tokens:
            Max completion tokens.
        base_url:
            Override the API base URL (useful for Azure or proxies).
        timeout:
            HTTP timeout in seconds.
        max_retries:
            Retry budget for transient errors.
        **extra:
            Additional keyword arguments forwarded to ``ModelConfig.extra_params``.
        """
        config = ModelConfig(
            model_name=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            extra_params=extra,
        )
        super().__init__(config)

    def _build_headers(self) -> Dict[str, str]:
        """Return OpenAI authorization headers."""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload(self, conversation: Conversation, *, stream: bool = False) -> Dict[str, Any]:
        """Construct the JSON body for a chat completion request."""
        payload: Dict[str, Any] = {
            "model": self.config.model_name,
            "messages": conversation.to_openai_messages(),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": stream,
        }
        if self.config.top_p is not None:
            payload["top_p"] = self.config.top_p
        if self.config.frequency_penalty is not None:
            payload["frequency_penalty"] = self.config.frequency_penalty
        if self.config.presence_penalty is not None:
            payload["presence_penalty"] = self.config.presence_penalty
        if self.config.stop_sequences:
            payload["stop"] = self.config.stop_sequences
        if self.config.seed is not None:
            payload["seed"] = self.config.seed
        payload.update(self.config.extra_params)
        return payload

    async def query(self, conversation: Conversation) -> ModelResponse:
        """Send a non-streaming chat completion request to OpenAI.

        Returns a normalised :class:`ModelResponse`.
        """
        url = f"{self._effective_base_url(OPENAI_BASE_URL)}/v1/chat/completions"
        payload = self._build_payload(conversation, stream=False)
        headers = self._merge_headers()

        t0 = time.perf_counter()
        resp = await self._http.post(url, json=payload, headers=headers)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        self._parse_rate_limits(resp.headers)

        if resp.status_code != 200:
            try:
                body = resp.json()
            except Exception:
                body = {"error": resp.text}
            raise _classify_http_error(resp.status_code, body, resp.headers)

        data = resp.json()
        choice = data["choices"][0]
        usage_data = data.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )
        self._record_usage(usage)

        return ModelResponse(
            content=choice["message"]["content"] or "",
            finish_reason=FinishReason.from_openai(choice.get("finish_reason")),
            token_usage=usage,
            latency_ms=latency_ms,
            model=data.get("model", self.config.model_name),
            raw_response=data,
            request_id=resp.headers.get("x-request-id"),
        )

    async def stream(self, conversation: Conversation) -> AsyncGenerator[StreamChunk, None]:
        """Stream chat completion from OpenAI using SSE.

        Yields :class:`StreamChunk` objects; the final chunk carries a
        non-``None`` ``finish_reason``.
        """
        url = f"{self._effective_base_url(OPENAI_BASE_URL)}/v1/chat/completions"
        payload = self._build_payload(conversation, stream=True)
        headers = self._merge_headers()

        async with self._http.stream("POST", url, json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                body_bytes = await resp.aread()
                try:
                    body = json.loads(body_bytes)
                except Exception:
                    body = {"error": body_bytes.decode(errors="replace")}
                raise _classify_http_error(resp.status_code, body, resp.headers)

            self._parse_rate_limits(resp.headers)
            buffer = ""
            async for raw_bytes in resp.aiter_bytes():
                buffer += raw_bytes.decode(errors="replace")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        return
                    try:
                        chunk_data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk_data.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content_piece = delta.get("content", "")
                    fr_raw = choices[0].get("finish_reason")
                    fr = FinishReason.from_openai(fr_raw) if fr_raw else None
                    yield StreamChunk(
                        delta=content_piece or "",
                        finish_reason=fr,
                        index=choices[0].get("index", 0),
                    )

    def count_tokens(self, text: str) -> int:
        """Improved token estimator for OpenAI models.

        Splits on whitespace *and* punctuation and multiplies by 1.3 to
        approximate BPE tokenisation.
        """
        if not text:
            return 0
        parts = _PUNCTUATION_RE.split(text)
        parts = [p for p in parts if p]
        return max(1, int(math.ceil(len(parts) * 1.3)))


# ---------------------------------------------------------------------------
# Anthropic client
# ---------------------------------------------------------------------------


class AnthropicClient(ModelClient):
    """Async client for the Anthropic Messages API.

    Defaults to ``claude-3-5-sonnet-20241022``.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        base_url: str = "",
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        anthropic_version: str = ANTHROPIC_VERSION,
        **extra: Any,
    ) -> None:
        """Create an Anthropic client.

        Parameters
        ----------
        api_key:
            Anthropic API key.
        model:
            Model identifier.
        anthropic_version:
            Anthropic API version header value.
        """
        config = ModelConfig(
            model_name=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            extra_params=extra,
        )
        super().__init__(config)
        self._anthropic_version = anthropic_version

    def _build_headers(self) -> Dict[str, str]:
        """Return Anthropic-specific auth and version headers."""
        return {
            "x-api-key": self.config.api_key,
            "anthropic-version": self._anthropic_version,
            "Content-Type": "application/json",
        }

    def _build_payload(self, conversation: Conversation, *, stream: bool = False) -> Dict[str, Any]:
        """Construct the JSON body for an Anthropic messages request."""
        payload: Dict[str, Any] = {
            "model": self.config.model_name,
            "messages": conversation.to_anthropic_messages(),
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        if conversation.system_prompt:
            payload["system"] = conversation.system_prompt
        if stream:
            payload["stream"] = True
        if self.config.top_p is not None:
            payload["top_p"] = self.config.top_p
        if self.config.stop_sequences:
            payload["stop_sequences"] = self.config.stop_sequences
        payload.update(self.config.extra_params)
        return payload

    async def query(self, conversation: Conversation) -> ModelResponse:
        """Send a non-streaming request to the Anthropic messages API.

        System prompt is submitted via the top-level ``system`` field rather
        than as a message with role ``system``.
        """
        url = f"{self._effective_base_url(ANTHROPIC_BASE_URL)}/v1/messages"
        payload = self._build_payload(conversation, stream=False)
        headers = self._merge_headers()

        t0 = time.perf_counter()
        resp = await self._http.post(url, json=payload, headers=headers)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        self._parse_rate_limits(resp.headers)

        if resp.status_code != 200:
            try:
                body = resp.json()
            except Exception:
                body = {"error": resp.text}
            raise _classify_http_error(resp.status_code, body, resp.headers)

        data = resp.json()

        # Anthropic returns content as a list of blocks
        content_blocks = data.get("content", [])
        text_parts = [
            block.get("text", "")
            for block in content_blocks
            if block.get("type") == "text"
        ]
        content = "".join(text_parts)

        usage_data = data.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
        )
        self._record_usage(usage)

        return ModelResponse(
            content=content,
            finish_reason=FinishReason.from_anthropic(data.get("stop_reason")),
            token_usage=usage,
            latency_ms=latency_ms,
            model=data.get("model", self.config.model_name),
            raw_response=data,
            request_id=resp.headers.get("request-id"),
        )

    async def stream(self, conversation: Conversation) -> AsyncGenerator[StreamChunk, None]:
        """Stream a response from the Anthropic messages API.

        Parses the Anthropic SSE event stream and yields
        :class:`StreamChunk` objects.
        """
        url = f"{self._effective_base_url(ANTHROPIC_BASE_URL)}/v1/messages"
        payload = self._build_payload(conversation, stream=True)
        headers = self._merge_headers()

        async with self._http.stream("POST", url, json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                body_bytes = await resp.aread()
                try:
                    body = json.loads(body_bytes)
                except Exception:
                    body = {"error": body_bytes.decode(errors="replace")}
                raise _classify_http_error(resp.status_code, body, resp.headers)

            self._parse_rate_limits(resp.headers)
            buffer = ""
            current_event: Optional[str] = None
            async for raw_bytes in resp.aiter_bytes():
                buffer += raw_bytes.decode(errors="replace")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()

                    if not line:
                        current_event = None
                        continue

                    if line.startswith("event:"):
                        current_event = line[len("event:"):].strip()
                        continue

                    if line.startswith("data:"):
                        data_str = line[len("data:"):].strip()
                        try:
                            event_data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        if current_event == "content_block_delta":
                            delta = event_data.get("delta", {})
                            text = delta.get("text", "")
                            yield StreamChunk(delta=text, index=event_data.get("index", 0))

                        elif current_event == "message_delta":
                            delta = event_data.get("delta", {})
                            stop_reason = delta.get("stop_reason")
                            fr = FinishReason.from_anthropic(stop_reason) if stop_reason else None
                            yield StreamChunk(delta="", finish_reason=fr, index=0)

                        elif current_event == "message_stop":
                            return

    def count_tokens(self, text: str) -> int:
        """Token estimation tuned for Anthropic's Claude tokeniser.

        Uses a character-based heuristic: roughly 3.5 characters per token
        for English text, with an adjustment for whitespace density.
        """
        if not text:
            return 0
        char_count = len(text)
        word_count = len(text.split())
        # Blend character- and word-based estimates
        char_estimate = char_count / 3.5
        word_estimate = word_count * 1.35
        blended = (char_estimate + word_estimate) / 2.0
        return max(1, int(math.ceil(blended)))


# ---------------------------------------------------------------------------
# HuggingFace Inference API client
# ---------------------------------------------------------------------------


class HuggingFaceClient(ModelClient):
    """Async client for the HuggingFace Inference API.

    Supports both text-generation and conversational model endpoints.
    """

    def __init__(
        self,
        api_key: str,
        model_id: str,
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        base_url: str = "",
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        use_chat_format: bool = True,
        **extra: Any,
    ) -> None:
        """Create a HuggingFace Inference API client.

        Parameters
        ----------
        api_key:
            HuggingFace access token.
        model_id:
            Full model identifier on HuggingFace Hub (e.g. ``meta-llama/Llama-2-70b-chat-hf``).
        use_chat_format:
            If *True*, uses the ``/v1/chat/completions`` compatible endpoint
            provided by TGI / inference endpoints.  Otherwise falls back to
            the raw ``/models/{model_id}`` endpoint.
        """
        config = ModelConfig(
            model_name=model_id,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            extra_params=extra,
        )
        super().__init__(config)
        self._model_id = model_id
        self._use_chat_format = use_chat_format

    def _build_headers(self) -> Dict[str, str]:
        """Return HuggingFace authorization headers."""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def _build_payload_chat(self, conversation: Conversation, *, stream: bool = False) -> Dict[str, Any]:
        """Build a TGI-compatible chat completion payload."""
        payload: Dict[str, Any] = {
            "model": self._model_id,
            "messages": conversation.to_openai_messages(),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": stream,
        }
        if self.config.top_p is not None:
            payload["top_p"] = self.config.top_p
        if self.config.stop_sequences:
            payload["stop"] = self.config.stop_sequences
        payload.update(self.config.extra_params)
        return payload

    def _build_payload_raw(self, conversation: Conversation) -> Dict[str, Any]:
        """Build a raw inference API payload (non-chat)."""
        # Concatenate conversation into a single prompt string
        parts: List[str] = []
        if conversation.system_prompt:
            parts.append(f"System: {conversation.system_prompt}")
        for msg in conversation.messages:
            parts.append(f"{msg.role.value.capitalize()}: {msg.content}")
        parts.append("Assistant:")
        prompt = "\n".join(parts)

        return {
            "inputs": prompt,
            "parameters": {
                "temperature": self.config.temperature,
                "max_new_tokens": self.config.max_tokens,
                "return_full_text": False,
            },
        }

    def _chat_url(self) -> str:
        """Return the chat completions URL."""
        base = self._effective_base_url(HUGGINGFACE_BASE_URL)
        return f"{base}/v1/chat/completions"

    def _raw_url(self) -> str:
        """Return the raw model inference URL."""
        base = self._effective_base_url(HUGGINGFACE_BASE_URL)
        return f"{base}/models/{self._model_id}"

    async def query(self, conversation: Conversation) -> ModelResponse:
        """Send a request to the HuggingFace Inference API.

        Chooses chat-completion or raw-inference format based on
        ``use_chat_format``.
        """
        headers = self._merge_headers()
        t0 = time.perf_counter()

        if self._use_chat_format:
            url = self._chat_url()
            payload = self._build_payload_chat(conversation, stream=False)
            resp = await self._http.post(url, json=payload, headers=headers)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            self._parse_rate_limits(resp.headers)
            if resp.status_code != 200:
                try:
                    body = resp.json()
                except Exception:
                    body = {"error": resp.text}
                raise _classify_http_error(resp.status_code, body, resp.headers)

            data = resp.json()
            choice = data["choices"][0]
            usage_data = data.get("usage", {})
            usage = TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )
            self._record_usage(usage)
            return ModelResponse(
                content=choice["message"]["content"] or "",
                finish_reason=FinishReason.from_openai(choice.get("finish_reason")),
                token_usage=usage,
                latency_ms=latency_ms,
                model=data.get("model", self._model_id),
                raw_response=data,
            )
        else:
            url = self._raw_url()
            payload = self._build_payload_raw(conversation)
            resp = await self._http.post(url, json=payload, headers=headers)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            self._parse_rate_limits(resp.headers)
            if resp.status_code != 200:
                try:
                    body = resp.json()
                except Exception:
                    body = {"error": resp.text}
                raise _classify_http_error(resp.status_code, body, resp.headers)

            data = resp.json()
            # HF raw API returns a list of generated texts
            if isinstance(data, list) and len(data) > 0:
                generated = data[0].get("generated_text", "")
            elif isinstance(data, dict):
                generated = data.get("generated_text", "")
            else:
                generated = str(data)

            estimated_prompt_tokens = self.count_conversation_tokens(conversation)
            estimated_completion_tokens = self.count_tokens(generated)
            usage = TokenUsage(
                prompt_tokens=estimated_prompt_tokens,
                completion_tokens=estimated_completion_tokens,
            )
            self._record_usage(usage)
            return ModelResponse(
                content=generated,
                finish_reason=FinishReason.STOP,
                token_usage=usage,
                latency_ms=latency_ms,
                model=self._model_id,
                raw_response=data if isinstance(data, dict) else {"results": data},
            )

    async def stream(self, conversation: Conversation) -> AsyncGenerator[StreamChunk, None]:
        """Stream from the HuggingFace inference endpoint.

        Uses the TGI-compatible SSE streaming format when ``use_chat_format``
        is *True*.  For raw mode, falls back to a non-streaming query and
        yields the result in a single chunk.
        """
        if not self._use_chat_format:
            # Fallback: no true streaming for raw endpoints
            response = await self.query(conversation)
            yield StreamChunk(
                delta=response.content,
                finish_reason=response.finish_reason,
                index=0,
            )
            return

        url = self._chat_url()
        payload = self._build_payload_chat(conversation, stream=True)
        headers = self._merge_headers()

        async with self._http.stream("POST", url, json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                body_bytes = await resp.aread()
                try:
                    body = json.loads(body_bytes)
                except Exception:
                    body = {"error": body_bytes.decode(errors="replace")}
                raise _classify_http_error(resp.status_code, body, resp.headers)

            self._parse_rate_limits(resp.headers)
            buffer = ""
            async for raw_bytes in resp.aiter_bytes():
                buffer += raw_bytes.decode(errors="replace")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        return
                    try:
                        chunk_data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk_data.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content_piece = delta.get("content", "")
                    fr_raw = choices[0].get("finish_reason")
                    fr = FinishReason.from_openai(fr_raw) if fr_raw else None
                    yield StreamChunk(
                        delta=content_piece or "",
                        finish_reason=fr,
                        index=choices[0].get("index", 0),
                    )

    def count_tokens(self, text: str) -> int:
        """Token estimate for HuggingFace models using a word-piece heuristic.

        Most HF models use some variant of BPE/SentencePiece.  This
        approximation splits on whitespace and punctuation, then applies a
        1.2x multiplier.
        """
        if not text:
            return 0
        parts = _PUNCTUATION_RE.split(text)
        parts = [p for p in parts if p]
        return max(1, int(math.ceil(len(parts) * 1.2)))


# ---------------------------------------------------------------------------
# Mock client (for testing / CABER behavioural audits)
# ---------------------------------------------------------------------------


class MockClient(ModelClient):
    """Deterministic mock client for testing and offline auditing.

    Maps input patterns to canned responses.  Records every query for later
    inspection, and can optionally inject simulated latency and random
    failures.
    """

    def __init__(
        self,
        response_map: Optional[Dict[str, str]] = None,
        default_response: str = "This is a mock response.",
        *,
        latency_ms: float = 0.0,
        failure_rate: float = 0.0,
        model_name: str = "mock-model",
    ) -> None:
        """Create a MockClient.

        Parameters
        ----------
        response_map:
            Mapping of substring patterns to response text.  The *first*
            pattern that appears (as a substring) in the last user message
            wins.
        default_response:
            Fallback response when no pattern matches.
        latency_ms:
            Simulated latency injected via ``asyncio.sleep``.
        failure_rate:
            Probability in ``[0, 1]`` that a query raises a
            :class:`ServerError` to simulate transient failures.
        model_name:
            Model name reported in responses.
        """
        config = ModelConfig(model_name=model_name, api_key="mock-key")
        super().__init__(config)
        self._response_map: Dict[str, str] = dict(response_map or {})
        self._default_response = default_response
        self._latency_ms = latency_ms
        self._failure_rate = failure_rate
        self._query_count: int = 0
        self._query_log: List[Tuple[Conversation, ModelResponse]] = []
        self._stream_chunk_delay_ms: float = 10.0

    # -- public helpers -------------------------------------------------------

    def set_response(self, pattern: str, response: str) -> None:
        """Register or update a pattern → response mapping."""
        self._response_map[pattern] = response

    def remove_response(self, pattern: str) -> bool:
        """Remove a pattern mapping.  Returns *True* if it existed."""
        return self._response_map.pop(pattern, None) is not None

    def set_failure_rate(self, rate: float) -> None:
        """Update the simulated failure rate (0.0–1.0)."""
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"failure_rate must be in [0, 1], got {rate}")
        self._failure_rate = rate

    def set_latency(self, latency_ms: float) -> None:
        """Update the simulated latency in milliseconds."""
        if latency_ms < 0:
            raise ValueError(f"latency_ms must be >= 0, got {latency_ms}")
        self._latency_ms = latency_ms

    def get_query_log(self) -> List[Tuple[Conversation, ModelResponse]]:
        """Return the full list of ``(conversation, response)`` pairs."""
        return list(self._query_log)

    @property
    def query_count(self) -> int:
        """Number of queries processed (including failures)."""
        return self._query_count

    def reset(self) -> None:
        """Clear the query log and reset counters."""
        self._query_count = 0
        self._query_log.clear()
        self._total_requests = 0
        self._total_tokens_used = 0
        self._cumulative_usage = TokenUsage()

    # -- internal -------------------------------------------------------------

    def _match_response(self, conversation: Conversation) -> str:
        """Find the first matching response for the conversation."""
        user_msg = conversation.last_user_message() or ""
        for pattern, response in self._response_map.items():
            if pattern in user_msg:
                return response
        return self._default_response

    def _maybe_fail(self) -> None:
        """Raise a ServerError with probability ``failure_rate``."""
        if self._failure_rate > 0 and random.random() < self._failure_rate:
            raise ServerError(
                "Simulated server error",
                status_code=500,
                raw={"error": "mock_failure"},
            )

    def _build_headers(self) -> Dict[str, str]:
        """Mock headers (no real auth needed)."""
        return {"Content-Type": "application/json"}

    # -- query / stream -------------------------------------------------------

    async def query(self, conversation: Conversation) -> ModelResponse:
        """Return a mock response based on pattern matching.

        Simulates latency and optional random failures.  Records every
        query in the internal log.
        """
        self._query_count += 1
        self._maybe_fail()

        if self._latency_ms > 0:
            await asyncio.sleep(self._latency_ms / 1000.0)

        content = self._match_response(conversation)
        prompt_tokens = self.count_conversation_tokens(conversation)
        completion_tokens = self.count_tokens(content)
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        self._record_usage(usage)

        response = ModelResponse(
            content=content,
            finish_reason=FinishReason.STOP,
            token_usage=usage,
            latency_ms=self._latency_ms,
            model=self.config.model_name,
        )
        self._query_log.append((conversation, response))
        return response

    async def stream(self, conversation: Conversation) -> AsyncGenerator[StreamChunk, None]:
        """Yield mock stream chunks, one word at a time.

        Simulates realistic streaming by introducing a short delay between
        chunks.
        """
        self._query_count += 1
        self._maybe_fail()

        if self._latency_ms > 0:
            await asyncio.sleep(self._latency_ms / 1000.0)

        content = self._match_response(conversation)
        words = content.split()
        for i, word in enumerate(words):
            is_last = i == len(words) - 1
            separator = "" if is_last else " "
            yield StreamChunk(
                delta=word + separator,
                finish_reason=FinishReason.STOP if is_last else None,
                index=0,
            )
            if self._stream_chunk_delay_ms > 0:
                await asyncio.sleep(self._stream_chunk_delay_ms / 1000.0)

    def count_tokens(self, text: str) -> int:
        """Simple word-count based token estimator for the mock client."""
        if not text:
            return 0
        return max(1, len(text.split()))


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


_PROVIDER_REGISTRY: Dict[str, type] = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
    "huggingface": HuggingFaceClient,
    "hf": HuggingFaceClient,
    "mock": MockClient,
}


def create_client(provider: str, **kwargs: Any) -> ModelClient:
    """Factory: instantiate a :class:`ModelClient` by provider name.

    Parameters
    ----------
    provider:
        One of ``"openai"``, ``"anthropic"``, ``"huggingface"`` /
        ``"hf"``, or ``"mock"``.
    **kwargs:
        Forwarded to the provider's ``__init__``.

    Returns
    -------
    ModelClient
        An instance of the appropriate client subclass.

    Raises
    ------
    ValueError
        If *provider* is not recognised.

    Examples
    --------
    >>> client = create_client("openai", api_key="sk-...", model="gpt-4o")
    >>> client = create_client("mock", default_response="hello")
    """
    key = provider.lower().strip()
    cls = _PROVIDER_REGISTRY.get(key)
    if cls is None:
        supported = ", ".join(sorted(_PROVIDER_REGISTRY.keys()))
        raise ValueError(
            f"Unknown provider {provider!r}. Supported providers: {supported}"
        )

    # MockClient has a different signature (no required api_key)
    if cls is MockClient:
        # Filter out kwargs that MockClient doesn't expect
        mock_kwargs: Dict[str, Any] = {}
        accepted = {
            "response_map",
            "default_response",
            "latency_ms",
            "failure_rate",
            "model_name",
        }
        for k, v in kwargs.items():
            if k in accepted:
                mock_kwargs[k] = v
        return cls(**mock_kwargs)

    # HuggingFaceClient requires model_id instead of model
    if cls is HuggingFaceClient:
        if "model" in kwargs and "model_id" not in kwargs:
            kwargs["model_id"] = kwargs.pop("model")
        elif "model_id" not in kwargs:
            raise ValueError("HuggingFaceClient requires 'model_id' (or 'model') parameter.")

    return cls(**kwargs)


def register_provider(name: str, cls: type) -> None:
    """Register a custom provider class in the factory.

    Parameters
    ----------
    name:
        Short name for the provider (used with :func:`create_client`).
    cls:
        A subclass of :class:`ModelClient`.
    """
    if not issubclass(cls, ModelClient):
        raise TypeError(f"{cls} is not a subclass of ModelClient")
    _PROVIDER_REGISTRY[name.lower().strip()] = cls


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------


def estimate_cost(model: str, usage: TokenUsage) -> float:
    """Estimate the USD cost of a query given the model and token usage.

    Uses the :data:`MODEL_COST_TABLE` lookup.  Returns ``0.0`` for
    unknown models.

    Parameters
    ----------
    model:
        Model identifier string (e.g. ``"gpt-4o"``).
    usage:
        Token usage from the response.

    Returns
    -------
    float
        Estimated cost in USD.

    Examples
    --------
    >>> usage = TokenUsage(prompt_tokens=1000, completion_tokens=500)
    >>> estimate_cost("gpt-4o", usage)
    0.0125
    """
    costs = MODEL_COST_TABLE.get(model)
    if costs is None:
        # Try prefix matching for versioned model names
        for known_model, known_costs in MODEL_COST_TABLE.items():
            if model.startswith(known_model):
                costs = known_costs
                break
    if costs is None:
        logger.warning("No cost data for model %r; returning 0.0", model)
        return 0.0

    input_cost_per_1k, output_cost_per_1k = costs
    prompt_cost = (usage.prompt_tokens / 1000.0) * input_cost_per_1k
    completion_cost = (usage.completion_tokens / 1000.0) * output_cost_per_1k
    return prompt_cost + completion_cost


def estimate_batch_cost(model: str, responses: Sequence[ModelResponse]) -> float:
    """Estimate total cost for a batch of responses.

    Parameters
    ----------
    model:
        Model identifier.
    responses:
        Sequence of model responses with token usage data.

    Returns
    -------
    float
        Total estimated cost in USD.
    """
    total = 0.0
    for resp in responses:
        total += estimate_cost(model, resp.token_usage)
    return total


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def merge_conversations(convs: List[Conversation]) -> Conversation:
    """Merge multiple conversations into a single one.

    Uses the system prompt from the *first* conversation that has one.
    Messages are concatenated in order.  Metadata dicts are merged
    left-to-right with later values overriding earlier ones.
    """
    merged_messages: List[Message] = []
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = {}

    for conv in convs:
        if system_prompt is None and conv.system_prompt:
            system_prompt = conv.system_prompt
        merged_messages.extend(conv.messages)
        metadata.update(conv.metadata)

    return Conversation(
        messages=merged_messages,
        system_prompt=system_prompt,
        metadata=metadata,
    )


def conversation_from_prompt(prompt: str, *, system: Optional[str] = None) -> Conversation:
    """Create a single-turn conversation from a user prompt.

    Parameters
    ----------
    prompt:
        User message text.
    system:
        Optional system prompt.
    """
    conv = Conversation(system_prompt=system)
    conv.add_user(prompt)
    return conv


def format_conversation(conversation: Conversation) -> str:
    """Pretty-print a conversation for debugging / logging.

    Returns a multi-line string with role labels.
    """
    lines: List[str] = []
    if conversation.system_prompt:
        lines.append(f"[system] {conversation.system_prompt}")
    for msg in conversation.messages:
        label = msg.role.value.upper()
        name_suffix = f" ({msg.name})" if msg.name else ""
        lines.append(f"[{label}{name_suffix}] {msg.content}")
    return "\n".join(lines)


def aggregate_usage(responses: Sequence[ModelResponse]) -> TokenUsage:
    """Sum token usage across a sequence of responses.

    Parameters
    ----------
    responses:
        Responses whose usage should be aggregated.

    Returns
    -------
    TokenUsage
        Combined token counts.
    """
    total = TokenUsage()
    for resp in responses:
        total = total + resp.token_usage
    return total


async def collect_stream(stream_gen: AsyncGenerator[StreamChunk, None]) -> Tuple[str, FinishReason]:
    """Consume a stream generator and return the full text plus finish reason.

    Parameters
    ----------
    stream_gen:
        Async generator of :class:`StreamChunk` objects.

    Returns
    -------
    tuple[str, FinishReason]
        Concatenated text and the final finish reason.
    """
    parts: List[str] = []
    finish = FinishReason.UNKNOWN
    async for chunk in stream_gen:
        parts.append(chunk.delta)
        if chunk.finish_reason is not None:
            finish = chunk.finish_reason
    return "".join(parts), finish


# ---------------------------------------------------------------------------
# Conversation builder (fluent API)
# ---------------------------------------------------------------------------


class ConversationBuilder:
    """Fluent builder for constructing :class:`Conversation` objects.

    Example
    -------
    >>> conv = (
    ...     ConversationBuilder()
    ...     .system("You are a helpful assistant.")
    ...     .user("Hello!")
    ...     .assistant("Hi there!")
    ...     .user("How are you?")
    ...     .with_metadata("audit_id", "abc-123")
    ...     .build()
    ... )
    """

    def __init__(self) -> None:
        """Initialise an empty builder."""
        self._system_prompt: Optional[str] = None
        self._messages: List[Message] = []
        self._metadata: Dict[str, Any] = {}

    def system(self, content: str) -> "ConversationBuilder":
        """Set the system prompt."""
        self._system_prompt = content
        return self

    def user(self, content: str, *, name: Optional[str] = None) -> "ConversationBuilder":
        """Append a user message."""
        self._messages.append(Message(role=MessageRole.USER, content=content, name=name))
        return self

    def assistant(self, content: str) -> "ConversationBuilder":
        """Append an assistant message."""
        self._messages.append(Message(role=MessageRole.ASSISTANT, content=content))
        return self

    def message(self, role: MessageRole, content: str, *, name: Optional[str] = None) -> "ConversationBuilder":
        """Append a message with an arbitrary role."""
        self._messages.append(Message(role=role, content=content, name=name))
        return self

    def with_metadata(self, key: str, value: Any) -> "ConversationBuilder":
        """Add a metadata key-value pair."""
        self._metadata[key] = value
        return self

    def build(self) -> Conversation:
        """Construct and return the :class:`Conversation`."""
        return Conversation(
            messages=list(self._messages),
            system_prompt=self._system_prompt,
            metadata=dict(self._metadata),
        )


# ---------------------------------------------------------------------------
# Multi-provider fan-out
# ---------------------------------------------------------------------------


async def fan_out_query(
    clients: List[ModelClient],
    conversation: Conversation,
    *,
    return_first: bool = False,
) -> List[ModelResponse]:
    """Query multiple clients in parallel with the same conversation.

    Parameters
    ----------
    clients:
        Model clients to query concurrently.
    conversation:
        The conversation to send to every client.
    return_first:
        If *True*, cancel remaining tasks once the first result arrives
        and return a single-element list.

    Returns
    -------
    list[ModelResponse]
        Responses in the same order as *clients* (unless *return_first*).
    """
    if not clients:
        return []

    if return_first:
        done: Optional[ModelResponse] = None
        tasks = [asyncio.create_task(c.query_with_retry(conversation)) for c in clients]

        finished, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        for t in finished:
            if t.exception() is None:
                done = t.result()
                break
        if done is None:
            # All finished tasks raised — re-raise the first
            for t in finished:
                exc = t.exception()
                if exc is not None:
                    raise exc
        return [done] if done is not None else []

    tasks = [asyncio.create_task(c.query_with_retry(conversation)) for c in clients]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    responses: List[ModelResponse] = []
    first_error: Optional[Exception] = None
    for r in results:
        if isinstance(r, Exception):
            if first_error is None:
                first_error = r
            logger.error("fan_out_query: one client failed: %s", r)
        else:
            responses.append(r)

    if not responses and first_error is not None:
        raise first_error
    return responses


# ---------------------------------------------------------------------------
# Token-budget aware conversation trimming
# ---------------------------------------------------------------------------


def trim_conversation_to_budget(
    conversation: Conversation,
    max_tokens: int,
    *,
    token_counter: Optional[Callable[[str], int]] = None,
    preserve_last_n: int = 1,
) -> Conversation:
    """Trim a conversation so its estimated tokens fit within a budget.

    Removes the *oldest* messages first, but always preserves at least
    the last *preserve_last_n* user messages.

    Parameters
    ----------
    conversation:
        The conversation to trim.
    max_tokens:
        Maximum allowed token count.
    token_counter:
        Optional custom token-counting function.  Defaults to a simple
        whitespace splitter.
    preserve_last_n:
        Minimum number of trailing messages to keep.

    Returns
    -------
    Conversation
        A new conversation that fits within the budget.
    """
    if token_counter is None:
        def token_counter(text: str) -> int:
            return max(1, int(math.ceil(len(text.split()) / 0.75)))

    system_tokens = 0
    if conversation.system_prompt:
        system_tokens = token_counter(conversation.system_prompt) + 4

    messages = list(conversation.messages)
    preserve_last_n = max(1, min(preserve_last_n, len(messages)))

    # Calculate tokens for all messages
    msg_tokens = [token_counter(m.content) + 4 for m in messages]
    total = system_tokens + sum(msg_tokens)

    if total <= max_tokens:
        return conversation.copy_with_messages(list(messages))

    # Remove from the front until we fit (or hit the preserve limit)
    removable = len(messages) - preserve_last_n
    idx = 0
    while total > max_tokens and idx < removable:
        total -= msg_tokens[idx]
        idx += 1

    kept = messages[idx:]
    return conversation.copy_with_messages(kept)


# ---------------------------------------------------------------------------
# __main__ : self-test suite
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys
    import traceback

    _pass = 0
    _fail = 0

    def _assert(condition: bool, label: str) -> None:
        global _pass, _fail
        if condition:
            _pass += 1
            print(f"  ✓ {label}")
        else:
            _fail += 1
            print(f"  ✗ {label}")

    async def _run_tests() -> None:
        global _pass, _fail

        # ==================================================================
        # 1. Pydantic model basics
        # ==================================================================
        print("\n── Pydantic models ──")

        cfg = ModelConfig(model_name="test-model", api_key=" sk-abc ", temperature=0.5)
        _assert(cfg.api_key == "sk-abc", "ModelConfig strips api_key whitespace")
        _assert(cfg.temperature == 0.5, "ModelConfig stores temperature")
        _assert(cfg.max_tokens == DEFAULT_MAX_TOKENS, "ModelConfig default max_tokens")

        msg = Message(role=MessageRole.USER, content="Hello world")
        _assert(msg.to_openai_dict() == {"role": "user", "content": "Hello world"}, "Message.to_openai_dict")
        _assert(msg.token_estimate() > 0, "Message.token_estimate > 0")

        msg_named = Message(role=MessageRole.USER, content="Hi", name="alice")
        d = msg_named.to_openai_dict()
        _assert(d.get("name") == "alice", "Message.to_openai_dict includes name")

        usage1 = TokenUsage(prompt_tokens=100, completion_tokens=50)
        _assert(usage1.total_tokens == 150, "TokenUsage auto-computes total")
        usage2 = TokenUsage(prompt_tokens=200, completion_tokens=100)
        combined = usage1 + usage2
        _assert(combined.prompt_tokens == 300, "TokenUsage addition (prompt)")
        _assert(combined.total_tokens == 450, "TokenUsage addition (total)")

        resp = ModelResponse(content="Hi!", finish_reason=FinishReason.STOP, model="test")
        _assert(resp.is_complete(), "ModelResponse.is_complete for STOP")
        _assert(not resp.is_truncated(), "ModelResponse.is_truncated false for STOP")

        resp_trunc = ModelResponse(content="...", finish_reason=FinishReason.LENGTH, model="test")
        _assert(resp_trunc.is_truncated(), "ModelResponse.is_truncated for LENGTH")

        chunk = StreamChunk(delta="hello", finish_reason=None, index=0)
        _assert(not chunk.is_final(), "StreamChunk.is_final false when no finish_reason")
        chunk_final = StreamChunk(delta="", finish_reason=FinishReason.STOP, index=0)
        _assert(chunk_final.is_final(), "StreamChunk.is_final true when STOP")

        rl = RateLimitInfo(requests_remaining=0, tokens_remaining=100)
        _assert(rl.is_exhausted(), "RateLimitInfo.is_exhausted when requests=0")
        rl2 = RateLimitInfo(requests_remaining=5, tokens_remaining=500)
        _assert(not rl2.is_exhausted(), "RateLimitInfo not exhausted when buckets > 0")

        # ==================================================================
        # 2. Conversation management
        # ==================================================================
        print("\n── Conversation management ──")

        conv = Conversation()
        conv.add_system("You are a bot.")
        conv.add_user("Hi!")
        conv.add_assistant("Hello!")
        conv.add_user("How are you?")

        _assert(conv.system_prompt == "You are a bot.", "Conversation.add_system")
        _assert(conv.message_count() == 3, "Conversation.message_count (excludes system)")
        _assert(conv.last_user_message() == "How are you?", "Conversation.last_user_message")
        _assert(conv.last_assistant_message() == "Hello!", "Conversation.last_assistant_message")

        oai_msgs = conv.to_openai_messages()
        _assert(oai_msgs[0]["role"] == "system", "OpenAI messages start with system")
        _assert(len(oai_msgs) == 4, "OpenAI messages count (system + 3)")

        anth_msgs = conv.to_anthropic_messages()
        _assert(all(m["role"] != "system" for m in anth_msgs), "Anthropic messages exclude system role")
        _assert(len(anth_msgs) == 3, "Anthropic messages count")

        _assert(conv.total_content_length() > 0, "Conversation.total_content_length > 0")

        truncated = conv.truncate(2)
        _assert(truncated.message_count() == 2, "Conversation.truncate keeps last N")
        _assert(truncated.system_prompt == conv.system_prompt, "Truncated keeps system prompt")

        # ConversationBuilder
        conv_b = (
            ConversationBuilder()
            .system("System instruction.")
            .user("Question?")
            .assistant("Answer.")
            .with_metadata("audit_id", "xyz")
            .build()
        )
        _assert(conv_b.system_prompt == "System instruction.", "ConversationBuilder system")
        _assert(conv_b.message_count() == 2, "ConversationBuilder message count")
        _assert(conv_b.metadata.get("audit_id") == "xyz", "ConversationBuilder metadata")

        # conversation_from_prompt
        simple = conversation_from_prompt("Hello", system="Be helpful")
        _assert(simple.system_prompt == "Be helpful", "conversation_from_prompt system")
        _assert(simple.last_user_message() == "Hello", "conversation_from_prompt user msg")

        # format_conversation
        formatted = format_conversation(conv)
        _assert("[system]" in formatted, "format_conversation includes system label")
        _assert("[USER]" in formatted, "format_conversation includes USER label")

        # merge_conversations
        conv_a = Conversation(system_prompt="First system")
        conv_a.add_user("msg A")
        conv_c = Conversation()
        conv_c.add_user("msg B")
        merged = merge_conversations([conv_a, conv_c])
        _assert(merged.system_prompt == "First system", "merge_conversations keeps first system prompt")
        _assert(merged.message_count() == 2, "merge_conversations merges messages")

        # ==================================================================
        # 3. Token counting
        # ==================================================================
        print("\n── Token counting ──")

        mock = MockClient(default_response="word " * 10)

        # Base class whitespace estimator
        base_count = mock.count_tokens("one two three four")
        _assert(base_count >= 4, "MockClient token count >= word count")
        _assert(mock.count_tokens("") == 0, "Empty string → 0 tokens")

        conv_tc = Conversation(system_prompt="System prompt here")
        conv_tc.add_user("Hello world this is a test")
        tc_total = mock.count_conversation_tokens(conv_tc)
        _assert(tc_total > 0, "count_conversation_tokens > 0")
        # Should include both system and user tokens plus overhead
        _assert(tc_total > mock.count_tokens("Hello world this is a test"), "Conversation tokens > single msg tokens")

        # OpenAI improved tokeniser
        oai = OpenAIClient(api_key="fake-key", model="gpt-4o")
        oai_count = oai.count_tokens("Hello, world! This is a test.")
        _assert(oai_count > 0, "OpenAI count_tokens > 0")
        _assert(oai_count != mock.count_tokens("Hello, world! This is a test."), "OpenAI uses different heuristic")
        await oai.close()

        # Anthropic tokeniser
        anth = AnthropicClient(api_key="fake-key")
        anth_count = anth.count_tokens("Hello, world! This is a test.")
        _assert(anth_count > 0, "Anthropic count_tokens > 0")
        await anth.close()

        # ==================================================================
        # 4. MockClient query + response matching
        # ==================================================================
        print("\n── MockClient query & matching ──")

        response_map = {
            "hello": "Hi there!",
            "weather": "It's sunny today.",
            "math": "2 + 2 = 4",
        }
        mock = MockClient(response_map=response_map, default_response="I don't understand.")

        # Test pattern matching
        conv1 = conversation_from_prompt("Say hello to me")
        r1 = await mock.query(conv1)
        _assert(r1.content == "Hi there!", "Pattern match 'hello'")
        _assert(r1.is_complete(), "Mock response has STOP finish_reason")
        _assert(r1.token_usage.prompt_tokens > 0, "Mock tracks prompt tokens")
        _assert(r1.token_usage.completion_tokens > 0, "Mock tracks completion tokens")

        conv2 = conversation_from_prompt("What's the weather like?")
        r2 = await mock.query(conv2)
        _assert(r2.content == "It's sunny today.", "Pattern match 'weather'")

        conv3 = conversation_from_prompt("Tell me a joke")
        r3 = await mock.query(conv3)
        _assert(r3.content == "I don't understand.", "Default response when no match")

        # Query log
        log = mock.get_query_log()
        _assert(len(log) == 3, "Query log has 3 entries")
        _assert(mock.query_count == 3, "Query count is 3")
        _assert(mock.total_requests == 3, "Total requests is 3")
        _assert(mock.cumulative_usage.total_tokens > 0, "Cumulative usage tracked")

        # set_response
        mock.set_response("joke", "Why did the chicken cross the road?")
        conv4 = conversation_from_prompt("Tell me a joke")
        r4 = await mock.query(conv4)
        _assert(r4.content == "Why did the chicken cross the road?", "set_response works")

        # Reset
        mock.reset()
        _assert(mock.query_count == 0, "Reset clears query_count")
        _assert(len(mock.get_query_log()) == 0, "Reset clears query_log")
        _assert(mock.total_requests == 0, "Reset clears total_requests")

        # ==================================================================
        # 5. MockClient streaming
        # ==================================================================
        print("\n── MockClient streaming ──")

        mock_stream = MockClient(
            response_map={"stream": "one two three four five"},
            default_response="default",
        )
        mock_stream._stream_chunk_delay_ms = 0  # speed up test

        conv_s = conversation_from_prompt("stream test")
        chunks: List[StreamChunk] = []
        async for chunk in mock_stream.stream(conv_s):
            chunks.append(chunk)

        _assert(len(chunks) == 5, f"Stream yields 5 chunks (got {len(chunks)})")
        full_text = "".join(c.delta for c in chunks)
        _assert(full_text.strip() == "one two three four five", "Stream text matches")
        _assert(chunks[-1].is_final(), "Last stream chunk is final")
        _assert(not chunks[0].is_final(), "First stream chunk is not final")

        # collect_stream helper
        mock_stream2 = MockClient(default_response="alpha beta gamma")
        mock_stream2._stream_chunk_delay_ms = 0
        conv_cs = conversation_from_prompt("anything")
        text, fr = await collect_stream(mock_stream2.stream(conv_cs))
        _assert(text.strip() == "alpha beta gamma", "collect_stream concatenates text")
        _assert(fr == FinishReason.STOP, "collect_stream returns finish reason")

        # ==================================================================
        # 6. Batch query
        # ==================================================================
        print("\n── Batch query ──")

        mock_batch = MockClient(
            response_map={"q1": "a1", "q2": "a2", "q3": "a3"},
            default_response="default",
        )

        conversations = [
            conversation_from_prompt("q1"),
            conversation_from_prompt("q2"),
            conversation_from_prompt("q3"),
            conversation_from_prompt("unknown"),
        ]
        results = await mock_batch.batch_query(conversations, max_concurrent=2)
        _assert(len(results) == 4, "Batch returns 4 results")
        _assert(results[0].content == "a1", "Batch result 0")
        _assert(results[1].content == "a2", "Batch result 1")
        _assert(results[2].content == "a3", "Batch result 2")
        _assert(results[3].content == "default", "Batch result 3 (default)")
        _assert(mock_batch.total_requests == 4, "Batch total_requests after 4 queries")

        # Empty batch
        empty_results = await mock_batch.batch_query([], max_concurrent=5)
        _assert(len(empty_results) == 0, "Empty batch returns empty list")

        # ==================================================================
        # 7. Factory function
        # ==================================================================
        print("\n── Factory function ──")

        mock_f = create_client("mock", default_response="factory response")
        _assert(isinstance(mock_f, MockClient), "create_client('mock') returns MockClient")
        conv_f = conversation_from_prompt("test")
        rf = await mock_f.query(conv_f)
        _assert(rf.content == "factory response", "Factory-created mock responds correctly")
        await mock_f.close()

        oai_f = create_client("openai", api_key="sk-test", model="gpt-4o-mini")
        _assert(isinstance(oai_f, OpenAIClient), "create_client('openai') returns OpenAIClient")
        _assert(oai_f.config.model_name == "gpt-4o-mini", "Factory passes model name")
        await oai_f.close()

        anth_f = create_client("anthropic", api_key="sk-ant-test")
        _assert(isinstance(anth_f, AnthropicClient), "create_client('anthropic') returns AnthropicClient")
        await anth_f.close()

        hf_f = create_client("huggingface", api_key="hf-test", model="meta-llama/Llama-2-7b")
        _assert(isinstance(hf_f, HuggingFaceClient), "create_client('huggingface') returns HuggingFaceClient")
        await hf_f.close()

        try:
            create_client("nonexistent")
            _assert(False, "Factory raises ValueError for unknown provider")
        except ValueError:
            _assert(True, "Factory raises ValueError for unknown provider")

        # ==================================================================
        # 8. Cost estimation
        # ==================================================================
        print("\n── Cost estimation ──")

        usage_cost = TokenUsage(prompt_tokens=1000, completion_tokens=500)
        cost_4o = estimate_cost("gpt-4o", usage_cost)
        expected_4o = (1000 / 1000.0) * 0.005 + (500 / 1000.0) * 0.015
        _assert(abs(cost_4o - expected_4o) < 1e-6, f"GPT-4o cost = {cost_4o:.6f} (expected {expected_4o:.6f})")

        cost_unknown = estimate_cost("unknown-model-xyz", usage_cost)
        _assert(cost_unknown == 0.0, "Unknown model returns 0.0 cost")

        cost_claude = estimate_cost("claude-3-5-sonnet-20241022", usage_cost)
        _assert(cost_claude > 0, f"Claude cost = {cost_claude:.6f}")

        # estimate_batch_cost
        responses_batch = [
            ModelResponse(content="a", model="gpt-4o", token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50)),
            ModelResponse(content="b", model="gpt-4o", token_usage=TokenUsage(prompt_tokens=200, completion_tokens=100)),
        ]
        batch_cost = estimate_batch_cost("gpt-4o", responses_batch)
        _assert(batch_cost > 0, "Batch cost > 0")

        # ==================================================================
        # 9. Retry logic (with mock failure)
        # ==================================================================
        print("\n── Retry logic ──")

        mock_retry = MockClient(default_response="success", failure_rate=0.0)
        mock_retry.config = ModelConfig(model_name="mock", api_key="mock", max_retries=2)
        conv_retry = conversation_from_prompt("retry test")
        r_retry = await mock_retry.query_with_retry(conv_retry)
        _assert(r_retry.content == "success", "query_with_retry succeeds without failures")

        # ==================================================================
        # 10. Context manager
        # ==================================================================
        print("\n── Context manager ──")

        async with MockClient(default_response="ctx") as ctx_client:
            _assert(not ctx_client.is_closed, "Client open inside context")
            r_ctx = await ctx_client.query(conversation_from_prompt("test"))
            _assert(r_ctx.content == "ctx", "Query works inside context")
        _assert(ctx_client.is_closed, "Client closed after context exit")

        # ==================================================================
        # 11. Finish reason mapping
        # ==================================================================
        print("\n── Finish reason mapping ──")

        _assert(FinishReason.from_openai("stop") == FinishReason.STOP, "OpenAI 'stop' → STOP")
        _assert(FinishReason.from_openai("length") == FinishReason.LENGTH, "OpenAI 'length' → LENGTH")
        _assert(FinishReason.from_openai(None) == FinishReason.UNKNOWN, "OpenAI None → UNKNOWN")
        _assert(FinishReason.from_anthropic("end_turn") == FinishReason.STOP, "Anthropic 'end_turn' → STOP")
        _assert(FinishReason.from_anthropic("max_tokens") == FinishReason.LENGTH, "Anthropic 'max_tokens' → LENGTH")

        # ==================================================================
        # 12. Rate limit parsing
        # ==================================================================
        print("\n── Rate limit parsing ──")

        mock_rl = MockClient()
        rl_info = mock_rl._parse_rate_limits({
            "x-ratelimit-remaining-requests": "42",
            "x-ratelimit-remaining-tokens": "10000",
        })
        _assert(rl_info is not None, "Rate limit info parsed")
        _assert(rl_info.requests_remaining == 42, "Remaining requests = 42")
        _assert(rl_info.tokens_remaining == 10000, "Remaining tokens = 10000")

        rl_none = mock_rl._parse_rate_limits({})
        _assert(rl_none is None, "No rate limit headers → None")

        # ==================================================================
        # 13. Trim conversation to budget
        # ==================================================================
        print("\n── Trim conversation to budget ──")

        conv_trim = Conversation(system_prompt="sys")
        for i in range(10):
            conv_trim.add_user(f"message {i} " + "word " * 50)

        trimmed = trim_conversation_to_budget(conv_trim, max_tokens=100, preserve_last_n=2)
        _assert(trimmed.message_count() <= conv_trim.message_count(), "Trimmed has fewer messages")
        _assert(trimmed.message_count() >= 2, "Trimmed preserves at least 2 messages")

        no_trim = trim_conversation_to_budget(
            conversation_from_prompt("short"), max_tokens=10000
        )
        _assert(no_trim.message_count() == 1, "No trimming when under budget")

        # ==================================================================
        # 14. Exception hierarchy
        # ==================================================================
        print("\n── Exception hierarchy ──")

        _assert(issubclass(AuthenticationError, ModelClientError), "AuthenticationError is ModelClientError")
        _assert(issubclass(RateLimitError, ModelClientError), "RateLimitError is ModelClientError")
        _assert(issubclass(ContentFilterError, ModelClientError), "ContentFilterError is ModelClientError")
        _assert(issubclass(InvalidRequestError, ModelClientError), "InvalidRequestError is ModelClientError")
        _assert(issubclass(ServerError, ModelClientError), "ServerError is ModelClientError")

        rl_exc = RateLimitError("too fast", retry_after=5.0)
        _assert(rl_exc.retry_after == 5.0, "RateLimitError stores retry_after")
        _assert(rl_exc.status_code == 429, "RateLimitError default status=429")

        # ==================================================================
        # 15. Provider registration
        # ==================================================================
        print("\n── Provider registration ──")

        class CustomClient(MockClient):
            pass

        register_provider("custom", CustomClient)
        custom = create_client("custom", default_response="custom!")
        _assert(isinstance(custom, CustomClient), "Custom provider registered and created")
        r_custom = await custom.query(conversation_from_prompt("test"))
        _assert(r_custom.content == "custom!", "Custom provider responds")
        await custom.close()

        try:
            register_provider("bad", str)  # type: ignore
            _assert(False, "register_provider rejects non-ModelClient subclass")
        except TypeError:
            _assert(True, "register_provider rejects non-ModelClient subclass")

        # ==================================================================
        # 16. Aggregate usage
        # ==================================================================
        print("\n── Aggregate usage ──")

        resps = [
            ModelResponse(content="a", token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5)),
            ModelResponse(content="b", token_usage=TokenUsage(prompt_tokens=20, completion_tokens=10)),
        ]
        agg = aggregate_usage(resps)
        _assert(agg.prompt_tokens == 30, "Aggregate prompt tokens")
        _assert(agg.completion_tokens == 15, "Aggregate completion tokens")
        _assert(agg.total_tokens == 45, "Aggregate total tokens")

        # ==================================================================
        # 17. Error classification
        # ==================================================================
        print("\n── Error classification ──")

        err_401 = _classify_http_error(401, {"error": {"message": "bad key"}}, httpx.Headers())
        _assert(isinstance(err_401, AuthenticationError), "401 → AuthenticationError")

        err_429 = _classify_http_error(429, {"error": {"message": "slow down"}}, httpx.Headers({"retry-after": "2"}))
        _assert(isinstance(err_429, RateLimitError), "429 → RateLimitError")
        _assert(err_429.retry_after == 2.0, "429 parses Retry-After header")

        err_400 = _classify_http_error(400, {"error": {"message": "bad request"}}, httpx.Headers())
        _assert(isinstance(err_400, InvalidRequestError), "400 → InvalidRequestError")

        err_500 = _classify_http_error(500, {"error": {"message": "internal"}}, httpx.Headers())
        _assert(isinstance(err_500, ServerError), "500 → ServerError")

        await mock_rl.close()

    # Run the tests
    print("=" * 60)
    print("CABER ModelClient — Self-Test Suite")
    print("=" * 60)

    try:
        asyncio.run(_run_tests())
    except Exception:
        traceback.print_exc()
        _fail += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {_pass} passed, {_fail} failed")
    print(f"{'=' * 60}")
    sys.exit(1 if _fail > 0 else 0)
