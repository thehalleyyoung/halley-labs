"""
Interface module — LLM API clients, query generation, and response parsing.

Provides:
    ModelClient      — Base class for LLM interaction
    OpenAIClient     — OpenAI API client (GPT-4o, GPT-4o-mini)
    AnthropicClient  — Anthropic API client (Claude)
    HuggingFaceClient — HuggingFace Inference API client
    MockClient       — Deterministic mock for testing
    QueryGenerator   — Query construction for PCL* learning
    ResponseParser   — Response classification and parsing
"""

from caber.interface.model_client import (
    ModelClient,
    OpenAIClient,
    AnthropicClient,
    HuggingFaceClient,
    MockClient,
)
from caber.interface.query_generator import QueryGenerator
from caber.interface.response_parser import ResponseParser

__all__ = [
    "ModelClient",
    "OpenAIClient",
    "AnthropicClient",
    "HuggingFaceClient",
    "MockClient",
    "QueryGenerator",
    "ResponseParser",
]
