"""
Pipeline integration for diverse LLM response generation.

Provides middleware and utilities for integrating diversity-aware
response selection into LangChain, LlamaIndex, and direct API calls
to OpenAI/Anthropic endpoints.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from .embedding import TextEmbedder, embed_texts
from .dpp import DPP, greedy_map
from .diversity_metrics import cosine_diversity, log_det_diversity
from .kernels import RBFKernel
from .utils import log_det_safe


# ---------------------------------------------------------------------------
# Core diversity selection used across all integrations
# ---------------------------------------------------------------------------

def _select_diverse_subset(
    texts: List[str],
    k: int,
    method: str = "dpp",
    embedder: Optional[TextEmbedder] = None,
    quality_scores: Optional[np.ndarray] = None,
) -> List[str]:
    """Select k diverse texts from a candidate pool.

    Parameters
    ----------
    texts : list of str
        Candidate responses.
    k : int
        Number of diverse responses to return.
    method : str
        Selection method: 'dpp', 'greedy_maxmin', or 'mmr'.
    embedder : TextEmbedder, optional
        Custom embedder; a default is created if *None*.
    quality_scores : np.ndarray, optional
        Per-candidate quality weights (higher is better).

    Returns
    -------
    list of str
        The *k* selected diverse responses.
    """
    if len(texts) <= k:
        return list(texts)

    if embedder is None:
        embedder = TextEmbedder(dim=64)
    embeddings = embedder.embed_batch(texts)

    if method == "dpp":
        return _select_dpp(texts, embeddings, k, quality_scores)
    elif method == "greedy_maxmin":
        return _select_greedy_maxmin(texts, embeddings, k)
    elif method == "mmr":
        return _select_mmr(texts, embeddings, k, quality_scores)
    else:
        raise ValueError(f"Unknown selection method: {method}")


def _select_dpp(
    texts: List[str],
    embeddings: np.ndarray,
    k: int,
    quality_scores: Optional[np.ndarray] = None,
) -> List[str]:
    """DPP-based diverse selection."""
    kernel = RBFKernel()
    S = kernel.gram_matrix(embeddings)
    if quality_scores is not None:
        q = np.asarray(quality_scores, dtype=float)
        q = q / (q.max() + 1e-12)
        Q = np.diag(q)
        S = Q @ S @ Q
    # Numerical stability
    S += np.eye(S.shape[0]) * 1e-6
    indices = greedy_map(S, k)
    return [texts[i] for i in indices]


def _select_greedy_maxmin(
    texts: List[str], embeddings: np.ndarray, k: int
) -> List[str]:
    """Greedy max-min distance selection."""
    n = len(texts)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.maximum(norms, 1e-12)
    sim = normed @ normed.T
    dist = 1.0 - sim

    selected: List[int] = [np.random.randint(n)]
    for _ in range(k - 1):
        min_dists = dist[selected].min(axis=0)
        min_dists[selected] = -np.inf
        selected.append(int(np.argmax(min_dists)))
    return [texts[i] for i in selected]


def _select_mmr(
    texts: List[str],
    embeddings: np.ndarray,
    k: int,
    quality_scores: Optional[np.ndarray] = None,
    lam: float = 0.5,
) -> List[str]:
    """Maximal Marginal Relevance selection."""
    n = len(texts)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.maximum(norms, 1e-12)
    sim = normed @ normed.T

    if quality_scores is None:
        quality_scores = np.ones(n)
    quality_scores = np.asarray(quality_scores, dtype=float)

    selected: List[int] = [int(np.argmax(quality_scores))]
    remaining = set(range(n)) - set(selected)

    for _ in range(k - 1):
        best_idx, best_score = -1, -np.inf
        for idx in remaining:
            relevance = quality_scores[idx]
            redundancy = max(sim[idx, s] for s in selected)
            mmr_score = lam * relevance - (1 - lam) * redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.discard(best_idx)

    return [texts[i] for i in selected]


# ---------------------------------------------------------------------------
# DiversityMiddleware — generic middleware for pipeline integration
# ---------------------------------------------------------------------------

class DiversityMiddleware:
    """Middleware that wraps an LLM call to add diversity-aware selection.

    Parameters
    ----------
    k : int
        Number of diverse outputs to keep.
    method : str
        Diversity selection method ('dpp', 'greedy_maxmin', 'mmr').
    overgenerate_factor : int
        How many candidates to generate per desired output.
    embedder : TextEmbedder, optional
        Custom text embedder.
    """

    def __init__(
        self,
        k: int = 5,
        method: str = "dpp",
        overgenerate_factor: int = 4,
        embedder: Optional[TextEmbedder] = None,
    ) -> None:
        self.k = k
        self.method = method
        self.overgenerate_factor = overgenerate_factor
        self.embedder = embedder or TextEmbedder(dim=64)
        self._history: List[List[str]] = []

    # -- public API ----------------------------------------------------------

    def wrap(self, generate_fn: Callable[..., List[str]]) -> Callable[..., List[str]]:
        """Return a wrapped version of *generate_fn* that applies diversity."""

        def wrapped(*args: Any, **kwargs: Any) -> List[str]:
            # Ask the underlying function for more candidates
            kwargs["n"] = self.k * self.overgenerate_factor
            candidates = generate_fn(*args, **kwargs)
            selected = self.select(candidates)
            return selected

        return wrapped

    def select(self, candidates: List[str]) -> List[str]:
        """Select *k* diverse responses from *candidates*."""
        selected = _select_diverse_subset(
            candidates, self.k, method=self.method, embedder=self.embedder
        )
        self._history.append(selected)
        return selected

    def diversity_score(self, texts: Optional[List[str]] = None) -> float:
        """Compute diversity of the most recent (or given) selection."""
        if texts is None:
            if not self._history:
                return 0.0
            texts = self._history[-1]
        embs = self.embedder.embed_batch(texts)
        return float(cosine_diversity(embs))

    def reset(self) -> None:
        self._history.clear()


# ---------------------------------------------------------------------------
# LangChain integration
# ---------------------------------------------------------------------------

def langchain_diverse_chain(
    llm: Any,
    prompt: str,
    k: int = 5,
    n: int = 20,
    method: str = "dpp",
    temperature: float = 1.0,
) -> List[str]:
    """Generate *k* diverse completions via a LangChain-compatible LLM.

    Parameters
    ----------
    llm : object
        A LangChain ``BaseLLM`` or ``BaseChatModel`` instance.  Must expose
        either ``generate`` or ``predict`` / ``invoke``.
    prompt : str
        The user prompt.
    k : int
        Number of diverse outputs.
    n : int
        Number of raw candidates to sample before selection.
    method : str
        Diversity method forwarded to ``_select_diverse_subset``.
    temperature : float
        Sampling temperature (set on ``llm`` if supported).

    Returns
    -------
    list of str
        *k* diverse completions.
    """
    # Try to set temperature on the model
    if hasattr(llm, "temperature"):
        llm.temperature = temperature

    candidates: List[str] = []

    # LangChain LLMs expose a ``generate`` batch method
    if hasattr(llm, "generate"):
        prompts_batch = [prompt] * n
        result = llm.generate(prompts_batch)
        for gen_list in result.generations:
            for gen in gen_list:
                text = gen.text if hasattr(gen, "text") else str(gen)
                candidates.append(text)
    elif hasattr(llm, "invoke"):
        for _ in range(n):
            resp = llm.invoke(prompt)
            text = resp.content if hasattr(resp, "content") else str(resp)
            candidates.append(text)
    elif hasattr(llm, "predict"):
        for _ in range(n):
            candidates.append(llm.predict(prompt))
    else:
        raise TypeError(
            "Unsupported LangChain LLM type: must have generate, invoke, or predict"
        )

    return _select_diverse_subset(candidates, k, method=method)


# ---------------------------------------------------------------------------
# LlamaIndex integration
# ---------------------------------------------------------------------------

def llamaindex_diverse_query(
    index: Any,
    query: str,
    k: int = 5,
    n: int = 20,
    method: str = "dpp",
) -> List[str]:
    """Query a LlamaIndex index and return *k* diverse results.

    Parameters
    ----------
    index : object
        A LlamaIndex index with a ``as_query_engine`` method.
    query : str
        The user query.
    k : int
        Desired diverse outputs.
    n : int
        Number of candidate queries to run (with varied phrasing).
    method : str
        Diversity selection method.

    Returns
    -------
    list of str
        *k* diverse response strings.
    """
    candidates: List[str] = []

    # Build a query engine — if the index supports similarity_top_k, set it
    engine_kwargs: Dict[str, Any] = {}
    engine_kwargs["similarity_top_k"] = max(n, 10)
    try:
        engine = index.as_query_engine(**engine_kwargs)
    except TypeError:
        engine = index.as_query_engine()

    # Generate candidate answers by varying query phrasing
    rephrasings = _rephrase_query(query, n)
    for q in rephrasings:
        try:
            resp = engine.query(q)
            text = str(resp) if not hasattr(resp, "response") else resp.response
            if text and text.strip():
                candidates.append(text.strip())
        except Exception:
            continue

    if not candidates:
        # Fallback: just query once
        resp = engine.query(query)
        text = str(resp) if not hasattr(resp, "response") else resp.response
        return [text]

    return _select_diverse_subset(candidates, k, method=method)


def _rephrase_query(query: str, n: int) -> List[str]:
    """Create *n* lightweight rephrasings of *query* for diversity."""
    prefixes = [
        "",
        "Explain: ",
        "Describe in detail: ",
        "Give an alternative perspective on: ",
        "Summarize: ",
        "List key points about: ",
        "What are different viewpoints on: ",
        "Provide a comprehensive answer to: ",
        "From a critical standpoint, ",
        "In simple terms, ",
    ]
    rephrasings: List[str] = []
    for i in range(n):
        prefix = prefixes[i % len(prefixes)]
        rephrasings.append(prefix + query)
    return rephrasings


# ---------------------------------------------------------------------------
# OpenAI direct integration
# ---------------------------------------------------------------------------

def openai_diverse_complete(
    prompt: str,
    model: str = "gpt-3.5-turbo",
    k: int = 5,
    n: int = 20,
    method: str = "dpp",
    temperature: float = 1.0,
    api_key: Optional[str] = None,
) -> List[str]:
    """Generate *k* diverse completions using the OpenAI API.

    Generates *n* candidates via the ``n`` parameter of the chat completions
    endpoint and selects the *k* most diverse.

    Parameters
    ----------
    prompt : str
        User prompt.
    model : str
        OpenAI model name.
    k, n : int
        Desired diverse outputs and candidate pool size.
    method : str
        Diversity selection method.
    temperature : float
        Sampling temperature.
    api_key : str, optional
        OpenAI API key; falls back to ``OPENAI_API_KEY`` env var.

    Returns
    -------
    list of str
    """
    try:
        import openai
    except ImportError:
        raise ImportError("pip install openai  is required for openai_diverse_complete")

    client_kwargs: Dict[str, Any] = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    client = openai.OpenAI(**client_kwargs)

    # The API supports up to n=128 in a single call
    batch_size = min(n, 128)
    candidates: List[str] = []

    while len(candidates) < n:
        this_n = min(batch_size, n - len(candidates))
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            n=this_n,
            temperature=temperature,
        )
        for choice in response.choices:
            candidates.append(choice.message.content or "")

    return _select_diverse_subset(candidates, k, method=method)


# ---------------------------------------------------------------------------
# Anthropic direct integration
# ---------------------------------------------------------------------------

def anthropic_diverse_complete(
    prompt: str,
    model: str = "claude-3-haiku-20240307",
    k: int = 5,
    n: int = 20,
    method: str = "dpp",
    temperature: float = 1.0,
    api_key: Optional[str] = None,
    max_tokens: int = 1024,
) -> List[str]:
    """Generate *k* diverse completions using the Anthropic API.

    Anthropic does not support the ``n`` parameter, so we make *n* individual
    requests and select the *k* most diverse.

    Parameters
    ----------
    prompt : str
        User prompt.
    model : str
        Anthropic model name.
    k, n : int
        Desired diverse outputs and candidate pool size.
    method : str
        Diversity selection method.
    temperature : float
        Sampling temperature (clamped to [0, 1]).
    api_key : str, optional
        Anthropic API key; falls back to ``ANTHROPIC_API_KEY`` env var.
    max_tokens : int
        Max tokens per response.

    Returns
    -------
    list of str
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError(
            "pip install anthropic  is required for anthropic_diverse_complete"
        )

    client_kwargs: Dict[str, Any] = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    client = anthropic.Anthropic(**client_kwargs)

    temperature = float(np.clip(temperature, 0.0, 1.0))

    candidates: List[str] = []
    for _ in range(n):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text
        candidates.append(text)

    return _select_diverse_subset(candidates, k, method=method)
