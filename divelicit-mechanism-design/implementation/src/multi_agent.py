"""Multi-agent diversity elicitation.

Run the same prompt through multiple LLM providers with temperature diversity,
model diversity, and prompt diversity. Ensemble responses with deduplication
and track which agents produce the most unique insights.

Supports OpenAI, Anthropic, Google, and local (Ollama/vLLM) providers.
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .embedding import TextEmbedder, embed_texts, project_to_sphere
from .transport import sinkhorn_candidate_scores
from .diversity_metrics import cosine_diversity, sinkhorn_diversity_metric as sinkhorn_diversity
from .coverage import estimate_coverage, CoverageCertificate
from .kernels import AdaptiveRBFKernel


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""
    provider: str
    model: str
    temperatures: List[float] = field(default_factory=lambda: [0.7])
    system_prompt: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 1024
    n_samples_per_temp: int = 3

    @property
    def total_samples(self) -> int:
        return len(self.temperatures) * self.n_samples_per_temp


@dataclass
class AgentResponse:
    """A single response from an agent with metadata."""
    text: str
    provider: str
    model: str
    temperature: float
    prompt_variant: str
    embedding: Optional[np.ndarray] = None
    quality_score: float = 0.0
    latency_ms: float = 0.0
    is_duplicate: bool = False
    cluster_id: int = -1


@dataclass
class ProviderStats:
    """Statistics for a single provider's contributions."""
    provider: str
    model: str
    total_generated: int = 0
    unique_selected: int = 0
    duplicates_removed: int = 0
    avg_quality: float = 0.0
    avg_latency_ms: float = 0.0
    uniqueness_ratio: float = 0.0


@dataclass
class MultiAgentResult:
    """Result from multi-agent diverse elicitation."""
    selected_responses: List[AgentResponse]
    all_responses: List[AgentResponse]
    diversity_score: float
    coverage: Optional[CoverageCertificate]
    provider_stats: Dict[str, ProviderStats]
    prompt_variants_used: List[str]
    dedup_count: int
    total_generated: int


# ---------------------------------------------------------------------------
# Provider backends
# ---------------------------------------------------------------------------

class ProviderBackend(ABC):
    """Abstract backend for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, config: ProviderConfig,
                 temperature: float) -> Tuple[str, float]:
        """Generate a response. Returns (text, latency_ms)."""
        ...


class OpenAIBackend(ProviderBackend):
    """OpenAI API backend (GPT-4, GPT-4.1, etc.)."""

    def generate(self, prompt: str, config: ProviderConfig,
                 temperature: float) -> Tuple[str, float]:
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai")
        client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        messages = []
        if config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})
        messages.append({"role": "user", "content": prompt})
        t0 = time.time()
        resp = client.chat.completions.create(
            model=config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=config.max_tokens,
        )
        latency = (time.time() - t0) * 1000
        return resp.choices[0].message.content or "", latency


class AnthropicBackend(ProviderBackend):
    """Anthropic API backend (Claude)."""

    def generate(self, prompt: str, config: ProviderConfig,
                 temperature: float) -> Tuple[str, float]:
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        client = anthropic.Anthropic(api_key=config.api_key)
        t0 = time.time()
        resp = client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            system=config.system_prompt or "",
            temperature=temperature,
        )
        latency = (time.time() - t0) * 1000
        text = resp.content[0].text if resp.content else ""
        return text, latency


class GoogleBackend(ProviderBackend):
    """Google Generative AI backend (Gemini)."""

    def generate(self, prompt: str, config: ProviderConfig,
                 temperature: float) -> Tuple[str, float]:
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("pip install google-generativeai")
        if config.api_key:
            genai.configure(api_key=config.api_key)
        model = genai.GenerativeModel(config.model)
        t0 = time.time()
        resp = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=config.max_tokens,
            ),
        )
        latency = (time.time() - t0) * 1000
        return resp.text or "", latency


class LocalBackend(ProviderBackend):
    """Local model backend via OpenAI-compatible API (Ollama, vLLM, etc.)."""

    def generate(self, prompt: str, config: ProviderConfig,
                 temperature: float) -> Tuple[str, float]:
        try:
            import openai
        except ImportError:
            raise ImportError("pip install openai")
        base_url = config.base_url or "http://localhost:11434/v1"
        client = openai.OpenAI(
            api_key=config.api_key or "unused",
            base_url=base_url,
        )
        t0 = time.time()
        resp = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=config.max_tokens,
        )
        latency = (time.time() - t0) * 1000
        return resp.choices[0].message.content or "", latency


class SimulatedBackend(ProviderBackend):
    """Simulated backend for testing without API keys."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self._call_count = 0

    def generate(self, prompt: str, config: ProviderConfig,
                 temperature: float) -> Tuple[str, float]:
        self._call_count += 1
        h = hashlib.md5(
            f"{prompt}:{config.model}:{temperature}:{self._call_count}".encode()
        ).hexdigest()
        n_words = 20 + int(temperature * 40) + self.rng.randint(10)
        words = []
        r = np.random.RandomState(int(h[:8], 16))
        vocab = [
            "the", "a", "an", "this", "approach", "solution", "method",
            "system", "design", "framework", "model", "algorithm",
            "optimization", "transport", "diversity", "selection",
            "quality", "coverage", "efficient", "novel", "robust",
            "scalable", "distributed", "parallel", "adaptive",
            "iterative", "convergent", "bounded", "optimal",
            "stochastic", "deterministic", "hybrid", "multi-modal",
        ]
        for _ in range(n_words):
            words.append(vocab[r.randint(len(vocab))])
        text = " ".join(words).capitalize() + "."
        latency = 50 + self.rng.exponential(100)
        return text, latency


_BACKENDS: Dict[str, type] = {
    "openai": OpenAIBackend,
    "anthropic": AnthropicBackend,
    "google": GoogleBackend,
    "local": LocalBackend,
    "simulated": SimulatedBackend,
}


# ---------------------------------------------------------------------------
# Prompt diversity
# ---------------------------------------------------------------------------

def generate_prompt_variants(
    base_prompt: str,
    n_variants: int = 3,
) -> List[str]:
    """Generate prompt variants for diversity.

    Uses lightweight rephrasing strategies (no LLM call):
    - Original prompt
    - "Think step by step" variant
    - Devil's advocate variant
    - Brainstorm variant
    - Contrarian variant
    """
    variants = [base_prompt]
    templates = [
        "Think step by step about the following, then give your answer:\n{prompt}",
        "Play devil's advocate on this topic. Challenge conventional thinking:\n{prompt}",
        "Brainstorm as many different angles as possible:\n{prompt}",
        "What would a contrarian or unconventional thinker say about this?\n{prompt}",
        "Consider this from multiple disciplines (economics, psychology, engineering, philosophy):\n{prompt}",
        "What are the second-order effects and non-obvious consequences?\n{prompt}",
        "Provide a minority viewpoint that most people would overlook:\n{prompt}",
    ]
    for i in range(min(n_variants - 1, len(templates))):
        variants.append(templates[i].format(prompt=base_prompt))
    return variants[:n_variants]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_responses(
    responses: List[AgentResponse],
    similarity_threshold: float = 0.85,
    embed_dim: int = 64,
) -> Tuple[List[AgentResponse], int]:
    """Remove near-duplicate responses based on embedding similarity.

    Returns (deduplicated_list, n_removed).
    """
    if len(responses) <= 1:
        return responses, 0

    # Compute embeddings if not already present
    embedder = TextEmbedder(dim=embed_dim)
    texts = [r.text for r in responses]
    embeddings = embedder.embed_batch(texts)
    embeddings = project_to_sphere(embeddings)

    for i, resp in enumerate(responses):
        resp.embedding = embeddings[i]

    # Greedy deduplication: keep first, remove later duplicates
    kept: List[int] = [0]
    removed = 0
    for i in range(1, len(responses)):
        sim_to_kept = embeddings[i] @ embeddings[kept].T
        if np.max(sim_to_kept) < similarity_threshold:
            kept.append(i)
        else:
            responses[i].is_duplicate = True
            removed += 1

    deduplicated = [responses[i] for i in kept]
    return deduplicated, removed


# ---------------------------------------------------------------------------
# Multi-agent elicitor
# ---------------------------------------------------------------------------

class MultiAgentElicitor:
    """Orchestrate diverse elicitation across multiple LLM providers.

    Combines three diversity axes:
    1. **Model diversity** — different providers/models see the world differently.
    2. **Temperature diversity** — same model at different temperatures.
    3. **Prompt diversity** — rephrased prompts elicit different framings.

    Example::

        elicitor = MultiAgentElicitor()
        elicitor.add_provider("openai", model="gpt-4.1-nano",
                              temperatures=[0.3, 0.7, 1.0])
        elicitor.add_provider("anthropic", model="claude-sonnet-4-20250514")
        result = elicitor.elicit("Propose solutions to traffic", k=8)
    """

    def __init__(
        self,
        embed_dim: int = 64,
        dedup_threshold: float = 0.85,
        quality_weight: float = 0.3,
        sinkhorn_epsilon: float = 0.1,
        prompt_variants: int = 3,
    ):
        self.embed_dim = embed_dim
        self.dedup_threshold = dedup_threshold
        self.quality_weight = quality_weight
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.prompt_variants = prompt_variants
        self._providers: List[ProviderConfig] = []
        self._backends: Dict[str, ProviderBackend] = {}

    def add_provider(
        self,
        provider: str,
        model: str,
        temperatures: Optional[List[float]] = None,
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 1024,
        n_samples_per_temp: int = 3,
    ) -> None:
        """Register an LLM provider."""
        if temperatures is None:
            temperatures = [0.7]
        config = ProviderConfig(
            provider=provider,
            model=model,
            temperatures=temperatures,
            system_prompt=system_prompt,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            n_samples_per_temp=n_samples_per_temp,
        )
        self._providers.append(config)
        if provider not in self._backends:
            backend_cls = _BACKENDS.get(provider)
            if backend_cls is None:
                raise ValueError(
                    f"Unknown provider '{provider}'. "
                    f"Supported: {list(_BACKENDS.keys())}"
                )
            self._backends[provider] = backend_cls()

    def elicit(
        self,
        prompt: str,
        k: int,
        *,
        use_prompt_variants: bool = True,
    ) -> MultiAgentResult:
        """Run diverse elicitation across all registered providers.

        Args:
            prompt: Base prompt to send.
            k: Number of diverse responses to select.
            use_prompt_variants: Whether to use prompt rephrasing.

        Returns:
            MultiAgentResult with selected responses and analytics.
        """
        if not self._providers:
            raise ValueError("No providers registered. Call add_provider() first.")

        # Generate prompt variants
        if use_prompt_variants and self.prompt_variants > 1:
            variants = generate_prompt_variants(prompt, self.prompt_variants)
        else:
            variants = [prompt]

        # Collect responses from all providers
        all_responses: List[AgentResponse] = []
        for config in self._providers:
            backend = self._backends[config.provider]
            for temp in config.temperatures:
                for variant in variants:
                    for _ in range(config.n_samples_per_temp):
                        try:
                            text, latency = backend.generate(variant, config, temp)
                            if text.strip():
                                resp = AgentResponse(
                                    text=text,
                                    provider=config.provider,
                                    model=config.model,
                                    temperature=temp,
                                    prompt_variant=variant[:80],
                                    latency_ms=latency,
                                )
                                all_responses.append(resp)
                        except Exception:
                            continue

        total_generated = len(all_responses)
        if total_generated == 0:
            raise RuntimeError("All providers failed to generate responses")

        # Deduplicate
        deduped, n_removed = deduplicate_responses(
            all_responses,
            similarity_threshold=self.dedup_threshold,
            embed_dim=self.embed_dim,
        )

        # Assign quality scores (length + lexical diversity heuristic)
        for resp in deduped:
            words = resp.text.split()
            length_score = min(len(words) / 100.0, 1.0)
            unique_ratio = len(set(words)) / max(len(words), 1)
            resp.quality_score = 0.5 * length_score + 0.5 * unique_ratio

        # Embed all deduplicated responses
        embedder = TextEmbedder(dim=self.embed_dim)
        texts = [r.text for r in deduped]
        embeddings = project_to_sphere(embedder.embed_batch(texts))
        for i, resp in enumerate(deduped):
            resp.embedding = embeddings[i]

        # Select diverse subset via Sinkhorn
        k = min(k, len(deduped))
        selected_indices = self._sinkhorn_select(
            embeddings,
            np.array([r.quality_score for r in deduped]),
            k,
        )

        selected = [deduped[i] for i in selected_indices]
        sel_emb = embeddings[selected_indices]

        # Diversity & coverage
        div_score = float(sinkhorn_diversity(sel_emb, reg=self.sinkhorn_epsilon))
        cert = estimate_coverage(sel_emb, epsilon=0.3)

        # Provider stats
        stats = self._compute_provider_stats(all_responses, selected)

        return MultiAgentResult(
            selected_responses=selected,
            all_responses=all_responses,
            diversity_score=div_score,
            coverage=cert,
            provider_stats=stats,
            prompt_variants_used=variants,
            dedup_count=n_removed,
            total_generated=total_generated,
        )

    def _sinkhorn_select(
        self,
        embeddings: np.ndarray,
        quality_scores: np.ndarray,
        k: int,
    ) -> List[int]:
        """Greedy selection using Sinkhorn dual potentials."""
        n = embeddings.shape[0]
        selected: List[int] = []
        remaining = set(range(n))
        reference = embeddings

        for _ in range(k):
            if not remaining:
                break
            rem_list = sorted(remaining)
            if not selected:
                best = rem_list[int(np.argmax(quality_scores[rem_list]))]
            else:
                scores = sinkhorn_candidate_scores(
                    embeddings[rem_list],
                    embeddings[selected],
                    reference,
                    reg=self.sinkhorn_epsilon,
                )
                combined = (
                    (1 - self.quality_weight) * scores
                    + self.quality_weight * quality_scores[rem_list]
                )
                best = rem_list[int(np.argmax(combined))]
            selected.append(best)
            remaining.discard(best)
        return selected

    def _compute_provider_stats(
        self,
        all_responses: List[AgentResponse],
        selected: List[AgentResponse],
    ) -> Dict[str, ProviderStats]:
        """Compute per-provider contribution statistics."""
        stats: Dict[str, ProviderStats] = {}
        selected_set = {id(r) for r in selected}

        for resp in all_responses:
            key = f"{resp.provider}:{resp.model}"
            if key not in stats:
                stats[key] = ProviderStats(
                    provider=resp.provider, model=resp.model
                )
            s = stats[key]
            s.total_generated += 1
            s.avg_latency_ms += resp.latency_ms
            s.avg_quality += resp.quality_score
            if resp.is_duplicate:
                s.duplicates_removed += 1
            if id(resp) in selected_set:
                s.unique_selected += 1

        for s in stats.values():
            if s.total_generated > 0:
                s.avg_latency_ms /= s.total_generated
                s.avg_quality /= s.total_generated
                s.uniqueness_ratio = (
                    s.unique_selected / max(s.total_generated - s.duplicates_removed, 1)
                )
        return stats


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def quick_multi_agent(
    prompt: str,
    k: int = 5,
    providers: Optional[List[Dict[str, Any]]] = None,
) -> MultiAgentResult:
    """One-shot multi-agent elicitation with minimal configuration.

    Args:
        prompt: The prompt to elicit on.
        k: Number of diverse responses.
        providers: List of dicts with keys: provider, model, and optionally
            temperatures, api_key, etc. If None, uses simulated agents.

    Returns:
        MultiAgentResult.
    """
    elicitor = MultiAgentElicitor()
    if providers is None:
        elicitor.add_provider("simulated", model="sim-a", temperatures=[0.3, 0.7, 1.0])
        elicitor.add_provider("simulated", model="sim-b", temperatures=[0.5, 0.9])
    else:
        for p in providers:
            elicitor.add_provider(**p)
    return elicitor.elicit(prompt, k)


def compare_providers(
    prompt: str,
    providers: List[Dict[str, Any]],
    k: int = 5,
) -> Dict[str, ProviderStats]:
    """Compare which providers contribute most unique insights.

    Runs each provider separately, then jointly, and reports uniqueness.
    """
    # Run jointly
    joint_result = quick_multi_agent(prompt, k=k, providers=providers)

    # Also run each individually for comparison
    individual_stats: Dict[str, ProviderStats] = {}
    for p in providers:
        solo = MultiAgentElicitor()
        solo.add_provider(**p)
        try:
            result = solo.elicit(prompt, k=k)
            key = f"{p['provider']}:{p['model']}"
            if result.provider_stats:
                individual_stats[key] = list(result.provider_stats.values())[0]
        except Exception:
            continue

    # Merge joint stats with individual comparison
    merged = dict(joint_result.provider_stats)
    for key, indiv in individual_stats.items():
        if key in merged:
            merged[key].uniqueness_ratio = (
                merged[key].unique_selected / max(indiv.total_generated, 1)
            )
    return merged
