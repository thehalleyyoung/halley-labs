"""Structured diverse debate with optimal-transport diversity constraints.

Multi-round debate framework where agents are assigned diverse roles
(including devil's advocate), diversity is enforced between rounds,
consensus is detected via embedding convergence, and arguments are
organized into a directed graph.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from .embedding import TextEmbedder, embed_texts, project_to_sphere
from .diversity_metrics import cosine_diversity, sinkhorn_diversity_metric as sinkhorn_diversity
from .coverage import estimate_coverage, CoverageCertificate
from .transport import sinkhorn_candidate_scores


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class ArgumentType(Enum):
    CLAIM = auto()
    SUPPORT = auto()
    COUNTER = auto()
    REBUTTAL = auto()
    SYNTHESIS = auto()
    DEVILS_ADVOCATE = auto()


class DebateRole(Enum):
    PROPONENT = "proponent"
    OPPONENT = "opponent"
    DEVILS_ADVOCATE = "devils_advocate"
    SYNTHESIZER = "synthesizer"
    NEUTRAL = "neutral"


@dataclass
class Argument:
    """A single argument in the debate."""
    id: str
    text: str
    agent_id: int
    round_num: int
    arg_type: ArgumentType
    role: DebateRole
    embedding: Optional[np.ndarray] = None
    quality_score: float = 0.0
    novelty_score: float = 0.0
    parent_ids: List[str] = field(default_factory=list)


@dataclass
class ArgumentEdge:
    """Directed edge in the argument graph."""
    source_id: str
    target_id: str
    relation: str  # "supports", "counters", "rebuts", "synthesizes"
    weight: float = 1.0


@dataclass
class ArgumentGraph:
    """Directed graph of arguments and their relationships."""
    arguments: Dict[str, Argument]
    edges: List[ArgumentEdge]

    @property
    def n_nodes(self) -> int:
        return len(self.arguments)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def get_children(self, arg_id: str) -> List[str]:
        return [e.target_id for e in self.edges if e.source_id == arg_id]

    def get_parents(self, arg_id: str) -> List[str]:
        return [e.source_id for e in self.edges if e.target_id == arg_id]

    def roots(self) -> List[str]:
        """Arguments with no parents (initial claims)."""
        has_parent = {e.target_id for e in self.edges}
        return [aid for aid in self.arguments if aid not in has_parent]

    def leaves(self) -> List[str]:
        """Arguments with no children (final points)."""
        has_child = {e.source_id for e in self.edges}
        return [aid for aid in self.arguments if aid not in has_child]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arguments": {
                aid: {
                    "text": a.text[:200],
                    "agent": a.agent_id,
                    "round": a.round_num,
                    "type": a.arg_type.name,
                    "role": a.role.value,
                }
                for aid, a in self.arguments.items()
            },
            "edges": [
                {"from": e.source_id, "to": e.target_id, "relation": e.relation}
                for e in self.edges
            ],
        }


@dataclass
class ConsensusPoint:
    """A point of agreement detected in the debate."""
    text: str
    supporting_agents: List[int]
    confidence: float
    round_detected: int


@dataclass
class DebateResult:
    """Complete result from a diverse debate."""
    arguments: List[Argument]
    argument_graph: ArgumentGraph
    consensus_points: List[ConsensusPoint]
    diversity_per_round: List[float]
    overall_diversity: float
    coverage: Optional[CoverageCertificate]
    n_rounds: int
    n_agents: int
    converged: bool
    convergence_round: Optional[int]


# ---------------------------------------------------------------------------
# Agent simulator for debate
# ---------------------------------------------------------------------------

class DebateAgent:
    """A simulated debate agent with a role and personality.

    For real LLM usage, replace ``_generate_argument`` with an API call.
    """

    def __init__(
        self,
        agent_id: int,
        role: DebateRole,
        embed_dim: int = 64,
        seed: Optional[int] = None,
    ):
        self.agent_id = agent_id
        self.role = role
        self.embed_dim = embed_dim
        self.rng = np.random.RandomState(seed or agent_id)
        self._base_direction = self.rng.randn(embed_dim)
        self._base_direction /= np.linalg.norm(self._base_direction) + 1e-12

    def generate_argument(
        self,
        prompt: str,
        context: List[Argument],
        round_num: int,
        generate_fn: Optional[Callable] = None,
    ) -> Argument:
        """Generate an argument given the debate context.

        If ``generate_fn`` is provided, calls it to get text; otherwise
        produces a simulated argument.
        """
        if generate_fn is not None:
            context_str = "\n".join(
                f"[Agent {a.agent_id}, {a.role.value}]: {a.text[:100]}"
                for a in context[-6:]
            )
            full_prompt = (
                f"You are debating as a {self.role.value}.\n"
                f"Topic: {prompt}\n\n"
                f"Previous arguments:\n{context_str}\n\n"
                f"Provide your argument (round {round_num}):"
            )
            text = generate_fn(full_prompt)
        else:
            text = self._simulate_argument(prompt, context, round_num)

        # Determine argument type from role
        arg_type = {
            DebateRole.PROPONENT: ArgumentType.CLAIM,
            DebateRole.OPPONENT: ArgumentType.COUNTER,
            DebateRole.DEVILS_ADVOCATE: ArgumentType.DEVILS_ADVOCATE,
            DebateRole.SYNTHESIZER: ArgumentType.SYNTHESIS,
            DebateRole.NEUTRAL: ArgumentType.CLAIM,
        }.get(self.role, ArgumentType.CLAIM)

        if round_num > 0 and self.role in (DebateRole.PROPONENT, DebateRole.NEUTRAL):
            arg_type = ArgumentType.REBUTTAL

        # Compute parent references
        parent_ids = []
        if context:
            recent = context[-min(3, len(context)):]
            parent_ids = [a.id for a in recent if a.agent_id != self.agent_id]

        arg_id = f"arg_{self.agent_id}_{round_num}"
        return Argument(
            id=arg_id,
            text=text,
            agent_id=self.agent_id,
            round_num=round_num,
            arg_type=arg_type,
            role=self.role,
            parent_ids=parent_ids,
        )

    def _simulate_argument(
        self, prompt: str, context: List[Argument], round_num: int
    ) -> str:
        """Generate a simulated argument with role-appropriate content."""
        h = int(hashlib.md5(f"{prompt}:{self.agent_id}:{round_num}".encode()).hexdigest()[:8], 16)
        r = np.random.RandomState(h)

        role_prefixes = {
            DebateRole.PROPONENT: "I argue in favor: ",
            DebateRole.OPPONENT: "I argue against: ",
            DebateRole.DEVILS_ADVOCATE: "Playing devil's advocate: ",
            DebateRole.SYNTHESIZER: "Synthesizing the discussion: ",
            DebateRole.NEUTRAL: "From a neutral perspective: ",
        }
        prefix = role_prefixes.get(self.role, "")

        vocab = [
            "consider", "however", "furthermore", "evidence", "suggests",
            "perspective", "analysis", "impact", "contrary", "moreover",
            "alternative", "framework", "assumption", "challenge", "consensus",
            "empirical", "theoretical", "practical", "systemic", "nuanced",
        ]
        n_words = 15 + r.randint(20)
        words = [vocab[r.randint(len(vocab))] for _ in range(n_words)]
        return prefix + " ".join(words) + "."


# ---------------------------------------------------------------------------
# Consensus detection
# ---------------------------------------------------------------------------

def detect_consensus(
    arguments: List[Argument],
    embeddings: np.ndarray,
    threshold: float = 0.8,
) -> List[ConsensusPoint]:
    """Detect consensus points where multiple agents converge.

    Finds clusters of high-similarity arguments from different agents
    and reports them as consensus points.
    """
    if len(arguments) < 2:
        return []

    sim = embeddings @ embeddings.T
    consensus_points: List[ConsensusPoint] = []
    visited: Set[int] = set()

    for i in range(len(arguments)):
        if i in visited:
            continue
        cluster_indices = [i]
        cluster_agents = {arguments[i].agent_id}
        for j in range(i + 1, len(arguments)):
            if j in visited:
                continue
            if sim[i, j] > threshold and arguments[j].agent_id not in cluster_agents:
                cluster_indices.append(j)
                cluster_agents.add(arguments[j].agent_id)
                visited.add(j)
        visited.add(i)

        if len(cluster_agents) >= 2:
            # Multiple agents agree — this is a consensus point
            avg_sim = np.mean([sim[i, j] for j in cluster_indices if j != i])
            consensus_points.append(ConsensusPoint(
                text=arguments[i].text[:200],
                supporting_agents=sorted(cluster_agents),
                confidence=float(avg_sim),
                round_detected=max(arguments[idx].round_num for idx in cluster_indices),
            ))

    return consensus_points


# ---------------------------------------------------------------------------
# Diversity enforcement
# ---------------------------------------------------------------------------

def enforce_diversity_constraint(
    candidate_args: List[Argument],
    existing_args: List[Argument],
    embeddings_candidates: np.ndarray,
    embeddings_existing: np.ndarray,
    min_novelty: float = 0.3,
) -> List[int]:
    """Filter out arguments that are too similar to existing ones.

    Returns indices of candidate arguments that pass the novelty threshold.
    """
    if len(existing_args) == 0:
        return list(range(len(candidate_args)))

    sim = embeddings_candidates @ embeddings_existing.T
    max_sim = np.max(sim, axis=1)
    passing = [i for i in range(len(candidate_args)) if (1 - max_sim[i]) >= min_novelty]
    return passing if passing else [int(np.argmin(max_sim))]


# ---------------------------------------------------------------------------
# Main debate class
# ---------------------------------------------------------------------------

class DiverseDebate:
    """Orchestrate a structured, diversity-constrained debate.

    Assigns agents to diverse roles (proponent, opponent, devil's advocate,
    synthesizer), runs multiple rounds with diversity enforcement between
    rounds, detects consensus, and builds an argument graph.

    Example::

        debate = DiverseDebate(n_rounds=3, n_agents=4)
        result = debate.run("Should cities ban private cars?")
        print(result.argument_graph.to_dict())
        print(result.consensus_points)
    """

    def __init__(
        self,
        n_rounds: int = 3,
        n_agents: int = 4,
        embed_dim: int = 64,
        min_novelty: float = 0.2,
        consensus_threshold: float = 0.8,
        convergence_threshold: float = 0.05,
        generate_fn: Optional[Callable[[str], str]] = None,
    ):
        self.n_rounds = n_rounds
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        self.min_novelty = min_novelty
        self.consensus_threshold = consensus_threshold
        self.convergence_threshold = convergence_threshold
        self.generate_fn = generate_fn
        self._embedder = TextEmbedder(dim=embed_dim)

    def _assign_roles(self) -> List[DebateRole]:
        """Assign roles to maximize debate diversity."""
        if self.n_agents == 1:
            return [DebateRole.NEUTRAL]
        roles = [
            DebateRole.PROPONENT,
            DebateRole.OPPONENT,
            DebateRole.DEVILS_ADVOCATE,
            DebateRole.SYNTHESIZER,
        ]
        assigned = []
        for i in range(self.n_agents):
            assigned.append(roles[i % len(roles)])
        return assigned

    def run(
        self,
        topic: str,
        *,
        agent_generate_fns: Optional[List[Callable]] = None,
    ) -> DebateResult:
        """Run the diverse debate.

        Args:
            topic: The debate topic/question.
            agent_generate_fns: Optional per-agent generation functions.
                Each callable takes a prompt string and returns a response string.

        Returns:
            DebateResult with argument graph, consensus, and diversity metrics.
        """
        roles = self._assign_roles()
        agents = [
            DebateAgent(i, roles[i], embed_dim=self.embed_dim, seed=i * 17)
            for i in range(self.n_agents)
        ]

        all_arguments: List[Argument] = []
        all_embeddings: List[np.ndarray] = []
        diversity_per_round: List[float] = []
        converged = False
        convergence_round: Optional[int] = None

        for round_num in range(self.n_rounds):
            round_args: List[Argument] = []

            for i, agent in enumerate(agents):
                gen_fn = None
                if agent_generate_fns and i < len(agent_generate_fns):
                    gen_fn = agent_generate_fns[i]
                elif self.generate_fn:
                    gen_fn = self.generate_fn

                arg = agent.generate_argument(
                    topic, all_arguments, round_num, generate_fn=gen_fn,
                )
                # Embed
                arg.embedding = project_to_sphere(
                    self._embedder.embed(arg.text).reshape(1, -1)
                )[0]
                round_args.append(arg)

            # Embed all round arguments
            round_embeddings = np.array([a.embedding for a in round_args])

            # Diversity enforcement: filter near-duplicates
            if all_embeddings:
                existing_emb = np.array(all_embeddings)
                passing = enforce_diversity_constraint(
                    round_args, all_arguments,
                    round_embeddings, existing_emb,
                    min_novelty=self.min_novelty,
                )
                round_args = [round_args[i] for i in passing]
                round_embeddings = round_embeddings[passing]

            # Compute per-round diversity
            if len(round_embeddings) > 1:
                rd = float(cosine_diversity(round_embeddings))
            else:
                rd = 0.0
            diversity_per_round.append(rd)

            # Convergence check
            if len(diversity_per_round) >= 2:
                delta = abs(diversity_per_round[-1] - diversity_per_round[-2])
                if delta < self.convergence_threshold:
                    converged = True
                    convergence_round = round_num

            # Add to global state
            for arg in round_args:
                all_arguments.append(arg)
                all_embeddings.append(arg.embedding)

        # Build argument graph
        graph = self._build_argument_graph(all_arguments)

        # Detect consensus
        if all_embeddings:
            all_emb = np.array(all_embeddings)
            all_emb = project_to_sphere(all_emb)
            consensus = detect_consensus(
                all_arguments, all_emb, threshold=self.consensus_threshold,
            )
        else:
            all_emb = np.empty((0, self.embed_dim))
            consensus = []

        # Overall diversity
        if len(all_emb) > 1:
            overall_div = float(sinkhorn_diversity(all_emb, reg=0.1))
        else:
            overall_div = 0.0

        # Coverage
        cert = estimate_coverage(all_emb, epsilon=0.3) if len(all_emb) > 0 else None

        return DebateResult(
            arguments=all_arguments,
            argument_graph=graph,
            consensus_points=consensus,
            diversity_per_round=diversity_per_round,
            overall_diversity=overall_div,
            coverage=cert,
            n_rounds=self.n_rounds,
            n_agents=self.n_agents,
            converged=converged,
            convergence_round=convergence_round,
        )

    def _build_argument_graph(
        self, arguments: List[Argument]
    ) -> ArgumentGraph:
        """Build a directed argument graph from parent references."""
        arg_dict = {a.id: a for a in arguments}
        edges: List[ArgumentEdge] = []

        for arg in arguments:
            for pid in arg.parent_ids:
                if pid in arg_dict:
                    relation = self._infer_relation(arg, arg_dict[pid])
                    edges.append(ArgumentEdge(
                        source_id=pid,
                        target_id=arg.id,
                        relation=relation,
                    ))

        return ArgumentGraph(arguments=arg_dict, edges=edges)

    @staticmethod
    def _infer_relation(child: Argument, parent: Argument) -> str:
        """Infer the relation type between two arguments."""
        if child.arg_type == ArgumentType.COUNTER:
            return "counters"
        if child.arg_type == ArgumentType.REBUTTAL:
            return "rebuts"
        if child.arg_type == ArgumentType.SYNTHESIS:
            return "synthesizes"
        if child.arg_type == ArgumentType.SUPPORT:
            return "supports"
        if child.arg_type == ArgumentType.DEVILS_ADVOCATE:
            return "challenges"
        return "responds_to"
