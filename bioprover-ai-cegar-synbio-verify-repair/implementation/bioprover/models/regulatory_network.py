"""Gene Regulatory Network module for BioProver.

Provides data structures and algorithms for representing, analyzing, and
decomposing gene regulatory networks (GRNs) used in synthetic biology
verification and counterexample-guided refinement.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class InteractionType(Enum):
    """Type of regulatory interaction between two genes."""
    ACTIVATION = auto()
    REPRESSION = auto()
    DUAL = auto()


class InteractionSign(Enum):
    """Sign (monotonicity direction) of an interaction."""
    POSITIVE = auto()
    NEGATIVE = auto()
    UNKNOWN = auto()


class LoopType(Enum):
    """Classification of a feedback loop."""
    POSITIVE_FEEDBACK = auto()
    NEGATIVE_FEEDBACK = auto()
    UNKNOWN_FEEDBACK = auto()


class MotifType(Enum):
    """Common network motif archetypes."""
    COHERENT_FFL = auto()
    INCOHERENT_FFL = auto()
    TOGGLE_SWITCH = auto()
    REPRESSILATOR = auto()
    AUTOREGULATION_POSITIVE = auto()
    AUTOREGULATION_NEGATIVE = auto()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RegulatoryInteraction:
    """A single directed regulatory interaction between two genes.

    Attributes:
        source: Name of the regulator gene.
        target: Name of the regulated gene.
        interaction_type: Whether the interaction activates, represses, or
            has dual (context-dependent) effect on the target.
        strength: Optional interaction strength / fold-change.
        hill_coefficient: Optional Hill coefficient for the dose–response.
        threshold: Optional activation / repression threshold concentration.
    """

    source: str
    target: str
    interaction_type: InteractionType
    strength: Optional[float] = None
    hill_coefficient: Optional[float] = None
    threshold: Optional[float] = None

    @property
    def sign(self) -> InteractionSign:
        """Return the sign of this interaction."""
        if self.interaction_type == InteractionType.ACTIVATION:
            return InteractionSign.POSITIVE
        if self.interaction_type == InteractionType.REPRESSION:
            return InteractionSign.NEGATIVE
        return InteractionSign.UNKNOWN

    @property
    def is_monotone(self) -> bool:
        """Return ``True`` if the interaction is monotone (not DUAL)."""
        return self.interaction_type != InteractionType.DUAL


@dataclass
class FeedbackLoop:
    """An ordered feedback loop in a gene regulatory network.

    Attributes:
        genes: Ordered list of gene names forming the loop.
        interactions: Ordered list of interactions along the loop.
    """

    genes: List[str]
    interactions: List[RegulatoryInteraction]

    @property
    def loop_type(self) -> LoopType:
        """Classify the loop based on interaction signs.

        A loop with an even number of repressions is positive feedback;
        an odd number yields negative feedback.  If any interaction is
        DUAL the loop type is unknown.
        """
        if any(i.interaction_type == InteractionType.DUAL for i in self.interactions):
            return LoopType.UNKNOWN_FEEDBACK
        num_repressions = sum(
            1 for i in self.interactions
            if i.interaction_type == InteractionType.REPRESSION
        )
        if num_repressions % 2 == 0:
            return LoopType.POSITIVE_FEEDBACK
        return LoopType.NEGATIVE_FEEDBACK

    @property
    def length(self) -> int:
        """Number of genes (equivalently, edges) in the loop."""
        return len(self.genes)


@dataclass
class NetworkMotif:
    """A recognised small network motif.

    Attributes:
        motif_type: The archetype this motif matches.
        genes: Genes participating in the motif.
        interactions: Interactions forming the motif.
    """

    motif_type: MotifType
    genes: List[str]
    interactions: List[RegulatoryInteraction]


# ---------------------------------------------------------------------------
# GeneRegulatoryNetwork
# ---------------------------------------------------------------------------

class GeneRegulatoryNetwork:
    """Directed graph representation of a gene regulatory network.

    Internally stores genes as nodes and :class:`RegulatoryInteraction`
    objects as edge data in a :class:`networkx.DiGraph`.
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()

    # -- Mutators -----------------------------------------------------------

    def add_gene(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a gene node to the network.

        Args:
            name: Unique gene identifier.
            metadata: Optional dictionary of gene-level annotations.
        """
        self._graph.add_node(name, **(metadata or {}))

    def add_interaction(self, interaction: RegulatoryInteraction) -> None:
        """Add a regulatory interaction (edge) to the network.

        Both source and target genes are added automatically if they are
        not already present.

        Args:
            interaction: The interaction to add.
        """
        if interaction.source not in self._graph:
            self._graph.add_node(interaction.source)
        if interaction.target not in self._graph:
            self._graph.add_node(interaction.target)
        self._graph.add_edge(
            interaction.source,
            interaction.target,
            interaction=interaction,
        )

    def remove_interaction(self, source: str, target: str) -> None:
        """Remove the interaction edge between *source* and *target*.

        Args:
            source: Regulator gene name.
            target: Regulated gene name.

        Raises:
            nx.NetworkXError: If the edge does not exist.
        """
        self._graph.remove_edge(source, target)

    # -- Properties ---------------------------------------------------------

    @property
    def genes(self) -> List[str]:
        """Return the list of gene names in the network."""
        return list(self._graph.nodes)

    @property
    def interactions(self) -> List[RegulatoryInteraction]:
        """Return all interactions stored in the network."""
        return [
            data["interaction"]
            for _, _, data in self._graph.edges(data=True)
            if "interaction" in data
        ]

    @property
    def num_genes(self) -> int:
        """Number of genes in the network."""
        return self._graph.number_of_nodes()

    @property
    def num_interactions(self) -> int:
        """Number of regulatory interactions in the network."""
        return self._graph.number_of_edges()

    # -- Neighbourhood queries ----------------------------------------------

    def get_regulators(self, gene: str) -> List[RegulatoryInteraction]:
        """Return interactions where *gene* is the target (incoming edges).

        Args:
            gene: Gene name to query.

        Returns:
            List of :class:`RegulatoryInteraction` objects targeting *gene*.
        """
        result: List[RegulatoryInteraction] = []
        for pred in self._graph.predecessors(gene):
            data = self._graph.edges[pred, gene]
            if "interaction" in data:
                result.append(data["interaction"])
        return result

    def get_targets(self, gene: str) -> List[RegulatoryInteraction]:
        """Return interactions where *gene* is the source (outgoing edges).

        Args:
            gene: Gene name to query.

        Returns:
            List of :class:`RegulatoryInteraction` objects originating from *gene*.
        """
        result: List[RegulatoryInteraction] = []
        for succ in self._graph.successors(gene):
            data = self._graph.edges[gene, succ]
            if "interaction" in data:
                result.append(data["interaction"])
        return result

    # -- Kinetics inference -------------------------------------------------

    @staticmethod
    def interaction_type_from_kinetics(kinetic_law: Any) -> InteractionType:
        """Infer the interaction type from a kinetic law object.

        Recognises common naming conventions used in SBML / BioProver
        kinetic-law classes (e.g. ``HillActivation``, ``HillRepression``).

        Args:
            kinetic_law: An object whose class name encodes the kinetics type.

        Returns:
            The inferred :class:`InteractionType`.
        """
        cls_name = type(kinetic_law).__name__.lower()
        if "activation" in cls_name or "hillactivation" in cls_name:
            return InteractionType.ACTIVATION
        if "repression" in cls_name or "hillrepression" in cls_name:
            return InteractionType.REPRESSION
        return InteractionType.DUAL

    # -- Monotonicity -------------------------------------------------------

    def is_monotone(self) -> bool:
        """Return ``True`` if every interaction in the network is monotone.

        A network is monotone when it contains no DUAL interactions, which
        simplifies formal verification.
        """
        return all(i.is_monotone for i in self.interactions)

    def get_non_monotone_interactions(self) -> List[RegulatoryInteraction]:
        """Return interactions that are not monotone (i.e. DUAL)."""
        return [i for i in self.interactions if not i.is_monotone]

    # -- Feedback loops -----------------------------------------------------

    def find_feedback_loops(self, max_length: int = 10) -> List[FeedbackLoop]:
        """Find all simple feedback loops up to *max_length* genes.

        Uses :func:`networkx.simple_cycles` and filters by length.

        Args:
            max_length: Maximum number of genes in a loop to consider.

        Returns:
            List of :class:`FeedbackLoop` instances.
        """
        loops: List[FeedbackLoop] = []
        for cycle in nx.simple_cycles(self._graph, length_bound=max_length):
            if len(cycle) > max_length:
                continue
            interactions: List[RegulatoryInteraction] = []
            valid = True
            for i in range(len(cycle)):
                src = cycle[i]
                tgt = cycle[(i + 1) % len(cycle)]
                edge_data = self._graph.get_edge_data(src, tgt)
                if edge_data is None or "interaction" not in edge_data:
                    valid = False
                    break
                interactions.append(edge_data["interaction"])
            if valid:
                loops.append(FeedbackLoop(genes=list(cycle), interactions=interactions))
        return loops

    def find_positive_feedback_loops(self) -> List[FeedbackLoop]:
        """Return only positive-feedback loops."""
        return [
            loop for loop in self.find_feedback_loops()
            if loop.loop_type == LoopType.POSITIVE_FEEDBACK
        ]

    def find_negative_feedback_loops(self) -> List[FeedbackLoop]:
        """Return only negative-feedback loops."""
        return [
            loop for loop in self.find_feedback_loops()
            if loop.loop_type == LoopType.NEGATIVE_FEEDBACK
        ]

    # -- Motif detection ----------------------------------------------------

    def detect_motifs(self) -> List[NetworkMotif]:
        """Detect common regulatory motifs in the network.

        Detected motif types:
        * **Feed-forward loop (FFL):** Three genes *A → B → C* with a
          shortcut *A → C*.  Classified as *coherent* when the direct and
          indirect paths have the same overall sign, otherwise *incoherent*.
        * **Toggle switch:** Two genes that mutually repress each other.
        * **Repressilator:** A three-gene cycle where every interaction is
          repression.
        * **Autoregulation:** A gene that regulates itself (positive or
          negative).

        Returns:
            List of :class:`NetworkMotif` instances found.
        """
        motifs: List[NetworkMotif] = []

        # --- Autoregulation (self-loops) ---
        for gene in self._graph.nodes:
            if self._graph.has_edge(gene, gene):
                edge_data = self._graph.edges[gene, gene]
                if "interaction" not in edge_data:
                    continue
                inter = edge_data["interaction"]
                if inter.interaction_type == InteractionType.ACTIVATION:
                    mtype = MotifType.AUTOREGULATION_POSITIVE
                elif inter.interaction_type == InteractionType.REPRESSION:
                    mtype = MotifType.AUTOREGULATION_NEGATIVE
                else:
                    # DUAL autoregulation – classify as positive by convention
                    mtype = MotifType.AUTOREGULATION_POSITIVE
                motifs.append(NetworkMotif(
                    motif_type=mtype,
                    genes=[gene],
                    interactions=[inter],
                ))

        # --- Toggle switch (mutual repression) ---
        seen_pairs: Set[Tuple[str, str]] = set()
        for u, v, data in self._graph.edges(data=True):
            if u == v:
                continue
            if "interaction" not in data:
                continue
            pair = (min(u, v), max(u, v))
            if pair in seen_pairs:
                continue
            if not self._graph.has_edge(v, u):
                continue
            rev_data = self._graph.edges[v, u]
            if "interaction" not in rev_data:
                continue
            i_uv = data["interaction"]
            i_vu = rev_data["interaction"]
            if (i_uv.interaction_type == InteractionType.REPRESSION
                    and i_vu.interaction_type == InteractionType.REPRESSION):
                seen_pairs.add(pair)
                motifs.append(NetworkMotif(
                    motif_type=MotifType.TOGGLE_SWITCH,
                    genes=[u, v],
                    interactions=[i_uv, i_vu],
                ))

        # --- Feed-forward loops & Repressilator ---
        gene_list = list(self._graph.nodes)
        for a, b, c in itertools.permutations(gene_list, 3):
            # FFL: A -> B, B -> C, A -> C
            if (self._graph.has_edge(a, b)
                    and self._graph.has_edge(b, c)
                    and self._graph.has_edge(a, c)):
                e_ab = self._graph.edges[a, b]
                e_bc = self._graph.edges[b, c]
                e_ac = self._graph.edges[a, c]
                if all("interaction" in e for e in (e_ab, e_bc, e_ac)):
                    i_ab = e_ab["interaction"]
                    i_bc = e_bc["interaction"]
                    i_ac = e_ac["interaction"]
                    indirect_sign = self._compose_signs(i_ab.sign, i_bc.sign)
                    direct_sign = i_ac.sign
                    if (indirect_sign == InteractionSign.UNKNOWN
                            or direct_sign == InteractionSign.UNKNOWN):
                        ffl_type = MotifType.INCOHERENT_FFL
                    elif indirect_sign == direct_sign:
                        ffl_type = MotifType.COHERENT_FFL
                    else:
                        ffl_type = MotifType.INCOHERENT_FFL
                    motifs.append(NetworkMotif(
                        motif_type=ffl_type,
                        genes=[a, b, c],
                        interactions=[i_ab, i_bc, i_ac],
                    ))

        # --- Repressilator: 3-gene cycle of all repressions ---
        for cycle in nx.simple_cycles(self._graph, length_bound=3):
            if len(cycle) != 3:
                continue
            edges = []
            all_repression = True
            for i in range(3):
                src = cycle[i]
                tgt = cycle[(i + 1) % 3]
                edge_data = self._graph.get_edge_data(src, tgt)
                if edge_data is None or "interaction" not in edge_data:
                    all_repression = False
                    break
                inter = edge_data["interaction"]
                if inter.interaction_type != InteractionType.REPRESSION:
                    all_repression = False
                    break
                edges.append(inter)
            if all_repression and len(edges) == 3:
                motifs.append(NetworkMotif(
                    motif_type=MotifType.REPRESSILATOR,
                    genes=list(cycle),
                    interactions=edges,
                ))

        return motifs

    @staticmethod
    def _compose_signs(s1: InteractionSign, s2: InteractionSign) -> InteractionSign:
        """Compose two interaction signs along a path.

        POSITIVE * POSITIVE = POSITIVE, NEGATIVE * NEGATIVE = POSITIVE,
        POSITIVE * NEGATIVE = NEGATIVE, anything with UNKNOWN = UNKNOWN.
        """
        if s1 == InteractionSign.UNKNOWN or s2 == InteractionSign.UNKNOWN:
            return InteractionSign.UNKNOWN
        if s1 == s2:
            return InteractionSign.POSITIVE
        return InteractionSign.NEGATIVE

    # -- Structural analysis ------------------------------------------------

    def strongly_connected_components(self) -> List[Set[str]]:
        """Return the strongly connected components of the network.

        Returns:
            List of sets, each containing gene names belonging to one SCC.
        """
        return [set(c) for c in nx.strongly_connected_components(self._graph)]

    def decompose_for_verification(self) -> List[GeneRegulatoryNetwork]:
        """Decompose the network into sub-networks per SCC.

        Each returned :class:`GeneRegulatoryNetwork` contains the genes
        and interactions of one strongly connected component, enabling
        independent verification of each component.

        Returns:
            List of sub-networks, one per SCC.
        """
        sub_networks: List[GeneRegulatoryNetwork] = []
        for component in self.strongly_connected_components():
            sub = GeneRegulatoryNetwork()
            for gene in component:
                meta = dict(self._graph.nodes[gene])
                sub.add_gene(gene, metadata=meta if meta else None)
            for u, v, data in self._graph.edges(data=True):
                if u in component and v in component and "interaction" in data:
                    sub.add_interaction(data["interaction"])
            sub_networks.append(sub)
        return sub_networks

    def topological_order(self) -> Optional[List[str]]:
        """Return a topological ordering of the genes, if one exists.

        If the network contains cycles a topological sort is not possible
        on the full graph.  In that case the method falls back to the
        condensation DAG (each SCC collapsed to one node) and returns an
        ordering of *SCCs* flattened into a gene list.  Returns ``None``
        only when the condensation itself cannot be ordered (should not
        happen for a DAG).

        Returns:
            Ordered list of gene names, or ``None`` if ordering fails.
        """
        if nx.is_directed_acyclic_graph(self._graph):
            return list(nx.topological_sort(self._graph))

        # Fall back to condensation DAG
        condensation = nx.condensation(self._graph)
        mapping = condensation.graph.get("mapping", {})
        # mapping: original node -> SCC index
        scc_members: Dict[int, List[str]] = {}
        for node, scc_id in mapping.items():
            scc_members.setdefault(scc_id, []).append(node)

        try:
            topo = list(nx.topological_sort(condensation))
        except nx.NetworkXUnfeasible:
            return None

        result: List[str] = []
        for scc_id in topo:
            result.extend(sorted(scc_members.get(scc_id, [])))
        return result

    def signal_flow_paths(self, source: str, target: str) -> List[List[str]]:
        """Find all simple paths from *source* to *target*.

        Args:
            source: Start gene.
            target: End gene.

        Returns:
            List of paths, where each path is a list of gene names.
        """
        return list(nx.all_simple_paths(self._graph, source, target))

    def adjacency_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Build a signed adjacency matrix for the network.

        Entries are ``+1`` for activation, ``-1`` for repression, and
        ``0`` for dual / absent edges.

        Returns:
            Tuple of (matrix, gene_order) where *gene_order* maps column/row
            indices to gene names.
        """
        gene_order = sorted(self._graph.nodes)
        idx = {g: i for i, g in enumerate(gene_order)}
        n = len(gene_order)
        matrix = np.zeros((n, n), dtype=float)
        for inter in self.interactions:
            r, c = idx[inter.source], idx[inter.target]
            if inter.interaction_type == InteractionType.ACTIVATION:
                matrix[r, c] = 1.0
            elif inter.interaction_type == InteractionType.REPRESSION:
                matrix[r, c] = -1.0
            # DUAL -> 0.0 (already default)
        return matrix, gene_order

    # -- Construction helpers -----------------------------------------------

    @classmethod
    def from_reactions(
        cls,
        reactions: List[Any],
        species_names: List[str],
    ) -> GeneRegulatoryNetwork:
        """Build a network by inferring interactions from reaction kinetics.

        Each reaction object is expected to expose ``reactants``,
        ``products``, and ``kinetic_law`` attributes.  Species names are
        used to create the initial gene set.

        Args:
            reactions: Iterable of reaction-like objects.
            species_names: Names of species (genes) in the model.

        Returns:
            A new :class:`GeneRegulatoryNetwork`.
        """
        network = cls()
        for name in species_names:
            network.add_gene(name)

        for reaction in reactions:
            kinetic_law = getattr(reaction, "kinetic_law", None)
            if kinetic_law is None:
                continue

            itype = cls.interaction_type_from_kinetics(kinetic_law)

            reactant_names: List[str] = []
            product_names: List[str] = []

            # Support both list-of-str and list-of-objects with .name
            for r in getattr(reaction, "reactants", []):
                rname = r if isinstance(r, str) else getattr(r, "name", str(r))
                if rname in species_names:
                    reactant_names.append(rname)

            for p in getattr(reaction, "products", []):
                pname = p if isinstance(p, str) else getattr(p, "name", str(p))
                if pname in species_names:
                    product_names.append(pname)

            # Create interactions: each reactant regulates each product
            for src in reactant_names:
                for tgt in product_names:
                    if src == tgt:
                        continue
                    interaction = RegulatoryInteraction(
                        source=src,
                        target=tgt,
                        interaction_type=itype,
                    )
                    network.add_interaction(interaction)

        return network

    # -- Dunder -------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"GeneRegulatoryNetwork(genes={self.num_genes}, "
            f"interactions={self.num_interactions})"
        )
