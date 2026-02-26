"""
Message-passing algorithms for junction-tree inference.

Implements both Hugin-style (collect/distribute with in-place updates)
and Shafer-Shenoy (explicit message passing) algorithms.  Includes
log-space computation for numerical stability, parallel scheduling,
and convergence checking for an optional loopy-BP fallback.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

from .potential_table import PotentialTable, multiply_potentials, marginalize_to
from .clique_tree import CliqueTree, CliqueNode, Separator
from .cache import InferenceCache, CacheKey


# ------------------------------------------------------------------ #
#  Algorithm variant
# ------------------------------------------------------------------ #

class MessagePassingVariant(Enum):
    HUGIN = "hugin"
    SHAFER_SHENOY = "shafer_shenoy"


# ------------------------------------------------------------------ #
#  Message record
# ------------------------------------------------------------------ #

@dataclass
class Message:
    """An immutable record of one message sent between two cliques."""

    sender_idx: int
    receiver_idx: int
    separator_vars: FrozenSet[str]
    potential: PotentialTable
    timestamp: float = field(default_factory=time.time)
    from_cache: bool = False


# ------------------------------------------------------------------ #
#  Statistics
# ------------------------------------------------------------------ #

@dataclass
class PassingStats:
    """Accumulated statistics for a message-passing run."""

    messages_sent: int = 0
    cache_hits: int = 0
    total_time_s: float = 0.0
    max_message_size: int = 0
    total_message_entries: int = 0
    collect_time_s: float = 0.0
    distribute_time_s: float = 0.0
    convergence_iterations: int = 0

    def summary(self) -> Dict[str, Any]:
        return {
            "messages_sent": self.messages_sent,
            "cache_hits": self.cache_hits,
            "total_time_s": round(self.total_time_s, 6),
            "max_message_size": self.max_message_size,
            "collect_time_s": round(self.collect_time_s, 6),
            "distribute_time_s": round(self.distribute_time_s, 6),
        }


# ------------------------------------------------------------------ #
#  Message passer
# ------------------------------------------------------------------ #

class MessagePasser:
    """Message-passing engine for calibrating a junction tree.

    Parameters
    ----------
    tree : CliqueTree
        The junction tree to calibrate.
    variant : MessagePassingVariant
        Which algorithm variant to use.
    use_log_space : bool
        If *True*, all computations are performed in log-space.
    cache : InferenceCache or None
        Optional memoization cache for messages.
    convergence_tol : float
        Tolerance for convergence checking (loopy BP fallback).
    max_iterations : int
        Maximum iterations for loopy BP.
    """

    def __init__(
        self,
        tree: CliqueTree,
        variant: MessagePassingVariant = MessagePassingVariant.HUGIN,
        use_log_space: bool = False,
        cache: Optional[InferenceCache] = None,
        convergence_tol: float = 1e-8,
        max_iterations: int = 100,
    ) -> None:
        self.tree = tree
        self.variant = variant
        self.use_log_space = use_log_space
        self.cache = cache
        self.convergence_tol = convergence_tol
        self.max_iterations = max_iterations

        self._messages: Dict[Tuple[int, int], Message] = {}
        self._stats = PassingStats()

        # Evidence signature for cache keying
        self._evidence_sig: str = ""
        self._intervention_sig: str = ""

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def calibrate(
        self,
        root: Optional[int] = None,
        evidence: Optional[Dict[str, int]] = None,
        intervention: Optional[Dict[str, float]] = None,
    ) -> PassingStats:
        """Fully calibrate the junction tree.

        1. Enter evidence into clique potentials.
        2. Collect evidence (leaves → root).
        3. Distribute evidence (root → leaves).
        """
        t0 = time.time()
        self._stats = PassingStats()

        if root is None:
            root = self.tree.find_root()

        # Convert potentials to log-space if requested
        if self.use_log_space:
            self._convert_to_log_space()

        # Enter evidence
        if evidence:
            self._enter_evidence(evidence)

        # Run message passing
        if self.variant == MessagePassingVariant.HUGIN:
            self._hugin_calibrate(root)
        else:
            self._shafer_shenoy_calibrate(root)

        self._stats.total_time_s = time.time() - t0
        return self._stats

    def collect_evidence(
        self, root: int, evidence: Optional[Dict[str, int]] = None
    ) -> None:
        """Upward pass only (leaves → root)."""
        if evidence:
            self._enter_evidence(evidence)

        collect_order, _ = self.tree.get_message_schedule(root)
        for sender, receiver in collect_order:
            self._send_message(sender, receiver)

    def distribute_evidence(self, root: int) -> None:
        """Downward pass only (root → leaves)."""
        _, distribute_order = self.tree.get_message_schedule(root)
        for sender, receiver in distribute_order:
            self._send_message(sender, receiver)

    def get_messages(self) -> Dict[Tuple[int, int], Message]:
        """Return all computed messages."""
        return dict(self._messages)

    @property
    def stats(self) -> PassingStats:
        return self._stats

    def reset(self) -> None:
        """Clear all messages (potentials are untouched)."""
        self._messages.clear()
        self._stats = PassingStats()

    # ------------------------------------------------------------------ #
    #  Evidence entry
    # ------------------------------------------------------------------ #

    def _enter_evidence(self, evidence: Dict[str, int]) -> None:
        """Enter hard evidence by reducing clique potentials.

        For each observed variable, exactly one clique is reduced.
        """
        assigned: Set[str] = set()
        for clique in self.tree.cliques:
            if clique.potential is None:
                continue
            for var, val in evidence.items():
                if var in clique.variables and var not in assigned:
                    clique.potential = clique.potential.reduce({var: val})
                    assigned.add(var)

    # ------------------------------------------------------------------ #
    #  Hugin-style calibration
    # ------------------------------------------------------------------ #

    def _hugin_calibrate(self, root: int) -> None:
        """Hugin algorithm: modify clique/separator potentials in-place."""
        collect_order, distribute_order = self.tree.get_message_schedule(root)

        t1 = time.time()
        for sender, receiver in collect_order:
            self._hugin_pass(sender, receiver)
        self._stats.collect_time_s = time.time() - t1

        t2 = time.time()
        for sender, receiver in distribute_order:
            self._hugin_pass(sender, receiver)
        self._stats.distribute_time_s = time.time() - t2

    def _hugin_pass(self, sender_idx: int, receiver_idx: int) -> None:
        """Single Hugin message: project sender onto separator, update
        receiver by multiplying the ratio new_sep / old_sep."""
        sender = self.tree.cliques[sender_idx]
        receiver = self.tree.cliques[receiver_idx]
        sep = self.tree.get_separator(sender_idx, receiver_idx)

        if sender.potential is None or receiver.potential is None or sep is None:
            return

        sep_vars = list(sep.variables & set(sender.potential.variables))
        if not sep_vars:
            self._stats.messages_sent += 1
            return

        # Project sender onto separator
        margin_vars = [
            v for v in sender.potential.variables if v not in sep.variables
        ]
        new_sep_potential = sender.potential.marginalize(margin_vars)

        # Cache check
        cache_key = self._make_cache_key(sender_idx, receiver_idx)
        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                new_sep_potential = PotentialTable(
                    cached.variables,
                    {v: self.tree.cardinalities[v] for v in cached.variables},
                    cached.value,
                    log_space=self.use_log_space,
                )
                self._stats.cache_hits += 1

        # Compute ratio and update receiver
        old_sep = sep.potential
        if old_sep is not None and old_sep.size > 0:
            ratio = new_sep_potential.divide(old_sep)
            receiver.potential = receiver.potential.multiply(ratio)
        else:
            receiver.potential = receiver.potential.multiply(new_sep_potential)

        # Update separator potential
        sep.potential = new_sep_potential

        # Record message
        msg = Message(
            sender_idx=sender_idx,
            receiver_idx=receiver_idx,
            separator_vars=frozenset(sep_vars),
            potential=new_sep_potential,
        )
        self._messages[(sender_idx, receiver_idx)] = msg
        self._stats.messages_sent += 1
        self._stats.max_message_size = max(
            self._stats.max_message_size, new_sep_potential.size
        )
        self._stats.total_message_entries += new_sep_potential.size

        # Store in cache
        if self.cache is not None:
            self.cache.put(
                cache_key,
                new_sep_potential.values,
                new_sep_potential.variables,
                boundary_vars=set(sep_vars),
            )

    # ------------------------------------------------------------------ #
    #  Shafer-Shenoy calibration
    # ------------------------------------------------------------------ #

    def _shafer_shenoy_calibrate(self, root: int) -> None:
        """Shafer-Shenoy: compute explicit messages without modifying
        clique potentials until both passes are complete."""
        collect_order, distribute_order = self.tree.get_message_schedule(root)

        t1 = time.time()
        for sender, receiver in collect_order:
            self._shafer_shenoy_message(sender, receiver)
        self._stats.collect_time_s = time.time() - t1

        t2 = time.time()
        for sender, receiver in distribute_order:
            self._shafer_shenoy_message(sender, receiver)
        self._stats.distribute_time_s = time.time() - t2

        # Update clique beliefs
        self._update_beliefs_from_messages()

    def _shafer_shenoy_message(
        self, sender_idx: int, receiver_idx: int
    ) -> None:
        """Compute a single Shafer-Shenoy message:

        msg(S→R) = Σ_{S\\sep} [ ψ_S · ∏_{N≠R} msg(N→S) ]
        """
        sender = self.tree.cliques[sender_idx]
        sep = self.tree.get_separator(sender_idx, receiver_idx)
        if sender.potential is None or sep is None:
            return

        # Cache check
        cache_key = self._make_cache_key(sender_idx, receiver_idx)
        if self.cache is not None:
            cached = self.cache.get(cache_key)
            if cached is not None:
                msg = Message(
                    sender_idx=sender_idx,
                    receiver_idx=receiver_idx,
                    separator_vars=frozenset(sep.variables),
                    potential=PotentialTable(
                        cached.variables,
                        {v: self.tree.cardinalities[v] for v in cached.variables},
                        cached.value,
                        log_space=self.use_log_space,
                    ),
                    from_cache=True,
                )
                self._messages[(sender_idx, receiver_idx)] = msg
                self._stats.cache_hits += 1
                self._stats.messages_sent += 1
                return

        # Start with sender's original potential
        combined = sender.potential.copy()

        # Multiply in all incoming messages except from receiver
        for nb_idx in self.tree.neighbors(sender_idx):
            if nb_idx == receiver_idx:
                continue
            incoming = self._messages.get((nb_idx, sender_idx))
            if incoming is not None:
                combined = combined.multiply(incoming.potential)

        # Marginalize to separator
        margin_out = [
            v for v in combined.variables if v not in sep.variables
        ]
        message_potential = combined.marginalize(margin_out)

        msg = Message(
            sender_idx=sender_idx,
            receiver_idx=receiver_idx,
            separator_vars=frozenset(sep.variables),
            potential=message_potential,
        )
        self._messages[(sender_idx, receiver_idx)] = msg
        self._stats.messages_sent += 1
        self._stats.max_message_size = max(
            self._stats.max_message_size, message_potential.size
        )
        self._stats.total_message_entries += message_potential.size

        if self.cache is not None:
            self.cache.put(
                cache_key,
                message_potential.values,
                message_potential.variables,
                boundary_vars=set(sep.variables),
            )

    def _update_beliefs_from_messages(self) -> None:
        """After Shafer-Shenoy passes, update each clique's belief by
        multiplying its original potential with all incoming messages."""
        for clique in self.tree.cliques:
            belief = clique.potential
            if belief is None:
                continue
            for nb_idx in self.tree.neighbors(clique.index):
                incoming = self._messages.get((nb_idx, clique.index))
                if incoming is not None:
                    belief = belief.multiply(incoming.potential)
            clique.potential = belief

    # ------------------------------------------------------------------ #
    #  Loopy BP fallback
    # ------------------------------------------------------------------ #

    def loopy_bp(
        self,
        evidence: Optional[Dict[str, int]] = None,
        damping: float = 0.5,
    ) -> PassingStats:
        """Loopy belief propagation for non-tree-structured graphs.

        Iterates message passing until convergence or ``max_iterations``.
        Uses damping to improve convergence:
            new_msg = damping * old_msg + (1-damping) * computed_msg
        """
        t0 = time.time()
        self._stats = PassingStats()

        if evidence:
            self._enter_evidence(evidence)

        # Initialise messages to uniform
        self._initialize_uniform_messages()

        converged = False
        for iteration in range(self.max_iterations):
            old_messages = {
                k: msg.potential.values.copy()
                for k, msg in self._messages.items()
            }

            # Forward and backward over all edges
            for clique in self.tree.cliques:
                for nb_idx in self.tree.neighbors(clique.index):
                    self._compute_loopy_message(
                        clique.index, nb_idx, damping
                    )

            # Check convergence
            max_diff = 0.0
            for k, msg in self._messages.items():
                if k in old_messages:
                    diff = float(
                        np.max(np.abs(msg.potential.values - old_messages[k]))
                    )
                    max_diff = max(max_diff, diff)

            self._stats.convergence_iterations = iteration + 1
            if max_diff < self.convergence_tol:
                converged = True
                break

        # Update beliefs
        self._update_beliefs_from_messages()
        self._stats.total_time_s = time.time() - t0
        return self._stats

    def _compute_loopy_message(
        self, sender_idx: int, receiver_idx: int, damping: float
    ) -> None:
        """Compute a single loopy-BP message with damping."""
        sender = self.tree.cliques[sender_idx]
        sep = self.tree.get_separator(sender_idx, receiver_idx)
        if sender.potential is None or sep is None:
            return

        combined = sender.potential.copy()
        for nb_idx in self.tree.neighbors(sender_idx):
            if nb_idx == receiver_idx:
                continue
            incoming = self._messages.get((nb_idx, sender_idx))
            if incoming is not None:
                combined = combined.multiply(incoming.potential)

        margin_out = [
            v for v in combined.variables if v not in sep.variables
        ]
        new_potential = combined.marginalize(margin_out)

        # Normalise to avoid numerical blow-up
        new_potential = new_potential.normalize()

        # Apply damping
        old_msg = self._messages.get((sender_idx, receiver_idx))
        if old_msg is not None and damping > 0:
            damped_values = (
                damping * old_msg.potential.values
                + (1 - damping) * new_potential.values
            )
            new_potential = PotentialTable(
                new_potential.variables,
                new_potential.cardinalities,
                damped_values,
                log_space=new_potential.log_space,
            )

        msg = Message(
            sender_idx=sender_idx,
            receiver_idx=receiver_idx,
            separator_vars=frozenset(sep.variables),
            potential=new_potential,
        )
        self._messages[(sender_idx, receiver_idx)] = msg
        self._stats.messages_sent += 1

    def _initialize_uniform_messages(self) -> None:
        """Initialize all messages to uniform distributions."""
        for clique in self.tree.cliques:
            for nb_idx in self.tree.neighbors(clique.index):
                sep = self.tree.get_separator(clique.index, nb_idx)
                if sep is None:
                    continue
                sep_vars = sorted(sep.variables)
                cards = {v: self.tree.cardinalities[v] for v in sep_vars}
                uniform = PotentialTable(
                    sep_vars, cards, log_space=self.use_log_space
                )
                self._messages[(clique.index, nb_idx)] = Message(
                    sender_idx=clique.index,
                    receiver_idx=nb_idx,
                    separator_vars=frozenset(sep.variables),
                    potential=uniform,
                )

    # ------------------------------------------------------------------ #
    #  Marginal extraction
    # ------------------------------------------------------------------ #

    def compute_message(
        self, sender_idx: int, receiver_idx: int
    ) -> PotentialTable:
        """Compute a single message from *sender* to *receiver*
        without modifying potentials.  Useful for external callers."""
        sender = self.tree.cliques[sender_idx]
        sep = self.tree.get_separator(sender_idx, receiver_idx)
        if sender.potential is None or sep is None:
            raise ValueError("Clique potential or separator not initialized")

        combined = sender.potential.copy()
        for nb_idx in self.tree.neighbors(sender_idx):
            if nb_idx == receiver_idx:
                continue
            incoming = self._messages.get((nb_idx, sender_idx))
            if incoming is not None:
                combined = combined.multiply(incoming.potential)

        margin_out = [
            v for v in combined.variables if v not in sep.variables
        ]
        return combined.marginalize(margin_out)

    def update_potential(
        self, clique_idx: int, message: PotentialTable
    ) -> None:
        """Multiply a message into a clique's potential in-place."""
        clique = self.tree.cliques[clique_idx]
        if clique.potential is None:
            clique.potential = message.copy()
        else:
            clique.potential = clique.potential.multiply(message)

    def get_marginal(self, variable: str) -> PotentialTable:
        """Extract the marginal distribution for a single variable
        from the calibrated tree."""
        # Find the smallest clique containing the variable
        best_clique: Optional[CliqueNode] = None
        best_size = float("inf")
        for clique in self.tree.cliques:
            if variable in clique.variables and clique.size < best_size:
                best_clique = clique
                best_size = clique.size

        if best_clique is None or best_clique.potential is None:
            raise ValueError(
                f"Variable '{variable}' not found in any calibrated clique"
            )

        margin_out = [v for v in best_clique.potential.variables if v != variable]
        marginal = best_clique.potential.marginalize(margin_out)
        return marginal.normalize()

    def get_joint_marginal(self, variables: Set[str]) -> PotentialTable:
        """Extract the joint marginal over a set of variables.

        The variables must all belong to some single clique (guaranteed
        by the junction-tree property for any subset of a clique).
        """
        clique = self.tree.clique_containing(variables)
        if clique is None or clique.potential is None:
            raise ValueError(
                f"No clique contains all variables: {variables}"
            )

        margin_out = [
            v for v in clique.potential.variables if v not in variables
        ]
        joint = clique.potential.marginalize(margin_out)
        return joint.normalize()

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _convert_to_log_space(self) -> None:
        """Convert all clique potentials to log-space."""
        for clique in self.tree.cliques:
            if clique.potential is not None and not clique.potential.log_space:
                clique.potential = clique.potential.to_log_space()

    def _make_cache_key(
        self, sender_idx: int, receiver_idx: int
    ) -> CacheKey:
        sender = self.tree.cliques[sender_idx]
        receiver = self.tree.cliques[receiver_idx]
        return CacheKey(
            sender=sender.variables,
            receiver=receiver.variables,
            evidence_sig=self._evidence_sig,
            intervention_sig=self._intervention_sig,
        )

    def set_evidence_signature(self, sig: str) -> None:
        self._evidence_sig = sig

    def set_intervention_signature(self, sig: str) -> None:
        self._intervention_sig = sig

    def __repr__(self) -> str:
        return (
            f"MessagePasser(variant={self.variant.value}, "
            f"msgs={len(self._messages)})"
        )
