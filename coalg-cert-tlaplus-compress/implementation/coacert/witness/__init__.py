"""
Bisimulation witness emitter for CoaCert-TLA.

Provides Merkle-tree-backed certificates that attest correctness of
quotient constructions produced by the bisimulation engine.
"""

from .merkle_tree import (
    MerkleTree,
    LeafNode,
    InternalNode,
    SparseMerkleTree,
    MerkleProof,
)
from .equivalence_binding import EquivalenceBinding, ClassBinding
from .transition_witness import (
    TransitionWitness,
    StutterWitness,
    FairnessWitness,
    WitnessSet,
)
from .hash_chain import HashChain, EquivalenceBlock, TransitionBlock, FairnessBlock
from .witness_format import WitnessFormat
from .compact_repr import CompactWitness, BloomFilter

__all__ = [
    "MerkleTree",
    "LeafNode",
    "InternalNode",
    "SparseMerkleTree",
    "MerkleProof",
    "EquivalenceBinding",
    "ClassBinding",
    "TransitionWitness",
    "StutterWitness",
    "FairnessWitness",
    "WitnessSet",
    "HashChain",
    "EquivalenceBlock",
    "TransitionBlock",
    "FairnessBlock",
    "WitnessFormat",
    "CompactWitness",
    "BloomFilter",
]
