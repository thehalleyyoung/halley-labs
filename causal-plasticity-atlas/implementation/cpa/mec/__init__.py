"""CPA MEC subpackage.

Markov Equivalence Class operations including CPDAG construction,
PAG handling, Meek orientation rules, DAG enumeration, and
d-separation algorithms.

Modules
-------
cpdag
    CPDAG construction and operations.
pag
    Partial Ancestral Graph handling.
orientation
    Meek and Zhang orientation rules.
enumeration
    DAG enumeration within a MEC.
separation
    D-separation and m-separation algorithms.
"""

from __future__ import annotations

from cpa.mec.cpdag import CPDAG, dag_to_cpdag, cpdag_to_dags
from cpa.mec.pag import PAG, dag_to_pag
from cpa.mec.orientation import MeekRules, ZhangRules, apply_orientation_rules
from cpa.mec.enumeration import MECEnumerator, MECSampler, mec_size
from cpa.mec.separation import (
    d_separation,
    m_separation,
    find_dsep_set,
    markov_blanket,
)

__all__ = [
    # cpdag.py
    "CPDAG",
    "dag_to_cpdag",
    "cpdag_to_dags",
    # pag.py
    "PAG",
    "dag_to_pag",
    # orientation.py
    "MeekRules",
    "ZhangRules",
    "apply_orientation_rules",
    # enumeration.py
    "MECEnumerator",
    "MECSampler",
    "mec_size",
    # separation.py
    "d_separation",
    "m_separation",
    "find_dsep_set",
    "markov_blanket",
]
