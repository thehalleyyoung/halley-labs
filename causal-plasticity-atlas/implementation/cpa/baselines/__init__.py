"""CPA baselines subpackage.

Evaluation baselines for comparing the CPA engine against existing
causal discovery methods including independent per-context, pooled,
ICP, CD-NOD, JCI, GES, and linear SEM approaches.

Modules
-------
ind_phc
    Independent per-context + post-hoc comparison.
pooled
    Pooled data baseline (ignore context).
icp_baseline
    Invariant Causal Prediction baseline.
cd_nod
    CD-NOD baseline.
jci
    Joint Causal Inference baseline.
ges_baseline
    GES-based baseline with multi-context extension.
lsem_pool
    Linear SEM pooled estimation.
"""

from cpa.baselines.ind_phc import IndependentPHC
from cpa.baselines.pooled import PooledBaseline
from cpa.baselines.icp_baseline import ICPBaseline
from cpa.baselines.cd_nod import CDNODBaseline
from cpa.baselines.jci import JCIBaseline
from cpa.baselines.ges_baseline import GESBaseline
from cpa.baselines.lsem_pool import LSEMPooled

__all__ = [
    # ind_phc.py
    "IndependentPHC",
    # pooled.py
    "PooledBaseline",
    # icp_baseline.py
    "ICPBaseline",
    # cd_nod.py
    "CDNODBaseline",
    # jci.py
    "JCIBaseline",
    # ges_baseline.py
    "GESBaseline",
    # lsem_pool.py
    "LSEMPooled",
]
