"""CPA Visualization — Atlas, DAG, and descriptor plotting.

Provides matplotlib-based visualizations for the Causal-Plasticity
Atlas, including heatmaps, DAG overlays, descriptor scatter plots,
and tipping-point timelines.

All visualizers support both interactive display and file output.

Modules
-------
atlas_viz
    AtlasVisualizer for overall atlas visualization.
dag_viz
    DAGVisualizer for causal graph drawing.
descriptor_viz
    DescriptorVisualizer for plasticity descriptor plots.
"""

from cpa.visualization.atlas_viz import AtlasVisualizer
from cpa.visualization.dag_viz import DAGVisualizer
from cpa.visualization.descriptor_viz import DescriptorVisualizer

__all__ = [
    "AtlasVisualizer",
    "DAGVisualizer",
    "DescriptorVisualizer",
]
