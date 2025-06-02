"""
neurophorm: Topological Data Analysis and Visualization Tools for Brain Connectivity

This package provides functions for computing and visualizing topological features
from brain connectivity data, including persistence diagrams, Betti curves, entropy,
distances, amplitudes, and more.
"""

from .persistence import (
    corr_to_distance_matrices,
    rips_persistence_diagrams,
    betti_curves,
    persistence_entropy,
    diagram_distances,
    diagram_amplitudes,
    persistence_images,
    save_tda_results,
    individual_tda_features,
    batch_tda_features,
    load_tda_results,
    load_and_interpolate_betti_curves,
)

from .visualization import (
    plot_betti_curves,
    plot_p_values,
    plot_grouped_p_value_heatmaps,
    plot_grouped_distance_heatmaps,
    plot_swarm_violin,
    plot_kde_dist,
)

from .node_removal import (
    node_removal_persistence,
    node_removal_differences,
    load_removal_data,
)

__all__ = [
    # persistence
    "corr_to_distance_matrices",
    "rips_persistence_diagrams",
    "betti_curves",
    "persistence_entropy",
    "diagram_distances",
    "diagram_amplitudes",
    "persistence_images",
    "save_tda_results",
    "individual_tda_features",
    "batch_tda_features",
    "load_tda_results",
    "load_and_interpolate_betti_curves",
    # visualization
    "plot_betti_curves",
    "plot_p_values",
    "plot_grouped_p_value_heatmaps",
    "plot_grouped_distance_heatmaps",
    "plot_swarm_violin",
    "plot_kde_dist",
    # node_removal
    "node_removal_persistence",
    "node_removal_differences",
    "load_removal_data",
]
