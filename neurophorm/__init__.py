"""
neurophorm
Topological data analysis utilities with logging-enabled submodules.

Submodules
----------
persistence
    Core TDA computations (diagrams, Betti curves, entropy, distances, amplitudes, images),
    batch/individual pipelines, saving/loading helpers.
visualization
    Publication-ready plotting helpers for Betti curves, p-values, grouped heatmaps, KDEs, etc.
node_removal
    Node-removal persistence analysis and aggregation utilities.

Logging
-------
By default, modules attach a StreamHandler if none exists and log at INFO.
Use `configure_logging` to control log level and handlers from your app or tests.
"""

from __future__ import annotations

import logging
from typing import Optional, Union, IO

# Expose high-level APIs
from .persistence import (  # noqa: F401
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
    compute_betti_stat_features,
)
from .visualization import (  # noqa: F401
    plot_betti_curves,
    plot_p_values,
    plot_grouped_p_value_heatmaps,
    plot_grouped_distance_heatmaps,
    plot_swarm_violin,
    plot_kde_dist,
    plot_betti_stats_pvalues,
    plot_node_removal,
    plot_node_removal_p_values,
)
from .node_removal import (  # noqa: F401
    node_removal_persistence,
    node_removal_differences,
    load_removal_data,
)

from .collect_pvalue_counts import ( # noqa: F401
    collect_pairwise_pvalues,
    collect_pvalue_counts,
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
    "compute_betti_stat_features",
    # visualization
    "plot_betti_curves",
    "plot_p_values",
    "plot_grouped_p_value_heatmaps",
    "plot_grouped_distance_heatmaps",
    "plot_swarm_violin",
    "plot_kde_dist",
    "plot_betti_stats_pvalues",
    "plot_node_removal",
    "plot_node_removal_p_values",
    # node_removal
    "node_removal_persistence",
    "node_removal_differences",
    "load_removal_data",
    # pvalue counts
    "collect_pvalue_counts",
    "collect_pairwise_pvalues",
    # utils
    "configure_logging",
]

__version__ = "0.1.0"


def configure_logging(
    level: int = logging.INFO,
    stream: Optional[IO[str]] = None,
    filename: Optional[str] = None,
    fmt: str = "[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    datefmt: Optional[str] = None,
    propagate: bool = False,
) -> None:
    """
    Configure package-wide logging.

    Parameters
    ----------
    level : int
        Logging level (e.g., logging.DEBUG, logging.INFO).
    stream : IO[str], optional
        Stream for a StreamHandler (defaults to sys.stderr if handler added).
    filename : str, optional
        If provided, also attach a FileHandler to this path.
    fmt : str
        Log message format string.
    datefmt : str, optional
        Date/time format string.
    propagate : bool
        Whether package loggers should propagate to root handlers.

    Notes
    -----
    This function configures the top-level package logger ("neurophorm") and
    leaves submodule-specific handlers intact. You can call it multiple times;
    it will avoid duplicate handlers of the same class/target.
    """
    logger = logging.getLogger(__name__.split(".")[0])  # "neurophorm"
    logger.setLevel(level)
    logger.propagate = propagate

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # Helper to check if a similar handler already exists
    def _has_handler(cls, target=None) -> bool:
        for h in logger.handlers:
            if isinstance(h, cls):
                if target is None:
                    return True
                if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is target:
                    return True
                if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == target:
                    return True
        return False

    if stream is not None and not _has_handler(logging.StreamHandler, stream):
        sh = logging.StreamHandler(stream)
        sh.setFormatter(formatter)
        sh.setLevel(level)
        logger.addHandler(sh)
    elif stream is None and not _has_handler(logging.StreamHandler):
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        sh.setLevel(level)
        logger.addHandler(sh)

    if filename is not None and not _has_handler(logging.FileHandler, filename):
        fh = logging.FileHandler(filename)
        fh.setFormatter(formatter)
        fh.setLevel(level)
        logger.addHandler(fh)
