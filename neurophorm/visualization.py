"""
visualization.py

This module provides visualization utilities for Topological Data Analysis (TDA) results.
It includes functions for plotting Betti curves, p-value heatmaps, distance matrices,
swarm/violin plots, and kernel density estimates for TDA features across groups.

Dependencies
    numpy, pandas, pillow (PIL), matplotlib, seaborn, scipy
"""

from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import logging
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import ttest_ind, wilcoxon, shapiro, levene, mannwhitneyu
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import ListedColormap
import seaborn as sns
import warnings
import copy

# -------------------------------------------------------------------------
# Global Matplotlib style settings
# -------------------------------------------------------------------------
def set_mpl_style(usetex: bool = False, interactive: bool = False) -> None:
    if not interactive:
        plt.ioff()  # disable interactive popups

    # Base style
    mpl.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.unicode_minus": False,
    })

    if usetex:
        # Requires a LaTeX install; ensures Times-like text and math
        mpl.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times"],
            "text.latex.preamble": r"\usepackage{newtxtext,newtxmath}",
        })
    else:
        # Pure Matplotlib fallback close to Times
        mpl.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times"],
            "mathtext.fontset": "stix",
        })

set_mpl_style(usetex=True, interactive=False)


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

warnings.filterwarnings('ignore', category=RuntimeWarning)


def _infer_dimensions(data: Dict[str, Dict[str, npt.NDArray]]) -> List[int]:
    """
    Infer homology dimensions from persistence diagrams or Betti curves.

    This helper inspects the provided dictionary of group data and extracts the set
    of unique homology dimensions, either from the third column of persistence diagrams
    or the shape of Betti curves.

    Parameters
    ----------
    data : Dict[str, Dict[str, np.ndarray]]
        Mapping of group label → feature dict. Each dict must contain either
        'persistence_diagrams' (list of arrays of shape (m, 3)) or "betti_curves_shared"
        (array of shape (n_dims, n_bins) or (n_samples, n_dims, n_bins)).

    Returns
    -------
    List[int]
        Sorted list of unique homology dimensions found.

    Raises
    ------
    ValueError
        If no dimensions can be inferred from the input.

    Examples
    --------
    >>> import numpy as np
    >>> data = {"A": {"persistence_diagrams": [np.array([[0, 1, 0], [0, 2, 1]])]}}
    >>> _infer_dimensions(data)
    [0, 1]
    """
    logger.debug("_infer_dimensions: start | keys=%s", list(data.keys()))
    dimensions = set()
    for label, label_data in data.items():
        if "persistence_diagrams" in label_data:
            for diagram in label_data["persistence_diagrams"]:
                if diagram.size > 0:  # Ensure diagram is not empty
                    dimensions.update(diagram[:, 2].astype(int))
        elif "betti_curves_shared" in label_data:
            betti_curves = label_data["betti_curves_shared"]
            if betti_curves.ndim == 3:
                dimensions.update(range(betti_curves.shape[1]))
            elif betti_curves.ndim == 2:
                dimensions.update(range(betti_curves.shape[0]))
    if not dimensions:
        logger.error("_infer_dimensions: no dimensions found")
        raise ValueError("Could not infer dimensions from data. Ensure 'persistence_diagrams' or 'betti_curves_shared' are present.")
    dims = sorted(dimensions)
    logger.debug("_infer_dimensions: done | dims=%s", dims)
    return dims


def _compute_betti_auc(
    data: Dict[str, Dict[str, np.ndarray]],
    labels: List[str],
    dimensions: Optional[List[int]] = None
) -> Dict[str, np.ndarray]:
    """
    Compute area under the curve (AUC) for Betti curves.

    For each group and each homology dimension, integrates the Betti curve using
    the trapezoidal rule, producing one scalar AUC per sample per dimension.

    Parameters
    ----------
    data : Dict[str, Dict[str, np.ndarray]]
        Mapping of group label → feature dict containing:
            "betti_curves_shared" : ndarray of shape (n_samples, n_dims, n_bins)
            "betti_x_shared"      : ndarray of shape (n_dims, n_bins) or (n_bins,)
    labels : List[str]
        Group labels to process.
    dimensions : List[int], optional
        Dimensions to compute AUC for. If None, inferred automatically.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping group label → array of shape (n_samples, n_dims_selected).

    Raises
    ------
    ValueError
        If "betti_curves_shared" or "betti_x_shared" are missing for a label.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 1, 5)
    >>> curves = np.ones((2, 1, 5))  # 2 samples, 1 dim, 5 bins
    >>> data = {"A": {"betti_curves": curves, "betti_x": x}}
    >>> aucs = _compute_betti_auc(data, ["A"], dimensions=[0])
    >>> aucs["A"].shape
    (2, 1)
    """

    logger.info("_compute_betti_auc: start | labels=%s | dims=%s", labels, dimensions)
    if dimensions is None:
        dimensions = _infer_dimensions(data)  # Assume _infer_dimensions is defined

    auc_data = {}
    for label in labels:
        if "betti_curves_shared" not in data[label] or "betti_x_shared" not in data[label]:
            logger.error("_compute_betti_auc: missing betti data for label=%s", label)
            raise ValueError(f"Missing betti_curves_shared or betti_x_shared for {label}")

        curves = data[label]["betti_curves_shared"]  # Shape: (n_samples, n_dims, n_bins)
        x = data[label]["betti_x_shared"]  # Shape: (n_bins,) or (n_dims, n_bins)

        n_samples, n_dims, n_bins = curves.shape
        auc = np.zeros((n_samples, len(dimensions)))

        for dim_idx, dim in enumerate(dimensions):
            if dim >= n_dims:
                logger.debug("_compute_betti_auc: skip dim %d >= n_dims %d", dim, n_dims)
                continue
            # Select x-values for this dimension
            x_dim = x[dim] if hasattr(x, "ndim") and x.ndim > 1 else x
            for sample_idx in range(n_samples):
                y = curves[sample_idx, dim, :]
                valid_mask = ~np.isnan(y) & ~np.isnan(x_dim)
                if np.sum(valid_mask) < 2:  # Need at least 2 points for AUC
                    auc[sample_idx, dim_idx] = np.nan
                else:
                    auc[sample_idx, dim_idx] = np.trapz(y[valid_mask], x_dim[valid_mask])

        auc_data[label] = auc
        logger.debug("_compute_betti_auc: label=%s | auc_shape=%s", label, auc.shape)

    logger.info("_compute_betti_auc: done")
    return auc_data


def plot_betti_curves(
    data: Dict[str, Dict[str, npt.NDArray]],
    dimensions: Optional[List[int]] = None,
    labels: Optional[List[str]] = None,
    label_styles: Optional[Dict[str, Tuple[str, str]]] = None,
    output_directory: Union[str, Path] = "./",
    show_plot: bool = True,
    save_plot: bool = False,
    save_format: str = "pdf",
    same_size: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (7, None),
) -> None:
    """
    Plot mean Betti curves with standard deviation shading.

    Produces one subplot per homology dimension, showing mean Betti curves for each
    group, shaded by ± standard error. Supports multiple groups and custom styles.

    Parameters
    ----------
    data : Dict[str, Dict[str, np.ndarray]]
        Grouped Betti curve data. Each group must contain:
            "betti_curves_shared" : ndarray, shape (n_samples, n_dims, n_bins) or (n_dims, n_bins)
            'betti_x_shared'      : ndarray, shape (n_dims, n_bins)
    dimensions : List[int], optional
        Dimensions to plot. If None, inferred.
    labels : List[str], optional
        Subset of groups to plot. Defaults to all keys of `data`.
    label_styles : Dict[str, Tuple[str, str]], optional
        Mapping label → (color, linestyle). If None, generated automatically.
    output_directory : str or Path, default="./"
        Where to save plots if `save_plot=True`.
    show_plot : bool, default=True
        Whether to display the plot interactively.
    save_plot : bool, default=False
        Whether to save the plot.
    save_format : str, default="pdf"
        File format to save ("pdf", "png", "svg", "jpg").
    same_size : bool, default=False
        If True, enforce same x/y axis limits across dimensions.
    xlim, ylim : tuple, optional
        Axis limits.
    figsize : tuple, default=(7, None)
        Width × height in inches. Height auto-computed if None.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If `labels` is not a list.
    ValueError
        If requested dimensions are invalid, or save_format unsupported.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 1, 10)
    >>> curves = np.random.rand(5, 1, 10)
    >>> data = {"A": {"betti_curves_shared": curves, "betti_x_shared": np.array([x])}}
    >>> plot_betti_curves(data, dimensions=[0], show_plot=False)
    """

    logger.info(
        "plot_betti_curves: start | groups=%s | dims=%s | save=%s(%s) | show=%s",
        None if labels is None else labels, dimensions, save_plot, save_format, show_plot
    )
    labels = labels if labels is not None else list(data.keys())
    if not isinstance(labels, list):
        logger.error("plot_betti_curves: labels must be a list of strings")
        raise TypeError("labels must be a list of strings")

    data = {k: data[k] for k in labels if k in data}

    # Infer dimensions if not provided
    if dimensions is None:
        dimensions = _infer_dimensions(data)

    # Validate dimensions
    available_dims = _infer_dimensions(data)
    invalid_dims = [d for d in dimensions if d not in available_dims]
    if invalid_dims:
        logger.error("plot_betti_curves: invalid dims %s | available=%s", invalid_dims, available_dims)
        raise ValueError(f"Specified dimensions {invalid_dims} not found in data. Available: {available_dims}")

    if label_styles:
        label_styles = {k: label_styles[k] for k in labels if k in label_styles}
    else:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        linestyles = ['-', '--', '-.', ':']
        label_styles = {label: (colors[i % len(colors)], linestyles[i % len(linestyles)])
                        for i, label in enumerate(data.keys())}

    # Compute height if not provided
    height = figsize[1] if figsize[1] is not None else len(dimensions) * 2.5
    fig, axs = plt.subplots(len(dimensions), 1, figsize=(figsize[0], height))
    axs = [axs] if len(dimensions) == 1 else axs

    # Compute global limits if same_size is True and no custom limits provided
    if same_size and (xlim is None or ylim is None):
        all_x = []
        all_y = []
        for label in data:
            for dim in dimensions:
                betti_curves = data[label]["betti_curves_shared"]
                if betti_curves.ndim == 3:
                    mean_curve = np.mean(betti_curves[:, dim, :], axis=0)
                    std_curve = np.std(betti_curves[:, dim, :], axis=0) / np.sqrt(betti_curves.shape[0])
                    all_y.extend(mean_curve - std_curve)
                    all_y.extend(mean_curve + std_curve)
                elif betti_curves.ndim == 2:
                    mean_curve = betti_curves[dim, :]
                    all_y.extend(mean_curve)
                filtration_values = data[label]["betti_x_shared"][dim]
                all_x.extend(filtration_values)

        computed_xlim = (min(all_x), max(all_x)) if xlim is None else xlim
        computed_ylim = (min(0, min(all_y)), max(all_y) * 1.1) if ylim is None else ylim
    else:
        computed_xlim = xlim
        computed_ylim = ylim

    for i, dim in enumerate(dimensions):
        for label, (color, linestyle) in label_styles.items():
            betti_curves = data[label]["betti_curves_shared"]
            filtration_values = data[label]["betti_x_shared"][dim]
            if betti_curves.ndim == 3:
                mean_curve = np.mean(betti_curves[:, dim, :], axis=0)
                std_curve = np.std(betti_curves[:, dim, :], axis=0) / np.sqrt(betti_curves.shape[0])
                axs[i].fill_between(filtration_values, mean_curve - std_curve, mean_curve + std_curve,
                                    alpha=0.3, color=color)
            elif betti_curves.ndim == 2:
                mean_curve = betti_curves[dim, :]

            axs[i].plot(filtration_values, mean_curve, c=color, ls=linestyle,
                        label=label.replace("_", " "))

        axs[i].set_xlabel("Filtration Parameter")
        axs[i].set_ylabel("Betti Number")
        axs[i].set_title(fr"Mean Betti Curves $H_{dim}$")
        axs[i].set_xlim(computed_xlim)
        axs[i].set_ylim(computed_ylim)

    if len(label_styles) > 1:
        fig.legend(*axs[0].get_legend_handles_labels(), loc='upper right')

    plt.suptitle("Mean Betti Curves by Group")
    plt.tight_layout()

    if save_plot and output_directory:
        output_directory = Path(output_directory)
        plot_dir = output_directory / "betti_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        valid_formats = ['pdf', 'png', 'svg', 'jpg']
        if save_format.lower() not in valid_formats:
            logger.error("plot_betti_curves: invalid save_format=%s", save_format)
            raise ValueError(f"save_format must be one of {valid_formats}")
        save_path = plot_dir / f"mean_betti_curves.{save_format.lower()}"
        plt.savefig(save_path, format=save_format.lower(), bbox_inches='tight')
        logger.info("plot_betti_curves: saved plot to %s", save_path)

    if show_plot:
        plt.show()
    plt.close("all")
    logger.info("plot_betti_curves: done")


def plot_p_values(
    data: Dict[str, Dict[str, npt.NDArray]],
    feature_name: str,
    labels: Optional[List[str]] = None,
    dimensions: Optional[List[int]] = None,
    output_directory: Union[str, Path] = "./",
    test: str = "auto",
    show_plot: bool = True,
    save_plot: bool = False,
    save_format: str = "pdf",
    figsize: Tuple[float, float] = (None, None),
    heatmap_kwargs: Optional[Dict] = None
) -> List[pd.DataFrame]:
    """
    Compute and visualize p-values comparing TDA features between groups.

    Performs pairwise group comparisons across specified homology dimensions using
    t-tests or Wilcoxon signed-rank tests (auto-selected if `test="auto"`). Produces
    heatmaps of p-values.

    For Betti curves, compare AUCs (area under curve) instead of curve means.

    Parameters
    ----------
    data : Dict[str, Dict[str, np.ndarray]]
        Grouped TDA features. Each group must contain `feature_name`.
    feature_name : str
        Name of feature to compare ("betti_curves_shared", "persistence_entropy", etc.).
    labels : List[str], optional
        Groups to include. Defaults to all.
    dimensions : List[int], optional
        Homology dimensions. Inferred if None.
    output_directory : str or Path
        Directory for saving plots.
    test : {"t_test", "wilcoxon", "auto"}, default="auto"
        Which statistical test to use.
    show_plot : bool, default=True
        Show interactively.
    save_plot : bool, default=False
        Save figure to disk.
    save_format : str, default="pdf"
        File format to save.
    figsize : tuple, default=(None, None)
        Width × height in inches. Auto-computed if None.
    heatmap_kwargs : dict, optional
        Extra options for seaborn.heatmap.

    Returns
    -------
    List[pd.DataFrame]
        One symmetric matrix of p-values per dimension.

    Raises
    ------
    ValueError
        If feature missing, shapes mismatch, or invalid test/save_format.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.rand(5, 1)
    >>> data = {"A": {"persistence_entropy": x}, "B": {"persistence_entropy": x}}
    >>> res = plot_p_values(data, "persistence_entropy", dimensions=[0], show_plot=False)
    >>> isinstance(res, list)
    True
    """

    logger.info(
        "plot_p_values: start | feature=%s | test=%s | save=%s | groups=%s",
        feature_name, test, save_plot, None if labels is None else labels
    )
    labels = labels if labels is not None else list(data.keys())
    data = {k: data[k] for k in labels if k in data}

    # Validate feature_name
    if not all(feature_name in data[label] for label in labels):
        logger.error("plot_p_values: feature '%s' missing in some groups", feature_name)
        raise ValueError(f"Feature '{feature_name}' not found in all group data")

    # Infer dimensions if not provided
    if dimensions is None:
        dimensions = _infer_dimensions(data)

    # Validate dimensions
    available_dims = _infer_dimensions(data)
    invalid_dims = [d for d in dimensions if d not in available_dims]
    if invalid_dims:
        logger.error("plot_p_values: invalid dims %s | available=%s", invalid_dims, available_dims)
        raise ValueError(f"Specified dimensions {invalid_dims} not found in data. Available: {available_dims}")

    p_values = [pd.DataFrame(np.ones((len(labels), len(labels))), index=labels, columns=labels)
                for _ in dimensions]
    cmap = ListedColormap("#3b4cc0")

    if feature_name == "betti_curves_shared":
        data = _compute_betti_auc(data, labels, dimensions)

    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels[i+1:], i+1):
            if feature_name == "betti_curves_shared":
                curves1 = data[label1]
                curves2 = data[label2]
            else:
                curves1 = data[label1][feature_name]
                curves2 = data[label2][feature_name]

            # Validate shapes
            if curves1.shape != curves2.shape:
                logger.error("plot_p_values: shape mismatch between %s and %s", label1, label2)
                raise ValueError(f"Incompatible shapes for '{feature_name}' between {label1} and {label2}")

            for dim_idx, dim in enumerate(dimensions):
                data1 = curves1[:, dim, :].mean(axis=0) if curves1.ndim > 2 else curves1[:, dim]
                data2 = curves2[:, dim, :].mean(axis=0) if curves2.ndim > 2 else curves2[:, dim]

                test_to_use = test
                if test == "auto":
                    _, p1 = shapiro(data1)
                    _, p2 = shapiro(data2)
                    test_to_use = "t_test" if p1 > 0.05 and p2 > 0.05 else "wilcoxon"

                if test_to_use == "t_test":
                    _, p = levene(data1, data2)
                    _, p_val = ttest_ind(data1, data2, equal_var=p > 0.05)
                elif test_to_use == "wilcoxon":
                    _, p_val = wilcoxon(data1, data2)
                else:
                    logger.error("plot_p_values: invalid test '%s'", test_to_use)
                    raise ValueError("Test must be 't_test', 'wilcoxon', or 'auto'")

                p_values[dim_idx].loc[label1, label2] = p_values[dim_idx].loc[label2, label1] = p_val

    # Compute figure size if not provided
    width = figsize[0] if figsize[0] is not None else len(labels) * 1.2 * len(dimensions)
    height = figsize[1] if figsize[1] is not None else len(labels) * 1.2
    fig, axs = plt.subplots(1, len(dimensions), figsize=(width, height))
    axs = [axs] if len(dimensions) == 1 else axs

    heatmap_kwargs = heatmap_kwargs or {}

    for dim_idx, dim in enumerate(dimensions):
        mask = p_values[dim_idx] < 0.05
        pretty_labels = [l.replace("_", " ") for l in labels]
        sns.heatmap(p_values[dim_idx], ax=axs[dim_idx], xticklabels=pretty_labels, yticklabels=pretty_labels,
                    annot=True, fmt=".2f", cbar=False, mask=mask, vmin=0, vmax=1, **heatmap_kwargs)
        sns.heatmap(p_values[dim_idx], ax=axs[dim_idx], xticklabels=pretty_labels, yticklabels=pretty_labels,
                    annot=True, fmt=".2f", cbar=False, cmap=cmap, mask=~mask, **heatmap_kwargs)
        axs[dim_idx].set_title(fr"$H_{dim}$", fontsize=18)

    if feature_name == "betti_curves_shared":
        feature_name = "Betti Curve AUC"

    title = (f"p-Values from {'T-Test or Wilcoxon' if test == 'auto' else test} "
             f"for {feature_name.replace('_', ' ')}")
    plt.suptitle(title)
    plt.tight_layout()

    if save_plot and output_directory:
        output_directory = Path(output_directory)
        plot_dir = output_directory / "p_value_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        valid_formats = ['pdf', 'png', 'svg', 'jpg']
        if save_format.lower() not in valid_formats:
            logger.error("plot_p_values: invalid save_format=%s", save_format)
            raise ValueError(f"save_format must be one of {valid_formats}")
        save_path = plot_dir / f"{test}_p_values_for_{feature_name}.{save_format.lower()}"
        plt.savefig(save_path, format=save_format.lower(), bbox_inches='tight')
        logger.info("plot_p_values: saved plot to %s", save_path)

    if show_plot:
        plt.show()
    plt.close("all")
    logger.info("plot_p_values: done")
    return p_values


def plot_grouped_p_value_heatmaps(
    p_values: List[pd.DataFrame],
    group_ranges: Dict[str, Tuple[List[int], List[str]]],
    name: str,
    ncol: int = 2,
    dimensions: Optional[List[int]] = None,
    output_directory: Union[str, Path] = "./",
    show_plot: bool = True,
    save_plot: bool = False,
    save_format: str = "pdf",
    subplot_layout: Optional[List[Tuple[str, str]]] = None,
    figsize: Tuple[float, float] = (None, None),
    heatmap_kwargs: Optional[Dict] = None
) -> None:
    """
    Visualize grouped subsets of p-values as heatmaps.

    Given precomputed p-value DataFrames (one per dimension), generate grouped
    heatmaps for specified subsets of labels.

    Parameters
    ----------
    p_values : List[pd.DataFrame]
        Symmetric p-value matrices (output of `plot_p_values`).
    group_ranges : Dict[str, Tuple[List[int], List[str]]]
        Mapping group_name → (indices, labels).
    name : str
        Descriptor for output files and titles.
    ncol : int, default=2
        Number of subplot columns.
    dimensions : List[int], optional
        Dimensions to include. Defaults to all.
    output_directory : str or Path
        Directory to save plots.
    show_plot : bool, default=True
        Show interactively.
    save_plot : bool, default=False
        Save to disk.
    save_format : str, default="pdf"
        File format.
    subplot_layout : List[Tuple[str, str]], optional
        Custom mosaic layout as (key, title) pairs.
    figsize : tuple, optional
        Width × height in inches.
    heatmap_kwargs : dict, optional
        Extra arguments for seaborn.heatmap.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If group_ranges empty, p_values empty, or dimensions invalid.

    Examples
    --------
    >>> import pandas as pd
    >>> mat = pd.DataFrame([[1,0.1],[0.1,1]], index=["A","B"], columns=["A","B"])
    >>> plot_grouped_p_value_heatmaps([mat], {"AB": ([0,1], ["A","B"])}, "test", show_plot=False)
    """

    logger.info(
        "plot_grouped_p_value_heatmaps: start | groupsets=%d | dims=%s | save=%s",
        len(group_ranges), dimensions, save_plot
    )
    if not group_ranges:
        logger.error("plot_grouped_p_value_heatmaps: empty group_ranges")
        raise ValueError("group_ranges dictionary cannot be empty")
    if not p_values:
        logger.error("plot_grouped_p_value_heatmaps: empty p_values")
        raise ValueError("p_values list cannot be empty")

    # Infer dimensions if not provided
    if dimensions is None:
        dimensions = list(range(len(p_values)))

    # Validate dimensions
    if max(dimensions) >= len(p_values):
        logger.error("plot_grouped_p_value_heatmaps: dimensions exceed available")
        raise ValueError(f"Specified dimensions {dimensions} exceed number of p-value DataFrames {len(p_values)}")

    cmap = ListedColormap("#3b4cc0")
    extract_subset = lambda dist, indices: dist[np.ix_(indices, indices)]

    # Default subplot layout if none provided
    if subplot_layout is None:
        subplot_layout = [(key, key) for key in group_ranges.keys()]

    # Create mosaic layout dynamically
    n_subplots = len(subplot_layout)
    rows = (n_subplots + 1) // ncol
    mosaic = [["."] * ncol for _ in range(rows)]
    for i, (key, _) in enumerate(subplot_layout):
        row, col = divmod(i, ncol)
        mosaic[row][col] = key

    heatmap_kwargs = heatmap_kwargs or {}

    for dim in dimensions:
        # Compute figure size
        width = figsize[0] if figsize[0] is not None else 2 * len(mosaic[0])
        height = figsize[1] if figsize[1] is not None else 2 * len(mosaic)
        fig, axes = plt.subplot_mosaic(mosaic, figsize=(width, height))
        p_value = p_values[dimensions.index(dim)].to_numpy()

        for key, title in subplot_layout:
            if key not in group_ranges:
                continue
            indices, labels = group_ranges[key]
            data = extract_subset(p_value, indices)

            ax = axes[key]
            mask = data < 0.05
            sns.heatmap(
                data, ax=ax, xticklabels=labels, yticklabels=labels, annot=True, fmt=".2f",
                cbar=False, mask=mask, vmin=0, vmax=1, **heatmap_kwargs
            )
            sns.heatmap(
                data, ax=ax, xticklabels=labels, yticklabels=labels, annot=True, fmt=".2f",
                cbar=False, cmap=cmap, mask=~mask, **heatmap_kwargs
            )
            ax.set_title(title)
            ax.tick_params(axis='x', rotation=0)
            ax.tick_params(axis='y', rotation=0)

        plt.suptitle(f"p-Values for {name.replace('_', ' ')} $H_{dim}$ Across Groups")
        plt.tight_layout()

        if save_plot and output_directory:
            output_directory = Path(output_directory)
            plot_dir = output_directory / "grouped_heatmap_plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            valid_formats = ['pdf', 'png', 'svg', 'jpg']
            if save_format.lower() not in valid_formats:
                logger.error("plot_grouped_p_value_heatmaps: invalid save_format=%s", save_format)
                raise ValueError(f"save_format must be one of {valid_formats}")
            save_path = plot_dir / f"{name}_pvalue_heatmap_dim_{dim}.{save_format.lower()}"
            plt.savefig(save_path, format=save_format.lower(), bbox_inches='tight')
            logger.info("plot_grouped_p_value_heatmaps: saved plot to %s", save_path)

        if show_plot:
            plt.show()
        plt.close("all")
    logger.info("plot_grouped_p_value_heatmaps: done")


def plot_grouped_distance_heatmaps(
    distances: npt.NDArray,
    group_ranges: Dict[str, Tuple[List[int], List[str]]],
    name: str,
    ncol: int = 2,
    dimensions: Optional[List[int]] = None,
    output_directory: Union[str, Path] = "./",
    show_plot: bool = True,
    save_plot: bool = False,
    save_format: str = "pdf",
    subplot_layout: Optional[List[Tuple[str, str]]] = None,
    block_size: int = 40,
    figsize: Tuple[float, float] = (None, None),
    heatmap_kwargs: Optional[Dict] = None
) -> None:
    """
    Visualize grouped subsets of distance matrices as heatmaps.

    Can display per-dimension distances or the average across all dimensions,
    overlaying block averages for clarity.

    Parameters
    ----------
    distances : np.ndarray
        Array of shape (n_dims, n, n) with pairwise distances.
    group_ranges : Dict[str, Tuple[List[int], List[str]]]
        Mapping group_name → (indices, labels). Indices are block numbers.
    name : str
        Descriptor for output.
    ncol : int, default=2
        Number of subplot columns.
    dimensions : List[int], optional
        Dimensions to plot. Use -1 for the mean.
    output_directory : str or Path
        Save directory.
    show_plot : bool, default=True
        Show interactively.
    save_plot : bool, default=False
        Save to disk.
    save_format : str, default="pdf"
        File format.
    subplot_layout : list, optional
        Custom mosaic layout.
    block_size : int, default=40
        Number of samples per group.
    figsize : tuple, optional
        Width × height.
    heatmap_kwargs : dict, optional
        Extra seaborn.heatmap arguments.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If group_ranges empty, distances empty, or invalid dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> dist = np.random.rand(2, 80, 80)  # 2 dims, 80x80
    >>> groups = {"X": ([0,1], ["A","B"])}
    >>> plot_grouped_distance_heatmaps(dist, groups, "test", block_size=40, show_plot=False)
    """
    logger.info(
        "plot_grouped_distance_heatmaps: start | name=%s | dims=%s | save=%s",
        name, dimensions, save_plot
    )
    if not group_ranges:
        logger.error("plot_grouped_distance_heatmaps: empty group_ranges")
        raise ValueError("group_ranges dictionary cannot be empty")
    if not distances.size:
        logger.error("plot_grouped_distance_heatmaps: empty distances")
        raise ValueError("distances array cannot be empty")
    if block_size <= 0:
        logger.error("plot_grouped_distance_heatmaps: non-positive block_size")
        raise ValueError("block_size must be positive")

    # Copy group_ranges to avoid modifying the input
    group_ranges = copy.deepcopy(group_ranges)

    # Infer dimensions if not provided
    if dimensions is None:
        dimensions = [-1] + list(range(distances.shape[0]))

    # Validate dimensions
    max_dim = distances.shape[0] - 1
    invalid_dims = [d for d in dimensions if d != -1 and d > max_dim]
    if invalid_dims:
        logger.error("plot_grouped_distance_heatmaps: invalid dims %s > %d", invalid_dims, max_dim)
        raise ValueError(f"Specified dimensions {invalid_dims} exceed number of distance matrices {max_dim}")

    for label, (indices, groups) in group_ranges.items():
        for inx in range(len(indices)):
            indices[inx] = np.arange(indices[inx] * block_size, indices[inx] * block_size + block_size)
        indices = np.concatenate(indices).astype(int)
        group_ranges[label] = (indices, groups)

    extract_subset = lambda dist, indices: dist[np.ix_(indices, indices)]

    # Default subplot layout if none provided
    if subplot_layout is None:
        subplot_layout = [(key, key) for key in group_ranges.keys()]

    # Create mosaic layout dynamically
    n_subplots = len(subplot_layout)
    rows = (n_subplots + 1) // ncol
    mosaic = [["."] * ncol for _ in range(rows)]
    for i, (key, _) in enumerate(subplot_layout):
        row, col = divmod(i, ncol)
        mosaic[row][col] = key

    heatmap_kwargs = heatmap_kwargs or {"cmap": "coolwarm"}

    for dim in dimensions:
        # Compute figure size
        width = figsize[0] if figsize[0] is not None else 3 * len(mosaic[0])
        height = figsize[1] if figsize[1] is not None else 3 * len(mosaic)
        fig, axes = plt.subplot_mosaic(mosaic, figsize=(width, height))
        distance = distances.mean(axis=0) if dim == -1 else distances[dim]

        for key, title in subplot_layout:
            if key not in group_ranges:
                continue
            indices, labels = group_ranges[key]
            data = extract_subset(distance, indices)
            ax = axes[key]
            group_indices = [
                f"{label}" if i % block_size == block_size // 2 else ""
                for i, label in enumerate(np.repeat(labels, block_size))
            ]

            sns.heatmap(
                data, xticklabels=group_indices, yticklabels=group_indices,
                cbar=False, ax=ax, **heatmap_kwargs
            )
            ax.set_title(title)
            ax.tick_params(left=False, bottom=False)
            ax.tick_params(axis='x', rotation=0)
            ax.tick_params(axis='y', rotation=0)

            num_blocks = data.shape[0] // block_size
            if num_blocks > 0:
                block_means = data.reshape(num_blocks, block_size, num_blocks, block_size).mean(axis=(1, 3))
                for i in range(num_blocks):
                    for j in range(num_blocks):
                        ax.text(
                            j * block_size + block_size // 2, i * block_size + block_size // 2,
                            f"{block_means[i, j]:.2f}", va="center", fontsize=block_size // 3,
                            color="black", fontweight="bold", horizontalalignment='center',
                            path_effects=[path_effects.Stroke(linewidth=1, foreground='white'),
                                          path_effects.Normal()]
                        )
                for i in range(block_size, data.shape[0], block_size):
                    ax.axhline(i, color='black', linestyle='--', linewidth=1)
                    ax.axvline(i, color='black', linestyle='--', linewidth=1)

        title = (f"Mean {name.replace('_', ' ')} Distance" if dim == -1 else
                 f"{name.replace('_', ' ')} Distance $H_{dim}$")

        plt.suptitle(f"{title} Across Groups (Mean Values Overlaid)")
        plt.tight_layout()

        if save_plot and output_directory:
            output_directory = Path(output_directory)
            plot_dir = output_directory / "grouped_heatmap_plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            valid_formats = ['pdf', 'png', 'svg', 'jpg']
            if save_format.lower() not in valid_formats:
                logger.error("plot_grouped_distance_heatmaps: invalid save_format=%s", save_format)
                raise ValueError(f"save_format must be one of {valid_formats}")
            save_path = plot_dir / f"{name}_distance_heatmap_dim_{dim}.{save_format.lower()}"
            plt.savefig(save_path, format=save_format.lower(), bbox_inches='tight')
            logger.info("plot_grouped_distance_heatmaps: saved plot to %s", save_path)

        if show_plot:
            plt.show()
        plt.close("all")
    logger.info("plot_grouped_distance_heatmaps: done")


def plot_swarm_violin(
    data: Dict[str, Dict[str, npt.NDArray]],
    feature_name: str,
    dimensions: Optional[List[int]] = None,
    labels: Optional[List[str]] = None,
    label_styles: Optional[Dict[str, Tuple[str, str]]] = None,
    output_directory: Union[str, Path] = "./",
    show_plot: bool = True,
    save_plot: bool = False,
    save_format: str = "pdf",
    figsize: Tuple[float, float] = (None, None),
    swarm_plot_kwargs: Optional[Dict] = None,
    violin_plot_kwargs: Optional[Dict] = None
) -> None:
    """
    Plot swarm + violin plots for TDA features.

    Displays sample-level feature distributions across groups using violin plots
    and overlaid swarm plots, grouped by homology dimension.

    Parameters
    ----------
    data : Dict[str, Dict[str, np.ndarray]]
        Grouped features. Each group must contain `feature_name` with shape
        (n_samples, n_dims).
    feature_name : str
        Feature to visualize.
    dimensions : List[int], optional
        Dimensions to plot. Inferred if None.
    labels : List[str], optional
        Subset of groups. Defaults to all.
    label_styles : Dict[str, Tuple[str, str]], optional
        Colors/styles. Defaults to seaborn pastel.
    output_directory : str or Path
        Save directory.
    show_plot : bool, default=True
        Show interactively.
    save_plot : bool, default=False
        Save to disk.
    save_format : str, default="pdf"
        File format.
    figsize : tuple, optional
        Width × height in inches.
    swarm_plot_kwargs, violin_plot_kwargs : dict, optional
        Extra plotting arguments.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If feature missing, dimensions invalid, or save_format unsupported.

    Examples
    --------
    >>> import numpy as np
    >>> feat = np.random.rand(5, 2)
    >>> data = {"A": {"persistence_entropy": feat}, "B": {"persistence_entropy": feat}}
    >>> plot_swarm_violin(data, "persistence_entropy", dimensions=[0], show_plot=False)
    """
    logger.info(
        "plot_swarm_violin: start | feature=%s | groups=%s | dims=%s | save=%s",
        feature_name, None if labels is None else labels, dimensions, save_plot
    )
    labels = labels if labels is not None else list(data.keys())
    data = {k: data[k] for k in labels if k in data}

    # Validate feature_name
    if not all(feature_name in data[label] for label in labels):
        logger.error("plot_swarm_violin: feature '%s' missing in some groups", feature_name)
        raise ValueError(f"Feature '{feature_name}' not found in all group data")

    # Infer dimensions if not provided
    if dimensions is None:
        dimensions = _infer_dimensions(data)

    # Validate dimensions
    available_dims = _infer_dimensions(data)
    invalid_dims = [d for d in dimensions if d not in available_dims]
    if invalid_dims:
        logger.error("plot_swarm_violin: invalid dims %s | available=%s", invalid_dims, available_dims)
        raise ValueError(f"Specified dimensions {invalid_dims} not found in data. Available: {available_dims}")

    palette = ([c for c, _ in label_styles.values()] if label_styles else "pastel")
    width = figsize[0] if figsize[0] is not None else len(data) * 1.2
    height = figsize[1] if figsize[1] is not None else 3 * len(dimensions)
    fig, axs = plt.subplots(len(dimensions), 1, figsize=(width, height))
    axs = [axs] if len(dimensions) == 1 else axs

    swarm_plot_kwargs = swarm_plot_kwargs or {"edgecolor": "black", "size": 5, "linewidth": 1, "alpha": 0.9}
    violin_plot_kwargs = violin_plot_kwargs or {}

    for i, dim in enumerate(dimensions):
        data_to_plot = [np.array(data[label][feature_name])[:, dim] for label in data]
        pretty_labels = [label.replace("_", " ") for label in labels]

        sns.violinplot(data=data_to_plot, ax=axs[i], inner=None, palette=palette, **violin_plot_kwargs)
        sns.swarmplot(data=data_to_plot, ax=axs[i],  palette=palette, **swarm_plot_kwargs)

        axs[i].grid(False)
        axs[i].set_xticks(np.arange(len(pretty_labels)))
        axs[i].set_xticklabels(pretty_labels)
        axs[i].set_title(fr"$H_{dim}$")

    plt.suptitle(f"Swarm and Violin Plots of {feature_name.replace('_', ' ').title()}")
    plt.tight_layout()

    if save_plot and output_directory:
        output_directory = Path(output_directory)
        plot_dir = output_directory / "swarm_violin_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        valid_formats = ['pdf', 'png', 'svg', 'jpg']
        if save_format.lower() not in valid_formats:
            logger.error("plot_swarm_violin: invalid save_format=%s", save_format)
            raise ValueError(f"save_format must be one of {valid_formats}")
        save_path = plot_dir / f"{feature_name}_swarm_violin.{save_format.lower()}"
        plt.savefig(save_path, format=save_format.lower(), bbox_inches='tight', **violin_plot_kwargs)
        logger.info("plot_swarm_violin: saved plot to %s", save_path)

    if show_plot:
        plt.show()
    plt.close("all")
    logger.info("plot_swarm_violin: done")


def plot_kde_dist(
    data: Dict[str, Dict[str, npt.NDArray]],
    feature_name: str,
    labels: Optional[List[str]] = None,
    label_styles: Optional[Dict[str, Tuple[str, str]]] = None,
    output_directory: Union[str, Path] = "./",
    show_plot: bool = True,
    save_plot: bool = False,
    save_format: str = "pdf",
    figsize: Tuple[float, float] = (8, 4),
    kde_kwargs: Optional[Dict] = None
) -> None:
    """
    Plot kernel density estimates (KDE) of a TDA feature distribution across groups.

    Uses seaborn.kdeplot to display estimated densities.

    Parameters
    ----------
    data : Dict[str, Dict[str, np.ndarray]]
        Grouped features. Each group must contain `feature_name`.
    feature_name : str
        Feature to visualize.
    labels : List[str], optional
        Subset of groups. Defaults to all.
    label_styles : Dict[str, Tuple[str, str]], optional
        Mapping group → (color, linestyle). Generated if None.
    output_directory : str or Path
        Directory for saving plots.
    show_plot : bool, default=True
        Show interactively.
    save_plot : bool, default=False
        Save to disk.
    save_format : str, default="pdf"
        File format.
    figsize : tuple, default=(8, 4)
        Width × height.
    kde_kwargs : dict, optional
        Extra arguments for seaborn.kdeplot.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If feature missing, non-numeric, or save_format unsupported.

    Examples
    --------
    >>> import numpy as np
    >>> feat = np.random.rand(5, 2, 2)  # shape (n_samples, n_dims, n_bins)
    >>> data = {"A": {"dummy": feat}, "B": {"dummy": feat}}
    >>> plot_kde_dist(data, "dummy", show_plot=False)
    """

    logger.info(
        "plot_kde_dist: start | feature=%s | groups=%s | save=%s",
        feature_name, None if labels is None else labels, save_plot
    )
    labels = labels if labels is not None else list(data.keys())
    data = {k: data[k] for k in labels if k in data}

    # Validate feature_name
    if not all(feature_name in data[label] for label in labels):
        logger.error("plot_kde_dist: feature '%s' missing in some groups", feature_name)
        raise ValueError(f"Feature '{feature_name}' not found in all group data")

    if not label_styles:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        linestyles = ['-', '--', '-.', ':']
        label_styles = {label: (colors[i % len(colors)], linestyles[i % len(linestyles)])
                        for i, label in enumerate(data.keys())}

    fig, ax = plt.subplots(figsize=figsize)
    kde_kwargs = kde_kwargs or {}

    for label, (color, linestyle) in label_styles.items():
        distances = data[label][feature_name]
        try:
            distances = distances.mean(axis=0).flatten()
            if not np.issubdtype(distances.dtype, np.number):
                raise ValueError(f"Feature '{feature_name}' for {label} must be numeric")
        except (AttributeError, ValueError) as e:
            logger.exception("plot_kde_dist: cannot process '%s' for %s", feature_name, label)
            raise ValueError(f"Cannot process '{feature_name}' for {label}: {str(e)}")

        sns.kdeplot(distances, ax=ax, color=color, linestyle=linestyle,
                    label=label.replace("_", " "), **kde_kwargs)

    ax.set_title(f"KDE Plot of {feature_name.replace('_', ' ').title()}")
    ax.set_xlabel(feature_name.replace('_', ' '))
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_touch = hasattr(plt, "tight_layout")  # defensive
    plt.tight_layout()

    if save_plot and output_directory:
        output_directory = Path(output_directory)
        plot_dir = output_directory / "kde_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        valid_formats = ['pdf', 'png', 'svg', 'jpg']
        if save_format.lower() not in valid_formats:
            logger.error("plot_kde_dist: invalid save_format=%s", save_format)
            raise ValueError(f"save_format must be one of {valid_formats}")
        save_path = plot_dir / f"{feature_name}_kde.{save_format.lower()}"
        plt.savefig(save_path, format=save_format.lower(), bbox_inches='tight')
        logger.info("plot_kde_dist: saved plot to %s", save_path)

    if show_plot:
        plt.show()
    plt.close("all")
    logger.info("plot_kde_dist: done")


def _infer_dimensions_from_betti_stats(data: Dict[str, Dict[str, npt.NDArray]], labels: List[str]) -> List[int]:
    """Find homology dimensions present across all groups when using betti-stat tables."""
    common = None
    for g in labels:
        dims_g = set()
        for k in data[g].keys():
            if isinstance(k, str) and k.startswith("H"):
                try:
                    dims_g.add(int(k[1:]))
                except Exception:
                    pass
        common = dims_g if common is None else (common & dims_g)
    return sorted(common or [])


def plot_betti_stats_pvalues(
    data: Dict[str, Dict[str, npt.NDArray]],
    feature_name: str,
    labels: Optional[List[str]] = None,
    dimensions: Optional[List[int]] = None,
    output_directory: Union[str, Path] = "./",
    test: str = "auto",
    show_plot: bool = True,
    save_plot: bool = False,
    save_format: str = "pdf",
    figsize: Tuple[float, float] = (None, None),
    heatmap_kwargs: Optional[Dict] = None
) -> List[pd.DataFrame]:
    """
    Compute and visualize p-values comparing TDA features between groups.

    If `data[group]` contains keys "feature_names" and "H{d}" (the output of
    compute_betti_stat_features), this compares the specified *Betti-stat feature*
    across groups for each homology dimension. Valid feature_name examples:
      "auc_trapz", "centroid_x", "peak_y", "std_y", "skewness_y", "kurtosis_excess_y".

    Otherwise, falls back to comparing raw features as in your original function.
    """
    logger.info(
        "plot_betti_stats_pvalues: start | feature=%s | test=%s | save=%s | groups=%s",
        feature_name, test, save_plot, None if labels is None else labels
    )

    # Select labels
    labels = labels if labels is not None else list(data.keys())
    data = {k: data[k] for k in labels if k in data}
    if len(data) < 2:
        raise ValueError("Need at least two groups for p-value comparison")

    # Detect "betti-stats mode"
    is_betti_stats_mode = all(
        isinstance(data[g], dict) and "feature_names" in data[g] for g in labels
    )

    if is_betti_stats_mode:
        # ----- Betti-stats path: use tables from compute_betti_stat_features -----
        # Validate feature exists everywhere, get column index per group
        feat_idx = {}
        for g in labels:
            fns = [str(x) for x in data[g]["feature_names"].tolist()]
            if feature_name not in fns:
                logger.error("plot_p_values: feature '%s' missing in group %s", feature_name, g)
                raise ValueError(f"Feature '{feature_name}' not found in group '{g}'. Available: {fns}")
            feat_idx[g] = fns.index(feature_name)

        # Dimensions
        if dimensions is None:
            dimensions = _infer_dimensions_from_betti_stats(data, labels)
        # Validate dims exist across groups
        missing_dims = [d for d in (dimensions or []) if not all(f"H{d}" in data[g] for g in labels)]
        if missing_dims:
            raise ValueError(f"Dimensions {missing_dims} not found across all groups")

        # Prepare output containers
        p_values = [pd.DataFrame(np.ones((len(labels), len(labels))), index=labels, columns=labels)
                    for _ in dimensions]
        cmap_sig = ListedColormap(["#3b4cc0"])  # significant overlay color

        # Pairwise tests
        for i, g1 in enumerate(labels):
            for j, g2 in enumerate(labels[i+1:], i+1):
                for dim_idx, d in enumerate(dimensions):
                    a = np.asarray(data[g1][f"H{d}"])[:, feat_idx[g1]].astype(float)
                    b = np.asarray(data[g2][f"H{d}"])[:, feat_idx[g2]].astype(float)
                    # Drop NaNs
                    a = a[~np.isnan(a)]
                    b = b[~np.isnan(b)]
                    if a.size == 0 or b.size == 0:
                        raise ValueError(f"Empty values for {feature_name} in H{d} for groups {g1} or {g2}")

                    # Choose test
                    test_to_use = test
                    if test == "auto":
                        pnorm_a = shapiro(a)[1] if a.size >= 3 else 0.0
                        pnorm_b = shapiro(b)[1] if b.size >= 3 else 0.0
                        test_to_use = "t_test" if (pnorm_a > 0.05 and pnorm_b > 0.05) else "wilcoxon"

                    if test_to_use == "t_test":
                        # Welch by default when variances unequal
                        plevene = levene(a, b, center="median")[1] if (a.size >= 2 and b.size >= 2) else 0.0
                        p_val = ttest_ind(a, b, equal_var=plevene > 0.05)[1]
                    elif test_to_use == "wilcoxon":
                        if a.size == b.size and a.size > 0:
                            p_val = wilcoxon(a, b)[1]
                        else:
                            logger.warning(
                                    "plot_p_values: wilcoxon requested but sizes differ; using Mann–Whitney"
                                )
                            p_val = mannwhitneyu(a, b, alternative="two-sided")[1]
                    else:
                        raise ValueError("Test must be 't_test', 'wilcoxon', or 'auto'")

                    p_values[dim_idx].loc[g1, g2] = p_values[dim_idx].loc[g2, g1] = float(p_val)

        # --- Plotting ---
        width = figsize[0] if figsize[0] is not None else len(labels) * 1.2 * max(1, len(dimensions))
        height = figsize[1] if figsize[1] is not None else len(labels) * 1.2
        fig, axs = plt.subplots(1, len(dimensions), figsize=(width, height))
        axs = [axs] if len(dimensions) == 1 else axs
        heatmap_kwargs = heatmap_kwargs or {}

        for ax, dim_idx in zip(axs, range(len(dimensions))):
            mat = p_values[dim_idx]
            mask_sig = mat.values < 0.05
            pretty_labels = [l.replace("_", " ") for l in labels]
            # base (non-significant) layer
            sns.heatmap(mat, ax=ax, xticklabels=pretty_labels, yticklabels=pretty_labels,
                        annot=True, fmt=".2f", cbar=False, mask=mask_sig, vmin=0, vmax=1, **heatmap_kwargs)
            # significant overlay
            sns.heatmap(mat, ax=ax, xticklabels=pretty_labels, yticklabels=pretty_labels,
                        annot=True, fmt=".2f", cbar=False, cmap=cmap_sig, mask=~mask_sig, vmin=0, vmax=1, **heatmap_kwargs)
            ax.set_title(fr"$H_{dimensions[dim_idx]}$", fontsize=18)

        title = (f"p-Values from {'T-Test or Wilcoxon' if test == 'auto' else test} "
                 f"for {feature_name.replace('_', ' ')}")
        plt.suptitle(title)
        plt.tight_layout()

        if save_plot and output_directory:
            output_directory = Path(output_directory)
            plot_dir = output_directory / "p_value_plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            valid_formats = ['pdf', 'png', 'svg', 'jpg']
            if save_format.lower() not in valid_formats:
                logger.error("plot_p_values: invalid save_format=%s", save_format)
                raise ValueError(f"save_format must be one of {valid_formats}")
            save_path = plot_dir / f"{test}_p_values_for_{feature_name}.{save_format.lower()}"
            plt.savefig(save_path, format=save_format.lower(), bbox_inches='tight')
            logger.info("plot_p_values: saved plot to %s", save_path)

        if show_plot:
            plt.show()
        plt.close("all")
        logger.info("plot_p_values: done (betti-stats mode)")
        return p_values

    
