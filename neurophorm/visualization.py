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
from scipy.stats import t as t_dist
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import ListedColormap, to_hex
import seaborn as sns
import warnings
import copy
import math

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

# ---------------------------
# helpers
# ---------------------------

def _colorblind_palette(n: int) -> List[str]:
    """
    Return a colorblind-safe categorical palette with n colors.
    Tries seaborn 'colorblind', falls back to Matplotlib 'tab10'.
    """
    try:
        import seaborn as sns
        pal = sns.color_palette("colorblind", n)
        try:
            return pal.as_hex()
        except AttributeError:
            return [sns.utils.rgb2hex(c) for c in pal]
    except Exception:
        cmap = plt.get_cmap("tab10")
        base = list(getattr(cmap, "colors", [])) or [cmap(i) for i in range(cmap.N)]
        from matplotlib.colors import to_hex
        return [to_hex(c) for c in (base * ((n + len(base) - 1)//len(base)))[:n]]

def _solid_cmap(color: str) -> ListedColormap:
    """One-color ListedColormap used to render significant cells as a solid patch."""
    return ListedColormap([color])

def _validate_save_format(fmt: str) -> None:
    valid = {"pdf","png","svg","jpg"}
    if fmt.lower() not in valid:
        raise ValueError(f"save_format must be one of {sorted(valid)}")

def _upper_tri_indices(n: int) -> List[Tuple[int,int]]:
    """Indices of the upper triangle (i<j) for an n×n symmetric matrix."""
    return [(i, j) for i in range(n) for j in range(i+1, n)]

def _apply_multitest_inplace(mat: pd.DataFrame, alpha: float, method: Optional[str]) -> None:
    """
    Multiple-comparisons correction in-place on the upper triangle of a symmetric matrix.
    method in {"fdr_bh", "bonferroni", None}.
    """
    if method is None:
        return
    try:
        from statsmodels.stats.multitest import multipletests
    except Exception:
        logger.warning("statsmodels not available; skipping multiple-comparison correction.")
        return
    idxs = _upper_tri_indices(len(mat))
    p = np.array([mat.iat[i, j] for i, j in idxs], dtype=float)
    if p.size == 0:
        return
    meth = "fdr_bh" if method == "fdr_bh" else "bonferroni"
    _, p_adj, _, _ = multipletests(p, alpha=alpha, method=meth)
    k = 0
    for i, j in idxs:
        mat.iat[i, j] = mat.iat[j, i] = float(p_adj[k])
        k += 1

def _choose_test(a: np.ndarray, b: np.ndarray, test: str) -> Tuple[str, float]:
    """
    Decide which statistical test to use and return (name, p_value).
    test in {"auto","t_test","mannwhitney","wilcoxon"}.
    'auto' -> Welch t-test if both look normal (Shapiro p>0.05 for n>=3), else Mann–Whitney.
    'wilcoxon' only used when len(a)==len(b); otherwise falls back to Mann–Whitney.
    """
    test = test.lower()
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return ("empty", 1.0)

    if test == "auto":
        pa = shapiro(a)[1] if a.size >= 3 else 0.0
        pb = shapiro(b)[1] if b.size >= 3 else 0.0
        test = "t_test" if (pa > 0.05 and pb > 0.05) else "mannwhitney"

    if test == "t_test":
        # Welch if variances differ
        peq = (levene(a, b, center="median")[1] if (a.size >= 2 and b.size >= 2) else 0.0) > 0.05
        return ("t_test", ttest_ind(a, b, equal_var=peq)[1])

    if test == "wilcoxon":
        if a.size == b.size and a.size > 0:
            return ("wilcoxon", wilcoxon(a, b)[1])
        logger.warning("wilcoxon requested but sizes differ; using Mann–Whitney")
        return ("mannwhitney", mannwhitneyu(a, b, alternative="two-sided")[1])

    if test == "mannwhitney":
        return ("mannwhitney", mannwhitneyu(a, b, alternative="two-sided")[1])

    raise ValueError("test must be one of {'auto','t_test','mannwhitney','wilcoxon'}")

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

def _welch_from_summary(m1: float, s1: float, n1: int,
                        m2: float, s2: float, n2: int) -> float:
    """
    Two-sided Welch's t-test p-value computed from summary stats.
      s? are sample standard deviations (not SE).
    """
    if n1 < 2 or n2 < 2:
        return 1.0
    denom = np.sqrt((s1**2)/n1 + (s2**2)/n2)
    if denom == 0:
        return 1.0
    t = (m1 - m2) / denom
    # Welch–Satterthwaite degrees of freedom
    v1, v2 = (s1**2)/n1, (s2**2)/n2
    df = (v1 + v2)**2 / ((v1**2)/(n1 - 1) + (v2**2)/(n2 - 1))
    p = 2.0 * t_dist.sf(np.abs(t), df)
    return float(p)

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
        colors = _colorblind_palette(len(labels))
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
        _validate_save_format(save_format)
        save_path = plot_dir / f"mean_betti_curves.{save_format.lower()}"
        plt.savefig(save_path, format=save_format.lower(), bbox_inches='tight')
        logger.info("plot_betti_curves: saved plot to %s", save_path)

    if show_plot:
        plt.show()
    plt.close("all")
    logger.info("plot_betti_curves: done")


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
            _validate_save_format(save_format)
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

    palette = _colorblind_palette(len(labels))
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
        _validate_save_format(save_format)
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
        colors = _colorblind_palette(len(labels))
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
        _validate_save_format(save_format)
        save_path = plot_dir / f"{feature_name}_kde.{save_format.lower()}"
        plt.savefig(save_path, format=save_format.lower(), bbox_inches='tight')
        logger.info("plot_kde_dist: saved plot to %s", save_path)

    if show_plot:
        plt.show()
    plt.close("all")
    logger.info("plot_kde_dist: done")


def plot_node_removal(
    mean_df: pd.DataFrame,
    error_df: pd.DataFrame,
    *,
    title: str = "Mean Distance of Removed Subnetworks by Groups/Conditions",
    x_label: str = "Removed Subnetwork",
    y_label: str = "Distance to the original network",
    group_order: Optional[List] = None,
    col_order: Optional[List[str]] = None,
    bar_colors: Optional[List[str]] = None,    # if None -> colorblind palette
    title_colors: Optional[List[str]] = None,  # if None -> colorblind palette
    layout: Optional[Tuple[int, int]] = None,
    figsize: Tuple[float, float] = (7, 10),
    ylim: Optional[Tuple[float, float]] = None,
    width: float = 0.9,
    edgecolor: str = "gray",
    linewidth: float = 0.5,
    capsize: float = 4.0,
    grid: bool = True,
    grid_kwargs: Optional[Dict] = None,
    tick_rotation: int = 0,
    bar_kwargs: Optional[Dict] = None,         
    output_directory: Union[str, Path] = "./", 
    show_plot: bool = True,                    
    save_plot: bool = False,                   
    save_format: str = "pdf"                   
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot node-removal summaries returned by load_removal_data.

    Color policy:
      - If bar_colors/title_colors are None, use a standard colorblind-safe palette.
      - No custom hand-picked colors are introduced in defaults.

    Saving/Showing policy: identical to other plot_* helpers.
    Saves to <output_directory>/node_removal_plots/ if save_plot=True.
    """
    logger.info("plot_node_removal: start | mean=%s error=%s | save=%s(%s) | show=%s",
                mean_df.shape, error_df.shape, save_plot, save_format, show_plot)

    # Align shapes & order
    error_df = error_df.reindex(index=mean_df.index, columns=mean_df.columns)
    if group_order is not None:
        mean_df = mean_df.reindex(group_order)
        error_df = error_df.reindex(group_order)
    if col_order is not None:
        mean_df = mean_df.reindex(columns=col_order)
        error_df = error_df.reindex(columns=col_order)

    groups = mean_df.index.to_list()
    cols   = mean_df.columns.to_list()
    n_groups, n_cols = len(groups), len(cols)
    if n_groups == 0 or n_cols == 0:
        raise ValueError("Empty mean_df: nothing to plot")

    # Colors: use colorblind-safe defaults only if not provided
    if bar_colors is None:
        bar_colors = _colorblind_palette(n_groups)
    else:
        if len(bar_colors) < n_groups:
            raise ValueError("bar_colors length must be >= number of groups")

    if title_colors is None:
        title_colors = _colorblind_palette(n_cols)
    else:
        if len(title_colors) < n_cols:
            raise ValueError("title_colors length must be >= number of columns")

    # Layout (auto grid similar to other helpers)
    if layout is None:
        r = int(math.floor(math.sqrt(n_cols)))
        c = int(math.ceil(n_cols / max(r, 1))) if r else n_cols
        layout = (r, c) if r and r * c >= n_cols else (math.ceil(n_cols / 2), 2)
    nrows, ncols = layout

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True, sharey=True)
    ax_array = np.atleast_1d(axes).ravel()

    x = np.arange(n_groups)
    bar_kwargs = bar_kwargs or {}

    for j, col in enumerate(cols):
        ax = ax_array[j]
        y = mean_df[col].values.astype(float)
        yerr = error_df[col].values.astype(float)

        bars = ax.bar(
            x, y, yerr=yerr, capsize=capsize,
            width=width, edgecolor=edgecolor, linewidth=linewidth,
            **bar_kwargs
        )
        # Apply bar colors (colorblind default if none provided)
        for i, b in enumerate(bars):
            b.set_facecolor(bar_colors[i])

        if grid:
            gkw = dict(axis="y", linestyle="--", alpha=0.6)
            if grid_kwargs:
                gkw.update(grid_kwargs)
            ax.grid(**gkw)

        ax.set_title(str(col), fontsize=12, color=title_colors[j])
        for spine in ax.spines.values():
            spine.set_color(title_colors[j])

        ax.set_xticks(x)
        ax.set_xticklabels([str(g) for g in groups], rotation=tick_rotation)

        if ylim is not None:
            ax.set_ylim(*ylim)

        # No legend by default
        if ax.get_legend():
            ax.get_legend().set_visible(False)

    # Remove unused axes
    for k in range(n_cols, nrows * ncols):
        fig.delaxes(ax_array[k])

    # Shared labels & layout
    fig.suptitle(title)
    fig.supxlabel(x_label)
    fig.supylabel(y_label)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Save / Show consistent with other helpers
    if save_plot and output_directory:
        output_directory = Path(output_directory)
        plot_dir = output_directory / "node_removal_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        _validate_save_format(save_format)
        # use a stable filename
        save_path = plot_dir / f"node_removal_bars.{save_format.lower()}"
        fig.savefig(save_path, format=save_format.lower(), bbox_inches="tight")
        logger.info("plot_node_removal: saved plot to %s", save_path)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    logger.info("plot_node_removal: done")


def plot_p_values(
    data: Dict[str, Dict[str, npt.NDArray]],
    feature_name: str,
    labels: Optional[List[str]] = None,
    dimensions: Optional[List[int]] = None,
    output_directory: Union[str, Path] = "./",
    test: str = "auto",
    alpha: float = 0.05,
    multitest: Optional[str] = None,  # {"fdr_bh","bonferroni", None}
    show_plot: bool = True,
    save_plot: bool = False,
    save_format: str = "pdf",
    figsize: Tuple[float, float] = (None, None),
    base_cmap: str = "viridis",
    sig_color: str = "#3b4cc0",
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
    logger.info("plot_p_values: start | feature=%s | test=%s", feature_name, test)

    labels = labels if labels is not None else list(data.keys())
    data = {k: data[k] for k in labels if k in data}
    if not all(feature_name in data[g] or feature_name == "betti_curves_shared" for g in labels):
        raise ValueError(f"Feature '{feature_name}' not found in all groups")

    if dimensions is None:
        dimensions = _infer_dimensions(data)

    # Compute quantity to test
    prepared = None
    if feature_name == "betti_curves_shared":
        prepared = _compute_betti_auc(data, labels, dimensions)  # shape per group: (n_samples, n_dims_sel)
    # Build containers
    p_values = [pd.DataFrame(np.ones((len(labels), len(labels))), index=labels, columns=labels)
                for _ in dimensions]

    for i, g1 in enumerate(labels):
        for j, g2 in enumerate(labels[i+1:], i+1):
            arr1 = prepared[g1] if prepared is not None else data[g1][feature_name]
            arr2 = prepared[g2] if prepared is not None else data[g2][feature_name]
            if arr1.shape != arr2.shape and arr1.ndim == arr2.ndim:
                raise ValueError(f"Incompatible shapes for '{feature_name}' between {g1} and {g2}")

            for d_idx, d in enumerate(dimensions):
                x = arr1[:, d, :].mean(axis=0) if arr1.ndim > 2 else arr1[:, d]
                y = arr2[:, d, :].mean(axis=0) if arr2.ndim > 2 else arr2[:, d]
                _, p = _choose_test(x, y, test)
                p_values[d_idx].iat[i, j] = p_values[d_idx].iat[j, i] = float(p)

    # Multiple-comparison correction per dimension
    for mat in p_values:
        _apply_multitest_inplace(mat, alpha=alpha, method=multitest)

    # ---- Plotting ----
    width = figsize[0] if figsize[0] is not None else max(3.0, 1.2 * len(labels) * max(1, len(dimensions)))
    height = figsize[1] if figsize[1] is not None else max(3.0, 1.2 * len(labels))
    fig, axs = plt.subplots(1, len(dimensions), figsize=(width, height))
    axs = [axs] if len(dimensions) == 1 else axs
    heatmap_kwargs = heatmap_kwargs or {}

    for ax, d_idx in zip(axs, range(len(dimensions))):
        mat = p_values[d_idx]
        mask_sig = mat.values < alpha
        pretty = [l.replace("_", " ") for l in labels]
        # non-significant layer
        sns.heatmap(mat, ax=ax, xticklabels=pretty, yticklabels=pretty,
                    annot=True, fmt=".3f", cbar=False, mask=mask_sig, vmin=0, vmax=1,
                    cmap=base_cmap, **heatmap_kwargs)
        # significant overlay (solid)
        sns.heatmap(mat, ax=ax, xticklabels=pretty, yticklabels=pretty,
                    annot=True, fmt=".3f", cbar=False, mask=~mask_sig, vmin=0, vmax=1,
                    cmap=_solid_cmap(sig_color), **heatmap_kwargs)
        ax.set_title(fr"$H_{dimensions[d_idx]}$", fontsize=16)

    nice_name = "Betti Curve AUC" if feature_name == "betti_curves_shared" else feature_name.replace("_", " ")
    plt.suptitle(f"p-values ({test}) for {nice_name}")
    plt.tight_layout()

    if save_plot and output_directory:
        _validate_save_format(save_format)
        out = Path(output_directory) / "p_value_plots"
        out.mkdir(parents=True, exist_ok=True)
        fp = out / f"{test}_p_values_for_{feature_name}.{save_format.lower()}"
        plt.savefig(fp, format=save_format.lower(), bbox_inches="tight")
        logger.info("plot_p_values: saved %s", fp)

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
    figsize: Tuple[float, float] = (None, None),
    alpha: float = 0.05,
    base_cmap: str = "viridis",
    sig_color: str = "#3b4cc0",
    heatmap_kwargs: Optional[Dict] = None,
    subplot_layout: Optional[List[Tuple[str, str]]] = None,
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
    logger.info("plot_grouped_p_value_heatmaps: start")

    if not group_ranges or not p_values:
        raise ValueError("group_ranges and p_values cannot be empty")
    if dimensions is None:
        dimensions = list(range(len(p_values)))
    if max(dimensions) >= len(p_values):
        raise ValueError("Some requested dimensions exceed available p-value matrices")

    # layout keys
    if subplot_layout is None:
        subplot_layout = [(key, key) for key in group_ranges.keys()]

    # compute mosaic
    n_sub = len(subplot_layout)
    rows = (n_sub + ncol - 1) // ncol
    mosaic = [["."] * ncol for _ in range(rows)]
    for i, (key, _) in enumerate(subplot_layout):
        r, c = divmod(i, ncol)
        mosaic[r][c] = key

    heatmap_kwargs = heatmap_kwargs or {}

    for dim in dimensions:
        width = figsize[0] if figsize[0] is not None else 2.0 * len(mosaic[0])
        height = figsize[1] if figsize[1] is not None else 2.0 * len(mosaic)
        fig, axes = plt.subplot_mosaic(mosaic, figsize=(width, height))
        mat = p_values[dimensions.index(dim)].to_numpy()

        for key, title in subplot_layout:
            if key not in group_ranges:
                continue
            idxs, lbls = group_ranges[key]
            sub = mat[np.ix_(idxs, idxs)]
            ax = axes[key]
            mask_sig = sub < alpha
            sns.heatmap(sub, ax=ax, xticklabels=lbls, yticklabels=lbls,
                        annot=True, fmt=".3f", cbar=False, mask=mask_sig, vmin=0, vmax=1,
                        cmap=base_cmap, **heatmap_kwargs)
            sns.heatmap(sub, ax=ax, xticklabels=lbls, yticklabels=lbls,
                        annot=True, fmt=".3f", cbar=False, mask=~mask_sig, vmin=0, vmax=1,
                        cmap=_solid_cmap(sig_color), **heatmap_kwargs)
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=0)
            ax.tick_params(axis="y", rotation=0)

        plt.suptitle(f"p-values for {name.replace('_',' ')}  $H_{dim}$")
        plt.tight_layout()

        if save_plot and output_directory:
            _validate_save_format(save_format)
            out = Path(output_directory) / "grouped_heatmap_plots"
            out.mkdir(parents=True, exist_ok=True)
            fp = out / f"{name}_pvalue_heatmap_dim_{dim}.{save_format.lower()}"
            plt.savefig(fp, format=save_format.lower(), bbox_inches="tight")
            logger.info("plot_grouped_p_value_heatmaps: saved %s", fp)

        if show_plot:
            plt.show()
        plt.close("all")
    logger.info("plot_grouped_p_value_heatmaps: done")


def plot_betti_stats_pvalues(
    data: Dict[str, Dict[str, npt.NDArray]],
    feature_name: str,
    labels: Optional[List[str]] = None,
    dimensions: Optional[List[int]] = None,
    output_directory: Union[str, Path] = "./",
    test: str = "auto",
    alpha: float = 0.05,
    multitest: Optional[str] = None,
    show_plot: bool = True,
    save_plot: bool = False,
    save_format: str = "pdf",
    figsize: Tuple[float, float] = (None, None),
    base_cmap: str = "viridis",
    sig_color: str = "#3b4cc0",
    heatmap_kwargs: Optional[Dict] = None
) -> List[pd.DataFrame]:
    """
    Compare a chosen Betti-stat feature across groups and plot p-value heatmaps.

    Expects each group mapping to contain:
      - 'feature_names'  list-like of column names for each H{d} table
      - 'H{d}'           2D array (n_samples × n_features) for each dimension

    feature_name examples
      'auc_trapz','centroid_x','peak_y','std_y','skewness_y','kurtosis_excess_y'
    """
    logger.info("plot_betti_stats_pvalues: start | feature=%s", feature_name)

    labels = labels if labels is not None else list(data.keys())
    data = {k: data[k] for k in labels if k in data}
    if len(data) < 2:
        raise ValueError("Need at least two groups for comparison")

    # verify feature exists and get per-group column index
    feat_idx = {}
    for g in labels:
        fns = [str(x) for x in data[g]["feature_names"]]
        if feature_name not in fns:
            raise ValueError(f"Feature '{feature_name}' not in group '{g}'. Available: {fns}")
        feat_idx[g] = fns.index(feature_name)

    if dimensions is None:
        dimensions = _infer_dimensions_from_betti_stats(data, labels)

    p_values = [pd.DataFrame(np.ones((len(labels), len(labels))), index=labels, columns=labels)
                for _ in dimensions]

    for i, g1 in enumerate(labels):
        for j, g2 in enumerate(labels[i+1:], i+1):
            for d_idx, d in enumerate(dimensions):
                a = np.asarray(data[g1][f"H{d}"])[:, feat_idx[g1]].astype(float)
                b = np.asarray(data[g2][f"H{d}"])[:, feat_idx[g2]].astype(float)
                _, p = _choose_test(a, b, test)
                p_values[d_idx].iat[i, j] = p_values[d_idx].iat[j, i] = float(p)

    for mat in p_values:
        _apply_multitest_inplace(mat, alpha=alpha, method=multitest)

    width = figsize[0] if figsize[0] is not None else max(3.0, 1.2 * len(labels) * max(1, len(dimensions)))
    height = figsize[1] if figsize[1] is not None else max(3.0, 1.2 * len(labels))
    fig, axs = plt.subplots(1, len(dimensions), figsize=(width, height))
    axs = [axs] if len(dimensions) == 1 else axs
    heatmap_kwargs = heatmap_kwargs or {}

    pretty = [l.replace("_", " ") for l in labels]
    for ax, d_idx in zip(axs, range(len(dimensions))):
        mat = p_values[d_idx]
        mask_sig = mat.values < alpha
        sns.heatmap(mat, ax=ax, xticklabels=pretty, yticklabels=pretty,
                    annot=True, fmt=".3f", cbar=False, mask=mask_sig, vmin=0, vmax=1,
                    cmap=base_cmap, **heatmap_kwargs)
        sns.heatmap(mat, ax=ax, xticklabels=pretty, yticklabels=pretty,
                    annot=True, fmt=".3f", cbar=False, mask=~mask_sig, vmin=0, vmax=1,
                    cmap=_solid_cmap(sig_color), **heatmap_kwargs)
        ax.set_title(fr"$H_{dimensions[d_idx]}$", fontsize=16)

    plt.suptitle(f"p-values ({test}) for Betti-stats • {feature_name}")
    plt.tight_layout()

    if save_plot and output_directory:
        _validate_save_format(save_format)
        out = Path(output_directory) / "p_value_plots"
        out.mkdir(parents=True, exist_ok=True)
        fp = out / f"{test}_p_values_for_betti_stats_{feature_name}.{save_format.lower()}"
        plt.savefig(fp, format=save_format.lower(), bbox_inches="tight")
        logger.info("plot_betti_stats_pvalues: saved %s", fp)

    if show_plot:
        plt.show()
    plt.close("all")
    logger.info("plot_betti_stats_pvalues: done")
    return p_values


def plot_node_removal_p_values(
    mean_df: pd.DataFrame,
    error_df: pd.DataFrame,
    atlas: np.ndarray,
    *,
    labels: Optional[List[str]] = None,
    group_order: Optional[List] = None,
    alpha: float = 0.05,
    multitest: Optional[str] = "fdr_bh",
    title: str = "p-values across conditions per removed subnetwork",
    title_colors: Optional[List[str]] = None,
    layout: Optional[Tuple[int, int]] = None,
    figsize: Tuple[float, float] = (None, None),
    base_cmap: str = "viridis",
    sig_color: str = "#3b4cc0",
    heatmap_kwargs: Optional[Dict] = None,
    output_directory: Union[str, Path] = "./",
    show_plot: bool = True,
    save_plot: bool = False,
    save_format: str = "pdf",
) -> List[pd.DataFrame]:
    """
    Rebuild pairwise p-value matrices between condition columns for each atlas group
    using Welch's t-test from summary statistics, then plot heatmaps.

    mean_df  mean per atlas label × condition
    error_df standard error per atlas label × condition
    atlas    vector of node → atlas label (provides sample counts per group)
    """
    logger.info("plot_node_removal_p_values: start")

    # Align and subset
    error_df = error_df.reindex(index=mean_df.index, columns=mean_df.columns)
    if labels is not None:
        mean_df = mean_df.reindex(columns=labels)
        error_df = error_df.reindex(columns=labels)

    cols = list(mean_df.columns)
    if len(cols) < 2:
        raise ValueError("Need at least two conditions to compute p-values")

    if group_order is not None:
        mean_df = mean_df.reindex(group_order)
        error_df = error_df.reindex(group_order)

    groups = list(mean_df.index)
    if not groups:
        raise ValueError("No atlas groups to plot")

    # sample counts per atlas label
    counts = pd.Series(atlas).value_counts().sort_index()
    missing = [g for g in groups if g not in counts.index]
    if missing:
        raise ValueError(f"Atlas does not provide counts for groups: {missing}")

    # compute p-value matrices
    def _welch_from_summary(m1, s1, n1, m2, s2, n2) -> float:
        if n1 < 2 or n2 < 2:
            return 1.0
        denom = np.sqrt((s1*s1)/n1 + (s2*s2)/n2)
        if denom == 0:
            return 1.0
        t = (m1 - m2) / denom
        v1, v2 = (s1*s1)/n1, (s2*s2)/n2
        df = (v1 + v2)**2 / ((v1**2)/(n1-1) + (v2**2)/(n2-1))
        return float(2.0 * t_dist.sf(np.abs(t), df))

    p_mats: List[pd.DataFrame] = []
    for g in groups:
        n = int(counts.loc[g])
        std_row = (error_df.loc[g].astype(float) * np.sqrt(n))
        mat = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols, dtype=float)
        for i, c1 in enumerate(cols):
            for j in range(i+1, len(cols)):
                c2 = cols[j]
                p = _welch_from_summary(float(mean_df.loc[g, c1]), float(std_row.loc[c1]), n,
                                        float(mean_df.loc[g, c2]), float(std_row.loc[c2]), n)
                mat.iat[i, j] = mat.iat[j, i] = p
        _apply_multitest_inplace(mat, alpha=alpha, method=multitest)
        p_mats.append(mat)

    # figure layout
    nG = len(groups)
    if layout is None:
        r = int(math.floor(math.sqrt(nG)))
        c = int(math.ceil(nG / max(r, 1))) if r else nG
        layout = (r, c) if r and r * c >= nG else (math.ceil(nG / 2), 2)
    nrows, ncols = layout

    if figsize[0] is None or figsize[1] is None:
        w = max(4.0, 1.2 * len(cols) * ncols)
        h = max(3.5, 1.2 * len(cols) * nrows)
        figsize = (w, h)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    ax_array = np.atleast_1d(axes).ravel()
    heatmap_kwargs = heatmap_kwargs or {}

    # title colors
    if title_colors is None:
        title_colors = _colorblind_palette(nG)
    elif len(title_colors) < nG:
        raise ValueError("title_colors length must be >= number of groups")

    for k, (g, mat) in enumerate(zip(groups, p_mats)):
        ax = ax_array[k]
        mask_sig = mat.values < alpha
        pretty = [c.replace("_", " ") for c in cols]
        sns.heatmap(mat, ax=ax, xticklabels=pretty, yticklabels=pretty,
                    annot=True, fmt=".3f", cbar=False, mask=mask_sig, vmin=0, vmax=1,
                    cmap=base_cmap, **heatmap_kwargs)
        sns.heatmap(mat, ax=ax, xticklabels=pretty, yticklabels=pretty,
                    annot=True, fmt=".3f", cbar=False, mask=~mask_sig, vmin=0, vmax=1,
                    cmap=_solid_cmap(sig_color), **heatmap_kwargs)
        ax.set_title(str(g), fontsize=12, color=title_colors[k])

    # remove any unused axes
    for k in range(nG, nrows * ncols):
        fig.delaxes(ax_array[k])

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_plot and output_directory:
        _validate_save_format(save_format)
        out = Path(output_directory) / "node_removal_p_value_plots"
        out.mkdir(parents=True, exist_ok=True)
        fp = out / f"node_removal_pvalues_{multitest or 'uncorrected'}.{save_format.lower()}"
        fig.savefig(fp, format=save_format.lower(), bbox_inches="tight")
        logger.info("plot_node_removal_p_values: saved %s", fp)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    logger.info("plot_node_removal_p_values: done")
    return p_mats
