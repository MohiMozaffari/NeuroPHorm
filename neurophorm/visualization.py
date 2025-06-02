from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import ttest_ind, wilcoxon, shapiro, levene
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import ListedColormap
import seaborn as sns
import warnings
import copy

warnings.filterwarnings('ignore', category=RuntimeWarning)

def _infer_dimensions(data: Dict[str, Dict[str, npt.NDArray]]) -> List[int]:
    """Infer homology dimensions from persistence diagrams or Betti curves.

    This function extracts unique homology dimensions from the input data, either
    from persistence diagrams (third column) or Betti curves (second dimension for
    3D arrays, first for 2D arrays). It is used to automatically determine dimensions
    for plotting functions when not specified by the user.

    Parameters
    ----------
    data : Dict[str, Dict[str, npt.NDArray]]
        Dictionary with labels as keys and nested dictionaries containing
        'persistence_diagrams' or 'betti_curves' arrays. Persistence diagrams have
        shape (n_points, 3) with the third column as the homology dimension. Betti
        curves are 2D (n_dims, n_bins) or 3D (n_samples, n_dims, n_bins).

    Returns
    -------
    List[int]
        Sorted list of unique homology dimensions (e.g., [0, 1, 2]).

    Raises
    ------
    ValueError
        If no valid dimensions can be inferred from the data (e.g., missing or
        empty 'persistence_diagrams' or 'betti_curves').
    """
    dimensions = set()
    for label, label_data in data.items():
        if "persistence_diagrams" in label_data:
            for diagram in label_data["persistence_diagrams"]:
                if diagram.size > 0:  # Ensure diagram is not empty
                    dimensions.update(diagram[:, 2].astype(int))
        elif "betti_curves" in label_data:
            betti_curves = label_data["betti_curves"]
            if betti_curves.ndim == 3:
                dimensions.update(range(betti_curves.shape[1]))
            elif betti_curves.ndim == 2:
                dimensions.update(range(betti_curves.shape[0]))
    
    if not dimensions:
        raise ValueError("Could not infer dimensions from data. Ensure 'persistence_diagrams' or 'betti_curves' are present.")
    
    return sorted(dimensions)

def _compute_betti_auc(
    data: Dict[str, Dict[str, np.ndarray]],
    labels: List[str],
    dimensions: Optional[List[int]] = None
) -> Dict[str, np.ndarray]:
    """
    Compute AUC for Betti curves for each sample, dimension, and group.

    Args:
        data: Dictionary with group labels as keys, containing 'betti_curves' and 'betti_x'.
        labels: List of group labels to process.
        dimensions: Homology dimensions to analyze. If None, inferred from data.

    Returns:
        Dictionary with group labels as keys and AUC arrays of shape (n_samples, n_dims).
    """
    if dimensions is None:
        dimensions = _infer_dimensions(data)  # Assume _infer_dimensions is defined

    auc_data = {}
    for label in labels:
        if "betti_curves" not in data[label] or "betti_x" not in data[label]:
            raise ValueError(f"Missing 'betti_curves' or 'betti_x' for {label}")

        curves = data[label]["betti_curves"]  # Shape: (n_samples, n_dims, n_bins)
        x = data[label]["betti_x"]  # Shape: (n_bins,) or (n_dims, n_bins)

        n_samples, n_dims, n_bins = curves.shape
        auc = np.zeros((n_samples, len(dimensions)))

        for dim_idx, dim in enumerate(dimensions):
            if dim >= n_dims:
                continue
            # Select x-values for this dimension
            x_dim = x[dim] if x.ndim > 1 else x
            for sample_idx in range(n_samples):
                y = curves[sample_idx, dim, :]
                valid_mask = ~np.isnan(y) & ~np.isnan(x_dim)
                if np.sum(valid_mask) < 2:  # Need at least 2 points for AUC
                    auc[sample_idx, dim_idx] = np.nan
                else:
                    auc[sample_idx, dim_idx] = np.trapz(y[valid_mask], x_dim[valid_mask])

        auc_data[label] = auc

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
    """Plot mean Betti curves with standard deviation shading for multiple groups.

    This function visualizes Betti curves, which represent the number of topological
    features (e.g., connected components, loops, voids) at different filtration
    parameters in topological data analysis (TDA). For each homology dimension, it
    plots the mean Betti curve across samples with shaded standard error regions
    for each group.

    Parameters
    ----------
    data : Dict[str, Dict[str, npt.NDArray]]
        Dictionary with group labels as keys and nested dictionaries containing
        'betti_curves' (2D or 3D array) and 'betti_x' (filtration values per
        dimension). 'betti_curves' can be 2D (n_dims, n_bins) for single curves or
        3D (n_samples, n_dims, n_bins) for multiple samples.
    dimensions : Optional[List[int]], optional
        Homology dimensions to plot (e.g., [0, 1] for H0 and H1). If None, dimensions
        are inferred from 'persistence_diagrams' or 'betti_curves' in the data.
        Default is None.
    labels : Optional[List[str]], optional
        List of group labels to plot. If None, uses all keys in `data`. Default is None.
    label_styles : Optional[Dict[str, Tuple[str, str]]], optional
        Dictionary mapping labels to (color, linestyle) tuples for plotting. If None,
        assigns default colors and linestyles cyclically. Default is None.
    output_directory : Union[str, Path], optional
        Directory to save the plot. Created if it doesn't exist. Default is "./".
    show_plot : bool, optional
        Whether to display the plot. Default is True.
    save_plot : bool, optional
        Whether to save the plot to `output_directory`. Default is False.
    save_format : str, optional
        File format for saving the plot ('pdf', 'png', 'svg', 'jpg'). Default is 'pdf'.
    same_size : bool, optional
        If True, uses consistent x and y limits across all subplots based on data
        ranges. Default is False.
    xlim : Optional[Tuple[float, float]], optional
        Custom x-axis limits (min, max). If None, limits are computed from data.
        Default is None.
    ylim : Optional[Tuple[float, float]], optional
        Custom y-axis limits (min, max). If None, limits are computed from data.
        Default is None.
    figsize : Tuple[float, float], optional
        Figure size (width, height). If height is None, it is computed as
        len(dimensions) * 2.5. Default is (7, None).

    Raises
    ------
    TypeError
        If `labels` is not a list of strings.
    ValueError
        If specified `dimensions` are not found in the data, or if `save_format` is
        invalid.
    """
    labels = labels if labels is not None else list(data.keys())
    if not isinstance(labels, list):
        raise TypeError("labels must be a list of strings")
    
    data = {k: data[k] for k in labels if k in data}
    
    # Infer dimensions if not provided
    if dimensions is None:
        dimensions = _infer_dimensions(data)
    
    # Validate dimensions
    available_dims = _infer_dimensions(data)
    invalid_dims = [d for d in dimensions if d not in available_dims]
    if invalid_dims:
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
                betti_curves = data[label]["betti_curves"]
                if betti_curves.ndim == 3:
                    mean_curve = np.mean(betti_curves[:, dim, :], axis=0)
                    std_curve = np.std(betti_curves[:, dim, :], axis=0) / np.sqrt(betti_curves.shape[0])
                    all_y.extend(mean_curve - std_curve)
                    all_y.extend(mean_curve + std_curve)
                elif betti_curves.ndim == 2:
                    mean_curve = betti_curves[dim, :]
                    all_y.extend(mean_curve)
                
                filtration_values = data[label]["betti_x"][dim]
                all_x.extend(filtration_values)

        computed_xlim = (min(all_x), max(all_x)) if xlim is None else xlim
        computed_ylim = (min(0, min(all_y)), max(all_y) * 1.1) if ylim is None else ylim
    else:
        computed_xlim = xlim
        computed_ylim = ylim

    for i, dim in enumerate(dimensions):
        for label, (color, linestyle) in label_styles.items():
            betti_curves = data[label]["betti_curves"]
            filtration_values = data[label]["betti_x"][dim]
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
            raise ValueError(f"save_format must be one of {valid_formats}")
        save_path = plot_dir / f"mean_betti_curves.{save_format.lower()}"
        plt.savefig(save_path, format=save_format.lower(), bbox_inches='tight')
        print(f"Betti curve plot saved to {save_path}")

    if show_plot:
        plt.show()
    plt.close()

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
    """Visualize p-values from statistical tests comparing TDA features between groups.

    This function performs statistical tests (t-test or Wilcoxon) on a specified TDA
    feature (e.g., Betti curves, persistence entropy) across groups and visualizes the
    resulting p-values as heatmaps for each homology dimension. Significant p-values
    (< 0.05) are highlighted. The test type can be specified or chosen automatically
    based on normality (Shapiro-Wilk test).

    Parameters
    ----------
    data : Dict[str, Dict[str, npt.NDArray]]
        Dictionary with group labels as keys and nested dictionaries containing TDA
        features (e.g., 'betti_curves', 'persistence_entropy'). Features can be 2D
        (n_samples, n_dims) or 3D (n_samples, n_dims, n_bins).
    feature_name : str
        Name of the TDA feature to analyze (e.g., 'betti_curves', 'persistence_entropy').
    labels : Optional[List[str]], optional
        List of group labels to compare. If None, uses all keys in `data`. Default is None.
    dimensions : Optional[List[int]], optional
        Homology dimensions to analyze (e.g., [0, 1]). If None, dimensions are inferred
        from 'persistence_diagrams' or the feature data. Default is None.
    output_directory : Union[str, Path], optional
        Directory to save the plot. Created if it doesn't exist. Default is "./".
    test : str, optional
        Statistical test to use: 't_test', 'wilcoxon', or 'auto'. If 'auto', chooses
        t-test for normally distributed data (Shapiro-Wilk p > 0.05) or Wilcoxon
        otherwise. Default is 'auto'.
    show_plot : bool, optional
        Whether to display the plot. Default is True.
    save_plot : bool, optional
        Whether to save the plot to `output_directory`. Default is False.
    save_format : str, optional
        File format for saving the plot ('pdf', 'png', 'svg', 'jpg'). Default is 'pdf'.
    figsize : Tuple[float, float], optional
        Figure size (width, height). If None, computed as (len(labels) * 1.2 * len(dimensions),
        len(labels) * 1.2). Default is (None, None).
    heatmap_kwargs : Optional[Dict], optional
        Additional keyword arguments for seaborn.heatmap (e.g., {'annot_kws': {'size': 8}}).
        Default is None.

    Returns
    -------
    List[pd.DataFrame]
        List of DataFrames, one per homology dimension, containing p-values for pairwise
        comparisons between groups.

    Raises
    ------
    ValueError
        If `feature_name` is not in `data`, if `dimensions` are invalid, if `test` is
        invalid, if `save_format` is invalid, or if feature shapes are incompatible.
    """
    labels = labels if labels is not None else list(data.keys())
    data = {k: data[k] for k in labels if k in data}
    
    # Validate feature_name
    if not all(feature_name in data[label] for label in labels):
        raise ValueError(f"Feature '{feature_name}' not found in all group data")
    
    # Infer dimensions if not provided
    if dimensions is None:
        dimensions = _infer_dimensions(data)
    
    # Validate dimensions
    available_dims = _infer_dimensions(data)
    invalid_dims = [d for d in dimensions if d not in available_dims]
    if invalid_dims:
        raise ValueError(f"Specified dimensions {invalid_dims} not found in data. Available: {available_dims}")
    
    p_values = [pd.DataFrame(np.ones((len(labels), len(labels))), index=labels, columns=labels) 
                for _ in dimensions]
    cmap = ListedColormap("#3b4cc0")

    if feature_name == "betti_curves":
        data = _compute_betti_auc(data, labels, dimensions)

    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels[i+1:], i+1):
            if feature_name == "betti_curves":
                curves1 = data[label1]
                curves2 = data[label2]
            else:
                curves1 = data[label1][feature_name]
                curves2 = data[label2][feature_name]

            # Validate shapes
            if curves1.shape != curves2.shape:
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
            raise ValueError(f"save_format must be one of {valid_formats}")
        save_path = plot_dir / f"{test}_p_values_for_{feature_name}.{save_format.lower()}"
        plt.savefig(save_path, format=save_format.lower(), bbox_inches='tight')
        print(f"p-value plot saved to {save_path}")

    if show_plot:
        plt.show()
    plt.close()
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
    """Create heatmaps of p-values for user-defined group subsets across homology dimensions.

    This function generates heatmaps to compare p-values of TDA features between
    subsets of groups, defined by `group_ranges`. Each heatmap corresponds to a
    homology dimension and a group subset, with significant p-values (< 0.05)
    highlighted. Subplots are arranged in a mosaic layout.

    Parameters
    ----------
    p_values : List[pd.DataFrame]
        List of p-value DataFrames, one per homology dimension, with group labels
        as indices and columns.
    group_ranges : Dict[str, Tuple[List[int], List[str]]]
        Dictionary mapping group names to tuples of (indices, labels) for selecting
        subsets of p-values. Indices correspond to rows/columns in `p_values`.
    name : str
        Name of the TDA feature for visualization (e.g., 'betti_curves').
    ncol : int, optional
        Number of columns in the mosaic layout. Default is 2.
    dimensions : Optional[List[int]], optional
        Homology dimensions to plot (e.g., [0, 1]). If None, inferred from the length
        of `p_values`. Default is None.
    output_directory : Union[str, Path], optional
        Directory to save the plot. Created if it doesn't exist. Default is "./".
    show_plot : bool, optional
        Whether to display the plot. Default is True.
    save_plot : bool, optional
        Whether to save the plot to `output_directory`. Default is False.
    save_format : str, optional
        File format for saving the plot ('pdf', 'png', 'svg', 'jpg'). Default is 'pdf'.
    subplot_layout : Optional[List[Tuple[str, str]]], optional
        List of (key, title) tuples defining subplot arrangement and titles. If None,
        uses `group_ranges` keys with title-cased names. Default is None.
    figsize : Tuple[float, float], optional
        Figure size (width, height). If None, computed as (2 * n_columns, 2 * n_rows)
        based on subplot layout. Default is (None, None).
    heatmap_kwargs : Optional[Dict], optional
        Additional keyword arguments for seaborn.heatmap. Default is None.

    Raises
    ------
    ValueError
        If `group_ranges` or `p_values` is empty, if `dimensions` exceed `p_values`
        length, or if `save_format` is invalid.
    """
    if not group_ranges:
        raise ValueError("group_ranges dictionary cannot be empty")
    if not p_values:
        raise ValueError("p_values list cannot be empty")
    
    # Infer dimensions if not provided
    if dimensions is None:
        dimensions = list(range(len(p_values)))
    
    # Validate dimensions
    if max(dimensions) >= len(p_values):
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
                raise ValueError(f"save_format must be one of {valid_formats}")
            save_path = plot_dir / f"{name}_pvalue_heatmap_dim_{dim}.{save_format.lower()}"
            plt.savefig(save_path, format=save_format.lower(), bbox_inches='tight')
            print(f"Heatmap plot saved to {save_path}")

        if show_plot:
            plt.show()
        plt.close()

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
    """Create heatmaps of distance matrices for user-defined group subsets.

    This function visualizes distance matrices (e.g., Wasserstein distances between
    persistence diagrams) for subsets of groups defined by `group_ranges`. Each
    heatmap corresponds to a homology dimension or the mean across dimensions (if
    dim = -1). Block averages are overlaid with grid lines to highlight group
    comparisons.

    Parameters
    ----------
    distances : npt.NDArray
        Array of distance matrices with shape (n_dims, n_samples, n_samples).
    group_ranges : Dict[str, Tuple[List[int], List[str]]]
        Dictionary mapping group names to tuples of (indices, labels). Indices are
        expanded into ranges based on `block_size` for selecting distance matrix
        subsets.
    name : str
        Name of the distance metric for visualization (e.g., 'wasserstein_distance').
    ncol : int, optional
        Number of columns in the mosaic layout. Default is 2.
    dimensions : Optional[List[int]], optional
        Dimensions to plot (e.g., [0, 1]). Use -1 for mean across all dimensions.
        If None, includes -1 and all available dimensions. Default is None.
    output_directory : Union[str, Path], optional
        Directory to save the plot. Created if it doesn't exist. Default is "./".
    show_plot : bool, optional
        Whether to display the plot. Default is True.
    save_plot : bool, optional
        Whether to save the plot to `output_directory`. Default is False.
    save_format : str, optional
        File format for saving the plot ('pdf', 'png', 'svg', 'jpg'). Default is 'pdf'.
    subplot_layout : Optional[List[Tuple[str, str]]], optional
        List of (key, title) tuples defining subplot arrangement and titles. If None,
        uses `group_ranges` keys with title-cased names. Default is None.
    block_size : int, optional
        Size of blocks for averaging and drawing grid lines. Default is 40.
    figsize : Tuple[float, float], optional
        Figure size (width, height). If None, computed as (3 * n_columns, 3 * n_rows)
        based on subplot layout. Default is (None, None).
    heatmap_kwargs : Optional[Dict], optional
        Additional keyword arguments for seaborn.heatmap. Default is None.

    Raises
    ------
    ValueError
        If `group_ranges` is empty, `distances` is empty, `block_size` is non-positive,
        `dimensions` are invalid, or `save_format` is invalid.
    """
    if not group_ranges:
        raise ValueError("group_ranges dictionary cannot be empty")
    if not distances.size:
        raise ValueError("distances array cannot be empty")
    if block_size <= 0:
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

    heatmap_kwargs = heatmap_kwargs or {"cmap":"coolwarm"}

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
                raise ValueError(f"save_format must be one of {valid_formats}")
            save_path = plot_dir / f"{name}_distance_heatmap_dim_{dim}.{save_format.lower()}"
            plt.savefig(save_path, format=save_format.lower(), bbox_inches='tight')
            print(f"Heatmap plot saved to {save_path}")

        if show_plot:
            plt.show()
        plt.close()

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
    """Plot swarm and violin plots for TDA features across groups.

    This function creates combined swarm and violin plots to visualize the
    distribution of a TDA feature (e.g., persistence entropy, amplitudes) across
    groups for each homology dimension. Violin plots show the density, while swarm
    plots display individual data points.

    Parameters
    ----------
    data : Dict[str, Dict[str, npt.NDArray]]
        Dictionary with group labels as keys and nested dictionaries containing TDA
        features (e.g., 'persistence_entropy', 'amplitudes'). Features are typically
        2D (n_samples, n_dims).
    feature_name : str
        Name of the TDA feature to plot (e.g., 'persistence_entropy').
    dimensions : Optional[List[int]], optional
        Homology dimensions to plot (e.g., [0, 1]). If None, dimensions are inferred
        from 'persistence_diagrams' or the feature data. Default is None.
    labels : Optional[List[str]], optional
        List of group labels to include. If None, uses all keys in `data`. Default is None.
    label_styles : Optional[Dict[str, Tuple[str, str]]], optional
        Dictionary mapping labels to (color, linestyle) tuples for plotting. If None,
        uses a default 'pastel' palette. Default is None.
    output_directory : Union[str, Path], optional
        Directory to save the plot. Created if it doesn't exist. Default is "./".
    show_plot : bool, optional
        Whether to display the plot. Default is True.
    save_plot : bool, optional
        Whether to save the plot to `output_directory`. Default is False.
    save_format : str, optional
        File format for saving the plot ('pdf', 'png', 'svg', 'jpg'). Default is 'pdf'.
    figsize : Tuple[float, float], optional
        Figure size (width, height). If None, computed as (len(data) * 1.2,
        3 * len(dimensions)). Default is (None, None).
    swarm_plot_kwargs : Optional[Dict], optional
        Additional keyword arguments for seaborn.swarmplot. Default is None.
    violin_plot_kwargs : Optional[Dict], optional
        Additional keyword arguments for seaborn.violinplot. Default is None.

    Raises
    ------
    ValueError
        If `feature_name` is not in `data`, if `dimensions` are invalid, or if
        `save_format` is invalid.
    """
    labels = labels if labels is not None else list(data.keys())
    data = {k: data[k] for k in labels if k in data}
    
    # Validate feature_name
    if not all(feature_name in data[label] for label in labels):
        raise ValueError(f"Feature '{feature_name}' not found in all group data")
    
    # Infer dimensions if not provided
    if dimensions is None:
        dimensions = _infer_dimensions(data)
    
    # Validate dimensions
    available_dims = _infer_dimensions(data)
    invalid_dims = [d for d in dimensions if d not in available_dims]
    if invalid_dims:
        raise ValueError(f"Specified dimensions {invalid_dims} not found in data. Available: {available_dims}")
    
    palette = ([c for c, _ in label_styles.values()] if label_styles else "pastel")
    width = figsize[0] if figsize[0] is not None else len(data) * 1.2
    height = figsize[1] if figsize[1] is not None else 3 * len(dimensions)
    fig, axs = plt.subplots(len(dimensions), 1, figsize=(width, height))
    axs = [axs] if len(dimensions) == 1 else axs

    swarm_plot_kwargs = swarm_plot_kwargs or {}
    violin_plot_kwargs = violin_plot_kwargs or {}

    for i, dim in enumerate(dimensions):
        data_to_plot = [np.array(data[label][feature_name])[:, dim] for label in data]
        pretty_labels = [label.replace("_", " ") for label in labels]
        
        sns.violinplot(data=data_to_plot, ax=axs[i], inner=None, palette=palette, **violin_plot_kwargs)
        sns.swarmplot(data=data_to_plot, ax=axs[i], edgecolor="black", palette=palette, **swarm_plot_kwargs)
        
        axs[i].grid(False)
        axs[i].set_xticks(np.arange(len(pretty_labels)))
        axs[i].set_xticklabels(pretty_labels)
        axs[i].set_title(fr"$H_{dim}$")

    plt.suptitle(f"Swarm and Violin Plots of {feature_name.replace('_', ' ')}")
    plt.tight_layout()

    if save_plot and output_directory:
        output_directory = Path(output_directory)
        plot_dir = output_directory / "swarm_violin_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        valid_formats = ['pdf', 'png', 'svg', 'jpg']
        if save_format.lower() not in valid_formats:
            raise ValueError(f"save_format must be one of {valid_formats}")
        save_path = plot_dir / f"{feature_name}_swarm_violin.{save_format.lower()}"
        plt.savefig(save_path, format=save_format.lower(), bbox_inches='tight', **violin_plot_kwargs)
        print(f"Swarm and violin plot saved to {save_path}")

    if show_plot:
        plt.show()
    plt.close()

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
    """Plot kernel density estimation (KDE) distributions for a TDA feature across groups.

    This function creates KDE plots to visualize the distribution of a TDA feature
    (e.g., persistence entropy, amplitudes) across groups. The feature data is
    averaged over samples and flattened for plotting. Each group is represented by
    a distinct line style and color.

    Parameters
    ----------
    data : Dict[str, Dict[str, npt.NDArray]]
        Dictionary with group labels as keys and nested dictionaries containing TDA
        features (e.g., 'persistence_entropy', 'amplitudes'). Features are typically
        2D (n_samples, n_dims) or 3D arrays.
    feature_name : str
        Name of the TDA feature to plot (e.g., 'persistence_entropy').
    labels : Optional[List[str]], optional
        List of group labels to include. If None, uses all keys in `data`. Default is None.
    label_styles : Optional[Dict[str, Tuple[str, str]]], optional
        Dictionary mapping labels to (color, linestyle) tuples for plotting. If None,
        assigns default colors and linestyles cyclically. Default is None.
    output_directory : Union[str, Path], optional
        Directory to save the plot. Created if it doesn't exist. Default is "./".
    show_plot : bool, optional
        Whether to display the plot. Default is True.
    save_plot : bool, optional
        Whether to save the plot to `output_directory`. Default is False.
    save_format : str, optional
        File format for saving the plot ('pdf', 'png', 'svg', 'jpg'). Default is 'pdf'.
    figsize : Tuple[float, float], optional
        Figure size (width, height). Default is (8, 4).
    kde_kwargs : Optional[Dict], optional
        Additional keyword arguments for seaborn.kdeplot. Default is None.

    Raises
    ------
    ValueError
        If `feature_name` is not in `data`, if feature data is not numeric or cannot
        be flattened, or if `save_format` is invalid.
    """
    labels = labels if labels is not None else list(data.keys())
    data = {k: data[k] for k in labels if k in data}
    
    # Validate feature_name
    if not all(feature_name in data[label] for label in labels):
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
            raise ValueError(f"Cannot process '{feature_name}' for {label}: {str(e)}")
        
        sns.kdeplot(distances, ax=ax, color=color, linestyle=linestyle, 
                    label=label.replace("_", " "), **kde_kwargs)

    ax.set_title(f"KDE Plot of {feature_name.replace('_', ' ')}")
    ax.set_xlabel(feature_name.replace('_', ' '))
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()

    if save_plot and output_directory:
        output_directory = Path(output_directory)
        plot_dir = output_directory / "kde_plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        valid_formats = ['pdf', 'png', 'svg', 'jpg']
        if save_format.lower() not in valid_formats:
            raise ValueError(f"save_format must be one of {valid_formats}")
        save_path = plot_dir / f"{feature_name}_kde.{save_format.lower()}"
        plt.savefig(save_path, format=save_format.lower(), bbox_inches='tight')
        print(f"KDE plot saved to {save_path}")

    if show_plot:
        plt.show()
    plt.close()

