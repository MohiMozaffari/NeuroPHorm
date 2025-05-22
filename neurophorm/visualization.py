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
warnings.filterwarnings('ignore', category=RuntimeWarning)

def plot_betti_curves(
    data: Dict[str, Dict[str, npt.NDArray]],
    dimensions: List[int] = [0, 1, 2],
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
    Plots mean Betti curves with standard deviation shading for different labels.

    Args:
        data: Dictionary with labels as keys and dicts containing "betti_x" and "betti_curves" as values
        dimensions: Homology dimensions to plot
        labels: Specific labels to plot; if None, uses all keys in data
        label_styles: Dict mapping labels to (color, linestyle) tuples; if None, uses defaults
        output_directory: Directory to save plots (default: "./")
        show_plot: Whether to display the plot
        save_plot: Whether to save the plot
        save_format: File format for saving ('pdf', 'png', 'svg', 'jpg'; default: 'pdf')
        same_size: If True, enforces consistent x and y limits across plots
        xlim: Optional tuple of (min, max) for x-axis limits
        ylim: Optional tuple of (min, max) for y-axis limits
        figsize: Figure size (width, height); height auto-computed if None
    """
    labels = labels if labels is not None else list(data.keys())
    if type(labels) != list:
        raise TypeError("labels must be a list of strings")
    
    data = {k: data[k] for k in labels if k in data}
    
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
                       label=label.replace("_", " ").title())

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
    dimensions: List[int] = [0, 1, 2],
    output_directory: Union[str, Path] = "./",
    test: str = "auto",
    show_plot: bool = True,
    save_plot: bool = False,
    save_format: str = "pdf",
    figsize: Tuple[float, float] = (None, None),
    heatmap_kwargs: Optional[Dict] = None
) -> List[pd.DataFrame]:
    """
    Calculates and visualizes p-values from statistical tests between groups.

    Args:
        data: Dictionary with labels as keys and feature data as values
        feature_name: Name of the feature to analyze
        labels: Specific labels to compare; if None, uses all keys
        dimensions: Homology dimensions to analyze
        output_directory: Directory to save plots (default: "./")
        test: Statistical test ("t_test", "wilcoxon", or "auto")
        show_plot: Whether to display the plot
        save_plot: Whether to save the plot
        save_format: File format for saving ('pdf', 'png', 'svg', 'jpg'; default: 'pdf')
        figsize: Figure size (width, height); auto-computed if None
        heatmap_kwargs: Additional seaborn heatmap kwargs (e.g., {'annot_kws': {'size': 8}})
    """
    labels = labels if labels is not None else list(data.keys())
    data = {k: data[k] for k in labels if k in data}
    p_values = [pd.DataFrame(np.ones((len(labels), len(labels))), index=labels, columns=labels) 
                for _ in dimensions]
    cmap = ListedColormap("#3b4cc0")

    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels[i+1:], i+1):
            curves1 = data[label1][feature_name]
            curves2 = data[label2][feature_name]

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
        pretty_labels = [l.replace("_", " ").title() for l in labels]
        sns.heatmap(p_values[dim_idx], ax=axs[dim_idx], xticklabels=pretty_labels, yticklabels=pretty_labels,
                   annot=True, fmt=".2f", cbar=False, mask=mask, vmin=0, vmax=1, **heatmap_kwargs)
        sns.heatmap(p_values[dim_idx], ax=axs[dim_idx], xticklabels=pretty_labels, yticklabels=pretty_labels,
                   annot=True, fmt=".2f", cbar=False, cmap=cmap, mask=~mask, **heatmap_kwargs)
        axs[dim_idx].set_title(fr"$H_{dim}$", fontsize=18)

    title = (f"p-Values from {'T-Test or Wilcoxon' if test == 'auto' else test.title()} "
            f"for {feature_name.replace('_', ' ').title()}")
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
        plt.savefig(save_path, format=save_format.lower(), bbox_inches='tight', **heatmap_kwargs)
        print(f"p-value plot saved to {save_path}")

    if show_plot:
        plt.show()
    plt.close()
    return p_values


def plot_grouped_p_value_heatmaps(
    p_values: List[pd.DataFrame],
    group_ranges: Dict[str, Tuple[List[int], List[str]]],
    name: str,
    dimensions: List[int] = [0, 1, 2],
    output_directory: Union[str, Path] = "./",
    show_plot: bool = True,
    save_plot: bool = False,
    save_format: str = "pdf",
    subplot_layout: Optional[List[Tuple[str, str]]] = None,
    figsize: Tuple[float, float] = (None, None),
    heatmap_kwargs: Optional[Dict] = None
) -> None:
    """
    Creates heatmaps comparing p-values across user-defined groups.

    Args:
        p_values: List of p-value DataFrames, one per homology dimension
        group_ranges: Dict mapping group names to (indices, labels) tuples
        name: Feature name for visualization
        dimensions: Homology dimensions to plot
        output_directory: Directory to save plots (default: "./")
        show_plot: Whether to display plots
        save_plot: Whether to save plots
        save_format: File format for saving ('pdf', 'png', 'svg', 'jpg'; default: 'pdf')
        subplot_layout: List of (key, title) tuples defining subplot arrangement
        figsize: Figure size (width, height); auto-computed if None
        heatmap_kwargs: Additional seaborn heatmap kwargs
    """
    if not group_ranges:
        raise ValueError("group_ranges dictionary cannot be empty")
    if not dimensions:
        raise ValueError("dimensions list cannot be empty")

    cmap = ListedColormap("#3b4cc0")
    extract_subset = lambda dist, indices: dist[np.ix_(indices, indices)]

    # Default subplot layout if none provided
    if subplot_layout is None:
        subplot_layout = [(key, key.title()) for key in group_ranges.keys()]
    
    # Create mosaic layout dynamically
    n_subplots = len(subplot_layout)
    rows = (n_subplots + 1) // 2
    mosaic = [["."] * 2 for _ in range(rows)]
    for i, (key, _) in enumerate(subplot_layout):
        row, col = divmod(i, 2)
        mosaic[row][col] = key

    heatmap_kwargs = heatmap_kwargs or {}

    for dim in dimensions:
        if dim >= len(p_values):
            raise ValueError(f"Dimension {dim} exceeds number of p-value DataFrames")
        
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

        plt.suptitle(f"p-Values for {name.replace('_', ' ').title()} $H_{dim}$ Across Groups")
        plt.tight_layout()

        if save_plot and output_directory:
            output_directory = Path(output_directory)
            plot_dir = output_directory / "grouped_heatmap_plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            valid_formats = ['pdf', 'png', 'svg', 'jpg']
            if save_format.lower() not in valid_formats:
                raise ValueError(f"save_format must be one of {valid_formats}")
            save_path = plot_dir / f"{name}_pvalue_heatmap_dim_{dim}.{save_format.lower()}"
            plt.savefig(save_path, format=save_format.lower(), bbox_inches='tight', **heatmap_kwargs)
            print(f"Heatmap plot saved to {save_path}")

        if show_plot:
            plt.show()
        plt.close()


def plot_grouped_distance_heatmaps(
    distances: npt.NDArray,
    group_ranges: Dict[str, Tuple[List[int], List[str]]],
    name: str,
    dimensions: List[int] = [-1, 0, 1, 2],
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
    Creates heatmaps showing distance comparisons across user-defined groups.

    Args:
        distances: Array of distance matrices per dimension
        group_ranges: Dict mapping group names to (indices, labels) tuples
        name: Feature name for visualization
        dimensions: Dimensions to plot (-1 for mean across all)
        output_directory: Directory to save plots (default: "./")
        show_plot: Whether to display plots
        save_plot: Whether to save plots
        save_format: File format for saving ('pdf', 'png', 'svg', 'jpg'; default: 'pdf')
        subplot_layout: List of (key, title) tuples defining subplot arrangement
        block_size: Size of blocks for averaging and grid lines
        figsize: Figure size (width, height); auto-computed if None
        heatmap_kwargs: Additional seaborn heatmap kwargs
    """
    if not group_ranges:
        raise ValueError("group_ranges dictionary cannot be empty")
    if not dimensions:
        raise ValueError("dimensions list cannot be empty")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    
    for label, (indeces , groups) in group_ranges.items():
        for inx in range(len(indeces)):
            indeces[inx] = np.arange(indeces[inx] * block_size, indeces[inx] * block_size + block_size) 
        indeces = np.concatenate(indeces).astype(int)
        group_ranges[label] = (indeces, groups)

    extract_subset = lambda dist, indices: dist[np.ix_(indices, indices)]

    # Default subplot layout if none provided
    if subplot_layout is None:
        subplot_layout = [(key, key.title()) for key in group_ranges.keys()]
    
    # Create mosaic layout dynamically
    n_subplots = len(subplot_layout)
    rows = (n_subplots + 1) // 2
    mosaic = [["."] * 2 for _ in range(rows)]
    for i, (key, _) in enumerate(subplot_layout):
        row, col = divmod(i, 2)
        mosaic[row][col] = key

    heatmap_kwargs = heatmap_kwargs or {}

    for dim in dimensions:
        if dim >= distances.shape[0] and dim != -1:
            raise ValueError(f"Dimension {dim} exceeds number of distance matrices")
        
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

        title = (f"Mean {name.replace('_', ' ').title()} Distance" if dim == -1 else
                f"{name.replace('_', ' ').title()} Distance $H_{dim}$")
        
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
            plt.savefig(save_path, format=save_format.lower(), bbox_inches='tight', **heatmap_kwargs)
            print(f"Heatmap plot saved to {save_path}")

        if show_plot:
            plt.show()
        plt.close()


def plot_swarm_violin(
    data: Dict[str, Dict[str, npt.NDArray]],
    feature_name: str,
    dimensions: List[int] = [0, 1, 2],
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
    Plots swarm and violin plots for given data.

    Args:
        data: Dictionary with labels as keys and feature data as values
        feature_name: Name of the feature to plot
        dimensions: Homology dimensions to plot
        labels: Specific labels to include; if None, uses all keys
        label_styles: Dict mapping labels to (color, linestyle) tuples
        output_directory: Directory to save plots (default: "./")
        show_plot: Whether to display the plot
        save_plot: Whether to save the plot
        save_format: File format for saving ('pdf', 'png', 'svg', 'jpg'; default: 'pdf')
        figsize: Figure size (width, height); auto-computed if None
        swarm_plot_kwargs: Additional seaborn kwargs for swarm plots
        violin_plot_kwargs: Additional seaborn kwargs for violin plots
    """
    labels = labels if labels is not None else list(data.keys())
    data = {k: data[k] for k in labels if k in data}
    
    palette = ([c for c, _ in label_styles.values()] if label_styles else "pastel")
    width = figsize[0] if figsize[0] is not None else len(data) * 1.2
    height = figsize[1] if figsize[1] is not None else 3 * len(dimensions)
    fig, axs = plt.subplots(len(dimensions), 1, figsize=(width, height))
    axs = [axs] if len(dimensions) == 1 else axs

    swarm_plot_kwargs = swarm_plot_kwargs or {}
    violin_plot_kwargs = violin_plot_kwargs or {}

    for i, dim in enumerate(dimensions):
        data_to_plot = [np.array(data[label][feature_name])[:, dim] for label in data]
        pretty_labels = [label.replace("_", " ").title() for label in labels]
        
        sns.violinplot(data=data_to_plot, ax=axs[i], inner=None, palette=palette, **violin_plot_kwargs)
        sns.swarmplot(data=data_to_plot, ax=axs[i], edgecolor="black", palette=palette, **swarm_plot_kwargs)
        
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
            raise ValueError(f"save_format must be one of {valid_formats}")
        save_path = plot_dir / f"{feature_name}_swarm_violin.{save_format.lower()}"
        plt.savefig(save_path, format=save_format.lower(), bbox_inches='tight', **plot_kwargs)
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
    """
    Plots KDE plots for the given feature across different labels.

    Args:
        data: Dictionary with labels as keys and feature data as values
        feature_name: Name of the feature to plot
        labels: Specific labels to include; if None, uses all keys
        label_styles: Dict mapping labels to (color, linestyle) tuples
        output_directory: Directory to save plots (default: "./")
        show_plot: Whether to display the plot
        save_plot: Whether to save the plot
        save_format: File format for saving ('pdf', 'png', 'svg', 'jpg'; default: 'pdf')
        figsize: Figure size (width, height)
        kde_kwargs: Additional seaborn kdeplot kwargs
    """
    labels = labels if labels is not None else list(data.keys())
    data = {k: data[k] for k in labels if k in data}
    
    if not label_styles:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        linestyles = ['-', '--', '-.', ':']
        label_styles = {label: (colors[i % len(colors)], linestyles[i % len(linestyles)]) 
                       for i, label in enumerate(data.keys())}

    fig, ax = plt.subplots(figsize=figsize)
    kde_kwargs = kde_kwargs or {}

    for label, (color, linestyle) in label_styles.items():
        distances = data[label][feature_name]
        distances = distances.mean(axis=0).flatten()
        sns.kdeplot(distances, ax=ax, color=color, linestyle=linestyle, 
                   label=label.replace("_", " ").title(), **kde_kwargs)

    ax.set_title(f"KDE Plot of {feature_name.replace('_', ' ').title()}")
    ax.set_xlabel(feature_name.replace('_', ' ').title())
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
        plt.savefig(save_path, format=save_format.lower(), bbox_inches='tight', **kde_kwargs)
        print(f"KDE plot saved to {save_path}")

    if show_plot:
        plt.show()
    plt.close()
