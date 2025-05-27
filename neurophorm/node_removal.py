from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import numpy as np
import numpy.typing as npt
import pandas as pd
from neurophorm.persistence import (
    rips_persistence_diagrams,
    diagram_distances,
    save_tda_results
)

def node_removal_persistence(
    distance_matrix: npt.NDArray,
    output_directory: Union[str, Path]="./",
    homology_dimensions: List[int] = [0, 1, 2],
    mode: Optional[str] = None,
    infinity: float = 10.0,
    verbose: bool = True,
    return_data: bool = False,
    format="csv",
    **kwargs
) -> Union[str, List[npt.NDArray]]:
    """
    Compute persistence diagrams for a distance matrix with each node removed.

    Args:
        distance_matrix: Input distance matrix as a NumPy array (n_points, n_points)
        output_directory: Directory to save persistence diagrams
        homology_dimensions: Homology dimensions to compute (default: [0, 1, 2])
        mode: Mode for persistence computation (None, "sparse"; default: None)
        infinity: Value to set for removed nodes (default: 10.0)
        verbose: Whether to print progress messages (default: True)
        return_data: Return persistence diagrams instead of saving (default: False)
        format: Format for saving results (default: "csv")
        **kwargs: Additional arguments passed to rips_persistence_diagrams

    Returns:
        Union[str, List[npt.NDArray]]: Directory path if saved, else list of persistence diagrams

    Raises:
        ValueError: If distance_matrix is invalid or mode is not supported
        TypeError: If distance_matrix is not a NumPy array
    """
    valid_modes = {None, "sparse"}
    if mode not in valid_modes:
        raise ValueError(f"Mode must be one of {valid_modes}")

    if not isinstance(distance_matrix, np.ndarray):
        raise TypeError("distance_matrix must be a NumPy array")
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True) if return_data else None

    num_nodes = distance_matrix.shape[0]
    if num_nodes == 0:
        if verbose:
            print("Distance matrix is empty")
        return "" if not return_data else []
    
        # Compute persistence diagrams for node removals
    modified_distances = []
    for node_to_remove in range(num_nodes):
        modified_distance = distance_matrix.copy()
        modified_distance[node_to_remove, :] = infinity
        modified_distance[:, node_to_remove] = infinity
        np.fill_diagonal(modified_distance, 0)

        modified_distances.append(modified_distance)

    # Compute persistence diagram for original network
    all_diagrams = rips_persistence_diagrams(
        [distance_matrix] + modified_distances,
        mode=mode,
        homology_dimensions=homology_dimensions,
        **kwargs
    )

    results = {}
    persistence_folder = output_directory / "persistence_diagrams"
    persistence_folder.mkdir(exist_ok=True)

    # Save or return results
    for i, diagram in enumerate(all_diagrams):
        filename = f"persistence_diagram_node_{i:03d}"
        results[str(persistence_folder / filename)] = diagram

    if return_data:
        return all_diagrams
    else:
        save_tda_results(results, format=format, overwrite=True)
        if verbose:
            print(f"Saved persistence diagrams to {persistence_folder}")
        return str(persistence_folder)

def node_removal_differences(
    persistence_diagrams: List[npt.NDArray],
    output_directory: Union[str, Path] = "./",
    metrics: List[str] = ["wasserstein", "bottleneck"],
    return_data: bool = False,
    output_filename: Optional[str] = None,
    verbose: bool = True,
    format: str = "csv",
    kwargs: Optional[Dict] = {"order":None, "n_jobs": -1}
) -> Union[str, pd.DataFrame]:
    """
    Compute pairwise distances between original and node-removed persistence diagrams.

    Args:
        persistence_diagrams: List of persistence diagrams (original + node removals)
        output_directory: Directory to save results
        metrics: Distance metrics to compute (default: ["wasserstein", "bottleneck"])
        return_data: Return DataFrame instead of saving (default: False)
        output_filename: Filename for the output CSV (default: None)
        verbose: Whether to print progress messages (default: True)
        format: Format for saving results (default: "csv")
        kwargs: Additional arguments for diagram_distances

    Returns:
        Union[str, pd.DataFrame]: Path to saved CSV or DataFrame of distances

    Raises:
        ValueError: If persistence_diagrams list is too short
    """
    if len(persistence_diagrams) < 2:
        raise ValueError("At least two persistence diagrams required (original + one removal)")

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    homology_dimensions = list(range(np.max(persistence_diagrams[0], axis=1)[2].astype(int) + 1))

    all_distances = []
    for idx in range(1, len(persistence_diagrams)):
        distances_for_node = {}
        distance_results = diagram_distances(
            [persistence_diagrams[0], persistence_diagrams[idx]],
            metrics=metrics,
            **kwargs
        )
        if verbose:
            print(f"Computed distances for node {idx} removal")

        for metric in metrics:
            for dim in homology_dimensions:
                key = f"{metric}_distance_H{dim}"
                if key in distance_results:
                    distances_for_node[f"{metric}_H{dim}"] = distance_results[key][0, 1]
        
        all_distances.append(distances_for_node)


    df_distances = pd.DataFrame(all_distances)
    df_distances.index.name = "Removed Node"
    
    df_distances.drop(0, inplace=True)
    dict_distances = df_distances.to_dict(orient="dict")



    if output_filename is None:
        output_filename = "node_removal_distances"

    if format == "csv":
        output_filename += ".csv"
    elif format == "npy":
        output_filename += ".npy"

    elif format == "txt":
        output_filename += ".txt"
    else:
        raise ValueError("Unsupported format. Use 'csv' or 'npy or 'txt'.")


    if return_data:
        return df_distances
    else:
        output_filepath = output_directory / output_filename
        save_tda_results(
            {str(output_filepath): dict_distances},
            format=format,
            overwrite=True
        )
        if verbose:
            print(f"Saved distances to {output_filepath}")
        return str(output_filepath)

