from typing import List,  Optional, Union, Tuple
from pathlib import Path
import os
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
    output_directory: Union[str, Path] = "./",
    mode: Optional[str] = None,
    infinity: float = 10.0,
    verbose: bool = True,
    return_data: bool = False,
    save_format: str = "csv",
    persistence_diagrams_kwargs: Optional[dict] = None,
) -> Union[str, List[npt.NDArray]]:
    """
    Compute persistence diagrams for a distance matrix with each node removed.

    This function generates persistence diagrams for the input distance matrix and
    modified versions where each node's row and column are set to a large value
    (`infinity`) to simulate node removal. Results are either saved to a directory
    or returned as a list of persistence diagrams.

    Parameters
    ----------
    distance_matrix : npt.NDArray
        Square distance matrix of shape (n_points, n_points) representing pairwise
        distances between nodes.
    output_directory : Union[str, Path], optional
        Directory to save persistence diagrams (default: "./").
    mode : Optional[str], optional
        Mode for persistence computation, either None or "sparse" (default: None).
    infinity : float, optional
        Value to set for removed nodes' rows and columns (default: 10.0).
        Should be larger than any finite value in the distance matrix.
    verbose : bool, optional
        If True, print progress messages (default: True).
    return_data : bool, optional
        If True, return the list of persistence diagrams instead of saving them
        (default: False).
    save_format : str, optional
        Format for saving results, typically "csv" (default: "csv").
    persistence_diagrams_kwargs : Optional[dict], optional
        Additional arguments to pass to `rips_persistence_diagrams` (default: None).

    Returns
    -------
    Union[str, List[npt.NDArray]]
        If `return_data=False`, returns the path to the directory where diagrams
        are saved as a string. If `return_data=True`, returns a list of persistence
        diagrams as NumPy arrays, where the first diagram corresponds to the original
        distance matrix and subsequent diagrams correspond to each node removal.

    Raises
    ------
    ValueError
        If `mode` is not None or "sparse", or if `distance_matrix` is not a square
        2D array.
    TypeError
        If `distance_matrix` is not a NumPy array.

    Notes
    -----
    - The function assumes the `rips_persistence_diagrams` and `save_tda_results`
      functions are available from the `neurophorm.persistence` module.
    - The `infinity` value should be sufficiently large to effectively "remove"
      nodes by making their distances larger than any other in the matrix.
    - Diagrams are saved in a subdirectory named "persistence_diagrams" within
      `output_directory`.

    Examples
    --------
    >>> import numpy as np
    >>> dist_matrix = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    >>> result = node_removal_persistence(dist_matrix, return_data=True)
    >>> print(len(result))  # Original + 3 node removals
    4
    """
    valid_modes = {None, "sparse"}
    if mode not in valid_modes:
        raise ValueError(f"Mode must be one of {valid_modes}")

    if not isinstance(distance_matrix, np.ndarray):
        raise TypeError("distance_matrix must be a NumPy array")
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix must be a square 2D array")


    persistence_diagrams_kwargs = persistence_diagrams_kwargs or {}

    output_directory = Path(output_directory)
    if not return_data:
        output_directory.mkdir(parents=True, exist_ok=True)

    num_nodes = distance_matrix.shape[0]
    if num_nodes == 0:
        if verbose:
            print("Distance matrix is empty")
        return "" if not return_data else []

    # Compute persistence diagrams for node removals
    modified_distances = [distance_matrix]  # Include original matrix
    for node_to_remove in range(num_nodes):
        modified_distance = distance_matrix.copy()
        modified_distance[node_to_remove, :] = infinity
        modified_distance[:, node_to_remove] = infinity
        np.fill_diagonal(modified_distance, 0)
        modified_distances.append(modified_distance)

    # Compute persistence diagrams
    all_diagrams = rips_persistence_diagrams(
        modified_distances,
        mode=mode,
        **persistence_diagrams_kwargs
    )

    if return_data:
        return all_diagrams

    # Save results
    persistence_folder = output_directory / "persistence_diagrams"
    persistence_folder.mkdir(exist_ok=True)
    results = {}
    for i, diagram in enumerate(all_diagrams):
        filename = "persistence_diagrams_original" if i == 0 else f"persistence_diagrams_node_{i:03d}"
        results[str(persistence_folder / filename)] = diagram

    save_tda_results(results, format=save_format, overwrite=True)
    if verbose:
        print(f"Saved {len(all_diagrams)} persistence diagrams to {persistence_folder}")
    return str(persistence_folder)

def node_removal_differences(
    persistence_diagrams: List[npt.NDArray],
    output_directory: Union[str, Path] = "./",
    metrics: List[str] = ["wasserstein", "bottleneck"],
    return_data: bool = False,
    output_filename: Optional[str] = None,
    verbose: bool = True,
    save_format: str = "csv",
    pairwise_distances_kwargs: Optional[dict] = None,
) -> Union[str, pd.DataFrame]:
    """
    Compute pairwise distances between the original and node-removed persistence diagrams.

    This function calculates distances (e.g., Wasserstein, bottleneck) between the
    original persistence diagram and those generated by removing each node. Results
    are either saved to a file or returned as a pandas DataFrame.

    Parameters
    ----------
    persistence_diagrams : List[npt.NDArray]
        List of persistence diagrams, where the first diagram is the original
        (unmodified) diagram, and subsequent diagrams correspond to node removals.
    output_directory : Union[str, Path], optional
        Directory to save the distance results (default: "./").
    metrics : List[str], optional
        Distance metrics to compute, e.g., ["wasserstein", "bottleneck"]
        (default: ["wasserstein", "bottleneck"]).
    return_data : bool, optional
        If True, return the DataFrame of distances instead of saving it
        (default: False).
    output_filename : Optional[str], optional
        Filename for the output file (without extension). If None, defaults to
        "node_removal_distances" (default: None).
    verbose : bool, optional
        If True, print progress messages (default: True).
    save_format : str, optional
        Format for saving results, either "csv", "npy", or "txt" (default: "csv").
    pairwise_distances_kwargs : Optional[dict], optional
        Additional arguments to pass to `diagram_distances`, e.g., {"order": None, "n_jobs": -1}
        (default: {"order": None, "n_jobs": -1}).

    Returns
    -------
    Union[str, pd.DataFrame]
        If `return_data=False`, returns the path to the saved file as a string.
        If `return_data=True`, returns a pandas DataFrame containing distances
        with columns for each metric and homology dimension, indexed by removed node.

    Raises
    ------
    ValueError
        If `persistence_diagrams` has fewer than two diagrams, or if `save_format`
        is not "csv", "npy", or "txt".
    TypeError
        If `persistence_diagrams` contains non-NumPy arrays.

    Notes
    -----
    - Assumes `persistence_diagrams[0]` is the original diagram, and
      `persistence_diagrams[1:]` correspond to node removals in order.
    - The function relies on `diagram_distances` and `save_tda_results` from the
      `neurophorm.persistence` module.
    - Distances are computed for each homology dimension present in the diagrams.

    Examples
    --------
    >>> import numpy as np
    >>> diagrams = [np.array([[0, 1, 0], [1, 2, 0]]), np.array([[0, 1, 0], [1, 3, 0]])]
    >>> df = node_removal_differences(diagrams, return_data=True)
    >>> print(df)
       wasserstein_H0
    0             1.0
    """
    if len(persistence_diagrams) < 2:
        raise ValueError("At least two persistence diagrams required (original + one removal)")
    if not all(isinstance(d, np.ndarray) for d in persistence_diagrams):
        raise TypeError("All persistence diagrams must be NumPy arrays")
    valid_formats = {"csv", "npy", "txt"}
    if save_format not in valid_formats:
        raise ValueError(f"save_format must be one of {valid_formats}")

    pairwise_distances_kwargs = pairwise_distances_kwargs or {"order": None, "n_jobs": -1}
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Determine homology dimensions from the original diagram
    homology_dimensions = list(range(int(np.max(persistence_diagrams[0][:, 2])) + 1))

    all_distances = []
    for idx in range(1, len(persistence_diagrams)):
        distances_for_node = {}
        distance_results = diagram_distances(
            [persistence_diagrams[0], persistence_diagrams[idx]],
            metrics=metrics,
            **pairwise_distances_kwargs
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
    df_distances.index += 1  # Start indexing at 1 for removed nodes

    if output_filename is None:
        output_filename = "node_removal_distances"
    output_filename = output_filename.rsplit(".", 1)[0]  # Strip existing extension
    output_filepath = output_directory / f"{output_filename}.{save_format}"

    if return_data:
        return df_distances
    else:
        save_tda_results(
            {str(output_filepath): df_distances},
            format=save_format,
            overwrite=True
        )
        if verbose:
            print(f"Saved distances for {len(all_distances)} nodes to {output_filepath}")
        return str(output_filepath)
    
def load_removal_data(
    output_directory: Union[str, Path],
    atlas: npt.NDArray[np.int_],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load node removal data from disk, compute mean and error values grouped by atlas labels.

    This function searches for files that start with 'node_removal_distances_' in the specified directory,
    infers the file format (csv, npy, or txt), loads the data, and computes the mean and normalized standard
    deviation of distances for each atlas region.

    Parameters
    ----------
    output_directory : Union[str, Path]
        The directory containing the node removal data files.
    atlas : np.ndarray
        A 1D array of length N (number of nodes), where each entry is the atlas label of a node.

    Returns
    -------
    mean_df : pd.DataFrame
        A DataFrame containing the mean summed distances for each atlas label.
    error_df : pd.DataFrame
        A DataFrame containing the standard deviation of the summed distances divided by the square root of the 
        number of nodes per atlas label (standard error).
    """
    supported_formats = {".csv", ".npy", ".txt"}
    output_path = Path(output_directory)
    all_data = {}

    def detect_format(filename: str) -> Optional[str]:
        """Detect the file format from its extension."""
        for ext in supported_formats:
            if filename.endswith(ext):
                return ext.lstrip(".")
        return None

    def load_array(file_path: Path, file_format: str) -> np.ndarray:
        """Load an array from a file based on its format."""
        try:
            if file_format == "csv":
                return pd.read_csv(file_path).to_numpy()
            elif file_format == "npy":
                return np.load(file_path)
            elif file_format == "txt":
                return np.loadtxt(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
        except Exception as e:
            raise ValueError(f"Failed to load {file_path}: {str(e)}")

    for file in os.listdir(output_path):
        file_path = output_path / file
        file_format = detect_format(file)
        if not file_format:
            continue

        name = file.replace("_node_removal_distances", "").replace(f".{file_format}", "")
        data = load_array(file_path, file_format)
        all_data[name] = data.sum(axis=1)

    df = pd.DataFrame(all_data, index=atlas)
    mean_df = df.groupby(df.index).mean()
    std_df = df.groupby(df.index).std()
    sqrt_counts = np.sqrt(pd.Series(atlas).value_counts())
    error_df = std_df.div(sqrt_counts, axis=0)

    return mean_df, error_df

