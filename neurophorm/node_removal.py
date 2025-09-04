"""
node_removal.py
Utilities for node-removal analysis on persistence diagrams with comprehensive logging.

This module provides:
    1) node_removal_persistence
       Compute persistence diagrams for the original distance matrix and for each
       single-node "removal" (implemented by setting that node's row/column to a
       large value).
    2) node_removal_differences
       Compute pairwise distances (e.g., Wasserstein, bottleneck) between the
       original diagram and each node-removed diagram.
    3) load_removal_data
       Load saved node-removal distance results and aggregate by atlas labels
       to obtain mean and standard error per region.

Dependencies
    numpy, pandas
    neurophorm.persistence: rips_persistence_diagrams, diagram_distances, save_tda_results
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Union, Tuple, Dict

import numpy as np
import numpy.typing as npt
import pandas as pd

from neurophorm.persistence import (
    rips_persistence_diagrams,
    diagram_distances,
    save_tda_results,
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
# Default level; the host application can override this:
logger.setLevel(logging.INFO)


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
    Compute persistence diagrams for the original matrix and for each single-node removal.

    A node "removal" is simulated by setting the node's entire row and column to a large
    value `infinity` while keeping zeros on the diagonal. The function returns all diagrams
    (original first) or saves them to disk.

    Parameters
    ----------
    distance_matrix : numpy.ndarray
        Square distance matrix of shape (n_nodes, n_nodes) with non-negative entries.
    output_directory : str or pathlib.Path, default "./"
        Directory where results are written when `return_data=False`.
    mode : {"sparse", None}, optional
        Persistence backend. None uses dense Vietoris–Rips; "sparse" uses Sparse Rips.
    infinity : float, default 10.0
        Large distance to assign to the removed node's row/column.
    verbose : bool, default True
        If True, key progress messages are logged at INFO; otherwise at DEBUG.
    return_data : bool, default False
        If True, return a list of diagrams instead of saving to disk.
    save_format : {"csv", "npy", "txt"}, default "csv"
        File format used when saving numeric arrays.
    persistence_diagrams_kwargs : dict, optional
        Extra keyword arguments forwarded to `rips_persistence_diagrams`, e.g.
        homology_dimensions=(0,1), n_jobs=-1, max_edge_length=...

    Returns
    -------
    str or List[numpy.ndarray]
        If `return_data=True`, a list of length (n_nodes + 1) containing persistence
        diagrams as (m_i, 3) arrays [birth, death, dim], with the original diagram at
        index 0. Otherwise, the path to the "persistence_diagrams" folder is returned.

    Raises
    ------
    TypeError
        If `distance_matrix` is not a NumPy array.
    ValueError
        If the matrix is not square or `mode` is not one of {None, "sparse"}.

    Notes
    -----
    The first diagram corresponds to the unmodified input matrix. Diagrams 1..n correspond
    to removal of node indices 0..n-1, respectively.

    Examples
    --------
    >>> import numpy as np
    >>> dm = np.array([[0, 1, 2],
    ...                [1, 0, 3],
    ...                [2, 3, 0]], dtype=float)
    >>> res = node_removal_persistence(dm, return_data=True, mode=None)
    >>> isinstance(res, list) and len(res) == 4
    True
    """
    log = logger.info if verbose else logger.debug
    log("node_removal_persistence: start | shape=%s | mode=%s | infinity=%.3f | return_data=%s",
        getattr(distance_matrix, "shape", None), mode, infinity, return_data)

    valid_modes = {None, "sparse"}
    if mode not in valid_modes:
        logger.error("Invalid mode: %s", mode)
        raise ValueError(f"Mode must be one of {valid_modes}")

    if not isinstance(distance_matrix, np.ndarray):
        logger.error("distance_matrix must be a numpy.ndarray")
        raise TypeError("distance_matrix must be a NumPy array")
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        logger.error("distance_matrix must be square | shape=%s", distance_matrix.shape)
        raise ValueError("distance_matrix must be a square 2D array")

    persistence_diagrams_kwargs = persistence_diagrams_kwargs or {}

    output_directory = Path(output_directory)
    if not return_data:
        output_directory.mkdir(parents=True, exist_ok=True)

    n = distance_matrix.shape[0]
    if n == 0:
        log("node_removal_persistence: empty matrix")
        return "" if not return_data else []

    # Build modified matrices
    modified_distances: List[np.ndarray] = [distance_matrix]  # original first
    for node_to_remove in range(n):
        md = distance_matrix.copy()
        md[node_to_remove, :] = infinity
        md[:, node_to_remove] = infinity
        np.fill_diagonal(md, 0.0)
        modified_distances.append(md)
    log("node_removal_persistence: built %d modified matrices (incl. original)", len(modified_distances))

    # Compute diagrams
    all_diagrams = rips_persistence_diagrams(
        modified_distances,
        mode=mode,
        **persistence_diagrams_kwargs,
    )
    log("node_removal_persistence: computed %d diagrams", len(all_diagrams))

    if return_data:
        log("node_removal_persistence: done | returning diagrams")
        return all_diagrams

    # Save results
    persistence_folder = output_directory / "persistence_diagrams"
    persistence_folder.mkdir(exist_ok=True)
    results: Dict[str, np.ndarray] = {}
    for i, diagram in enumerate(all_diagrams):
        # file stem without suffix; save_tda_results will add suffix
        stem = "persistence_diagrams_original" if i == 0 else f"persistence_diagrams_node_{i:03d}"
        results[str(persistence_folder / stem)] = diagram

    save_tda_results(results, format=save_format, overwrite=True)
    log("node_removal_persistence: saved to %s", persistence_folder)
    return str(persistence_folder)


def node_removal_differences(
    persistence_diagrams: List[npt.NDArray],
    output_directory: Union[str, Path] = "./",
    metrics: List[str] = ("wasserstein", "bottleneck"),
    return_data: bool = False,
    output_filename: Optional[str] = None,
    verbose: bool = True,
    save_format: str = "csv",
    pairwise_distances_kwargs: Optional[dict] = None,
) -> Union[str, pd.DataFrame]:
    """
    Compute distances from the original diagram to each node-removed diagram.

    For each node i, compute diagram distances between the original diagram
    (index 0) and the diagram with node i removed (index i+1), for each metric
    and each homology dimension present.

    Parameters
    ----------
    persistence_diagrams : List[numpy.ndarray]
        List of persistence diagrams with the original diagram at index 0, and
        node-removed diagrams at indices 1..n (one per removed node). Each diagram
        has shape (m_i, 3) with columns [birth, death, dim].
    output_directory : str or pathlib.Path, default "./"
        Directory where results are written if `return_data=False`.
    metrics : Iterable[str], default ("wasserstein", "bottleneck")
        Diagram distance metrics to compute.
    return_data : bool, default False
        If True, return a pandas DataFrame instead of saving to disk.
    output_filename : str, optional
        Stem of the output file. Defaults to "node_removal_distances".
    verbose : bool, default True
        If True, key progress messages are logged at INFO; otherwise at DEBUG.
    save_format : {"csv", "npy", "txt"}, default "csv"
        File format for saved results.
    pairwise_distances_kwargs : dict, optional
        Extra keyword arguments forwarded to `diagram_distances`, e.g. order=1, n_jobs=-1.

    Returns
    -------
    str or pandas.DataFrame
        If `return_data=True`, a DataFrame indexed by removed node (1..n)
        with columns like "wasserstein_H0" etc. Otherwise the path to the saved file.

    Raises
    ------
    TypeError
        If any entry in `persistence_diagrams` is not a NumPy array.
    ValueError
        If fewer than two diagrams are provided or `save_format` unsupported.

    Notes
    -----
    The function detects homology dimensions from the original diagram (index 0).

    Examples
    --------
    >>> import numpy as np
    >>> PD0 = np.array([[0.0, 1.0, 0], [0.2, 0.9, 1]])
    >>> PD1 = np.array([[0.0, 1.1, 0], [0.3, 1.0, 1]])
    >>> df = node_removal_differences([PD0, PD1], return_data=True)
    >>> isinstance(df, pd.DataFrame)
    True
    """
    log = logger.info if verbose else logger.debug
    log("node_removal_differences: start | n_diagrams=%d | metrics=%s | return_data=%s",
        len(persistence_diagrams), metrics, return_data)

    if len(persistence_diagrams) < 2:
        logger.error("At least two diagrams required (original + one removal)")
        raise ValueError("At least two persistence diagrams required (original + one removal)")
    if not all(isinstance(d, np.ndarray) for d in persistence_diagrams):
        logger.error("All persistence diagrams must be numpy arrays")
        raise TypeError("All persistence diagrams must be NumPy arrays")

    valid_formats = {"csv", "npy", "txt"}
    if save_format not in valid_formats:
        logger.error("Unsupported save_format: %s", save_format)
        raise ValueError(f"save_format must be one of {valid_formats}")

    pairwise_distances_kwargs = pairwise_distances_kwargs or {"order": None, "n_jobs": -1}

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Homology dimensions present in the original diagram
    if persistence_diagrams[0].size == 0:
        homology_dimensions = []
    else:
        homology_dimensions = list(range(int(np.max(persistence_diagrams[0][:, 2])) + 1))
    log("node_removal_differences: homology_dims=%s", homology_dimensions)

    all_distances: List[Dict[str, float]] = []
    for idx in range(1, len(persistence_diagrams)):
        distance_results = diagram_distances(
            [persistence_diagrams[0], persistence_diagrams[idx]],
            metrics=list(metrics),
            **pairwise_distances_kwargs,
        )
        log("node_removal_differences: computed distances for removal %d", idx)

        distances_for_node: Dict[str, float] = {}
        for metric in metrics:
            for dim in homology_dimensions:
                key = f"{metric}_distance_H{dim}"
                if key in distance_results:
                    distances_for_node[f"{metric}_H{dim}"] = float(distance_results[key][0, 1])
        all_distances.append(distances_for_node)

    df_distances = pd.DataFrame(all_distances)
    df_distances.index.name = "Removed Node"
    df_distances.index += 1  # human-friendly indexing (1..n)

    if return_data:
        log("node_removal_differences: done | returning DataFrame shape=%s", df_distances.shape)
        return df_distances

    if output_filename is None:
        output_filename = "node_removal_distances"
    output_filename = output_filename.rsplit(".", 1)[0]  # strip extension if any
    output_filepath = output_directory / f"{output_filename}.{save_format}"

    # Persist via neurophorm.persistence.save_tda_results; ensure numpy array/tabular
    payload: Dict[str, np.ndarray] = {str(output_filepath): df_distances.to_numpy()}
    save_tda_results(payload, format=save_format, overwrite=True)
    log("node_removal_differences: saved distances for %d removals to %s",
        len(all_distances), output_filepath)
    return str(output_filepath)


def load_removal_data(
    output_directory: Union[str, Path],
    atlas: npt.NDArray[np.int_],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load node-removal distance files and aggregate mean and standard error by atlas label.

    This function scans `output_directory` for files with extensions among {csv, npy, txt},
    loads each as a 2D array where rows correspond to removed nodes (1..n), and computes:
        1) Sum across columns per removed node (collapsing metrics/dims into a single score)
        2) Group-wise mean and standard error (std / sqrt(N)) according to `atlas` labels

    Parameters
    ----------
    output_directory : str or pathlib.Path
        Directory containing saved node-removal distance files. Each file should be a
        rectangular numeric table with shape (n_nodes, n_features).
    atlas : numpy.ndarray of int
        A 1D array of length n_nodes, giving an integer atlas/region label for each node.
        These labels are used to group nodes when computing mean and standard error.

    Returns
    -------
    mean_df : pandas.DataFrame
        DataFrame indexed by atlas label with the mean of the per-node sums for each file.
        Columns correspond to the discovered files (stem names).
    error_df : pandas.DataFrame
        DataFrame indexed by atlas label with the standard error (std / sqrt(count))
        of the per-node sums for each file.

    Raises
    ------
    ValueError
        If a file cannot be loaded or has an unsupported extension.
    FileNotFoundError
        If `output_directory` does not exist or contains no supported files.

    Notes
    -----
    File names are used to generate column labels by removing the trailing
    "_node_removal_distances" piece (if present) and the extension.

    Examples
    --------
    >>> import numpy as np, pandas as pd, os, tempfile
    >>> tmp = tempfile.mkdtemp()
    >>> # Simulate a CSV with 4 removed nodes and 3 metrics
    >>> arr = np.random.rand(4, 3)
    >>> pd.DataFrame(arr).to_csv(os.path.join(tmp, "example_node_removal_distances.csv"), index=False)
    >>> atlas = np.array([1, 1, 2, 2], dtype=int)
    >>> mean_df, err_df = load_removal_data(tmp, atlas)
    >>> set(mean_df.index) == {1, 2} and mean_df.shape[1] == 1
    True
    """
    logger.info("load_removal_data: start | out_dir=%s", output_directory)

    supported_formats = {".csv", ".npy", ".txt"}
    output_path = Path(output_directory)
    if not output_path.exists() or not output_path.is_dir():
        logger.error("Directory does not exist or is not a directory: %s", output_path)
        raise FileNotFoundError(f"Directory not found: {output_path}")

    files = [f for f in os.listdir(output_path) if Path(f).suffix in supported_formats]
    if not files:
        logger.error("No supported files found in %s", output_path)
        raise FileNotFoundError(f"No supported files (.csv/.npy/.txt) found in {output_path}")

    all_data: Dict[str, np.ndarray] = {}

    def detect_format(filename: str) -> Optional[str]:
        """Detect the file format from its extension (csv, npy, or txt)."""
        for ext in supported_formats:
            if filename.endswith(ext):
                return ext.lstrip(".")
        return None

    def load_array(file_path: Path, file_format: str) -> np.ndarray:
        """Load an array from a file, normalizing to a float NumPy array."""
        try:
            if file_format == "csv":
                return pd.read_csv(file_path).to_numpy(dtype=float)
            if file_format == "npy":
                return np.load(file_path).astype(float)
            if file_format == "txt":
                return np.loadtxt(file_path).astype(float)
        except Exception as e:
            logger.exception("Failed to load %s | %s", file_path, e)
            raise ValueError(f"Failed to load {file_path}: {str(e)}")
        raise ValueError(f"Unsupported file format: {file_format}")

    for file in files:
        file_path = output_path / file
        file_format = detect_format(file)
        if not file_format:
            logger.debug("Skipping file with unknown format: %s", file)
            continue

        # Normalize column name (remove suffix and extension)
        name = file.replace("_node_removal_distances", "")
        name = name[: -(len(file_path.suffix))] if file_path.suffix else name

        data = load_array(file_path, file_format)
        if data.ndim != 2:
            logger.error("Loaded array must be 2D | file=%s | shape=%s", file_path, data.shape)
            raise ValueError(f"File {file_path} does not contain a 2D array")

        # Collapse per-node across features via sum
        all_data[name] = data.sum(axis=1)

    if not all_data:
        logger.error("No valid data loaded from %s", output_path)
        raise FileNotFoundError(f"No valid node removal data found in {output_path}")

    if len(atlas.shape) != 1:
        logger.error("atlas must be 1D | shape=%s", atlas.shape)
        raise ValueError("atlas must be a 1D array of integer labels")

    df = pd.DataFrame(all_data, index=atlas)
    mean_df = df.groupby(df.index).mean()
    std_df = df.groupby(df.index).std()

    # Standard error = std / sqrt(count)
    counts = pd.Series(atlas).value_counts().sort_index()
    sqrt_counts = np.sqrt(counts)
    error_df = std_df.div(sqrt_counts, axis=0)

    logger.info("load_removal_data: done | mean_df=%s | error_df=%s",
                mean_df.shape, error_df.shape)
    return mean_df, error_df
