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
from collections import defaultdict

from neurophorm.persistence import (
    rips_persistence_diagrams,
    diagram_distances,
    save_tda_results,
    _detect_format,
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


def _load_array(file_path: Path, file_format: str, expected_rows: int) -> np.ndarray:
    """
    Load array from file based on format, dynamically handling potential header row.
    """
    if file_format == "npy":
        data = np.load(file_path)
        if data.shape[0] != expected_rows:
            raise ValueError(f"NPY file {file_path} has {data.shape[0]} rows, expected {expected_rows}")
        return data
    elif file_format in {"csv", "txt"}:
        # Load without skipping header first
        if file_format == "csv":
            df = pd.read_csv(file_path, header=None)
        else:  # txt
            df = pd.read_csv(file_path, sep='\s+', header=None)  # Assume whitespace separated for TXT
        data = df.values
        if data.shape[0] == expected_rows:
            # Exact match, no header
            if data.shape[1] == 0:
                raise ValueError(f"File {file_path} has no columns after loading")
            return data
        elif data.shape[0] == expected_rows + 1:
            # Likely has header, skip first row
            data_no_header = data[1:, :]
            if data_no_header.shape[1] == 0:
                raise ValueError(f"File {file_path} has no data columns after skipping header")
            return data_no_header
        else:
            raise ValueError(f"File {file_path} has {data.shape[0]} rows, expected {expected_rows} or {expected_rows + 1}")
    raise ValueError(f"Unsupported format: {file_format}")


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
    per_subject: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load node-removal distance files and aggregate mean and standard error by atlas label.

    This function scans `output_directory` recursively for files with extensions among {csv, npy, txt},
    loads each as a 2D array where rows correspond to removed nodes (1..n), and computes:
        1) Sum across columns per removed node (collapsing metrics/dims into a single score)
        2) Group-wise mean and standard error according to `atlas` labels.

    If the directory contains subfolders (assumed to be group folders with subject subfolders),
    it processes files as in the original function, using identifiers like "{group}_{subject_no}".
    If the directory directly contains files (no subfolders), it uses the immediate folder name
    as the identifier for each file, ignoring group/subject extraction.

    Parameters
    ----------
    output_directory : str or pathlib.Path
        Directory containing saved node-removal distance files (recursively searched if subfolders exist).
        Each file should be a rectangular numeric table with shape (n_nodes, n_features).
    atlas : numpy.ndarray of int
        A 1D array of length n_nodes, giving an integer atlas/region label for each node.
    per_subject : bool, default True
        If True, return per-subject DataFrames with columns named either "{group}_{subject_no}"
        (if subfolders exist) or "{folder_name}" (if files are directly in output_directory).
        If False, return per-group aggregated DataFrames with columns named "{group}" or "{folder_name}".

    Returns
    -------
    mean_df : pandas.DataFrame
        DataFrame indexed by atlas label with means per label.
    error_df : pandas.DataFrame
        DataFrame indexed by atlas label with standard errors per label.

    Raises
    ------
    ValueError
        If a file cannot be loaded, has an unsupported extension, or has shape mismatch with atlas.
    FileNotFoundError
        If `output_directory` does not exist or contains no supported files.

    Notes
    -----
    When files are directly in `output_directory`, the folder name (not filename) is used as the identifier.
    """
    logger.info("load_removal_data: start | out_dir=%s | per_subject=%s", output_directory, per_subject)

    expected_rows = len(atlas)
    supported_formats = {".csv", ".npy", ".txt"}
    output_path = Path(output_directory)
    if not output_path.exists() or not output_path.is_dir():
        logger.error("Directory does not exist or is not a directory: %s", output_path)
        raise FileNotFoundError(f"Directory not found: {output_path}")

    # Check if directory contains subfolders or only files
    contains_subdirs = any(p.is_dir() for p in output_path.iterdir())
    all_file_paths = []
    for ext in supported_formats:
        if contains_subdirs:
            all_file_paths.extend(output_path.rglob(f'*{ext}'))
        else:
            all_file_paths.extend(output_path.glob(f'*{ext}'))
    if not all_file_paths:
        logger.error("No supported files found in %s", output_path)
        raise FileNotFoundError(f"No supported files (.csv/.npy/.txt) found in {output_path}")

    all_sums_per_subject: Dict[str, List[np.ndarray]] = defaultdict(list)
    all_sums_per_group: Dict[str, List[tuple]] = defaultdict(list)

    for file_path in all_file_paths:
        file_format = _detect_format(file_path.parent, "*node_removal_distances")
        if not file_format:
            logger.debug("Skipping file with unknown format: %s", file_path.name)
            continue

        # Determine identifier based on directory structure
        if contains_subdirs:
            # Subfolder case: extract group and subject_no
            try:
                parents = list(file_path.parents)
                if len(parents) < 2:
                    raise ValueError("Insufficient parent directories to extract group")
                group_folder = parents[1].name  # Immediate parent of subject folder
                name = file_path.stem.replace("_node_removal_distances", "")
                subject_no = name.split("_")[-1]  # Last part after _ in filename stem
                identifier = f"{group_folder}_{subject_no}"
                group_key = group_folder
            except (IndexError, ValueError) as e:
                logger.warning("Could not extract group/subject_no for %s: %s, skipping", file_path, e)
                continue
        else:
            # Direct files case: use folder name as identifier
            group_key = output_path.name
            identifier = file_path.stem.replace("_node_removal_distances", "")

        try:
            data = _load_array(file_path, file_format, expected_rows)
        except ValueError as e:
            logger.error("Failed to load array from %s: %s", file_path, e)
            continue

        if data.ndim != 2:
            logger.error("Loaded array must be 2D | file=%s | shape=%s", file_path, data.shape)
            continue

        # Collapse per-node across features via sum
        sums = data.sum(axis=1)
        all_sums_per_subject[identifier].append(sums)
        all_sums_per_group[group_key].append((identifier, sums))

    if not all_sums_per_subject:
        logger.error("No valid data loaded from %s", output_path)
        raise FileNotFoundError(f"No valid node removal data found in {output_path}")

    if len(atlas.shape) != 1:
        logger.error("atlas must be 1D | shape=%s", atlas.shape)
        raise ValueError("atlas must be a 1D array of integer labels")

    unique_labels = sorted(np.unique(atlas))

    if per_subject:
        # Per subject: compute mean and SE across nodes (pooling files per identifier)
        mean_dict: Dict[str, List[float]] = {}
        error_dict: Dict[str, List[float]] = {}
        for identifier, sums_list in all_sums_per_subject.items():
            if not sums_list:
                continue
            means = []
            ses = []
            for label in unique_labels:
                node_mask = (atlas == label)
                all_values = np.concatenate([s[node_mask] for s in sums_list])
                n_obs = len(all_values)
                if n_obs == 0:
                    mean_val = np.nan
                    se_val = np.nan
                else:
                    mean_val = np.mean(all_values)
                    std_val = np.std(all_values, ddof=1)
                    se_val = std_val / np.sqrt(n_obs) if n_obs > 1 else 0.0
                means.append(mean_val)
                ses.append(se_val)
            mean_dict[identifier] = means
            error_dict[identifier] = ses
        mean_df = pd.DataFrame(mean_dict, index=unique_labels)
        error_df = pd.DataFrame(error_dict, index=unique_labels)
    else:
        # Per group: compute per-identifier means per label, then aggregate across identifiers per group
        mean_dict = {}
        error_dict = {}
        for group_key, subject_data_list in all_sums_per_group.items():
            if not subject_data_list:
                continue
            # Collect per-identifier sums for this group
            group_subject_sums: Dict[str, List[np.ndarray]] = defaultdict(list)
            for identifier, sums in subject_data_list:
                group_subject_sums[identifier].append(sums)

            # Compute identifier means for this group
            group_subject_means: Dict[int, List[float]] = defaultdict(list)
            for identifier, sums_list in group_subject_sums.items():
                if not sums_list:
                    continue
                for label in unique_labels:
                    node_mask = (atlas == label)
                    all_values = np.concatenate([s[node_mask] for s in sums_list])
                    n_obs = len(all_values)
                    if n_obs > 0:
                        subject_mean = np.mean(all_values)
                        group_subject_means[label].append(subject_mean)

            # Aggregate across identifiers for this group
            means = []
            ses = []
            for label in unique_labels:
                subject_means = group_subject_means[label]
                n_subjects = len(subject_means)
                if n_subjects == 0:
                    overall_mean = np.nan
                    overall_se = np.nan
                else:
                    overall_mean = np.mean(subject_means)
                    if n_subjects > 1:
                        overall_std = np.std(subject_means, ddof=1)
                        overall_se = overall_std / np.sqrt(n_subjects)
                    else:
                        overall_se = 0.0
                means.append(overall_mean)
                ses.append(overall_se)
            mean_dict[group_key] = means
            error_dict[group_key] = ses

        mean_df = pd.DataFrame(mean_dict, index=unique_labels)
        error_df = pd.DataFrame(error_dict, index=unique_labels)

    if mean_df.empty:
        logger.error("No data columns after processing in %s", output_path)
        raise FileNotFoundError(f"No valid node removal data columns after processing in {output_path}")

    logger.info("load_removal_data: done | mean_df=%s | error_df=%s",
                mean_df.shape, error_df.shape)
    return mean_df, error_df