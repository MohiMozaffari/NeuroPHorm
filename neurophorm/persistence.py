"""
persistence.py
Topological Data Analysis utilities.

Dependencies
    numpy, pandas, pillow (PIL), matplotlib, scipy, giotto-tda (gtda)
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from PIL import Image
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from gtda.homology import VietorisRipsPersistence, SparseRipsPersistence
from gtda.diagrams import (
    BettiCurve,
    PersistenceEntropy,
    PairwiseDistance,
    Amplitude,
    PersistenceImage,
)

# ---------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "[%(levelname)s] %(asctime)s - %(name)s - %(message)s"
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
# Default level; override from host app as needed
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------
def corr_to_distance_matrices(
    correlation_matrices: List[npt.NDArray],
    mode: Optional[str] = None,
) -> List[npt.NDArray]:
    """
    Convert correlation matrices to distance matrices for TDA.

    This maps correlations in [-1, 1] to non-negative distances according to `mode`
    and zeros the diagonal. Use this before Vietoris–Rips persistence on
    correlation-based networks.

    Parameters
        correlation_matrices
            List of square NumPy arrays of shape (n, n) with entries in [-1, 1].
        mode
            Distance mapping
                None  →  1 - ρ  standard mapping
                "positive"  →  keep positive ρ, set others to a large distance 10
                "negative"  →  keep |negative ρ|, set others to a large distance 10

    Returns
        List[numpy.ndarray]
            Distance matrices with shape (n, n), diagonal set to 0.

    Raises
        ValueError
            If `mode` is not one of {None, "positive", "negative"}.

    Examples
        >>> import numpy as np
        >>> corr = [np.array([[1.0, 0.5], [0.5, 1.0]])]
        >>> dists = corr_to_distance_matrices(corr)
        >>> dists[0].round(3)
        array([[0. , 0.5],
            [0.5, 0. ]])
        >>> dists_pos = corr_to_distance_matrices(corr, mode="positive")
        >>> float(dists_pos[0][0, 1]) == 0.5
        True
    """
    logger.info("corr_to_distance_matrices: start | n=%d | mode=%s",
                len(correlation_matrices), mode)

    valid_modes = {None, "positive", "negative"}
    if mode not in valid_modes:
        logger.error("Invalid mode for corr_to_distance_matrices: %s", mode)
        raise ValueError(f"Mode must be one of {valid_modes}")

    all_distance_matrix: List[npt.NDArray] = []
    for i, data in enumerate(correlation_matrices):
        if mode is None:
            distance = 1 - data
        elif mode == "positive":
            distance = np.where(data > 0, data, 10)
        else:  # mode == "negative"
            distance = np.where(data < 0, np.abs(data), 10)

        np.fill_diagonal(distance, 0)
        all_distance_matrix.append(distance)
        logger.debug("corr_to_distance_matrices: idx=%d | shape=%s", i, distance.shape)

    logger.info("corr_to_distance_matrices: done")
    return all_distance_matrix


def rips_persistence_diagrams(
    distance_matrices: List[npt.NDArray],
    mode: Optional[str] = None,
    **kwargs,
) -> List[npt.NDArray]:
    """
    Compute Vietoris–Rips (or Sparse Rips) persistence diagrams from distance matrices.

    The input must be precomputed distance matrices. Internally dispatches to
    `gtda.homology.VietorisRipsPersistence` or `SparseRipsPersistence` depending on `mode`.

    Parameters
        distance_matrices
            List of square (n, n) distance matrices with non-negative entries.
        mode
            None → dense Vietoris–Rips
            "sparse" → Sparse Rips for large datasets
        **kwargs
            Passed to the Giotto-TDA transformer, e.g.
            homology_dimensions=(0, 1, 2), n_jobs=-1, max_edge_length=..., approx_mode=...

    Returns
        List[numpy.ndarray]
            One diagram per input matrix. Each diagram has shape (m, 3) with columns
            [birth, death, homology_dimension].

    Raises
        ValueError
            If `mode` not in {None, "sparse"} or any matrix is non-square.

    Examples
        >>> import numpy as np
        >>> dm = [np.array([[0.0, 1.0], [1.0, 0.0]])]
        >>> diags = rips_persistence_diagrams(dm, homology_dimensions=(0,))
        >>> isinstance(diags, list) and diags[0].shape[1] == 3
        True
    """

    logger.info(
        "rips_persistence_diagrams: start | n=%d | mode=%s | kwargs=%s",
        len(distance_matrices),
        mode,
        {k: v for k, v in kwargs.items()},
    )

    valid_modes = {None, "sparse"}
    if mode not in valid_modes:
        logger.error("Invalid mode for rips_persistence_diagrams: %s", mode)
        raise ValueError(f"Mode must be one of {valid_modes}")

    # Validate distance matrices
    for i, matrix in enumerate(distance_matrices):
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            logger.error("Non-square distance matrix at index %d with shape %s", i, matrix.shape)
            raise ValueError("Each distance matrix must be square (n_points, n_points)")

    if mode is None:
        vr_persistence = VietorisRipsPersistence(metric="precomputed", **kwargs)
        logger.debug("Using VietorisRipsPersistence")
    else:
        vr_persistence = SparseRipsPersistence(metric="precomputed", **kwargs)
        logger.debug("Using SparseRipsPersistence")

    # gtda expects input as (n_samples, n_points, n_points)
    diagrams = vr_persistence.fit_transform(np.array(distance_matrices))
    out = [diagrams[i] for i in range(len(distance_matrices))]

    logger.info("rips_persistence_diagrams: done | produced=%d", len(out))
    return out


def betti_curves(
    persistence_diagrams: List[npt.NDArray],
    **kwargs,
) -> Tuple[List[npt.NDArray], npt.NDArray]:
    """
    Compute Betti curves from persistence diagrams and return curves plus x-axis.

    Uses `gtda.diagrams.BettiCurve`. The returned Betti array is aligned with the
    homology dimensions configured in the transformer. The x-axis grids are extracted
    from the transformer’s plot method, deduplicated across homology dimensions, and
    stacked into a strict 2D numeric array.

    Parameters
    ----------
    persistence_diagrams : list of numpy.ndarray
        Each array has shape (m, 3) with [birth, death, dim].
    **kwargs :
        Passed to `BettiCurve`, e.g. n_bins=200, sampling=..., n_jobs=-1.

    Returns
    -------
    betti_numbers : list of numpy.ndarray
        A list of arrays of shape (n_homology_dims, n_bins) for each input diagram.
        The order of dimensions corresponds to the homology dimensions provided to
        the transformer.
    x_data : numpy.ndarray
        Array of shape (n_homology_dims, n_bins) containing the x-values
        (filtration grid) used for each homology dimension. Guaranteed to be
        numeric (dtype=float). This replaces the earlier object-dtype return,
        which caused downstream plotting errors.

    Raises
    ------
    ValueError
        If the extracted x-grids across dimensions have inconsistent lengths.

    Examples
    --------
    >>> import numpy as np
    >>> # Fake diagram with two features in H0 and H1
    >>> PD = [np.array([[0.0, 0.5, 0], [0.2, 0.8, 1]])]
    >>> curves, xvals = betti_curves(PD, n_bins=10)
    >>> curves[0].shape  # (n_dims, n_bins)
    (2, 10)
    >>> xvals.shape  # (n_dims, n_bins)
    (2, 10)
    """


    logger.info(
        "betti_curves: start | n=%d | kwargs=%s",
        len(persistence_diagrams),
        {k: v for k, v in kwargs.items()},
    )
    betti_curve = BettiCurve(**kwargs)
    betti_numbers = betti_curve.fit_transform(persistence_diagrams)
    fig = betti_curve.plot(persistence_diagrams)

    # Collect x arrays from traces, coerce to float 1D
    x_traces = []
    for tr in fig.data:
        xa = np.asarray(tr.x, dtype=float).ravel()
        x_traces.append(xa)

    # Keep unique x grids (plot may emit multiple traces per dim)
    unique_x = []
    for xa in x_traces:
        if not any(np.array_equal(xa, u) for u in unique_x):
            unique_x.append(xa)

    # Sanity check: all x grids must have same length
    n_bins_set = {len(u) for u in unique_x}
    if len(n_bins_set) != 1:
        raise ValueError(
            f"Inconsistent Betti x-grid lengths across dimensions: {sorted(n_bins_set)}"
        )

    # Final numeric 2D array: shape (n_dims, n_bins)
    x_data = np.vstack(unique_x)

    logger.debug(
        "betti_curves: shapes | betti_numbers=%s | x_data_shape=%s",
        getattr(betti_numbers, "shape", None),
        x_data.shape,
    )

    logger.info("betti_curves: done")
    return betti_numbers, x_data


def persistence_entropy(
    persistence_diagrams: List[npt.NDArray],
    **kwargs,
) -> Dict[str, npt.NDArray]:
    """
    Compute persistence entropy as a scalar summary per homology dimension.

    Parameters
        persistence_diagrams
            List of diagrams, each with shape (m, 3) [birth, death, dim].
        **kwargs
            Passed to `PersistenceEntropy`, e.g. normalize=True, n_jobs=-1.

    Returns
        Dict[str, numpy.ndarray]
            {"persistence_entropy": E} where E has shape (n_samples, n_homology_dims).

    Examples
        >>> import numpy as np
        >>> PD = [np.array([[0.0, 0.5, 0], [0.2, 0.8, 1]])]
        >>> out = persistence_entropy(PD, normalize=True)
        >>> "persistence_entropy" in out and out["persistence_entropy"].ndim == 2
        True
    """

    logger.info(
        "persistence_entropy: start | n=%d | kwargs=%s",
        len(persistence_diagrams),
        {k: v for k, v in kwargs.items()},
    )
    pe = PersistenceEntropy(**kwargs)
    entropy = pe.fit_transform(np.array(persistence_diagrams))
    logger.debug("persistence_entropy: shape=%s", getattr(entropy, "shape", None))
    logger.info("persistence_entropy: done")
    return {"persistence_entropy": entropy}


def diagram_distances(
    persistence_diagrams: List[npt.NDArray],
    metrics: List[str] = ("wasserstein", "bottleneck"),
    **kwargs,
) -> Dict[str, npt.NDArray]:
    """
    Compute pairwise distances between persistence diagrams for given metrics.

    Uses `gtda.diagrams.PairwiseDistance`. Produces a square matrix per metric and
    per homology dimension.

    Parameters
        persistence_diagrams
            List of diagrams, each (m, 3) with [birth, death, dim].
        metrics
            Iterable of metrics to compute, e.g. ("wasserstein", "bottleneck").
        **kwargs
            Passed to `PairwiseDistance`, e.g. order=1, n_jobs=-1.

    Returns
        Dict[str, numpy.ndarray]
            Keys in the form "{metric}_distance_H{dim}" each mapping to an array with
            shape (n_samples, n_samples).

    Examples
        >>> import numpy as np
        >>> PD = [np.array([[0.0, 0.5, 0]]), np.array([[0.0, 0.6, 0]])]
        >>> D = diagram_distances(PD, metrics=("wasserstein",))
        >>> all(k.startswith("wasserstein_distance_H") for k in D.keys())
        True
        >>> next(iter(D.values())).shape  # 2x2 matrix
        (2, 2)
    """

    logger.info(
        "diagram_distances: start | n=%d | metrics=%s | kwargs=%s",
        len(persistence_diagrams),
        metrics,
        {k: v for k, v in kwargs.items()},
    )
    distances_dict: Dict[str, npt.NDArray] = {}
    diagrams = np.array(persistence_diagrams)

    for metric in metrics:
        logger.debug("diagram_distances: computing metric=%s", metric)
        pd_calc = PairwiseDistance(metric=metric, **kwargs)
        distances = pd_calc.fit_transform(diagrams)  # (n, n, n_homology_dims)
        for dim in range(distances.shape[2]):
            key = f"{metric}_distance_H{dim}"
            distances_dict[key] = distances[:, :, dim]
            logger.debug("diagram_distances: %s shape=%s", key, distances_dict[key].shape)

    logger.info("diagram_distances: done")
    return distances_dict


def diagram_amplitudes(
    persistence_diagrams: List[npt.NDArray],
    metrics: List[str] = ("wasserstein", "bottleneck"),
    **kwargs,
) -> Dict[str, npt.NDArray]:
    """
    Compute amplitude vectors of persistence diagrams relative to the empty diagram.

    Uses `gtda.diagrams.Amplitude` and returns one vector per homology dimension.

    Parameters
        persistence_diagrams
            List of diagrams, each (m, 3) with [birth, death, dim].
        metrics
            Iterable of metric names, e.g. ("wasserstein", "bottleneck").
        **kwargs
            Passed to `Amplitude`, e.g. order=1, n_jobs=-1.

    Returns
        Dict[str, numpy.ndarray]
            Keys like "{metric}_amplitude" mapping to arrays of shape
            (n_samples, n_homology_dims).

    Examples
        >>> import numpy as np
        >>> PD = [np.array([[0.0, 0.5, 0]]), np.array([[0.0, 0.6, 0]])]
        >>> A = diagram_amplitudes(PD, metrics=("bottleneck",))
        >>> list(A.keys()) == ["bottleneck_amplitude"] and A["bottleneck_amplitude"].ndim == 2
        True
    """

    logger.info(
        "diagram_amplitudes: start | n=%d | metrics=%s | kwargs=%s",
        len(persistence_diagrams),
        metrics,
        {k: v for k, v in kwargs.items()},
    )
    amplitudes_dict: Dict[str, npt.NDArray] = {}
    diagrams = np.array(persistence_diagrams)

    for metric in metrics:
        logger.debug("diagram_amplitudes: computing metric=%s", metric)
        amplitude = Amplitude(metric=metric, **kwargs)
        amplitudes = amplitude.fit_transform(diagrams)  # (n, n_homology_dims)
        key = f"{metric}_amplitude"
        amplitudes_dict[key] = amplitudes
        logger.debug("diagram_amplitudes: %s shape=%s", key, amplitudes.shape)

    logger.info("diagram_amplitudes: done")
    return amplitudes_dict


def persistence_images(
    persistence_diagrams: List[npt.NDArray],
    **kwargs,
) -> List[npt.NDArray]:
    """
    Rasterize persistence diagrams into persistence images.

    Uses `gtda.diagrams.PersistenceImage` to map persistence points onto a
    Gaussian-smeared grid per homology dimension.

    Parameters
        persistence_diagrams
            List of diagrams, each (m, 3) with [birth, death, dim].
        **kwargs
            Passed to `PersistenceImage`, e.g. n_bins=64, sigma=0.1, n_jobs=-1,
            weight_function=callable.

    Returns
        List[numpy.ndarray]
            One image stack per input, each with shape (n_homology_dims, n_bins, n_bins).

    Examples
        >>> import numpy as np
        >>> PD = [np.array([[0.0, 0.5, 0]])]
        >>> imgs = persistence_images(PD, n_bins=8, sigma=0.05)
        >>> isinstance(imgs, list) and imgs[0].ndim == 3
        True
    """
    logger.info(
        "persistence_images: start | n=%d | kwargs=%s",
        len(persistence_diagrams),
        {k: v for k, v in kwargs.items()},
    )
    pi = PersistenceImage(**kwargs)
    images = pi.fit_transform(np.array(persistence_diagrams))  # (n, n_h, n_bins, n_bins)
    out = [images[i] for i in range(len(persistence_diagrams))]
    logger.debug(
        "persistence_images: produced %d images | example_shape=%s",
        len(out),
        None if not out else out[0].shape,
    )
    logger.info("persistence_images: done")
    return out


def save_tda_results(
    data_dict: Dict[str, npt.NDArray],
    overwrite: bool = True,
    format: str = "csv",
    **kwargs,
) -> None:
    """
    Save TDA results to disk as tables or images with optional overwrite control.

    When saving images the input arrays must be 2D. Parent directories are created
    as needed and file extensions are adjusted to match the chosen `format`.

    Parameters
        data_dict
            Mapping from output base paths to arrays. The appropriate suffix is added.
        overwrite
            If False, existing files are skipped.
        format
            One of {"csv", "npy", "txt", "png", "jpg", "jpeg"}.
        **kwargs
            Extra options passed to writers. For images, forwarded to `matplotlib.pyplot.imsave`,
            e.g. cmap="gray", vmin=..., vmax=..., dpi=...

    Raises
        ValueError
            If `format` is unsupported or an image save is requested with non-2D data.

    Examples
        >>> import numpy as np, tempfile, os
        >>> tmp = tempfile.mkdtemp()
        >>> path = os.path.join(tmp, "results/data")  # suffix added automatically
        >>> save_tda_results({path: np.array([[1, 2], [3, 4]])}, format="csv")
        >>> os.path.exists(path + ".csv")
        True
    """

    logger.info(
        "save_tda_results: start | items=%d | format=%s | overwrite=%s | kwargs=%s",
        len(data_dict),
        format,
        overwrite,
        {k: v for k, v in kwargs.items()},
    )
    supported_formats = {"csv", "npy", "txt", "png", "jpg", "jpeg"}
    fmt = format.lower()

    if fmt not in supported_formats:
        logger.error("Unsupported format requested: %s", fmt)
        raise ValueError(f"Unsupported format '{fmt}'. Supported formats are {supported_formats}")

    for raw_path, data in data_dict.items():
        file_path = Path(raw_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path = file_path.with_suffix(f".{fmt}")

        if not overwrite and file_path.exists():
            logger.info("save_tda_results: skip existing file (overwrite=False) | %s", file_path)
            continue

        try:
            if fmt == "csv":
                pd.DataFrame(data).to_csv(file_path, index=False)
            elif fmt == "npy":
                np.save(file_path, data)
            elif fmt == "txt":
                np.savetxt(file_path, data)
            elif fmt in {"png", "jpg", "jpeg"}:
                # Ensure data is 2D for image saving
                if getattr(data, "ndim", 0) > 2:
                    logger.error("Cannot save as image, data must be 2D | %s", file_path)
                    raise ValueError(f"Cannot save {file_path} as image: data must be 2D")
                plt.imsave(file_path, data, format=fmt, **kwargs)

            logger.info("save_tda_results: saved | %s", file_path)
        except Exception as e:
            logger.exception("save_tda_results: error saving %s | %s", file_path, e)

    logger.info("save_tda_results: done")


def individual_tda_features(
    distance_matrices: List[npt.NDArray],
    name: str,
    output_directory: Union[str, Path] = "./",
    mode: Optional[str] = None,
    return_data: bool = False,
    compute_persistence: bool = True,
    compute_betti: bool = False,
    compute_entropy: bool = False,
    compute_amplitude: bool = False,
    compute_image: bool = False,
    save_format: str = "csv",
    image_save_format: str = "png",
    persistence_diagrams_kwargs: Optional[dict] = None,
    betti_curves_kwargs: Optional[dict] = None,
    persistence_entropy_kwargs: Optional[dict] = None,
    amplitudes_kwargs: Optional[dict] = None,
    persistence_images_kwargs: Optional[dict] = None,
    save_tda_results_kwargs: Optional[dict] = None,
) -> Optional[Dict[str, npt.NDArray]]:
    """
    Compute TDA features per matrix independently and save or return the results.

    For each matrix computes
        persistence diagrams  optional
        Betti curves          optional
        persistence entropy   optional
        amplitudes            optional
        persistence images    optional

    Results are organized in subfolders under `output_directory/name`.

    Parameters
        distance_matrices
            List of square (n, n) distance matrices.
        name
            Dataset label used as a subfolder name.
        output_directory
            Base directory for outputs.
        mode
            Persistence backend None for dense or "sparse" for Sparse Rips.
        return_data
            If True, return a dict of arrays instead of saving to disk.
        compute_persistence, compute_betti, compute_entropy, compute_amplitude, compute_image
            Feature toggles.
        save_format
            File format for tables among {"csv", "npy", "txt"}.
        image_save_format
            File format for images among {"png", "jpg", "jpeg"}.
        *_kwargs
            Passed through to the corresponding computation or saver.

    Returns
        Optional[Dict[str, numpy.ndarray]]
            If `return_data=True`, a flat mapping of output paths to arrays. Otherwise None.

    Raises
        ValueError
            If any matrix is not square.

    Examples
        >>> import numpy as np, tempfile
        >>> dm = [np.array([[0.0, 1.0], [1.0, 0.0]])]
        >>> out = tempfile.mkdtemp()
        >>> _ = individual_tda_features(dm, name="test_ind", output_directory=out,
        ...                             mode=None, compute_image=False, return_data=False)
        >>> import os
        >>> os.path.isdir(os.path.join(out, "test_ind", "persistence_diagrams"))
        True
    """

    logger.info(
        "individual_tda_features: start | n=%d | name=%s | out_dir=%s | mode=%s",
        len(distance_matrices), name, output_directory, mode
    )

    betti_curves_kwargs = betti_curves_kwargs or {"n_bins": 200, "n_jobs": -1}
    persistence_entropy_kwargs = persistence_entropy_kwargs or {"normalize": True, "n_jobs": -1}
    amplitudes_kwargs = amplitudes_kwargs or {"order": None, "n_jobs": -1}
    persistence_images_kwargs = persistence_images_kwargs or {"n_jobs": -1, "n_bins": 200, "sigma": 0.0005}
    save_tda_results_kwargs = save_tda_results_kwargs or {"cmap": "gray", "dpi": 300}

    output_directory = Path(output_directory)
    output_csv_directory = output_directory / name
    output_csv_directory.mkdir(parents=True, exist_ok=True)

    # Initialize folder paths conditionally
    persistence_folder = output_csv_directory / "persistence_diagrams" if compute_persistence else None
    betti_folder = output_csv_directory / "betti_curves" if compute_betti else None
    image_folder = output_csv_directory / "persistence_images" if compute_image else None
    entropy_folder = output_csv_directory / "persistence_entropy" if compute_entropy else None
    amplitude_folder = output_csv_directory / "amplitudes" if compute_amplitude else None

    # Create only required folders
    for folder in [persistence_folder, betti_folder, image_folder, entropy_folder, amplitude_folder]:
        if folder:
            folder.mkdir(exist_ok=True)

    # Initialize result dictionaries
    results: Dict[str, npt.NDArray] = {}
    image_results: Dict[str, npt.NDArray] = {}
    persistence_diagrams_kwargs = persistence_diagrams_kwargs or {}

    for idx, matrix in enumerate(distance_matrices):
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            logger.error("individual_tda_features: non-square matrix at idx=%d | shape=%s", idx, matrix.shape)
            raise ValueError("Each distance matrix must be square (n_points, n_points)")

        logger.debug("individual_tda_features: computing diagram | idx=%d", idx)
        persistence_diagrams = rips_persistence_diagrams([matrix], mode, **persistence_diagrams_kwargs)

        # Save persistence diagrams
        if compute_persistence:
            key = str(persistence_folder / f"persistence_diagram_{idx:03d}")
            results[key] = persistence_diagrams[0]
            logger.debug("individual_tda_features: stored diagram | %s", key)

        # Compute Betti curves
        if compute_betti:
            betti_numbers, x_data = betti_curves(persistence_diagrams, **betti_curves_kwargs)
            key_curve = str(betti_folder / f"betti_curve_{idx:03d}")
            key_x = str(betti_folder / f"betti_x_{idx:03d}")
            results[key_curve] = betti_numbers[0]
            results[key_x] = x_data
            logger.debug("individual_tda_features: stored betti | %s, %s", key_curve, key_x)

        # Compute persistence entropy
        if compute_entropy:
            entropy_results = persistence_entropy(persistence_diagrams, **persistence_entropy_kwargs)
            for filename, data in entropy_results.items():
                key = str(entropy_folder / f"{filename}_{idx:03d}")
                results[key] = data
                logger.debug("individual_tda_features: stored entropy | %s", key)

        # Compute amplitudes
        if compute_amplitude:
            amplitude_results = diagram_amplitudes(
                persistence_diagrams,
                metrics=["wasserstein", "bottleneck"],
                **amplitudes_kwargs,
            )
            for filename, data in amplitude_results.items():
                key = str(amplitude_folder / f"{filename}_{idx:03d}")
                results[key] = data
                logger.debug("individual_tda_features: stored amplitude | %s", key)

        # Compute persistence images
        if compute_image:
            images_results = persistence_images(persistence_diagrams, **persistence_images_kwargs)
            for dim in range(images_results[0].shape[0]):
                save_path = str(image_folder / f"persistence_image_{idx:03d}_H{dim}")
                image_results[save_path] = images_results[0][dim]
                logger.debug("individual_tda_features: stored image | %s", save_path)

    if return_data:
        logger.info("individual_tda_features: done | returning data")
        return {**results, **image_results}

    save_tda_results(results, format=save_format)
    if compute_image:
        save_tda_results(image_results, format=image_save_format, **save_tda_results_kwargs)
    logger.info(
        "individual_tda_features: done | saved for %s in %s and %s",
        name, save_format, image_save_format
    )
    return None


def batch_tda_features(
    distance_matrices: List[npt.NDArray],
    name: str,
    output_directory: Union[str, Path],
    mode: Optional[str] = None,
    metrics: List[str] = ("wasserstein", "bottleneck"),
    return_data: bool = False,
    compute_persistence: bool = True,
    compute_betti: bool = False,
    compute_entropy: bool = False,
    compute_distance: bool = False,
    compute_amplitude: bool = False,
    compute_image: bool = False,
    save_format: str = "csv",
    image_save_format: str = "png",
    persistence_diagrams_kwargs: Optional[dict] = None,
    betti_curves_kwargs: Optional[dict] = None,
    persistence_entropy_kwargs: Optional[dict] = None,
    pairwise_distances_kwargs: Optional[dict] = None,
    amplitudes_kwargs: Optional[dict] = None,
    persistence_images_kwargs: Optional[dict] = None,
    save_tda_results_kwargs: Optional[dict] = None,
) -> Optional[Dict[str, npt.NDArray]]:
    """
    Compute TDA features collectively for a batch of matrices and save or return.

    Computes persistence diagrams once for the batch then optionally
        Betti curves with a shared x grid
        persistence entropy
        pairwise diagram distances
        amplitudes
        persistence images

    Parameters
        distance_matrices
            List of square (n, n) distance matrices.
        name, output_directory, mode
            See `individual_tda_features`.
        metrics
            Metrics for distances and amplitudes, e.g. ("wasserstein", "bottleneck").
        return_data
            If True, return arrays instead of saving to disk.
        compute_*
            Feature toggles.
        save_format, image_save_format, *_kwargs
            See `individual_tda_features`.

    Returns
        Optional[Dict[str, numpy.ndarray]]
            If `return_data=True`, mapping of file paths to arrays. Otherwise None.

    Raises
        ValueError
            If any matrix is not square.

    Examples
        >>> import numpy as np, tempfile, os
        >>> dms = [np.array([[0.0, 1.0], [1.0, 0.0]]),
        ...        np.array([[0.0, 0.8], [0.8, 0.0]])]
        >>> out = tempfile.mkdtemp()
        >>> _ = batch_tda_features(dms, name="test_batch", output_directory=out,
        ...                        mode=None, compute_image=False, return_data=False)
        >>> os.path.isdir(os.path.join(out, "test_batch", "persistence_diagrams"))
        True
    """
    logger.info(
        "batch_tda_features: start | n=%d | name=%s | out_dir=%s | mode=%s",
        len(distance_matrices), name, output_directory, mode
    )

    # Defaults
    betti_curves_kwargs = betti_curves_kwargs or {"n_bins": 200, "n_jobs": -1}
    persistence_entropy_kwargs = persistence_entropy_kwargs or {"normalize": True, "n_jobs": -1}
    pairwise_distances_kwargs = pairwise_distances_kwargs or {"order": None, "n_jobs": -1}
    amplitudes_kwargs = amplitudes_kwargs or {"order": None, "n_jobs": -1}
    persistence_images_kwargs = persistence_images_kwargs or {"n_jobs": -1, "n_bins": 200, "sigma": 0.0005}
    save_tda_results_kwargs = save_tda_results_kwargs or {"cmap": "gray", "dpi": 300}
    persistence_diagrams_kwargs = persistence_diagrams_kwargs or {}

    out_root = Path(output_directory) / name
    out_root.mkdir(parents=True, exist_ok=True)

    # Feature subfolders (mirror individual_tda_features)
    persistence_folder = out_root / "persistence_diagrams"
    betti_folder       = out_root / "betti_curves"
    entropy_folder     = out_root / "persistence_entropy"
    distances_folder   = out_root / "pairwise_distances"
    amplitude_folder   = out_root / "amplitudes"
    image_folder       = out_root / "persistence_images"

    for folder, enabled in [
        (persistence_folder, compute_persistence),
        (betti_folder, compute_betti),
        (entropy_folder, compute_entropy),
        (distances_folder, compute_distance),
        (amplitude_folder, compute_amplitude),
        (image_folder, compute_image),
    ]:
        if enabled:
            folder.mkdir(exist_ok=True)

    # 1) Persistence diagrams (computed once for whole batch)
    logger.debug("batch_tda_features: computing persistence diagrams")
    diagrams = rips_persistence_diagrams(
        distance_matrices, mode=mode, **persistence_diagrams_kwargs
    )

    # Prepare collectors if returning data
    results: Dict[str, npt.NDArray] = {}
    image_results: Dict[str, npt.NDArray] = {}

    # Save diagrams per sample
    if compute_persistence:
        items = {}
        for i, diag in enumerate(diagrams):
            stem = persistence_folder / f"persistence_diagram_{i:03d}"
            items[str(stem)] = diag
            if return_data:
                results[str(stem)] = diag
        save_tda_results(items, format=save_format)

    # 2) Betti curves (shared x grid)
    if compute_betti:
        betti_numbers, x_data = betti_curves(diagrams, **betti_curves_kwargs)
        # Ensure list->array for consistent handling
        if isinstance(betti_numbers, list):
            betti_numbers = np.stack(betti_numbers)  # (n_samples, n_dims, n_bins)

        # Save per-sample curves + one shared x grid
        items = {str(betti_folder / "betti_x"): x_data}
        for i in range(betti_numbers.shape[0]):
            items[str(betti_folder / f"betti_curve_{i:03d}")] = betti_numbers[i]
            if return_data:
                results[str(betti_folder / f"betti_curve_{i:03d}")] = betti_numbers[i]
        save_tda_results(items, format=save_format)

    # 3) Persistence entropy
    if compute_entropy:
        pe = persistence_entropy(diagrams, **persistence_entropy_kwargs)["persistence_entropy"]  # (n, n_dims)
        items = {str(entropy_folder / "persistence_entropy"): pe}
        if return_data:
            results[str(entropy_folder / "persistence_entropy")] = pe
        save_tda_results(items, format=save_format)

    # 4) Pairwise distances
    if compute_distance:
        distmats = diagram_distances(diagrams, metrics=list(metrics), **pairwise_distances_kwargs)
        items = {}
        for key, mat in distmats.items():
            # key like "wasserstein_distance_H0"
            stem = distances_folder / key
            items[str(stem)] = mat
            if return_data:
                results[str(stem)] = mat
        save_tda_results(items, format=save_format)

    # 5) Amplitudes
    if compute_amplitude:
        amps = diagram_amplitudes(diagrams, metrics=list(metrics), **amplitudes_kwargs)
        items = {}
        for key, arr in amps.items():
            stem = amplitude_folder / key  # e.g., "wasserstein_amplitude"
            items[str(stem)] = arr
            if return_data:
                results[str(stem)] = arr
        save_tda_results(items, format=save_format)

    # 6) Persistence images
    if compute_image:
        imgs = persistence_images(diagrams, **persistence_images_kwargs)  # list of (n_dims, n_bins, n_bins)
        # Save each sample × dimension as a separate image file
        items = {}
        for i, img_stack in enumerate(imgs):
            for dim in range(img_stack.shape[0]):
                stem = image_folder / f"persistence_image_{i:03d}_H{dim}"
                items[str(stem)] = img_stack[dim]  # 2D array required for image writers
                if return_data:
                    image_results[str(stem)] = img_stack[dim]
        save_tda_results(items, format=image_save_format, **save_tda_results_kwargs)

    logger.info("batch_tda_features: done | saved to %s", out_root)

    if return_data:
        return {**results, **image_results}
    return None


def load_tda_results(
    output_directory: Union[str, Path],
    dataset_names: Optional[List[str]] = None,
    metrics: List[str] = ("wasserstein", "bottleneck"),
    load_all: bool = True,
    load_diagrams: Optional[bool] = None,
    load_betti: Optional[bool] = None,
    load_entropy: Optional[bool] = None,
    load_distance: Optional[bool] = None,
    load_amplitude: Optional[bool] = None,
    load_image: Optional[bool] = None,
    interpolate_betti_to_shared_x: Optional[bool] = None,
    include_all_betti_x: bool = False,
) -> Dict[str, Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]]:
    """
    Load TDA outputs from a directory (or a single file) into structured NumPy arrays.

    This function supports both a flat output layout (all artifact folders directly under
    the given directory) and a nested layout with multiple dataset subfolders. It detects
    file formats automatically among {csv, npy, txt} for tabular data and {png, jpg, jpeg}
    for images. You can also pass a single file path to directly load that file.

    Betti curve handling policy
        The function can produce *two* representations of Betti curves
          1) Shared-x version
             Keys:  "betti_curves_shared"  shape (n_samples, n_dims, shared_bins)
                    "betti_x_shared"       shape (n_dims, shared_bins)
             This is used by default. If a shared x exists on disk it is used. Otherwise,
             if per-sample x files exist and are not identical, a shared x is built and
             curves are interpolated onto it. If per-sample x grids are identical, no
             interpolation is performed and that common x is used as shared.
          2) Originals (per-sample x)
             Keys:  "betti_curves_original"  stacked originals
                    "betti_x_list"           list of length n_samples, each (n_dims, n_bins_i)

        You control what is returned with these flags
          • Default (include_all_betti_x=False and interpolate_betti_to_shared_x not False)
                Return only the shared-x version: "betti_curves_shared", "betti_x_shared".
                If interpolation was needed (because per-sample x differ), originals are
                ALSO returned to avoid loss  keys "betti_curves_original", "betti_x_list".
          • include_all_betti_x=True and interpolate_betti_to_shared_x=True/None
                Return BOTH shared and originals  four keys.
          • include_all_betti_x=True and interpolate_betti_to_shared_x=False
                Return originals AND also compute shared in addition  four keys.
          • include_all_betti_x=False and interpolate_betti_to_shared_x=False
                Return ONLY originals  "betti_curves_original", "betti_x_list".
          • If only a shared x exists on disk (no per-sample x files), originals cannot
            be reconstructed, so only the shared keys will be returned.

    Accepted inputs
        • Directory path
            The directory should contain either artifact folders directly or dataset
            subfolders. Supported artifact folders
              "persistence_diagrams", "betti_curves", "persistence_entropy",
              "pairwise_distances", "amplitudes", "persistence_images".
        • Single file path
            If a single file path with extension in {csv, npy, txt, png, jpg, jpeg} is
            provided, that file is loaded and returned.

    Parameters
        output_directory
            Directory containing TDA artifacts (flat or per-dataset) or a single supported file.
        dataset_names
            Optional subset of dataset folders to load when the directory contains multiple datasets.
        metrics
            Metrics to load for distances and amplitudes (e.g., ("wasserstein", "bottleneck")).
        load_all
            If True, load all artifact groups unless overridden by the fine-grained flags below.
        load_diagrams, load_betti, load_entropy, load_distance, load_amplitude, load_image
            Fine-grained booleans to enable or disable each artifact group. If any is None,
            it inherits the value of load_all.
        interpolate_betti_to_shared_x
            Controls interpolation for Betti curves when only per-sample x files exist.
              - True  force building a shared x and interpolate curves onto it.
              - False keep originals only (unless include_all_betti_x=True, which will also
                      compute a shared version and return both).
              - None  auto
                        * if shared "betti_x" exists on disk, use it
                        * else if per-sample x exist and differ, build shared and interpolate
                        * else if per-sample x are identical, use that as shared (no interpolation)
        include_all_betti_x
            If True, always include the original per-sample curves/x (when available).
            Combined with interpolate_betti_to_shared_x it yields the behavior matrix above.

    Returns
        dict  mapping dataset_name → dict of arrays. Possible keys
            Persistence diagrams
              "persistence_diagrams"                 list of variable-length arrays
            Betti curves
              "betti_curves_shared"                  (n_samples, n_dims, shared_bins)
              "betti_x_shared"                       (n_dims, shared_bins)
              "betti_curves_original"                stacked originals (shape depends on on-disk layout)
              "betti_x_list"                         list of per-sample x arrays
            Persistence entropy
              "persistence_entropy"                  (n_samples, n_dims)
            Pairwise distances (per metric and homology dim)
              "{metric}_distance_H{d}"               (n_samples, n_samples)
            Amplitudes (per metric)
              "{metric}_amplitude"                   (n_samples, n_dims)
            Persistence images
              "persistence_images"                   dict: stem → 2-D array (float in [0, 1])

        If a single file path was provided
            { "<full_file_path>": { "data": np.ndarray } }

    Raises
        FileNotFoundError  if directory path does not exist.
        ValueError         if unsupported file type/format is encountered or array shapes are invalid.

    Example
        >>> # Minimal structure check with an empty temp tree
        >>> import tempfile, os, shutil, pathlib
        >>> root = tempfile.mkdtemp()
        >>> pathlib.Path(root, "dummy", "persistence_diagrams").mkdir(parents=True, exist_ok=True)
        >>> res = load_tda_results(root, dataset_names=["dummy"], load_all=False, load_diagrams=True)
        >>> "dummy" in res or res == {}
        True
        >>> shutil.rmtree(root)
    """
    # Local imports to keep function self-contained as a drop-in
    import logging
    import re
    from pathlib import Path
    from typing import Dict, List, Optional, Union
    import numpy as np
    import pandas as pd
    from PIL import Image
    from scipy.interpolate import interp1d

    logger = logging.getLogger(__name__)

    logger.info(
        "load_tda_results: start | out_dir=%s | dataset_names=%s | load_all=%s",
        output_directory, dataset_names, load_all
    )

    output_path = Path(output_directory)

    # Supported formats
    supported_tabular_formats = {"csv", "npy", "txt"}
    supported_image_formats = {"png", "jpg", "jpeg"}
    data_folders = {
        "persistence_diagrams",
        "betti_curves",
        "persistence_entropy",
        "pairwise_distances",
        "amplitudes",
        "persistence_images",
    }

    # ---------- helpers ----------
    def _load_array(file: Path, ext: str) -> np.ndarray:
        try:
            if ext == "csv":
                return pd.read_csv(file).to_numpy()
            if ext == "npy":
                return np.load(file, allow_pickle=False)
            if ext == "txt":
                return np.loadtxt(file)
            if ext in supported_image_formats:
                img = Image.open(file).convert("L")
                return (np.array(img).astype(np.float32) / 255.0)
        except Exception as e:
            logger.exception("load_tda_results: failed to load %s | %s", file, e)
            raise
        raise ValueError(f"Unsupported file format: {ext}")

    def _detect_format(folder: Path, pattern_glob: str) -> "Optional[str]":
        for ext in (supported_tabular_formats | supported_image_formats):
            if any(folder.glob(f"{pattern_glob}.{ext}")):
                return ext
        return None

    def _ensure_2d_samples_dims(arr: np.ndarray) -> np.ndarray:
        """Coerce to (n_samples, n_dims). Handles (n,1,n_dims), (n_dims,), (n,) etc."""
        a = np.asarray(arr)
        if a.ndim == 3 and a.shape[1] == 1:
            a = a[:, 0, :]
        elif a.ndim == 1:
            a = a[:, None]
        elif a.ndim == 2:
            pass
        else:
            a = np.squeeze(a)
            if a.ndim == 1:
                a = a[:, None]
            elif a.ndim != 2:
                raise ValueError(f"Expected 2-D (n_samples, n_dims), got shape {a.shape}")
        return a

    def _all_x_equal(xs: "List[np.ndarray]") -> bool:
        """Return True if all per-sample x grids are identically equal across samples and dims."""
        if not xs:
            return False
        first = xs[0]
        if any(x.shape != first.shape for x in xs):
            return False
        return all(np.array_equal(x, first) for x in xs)

    # ---------- single-file mode ----------
    if output_path.is_file():
        ext = output_path.suffix.lower().lstrip(".")
        all_supported = supported_tabular_formats | supported_image_formats
        if ext not in all_supported:
            raise ValueError(f"Unsupported file type for load_tda_results: {output_path.suffix}")
        arr = _load_array(output_path, ext)
        return {str(output_path): {"data": arr}}

    # ---------- directory mode ----------
    if not output_path.exists():
        raise FileNotFoundError(f"Output path not found: {output_path}")

    subdirs = [d for d in output_path.iterdir() if d.is_dir()]
    # Flat layout detection
    if any(d.name in data_folders for d in subdirs):
        subdirs = [output_path]
    elif dataset_names:
        subdirs = [d for d in subdirs if d.name in dataset_names]

    results: Dict[str, Dict[str, "Union[np.ndarray, Dict[str, np.ndarray]]"]] = {}

    for subdir in subdirs:
        name = subdir.name
        label_data: Dict[str, "Union[np.ndarray, Dict[str, np.ndarray]]"] = {}

        _load_diagrams = load_diagrams if load_diagrams is not None else load_all
        _load_betti    = load_betti    if load_betti    is not None else load_all
        _load_entropy  = load_entropy  if load_entropy  is not None else load_all
        _load_distance = load_distance if load_distance is not None else load_all
        _load_amplitude= load_amplitude if load_amplitude is not None else load_all
        _load_image    = load_image    if load_image    is not None else load_all

        # Persistence diagrams (variable-length lists)
        pd_folder = subdir / "persistence_diagrams"
        if _load_diagrams and pd_folder.exists():
            fmt = _detect_format(pd_folder, "persistence_diagram_*")
            if fmt:
                files = sorted(pd_folder.glob(f"persistence_diagram_*.{fmt}"))
                label_data["persistence_diagrams"] = [_load_array(f, fmt) for f in files]

        # --------------------- Betti curves + x  (NAMING & LOGIC) ---------------------
        betti_folder = subdir / "betti_curves"
        if _load_betti and betti_folder.exists():
            curve_fmt = _detect_format(betti_folder, "betti_curve_*")
            if curve_fmt:
                curve_files = sorted(betti_folder.glob(f"betti_curve_*.{curve_fmt}"))
                curves = [_load_array(f, curve_fmt) for f in curve_files]

                # Look for shared and per-sample x on disk
                x_shared_fmt = _detect_format(betti_folder, "betti_x")
                x_shared_file = (betti_folder / f"betti_x.{x_shared_fmt}") if x_shared_fmt else None

                x_per_fmt = _detect_format(betti_folder, "betti_x_*")
                x_per_files = sorted(betti_folder.glob(f"betti_x_*.{x_per_fmt}")) if x_per_fmt else []
                have_per_sample = (x_per_files and len(x_per_files) == len(curve_files))
                xs = [_load_array(f, x_per_fmt) for f in x_per_files] if have_per_sample else []

                # Convenience setters
                def _set_originals(xs_list: "List[np.ndarray]") -> None:
                    label_data["betti_curves_original"] = np.array(curves)
                    label_data["betti_x_list"] = xs_list

                def _set_shared_from(curves_arr: np.ndarray, x_shared: np.ndarray) -> None:
                    label_data["betti_curves_shared"] = curves_arr
                    label_data["betti_x_shared"] = x_shared

                # Core cases matrix
                if include_all_betti_x and (interpolate_betti_to_shared_x is False):
                    # Originals required; also compute shared in addition (four keys when possible)
                    if have_per_sample:
                        _set_originals(xs)
                        if _all_x_equal(xs):
                            _set_shared_from(np.array(curves), xs[0])
                        else:
                            # Build shared and interpolate
                            max_dim, n_bins = curves[0].shape
                            mins = [min(x[d].min() for x in xs) for d in range(max_dim)]
                            maxs = [max(x[d].max() for x in xs) for d in range(max_dim)]
                            x_common = [np.linspace(mn, mx, n_bins) for mn, mx in zip(mins, maxs)]
                            interpolated = []
                            for curve, x_vals in zip(curves, xs):
                                dims_interp = [
                                    interp1d(x_vals[d], curve[d], kind="linear",
                                             bounds_error=False, fill_value=0.0)(x_common[d])
                                    for d in range(max_dim)
                                ]
                                interpolated.append(np.array(dims_interp))
                            _set_shared_from(np.array(interpolated), np.vstack([xc for xc in x_common]))
                    elif x_shared_file and x_shared_file.exists():
                        # No per-sample x on disk; only shared is possible
                        _set_shared_from(np.array(curves), _load_array(x_shared_file, x_shared_fmt))
                    else:
                        # No x files at all
                        label_data["betti_curves_original"] = np.array(curves)

                elif include_all_betti_x and (interpolate_betti_to_shared_x is not False):
                    # Want originals + shared (four keys when both available)
                    if have_per_sample:
                        _set_originals(xs)
                        if x_shared_file and x_shared_file.exists():
                            _set_shared_from(np.array(curves), _load_array(x_shared_file, x_shared_fmt))
                        elif _all_x_equal(xs):
                            _set_shared_from(np.array(curves), xs[0])
                        else:
                            max_dim, n_bins = curves[0].shape
                            mins = [min(x[d].min() for x in xs) for d in range(max_dim)]
                            maxs = [max(x[d].max() for x in xs) for d in range(max_dim)]
                            x_common = [np.linspace(mn, mx, n_bins) for mn, mx in zip(mins, maxs)]
                            interpolated = []
                            for curve, x_vals in zip(curves, xs):
                                dims_interp = [
                                    interp1d(x_vals[d], curve[d], kind="linear",
                                             bounds_error=False, fill_value=0.0)(x_common[d])
                                    for d in range(max_dim)
                                ]
                                interpolated.append(np.array(dims_interp))
                            _set_shared_from(np.array(interpolated), np.vstack([xc for xc in x_common]))
                    elif x_shared_file and x_shared_file.exists():
                        _set_shared_from(np.array(curves), _load_array(x_shared_file, x_shared_fmt))
                    else:
                        label_data["betti_curves_original"] = np.array(curves)

                elif (not include_all_betti_x) and (interpolate_betti_to_shared_x is False):
                    # Originals only
                    if have_per_sample:
                        _set_originals(xs)
                    else:
                        label_data["betti_curves_original"] = np.array(curves)

                else:
                    # Default  shared only
                    if x_shared_file and x_shared_file.exists():
                        _set_shared_from(np.array(curves), _load_array(x_shared_file, x_shared_fmt))
                    elif have_per_sample:
                        if _all_x_equal(xs):
                            _set_shared_from(np.array(curves), xs[0])
                        else:
                            # Build shared and interpolate
                            max_dim, n_bins = curves[0].shape
                            mins = [min(x[d].min() for x in xs) for d in range(max_dim)]
                            maxs = [max(x[d].max() for x in xs) for d in range(max_dim)]
                            x_common = [np.linspace(mn, mx, n_bins) for mn, mx in zip(mins, maxs)]
                            interpolated = []
                            for curve, x_vals in zip(curves, xs):
                                dims_interp = [
                                    interp1d(x_vals[d], curve[d], kind="linear",
                                             bounds_error=False, fill_value=0.0)(x_common[d])
                                    for d in range(max_dim)
                                ]
                                interpolated.append(np.array(dims_interp))
                            _set_shared_from(np.array(interpolated), np.vstack([xc for xc in x_common]))
                            # Also keep originals to avoid loss when values changed
                            label_data["betti_curves_original"] = np.array(curves)
                            label_data["betti_x_list"] = xs
                    else:
                        # No x available; return curves as shared (no x)
                        label_data["betti_curves_shared"] = np.array(curves)
        # ------------------- end Betti block -------------------------

        # Persistence entropy → enforce (n_samples, n_dims)
        entropy_folder = subdir / "persistence_entropy"
        if _load_entropy and entropy_folder.exists():
            single_fmt = _detect_format(entropy_folder, "persistence_entropy")
            if single_fmt and (entropy_folder / f"persistence_entropy.{single_fmt}").exists():
                pe = _load_array(entropy_folder / f"persistence_entropy.{single_fmt}", single_fmt)
                label_data["persistence_entropy"] = _ensure_2d_samples_dims(pe)
            else:
                fmt = _detect_format(entropy_folder, "persistence_entropy_*")
                if fmt:
                    files = sorted(entropy_folder.glob(f"persistence_entropy_*.{fmt}"))
                    arr = np.array([_load_array(f, fmt) for f in files])
                    if arr.ndim == 3 and arr.shape[1] == 1:
                        arr = arr[:, 0, :]
                    elif arr.ndim == 2:
                        pass
                    elif arr.ndim == 1:
                        arr = arr[:, None]
                    else:
                        arr = np.squeeze(arr)
                        if arr.ndim == 1:
                            arr = arr[:, None]
                    label_data["persistence_entropy"] = arr

        # Pairwise distances → per dim matrices
        dist_folder = subdir / "pairwise_distances"
        if _load_distance and dist_folder.exists():
            for metric in metrics:
                fmt = _detect_format(dist_folder, f"{metric}_distance_*")
                if fmt:
                    files = sorted(dist_folder.glob(f"{metric}_distance_*.{fmt}"))
                    for f in files:
                        m = re.search(r"_H(\d+)\.", f.name)
                        mat = _load_array(f, fmt)
                        if mat.ndim != 2:
                            raise ValueError(f"{f} should be a 2-D (n×n) matrix, got {mat.shape}")
                        if m:
                            label_data[f"{metric}_distance_H{int(m.group(1))}"] = mat
                        else:
                            label_data[f"{metric}_distance_H-1"] = mat  # unknown dim

        # Amplitudes → coerce to (n_samples, n_dims)
        amp_folder = subdir / "amplitudes"
        if _load_amplitude and amp_folder.exists():
            for metric in metrics:
                single_fmt = _detect_format(amp_folder, f"{metric}_amplitude")
                if single_fmt and (amp_folder / f"{metric}_amplitude.{single_fmt}").exists():
                    pe = _load_array(amp_folder / f"{metric}_amplitude.{single_fmt}", single_fmt)
                    label_data[f"{metric}_amplitude"] = _ensure_2d_samples_dims(pe)
                else:
                    fmt = _detect_format(amp_folder, f"{metric}_amplitude_*")
                    if fmt:
                        files = sorted(amp_folder.glob(f"{metric}_amplitude_*.{fmt}"))
                        arr = np.array([_load_array(f, fmt) for f in files])
                        if arr.ndim == 3 and arr.shape[1] == 1:
                            arr = arr[:, 0, :]
                        elif arr.ndim == 2:
                            pass
                        elif arr.ndim == 1:
                            arr = arr[:, None]
                        else:
                            arr = np.squeeze(arr)
                            if arr.ndim == 1:
                                arr = arr[:, None]
                        label_data[f"{metric}_amplitude"] = arr
        # ------------------- end Amplitude block -------------------------
        # Persistence images
        img_folder = subdir / "persistence_images"
        if _load_image and img_folder.exists():
            fmt = _detect_format(img_folder, "persistence_image_*")
            if fmt:
                label_data["persistence_images"] = {
                    file.stem: _load_array(file, fmt)
                    for file in sorted(img_folder.glob(f"persistence_image_*.{fmt}"))
                }

        if label_data:
            results[name] = label_data

    logger.info("load_tda_results: done | datasets_loaded=%d", len(results))
    return results


def compute_betti_stat_features(
    tda_results: Dict[str, Dict[str, np.ndarray]],
    *,
    save: bool = False,
    save_root: Union[str, Path] = "./",
    save_format: str = "csv",   # {"csv","npy","txt"}
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute compact statistical features from Betti curves already loaded in memory.

    Input
      tda_results
        A dict mapping dataset_name -> artifact dict, as returned by your loader.
        This function looks for Betti curves in the following priority order
          1) "betti_curves_original"  with "betti_x_list"   (native per-sample x, preferred)
          2) "betti_curves_shared"    with "betti_x_shared" (shared x fallback)
          3) Back-compat keys: "betti_curves" with either "betti_x_list" or "betti_x"

        Accepted shapes
          betti_curves_*  (n_samples, n_dims, n_bins_i or shared_bins)
          betti_x_list    list of length n_samples, each (n_dims, n_bins_i)
          betti_x_shared  (n_dims, shared_bins)

        No interpolation is performed. If only the shared version exists, features are
        computed on the shared x (that’s still “no interpolation”).

    Computed features per sample × homology dimension (columns in this order)
      0  auc_trapz            ∫ y dx  using trapezoidal rule on sorted x
      1  centroid_x           (∫ x y dx) / (∫ y dx)  NaN if ∑y == 0
      2  peak_y               max(y)
      3  std_y                standard deviation of y (ddof=1 if size>1 else 0)
      4  skewness_y           standardized 3rd moment
      5  kurtosis_excess_y    standardized 4th moment minus 3

    Parameters
      save
        If True, saves per-dimension feature tables for each dataset to disk.
      save_root
        Root directory under which per-dataset folders will be written.
      save_format
        One of {"csv","npy","txt"} for saving when save=True.

    Returns
      features_by_dataset
        dict mapping dataset_name → dict with keys
          "feature_names" : np.ndarray, shape (6,)
          "H{d}"          : np.ndarray, shape (n_samples, 6), for each homology dim d

    Raises
      ValueError
        If Betti data are missing or inconsistent.
    """

    feature_names = np.array(
        ["auc_trapz", "centroid_x", "peak_y", "std_y", "skewness_y", "kurtosis_excess_y"],
        dtype=object
    )
    _EPS = 1e-12

    def _curve_features(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute stats for a single 1D Betti curve y(x)."""
        idx = np.argsort(x)
        xs = np.asarray(x, dtype=float)[idx]
        ys = np.asarray(y, dtype=float)[idx]

        auc = float(np.trapz(ys, xs))
        wy = float(np.trapz(xs * ys, xs))
        denom = float(np.trapz(ys, xs))
        centroid_x = float(wy / denom) if abs(denom) > _EPS else float("nan")

        peak_y = float(np.max(ys)) if ys.size else 0.0
        std_y = float(np.std(ys, ddof=1)) if ys.size > 1 else 0.0

        if ys.size >= 3 and std_y > 0:
            y_mean = float(np.mean(ys))
            m = ys - y_mean
            m2 = float(np.mean(m**2))
            m3 = float(np.mean(m**3))
            m4 = float(np.mean(m**4))
            skew = float(m3 / (m2 ** 1.5 + _EPS))
            kurt_excess = float(m4 / (m2 ** 2 + _EPS) - 3.0)
        else:
            skew, kurt_excess = 0.0, -3.0

        return np.array([auc, centroid_x, peak_y, std_y, skew, kurt_excess], dtype=float)

    def _pick_betti_source(art: Dict[str, np.ndarray]) -> tuple[np.ndarray, Optional[List[np.ndarray]]]:
        """
        Return (curves, xs_list_or_shared) where:
          curves: np.ndarray (n_samples, n_dims, n_bins_i or shared_bins)
          xs_list_or_shared:
             - list of per-sample (n_dims, n_bins_i) when originals exist
             - np.ndarray (n_dims, shared_bins) when shared exists
             - None if no x provided (allowed; features not needing x would still work,
               but we need x for integrals, so we will error if None)
        """
        # Preferred: originals
        if "betti_curves_original" in art and "betti_x_list" in art:
            curves = np.asarray(art["betti_curves_original"])
            xs_list = art["betti_x_list"]
            if not isinstance(xs_list, list) or len(xs_list) != curves.shape[0]:
                raise ValueError("betti_x_list must be a list with length n_samples")
            return curves, xs_list

        # Fallback: shared
        if "betti_curves_shared" in art and "betti_x_shared" in art:
            return np.asarray(art["betti_curves_shared"]), np.asarray(art["betti_x_shared"])

        raise ValueError("No Betti curves found. Expected originals or shared keys in tda_results.")

    features_by_dataset: Dict[str, Dict[str, np.ndarray]] = {}
    save_root = Path(save_root)

    for ds_name, art in tda_results.items():
        # pick source
        curves, xref = _pick_betti_source(art)
        if curves.ndim != 3:
            raise ValueError(f"{ds_name}: betti_curves must be 3D (n_samples, n_dims, n_bins), got {curves.shape}")

        n_samples, n_dims, _ = curves.shape

        # Build per-dimension tables
        per_dim_rows: Dict[str, List[np.ndarray]] = {f"H{d}": [] for d in range(n_dims)}

        if isinstance(xref, list):
            # Original per-sample x
            if len(xref) != n_samples:
                raise ValueError(f"{ds_name}: len(betti_x_list) must equal n_samples")
            for i in range(n_samples):
                xi = np.asarray(xref[i])
                yi = curves[i]
                if xi.shape != yi.shape:
                    raise ValueError(f"{ds_name}: x and curve shape mismatch at sample {i}: {xi.shape} vs {yi.shape}")
                for d in range(n_dims):
                    feats = _curve_features(xi[d], yi[d])
                    per_dim_rows[f"H{d}"].append(feats)
        else:
            # Shared x (np.ndarray) required
            if xref is None:
                raise ValueError(f"{ds_name}: missing x for Betti curves; required for integrals.")
            x_shared = np.asarray(xref)
            if x_shared.ndim != 2 or x_shared.shape[0] != n_dims:
                raise ValueError(f"{ds_name}: betti_x_shared must be (n_dims, n_bins), got {x_shared.shape}")
            for i in range(n_samples):
                yi = curves[i]
                if yi.shape[0] != n_dims or yi.shape[1] != x_shared.shape[1]:
                    raise ValueError(
                        f"{ds_name}: curve bins do not match shared x at sample {i}: {yi.shape} vs {x_shared.shape}"
                    )
                for d in range(n_dims):
                    feats = _curve_features(x_shared[d], yi[d])
                    per_dim_rows[f"H{d}"].append(feats)

        # Pack results
        pack: Dict[str, np.ndarray] = {"feature_names": feature_names}
        for d in range(n_dims):
            pack[f"H{d}"] = np.vstack(per_dim_rows[f"H{d}"]) if per_dim_rows[f"H{d}"] else np.empty((0, len(feature_names)))
        features_by_dataset[ds_name] = pack

        # Optional saving
        if save:
            ds_dir = save_root / ds_name / "betti_stats_pval"
            ds_dir.mkdir(parents=True, exist_ok=True)
            if save_format == "csv":
                for d in range(n_dims):
                    pd.DataFrame(pack[f"H{d}"], columns=feature_names).to_csv(
                        ds_dir / f"betti_stats_H{d}.csv", index=False
                    )
            elif save_format == "npy":
                for d in range(n_dims):
                    np.save(ds_dir / f"betti_stats_H{d}.npy", pack[f"H{d}"])
            elif save_format == "txt":
                header = " ".join(feature_names.tolist())
                for d in range(n_dims):
                    np.savetxt(ds_dir / f"betti_stats_H{d}.txt", pack[f"H{d}"], header=header)
            else:
                raise ValueError("save_format must be one of {'csv','npy','txt'}")

    logger.info("compute_betti_stat_features: done | datasets=%d", len(features_by_dataset))
    return features_by_dataset

