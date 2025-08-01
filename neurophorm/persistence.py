from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.interpolate import interp1d
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
from gtda.homology import VietorisRipsPersistence, SparseRipsPersistence
from gtda.diagrams import (
    BettiCurve,
    PersistenceEntropy,
    PairwiseDistance,
    Amplitude,
    PersistenceImage
)

def corr_to_distance_matrices(
    correlation_matrices: List[npt.NDArray],
    mode: Optional[str] = None
) -> List[npt.NDArray]:
    """
    Convert correlation matrices to distance matrices for topological data analysis.

    This function transforms a list of correlation matrices into distance matrices
    using one of three modes: standard (1 - correlation), positive correlations only,
    or negative correlations only. The resulting distance matrices are suitable for
    persistence homology computations.

    Args:
        correlation_matrices: A list of correlation matrices, where each matrix is a
            square NumPy array of shape (n_points, n_points) with values in [-1, 1].
        mode: The method for computing distances. Options are:
            - None (default): Computes distances as 1 - correlation.
            - "positive": Uses positive correlations only, setting negative correlations
                to a large value (10).
            - "negative": Uses absolute value of negative correlations only, setting
                positive correlations to a large value (10).

    Returns:
        A list of distance matrices, where each matrix is a NumPy array of shape
        (n_points, n_points) with non-negative values and zeros on the diagonal.

    Raises:
        ValueError: If `mode` is not None, "positive", or "negative".

    Example:
        >>> import numpy as np
        >>> corr_matrix = [np.array([[1.0, 0.5], [0.5, 1.0]])]
        >>> dist_matrices = corr_to_distance_matrices(corr_matrix, mode=None)
        >>> print(dist_matrices[0])
        [[0.  0.5]
         [0.5 0. ]]
    """
    valid_modes = {None, "positive", "negative"}
    if mode not in valid_modes:
        raise ValueError(f"Mode must be one of {valid_modes}")

    all_distance_matrix = []
    for data in correlation_matrices:
        if mode is None:
            distance = 1 - data
        elif mode == "positive":
            distance = np.where(data > 0, data, 10)
        elif mode == "negative":
            distance = np.where(data < 0, abs(data), 10)
        
        np.fill_diagonal(distance, 0)
        all_distance_matrix.append(distance)

    return all_distance_matrix


def rips_persistence_diagrams(
    distance_matrices: List[npt.NDArray],
    mode: Optional[str] = None,
    **kwargs
) -> List[npt.NDArray]:
    """
    Compute persistence diagrams from distance matrices using Vietoris-Rips or sparse Rips persistence.

    This function applies topological data analysis to compute persistence diagrams
    from a list of distance matrices. It supports both standard Vietoris-Rips and
    sparse Rips persistence computations, configurable via the mode parameter.

    Args:
        distance_matrices: A list of distance matrices, where each matrix is a square
            NumPy array of shape (n_points, n_points) with non-negative values.
        mode: The persistence computation method. Options are:
            - None (default): Uses standard Vietoris-Rips persistence.
            - "sparse": Uses sparse Rips persistence for improved efficiency on large datasets.
        **kwargs: Additional parameters for persistence computation, including:
            - homology_dimensions: Tuple of integers specifying homology dimensions to compute (default: (0, 1, 2)).
            - n_jobs: Number of jobs for parallel computation (default: None).
            - collapse_edges: Whether to collapse edges in sparse filtrations (bool, for SparseRipsPersistence only).
            - max_edge_length: Maximum edge length for sparse filtrations (float, for SparseRipsPersistence only).
            - approx_mode: Approximation mode for sparse filtrations (str, for SparseRipsPersistence only).

    Returns:
        A list of persistence diagrams, where each diagram is a NumPy array of shape
        (n_features, 3), with columns representing birth, death, and homology dimension.

    Raises:
        ValueError: If `mode` is not None or "sparse", or if any distance matrix is not square.

    Example:
        >>> import numpy as np
        >>> dist_matrix = [np.array([[0.0, 1.0], [1.0, 0.0]])]
        >>> diagrams = rips_persistence_diagrams(dist_matrix, mode=None, homology_dimensions=(0, 1))
        >>> print(diagrams[0].shape)
        (n_features, 3)
    """
    valid_modes = {None, "sparse"}
    if mode not in valid_modes:
        raise ValueError(f"Mode must be one of {valid_modes}")

    # Validate distance matrices
    for matrix in distance_matrices:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Each distance matrix must be square (n_points, n_points)")

    if mode is None:
        vr_persistence = VietorisRipsPersistence(
            metric='precomputed',
            **kwargs
        )
    else:
        vr_persistence = SparseRipsPersistence(
            metric='precomputed',
            **kwargs
        )
    
    # gtda expects input as (n_samples, n_points, n_points)
    diagrams = vr_persistence.fit_transform(np.array(distance_matrices))
    return [diagrams[i] for i in range(len(distance_matrices))]


def betti_curves(
    persistence_diagrams: List[npt.NDArray],
    **kwargs
) -> Tuple[List[npt.NDArray], npt.NDArray]:
    """
    Compute Betti curves from persistence diagrams for topological feature analysis.

    Betti curves represent the number of topological features (connected components,
    loops, voids) at different filtration values, derived from persistence diagrams.

    Args:
        persistence_diagrams: A list of persistence diagrams, where each diagram is a
            NumPy array of shape (n_features, 3) with columns for birth, death, and
            homology dimension.
        **kwargs: Additional parameters for Betti curve computation, including:
            - n_bins: Number of bins for discretizing the filtration (int, default: 100).
            - sampling: Custom sampling points for the curves (NumPy array).

    Returns:
        A tuple containing:
            - A list of Betti curves, where each curve is a NumPy array of shape
              (n_homology_dims, n_bins) representing Betti numbers for each homology dimension.
            - A NumPy array of shape (n_bins,) containing the x-values (filtration values)
              for the Betti curves.

    Example:
        >>> import numpy as np
        >>> diagrams = [np.array([[0.0, 1.0, 0], [0.5, 1.5, 1]])]
        >>> curves, x_vals = betti_curves(diagrams, n_bins=100)
        >>> print(curves[0].shape)
        (2, 100)
        >>> print(x_vals.shape)
        (100,)
    """
    betti_curve = BettiCurve(**kwargs)
    betti_numbers = betti_curve.fit_transform(persistence_diagrams)
    fig = betti_curve.plot(persistence_diagrams)
    x_data = [trace.x for trace in fig.data]
    return betti_numbers, x_data


def persistence_entropy(
    persistence_diagrams: List[npt.NDArray],
    **kwargs
) -> Dict[str, npt.NDArray]:
    """
    Calculate persistence entropy from persistence diagrams to quantify topological complexity.

    Persistence entropy measures the complexity of the topological features in a
    persistence diagram, providing a scalar summary for each homology dimension.

    Args:
        persistence_diagrams: A list of persistence diagrams, where each diagram is a
            NumPy array of shape (n_features, 3) with columns for birth, death, and
            homology dimension.
        **kwargs: Additional parameters for entropy computation, including:
            - normalize: Whether to normalize entropy values (bool, default: False).
            - nan_fill_value: Value to replace NaNs in entropy calculations (float, default: 0.0).

    Returns:
        A dictionary with a single key "persistence_entropy" mapping to a NumPy array
        of shape (n_samples, n_homology_dims) containing entropy values for each diagram
        and homology dimension.

    Example:
        >>> import numpy as np
        >>> diagrams = [np.array([[0.0, 1.0, 0], [0.5, 1.5, 1]])]
        >>> entropy_dict = persistence_entropy(diagrams, normalize=True)
        >>> print(entropy_dict["persistence_entropy"].shape)
        (1, 2)
    """
    pe = PersistenceEntropy(**kwargs)
    entropy = pe.fit_transform(np.array(persistence_diagrams))
    return {"persistence_entropy": entropy}


def diagram_distances(
    persistence_diagrams: List[npt.NDArray],
    metrics: List[str] = ["wasserstein", "bottleneck"],
    **kwargs,
) -> Dict[str, npt.NDArray]:
    """
    Compute pairwise distances between persistence diagrams using specified metrics.

    This function calculates distances between persistence diagrams to quantify their
    topological similarity, supporting metrics like Wasserstein and bottleneck distances.

    Args:
        persistence_diagrams: A list of persistence diagrams, where each diagram is a
            NumPy array of shape (n_features, 3) with columns for birth, death, and
            homology dimension.
        metrics: A list of distance metrics to compute. Supported options are
            "wasserstein" and "bottleneck" (default: ["wasserstein", "bottleneck"]).
        **kwargs: Additional parameters for distance computation, including:
            - order: Order of the distance metric (int or None, default: None).
            - n_jobs: Number of jobs for parallel computation (int or None, default: None).

    Returns:
        A dictionary where each key is formatted as "{metric}_distance_H{dim}" and maps
        to a NumPy array of shape (n_samples, n_samples) containing pairwise distances
        for the specified metric and homology dimension.

    Example:
        >>> import numpy as np
        >>> diagrams = [np.array([[0.0, 1.0, 0]]), np.array([[0.0, 1.0, 0]])]
        >>> distances = diagram_distances(diagrams, metrics=["wasserstein"])
        >>> print(distances["wasserstein_distance_H0"].shape)
        (2, 2)
    """
    distances_dict = {}
    diagrams = np.array(persistence_diagrams)
    for metric in metrics:
        pd_calc = PairwiseDistance(metric=metric, **kwargs)
        distances = pd_calc.fit_transform(diagrams)
        # distances shape: (n_samples, n_samples, n_homology_dims)
        for dim in range(distances.shape[2]):
            distances_dict[f"{metric}_distance_H{dim}"] = distances[:, :, dim]
    return distances_dict


def diagram_amplitudes(
    persistence_diagrams: List[npt.NDArray],
    metrics: List[str] = ["wasserstein", "bottleneck"],
    **kwargs
) -> Dict[str, npt.NDArray]:
    """
    Compute amplitudes of persistence diagrams to summarize topological features.

    Amplitudes provide a scalar summary of the persistence diagrams' topological
    features, computed using specified distance metrics relative to an empty diagram.

    Args:
        persistence_diagrams: A list of persistence diagrams, where each diagram is a
            NumPy array of shape (n_features, 3) with columns for birth, death, and
            homology dimension.
        metrics: A list of metrics for amplitude computation. Supported options are
            "wasserstein" and "bottleneck" (default: ["wasserstein", "bottleneck"]).
        **kwargs: Additional parameters for amplitude computation, including:
            - order: Order of the distance metric (int or None, default: None).
            - n_jobs: Number of jobs for parallel computation (int or None, default: None).

    Returns:
        A dictionary where each key is formatted as "{metric}_amplitude" and maps to a
        NumPy array of shape (n_samples, n_homology_dims) containing amplitude values
        for each diagram and homology dimension.

    Example:
        >>> import numpy as np
        >>> diagrams = [np.array([[0.0, 1.0, 0]])]
        >>> amplitudes = diagram_amplitudes(diagrams, metrics=["bottleneck"])
        >>> print(amplitudes["bottleneck_amplitude"].shape)
        (1, 1)
    """
    amplitudes_dict = {}
    diagrams = np.array(persistence_diagrams)
    for metric in metrics:
        amplitude = Amplitude(metric=metric, **kwargs)
        amplitudes = amplitude.fit_transform(diagrams)
        amplitudes_dict[f"{metric}_amplitude"] = amplitudes
    return amplitudes_dict


def persistence_images(
    persistence_diagrams: List[npt.NDArray],
    **kwargs
) -> List[npt.NDArray]:
    """
    Generate persistence images from persistence diagrams for visualization and analysis.

    Persistence images represent persistence diagrams as 2D arrays, where topological
    features are convolved with a Gaussian kernel, suitable for machine learning or visualization.

    Args:
        persistence_diagrams: A list of persistence diagrams, where each diagram is a
            NumPy array of shape (n_features, 3) with columns for birth, death, and
            homology dimension.
        **kwargs: Additional parameters for image computation, including:
            - sigma: Width of the Gaussian kernel for image generation (float, default: 0.1).
            - n_bins: Number of bins per axis for the image (int, default: 100).
            - weight_function: Custom weight function for persistence points (callable, optional).

    Returns:
        A list of persistence images, where each image is a NumPy array of shape
        (n_homology_dims, n_bins, n_bins) representing the convolved topological features.

    Example:
        >>> import numpy as np
        >>> diagrams = [np.array([[0.0, 1.0, 0]])]
        >>> images = persistence_images(diagrams, n_bins=50, sigma=0.1)
        >>> print(images[0].shape)
        (1, 50, 50)
    """
    pi = PersistenceImage(**kwargs)
    images = pi.fit_transform(np.array(persistence_diagrams))
    return [images[i] for i in range(len(persistence_diagrams))]


def save_tda_results(
    data_dict: Dict[str, npt.NDArray],
    overwrite: bool = True,
    format: str = "csv",
    **kwargs
) -> None:
    """
    Save topological analysis results to files in specified formats.

    This function saves a dictionary of NumPy arrays to files, supporting tabular
    formats (CSV, NPY, TXT) and image formats (PNG, JPG, JPEG). It ensures proper
    directory creation and handles overwriting existing files.

    Args:
        data_dict: A dictionary where keys are file paths (as strings) and values are
            NumPy arrays containing the data to save.
        overwrite: If True, overwrites existing files; if False, skips existing files
            (default: True).
        format: The file format for saving data. Supported options are "csv", "npy",
            "txt", "png", "jpg", "jpeg" (default: "csv").
        **kwargs: Additional parameters for saving, particularly for images:
            - cmap: Colormap for saving persistence images (str, default: "gray").
            - vmin: Minimum value for image normalization (float, optional).
            - vmax: Maximum value for image normalization (float, optional).

    Raises:
        ValueError: If `format` is not supported or if image data is not 2D.

    Example:
        >>> import numpy as np
        >>> data = {"results/data.csv": np.array([[1, 2], [3, 4]])}
        >>> save_tda_results(data, format="csv", overwrite=True)
    """
    supported_formats = {"csv", "npy", "txt", "png", "jpg", "jpeg"}
    format = format.lower()

    if format not in supported_formats:
        raise ValueError(f"Unsupported format '{format}'. Supported formats are {supported_formats}")

    for file_path, data in data_dict.items():
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Adjust file extension based on format
        file_path = file_path.with_suffix(f".{format}")

        if overwrite or not file_path.exists():
            try:
                if format == "csv":
                    pd.DataFrame(data).to_csv(file_path, index=False)
                elif format == "npy":
                    np.save(file_path, data)
                elif format == "txt":
                    np.savetxt(file_path, data)
                elif format in {"png", "jpg", "jpeg"}:
                    # Ensure data is 2D for image saving
                    if data.ndim > 2:
                        raise ValueError(f"Cannot save {file_path} as image: data must be 2D")
                    plt.imsave(
                        file_path,
                        data,
                        format=format,
                        **kwargs
                    )
            except Exception as e:
                print(f"Error saving {file_path}: {str(e)}")


def individual_tda_features(
    distance_matrices: List[npt.NDArray],
    name: str,
    output_directory: Union[str, Path] = "./",
    mode: Optional[str] = None,
    return_data: bool = False,
    compute_persistence: bool = True,
    compute_betti: bool = True,
    compute_entropy: bool = True,
    compute_amplitude: bool = True,
    compute_image: bool = True,
    save_format: str = "csv",
    image_save_format: str = "png",
    persistence_diagrams_kwargs: Optional[dict] = None,
    betti_curves_kwargs: Optional[dict] = {"n_bins": 200, "n_jobs": -1},
    persistence_entropy_kwargs: Optional[dict] = {"normalize": True, "n_jobs": -1},
    amplitudes_kwargs: Optional[dict] = {"order": None, "n_jobs": -1},
    persistence_images_kwargs: Optional[dict] = {"n_jobs": -1, "n_bins": 200, "sigma": 0.0005},
    save_tda_results_kwargs: Optional[dict] = {"cmap": "gray", "dpi": 300},
) -> Optional[Dict[str, npt.NDArray]]:
    """
    Compute topological features independently for each distance matrix and save or return results.

    This function processes each distance matrix individually to compute topological
    features such as persistence diagrams, Betti curves, persistence entropy, amplitudes,
    and persistence images. Results are organized into subdirectories and can be saved
    to files or returned as a dictionary.

    Args:
        distance_matrices: A list of distance matrices, where each matrix is a square
            NumPy array of shape (n_points, n_points) with non-negative values.
        name: A string identifier for the dataset or group, used as a subdirectory name.
        output_directory: The base directory (str or Path) where results will be saved.
        mode: The persistence computation method (None for Vietoris-Rips, "sparse" for sparse Rips).
        return_data: If True, returns results as a dictionary; if False, saves to files (default: False).
        compute_persistence: If True, computes persistence diagrams (default: True).
        compute_betti: If True, computes Betti curves (default: True).
        compute_entropy: If True, computes persistence entropy (default: True).
        compute_amplitude: If True, computes amplitudes (default: True).
        compute_image: If True, computes persistence images (default: True).
        save_format: File format for tabular data ("csv", "npy", "txt"; default: "csv").
        image_save_format: File format for images ("png", "jpg", "jpeg"; default: "png").
        persistence_diagrams_kwargs: Additional parameters for persistence diagram computation (dict, optional).
        betti_curves_kwargs: Additional parameters for Betti curve computation (dict, default: {"n_bins": 200, "n_jobs": -1}).
        persistence_entropy_kwargs: Additional parameters for entropy computation (dict, default: {"normalize": True, "n_jobs": -1}).
        amplitudes_kwargs: Additional parameters for amplitude computation (dict, default: {"order": None, "n_jobs": -1}).
        persistence_images_kwargs: Additional parameters for image computation (dict, default: {"n_jobs": -1, "n_bins": 200, "sigma": 0.0005}).
        save_tda_results_kwargs: Additional parameters for saving results, especially images (dict, default: {"cmap": "gray", "dpi": 300}).

    Returns:
        If `return_data` is True, a dictionary where keys are file paths (as strings) and
        values are NumPy arrays containing the computed topological features. Otherwise,
        None, with results saved to the specified output directory.

    Raises:
        ValueError: If any distance matrix is not square.

    Example:
        >>> import numpy as np
        >>> dist_matrices = [np.array([[0.0, 1.0], [1.0, 0.0]])]
        >>> individual_tda_features(dist_matrices, "test", "output", mode=None, return_data=False)
        Data processed and saved for test in csv and png formats.
    """
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
            raise ValueError("Each distance matrix must be square (n_points, n_points)")

        persistence_diagrams = rips_persistence_diagrams([matrix], mode, **persistence_diagrams_kwargs)

        # Save persistence diagrams
        if compute_persistence:
            results[str(persistence_folder / f"persistence_diagram_{idx:03d}")] = persistence_diagrams[0]

        # Compute Betti curves
        if compute_betti:
            betti_numbers, x_data = betti_curves(persistence_diagrams, **betti_curves_kwargs)
            results[str(betti_folder / f"betti_curve_{idx:03d}")] = betti_numbers[0]
            results[str(betti_folder / f"betti_x_{idx:03d}")] = x_data

        # Compute persistence entropy
        if compute_entropy:
            entropy_results = persistence_entropy(persistence_diagrams, **persistence_entropy_kwargs)
            for filename, data in entropy_results.items():
                results[str(entropy_folder / f"{filename}_{idx:03d}")] = data

        # Compute amplitudes
        if compute_amplitude:
            amplitude_results = diagram_amplitudes(persistence_diagrams, metrics=["wasserstein", "bottleneck"], **amplitudes_kwargs)
            for filename, data in amplitude_results.items():
                results[str(amplitude_folder / f"{filename}_{idx:03d}")] = data

        # Compute persistence images
        if compute_image:
            images_results = persistence_images(persistence_diagrams, **persistence_images_kwargs)
            for dim in range(images_results[0].shape[0]):
                save_path = str(image_folder / f"persistence_image_{idx:03d}_H{dim}")
                image_results[save_path] = images_results[0][dim]

    if return_data:
        return {**results, **image_results}
    else:
        save_tda_results(results, format=save_format)
        if compute_image:
            save_tda_results(image_results, format=image_save_format, **save_tda_results_kwargs)
        print(f"Data processed and saved for {name} in {save_format} and {image_save_format} formats.")


def batch_tda_features(
    distance_matrices: List[npt.NDArray],
    name: str,
    output_directory: Union[str, Path],
    mode: Optional[str] = None,
    metrics: List[str] = ["wasserstein", "bottleneck"],
    return_data: bool = False,
    compute_persistence: bool = True,
    compute_betti: bool = True,
    compute_entropy: bool = True,
    compute_distance: bool = True,
    compute_amplitude: bool = True,
    compute_image: bool = True,
    save_format: str = "csv",
    image_save_format: str = "png",
    persistence_diagrams_kwargs: Optional[dict] = None,
    betti_curves_kwargs: Optional[dict] = {"n_bins": 200, "n_jobs": -1},
    persistence_entropy_kwargs: Optional[dict] = {"normalize": True, "n_jobs": -1},
    pairwise_distances_kwargs: Optional[dict] = {"order": None, "n_jobs": -1},
    amplitudes_kwargs: Optional[dict] = {"order": None, "n_jobs": -1},
    persistence_images_kwargs: Optional[dict] = {"n_jobs": -1, "n_bins": 200, "sigma": 0.0005},
    save_tda_results_kwargs: Optional[dict] = {"cmap": "gray", "dpi": 300},
) -> Optional[Dict[str, npt.NDArray]]:
    """
    Compute topological features collectively for a list of distance matrices and organize results.

    This function processes a list of distance matrices collectively to compute topological
    features such as persistence diagrams, Betti curves, persistence entropy, pairwise
    distances, amplitudes, and persistence images. Results are organized into subdirectories
    and can be saved to files or returned as a dictionary.

    Args:
        distance_matrices: A list of distance matrices, where each matrix is a square
            NumPy array of shape (n_points, n_points) with non-negative values.
        name: A string identifier for the dataset or group, used as a subdirectory name.
        output_directory: The base directory (str or Path) where results will be saved.
        mode: The persistence computation method (None for Vietoris-Rips, "sparse" for sparse Rips).
        metrics: A list of distance metrics for pairwise distances and amplitudes
            (default: ["wasserstein", "bottleneck"]).
        return_data: If True, returns results as a dictionary; if False, saves to files (default: False).
        compute_persistence: If True, computes persistence diagrams (default: True).
        compute_betti: If True, computes Betti curves (default: True).
        compute_entropy: If True, computes persistence entropy (default: True).
        compute_distance: If True, computes pairwise distances (default: True).
        compute_amplitude: If True, computes amplitudes (default: True).
        compute_image: If True, computes persistence images (default: True).
        save_format: File format for tabular data ("csv", "npy", "txt"; default: "csv").
        image_save_format: File format for images ("png", "jpg", "jpeg"; default: "png").
        persistence_diagrams_kwargs: Additional parameters for persistence diagram computation (dict, optional).
        betti_curves_kwargs: Additional parameters for Betti curve computation (dict, default: {"n_bins": 200, "n_jobs": -1}).
        persistence_entropy_kwargs: Additional parameters for entropy computation (dict, default: {"normalize": True, "n_jobs": -1}).
        pairwise_distances_kwargs: Additional parameters for pairwise distance computation (dict, default: {"order": None, "n_jobs": -1}).
        amplitudes_kwargs: Additional parameters for amplitude computation (dict, default: {"order": None, "n_jobs": -1}).
        persistence_images_kwargs: Additional parameters for image computation (dict, default: {"n_jobs": -1, "n_bins": 200, "sigma": 0.0005}).
        save_tda_results_kwargs: Additional parameters for saving results, especially images (dict, default: {"cmap": "gray", "dpi": 300}).

    Returns:
        If `return_data` is True, a dictionary where keys are file paths (as strings) and
        values are NumPy arrays containing the computed topological features. Otherwise,
        None, with results saved to the specified output directory.

    Raises:
        ValueError: If any distance matrix is not square.

    Example:
        >>> import numpy as np
        >>> dist_matrices = [np.array([[0.0, 1.0], [1.0, 0.0]])]
        >>> batch_tda_features(dist_matrices, "test", "output", mode=None, return_data=False)
        Data processed and saved for test in csv and png formats.
    """
    for matrix in distance_matrices:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Each distance matrix must be square (n_points, n_points)")

    output_directory = Path(output_directory)
    output_csv_directory = output_directory / name
    output_csv_directory.mkdir(parents=True, exist_ok=True)

    # Initialize result containers
    results: Dict[str, npt.NDArray] = {}
    image_results: Dict[str, npt.NDArray] = {}
    persistence_diagrams_kwargs = persistence_diagrams_kwargs or {}

    # Compute persistence diagrams once
    persistence_diagrams = rips_persistence_diagrams(distance_matrices, mode, **persistence_diagrams_kwargs)

    for number in range(len(distance_matrices)):

        if compute_persistence:
            persistence_folder = output_csv_directory / "persistence_diagrams"
            persistence_folder.mkdir(exist_ok=True)
            results[str(persistence_folder / f"persistence_diagram_{number:03d}")] = persistence_diagrams[number]

        if compute_betti:
            betti_folder = output_csv_directory / "betti_curves"
            betti_folder.mkdir(exist_ok=True)
            betti_numbers, x_data = betti_curves(persistence_diagrams, **betti_curves_kwargs)
            results[str(betti_folder / f"betti_curve_{number:03d}")] = betti_numbers[number]
            if number == 0:
                results[str(betti_folder / "betti_x_data")] = x_data

        if compute_image:
            image_folder = output_csv_directory / "persistence_images"
            image_folder.mkdir(exist_ok=True)
            images_results = persistence_images(persistence_diagrams, **persistence_images_kwargs)
            for dim in range(images_results[number].shape[0]):
                save_path = str(image_folder / f"persistence_image_{number:03d}_H{dim}")
                image_results[save_path] = images_results[number][dim]

    if compute_entropy:
        entropy_folder = output_csv_directory / "persistence_entropy"
        entropy_folder.mkdir(exist_ok=True)
        entropy_results = persistence_entropy(persistence_diagrams, **persistence_entropy_kwargs)
        for filename, data in entropy_results.items():
            results[str(entropy_folder / filename)] = data

    if compute_distance:
        distance_folder = output_csv_directory / "pairwise_distances"
        distance_folder.mkdir(exist_ok=True)
        distance_results = diagram_distances(persistence_diagrams, metrics=metrics, **pairwise_distances_kwargs)
        for filename, data in distance_results.items():
            results[str(distance_folder / filename)] = data

    if compute_amplitude:
        amplitude_folder = output_csv_directory / "amplitudes"
        amplitude_folder.mkdir(exist_ok=True)
        amplitude_results = diagram_amplitudes(persistence_diagrams, metrics=metrics, **amplitudes_kwargs)
        for filename, data in amplitude_results.items():
            results[str(amplitude_folder / filename)] = data

    if return_data:
        return {**results, **image_results}
    else:
        save_tda_results(results, format=save_format)
        if compute_image:
            save_tda_results(image_results, format=image_save_format, **save_tda_results_kwargs)
        print(f"Data processed and saved for {name} in {save_format} and {image_save_format} formats.")


def load_tda_results(
    output_directory: Union[str, Path],
    dataset_names: Optional[List[str]] = None,
    metrics: List[str] = ["wasserstein", "bottleneck"],
    load_all: bool = True,
    load_diagrams: Optional[bool] = None,
    load_betti: Optional[bool] = None,
    load_entropy: Optional[bool] = None,
    load_distance: Optional[bool] = None,
    load_amplitude: Optional[bool] = None,
    load_image: Optional[bool] = None
) -> Dict[str, Dict[str, Union[np.ndarray, Dict[str, np.ndarray]]]]:
    """
    Load topological data analysis (TDA) results from an output directory.

    This function loads TDA results (persistence diagrams, Betti curves, entropy,
    distances, amplitudes, and images) from a specified directory, supporting both
    collective and independent data structures. It automatically detects the data
    layout and file formats, with optional filtering by dataset names and feature types.

    Args:
        output_directory: The base directory (str or Path) containing TDA results.
        dataset_names: A list of subfolder names to filter datasets (optional; if None, loads all subdirectories).
        metrics: A list of distance metrics to load for distances and amplitudes
            (default: ["wasserstein", "bottleneck"]).
        load_all: If True, loads all available data types unless overridden by specific flags (default: True).
        load_diagrams: If True, loads persistence diagrams (default: None, uses load_all).
        load_betti: If True, loads Betti curves and their x-values (default: None, uses load_all).
        load_entropy: If True, loads persistence entropy (default: None, uses load_all).
        load_distance: If True, loads pairwise distances (default: None, uses load_all).
        load_amplitude: If True, loads amplitudes (default: None, uses load_all).
        load_image: If True, loads persistence images (default: None, uses load_all).

    Returns:
        A dictionary where keys are dataset/subdirectory names and values are dictionaries
        containing loaded features. Each feature dictionary may include:
            - "persistence_diagrams": List of NumPy arrays with shape (n_features, 3).
            - "betti_curves": NumPy array with shape (n_samples, n_homology_dims, n_bins).
            - "betti_x": NumPy array with shape (n_bins,) or (n_homology_dims, n_bins).
            - "persistence_entropy": NumPy array with shape (n_samples, n_homology_dims).
            - "{metric}_distance": NumPy array with shape (n_samples, n_samples) for each metric.
            - "{metric}_amplitude": NumPy array with shape (n_samples, n_homology_dims) for each metric.
            - "persistence_images": Dictionary mapping image filenames to NumPy arrays with shape (n_bins, n_bins).

    Raises:
        ValueError: If an unsupported file format is encountered.

    Example:
        >>> results = load_tda_results("output", dataset_names=["test"], load_all=True)
        >>> print(results["test"].keys())
        dict_keys(['persistence_diagrams', 'betti_curves', 'betti_x', 'persistence_entropy', ...])
    """
    output_directory = Path(output_directory)
    supported_tabular_formats = {"csv", "npy", "txt"}
    supported_image_formats = {"png", "jpg", "jpeg"}
    data_folders = {
        "persistence_diagrams", "betti_curves", "persistence_entropy",
        "pairwise_distances", "amplitudes", "persistence_images"
    }

    def detect_format(folder: Path, pattern: str) -> Optional[str]:
        for ext in supported_tabular_formats | supported_image_formats:
            if any(folder.glob(f"{pattern}.{ext}")):
                return ext
        return None

    def load_array(file: Path, file_format: str) -> np.ndarray:
        if file_format == "csv":
            return pd.read_csv(file).to_numpy()
        elif file_format == "npy":
            return np.load(file)
        elif file_format == "txt":
            return np.loadtxt(file)
        elif file_format in supported_image_formats:
            img = Image.open(file).convert("L")
            array = np.array(img).astype(np.float32) / 255.0 
            return array
        raise ValueError(f"Unsupported file format: {file_format}")

    # Determine folders to load
    subdirs = [d for d in output_directory.iterdir() if d.is_dir()]
    if any(d.name in data_folders for d in subdirs):  # flat structure (collective)
        subdirs = [output_directory]  # treat as one dataset
    elif dataset_names:
        subdirs = [d for d in subdirs if d.name in dataset_names]

    results = {}

    for subdir in subdirs:
        label_data = {}
        name = subdir.name

        _load_diagrams = load_diagrams if load_diagrams is not None else load_all
        _load_betti = load_betti if load_betti is not None else load_all
        _load_entropy = load_entropy if load_entropy is not None else load_all
        _load_distance = load_distance if load_distance is not None else load_all
        _load_amplitude = load_amplitude if load_amplitude is not None else load_all
        _load_image = load_image if load_image is not None else load_all

        # Persistence diagrams
        if _load_diagrams and (pd_folder := subdir / "persistence_diagrams").exists():
            fmt = detect_format(pd_folder, "persistence_diagram_*")
            if fmt:
                label_data["persistence_diagrams"] = [
                    load_array(f, fmt) for f in sorted(pd_folder.glob(f"persistence_diagram_*.{fmt}"))
                ]

        # Betti curves
        if _load_betti and (betti_folder := subdir / "betti_curves").exists():
            curve_fmt = detect_format(betti_folder, "betti_curve_*")
            x_file = betti_folder / f"betti_x_data.{curve_fmt}" if curve_fmt else None

            if curve_fmt:
                curve_files = sorted(betti_folder.glob(f"betti_curve_*.{curve_fmt}"))

                # Case 1: Shared X data for all curves → no interpolation
                if x_file and x_file.exists():
                    label_data["betti_curves"] = np.array([
                        load_array(f, curve_fmt) for f in curve_files
                    ])
                    label_data["betti_x"] = load_array(x_file, curve_fmt)

                # Case 2: Individual X data per curve → interpolate to common X
                else:
                    x_fmt = detect_format(betti_folder, "betti_x_*")
                    x_files = sorted(betti_folder.glob(f"betti_x_*.{x_fmt}")) if x_fmt else []

                    if x_files and len(x_files) == len(curve_files):
                        curves = [load_array(f, curve_fmt) for f in curve_files]
                        xs = [load_array(f, x_fmt) for f in x_files]

                        max_dim, n_bins = curves[0].shape
                        xs = np.array(xs)
                        min_x = np.min(xs, axis=(0, 2))
                        max_x = np.max(xs, axis=(0, 2))
                        x_common = [np.linspace(mn, mx, n_bins) for mn, mx in zip(min_x, max_x)]

                        interpolated = []
                        for curve, x_vals in zip(curves, xs):
                            dims_interp = [
                                interp1d(x_vals[d], curve[d], kind='linear', bounds_error=False, fill_value=0.0)(x_common[d])
                                for d in range(max_dim)
                            ]
                            interpolated.append(np.array(dims_interp))

                        label_data["betti_curves"] = np.array(interpolated)
                        label_data["betti_x"] = np.array(x_common)


        # Persistence entropy
        if _load_entropy and (entropy_folder := subdir / "persistence_entropy").exists():
            fmt = detect_format(entropy_folder, "persistence_entropy_*")
            if fmt:
                # Check for single file (batch mode) or multiple files (individual mode)
                single_file = entropy_folder / f"persistence_entropy.{fmt}"
                if single_file.exists():
                    print("Heloooo")
                    label_data["persistence_entropy"] = load_array(single_file, fmt)
                else:
                    entropy_files = sorted(entropy_folder.glob(f"persistence_entropy*.{fmt}"))
                    label_data["persistence_entropy"] = np.array([load_array(f, fmt) for f in entropy_files])

        # Distances
        if _load_distance and (dist_folder := subdir / "pairwise_distances").exists():
            for metric in metrics:
                fmt = detect_format(dist_folder, f"{metric}_distance_*")
                if fmt:
                    label_data[f"{metric}_distance"] = np.array([
                        load_array(f, fmt)
                        for f in sorted(dist_folder.glob(f"{metric}_distance_*.{fmt}"))
                    ])

        # Amplitudes
        if _load_amplitude and (amp_folder := subdir / "amplitudes").exists():
            for metric in metrics:
                fmt = detect_format(amp_folder, f"{metric}_amplitude_*")
                if not fmt:
                    fmt = detect_format(amp_folder, f"{metric}_amplitude")
                files = sorted(amp_folder.glob(f"{metric}_amplitude*.{fmt}")) if fmt else []
                if files:
                    label_data[f"{metric}_amplitude"] = np.array([load_array(f, fmt) for f in files])

        # Persistence images
        if _load_image and (img_folder := subdir / "persistence_images").exists():
            fmt = detect_format(img_folder, "persistence_image_*")
            if fmt:
                label_data["persistence_images"] = {
                    file.stem: load_array(file, fmt)
                    for file in sorted(img_folder.glob(f'persistence_image_*.{fmt}'))
                }

        if label_data:
            results[name] = label_data

    return results


def load_and_interpolate_betti_curves(
    output_directory: Union[str, Path],
    dataset_names: Optional[List[str]] = None,
    num_points: int = 200
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Load Betti curves and their corresponding x-values from multiple datasets,
    compute a shared common x-axis, and interpolate all curves to this axis.

    This function processes Betti curve data stored in subdirectories, ensuring all
    curves are interpolated to a uniform x-axis for consistent analysis. It supports
    multiple file formats (CSV, NPY, TXT) and handles both flat and nested directory
    structures.

    Args:
        output_directory (Union[str, Path]): Path to the directory containing dataset
            subfolders or a flat structure with Betti curve files.
        dataset_names (Optional[List[str]], optional): List of specific dataset folder
            names to process. If None, processes all subdirectories or assumes a flat
            structure if a 'betti_curves' folder is found. Defaults to None.
        num_points (int, optional): Number of points for the interpolated x-axis.
            Must be positive. Defaults to 200.

    Returns:
        Tuple[Dict[str, np.ndarray], np.ndarray]:
            - Dictionary mapping dataset names to interpolated Betti curves with shape
              (n_samples, n_dims, num_points), where n_samples is the number of curves
              per dataset, n_dims is the number of dimensions, and num_points is the
              number of interpolated points.
            - Common x-axis array with shape (n_dims, num_points).

    Raises:
        ValueError: If output_directory doesn't exist, num_points is non-positive,
            no valid datasets are found, or file formats are unsupported.
        FileNotFoundError: If no Betti curve or x-value files are found in a dataset.
        RuntimeError: If Betti curve dimensions are inconsistent across datasets.

    Examples:
        >>> curves, x_common = load_and_interpolate_betti_curves_common_x(
        ...     output_directory="path/to/data",
        ...     dataset_names=["dataset1", "dataset2"],
        ...     num_points=100
        ... )
        >>> curves["dataset1"].shape
        (n_samples, n_dims, 100)
        >>> x_common.shape
        (n_dims, 100)
    """
    # Input validation
    output_directory = Path(output_directory)
    if not output_directory.exists():
        raise ValueError(f"Directory {output_directory} does not exist")
    if not isinstance(num_points, int) or num_points <= 0:
        raise ValueError("num_points must be a positive integer")

    supported_formats = {".csv", ".npy", ".txt"}

    def detect_format(folder: Path, pattern: str) -> Optional[str]:
        """Detect the file format for a given pattern in the folder."""
        for ext in supported_formats:
            if any(folder.glob(f"{pattern}{ext}")):
                return ext.lstrip(".")
        return None

    def load_array(file: Path, file_format: str) -> np.ndarray:
        """Load an array from a file based on its format."""
        try:
            if file_format == "csv":
                return pd.read_csv(file).to_numpy()
            elif file_format == "npy":
                return np.load(file)
            elif file_format == "txt":
                return np.loadtxt(file)
        except Exception as e:
            raise ValueError(f"Failed to load {file}: {str(e)}")
        raise ValueError(f"Unsupported file format: {file_format}")

    # Determine subdirectories to process
    subdirs = [d for d in output_directory.iterdir() if d.is_dir()]
    if any(d.name == "betti_curves" for d in subdirs):
        subdirs = [output_directory]  # Flat structure
        dataset_names = [output_directory.name] if dataset_names is None else dataset_names
    elif dataset_names:
        subdirs = [d for d in subdirs if d.name in dataset_names]
    else:
        dataset_names = [d.name for d in subdirs]

    if not subdirs:
        raise ValueError("No valid datasets found in the specified directory")

    all_curves, all_x = [], []
    n_dims = None
    n_samples_by_group = {}

    # Load Betti curves and x-values
    for subdir in subdirs:
        name = subdir.name if subdir != output_directory else dataset_names[0]
        group_dir = subdir / "betti_curves" if subdir != output_directory else subdir
        if not group_dir.exists():
            continue

        curve_format = detect_format(group_dir, "betti_curve_*")
        if not curve_format:
            raise ValueError(f"No supported Betti curve files found in {group_dir}")

        curve_files = sorted(group_dir.glob(f"betti_curve_*.{curve_format}"))
        x_files = sorted(group_dir.glob(f"betti_x_*.{curve_format}"))

        if not curve_files or not x_files:
            raise FileNotFoundError(f"No Betti curve or x-value files found in {group_dir}")

        if len(curve_files) != len(x_files):
            raise ValueError(f"Mismatch between curve ({len(curve_files)}) and x-value ({len(x_files)}) files in {group_dir}")

        curves = [load_array(f, curve_format) for f in curve_files]
        xs = [load_array(f, curve_format) for f in x_files]

        # Validate dimensions
        if n_dims is None:
            n_dims = curves[0].shape[0]
        elif any(c.shape[0] != n_dims for c in curves):
            raise RuntimeError(f"Inconsistent number of dimensions in {name} curves")

        all_curves.extend(curves)
        all_x.extend(xs)
        n_samples_by_group[name] = len(curves)

    if not all_curves:
        raise ValueError("No valid Betti curves loaded from any dataset")

    # Compute common x-axis
    all_x_arr = np.array(all_x)
    min_x = np.min(all_x_arr, axis=(0, 2))
    max_x = np.max(all_x_arr, axis=(0, 2))
    x_common = np.array([np.linspace(mn, mx, num_points) for mn, mx in zip(min_x, max_x)])

    # Interpolate curves to common x-axis
    interpolated_curves = np.zeros((len(all_curves), n_dims, num_points))
    for i, (curve, x_vals) in enumerate(zip(all_curves, all_x)):
        for d in range(n_dims):
            interpolator = interp1d(
                x_vals[d], curve[d], kind="linear", bounds_error=False, fill_value=0.0
            )
            interpolated_curves[i, d] = interpolator(x_common[d])

    # Organize results by dataset
    result = {}
    idx = 0
    for name in dataset_names:
        if name in n_samples_by_group:
            count = n_samples_by_group[name]
            result[name] = interpolated_curves[idx:idx + count]
            idx += count

    return result, x_common
    
