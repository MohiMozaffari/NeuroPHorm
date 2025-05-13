from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import numpy as np
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

def compute_distance_matrices(
    correlation_matrices: List[npt.NDArray],
    mode: Optional[str] = None
) -> List[npt.NDArray]:
    """
    Compute distance matrices from correlation matrices using different modes.

    Args:
        correlation_matrices: List of correlation matrices as numpy arrays
        mode: Mode for computing distances. Options are:
            - None: Standard distance (1 - correlation) (default)
            - "positive": Keep only positive correlations
            - "negative": Keep only negative correlations

    Returns:
        List of distance matrices as numpy arrays

    Raises:
        ValueError: If mode is not None, 'positive', or 'negative'
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


def compute_persistence_diagrams(
    distance_matrices: List[npt.NDArray],
    mode: Optional[str] = None,
    **kwargs
) -> List[npt.NDArray]:
    """
    Compute persistence diagrams from distance matrices.

    Args:
        distance_matrices: List of distance matrices as numpy arrays (shape: (n_points, n_points))
        mode: Mode for computing persistence diagrams. Options are:
            - None: Standard Vietoris-Rips persistence (default)
            - sparse: Sparse Rips persistence
        **kwargs: Additional arguments for persistence computation:
            - homology_dimensions: Homology dimensions to compute
            - n_jobs: Number of jobs for parallel computation (int or None)
            - collapse_edges: Collapse edges in sparse filtrations (bool, SparseRipsPersistence only)
            - max_edge_length: Maximum edge length (float, SparseRipsPersistence only)
            - approx_mode: Approximation mode (str, SparseRipsPersistence only)

    Returns:
        List of persistence diagrams, one per distance matrix (shape: (n_features, 3))

    Raises:
        ValueError: If mode is invalid or distance matrices are not square
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


def compute_betti_curves(
    persistence_diagrams: List[npt.NDArray],
    **kwargs
) -> Tuple[List[npt.NDArray], npt.NDArray]:
    """
    Compute Betti curves from persistence diagrams.

    Args:
        persistence_diagrams: List of persistence diagrams (shape: (n_features, n_homology_dims))
        **kwargs: Additional arguments for Betti curve computation:
            - n_bins: Number of bins for discretizing the filtration (int)
            - sampling: Custom sampling points for the curves (np.ndarray)

    Returns:
        Tuple of (list of Betti curves (shape: (n_homology_dims, n_bins)), x-values (shape: (n_bins,)))
    """
    betti_curve = BettiCurve(**kwargs)
    betti_numbers = betti_curve.fit_transform(persistence_diagrams)
    fig = betti_curve.plot(persistence_diagrams)
    x_data = [trace.x for trace in fig.data]
    return betti_numbers, x_data


def compute_persistence_entropy(
    persistence_diagrams: List[npt.NDArray],
    **kwargs
) -> Dict[str, npt.NDArray]:
    """
    Compute persistence entropy from persistence diagrams.

    Args:
        persistence_diagrams: List of persistence diagrams (shape: (n_features, 3))
        **kwargs: Additional arguments for entropy computation:
            - normalize: Whether to normalize the entropy values (bool)
            - nan_fill_value: Value to replace NaNs in entropy calculations (float)

    Returns:
        Dictionary with persistence entropy values (shape: (n_samples, n_homology_dims))
    """
    pe = PersistenceEntropy(**kwargs)
    entropy = pe.fit_transform(np.array(persistence_diagrams))
    return {"persistence_entropy": entropy}


def compute_pairwise_distances(
    persistence_diagrams: List[npt.NDArray],
    metrics: List[str] = ["wasserstein", "bottleneck"],
    kwargs: Optional[Dict] = {"order":None, "n_jobs": -1}
) -> Dict[str, npt.NDArray]:
    """
    Compute pairwise distances between persistence diagrams.

    Args:
        persistence_diagrams: List of persistence diagrams (shape: (n_features, 3))
        metrics: List of metrics to compute distances (default: ["wasserstein", "bottleneck"])
        kwargs: Additional arguments for distance computation:
            - order: Order of the distance metric (int or None)
            - n_jobs: Number of jobs for parallel computation (int or None)

    Returns:
        Dictionary of pairwise distances for each metric and homology dimension
        (shape: (n_samples, n_samples))
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


def compute_amplitudes(
    persistence_diagrams: List[npt.NDArray],
    metrics: List[str] = ["wasserstein", "bottleneck"],
    **kwargs
) -> Dict[str, npt.NDArray]:
    """
    Compute amplitudes of persistence diagrams.

    Args:
        persistence_diagrams: List of persistence diagrams (shape: (n_features, 3))
        metrics: List of metrics to compute amplitudes (default: ["wasserstein", "bottleneck"])
        **kwargs: Additional arguments for amplitude computation:
            - order: Order of the distance metric (int or None)
            - n_jobs: Number of jobs for parallel computation (int or None)

    Returns:
        Dictionary of amplitudes for each metric (shape: (n_samples, n_homology_dims))
    """
    amplitudes_dict = {}
    diagrams = np.array(persistence_diagrams)
    for metric in metrics:
        amplitude = Amplitude(metric=metric, **kwargs)
        amplitudes = amplitude.fit_transform(diagrams)
        amplitudes_dict[f"{metric}_amplitude"] = amplitudes
    return amplitudes_dict


def compute_persistence_images(
    persistence_diagrams: List[npt.NDArray],
    **kwargs
) -> List[npt.NDArray]:
    """
    Compute persistence images from persistence diagrams.

    Args:
        persistence_diagrams: List of persistence diagrams (shape: (n_features, 3))
        **kwargs: Additional arguments for image computation:
            - sigma: Gaussian kernel width for the image (float)
            - n_bins: Number of bins per axis for the image (int)
            - weight_function: Custom weight function for persistence points (callable)

    Returns:
        List of persistence images (shape: (n_homology_dims, n_bins, n_bins))
    """
    pi = PersistenceImage(**kwargs)
    images = pi.fit_transform(np.array(persistence_diagrams))
    return [images[i] for i in range(len(persistence_diagrams))]


def save_results(
    data_dict: Dict[str, npt.NDArray],
    overwrite: bool = True,
    format: str = "csv",
    **kwargs
) -> None:
    """
    Save results to different file formats.

    Args:
        data_dict: Dictionary with file paths as keys and data as values
        overwrite: Whether to overwrite existing files (default: True)
        format: File format to save ("csv", "npy", "txt", "png", "jpg", "jpeg")
        cmap: Colormap for saving persistence images
        vmin: Minimum value for image normalization
        vmax: Maximum value for image normalization
        **kwargs: Additional arguments passed to the save function
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
                    pd.DataFrame(data).to_csv(file_path, index=False,)
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


def compute_persistence(
    distance_matrices: List[npt.NDArray],
    name: str,
    output_directory: Union[str, Path],
    mode: Optional[str] = None,
    metrics: List[str] = ["wasserstein", "bottleneck"],
    return_data: bool = False,
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
    save_results_kwargs: Optional[dict] = {"cmap": "gray", "dpi": 300},
) -> Optional[Dict[str, npt.NDArray]]:
    """
    ...
    Computes and optionally saves various topological features from a list of distance matrices.
    This function processes a list of distance matrices to compute persistence diagrams and 
    other topological features such as Betti curves, persistence entropy, pairwise distances, 
    amplitudes, and persistence images. The results can be saved to disk in specified formats 
    or returned as a dictionary.
    Parameters:
        distance_matrices (List[npt.NDArray]): A list of square distance matrices (n_points, n_points).
        name (str): A name for the output directory where results will be saved.
        output_directory (Union[str, Path]): The base directory where results will be stored.
        mode (Optional[str]): The mode for computing persistence diagrams (e.g., "sparse").
        metrics (List[str]): Metrics for pairwise distances and amplitudes (default: ["wasserstein", "bottleneck"]).
        return_data (bool): If True, returns the computed data as a dictionary instead of saving to disk.
        compute_betti (bool): If True, computes Betti curves.
        compute_entropy (bool): If True, computes persistence entropy.
        compute_distance (bool): If True, computes pairwise distances between persistence diagrams.
        compute_amplitude (bool): If True, computes amplitudes of persistence diagrams.
        compute_image (bool): If True, computes persistence images.
        save_format (str): Format for saving tabular data (default: "csv").
        image_save_format (str): Format for saving image data (default: "png").
        persistence_diagrams_kwargs (Optional[dict]): Additional arguments for computing persistence diagrams.
        betti_curves_kwargs (Optional[dict]): Additional arguments for computing Betti curves 
            (default: {"n_bins": 200, "n_jobs": -1}).
        persistence_entropy_kwargs (Optional[dict]): Additional arguments for computing persistence entropy 
            (default: {"normalize": True, "n_jobs": -1}).
        pairwise_distances_kwargs (Optional[dict]): Additional arguments for computing pairwise distances 
            (default: {"order": None, "n_jobs": -1}).
        amplitudes_kwargs (Optional[dict]): Additional arguments for computing amplitudes 
            (default: {"order": None, "n_jobs": -1}).
        persistence_images_kwargs (Optional[dict]): Additional arguments for computing persistence images 
            (default: {"n_jobs": -1, "n_bins": 200, "sigma": 0.0005}).
        save_results_kwargs (Optional[dict]): Additional arguments for saving results 
            (default: {"cmap": "gray", "dpi": 300}).
    Returns:
        Optional[Dict[str, npt.NDArray]]: A dictionary containing computed results if `return_data` is True. 
        Otherwise, results are saved to disk, and the function returns None.
    Raises:
        ValueError: If any distance matrix is not square (n_points, n_points).
    Notes:
        - The function creates subdirectories within the specified output directory for each type of result.
        - If `return_data` is False, results are saved in the specified formats (`save_format` and `image_save_format`).
        - The function supports parallel computation for certain features using the `n_jobs` parameter in kwargs.
    Example:
        >>> results = compute_persistence(
        ...     distance_matrices=[matrix1, matrix2],
        ...     name="example",
        ...     output_directory="results",
        ...     return_data=True
        ... )
    """
    # Validate distance matrices
    for matrix in distance_matrices:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Each distance matrix must be square (n_points, n_points)")

    output_directory = Path(output_directory)
    output_csv_directory = output_directory / name
    output_csv_directory.mkdir(parents=True, exist_ok=True)

    # Set default empty dictionaries if None provided
    persistence_diagrams_kwargs = persistence_diagrams_kwargs or {}

    # Compute persistence diagrams
    persistence_diagrams = compute_persistence_diagrams(distance_matrices, mode, **persistence_diagrams_kwargs)

    # Initialize result dicts
    results: Dict[str, npt.NDArray] = {}
    image_results: Dict[str, npt.NDArray] = {}

    # Define folder paths
    persistence_folder = output_csv_directory / "persistence_diagrams"
    betti_folder = output_csv_directory / "betti_curves"
    image_folder = output_csv_directory / "persistence_images"
    entropy_folder = output_csv_directory / "persistence_entropy"
    distance_folder = output_csv_directory / "pairwise_distances"
    amplitude_folder = output_csv_directory / "amplitudes"

    if compute_betti:
        betti_numbers, x_data = compute_betti_curves(persistence_diagrams, **betti_curves_kwargs)

    if compute_image:
        persistence_images = compute_persistence_images(persistence_diagrams, **persistence_images_kwargs)

    # Save per-matrix results
    for number in range(len(distance_matrices)):
        persistence_folder.mkdir(exist_ok=True)
        results[str(persistence_folder / f"persistence_diagram_{number:03d}")] = persistence_diagrams[number]

        if compute_betti:
            betti_folder.mkdir(exist_ok=True)
            results[str(betti_folder / f"betti_curve_{number:03d}")] = betti_numbers[number]
            if number == 0:
                results[str(betti_folder / "betti_x_data")] = x_data

        if compute_image:
            image_folder.mkdir(exist_ok=True)
            for dim in range(persistence_images[number].shape[0]):
                save_path = str(image_folder / f"persistence_image_{number:03d}_H{dim}")
                image_results[save_path] = persistence_images[number][dim]

    # Save aggregate results
    if compute_entropy:
        entropy_folder.mkdir(exist_ok=True)
        entropy_results = compute_persistence_entropy(persistence_diagrams, **persistence_entropy_kwargs)
        for filename, data in entropy_results.items():
            results[str(entropy_folder / filename)] = data

    if compute_distance:
        distance_folder.mkdir(exist_ok=True)
        distance_results = compute_pairwise_distances(persistence_diagrams, metrics=metrics, **pairwise_distances_kwargs)
        for filename, data in distance_results.items():
            results[str(distance_folder / filename)] = data

    if compute_amplitude:
        amplitude_folder.mkdir(exist_ok=True)
        amplitude_results = compute_amplitudes(persistence_diagrams, metrics=metrics, **amplitudes_kwargs)
        for filename, data in amplitude_results.items():
            results[str(amplitude_folder / filename)] = data

    # Return or Save
    if return_data:
        return {**results, **image_results}
    else:
        # Save tabular data
        save_results(results, format=save_format)
        # Save image data
        if compute_image:
            save_results(image_results, format=image_save_format, **save_results_kwargs)
        print(f"Data processed and saved for {name} in {save_format} and {image_save_format} formats.")


def load_results(
    output_directory: Union[str, Path],
    dataset_names: Union[str, List[str]] = None,
    metrics: List[str] = ["wasserstein", "bottleneck"],
    load_diagrams: bool = False,
    load_betti: bool = False,
    load_entropy: bool = False,
    load_distance: bool = False,
    load_amplitude: bool = False,
    load_image: bool = False
) -> Dict[str, Dict[str, npt.NDArray]]:
    """
    Load processed TDA data by detecting file formats in folders.

    Args:
        output_directory: Path to the main output folder
        dataset_names: Single dataset name (str) or list of dataset names (List[str]) to load.
                       If None or empty, loads all dataset subdirectories or treats
                       output_directory as a single dataset if no dataset subdirectories exist.
        metrics: List of metrics for distance and amplitude data
        load_diagrams: Whether to load persistence diagrams
        load_betti: Whether to load Betti curves
        load_entropy: Whether to load persistence entropy
        load_distance: Whether to load pairwise distances
        load_amplitude: Whether to load amplitudes
        load_image: Whether to load persistence images

    Returns:
        Dictionary containing loaded data organized by dataset names
    """
    supported_tabular_formats = {"csv", "npy", "txt"}
    supported_image_formats = {"png", "jpg", "jpeg"}
    # Known data folders that should not be treated as dataset names
    data_folders = {"persistence_diagrams", "betti_curves", "persistence_entropy", "pairwise_distances", "amplitudes", "persistence_images"}

    def detect_format(folder: Path, pattern: str) -> Optional[str]:
        """Detect the file format in the folder by checking extensions."""
        for ext in supported_tabular_formats | supported_image_formats:
            if any(folder.glob(f"{pattern}.{ext}")):
                return ext
        return None

    def load_array(file: Path, file_format: str) -> np.ndarray:
        """Load array based on file format."""
        if file_format == "csv":
            return pd.read_csv(file).to_numpy()
        elif file_format == "npy":
            return np.load(file)
        elif file_format == "txt":
            return np.loadtxt(file)
        elif file_format in {"png", "jpg", "jpeg"}:
            return plt.imread(file)
        raise ValueError(f"Unsupported file format: {file_format}")

    output_directory = Path(output_directory)
    dataset_results = {}

    # Handle dataset_names input
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]  # Convert single string to list
    elif dataset_names is None or len(dataset_names) == 0:
        # Check for subdirectories, excluding known data folders
        subdirs = [d for d in output_directory.iterdir() if d.is_dir() and d.name not in data_folders]
        if subdirs:
            # Load all dataset subdirectories
            dataset_names = [d.name for d in subdirs]
        else:
            # No dataset subdirectories; treat output_directory as a single dataset
            # Check if it contains any data folders
            if any((output_directory / data_folder).exists() for data_folder in data_folders):
                dataset_names = [output_directory.name]
            else:
                dataset_names = []  # No data to process

    for dataset_name in dataset_names:
        dataset = {}
        # Use output_directory directly if dataset_name is its name
        dataset_path = output_directory if dataset_name == output_directory.name else output_directory / dataset_name

        if not dataset_path.is_dir():
            continue  # Skip if the dataset path is not a directory

        if load_diagrams and (persistence_folder := dataset_path / "persistence_diagrams").exists():
            file_format = detect_format(persistence_folder, "persistence_diagram_*")
            if file_format in supported_tabular_formats:
                dataset['persistence_diagrams'] = np.array([
                    load_array(file, file_format)
                    for file in sorted(persistence_folder.glob(f"persistence_diagram_*.{file_format}"))
                ])

        if load_betti and (betti_folder := dataset_path / "betti_curves").exists():
            file_format = detect_format(betti_folder, "betti_curve_*")
            if file_format in supported_tabular_formats:
                dataset['betti_curves'] = np.array([
                    load_array(file, file_format)
                    for file in sorted(betti_folder.glob(f"betti_curve_*.{file_format}"))
                ])
                betti_x_file = betti_folder / f"betti_x_data.{file_format}"
                if betti_x_file.exists():
                    dataset['betti_x'] = load_array(betti_x_file, file_format)

        if load_entropy and (entropy_folder := dataset_path / "persistence_entropy").exists():
            file_format = detect_format(entropy_folder, "persistence_entropy")
            if file_format in supported_tabular_formats:
                entropy_file = entropy_folder / f"persistence_entropy.{file_format}"
                if entropy_file.exists():
                    dataset['persistence_entropy'] = load_array(entropy_file, file_format)

        if load_distance and (distance_folder := dataset_path / "pairwise_distances").exists():
            for metric in metrics:
                file_format = detect_format(distance_folder, f"{metric}_distance_*")
                if file_format in supported_tabular_formats:
                    dataset[f"{metric}_distance"] = np.array([
                        load_array(file, file_format)
                        for file in sorted(distance_folder.glob(f"{metric}_distance_*.{file_format}"))
                    ])

        if load_amplitude and (amplitude_folder := dataset_path / "amplitudes").exists():
            for metric in metrics:
                file_format = detect_format(amplitude_folder, f"{metric}_amplitude")
                if file_format in supported_tabular_formats:
                    amplitude_file = amplitude_folder / f"{metric}_amplitude.{file_format}"
                    if amplitude_file.exists():
                        dataset[f"{metric}_amplitude"] = load_array(amplitude_file, file_format)

        if load_image and (image_folder := dataset_path / "persistence_images").exists():
            file_format = detect_format(image_folder, "persistence_image_*")
            if file_format:
                dataset['persistence_images'] = {
                    file.stem: load_array(file, file_format)
                    for file in sorted(image_folder.glob(f'persistence_image_*.{file_format}'))
                }

        if dataset:  # Only add non-empty datasets
            dataset_results[dataset_name] = dataset

    return dataset_results