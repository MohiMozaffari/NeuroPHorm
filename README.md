# NeuroPHorm

NeuroPHorm is a Python package designed for topological brain network analysis using persistent homology. It provides tools to analyze and visualize brain networks, focusing on their topological features such as persistence diagrams, Betti curves, and pairwise distances. The package is particularly suited for researchers in neuroscience and topological data analysis (TDA) who aim to explore the structural properties of brain connectivity matrices.

<p align="center">
    <img src="assets/NeuroPHorm.png" alt="Package Logo" width="300">
</p>

## Features

- **Node Removal Analysis**: Compute persistence diagrams for brain networks with sequential node removals to study the impact of individual nodes on topological features.
- **Persistence Analysis**: Transform correlation matrices into distance matrices and compute a variety of topological features, including:
  - Persistence diagrams
  - Betti curves
  - Persistence entropy
  - Pairwise distances (Wasserstein, bottleneck)
  - Amplitudes
  - Persistence images
- **Visualization**: Generate insightful plots to visualize TDA results, including:
  - Mean Betti curves with standard deviation shading
  - P-value heatmaps for statistical comparisons
  - Swarm and violin plots for feature distributions
  - Kernel density estimation (KDE) plots
  - Grouped distance heatmaps with block averages
- **Flexible Data Processing**: Support for both individual and batch processing of distance matrices, with options to save results in multiple formats (CSV, NPY, TXT, PNG).
- **Extensible Architecture**: Built with modularity in mind, allowing users to extend functionality for custom TDA analyses.

## Installation

To install NeuroPHorm, use pip:

```bash
pip install neurophorm
```

Alternatively, clone the repository and install it locally:

```bash
git clone https://github.com/mohimozaffari/neurophorm.git
cd neurophorm
pip install .
```

## Requirements

NeuroPHorm depends on the following Python packages:

- `numpy==1.23.5`
- `pandas==1.5.3`
- `scipy==1.10.1`
- `matplotlib==3.7.1`
- `seaborn==0.12.2`
- `scikit-learn==1.2.2`
- `gtda==0.3.1`
- `Pillow==9.3.0`

You can install these dependencies automatically with:

```bash
pip install -r requirements.txt
```

## Usage

Below are example scripts demonstrating how to use NeuroPHorm for node removal analysis, persistence analysis, and visualization of topological features.

### Node Removal Analysis

This example loads correlation matrices, computes persistence diagrams with node removals, and calculates distances between original and node-removed diagrams.

```python
from neurophorm.persistence import corr_to_distance_matrices, node_removal_persistence, node_removal_differences
import numpy as np
from pathlib import Path
import os

# Define output directory and data paths
output_directory = Path("output")
data_paths = {
    "con_chi": "data/con_chi",
    "asd_chi": "data/asd_chi",
    "con_ado": "data/con_ado",
    "asd_ado": "data/asd_ado",
    "con_adu": "data/con_adu",
    "asd_adu": "data/asd_adu"
}

# Process each group
for name, data_path in data_paths.items():
    data_path = Path(data_path)
    if not data_path.is_dir():
        print(f"Directory {data_path} does not exist. Skipping {name}.")
        continue

    # Load correlation matrices
    correlation_matrices = []
    for filename in os.listdir(data_path):
        if filename.startswith("corr"):
            file_path = data_path / filename
            matrix = np.loadtxt(file_path)
            if matrix.shape[0] == matrix.shape[1] and np.all((matrix >= -1) & (matrix <= 1)):
                correlation_matrices.append(matrix)

    if not correlation_matrices:
        print(f"No valid correlation matrices found in {data_path}. Skipping {name}.")
        continue

    # Average correlation matrices
    correlation_matrix = np.array(correlation_matrices).mean(axis=0)
    
    # Convert to distance matrix
    distances_matrix = corr_to_distance_matrices([correlation_matrix])[0]
    
    # Compute persistence diagrams with node removals
    max_distance = np.max(distances_matrix[~np.isinf(distances_matrix)])
    diagrams = node_removal_persistence(
        distances_matrix,
        output_directory=output_directory,
        return_data=True,
        persistence_diagrams_kwargs={"homology_dimensions": [0, 1], "max_edge_length": max_distance * 1.5},
        infinity=max_distance * 2.0
    )
    
    # Compute node removal differences
    node_removal_differences(
        diagrams,
        output_directory=output_directory,
        output_filename=f"{name}_node_removal_distances",
        save_format="txt"
    )
    print(f"Completed processing for {name}")
```

### Persistence Analysis

This example computes topological features for individual distance matrices using `individual_tda_features`.

```python
from neurophorm.persistence import corr_to_distance_matrices, individual_tda_features
import numpy as np
from pathlib import Path
import os

# Define output directory and data paths
output_directory = Path("output/pos")
data_paths = {
    "con_chi": "data/con_chi",
    "asd_chi": "data/asd_chi"
}

# Process each group
for name, data_path in data_paths.items():
    data_path = Path(data_path)
    if not data_path.is_dir():
        print(f"Directory {data_path} does not exist. Skipping {name}.")
        continue

    # Load correlation matrices
    correlation_matrices = []
    for filename in os.listdir(data_path):
        if filename.startswith("corr"):
            file_path = data_path / filename
            matrix = np.loadtxt(file_path)
            if matrix.shape[0] == matrix.shape[1] and np.all((matrix >= -1) & (matrix <= 1)):
                correlation_matrices.append(matrix)

    if not correlation_matrices:
        print(f"No valid correlation matrices found in {data_path}. Skipping {name}.")
        continue

    # Convert to distance matrices
    distance_matrices = corr_to_distance_matrices(correlation_matrices, mode="positive")
    
    # Compute TDA features
    max_distance = max(np.max(m[~np.isinf(m)]) for m in distance_matrices)
    individual_tda_features(
        distance_matrices,
        name,
        output_directory=output_directory,
        mode="sparse",
        persistence_diagrams_kwargs={"homology_dimensions": [0], "max_edge_length": max_distance * 1.5},
        save_format="txt"
    )
    print(f"Completed TDA feature computation for {name}")
```

### Batch TDA Analysis

This example computes topological features collectively for a list of distance matrices using `batch_tda_features`.

```python
from neurophorm.persistence import corr_to_distance_matrices, batch_tda_features
import numpy as np
from pathlib import Path
import os

# Define output directory and data paths
output_directory = Path("output")
data_paths = {
    "con_chi": "data/con_chi",
    "asd_chi": "data/asd_chi"
}

# Load correlation matrices from all groups
correlation_matrices = []
for name, data_path in data_paths.items():
    data_path = Path(data_path)
    if not data_path.is_dir():
        continue
    for filename in os.listdir(data_path):
        if filename.startswith("corr"):
            file_path = data_path / filename
            matrix = np.loadtxt(file_path)
            if matrix.shape[0] == matrix.shape[1] and np.all((matrix >= -1) & (matrix <= 1)):
                correlation_matrices.append(matrix)

if not correlation_matrices:
    print("No valid correlation matrices found. Exiting.")
    exit(1)

# Convert to distance matrices
distance_matrices = corr_to_distance_matrices(correlation_matrices)

# Compute TDA features
max_distance = max(np.max(m[~np.isinf(m)]) for m in distance_matrices)
batch_tda_features(
    distance_matrices,
    "all",
    output_directory=output_directory,
    persistence_diagrams_kwargs={"homology_dimensions": [0], "max_edge_length": max_distance * 1.5},
    save_format="txt",
    compute_persistence=False,
    compute_distance=True
)
print("Completed TDA feature computation for all groups")
```

### Visualization

This example loads TDA results and generates visualizations such as Betti curves and p-value heatmaps.

```python
from neurophorm.persistence import load_tda_results
from neurophorm.visualization import plot_betti_curves, plot_p_values

# Load TDA results
results = load_tda_results(
    output_directory="output",
    names=["con_chi", "asd_chi"],
    load_all=True
)

# Plot mean Betti curves
plot_betti_curves(
    data=results,
    labels=["con_chi", "asd_chi"],
    output_directory="output/plots",
    save_plot=True,
    save_format="png"
)

# Plot p-value heatmaps for Betti curves
p_values = plot_p_values(
    data=results,
    feature_name="betti_curves",
    labels=["con_chi", "asd_chi"],
    output_directory="output/plots",
    test="auto",
    save_plot=True,
    save_format="png"
)
```

## Directory Structure

Organize your data and output directories as follows:

```
neurophorm_project/
├── data/
│   ├── con_chi/
│   │   ├── corr1.txt
│   │   ├── corr2.txt
│   ├── asd_chi/
│   │   ├── corr1.txt
│   │   ├── corr2.txt
│   └── ...
├── output/
│   ├── persistence_diagrams/
│   ├── betti_curves/
│   ├── pairwise_distances/
│   └── ...
├── scripts/
│   ├── node_removal.py
│   ├── persistence_analysis.py
│   └── visualization.py
└── assets/
    └── NeuroPHorm.png
```

Correlation matrices should be stored as text files (e.g., `corr1.txt`) in group-specific subdirectories, with values in [-1, 1] and square shape.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please ensure your code follows PEP 8 style guidelines and includes tests where applicable. Report issues or suggest features via the [issue tracker](https://github.com/mohimozaffari/neurophorm/issues).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

Developed by Mohaddeseh Mozaffari. For inquiries, contact [mohaddeseh.mozaffarii@gmail.com](mailto:mohaddeseh.mozaffarii@gmail.com).

## Acknowledgments

- The [giotto-tda](https://github.com/giotto-ai/giotto-tda) library for providing robust TDA tools.
- The neuroscience community for inspiring this work.
- Contributors and users who provide feedback to improve NeuroPHorm.

## Citation

If you use NeuroPHorm in your research, please cite it as:

```
@software{neurophorm,
  author = {Mohaddeseh Mozaffari},
  title = {NeuroPHorm: A Python Package for Topological Brain Network Analysis},
  year = {2025},
  url = {https://github.com/mohimozaffari/neurophorm}
}
```

