
# NeuroPHorm

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](#license) ![Python](https://img.shields.io/badge/Python-3.9%2B-informational) ![Status](https://img.shields.io/badge/status-beta-yellow)

<p align="center">
    <img src="assets/NeuroPHorm.png" alt="NeuroPHorm Logo" width="250">
</p>

**NeuroPHorm** is a modern Python package for topological brain network analysis using persistent homology. It provides robust tools for analyzing and visualizing brain networks, focusing on topological features such as persistence diagrams, Betti curves, and pairwise distances. NeuroPHorm is ideal for neuroscientists and TDA researchers exploring the structure of brain connectivity matrices.

---

## 🚀 Features

- **Node Removal Analysis**: Study the impact of individual nodes on topological features by computing persistence diagrams after sequential node removals.
- **Persistence Analysis**: Transform correlation matrices into distance matrices and compute:
    - Persistence diagrams
    - Betti curves
    - Persistence entropy
    - Pairwise distances (Wasserstein, bottleneck)
    - Amplitudes
    - Persistence images
- **Visualization**: Generate publication-ready plots:
    - Mean Betti curves with standard deviation
    - P-value heatmaps
    - Swarm, violin, and KDE plots
    - Grouped distance heatmaps
- **Flexible Data Processing**: Batch and individual processing, with results saved in CSV, TXT, NPY, or PNG formats.
- **Extensible & Modular**: Easily extend for custom TDA analyses.

---

## 📦 Installation

Install from source:

```bash
git clone https://github.com/mohimozaffari/neurophorm.git
cd neurophorm
pip install .
```

**Dependencies:**

- numpy >= 1.23.5
- pandas >= 1.5.3
- scipy >= 1.10.1
- matplotlib >= 3.7.1
- seaborn >= 0.12.2
- scikit-learn >= 1.2.2
- gtda >= 0.3.1
- Pillow >= 9.3.0

Install all requirements:

```bash
pip install -r requirements.txt
```

---

## 🗂️ Example Usage

NeuroPHorm comes with ready-to-use example scripts and notebooks in the `examples/` directory. These demonstrate typical workflows for TDA on brain networks.

### Example Scripts

#### 1. Compute Individual TDA Features

File: [`examples/compute_individual_tda.py`](examples/compute_individual_tda.py)

Compute topological features (Betti curves, amplitudes, entropy, etc.) for each subject in a group:

```python
from neurophorm.persistence import corr_to_distance_matrices, individual_tda_features
import numpy as np
import os

group_dir = "examples/groups_data_txt/Group_A"
output_dir = "examples/persistences/A"

matrices = [np.loadtxt(os.path.join(group_dir, f)) for f in os.listdir(group_dir) if f.endswith('.txt')]
dist_matrices = corr_to_distance_matrices(matrices)
individual_tda_features(dist_matrices, "A", output_directory=output_dir)
```

#### 2. Compute Node Removal Distances

File: [`examples/compute_node_removal.py`](examples/compute_node_removal.py)

Analyze the effect of removing each node on the persistence diagram:

```python
from neurophorm.persistence import corr_to_distance_matrices, node_removal_persistence, node_removal_differences
import numpy as np
import os

group_dir = "examples/groups_data_txt/Group_B"
output_dir = "examples/node_removal"

matrices = [np.loadtxt(os.path.join(group_dir, f)) for f in os.listdir(group_dir) if f.endswith('.txt')]
dist_matrix = corr_to_distance_matrices([matrices[0]])[0]
diagrams = node_removal_persistence(dist_matrix, output_directory=output_dir, return_data=True)
node_removal_differences(diagrams, output_directory=output_dir, output_filename="B_node_removal_distances")
```

#### 3. Visualization Notebooks

- [`examples/plot_node_removal.ipynb`](examples/plot_node_removal.ipynb): Visualize node removal results.
- [`examples/plot_saved.ipynb`](examples/plot_saved.ipynb): Plot saved TDA features and compare groups.
- [`examples/p_value_selection.ipynb`](examples/p_value_selection.ipynb): Explore p-value selection and statistical analysis.

Open these notebooks in Jupyter for interactive exploration.

---

## 📁 Directory Structure

```
NeuroPHorm/
├── neurophorm/           # Core package
├── examples/             # Example scripts, notebooks, and data
│   ├── compute_individual_tda.py
│   ├── compute_node_removal.py
│   ├── plot_node_removal.ipynb
│   ├── plot_saved.ipynb
│   ├── p_value_selection.ipynb
│   └── groups_data_txt/  # Example data (Group_A, Group_B)
│   └── persistences/     # Example output
│   └── node_removal/     # Example output
├── requirements.txt
├── setup.py
├── README.md
└── assets/
```

---

## 🧑‍💻 Contributing

Contributions are welcome! Please:

1. Fork the repo and create a new branch.
2. Make your changes (PEP8 style, add tests if possible).
3. Open a pull request.

For issues or feature requests, use the [GitHub issue tracker](https://github.com/mohimozaffari/neurophorm/issues).

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.

---

## 👩‍🔬 Author

Developed by Mohaddeseh Mozaffari  
Contact: [mohaddeseh.mozaffarii@gmail.com](mailto:mohaddeseh.mozaffarii@gmail.com)

---

## 🙏 Acknowledgments

- [giotto-tda](https://github.com/giotto-ai/giotto-tda) for TDA tools
- The neuroscience community
- All contributors and users

---

## 📖 Citation

If you use NeuroPHorm in your research, please cite:

```bibtex
@software{neurophorm,
    author = {Mohaddeseh Mozaffari},
    title = {NeuroPHorm: A Python Package for Topological Brain Network Analysis},
    year = {2025},
    url = {https://github.com/mohimozaffari/neurophorm}
}
```

