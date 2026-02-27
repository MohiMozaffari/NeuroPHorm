# NeuroPHorm

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](#license) ![Python](https://img.shields.io/badge/Python-3.9%2B-informational) ![Status](https://img.shields.io/badge/status-beta-yellow) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17542597.svg)](https://doi.org/10.5281/zenodo.17542597)


<p align="center">
    <img src="assets/NeuroPHorm.png" alt="NeuroPHorm Logo" width="250">
</p>

**NeuroPHorm** is a modern Python package for topological brain network analysis using persistent homology. It provides robust tools for analyzing and visualizing brain networks, focusing on topological features such as persistence diagrams, Betti curves, and pairwise distances. NeuroPHorm is ideal for neuroscientists and TDA researchers exploring the structure of brain connectivity matrices.

---

## 📋 Table of contents

1. [Key capabilities](#-key-capabilities)
2. [Why NeuroPHorm?](#-why-neurophorm)
3. [Installation](#-installation)
4. [Quick start](#-quick-start)
5. [Worked examples](#-worked-examples)
6. [API highlights](#-api-highlights)
7. [Project structure](#-project-structure)
8. [Testing & quality checks](#-testing--quality-checks)
9. [Contributing](#-contributing)
10. [License](#-license)
11. [Author](#-author)
12. [Acknowledgments](#-acknowledgments)
13. [Citation](#-citation)

---

## 🚀 Key capabilities

- **Node removal analysis** – quantify how the removal of individual nodes changes persistent topological signatures.
- **Persistence pipelines** – transform correlation matrices into distance matrices and compute
  - Persistence diagrams
  - Betti curves (per subject and across cohorts)
  - Persistence entropy
  - Pairwise Wasserstein & bottleneck distances
  - Amplitudes
  - Persistence images
- **Visualization suite** – produce publication-ready plots:
  - Cohort-level Betti summaries with standard deviation bands
  - P-value heatmaps and grouped comparisons
  - Swarm, violin, and KDE distributions
  - Node-removal dashboards
- **Flexible I/O** – load/save results in CSV, TXT, NPY, or PNG formats with consistent naming conventions.
- **Extensible architecture** – modular functions and logging-aware utilities make it easy to integrate NeuroPHorm into larger analysis pipelines.

---

## 💡 Why NeuroPHorm?

Traditional graph-theoretic measures often miss higher-order interactions encoded in brain connectivity data. NeuroPHorm bridges that gap with:

- Batteries of persistent homology descriptors ready for statistical analysis.
- Utilities to aggregate, visualise, and compare topological signatures between groups.
- Reproducible workflows backed by example datasets and notebooks.

Whether you are prototyping new biomarkers or preparing results for publication, NeuroPHorm gives you a cohesive toolkit built on top of [giotto-tda](https://github.com/giotto-ai/giotto-tda).

---

## 📦 Installation

### From source

```bash
git clone https://github.com/MohiMozaffari/NeuroPHorm.git
cd neurophorm
pip install .
```

### Install all research dependencies

```bash
pip install -r requirements.txt
```

The package targets **Python 3.9+** and relies on the scientific Python stack:

| Dependency | Minimum version |
|------------|-----------------|
| numpy      | 1.23.5          |
| pandas     | 1.5.3           |
| scipy      | 1.10.1          |
| matplotlib | 3.7.1           |
| seaborn    | 0.12.2          |
| scikit-learn | 1.2.2        |
| giotto-tda | 0.3.1           |
| Pillow     | 9.3.0           |

NeuroPHorm does not ship with GPU-specific dependencies; the default CPU installation is sufficient for the provided examples.

---

## ⚡ Quick start

This minimal example walks through computing persistence features for a cohort and plotting summary Betti curves. The example data shipped with the repository represents toy functional connectivity matrices.

```python
import os
import numpy as np
from neurophorm import (
    corr_to_distance_matrices,
    individual_tda_features,
    load_tda_results,
    plot_betti_curves,
)

# 1. Load correlation matrices from the example dataset
group_dir = "examples/groups_data_txt/Group_A"
matrices = [
    np.loadtxt(os.path.join(group_dir, f))
    for f in sorted(os.listdir(group_dir))
    if f.endswith(".txt")
]

# 2. Convert correlations to distance matrices
distance_mats = corr_to_distance_matrices(matrices)

# 3. Compute per-subject topological summaries
output_dir = "examples/persistences/A"
individual_tda_features(distance_mats, group_name="Group_A", output_directory=output_dir)

# 4. Reload results and generate a publication-ready plot
tda_results = load_tda_results(output_dir)
fig = plot_betti_curves(tda_results, title="Group A Betti curves")
fig.show()
```

💡 **Tip:** Replace the `group_dir` with your own connectivity matrices (TXT, CSV, or NPY). NeuroPHorm handles shape normalisation internally.

---

## 🧪 Worked examples

The `examples/` directory contains scripts, notebooks, and sample outputs showcasing typical workflows.

| File | Description |
|------|-------------|
| [`examples/compute_individual_tda.py`](examples/compute_individual_tda.py) | Batch-compute Betti curves, amplitudes, entropy, and distance summaries for each subject in a cohort. |
| [`examples/compute_node_removal.py`](examples/compute_node_removal.py) | Perform node-removal experiments and derive pairwise persistence distances. |
| [`examples/plot_node_removal.ipynb`](examples/plot_node_removal.ipynb) | Notebook for visualising node-removal results. |
| [`examples/plot_saved.ipynb`](examples/plot_saved.ipynb) | Plot stored TDA features and compare cohorts. |
| [`examples/p_value_selection.ipynb`](examples/p_value_selection.ipynb) | Explore p-value thresholds and statistical filtering of persistence descriptors. |

### Reproducing the node-removal pipeline

```bash
python examples/compute_node_removal.py
```

The script reads the default configuration inside the file (pointing to both `Group_A` and `Group_B`) and stores diagrams plus pairwise distances in `examples/node_removal/`. Modify the configuration block at the top of the script to analyse your own cohorts, then visualise the results through the `plot_node_removal` utilities.

---

## 🧰 API highlights

NeuroPHorm exposes a concise API through `neurophorm.__all__`. Common entry points include:

### Persistence utilities (`neurophorm.persistence`)

- `corr_to_distance_matrices` – convert correlation matrices to valid distance matrices.
- `rips_persistence_diagrams` – run Vietoris–Rips or sparse persistence on distance matrices.
- `individual_tda_features` / `batch_tda_features` – compute and store per-subject or aggregated descriptors.
- `load_tda_results` & `save_tda_results` – serialise/deserialize results to disk.
- `compute_betti_stat_features` – derive summary statistics (AUC, centroid, skewness, etc.) from Betti curves.

### Node removal (`neurophorm.node_removal`)

- `node_removal_persistence` – recompute persistence diagrams after sequentially removing nodes.
- `node_removal_differences` – evaluate how diagrams change and persist across removals.
- `load_node_removal_data` – load cached node-removal experiments with flexible aggregation options for plotting.
- `load_removal_data` – deprecated alias kept for compatibility; prefer `load_node_removal_data`.

### Visualisation (`neurophorm.visualization`)

- `plot_betti_curves` – overlay mean Betti curves with uncertainty bands.
- `plot_p_values` & `plot_grouped_p_value_heatmaps` – display statistical significance across cohorts.
- `plot_swarm_violin` / `plot_kde_dist` – inspect distributions of persistence-derived metrics.
- `plot_node_removal` – summarise node-removal persistence diagrams and differences.

Refer to the docstrings for arguments, return types, and advanced options such as sparse persistence or saving plots to disk.

---

## 🗂️ Project structure

```
NeuroPHorm/
├── neurophorm/              # Core package (persistence, node removal, visualisation, logging helpers)
├── examples/                # Example scripts, notebooks, synthetic datasets, and generated outputs
│   ├── groups_data_txt/     # Sample correlation matrices for two cohorts
│   ├── persistences/        # Example persistence outputs generated by the scripts
│   └── node_removal/        # Sample node-removal results
├── assets/                  # Logos and artwork
├── requirements.txt         # Research dependency lockstep
├── pyproject.toml / setup.py# Packaging metadata
└── README.md                # You are here
```

---

## ✅ Testing & quality checks

Basic smoke tests are included in `tests/` and run in CI. To verify your environment locally:

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run smoke checks
python -m compileall -q neurophorm examples
pytest
flake8 neurophorm
```

If you contribute additional checks (e.g., expanded unit tests, type checking), document them in your pull request so others can reproduce the setup.

---

## 🧑‍💻 Contributing

We welcome contributions of all sizes—from bug reports to new analysis pipelines. Please read the [contribution guide](CONTRIBUTING.md) for details on setting up a development environment, running tests, and coding standards. Issues and feature requests are tracked on [GitHub](https://github.com/MohiMozaffari/NeuroPHorm/issues).

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

## 📚 Citation

Use the following DOIs depending on your citation goal:

- Latest archived release (concept DOI): [10.5281/zenodo.17542597](https://doi.org/10.5281/zenodo.17542597)
- Exact `v1.0.0` release (version DOI): [10.5281/zenodo.17542598](https://doi.org/10.5281/zenodo.17542598)

Recent updates on `main` may not yet be archived on Zenodo.

For citation metadata, see [CITATION.cff](CITATION.cff).

---
