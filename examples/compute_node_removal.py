from __future__ import annotations
from typing import List
import os
import sys
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import neurophorm as nf


# ---------- CONFIG ----------
output_directory = Path("examples/node_removal")    
data_paths = {
    "A": "examples/groups_data_txt/Group_A",
    "B": "examples/groups_data_txt/Group_B"
}

def load_group_corrs(folder: Path) -> List[np.ndarray]:
    """Load all files as square correlation matrices in [-1, 1]."""
    corrs: List[np.ndarray] = []
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".txt"):
            continue
        fp = folder / fn
        try:
            M = np.loadtxt(fp)
            if M.ndim != 2 or M.shape[0] != M.shape[1]:
                print(f"  skip {fp} not square: {M.shape}")
                continue
            if not np.all((M >= -1) & (M <= 1)):
                print(f"  skip {fp} values outside [-1,1]")
                continue
            corrs.append(M.astype(float))
        except Exception as e:
            print(f"  error loading {fp}: {e}")
    return corrs

def main():
    output_directory.mkdir(parents=True, exist_ok=True)
    nf.configure_logging(filename=str(output_directory / "compute_node_removal.log"))

    for name, pathstr in data_paths.items():
        folder = Path(pathstr)
        if not folder.is_dir():
            print(f"[skip] {name} → folder does not exist: {folder}")
            continue

        print(f"[load] {name} from {folder}")
        corrs = load_group_corrs(folder)
        if not corrs:
            print(f"[skip] {name} → no valid 'txt' files")
            continue

        shapes = {m.shape for m in corrs}
        if len(shapes) > 1:
            print(f"[skip] {name} → inconsistent shapes: {shapes}")
            continue

        print(f"[tda ] {name} → n={len(corrs)}, shape={corrs[0].shape}")

        dists = nf.corr_to_distance_matrices(corrs)  # standard 1 - rho
        dist = np.mean(dists, axis=0)  # group average distance matrix
        diagrams = nf.node_removal_persistence(
            dist,
            output_directory=output_directory,
            return_data=True,
            persistence_diagrams_kwargs={
                "homology_dimensions": [0, 1],
                "max_edge_length": 1.0,
                "n_jobs": -1
            },
            verbose=True
        )

        nf.node_removal_differences(
            diagrams,
            output_directory=output_directory,
            output_filename=f"{name}_node_removal_distances",
            save_format="txt",
            verbose=True
        )
    return


if __name__ == "__main__":
    main()
