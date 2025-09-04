"""
compute_individual_tda.py
Load correlation TXT files per-group, compute TDA with individual_tda_features
and save them in examples/persistences.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import List
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import neurophorm as nf


# ---------- CONFIG ----------
output_directory = Path("examples/persistences")     # where figures will be saved
data_paths = {
    "A": "examples/groups_data_txt/Group_A",
    "B": "examples/groups_data_txt/Group_B"
}

# persistence backend + params
VR_KW = dict(homology_dimensions=(0, 1, 2), n_jobs=-1)  # keep dims light for plotting
BETTI_KW = dict(n_bins=80, n_jobs=-1)
PE_KW = dict(normalize=True, n_jobs=-1)



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
    nf.configure_logging(filename=str(output_directory / "individual_ppersistence.log"))

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
        nf.individual_tda_features(
        dists,
        name=name,
        output_directory=output_directory,   # not used since return_data=True
        compute_persistence=False,
        compute_betti=True,
        compute_entropy=True,
        compute_amplitude=True,
        persistence_diagrams_kwargs=VR_KW,
        betti_curves_kwargs=BETTI_KW,
        persistence_entropy_kwargs=PE_KW,
        )
    return


if __name__ == "__main__":
    main()
