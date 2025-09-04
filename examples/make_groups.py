"""
make_groups.py
Generate two groups (A and B), each with 10 random 25x25 correlation matrices,
and save them as plain text (.txt) files for later use.
"""

import numpy as np
from pathlib import Path

OUTDIR = Path("examples/groups_data_txt")
(OUTDIR / "Group_A").mkdir(parents=True, exist_ok=True)
(OUTDIR / "Group_B").mkdir(parents=True, exist_ok=True)


def make_group(n_samples: int = 10, n_nodes: int = 25, seed: int = 0):
    """Create a group of symmetric correlation-like matrices in [-1,1]."""
    rng = np.random.default_rng(seed)
    mats = []
    for _ in range(n_samples):
        X = rng.normal(size=(n_nodes, n_nodes))
        M = (X + X.T) / 2.0                # symmetrize
        M /= (np.max(np.abs(M)) + 1e-9)    # scale into [-1,1]
        np.fill_diagonal(M, 1.0)           # diag=1 for correlation
        mats.append(M)
    return np.array(mats)


def main():
    # Group A: 10 random correlation matrices
    A = make_group(n_samples=10, n_nodes=25, seed=1)

    # Group B: perturbed version of A
    rng = np.random.default_rng(42)
    B = A + 0.1 * rng.normal(size=A.shape)
    B = np.clip(B, -1.0, 1.0)
    for m in B:
        np.fill_diagonal(m, 1.0)

    # Save each sample as a separate TXT file
    for i, mat in enumerate(A):
        np.savetxt(OUTDIR / "Group_A" / f"A_sample_{i:02d}.txt", mat, fmt="%.6f")

    for i, mat in enumerate(B):
        np.savetxt(OUTDIR / "Group_B" / f"B_sample_{i:02d}.txt", mat, fmt="%.6f")

    print(f"Saved Group A and Group B matrices (10x 25x25 each) as .txt in {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
