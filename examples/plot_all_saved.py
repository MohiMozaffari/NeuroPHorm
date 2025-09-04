"""
plot_all_saved.py
Load previously saved TDA outputs (from individual_tda_features) for groups A & B
and save ALL supported plots: Betti curves, p-values, KDE, swarm/violin, grouped heatmaps.
"""

from __future__ import annotations
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from neurophorm.persistence import load_tda_results
from neurophorm import visualization as vis
import neurophorm as nf
# ---------- CONFIG ----------
output_directory = Path("examples/plots")     # where figures will be saved
OUTDIR = Path("examples/persistences")   # where data is loaded from
SAVE_FMT = "png"
LABEL_STYLES = {
    "A": ("#1f77b4", "-"),
    "B": ("#ff7f0e", "--"),
}
HOMOLOGY_DIMS = [0, 1]   # adjust to [0,1,2] if you computed H2 as well

def main():
    nf.configure_logging(filename=str(output_directory / "individual_plots.log"))

    # load all features (betti, entropy, amplitudes, distances, images if available)
    data = load_tda_results(
        output_directory=OUTDIR,
        load_all=False,
        load_betti=True,
        load_entropy=True,
        load_amplitude=True,
    )

    labels = list(data.keys())

    # 1. Mean Betti curves
    vis.plot_betti_curves(
        data,
        dimensions=HOMOLOGY_DIMS,
        labels=labels,
        label_styles=LABEL_STYLES,
        output_directory=output_directory,
        save_plot=True,
        show_plot=False,
        save_format=SAVE_FMT,
        same_size=True,
    )

    # 2. p-value heatmaps from Betti AUCs
    p_betti = vis.plot_p_values(
        data,
        feature_name="betti_curves",
        labels=labels,
        dimensions=HOMOLOGY_DIMS,
        output_directory=output_directory,
        save_plot=True,
        show_plot=False,
        save_format=SAVE_FMT,
    )

    # 3. Persistence entropy plots
    vis.plot_kde_dist(
        data,
        feature_name="persistence_entropy",
        labels=labels,
        label_styles=LABEL_STYLES,
        output_directory=output_directory,
        save_plot=True,
        show_plot=False,
        save_format=SAVE_FMT,
    )
    vis.plot_swarm_violin(
        data,
        feature_name="persistence_entropy",
        labels=labels,
        dimensions=[0],
        label_styles=LABEL_STYLES,
        output_directory=output_directory,
        save_plot=True,
        show_plot=False,
        save_format=SAVE_FMT,
    )
    p_entropy = vis.plot_p_values(
        data,
        feature_name="persistence_entropy",
        labels=labels,
        dimensions=HOMOLOGY_DIMS,
        output_directory=output_directory,
        save_plot=True,
        show_plot=False,
        save_format=SAVE_FMT,
    )

    # 4. Amplitudes if available
    for amp in ["bottleneck_amplitude", "wasserstein_amplitude"]:
        vis.plot_swarm_violin(
            data,
            feature_name=amp,
            dimensions=HOMOLOGY_DIMS,
            labels=labels,
            label_styles=LABEL_STYLES,
            output_directory=output_directory,
            save_plot=True,
            show_plot=False,
            save_format=SAVE_FMT,
        )
        vis.plot_p_values(
            data,
            feature_name=amp,
            labels=labels,
            dimensions=HOMOLOGY_DIMS,
            output_directory=output_directory,
            save_plot=True,
            show_plot=False,
            save_format=SAVE_FMT,
        )

    print(f"Plots saved under {output_directory.resolve()}")

if __name__ == "__main__":
    main()
