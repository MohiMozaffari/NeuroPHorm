"""
collect_pvalue_counts.py
Auto-discovers features in your data dict, computes p-value matrices using your existing
plotters (without drawing), counts unique significant entries, and returns ONE tidy DataFrame,
optionally saving to CSV.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import logging

# Local imports (adjust the module names if your package layout differs)
from neurophorm.visualization import (
    plot_p_values,
    plot_betti_stats_pvalues,
    plot_node_removal_p_values,
    _infer_dimensions,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

# Keys that are meta / not direct per-sample features
EXCLUDE_GENERIC_KEYS = {
    "persistence_diagrams",
    "betti_curves_shared",      # handled specially through plot_p_values
    "betti_curves_original",
    "betti_x_shared",
    "betti_x_list",
    "feature_names",            # Betti-stats helper
    "persistence_images",       # dict of 2D images
}

def _is_array_like(x) -> bool:
    try:
        arr = np.asarray(x)
        return isinstance(arr, np.ndarray) and arr.size > 0
    except Exception:
        return False

def _discover_generic_features(data: Dict[str, Dict]) -> List[str]:
    """
    Discover ndarray-like keys present in ALL groups, excluding meta keys.
    Adds 'betti_curves_shared' pseudo-feature if shared/original betti curves
    are present in all groups.
    """
    if not data:
        return []
    # intersect of keys across groups to avoid missing data
    key_sets = []
    for g, d in data.items():
        if not isinstance(d, dict):
            continue
        keys = set(k for k, v in d.items() if k not in EXCLUDE_GENERIC_KEYS and _is_array_like(v))
        key_sets.append(keys)
    if not key_sets:
        candidates = set()
    else:
        candidates = set.intersection(*key_sets) if len(key_sets) > 1 else key_sets[0]

    # Betti curves pseudo-feature if curves present across all groups
    have_betti = all(
        ("betti_curves_shared" in (data.get(g) or {}) or "betti_curves_original" in (data.get(g) or {}))
        for g in data.keys()
    )
    features = sorted(candidates)
    if have_betti:
        features.append("betti_curves_shared")
    return features

def _count_unique_significant(df: pd.DataFrame, alpha: float) -> int:
    """
    Count significant entries in the upper triangle (k=1), excluding NaN.
    Avoids double counting symmetric matrices; diagonal ignored.
    """
    if df.empty:
        return 0
    m = df.values
    mask = ~np.isnan(m)
    iu = np.triu_indices_from(m, k=1)
    sig = (m[iu] < alpha) & (mask[iu])
    return int(np.count_nonzero(sig))

def collect_pvalue_counts(
    *,
    # Already materialized data structures
    data: Optional[Dict[str, Dict]] = None,
    betti_stats: Optional[Dict[str, Dict]] = None,
    node_mean: Optional[pd.DataFrame] = None,
    node_se: Optional[pd.DataFrame] = None,
    atlas: Optional[np.ndarray] = None,
    # unified testing policy
    test: str = "auto",
    alpha: float = 0.05,
    multitest: Optional[str] = "fdr_bh",
    # output
    output_csv: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Build ONE tidy DataFrame of feature-selection signals by counting unique
    significant entries in the pairwise p-value matrices produced by your existing
    plotters (no figures drawn).

    Behavior
      • Generic features are auto-discovered from `data` keys and tested via `plot_p_values`.
        If Betti curves are present, Betti AUC is included as the pseudo-feature
        "betti_curves_shared" — unless Betti-stats are provided (see next).
      • If `betti_stats` is provided, Betti-stats are used via `plot_betti_stats_pvalues`
        and "betti_curves_shared" is removed from generic features to avoid duplication.
      • If `node_mean`, `node_se`, and `atlas` are provided, node-removal matrices are
        added via `plot_node_removal_p_values`.
      • Any number of labels (≥ 2) is supported. Counts are computed on the upper triangle
        (k = 1), excluding diagonal and NaNs, so symmetric matrices are not double-counted.

    Returned columns (tidy rows; one row per p-value matrix)
      Method                     {"generic","betti_stats","node_removal"}
      Subnetwork                 removed subnetwork name for node_removal else <NA>
      Feature                    feature key or Betti-stats column name
      Homology_Dimension         TRUE homology dimension from the data (e.g., 0,1,2);
                                 <NA> for node_removal; falls back to positional index
                                 if a true label cannot be inferred for Betti-stats
      Count                      number of unique significant cell pairs in the matrix
      Alpha                      α used
      Multitest                  multiple-testing method name

    Parameters
    ----------
    data : dict, optional
        label → dict of per-feature arrays (e.g., from `load_tda_results`).
    betti_stats : dict, optional
        label → {"feature_names":[...], "H0":..., "H1":..., ...}
    node_mean, node_se, atlas : optional
        Inputs for node-removal p-values.
    test : str
        Statistical test policy ("auto" delegates to the plotters).
    alpha : float
        Significance threshold used for counting.
    multitest : str or None
        Multiple-testing correction ("fdr_bh", "bonferroni", or None).
    output_csv : str or Path, optional
        If provided, save the tidy DataFrame to this CSV path.

    Returns
    -------
    pandas.DataFrame
        The tidy table described above, sorted by Count (desc), then Method, Feature, Homology_Dimension.

    Examples
    --------
    >>> df = collect_pvalue_counts(
    ...     data=tda_results_by_group,            # {"ASD": {...}, "TD": {...}, "ADHD": {...}}
    ...     alpha=0.05, multitest="fdr_bh",
    ... )
    >>> df.head()

    >>> df = collect_pvalue_counts(
    ...     data=tda_results_by_group,
    ...     betti_stats=betti_stats_by_group,     # uses Betti-stats; skips Betti AUC
    ...     node_mean=mean_df, node_se=se_df, atlas=atlas_vec,
    ...     alpha=0.01, multitest="bonferroni",
    ...     output_csv="outputs/all_pvalue_counts.csv",
    ... )
    """
    results: List[Dict] = []

    # ---------- A) generic features via plot_p_values ----------
    if data:
        labels_g = list(data.keys())
        data_g = {k: data[k] for k in labels_g if k in data}
        feats = _discover_generic_features(data_g)
        dims_g = _infer_dimensions(data_g)  # TRUE dims, e.g., [0,1,2]

        # If Betti-stats exist, do NOT also compute Betti AUC from curves.
        if betti_stats and "betti_curves_shared" in feats:
            feats = [f for f in feats if f != "betti_curves_shared"]

        for feature_name in feats:
            p_mats = plot_p_values(
                data=data_g,
                feature_name=feature_name,
                labels=labels_g,
                dimensions=dims_g,           # ensures true dims order
                test=test,
                alpha=alpha,
                multitest=multitest,
                show_plot=False,
                save_plot=False,
            )
            for dim_idx, df in enumerate(p_mats):
                cnt = _count_unique_significant(df, alpha)
                results.append({
                    "Method": "generic",
                    "Subnetwork": pd.NA,
                    "Feature": feature_name,
                    "Homology_Dimension": int(dims_g[dim_idx]),  # TRUE homology dim
                    "Count": cnt,
                    "Alpha": alpha,
                    "Multitest": multitest,
                })

    # ---------- B) Betti-stats via plot_betti_stats_pvalues ----------
    if betti_stats:
        labels_b = list(betti_stats.keys())
        data_b = {k: betti_stats[k] for k in labels_b if k in betti_stats}

        # Intersection of feature_names
        common = None
        for g, d in data_b.items():
            cols = [str(x) for x in d.get("feature_names", [])]
            common = set(cols) if common is None else (common & set(cols))
        stat_feats = sorted(common or [])

        # Try to infer TRUE homology dims from common H* keys (H0, H1, ...)
        hkeys_common = None
        for g, d in data_b.items():
            hkeys = set(k for k in d.keys() if k.upper().startswith("H") and k[1:].isdigit())
            hkeys_common = hkeys if hkeys_common is None else (hkeys_common & hkeys)
        dims_labels = sorted([int(k[1:]) for k in (hkeys_common or [])])  # e.g., [0,1,2]

        # Let the plotter infer internal ordering if needed by passing None
        dims_b = None

        for feature_name in stat_feats:
            p_mats = plot_betti_stats_pvalues(
                data=data_b,
                feature_name=feature_name,
                labels=labels_b,
                dimensions=dims_b,
                test=test,
                alpha=alpha,
                multitest=multitest,
                show_plot=False,
                save_plot=False,
            )
            for dim_idx, df in enumerate(p_mats):
                cnt = _count_unique_significant(df, alpha)
                # map positional index to true dim if available; else fall back
                true_dim = (dims_labels[dim_idx] if dim_idx < len(dims_labels) else dim_idx)
                results.append({
                    "Method": "betti_stats",
                    "Subnetwork": pd.NA,
                    "Feature": feature_name,
                    "Homology_Dimension": int(true_dim),
                    "Count": cnt,
                    "Alpha": alpha,
                    "Multitest": multitest,
                })

    # ---------- C) node-removal via plot_node_removal_p_values ----------
    if (node_mean is not None) and (node_se is not None) and (atlas is not None):
        p_mats = plot_node_removal_p_values(
            mean_df=node_mean,
            error_df=node_se,
            atlas=np.asarray(atlas),
            labels=list(node_mean.columns),
            alpha=alpha,
            multitest=multitest,
            show_plot=False,
            save_plot=False,
        )
        subnets = list(node_mean.index)
        for g, df in zip(subnets, p_mats):
            cnt = _count_unique_significant(df, alpha)
            results.append({
                "Method": "node_removal",
                "Subnetwork": g,
                "Feature": "node_removal",
                "Homology_Dimension": pd.NA,  # not applicable
                "Count": cnt,
                "Alpha": alpha,
                "Multitest": multitest,
            })

    tidy = pd.DataFrame(results, columns=[
        "Method", "Subnetwork", "Feature", "Homology_Dimension", "Count", "Alpha", "Multitest"
    ])

    # Sort by informativeness
    tidy = tidy.sort_values(
        ["Count", "Method", "Feature", "Homology_Dimension"],
        ascending=[False, True, True, True]
    ).reset_index(drop=True)

    if output_csv:
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tidy.to_csv(out_path, index=False)
        logger.info("Saved tidy counts to %s", out_path)

    return tidy
