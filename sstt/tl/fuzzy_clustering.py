from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence, Tuple

import gseapy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patheffects import withStroke
from scipy import sparse


def create_gene_distance_matrix(
    adata,
    distance_col: str = "normalized_dist",
    nbins: int = 50,
    distance_range: Optional[Tuple[float, float]] = None,
    segment_col: Optional[str] = None,
    layer: Optional[str] = None,
    use_raw: bool = False,
    fill_empty: float = 0.0,
):
    if distance_col not in adata.obs.columns:
        raise KeyError(f"'{distance_col}' not found in adata.obs.")

    if segment_col is not None and segment_col not in adata.obs.columns:
        raise KeyError(f"'{segment_col}' not found in adata.obs.")

    if nbins <= 0:
        raise ValueError("`nbins` must be positive.")

    X, var_names = _get_expression_matrix(
        adata,
        layer=layer,
        use_raw=use_raw,
    )

    distances = pd.to_numeric(adata.obs[distance_col], errors="coerce")
    finite_mask = np.isfinite(distances.to_numpy())

    if finite_mask.sum() == 0:
        raise ValueError(f"No finite values found in adata.obs['{distance_col}'].")

    if distance_range is None:
        dmin = float(np.nanmin(distances[finite_mask]))
        dmax = float(np.nanmax(distances[finite_mask]))
    else:
        dmin, dmax = map(float, distance_range)

    if not dmin < dmax:
        raise ValueError("`distance_range` must satisfy min < max.")

    bin_edges = np.linspace(dmin, dmax, nbins + 1)
    distance_bins = pd.IntervalIndex.from_breaks(bin_edges, closed="left")

    binned = pd.cut(
        distances,
        bins=bin_edges,
        labels=distance_bins,
        right=False,
        include_lowest=True,
    )

    # Include the rightmost endpoint.
    right_edge_mask = np.isclose(distances.to_numpy(), dmax)
    if right_edge_mask.any():
        binned = binned.astype(object)
        binned.iloc[right_edge_mask] = distance_bins[-1]

    obs_tmp = adata.obs.copy()
    obs_tmp["_distance_bin"] = pd.Categorical(
        binned,
        categories=distance_bins,
        ordered=True,
    )

    valid_cells = obs_tmp["_distance_bin"].notna().to_numpy()
    if valid_cells.sum() == 0:
        raise ValueError("No cells remain after distance binning.")

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    if segment_col is None:
        gene_distance_matrix = np.full(
            (len(var_names), len(distance_bins)),
            fill_empty,
            dtype=float,
        )

        for i, interval in enumerate(distance_bins):
            mask = (obs_tmp["_distance_bin"] == interval).to_numpy()

            if mask.sum() > 0:
                gene_distance_matrix[:, i] = _mean_expression_by_mask(X, mask)

        gene_distance_df = pd.DataFrame(
            gene_distance_matrix,
            index=var_names,
            columns=bin_centers,
        )

    else:
        segments = _get_category_order(obs_tmp[segment_col])

        multi_cols = pd.MultiIndex.from_product(
            [segments, bin_centers],
            names=[segment_col, distance_col],
        )

        gene_distance_matrix = np.full(
            (len(var_names), len(multi_cols)),
            fill_empty,
            dtype=float,
        )

        col_idx = 0
        for seg in segments:
            for interval in distance_bins:
                mask = (
                    (obs_tmp[segment_col] == seg)
                    & (obs_tmp["_distance_bin"] == interval)
                ).to_numpy()

                if mask.sum() > 0:
                    gene_distance_matrix[:, col_idx] = _mean_expression_by_mask(
                        X,
                        mask,
                    )

                col_idx += 1

        gene_distance_df = pd.DataFrame(
            gene_distance_matrix,
            index=var_names,
            columns=multi_cols,
        )

    gene_distance_df.attrs["distance_col"] = distance_col
    gene_distance_df.attrs["distance_range"] = (dmin, dmax)
    gene_distance_df.attrs["nbins"] = nbins
    gene_distance_df.attrs["bin_edges"] = bin_edges
    gene_distance_df.attrs["bin_centers"] = bin_centers

    return gene_distance_df, distance_bins, bin_centers


def filter_spatially_variable_genes(
    gene_distance_matrix: pd.DataFrame,
    min_mean: float = 0.05,
    min_amplitude: float = 0.2,
    min_std: float = 0.05,
    amplitude_quantiles: Tuple[float, float] = (5, 95),
    top_n: Optional[int] = 1000,
    return_metrics: bool = True,
):
    if gene_distance_matrix.empty:
        raise ValueError("`gene_distance_matrix` is empty.")
    q_low, q_high = amplitude_quantiles
    if not (0 <= q_low < q_high <= 100):
        raise ValueError("`amplitude_quantiles` must satisfy 0 <= low < high <= 100.")
    matrix = gene_distance_matrix.to_numpy(dtype=float)
    mean_expr = np.nanmean(matrix, axis=1)
    q_low_values = np.nanpercentile(matrix, q_low, axis=1)
    q_high_values = np.nanpercentile(matrix, q_high, axis=1)
    raw_amplitude = q_high_values - q_low_values
    raw_std = np.nanstd(matrix, axis=1)
    # Rank genes by robust amplitude and bin-to-bin variation.
    trend_score = raw_amplitude * raw_std
    metrics_df = pd.DataFrame(
        {
            "gene": gene_distance_matrix.index,
            "mean_expr": mean_expr,
            "raw_amplitude": raw_amplitude,
            "raw_std": raw_std,
            "trend_score": trend_score,
        },
        index=gene_distance_matrix.index,
    )
    keep = (
        (metrics_df["mean_expr"] >= min_mean)
        & (metrics_df["raw_amplitude"] >= min_amplitude)
        & (metrics_df["raw_std"] >= min_std)
    )
    metrics_df = metrics_df.loc[keep].sort_values("trend_score", ascending=False)
    if top_n is not None:
        metrics_df = metrics_df.head(top_n)
    filtered_matrix = gene_distance_matrix.loc[metrics_df.index].copy()
    if return_metrics:
        return filtered_matrix, metrics_df
    return filtered_matrix


def run_fuzzy_clustering(
    gene_distance_matrix: pd.DataFrame,
    n_clusters: int = 6,
    m: float = 2.0,
    max_iter: int = 150,
    error: float = 1e-4,
    seed: int = 42,
    scale_range: Tuple[float, float] = (-1.0, 1.0),
):
    try:
        from skfuzzy import cmeans
    except ImportError as exc:
        raise ImportError(
            "Please install scikit-fuzzy first: pip install scikit-fuzzy"
        ) from exc

    if gene_distance_matrix.empty:
        raise ValueError("`gene_distance_matrix` is empty.")

    if isinstance(gene_distance_matrix.columns, pd.MultiIndex):
        raise ValueError(
            "`run_fuzzy_clustering` expects a simple distance-bin matrix. "
            "If you used `segment_col`, select one segment before clustering."
        )

    matrix = gene_distance_matrix.to_numpy(dtype=float)
    normalized_matrix = _scale_rows_minmax(matrix, feature_range=scale_range)

    data = normalized_matrix.T

    cntr, u, u0, d, jm, p, fpc = cmeans(
        data,
        c=n_clusters,
        m=m,
        error=error,
        maxiter=max_iter,
        seed=seed,
    )

    assigned_clusters = np.argmax(u, axis=0)
    max_membership = np.max(u, axis=0)

    results_df = pd.DataFrame(
        {
            "gene": gene_distance_matrix.index,
            "cluster": assigned_clusters,
            "max_membership": max_membership,
        },
        index=gene_distance_matrix.index,
    )

    for i in range(n_clusters):
        results_df[f"cluster_{i}_membership"] = u[i, :]

    cluster_centers = pd.DataFrame(
        cntr,
        columns=gene_distance_matrix.columns,
        index=[f"Cluster_{i}" for i in range(n_clusters)],
    )

    results_df.attrs["fpc"] = float(fpc)
    results_df.attrs["n_clusters"] = int(n_clusters)
    results_df.attrs["m"] = float(m)

    return normalized_matrix, results_df, cluster_centers


def run_fuzzy_cluster_enrichment(
    results_df: pd.DataFrame,
    n_clusters: Optional[int] = None,
    species: str = "mouse",
    top_n_genes: int = 100,
    output_dir: str = "enrichment_results",
    pval_cutoff: float = 0.05,
    top_terms: int = 10,
    gene_col: str = "gene",
    cluster_col: str = "cluster",
    gene_sets: Optional[Dict[str, Sequence[str] | str]] = None,
    plot: bool = True,
    save_excel: bool = True,
    dpi: int = 600,
    verbose: bool = True,
) -> Dict[int, Any]:
    if gene_col not in results_df.columns:
        if "Gene" in results_df.columns:
            gene_col = "Gene"
        else:
            raise KeyError(f"'{gene_col}' not found in results_df.")

    if cluster_col not in results_df.columns:
        if "Assigned_Cluster" in results_df.columns:
            cluster_col = "Assigned_Cluster"
        else:
            raise KeyError(f"'{cluster_col}' not found in results_df.")

    membership_cols = _find_membership_columns(results_df)

    if n_clusters is None:
        n_clusters = len(membership_cols)

    if n_clusters <= 0:
        raise ValueError("`n_clusters` must be positive.")

    if len(membership_cols) < n_clusters:
        raise ValueError(
            f"Expected at least {n_clusters} membership columns, "
            f"but found {len(membership_cols)}."
        )

    species_key, organism = _normalize_species(species)

    if gene_sets is None:
        species_to_db = {
            "Mouse": {
                "go": [
                    "GO_Biological_Process_2025",
                    "GO_Cellular_Component_2025",
                    "GO_Molecular_Function_2025",
                ],
                "kegg": "KEGG_2019_Mouse",
            },
            "Human": {
                "go": [
                    "GO_Biological_Process_2025",
                    "GO_Cellular_Component_2025",
                    "GO_Molecular_Function_2025",
                ],
                "kegg": "KEGG_2021_Human",
            },
        }

        if species_key not in species_to_db:
            raise ValueError(
                f"Unsupported species: {species_key}. "
                f"Supported species: {list(species_to_db.keys())}"
            )

        gene_sets = species_to_db[species_key]

    if "go" not in gene_sets:
        raise KeyError("`gene_sets` must contain a 'go' entry.")

    os.makedirs(output_dir, exist_ok=True)

    enrichment_results = {}

    for cluster_id in range(n_clusters):
        if verbose:
            print(f"\n[Processing] Cluster {cluster_id}...")

        membership_col = _get_membership_col(results_df, cluster_id)

        cluster_df = results_df[
            results_df[cluster_col].astype(int) == int(cluster_id)
        ].copy()

        if cluster_df.empty:
            if verbose:
                print(
                    f"  [Warning] Cluster {cluster_id} has no assigned genes. Skipped."
                )
            continue

        # Prefer max_membership if available, otherwise use cluster-specific membership.
        rank_col = (
            "max_membership"
            if "max_membership" in cluster_df.columns
            else membership_col
        )

        cluster_df = cluster_df.sort_values(rank_col, ascending=False)

        if top_n_genes is not None:
            cluster_df = cluster_df.head(top_n_genes)

        cluster_genes = cluster_df[gene_col].dropna().astype(str).tolist()

        cluster_genes = list(dict.fromkeys(cluster_genes))

        if verbose:
            print(
                f"  Selected genes: {len(cluster_genes)} "
                f"from assigned Cluster {cluster_id} genes."
            )

        cluster_genes = list(dict.fromkeys(cluster_genes))

        if len(cluster_genes) == 0:
            if verbose:
                print(f"  [Warning] Cluster {cluster_id} has no valid genes. Skipped.")
            continue

        cluster_dir = os.path.join(output_dir, f"cluster_{cluster_id}")
        os.makedirs(cluster_dir, exist_ok=True)

        try:
            go_enr = gp.enrichr(
                gene_list=cluster_genes,
                gene_sets=gene_sets["go"],
                organism=organism,
                cutoff=pval_cutoff,
                outdir=None,
            )

            enrichment_results[cluster_id] = go_enr

            if go_enr.results is None or go_enr.results.empty:
                if verbose:
                    print(
                        f"  [Warning] No enrichment terms found for cluster {cluster_id}."
                    )
                continue

            if save_excel:
                excel_path = os.path.join(
                    cluster_dir,
                    f"cluster_{cluster_id}_GO_enrichment.xlsx",
                )

                with pd.ExcelWriter(excel_path) as writer:
                    for db_name in gene_sets["go"]:
                        res_df = go_enr.results[
                            go_enr.results["Gene_set"] == db_name
                        ].copy()

                        if not res_df.empty:
                            sheet_name = _safe_excel_sheet_name(db_name)
                            res_df.to_excel(writer, sheet_name=sheet_name, index=False)

                if verbose:
                    print(f"  Saved table: {os.path.basename(excel_path)}")

            if plot:
                for db_name in gene_sets["go"]:
                    term_results = go_enr.results[
                        go_enr.results["Gene_set"] == db_name
                    ].copy()

                    if term_results.empty:
                        continue

                    term_results = term_results.sort_values("Adjusted P-value")

                    db_short = _short_go_name(db_name)
                    save_path = os.path.join(
                        cluster_dir,
                        f"cluster_{cluster_id}_{db_short}_enrichment_bar.pdf",
                    )

                    _plot_enrichment_bar(
                        term_results,
                        title=f"Cluster {cluster_id}: {db_short}",
                        save_path=save_path,
                        top_n=top_terms,
                        dpi=dpi,
                        show=False,
                    )
                    plt.close()

                    if verbose:
                        print(
                            f"  Saved plot: {os.path.basename(save_path)} "
                            f"(terms={len(term_results)})"
                        )

        except Exception as e:
            if verbose:
                print(f"  [Error] Cluster {cluster_id} failed: {e}")

    if verbose:
        print(f"\n[Done] Results saved to: {output_dir}")

    return enrichment_results


def _get_expression_matrix(
    adata,
    layer: Optional[str] = None,
    use_raw: bool = False,
):
    if use_raw:
        if adata.raw is None:
            raise ValueError("`use_raw=True` but adata.raw is None.")
        X = adata.raw.X
        var_names = adata.raw.var_names
    elif layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer '{layer}' not found in adata.layers.")
        X = adata.layers[layer]
        var_names = adata.var_names
    else:
        X = adata.X
        var_names = adata.var_names

    return X, pd.Index(var_names)


def _mean_expression_by_mask(X, mask: np.ndarray) -> np.ndarray:
    X_sub = X[mask]

    if sparse.issparse(X_sub):
        mean_values = np.asarray(X_sub.mean(axis=0)).ravel()
    else:
        mean_values = np.asarray(X_sub).mean(axis=0)

    return np.asarray(mean_values).ravel()


def _get_category_order(series: pd.Series) -> list:
    if pd.api.types.is_categorical_dtype(series):
        return list(series.cat.categories)

    return sorted(series.dropna().unique().tolist())


def _scale_rows_minmax(
    matrix: np.ndarray,
    feature_range: Tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)

    min_val = np.nanmin(matrix, axis=1, keepdims=True)
    max_val = np.nanmax(matrix, axis=1, keepdims=True)

    denom = max_val - min_val
    denom[denom == 0] = 1.0

    scaled = (matrix - min_val) / denom

    low, high = feature_range
    scaled = scaled * (high - low) + low

    return scaled


def _plot_enrichment_bar(
    enrichment_df: pd.DataFrame,
    title: str,
    top_n: int = 10,
    term_col: str = "Term",
    pval_col: str = "Adjusted P-value",
    figsize: tuple[float, float] = (8, 6),
    cmap: str = "viridis",
    text_max_length: int = 65,
    save_path: Optional[str] = None,
    dpi: int = 600,
    ax: Optional[Axes] = None,
    show: bool = True,
) -> Axes:
    if enrichment_df is None or enrichment_df.empty:
        raise ValueError("`enrichment_df` is empty.")

    if term_col not in enrichment_df.columns:
        raise KeyError(f"'{term_col}' not found in enrichment_df.")

    if pval_col not in enrichment_df.columns:
        raise KeyError(f"'{pval_col}' not found in enrichment_df.")

    df = enrichment_df.copy()
    df[pval_col] = pd.to_numeric(df[pval_col], errors="coerce")
    df = df.dropna(subset=[pval_col])

    if df.empty:
        raise ValueError(f"No valid values found in '{pval_col}'.")

    df["minus_log10_padj"] = -np.log10(df[pval_col] + 1e-100)
    df = df.sort_values("minus_log10_padj", ascending=True).tail(top_n)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    values = df["minus_log10_padj"].to_numpy()
    colors = plt.get_cmap(cmap)(values / (values.max() + 0.1))

    bars = ax.barh(
        range(len(df)),
        values,
        color=colors,
        height=0.7,
    )

    for bar, term in zip(bars, df[term_col]):
        term = str(term)
        clean_term = (
            term if len(term) < text_max_length else term[: text_max_length - 3] + "..."
        )

        ax.text(
            x=0.2,
            y=bar.get_y() + bar.get_height() / 2,
            s=clean_term,
            va="center",
            ha="left",
            fontsize=12,
            color="white",
            fontweight="bold",
            path_effects=[withStroke(linewidth=2, foreground="black")],
        )

    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("-log10(adjusted P-value)", fontsize=12)
    ax.set_yticks([])
    ax.grid(axis="x", linestyle=":", alpha=0.5)

    if save_path is not None:
        ax.figure.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def _normalize_species(species: str) -> tuple[str, str]:
    species = str(species).strip().lower()

    species_alias = {
        "human": ("Human", "human"),
        "homo sapiens": ("Human", "human"),
        "hs": ("Human", "human"),
        "hsapiens": ("Human", "human"),
        "h. sapiens": ("Human", "human"),
        "mouse": ("Mouse", "mouse"),
        "mus musculus": ("Mouse", "mouse"),
        "mm": ("Mouse", "mouse"),
        "mmusculus": ("Mouse", "mouse"),
        "m. musculus": ("Mouse", "mouse"),
    }

    if species not in species_alias:
        raise ValueError(
            f"Unsupported species: {species}. "
            "Supported examples: 'human', 'mouse', 'Homo sapiens', 'Mus musculus'."
        )

    return species_alias[species]


def _find_membership_columns(results_df: pd.DataFrame) -> list[str]:
    new_cols = [
        col
        for col in results_df.columns
        if col.startswith("cluster_") and col.endswith("_membership")
    ]

    if len(new_cols) > 0:
        return sorted(
            new_cols,
            key=lambda x: int(x.split("_")[1]),
        )

    old_cols = [
        col
        for col in results_df.columns
        if col.startswith("Cluster_") and col.endswith("_Membership")
    ]

    if len(old_cols) > 0:
        return sorted(
            old_cols,
            key=lambda x: int(x.split("_")[1]),
        )

    raise KeyError(
        "No fuzzy membership columns found. Expected columns like "
        "'cluster_0_membership' or 'Cluster_0_Membership'."
    )


def _get_membership_col(results_df: pd.DataFrame, cluster_id: int) -> str:
    new_col = f"cluster_{cluster_id}_membership"
    old_col = f"Cluster_{cluster_id}_Membership"

    if new_col in results_df.columns:
        return new_col

    if old_col in results_df.columns:
        return old_col

    raise KeyError(
        f"No membership column found for cluster {cluster_id}. "
        f"Expected '{new_col}' or '{old_col}'."
    )


def _safe_excel_sheet_name(name: str) -> str:
    name = str(name)

    replacements = {
        "[": "",
        "]": "",
        ":": "_",
        "*": "_",
        "?": "_",
        "/": "_",
        "\\": "_",
    }

    for old, new in replacements.items():
        name = name.replace(old, new)

    return name[:31]


def _short_go_name(db_name: str) -> str:
    if "Biological_Process" in db_name:
        return "BP"
    if "Cellular_Component" in db_name:
        return "CC"
    if "Molecular_Function" in db_name:
        return "MF"

    return db_name
