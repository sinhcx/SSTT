from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.stats import norm


def niche_enrichment_heatmap(
    adata,
    niche_col: str = "niches",
    celltype_col: str = "celltype",
    niche_order: Optional[Sequence[str]] = None,
    celltype_order: Optional[Sequence[str]] = None,
    eps: float = 1e-4,
    enrich_clip: Optional[Tuple[float, float]] = None,
    cmap_name: str = "OrRd",
    show_only_significant: bool = True,
    niche_fontsize: int = 15,
    celltype_fontsize: int = 14,
    star_fontsize: int = 10,
    xlabel_rotation: float = 45,
    return_data: bool = False,
):
    # Fixed layout parameters.
    square_in = 0.45
    cbar_w_in = 0.18
    cbar_shrink = 0.42
    grid_linewidth = 1.0
    name_h_in = 2.0
    strip_h_in = 0.22
    name_strip_gap_in = 0.08
    left_margin_in = 0.5
    bottom_margin_in = 0.5

    enrich_mat, star_mat, niche_order, celltype_order = _extract_enrichment_data(
        adata=adata,
        niche_col=niche_col,
        celltype_col=celltype_col,
        niche_order=niche_order,
        celltype_order=celltype_order,
        eps=eps,
        enrich_clip=enrich_clip,
    )

    em_t = enrich_mat.reindex(index=celltype_order, columns=niche_order).T
    sm_t = star_mat.reindex(index=celltype_order, columns=niche_order).T

    niches = em_t.index.tolist()
    celltypes = em_t.columns.tolist()
    n_niche = len(niches)
    n_celltype = len(celltypes)

    niche_palette = _get_category_colors(
        adata=adata,
        column=niche_col,
        categories=niches,
    )
    celltype_palette = _get_category_colors(
        adata=adata,
        column=celltype_col,
        categories=celltypes,
    )

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("white")

    heat_w_in = n_celltype * square_in
    heat_h_in = n_niche * square_in
    niche_strip_w_in = square_in

    fig_w = left_margin_in + cbar_w_in + 0.6 + niche_strip_w_in + heat_w_in + 0.5
    fig_h = (
        bottom_margin_in + name_h_in + name_strip_gap_in + strip_h_in + heat_h_in + 0.5
    )

    fig = plt.figure(figsize=(fig_w, fig_h))

    y_heat_base = (
        bottom_margin_in + name_h_in + name_strip_gap_in + strip_h_in
    ) / fig_h
    h_heat_rel = heat_h_in / fig_h
    x_heat_start = (left_margin_in + cbar_w_in + 0.6 + niche_strip_w_in) / fig_w

    cbar_h_rel = h_heat_rel * cbar_shrink
    cbar_y_start = y_heat_base + (h_heat_rel - cbar_h_rel) / 2

    axes = {}
    axes["cbar"] = fig.add_axes(
        [
            left_margin_in / fig_w,
            cbar_y_start,
            cbar_w_in / fig_w,
            cbar_h_rel,
        ]
    )
    axes["niche_strip"] = fig.add_axes(
        [
            (left_margin_in + cbar_w_in + 0.6) / fig_w,
            y_heat_base,
            niche_strip_w_in / fig_w,
            h_heat_rel,
        ]
    )
    axes["heatmap"] = fig.add_axes(
        [
            x_heat_start,
            y_heat_base,
            heat_w_in / fig_w,
            h_heat_rel,
        ]
    )
    axes["celltype_strip"] = fig.add_axes(
        [
            x_heat_start,
            (bottom_margin_in + name_h_in + name_strip_gap_in) / fig_h,
            heat_w_in / fig_w,
            strip_h_in / fig_h,
        ]
    )
    axes["celltype_names"] = fig.add_axes(
        [
            x_heat_start,
            bottom_margin_in / fig_h,
            heat_w_in / fig_w,
            name_h_in / fig_h,
        ]
    )

    # Left niche color strip.
    ax = axes["niche_strip"]
    ax.pcolormesh(
        np.arange(2),
        np.arange(n_niche + 1),
        np.arange(n_niche).reshape(n_niche, 1),
        cmap=ListedColormap(niche_palette),
        edgecolors="none",
    )
    ax.hlines(
        np.arange(n_niche + 1),
        0,
        1,
        colors="black",
        lw=grid_linewidth,
        zorder=10,
    )
    ax.vlines(
        [0, 1],
        0,
        n_niche,
        colors="black",
        lw=grid_linewidth,
        zorder=10,
    )

    for i, niche in enumerate(niches):
        ax.text(
            0.5,
            i + 0.5,
            str(niche),
            ha="center",
            va="center",
            fontsize=niche_fontsize,
            fontweight="bold",
        )

    ax.invert_yaxis()
    ax.axis("off")

    # Bottom cell type color strip.
    ax = axes["celltype_strip"]
    ax.pcolormesh(
        np.arange(n_celltype + 1),
        np.arange(2),
        np.arange(n_celltype).reshape(1, n_celltype),
        cmap=ListedColormap(celltype_palette),
        edgecolors="none",
    )
    ax.hlines(
        [0, 1],
        0,
        n_celltype,
        colors="black",
        lw=grid_linewidth,
        zorder=10,
    )
    ax.vlines(
        np.arange(n_celltype + 1),
        0,
        1,
        colors="black",
        lw=grid_linewidth,
        zorder=10,
    )
    ax.set_xlim(0, n_celltype)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Main heatmap.
    ax = axes["heatmap"]
    z = em_t.values.astype(float).copy()

    if show_only_significant:
        z[sm_t.values == ""] = np.nan

    valid_vals = z[~np.isnan(z)]
    if valid_vals.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.nanmin(valid_vals))
        vmax = float(np.nanmax(valid_vals))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-6

    pcm = ax.pcolormesh(
        z,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
    )
    ax.hlines(
        [0, n_niche],
        0,
        n_celltype,
        colors="black",
        lw=1.2,
        zorder=10,
    )
    ax.vlines(
        [0, n_celltype],
        0,
        n_niche,
        colors="black",
        lw=1.2,
        zorder=10,
    )
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")

    for i in range(n_niche):
        for j in range(n_celltype):
            s = sm_t.values[i, j]
            if s != "":
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    s,
                    ha="center",
                    va="center",
                    fontsize=star_fontsize,
                )

    # Cell type names.
    ax = axes["celltype_names"]
    ax.set_xlim(0, n_celltype)
    ax.set_ylim(0, 1)

    for i, ct in enumerate(celltypes):
        ax.text(
            i + 0.5,
            1.0,
            str(ct),
            rotation=xlabel_rotation,
            ha="right" if xlabel_rotation != 90 else "center",
            va="top",
            rotation_mode="anchor",
            fontsize=celltype_fontsize,
        )

    ax.axis("off")

    # Colorbar.
    cb = plt.colorbar(pcm, cax=axes["cbar"])
    cb.ax.tick_params(labelsize=8)
    axes["cbar"].set_ylabel("Enrichment score", fontsize=9)

    if return_data:
        return enrich_mat, star_mat

    return None


def _extract_enrichment_data(
    adata,
    niche_col: str = "niches",
    celltype_col: str = "celltype",
    niche_order: Optional[Sequence[str]] = None,
    celltype_order: Optional[Sequence[str]] = None,
    eps: float = 1e-4,
    enrich_clip: Optional[Tuple[float, float]] = None,
):
    if niche_col not in adata.obs.columns:
        raise KeyError(f"'{niche_col}' not found in adata.obs.")

    if celltype_col not in adata.obs.columns:
        raise KeyError(f"'{celltype_col}' not found in adata.obs.")

    df = adata.obs[[niche_col, celltype_col]].copy()
    df = df.dropna(subset=[niche_col, celltype_col])

    if df.empty:
        raise ValueError(
            f"No valid rows found after removing missing values in "
            f"'{niche_col}' and '{celltype_col}'."
        )

    df[niche_col] = df[niche_col].astype(str)
    df[celltype_col] = df[celltype_col].astype(str)

    if niche_order is None:
        niche_order = _get_default_category_order(
            adata=adata,
            column=niche_col,
            values=df[niche_col],
        )
    else:
        niche_order = [str(x) for x in niche_order]

    if celltype_order is None:
        celltype_order = _get_default_category_order(
            adata=adata,
            column=celltype_col,
            values=df[celltype_col],
        )
    else:
        celltype_order = [str(x) for x in celltype_order]

    niche_celltype_count = (
        df.groupby([niche_col, celltype_col])
        .size()
        .unstack(fill_value=0)
        .reindex(index=niche_order, columns=celltype_order, fill_value=0)
    )

    p_celltype_given_niche = niche_celltype_count.div(
        niche_celltype_count.sum(axis=1), axis=0
    ).fillna(0)

    global_counts = (
        df[celltype_col].value_counts().reindex(celltype_order, fill_value=0)
    )
    p_celltype_global = (global_counts / global_counts.sum()).values

    enrich = np.log2(
        (p_celltype_given_niche.values + eps) / (p_celltype_global[None, :] + eps)
    )

    if enrich_clip is not None:
        enrich = np.clip(enrich, enrich_clip[0], enrich_clip[1])

    enrich_mat = pd.DataFrame(
        enrich.T,
        index=celltype_order,
        columns=niche_order,
    )

    star_mat = pd.DataFrame("", index=celltype_order, columns=niche_order)

    n_total = len(df)
    niche_sizes = df[niche_col].value_counts().reindex(niche_order, fill_value=0)

    for celltype in celltype_order:
        k_total = int(global_counts[celltype])

        for niche in niche_order:
            k_in = int(niche_celltype_count.loc[niche, celltype])
            n_in = int(niche_sizes[niche])

            k_out = k_total - k_in
            n_out = n_total - n_in

            u = k_in * (n_out - k_out) + 0.5 * (
                k_in * k_out + (n_in - k_in) * (n_out - k_out)
            )
            mu = n_in * n_out / 2
            var = (n_in * n_out / 12.0) * (n_total + 1)

            z = (u - mu - 0.5) / np.sqrt(var) if var > 0 else 0.0
            p = float(norm.sf(z))

            if p <= 1e-4:
                star_mat.loc[celltype, niche] = "****"
            elif p <= 1e-3:
                star_mat.loc[celltype, niche] = "***"
            elif p <= 1e-2:
                star_mat.loc[celltype, niche] = "**"
            elif p <= 5e-2:
                star_mat.loc[celltype, niche] = "*"

    return enrich_mat, star_mat, list(niche_order), list(celltype_order)


def _get_default_category_order(
    adata,
    column: str,
    values: pd.Series,
) -> list[str]:
    if pd.api.types.is_categorical_dtype(adata.obs[column]):
        return [str(x) for x in adata.obs[column].cat.categories]

    return sorted(values.astype(str).unique().tolist())


def _get_category_colors(
    adata,
    column: str,
    categories: Sequence[str],
) -> list:
    color_key = f"{column}_colors"

    if (
        pd.api.types.is_categorical_dtype(adata.obs[column])
        and color_key in adata.uns
        and len(adata.uns[color_key]) >= len(adata.obs[column].cat.categories)
    ):
        all_categories = [str(x) for x in adata.obs[column].cat.categories]
        color_map = dict(zip(all_categories, adata.uns[color_key]))

        return [
            color_map.get(str(cat), plt.get_cmap("tab20")(i % 20))
            for i, cat in enumerate(categories)
        ]

    base_cmap = plt.get_cmap("tab20")
    return [base_cmap(i % base_cmap.N) for i, _ in enumerate(categories)]
