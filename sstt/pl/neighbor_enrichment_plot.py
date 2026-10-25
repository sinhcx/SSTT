from __future__ import annotations

import math
import re
import warnings
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, TwoSlopeNorm, to_rgba
from matplotlib.patches import Patch, Rectangle
from scipy.spatial import cKDTree


__all__ = [
    "plot_component_distribution",
    "plot_component_closure_index",
    "plot_multihop_distribution_from_membership",
    "plot_multihop_enrichment_heatmap",
    "plot_observed_multihop_enrichment_heatmap",
    "plot_component_distance_ridges",
    "compute_cellwise_spatial_density",
    "plot_density_pair_panel",
    "plot_multihop_gene_trajectory",
    "plot_multihop_genes_trajectory",
]


_REQUIRED_COLUMNS = {
    "observed_celltype",
    "randomized_celltype",
    "hop",
    "enrichment",
    "qvalue",
}


def plot_component_distribution(
    adata,
    membership,
    observed_celltype,
    cell_id_col="cell_id",
    celltype_col="observed_celltype",
    component_col="component_id",
    hop_col="hop_label",
    target_label="Target",
    component_ids=None,
    spatial_key="spatial",
    spot_size=2,
    img_key=None,
    figsize=(8, 8),
    show_component_labels=True,
    label_prefix="C",
    label_size=8,
    label_color="black",
    label_bold=True,
    label_bbox=True,
    label_bbox_alpha=0.7,
    background_color="#D9D9D9",
    frameon=False,
    return_adata=False,
):
    adata_plot = adata.copy()
    comp_df = membership[
        (membership[celltype_col] == observed_celltype)
        & (membership[hop_col] == target_label)
    ].copy()
    if component_ids is not None:
        comp_df = comp_df[comp_df[component_col].isin(component_ids)]
    if comp_df.empty:
        raise ValueError("No component found.")
    labels = np.full(adata_plot.n_obs, "background", dtype=object)
    for cid, group in comp_df.groupby(component_col):
        cells = group[cell_id_col].values
        mask = adata_plot.obs_names.isin(cells)
        labels[mask] = f"{label_prefix}{cid}"
    plot_key = f"{observed_celltype}_component"
    adata_plot.obs[plot_key] = labels
    components = sorted(comp_df[component_col].unique().tolist())
    categories = [f"{label_prefix}{x}" for x in components]
    categories.append("background")
    adata_plot.obs[plot_key] = (
        adata_plot.obs[plot_key].astype("category").cat.reorder_categories(categories)
    )
    colors = sc.pl.palettes.default_20
    palette = [colors[i % len(colors)] for i in range(len(components))]
    palette.append(background_color)
    fig, ax = plt.subplots(figsize=figsize)
    sc.pl.spatial(
        adata_plot,
        color=plot_key,
        spot_size=spot_size,
        img_key=img_key,
        palette=palette,
        frameon=frameon,
        add_outline=False,
        ax=ax,
        show=False,
    )
    if show_component_labels:
        for cid, group in comp_df.groupby(component_col):
            cells = group[cell_id_col].values
            coords = adata_plot[cells, :].obsm[spatial_key]
            center = coords.mean(axis=0)
            text_kwargs = dict(
                fontsize=label_size, color=label_color, ha="center", va="center"
            )
            if label_bold:
                text_kwargs["fontweight"] = "bold"
            if label_bbox:
                text_kwargs["bbox"] = dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    edgecolor="none",
                    alpha=label_bbox_alpha,
                )
            ax.text(center[0], center[1], f"{label_prefix}{cid}", **text_kwargs)
    ax.set_title(f"{observed_celltype} connected components", fontsize=14)
    if return_adata:
        return adata_plot
    return ax


def plot_component_closure_index(
    adata,
    membership,
    observed_celltype,
    component_summary=None,
    cell_id_col="cell_id",
    celltype_col="observed_celltype",
    component_col="component_id",
    target_filter_col="hop_label",
    target_label="Target",
    closure_col="closure_index",
    plot_key=None,
    spot_size=2,
    img_key=None,
    cmap="viridis",
    na_color="#D9D9D9",
    vmin=None,
    vmax=None,
    figsize=(8, 8),
    frameon=False,
    return_adata=False,
):
    if component_summary is None:
        if "multihop_component_summary" not in adata.uns:
            raise ValueError(
                'adata.uns["multihop_component_summary"] not found. Please provide component_summary explicitly.'
            )
        component_summary = adata.uns["multihop_component_summary"]
    adata_plot = adata.copy()
    if plot_key is None:
        plot_key = f"{observed_celltype}_{closure_col}"
    mem = membership.copy()
    if celltype_col not in mem.columns:
        raise ValueError(f"{celltype_col!r} not found in membership.")
    if component_col not in mem.columns:
        raise ValueError(f"{component_col!r} not found in membership.")
    if cell_id_col not in mem.columns:
        raise ValueError(f"{cell_id_col!r} not found in membership.")
    if target_filter_col not in mem.columns:
        raise ValueError(
            f"{target_filter_col!r} not found in membership. Please set target_filter_col correctly."
        )
    mem = mem[
        (mem[celltype_col] == observed_celltype)
        & (mem[target_filter_col] == target_label)
    ].copy()
    if mem.empty:
        raise ValueError(
            f"No Target cells found for observed_celltype={observed_celltype!r}."
        )
    comp = component_summary.copy()
    required_cols = [celltype_col, component_col, closure_col]
    missing = [x for x in required_cols if x not in comp.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns in component_summary: {missing}")
    comp = comp[comp[celltype_col] == observed_celltype][
        [component_col, closure_col]
    ].copy()
    if comp.empty:
        raise ValueError(
            f"No component summary found for observed_celltype={observed_celltype!r}."
        )
    mem = mem.merge(comp, on=component_col, how="left")
    closure_values = pd.Series(np.nan, index=adata_plot.obs_names, dtype=float)
    valid_mask = mem[cell_id_col].isin(adata_plot.obs_names)
    if valid_mask.sum() != mem.shape[0]:
        missing_cells = mem.loc[~valid_mask, cell_id_col].head().tolist()
        raise ValueError(
            f"{(~valid_mask).sum()} cell_id not found in adata.obs_names. Examples: {missing_cells}"
        )
    closure_values.loc[mem[cell_id_col].values] = mem[closure_col].values
    adata_plot.obs[plot_key] = closure_values.values
    fig, ax = plt.subplots(figsize=figsize)
    sc.pl.spatial(
        adata_plot,
        color=plot_key,
        spot_size=spot_size,
        img_key=img_key,
        cmap=cmap,
        na_color=na_color,
        vmin=vmin,
        vmax=vmax,
        frameon=frameon,
        add_outline=False,
        ax=ax,
        show=False,
    )
    ax.set_title(f"{observed_celltype} component {closure_col}", fontsize=14)
    if return_adata:
        return adata_plot
    return ax


def plot_multihop_distribution_from_membership(
    adata,
    membership,
    observed_celltype,
    tie_policy="first",
    groups=None,
    max_hop=None,
    spot_size=2,
    img_key=None,
    figsize=(8, 8),
    frameon=False,
    include_outside=False,
    palette=None,
    inhop_cmap="Blues",
    outhop_cmap="Reds",
    inhop_range=(0.45, 0.9),
    outhop_range=(0.45, 0.9),
    target_color="#B238E0",
    bridge_color="#17A34A",
    closure_bridge_color="#F39C12",
    outside_color="#D9D9D9",
    ambiguous_color="#808080",
    title=None,
    ax=None,
    return_data=False,
    show=False,
):
    aggregated = _aggregate_membership_for_plot(
        membership=membership,
        observed_celltype=observed_celltype,
        tie_policy=tie_policy,
    )
    cell_indices = aggregated["cell_index"].to_numpy(dtype=np.int64)
    invalid = (cell_indices < 0) | (cell_indices >= adata.n_obs)
    if invalid.any():
        raise IndexError("`membership.cell_index` contains indices outside `adata`.")
    expected_ids = adata.obs_names[cell_indices].astype(str).to_numpy()
    actual_ids = aggregated["cell_id"].astype(str).to_numpy()
    if not np.array_equal(expected_ids, actual_ids):
        raise ValueError(
            "`membership.cell_index` and `membership.cell_id` do not match `adata.obs_names`."
        )
    adata_plot = adata.copy()
    labels = np.full(adata.n_obs, "outside", dtype=object)
    labels[cell_indices] = aggregated["hop_label"].astype(str).to_numpy()
    plot_key = f"__{observed_celltype}_stage2_aggregated_hop"
    unique_labels = pd.unique(labels)
    if max_hop is None:
        max_hop = _infer_max_hop(unique_labels)
    category_order = _ordered_multihop_groups(
        max_hop=max_hop, include_outside=True, include_ambiguous=True
    )
    category_order = [
        category for category in category_order if category in unique_labels
    ]
    adata_plot.obs[plot_key] = pd.Categorical(
        labels, categories=category_order, ordered=True
    )
    if palette is None:
        palette = _make_multihop_signed_palette(
            max_hop=max_hop,
            inhop_cmap=inhop_cmap,
            outhop_cmap=outhop_cmap,
            inhop_range=inhop_range,
            outhop_range=outhop_range,
            target_color=target_color,
            bridge_color=bridge_color,
            closure_bridge_color=closure_bridge_color,
            outside_color=outside_color,
            ambiguous_color=ambiguous_color,
        )
    if groups is None:
        groups = _ordered_multihop_groups(
            max_hop=max_hop, include_outside=include_outside, include_ambiguous=True
        )
        groups = [group for group in groups if group in category_order]
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    sc.pl.spatial(
        adata_plot,
        color=plot_key,
        groups=groups,
        palette=palette,
        spot_size=spot_size,
        img_key=img_key,
        frameon=frameon,
        add_outline=False,
        ax=ax,
        show=False,
    )
    if title is None:
        title = f"{observed_celltype} multi-hop spatial distribution"
    ax.set_title(title, fontsize=14)
    diagnostics = _build_plot_diagnostics(
        membership=membership,
        aggregated=aggregated,
        observed_celltype=observed_celltype,
    )
    if show:
        plt.show()
    if return_data:
        return {
            "ax": ax,
            "palette": palette,
            "adata_plot": adata_plot,
            "plot_key": plot_key,
            "aggregated": aggregated,
            "diagnostics": diagnostics,
        }
    return (ax, palette)


def plot_multihop_enrichment_heatmap(
    enrichment_df,
    observed_celltypes=None,
    randomized_celltypes=None,
    hop_order=None,
    observed_key="observed_celltype",
    randomized_key="randomized_celltype",
    hop_key="hop",
    value_key="enrichment",
    qvalue_key="qvalue",
    qvalue_cutoff=0.05,
    clip_quantile=0.95,
    cmap="RdBu_r",
    center=0,
    vmin=None,
    vmax=None,
    na_color="#F2F2F2",
    nonsig_color="#B8B8B8",
    nonsig_line_fraction=0.22,
    cell_padding=0.055,
    hop_gap=0.012,
    cell_edgecolor="#B7B7B7",
    cell_linewidth=0.75,
    figsize=None,
    title=None,
    ylabel="Observed cell type",
    cbar_label="log$_2$ enrichment",
    clean_labels=True,
    xtick_rotation=45,
    title_fontsize=15,
    label_fontsize=12,
    tick_fontsize=10,
    show_hop_label=True,
    show_significance_label=True,
    save_path=None,
    dpi=300,
    show=True,
):
    required = {observed_key, randomized_key, hop_key, value_key}
    if qvalue_cutoff is not None:
        required.add(qvalue_key)
    missing = required - set(enrichment_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not 0 < clip_quantile <= 1:
        raise ValueError("clip_quantile must be in the range (0, 1].")
    if qvalue_cutoff is not None and (not 0 <= qvalue_cutoff <= 1):
        raise ValueError("qvalue_cutoff must be in the range [0, 1].")
    if not 0 < nonsig_line_fraction <= 1:
        raise ValueError("nonsig_line_fraction must be in the range (0, 1].")
    data = enrichment_df.copy()
    available_observed = list(dict.fromkeys(data[observed_key]))
    available_randomized = list(dict.fromkeys(data[randomized_key]))
    if observed_celltypes is None:
        observed_celltypes = available_observed
    else:
        observed_celltypes = list(observed_celltypes)
        missing_observed = set(observed_celltypes) - set(available_observed)
        if missing_observed:
            raise ValueError(
                f"Observed cell types not found: {sorted(missing_observed)}"
            )
    if randomized_celltypes is None:
        randomized_celltypes = available_randomized
    else:
        randomized_celltypes = list(randomized_celltypes)
        missing_randomized = set(randomized_celltypes) - set(available_randomized)
        if missing_randomized:
            raise ValueError(
                f"Randomized cell types not found: {sorted(missing_randomized)}"
            )
    hop_order = _expand_heatmap_hop_order(hop_order)
    if not observed_celltypes:
        raise ValueError("No observed cell types are available for plotting.")
    if not randomized_celltypes:
        raise ValueError("No randomized cell types are available for plotting.")
    if not hop_order:
        raise ValueError("No hop values are available for plotting.")
    data = data[
        data[observed_key].isin(observed_celltypes)
        & data[randomized_key].isin(randomized_celltypes)
        & data[hop_key].isin(hop_order)
    ].copy()
    full_index = pd.MultiIndex.from_product(
        [observed_celltypes, randomized_celltypes, hop_order],
        names=[observed_key, randomized_key, hop_key],
    )
    aggregation = {value_key: "mean"}
    if qvalue_key in data.columns:
        aggregation[qvalue_key] = "min"
    plot_data = (
        data.groupby([observed_key, randomized_key, hop_key], observed=True)
        .agg(aggregation)
        .reindex(full_index)
    )
    n_rows = len(observed_celltypes)
    n_cols = len(randomized_celltypes)
    n_hops = len(hop_order)
    has_negative_hops = any((hop < 0 for hop in hop_order))
    has_positive_hops = any((hop > 0 for hop in hop_order))
    show_signed_separator = has_negative_hops and has_positive_hops
    negative_hop_count = sum((hop < 0 for hop in hop_order))
    values = plot_data[value_key].to_numpy(dtype=float)
    values = values.reshape(n_rows, n_cols, n_hops)
    if qvalue_key in plot_data.columns:
        qvalues = plot_data[qvalue_key].to_numpy(dtype=float)
        qvalues = qvalues.reshape(n_rows, n_cols, n_hops)
    else:
        qvalues = np.full_like(values, np.nan)
    finite_mask = np.isfinite(values)
    if qvalue_cutoff is None:
        significant_mask = finite_mask.copy()
    else:
        significant_mask = finite_mask & np.isfinite(qvalues)
        significant_mask &= qvalues <= qvalue_cutoff
    scale_values = values[significant_mask]
    if scale_values.size == 0:
        scale_values = values[finite_mask]
    if vmin is None and vmax is None:
        if scale_values.size == 0:
            limit = 1.0
        else:
            limit = np.quantile(np.abs(scale_values - center), clip_quantile)
            if not np.isfinite(limit) or limit <= 0:
                limit = 1.0
        vmin = center - limit
        vmax = center + limit
    elif vmin is None:
        vmin = 2 * center - vmax
    elif vmax is None:
        vmax = 2 * center - vmin
    if not vmin < center < vmax:
        raise ValueError("vmin, center, and vmax must satisfy vmin < center < vmax.")
    if figsize is None:
        figsize = (max(9, 0.72 * n_cols + 2.8), max(4.5, 0.72 * n_rows + 2.0))
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 0.025], wspace=0.08)
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])
    cmap_obj = plt.get_cmap(cmap).copy()
    norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
    inner_width = 1 - 2 * cell_padding
    inner_height = 1 - 2 * cell_padding
    tile_width = (inner_width - hop_gap * (n_hops - 1)) / n_hops
    if tile_width <= 0:
        raise ValueError("cell_padding or hop_gap is too large for the number of hops.")
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            x0 = col_idx
            y0 = row_idx
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    1,
                    1,
                    facecolor="white",
                    edgecolor=cell_edgecolor,
                    linewidth=cell_linewidth,
                )
            )
            for hop_idx in range(n_hops):
                value = values[row_idx, col_idx, hop_idx]
                is_significant = significant_mask[row_idx, col_idx, hop_idx]
                tile_x = x0 + cell_padding + hop_idx * (tile_width + hop_gap)
                tile_y = y0 + cell_padding
                if not np.isfinite(value):
                    ax.add_patch(
                        Rectangle(
                            (tile_x, tile_y),
                            tile_width,
                            inner_height,
                            facecolor=na_color,
                            edgecolor="none",
                        )
                    )
                    continue
                if is_significant:
                    ax.add_patch(
                        Rectangle(
                            (tile_x, tile_y),
                            tile_width,
                            inner_height,
                            facecolor=cmap_obj(norm(value)),
                            edgecolor="none",
                        )
                    )
                    continue
                line_width = tile_width * nonsig_line_fraction
                line_x = tile_x + (tile_width - line_width) / 2
                ax.add_patch(
                    Rectangle(
                        (line_x, tile_y),
                        line_width,
                        inner_height,
                        facecolor=nonsig_color,
                        edgecolor="none",
                    )
                )
            if show_signed_separator:
                separator_x = (
                    x0
                    + cell_padding
                    + negative_hop_count * tile_width
                    + (negative_hop_count - 0.5) * hop_gap
                )
                ax.plot(
                    [separator_x, separator_x],
                    [y0 + cell_padding, y0 + 1 - cell_padding],
                    color="#4D4D4D",
                    linewidth=0.8,
                    linestyle=(0, (2.0, 1.6)),
                    solid_capstyle="butt",
                    zorder=5,
                )
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)

    def format_label(label):
        label = str(label)
        return label.replace("_", " ") if clean_labels else label

    ax.set_xticklabels(
        [format_label(x) for x in randomized_celltypes],
        rotation=xtick_rotation,
        ha="right" if xtick_rotation else "center",
        rotation_mode="anchor",
        fontsize=tick_fontsize,
    )
    ax.set_yticklabels(
        [format_label(x) for x in observed_celltypes], fontsize=tick_fontsize
    )
    ax.xaxis.tick_bottom()
    ax.tick_params(
        axis="x",
        top=False,
        bottom=True,
        labeltop=False,
        labelbottom=True,
        length=0,
        pad=7,
    )
    ax.tick_params(axis="y", left=True, length=0, pad=8)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_aspect("auto")
    for spine in ax.spines.values():
        spine.set_visible(False)
    scalar_mappable = ScalarMappable(norm=norm, cmap=cmap_obj)
    scalar_mappable.set_array([])
    cbar = fig.colorbar(scalar_mappable, cax=cax)
    cbar.ax.set_title(cbar_label, fontsize=label_fontsize, pad=10)
    cbar.set_ticks([vmin, center, vmax])
    cbar.set_ticklabels([f"{vmin:.2g}", f"{center:g}", f"{vmax:.2g}"])
    cbar.ax.tick_params(labelsize=tick_fontsize, length=3, width=0.7)
    cbar.outline.set_linewidth(0.7)
    bottom = 0.22 if xtick_rotation else 0.12
    annotation = []
    if show_hop_label:
        if show_signed_separator:
            max_out_hop = max((hop for hop in hop_order if hop > 0))
            annotation.append(f"InHop1  →  OutHop{max_out_hop}")
        else:
            first_hop = hop_order[0]
            last_hop = hop_order[-1]
            if first_hop < 0 and last_hop < 0:
                annotation.append(
                    f"Within each celltype pair: InHop{abs(first_hop)}  →  InHop{abs(last_hop)}"
                )
            else:
                annotation.append(
                    f"Within each celltype pair: Hop{first_hop}  →  Hop{last_hop}"
                )
    if show_significance_label and qvalue_cutoff is not None:
        annotation.append(
            f"Colored: q_value ≤ {qvalue_cutoff:g}; gray: q_value > {qvalue_cutoff:g}"
        )
    has_annotation = bool(annotation)
    top = 0.82 if has_annotation else 0.88
    fig.subplots_adjust(left=0.13, right=0.93, bottom=bottom, top=top)
    if title:
        fig.suptitle(title, fontsize=title_fontsize, y=0.985)
    if has_annotation:
        annotation_y = 0.915 if title else 0.965
        fig.text(
            0.5,
            annotation_y,
            "    |    ".join(annotation),
            ha="center",
            va="center",
            fontsize=tick_fontsize,
        )
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return (fig, ax)


def plot_observed_multihop_enrichment_heatmap(
    enrichment_df,
    observed_celltype,
    neighbor_celltypes: Sequence[str] | None = None,
    hop_order=None,
    qvalue_cutoff=0.05,
    min_enrichment=0.0,
    drop_without_enrichment=True,
    exclude_self=False,
    sort_mode="peak_hop",
    title=None,
    cmap="RdBu_r",
    center=0.0,
    clip_quantile=0.95,
    vmin=None,
    vmax=None,
    inactive_color="white",
    show_values=False,
    value_format=".2f",
    show_group_separators=False,
    grid_color="white",
    grid_linewidth=0.8,
    separator_color="black",
    separator_linewidth=0.8,
    xtick_rotation=45,
    figsize=None,
    tick_fontsize=9,
    annotation_fontsize=7,
    cbar_label="Log2 enrichment",
    save_path=None,
    dpi=300,
    show=True,
    return_data=False,
):
    if not 0 < clip_quantile <= 1:
        raise ValueError("`clip_quantile` must be in the range (0, 1].")
    prepared = _prepare_observed_multihop_profile(
        enrichment_df=enrichment_df,
        observed_celltype=observed_celltype,
        neighbor_celltypes=neighbor_celltypes,
        hop_order=hop_order,
        qvalue_cutoff=qvalue_cutoff,
        min_enrichment=min_enrichment,
        drop_without_enrichment=drop_without_enrichment,
        exclude_self=exclude_self,
        sort_mode=sort_mode,
    )
    values = prepared["value_matrix"].to_numpy(dtype=float)
    finite_mask = prepared["finite_mask"]
    significant_mask = prepared["significant_mask"]
    resolved_hops = prepared["hop_order"]
    signed_hops = prepared["signed_hops"]
    n_rows, n_hops = values.shape
    finite_values = values[finite_mask]
    if finite_values.size == 0:
        raise ValueError("No finite enrichment values are available for plotting.")
    scale_values = values[significant_mask]
    if scale_values.size == 0:
        scale_values = finite_values
    if vmin is None and vmax is None:
        limit = float(np.quantile(np.abs(scale_values - center), clip_quantile))
        if not np.isfinite(limit) or limit <= 0:
            limit = 1.0
        vmin, vmax = (center - limit, center + limit)
    elif vmin is None:
        vmin = 2 * center - float(vmax)
    elif vmax is None:
        vmax = 2 * center - float(vmin)
    if not vmin < center < vmax:
        raise ValueError(
            "`vmin`, `center`, and `vmax` must satisfy vmin < center < vmax."
        )
    if figsize is None:
        figsize = (max(5.0, 0.55 * n_hops + 2.7), max(3.2, 0.34 * n_rows + 1.8))
    fig, ax = plt.subplots(figsize=figsize)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
    plot_cmap = plt.get_cmap(cmap)
    rgba = np.empty((n_rows, n_hops, 4), dtype=float)
    rgba[...] = to_rgba(inactive_color)
    rgba[significant_mask] = plot_cmap(norm(values[significant_mask]))
    ax.imshow(
        rgba,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=(-0.5, n_hops - 0.5, n_rows - 0.5, -0.5),
        zorder=1,
    )
    if show_values:
        for row_index in range(n_rows):
            for col_index in range(n_hops):
                if significant_mask[row_index, col_index]:
                    value = values[row_index, col_index]
                    ax.text(
                        col_index,
                        row_index,
                        format(value, value_format),
                        ha="center",
                        va="center",
                        fontsize=annotation_fontsize,
                        zorder=4,
                    )
    ax.set_xticks(np.arange(n_hops))
    ax.set_xticklabels(
        [_format_hop(value, signed=signed_hops) for value in resolved_hops],
        rotation=xtick_rotation,
        ha="right" if xtick_rotation else "center",
        rotation_mode="anchor",
        fontsize=tick_fontsize,
    )
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(prepared["row_order"], fontsize=tick_fontsize)
    ax.set_xlabel("Hop")
    ax.set_ylabel("Neighbor cell type")
    ax.set_title(
        title or f"{prepared['observed_celltype']} multi-hop neighborhood enrichment"
    )
    x_boundaries = np.arange(-0.5, n_hops + 0.5, 1.0)
    y_boundaries = np.arange(-0.5, n_rows + 0.5, 1.0)
    ax.vlines(
        x_boundaries,
        ymin=-0.5,
        ymax=n_rows - 0.5,
        colors=grid_color,
        linewidth=grid_linewidth,
        antialiased=False,
        zorder=3,
    )
    ax.hlines(
        y_boundaries,
        xmin=-0.5,
        xmax=n_hops - 0.5,
        colors=grid_color,
        linewidth=grid_linewidth,
        antialiased=False,
        zorder=3,
    )
    if signed_hops:
        negative_hop_count = sum((hop < 0 for hop in resolved_hops))
        signed_boundary = negative_hop_count - 0.5
        ax.vlines(
            signed_boundary,
            ymin=-0.5,
            ymax=n_rows - 0.5,
            colors=separator_color,
            linewidth=separator_linewidth,
            linestyles=(0, (3.0, 2.2)),
            zorder=6,
        )
    ax.set_xlim(-0.5, n_hops - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.tick_params(axis="both", length=0)
    if show_group_separators and n_rows > 1:
        sort_groups = prepared["row_metadata"]["sort_hop_index"].to_numpy()
        boundaries = np.flatnonzero(sort_groups[1:] != sort_groups[:-1]) + 0.5
        ax.hlines(
            boundaries,
            xmin=-0.5,
            xmax=n_hops - 0.5,
            colors=separator_color,
            linewidth=separator_linewidth,
            zorder=5,
        )
    scalar_mappable = ScalarMappable(norm=norm, cmap=plot_cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax, fraction=0.045, pad=0.025)
    colorbar.set_label(cbar_label)
    if signed_hops:
        hop_text = f"Hops: InHop{abs(resolved_hops[0])} → InHop1 | OutHop1 → OutHop{resolved_hops[-1]}"
    elif resolved_hops[-1] < 0:
        hop_text = f"Hops: InHop{abs(resolved_hops[0])} → InHop{abs(resolved_hops[-1])}"
    else:
        hop_text = f"Hops: Hop{resolved_hops[0]} → Hop{resolved_hops[-1]}"
    q_text = f"q ≤ {qvalue_cutoff:g}"
    annotation_text = f"{hop_text}; rows: ordered by {sort_mode.replace('_', ' ')}; colored: {q_text}; blank: non-significant or unavailable"
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    axes_box = ax.get_position()
    fig.text(
        axes_box.x0,
        0.018,
        annotation_text,
        ha="left",
        va="bottom",
        fontsize=max(6, tick_fontsize - 1),
    )
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    if return_data:
        return (fig, ax, prepared)
    return (fig, ax)


def plot_component_distance_ridges(
    distance_data: pd.DataFrame,
    group_col: str = "observed_celltype",
    celltype_col: str = "neighbor_celltype",
    signed_distance_col: str = "signed_physical_distance",
    distance_col: str = "physical_distance",
    side_col: str = "side",
    component_col: str = "component_uid",
    distance_range: tuple[float, float] = (-50, 200),
    aggregation_mode: str = "local",
    group_order: Sequence[str] | None = None,
    celltype_order: Sequence[str] | None = None,
    colors: Mapping[str, Any] | Sequence[Any] | None = None,
    bandwidth: float | str = "scott",
    grid_size: int = 500,
    min_cells_per_component: int = 5,
    min_cells_pool: int = 5,
    min_components_per_group: int = 1,
    height_mode: str = "global",
    height_quantile: float = 1.0,
    ridge_scale: float = 1.55,
    row_spacing: float = 0.68,
    alpha: float = 0.85,
    linewidth: float = 1.1,
    outline_color: str = "#2B2B2B",
    baseline_color: str = "#505050",
    baseline_linewidth: float = 1.0,
    show_baseline: bool = True,
    show_quantiles: Sequence[float] | None = None,
    quantile_linewidth: float = 1.7,
    layer_order: str = "far_on_top",
    layer_quantile: float = 0.9,
    boundary_color: str = "#303030",
    boundary_linewidth: float = 1.2,
    boundary_linestyle: str = "--",
    xgrid: bool = False,
    grid_alpha: float = 0.2,
    background_color: str = "white",
    legend: bool = True,
    legend_title: str | None = None,
    title: str | None = None,
    xlabel: str = "Signed physical distance from component boundary (μm)",
    ylabel: str | None = None,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    save_path: str | None = None,
    dpi: int = 300,
    show: bool = True,
    return_data: bool = False,
):
    if aggregation_mode not in {"local", "pool"}:
        raise ValueError("aggregation_mode必须是'local'或'pool'")
    if height_mode not in {"global", "row"}:
        raise ValueError("height_mode必须是'global'或'row'")
    if layer_order not in {"far_on_top", "near_on_top", "group_order"}:
        raise ValueError("layer_order必须是'far_on_top'、'near_on_top'或'group_order'")
    lower, upper = map(float, distance_range)
    if not np.isfinite([lower, upper]).all() or lower >= upper:
        raise ValueError("distance_range必须满足lower < upper")
    if not lower <= 0 <= upper:
        raise ValueError("distance_range必须包含0，例如(-50, 200)")
    if grid_size < 100:
        raise ValueError("grid_size不能小于100")
    if not 0 < float(height_quantile) <= 1:
        raise ValueError("height_quantile必须位于(0, 1]")
    if not 0 < float(layer_quantile) < 1:
        raise ValueError("layer_quantile必须位于(0, 1)")
    data = _prepare_ridge_data(
        distance_data,
        group_col,
        celltype_col,
        component_col,
        signed_distance_col,
        distance_col,
        side_col,
        aggregation_mode,
    )
    available_groups = _ordered_unique(data[group_col])
    available_celltypes = _ordered_unique(data[celltype_col])
    group_order = (
        available_groups if group_order is None else _ordered_unique(group_order)
    )
    celltype_order = (
        available_celltypes
        if celltype_order is None
        else _ordered_unique(celltype_order)
    )
    missing_groups = set(group_order).difference(available_groups)
    missing_celltypes = set(celltype_order).difference(available_celltypes)
    if missing_groups:
        raise ValueError(f"未找到分组：{sorted(missing_groups)}")
    if missing_celltypes:
        raise ValueError(f"未找到细胞类型：{sorted(missing_celltypes)}")
    data = data[
        data[group_col].isin(group_order) & data[celltype_col].isin(celltype_order)
    ].copy()
    data["_in_range"] = data[signed_distance_col].between(lower, upper)
    range_values = data.loc[data["_in_range"], signed_distance_col].to_numpy(float)
    if not len(range_values):
        raise ValueError(f"distance_range={distance_range}内没有细胞")
    grid = _make_signed_grid(lower, upper, grid_size)
    resolved_bandwidth = _resolve_bandwidth(
        range_values, bandwidth, upper - lower, grid_size
    )
    quantiles = [] if show_quantiles is None else [float(q) for q in show_quantiles]
    if any((not 0 < q < 1 for q in quantiles)):
        raise ValueError("show_quantiles中的数值必须位于(0, 1)")

    def weighted_curve(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        selected = values[(values >= lower) & (values <= upper)]
        n_all, n_range = (len(values), len(selected))
        share = n_range / n_all if n_all else 0.0
        curve = (
            _signed_kde(selected, grid, resolved_bandwidth) * share
            if n_range
            else np.zeros_like(grid)
        )
        return (curve, selected, n_all, n_range, share)

    density_records, component_records, support_records = ([], [], [])
    for celltype in celltype_order:
        celltype_data = data[data[celltype_col] == celltype]
        for group in group_order:
            block = celltype_data[celltype_data[group_col] == group]
            if block.empty:
                continue
            if aggregation_mode == "local":
                curves, shares = ([], [])
                pooled_n_all = pooled_n_range = 0
                for component_id, component in block.groupby(
                    component_col, observed=True, sort=False
                ):
                    values = component[signed_distance_col].to_numpy(float)
                    if len(values) < min_cells_per_component:
                        continue
                    curve, selected, n_all, n_range, share = weighted_curve(values)
                    curves.append(curve)
                    shares.append(share)
                    pooled_n_all += n_all
                    pooled_n_range += n_range
                    record = {
                        group_col: group,
                        celltype_col: celltype,
                        component_col: component_id,
                        "n_all": n_all,
                        "n_in_range": n_range,
                        "n_out_range": n_all - n_range,
                        "range_share": share,
                        "curve_area": _curve_area(curve, grid),
                        "layer_distance": (
                            float(np.quantile(selected, layer_quantile))
                            if n_range
                            else np.nan
                        ),
                    }
                    for q in quantiles:
                        record[f"q{q:g}"] = (
                            float(np.quantile(selected, q)) if n_range else np.nan
                        )
                    component_records.append(record)
                n_components = len(curves)
                if n_components < min_components_per_group:
                    continue
                mean_density = np.mean(np.stack(curves), axis=0)
                pooled_share = pooled_n_range / pooled_n_all if pooled_n_all else np.nan
                mean_component_share = float(np.mean(shares))
            else:
                values = block[signed_distance_col].to_numpy(float)
                if len(values) < min_cells_pool:
                    continue
                mean_density, selected, pooled_n_all, pooled_n_range, pooled_share = (
                    weighted_curve(values)
                )
                n_components = (
                    block[component_col].nunique() if component_col in block else 1
                )
                mean_component_share = np.nan
                record = {
                    group_col: group,
                    celltype_col: celltype,
                    component_col: "__pooled__",
                    "n_all": pooled_n_all,
                    "n_in_range": pooled_n_range,
                    "n_out_range": pooled_n_all - pooled_n_range,
                    "range_share": pooled_share,
                    "curve_area": _curve_area(mean_density, grid),
                    "layer_distance": (
                        float(np.quantile(selected, layer_quantile))
                        if pooled_n_range
                        else np.nan
                    ),
                }
                for q in quantiles:
                    record[f"q{q:g}"] = (
                        float(np.quantile(selected, q)) if pooled_n_range else np.nan
                    )
                component_records.append(record)
            for x, density in zip(grid, mean_density):
                density_records.append(
                    {
                        group_col: group,
                        celltype_col: celltype,
                        signed_distance_col: float(x),
                        "density": float(density),
                        "aggregation_mode": aggregation_mode,
                        "n_components": n_components,
                        "n_all": pooled_n_all,
                        "n_in_range": pooled_n_range,
                    }
                )
            support_records.append(
                {
                    group_col: group,
                    celltype_col: celltype,
                    "aggregation_mode": aggregation_mode,
                    "n_components": n_components,
                    "n_all": pooled_n_all,
                    "n_in_range": pooled_n_range,
                    "n_out_range": pooled_n_all - pooled_n_range,
                    "pooled_range_share": pooled_share,
                    "mean_component_range_share": mean_component_share,
                    "curve_area": _curve_area(mean_density, grid),
                }
            )
    if not density_records:
        raise ValueError("没有满足条件的曲线，请降低细胞数或component数量阈值")
    density_data = pd.DataFrame(density_records)
    component_summary = pd.DataFrame(component_records)
    support_summary = pd.DataFrame(support_records)
    valid_celltypes = [
        ct for ct in celltype_order if density_data[celltype_col].eq(ct).any()
    ]
    omitted = [ct for ct in celltype_order if ct not in valid_celltypes]
    if omitted:
        warnings.warn(
            f"以下细胞类型没有满足条件的曲线，已省略：{omitted}",
            RuntimeWarning,
            stacklevel=2,
        )
    celltype_order = valid_celltypes
    palette = _resolve_palette(group_order, colors)
    if ax is None:
        figsize = figsize or (9, max(3, 0.5 * len(celltype_order) + 1.6))
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.set_facecolor(background_color)
    positions = {
        ct: (len(celltype_order) - 1 - i) * row_spacing
        for i, ct in enumerate(celltype_order)
    }
    positive = density_data.loc[density_data["density"] > 0, "density"].to_numpy()
    global_reference = (
        max(float(np.quantile(positive, height_quantile)), np.finfo(float).eps)
        if len(positive)
        else 1.0
    )
    for celltype_index, celltype in enumerate(celltype_order):
        baseline = positions[celltype]
        row = density_data[density_data[celltype_col] == celltype]
        if show_baseline:
            ax.hlines(
                baseline,
                lower,
                upper,
                color=baseline_color,
                linewidth=baseline_linewidth,
                zorder=1,
            )
        available = [g for g in group_order if row[group_col].eq(g).any()]
        summaries = component_summary[component_summary[celltype_col] == celltype]
        scores = {}
        for group in available:
            values = pd.to_numeric(
                summaries.loc[summaries[group_col] == group, "layer_distance"],
                errors="coerce",
            ).dropna()
            scores[group] = float(values.median()) if len(values) else np.nan
        if layer_order == "group_order":
            draw_groups = available
        elif layer_order == "far_on_top":
            draw_groups = sorted(
                available, key=lambda g: -np.inf if np.isnan(scores[g]) else scores[g]
            )
        else:
            draw_groups = sorted(
                available,
                key=lambda g: np.inf if np.isnan(scores[g]) else scores[g],
                reverse=True,
            )
        if height_mode == "row":
            row_positive = row.loc[row["density"] > 0, "density"].to_numpy()
            reference = (
                max(
                    float(np.quantile(row_positive, height_quantile)),
                    np.finfo(float).eps,
                )
                if len(row_positive)
                else 1.0
            )
        else:
            reference = global_reference
        for rank, group in enumerate(draw_groups):
            curve = row[row[group_col] == group].sort_values(signed_distance_col)
            x = curve[signed_distance_col].to_numpy(float)
            density = curve["density"].to_numpy(float)
            y = baseline + ridge_scale * np.clip(density / reference, 0, 1)
            zorder = 10 + celltype_index * max(len(group_order), 1) + rank
            ax.fill_between(
                x,
                baseline,
                y,
                facecolor=palette[group],
                edgecolor=outline_color,
                linewidth=linewidth,
                alpha=alpha,
                zorder=zorder,
            )
            if quantiles:
                group_summary = summaries[summaries[group_col] == group]
                for qi, q in enumerate(quantiles):
                    values = pd.to_numeric(
                        group_summary[f"q{q:g}"], errors="coerce"
                    ).dropna()
                    if len(values):
                        marker_x = float(values.median())
                        marker_height = ridge_scale * (0.12 + 0.08 * qi)
                        ax.vlines(
                            marker_x,
                            baseline,
                            baseline + marker_height,
                            color=outline_color,
                            linewidth=quantile_linewidth,
                            zorder=zorder + 0.2,
                        )
    ax.axvline(
        0,
        color=boundary_color,
        linewidth=boundary_linewidth,
        linestyle=boundary_linestyle,
        zorder=100,
    )
    ax.set_xlim(lower, upper)
    ax.set_ylim(
        -0.25 * row_spacing,
        (len(celltype_order) - 1) * row_spacing + ridge_scale * 1.08,
    )
    ax.set_yticks([positions[ct] for ct in celltype_order])
    ax.set_yticklabels(celltype_order)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("" if ylabel is None else ylabel)
    ax.tick_params(axis="y", length=0)
    if title:
        ax.set_title(title)
    if xgrid:
        ax.grid(axis="x", linewidth=0.8, alpha=grid_alpha, zorder=0)
    if legend:
        handles = [
            Patch(facecolor=palette[g], edgecolor=outline_color, label=g)
            for g in group_order
        ]
        ax.legend(handles=handles, title=legend_title, frameon=False, loc="upper right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    if return_data:
        return (
            fig,
            ax,
            {
                "density": density_data,
                "component_summary": component_summary,
                "support_summary": support_summary,
                "bandwidth": resolved_bandwidth,
                "distance_range": (lower, upper),
                "aggregation_mode": aggregation_mode,
                "group_order": list(group_order),
                "celltype_order": list(celltype_order),
                "height_mode": height_mode,
                "height_reference": (
                    global_reference if height_mode == "global" else None
                ),
            },
        )
    return (fig, ax)


def compute_cellwise_spatial_density(
    adata,
    group_definitions: Mapping[str, str | Sequence[str]] | None = None,
    spatial_key: str = "spatial",
    celltype_key: str = "celltype",
    groups: Sequence[str] | None = None,
    radius_um: float = 25.0,
    sample_key: str | None = None,
    sample: Any | None = None,
    exclude_self: bool = True,
    all_cells_group_name: str = "All_cells",
):
    if spatial_key not in adata.obsm:
        raise KeyError(f"`{spatial_key}` was not found in `adata.obsm`.")
    if not np.isfinite(radius_um) or radius_um <= 0:
        raise ValueError("`radius_um` must be finite and > 0.")
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"`adata.obsm[{spatial_key!r}]` must have shape (n_obs, >=2).")
    coords = coords[:, :2]
    finite_mask = np.isfinite(coords).all(axis=1)
    sample_mask, selected_sample = _resolve_sample_mask(
        adata.obs, sample_key=sample_key, sample=sample
    )
    base_mask = finite_mask & sample_mask
    if not base_mask.any():
        raise ValueError("No finite spatial coordinates remain after filtering.")
    base_coords = coords[base_mask]
    extent = (
        float(base_coords[:, 0].min()),
        float(base_coords[:, 0].max()),
        float(base_coords[:, 1].min()),
        float(base_coords[:, 1].max()),
    )
    obs_names = np.asarray(adata.obs_names.astype(str))
    area = math.pi * radius_um * radius_um
    if group_definitions is None:
        if groups is not None:
            raise ValueError(
                "`groups` should not be specified when `group_definitions=None`."
            )
        group_name = str(all_cells_group_name)
        group_indices = np.flatnonzero(base_mask)
        group_coords = coords[group_indices]
        counts = _radius_neighbor_counts(
            reference_coords=group_coords,
            query_coords=group_coords,
            radius_um=radius_um,
        )
        if exclude_self:
            counts = np.maximum(counts - 1, 0)
        density = counts.astype(np.float64) / area
        result = pd.DataFrame(
            {
                "group": group_name,
                "cell_index": group_indices.astype(np.int64),
                "cell_id": obs_names[group_indices],
                "x": group_coords[:, 0],
                "y": group_coords[:, 1],
                "neighbor_count": counts,
                "density_per_um2": density,
            }
        )
        result.attrs["extent"] = extent
        result.attrs["radius_um"] = float(radius_um)
        result.attrs["group_definitions"] = {group_name: "__all_cells__"}
        result.attrs["sample_key"] = sample_key
        result.attrs["sample"] = selected_sample
        result.attrs["mode"] = "all_cells"
        return result
    if celltype_key not in adata.obs:
        raise KeyError(f"`{celltype_key}` was not found in `adata.obs`.")
    definitions = _normalize_group_definitions(group_definitions)
    if groups is None:
        groups = list(definitions)
    else:
        groups = _ordered_unique(groups)
        missing_groups = [group for group in groups if group not in definitions]
        if missing_groups:
            raise KeyError(
                f"Groups were not found in `group_definitions`: {missing_groups}"
            )
    labels = adata.obs[celltype_key].astype(str).to_numpy()
    records = []
    for group in groups:
        group_indices = np.flatnonzero(base_mask & np.isin(labels, definitions[group]))
        if group_indices.size == 0:
            continue
        group_coords = coords[group_indices]
        counts = _radius_neighbor_counts(
            reference_coords=group_coords,
            query_coords=group_coords,
            radius_um=radius_um,
        )
        if exclude_self:
            counts = np.maximum(counts - 1, 0)
        density = counts.astype(np.float64) / area
        group_frame = pd.DataFrame(
            {
                "group": group,
                "cell_index": group_indices.astype(np.int64),
                "cell_id": obs_names[group_indices],
                "x": group_coords[:, 0],
                "y": group_coords[:, 1],
                "neighbor_count": counts,
                "density_per_um2": density,
            }
        )
        records.append(group_frame)
    if not records:
        raise ValueError(
            "No cells matching the requested group definitions were found."
        )
    result = pd.concat(records, ignore_index=True)
    result.attrs["extent"] = extent
    result.attrs["radius_um"] = float(radius_um)
    result.attrs["group_definitions"] = {
        group: list(definitions[group]) for group in groups
    }
    result.attrs["sample_key"] = sample_key
    result.attrs["sample"] = selected_sample
    result.attrs["mode"] = "groups"
    return result


def plot_density_pair_panel(
    density_data: pd.DataFrame,
    tumor_group: str,
    immune_group: str,
    region: tuple[Sequence[float], Sequence[float]] | None = None,
    colors: Mapping[str, Any] | None = None,
    value_col: str = "neighbor_count",
    scale_mode: str = "per_group_quantile",
    clip_quantile: float = 0.97,
    gamma: float = 0.9,
    point_size: float = 7.0,
    point_alpha: float = 1.0,
    point_edgecolor: Any = "none",
    point_linewidth: float = 0.0,
    tumor_on_top: bool = False,
    figsize: tuple[float, float] = (8.2, 5.2),
    xlabel: str = "Spatial x",
    ylabel: str = "Spatial y",
    show_axes: bool = True,
    invert_y: bool = True,
    equal_aspect: bool = True,
    full_frame: bool = True,
    frame_color: str = "#3F3F3F",
    frame_linewidth: float = 1.0,
    add_colorbar: bool = True,
    colorbar_y: float = 1.08,
    colorbar_tumor_x: float = 0.05,
    colorbar_immune_x: float = 0.57,
    colorbar_length: float = 0.38,
    colorbar_thickness: float = 0.035,
    colorbar_top_margin: float = 0.19,
    colorbar_title_fontsize: float = 11.5,
    colorbar_tick_fontsize: float = 8.5,
    colorbar_title_align: str = "center",
    colorbar_tick_mode: str = "midmax",
    rasterized: bool = False,
    save_path: str | None = None,
    dpi: int = 300,
    show: bool = True,
    return_data: bool = False,
):
    required = {
        "group",
        "cell_index",
        "cell_id",
        "x",
        "y",
        "neighbor_count",
        "density_per_um2",
    }
    missing = required.difference(density_data.columns)
    if missing:
        raise KeyError(f"`density_data` is missing columns: {sorted(missing)}")
    tumor_group = str(tumor_group)
    immune_group = str(immune_group)
    if value_col not in {"neighbor_count", "density_per_um2"}:
        raise ValueError("`value_col` must be 'neighbor_count' or 'density_per_um2'.")
    if gamma <= 0:
        raise ValueError("`gamma` must be greater than 0.")
    if colorbar_tick_mode not in {"midmax", "minmidmax"}:
        raise ValueError("`colorbar_tick_mode` must be 'midmax' or 'minmidmax'.")
    if colorbar_title_align not in {"left", "center", "right"}:
        raise ValueError("`colorbar_title_align` must be 'left', 'center', or 'right'.")
    if colors is None:
        raise ValueError(
            "`colors` must provide entries for both `tumor_group` and `immune_group`."
        )
    for group in [tumor_group, immune_group]:
        if group not in colors:
            raise KeyError(f"`colors` is missing a value for group {group!r}.")
    available_groups = set(density_data["group"].astype(str))
    for group in [tumor_group, immune_group]:
        if group not in available_groups:
            raise KeyError(f"Group {group!r} was not found in `density_data['group']`.")
    extent = _resolve_region(density_data, region)
    x_min, x_max, y_min, y_max = extent
    panel = density_data[
        density_data["group"].isin([tumor_group, immune_group])
        & density_data["x"].between(x_min, x_max, inclusive="both")
        & density_data["y"].between(y_min, y_max, inclusive="both")
    ].copy()
    if panel.empty:
        raise ValueError("No selected cells were found in the requested `region`.")
    # Use the full density table for color scales, independent of the displayed crop.
    scales = _compute_group_scales(
        density_data=density_data,
        groups=[tumor_group, immune_group],
        value_col=value_col,
        scale_mode=scale_mode,
        clip_quantile=clip_quantile,
    )
    tumor = panel[panel["group"] == tumor_group].copy()
    immune = panel[panel["group"] == immune_group].copy()
    tumor = tumor.sort_values(value_col, ascending=True, kind="mergesort")
    immune = immune.sort_values(value_col, ascending=True, kind="mergesort")
    fig, ax = plt.subplots(figsize=figsize)
    if add_colorbar:
        fig.subplots_adjust(top=1.0 - colorbar_top_margin)
    cax_top = None
    cax_bottom = None
    cmap_tumor = _white_to_color_cmap(
        color=colors[tumor_group], name=f"{tumor_group}_white_to_color"
    )
    cmap_immune = _white_to_color_cmap(
        color=colors[immune_group], name=f"{immune_group}_white_to_color"
    )
    norm_tumor = PowerNorm(gamma=gamma, vmin=0.0, vmax=scales[tumor_group], clip=True)
    norm_immune = PowerNorm(gamma=gamma, vmin=0.0, vmax=scales[immune_group], clip=True)
    layers = [
        ("tumor", tumor, cmap_tumor, norm_tumor),
        ("immune", immune, cmap_immune, norm_immune),
    ]
    if not tumor_on_top:
        layers = [
            ("tumor", tumor, cmap_tumor, norm_tumor),
            ("immune", immune, cmap_immune, norm_immune),
        ]
    else:
        layers = [
            ("immune", immune, cmap_immune, norm_immune),
            ("tumor", tumor, cmap_tumor, norm_tumor),
        ]
    for _, frame, cmap, norm in layers:
        if frame.empty:
            continue
        ax.scatter(
            frame["x"],
            frame["y"],
            s=point_size,
            c=frame[value_col],
            cmap=cmap,
            norm=norm,
            alpha=point_alpha,
            edgecolors=point_edgecolor,
            linewidths=point_linewidth,
            rasterized=rasterized,
        )
    ax.set_xlim(x_min, x_max)
    if invert_y:
        ax.set_ylim(y_max, y_min)
    else:
        ax.set_ylim(y_min, y_max)
    if equal_aspect:
        ax.set_aspect("equal")
    else:
        ax.set_aspect("auto")
    if show_axes:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.grid(False)
    ax.set_facecolor("white")
    for spine_name in ["top", "right", "bottom", "left"]:
        ax.spines[spine_name].set_visible(full_frame)
        ax.spines[spine_name].set_color(frame_color)
        ax.spines[spine_name].set_linewidth(frame_linewidth)
    cb_tumor = None
    cb_immune = None
    if add_colorbar:
        cax_top = ax.inset_axes(
            [colorbar_tumor_x, colorbar_y, colorbar_length, colorbar_thickness],
            transform=ax.transAxes,
        )
        cax_bottom = ax.inset_axes(
            [colorbar_immune_x, colorbar_y, colorbar_length, colorbar_thickness],
            transform=ax.transAxes,
        )
        sm_tumor = ScalarMappable(norm=norm_tumor, cmap=cmap_tumor)
        sm_immune = ScalarMappable(norm=norm_immune, cmap=cmap_immune)
        cb_tumor, cb_immune = _draw_horizontal_colorbar_pair(
            fig=fig,
            cax_top=cax_top,
            cax_bottom=cax_bottom,
            mappable_a=sm_tumor,
            mappable_b=sm_immune,
            label_a=tumor_group,
            label_b=immune_group,
            color_a=colors[tumor_group],
            color_b=colors[immune_group],
            tick_mode=colorbar_tick_mode,
            title_fontsize=colorbar_title_fontsize,
            tick_fontsize=colorbar_tick_fontsize,
            title_align=colorbar_title_align,
        )
    fig.patch.set_facecolor("white")
    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
    if show:
        plt.show()
    if return_data:
        return (
            fig,
            ax,
            {
                "tumor_data": tumor,
                "immune_data": immune,
                "region": extent,
                "value_col": value_col,
                "group_scales": scales,
                "radius_um": density_data.attrs.get("radius_um"),
                "tumor_group": tumor_group,
                "immune_group": immune_group,
                "tumor_colorbar": cb_tumor,
                "immune_colorbar": cb_immune,
                "gamma": gamma,
                "scale_mode": scale_mode,
                "clip_quantile": clip_quantile,
                "colorbar_y": colorbar_y if add_colorbar else None,
                "colorbar_tumor_x": colorbar_tumor_x if add_colorbar else None,
                "colorbar_immune_x": colorbar_immune_x if add_colorbar else None,
                "colorbar_length": colorbar_length if add_colorbar else None,
                "colorbar_thickness": colorbar_thickness if add_colorbar else None,
                "colorbar_title_align": colorbar_title_align if add_colorbar else None,
            },
        )
    return (fig, ax)


def plot_multihop_gene_trajectory(
    gradient_result,
    gene,
    metric="mean_expression",
    observed_celltypes=None,
    celltype=None,
    colors=None,
    show_sem=True,
    figsize=(6, 4),
    linewidth=2.0,
    markersize=6,
    xlabel="Topological hop",
    ylabel=None,
    title=None,
    fontsize=11,
    tick_fontsize=10,
    legend=True,
    save_path=None,
    dpi=300,
    show=True,
):
    summary = gradient_result["summary"].copy()
    if metric not in summary.columns:
        raise KeyError(f"`metric={metric}` was not found.")
    summary = summary.loc[summary["gene"].astype(str) == str(gene)].copy()
    if observed_celltypes is not None:
        if isinstance(observed_celltypes, str):
            observed_celltypes = [observed_celltypes]
        summary = summary.loc[
            summary["observed_celltype"].astype(str).isin(observed_celltypes)
        ].copy()
    if celltype is not None:
        if "celltype" not in summary.columns:
            raise ValueError(
                "`celltype` can only be used when `stratify_celltype=True` was used during computation."
            )
        summary = summary.loc[summary["celltype"].astype(str) == str(celltype)].copy()
    if summary.empty:
        raise ValueError("No data remain after filtering.")
    sem_col = f"{metric}_sem"
    fig, ax = plt.subplots(figsize=figsize)
    groups = pd.unique(summary["observed_celltype"].astype(str))
    for group in groups:
        block = summary.loc[summary["observed_celltype"].astype(str) == group].copy()
        block = block.sort_values("signed_hop")
        x = block["signed_hop"].to_numpy()
        y = block[metric].to_numpy(dtype=float)
        plot_kwargs = {}
        if colors is not None:
            plot_kwargs["color"] = colors[group]
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=linewidth,
            markersize=markersize,
            label=group,
            **plot_kwargs,
        )
        if show_sem and sem_col in block.columns:
            sem = block[sem_col].to_numpy(dtype=float)
            fill_kwargs = {}
            if colors is not None:
                fill_kwargs["color"] = colors[group]
            ax.fill_between(x, y - sem, y + sem, alpha=0.15, linewidth=0, **fill_kwargs)
    hop_table = (
        summary[["signed_hop", "hop_label"]].drop_duplicates().sort_values("signed_hop")
    )
    ax.set_xticks(hop_table["signed_hop"])
    ax.set_xticklabels(hop_table["hop_label"], rotation=45, ha="right")
    ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is None:
        ylabel_map = {
            "mean_expression": "Mean expression",
            "positive_fraction": "Positive fraction",
            "positive_mean_expression": "Mean expression in positive cells",
        }
        ylabel = ylabel_map.get(metric, metric)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if title is None:
        title = str(gene)
    ax.set_title(title, fontsize=fontsize + 1)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    if legend:
        ax.legend(frameon=False, fontsize=tick_fontsize)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return ax


def plot_multihop_genes_trajectory(
    gradient_result,
    genes,
    observed_celltype,
    metric="mean_expression",
    celltype=None,
    normalize=True,
    colors=None,
    show_sem=False,
    sem_alpha=0.12,
    figsize=(7, 5),
    linewidth=2.0,
    markersize=5,
    xlabel="Topological hop",
    ylabel=None,
    title=None,
    fontsize=11,
    tick_fontsize=10,
    legend=True,
    legend_fontsize=None,
    save_path=None,
    dpi=300,
    show=True,
):
    if "summary" not in gradient_result:
        raise KeyError("`gradient_result` does not contain 'summary'.")
    summary = gradient_result["summary"].copy()
    if metric not in summary.columns:
        raise KeyError(
            f"`metric={metric!r}` was not found in `gradient_result['summary']`."
        )
    if isinstance(genes, str):
        genes = [genes]
    else:
        genes = list(genes)
    genes = list(dict.fromkeys((str(gene) for gene in genes)))
    if len(genes) == 0:
        raise ValueError("`genes` cannot be empty.")
    summary = summary.loc[
        summary["observed_celltype"].astype(str) == str(observed_celltype)
    ].copy()
    if summary.empty:
        raise ValueError(
            f"No data were found for observed_celltype={observed_celltype!r}."
        )
    if celltype is not None:
        if "celltype" not in summary.columns:
            raise ValueError(
                "`celltype` can only be used when `stratify_celltype=True` was used during `compute_multihop_gene_gradient`."
            )
        summary = summary.loc[summary["celltype"].astype(str) == str(celltype)].copy()
        if summary.empty:
            raise ValueError(f"No data were found for celltype={celltype!r}.")
    summary = summary.loc[summary["gene"].astype(str).isin(genes)].copy()
    available_genes = set(summary["gene"].astype(str))
    missing_genes = [gene for gene in genes if gene not in available_genes]
    if missing_genes:
        print(
            f"Warning: the following genes were not found in gradient_result: {missing_genes}"
        )
    gene_order = [gene for gene in genes if gene in available_genes]
    if len(gene_order) == 0:
        raise ValueError("None of the requested genes were found in gradient_result.")
    fig, ax = plt.subplots(figsize=figsize)
    sem_col = f"{metric}_sem"
    for gene in gene_order:
        block = summary.loc[summary["gene"].astype(str) == gene].copy()
        block = block.sort_values("signed_hop")
        x = block["signed_hop"].to_numpy(dtype=float)
        y = block[metric].to_numpy(dtype=float)
        if show_sem and sem_col in block.columns:
            sem = block[sem_col].to_numpy(dtype=float)
        else:
            sem = None
        if normalize:
            finite_y = y[np.isfinite(y)]
            if len(finite_y) > 0:
                scale = np.nanmax(np.abs(finite_y))
            else:
                scale = np.nan
            if np.isfinite(scale) and scale > 0:
                y = y / scale
                if sem is not None:
                    sem = sem / scale
        plot_kwargs = {}
        if colors is not None and gene in colors:
            plot_kwargs["color"] = colors[gene]
        line = ax.plot(
            x,
            y,
            marker="o",
            linewidth=linewidth,
            markersize=markersize,
            label=gene,
            **plot_kwargs,
        )
        if show_sem and sem is not None:
            valid = np.isfinite(y) & np.isfinite(sem)
            if valid.any():
                ax.fill_between(
                    x[valid],
                    y[valid] - sem[valid],
                    y[valid] + sem[valid],
                    color=line[0].get_color(),
                    alpha=sem_alpha,
                    linewidth=0,
                )
    hop_table = (
        summary[["signed_hop", "hop_label"]].drop_duplicates().sort_values("signed_hop")
    )
    ax.set_xticks(hop_table["signed_hop"].to_numpy())
    ax.set_xticklabels(hop_table["hop_label"].astype(str), rotation=45, ha="right")
    ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is None:
        if normalize:
            if metric == "mean_expression":
                ylabel = "Relative mean expression"
            elif metric == "positive_fraction":
                ylabel = "Relative positive fraction"
            elif metric == "positive_mean_expression":
                ylabel = "Relative expression in positive cells"
            else:
                ylabel = f"Relative {metric}"
        else:
            ylabel_map = {
                "mean_expression": "Mean expression",
                "positive_fraction": "Positive fraction",
                "positive_mean_expression": "Mean expression in positive cells",
            }
            ylabel = ylabel_map.get(metric, metric)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if title is None:
        title = str(observed_celltype)
        if celltype is not None:
            title += f" – {celltype}"
    ax.set_title(title, fontsize=fontsize + 1)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    if legend:
        if legend_fontsize is None:
            legend_fontsize = tick_fontsize
        ax.legend(
            frameon=False,
            fontsize=legend_fontsize,
            bbox_to_anchor=(1.02, 1.0),
            loc="upper left",
        )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return ax


def _rgba_to_hex(rgba):
    r, g, b = rgba[:3]
    return "#{:02X}{:02X}{:02X}".format(
        int(round(r * 255)), int(round(g * 255)), int(round(b * 255))
    )


def _make_multihop_signed_palette(
    max_hop,
    inhop_cmap="Blues",
    outhop_cmap="Reds",
    inhop_range=(0.45, 0.9),
    outhop_range=(0.45, 0.9),
    target_color="#B238E0",
    bridge_color="#17A34A",
    closure_bridge_color="#F39C12",
    outside_color="#D9D9D9",
    ambiguous_color="#808080",
):
    if max_hop < 1:
        raise ValueError("`max_hop` must be >= 1.")
    in_cmap = plt.get_cmap(inhop_cmap)
    out_cmap = plt.get_cmap(outhop_cmap)
    in_vals = np.linspace(inhop_range[0], inhop_range[1], max_hop)
    out_vals = np.linspace(outhop_range[0], outhop_range[1], max_hop)
    palette = {}
    for hop, value in enumerate(in_vals, start=1):
        palette[f"InHop{hop}"] = _rgba_to_hex(in_cmap(value))
    for hop, value in enumerate(out_vals, start=1):
        palette[f"OutHop{hop}"] = _rgba_to_hex(out_cmap(value))
    palette["Target"] = target_color
    palette["Bridge"] = bridge_color
    palette["ClosureBridge"] = closure_bridge_color
    palette["outside"] = outside_color
    palette["ambiguous"] = ambiguous_color
    return palette


def _infer_max_hop(labels):
    hops = []
    for label in labels:
        match = re.match("^(InHop|OutHop)(\\d+)$", str(label))
        if match is not None:
            hops.append(int(match.group(2)))
    return max(hops) if hops else 1


def _ordered_multihop_groups(max_hop, include_outside=True, include_ambiguous=True):
    groups = (
        [f"InHop{hop}" for hop in range(max_hop, 0, -1)]
        + ["Bridge", "ClosureBridge", "Target"]
        + [f"OutHop{hop}" for hop in range(1, max_hop + 1)]
    )
    if include_ambiguous:
        groups.append("ambiguous")
    if include_outside:
        groups.append("outside")
    return groups


def _aggregate_membership_for_plot(membership, observed_celltype, tie_policy="first"):
    required = {
        "observed_celltype",
        "component_id",
        "cell_index",
        "cell_id",
        "signed_hop",
        "hop",
        "side",
        "hop_label",
    }
    missing = required.difference(membership.columns)
    if missing:
        raise KeyError(f"`membership` is missing columns: {sorted(missing)}")
    if tie_policy not in {"first", "ambiguous", "in", "out", "error"}:
        raise ValueError(
            "`tie_policy` must be one of 'first', 'ambiguous', 'in', 'out', or 'error'."
        )
    selected = membership[
        membership["observed_celltype"].astype(str) == str(observed_celltype)
    ].copy()
    if selected.empty:
        raise ValueError(f"No membership found for {observed_celltype!r}.")
    selected["cell_index"] = selected["cell_index"].astype(np.int64)
    selected["component_id"] = selected["component_id"].astype(np.int64)
    selected["hop"] = selected["hop"].astype(np.int16)
    selected["signed_hop"] = selected["signed_hop"].astype(np.int16)
    selected["side"] = selected["side"].astype(str)
    selected["hop_label"] = selected["hop_label"].astype(str)
    # Keep the first role per cell and component before resolving overlaps.
    selected = selected.drop_duplicates(
        subset=["component_id", "cell_index"], keep="first"
    )
    nearest_hop = selected.groupby("cell_index", sort=False)["hop"].transform("min")
    nearest = selected[selected["hop"] == nearest_hop].copy()
    role_priority = (
        nearest["side"]
        .map({"Target": 0, "Bridge": 1, "ClosureBridge": 2, "In": 3, "Out": 4})
        .fillna(5)
    )
    nearest = nearest.assign(_role_priority=role_priority).sort_values(
        ["cell_index", "_role_priority", "component_id", "signed_hop"], kind="stable"
    )
    label_counts = nearest.groupby("cell_index", sort=False)["hop_label"].nunique()
    nearest_counts = nearest.groupby("cell_index", sort=False).size()
    conflict_indices = label_counts.index[label_counts > 1].to_numpy(dtype=np.int64)
    if tie_policy == "error" and len(conflict_indices) > 0:
        raise ValueError(f"Found {len(conflict_indices)} equal-distance conflicts.")
    chosen = nearest.drop_duplicates("cell_index", keep="first").set_index("cell_index")
    if len(conflict_indices) > 0 and tie_policy in {"in", "out"}:
        preferred_side = "In" if tie_policy == "in" else "Out"
        preferred = nearest[
            nearest["cell_index"].isin(conflict_indices)
            & (nearest["side"] == preferred_side)
        ]
        preferred = preferred.drop_duplicates("cell_index", keep="first").set_index(
            "cell_index"
        )
        if not preferred.empty:
            chosen.update(preferred)
    chosen["n_nearest_components"] = nearest_counts.reindex(chosen.index).astype(
        np.int32
    )
    chosen["conflict"] = chosen.index.isin(conflict_indices)
    if tie_policy == "ambiguous" and len(conflict_indices) > 0:
        chosen.loc[conflict_indices, "hop_label"] = "ambiguous"
        chosen.loc[conflict_indices, "side"] = "Ambiguous"
        chosen.loc[conflict_indices, "component_id"] = -1
        chosen.loc[conflict_indices, "signed_hop"] = 0
    columns = [
        "cell_id",
        "hop_label",
        "signed_hop",
        "hop",
        "side",
        "component_id",
        "n_nearest_components",
        "conflict",
    ]
    aggregated = (
        chosen[columns].reset_index().sort_values("cell_index").reset_index(drop=True)
    )
    return aggregated


def _build_plot_diagnostics(membership, aggregated, observed_celltype):
    raw = membership[
        membership["observed_celltype"].astype(str) == str(observed_celltype)
    ].copy()
    component_counts = raw.groupby("cell_index")["component_id"].nunique()
    multi_component_cells = component_counts[component_counts > 1]
    target_cells = set(raw.loc[raw["side"] == "Target", "cell_index"].astype(int))
    hop_cells = set(raw.loc[raw["side"].isin(["In", "Out"]), "cell_index"].astype(int))
    target_hop_overlap = target_cells & hop_cells
    agg_lookup = aggregated.set_index("cell_index")["hop_label"].astype(str)
    target_not_target = [
        cell_index
        for cell_index in target_cells
        if cell_index in agg_lookup.index and agg_lookup.loc[cell_index] != "Target"
    ]
    return {
        "n_membership_rows": int(len(raw)),
        "n_unique_cells": int(raw["cell_index"].nunique()),
        "n_multi_component_cells": int(len(multi_component_cells)),
        "n_raw_target_hop_overlap": int(len(target_hop_overlap)),
        "n_target_not_target_after_aggregation": int(len(target_not_target)),
        "multi_component_cells": multi_component_cells.rename(
            "n_components"
        ).reset_index(),
        "raw_target_hop_overlap": np.array(sorted(target_hop_overlap), dtype=np.int64),
        "target_not_target_after_aggregation": np.array(
            target_not_target, dtype=np.int64
        ),
    }


def _expand_heatmap_hop_order(hop_order):
    if hop_order is None:
        hop_order = (1, 15)
    if not isinstance(hop_order, (tuple, list)) or len(hop_order) != 2:
        raise ValueError(
            "hop_order must be a two-element inclusive interval, for example (1, 10) or (-5, 10)."
        )
    start, stop = hop_order
    if (
        isinstance(start, bool)
        or isinstance(stop, bool)
        or (not isinstance(start, (int, np.integer)))
        or (not isinstance(stop, (int, np.integer)))
    ):
        raise TypeError("Both hop_order bounds must be integers.")
    start = int(start)
    stop = int(stop)
    if start > stop:
        raise ValueError("hop_order must satisfy start <= stop.")
    expanded = [hop for hop in range(start, stop + 1) if hop != 0]
    if not expanded:
        raise ValueError("hop_order cannot contain only hop 0.")
    return expanded


def _ordered_profile_values(values):
    return list(dict.fromkeys(values))


def _expand_profile_hop_order(hop_order):
    if hop_order is None:
        hop_order = (1, 15)
    if not isinstance(hop_order, (tuple, list)) or len(hop_order) != 2:
        raise ValueError(
            "`hop_order` must be a two-element inclusive interval, for example (1, 10) or (-5, 10)."
        )
    start, stop = hop_order
    if (
        isinstance(start, bool)
        or isinstance(stop, bool)
        or (not isinstance(start, (int, np.integer)))
        or (not isinstance(stop, (int, np.integer)))
    ):
        raise TypeError("Both `hop_order` bounds must be integers.")
    start = int(start)
    stop = int(stop)
    if start > stop:
        raise ValueError("`hop_order` must satisfy start <= stop.")
    expanded = [hop for hop in range(start, stop + 1) if hop != 0]
    if not expanded:
        raise ValueError("`hop_order` cannot contain only hop 0.")
    return expanded


def _format_hop(value, signed=False):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return f"Hop {value}"
    if not numeric.is_integer():
        return f"Hop {value}"
    hop = int(numeric)
    if signed:
        if hop < 0:
            return f"InHop {abs(hop)}"
        if hop > 0:
            return f"OutHop {hop}"
    return f"Hop {hop}"


def _prepare_observed_multihop_profile(
    enrichment_df,
    observed_celltype,
    neighbor_celltypes: Sequence[str] | None = None,
    hop_order=None,
    qvalue_cutoff=0.05,
    min_enrichment=0.0,
    drop_without_enrichment=True,
    exclude_self=False,
    sort_mode="peak_hop",
):
    missing = sorted(_REQUIRED_COLUMNS.difference(enrichment_df.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    if qvalue_cutoff <= 0 or qvalue_cutoff > 1:
        raise ValueError("`qvalue_cutoff` must be in (0, 1].")
    if sort_mode not in {"peak_hop", "first_significant_hop"}:
        raise ValueError(
            "`sort_mode` must be either 'peak_hop' or 'first_significant_hop'."
        )
    observed_text = str(observed_celltype)
    data = enrichment_df.loc[
        enrichment_df["observed_celltype"].astype(str).eq(observed_text)
    ].copy()
    if data.empty:
        available = _ordered_profile_values(
            enrichment_df["observed_celltype"].astype(str).tolist()
        )
        raise ValueError(
            f"Observed cell type {observed_text!r} was not found. Available values include: {available[:10]}"
        )
    data["randomized_celltype"] = data["randomized_celltype"].astype(str)
    hop_values = pd.to_numeric(data["hop"], errors="coerce")
    if hop_values.isna().any():
        raise ValueError("The `hop` column must contain numeric hop values.")
    if not np.all(np.equal(hop_values.to_numpy(), np.floor(hop_values.to_numpy()))):
        raise ValueError("The `hop` column must contain integer hop values.")
    data["hop"] = hop_values.astype(int)
    if exclude_self:
        data = data.loc[~data["randomized_celltype"].eq(observed_text)].copy()
    if neighbor_celltypes is not None:
        requested = _ordered_profile_values(
            (str(value) for value in neighbor_celltypes)
        )
        available = set(data["randomized_celltype"])
        missing = [value for value in requested if value not in available]
        if missing:
            raise ValueError(f"Neighbor cell types were not found: {missing}")
        data = data.loc[data["randomized_celltype"].isin(requested)].copy()
        base_neighbor_order = requested
    else:
        base_neighbor_order = _ordered_profile_values(
            data["randomized_celltype"].tolist()
        )
    hop_order = _expand_profile_hop_order(hop_order)
    signed_hops = any((hop < 0 for hop in hop_order)) and any(
        (hop > 0 for hop in hop_order)
    )
    value_matrix = data.pivot_table(
        index="randomized_celltype",
        columns="hop",
        values="enrichment",
        aggfunc="first",
        observed=False,
    ).reindex(index=base_neighbor_order, columns=hop_order)
    qvalue_matrix = data.pivot_table(
        index="randomized_celltype",
        columns="hop",
        values="qvalue",
        aggfunc="first",
        observed=False,
    ).reindex(index=base_neighbor_order, columns=hop_order)
    value_array = value_matrix.to_numpy(dtype=float)
    qvalue_array = qvalue_matrix.to_numpy(dtype=float)
    finite_mask = np.isfinite(value_array) & np.isfinite(qvalue_array)
    significant_mask = finite_mask & (qvalue_array <= qvalue_cutoff)
    enriched_mask = significant_mask & (value_array > float(min_enrichment))
    sort_rows = []
    n_hops = len(hop_order)
    for row_index, celltype in enumerate(value_matrix.index):
        row_values = value_array[row_index]
        row_enriched = enriched_mask[row_index]
        if row_enriched.any():
            eligible_indices = np.flatnonzero(row_enriched)
            peak_local = eligible_indices[np.nanargmax(row_values[eligible_indices])]
            first_local = int(eligible_indices[0])
            peak_hop_index = int(peak_local)
            first_hop_index = first_local
            peak_enrichment = float(row_values[peak_local])
        else:
            peak_hop_index = n_hops
            first_hop_index = n_hops
            peak_enrichment = -np.inf
        primary_hop_index = (
            peak_hop_index if sort_mode == "peak_hop" else first_hop_index
        )
        sort_rows.append(
            {
                "randomized_celltype": str(celltype),
                "sort_hop_index": primary_hop_index,
                "peak_hop_index": peak_hop_index,
                "first_significant_hop_index": first_hop_index,
                "peak_hop": (
                    hop_order[peak_hop_index] if peak_hop_index < n_hops else np.nan
                ),
                "first_significant_hop": (
                    hop_order[first_hop_index] if first_hop_index < n_hops else np.nan
                ),
                "peak_enrichment": peak_enrichment,
                "has_enrichment": bool(row_enriched.any()),
                "original_order": row_index,
            }
        )
    row_metadata = pd.DataFrame(sort_rows).set_index("randomized_celltype")
    if drop_without_enrichment:
        keep_rows = row_metadata["has_enrichment"]
        value_matrix = value_matrix.loc[keep_rows.index[keep_rows]]
        qvalue_matrix = qvalue_matrix.loc[value_matrix.index]
        row_metadata = row_metadata.loc[value_matrix.index]
    if value_matrix.empty:
        raise ValueError(
            "No neighbor cell type passed the enrichment filters. Consider increasing `qvalue_cutoff`, lowering `min_enrichment`, or setting `drop_without_enrichment=False`."
        )
    row_metadata = row_metadata.sort_values(
        ["sort_hop_index", "peak_enrichment", "original_order"],
        ascending=[True, False, True],
        kind="stable",
    )
    row_order = row_metadata.index.tolist()
    value_matrix = value_matrix.loc[row_order]
    qvalue_matrix = qvalue_matrix.loc[row_order]
    value_array = value_matrix.to_numpy(dtype=float)
    qvalue_array = qvalue_matrix.to_numpy(dtype=float)
    finite_mask = np.isfinite(value_array) & np.isfinite(qvalue_array)
    significant_mask = finite_mask & (qvalue_array <= qvalue_cutoff)
    enriched_mask = significant_mask & (value_array > float(min_enrichment))
    return {
        "observed_celltype": observed_text,
        "value_matrix": value_matrix,
        "qvalue_matrix": qvalue_matrix,
        "finite_mask": finite_mask,
        "significant_mask": significant_mask,
        "enriched_mask": enriched_mask,
        "row_metadata": row_metadata.loc[row_order],
        "row_order": row_order,
        "hop_order": list(hop_order),
        "signed_hops": signed_hops,
        "parameters": {
            "hop_interval": (hop_order[0], hop_order[-1]),
            "qvalue_cutoff": float(qvalue_cutoff),
            "min_enrichment": float(min_enrichment),
            "drop_without_enrichment": bool(drop_without_enrichment),
            "exclude_self": bool(exclude_self),
            "sort_mode": sort_mode,
        },
    }


def _ordered_unique(values) -> list[str]:
    return list(dict.fromkeys((str(value) for value in values)))


def _resolve_palette(groups, colors):
    groups = list(groups)
    if colors is None:
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        cycle = cycle or [f"C{i}" for i in range(max(1, len(groups)))]
        return {group: cycle[i % len(cycle)] for i, group in enumerate(groups)}
    if isinstance(colors, Mapping):
        missing = [group for group in groups if group not in colors]
        if missing:
            raise KeyError(f"colors中缺少以下分组：{missing}")
        return {group: colors[group] for group in groups}
    colors = list(colors)
    if len(colors) < len(groups):
        raise ValueError("colors的数量不能少于分组数量")
    return {group: colors[i] for i, group in enumerate(groups)}


def _resolve_bandwidth(values, bandwidth, domain_span, grid_size):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    fallback = max(
        domain_span / max(grid_size - 1, 1), domain_span / 100, np.finfo(float).eps
    )
    if isinstance(bandwidth, Real):
        bandwidth = float(bandwidth)
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("数值型bandwidth必须有限且大于0")
        return bandwidth
    method = str(bandwidth).lower()
    if method not in {"scott", "silverman"}:
        raise ValueError("bandwidth必须为正数、'scott'或'silverman'")
    if len(values) < 2:
        return fallback
    std = float(np.std(values, ddof=1))
    if not np.isfinite(std) or std <= 0:
        return fallback
    if method == "scott":
        result = std * len(values) ** (-1 / 5)
    else:
        q25, q75 = np.quantile(values, [0.25, 0.75])
        robust_std = float((q75 - q25) / 1.349)
        scale = min(std, robust_std) if robust_std > 0 else std
        result = 0.9 * scale * len(values) ** (-1 / 5)
    return max(float(result), fallback)


def _signed_kde(values, grid, bandwidth, chunk_size=4096):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.zeros_like(grid, dtype=np.float64)
    density = np.zeros_like(grid, dtype=np.float64)
    normalizer = len(values) * bandwidth * np.sqrt(2 * np.pi)
    for start in range(0, len(values), chunk_size):
        current = values[start : start + chunk_size]
        z = (grid[:, None] - current[None, :]) / bandwidth
        density += np.exp(-0.5 * np.square(z)).sum(axis=1)
    return density / normalizer


def _make_signed_grid(lower, upper, grid_size):
    span = upper - lower
    n_left = int(round((grid_size - 1) * abs(lower) / span))
    n_left = min(max(n_left, 0), grid_size - 1)
    n_right = grid_size - 1 - n_left
    left = np.linspace(lower, 0, n_left + 1)[:-1] if n_left else np.empty(0)
    right = np.linspace(0, upper, n_right + 1)
    return np.r_[left, right]


def _curve_area(density, grid):
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(density, grid))
    return float(np.trapz(density, grid))


def _prepare_ridge_data(
    distance_data,
    group_col,
    celltype_col,
    component_col,
    signed_distance_col,
    distance_col,
    side_col,
    aggregation_mode,
):
    required = {group_col, celltype_col}
    if aggregation_mode == "local":
        required.add(component_col)
    missing = required.difference(distance_data.columns)
    if missing:
        raise KeyError(f"distance_data缺少必要列：{sorted(missing)}")
    columns = [group_col, celltype_col]
    for col in (
        component_col,
        signed_distance_col,
        distance_col,
        side_col,
        "cell_index",
    ):
        if col in distance_data.columns and col not in columns:
            columns.append(col)
    data = distance_data.loc[:, columns].copy()
    data[group_col] = data[group_col].astype(str)
    data[celltype_col] = data[celltype_col].astype(str)
    if component_col in data:
        data[component_col] = data[component_col].astype(str)
    if signed_distance_col in data:
        data[signed_distance_col] = pd.to_numeric(
            data[signed_distance_col], errors="coerce"
        )
    else:
        if distance_col not in data or side_col not in data:
            raise KeyError(
                f"需要{signed_distance_col}，或者同时提供{distance_col}和{side_col}"
            )
        physical = pd.to_numeric(data[distance_col], errors="coerce")
        side = data[side_col].astype(str)
        valid = side.isin(["In", "Out"]) & physical.notna() & physical.ge(0)
        data.loc[~valid, signed_distance_col] = np.nan
        data.loc[valid, signed_distance_col] = np.where(
            side[valid].eq("In"), -physical[valid], physical[valid]
        )
    data = data[np.isfinite(data[signed_distance_col].to_numpy(float))].copy()
    if data.empty:
        raise ValueError("没有有效的有符号距离")
    if "cell_index" in data:
        dedup = [group_col, celltype_col]
        if component_col in data:
            dedup.append(component_col)
        dedup.append("cell_index")
        data = data.drop_duplicates(dedup, keep="first")
    return data


def _normalize_group_definitions(
    group_definitions: Mapping[str, str | Sequence[str]]
) -> dict[str, list[str]]:
    if not isinstance(group_definitions, Mapping) or not group_definitions:
        raise TypeError(
            "`group_definitions` must be a non-empty mapping from group names to one cell type or a sequence of cell types."
        )
    normalized = {}
    for group_name, labels in group_definitions.items():
        group_name = str(group_name)
        if isinstance(labels, str):
            labels = [labels]
        else:
            labels = list(labels)
        labels = _ordered_unique(labels)
        if not labels:
            raise ValueError(f"`group_definitions[{group_name!r}]` cannot be empty.")
        normalized[group_name] = labels
    return normalized


def _resolve_sample_mask(
    obs: pd.DataFrame, sample_key: str | None, sample: Any | None
) -> tuple[np.ndarray, str | None]:
    n_obs = len(obs)
    if sample_key is None:
        if sample is not None:
            raise ValueError("`sample` was provided but `sample_key` is None.")
        return (np.ones(n_obs, dtype=bool), None)
    if sample_key not in obs:
        raise KeyError(f"`{sample_key}` was not found in `adata.obs`.")
    sample_labels = obs[sample_key].astype(str)
    available = _ordered_unique(sample_labels)
    if sample is None:
        if len(available) != 1:
            raise ValueError(
                "`adata` contains multiple samples. Specify one sample with `sample=...` so independent tissue sections are not pooled."
            )
        selected = available[0]
    else:
        selected = str(sample)
        if selected not in set(available):
            raise ValueError(
                f"Sample {selected!r} was not found in `adata.obs[{sample_key!r}]`."
            )
    return (sample_labels.to_numpy() == selected, selected)


def _radius_neighbor_counts(
    reference_coords: np.ndarray,
    query_coords: np.ndarray,
    radius_um: float,
    chunk_size: int = 100000,
) -> np.ndarray:
    tree = cKDTree(reference_coords)
    result = np.empty(query_coords.shape[0], dtype=np.int32)
    for start in range(0, query_coords.shape[0], chunk_size):
        stop = min(start + chunk_size, query_coords.shape[0])
        query_chunk = query_coords[start:stop]
        counts = tree.query_ball_point(
            query_chunk, r=radius_um, return_length=True, workers=-1
        )
        counts = np.asarray(counts, dtype=np.int32)
        result[start:stop] = counts
    return result


def _resolve_region(
    density_data: pd.DataFrame, region: tuple[Sequence[float], Sequence[float]] | None
) -> tuple[float, float, float, float]:
    full_extent = density_data.attrs.get("extent")
    if full_extent is None:
        full_extent = (
            float(density_data["x"].min()),
            float(density_data["x"].max()),
            float(density_data["y"].min()),
            float(density_data["y"].max()),
        )
    if region is None:
        return tuple(full_extent)
    if len(region) != 2:
        raise ValueError("`region` must be None or ([x_min, y_min], [x_max, y_max]).")
    minimum = np.asarray(region[0], dtype=np.float64)
    maximum = np.asarray(region[1], dtype=np.float64)
    if minimum.shape != (2,) or maximum.shape != (2,):
        raise ValueError("`region` must be None or ([x_min, y_min], [x_max, y_max]).")
    if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
        raise ValueError("`region` contains non-finite values.")
    if np.any(maximum <= minimum):
        raise ValueError("`region` has invalid bounds.")
    return (float(minimum[0]), float(maximum[0]), float(minimum[1]), float(maximum[1]))


def _compute_group_scales(
    density_data: pd.DataFrame,
    groups: Sequence[str],
    value_col: str,
    scale_mode: str,
    clip_quantile: float,
) -> dict[str, float]:
    if scale_mode not in {
        "per_group_quantile",
        "global_quantile",
        "per_group_max",
        "global_max",
    }:
        raise ValueError(
            "`scale_mode` must be one of: 'per_group_quantile', 'global_quantile', 'per_group_max', or 'global_max'."
        )
    if not 0 < clip_quantile <= 1:
        raise ValueError("`clip_quantile` must be in (0, 1].")
    groups = [str(group) for group in groups]
    scales = {}
    if scale_mode.startswith("global"):
        values = density_data.loc[
            density_data["group"].isin(groups), value_col
        ].to_numpy(dtype=np.float64)
        positive = values[values > 0]
        if positive.size == 0:
            scale = 1.0
        elif scale_mode == "global_quantile":
            scale = float(np.quantile(positive, clip_quantile))
        else:
            scale = float(positive.max())
        scale = max(scale, np.finfo(float).eps)
        scales = {group: scale for group in groups}
    else:
        for group in groups:
            values = density_data.loc[
                density_data["group"] == group, value_col
            ].to_numpy(dtype=np.float64)
            positive = values[values > 0]
            if positive.size == 0:
                scale = 1.0
            elif scale_mode == "per_group_quantile":
                scale = float(np.quantile(positive, clip_quantile))
            else:
                scale = float(positive.max())
            scales[group] = max(scale, np.finfo(float).eps)
    return scales


def _white_to_color_cmap(color: Any, name: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, ["#FFFFFF", color], N=256)


def _draw_horizontal_colorbar_pair(
    fig,
    cax_top,
    cax_bottom,
    mappable_a,
    mappable_b,
    label_a: str,
    label_b: str,
    color_a: Any,
    color_b: Any,
    tick_mode: str = "midmax",
    title_fontsize: float = 13.0,
    tick_fontsize: float = 9.5,
    title_align: str = "center",
):

    def _one_bar(cax, mappable, label, color):
        cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
        vmax = float(mappable.norm.vmax)
        # Place the middle tick halfway along the nonlinear colorbar.
        center_value = float(mappable.norm.inverse(0.5))
        if tick_mode == "midmax":
            ticks = [center_value, vmax]
        else:
            ticks = [0.0, center_value, vmax]
        ticks = [tick for tick in ticks if np.isfinite(tick)]
        cb.set_ticks(ticks)
        cb.ax.tick_params(labelsize=tick_fontsize, pad=1.0, length=3)
        cb.outline.set_edgecolor("#4B4B4B")
        cb.outline.set_linewidth(0.8)
        title_x = {"left": 0.0, "center": 0.5, "right": 1.0}[title_align]
        title_ha = {"left": "left", "center": "center", "right": "right"}[title_align]
        cb.ax.set_title(
            label,
            color=color,
            fontsize=title_fontsize,
            pad=3,
            loc=title_align,
            x=title_x,
            ha=title_ha,
        )
        return cb

    cb_top = _one_bar(cax_top, mappable_a, label_a, color_a)
    cb_bottom = _one_bar(cax_bottom, mappable_b, label_b, color_b)
    return (cb_top, cb_bottom)


def _plot_density_single_panel(
    density_data: pd.DataFrame,
    group: str | None = None,
    region: tuple[Sequence[float], Sequence[float]] | None = None,
    color: Any = "#4AB4F8",
    value_col: str = "neighbor_count",
    scale_mode: str = "per_group_quantile",
    clip_quantile: float = 0.97,
    gamma: float = 0.9,
    point_size: float = 7.0,
    point_alpha: float = 1.0,
    point_edgecolor: Any = "none",
    point_linewidth: float = 0.0,
    figsize: tuple[float, float] = (8.2, 5.2),
    xlabel: str = "Spatial x",
    ylabel: str = "Spatial y",
    title: str | None = None,
    show_axes: bool = True,
    invert_y: bool = True,
    equal_aspect: bool = True,
    full_frame: bool = True,
    frame_color: str = "#3F3F3F",
    frame_linewidth: float = 1.0,
    add_colorbar: bool = True,
    colorbar_label: str | None = None,
    colorbar_title_fontsize: float = 11.5,
    colorbar_tick_fontsize: float = 8.5,
    rasterized: bool = False,
    save_path: str | None = None,
    dpi: int = 300,
    show: bool = True,
    return_data: bool = False,
):
    required = {
        "group",
        "cell_index",
        "cell_id",
        "x",
        "y",
        "neighbor_count",
        "density_per_um2",
    }
    missing = required.difference(density_data.columns)
    if missing:
        raise KeyError(f"`density_data` is missing columns: {sorted(missing)}")
    if value_col not in {"neighbor_count", "density_per_um2"}:
        raise ValueError("`value_col` must be 'neighbor_count' or 'density_per_um2'.")
    if gamma <= 0:
        raise ValueError("`gamma` must be greater than 0.")
    available_groups = _ordered_unique(density_data["group"].astype(str))
    if group is None:
        if len(available_groups) != 1:
            raise ValueError(
                "`density_data` contains multiple groups. Specify one with `group=...`."
            )
        group = available_groups[0]
    else:
        group = str(group)
        if group not in set(available_groups):
            raise KeyError(f"Group {group!r} was not found in `density_data['group']`.")
    extent = _resolve_region(density_data=density_data, region=region)
    x_min, x_max, y_min, y_max = extent
    panel = density_data[
        density_data["group"].astype(str).eq(group)
        & density_data["x"].between(x_min, x_max, inclusive="both")
        & density_data["y"].between(y_min, y_max, inclusive="both")
    ].copy()
    if panel.empty:
        raise ValueError("No selected cells were found in the requested `region`.")
    # Use the full density table for color scales, independent of the displayed crop.
    scales = _compute_group_scales(
        density_data=density_data,
        groups=[group],
        value_col=value_col,
        scale_mode=scale_mode,
        clip_quantile=clip_quantile,
    )
    scale = scales[group]
    panel = panel.sort_values(value_col, ascending=True, kind="mergesort")
    cmap = _white_to_color_cmap(color=color, name=f"{group}_white_to_color")
    norm = PowerNorm(gamma=gamma, vmin=0.0, vmax=scale, clip=True)
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        panel["x"],
        panel["y"],
        s=point_size,
        c=panel[value_col],
        cmap=cmap,
        norm=norm,
        alpha=point_alpha,
        edgecolors=point_edgecolor,
        linewidths=point_linewidth,
        rasterized=rasterized,
    )
    ax.set_xlim(x_min, x_max)
    if invert_y:
        ax.set_ylim(y_max, y_min)
    else:
        ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal" if equal_aspect else "auto")
    if show_axes:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    if title is not None:
        ax.set_title(title)
    ax.grid(False)
    ax.set_facecolor("white")
    for spine_name in ["top", "right", "bottom", "left"]:
        ax.spines[spine_name].set_visible(full_frame)
        ax.spines[spine_name].set_color(frame_color)
        ax.spines[spine_name].set_linewidth(frame_linewidth)
    cb = None
    if add_colorbar:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.035, pad=0.025)
        if colorbar_label is None:
            if value_col == "neighbor_count":
                radius_um = density_data.attrs.get("radius_um")
                if radius_um is None:
                    colorbar_label = "Local neighbor count"
                else:
                    colorbar_label = f"Neighbor count within {radius_um:g} μm"
            else:
                colorbar_label = "Cell density (cells / μm²)"
        cb.set_label(colorbar_label, fontsize=colorbar_title_fontsize)
        cb.ax.tick_params(labelsize=colorbar_tick_fontsize)
        cb.outline.set_linewidth(0.8)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            transparent=False,
        )
    if show:
        plt.show()
    if return_data:
        return (
            fig,
            ax,
            {
                "plot_data": panel,
                "group": group,
                "region": extent,
                "value_col": value_col,
                "group_scale": scale,
                "radius_um": density_data.attrs.get("radius_um"),
                "gamma": gamma,
                "scale_mode": scale_mode,
                "clip_quantile": clip_quantile,
                "colorbar": cb,
            },
        )
    return (fig, ax)
