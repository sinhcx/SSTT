from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import is_color_like, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
from pygam import LinearGAM, LogisticGAM, s
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import make_interp_spline


_DEFAULT_COLORS = {
    "SCC_MetInv": "#2171B5",
    "SCC_MigInv": "#9ECAE1",
    "SCC_Hypoxic": "#F16913",
    "SCC_Diff": "#FDBB84",
    "SCC_Basal": "#2F9E44",
    "SCC_Dysker": "#9BD18B",
}


def selected_border_curves(
    *curve_results,
    show_clicked: bool = True,
    show_guide: bool = True,
    show_snapped: bool = True,
    show_legend: bool = True,
    figsize: Tuple[float, float] = (6, 6),
    clicked_color: str = "red",
    guide_color: str = "orange",
    final_color: str = "blue",
    snapped_color: str = "gray",
    clicked_size: float = 5,
    snapped_size: float = 12,
    guide_width: float = 2,
    final_width: float = 3,
    alpha: float = 0.9,
    invert_y: bool = True,
    title: str = "Selected border curve",
    ax: Optional[Axes] = None,
    show: bool = True,
) -> Axes:
    if len(curve_results) == 0:
        raise ValueError("At least one curve result must be provided.")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    valid_count = 0

    for i, result in enumerate(curve_results):
        if result is None:
            print(f"[Warning] curve result {i + 1} is None and was skipped.")
            continue

        if not isinstance(result, dict):
            raise TypeError(
                f"curve result {i + 1} must be a dictionary returned by "
                "`InteractiveCurveSelector.get_curve()`."
            )

        final_curve = result.get("final_curve", None)
        if final_curve is None:
            print(f"[Warning] curve result {i + 1} has no final_curve and was skipped.")
            continue

        clicked_points = result.get("clicked_points", None)
        guide_curve = result.get("guide_curve", None)
        snapped_coords = result.get("snapped_coords", None)

        mode = result.get("mode", "unknown")
        side = result.get("side", "unknown")
        closed = result.get("closed", False)

        label_suffix = f"{i + 1}: {mode}, {side}, closed={closed}"

        if show_clicked and clicked_points is not None:
            clicked_points = np.asarray(clicked_points)
            ax.plot(
                clicked_points[:, 0],
                clicked_points[:, 1],
                "o-",
                color=clicked_color,
                linewidth=1.5,
                markersize=clicked_size,
                alpha=alpha,
                label=f"Clicked {label_suffix}",
            )

        if show_guide and guide_curve is not None:
            guide_curve = np.asarray(guide_curve)
            ax.plot(
                guide_curve[:, 0],
                guide_curve[:, 1],
                "--",
                color=guide_color,
                linewidth=guide_width,
                alpha=0.8,
                label=f"Guide {label_suffix}",
            )

        if show_snapped and snapped_coords is not None:
            snapped_coords = np.asarray(snapped_coords)
            ax.scatter(
                snapped_coords[:, 0],
                snapped_coords[:, 1],
                s=snapped_size,
                color=snapped_color,
                alpha=0.7,
                edgecolors="none",
                label=f"Snapped {label_suffix}",
            )

        final_curve = np.asarray(final_curve)
        ax.plot(
            final_curve[:, 0],
            final_curve[:, 1],
            "-",
            color=final_color,
            linewidth=final_width,
            alpha=alpha,
            label=f"Final {label_suffix}",
        )

        valid_count += 1

    if valid_count == 0:
        raise ValueError("No valid curve result was found.")

    ax.set_aspect("equal")

    if invert_y:
        ax.invert_yaxis()

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if show_legend:
        ax.legend(fontsize=8, frameon=False)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def border_normalized_distance(
    adata,
    dist_key: str = "normalized_dist",
    spatial_key: str = "spatial",
    title: str = "Digital layer along border",
    figsize: Tuple[float, float] = (12, 10),
    point_size: float = 30,
    colormap: Optional[str] = "coolwarm",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center_zero: bool = True,
    zero_value: float = 0.0,
    cbar_step: Optional[float] = 0.25,
    save_path: Optional[str] = None,
    dpi: int = 300,
    ax: Optional[Axes] = None,
    show: bool = True,
) -> Axes:
    if spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{spatial_key}'] not found.")

    if dist_key not in adata.obs.columns:
        raise KeyError(f"`dist_key='{dist_key}'` not found in adata.obs.")

    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)

    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            f"adata.obsm['{spatial_key}'] must be a 2D array with at least two columns."
        )

    x = coords[:, 0]
    y = coords[:, 1]

    dist_data = adata.obs[dist_key]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    fig = ax.figure

    if dist_key == "distance_interval":
        if colormap is None:
            colormap = "tab20"

        dist_data = dist_data.astype("category")
        categories = dist_data.cat.categories
        n_cats = len(categories)

        cmap = plt.get_cmap(colormap, n_cats)
        bounds = np.arange(n_cats + 1)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        scatter = ax.scatter(
            x,
            y,
            c=dist_data.cat.codes,
            cmap=cmap,
            norm=norm,
            s=point_size,
            edgecolor="none",
            alpha=0.9,
        )

        tick_positions = np.arange(n_cats) + 0.5

        cbar = fig.colorbar(
            scatter,
            ax=ax,
            boundaries=bounds,
            ticks=tick_positions,
            fraction=0.04,
            pad=0.02,
        )

        cbar.ax.set_yticklabels(categories)
        cbar.set_label("Distance interval", fontsize=11)
        cbar.ax.tick_params(labelsize=9)

    else:
        values = dist_data.astype(float).values
        finite_values = values[np.isfinite(values)]

        if finite_values.size == 0:
            raise ValueError(f"No finite values found in adata.obs['{dist_key}'].")

        data_min = float(np.nanmin(finite_values))
        data_max = float(np.nanmax(finite_values))

        if vmin is None:
            vmin = data_min
        else:
            vmin = float(vmin)

        if vmax is None:
            vmax = data_max
        else:
            vmax = float(vmax)

        if np.isclose(vmin, vmax):
            raise ValueError("Color range has zero width. Check vmin/vmax or data.")

        if colormap is None:
            colormap = "coolwarm" if center_zero else "viridis"

        if center_zero:
            if not (vmin < zero_value < vmax):
                raise ValueError(
                    f"`center_zero=True` requires vmin < zero_value < vmax. "
                    f"Got vmin={vmin}, zero_value={zero_value}, vmax={vmax}. "
                    f"If your data are one-sided, set center_zero=False."
                )

            cmap = _make_centered_compressed_cmap(
                base_cmap=colormap,
                vmin=vmin,
                vmax=vmax,
                center=zero_value,
                name=f"compressed_{colormap}",
            )

            # Keep linear normalization so colorbar tick spacing is true to
            # the numeric distance scale.
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        else:
            cmap = plt.get_cmap(colormap)
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        scatter = ax.scatter(
            x,
            y,
            c=values,
            cmap=cmap,
            norm=norm,
            s=point_size,
            edgecolor="none",
            alpha=0.9,
        )

        cbar = fig.colorbar(
            scatter,
            ax=ax,
            fraction=0.04,
            pad=0.02,
        )

        ticks = _make_even_ticks(
            vmin=vmin,
            vmax=vmax,
            step=cbar_step,
            include_zero=center_zero,
        )

        if ticks is not None:
            cbar.set_ticks(ticks)

        if dist_key == "normalized_dist":
            cbar.set_label("Normalized Distance", fontsize=11)
        elif dist_key == "signed_dist":
            cbar.set_label("Signed Distance", fontsize=11)
        elif dist_key == "abs_dist":
            cbar.set_label("Absolute Distance", fontsize=11)
        else:
            cbar.set_label(dist_key, fontsize=11)

        cbar.ax.tick_params(labelsize=9)
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax.invert_yaxis()
    ax.axis("equal")
    ax.grid(False)
    ax.set_xlabel("X Position", fontsize=12)
    ax.set_ylabel("Y Position", fontsize=12)
    ax.set_title(title, fontsize=16)

    if show:
        plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return ax


def celltype_composition_trend(
    adata,
    dist_key: str = "normalized_dist",
    celltype_key: str = "celltype",
    selected_celltypes: Optional[Sequence[str]] = None,
    n_bins: int = 50,
    distance_range: Optional[Tuple[float, float]] = None,
    min_cells_per_bin: int = 1,
    colormap: str = "tab20",
    custom_colors: Optional[Mapping[str, str]] = None,
    figsize: Tuple[float, float] = (6, 5),
    smooth: bool = True,
    gam_type: str = "logistic",
    lam: float = 0.6,
    line_width: float = 3.0,
    alpha: float = 1.0,
    xlabel: str = "Normalized distance to border",
    ylabel: str = "Cell type proportion",
    title: str = "Cell type composition along distance",
    legend: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
    return_data: bool = False,
):
    if dist_key not in adata.obs.columns:
        raise KeyError(f"'{dist_key}' not found in adata.obs.")

    if celltype_key not in adata.obs.columns:
        raise KeyError(f"'{celltype_key}' not found in adata.obs.")

    if n_bins <= 1:
        raise ValueError("`n_bins` must be greater than 1.")

    if min_cells_per_bin <= 0:
        raise ValueError("`min_cells_per_bin` must be positive.")

    if gam_type not in {"logistic", "linear"}:
        raise ValueError("`gam_type` must be either 'logistic' or 'linear'.")

    df = adata.obs[[dist_key, celltype_key]].copy()
    df[dist_key] = pd.to_numeric(df[dist_key], errors="coerce")
    df[celltype_key] = df[celltype_key].astype(str)
    df = df.dropna(subset=[dist_key, celltype_key])

    if df.empty:
        raise ValueError("No valid cells found after removing missing values.")

    if distance_range is None:
        dmin = float(df[dist_key].min())
        dmax = float(df[dist_key].max())
    else:
        dmin, dmax = map(float, distance_range)

    if not dmin < dmax:
        raise ValueError("`distance_range` must satisfy min < max.")

    bin_edges = np.linspace(dmin, dmax, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    df["_dist_bin"] = pd.cut(
        df[dist_key],
        bins=bin_edges,
        labels=np.arange(n_bins),
        include_lowest=True,
        right=False,
    )

    # Include rightmost endpoint.
    right_edge_mask = np.isclose(df[dist_key].to_numpy(), dmax)
    if right_edge_mask.any():
        df.loc[right_edge_mask, "_dist_bin"] = n_bins - 1

    df = df.dropna(subset=["_dist_bin"])
    df["_dist_bin"] = df["_dist_bin"].astype(int)

    bin_total = (
        df.groupby("_dist_bin", observed=False)
        .size()
        .reindex(np.arange(n_bins), fill_value=0)
    )

    valid_bins = bin_total >= min_cells_per_bin

    all_celltypes = _get_celltype_order(adata, celltype_key)

    if selected_celltypes is not None:
        selected_celltypes = [str(x) for x in selected_celltypes]
        celltypes = [ct for ct in selected_celltypes if ct in set(df[celltype_key])]
    else:
        celltypes = [ct for ct in all_celltypes if ct in set(df[celltype_key])]

    if len(celltypes) == 0:
        raise ValueError("No selected cell types found in the data.")

    count_table = (
        df.groupby(["_dist_bin", celltype_key], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=np.arange(n_bins), columns=celltypes, fill_value=0)
    )

    proportion_table = count_table.div(bin_total.replace(0, np.nan), axis=0)
    proportion_table.loc[~valid_bins, :] = np.nan

    proportion_df = proportion_table.reset_index().melt(
        id_vars="_dist_bin",
        var_name=celltype_key,
        value_name="proportion",
    )
    proportion_df["dist_mid"] = proportion_df["_dist_bin"].map(
        dict(zip(np.arange(n_bins), bin_centers))
    )
    proportion_df["bin_total"] = proportion_df["_dist_bin"].map(bin_total)

    color_dict = _get_celltype_colors(
        adata=adata,
        celltype_key=celltype_key,
        celltypes=celltypes,
        custom_colors=custom_colors,
        colormap=colormap,
    )

    fig, ax = plt.subplots(figsize=figsize)

    for ct in celltypes:
        subset = (
            proportion_df[proportion_df[celltype_key] == ct]
            .dropna(subset=["dist_mid", "proportion"])
            .sort_values("dist_mid")
        )

        if subset.shape[0] < 2:
            continue

        x = subset["dist_mid"].to_numpy(dtype=float)
        y = subset["proportion"].to_numpy(dtype=float)
        weights = subset["bin_total"].to_numpy(dtype=float)

        color = color_dict.get(ct, "#000000")

        if smooth and subset.shape[0] >= 4:
            x_grid = np.linspace(x.min(), x.max(), 300)
            X = x.reshape(-1, 1)
            X_grid = x_grid.reshape(-1, 1)

            try:
                if gam_type == "logistic":
                    gam = LogisticGAM(s(0), lam=lam).fit(X, y, weights=weights)
                    y_grid = gam.predict_mu(X_grid)
                else:
                    gam = LinearGAM(s(0), lam=lam).fit(X, y, weights=weights)
                    y_grid = gam.predict(X_grid)
                    y_grid = np.clip(y_grid, 0, 1)

                ax.plot(
                    x_grid,
                    y_grid,
                    label=ct,
                    color=color,
                    lw=line_width,
                    alpha=alpha,
                )

            except Exception:
                ax.plot(
                    x,
                    y,
                    label=ct,
                    color=color,
                    lw=line_width,
                    alpha=alpha,
                )
        else:
            ax.plot(
                x,
                y,
                label=ct,
                color=color,
                lw=line_width,
                alpha=alpha,
            )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=16)

    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)

    ax.tick_params(
        axis="both",
        which="major",
        direction="out",
        length=6,
        width=1,
        colors="black",
        left=True,
        bottom=True,
    )

    ax.grid(False)

    if legend:
        ax.legend(
            ncol=1,
            frameon=False,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            handlelength=0.8,
        )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    if return_data:
        return ax, proportion_df

    return ax


def celltype_composition_consensus(
    adatas,
    dist_key: str = "normalized_dist",
    celltype_key: str = "custom_celltype",
    selected_celltypes: Optional[Sequence[str]] = None,
    sample_names: Optional[Sequence[str]] = None,
    distance_range=(0.0, 1.0),
    n_bins: int = 20,
    normalize_selected: bool = True,
    smooth_sigma: float = 1.0,
    min_cells_per_bin: int = 5,
    celltype_order: str | Sequence[str] = "auto",
    order_metric: str = "center",
    colors=None,
    use_adata_colors: bool = True,
    figsize=(9.0, 5.6),
    plot_resolution: int = 600,
    full_frame: bool = True,
    show_sample_boundaries: bool = True,
    sample_boundary_color: str = "white",
    sample_boundary_alpha: float = 0.55,
    sample_boundary_linewidth: float = 0.6,
    show_dominant_strip: bool = True,
    strip_height_ratio: float = 0.18,
    strip_resolution: int = 600,
    strip_min_run_bins: int = 2,
    strip_row_separator_width: float = 1.0,
    show_strip_transition_guides: bool = True,
    strip_guide_color: str = "0.2",
    strip_guide_alpha: float = 0.45,
    strip_guide_linewidth: float = 0.7,
    strip_guide_linestyle=(0, (4, 4)),
    strip_tick_fontsize: float = 9.0,
    strip_title: Optional[str] = "Dominant cell type",
    strip_render_mode: str = "vector",
    strip_title_fontsize: float = 9.0,
    show_area_labels: bool = True,
    min_label_fraction: float = 0.07,
    title: Optional[str] = "Cross-sample consensus spatial transition",
    xlabel: str = "Normalized distance",
    ylabel: str = "Cell type proportion",
    legend: bool = True,
    legend_bbox_to_anchor=(0.0, 1.0),
    legend_width_ratio: float = 0.24,
    legend_marker_size: float = 10.0,
    save_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
    return_data: bool = False,
):
    prepared = _prepare_celltype_composition_consensus(
        adatas=adatas,
        dist_key=dist_key,
        celltype_key=celltype_key,
        selected_celltypes=selected_celltypes,
        sample_names=sample_names,
        distance_range=distance_range,
        n_bins=n_bins,
        normalize_selected=normalize_selected,
        smooth_sigma=smooth_sigma,
        min_cells_per_bin=min_cells_per_bin,
        celltype_order=celltype_order,
        order_metric=order_metric,
    )

    names = prepared["sample_names"]
    celltypes = prepared["celltype_order"]
    x = prepared["bin_centers"]
    sample_profiles = prepared["sample_profiles"]
    consensus = prepared["consensus_profile"]

    objects, _ = _coerce_adatas_and_names(adatas, sample_names)
    color_map = _resolve_colors(
        celltypes,
        colors,
        adatas=objects,
        celltype_key=celltype_key,
        use_adata_colors=use_adata_colors,
    )
    ordered_colors = [color_map[celltype] for celltype in celltypes]
    prepared["colors"] = color_map

    if plot_resolution < max(50, n_bins):
        raise ValueError("`plot_resolution` must be at least max(50, n_bins).")

    lo, hi = prepared["parameters"]["distance_range"]
    x_plot = np.linspace(lo, hi, int(plot_resolution))
    consensus_plot = _interpolate_profiles(consensus[None, :, :], x, x_plot)[0]
    sample_profiles_plot = _interpolate_profiles(sample_profiles, x, x_plot)

    if normalize_selected:
        consensus_total = consensus_plot.sum(axis=0)
        valid = consensus_total > 0
        consensus_plot[:, valid] /= consensus_total[valid]
        consensus_plot[:, ~valid] = 0.0

        sample_totals = sample_profiles_plot.sum(axis=1, keepdims=True)
        sample_profiles_plot = np.divide(
            sample_profiles_plot,
            sample_totals,
            out=np.zeros_like(sample_profiles_plot),
            where=sample_totals > 0,
        )

    if legend:
        ncols = 2
        width_ratios = [1.0, float(legend_width_ratio)]
        wspace = 0.03
    else:
        ncols = 1
        width_ratios = [1.0]
        wspace = 0.0

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    if show_dominant_strip:
        grid = fig.add_gridspec(
            nrows=2,
            ncols=ncols,
            height_ratios=[1.0, strip_height_ratio],
            width_ratios=width_ratios,
            hspace=0.03,
            wspace=wspace,
        )
        ax = fig.add_subplot(grid[0, 0])
        ax_strip = fig.add_subplot(grid[1, 0], sharex=ax)
        ax_legend = fig.add_subplot(grid[:, 1]) if legend else None
    else:
        grid = fig.add_gridspec(
            nrows=1,
            ncols=ncols,
            width_ratios=width_ratios,
            wspace=wspace,
        )
        ax = fig.add_subplot(grid[0, 0])
        ax_strip = None
        ax_legend = fig.add_subplot(grid[0, 1]) if legend else None

    if ax_legend is not None:
        ax_legend.axis("off")

    ax.stackplot(
        x_plot,
        *consensus_plot,
        colors=ordered_colors,
        baseline="zero",
        edgecolor="white",
        linewidth=0.7,
        alpha=0.97,
    )

    if show_sample_boundaries and len(names) > 0:
        for sample_idx in range(len(names)):
            cumulative = np.cumsum(sample_profiles_plot[sample_idx], axis=0)
            n_boundaries = len(celltypes) - 1 if normalize_selected else len(celltypes)
            for boundary_idx in range(n_boundaries):
                ax.plot(
                    x_plot,
                    cumulative[boundary_idx],
                    color=sample_boundary_color,
                    alpha=sample_boundary_alpha,
                    linewidth=sample_boundary_linewidth,
                    zorder=4,
                )

    if show_area_labels:
        lower = np.vstack(
            [
                np.zeros_like(x_plot),
                np.cumsum(consensus_plot, axis=0)[:-1],
            ]
        )
        upper = np.cumsum(consensus_plot, axis=0)
        for type_idx, celltype in enumerate(celltypes):
            thickness = consensus_plot[type_idx]
            peak_idx = int(np.argmax(thickness))
            if x_plot.size >= 3:
                safe_min = max(1, int(round(0.08 * (x_plot.size - 1))))
                safe_max = min(x_plot.size - 2, int(round(0.88 * (x_plot.size - 1))))
                peak_idx = int(np.clip(peak_idx, safe_min, safe_max))
            if thickness[peak_idx] < min_label_fraction:
                continue
            y = 0.5 * (lower[type_idx, peak_idx] + upper[type_idx, peak_idx])
            ax.text(
                x_plot[peak_idx],
                y,
                celltype,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="semibold",
                color=_text_color_for_fill(ordered_colors[type_idx]),
                zorder=5,
                clip_on=True,
            )

    ax.set_xlim(lo, hi)
    if normalize_selected:
        ax.set_ylim(0.0, 1.0)
    else:
        total_max = float(np.max(consensus.sum(axis=0)))
        ax.set_ylim(0.0, max(1.0, total_max * 1.05))

    ax.set_ylabel(ylabel)
    if not show_dominant_strip:
        ax.set_xlabel(xlabel)
    else:
        ax.tick_params(axis="x", labelbottom=False)

    if title:
        ax.set_title(title, pad=10, fontweight="semibold")

    if full_frame:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.9)
    else:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.grid(False)

    if legend:
        if legend_marker_size <= 0:
            raise ValueError("`legend_marker_size` must be greater than 0.")

        handles = [
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="None",
                markersize=float(legend_marker_size),
                markerfacecolor=color_map[celltype],
                markeredgecolor="none",
                label=celltype,
            )
            for celltype in reversed(celltypes)
        ]
        if show_sample_boundaries:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color="0.45",
                    alpha=0.9,
                    linewidth=1.0,
                    label="Individual samples",
                )
            )
        legend_target = ax_legend if ax_legend is not None else ax
        legend_target.legend(
            handles=handles,
            title="Cell type",
            bbox_to_anchor=legend_bbox_to_anchor,
            loc="upper left",
            frameon=False,
            borderaxespad=0.0,
            handlelength=1.0,
            handletextpad=0.7,
            labelspacing=0.8,
        )

    if ax_strip is not None:
        if strip_resolution < 50:
            raise ValueError("`strip_resolution` must be at least 50.")
        if strip_min_run_bins < 1:
            raise ValueError("`strip_min_run_bins` must be at least 1.")

        if strip_render_mode not in {"vector", "rgb"}:
            raise ValueError("`strip_render_mode` must be 'vector' or 'rgb'.")

        x_dense = np.linspace(lo, hi, int(strip_resolution))
        dense_profiles = _interpolate_profiles(sample_profiles, x, x_dense)
        dominant_dense = np.argmax(dense_profiles, axis=1)

        min_run_points = max(
            1,
            int(round(strip_min_run_bins * strip_resolution / max(n_bins, 1))),
        )
        if strip_min_run_bins > 1:
            dominant_dense = np.vstack(
                [
                    _merge_short_dominant_runs(
                        dominant_dense[sample_idx],
                        dense_profiles[sample_idx],
                        min_run_points=min_run_points,
                    )
                    for sample_idx in range(len(names))
                ]
            )

        if strip_render_mode == "vector":
            # Draw each contiguous dominant-state run as a solid rectangle.
            x_edges_dense = np.empty(x_dense.size + 1, dtype=float)
            x_edges_dense[0] = lo
            x_edges_dense[-1] = hi
            x_edges_dense[1:-1] = 0.5 * (x_dense[:-1] + x_dense[1:])

            for sample_idx in range(len(names)):
                states = dominant_dense[sample_idx]
                change_points = np.flatnonzero(states[1:] != states[:-1]) + 1
                run_starts = np.r_[0, change_points]
                run_ends = np.r_[change_points, states.size]

                for start, end in zip(run_starts, run_ends):
                    state_idx = int(states[start])
                    ax_strip.add_patch(
                        Rectangle(
                            (x_edges_dense[start], sample_idx),
                            x_edges_dense[end] - x_edges_dense[start],
                            1.0,
                            facecolor=ordered_colors[state_idx],
                            edgecolor="none",
                            linewidth=0.0,
                            antialiased=False,
                            zorder=1,
                        )
                    )
        else:
            # Render the dominant-state strip as an RGB image.
            rgb_lut = np.asarray([to_rgb(color) for color in ordered_colors])
            strip_rgb = rgb_lut[dominant_dense]
            ax_strip.imshow(
                strip_rgb,
                interpolation="nearest",
                aspect="auto",
                extent=(lo, hi, len(names), 0),
                rasterized=True,
                zorder=1,
            )

        if strip_row_separator_width > 0:
            for y in range(1, len(names)):
                ax_strip.axhline(
                    y,
                    color="white",
                    linewidth=strip_row_separator_width,
                    zorder=3,
                )

        consensus_dense = _interpolate_profiles(consensus[None, :, :], x, x_dense)[0]
        consensus_dominant = np.argmax(consensus_dense, axis=0)
        if strip_min_run_bins > 1:
            consensus_dominant = _merge_short_dominant_runs(
                consensus_dominant,
                consensus_dense,
                min_run_points=min_run_points,
            )

        transition_positions = _dominant_transition_positions(
            consensus_dominant, x_dense
        )
        if show_strip_transition_guides:
            for transition_x in transition_positions:
                ax_strip.axvline(
                    transition_x,
                    color=strip_guide_color,
                    alpha=strip_guide_alpha,
                    linewidth=strip_guide_linewidth,
                    linestyle=strip_guide_linestyle,
                    zorder=4,
                )

        ax_strip.set_ylim(len(names), 0)
        ax_strip.set_yticks(np.arange(len(names)) + 0.5)
        ax_strip.set_yticklabels(names, fontsize=strip_tick_fontsize)
        if strip_title:
            ax_strip.set_title(
                strip_title,
                loc="left",
                pad=4,
                fontsize=strip_title_fontsize,
                fontweight="semibold",
            )
        ax_strip.set_xlabel(xlabel)
        ax_strip.set_xticks(np.linspace(lo, hi, 5))
        ax_strip.tick_params(
            axis="x",
            labelsize=strip_tick_fontsize,
            width=0.9,
            length=4,
        )
        ax_strip.tick_params(axis="y", length=0, pad=6)
        ax_strip.spines["top"].set_visible(False)
        ax_strip.spines["right"].set_visible(False)
        ax_strip.spines["left"].set_visible(False)
        ax_strip.spines["bottom"].set_linewidth(0.9)
        ax_strip.grid(False)

        prepared["strip_x"] = x_dense
        prepared["strip_dominant_indices"] = dominant_dense
        prepared["strip_transition_positions"] = transition_positions
        prepared["strip_render_mode"] = strip_render_mode

    prepared["plot_x"] = x_plot
    prepared["plot_consensus_profile"] = consensus_plot
    prepared["plot_sample_profiles"] = sample_profiles_plot

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    axes = (ax, ax_strip) if ax_strip is not None else ax
    if return_data:
        return fig, axes, prepared
    return fig, axes


def distance_trend(
    adatas,
    key,
    dist_key="normalized_dist",
    layer=None,
    sample_names=None,
    colors=None,
    figsize=(5, 3.5),
    lam=5,
    normalize=True,
    xlabel="Normalized distance to border",
    ylabel="Relative value",
    title=None,
    legend=True,
    save_path=None,
    dpi=600,
    show=True,
):
    if not isinstance(adatas, (list, tuple)):
        adatas = [adatas]
    n_samples = len(adatas)
    if n_samples < 1:
        raise ValueError("`adatas` must contain at least one AnnData object.")
    if sample_names is None:
        sample_names = [f"Sample {i + 1}" for i in range(n_samples)]
    if len(sample_names) != n_samples:
        raise ValueError("`sample_names` must have the same length as `adatas`.")
    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % cmap.N) for i in range(n_samples)]
    if len(colors) != n_samples:
        raise ValueError("`colors` must have the same length as `adatas`.")

    dfs = []
    value_types = []
    for adata in adatas:
        df, value_type = _get_distance_trend_data(adata, key, dist_key, layer)
        dfs.append(df)
        value_types.append(value_type)
    if len(set(value_types)) != 1:
        raise ValueError(
            f"`key` has inconsistent source types across samples: {value_types}"
        )

    all_vals = np.concatenate([df["value"].to_numpy() for df in dfs])
    # Use a common value range across all cleaned samples.
    if normalize:
        v_min, v_max = (np.nanmin(all_vals), np.nanmax(all_vals))
        for df in dfs:
            if np.isclose(v_min, v_max):
                df["y_plot"] = 0.0
            else:
                df["y_plot"] = (df["value"] - v_min) / (v_max - v_min)
        y_label = ylabel
    else:
        for df in dfs:
            df["y_plot"] = df["value"]
        y_label = key

    fig, ax = plt.subplots(figsize=figsize)
    x_min = min((df["dist"].min() for df in dfs))
    x_max = max((df["dist"].max() for df in dfs))
    x_grid = np.linspace(x_min, x_max, 300).reshape(-1, 1)

    for df, label, color in zip(dfs, sample_names, colors):
        if df.shape[0] < 4:
            continue
        gam = LinearGAM(s(0, lam=lam)).fit(
            df["dist"].to_numpy().reshape(-1, 1), df["y_plot"].to_numpy()
        )
        y_pred = gam.predict(x_grid)
        ax.plot(x_grid.ravel(), y_pred, label=label, color=color, lw=3.2, zorder=3)

    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    if title is None:
        title = f"{key}"
    ax.set_title(title, fontsize=15, pad=25, fontweight="bold")
    if legend:
        leg = ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=min(n_samples, 4),
            frameon=False,
            fontsize=10,
            handletextpad=0.4,
            columnspacing=1.5,
            borderaxespad=0.15,
        )
        for line in leg.get_lines():
            line.set_linewidth(4.5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return ax


def fuzzy_clusters(
    normalized_matrix: np.ndarray,
    cluster_centers: pd.DataFrame,
    results_df: pd.DataFrame,
    bin_centers: Optional[Sequence[float]] = None,
    n_cols: int = 3,
    figsize_per_panel: Tuple[float, float] = (4, 3),
    smooth: bool = True,
    smooth_points: int = 300,
    line_color: str = "#B22222",
    fill_color: str = "salmon",
    fill_alpha: float = 0.2,
    line_width: float = 3.5,
    title_prefix: str = "Cluster",
    xlabel: str = "Normalized distance to border",
    ylabel: str = "Normalized expression",
    save_path: Optional[str] = None,
    dpi: int = 300,
    show: bool = True,
):
    if "cluster" not in results_df.columns:
        raise KeyError("`results_df` must contain a 'cluster' column.")

    n_clusters = cluster_centers.shape[0]
    assigned_clusters = results_df["cluster"].to_numpy()

    x_raw = _infer_x_positions(
        cluster_centers=cluster_centers,
        bin_centers=bin_centers,
    )

    if len(x_raw) != cluster_centers.shape[1]:
        raise ValueError("`bin_centers` length must match the number of distance bins.")

    n_rows = (n_clusters + n_cols - 1) // n_cols
    fig_w = figsize_per_panel[0] * n_cols
    fig_h = figsize_per_panel[1] * n_rows

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )

    axes_flat = axes.ravel()

    for i in range(n_clusters):
        ax = axes_flat[i]

        cluster_indices = np.where(assigned_clusters == i)[0]

        if len(cluster_indices) == 0:
            ax.set_title(f"{title_prefix} {i} (0 genes)")
            ax.axis("off")
            continue

        cluster_data = normalized_matrix[cluster_indices]
        center_line = cluster_centers.iloc[i].to_numpy(dtype=float)
        std_dev = np.std(cluster_data, axis=0)

        if smooth and len(x_raw) >= 4:
            x_plot, center_plot = _smooth_line(x_raw, center_line, smooth_points)
            _, std_plot = _smooth_line(x_raw, std_dev, smooth_points)
        else:
            x_plot = x_raw
            center_plot = center_line
            std_plot = std_dev

        ax.fill_between(
            x_plot,
            center_plot - std_plot,
            center_plot + std_plot,
            color=fill_color,
            alpha=fill_alpha,
            zorder=1,
        )

        ax.plot(
            x_plot,
            center_plot,
            color=line_color,
            linewidth=line_width,
            label=f"{title_prefix} {i}",
            zorder=2,
        )

        ax.set_title(
            f"{title_prefix} {i} ({len(cluster_indices)} genes)",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

        if i % n_cols == 0:
            ax.set_ylabel(ylabel, fontsize=12)

        if i >= (n_rows - 1) * n_cols:
            ax.set_xlabel(xlabel, fontsize=12)

        ax.axhline(
            y=0,
            color="black",
            linestyle="--",
            alpha=0.2,
            lw=1,
        )

        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=10)

    for j in range(n_clusters, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()


def _make_centered_compressed_cmap(
    base_cmap,
    vmin: float,
    vmax: float,
    center: float = 0.0,
    name: str = "centered_compressed_cmap",
    n_colors: int = 256,
):
    if not (vmin < center < vmax):
        raise ValueError(
            f"Require vmin < center < vmax. "
            f"Got vmin={vmin}, center={center}, vmax={vmax}."
        )

    base = plt.get_cmap(base_cmap)
    center_pos = (center - vmin) / (vmax - vmin)

    n_left = max(2, int(np.round(n_colors * center_pos)))
    n_right = max(2, n_colors - n_left)

    left_colors = base(np.linspace(0.0, 0.5, n_left, endpoint=False))
    right_colors = base(np.linspace(0.5, 1.0, n_right))

    colors = np.vstack([left_colors, right_colors])

    return mcolors.ListedColormap(colors, name=name)


def _make_even_ticks(
    vmin: float,
    vmax: float,
    step: Optional[float] = 0.25,
    include_zero: bool = True,
):
    if step is None:
        return None

    step = float(step)
    if step <= 0:
        raise ValueError("`cbar_step` must be positive.")

    tick_start = np.ceil(vmin / step) * step
    tick_end = np.floor(vmax / step) * step

    ticks = np.arange(
        tick_start,
        tick_end + step * 0.5,
        step,
    )

    if include_zero and vmin <= 0 <= vmax:
        ticks = np.concatenate([ticks, [0.0]])

    ticks = np.unique(np.round(ticks, 10))
    ticks = ticks[(ticks >= vmin - 1e-9) & (ticks <= vmax + 1e-9)]
    ticks[np.isclose(ticks, 0)] = 0.0

    return ticks


def _get_celltype_order(adata, celltype_key: str) -> list[str]:
    values = adata.obs[celltype_key]

    if pd.api.types.is_categorical_dtype(values):
        return [str(x) for x in values.cat.categories]

    return sorted(values.dropna().astype(str).unique().tolist())


def _get_celltype_colors(
    adata,
    celltype_key: str,
    celltypes: Sequence[str],
    custom_colors: Optional[Mapping[str, str]] = None,
    colormap: str = "tab20",
) -> dict[str, str]:
    color_dict = {}

    if custom_colors is not None:
        color_dict.update({str(k): v for k, v in custom_colors.items()})

    uns_key = f"{celltype_key}_colors"

    if uns_key in adata.uns and pd.api.types.is_categorical_dtype(
        adata.obs[celltype_key]
    ):
        cats = [str(x) for x in adata.obs[celltype_key].cat.categories]
        uns_colors = list(adata.uns[uns_key])

        if len(uns_colors) >= len(cats):
            color_dict.update(
                {
                    cat: uns_colors[i]
                    for i, cat in enumerate(cats)
                    if cat not in color_dict
                }
            )

    missing = [ct for ct in celltypes if ct not in color_dict]

    if len(missing) > 0:
        palette = sns.color_palette(colormap, len(missing))
        color_dict.update({ct: palette[i] for i, ct in enumerate(missing)})

    return color_dict


def _coerce_adatas_and_names(adatas, sample_names=None):
    if isinstance(adatas, Mapping):
        names = [str(x) for x in adatas.keys()]
        objects = list(adatas.values())
    else:
        objects = list(adatas)
        if sample_names is None:
            names = [f"S{i + 1}" for i in range(len(objects))]
        else:
            names = [str(x) for x in sample_names]

    if not objects:
        raise ValueError("`adatas` must contain at least one AnnData object.")
    if len(names) != len(objects):
        raise ValueError("`sample_names` must have the same length as `adatas`.")

    return objects, names


def _smooth_nan_curve(values: np.ndarray, sigma: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)

    if not valid.any():
        return np.zeros_like(values)

    x = np.arange(values.size, dtype=float)
    filled = np.interp(x, x[valid], values[valid])

    if sigma is not None and sigma > 0:
        filled = gaussian_filter1d(filled, sigma=float(sigma), mode="nearest")

    return np.clip(filled, 0.0, None)


def _extract_adata_color_map(adatas, celltype_key: str) -> dict[str, Any]:
    color_key = f"{celltype_key}_colors"
    color_map: dict[str, Any] = {}

    for adata in adatas:
        if color_key not in adata.uns or celltype_key not in adata.obs.columns:
            continue

        series = adata.obs[celltype_key]
        if not isinstance(series.dtype, pd.CategoricalDtype):
            continue

        categories = [str(x) for x in series.cat.categories]
        category_colors = list(adata.uns[color_key])
        if len(category_colors) < len(categories):
            continue

        for category, color in zip(categories, category_colors):
            if category not in color_map and is_color_like(color):
                color_map[category] = color

    return color_map


def _resolve_colors(
    celltypes,
    colors=None,
    *,
    adatas=None,
    celltype_key: Optional[str] = None,
    use_adata_colors: bool = True,
):
    fallback = plt.get_cmap("tab20")

    adata_colors = {}
    if colors is None and use_adata_colors and adatas is not None and celltype_key:
        adata_colors = _extract_adata_color_map(adatas, celltype_key)

    if colors is None:
        color_map = {
            celltype: adata_colors.get(
                celltype,
                _DEFAULT_COLORS.get(celltype, fallback(i % 20)),
            )
            for i, celltype in enumerate(celltypes)
        }
    elif isinstance(colors, Mapping):
        color_map = {
            celltype: colors.get(
                celltype,
                adata_colors.get(
                    celltype,
                    _DEFAULT_COLORS.get(celltype, fallback(i % 20)),
                ),
            )
            for i, celltype in enumerate(celltypes)
        }
    else:
        colors = list(colors)
        if len(colors) < len(celltypes):
            raise ValueError(
                "`colors` must contain at least one color per selected cell type."
            )
        color_map = dict(zip(celltypes, colors))

    return color_map


def _text_color_for_fill(color) -> str:
    r, g, b = to_rgb(color)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 0.62 else "white"


def _interpolate_profiles(
    profiles: np.ndarray,
    x: np.ndarray,
    x_dense: np.ndarray,
) -> np.ndarray:
    profiles = np.asarray(profiles, dtype=float)
    flat = profiles.reshape(-1, profiles.shape[-1])
    dense = np.vstack([np.interp(x_dense, x, row) for row in flat])
    return dense.reshape(*profiles.shape[:-1], x_dense.size)


def _merge_short_dominant_runs(
    labels: np.ndarray,
    profiles: np.ndarray,
    min_run_points: int,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=int).copy()
    profiles = np.asarray(profiles, dtype=float)

    if min_run_points <= 1 or labels.size < 3:
        return labels

    for _ in range(12):
        starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
        ends = np.r_[starts[1:], labels.size]
        changed = False

        for run_idx, (start, end) in enumerate(zip(starts, ends)):
            if end - start >= min_run_points:
                continue

            candidates = []
            if run_idx > 0:
                candidates.append(int(labels[starts[run_idx - 1]]))
            if run_idx + 1 < len(starts):
                candidates.append(int(labels[starts[run_idx + 1]]))
            candidates = list(dict.fromkeys(candidates))

            if not candidates:
                continue

            scores = [profiles[candidate, start:end].mean() for candidate in candidates]
            replacement = candidates[int(np.argmax(scores))]
            labels[start:end] = replacement
            changed = True

        if not changed:
            break

    return labels


def _dominant_transition_positions(labels: np.ndarray, x: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    change_idx = np.flatnonzero(labels[1:] != labels[:-1])
    if change_idx.size == 0:
        return np.empty(0, dtype=float)
    return 0.5 * (x[change_idx] + x[change_idx + 1])


def _get_distance_trend_data(adata, key, dist_key, layer):
    if dist_key not in adata.obs.columns:
        raise KeyError(f"'{dist_key}' not found in adata.obs.")
    if key in adata.var_names:
        values = adata[:, key].layers[layer] if layer else adata[:, key].X
        if hasattr(values, "toarray"):
            values = values.toarray()
        values = np.asarray(values).reshape(-1)
        value_type = "gene"
    elif key in adata.obs.columns:
        values = pd.to_numeric(adata.obs[key], errors="coerce").to_numpy()
        value_type = "obs"
    else:
        raise KeyError(f"'{key}' is neither in adata.var_names nor adata.obs.columns.")
    df = pd.DataFrame(
        {
            "value": values,
            "dist": pd.to_numeric(adata.obs[dist_key], errors="coerce").to_numpy(),
        }
    ).dropna()
    return (df, value_type)


def _infer_x_positions(
    cluster_centers: pd.DataFrame,
    bin_centers: Optional[Sequence[float]] = None,
) -> np.ndarray:
    if bin_centers is not None:
        return np.asarray(bin_centers, dtype=float)

    try:
        return cluster_centers.columns.astype(float).to_numpy()
    except Exception:
        return np.arange(cluster_centers.shape[1], dtype=float)


def _smooth_line(
    x: np.ndarray,
    y: np.ndarray,
    smooth_points: int = 300,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    unique_x, unique_idx = np.unique(x, return_index=True)
    unique_y = y[unique_idx]

    if len(unique_x) < 4:
        return unique_x, unique_y

    k = min(3, len(unique_x) - 1)

    x_smooth = np.linspace(unique_x.min(), unique_x.max(), smooth_points)
    spline = make_interp_spline(unique_x, unique_y, k=k)

    return x_smooth, spline(x_smooth)


def _prepare_celltype_composition_consensus(
    adatas,
    dist_key: str = "normalized_dist",
    celltype_key: str = "custom_celltype",
    selected_celltypes: Optional[Sequence[str]] = None,
    sample_names: Optional[Sequence[str]] = None,
    distance_range=(0.0, 1.0),
    n_bins: int = 20,
    normalize_selected: bool = True,
    smooth_sigma: float = 1.0,
    min_cells_per_bin: int = 5,
    celltype_order: str | Sequence[str] = "auto",
    order_metric: str = "center",
):
    objects, names = _coerce_adatas_and_names(adatas, sample_names)

    if n_bins < 2:
        raise ValueError("`n_bins` must be at least 2.")
    if min_cells_per_bin < 1:
        raise ValueError("`min_cells_per_bin` must be at least 1.")

    lo, hi = map(float, distance_range)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError("`distance_range` must satisfy finite lower < upper.")

    for name, adata in zip(names, objects):
        missing = [
            key for key in (dist_key, celltype_key) if key not in adata.obs.columns
        ]
        if missing:
            raise KeyError(f"Sample {name!r} is missing obs column(s): {missing}")

    if selected_celltypes is None:
        shared = None
        for adata in objects:
            current = set(adata.obs[celltype_key].dropna().astype(str))
            shared = current if shared is None else shared.intersection(current)
        selected_celltypes = sorted(shared or [])
    else:
        selected_celltypes = [str(x) for x in selected_celltypes]

    if not selected_celltypes:
        raise ValueError("No selected cell types are available for plotting.")
    if len(set(selected_celltypes)) != len(selected_celltypes):
        raise ValueError("`selected_celltypes` contains duplicates.")

    bin_edges = np.linspace(lo, hi, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n_types = len(selected_celltypes)

    sample_profiles = np.zeros((len(objects), n_types, n_bins), dtype=float)
    sample_bin_counts = np.zeros((len(objects), n_bins), dtype=int)

    for sample_idx, (name, adata) in enumerate(zip(names, objects)):
        obs = adata.obs[[dist_key, celltype_key]].copy()
        obs[dist_key] = pd.to_numeric(obs[dist_key], errors="coerce")
        obs[celltype_key] = obs[celltype_key].astype("object").astype(str)
        obs = obs.loc[
            obs[dist_key].notna() & obs[dist_key].between(lo, hi, inclusive="both")
        ].copy()

        if obs.empty:
            raise ValueError(
                f"Sample {name!r} has no cells in distance_range={distance_range}."
            )

        distances = obs[dist_key].to_numpy(dtype=float)
        labels = obs[celltype_key].to_numpy(dtype=object)

        bin_idx = np.searchsorted(bin_edges, distances, side="right") - 1
        bin_idx[distances == hi] = n_bins - 1
        valid_bin = (bin_idx >= 0) & (bin_idx < n_bins)
        bin_idx = bin_idx[valid_bin]
        labels = labels[valid_bin]

        all_counts = np.bincount(bin_idx, minlength=n_bins).astype(float)
        sample_bin_counts[sample_idx] = all_counts.astype(int)

        selected_mask = np.isin(labels, selected_celltypes)
        selected_counts = np.bincount(bin_idx[selected_mask], minlength=n_bins).astype(
            float
        )
        denominator = selected_counts if normalize_selected else all_counts

        raw = np.full((n_types, n_bins), np.nan, dtype=float)
        valid_denominator = denominator >= min_cells_per_bin

        for type_idx, celltype in enumerate(selected_celltypes):
            type_counts = np.bincount(
                bin_idx[labels == celltype],
                minlength=n_bins,
            ).astype(float)
            raw[type_idx, valid_denominator] = (
                type_counts[valid_denominator] / denominator[valid_denominator]
            )

        smoothed = np.vstack(
            [
                _smooth_nan_curve(raw[type_idx], smooth_sigma)
                for type_idx in range(n_types)
            ]
        )

        if normalize_selected:
            totals = smoothed.sum(axis=0)
            valid_total = totals > 0
            smoothed[:, valid_total] /= totals[valid_total]
            smoothed[:, ~valid_total] = 0.0

        sample_profiles[sample_idx] = smoothed

    consensus = sample_profiles.mean(axis=0)
    if normalize_selected:
        totals = consensus.sum(axis=0)
        valid_total = totals > 0
        consensus[:, valid_total] /= totals[valid_total]
        consensus[:, ~valid_total] = 0.0

    if isinstance(celltype_order, str):
        if celltype_order != "auto":
            raise ValueError("String `celltype_order` must be 'auto'.")
        if order_metric not in {"center", "peak"}:
            raise ValueError("`order_metric` must be 'center' or 'peak'.")

        if order_metric == "peak":
            scores = np.array(
                [
                    (
                        bin_centers[np.argmax(consensus[i])]
                        if np.any(consensus[i] > 0)
                        else np.inf
                    )
                    for i in range(n_types)
                ]
            )
        else:
            scores = np.array(
                [
                    (
                        np.sum(bin_centers * consensus[i]) / np.sum(consensus[i])
                        if np.sum(consensus[i]) > 0
                        else np.inf
                    )
                    for i in range(n_types)
                ]
            )
        order_idx = np.argsort(scores, kind="stable")
        ordered_celltypes = [selected_celltypes[i] for i in order_idx]
    else:
        ordered_celltypes = [str(x) for x in celltype_order]
        if set(ordered_celltypes) != set(selected_celltypes):
            raise ValueError(
                "Custom `celltype_order` must contain exactly the selected cell types."
            )
        order_idx = np.array([selected_celltypes.index(x) for x in ordered_celltypes])

    ordered_sample_profiles = sample_profiles[:, order_idx, :]
    ordered_consensus = consensus[order_idx, :]
    dominant_indices = np.argmax(ordered_sample_profiles, axis=1)
    dominant_states = np.asarray(ordered_celltypes, dtype=object)[dominant_indices]

    sample_long = []
    for sample_idx, sample_name in enumerate(names):
        for type_idx, celltype in enumerate(ordered_celltypes):
            sample_long.append(
                pd.DataFrame(
                    {
                        "sample": sample_name,
                        "distance": bin_centers,
                        "celltype": celltype,
                        "proportion": ordered_sample_profiles[sample_idx, type_idx],
                    }
                )
            )

    consensus_long = []
    for type_idx, celltype in enumerate(ordered_celltypes):
        consensus_long.append(
            pd.DataFrame(
                {
                    "distance": bin_centers,
                    "celltype": celltype,
                    "mean_proportion": ordered_consensus[type_idx],
                    "sd_proportion": ordered_sample_profiles[:, type_idx, :].std(
                        axis=0, ddof=0
                    ),
                }
            )
        )

    return {
        "sample_names": names,
        "celltype_order": ordered_celltypes,
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "sample_profiles": ordered_sample_profiles,
        "consensus_profile": ordered_consensus,
        "dominant_indices": dominant_indices,
        "dominant_states": dominant_states,
        "sample_bin_counts": sample_bin_counts,
        "sample_long": pd.concat(sample_long, ignore_index=True),
        "consensus_long": pd.concat(consensus_long, ignore_index=True),
        "parameters": {
            "dist_key": dist_key,
            "celltype_key": celltype_key,
            "distance_range": (lo, hi),
            "n_bins": n_bins,
            "normalize_selected": normalize_selected,
            "smooth_sigma": smooth_sigma,
            "min_cells_per_bin": min_cells_per_bin,
            "order_metric": order_metric,
        },
    }
