from __future__ import annotations

import math
import warnings
from pathlib import Path as FilePath
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spatialdata_plot  # noqa: F401  # Register the SpatialData plotting accessor.
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import to_hex, to_rgba
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch, Polygon as MplPolygon
from matplotlib.path import Path
from matplotlib.ticker import PercentFormatter
from scipy import sparse
from spatialdata import bounding_box_query
from spatialdata.models import PointsModel
from spatialdata.transformations import get_transformation, set_transformation


def spatial_network(
    adata,
    network_key: str = "sstt_neighbor_network",
    spatial_key: str = "spatial",
    color: Optional[Union[str, Sequence]] = None,
    cells: Optional[Iterable] = None,
    min_coordinate: Optional[Sequence[float]] = None,
    max_coordinate: Optional[Sequence[float]] = None,
    max_nodes: Optional[int] = None,
    max_edges: Optional[int] = 100_000,
    random_state: int = 0,
    node_size: float = 2.0,
    node_alpha: float = 0.9,
    edge_width: float = 0.2,
    edge_alpha: float = 0.18,
    edge_color: str = "lightgray",
    cmap: str = "viridis",
    legend: bool = False,
    max_legend_categories: int = 20,
    invert_y: bool = True,
    rasterized: bool = True,
    figsize: Tuple[float, float] = (6, 6),
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    show: bool = True,
    cell_boundary: Optional[Union[Mapping, Sequence]] = None,
    cell_id_col: Optional[str] = "cell_id",
    boundary_color: str = "#444444",
    boundary_width: float = 0.45,
    boundary_alpha: float = 0.75,
    boundary_fill_alpha: float = 0.3,
) -> Axes:
    if spatial_key not in adata.obsm:
        raise KeyError(f"Spatial coordinates not found in adata.obsm['{spatial_key}'].")

    if network_key not in adata.obsp:
        raise KeyError(f"Network not found in adata.obsp['{network_key}'].")

    if not 0 <= boundary_alpha <= 1 or not 0 <= boundary_fill_alpha <= 1:
        raise ValueError(
            "`boundary_alpha` and `boundary_fill_alpha` must be in [0, 1]."
        )

    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            f"adata.obsm['{spatial_key}'] must be a 2D array with at least two columns."
        )

    coords = coords[:, :2]
    n_obs = adata.n_obs
    rng = np.random.default_rng(random_state)

    selected_mask = np.ones(n_obs, dtype=bool)

    if cells is not None:
        cell_index = pd.Index(adata.obs_names)
        selected_cells = pd.Index(cells)
        loc = cell_index.get_indexer(selected_cells)

        if np.any(loc < 0):
            missing = selected_cells[loc < 0]
            raise KeyError(
                f"{len(missing)} cells were not found in adata.obs_names. "
                f"First missing cell: {missing[0]}"
            )

        selected_mask[:] = False
        selected_mask[loc] = True

    if (min_coordinate is None) ^ (max_coordinate is None):
        raise ValueError(
            "`min_coordinate` and `max_coordinate` must be provided together."
        )

    crop_bounds = None
    if min_coordinate is not None and max_coordinate is not None:
        if len(min_coordinate) != 2 or len(max_coordinate) != 2:
            raise ValueError(
                "`min_coordinate` and `max_coordinate` must each be [x, y]."
            )

        xmin, ymin = map(float, min_coordinate)
        xmax, ymax = map(float, max_coordinate)

        if xmin > xmax or ymin > ymax:
            raise ValueError(
                "`min_coordinate` must not exceed `max_coordinate` on either axis."
            )

        crop_bounds = (xmin, xmax, ymin, ymax)

        coordinate_mask = _get_view_mask(
            x=coords[:, 0],
            y=coords[:, 1],
            min_coordinate=(xmin, ymin),
            max_coordinate=(xmax, ymax),
        )
        selected_mask &= coordinate_mask

    selected_idx = np.flatnonzero(selected_mask)

    if selected_idx.size == 0:
        raise ValueError(
            "No cells remain after applying `cells` and/or coordinate filters."
        )

    if max_nodes is not None and selected_idx.size > max_nodes:
        selected_idx = rng.choice(selected_idx, size=max_nodes, replace=False)
        selected_idx = np.sort(selected_idx)

        selected_mask[:] = False
        selected_mask[selected_idx] = True

    A = adata.obsp[network_key]
    if not sparse.issparse(A):
        A = sparse.csr_matrix(A)
    else:
        A = A.tocsr()

    A_upper = sparse.triu(A, k=1).tocoo()

    edge_mask = selected_mask[A_upper.row] & selected_mask[A_upper.col]
    edge_src = A_upper.row[edge_mask]
    edge_dst = A_upper.col[edge_mask]

    if max_edges is not None and edge_src.size > max_edges:
        edge_keep = rng.choice(edge_src.size, size=max_edges, replace=False)
        edge_src = edge_src[edge_keep]
        edge_dst = edge_dst[edge_keep]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Edges are always drawn below nodes.
    if edge_src.size > 0:
        segments = np.stack(
            [coords[edge_src], coords[edge_dst]],
            axis=1,
        )

        edge_collection = LineCollection(
            segments,
            colors=edge_color,
            linewidths=edge_width,
            alpha=edge_alpha,
            rasterized=rasterized,
            zorder=1,
        )
        ax.add_collection(edge_collection)

    plot_coords = coords[selected_idx]
    fill_colors = ["lightgray"] * len(selected_idx)

    # Nodes are always drawn above edges.
    if color is None:
        ax.scatter(
            plot_coords[:, 0],
            plot_coords[:, 1],
            s=node_size,
            c="lightgray",
            alpha=node_alpha,
            linewidths=0,
            rasterized=rasterized,
            zorder=3,
        )

    else:
        if isinstance(color, str) and color in adata.obs.columns:
            values = adata.obs[color].iloc[selected_idx]
            color_label = color
        else:
            values = np.asarray(color)
            if values.shape[0] != n_obs:
                raise ValueError(
                    "`color` must be either an adata.obs column name or a sequence "
                    "with length adata.n_obs."
                )
            values = pd.Series(
                values[selected_idx], index=adata.obs_names[selected_idx]
            )
            color_label = None

        if pd.api.types.is_numeric_dtype(values):
            scatter = ax.scatter(
                plot_coords[:, 0],
                plot_coords[:, 1],
                s=node_size,
                c=np.asarray(values),
                cmap=cmap,
                alpha=node_alpha,
                linewidths=0,
                rasterized=rasterized,
                zorder=3,
            )
            cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            if color_label is not None:
                cbar.set_label(color_label)
            fill_colors = scatter.to_rgba(np.asarray(values))

        else:
            if isinstance(color, str) and color in adata.obs.columns:
                full_values = adata.obs[color]

                categories = _get_observation_categories(full_values)

                palette_key = f"{color}_colors"
                if palette_key not in adata.uns:
                    raise KeyError(
                        f"Colors for '{color}' were not found in "
                        f"adata.uns['{palette_key}']."
                    )

                stored_colors = list(adata.uns[palette_key])
                if len(stored_colors) < len(categories):
                    raise ValueError(
                        f"adata.uns['{palette_key}'] has {len(stored_colors)} colors, "
                        f"but '{color}' has {len(categories)} categories."
                    )

                palette = {cat: stored_colors[i] for i, cat in enumerate(categories)}

                selected_values = full_values.iloc[selected_idx]
                mapped_colors = [
                    palette[val] if pd.notna(val) else "lightgray"
                    for val in selected_values
                ]
            else:
                values = pd.Categorical(values)
                categories = list(values.categories)

                base_cmap = plt.get_cmap("tab20")
                palette = {
                    cat: base_cmap(i % base_cmap.N) for i, cat in enumerate(categories)
                }
                mapped_colors = [
                    palette[val] if pd.notna(val) else "lightgray" for val in values
                ]

            fill_colors = mapped_colors

            ax.scatter(
                plot_coords[:, 0],
                plot_coords[:, 1],
                s=node_size,
                c=mapped_colors,
                alpha=node_alpha,
                linewidths=0,
                rasterized=rasterized,
                zorder=3,
            )

            if legend and len(categories) <= max_legend_categories:
                handles = [
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        linestyle="",
                        markersize=5,
                        markerfacecolor=palette[cat],
                        markeredgecolor="none",
                        label=str(cat),
                    )
                    for cat in categories
                ]
                ax.legend(
                    handles=handles,
                    title=color_label,
                    frameon=False,
                    bbox_to_anchor=(1.02, 1),
                    loc="upper left",
                    borderaxespad=0,
                )

    if cell_boundary is not None:
        selected_geometries = _selected_boundary_geometries(
            adata, selected_idx, cell_boundary, cell_id_col
        )
        color_by_index = dict(zip(selected_idx, fill_colors))
        patches = []
        patch_colors = []
        outline_segments = []
        for idx, geom in selected_geometries:
            outline_segments.extend(_cell_outline_segments(geom))
            for patch in _cell_polygon_patches(geom):
                patches.append(patch)
                patch_colors.append(
                    to_rgba(color_by_index[idx], alpha=boundary_fill_alpha)
                )
        if patches and boundary_fill_alpha > 0:
            ax.add_collection(
                PatchCollection(
                    patches,
                    facecolors=patch_colors,
                    edgecolors="none",
                    linewidths=0,
                    rasterized=rasterized,
                    zorder=0.5,
                )
            )
        if outline_segments:
            ax.add_collection(
                LineCollection(
                    outline_segments,
                    colors=boundary_color,
                    linewidths=boundary_width,
                    alpha=boundary_alpha,
                    rasterized=rasterized,
                    zorder=2,
                )
            )

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if crop_bounds is not None:
        xmin, xmax, ymin, ymax = crop_bounds
        ax.set_xlim(xmin, xmax)
        if invert_y:
            ax.set_ylim(ymax, ymin)
        else:
            ax.set_ylim(ymin, ymax)
    elif invert_y:
        ax.invert_yaxis()

    if title is None:
        n_edges_total = A_upper.nnz
        n_edges_drawn = edge_src.size
        title = (
            f"{network_key}: "
            f"{selected_idx.size:,} cells, "
            f"{n_edges_drawn:,}/{n_edges_total:,} edges shown"
        )

    ax.set_title(title)
    ax.margins(0.02)

    if show:
        plt.show()

    return ax


def spatial_polygons(
    adata,
    color: Union[str, Sequence[str]],
    cell_boundary_dict: Optional[Mapping[str, Any]] = None,
    spatial_key: str = "spatial",
    min_coordinate: Optional[Sequence[float]] = None,
    max_coordinate: Optional[Sequence[float]] = None,
    cell_type_colors: Optional[Mapping[str, str]] = None,
    cmap: Union[str, Any] = "Spectral_r",
    spot_size: float = 8,
    boundary_linewidth: float = 0.05,
    boundary_edgecolor: str = "white",
    alpha: float = 1.0,
    na_color: str = "#d9d9d9",
    background_color: str = "#d9d9d9",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vmax_percentile: Optional[float] = 99,
    figsize: Optional[Tuple[float, float]] = None,
    ncols: int = 1,
    title: Optional[Union[str, Sequence[str]]] = None,
    legend: bool = True,
    legend_loc: str = "right_upper",
    colorbar: bool = True,
    invert_y: bool = True,
    show_axis: bool = False,
    focus: Optional[Union[str, Sequence[str]]] = None,
    focus_key: Optional[str] = None,
    overlay_key: Optional[str] = None,
    overlay: Optional[Mapping[str, str]] = None,
    overlay_alpha: float = 1.0,
    overlay_boundary_linewidth: Optional[float] = None,
    overlay_boundary_edgecolor: Optional[str] = None,
    overlay_spot_size: Optional[float] = None,
    overlay_legend: bool = True,
    overlay_legend_loc: str = "right_lower",
    save_path: Optional[str] = None,
    dpi: int = 300,
    legend_layout_rect: Optional[Tuple[float, float, float, float]] = None,
    show: bool = True,
    return_fig: bool = False,
):
    color_keys = _as_list(color)

    # Resolve stored colors before cropping for a single observation column.
    if cell_type_colors is None:
        if len(color_keys) == 1 and color_keys[0] in adata.obs.columns:
            palette_key = f"{color_keys[0]}_colors"
            if palette_key in adata.uns:
                categories = _get_observation_categories(adata.obs[color_keys[0]])
                stored_colors = list(adata.uns[palette_key])
                cell_type_colors = {
                    str(cat): stored_colors[i]
                    for i, cat in enumerate(categories)
                    if i < len(stored_colors)
                }

    if spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm 中没有 '{spatial_key}'。")

    coords = np.asarray(adata.obsm[spatial_key])
    if coords.shape[1] < 2:
        raise ValueError(f"adata.obsm['{spatial_key}'] 至少需要两列 x, y 坐标。")

    x = coords[:, 0]
    y = coords[:, 1]

    view_mask = _get_view_mask(
        x=x,
        y=y,
        min_coordinate=min_coordinate,
        max_coordinate=max_coordinate,
    )

    n_panels = len(color_keys)
    ncols = min(ncols, n_panels)
    nrows = math.ceil(n_panels / ncols)

    if figsize is None:
        figsize = (6 * ncols, 6 * nrows)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        squeeze=False,
    )

    axes_flat = axes.ravel()
    titles = _prepare_titles(title, color_keys)

    for i, color_key in enumerate(color_keys):
        ax = axes_flat[i]

        plot_info = _get_color_values(adata, color_key)

        _plot_single_panel(
            ax=ax,
            adata=adata,
            color_key=color_key,
            plot_info=plot_info,
            x=x,
            y=y,
            view_mask=view_mask,
            cell_boundary_dict=cell_boundary_dict,
            cell_type_colors=cell_type_colors,
            cmap=cmap,
            spot_size=spot_size,
            boundary_linewidth=boundary_linewidth,
            boundary_edgecolor=boundary_edgecolor,
            alpha=alpha,
            na_color=na_color,
            background_color=background_color,
            vmin=vmin,
            vmax=vmax,
            vmax_percentile=vmax_percentile,
            legend=legend,
            legend_loc=legend_loc,
            colorbar=colorbar,
            focus=focus,
            focus_key=focus_key,
            overlay_key=overlay_key,
            overlay=overlay,
            overlay_alpha=overlay_alpha,
            overlay_boundary_linewidth=overlay_boundary_linewidth,
            overlay_boundary_edgecolor=overlay_boundary_edgecolor,
            overlay_spot_size=overlay_spot_size,
            overlay_legend=overlay_legend,
            overlay_legend_loc=overlay_legend_loc,
        )

        if min_coordinate is not None and max_coordinate is not None:
            ax.set_xlim(min_coordinate[0], max_coordinate[0])
            ax.set_ylim(min_coordinate[1], max_coordinate[1])

        ax.set_aspect("equal")

        if invert_y:
            ax.invert_yaxis()

        if show_axis:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.tick_params(
                axis="both",
                which="both",
                labelsize=9,
                length=3,
                width=0.8,
            )
        else:
            ax.axis("off")

        ax.set_title(titles[i], fontsize=14)

    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].axis("off")

    if legend_layout_rect is not None:
        fig.tight_layout(rect=legend_layout_rect)
    else:
        fig.tight_layout()

    # Include all legends in the saved figure bounds.
    if save_path is not None:
        fig.canvas.draw()

        legends = []
        for ax in axes_flat:
            legends.extend(ax.findobj(match=Legend))

        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            bbox_extra_artists=legends if len(legends) > 0 else None,
        )

    if show:
        plt.show()

    if return_fig:
        return fig, axes


def spatial_bins(
    adata,
    color: str,
    spatial_key: str = "spatial",
    min_coordinate: Optional[Tuple[float, float]] = None,
    max_coordinate: Optional[Tuple[float, float]] = None,
    bin_size: float = 20.0,
    agg: str = "mean",
    min_cells: int = 1,
    cmap: Union[str, Any] = "Spectral_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    vmax_percentile: Optional[float] = 99,
    na_color: str = "#ffffff",
    figsize: Tuple[float, float] = (7, 7),
    title: Optional[str] = None,
    colorbar: bool = True,
    invert_y: bool = True,
    show_axis: bool = True,
    save_path: Optional[str] = None,
    dpi: int = 300,
    ax: Optional[Axes] = None,
    show: bool = True,
    return_data: bool = False,
    *,
    category: Optional[Any] = None,
    colorbar_label: Optional[str] = None,
):
    if spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{spatial_key}'] not found.")
    if not np.isfinite(bin_size) or bin_size <= 0:
        raise ValueError("bin_size must be finite and positive.")
    if (
        isinstance(min_cells, (bool, np.bool_))
        or not isinstance(min_cells, (int, np.integer))
        or min_cells <= 0
    ):
        raise ValueError("min_cells must be a positive integer.")
    if vmax_percentile is not None and (
        not np.isfinite(vmax_percentile) or not 0 <= vmax_percentile <= 100
    ):
        raise ValueError("vmax_percentile must be between 0 and 100, or None.")

    category_mode = category is not None
    fraction_mode = False
    if category_mode:
        if color not in adata.obs.columns:
            raise KeyError(f"Category mode requires an adata.obs column: {color!r}.")
        if agg in {"fraction", "proportion", "mean"}:
            aggregation = "mean"
            fraction_mode = True
        elif agg in {"count", "sum"}:
            aggregation = "sum"
        else:
            raise ValueError(
                "With category, agg must be fraction, proportion, mean, count, or sum."
            )
        annotations = adata.obs[color]
        if not annotations.isin([category]).any():
            raise ValueError(
                f"Category {category!r} was not found in adata.obs[{color!r}]."
            )
        values = annotations.isin([category]).to_numpy(dtype=float)
        default_label = f"{category} fraction" if fraction_mode else f"{category} count"
    else:
        if agg not in {"mean", "sum", "median", "max"}:
            raise ValueError("Without category, agg must be mean, sum, median, or max.")
        aggregation = agg
        values = _get_continuous_values_for_bin_plot(adata, color)
        default_label = f"{agg} {color}"

    coords = np.asarray(adata.obsm[spatial_key], dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"adata.obsm[{spatial_key!r}] must have at least two columns.")
    if len(values) != len(adata.obs):
        raise ValueError(
            "Coordinates, values, and adata.obs must have the same cell count."
        )
    x, y = coords[:, 0], coords[:, 1]
    finite_coords = np.isfinite(x) & np.isfinite(y)
    if not finite_coords.any():
        raise ValueError("No cells have finite spatial coordinates.")

    xmin, ymin = (
        (float(x[finite_coords].min()), float(y[finite_coords].min()))
        if min_coordinate is None
        else map(float, min_coordinate)
    )
    xmax, ymax = (
        (float(x[finite_coords].max()), float(y[finite_coords].max()))
        if max_coordinate is None
        else map(float, max_coordinate)
    )
    if not np.isfinite([xmin, ymin, xmax, ymax]).all() or not (
        xmin < xmax and ymin < ymax
    ):
        raise ValueError("Require finite limits with xmin < xmax and ymin < ymax.")
    view_mask = (
        finite_coords
        & _get_view_mask(x, y, (xmin, ymin), (xmax, ymax))
        & np.isfinite(values)
    )
    if not view_mask.any():
        raise ValueError("No valid cells found in the selected region.")

    n_x = max(1, int(np.ceil((xmax - xmin) / bin_size)))
    n_y = max(1, int(np.ceil((ymax - ymin) / bin_size)))
    x_edges = xmin + np.arange(n_x + 1) * bin_size
    y_edges = ymin + np.arange(n_y + 1) * bin_size
    # Right-open bins internally; include the outer maximum in the last bin.
    x_bin = np.clip(
        np.searchsorted(x_edges, x[view_mask], side="right") - 1, 0, n_x - 1
    )
    y_bin = np.clip(
        np.searchsorted(y_edges, y[view_mask], side="right") - 1, 0, n_y - 1
    )
    bin_df = pd.DataFrame({"x_bin": x_bin, "y_bin": y_bin, "value": values[view_mask]})
    grouped = bin_df.groupby(["y_bin", "x_bin"], sort=False)["value"]
    agg_values = grouped.agg(aggregation)
    count_values = grouped.size().reindex(agg_values.index)
    grid_y = agg_values.index.get_level_values("y_bin").to_numpy(dtype=int)
    grid_x = agg_values.index.get_level_values("x_bin").to_numpy(dtype=int)
    value_grid = np.full((n_y, n_x), np.nan, dtype=float)
    count_grid = np.zeros((n_y, n_x), dtype=int)
    value_grid[grid_y, grid_x] = agg_values.to_numpy(dtype=float)
    count_grid[grid_y, grid_x] = count_values.to_numpy(dtype=int)
    value_grid[count_grid < min_cells] = np.nan

    finite_values = value_grid[np.isfinite(value_grid)]
    if not finite_values.size:
        raise ValueError(
            "No bins passed min_cells. Increase bin_size or reduce min_cells."
        )
    current_vmin = (
        float(vmin)
        if vmin is not None
        else (0.0 if fraction_mode else float(finite_values.min()))
    )
    if vmax is not None:
        current_vmax = float(vmax)
    elif fraction_mode:
        current_vmax = 1.0
    elif vmax_percentile is None:
        current_vmax = float(finite_values.max())
    else:
        current_vmax = float(np.percentile(finite_values, vmax_percentile))
    if (
        not np.isfinite([current_vmin, current_vmax]).all()
        or current_vmax < current_vmin
    ):
        raise ValueError("Color limits must be finite and vmax must be >= vmin.")
    if current_vmin == current_vmax:
        current_vmax = current_vmin + max(1e-9, abs(current_vmin) * 1e-9)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(na_color)
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        value_grid,
        cmap=cmap_obj,
        vmin=current_vmin,
        vmax=current_vmax,
        shading="flat",
    )
    if colorbar:
        cbar = ax.figure.colorbar(mesh, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(
            default_label if colorbar_label is None else colorbar_label, fontsize=12
        )
        if fraction_mode:
            cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.set_aspect("equal")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim((ymax, ymin) if invert_y else (ymin, ymax))
    if show_axis:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        ax.axis("off")
    if title is None:
        title = (
            f"{default_label} (bin={bin_size:g})"
            if category_mode
            else f"{color} ({agg}, bin={bin_size:g})"
        )
    ax.set_title(title, fontsize=14)
    if save_path is not None:
        ax.figure.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        ax.figure.tight_layout()
        plt.show()
    return (ax, bin_df, value_grid, count_grid) if return_data else ax


def spatial_transcripts(
    sdata,
    genes,
    celltypes=None,
    celltype_key="celltype",
    table_name="table",
    transcript_scope="all",
    transcript_key="transcripts",
    shape_key="cell_boundaries",
    min_coordinate=None,
    max_coordinate=None,
    coordinate_input="anndata",
    coordinate_reference="shapes",
    show_anndata_coordinates=True,
    show_boundaries=True,
    show_transcripts=True,
    celltype_palette=None,
    cell_fill_alpha=0.18,
    boundary_width=0.5,
    boundary_alpha=1.0,
    transcript_palette=None,
    transcript_size=3,
    transcript_alpha=1.0,
    coordinate_system="global",
    point_method="datashader",
    auto_ticks=True,
    x_tick_interval=None,
    y_tick_interval=None,
    target_tick_intervals=6,
    tick_labelsize=10,
    show_legends=True,
    cell_legend_title="Cell type",
    transcript_legend_title="Transcript",
    cell_legend_bbox=(1.02, 1.0),
    transcript_legend_bbox=None,
    legend_gap=0.06,
    legend_fontsize=9,
    legend_title_fontsize=10,
    transcript_legend_size=6,
    cell_legend_ncol=1,
    transcript_legend_ncol=1,
    figsize=(8, 8),
    save_path=None,
    save_dpi=300,
    return_ax=True,
):
    genes = list(dict.fromkeys((str(gene) for gene in genes)))
    if len(genes) == 0:
        raise ValueError("`genes` cannot be empty.")
    if not (show_boundaries or show_transcripts):
        raise ValueError(
            "At least one of `show_boundaries` or `show_transcripts` must be True."
        )
    if transcript_scope not in {"all", "inside_cells"}:
        raise ValueError("`transcript_scope` must be 'all' or 'inside_cells'.")
    if coordinate_input not in {"anndata", "global"}:
        raise ValueError("`coordinate_input` must be 'anndata' or 'global'.")
    if coordinate_reference not in {"shapes", "transcripts"}:
        raise ValueError("`coordinate_reference` must be 'shapes' or 'transcripts'.")
    for name, value in {
        "cell_fill_alpha": cell_fill_alpha,
        "boundary_alpha": boundary_alpha,
        "transcript_alpha": transcript_alpha,
    }.items():
        if not 0 <= value <= 1:
            raise ValueError(f"`{name}` must be between 0 and 1.")
    if show_transcripts and transcript_key not in sdata.points:
        raise KeyError(f"'{transcript_key}' not found in sdata.points.")
    if show_boundaries and shape_key not in sdata.shapes:
        raise KeyError(f"'{shape_key}' not found in sdata.shapes.")
    if (
        min_coordinate is None
        and max_coordinate is not None
        or (min_coordinate is not None and max_coordinate is None)
    ):
        raise ValueError(
            "`min_coordinate` and `max_coordinate` must either both be provided or both be None."
        )
    if x_tick_interval is not None and x_tick_interval <= 0:
        raise ValueError("`x_tick_interval` must be > 0.")
    if y_tick_interval is not None and y_tick_interval <= 0:
        raise ValueError("`y_tick_interval` must be > 0.")
    if target_tick_intervals < 1:
        raise ValueError("`target_tick_intervals` must be >= 1.")

    if coordinate_reference == "shapes":
        if shape_key not in sdata.shapes:
            raise KeyError(f"'{shape_key}' not found in sdata.shapes.")
        reference_element = sdata.shapes[shape_key]
    else:
        if transcript_key not in sdata.points:
            raise KeyError(f"'{transcript_key}' not found in sdata.points.")
        reference_element = sdata.points[transcript_key]
    affine_matrix = None
    if coordinate_input == "anndata":
        affine_matrix = _get_xy_affine_matrix(
            reference_element, target_coordinate_system=coordinate_system
        )

    crop_min = None
    crop_max = None
    if min_coordinate is not None:
        if len(min_coordinate) != 2 or len(max_coordinate) != 2:
            raise ValueError(
                "`min_coordinate` and `max_coordinate` must have the form [x, y]."
            )
        xmin, ymin = map(float, min_coordinate)
        xmax, ymax = map(float, max_coordinate)
        if xmin >= xmax or ymin >= ymax:
            raise ValueError("`min_coordinate` must be smaller than `max_coordinate`.")
        if coordinate_input == "anndata":
            crop_min, crop_max = _transform_bbox(
                min_coordinate, max_coordinate, affine_matrix
            )
        else:
            crop_min = [xmin, ymin]
            crop_max = [xmax, ymax]
        plot_sdata = bounding_box_query(
            sdata,
            axes=("x", "y"),
            min_coordinate=crop_min,
            max_coordinate=crop_max,
            target_coordinate_system=coordinate_system,
            filter_table=False,
        )
    else:
        plot_sdata = sdata

    if table_name not in plot_sdata.tables:
        raise KeyError(f"'{table_name}' not found in sdata.tables.")
    table = plot_sdata.tables[table_name]
    obs = table.obs
    if celltype_key not in obs.columns:
        raise KeyError(
            f"'{celltype_key}' not found in sdata.tables['{table_name}'].obs."
        )

    available_celltypes = set(obs[celltype_key].dropna().astype(str).unique())
    if celltypes is None:
        series = obs[celltype_key]
        if hasattr(series.dtype, "categories"):
            allowed_celltypes = [
                str(celltype)
                for celltype in series.cat.categories
                if str(celltype) in available_celltypes
            ]
        else:
            allowed_celltypes = series.dropna().astype(str).drop_duplicates().tolist()
    else:
        allowed_celltypes = list(
            dict.fromkeys((str(celltype) for celltype in celltypes))
        )
        missing_celltypes = [
            celltype
            for celltype in allowed_celltypes
            if celltype not in available_celltypes
        ]
        if missing_celltypes:
            raise ValueError(
                f"Requested cell types were not found: {missing_celltypes}"
            )
    if len(allowed_celltypes) == 0:
        raise ValueError(f"No valid cell types found in '{celltype_key}'.")

    celltype_mask = obs[celltype_key].astype(str).isin(allowed_celltypes)
    selected_cells = set(obs.index[celltype_mask].astype(str))
    if len(selected_cells) == 0:
        raise ValueError("No cells matched the selected cell types.")

    resolved_celltype_palette = _get_celltype_palette(
        table=table,
        celltype_key=celltype_key,
        celltypes=allowed_celltypes,
        palette=celltype_palette,
    )
    resolved_transcript_palette = _get_transcript_palette(
        genes=genes, palette=transcript_palette
    )

    tmp_points_name = "_sstt_selected_transcripts"
    tmp_shape_name = "_sstt_selected_boundaries"
    added_tmp_points = False
    added_tmp_shapes = False
    display_celltypes = []
    display_genes = []

    # Remove temporary elements even if rendering or saving fails.
    try:
        if show_transcripts:
            original_points = plot_sdata.points[transcript_key]
            # Cell-type filtering applies only to assigned-transcript mode.
            if transcript_scope == "all":
                points_element = transcript_key
                display_genes = _get_present_values(
                    original_points["feature_name"], genes
                )
            else:
                tx = original_points[original_points["feature_name"].isin(genes)]
                tx = tx[tx["cell_id"].astype(str).isin(list(selected_cells))]
                display_genes = _get_present_values(tx["feature_name"], genes)
                selected_points = PointsModel.parse(
                    tx, coordinates={"x": "x", "y": "y"}
                )
                transformations = get_transformation(original_points, get_all=True)
                set_transformation(selected_points, transformations, set_all=True)
                if tmp_points_name in plot_sdata.points:
                    raise KeyError(
                        f"Temporary point element '{tmp_points_name}' already exists."
                    )
                plot_sdata.points[tmp_points_name] = selected_points
                added_tmp_points = True
                points_element = tmp_points_name

        if show_boundaries:
            original_shapes = plot_sdata.shapes[shape_key]
            shape_ids = original_shapes.index.astype(str)
            keep = shape_ids.isin(selected_cells)
            selected_shapes = original_shapes.loc[keep].copy()
            if selected_shapes.empty:
                raise ValueError(
                    "No selected cell boundaries were found inside the requested ROI."
                )
            celltype_lookup = obs[celltype_key].astype(str)
            celltype_lookup.index = celltype_lookup.index.astype(str)
            shape_color_key = "_sstt_celltype"
            selected_shapes[shape_color_key] = selected_shapes.index.astype(str).map(
                celltype_lookup
            )
            present_set = set(selected_shapes[shape_color_key].dropna().astype(str))
            display_celltypes = [
                celltype for celltype in allowed_celltypes if celltype in present_set
            ]
            if len(display_celltypes) == 0:
                raise ValueError(
                    "No requested cell types are present inside the selected ROI."
                )
            selected_shapes[shape_color_key] = (
                selected_shapes[shape_color_key]
                .astype("category")
                .cat.set_categories(display_celltypes)
            )
            transformations = get_transformation(original_shapes, get_all=True)
            set_transformation(selected_shapes, transformations, set_all=True)
            if tmp_shape_name in plot_sdata.shapes:
                raise KeyError(
                    f"Temporary shape element '{tmp_shape_name}' already exists."
                )
            plot_sdata.shapes[tmp_shape_name] = selected_shapes
            added_tmp_shapes = True
            shape_element = tmp_shape_name

        plot_obj = plot_sdata
        if show_boundaries:
            plot_obj = plot_obj.pl.render_shapes(
                shape_element,
                color=shape_color_key,
                groups=display_celltypes,
                palette=resolved_celltype_palette,
                fill_alpha=cell_fill_alpha,
                outline_color=shape_color_key,
                outline_width=boundary_width,
                outline_alpha=boundary_alpha,
            )
        if show_transcripts and len(display_genes) > 0:
            plot_obj = plot_obj.pl.render_points(
                points_element,
                color="feature_name",
                groups=display_genes,
                palette=resolved_transcript_palette,
                size=transcript_size,
                alpha=transcript_alpha,
                method=point_method,
            )

        ax = plot_obj.pl.show(
            coordinate_systems=coordinate_system,
            figsize=figsize,
            legend_loc=None,
            na_in_legend=False,
            return_ax=True,
        )
        if isinstance(ax, (list, tuple)):
            if len(ax) == 0:
                raise RuntimeError("SpatialData returned an empty Axes list.")
            ax = ax[0]
        if ax is None:
            raise RuntimeError("SpatialData plotting did not return an Axes object.")

        ax.set_aspect("equal")
        if auto_ticks:
            _set_spatial_ticks(
                ax=ax,
                affine_matrix=affine_matrix,
                coordinate_input=coordinate_input,
                min_coordinate=min_coordinate,
                max_coordinate=max_coordinate,
                x_tick_interval=x_tick_interval,
                y_tick_interval=y_tick_interval,
                target_tick_intervals=target_tick_intervals,
            )
        elif coordinate_input == "anndata" and show_anndata_coordinates:
            warnings.warn(
                "`auto_ticks=False`; AnnData-style tick conversion has therefore not been applied.",
                RuntimeWarning,
                stacklevel=2,
            )
        ax.tick_params(axis="both", labelsize=tick_labelsize)

        if show_legends:
            _add_separate_legends(
                ax=ax,
                celltypes=display_celltypes,
                celltype_palette=resolved_celltype_palette,
                genes=display_genes,
                transcript_palette=resolved_transcript_palette,
                cell_fill_alpha=cell_fill_alpha,
                show_boundaries=show_boundaries,
                show_transcripts=show_transcripts,
                cell_legend_title=cell_legend_title,
                transcript_legend_title=transcript_legend_title,
                cell_legend_bbox=cell_legend_bbox,
                transcript_legend_bbox=transcript_legend_bbox,
                legend_gap=legend_gap,
                legend_fontsize=legend_fontsize,
                legend_title_fontsize=legend_title_fontsize,
                transcript_legend_size=transcript_legend_size,
                cell_legend_ncol=cell_legend_ncol,
                transcript_legend_ncol=transcript_legend_ncol,
            )

        if save_path is not None:
            save_path = FilePath(save_path)
            ax.figure.savefig(save_path, dpi=save_dpi, bbox_inches="tight")
        if return_ax:
            return ax
        return None
    finally:
        if added_tmp_points and tmp_points_name in plot_sdata.points:
            del plot_sdata.points[tmp_points_name]
        if added_tmp_shapes and tmp_shape_name in plot_sdata.shapes:
            del plot_sdata.shapes[tmp_shape_name]


def _get_view_mask(x, y, min_coordinate=None, max_coordinate=None):
    mask = np.ones(len(x), dtype=bool)

    if min_coordinate is not None:
        xmin, ymin = min_coordinate
        mask &= x >= xmin
        mask &= y >= ymin

    if max_coordinate is not None:
        xmax, ymax = max_coordinate
        mask &= x <= xmax
        mask &= y <= ymax

    return mask


def _get_observation_categories(values):
    if isinstance(values.dtype, pd.CategoricalDtype):
        return list(values.cat.categories)
    return list(pd.unique(values.dropna()))


def _plot_single_panel(
    ax,
    adata,
    color_key,
    plot_info,
    x,
    y,
    view_mask,
    cell_boundary_dict=None,
    cell_type_colors=None,
    cmap="Spectral_r",
    spot_size=8,
    boundary_linewidth=0.05,
    boundary_edgecolor="white",
    alpha=1.0,
    na_color="#d9d9d9",
    background_color="#d9d9d9",
    vmin=None,
    vmax=None,
    vmax_percentile=99,
    legend=True,
    legend_loc="right_upper",
    colorbar=True,
    focus=None,
    focus_key=None,
    overlay_key=None,
    overlay=None,
    overlay_alpha=1.0,
    overlay_boundary_linewidth=None,
    overlay_boundary_edgecolor=None,
    overlay_spot_size=None,
    overlay_legend=True,
    overlay_legend_loc="right_lower",
):
    plot_type = plot_info["plot_type"]
    source_type = plot_info["source_type"]
    values = plot_info["values"]

    focus_mask = _get_focus_mask(
        adata=adata,
        values=values,
        plot_type=plot_type,
        focus=focus,
        focus_key=focus_key,
    )

    if cell_boundary_dict is not None:
        _draw_cell_boundaries(
            ax=ax,
            adata=adata,
            color_key=color_key,
            plot_type=plot_type,
            source_type=source_type,
            values=values,
            view_mask=view_mask,
            cell_boundary_dict=cell_boundary_dict,
            cell_type_colors=cell_type_colors,
            cmap=cmap,
            boundary_linewidth=boundary_linewidth,
            boundary_edgecolor=boundary_edgecolor,
            alpha=alpha,
            na_color=na_color,
            background_color=background_color,
            vmin=vmin,
            vmax=vmax,
            vmax_percentile=vmax_percentile,
            legend=legend,
            legend_loc=legend_loc,
            colorbar=colorbar,
            focus_mask=focus_mask,
            overlay_key=overlay_key,
            overlay=overlay,
            overlay_alpha=overlay_alpha,
            overlay_boundary_linewidth=overlay_boundary_linewidth,
            overlay_boundary_edgecolor=overlay_boundary_edgecolor,
            overlay_legend=overlay_legend,
            overlay_legend_loc=overlay_legend_loc,
        )
    else:
        _draw_spots(
            ax=ax,
            adata=adata,
            color_key=color_key,
            plot_type=plot_type,
            source_type=source_type,
            values=values,
            x=x,
            y=y,
            view_mask=view_mask,
            cell_type_colors=cell_type_colors,
            cmap=cmap,
            spot_size=spot_size,
            alpha=alpha,
            na_color=na_color,
            background_color=background_color,
            vmin=vmin,
            vmax=vmax,
            vmax_percentile=vmax_percentile,
            legend=legend,
            legend_loc=legend_loc,
            colorbar=colorbar,
            focus_mask=focus_mask,
            overlay_key=overlay_key,
            overlay=overlay,
            overlay_alpha=overlay_alpha,
            overlay_spot_size=overlay_spot_size,
            overlay_legend=overlay_legend,
            overlay_legend_loc=overlay_legend_loc,
        )


def _draw_cell_boundaries(
    ax,
    adata,
    color_key,
    plot_type,
    source_type,
    values,
    view_mask,
    cell_boundary_dict,
    cell_type_colors=None,
    cmap="Spectral_r",
    boundary_linewidth=0.05,
    boundary_edgecolor="white",
    alpha=1.0,
    na_color="#d9d9d9",
    background_color="#d9d9d9",
    vmin=None,
    vmax=None,
    vmax_percentile=99,
    legend=True,
    legend_loc="right_upper",
    colorbar=True,
    focus_mask=None,
    overlay_key=None,
    overlay=None,
    overlay_alpha=1.0,
    overlay_boundary_linewidth=None,
    overlay_boundary_edgecolor=None,
    overlay_legend=True,
    overlay_legend_loc="right_lower",
):
    cell_ids = adata.obs_names.astype(str).to_numpy()
    selected_cell_ids = cell_ids[view_mask]

    patches = []
    patch_values = []
    patch_focus = []

    for cell_id in selected_cell_ids:
        if cell_id not in cell_boundary_dict:
            continue

        geom = cell_boundary_dict[cell_id]
        polygons = _normalize_geometry_to_polygons(geom)

        if len(polygons) == 0:
            continue

        value = values.loc[cell_id]

        if focus_mask is None:
            is_focus = True
        else:
            is_focus = bool(focus_mask.loc[cell_id])

        for poly in polygons:
            coords = np.asarray(poly.exterior.coords)
            patches.append(MplPolygon(coords, closed=True))
            patch_values.append(value)
            patch_focus.append(is_focus)

    if len(patches) == 0:
        raise ValueError(
            "没有找到可绘制的细胞边界。请检查 cell_boundary_dict 的 key 是否与 adata.obs_names 一致。"
        )

    patch_focus = np.asarray(patch_focus, dtype=bool)

    if plot_type == "continuous":
        numeric_values = pd.to_numeric(
            pd.Series(patch_values), errors="coerce"
        ).to_numpy()

        limit_values = (
            numeric_values[patch_focus] if focus_mask is not None else numeric_values
        )

        current_vmin, current_vmax = _get_continuous_limits(
            limit_values,
            vmin=vmin,
            vmax=vmax,
            vmax_percentile=vmax_percentile,
        )

        pc_for_colorbar = None

        if focus_mask is None:
            pc = PatchCollection(
                patches,
                cmap=cmap,
                edgecolor=boundary_edgecolor,
                linewidth=boundary_linewidth,
                alpha=alpha,
                zorder=1,
            )
            pc.set_array(numeric_values)
            pc.set_clim(current_vmin, current_vmax)
            ax.add_collection(pc)
            pc_for_colorbar = pc

        else:
            background_patches = [
                p for p, keep in zip(patches, patch_focus) if not keep
            ]

            if len(background_patches) > 0:
                bg_pc = PatchCollection(
                    background_patches,
                    facecolor=background_color,
                    edgecolor=boundary_edgecolor,
                    linewidth=boundary_linewidth,
                    alpha=alpha,
                    zorder=1,
                )
                ax.add_collection(bg_pc)

            focus_patches = [p for p, keep in zip(patches, patch_focus) if keep]
            focus_values = numeric_values[patch_focus]

            if len(focus_patches) > 0:
                pc = PatchCollection(
                    focus_patches,
                    cmap=cmap,
                    edgecolor=boundary_edgecolor,
                    linewidth=boundary_linewidth,
                    alpha=alpha,
                    zorder=2,
                )
                pc.set_array(focus_values)
                pc.set_clim(current_vmin, current_vmax)
                ax.add_collection(pc)
                pc_for_colorbar = pc

        ax.autoscale()

        if colorbar and pc_for_colorbar is not None:
            cbar = plt.colorbar(pc_for_colorbar, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label(color_key, fontsize=12)

    else:
        patch_values_series = pd.Series(patch_values).astype("category")

        color_map = _get_category_color_map(
            adata=adata,
            color_key=color_key,
            categories=patch_values_series.cat.categories,
            user_colors=cell_type_colors,
            na_color=na_color,
        )

        facecolors = []
        for v, keep in zip(patch_values, patch_focus):
            if pd.isna(v):
                facecolors.append(na_color)
            else:
                v_str = str(v)

                if focus_mask is not None:
                    if keep:
                        facecolors.append(color_map.get(v_str, na_color))
                    else:
                        facecolors.append(background_color)
                else:
                    facecolors.append(color_map.get(v_str, na_color))

        pc = PatchCollection(
            patches,
            facecolor=facecolors,
            edgecolor=boundary_edgecolor,
            linewidth=boundary_linewidth,
            alpha=alpha,
            zorder=1,
        )

        ax.add_collection(pc)
        ax.autoscale()

        if legend:
            if focus_mask is not None:
                focused_values = [
                    str(v)
                    for v, keep in zip(patch_values, patch_focus)
                    if keep and not pd.isna(v)
                ]
                focused_categories = set(focused_values)
                legend_color_map = {
                    k: v for k, v in color_map.items() if k in focused_categories
                }
            else:
                legend_color_map = color_map

            _add_category_legend(
                ax=ax,
                color_map=legend_color_map,
                legend_loc=legend_loc,
                title=color_key,
            )

    if overlay is not None:
        _add_boundary_overlay(
            ax=ax,
            adata=adata,
            view_mask=view_mask,
            cell_boundary_dict=cell_boundary_dict,
            overlay_key=overlay_key,
            overlay=overlay,
            alpha=overlay_alpha,
            edgecolor=(
                overlay_boundary_edgecolor
                if overlay_boundary_edgecolor is not None
                else boundary_edgecolor
            ),
            linewidth=(
                overlay_boundary_linewidth
                if overlay_boundary_linewidth is not None
                else boundary_linewidth
            ),
            legend=overlay_legend,
            legend_loc=overlay_legend_loc,
        )


def _draw_spots(
    ax,
    adata,
    color_key,
    plot_type,
    source_type,
    values,
    x,
    y,
    view_mask,
    cell_type_colors=None,
    cmap="Spectral_r",
    spot_size=8,
    alpha=1.0,
    na_color="#d9d9d9",
    background_color="#d9d9d9",
    vmin=None,
    vmax=None,
    vmax_percentile=99,
    legend=True,
    legend_loc="right_upper",
    colorbar=True,
    focus_mask=None,
    overlay_key=None,
    overlay=None,
    overlay_alpha=1.0,
    overlay_spot_size=None,
    overlay_legend=True,
    overlay_legend_loc="right_lower",
):
    x_plot = x[view_mask]
    y_plot = y[view_mask]
    values_plot = values.iloc[view_mask]

    if focus_mask is not None:
        focus_mask_plot = focus_mask.iloc[view_mask].to_numpy(dtype=bool)
    else:
        focus_mask_plot = None

    if plot_type == "continuous":
        numeric_values = pd.to_numeric(values_plot, errors="coerce").to_numpy()

        limit_values = (
            numeric_values[focus_mask_plot]
            if focus_mask_plot is not None
            else numeric_values
        )

        current_vmin, current_vmax = _get_continuous_limits(
            limit_values,
            vmin=vmin,
            vmax=vmax,
            vmax_percentile=vmax_percentile,
        )

        if focus_mask_plot is None:
            sc = ax.scatter(
                x_plot,
                y_plot,
                c=numeric_values,
                s=spot_size,
                cmap=cmap,
                vmin=current_vmin,
                vmax=current_vmax,
                alpha=alpha,
                linewidths=0,
                zorder=1,
            )
        else:
            ax.scatter(
                x_plot[~focus_mask_plot],
                y_plot[~focus_mask_plot],
                c=background_color,
                s=spot_size,
                alpha=alpha,
                linewidths=0,
                zorder=1,
            )

            sc = ax.scatter(
                x_plot[focus_mask_plot],
                y_plot[focus_mask_plot],
                c=numeric_values[focus_mask_plot],
                s=spot_size,
                cmap=cmap,
                vmin=current_vmin,
                vmax=current_vmax,
                alpha=alpha,
                linewidths=0,
                zorder=2,
            )

        if colorbar:
            cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label(color_key, fontsize=12)

    else:
        categories = values_plot.astype("category")

        color_map = _get_category_color_map(
            adata=adata,
            color_key=color_key,
            categories=categories.cat.categories,
            user_colors=cell_type_colors,
            na_color=na_color,
        )

        point_colors = []
        for idx, v in enumerate(values_plot):
            if pd.isna(v):
                point_colors.append(na_color)
            else:
                v_str = str(v)

                if focus_mask_plot is not None:
                    if focus_mask_plot[idx]:
                        point_colors.append(color_map.get(v_str, na_color))
                    else:
                        point_colors.append(background_color)
                else:
                    point_colors.append(color_map.get(v_str, na_color))

        ax.scatter(
            x_plot,
            y_plot,
            c=point_colors,
            s=spot_size,
            alpha=alpha,
            linewidths=0,
            zorder=1,
        )

        if legend:
            if focus_mask_plot is not None:
                focused_values = values_plot.iloc[focus_mask_plot]
                focused_categories = set(map(str, focused_values.dropna().unique()))
                legend_color_map = {
                    k: v for k, v in color_map.items() if k in focused_categories
                }
            else:
                legend_color_map = color_map

            _add_category_legend(
                ax=ax,
                color_map=legend_color_map,
                legend_loc=legend_loc,
                title=color_key,
            )

    if overlay is not None:
        _add_spot_overlay(
            ax=ax,
            adata=adata,
            x=x,
            y=y,
            view_mask=view_mask,
            overlay_key=overlay_key,
            overlay=overlay,
            spot_size=overlay_spot_size if overlay_spot_size is not None else spot_size,
            alpha=overlay_alpha,
            legend=overlay_legend,
            legend_loc=overlay_legend_loc,
        )


def _add_boundary_overlay(
    ax,
    adata,
    view_mask,
    cell_boundary_dict,
    overlay_key,
    overlay,
    alpha=1.0,
    edgecolor="white",
    linewidth=0.05,
    legend=True,
    legend_loc="right_lower",
):
    if overlay_key is None:
        raise ValueError(
            "使用 overlay 时必须提供 overlay_key，例如 overlay_key='cell_type'。"
        )

    if overlay_key not in adata.obs.columns:
        raise KeyError(f"overlay_key='{overlay_key}' 不在 adata.obs.columns 中。")

    overlay_values = adata.obs[overlay_key].copy()
    overlay_values.index = overlay_values.index.astype(str)
    overlay_values = overlay_values.astype(str)

    overlay = {str(k): v for k, v in overlay.items()}

    cell_ids = adata.obs_names.astype(str).to_numpy()
    selected_cell_ids = cell_ids[view_mask]

    overlay_patches = []
    overlay_colors = []
    used_labels = set()

    for cell_id in selected_cell_ids:
        if cell_id not in cell_boundary_dict:
            continue

        label = overlay_values.loc[cell_id]

        if label not in overlay:
            continue

        geom = cell_boundary_dict[cell_id]
        polygons = _normalize_geometry_to_polygons(geom)

        for poly in polygons:
            coords = np.asarray(poly.exterior.coords)
            overlay_patches.append(MplPolygon(coords, closed=True))
            overlay_colors.append(overlay[label])
            used_labels.add(label)

    if len(overlay_patches) == 0:
        return

    pc = PatchCollection(
        overlay_patches,
        facecolor=overlay_colors,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        zorder=10,
    )

    ax.add_collection(pc)
    ax.autoscale()

    if legend and len(used_labels) > 0:
        overlay_color_map = {
            label: overlay[label] for label in overlay if label in used_labels
        }

        _add_category_legend(
            ax=ax,
            color_map=overlay_color_map,
            legend_loc=legend_loc,
            title=overlay_key,
        )


def _add_spot_overlay(
    ax,
    adata,
    x,
    y,
    view_mask,
    overlay_key,
    overlay,
    spot_size=8,
    alpha=1.0,
    legend=True,
    legend_loc="right_lower",
):
    if overlay_key is None:
        raise ValueError(
            "使用 overlay 时必须提供 overlay_key，例如 overlay_key='cell_type'。"
        )

    if overlay_key not in adata.obs.columns:
        raise KeyError(f"overlay_key='{overlay_key}' 不在 adata.obs.columns 中。")

    overlay_values = adata.obs[overlay_key].copy()
    overlay_values.index = overlay_values.index.astype(str)
    overlay_values = overlay_values.astype(str)

    overlay = {str(k): v for k, v in overlay.items()}

    cell_ids = adata.obs_names.astype(str).to_numpy()
    selected_ids = cell_ids[view_mask]

    selected_values = overlay_values.loc[selected_ids]
    mask = selected_values.isin(set(overlay.keys())).to_numpy()

    if mask.sum() == 0:
        return

    x_plot = x[view_mask]
    y_plot = y[view_mask]

    point_colors = [overlay[str(v)] for v in selected_values[mask]]

    ax.scatter(
        x_plot[mask],
        y_plot[mask],
        c=point_colors,
        s=spot_size,
        alpha=alpha,
        linewidths=0,
        zorder=10,
    )

    if legend:
        used_labels = set(map(str, selected_values[mask].unique()))
        overlay_color_map = {
            label: overlay[label] for label in overlay if label in used_labels
        }

        _add_category_legend(
            ax=ax,
            color_map=overlay_color_map,
            legend_loc=legend_loc,
            title=overlay_key,
        )


def _get_color_values(adata, color_key):
    if color_key in adata.var_names:
        values = _get_gene_values(adata, color_key)

        return {
            "plot_type": "continuous",
            "source_type": "gene",
            "values": pd.Series(values, index=adata.obs_names.astype(str)),
        }

    if color_key in adata.obs.columns:
        values = adata.obs[color_key].copy()
        values.index = values.index.astype(str)

        if pd.api.types.is_numeric_dtype(values):
            return {
                "plot_type": "continuous",
                "source_type": "obs",
                "values": values,
            }
        else:
            return {
                "plot_type": "categorical",
                "source_type": "obs",
                "values": values.astype("category"),
            }

    raise KeyError(
        f"'{color_key}' 既不在 adata.var_names 中，也不在 adata.obs.columns 中。"
    )


def _get_focus_mask(
    adata,
    values,
    plot_type,
    focus=None,
    focus_key=None,
):
    if focus is None:
        return None

    focus_set = _as_focus_set(focus)

    if focus_key is None:
        if plot_type != "categorical":
            raise ValueError(
                "当 color 是连续变量或基因时，如果要使用 focus，"
                "请同时提供 focus_key，例如 focus_key='cell_type', focus='B cell'。"
            )

        focus_values = values.astype(str)
        return focus_values.isin(focus_set)

    if focus_key not in adata.obs.columns:
        raise KeyError(f"focus_key='{focus_key}' 不在 adata.obs.columns 中。")

    focus_values = adata.obs[focus_key].copy()
    focus_values.index = focus_values.index.astype(str)
    focus_values = focus_values.astype(str)

    return focus_values.isin(focus_set)


def _normalize_geometry_to_polygons(geom):
    if geom is None:
        return []

    if isinstance(geom, (list, tuple)):
        polygons = []
        for g in geom:
            polygons.extend(_normalize_geometry_to_polygons(g))
        return polygons

    if hasattr(geom, "geom_type"):
        if geom.geom_type == "Polygon":
            return [geom]
        elif geom.geom_type == "MultiPolygon":
            return list(geom.geoms)
        else:
            return []

    return []


def _get_category_color_map(
    adata,
    color_key,
    categories,
    user_colors=None,
    na_color="#d9d9d9",
):
    categories = [str(c) for c in categories if not pd.isna(c)]
    color_map = {}

    if user_colors is not None:
        for cat in categories:
            if cat in user_colors:
                color_map[cat] = user_colors[cat]

    uns_key = f"{color_key}_colors"

    if adata is not None and uns_key in adata.uns:
        adata_colors = adata.uns[uns_key]

        if len(adata_colors) >= len(categories):
            for cat, col in zip(categories, adata_colors):
                if cat not in color_map:
                    color_map[cat] = col

    missing_categories = [cat for cat in categories if cat not in color_map]

    if len(missing_categories) > 0:
        base_cmap = plt.get_cmap("tab20", max(len(missing_categories), 1))

        for i, cat in enumerate(missing_categories):
            color_map[cat] = to_hex(base_cmap(i))

    return color_map


def _get_continuous_limits(values, vmin=None, vmax=None, vmax_percentile=99):
    values = np.asarray(values, dtype=float)
    finite_values = values[np.isfinite(values)]

    if finite_values.size == 0:
        return 0, 1

    if vmin is None:
        current_vmin = np.nanmin(finite_values)
    else:
        current_vmin = vmin

    if vmax is None:
        if vmax_percentile is None:
            current_vmax = np.nanmax(finite_values)
        else:
            current_vmax = np.nanpercentile(finite_values, vmax_percentile)
    else:
        current_vmax = vmax

    if current_vmin == current_vmax:
        current_vmax = current_vmin + 1e-9

    return current_vmin, current_vmax


def _add_category_legend(
    ax,
    color_map,
    legend_loc="right",
    title=None,
):
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=7,
            label=cat,
        )
        for cat, color in color_map.items()
    ]

    if len(handles) == 0:
        return None

    if legend_loc == "right":
        leg = ax.legend(
            handles=handles,
            title=title,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )

    elif legend_loc == "right_upper":
        leg = ax.legend(
            handles=handles,
            title=title,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            frameon=False,
        )

    elif legend_loc == "right_lower":
        leg = ax.legend(
            handles=handles,
            title=title,
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0),
            frameon=False,
        )

    elif legend_loc == "best":
        leg = ax.legend(
            handles=handles,
            title=title,
            loc="best",
            frameon=False,
        )

    else:
        leg = ax.legend(
            handles=handles,
            title=title,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )

    ax.add_artist(leg)
    return leg


def _as_list(x):
    if isinstance(x, str):
        return [x]
    return list(x)


def _as_focus_set(focus):
    if focus is None:
        return None

    if isinstance(focus, str):
        return {focus}

    return set(map(str, focus))


def _prepare_titles(title, color_keys):
    if title is None:
        return color_keys

    if isinstance(title, str):
        if len(color_keys) == 1:
            return [title]
        else:
            return [f"{title}: {c}" for c in color_keys]

    title = list(title)

    if len(title) != len(color_keys):
        raise ValueError("当 title 是列表时，长度必须和 color 数量一致。")

    return title


def _cell_outline_segments(geom):
    if geom is None or getattr(geom, "is_empty", True):
        return []
    geom_type = getattr(geom, "geom_type", None)
    if geom_type == "Polygon":
        rings = [geom.exterior, *geom.interiors]
        return [np.asarray(r.coords, dtype=float)[:, :2] for r in rings]
    if geom_type in ("MultiPolygon", "GeometryCollection"):
        return [
            segment for part in geom.geoms for segment in _cell_outline_segments(part)
        ]
    raise TypeError(
        "`cell_boundary` must contain Polygon or MultiPolygon geometries; "
        f"received {geom_type!r}."
    )


def _selected_boundary_geometries(adata, selected_idx, cell_boundary, cell_id_col):
    if isinstance(cell_boundary, Mapping):
        if cell_id_col is not None and cell_id_col in adata.obs:
            ids = np.asarray(adata.obs[cell_id_col])
        else:
            ids = np.asarray(adata.obs_names)
        selected = []
        for idx in selected_idx:
            key = ids[idx]
            geom = cell_boundary.get(key)
            if geom is None:
                geom = cell_boundary.get(str(key))
            if geom is not None and not geom.is_empty:
                selected.append((idx, geom))
        return selected
    if isinstance(cell_boundary, (str, bytes)) or not isinstance(
        cell_boundary, Sequence
    ):
        raise TypeError(
            "`cell_boundary` must be a cell-ID mapping or an adata-ordered sequence."
        )
    if len(cell_boundary) != adata.n_obs:
        raise ValueError(
            "An ordered `cell_boundary` sequence must have length adata.n_obs."
        )
    return [
        (idx, cell_boundary[idx])
        for idx in selected_idx
        if cell_boundary[idx] is not None and not cell_boundary[idx].is_empty
    ]


def _cell_polygon_patches(geom):
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        return [patch for part in geom.geoms for patch in _cell_polygon_patches(part)]
    if geom.geom_type != "Polygon":
        raise TypeError(f"Unsupported cell boundary geometry: {geom.geom_type!r}")
    vertices = []
    codes = []
    for i, ring in enumerate([geom.exterior, *geom.interiors]):
        xy = np.asarray(ring.coords, dtype=float)[:, :2]
        signed_area = np.sum(xy[:-1, 0] * xy[1:, 1] - xy[1:, 0] * xy[:-1, 1])
        if (i == 0 and signed_area < 0) or (i > 0 and signed_area > 0):
            xy = xy[::-1]
        ring_codes = np.full(len(xy), Path.LINETO, dtype=np.uint8)
        ring_codes[0] = Path.MOVETO
        ring_codes[-1] = Path.CLOSEPOLY
        vertices.append(xy)
        codes.append(ring_codes)
    return [PathPatch(Path(np.concatenate(vertices), np.concatenate(codes)))]


def _get_continuous_values_for_bin_plot(adata, color: str) -> np.ndarray:
    if color in adata.var_names:
        values = _get_gene_values(adata, color).astype(float)
        if len(values) != len(adata.obs):
            raise ValueError(
                f"Gene {color!r} must identify exactly one expression column."
            )
        return values
    if color in adata.obs.columns:
        values = adata.obs[color]
        if not pd.api.types.is_numeric_dtype(values.dtype):
            raise ValueError(
                f"adata.obs[{color!r}] is not numeric. For cell-type proportions, "
                "set category='your cell-type label' and agg='fraction'."
            )
        return pd.to_numeric(values, errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
    raise KeyError(f"{color!r} is neither a gene nor a numeric adata.obs column.")


def _normalize_color(color):
    return to_hex(color, keep_alpha=False)


def _get_celltype_palette(table, celltype_key, celltypes, palette=None):
    celltypes = list(celltypes)
    if palette is not None:
        return _resolve_explicit_palette(
            celltypes, palette, "celltype_palette", "cell type"
        )
    color_key = f"{celltype_key}_colors"
    series = table.obs[celltype_key]
    if color_key in table.uns and hasattr(series.dtype, "categories"):
        categories = [str(x) for x in series.cat.categories]
        colors = list(table.uns[color_key])
        if len(colors) >= len(categories):
            adata_palette = {
                category: _normalize_color(color)
                for category, color in zip(categories, colors)
            }
            if all((x in adata_palette for x in celltypes)):
                return {x: adata_palette[x] for x in celltypes}
    cmap = plt.get_cmap("tab20")
    return {x: to_hex(cmap(i % cmap.N)) for i, x in enumerate(celltypes)}


def _get_transcript_palette(genes, palette=None):
    genes = list(genes)
    if palette is not None:
        return _resolve_explicit_palette(genes, palette, "transcript_palette", "gene")
    cmap = plt.get_cmap("tab10") if len(genes) <= 10 else plt.get_cmap("tab20")
    return {gene: to_hex(cmap(i % cmap.N)) for i, gene in enumerate(genes)}


def _get_xy_affine_matrix(element, target_coordinate_system="global"):
    transformation = get_transformation(
        element, to_coordinate_system=target_coordinate_system
    )
    matrix = np.asarray(
        transformation.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
        dtype=float,
    )
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 affine matrix, received {matrix.shape}.")
    return matrix


def _transform_bbox(min_coordinate, max_coordinate, affine_matrix):
    xmin, ymin = map(float, min_coordinate)
    xmax, ymax = map(float, max_coordinate)
    corners = np.array(
        [[xmin, ymin, 1.0], [xmin, ymax, 1.0], [xmax, ymin, 1.0], [xmax, ymax, 1.0]],
        dtype=float,
    )
    transformed = (affine_matrix @ corners.T).T
    target_min = [float(transformed[:, 0].min()), float(transformed[:, 1].min())]
    target_max = [float(transformed[:, 0].max()), float(transformed[:, 1].max())]
    return (target_min, target_max)


def _nice_tick_step(vmin, vmax, target_intervals=6):
    span = abs(float(vmax) - float(vmin))
    if span <= 0:
        return 1.0
    raw_step = span / max(int(target_intervals), 1)
    exponent = int(np.floor(np.log10(raw_step)))
    candidates = []
    for exp in range(exponent - 2, exponent + 3):
        base = 10.0**exp
        for multiplier in (1.0, 2.0, 5.0):
            step = multiplier * base
            if step < 1:
                continue
            n_intervals = span / step
            candidates.append((abs(n_intervals - target_intervals), step))
    if len(candidates) == 0:
        return 1.0
    candidates.sort(key=lambda x: (x[0], x[1]))
    return float(candidates[0][1])


def _generate_ticks(vmin, vmax, step):
    vmin = float(vmin)
    vmax = float(vmax)
    step = float(step)
    if step <= 0:
        raise ValueError("Tick interval must be > 0.")
    start = np.ceil(vmin / step) * step
    end = np.floor(vmax / step) * step
    if start > end:
        return np.array([(vmin + vmax) / 2], dtype=float)
    ticks = np.arange(start, end + step * 0.5, step)
    return ticks.astype(float)


def _set_spatial_ticks(
    ax,
    affine_matrix=None,
    coordinate_input="anndata",
    min_coordinate=None,
    max_coordinate=None,
    x_tick_interval=None,
    y_tick_interval=None,
    target_tick_intervals=6,
):
    if coordinate_input == "anndata":
        if affine_matrix is None:
            return
        linear = affine_matrix[:2, :2]
        translation = affine_matrix[:2, 2]
        if not (np.isclose(linear[0, 1], 0) and np.isclose(linear[1, 0], 0)):
            warnings.warn(
                "The SpatialData transformation contains rotation or shear. Automatic AnnData-coordinate ticks cannot be applied.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        sx = float(linear[0, 0])
        sy = float(linear[1, 1])
        tx = float(translation[0])
        ty = float(translation[1])
        if np.isclose(sx, 0) or np.isclose(sy, 0):
            return
        if min_coordinate is not None and max_coordinate is not None:
            xmin, ymin = map(float, min_coordinate)
            xmax, ymax = map(float, max_coordinate)
        else:
            global_xlim = ax.get_xlim()
            global_ylim = ax.get_ylim()
            intrinsic_x = [(x - tx) / sx for x in global_xlim]
            intrinsic_y = [(y - ty) / sy for y in global_ylim]
            xmin = min(intrinsic_x)
            xmax = max(intrinsic_x)
            ymin = min(intrinsic_y)
            ymax = max(intrinsic_y)
        if x_tick_interval is None:
            x_tick_interval = _nice_tick_step(
                xmin, xmax, target_intervals=target_tick_intervals
            )
        if y_tick_interval is None:
            y_tick_interval = _nice_tick_step(
                ymin, ymax, target_intervals=target_tick_intervals
            )
        x_values = _generate_ticks(xmin, xmax, x_tick_interval)
        y_values = _generate_ticks(ymin, ymax, y_tick_interval)
        x_positions = sx * x_values + tx
        y_positions = sy * y_values + ty
        ax.set_xticks(x_positions)
        ax.set_yticks(y_positions)
        ax.set_xticklabels([str(int(round(value))) for value in x_values])
        ax.set_yticklabels([str(int(round(value))) for value in y_values])
        ax.set_xlabel("Spatial x")
        ax.set_ylabel("Spatial y")
        return
    if min_coordinate is not None and max_coordinate is not None:
        xmin, ymin = map(float, min_coordinate)
        xmax, ymax = map(float, max_coordinate)
    else:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmin = min(xlim)
        xmax = max(xlim)
        ymin = min(ylim)
        ymax = max(ylim)
    if x_tick_interval is None:
        x_tick_interval = _nice_tick_step(
            xmin, xmax, target_intervals=target_tick_intervals
        )
    if y_tick_interval is None:
        y_tick_interval = _nice_tick_step(
            ymin, ymax, target_intervals=target_tick_intervals
        )
    x_values = _generate_ticks(xmin, xmax, x_tick_interval)
    y_values = _generate_ticks(ymin, ymax, y_tick_interval)
    ax.set_xticks(x_values)
    ax.set_yticks(y_values)
    ax.set_xticklabels([str(int(round(value))) for value in x_values])
    ax.set_yticklabels([str(int(round(value))) for value in y_values])


def _get_present_values(values, requested_order):
    subset = values[values.isin(requested_order)]
    unique_values = subset.dropna().unique()
    if hasattr(unique_values, "compute"):
        unique_values = unique_values.compute()
    if hasattr(unique_values, "tolist"):
        unique_values = unique_values.tolist()
    present_set = {str(value) for value in unique_values}
    return [value for value in requested_order if value in present_set]


def _add_separate_legends(
    ax,
    celltypes,
    celltype_palette,
    genes,
    transcript_palette,
    cell_fill_alpha=0.18,
    show_boundaries=True,
    show_transcripts=True,
    cell_legend_title="Cell type",
    transcript_legend_title="Transcript",
    cell_legend_bbox=(1.02, 1.0),
    transcript_legend_bbox=None,
    legend_gap=0.06,
    legend_fontsize=9,
    legend_title_fontsize=10,
    transcript_legend_size=6,
    cell_legend_ncol=1,
    transcript_legend_ncol=1,
):
    old_legend = ax.get_legend()
    if old_legend is not None:
        old_legend.remove()
    cell_legend = None
    if show_boundaries and len(celltypes) > 0:
        cell_handles = [
            Patch(
                facecolor=to_rgba(celltype_palette[celltype], alpha=cell_fill_alpha),
                edgecolor=celltype_palette[celltype],
                linewidth=1.0,
                label=celltype,
            )
            for celltype in celltypes
        ]
        cell_legend = ax.legend(
            handles=cell_handles,
            title=cell_legend_title,
            loc="upper left",
            bbox_to_anchor=cell_legend_bbox,
            frameon=False,
            fontsize=legend_fontsize,
            title_fontsize=legend_title_fontsize,
            ncol=cell_legend_ncol,
            borderaxespad=0,
        )
        ax.add_artist(cell_legend)
    if show_transcripts and len(genes) > 0:
        transcript_handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                markerfacecolor=transcript_palette[gene],
                markeredgecolor="none",
                markersize=transcript_legend_size,
                label=gene,
            )
            for gene in genes
        ]
        if transcript_legend_bbox is None and cell_legend is not None:
            ax.figure.canvas.draw()
            renderer = ax.figure.canvas.get_renderer()
            bbox_display = cell_legend.get_window_extent(renderer=renderer)
            bbox_axes = bbox_display.transformed(ax.transAxes.inverted())
            transcript_legend_bbox = (bbox_axes.x1 + legend_gap, 1.0)
        elif transcript_legend_bbox is None:
            transcript_legend_bbox = (1.02, 1.0)
        ax.legend(
            handles=transcript_handles,
            title=transcript_legend_title,
            loc="upper left",
            bbox_to_anchor=transcript_legend_bbox,
            frameon=False,
            fontsize=legend_fontsize,
            title_fontsize=legend_title_fontsize,
            ncol=transcript_legend_ncol,
            borderaxespad=0,
        )


def _resolve_explicit_palette(labels, palette, parameter_name, item_name):
    if isinstance(palette, dict):
        missing = [label for label in labels if label not in palette]
        if missing:
            raise KeyError(f"`{parameter_name}` is missing colors for: {missing}")
        return {label: _normalize_color(palette[label]) for label in labels}
    palette = list(palette)
    if len(palette) < len(labels):
        raise ValueError(
            f"`{parameter_name}` must contain at least one color per {item_name}."
        )
    return {label: _normalize_color(palette[i]) for i, label in enumerate(labels)}


def _get_gene_values(adata, gene):
    expression = adata[:, gene].X
    if sparse.issparse(expression):
        expression = expression.toarray()
    return np.asarray(expression).reshape(-1)
