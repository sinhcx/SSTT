from __future__ import annotations

from collections.abc import Mapping
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import is_color_like
from matplotlib.patches import Rectangle


def spatial_sampling_preview(
    adata,
    spatial_key: str = "spatial",
    color: Optional[Union[str, Sequence]] = None,
    max_points: int = 50_000,
    random_state: int = 0,
    point_size: float = 1.0,
    point_alpha: float = 0.7,
    point_color: str = "black",
    cmap: str = "viridis",
    region: Optional[Tuple[float, float, float, float]] = None,
    region_edgecolor: str = "red",
    region_linewidth: float = 1.5,
    show_grid: bool = True,
    grid_alpha: float = 0.25,
    invert_y: bool = True,
    rasterized: bool = True,
    figsize: Tuple[float, float] = (7, 7),
    title: Optional[str] = None,
    ax: Optional[Axes] = None,
    show: bool = True,
) -> Axes:
    if spatial_key not in adata.obsm:
        raise KeyError(f"Spatial coordinates not found in adata.obsm['{spatial_key}'].")

    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            f"adata.obsm['{spatial_key}'] must be a 2D array with at least two columns."
        )

    coords = coords[:, :2]
    n_obs = adata.n_obs

    if max_points is None:
        selected_idx = np.arange(n_obs)
    else:
        if max_points <= 0:
            raise ValueError("`max_points` must be positive or None.")

        rng = np.random.default_rng(random_state)
        if n_obs > max_points:
            selected_idx = rng.choice(n_obs, size=max_points, replace=False)
            selected_idx = np.sort(selected_idx)
        else:
            selected_idx = np.arange(n_obs)

    plot_coords = coords[selected_idx]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if color is None:
        ax.scatter(
            plot_coords[:, 0],
            plot_coords[:, 1],
            s=point_size,
            c=point_color,
            alpha=point_alpha,
            linewidths=0,
            rasterized=rasterized,
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
                s=point_size,
                c=np.asarray(values),
                cmap=cmap,
                alpha=point_alpha,
                linewidths=0,
                rasterized=rasterized,
            )
            cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            if color_label is not None:
                cbar.set_label(color_label)

        else:
            values = pd.Categorical(values)
            categories = list(values.categories)

            base_cmap = plt.get_cmap("tab20")
            palette = {
                cat: base_cmap(i % base_cmap.N) for i, cat in enumerate(categories)
            }
            mapped_colors = [palette[cat] for cat in values]

            ax.scatter(
                plot_coords[:, 0],
                plot_coords[:, 1],
                s=point_size,
                c=mapped_colors,
                alpha=point_alpha,
                linewidths=0,
                rasterized=rasterized,
            )

    if region is not None:
        if len(region) != 4:
            raise ValueError("`region` must be provided as (xmin, xmax, ymin, ymax).")

        xmin, xmax, ymin, ymax = region
        rect = Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor=region_edgecolor,
            linewidth=region_linewidth,
        )
        ax.add_patch(rect)

    x_min, x_max = np.nanmin(coords[:, 0]), np.nanmax(coords[:, 0])
    y_min, y_max = np.nanmin(coords[:, 1]), np.nanmax(coords[:, 1])

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if show_grid:
        ax.grid(True, alpha=grid_alpha, linewidth=0.6)

    if invert_y:
        ax.invert_yaxis()

    if title is None:
        title = (
            f"Spatial sampling preview: " f"{selected_idx.size:,}/{n_obs:,} cells shown"
        )

    ax.set_title(title)

    if show:
        plt.show()

    return ax


def celltype_composition_stack(
    adata,
    niche_col: str = "niches",
    celltype_col: str = "celltype",
    normalize: bool = True,
    sort_index: bool = True,
    celltypes: Optional[Sequence[str]] = None,
    stat_celltypes: Optional[Sequence[str]] = None,
    stat_niches: Optional[Sequence[str]] = None,
    drop_empty_niches: bool = True,
    colors: Optional[Union[Sequence, Mapping]] = None,
    cmap: str = "tab20",
    figsize: Tuple[float, float] = (8, 5),
    width: float = 0.7,
    edgecolor: str = "white",
    title: str = "Cell type composition per niche",
    xlabel: str = "Niche",
    ylabel: Optional[str] = None,
    legend: bool = True,
    legend_bbox_to_anchor: Tuple[float, float] = (1.05, 1),
    legend_loc: str = "upper left",
    xtick_rotation: float = 0,
    ax: Optional[Axes] = None,
    show: bool = True,
) -> Axes:
    for col in (niche_col, celltype_col):
        if col not in adata.obs.columns:
            raise KeyError(f"'{col}' not found in adata.obs.")

    # Track whether the statistical cell-type filter was supplied.
    has_stat_celltype_filter = stat_celltypes is not None
    niche_series = adata.obs[niche_col]
    celltype_series = adata.obs[celltype_col]
    all_niches = (
        list(niche_series.cat.categories)
        if isinstance(niche_series.dtype, pd.CategoricalDtype)
        else list(pd.unique(niche_series.dropna()))
    )
    all_celltypes = (
        list(celltype_series.cat.categories)
        if isinstance(celltype_series.dtype, pd.CategoricalDtype)
        else list(pd.unique(celltype_series.dropna()))
    )

    df = adata.obs[[niche_col, celltype_col]].dropna().copy()
    if df.empty:
        raise ValueError("No valid rows remain after removing missing annotations.")

    selected_niches = list(stat_niches) if stat_niches is not None else all_niches
    if stat_niches is not None:
        missing = [n for n in selected_niches if n not in all_niches]
        if missing:
            raise ValueError(
                f"The following stat_niches were not found in '{niche_col}': {missing}"
            )
        df = df.loc[df[niche_col].isin(selected_niches)]
        if df.empty:
            raise ValueError("No cells remain after filtering by stat_niches.")

    stat_celltypes = (
        list(stat_celltypes) if stat_celltypes is not None else all_celltypes
    )
    missing = [ct for ct in stat_celltypes if ct not in all_celltypes]
    if missing:
        raise ValueError(
            f"The following stat_celltypes were not found in "
            f"'{celltype_col}': {missing}"
        )
    df = df.loc[df[celltype_col].isin(stat_celltypes)]
    if df.empty:
        raise ValueError("No cells remain after filtering by stat_celltypes.")

    count_table = (
        pd.crosstab(df[niche_col], df[celltype_col])
        .reindex(index=selected_niches, columns=stat_celltypes, fill_value=0)
        .fillna(0)
    )
    if drop_empty_niches:
        count_table = count_table.loc[count_table.sum(axis=1) > 0]
    if count_table.empty:
        raise ValueError("No valid niche remains for plotting.")

    display_celltypes = list(celltypes) if celltypes is not None else stat_celltypes
    missing = [ct for ct in display_celltypes if ct not in stat_celltypes]
    if missing:
        raise ValueError(
            f"The following celltypes are not included in stat_celltypes: {missing}"
        )
    if not display_celltypes:
        raise ValueError("At least one cell type must be selected for display.")

    plot_table = count_table.loc[:, display_celltypes].copy()
    if normalize:
        denominator = count_table.sum(axis=1).replace(0, np.nan)
        plot_table = plot_table.div(denominator, axis=0).fillna(0)
    if sort_index and stat_niches is None:
        plot_table = plot_table.sort_index()

    # Resolve colors against the original annotation order, before filtering.
    color_map = {}
    if colors is None or isinstance(colors, Mapping):
        cm = plt.get_cmap(cmap)
        palette = getattr(cm, "colors", None)
        if palette is None:
            palette = cm(np.linspace(0, 1, max(len(all_celltypes), 1)))
        color_map = {
            ct: palette[i % len(palette)] for i, ct in enumerate(all_celltypes)
        }
        stored_colors = adata.uns.get(f"{celltype_col}_colors")
        if isinstance(stored_colors, Mapping):
            stored_items = stored_colors.items()
        elif stored_colors is not None and not isinstance(stored_colors, (str, bytes)):
            stored_items = zip(all_celltypes, stored_colors)
        else:
            stored_items = ()
        for ct, color in stored_items:
            if ct in color_map and is_color_like(color):
                color_map[ct] = color

    if isinstance(colors, Mapping):
        for ct in display_celltypes:
            if ct in colors:
                if not is_color_like(colors[ct]):
                    raise ValueError(
                        f"Invalid explicit color for {ct!r}: {colors[ct]!r}"
                    )
                color_map[ct] = colors[ct]
        plot_colors = [color_map[ct] for ct in display_celltypes]
    elif colors is None:
        plot_colors = [color_map[ct] for ct in display_celltypes]
    else:
        explicit_colors = list(colors)
        if not explicit_colors:
            raise ValueError("colors must contain at least one color.")
        if not all(is_color_like(c) for c in explicit_colors):
            raise ValueError("colors contains an invalid Matplotlib color.")
        plot_colors = [
            explicit_colors[i % len(explicit_colors)]
            for i in range(len(display_celltypes))
        ]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(plot_table))
    bottom = np.zeros(len(plot_table), dtype=float)
    bar_handles = []
    for ct, color in zip(plot_table.columns, plot_colors):
        values = plot_table[ct].to_numpy(dtype=float)
        bars = ax.bar(
            x,
            values,
            bottom=bottom,
            label=str(ct),
            color=color,
            edgecolor=edgecolor,
            width=width,
        )
        bar_handles.append(bars)
        bottom += values

    if ylabel is None:
        ylabel = (
            "Fraction within selected cell types"
            if normalize and has_stat_celltype_filter
            else "Fraction of cells" if normalize else "Cell count"
        )
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_table.index.astype(str))
    ax.tick_params(axis="x", rotation=xtick_rotation)
    if normalize:
        ax.set_ylim(0, 1)
    if legend:
        # Match the legend order to the stack from top to bottom.
        ax.legend(
            handles=bar_handles[::-1],
            labels=[str(ct) for ct in plot_table.columns[::-1]],
            bbox_to_anchor=legend_bbox_to_anchor,
            loc=legend_loc,
            frameon=False,
        )
    if show:
        ax.figure.tight_layout()
        plt.show()
    return ax
