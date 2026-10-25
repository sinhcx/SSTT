from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from shapely.geometry import LineString, Point


def calculate_normalized_distances(
    adata,
    curve_result: dict,
    spatial_key: str = "spatial",
    side: Literal["both", "positive", "negative"] = "both",
    max_pos: float = 750,
    max_neg: float = 750,
    curve_key: str = "final_curve",
    closed: Optional[bool] = None,
    clip_open_ends: bool = True,
    normalization: Literal["global", "local"] = "local",
    local_window: float = 50,
    local_scale_quantile: float = 0.95,
    local_min_cells: int = 5,
    use_observed_max: bool = True,
    min_effective_max: float = 1e-8,
    verbose: bool = True,
):
    if side not in {"both", "positive", "negative"}:
        raise ValueError("`side` must be one of {'both', 'positive', 'negative'}.")

    if normalization not in {"global", "local"}:
        raise ValueError("`normalization` must be either 'global' or 'local'.")

    if spatial_key not in adata.obsm:
        raise KeyError(f"adata.obsm['{spatial_key}'] not found.")

    if curve_key not in curve_result:
        raise KeyError(
            f"`curve_result` does not contain key '{curve_key}'. "
            f"Available keys: {list(curve_result.keys())}"
        )

    if max_pos <= 0 or max_neg <= 0:
        raise ValueError("`max_pos` and `max_neg` must be positive.")

    if local_window <= 0:
        raise ValueError("`local_window` must be positive.")

    if not (0 < local_scale_quantile <= 1):
        raise ValueError("`local_scale_quantile` must be in the range (0, 1].")

    if local_min_cells <= 0:
        raise ValueError("`local_min_cells` must be positive.")

    curve = np.asarray(curve_result[curve_key], dtype=np.float64)

    if curve.ndim != 2 or curve.shape[1] != 2:
        raise ValueError(
            f"curve_result['{curve_key}'] must be an array with shape (n_points, 2)."
        )

    if curve.shape[0] < 2:
        raise ValueError("The curve must contain at least two points.")

    if closed is None:
        closed = bool(curve_result.get("closed", False))

    requested_max_pos = float(max_pos)
    requested_max_neg = float(max_neg)

    if closed and not np.allclose(curve[0], curve[-1]):
        curve_for_line = np.vstack([curve, curve[0]])
    else:
        curve_for_line = curve

    line = LineString(curve_for_line)

    if line.length == 0:
        raise ValueError("The curve length is zero. Please check curve coordinates.")

    cell_coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)

    if cell_coords.ndim != 2 or cell_coords.shape[1] < 2:
        raise ValueError(
            f"adata.obsm['{spatial_key}'] must be a 2D array with at least two columns."
        )

    cell_coords = cell_coords[:, :2]

    signed_dists = np.zeros(cell_coords.shape[0], dtype=float)
    curve_positions = np.zeros(cell_coords.shape[0], dtype=float)

    delta = 1e-6 * line.length

    for i, (x, y) in enumerate(cell_coords):
        pt = Point(float(x), float(y))

        proj_dist = line.project(pt)
        curve_positions[i] = proj_dist

        nearest_pt = line.interpolate(proj_dist)
        nx, ny = nearest_pt.x, nearest_pt.y

        p1 = line.interpolate(max(proj_dist - delta, 0))
        p2 = line.interpolate(min(proj_dist + delta, line.length))

        tangent = np.array([p2.x - p1.x, p2.y - p1.y], dtype=float)
        vec = np.array([x - nx, y - ny], dtype=float)

        dist = np.linalg.norm(vec)

        if np.linalg.norm(tangent) < 1e-12:
            signed_dists[i] = dist
            continue

        cross_z = tangent[0] * vec[1] - tangent[1] * vec[0]

        # The curve direction determines the sign: nonnegative cross products
        # are negative, and negative cross products are positive.
        signed_dists[i] = -dist if cross_z >= 0 else dist

    if (not closed) and clip_open_ends:
        start_tangent = _get_endpoint_tangent(curve, which="start")
        end_tangent = _get_endpoint_tangent(curve, which="end")

        start_vec = cell_coords - curve[0]
        end_vec = cell_coords - curve[-1]

        within_start = np.sum(start_vec * start_tangent[None, :], axis=1) >= 0
        within_end = np.sum(end_vec * end_tangent[None, :], axis=1) <= 0

        end_clip_mask = within_start & within_end
    else:
        end_clip_mask = np.ones(cell_coords.shape[0], dtype=bool)

    if side == "both":
        mask = (
            end_clip_mask
            & (signed_dists >= -requested_max_neg)
            & (signed_dists < requested_max_pos)
        )
    elif side == "positive":
        mask = end_clip_mask & (signed_dists >= 0) & (signed_dists < requested_max_pos)
    else:
        mask = end_clip_mask & (signed_dists >= -requested_max_neg) & (signed_dists < 0)

    adata_filtered = adata[mask].copy()

    filtered_dists = signed_dists[mask]
    filtered_positions = curve_positions[mask]

    if filtered_dists.size == 0:
        raise ValueError(
            "No cells remain after distance filtering. "
            "Try increasing `max_pos`/`max_neg`, changing `side`, "
            "or setting `clip_open_ends=False`."
        )

    pos_dists = filtered_dists[filtered_dists >= 0]
    neg_dists = filtered_dists[filtered_dists < 0]

    observed_max_pos = float(np.max(pos_dists)) if pos_dists.size > 0 else 0.0
    observed_max_neg = float(abs(np.min(neg_dists))) if neg_dists.size > 0 else 0.0

    if use_observed_max:
        if side in {"both", "positive"} and observed_max_pos > 0:
            effective_max_pos = min(requested_max_pos, observed_max_pos)
        else:
            effective_max_pos = requested_max_pos

        if side in {"both", "negative"} and observed_max_neg > 0:
            effective_max_neg = min(requested_max_neg, observed_max_neg)
        else:
            effective_max_neg = requested_max_neg
    else:
        effective_max_pos = requested_max_pos
        effective_max_neg = requested_max_neg

    effective_max_pos = max(effective_max_pos, min_effective_max)
    effective_max_neg = max(effective_max_neg, min_effective_max)

    if verbose and use_observed_max:
        if (
            side in {"both", "positive"}
            and observed_max_pos > 0
            and observed_max_pos < requested_max_pos
        ):
            print(
                f"[Info] positive max adjusted: "
                f"requested={requested_max_pos:.3f}, observed={observed_max_pos:.3f}. "
                f"Use effective_max_pos={effective_max_pos:.3f} for global fallback."
            )

        if (
            side in {"both", "negative"}
            and observed_max_neg > 0
            and observed_max_neg < requested_max_neg
        ):
            print(
                f"[Info] negative max adjusted: "
                f"requested={requested_max_neg:.3f}, observed={observed_max_neg:.3f}. "
                f"Use effective_max_neg={effective_max_neg:.3f} for global fallback."
            )

    if normalization == "global":
        if side == "positive":
            scales = np.full(filtered_dists.shape[0], effective_max_pos)
        elif side == "negative":
            scales = np.full(filtered_dists.shape[0], effective_max_neg)
        else:
            scales = np.where(
                filtered_dists >= 0,
                effective_max_pos,
                effective_max_neg,
            )

    else:
        scales = _compute_local_normalization_scales(
            signed_dists=filtered_dists,
            curve_positions=filtered_positions,
            side=side,
            line_length=line.length,
            closed=closed,
            local_window=local_window,
            local_scale_quantile=local_scale_quantile,
            local_min_cells=local_min_cells,
            fallback_pos=effective_max_pos,
            fallback_neg=effective_max_neg,
            min_effective_max=min_effective_max,
        )

    scales = np.maximum(scales, min_effective_max)

    normalized_dist = filtered_dists / scales

    if side == "positive":
        normalized_dist = np.clip(normalized_dist, 0.0, 1.0)

    elif side == "negative":
        normalized_dist = np.clip(normalized_dist, -1.0, 0.0)

    else:
        normalized_dist = np.clip(normalized_dist, -1.0, 1.0)

    adata_filtered.obs["signed_dist"] = filtered_dists
    adata_filtered.obs["abs_dist"] = np.abs(filtered_dists)

    adata_filtered.obs["dist_side"] = np.where(
        filtered_dists >= 0,
        "positive",
        "negative",
    )

    adata_filtered.obs["curve_position"] = filtered_positions
    adata_filtered.obs["normalization_scale"] = scales

    if normalization == "local":
        adata_filtered.obs["local_scale"] = scales

    adata_filtered.obs["normalized_dist"] = normalized_dist

    adata_filtered.uns["distance_stats"] = {
        "spatial_key": spatial_key,
        "side": side,
        "requested_max_pos": requested_max_pos,
        "requested_max_neg": requested_max_neg,
        "observed_max_pos": observed_max_pos,
        "observed_max_neg": observed_max_neg,
        "effective_max_pos": effective_max_pos,
        "effective_max_neg": effective_max_neg,
        "use_observed_max": use_observed_max,
        "clip_open_ends": clip_open_ends,
        "normalization": normalization,
        "local_window": local_window if normalization == "local" else None,
        "local_scale_quantile": (
            local_scale_quantile if normalization == "local" else None
        ),
        "local_min_cells": local_min_cells if normalization == "local" else None,
        "normalized_dist_min": float(np.nanmin(normalized_dist)),
        "normalized_dist_max": float(np.nanmax(normalized_dist)),
        "expected_normalized_range": ("positive: 0~1; negative: -1~0; both: -1~1"),
        "n_input_cells": int(adata.n_obs),
        "n_selected_cells": int(adata_filtered.n_obs),
        "mean_dist": float(np.mean(filtered_dists)),
        "mean_abs_dist": float(np.mean(np.abs(filtered_dists))),
        "mean_normalized_dist": float(
            np.mean(normalized_dist[~np.isnan(normalized_dist)])
        ),
        "mean_normalization_scale": float(np.mean(scales)),
        "median_normalization_scale": float(np.median(scales)),
        "curve_key": curve_key,
        "closed": closed,
        "sign_rule": (
            "cross_z >= 0 -> negative; cross_z < 0 -> positive. "
            "The curve direction follows curve_result[curve_key] point order."
        ),
        "open_end_rule": (
            "For open curves with clip_open_ends=True, cells are retained only "
            "between the endpoint-normal boundaries."
        ),
    }

    return adata_filtered


def _compute_local_normalization_scales(
    signed_dists: np.ndarray,
    curve_positions: np.ndarray,
    side: Literal["both", "positive", "negative"],
    line_length: float,
    closed: bool,
    local_window: float,
    local_scale_quantile: float,
    local_min_cells: int,
    fallback_pos: float,
    fallback_neg: float,
    min_effective_max: float = 1e-8,
) -> np.ndarray:
    signed_dists = np.asarray(signed_dists, dtype=float)
    curve_positions = np.asarray(curve_positions, dtype=float)

    scales = np.zeros_like(signed_dists, dtype=float)

    side_labels = np.where(signed_dists >= 0, "positive", "negative")

    if side == "positive":
        target_sides = ["positive"]
    elif side == "negative":
        target_sides = ["negative"]
    else:
        target_sides = ["positive", "negative"]

    for target_side in target_sides:
        side_mask = side_labels == target_side

        if not np.any(side_mask):
            continue

        fallback = fallback_pos if target_side == "positive" else fallback_neg

        idx_global = np.where(side_mask)[0]
        pos = curve_positions[idx_global]
        abs_dist = np.abs(signed_dists[idx_global])

        local_scales = _compute_local_scale_one_side(
            positions=pos,
            abs_distances=abs_dist,
            line_length=line_length,
            closed=closed,
            local_window=local_window,
            local_scale_quantile=local_scale_quantile,
            local_min_cells=local_min_cells,
            fallback=fallback,
            min_effective_max=min_effective_max,
        )

        scales[idx_global] = local_scales

    zero_mask = scales <= 0

    if np.any(zero_mask):
        fallback_values = np.where(
            signed_dists[zero_mask] >= 0,
            fallback_pos,
            fallback_neg,
        )
        scales[zero_mask] = fallback_values

    scales = np.maximum(scales, min_effective_max)

    return scales


def _compute_local_scale_one_side(
    positions: np.ndarray,
    abs_distances: np.ndarray,
    line_length: float,
    closed: bool,
    local_window: float,
    local_scale_quantile: float,
    local_min_cells: int,
    fallback: float,
    min_effective_max: float = 1e-8,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    abs_distances = np.asarray(abs_distances, dtype=float)

    n = len(positions)

    if n == 0:
        return np.array([], dtype=float)

    if n == 1:
        return np.full(1, max(abs_distances[0], fallback, min_effective_max))

    order = np.argsort(positions)
    pos_sorted = positions[order]
    dist_sorted = abs_distances[order]

    scale_sorted = np.zeros(n, dtype=float)

    if closed:
        pos_ext = np.concatenate(
            [
                pos_sorted - line_length,
                pos_sorted,
                pos_sorted + line_length,
            ]
        )
        dist_ext = np.concatenate([dist_sorted, dist_sorted, dist_sorted])

        offset = n

        for sorted_i in range(n):
            center_pos = pos_sorted[sorted_i]
            center_idx = offset + sorted_i

            left = np.searchsorted(
                pos_ext,
                center_pos - local_window,
                side="left",
            )
            right = np.searchsorted(
                pos_ext,
                center_pos + local_window,
                side="right",
            )

            local_values = dist_ext[left:right]

            if local_values.size < local_min_cells:
                half = local_min_cells // 2

                left2 = max(0, center_idx - half)
                right2 = min(len(pos_ext), left2 + local_min_cells)

                if right2 - left2 < local_min_cells:
                    left2 = max(0, right2 - local_min_cells)

                local_values = dist_ext[left2:right2]

            scale = np.nanquantile(local_values, local_scale_quantile)

            if not np.isfinite(scale) or scale <= 0:
                scale = fallback

            scale_sorted[sorted_i] = max(scale, min_effective_max)

    else:
        for sorted_i in range(n):
            center_pos = pos_sorted[sorted_i]

            left = np.searchsorted(
                pos_sorted,
                center_pos - local_window,
                side="left",
            )
            right = np.searchsorted(
                pos_sorted,
                center_pos + local_window,
                side="right",
            )

            local_values = dist_sorted[left:right]

            if local_values.size < local_min_cells:
                half = local_min_cells // 2

                left2 = max(0, sorted_i - half)
                right2 = min(n, left2 + local_min_cells)

                if right2 - left2 < local_min_cells:
                    left2 = max(0, right2 - local_min_cells)

                local_values = dist_sorted[left2:right2]

            scale = np.nanquantile(local_values, local_scale_quantile)

            if not np.isfinite(scale) or scale <= 0:
                scale = fallback

            scale_sorted[sorted_i] = max(scale, min_effective_max)

    scales = np.zeros(n, dtype=float)
    scales[order] = scale_sorted

    return scales


def _get_endpoint_tangent(
    curve: np.ndarray,
    which: Literal["start", "end"],
    eps: float = 1e-12,
) -> np.ndarray:
    curve = np.asarray(curve, dtype=np.float64)

    if which == "start":
        anchor = curve[0]
        candidates = curve[1:] - anchor
    elif which == "end":
        anchor = curve[-1]
        candidates = anchor - curve[:-1]
        candidates = candidates[::-1]
    else:
        raise ValueError("`which` must be either 'start' or 'end'.")

    for vec in candidates:
        norm = np.linalg.norm(vec)
        if norm > eps:
            return vec / norm

    raise ValueError("Cannot compute endpoint tangent from degenerate curve.")
