from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix, csgraph
from scipy.spatial import Delaunay
from shapely import area, buffer, intersection, length, linestrings
from shapely.geometry import Polygon
from shapely.strtree import STRtree


def create_polygons(
    boundary_df: pd.DataFrame,
    verbose: bool = True,
) -> Dict[Any, Polygon]:
    columns = ["cell_id", "vertex_x", "vertex_y"]
    if "label_id" in boundary_df.columns:
        columns.insert(1, "label_id")

    df = boundary_df[columns]

    if "label_id" in df.columns:
        df = df.sort_values(["cell_id", "label_id"], kind="stable")

    cell_ids = df["cell_id"].to_numpy()
    vertices = df[["vertex_x", "vertex_y"]].to_numpy()

    unique_ids, start_idx = np.unique(cell_ids, return_index=True)
    end_idx = np.r_[start_idx[1:], len(cell_ids)]

    cell_id_to_polygon: Dict[Any, Polygon] = {}

    for cell_id, start, end in zip(unique_ids, start_idx, end_idx):
        coords = vertices[start:end]

        if coords.shape[0] < 3:
            continue

        polygon = Polygon(coords)
        if polygon.is_valid and not polygon.is_empty:
            cell_id_to_polygon[cell_id] = polygon

    if verbose:
        print(f"[INFO] Created {len(cell_id_to_polygon)} valid cell polygons.")

    return cell_id_to_polygon


def build_neighborhood_network(
    adata,
    polys_dict: Optional[Dict[Any, Polygon]] = None,
    spatial_key: str = "spatial",
    delaunay_dist_thr: Optional[float] = None,
    delaunay_q: float = 0.98,
    delaunay_visibility_filter: bool = True,
    delaunay_corridor_width: float = 0.0,
    boundary_thr: float = 2.0,
    save_boundary_network: bool = False,
    verbose: bool = True,
    network_key: str = "sstt_neighbor_network",
    distance_key: str = "sstt_neighbor_distance",
    boundary_key: str = "sstt_boundary_network",
) -> None:
    if not isinstance(save_boundary_network, (bool, np.bool_)):
        raise TypeError("save_boundary_network must be a boolean.")

    output_keys = [network_key, distance_key]
    if save_boundary_network:
        output_keys.append(boundary_key)
    if len(set(output_keys)) != len(output_keys):
        raise ValueError("Output network keys must be distinct.")

    edges_delaunay = _build_delaunay_edges(
        adata=adata,
        spatial_key=spatial_key,
        distance_threshold=delaunay_dist_thr,
        auto_q=delaunay_q,
        polys_dict=polys_dict,
        visibility_filter=delaunay_visibility_filter,
        corridor_width=delaunay_corridor_width,
        verbose=verbose,
    )

    edges_list = [edges_delaunay]
    edges_boundary = np.empty((0, 2), dtype=np.int64)

    if polys_dict:
        edges_boundary = _build_boundary_edges(
            adata=adata,
            polys_dict=polys_dict,
            distance_threshold=boundary_thr,
            verbose=verbose,
        )
        edges_list.append(edges_boundary)
    elif verbose:
        print(
            "[WARN] No valid `polys_dict` provided. Only Delaunay graph will be used."
        )

    edges_union, edge_distances = _build_union_edges(
        adata=adata,
        edges_list=edges_list,
        spatial_key=spatial_key,
    )

    _write_neighbor_network_to_adata(
        adata=adata,
        edges=edges_union,
        edge_distances=edge_distances,
        spatial_key=spatial_key,
        network_key=network_key,
        distance_key=distance_key,
        verbose=verbose,
    )

    if save_boundary_network:
        adata.obsp[boundary_key] = _edge_array_to_binary_csr(
            n_nodes=adata.n_obs,
            edges=edges_boundary,
        )


def summarize_neighborhood_network(
    adata,
    network_key: str = "sstt_neighbor_network",
    distance_key: Optional[str] = "sstt_neighbor_distance",
    distance_quantiles: Sequence[float] = (0.95, 0.99),
    store_key: Optional[str] = None,
    store: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    if network_key not in adata.obsp:
        raise KeyError(f"Network not found in adata.obsp['{network_key}'].")

    distance_quantiles = tuple(float(q) for q in distance_quantiles)
    if any((q < 0 or q > 1) for q in distance_quantiles):
        raise ValueError("All values in `distance_quantiles` must be between 0 and 1.")

    A = adata.obsp[network_key]
    if not sparse.issparse(A):
        A = sparse.csr_matrix(A)
    else:
        A = A.tocsr()

    n_cells = adata.n_obs

    A = A.copy()
    A.setdiag(0)
    A.eliminate_zeros()

    # Treat the network as undirected for summary.
    A_bool = (A > 0).astype(np.float32).tocsr()
    A_bool = A_bool.maximum(A_bool.T)
    A_bool.setdiag(0)
    A_bool.eliminate_zeros()

    n_edges = A_bool.nnz // 2
    possible_edges = n_cells * (n_cells - 1) / 2
    density = n_edges / possible_edges if possible_edges > 0 else 0.0

    degree = np.asarray(A_bool.sum(axis=1)).ravel().astype(np.int64)
    island_mask = degree == 0
    island_cells = np.asarray(adata.obs_names)[island_mask]

    n_components, component_labels = csgraph.connected_components(
        A_bool,
        directed=False,
        return_labels=True,
    )

    component_sizes = np.bincount(component_labels, minlength=n_components)
    component_order = np.argsort(component_sizes)[::-1]

    component_table = pd.DataFrame(
        {
            "component_id": component_order,
            "n_cells": component_sizes[component_order],
            "fraction": component_sizes[component_order] / n_cells,
            "is_island_component": component_sizes[component_order] == 1,
        }
    )

    largest_component_size = (
        int(component_table["n_cells"].iloc[0]) if n_components > 0 else 0
    )
    largest_component_fraction = (
        largest_component_size / n_cells if n_cells > 0 else 0.0
    )

    degree_summary = pd.Series(degree).describe(
        percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    )

    summary = {
        "network_key": network_key,
        "n_cells": int(n_cells),
        "n_edges": int(n_edges),
        "density": float(density),
        "n_components": int(n_components),
        "largest_component_size": int(largest_component_size),
        "largest_component_fraction": float(largest_component_fraction),
        "n_island_cells": int(island_mask.sum()),
        "island_fraction": float(island_mask.mean()),
        "mean_degree": float(degree.mean()) if degree.size > 0 else 0.0,
        "median_degree": float(np.median(degree)) if degree.size > 0 else 0.0,
        "max_degree": int(degree.max()) if degree.size > 0 else 0,
    }

    distance_summary = None
    distance_quantile_table = None
    distance_values = None

    if distance_key is not None and distance_key in adata.obsp:
        D = adata.obsp[distance_key]
        if not sparse.issparse(D):
            D = sparse.csr_matrix(D)
        else:
            D = D.tocsr()

        D_upper = sparse.triu(D, k=1).tocoo()
        distance_values = np.asarray(D_upper.data, dtype=np.float64)

        if distance_values.size > 0:
            distance_summary = pd.Series(distance_values).describe(
                percentiles=list(distance_quantiles)
            )

            quantile_values = np.quantile(distance_values, distance_quantiles)
            distance_quantile_table = pd.DataFrame(
                {
                    "quantile": distance_quantiles,
                    "edge_distance": quantile_values,
                }
            )

            summary.update(
                {
                    "distance_key": distance_key,
                    "mean_edge_distance": float(distance_values.mean()),
                    "median_edge_distance": float(np.median(distance_values)),
                    "min_edge_distance": float(distance_values.min()),
                    "max_edge_distance": float(distance_values.max()),
                }
            )

            for q, value in zip(distance_quantiles, quantile_values):
                q_label = f"q{int(round(q * 100)):02d}_edge_distance"
                summary[q_label] = float(value)

        else:
            summary["distance_key"] = distance_key
            summary["mean_edge_distance"] = np.nan
            summary["median_edge_distance"] = np.nan
            summary["min_edge_distance"] = np.nan
            summary["max_edge_distance"] = np.nan

            for q in distance_quantiles:
                q_label = f"q{int(round(q * 100)):02d}_edge_distance"
                summary[q_label] = np.nan

    elif distance_key is not None:
        summary["distance_key"] = None
        if verbose:
            print(
                f"[WARN] Distance matrix adata.obsp['{distance_key}'] was not found. "
                "Distance statistics were skipped."
            )

    result = {
        "summary": summary,
        "component_table": component_table,
        "degree_summary": degree_summary,
        "distance_summary": distance_summary,
        "distance_quantile_table": distance_quantile_table,
        "island_cells": island_cells,
    }

    if store:
        if store_key is None:
            store_key = f"{network_key}_summary"
        adata.uns[store_key] = result

    if verbose:
        print(f"[INFO] Network summary for adata.obsp['{network_key}']")
        print(f"Cells: {summary['n_cells']:,}")
        print(f"Edges: {summary['n_edges']:,}")
        print(f"Components: {summary['n_components']:,}")
        print(
            f"Largest component: {summary['largest_component_size']:,} cells "
            f"({summary['largest_component_fraction']:.2%})"
        )
        print(
            f"Island cells: {summary['n_island_cells']:,} "
            f"({summary['island_fraction']:.2%})"
        )
        print(
            f"Degree: mean={summary['mean_degree']:.2f}, "
            f"median={summary['median_degree']:.2f}, "
            f"max={summary['max_degree']}"
        )

        if distance_quantile_table is not None:
            quantile_text = ", ".join(
                [
                    f"q{int(round(q * 100)):02d}={v:.2f}"
                    for q, v in zip(
                        distance_quantile_table["quantile"],
                        distance_quantile_table["edge_distance"],
                    )
                ]
            )
            print(
                f"Edge distance: mean={summary['mean_edge_distance']:.2f}, "
                f"median={summary['median_edge_distance']:.2f}, "
                f"max={summary['max_edge_distance']:.2f}; "
                f"{quantile_text}"
            )
        else:
            print("Edge distance: not available")

    return result


def _build_delaunay_edges(
    adata,
    spatial_key: str = "spatial",
    distance_threshold: Optional[float] = None,
    auto_q: float = 0.98,
    polys_dict: Optional[Dict[Any, Polygon]] = None,
    visibility_filter: bool = True,
    corridor_width: float = 0.0,
    verbose: bool = True,
) -> np.ndarray:
    if spatial_key not in adata.obsm:
        raise KeyError(f"Spatial coordinates not found in adata.obsm['{spatial_key}'].")
    if not 0 < auto_q <= 1:
        raise ValueError("`auto_q` must be in the range (0, 1].")
    if corridor_width < 0:
        raise ValueError("`corridor_width` must be >= 0.")

    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    n = coords.shape[0]

    if n < 3:
        if verbose:
            print(
                "[WARN] Fewer than 3 cells detected. "
                "Returning an empty Delaunay edge set."
            )
        return np.empty((0, 2), dtype=np.int64)

    tri = Delaunay(coords)
    edges_list: List[Tuple[int, int]] = []

    for simplex in tri.simplices:
        a, b, c = map(int, simplex)
        edges_list.append((min(a, b), max(a, b)))
        edges_list.append((min(a, c), max(a, c)))
        edges_list.append((min(b, c), max(b, c)))

    if not edges_list:
        return np.empty((0, 2), dtype=np.int64)

    edges = np.unique(np.asarray(edges_list, dtype=np.int64), axis=0)
    dists = np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=1)

    threshold = (
        float(distance_threshold)
        if distance_threshold is not None
        else float(np.quantile(dists, auto_q))
    )
    edges_distance_filtered = edges[dists <= threshold]

    edges_filtered = edges_distance_filtered
    n_occluded = 0
    n_unchecked = 0

    if visibility_filter and polys_dict and edges_distance_filtered.size > 0:
        edges_filtered, n_occluded, n_unchecked = _filter_delaunay_edges_by_visibility(
            adata=adata,
            edges=edges_distance_filtered,
            polys_dict=polys_dict,
            corridor_width=corridor_width,
            spatial_key=spatial_key,
        )

    if verbose:
        message = (
            f"[INFO] _build_delaunay_edges: {edges.shape[0]} candidates -> "
            f"{edges_distance_filtered.shape[0]} after distance filtering "
            f"(thr={threshold:.2f})"
        )
        if visibility_filter and polys_dict:
            message += (
                f" -> {edges_filtered.shape[0]} after visibility filtering "
                f"(removed={n_occluded}, unchecked={n_unchecked}, "
                f"corridor_width={corridor_width:g})"
            )
        print(message)

    return edges_filtered


def _filter_delaunay_edges_by_visibility(
    adata,
    edges: np.ndarray,
    polys_dict: Dict[Any, Polygon],
    corridor_width: float = 0.0,
    spatial_key: str = "spatial",
) -> Tuple[np.ndarray, int, int]:
    if edges is None or edges.size == 0:
        return np.empty((0, 2), dtype=np.int64), 0, 0

    if "cell_id" in adata.obs.columns:
        idx_to_id = np.asarray(adata.obs["cell_id"])
    else:
        idx_to_id = np.asarray(adata.obs_names)

    polygons: List[Polygon] = []
    node_to_tree_index = np.full(adata.n_obs, -1, dtype=np.int64)

    for node_index, cell_id in enumerate(idx_to_id):
        polygon = polys_dict.get(cell_id)
        if polygon is None or polygon.is_empty or not polygon.is_valid:
            continue

        node_to_tree_index[node_index] = len(polygons)
        polygons.append(polygon)

    if not polygons:
        return edges.copy(), 0, int(edges.shape[0])

    polygon_array = np.asarray(polygons, dtype=object)
    tree = STRtree(polygon_array)
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)[:, :2]

    src_tree_indices = node_to_tree_index[edges[:, 0]]
    dst_tree_indices = node_to_tree_index[edges[:, 1]]
    checkable_mask = (src_tree_indices >= 0) & (dst_tree_indices >= 0)

    checked_edge_indices = np.flatnonzero(checkable_mask)
    n_unchecked = int((~checkable_mask).sum())
    keep_mask = np.ones(edges.shape[0], dtype=bool)

    # Limit temporary geometry arrays for large spatial datasets.
    batch_size = 50_000
    overlap_tolerance = 1e-12

    for batch_start in range(0, checked_edge_indices.size, batch_size):
        batch_edge_indices = checked_edge_indices[
            batch_start : batch_start + batch_size
        ]
        batch_src_tree = src_tree_indices[batch_edge_indices]
        batch_dst_tree = dst_tree_indices[batch_edge_indices]

        batch_edges = edges[batch_edge_indices]
        connections = linestrings(
            np.stack(
                (coords[batch_edges[:, 0]], coords[batch_edges[:, 1]]),
                axis=1,
            )
        )
        test_geometries = (
            buffer(connections, corridor_width, cap_style="flat")
            if corridor_width > 0
            else connections
        )

        query_pairs = tree.query(test_geometries)
        if query_pairs.size == 0:
            continue

        query_indices = query_pairs[0]
        blocker_tree_indices = query_pairs[1]

        endpoint_mask = (blocker_tree_indices != batch_src_tree[query_indices]) & (
            blocker_tree_indices != batch_dst_tree[query_indices]
        )
        if not endpoint_mask.any():
            continue

        query_indices = query_indices[endpoint_mask]
        blocker_tree_indices = blocker_tree_indices[endpoint_mask]

        overlaps = intersection(
            test_geometries[query_indices],
            polygon_array[blocker_tree_indices],
        )

        if corridor_width > 0:
            blocking_mask = area(overlaps) > overlap_tolerance
        else:
            blocking_mask = length(overlaps) > overlap_tolerance

        if blocking_mask.any():
            blocked_batch_indices = np.unique(query_indices[blocking_mask])
            keep_mask[batch_edge_indices[blocked_batch_indices]] = False

    n_occluded = int((~keep_mask).sum())
    return edges[keep_mask], n_occluded, n_unchecked


def _build_boundary_edges(
    adata,
    polys_dict: Dict[Any, Polygon],
    distance_threshold: float = 5.0,
    verbose: bool = True,
) -> np.ndarray:
    if "cell_id" in adata.obs.columns:
        idx_to_id = np.asarray(adata.obs["cell_id"])
    else:
        idx_to_id = np.asarray(adata.obs_names)

    polygons: List[Polygon] = []
    valid_idx: List[int] = []

    for i, cell_id in enumerate(idx_to_id):
        if cell_id in polys_dict:
            polygons.append(polys_dict[cell_id])
            valid_idx.append(i)

    if not polygons:
        return np.empty((0, 2), dtype=np.int64)

    valid_idx = np.asarray(valid_idx, dtype=np.int64)

    buffer_radius = distance_threshold / 2.0
    buffered_polygons = [poly.buffer(buffer_radius) for poly in polygons]
    tree = STRtree(buffered_polygons)

    index_pairs = tree.query(buffered_polygons, predicate="intersects").T
    pairs = index_pairs[index_pairs[:, 0] < index_pairs[:, 1]]

    src = valid_idx[pairs[:, 0]]
    dst = valid_idx[pairs[:, 1]]

    edges = np.stack([src, dst], axis=1).astype(np.int64)
    edges = np.unique(edges, axis=0)

    if verbose:
        print(
            f"[INFO] _build_boundary_edges: kept {edges.shape[0]} edges "
            f"(boundary <= {distance_threshold} μm)"
        )

    return edges


def _build_union_edges(
    adata,
    edges_list: List[np.ndarray],
    spatial_key: str = "spatial",
) -> Tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)

    valid_edges = [
        edges for edges in edges_list if edges is not None and edges.size > 0
    ]
    if not valid_edges:
        return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.float32)

    edges_union = np.unique(np.vstack(valid_edges).astype(np.int64), axis=0)
    edge_distances = np.linalg.norm(
        coords[edges_union[:, 0]] - coords[edges_union[:, 1]],
        axis=1,
    ).astype(np.float32)

    return edges_union, edge_distances


def _write_neighbor_network_to_adata(
    adata,
    edges: np.ndarray,
    edge_distances: np.ndarray,
    spatial_key: str = "spatial",
    network_key: str = "sstt_neighbor_network",
    distance_key: str = "sstt_neighbor_distance",
    verbose: bool = True,
) -> None:
    n_nodes = adata.n_obs
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)

    A_bin = _edge_array_to_binary_csr(n_nodes=n_nodes, edges=edges)
    A_dist = _edge_array_to_distance_csr(
        n_nodes=n_nodes,
        edges=edges,
        coords=coords,
        dists=edge_distances,
    )

    adata.obsp[network_key] = A_bin
    adata.obsp[distance_key] = A_dist

    if verbose:
        print(
            f"[INFO] _write_neighbor_network_to_adata: "
            f"{network_key} edges = {A_bin.nnz // 2}"
        )


def _edge_array_to_binary_csr(
    n_nodes: int,
    edges: np.ndarray,
) -> csr_matrix:
    if edges is None or edges.size == 0:
        return sparse.csr_matrix((n_nodes, n_nodes), dtype=np.float32)

    src = edges[:, 0]
    dst = edges[:, 1]
    data = np.ones(edges.shape[0], dtype=np.float32)

    adjacency = sparse.csr_matrix(
        (data, (src, dst)),
        shape=(n_nodes, n_nodes),
        dtype=np.float32,
    )
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    return adjacency


def _edge_array_to_distance_csr(
    n_nodes: int,
    edges: np.ndarray,
    coords: np.ndarray,
    dists: Optional[np.ndarray] = None,
) -> csr_matrix:
    if edges is None or edges.size == 0:
        return sparse.csr_matrix((n_nodes, n_nodes), dtype=np.float32)

    src = edges[:, 0]
    dst = edges[:, 1]

    if dists is None:
        dists = np.linalg.norm(coords[src] - coords[dst], axis=1).astype(np.float32)

    adjacency = sparse.csr_matrix(
        (dists, (src, dst)),
        shape=(n_nodes, n_nodes),
        dtype=np.float32,
    )
    adjacency = adjacency + adjacency.T
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    return adjacency
