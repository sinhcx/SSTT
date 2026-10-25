from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pandas as pd
import scipy.sparse as sp


def neighbors(
    adata,
    n_neighbors=50,
    use_rep="X_pca",
    metric="euclidean",
    key_added=None,
    random_state=0,
):
    import cupy as cp
    from cuml.manifold.umap import fuzzy_simplicial_set
    from cuml.neighbors import NearestNeighbors

    values = np.asarray(adata.obsm[use_rep], dtype=np.float32, order="C")
    k = min(int(n_neighbors), values.shape[0])
    values_gpu = cp.asarray(values)
    estimator = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    estimator.fit(values_gpu)
    distances_gpu, indices_gpu = estimator.kneighbors(values_gpu)
    indices_gpu[:, 0] = cp.arange(values.shape[0])
    distances_gpu[:, 0] = 0.0
    graph = fuzzy_simplicial_set(
        cp.zeros((values.shape[0], 1), dtype=cp.float32),
        k,
        random_state,
        knn_indices=indices_gpu,
        knn_dists=distances_gpu,
        set_op_mix_ratio=1.0,
        local_connectivity=1.0,
    )
    if isinstance(graph, tuple):
        graph = graph[0]
    connectivities = sp.csr_matrix(graph.get(), dtype=np.float32)

    indices = cp.asnumpy(indices_gpu)
    distances = cp.asnumpy(distances_gpu)
    rows = np.repeat(np.arange(values.shape[0]), k)
    cols = indices.reshape(-1)
    data = distances.reshape(-1)
    keep = rows != cols
    distance_matrix = sp.csr_matrix(
        (data[keep], (rows[keep], cols[keep])),
        shape=(values.shape[0], values.shape[0]),
        dtype=np.float32,
    )
    if key_added is None or key_added == "neighbors":
        key_added = "neighbors"
        connectivity_key = "connectivities"
        distance_key = "distances"
    else:
        connectivity_key = f"{key_added}_connectivities"
        distance_key = f"{key_added}_distances"
    adata.obsp[connectivity_key] = connectivities
    adata.obsp[distance_key] = distance_matrix
    adata.uns[key_added] = {
        "connectivities_key": connectivity_key,
        "distances_key": distance_key,
        "params": {
            "n_neighbors": k,
            "metric": metric,
            "use_rep": use_rep,
            "method": "rapids_cuml",
        },
    }


def umap(
    adata,
    min_dist=0.5,
    spread=1.0,
    n_components=2,
    n_epochs=None,
    learning_rate=1.0,
    repulsion_strength=1.0,
    negative_sample_rate=5,
    init="spectral",
    random_state=0,
    key_added=None,
    neighbors_key="neighbors",
    verbose=False,
):
    import cupy as cp

    scales = (min_dist, spread, learning_rate, repulsion_strength)
    if (
        not np.isfinite(scales).all()
        or not 0 <= min_dist <= spread
        or min(spread, learning_rate, repulsion_strength) <= 0
    ):
        raise ValueError(
            "UMAP scales must be finite and positive, with 0 <= min_dist <= spread."
        )
    if not isinstance(n_components, (int, np.integer)) or n_components < 1:
        raise ValueError("n_components must be a positive integer.")
    if (
        not isinstance(negative_sample_rate, (int, np.integer))
        or negative_sample_rate < 1
    ):
        raise ValueError("negative_sample_rate must be a positive integer.")
    if n_epochs is not None and (
        not isinstance(n_epochs, (int, np.integer)) or n_epochs < 1
    ):
        raise ValueError("n_epochs must be a positive integer or None.")
    if neighbors_key is None:
        neighbors_key = "neighbors"
    key_obsm, key_uns = (
        ("X_umap", "umap") if key_added is None else (key_added, key_added)
    )
    if key_uns == neighbors_key:
        raise ValueError(
            "key_added must differ from neighbors_key to preserve graph metadata."
        )

    neighbors = adata.uns[neighbors_key]
    graph_key = neighbors["connectivities_key"]
    graph = sp.csr_matrix(adata.obsp[graph_key], dtype=np.float32, copy=True)
    graph.sum_duplicates()
    graph.eliminate_zeros()
    if adata.n_obs < 2 or graph.nnz == 0:
        raise ValueError(
            "UMAP requires at least two cells and a nonempty neighbor graph."
        )
    if not np.isfinite(graph.data).all() or np.any(graph.data < 0):
        raise ValueError("Neighbor graph weights must be finite and nonnegative.")

    if isinstance(init, str):
        if init not in {"spectral", "random"}:
            raise ValueError(
                "init must be 'spectral', 'random', or an initial coordinate array."
            )
        initial_positions = init
    else:
        positions = np.asarray(init, dtype=np.float32, order="C")
        if (
            positions.shape != (adata.n_obs, n_components)
            or not np.isfinite(positions).all()
        ):
            raise ValueError(
                "Initial coordinates must be finite with shape (n_obs, n_components)."
            )
        initial_positions = positions

    params = neighbors["params"]
    values = np.asarray(adata.obsm[params["use_rep"]], dtype=np.float32, order="C")
    epochs = (
        (500 if adata.n_obs <= 10_000 else 200) if n_epochs is None else int(n_epochs)
    )
    with _umap_accelerators(cp):
        from cupyx.scipy import sparse as gpu_sparse
        from cuml.manifold.umap import find_ab_params, simplicial_set_embedding

        if not isinstance(initial_positions, str):
            initial_positions = cp.array(initial_positions, dtype=cp.float32, copy=True)
        a, b = find_ab_params(spread, min_dist)

        # Optimize the stored fuzzy graph without repeating neighbor construction.
        embedding_gpu = simplicial_set_embedding(
            data=cp.asarray(values),
            graph=gpu_sparse.coo_matrix(graph.tocoo()),
            n_components=int(n_components),
            initial_alpha=float(learning_rate),
            a=float(a),
            b=float(b),
            gamma=float(repulsion_strength),
            negative_sample_rate=int(negative_sample_rate),
            n_epochs=epochs,
            init=initial_positions,
            random_state=random_state,
            metric=params["metric"],
            verbose=verbose,
        )
        coordinates = cp.asnumpy(cp.asarray(embedding_gpu))
    adata.obsm[key_obsm] = coordinates
    adata.uns[key_uns] = {
        "params": {
            "min_dist": float(min_dist),
            "spread": float(spread),
            "n_components": int(n_components),
            "n_epochs": epochs,
            "learning_rate": float(learning_rate),
            "repulsion_strength": float(repulsion_strength),
            "negative_sample_rate": int(negative_sample_rate),
            "init": init if isinstance(init, str) else "array",
            "random_state": random_state,
            "a": float(a),
            "b": float(b),
            "neighbors_key": neighbors_key,
            "connectivities_key": graph_key,
            "backend": "cuml",
        },
    }


def leiden(
    adata,
    resolution=1.0,
    random_state=0,
    key_added="leiden",
    neighbors_key=None,
    n_iterations=100,
):
    import cudf
    from cugraph import Graph
    from cugraph import leiden as cugraph_leiden

    if neighbors_key is None:
        neighbors_key = "neighbors"

    adjacency = adata.obsp[adata.uns[neighbors_key]["connectivities_key"]].tocsr()
    adjacency = adjacency.maximum(adjacency.T)
    upper = sp.triu(adjacency, k=1).tocoo()
    edges = cudf.DataFrame(
        {
            "source": upper.row.astype(np.int32),
            "destination": upper.col.astype(np.int32),
            "weight": upper.data.astype(np.float32),
        }
    )
    graph = Graph(directed=False)
    graph.from_cudf_edgelist(
        edges,
        source="source",
        destination="destination",
        weight="weight",
        vertices=cudf.Series(np.arange(adata.n_obs, dtype=np.int32)),
        renumber=False,
    )
    parts, modularity = cugraph_leiden(
        graph,
        resolution=float(resolution),
        random_state=int(random_state),
        max_iter=int(n_iterations),
    )
    parts = parts.to_pandas().sort_values("vertex")
    labels = np.empty(adata.n_obs, dtype=np.int64)
    labels[parts["vertex"].to_numpy()] = parts["partition"].to_numpy()
    unique = sorted(np.unique(labels), key=lambda x: np.flatnonzero(labels == x)[0])
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.asarray([remap[x] for x in labels])
    adata.obs[key_added] = pd.Categorical(labels.astype(str))
    adata.uns[key_added] = {
        "modularity": float(modularity),
        "params": {
            "resolution": float(resolution),
            "random_state": int(random_state),
            "n_iterations": int(n_iterations),
            "backend": "cugraph",
        },
    }


@contextmanager
def _umap_accelerators(cp):
    elementwise = list(cp._core.get_elementwise_accelerators())
    reduction = list(cp._core.get_reduction_accelerators())
    routine = list(cp._core.get_routine_accelerators())
    try:
        # Avoid the CUB compilation path during UMAP.
        cp._core.set_elementwise_accelerators([])
        cp._core.set_reduction_accelerators([])
        cp._core.set_routine_accelerators([])
        yield
    finally:
        cp._core.set_elementwise_accelerators(elementwise)
        cp._core.set_reduction_accelerators(reduction)
        cp._core.set_routine_accelerators(routine)


__all__ = ["neighbors", "umap", "leiden"]
