from __future__ import annotations

import random

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F


def _set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _row_normalize(matrix):
    row_sum = np.asarray(matrix.sum(axis=1)).ravel()
    return matrix.multiply((1.0 / np.maximum(row_sum, 1e-12))[:, None]).tocsr()


def _to_torch_sparse(matrix, device):
    matrix = matrix.tocoo()
    indices = torch.from_numpy(np.vstack([matrix.row, matrix.col]).astype(np.int64))
    values = torch.from_numpy(matrix.data.astype(np.float32, copy=False))
    return torch.sparse_coo_tensor(
        indices,
        values,
        matrix.shape,
        device=device,
    ).coalesce()


def _distance_key(network_key):
    if network_key.endswith("_network"):
        return network_key[:-8] + "_distance"
    return network_key.replace("network", "distance")


def _first_order_weights(graph, distance):
    graph = graph.tocsr().copy()
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph.data[:] = 1.0
    graph.sort_indices()

    distance = distance.tocsr().multiply(graph).tocsr()
    distance.eliminate_zeros()
    distance.sort_indices()
    sigma = max(float(np.median(distance.data)), 1e-6)
    weights = distance.copy()
    weights.data = np.exp(-weights.data / sigma).astype(np.float32)
    weights = _row_normalize(weights)
    return graph, weights, sigma


def _exact_second_order_weights(
    graph,
    spatial,
    scale_factor=1.0,
):
    binary = graph.astype(np.float32, copy=True)
    binary.data[:] = 1.0
    exact2 = (binary @ binary).tocsr()
    exact2.data[:] = 1.0
    exact2.setdiag(0)
    exact2.eliminate_zeros()
    exact2 = (exact2 - exact2.multiply(binary)).tocsr()
    exact2.eliminate_zeros()
    exact2.data[:] = 1.0
    exact2.sort_indices()

    if exact2.nnz == 0:
        return exact2, exact2.copy(), np.nan

    counts = np.diff(exact2.indptr)
    rows = np.repeat(np.arange(exact2.shape[0]), counts)
    cols = exact2.indices
    distances = np.linalg.norm(spatial[rows] - spatial[cols], axis=1)
    sigma = max(float(np.median(distances)) * scale_factor, 1e-6)
    data = np.exp(-distances / sigma).astype(np.float32)
    weights = sp.csr_matrix((data, (rows, cols)), shape=exact2.shape)
    weights = _row_normalize(weights)
    return exact2, weights, sigma


def _prepare_inputs(
    adata,
    x_key,
    network_key,
    spatial_key,
    n_pcs,
    second_order_scale,
    device,
):
    x = np.asarray(adata.obsm[x_key], dtype=np.float32)
    if n_pcs is not None:
        x = x[:, :n_pcs]
    x = np.ascontiguousarray(x)
    graph, first_weights, sigma1 = _first_order_weights(
        adata.obsp[network_key],
        adata.obsp[_distance_key(network_key)],
    )
    _, second_weights, sigma2 = _exact_second_order_weights(
        graph,
        np.asarray(adata.obsm[spatial_key], dtype=np.float32),
        scale_factor=second_order_scale,
    )
    result = {
        "x": torch.tensor(x, dtype=torch.float32, device=device),
        "first_weights": _to_torch_sparse(first_weights, device),
        "first_weights_np": first_weights,
        "neighbor_ptr": torch.as_tensor(graph.indptr, dtype=torch.long, device=device),
        "neighbor_edges": torch.as_tensor(
            graph.indices, dtype=torch.long, device=device
        ),
        "second_weights": _to_torch_sparse(second_weights, device),
        "second_weights_np": second_weights,
        "sigma1": sigma1,
        "sigma2": sigma2,
    }
    return result


def _directional_reconstruction_loss(prediction, target):
    prediction = F.normalize(prediction, dim=1)
    target = F.normalize(target, dim=1)
    return (prediction - target).pow(2).sum(dim=1).mean()


def _reconstruction_cosine(prediction, target):
    prediction = F.normalize(prediction, dim=1)
    target = F.normalize(target, dim=1)
    return (prediction * target).sum(dim=1)


def _sample_neighbors(ptr, edges, nodes, k):
    start = ptr[nodes]
    count = ptr[nodes + 1] - start
    safe_count = count.clamp_min(1)
    offset = (
        torch.rand(nodes.numel(), k, device=nodes.device) * safe_count[:, None]
    ).long()
    index = (start[:, None] + offset).clamp_max(edges.numel() - 1)
    sampled = edges[index]
    return torch.where(count[:, None] > 0, sampled, nodes[:, None])


def _neighbor_infonce(
    embedding,
    x,
    ptr,
    edges,
    nodes,
    tau,
    k_pos,
    k_neg,
    similarity_power,
    use_similarity_weighted_contrastive,
):
    if edges.numel() == 0:
        return embedding.new_zeros(())
    query = F.normalize(embedding[nodes], dim=1)
    positive_index = _sample_neighbors(ptr, edges, nodes, k_pos)
    positive = F.normalize(embedding[positive_index], dim=2)
    positive_pair_logits = (positive * query[:, None, :]).sum(dim=2)

    if use_similarity_weighted_contrastive:
        x_query = F.normalize(x[nodes], dim=1)
        x_positive = F.normalize(x[positive_index], dim=2)
        similarity = (x_positive * x_query[:, None, :]).sum(dim=2).clamp(0, 1)
        weights = similarity.pow(similarity_power)
        positive_logit = (positive_pair_logits * weights).sum(
            dim=1, keepdim=True
        ) / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        anchor_weight = weights.mean(dim=1)
    else:
        positive_logit = positive_pair_logits.mean(dim=1, keepdim=True)
        anchor_weight = torch.ones(nodes.numel(), device=nodes.device)

    negative_index = torch.randint(
        0, embedding.size(0), (nodes.numel(), k_neg), device=nodes.device
    )
    negative = F.normalize(embedding[negative_index], dim=2)
    negative_logits = torch.bmm(negative, query[:, :, None]).squeeze(2)
    logits = torch.cat([positive_logit, negative_logits], dim=1) / tau
    per_node_loss = -F.log_softmax(logits, dim=1)[:, 0]
    return (per_node_loss * anchor_weight).sum() / anchor_weight.sum().clamp_min(1e-12)
