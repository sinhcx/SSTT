from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy.sparse
import torch
from scipy.spatial import cKDTree
from tqdm import tqdm

__all__ = ["calculate_secreted_signaling_score", "calculate_contact_signaling_score"]


def calculate_secreted_signaling_score(
    adata,
    LR_df,
    interaction_type="Secreted Signaling",
    type_col="annotation",
    layer_key=None,
    spatial_key="spatial",
    celltype_key="celltype",
    domain_col=None,
    domain_id=None,
    effective_range=100.0,
    min_expr_pct=0.05,
    n_permutations=1000,
    permutation_group_col=None,
    random_state=3407,
    chunk_size=5000,
    use_gpu=True,
):
    output_columns = [
        "Interaction",
        "Domain",
        "Sender",
        "Receiver",
        "Sender_Receiver",
        "Base_Score",
        "Mean_Ligand_Field",
        "Mean_Receptor_Expression",
        "Receiver_Expr_Pct",
        "N_Field_Receiver",
        "N_Total_Receiver",
        "Permutation_Mean",
        "Permutation_STD",
        "Z_score",
        "P_enrichment",
        "P_depletion",
        "P_value",
        "P_value_Two_Sided",
        "FDR",
        "Direction",
        "N_Permutations",
    ]
    if effective_range <= 0:
        raise ValueError("effective_range must be > 0")
    if int(n_permutations) < 1:
        raise ValueError("n_permutations must be >= 1")
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be >= 1")
    required_lr_columns = {"ligand.symbol", "receptor.symbol"}
    missing_lr_columns = required_lr_columns.difference(LR_df.columns)
    if missing_lr_columns:
        raise KeyError(
            "LR_df is missing required columns: "
            + ", ".join(sorted(missing_lr_columns))
        )
    if celltype_key not in adata.obs.columns:
        raise KeyError(f"{celltype_key!r} was not found in adata.obs")
    if spatial_key not in adata.obsm:
        raise KeyError(f"{spatial_key!r} was not found in adata.obsm")
    if permutation_group_col is not None:
        if permutation_group_col not in adata.obs.columns:
            raise KeyError(f"{permutation_group_col!r} was not found in adata.obs")
    selected_domain_id = domain_id
    if domain_col is not None and domain_id is not None:
        if domain_col not in adata.obs.columns:
            raise KeyError(f"{domain_col!r} was not found in adata.obs")
        domain_values = (
            [domain_id]
            if isinstance(domain_id, (str, int, np.integer))
            else list(domain_id)
        )
        mask = adata.obs[domain_col].isin(domain_values)
        _adata = adata[mask]
        selected_domain_id = domain_values
        print(
            f"Domain-specific analysis: {domain_col} in {domain_values}; n_cells = {_adata.n_obs}"
        )
    else:
        _adata = adata
        print(f"Global analysis; n_cells = {_adata.n_obs}")
    if _adata.n_obs == 0:
        return pd.DataFrame(columns=output_columns)
    domain_label = "All" if selected_domain_id is None else str(selected_domain_id)
    if type_col in LR_df.columns:
        n_total = len(LR_df)
        filtered_LR = LR_df[LR_df[type_col] == interaction_type].copy()
        print(
            f"Interaction filtering: {interaction_type}; retained {len(filtered_LR)}/{n_total} pairs"
        )
    else:
        filtered_LR = LR_df.copy()
        print("Interaction filtering skipped: type column not found")
    if len(filtered_LR) == 0:
        return pd.DataFrame(columns=output_columns)
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Computation device: {device}")
    sigma = float(effective_range) / 3.0
    sigma_sq = sigma * sigma
    coords_cpu = np.asarray(_adata.obsm[spatial_key], dtype=np.float64, order="C")
    num_cells = int(coords_cpu.shape[0])
    coords_device = torch.as_tensor(coords_cpu, dtype=torch.float32, device=device)
    spatial_tree = cKDTree(coords_cpu)
    query_workers = -1 if num_cells >= 50000 else 1
    sample_n = min(512, num_cells)
    sample_idx = np.linspace(0, num_cells - 1, sample_n, dtype=np.int64)
    neighbor_counts = spatial_tree.query_ball_point(
        coords_cpu[sample_idx],
        r=effective_range,
        workers=query_workers,
        return_length=True,
    )
    mean_neighbors = max(float(np.mean(neighbor_counts)), 1.0)
    if device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(device)
        edge_budget = int(np.clip(free_bytes * 0.05 / 48.0, 250000, 2500000))
    else:
        edge_budget = 500000
    adaptive_chunk_size = max(
        1, min(int(chunk_size), int(edge_budget / max(mean_neighbors * 2.0, 1.0)))
    )
    print(
        f"Sparse radius backend: cKDTree + {device.type} scatter; mean_neighbors ≈ {mean_neighbors:.1f}; source_chunk = {adaptive_chunk_size}"
    )
    expression_matrix = _adata.X if layer_key is None else _adata.layers[layer_key]
    var_names = _adata.var_names
    var_name_set = set(var_names)
    ligand_genes = set(filtered_LR["ligand.symbol"].astype(str))
    receptor_genes = {
        gene.strip()
        for receptor in filtered_LR["receptor.symbol"]
        for gene in str(receptor).split(",")
    }
    gene_to_col = {
        gene: int(var_names.get_loc(gene))
        for gene in ligand_genes | receptor_genes
        if gene in var_name_set
    }
    if not gene_to_col:
        return pd.DataFrame(columns=output_columns)
    gene_cache = OrderedDict()
    receptor_cache = OrderedDict()
    gene_cache_limit = 8
    receptor_cache_limit = 8
    cell_types = _adata.obs[celltype_key].astype(str).to_numpy()
    unique_types = np.unique(cell_types)
    type_indices = {
        cell_type: np.flatnonzero(cell_types == cell_type).astype(np.int64, copy=False)
        for cell_type in unique_types
    }
    type_indices_t = {
        cell_type: torch.as_tensor(indices, dtype=torch.long, device=device)
        for cell_type, indices in type_indices.items()
    }
    if permutation_group_col is None:
        permutation_groups = np.full(num_cells, "__all__", dtype=object)
    else:
        permutation_groups = (
            _adata.obs[permutation_group_col]
            .astype(str)
            .fillna("__missing__")
            .to_numpy()
        )
    type_group_indices = {}
    type_group_indices_t = {}
    for receiver_type, receiver_indices in type_indices.items():
        receiver_groups = permutation_groups[receiver_indices]
        group_dict = {}
        group_dict_t = {}
        for group_value in np.unique(receiver_groups):
            group_global_indices = receiver_indices[receiver_groups == group_value]
            group_dict[group_value] = group_global_indices
            group_dict_t[group_value] = torch.as_tensor(
                group_global_indices, dtype=torch.long, device=device
            )
        type_group_indices[receiver_type] = group_dict
        type_group_indices_t[receiver_type] = group_dict_t
    permutation_generator = torch.Generator(device=device)
    permutation_generator.manual_seed(int(random_state))
    all_results = []
    # Reuse each sender ligand exposure across its receptor pairs.
    grouped_lr = filtered_LR.groupby(
        filtered_LR["ligand.symbol"].astype(str), sort=False
    )
    with torch.inference_mode(), tqdm(
        total=len(filtered_LR), desc="Ligand field + receiver permutation"
    ) as progress:
        for ligand_gene, ligand_rows in grouped_lr:
            ligand_values = _get_gene_vector(
                ligand_gene,
                expression_matrix=expression_matrix,
                gene_to_col=gene_to_col,
                gene_cache=gene_cache,
                gene_cache_limit=gene_cache_limit,
            )
            if ligand_values is None:
                progress.update(len(ligand_rows))
                continue
            valid_senders = []
            for sender_type in unique_types:
                sender_indices = type_indices[sender_type]
                sender_positive = ligand_values[sender_indices] > 0
                sender_expr_pct = float(np.mean(sender_positive))
                if sender_expr_pct < min_expr_pct:
                    continue
                active_sender_indices = sender_indices[sender_positive]
                if active_sender_indices.size == 0:
                    continue
                valid_senders.append((sender_type, active_sender_indices))
            if not valid_senders:
                progress.update(len(ligand_rows))
                continue
            for sender_type, active_sender_indices in valid_senders:
                field_result = _compute_ligand_field(
                    source_indices=active_sender_indices,
                    source_values=ligand_values,
                    num_cells=num_cells,
                    device=device,
                    coords_device=coords_device,
                    sigma_sq=sigma_sq,
                    coords_cpu=coords_cpu,
                    spatial_tree=spatial_tree,
                    effective_range=effective_range,
                    adaptive_chunk_size=adaptive_chunk_size,
                    query_workers=query_workers,
                )
                if field_result is None:
                    continue
                field_indices, field_indices_t, ligand_field = field_result
                field_types = cell_types[field_indices]
                field_type_positions = {
                    receiver_type: np.flatnonzero(field_types == receiver_type).astype(
                        np.int64, copy=False
                    )
                    for receiver_type in unique_types
                }
                for _, row in ligand_rows.iterrows():
                    receptor_string = str(row["receptor.symbol"])
                    receptor_values = _get_receptor_vector(
                        receptor_string,
                        num_cells=num_cells,
                        expression_matrix=expression_matrix,
                        gene_to_col=gene_to_col,
                        gene_cache=gene_cache,
                        gene_cache_limit=gene_cache_limit,
                        receptor_cache=receptor_cache,
                        receptor_cache_limit=receptor_cache_limit,
                    )
                    if receptor_values is None or not np.any(receptor_values > 0):
                        continue
                    receptor_values_t = torch.as_tensor(
                        receptor_values, dtype=torch.float32, device=device
                    )
                    for receiver_type in unique_types:
                        receiver_local_positions = field_type_positions[receiver_type]
                        n_field_receiver = int(receiver_local_positions.size)
                        if n_field_receiver == 0:
                            continue
                        receiver_local_positions_t = torch.as_tensor(
                            receiver_local_positions, dtype=torch.long, device=device
                        )
                        field_receiver_global_indices = field_indices[
                            receiver_local_positions
                        ]
                        field_receiver_global_indices_t = field_indices_t.index_select(
                            0, receiver_local_positions_t
                        )
                        receiver_receptor_values = receptor_values_t.index_select(
                            0, field_receiver_global_indices_t
                        )
                        receptor_positive = receiver_receptor_values > 0
                        receiver_expr_pct = float(
                            receptor_positive.float().mean().item()
                        )
                        if receiver_expr_pct < min_expr_pct:
                            continue
                        receiver_ligand_field = ligand_field.index_select(
                            0, receiver_local_positions_t
                        )
                        receiver_scores = (
                            receiver_ligand_field * receiver_receptor_values
                        )
                        observed_score = float(receiver_scores.mean().item())
                        if not np.isfinite(observed_score) or observed_score <= 1e-07:
                            continue
                        permutation_result = _receiver_position_permutation(
                            receiver_type=receiver_type,
                            receiver_indices=field_receiver_global_indices,
                            exposure_values=receiver_ligand_field,
                            receptor_values_t=receptor_values_t,
                            observed_score=observed_score,
                            permutation_groups=permutation_groups,
                            type_group_indices_t=type_group_indices_t,
                            n_permutations=n_permutations,
                            permutation_generator=permutation_generator,
                            device=device,
                        )
                        all_results.append(
                            {
                                "Interaction": f"{ligand_gene}_{receptor_string}",
                                "Domain": domain_label,
                                "Sender": sender_type,
                                "Receiver": receiver_type,
                                "Sender_Receiver": f"{sender_type}->{receiver_type}",
                                "Base_Score": observed_score,
                                "Mean_Ligand_Field": float(
                                    receiver_ligand_field.mean().item()
                                ),
                                "Mean_Receptor_Expression": float(
                                    receiver_receptor_values.mean().item()
                                ),
                                "Receiver_Expr_Pct": receiver_expr_pct,
                                "N_Field_Receiver": n_field_receiver,
                                "N_Total_Receiver": int(
                                    type_indices_t[receiver_type].numel()
                                ),
                                **permutation_result,
                                "N_Permutations": int(n_permutations),
                            }
                        )
            progress.update(len(ligand_rows))
    result_df = pd.DataFrame(all_results)
    if result_df.empty:
        return pd.DataFrame(columns=output_columns)
    return _finalize_signaling_results(result_df, output_columns)


def calculate_contact_signaling_score(
    adata,
    LR_df,
    interaction_type="Cell-Cell Contact",
    type_col="annotation",
    layer_key=None,
    celltype_key="celltype",
    adj_key="sstt_boundary_network",
    domain_col=None,
    domain_id=None,
    min_expr_pct=0.05,
    n_permutations=1000,
    permutation_group_col=None,
    random_state=3407,
    use_gpu=True,
):
    output_columns = [
        "Interaction",
        "Domain",
        "Sender",
        "Receiver",
        "Sender_Receiver",
        "Base_Score",
        "Mean_Contact_Ligand_Exposure",
        "Mean_Receptor_Expression",
        "Receiver_Expr_Pct",
        "N_Contact_Receiver",
        "N_Total_Receiver",
        "Permutation_Mean",
        "Permutation_STD",
        "Z_score",
        "P_enrichment",
        "P_depletion",
        "P_value",
        "P_value_Two_Sided",
        "FDR",
        "Direction",
        "N_Permutations",
    ]
    n_permutations = int(n_permutations)
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")
    if not 0 <= float(min_expr_pct) <= 1:
        raise ValueError("min_expr_pct must be between 0 and 1")
    required_lr_columns = {"ligand.symbol", "receptor.symbol"}
    missing_lr_columns = required_lr_columns.difference(LR_df.columns)
    if missing_lr_columns:
        raise KeyError(
            "LR_df is missing required columns: "
            + ", ".join(sorted(missing_lr_columns))
        )
    if celltype_key not in adata.obs.columns:
        raise KeyError(f"{celltype_key!r} was not found in adata.obs")
    if adj_key not in adata.obsp:
        raise KeyError(f"{adj_key!r} was not found in adata.obsp")
    if permutation_group_col is not None:
        if permutation_group_col not in adata.obs.columns:
            raise KeyError(f"{permutation_group_col!r} was not found in adata.obs")
    selected_domain_id = domain_id
    if domain_col is not None and domain_id is not None:
        if domain_col not in adata.obs.columns:
            raise KeyError(f"{domain_col!r} was not found in adata.obs")
        domain_values = (
            [domain_id]
            if isinstance(domain_id, (str, int, np.integer))
            else list(domain_id)
        )
        domain_mask = adata.obs[domain_col].isin(domain_values)
        _adata = adata[domain_mask].copy()
        selected_domain_id = domain_values
        print(
            f"Domain-specific analysis: {domain_col} in {domain_values}; n_cells = {_adata.n_obs}"
        )
    else:
        _adata = adata
        print(f"Global analysis; n_cells = {_adata.n_obs}")
    if _adata.n_obs == 0:
        return pd.DataFrame(columns=output_columns)
    domain_label = "All" if selected_domain_id is None else str(selected_domain_id)
    if type_col in LR_df.columns:
        n_total = len(LR_df)
        filtered_LR = LR_df[LR_df[type_col] == interaction_type].copy()
        print(
            f"Interaction filtering: {interaction_type}; retained {len(filtered_LR)}/{n_total} pairs"
        )
    else:
        filtered_LR = LR_df.copy()
        print("Interaction filtering skipped: type column not found")
    if filtered_LR.empty:
        return pd.DataFrame(columns=output_columns)
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Computation device: {device}")
    num_cells = int(_adata.n_obs)
    adj_matrix = _adata.obsp[adj_key]
    if scipy.sparse.issparse(adj_matrix):
        adj_binary = adj_matrix.tocsr(copy=True)
        if not np.all(np.isfinite(adj_binary.data)):
            raise ValueError(f"{adj_key!r} contains non-finite edge values")
        if np.any(adj_binary.data < 0):
            raise ValueError(f"{adj_key!r} contains negative edge values")
        # Stored sparse entries define contact, including explicit zero entries.
        adj_binary.data[:] = 1.0
        adj_binary.setdiag(0.0)
        adj_binary.eliminate_zeros()
        adj_coo = adj_binary.tocoo(copy=False)
        adj_coo.sum_duplicates()
        indices_np = np.vstack((adj_coo.row, adj_coo.col)).astype(np.int64, copy=False)
        values_np = np.ones(adj_coo.nnz, dtype=np.float32)
        adj_t = torch.sparse_coo_tensor(
            torch.as_tensor(indices_np, dtype=torch.long, device=device),
            torch.as_tensor(values_np, dtype=torch.float32, device=device),
            size=adj_coo.shape,
            device=device,
        ).coalesce()
        sparse_contact_graph = True
    else:
        adj_values = np.asarray(adj_matrix)
        if not np.all(np.isfinite(adj_values)):
            raise ValueError(f"{adj_key!r} contains non-finite edge values")
        if np.any(adj_values < 0):
            raise ValueError(f"{adj_key!r} contains negative edge values")
        adj_binary = (adj_values > 0).astype(np.float32, copy=False)
        np.fill_diagonal(adj_binary, 0.0)
        adj_t = torch.as_tensor(adj_binary, dtype=torch.float32, device=device)
        sparse_contact_graph = False
    expression_matrix = _adata.X if layer_key is None else _adata.layers[layer_key]
    var_names = _adata.var_names
    var_name_set = set(var_names)
    ligand_genes = set(filtered_LR["ligand.symbol"].astype(str))
    receptor_genes = {
        gene.strip()
        for receptor in filtered_LR["receptor.symbol"]
        for gene in str(receptor).split(",")
    }
    gene_to_col = {
        gene: int(var_names.get_loc(gene))
        for gene in ligand_genes | receptor_genes
        if gene in var_name_set
    }
    if not gene_to_col:
        return pd.DataFrame(columns=output_columns)
    gene_cache = OrderedDict()
    receptor_cache = OrderedDict()
    gene_cache_limit = 8
    receptor_cache_limit = 8
    cell_types = _adata.obs[celltype_key].astype(str).to_numpy()
    unique_types = np.unique(cell_types)
    type_indices = {
        cell_type: np.flatnonzero(cell_types == cell_type).astype(np.int64, copy=False)
        for cell_type in unique_types
    }
    type_indices_t = {
        cell_type: torch.as_tensor(indices, dtype=torch.long, device=device)
        for cell_type, indices in type_indices.items()
    }
    if permutation_group_col is None:
        permutation_groups = np.full(num_cells, "__all__", dtype=object)
    else:
        permutation_groups = (
            _adata.obs[permutation_group_col]
            .astype(str)
            .fillna("__missing__")
            .to_numpy()
        )
    type_group_indices_t = {}
    for receiver_type, receiver_indices in type_indices.items():
        receiver_groups = permutation_groups[receiver_indices]
        group_dict_t = {}
        for group_value in np.unique(receiver_groups):
            group_global_indices = receiver_indices[receiver_groups == group_value]
            group_dict_t[group_value] = torch.as_tensor(
                group_global_indices, dtype=torch.long, device=device
            )
        type_group_indices_t[receiver_type] = group_dict_t
    permutation_generator = torch.Generator(device=device)
    permutation_generator.manual_seed(int(random_state))
    all_results = []
    # Reuse each sender ligand exposure across its receptor pairs.
    grouped_lr = filtered_LR.groupby(
        filtered_LR["ligand.symbol"].astype(str), sort=False
    )
    with torch.inference_mode(), tqdm(
        total=len(filtered_LR), desc="Contact exposure + receiver permutation"
    ) as progress:
        for ligand_gene, ligand_rows in grouped_lr:
            ligand_values = _get_gene_vector(
                ligand_gene,
                expression_matrix=expression_matrix,
                gene_to_col=gene_to_col,
                gene_cache=gene_cache,
                gene_cache_limit=gene_cache_limit,
            )
            if ligand_values is None:
                progress.update(len(ligand_rows))
                continue
            ligand_values_t = torch.as_tensor(
                ligand_values, dtype=torch.float32, device=device
            )
            valid_senders = []
            for sender_type in unique_types:
                sender_indices = type_indices[sender_type]
                sender_expr_pct = float(np.mean(ligand_values[sender_indices] > 0))
                if sender_expr_pct >= min_expr_pct:
                    valid_senders.append(sender_type)
            if not valid_senders:
                progress.update(len(ligand_rows))
                continue
            for sender_type in valid_senders:
                sender_indices_t = type_indices_t[sender_type]
                sender_ligand_t = torch.zeros(
                    num_cells, dtype=torch.float32, device=device
                )
                sender_ligand_t.index_copy_(
                    0,
                    sender_indices_t,
                    ligand_values_t.index_select(0, sender_indices_t),
                )
                contact_exposure_t = _contact_mm(
                    sender_ligand_t.unsqueeze(1),
                    sparse_contact_graph=sparse_contact_graph,
                    adj_t=adj_t,
                ).squeeze(1)
                contact_indices_t = torch.nonzero(
                    contact_exposure_t > 0, as_tuple=False
                ).flatten()
                if contact_indices_t.numel() == 0:
                    continue
                contact_indices = contact_indices_t.detach().cpu().numpy()
                contact_types = cell_types[contact_indices]
                contact_type_positions = {
                    receiver_type: np.flatnonzero(
                        contact_types == receiver_type
                    ).astype(np.int64, copy=False)
                    for receiver_type in unique_types
                }
                for _, row in ligand_rows.iterrows():
                    receptor_string = str(row["receptor.symbol"])
                    receptor_values = _get_receptor_vector(
                        receptor_string,
                        num_cells=num_cells,
                        expression_matrix=expression_matrix,
                        gene_to_col=gene_to_col,
                        gene_cache=gene_cache,
                        gene_cache_limit=gene_cache_limit,
                        receptor_cache=receptor_cache,
                        receptor_cache_limit=receptor_cache_limit,
                    )
                    if receptor_values is None or not np.any(receptor_values > 0):
                        continue
                    receptor_values_t = torch.as_tensor(
                        receptor_values, dtype=torch.float32, device=device
                    )
                    for receiver_type in unique_types:
                        receiver_local_positions = contact_type_positions[receiver_type]
                        n_contact_receiver = int(receiver_local_positions.size)
                        if n_contact_receiver == 0:
                            continue
                        receiver_local_positions_t = torch.as_tensor(
                            receiver_local_positions, dtype=torch.long, device=device
                        )
                        contact_receiver_global_indices = contact_indices[
                            receiver_local_positions
                        ]
                        contact_receiver_global_indices_t = (
                            contact_indices_t.index_select(
                                0, receiver_local_positions_t
                            )
                        )
                        receiver_receptor_values = receptor_values_t.index_select(
                            0, contact_receiver_global_indices_t
                        )
                        receiver_expr_pct = float(
                            (receiver_receptor_values > 0).float().mean().item()
                        )
                        if receiver_expr_pct < min_expr_pct:
                            continue
                        receiver_contact_exposure = contact_exposure_t.index_select(
                            0, contact_receiver_global_indices_t
                        )
                        receiver_scores = (
                            receiver_contact_exposure * receiver_receptor_values
                        )
                        observed_score = float(receiver_scores.mean().item())
                        if not np.isfinite(observed_score) or observed_score <= 1e-07:
                            continue
                        permutation_result = _receiver_position_permutation(
                            receiver_type=receiver_type,
                            receiver_indices=contact_receiver_global_indices,
                            exposure_values=receiver_contact_exposure,
                            receptor_values_t=receptor_values_t,
                            observed_score=observed_score,
                            permutation_groups=permutation_groups,
                            type_group_indices_t=type_group_indices_t,
                            n_permutations=n_permutations,
                            permutation_generator=permutation_generator,
                            device=device,
                        )
                        all_results.append(
                            {
                                "Interaction": f"{ligand_gene}_{receptor_string}",
                                "Domain": domain_label,
                                "Sender": sender_type,
                                "Receiver": receiver_type,
                                "Sender_Receiver": f"{sender_type}->{receiver_type}",
                                "Base_Score": observed_score,
                                "Mean_Contact_Ligand_Exposure": float(
                                    receiver_contact_exposure.mean().item()
                                ),
                                "Mean_Receptor_Expression": float(
                                    receiver_receptor_values.mean().item()
                                ),
                                "Receiver_Expr_Pct": receiver_expr_pct,
                                "N_Contact_Receiver": n_contact_receiver,
                                "N_Total_Receiver": int(
                                    type_indices_t[receiver_type].numel()
                                ),
                                **permutation_result,
                                "N_Permutations": n_permutations,
                            }
                        )
            progress.update(len(ligand_rows))
    result_df = pd.DataFrame(all_results)
    if result_df.empty:
        return pd.DataFrame(columns=output_columns)
    return _finalize_signaling_results(result_df, output_columns)


def _get_gene_vector(
    gene, expression_matrix, gene_to_col, gene_cache, gene_cache_limit
):
    if gene in gene_cache:
        gene_cache.move_to_end(gene)
        return gene_cache[gene]
    col_idx = gene_to_col.get(gene)
    if col_idx is None:
        return None
    gene_data = expression_matrix[:, col_idx]
    if scipy.sparse.issparse(gene_data):
        values = gene_data.toarray().reshape(-1)
    else:
        values = np.asarray(gene_data).reshape(-1)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.ascontiguousarray(values, dtype=np.float32)
    gene_cache[gene] = values
    gene_cache.move_to_end(gene)
    while len(gene_cache) > gene_cache_limit:
        gene_cache.popitem(last=False)
    return values


def _get_receptor_vector(
    receptor_string,
    num_cells,
    expression_matrix,
    gene_to_col,
    gene_cache,
    gene_cache_limit,
    receptor_cache,
    receptor_cache_limit,
):
    if receptor_string in receptor_cache:
        receptor_cache.move_to_end(receptor_string)
        return receptor_cache[receptor_string]
    # Missing subunits are omitted, matching the input scoring convention.
    subunits = [
        gene.strip()
        for gene in str(receptor_string).split(",")
        if gene.strip() in gene_to_col
    ]
    if not subunits:
        return None
    valid = np.ones(num_cells, dtype=bool)
    summed = np.zeros(num_cells, dtype=np.float32)
    for gene in subunits:
        values = _get_gene_vector(
            gene,
            expression_matrix=expression_matrix,
            gene_to_col=gene_to_col,
            gene_cache=gene_cache,
            gene_cache_limit=gene_cache_limit,
        )
        valid &= values > 0
        summed += values
    receptor_values = np.zeros(num_cells, dtype=np.float32)
    receptor_values[valid] = summed[valid] / float(len(subunits))
    result = receptor_values
    receptor_cache[receptor_string] = result
    receptor_cache.move_to_end(receptor_string)
    while len(receptor_cache) > receptor_cache_limit:
        receptor_cache.popitem(last=False)
    return result


def _receiver_position_permutation(
    receiver_type,
    receiver_indices,
    exposure_values,
    receptor_values_t,
    observed_score,
    permutation_groups,
    type_group_indices_t,
    n_permutations,
    permutation_generator,
    device,
):
    receiver_indices = np.asarray(receiver_indices, dtype=np.int64)
    n_receivers = int(receiver_indices.size)
    receiver_group_values = permutation_groups[receiver_indices]
    group_data = []
    for group_value in np.unique(receiver_group_values):
        receiver_positions = np.flatnonzero(
            receiver_group_values == group_value
        ).astype(np.int64, copy=False)
        pool_indices_t = type_group_indices_t[receiver_type].get(group_value)
        receiver_count = int(receiver_positions.size)
        pool_count = int(pool_indices_t.numel())
        receiver_positions_t = torch.as_tensor(
            receiver_positions, dtype=torch.long, device=device
        )
        group_exposure = exposure_values.index_select(0, receiver_positions_t)
        pool_values = receptor_values_t.index_select(0, pool_indices_t)
        group_data.append((group_exposure, pool_values, receiver_count, pool_count))
    # Exposure stays fixed while receptor scores move within receiver pools.
    null_scores = torch.empty(int(n_permutations), dtype=torch.float32, device=device)
    denominator = float(n_receivers)
    for permutation_id in range(int(n_permutations)):
        weighted_sum = torch.zeros((), dtype=torch.float32, device=device)
        for group_exposure, pool_values, receiver_count, pool_count in group_data:
            permutation = torch.randperm(
                pool_count, generator=permutation_generator, device=device
            )
            sampled_values = pool_values.index_select(0, permutation[:receiver_count])
            weighted_sum += torch.sum(group_exposure * sampled_values)
        null_scores[permutation_id] = weighted_sum / denominator
    null_mean_t = torch.mean(null_scores)
    null_std_t = torch.std(null_scores, unbiased=False)
    observed_t = torch.as_tensor(observed_score, dtype=torch.float32, device=device)
    p_enrichment = (1.0 + torch.sum(null_scores >= observed_t).item()) / (
        int(n_permutations) + 1.0
    )
    p_depletion = (1.0 + torch.sum(null_scores <= observed_t).item()) / (
        int(n_permutations) + 1.0
    )
    p_two_sided = min(1.0, 2.0 * min(p_enrichment, p_depletion))
    null_mean = float(null_mean_t.item())
    null_std = float(null_std_t.item())
    if np.isfinite(null_std) and null_std > 1e-12:
        z_score = (float(observed_score) - null_mean) / null_std
    else:
        z_score = np.nan
    if float(observed_score) > null_mean + 1e-12:
        direction = "enrichment"
    elif float(observed_score) < null_mean - 1e-12:
        direction = "depletion"
    else:
        direction = "neutral"
    return {
        "Permutation_Mean": null_mean,
        "Permutation_STD": null_std,
        "Z_score": z_score,
        "P_enrichment": float(p_enrichment),
        "P_depletion": float(p_depletion),
        "P_value": float(p_enrichment),
        "P_value_Two_Sided": float(p_two_sided),
        "Direction": direction,
    }


def _iter_radius_edges(
    source_indices,
    coords_cpu,
    spatial_tree,
    effective_range,
    adaptive_chunk_size,
    query_workers,
):
    source_indices = np.asarray(source_indices, dtype=np.int64)
    for start in range(0, len(source_indices), adaptive_chunk_size):
        end = min(start + adaptive_chunk_size, len(source_indices))
        source_chunk = source_indices[start:end]
        neighbor_lists = spatial_tree.query_ball_point(
            coords_cpu[source_chunk],
            r=effective_range,
            workers=query_workers,
            return_sorted=False,
        )
        lengths = np.fromiter(
            (len(indices) for indices in neighbor_lists),
            dtype=np.int64,
            count=len(source_chunk),
        )
        if int(lengths.sum()) == 0:
            continue
        source_local = np.repeat(np.arange(len(source_chunk), dtype=np.int64), lengths)
        target_global = np.concatenate(neighbor_lists).astype(np.int64, copy=False)
        yield (source_chunk, source_local, target_global)


def _compute_ligand_field(
    source_indices,
    source_values,
    num_cells,
    device,
    coords_device,
    sigma_sq,
    coords_cpu,
    spatial_tree,
    effective_range,
    adaptive_chunk_size,
    query_workers,
):
    field_value = torch.zeros(num_cells, dtype=torch.float32, device=device)
    for source_chunk, source_local, target_global in _iter_radius_edges(
        source_indices,
        coords_cpu=coords_cpu,
        spatial_tree=spatial_tree,
        effective_range=effective_range,
        adaptive_chunk_size=adaptive_chunk_size,
        query_workers=query_workers,
    ):
        source_chunk_t = torch.as_tensor(source_chunk, dtype=torch.long, device=device)
        source_local_t = torch.as_tensor(source_local, dtype=torch.long, device=device)
        target_global_t = torch.as_tensor(
            target_global, dtype=torch.long, device=device
        )
        chunk_values_t = torch.as_tensor(
            source_values[source_chunk], dtype=torch.float32, device=device
        )
        edge_values = chunk_values_t.index_select(0, source_local_t)
        source_coords = coords_device.index_select(0, source_chunk_t).index_select(
            0, source_local_t
        )
        target_coords = coords_device.index_select(0, target_global_t)
        diff = target_coords - source_coords
        distance_sq = torch.sum(diff * diff, dim=1)
        weights = torch.exp(-distance_sq / (2.0 * sigma_sq))
        weighted_values = weights * edge_values
        field_value.index_add_(0, target_global_t, weighted_values)
    field_indices_t = torch.nonzero(field_value > 0, as_tuple=False).flatten()
    if field_indices_t.numel() == 0:
        return None
    return (
        field_indices_t.detach().cpu().numpy(),
        field_indices_t,
        field_value.index_select(0, field_indices_t),
    )


def _finalize_signaling_results(result_df, output_columns):
    p_values = result_df["P_value"].to_numpy(dtype=float)
    order = np.argsort(p_values)
    ranked_p = p_values[order]
    m = len(ranked_p)
    ranked_fdr = ranked_p * m / np.arange(1, m + 1)
    ranked_fdr = np.minimum.accumulate(ranked_fdr[::-1])[::-1]
    ranked_fdr = np.clip(ranked_fdr, 0.0, 1.0)
    fdr = np.empty(m, dtype=float)
    fdr[order] = ranked_fdr
    result_df["FDR"] = fdr
    result_df = result_df.sort_values(
        ["FDR", "P_value", "Base_Score"], ascending=[True, True, False]
    ).reset_index(drop=True)
    return result_df.reindex(columns=output_columns)


def _contact_mm(source_matrix_t, sparse_contact_graph, adj_t):
    if sparse_contact_graph:
        return torch.sparse.mm(adj_t, source_matrix_t)
    return torch.matmul(adj_t, source_matrix_t)
