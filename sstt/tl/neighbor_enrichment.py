from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from tqdm.auto import tqdm


__all__ = [
    "build_candidate_multihop_neighborhoods",
    "filter_components_by_closure",
    "filter_components_manually",
    "build_component_distance_data",
    "run_multihop_enrichment",
    "compute_multihop_gene_gradient",
]


def build_candidate_multihop_neighborhoods(
    adata,
    observed_celltypes: Sequence[str],
    celltype_key="celltype",
    connectivity_key="sstt_neighbor_network",
    spatial_key="spatial",
    max_hop=10,
    min_component_size=10,
    margin=50.0,
    sample_key=None,
    barrier_mode="current_component",
    bridge_components=True,
    closure_bridges=True,
    closure_min_barrier_contacts=2,
    show_progress=True,
):
    membership, component_summary = _build_all_candidate_neighborhoods(
        adata=adata,
        observed_celltypes=observed_celltypes,
        celltype_key=celltype_key,
        connectivity_key=connectivity_key,
        spatial_key=spatial_key,
        max_hop=max_hop,
        min_component_size=min_component_size,
        min_closure_index=0.0,
        margin=margin,
        sample_key=sample_key,
        barrier_mode=barrier_mode,
        bridge_components=bridge_components,
        closure_bridges=closure_bridges,
        closure_min_barrier_contacts=closure_min_barrier_contacts,
        show_progress=show_progress,
    )
    component_summary = component_summary.copy()
    if not component_summary.empty:
        component_summary["min_component_size"] = int(min_component_size)
        component_summary["closure_filter_applied"] = False
        component_summary["manual_filter_applied"] = False
        component_summary["passes_manual_filter"] = True
        component_summary["used_in_analysis"] = True
        component_summary["selection_reason"] = "candidate"
    return membership, component_summary


def filter_components_by_closure(
    candidate_membership,
    candidate_component_summary,
    min_closure_index: float | None,
):
    required_membership = {
        "observed_celltype",
        "component_id",
        "cell_index",
        "cell_id",
        "hop",
        "signed_hop",
        "side",
        "hop_label",
    }
    missing = required_membership.difference(candidate_membership.columns)
    if missing:
        raise KeyError(f"candidate_membership is missing columns: {sorted(missing)}")
    required_summary = {
        "observed_celltype",
        "component_id",
        "sample",
        "closure_index",
    }
    missing = required_summary.difference(candidate_component_summary.columns)
    if missing:
        raise KeyError(
            "candidate_component_summary is missing columns: " f"{sorted(missing)}"
        )
    if min_closure_index is not None and not 0.0 <= float(min_closure_index) <= 1.0:
        raise ValueError("min_closure_index must be between 0 and 1 or None.")

    summary = candidate_component_summary.copy()
    summary["closure_filter_applied"] = min_closure_index is not None
    summary["min_closure_index"] = (
        np.nan if min_closure_index is None else float(min_closure_index)
    )
    if min_closure_index is None:
        summary["passes_closure_filter"] = True
    else:
        closure_values = summary["closure_index"].to_numpy(dtype=float)
        summary["passes_closure_filter"] = np.isfinite(closure_values) & (
            closure_values >= float(min_closure_index)
        )

    summary["manual_filter_applied"] = False
    summary["manual_selection_mode"] = "none"
    summary["passes_manual_filter"] = True
    summary["used_in_analysis"] = summary["passes_closure_filter"].astype(bool)
    summary["selection_reason"] = np.where(
        summary["used_in_analysis"],
        "retained_after_closure",
        "below_closure_threshold",
    )

    retained_keys = summary.loc[
        summary["used_in_analysis"],
        ["observed_celltype", "component_id"],
    ].drop_duplicates()
    closure_membership = candidate_membership.merge(
        retained_keys,
        on=["observed_celltype", "component_id"],
        how="inner",
        sort=False,
    )
    closure_membership = closure_membership.sort_values(
        ["observed_celltype", "component_id", "signed_hop", "cell_index"]
    ).reset_index(drop=True)
    summary = summary.sort_values(
        ["observed_celltype", "sample", "component_id"]
    ).reset_index(drop=True)
    return closure_membership, summary


def filter_components_manually(
    closure_membership,
    closure_component_summary,
    include_component_ids: Mapping[str, Sequence[int]] | None = None,
    exclude_component_ids: Mapping[str, Sequence[int]] | None = None,
):
    required_membership = {
        "observed_celltype",
        "component_id",
        "cell_index",
        "cell_id",
    }
    missing = required_membership.difference(closure_membership.columns)
    if missing:
        raise KeyError(f"closure_membership is missing columns: {sorted(missing)}")
    required_summary = {
        "observed_celltype",
        "component_id",
        "sample",
        "passes_closure_filter",
    }
    missing = required_summary.difference(closure_component_summary.columns)
    if missing:
        raise KeyError(
            "closure_component_summary is missing columns: " f"{sorted(missing)}"
        )

    for name, mapping in (
        ("include_component_ids", include_component_ids),
        ("exclude_component_ids", exclude_component_ids),
    ):
        if mapping is not None and not isinstance(mapping, Mapping):
            raise TypeError(f"{name} must map cell types to component IDs or be None.")

    include_component_ids = include_component_ids or {}
    exclude_component_ids = exclude_component_ids or {}
    include_celltypes = {str(key) for key in include_component_ids}
    exclude_celltypes = {str(key) for key in exclude_component_ids}
    conflicting_celltypes = sorted(include_celltypes & exclude_celltypes)
    if conflicting_celltypes:
        raise ValueError(
            "The same cell type cannot use include and exclude selection: "
            f"{conflicting_celltypes}"
        )

    summary = closure_component_summary.copy()
    available_celltypes = set(summary["observed_celltype"].astype(str))
    unknown_celltypes = sorted(
        (include_celltypes | exclude_celltypes).difference(available_celltypes)
    )
    if unknown_celltypes:
        raise ValueError(
            f"Manual-selection cell types were not found: {unknown_celltypes}"
        )

    summary["manual_filter_applied"] = False
    summary["manual_selection_mode"] = "none"
    summary["passes_manual_filter"] = True

    def apply_mapping(mapping, mode):
        for observed_celltype, component_ids in mapping.items():
            observed_celltype = str(observed_celltype)
            selected_ids = {int(value) for value in component_ids}
            celltype_mask = (
                summary["observed_celltype"].astype(str) == observed_celltype
            )
            closure_mask = celltype_mask & summary["passes_closure_filter"].astype(bool)
            available_ids = set(summary.loc[closure_mask, "component_id"].astype(int))
            unknown_ids = sorted(selected_ids.difference(available_ids))
            if unknown_ids:
                raise ValueError(
                    f"Component IDs are not available after closure filtering "
                    f"for {observed_celltype!r}: {unknown_ids}"
                )

            listed = summary["component_id"].astype(int).isin(selected_ids)
            summary.loc[celltype_mask, "manual_filter_applied"] = True
            summary.loc[celltype_mask, "manual_selection_mode"] = mode
            if mode == "include":
                summary.loc[celltype_mask, "passes_manual_filter"] = listed[
                    celltype_mask
                ]
            else:
                summary.loc[celltype_mask, "passes_manual_filter"] = ~listed[
                    celltype_mask
                ]

    apply_mapping(include_component_ids, "include")
    apply_mapping(exclude_component_ids, "exclude")

    closure_pass = summary["passes_closure_filter"].astype(bool)
    manual_pass = summary["passes_manual_filter"].astype(bool)
    summary["used_in_analysis"] = closure_pass & manual_pass

    reasons = np.full(len(summary), "retained", dtype=object)
    reasons[~closure_pass.to_numpy()] = "below_closure_threshold"
    include_rejected = (
        closure_pass.to_numpy()
        & ~manual_pass.to_numpy()
        & (summary["manual_selection_mode"].to_numpy() == "include")
    )
    exclude_rejected = (
        closure_pass.to_numpy()
        & ~manual_pass.to_numpy()
        & (summary["manual_selection_mode"].to_numpy() == "exclude")
    )
    reasons[include_rejected] = "not_manually_included"
    reasons[exclude_rejected] = "manually_excluded"
    summary["selection_reason"] = reasons

    retained_keys = summary.loc[
        summary["used_in_analysis"],
        ["observed_celltype", "component_id"],
    ].drop_duplicates()
    selected_membership = closure_membership.merge(
        retained_keys,
        on=["observed_celltype", "component_id"],
        how="inner",
        sort=False,
    )
    selected_membership = selected_membership.sort_values(
        ["observed_celltype", "component_id", "signed_hop", "cell_index"]
    ).reset_index(drop=True)
    summary = summary.sort_values(
        ["observed_celltype", "sample", "component_id"]
    ).reset_index(drop=True)
    return selected_membership, summary


def build_component_distance_data(
    adata,
    membership,
    celltype_key="celltype",
    spatial_key="spatial",
    observed_celltypes: Sequence[str] | None = None,
    neighbor_celltypes: Sequence[str] | None = None,
    max_hop: int | None = None,
    distance_reference="target",
):
    required_columns = {
        "observed_celltype",
        "component_id",
        "sample",
        "cell_index",
        "cell_id",
        "signed_hop",
        "hop",
        "side",
        "hop_label",
        "is_target",
        "is_barrier",
    }
    missing = required_columns.difference(membership.columns)
    if missing:
        raise KeyError(f"`membership` is missing columns: {sorted(missing)}")

    if celltype_key not in adata.obs:
        raise KeyError(f"`{celltype_key}` was not found in `adata.obs`.")
    if spatial_key not in adata.obsm:
        raise KeyError(f"`{spatial_key}` was not found in `adata.obsm`.")
    if adata.obs[celltype_key].isna().any():
        raise ValueError(f"`adata.obs[{celltype_key!r}]` contains missing values.")

    if distance_reference not in {"target", "barrier"}:
        raise ValueError("`distance_reference` must be either 'target' or 'barrier'.")

    if max_hop is not None:
        if not isinstance(max_hop, (int, np.integer)) or int(max_hop) < 1:
            raise ValueError("`max_hop` must be a positive integer or None.")
        max_hop = int(max_hop)

    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"`adata.obsm[{spatial_key!r}]` must have shape (n_obs, >=2).")
    coords = coords[:, :2]
    if not np.isfinite(coords).all():
        raise ValueError(
            f"`adata.obsm[{spatial_key!r}]` contains non-finite coordinates."
        )

    celltype_labels = adata.obs[celltype_key].astype(str).to_numpy()
    obs_names = adata.obs_names.astype(str).to_numpy()

    selected = membership.loc[:, sorted(required_columns)].copy()
    selected["observed_celltype"] = selected["observed_celltype"].astype(str)
    selected["sample"] = selected["sample"].astype(str)
    selected["component_id"] = selected["component_id"].astype(np.int64)
    selected["cell_index"] = selected["cell_index"].astype(np.int64)
    selected["signed_hop"] = selected["signed_hop"].astype(np.int16)
    selected["hop"] = selected["hop"].astype(np.int16)
    selected["side"] = selected["side"].astype(str)
    selected["hop_label"] = selected["hop_label"].astype(str)
    selected["is_target"] = selected["is_target"].astype(bool)
    selected["is_barrier"] = selected["is_barrier"].astype(bool)

    cell_indices = selected["cell_index"].to_numpy(dtype=np.int64)
    invalid_indices = (cell_indices < 0) | (cell_indices >= adata.n_obs)
    if invalid_indices.any():
        raise IndexError("`membership.cell_index` contains values outside `adata`.")

    if observed_celltypes is not None:
        observed_celltypes = _ordered_unique(observed_celltypes)
        available_observed = set(selected["observed_celltype"].unique())
        missing_observed = [
            value for value in observed_celltypes if value not in available_observed
        ]
        if missing_observed:
            raise ValueError(
                "Observed cell types were not found in `membership`: "
                f"{missing_observed}"
            )
        selected = selected[selected["observed_celltype"].isin(observed_celltypes)]

    if neighbor_celltypes is not None:
        neighbor_celltypes = _ordered_unique(neighbor_celltypes)
        available_neighbors = set(celltype_labels)
        missing_neighbors = [
            value for value in neighbor_celltypes if value not in available_neighbors
        ]
        if missing_neighbors:
            raise ValueError(
                "Neighbor cell types were not found in "
                f"`adata.obs[{celltype_key!r}]`: {missing_neighbors}"
            )
        neighbor_celltypes = np.asarray(neighbor_celltypes, dtype=object)

    output_columns = [
        "observed_celltype",
        "component_id",
        "sample",
        "component_uid",
        "cell_index",
        "cell_id",
        "neighbor_celltype",
        "signed_hop",
        "hop",
        "side",
        "hop_label",
        "physical_distance",
        "signed_physical_distance",
        "distance_reference",
        "nearest_reference_cell_index",
        "nearest_reference_cell_id",
        "component_target_n",
        "component_barrier_n",
    ]

    if selected.empty:
        return pd.DataFrame(columns=output_columns)

    distance_blocks = []
    grouping_columns = ["observed_celltype", "sample", "component_id"]

    for (observed_celltype, sample, component_id), block in selected.groupby(
        grouping_columns,
        observed=True,
        sort=False,
    ):
        # Preserve distinct inward/outward relationships when the two global
        # expansions meet at the same cell. Remove only exact role duplicates.
        block = block.drop_duplicates(
            ["cell_index", "side", "hop", "hop_label"],
            keep="first",
        )

        if distance_reference == "target":
            reference_rows = block[block["is_target"]]
            reference_label = "nearest_target_cell_center"
        else:
            reference_rows = block[block["is_barrier"]]
            reference_label = "nearest_barrier_cell_center"

        # Retain both directions, while Target/Bridge/ClosureBridge remain
        # excluded as distance observations.
        neighbor_rows = block[block["side"].isin(["In", "Out"])].copy()
        if max_hop is not None:
            neighbor_rows = neighbor_rows[neighbor_rows["hop"] <= max_hop]

        if reference_rows.empty or neighbor_rows.empty:
            continue

        neighbor_indices = neighbor_rows["cell_index"].to_numpy(dtype=np.int64)
        neighbor_labels = celltype_labels[neighbor_indices]

        if neighbor_celltypes is not None:
            keep = np.isin(neighbor_labels, neighbor_celltypes)
            if not keep.any():
                continue

            keep_positions = np.flatnonzero(keep)
            neighbor_rows = neighbor_rows.iloc[keep_positions].copy()
            neighbor_indices = neighbor_indices[keep]
            neighbor_labels = neighbor_labels[keep]

        reference_indices = reference_rows["cell_index"].to_numpy(dtype=np.int64)
        reference_tree = cKDTree(coords[reference_indices])
        distances, nearest_positions = reference_tree.query(
            coords[neighbor_indices],
            k=1,
        )

        distances = np.asarray(distances, dtype=np.float64)
        nearest_positions = np.asarray(nearest_positions, dtype=np.int64)
        nearest_indices = reference_indices[nearest_positions]

        side_values = neighbor_rows["side"].astype(str).to_numpy()
        distance_sign = np.where(side_values == "In", -1.0, 1.0)

        component_uid = f"{sample}::{observed_celltype}::{int(component_id)}"

        distance_blocks.append(
            pd.DataFrame(
                {
                    "observed_celltype": str(observed_celltype),
                    "component_id": int(component_id),
                    "sample": str(sample),
                    "component_uid": component_uid,
                    "cell_index": neighbor_indices,
                    "cell_id": obs_names[neighbor_indices],
                    "neighbor_celltype": neighbor_labels,
                    "signed_hop": neighbor_rows["signed_hop"].to_numpy(dtype=np.int16),
                    "hop": neighbor_rows["hop"].to_numpy(dtype=np.int16),
                    "side": side_values,
                    "hop_label": neighbor_rows["hop_label"].astype(str).to_numpy(),
                    "physical_distance": distances,
                    "signed_physical_distance": distances * distance_sign,
                    "distance_reference": reference_label,
                    "nearest_reference_cell_index": nearest_indices,
                    "nearest_reference_cell_id": obs_names[nearest_indices],
                    "component_target_n": int(block["is_target"].sum()),
                    "component_barrier_n": int(block["is_barrier"].sum()),
                }
            )
        )

    if not distance_blocks:
        return pd.DataFrame(columns=output_columns)

    distance_data = pd.concat(distance_blocks, ignore_index=True)
    distance_data = distance_data.loc[:, output_columns]
    distance_data = distance_data.sort_values(
        [
            "observed_celltype",
            "sample",
            "component_id",
            "signed_hop",
            "cell_index",
        ],
        kind="stable",
    ).reset_index(drop=True)

    return distance_data


def run_multihop_enrichment(
    adata,
    membership,
    component_summary,
    observed_celltypes: Sequence[str],
    celltype_key="celltype",
    max_hop=10,
    randomized_celltypes: Sequence[str] | None = None,
    component_mode="pool",
    n_permutations=5000,
    sample_key=None,
    pseudocount=0.5,
    device="auto",
    permutation_batch_size=128,
    max_batch_elements=5_000_000,
    random_state=0,
    key_suffix="_hop_group",
    show_progress=True,
    use_negative_hops=True,
    overlap_tie_policy="first",
):
    required_membership = {
        "observed_celltype",
        "component_id",
        "sample",
        "cell_index",
        "cell_id",
        "signed_hop",
        "hop",
        "side",
        "hop_label",
        "is_target",
        "is_barrier",
    }
    missing_membership = required_membership.difference(membership.columns)
    if missing_membership:
        raise KeyError(f"`membership` is missing columns: {sorted(missing_membership)}")
    if celltype_key not in adata.obs:
        raise KeyError(f"`{celltype_key}` was not found in `adata.obs`.")
    if max_hop < 1:
        raise ValueError("`max_hop` must be at least 1.")
    if component_mode not in {"pool", "local"}:
        raise ValueError("`component_mode` must be either 'pool' or 'local'.")
    if n_permutations < 1:
        raise ValueError("`n_permutations` must be at least 1.")
    if pseudocount <= 0:
        raise ValueError("`pseudocount` must be greater than 0.")
    if permutation_batch_size < 1:
        raise ValueError("`permutation_batch_size` must be at least 1.")
    if max_batch_elements < 1:
        raise ValueError("`max_batch_elements` must be at least 1.")
    if not isinstance(use_negative_hops, (bool, np.bool_)):
        raise TypeError("`use_negative_hops` must be a boolean.")
    if overlap_tie_policy not in {"ambiguous", "first", "in", "out", "error"}:
        raise ValueError(
            "`overlap_tie_policy` must be 'ambiguous', 'first', 'in', "
            "'out', or 'error'."
        )
    if adata.obs[celltype_key].isna().any():
        raise ValueError(f"`adata.obs[{celltype_key!r}]` contains missing values.")

    observed_celltypes = _ordered_unique(observed_celltypes)
    if not observed_celltypes:
        raise ValueError("`observed_celltypes` cannot be empty.")

    celltype_labels = adata.obs[celltype_key].astype(str).to_numpy()
    if isinstance(adata.obs[celltype_key].dtype, pd.CategoricalDtype):
        all_celltypes = [str(value) for value in adata.obs[celltype_key].cat.categories]
    else:
        all_celltypes = _ordered_unique(celltype_labels)

    missing_observed = [
        celltype for celltype in observed_celltypes if celltype not in all_celltypes
    ]
    if missing_observed:
        raise ValueError(f"Observed cell types were not found: {missing_observed}")

    if randomized_celltypes is None:
        randomized_celltypes = all_celltypes
    else:
        randomized_celltypes = _ordered_unique(randomized_celltypes)
        missing_randomized = [
            celltype
            for celltype in randomized_celltypes
            if celltype not in all_celltypes
        ]
        if missing_randomized:
            raise ValueError(
                f"Randomized cell types were not found: {missing_randomized}"
            )

    resolved_device = _resolve_device(device)
    celltype_codes = pd.Categorical(
        celltype_labels, categories=all_celltypes
    ).codes.astype(np.int32)
    if sample_key is None:
        sample_codes = np.zeros(adata.n_obs, dtype=np.int32)
    else:
        if sample_key not in adata.obs:
            raise KeyError(f"`{sample_key}` was not found in `adata.obs`.")
        if adata.obs[sample_key].isna().any():
            raise ValueError(f"`adata.obs[{sample_key!r}]` contains missing values.")
        sample_codes = pd.Categorical(adata.obs[sample_key]).codes.astype(np.int32)

    randomized_indices = np.array(
        [all_celltypes.index(x) for x in randomized_celltypes], dtype=np.int64
    )

    adata.uns["multihop_component_summary"] = component_summary.copy()
    if component_summary.empty:
        component_filter_counts = pd.DataFrame(
            columns=[
                "observed_celltype",
                "sample",
                "components_after_size",
                "components_after_closure",
                "components_used",
                "components_filtered_by_closure",
                "components_filtered_manually",
            ]
        )
    else:
        required_summary = {
            "observed_celltype",
            "sample",
            "component_id",
            "used_in_analysis",
        }
        missing_summary = required_summary.difference(component_summary.columns)
        if missing_summary:
            raise KeyError(
                f"`component_summary` is missing columns: {sorted(missing_summary)}"
            )
        count_source = component_summary.copy()
        count_source["_passes_closure"] = (
            count_source["passes_closure_filter"].astype(bool)
            if "passes_closure_filter" in count_source
            else True
        )
        component_filter_counts = (
            count_source.groupby(
                ["observed_celltype", "sample"],
                observed=True,
                sort=False,
            )
            .agg(
                components_after_size=("component_id", "size"),
                components_after_closure=("_passes_closure", "sum"),
                components_used=("used_in_analysis", "sum"),
            )
            .reset_index()
        )
        component_filter_counts["components_after_size"] = component_filter_counts[
            "components_after_size"
        ].astype(np.int64)
        component_filter_counts["components_used"] = component_filter_counts[
            "components_used"
        ].astype(np.int64)
        component_filter_counts["components_after_closure"] = component_filter_counts[
            "components_after_closure"
        ].astype(np.int64)
        component_filter_counts["components_filtered_by_closure"] = (
            component_filter_counts["components_after_size"]
            - component_filter_counts["components_after_closure"]
        )
        component_filter_counts["components_filtered_manually"] = (
            component_filter_counts["components_after_closure"]
            - component_filter_counts["components_used"]
        )
    adata.uns["multihop_component_filter_counts"] = component_filter_counts.copy()
    adata.uns["multihop_overlap_tie_policy"] = str(overlap_tie_policy)

    hop_values = _statistical_hop_values(
        max_hop=max_hop, use_negative_hops=use_negative_hops
    )
    result_blocks = []
    iterator = observed_celltypes
    if show_progress:
        iterator = tqdm(observed_celltypes, desc="Enrichment tests")

    for observed_index, observed_celltype in enumerate(iterator):
        selected = membership[
            membership["observed_celltype"].astype(str) == str(observed_celltype)
        ]
        if selected.empty:
            aggregated = pd.DataFrame(
                columns=[
                    "cell_index",
                    "cell_id",
                    "hop_label",
                    "signed_hop",
                    "hop",
                    "side",
                    "component_id",
                    "n_nearest_components",
                    "conflict",
                ]
            )
        else:
            aggregated = _aggregate_signed_celltype_membership(
                membership=membership,
                observed_celltype=observed_celltype,
                tie_policy=overlap_tie_policy,
                use_negative_hops=use_negative_hops,
            )

        _write_aggregated_hop_group(
            adata=adata,
            aggregated=aggregated,
            column=f"{observed_celltype}{key_suffix}",
            max_hop=max_hop,
            use_negative_hops=use_negative_hops,
        )

        global_hop_codes = np.zeros(adata.n_obs, dtype=np.int16)
        local_component_codes = np.full(adata.n_obs, -1, dtype=np.int32)
        if not aggregated.empty:
            cell_indices = aggregated["cell_index"].to_numpy(dtype=np.int64)
            if use_negative_hops:
                statistical_hops = aggregated["signed_hop"].to_numpy(dtype=np.int16)
            else:
                statistical_hops = aggregated["hop"].to_numpy(dtype=np.int16)
            global_hop_codes[cell_indices] = statistical_hops
            local_component_codes[cell_indices] = aggregated["component_id"].to_numpy(
                dtype=np.int32
            )

        stratum_codes = (
            sample_codes if component_mode == "pool" else local_component_codes
        )
        mode_name = "signed" if use_negative_hops else "unsigned"
        enrichment, pvalues = _multihop_enrichment_test(
            hop_codes=global_hop_codes,
            celltype_codes=celltype_codes,
            stratum_codes=stratum_codes,
            max_hop=max_hop,
            n_celltypes=len(all_celltypes),
            n_permutations=n_permutations,
            pseudocount=pseudocount,
            device=resolved_device,
            permutation_batch_size=permutation_batch_size,
            max_batch_elements=max_batch_elements,
            random_state=random_state + observed_index * 100_003,
            show_progress=show_progress,
            description=(
                f"{observed_celltype} {component_mode} {mode_name} "
                f"permutations ({resolved_device.type})"
            ),
            use_negative_hops=use_negative_hops,
        )

        selected_enrichment = enrichment[:, randomized_indices].T.reshape(-1)
        selected_pvalues = pvalues[:, randomized_indices].T.reshape(-1)
        n_rows = len(randomized_celltypes) * len(hop_values)
        result_blocks.append(
            pd.DataFrame(
                {
                    "observed_celltype": np.repeat(observed_celltype, n_rows),
                    "randomized_celltype": np.repeat(
                        randomized_celltypes, len(hop_values)
                    ),
                    "hop": np.tile(hop_values, len(randomized_celltypes)),
                    "enrichment": selected_enrichment,
                    "pvalue": selected_pvalues,
                    "component_mode": np.repeat(component_mode, n_rows),
                }
            )
        )

    result = pd.concat(result_blocks, ignore_index=True)
    result["qvalue"] = _bh_adjust(result["pvalue"].to_numpy())
    result["observed_celltype"] = pd.Categorical(
        result["observed_celltype"], categories=observed_celltypes, ordered=True
    )
    result["randomized_celltype"] = pd.Categorical(
        result["randomized_celltype"], categories=randomized_celltypes, ordered=True
    )
    result = result.sort_values(["observed_celltype", "randomized_celltype", "hop"])
    result = result.loc[
        :,
        [
            "observed_celltype",
            "randomized_celltype",
            "hop",
            "enrichment",
            "pvalue",
            "qvalue",
            "component_mode",
        ],
    ].reset_index(drop=True)

    return result


def compute_multihop_gene_gradient(
    adata,
    membership,
    genes,
    observed_celltypes=None,
    neighbor_celltypes=None,
    celltype_key="celltype",
    direction="out",
    include_target=True,
    max_hop=None,
    layer=None,
    positive_threshold=0.0,
    stratify_celltype=False,
):
    if isinstance(genes, str):
        genes = [genes]
    else:
        genes = list(genes)
    genes = list(dict.fromkeys((str(gene) for gene in genes)))
    if len(genes) == 0:
        raise ValueError("`genes` cannot be empty.")
    if isinstance(observed_celltypes, str):
        observed_celltypes = [observed_celltypes]
    if isinstance(neighbor_celltypes, str):
        neighbor_celltypes = [neighbor_celltypes]
    if direction not in {"out", "in", "both"}:
        raise ValueError("`direction` must be 'out', 'in', or 'both'.")
    if max_hop is not None:
        max_hop = int(max_hop)
        if max_hop < 1:
            raise ValueError("`max_hop` must be >= 1.")
    if celltype_key not in adata.obs:
        raise KeyError(f"`adata.obs[{celltype_key!r}]` was not found.")
    required_columns = {
        "observed_celltype",
        "component_id",
        "sample",
        "cell_index",
        "cell_id",
        "signed_hop",
        "hop",
        "side",
        "hop_label",
        "is_target",
    }
    missing_columns = required_columns - set(membership.columns)
    if missing_columns:
        raise KeyError(f"`membership` is missing columns: {sorted(missing_columns)}")
    available_genes = set(adata.var_names.astype(str))
    missing_genes = [gene for gene in genes if gene not in available_genes]
    if missing_genes:
        raise KeyError(f"Genes were not found in adata.var_names: {missing_genes}")
    selected = membership.copy()
    selected["observed_celltype"] = selected["observed_celltype"].astype(str)
    selected["sample"] = selected["sample"].astype(str)
    selected["side"] = selected["side"].astype(str)
    selected["hop_label"] = selected["hop_label"].astype(str)
    selected["cell_index"] = selected["cell_index"].astype(np.int64)
    selected["component_id"] = selected["component_id"].astype(np.int64)
    selected["hop"] = selected["hop"].astype(np.int16)
    selected["signed_hop"] = selected["signed_hop"].astype(np.int16)
    if observed_celltypes is not None:
        observed_celltypes = [str(x) for x in observed_celltypes]
        selected = selected.loc[
            selected["observed_celltype"].isin(observed_celltypes)
        ].copy()
    if direction == "out":
        valid_sides = ["Out"]
    elif direction == "in":
        valid_sides = ["In"]
    else:
        valid_sides = ["In", "Out"]
    if include_target:
        valid_sides.append("Target")
    selected = selected.loc[selected["side"].isin(valid_sides)].copy()
    if max_hop is not None:
        selected = selected.loc[
            (selected["side"] == "Target") | (selected["hop"] <= max_hop)
        ].copy()
    if selected.empty:
        raise ValueError("No membership rows remain after filtering.")
    selected = selected.drop_duplicates(
        subset=["observed_celltype", "sample", "component_id", "cell_index"],
        keep="first",
    ).copy()
    cell_indices = selected["cell_index"].to_numpy(dtype=np.int64)
    invalid = (cell_indices < 0) | (cell_indices >= adata.n_obs)
    if invalid.any():
        raise IndexError(
            "`membership.cell_index` contains indices outside the AnnData object."
        )
    adata_celltypes = adata.obs[celltype_key].astype(str).to_numpy()
    selected["celltype"] = adata_celltypes[
        selected["cell_index"].to_numpy(dtype=np.int64)
    ]
    if neighbor_celltypes is not None:
        neighbor_celltypes = [str(x) for x in neighbor_celltypes]
        selected = selected.loc[selected["celltype"].isin(neighbor_celltypes)].copy()
    if selected.empty:
        raise ValueError("No cells remain after cell-type filtering.")
    selected["component_uid"] = (
        selected["sample"].astype(str)
        + "::"
        + selected["observed_celltype"].astype(str)
        + "::"
        + selected["component_id"].astype(str)
    )
    unique_cell_indices = np.sort(selected["cell_index"].unique())
    if layer is None:
        expression_matrix = adata[unique_cell_indices, genes].X
    else:
        if layer not in adata.layers:
            raise KeyError(f"`adata.layers[{layer!r}]` was not found.")
        expression_matrix = adata[unique_cell_indices, genes].layers[layer]
    if sp.issparse(expression_matrix):
        expression_matrix = expression_matrix.toarray()
    expression_matrix = np.asarray(expression_matrix, dtype=np.float64)
    index_lookup = pd.Series(
        np.arange(len(unique_cell_indices), dtype=np.int64), index=unique_cell_indices
    )
    expression_row = selected["cell_index"].map(index_lookup).to_numpy(dtype=np.int64)
    membership_expression = expression_matrix[expression_row, :]
    n_rows = len(selected)
    n_genes = len(genes)
    metadata_columns = [
        "observed_celltype",
        "sample",
        "component_id",
        "component_uid",
        "cell_index",
        "cell_id",
        "celltype",
        "signed_hop",
        "hop",
        "side",
        "hop_label",
    ]
    cell_level = (
        selected[metadata_columns]
        .loc[selected.index.repeat(n_genes)]
        .reset_index(drop=True)
    )
    cell_level["gene"] = np.tile(genes, n_rows)
    cell_level["expression"] = membership_expression.reshape(-1)
    cell_level["positive"] = cell_level["expression"] > positive_threshold
    cell_level["positive_expression"] = cell_level["expression"].where(
        cell_level["positive"]
    )
    component_group_columns = [
        "observed_celltype",
        "sample",
        "component_id",
        "component_uid",
        "signed_hop",
        "hop",
        "side",
        "hop_label",
    ]
    if stratify_celltype:
        component_group_columns.append("celltype")
    component_group_columns.append("gene")
    component_stats = (
        cell_level.groupby(component_group_columns, observed=True, sort=False)
        .agg(
            n_cells=("cell_id", "nunique"),
            mean_expression=("expression", "mean"),
            positive_fraction=("positive", "mean"),
            positive_mean_expression=("positive_expression", "mean"),
        )
        .reset_index()
    )
    summary_group_columns = [
        "observed_celltype",
        "signed_hop",
        "hop",
        "side",
        "hop_label",
    ]
    if stratify_celltype:
        summary_group_columns.append("celltype")
    summary_group_columns.append("gene")
    summary = (
        component_stats.groupby(summary_group_columns, observed=True, sort=False)
        .agg(
            n_components=("component_uid", "nunique"),
            total_cells=("n_cells", "sum"),
            mean_expression=("mean_expression", "mean"),
            mean_expression_median=("mean_expression", "median"),
            mean_expression_std=("mean_expression", "std"),
            positive_fraction=("positive_fraction", "mean"),
            positive_fraction_median=("positive_fraction", "median"),
            positive_fraction_std=("positive_fraction", "std"),
            positive_mean_expression=("positive_mean_expression", "mean"),
            positive_mean_expression_median=("positive_mean_expression", "median"),
            positive_mean_expression_std=("positive_mean_expression", "std"),
        )
        .reset_index()
    )
    sqrt_n = np.sqrt(summary["n_components"].astype(float))
    summary["mean_expression_sem"] = summary["mean_expression_std"] / sqrt_n
    summary["positive_fraction_sem"] = summary["positive_fraction_std"] / sqrt_n
    summary["positive_mean_expression_sem"] = (
        summary["positive_mean_expression_std"] / sqrt_n
    )
    sort_columns = ["observed_celltype", "signed_hop"]
    if stratify_celltype:
        sort_columns.append("celltype")
    sort_columns.append("gene")
    summary = summary.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    component_stats = component_stats.sort_values(
        ["observed_celltype", "sample", "component_id", "signed_hop", "gene"],
        kind="stable",
    ).reset_index(drop=True)
    return {"cell_level": cell_level, "component": component_stats, "summary": summary}


def _ordered_unique(values):
    return list(dict.fromkeys(str(value) for value in values))


def _prepare_adjacency(adata, connectivity_key, sample_key=None):
    if connectivity_key not in adata.obsp:
        raise KeyError(f"`{connectivity_key}` was not found in `adata.obsp`.")

    adjacency = sp.csr_matrix(adata.obsp[connectivity_key])

    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()

    if sample_key is not None:
        if sample_key not in adata.obs:
            raise KeyError(f"`{sample_key}` was not found in `adata.obs`.")

        sample_codes = pd.Categorical(adata.obs[sample_key]).codes
        coo = adjacency.tocoo()
        keep = sample_codes[coo.row] == sample_codes[coo.col]
        adjacency = sp.csr_matrix(
            (
                np.ones(int(keep.sum()), dtype=np.uint8),
                (coo.row[keep], coo.col[keep]),
            ),
            shape=adjacency.shape,
        )
    else:
        adjacency.data = np.ones(adjacency.nnz, dtype=np.uint8)

    adjacency.sort_indices()
    return adjacency


def _find_bridged_target_components(
    adjacency,
    celltype_labels,
    target_celltype,
    min_component_size,
    bridge_components=True,
):
    n_cells = len(celltype_labels)
    target_indices = np.flatnonzero(celltype_labels == target_celltype)

    valid_target = np.zeros(n_cells, dtype=bool)
    valid_bridge = np.zeros(n_cells, dtype=bool)
    target_component_codes = np.full(n_cells, -1, dtype=np.int32)
    bridge_component_codes = np.full(n_cells, -1, dtype=np.int32)

    empty_info = {
        "core_target_n": np.zeros(0, dtype=np.int64),
        "bridge_n": np.zeros(0, dtype=np.int64),
        "original_component_n": np.zeros(0, dtype=np.int64),
    }

    if len(target_indices) == 0:
        return (
            valid_target,
            target_component_codes,
            valid_bridge,
            bridge_component_codes,
            empty_info,
        )

    target_graph = adjacency[target_indices][:, target_indices]
    n_original, original_labels = connected_components(
        target_graph,
        directed=False,
        return_labels=True,
    )
    original_labels = original_labels.astype(np.int32, copy=False)

    bridge_indices = np.zeros(0, dtype=np.int64)
    bridge_original_components = []

    if bridge_components and n_original > 1:
        # Collect unique (non-target cell, contacted original component) pairs.
        target_edges = adjacency[target_indices].tocoo()
        neighbor_indices = target_edges.col.astype(np.int64, copy=False)
        contacted_components = original_labels[target_edges.row]

        non_target_edge = celltype_labels[neighbor_indices] != target_celltype
        neighbor_indices = neighbor_indices[non_target_edge]
        contacted_components = contacted_components[non_target_edge]

        if len(neighbor_indices) > 0:
            order = np.lexsort((contacted_components, neighbor_indices))
            pair_cells = neighbor_indices[order]
            pair_components = contacted_components[order]

            unique_pair = np.ones(len(pair_cells), dtype=bool)
            if len(pair_cells) > 1:
                unique_pair[1:] = (pair_cells[1:] != pair_cells[:-1]) | (
                    pair_components[1:] != pair_components[:-1]
                )
            pair_cells = pair_cells[unique_pair]
            pair_components = pair_components[unique_pair]

            unique_cells, starts, counts = np.unique(
                pair_cells,
                return_index=True,
                return_counts=True,
            )
            is_bridge = counts >= 2
            bridge_indices = unique_cells[is_bridge].astype(np.int64, copy=False)
            bridge_starts = starts[is_bridge]
            bridge_counts = counts[is_bridge]

            bridge_original_components = [
                pair_components[start : start + count]
                for start, count in zip(bridge_starts, bridge_counts)
            ]

    # Union original target components connected by bridge cells. Always attach
    # the larger root to the smaller root so merged IDs are deterministic.
    parent = np.arange(n_original, dtype=np.int32)

    def find_root(x):
        x = int(x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union_components(a, b):
        root_a = find_root(a)
        root_b = find_root(b)
        if root_a == root_b:
            return
        if root_a < root_b:
            parent[root_b] = root_a
        else:
            parent[root_a] = root_b

    for contacted in bridge_original_components:
        anchor = int(contacted[0])
        for other in contacted[1:]:
            union_components(anchor, int(other))

    roots = np.array([find_root(i) for i in range(n_original)], dtype=np.int32)
    merged_roots = np.unique(roots)
    root_to_merged = np.full(n_original, -1, dtype=np.int32)
    root_to_merged[merged_roots] = np.arange(len(merged_roots), dtype=np.int32)
    original_to_merged = root_to_merged[roots]
    target_merged_codes = original_to_merged[original_labels]

    merged_target_sizes = np.bincount(
        target_merged_codes,
        minlength=len(merged_roots),
    ).astype(np.int64)
    valid_merged_ids = np.flatnonzero(merged_target_sizes >= min_component_size)

    if len(valid_merged_ids) == 0:
        return (
            valid_target,
            target_component_codes,
            valid_bridge,
            bridge_component_codes,
            empty_info,
        )

    merged_to_valid = np.full(len(merged_roots), -1, dtype=np.int32)
    merged_to_valid[valid_merged_ids] = np.arange(
        len(valid_merged_ids),
        dtype=np.int32,
    )

    mapped_target_codes = merged_to_valid[target_merged_codes]
    target_keep = mapped_target_codes >= 0
    kept_target_indices = target_indices[target_keep]
    valid_target[kept_target_indices] = True
    target_component_codes[kept_target_indices] = mapped_target_codes[target_keep]

    # Every candidate bridge induced unions between all components it contacts,
    # so all contacted components must map to the same merged group.
    if len(bridge_indices) > 0:
        bridge_merged_codes = np.array(
            [
                original_to_merged[int(contacted[0])]
                for contacted in bridge_original_components
            ],
            dtype=np.int32,
        )
        mapped_bridge_codes = merged_to_valid[bridge_merged_codes]
        bridge_keep = mapped_bridge_codes >= 0
        kept_bridge_indices = bridge_indices[bridge_keep]
        valid_bridge[kept_bridge_indices] = True
        bridge_component_codes[kept_bridge_indices] = mapped_bridge_codes[bridge_keep]

    n_valid = len(valid_merged_ids)
    valid_original_counts = np.zeros(n_valid, dtype=np.int64)
    for original_id in range(n_original):
        valid_id = merged_to_valid[original_to_merged[original_id]]
        if valid_id >= 0:
            valid_original_counts[valid_id] += 1

    valid_bridge_counts = np.bincount(
        bridge_component_codes[bridge_component_codes >= 0],
        minlength=n_valid,
    ).astype(np.int64)

    component_info = {
        "core_target_n": merged_target_sizes[valid_merged_ids],
        "bridge_n": valid_bridge_counts,
        "original_component_n": valid_original_counts,
    }

    return (
        valid_target,
        target_component_codes,
        valid_bridge,
        bridge_component_codes,
        component_info,
    )


def _build_sample_spatial_index(coords, sample_codes):
    spatial_index = {}

    for sample_code in np.unique(sample_codes):
        sample_indices = np.flatnonzero(sample_codes == sample_code)
        spatial_index[int(sample_code)] = (
            sample_indices,
            cKDTree(coords[sample_indices]),
        )

    return spatial_index


def _query_component_window(
    coords,
    target_indices,
    sample_code,
    spatial_index,
    margin,
):
    target_coords = coords[target_indices, :2]
    bbox_min = target_coords.min(axis=0) - margin
    bbox_max = target_coords.max(axis=0) + margin

    center = (bbox_min + bbox_max) / 2.0
    half_size = (bbox_max - bbox_min) / 2.0
    query_radius = float(np.linalg.norm(half_size))

    sample_indices, tree = spatial_index[int(sample_code)]
    sample_positions = tree.query_ball_point(center, r=query_radius)

    if len(sample_positions) == 0:
        return np.array([], dtype=np.int64), bbox_min, bbox_max

    candidate_indices = sample_indices[np.asarray(sample_positions, dtype=np.int64)]
    candidate_coords = coords[candidate_indices, :2]

    keep = np.all(
        (candidate_coords >= bbox_min) & (candidate_coords <= bbox_max),
        axis=1,
    )

    local_indices = np.sort(candidate_indices[keep].astype(np.int64, copy=False))
    return local_indices, bbox_min, bbox_max


def _outside_seed_mask_from_window_edges(
    global_degree,
    local_graph,
    local_indices,
    background_local_positions,
):
    if len(background_local_positions) == 0:
        return np.zeros(0, dtype=bool)

    background_global_indices = local_indices[background_local_positions]
    local_rows = local_graph[background_local_positions]
    local_degree = np.diff(local_rows.indptr)

    return global_degree[background_global_indices] > local_degree


def _split_local_background(
    adjacency,
    global_degree,
    local_indices,
    barrier_global_mask,
):
    local_graph = adjacency[local_indices][:, local_indices].tocsr()
    local_barrier = barrier_global_mask[local_indices]

    background_local_positions = np.flatnonzero(~local_barrier)
    n_local = len(local_indices)

    local_inside = np.zeros(n_local, dtype=bool)
    local_outside = np.zeros(n_local, dtype=bool)

    if len(background_local_positions) == 0:
        return {
            "local_graph": local_graph,
            "background_graph": sp.csr_matrix((0, 0), dtype=np.uint8),
            "background_local_positions": background_local_positions,
            "background_inside": np.zeros(0, dtype=bool),
            "background_outside": np.zeros(0, dtype=bool),
            "outside_seed_mask": np.zeros(0, dtype=bool),
            "local_inside": local_inside,
            "local_outside": local_outside,
            "n_background_regions": 0,
            "n_inside_regions": 0,
            "n_outside_regions": 0,
            "n_outside_seeds": 0,
            "outside_seed_found": False,
        }

    background_graph = local_graph[background_local_positions][
        :, background_local_positions
    ].tocsr()

    n_regions, region_labels = connected_components(
        background_graph,
        directed=False,
        return_labels=True,
    )

    outside_seed_mask = _outside_seed_mask_from_window_edges(
        global_degree=global_degree,
        local_graph=local_graph,
        local_indices=local_indices,
        background_local_positions=background_local_positions,
    )

    if outside_seed_mask.any():
        outside_region_labels = np.unique(region_labels[outside_seed_mask])
        background_outside = np.isin(region_labels, outside_region_labels)
        background_inside = ~background_outside
    else:
        # Without an exterior seed the graph has no directional reference.
        # Keep all background cells unresolved instead of falsely calling them inside.
        outside_region_labels = np.array([], dtype=np.int32)
        background_outside = np.zeros(len(background_local_positions), dtype=bool)
        background_inside = np.zeros(len(background_local_positions), dtype=bool)

    local_outside[background_local_positions] = background_outside
    local_inside[background_local_positions] = background_inside

    return {
        "local_graph": local_graph,
        "background_graph": background_graph,
        "background_local_positions": background_local_positions,
        "background_inside": background_inside,
        "background_outside": background_outside,
        "outside_seed_mask": outside_seed_mask,
        "local_inside": local_inside,
        "local_outside": local_outside,
        "n_background_regions": int(n_regions),
        "n_inside_regions": int(
            len(np.unique(region_labels[background_inside]))
            if background_inside.any()
            else 0
        ),
        "n_outside_regions": int(len(outside_region_labels)),
        "n_outside_seeds": int(outside_seed_mask.sum()),
        "outside_seed_found": bool(outside_seed_mask.any()),
    }


def _outside_articulation_points(background_graph, outside_seed_mask):
    n_nodes = int(background_graph.shape[0])
    articulation = np.zeros(n_nodes, dtype=bool)

    seed_nodes = np.flatnonzero(outside_seed_mask).astype(np.int64, copy=False)
    if n_nodes == 0 or len(seed_nodes) == 0:
        return articulation

    root = n_nodes
    disc = np.full(n_nodes + 1, -1, dtype=np.int64)
    low = np.full(n_nodes + 1, -1, dtype=np.int64)
    parent = np.full(n_nodes + 1, -1, dtype=np.int64)
    child_count = np.zeros(n_nodes + 1, dtype=np.int32)
    next_edge = background_graph.indptr[:-1].astype(np.int64, copy=True)
    root_edge_done = np.zeros(n_nodes, dtype=bool)

    disc[root] = 0
    low[root] = 0
    time = 1
    root_next = 0
    stack = [root]

    while stack:
        node = stack[-1]
        neighbor = -1

        if node == root:
            if root_next < len(seed_nodes):
                neighbor = int(seed_nodes[root_next])
                root_next += 1
        else:
            edge_end = int(background_graph.indptr[node + 1])
            if next_edge[node] < edge_end:
                neighbor = int(background_graph.indices[next_edge[node]])
                next_edge[node] += 1
            elif outside_seed_mask[node] and not root_edge_done[node]:
                neighbor = root
                root_edge_done[node] = True

        if neighbor >= 0:
            if disc[neighbor] < 0:
                parent[neighbor] = node
                child_count[node] += 1
                disc[neighbor] = time
                low[neighbor] = time
                time += 1
                stack.append(neighbor)
            elif neighbor != parent[node]:
                low[node] = min(low[node], disc[neighbor])
            continue

        stack.pop()
        parent_node = int(parent[node])
        if parent_node >= 0:
            if parent_node != root and low[node] >= disc[parent_node]:
                articulation[parent_node] = True
            low[parent_node] = min(low[parent_node], low[node])

    return articulation


def _find_closure_bridges(
    split,
    local_indices,
    current_barrier_global_mask,
    eligible_global_mask,
    min_barrier_contacts=2,
):
    background_positions = split["background_local_positions"]
    if len(background_positions) == 0 or not split["outside_seed_found"]:
        return np.zeros(0, dtype=np.int64), 0, 0

    current_barrier_local = current_barrier_global_mask[local_indices]
    barrier_positions = np.flatnonzero(current_barrier_local)
    if len(barrier_positions) == 0:
        return np.zeros(0, dtype=np.int64), 0, 0

    local_graph = split["local_graph"]
    contact_counts = np.asarray(
        local_graph[background_positions][:, barrier_positions].sum(axis=1)
    ).ravel()

    eligible_background = eligible_global_mask[local_indices[background_positions]]

    candidate_mask = (
        split["background_outside"]
        & eligible_background
        & (contact_counts >= int(min_barrier_contacts))
        & ~split["outside_seed_mask"]
    )
    candidate_n = int(candidate_mask.sum())
    if candidate_n == 0:
        return np.zeros(0, dtype=np.int64), 0, 0

    articulation = _outside_articulation_points(
        background_graph=split["background_graph"],
        outside_seed_mask=split["outside_seed_mask"],
    )
    closure_background_positions = np.flatnonzero(candidate_mask & articulation)
    articulation_candidate_n = int(closure_background_positions.size)

    if articulation_candidate_n == 0:
        return np.zeros(0, dtype=np.int64), candidate_n, 0

    closure_local_positions = background_positions[closure_background_positions]
    closure_global_indices = local_indices[closure_local_positions]
    return (
        closure_global_indices.astype(np.int64, copy=False),
        candidate_n,
        articulation_candidate_n,
    )


def _component_hop1_indices(adjacency, barrier_indices, current_barrier_mask):
    barrier_rows = adjacency[barrier_indices]
    if barrier_rows.nnz == 0:
        return np.array([], dtype=np.int64)

    hop1_indices = np.unique(barrier_rows.indices)
    hop1_indices = hop1_indices[~current_barrier_mask[hop1_indices]]
    return hop1_indices.astype(np.int64, copy=False)


def _evaluate_component_closure_state(
    *,
    adjacency,
    global_degree,
    coords,
    celltype_labels,
    observed_celltype,
    component_id,
    target_component_codes,
    bridge_component_codes,
    valid_target,
    valid_bridge,
    sample_codes,
    sample_labels,
    spatial_index,
    local_lookup,
    margin,
    barrier_mode,
    closure_bridges,
    closure_min_barrier_contacts,
    min_closure_index,
    component_info,
):
    current_target_mask = target_component_codes == component_id
    current_bridge_mask = bridge_component_codes == component_id
    current_barrier_mask = current_target_mask | current_bridge_mask

    target_indices = np.flatnonzero(current_target_mask)
    bridge_indices = np.flatnonzero(current_bridge_mask)
    barrier_indices = np.flatnonzero(current_barrier_mask)

    component_sample_codes = np.unique(sample_codes[barrier_indices])
    if len(component_sample_codes) != 1:
        raise RuntimeError(
            "A target component spans multiple samples. Check "
            "`sample_key` and the adjacency matrix."
        )
    component_sample_code = int(component_sample_codes[0])

    local_indices, bbox_min, bbox_max = _query_component_window(
        coords=coords,
        target_indices=barrier_indices,
        sample_code=component_sample_code,
        spatial_index=spatial_index,
        margin=float(margin),
    )

    if len(local_indices) == 0:
        return None

    local_lookup[local_indices] = np.arange(
        len(local_indices),
        dtype=np.int32,
    )

    try:
        if barrier_mode == "current_component":
            barrier_mask = current_barrier_mask
        else:
            barrier_mask = valid_target | valid_bridge

        initial_split = _split_local_background(
            adjacency=adjacency,
            global_degree=global_degree,
            local_indices=local_indices,
            barrier_global_mask=barrier_mask,
        )

        closure_indices = np.zeros(0, dtype=np.int64)
        closure_candidate_n = 0
        closure_articulation_n = 0

        if closure_bridges:
            (
                closure_indices,
                closure_candidate_n,
                closure_articulation_n,
            ) = _find_closure_bridges(
                split=initial_split,
                local_indices=local_indices,
                current_barrier_global_mask=current_barrier_mask,
                eligible_global_mask=(
                    (celltype_labels != observed_celltype) & ~valid_bridge
                ),
                min_barrier_contacts=closure_min_barrier_contacts,
            )

        if len(closure_indices) > 0:
            current_barrier_mask = current_barrier_mask.copy()
            current_barrier_mask[closure_indices] = True
            barrier_indices = np.concatenate([barrier_indices, closure_indices]).astype(
                np.int64, copy=False
            )

            if barrier_mode == "current_component":
                barrier_mask = current_barrier_mask
            else:
                barrier_mask = (valid_target | valid_bridge).copy()
                barrier_mask[closure_indices] = True

            split = _split_local_background(
                adjacency=adjacency,
                global_degree=global_degree,
                local_indices=local_indices,
                barrier_global_mask=barrier_mask,
            )
        else:
            split = initial_split

        local_inside = split["local_inside"]
        local_outside = split["local_outside"]

        hop1_indices = _component_hop1_indices(
            adjacency=adjacency,
            barrier_indices=barrier_indices,
            current_barrier_mask=current_barrier_mask,
        )

        hop1_local_positions = local_lookup[hop1_indices]
        hop1_outside_window = hop1_local_positions < 0

        if hop1_outside_window.any():
            missing_n = int(hop1_outside_window.sum())
            raise ValueError(
                f"{observed_celltype} component {int(component_id)} has "
                f"{missing_n} Hop1 cells outside the local window. Increase "
                f"`margin` above {margin}."
            )

        in_hop1_local = hop1_local_positions[local_inside[hop1_local_positions]]
        out_hop1_local = hop1_local_positions[local_outside[hop1_local_positions]]

        unresolved_hop1 = (
            len(hop1_local_positions) - len(in_hop1_local) - len(out_hop1_local)
        )

        out_hop1_indices = local_indices[out_hop1_local]
        if len(out_hop1_indices) > 0:
            target_out_degree = np.diff(
                adjacency[target_indices][:, out_hop1_indices].indptr
            )
            target_touching_out_n = int(np.count_nonzero(target_out_degree))
        else:
            target_touching_out_n = 0

        outside_exposure = (
            target_touching_out_n / len(target_indices)
            if len(target_indices) > 0
            else np.nan
        )
        closure_index = (
            float(np.clip(1.0 - outside_exposure, 0.0, 1.0))
            if np.isfinite(outside_exposure)
            else np.nan
        )
        passes_closure_filter = bool(
            np.isfinite(closure_index) and closure_index >= float(min_closure_index)
        )

        sample_indices, _ = spatial_index[component_sample_code]
        sample_bbox_min = coords[sample_indices].min(axis=0)
        sample_bbox_max = coords[sample_indices].max(axis=0)
        window_truncated = bool(
            np.any(bbox_min <= sample_bbox_min) or np.any(bbox_max >= sample_bbox_max)
        )

        summary = {
            "observed_celltype": observed_celltype,
            "component_id": int(component_id),
            "sample": str(sample_labels[component_sample_code]),
            "target_n": int(len(target_indices)),
            "bridge_n": int(len(bridge_indices)),
            "closure_bridge_n": int(len(closure_indices)),
            "barrier_n": int(len(barrier_indices)),
            "closure_candidate_n": int(closure_candidate_n),
            "closure_articulation_n": int(closure_articulation_n),
            "original_component_n": int(
                component_info["original_component_n"][int(component_id)]
            ),
            "local_n": int(len(local_indices)),
            "hop1_n": int(len(hop1_indices)),
            "in_hop1_n": int(len(in_hop1_local)),
            "out_hop1_n": int(len(out_hop1_local)),
            "target_touching_out_n": int(target_touching_out_n),
            "outside_exposure": float(outside_exposure),
            "closure_index": float(closure_index),
            "min_closure_index": float(min_closure_index),
            "passes_closure_filter": bool(passes_closure_filter),
            "used_in_analysis": bool(passes_closure_filter),
            "multihop_expanded": False,
            "unresolved_hop1_n": int(unresolved_hop1),
            "inside_region_n": int(local_inside.sum()),
            "outside_region_n": int(local_outside.sum()),
            "n_background_regions": split["n_background_regions"],
            "n_inside_regions": split["n_inside_regions"],
            "n_outside_regions": split["n_outside_regions"],
            "n_outside_seeds": split["n_outside_seeds"],
            "outside_seed_found": split["outside_seed_found"],
            "max_in_hop_found": int(len(in_hop1_local) > 0),
            "max_out_hop_found": int(len(out_hop1_local) > 0),
            "window_truncated": window_truncated,
            "margin": float(margin),
            "bbox_xmin": float(bbox_min[0]),
            "bbox_xmax": float(bbox_max[0]),
            "bbox_ymin": float(bbox_min[1]),
            "bbox_ymax": float(bbox_max[1]),
            "barrier_mode": barrier_mode,
            "closure_bridges": bool(closure_bridges),
            "closure_min_barrier_contacts": int(closure_min_barrier_contacts),
        }

        return {
            "target_indices": target_indices,
            "bridge_indices": bridge_indices,
            "closure_indices": closure_indices,
            "barrier_indices": barrier_indices,
            "component_sample_code": component_sample_code,
            "local_indices": local_indices,
            "split": split,
            "in_hop1_local": in_hop1_local,
            "out_hop1_local": out_hop1_local,
            "summary": summary,
        }
    finally:
        local_lookup[local_indices] = -1


def _expand_side_hops_global(
    adjacency,
    seed_indices,
    max_hop,
    blocked_indices=None,
):
    n_cells = adjacency.shape[0]
    hop_codes = np.zeros(n_cells, dtype=np.int16)

    if len(seed_indices) == 0:
        return hop_codes

    seed_indices = np.asarray(seed_indices, dtype=np.int64)

    blocked = np.zeros(n_cells, dtype=bool)
    if blocked_indices is not None and len(blocked_indices) > 0:
        blocked[np.asarray(blocked_indices, dtype=np.int64)] = True

    frontier = np.zeros(n_cells, dtype=bool)
    frontier[seed_indices] = True
    frontier &= ~blocked

    if not frontier.any():
        return hop_codes

    visited = frontier.copy()
    visited |= blocked

    hop_codes[frontier] = 1

    for hop in range(2, max_hop + 1):
        neighbors = np.asarray(adjacency.dot(frontier.astype(np.int8))).ravel() > 0

        ring = neighbors & ~visited

        if not ring.any():
            break

        hop_codes[ring] = hop
        visited[ring] = True
        frontier = ring

    return hop_codes


def _build_all_candidate_neighborhoods(
    adata,
    observed_celltypes: Sequence[str],
    celltype_key="celltype",
    connectivity_key="sstt_neighbor_network",
    spatial_key="spatial",
    max_hop=10,
    min_component_size=10,
    min_closure_index=0.0,
    margin=50.0,
    sample_key=None,
    barrier_mode="current_component",
    bridge_components=True,
    closure_bridges=True,
    closure_min_barrier_contacts=2,
    show_progress=True,
):
    if celltype_key not in adata.obs:
        raise KeyError(f"`{celltype_key}` was not found in `adata.obs`.")

    if spatial_key not in adata.obsm:
        raise KeyError(f"`{spatial_key}` was not found in `adata.obsm`.")

    if max_hop < 1:
        raise ValueError("`max_hop` must be at least 1.")

    if min_component_size < 2:
        raise ValueError("`min_component_size` must be at least 2.")

    if not 0.0 <= float(min_closure_index) <= 1.0:
        raise ValueError("`min_closure_index` must be between 0 and 1.")

    if margin <= 0:
        raise ValueError("`margin` must be greater than 0.")

    if closure_min_barrier_contacts < 2:
        raise ValueError("`closure_min_barrier_contacts` must be at least 2.")

    if barrier_mode not in {"current_component", "all_valid_components"}:
        raise ValueError(
            "`barrier_mode` must be 'current_component' or " "'all_valid_components'."
        )

    if adata.obs[celltype_key].isna().any():
        raise ValueError(f"`adata.obs[{celltype_key!r}]` contains missing values.")

    observed_celltypes = _ordered_unique(observed_celltypes)
    if not observed_celltypes:
        raise ValueError("`observed_celltypes` cannot be empty.")

    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"`adata.obsm[{spatial_key!r}]` must have shape (n_obs, >=2).")
    coords = coords[:, :2]

    celltype_labels = adata.obs[celltype_key].astype(str).to_numpy()
    all_celltypes = set(celltype_labels)
    missing = [x for x in observed_celltypes if x not in all_celltypes]
    if missing:
        raise ValueError(f"Observed cell types were not found: {missing}")

    adjacency = _prepare_adjacency(
        adata=adata,
        connectivity_key=connectivity_key,
        sample_key=sample_key,
    )

    global_degree = np.diff(adjacency.indptr).astype(np.int32, copy=False)

    if sample_key is None:
        sample_codes = np.zeros(adata.n_obs, dtype=np.int32)
        sample_labels = np.array(["sample_0"], dtype=object)
    else:
        if adata.obs[sample_key].isna().any():
            raise ValueError(f"`adata.obs[{sample_key!r}]` contains missing values.")
        sample_categorical = pd.Categorical(adata.obs[sample_key])
        sample_codes = sample_categorical.codes.astype(np.int32)
        sample_labels = np.asarray(sample_categorical.categories, dtype=object)

    spatial_index = _build_sample_spatial_index(coords, sample_codes)

    membership_blocks = []
    summary_by_component = {}
    local_lookup = np.full(adata.n_obs, -1, dtype=np.int32)
    obs_names = adata.obs_names.astype(str).to_numpy()

    observed_iterator = observed_celltypes
    if show_progress:
        observed_iterator = tqdm(
            observed_celltypes,
            desc="Observed cell types",
        )

    for observed_celltype in observed_iterator:
        (
            valid_target,
            target_component_codes,
            valid_bridge,
            bridge_component_codes,
            component_info,
        ) = _find_bridged_target_components(
            adjacency=adjacency,
            celltype_labels=celltype_labels,
            target_celltype=observed_celltype,
            min_component_size=min_component_size,
            bridge_components=bridge_components,
        )

        candidate_component_ids = np.unique(
            target_component_codes[target_component_codes >= 0]
        )

        if float(min_closure_index) > 0.0:
            retained_component_ids = []

            # Screening intentionally stops at Hop1. No multi-hop expansion or
            # membership DataFrame is created for rejected components.
            for component_id in candidate_component_ids:
                state = _evaluate_component_closure_state(
                    adjacency=adjacency,
                    global_degree=global_degree,
                    coords=coords,
                    celltype_labels=celltype_labels,
                    observed_celltype=observed_celltype,
                    component_id=int(component_id),
                    target_component_codes=target_component_codes,
                    bridge_component_codes=bridge_component_codes,
                    valid_target=valid_target,
                    valid_bridge=valid_bridge,
                    sample_codes=sample_codes,
                    sample_labels=sample_labels,
                    spatial_index=spatial_index,
                    local_lookup=local_lookup,
                    margin=margin,
                    barrier_mode=barrier_mode,
                    closure_bridges=closure_bridges,
                    closure_min_barrier_contacts=closure_min_barrier_contacts,
                    min_closure_index=min_closure_index,
                    component_info=component_info,
                )

                if state is None:
                    continue

                summary = state["summary"]
                summary["bridge_components"] = bool(bridge_components)
                summary_by_component[(observed_celltype, int(component_id))] = summary

                if summary["passes_closure_filter"]:
                    retained_component_ids.append(int(component_id))

            retained_component_ids = np.asarray(
                retained_component_ids,
                dtype=np.int32,
            )
        else:
            # With the default threshold, every post-size-filter component is
            # used. Evaluate each component only once during expansion.
            retained_component_ids = candidate_component_ids.astype(
                np.int32,
                copy=False,
            )

        if show_progress:
            if float(min_closure_index) > 0.0:
                tqdm.write(
                    f"[INFO] {observed_celltype}: "
                    f"{len(candidate_component_ids)} components after "
                    f"min_component_size={int(min_component_size)}; "
                    f"{len(retained_component_ids)} retained with "
                    f"closure_index >= {float(min_closure_index):.3f}."
                )
            else:
                tqdm.write(
                    f"[INFO] {observed_celltype}: "
                    f"{len(candidate_component_ids)} candidate components after "
                    f"min_component_size={int(min_component_size)}; "
                    "closure filter not applied."
                )

        component_iterator = retained_component_ids
        if show_progress and len(retained_component_ids) > 0:
            component_iterator = tqdm(
                retained_component_ids,
                desc=f"{observed_celltype} retained components",
                leave=False,
            )

        for component_id in component_iterator:
            state = _evaluate_component_closure_state(
                adjacency=adjacency,
                global_degree=global_degree,
                coords=coords,
                celltype_labels=celltype_labels,
                observed_celltype=observed_celltype,
                component_id=int(component_id),
                target_component_codes=target_component_codes,
                bridge_component_codes=bridge_component_codes,
                valid_target=valid_target,
                valid_bridge=valid_bridge,
                sample_codes=sample_codes,
                sample_labels=sample_labels,
                spatial_index=spatial_index,
                local_lookup=local_lookup,
                margin=margin,
                barrier_mode=barrier_mode,
                closure_bridges=closure_bridges,
                closure_min_barrier_contacts=closure_min_barrier_contacts,
                min_closure_index=min_closure_index,
                component_info=component_info,
            )

            if state is None:
                continue

            summary = state["summary"]
            summary["bridge_components"] = bool(bridge_components)
            if not summary["passes_closure_filter"]:
                raise RuntimeError(
                    "A component passed closure screening but failed during "
                    "retained-component expansion. The topology changed "
                    "between deterministic evaluation passes."
                )

            # ---------------------------------------------------------
            # Global topology expansion
            #
            # margin/window is only used for Hop1 inside/outside assignment.
            # Hop2+ expansion must follow the original global cell graph and
            # cannot be truncated by the component bounding box.
            # ---------------------------------------------------------
            in_hop1_indices = state["local_indices"][state["in_hop1_local"]]
            out_hop1_indices = state["local_indices"][state["out_hop1_local"]]

            # Protect all valid Target/Bridge components during global BFS.
            #
            # Hop2+ expansion follows the complete adjacency graph rather than
            # the local margin window. Therefore, Target or Bridge cells from
            # other valid components must also remain unavailable to the current
            # component's InHop/OutHop expansion.
            #
            # ClosureBridge cells are component-specific and only the current
            # component's ClosureBridge cells are protected here.
            protected_barrier_indices = np.concatenate(
                [
                    np.flatnonzero(valid_target),
                    np.flatnonzero(valid_bridge),
                    state["closure_indices"],
                ]
            ).astype(np.int64, copy=False)

            in_hop_codes = _expand_side_hops_global(
                adjacency=adjacency,
                seed_indices=in_hop1_indices,
                max_hop=max_hop,
                blocked_indices=protected_barrier_indices,
            )

            out_hop_codes = _expand_side_hops_global(
                adjacency=adjacency,
                seed_indices=out_hop1_indices,
                max_hop=max_hop,
                blocked_indices=protected_barrier_indices,
            )

            target_indices = state["target_indices"]
            bridge_indices = state["bridge_indices"]
            closure_indices = state["closure_indices"]
            component_sample_code = state["component_sample_code"]

            component_rows = [
                pd.DataFrame(
                    {
                        "observed_celltype": observed_celltype,
                        "component_id": int(component_id),
                        "sample": str(sample_labels[component_sample_code]),
                        "cell_index": target_indices.astype(np.int64),
                        "cell_id": obs_names[target_indices],
                        "signed_hop": np.zeros(len(target_indices), dtype=np.int16),
                        "hop": np.zeros(len(target_indices), dtype=np.int16),
                        "side": "Target",
                        "hop_label": "Target",
                        "is_target": True,
                        "is_bridge": False,
                        "is_closure_bridge": False,
                        "is_barrier": True,
                    }
                )
            ]

            if len(bridge_indices) > 0:
                component_rows.append(
                    pd.DataFrame(
                        {
                            "observed_celltype": observed_celltype,
                            "component_id": int(component_id),
                            "sample": str(sample_labels[component_sample_code]),
                            "cell_index": bridge_indices.astype(np.int64),
                            "cell_id": obs_names[bridge_indices],
                            "signed_hop": np.full(
                                len(bridge_indices), -1, dtype=np.int16
                            ),
                            "hop": np.ones(len(bridge_indices), dtype=np.int16),
                            "side": "Bridge",
                            "hop_label": "Bridge",
                            "is_target": False,
                            "is_bridge": True,
                            "is_closure_bridge": False,
                            "is_barrier": True,
                        }
                    )
                )

            if len(closure_indices) > 0:
                component_rows.append(
                    pd.DataFrame(
                        {
                            "observed_celltype": observed_celltype,
                            "component_id": int(component_id),
                            "sample": str(sample_labels[component_sample_code]),
                            "cell_index": closure_indices.astype(np.int64),
                            "cell_id": obs_names[closure_indices],
                            "signed_hop": np.full(
                                len(closure_indices), -1, dtype=np.int16
                            ),
                            "hop": np.ones(len(closure_indices), dtype=np.int16),
                            "side": "ClosureBridge",
                            "hop_label": "ClosureBridge",
                            "is_target": False,
                            "is_bridge": False,
                            "is_closure_bridge": True,
                            "is_barrier": True,
                        }
                    )
                )

            for hop in range(1, max_hop + 1):
                in_global_indices = np.flatnonzero(in_hop_codes == hop)
                if len(in_global_indices) > 0:
                    component_rows.append(
                        pd.DataFrame(
                            {
                                "observed_celltype": observed_celltype,
                                "component_id": int(component_id),
                                "sample": str(sample_labels[component_sample_code]),
                                "cell_index": in_global_indices.astype(np.int64),
                                "cell_id": obs_names[in_global_indices],
                                "signed_hop": np.full(
                                    len(in_global_indices), -hop, dtype=np.int16
                                ),
                                "hop": np.full(
                                    len(in_global_indices), hop, dtype=np.int16
                                ),
                                "side": "In",
                                "hop_label": f"InHop{hop}",
                                "is_target": False,
                                "is_bridge": False,
                                "is_closure_bridge": False,
                                "is_barrier": False,
                            }
                        )
                    )

                out_global_indices = np.flatnonzero(out_hop_codes == hop)
                if len(out_global_indices) > 0:
                    component_rows.append(
                        pd.DataFrame(
                            {
                                "observed_celltype": observed_celltype,
                                "component_id": int(component_id),
                                "sample": str(sample_labels[component_sample_code]),
                                "cell_index": out_global_indices.astype(np.int64),
                                "cell_id": obs_names[out_global_indices],
                                "signed_hop": np.full(
                                    len(out_global_indices), hop, dtype=np.int16
                                ),
                                "hop": np.full(
                                    len(out_global_indices), hop, dtype=np.int16
                                ),
                                "side": "Out",
                                "hop_label": f"OutHop{hop}",
                                "is_target": False,
                                "is_bridge": False,
                                "is_closure_bridge": False,
                                "is_barrier": False,
                            }
                        )
                    )

            membership_blocks.append(pd.concat(component_rows, ignore_index=True))

            summary["max_in_hop_found"] = int(in_hop_codes.max(initial=0))
            summary["max_out_hop_found"] = int(out_hop_codes.max(initial=0))
            summary["multihop_expanded"] = True
            summary["used_in_analysis"] = True
            summary_by_component[(observed_celltype, int(component_id))] = summary

    membership_columns = [
        "observed_celltype",
        "component_id",
        "sample",
        "cell_index",
        "cell_id",
        "signed_hop",
        "hop",
        "side",
        "hop_label",
        "is_target",
        "is_bridge",
        "is_closure_bridge",
        "is_barrier",
    ]

    if membership_blocks:
        membership = pd.concat(membership_blocks, ignore_index=True)
        membership = membership.loc[:, membership_columns]
        membership = membership.sort_values(
            ["observed_celltype", "component_id", "signed_hop", "cell_index"]
        ).reset_index(drop=True)
    else:
        membership = pd.DataFrame(columns=membership_columns)

    if summary_by_component:
        component_summary = pd.DataFrame(summary_by_component.values())
        component_summary = component_summary.sort_values(
            ["observed_celltype", "sample", "component_id"]
        ).reset_index(drop=True)
    else:
        component_summary = pd.DataFrame()

    return membership, component_summary


def _write_signed_component_to_obs(
    adata,
    membership,
    observed_celltype,
    component_id,
    column=None,
    outside_label="outside",
    use_negative_hops=True,
):
    required_columns = {
        "observed_celltype",
        "component_id",
        "cell_index",
        "signed_hop",
        "hop",
        "side",
        "hop_label",
    }
    missing = required_columns.difference(membership.columns)
    if missing:
        raise KeyError(f"`membership` is missing columns: {sorted(missing)}")

    selected = membership[
        (membership["observed_celltype"].astype(str) == str(observed_celltype))
        & (membership["component_id"].astype(int) == int(component_id))
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No membership rows found for observed_celltype="
            f"{observed_celltype!r}, component_id={component_id}."
        )

    selected["hop"] = selected["hop"].astype(np.int16)
    max_hop = int(selected["hop"].max())

    if use_negative_hops:
        categories = (
            [f"InHop{hop}" for hop in range(max_hop, 0, -1)]
            + ["Bridge", "ClosureBridge", "Target"]
            + [f"OutHop{hop}" for hop in range(1, max_hop + 1)]
            + [outside_label]
        )
        labels = selected["hop_label"].astype(str).to_numpy()
    else:
        categories = (
            ["Target"]
            + [f"Hop{hop}" for hop in range(1, max_hop + 1)]
            + [outside_label]
        )
        hops = selected["hop"].to_numpy(dtype=np.int16)
        labels = np.where(
            hops == 0,
            "Target",
            np.char.add("Hop", hops.astype(str)),
        )

    values = np.full(adata.n_obs, outside_label, dtype=object)
    cell_indices = selected["cell_index"].to_numpy(dtype=np.int64)
    values[cell_indices] = labels

    if column is None:
        safe_celltype = str(observed_celltype).replace(" ", "_")
        column = f"{safe_celltype}_component_{int(component_id)}_signed_hop"

    adata.obs[column] = pd.Categorical(
        values,
        categories=categories,
        ordered=True,
    )
    return column


def _aggregate_signed_celltype_membership(
    membership,
    observed_celltype,
    tie_policy="ambiguous",
    ambiguous_label="ambiguous",
    use_negative_hops=True,
):

    selected = membership[
        membership["observed_celltype"].astype(str) == str(observed_celltype)
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No membership rows found for observed_celltype={observed_celltype!r}."
        )

    selected["cell_index"] = selected["cell_index"].astype(np.int64)
    selected["component_id"] = selected["component_id"].astype(np.int64)
    selected["hop"] = selected["hop"].astype(np.int16)
    selected["signed_hop"] = selected["signed_hop"].astype(np.int16)
    selected["hop_label"] = selected["hop_label"].astype(str)
    selected["side"] = selected["side"].astype(str)

    # Preserve distinct component/side assignments. Inward and outward global
    # searches may meet, so silently keeping the first row would make the
    # result depend on sort order rather than graph distance and tie policy.
    selected = selected.drop_duplicates(
        subset=["component_id", "cell_index", "side", "hop", "hop_label"],
        keep="first",
    )

    if not use_negative_hops:
        non_target = selected["hop"].to_numpy(dtype=np.int16) > 0
        unsigned_hops = selected["hop"].to_numpy(dtype=np.int16)

        selected.loc[non_target, "signed_hop"] = unsigned_hops[non_target]
        selected.loc[non_target, "hop_label"] = np.char.add(
            "Hop",
            unsigned_hops[non_target].astype(str),
        )
        selected.loc[non_target, "side"] = "Hop"
        selected.loc[~non_target, "signed_hop"] = 0
        selected.loc[~non_target, "hop_label"] = "Target"
        selected.loc[~non_target, "side"] = "Target"

    nearest_hop = selected.groupby("cell_index", sort=False)["hop"].transform("min")
    nearest = selected[selected["hop"] == nearest_hop].copy()

    if use_negative_hops:
        role_priority = (
            nearest["side"]
            .map({"Target": 0, "Bridge": 1, "ClosureBridge": 2, "In": 3, "Out": 4})
            .fillna(5)
        )
    else:
        role_priority = nearest["side"].map({"Target": 0, "Hop": 1}).fillna(2)

    nearest = nearest.assign(_role_priority=role_priority).sort_values(
        ["cell_index", "_role_priority", "component_id", "signed_hop"],
        kind="stable",
    )

    label_counts = nearest.groupby("cell_index", sort=False)["hop_label"].nunique()
    nearest_counts = nearest.groupby("cell_index", sort=False)["component_id"].nunique()
    conflict_indices = label_counts.index[label_counts > 1].to_numpy(dtype=np.int64)

    if tie_policy == "error" and len(conflict_indices) > 0:
        examples = conflict_indices[:10].tolist()
        raise ValueError(
            f"Found {len(conflict_indices)} equal-distance assignment conflicts. "
            f"Example cell indices: {examples}"
        )

    chosen = nearest.drop_duplicates("cell_index", keep="first").set_index("cell_index")

    if use_negative_hops and len(conflict_indices) > 0 and tie_policy in {"in", "out"}:
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
        chosen.loc[conflict_indices, "hop_label"] = str(ambiguous_label)
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
    aggregated = chosen.loc[:, columns].reset_index()
    return aggregated.sort_values("cell_index").reset_index(drop=True)


def _write_signed_celltype_to_obs(
    adata,
    membership,
    observed_celltype,
    column=None,
    outside_label="outside",
    ambiguous_label="ambiguous",
    tie_policy="ambiguous",
    component_column=None,
    conflict_column=None,
    use_negative_hops=True,
):
    aggregated = _aggregate_signed_celltype_membership(
        membership=membership,
        observed_celltype=observed_celltype,
        tie_policy=tie_policy,
        ambiguous_label=ambiguous_label,
        use_negative_hops=use_negative_hops,
    )

    max_hop = int(aggregated["hop"].max())
    if use_negative_hops:
        categories = (
            [f"InHop{hop}" for hop in range(max_hop, 0, -1)]
            + ["Bridge", "ClosureBridge", "Target"]
            + [f"OutHop{hop}" for hop in range(1, max_hop + 1)]
        )
    else:
        categories = ["Target"] + [f"Hop{hop}" for hop in range(1, max_hop + 1)]

    if tie_policy == "ambiguous" and aggregated["conflict"].any():
        categories.append(str(ambiguous_label))
    categories.append(str(outside_label))

    safe_celltype = str(observed_celltype).replace(" ", "_")
    if column is None:
        column = f"{safe_celltype}_signed_hop"
    if component_column is None:
        component_column = f"{safe_celltype}_signed_component"
    if conflict_column is None:
        conflict_column = f"{safe_celltype}_signed_conflict"

    labels = np.full(adata.n_obs, outside_label, dtype=object)
    components = np.full(adata.n_obs, -1, dtype=np.int32)
    conflicts = np.zeros(adata.n_obs, dtype=bool)

    cell_indices = aggregated["cell_index"].to_numpy(dtype=np.int64)
    if np.any((cell_indices < 0) | (cell_indices >= adata.n_obs)):
        raise IndexError("`membership.cell_index` contains values outside `adata`.")

    labels[cell_indices] = aggregated["hop_label"].astype(str).to_numpy()
    components[cell_indices] = aggregated["component_id"].to_numpy(dtype=np.int32)
    conflicts[cell_indices] = aggregated["conflict"].to_numpy(dtype=bool)

    adata.obs[column] = pd.Categorical(
        labels,
        categories=categories,
        ordered=True,
    )
    adata.obs[component_column] = components
    adata.obs[conflict_column] = conflicts

    return {
        "label_column": column,
        "component_column": component_column,
        "conflict_column": conflict_column,
        "n_assigned": int(len(aggregated)),
        "n_conflicts": int(aggregated["conflict"].sum()),
        "tie_policy": tie_policy,
        "use_negative_hops": bool(use_negative_hops),
    }


def _resolve_device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but no CUDA device is available to PyTorch."
        )
    return resolved


def _effective_batch_size(
    permutation_batch_size,
    max_stratum_size,
    max_batch_elements,
):
    if max_stratum_size <= 0:
        return permutation_batch_size

    memory_limited = max(1, max_batch_elements // max_stratum_size)
    return max(1, min(permutation_batch_size, memory_limited))


def _statistical_hop_values(max_hop, use_negative_hops):
    if use_negative_hops:
        return np.concatenate(
            [
                np.arange(-max_hop, 0, dtype=np.int16),
                np.arange(1, max_hop + 1, dtype=np.int16),
            ]
        )
    return np.arange(1, max_hop + 1, dtype=np.int16)


def _hop_codes_to_group_indices(hop_codes, max_hop, use_negative_hops):
    hop_codes = np.asarray(hop_codes, dtype=np.int16)
    group_indices = np.full(len(hop_codes), -1, dtype=np.int64)

    if use_negative_hops:
        negative = (hop_codes >= -max_hop) & (hop_codes <= -1)
        positive = (hop_codes >= 1) & (hop_codes <= max_hop)

        group_indices[negative] = hop_codes[negative].astype(np.int64) + max_hop
        group_indices[positive] = max_hop + hop_codes[positive].astype(np.int64) - 1
    else:
        positive = (hop_codes >= 1) & (hop_codes <= max_hop)
        group_indices[positive] = hop_codes[positive].astype(np.int64) - 1

    return group_indices


def _signed_hop_values(max_hop):
    return _statistical_hop_values(max_hop, use_negative_hops=True)


def _signed_hops_to_group_indices(signed_hops, max_hop):
    return _hop_codes_to_group_indices(
        signed_hops,
        max_hop=max_hop,
        use_negative_hops=True,
    )


def _count_by_group(group_indices, celltype_codes, n_groups, n_celltypes):
    counts = np.zeros((n_groups, n_celltypes), dtype=np.int64)
    np.add.at(counts, (group_indices, celltype_codes), 1)
    return counts


def _analytic_expected_counts_stratified(
    local_groups,
    local_celltypes,
    local_strata,
    n_groups,
    n_celltypes,
):
    expected = np.zeros((n_groups, n_celltypes), dtype=np.float64)

    for stratum in np.unique(local_strata):
        mask = local_strata == stratum
        n_cells = int(mask.sum())
        if n_cells == 0:
            continue

        group_counts = np.bincount(local_groups[mask], minlength=n_groups).astype(
            np.float64
        )
        celltype_counts = np.bincount(
            local_celltypes[mask], minlength=n_celltypes
        ).astype(np.float64)
        expected += np.outer(group_counts, celltype_counts / n_cells)

    return expected


def _permutation_pvalues_numpy(
    local_groups,
    local_celltypes,
    local_strata,
    observed_counts,
    expected_counts,
    n_groups,
    n_celltypes,
    n_permutations,
    random_state,
    show_progress,
    description,
):
    rng = np.random.default_rng(random_state)
    observed_deviation = np.abs(observed_counts - expected_counts)
    extreme_counts = np.zeros((n_groups, n_celltypes), dtype=np.int64)
    strata = [
        np.flatnonzero(local_strata == stratum) for stratum in np.unique(local_strata)
    ]

    iterator = range(n_permutations)
    if show_progress:
        iterator = tqdm(iterator, desc=description, leave=False)

    for _ in iterator:
        permuted = local_celltypes.copy()
        for indices in strata:
            permuted[indices] = rng.permutation(local_celltypes[indices])

        permuted_counts = _count_by_group(
            local_groups,
            permuted,
            n_groups,
            n_celltypes,
        )
        extreme_counts += (
            np.abs(permuted_counts - expected_counts) >= observed_deviation - 1e-12
        )

    return (extreme_counts + 1) / (n_permutations + 1)


def _permutation_pvalues_torch(
    local_groups,
    local_celltypes,
    local_strata,
    observed_counts,
    expected_counts,
    n_groups,
    n_celltypes,
    n_permutations,
    device,
    permutation_batch_size,
    max_batch_elements,
    random_state,
    show_progress,
    description,
):
    strata = []
    for stratum in np.unique(local_strata):
        indices = np.flatnonzero(local_strata == stratum)
        groups = torch.as_tensor(local_groups[indices], dtype=torch.long, device=device)
        labels = torch.as_tensor(
            local_celltypes[indices], dtype=torch.long, device=device
        )
        strata.append((groups, labels))

    max_stratum_size = max((len(labels) for _, labels in strata), default=0)
    batch_size = _effective_batch_size(
        permutation_batch_size,
        max_stratum_size,
        max_batch_elements,
    )

    expected = torch.as_tensor(
        expected_counts.reshape(-1), dtype=torch.float32, device=device
    )
    observed = torch.as_tensor(
        observed_counts.reshape(-1), dtype=torch.float32, device=device
    )
    observed_deviation = torch.abs(observed - expected)
    extreme_counts = torch.zeros(
        n_groups * n_celltypes,
        dtype=torch.int64,
        device=device,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(int(random_state))

    n_batches = (n_permutations + batch_size - 1) // batch_size
    iterator = range(n_batches)
    if show_progress:
        iterator = tqdm(iterator, desc=description, leave=False)

    completed = 0
    for _ in iterator:
        current_batch = min(batch_size, n_permutations - completed)
        batch_counts = torch.zeros(
            (current_batch, n_groups * n_celltypes),
            dtype=torch.int32,
            device=device,
        )

        for groups, labels in strata:
            n_cells = len(labels)
            if n_cells == 0:
                continue

            if n_cells == 1:
                permuted = labels.expand(current_batch, 1)
            else:
                random_keys = torch.rand(
                    (current_batch, n_cells),
                    dtype=torch.float32,
                    device=device,
                    generator=generator,
                )
                order = torch.argsort(random_keys, dim=1)
                permuted = labels[order]

            flat_indices = groups.unsqueeze(0) * n_celltypes + permuted
            source = torch.ones_like(flat_indices, dtype=torch.int32)
            batch_counts.scatter_add_(1, flat_indices, source)

            if n_cells > 1:
                del random_keys, order
            del permuted, flat_indices, source

        null_deviation = torch.abs(batch_counts.float() - expected.unsqueeze(0))
        extreme_counts += (
            null_deviation >= observed_deviation.unsqueeze(0) - 1e-7
        ).sum(dim=0)
        completed += current_batch

    pvalues = extreme_counts.cpu().numpy().reshape(n_groups, n_celltypes) + 1
    pvalues = pvalues / (n_permutations + 1)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return pvalues


def _multihop_enrichment_test(
    hop_codes,
    celltype_codes,
    stratum_codes,
    max_hop,
    n_celltypes,
    n_permutations,
    pseudocount,
    device,
    permutation_batch_size,
    max_batch_elements,
    random_state,
    show_progress,
    description,
    use_negative_hops,
):
    group_codes = _hop_codes_to_group_indices(
        hop_codes,
        max_hop=max_hop,
        use_negative_hops=use_negative_hops,
    )
    local_mask = (group_codes >= 0) & (stratum_codes >= 0)

    n_groups = 2 * max_hop if use_negative_hops else max_hop
    shape = (n_groups, n_celltypes)
    enrichment = np.full(shape, np.nan, dtype=np.float32)
    pvalues = np.full(shape, np.nan, dtype=np.float64)

    if not local_mask.any():
        return enrichment, pvalues

    local_groups = group_codes[local_mask].astype(np.int64, copy=False)
    local_celltypes = celltype_codes[local_mask].astype(np.int64, copy=False)
    local_strata = stratum_codes[local_mask].astype(np.int64, copy=False)

    observed_counts = _count_by_group(
        local_groups,
        local_celltypes,
        n_groups,
        n_celltypes,
    )
    expected_counts = _analytic_expected_counts_stratified(
        local_groups=local_groups,
        local_celltypes=local_celltypes,
        local_strata=local_strata,
        n_groups=n_groups,
        n_celltypes=n_celltypes,
    )

    enrichment = np.log2(
        (observed_counts + pseudocount) / (expected_counts + pseudocount)
    ).astype(np.float32)

    if device.type == "cuda":
        pvalues = _permutation_pvalues_torch(
            local_groups=local_groups,
            local_celltypes=local_celltypes,
            local_strata=local_strata,
            observed_counts=observed_counts,
            expected_counts=expected_counts,
            n_groups=n_groups,
            n_celltypes=n_celltypes,
            n_permutations=n_permutations,
            device=device,
            permutation_batch_size=permutation_batch_size,
            max_batch_elements=max_batch_elements,
            random_state=random_state,
            show_progress=show_progress,
            description=description,
        )
    else:
        pvalues = _permutation_pvalues_numpy(
            local_groups=local_groups,
            local_celltypes=local_celltypes,
            local_strata=local_strata,
            observed_counts=observed_counts,
            expected_counts=expected_counts,
            n_groups=n_groups,
            n_celltypes=n_celltypes,
            n_permutations=n_permutations,
            random_state=random_state,
            show_progress=show_progress,
            description=description,
        )

    group_sizes = np.bincount(local_groups, minlength=n_groups)
    celltype_totals = np.bincount(local_celltypes, minlength=n_celltypes)
    invalid = (group_sizes[:, None] == 0) | (celltype_totals[None, :] == 0)
    enrichment[invalid] = np.nan
    pvalues[invalid] = np.nan
    return enrichment, pvalues


def _signed_multihop_enrichment_test(
    signed_hop_codes,
    celltype_codes,
    stratum_codes,
    max_hop,
    n_celltypes,
    n_permutations,
    pseudocount,
    device,
    permutation_batch_size,
    max_batch_elements,
    random_state,
    show_progress,
    description,
):
    return _multihop_enrichment_test(
        hop_codes=signed_hop_codes,
        celltype_codes=celltype_codes,
        stratum_codes=stratum_codes,
        max_hop=max_hop,
        n_celltypes=n_celltypes,
        n_permutations=n_permutations,
        pseudocount=pseudocount,
        device=device,
        permutation_batch_size=permutation_batch_size,
        max_batch_elements=max_batch_elements,
        random_state=random_state,
        show_progress=show_progress,
        description=description,
        use_negative_hops=True,
    )


def _bh_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(pvalues, np.nan)
    valid = np.isfinite(pvalues)

    if not valid.any():
        return adjusted

    values = pvalues[valid]
    order = np.argsort(values)
    ranked = values[order]
    n_tests = len(ranked)
    corrected = ranked * n_tests / np.arange(1, n_tests + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    corrected = np.clip(corrected, 0, 1)

    restored = np.empty(n_tests, dtype=float)
    restored[order] = corrected
    adjusted[valid] = restored
    return adjusted


def _write_aggregated_hop_group(
    adata,
    aggregated,
    column,
    max_hop,
    use_negative_hops,
):

    values = np.full(adata.n_obs, "outside", dtype=object)

    if use_negative_hops:
        categories = (
            [f"InHop{hop}" for hop in range(max_hop, 0, -1)]
            + ["Bridge", "ClosureBridge", "Target"]
            + [f"OutHop{hop}" for hop in range(1, max_hop + 1)]
            + ["outside"]
        )

        if not aggregated.empty:
            cell_indices = aggregated["cell_index"].to_numpy(dtype=np.int64)
            values[cell_indices] = aggregated["hop_label"].astype(str).to_numpy()
    else:
        categories = (
            ["Target"] + [f"Hop{hop}" for hop in range(1, max_hop + 1)] + ["outside"]
        )

        if not aggregated.empty:
            cell_indices = aggregated["cell_index"].to_numpy(dtype=np.int64)
            unsigned_hops = aggregated["hop"].to_numpy(dtype=np.int16)

            invalid = (unsigned_hops < 0) | (unsigned_hops > max_hop)
            if invalid.any():
                bad = np.unique(unsigned_hops[invalid]).tolist()
                raise ValueError(
                    "`aggregated.hop` contains values outside " f"0...{max_hop}: {bad}"
                )

            unsigned_labels = np.where(
                unsigned_hops == 0,
                "Target",
                np.char.add("Hop", unsigned_hops.astype(str)),
            )
            values[cell_indices] = unsigned_labels

    adata.obs[column] = pd.Categorical(
        values,
        categories=categories,
        ordered=True,
    )


def _write_aggregated_signed_hop_group(
    adata,
    aggregated,
    column,
    max_hop,
):
    _write_aggregated_hop_group(
        adata=adata,
        aggregated=aggregated,
        column=column,
        max_hop=max_hop,
        use_negative_hops=True,
    )
