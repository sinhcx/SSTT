from __future__ import annotations

import os

# Keep factorization workers from multiplying BLAS threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import shutil
import tempfile
import warnings
from collections.abc import Sequence

import anndata as ad
import gseapy as gp
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import statsmodels.api as sm
from cnmf import cNMF
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits

__all__ = [
    "run_cnmf_meta_analysis",
    "run_mp_go_enrichment",
    "compute_stable_meta_program_scores",
    "build_distance_weighted_exposure_matrix",
    "calculate_glm_associations",
]


def run_cnmf_meta_analysis(
    counts_path: str,
    output_dir: str | None = None,
    ks: Sequence[int] = (5, 6, 7, 8, 9, 10, 11, 12),
    dt: float = 0.1,
    n_iter: int = 20,
    seed: int = 14,
    num_hvg: int = 3000,
    min_expr_pct: float = 0.05,
    meta_dist_threshold: float = 0.7,
    min_k_stability: int = 4,
    top_n_genes: int = 50,
    program_top_n_genes: int = 100,
    n_workers: int = 8,
    save_usage: bool = True,
    name: str = "cnmf",
):
    if not ks:
        raise ValueError("`ks` cannot be empty.")
    if any((int(k) < 2 for k in ks)):
        raise ValueError("Every value in `ks` must be at least 2.")
    if int(n_iter) < 1:
        raise ValueError("`n_iter` must be at least 1.")
    if int(num_hvg) < 1:
        raise ValueError("`num_hvg` must be at least 1.")
    if int(min_k_stability) < 1:
        raise ValueError("`min_k_stability` must be at least 1.")
    if not isinstance(top_n_genes, (int, np.integer)):
        raise TypeError("`top_n_genes` must be an integer.")
    if int(top_n_genes) < 1:
        raise ValueError("`top_n_genes` must be at least 1.")
    if not isinstance(program_top_n_genes, (int, np.integer)):
        raise TypeError("`program_top_n_genes` must be an integer.")
    if int(program_top_n_genes) < 1:
        raise ValueError("`program_top_n_genes` must be at least 1.")
    if not isinstance(n_workers, (int, np.integer)):
        raise TypeError("`n_workers` must be an integer.")
    if int(n_workers) < 1:
        raise ValueError("`n_workers` must be at least 1.")
    ks = sorted(set((int(k) for k in ks)))
    top_n_genes = int(top_n_genes)
    program_top_n_genes = int(program_top_n_genes)
    total_nmf_runs = len(ks) * int(n_iter)
    n_workers = min(int(n_workers), total_nmf_runs)
    is_temp = False
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="cnmf_analysis_")
        is_temp = True
        warnings.warn(
            "`output_dir=None`: cNMF intermediate files and saved usage matrices will be deleted after the function finishes. Use a permanent output_dir if these files should be retained.",
            RuntimeWarning,
            stacklevel=2,
        )
    else:
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    meta_dir = os.path.join(output_dir, "meta_program_cosine")
    os.makedirs(meta_dir, exist_ok=True)
    usage_dir = os.path.join(meta_dir, "usage_by_k")
    if save_usage:
        os.makedirs(usage_dir, exist_ok=True)
    filtered_counts_path = os.path.join(output_dir, f"{name}.min_expr_filtered.h5ad")
    try:
        filtered_adata, gene_filter_stats, min_detected_cells = (
            _filter_genes_by_detection(
                counts_path=counts_path, min_expr_pct=min_expr_pct
            )
        )
        n_cells = int(filtered_adata.n_obs)
        n_genes_input = int(len(gene_filter_stats))
        n_genes_passed = int(filtered_adata.n_vars)
        effective_num_hvg = min(int(num_hvg), n_genes_passed)
        if effective_num_hvg < int(num_hvg):
            warnings.warn(
                f"Only {n_genes_passed} genes passed `min_expr_pct={min_expr_pct}`; using all passing genes rather than `num_hvg={num_hvg}`.",
                RuntimeWarning,
                stacklevel=2,
            )
        gene_filter_stats.to_csv(
            os.path.join(meta_dir, "gene_detection_filter.tsv"), sep="\t", index=False
        )
        filter_summary = pd.DataFrame(
            [
                {
                    "n_cells": n_cells,
                    "n_genes_input": n_genes_input,
                    "n_genes_passed": n_genes_passed,
                    "n_genes_filtered": n_genes_input - n_genes_passed,
                    "min_expr_pct": float(min_expr_pct),
                    "min_detected_cells": int(min_detected_cells),
                    "requested_num_hvg": int(num_hvg),
                    "effective_num_hvg": int(effective_num_hvg),
                }
            ]
        )
        filter_summary.to_csv(
            os.path.join(meta_dir, "gene_detection_filter_summary.tsv"),
            sep="\t",
            index=False,
        )
        filtered_adata.write_h5ad(filtered_counts_path)
        del filtered_adata
        print(
            f"[Gene filter] {n_genes_passed}/{n_genes_input} genes retained; threshold = {min_detected_cells}/{n_cells} cells ({float(min_expr_pct):.1%})."
        )
        print(
            f"[HVG] cNMF will select {effective_num_hvg} HVGs from the filtered genes."
        )
        print(
            f"[NMF] {len(ks)} K values × {n_iter} iterations = {total_nmf_runs} independent NMF runs."
        )
        print(f"[Parallel] Using {n_workers} worker(s).")
        cnmf_obj = cNMF(output_dir=output_dir, name=name)
        cnmf_obj.prepare(
            counts_fn=filtered_counts_path,
            components=ks,
            n_iter=n_iter,
            seed=seed,
            num_highvar_genes=effective_num_hvg,
        )
        if os.path.exists(filtered_counts_path):
            os.remove(filtered_counts_path)
        print("[Factorization] Starting...")
        with threadpool_limits(limits=1):
            if n_workers == 1:
                cnmf_obj.factorize(worker_i=0, total_workers=1)
            else:
                cnmf_obj.factorize_multi_process(total_workers=n_workers)
        print("[Factorization] Complete.")
        print("[Combine] Combining NMF runs...")
        cnmf_obj.combine()
        print("[Consensus] Calculating consensus programs...")
        for k in ks:
            print(f"[Consensus] K={k}")
            cnmf_obj.consensus(k=k, density_threshold=dt, show_clustering=False)
        all_programs = []
        program_vectors = []
        usage_matrices = {}
        for k in ks:
            usage, spectra_scores, _, _ = cnmf_obj.load_results(
                K=k, density_threshold=dt
            )
            usage = usage.copy()
            original_usage_columns = list(usage.columns)
            usage.columns = [f"K{k}_{program}" for program in original_usage_columns]
            usage_matrices[k] = usage
            if save_usage:
                usage.to_csv(
                    os.path.join(usage_dir, f"usage_K{k}.tsv.gz"),
                    sep="\t",
                    compression="gzip",
                )
            for program in spectra_scores.columns:
                program_name = f"K{k}_{program}"
                vector = spectra_scores[program].copy()
                vector.name = program_name
                program_vectors.append(vector)
                all_programs.append(
                    {
                        "program": program_name,
                        "k": k,
                        "top_genes": vector.sort_values(ascending=False)
                        .head(program_top_n_genes)
                        .index.tolist(),
                    }
                )
        if not program_vectors:
            raise RuntimeError("No cNMF programs were loaded.")
        usage_all = pd.concat([usage_matrices[k] for k in ks], axis=1)
        if save_usage:
            usage_all.to_csv(
                os.path.join(meta_dir, "usage_all_K.tsv.gz"),
                sep="\t",
                compression="gzip",
            )
        print(f"[Usage] Saved consensus usage for {len(usage_matrices)} K values.")
        print(
            f"[Usage] Combined matrix: {usage_all.shape[0]} cells × {usage_all.shape[1]} programs."
        )
        program_df = pd.DataFrame(all_programs)
        program_matrix = pd.concat(program_vectors, axis=1).T
        # Compare full gene-score vectors across all tested K values.
        cosine_sim = cosine_similarity(program_matrix.values)
        cos_sim_df = pd.DataFrame(
            cosine_sim, index=program_matrix.index, columns=program_matrix.index
        )
        distance_matrix = 1.0 - cosine_sim
        distance_matrix = np.clip(distance_matrix, 0.0, 2.0)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2.0
        np.fill_diagonal(distance_matrix, 0.0)
        linkage_matrix = linkage(
            squareform(distance_matrix, checks=False), method="average"
        )
        clusters = fcluster(
            linkage_matrix, t=float(meta_dist_threshold), criterion="distance"
        )
        program_df["meta_cluster"] = clusters
        meta_summary = (
            program_df.groupby("meta_cluster", observed=True)
            .agg(
                n_programs=("program", "count"),
                n_unique_k=("k", "nunique"),
                ks=("k", lambda values: sorted(set(values))),
            )
            .reset_index()
        )
        meta_summary["stable"] = meta_summary["n_unique_k"] >= int(min_k_stability)
        stable_clusters = meta_summary.loc[
            meta_summary["stable"], "meta_cluster"
        ].tolist()
        meta_gene_rows = []
        meta_program_vectors = []
        for cluster_id, cluster_programs in program_df.groupby(
            "meta_cluster", observed=True, sort=True
        ):
            mean_vector = program_matrix.loc[cluster_programs["program"]].mean(axis=0)
            mean_vector.name = f"MP{cluster_id}"
            meta_program_vectors.append(mean_vector)
            top_gene_scores = mean_vector.sort_values(ascending=False).head(top_n_genes)
            for rank, (gene, score) in enumerate(top_gene_scores.items(), start=1):
                meta_gene_rows.append(
                    {
                        "meta_program": f"MP{cluster_id}",
                        "meta_cluster": int(cluster_id),
                        "stable": bool(cluster_id in stable_clusters),
                        "gene": gene,
                        "rank": rank,
                        "score": float(score),
                    }
                )
        mp_spectra = pd.concat(meta_program_vectors, axis=1).T
        mp_gene_lists = pd.DataFrame(meta_gene_rows)
        analysis_parameters = pd.DataFrame(
            [
                {
                    "counts_path": os.path.abspath(counts_path),
                    "ks": ",".join(map(str, ks)),
                    "density_threshold": float(dt),
                    "n_iter": int(n_iter),
                    "seed": int(seed),
                    "requested_num_hvg": int(num_hvg),
                    "effective_num_hvg": int(effective_num_hvg),
                    "min_expr_pct": float(min_expr_pct),
                    "min_detected_cells": int(min_detected_cells),
                    "meta_dist_threshold": float(meta_dist_threshold),
                    "min_k_stability": int(min_k_stability),
                    "top_n_genes": int(top_n_genes),
                    "program_top_n_genes": int(program_top_n_genes),
                    "n_workers": int(n_workers),
                    "save_usage": bool(save_usage),
                    "name": str(name),
                }
            ]
        )
        mp_spectra.to_csv(os.path.join(meta_dir, "mp_spectra_matrix.tsv"), sep="\t")
        mp_gene_lists.to_csv(
            os.path.join(meta_dir, "mp_gene_lists.tsv"), sep="\t", index=False
        )
        meta_summary.to_csv(
            os.path.join(meta_dir, "mp_summary.tsv"), sep="\t", index=False
        )
        program_df.to_csv(
            os.path.join(meta_dir, "program_assignments.tsv"), sep="\t", index=False
        )
        cos_sim_df.to_csv(
            os.path.join(meta_dir, "program_cosine_similarity.tsv"), sep="\t"
        )
        analysis_parameters.to_csv(
            os.path.join(meta_dir, "analysis_parameters.tsv"), sep="\t", index=False
        )
        print(
            f"[Gene lists] Each final MP returns Top {top_n_genes} genes; each original K-program stores Top {program_top_n_genes} genes."
        )
        print(f"分析完成。结果保存在: {meta_dir}")
        return {
            "mp_summary": meta_summary,
            "mp_gene_lists": mp_gene_lists,
            "mp_spectra": mp_spectra,
            "program_df": program_df,
            "usage_by_k": usage_matrices,
            "usage_all": usage_all,
            "stable_clusters": stable_clusters,
            "cos_sim_df": cos_sim_df,
            "gene_filter_stats": gene_filter_stats,
            "gene_filter_summary": filter_summary,
            "analysis_parameters": analysis_parameters,
        }
    finally:
        if os.path.exists(filtered_counts_path):
            os.remove(filtered_counts_path)
        if is_temp and os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"已清理临时文件夹: {output_dir}")


def run_mp_go_enrichment(mp_gene_lists, stable_clusters, top_n=3):
    all_go_results = []
    mp_df = mp_gene_lists[mp_gene_lists["meta_cluster"].isin(stable_clusters)]
    for mp_id in stable_clusters:
        genes = mp_df[mp_df["meta_cluster"] == mp_id]["gene"].tolist()
        if len(genes) < 5:
            print(f"Skipping {mp_id}: Too few genes ({len(genes)})")
            continue
        print(f"Running GO enrichment for {mp_id}...")
        try:
            enr = gp.enrichr(
                gene_list=genes,
                gene_sets=["GO_Biological_Process_2025"],
                organism="homo sapiens",
                outdir=None,
                cutoff=0.05,
            )
            res = enr.results
            if not res.empty:
                res = res[res["Adjusted P-value"] < 0.05].sort_values(
                    "Adjusted P-value"
                )
                top_go = res.head(top_n).copy()
                top_go["MP_Module"] = mp_id
                all_go_results.append(
                    top_go[["MP_Module", "Term", "Adjusted P-value", "Genes"]]
                )
        except Exception as e:
            print(f"Error analyzing {mp_id}: {e}")
    if all_go_results:
        final_df = pd.concat(all_go_results).reset_index(drop=True)
        return final_df
    else:
        return pd.DataFrame()


def compute_stable_meta_program_scores(adata, results, mode="usage_matrix"):
    valid_modes = {"usage_matrix", "gene_score"}
    if mode not in valid_modes:
        raise ValueError(f"`mode` must be one of {sorted(valid_modes)}, got {mode!r}.")
    if "stable_clusters" not in results:
        raise KeyError("`results` is missing `stable_clusters`.")
    stable_clusters = sorted([int(cluster) for cluster in results["stable_clusters"]])
    stable_mp_names = [f"MP{cluster}" for cluster in stable_clusters]
    print(f"Scoring stable Meta-Programs using mode='{mode}': {stable_mp_names}")
    computed_columns = []
    if mode == "usage_matrix":
        required_keys = {"program_df", "usage_by_k"}
        missing_keys = required_keys - set(results.keys())
        if missing_keys:
            raise KeyError(
                f"For mode='usage_matrix', `results` is missing: {sorted(missing_keys)}"
            )
        program_df = results["program_df"].copy()
        usage_by_k_raw = results["usage_by_k"]
        usage_by_k = {int(k): usage.copy() for k, usage in usage_by_k_raw.items()}
        adata_index = pd.Index(adata.obs_names.astype(str))
        for cluster_id in stable_clusters:
            mp_name = f"MP{cluster_id}"
            score_name = f"{mp_name}_score"
            mp_programs = program_df.loc[
                program_df["meta_cluster"].astype(int) == cluster_id
            ].copy()
            if mp_programs.empty:
                warnings.warn(
                    f"{mp_name} has no corresponding programs in `program_df`; skipped."
                )
                continue
            usage_per_k = []
            for k, sub_k in mp_programs.groupby("k", observed=True, sort=True):
                k = int(k)
                if k not in usage_by_k:
                    warnings.warn(
                        f"{mp_name}: usage for K={k} was not found; this K was skipped."
                    )
                    continue
                usage = usage_by_k[k].copy()
                usage.index = usage.index.astype(str)
                program_names = sub_k["program"].astype(str).tolist()
                resolved_columns = []
                for program_name in program_names:
                    if program_name in usage.columns:
                        resolved_columns.append(program_name)
                        continue
                    prefix = f"K{k}_"
                    if program_name.startswith(prefix):
                        suffix = program_name[len(prefix) :]
                        candidates = [suffix]
                        try:
                            candidates.append(int(suffix))
                        except ValueError:
                            pass
                        matched = False
                        for candidate in candidates:
                            if candidate in usage.columns:
                                resolved_columns.append(candidate)
                                matched = True
                                break
                        if matched:
                            continue
                    warnings.warn(
                        f"{mp_name}, K={k}: usage column corresponding to {program_name!r} was not found."
                    )
                if len(resolved_columns) == 0:
                    continue
                k_mp_usage = usage[resolved_columns].sum(axis=1)
                k_mp_usage.name = f"K{k}"
                usage_per_k.append(k_mp_usage)
            if len(usage_per_k) == 0:
                warnings.warn(
                    f"{mp_name} has no usable K-specific usage and was skipped."
                )
                continue
            mp_usage_matrix = pd.concat(usage_per_k, axis=1)
            # Each available K contributes equally after summing its programs.
            mp_usage = mp_usage_matrix.mean(axis=1)
            score = mp_usage.reindex(adata_index)
            score.index = adata.obs_names
            adata.obs[score_name] = score.astype(float)
            computed_columns.append(score_name)
            n_matched = int(adata.obs[score_name].notna().sum())
            print(
                f"{mp_name}: {len(usage_per_k)} K values, {len(mp_programs)} constituent programs, {n_matched}/{adata.n_obs} cells matched"
            )
    elif mode == "gene_score":
        if "mp_gene_lists" not in results:
            raise KeyError(
                "For mode='gene_score', `results` is missing `mp_gene_lists`."
            )
        meta_program_genes = results["mp_gene_lists"].copy()
        required_columns = {"meta_program", "gene"}
        missing_columns = required_columns - set(meta_program_genes.columns)
        if missing_columns:
            raise KeyError(
                f"`results['mp_gene_lists']` is missing columns: {sorted(missing_columns)}"
            )
        adata_genes = set(adata.var_names.astype(str))
        for mp_name in stable_mp_names:
            score_name = f"{mp_name}_score"
            genes = (
                meta_program_genes.loc[
                    meta_program_genes["meta_program"].astype(str) == mp_name, "gene"
                ]
                .astype(str)
                .tolist()
            )
            genes = list(dict.fromkeys(genes))
            valid_genes = [gene for gene in genes if gene in adata_genes]
            if len(valid_genes) == 0:
                warnings.warn(
                    f"{mp_name} has no genes present in `adata.var_names`; skipped."
                )
                continue
            sc.tl.score_genes(adata, gene_list=valid_genes, score_name=score_name)
            computed_columns.append(score_name)
            print(f"{mp_name}: {len(valid_genes)}/{len(genes)} genes matched")
    mp_scores = adata.obs[computed_columns].copy()
    return mp_scores


def build_distance_weighted_exposure_matrix(
    adata,
    celltype_col="celltype",
    coord_key="spatial",
    radius_um=50.0,
    sigma_um=20.0,
    target_celltypes=None,
    include_self=False,
    return_neighbor_list=False,
):
    if celltype_col not in adata.obs.columns:
        raise ValueError(f'adata.obs 中不存在列: "{celltype_col}"')
    if coord_key not in adata.obsm:
        raise ValueError(f'adata.obsm 中不存在 key: "{coord_key}"')
    coords = np.asarray(adata.obsm[coord_key])
    if coords.shape[1] < 2:
        raise ValueError(f'adata.obsm["{coord_key}"] 至少需要二维坐标')
    coords = coords[:, :2]
    cell_ids = adata.obs_names.to_numpy()
    celltypes = adata.obs[celltype_col].astype(str).to_numpy()
    all_neighbor_types = sorted(pd.unique(celltypes))
    if target_celltypes is None:
        center_mask = np.ones(len(cell_ids), dtype=bool)
    else:
        center_mask = np.isin(celltypes, target_celltypes)
    center_indices = np.where(center_mask)[0]
    center_ids = cell_ids[center_indices]
    exposure_cols = [f"{ct}" for ct in all_neighbor_types]
    exposure_df = pd.DataFrame(
        0.0, index=center_ids, columns=exposure_cols, dtype=float
    )
    exposure_df["total_exposure"] = 0.0
    exposure_df["n_neighbors"] = 0
    tree = cKDTree(coords)
    neighbor_info = {} if return_neighbor_list else None
    for i in center_indices:
        cid = cell_ids[i]
        center_xy = coords[i]
        nb_idx = tree.query_ball_point(center_xy, r=radius_um)
        if not include_self:
            nb_idx = [j for j in nb_idx if j != i]
        if len(nb_idx) == 0:
            if return_neighbor_list:
                neighbor_info[cid] = {"neighbor_ids": [], "distances": []}
            continue
        nb_idx = np.asarray(nb_idx, dtype=int)
        dists = np.linalg.norm(coords[nb_idx] - center_xy, axis=1)
        dists = np.maximum(dists, 1e-08)
        weights = np.exp(-dists / sigma_um)
        nb_types = celltypes[nb_idx]
        exposure_df.at[cid, "n_neighbors"] = len(nb_idx)
        exposure_df.at[cid, "total_exposure"] = float(weights.sum())
        for ct in np.unique(nb_types):
            mask = nb_types == ct
            exposure_df.at[cid, f"{ct}"] = float(weights[mask].sum())
        if return_neighbor_list:
            neighbor_info[cid] = {
                "neighbor_ids": cell_ids[nb_idx].tolist(),
                "distances": dists.tolist(),
            }
    if return_neighbor_list:
        return (exposure_df, neighbor_info)
    return exposure_df


def calculate_glm_associations(X_df, Y_df):
    common_cells = X_df.index.intersection(Y_df.index)
    X_sub = X_df.loc[common_cells].copy()
    Y_sub = Y_df.loc[common_cells].copy()
    exclude_cols = ["total_exposure", "n_neighbors"]
    predictor_cols = [c for c in X_sub.columns if c not in exclude_cols]
    X = X_sub[predictor_cols].copy()
    X = (X - X.mean()) / X.std()
    results = []
    for module in Y_sub.columns:
        y = (Y_sub[module].astype(float) - Y_sub[module].mean()) / Y_sub[module].std()
        X_model = sm.add_constant(X)
        # Fit standardized OLS associations with an intercept.
        model = sm.OLS(y, X_model).fit()
        for var in predictor_cols:
            results.append(
                {
                    "module": module,
                    "neighbor": var,
                    "coef": model.params[var],
                    "pvalue": model.pvalues[var],
                    "tvalue": model.tvalues[var],
                    "r2": model.rsquared,
                }
            )
    glm_df = pd.DataFrame(results)
    glm_df["FDR"] = multipletests(glm_df["pvalue"], method="fdr_bh")[1]
    return glm_df


def _filter_genes_by_detection(
    counts_path: str, min_expr_pct: float
) -> tuple[ad.AnnData, pd.DataFrame, int]:
    if not 0 <= float(min_expr_pct) <= 1:
        raise ValueError("`min_expr_pct` must be between 0 and 1.")
    if not str(counts_path).lower().endswith(".h5ad"):
        raise ValueError(
            "Gene detection filtering currently requires `counts_path` to be an .h5ad file."
        )
    adata = ad.read_h5ad(counts_path)
    if adata.n_obs == 0:
        raise ValueError("The input AnnData contains no cells.")
    if adata.n_vars == 0:
        raise ValueError("The input AnnData contains no genes.")
    if not adata.var_names.is_unique:
        raise ValueError("`adata.var_names` must be unique before cNMF.")
    matrix = adata.X
    if sp.issparse(matrix):
        matrix = matrix.tocsr(copy=True)
        matrix.eliminate_zeros()
        detected_cells = np.asarray((matrix > 0).sum(axis=0)).ravel()
    else:
        detected_cells = np.count_nonzero(np.asarray(matrix) > 0, axis=0)
    min_detected_cells = max(1, int(np.ceil(float(min_expr_pct) * adata.n_obs)))
    detected_pct = detected_cells / adata.n_obs
    passed = detected_cells >= min_detected_cells
    gene_filter_stats = pd.DataFrame(
        {
            "gene": adata.var_names.astype(str),
            "detected_cells": detected_cells.astype(np.int64),
            "detected_pct": detected_pct.astype(np.float64),
            "min_detected_cells": int(min_detected_cells),
            "min_expr_pct": float(min_expr_pct),
            "passed_min_expr": passed,
        }
    )
    n_passed = int(passed.sum())
    if n_passed == 0:
        raise ValueError(
            f"No genes passed the detection filter. `min_expr_pct={min_expr_pct}` corresponds to at least {min_detected_cells}/{adata.n_obs} cells."
        )
    return (adata[:, passed].copy(), gene_filter_stats, min_detected_cells)
