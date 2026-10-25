from __future__ import annotations

import re

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from sklearn.preprocessing import StandardScaler

__all__ = [
    "plot_cnmf_heatmap",
    "plot_mp_gene_heatmap",
    "plot_mp_go_dotplot_refined",
    "plot_single_mp_go_barplot",
    "plot_glm_heatmap",
]


def plot_cnmf_heatmap(results, figsize=(12, 12), title="NMF meta program"):
    program_df = results["program_df"]
    cos_sim_df = results["cos_sim_df"]
    stable_clusters = results["stable_clusters"]
    ordered_programs = []
    for cluster_id in sorted(program_df["meta_cluster"].unique()):
        progs = program_df[program_df["meta_cluster"] == cluster_id]["program"].tolist()
        if len(progs) > 1:
            sub_dist = 1 - cos_sim_df.loc[progs, progs].values
            np.fill_diagonal(sub_dist, 0)
            from scipy.cluster.hierarchy import linkage, leaves_list
            from scipy.spatial.distance import squareform

            Z_sub = linkage(squareform(sub_dist, checks=False), method="average")
            ordered_programs.extend([progs[i] for i in leaves_list(Z_sub)])
        else:
            ordered_programs.extend(progs)
    unique_clusters = sorted(program_df["meta_cluster"].unique())
    colors = list(mcolors.TABLEAU_COLORS.values())
    cluster_color_map = {
        cluster: colors[i % len(colors)] for i, cluster in enumerate(unique_clusters)
    }
    ordered_df = program_df.set_index("program").loc[ordered_programs]
    row_colors = ordered_df["meta_cluster"].map(cluster_color_map)
    g = sns.clustermap(
        cos_sim_df.loc[ordered_programs, ordered_programs],
        row_cluster=False,
        col_cluster=False,
        row_colors=row_colors,
        cmap="Reds",
        vmin=0,
        vmax=1,
        figsize=figsize,
        cbar_pos=(0.02, 0.8, 0.03, 0.12),
        xticklabels=False,
        yticklabels=False,
    )
    g.ax_row_colors.set_xlabel("")
    g.ax_row_colors.set_xticks([])
    g.ax_row_colors.set_yticks([])
    target_lw = 1.0
    ax_heatmap = g.ax_heatmap
    for spine in ax_heatmap.spines.values():
        spine.set_visible(False)
    pos = ax_heatmap.get_position()
    heatmap_box = Rectangle(
        (pos.x0, pos.y0),
        pos.width,
        pos.height,
        fill=False,
        edgecolor="black",
        linewidth=target_lw,
        transform=g.fig.transFigure,
        zorder=10,
    )
    g.fig.patches.append(heatmap_box)
    title_height = 0.035
    title_box = Rectangle(
        (pos.x0, pos.y1),
        pos.width,
        title_height,
        facecolor="lightgrey",
        edgecolor="black",
        linewidth=target_lw,
        transform=g.fig.transFigure,
        zorder=10,
    )
    g.fig.patches.append(title_box)
    g.fig.text(
        pos.x0 + pos.width / 2,
        pos.y1 + title_height / 2,
        title,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        zorder=20,
    )
    start = 0
    for cluster_id, size in (
        ordered_df.groupby("meta_cluster", sort=False).size().items()
    ):
        if cluster_id in stable_clusters:
            rect = Rectangle(
                (start, start),
                size,
                size,
                fill=False,
                edgecolor="black",
                linewidth=2.0,
                clip_on=False,
            )
            ax_heatmap.add_patch(rect)
            ax_heatmap.text(
                -0.05,
                start + size / 2,
                f"MP{cluster_id}",
                va="center",
                ha="right",
                fontsize=10,
                fontweight="bold",
                transform=ax_heatmap.get_yaxis_transform(),
            )
        start += size
    return g


def plot_mp_gene_heatmap(results, top_n_genes=15, figsize=(8, 6)):
    mp_spectra = results["mp_spectra"]
    stable_clusters = results["stable_clusters"]
    stable_mp_names = [f"MP{c}" for c in sorted(stable_clusters)]
    stable_spectra = mp_spectra.loc[stable_mp_names]
    selected_genes = []
    for mp in stable_mp_names:
        top_genes = (
            stable_spectra.loc[mp]
            .sort_values(ascending=False)
            .head(top_n_genes)
            .index.tolist()
        )
        for g in top_genes:
            if g not in selected_genes:
                selected_genes.append(g)
    plot_df = stable_spectra[selected_genes].T
    scaler = StandardScaler()
    plot_df_scaled = pd.DataFrame(
        scaler.fit_transform(plot_df.T).T, index=plot_df.index, columns=plot_df.columns
    )
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        plot_df_scaled,
        cmap="RdBu_r",
        center=0,
        cbar_kws={"label": "Z-score of Gene Score", "shrink": 0.5},
        yticklabels=False,
        xticklabels=True,
    )
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    plt.xticks(rotation=45, ha="left")
    plt.title(
        f"Top {top_n_genes} Genes for Stable Meta-Programs",
        fontsize=14,
        fontweight="bold",
        pad=40,
    )
    plt.xlabel("")
    plt.ylabel("Genes", fontsize=12)
    plt.tight_layout()
    return ax


def plot_mp_go_dotplot_refined(
    go_df,
    stable_clusters,
    figsize=None,
    remove_go_id=True,
    bubble_scale=(120, 520),
    bubble_panel_right=0.84,
    cbar_fraction=0.03,
    cbar_pad=0.01,
    cbar_aspect=25,
):
    if go_df.empty:
        print("No GO terms to plot.")
        return None
    plot_data = go_df.copy()
    plot_data["-log10_P"] = -np.log10(plot_data["Adjusted P-value"].astype(float))
    if remove_go_id:
        plot_data["Term"] = (
            plot_data["Term"]
            .astype(str)
            .str.replace("\\s*\\(?(GO:\\d+)\\)?", "", regex=True)
            .str.replace("\\bGO:\\d+\\b", "", regex=True)
            .str.strip()
        )
    stable_clusters = list(stable_clusters)
    plot_data = plot_data[plot_data["MP_Module"].isin(stable_clusters)].copy()
    x_map = {mp: i for i, mp in enumerate(stable_clusters)}
    plot_data["x_pos"] = plot_data["MP_Module"].map(x_map)
    term_order = plot_data["Term"].drop_duplicates().tolist()
    y_map = {t: i for i, t in enumerate(term_order)}
    plot_data["y_pos"] = plot_data["Term"].map(y_map)
    if figsize is None:
        figsize = (max(len(stable_clusters) * 1.0, 9), max(len(term_order) * 0.45, 6))
    fig, ax = plt.subplots(figsize=figsize)
    if plot_data["-log10_P"].nunique() > 1:
        sizes = np.interp(
            plot_data["-log10_P"],
            (plot_data["-log10_P"].min(), plot_data["-log10_P"].max()),
            bubble_scale,
        )
    else:
        sizes = np.full(len(plot_data), np.mean(bubble_scale))
    sc = ax.scatter(
        x=plot_data["x_pos"],
        y=plot_data["y_pos"],
        s=sizes,
        c=plot_data["-log10_P"],
        cmap="magma",
        edgecolors="black",
        linewidths=0.5,
    )
    ax.set_xticks(range(len(stable_clusters)))
    ax.set_xticklabels(stable_clusters, rotation=0, ha="center")
    ax.set_yticks(range(len(term_order)))
    ax.set_yticklabels(term_order)
    ax.invert_yaxis()
    ax.set_xlim(-0.5, len(stable_clusters) - 0.5)
    ax.grid(axis="both", linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Meta-program (MP) Modules", fontsize=12, fontweight="bold")
    ax.set_ylabel("GO Biological Process (Top 3)", fontsize=12, fontweight="bold")
    ax.set_title("Functional Enrichment per MP Module", fontsize=14, pad=16)
    plt.subplots_adjust(right=bubble_panel_right)
    cbar = fig.colorbar(
        sc, ax=ax, fraction=cbar_fraction, pad=cbar_pad, aspect=cbar_aspect
    )
    cbar.set_label("-log10(Adj.P)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    return ax


def plot_single_mp_go_barplot(
    go_df,
    mp_name,
    top_n=3,
    term_col="Term",
    p_col="Adjusted P-value",
    mp_col="MP_Module",
    figsize=(7, 3.8),
    remove_go_id=True,
    color="#C06C50",
):
    sub = go_df[go_df[mp_col] == mp_name].copy()
    if sub.empty:
        print(f"No GO results found for {mp_name}.")
        return None
    sub = sub.sort_values(p_col, ascending=True).head(top_n).copy()
    sub["-log10_P"] = -np.log10(sub[p_col].astype(float))
    if remove_go_id:
        sub[term_col] = (
            sub[term_col]
            .astype(str)
            .str.replace("\\s*\\(?(GO:\\d+)\\)?", "", regex=True)
            .str.replace("\\bGO:\\d+\\b", "", regex=True)
            .str.strip()
        )
    sub = sub.sort_values("-log10_P", ascending=True)
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(
        y=sub[term_col],
        width=sub["-log10_P"],
        color=color,
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_xlabel("-log10(Adjusted P-value)", fontsize=11, fontweight="bold")
    ax.set_ylabel("")
    ax.set_title(f"MP_{mp_name} GO enrichment (Top {top_n})", fontsize=13, pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    return ax


def plot_glm_heatmap(
    glm_df,
    neighbor_celltypes=None,
    figsize=(9, 5),
    title="GLM Analysis of Meta-Program Associations with TME",
):
    required_columns = {"module", "neighbor", "coef"}
    missing_columns = required_columns.difference(glm_df.columns)
    if missing_columns:
        raise KeyError(f"`glm_df` is missing columns: {sorted(missing_columns)}")
    plot_df = glm_df.loc[:, ["module", "neighbor", "coef"]].copy()
    plot_df["module"] = plot_df["module"].astype(str)
    plot_df["neighbor"] = plot_df["neighbor"].astype(str)
    plot_df["coef"] = pd.to_numeric(plot_df["coef"], errors="coerce")
    plot_df = plot_df.dropna(subset=["module", "neighbor", "coef"])
    if plot_df.empty:
        raise ValueError("No valid GLM coefficients are available for plotting.")
    available_neighbors = list(dict.fromkeys(plot_df["neighbor"]))
    if neighbor_celltypes is None:
        neighbor_order = available_neighbors
    else:
        neighbor_order = list(
            dict.fromkeys((str(value) for value in neighbor_celltypes))
        )
        if not neighbor_order:
            raise ValueError("`neighbor_celltypes` cannot be empty.")
        missing_neighbors = [
            neighbor
            for neighbor in neighbor_order
            if neighbor not in set(available_neighbors)
        ]
        if missing_neighbors:
            raise ValueError(
                f"The following neighbor cell types were not found in `glm_df`: {missing_neighbors}"
            )
        plot_df = plot_df[plot_df["neighbor"].isin(neighbor_order)].copy()
    plot_df["neighbor"] = pd.Categorical(
        plot_df["neighbor"], categories=neighbor_order, ordered=True
    )
    module_order = sorted(
        plot_df["module"].unique(),
        key=lambda value: (_extract_module_number(value), str(value)),
    )
    plot_df["module"] = pd.Categorical(
        plot_df["module"], categories=module_order, ordered=True
    )
    duplicated = plot_df.duplicated(subset=["module", "neighbor"], keep=False)
    if duplicated.any():
        duplicate_pairs = (
            plot_df.loc[duplicated, ["module", "neighbor"]].drop_duplicates().head(10)
        )
        raise ValueError(
            f"`glm_df` contains duplicated module-neighbor combinations. Each combination must have one coefficient.\nExamples:\n{duplicate_pairs.to_string(index=False)}"
        )
    coef_mat = plot_df.pivot(index="module", columns="neighbor", values="coef")
    coef_mat = coef_mat.reindex(index=module_order, columns=neighbor_order)
    coef_values = coef_mat.to_numpy(dtype=float)
    finite_values = np.abs(coef_values[np.isfinite(coef_values)])
    abs_max = float(finite_values.max()) if finite_values.size > 0 else 1.0
    if abs_max == 0:
        abs_max = 1.0
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        coef_mat,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-abs_max,
        vmax=abs_max,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Standardized coefficient"},
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor("black")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=15)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=15)
    ax.set_title(title)
    ax.set_xlabel("Neighbor Cell Type")
    ax.set_ylabel("Meta-Program (MP)")
    fig.tight_layout()
    return ax


def _extract_module_number(value):
    match = re.search("\\d+", str(value))
    return int(match.group()) if match else np.inf
